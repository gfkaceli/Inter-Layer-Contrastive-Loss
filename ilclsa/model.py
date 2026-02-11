import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from torch.nn import MSELoss
from transformers.models.llama.modeling_llama import (
LlamaPreTrainedModel,
LlamaModel,
LlamaConfig
)
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

# A tiny gate to score each layer’s pooled representation
class LayerGate(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        # map from pooled-dim → num_layers logits
        self.fc = nn.Linear(hidden_size, num_layers)

    def forward(self, layer_feats):
        # layer_feats: [batch, num_layers, hidden_size]
        # we average over tokens already, so just pool across layers -> logits of shape [batch, num_layers]
        return self.fc(layer_feats.mean(dim=-1))

class MLPLayer(nn.Module):
    """Head for getting sentence representations from CLS representation."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x

class DropoutLayer(nn.Module):
    """Dropout layer for sentence representations."""
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
    def forward(self, features):
        return self.dropout(features)

class ProjectionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config.hidden_size
        hidden_dim = config.hidden_size * 2
        out_dim = config.hidden_size
        affine = False
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=affine),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)

class Similarity(nn.Module):
    """Dot product or cosine similarity with temperature scaling."""
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
    Parameter-free pooler to get the sentence embedding from token embeddings.
    Supported types:
      - 'cls': use [CLS] token (after transformer encoder)
      - 'cls_before_pooler': [CLS] token without the encoder’s built-in pooler
      - 'avg': average of last layer token embeddings (masked average)
      - 'avg_top2': average of the last two layers’ token embeddings
      - 'avg_first_last': average of the first and last layers’ token embeddings
    """
    def __init__(self, pooler_type):
        super().__init__()
        assert pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], f"unrecognized pooling type {pooler_type}"
        self.pooler_type = pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state      # [bs*num_sent, seq_len, hid]
        hidden_states = outputs.hidden_states if outputs.hidden_states is not None else last_hidden
        # Always return a tuple: (pooled_output, hidden_states)
        if self.pooler_type in ['cls_before_pooler', 'cls']:
            pooled = last_hidden[:, 0]  # CLS token
            return pooled, hidden_states
        elif self.pooler_type == "avg":
            pooled = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled, hidden_states
        elif self.pooler_type == "avg_first_last":
            # average of first and last layer hidden states
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled, hidden_states
        elif self.pooler_type == "avg_top2":
            # average of last two layers
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled, hidden_states
        else:
            raise NotImplementedError

def construct_negative(cls, batch_size, attention_mask, z1, cos_sim, loss_fct, hidden_states, num_sent, labels):
    # Use hidden representations from specified negative layers as additional negatives (SSCL)
    for i in range(cls.negative_layers):
        outputs = hidden_states[-2 - i]  # take last i+1-th layer (excluding final)
        if cls.pooler_type == 'cls':
            pooler_outputs = outputs[:, 0]
        elif cls.pooler_type == 'avg':
            pooler_outputs = (outputs * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        else:
            # default to cls for other types
            pooler_outputs = outputs[:, 0]
        pooler_outputs = pooler_outputs.view((batch_size, num_sent, outputs.size(-1)))
        if cls.pooler_type == "cls":
            pooler_outputs = cls.mlp(pooler_outputs)
        # separate representations
        n1, n2 = pooler_outputs[:, 0], pooler_outputs[:, 1]
        cos_sim1 = cls.sim(z1.unsqueeze(1), n1.unsqueeze(0))
        cos_sim2 = cls.sim(z1.unsqueeze(1), n2.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, cos_sim1, cos_sim2], dim=-1)
    loss = loss_fct(cos_sim, labels)
    return loss

def cl_init(cls, config):
    cls.pooler_type = cls.model_args.pooler_type
    cls.negative_layers = cls.model_args.hard_negative_layers
    cls.do_neg = cls.model_args.do_neg
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
        cls.dropout = DropoutLayer(config)
        cls.proj_mlp = ProjectionMLP(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()
    # New arguments for ILCL-SA
    cls.ilcl_sa = getattr(cls.model_args, "ilcl_sa", False)
    cls.ilcl_layers = getattr(cls.model_args, "ilcl_layers", [])
    cls.ilcl_weight = getattr(cls.model_args, "ilcl_weight", 1.0)
    cls.normalize_emb = getattr(cls.model_args, "normalize_emb", False)
    # For backward compatibility or default selection: if ILCL-SA enabled but no layers specified, use top_k_layers if available
    cls.top_k_layers = getattr(cls.model_args, "top_k_layers", 3)
    if cls.ilcl_sa and len(cls.ilcl_layers) == 0:
        # default to last `top_k_layers` intermediate layers
        if hasattr(config, "num_hidden_layers"):
            total_layers = config.num_hidden_layers
        elif hasattr(config, "num_layers"):
            total_layers = config.num_layers
        else:
            total_layers = None
        if total_layers:
            # Select intermediate transformer layers (not the final layer used for anchors)
            # For a 12-layer model with top_k_layers=3, we want indices like [4, 7, 10]
            # (transformer layers 4, 7, 10 from hidden_states tuple)
            # This gives better coverage than just the last few layers
            if cls.top_k_layers == 1:
                # Use middle layer
                cls.ilcl_layers = [total_layers // 2]
            else:
                # Evenly distribute across layers, avoiding first 2 and last layer
                start_layer = 2  # Skip embedding (0) and first transformer layer (1)
                end_layer = total_layers - 1  # Don't include final layer (that's our anchor)
                if end_layer > start_layer:
                    step = (end_layer - start_layer) / (cls.top_k_layers - 1) if cls.top_k_layers > 1 else 1
                    cls.ilcl_layers = [int(start_layer + step * i) for i in range(cls.top_k_layers)]
                else:
                    # Fallback if model is too shallow
                    cls.ilcl_layers = list(range(1, min(total_layers, cls.top_k_layers + 1)))
            # Ensure indices are valid and sorted
            cls.ilcl_layers = sorted([idx for idx in cls.ilcl_layers if 0 < idx < total_layers])
    # (Optional anchor/momentum fields for future)
    cls.use_anchor = getattr(cls.model_args, "use_anchor", False)
    cls.anchor_weight = getattr(cls.model_args, "anchor_weight", 0.2)
    cls.iso_weight = getattr(cls.model_args, "iso_weight", 0.01)
    # Prepare layer gating (not used in current implementation but reserved)
    if hasattr(config, "num_hidden_layers"):
        cls.num_layers = config.num_hidden_layers
    elif hasattr(config, "num_layers"):
        cls.num_layers = config.num_layers
    else:
        cls.num_layers = None
    cls.layer_gate = LayerGate(config.hidden_size, cls.num_layers) if cls.num_layers else None
    cls.anchor_loss_fn = MSELoss()
    # Validate ILCL layer indices
    if cls.ilcl_sa and cls.ilcl_layers:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"ILCL-SA enabled with layers: {cls.ilcl_layers}")
        # hidden_states is a tuple: (embedding_layer, layer_1, layer_2, ..., layer_N)
        # So we need indices in range [0, num_layers] where 0 is embedding, 1-N are transformer layers
        max_idx = cls.num_layers if cls.num_layers else 12
        for idx in cls.ilcl_layers:
            assert 0 <= idx <= max_idx, \
                f"Invalid ILCL layer index {idx}. Must be in range [0, {max_idx}] for {max_idx}-layer model."
        logger.info(f"ILCL layer indices validated successfully.")


def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = input_ids.size(0)
    num_sent = input_ids.size(1)  # 2 for pair (SimCSE), 3 if an extra hard negative
    # Flatten input for encoding
    input_ids = input_ids.view(-1, input_ids.size(-1))             # [bs * num_sent, seq_len]
    attention_mask = attention_mask.view(-1, attention_mask.size(-1))
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
    # Encode all sentences (and optional MLM augmentation)
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True,  # always get hidden states for ILCL
        return_dict=True,
    )
    mlm_outputs = None
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view(-1, mlm_input_ids.size(-1))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
    # Pooling to get sentence embeddings and retrieve hidden states
    pooler_output, hidden_states = cls.pooler(attention_mask, outputs)
    # Reshape to [batch_size, num_sent, hidden_size]
    pooler_output = pooler_output.view(batch_size, num_sent, pooler_output.size(-1))
    # Apply projection MLP if using 'cls' pooler
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)
    # Separate embeddings for each sentence in the pair (and hard negative if present)
    z1, z2 = pooler_output[:, 0], pooler_output[:, 1]  # (batch_size, hidden_size) each
    if num_sent == 3:
        z3 = pooler_output[:, 2]
    # Optionally normalize embeddings
    if cls.normalize_emb:
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        if num_sent == 3:
            z3 = F.normalize(z3, p=2, dim=-1)
    # Gather embeddings from all processes for a larger effective batch (distributed training)
    if dist.is_initialized() and cls.training:
        # Gather z3 as well if present
        if num_sent == 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3  # replace the current rank's portion with original (to keep gradient)
            z3 = torch.cat(z3_list, dim=0)
        # Gather z1 and z2
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        # Update z1, z2 to contain embeddings from all processes
        z1 = torch.cat(z1_list, dim=0)
        z2 = torch.cat(z2_list, dim=0)
    # Compute cosine similarity between all z1 and z2 in the batch
    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # [total_samples, total_samples] similarity matrix
    if num_sent >= 3:
        # Also compute similarity between z1 and z3 (hard negatives) and append as additional columns
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], dim=1)
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
    loss_fct = nn.CrossEntropyLoss()

    # Handle hard negative weighting if provided
    if num_sent == 3:
        z3_weight = cls.model_args.hard_negative_weight
        if z3_weight != 0:
            # Increase logits for hard negatives on their corresponding positions
            weights = torch.zeros_like(cos_sim)
            for i in range(z1_z3_cos.size(0)):
                # place z3_weight at the hard negative position for each example i
                weights[i, cos_sim.size(1) - z1_z3_cos.size(1) + i] = z3_weight
            cos_sim = cos_sim + weights

    # Contrastive loss for sentence-level (SimCSE objective)
    r_loss = loss_fct(cos_sim, labels)
    if cls.do_neg:
        # Include intermediate negatives from specified layers (SSCL)
        neg_loss = construct_negative(cls, batch_size, attention_mask, z1, cos_sim, loss_fct, hidden_states, num_sent, labels)
        simcse_loss = 0.5 * (r_loss + neg_loss)
    else:
        simcse_loss = r_loss
    # MLM auxiliary loss (if enabled)
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        simcse_loss = simcse_loss + cls.model_args.mlm_weight * masked_lm_loss

    # Inter-layer Contrastive Learning with Semantic Anchors (ILCL-SA)
    ilcl_loss = 0.0
    if cls.ilcl_sa:
        # Prepare anchor embeddings (stop-gradient)
        anchor_list = [z1.detach(), z2.detach()]
        if num_sent == 3:
            anchor_list.append(z3.detach())
        anchor_stack = torch.cat(anchor_list, dim=0)  # shape: [total_samples, hidden_size]
        intermediate_losses = []
        for layer_idx in cls.ilcl_layers:
            # Get intermediate layer hidden outputs
            # Note: hidden_states is a tuple from the transformer:
            # hidden_states[0] = embedding layer output
            # hidden_states[1] = transformer layer 1 output
            # hidden_states[2] = transformer layer 2 output
            # ...
            # hidden_states[N] = transformer layer N output (final layer)
            # layer_idx should be in range [1, N-1] to select intermediate transformer layers
            inter_hidden = hidden_states[layer_idx]  # tensor shape [batch*num_sent, seq_len, hid]
            # Pool the intermediate layer representation
            if cls.pooler_type == 'cls':
                inter_pooled = inter_hidden[:, 0]
            elif cls.pooler_type == 'avg':
                inter_pooled = (inter_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            else:
                # default: use average pooling for other pooler types
                inter_pooled = (inter_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            inter_pooled = inter_pooled.view(batch_size, num_sent, inter_pooled.size(-1))
            if cls.pooler_type == 'cls':
                inter_pooled = cls.mlp(inter_pooled)
            n1, n2 = inter_pooled[:, 0], inter_pooled[:, 1]
            if num_sent == 3:
                n3 = inter_pooled[:, 2]
            # Gather intermediate pooled embeddings across processes if distributed
            if dist.is_initialized() and cls.training:
                n1_list = [torch.zeros_like(n1) for _ in range(dist.get_world_size())]
                n2_list = [torch.zeros_like(n2) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=n1_list, tensor=n1.contiguous())
                dist.all_gather(tensor_list=n2_list, tensor=n2.contiguous())
                n1_list[dist.get_rank()] = n1
                n2_list[dist.get_rank()] = n2
                n1 = torch.cat(n1_list, dim=0)
                n2 = torch.cat(n2_list, dim=0)
                if num_sent == 3:
                    n3_list = [torch.zeros_like(n3) for _ in range(dist.get_world_size())]
                    dist.all_gather(tensor_list=n3_list, tensor=n3.contiguous())
                    n3_list[dist.get_rank()] = n3
                    n3 = torch.cat(n3_list, dim=0)
            # Normalize intermediate embeddings if required
            if cls.normalize_emb:
                n1 = F.normalize(n1, p=2, dim=-1)
                n2 = F.normalize(n2, p=2, dim=-1)
                if num_sent == 3:
                    n3 = F.normalize(n3, p=2, dim=-1)
            # Stack intermediate embeddings similar to anchors
            inter_list = [n1, n2]
            if num_sent == 3:
                inter_list.append(n3)
            intermediate_stack = torch.cat(inter_list, dim=0)  # [total_samples, hidden_size]
            # Direct pairwise alignment: each anchor aligns with its corresponding intermediate representation
            # anchor_stack[i] should match intermediate_stack[i] (same input position)
            # We use cosine similarity divided by temperature, then maximize similarity
            sim_scores = F.cosine_similarity(anchor_stack, intermediate_stack, dim=-1) / cls.sim.temp
            # Negative mean to maximize similarity (minimize negative similarity)
            layer_loss = -sim_scores.mean()
            intermediate_losses.append(layer_loss)
        ilcl_loss = sum(intermediate_losses) / len(intermediate_losses)
        # Add ILCL loss (with stop-gradient anchor) to total loss
        simcse_loss = simcse_loss + cls.ilcl_weight * ilcl_loss

    # Total loss is combination of SimCSE (and SSCL if enabled) + ILCL-SA loss
    loss = simcse_loss
    if not return_dict:
        # Return tuple for compatibility
        output_tuple = (cos_sim,)
        if outputs.hidden_states is not None or outputs.attentions is not None:
            # Append hidden_states and attentions if present
            output_tuple += (outputs.hidden_states,) if outputs.hidden_states is not None else ()
            output_tuple += (outputs.attentions,) if outputs.attentions is not None else ()
        return (loss,) + output_tuple
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def sentemb_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )
    # Pooler returns (pooled_output, hidden_states)
    pooled_output, hidden_states = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooled_output = cls.mlp(pooled_output)
    if not return_dict:
        # Convert to tuple output
        result = (outputs.last_hidden_state, pooled_output)
        if outputs.hidden_states is not None:
            result += (outputs.hidden_states,)
        if outputs.attentions is not None:
            result += (outputs.attentions,)
        return result
    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooled_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)
        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)
        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels)

class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)
        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels)


class LlamaForCL(LlamaPreTrainedModel):
    """
    LLaMA model for contrastive learning with ILCL-SA.

    This extends the base LLaMA model with:
    - Contrastive learning objective (SimCSE-style)
    - Inter-layer Contrastive Learning with Semantic Anchors (ILCL-SA)
    - Support for sentence embedding generation

    Key differences from BERT/RoBERTa:
    - Uses 'avg' or 'last_token' pooling instead of CLS token
    - No token_type_ids
    - Decoder architecture with causal attention

    Usage:
        model = LlamaForCL.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            model_args=model_args
        )
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]

        # LLaMA encoder (decoder-only transformer)
        self.model = LlamaModel(config)

        # Optional: Language modeling head for auxiliary CLM task
        # Note: LLaMA is decoder-only, so we use CLM instead of MLM
        if self.model_args.do_mlm:
            # For LLaMA, this would actually be CLM (Causal Language Modeling)
            # We keep the parameter name 'do_mlm' for compatibility
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize ILCL-SA components
        # This sets up: pooler, MLP, similarity function, ILCL parameters, etc.
        cl_init(self, config)

        # Gradient checkpointing for memory efficiency (optional)
        # Uncomment if needed for large models:
        # self.model.gradient_checkpointing_enable()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                sent_emb=False,
                mlm_input_ids=None,
                mlm_labels=None,
                **kwargs):
        """
        Forward pass for LLaMA with ILCL-SA.

        Args:
            input_ids: Token indices of shape [batch_size, seq_length]
                      or [batch_size, num_sent, seq_length] for contrastive learning
            attention_mask: Attention mask of same shape as input_ids
            position_ids: Position indices (optional)
            sent_emb: If True, return sentence embeddings only (for inference)
                     If False, compute contrastive loss (for training)
            mlm_input_ids: Input IDs for auxiliary language modeling task
            mlm_labels: Labels for auxiliary language modeling task

        Returns:
            If sent_emb=True: BaseModelOutputWithPoolingAndCrossAttentions with sentence embeddings
            If sent_emb=False: SequenceClassifierOutput with contrastive loss
        """

        if sent_emb:
            # Inference mode: Generate sentence embeddings
            return sentemb_forward(self, self.model,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   position_ids=position_ids,
                                   inputs_embeds=inputs_embeds,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)
        else:
            # Training mode: Contrastive learning with ILCL-SA
            return cl_forward(self, self.model,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              position_ids=position_ids,
                              inputs_embeds=inputs_embeds,
                              labels=labels,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict,
                              mlm_input_ids=mlm_input_ids,
                              mlm_labels=mlm_labels)

    def get_sentence_embedding(self, input_ids, attention_mask=None):
        """
        Convenience method to get sentence embeddings.

        Args:
            input_ids: Token indices [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            Sentence embeddings: [batch_size, hidden_size]
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            sent_emb=True
        )
        return outputs.pooler_output


class MistralForCL(LlamaPreTrainedModel):
    """
    Mistral model for contrastive learning with ILCL-SA.

    Mistral has the same architecture as LLaMA with some improvements
    (sliding window attention, grouped-query attention).
    We can reuse the LLaMA implementation with Mistral's base model.
    """

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config, *model_args, **model_kargs):
        # Import Mistral model
        from transformers.models.mistral.modeling_mistral import MistralModel

        super().__init__(config)
        self.model_args = model_kargs["model_args"]

        # Mistral encoder
        self.model = MistralModel(config)

        if self.model_args.do_mlm:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        cl_init(self, config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                sent_emb=False,
                mlm_input_ids=None,
                mlm_labels=None,
                **kwargs):

        if sent_emb:
            return sentemb_forward(self, self.model,
                                   input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   position_ids=position_ids,
                                   inputs_embeds=inputs_embeds,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)
        else:
            return cl_forward(self, self.model,
                              input_ids=input_ids,
                              attention_mask=attention_mask,
                              position_ids=position_ids,
                              inputs_embeds=inputs_embeds,
                              labels=labels,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict,
                              mlm_input_ids=mlm_input_ids,
                              mlm_labels=mlm_labels)
