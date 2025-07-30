
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from openpyxl.styles.builtins import output
from torch.nn import MSELoss
from transformers import RobertaTokenizer
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
        # we average over tokens already, so just pool across layers
        # → logits of shape [batch, num_layers]
        return self.fc(layer_feats.mean(dim=-1))

class MLPLayer(nn.Module):
    """
     Head for getting sentence representations over RoBERTa/BERT's CLS representation.
     """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #         self.scale = nn.Linear(config.hidden_size, 256)
        #         self.dropout = nn.Dropout(0.3)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        #         x = self.scale(features)
        #         x = self.dropout(x)
        x = self.activation(x)

        return x

class DropoutLayer(nn.Module):
    """
    Head for dropout getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(0.2)

    #         self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        #         x = self.activation(x)

        return x


class ProjectionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config.hidden_size
        hidden_dim = config.hidden_size * 2
        out_dim = config.hidden_size
        affine=False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine),nn.Tanh()]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class Pooler(nn.Module):
    """
       Parameter-free poolers to get the sentence embedding
       'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
       'cls_before_pooler': [CLS] representation without the original MLP pooler.
       'avg': average of the last layers' hidden states at each token.
       'avg_top2': average of the last two layers.
       'avg_first_last': average of the first and the last layers.
       """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        #         if pooler_output:
        #             print(pooler_output.size(),'----')
        hidden_states = outputs.hidden_states if outputs.hidden_states else last_hidden

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0], hidden_states
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(
                -1), hidden_states
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def construct_negative(cls, batch_size, attention_mask, z1, cos_sim, loss_fct, hidden_states, num_sent, labels):
    for i in range(cls.negative_layers):
        outputs = hidden_states[-2-i]
        if cls.pooler_type == 'cls':
            pooler_outputs = outputs[:, 0]
        elif cls.pooler_type == 'avg':
            pooler_outputs = ((outputs * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

        pooler_outputs = pooler_outputs.view((batch_size, num_sent, outputs.size(-1)))

        if cls.pooler_type == "cls":
            pooler_outputs =cls.mlp(pooler_outputs)

        # separate representations
        n1, n2 = pooler_outputs[:, 0], pooler_outputs[:, 1]

        cos_sim1 = cls.sim(z1.unsqueeze(1), n1.unsqueeze(0))
        cos_sim2 = cls.sim(z1.unsqueeze(1), n2.unsqueeze(0))
        cos_sim = torch.cat((cos_sim, cos_sim1), dim=-1)
        cos_sim = torch.cat((cos_sim, cos_sim2), dim=-1)
    loss = loss_fct(cos_sim, labels)
    return loss



def construct_negative_layers(cls, batch_size, attention_mask, z1,
                              cos_sim, loss_fct, hidden_states, num_sent, labels):
    outputs = hidden_states[cls.negative_layers]

    if cls.pooler_type == 'cls':
        pooler_outputs = hidden_states[:, 0]

    elif cls.pooler_type == 'avg':
        pooler_ouputs = ((outputs * attention_mask.unsqueeze.sum(1)) / attention_mask.unsqueeze.sum(1))

    pooler_outputs = pooler_outputs.view((batch_size, num_sent, outputs.size(-1)))  # (bs, num_sent, hidden)

    if cls.pooler_type == "cls":
        pooler_outputs = cls.mlp(pooler_outputs)
    #         pooler_outputs = cls.proj_mlp(pooler_outputs)
    # Separate representation
    n1, n2 = pooler_ouputs[:, 0], pooler_outputs[:, 1]
    ## to do get batch embeddings and calculate loss

    if num_sent == 3:
        z3 = pooler_outputs[:, 2]

    if dist.is_initialized() and cls.training:
        # gather hard negatives
        if num_sent >= 3:
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
            z3_list[dist.get_rank()] = z3
            z3 = torch.cat(z3_list, 0)

        # dummy vectors
        n1_list = [torch.zeros_like(n1) for _ in range(dist.get_world_size())]
        n2_list = [torch.zeros_like(n2) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=n1_list, tensor=n1.contiguous())
        dist.all_gather(tensor_list=n2_list, tensor=n2.contiguous())

        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        n1_list[dist.get_rank()] = n1
        n2_list[dist.get_rank()] = n2
        # Get full batch embeddings: (bs x N, hidden)
        n1 = torch.cat(n1_list, 0)
        n2 = torch.cat(n2_list, 0)
    cos_sim1 =  cls.sim(z1.unsqueeze(1), n1.unsqueeze(0))
    cos_sim2 =  cls.sim(z1.unsqueeze(1), n2.unsqueeze(0))
    cos_sim = torch.cat((cos_sim, cos_sim1), dim=-1)
    cos_sim = torch.cat((cos_sim, cos_sim2), dim=-1)
    if num_sent==3 :
        z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
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
    # new for ILCL-SA
    cls.use_anchor = getattr(cls.model_args, "use_anchor", False)
    cls.anchor_weight = getattr(cls.model_args, "anchor_weight", 0.2)
    cls.iso_weight = getattr(cls.model_args, "iso_weight", 0.01)
    cls.top_k_layers = getattr(cls.model_args, "top_k_layers", 3)
    cls.layer_selector = getattr(cls.model_args, "layer_selector", "topk")
    cls.num_layers = len(config.num_hidden_layers) if hasattr(config, "num_hidden_layers") else config.num_hidden_layers
    # gating MLP over layers
    cls.layer_gate = LayerGate(config.hidden_size, cls.num_layers)
    # supervision on which layer-pairs to pull
    cls.anchor_loss_fn = MSELoss()


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
    mlm_labels=None,):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)

    num_sent = input_ids.size(1)

    mlm_outputs = None

    # flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1)))

    attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))

    # get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=True,
    )

    # MLM auxiliary objectives
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
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

        # Pooling
    pooler_output, hidden_states = cls.pooler(attention_mask, outputs)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

    # optional MLP on cls
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

        # split z1, z2 (and z3 if present)
    z1 = pooler_output[:, 0]
    z2 = pooler_output[:, 1]
    if num_sent == 3:
        z3 = pooler_output[:, 2]

    return SequenceClassifierOutput()

def sentemb_forward( cls,
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
    return_dict=None,):

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

    pooler_output = cls.pooler(attention_mask, outputs)
    #     print(pooler_output.shape)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
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
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
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
            )
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
                mlm_labels=mlm_labels,
            )



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
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
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
            )
        else:
#             print('i am here else')
#             exit()
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
                mlm_labels=mlm_labels,
            )
