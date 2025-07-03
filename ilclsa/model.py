import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
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


def construct_negative():
    pass

def construct_negative_layers():
    pass

def cl_init():
    pass

def cl_forward():
    pass

def sentemb_forward():
    pass

class BertForCL(nn.Module):
    pass

class RobertaForCL(nn.Module):
    pass
