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

class MLPLayer(nn.Module):
    pass

class DropoutLayer(nn.Module):
    pass

class ProjectionMLP(nn.Module):
    pass

class Similarity(nn.Module):
    pass

class Pooler(nn.Module):
    pass

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
