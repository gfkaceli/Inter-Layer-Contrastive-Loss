import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List

import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)
from transformers.trainer_utils import is_main_process

from ilclsa.model import RobertaForCL, BertForCL
from ilclsa.trainers import CLTrainer

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments for model configuration and training.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Model type if training from scratch (e.g., bert, roberta)."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the pretrained models downloaded"}
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Whether to use a fast tokenizer."}
    )
    model_revision: str = field(
        default="main", metadata={"help": "The specific model version to use (branch name, tag name or commit id)."}
    )
    use_auth_token: bool = field(
        default=False, metadata={"help": "Use token from `transformers-cli login` for private models."}
    )
    temp: float = field(
        default=0.05, metadata={"help": "Temperature for the contrastive softmax."}
    )
    pooler_type: str = field(
        default="cls", metadata={"help": "Pooling strategy (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."}
    )
    hard_negative_layers: int = field(
        default=2, metadata={"help": "Number of top layers to use as negatives (for SSCL)."}
    )
    hard_negative_weight: float = field(
        default=0.0, metadata={"help": "Logit weight for hard negatives (if any)."}
    )
    do_mlm: bool = field(
        default=False, metadata={"help": "Whether to use MLM auxiliary objective during training."}
    )
    mlm_weight: float = field(
        default=0.1, metadata={"help": "Weight for MLM loss (if --do_mlm is True)."}
    )
    mlp_only_train: bool = field(
        default=False, metadata={"help": "Use the MLP projection head only during training (not for evaluation)."}
    )
    do_neg: bool = field(
        default=False, metadata={"help": "Whether to use self-contrastive negatives (SSCL)."}
    )
    ilcl_sa: bool = field(
        default=False, metadata={"help": "Enable Inter-layer Contrastive Learning with Semantic Anchors (ILCL-SA)."}
    )
    ilcl_layers: List[int] = field(
        default_factory=lambda: [], metadata={"help": "List of intermediate layer indices to use for ILCL-SA."}
    )
    ilcl_weight: float = field(
        default=1.0, metadata={"help": "Weight factor for the ILCL-SA loss term."}
    )
    normalize_emb: bool = field(
        default=False, metadata={"help": "Whether to L2 normalize embeddings before computing contrastive losses."}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments for data inputs during training and evaluation.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (text file or jsonlines)."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Optional evaluation data file for validation."}
    )
    # ... (other data arguments)
    # (no changes needed in data arguments for this task)

@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation and logging extensions (if any)
    eval_transfer: bool = field(
        default=False, metadata={"help": "Evaluate transfer tasks during training if set true."}
    )
    # (other custom training arguments if present)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Check output directory
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    # ... (dataset loading logic unchanged)

    # Load pretrained model and tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("No tokenizer specified. Provide --tokenizer_name if training from scratch.")

    # Initialize model
    if model_args.model_name_or_path:
        if 'roberta' in model_args.model_name_or_path:
            model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
        elif 'bert' in model_args.model_name_or_path:
            model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            if model_args.do_mlm:
                # initialize MLM head from pretrained weights
                pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
                model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
        else:
            raise NotImplementedError("Only BERT and RoBERTa models are supported for CL.")
    else:
        # training from scratch not supported in this script
        raise NotImplementedError("Training from scratch is not supported in this script.")
    model.resize_token_embeddings(len(tokenizer))

    # Prepare datasets and training
    # ... (the rest of the training setup, trainer initialization, etc., remains largely unchanged)

    # Use custom CLTrainer for training and evaluation
    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"] if training_args.do_train else None,
        eval_dataset=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # Also save tokenizer
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation (if any)
    if training_args.do_eval:
        results = {}
        logger.info("*** Evaluate ***")
        eval_metrics = trainer.evaluate(eval_senteval_transfer=training_args.eval_transfer)
        results.update(eval_metrics)
        logger.info(results)
        # (printing or returning results as needed)
