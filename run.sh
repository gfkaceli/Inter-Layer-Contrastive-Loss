#!/bin/bash
# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# about how to use PyTorch's distributed data parallel.

python -m torch.distributed.launch --nproc_per_node=5 train.py \
  --train_file data/wiki1m_for_simcse.txt \
  --model_name_or_path bert-base-uncased \
  --num_train_epochs 100 \
  --per_device_train_batch_size 16 \
  gradient_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --max_seq_length 32 \
  --pooler_type avg \
  --temp 0.05 \
  --ilcl_sa \
  --ilcl_layers 4 8 12 \
  --ilcl_weight 0.4 \
  --do_neg \
  --mlp_only_train \
  --do_train \
  --logging_steps 10 \
  --logging_strategy steps \
  --eval_strategy no \
  --eval_steps 100 \
  --save_strategy steps \
  --save_steps 100 \
  --output_dir result \
  --overwrite_output_dir \
  --fp16 \
  --use_cpu
