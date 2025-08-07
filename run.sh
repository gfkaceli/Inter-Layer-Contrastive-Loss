#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# about how to use PyTorch's distributed data parallel.

python train.py \
  --train_file data/wiki1m_for_simcse.txt \
  --model_name_or_path bert-base-uncased \
  --num_train_epochs 1 \
  --per_device_train_batch_size 64 \
  --learning_rate 3e-5 \
  --max_seq_length 32 \
  --pooler_type avg \
  --temp 0.05 \
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
