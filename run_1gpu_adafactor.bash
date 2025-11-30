#!/bin/bash
set -e
source asyncpp/bin/activate

echo ">>> 1-GPU Adafactor Optimizer Test Started..."

# 1-GPU, 1-Process 실행
python -m torch.distributed.run \
  --nproc_per_node=1 \
  --nnodes=1 \
  --rdzv_endpoint=localhost:12345 \
  /home/leemyl2002/AsyncPP/main_with_runtime.py \
  --module models.gptn.gpus=1 \
  --config_path models/gptn/gpus=1/mp_conf.json \
  --master_addr 127.0.0.1 \
  --master_port 12345 \
  --block_size 512 \
  --n_embd 384 \
  --n_head 12 \
  --n_layer 8 \
  -b 8 \
  --eval-batch-size 8 \
  --dataset_name wikitext-103-v1 \
  --distributed_backend gloo \
  --lr 3e-4 \
  --lr_warmup \
  --optimizer adafactor \
  --epochs 2 \
  --num_minibatches 1000 \
  --num_eval_minibatches 25 \
  --clip_grad 10 \
  --log_tb \
  --recompute \
  --lr_policy cosine \
  --momentum 0.99 \
  --tb_dir ./runs \
  --exp_name wikitext-103-v1_gptn_512_384_12_8_b8/gpus=1/adafactor_test/

echo ">>> Test Completed."