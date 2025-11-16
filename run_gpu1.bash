#!/bin/bash
# -----------------------------------------------------------------
# AsyncPP 1-GPU 테스트 스크립트 (n_embd 384)
# -----------------------------------------------------------------

# 1. 스크립트가 실패하면 즉시 중지
set -e

# 2. 가상환경 활성화 (필수!)
echo ">>> 가상환경(asyncpp)을 활성화합니다..."
source asyncpp/bin/activate

# 3. 메인 파이썬 스크립트 실행
echo ">>> 1-GPU 테스트를 시작합니다 (Epochs=2)..."
python main_with_runtime.py \
  --module models.gptn.gpus=1 \
  --block_size 512 \
  --n_embd 384 \
  --n_head 12 \
  --n_layer 8 \
  --config_path models/gptn/gpus=1/mp_conf.json \
  -b 8 \
  --eval-batch-size 8 \
  -d wikitext-103-v1 \
  --master_addr localhost \
  --distributed_backend gloo \
  --lr 3e-4 \
  --lr_warmup \
  --optimizer nadamw \
  --epochs 2 \
  --num_minibatches 1000 \
  --num_eval_minibatches 25 \
  --clip_grad 10 \
  --log_tb \
  --tb_dir ./runs \
  --recompute \
  --lr_policy cosine \
  --momentum 0.99 \
  --exp_name wikitext-103-v1_gptn_512_384_12_8_b8/gpus=1/no_checkpoint_test/ \
  --rank 0 \
  --local_rank 0

echo ">>> 테스트가 성공적으로 완료되었습니다."