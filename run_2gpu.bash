#!/bin/bash

# -----------------------------------------------------------------
# AsyncPP 4-GPU 파이프라인 환경 변수 설정 스크립트
# 
# 사용법: 
# 1. 가상환경 활성화: source asyncpp/bin/activate
# 2. 이 스크립트 불러오기: source setup_4gpu.bash
# 3. eval로 랭크 실행: eval "$BASE_CMD --rank 0 --local_rank 0"
# -----------------------------------------------------------------
export WORLD_SIZE=2
# (마스터 IP 주소 - Rank 0의 내부 IP로 수정하세요!)
MASTER_IP="35.229.197.138"
MASTER_PORT="12345"

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

export GLOO_SOCKET_IFNAME=eth0

# (실험 이름 - 이전과 겹치지 않게 새로 지정)
EXP_NAME="wikitext-103-v1_gptn_512_384_12_8_b8/gpus=2/2gpu_pipeline_test/"


# 공통 명령어 변수 (BASE_CMD) 내보내기
export BASE_CMD="python main_with_runtime.py \
  --module models.gptn.gpus=2 \
  --config_path models/gptn/gpus=2/mp_conf.json \
  --master_addr $MASTER_IP \
  --master_port $MASTER_PORT \
  --block_size 512 \
  --n_embd 384 \
  --n_head 12 \
  --n_layer 8 \
  -b 8 \
  --eval-batch-size 8 \
  -d wikitext-103-v1 \
  --distributed_backend nccl \
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
  --exp_name $EXP_NAME"

echo ">>> 4-GPU 환경 변수(BASE_CMD)가 설정되었습니다."
echo ">>> 마스터 주소: $MASTER_IP:$MASTER_PORT"
echo ">>> 실험 이름: $EXP_NAME"