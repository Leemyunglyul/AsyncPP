#!/bin/bash

# -----------------------------------------------------------------
# AsyncPP 2-GPU 파이프라인 설정 (Tailscale VPN 전용)
# -----------------------------------------------------------------

# [중요] Rank 0 (마스터 노드)의 Tailscale IP 주소를 입력하세요!
# `tailscale ip -4` 명령어로 확인 가능합니다.
MASTER_IP="10.140.0.2"  # <--- 여기에 Rank 0의 Tailscale IP 입력
MASTER_PORT="12345"

export WORLD_SIZE=2

# 💡 [핵심] VPC 피어링이 되었으므로 'eth0'를 사용합니다.
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

# 실험 이름
EXP_NAME="wikitext-103-v1_gptn_512_384_12_8_b8/gpus=4/vpc_peering_final/"

# 공통 명령어 변수 (BASE_CMD)
# 주의: -d wikitext-103-v1 대신 --dataset_name을 사용하여 모호함 방지
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
  --dataset_name wikitext-103-v1 \
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
  --exp_name $EXP_NAME"

echo ">>> Tailscale 2-GPU 환경 변수가 설정되었습니다."
echo ">>> 마스터(Rank 0) IP: $MASTER_IP (Interface: tailscale0)"
