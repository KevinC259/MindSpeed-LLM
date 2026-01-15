#!/bin/bash
set -e

# Positional overrides: 1=MASTER_PORT, 2=LR, 3=ASCEND_RT_VISIBLE_DEVICES
USER_MASTER_PORT=$1
USER_LR=$2
USER_VISIBLE=$3

# Allow overriding visible devices
if [[ -n "${USER_VISIBLE}" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES=${USER_VISIBLE}
fi

############################################
# 基础环境
############################################
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Auto-detect NPUs unless NPUS_PER_NODE is preset
if [[ -z "${NPUS_PER_NODE:-}" || "${NPUS_PER_NODE}" == "auto" ]]; then
    if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -ra _asc_dev <<<"${ASCEND_RT_VISIBLE_DEVICES}"
        NPUS_PER_NODE=${#_asc_dev[@]}
    elif command -v npu-smi >/dev/null 2>&1; then
        NPUS_PER_NODE=$(npu-smi info 2>/dev/null | grep -c "Device ID")
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        NPUS_PER_NODE=1
    fi
fi

NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=${USER_MASTER_PORT:-6002}
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

############################################
# 训练超参
############################################
LR=${USER_LR:-1e-3}
OPTIMIZER="muon"


############################################
# 路径配置（请按实际环境修改）
############################################
DATA_PATH="/sharedata/ckw/dataset/openwebtext-gpt_text_document"
TOKENIZER_PATH="/sharedata/ckw/model_from_hf/gpt2"   # GPT-2 tokenizer
CKPT_LOAD_DIR=""                                    # 从头训练可留空
EXP_NAME="gpt2_medium_${OPTIMIZER}_lr${LR}"
CKPT_SAVE_DIR="/sharedata/ckw/ckpt/${EXP_NAME}"
TENSORBOARD_DIR="./tensorboard/${EXP_NAME}"
WANDB_DIR="./wandb/${EXP_NAME}"

mkdir -p ${CKPT_SAVE_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${WANDB_DIR}

############################################
# 并行配置
############################################
TP=1
PP=1

############################################
# Batch / Token 对齐
############################################
SEQ_LEN=1024          # block_size = 1024
MBS=12                 # micro batch size
GBS=480               # global batch size
# tokens / step = 480 × 1024 = 491,520

############################################
# 分布式参数
############################################
DISTRIBUTED_ARGS="
  --nproc_per_node ${NPUS_PER_NODE} \
  --nnodes ${NNODES} \
  --node_rank ${NODE_RANK} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT}
"

############################################
# GPT-2 medium 模型参数（严格对齐）
############################################
GPT_ARGS="
  --use-mcore-models \
  --tensor-model-parallel-size ${TP} \
  --pipeline-model-parallel-size ${PP} \

  --num-layers 24 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --ffn-hidden-size 4096 \

  --seq-length ${SEQ_LEN} \
  --max-position-embeddings ${SEQ_LEN} \
  --position-embedding-type learned_absolute \

  --normalization LayerNorm \
  --norm-epsilon 1e-5 \


  --disable-bias-linear \
  --untie-embeddings-and-output-weights \

  --micro-batch-size ${MBS} \
  --global-batch-size ${GBS} \

  --tokenizer-type PretrainedFromHF \
  --tokenizer-name-or-path ${TOKENIZER_PATH} \
  --padded-vocab-size 50257 \

  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \

  --no-masked-softmax-fusion \
  --attention-softmax-in-fp32 \
  --no-gradient-accumulation-fusion \
  --no-load-optim \
  --no-load-rng \

  --bf16
"

############################################
# SophiaG 优化器（与你 config 对齐）
############################################
OPTIM_ARGS="
  --lr ${LR} \
  --min-lr 1e-5 \
  --lr-decay-style cosine \
  --lr-warmup-fraction 0.02 \
  --train-iters 100000 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --weight-decay 2e-1 \
  --optimizer-selection muon \
"

############################################
# 数据参数
############################################
DATA_ARGS="
  --data-path ${DATA_PATH} \
  --split 100,0,0
"

############################################
# 日志 / 评测 / 保存
############################################
OUTPUT_ARGS="
  --log-interval 1 \
  --eval-interval 1000 \
  --eval-iters 200 \
  --save-interval 2000 \

  --tensorboard-dir ${TENSORBOARD_DIR} \
  --log-timers-to-tensorboard \
  --log-throughput \

  --use-wandb \
  --wandb-project gpt2-medium \
  --wandb-exp-name ${EXP_NAME} \
  --wandb-save-dir ${WANDB_DIR}
"

############################################
# 启动训练
############################################
torchrun ${DISTRIBUTED_ARGS} pretrain_gpt.py \
  ${GPT_ARGS} \
  ${OPTIM_ARGS} \
  ${DATA_ARGS} \
  ${OUTPUT_ARGS} \
  --seed 42 \
  --distributed-backend nccl \
  --save ${CKPT_SAVE_DIR} \
  --load ${CKPT_SAVE_DIR} \
  | tee logs/${EXP_NAME}.log
