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
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0

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
MASTER_PORT=${USER_MASTER_PORT:-6000}
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

############################################
# 训练超参
############################################
LR=${USER_LR:-6e-4}
OPTIMIZER="sophiamuon"

############################################
# 路径配置
############################################
# Please fill these path configurations or ensure they are set in environment
DATA_PATH="/sharedata/ckw/dataset/openwebtext_text_document"
TOKENIZER_PATH="/sharedata/data/models/Qwen3-0.6B-Base"
CKPT_LOAD_DIR="/sharedata/ckw/ckpt/${EXP_NAME}"

EXP_NAME="qwen3_1point7b_4k_${OPTIMIZER}_lr${LR}"
CKPT_SAVE_DIR="/sharedata/ckw/ckpt/${EXP_NAME}"
TENSORBOARD_DIR="./tensorboard/${EXP_NAME}"
WANDB_DIR="./wandb/${EXP_NAME}"

mkdir -p ${CKPT_SAVE_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${WANDB_DIR}
mkdir -p logs

############################################
# 并行配置
############################################
TP=1
PP=4

############################################
# Batch / Token 对齐
############################################
SEQ_LEN=4096
MBS=2
GBS=256
TRAIN_ITERS=20000

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
# 模型参数 (Qwen3 1.7B)
############################################
GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    
    --num-layers 28 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --ffn-hidden-size 6144 \
    
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    
    --normalization RMSNorm \
    --swiglu \
    --disable-bias-linear \
    
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8 \
    
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --init-method-std 0.01 \
    
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --bf16
"

############################################
# 优化器参数 (Adamuon)
############################################
OPTIM_ARGS="
    --lr ${LR} \
    --min-lr 1.25e-7 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --train-iters ${TRAIN_ITERS} \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    
    --optimizer-selection ${OPTIMIZER} \
    
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion
"

############################################
# 数据参数
############################################
DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
    --no-shared-storage
"

############################################
# 日志 / 评测 / 保存
############################################
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-timers-to-tensorboard \
    --log-throughput \
    
    --use-wandb \
    --wandb-project qwen3-1.7b \
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
