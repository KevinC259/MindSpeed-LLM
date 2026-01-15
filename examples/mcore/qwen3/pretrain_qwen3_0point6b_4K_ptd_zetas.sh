#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11
NPUS_PER_NODE=${1:-4}
MASTER_ADDR=localhost
MASTER_PORT=6005
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="/sharedata/ckw/ckpts/qwen3_0.6B_zetas_lr3e-4_openwebtext/"
CKPT_SAVE_DIR="/sharedata/ckw/ckpts/qwen3_0.6B_zetas_lr3e-4_openwebtext"
#DATA_PATH="/sharedata/zimoliu/data/alpaca_zh_text_document"
DATA_PATH="/sharedata/ckw/dataset/openwebtext_text_document"
#TOKENIZER_PATH="/sharedata/zimoliu/models/Qwen2.5-7B-Instruct"
TOKENIZER_PATH="/sharedata/data/models/Qwen3-0.6B-Base"
TENSORBOARD_DIR="./tensorboard/qwen3_06b_4k_mcore_ptd"


TP=1
PP=4
MBS=1
GBS=64
SEQ_LENGTH=4096
TRAIN_ITERS=20000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

PROJECT_DIR="qwen3"
EXP_NAME="zetas_qwen3_0.6b_4k_3e-4_openwebtext"
ENTITY_NAME="584272225-south-china-university-of-technology"
GROUP_NAME="private"

WANDB_DIR="./wandb/${EXP_NAME}"
mkdir -p ${WANDB_DIR}
OPT="zetas-vfix"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 3e-4 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --lr-warmup-fraction 0.01 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

GPT_ARGS="
    --use-mcore-models \
    --sequence-parallel \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --num-layers 28 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --ffn-hidden-size 3072 \
    --max-position-embeddings 32768 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --norm-epsilon 1e-6 \
    --no-gradient-accumulation-fusion \
    --attention-softmax-in-fp32 \
    --exit-on-missing-checkpoint \
    --group-query-attention \
    --num-query-groups 8 \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --optimizer-selection zetas \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 400 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --tensorboard-dir $TENSORBOARD_DIR \
    --log-timers-to-tensorboard \
    --log-throughput
    --use-wandb \
    --wandb-project $PROJECT_DIR \
    --wandb-exp-name $EXP_NAME \
    --wandb-entity $ENTITY_NAME \
    --wandb-group $GROUP_NAME \
    --wandb-save-dir $WANDB_DIR \
"

pre=/sharedata/ckw/MindSpeed-LLM
dir=/sharedata/shareenvs/qiuwu-optimizer-dev/bin/
$dir/torchrun $DISTRIBUTED_ARGS $pre/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --lr-warmup-fraction 0.003 \
    --lr-warmup-init 6e-6 \
    --distributed-backend nccl \
    --log-throughput \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/train_mcore_qwen3_0point6b.log