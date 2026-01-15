#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7    

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="/sharedata/ckw/ckpt/qwen2.5-0.5b-hf-zetas-lr6e-4_openwebtext/"
CKPT_SAVE_DIR="/sharedata/ckw/ckpt/qwen2.5-0.5b-hf-zetas-lr6e-4_openwebtext"
DATA_PATH="/sharedata/ckw/dataset/openwebtext_text_document"
TOKENIZER_PATH="/sharedata/data/models/Qwen3-0.6B-Base"
TENSORBOARD_DIR="./tensorboard/qwen3_7b_4k_mcore_ptd_test_opt"

TP=1
PP=4
SEQ_LEN=513
MBS=1
GBS=64

PROJECT_DIR="qwen3-0dot6b-optimizer"
OPT=$1
EXP_NAME="zetas_qwen25_0.5b_4k_6e-4_openwebtext"
ENTITY_NAME="584272225-south-china-university-of-technology"
GROUP_NAME="private"

WANDB_DIR="./wandb/${EXP_NAME}"
mkdir -p ${WANDB_DIR}
mkdir -p ${TENSORBOARD_DIR}


DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 12  \
    --hidden-size 1024  \
    --ffn-hidden-size 4864 \
    --num-attention-heads 16  \
    --max-position-embeddings ${SEQ_LEN} \
    --seq-length ${SEQ_LEN} \
    --disable-bias-linear \
    --add-qkv-bias \
    --group-query-attention \
    --num-query-groups 4 \
    --use-flash-attn \
    --swiglu \
    --use-fused-swiglu \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --use-fused-rmsnorm \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --use-fused-rotary-pos-emb \
    --untie-embeddings-and-output-weights \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --lr 6e-4 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --lr-warmup-fraction 0.01 \
    --init-method-std 0.01 \
    --train-iters 20000 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --optimizer-selection zetas \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

CKPT_ARGS="
    \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
    
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

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/pretrain_mcore_qwen25_7b_32k.log
