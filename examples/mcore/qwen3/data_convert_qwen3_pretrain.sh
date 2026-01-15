# 修改 ascend-toolkit 路径

python ./preprocess_data.py \
    --input /sharedata/ckw/dataset/openwebtext/plain_text/1.0.0/6f68e85c16ccc770c0dd489f4008852ea9633604995addd0cd76e293aed9e521/ \
    --tokenizer-name-or-path /sharedata/data/models/Qwen3-0.6B-Base \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix /sharedata/ckw/dataset/openwebtext \
    --json-keys text \
    --workers 4 \
    --log-interval 1000