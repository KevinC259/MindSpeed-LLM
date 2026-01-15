python ./preprocess_data.py \
	--input /sharedata/ckw/dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
	--tokenizer-name-or-path /sharedata/ckw/model_from_hf/qwen2.5-7b-hf \
	--output-prefix /sharedata/ckw/dataset/alpaca  \
	--tokenizer-type PretrainedFromHF \
	--workers 4 \
	--log-interval 1000
