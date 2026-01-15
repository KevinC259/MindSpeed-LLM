python ./preprocess_data.py \
	--input /sharedata/ckw/dataset/openwebtext/plain_text/1.0.0/6f68e85c16ccc770c0dd489f4008852ea9633604995addd0cd76e293aed9e521 \
	--tokenizer-name-or-path /sharedata/ckw/model_from_hf/gpt2 \
	--output-prefix /sharedata/ckw/dataset/openwebtext-gpt  \
	--tokenizer-type PretrainedFromHF \
	--workers 40 \
	--log-interval 1000
