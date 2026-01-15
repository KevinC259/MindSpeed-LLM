import torch
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["USE_BITSANDBYTES"] = "0"
print("CUDA available:", torch.cuda.is_available())

# 模拟你的 import 顺序
from mindspeed_llm.core.transformer.transformer_block import TENorm  # 会触发 megatron 警告
print("After import, CUDA available:", torch.cuda.is_available())
