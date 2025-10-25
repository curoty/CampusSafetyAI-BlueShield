import torch
import bitsandbytes as bnb

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("当前GPU:", torch.cuda.get_device_name(0))
print("bitsandbytes版本:", bnb.__version__)

# 更可靠的bitsandbytes CUDA验证方式
try:
    # 尝试创建一个CUDA张量来验证
    bnb_tensor = bnb.nn.Linear(10, 10).to("cuda")
    print("bitsandbytes支持CUDA: True")
except Exception as e:
    print("bitsandbytes支持CUDA: False，原因:", str(e))