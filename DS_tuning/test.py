import torch, gc
gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.is_available())
print(torch.cuda.memory_summary())