import torch
from transformers import AutoModelForCausalLM
# 确保导入 LoRALinear
from LoRA_4bit.DS_tuning4 import LoRALinear

# 基座模型路径
model_path = r"D:\PyCharm 2025.1.1.1\DS_tuning\DS_model\deepseek-ai\deepseek-llm-7b-chat"
# LoRA 微调后的权重文件路径
lora_weights_path = r"D:\PyCharm 2025.1.1.1\DS_tuning\fine_tuned_lora_fast\lora_adapters_final.pt"
# 输出保存的纯模型路径
output_model_path = r"D:\PyCharm 2025.1.1.1\DS_tuning\merged_model"

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained(model_path)

# 加载 LoRA 微调权重
lora_weights = torch.load(lora_weights_path)

# 将 LoRA 权重加载到模型中
for name, module in model.named_modules():
    if isinstance(module, LoRALinear):
        # 加载权重
        if f"{name}.A" in lora_weights:
            module.A.data = lora_weights[f"{name}.A"]
        if f"{name}.B" in lora_weights:
            module.B.data = lora_weights[f"{name}.B"]

# 保存合并后的模型（一个完整的模型，不再依赖 LoRA 文件）
model.save_pretrained(output_model_path)

print(f"✅ 纯模型已保存到 {output_model_path}")
