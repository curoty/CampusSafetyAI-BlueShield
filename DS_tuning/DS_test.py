import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from LoRA_4bit.DS_tuning1 import LoRALinear, find_and_replace_modules, target_module_keywords, lora_r, lora_alpha

# ========= 配置 =========
base_model_path = r"C:\Users\xunyi\.cache\huggingface\hub\models--deepseek-ai--deepseek-llm-7b-chat\snapshots\afbda8b347ec881666061fa67447046fc5164ec8"
lora_path = r"E:\pycharm pro\PyCharm 2025.1.2\DS_tuning\fine_tuned_lora_fixed\lora_adapters_final.pt"
output_file = r"E:\pycharm pro\PyCharm 2025.1.2\DS_tuning\test_results.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 加载量化基座模型 =========
print("🚀 正在加载基座模型 (4bit QLoRA)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

# ========= 注入 LoRA 层 =========
print("🔧 正在注入 LoRA 层 ...")
find_and_replace_modules(model, target_module_keywords, r=lora_r, alpha=lora_alpha)

# ========= 加载 LoRA 权重 =========
print("📦 正在加载 LoRA 权重 ...")
lora_state = torch.load(lora_path, map_location="cpu")

missing = 0
for n, m in model.named_modules():
    if isinstance(m, LoRALinear):
        key_A = f"{n}.A"
        key_B = f"{n}.B"
        if key_A in lora_state and key_B in lora_state:
            m.A.data = lora_state[key_A].to(device)
            m.B.data = lora_state[key_B].to(device)
        else:
            missing += 1
print(f"✅ LoRA 权重加载完成（未匹配 {missing} 层）")

# ========= 推理函数 =========
def generate_response(prompt: str, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ========= 测试语句（可改成你自己的） =========
test_prompts = [
    "你好"
]

# ========= 开始测试 =========
print("\n🧠 开始生成回答...\n")
results = []
for i, prompt in enumerate(test_prompts, 1):
    print(f"【输入 {i}】{prompt}")
    response = generate_response(prompt)
    print(f"【输出 {i}】{response}\n")
    results.append(f"【输入{i}】{prompt}\n【输出{i}】{response}\n\n")

# ========= 保存结果 =========
with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(results)
print(f"✅ 所有结果已保存到：{output_file}")
