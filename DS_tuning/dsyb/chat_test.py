from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ✅ 你的模型路径
merged_model_path = r"D:\PyCharm 2025.1.1.1\DS_tuning\merged_model"

print("🚀 正在加载模型...")

# ✅ 自动识别 JSON 格式分词器（不要手动关 fast）
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)

model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    torch_dtype=torch.bfloat16,  # 若你的显卡支持 bfloat16 (RTX 40系列支持)
    device_map="auto"
)

print("✅ 模型加载完成！开始聊天测试。\n")

while True:
    query = input("🧠 你：").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 结束测试。")
        break
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("🤖 DS模型：", response)
    print("-" * 60)
