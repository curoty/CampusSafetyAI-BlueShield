import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ==== 配置部分 ====
BASE_MODEL = r"C:\Users\xunyi\Desktop\FlaskProject1\model\deepseek-ai\deepseek-llm-7b-chat"   # ✅ 改成本地路径（无命名空间）
LORA_PATH = r"C:\Users\xunyi\Desktop\FlaskProject1\model\deepseek-lora-finetuned\checkpoint-57600"          # 训练好的 LoRA 权重路径
USE_4BIT = True                                   # 是否使用4bit量化（省显存）

# ==== 加载 Tokenizer ====
print("🔹 正在加载 Tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# ==== 加载基础模型 ====
print("🔹 正在加载基础模型 ...")
bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16 if USE_4BIT else torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# ==== 合并 LoRA 权重 ====
print("🔹 正在合并 LoRA 权重 ...")
model = PeftModel.from_pretrained(model, LORA_PATH)

# ⚠️ 如果只是想加载LoRA微调，不必 merge
# model = model.merge_and_unload()

model.eval()
print("✅ 模型加载完成，开始聊天！\n")

# ==== 聊天循环 ====
def chat():
    # 提示词前缀：明确要求结合微调知识和通用知识
    # 替换为你的领域（如“安全监控告警”“设备运维”等）
    domain = "安全监控告警"
    forbidden_phrases = ["系统性能分析", "视觉检测与语言生成", "全面的安全监护", "高效的异常处理"]
    system_prompt = f"""
    你是{domain}领域的专家，回答必须严格遵循：
    1. 绝对禁止使用以下套话：{', '.join(forbidden_phrases)}
    2. 必须先理解用户的真实需求（如“摔倒后怎么办”是要自救建议），再针对性回答；
    3. 若问题涉及用户自身行动（如摔倒、求助），优先提供具体可操作的步骤；
    4. 若涉及系统，简要说明即可，重点放在用户能理解的内容上。
    """

    while True:
        user_input = input("🧑‍💬 你：").strip()
        if user_input.lower() in ["exit", "quit", "退出", "q"]:
            print("👋 再见！")
            break

        # 拼接系统提示和用户输入，引导模型平衡两种知识
        prompt = f"{system_prompt}\n用户问：{user_input}\n回答："
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,  # 保留一定随机性，兼顾两种知识的灵活性
                temperature=0.6,  # 中等温度（0.5-0.7），平衡确定性和多样性
                top_p=0.8,  # 适当扩大采样范围，允许通用知识参与
                repetition_penalty=1.1,  # 轻微惩罚重复，避免单一知识占比过高
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 解析并清洗回答
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除提示词前缀，只保留核心回答
        if "回答：" in response:
            response = response.split("回答：")[-1].strip()
        # 移除可能残留的用户输入
        if response.startswith(user_input):
            response = response[len(user_input):].strip()

        print(f"🤖 模型：{response}\n")


if __name__ == "__main__":
    chat()



