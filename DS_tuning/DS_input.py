from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

# -------------------------- 1. 配置路径（与训练一致） --------------------------
MODEL_NAME = "DS_model/deepseek-ai/deepseek-llm-7b-chat"  # 本地基础模型路径
SAVE_DIR = "./deepseek-lora-finetuned"  # 训练好的LoRA权重路径

# -------------------------- 2. 关键：4bit量化配置（压显存至~4GB） --------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4bit量化核心配置
    bnb_4bit_use_double_quant=True,  # 双重量化，进一步降内存
    bnb_4bit_quant_type="nf4",  # 适配大模型的量化类型
    bnb_4bit_compute_dtype=torch.float16  # 计算时用float16，平衡速度和精度
)

# -------------------------- 3. GPU加载基础模型（无卸载，纯GPU） --------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,  # 启用4bit量化
    trust_remote_code=True,  # DeepSeek模型必需
    dtype=torch.float16,
    device_map="cuda:0",  # 强制加载到第0块GPU，不用auto
    local_files_only=True,  # 强制本地加载，不联网
    low_cpu_mem_usage=True  # 减少CPU内存占用（仅辅助加载）
)

# -------------------------- 4. 加载LoRA权重（与GPU对齐） --------------------------
peft_model = PeftModel.from_pretrained(
    base_model,
    SAVE_DIR,
    device_map="cuda:0"  # 同样加载到GPU
)
peft_model.eval()  # 推理模式，禁用dropout
print("模型已加载到GPU！显存占用约4GB")

# -------------------------- 5. 加载分词器（与训练一致） --------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="right",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token


# -------------------------- 6. GPU推理函数（速度比CPU快10倍+） --------------------------
def generate_alert(instruction, input_info):
    prompt = f"用户指令：{instruction}\n输入信息：{input_info}\n输出："

    # 输入也放到GPU
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    ).to("cuda:0")

    with torch.no_grad():  # 推理时禁用梯度，节省显存
        outputs = peft_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.split("输出：")[-1].strip()


# -------------------------- 7. 测试GPU推理 --------------------------
test_instruction = "你是本项目的告警助手，根据输入的高危信息，进行告警并输出到最终结果到前端用户"
test_input = "你的职责是什么"
result = generate_alert(test_instruction, test_input)
print("告警生成结果：", result)