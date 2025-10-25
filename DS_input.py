import queue
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from queue import Queue

# ==== 配置部分 ====
BASE_MODEL = r"model/deepseek-ai/deepseek-llm-7b-chat"
LORA_PATH = r"model/deepseek-lora-finetuned/checkpoint-57600"
USE_4BIT = True

model = None
tokenizer = None

# ==== 加载 Tokenizer ====
def run():
    """加载DS模型的线程函数"""
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # ==== 加载基础模型 ====
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
        dtype=torch.bfloat16 if USE_4BIT else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ==== 合并 LoRA 权重 ====
    print("🔹 正在合并 LoRA 权重 ...")
    model = PeftModel.from_pretrained(model, LORA_PATH)


    model.eval()

def ask_ai_question(result_queue: Queue, *args):
    """问答接口（需在模型加载完成后调用）"""
    global model, tokenizer
    if model is None or tokenizer is None:
        result_queue.put("错误：DS模型未加载完成，请稍后再试")
        return

    try:
        # （保持原有的ask_ai_question逻辑不变）
        if len(args) not in (1, 2):
            result_queue.put("参数错误：需传入1个（内容）或2个（指令, 内容）参数")
            return

        if len(args) == 1:
            content = args[0]
            domain = "安全监控告警"
            forbidden_phrases = ["系统性能分析", "视觉检测与语言生成", "全面的安全监护", "高效的异常处理"]
            system_prompt = f"""
                你是{domain}领域的专家，回答必须严格遵循：
                1. 绝对禁止使用以下套话：{', '.join(forbidden_phrases)}
                2. 必须先理解用户的真实需求（如“摔倒后怎么办”是要自救建议），再针对性回答；
                3. 若问题涉及用户自身行动（如摔倒、求助），优先提供具体可操作的步骤；
                4. 若涉及系统，简要说明即可，重点放在用户能理解的内容上。
                """
            prompt = f"{system_prompt}\n用户问{content}\n回答："

        else:
            instruction, content = args
            prompt = f"指令：{instruction}\n输入：{content}\n回答："

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
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "回答：" in response:
            response = response.split("回答：")[-1].strip()
        # 移除可能残留的用户输入
        if response.startswith(content):
            response = response[len(content):].strip()
        result_queue.put(response)
        print(response)

    except Exception as e:
        result_queue.put(f"处理错误：{str(e)}")


def test():
    run()
    a = queue.Queue(maxsize=2)
    ask_ai_question(a, "你是本项目的告警助手，根据输入的高危信息，进行告警并输出到最终结果到前端用户", "监控告警触发：00:01:41于老人居住区识别到摔倒高危事件，帧比例验证通过")
