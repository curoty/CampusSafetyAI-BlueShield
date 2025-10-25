from datasets import load_dataset  # 修正笔误
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import logging
import os
from typing import List, Dict
# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------- 1. 配置基础参数（适配8G显存） --------------------------
MODEL_NAME = "DS_model/deepseek-ai/deepseek-llm-7b-chat"
DATASET_PATH = "../dsyb/train_new3.jsonl"
SAVE_DIR = "../deepseek-lora-finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256  # 关键：缩短序列长度（告警文本短，128足够）
GRADIENT_ACCUMULATION_STEPS = 1  # 关闭梯度累积，减少中间变量显存占用


# -------------------------- 2. 加载与处理数据集 --------------------------
def load_and_clean_dataset(path):
    dataset = load_dataset("json", data_files=path)["train"]
    logger.info(f"原始数据集大小：{len(dataset)}")

    def filter_valid(example):
        required = ["instruction", "input", "output"]
        return all(
            field in example and isinstance(example[field], str) and len(example[field].strip()) > 0
            for field in required
        )

    clean_dataset = dataset.filter(filter_valid)
    logger.info(f"清洗后数据集大小：{len(clean_dataset)}（过滤掉 {len(dataset) - len(clean_dataset)} 条）")
    return clean_dataset


dataset = load_and_clean_dataset(DATASET_PATH)

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side="right",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token


# 数据格式化与校验
def format_and_validate(example) -> Dict[str, List[int]]:
    instruction = example["instruction"].strip()
    input_info = example["input"].strip()
    output_info = example["output"].strip()

    full_text = f"用户指令：{instruction}\n输入信息：{input_info}\n输出：{output_info}"

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,  # 使用缩短后的长度
        padding=False,
        return_tensors=None,
        add_special_tokens=True
    )

    if not isinstance(tokenized["input_ids"], list) or not all(isinstance(x, int) for x in tokenized["input_ids"]):
        raise ValueError(f"input_ids格式错误：{type(tokenized['input_ids'])}")

    tokenized["labels"] = tokenized["input_ids"].copy()
    if len(tokenized["labels"]) != len(tokenized["input_ids"]):
        raise ValueError(f"长度不匹配：labels={len(tokenized['labels'])}，input_ids={len(tokenized['input_ids'])}")

    # 在format_and_validate函数中添加
    if len(tokenized["input_ids"]) == MAX_LENGTH:
        logger.warning(f"样本被截断至{MAX_LENGTH}token，原始长度可能过长")

    return tokenized


# 安全处理数据集
def safe_map_function(example):
    try:
        return format_and_validate(example)
    except Exception as e:
        logger.warning(f"样本处理失败，跳过：{str(e)}")
        return None


tokenized_dataset = dataset.map(
    safe_map_function,
    remove_columns=dataset.column_names,
    batched=False,
    load_from_cache_file=False
).filter(lambda x: x is not None)

tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
logger.info(f"训练集：{len(tokenized_dataset['train'])}，验证集：{len(tokenized_dataset['test'])}")


# -------------------------- 3. 自定义数据Collator --------------------------
class CustomDataCollator:
    def __init__(self, tokenizer, max_length=256, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer.pad_token_id
        self.ignore_label = -100

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        # 填充input_ids（适配短序列）
        batch_input = self.tokenizer.pad(
            {"input_ids": input_ids},
            max_length=self.max_length,
            padding="max_length",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        input_ids = batch_input["input_ids"]
        attention_mask = batch_input["attention_mask"]

        # 处理labels
        batch_labels = []
        for label in labels:
            if len(label) > self.max_length:
                label = label[:self.max_length]
            else:
                pad_len = self.max_length - len(label)
                label += [self.ignore_label] * pad_len
            batch_labels.append(label)
        labels = torch.tensor(batch_labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


data_collator = CustomDataCollator(
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    pad_to_multiple_of=8
)

# -------------------------- 4. 模型与训练配置（彻底解决梯度问题） --------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    dtype=torch.float16
)

# 1. 应用LoRA
model = get_peft_model(model, lora_config)

# 2. 关键：强制启用所有LoRA参数的梯度，并设置正确精度
for name, param in model.named_parameters():
    if "lora" in name or "adapter" in name:
        param.requires_grad = True  # 强制启用梯度
        # 4bit量化下必须用float32才能计算梯度
        if param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.to(torch.float32)
    else:
        param.requires_grad = False  # 冻结非LoRA参数

# 3. 禁用梯度检查点（与4bit量化不兼容，是梯度丢失的主因）
# model.gradient_checkpointing_enable()  # 必须注释掉这行

# 确认可训练参数
model.print_trainable_parameters()


# -------------------------- 5. 训练参数（优化器适配） --------------------------
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=1,  # 8G显存安全值
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=2,
    num_train_epochs=4,
    learning_rate=5e-5,
    logging_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=False,  # 关闭FP16，避免4bit量化下的精度冲突
    optim="adamw_torch",  # 改用基础优化器，兼容性更好
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none"
)
# -------------------------- 6. 执行训练并保存模型 --------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator
)

# 执行训练
trainer.train()

# 强制保存LoRA权重
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
logger.info(f"LoRA权重已保存至：{SAVE_DIR}")
if not os.path.exists(os.path.join(SAVE_DIR, "adapter_config.json")):
    logger.error("保存失败：未找到adapter_config.json")
else:
    logger.info("保存成功，文件列表：" + str(os.listdir(SAVE_DIR)))


# -------------------------- 7. 推理函数 --------------------------
def generate_alert(instruction, input_info):
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16
    )

    if not os.path.exists(os.path.join(SAVE_DIR, "adapter_config.json")):
        raise FileNotFoundError(f"LoRA权重文件缺失：{SAVE_DIR}")

    peft_model = PeftModel.from_pretrained(
        base_model,
        SAVE_DIR,
        from_transformers=True
    )
    peft_model.eval()

    prompt = f"用户指令：{instruction}\n输入信息：{input_info}\n输出："
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
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


# 测试推理
if os.path.exists(os.path.join(SAVE_DIR, "adapter_config.json")):
    test_instruction = "你是本项目的告警助手，根据输入的高危信息，进行告警并输出到最终结果到前端用户"
    test_input = "安全事件：康复器材区在09:45:10检测到持续摔倒状态，需要生成告警"
    print("生成结果：", generate_alert(test_instruction, test_input))
else:
    logger.warning("未找到LoRA权重文件，跳过推理测试")