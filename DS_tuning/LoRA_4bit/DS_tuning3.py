# ====================================================
# ✅ DS_tuning5_lora_safe.py
# - 支持 4bit 量化 + 正确替换 Linear4bit 层
# - 可在 8GB 显卡上安全微调 DeepSeek / LLaMA / Qwen
# ====================================================

import os
import gc
import math
import json
import random
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_scheduler

# ================== 基础配置 ==================
model_path = r"D:\PyCharm 2025.1.1.1\DS_tuning\DS_model\deepseek-ai\deepseek-llm-7b-chat"
train_jsonl_path = r"D:\PyCharm 2025.1.1.1\DS_tuning\dsyb\train_new2.jsonl"
output_dir = r"D:\PyCharm 2025.1.1.1\DS_tuning\fine_tuned_lora_safe1"

batch_size = 1
gradient_accumulation_steps = 4
num_epochs = 4
learning_rate = 6e-6
max_length = 256
lora_r = 8
lora_alpha = 16
target_module_keywords = ["q_proj", "v_proj", "k_proj", "o_proj"]  # 推荐只在 Attention 注入 LoRA

torch.manual_seed(42)
random.seed(42)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== LoRA 核心类 ==================
class LoRALinear(nn.Module):
    def __init__(self, wrapped_linear, r=8, alpha=16):
        super().__init__()
        self.wrapped = wrapped_linear  # Linear4bit or Linear8bitLt
        for p in self.wrapped.parameters():
            p.requires_grad = False

        # 获取包裹层的设备
        self.device = next(wrapped_linear.parameters()).device

        in_features = getattr(wrapped_linear, "in_features", None)
        out_features = getattr(wrapped_linear, "out_features", None)
        if in_features is None or out_features is None:
            raise ValueError(f"LoRALinear 找不到有效维度: {type(wrapped_linear)}")

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)

        # LoRA 参数初始化并移动到正确设备
        dtype = torch.float16
        self.A = nn.Parameter(torch.zeros(out_features, r, dtype=dtype, device=self.device))
        self.B = nn.Parameter(torch.randn(r, in_features, dtype=dtype, device=self.device) * 0.01)

    def forward(self, x):
        base = self.wrapped(x)
        lora_out = (x @ self.B.t()) @ self.A.t()
        return base + self.scaling * lora_out


# ================== 模型替换逻辑（关键） ==================
def replace_with_lora_layers(model: nn.Module, target_keywords, r, alpha):
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
    replaced = 0
    modules_dict = dict(model.named_modules())

    for name, module in list(modules_dict.items()):
        if not any(k in name.lower() for k in target_keywords):
            continue

        if isinstance(module, (nn.Linear, Linear4bit, Linear8bitLt)):
            parent_name, _, child_name = name.rpartition(".")
            parent = modules_dict[parent_name] if parent_name else model
            setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha))
            replaced += 1

    return replaced


# ================== 数据集 ==================
class JsonlDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                jd = json.loads(line)
                instr = jd.get("instruction", "") or jd.get("prompt", "")
                resp = jd.get("output", "") or jd.get("response", "")
                if not resp:
                    continue
                text = f"### Instruction:\n{instr}\n\n### Response:\n{resp}"
                self.samples.append(text)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        labels = item["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        item["labels"] = labels
        return item


# ================== 保存 LoRA ==================
def save_lora_weights(model, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    state = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            state[f"{name}.A"] = mod.A.detach().cpu()
            state[f"{name}.B"] = mod.B.detach().cpu()
    torch.save(state, os.path.join(out_dir, f"lora_{tag}.pt"))
    print(f"✅ Saved LoRA weights: {tag}")


# ================== 主训练 ==================
def main():
    gc.collect()
    torch.cuda.empty_cache()

    print(f"🔥 Using device: {device}")
    print(f"Initial CUDA memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    replaced = replace_with_lora_layers(model, target_module_keywords, lora_r, lora_alpha)
    print(f"✅ LoRA layers replaced: {replaced}")
    assert replaced > 0, "❌ 没有替换任何层，请检查 target_module_keywords"

    for n, p in model.named_parameters():
        module_name = n.rsplit('.', 1)[0]
        try:
            module = model.get_submodule(module_name)
            p.requires_grad = isinstance(module, LoRALinear)
        except Exception:
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    ds = JsonlDataset(train_jsonl_path, tokenizer, max_len=max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(trainable, lr=learning_rate)
    total_steps = math.ceil(len(dl) / gradient_accumulation_steps) * num_epochs
    scheduler = get_scheduler("linear", optimizer, 0, total_steps)

    model.train()
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix({"loss": round(loss.item() * gradient_accumulation_steps, 6)})

        save_lora_weights(model, output_dir, f"epoch_{epoch}")
        torch.cuda.empty_cache()

    save_lora_weights(model, output_dir, "final")
    print("✅ Training complete.")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🧠 Trainable parameters: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M ({100 * trainable / total:.4f}%)")


if __name__ == "__main__":
    main()
