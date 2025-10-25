# ====================================================
# ✅ DS_tuning2_lora_save_fixed.py
# - 仅保存每轮结束后的 lora_adapters_epoch_X.pt + 最终 lora_adapters_final.pt
# - 自动生成 adapter_config.json
# ====================================================

import os
import json
import math
import random
import numpy as np
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    BitsAndBytesConfig
)
from torch.optim import AdamW
from tqdm.auto import tqdm


# ================== 配置 ==================
model_path = r"/DS_model/deepseek-ai/deepseek-llm-7b-chat"
jsonl_path = r"D:\PyCharm 2025.1.1.1\DS_tuning\dsyb\train_new2.jsonl"
output_dir = r"D:\PyCharm 2025.1.1.1\DS_tuning\fine_tuned_lora_fixed"

batch_size = 1
gradient_accumulation_steps = 8
num_epochs = 4
learning_rate = 3e-6
weight_decay = 0.0
max_length = 192

lora_r = 8
lora_alpha = 16
target_module_keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv"]

MASK_INSTRUCTION = True
INSTRUCTION_MARKER = "### Response:"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
os.makedirs(output_dir, exist_ok=True)


# ================== LoRA 实现 ==================
class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.orig = orig_linear
        for p in self.orig.parameters():
            p.requires_grad = False

        self.in_features = getattr(orig_linear, "in_features", None)
        self.out_features = getattr(orig_linear, "out_features", None)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)

        w = getattr(orig_linear, "weight", None)
        if w is not None:
            param_device = w.device
            dtype = w.dtype if torch.is_floating_point(w) else torch.float32
        else:
            param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32

        self.B = nn.Parameter(torch.randn(r, self.in_features, device=param_device, dtype=dtype) * 0.001)
        self.A = nn.Parameter(torch.zeros(self.out_features, r, device=param_device, dtype=dtype))

        if getattr(orig_linear, "bias", None) is not None:
            self.bias = orig_linear.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

    def forward(self, x):
        # 强制 A 和 B 与输入 x 的数据类型一致
        A, B = self.A.to(x.dtype), self.B.to(x.dtype)
        out = self.orig(x)
        lora_out = (x @ B.t()) @ A.t()
        return out + self.scaling * lora_out


def find_and_replace_modules(model: nn.Module, keywords: List[str], r: int, alpha: int):
    replace_count = 0
    modules_dict = dict(model.named_modules())
    for name, module in list(modules_dict.items()):
        if not name:
            continue
        parent_name, _, child_name = name.rpartition(".")
        if not child_name:
            continue
        parent = modules_dict.get(parent_name, None) if parent_name else model
        if parent is None:
            continue
        mod = getattr(parent, child_name, None)
        if mod is None:
            continue
        if isinstance(mod, nn.Linear) and any(k in name.lower() for k in keywords):
            setattr(parent, child_name, LoRALinear(mod, r=r, alpha=alpha))
            replace_count += 1
    return replace_count


# ================== Dataset ==================
class JsonlInstructionDataset(Dataset):
    def __init__(self, jsonl_file: str, tokenizer: AutoTokenizer, max_len: int = 256, mask_instruction: bool = True):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_instruction = mask_instruction

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    jd = json.loads(line)
                except Exception:
                    continue
                instruction = jd.get("instruction") or jd.get("prompt") or jd.get("input") or ""
                output = jd.get("output") or jd.get("response") or jd.get("answer") or ""
                if not output:
                    continue
                if instruction:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                else:
                    text = output
                self.examples.append(text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        txt = self.examples[idx]
        enc = self.tokenizer(txt, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        labels = item["input_ids"].clone()

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        labels[labels == pad_id] = -100

        if self.mask_instruction:
            try:
                split_pos = txt.find(INSTRUCTION_MARKER)
                if split_pos != -1:
                    pre = txt[:split_pos + len(INSTRUCTION_MARKER)]
                    pre_enc = self.tokenizer(pre, truncation=True, max_length=self.max_len, padding=False, return_tensors="pt")
                    start_tok = pre_enc["input_ids"].shape[1]
                    labels[:start_tok] = -100
            except Exception:
                pass

        item["labels"] = labels
        return item


# ================== 保存 LoRA 权重 + 配置 ==================
def save_lora_state(model: nn.Module, out_dir: str, tag: str):
    """保存 LoRA 权重文件 + adapter_config.json"""
    os.makedirs(out_dir, exist_ok=True)
    sd = {}
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            sd[f"{n}.A"] = m.A.detach().cpu()
            sd[f"{n}.B"] = m.B.detach().cpu()
            sd[f"{n}.alpha"] = torch.tensor(m.alpha)
            sd[f"{n}.r"] = torch.tensor(m.r)
    fname = os.path.join(out_dir, f"lora_adapters_{tag}.pt")
    torch.save(sd, fname)
    print(f"✅ Saved LoRA weights to {fname}")

    # 自动生成 adapter_config.json
    config = {
        "r": lora_r,
        "alpha": lora_alpha,
        "target_modules": target_module_keywords,
        "fan_in_fan_out": False,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    config_path = os.path.join(out_dir, "adapter_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved adapter_config.json to {config_path}")


def freeze_model_except_lora(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad = True
            m.B.requires_grad = True


# ================== 主训练 ==================
def main():
    print("Device:", device)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    print("Applying LoRA...")
    replaced = find_and_replace_modules(model, target_module_keywords, r=lora_r, alpha=lora_alpha)
    print(f"Replaced {replaced} layers with LoRA.")

    freeze_model_except_lora(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable))

    ds = JsonlInstructionDataset(jsonl_path, tokenizer, max_len=max_length, mask_instruction=MASK_INSTRUCTION)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    optimizer = AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    total_steps = num_epochs * math.ceil(len(dl) / gradient_accumulation_steps)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps)

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                running_loss += loss.item()
                pbar.set_postfix({"loss": round(loss.item() * gradient_accumulation_steps, 6)})

        # 每个 epoch 保存一次
        save_lora_state(model, output_dir, f"epoch_{epoch+1}")

    # 最终保存
    save_lora_state(model, output_dir, "final")
    print("✅ Training complete.")


if __name__ == "__main__":
    main()
