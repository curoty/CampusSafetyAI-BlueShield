# ====================================================
# DS_tuning4_fast.py
# - 仅训练 + 保存 LoRA
# - 不使用验证集，提高速度
# - 8GB GPU 安全
# ====================================================

import os
import json
import math
import random
import time
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, BitsAndBytesConfig
from torch.optim import AdamW
from tqdm.auto import tqdm

# ================== 配置 ==================
model_path = r"/DS_model/deepseek-ai/deepseek-llm-7b-chat"
train_jsonl_path = r"/train_new1.jsonl"
output_dir = r"D:\PyCharm 2025.1.1.1\DS_tuning\fine_tuned_lora_fast1"

batch_size = 1
gradient_accumulation_steps = 8
num_epochs = 5
learning_rate = 1e-5
weight_decay = 0.0
max_length = 256

lora_r = 8
lora_alpha = 16
target_module_keywords = ["q_proj", "k_proj", "v_proj", "o_proj"]

MASK_INSTRUCTION = True
INSTRUCTION_MARKER = "### Response:"

torch.manual_seed(42)
random.seed(42)
os.makedirs(output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== LoRA ==================
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
        param_device = w.device if w is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = w.dtype if (w is not None and torch.is_floating_point(w)) else torch.float32

        self.B = nn.Parameter(torch.randn(r, self.in_features, device=param_device, dtype=dtype) * 0.001)
        self.A = nn.Parameter(torch.zeros(self.out_features, r, device=param_device, dtype=dtype))

        self.bias = getattr(orig_linear, "bias", None)
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
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

def freeze_model_except_lora(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad = True
            m.B.requires_grad = True

# ================== Dataset ==================
class JsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_len: int = 256, mask_instr: bool = True):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_instr = mask_instr
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    jd = json.loads(line)
                except:
                    continue
                instr = jd.get("instruction") or jd.get("prompt") or jd.get("input") or ""
                output = jd.get("output") or jd.get("response") or jd.get("answer") or ""
                if not output:
                    continue
                if instr:
                    text = f"### Instruction:\n{instr}\n\n### Response:\n{output}"
                else:
                    text = output
                self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        txt = self.samples[idx]
        enc = self.tokenizer(txt, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        labels = item["input_ids"].clone()
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        labels[labels == pad_id] = -100

        if self.mask_instr:
            try:
                pos = txt.find(INSTRUCTION_MARKER)
                if pos != -1:
                    prefix = txt[:pos + len(INSTRUCTION_MARKER)]
                    pre_enc = self.tokenizer(prefix, truncation=True, max_length=self.max_len, padding=False, return_tensors="pt")
                    n = pre_enc["input_ids"].shape[1]
                    labels[:n] = -100
            except:
                pass
        item["labels"] = labels
        return item

# ================== 保存 LoRA ==================
def save_lora_state(model: nn.Module, out_dir: str, tag: str):
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
    cfg = {"r": lora_r, "alpha": lora_alpha, "target_modules": target_module_keywords,
           "fan_in_fan_out": False, "bias": "none", "task_type": "CAUSAL_LM"}
    with open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved LoRA weights to {fname}")

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

    replaced = find_and_replace_modules(model, target_module_keywords, lora_r, lora_alpha)
    print(f"✅ Replaced {replaced} layers with LoRA")

    freeze_model_except_lora(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable))

    ds = JsonlDataset(train_jsonl_path, tokenizer, max_len=max_length, mask_instr=MASK_INSTRUCTION)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    optimizer = AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    total_steps = math.ceil(len(dl)/gradient_accumulation_steps)*num_epochs
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=int(0.03*total_steps), num_training_steps=total_steps)

    model.train()
    global_step = 0
    for epoch in range(1, num_epochs+1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{num_epochs}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (step+1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix({"loss": round(loss.item()*gradient_accumulation_steps, 6)})

        # 保存 LoRA
        save_lora_state(model, output_dir, f"epoch_{epoch}")

    save_lora_state(model, output_dir, "final")
    print("✅ Training complete")

if __name__ == "__main__":
    main()
