# DS_tuning1_fixed.py
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
model_path = r"C:\Users\xunyi\.cache\huggingface\hub\models--deepseek-ai--deepseek-llm-7b-chat\snapshots\afbda8b347ec881666061fa67447046fc5164ec8"
jsonl_path = r"E:\pycharm pro\PyCharm 2025.1.2\DS_tuning\deepseek_finetune_data.jsonl"
output_dir = r"E:\pycharm pro\PyCharm 2025.1.2\DS_tuning\fine_tuned_lora_fixed"

# 4060(8GB) 推荐保守配置
batch_size = 1
gradient_accumulation_steps = 8
num_epochs = 3
learning_rate = 3e-6   # 更保守
weight_decay = 0.0
max_length = 256

# LoRA 超参
lora_r = 8
lora_alpha = 16
target_module_keywords = ["q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv"]

# 如果你的 jsonl 每行是 instruction/response 格式，建议开启：
MASK_INSTRUCTION = True   # True = 只对 Response 部分计算 loss（更稳）
INSTRUCTION_MARKER = "### Response:"  # 用于分割 instruction/response 的标识

# reproducible
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
os.makedirs(output_dir, exist_ok=True)

# ================== 稳健 LoRA 实现 ==================
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
            # 如果原权重是浮点就继承 dtype，否则回退到 float32（更稳）
            if torch.is_floating_point(w):
                dtype = w.dtype
            else:
                dtype = torch.float32
        else:
            param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32

        # 更保守的初始化 scale（减少初始波动）
        self.B = nn.Parameter(torch.randn(r, self.in_features, device=param_device, dtype=dtype) * 0.001)
        self.A = nn.Parameter(torch.zeros(self.out_features, r, device=param_device, dtype=dtype))

        if getattr(orig_linear, "bias", None) is not None:
            self.bias = orig_linear.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

    def forward(self, x):
        A, B = self.A, self.B
        if A.device != x.device or A.dtype != x.dtype:
            A = A.to(x.device, dtype=x.dtype)
            B = B.to(x.device, dtype=x.dtype)

        out = self.orig(x)
        lora_out = (x @ B.t()) @ A.t()
        return out + self.scaling * lora_out

# ========== 替换函数 ==========
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

# ================== Dataset（同时创建 labels mask） ==================
class JsonlInstructionDataset(Dataset):
    def __init__(self, jsonl_file: str, tokenizer: AutoTokenizer, max_len: int = 256, mask_instruction: bool = True):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_instruction = mask_instruction

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    jd = json.loads(line)
                except Exception:
                    # 如果不是 json 行，跳过
                    continue
                instruction = jd.get("instruction") or jd.get("prompt") or jd.get("input") or ""
                output = jd.get("output") or jd.get("response") or jd.get("label") or jd.get("answer") or ""
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
        enc = self.tokenizer(
            txt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        labels = item["input_ids"].clone()

        # 将 padding token 的 label 设为 -100（不会计入 loss）
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        labels[labels == pad_id] = -100

        # 可选：只对 Response 部分计算 loss（如果 dataset 使用 Instruction/Response 格式）
        if self.mask_instruction:
            # 找到分割位置的 token index（基于文本检索）
            # 注意：这里用文本级别查找，随后把 instruction 部分对应 token label 设为 -100
            try:
                # 找响应开始在原文本的位置
                split_pos = txt.find(INSTRUCTION_MARKER)
                if split_pos != -1:
                    # 截取 response 文本并 tokenize to get start token index
                    pre = txt[:split_pos + len(INSTRUCTION_MARKER)]
                    pre_enc = self.tokenizer(pre, truncation=True, max_length=self.max_len, padding=False, return_tensors="pt")
                    start_tok = pre_enc["input_ids"].shape[1] if "input_ids" in pre_enc else pre_enc["input_ids"].size(1)
                    # set all labels before start_tok to -100
                    labels[:start_tok] = -100
            except Exception:
                # 如果任何异常，继续保持默认 labels
                pass

        item["labels"] = labels
        return item

# ========== 保存 LoRA ==========
def save_lora_state(model: nn.Module, out_dir: str, tag: Optional[str] = None):
    sd = {}
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            sd[f"{n}.A"] = m.A.detach().cpu()
            sd[f"{n}.B"] = m.B.detach().cpu()
            sd[f"{n}.alpha"] = torch.tensor(m.alpha)
            sd[f"{n}.r"] = torch.tensor(m.r)
    fname = os.path.join(out_dir, f"lora_adapters_{tag}.pt")
    torch.save(sd, fname)
    print(f"✅ Saved LoRA state to {fname}")

# ========== Freeze except LoRA ==========
def freeze_model_except_lora(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad = True
            m.B.requires_grad = True

# ================== 主训练流程 ==================
def main():
    print("Device:", device)
    if use_cuda:
        print("CUDA available. GPU count:", torch.cuda.device_count())

    # BitsAndBytes 4-bit 配置：compute dtype 选 float16（兼顾速度/显存）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print("Loading tokenizer and 4-bit model ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # device_map="auto" 一般已把权重放到正确设备，避免强制 .to(device)
    try:
        if use_cuda:
            model = model.to(device)
    except Exception:
        pass

    print("Applying LoRA ...")
    replaced = find_and_replace_modules(model, target_module_keywords, r=lora_r, alpha=lora_alpha)
    print(f"Replaced {replaced} linear layers with LoRA.")

    # 冻结除 LoRA 外的参数
    freeze_model_except_lora(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable)
    print("Trainable param count:", trainable_param_count)

    ds = JsonlInstructionDataset(jsonl_path, tokenizer, max_len=max_length, mask_instruction=MASK_INSTRUCTION)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    print(f"Samples: {len(ds)}")

    optimizer = AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    num_update_steps_per_epoch = math.ceil(len(dl) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=math.ceil(0.03 * max_train_steps),
        num_training_steps=max_train_steps,
    )

    # 关闭 AMP/GradScaler（4-bit + bnb 在此环境下与 AMP 常冲突）
    use_amp = False

    save_every_steps = 200
    global_step = 0

    try:
        model.train()
        for epoch in range(num_epochs):
            pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{num_epochs}")
            optimizer.zero_grad()
            running_loss = 0.0
            for step, batch in enumerate(pbar):
                # move tensors
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                # forward
                try:
                    outputs = model(**batch)
                    loss = outputs.loss
                except Exception as e:
                    print(f"⚠️ Forward failed at step {step} with error: {e}. Skipping step.")
                    optimizer.zero_grad()
                    continue

                # 基本检查
                if loss is None:
                    print("Warning: loss is None; skipping step")
                    optimizer.zero_grad()
                    continue
                if not torch.isfinite(loss):
                    # 打印被跳过样本的部分文本，帮助定位数据问题
                    try:
                        input_ids = batch.get("input_ids")
                        if input_ids is not None and input_ids.ndim == 2:
                            # 打印第一条样本的解码（前512字符）
                            text_preview = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                            print("⚠️ Invalid loss detected. Sample preview (truncated):")
                            print(text_preview[:512])
                            print(f"input_ids_len={input_ids.shape[1]}")
                    except Exception:
                        pass
                    print("Warning: invalid loss (NaN/Inf); skipping step")
                    optimizer.zero_grad()
                    continue

                # scale for accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()

                # accumulate
                if (step + 1) % gradient_accumulation_steps == 0:
                    # clip grads and check finite
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

                    grads_finite = True
                    for p in trainable:
                        if p.grad is not None:
                            if not torch.all(torch.isfinite(p.grad)):
                                grads_finite = False
                                break

                    if not grads_finite:
                        print("⚠️ Non-finite gradients detected; zeroing grads and skipping optimizer step.")
                        optimizer.zero_grad()
                        continue

                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    global_step += 1

                    cur_loss = loss.item() * gradient_accumulation_steps
                    running_loss += cur_loss
                    pbar.set_postfix({"loss": round(cur_loss, 6), "step": global_step})

                # checkpoint
                if global_step > 0 and global_step % save_every_steps == 0:
                    save_lora_state(model, output_dir, f"step_{global_step}")

            save_lora_state(model, output_dir, f"epoch_{epoch+1}")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — saving interrupted state...")
        save_lora_state(model, output_dir, f"interrupted_step_{global_step}")

    print("Training finished — saving final.")
    save_lora_state(model, output_dir, "final")


if __name__ == "__main__":
    main()
