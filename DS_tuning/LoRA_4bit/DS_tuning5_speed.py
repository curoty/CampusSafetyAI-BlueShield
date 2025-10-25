# DS_tuning5_speed_opt.py
# 说明：
# - 预先 token 化并保存为 train_tokens.pt（如果存在则跳过）
# - 自动尝试扩大 batch（逐步试探），避免 OOM 崩溃
# - 使用 AMP + GradScaler、pin_memory、num_workers 优化
# - 可启用 gradient_checkpointing（节省显存，牺牲部分速度）
# 运行前：确认 transformers, bitsandbytes, torch 已正确安装并支持 4bit
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
# 或更保守：
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc, torch
gc.collect()
torch.cuda.empty_cache()

import os
import gc
import json
import math
import time
import random
import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_scheduler
from torch.cuda.amp import autocast, GradScaler

# ================== 配置 ==================
model_path = r"D:\PyCharm 2025.1.1.1\DS_tuning\DS_model\deepseek-ai\deepseek-llm-7b-chat"
train_jsonl_path = r"D:\PyCharm 2025.1.1.1\DS_tuning\dsyb\train_new2.jsonl"
token_cache_path = train_jsonl_path + ".tokens.pt"   # token cache 文件
output_dir = r"D:\PyCharm 2025.1.1.1\DS_tuning\fine_tuned_lora_safe_speed_opt"

# 初始训练参数（脚本会尝试增大 batch）
init_batch_size = 2            # 起始 batch
max_trial_batch = 6            # 最多尝试到的 batch（注意：受显存限制）
gradient_accumulation_steps = 4
num_epochs = 3
learning_rate = 5e-5
max_length = 224
warmup_ratio = 0.05
weight_decay = 0.01
max_grad_norm = 1.0
lr_scheduler_type = "cosine"

# LoRA（保守稳定配置）
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
target_module_keywords = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 其它
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Windows 和 Linux 上的 num_workers 建议
if os.name == "nt":
    default_num_workers = 2  # Windows 下稳定值（可改为 0~2）
else:
    default_num_workers = 4  # Linux 可更高

# ================== LoRA wrapper（兼容 Linear4bit/8bitLt） ==================
class LoRALinear(nn.Module):
    def __init__(self, wrapped_linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.wrapped = wrapped_linear
        for p in self.wrapped.parameters():
            p.requires_grad = False

        w = getattr(wrapped_linear, "weight", None)
        if w is None or w.ndim != 2:
            # 尝试用属性推断
            in_f = getattr(wrapped_linear, "in_features", None)
            out_f = getattr(wrapped_linear, "out_features", None)
            if in_f is None or out_f is None:
                raise RuntimeError("无法推断 wrapped_linear 维度")
            out_features, in_features = out_f, in_f
        else:
            out_features, in_features = w.shape

        self.device = next(wrapped_linear.parameters()).device
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        dtype = torch.float16
        self.A = nn.Parameter(torch.zeros(out_features, r, dtype=dtype, device=self.device))
        self.B = nn.Parameter(torch.randn(r, in_features, dtype=dtype, device=self.device) * 0.01)

    def forward(self, x):
        base = self.wrapped(x)
        lora_out = (self.dropout(x) @ self.B.t()) @ self.A.t()
        return base + self.scaling * lora_out

# ================== 替换模块 ==================
def replace_with_lora_layers(model, target_keywords, r, alpha, dropout):
    try:
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
    except Exception:
        Linear4bit, Linear8bitLt = (), ()
    replaced = 0
    modules = dict(model.named_modules())
    for name, module in list(modules.items()):
        if not any(k in name.lower() for k in target_keywords):
            continue
        if isinstance(module, (nn.Linear, ) + (Linear4bit, Linear8bitLt)):
            parent_name, _, child_name = name.rpartition(".")
            parent = modules[parent_name] if parent_name else model
            setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
    return replaced

# ================== 预处理：tokenize 并缓存 ==================
def build_or_load_token_cache(tokenizer, jsonl_path, cache_path, max_len):
    if os.path.exists(cache_path):
        print("加载 token 缓存:", cache_path)
        data = torch.load(cache_path)
        return data
    print("开始预处理并构建 token 缓存（这将花一点时间）...")
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenize lines"):
            line = line.strip()
            if not line:
                continue
            try:
                jd = json.loads(line)
            except:
                continue
            instr = jd.get("instruction") or jd.get("prompt") or ""
            out = jd.get("output") or jd.get("response") or jd.get("answer") or ""
            if not out:
                continue
            text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
            enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            # squeeze to small dict of tensors
            samples.append({k: v.squeeze(0) for k, v in enc.items()})
    print(f"保存 token 缓存到 {cache_path} （样本数：{len(samples)})")
    torch.save(samples, cache_path)
    return samples

# ================== Dataset（直接读取 token tensors） ==================
class TokenDataset(Dataset):
    def __init__(self, token_list):
        self.data = token_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        labels = item["input_ids"].clone()
        # pad token id 可能为 None
        pad_id = item.get("attention_mask", None)  # not use, just placeholder
        # replace pad tokens with -100, but we used padding="max_length" so pad token exists
        # get pad id using tokenizer later if needed
        item["labels"] = labels
        return item

# ================== 主训练 ==================
def main():
    gc.collect(); torch.cuda.empty_cache()
    print("开始训练（优化版）")

    # BitsAndBytes 4bit 配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 预处理并缓存 tokens（若存在则加载）
    token_list = build_or_load_token_cache(tokenizer, train_jsonl_path, token_cache_path, max_length)
    dataset = TokenDataset(token_list)
    print("样本数量:", len(dataset))

    # 加载量化模型（单次）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},  # <-- 强制单卡
        trust_remote_code=True
    )

    # 可选：启用 gradient checkpoint（注释掉可提高速度但更吃显存）
    # model.gradient_checkpointing_enable()

    replaced = replace_with_lora_layers(model, target_module_keywords, lora_r, lora_alpha, lora_dropout)
    print(f"✅ LoRA layers replaced: {replaced}")
    if replaced == 0:
        raise RuntimeError("没有替换任何层，请检查 target_module_keywords 或 bitsandbytes 是否正确安装。")

    # 设置需要训练的参数（仅 LoRA）
    for n, p in model.named_parameters():
        module_name = n.rsplit('.', 1)[0]
        try:
            module = model.get_submodule(module_name)
            p.requires_grad = isinstance(module, LoRALinear)
        except Exception:
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {total_trainable:,} ({100*total_trainable/total_params:.6f}%)")

    # 自动尝试扩增 batch（从 init_batch_size 开始）
    chosen_batch = None
    for trial_batch in range(init_batch_size, max_trial_batch+1):
        try:
            dl_tmp = DataLoader(dataset, batch_size=trial_batch, shuffle=False, num_workers=0, pin_memory=True)
            # 尝试一次前向（取第一个 batch）来检测 OOM
            sample = next(iter(dl_tmp))
            # 将 sample 移到 device（按字段）
            sample = {k: v.to(device) for k,v in sample.items()}
            with torch.no_grad():
                # use autocast to reflect runtime
                with autocast(dtype=torch.float16):
                    _ = model(**sample)
            # 如果没有抛 OOM，则接受该 batch
            chosen_batch = trial_batch
            print(f"可用 batch_size 试验通过: {trial_batch}")
            del dl_tmp, sample
            torch.cuda.empty_cache()
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda out of memory" in msg:
                print(f"batch_size={trial_batch} OOM，尝试下一个较小值（skip）")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                # 其他错误抛出
                raise
        except StopIteration:
            chosen_batch = trial_batch
            break

    if chosen_batch is None:
        chosen_batch = init_batch_size
        print("未能通过增大 batch 测试，使用初始 batch_size:", chosen_batch)
    else:
        print("最终选定 batch_size:", chosen_batch)

    # DataLoader 最终配置（使用发现的 batch）
    dl = DataLoader(dataset, batch_size=chosen_batch, shuffle=True,
                    num_workers=default_num_workers, pin_memory=True)

    effective_batch = chosen_batch * gradient_accumulation_steps
    print(f"Effective batch size = {chosen_batch} * {gradient_accumulation_steps} = {effective_batch}")

    optimizer = AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    total_steps = math.ceil(len(dl) / gradient_accumulation_steps) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_scheduler(lr_scheduler_type, optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = GradScaler()
    model.train()
    global_step = 0
    best_loss = float('inf')

    # 训练循环（使用 autocast + scaler）
    for epoch in range(1, num_epochs+1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{num_epochs}")
        optimizer.zero_grad()
        running_epoch_loss = 0.0
        step_in_epoch = 0

        for step, batch in enumerate(pbar):
            # move batch to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with autocast(dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                current_loss = loss.item() * gradient_accumulation_steps
                running_epoch_loss += current_loss
                # 每 500 global_step 打印一次，减少开销
                if global_step % 500 == 0:
                    print(f"[Global step {global_step}] loss={current_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
                global_step += 1
                step_in_epoch += 1

        # 计算并打印 epoch 平均 loss（按已更新步数）
        if step_in_epoch > 0:
            avg_epoch_loss = running_epoch_loss / step_in_epoch
        else:
            avg_epoch_loss = float("nan")
        print(f"Epoch {epoch} avg loss: {avg_epoch_loss:.4f}")

        # 保存 LoRA
        state = {}
        for name, mod in model.named_modules():
            if isinstance(mod, LoRALinear):
                state[f"{name}.A"] = mod.A.detach().cpu()
                state[f"{name}.B"] = mod.B.detach().cpu()
        torch.save(state, os.path.join(output_dir, f"lora_epoch{epoch}.pt"))
        print("Saved LoRA epoch:", epoch)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(state, os.path.join(output_dir, "lora_best.pt"))
            print("New best loss:", best_loss)

        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)

    print("训练结束，保存 final LoRA")
    torch.save(state, os.path.join(output_dir, "lora_final.pt"))
    print("Done. Best loss:", best_loss)

if __name__ == "__main__":
    main()
