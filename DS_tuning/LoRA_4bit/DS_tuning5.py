# ====================================================
# ✅ DS_tuning5_lora_safe_optimized.py
# - 优化微调参数，平衡原模型能力保留和项目定制能力
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
import numpy as np

# ================== 基础配置 ==================
model_path = r"/DS_model/deepseek-ai/deepseek-llm-7b-chat"
train_jsonl_path = r"/dsyb/train_new2.jsonl"
output_dir = r"D:\PyCharm 2025.1.1.1\DS_tuning\fine_tuned_lora_safe_optimized"

# ================== 核心训练参数 ==================
batch_size = 2
gradient_accumulation_steps = 8
num_epochs = 3
learning_rate = 2e-4
max_length = 224
warmup_ratio = 0.1
weight_decay = 0.01

# ================== LoRA 参数优化 ==================
lora_r = 16
lora_alpha = 32
lora_dropout = 0.1
target_module_keywords = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ================== 优化器参数 ==================
max_grad_norm = 1.0
lr_scheduler_type = "cosine"

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================== LoRA 核心类（带Dropout） ==================
class LoRALinear(nn.Module):
    def __init__(self, wrapped_linear, r=16, alpha=32, dropout=0.1):
        super().__init__()
        self.wrapped = wrapped_linear
        for p in self.wrapped.parameters():
            p.requires_grad = False

        self.device = next(wrapped_linear.parameters()).device
        in_features = getattr(wrapped_linear, "in_features", None)
        out_features = getattr(wrapped_linear, "out_features", None)

        if in_features is None or out_features is None:
            raise ValueError(f"LoRALinear 找不到有效维度: {type(wrapped_linear)}")

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


# ================== 模型替换逻辑 ==================
def replace_with_lora_layers(model: nn.Module, target_keywords, r, alpha, dropout):
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
    replaced = 0
    modules_dict = dict(model.named_modules())

    for name, module in list(modules_dict.items()):
        if not any(k in name.lower() for k in target_keywords):
            continue

        if isinstance(module, (nn.Linear, Linear4bit, Linear8bitLt)):
            parent_name, _, child_name = name.rpartition(".")
            parent = modules_dict[parent_name] if parent_name else model
            setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            replaced += 1

    return replaced


# ================== 数据集 ==================
class JsonlDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=512):
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


# ================== 训练监控 ==================
class TrainingMonitor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.losses = []
        self.learning_rates = []

    def log_step(self, step, loss, lr):
        self.losses.append(loss)
        self.learning_rates.append(lr)

        if step % 100 == 0:
            avg_loss = np.mean(self.losses[-100:])
            print(f"Step {step}: loss = {avg_loss:.4f}, lr = {lr:.2e}")

    def save_training_curve(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates)
        plt.title('Learning Rate')
        plt.xlabel('Step')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close()


# ================== 主训练 ==================
def main():
    gc.collect()
    torch.cuda.empty_cache()

    print(f"🔥 Using device: {device}")
    print(f"Initial CUDA memory: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")

    # 打印训练配置
    print("\n=== 训练配置 ===")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA r: {lora_r}, alpha: {lora_alpha}")
    print(f"Target modules: {target_module_keywords}")

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

    replaced = replace_with_lora_layers(model, target_module_keywords, lora_r, lora_alpha, lora_dropout)
    print(f"✅ LoRA layers replaced: {replaced}")
    assert replaced > 0, "❌ 没有替换任何层，请检查 target_module_keywords"

    # 设置参数梯度
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
    print(f"Trainable params: {total_trainable:,} ({100 * total_trainable / total_params:.4f}%)")

    # 数据集
    ds = JsonlDataset(train_jsonl_path, tokenizer, max_len=max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    print(f"Training samples: {len(ds)}")

    # 优化器
    optimizer = AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

    # 学习率调度
    total_steps = math.ceil(len(dl) / gradient_accumulation_steps) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 训练监控
    monitor = TrainingMonitor(output_dir)

    # 训练循环
    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        optimizer.zero_grad()

        epoch_loss = 0
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                current_lr = scheduler.get_last_lr()[0]
                current_loss = loss.item() * gradient_accumulation_steps

                monitor.log_step(global_step, current_loss, current_lr)
                epoch_loss += current_loss

                pbar.set_postfix({
                    "loss": round(current_loss, 4),
                    "lr": f"{current_lr:.2e}"
                })

                global_step += 1

        # 计算epoch平均损失
        avg_epoch_loss = epoch_loss / (len(dl) / gradient_accumulation_steps)
        print(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")

        # 保存checkpoint
        save_lora_weights(model, output_dir, f"epoch_{epoch}")

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_lora_weights(model, output_dir, "best")
            print(f"🎉 New best model with loss: {best_loss:.4f}")

        torch.cuda.empty_cache()

    # 保存最终模型和训练曲线
    save_lora_weights(model, output_dir, "final")
    monitor.save_training_curve()

    print("✅ Training complete.")
    print(
        f"🧠 Trainable parameters: {total_trainable / 1e6:.2f}M / {total_params / 1e6:.2f}M ({100 * total_trainable / total_params:.4f}%)")
    print(f"📊 Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()