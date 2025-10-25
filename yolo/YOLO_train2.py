from ultralytics import YOLO
import os
import pandas as pd
import torch
import shutil


def train_start():
    # ============ 配置 ============
    save_dir = r"fall_dataset_train_3rd/exp_6th"
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train_summary_6th.txt")


    if os.path.exists(save_dir):
        print(f"⚠️ 检测到已有旧目录 {save_dir}，正在清空...")
        shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    torch.cuda.empty_cache()

    # ============ 加载模型 ============
    model = YOLO("yolov10s.pt")

    # ============ 开始训练 ============
    results = model.train(
        data=r"E:\pycharm pro\PyCharm 2025.1.2\yolo\yolo_train\fall_dataset2\fall_dataset2.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        project="fall_dataset_train_3rd",
        name="exp_6th",
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=5,
        patience=150,

        # 数据增强
        degrees=10,
        translate=0.05,
        scale=0.3,
        shear=1.0,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,

        exist_ok=False  # ✅ 覆盖旧目录，不创建 exp_6th2
    )

    # ============ 训练后再验证 ============
    metrics = model.val()  # DetMetrics 对象
    results_dict = getattr(metrics, "results_dict", None)

    # ============ 读取 results.csv ============
    results_csv = os.path.join(save_dir, "results.csv")
    df = None
    final_metrics = None
    if os.path.exists(results_csv):
        try:
            df = pd.read_csv(results_csv)
            if len(df) > 0:
                final_metrics = df.iloc[-1]
        except Exception as e:
            print("读取 results.csv 失败：", e)

    # -------- 辅助函数：找 results_dict 指标 --------
    def find_from_results_dict(include: list, exclude: list = []):
        if not results_dict:
            return None, None
        for k, v in results_dict.items():
            kl = k.lower()
            if all(p.lower() in kl for p in include) and all(p.lower() not in kl for p in exclude):
                return k, v
        return None, None

    _, map50 = find_from_results_dict(['map50'], ['map50-95'])
    _, map50_95 = find_from_results_dict(['map50-95'])
    _, precision = find_from_results_dict(['precision'])
    _, recall = find_from_results_dict(['recall'])

    # -------- 辅助函数：找 results.csv 列 --------
    def find_csv_col(df, substrs):
        if df is None:
            return None
        for col in df.columns:
            lc = col.lower()
            if all(s.lower() in lc for s in substrs):
                return col
        return None

    train_box_col = find_csv_col(df, ['train', 'box'])
    train_cls_col = find_csv_col(df, ['train', 'cls'])
    train_dfl_col = find_csv_col(df, ['train', 'dfl'])
    val_box_col = find_csv_col(df, ['val', 'box'])
    val_cls_col = find_csv_col(df, ['val', 'cls'])
    val_dfl_col = find_csv_col(df, ['val', 'dfl'])

    # ============ 写日志 ============
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("="*20 + " 训练配置 " + "="*20 + "\n")
        f.write("训练集路径：fall_dataset2.yaml\n")
        f.write("训练轮数(Epochs)：150\n")
        f.write("输入图片尺寸(imgsz)：640*640\n")
        f.write("批次大小(Batch)：16\n")
        f.write("优化器(Optimizer)：AdamW\n")
        f.write("初始学习率(lr0)：0.001\n\n")

        f.write("="*20 + " 核心性能指标 " + "="*20 + "\n")
        if map50 is not None:
            f.write(f"最终 mAP@0.5：{map50:.4f}\n")
        if map50_95 is not None:
            f.write(f"最终 mAP@0.5:0.95：{map50_95:.4f}\n")
        if precision is not None:
            f.write(f"最终 Precision：{precision:.4f}\n")
        if recall is not None:
            f.write(f"最终 Recall：{recall:.4f}\n")

        f.write("\n" + "="*20 + " Loss（来自 results.csv） " + "="*20 + "\n")
        if final_metrics is not None:
            def write_csv_item(col, nice_name):
                if col and col in final_metrics.index:
                    try:
                        f.write(f"{nice_name}：{final_metrics[col]:.4f}\n")
                    except Exception:
                        f.write(f"{nice_name}：{final_metrics[col]}\n")
                else:
                    f.write(f"{nice_name}：未找到对应列\n")

            write_csv_item(train_box_col, "训练 - 边界框损失 (train/box_loss)")
            write_csv_item(train_cls_col, "训练 - 类别损失 (train/cls_loss)")
            write_csv_item(train_dfl_col, "训练 - DFL损失 (train/dfl_loss)")

            write_csv_item(val_box_col, "验证 - 边界框损失 (val/box_loss)")
            write_csv_item(val_cls_col, "验证 - 类别损失 (val/cls_loss)")
            write_csv_item(val_dfl_col, "验证 - DFL损失 (val/dfl_loss)")
        else:
            f.write("未找到 results.csv，无法输出 train/val loss。\n")

    print(f"✅ 训练与日志已完成！日志路径：{log_path}")


if __name__ == "__main__":
    train_start()
