import os
from collections import Counter

# 标签文件夹路径
train_labels = r"E:\pycharm pro\PyCharm 2025.1.2\yolo\yolo_train\fall_dataset2\labels\train"
val_labels = r"E:\pycharm pro\PyCharm 2025.1.2\yolo\yolo_train\fall_dataset2\labels\val"

# 类别映射
class_map = {
    0: "Fall Detected",
    1: "Normal",
    2: "Resting"
}

def count_labels(label_dir):
    counts = Counter()
    total = 0
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file), "r") as f:
                for line in f:
                    if line.strip():  # 避免空文件
                        cls_id = int(line.split()[0])  # 取类别 ID
                        counts[cls_id] += 1
                        total += 1
    return counts, total

# 统计 train
train_counts, train_total = count_labels(train_labels)
# 统计 val
val_counts, val_total = count_labels(val_labels)

print("📊 Train集样本分布：")
for cls_id, name in class_map.items():
    cnt = train_counts.get(cls_id, 0)
    ratio = cnt / train_total * 100 if train_total > 0 else 0
    print(f"  {name} ({cls_id}): {cnt} ({ratio:.2f}%)")

print("\n📊 Val集样本分布：")
for cls_id, name in class_map.items():
    cnt = val_counts.get(cls_id, 0)
    ratio = cnt / val_total * 100 if val_total > 0 else 0
    print(f"  {name} ({cls_id}): {cnt} ({ratio:.2f}%)")

print("\n✅ 总结：")
print(f"Train 总样本: {train_total}, Val 总样本: {val_total}")
