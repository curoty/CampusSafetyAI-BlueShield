import json

# 配置参数
SOURCE_FILE = r"D:\PyCharm 2025.1.1.1\DS_tuning\dsyb\train_new2.jsonl"
DEST_FILE = r"D:\PyCharm 2025.1.1.1\DS_tuning\dsyb\filtered_samples.jsonl"  # 目标文件路径
TARGET_INSTRUCTION = "你是本项目的告警助手，根据输入的高危信息，进行告警并输出到最终结果到前端用户"


def filter_and_move_samples():
    matched_samples = []
    remaining_samples = []

    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 检查instruction字段是否完全匹配目标文本
                if "instruction" in data and data["instruction"] == TARGET_INSTRUCTION:
                    matched_samples.append(line)
                else:
                    remaining_samples.append(line)
            except json.JSONDecodeError:
                print(f"警告：无法解析JSON行 - {line}")
                remaining_samples.append(line)  # 解析失败的行保留在原文件

    # 写入匹配样本到目标文件
    with open(DEST_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(matched_samples) + '\n')

    # 写回剩余样本到原文件
    with open(SOURCE_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(remaining_samples) + '\n')

    print(f"处理完成：")
    print(f" - 匹配到 {len(matched_samples)} 个符合条件的样本，已移动到 {DEST_FILE}")
    print(f" - 剩余 {len(remaining_samples)} 个样本保留在原文件")


if __name__ == "__main__":
    filter_and_move_samples()