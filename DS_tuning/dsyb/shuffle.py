import random
import os


def read_jsonl(file_path):
    """读取JSONL文件，返回非空行列表（每条样本为一行）"""
    data = []
    try:
        # 以UTF-8编码读取文件，避免中文乱码
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # 去除行首尾空白（避免空行、换行符干扰），非空行才保留
                clean_line = line.strip()
                if clean_line:
                    data.append(clean_line)
                else:
                    print(f"警告：{file_path} 的第{line_num}行为空行，已跳过")
        print(f"成功读取 {file_path}，共获取 {len(data)} 条有效样本")
        return data
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}，请检查路径是否正确")
        exit()  # 文件不存在时终止程序，避免后续错误
    except Exception as e:
        print(f"读取 {file_path} 时出错：{str(e)}")
        exit()


def merge_and_shuffle_jsonl(file1_path, file2_path, output_path="merged_shuffled_dataset.jsonl"):
    """合并两个JSONL文件，随机打乱后写入新文件"""
    # 1. 读取两个文件的样本数据
    data1 = read_jsonl(file1_path)
    data2 = read_jsonl(file2_path)

    # 2. 合并样本（确保无重复逻辑，若有重复需求可额外添加去重）
    merged_data = data1 + data2
    total_count = len(merged_data)
    print(f"\n合并后总样本数：{total_count} 条")

    # 3. 随机打乱（random.shuffle 原地打乱，无需额外赋值）
    # 若需可复现结果，可添加 random.seed(123)（123为自定义种子）
    random.shuffle(merged_data)
    print("样本已完成随机打乱")

    # 4. 写入新JSONL文件（每条样本独占一行）
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # 逐行写入，每条样本后添加换行符
            for line in merged_data:
                f.write(line + '\n')
        print(f"\n新文件已生成：{os.path.abspath(output_path)}")
        print(f"最终写入样本数：{len(merged_data)} 条")
    except Exception as e:
        print(f"写入新文件时出错：{str(e)}")
        exit()


# -------------------------- 配置文件路径 --------------------------
# 替换为你的两个JSONL文件路径（保持原始路径格式）
FILE1 = r"D:\PyCharm 2025.1.1.1\DS_tuning\dsyb\train_new2.jsonl"
FILE2 = r"D:\PyCharm 2025.1.1.1\DS_tuning\dsyb\ddddddddddd.jsonl"
# 输出文件路径（默认与脚本同目录，可自定义）
OUTPUT_FILE = r"/dsyb/train_new3.jsonl"

# -------------------------- 执行合并打乱 --------------------------
if __name__ == "__main__":
    merge_and_shuffle_jsonl(FILE1, FILE2, OUTPUT_FILE)