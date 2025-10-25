import json

input_json_path = r"D:\PyCharm 2025.1.1.1\DS_tuning\dsyb\extended_5000_training_data.json"  # 原始 JSON 文件路径
output_jsonl_path = "train_new2.jsonl"  # 转换后的 JSONL 文件路径

with open(input_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    with open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
        for sample in data:
            out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print("done")