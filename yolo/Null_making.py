import os

# 图片文件夹
img_dir = r"C:\Users\xunyi\Desktop\null_making"
# 标签文件夹（生成的空txt要放的路径，最好和你的标注目录一致）
label_dir = r"C:\Users\xunyi\Desktop\train_tizhen_save"

os.makedirs(label_dir, exist_ok=True)

for img_file in os.listdir(img_dir):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        name, _ = os.path.splitext(img_file)
        txt_file = name + ".txt"
        txt_path = os.path.join(label_dir, txt_file)

        # 如果不存在对应的 txt 文件，就创建一个空文件
        if not os.path.exists(txt_path):
            with open(txt_path, "w") as f:
                pass  # 创建空文件
            print(f"生成空标签文件: {txt_file}")

print("所有空样本的 txt 文件已生成完成 ✅")
