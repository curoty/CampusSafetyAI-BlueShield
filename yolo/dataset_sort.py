import os
import shutil

# 文件夹路径
img_dir = r"C:\Users\xunyi\Desktop\null"
label_dir = r"C:\Users\xunyi\Desktop\qian"
null_dir = r"C:\Users\xunyi\Desktop\unprocessed"

# 如果 null 文件夹不存在，创建它
os.makedirs(null_dir, exist_ok=True)

# 遍历图片文件夹
for img_file in os.listdir(img_dir):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        # 图片名（去掉扩展名）
        name, _ = os.path.splitext(img_file)
        label_file = name + ".txt"
        label_path = os.path.join(label_dir, label_file)

        # 如果没有对应的标签文件，就移动图片
        if not os.path.exists(label_path):
            src_img_path = os.path.join(img_dir, img_file)
            dst_img_path = os.path.join(null_dir, img_file)
            shutil.move(src_img_path, dst_img_path)
            print(f"移动无标签图片: {img_file}")

print("处理完成！")
