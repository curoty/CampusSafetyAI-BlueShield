import os 
from glob import glob

# 训练集和验证集图片目录
train_img_dir = r'E:\pycharm pro\PyCharm 2025.1.2\yolo\yolo_train\fall_dataset2\images\train'
val_img_dir = r'E:\pycharm pro\PyCharm 2025.1.2\yolo\yolo_train\fall_dataset2\images\val'

# 支持常见图片格式
train_imgs = glob(os.path.join(train_img_dir, '*.jpg')) + glob(os.path.join(train_img_dir, '*.png'))
val_imgs = glob(os.path.join(val_img_dir, '*.jpg')) + glob(os.path.join(val_img_dir, '*.png'))

print(f"Train集图片数: {len(train_imgs)}")
print(f"Val集图片数: {len(val_imgs)}")