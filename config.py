import torch

# 模型配置
YOLO_MODEL_PATH = r"model/yolo/best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 视频配置
VIDEO_URL = 1  # 可以替换为RTSP地址
FRAME_QUEUE_MAXSIZE = 100
TARGET_FPS = 30
FPS_INTERVAL = 1.0 / TARGET_FPS

# Flask配置
SECRET_KEY = 'your_secret_key'
HOST = '0.0.0.0'
PORT = 5000

# 检测配置
CONFIDENCE_THRESHOLD = 0.75