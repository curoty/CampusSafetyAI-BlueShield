from ultralytics import YOLO
import fall_alarm
from config import *
from logger import get_logger
from threading import Lock
import cv2
import alert_send

logger = get_logger(__name__)

# 全局变量
yolo_model = None
alert = None
number_frame = 0
lock = Lock()


def load_yolo_model():
    """加载YOLO模型"""
    global yolo_model
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        yolo_model = yolo_model.to(device=DEVICE)
        logger.info(f"YOLO模型加载成功，使用设备: {DEVICE}")
        return True
    except Exception as e:
        logger.error(f"YOLO加载失败：{e}")
        return False


def process_video_frames():
    """处理视频帧的线程函数"""
    global alert, number_frame, yolo_model
    from video_processor import get_frame_with_reconnect, FRAME_QUEUE

    logger.info("开始处理视频帧...")
    if not yolo_model:
        logger.error("模型未加载，无法处理视频帧")
        return

    while True:
        frame = get_frame_with_reconnect()
        if frame is None:
            continue

        # 模型推理
        results = yolo_model(frame)
        logger.info(f"YOLO检测成功")  # 新增日志
        result = results[0]

        # 跌倒检测
        with lock:
            alert = fall_alarm.fall_detect(result, yolo_model)

            if alert is not None:
                alert_send.set_alert(alert)

        # 筛选高置信度目标
        high_conf_mask = result.boxes.conf > CONFIDENCE_THRESHOLD
        filtered_boxes = result.boxes[high_conf_mask]

        with lock:
            number_frame = len(filtered_boxes)

        # 替换结果集，绘制边界框
        result.boxes = filtered_boxes
        frame = result.plot(
            conf=True,
            line_width=2,
            font_size=10,
            labels=True
        )

        # 入队处理
        if FRAME_QUEUE.qsize() < 30:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                FRAME_QUEUE.put(buffer.tobytes())
                logger.debug(f"新帧入队，队列剩余 {FRAME_QUEUE.qsize()} 帧")
        else:
            logger.debug("队列已满，丢弃当前帧")

def get_number_frame():
    global number_frame, logger
    return number_frame
