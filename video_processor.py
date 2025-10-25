import cv2
import time
from queue import Queue
from threading import Lock
from config import *
from logger import get_logger

logger = get_logger(__name__)

# 全局变量
FRAME_QUEUE = Queue(maxsize=FRAME_QUEUE_MAXSIZE)
last_frame = None
cap = None
status = "初始化中"
status_lock = Lock()


def init_video_capture():
    global cap, status
    try:
        cap = cv2.VideoCapture(VIDEO_URL)
        if cap.isOpened():
            with status_lock:
                status = "连接正常"  # 成功则更新为“连接正常”
            return True
        else:
            with status_lock:
                status = "连接异常"  # 失败则更新为“连接异常”
            logger.error(f"视频源 {VIDEO_URL} 初始化失败（无法打开）")
            return False
    except Exception as e:
        with status_lock:
            status = "连接异常"  # 报错也更新为“连接异常”
        logger.error(f"初始化视频源报错: {e}")
        return False


def reconnect_worker():
    """视频重连工作线程"""
    global cap, status
    reconnect_delay = 1
    while True:
        if not cap or not cap.isOpened():
            with status_lock:
                status = "连接异常"
            logger.debug(f"流断开，{reconnect_delay}秒后重连...")
            time.sleep(reconnect_delay)
            try:
                if cap:
                    with status_lock:
                        cap.release()
                cap = cv2.VideoCapture(VIDEO_URL)
                if cap.isOpened():
                    logger.info("视频流重连成功")
                    with status_lock:
                        status = "连接正常"
            except Exception as e:
                logger.error(f"重连失败: {e}")
            reconnect_delay = min(reconnect_delay * 2, 10)
        else:
            reconnect_delay = 1  # 重置间隔
        time.sleep(1)


def get_frame_with_reconnect():
    """获取视频帧，带重连机制"""
    global cap, last_frame
    is_connected = cap and cap.isOpened()

    if is_connected:
        with status_lock:
            ret, frame = cap.read()
        if ret:
            last_frame = frame
            return frame
        else:
            logger.debug("获取帧失败，使用最后一帧")
            return last_frame
    else:
        return last_frame


def generate_frames():
    """生成视频流帧，供前端使用"""
    logger.info("客户端已连接到视频流")
    last_one_frame = None

    while True:
        start_time = time.time()

        try:
            frame_bytes = FRAME_QUEUE.get(timeout=1.0)
            logger.debug(f"成功取帧，大小: {len(frame_bytes)} 字节")
            last_one_frame = frame_bytes
            logger.debug(f"队列剩余 {FRAME_QUEUE.qsize()} 帧")
        except Exception as err:
            logger.debug(f"队列空，重复发送最后一帧: {err}")
            if last_one_frame is None:
                time.sleep(0.1)
                continue
            frame_bytes = last_one_frame

        # 发送帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'\r\n' + frame_bytes + b'\r\n')

        elapsed_time = time.time() - start_time
        sleep_time = max(0, FPS_INTERVAL - elapsed_time)
        time.sleep(sleep_time)

def get_status():
    return status