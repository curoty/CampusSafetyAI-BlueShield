import logging
import time
from typing import TypedDict, Optional


logger = logging.getLogger(__name__)

FALL_CLASS_NAME = "Fall Detected"
FALL_CONFIRMATION_DELAY = 6  # 10秒检测窗口
FALL_RATIO_THRESHOLD = 0.8    # 摔倒帧占比阈值（80%）
position = "未知位置"
COOL_DOWN_TIME = 30  # 检测冷却时间
last_alert_end_time = 0 # 上一次告警窗口结束的时间

class DetectionWindow(TypedDict):
    start_time: Optional[float]  # 允许为float或None
    total_frames: int
    fall_frames: int
    is_window_active: bool

# 用定义的类型初始化全局变量
fall_detection_window: DetectionWindow = {
    "start_time": None,
    "total_frames": 0,
    "fall_frames": 0,
    "is_window_active": False
}
# 线程锁保护全局状态
from threading import Lock
state_lock = Lock()


def fall_detect(result, model):
    global fall_detection_window, COOL_DOWN_TIME, last_alert_end_time

    with state_lock:
        # 1. 解析当前帧的检测结果
        current_frame_fall = False  # 当前帧是否检测到摔倒
        for box in result.boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls]
            if class_name == FALL_CLASS_NAME and confidence > 0.75:
                current_frame_fall = True
                break

        current_time = time.time()
        window = fall_detection_window  # 简化引用

        # 2. 处理“未处于检测窗口”的情况（首次检测到摔倒时开启窗口）
        if not window["is_window_active"] and current_frame_fall:

            time_since_last_alert = current_time - last_alert_end_time
            if time_since_last_alert < COOL_DOWN_TIME:
                logger.info(f"冷却中：剩余{COOL_DOWN_TIME - time_since_last_alert:.1f}秒，不启动窗口")
                return None

            window["start_time"] = current_time
            window["total_frames"] = 1  # 当前帧计入总帧数
            window["fall_frames"] = 1   # 当前帧计入摔倒帧数
            window["is_window_active"] = True
            return None  # 未开启窗口，不返回警告

        # 3. 处于10秒检测窗口内：累计帧计数
        else:

            if window["is_window_active"]:
                if window["start_time"] is None:
                    logger.error("窗口状态异常：is_window_active=True，但 start_time 为 None，强制重置")
                    window["start_time"] = None
                    window["total_frames"] = 0
                    window["fall_frames"] = 0
                    window["is_window_active"] = False
                    return None

                 # 3.1 累计总帧数和摔倒帧数
                window["total_frames"] += 1
                if current_frame_fall:
                    window["fall_frames"] += 1

                # 3.2 判断10秒窗口是否结束
                time_elapsed = current_time - window["start_time"]
                if time_elapsed < FALL_CONFIRMATION_DELAY:
                    # 窗口未结束：继续计数，不返回警告
                    return None

                # 3.3 窗口结束：判断是否满足摔倒条件
                else:
                    # 计算摔倒帧占比（避免除零错误）
                    if window["total_frames"] == 0:
                        fall_ratio = 0.0
                    else:
                        fall_ratio = window["fall_frames"] / window["total_frames"]

                    logger.info(
                        f"10秒检测窗口结束：总帧数={window['total_frames']}, "
                        f"摔倒帧数={window['fall_frames']}, 占比={fall_ratio:.2f}"
                    )

                    # 重置窗口状态（无论是否报警，都重新开始检测）
                    window["start_time"] = None
                    window["total_frames"] = 0
                    window["fall_frames"] = 0
                    window["is_window_active"] = False


                    # 若占比超过80%：发送警告
                    if fall_ratio >= FALL_RATIO_THRESHOLD:
                        last_alert_end_time = current_time
                        alert_msg = f"在{position}, {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}，检测到行人摔倒事件"
                        return alert_msg

                # 占比未达标：不发送警告
                    else:
                        return None
            else:
                return None


def set_position(new_position):
    """更新当前位置信息（如“客厅”“卧室”）"""
    global position
    with state_lock:
        position = new_position