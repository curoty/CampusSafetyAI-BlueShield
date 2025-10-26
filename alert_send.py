import time
from queue import Queue, Empty
from threading import Lock
from logger import get_logger
import DS_input
import threading
import winsound

logger = get_logger(__name__)

# 全局变量
alert = None
position = "None"
lock = Lock()

def set_alert(new_alert):
    """设置告警信息"""
    global alert
    with lock:
        alert = new_alert
    if new_alert is not None:
        logger.info(f"set_alert 已接收告警：{new_alert}")  # 新增日志

def get_alert():
    """获取当前告警信息"""
    global alert
    with lock:
        return alert

def get_position(data):
    global position
    position = data

def fall_deal(socketio):
    """跌倒告警处理线程"""
    alert_queue = Queue(maxsize=5)
    while True:
        time.sleep(5)
        current_alert = get_alert()
        if current_alert is not None:
            threading.Thread(
                target=DS_input.ask_ai_question,  # 只传函数名，不带括号
                args=(
                    alert_queue,
                    "你是本项目的告警助手，根据输入的高危信息，进行告警并输出到最终结果到前端用户",
                    current_alert
                ),  # 位置参数用元组传递
                daemon=True
            ).start()
            try:
                reply = alert_queue.get(timeout=30)
                set_alert(None)
            except Empty:
                set_alert(None)  # 清空告警，避免重复尝试
                continue  # 继续下一次循环
            # 生成警告信息
            socketio.emit('fall_alert', {'message': reply.replace('未知区域', position)})
            # 异步播放警告音
            threading.Thread(
                target=lambda: winsound.PlaySound("Alarm _audio/alarm.wav", winsound.SND_FILENAME),
                daemon=True
            ).start()
            logger.warning(f"{reply.replace('未知区域', position)}")
        else:
            # 新增日志，明确无告警时的状态
            logger.debug("当前无跌倒告警信息")
        time.sleep(0.1)