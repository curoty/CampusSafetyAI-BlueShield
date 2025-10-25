import threading
from datetime import datetime
from queue import Queue  # 修正：小写 queue 模块
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO

import alert_send
import fall_alarm
from config import *
from logger import get_logger
from video_processor import (
    init_video_capture, reconnect_worker, generate_frames, get_status, status_lock
)
from model_processor import (
    load_yolo_model, process_video_frames, get_number_frame, lock as model_lock
)
from alert_send import fall_deal
import DS_input

logger = get_logger(__name__)

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*")

# 路由定义
@app.after_request
def after_request(response):
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    logger.info(f"{timestamp} - {request.remote_addr} - {request.method} {request.path} - {response.status_code}")
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    with model_lock:
        current_number = get_number_frame()
    with status_lock:
        current_status = get_status()
    return jsonify({
        "number_frame": current_number,
        "status": current_status
    })

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    result_queue = Queue(maxsize=5)
    data = request.get_json()
    question = data.get('message')
    DS_input.ask_ai_question(result_queue, question)
    reply = result_queue.get(timeout=10)
    logger.info(f"AI回复: {reply}")  # 新增日志
    return jsonify({"reply": reply})

@app.route('/position', methods=['POST'])
def receive_position():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "请发送有效的JSON数据"}), 400  # 400表示客户端请求错误
    position = data.get('current_position')  # 安全获取字段
    fall_alarm.set_position(position)
    alert_send.get_position(position)
    return jsonify({"ok": "ok"})



def main():
    """主函数"""
    # 初始化视频捕获
    if not init_video_capture():
        logger.error("无法初始化视频捕获，程序退出")
        return

    reconnect_thread = threading.Thread(target=reconnect_worker)
    reconnect_thread.start()

    logger.info("===== 启动 Flask-YOLO 应用 =====")

    # 启动DS模型线程
    ds_thread = threading.Thread(target=DS_input.run, daemon=True, name="DS_model_loaded")
    ds_thread.start()

    # 加载YOLO模型
    if not load_yolo_model():
        logger.error("YOLO模型加载失败，程序退出")
        return

    #等待DS线程启动完成
    ds_thread.join()

    # 启动后台任务
    socketio.start_background_task(target=process_video_frames)
    socketio.start_background_task(target=fall_deal, socketio=socketio)

    # 运行应用
    socketio.run(app, host=HOST, port=PORT, debug=False)

if __name__ == '__main__':
    main()