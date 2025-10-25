import logging
from logging.handlers import TimedRotatingFileHandler  # 按时间+大小切割
import os
from datetime import datetime

def setup_logging():
    # 1. 基础配置（不指定文件，避免与后续 handler 冲突）
    logging.basicConfig(
        level=logging.WARNING,  # 全局默认级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 2. 日志格式（包含线程名、模块名，方便多线程调试）
    log_format = "%(asctime)s - %(levelname)s - %(threadName)s - %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # 3. 创建日志目录
    log_dir = "../FlaskProject1/app_logs"
    os.makedirs(log_dir, exist_ok=True)

    # 4. 控制台输出（实时查看，INFO及以上）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)  # 显示正常流程+警告+错误

    # 5. 文件输出（按天切割，单文件超50MB强制切割，保留7天）
    file_handler = TimedRotatingFileHandler(
        filename=f"{log_dir}/app.log_{datetime.now().strftime('%Y%m%d')}",
        when="D",  # 按天
        interval=1,
        backupCount=7,  # 保留7天
        encoding="utf-8",
    )
    file_handler.maxBytes = 50 * 1024 * 1024  # 单文件最大50MB
    file_handler.backupCount = 5  # 单天内超过大小，保留5个切割文件
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.WARNING)  # 记录所有关键信息

    # 6. 配置全局日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 全局最低级别（不影响 handler 过滤）
    # 先移除默认 handler（避免重复输出）
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # 7. 降低第三方库日志级别（减少冗余）
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("flask").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # Flask 服务器日志