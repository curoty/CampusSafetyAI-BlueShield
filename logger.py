import logging
import log_n

# 初始化日志配置
log_n.setup_logging()

def get_logger(name):
    """获取指定名称的日志器"""
    return logging.getLogger(name)