import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Union


DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_DIR = "logs"
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FMT = '%Y-%m-%d %H:%M:%S'
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5


def get_logger(
        logger_name: str = 'llm_logger',
        level: Union[int, str] = None,
        log_file: str = None,
        log_dir: str = DEFAULT_LOG_DIR,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        formatter: logging.Formatter = None
    ) -> logging.Logger:

    def _resolve_log_level(level: Union[int, str]) -> int:
        """解析日志级别"""
        # 优先级：参数 > 环境变量 > 默认值
        if level is None:
            level = os.getenv('LOG_LEVEL', '')
        if isinstance(level, str):
            level = level.upper()
            return getattr(logging, level, DEFAULT_LOG_LEVEL)
        return level or DEFAULT_LOG_LEVEL

    logger = logging.getLogger(logger_name)
    
    # 避免重复添加处理器
    if logger.hasHandlers():
        return logger

    # 解析日志级别
    resolved_level = _resolve_log_level(level)
    logger.setLevel(resolved_level)

    # 创建格式化器
    formatter = formatter or logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FMT)

    # 始终添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 添加文件处理器（如果配置了日志文件）
    if log_file:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_file)
            file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error("Failed to create file handler: %s", e, exc_info=True)

    return logger


# 创建全局logger实例
llm_logger = get_logger()
