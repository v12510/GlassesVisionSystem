# === src/utils/logger.py ===
import logging
import logging.handlers
from logging import Logger
from typing import Dict, Optional, Any
from pathlib import Path
import sys
import threading
import queue
from datetime import datetime
import traceback
from .config import ConfigManager
from .events import EventBus, EventType, EventFactory


class ThreadSafeLogger:
    """线程安全的异步日志记录器"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[ConfigManager] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init(config or ConfigManager())
            return cls._instance

    def _init(self, config: ConfigManager):
        """初始化日志系统"""
        self.config = config
        self.event_bus = None  # 可选的事件总线引用
        self._log_queue = queue.Queue(-1)
        self._handlers = []

        # 初始化根记录器
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(self._parse_level(
            self.config.get("log.level", "INFO")
        ))

        # 设置异步处理
        self._setup_async_handlers()

        # 捕获未处理异常
        sys.excepthook = self._global_except_hook

        # 初始化各模块日志记录器缓存
        self.module_loggers: Dict[str, logging.Logger] = {}

    def connect_event_bus(self, bus: EventBus):
        """连接事件总线用于错误警报"""
        self.event_bus = bus

    def _setup_async_handlers(self):
        """配置异步日志处理器"""
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(module)s:%(lineno)d] [%(context)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 控制台处理器
        console = logging.StreamHandler()
        console.setFormatter(formatter)

        # 文件处理器（带轮转）
        log_path = Path(self.config.get("log.path", "logs/app.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file = logging.handlers.TimedRotatingFileHandler(
            filename=log_path,
            when="midnight",
            backupCount=self.config.get("log.backup_count", 7),
            encoding="utf-8"
        )
        file.setFormatter(formatter)

        # 使用队列监听器
        self._handlers = [console, file]
        self.queue_listener = logging.handlers.QueueListener(
            self._log_queue, *self._handlers)
        self.queue_listener.start()

    def get_logger(self, module: str) -> logging.Logger:
        """获取模块日志记录器"""
        if module not in self.module_loggers:
            logger = logging.getLogger(module)
            logger.propagate = False
            logger.setLevel(self.root_logger.level)

            # 添加队列处理器
            handler = logging.handlers.QueueHandler(self._log_queue)
            handler.addFilter(ContextFilter(module))
            logger.addHandler(handler)

            self.module_loggers[module] = logger
        return self.module_loggers[module]

    def _global_except_hook(self, exc_type, exc_value, exc_traceback):
        """全局异常捕获"""
        logger = self.get_logger("sys")
        logger.critical(
            "未捕获的全局异常",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

        # 发布系统警报事件
        if self.event_bus:
            stack = "".join(traceback.format_exception(
                exc_type, exc_value, exc_traceback))
            self.event_bus.publish(
                EventFactory.create_alert_event(
                    "Logger",
                    "CRITICAL",
                    f"未处理异常: {exc_value}\n{stack}"
                ),
                priority=1
            )

    def update_config(self, new_config: Dict[str, Any]):
        """动态更新日志配置"""
        if "log.level" in new_config:
            level = self._parse_level(new_config["log.level"])
            self.root_logger.setLevel(level)
            for logger in self.module_loggers.values():
                logger.setLevel(level)

    @staticmethod
    def _parse_level(level: str) -> int:
        """转换日志级别字符串为常量"""
        return getattr(logging, level.upper(), logging.INFO)


class ContextFilter(logging.Filter):
    """上下文信息过滤器"""

    def __init__(self, module: str):
        super().__init__()
        self.module = module

    def filter(self, record):
        record.context = self._get_context()
        record.module = self.module
        return True

    def _get_context(self) -> str:
        """获取运行时上下文信息"""
        thread = threading.current_thread()
        return f"{thread.name}:{thread.ident}"


def log_call(log_level: int = logging.DEBUG):
    """记录函数调用的装饰器"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.log(
                log_level,
                f"调用 {func.__name__} 参数: {args} {kwargs}"
            )
            try:
                result = func(*args, **kwargs)
                logger.log(
                    log_level,
                    f"函数 {func.__name__} 返回: {result}"
                )
                return result
            except Exception as e:
                logger.exception(f"函数 {func.__name__} 抛出异常: {e}")
                raise

        return wrapper

    return decorator
