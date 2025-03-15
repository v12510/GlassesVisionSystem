import threading
import queue
from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type, Optional
from datetime import datetime
import logging

# 初始化日志
event_logger = logging.getLogger("EventSystem")

class EventType(Enum):
    """系统核心事件类型"""
    FRAME_CAPTURED = 1           # 携带原始帧数据
    IMAGE_PREPROCESSED = 2       # 携带预处理后的图像
    OBJECTS_DETECTED = 3         # 携带检测结果列表
    SCENE_ANALYZED = 4           # 携带场景分析结果
    NARRATION_GENERATED = 5      # 携带文本描述
    AUDIO_SYNTHESIZED = 6        # 携带音频数据
    USER_COMMAND = 7             # 携带语音指令文本
    SYSTEM_ALERT = 8             # 携带警报信息
    LOW_POWER_MODE = 9           # 进入低功耗模式

@dataclass(frozen=True)
class Event:
    """事件基类"""
    type: EventType
    timestamp: datetime
    data: Any
    source: str                  # 事件来源模块

class EventBus:
    """线程安全的事件总线"""
    def __init__(self):
        self._subscriptions: Dict[EventType, List[Callable]] = {}
        self._queue = queue.PriorityQueue()  # (优先级, 事件)
        self._lock = threading.RLock()
        self._worker = threading.Thread(target=self._process_events, daemon=True)
        self._worker.start()
        self._handlers = {}

    def subscribe(self,
                event_type: EventType,
                handler: Callable[[Event], None],
                priority: int = 5):
        """订阅指定类型事件"""
        with self._lock:
            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []
            self._subscriptions[event_type].append((priority, handler))
            self._subscriptions[event_type].sort(reverse=True, key=lambda x: x[0])

    def publish(self, event: Event, priority: int = 5):
        """发布事件到总线"""
        self._queue.put((-priority, event))  # 负号实现优先级队列

    def _process_events(self):
        """事件处理线程"""
        while True:
            try:
                priority, event = self._queue.get()
                self._dispatch(event)
            except Exception as e:
                event_logger.error(f"事件处理失败: {e}")

    def _dispatch(self, event: Event):
        """分发事件给订阅者"""
        with self._lock:
            handlers = self._subscriptions.get(event.type, [])
            for _, handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    event_logger.error(f"事件处理回调错误: {e}")

    def register_handler(self,
                       event_type: EventType,
                       handler: Callable,
                       priority: int = 5):
        """装饰器注册处理器"""
        def decorator(func):
            self.subscribe(event_type, func, priority)
            return func
        return decorator

class EventFactory:
    """事件生成工厂"""
    @staticmethod
    def create_frame_event(source: str, frame: np.ndarray) -> Event:
        return Event(
            type=EventType.FRAME_CAPTURED,
            timestamp=datetime.now(),
            data={"frame": frame},
            source=source
        )

    @staticmethod
    def create_alert_event(source: str, level: str, message: str) -> Event:
        return Event(
            type=EventType.SYSTEM_ALERT,
            timestamp=datetime.now(),
            data={"level": level, "msg": message},
            source=source
        )

    @staticmethod
    def create_command_event(command: str) -> Event:
        return Event(
            type=EventType.USER_COMMAND,
            timestamp=datetime.now(),
            data={"command": command},
            source="UserInteraction"
        )

