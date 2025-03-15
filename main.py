# === src/main.py ===
import time
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
import numpy as np

from utils.config import ConfigManager
from utils.events import EventBus, EventType, EventFactory
from utils.logger import ThreadSafeLogger

from core.camera import CameraController
from core.processing import ImagePreprocessor, HybridDetector, SceneAnalyzer
from core.narration import NarrativeEngine, TTSService
from core.user_interaction import UserInteraction


class VisionAssistant:
    """主控制系统，协调所有模块工作"""

    def __init__(self):
        # 初始化基础组件
        self._init_shutdown_handler()
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 加载配置和核心模块
        self.config = ConfigManager("config/app_config.yaml")
        self.event_bus = EventBus()
        self.logger = ThreadSafeLogger(self.config)
        self.logger.connect_event_bus(self.event_bus)

        # 初始化硬件相关模块
        self._init_hardware_modules()

        # 初始化处理流水线
        self._init_processing_pipeline()

        # 注册事件监听
        self._register_event_handlers()

        # 系统状态监控
        self.frame_counter = 0
        self.last_throughput = 0.0

    def _init_shutdown_handler(self):
        """注册优雅关机处理"""
        signal.signal(signal.SIGINT, self._graceful_shutdown)
        signal.signal(signal.SIGTERM, self._graceful_shutdown)

    def _init_hardware_modules(self):
        """初始化硬件相关模块"""
        self.camera = CameraController(self.config, self.event_bus)
        self.user_input = UserInteraction(self.config, self.event_bus)
        self.tts = TTSService(self.config, self.event_bus)

        # 等待硬件初始化完成
        self._wait_for_hardware_ready()

    def _init_processing_pipeline(self):
        """初始化图像处理流水线"""
        self.preprocessor = ImagePreprocessor(self.config)
        self.detector = HybridDetector(self.config)
        self.scene_analyzer = SceneAnalyzer(self.config)
        self.narrator = NarrativeEngine(self.config)

    def _register_event_handlers(self):
        """注册核心事件处理器"""
        # 图像处理流水线
        self.event_bus.subscribe(
            EventType.FRAME_CAPTURED,
            self._process_frame
        )

        # 用户指令处理
        self.event_bus.subscribe(
            EventType.USER_COMMAND,
            self._handle_user_command
        )

        # 系统警报处理
        self.event_bus.subscribe(
            EventType.SYSTEM_ALERT,
            self._handle_system_alert
        )

    def _wait_for_hardware_ready(self):
        """等待硬件初始化完成"""
        retries = 5
        while not self.camera.is_ready() and retries > 0:
            self.logger.get_logger("Main").warning("等待摄像头初始化...")
            time.sleep(1)
            retries -= 1

        if not self.camera.is_ready():
            raise RuntimeError("摄像头初始化失败")

    def start(self):
        """启动系统主循环"""
        self.running = True
        self.logger.get_logger("Main").info("系统启动")

        # 启动用户交互线程
        self.user_input.start_listening()

        # 启动摄像头采集线程
        self.camera.start_capturing()

        # 主监控循环
        while self.running:
            self._monitor_system_status()
            time.sleep(1)  # 降低监控频率

    def _process_frame(self, event):
        """处理图像帧的完整流水线"""
        start_time = time.time()
        frame = event.data["frame"]

        try:
            # 阶段1：图像预处理
            processed_frame = self.preprocessor.process(frame)

            # 阶段2：物体检测（并行执行）
            detection_future = self.executor.submit(
                self.detector.detect, processed_frame)

            # 阶段3：场景分析（依赖检测结果）
            detection_results = detection_future.result()
            scene_data = self.scene_analyzer.analyze(detection_results, processed_frame)

            # 阶段4：生成描述
            narration = self.narrator.generate(scene_data)

            # 阶段5：语音合成（异步）
            self.tts.speak(narration, priority=scene_data.get("priority", 1))

            # 性能统计
            self._update_performance_metrics(start_time)

        except Exception as e:
            self.logger.get_logger("Pipeline").error(
                f"处理流程异常: {str(e)}", exc_info=True)
            self.event_bus.publish(EventFactory.create_alert_event(
                "Main", "ERROR", f"处理流程异常: {str(e)}"))

    def _update_performance_metrics(self, start_time: float):
        """更新性能指标并自适应调整"""
        latency = time.time() - start_time
        self.frame_counter += 1

        # 每10帧调整一次
        if self.frame_counter % 10 == 0:
            current_throughput = 10 / (time.time() - self.last_throughput)
            self.last_throughput = time.time()

            self.logger.get_logger("Perf").info(
                f"系统吞吐量: {current_throughput:.1f}fps, 延迟: {latency:.2f}s")

            # 自适应调整策略
            if latency > 1.0:
                self._adjust_processing_params("reduce")
            elif latency < 0.5 and current_throughput < 15:
                self._adjust_processing_params("increase")

    def _adjust_processing_params(self, action: str):
        """动态调整处理参数"""
        current_res = self.config.get("processing.resolution", (1280, 720))

        if action == "reduce":
            new_res = (
                max(current_res[0] // 2, 640),
                max(current_res[1] // 2, 480)
            )
            self.config.update("processing.resolution", new_res)
            self.logger.get_logger("Adaptive").warning(
                f"降低处理分辨率至 {new_res}")

        elif action == "increase":
            new_res = (
                min(current_res[0] * 2, 1920),
                min(current_res[1] * 2, 1080)
            )
            self.config.update("processing.resolution", new_res)
            self.logger.get_logger("Adaptive").info(
                f"提升处理分辨率至 {new_res}")

    def _handle_user_command(self, event):
        """处理用户语音指令"""
        command = event.data["command"]
        self.logger.get_logger("UI").info(f"收到用户指令: {command}")

        match command.lower():
            case "启动":
                self._start_processing()
            case "停止":
                self._stop_processing()
            case "切换模式":
                self._toggle_processing_mode()
            case "电量查询":
                self._report_battery_status()
            case _:
                self.tts.speak("无法识别的指令")

    def _handle_system_alert(self, event):
        """处理系统警报事件"""
        alert_msg = f"[{event.data['level']}] {event.data['msg']}"
        self.logger.get_logger("Alert").warning(alert_msg)

        # 高风险警报立即语音提示
        if event.data["level"] in ["CRITICAL", "HIGH"]:
            self.tts.speak(alert_msg, priority=0)

    def _start_processing(self):
        """启动处理流程"""
        if not self.running:
            self.running = True
            self.camera.start_capturing()
            self.tts.speak("系统已启动")

    def _stop_processing(self):
        """停止处理流程"""
        self.running = False
        self.camera.stop_capturing()
        self.tts.speak("系统已暂停")

    def _toggle_processing_mode(self):
        """切换处理模式"""
        current_mode = self.config.get("processing.mode", "balanced")
        new_mode = "fast" if current_mode == "quality" else "quality"
        self.config.update("processing.mode", new_mode)
        self.tts.speak(f"已切换至{new_mode}模式")

    def _report_battery_status(self):
        """报告电量状态"""
        # 假设电源管理模块提供电量信息
        battery_level = 75  # 从PowerManager获取
        self.tts.speak(f"当前电量剩余{battery_level}%")

    def _monitor_system_status(self):
        """系统健康状态监控"""
        # 检查摄像头状态
        if not self.camera.is_ready():
            self.event_bus.publish(EventFactory.create_alert_event(
                "Main", "CRITICAL", "摄像头失去连接"))

        # 检查处理延迟
        # （实际实现需要更详细的健康检查）

    def _graceful_shutdown(self, signum, frame):
        """优雅关闭系统"""
        self.logger.get_logger("Main").info("收到关机信号，开始清理...")
        self.running = False

        # 关闭所有模块
        self.camera.release()
        self.user_input.stop_listening()
        self.tts.stop()
        self.executor.shutdown(wait=True)

        # 等待资源释放
        time.sleep(1)
        self.logger.get_logger("Main").info("系统安全关闭")
        exit(0)


if __name__ == "__main__":
    try:
        assistant = VisionAssistant()
        assistant.start()
    except Exception as e:
        ThreadSafeLogger().get_logger("Main").critical(
            "系统启动失败", exc_info=True)
        raise

