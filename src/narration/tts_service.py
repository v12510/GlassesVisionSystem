import os
import queue
import threading
import time
import sounddevice as sd
import numpy as np
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass
from .text_generator import SceneContext  # 依赖文本生成模块


@dataclass
class VoiceProfile:
    """语音合成配置参数"""
    voice_id: str = "female_01"
    speed: float = 1.0  # 0.5 ~ 2.0
    pitch: float = 0.0  # -1.0 ~ 1.0
    emotion: str = "neutral"  # happy/sad/angry


class TTSService:
    """多引擎语音合成服务"""

    def __init__(self, config_manager):
        self.config = config_manager
        self.profile = self._load_voice_profile()
        self.audio_queue = queue.Queue(maxsize=10)  # 缓冲队列
        self.cache_dir = "audio_cache"
        self.running = True
        self._init_audio_device()
        self._start_consumer_thread()

        # 初始化引擎
        self.online_enabled = self.config.get("tts.use_online", True)
        self.offline_engine = self._init_offline_engine()

    def _load_voice_profile(self) -> VoiceProfile:
        """从配置加载语音参数"""
        return VoiceProfile(
            voice_id=self.config.get("tts.voice_id", "female_02"),
            speed=self.config.get("tts.speed", 1.0),
            pitch=self.config.get("tts.pitch", 0.0),
            emotion=self.config.get("tts.emotion", "neutral")
        )

    def _init_audio_device(self):
        """初始化音频输出设备"""
        try:
            self.device_info = sd.query_devices(
                device=self.config.get("audio.output_device", None),
                kind='output'
            )
            sd.check_input_settings(
                device=self.device_info['name'],
                channels=1,
                dtype='float32'
            )
        except sd.PortAudioError as e:
            print(f"Audio device error: {e}")
            raise

    def _init_offline_engine(self):
        """加载本地TTS引擎"""
        try:
            # 示例：加载VITS本地模型
            from TTS.api import TTS
            return TTS(model_name="vits_zh", progress_bar=False, gpu=False)
        except ImportError:
            return None

    def _start_consumer_thread(self):
        """启动音频播放线程"""
        self.consumer_thread = threading.Thread(
            target=self._audio_consumer,
            daemon=True
        )
        self.consumer_thread.start()

    def speak(self, text: str, priority: int = 0):
        """提交文本到合成队列（优先级机制）"""
        if not text.strip():
            return

        # 检查缓存
        audio_data = self._check_cache(text)
        if audio_data is not None:
            self.audio_queue.put((audio_data, priority))
            return

        # 提交合成任务
        threading.Thread(
            target=self._synthesize_task,
            args=(text, priority),
            daemon=True
        ).start()

    def _synthesize_task(self, text: str, priority: int):
        """合成任务处理线程"""
        try:
            # 优先使用在线引擎
            if self.online_enabled:
                audio = self._online_synthesis(text)
            else:
                audio = self._offline_synthesis(text)

            if audio is not None:
                self._add_to_cache(text, audio)
                self.audio_queue.put((audio, priority))
        except Exception as e:
            print(f"TTS synthesis failed: {e}")

    def _online_synthesis(self, text: str) -> Optional[np.ndarray]:
        """SenseVoice在线合成"""
        api_key = self.config.get("apis.sensevoice.key")
        if not api_key:
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "voice": self.profile.voice_id,
            "speed": self.profile.speed,
            "pitch": self.profile.pitch,
            "emotion": self.profile.emotion,
            "format": "wav"
        }

        try:
            response = requests.post(
                "https://api.sensevoice.ai/v1/synthesize",
                json=payload,
                headers=headers,
                timeout=5
            )
            response.raise_for_status()
            return self._convert_audio(response.content)
        except requests.exceptions.RequestException as e:
            print(f"Online TTS failed: {e}")
            return None

    def _offline_synthesis(self, text: str) -> Optional[np.ndarray]:
        """离线合成后备方案"""
        if self.offline_engine is None:
            return None

        try:
            # 使用本地TTS引擎生成
            output_path = "temp_audio.wav"
            self.offline_engine.tts_to_file(
                text=text,
                speaker=self.profile.voice_id,
                file_path=output_path
            )
            with open(output_path, "rb") as f:
                return self._convert_audio(f.read())
        except Exception as e:
            print(f"Offline TTS failed: {e}")
            return None

    def _convert_audio(self, data: bytes) -> np.ndarray:
        """转换音频数据到numpy数组"""
        import io
        from scipy.io import wavfile

        buffer = io.BytesIO(data)
        rate, audio = wavfile.read(buffer)
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        return audio

    def _audio_consumer(self):
        """音频播放线程"""
        while self.running:
            try:
                # 按优先级排序播放
                items = []
                while not self.audio_queue.empty():
                    items.append(self.audio_queue.get())

                # 排序：高优先级先播放
                items.sort(key=lambda x: -x[1])

                for audio, _ in items:
                    sd.play(audio, samplerate=24000)
                    sd.wait()
            except Exception as e:
                print(f"Audio playback error: {e}")
            time.sleep(0.1)

    def _check_cache(self, text: str) -> Optional[np.ndarray]:
        """检查音频缓存"""
        if not self.config.get("tts.cache_enabled", True):
            return None

        cache_file = os.path.join(
            self.cache_dir,
            f"{hash(text)}.npy"
        )
        if os.path.exists(cache_file):
            try:
                return np.load(cache_file)
            except:
                return None
        return None

    def _add_to_cache(self, text: str, audio: np.ndarray):
        """添加音频到缓存"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        cache_file = os.path.join(
            self.cache_dir,
            f"{hash(text)}.npy"
        )
        np.save(cache_file, audio)

    def update_profile(self, new_profile: Dict[str, Any]):
        """动态更新语音参数"""
        self.profile = VoiceProfile(
            voice_id=new_profile.get("voice_id", self.profile.voice_id),
            speed=new_profile.get("speed", self.profile.speed),
            pitch=new_profile.get("pitch", self.profile.pitch),
            emotion=new_profile.get("emotion", self.profile.emotion)
        )

    def stop(self):
        """停止服务"""
        self.running = False
        sd.stop()
        self.consumer_thread.join(timeout=1)


# === 使用示例 ===
if __name__ == "__main__":
    from utils.config import ConfigManager

    config = ConfigManager("config/default.yaml")
    tts = TTSService(config)

    # 合成并播放
    tts.speak("前方三米检测到行人，建议减速", priority=2)
    time.sleep(5)
    tts.stop()