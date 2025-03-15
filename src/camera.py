import cv2
import numpy as np
from typing import Optional

class CameraController:
    """多摄像头管理"""
    def __init__(self, config: ConfigManager):
        self.cap = cv2.VideoCapture(config.get("camera.index", 0))
        self.resolution = tuple(config.get("camera.resolution", (1280, 720)))
        self.set_properties()

    def set_properties(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    def capture_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def release(self):
        self.cap.release()