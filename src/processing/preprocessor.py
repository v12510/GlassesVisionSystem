import cv2
import numpy as np

class ImagePreprocessor:
    """图像预处理流水线"""
    def __init__(self, config: ConfigManager):
        self.denoise_strength = config.get("processing.denoise", 3)
        self.clahe = cv2.createCLAHE(
            clipLimit=config.get("processing.clahe_clip", 2.0),
            tileGridSize=(8,8)
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        img = self._denoise(image)
        img = self._enhance_contrast(img)
        return self._white_balance(img)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(img, None,
            self.denoise_strength, self.denoise_strength, 7, 21)

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    def _white_balance(self, img: np.ndarray) -> np.ndarray:
        # 自定义白平衡算法
        pass