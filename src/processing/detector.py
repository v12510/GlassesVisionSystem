from ultralytics import YOLO
import requests
from typing import List, Dict


class HybridDetector:
    """混合检测系统"""

    def __init__(self, config: ConfigManager):
        self.local_model = YOLO(config.get("models.yolo_path"))
        self.api_key = config.get("apis.deepseek.key")
        self.use_cloud = config.get("models.use_cloud", False)

    def detect(self, image: np.ndarray) -> List[Dict]:
        # 本地快速检测
        local_results = self._local_detection(image)

        if self.use_cloud:
            cloud_results = self._cloud_analysis(image)
            return self._merge_results(local_results, cloud_results)
        return local_results

    def _local_detection(self, image: np.ndarray):
        results = self.local_model.predict(image)
        return self._parse_yolo_results(results)

    def _cloud_analysis(self, image: np.ndarray):
        # 调用DeepSeek API
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"image": cv2.imencode('.jpg', image)[1].tobytes()}
        response = requests.post(
            "https://api.deepseek.com/v1/vision",
            files=files,
            headers=headers
        )
        return response.json()

    def _parse_yolo_results(self, results):
        # 解析YOLO输出格式
        pass