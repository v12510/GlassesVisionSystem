import numpy as np
from typing import List, Dict, Tuple, Deque
from collections import deque, defaultdict
from dataclasses import dataclass
import time


@dataclass
class ObjectTrack:
    """物体跟踪数据"""
    id: int
    cls: str
    positions: Deque[Tuple[float, float]]  # 中心点坐标队列
    timestamps: Deque[float]
    attributes: Dict[str, any]


class SceneAnalyzer:
    """多模态场景理解引擎"""

    def __init__(self, config):
        self.config = config
        self.context_window = config.get("scene.context_window", 5)  # 跟踪帧数
        self.tracks: Dict[int, ObjectTrack] = {}  # 当前跟踪的物体
        self.last_update = time.time()

        # 场景规则配置
        self.scene_rules = config.get("scene.rules", {
            "crosswalk": {"required": ["person", "traffic_light"],
                          "threshold": 0.7},
            "office": {"required": ["chair", "computer"],
                       "optional": ["desk", "book"]}
        })

        # 风险判定参数
        self.risk_params = {
            "speed_threshold": 0.5,  # 像素/秒
            "distance_threshold": 200  # 像素距离
        }

    def analyze(self,
                current_objects: List[Dict],
                frame: np.ndarray) -> Dict:
        """主分析流程"""
        # 更新跟踪状态
        self._update_tracks(current_objects)

        # 多维度分析
        scene_type = self._classify_scene(current_objects)
        risks = self._assess_risks()
        relations = self._analyze_spatial_relations(current_objects)
        activities = self._detect_activities()

        return {
            "scene_type": scene_type,
            "potential_risks": risks,
            "spatial_relations": relations,
            "active_objects": activities,
            "timestamp": time.time()
        }

    def _update_tracks(self, objects: List[Dict]):
        """物体跟踪与状态更新"""
        current_ids = set()
        for obj in objects:
            obj_id = obj["id"]
            bbox = obj["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if obj_id not in self.tracks:
                self.tracks[obj_id] = ObjectTrack(
                    id=obj_id,
                    cls=obj["class"],
                    positions=deque(maxlen=self.context_window),
                    timestamps=deque(maxlen=self.context_window),
                    attributes=obj.get("attributes", {})
                )
            else:
                self.tracks[obj_id].attributes.update(obj["attributes"])

            self.tracks[obj_id].positions.append((cx, cy))
            self.tracks[obj_id].timestamps.append(time.time())
            current_ids.add(obj_id)

        # 移除丢失的跟踪目标
        lost_ids = set(self.tracks.keys()) - current_ids
        for lid in lost_ids:
            del self.tracks[lid]

    def _classify_scene(self, objects: List[Dict]) -> str:
        """基于规则和机器学习的场景分类"""
        # 统计物体类别
        class_counts = defaultdict(int)
        for obj in objects:
            class_counts[obj["class"]] += 1

        # 应用规则判断
        for scene_type, rule in self.scene_rules.items():
            required_match = all(
                cls in class_counts
                for cls in rule["required"]
            )
            optional_match = any(
                cls in class_counts
                for cls in rule.get("optional", [])
            )

            if required_match and optional_match:
                return scene_type

        # 使用机器学习模型（示例）
        # return self._ml_scene_classification(frame)

        return "unknown"

    def _assess_risks(self) -> List[str]:
        """风险因素分析"""
        risks = []
        for track in self.tracks.values():
            # 移动速度分析
            if len(track.positions) >= 2:
                dx = track.positions[-1][0] - track.positions[0][0]
                dy = track.positions[-1][1] - track.positions[0][1]
                dt = track.timestamps[-1] - track.timestamps[0]
                speed = np.sqrt(dx ** 2 + dy ** 2) / dt if dt > 0 else 0

                if speed > self.risk_params["speed_threshold"]:
                    risks.append(f"fast_moving_{track.cls}")

        # 近距离物体检测
        main_center = (frame.shape[1] // 2, frame.shape[0] // 2)  # 假设用户位置在画面中心
        for obj in current_objects:
            obj_center = ((obj["bbox"][0] + obj["bbox"][2]) // 2,
                          (obj["bbox"][1] + obj["bbox"][3]) // 2)
            distance = np.sqrt(
                (obj_center[0] - main_center[0]) ** 2 +
                (obj_center[1] - main_center[1]) ** 2
            )
            if distance < self.risk_params["distance_threshold"]:
                risks.append(f"nearby_{obj['class']}")

        return list(set(risks))

    def _analyze_spatial_relations(self,
                                   objects: List[Dict]) -> Dict[str, List]:
        """空间关系分析"""
        relations = defaultdict(list)
        reference_point = (frame.shape[1] // 2, frame.shape[0] // 2)  # 以用户为中心

        for obj in objects:
            # 相对位置判断
            obj_center = ((obj["bbox"][0] + obj["bbox"][2]) // 2,
                          (obj["bbox"][1] + obj["bbox"][3]) // 2)

            # 水平方向
            if obj_center[0] < reference_point[0] - 100:
                relations["left"].append(obj["class"])
            elif obj_center[0] > reference_point[0] + 100:
                relations["right"].append(obj["class"])

            # 垂直方向
            if obj_center[1] < reference_point[1] - 50:
                relations["front"].append(obj["class"])
            elif obj_center[1] > reference_point[1] + 50:
                relations["back"].append(obj["class"])

        return dict(relations)

    def _detect_activities(self) -> List[Dict]:
        """活动物体检测"""
        activities = []
        for track in self.tracks.values():
            if len(track.positions) < 2:
                continue

            # 计算移动轨迹方差
            positions = np.array(track.positions)
            variance = np.var(positions, axis=0).mean()

            # 分析属性变化
            if "action" in track.attributes:
                activities.append({
                    "id": track.id,
                    "class": track.cls,
                    "activity": track.attributes["action"],
                    "intensity": variance
                })
        return activities

    def _ml_scene_classification(self, frame: np.ndarray) -> str:
        """基于深度学习的场景分类（示例）"""
        # 此处可以集成场景分类模型
        # 示例使用颜色直方图简单判断
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

        if hist[120:150, 200:].sum() > 0.3 * hist.sum():
            return "outdoor"
        else:
            return "indoor"

    def _analyze_social_context(self):
        """社交场景分析（如人群密度）"""
        person_count = sum(1 for obj in self.tracks.values()
                           if obj.cls == "person")
        if person_count > 5:
            return "crowded"
        return "normal"

    def _predict_trajectory(self, track: ObjectTrack) -> Tuple[float, float]:
        """基于卡尔曼滤波的轨迹预测"""
        if len(track.positions) < 3:
            return track.positions[-1] if track.positions else (0, 0)

        # 实现简单的线性预测
        dx = track.positions[-1][0] - track.positions[-2][0]
        dy = track.positions[-1][1] - track.positions[-2][1]
        return (
            track.positions[-1][0] + dx,
            track.positions[-1][1] + dy
        )

    def _assess_lighting_condition(self, frame: np.ndarray) -> str:
        """环境光照评估"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray)
        if avg_brightness < 50:
            return "low_light"
        elif avg_brightness > 200:
            return "overexposed"
        return "normal"

