from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SceneContext:
    objects: List[Dict]
    scene_type: str
    risks: List[str]

class NarrativeEngine:
    """多层级文本生成"""
    def __init__(self, config: ConfigManager):
        self.language = config.get("narration.language", "en")
        self.verbosity = config.get("narration.verbosity", 2)
        self._load_templates()

    def generate(self, context: SceneContext) -> str:
        critical = self._critical_alerts(context.risks)
        priority = self._priority_objects(context.objects)
        summary = self._scene_summary(context.scene_type)
        return " ".join(filter(None, [critical, priority, summary]))

    def _critical_alerts(self, risks: List[str]) -> str:
        if "moving_vehicle" in risks:
            return "Warning: Approaching vehicle detected!"
        return ""

    def _priority_objects(self, objects: List[Dict]) -> str:
        # 根据对象优先级生成描述
        pass

    def _scene_summary(self, scene_type: str) -> str:
        # 场景类型描述
        pass

    def _load_templates(self):
        # 加载多语言模板
        pass