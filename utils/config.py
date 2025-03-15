import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    """动态配置管理"""
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = self._load_config(config_path)
        self.watcher = self._start_file_watcher()

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(Path(__file__).parent.parent / path) as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default

    def _start_file_watcher(self):
        # 实现配置文件热重载
        pass