import argparse
import json
from pathlib import Path

from src.configs.constants import OUTPUT_DIR
from src.modules.utils import load_file_as_string


class Base:
    def __init__(self, task_id: str | None = None):
        tmp_config = Base.load_tmp_config(task_id)

        self.task_id = task_id
        self.title = tmp_config["title"]
        self.key_words = tmp_config["key_words"]
        self.topic = tmp_config["topic"]

    @staticmethod
    def load_tmp_config(task_id: str) -> dict:
        path = Path(OUTPUT_DIR) / task_id / "tmp_config.json"
        
        # 如果文件不存在，尝试从父目录查找或创建默认配置
        if not path.exists():
            from src.configs.logger import get_logger
            logger = get_logger("src.schemas.base")
            logger.warning(
                f"tmp_config.json not found at {path}. "
                f"This file should be created by create_tmp_config() before calling offline_generate(). "
                f"Please ensure you're running the full pipeline from offline_run.py"
            )
            # 尝试从父目录查找（可能 task_id 路径不同）
            parent_dir = path.parent
            if parent_dir.exists():
                # 查找是否有其他 task_id 的 tmp_config.json
                config_files = list(parent_dir.glob("*/tmp_config.json"))
                if config_files:
                    logger.warning(f"Found tmp_config.json in other task directories: {[f.parent.name for f in config_files]}")
            
            raise FileNotFoundError(
                f"tmp_config.json not found at {path}. "
                f"Please ensure create_tmp_config() is called before initializing generators. "
                f"If you're running individual workflow steps, you need to create this file first."
            )
        
        dic = json.loads(load_file_as_string(path))

        missing_keys = [
            key for key in dic if key not in ["task_id", "title", "key_words", "topic"]
        ]
        assert not missing_keys, f"Missing keys: {', '.join(missing_keys)}"

        return dic
