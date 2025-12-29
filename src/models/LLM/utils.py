import base64
import os
from pathlib import Path

import tiktoken

from src.configs.config import CUT_WORD_LENGTH
from src.configs.logger import get_logger

logger = get_logger("src.modules.LLM.utils")


def load_prompt(file_path: Path, **kwargs):
    """读取prompt模板"""
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as f:
            return f.read().format(**kwargs)
    else:
        logger.error(f"Prompt template not found at {file_path}")
        return ""


def num_token_from_string(text: str, model: str = "gpt-4o-mini") -> int:
    """Return token nums of a string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        # 允许特殊 token 作为普通文本处理
        encoded_text = encoding.encode(text, disallowed_special=())
        return len(encoded_text)
    except Exception as e:
        logger.error(f"Error encoding text for token count: {e}")
        # 如果编码失败，使用字符数估算（fallback）
        return len(text) // CUT_WORD_LENGTH


def cut_text_by_token(text: str, max_tokens: int, model: str = "gpt-4o-mini"):
    """Cut text by token num."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        # 允许特殊 token 作为普通文本处理，避免编码错误
        # 这通常发生在参考文档中包含 GPT 技术报告等包含特殊 token 的文本时
        encoded_text = encoding.encode(text, disallowed_special=())
        cut_text = encoding.decode(encoded_text[:max_tokens])
    except Exception as e:
        logger.error(e)
        # 如果编码失败，使用字符数估算（fallback）
        cut_text = text[: CUT_WORD_LENGTH * max_tokens]
    return cut_text


# 图片转base64函数
def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
