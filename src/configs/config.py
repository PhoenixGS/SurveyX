import json
from pathlib import Path
import os

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent

# huggingface mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # Uncomment this line if you want to use a specific Hugging Face mirror
os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache/")

# 支持从环境变量读取API key，如果没有设置则使用默认值
# Paratera API endpoint (OpenAI兼容格式)
# 默认使用 Paratera API，也可以通过环境变量覆盖
REMOTE_URL = os.environ.get("LLM_API_URL", "https://llmapi.paratera.com/v1/chat/completions")
# 优先从环境变量读取TOKEN，如果没有则使用默认值
# Paratera API Key
# 优先级：PARATERA_API_KEY > 默认值（sk-eDlafAhDK-FfG08GHEyCfw）
# 注意：如果环境变量中设置了 ZHIPUAI_API_KEY，请取消设置或使用 PARATERA_API_KEY 来覆盖
# 默认使用 Paratera API Key
TOKEN = os.environ.get("PARATERA_API_KEY", "sk-eDlafAhDK-FfG08GHEyCfw")

# 多 API Key 支持（用于负载均衡和避免限流）
# 可以通过环境变量 PARATERA_API_KEYS 设置多个 key，用逗号分隔
# 例如：export PARATERA_API_KEYS="sk-key1,sk-key2,sk-key3"
# 如果设置了多个 key，系统会随机选择使用
_PARATERA_API_KEYS_STR = os.environ.get("PARATERA_API_KEYS", None)
if _PARATERA_API_KEYS_STR:
    # 从环境变量读取多个 key
    PARATERA_API_KEYS = [key.strip() for key in _PARATERA_API_KEYS_STR.split(",") if key.strip()]
else:
    # 默认使用多个 key（用户提供的多个 key）
    PARATERA_API_KEYS = [
        "sk-eDlafAhDK-FfG08GHEyCfw",  # 原始 key
        "sk-E1b_eA5m89SLKIf0MMMVYw",  # 新申请的 key
        "sk-IxW6eWaQEYETG5ZqIv5BDQ",  # 新增 key
        "sk-GF5Rxwct91kwfyKqidEPgg",  # 新增 key
        "sk-rY7L9bKpPLpimsaH8F_7AA",  # 新增 key
        "sk-XTLXwUqYjY_hlwvKibj9FA",  # 新增 key
    ]

# 如果只设置了单个 PARATERA_API_KEY，也添加到列表中
if TOKEN and TOKEN not in PARATERA_API_KEYS:
    PARATERA_API_KEYS.insert(0, TOKEN)

# 备用 API 配置（用于内容安全错误时的切换）
# 可以通过环境变量设置备用 API URL 和 TOKEN
FALLBACK_REMOTE_URL = os.environ.get("FALLBACK_LLM_API_URL", None)  # 例如: "https://api.openai.com/v1/chat/completions"
FALLBACK_TOKEN = os.environ.get("FALLBACK_API_KEY", None)  # 备用 API Key
# 模型名称配置
# 默认模型（快速，用于常规任务）：GLM-4-Flash 或 GLM-4-FlashX
DEFAULT_CHATAGENT_MODEL = os.environ.get("LLM_MODEL", "GLM-4-Flash")
# 高级模型（高质量，用于重要任务如生成大纲、内容等）：GLM-4-Plus 或 GLM-4.5
# ADVANCED_CHATAGENT_MODEL = os.environ.get("ADVANCED_LLM_MODEL", "GLM-4-Plus")
ADVANCED_CHATAGENT_MODEL = os.environ.get("ADVANCED_LLM_MODEL", "DeepSeek-V3.2")

# ==================== Multi-Agent 高质量模式配置 (High Quality Mode) ====================
# 此配置用于 --quality_mode high 模式，实现 Writer-Critic 循环验证
# 与 default 模式分离，不影响原有逻辑

# Writer Agent 配置（用于创造性写作，使用较快的模型）
# DeepSeek-V3.2 具有良好的写作能力和性价比
HIGH_QUALITY_WRITER_MODEL = os.environ.get("WRITER_MODEL", "DeepSeek-V3.2")
HIGH_QUALITY_WRITER_TEMPERATURE = 0.7  # 较高温度获得创造力

# Critic Agent 配置（用于严格核验，使用强推理模型）
# DeepSeek-R1 具有强大的推理和事实核验能力
HIGH_QUALITY_CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "DeepSeek-R1")
HIGH_QUALITY_CRITIC_TEMPERATURE = 0.1  # 低温度确保核验的严谨性和一致性

# 循环控制参数
HIGH_QUALITY_MAX_RETRIES = int(os.environ.get("CRITIC_MAX_RETRIES", "2"))  # 最多修正2次，避免死循环
HIGH_QUALITY_CONTEXT_WINDOW = 200  # 上下文衔接窗口大小（字符数）

# 质量模式选择：default | high
QUALITY_MODE = os.environ.get("QUALITY_MODE", "default")

# ==================== CoT (Chain of Thought) 配置 ====================
# 此配置用于 --cot 模式，在High模式下启用Planner Agent进行逻辑规划

# Planner Agent 配置（仅用于 High + CoT 模式）
# 使用强推理模型进行逻辑规划
HIGH_QUALITY_PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "DeepSeek-R1")
HIGH_QUALITY_PLANNER_TEMPERATURE = 0.3  # 低温度保证计划严谨

# CoT 开关（可通过环境变量设置默认值）
USE_COT = os.environ.get("USE_COT", "false").lower() in ("true", "1", "yes")

LOCAL_URL = "LOCAL_URL"
LOCAL_LLM = "LOCAL_LLM"
DEFAULT_EMBED_LOCAL_MODEL = "DEFAULT_EMBED_LOCAL_MODEL"

## for embedding model
# 默认使用本地HuggingFace模型，如果网络可用会自动下载
DEFAULT_EMBED_ONLINE_MODEL = "BAAI/bge-base-en-v1.5"
# Embedding API（如果需要使用在线embedding服务，可以配置SiliconFlow等）
EMBED_REMOTE_URL = os.environ.get("EMBED_API_URL", "https://api.siliconflow.cn/v1/embeddings")
EMBED_TOKEN = os.environ.get("EMBED_TOKEN", "your embed token here")
SPLITTER_WINDOW_SIZE = 6
SPLITTER_CHUNK_SIZE = 2048

## for preprocessing
CRAWLER_BASE_URL = ""
CRAWLER_GOOGLE_SCHOLAR_SEND_TASK_URL = ""
DEFAULT_DATA_FETCHER_ENABLE_CACHE = True
CUT_WORD_LENGTH = 10
MD_TEXT_LENGTH = 20000
ARXIV_PROJECTION = (
    "_id, title, authors, detail_url, abstract, md_text, reference, detail_id, image"
)

## Iteration and paper pool limits
DEFAULT_ITERATION_LIMIT = 3
DEFAULT_PAPER_POOL_LIMIT = 1024

## llamaindex OpenAI
DEFAULT_LLAMAINDEX_OPENAI_MODEL = os.environ.get("LLAMAINDEX_MODEL", "GLM-4-Plus")
# DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
CHAT_AGENT_WORKERS = 4

## survey generation
COARSE_GRAINED_TOPK = 200
MIN_FILTERED_LIMIT = 150
NUM_PROCESS_LIMIT = 10

## fig retrieving
FIG_RETRIEVE_URL = ""
ENHANCED_FIG_RETRIEVE_URL = ""
FIG_CHUNK_SIZE = 8192
MATCH_TOPK = 3
FIG_RETRIEVE_Authorization = ""
FIG_RETRIEVE_TOKEN = ""

## device configuration
# Global device configuration (can be set via command line arguments)
_DEVICE_CONFIG = {
    "device": "auto",  # "auto", "cpu", or "cuda"
    "gpu_ids": None,   # None, or string like "0" or "0,1,2"
}

def set_device_config(device: str = "auto", gpu_ids: str = None):
    """
    Set global device configuration for embedding models.
    
    Args:
        device: "auto" (auto-detect), "cpu", or "cuda"
        gpu_ids: GPU device IDs string, e.g., "0" or "0,1,2". Only used when device is "cuda" or "auto".
    """
    global _DEVICE_CONFIG
    _DEVICE_CONFIG["device"] = device
    _DEVICE_CONFIG["gpu_ids"] = gpu_ids

def get_device_config():
    """
    Get the device string to use for embedding models.
    Returns: device string like "cpu", "cuda:0", "cuda:1", etc.
    """
    import torch
    device_type = _DEVICE_CONFIG["device"]
    gpu_ids = _DEVICE_CONFIG["gpu_ids"]
    
    # Determine device type
    if device_type == "cpu":
        return "cpu"
    elif device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device cpu or --device auto.")
        if gpu_ids:
            # Use specified GPU(s) - for now, use the first one
            # Multi-GPU support would require model parallelism
            gpu_id = gpu_ids.split(",")[0].strip()
            return f"cuda:{gpu_id}"
        else:
            return "cuda:0"  # Default to first GPU
    else:  # auto
        if torch.cuda.is_available():
            if gpu_ids:
                gpu_id = gpu_ids.split(",")[0].strip()
                return f"cuda:{gpu_id}"
            else:
                return "cuda:0"
        else:
            return "cpu"
