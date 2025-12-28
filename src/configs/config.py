import json
from pathlib import Path
import os
from zai import ZhipuAiClient

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent

# huggingface mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # Uncomment this line if you want to use a specific Hugging Face mirror
os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache/")

# REMOTE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
REMOTE_URL = "https://llmapi.paratera.com/v1/chat/completions"
TOKEN = "sk-Gu92tzt7fkKMYls1gwk2Hg"
# TOKEN = "54ff878e35184e3fb5dd5779b7997131.QePWYwiESCFZJKS9"
# DEFAULT_CHATAGENT_MODEL = "glm-4.5-flash"
DEFAULT_CHATAGENT_MODEL = "GLM-4.5"
ADVANCED_CHATAGENT_MODEL = "GLM-4.5"

LOCAL_URL = "LOCAL_URL"
LOCAL_LLM = " "
DEFAULT_EMBED_LOCAL_MODEL = "glm-4.5-flash"

## for embedding model
DEFAULT_EMBED_ONLINE_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_REMOTE_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_TOKEN = "sk-vzjuwtcnqpszzdpftljbutreqmbqhckixmoshijzsaiocxjd"
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
DEFAULT_LLAMAINDEX_OPENAI_MODEL = "gpt-4o"
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
