import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from src.configs.constants import BASE_DIR, OUTPUT_DIR
from src.configs.logger import get_logger
from src.models.LLM import ChatAgent
from src.configs.config import ADVANCED_CHATAGENT_MODEL
from src.models.LLM.utils import load_prompt
from src.modules.utils import save_result, update_config
from src.configs.config import DEFAULT_DATA_FETCHER_ENABLE_CACHE

logger = get_logger("src.modules.preprocessor.utils")


class ArgsNamespace(argparse.Namespace):
    title: str
    key_words: str
    page: int
    time_s: str
    time_e: str
    enable_cache: bool
    device: str
    gpu_ids: str
    quality_mode: str  # "default" | "high"
    cot: bool  # Chain of Thought 开关


def parse_arguments_for_preprocessor() -> ArgsNamespace:
    parser = argparse.ArgumentParser(description="Fetch data and Clean them.")
    parser.add_argument(
        "--title",
        type=str,
        default="Attention Heads of Large Language Models: A Survey",
        help="Input the title to generate survey.",
    )
    parser.add_argument(
        "--key_words",
        type=str,
        default="",
        help="Input the key_words to search on databases.",
    )
    parser.add_argument(
        "--page",
        type=str,
        default="5",
        help="Number of pages to crawl on Google Scholar.",
    )
    parser.add_argument(
        "--time_s",
        type=str,
        default="2017",
        help="Start year for filtering search results.",
    )
    parser.add_argument(
        "--time_e",
        type=str,
        default="2024",
        help="End year for filtering search results.",
    )
    parser.add_argument(
        "--enable_cache",
        type=bool,
        default=DEFAULT_DATA_FETCHER_ENABLE_CACHE,
        help="Whether import cache for preprocessing.",
    )
    return parser.parse_args()


def parse_arguments_for_integration_test() -> str:
    parser = argparse.ArgumentParser(description="Give the --task_id parameter.")
    parser.add_argument("--task_id", type=str, required=True, help="The ID of the task")
    args = parser.parse_args()
    return args.task_id


def parse_arguments_for_offline() -> ArgsNamespace:
    parser = argparse.ArgumentParser(description="Fetch data and Clean them.")
    parser.add_argument(
        "--title",
        type=str,
        default="Attention Heads of Large Language Models: A Survey",
        help="Input the title to generate survey.",
    )
    parser.add_argument(
        "--key_words",
        type=str,
        default="",
        help="Input the key_words to search on databases.",
    )
    parser.add_argument(
        "--ref_path",
        type=str,
        default="",
        help="Path to the reference papers for offline run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for embedding model. 'auto' will auto-detect (use GPU if available), 'cpu' forces CPU, 'cuda' forces GPU.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="GPU device IDs to use (e.g., '0' for single GPU, '0,1,2' for multiple GPUs). Only effective when device is 'cuda' or 'auto'.",
    )
    parser.add_argument(
        "--quality_mode",
        type=str,
        default="default",
        choices=["default", "high"],
        help="Quality mode for content generation. 'default' uses standard single-pass generation. "
             "'high' enables Multi-Agent Writer-Critic verification loop for higher accuracy (costs ~2-3x more API calls).",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        default=False,
        help="Enable Chain of Thought (CoT) reasoning. "
             "In 'default' mode: adds self-reflection thinking before writing. "
             "In 'high' mode: adds Planner agent for logical planning before Writer generates content. "
             "Costs ~15%% more in default mode, ~65%% more in high mode.",
    )

    return parser.parse_args()


def create_tmp_config(title: str, key_word: str):
    tmp_config = {}
    tmp_config["title"] = title
    
    # 尝试生成关键词，如果失败则使用原始关键词
    try:
        tmp_config["key_words"] = gen_keyword(title, key_word)
    except Exception as e:
        logger.warning(f"Failed to generate keywords: {e}, using original key_words")
        tmp_config["key_words"] = key_word if key_word else title[:50]
    
    # 尝试生成主题，如果失败则使用标题作为主题
    try:
        tmp_config["topic"] = gen_topic(title, tmp_config["key_words"])
    except Exception as e:
        logger.warning(f"Failed to generate topic: {e}, using title as topic")
        tmp_config["topic"] = title
    
    # 生成 task_id（基于关键词的前5个字符）
    task_id = datetime.now().strftime("%Y-%m-%d-%H%M_") + tmp_config["key_words"][
        :5
    ].replace(" ", "_")
    tmp_config["task_id"] = task_id

    # 确保目录存在，然后保存配置
    config_path = Path(OUTPUT_DIR) / task_id / "tmp_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    update_config(tmp_config, config_path)
    logger.info(f"Created tmp_config: {json.dumps(tmp_config, indent=4)}")
    return tmp_config


def gen_keyword(title: str, key_words: str) -> str:
    if len(key_words.split(",")) >= 6:
        return key_words

    chat_agent = ChatAgent()
    prompt = load_prompt(
        Path(f"{BASE_DIR}/resources/LLM/prompts/preprocessor/generate_keyword.md"),
        title=title,
        key_words=key_words,
    )

    res = chat_agent.remote_chat(prompt, model=ADVANCED_CHATAGENT_MODEL)
    new_keywords = re.findall(r"<Answer>(.*?)</Answer>", res)[0]
    if key_words:
        final_keywords = ",".join(
            key_words.split(",") + new_keywords.split(",")[:3]
        )  # only select first 3 generated keyword, to avoid misunderstanding
    else:
        final_keywords = new_keywords
    logger.info(f"Keywords: {final_keywords}")
    return final_keywords


def gen_topic(title: str, key_word: str) -> str:
    """Generate a detail description of the keyword in one sentence.
    This description is used to provide more infos about keyword.
    """
    chat = ChatAgent()
    prompt = load_prompt(
        Path(f"{BASE_DIR}/resources/LLM/prompts/preprocessor/generate_topic.md"),
        title=title,
        key_word=key_word,
    )
    topic = chat.remote_chat(prompt, model=ADVANCED_CHATAGENT_MODEL)
    return topic


def wait_for_crawling(seconds: int):
    """Sleep system for seconds."""
    for i in range(seconds):
        print(
            f"\rWaiting for crawling... remaining {seconds - i} seconds.   ",
            end="",
            flush=True,
        )
        time.sleep(1)
    print()


def save_papers(papers: list[dict], dir_path: Path):
    for paper in papers:
        p = Path(dir_path) / f"{paper.get('_id', paper['title'])}.json"
        save_result(json.dumps(paper, indent=4), p)


def get_tmp_config(task_id: str):
    tmp_config_path = Path(OUTPUT_DIR) / task_id / "tmp_config.json"
    with open(tmp_config_path, "r", encoding="utf-8") as file:
        tmp_config = json.load(file)
    return tmp_config
