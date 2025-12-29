import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 确保使用 HuggingFace 镜像（必须在导入其他库之前设置）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache/")

import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm

from src.configs.config import (
    DEFAULT_EMBED_LOCAL_MODEL,
    DEFAULT_EMBED_ONLINE_MODEL,
    EMBED_REMOTE_URL,
    EMBED_TOKEN,
    get_device_config,
)
from src.configs.logger import get_logger

logger = get_logger("src.models.LLM.EmbedAgent")


class EmbedAgent:
    """
    A class to handle remote text embedding using a specified API.
    Supports multi-threading for batch processing.
    """

    def __init__(self, token=EMBED_TOKEN, remote_url=EMBED_REMOTE_URL) -> None:
        """
        Initialize the EmbedAgent.

        Args:
            token (str): Authentication token for the remote API.
            remote_url (str): URL of the remote embedding API.
        """
        self.remote_url = remote_url
        self.token = token
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        # 使用全局设备配置
        device = get_device_config()
        
        try:
            # HuggingFaceEmbedding 直接使用 device 参数，而不是通过 model_kwargs
            self.local_embedding_model = HuggingFaceEmbedding(
                model_name=DEFAULT_EMBED_ONLINE_MODEL,
                device=device
            )
            logger.info(f"Loaded embedding model on {device}")
        except Exception as e:
            logger.error(
                f"Failed to load embedding model {DEFAULT_EMBED_ONLINE_MODEL}: {e}"
            )
            # 如果主模型加载失败，尝试使用 CPU 作为备用
            try:
                logger.info("Trying to load embedding model on CPU as fallback...")
                self.local_embedding_model = HuggingFaceEmbedding(
                    model_name=DEFAULT_EMBED_ONLINE_MODEL,
                    device="cpu"
                )
                logger.info("Loaded embedding model on CPU (fallback)")
            except Exception as e2:
                logger.error(f"Failed to load embedding model on CPU: {e2}")
                raise RuntimeError(
                    f"Unable to load embedding model {DEFAULT_EMBED_ONLINE_MODEL}. "
                    f"Original error: {e}, Fallback error: {e2}"
                )

    def remote_embed(
        self,
        text: str,
        max_try: int = 15,
        debug: bool = False,
        model: str = "BAAI/bge-m3",
    ) -> list:
        """
        Embed text using the remote API.

        Args:
            text (str): Input text to embed.
            max_try (int, optional): Maximum number of retry attempts.
            debug (bool, optional): Whether to return debug information.
            model (str, optional): Model name for the remote API.

        Returns:
            list: Embedding vector or error message.
        """
        url = self.remote_url
        json_data = json.dumps(
            {"model": model, "input": text, "encoding_format": "float"}
        )

        try:
            response = requests.post(url, headers=self.header, data=json_data)
        except Exception as e:
            logger.error(f"Initial request failed: {e}")
            response = None
            for attempt in range(max_try):
                try:
                    response = requests.post(url, headers=self.header, data=json_data)
                    if response.status_code == 200:
                        logger.info(f"Retry {attempt + 1}/{max_try} succeeded.")
                        break
                except Exception as e:
                    logger.error(f"Retry {attempt + 1}/{max_try} failed: {e}")
                    response = None

        if response is None:
            error_msg = "embed response code: 000"
            if debug:
                return error_msg, response
            return []

        if response.status_code != 200:
            error_msg = f"embed response code: {response.status_code}\n{response.text}"
            logger.error(error_msg)
            if debug:
                return error_msg, response
            return []

        try:
            res = response.json()
            embedding = res["data"][0]["embedding"]
            if debug:
                return embedding, response
            return embedding
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}")
            if debug:
                return "JSON decoding failed", response
            return []

    def __remote_embed_task(self, index: int, text: str):
        """
        Internal method to handle embedding tasks in threads.

        Args:
            index (int): Index of the text in the input list.
            text (str): Text to embed.

        Returns:
            tuple: (index, embedding)
        """
        embedding = self.remote_embed(text)
        return index, embedding

    def batch_remote_embed(
        self, texts: list[str], worker: int = 10, desc: str = "Batch Embedding..."
    ) -> list:
        """
        Batch process text embeddings using multi-threading.

        Args:
            texts (list[str]): List of texts to embed.
            worker (int, optional): Number of worker threads.
            desc (str, optional): Description for the progress bar.

        Returns:
            list: List of embedding vectors.
        """
        embeddings = ["no response"] * len(texts)
        with ThreadPoolExecutor(max_workers=worker) as executor:
            future_l = [
                executor.submit(self.__remote_embed_task, i, texts[i])
                for i in range(len(texts))
            ]
            for future in tqdm(
                as_completed(future_l),
                desc=desc,
                total=len(future_l),
                dynamic_ncols=True,
            ):
                i, embedding = future.result()
                embeddings[i] = embedding
        return embeddings

    def local_embed(self, text: str) -> list[float]:
        embedding = self.local_embedding_model.get_text_embedding(text)
        return embedding

    def batch_local_embed(self, text_l: list[str]) -> list[list[float]]:
        embed_documents = self.local_embedding_model.get_text_embedding_batch(
            text_l, show_progress=True
        )
        return embed_documents


if __name__ == "__main__":
    text_list = ["text1", "text2", "text4"]
    embed_agent = EmbedAgent()
    embedding = embed_agent.batch_remote_embed(text_list)
    print(embedding)
    logger.info("Embedding complete.")

    embedding = embed_agent.batch_local_embed(text_list)
    print(embedding)
