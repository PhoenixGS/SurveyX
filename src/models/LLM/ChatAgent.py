"""
@reference:
1.发送本地图片： https://www.cnblogs.com/Vicrooor/p/18227547
"""

import fcntl
import requests
import json
import pickle
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm
from pathlib import Path

from src.configs.config import (
    REMOTE_URL,
    LOCAL_URL,
    TOKEN,
    BASE_DIR,
    DEFAULT_CHATAGENT_MODEL,
    CHAT_AGENT_WORKERS,
    FALLBACK_REMOTE_URL,
    FALLBACK_TOKEN,
    PARATERA_API_KEYS,
)
from src.configs.constants import OUTPUT_DIR

from src.configs.logger import get_logger
from src.models.LLM.utils import encode_image
from src.models.monitor.token_monitor import TokenMonitor

logger = get_logger("src.models.LLM.ChatAgent")
logger.debug(f"ChatAgent pid={os.getpid()}")


class ChatAgent:
    Cost_file = Path(f"{OUTPUT_DIR}/tmp/cost.txt")
    Request_stats_file = Path(f"{OUTPUT_DIR}/tmp/request_stats.txt")
    Record_splitter = "||"
    Record_show_length = 200
    
    # 类级别的 API key 池和锁（线程安全）
    _api_keys_pool = PARATERA_API_KEYS.copy() if PARATERA_API_KEYS else [TOKEN]
    _api_keys_lock = threading.Lock()
    _key_usage_counter = {key: 0 for key in _api_keys_pool}  # 记录每个 key 的使用次数

    def __init__(
        self,
        token_monitor: TokenMonitor | None = None,
        token: str = None,  # 如果为 None，则从池中随机选择
        remote_url: str = REMOTE_URL,
        local_url: str = LOCAL_URL,
    ) -> None:
        self.remote_url = remote_url
        self.local_url = local_url
        
        # API key 选择逻辑
        if token is not None:
            # 如果明确指定了 token，使用指定的
            self.token = token
        elif len(self._api_keys_pool) > 1:
            # 如果有多个 key，随机选择一个
            with self._api_keys_lock:
                self.token = random.choice(self._api_keys_pool)
                self._key_usage_counter[self.token] = self._key_usage_counter.get(self.token, 0) + 1
            logger.debug(f"随机选择 API key: {self.token[:20]}... (池中共 {len(self._api_keys_pool)} 个 key)")
        else:
            # 只有一个 key 或使用默认值
            self.token = self._api_keys_pool[0] if self._api_keys_pool else TOKEN
        
        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        self.batch_workers = CHAT_AGENT_WORKERS
        self.token_monitor = token_monitor
        
        # 备用 API 配置
        self.fallback_url = FALLBACK_REMOTE_URL
        self.fallback_token = FALLBACK_TOKEN
        self.use_fallback = False  # 是否已切换到备用 API
        self.content_safety_errors = []  # 记录内容安全错误
    
    @classmethod
    def get_random_token(cls) -> str:
        """从 API key 池中随机选择一个 key（线程安全）"""
        if len(cls._api_keys_pool) == 0:
            return TOKEN
        with cls._api_keys_lock:
            token = random.choice(cls._api_keys_pool)
            cls._key_usage_counter[token] = cls._key_usage_counter.get(token, 0) + 1
            return token
    
    @classmethod
    def get_key_usage_stats(cls) -> dict:
        """获取 API key 使用统计"""
        with cls._api_keys_lock:
            return cls._key_usage_counter.copy()

    def _extract_safety_reason(self, error_text: str) -> str:
        """从错误响应中提取敏感原因"""
        try:
            import json
            error_json = json.loads(error_text)
            error_msg = error_json.get("error", {}).get("message", "")
            
            # 尝试提取具体原因
            if "敏感内容" in error_msg:
                # 查找可能的敏感原因关键词
                reasons = []
                if "输入" in error_msg or "input" in error_msg.lower():
                    reasons.append("输入内容")
                if "生成" in error_msg or "output" in error_msg.lower() or "generate" in error_msg.lower():
                    reasons.append("可能生成的内容")
                if "提示语" in error_msg or "prompt" in error_msg.lower():
                    reasons.append("提示语")
                
                if reasons:
                    return f"检测到敏感内容（可能原因：{', '.join(reasons)}）"
                return "检测到敏感内容（具体原因未明确）"
        except:
            pass
        return "内容安全检测失败（无法解析具体原因）"
    
    def _switch_to_fallback_api(self):
        """切换到备用 API"""
        if self.use_fallback:
            return False  # 已经切换过了
        
        if self.fallback_url and self.fallback_token:
            logger.warning(
                f"检测到批量内容安全错误，切换到备用 API: {self.fallback_url[:50]}..."
            )
            self.remote_url = self.fallback_url
            self.token = self.fallback_token
            self.header = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.fallback_token}",
            }
            self.use_fallback = True
            return True
        else:
            logger.warning(
                "检测到批量内容安全错误，但未配置备用 API。"
                "请设置环境变量 FALLBACK_LLM_API_URL 和 FALLBACK_API_KEY"
            )
            return False

    @retry(
        stop=stop_after_attempt(30),
        wait=wait_exponential(min=1, max=300),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def remote_chat(
        self,
        text_content: str,
        image_urls: list[str] = None,
        local_images: list[Path] = None,
        temperature: float = 0.5,
        debug: bool = False,
        model=DEFAULT_CHATAGENT_MODEL,
        skip_retry_on_safety_error: bool = True,  # 内容安全错误不重试
    ) -> str:
        """chat with remote LLM, return result."""
        url = self.remote_url
        header = self.header
        # text content
        messages = [{"role": "user", "content": text_content}]
        # insert image urls ----
        if (
            image_urls is not None
            and isinstance(image_urls, list)
            and len(image_urls) > 0
        ):
            image_url_frame = []
            for url_ in image_urls:
                image_url_frame.append(
                    {"type": "image_url", "image_url": {"url": url_}}
                )
            image_message_frame = {"role": "user", "content": image_url_frame}
            messages.append(image_message_frame)

        # insert local images ----
        if (
            local_images is not None
            and isinstance(local_images, list)
            and len(local_images) > 0
        ):
            local_image_frame = []
            for local_image in local_images:
                local_encoded_image = encode_image(local_image)
                local_image_frame.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{local_encoded_image}"
                        },
                    }
                )
            image_message_frame = {"role": "user", "content": local_image_frame}
            messages.append(image_message_frame)

        payload = {"model": model, "messages": messages, "temperature": temperature}

        response = requests.post(url, headers=header, json=payload)

        if response.status_code != 200:
            error_text = response.text[:500]
            logger.error(
                f"chat response code: {response.status_code}\n{error_text}, retrying..."
            )
            status_code = 0 if response.status_code != 200 else 1

            # 加了线程锁
            self.update_record(
                status_code=status_code,
                response_code=response.status_code,
                request=text_content,
                response=response.text,
            )
            
            # 检查是否是内容安全检测错误（400 BadRequest）
            if response.status_code == 400 and "敏感内容" in error_text:
                safety_reason = self._extract_safety_reason(response.text)
                error_detail = f"Content Safety Check Failed (400): {safety_reason}"
                
                # 记录内容安全错误
                self.content_safety_errors.append({
                    "reason": safety_reason,
                    "error_text": error_text,
                    "content_preview": text_content[:200] if text_content else ""
                })
                
                logger.warning(
                    f"API 内容安全检测触发：{safety_reason}\n"
                    f"错误详情：{error_text[:300]}\n"
                    f"内容预览：{text_content[:200] if text_content else 'N/A'}..."
                )
                
                # 对于内容安全错误，不重试（因为重试也会失败）
                # 如果 skip_retry_on_safety_error=True，直接抛出异常，不触发 retry 装饰器
                if skip_retry_on_safety_error:
                    # 抛出特殊异常，让调用者决定如何处理
                    # 这个异常不会被 retry 装饰器捕获（因为不是 requests.RequestException）
                    class ContentSafetyError(Exception):
                        """内容安全检测错误，不进行重试"""
                        pass
                    raise ContentSafetyError(error_detail)
                else:
                    # 如果允许重试，抛出 HTTPError（会被 retry 装饰器捕获）
                    raise requests.HTTPError(error_detail, response=response)
            
            response.raise_for_status()
        try:
            res = json.loads(response.text)
            res_text = res["choices"][0]["message"]["content"]
            # 更新总开销
            # token monitor
            if self.token_monitor:
                self.token_monitor.add_token(
                    model=model,
                    input_tokens=res["usage"]["prompt_tokens"],
                    output_tokens=res["usage"]["completion_tokens"],
                )
        except Exception as e:
            res_text = f"Error: {e}"
            logger.error(f"There is an error: {e}")

        status_code = 0 if response.status_code != 200 else 1
        self.update_record(
            status_code=status_code,
            response_code=response.status_code,
            request=text_content,
            response=res_text,
        )

        if debug:
            return res_text, response
        return res_text

    # map chat index
    def __remote_chat(
        self,
        index,
        content,
        temperature: float = 0.5,
        debug: bool = False,
        model=DEFAULT_CHATAGENT_MODEL,
    ):
        # 为每个请求随机选择 API key（如果池中有多个 key）
        # 这样可以实现负载均衡，避免单个 key 限流
        original_header = None
        if len(self._api_keys_pool) > 1:
            current_token = self.get_random_token()
            # 临时更新 header
            original_header = self.header.copy()
            self.header["Authorization"] = f"Bearer {current_token}"
        
        try:
            resp = self.remote_chat(
                text_content=content,
                image_urls=None,
                local_images=None,
                temperature=temperature,
                debug=debug,
                model=model,
                skip_retry_on_safety_error=True,  # 内容安全错误不重试
            )
            return index, resp
        except Exception as e:
            error_str = str(e)
            # 检查是否是内容安全错误
            if "Content Safety" in error_str or "敏感内容" in error_str:
                # 提取敏感原因
                safety_reason = "未知原因"
                for err_info in self.content_safety_errors:
                    if err_info["content_preview"] in content[:200]:
                        safety_reason = err_info["reason"]
                        break
                
                error_msg = f"Content Safety Check Failed: {safety_reason}"
                logger.warning(f"论文索引 {index} 触发内容安全检测: {safety_reason}")
                return index, error_msg
            else:
                # 其他错误，重新抛出
                raise
        finally:
            # 恢复原始 header（如果修改过）
            if original_header is not None:
                self.header = original_header

    def batch_remote_chat(
        self,
        prompt_l: list[str],
        desc: str = "batch_chating...",
        workers: int = CHAT_AGENT_WORKERS,
        temperature: float = 0.5,
        model: str = DEFAULT_CHATAGENT_MODEL,
    ) -> list[str]:
        """
        开启多线程进行对话
        
        Args:
            prompt_l: 提示词列表
            desc: 进度条描述
            workers: 工作线程数
            temperature: 温度参数
            model: 使用的模型，默认为 DEFAULT_CHATAGENT_MODEL
        """
        if workers is None:
            workers = self.batch_workers
        
        # 清空之前的内容安全错误记录
        self.content_safety_errors = []
        
        # 创建线程池
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # 提交任务
            future_l = [
                executor.submit(self.__remote_chat, i, prompt_l[i], temperature, False, model)
                for i in range(len(prompt_l))
            ]
            # 领取任务结果
            res_l = ["no response"] * len(prompt_l)
            safety_error_count = 0
            
            for future in tqdm(
                as_completed(future_l),
                desc=desc,
                total=len(future_l),
                dynamic_ncols=True,
            ):
                try:
                    i, resp = future.result()
                    res_l[i] = resp
                    
                    # 统计内容安全错误
                    if "Content Safety Check Failed" in resp:
                        safety_error_count += 1
                        
                except Exception as e:
                    logger.error(f"批量处理时发生异常: {e}")
                    # 尝试找到对应的索引
                    for idx, f in enumerate(future_l):
                        if f == future:
                            res_l[idx] = f"Error: {str(e)[:200]}"
                            break
            
            # 如果内容安全错误超过阈值（例如 30%），尝试切换 API
            if safety_error_count > 0:
                error_rate = safety_error_count / len(prompt_l)
                logger.warning(
                    f"批量处理完成，内容安全错误: {safety_error_count}/{len(prompt_l)} "
                    f"({error_rate*100:.1f}%)"
                )
                
                # 如果错误率超过 30% 且未使用备用 API，尝试切换
                if error_rate > 0.3 and not self.use_fallback:
                    if self._switch_to_fallback_api():
                        logger.info("已切换到备用 API，建议重新运行失败的请求")
                
                # 显示所有内容安全错误的详细信息
                if self.content_safety_errors:
                    logger.warning("=" * 60)
                    logger.warning("内容安全错误详情：")
                    for i, err_info in enumerate(self.content_safety_errors[:10], 1):  # 最多显示10个
                        logger.warning(
                            f"  错误 {i}: {err_info['reason']}\n"
                            f"    内容预览: {err_info['content_preview']}..."
                        )
                    if len(self.content_safety_errors) > 10:
                        logger.warning(f"  ... 还有 {len(self.content_safety_errors) - 10} 个错误未显示")
                    logger.warning("=" * 60)
        
        return res_l

    @classmethod
    def update_record(
        cls, status_code: int, response_code: int, request: str, response: str
    ):
        "维护记录文件"
        content = (
            f"{status_code}{cls.Record_splitter}{response_code}{cls.Record_splitter}{request[: cls.Record_show_length]}{cls.Record_splitter}{response[: cls.Record_show_length]}".replace(
                "\n", ""
            )
            + "\n"
        )
        # 检查文件是否存在
        if not os.path.exists(cls.Request_stats_file):
            parent_dir = Path(cls.Request_stats_file).parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            with open(cls.Request_stats_file, "w", encoding="utf-8") as fw:
                fcntl.flock(fw, fcntl.LOCK_EX)  # 加锁
                fw.write(content)
                logger.info(
                    f"record file {cls.Request_stats_file} did not exist, created and initialized with 0.0"
                )
                fcntl.flock(fw, fcntl.LOCK_UN)
        # 更新开销总计
        try:
            with open(cls.Request_stats_file, "a", encoding="utf-8") as fw:
                fcntl.flock(fw, fcntl.LOCK_EX)
                fw.write(content)
                fcntl.flock(fw, fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Failed to update cost: {e}")

    def local_chat(self, query, debug=False) -> str:
        """
        调用本地LLM进行推理, 保证端口已开启
        """
        query = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
            {}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""".format(query)

        payload = json.dumps(
            {
                "prompt": query,
                "temperature": 1.0,
                "max_tokens": 102400,
                "n": 1,
                # 可选的参数在这里：https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
            }
        )
        headers = {"Content-Type": "application/json"}
        res = requests.request("POST", self.local_url, headers=headers, data=payload)
        if res.status_code != 200:
            logger.info("chat response code: {}".format(res.status_code), query[:20])
            return "chat response code: {}".format(res.status_code)
        if debug:
            return res
        return res.json()["text"][0].replace(query, "")

    def __local_chat(self, index, query):
        return index, self.local_chat(query, debug=True)

    def batch_local_chat(self, query_l, worker=16, desc="bach local inferencing..."):
        """
        多线程本地推理
        """
        with ThreadPoolExecutor(max_workers=worker) as executor:
            # 提交任务
            future_l = [
                executor.submit(self.__local_chat, i, query_l[i])
                for i in range(len(query_l))
            ]
            # 领取任务结果
            res_l = ["no response"] * len(query_l)
            for future in tqdm(as_completed(future_l), desc=desc, total=len(future_l)):
                i, resp = future.result()
                res_l[i] = resp
        return res_l

    @staticmethod
    def show_request_stats():
        stats_file = ChatAgent.Request_stats_file
        logger.info(f"stats_file: {stats_file}")

        with stats_file.open("r", encoding="utf-8") as fr:
            succ_count = 0
            total_count = 0
            for line in fr:
                elements = line.strip().split(ChatAgent.Record_splitter)
                succ_count += int(elements[0])
                total_count += 1
            logger.info(f"请求成功率：{round(succ_count / total_count * 100, 2)}%")

    @staticmethod
    def clean_request_stats():
        stats_file = ChatAgent.Request_stats_file
        if stats_file.exists():
            logger.info(f"remove {stats_file}.")


if __name__ == "__main__":
    agent = ChatAgent()
    text_content = "图片里面有什么"

    # result = agent.remote_chat(text_content="今天天气怎么样",  model="gpt-4o")
    # print(result)
    #
    # image_urls = ["https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"]
    # result = agent.remote_chat( text_content=text_content, image_urls=image_urls, temperature=0.5, model="gpt-4o")
    # print(result)

    local_images = [f"{BASE_DIR}/resources/dummy_data/figs/dog_and_girl.jpeg"]
    result = agent.remote_chat(
        text_content=text_content,
        local_images=local_images,
        temperature=0.5,
        model="gpt-4o",
    )
    print(result)

    ChatAgent.show_request_stats()
