import json
import os
import re
from pathlib import Path
from typing import Union

from tqdm import tqdm

from src.configs.config import BASE_DIR, CHAT_AGENT_WORKERS, MD_TEXT_LENGTH
from src.configs.constants import OUTPUT_DIR
from src.configs.logger import get_logger
from src.models.LLM import ChatAgent
from src.models.LLM.utils import cut_text_by_token, load_prompt
from src.models.monitor.time_monitor import TimeMonitor
from src.modules.utils import (
    clean_chat_agent_format,
    load_file_as_string,
    sanitize_filename,
    save_result,
)

logger = get_logger("src.modules.preprocessor.DataCleaner")


class DataCleaner:
    def __init__(self, papers: list[dict] = []):
        self.papers: list[dict] = papers
        self.chat_agent_workers = CHAT_AGENT_WORKERS

    def load_json_dir(self, json_path_dir: Path):
        """load papers from json directory."""
        papers = []
        cnt_total = 0
        for file in os.listdir(json_path_dir):
            if file.endswith(".json"):
                p = os.path.join(json_path_dir, file)
                dic = json.loads(load_file_as_string(p))
                if (
                    "md_text" in dic
                ):  # Only consider those with `md_text` as available papers.
                    papers.append(dic)
                cnt_total += 1
        logger.info(f"Find {len(papers)} out of {cnt_total} papers available.")
        self.papers = papers

    def complete_title(self):
        for paper in tqdm(self.papers, desc="completing title..."):
            if "title" not in paper:
                md_text = paper.get("md_text", "").strip()
                if md_text:
                    lines = md_text.splitlines()
                    if lines:
                        paper["title"] = lines[0].strip(" #")
                    else:
                        paper["title"] = "Untitled"
                        logger.warning(f"Empty md_text found, using default title 'Untitled'")
                else:
                    paper["title"] = "Untitled"
                    logger.warning(f"Empty or missing md_text found, using default title 'Untitled'")
                
                # 避免标题过长
                if "title" in paper:
                    paper["title"] = paper["title"][:32]

    def complete_abstract(self):
        pattern = r"\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*"  # find "abstract" substring, with whitespace bettween letters.
        for paper in tqdm(self.papers, desc="completing abstract..."):
            if "abstract" in paper and len(paper["abstract"]) > 500:
                continue
            match = re.search(pattern, paper["md_text"], re.IGNORECASE)
            if match:
                index = match.start()
                paper["abstract"] = paper["md_text"][index : index + 2000]
            else:
                paper["abstract"] = paper["md_text"][:2000]

    def complete_bib(self, bib_file_save_path: str, chat_agent: ChatAgent = None, use_llm: bool = True):
        """
        Not only complete the bib_name, also need to save all bibnames into a references.bib file.
        
        Args:
            bib_file_save_path: Path to save the references.bib file
            chat_agent: Optional ChatAgent for LLM-based BibTeX completion
            use_llm: Whether to use LLM to complete BibTeX entries (default: True)
        """
        var_name_i = 0
        bib_all = []
        remove_non_ascii_chars = (
            lambda input_string: input_string.replace(",", "")
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        
        # 如果使用 LLM 补全但没有提供 chat_agent，创建一个
        if use_llm and chat_agent is None:
            chat_agent = ChatAgent()

        for paper in tqdm(self.papers, desc="completing bibname..."):
            if "reference" in paper:
                # 已有 reference，提取并清理 bib_name
                bib_name = paper["reference"].splitlines()[0].split("{")[1].strip(",")
                new_bib_name = remove_non_ascii_chars(bib_name)

                paper["bib_name"] = new_bib_name
                paper["reference"] = paper["reference"].replace(bib_name, new_bib_name)
                
                # 如果使用 LLM 且现有条目不完整，尝试补全
                if use_llm and chat_agent:
                    # 检查是否只有 title 字段（简单条目）
                    if "author" not in paper["reference"].lower() and "year" not in paper["reference"].lower():
                        try:
                            enhanced_bib = self._enhance_bibtex_with_llm(
                                paper, chat_agent, new_bib_name, remove_non_ascii_chars
                            )
                            if enhanced_bib:
                                paper["reference"] = enhanced_bib
                                logger.debug(f"Enhanced BibTeX for: {paper.get('title', 'Unknown')[:50]}")
                        except Exception as e:
                            logger.warning(f"Failed to enhance BibTeX for {paper.get('title', 'Unknown')[:50]}: {e}")
            else:
                # 没有 reference，生成新的
                title = remove_non_ascii_chars(paper["title"])
                bib_name = "".join([c for c in title if not c.isspace()][:10]) + str(
                    var_name_i
                )
                var_name_i += 1
                
                # 如果使用 LLM，尝试生成完整的 BibTeX 条目
                if use_llm and chat_agent:
                    try:
                        bib_tex = self._enhance_bibtex_with_llm(
                            paper, chat_agent, bib_name, remove_non_ascii_chars
                        )
                        if not bib_tex:
                            # LLM 失败，使用简单格式
                            bib_tex = f"@article{{{bib_name},\ntitle={{{title}}}\n}}"
                    except Exception as e:
                        logger.warning(f"Failed to generate BibTeX with LLM for {title[:50]}: {e}")
                        # 失败时使用简单格式
                        bib_tex = f"@article{{{bib_name},\ntitle={{{title}}}\n}}"
                else:
                    # 不使用 LLM，使用简单格式
                    bib_tex = f"@article{{{bib_name},\ntitle={{{title}}}\n}}"

                paper["reference"] = bib_tex
                paper["bib_name"] = bib_name

            bib_all.append(paper["reference"])

        save_result("\n".join(bib_all), bib_file_save_path)
    
    def _enhance_bibtex_with_llm(
        self, paper: dict, chat_agent: ChatAgent, bib_name: str, remove_non_ascii_chars
    ) -> str:
        """
        使用 LLM 从论文文本中提取信息并生成完整的 BibTeX 条目
        
        Args:
            paper: 论文字典，包含 title, abstract, md_text 等字段
            chat_agent: ChatAgent 实例
            bib_name: BibTeX 条目名称
            remove_non_ascii_chars: 清理函数
        
        Returns:
            完整的 BibTeX 条目字符串，如果失败返回 None
        """
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        md_text = paper.get("md_text", "")
        
        # 限制 md_text 长度，避免 prompt 过长
        md_text_preview = md_text[:2000] if md_text else ""
        
        # 清理标题中的特殊字符
        clean_title = remove_non_ascii_chars(title)
        
        prompt = load_prompt(
            f"{BASE_DIR}/resources/LLM/prompts/preprocessor/complete_bibtex.md",
            title=title,
            abstract=abstract[:1000] if abstract else "",  # 限制摘要长度
            md_text_preview=md_text_preview,
            bib_name=bib_name,
        )
        
        try:
            response = chat_agent.remote_chat(prompt)
            response = clean_chat_agent_format(content=response)
            
            # 验证返回的是有效的 BibTeX 格式
            if "@" in response and "{" in response and bib_name in response:
                # 确保 bib_name 正确
                if f"{{{bib_name}," not in response:
                    # 尝试修复 bib_name
                    response = re.sub(r"@\w+\{([^,]+),", f"@article{{{bib_name},", response, count=1)
                return response.strip()
            else:
                logger.warning(f"LLM returned invalid BibTeX format for {title[:50]}")
                return None
        except Exception as e:
            logger.warning(f"Error calling LLM for BibTeX completion: {e}")
            return None

    def check_md_text_length(self):
        for paper in self.papers:
            if "md_text" not in paper:
                continue
            md_text = paper["md_text"]
            paper["md_text"] = cut_text_by_token(md_text, MD_TEXT_LENGTH)

    def __process_paper_type_response(self, res: str, paper_index: int):
        kinds = ["method", "benchmark", "theory", "survey"]
        for k in kinds:
            if k in res.lower():
                self.papers[paper_index]["paper_type"] = k
                return True
        logger.error(
            f"failed to extract papertype of {self.papers[paper_index]['title']}"
        )
        logger.error(f"The response from gpt is {res}")
        return False

    def get_paper_type(self, chat_agent: ChatAgent):
        """complete the paper type field with chatgpt."""
        # load prompts
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            abstract = paper["abstract"]
            prompt = load_prompt(
                f"{BASE_DIR}/resources/LLM/prompts/preprocessor/paper_type_classification.md",
                abstract=abstract,
            )
            prompts_and_index.append([prompt, i])
        # batch_chat
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = chat_agent.batch_remote_chat(prompts, desc="getting paper type...")
            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_paper_type_response(res, paper_index)
            ]
            cnt += 1
        
        # 为所有没有 paper_type 的论文设置默认值（避免后续 KeyError）
        for i, paper in enumerate(self.papers):
            if "paper_type" not in paper:
                logger.warning(
                    f"论文 '{paper.get('title', 'Unknown')[:50]}' 无法分类类型，使用默认值 'method'"
                )
                paper["paper_type"] = "method"  # 默认使用 method 类型

    def __process_attri_response(self, res: str, paper_index: int):
        # 检查是否是内容安全错误
        if "敏感内容" in res or "BadRequestError" in res or "Content Safety" in res:
            paper_title = self.papers[paper_index].get('title', 'Unknown')
            
            # 尝试提取敏感原因
            safety_reason = "未知原因"
            if "Content Safety Check Failed:" in res:
                try:
                    reason_part = res.split("Content Safety Check Failed:")[1].strip()
                    safety_reason = reason_part.split("\n")[0] if "\n" in reason_part else reason_part
                except:
                    pass
            
            logger.warning(
                f"论文 '{paper_title}' 触发内容安全检测，跳过该论文的属性提取\n"
                f"  敏感原因: {safety_reason}"
            )
            # 为该论文设置一个空的属性，避免后续处理失败
            self.papers[paper_index]["attri"] = None
            return True  # 返回 True 表示已处理（虽然失败了，但不需要重试）
        
        res = clean_chat_agent_format(content=res)
        try:
            res_dic = json.loads(res)
            self.papers[paper_index]["attri"] = {**res_dic}
            return True
        except Exception as e:
            logger.debug(
                f"Failed to process {self.papers[paper_index]['title']}; The res: {res[:100]}; {e}"
            )
            return False

    def get_attri(self, chat_agent: ChatAgent):
        """extract attribute tree from paper"""
        # 获取所有含 "md_text" 的文件并生成 prompts
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            # 检查 paper_type 是否存在，如果不存在则使用默认值
            paper_type = paper.get("paper_type", "method").lower()
            
            # 验证 paper_type 是否在支持的列表中
            supported_types = ["method", "benchmark", "theory", "survey"]
            if paper_type not in supported_types:
                logger.warning(
                    f"论文 '{paper.get('title', 'Unknown')[:50]}' 的类型 '{paper_type}' 不在支持列表中，使用默认值 'method'"
                )
                paper_type = "method"
            
            prompt = load_prompt(
                f"{BASE_DIR}/resources/LLM/prompts/preprocessor/attri_tree_for_{paper_type}.md",
                paper=paper["md_text"],
            )
            prompts_and_index.append([prompt, i])

        # 批量处理 prompts
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = chat_agent.batch_remote_chat(
                prompts, desc="getting attribute tree from paper......"
            )

            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_attri_response(res, paper_index)
            ]
            cnt += 1

    def save_papers(
        self, save_dir: Union[str, Path], file_name_attr: str = "title"
    ) -> None:
        """save every cleaned paper."""
        # 确保保存目录存在
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filter_field = [
            "from",
            "scholar_id",
            "detail_id",
            "title",
            "abstract",
            "bib_name",
            "md_text",
            "paper_type",
            "attri",
            "mount_outline",
            "similarity_score",
            "image",
        ]
        for paper in self.papers:
            try:
                file_name = paper[file_name_attr] + ".json"
                file_name = sanitize_filename(file_name)
                file_path = os.path.join(save_dir, file_name)
                save_dic = {key: paper.get(key, None) for key in filter_field}
                save_result(json.dumps(save_dic, indent=4), file_path)
            except Exception as e:
                logger.error(
                    f"There is an error when saving {file_path}. The error is: {e}"
                )
        return self.papers

    def quick_check(self) -> list[dict]:
        """Used in PaperRecaller for quick check"""
        papers_with_md = [paper for paper in self.papers if "md_text" in paper]
        self.papers = papers_with_md
        self.complete_title()
        self.complete_abstract()
        return self.papers

    def offline_proc(self, task_id: str, ref_path: str) -> None:
        ref_data_path = Path(ref_path)
        if not ref_data_path.exists():
            raise FileNotFoundError(
                f"Reference directory does not exist: {ref_data_path}. "
                f"Please make sure the path is correct and contains .md files."
            )
        if not ref_data_path.is_dir():
            raise ValueError(f"Reference path is not a directory: {ref_data_path}")
        
        md_texts = []
        for p in ref_data_path.glob("*.md"):
            if p.is_file():
                text = p.read_text(encoding='utf-8')
                if text.strip():  # 只添加非空文件
                    md_texts.append(text)
                else:
                    logger.warning(f"Skipping empty file: {p.name}")
        
        if len(md_texts) == 0:
            logger.warning(
                f"No .md files found in {ref_data_path}. "
                f"Please make sure you have placed your reference documents (.md files) in this directory."
            )
        self.papers = [{"md_text": md_text} for md_text in md_texts]

        self.complete_title()
        self.complete_abstract()
        bib_file_path = Path(OUTPUT_DIR) / task_id / "latex" / "references.bib"
        # 创建 chat_agent 用于 BibTeX 补全和其他任务
        chat_agent = ChatAgent()
        self.complete_bib(bib_file_path, chat_agent=chat_agent, use_llm=True)

        self.check_md_text_length()
        self.get_paper_type(chat_agent=chat_agent)
        self.get_attri(chat_agent=chat_agent)

        save_path = Path(f"{OUTPUT_DIR}/{task_id}/papers")
        self.save_papers(save_dir=save_path)
        logger.info(f"========== {len(self.papers)} remain after cleaning. ==========")

    def run(self, task_id: str, chat_agent: ChatAgent = None):
        time_monitor = TimeMonitor(task_id)
        time_monitor.start("clean paper")

        self.load_json_dir(Path(OUTPUT_DIR) / task_id / "jsons")
        self.complete_title()
        self.complete_abstract()
        bib_file_path = Path(OUTPUT_DIR) / task_id / "latex" / "references.bib"
        
        # 如果 chat_agent 为 None，创建一个用于 BibTeX 补全
        if chat_agent is None:
            chat_agent = ChatAgent()
        
        self.complete_bib(bib_file_path, chat_agent=chat_agent, use_llm=True)

        self.check_md_text_length()
        self.get_paper_type(chat_agent=chat_agent)
        self.get_attri(chat_agent=chat_agent)

        save_path = Path(f"{OUTPUT_DIR}/{task_id}/papers")
        self.save_papers(save_dir=save_path)
        logger.info(f"========== {len(self.papers)} remain after cleaning. ==========")

        time_monitor.end("clean paper")


# python -m src.modules.preprocessor.data_cleaner
if __name__ == "__main__":
    dc = DataCleaner()
    dc.offline_proc("ref1")
    print(len(dc.papers))
