"""
Multi-Agent Content Generator - High Quality Mode

实现 Writer-Critic 循环验证机制：
1. Writer Agent 生成初稿（创造性写作，温度较高）
2. Critic Agent 核验事实/逻辑（严格验证，温度极低）
3. 根据 Critic 反馈迭代修正

核心设计原则：
- 事实 > 逻辑 > 语言（验证优先级）
- 以 Subsection 为操作单位
- 最多重试 2 次，避免死循环
- 传入上下文窗口确保连贯性
"""

import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.configs.config import (
    BASE_DIR,
    CHAT_AGENT_WORKERS,
    HIGH_QUALITY_WRITER_MODEL,
    HIGH_QUALITY_WRITER_TEMPERATURE,
    HIGH_QUALITY_CRITIC_MODEL,
    HIGH_QUALITY_CRITIC_TEMPERATURE,
    HIGH_QUALITY_MAX_RETRIES,
    HIGH_QUALITY_CONTEXT_WINDOW,
    HIGH_QUALITY_PLANNER_MODEL,
    HIGH_QUALITY_PLANNER_TEMPERATURE,
)
from src.configs.constants import OUTPUT_DIR
from src.configs.logger import get_logger
from src.models.LLM import ChatAgent
from src.models.LLM.utils import load_prompt
from src.models.monitor.time_monitor import TimeMonitor
from src.models.monitor.token_monitor import TokenMonitor
from src.models.generator.content_generator import ContentGenerator
from src.modules.utils import clean_chat_agent_format, load_file_as_string, save_result
from src.schemas.outlines import Outlines, SingleOutline

logger = get_logger("src.models.generator.MultiAgentGenerator")


@dataclass
class CriticReviewResult:
    """Critic 评审结果的结构化表示"""
    verdict: str  # "PASS" or "FAIL"
    fact_errors: list = field(default_factory=list)  # 事实错误列表
    missing_points: list = field(default_factory=list)  # 遗漏的重要属性
    logic_gaps: str = ""  # 逻辑不连贯之处
    action_plan: list = field(default_factory=list)  # 改写建议
    raw_response: str = ""  # 原始响应（用于调试）
    
    @property
    def is_pass(self) -> bool:
        return self.verdict.upper() == "PASS"
    
    @staticmethod
    def from_json_response(response: str) -> "CriticReviewResult":
        """从 Critic 的 JSON 响应解析结果"""
        try:
            # 尝试提取 JSON 块
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析
                json_str = response.strip()
                # 移除可能的前后缀
                if json_str.startswith('{') and json_str.endswith('}'):
                    pass
                else:
                    # 尝试找到 JSON 对象
                    start = json_str.find('{')
                    end = json_str.rfind('}') + 1
                    if start != -1 and end > start:
                        json_str = json_str[start:end]
            
            data = json.loads(json_str)
            
            return CriticReviewResult(
                verdict=data.get("verdict", "FAIL"),
                fact_errors=data.get("fact_errors", []),
                missing_points=data.get("missing_points", []),
                logic_gaps=data.get("logic_gaps", ""),
                action_plan=data.get("action_plan", []),
                raw_response=response
            )
        except Exception as e:
            logger.warning(f"Failed to parse Critic response as JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}")
            
            # 尝试从文本中提取 verdict
            if "PASS" in response.upper():
                return CriticReviewResult(verdict="PASS", raw_response=response)
            else:
                return CriticReviewResult(
                    verdict="FAIL",
                    action_plan=["Unable to parse structured feedback, please review the draft manually."],
                    raw_response=response
                )
    
    def to_feedback_string(self) -> str:
        """将评审结果转换为供 Writer 参考的反馈字符串"""
        feedback_parts = []
        
        if self.fact_errors:
            feedback_parts.append("【事实错误】需要修正以下与原始数据不符的内容：")
            for i, error in enumerate(self.fact_errors, 1):
                feedback_parts.append(f"  {i}. {error}")
        
        if self.missing_points:
            feedback_parts.append("\n【遗漏要点】以下重要属性未被覆盖：")
            for i, point in enumerate(self.missing_points, 1):
                feedback_parts.append(f"  {i}. {point}")
        
        if self.logic_gaps:
            feedback_parts.append(f"\n【逻辑问题】{self.logic_gaps}")
        
        if self.action_plan:
            feedback_parts.append("\n【改写指南】")
            for i, action in enumerate(self.action_plan, 1):
                feedback_parts.append(f"  {i}. {action}")
        
        return "\n".join(feedback_parts) if feedback_parts else "No specific feedback provided."


class MultiAgentGenerator(ContentGenerator):
    """
    Multi-Agent 高质量内容生成器
    
    继承 ContentGenerator，重写核心生成逻辑，实现 Writer-Critic 循环：
    1. Writer 生成初稿（使用 DeepSeek-V3.2，温度 0.7）
    2. Critic 核验事实/逻辑（使用 DeepSeek-R1，温度 0.1）
    3. 根据结构化反馈迭代修正
    
    关键特性：
    - 结构化 JSON 评审协议
    - 上下文窗口确保连贯性
    - 最多重试 N 次避免死循环
    """
    
    def __init__(
        self,
        task_id: str,
        max_retries: int = None,
        context_window_size: int = None,
        use_cot: bool = False,
    ):
        super().__init__(task_id)
        
        self.use_cot = use_cot  # CoT (Chain of Thought) 开关
        self.max_retries = max_retries or HIGH_QUALITY_MAX_RETRIES
        self.context_window_size = context_window_size or HIGH_QUALITY_CONTEXT_WINDOW
        
        # 创建 Writer Agent（创造性写作）
        self.writer_token_monitor = TokenMonitor(task_id, "multi_agent_writer")
        self.writer_agent = ChatAgent(token_monitor=self.writer_token_monitor)
        
        # 创建 Critic Agent（严格核验）
        self.critic_token_monitor = TokenMonitor(task_id, "multi_agent_critic")
        self.critic_agent = ChatAgent(token_monitor=self.critic_token_monitor)
        
        # CoT模式：创建 Planner Agent（逻辑规划）
        if self.use_cot:
            self.planner_token_monitor = TokenMonitor(task_id, "multi_agent_planner")
            self.planner_agent = ChatAgent(token_monitor=self.planner_token_monitor)
        
        # Prompt 路径
        self.prompt_dir = Path(BASE_DIR) / "resources" / "LLM" / "prompts" / "multi_agent"
        
        # 统计信息
        self.stats = {
            "total_subsections": 0,
            "passed_first_try": 0,
            "passed_after_revision": 0,
            "max_retries_reached": 0,
            "total_writer_calls": 0,
            "total_critic_calls": 0,
            "total_planner_calls": 0,  # CoT模式新增
        }
        
        cot_info = f"\n  CoT Mode: Enabled (Planner Model: {HIGH_QUALITY_PLANNER_MODEL})" if self.use_cot else ""
        logger.info(
            f"MultiAgentGenerator initialized with:\n"
            f"  Writer Model: {HIGH_QUALITY_WRITER_MODEL} (temp={HIGH_QUALITY_WRITER_TEMPERATURE})\n"
            f"  Critic Model: {HIGH_QUALITY_CRITIC_MODEL} (temp={HIGH_QUALITY_CRITIC_TEMPERATURE})\n"
            f"  Max Retries: {self.max_retries}\n"
            f"  Context Window: {self.context_window_size} chars{cot_info}"
        )
    
    def _get_context_window(self, written_content: str) -> str:
        """获取上下文窗口（上一小节的最后 N 个字符）用于衔接"""
        if not written_content or len(written_content) < 10:
            return ""
        
        # 获取最后 context_window_size 个字符
        context = written_content[-self.context_window_size:]
        
        # 尝试从完整句子开始
        sentence_start = context.find('. ')
        if sentence_start != -1 and sentence_start < len(context) // 2:
            context = context[sentence_start + 2:]
        
        return context.strip()
    
    def _planner_generate_plan(
        self,
        subsection_title: str,
        subsection_desc: str,
        attribute_tree_facts: str,
        outlines: Outlines,
        written_content: str,
    ) -> str:
        """
        Planner Agent 生成写作计划（仅在CoT模式下调用）
        
        Args:
            subsection_title: 小节标题
            subsection_desc: 小节描述
            attribute_tree_facts: Attribute Tree事实数据
            outlines: 完整大纲
            written_content: 已生成内容
        
        Returns:
            结构化写作计划（纯文本，200-500字）
        """
        prompt = load_prompt(
            self.prompt_dir / "planner_generate_plan.md",
            topic=self.topic,
            outlines=str(outlines),
            written_content=written_content[-2000:] if len(written_content) > 2000 else written_content,  # 限制长度
            attribute_facts=attribute_tree_facts,
            section_title=subsection_title,
            section_desc=subsection_desc,
        )
        
        self.stats["total_planner_calls"] += 1
        
        response = self.planner_agent.remote_chat(
            text_content=prompt,
            temperature=HIGH_QUALITY_PLANNER_TEMPERATURE,
            model=HIGH_QUALITY_PLANNER_MODEL,
        )
        
        # 提取计划内容（清洗格式）
        plan = clean_chat_agent_format(content=response)
        
        # 确保计划简洁（不超过800字符）
        max_plan_length = 800
        if len(plan) > max_plan_length:
            logger.debug(f"[Planner] Plan truncated from {len(plan)} to {max_plan_length} chars")
            plan = plan[:max_plan_length] + "..."
        
        logger.debug(f"[Planner] Generated plan ({len(plan)} chars) for: {subsection_title}")
        return plan
    
    def _writer_generate(
        self,
        subsection_title: str,
        subsection_desc: str,
        attribute_tree_facts: str,
        context_window: str,
        outlines: Outlines,
        written_content: str,
        writing_plan: str = "",
    ) -> str:
        """
        Writer Agent 生成初稿
        
        Args:
            subsection_title: 当前小节标题
            subsection_desc: 小节描述
            attribute_tree_facts: 从 Attribute Tree 提取的事实数据
            context_window: 上一小节末尾的上下文（用于衔接）
            outlines: 完整大纲
            written_content: 已生成的内容
            writing_plan: 写作计划（CoT模式下由Planner生成）
        
        Returns:
            生成的 LaTeX 格式内容
        """
        # 根据是否有Plan选择不同的prompt模板
        if writing_plan:
            prompt_template = self.prompt_dir / "writer_generate_with_plan.md"
        else:
            prompt_template = self.prompt_dir / "writer_generate.md"
        
        # 获取所有可用的 bib_name 列表
        available_bib_names = self.get_available_bib_names()
        available_bib_names_str = ", ".join(available_bib_names) if available_bib_names else "无可用引用"
        
        prompt = load_prompt(
            prompt_template,
            topic=self.topic,
            outlines=str(outlines),
            written_content=written_content,
            context_window=context_window,
            attribute_facts=attribute_tree_facts,
            section_title=subsection_title,
            section_desc=subsection_desc,
            writing_plan=writing_plan,  # 传入Plan（如果有）
            available_bib_names=available_bib_names_str,  # 添加可用的 bib_name 列表
        )
        
        self.stats["total_writer_calls"] += 1
        
        response = self.writer_agent.remote_chat(
            text_content=prompt,
            temperature=HIGH_QUALITY_WRITER_TEMPERATURE,
            model=HIGH_QUALITY_WRITER_MODEL,
        )
        
        # 清理响应格式
        response = clean_chat_agent_format(content=response)
        response = response.replace("\\subsection{Conclusion}", "")
        
        # 验证引用
        validation_result = self.validate_citations(response)
        if not validation_result["valid"]:
            logger.warning(
                f"[Writer] Generated draft for '{subsection_title}' contains {len(validation_result['invalid_citations'])} invalid citations. "
                f"Invalid citations: {validation_result['invalid_citations'][:5]}"
            )
        
        return response
    
    def _critic_review(
        self,
        draft: str,
        attribute_tree_facts: str,
        subsection_title: str,
    ) -> CriticReviewResult:
        """
        Critic Agent 核验草稿
        
        要求 Critic 输出结构化 JSON：
        {
            "verdict": "PASS" | "FAIL",
            "fact_errors": [...],
            "missing_points": [...],
            "logic_gaps": "...",
            "action_plan": [...]
        }
        
        Args:
            draft: Writer 生成的草稿
            attribute_tree_facts: 原始 Attribute Tree 数据（真相来源）
            subsection_title: 小节标题
        
        Returns:
            CriticReviewResult 结构化评审结果
        """
        prompt = load_prompt(
            self.prompt_dir / "critic_review.md",
            draft=draft,
            attribute_facts=attribute_tree_facts,
            section_title=subsection_title,
        )
        
        self.stats["total_critic_calls"] += 1
        
        response = self.critic_agent.remote_chat(
            text_content=prompt,
            temperature=HIGH_QUALITY_CRITIC_TEMPERATURE,
            model=HIGH_QUALITY_CRITIC_MODEL,
        )
        
        return CriticReviewResult.from_json_response(response)
    
    def _writer_revise(
        self,
        original_draft: str,
        attribute_tree_facts: str,
        critic_feedback: CriticReviewResult,
        subsection_title: str,
        context_window: str,
    ) -> str:
        """
        Writer Agent 根据 Critic 反馈修正草稿
        
        Prompt 构造：
        - 【原始属性数据】作为真相来源
        - 【上一轮失败草稿】作为基础
        - 【Critic 的 action_plan】作为改写指南
        - 强调保留优点，逐一修复错误
        
        Args:
            original_draft: 上一轮的草稿
            attribute_tree_facts: 原始 Attribute Tree 数据
            critic_feedback: Critic 的结构化反馈
            subsection_title: 小节标题
            context_window: 上下文窗口
        
        Returns:
            修正后的内容
        """
        # 获取所有可用的 bib_name 列表
        try:
            available_bib_names = self.get_available_bib_names()
            available_bib_names_str = ", ".join(available_bib_names) if available_bib_names else "无可用引用"
        except Exception as e:
            logger.warning(f"Failed to get available_bib_names: {e}, using empty list")
            available_bib_names_str = "无可用引用"
        
        prompt = load_prompt(
            self.prompt_dir / "writer_revise.md",
            original_draft=original_draft,
            attribute_facts=attribute_tree_facts,
            critic_feedback=critic_feedback.to_feedback_string(),
            section_title=subsection_title,
            context_window=context_window,
            available_bib_names=available_bib_names_str,  # 添加可用的 bib_name 列表
        )
        
        self.stats["total_writer_calls"] += 1
        
        response = self.writer_agent.remote_chat(
            text_content=prompt,
            temperature=HIGH_QUALITY_WRITER_TEMPERATURE,
            model=HIGH_QUALITY_WRITER_MODEL,
        )
        
        # 清理响应格式
        response = clean_chat_agent_format(content=response)
        response = response.replace("\\subsection{Conclusion}", "")
        
        # 验证引用
        validation_result = self.validate_citations(response)
        if not validation_result["valid"]:
            logger.warning(
                f"[Writer Revise] Revised draft for '{subsection_title}' contains {len(validation_result['invalid_citations'])} invalid citations. "
                f"Invalid citations: {validation_result['invalid_citations'][:5]}"
            )
        
        return response
    
    def generate_subsection_with_verification(
        self,
        subsection_title: str,
        subsection_desc: str,
        attribute_tree_facts: str,
        outlines: Outlines,
        written_content: str,
    ) -> str:
        """
        核心方法：带验证的小节生成
        
        实现 [Planner] - Writer - Critic 循环：
        0. (CoT模式) Planner 生成写作计划
        1. 获取上下文窗口（上一小节末尾 200 字）
        2. Writer 生成初稿（根据Plan）
        3. Critic 核验（输出结构化 JSON）
        4. 如果不通过，Writer 根据 action_plan 修正
        5. 重复直到通过或达到最大重试次数
        
        Args:
            subsection_title: 小节标题
            subsection_desc: 小节描述
            attribute_tree_facts: Attribute Tree 事实数据
            outlines: 完整大纲
            written_content: 已生成的内容
        
        Returns:
            最终生成的内容
        """
        self.stats["total_subsections"] += 1
        
        # Step 0: 获取上下文窗口用于衔接
        context_window = self._get_context_window(written_content)
        
        # Step 0.5 (CoT模式): Planner 生成写作计划
        writing_plan = ""
        if self.use_cot:
            logger.debug(f"[Planner] Generating plan for: {subsection_title}")
            try:
                writing_plan = self._planner_generate_plan(
                    subsection_title=subsection_title,
                    subsection_desc=subsection_desc,
                    attribute_tree_facts=attribute_tree_facts,
                    outlines=outlines,
                    written_content=written_content,
                )
            except Exception as e:
                logger.warning(f"[Planner] Failed to generate plan: {e}, proceeding without plan")
                writing_plan = ""
        
        # Step 1: Writer 生成初稿
        logger.debug(f"[Writer] Generating draft for: {subsection_title}")
        draft = self._writer_generate(
            subsection_title=subsection_title,
            subsection_desc=subsection_desc,
            attribute_tree_facts=attribute_tree_facts,
            context_window=context_window,
            outlines=outlines,
            written_content=written_content,
            writing_plan=writing_plan,  # 传入Plan（CoT模式下有效）
        )
        
        # 如果没有 attribute_tree_facts，跳过验证
        if not attribute_tree_facts or attribute_tree_facts.strip() == "":
            logger.debug(f"[Skip Verification] No attribute facts for: {subsection_title}")
            self.stats["passed_first_try"] += 1
            return draft
        
        # Step 2: Critic 核验
        logger.debug(f"[Critic] Reviewing draft for: {subsection_title}")
        review_result = self._critic_review(
            draft=draft,
            attribute_tree_facts=attribute_tree_facts,
            subsection_title=subsection_title,
        )
        
        # Step 3: 判断与迭代
        retries = 0
        while not review_result.is_pass and retries < self.max_retries:
            retries += 1
            logger.info(
                f"[Revision {retries}/{self.max_retries}] {subsection_title}\n"
                f"  Verdict: {review_result.verdict}\n"
                f"  Fact Errors: {len(review_result.fact_errors)}\n"
                f"  Missing Points: {len(review_result.missing_points)}"
            )
            
            # Writer 根据反馈修正
            draft = self._writer_revise(
                original_draft=draft,
                attribute_tree_facts=attribute_tree_facts,
                critic_feedback=review_result,
                subsection_title=subsection_title,
                context_window=context_window,
            )
            
            # 再次验证
            review_result = self._critic_review(
                draft=draft,
                attribute_tree_facts=attribute_tree_facts,
                subsection_title=subsection_title,
            )
        
        # 记录统计信息
        if review_result.is_pass:
            if retries == 0:
                self.stats["passed_first_try"] += 1
            else:
                self.stats["passed_after_revision"] += 1
                logger.info(f"[PASSED] {subsection_title} after {retries} revision(s)")
        else:
            self.stats["max_retries_reached"] += 1
            logger.warning(
                f"[MAX RETRIES] {subsection_title} still has issues after {retries} revisions.\n"
                f"  Final verdict: {review_result.verdict}\n"
                f"  Remaining issues: {review_result.to_feedback_string()[:200]}..."
            )
        
        return draft
    
    def content_fulfill_with_verification(
        self,
        paper_dir: str,
        outlines: Outlines,
        mainbody_save_path: str,
    ) -> None:
        """
        带验证的内容生成（重写父类方法）
        
        以 Subsection 为单位进行 Writer-Critic 循环
        """
        sec2info = self.map_section_to_papers(outlines, paper_dir)
        
        tqdm_bar = tqdm(
            total=sum(len(section.sub) + 1 for section in outlines.sections),
            desc="[High Quality] Writing with verification...",
            position=0,
        )
        
        written_content = f"\\title{{{outlines.title}}}\n"
        mainbody = []
        
        for i, section in enumerate(outlines.sections):
            out1_title = section.title
            written_content += f"\n\\section{{{out1_title}}}\n"
            mainbody.append(f"\\section{{{out1_title}}}")
            
            if section.sub == []:
                section.sub.append(SingleOutline(section.title, section.desc))
            
            for j, subsection in enumerate(section.sub):
                out2_title = subsection.title
                written_content += f"\n\\subsection{{{out2_title}}}\n"
                
                # 获取该小节的 Attribute Tree 数据
                if out2_title in sec2info:
                    papers = sec2info[out2_title]
                    attribute_tree_facts = "\n\n".join(papers)
                else:
                    attribute_tree_facts = ""
                
                # 使用 Writer-Critic 循环生成内容
                with tqdm(
                    total=1,
                    desc=f"{i+1}.{j+1} {subsection.title[:15]}",
                    position=1,
                    leave=False,
                ) as bar_2:
                    result = self.generate_subsection_with_verification(
                        subsection_title=subsection.title,
                        subsection_desc=subsection.desc,
                        attribute_tree_facts=attribute_tree_facts,
                        outlines=outlines,
                        written_content=written_content,
                    )
                    bar_2.update(1)
                
                mainbody.append(result)
                written_content = "\n\n".join(mainbody)
                tqdm_bar.update()
        
        tqdm_bar.close()
        mainbody_text = "\n\n".join(mainbody)
        
        # 生成 section words（复用父类方法）
        chat_agent = ChatAgent(TokenMonitor(self.task_id, "section_words"))
        mainbody_text = self.gen_section_words(mainbody_text, chat_agent)
        
        # 保存结果
        save_result(mainbody_text, mainbody_save_path)
        
        # 输出统计信息
        planner_info = f"\n  Total Planner Calls: {self.stats['total_planner_calls']}" if self.use_cot else ""
        logger.info(
            f"\n{'='*60}\n"
            f"Multi-Agent Generation Statistics:\n"
            f"  CoT Mode: {'Enabled' if self.use_cot else 'Disabled'}\n"
            f"  Total Subsections: {self.stats['total_subsections']}\n"
            f"  Passed First Try: {self.stats['passed_first_try']} "
            f"({self.stats['passed_first_try']/max(1,self.stats['total_subsections'])*100:.1f}%)\n"
            f"  Passed After Revision: {self.stats['passed_after_revision']}\n"
            f"  Max Retries Reached: {self.stats['max_retries_reached']}\n"
            f"  Total Writer Calls: {self.stats['total_writer_calls']}\n"
            f"  Total Critic Calls: {self.stats['total_critic_calls']}{planner_info}\n"
            f"{'='*60}"
        )
    
    def run(self):
        """运行 Multi-Agent 高质量内容生成"""
        time_monitor = TimeMonitor(self.task_id)
        time_monitor.start("multi_agent_generate_content")
        
        # 复用父类的 ChatAgent 用于 mount 等操作
        chat_agent = ChatAgent(TokenMonitor(self.task_id, "multi_agent_mount"))
        
        # 1. Mount trees on outlines
        outlines = Outlines.from_saved(self.outlines_path)
        self.mount_trees_on_outlines(self.paper_dir, outlines, chat_agent)
        
        tmp_dir = self.work_dir / "tmp"
        if not tmp_dir.exists():
            tmp_dir.mkdir(exist_ok=True, parents=True)
        
        # 2. Overview the mount details
        mount_detail_fig_path = tmp_dir / "mount_details.jpg"
        self.draw_mount_details(self.paper_dir, mount_detail_fig_path)
        
        # 3. Content fulfill with verification (核心区别)
        main_body_raw_path = tmp_dir / "mainbody.raw.tex"
        self.content_fulfill_with_verification(
            self.paper_dir, outlines, main_body_raw_path
        )
        
        # 4. Generate abstract
        abstract_save_path = tmp_dir / "abstract.tex"
        self.gen_abstract(main_body_raw_path, abstract_save_path, chat_agent)
        
        # 5. Post revise
        main_body_save_path = tmp_dir / "mainbody.tex"
        self.post_revise(main_body_raw_path, main_body_save_path, self.paper_dir)
        
        time_monitor.end("multi_agent_generate_content")
        logger.info("Multi-Agent content generation completed.")


# 测试入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Multi-Agent Generator")
    parser.add_argument("--task_id", type=str, required=True, help="Task ID to process")
    args = parser.parse_args()
    
    generator = MultiAgentGenerator(args.task_id)
    generator.run()

