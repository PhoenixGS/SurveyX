"""
@Reference:
1. How to create llama index templates: https://blog.csdn.net/lovechris00/article/details/137782020
"""

import json
import os
import re
import random as normalrandom
from pathlib import Path

from typing import List, Union, Dict
from tqdm import tqdm
from llama_index.core import ChatPromptTemplate, PromptTemplate

from src.configs.config import BASE_DIR
from src.configs.constants import OUTPUT_DIR, RESOURCE_DIR
from src.configs.logger import get_logger
from src.configs.utils import load_latest_task_id
from src.modules.utils import (
    load_file_as_string,
    save_result,
    load_prompt,
    clean_chat_agent_format,
)
from src.schemas.paragraph import Paragraph
from src.modules.heuristic_modules import AbbrReplacer, BibNameReplacer
from src.modules.post_refine import BaseRefiner

logger = get_logger("src.modules.post_refine.RuleBasedRefiner")


class RuleBasedRefiner(BaseRefiner):
    def __init__(self, task_id: str = None, **kwargs) -> None:
        super().__init__(task_id=task_id, **kwargs)
        self.refined_mainbody_path = Path(
            f"{OUTPUT_DIR}/{self.task_id}/tmp/mainbody_rule_refined.tex"
        )

        # functions
        self.abbr_replacer = AbbrReplacer()
        self.bib_name_replacer = BibNameReplacer(task_id=self.task_id)
        self.sp_phrases_to_be_rm = [
            "Certainly! Below is the rewritten content following your instructions:"
        ]

        # replace rules to be executed
        self.re_pattern_rm_tokens_1 = re.compile(
            pattern=r"\*\*.*?\*\*", flags=re.DOTALL
        )
        self.re_pattern_rm_tokens_2 = re.compile(
            pattern=r"Certainly!.*?following your instructions:", flags=re.DOTALL
        )

        # 使用self.abbr_replacer.process会出现未知的重复bug，只能去掉
        self.replace_rules = [
            # self.abbr_replacer.process,
            self.bib_name_replacer.process,
            self.remove_unexpected_tokens,
        ]

    def rule_replace_pipeline(self, content: str, funcs: list):
        for func in funcs:
            content = func(content=content)
        return content

    def remove_unexpected_tokens(self, content: str):
        for one in self.sp_phrases_to_be_rm:
            content = content.replace(one, "")
        # remove LLM redundant words
        content = re.sub(self.re_pattern_rm_tokens_1, "", content)
        content = re.sub(self.re_pattern_rm_tokens_2, "", content)
        return clean_chat_agent_format(content=content)

    def find_differences(self, str1, str2):
        # 找出较短的字符串长度
        sents1 = str1.strip().split(".")
        sents2 = str2.strip().split(".")
        # 比较两个字符串并记录差异
        differences = []
        for i, (sent_1, sent_2) in enumerate(zip(sents1, sents2)):
            if sent_1 != sent_2:
                differences.append(
                    (
                        i,
                        sent_1.strip().replace("\n\n", ""),
                        sent_2.strip().replace("\n\n", ""),
                    )
                )

        return differences

    def show_differences(
        self, differences, str1_label="revised_content", str2_label="sec_content"
    ):
        for index, sent_1, sent_2 in differences:
            logger.info(
                f"Position {index}: {str1_label} has '{sent_1}'|||||| {str2_label} has '{sent_2}'"
            )

    def run(self, mainbody_path=None, debug=False):
        if mainbody_path is None:
            mainbody_path = self.mainbody_path
        survey_sections = self.load_survey_sections(mainbody_path)
        revised_content_list = []
        
        # 统计信息
        total_sections = len(survey_sections)
        sections_with_invalid_citations = 0
        
        for sec in tqdm(survey_sections, desc="Rule based refining..."):
            # 在处理前验证引用
            validation_before = self.bib_name_replacer.validate_all_citations(sec.content)
            
            revised_content = self.rule_replace_pipeline(
                content=sec.content, funcs=self.replace_rules
            )
            
            # 在处理后再次验证引用
            validation_after = self.bib_name_replacer.validate_all_citations(revised_content)
            
            if debug:
                differences = self.find_differences(
                    str1=sec.content, str2=revised_content
                )
                self.show_differences(differences=differences)
                
                # 显示引用验证信息
                if not validation_after["valid"]:
                    logger.warning(
                        f"Section '{sec.title[:50] if hasattr(sec, 'title') else 'Unknown'}' "
                        f"contains {len(validation_after['invalid_citations'])} invalid citations: "
                        f"{validation_after['invalid_citations'][:3]}"
                    )
            
            if not validation_after["valid"]:
                sections_with_invalid_citations += 1
                logger.warning(
                    f"处理后的内容仍包含无效引用：{len(validation_after['invalid_citations'])} 个"
                )
            
            revised_content_list.append(revised_content)
        
        refined_survey_text = "\n".join(revised_content_list)
        
        # 最终验证整个文档
        final_validation = self.bib_name_replacer.validate_all_citations(refined_survey_text)
        if not final_validation["valid"]:
            logger.error(
                f"RuleBasedRefiner 处理完成，但仍有 {len(final_validation['invalid_citations'])} 个无效引用："
                f"{final_validation['invalid_citations'][:10]}"
            )
            logger.info(
                f"建议替换：{final_validation['replacement_suggestions'][:5]}"
            )
        else:
            logger.info(
                f"RuleBasedRefiner 处理完成：所有 {final_validation['total_citations']} 个引用均有效"
            )
        
        save_result(refined_survey_text, self.refined_mainbody_path)
        logger.debug(f"Save content to {self.refined_mainbody_path}.")
        return refined_survey_text


if __name__ == "__main__":
    task_id = load_latest_task_id()
    print(f"task_id: {task_id}")
    # store vector index into local directory for the convenience of debugging
    temp_rewriter = RuleBasedRefiner(task_id=task_id, llamaindex_store_local=False)
    mainbody_path = Path(f"{OUTPUT_DIR}/{task_id}/tmp/mainbody_fig_refined.tex")
    temp_rewriter.run(mainbody_path=mainbody_path, debug=True)
