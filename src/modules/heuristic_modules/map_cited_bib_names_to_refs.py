import re
from pathlib import Path

from src.configs.utils import load_latest_task_id
from src.configs.constants import OUTPUT_DIR
from src.configs.logger import get_logger
from src.modules.utils import save_result, load_file_as_string
from src.modules.latex_handler.utils import fuzzy_match

logger = get_logger("src.modules.heuristic_modules.BibNameReplacer")


class BibNameReplacer(object):
    def __init__(self, task_id: str = None):
        self.ref_bibs = None
        self.task_id = task_id

        # ======== settings ========
        self.pattern_of_bib_name_in_paper = r"\\cite[t|p]*\{(.*?)\}"
        self.pattern_of_bib_name_in_references = (
            r"@[\w\-]+\{([^,]+),"  # 匹配 @xxx{ 后面的内容直到第一个逗号
        )
        self.ref_file_path = Path(f"{OUTPUT_DIR}/{task_id}/latex/references.bib")

        self.collect_ref_bibs()

    def collect_ref_bibs(self):
        ref_content = load_file_as_string(path=self.ref_file_path)
        bib_names = re.findall(
            pattern=self.pattern_of_bib_name_in_references, string=ref_content
        )
        self.ref_bibs = bib_names

    def process(self, content: str):
        """
        处理内容中的引用名称，替换无效引用并记录警告
        
        Returns:
            处理后的内容字符串
        """
        # 先收集新的缩写对
        bibs_in_content = re.findall(
            pattern=self.pattern_of_bib_name_in_paper, string=content
        )
        bibs_in_content = set(bibs_in_content)
        
        # 统计信息
        total_citations = 0
        invalid_citations = []
        replaced_citations = []
        
        for bib_name_content in bibs_in_content:
            bib_names = [one.strip() for one in bib_name_content.split(",")]
            for bib_name in bib_names:
                total_citations += 1
                if bib_name not in self.ref_bibs:
                    # 使用模糊匹配找到最接近的引用名称
                    closet_ref_bib_name = fuzzy_match(
                        text=bib_name, candidates=self.ref_bibs
                    )[0]
                    
                    if closet_ref_bib_name != bib_name:
                        invalid_citations.append(bib_name)
                        replaced_citations.append((bib_name, closet_ref_bib_name))
                        logger.warning(
                            f"引用名称 '{bib_name}' 不在 reference.bib 中；已替换为 '{closet_ref_bib_name}'"
                        )
                        content = content.replace(bib_name, closet_ref_bib_name)
                    else:
                        # 模糊匹配也找不到，记录错误
                        invalid_citations.append(bib_name)
                        logger.error(
                            f"引用名称 '{bib_name}' 不在 reference.bib 中，且无法找到相似的引用名称进行替换"
                        )
        
        # 输出统计信息
        if invalid_citations:
            logger.warning(
                f"BibNameReplacer 处理完成：共 {total_citations} 个引用，"
                f"其中 {len(invalid_citations)} 个无效引用，"
                f"已替换 {len(replaced_citations)} 个"
            )
            if len(invalid_citations) > len(replaced_citations):
                logger.error(
                    f"仍有 {len(invalid_citations) - len(replaced_citations)} 个无效引用无法修复"
                )
        else:
            logger.debug(f"BibNameReplacer 处理完成：所有 {total_citations} 个引用均有效")
        
        return content
    
    def validate_all_citations(self, content: str) -> dict:
        """
        验证内容中的所有引用是否都在 references.bib 中
        
        Args:
            content: 要验证的内容
        
        Returns:
            包含验证结果的字典：
            {
                "valid": bool,
                "invalid_citations": list[str],
                "total_citations": int,
                "valid_citations": int,
                "replacement_suggestions": list[tuple[str, str]]  # (invalid, suggested)
            }
        """
        bibs_in_content = re.findall(
            pattern=self.pattern_of_bib_name_in_paper, string=content
        )
        bibs_in_content = set(bibs_in_content)
        
        all_citation_names = []
        for bib_name_content in bibs_in_content:
            bib_names = [one.strip() for one in bib_name_content.split(",")]
            all_citation_names.extend(bib_names)
        
        all_citation_names = list(set(all_citation_names))
        
        invalid_citations = []
        replacement_suggestions = []
        
        for bib_name in all_citation_names:
            if bib_name not in self.ref_bibs:
                invalid_citations.append(bib_name)
                # 尝试找到最接近的引用名称
                closest = fuzzy_match(text=bib_name, candidates=self.ref_bibs)[0]
                if closest:
                    replacement_suggestions.append((bib_name, closest))
        
        valid_citations = [
            name for name in all_citation_names 
            if name in self.ref_bibs
        ]
        
        return {
            "valid": len(invalid_citations) == 0,
            "invalid_citations": invalid_citations,
            "total_citations": len(all_citation_names),
            "valid_citations": len(valid_citations),
            "replacement_suggestions": replacement_suggestions,
        }


# 示例使用
if __name__ == "__main__":
    task_id = load_latest_task_id()
    print(f"task_id: {task_id}")
    replacer = BibNameReplacer(task_id=task_id)
    content = """{
\begin{figure}[ht!]
\centering
            \subfloat[Memory Usage and Throughput of Existing Systems vs vLLM\cite{kwon2023efficiently}]{\includegraphics[width=0.28\textwidth]{figs/Memory Usage and Throughput of Existing Systems vs vLLM.jpg}}\hspace{0.03\textwidth}
            
            \subfloat[Architecture of a Distributed System\cite{kwon2023efficient}]{\includegraphics[width=0.28\textwidth]{figs/Architecture of a Distributed System.jpg}}\hspace{0.03\textwidth}
            
            \subfloat[Model Structure Overview\cite{pope2022efficiently}]{\includegraphics[width=0.28\textwidth]{figs/Model Structure Overview.jpg}}\hspace{0.03\textwidth}
            
\end{figure}
}
"""
    processed_text = replacer.process(content)
