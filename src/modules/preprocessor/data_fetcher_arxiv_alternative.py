"""
使用 arxiv 包实现的替代方案示例
这个文件展示了如何使用 pip 的 arxiv 包来替代内部 API
"""
import sys
from pathlib import Path

# 添加项目根目录到 sys.path，以便可以直接运行此文件
FILE_PATH = Path(__file__).absolute()
# 向上查找包含 src 目录的项目根目录
BASE_DIR = FILE_PATH.parent
while BASE_DIR != BASE_DIR.parent and not (BASE_DIR / "src").exists():
    BASE_DIR = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

import arxiv
import time
import re
import requests
from typing import List, Dict, Optional
from collections import Counter
from pathlib import Path
import json

from src.configs.logger import get_logger

logger = get_logger("src.modules.preprocessor.DataFetcherArxiv")

# 可选依赖：用于 PDF 和 LaTeX 处理
try:
    import pymupdf  # PyMuPDF (fitz)
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF not installed. PDF extraction will be limited.")

try:
    from markdownify import markdownify as md
    HAS_MARKDOWNIFY = True
except ImportError:
    HAS_MARKDOWNIFY = False


class DataFetcherArxivAlternative:
    """使用 arxiv 包实现的替代方案"""
    
    BATCH_SIZE = 200
    SINGLE_WORD_LIMIT = 1000
    # arXiv API 速率限制：每秒最多 1 个请求
    RATE_LIMIT_DELAY = 1.1  # 稍微大于 1 秒，确保不超限
    
    def __init__(self):
        self.client = arxiv.Client(
            page_size=100,  # arXiv API 每页最多 100 条
            delay_seconds=self.RATE_LIMIT_DELAY,
            num_retries=3
        )
    
    def _get_data_arxiv(
        self,
        keyword: str,
        projection: str = "",
        last_id: str = "00000000000000000000000000000000",
        start_index: int = 0,  # 使用索引而不是 last_id
    ) -> List[Dict]:
        """
        使用 arxiv 包获取论文数据
        
        注意：arxiv 包不支持基于 last_id 的分页，而是使用 start_index
        """
        try:
            # 构建查询字符串：在标题或摘要中搜索关键词
            query = f'all:"{keyword}"'
            
            # 创建搜索对象
            search = arxiv.Search(
                query=query,
                max_results=min(self.BATCH_SIZE, self.SINGLE_WORD_LIMIT - start_index),
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            # 由于 arxiv 包的限制，我们需要手动实现分页
            # 通过跳过前面的结果来实现
            results = list(self.client.results(search))
            
            # 从 start_index 开始获取结果
            for i, result in enumerate(results[start_index:start_index + self.BATCH_SIZE]):
                paper = self._convert_arxiv_result_to_dict(result)
                papers.append(paper)
                
                # 如果达到限制，停止
                if len(papers) >= self.BATCH_SIZE:
                    break
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to fetch papers batch from arxiv: {str(e)}")
            return []
    
    def _convert_arxiv_result_to_dict(self, result: arxiv.Result, extract_full_content: bool = False) -> Dict:
        """
        将 arxiv.Result 转换为与原始格式兼容的字典
        
        Args:
            result: arxiv.Result 对象
            extract_full_content: 是否提取完整内容（md_text, reference, image）
                                注意：这会下载 PDF，速度较慢
        """
        # 提取 arxiv ID（格式：arXiv:1234.5678v1）
        arxiv_id = result.entry_id.split('/')[-1] if '/' in result.entry_id else result.entry_id
        
        paper = {
            "_id": arxiv_id,
            "title": result.title,
            "authors": [str(author) for author in result.authors],
            "abstract": result.summary,
            "detail_url": result.entry_id,
            "detail_id": arxiv_id,
            # 以下字段 arxiv 包不直接提供，需要额外处理
            "md_text": "",  # Markdown 格式的论文全文
            "reference": "",  # BibTeX 格式的参考文献条目（字符串）
            "image": [],  # 图片 URL 列表
        }
        
        # 如果需要提取完整内容，下载并处理 PDF
        if extract_full_content:
            try:
                md_text, reference, images = self._extract_full_content(result)
                paper["md_text"] = md_text
                paper["reference"] = reference
                paper["image"] = images
            except Exception as e:
                logger.warning(f"Failed to extract full content for {arxiv_id}: {e}")
        
        return paper
    
    def _extract_full_content(self, result: arxiv.Result) -> tuple[str, str, List[Dict]]:
        """
        从 arXiv 论文中提取完整内容
        
        Returns:
            tuple: (md_text, reference, images)
            - md_text: Markdown 格式的论文全文
            - reference: BibTeX 格式的参考文献（字符串）
            - images: 图片信息列表，每个元素是包含 figure_link, figure_desc 等的字典
        """
        md_text = ""
        reference = ""
        images = []
        
        # 方法1: 尝试从 LaTeX 源码获取（更准确）
        try:
            # arXiv 提供 LaTeX 源码访问
            latex_url = result.entry_id.replace('/abs/', '/src/').replace('/pdf/', '/src/')
            # 注意：arXiv 的 LaTeX 源码访问需要特殊权限，这里仅作示例
            # 实际使用时可能需要使用 arXiv API 的其他端点
        except:
            pass
        
        # 方法2: 从 PDF 提取（更通用但可能不够准确）
        if result.pdf_url and HAS_PYMUPDF:
            try:
                # 下载 PDF
                pdf_response = requests.get(result.pdf_url, timeout=30)
                pdf_response.raise_for_status()
                
                # 使用 PyMuPDF 提取文本和图片
                import io
                pdf_bytes = io.BytesIO(pdf_response.content)
                doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                
                # 提取文本并转换为 Markdown
                md_text_parts = []
                md_text_parts.append(f"# {result.title}\n\n")
                md_text_parts.append(f"**Authors:** {', '.join(str(a) for a in result.authors)}\n\n")
                md_text_parts.append(f"**Abstract:**\n\n{result.summary}\n\n")
                
                # 提取正文
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        md_text_parts.append(f"## Page {page_num + 1}\n\n{text}\n\n")
                
                md_text = "\n".join(md_text_parts)
                
                # 提取图片
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            # 图片数据在 base_image["image"] 中
                            # 可以保存到本地或上传到图床
                            images.append({
                                "figure_link": f"{result.entry_id}#page={page_num+1}&image={img_index}",
                                "figure_desc": f"Figure from page {page_num + 1}",
                                "figure_size": f"{base_image.get('width', 'unknown')}x{base_image.get('height', 'unknown')}"
                            })
                        except Exception as e:
                            logger.debug(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                
                doc.close()
                
            except Exception as e:
                logger.warning(f"Failed to extract from PDF: {e}")
        
        # 生成 BibTeX 格式的参考文献
        reference = self._generate_bibtex(result)
        
        return md_text, reference, images
    
    def _generate_bibtex(self, result: arxiv.Result) -> str:
        """
        生成 BibTeX 格式的参考文献条目
        
        格式示例：
        @article{author2024title,
            title={Title},
            author={Author1 and Author2},
            journal={arXiv preprint arXiv:1234.5678},
            year={2024}
        }
        """
        # 生成 bib_name（用于引用）
        first_author = str(result.authors[0]) if result.authors else "unknown"
        # 提取姓氏
        author_lastname = first_author.split()[-1] if first_author.split() else "unknown"
        year = result.published.year if result.published else "unknown"
        bib_name = f"{author_lastname}{year}"
        
        # 清理标题中的特殊字符
        title = result.title.replace("{", "").replace("}", "")
        
        # 格式化作者列表
        authors_str = " and ".join(str(author) for author in result.authors)
        
        # 生成 BibTeX
        bibtex = f"""@article{{{bib_name},
    title={{{title}}},
    author={{{authors_str}}},
    journal={{arXiv preprint {result.entry_id.split('/')[-1]}}},
    year={{{year}}},
    url={{{result.entry_id}}}
}}"""
        
        return bibtex
    
    def search_on_arxiv_single_word(
        self, key_word: str, projection: str = ""
    ) -> List[Dict]:
        """
        搜索单个关键词的论文
        注意：这个方法需要修改调用方式，因为分页机制不同
        """
        papers = []
        start_index = 0
        
        logger.debug(f"Searching papers from arxiv which keyword is {key_word}")
        
        while len(papers) < self.SINGLE_WORD_LIMIT:
            batch = self._get_data_arxiv(
                keyword=key_word,
                projection=projection,
                start_index=start_index
            )
            
            if not batch:
                break
            
            for paper in batch:
                paper["from"] = "arxiv"
            papers.extend(batch)
            start_index += len(batch)
            
            # 如果返回的数量少于批次大小，说明没有更多结果了
            if len(batch) < self.BATCH_SIZE:
                break
            
            # 速率限制：等待一下再继续
            time.sleep(self.RATE_LIMIT_DELAY)
        
        logger.debug(
            f"arxiv: Retrieved {len(papers)} papers which key_word is {key_word}"
        )
        return papers
    
    def search_on_arxiv(self, key_words: str) -> List[Dict]:
        """
        搜索多个关键词，返回重叠的论文
        与原方法功能相同
        """
        key_words = key_words.split(",")
        id_counter = Counter()
        id2paper = {}
        
        for key_word in key_words:
            papers = self.search_on_arxiv_single_word(key_word.strip())
            
            _ids = [paper["_id"] for paper in papers]
            id_counter.update(_ids)
            id2paper.update({paper["_id"]: paper for paper in papers})
        
        overlaped_papers = [
            paper for _id, paper in id2paper.items() if id_counter[_id] >= 1
        ]
        logger.debug(
            f"Searched {len(id_counter)} papers for {key_words}, "
            f"return {len(overlaped_papers)} overlaped papers"
        )
        return overlaped_papers


# 使用示例
if __name__ == "__main__":
    fetcher = DataFetcherArxivAlternative()
    papers = fetcher.search_on_arxiv("machine learning,deep learning")
    print(f"Found {len(papers)} papers")

    # show the first paper
    print(papers[0])