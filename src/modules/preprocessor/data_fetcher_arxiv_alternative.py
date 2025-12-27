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

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logger.warning("BeautifulSoup4 not installed. HTML extraction will be limited.")

try:
    import tarfile
    HAS_TARFILE = True
except ImportError:
    HAS_TARFILE = False


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
        
        # 提取 arxiv ID
        arxiv_id = result.entry_id.split('/')[-1] if '/' in result.entry_id else result.entry_id
        
        # 方法1: 尝试从 LaTeX 源码获取（最准确）
        try:
            md_text, reference, images = self._extract_from_latex_source(result, arxiv_id)
            if md_text:
                logger.debug(f"Successfully extracted from LaTeX source for {arxiv_id}")
                return md_text, reference, images
        except Exception as e:
            logger.debug(f"Failed to extract from LaTeX source: {e}")
        
        # 方法2: 从 HTML 页面提取（较准确）
        try:
            md_text, reference, images = self._extract_from_html(result, arxiv_id)
            if md_text:
                logger.debug(f"Successfully extracted from HTML for {arxiv_id}")
                return md_text, reference, images
        except Exception as e:
            logger.debug(f"Failed to extract from HTML: {e}")
        
        # 方法3: 从 PDF 提取（备用方案，可能不够准确）
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
        
        # 如果所有方法都失败，至少生成基本的 BibTeX
        if not reference:
            reference = self._generate_bibtex(result)
        
        return md_text, reference, images
    
    def _extract_from_latex_source(self, result: arxiv.Result, arxiv_id: str) -> tuple[str, str, List[Dict]]:
        """
        从 LaTeX 源码提取内容（最准确的方法）
        
        arXiv LaTeX 源码下载 URL: https://arxiv.org/src/{arxiv_id}
        或: https://arxiv.org/e-print/{arxiv_id}
        """
        md_text = ""
        reference = ""
        images = []
        
        try:
            # 尝试下载 LaTeX 源码（通常是 tar.gz 格式）
            # 注意：不是所有论文都有源码，有些只有 PDF
            latex_url = f"https://arxiv.org/src/{arxiv_id}"
            response = requests.get(latex_url, timeout=30, allow_redirects=True)
            
            # 如果返回的是 tar.gz 文件
            if response.headers.get('content-type', '').startswith('application/x-tar') or \
               response.headers.get('content-type', '').startswith('application/gzip'):
                import io
                import tempfile
                import os
                
                # 解压 tar.gz
                with tempfile.TemporaryDirectory() as tmpdir:
                    tar_path = os.path.join(tmpdir, f"{arxiv_id}.tar.gz")
                    with open(tar_path, 'wb') as f:
                        f.write(response.content)
                    
                    # 解压
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        tar.extractall(tmpdir)
                    
                    # 查找主 .tex 文件
                    tex_files = list(Path(tmpdir).rglob("*.tex"))
                    if not tex_files:
                        raise ValueError("No .tex files found in source")
                    
                    # 找到主文件（通常是最大的或包含 \documentclass 的）
                    main_tex = None
                    for tex_file in tex_files:
                        content = tex_file.read_text(encoding='utf-8', errors='ignore')
                        if '\\documentclass' in content:
                            if main_tex is None or len(content) > len(main_tex.read_text(encoding='utf-8', errors='ignore')):
                                main_tex = tex_file
                    
                    if main_tex is None:
                        main_tex = max(tex_files, key=lambda f: f.stat().st_size)
                    
                    # 读取主文件内容
                    tex_content = main_tex.read_text(encoding='utf-8', errors='ignore')
                    
                    # 转换为 Markdown
                    md_text = self._latex_to_markdown(tex_content, result)
                    
                    # 提取参考文献
                    reference = self._extract_references_from_latex(tex_content, result)
                    
                    # 提取图片引用
                    images = self._extract_images_from_latex(tex_files, tmpdir, result)
            
            else:
                # 可能是 HTML 页面，尝试解析
                raise ValueError("Source is not a tar.gz file")
                
        except Exception as e:
            logger.debug(f"LaTeX source extraction failed: {e}")
            raise
        
        return md_text, reference, images
    
    def _extract_from_html(self, result: arxiv.Result, arxiv_id: str) -> tuple[str, str, List[Dict]]:
        """
        从 arXiv HTML 页面提取内容
        """
        md_text = ""
        reference = ""
        images = []
        
        if not HAS_BS4:
            raise ImportError("BeautifulSoup4 is required for HTML extraction")
        
        try:
            # 访问 HTML 页面
            html_url = f"https://arxiv.org/abs/{arxiv_id}"
            response = requests.get(html_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取标题
            title = result.title
            if not title:
                title_elem = soup.find('h1', class_='title')
                if title_elem:
                    title = title_elem.get_text().replace('Title:', '').strip()
            
            # 提取作者
            authors = result.authors
            if not authors:
                authors_elem = soup.find('div', class_='authors')
                if authors_elem:
                    authors = [a.get_text().strip() for a in authors_elem.find_all('a')]
            
            # 提取摘要
            abstract = result.summary
            if not abstract:
                abstract_elem = soup.find('blockquote', class_='abstract')
                if abstract_elem:
                    abstract = abstract_elem.get_text().replace('Abstract:', '').strip()
            
            # 构建 Markdown
            md_parts = [f"# {title}\n\n"]
            if authors:
                authors_str = ', '.join(str(a) for a in authors)
                md_parts.append(f"**Authors:** {authors_str}\n\n")
            if abstract:
                md_parts.append(f"**Abstract:**\n\n{abstract}\n\n")
            
            # 尝试提取正文（如果有的话）
            # arXiv HTML 页面通常不包含完整正文，只有摘要
            # 但我们可以尝试从其他部分提取信息
            content_div = soup.find('div', class_='full-text')
            if content_div:
                # 如果有完整文本，转换为 Markdown
                if HAS_MARKDOWNIFY:
                    md_parts.append(md(str(content_div)))
                else:
                    md_parts.append(content_div.get_text())
            
            md_text = "\n".join(md_parts)
            
            # 提取参考文献（从 HTML 中查找）
            ref_section = soup.find('div', id='bibtex')
            if ref_section:
                reference = ref_section.get_text().strip()
            else:
                # 生成基本的 BibTeX
                reference = self._generate_bibtex(result)
            
            # 提取图片（从 HTML 中查找）
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('src', '')
                if src and ('arxiv' in src or 'figure' in src.lower()):
                    images.append({
                        "figure_link": src if src.startswith('http') else f"https://arxiv.org{src}",
                        "figure_desc": img.get('alt', 'Figure'),
                        "figure_size": ""
                    })
        
        except Exception as e:
            logger.debug(f"HTML extraction failed: {e}")
            raise
        
        return md_text, reference, images
    
    def _latex_to_markdown(self, tex_content: str, result: arxiv.Result) -> str:
        """
        将 LaTeX 内容转换为 Markdown
        """
        md_parts = []
        
        # 添加标题和作者
        md_parts.append(f"# {result.title}\n\n")
        authors_str = ', '.join(str(a) for a in result.authors)
        md_parts.append(f"**Authors:** {authors_str}\n\n")
        md_parts.append(f"**Abstract:**\n\n{result.summary}\n\n")
        
        # 移除 LaTeX 命令，保留文本内容
        # 这是一个简化的转换，可以改进
        text = tex_content
        
        # 移除注释
        text = re.sub(r'%.*?$', '', text, flags=re.MULTILINE)
        
        # 移除常见的 LaTeX 命令，保留内容
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)  # \command{content} -> content
        text = re.sub(r'\\[a-zA-Z]+', '', text)  # 移除单独的 LaTeX 命令
        
        # 处理章节
        text = re.sub(r'\\section\*?\{([^}]*)\}', r'## \1', text)
        text = re.sub(r'\\subsection\*?\{([^}]*)\}', r'### \1', text)
        text = re.sub(r'\\subsubsection\*?\{([^}]*)\}', r'#### \1', text)
        
        # 处理强调
        text = re.sub(r'\\textbf\{([^}]*)\}', r'**\1**', text)
        text = re.sub(r'\\textit\{([^}]*)\}', r'*\1*', text)
        
        # 处理数学公式（简化处理）
        text = re.sub(r'\$([^$]*)\$', r'$\1$', text)  # 保留行内公式
        text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)  # 块级公式
        
        # 清理多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        md_parts.append(text)
        
        return "\n".join(md_parts)
    
    def _extract_references_from_latex(self, tex_content: str, result: arxiv.Result) -> str:
        """
        从 LaTeX 源码中提取参考文献
        """
        # 查找 \bibliography{} 命令
        bib_match = re.search(r'\\bibliography\{([^}]+)\}', tex_content)
        if bib_match:
            bib_file = bib_match.group(1)
            # 注意：需要读取对应的 .bib 文件
            # 这里先返回生成的 BibTeX
            pass
        
        # 查找 \begin{thebibliography} 环境
        bib_match = re.search(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', 
                             tex_content, re.DOTALL)
        if bib_match:
            bib_content = bib_match.group(0)
            # 可以进一步解析，但这里先返回生成的
            pass
        
        # 如果找不到，生成基本的 BibTeX
        return self._generate_bibtex(result)
    
    def _extract_images_from_latex(self, tex_files: List[Path], base_dir: str, result: arxiv.Result) -> List[Dict]:
        """
        从 LaTeX 源码中提取图片引用
        """
        images = []
        
        for tex_file in tex_files:
            content = tex_file.read_text(encoding='utf-8', errors='ignore')
            
            # 查找 \includegraphics 命令
            pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
            matches = re.findall(pattern, content)
            
            for img_path in matches:
                # 移除可能的扩展名（LaTeX 会自动添加）
                img_path_clean = img_path.replace('.pdf', '').replace('.png', '').replace('.jpg', '')
                
                # 查找实际文件
                possible_exts = ['.pdf', '.png', '.jpg', '.jpeg', '.eps']
                actual_file = None
                for ext in possible_exts:
                    full_path = Path(base_dir) / img_path_clean
                    if full_path.with_suffix(ext).exists():
                        actual_file = full_path.with_suffix(ext)
                        break
                    # 也尝试相对路径
                    for tex_dir in [tex_file.parent]:
                        test_path = tex_dir / img_path_clean
                        if test_path.with_suffix(ext).exists():
                            actual_file = test_path.with_suffix(ext)
                            break
                
                if actual_file:
                    images.append({
                        "figure_link": str(actual_file),
                        "figure_desc": f"Figure: {img_path_clean}",
                        "figure_size": ""
                    })
        
        return images
    
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