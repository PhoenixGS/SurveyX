import sys
from pathlib import Path

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
    RATE_LIMIT_DELAY = 1.5  # 稍微大于 1 秒，确保不超限
    
    def __init__(self, max_papers_per_search: int = None):
        """
        Args:
            max_papers_per_search: 每次搜索返回的最大论文数量（None 表示不限制）
        """
        self.client = arxiv.Client(
            page_size=100,  # arXiv API 每页最多 100 条
            delay_seconds=self.RATE_LIMIT_DELAY,
            num_retries=3
        )
        self.max_papers_per_search = max_papers_per_search or self.SINGLE_WORD_LIMIT
    
    def _get_data_arxiv(
        self,
        keyword: str,
        projection: str = "",
        last_id: str = "00000000000000000000000000000000",
        start_index: int = 0,
    ) -> List[Dict]:
        """
        使用 arxiv 包获取论文数据
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
            # will be processed later
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
        latex source -> html -> pdf
        
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
        
        # extract from /html/ page
        try:
            md_text, reference, images = self._extract_from_html(result, arxiv_id)
            if md_text:
                logger.debug(f"Successfully extracted from HTML for {arxiv_id}")
                return md_text, reference, images
        except Exception as e:
            logger.debug(f"Failed to extract from HTML: {e}")

        # extract from /src/ page
        try:
            md_text, reference, images = self._extract_from_latex_source(result, arxiv_id)
            if md_text:
                logger.debug(f"Successfully extracted from LaTeX source for {arxiv_id}")
                return md_text, reference, images
        except Exception as e:
            logger.debug(f"Failed to extract from LaTeX source: {e}")
        
        # extract from /pdf/ page
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
        

        # TODO: need to be verified
        # reference is the internal references of the paper, not the reference of the paper itself
        # or return reference of the paper itself
        
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
                    
                    # 提取参考文献（论文内部的参考文献列表）
                    reference = self._extract_references_from_latex(tex_content, result, tmpdir)
                    
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
        注意：这里使用的是 /html/ 页面（完整论文HTML），而不是 /abs/ 页面（摘要页面）
        """
        md_text = ""
        reference = ""
        images = []
        
        if not HAS_BS4:
            raise ImportError("BeautifulSoup4 is required for HTML extraction")
        
        try:
            # 访问 HTML 页面（/html/ 页面包含完整的论文内容）
            # 移除版本号后缀（如果有的话），因为 /html/ 页面不需要版本号
            arxiv_id_clean = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
            html_url = f"https://arxiv.org/html/{arxiv_id_clean}"
            response = requests.get(html_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 获取 base URL（用于解析相对路径的图片）
            # HTML 中的 base 标签通常是 /html/{arxiv_id}v1/ 这样的格式
            base_tag = soup.find('base')
            if base_tag:
                base_url = base_tag.get('href', '')
                # 构建完整的 base URL
                if base_url.startswith('/'):
                    base_url_full = f"https://arxiv.org{base_url}"
                elif base_url.startswith('http'):
                    base_url_full = base_url
                else:
                    base_url_full = f"https://arxiv.org/html/{arxiv_id_clean}/{base_url}"
                # 确保以 / 结尾
                if not base_url_full.endswith('/'):
                    base_url_full += '/'
            else:
                # 如果没有 base 标签，使用默认路径
                base_url_full = f"https://arxiv.org/html/{arxiv_id_clean}/"
            
            # 提取论文主内容区域（article标签包含完整的论文内容）
            article = soup.find('article', class_='ltx_document')
            if not article:
                # 如果找不到 article，尝试查找其他可能的内容容器
                article = soup.find('article') or soup.find('div', class_='ltx_page_content')
            
            if not article:
                logger.warning(f"Could not find article content in HTML page for {arxiv_id}")
                # 如果没有找到内容，返回基本信息
                title = result.title or ""
                authors = result.authors or []
                abstract = result.summary or ""
                md_text = f"# {title}\n\n**Authors:** {', '.join(str(a) for a in authors)}\n\n**Abstract:**\n\n{abstract}\n\n"
                return md_text, reference, images
            
            # 提取标题
            title = result.title
            if not title:
                title_elem = article.find('h1', class_='ltx_title')
                if title_elem:
                    title = title_elem.get_text().strip()
            
            # 提取作者
            authors = result.authors
            if not authors:
                authors_elem = article.find('div', class_='ltx_authors')
                if authors_elem:
                    # 提取作者姓名（可能有复杂的结构，尝试提取文本）
                    authors_text = authors_elem.get_text()
                    # 简化处理：如果无法提取结构化作者，使用摘要页面的作者
                    pass
            
            # 提取摘要
            abstract = result.summary
            if not abstract:
                abstract_elem = article.find('div', class_='ltx_abstract')
                if abstract_elem:
                    # 移除 "Abstract" 标题
                    abstract_text = abstract_elem.get_text()
                    abstract = abstract_text.replace('Abstract', '').strip()
            
            # 构建 Markdown
            md_parts = [f"# {title}\n\n"]
            if authors:
                authors_str = ', '.join(str(a) for a in authors)
                md_parts.append(f"**Authors:** {authors_str}\n\n")
            if abstract:
                md_parts.append(f"**Abstract:**\n\n{abstract}\n\n")
            
            # 提取正文内容（转换为 Markdown）
            if HAS_MARKDOWNIFY:
                # 使用 markdownify 将 HTML 转换为 Markdown
                content_html = str(article)
                md_content = md(content_html, heading_style="ATX")
                # 移除标题和作者部分（已经在上面添加了）
                md_parts.append(md_content)
            else:
                # 如果没有 markdownify，使用纯文本
                md_parts.append(article.get_text())
            
            md_text = "\n".join(md_parts)
            
            # 提取参考文献（从 HTML 中查找）
            # arXiv HTML 页面可能包含参考文献列表，通常在最后的 section 或单独的 div 中
            ref_section = article.find('div', id='bibtex') or article.find('section', class_='ltx_bibliography')
            if ref_section:
                reference = ref_section.get_text().strip()
                # 检查是否是论文本身的引用（通常包含 arXiv ID）
                if arxiv_id_clean in reference or arxiv_id in reference:
                    # 这可能是论文本身的引用，不是参考文献列表，尝试查找真正的参考文献
                    # 查找 bibliography 环境或类似的内容
                    bib_items = ref_section.find_all('div', class_='ltx_bibitem')
                    if not bib_items:
                        bib_items = ref_section.find_all('li', class_='ltx_bibitem')
                    if bib_items:
                        # 构建参考文献列表
                        ref_list = []
                        for item in bib_items:
                            ref_list.append(item.get_text().strip())
                        reference = "\n\n".join(ref_list)
                    else:
                        reference = ""
            else:
                # 如果找不到参考文献区域，返回空字符串
                reference = ""
            
            # 提取图片（从 article 内容区域中查找）
            # /html/ 页面包含论文中的实际图片
            img_tags = article.find_all('img')
            
            # 定义需要排除的路径模式（logo、icon等，虽然这些通常不在 article 内）
            excluded_patterns = [
                'arxiv-logo',
                'logomark',
                'logo',
                '/icons/licenses/',
                '/static/browse/',
                'favicon',
                'icon',
            ]
            
            for img in img_tags:
                src = img.get('src', '')
                if not src:
                    continue
                
                # 检查是否是被排除的图片（logo/icon等）
                src_lower = src.lower()
                if any(pattern in src_lower for pattern in excluded_patterns):
                    continue
                
                # 构建完整的URL（处理相对路径）
                if src.startswith('http'):
                    full_url = src
                elif src.startswith('//'):
                    full_url = f"https:{src}"
                elif src.startswith('/'):
                    full_url = f"https://arxiv.org{src}"
                else:
                    # 相对路径，使用 base URL
                    full_url = f"{base_url_full}{src}"
                
                # 提取图片描述（从 figcaption 或 alt 属性）
                figure_desc = img.get('alt', '')
                if not figure_desc:
                    # 尝试从父 figure 元素获取 caption
                    parent_figure = img.find_parent('figure')
                    if parent_figure:
                        figcaption = parent_figure.find('figcaption')
                        if figcaption:
                            figure_desc = figcaption.get_text().strip()
                    if not figure_desc:
                        figure_desc = 'Figure'
                
                # 提取图片尺寸
                width = img.get('width', '')
                height = img.get('height', '')
                figure_size = f"{width}x{height}" if width and height else ""
                
                images.append({
                    "figure_link": full_url,
                    "figure_desc": figure_desc,
                    "figure_size": figure_size
                })
        
        except Exception as e:
            logger.debug(f"HTML extraction failed: {e}")
            raise
        
        return md_text, reference, images
    
    def _latex_to_markdown(self, tex_content: str, result: arxiv.Result) -> str:
        """
        将 LaTeX 内容转换为 Markdown
        """
        # TODO: need to be verified
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
    
    def _extract_references_from_latex(self, tex_content: str, result: arxiv.Result, base_dir: str = None) -> str:
        """
        从 LaTeX 源码中提取论文内部的参考文献列表（不是论文本身的引用）
        
        Returns:
            str: 参考文献的 BibTeX 字符串（多个条目，用换行分隔）
        """
        references = []
        
        # 方法1: 查找 \bibliography{filename} 命令，读取 .bib 文件
        bib_match = re.search(r'\\bibliography\{([^}]+)\}', tex_content)
        if bib_match and base_dir:
            bib_file_name = bib_match.group(1)
            # 尝试查找 .bib 文件（可能没有扩展名）
            bib_paths = [
                Path(base_dir) / f"{bib_file_name}.bib",
                Path(base_dir) / bib_file_name,
            ]
            # 也尝试在子目录中查找
            for bib_path in bib_paths:
                if not bib_path.exists():
                    # 尝试在子目录中查找
                    for subdir in Path(base_dir).rglob("*"):
                        if subdir.is_dir():
                            test_path = subdir / bib_path.name
                            if test_path.exists():
                                bib_path = test_path
                                break
                
                if bib_path.exists():
                    try:
                        # 尝试多种编码
                        bib_content = None
                        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                            try:
                                bib_content = bib_path.read_text(encoding=encoding, errors='ignore')
                                break
                            except (UnicodeDecodeError, Exception):
                                continue
                        
                        if bib_content is None:
                            logger.warning(f"Failed to read .bib file with any encoding: {bib_path}")
                            continue
                        
                        # 解析 .bib 文件，提取所有条目
                        references = self._parse_bib_file(bib_content)
                        if references:
                            logger.debug(f"Found {len(references)} references from .bib file: {bib_path}")
                            return "\n\n".join(references)
                    except Exception as e:
                        logger.debug(f"Failed to parse .bib file {bib_path}: {e}")
                        # 继续尝试其他方法
        
        # 方法2: 查找 \begin{thebibliography} 环境
        bib_match = re.search(
            r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', 
            tex_content, 
            re.DOTALL
        )
        if bib_match:
            bib_content = bib_match.group(0)
            # 解析 thebibliography 环境中的参考文献
            references = self._parse_thebibliography(bib_content)
            if references:
                logger.debug(f"Found {len(references)} references from thebibliography environment")
                return "\n\n".join(references)
        
        # 方法3: 如果找不到，返回空字符串（而不是论文本身的引用）
        # 因为 reference 字段应该是论文内部的参考文献列表
        logger.debug("No references found in LaTeX source")
        return ""
    
    def _parse_bib_file(self, bib_content: str) -> List[str]:
        """
        解析 .bib 文件，提取所有 BibTeX 条目
        使用更健壮的方法处理嵌套大括号
        
        Returns:
            List[str]: BibTeX 条目列表
        """
        entries = []
        i = 0
        content_len = len(bib_content)
        
        while i < content_len:
            # 查找下一个 @ 符号（BibTeX 条目的开始）
            if bib_content[i] != '@':
                i += 1
                continue
            
            # 找到 @，开始解析条目
            start_pos = i
            i += 1  # 跳过 @
            
            # 提取条目类型（@article, @inproceedings 等）
            entry_type = ""
            while i < content_len and (bib_content[i].isalnum() or bib_content[i] == '_'):
                entry_type += bib_content[i]
                i += 1
            
            if not entry_type:
                i += 1
                continue
            
            # 跳过空白字符
            while i < content_len and bib_content[i].isspace():
                i += 1
            
            # 应该遇到 {
            if i >= content_len or bib_content[i] != '{':
                i += 1
                continue
            
            i += 1  # 跳过 {
            
            # 提取条目 key（直到第一个逗号）
            entry_key = ""
            while i < content_len and bib_content[i] != ',':
                entry_key += bib_content[i]
                i += 1
            
            if i >= content_len:
                break
            
            i += 1  # 跳过逗号
            
            # 提取条目内容（需要匹配嵌套的大括号）
            entry_content = ""
            brace_count = 1  # 已经有一个开括号
            content_start = i
            
            while i < content_len and brace_count > 0:
                if bib_content[i] == '{':
                    brace_count += 1
                elif bib_content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # 找到了匹配的闭括号，但不包括它
                        entry_content = bib_content[content_start:i].strip()
                        break
                i += 1
            
            if entry_content:
                # 构建完整的 BibTeX 条目
                entry = f"@{entry_type}{{{entry_key.strip()},\n{entry_content}\n}}"
                entries.append(entry)
            
            # 跳过闭括号
            if i < content_len and bib_content[i] == '}':
                i += 1
        
        return entries
    
    def _parse_thebibliography(self, bib_content: str) -> List[str]:
        """
        解析 thebibliography 环境中的参考文献
        
        Returns:
            List[str]: BibTeX 条目列表（简化格式）
        """
        entries = []
        # 提取 \bibitem 条目
        bibitem_pattern = r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})'
        matches = re.finditer(bibitem_pattern, bib_content, re.DOTALL)
        
        for match in matches:
            bib_key = match.group(1)
            bib_text = match.group(2).strip()
            
            # 尝试从文本中提取信息，构建简化的 BibTeX 条目
            # 这是一个简化的解析，可以改进
            entry = f"@article{{{bib_key},\n    note={{{bib_text}}}\n}}"
            entries.append(entry)
        
        return entries
    
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
        self, key_word: str, projection: str = "", extract_full_content: bool = False
    ) -> List[Dict]:
        """
        搜索单个关键词的论文
        
        Args:
            key_word: 搜索关键词
            projection: 字段投影（未使用，保持兼容性）
            extract_full_content: 是否提取完整内容（md_text, reference, image）
                                注意：这会很慢，建议在 paper_recaller 中批量提取
        """
        papers = []
        start_index = 0
        
        logger.debug(f"Searching papers from arxiv which keyword is {key_word}")
        
        while len(papers) < min(self.SINGLE_WORD_LIMIT, self.max_papers_per_search):
            batch = self._get_data_arxiv(
                keyword=key_word,
                projection=projection,
                start_index=start_index
            )
            
            if not batch:
                break
            
            for paper in batch:
                paper["from"] = "arxiv"
                # 如果需要提取完整内容，在这里提取（但会很慢）
                if extract_full_content and (not paper.get("md_text") or not paper["md_text"].strip()):
                    try:
                        import arxiv
                        arxiv_id = paper.get("_id") or paper.get("detail_id")
                        if arxiv_id:
                            arxiv_id_clean = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
                            result = next(arxiv.Search(id_list=[arxiv_id_clean]).results())
                            full_paper = self._convert_arxiv_result_to_dict(result, extract_full_content=True)
                            if full_paper.get("md_text"):
                                paper["md_text"] = full_paper["md_text"]
                            if full_paper.get("reference"):
                                paper["reference"] = full_paper["reference"]
                            if full_paper.get("image"):
                                paper["image"] = full_paper["image"]
                    except Exception as e:
                        logger.debug(f"Failed to extract content for {paper.get('_id')}: {e}")
            
            papers.extend(batch)
            start_index += len(batch)
            
            # 如果达到上限，停止搜索
            if len(papers) >= self.max_papers_per_search:
                logger.debug(f"Reached max_papers_per_search limit: {self.max_papers_per_search}")
                break
            
            # 如果返回的数量少于批次大小，说明没有更多结果了
            if len(batch) < self.BATCH_SIZE:
                break
            
            # 速率限制：等待一下再继续
            time.sleep(self.RATE_LIMIT_DELAY)
        
        logger.debug(
            f"arxiv: Retrieved {len(papers)} papers which key_word is {key_word}"
        )
        return papers
    
    def search_on_arxiv(self, key_words: str, extract_full_content: bool = False) -> List[Dict]:
        """
        搜索多个关键词，返回重叠的论文
        与原方法功能相同
        
        Args:
            key_words: 逗号分隔的关键词
            extract_full_content: 是否提取完整内容（默认 False，建议在 paper_recaller 中批量提取）
        """
        key_words = key_words.split(",")
        id_counter = Counter()
        id2paper = {}
        
        for key_word in key_words:
            papers = self.search_on_arxiv_single_word(
                key_word.strip(), 
                extract_full_content=extract_full_content
            )
            
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


    def save_papers_for_cleaner(self, papers: List[Dict], task_id: str) -> None:
        """
        将论文保存到 data_cleaner 期望的位置和格式
        
        Args:
            papers: 论文列表
            task_id: 任务 ID
        """
        from src.configs.constants import OUTPUT_DIR
        from src.modules.utils import save_result, sanitize_filename
        import json
        
        # 创建输出目录
        output_dir = Path(OUTPUT_DIR) / task_id / "jsons"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        skipped_count = 0
        
        for paper in papers:
            # 检查 md_text 是否存在且不为空
            if "md_text" not in paper or not paper["md_text"] or not paper["md_text"].strip():
                logger.warning(f"Paper {paper.get('_id', 'unknown')} has no md_text, skipping...")
                skipped_count += 1
                continue
            
            # 确保必要的字段存在
            if "_id" not in paper:
                paper["_id"] = paper.get("detail_id", f"paper_{saved_count}")
            
            # 保存为 JSON 文件
            file_id = paper["_id"]
            filename = f"{file_id}.json"
            filename = sanitize_filename(filename)
            file_path = output_dir / filename
            
            save_result(json.dumps(paper, indent=4, ensure_ascii=False), file_path)
            saved_count += 1
        
        logger.info(f"Saved {saved_count} papers to {output_dir}, skipped {skipped_count} papers without md_text")


# 使用示例
if __name__ == "__main__":
    fetcher = DataFetcherArxivAlternative()
    papers = fetcher.search_on_arxiv("machine learning,deep learning")
    print(f"Found {len(papers)} papers")
    
    # 如果需要保存供 data_cleaner 使用
    # fetcher.save_papers_for_cleaner(papers, task_id="test_task")

    # show the first paper
    print(papers[0])