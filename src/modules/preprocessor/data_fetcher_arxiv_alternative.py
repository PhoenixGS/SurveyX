"""
使用 arxiv 包实现的替代方案示例
这个文件展示了如何使用 pip 的 arxiv 包来替代内部 API
"""
import arxiv
import time
from typing import List, Dict
from collections import Counter

from src.configs.logger import get_logger

logger = get_logger("src.modules.preprocessor.DataFetcherArxiv")


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
    
    def _convert_arxiv_result_to_dict(self, result: arxiv.Result) -> Dict:
        """
        将 arxiv.Result 转换为与原始格式兼容的字典
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
            "md_text": "",  # 需要从 PDF 或 LaTeX 源码获取
            "reference": [],  # 需要从 PDF 解析
            "image": [],  # 需要从 PDF 提取
        }
        return paper
    
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

