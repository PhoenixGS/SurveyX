#!/usr/bin/env python3
"""
更新 log.log 和 EVALUATION_EXAMPLES.md 中的标题为人类真实标题
"""
import re
from pathlib import Path

# 主题名称到人类真实标题的映射
TITLE_MAPPING = {
    "LLMs for Recommendation": "A Survey on Large Language Models for Recommendation",
    "Evaluation of LLMs": "A Survey on Evaluation of Large Language Models",
    "Hallucination in LLMs": "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Qestions",
    "Chain of Thought": "A Survey of Chain of Thought Reasoning: Advances, Frontiers and Future",
    "In-context Learning": "A Survey on In-context Learning",
    "Instruction Tuning for LLMs": "Instruction Tuning for Large Language Models: A Survey",
    "LLMs-based Agents": "A Survey on Large Language Model based Autonomous Agents",
    "Alignment of LLMs": "Aligning Large Language Models with Human: A Survey",
    "Bias and Fairness in LLMs": "Bias and Fairness in Large Language Models: A Survey",
    "Explainability for LLMs": "Explainability for Large Language Models: A Survey",
    "Acceleration for LLMs": "A Survey on Model Compression and Acceleration for Pretrained Language Models",
    "Domain Specialization of LLMs": "Domain Specialization as the Key to Make Large Language Models Disruptive: A Comprehensive Survey",
    "LLMs for Information Retrieval": "Large Language Models for Information Retrieval: A Survey",
    "LLMs for Software Engineering": "Large Language Models for Sofware Engineering: A Systematic Literature Review",
    "LLMs in Medicine": "A Survey of Large Language Models in Medicine: Progress, Application, and Challenge",
    "Large Multi-Modal Language Models": "Large-scale Multi-Modal Pre-trained Models: A Comprehensive Survey",
    "ChatGPT": "Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond",
    "Challenges of LLMs in Education": "Practical and Ethical Challenges of Large Language Models in Education: A Systematic Scoping Review",
}

def update_log_file():
    """更新 log.log 文件"""
    log_file = Path("log.log")
    content = log_file.read_text(encoding='utf-8')
    
    # 替换每个主题的标题
    for topic, real_title in TITLE_MAPPING.items():
        # 匹配 --title "..." 模式
        pattern = rf'(python tasks/offline_run\.py[^\n]*--title ")[^"]+(".*?{re.escape(topic)}[^\n]*)'
        replacement = rf'\1{real_title}\2'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    log_file.write_text(content, encoding='utf-8')
    print(f"✓ 已更新 log.log")

def update_evaluation_examples():
    """更新 EVALUATION_EXAMPLES.md 文件"""
    md_file = Path("EVALUATION_EXAMPLES.md")
    content = md_file.read_text(encoding='utf-8')
    
    # 替换每个主题的标题
    for topic, real_title in TITLE_MAPPING.items():
        # 匹配 Markdown 代码块中的 --title "..."
        pattern = rf'(--title ")[^"]+(".*?{re.escape(topic)}[^\n]*)'
        replacement = rf'\1{real_title}\2'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    md_file.write_text(content, encoding='utf-8')
    print(f"✓ 已更新 EVALUATION_EXAMPLES.md")

if __name__ == "__main__":
    print("开始更新标题...")
    update_log_file()
    update_evaluation_examples()
    print("\n✅ 所有标题已更新为人类真实标题！")

