#!/usr/bin/env python3
"""
对比人类真实综述标题和 log.log 中的标题
"""
from pathlib import Path
import re

def extract_human_titles():
    """提取人类真实综述的标题"""
    human_dir = Path("eval/data/human")
    titles = {}
    for file in human_dir.glob("*.md"):
        with open(file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line.startswith('#'):
                title = first_line[1:].strip()
                topic = file.stem
                titles[topic] = title
    return titles

def extract_log_titles():
    """提取 log.log 中的标题"""
    log_file = Path("log.log")
    titles = {}
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # 匹配主题和标题
        pattern = r'# (\d+)\. (.+?)\n.*?--title "([^"]+)"'
        matches = re.findall(pattern, content)
        for num, topic, title in matches:
            titles[topic] = title
    return titles

if __name__ == "__main__":
    human_titles = extract_human_titles()
    log_titles = extract_log_titles()
    
    print("=" * 80)
    print("标题对比分析")
    print("=" * 80)
    print()
    
    # 找到匹配的主题
    common_topics = set(human_titles.keys()) & set(log_titles.keys())
    
    print(f"找到 {len(common_topics)} 个共同主题\n")
    
    for topic in sorted(common_topics):
        human_title = human_titles[topic]
        log_title = log_titles[topic]
        match = "✓" if human_title == log_title else "✗"
        
        print(f"{match} {topic}:")
        print(f"  人类真实: {human_title}")
        print(f"  log.log:  {log_title}")
        if human_title != log_title:
            print(f"  差异: {'相似' if human_title.lower() in log_title.lower() or log_title.lower() in human_title.lower() else '不同'}")
        print()

