#!/usr/bin/env python3
"""
将 eval/data/ref 目录中的 JSON 文件转换为 .md 文件
用于准备参考文档
"""
import json
import os
from pathlib import Path

def convert_json_to_md(json_file: Path, output_dir: Path):
    """将单个 JSON 文件转换为 .md 文件"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取 md_text
        if 'md_text' not in data:
            print(f"警告: {json_file} 中没有 md_text 字段，跳过")
            return False
        
        md_text = data['md_text']
        
        # 生成输出文件名（使用标题或原文件名）
        if 'title' in data and data['title']:
            # 使用标题作为文件名，清理特殊字符
            safe_title = "".join(c for c in data['title'] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_')[:100]  # 限制长度
            output_file = output_dir / f"{safe_title}.md"
        else:
            # 使用原文件名
            output_file = output_dir / f"{json_file.stem}.md"
        
        # 写入 .md 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_text)
        
        print(f"✓ 转换: {json_file.name} -> {output_file.name}")
        return True
    except Exception as e:
        print(f"✗ 错误: 转换 {json_file} 时出错: {e}")
        return False

def convert_ref_dir_to_md(ref_dir: str, output_dir: str, topic: str = None):
    """
    将 ref 目录中的 JSON 文件转换为 .md 文件
    
    Args:
        ref_dir: ref 目录路径，例如 "eval/data/ref/LLMs for Recommendation"
        output_dir: 输出目录路径
        topic: 主题名称（可选），如果指定则只处理该主题
    """
    ref_path = Path(ref_dir)
    output_path = Path(output_dir)
    
    if not ref_path.exists():
        print(f"错误: 目录不存在: {ref_dir}")
        return
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 如果指定了主题，只处理该主题目录
    if topic:
        topic_dir = ref_path / topic
        if topic_dir.exists() and topic_dir.is_dir():
            json_files = list(topic_dir.glob("*.json"))
            print(f"处理主题: {topic}, 找到 {len(json_files)} 个 JSON 文件")
            for json_file in json_files:
                convert_json_to_md(json_file, output_path)
        else:
            print(f"错误: 主题目录不存在: {topic_dir}")
    else:
        # 处理所有子目录
        topic_dirs = [d for d in ref_path.iterdir() if d.is_dir()]
        print(f"找到 {len(topic_dirs)} 个主题目录")
        
        for topic_dir in topic_dirs:
            json_files = list(topic_dir.glob("*.json"))
            print(f"\n处理主题: {topic_dir.name}, 找到 {len(json_files)} 个 JSON 文件")
            for json_file in json_files:
                convert_json_to_md(json_file, output_path)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法:")
        print("  python scripts/convert_ref_to_md.py <ref_dir> <output_dir> [topic]")
        print("\n示例:")
        print("  # 转换单个主题")
        print("  python scripts/convert_ref_to_md.py eval/data/ref references 'LLMs for Recommendation'")
        print("\n  # 转换所有主题")
        print("  python scripts/convert_ref_to_md.py eval/data/ref references")
        sys.exit(1)
    
    ref_dir = sys.argv[1]
    output_dir = sys.argv[2]
    topic = sys.argv[3] if len(sys.argv) > 3 else None
    
    convert_ref_dir_to_md(ref_dir, output_dir, topic)
    print(f"\n✓ 转换完成！输出目录: {output_dir}")

