#!/usr/bin/env python3
"""
检查 Token 使用统计
用法: python scripts/check_token_stats.py [task_id]
"""
import json
import sys
from pathlib import Path

def check_token_stats(task_id=None):
    """检查指定任务或最新任务的 Token 统计"""
    base_dir = Path(__file__).parent.parent
    outputs_dir = base_dir / "outputs"
    
    if not outputs_dir.exists():
        print("错误: outputs 目录不存在")
        return
    
    # 如果没有指定 task_id，使用最新的
    if task_id is None:
        task_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], 
                          key=lambda x: x.stat().st_mtime, reverse=True)
        if not task_dirs:
            print("错误: 没有找到任何任务目录")
            return
        task_id = task_dirs[0].name
        print(f"使用最新任务: {task_id}\n")
    
    token_file = outputs_dir / task_id / "metrics" / "token_monitor.json"
    
    if not token_file.exists():
        print(f"错误: Token 统计文件不存在: {token_file}")
        print("\n可能的原因:")
        print("1. 任务尚未完成")
        print("2. TokenMonitor 未正确初始化")
        print("3. 任务运行出错")
        return
    
    # 读取统计文件
    try:
        with open(token_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取统计文件: {e}")
        return
    
    # 显示统计信息
    print("=" * 60)
    print(f"Token 使用统计 - {task_id}")
    print("=" * 60)
    print()
    
    total_input = 0
    total_output = 0
    total_cost = 0.0
    
    for label, models in stats.items():
        print(f"📊 {label}:")
        print("-" * 60)
        
        for model, data in models.items():
            input_tokens = data.get("input_tokens", 0)
            output_tokens = data.get("output_tokens", 0)
            cost = data.get("total_cost", 0.0)
            
            total_input += input_tokens
            total_output += output_tokens
            total_cost += cost
            
            print(f"  模型: {model}")
            print(f"    输入 Token: {input_tokens:,}")
            print(f"    输出 Token: {output_tokens:,}")
            print(f"    总 Token: {input_tokens + output_tokens:,}")
            print(f"    费用: ${cost:.6f}")
            print()
    
    print("=" * 60)
    print("总计:")
    print(f"  总输入 Token: {total_input:,}")
    print(f"  总输出 Token: {total_output:,}")
    print(f"  总 Token: {total_input + total_output:,}")
    print(f"  总费用: ${total_cost:.6f}")
    print("=" * 60)

if __name__ == "__main__":
    task_id = sys.argv[1] if len(sys.argv) > 1 else None
    check_token_stats(task_id)

