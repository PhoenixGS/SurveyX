#!/usr/bin/env python3
"""
测试多 API Key 随机选择功能
"""
import sys
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.configs.config import PARATERA_API_KEYS
from src.models.LLM import ChatAgent

def test_multi_api_keys():
    print("=" * 60)
    print("多 API Key 随机选择测试")
    print("=" * 60)
    
    print(f"\n📋 配置的 API Keys ({len(PARATERA_API_KEYS)} 个):")
    for i, key in enumerate(PARATERA_API_KEYS, 1):
        print(f"  {i}. {key[:20]}...{key[-10:]}")
    
    print(f"\n🧪 测试随机选择...")
    # 创建多个 ChatAgent 实例，观察 key 选择
    agents = []
    for i in range(10):
        agent = ChatAgent()
        agents.append(agent)
        print(f"  Agent {i+1}: {agent.token[:20]}...{agent.token[-10:]}")
    
    # 获取使用统计
    stats = ChatAgent.get_key_usage_stats()
    print(f"\n📊 API Key 使用统计:")
    for key, count in stats.items():
        print(f"  {key[:20]}...{key[-10:]}: {count} 次")
    
    print(f"\n✅ 测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_multi_api_keys()

