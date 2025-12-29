#!/usr/bin/env python3
"""
测试 Paratera API 配置
"""
import sys
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent
sys.path.insert(0, str(BASE_DIR))

from src.configs.config import REMOTE_URL, TOKEN, DEFAULT_CHATAGENT_MODEL, ADVANCED_CHATAGENT_MODEL
from src.models.LLM import ChatAgent

def test_paratera_config():
    """测试 Paratera API 配置"""
    print("=" * 60)
    print("Paratera API 配置测试")
    print("=" * 60)
    
    print(f"\n📋 配置信息:")
    print(f"  API URL: {REMOTE_URL}")
    print(f"  API Key: {TOKEN[:20]}...{TOKEN[-10:] if len(TOKEN) > 30 else '***'}")
    print(f"  默认模型: {DEFAULT_CHATAGENT_MODEL}")
    print(f"  高级模型: {ADVANCED_CHATAGENT_MODEL}")
    
    print(f"\n🧪 测试 API 连接...")
    try:
        chat_agent = ChatAgent()
        
        # 简单测试
        test_prompt = "你好，请用一句话介绍你自己。"
        print(f"\n发送测试请求...")
        print(f"  提示: {test_prompt}")
        
        response = chat_agent.remote_chat(
            test_prompt,
            model=DEFAULT_CHATAGENT_MODEL
        )
        
        print(f"\n✅ API 连接成功！")
        print(f"  响应: {response[:200]}...")
        
        # 测试高级模型
        print(f"\n🧪 测试高级模型 ({ADVANCED_CHATAGENT_MODEL})...")
        response_advanced = chat_agent.remote_chat(
            "请用一句话说明什么是学术综述。",
            model=ADVANCED_CHATAGENT_MODEL
        )
        print(f"✅ 高级模型测试成功！")
        print(f"  响应: {response_advanced[:200]}...")
        
        print(f"\n" + "=" * 60)
        print("✅ 所有测试通过！配置正确。")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败:")
        print(f"  错误: {e}")
        print(f"\n💡 故障排查建议:")
        print(f"  1. 检查 API Key 是否正确")
        print(f"  2. 检查 API URL 是否正确")
        print(f"  3. 检查网络连接")
        print(f"  4. 检查模型名称是否正确（注意大小写）")
        print(f"  5. 查看 Paratera 平台确认 API 状态")
        return False
    
    return True

if __name__ == "__main__":
    success = test_paratera_config()
    sys.exit(0 if success else 1)

