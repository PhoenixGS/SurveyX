#!/usr/bin/env python3
"""
测试设备参数解析功能
"""
import sys
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent
sys.path.insert(0, str(BASE_DIR))

from src.modules.preprocessor.utils import parse_arguments_for_offline
from src.configs.config import set_device_config, get_device_config

def test_device_args():
    """测试设备参数解析"""
    print("=" * 60)
    print("测试设备参数解析功能")
    print("=" * 60)
    
    # 模拟不同的命令行参数
    test_cases = [
        # (args_list, description)
        (["--title", "Test", "--key_words", "test", "--ref_path", "refs"], "默认参数（auto）"),
        (["--title", "Test", "--key_words", "test", "--ref_path", "refs", "--device", "cpu"], "强制使用CPU"),
        (["--title", "Test", "--key_words", "test", "--ref_path", "refs", "--device", "cuda"], "强制使用CUDA"),
        (["--title", "Test", "--key_words", "test", "--ref_path", "refs", "--device", "cuda", "--gpu_ids", "0"], "使用GPU 0"),
        (["--title", "Test", "--key_words", "test", "--ref_path", "refs", "--device", "cuda", "--gpu_ids", "1"], "使用GPU 1"),
        (["--title", "Test", "--key_words", "test", "--ref_path", "refs", "--device", "auto", "--gpu_ids", "2"], "自动检测，指定GPU 2"),
    ]
    
    for i, (args_list, description) in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {description}")
        print(f"  参数: {' '.join(args_list)}")
        
        # 保存原始sys.argv
        original_argv = sys.argv.copy()
        try:
            # 设置测试参数
            sys.argv = ["test_device_args.py"] + args_list
            args = parse_arguments_for_offline()
            
            print(f"  解析结果:")
            print(f"    title: {args.title}")
            print(f"    key_words: {args.key_words}")
            print(f"    ref_path: {args.ref_path}")
            print(f"    device: {args.device}")
            print(f"    gpu_ids: {args.gpu_ids}")
            
            # 测试设备配置设置
            set_device_config(device=args.device, gpu_ids=args.gpu_ids)
            try:
                device_str = get_device_config()
                print(f"    实际设备: {device_str}")
            except Exception as e:
                print(f"    设备配置错误: {e}")
                
        except Exception as e:
            print(f"  错误: {e}")
        finally:
            # 恢复原始sys.argv
            sys.argv = original_argv
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_device_args()

