import sys
import argparse
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.models.generator import ContentGenerator, MultiAgentGenerator
from src.models.LLM import ChatAgent
from src.configs.logger import get_logger

logger = get_logger("tasks.workflow.04_gen_content")


def parse_arguments():
    """Parse command line arguments for content generation."""
    parser = argparse.ArgumentParser(description="Generate survey content.")
    parser.add_argument(
        "--task_id", 
        type=str, 
        required=True, 
        help="The ID of the task"
    )
    parser.add_argument(
        "--quality_mode",
        type=str,
        default="default",
        choices=["default", "high"],
        help="Quality mode: 'default' for standard generation, 'high' for Multi-Agent Writer-Critic verification",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        default=False,
        help="Enable Chain of Thought (CoT) reasoning for better logical structure",
    )
    return parser.parse_args()


# python tasks/workflow/04_gen_content.py --task_id 2024-11-30-0022_atten
# python tasks/workflow/04_gen_content.py --task_id 2024-11-30-0022_atten --quality_mode high
# python tasks/workflow/04_gen_content.py --task_id 2024-11-30-0022_atten --quality_mode high --cot
if __name__ == "__main__":
    args = parse_arguments()
    
    cot_info = " with CoT" if args.cot else ""
    
    if args.quality_mode == "high":
        logger.info("=" * 60)
        logger.info(f"Using HIGH QUALITY mode with Multi-Agent Writer-Critic loop{cot_info}")
        if args.cot:
            logger.info("CoT enabled: Planner Agent will generate writing plans")
        logger.info("=" * 60)
        content_generator = MultiAgentGenerator(args.task_id, use_cot=args.cot)
    else:
        logger.info(f"Using DEFAULT mode with standard generation{cot_info}")
        if args.cot:
            logger.info("CoT enabled: Self-reflection thinking before writing")
        content_generator = ContentGenerator(args.task_id, use_cot=args.cot)
    
    content_generator.run()
