import subprocess
import sys
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.configs.config import BASE_DIR
from src.configs.logger import get_logger
from src.models.generator import (ContentGenerator, LatexGenerator,
                                  OutlinesGenerator, MultiAgentGenerator)
from src.models.LLM import ChatAgent
from src.models.post_refine import PostRefiner
from src.modules.preprocessor.preprocessor import single_preprocessing
from src.modules.preprocessor.utils import parse_arguments_for_preprocessor

logger = get_logger("tasks.full_run")

def check_latexmk_installed():
    try:
        # Try running the latexmk command with the --version option
        _ = subprocess.run(['latexmk', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.debug("latexmk is installed.")
        return True
    except subprocess.CalledProcessError as e:
        logger.debug("latexmk is not installed.")
        return False
    except FileNotFoundError:
        logger.debug("latexmk is not installed.")
        return False

def generate_single_survey(task_id: str, chat_agent: ChatAgent=None, quality_mode: str = "default", use_cot: bool = False):
    """
    Generate a single survey.
    
    Args:
        task_id: Unique identifier for this generation task
        chat_agent: Optional ChatAgent instance
        quality_mode: "default" for standard generation, "high" for Multi-Agent verification
        use_cot: Enable Chain of Thought reasoning for better logical structure
    """
    if chat_agent is None:
        chat_agent = ChatAgent()

    # generate outlines
    outline_generator = OutlinesGenerator(task_id)
    outline_generator.run()

    # generate survey - select generator based on quality mode
    cot_info = " with CoT" if use_cot else ""
    if quality_mode == "high":
        logger.info("=" * 60)
        logger.info(f"Using HIGH QUALITY mode with Multi-Agent Writer-Critic loop{cot_info}")
        if use_cot:
            logger.info("CoT enabled: Planner Agent will generate writing plans")
        logger.info("=" * 60)
        content_generator = MultiAgentGenerator(task_id=task_id, use_cot=use_cot)
    else:
        logger.info(f"Using DEFAULT mode with standard generation{cot_info}")
        if use_cot:
            logger.info("CoT enabled: Self-reflection thinking before writing")
        content_generator = ContentGenerator(task_id=task_id, use_cot=use_cot)
    
    content_generator.run()

    # post refine
    post_refiner = PostRefiner(task_id=task_id, chat_agent=chat_agent)
    post_refiner.run()

    # generate full survey
    latex_generator = LatexGenerator(task_id=task_id)
    latex_generator.generate_full_survey()

    # compile latex
    if check_latexmk_installed():
        logger.info(f"Start compiling with latexmk.")
        latex_generator.compile_single_survey()
    else:
        logger.error(f"Compiling failed, as there is no latexmk installed in this machine.")


if __name__ == "__main__":
    args = parse_arguments_for_preprocessor()
    task_id = single_preprocessing(args)
    generate_single_survey(task_id=task_id)
