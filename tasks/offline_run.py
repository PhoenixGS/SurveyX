import subprocess
import sys
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.configs.config import BASE_DIR, set_device_config
from src.configs.constants import OUTPUT_DIR
from src.configs.logger import get_logger
from src.models.generator import (ContentGenerator, LatexGenerator,
                                  OutlinesGenerator, MultiAgentGenerator)
from src.models.LLM import ChatAgent
from src.models.post_refine import PostRefiner
from src.modules.preprocessor.data_cleaner import DataCleaner
from src.modules.preprocessor.utils import (create_tmp_config,
                                            parse_arguments_for_offline)

logger = get_logger("tasks.offline_run")


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

def offline_generate(task_id: str, ref_path: str, quality_mode: str = "default", use_cot: bool = False):
    """
    Offline survey generation pipeline.
    
    Args:
        task_id: Unique identifier for this generation task
        ref_path: Path to reference papers directory
        quality_mode: "default" for standard generation, "high" for Multi-Agent verification
        use_cot: Enable Chain of Thought reasoning for better logical structure
    """
    # Create a mode marker file in the task directory for easy identification
    # Move this to the very beginning to ensure it's created even if subsequent steps fail
    mode_tag = quality_mode.upper()
    if use_cot:
        mode_tag += "_COT"
    
    task_dir = Path(OUTPUT_DIR) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / f"MODE_{mode_tag}", "w", encoding="utf-8") as f:
        f.write(f"This task was run in {quality_mode} mode with CoT={'True' if use_cot else 'False'}.\n")

    chat_agent = ChatAgent()
    
    # preprocess references
    dc = DataCleaner()
    dc.offline_proc(task_id=task_id, ref_path=ref_path)

    # generate outlines
    outline_generator = OutlinesGenerator(task_id)
    outline_generator.run()

    # generate survey - select generator based on quality mode
    cot_info = " with CoT" if use_cot else ""
    if quality_mode == "high":
        logger.info("=" * 60)
        logger.info(f"Using HIGH QUALITY mode with Multi-Agent Writer-Critic loop{cot_info}")
        logger.info("This will cost ~2-3x more API calls but significantly reduce hallucinations")
        if use_cot:
            logger.info("CoT enabled: Planner Agent will generate writing plans (+65% cost)")
        logger.info("=" * 60)
        content_generator = MultiAgentGenerator(task_id=task_id, use_cot=use_cot)
    else:
        logger.info(f"Using DEFAULT mode with standard single-pass generation{cot_info}")
        if use_cot:
            logger.info("CoT enabled: Self-reflection thinking before writing (+15% cost)")
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
    args = parse_arguments_for_offline()
    
    # Set device configuration from command line arguments
    set_device_config(device=args.device, gpu_ids=args.gpu_ids)
    logger.info(f"Device configuration: device={args.device}, gpu_ids={args.gpu_ids}")
    logger.info(f"Quality mode: {args.quality_mode}, CoT: {args.cot}")
    
    tmp_config = create_tmp_config(args.title, args.key_words)

    topic = tmp_config["topic"]
    task_id = tmp_config["task_id"]
    
    offline_generate(task_id=task_id, ref_path=args.ref_path, quality_mode=args.quality_mode, use_cot=args.cot)
