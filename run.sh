# ============================================================================
# SurveyX - Offline Run (Entire Pipeline)
# ============================================================================

# Basic usage (default mode):
python tasks/offline_run.py --title "Controllable Text Generation for Large Language Models: A Survey" --key_words "controlled text generation, text generation, large language model, LLM" --ref_path "dir/path/to/your/references-markdowns"

# ============================================================================
# HIGH QUALITY MODE (Multi-Agent Writer-Critic Verification)
# ============================================================================
# Use --quality_mode high to enable Writer-Critic loop for reduced hallucinations
# Note: This costs ~2-3x more API calls but significantly improves accuracy

# High quality mode (recommended for final output):
python tasks/offline_run.py --title "Your Title" --key_words "keywords" --ref_path "references" --quality_mode high

# High quality mode with GPU:
python tasks/offline_run.py --title "Your Title" --key_words "keywords" --ref_path "references" --quality_mode high --device cuda --gpu_ids "0"

# ============================================================================
# Device Selection Options
# ============================================================================

# Use CPU explicitly:
python tasks/offline_run.py --title "Your Title" --key_words "keywords" --ref_path "references" --device cpu

# Use GPU (auto-detect, use first available GPU):
python tasks/offline_run.py --title "Your Title" --key_words "keywords" --ref_path "references" --device cuda

# Use specific GPU (e.g., GPU 0):
python tasks/offline_run.py --title "Your Title" --key_words "keywords" --ref_path "references" --device cuda --gpu_ids "0"

# Use specific GPU (e.g., GPU 1):
python tasks/offline_run.py --title "Your Title" --key_words "keywords" --ref_path "references" --device cuda --gpu_ids "1"

# Auto-detect (default, will use GPU if available):
python tasks/offline_run.py --title "Your Title" --key_words "keywords" --ref_path "references" --device auto

# ============================================================================
# Workflow (Step by Step, for debugging)
# ============================================================================

python tasks/workflow/02_clean_data.py --title "Controllable Text Generation for Large Language Models: A Survey" --key_words "controlled text generation, text generation, large language model, LLM" --ref_path "../refs"
task_id="xxx" # You can check the task_id in the outputs/<task_id>/tmp_config.json of the previous command
python tasks/workflow/03_gen_outlines.py  --task_id $task_id

# Default mode:
python tasks/workflow/04_gen_content.py  --task_id $task_id
# OR High quality mode:
python tasks/workflow/04_gen_content.py  --task_id $task_id --quality_mode high

python tasks/workflow/05_post_refine.py  --task_id $task_id
python tasks/workflow/06_gen_latex.py  --task_id $task_id
