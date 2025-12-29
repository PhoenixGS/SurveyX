#!/bin/bash
# 批量评估脚本
# 用于批量运行多个 SurveyX 评估任务

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF_BASE_DIR="$BASE_DIR/eval/data/ref"
OUTPUT_BASE_DIR="$BASE_DIR/references"
RESULTS_DIR="$BASE_DIR/evaluation_results"

# 创建结果目录
mkdir -p "$RESULTS_DIR"
mkdir -p "$OUTPUT_BASE_DIR"

# 定义评估任务（主题名称:标题:关键词）
declare -a TASKS=(
    "Evaluation of LLMs:A Survey on Evaluation of Large Language Models:large language model, LLM evaluation, model assessment, benchmark, evaluation metrics"
    "Hallucination in LLMs:A Survey on Hallucination in Large Language Models:large language model, hallucination, factuality, misinformation, model reliability"
    "Chain of Thought:A Survey on Chain of Thought Reasoning in Large Language Models:chain of thought, reasoning, step-by-step reasoning, CoT, reasoning chain"
    "In-context Learning:A Survey on In-Context Learning for Large Language Models:in-context learning, few-shot learning, prompt learning, ICL, context learning"
    "Instruction Tuning for LLMs:A Survey on Instruction Tuning for Large Language Models:instruction tuning, fine-tuning, supervised fine-tuning, instruction following, model alignment"
)

# 计数器
TOTAL=${#TASKS[@]}
SUCCESS=0
FAILED=0

echo "=========================================="
echo "SurveyX 批量评估脚本"
echo "=========================================="
echo "总共任务数: $TOTAL"
echo "开始时间: $(date)"
echo "=========================================="
echo ""

# 遍历任务
for i in "${!TASKS[@]}"; do
    IFS=':' read -r TOPIC TITLE KEYWORDS <<< "${TASKS[$i]}"
    TASK_NUM=$((i + 1))
    
    echo ""
    echo "=========================================="
    echo "[$TASK_NUM/$TOTAL] 处理任务: $TOPIC"
    echo "=========================================="
    
    # 生成输出目录名（替换空格为下划线）
    OUTPUT_DIR_NAME=$(echo "$TOPIC" | tr ' ' '_')
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$OUTPUT_DIR_NAME"
    REF_DIR="$REF_BASE_DIR/$TOPIC"
    
    # 检查参考目录是否存在
    if [ ! -d "$REF_DIR" ]; then
        echo -e "${RED}✗ 错误: 参考目录不存在: $REF_DIR${NC}"
        ((FAILED++))
        continue
    fi
    
    # 步骤 1: 转换参考文档
    echo "步骤 1: 转换参考文档..."
    if python "$BASE_DIR/scripts/convert_ref_to_md.py" "$REF_BASE_DIR" "$OUTPUT_DIR" "$TOPIC"; then
        echo -e "${GREEN}✓ 参考文档转换成功${NC}"
    else
        echo -e "${RED}✗ 参考文档转换失败${NC}"
        ((FAILED++))
        continue
    fi
    
    # 步骤 2: 运行生成任务
    echo "步骤 2: 运行生成任务..."
    echo "  标题: $TITLE"
    echo "  关键词: $KEYWORDS"
    echo "  参考目录: $OUTPUT_DIR"
    
    if python "$BASE_DIR/tasks/offline_run.py" \
        --title "$TITLE" \
        --key_words "$KEYWORDS" \
        --ref_path "$OUTPUT_DIR"; then
        echo -e "${GREEN}✓ 生成任务成功${NC}"
        ((SUCCESS++))
        
        # 查找最新的 task_id
        LATEST_TASK=$(ls -t "$BASE_DIR/outputs" | head -1)
        if [ -n "$LATEST_TASK" ]; then
            echo "  结果保存在: outputs/$LATEST_TASK/"
            # 创建结果链接
            ln -sf "$BASE_DIR/outputs/$LATEST_TASK" "$RESULTS_DIR/$OUTPUT_DIR_NAME" 2>/dev/null || true
        fi
    else
        echo -e "${RED}✗ 生成任务失败${NC}"
        ((FAILED++))
    fi
    
    echo ""
done

# 总结
echo ""
echo "=========================================="
echo "评估完成"
echo "=========================================="
echo "总任务数: $TOTAL"
echo -e "${GREEN}成功: $SUCCESS${NC}"
echo -e "${RED}失败: $FAILED${NC}"
echo "结束时间: $(date)"
echo "=========================================="
echo ""
echo "结果目录: $RESULTS_DIR"
echo "参考文档目录: $OUTPUT_BASE_DIR"

