#!/bin/bash
# 快速评估脚本 - 运行单个任务
# 用法: ./scripts/quick_evaluate.sh <主题名称> [标题] [关键词]

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ $# -lt 1 ]; then
    echo "用法: $0 <主题名称> [标题] [关键词]"
    echo ""
    echo "示例:"
    echo "  $0 'Evaluation of LLMs'"
    echo "  $0 'Hallucination in LLMs' 'A Survey on Hallucination' 'hallucination, factuality'"
    exit 1
fi

TOPIC="$1"
TITLE="${2:-A Survey on ${TOPIC}}"
KEYWORDS="${3:-${TOPIC}, large language model, LLM}"

# 生成输出目录名
OUTPUT_DIR_NAME=$(echo "$TOPIC" | tr ' ' '_')
OUTPUT_DIR="$BASE_DIR/references/$OUTPUT_DIR_NAME"
REF_DIR="$BASE_DIR/eval/data/ref/$TOPIC"

echo "=========================================="
echo "快速评估: $TOPIC"
echo "=========================================="
echo "标题: $TITLE"
echo "关键词: $KEYWORDS"
echo "参考目录: $REF_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查参考目录
if [ ! -d "$REF_DIR" ]; then
    echo "错误: 参考目录不存在: $REF_DIR"
    exit 1
fi

# 步骤 1: 转换参考文档
echo "步骤 1: 转换参考文档..."
python "$BASE_DIR/scripts/convert_ref_to_md.py" "$BASE_DIR/eval/data/ref" "$OUTPUT_DIR" "$TOPIC"

# 步骤 2: 运行生成任务
echo ""
echo "步骤 2: 运行生成任务..."
python "$BASE_DIR/tasks/offline_run.py" \
    --title "$TITLE" \
    --key_words "$KEYWORDS" \
    --ref_path "$OUTPUT_DIR"

echo ""
echo -e "${GREEN}✓ 任务完成！${NC}"

# 查找最新的 task_id
LATEST_TASK=$(ls -t "$BASE_DIR/outputs" 2>/dev/null | head -1)
if [ -n "$LATEST_TASK" ]; then
    echo "结果保存在: outputs/$LATEST_TASK/"
fi

