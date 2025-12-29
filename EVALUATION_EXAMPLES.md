# SurveyX 评估示例

本文档提供了多个可运行的示例命令，用于评估 SurveyX 的效果。

## 📋 使用说明

### 基本流程

1. **转换参考文档**：将 JSON 格式的参考论文转换为 Markdown 格式
2. **运行生成任务**：使用转换后的参考文档生成综述

### 命令格式

```bash
# 步骤 1: 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref <输出目录> "<主题名称>"

# 步骤 2: 运行生成任务
python tasks/offline_run.py \
    --title "<综述标题>" \
    --key_words "<关键词1>, <关键词2>, ..." \
    --ref_path "<参考文档目录>"
```

---

## 🎯 评估示例

### 1. LLMs for Recommendation (已完成)
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/LLMs_for_Recommendation "LLMs for Recommendation"

# 运行生成任务
python tasks/offline_run.py \
    --title "A Survey on Large Language Models for Recommendation" \
    --key_words "large language model, LLM, recommendation system, recommender system, LLM-based recommendation" \
    --ref_path "./references/LLMs_for_Recommendation"
```

### 2. Evaluation of LLMs
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/Evaluation_of_LLMs "Evaluation of LLMs"

# 运行生成任务
python tasks/offline_run.py \
    --title "A Survey on Evaluation of Large Language Models" \
    --key_words "large language model, LLM evaluation, model assessment, benchmark, evaluation metrics" \
    --ref_path "./references/Evaluation_of_LLMs"
```

### 3. Hallucination in LLMs
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/Hallucination_in_LLMs "Hallucination in LLMs"

# 运行生成任务
python tasks/offline_run.py \
    --title "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Qestions" \
    --key_words "large language model, hallucination, factuality, misinformation, model reliability" \
    --ref_path "./references/Hallucination_in_LLMs"
```

### 4. Chain of Thought
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/Chain_of_Thought "Chain of Thought"

# 运行生成任务
python tasks/offline_run.py \
    --title "A Survey of Chain of Thought Reasoning: Advances, Frontiers and Future" \
    --key_words "chain of thought, reasoning, step-by-step reasoning, CoT, reasoning chain" \
    --ref_path "./references/Chain_of_Thought"
```

### 5. In-context Learning
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/In-context_Learning "In-context Learning"

# 运行生成任务
python tasks/offline_run.py \
    --title "A Survey on In-context Learning" \
    --key_words "in-context learning, few-shot learning, prompt learning, ICL, context learning" \
    --ref_path "./references/In-context_Learning"
```

### 6. Instruction Tuning for LLMs
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/Instruction_Tuning "Instruction Tuning for LLMs"

# 运行生成任务
python tasks/offline_run.py \
    --title "Instruction Tuning for Large Language Models: A Survey" \
    --key_words "instruction tuning, fine-tuning, supervised fine-tuning, instruction following, model alignment" \
    --ref_path "./references/Instruction_Tuning"
```

### 7. LLMs-based Agents
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/LLMs-based_Agents "LLMs-based Agents"

# 运行生成任务
python tasks/offline_run.py \
    --title "A Survey on Large Language Model based Autonomous Agents" \
    --key_words "LLM agents, AI agents, autonomous agents, agent framework, multi-agent systems" \
    --ref_path "./references/LLMs-based_Agents"
```

### 8. Alignment of LLMs
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/Alignment_of_LLMs "Alignment of LLMs"

# 运行生成任务
python tasks/offline_run.py \
    --title "Aligning Large Language Models with Human: A Survey" \
    --key_words "LLM alignment, human alignment, RLHF, preference learning, model alignment" \
    --ref_path "./references/Alignment_of_LLMs"
```

### 9. Bias and Fairness in LLMs
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/Bias_and_Fairness "Bias and Fairness in LLMs"

# 运行生成任务
python tasks/offline_run.py \
    --title "Bias and Fairness in Large Language Models: A Survey" \
    --key_words "bias, fairness, algorithmic bias, demographic parity, fairness metrics, bias mitigation" \
    --ref_path "./references/Bias_and_Fairness"
```

### 10. Explainability for LLMs
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/Explainability "Explainability for LLMs"

# 运行生成任务
python tasks/offline_run.py \
    --title "Explainability for Large Language Models: A Survey" \
    --key_words "explainability, interpretability, model explanation, attention visualization, XAI" \
    --ref_path "./references/Explainability"
```

### 11. Acceleration for LLMs
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/Acceleration "Acceleration for LLMs"

# 运行生成任务
python tasks/offline_run.py \
    --title "A Survey on Model Compression and Acceleration for Pretrained Language Models" \
    --key_words "model acceleration, inference speedup, quantization, pruning, model compression" \
    --ref_path "./references/Acceleration"
```

### 12. Domain Specialization of LLMs
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/Domain_Specialization "Domain Specialization of LLMs"

# 运行生成任务
python tasks/offline_run.py \
    --title "Domain Specialization as the Key to Make Large Language Models Disruptive: A Comprehensive Survey" \
    --key_words "domain adaptation, specialized models, domain-specific LLM, fine-tuning, transfer learning" \
    --ref_path "./references/Domain_Specialization"
```

### 13. LLMs for Information Retrieval
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/LLMs_for_IR "LLMs for Information Retrieval"

# 运行生成任务
python tasks/offline_run.py \
    --title "Large Language Models for Information Retrieval: A Survey" \
    --key_words "information retrieval, search, retrieval-augmented generation, RAG, semantic search" \
    --ref_path "./references/LLMs_for_IR"
```

### 14. LLMs for Software Engineering
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/LLMs_for_SE "LLMs for Software Engineering"

# 运行生成任务
python tasks/offline_run.py \
    --title "Large Language Models for Sofware Engineering: A Systematic Literature Review" \
    --key_words "software engineering, code generation, code understanding, program synthesis, software development" \
    --ref_path "./references/LLMs_for_SE"
```

### 15. LLMs in Medicine
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/LLMs_in_Medicine "LLMs in Medicine"

# 运行生成任务
python tasks/offline_run.py \
    --title "A Survey of Large Language Models in Medicine: Progress, Application, and Challenge" \
    --key_words "medical AI, healthcare, clinical NLP, medical diagnosis, health informatics" \
    --ref_path "./references/LLMs_in_Medicine"
```

### 16. Large Multi-Modal Language Models
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/Multi-Modal_LLMs "Large Multi-Modal Language Models"

# 运行生成任务
python tasks/offline_run.py \
    --title "Large-scale Multi-Modal Pre-trained Models: A Comprehensive Survey" \
    --key_words "multimodal learning, vision-language models, CLIP, image-text understanding, multimodal AI" \
    --ref_path "./references/Multi-Modal_LLMs"
```

### 17. ChatGPT
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/ChatGPT "ChatGPT"

# 运行生成任务
python tasks/offline_run.py \
    --title "Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond" \
    --key_words "ChatGPT, GPT, conversational AI, chat models, dialogue systems" \
    --ref_path "./references/ChatGPT"
```

### 18. Challenges of LLMs in Education
```bash
# 转换参考文档
python scripts/convert_ref_to_md.py eval/data/ref references/LLMs_in_Education "Challenges of LLMs in Education"

# 运行生成任务
python tasks/offline_run.py \
    --title "Practical and Ethical Challenges of Large Language Models in Education: A Systematic Scoping Review" \
    --key_words "educational technology, AI in education, learning systems, educational assessment, pedagogy" \
    --ref_path "./references/LLMs_in_Education"
```

---

## 🚀 批量运行脚本

可以使用以下脚本批量运行多个评估任务：

```bash
# 运行批量评估脚本
bash scripts/batch_evaluate.sh
```

---

## 📊 评估结果位置

所有生成的结果保存在 `outputs/<task_id>/` 目录下：

- `survey.pdf`: 最终生成的综述 PDF
- `outlines.json`: 生成的大纲
- `latex/`: LaTeX 源文件
- `tmp/`: 中间文件
- `logs/`: 日志文件
- `metrics/token_monitor.json`: Token 使用统计（**自动生成**）

---

## 💡 评估建议

1. **选择不同规模的主题**：
   - 小规模（<50 papers）：快速测试
   - 中等规模（50-100 papers）：标准评估
   - 大规模（>100 papers）：完整评估

2. **对比不同主题**：
   - 选择 3-5 个不同领域的主题
   - 比较生成质量、结构完整性、引用准确性

3. **检查关键指标**：
   - Token 使用量（**自动统计**，保存在 `outputs/<task_id>/metrics/token_monitor.json`）
   - 输出质量（人工评估）
   - 与参考综述的对比

### 查看 Token 统计

Token 使用量会自动统计，无需手动配置。查看方法：

```bash
# 方法 1: 使用脚本查看（推荐）
python scripts/check_token_stats.py [task_id]

# 方法 2: 直接查看 JSON 文件
cat outputs/<task_id>/metrics/token_monitor.json | python -m json.tool
```

统计文件包含：
- 按任务阶段分类（generate outline, generate content, generate table 等）
- 按模型分类（GLM-4-Flash, GLM-4-Plus 等）
- 每个模型的输入/输出 Token 数量和费用

---

## ⚙️ 使用 GPU 加速

如果使用 GPU，可以添加设备参数：

```bash
python tasks/offline_run.py \
    --title "Your Title" \
    --key_words "keywords" \
    --ref_path "references" \
    --device cuda \
    --gpu_ids "0"
```

---

## 📝 注意事项

1. **确保参考文档已转换**：在运行生成任务前，必须先运行转换脚本
2. **检查 API 配置**：确保 Paratera API 配置正确
3. **监控资源使用**：大规模任务可能需要较长时间和较多 Token
4. **保存结果**：定期备份 `outputs/` 目录中的结果

