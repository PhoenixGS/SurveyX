# Quality Mode 和 CoT 使用说明

## 概述

本文档说明 `quality_mode` 和 `use_cot` 参数在哪些阶段生效。

## 各阶段使用的模型

### 1. 数据预处理阶段（01_preprocess / DataCleaner）

**Attribute Tree 生成** (`DataCleaner.get_attri()`):
- ✅ **使用的模型**: `DEFAULT_CHATAGENT_MODEL` (默认: `GLM-4-Flash`)
- ❌ **不受 `quality_mode` 影响**: 即使在 `high` 模式下也使用默认模型
- ❌ **不支持 CoT**: 不涉及 CoT 功能

**原因**: Attribute tree 生成是预处理步骤，需要快速处理大量论文，使用快速模型即可。

### 2. 大纲生成阶段（02_gen_outline / OutlinesGenerator）

- ✅ **使用的模型**: `DEFAULT_CHATAGENT_MODEL` (默认: `GLM-4-Flash`)
- ❌ **不受 `quality_mode` 影响**
- ❌ **不支持 CoT**

### 3. 内容生成阶段（04_gen_content / ContentGenerator / MultiAgentGenerator）

**这是唯一使用 `quality_mode` 和 `use_cot` 的阶段！**

#### Default 模式 (`quality_mode="default"`):
- 使用 `ContentGenerator`
- 模型: `DEFAULT_CHATAGENT_MODEL` (默认: `GLM-4-Flash`)
- CoT 支持: ✅ 支持 `use_cot` 参数
  - 如果 `use_cot=True`: 使用 CoT prompt 模板，模型会先思考再生成内容
  - 如果 `use_cot=False`: 使用标准 prompt 模板

#### High 模式 (`quality_mode="high"`):
- 使用 `MultiAgentGenerator`
- Writer Agent 模型: `HIGH_QUALITY_WRITER_MODEL` (默认: `DeepSeek-V3.2`)
- Critic Agent 模型: `HIGH_QUALITY_CRITIC_MODEL` (默认: `DeepSeek-R1`)
- Planner Agent 模型: `HIGH_QUALITY_PLANNER_MODEL` (仅在 `use_cot=True` 时使用)
- CoT 支持: ✅ 支持 `use_cot` 参数
  - 如果 `use_cot=True`: 增加 Planner Agent 生成写作计划
  - 如果 `use_cot=False`: 仅使用 Writer-Critic 循环

### 4. 后处理阶段（05_post_refine / PostRefiner）

- ✅ **使用的模型**: `DEFAULT_CHATAGENT_MODEL` (默认: `GLM-4-Flash`)
- ❌ **不受 `quality_mode` 影响**: 即使在 `high` 模式下也使用默认模型
- ❌ **不支持 CoT**: 不涉及 CoT 功能

**原因**: Post refine 主要是基于规则的润色和 RAG 检索，不需要高质量模型。

## 总结

| 阶段 | 使用的模型 | 受 quality_mode 影响？ | 支持 CoT？ |
|------|-----------|----------------------|-----------|
| 01 预处理 (Attribute Tree) | `DEFAULT_CHATAGENT_MODEL` | ❌ 否 | ❌ 否 |
| 02 生成大纲 | `DEFAULT_CHATAGENT_MODEL` | ❌ 否 | ❌ 否 |
| **04 生成内容** | **根据 quality_mode 选择** | **✅ 是** | **✅ 是** |
| 05 后处理润色 | `DEFAULT_CHATAGENT_MODEL` | ❌ 否 | ❌ 否 |

## 使用建议

1. **Attribute Tree 生成**: 使用默认模型即可，因为需要处理大量论文，速度优先
2. **内容生成**: 
   - 如果追求质量，使用 `--quality_mode high`
   - 如果需要更好的逻辑结构，使用 `--cot`
   - 两者可以同时使用: `--quality_mode high --cot`
3. **Post Refine**: 使用默认模型即可，主要是规则和检索

## 配置示例

```bash
# 默认模式，不使用 CoT
python tasks/offline_run.py --title "..." --key_words "..." --ref_path "..."

# 高质量模式，不使用 CoT
python tasks/offline_run.py --title "..." --key_words "..." --ref_path "..." --quality_mode high

# 默认模式，使用 CoT
python tasks/offline_run.py --title "..." --key_words "..." --ref_path "..." --cot

# 高质量模式，使用 CoT（最完整配置）
python tasks/offline_run.py --title "..." --key_words "..." --ref_path "..." --quality_mode high --cot
```

