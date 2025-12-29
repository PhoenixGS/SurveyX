# Paratera API 配置说明

## 📋 配置概览

已成功配置 Paratera API，使用以下设置：

### API 配置
- **API URL**: `https://llmapi.paratera.com/v1/chat/completions`
- **API Key**: `sk-eDlafAhDK-FfG08GHEyCfw` (已配置在 `config.py` 中)

### 模型选择

根据 SurveyX 任务需求，已选择以下模型：

#### 1. 默认模型（快速任务）
- **模型名称**: `GLM-4-Flash`
- **用途**: 常规对话、快速响应
- **特点**: 速度快，成本低

#### 2. 高级模型（高质量任务）
- **模型名称**: `GLM-4-Plus`
- **用途**: 
  - 生成大纲（Outlines）
  - 生成内容（Content）
  - 后处理优化（Post-refinement）
  - 表格生成
  - 图表描述
- **特点**: 质量高，适合复杂任务

#### 3. 其他可用模型

Paratera API 提供了丰富的模型选择，可以根据需要调整：

**GLM 系列**:
- `GLM-4-Flash` / `GLM-4-FlashX`: 快速版本
- `GLM-4-Plus`: 高质量版本
- `GLM-4.5` / `GLM-4.5-Air` / `GLM-4.5-Flash`: 最新版本系列
- `GLM-4-Long`: 长文本处理
- `GLM-4V-Flash` / `GLM-4V-Plus`: 视觉理解模型
- `GLM-Z1` 系列: 最新 Z1 系列

**Qwen 系列**:
- `Qwen3-Next-80B-A3B-Instruct`: 大模型指令版本
- `Qwen3-235B-A22B-Instruct`: 超大模型
- `Qwen2.5-72B-Instruct`: 2.5 系列

**DeepSeek 系列**:
- `DeepSeek-V3.2-Instruct`: 高质量指令模型
- `DeepSeek-V3.2-Thinking`: 思维链模型
- `DeepSeek-R1`: 推理模型

## 🔧 如何修改配置

### 方法 1: 修改配置文件（推荐）

编辑 `src/configs/config.py`:

```python
# 修改默认模型
DEFAULT_CHATAGENT_MODEL = "GLM-4-Flash"  # 改为你想要的模型

# 修改高级模型
ADVANCED_CHATAGENT_MODEL = "GLM-4-Plus"  # 改为你想要的模型
```

### 方法 2: 使用环境变量

```bash
# 设置 API URL（如果需要）
export LLM_API_URL="https://llmapi.paratera.com/v1/chat/completions"

# 设置 API Key（如果需要更换）
export PARATERA_API_KEY="your-new-api-key"

# 设置默认模型
export LLM_MODEL="GLM-4-Flash"

# 设置高级模型
export ADVANCED_LLM_MODEL="GLM-4.5"
```

### 方法 3: 在代码中临时指定

```python
from src.models.LLM import ChatAgent

chat_agent = ChatAgent()
# 使用特定模型
response = chat_agent.remote_chat(
    prompt, 
    model="GLM-4.5"  # 直接指定模型
)
```

## 💰 定价信息

所有模型的定价信息已配置在 `src/configs/LLM.yaml` 中。

**注意**: 
- 当前定价是示例值，请根据 Paratera 实际定价调整
- 您提到有免费额度，超出免费额度后按实际计费
- 建议定期检查 Paratera 平台的定价更新

### 当前定价示例（每 100 万 tokens）:

- `GLM-4-Flash`: $0.01 (输入/输出)
- `GLM-4-Plus`: $0.1 (输入/输出)
- `GLM-4.5`: $0.15 (输入/输出)
- `GLM-4.5-Flash`: $0.02 (输入/输出)

## 📊 模型选择建议

### 根据任务类型选择：

1. **快速批量处理** (如论文映射、初步筛选)
   - 推荐: `GLM-4-Flash` 或 `GLM-4-FlashX`
   - 原因: 速度快，成本低

2. **高质量生成** (如大纲生成、内容生成)
   - 推荐: `GLM-4-Plus` 或 `GLM-4.5`
   - 原因: 质量高，适合复杂任务

3. **长文本处理**
   - 推荐: `GLM-4-Long` 或 `Qwen-Long`
   - 原因: 支持更长上下文

4. **视觉理解任务**
   - 推荐: `GLM-4V-Flash` 或 `GLM-4V-Plus`
   - 原因: 支持图像理解

5. **推理任务**
   - 推荐: `DeepSeek-V3.2-Thinking` 或 `DeepSeek-R1`
   - 原因: 擅长推理和思维链

## 🚀 使用示例

### 基本使用

```bash
# 运行离线生成（使用默认配置）
python tasks/offline_run.py \
    --title "Your Survey Title" \
    --key_words "keyword1, keyword2" \
    --ref_path "path/to/references"
```

### 使用环境变量覆盖

```bash
# 使用 GLM-4.5 作为高级模型
export ADVANCED_LLM_MODEL="GLM-4.5"

python tasks/offline_run.py \
    --title "Your Survey Title" \
    --key_words "keyword1, keyword2" \
    --ref_path "path/to/references"
```

## ⚠️ 注意事项

1. **API Key 安全**: 
   - 当前 API Key 已硬编码在 `config.py` 中
   - 建议使用环境变量或配置文件（不提交到 Git）

2. **免费额度**:
   - 注意监控使用量，避免超出免费额度
   - 可以在 Paratera 平台查看使用情况

3. **模型可用性**:
   - 某些模型可能在某些时间段不可用
   - 如果遇到模型不可用，可以切换到备用模型

4. **定价更新**:
   - 定期检查 `LLM.yaml` 中的定价是否准确
   - 根据实际使用情况调整定价信息

## 📝 配置文件位置

- **主配置**: `src/configs/config.py`
- **定价配置**: `src/configs/LLM.yaml`
- **日志**: `outputs/<task_id>/logs/`

## 🔍 故障排查

如果遇到问题：

1. **检查 API Key**: 确认 API Key 是否正确
2. **检查 API URL**: 确认 URL 格式是否正确
3. **检查模型名称**: 确认模型名称拼写正确（注意大小写）
4. **查看日志**: 检查 `outputs/<task_id>/logs/` 中的错误日志
5. **测试连接**: 可以先用简单的 API 调用测试连接

## 📞 支持

如有问题，可以：
- 查看 Paratera 平台文档
- 检查代码中的日志输出
- 查看 `outputs/<task_id>/logs/` 中的详细日志

