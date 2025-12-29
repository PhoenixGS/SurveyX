# Reference.bib 生成机制说明

## 生成时机

**是的，`reference.bib` 确实在生成 outlines 之前就创建了！**

### 执行流程

```python
# tasks/offline_run.py
def offline_generate(...):
    # 1. 预处理阶段（DataCleaner.offline_proc）
    dc = DataCleaner()
    dc.offline_proc(task_id=task_id, ref_path=ref_path)
    #    ↓ 在这个方法内部：
    #    - complete_title()          # 提取标题
    #    - complete_abstract()        # 提取摘要
    #    - complete_bib()             # ✅ 创建 references.bib（第 284-285 行）
    #    - get_paper_type()           # 分类论文类型
    #    - get_attri()                # 提取属性树
    
    # 2. 生成大纲（在 references.bib 创建之后）
    outline_generator = OutlinesGenerator(task_id)
    outline_generator.run()
    
    # 3. 生成内容
    content_generator.run()
    
    # 4. 后处理（包含引用名称修复）
    post_refiner.run()
```

## 生成机制

### 1. 初始生成（`DataCleaner.complete_bib()`）

位置：`src/modules/preprocessor/data_cleaner.py:77-107`

**生成逻辑**：
- 如果论文已有 `"reference"` 字段：使用该 BibTeX 条目，提取 `bib_name`
- 如果论文没有 `"reference"` 字段：根据标题自动生成简单的 BibTeX 条目
  ```python
  bib_name = title前10个字符（去空格） + 序号
  bib_tex = f"@article{{{bib_name},\ntitle={{{title}}}\n}}"
  ```

**问题**：
- 自动生成的 BibTeX 条目非常简单，只有 `title` 字段
- 没有作者、年份、期刊等完整信息

### 2. 正确性保证机制

#### 机制 1：引用名称修复（`BibNameReplacer`）

位置：`src/modules/heuristic_modules/map_cited_bib_names_to_refs.py`

**使用时机**：在 `post_refine` 阶段的 `RuleBasedRefiner` 中（第 42 行）

**工作原理**：
1. 从 `references.bib` 中提取所有已存在的 `bib_name`
2. 在生成的内容中查找所有 `\cite{...}` 引用
3. 对于每个引用名称，检查是否存在于 `references.bib` 中
4. 如果不存在，使用**模糊匹配**找到最接近的 `bib_name` 并替换

**示例**：
```python
# 生成的内容中使用了 \cite{LLMRG:ALa74}
# 但 references.bib 中没有这个名称
# BibNameReplacer 会找到最接近的 "LLM-R2:ALa74" 并替换
```

**局限性**：
- 模糊匹配可能不够准确
- 如果 `references.bib` 中没有相似的名称，可能替换错误
- 只能修复引用名称，不能修复 BibTeX 条目的内容

#### 机制 2：引用名称一致性

在生成内容时，LLM 应该使用 `papers` 目录中每篇论文的 `bib_name` 字段。但是：
- LLM 可能会生成新的引用名称（不在 `references.bib` 中）
- LLM 可能会使用错误的引用格式

## 潜在问题

### 问题 1：LLM 生成新引用

**场景**：LLM 在生成内容时，可能会引用一些不在原始参考论文列表中的论文，或者使用不同的引用名称。

**影响**：
- 这些引用名称不在 `references.bib` 中
- `BibNameReplacer` 会尝试用模糊匹配替换，但可能不准确

### 问题 2：BibTeX 条目不完整

**场景**：自动生成的 BibTeX 条目只有 `title` 字段，缺少作者、年份、期刊等信息。

**影响**：
- 编译后的 PDF 中引用信息不完整
- 不符合学术规范

### 问题 3：引用名称不匹配

**场景**：LLM 生成的引用名称与 `references.bib` 中的 `bib_name` 不一致。

**影响**：
- LaTeX 编译时可能出现 "Citation undefined" 警告
- `BibNameReplacer` 会尝试修复，但可能不够准确

## 改进建议

### 建议 1：在生成内容前验证引用

在 `ContentGenerator` 或 `MultiAgentGenerator` 中，可以：
1. 提取所有可用的 `bib_name` 列表
2. 在 prompt 中明确告诉 LLM 只能使用这些引用名称
3. 在生成后验证所有引用是否都在列表中

### 建议 2：增强 BibTeX 生成

在 `complete_bib()` 中，可以：
1. 尝试从 `md_text` 中提取更多信息（作者、年份、期刊等）
2. 使用 LLM 来补全 BibTeX 条目
3. 提供更完整的 BibTeX 格式

### 建议 3：后处理阶段验证

在 `post_refine` 阶段，可以：
1. 检查所有引用是否都在 `references.bib` 中
2. 对于不存在的引用，记录警告或错误
3. 提供更准确的引用名称修复机制

## 当前状态总结

✅ **已实现**：
- 在预处理阶段创建 `references.bib`
- 自动生成基本的 BibTeX 条目
- 后处理阶段使用模糊匹配修复引用名称

⚠️ **局限性**：
- BibTeX 条目可能不完整
- 引用名称修复依赖模糊匹配，可能不够准确
- 无法处理 LLM 生成的新引用（不在原始论文列表中）

## 相关代码位置

- `src/modules/preprocessor/data_cleaner.py:77-107` - `complete_bib()` 方法
- `src/modules/heuristic_modules/map_cited_bib_names_to_refs.py` - `BibNameReplacer` 类
- `src/modules/post_refine/rule_based_refiner.py:42` - 使用 `BibNameReplacer`

