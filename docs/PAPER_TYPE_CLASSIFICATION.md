# Paper Type 分类说明

## 支持的分类类型

代码中支持 4 种类型（`src/modules/preprocessor/data_cleaner.py:211`）：

1. **`method`** - 方法论文
   - 引入新方法、技术或算法来解决特定问题

2. **`benchmark`** - 基准测试论文
   - 提出新数据集、评估协议或性能标准
   - 用于衡量模型或方法的有效性

3. **`theory`** - 理论论文
   - 发展新的理论见解、框架或原理
   - 有助于理解现象或领域

4. **`survey`** - 综述论文
   - 提供现有文献的全面综述或分析
   - 总结特定领域的研究发现或趋势

## Prompt 模板

**文件位置**：`resources/LLM/prompts/preprocessor/paper_type_classification.md`

**关键要求**：
- 输出格式：单个词（Method, Benchmark, Theory, Survey）
- 约束：不能包含额外信息或解释
- 示例输出：`Method.` 或 `Theory.`

**Prompt 内容**：
```
- Role: Academic Paper Classifier
- Background: The user seeks to categorize a given paper abstract into one of four academic types: Method, Benchmark, Theory, or Survey.
- Goals: To accurately categorize the paper abstract into one of the four specified types based on its content.
- Constrains: The output must be a single category (Method, Benchmark, Theory, or Survey) and should not include additional information or explanations.
- OutputFormat: A single word representing the category (Method, Benchmark, Theory, Survey)
```

## 分类逻辑

代码中的分类逻辑（`src/modules/preprocessor/data_cleaner.py:210-220`）：

```python
def __process_paper_type_response(self, res: str, paper_index: int):
    kinds = ["method", "benchmark", "theory", "survey"]
    for k in kinds:
        if k in res.lower():  # 小写匹配
            self.papers[paper_index]["paper_type"] = k
            return True
    # 如果都不匹配，返回 False
    return False
```

## 常见问题

### 问题 1：LLM 返回意外的类型

**现象**：LLM 返回 "Dataset"、"Application" 等不在预期列表中的词

**原因**：
1. LLM 未严格遵循 prompt，返回了其他词
2. 某些论文确实难以归类（如数据集论文可能被误判为 benchmark）
3. 模型版本或 prompt 理解差异

**解决方案**：
- 代码已经添加了默认值处理：如果分类失败，使用 `"method"` 作为默认值
- 如果返回的类型不在支持列表中，也会使用 `"method"` 作为默认值

### 问题 2：分类不准确

**现象**：某些论文被错误分类

**可能原因**：
- Abstract 内容不够清晰
- 论文类型确实模糊（如既有方法又有数据集）

**建议**：
- 可以手动检查分类结果
- 如果发现大量错误分类，可以考虑改进 prompt 或添加更多示例

## 相关文件

- Prompt 模板：`resources/LLM/prompts/preprocessor/paper_type_classification.md`
- 分类逻辑：`src/modules/preprocessor/data_cleaner.py:210-251`
- 属性树提取：`src/modules/preprocessor/data_cleaner.py:280-304`
  - 根据 `paper_type` 加载对应的 prompt：
    - `attri_tree_for_method.md`
    - `attri_tree_for_benchmark.md`
    - `attri_tree_for_theory.md`
    - `attri_tree_for_survey.md`

