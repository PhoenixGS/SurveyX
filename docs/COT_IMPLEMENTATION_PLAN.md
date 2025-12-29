# CoT (Chain of Thought) 功能实现思路文档

## 1. 总体架构设计

### 1.1 核心思想

CoT功能作为独立开关（`--cot`），与质量模式（`--quality_mode`）组合使用，形成四种模式：

| 质量模式 | CoT开关 | 模式名称 | 实现方式 |
|---------|---------|---------|---------|
| `default` | `False` | 标准单兵模式 | 直接生成内容（现有逻辑） |
| `default` | `True` | 内省式写作 | Prompt Engineering + 正则清洗思维链 |
| `high` | `False` | 团队协作模式 | Writer-Critic循环（现有逻辑） |
| `high` | `True` | 策划-执行-审核模式 | Planner + Writer + Critic 三层架构 |

### 1.2 设计原则

1. **成本控制**: Default + CoT 使用单一模型，成本增加最小（只增加输出token）
2. **质量提升**: High + CoT 使用专业分工，Planner用强推理模型（R1/GPT-4o），Writer用高性价比模型（V3/Flash）
3. **向后兼容**: 不改变现有接口，通过参数控制行为
4. **模块化**: CoT逻辑独立，易于测试和维护

---

## 2. Default模式 + CoT 实现方案

### 2.1 工作原理

**"内省式写作" (Self-Reflection CoT)**:
- 同一个模型先输出思考过程（在`[THOUGHT]...[/THOUGHT]`标签中）
- 然后输出正文内容
- 代码使用正则表达式清洗掉思考部分，只保留正文

### 2.2 需要修改的位置

#### 2.2.1 ContentGenerator类修改

**文件**: `src/models/generator/content_generator.py`

**修改点1**: 构造函数添加`use_cot`参数
```python
def __init__(self, task_id: str, use_cot: bool = False):
    super().__init__(task_id)
    self.use_cot = use_cot
    # ... 其他初始化代码
```

**修改点2**: `write_content_iteratively()`方法
- 当前使用: `fulfill_content_iteratively.md`
- CoT模式使用: `fulfill_content_iteratively_cot.md` (新增)
- 添加后处理: `_clean_cot_tags()` 方法

**修改点3**: 新增`_clean_cot_tags()`方法
```python
def _clean_cot_tags(self, content: str) -> str:
    """
    清洗CoT标签，提取纯文本内容
    移除 [THOUGHT]...[/THOUGHT] 标签及其内容
    """
    # 使用正则表达式移除思考标签
    pattern = r'\[THOUGHT\].*?\[/THOUGHT\]'
    cleaned = re.sub(pattern, '', content, flags=re.DOTALL)
    return cleaned.strip()
```

#### 2.2.2 Prompt模板创建

**新建文件**: `resources/LLM/prompts/content_generator/fulfill_content_iteratively_cot.md`

**核心要求**:
1. 要求模型在`[THOUGHT]...[/THOUGHT]`标签中展示思考过程
2. 思考过程包括：
   - 分析当前小节在整个survey中的位置和作用
   - 梳理需要引用的论文信息
   - 规划段落结构和逻辑顺序
   - 确定关键论点
3. 正文部分要求与原有prompt一致

**Prompt模板结构**:
```markdown
# Academic Writing with Chain of Thought

## Instructions

You are an academic writing specialist. Before writing, please **think step by step** about how to structure this section. Show your thinking process in [THOUGHT]...[/THOUGHT] tags, then write the actual content.

### Thinking Process Template

[THOUGHT]
1. **Section Context Analysis**: What role does this section play in the overall survey?
2. **Information Mapping**: Which papers and facts should be cited? How do they relate?
3. **Structure Planning**: What is the logical flow? How should paragraphs be organized?
4. **Key Points Identification**: What are the main arguments to convey?
[/THOUGHT]

### Output Format

[THOUGHT]
[Your thinking process here]
[/THOUGHT]

[CONTENT]
\subsection{{section_title}}
[Actual LaTeX content here]
[/CONTENT]

## Original Requirements
[保留原有prompt的所有要求和上下文]
```

### 2.3 调用流程

```python
# content_generator.py - write_content_iteratively()
def write_content_iteratively(self, ..., use_cot: bool = None):
    use_cot = use_cot if use_cot is not None else self.use_cot
    
    # 选择prompt模板
    prompt_template = (
        "fulfill_content_iteratively_cot.md" if use_cot
        else "fulfill_content_iteratively.md"
    )
    
    prompt = load_prompt(prompt_template, ...)
    
    # 调用LLM
    res = chat.remote_chat(prompt, model=ADVANCED_CHATAGENT_MODEL)
    res = clean_chat_agent_format(content=res)
    
    # CoT模式：清洗思考标签
    if use_cot:
        res = self._clean_cot_tags(res)
    
    # 后续处理
    res = res.replace("\\subsection{Conclusion}", "")
    return res
```

---

## 3. High模式 + CoT 实现方案

### 3.1 工作原理

**"策划-执行-审核"模式 (Architect-Builder-Inspector)**:
1. **Planner Agent**: 分析需求，输出结构化写作计划（逻辑大纲）
2. **Writer Agent**: 根据Plan生成初稿
3. **Critic Agent**: 验证初稿是否符合Plan和事实要求

### 3.2 需要修改的位置

#### 3.2.1 MultiAgentGenerator类修改

**文件**: `src/models/generator/multi_agent_generator.py`

**修改点1**: 构造函数添加`use_cot`参数并初始化Planner
```python
def __init__(
    self,
    task_id: str,
    max_retries: int = None,
    context_window_size: int = None,
    use_cot: bool = False,  # 新增
):
    super().__init__(task_id)
    self.use_cot = use_cot  # 新增
    # ... 现有初始化代码
    
    # CoT模式：创建Planner Agent（如果启用）
    if self.use_cot:
        self.planner_token_monitor = TokenMonitor(task_id, "multi_agent_planner")
        self.planner_agent = ChatAgent(token_monitor=self.planner_token_monitor)
        # 使用强推理模型（DeepSeek-R1或GPT-4o）
        # 模型配置从config.py读取
```

**修改点2**: 新增`_planner_generate_plan()`方法
```python
def _planner_generate_plan(
    self,
    subsection_title: str,
    subsection_desc: str,
    attribute_tree_facts: str,
    outlines: Outlines,
    written_content: str,
) -> str:
    """
    Planner Agent 生成写作计划
    
    Args:
        subsection_title: 小节标题
        subsection_desc: 小节描述
        attribute_tree_facts: Attribute Tree事实数据
        outlines: 完整大纲
        written_content: 已生成内容
    
    Returns:
        结构化写作计划（纯文本，200-500字）
    """
    prompt = load_prompt(
        self.prompt_dir / "planner_generate_plan.md",
        topic=self.topic,
        outlines=str(outlines),
        written_content=written_content,
        attribute_facts=attribute_tree_facts,
        section_title=subsection_title,
        section_desc=subsection_desc,
    )
    
    self.stats["total_planner_calls"] = getattr(self.stats, "total_planner_calls", 0) + 1
    
    response = self.planner_agent.remote_chat(
        text_content=prompt,
        temperature=HIGH_QUALITY_PLANNER_TEMPERATURE,  # 0.3（低温度保证逻辑严谨）
        model=HIGH_QUALITY_PLANNER_MODEL,  # DeepSeek-R1或GPT-4o
    )
    
    # 提取计划内容（如果有标签，清洗掉）
    plan = clean_chat_agent_format(content=response)
    # 确保计划简洁（不超过500字）
    if len(plan) > 500:
        plan = plan[:500] + "..."
    
    return plan
```

**修改点3**: 修改`generate_subsection_with_verification()`方法
```python
def generate_subsection_with_verification(
    self,
    subsection_title: str,
    subsection_desc: str,
    attribute_tree_facts: str,
    outlines: Outlines,
    written_content: str,
) -> str:
    """带验证的小节生成（支持CoT模式）"""
    
    # ===== Step 0: CoT模式 - Planner生成计划 =====
    writing_plan = ""
    if self.use_cot:
        logger.debug(f"[Planner] Generating plan for: {subsection_title}")
        writing_plan = self._planner_generate_plan(
            subsection_title=subsection_title,
            subsection_desc=subsection_desc,
            attribute_tree_facts=attribute_tree_facts,
            outlines=outlines,
            written_content=written_content,
        )
        logger.debug(f"[Planner] Plan generated ({len(writing_plan)} chars)")
    
    # ===== Step 1: Writer生成初稿 =====
    logger.debug(f"[Writer] Generating draft for: {subsection_title}")
    draft = self._writer_generate(
        subsection_title=subsection_title,
        subsection_desc=subsection_desc,
        attribute_tree_facts=attribute_tree_facts,
        context_window=self._get_context_window(written_content),
        outlines=outlines,
        written_content=written_content,
        writing_plan=writing_plan,  # 新增参数：传入Plan
    )
    
    # ===== Step 2 & 3: Critic验证和Writer修正（保持原有逻辑）=====
    # ... 现有代码不变
```

**修改点4**: 修改`_writer_generate()`方法，支持Plan参数
```python
def _writer_generate(
    self,
    subsection_title: str,
    subsection_desc: str,
    attribute_tree_facts: str,
    context_window: str,
    outlines: Outlines,
    written_content: str,
    writing_plan: str = "",  # 新增参数
) -> str:
    """Writer Agent生成初稿（支持根据Plan生成）"""
    
    # 选择prompt模板
    prompt_template = (
        "writer_generate_with_plan.md" if writing_plan
        else "writer_generate.md"
    )
    
    prompt = load_prompt(
        self.prompt_dir / prompt_template,
        topic=self.topic,
        outlines=str(outlines),
        written_content=written_content,
        context_window=context_window,
        attribute_facts=attribute_tree_facts,
        section_title=subsection_title,
        section_desc=subsection_desc,
        writing_plan=writing_plan,  # 传入Plan
    )
    
    # ... 后续调用逻辑不变
```

**修改点5**: 统计信息更新
```python
self.stats = {
    "total_subsections": 0,
    "passed_first_try": 0,
    "passed_after_revision": 0,
    "max_retries_reached": 0,
    "total_writer_calls": 0,
    "total_critic_calls": 0,
    "total_planner_calls": 0,  # 新增
}
```

#### 3.2.2 Prompt模板创建

**新建文件1**: `resources/LLM/prompts/multi_agent/planner_generate_plan.md`

**核心要求**:
1. 输出简洁的结构化写作计划（200-500字）
2. 不写具体内容，只规划逻辑结构
3. 明确每个段落应该覆盖哪些关键点
4. 规划引用哪些论文和事实

**Prompt结构**:
```markdown
# Planner Agent - Writing Plan Generation

- Role: Writing Architect and Logic Planner
- Background: You are a Planner Agent in a Multi-Agent system. Your job is to create a logical writing plan BEFORE the Writer starts writing.

## Core Principle

**DO NOT write the actual content.** Only output a step-by-step plan that outlines:
1. Logical structure and flow
2. Key points to cover in each paragraph
3. Which papers/facts should be cited
4. How to connect ideas

## Input Context

### Survey Topic
{topic}

### Complete Outline
{outlines}

### Content Written So Far
{written_content}

### Attribute Tree Facts (Source Material)
{attribute_facts}

### Section to Plan
- **Title**: {section_title}
- **Description**: {section_desc}

## Output Format

Output a concise writing plan (200-500 words) following this structure:

### Logical Structure
[Describe the overall structure: how many paragraphs, what each covers]

### Key Points to Cover
1. [Point 1] - Cite: [relevant papers/facts]
2. [Point 2] - Cite: [relevant papers/facts]
3. ...

### Flow and Transitions
[How to connect different paragraphs and ideas]

### Critical Facts to Include
[List specific facts from Attribute Tree that MUST be mentioned]

**Important**: Keep the plan concise (max 500 words). This plan will guide the Writer, so focus on logic and structure, not detailed content.
```

**新建文件2**: `resources/LLM/prompts/multi_agent/writer_generate_with_plan.md`

**核心要求**:
1. 在原有`writer_generate.md`基础上，增加Plan输入
2. 要求Writer严格按照Plan执行
3. 同时保持与原有prompt的兼容性

**主要修改**:
```markdown
# Writer Agent - Initial Draft Generation (With Plan)

[保留原有writer_generate.md的所有内容]

### Writing Plan (Provided by Planner Agent)
```
{writing_plan}
```
**Important**: You MUST follow this plan when writing. The plan outlines the logical structure and key points to cover. Use it as a guide, but still write natural, fluent academic prose.

[其余要求与原prompt相同]
```

---

## 4. 配置文件修改

### 4.1 config.py 新增配置

**文件**: `src/configs/config.py`

```python
# ==================== CoT (Chain of Thought) 配置 ====================

# Planner Agent 配置（仅用于 High + CoT 模式）
# 使用强推理模型（逻辑规划能力强）
HIGH_QUALITY_PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "DeepSeek-R1")
HIGH_QUALITY_PLANNER_TEMPERATURE = 0.3  # 低温度保证计划严谨

# Default + CoT 模式使用的模型（与原default模式相同）
# 如果需要，可以单独配置
DEFAULT_COT_MODEL = os.environ.get("DEFAULT_COT_MODEL", ADVANCED_CHATAGENT_MODEL)
DEFAULT_COT_TEMPERATURE = 0.7  # 可适当提高温度以获得更多创造性思考
```

---

## 5. 命令行参数和接口修改

### 5.1 参数解析修改

**需要修改的文件**:
1. `src/modules/preprocessor/utils.py` - `parse_arguments_for_offline()`
2. `tasks/full_run.py` - 可能需要添加参数
3. `tasks/workflow/04_gen_content.py` - `parse_arguments()`

**修改示例** (`src/modules/preprocessor/utils.py`):
```python
parser.add_argument(
    "--cot",
    action="store_true",
    default=False,
    help="Enable Chain of Thought (CoT) reasoning. "
         "In default mode: adds self-reflection thinking. "
         "In high mode: adds Planner agent for logical planning."
)
```

### 5.2 接口修改

**文件**: `tasks/full_run.py` 和 `tasks/offline_run.py`

```python
def generate_single_survey(
    task_id: str, 
    chat_agent: ChatAgent = None, 
    quality_mode: str = "default",
    use_cot: bool = False,  # 新增参数
):
    """生成单个survey"""
    # ... 现有代码
    
    if quality_mode == "high":
        content_generator = MultiAgentGenerator(
            task_id=task_id,
            use_cot=use_cot,  # 传递CoT参数
        )
    else:
        content_generator = ContentGenerator(
            task_id=task_id,
            use_cot=use_cot,  # 传递CoT参数
        )
    
    content_generator.run()
```

### 5.3 参数传递链路

```
命令行参数 (--cot)
  ↓
parse_arguments_for_offline() / parse_arguments()
  ↓
offline_generate() / generate_single_survey()
  ↓
ContentGenerator.__init__(use_cot=...) / MultiAgentGenerator.__init__(use_cot=...)
  ↓
write_content_iteratively(use_cot=...) / generate_subsection_with_verification()
```

---

## 6. 成本评估

### 6.1 Default + CoT 成本

**假设**（每个小节）:
- Input tokens: 2000 tokens (与原Default相同)
- Output tokens: 
  - 思考部分: ~300 tokens (新增)
  - 正文部分: ~1000 tokens (与原Default相同)
  - **总计: ~1300 tokens** (增加30%)

**成本计算**（以DeepSeek-V3.2为例，假设价格为 $0.001/1K input tokens, $0.002/1K output tokens）:
- 原Default: 2000 * 0.001/1000 + 1000 * 0.002/1000 = $0.004/section
- Default + CoT: 2000 * 0.001/1000 + 1300 * 0.002/1000 = $0.0046/section
- **成本增加: +15%** (仅考虑输出token增加)

**整篇survey假设30个小节**:
- 原Default: $0.004 * 30 = $0.12
- Default + CoT: $0.0046 * 30 = $0.138
- **总成本增加: $0.018 (+15%)**

### 6.2 High + CoT 成本

**假设**（每个小节）:

**Planner阶段**:
- Input tokens: 2500 tokens (包含outlines, written_content, attribute_facts)
- Output tokens: 200 tokens (简洁的计划)
- 模型: DeepSeek-R1 (假设价格 $0.003/1K input, $0.006/1K output)

**Writer阶段**:
- Input tokens: 2500 tokens (包含plan)
- Output tokens: 1000 tokens (初稿)
- 模型: DeepSeek-V3.2 (假设价格 $0.001/1K input, $0.002/1K output)

**Critic阶段**（假设1次验证）:
- Input tokens: 4000 tokens (draft + attribute_facts + plan)
- Output tokens: 100 tokens (JSON反馈)
- 模型: DeepSeek-R1 (假设价格 $0.003/1K input, $0.006/1K output)

**成本计算**（每个小节）:
- Planner: 2500 * 0.003/1000 + 200 * 0.006/1000 = $0.0087
- Writer: 2500 * 0.001/1000 + 1000 * 0.002/1000 = $0.0045
- Critic: 4000 * 0.003/1000 + 100 * 0.006/1000 = $0.0126
- **总计: $0.0258/section**

**与原High模式对比**（原High模式无Planner）:
- 原High (Writer + Critic): (2500 * 0.001/1000 + 1000 * 0.002/1000) + (3500 * 0.003/1000 + 100 * 0.006/1000) = $0.0045 + $0.0111 = $0.0156/section
- High + CoT: $0.0258/section
- **成本增加: +65%** (主要是Planner阶段增加了R1模型的调用)

**整篇survey假设30个小节**:
- 原High: $0.0156 * 30 = $0.468
- High + CoT: $0.0258 * 30 = $0.774
- **总成本增加: $0.306 (+65%)**

### 6.3 成本优化建议

1. **Planner输出限制**: 严格要求Plan不超过200字，减少输出token
2. **缓存机制**: 相似小节的Plan可以复用或微调
3. **模型选择**: 
   - Planner可以用GPT-4o-mini（如果支持）替代R1，降低成本
   - 但建议保持R1，因为逻辑规划是关键环节

### 6.4 成本对比总结表

| 模式 | 每小节成本 | 30小节总成本 | 相对Default增加 |
|-----|-----------|-------------|----------------|
| Default | $0.004 | $0.12 | - |
| Default + CoT | $0.0046 | $0.138 | +15% |
| High | $0.0156 | $0.468 | +290% |
| High + CoT | $0.0258 | $0.774 | +545% |

**注意**: 以上价格仅为示例，实际价格请参考各模型提供商的最新定价。

---

## 7. 实现检查清单

### 7.1 Default + CoT 实现

- [ ] 修改`ContentGenerator.__init__()`添加`use_cot`参数
- [ ] 创建`_clean_cot_tags()`方法
- [ ] 修改`write_content_iteratively()`支持CoT prompt
- [ ] 创建`fulfill_content_iteratively_cot.md` prompt模板
- [ ] 测试CoT标签清洗功能
- [ ] 验证输出质量

### 7.2 High + CoT 实现

- [ ] 修改`MultiAgentGenerator.__init__()`添加`use_cot`参数和Planner初始化
- [ ] 创建`_planner_generate_plan()`方法
- [ ] 修改`generate_subsection_with_verification()`添加Planner调用
- [ ] 修改`_writer_generate()`支持Plan参数
- [ ] 创建`planner_generate_plan.md` prompt模板
- [ ] 创建`writer_generate_with_plan.md` prompt模板
- [ ] 更新统计信息收集
- [ ] 测试完整流程

### 7.3 配置和接口

- [ ] 在`config.py`添加Planner相关配置
- [ ] 修改参数解析函数添加`--cot`参数
- [ ] 修改`full_run.py`和`offline_run.py`传递CoT参数
- [ ] 更新文档说明

### 7.4 测试

- [ ] 单元测试：CoT标签清洗
- [ ] 集成测试：Default + CoT完整流程
- [ ] 集成测试：High + CoT完整流程
- [ ] 成本监控测试
- [ ] 输出质量对比测试

---

## 8. 关键注意事项

### 8.1 Context Window管理

**问题**: High + CoT模式下，上下文窗口可能爆炸
- Planner需要看: outlines + written_content + attribute_facts
- Writer需要看: outlines + written_content + attribute_facts + **plan**
- Critic需要看: draft + attribute_facts + **plan**

**解决方案**:
1. **Plan长度限制**: 严格要求Plan不超过200-500字
2. **Context截断**: 如果written_content过长，只传递最近N个字符
3. **Attribute Facts精简**: 只传递当前小节相关的facts

### 8.2 Prompt设计要点

**Default + CoT Prompt**:
- 明确要求思考过程放在`[THOUGHT]...[/THOUGHT]`标签中
- 正文放在`[CONTENT]...[/CONTENT]`标签中（可选，便于提取）
- 或者直接要求LaTeX格式输出，思考部分在前面

**Planner Prompt**:
- **关键**: 强调"DO NOT write content, only plan"
- 要求输出结构化但简洁的计划
- 明确字数限制（200-500字）

**Writer with Plan Prompt**:
- 强调严格按照Plan执行
- 但保持自然流畅的写作风格
- Plan作为指导，不是限制

### 8.3 错误处理

1. **CoT标签缺失**: 如果模型没有输出标签，fallback到原逻辑
2. **Plan格式错误**: 如果Plan过长或格式错误，进行截断或清洗
3. **Planner调用失败**: 如果Planner失败，fallback到无Plan模式

### 8.4 向后兼容性

- 所有修改都通过参数控制，默认行为不变
- 现有调用代码无需修改
- 新的CoT功能是可选的增强功能

---

## 9. 实施优先级

### Phase 1: Default + CoT（优先级高）
- **原因**: 实现简单，成本增加小，可快速验证效果
- **工作量**: 2-3天
- **文件**: 
  - `content_generator.py`
  - `fulfill_content_iteratively_cot.md`
  - 参数解析和配置

### Phase 2: High + CoT（优先级中）
- **原因**: 实现复杂，但效果可能更显著
- **工作量**: 3-5天
- **文件**:
  - `multi_agent_generator.py`
  - `planner_generate_plan.md`
  - `writer_generate_with_plan.md`
  - 配置更新

### Phase 3: 优化和测试（优先级中）
- **原因**: 确保稳定性和性能
- **工作量**: 2-3天
- **内容**: 
  - 性能优化
  - 错误处理完善
  - 测试用例
  - 文档更新

---

## 10. 预期效果

### 10.1 Default + CoT 预期效果

**质量提升**:
- 逻辑连贯性: +10-15%
- 结构合理性: +10-15%
- 内容完整性: +5-10%

**成本**: +15%

### 10.2 High + CoT 预期效果

**质量提升**:
- 逻辑连贯性: +20-30%
- 结构合理性: +25-35%
- 内容完整性: +15-20%
- 事实准确性: 保持High模式的水平（通过Critic保证）

**成本**: +65%（相比High模式）

---

## 11. 后续优化方向

1. **Plan复用**: 相似小节的Plan可以缓存和复用
2. **自适应CoT**: 根据小节复杂度决定是否启用CoT
3. **Plan细化**: 支持多级Plan（大纲级 -> 段落级）
4. **CoT评估**: 添加评估机制，判断CoT是否真正提升了质量

---

## 12. 参考资料

- CoT (Chain of Thought) 原始论文: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- Multi-Agent系统设计模式
- 模型定价参考（需要根据实际使用的API更新）

