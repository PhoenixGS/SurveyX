# A Survey of Chain of Thought Reasoning: Advances, Frontiers and Future  

Zheng $\mathbf{Chu^{1}}$ ,∗ Jingchang Chen1,∗ Qianglong Chen2,∗ Weijiang $\mathbf{Y}\mathbf{u}^{2}$ , Tao $\mathbf{H}\mathbf{e}^{1}$ Haotian Wang1, Weihua Peng2, Ming ${\bf L i u}^{1\dagger}$ , Bing $\mathbf{Qin}^{1}$ , Ting Liu1 1Harbin Institute of Technology, Harbin, China 2Huawei Inc., Shenzhen, China {zchu, jcchen, the, mliu† , qinb, tliu}@ir.hit.edu.cn {chenqianglong.ai, wanght1998, weijiangyu8, pengwh.hit}@gmail.com  

# Abstract  

Chain-of-thought reasoning, a cognitive process fundamental to human intelligence, has garnered signifcant attention in the realm of artifcial intelligence and natural language processing. However, there still remains a lack of a comprehensive survey for this arena. To this end, we take the frst step and present a thorough survey of this research feld carefully and widely. We use X-of-Thought to refer to Chain-of-Thought in a broad sense. In detail, we systematically organize the current research according to the taxonomies of methods, including XoT construction, XoT structure variants, and enhanced XoT. Additionally, we describe XoT with frontier applications, covering planning, tool use, and distillation. Furthermore, we address challenges and discuss some future directions, including faithfulness, multi-modal, and theory. We hope this survey serves as a valuable resource for researchers seeking to innovate within the domain of chain-of-thought reasoning1.  

# 1 Introduction  

Pre-trained language models (PLMs) can automatically learn general representations from unlabeled text and achieve excellent performance through fne-tuning on downstream tasks. (Devlin et al., 2019; Raffel et al., 2020; Radford and Narasimhan, 2018). Recently, scaling up language models significantly improves performance and brings many surprises, such as emergent abilities (Wei et al., 2022a; Schaeffer et al., 2023). Therefore, the paradigm of natural language processing is shifting from pretraining with fne-tuning to pre-training with incontext learning. However, as of now, large-scale language models (LLMs) still have considerable room for improvement on complex reasoning tasks, such as mathematical reasoning (Cobbe et al., 2021; Patel et al., 2021), commonsense reasoning (Talmor et al., 2021; Mihaylov et al., 2018), etc.  

To leverage LLMs for addressing complex reasoning tasks, Wei et al. (2022b) extends in-context learning with step-by-step reasoning processes, frst introducing the concept of chain-of-thought (CoT) prompting. Kojima et al. (2022) fnds that simply adding a magic phrase Let’s think step by step in prompts enables LLMs to perform zero-shot chain-of-thought reasoning without any human annotation. These studies have highlighted the signifcance of chain-of-thought in enhancing the model’s capability for complex reasoning and improving its reasoning and planning abilities.  

Subsequently, a substantial of works about Xof-thought (XoT) emerges like mushrooms after the rain in the NLP community, such as automatic XoT construction (Kojima et al., 2022; Zhang et al., 2023f; Xu et al., 2023), XoT structural variants (Chen et al., 2022a; Ning et al., 2023; Lei et al., 2023a; Yao et al., 2023b), etc. Note that to distinguish it from primitive CoT, we use XoT to refer to CoT in a broad sense, which is a collective term for the use of step-by-step reasoning methods.  

However, these methods and datasets have not yet undergone systematic review and analysis. To fll this gap, we propose this work to conduct a comprehensive and detailed analysis of the XoT family. Even though there have been some surveys discussing chain-of-thought, they are limited to specifc aspects, such as LLM reasoning with prompts (Qiao et al., 2023) and chain-of-thought prompt strategies (Yu et al., 2023c). In contrast, our survey not only provides a more thorough and comprehensive discussion of the topics they’ve already covered, but also includes additional topics and discussions, such as XoT construction, XoT structural variants and frontier application, etc. Concretely, in this paper, we frst introduce the relevant background and preliminary (§2). Furthermore, we carefully classify the XoT series of work from multiple perspectives and complete an in-depth analysis (§4), including XoT construction methods (§4.1), XoT structure variants (§4.2) and XoT enhancement methods (§4.3). Then, we provide practical applications of the XoT in the frontier felds (§5). In order to inspire the follow-up work of XoT, we offer insights into potential avenues for future research in this area (§6). Finally, we compare and discuss existing methods $(\S7)$ .  

# 2 Background and Preliminary  

# 2.1 Background  

In recent years, with the continuous expansion of computing power, large-scale language models have sprung up (Brown et al., 2020; OpenAI, 2023; Touvron et al., 2023a; Scao et al., 2022; Touvron et al., 2023b; Zhao et al., 2023b), and as the model size continues to grow, many new capabilities have emerged, such as in-context learning and chain-ofthought reasoning (Brown et al., 2020; Wei et al., 2022b,a; Schaeffer et al., 2023).  

Brown et al. (2020) fnds that large-scale language models have excellent in-context learning (ICL) ability. ICL incorporates input-output demonstrations into the prompt text. With ICL, off-the-shelf LLMs can be employed without additional fne-tuning while achieving comparable performance. Nevertheless, this end-to-end approach tends to underperform when faced with complex reasoning tasks.  

Wei et al. (2022b) fnds that the reasoning ability of LLMs can be improved by adding step-by-step reasoning processes to the demonstration, which is known as chain-of-thought prompting. CoT prompting enables the model to gain a more precise understanding of both the question’s intricacies and the reasoning process. Furthermore, the model generates a sequence of reasoning steps, which grants us a transparent view of the model’s cognitive process, further enhancing interpretability.  

# 2.2 Preliminary  

In this section, we introduce the preliminary chainof-thought reasoning with LLMs, and we refer to the formula defnition in (Qiao et al., 2023). Suppose there is a question $\mathcal{Q}$ , a prompt $\tau$ and a probabilistic language model $P_{L M}$ . The model takes the question and prompt as inputs to give the rationale $\mathcal{R}$ and answer $\boldsymbol{\mathcal{A}}$ . We frst consider in-context scenarios where the demonstrations do not contain reasoning chains. We need to maximize the likelihood of Answer $\boldsymbol{\mathcal{A}}$ , as shown in Equ (1,2).  

$$
\begin{array}{l}{{p(A\mid T,\mathcal{Q})=\displaystyle\prod_{i=1}^{|A|}p_{L M}(a_{i}\mid T,\mathcal{Q},a_{<i})}}\\ {{\mathcal{T}_{I C L}=\{I,(x_{1},y_{1}),\dots,(x_{n},y_{n})\}}}\end{array}
$$  

In the chain-of-thought reasoning scenario, where the demonstrations contain reasoning process, we need to maximize the likelihood of Answer $\boldsymbol{\mathcal{A}}$ and rationale $\mathcal{R}$ , as shown in Equ (3,4,5,6).  

$$
\begin{array}{l}{{p(A\mid T,Q)=p(A\mid T,Q,\mathcal{R})p(\mathcal{R}\mid T,Q)}}\\ {{\;}}\\ {{p(\mathcal{R}\mid T,Q)=\displaystyle\prod_{i=1}^{|\mathcal{R}|}p_{L M}(r_{i}\mid T,Q,r_{<i})}}\\ {{\;}}\\ {{p(A|T,Q,\mathcal{R})=\displaystyle\prod_{j=1}^{|A|}p_{L M}(a_{i}|T,Q,\mathcal{R},a_{<j})}}\\ {{\;}}\\ {{T_{\mathrm{COT}}=\{I,(x_{1},e_{1},y_{1}),\cdots,(x_{n},e_{n},y_{n})\}}}\end{array}
$$  

# 3 Benchmarks  

# 3.1 Mathematical Reasoning  

Mathematical reasoning is often used to measure the reasoning power of a model. Early benchmarks contain simple arithmetic operations (Hosseini et al., 2014; Koncel-Kedziorski et al., 2015; Roy and Roth, 2015; Koncel-Kedziorski et al., 2016). Ling et al. (2017) labels the reasoning process in natural language form, and Amini et al. (2019) builds on AQUA by labeling the reasoning process in program form. Later benchmarks (Miao et al., 2020; Patel et al., 2021; Cobbe et al., 2021; Gao et al., 2023) contain more complex and diverse questions. (Zhu et al., 2021; Chen et al., 2021, 2022b) require reasoning based on the table content. There are also general benchmarks (Hendrycks et al., 2021; Mishra et al., 2022a,b) and reading comprehension form benchmarks (Dua et al., 2019; Chen et al., 2023). Recently, (Yu et al., 2021a) endowed pre-trained model with the ability of mathematical reasoning by using hierarchical reasoning and knowledge.  

# 3.2 Commonsense Reasoning  

Commonsense reasoning is the process of making inferences, judgments, and understandings based on knowledge that is generally known and commonly perceived in the everyday world. How to acquire and understand commonsense knowledge is a major impediment to models facing commonsense reasoning. Many benchmarks and tasks are proposed focusing on commonsense understanding (Talmor et al., 2019, 2021; Bhakthavatsalam et al., 2021; Mihaylov et al., 2018; Geva et al., 2021; Huang et al., 2019; Bisk et al., 2020), event temporal commonsense reasoning (Rashkin et al., 2018; Zhou et al., 2019) , and commonsense verifcation (Wang et al., 2019).  

<html><body><table><tr><td>Task</td><td>Dataset</td><td>Size</td><td>Input</td><td>ndno</td><td>Rationale</td><td>Description</td></tr><tr><td rowspan="15">Mathematical Reasoning</td><td>AddSub (Hosseini et al., 2014) SingleEq (Koncel-Kedziorski et al., 2015)</td><td>395 508</td><td>Question Question</td><td>Number Number</td><td>Equation Equation</td><td>Simple arithmetic Simple arithmetic</td></tr><tr><td></td><td>600</td><td></td><td></td><td></td><td></td></tr><tr><td>MultiArith (Roy and Roth, 2015)</td><td></td><td>Question</td><td>Number</td><td>Equation</td><td>Simple arithmetic</td></tr><tr><td>MAWPS (Koncel-Kedziorski et al., 2016)</td><td>3320</td><td>Question</td><td>Number</td><td>Equation</td><td>Simple arithmetic</td></tr><tr><td>AQUA-RAT (Ling et al., 2017)</td><td>100,000</td><td>Question</td><td>Option</td><td>Natural Language</td><td>Math reasoning with NL rationale</td></tr><tr><td>ASDiv (Miao et al., 2020)</td><td>2305</td><td>Question</td><td>Number</td><td>Equation</td><td>Multi-step math reasoning</td></tr><tr><td>SVAMP (Patel et al., 2021)</td><td>1,000</td><td>Question</td><td>Number</td><td>Equation</td><td>Multi-step math reasoning</td></tr><tr><td>GSM8K (Cobbe et al., 2021)</td><td>8,792</td><td>Question</td><td>Number</td><td>Natural Language</td><td>Multi-step math reasoning</td></tr><tr><td>GSM-Hard (Gao et al., 2023)</td><td>936</td><td>Question</td><td>Number</td><td>Natural Language</td><td>GSM8K with larger number</td></tr><tr><td>MathQA (Amini et al., 2019)</td><td>37,297</td><td>Question</td><td>Number</td><td>Operation</td><td>Annotated based on AQUA</td></tr><tr><td>DROP (Dua et al., 2019)</td><td>96,567</td><td>Question+Passage</td><td>Number+Span</td><td>Equation</td><td>Reading comprehension form</td></tr><tr><td>TheoremQA (Chen et al., 2023)</td><td>800</td><td>Question+Theorem</td><td>Number</td><td></td><td>Answer based on theorems</td></tr><tr><td>TAT-QA (Zhu et al., 2021)</td><td>16,552</td><td>Question+Table+Text</td><td>Number+Span</td><td>Operation</td><td>Answer based on tables</td></tr><tr><td>FinQA (Chen et al., 2021)</td><td>8,281</td><td>Question+Table+Text</td><td>Number</td><td>Operation</td><td>Answer based on tables</td></tr><tr><td>ConvFinQA (Chen et al., 2022b) MATH (Hendrycks et al., 2021)</td><td>3892 12500</td><td>Question+Table+Dialog</td><td>Number</td><td>Operation</td><td>Multi-turn dialogs</td></tr><tr><td>NumGLUE (Mishra et al., 2022b)</td><td>101,835</td><td>Question Question+Text</td><td>Number Number+Span</td><td>Natural Language</td><td>Challenging competition math problems Multi-task benchmark</td></tr><tr><td>LILA (Mishra et al., 2022a)</td><td>133,815</td><td>Question+Text</td><td>Free-form</td><td>Program</td><td>Multi-task benchmark</td></tr><tr><td>ARC (Bhakthavatsalam et al., 2021) Commonsense Reasoning</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OpenBookQA (Mihaylov et al., 2018)</td><td></td><td>Question</td><td></td><td></td><td>From science exam</td></tr><tr><td>PIQA (Bisk et al., 2020)</td><td>7787 5,957 21000</td><td>Question+Context Goal+Solution</td><td>Option Option</td><td></td><td>Open-book knowledges</td></tr><tr><td>CommonsenseQA (Talmor et al., 2019)</td><td>12247</td><td>Question</td><td>Option</td><td></td><td>Physical commonsense knowledge</td></tr><tr><td>CommonsenseQA 2.0 (Talmor et al., 2021)</td><td>14343</td><td>Question</td><td>Option</td><td></td><td>Derived from ConceptNet</td></tr><tr><td>Event2Mind (Rashkin et al., 2018)</td><td>25000</td><td>Event</td><td>Yes/No</td><td></td><td>Gaming annotation with high quality</td></tr><tr><td>McTaco (Zhou et al., 2019)</td><td>13225</td><td>Question</td><td>Intent+Reaction</td><td></td><td>Intension commonsense reasoning</td></tr><tr><td>CosmosQA (Huang et al., 2019)</td><td>35588</td><td>Question+Paragraph</td><td>Option</td><td></td><td>Event temporal commonsense reasoning Narrative commonsense reasoning</td></tr><tr><td>Com Validation (Wang et al., 2019)</td><td>11997</td><td>Statement</td><td>Option</td><td></td><td></td></tr><tr><td>ComExplanation (Wang et al., 2019)</td><td>11997</td><td>Statement</td><td>Option</td><td></td><td>Commonsense verification</td></tr><tr><td>StrategyQA (Geva et al., 2021)</td><td>2,780</td><td>Question Words</td><td>Option/Free-form Yes/No Letters</td><td></td><td>Commonsense explanation Multi-hop commonsense reasoning</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Symbolic Reasoning</td><td>Last Letter Concat. (Wei et al., 2022b) Coin Flip (Wei et al., 2022b)</td><td></td><td></td><td>Yes/No</td><td>Rule-based</td></tr><tr><td>Reverse List (Wei et al., 2022b)</td><td></td><td>Statement</td><td></td><td></td><td>Rule-based</td></tr><tr><td>BigBench (Srivastava et al., 2022)</td><td></td><td>List</td><td>Reversed List</td><td></td><td>Rule-based</td></tr><tr><td>BigBench-Hard (Suzgun et al., 2023)</td><td></td><td></td><td></td><td></td><td>Contains multiple symbolic reasoning datasets Contains multiple symbolic reasoning datasets</td></tr><tr><td>ReClor (Yu et al., 2020)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LogiQA (Liu et al., 2020) Logical Reasoning</td><td></td><td>6,138 8,678</td><td>Question+Context</td><td></td><td>Questions from GMAT and LSAT</td></tr><tr><td>ProofWriter (Tafjord et al., 2021)</td><td></td><td>Question+Paragraph</td><td>Option Option</td><td></td><td>Questions from China Civil Service Exam</td></tr><tr><td>FOLIO (Han et al., 2022)</td><td>20192 1435</td><td>Question+Rule</td><td>Answer+Proof</td><td>Entailment Tree</td><td>Reasoning process generation</td></tr><tr><td>DEER (Yang et al., 2022)</td><td>1,200</td><td>Conclusion+Premise Fact</td><td>Yes/No Rule</td><td></td><td>First-order logic</td></tr><tr><td>PrOntoQA (Saparov and He, 2023)</td><td></td><td>Question+Context</td><td>Yes/No+Proccess</td><td>First-Order Logic</td><td>Inductive reasoning Deductive reasoning</td></tr><tr><td>Multimodal</td><td></td><td>264,720</td><td></td><td></td><td></td></tr><tr><td>VCR (Zellers et al., 2019) VisualCOMET (Park et al., 2020)</td><td></td><td>Question+Image</td><td>Option</td><td>Natural Language</td><td>Visual commonsense reasoning</td></tr><tr><td>PMR (Dong et al., 2022)</td><td>1,465,704</td><td>Image+Event</td><td>Action+Intent</td><td></td><td>Visual commonsense reasoning</td></tr><tr><td></td><td>15,360</td><td>Image+Background</td><td>Option</td><td></td><td>Premise-based multi-modal reasoning</td></tr><tr><td>ScienceQA (Lu et al.,2022)</td><td>21,208</td><td>Q+Image+Context</td><td>Option</td><td>Natural Language</td><td>Multi-modal reasoning with NL rationales</td></tr><tr><td>VLEP (Lei et al., 2020) Reasoning</td><td>28,726</td><td>Premise+Video</td><td>Option</td><td></td><td>Video event prediction</td></tr><tr><td>CLEVRER (Yi et al., 2020)</td><td>305,280</td><td>Question+Video</td><td>Option/Free-form</td><td>Program</td><td>Video temporal and causal reasoning</td></tr><tr><td>STAR (Wu et al., 2021)</td><td>600,000</td><td>Question+Video</td><td>Option</td><td></td><td>Video situated reasoning</td></tr><tr><td>NEXT-QA (Xiao et al., 2021)</td><td>47,692</td><td>Question+ Video</td><td>Option</td><td></td><td>Video temporal,causal.commonsense reasoning</td></tr><tr><td>Causal-VidQA (Li et al., 2022a)</td><td>107,600</td><td>Question+Video</td><td>Free-form</td><td>Natural Language</td><td>Video causal and commonsense reasoning</td></tr><tr><td>News-KVQA (Gupta and Gupta, 2022)</td><td>1,041,352</td><td>Q+V+KG</td><td>Option</td><td></td><td>Video reasoning with external knowledge</td></tr></table></body></html>

Table 1: An overview of benchmarks and tasks on reasoning.  

# 3.4 Logical Reasoning  

Logical reasoning is divided into deductive reasoning, inductive reasoning, and abductive reasoning (Yu et al., 2023a). Deductive reasoning derives conclusions from general premises (Liu et al., 2020; Yu et al., 2020; Tafjord et al., 2021; Han et al., 2022). Inductive reasoning derives general conclusions from special cases (Yang et al., 2022). Abductive reasoning gives rational explanations for observed phenomena (Saparov and He, 2023).  

# 3.3 Symbolic Reasoning  

Symbolic reasoning here refers specifcally to the simulation of some simple operations, which are simple for humans yet challenging for LLMs. Last letter concatenation, coin fip, and reverse list (Wei et al., 2022b) are the most commonly used symbolic reasoning tasks. In addition, the collaborative benchmark BigBench (Srivastava et al., 2022) and BigBench-Hard (Suzgun et al., 2023) also contain several symbolic reasoning datasets, such as state tracking and object counting.  

# 3.5 Multi-modal Reasoning  

In the real world, reasoning also involves information in modalities other than text, with visual modalities being the most prevalent. To this end, many benchmarks for visual multi-modal reasoning are proposed (Zellers et al., 2019; Park et al., 2020; Dong et al., 2022; Lu et al., 2022), and among them, ScienceQA (Lu et al., 2022) annotates reasoning process and is the most commonly used visual multi-modal reasoning benchmark. Video multi-modal reasoning (Lei et al., 2020; Yi et al.,  

2020; Wu et al., 2021; Xiao et al., 2021; Li et al., 2022a; Gupta and Gupta, 2022) is more challenging as it introduces additional temporal information compared to visual multi-modal reasoning.  

# 3.6 Metrics  

Accuracy Accuracy is used to assess a model’s ability on classifcation tasks and is commonly used for multi-choice (Ling et al., 2017; Mihaylov et al., 2018; Liu et al., 2020; Lu et al., 2022) and yes/no (Talmor et al., 2021; Geva et al., 2021; Han et al., 2022) tasks.  

$$
\mathrm{Accuracy}={\frac{\mathrm{N}_{\mathrm{correct}}}{\mathrm{N}_{\mathrm{total}}}}
$$  

EM and F1 EM and F1 are metrics used to evaluate free form (Mishra et al., 2022a; Wang et al., 2019; Yi et al., 2020) and span extraction (Dua et al., 2019; Zhu et al., 2021; Mishra et al., 2022b) tasks. Both are calculated at the token level.  

$$
\begin{array}{r}{\mathrm{F}1=\frac{2\cdot\mathrm{P}\cdot\mathrm{R}}{\mathrm{P}+\mathrm{R}}}\\ {\mathrm{EM}=\frac{\displaystyle\sum\mathbb{I}[A=A^{\prime}]}{\mathrm{N}_{\mathrm{total}}}}\end{array}
$$  

where $\mathrm{P}$ and $\mathbf{R}$ stand for precision and recall, and EM calculates the proportion of predictions and answers that are exactly the same.  

# 4 Methods  

In this section, we explore $\Chi$ -of-thought reasoning through three different categorizations: the construction of X-of-thought (§4.1), the structural variants of X-of-thought (§4.2), and the enhanced methods of X-of-thought (§4.3).  

# 4.1 Construction Approach  

After thorough analysis, we divide the construction of X-of-thought into three categories: 1) Manual XoT, 2) Automatic XoT, and 3) Semi-automatic XoT, described as follows.  

# 4.1.1 Manual XoT  

While large language models perform few-shot incontext learning via prompting, they are still limited in reasoning tasks. In order to explore the potential reasoning ability of large language models, one standard approach is to provide different forms of thoughts in demonstrations.  

Wei et al. (2022b) frst propose chain-of-thought prompting (Few-shot CoT) by manually providing natural language form rationales to the demonstrations. To further ensure certainty in the reasoning process and reduce inconsistencies between reasoning path and answers, PAL (Gao et al., 2023), PoT (Chen et al., 2022a) and NLEP (Zhang et al., 2023e) leverage programming language as annotated rationales, which transforms the problemsolving into an executable Python program. Meanwhile, to take both advantages of natural language and programming language and raise the confdence of reasoning output, MathPrompter (Imani et al., 2023) uses the zero-shot chain-of-thought prompting to generate multiple algebraic expressions or Python functions, which can verify each other and improve the reliability of results. Furthermore, since the reasoning complexity of samples in demonstrations, such as chains with more reasoning steps, results in performance improvement, Fu et al. (2023a) proposes complexity-based prompting, where voting among high-complexity rationales is performed to get the fnal answer.  

Manually constructed X-of-thought methods expand on in-context learning by adding different types of step-by-step intermediate reasoning processes to demonstrations. They allow LLMs to mimic and generate reasoning paths. Although manual XoT methods provide greater interpretability as well as trustworthiness for human understanding and outperform on complex tasks, i.e., mathematical reasoning, commonsense reasoning, symbolic reasoning, etc., manual annotating of rationales entails signifcant costs and suffers from drawbacks such as diffculty in demonstration selection and task generalization. Specifcally, different tasks require different ways of demonstrations. Therefore, other works attempt to construct the reasoning path automatically, as discussed in $\S4.1.2$ .  

# 4.1.2 Automatic XoT  

Chain-of-thought prompting (Wei et al., 2022b) elicits the complex reasoning ability of LLMs with task-specifc exemplars in a few-shot setting, which limits the scalability and generalization. To reduce the cost of hand-crafted few-shot exemplars, Kojima et al. (2022) proposes zero-shot CoT by introducing a magic phrase Let’s think step by step after question, which enables LLMs to generate reasoning chains in a zero-shot manner. However, zero-shot CoT suffers from poor-quality reasoning paths, coming with many mistakes. Since the diversity of demonstration plays a vital role in reasoning chains generation, Auto-CoT (Zhang et al.,  

![](images/38d9ab6826a9c705f5ab775708b0b775efad28dbe0cb491ae3a7ba14df794186.jpg)  
Figure 1: XoT Methods, Frontier Application, Future Direction, and Benchmarks.  

2023f) generates the demonstrations automatically via clustering and representative exemplars selection, which improves the diversity and consistently matches or exceeds the performance of Few-shot CoT. COSP (Wan et al., 2023) introduces the outcome entropy of the question to aid demonstration selection. Xu et al. (2023) proposes Reprompting to fnd the effective CoT prompt by employing Gibbs sampling iteratively. Meanwhile, some mistakes in reasoning chains come from missing-step errors, Wang et al. (2023f) extend the zero-shot CoT into Plan-and-Solve (PS) Prompting via devising a plan to divide the entire task into smaller sub-tasks and carrying out the sub-tasks according to the plan with more detailed instructions. LogiCoT (Zhao et al., 2023c) uses symbolic logic to validate the zero-shot reasoning process, thus reducing errors in reasoning. Besides, PoT (Chen et al., 2022a) also explore language models, such as Codex, to generate an executable Python program to solve math problems in zero-shot setting via adding Let’s write a Python program step by step..., which mitigates errors in intermediate reasoning steps. Some work introduces agents to solve reasoning problems. For example, Agent Instruct (Crispino et al., 2023a) utilizes agents to generate task-related, informative instructions, which guides LLMs to perform zero-shot reasoning.  

Unlike manual XoT, automatic XoT, using zeroshot prompt engineering or sampling, is scalable and can be generalized between domains without human intervention. However, due to the lack of human alignment, automatically generated chain-ofthought encounters challenges such as poor quality, hallucinations, and factual inconsistencies. Therefore, constructing XoT in a semi-automatic way is necessary, which is introduced in $\S4.1.3$ .  

# 4.1.3 Semi-automatic XoT  

Semi-automatic XoT methods integrate the advantages of both manual and automatic construction methods. Shao et al. (2023) proposes Synthetic Prompting, which leverages a few humanannotated examples to prompt models to generate more examples through an alternated forwardbackward process and selects effective demonstrations to elicit better reasoning, alleviating the lack of human alignment in AutoCoT. Although previous work solves the problem of manual annotating, demonstration selection can also signifcantly affect performance. Automate-CoT (Shum et al., 2023) employs reinforcement learning with a variance-reduced policy gradient strategy to estimate the signifcance of each example in a blackbox language model, eliciting better demonstration selection. Similarly, Lu et al. (2023b) proposes PromptPG, which utilizes policy gradient to learn to select demonstrations in tabular reasoning. Ye and Durrett (2023) initially uses two proxy metrics to evaluate each example and then searches over examples to fnd demonstrations that yield the best performance in a silver-labeled development set. Meanwhile, Pitis et al. (2023) proposes Boosted Prompting, a prompt ensembling way to improve the performance, which iteratively expands the examples when encountering the problem that the current demonstration is challenging to handle. Zou et al. (2023) introduce Meta-CoT, which automatically selects demonstrations based on the question category, eliminating the need for the task-specifc prompt design.  

The semi-automatic XoT methods reduce the workload of manual labeling while introducing human alignment signals and demonstration selection strategies to enhance the capability and stability of reasoning. Additionally, it enables cost-effective domain generalization. However, the demonstration selection problem has not been entirely resolved and requires more effort and research.  

# 4.2 XoT Structural Variants  

The most primitive chain-of-thought is a chain structure that describes intermediate reasoning steps in natural language. In this section, we introduce structural variants that modify the original chain structure, including chain structure variants, tree structure variants, and graph structure variants.  

Chain Structure PAL (Gao et al., 2023) and PoT (Chen et al., 2022a) introduce programming languages to describe the reasoning process, thereby converting the reasoning problem into the implementation of an executable program to obtain the fnal answer. Since the program execution is deterministic and performs arithmetic computations accurately, this approach shows excellent performance in mathematical reasoning. Besides, symbol sequence is another type of thought representation. Chain-of-Symbol (Hu et al., 2023a) represents the complex environments with condensed symbolic chain representations during planning, which reduces the complexity of the simulation environment. Chain structure variants are shown in Figure 2(c,d) Algorithm of Thought (Sel et al., 2023) injects algorithmic capabilities into the model, making the model’s reasoning more logical by adding examples based on algorithms. Its absence of the huge search space of tree search (Long, 2023; Yao et al., 2023b) saves computational resources and achieves excellent performance.  

Tree Structure The original chain structure inherently limits the scope of exploration. Through the incorporation of tree structures and tree search algorithms, models gain the capability to effciently explore and backtrack during the reasoning process (Long, 2023; Yao et al., 2023b), as shown in Figure 2(e). Combined with self-assessment of intermediate thoughts, models can achieve global optimum solutions. The reasoning process of ToT involves uncertainty, which can potentially lead to cascading errors. TouT (Mo and Xin, 2023) introduces Monte Carlo Dropout in reasoning, taking into account the uncertainty. Yu et al. (2023b) delves into analogous problems, harnessing their solutions to elevate the intricate reasoning abilities of LLMs. These analogous problems exhibit a tree-like structure, ultimately converging to solve the main problem. However, the current tree-ofthought has considerable limitations on task selection and requires specifc prompt designing for each task, which hinders its widespread application. SoT (Ning et al., 2023) is another variant of the tree structure, which decomposes a problem into subproblems that can be processed in parallel and solved simultaneously to speed up reasoning. However, its utility is restricted to parallel decomposable problems and is not suited for complex reasoning tasks.  

Graph Structure Compared to trees, graphs introduce loops and rings, which bring more complex topological relationships and allow for modeling more complex reasoning, as shown in Figure 2(f). GoT (Besta et al., 2023; Lei et al., 2023a) regards intermediate thought as nodes within a graph, combining exploration and backtracking operations, and additionally introduces aggregation and refnement operations compared to treeof-thought. The additional operations, aggregation and refnement elicit better reasoning in complex tasks. Nevertheless, it faces the same dilemmas as the tree-of-thought, i.e., task limitations and poor generalizability. Besides, it has increased reasoning costs. Unlike GoT, which explicitly constructs a thought graph, ResPrompt (Jiang et al., 2023a) introduces residual connections between thoughts in the prompt text, allowing the reasoning of different steps to interact with each other.  

![](images/a9fbf34b7e59b5848e3e896cef8ba7c7db69f15020a9c4b7b6556640b0fc2b1d.jpg)  
Figure 2: The evolution of reasoning, from direct I/O to chain structure, then to tree and graph structure.  

As models transition from linear chains to hierarchical trees and intricate graphs, the interplay of thoughts becomes progressively more complex, thereby gradually enhancing the capacity to address intricate problems. However, as the complexity of the topology increases, associated methods impose more constraints on task selection, leading to a signifcant reduction in their generalizability and making their application diffcult. Extending complex topology structure-based methods to general domains is a major challenge for future research.  

# 4.3 XoT Enhancement Methods  

In this section, we present the XoT enhancement methods. In total, we will provide an overview of fve categories, which are adding verifcation and refnement (§4.3.1), question decomposition $(\S4.3.2)$ , leveraging external knowledge $(\S4.3.3)$ , voting and ranking (§4.3.4), and improving effciency (§4.3.5).  

# 4.3.1 Verify and Refne  

Chain-of-thought reasoning often tends to be hallucinatory, producing incorrect reasoning steps. Errors in intermediate reasoning steps can, in turn, trigger a cascade of errors. Incorporating verifcation to obtain feedback and subsequently refning the reasoning process based on this feedback can be a highly effective strategy for mitigating this phenomenon, which is similar to the process of human refection. Figure 3 depicts the overview of verifcation and refnement.  

![](images/7ebc2a1b332bbbf9a60f05ad9e3e44aa01bb79fb1b00fa0e9e17bc9b1a05c031.jpg)  
Figure 3: Verifcation and refnement reduce cascading errors in reasoning.  

VerifyCoT (Ling et al., 2023) devises a Natural Program, a deductive reasoning form, which allows models to produce accurate reasoning steps, with each subsequent step strictly based on the previous steps. DIVERSE (Li et al., 2022c) utilizes a voting mechanism to eliminate incorrect answers, followed by a fne-grained verifcation of each reasoning step independently. SCREWS(Shridhar et al., 2023) thinks that the post-modifcation result may not necessarily be superior to the origin, so it introduces a selection module to select a better result between the origin and modifcation. To facilitate knowledge-intensive tasks, Verify-and-Edit (Zhao et al., 2023a) incorporates external knowledge to re-reason uncertain examples, reducing factual mistakes in reasoning. Some research efforts attempt to unearth the internal knowledge of models. Some research efforts attempt to unearth the internal knowledge of models. To address factual errors, some research attempts to unearth the intrinsic knowledge of LLMs. They acquire knowledge from the model before answering the questions (Dhuliawala et al., 2023; Zheng et al., 2023). Ji et al. (2023) further verifes the correctness of intrinsic knowledge, and Liu et al. (2023b) enhances the accuracy of intrinsic knowledge acquisition through reinforcement learning.  

Inconsistency is another major challenge in reasoning, Dua et al. (2022) iteratively uses previous reasoning results as prompts until the model gives a consistent answer. Paul et al. (2023) trains a critic model to provide structured feedback on the reasoning process. Self-Refne (Madaan et al., 2023) performs iterative self-feedback and refnement to alleviate errors in reasoning. Compared with Self-Refne, Refexion (Shinn et al., 2023) introduces reinforcement learning for refection, which additionally brings decision-making capability. Meanwhile, some work introduces backward reasoning (Yu et al., 2023a) for verifcation. RCoT (Xue et al., 2023) reconstructs the question according to the reasoning chains, and its inconsistency with the original question exposes errors in the reasoning process. FOBAR (Jiang et al., 2023b) and Self Verifcation (Weng et al., 2022) perform verifcation by deducing the conditions in the question from the answer. FOBAR infers the variables in the question, and Self Verifcation infers the conditions in the question. However, Huang et al. (2023a) fnds that LLMs struggle to self-correct without external feedback, and it could even lead to a performance decline.  

LLM reasoning is an unsupervised process in which feedback signals from intermediate reasoning steps play a crucial role in improving reasoning. Guidance from feedback signals can effectively reduce the hallucination phenomena in reasoning. There is still signifcant research space for obtaining appropriate feedback and making accurate corrections based on that feedback.  

# 4.3.2 Question Decomposition  

The essence of X-of-thought reasoning lies in its step-by-step problem-solving. However, the original chain-of-thought reasoning approach does not explicitly strip out the step-by-step reasoning process and still uses one-stage generation. In this section, we discuss the question decomposition approach, which explicitly solves questions stepby-step. The overview is shown in Figure 4.  

![](images/c10d877b7abc97775244a21e442828efe04e03fff6f4b591b91b7b659ae0a2ce.jpg)  
Figure 4: Question decomposition solves complex questions progressively by solving simple sub-questions.  

Wang et al. (2022a) iteratively acquires knowledge from the model, making progress in multi-hop QA. Zhou et al. (2023b) proposes Least-to-Most Prompting, which initially breaks down the question into sub-questions in a top-down fashion, and subsequently, it solves a sub-question once at a time and leverages their solutions to facilitate subsequent sub-questions. Successive Prompting (Dua et al., 2022) takes a similar approach to Least-toMost Prompting, and the difference is that it takes a decomposition with interleaved sub-questions and answers rather than two-stage decomposition. The above methods do not formulate tailored solutions for various sub-problems. Decomposed Prompting (Khot et al., 2023) designs a modular shared library, each dedicated to a class of subproblems, which can tailor more effective solutions to different classes of sub-problems. Apart from general tasks, some works focus on question decomposition on tabular reasoning. BINDER(Cheng et al., 2023) maps reasoning to a program in a neural-symbolic manner and obtains the fnal answer through a program executor such as Python or SQL. Ye et al. (2023) introduces DATER, which breaks down large tables into smaller ones and complex questions into simpler ones. The former reduces irrelevant information, while the latter reduces the complexity of reasoning.  

Providing direct answers to complex questions can be challenging. By decomposing the question into simple sub-questions and solving them stepby-step, the diffculty is reduced. Moreover, each sub-question can be traced back to a specifc reasoning step, making the reasoning process more transparent and explainable. Current work mostly uses top-down decomposition strategies, while bottomup decomposition strategies based on backward reasoning remain to be explored in future work.  

![](images/8573cf6284ceda2de866c85af2418c82a4b5a5dd6ca43c741915841250ba22a1.jpg)  
Figure 5: Introducing external knowledge reduces factual errors in reasoning.  

The parameterized knowledge within models is limited and outdated. Thus, factual mistakes often occur when facing knowledge-intensive tasks. Introducing external knowledge can mitigate this phenomenon, as shown in Figure 5.  

Lu et al. (2023a) introduces multilingual dictionaries in prompts to enhance machine translation. Li et al. (2023d) proposes chain-of-knowledge (CoK-Li), which obtains structured knowledge from a knowledge base via a query generator to perform knowledge-guided reasoning. Wang et al. (2023b) (CoK-Wang) also retrieves structured knowledge from KB. Moreover, it estimates the reasoning chains in terms of factuality and faithfulness and prompts models to rethink unreliable reasonings, which mitigates the knowledge retrieval errors in CoK-Li. KD-CoT (Wang et al., 2023c) addresses factual reasoning problems through a multi-turn QA approach. They design a feedbackaugmented retriever for retrieving relevant external knowledge in each round of QA to calibrate the reasoning process. Other studies use the model’s own memory as external knowledge. For example, Memory-of-Thought (Li and Qiu, 2023) frst performs pre-thinking to save the high-confdence thoughts into external memory, and during inference, it lets the LLM recall relevant memory to aid reasoning.  

The parameterized knowledge in the model is fxed at the end of the pre-training, which leads to its shortcomings in terms of knowledge capacity and knowledge updating. While introducing external knowledge can alleviate this to some extent, it remains an imperfect solution. To fundamentally tackle this issue, continual learning (Lange et al., 2022; Wang et al., $2023\mathrm{g}$ ) stands as a promising avenue for future research endeavors.  

![](images/737600f4f182ccad44e2d1e8742ba8df1b331e67c0f6f5d2763cc319d4b6f073.jpg)  
Figure 6: Voting and ranking reduce inconsistency by selecting fnal answers from multiple samplings.  

Owing to the inherent stochasticity in the generation process, LLM reasoning exhibits an element of randomness and uncertainty. This problem can be effectively alleviated through multiple sampling strategies, as shown in Figure 6.  

Some methods adopt ranking, such as (Cobbe et al., 2021), which trains a verifer to select high-confdence reasoning chains through ranking. Meanwhile, other methods select reasoning chains through a voting mechanism. Selfconsistency (Wang et al., 2023j) selects the most consistent answer by majority voting among sampled reasoning chains based on fnal answers. Furthermore, (Fu et al., 2023a) proposes Complex CoT, which utilizes a complexity-based voting strategy that leans towards selecting answers generated by more complex reasoning chains. However, answer-based voting mechanisms do not take into account the correctness of reasoning chains. Miao et al. (2023) takes the reasoning steps into account when voting, which can obtain both consistent answers and trustworthy reasoning processes simultaneously. Moreover, to consider the relations between intermediate steps across chains, Yoran et al. (2023) mixes information between reasoning chains and selects the most relevant facts to perform meta-reason over multiple reasoning chains. GRACE(Khalifa et al., 2023) trains a discriminator through contrastive learning and uses this discriminator to rank each intermediate reasoning step. Previous methods sample based on the probability distribution, while Diversity-of-Thought (Naik et al., 2023) obtains multiple reasoning paths by prompting with different instructions.  

Drawing inspiration from ensemble learning, the practice of voting and ranking following with multiple sampling serves to diminish uncertainty. Furthermore, it has showcased substantial performance improvements compared to the single-sample approach. Multiple sampling with voting has become a common technique in current X-of-thought studies. Integrating reasoning chains into voting remains a signifcant area of research for the future.  

# 4.3.5 Effciency  

LLM reasoning and manually annotated reasoning chains impose expensive overheads. Aggarwal et al. (2023) improves self-consistency by dynamically adjusting the number of samples, which can signifcantly reduce inference costs with marginal performance degradation. Ning et al. (2023) decomposed the questions in parallel and handled them simultaneously, reducing the reasoning time overhead. But it cannot handle complex questions. Zhang et al. (2023b) accelerates the reasoning by selectively skipping some intermediate layers and then verifes the draft in another forward pass. Diao et al. (2023) borrows ideas from active learning to annotate examples with high uncertainty, reducing the human annotating cost.  

Large-scale language models have showcased immense capabilities, but they also come with substantial overhead. Balancing the trade-off between performance and overhead may require signifcant attention in future research endeavors.  

# 5 Frontier Application  

# 5.1 Tool Use  

Despite the extensive knowledge exhibited by LLMs, it is accompanied by several challenges. These encompass the incapacity to access upto-the-minute news, proclivity towards hallucinations when responding to queries involving out-ofdomain knowledge, and the absence of sophisticated reasoning capacities like mathematical calculations or symbolic reasoning. By granting LLMs the ability to employ external tools, it becomes possible to augment the model’s reasoning capabilities and assimilate external knowledge, enabling it to engage in information retrieval and environmental interaction.  

MRKL (Karpas et al., 2022) introduces a novel framework comprising scalable modules (referred to as experts) and a router. These experts can take the form of neural networks or symbols. However, this study primarily focuses on conceptualization and training an LLM specifcally for mathematical computation while not delving into implementing other module contents. TALM (Parisi et al.,  

2022a) and Toolformer (Schick et al., 2023) integrate a text-centric methodology with supplementary tools to enhance the capabilities of language models. They employ a self-supervise mechanism to initiate performance enhancements, commencing with a limited set of tooltips. In a similar vein, HuggingGPT (Shen et al., 2023) leverages visual and speech models to process information from diverse modalities, thereby endowing LLMs with the capacity for multi-modal understanding and generation. Another question is how to select the appropriate tool. LATM (Cai et al., 2023) enables the tool-making ability of LLMs to make generalized API across different tasks, and GEAR (Lu et al., 2023c) considers the effciency of tool-using by using smaller models to delegate tool grounding and execution.  

However, converting a user request into API format is often not straightforward. The existing approaches mentioned above have limitations in facilitating multiple invocations of the tool and rectifying query errors. To tackle this problem, ReAct (Yao et al., 2023c) integrates the strengths of reasoning and action to enhance and complement each other, augmenting problem-solving capability mutually. ART (Paranjape et al., 2023) uses a task library to select relevant tool usage and reasoning chains. MM-REACT (Yang et al., 2023) further utilizes vision experts to enable multi-modal reasoning and action.  

The aforementioned research endeavors focus on designing tools (or APIs) to enhance the capabilities of LLMs in various domains. Combining XoT with tools effectively addresses the challenges faced by LLMs. X-of-thought reasoning enables models to effectively elicit, track, and update action plans while managing exceptions. Simultaneously, action operations facilitate the model’s interaction with external sources, such as knowledge bases and environments, enabling it to gather additional information. To assess the profciency of tools, APIBank (Li et al., 2023c) and MetaTool (Huang et al., 2023c) introduce comprehensive benchmarks, providing a robust foundation to evaluate the performance and effectiveness of tool-augmented LLMs.  

# 5.2 Planning  

LLMs face challenges in providing accurate responses directly for intricate problems, necessitating the need to decompose them into sequential steps and sub-tasks. While CoT offers a straightforward approach to planning, it falls short in addressing highly complex problems and lacks the ability to evaluate and rectify errors through backtracking.  

Numerous studies have extended the framework of chain-of-thought to various formats to enhance the capacity for planning further. Treeof-Thought (Yao et al., 2023b) enables LLMs to consider multiple reasoning paths in a tree and self-evaluate to determine the next course of action. In cases where global decisions are necessary, ToT allows forward or backward exploration through techniques like deep-frst search or breadthfrst search. Reasoning via Planning (RAP) (Hao et al., 2023) also divides the problem into a tree and explores them by Monto Carlo tree search algorithm, using LLMs as both world-model and reasoning agent. Another method, Graph of Thought (GoT) (Yao et al., 2023d), employs graph nodes to represent individual thoughts and external Graph Neural Networks for organization. ${\mathrm{LLM}}+{\mathrm{P}}$ (Liu et al., 2023a) and $\mathrm{LLM+DP}$ (Dagan et al., 2023) facilitate the generation of Planning Domain Defnition Language (PDDL) (Gerevini, 2020) by LLMs. PDDL assists in decomposing complex problems and utilizing specialized models for planning before converting the results into natural language for LLM processing. However, it is essential to note that these methods use tree/graph/PDDL nodes to represent thoughts, which have limitations regarding their representation forms and can only handle specifc planning problems.  

Another technique is to improve the model’s ability to correct errors and summarize historical experience. Self-Refne (Madaan et al., 2023) employs a unique approach where the output generated by the model is evaluated and provided with feedback using the same model. Refexion (Shinn et al., 2023) enables the model to refect on and rectify errors made in previous actions, resembles reinforcement learning in textual format, and involves dividing memory into long and short-term components. However, Refexion cannot update the plan when an out-of-plan error occurs. AdaPlanner (Sun et al., 2023) introduces adaptive closed-loop plan refnement, which iterative refnes the task plan based on the feedback of the environment. ISRLLM (Zhou et al., 2023c) combines Self-Refne with PDDL to achieve a better success rate in longhorizon sequential tasks. Meanwhile, LATS (Zhou et al., 2023a) utilizes LM-based Monte Carlo Tree Search for a more fexible planning procedure.  

Planning can be fexibly combined with tools (Ruan et al., 2023) or agents (Crispino et al., 2023b) to enrich reasoning ability. ToRA (Gou et al., 2023) designs mathematical specialized agents with external tools, and AutoUI (Zhang and Zhang, 2023) directly interacts with the multi-modal environment instead of converting visual inputs into text, which enhances the reasoning effciency and reduces error propagation.  

Planning augmented approaches have advanced conventional sequential planning by introducing search-based, graph-based, and defnition languagebased methods. On the other hand, some methods incorporate action, planning, refection, or tools, aiming to enhance LLMs’ long-term planning and error resilience capabilities.  

# 5.3 CoT Distillation  

LLM can be self-improved by distilling reasoning steps to solve complex problems. Huang et al. (2022) employs an LLM with self-consistency to generate reasoning chains from unlabeled data. These chains are subsequently utilized to fne-tune the model, enhancing its generalized reasoning capabilities. Zelikman et al. (2022) proposes STaR, a few-shot learning approach to improve LM’s reasoning capabilities using a self-loop bootstrap strategy. SECToR (Zhang and Parkes, 2023) uses chainof-thought to obtain arithmetic answers, then fnetune the model to generate the answer without CoT directly.  

Thought CoT is an emerging ability primarily observed in LLMs, with limited advancements in small models. However, enhancing small models’ CoT ability is conceivable through techniques like distillation. Magister et al. (2023) demonstrates that fne-tuning T5 with reasoning chains generated by larger teacher models and utilizing an external calculator for answer resolution can substantially enhance task performance across diverse datasets. Ho et al. (2023) generates and flters multiple reasoning paths to enrich the diversity.  

Numerous endeavors can be undertaken to reduce human costs using unannotated (or very few annotated) data by utilizing the selfconsistency (Wang et al., 2023j). Hsieh et al. (2023) employs prompts to generate answers from much fewer labeled/unlabeled data, followed by the generation of rationales that prompt the language model to provide reasoning for the given answer. SCoTD (Li et al., 2023b) fnds that sampling multiple reasoning chains per instance from teachers is paramount for improving the capability of students. SCOTT (Wang et al., 2023h) utilizes contrastive decoding (Li et al., 2022b; O’Brien and Lewis, 2023) during rationale generation for teacher models. Furthermore, to tackle the shortcut problem, it employs a counterfactual reasoning objective while training student models. DialCoT (Han et al., 2023) decomposes reasoning steps into a multi-round dialog and selects the correct path using the PPO algorithm. Jie et al. (2023); Wang et al. (2023i) add special tokens for mathematic problems. This high-level information improves the consistency of reasoning steps.  

The studies above adopt a shared paradigm wherein reasoning chains are generated through LLMs possessing superior reasoning capabilities. These reasoning chains are then distilled into smaller models. The effectiveness of the distillation process is improved by augmenting the sampling strategy from the larger model, for example, through the utilization of multiple sampling paths, consistency, or contrastive decoding, which leads to improved diversity and accuracy in the generated reasoning chains, ultimately benefting the distillation process to smaller models. It’s notable that language models have intricate tradeoffs and complex balances associated with multidimensional capabilities. Fu et al. (2023b) emphasizes that increasing task-specifc chain-of-thought capabilities through distillation may also adversely impact the models’ performance in solving generalized problems.  

# 6 Future Directions  

While chain-of-thought reasoning has showcased remarkable performance on numerous tasks, some challenges still require further exploration. In this section, we provide a concise overview of three promising avenues for future research: multimodal X-of-thought reasoning $(\S6.1)$ , faithful Xof-thought reasoning (§6.2), and X-of-thought reasoning theory (§6.3).  

# 6.1 Multi-modal CoT  

The shift from text unimodal to vision-text multimodal introduces richer information, meanwhile bringing more challenges. Some works have attempted to explore X-of-thought reasoning in multimodal scenarios by fne-tuning multi-modal models to generate a high-quality chain of thoughts.  

Multimodal-CoT (Zhang et al., $2023\mathrm{g}$ ) frstly fne-tunes multi-modal models to generate chainof-thoughts and then reasons over the rationales to obtain fnal answers. However, it suffers from the limitation of the linearity of the reasoning process and has diffculties in interacting between different modalities. To alleviate the challenges encountered by Multimodal-CoT, (Yao et al., 2023d) proposes Graph-of-Thought (GoT), which models the thought processes as a graph. It parses the reasoning chains into a thought graph, which enables a more realistic representation of thought processes by capturing non-sequential information interactions. This measure breaks the limitations of linear structure through graphical structures and further improves performance. Furthermore, Yao et al. (2023a) proposes Hypergraph-of-Thought (HoT), replacing thought graphs with hypergraphs, which enables models with better ability of highorder multi-hop reasoning and multi-modal comparative judgment. Meanwhile, some work takes an approach based on knowledge distillation. TSciQ (Wang et al., 2023d) generates high-quality CoT rationales from LLMs as fne-tuning signals and introduces a novel data mixing strategy to produce effective samples for different questions.  

The aforementioned studies explore multi-modal reasoning in small models and fne-tuning scenarios, which we regard as an initial endeavor in the realm of multi-modal chain-of-thought reasoning. We believe that video multi-modal reasoning combined with in-context learning should be the focus of future research. On the one hand, videos introduce additional temporal information with innate chaining relationships compared with images. Through chain-of-thought reasoning, the information in different frames can be naturally connected to explicitly model the temporal relationship, which is well-suited for video multi-modal reasoning. On the other hand, small models are capacity-limited and need fne-tuning to gain chainof-thought ability. Worse still, multi-modal reasoning chains are diffcult to obtain, which further exacerbates the challenge. In comparison, contemporary vision-language foundation models (VLMs) (Alayrac et al., 2022; Li et al., 2023a; Wang et al., 2022b; Huang et al., 2023b; Peng et al., 2023; Yu et al., 2021b) have strong visionlanguage comprehension and are already capable of in-context learning with interleaved text and images. They provide a solid foundation for chain-ofthought reasoning with in-context learning. Utilizing chain-of-thought for video reasoning remains an unexplored territory with only a few studies. CoMT (Hu et al., 2023b) combines fast-thinking and slow-thinking in video reasoning and introduces a tree search strategy for planning, which frstly applies CoT in video multi-modal reasoning.  

Although some works have started to utilize chain-of-thought reasoning and solve multi-modal reasoning tasks, previous works only focus on how to construct high-quality fne-tuned data, and there are still several challenges remaining:  

• How to unify visual and language features to elicit better multi-modal understanding. • How to use VLMs for chain-of-thought reasoning without fne-tuning. • How to adapt image multi-modal reasoning into video multi-modal reasoning.  

# 6.2 Faithfulness  

Extensive research indicates that chain-of-thought reasoning can lead to hallucination phenomena, such as factual mistakes and contextual inconsistencies. Considering that language models fundamentally belong to statistical models, and due to factors such as data noise and knowledge forgetting, hallucination phenomena are unavoidable.  

Some works focus on mitigating factual mistakes. He et al. (2023a) introduces external knowledge to evaluate reasoning chains and votes to flter out chains that contain factual mistakes but without correcting them Wang et al. (2023b) adopts a similar way, with the difference that it additionally introduces a refection mechanism to correct low-scoring reasoning. Zhao et al. (2023a) flters out low-confdence reasoning by consistency and guides models to re-reasoning based on relevant external knowledge. While the aforementioned methods work well on knowledge-intensive tasks, they fall short in addressing the challenge of contextual inconsistencies. Zhang et al. (2023d) explores the hallucination snowballing phenomena during the reasoning process. Others aim to address the inconsistency issues. Radhakrishnan et al. (2023) observes that models are more faithful when dealing with simple questions. Thus, it improves faithfulness through question decomposition. Faithful CoT (Lyu et al., 2023) initially generates symbolic reasoning chains and later deterministically executes symbolic functions, mitigating reasoning inconsistencies. Lanham et al. (2023) explores the factors that infuence faithfulness, which provides an empirical perspective. It fnds faithfulness varies on different tasks and decreases as the model size increases. CoNLI (Lei et al., 2023b) proposes a post-editing strategy to diminish the hallucinations. SynTra (Jones et al., 2023) performs prefx-tuning on a synthetic dataset designed to elicit hallucination easily, and then transfers this capability to real tasks.  

Despite numerous efforts aimed at addressing the hallucination issues in large language models, these works have only mitigated the problem to some extent. There is still a long way to fully enhance the faithfulness of large language models. We summarize the future directions as follows:  

• Improving the ability to recognize hallucination phenomena in the reasoning processes.   
• Improving the accuracy of external knowledge retrieval and utilization to reduce factual mistakes.   
• Improving the ability to recognize and correct contextual inconsistencies and logical mistakes, which is more challenging.   
• How to fundamentally eliminate hallucination phenomena from alternative approaches, e.g. specifc pre-training.  

# 6.3 CoT Theory  

Despite the impressive capability of chain-ofthought reasoning, the ability to generate chainof-thought following instructions still lacks a comprehensive explanation.  

Some work addresses from an empirical perspective and can serve as a practical guide. Madaan and Yazdanbakhsh (2022) decomposes prompts into three components: symbols, patterns, and text, exploring the impact of CoT through counterfactual prompting. Wang et al. (2023a) analyzes the impact of demonstration selection. They fnd that the correctness of reasoning chains has a negligible effect, while the relevance to the question and correct reasoning order matters. Tang et al. (2023) explores the role of semantics. They fnd that chain-of-thought reasoning relies heavily on semantic knowledge introduced during pre-training and performs poorly in symbolic reasoning.  

Others work analyze theoretically, exploring the underlying principles and internal mechanisms. Li et al. (2023e) deconstructs chain-of-thought reasoning as a multi-step combinatorial function. They demonstrate that chain-of-thought reduces the complexity of in-context learning to tackle complex questions. Feng et al. (2023) theoretically proves that a fxed-size Transformer is suffcient for computational tasks and dynamic planning with chainof-thought. Merrill and Sabharwal (2023) observes that chain-of-thought can boost reasoning ability, with the extent of improvement increasing as the number of intermediate reasoning steps grows. Wu et al. (2023) leverages gradient-based feature attribution methods to explore the impact of chainof-thought on outputs. The results indicate that chain-of-thought exhibits robustness to perturbations and variations in the question. In addition, there are some claims suggesting that the chainof-thought ability stems from code data during the pre-training phase (Madaan et al., 2022; Zhang et al., 2023c), but there is currently no systematic work to substantiate this opinion.  

Current research on chain-of-thought theory is still in its preliminary exploration stage. We summarize future research directions as follows:  

• Explore the sources of chain-of-thought ability to achieve targeted improvements in CoT reasoning.  

• Theoretically analyzing the advantages of chain-of-thought over in-context learning and exploring the boundaries of its capabilities.  

# 7 Discussion  

# 7.1 Comparison of XoT Construction  

There are three main ways of constructing an Xof-thought for existing methods: (1) Manual labeling reasoning chains. (2) Automatic generating reasoning chains by models. (3) Semi-automatic generation with automatic expansion on a small number of manually labeled reasoning chains.  

We observe that the manual construction methods (Wei et al., 2022b; Gao et al., 2023) face similar challenges to in-context learning, i.e., demonstration selection, instruction formatting, etc (Dong et al., 2023). This causes numerous diffculties in its application and hinders the transfer ability across different tasks. Automatic construction methods (Zhang et al., 2023f; Chen et al., 2022a; Xu et al., 2023) lack the guidance of high-quality annotations, resulting in performance defciencies. Benefting from the signals brought by manual annotations, semi-automatic methods (Shum et al., 2023; Shao et al., 2023) can generate high-quality reasoning chains through self-bootstrapping and similar techniques, effectively addressing the challenges faced by previous approaches. While achieving excellent performance, it allows for easy transfer across different tasks.  

# 7.2 Comparison between Verifcation/Refnement and Planning  

Numerous parallels exist between planning methods and verifcation/refnement-based methods, as both rely on feedback from intermediate processes to adjust and refne behavior. The distinction lies in the fact that planning methods encompass decisionmaking, while verifcation/refnement-based methods solely address intermediate errors without delving into higher-level cognitive processes.  

LLM reasoning processes are often hallucinatory, causing factual and logical mistakes. Verify and edit based methods (Ling et al., 2023; Zhao et al., 2023a; Madaan et al., 2023; Shinn et al., 2023) verify the correctness of the reasoning process and refne reasoning step that may cause hallucinatory. Through verifcation and refnement, cascading errors and hallucinatory phenomena in the reasoning process are signifcantly reduced.  

The planning methods (Long, 2023; Yao et al., 2023b,c; Liu et al., 2023a; Shinn et al., 2023) introduce a decision-making process in the reasoning. They evaluate the intermediate reasoning steps to get feedback, and based on the feedback, they engage in exploration and backtracking to achieve superior solutions at a global level. Their specialization lies in handling complex problems, enabling them to achieve remarkable performance, especially when confronted with intricate multi-hop reasoning and planning tasks.  

# 7.3 Compensate for Innate Weaknesses  

LLMs have many inherent limitations when it comes to reasoning, such as the inability to access external information, arithmetic errors, and inconsistent reasoning. These issues can be cleverly circumvented by entrusting specifc responsibilities to dedicated modules or models.  

In response to the models’ limitation in accessing external information, (Li et al., 2023d; Wang et al., 2023b; Lu et al., 2023a; Schick et al., 2023; Karpas et al., 2022; Yoran et al., 2023) utilizes external knowledge resources like knowledge base, search engines, and open-domain questionanswering systems. Some work introduces a calculator to address arithmetic errors (Schick et al.,  

2023; Karpas et al., 2022; Parisi et al., 2022b). Code execution is deterministic, and certain work enhances the consistency of the reasoning process by introducing code executor (Gao et al., 2023; Chen et al., 2022a; Bi et al., 2023; Imani et al., 2023). We believe that employing LLMs as an agent for central planning and reasoning, delegating specifc sub-tasks to dedicated sub-models, is a potential avenue for applying large models in complex scenarios in the future (Wang et al., 2023e; Xi et al., 2023).  

# 7.4 Other Work  

In this chapter, we will list other works that represent early attempts at chain-of-thought reasoning or are designed for specifc domains.  

Katz et al. (2022); Zhang et al. (2022) provide benchmarks and resources. Some work has empirically demonstrated the effectiveness of chainof-thought prompting (Lampinen et al., 2022; Ye and Durrett, 2022; Arora et al., 2023) and Shi et al. (2023) explores multi-lingual CoT reasoning. Other work focuses on specifc domains, such as machine translation (He et al., 2023b), sentiment analysis (Fei et al., 2023), sentence embeddings (Zhang et al., 2023a), summarization (Wang et al., 2023k), arithmetic (Lee and Kim, 2023), and tabular reasoning (Chen, 2023; Jin and Lu, 2023), etc. Besides, some research utilizes specifc pretraining to enhance certain capabilities, such as mathematical reasoning (Lewkowycz et al., 2022; Zhao et al., 2022).  

# 8 Conclusion  

In this paper, we conduct an extensive survey of existing research on X-of-thought reasoning, offering a comprehensive review of the feld. We introduce the concept of generalized chain-of-thought (X-of-Thought) and examine advances in X-ofthought reasoning from various angles. Additionally, we investigate the applications of X-ofthought in cutting-edge domains. Furthermore, we spotlight the current challenges confronting this research and provide future prospects. To the best of our knowledge, this survey represents the frst systematic exploration of chain-of-thought reasoning. Our objective is to furnish researchers interested in chain-of-thought reasoning with a thorough overview, with the hope that this survey will facilitate further research in this area.  

# References  

Pranjal Aggarwal, Aman Madaan, Yiming Yang, and Mausam. 2023. Let’s sample step by step: Adaptiveconsistency for effcient reasoning with llms. CoRR, abs/2305.11860.  

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. 2022. Flamingo: a visual language model for few-shot learning. In NeurIPS.  

Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. 2019. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 2357–2367. Association for Computational Linguistics.  

Simran Arora, Avanika Narayan, Mayee F. Chen, Laurel J. Orr, Neel Guha, Kush Bhatia, Ines Chami, and Christopher Ré. 2023. Ask me anything: A simple strategy for prompting language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.  

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefer. 2023. Graph of thoughts: Solving elaborate problems with large language models. CoRR, abs/2308.09687.  

Sumithra Bhakthavatsalam, Daniel Khashabi, Tushar Khot, Bhavana Dalvi Mishra, Kyle Richardson, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord, and Peter Clark. 2021. Think you have solved direct-answer question answering? try arc-da, the direct-answer AI2 reasoning challenge. CoRR, abs/2102.03315.  

Zhen Bi, Ningyu Zhang, Yinuo Jiang, Shumin Deng, Guozhou Zheng, and Huajun Chen. 2023. When do program-of-thoughts work for reasoning?  

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. PIQA: reasoning about physical commonsense in natural language. In The Thirty-Fourth AAAI Conference on Artifcial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artifcial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational  

Advances in Artifcial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 7432– 7439. AAAI Press.  

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.  

Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. 2023. Large language models as tool makers.  

Wenhu Chen. 2023. Large language models are few(1)- shot table reasoners. In Findings of the Association for Computational Linguistics: EACL 2023, Dubrovnik, Croatia, May 2-6, 2023, pages 1090– 1100. Association for Computational Linguistics.  

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. 2022a. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. CoRR, abs/2211.12588.  

Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. 2023. Theoremqa: A theorem-driven question answering dataset. CoRR, abs/2305.12524.  

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R. Routledge, and William Yang Wang. 2021. Finqa: A dataset of numerical reasoning over fnancial data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 3697–3711. Association for Computational Linguistics.  

Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. 2022b. Convfnqa: Exploring the chain of numerical reasoning in conversational fnance question answering. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 6279–6292. Association for Computational Linguistics.  

Zhoujun Cheng, Tianbao Xie, Peng Shi, Chengzu Li, Rahul Nadkarni, Yushi Hu, Caiming Xiong, Dragomir Radev, Mari Ostendorf, Luke Zettlemoyer,  

Noah A. Smith, and Tao Yu. 2023. Binding language models in symbolic languages. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.  

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifers to solve math word problems. CoRR, abs/2110.14168.  

Nicholas Crispino, Kyle Montgomery, Fankun Zeng, Dawn Song, and Chenguang Wang. 2023a. Agent instructs large language models to be general zeroshot reasoners. arXiv preprint arXiv:2310.03710.  

Nicholas Crispino, Kyle Montgomery, Fankun Zeng, Dawn Song, and Chenguang Wang. 2023b. Agent instructs large language models to be general zeroshot reasoners.  

Gautier Dagan, Frank Keller, and Alex Lascarides. 2023. Dynamic planning with a llm. ArXiv, abs/2308.06391.  

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171–4186. Association for Computational Linguistics.  

Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. 2023. Chain-of-verifcation reduces hallucination in large language models. arXiv preprint arXiv:2309.11495.  

Shizhe Diao, Pengcheng Wang, Yong Lin, and Tong Zhang. 2023. Active prompting with chainof-thought for large language models. CoRR, abs/2302.12246.  

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, and Zhifang Sui. 2023. A survey for in-context learning. CoRR, abs/2301.00234.  

Qingxiu Dong, Ziwei Qin, Heming Xia, Tian Feng, Shoujie Tong, Haoran Meng, Lin Xu, Zhongyu Wei, Weidong Zhan, Baobao Chang, Sujian Li, Tianyu Liu, and Zhifang Sui. 2022. Premise-based multimodal reasoning: Conditional inference on joint textual and visual clues. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 932–946. Association for Computational Linguistics.  

Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner. 2022. Successive prompting for decomposing complex questions. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 1251– 1265. Association for Computational Linguistics.  

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2368–2378, Minneapolis, Minnesota. Association for Computational Linguistics.  

Hao Fei, Bobo Li, Qian Liu, Lidong Bing, Fei Li, and Tat-Seng Chua. 2023. Reasoning implicit sentiment with chain-of-thought prompting. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 1171–1182. Association for Computational Linguistics.  

Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. 2023. Towards revealing the mystery behind chain of thought: a theoretical perspective. CoRR, abs/2305.15408.  

Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. 2023a. Complexity-based prompting for multi-step reasoning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.  

Yao Fu, Hao-Chun Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. 2023b. Specializing smaller language models towards multi-step reasoning. In International Conference on Machine Learning.  

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. PAL: Program-aided language models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 10764–10799. PMLR.  

Alfonso Emilio Gerevini. 2020. An introduction to the planning domain defnition language (PDDL): book review. Artif. Intell., 280:103221.  

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? A question answering benchmark with implicit reasoning strategies. Trans. Assoc. Comput. Linguistics, 9:346–361.  

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu  

Chen. 2023. Tora: A tool-integrated reasoning agent for mathematical problem solving.  

Pranay Gupta and Manish Gupta. 2022. Newskvqa: Knowledge-aware news video question answering. In Advances in Knowledge Discovery and Data Mining - 26th Pacifc-Asia Conference, PAKDD 2022, Chengdu, China, May 16-19, 2022, Proceedings, Part III, volume 13282 of Lecture Notes in Computer Science, pages 3–15. Springer.  

Chengcheng Han, Xiaowei Du, Che Zhang, Yixin Lian, Xiang Li, Ming Gao, and Baoyuan Wang. 2023. Dialcot meets ppo: Decomposing and exploring reasoning paths in smaller language models.  

Simeng Han, Hailey Schoelkopf, Yilun Zhao, Zhenting Qi, Martin Riddell, Luke Benson, Lucy Sun, Ekaterina Zubova, Yujie Qiao, Matthew Burtell, David Peng, Jonathan Fan, Yixin Liu, Brian Wong, Malcolm Sailor, Ansong Ni, Linyong Nan, Jungo Kasai, Tao Yu, Rui Zhang, Shafq R. Joty, Alexander R. Fabbri, Wojciech Kryscinski, Xi Victoria Lin, Caiming Xiong, and Dragomir Radev. 2022. FOLIO: natural language reasoning with frst-order logic. CoRR, abs/2209.00840.  

Shibo Hao, Yilan Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. 2023. Reasoning with language model is planning with world model. ArXiv, abs/2305.14992.  

Hangfeng He, Hongming Zhang, and Dan Roth. 2023a. Rethinking with retrieval: Faithful large language model inference. CoRR, abs/2301.00303.  

Zhiwei He, Tian Liang, Wenxiang Jiao, Zhuosheng Zhang, Yujiu Yang, Rui Wang, Zhaopeng Tu, Shuming Shi, and Xing Wang. 2023b. Exploring humanlike translation strategy with large language models. CoRR, abs/2305.04118.  

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual.  

Namgyu Ho, Laura Schmid, and Se-Young Yun. 2023. Large language models are reasoning teachers. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 14852–14882. Association for Computational Linguistics.  

Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. 2014. Learning to solve arithmetic word problems with verb categorization. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP 2014, October 25-29, 2014, Doha, Qatar, A meeting of SIGDAT, a Special Interest Group of the ACL, pages 523–533. ACL.  

Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander J. Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfster. 2023. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. ArXiv, abs/2305.02301.  

Hanxu Hu, Hongyuan Lu, Huajian Zhang, Wai Lam, and Yue Zhang. 2023a. Chain-of-symbol prompting elicits planning in large langauge models. CoRR, abs/2305.10276.  

Pengbo Hu, Ji Qi, Xingyu Li, Hong Li, Xinqi Wang, Bing Quan, Ruiyu Wang, and Yi Zhou. 2023b. Tree-of-mixed-thought: Combining fast and slow thinking for multi-hop visual reasoning. CoRR, abs/2308.09658.  

Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. 2022. Large language models can self-improve. CoRR, abs/2210.11610.  

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2023a. Large language models cannot self-correct reasoning yet. arXiv preprint arXiv:2310.01798.  

Lifu Huang, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2019. Cosmos QA: machine reading comprehension with contextual commonsense reasoning. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 2391– 2401. Association for Computational Linguistics.  

Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, and Furu Wei. 2023b. Language is not all you need: Aligning perception with language models. CoRR, abs/2302.14045.  

Yue Huang, Jiawen Shi, Yuan Li, Chenrui Fan, Siyuan Wu, Qihui Zhang, Yixin Liu, Pan Zhou, Yao Wan, Neil Zhenqiang Gong, and Lichao Sun. 2023c. Metatool benchmark: Deciding whether to use tools and which to use.  

Shima Imani, Liang Du, and Harsh Shrivastava. 2023. Mathprompter: Mathematical reasoning using large language models. In Proceedings of the The 61st Annual Meeting of the Association for Computational Linguistics: Industry Track, ACL 2023, Toronto, Canada, July 9-14, 2023, pages 37–42. Association for Computational Linguistics.  

Ziwei Ji, Tiezheng Yu, Yan Xu, Nayeon Lee, Etsuko Ishii, and Pascale Fung. 2023. Towards mitigating hallucination in large language models via selfrefection. arXiv preprint arXiv:2310.06271.  

Song Jiang, Zahra Shakeri, Aaron Chan, Maziar Sanjabi, Hamed Firooz, Yinglong Xia, Bugra Akyildiz, Yizhou Sun, Jinchao Li, Qifan Wang, et al. 2023a. Resprompt: Residual connection prompting advances multi-step reasoning in large language models. arXiv preprint arXiv:2310.04743.  

Weisen Jiang, Han Shi, Longhui Yu, Zhengying Liu, Yu Zhang, Zhenguo Li, and James T. Kwok. 2023b. Forward-backward reasoning in large language models for verifcation. CoRR, abs/2308.07758.  

Zhanming Jie, Trung Quoc Luong, Xinbo Zhang, Xiaoran Jin, and Hang Li. 2023. Design of chain-ofthought in math problem solving.  

Ziqi Jin and Wei Lu. 2023. Tab-cot: Zero-shot tabular chain of thought. In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 10259–10277. Association for Computational Linguistics.  

Erik Jones, Hamid Palangi, Clarisse Simões, Varun Chandrasekaran, Subhabrata Mukherjee, Arindam Mitra, Ahmed Awadallah, and Ece Kamar. 2023. Teaching language models to hallucinate less with synthetic tasks. arXiv preprint arXiv:2310.06827.  

Ehud D. Karpas, Omri Abend, Yonatan Belinkov, Barak Lenz, Opher Lieber, Nir Ratner, Yoav Shoham, Hoft Bata, Yoav Levine, Kevin Leyton-Brown, Dor Muhlgay, Noam Rozen, Erez Schwartz, Gal Shachaf, Shai Shalev-Shwartz, Amnon Shashua, and Moshe Tenenholtz. 2022. Mrkl systems: A modular, neurosymbolic architecture that combines large language models, external knowledge sources and discrete reasoning. ArXiv, abs/2205.00445.  

Uri Katz, Mor Geva, and Jonathan Berant. 2022. Inferring implicit relations in complex questions with language models. In Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 2548–2566. Association for Computational Linguistics.  

Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, and Lu Wang. 2023. Discriminator-guided multi-step reasoning with language models. arXiv preprint arXiv:2305.14934.  

Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. 2023. Decomposed prompting: A modular approach for solving complex tasks. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.  

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In NeurIPS.  

Rik Koncel-Kedziorski, Hannaneh Hajishirzi, Ashish Sabharwal, Oren Etzioni, and Siena Dumas Ang.  

2015. Parsing algebraic word problems into equations. Transactions of the Association for Computational Linguistics, 3:585–597.  

Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. 2016. MAWPS: A math word problem repository. In NAACL HLT 2016, The 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, San Diego California, USA, June 12-17, 2016, pages 1152–1157. The Association for Computational Linguistics.  

Andrew K. Lampinen, Ishita Dasgupta, Stephanie C. Y. Chan, Kory W. Mathewson, Mh Tessler, Antonia Creswell, James L. McClelland, Jane Wang, and Felix Hill. 2022. Can language models learn from explanations in context? In Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 537–563. Association for Computational Linguistics.  

Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Ales Leonardis, Gregory G. Slabaugh, and Tinne Tuytelaars. 2022. A continual learning survey: Defying forgetting in classifcation tasks. IEEE Trans. Pattern Anal. Mach. Intell., 44(7):3366–3385.  

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Lukosiute, Karina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfeld-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. 2023. Measuring faithfulness in chainof-thought reasoning. CoRR, abs/2307.13702.  

Soochan Lee and Gunhee Kim. 2023. Recursion of thought: A divide-and-conquer approach to multicontext reasoning with language models. In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 623–658. Association for Computational Linguistics.  

Bin Lei, Pei-Hung Lin, Chunhua Liao, and Caiwen Ding. 2023a. Boosting logical reasoning in large language models through a new framework: The graph of thought. CoRR, abs/2308.08614.  

Deren Lei, Yaxi Li, Mingyu Wang, Vincent Yun, Emily Ching, Eslam Kamal, et al. 2023b. Chain of natural language inference for reducing large language model ungrounded hallucinations. arXiv preprint arXiv:2310.03951.  

Jie Lei, Licheng Yu, Tamara L. Berg, and Mohit Bansal. 2020. What is more likely to happen next? videoand-language future event prediction. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online,  

November 16-20, 2020, pages 8769–8784. Association for Computational Linguistics.  

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. Solving quantitative reasoning problems with language models. In NeurIPS.  

Jiangtong Li, Li Niu, and Liqing Zhang. 2022a. From representation to reasoning: Towards both evidence and commonsense reasoning for video questionanswering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 21241– 21250. IEEE.  

Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. 2023a. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 19730–19742. PMLR.  

Liunian Harold Li, Jack Hessel, Youngjae Yu, Xiang Ren, Kai-Wei Chang, and Yejin Choi. 2023b. Symbolic chain-of-thought distillation: Small models can also "think" step-by-step. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 2665–2679. Association for Computational Linguistics.  

Minghao Li, Feifan Song, Bowen Yu, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. 2023c. Apibank: A benchmark for tool-augmented llms. ArXiv, abs/2304.08244.  

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2022b. Contrastive decoding: Open-ended text generation as optimization. In Annual Meeting of the Association for Computational Linguistics.  

Xiaonan Li and Xipeng Qiu. 2023. Mot: Pre-thinking and recalling enable chatgpt to self-improve with memory-of-thoughts. CoRR, abs/2305.05181.  

Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng Ding, Lidong Bing, Shafq R. Joty, and Soujanya Poria. 2023d. Chain of knowledge: A framework for grounding large language models with structured knowledge bases. CoRR, abs/2305.13269.  

Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, B. Chen, Jian-Guang Lou, and Weizhu Chen. 2022c. Making language models better reasoners with step-aware verifer. In Annual Meeting of the Association for Computational Linguistics.  

Yingcong Li, Kartik Sreenivasan, Angeliki Giannou, Dimitris S. Papailiopoulos, and Samet Oymak. 2023e. Dissecting chain-of-thought: A study on compositional in-context learning of mlps. CoRR, abs/2305.18869.  

Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. 2017. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers, pages 158–167. Association for Computational Linguistics.  

Zhan Ling, Yunhao Fang, Xuanlin Li, Zhiao Huang, Mingu Lee, Roland Memisevic, and Hao Su. 2023. Deductive verifcation of chain-of-thought reasoning. CoRR, abs/2306.03872.  

Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. 2023a. $\mathrm{Llm}{+}\mathrm{p}$ : Empowering large language models with optimal planning profciency.  

Jiacheng Liu, Ramakanth Pasunuru, Hannaneh Hajishirzi, Yejin Choi, and Asli Celikyilmaz. 2023b. Crystal: Introspective reasoners reinforced with selffeedback. arXiv preprint arXiv:2310.04921.  

Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. 2020. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. In Proceedings of the TwentyNinth International Joint Conference on Artifcial Intelligence, IJCAI 2020, pages 3622–3628. ijcai.org.  

Jieyi Long. 2023. Large language model guided tree-ofthought. CoRR, abs/2305.08291.  

Hongyuan Lu, Haoyang Huang, Dongdong Zhang, Haoran Yang, Wai Lam, and Furu Wei. 2023a. Chainof-dictionary prompting elicits translation in large language models. CoRR, abs/2305.06575.  

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS.  

Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. 2023b. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.  

Yining Lu, Haoping Yu, and Daniel Khashabi. 2023c. Gear: Augmenting language models with generalizable and effcient tool resolution.  

Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. 2023. Faithful chain-ofthought reasoning. CoRR, abs/2301.13379.  

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder, Shashank Gupta, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refne: Iterative refnement with self-feedback. CoRR, abs/2303.17651.  

Aman Madaan and Amir Yazdanbakhsh. 2022. Text and patterns: For effective chain of thought, it takes two to tango. CoRR, abs/2209.07686.  

Aman Madaan, Shuyan Zhou, Uri Alon, Yiming Yang, and Graham Neubig. 2022. Language models of code are few-shot commonsense learners. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 1384–1403. Association for Computational Linguistics.  

Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adámek, Eric Malmi, and Aliaksei Severyn. 2023. Teaching small language models to reason. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 1773–1781. Association for Computational Linguistics.  

William Merrill and Ashish Sabharwal. 2023. The expresssive power of transformers with chain of thought.  

Ning Miao, Yee Whye Teh, and Tom Rainforth. 2023. Selfcheck: Using llms to zero-shot check their own step-by-step reasoning. arXiv preprint arXiv:2308.00436.  

Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2020. A diverse corpus for evaluating and developing English math word problem solvers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 975–984, Online. Association for Computational Linguistics.  

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, Brussels, Belgium. Association for Computational Linguistics.  

Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, and Ashwin Kalyan. 2022a. LILA: A unifed benchmark for mathematical reasoning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 5807–5832. Association for Computational Linguistics.  

Swaroop Mishra, Arindam Mitra, Neeraj Varshney, Bhavdeep Singh Sachdeva, Peter Clark, Chitta Baral, and Ashwin Kalyan. 2022b. Numglue: A suite of fundamental yet challenging mathematical reasoning tasks. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 3505–3523. Association for Computational Linguistics.  

Shentong Mo and Miao Xin. 2023. Tree of uncertain thoughts reasoning for large language models. CoRR, abs/2309.07694.  

Ranjita Naik, Varun Chandrasekaran, Mert Yuksekgonul, Hamid Palangi, and Besmira Nushi. 2023. Diversity of thought improves reasoning abilities of large language models. arXiv preprint arXiv:2310.07088.  

Xuefei Ning, Zinan Lin, Zixuan Zhou, Huazhong Yang, and Yu Wang. 2023. Skeleton-of-thought: Large language models can do parallel decoding. CoRR, abs/2307.15337.  

Sean O’Brien and Mike Lewis. 2023. Contrastive decoding improves reasoning in large language models. ArXiv, abs/2309.09117.  

OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774.  

Bhargavi Paranjape, Scott Lundberg, Sameer Singh, Hannaneh Hajishirzi, Luke Zettlemoyer, and Marco Tulio Ribeiro. 2023. Art: Automatic multistep reasoning and tool-use for large language models.  

Aaron Parisi, Yao Zhao, and Noah Fiedel. 2022a. Talm: Tool augmented language models. ArXiv, abs/2205.12255.  

Aaron Parisi, Yao Zhao, and Noah Fiedel. 2022b. TALM: tool augmented language models. CoRR, abs/2205.12255.  

Jae Sung Park, Chandra Bhagavatula, Roozbeh Mottaghi, Ali Farhadi, and Yejin Choi. 2020. Visualcomet: Reasoning about the dynamic context of a still image. In Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part V, volume 12350 of Lecture Notes in Computer Science, pages 508–524. Springer.  

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, pages 2080–2094. Association for Computational Linguistics.  

Debjit Paul, Mete Ismayilzada, Maxime Peyrard, Beatriz Borges, Antoine Bosselut, Robert West, and Boi Faltings. 2023. REFINER: reasoning feedback on intermediate representations. CoRR, abs/2304.01904.  

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. 2023. Kosmos-2: Grounding multimodal large language models to the world. CoRR, abs/2306.14824.  

Silviu Pitis, Michael R. Zhang, Andrew Wang, and Jimmy Ba. 2023. Boosted prompt ensembles for large language models. CoRR, abs/2304.05970.  

Shuofei Qiao, Yixin Ou, Ningyu Zhang, Xiang Chen, Yunzhi Yao, Shumin Deng, Chuanqi Tan, Fei Huang, and Huajun Chen. 2023. Reasoning with language model prompting: A survey. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 5368– 5393. Association for Computational Linguistics.  

Alec Radford and Karthik Narasimhan. 2018. Improving language understanding by generative pretraining.  

Ansh Radhakrishnan, Karina Nguyen, Anna Chen, Carol Chen, Carson Denison, Danny Hernandez, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Lukosiute, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Sam McCandlish, Sheer El Showk, Tamera Lanham, Tim Maxwell, Venkatesa Chandrasekaran, Zac Hatfeld-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. 2023. Question decomposition improves the faithfulness of model-generated reasoning. CoRR, abs/2307.11768.  

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unifed text-to-text transformer. J. Mach. Learn. Res., 21(1).  

Hannah Rashkin, Maarten Sap, Emily Allaway, Noah A. Smith, and Yejin Choi. 2018. Event2mind: Commonsense inference on events, intents, and reactions. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 1: Long Papers, pages 463–473. Association for Computational Linguistics.  

Subhro Roy and Dan Roth. 2015. Solving general arithmetic word problems. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1743–1752, Lisbon, Portugal. Association for Computational Linguistics.  

Jingqing Ruan, Yihong Chen, Bin Zhang, Zhiwei Xu, Tianpeng Bao, Guoqing Du, Shiwei Shi, Hangyu Mao, Xingyu Zeng, and Rui Zhao. 2023. Tptu: Task planning and tool usage of large language modelbased ai agents.  

Abulhair Saparov and He He. 2023. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.  

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, and et al. 2022. BLOOM: A 176b-parameter open-access multilingual language model. CoRR, abs/2211.05100.  

Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. 2023. Are emergent abilities of large language models a mirage? CoRR, abs/2304.15004.  

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. CoRR, abs/2302.04761.  

Bilgehan Sel, Ahmad Al-Tawaha, Vanshaj Khattar, Lu Wang, Ruoxi Jia, and Ming Jin. 2023. Algorithm of thoughts: Enhancing exploration of ideas in large language models. CoRR, abs/2308.10379.  

Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Synthetic prompting: Generating chain-of-thought demonstrations for large language models. CoRR, abs/2302.00618.  

Yongliang Shen, Kaitao Song, Xu Tan, Dong Sheng Li, Weiming Lu, and Yue Ting Zhuang. 2023. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. ArXiv, abs/2303.17580.  

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. 2023. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.  

Noah Shinn, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Refexion: Language agents with verbal reinforcement learning.  

Kumar Shridhar, Harsh Jhamtani, Hao Fang, Benjamin Van Durme, Jason Eisner, and Patrick Xia. 2023. Screws: A modular framework for reasoning with revisions. arXiv preprint arXiv:2309.13075.  

Kashun Shum, Shizhe Diao, and Tong Zhang. 2023. Automatic prompt augmentation and selection with chain-of-thought from labeled data. CoRR, abs/2302.12822.  

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, Alicia Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ameet Rahane, Anantharaman S. Iyer, Anders Andreassen, Andrea Santilli, Andreas Stuhlmüller, Andrew M. Dai, Andrew La, Andrew K. Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karakas, and et al. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. CoRR, abs/2206.04615.  

Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, and Chao Zhang. 2023. Adaplanner: Adaptive planning from feedback with language models. ArXiv, abs/2305.16653.  

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed Chi, Denny Zhou, and Jason Wei. 2023. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 13003–13051. Association for Computational Linguistics.  

Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. 2021. Proofwriter: Generating implications, proofs, and abductive statements over natural language. In Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021, volume ACL/IJCNLP 2021 of Findings of ACL, pages 3621–3634. Association for Computational Linguistics.  

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4149–4158. Association for Computational Linguistics.  

Alon Talmor, Ori Yoran, Ronan Le Bras, Chandra Bhagavatula, Yoav Goldberg, Yejin Choi, and Jonathan Berant. 2021. Commonsenseqa 2.0: Exposing the limits of AI through gamifcation. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual.  

Xiaojuan Tang, Zilong Zheng, Jiaqi Li, Fanxu Meng, Song-Chun Zhu, Yitao Liang, and Muhan Zhang. 2023. Large language models are in-context semantic reasoners rather than symbolic reasoners. CoRR, abs/2305.14825.  

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and effcient foundation language models. CoRR, abs/2302.13971.  

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian CantonFerrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fne-tuned chat models. CoRR, abs/2307.09288.  

Xingchen Wan, Ruoxi Sun, Hanjun Dai, Sercan Ö. Arik, and Tomas Pfster. 2023. Better zero-shot reasoning with self-adaptive prompting. In Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023, pages 3493–3514. Association for Computational Linguistics.  

Boshi Wang, Xiang Deng, and Huan Sun. 2022a. Iteratively prompt pre-trained language models for chain of thought. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 2714–2730. Association for Computational Linguistics.  

Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. 2023a. Towards understanding chain-of-thought prompting: An empirical study of what matters. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages  

2717–2739. Association for Computational Linguistics.  

Cunxiang Wang, Shuailong Liang, Yue Zhang, Xiaonan Li, and Tian Gao. 2019. Does it make sense? and why? a pilot study for sense making and explanation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4020–4026, Florence, Italy. Association for Computational Linguistics.  

Jianing Wang, Qiushi Sun, Nuo Chen, Xiang Li, and Ming Gao. 2023b. Boosting language models reasoning with chain-of-knowledge prompting. CoRR, abs/2306.06427.  

Keheng Wang, Feiyu Duan, Sirui Wang, Peiguang Li, Yunsen Xian, Chuantao Yin, Wenge Rong, and Zhang Xiong. 2023c. Knowledge-driven cot: Exploring faithful reasoning in llms for knowledge-intensive question answering.  

Lei Wang, Yi Hu, Jiabang He, Xing Xu, Ning Liu, Hui Liu, and Heng Tao Shen. 2023d. T-sciq: Teaching multimodal chain-of-thought reasoning via large language model signals for science question answering. CoRR, abs/2305.03453.  

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. 2023e. A survey on large language model based autonomous agents. CoRR, abs/2308.11432.  

Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. 2023f. Plan-and-solve prompting: Improving zeroshot chain-of-thought reasoning by large language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 2609–2634. Association for Computational Linguistics.  

Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. $2023\mathrm{g}$ . A comprehensive survey of continual learning: Theory, method and application. CoRR, abs/2302.00487.  

Peifeng Wang, Zhengyang Wang, Zheng Li, Yifan Gao, Bing Yin, and Xiang Ren. 2023h. Scott: Selfconsistent chain-of-thought distillation. In Annual Meeting of the Association for Computational Linguistics.  

Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, and Furu Wei. 2022b. Image as a foreign language: Beit pretraining for all vision and visionlanguage tasks. CoRR, abs/2208.10442.  

Xinyi Wang, Lucas Caccia, Oleksiy Ostapenko, Xingdi Yuan, and Alessandro Sordoni. 2023i. Guiding language model reasoning with planning tokens.  

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023j. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.  

Yiming Wang, Zhuosheng Zhang, and Rui Wang. 2023k. Element-aware summarization with large language models: Expert-aligned evaluation and chain-ofthought method. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 8640–8665. Association for Computational Linguistics.  

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022a. Emergent abilities of large language models. Trans. Mach. Learn. Res., 2022.  

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022b. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS.  

Yixuan Weng, Minjun Zhu, Shizhu He, Kang Liu, and Jun Zhao. 2022. Large language models are reasoners with self-verifcation. arXiv preprint arXiv:2212.09561.  

Bo Wu, Shoubin Yu, Zhenfang Chen, Josh Tenenbaum, and Chuang Gan. 2021. STAR: A benchmark for situated reasoning in real-world videos. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual.  

Skyler Wu, Eric Meng Shen, Charumathi Badrinath, Jiaqi Ma, and Himabindu Lakkaraju. 2023. Analyzing chain-of-thought prompting in large language models via gradient-based feature attributions. CoRR, abs/2307.13339.  

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huan, and Tao Gui. 2023. The rise and potential of large language model based agents: A survey. CoRR, abs/2309.07864.  

Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. 2021. Next-qa: Next phase of questionanswering to explaining temporal actions. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 9777–9786. Computer Vision Foundation / IEEE.  

Weijia Xu, Andrzej Banburski-Fahey, and Nebojsa Jojic. 2023. Reprompting: Automated chain-of-thought prompt inference through gibbs sampling.  

Tianci Xue, Ziqi Wang, Zhenhailong Wang, Chi Han, Pengfei Yu, and Heng Ji. 2023. RCOT: detecting and rectifying factual inconsistency in reasoning by reversing chain-of-thought. CoRR, abs/2305.11499.  

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. 2023. MMREACT: prompting chatgpt for multimodal reasoning and action. CoRR, abs/2303.11381.  

Zonglin Yang, Li Dong, Xinya Du, Hao Cheng, Erik Cambria, Xiaodong Liu, Jianfeng Gao, and Furu Wei. 2022. Language models as inductive reasoners. CoRR, abs/2212.10923.  

Fanglong Yao, Changyuan Tian, Jintao Liu, Zequn Zhang, Qing Liu, Li Jin, Shuchao Li, Xiaoyu Li, and Xian Sun. 2023a. Thinking like an expert:multimodal hypergraph-of-thought (hot) reasoning to boost foundation modals.  

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffths, Yuan Cao, and Karthik Narasimhan. 2023b. Tree of thoughts: Deliberate problem solving with large language models. CoRR, abs/2305.10601.  

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023c. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.  

Yao Yao, Zuchao Li, and Hai Zhao. 2023d. Beyond chain-of-thought, effective graph-of-thought reasoning in large language models. CoRR, abs/2305.16582.  

Xi Ye and Greg Durrett. 2022. The unreliability of explanations in few-shot in-context learning. CoRR, abs/2205.03401.  

Xi Ye and Greg Durrett. 2023. Explanation selection using unlabeled data for in-context learning. CoRR, abs/2302.04813.  

Yunhu Ye, Binyuan Hui, Min Yang, Binhua Li, Fei Huang, and Yongbin Li. 2023. Large language models are versatile decomposers: Decomposing evidence and questions for table-based reasoning. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023, pages 174–184. ACM.  

Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B. Tenenbaum. 2020. CLEVRER: collision events for video representation and reasoning. In 8th International Conference on Learning Representations, ICLR 2020,  

Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.  

Ori Yoran, Tomer Wolfson, Ben Bogin, Uri Katz, Daniel Deutch, and Jonathan Berant. 2023. Answering questions by meta-reasoning over multiple chains of thought. CoRR, abs/2304.13007.  

Fei Yu, Hongbo Zhang, and Benyou Wang. 2023a. Nature language reasoning, A survey. CoRR, abs/2303.14725.  

Junchi Yu, Ran He, and Rex Ying. 2023b. Thought propagation: An analogical approach to complex reasoning with large language models. arXiv preprint arXiv:2310.03965.  

Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. 2020. Reclor: A reading comprehension dataset requiring logical reasoning. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.  

Weijiang Yu, Yingpeng Wen, Fudan Zheng, and Nong Xiao. 2021a. Improving math word problems with pre-trained knowledge and hierarchical reasoning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3384–3394, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.  

Weijiang Yu, Haoteng Zheng, Mengfei Li, Lei Ji, Lijun Wu, Nong Xiao, and Nan Duan. 2021b. Learning from inside: Self-driven siamese sampling and reasoning for video question answering. Advances in Neural Information Processing Systems, 34:26462– 26474.  

Zihan Yu, Liang He, Zhen Wu, Xinyu Dai, and Jiajun Chen. 2023c. Towards better chain-of-thought prompting strategies: A survey.  

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. 2022. Star: Bootstrapping reasoning with reasoning. In NeurIPS.  

Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. From recognition to cognition: Visual commonsense reasoning. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 6720–6731. Computer Vision Foundation / IEEE.  

Bowen Zhang, Kehua Chang, and Chunping Li. 2023a. Cot-bert: Enhancing unsupervised sentence representation through chain-of-thought. CoRR, abs/2309.11143.  

Hugh Zhang and David C. Parkes. 2023. Chain-ofthought reasoning is a policy improvement operator.  

Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. 2023b. Draft & verify: Lossless large language model acceleration via self-speculative decoding. arXiv preprint arXiv:2309.08168.  

Li Zhang, Liam Dugan, Hainiu Xu, and Chris CallisonBurch. 2023c. Exploring the curious case of code prompts. CoRR, abs/2304.13250.  

Muru Zhang, Ofr Press, William Merrill, Alisa Liu, and Noah A. Smith. 2023d. How language model hallucinations can snowball. CoRR, abs/2305.13534.  

Sarah J. Zhang, Reece Shuttleworth, Derek Austin, Yann Hicke, Leonard Tang, Sathwik Karnik, Darnell Granberry, and Iddo Drori. 2022. A dataset and benchmark for automatically answering and generating machine learning fnal exams. CoRR, abs/2206.05442.  

Tianhua Zhang, Jiaxin Ge, Hongyin Luo, YungSung Chuang, Mingye Gao, Yuan Gong, Xixin Wu, Yoon Kim, Helen Meng, and James Glass. 2023e. Natural language embedded programs for hybrid language symbolic reasoning. arXiv preprint arXiv:2309.10814.  

Zhuosheng Zhang and Aston Zhang. 2023. You only look at screens: Multimodal chain-of-action agents.  

Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2023f. Automatic chain of thought prompting in large language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.  

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. $2023\mathrm{g}$ . Multimodal chain-of-thought reasoning in language models. CoRR, abs/2302.00923.  

Ruochen Zhao, Xingxuan Li, Shafq Joty, Chengwei Qin, and Lidong Bing. 2023a. Verify-and-edit: A knowledge-enhanced chain-of-thought framework. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 5823–5840. Association for Computational Linguistics.  

Wayne Xin Zhao, Kun Zhou, Zheng Gong, Beichen Zhang, Yuanhang Zhou, Jing Sha, Zhigang Chen, Shijin Wang, Cong Liu, and Ji-Rong Wen. 2022. Jiuzhang: A chinese pre-trained language model for mathematical problem understanding. In KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14 - 18, 2022, pages 4571–4581. ACM.  

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023b. A survey of large language models. CoRR, abs/2303.18223.  

Xufeng Zhao, Mengdi Li, Wenhao Lu, Cornelius Weber, Jae Hee Lee, Kun Chu, and Stefan Wermter. 2023c. Enhancing zero-shot chain-of-thought reasoning in large language models through logic. CoRR, abs/2309.13339.  

Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H Chi, Quoc V Le, and Denny Zhou. 2023. Take a step back: Evoking reasoning via abstraction in large language models. arXiv preprint arXiv:2310.06117.  

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. 2023a. Language agent tree search unifes reasoning acting and planning in language models.  

Ben Zhou, Daniel Khashabi, Qiang Ning, and Dan Roth. 2019. "going on a vacation" takes longer than "going for a walk": A study of temporal commonsense understanding. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pages 3361– 3367. Association for Computational Linguistics.  

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V. Le, and Ed H. Chi. 2023b. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.  

Zhehua Zhou, Jiayang Song, Kunpeng Yao, Zhan Shu, and Lei Ma. 2023c. Isr-llm: Iterative self-refned large language model for long-horizon sequential task planning.  

Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in fnance. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 3277–3287. Association for Computational Linguistics.  

Anni Zou, Zhuosheng Zhang, Hai Zhao, and Xiangru Tang. 2023. Meta-cot: Generalizable chain-ofthought prompting in mixed-task scenarios with large language models. arXiv preprint arXiv:2310.06692.  