# Beyond Examples: High-level Automated Reasoning Paradigm in In-Context Learning via MCTS
Jinyang Wu * 1 Mingkuan Feng * 1 Shuai Zhang 1 Feihu Che 2 Zengqi Wen 2 Jianhua Tao 1 2
Abstract
In-context Learning (ICL) enables large language models (LLMs) to tackle downstream tasks through sophisticated prompting and high-quality demonstrations. However, this traditional ICL paradigm shows limitations when facing complex mathematical reasoning tasks, primarily due to its heavy dependence on example quality and the necessity for human intervention in challenging scenarios. To address these limitations, this paper presents HiAR-ICL, a High-level Automated Reasoning paradigm in ICL that shifts focus from specific examples to abstract thinking patterns, extending the conventional concept of context in ICL. HiAR-ICL introduces five atomic reasoning actions as fundamental components for constructing chain-structured patterns. Using Monte Carlo Tree Search, we explore reasoning paths and construct thought cards to guide subsequent inference. We then develop a cognitive complexity framework that dynamically matches problems with appropriate thought cards. Experimental results demonstrate HiAR-ICL’s effectiveness, achieving state-of-the-art accuracy (79.6%) on the MATH benchmark with Qwen2.5-7B-Instruct, surpassing GPT-4o (76.6%) and Claude 3.5 (71.1%).
arXiv:2411.18478v1
# 1. Introduction
“Give a man a fish and you feed him for a day. Teach a man to fish and you feed him for a lifetime.”
“Give a man a fish and you feed him for a day. Teach a man to fish and you feed him for a lifetime.” — An old proverb
Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks and domains (Zhao et al., 2023; OpenAI, 2023; Yang et al., 2024a; Dubey et al.,
*Equal contribution 1Department of Automation, Tsinghua University, Beijing, China 2Beijing National Research Center for Information Science and Technology, Beijing, China. Correspondence to: Shuai Zhang <zhang shuai@mail.tsinghua.edu.cn>, Jianhua Tao <jhtaoo@tsinghua.edu.cn>.
2024). Among these capabilities, complex reasoning proficiency, particularly in mathematical tasks, has emerged as a critical benchmark for evaluating these models’ fundamental cognitive abilities (Hao et al., 2023; Xi et al., 2024). This aptitude highlights their logical reasoning skills and reflects their ability to solve structured problems effectively (Fu et al., 2023; Plaat et al., 2024). The mastery of multi-step reasoning often demands rigorous adherence to intricate rules, precise execution, and application of various problem-solving strategies, which poses unique challenges for existing LLMs (Ahn et al., 2024).
multi-step reasoning often demands rigorous adherence to intricate rules, precise execution, and application of various problem-solving strategies, which poses unique challenges for existing LLMs (Ahn et al., 2024). Due to its simplicity and parameter-free nature (zero training cost), in-context learning (ICL), also known as few-shot prompting, has garnered increasing attention and emerged as a promising approach for eliciting the reasoning potential of LLMs (Zhou et al., 2024c; Zhao et al., 2024). Originally introduced by (Brown et al., 2020), the key idea of ICL is analogy-based learning (Dong et al., 2024). This approach expects LLMs to discern hidden patterns from carefully curated demonstration examples and subsequently generate appropriate reasoning steps for test problems. Extensive research has focused on enhancing ICL performance through improved prompt engineering, encompassing both instruction optimization (Wang et al., 2023c) and demonstration selection (Luo et al., 2024). A pivotal advancement in this domain is Chain-of-thought (CoT) reasoning (Wei et al., 2022; Kojima et al., 2022). By incorporating the prompt “Let’s think step by step” alongside step-by-step reasoning examples, this approach enables models to emulate human-like reasoning processes, achieving notable success in complex problem-solving, especially in mathematical reasoning tasks (Sprague et al., 2024). Despite these advances, current ICL paradigms face several limitations. First, ICL-based reasoning performance is highly contingent upon the provided demonstrations. Empirical studies (Wang et al., 2023a; Cui et al., 2024; Wang et al., 2024c) have revealed that LLMs exhibit high sensitivity to task-specific characteristics and multiple facets of ICL examples, including demonstration quantity, ordering, and label distributions. Consequently, suboptimal demonstrations may fail to elicit the best model performance and even hinder reasoning capabilities. Second, crafting high-quality
Despite these advances, current ICL paradigms face several limitations. First, ICL-based reasoning performance is highly contingent upon the provided demonstrations. Empirical studies (Wang et al., 2023a; Cui et al., 2024; Wang et al., 2024c) have revealed that LLMs exhibit high sensitivity to task-specific characteristics and multiple facets of ICL examples, including demonstration quantity, ordering, and label distributions. Consequently, suboptimal demonstrations may fail to elicit the best model performance and even hinder reasoning capabilities. Second, crafting high-quality
demonstrations often requires substantial human expertise, which can be time-consuming and labor-intensive for complex reasoning domains such as mathematical problemsolving. Third, ICL’s generalization ability remains limited. When encountering reasoning tasks that share similar logical structures but differ in presentation format, reconstructing corresponding demonstration examples is often necessary. To address these challenges, this paper proposes HiARICL, a High-level Automated Reasoning paradigm in ICL through Monte Carlo Tree Search (MCTS). We aim to expand the conventional conecept of ICL by redefining ‘context’ beyond mere demonstration examples to encompass higher-order cognitive reasoning patterns. This paradigm embodies the principle of “teaching people how to think, rather than merely what to think”, aligning with the proverb, ”Give a man a fish and you feed him for a day. Teach a man to fish and you feed him for a lifetime.” Furthermore, while traditional ICL approaches (e.g., CoT) typically follow a linear left-to-right reasoning trajectory (Chen & Li, 2024), we employ tree search to expand search spaces for reasoning patterns (´Swiechowski et al., 2023; Koh et al., 2024). Specifically, our method comprises four key steps: (1) define atom reasoning actions, (2) construct thought cards via MCTS, (3) select reasoning patterns, and (4) solve and verify. First, we define five atomic reasoning actions that collectively form the building blocks of chain-structured reasoning patterns (termed “thought card” in this paper). These actions are designed to simulate human-like cognitive behaviors, such as problem decomposition and reasoning step reflection. Second, using a small set of randomly sampled seed data, we leverage MCTS to derive reference reasoning patterns through its standard four steps: selection, expansion, simulation, and backpropagation, ultimately constructing multiple thought cards. Third, we introduce a cognitive complexity metric comprising three indicators: subquestion count, problem condition complexity, and semantic similarity. Based on this metric, we match and select the optimal three thought cards that best align with the target problem’s cognitive complexity. Finally, guided by the selected thought cards, we execute the reasoning process and validate the final solution through multiple verification mechanisms: output reward model (ORM), process reward model (PRM), and consistency checks, ensuring high-quality results. Empirical experiments across multiple complex reasoning datasets demonstrate that HiAR-ICL not only outperforms state-of-the-art (SOTA) methods but also achieves reduced time complexity. The core contributions are summarized as follows:
demonstrations often requires substantial human expertise, which can be time-consuming and labor-intensive for complex reasoning domains such as mathematical problemsolving. Third, ICL’s generalization ability remains limited. When encountering reasoning tasks that share similar logical structures but differ in presentation format, reconstructing corresponding demonstration examples is often necessary. To address these challenges, this paper proposes HiARICL, a High-level Automated Reasoning paradigm in ICL through Monte Carlo Tree Search (MCTS). We aim to expand the conventional conecept of ICL by redefining ‘context’ beyond mere demonstration examples to encompass higher-order cognitive reasoning patterns. This paradigm embodies the principle of “teaching people how to think, rather than merely what to think”, aligning with the proverb, ”Give a man a fish and you feed him for a day. Teach a man to fish and you feed him for a lifetime.” Furthermore, while traditional ICL approaches (e.g., CoT) typically follow a linear left-to-right reasoning trajectory (Chen & Li, 2024), we employ tree search to expand search spaces for reasoning patterns (´Swiechowski et al., 2023; Koh et al., 2024).
Specifically, our method comprises four key steps: (1) define atom reasoning actions, (2) construct thought cards via MCTS, (3) select reasoning patterns, and (4) solve and verify. First, we define five atomic reasoning actions that collectively form the building blocks of chain-structured reasoning patterns (termed “thought card” in this paper). These actions are designed to simulate human-like cognitive behaviors, such as problem decomposition and reasoning step reflection. Second, using a small set of randomly sampled seed data, we leverage MCTS to derive reference reasoning patterns through its standard four steps: selection, expansion, simulation, and backpropagation, ultimately constructing multiple thought cards. Third, we introduce a cognitive complexity metric comprising three indicators: subquestion count, problem condition complexity, and semantic similarity. Based on this metric, we match and select the optimal three thought cards that best align with the target problem’s cognitive complexity. Finally, guided by the selected thought cards, we execute the reasoning process and validate the final solution through multiple verification mechanisms: output reward model (ORM), process reward model (PRM), and consistency checks, ensuring high-quality results. Empirical experiments across multiple complex reasoning datasets demonstrate that HiAR-ICL not only outperforms state-of-the-art (SOTA) methods but also achieves reduced time complexity. The core contributions are summarized as follows:
• Novel ICL Framework: We extend the traditional concept of “context” from specific examples to higherlevel cognitive reasoning patterns, advancing the frontier of ICL research.
• Novel ICL Framework: We extend the traditional concept of “context” from specific examples to higherlevel cognitive reasoning patterns, advancing the frontier of ICL research.
• Automated Reasoning Paradigm: We propose a fully automated reasoning paradigm through MCTS, which eliminates human intervention in demonstration design, and aligns with LLM’s intrinsic reasoning capabilities. • Human-Like Reasoning Behavior: We introduce five atomic reasoning actions that emulate human cognitive processes, enabling more effective problem-solving.
eliminates human intervention in demonstration design, and aligns with LLM’s intrinsic reasoning capabilities. • Human-Like Reasoning Behavior: We introduce five atomic reasoning actions that emulate human cognitive processes, enabling more effective problem-solving.
• Human-Like Reasoning Behavior: We introduce five atomic reasoning actions that emulate human cognitive processes, enabling more effective problem-solving.
 Superior Performance: HiAR-ICL significantly outperforms existing methods on complex reasoning benchmarks, achieving 79.6% accuracy on MATH with Qwen2.5-7B-Instruct, substantially outperforming GPT-4o (76.6%).
# 2. Preliminary
In this section, we begin with the problem statement in Section 2.1, and then illustrate our motivation from two aspects (Section 2.2 and Section 2.3).
# 2.1. Problem Statement
For a pre-trained policy LLM πθ parameterized by θ, complex problem-solving can be framed as a multi-step reasoning process. Specifically, given a problem x and predefined instructions Φ like “let’s think step by step” (Kojima et al., 2022; Wei et al., 2022), we guide the model to generate a sequence of reasoning steps that lead to a final answer y ∼πθ(Φ(x)). In detail, this involves T intermediate reasoning steps s0...T = [s0, s1, s2, ..., sT ], where s0 := x and sT := y. At each time step t, the model receives a state St−1, which consists of the original input x and preceding reasoning steps (s1, s2, ..., st−1). The policy model πθ then generates the current action at = πθ(Φ(St−1)), which is used to prompt the LLM to produce the next reasoning step st. The entire process from the initial step s1 to the final output sT naturally forms a complete chain of thought. When direct mapping from input x to output y (e.g., in mathematical reasoning) is complex (Figure 1(a)), few-shot Chain-of-Thought (CoT) prompting (Wei et al., 2022) serves as an effective approach to guide reasoning. This approach requires high-quality demonstrations to facilitate analogical learning for test problems. However, LLMs often exhibit sensitivity to exemplar selection, necessitating human intervention in choosing appropriate samples (Sclar et al., 2024). Moreover, pure analogical learning may lead to rigid thinking patterns, hindering its ability to develop genuine problem-solving capabilities (illustrated in Figure 1(b)). This paper aims to teach LLMs high-level thinking patterns to reason about such problems rather than mere example imitation. We enhance in-context learning to (1) eliminate dependency on carefully curated examples; (2) foster ef-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a908/a9086ea3-20ae-42cf-b065-451bc6a8e2f9.png" style="width: 50%;"></div>
Figure 1. Schematic comparison between HiAR-ICL and traditional zero-shot and few-shot in-context learning methods. We illustrate the learning process through a teacher-student paradigm. (a) Direct Prompting (Zero-shot CoT) provides only generic “Let’s think step by step” instruction, proving insufficient for executing step-by-step reasoning; (b) In-Context Learning (Few-shot CoT) offers carefully selected examples but fails to generalize when encountering dissimilar problems; (c) Our Method (HiAR-ICL) teaches high-level thought patterns, enabling robust performance on various problems.
fective reasoning strategies; and (3) transcend linear CoT limitations by adopting a more precise reasoning paradigm.
# 2.2. Motivation 1: Traditional ICL suffers from example bias, human costs, and limited generalization
Our primary motivation stems from the limitations of InContext Learning (ICL), which can be intuitively illustrated through the teacher-student analogy. Here, the teacher typically represents a human, while the student refers to a model designed for problem-solving. Figure 1 provides a visual comparison of traditional ICL and our method.
Scenario (a) The teacher merely provides step-by-step instructions without explaining the underlying reasoning process or the appropriate thinking pattern for each step. Consequently, the student (particularly smaller models below 10B parameters) struggles to comprehend the teacher’s intent and internalize the problem-solving approach.
with robust problem-solving strategies, enabling them to effectively tackle novel and complex challenges, even in unfamiliar scenarios. Such a long-term perspective empowers students to independently adapt and solve similar types of problems efficiently.
# 2.3. Motivation 2: Precise reasoning paradigms unleash model potential
The recently introduced OpenAI o1 model1 has demonstrated outstanding performance in solving complex reasoning problems, which highlights that precise reasoning paradigms (i.e., Test-time Compute methods) can significantly unleash LLM potential and enhance reasoning capabilities (Wu et al., 2024; Qin et al., 2024). Given that CoT follows a linear left-to-right structure, numerous works have extended reasoning paradigms to tree structures to expand potential search spaces and achieve more precise reasoning, such as ToT (Yao et al., 2023), rStar (Qi et al., 2024), and ReST-MCTS* (Zhang et al., 2024c).
The recently introduced OpenAI o1 model1 has demonstrated outstanding performance in solving complex reasoning problems, which highlights that precise reasoning paradigms (i.e., Test-time Compute methods) can significantly unleash LLM potential and enhance reasoning capabilities (Wu et al., 2024; Qin et al., 2024). Given that CoT follows a linear left-to-right structure, numerous works have extended reasoning paradigms to tree structures to expand potential search spaces and achieve more precise reasoning, such as ToT (Yao et al., 2023), rStar (Qi et al., 2024), and ReST-MCTS* (Zhang et al., 2024c). Motivated by this, we also leverage tree structures (MCTS) to explore more comprehensive reasoning paths. By incorporating prior cognitive patterns into the ICL-based reasoning process, we significantly reduce the computational complexity of infinite search spaces compared to existing approaches while maintaining performance. Overall, this paper constructs a precise reasoning paradigm that effectively balances efficiency and accuracy.
Motivated by this, we also leverage tree structures (MCTS) to explore more comprehensive reasoning paths. By incorporating prior cognitive patterns into the ICL-based reasoning process, we significantly reduce the computational complexity of infinite search spaces compared to existing approaches while maintaining performance. Overall, this paper constructs a precise reasoning paradigm that effectively balances efficiency and accuracy.
1https://openai.com/o1/
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/52e3/52e35633-1209-454e-af55-8f3a1ea76d8d.png" style="width: 50%;"></div>
<div style="text-align: center;">[Action-chain guided reasoning] + [Verify (ORM, PRM, Consistency] [Action-chain guided reasoning] + [Verify (ORM, PRM, Consistency] [Action-chain guided reasoning] + [Verify (ORM, PRM, Consistency]</div>
<div style="text-align: center;">Figure 2. Flowchart of our proposed method HiAR-ICL. This framework consists of four main parts: (1) Define Atom Reasoning Actions; (2) Construct Thought Cards via MCTS; (3) Select Reasoning Patterns; (4) Solve and Verify.</div>
# 3. Methodology
Overview of HiAR-ICL In this section, we introduce HiARICL (High-level Automated Reasoning Paradigm in ICL via MCTS) in detail, and illustrate the whole process in Figure 2. Our method consists of four main components:
• Define Atom Reasoning Actions: Define five humanlike fundamental thought actions as building blocks for
• Construct Thought Cards via MCTS: Leverage MCTS (Selection, Expansion, Simulation, Backpropagation) to construct thought cards comprehensively. • Select Reasoning Patterns: Identify the optimal three reasoning patterns based on the problem’s cognitive complexity level.
• Solve and Verify: Perform reasoning process under the selected patterns, and validate candidate solutions using PRM, ORM, or consistency-based verification.
# 3.1. Define Atom Reasoning Actions
Understanding how humans engage in complex reasoning is essential for modeling human cognition (Jaffe et al., 2023). Existing studies often distinguish between two categories of reasoning or—more broadly—cognitive processes: System 1 and System 2 (Kahneman, 2011; Da Silva, 2023; Hagendorff et al., 2023). “System 1” refers to fast, intuitive, yet error-prone thinking, while “System 2” involves slow, deliberative thinking with superior reasoning performance. Given OpenAI o1 model’s impressive complex reasoning abilities, developing efficient ”System 2” approaches has gained increasing attention among researchers aiming to emulate human cognitive processes (Qin et al., 2024; Sun et al., 2024). Inspired by this, we introduce five atomic human-like reasoning actions to bridge the gap between model reasoning and human cognition as follows:
• (a1) System Analysis (SA): Analyzing the overall structure of the problem and identifying the constraints and conditions before addressing it, thereby clarifying task requirements effectively.
• (a2) One-Step Thought (OST): Generating the next one-step thought based on the given question and the preceding reasoning steps.
# • (a2) One-Step Thought (OST): Generating the next one-step thought based on the given question and the preceding reasoning steps.
• (a3) Chain-of-Thought (CoT): Facilitating step-by-step reasoning by constructing a logical sequence of intermediate thoughts, where each step incrementally builds on the previous ones.
• (a4) Divide and Conquer (DC): Breaking down a complex reasoning problem into several smaller subproblems and progressively solving them to achieve the overall solution.
• (a5) Self-Reflection and Refinement (SRR): Engaging in timely reflection of prior solutions and implementing necessary refinement during the reasoning process to ensure accuracy and reliability.
# 3.2. Construct Thought Cards via MCTS
Following the action definition, we establish prior reasoning patterns. For clarity of presentation, we introduce the concept of “thought cards”, which serve as reasoning templates in subsequent ICL-based reasoning processes. Specifically, using a small seed dataset of several hundred samples, we aim to derive their optimal reasoning paths and distill them into multiple thought cards. These thought cards then serve
as prior knowledge when encountering test cases, facilitating efficient reasoning by providing reference insights.
# Step 1: Acquiring Reasoning Paths for Seed dataset
As shown in Figure 2, for reasoning path acquisition, we employ Monte Carlo Tree Search (MCTS) to iteratively evaluate and optimize the solution search process and obtain high-quality reaonsing paths of given seed dataset. This design leverages both the iterative nature of MCTS and the inherent reasoning capabilities of LLMs, thereby leading to enhanced search outcomes (Ye et al., 2021; Zhou et al., 2024a). In our framework, following (Wang et al., 2024a; Qi et al., 2024), we formulate each complex reasoning question q as a tree search problem, where q represents the root node, and subsequent tree nodes denote reasoning steps (comprising actions and corresponding outcomes) generated by an LLM policy πθ. We define the state St−1 as an (incomplete) trajectory q, s1, ..., st−1, where S0 = q. The next step in the sequence can then be sampled as st ∼πθ(St−1). Another critical component of MCTS is the reward function, which guides tree expansion by evaluating action values. We define Q(s) as the reward value for node s. Initially, all unexplored nodes are assigned Q(si, ai) = 0. They will be updated as a weighted average of its current value and the Q-value of its child node:
(1)
where α serves as a discount factor, reflecting the importance of future rewards. For terminal nodes, following (Qi et al., 2024), we adopt the likelihood (confidence) of selfconsistency majority voting as the reward value. This design eliminates the need for external supervision, thereby enhancing the generalization of our approach.
Specifically, this step undergones four MCTS phases: selection, expansion, simulation, and backpropagation.
(1) Selection. This phase identifies the most suitable node for subsequent expansion. Starting from the root node, a child node is chosen at each tree level until reaching a leaf node, which we define as either achieving the maximum tree depth or arriving at an answer here. To balance the exploration and exploitation, we use the well-known Upper Confidence Bounds applied to Trees (UCT) (Kocsis & Szepesv´ari, 2006) for node selection:
(2)
where Q(s) is the reward value for node s, N(s) is the number of visits to s, p is the parent node of s, and w is the exploration weight. The node with the highest UCT value is selected for subsequent phases. (2) Expansion. During the expansion phase, the selected node s is expanded by sampling n actions from πθ and then
where Q(s) is the reward value for node s, N(s) is the number of visits to s, p is the parent node of s, and w is the exploration weight. The node with the highest UCT value is selected for subsequent phases.
generating corresponding reasoning outcomes, as described above. This results in n child nodes, which are added to the tree and stored in an external long-term memory structure. (3) Simulation. In MCTS, a complete simulation process involves iteratively selecting and expanding nodes from the initially selected node until reaching a terminal node. Here, we define a terminal node as either the maximum tree depth (d=5 in our implementation) or an answer node, indicated by a specific answer marker in the output, such as “the answer is” in our implementation. At each tree level, nodes are sampled and expanded following a consistent procedure. To enhance efficiency and avoid unnecessary expansion, we implement an early termination strategy. Inspired by (Fluri et al., 2023; Zhou et al., 2024a), we incorporate a heuristic based on self-consistency (Wang et al., 2023b). This strategy leverages the observation that actions sampled multiple times at the same state are more likely to be accurate, indicating that the model has likely completed the task successfully. Specifically, if the model’s consistency score exceeds a predefined threshold c, i.e., SC(s) > c, the simulation can be terminated early.
(4) Backpropagation. When the end of a rollout is reached, a backpropagation is carried out. During this phase, we update node information, including the number of visits and node values based on the outcome of a trajectory. Specifically, the number of visits of each node si along the simulation path s0, ...sd will be updated as Nnew(s) = Nold(s) + 1. The value Q(s) of a newly visited node is propagated backward to its parent node p, where the reward value of p is adjusted as Eq. 1. Note that, these updated values are used in the UCT formula (Eq. 2) to guide the selection of the next node.
# Step 2: Distilling Paths into Thought Cards
After executing the MCTS procedure, we obtain a tree structure for each question in the seed dataset, yielding multiple valid reasoning trajectories. To identify the optimal reasoning path per question, we draw inspiration from the concept of Value of Computation (VOC) (Russell & Wefald, 1991). This principle posits that intelligent systems should optimize the trade-off between computational benefits and costs, which leads us to propose a novel VOC-inspired selection metric as follows:
score(x, pa, y) = k·Reward(pa|x)−(1−k)·C(pa) (3
where x represents the task input, y denotes the target answer, pa is a candidate reasoning path, k is a balance factor between computational benefits and costs. Reward(pa|x) represents the final reward score of path pa in the generated tree, and C(pa) measures the path length of the reasoning sequence. Here, for simplicity, we define Reward(pa|x)as the Q-value of the leaf node, and C(pa) as the total number of utilized actions in the reasoning sequence. More complex settings are left for future research.
Then, for each question in the seed dataset, we select the path pa with the highest score(x, pa, y) value to construct the Question-path repository, establishing one-to-one mappings between questions and their optimal reasoning paths. Drawing inspiration from metareasoning approaches (Russell & Wefald, 1991; De Sabbata et al., 2024), which advocate for adaptive reasoning strategies based on problem characteristics, we perform a distillation process to these fundamental question-path pairs into more abstract thought cards for subsequent test-time reference. This transformation is guided by a novel cognitive complexity framework designed for complex reasoning tasks (Lee & Heyworth, 2000; Embretson & Daniel, 2008). The framework encompasses three key dimensions: (1) Subquestion Count (SC): Quantifying the number of subproblems; (2) Problem Condition Complexity (PCC): Measuring the number of known conditions; (3) Semantic Similarity (SS): Assessing semantic distance between the target problem and the seed dataset.
Here, using the first two metrics, we refine the Question-path Repository to generate multiple high-level thought cards, as shown in Figure 2. The third metric is reserved for the testing phase. The next section describes how we utilized these thought cards in the testing phase.
# 3.3. Select Reasoning Patterns
During the evaluation phase, we apply our cognitive complexity framework to each test question. We instruct Llama38B-instruct to compute its SC, PCC, and SS, with the latter metric being calculated based on a few dozen examples from the seed dataset. Using these metrics, we perform a matching process within our thought cards, selecting two to three cards that exhibit the closest alignment in scores for each metric. These selected cards are then utilized as reference templates in the final solution generation and verification process.
# 3.4. Solve and Verify
Solve y leveraging two to three selected thought cards, we generates candidate solutions for a given test problem. Extending the tree-structured reasoning framework introduced in Section 3.2, our method incorporates prior reasoning patterns, eliminating the need to expand multiple child nodes. Instead, we follow the action sequences specified on each thought card at each hierarchical level, as depicted in Figure 2. In this way, for a test problem qtest, we can obtain several candidate solutions.
Verify Identifying the most accurate reasoning trajectory among multiple candidate solutions represents a critical challenge (Uesato et al., 2022; Qi et al., 2024; Zhang et al., 2024c). We investigate three verification methodologies: process-supervision, outcome-supervision, and consistencybased approaches. Our empirical results demonstrate that
<div style="text-align: center;">Table 1. Evaluation of HiAR-ICL’s reasoning capabilities against ICL methods across four reasoning benchmarks. The best results in each box are highlighted in bold. Our method, HiAR-ICL consistently achieves the best performance across models and datasets.</div>
MODEL
SETTING
MATHEMATICS
ARITHMETIC
COMMONSENSE
AVERAGE
MATH
GSM8K
SVAMP
StrategyQA
Qwen2.5-14B-instruct
Zero-shot CoT
69.8
92.4
91.6
62.8
79.1
Few-shot CoT
80.0
94.8
91.3
53.1
79.8
CoT+SC@4
76.2
94.0
91.0
69.7
82.7
Ours
80.2
95.3
93.7
77.3
86.6
Qwen2.5-7B-instruct
Zero-shot CoT
64.8
86.2
91.3
52.8
73.7
Few-shot CoT
75.5
91.6
92.3
67.6
81.7
CoT+SC@4
76.4
92.0
92.3
73.2
83.4
Ours
79.6
92.8
93.0
76.0
85.4
Qwen2-7B-instruct
Zero-shot CoT
36.9
76.6
85.2
55.3
63.5
Few-shot CoT
52.9
85.7
87.3
62.3
72.0
CoT+SC@4
55.6
87.7
90.3
65.5
74.8
Ours
63.8
90.6
92.7
72.0
79.8
Yi-1.5-6B-Chat
Zero-shot CoT
30.4
76.4
64.4
46.2
54.3
Few-shot CoT
40.5
78.9
81.3
61.1
65.4
CoT+SC@4
42.2
79.4
87.6
65.2
68.6
Ours
54.0
81.4
90.0
70.3
74.0
Llama-3-8B-Instruct
Zero-shot CoT
5.8
68.3
70.9
57.2
50.5
Few-shot CoT
17.8
74.5
81.0
68.4
60.4
CoT+SC@4
28.8
80.6
88.0
66.8
66.0
Ours
43.2
89.6
92.7
73.0
74.6
Llama-3.1-8B-Instruct
Zero-shot CoT
18.0
61.5
69.3
52.4
50.3
Few-shot CoT
47.2
76.6
82.0
63.6
67.3
CoT+SC@4
44.2
80.5
85.6
69.8
70.0
Ours
55.0
90.7
93.0
73.2
78.0
even the simple consistency-based method can effectively select the most precise reasoning chains, achieving robust reasoning performance.
select the most precise reasoning chains, achieving robust reasoning performance. In summary, our approach can be seen as an optimized variant of tree search algorithms. Unlike traditional MCTS methods that require extensive exploration—often involving exhaustive search to identify effective paths, which is computationally expensive—our method reallocates computational complexity. By strategically incorporating prior knowledge, we significantly enhance search efficiency. As illustrated in the fourth section of Figure 2, our approach effectively traverses the solution tree by directly targeting promising trajectories. Therefore, this approach maintains high-performance standards while simultaneously reducing computational time complexity, representing a notable advancement in efficient reasoning strategies.
# 4. Experiments
# 4.1. Setups
Datasets To evaluate the effectiveness of our method, HiARICL, we conduct comprehensive experiments across diverse datasets and reasoning tasks. We sample several hundred
instances from the training sets as seed data for thought card construction (described in Section 3.2). The model’s performance was then evaluated on corresponding test sets. Our evaluation benchmarks encompass: (1) arithmetic reasoning: GSM8K1319 (Cobbe et al., 2021) and SVAMP300 (Patel et al., 2021); (2) complex mathematical reasoning: MATH500 (Hendrycks et al., 2021); (3) multi-hop commonsense reasoning: StrategyQA687 (Geva et al., 2021). Models HiAR-ICL is a general approach applicable to various LLMs. In our experiments, we evaluate its effectiveness using state-of-the-art (SOTA) models: Llama3-8BInstruct (Dubey et al., 2024), Llama-3.1-8B-Instruct (Meta AI, 2024), Yi-1.5-6B-Chat (Young et al., 2024), Qwen2-7BInstruct (Yang et al., 2024a), and Qwen2.5-7B/14B-Instruct (Qwen Team, 2024). By focusing on LLMs with parameter counts generally under 10B, we aim to demonstrate the robustness and efficiency of our method. We expect that applying HiAR-ICL to small language models will achieve results comparable to or exceeding closed-source LLMs. Baselines We evaluate HiAR-ICL against three strong baseline categories: (1) traditional example-based ICL methods, including zero-shot CoT (Kojima et al., 2022), few-shot CoT (Wei et al., 2022), and SC+CoT (Wang et al., 2023b); (2)
instances from the training sets as seed data for thought card construction (described in Section 3.2). The model’s performance was then evaluated on corresponding test sets. Our evaluation benchmarks encompass: (1) arithmetic reasoning: GSM8K1319 (Cobbe et al., 2021) and SVAMP300 (Patel et al., 2021); (2) complex mathematical reasoning: MATH500 (Hendrycks et al., 2021); (3) multi-hop commonsense reasoning: StrategyQA687 (Geva et al., 2021).
Table 2. Comparison with leading closed-source LLMs. The best results in each box are highlighted in bold. Results for closedsource models are sourced from corresponding official websites. ‘CS’ and ‘OS’ represent closed-source and open-source LLMs, respectively. Notably, the 7B model, Qwen2.5-7b-instruct, surpasses all closed-source models, achieving SOTA performance.
MODEL
SETTING
MATH
GSM8K
Claude-3-Opus
CS
60.1
95.0
Claude-3.5-Sonnet
CS
71.1
96.4
GPT-3.5
CS
43.1
81.6
GPT-4
CS
64.5
94.2
GPT-4o
CS
76.6
96.1
GPT-4o mini
CS
70.2
93.2
Gemini-1.5-Pro
CS
67.7
90.8
Llama-3.1-405B-Instruct
OS 405B
73.8
96.8
Llama-3.1-70B-Instruct
OS 70B
68.0
95.1
Llama-3-70B-Instruct
OS 70B
50.4
93.0
Nemotron4-340B-Instruct
OS 340B
41.1
92.3
Mixtral-large2-Instruct
OS 123B
69.9
92.7
Mixtral-8x22B-Instruct
OS 141B
54.1
88.2
NuminaMath-72B CoT
OS 72B
66.7
90.8
Qwen2-72B-Instruct
OS 72B
69.0
93.2
Yi-1.5-34B-Chat
OS 34B
50.1
90.2
Qwen2.5-14B-instruct
Ours
80.2
95.3
Qwen2.5-7B-instruct
Ours
79.6
92.8
Qwen2-7B-instruct
Ours
63.8
90.6
Yi-1.5-6B-Chat
Ours
54.0
81.4
Llama-3-8B-Instruct
Ours
43.2
89.6
Llama-3.1-8B-Instruct
Ours
55.0
90.7
tree-based methods, including ToT (Yao et al., 2023), RAP (Hao et al., 2023), ReST-MCTS∗(Zhang et al., 2024c), LiteSearch (Wang et al., 2024a), MCTSr (Zhang et al., 2024a), BEATS (Sun et al., 2024), LLaMA-Berry (Zhang et al., 2024b), and rStar (Qi et al., 2024). (3) powerful closedsource LLMs, including GPT-4 (OpenAI, 2023), GPT-4o (OpenAI, 2024), Claude-3.5 (Anthropic, 2024) and Gemini1.5-pro (Google DeepMind, 2024). Evaluation Metrics We evaluate our approach using two key metrics. First, we report accuracy as our primary evaluation metric, where correctness is determined by comparing the model’s final answer with the ground truth. To ensure consistent answer extraction, we require the LLM to explicitly state its solution following a predefined format (e.g., ”The answer is”). Additionally, we measure the average reasoning time to analyze our method’s computational complexity compared to existing search-based approaches. Implementation Details We utilize the vLLM framework2 with the following parameters: temperature set to 0.8, top p set to 0.9, and max tokens set to 1024. All experiments were conducted on a machine running Ubuntu 22.04, equipped with NVIDIA A100-80GB GPUs.
tree-based methods, including ToT (Yao et al., 2023), RAP (Hao et al., 2023), ReST-MCTS∗(Zhang et al., 2024c), LiteSearch (Wang et al., 2024a), MCTSr (Zhang et al., 2024a), BEATS (Sun et al., 2024), LLaMA-Berry (Zhang et al., 2024b), and rStar (Qi et al., 2024). (3) powerful closedsource LLMs, including GPT-4 (OpenAI, 2023), GPT-4o (OpenAI, 2024), Claude-3.5 (Anthropic, 2024) and Gemini1.5-pro (Google DeepMind, 2024).
# 4.2. Results on diverse reasoning benchmarks
As shown in Table 1, we evaluate the effectiveness of HiARICL across four mainstream reasoning benchmarks. We provide comprehensive comparisons between HiAR-ICL and SOTA closed-source LLMs’ performances (sourced from official websites and technical reports). We have two key findings: ♢HiAR-ICL consistently performs better than traditional ICL methods across all tasks. For example, Llama3-8B’s accuracy on the MATH benchmark improved from 17.8% (few-shot CoT) to 43.2% with HiAR-ICL, representing a substantial performance enhancement of 2.4 times. ♢Our method exhibits the most substantial performance improvements on relatively small language models. For example, Qwen2-7B-Instruct improved from 52.9% to 63.8%, Yi-1.5-6B-Chat from 40.5% to 54.0%, and Llama3-8BInstruct from 17.8% to 43.2%. These results underscore our approach’s potential to efficiently guide smaller language models in generating and selecting optimal solutions.
As shown in Table 1, we evaluate the effectiveness of HiARICL across four mainstream reasoning benchmarks. We provide comprehensive comparisons between HiAR-ICL and SOTA closed-source LLMs’ performances (sourced from official websites and technical reports). We have two key findings:
♢Our method exhibits the most substantial performance improvements on relatively small language models. For example, Qwen2-7B-Instruct improved from 52.9% to 63.8%, Yi-1.5-6B-Chat from 40.5% to 54.0%, and Llama3-8BInstruct from 17.8% to 43.2%. These results underscore our approach’s potential to efficiently guide smaller language models in generating and selecting optimal solutions.
# 4.3. Comparison with powerful Closed-source LLMs
To comprehensively demonstrate the effectiveness of our method, we compare our approach with current state-of-theart closed-source models and prominent open-source models wwith several hundred billion parameters. As shown in Table 2, HiAR-ICL-empowered LLMs achieved competitive performance comparable to models with several hundred billion parameters, even surpassing powerful closed-source models. Notably, the 7B Qwen2.5 model achieved 79.6% accuracy on the challenging MATH benchmark, exceeding GPT-4o’s performance. Similarly, Qwen2-7B-Instruct, utilizing the HiAR-ICL reasoning paradigm, outperformed Llama3-70B-Instruct.
# 4.4. Comparison with tree-based methods
We compare our approach with other tree-based reasoning methods on GSM8K and MATH. As shown in Table 3, our method demonstrates superior performance and notable generalizability across various models and datasets. Notably, as benchmark complexity increases, existing methods such as ToT and RAP face significant challenges. While methods like BEATs show comparable performance to ours on Llama3-8B-Instruct, performance disparities persist across other models like Yi-1.5-6B-Chat and Qwen2-7B-Instruct. We also compare with rStar (Qi et al., 2024) across four datasets. As a recently proposed powerful method, rStar significantly expands the solution search space and achieves state-of-the-art performance in tree-based reasoning. Our analysis in Figure 3 systematically assesses performance and computational efficiency. Notably, our approach achieves
Table 3. Evaluation of HiAR-ICL’s reasoning capabilities against state-of-the-art tree-based methods. Our results are marked in blue , with baseline results sourced from the original paper when accessible. The best results are highlighted in bold.
MODEL
METHOD
GSM8K
MATH
Yi-1.5-6B-Chat
BEATS
76.1
51.2
Ours
81.4
54.0
Qwen2-7B-instruct
BEATS
83.0
61.5
Ours
90.6
63.8
Llama-3-8B-Instruct
ToT
69.0
13.6
RAP
80.5
18.8
ReST-MCTS*
-
34.2
LiteSearch
82.3
-
LLaMA-Berry
88.1
39.6
BEATS
88.4
42.9
Ours
89.6
43.2
Llama-3.1-8B-Instruct
LLaMA-Berry
89.8
54.8
Ours
90.7
55.0
METHOD
1
2
3
4
5
AVERAGE
CoT
83.7
83.3
82.8
59.4
37.3
64.8
CoT+SC
95.3
90.0
91.4
73.4
52.2
76.4
HiAR-ICL
97.7
94.5
92.3
80.5
53.0
79.6
Table 4. The effect of verification method for HiAR-ICL. ‘PRM’, ‘ORM’, ’SC’ denote process-reward model, output-reward model, and self-consistency, respectively.
MODEL
SETTING
GSM8K
MATH
Llama3-8B-Instruct
ORM
86.4
38.6
PRM (min)
89.6
42.8
PRM (product)
87.9
41.4
SC
89.0
43.2
Qwen2-7B-instruct
ORM
88.7
56.6
PRM (min)
90.6
62.2
PRM (product)
90.4
61.4
SC
90.4
63.8
competitive performance with rStar while substantially reducing time complexity. For example, our method demonstrates significant time reduction on relatively straightforward datasets such as GSM8K and StrategyQA, with reductions of 27.6X, and 47.3X, respectively. This indicates that for simpler problems, our approach can efficiently complete reasoning without excessive exploration by adaptively selecting simple reasoning patterns. Similarly, we achieve a tenfold reduction in computational time on the more challenging MATH dataset. Consequently, our method presents a clear performance-efficiency trade-off.
# 4.5. The effect of verification method
We conduct experiments to assess the verification parts of HiAR-ICL. As illustrated in Section 3.4, we adopt three
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5900/590064aa-f785-4416-bcc3-f0ebf0714492.png" style="width: 50%;"></div>
Figure 3. Comparison with State-of-the-Art Tree-Based Method, rStar. (a) Performance Comparison (accuracy). (b) Time Cost Per Sample. The results demonstrate that our method achieves remarkable performance while significantly reducing computational time, with an overall 28-fold time reduction and an impressive 47-fold time reduction on the StrategyQA dataset.
Table 5. Performance variations of Qwen2.5-7B-Instruct across different difficulty levels on MATH. We list the result of Zero-shot CoT, fewshot CoT+SC, and our method.
methods: Process Reward Model (PRM), Output Reward Model (ORM), and Self-Consistency (SC). The PRM utilized math-shepherd-mistral-7b-prm3 and ORM employed Llama3.1-8B-ORM-Mistral-Data4. For PRM, two scoring strategies (Lightman et al., 2024; Wang et al., 2024b) were investigated: ‘product’ (final solution score computed by multiplying individual step scores) and ‘min’ (final solution score determined by the minimum step score). As illustrated in Table 4, the results reveal that even a naive consistencybased method demonstrates promising performance. More sophisticated verification methods are reserved for future exploration.
# 4.6. The effect of complexity levels
As shown in Table 5, we present the performance of Zeroshot CoT, Few-shot CoT+SC, and our method HiAR-ICL on the challenging MATH dataset at different difficulty levels. Compared to the first two methods, our approach improves performance across all levels, with an average accuracy boost of 2.6% for the easier levels 1-3. Notably, for the more difficult level 4, the improvement reaches +7.1%. This indicates that our method has the potential to solve more challenging problems and enhance reasoning performance. This may be due to the introduction of thinking patterns,
3huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm 4huggingface.co/RLHFlow/Llama3.1-8B-ORM-Mistral-Data
which help LLMs find a clearer solution more quickly.
# 5. Related Work
In-context learning via examples In-context learning enables LLMs to learn from few demonstrations without finetuning (Zhou et al., 2024c; Dong et al., 2024). For example, Chain-of-Thought (CoT) (Kojima et al., 2022; Wei et al., 2022) prompts LLMs with a simple instruction like “Let’s think step by step” to guide reasoning with few examples. Self-Consistency (Wang et al., 2023b) improves performance by generating multiple reasoning paths and selecting the most consistent answer. However, this traditional paradigm primarily focuses on example-level analogical learning, with performance constrained by demonstration selection and often requiring human expert intervention for complex reasoning tasks (Wang et al., 2023a; Yang et al., 2024b; Zhao et al., 2024). In contrast, we expand the context concept by shifting focus from specific examples to highlevel reasoning patterns. Therefore, our approach offers greater generalizability, enabling fully automated efficient inference without human intervention, even for small models under 10B parameters. Tree-based search LLMs have demonstrated remarkable capabilities but struggle with complex multi-step reasoning tasks (Zhao et al., 2023). Tree search algorithms, particularly Monte Carlo Tree Search (MCTS) (Chaslot et al., 2008), have emerged to expand search spaces and enhance reasoning capabilities (Koh et al., 2024; Zhang et al., 2024a; Zhou et al., 2024b). Recent approaches like Tree of Thought (ToT) (Yao et al., 2023) and GOT (Besta et al., 2024) extend reasoning by using multiple LLM queries to explore non-linear paths. AlphaMath (Chen et al., 2024) enhances LLM mathematical reasoning via MCTS and a value model, without human-annotated supervision. Similarly, rStar (Qi et al., 2024) leverages the model’s inherent capabilities for iterative exploration. However, these methods are often time-intensive. In contrast, our approach introduces a novel paradigm by frontloading computational resources and incorporating prior reasoning patterns, achieving competitive performance with reduced computational complexity.
In-context learning via examples In-context learning enables LLMs to learn from few demonstrations without finetuning (Zhou et al., 2024c; Dong et al., 2024). For example, Chain-of-Thought (CoT) (Kojima et al., 2022; Wei et al., 2022) prompts LLMs with a simple instruction like “Let’s think step by step” to guide reasoning with few examples. Self-Consistency (Wang et al., 2023b) improves performance by generating multiple reasoning paths and selecting the most consistent answer. However, this traditional paradigm primarily focuses on example-level analogical learning, with performance constrained by demonstration selection and often requiring human expert intervention for complex reasoning tasks (Wang et al., 2023a; Yang et al., 2024b; Zhao et al., 2024). In contrast, we expand the context concept by shifting focus from specific examples to highlevel reasoning patterns. Therefore, our approach offers greater generalizability, enabling fully automated efficient inference without human intervention, even for small models under 10B parameters.
Tree-based search LLMs have demonstrated remarkable capabilities but struggle with complex multi-step reasoning tasks (Zhao et al., 2023). Tree search algorithms, particularly Monte Carlo Tree Search (MCTS) (Chaslot et al., 2008), have emerged to expand search spaces and enhance reasoning capabilities (Koh et al., 2024; Zhang et al., 2024a; Zhou et al., 2024b). Recent approaches like Tree of Thought (ToT) (Yao et al., 2023) and GOT (Besta et al., 2024) extend reasoning by using multiple LLM queries to explore non-linear paths. AlphaMath (Chen et al., 2024) enhances LLM mathematical reasoning via MCTS and a value model, without human-annotated supervision. Similarly, rStar (Qi et al., 2024) leverages the model’s inherent capabilities for iterative exploration. However, these methods are often time-intensive. In contrast, our approach introduces a novel paradigm by frontloading computational resources and incorporating prior reasoning patterns, achieving competitive performance with reduced computational complexity.
# 6. Conclusion
In this work, we propose HiAR-ICL, a novel automated reasoning paradigm that extends the concept of context in ICL and enables LLMs to perform adaptive and efficient reasoning for challenging problems. By incorporating abstract thinking patterns rather than relying on human intervention and example quality, HiAR-ICL empowers LLMs to develop genuine reasoning capabilities instead of merely imitating demonstration examples. Extensive experimental results demonstrate that HiAR-ICL significantly outperforms
existing methods, including traditional example-based ICL and test-time OpenAI o1-like search methods. The success of HiAR-ICL in mathematical reasoning tasks establishes a solid foundation for advancing complex reasoning in LLMs. This work represents a promising starting point, paving the way for future research to explore more sophisticated highlevel automated reasoning paradigms and their applicability across a broader spectrum of complex domains.
# References
Besta, M., Blach, N., Kubicek, A., Gerstenberger, R., Podstawski, M., Gianinazzi, L., Gajda, J., Lehmann, T., Niewiadomski, H., Nyczyk, P., et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 17682–17690, 2024.
rown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020.
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. Cui, Y., He, P., Tang, X., He, Q., Luo, C., Tang, J., and Xing, Y. A theoretical understanding of chain-of-thought: Coherent reasoning and error-aware demonstration. arXiv preprint arXiv:2410.16540, 2024. Da Silva, S. System 1 vs. system 2 thinking. Psych, 5(4): 1057–1076, 2023. De Sabbata, C. N., Sumers, T. R., and Griffiths, T. L. Rational metareasoning for large language models. arXiv preprint arXiv:2410.05563, 2024. Dong, Q., Li, L., Dai, D., Zheng, C., Ma, J., Li, R., Xia, H., Xu, J., Wu, Z., Chang, B., Sun, X., Li, L., and Sui, Z. A survey on in-context learning. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 1107–1128, Miami, Florida, USA, November 2024. Association for Computational Linguistics. Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. Embretson, S. E. and Daniel, R. C. Understanding and quantifying cognitive complexity level in mathematical problem solving items. Psychology Science, 50(3):328, 2008. Fluri, L., Paleka, D., and Tram`er, F. Evaluating superhuman models with consistency checks. In Socially Responsible Language Modelling Research, 2023. Fu, Y., Peng, H., Ou, L., Sabharwal, A., and Khot, T. Specializing smaller language models towards multi-step reasoning. In International Conference on Machine Learning, pp. 10421–10430. PMLR, 2023. Geva, M., Khashabi, D., Segal, E., Khot, T., Roth, D., and Berant, J. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9: 346–361, 2021. Google DeepMind. Gemini models, May 2024. URL https://deepmind.google/technologies/ gemini/. Hagendorff, T., Fabi, S., and Kosinski, M. Human-like intuitive behavior and reasoning biases emerged in large language models but disappeared in chatgpt. Nature Computational Science, 3(10):833–838, 2023.
Hagendorff, T., Fabi, S., and Kosinski, M. Human-like intuitive behavior and reasoning biases emerged in large language models but disappeared in chatgpt. Nature Computational Science, 3(10):833–838, 2023.
Hagendorff, T., Fabi, S., and Kosinski, M. Human-like intuitive behavior and reasoning biases emerged in large language models but disappeared in chatgpt. Nature Computational Science, 3(10):833–838, 2023.
Hao, S., Gu, Y., Ma, H., Hong, J., Wang, Z., Wang, D., and Hu, Z. Reasoning with language model is planning with world model. In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 8154–8173, Singapore, December 2023. Association for Computational Linguistics. Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. Jaffe, P. I., Poldrack, R. A., Schafer, R. J., and et al. Modelling human behaviour in cognitive tasks with latent dynamical systems. Nature Human Behaviour, 7:986–1000, 2023. Kahneman, D. Thinking, Fast and Slow. Farrar, Straus and Giroux, New York, NY, 2011. ISBN 978-0374275631. Kocsis, L. and Szepesv´ari, C. Bandit based monte-carlo planning. In F¨urnkranz, J., Scheffer, T., and Spiliopoulou, M. (eds.), Machine Learning: ECML 2006, pp. 282–293, Berlin, Heidelberg, 2006. Springer Berlin Heidelberg. ISBN 978-3-540-46056-5. Koh, J. Y., McAleer, S., Fried, D., and Salakhutdinov, R. Tree search for language model agents. arXiv preprint arXiv:2407.01476, 2024. Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35: 22199–22213, 2022. Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann. Lee, F.-L. and Heyworth, R. Problem complexity: A measure of problem difficulty in algebra by using computer. EDUCATION JOURNAL-HONG KONG-CHINESE UNIVERSITY OF HONG KONG-, 28(1):85–108, 2000. Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. Luo, M., Xu, X., Liu, Y., Pasupat, P., and Kazemi, M. Incontext learning with retrieved demonstrations for language models: A survey. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. Survey Certification.
Meta AI. Introducing llama 3.1, July 2024. URL https: //ai.meta.com/blog/meta-llama-3-1/. OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. OpenAI. Hello gpt-4o, May 2024. URL https:// openai.com/index/hello-gpt-4o/. Patel, A., Bhattamishra, S., and Goyal, N. Are nlp models really able to solve simple math word problems? In North American Chapter of the Association for Computational Linguistics, 2021. Plaat, A., Wong, A., Verberne, S., Broekens, J., van Stein, N., and Back, T. Reasoning with large language models, a survey. arXiv preprint arXiv:2407.11511, 2024. Qi, Z., Ma, M., Xu, J., Zhang, L. L., Yang, F., and Yang, M. Mutual reasoning makes smaller llms stronger problemsolvers. arXiv preprint arXiv:2408.06195, 2024. Qin, Y., Li, X., Zou, H., Liu, Y., Xia, S., Huang, Z., Ye, Y., Yuan, W., Liu, H., Li, Y., et al. O1 replication journey: A strategic progress report–part 1. arXiv preprint arXiv:2410.18982, 2024. Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github. io/blog/qwen2.5/. Russell, S. and Wefald, E. Principles of metareasoning. Artificial Intelligence, 49(1):361–395, 1991. ISSN 00043702. Sclar, M., Choi, Y., Tsvetkov, Y., and Suhr, A. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In ICLR, 2024. Sprague, Z., Yin, F., Rodriguez, J. D., Jiang, D., Wadhwa, M., Singhal, P., Zhao, X., Ye, X., Mahowald, K., and Durrett, G. To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning. arXiv preprint arXiv:2409.12183, 2024. Sun, L., Liang, H., Wei, J., Yu, B., He, C., Zhou, Z., and Zhang, W. Beats: Optimizing llm mathematical capabilities with backverify and adaptive disambiguate based efficient tree search. arXiv preprint arXiv:2409.17972, 2024. ´Swiechowski, M., Godlewski, K., Sawicki, B., and Ma´ndziuk, J. Monte carlo tree search: A review of recent modifications and applications. Artificial Intelligence Review, 56(3):2497–2562, 2023.
Meta AI. Introducing llama 3.1, July 2024. URL https: //ai.meta.com/blog/meta-llama-3-1/.
Qi, Z., Ma, M., Xu, J., Zhang, L. L., Yang, F., and Yang, M. Mutual reasoning makes smaller llms stronger problemsolvers. arXiv preprint arXiv:2408.06195, 2024.
Qin, Y., Li, X., Zou, H., Liu, Y., Xia, S., Huang, Z., Ye, Y., Yuan, W., Liu, H., Li, Y., et al. O1 replication journey: A strategic progress report–part 1. arXiv preprint arXiv:2410.18982, 2024.
Russell, S. and Wefald, E. Principles of metareasoning. Artificial Intelligence, 49(1):361–395, 1991. ISSN 00043702.
Sclar, M., Choi, Y., Tsvetkov, Y., and Suhr, A. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In ICLR, 2024.
Sprague, Z., Yin, F., Rodriguez, J. D., Jiang, D., Wadhwa, M., Singhal, P., Zhao, X., Ye, X., Mahowald, K., and Durrett, G. To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning. arXiv preprint arXiv:2409.12183, 2024.
Sun, L., Liang, H., Wei, J., Yu, B., He, C., Zhou, Z., and Zhang, W. Beats: Optimizing llm mathematical capabilities with backverify and adaptive disambiguate based efficient tree search. arXiv preprint arXiv:2409.17972, 2024.
´Swiechowski, M., Godlewski, K., Sawicki, B., and Ma´ndziuk, J. Monte carlo tree search: A review of recent modifications and applications. Artificial Intelligence Review, 56(3):2497–2562, 2023.
Uesato, J., Kushman, N., Kumar, R., Song, F., Siegel, N., Wang, L., Creswell, A., Irving, G., and Higgins, I. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.
Wang, A., Song, L., Tian, Y., Peng, B., Yu, D., Mi, H., Su, J., and Yu, D. Litesearch: Efficacious tree search for llm. arXiv preprint arXiv:2407.00320, 2024a.
Wang, L., Li, L., Dai, D., Chen, D., Zhou, H., Meng, F., Zhou, J., and Sun, X. Label words are anchors: An information flow perspective for understanding in-context learning. In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 9840–9855, Singapore, December 2023a. Association for Computational Linguistics.
Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., and Sui, Z. Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9426–9439, Bangkok, Thailand, August 2024b. Association for Computational Linguistics.
Wang, S., Chen, Z., Shi, C., Shen, C., and Li, J. Mixture of demonstrations for in-context learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024c.
Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023b.
Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language models with self-generated instructions. In Rogers, A., Boyd-Graber, J., and Okazaki, N. (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13484–13508, Toronto, Canada, July 2023c. Association for Computational Linguistics.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
Wu, S., Peng, Z., Du, X., Zheng, T., Liu, M., Wu, J., Ma, J., Li, Y., Yang, J., Zhou, W., et al. A comparative study on reasoning patterns of openai’s o1 model. arXiv preprint arXiv:2410.13639, 2024.
Xi, Z., Chen, W., Hong, B., Jin, S., Zheng, R., He, W., Ding, Y., Liu, S., Guo, X., Wang, J., Guo, H., Shen, W., Fan, X., Zhou, Y., Dou, S., Wang, X., Zhang, X., peng sun, Gui, T., Zhang, Q., and Huang, X. Training large language models for reasoning through reverse curriculum reinforcement learning. In Forty-first International Conference on Machine Learning, 2024. Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., Li, C., Li, C., Liu, D., Huang, F., et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024a. Yang, L., Yu, Z., Zhang, T., Cao, S., Xu, M., Zhang, W., Gonzalez, J. E., and Cui, B. Buffer of thoughts: Thoughtaugmented reasoning with large language models. Advances in Neural Information Processing Systems, 2024b. Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 11809–11822. Curran Associates, Inc., 2023. Ye, W., Liu, S., Kurutach, T., Abbeel, P., and Gao, Y. Mastering atari games with limited data. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 25476–25488. Curran Associates, Inc., 2021. Young, A., Chen, B., Li, C., Huang, C., Zhang, G., Zhang, G., Li, H., Zhu, J., Chen, J., Chang, J., et al. Yi: Open foundation models by 01. ai. arXiv preprint arXiv:2403.04652, 2024. Zhang, D., Li, J., Huang, X., Zhou, D., Li, Y., and Ouyang, W. Accessing gpt-4 level mathematical olympiad solutions via monte carlo tree self-refine with llama-3 8b. arXiv preprint arXiv:2406.07394, 2024a. Zhang, D., Wu, J., Lei, J., Che, T., Li, J., Xie, T., Huang, X., Zhang, S., Pavone, M., Li, Y., et al. Llama-berry: Pairwise optimization for o1-like olympiad-level mathematical reasoning. arXiv preprint arXiv:2410.02884, 2024b. Zhang, D., Zhoubian, S., Hu, Z., Yue, Y., Dong, Y., and Tang, J. Rest-mcts*: Llm self-training via process reward guided tree search. Advances in Neural Information Processing Systems, 2024c. Zhao, A., Ye, F., Fu, J., and Shen, X. Unveiling in-context learning: A coordinate system to understand its working mechanism. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp.
12375–12400, Miami, Florida, USA, November 2024. Association for Computational Linguistics. Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.
12375–12400, Miami, Florida, USA, November 2024. Association for Computational Linguistics. Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023. Zhou, A., Yan, K., Shlapentokh-Rothman, M., Wang, H., and Wang, Y.-X. Language agent tree search unifies reasoning, acting, and planning in language models. In Forty-first International Conference on Machine Learning, 2024a. Zhou, A., Yan, K., Shlapentokh-Rothman, M., Wang, H., and Wang, Y.-X. Language agent tree search unifies reasoning, acting, and planning in language models. In Forty-first International Conference on Machine Learning, 2024b. Zhou, Y., Li, J., Xiang, Y., Yan, H., Gui, L., and He, Y. The mystery of in-context learning: A comprehensive survey on interpretation and analysis. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 14365–14378, Miami, Florida, USA, November 2024c. Association for Computational Linguistics.
Zhou, Y., Li, J., Xiang, Y., Yan, H., Gui, L., and He, Y. The mystery of in-context learning: A comprehensive survey on interpretation and analysis. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 14365–14378, Miami, Florida, USA, November 2024c. Association for Computational Linguistics.
