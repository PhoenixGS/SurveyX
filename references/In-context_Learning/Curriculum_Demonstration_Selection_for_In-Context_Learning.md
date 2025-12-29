# Curriculum Demonstration Selection for In-Context Learning
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/013b/013bc3d6-1b51-4b77-a835-6577be786f1a.png" style="width: 50%;"></div>
27 Nov 2024
]  27 Nov 20
# Abstract
Abstract
Large Language Models (LLMs) have shown strong in-context learning (ICL) abilities with a few demonstrations. However, one critical challenge is how to select demonstrations to elicit the full potential of LLMs. In this paper, we propose Curriculum Demonstration Selection (CDS), a novel demonstration selection method for ICL. Instead of merely using similarity, CDS additionally partitions samples by their complexity measurements. Following curriculum learning, CDS then selects demonstrations from easy to difficult. Thus the selected demonstrations cover a wide range of difficulty levels, enabling LLMs to learn from varied complexities within the training set. Experiments demonstrate that our CDS consistently outperforms baseline methods, achieving notable improvements across nine LLMs on three benchmarks. Moreover, CDS proves especially effective in enhancing LLM performance in solving challenging problems.
arXiv:2411.1812
arXiv:2
# Keywords
ACM proceedings, LATEX, text tagging
ACM Reference Format: Duc Anh Vu, Nguyen Tran Cong Duy, Xiaobao Wu, Hoang Minh Nhat, Du Mingzhe, Nguyen Thanh Thong, and Anh Tuan Luu. 2025. Curriculum Demonstration Selection for In-Context Learning. In Proceedings of ACM ∗Corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. SAC’25, March 31 –April 4, 2025, Sicily, Italy © 2025 ACM. ACM ISBN 979-8-4007-0629-5/25/03 https://doi.org/xx.xxx/xxx_x
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. SAC’25, March 31 –April 4, 2025, Sicily, Italy © 2025 ACM. ACM ISBN 979-8-4007-0629-5/25/03 https://doi.org/xx.xxx/xxx_x
SAC Conference (SAC’25). ACM, New York, NY, USA, Article 4, 8 pages. https://doi.org/xx.xxx/xxx_x
# 1 Introduction
Large language models (LLMs) have made significant strides in natural language processing (NLP) [34, 44, 49], excelling in tasks such as natural language understanding [3, 14, 16, 47], text generation [30, 31, 35, 51] and reasoning [20, 32, 33, 36]. Despite these advances, the full potential of LLMs remains limited by how they generalize across tasks of varying complexity. As LLMs tackle increasingly diverse and complex real-world challenges, optimizing their ability to handle varying task difficulties is crucial. In-Context Learning (ICL) has emerged as a powerful paradigm in which LLMs perform tasks by leveraging examples embedded within the input prompt, without the need for parameter updates. The success of ICL, however, is highly sensitive to the choice of demonstrations included in the prompt. Studies have shown that selecting the right set of demonstrations can significantly influence model performance [6, 25]. Despite this, existing methods for demonstration selection often rely on random selection or heuristic approaches that may lead to inconsistent and suboptimal outcomes, particularly in more complex tasks. Although various approaches have been proposed to improve demonstration selection, including unsupervised [25, 29, 41, 48] and supervised methods [28, 38, 45, 52], these techniques often fail to account for the varying difficulty of the demonstrations. As a result, LLMs are limited in their ability to generalize effectively across tasks that span different levels of complexity. This shortcoming becomes especially problematic when models must solve a wide range of problems, from simple to highly intricate. To address this gap, we draw inspiration from curriculum learning, where learners gradually progress from simpler to more challenging tasks. We propose Curriculum Demonstration Selection (CDS), a method that systematically selects demonstrations based on their complexity to ensure a balanced representation. It first
splits the dataset into distinct difficulty levels, then retrieves demonstrations from each group. By adopting this curriculum-based approach, CDS enables LLMs to incrementally refine their understanding, which improves their performance on both simple and complex tasks. Our experiments demonstrate that CDS consistently outperforms baseline methods, particularly on more difficult problems where traditional selection techniques often fall short. The contributions of this paper are threefold: (i) We introduce Curriculum Demonstration Selection (CDS), a novel method for optimizing demonstration selection in ICL, leveraging curriculum learning to enhance LLM performance. (ii) We empirically demonstrate the effectiveness of CDS across multiple benchmarks, showing significant improvements over existing methods. (iii) We provide insights into the behavior of LLMs when exposed to demonstrations of varying complexity, highlighting the potential of CDS to boost challenging problem-solving capabilities.
# 2 Related Work
# 2.1 Demonstration Selection
In-context learning (ICL) has garnered increasing attention due to its effectiveness in leveraging demonstrations for LLMs without necessitating parameter updates. A critical challenge in ICL lies in selecting optimal demonstrations, as research has shown that they can significantly influence performance [6]. Several strategies have been explored for demonstration selection. Early research use random selection [3, 22] or human-crafted examples [19, 46]. While straightforward to implement, these methods often result in inconsistent performance, as some demonstrations may not provide effective guidance to the model. To mitigate this issue, Liu et al. [25] proposed a retrieval-based method where examples are selected based on their semantic similarity to the query. Another variant [45] uses a latent variable model to explain the effectiveness of certain demonstrations, showing that semantically similar examples can enhance generalization across tasks. Beyond semantic similarity, other strategies have emerged. Complexity-based selection [10] emphasizes selecting demonstrations that involve more reasoning steps or higher complexity. Furthermore, Zhao et al. [53] presents kNN-ICL, a method that integrates ICL with a nearest-neighbor search to simplify prompt engineering and enhance compositional task-oriented parsing. In addition, there are more advanced selection processes based on metrics such as informativeness [24], perplexity [11] and concept learning [45].
# 2.2 Curriculum Learning
The concept of curriculum learning [2] has inspired extensive research across numerous NLP tasks including ICL. Drozdov et al. [7] prioritized harder demonstrations, estimated by Demonstration Query Likelihood (DQL). The most challenging samples, represented by low-probability queries, are selected assuming they contribute to larger updates in the learning process, emulating how gradient updates work during training. Liu et al. [27] introduced In-Context Curriculum Learning (ICCL), which advocates ordering already selected demonstrations from simple to complex. Additionally, other approaches to demonstration selection in ICL have emerged that align with curriculum learning principles. For
example, Sorensen et al. [41] proposed an information-theoretic framework designed to select demonstrations that maximize information gain, a strategy that can be interpreted as a form of curriculum learning where examples are chosen for their informativeness rather than difficulty. Despite these advancements, most existing methods focus primarily on selecting demonstrations similar to the test instance, according to predefined metrics, and overlook the importance of incorporating a set of demonstrations with varying complexity levels. This lack of diversity may hinder the model’s ability to generalize effectively across a broader spectrum of complexities.
# 3 Methodology
We introduce Curriculum Demonstration Selection (CDS), a novel strategy for selecting examples that draws inspiration from curriculum learning. CDS ensures that selected demonstrations cover a wide spectrum of difficulty levels, enabling LLMs to progressively learn from varied complexities within the training set.
# 3.1 Problem Formulation
Given an LLM 𝜃and a set of 𝑛training pairs {𝑥𝑖,𝑦𝑖}𝑛 𝑖=1, our objective is to select 𝑘demonstrations from this set to guide the model 𝜃in solving a specific task 𝑆. We hypothesize that demonstrations with diverse complexity levels will enhance the model’s performance. Therefore, CDS aims to acquire demonstrations from different difficulty levels.
# 3.2 Curriculum Construction
To construct the curriculum, we rely on human-annotated complexity measures. Complexity is assessed based on factors such as grade level, with higher grade levels indicating more challenging examples. In cases where examples share the same complexity level, we further differentiate their difficulty using human performance metrics, such as task completion rates (e.g., acceptance rates in coding tasks). The dataset is divided into 𝑘distinct difficulty partitions, where 𝑘is determined by the distribution of complexity scores in the dataset. This partitioning enables the construction of a curriculum that spans a broad spectrum of example complexities, ensuring that the LLM is exposed to both simpler and more complex tasks.
# 3.3 Demonstration Retrieval
Once the dataset is partitioned into 𝑘difficulty levels, the CDS algorithm, outlined in Algorithm 1, selects one demonstration from each partition. This process ensures that the selected demonstrations are representative of different complexities, allowing the LLM to leverage a balanced set of examples. The retrieval process can either be similarity-based or random. For similarity-based retrieval, we use CLS embeddings from a pre-trained Transformers model to represent the sentences, following KATE [25]. The algorithm then selects demonstrations that are most similar to the test question from each difficulty level by calculating the negative Euclidean distance. We then shuffle the demonstration order to prevent overfitting to specific patterns and ensure unbiasedness.
MATH
ARC-c
Algebra
Geometry
Precalculus
Number Theory
Probability
Avg.
Llama 2 7B
4.58±0.33
1.11±0.1
0.92±0.15
3.27±0.23
2.18±0.26
3.48±0.17
57.37±0.43
+ KATE
4.93±0.27
2.02±0.69
2.26±0.31
4.26±0.36
2.88±0.4
4.09±0.08
57.54±0.43
+ CDS (ours)
5.66±0.29
2.16±0.26
2.08±0.09
4.57±0.68
3.59±0.17
4.62±0.24
57.99±0.26
Llama 2 13B
6.7±0.06
2.92±0.59
1.04±0.09
3.46±0.78
3.09±0.2
5.03±0.09
65.02±0.12
+ KATE
7.69±0.19
2.99±0.86
2.08±0.38
4.57±0.23
4.64±0.17
6.0±0.19
66.24±0.32
+ CDS (ours)
7.94±0.1
3.2±0.1
2.32±0.17
5.0±0.4
4.92±0.5
6.27±0.06
66.47±0.64
Llama 3 8B
25.91±0.25
12.18±1.45
8.18±0.53
17.47±0.86
16.60±0.95
20.87±0.38
80.29±0.28
+ KATE
27.14±0.18
13.15±0.59
9.65±0.23
16.36±0.09
20.39±0.78
22.09±0.11
81.11±0.2
+ CDS (ours)
27.09±0.52
15.31±0.71
10.13±0.71
17.96±0.76
21.31±0.17
22.57±0.28
81.0±0.15
Mistral 7B
12.71±0.26
5.15±1.0
4.7±0.17
5.74±0.76
7.88±0.43
9.9±0.03
75.37±0.35
+ KATE
14.9±0.3
6.61±0.35
6.41±0.3
6.23±0.71
7.74±0.26
11.57±0.12
76.59±0.15
+ CDS (ours)
14.89±0.03
6.26±0.3
6.47±0.09
6.42±0.87
8.65±0.52
11.64±0.02
76.71±0.25
Qwen 7B
13.75±0.14
5.43±1.02
5.56±1.27
6.54±0.71
8.79±0.1
10.81±0.21
48.66±0.79
+ KATE
15.25±0.25
6.96±0.43
6.41±0.79
7.9±0.75
9.14±0.53
12.12±0.25
44.62±0.24
+ CDS (ours)
15.54±0.26
7.24±0.94
5.92±0.46
8.64±1.06
9.7±0.6
12.39±0.04
49.74±0.48
Table 1: Result on MATH and ARC-c dataset. Numbers in bold indicate the best scores among all methods.
± ± ± ± ± ± ± Table 1: Result on MATH and ARC-c dataset. Numbers in bold indicate the best scores among all methods.
± ± ± ± ± ± ± Table 1: Result on MATH and ARC-c dataset. Numbers in bold indicate the best scores among all methods.
± ± ± ± ± ± ± ult on MATH and ARC-c dataset. Numbers in bold indicate the best scores among all methods.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b9ca/b9caef9c-063a-47c9-b4ff-9e2d0aeb6ae8.png" style="width: 50%;"></div>
Algorithm 1 Curriculum Demonstration Selection (CDS)
1: Input: LLM𝜃, Training set𝑇, Testing set 𝐸, Difficulty Measurer
𝐷, Retrieval function 𝑅, number of demonstrations 𝑘
2: Output: Selected demonstrations for each test instance
3: {𝑇𝑖}𝑘
𝑖=1 = 𝐷(𝑇) ⊲Partition the training set 𝑇into 𝑘subsets by
difficulty using 𝐷
4: for each 𝑡𝑒𝑠𝑡instance in 𝐸do
5:
𝑑𝑒𝑚𝑜𝑛𝑠𝑡𝑟𝑎𝑡𝑖𝑜𝑛_𝑝𝑜𝑜𝑙←empty list
6:
for 𝑖= 1 to 𝑘do
7:
𝑑𝑒𝑚𝑜𝑛𝑠𝑡𝑟𝑎𝑡𝑖𝑜𝑛_𝑝𝑜𝑜𝑙.add(𝑅(𝑇𝑖))
8:
end for
9:
𝑑𝑒𝑚𝑜𝑛𝑠𝑡𝑟𝑎𝑡𝑖𝑜𝑛_𝑝𝑜𝑜𝑙.shuffle()
10:
𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛= 𝜃(𝑑𝑒𝑚𝑜𝑛𝑠𝑡𝑟𝑎𝑡𝑖𝑜𝑛_𝑝𝑜𝑜𝑙)
11: end for
# 4 Experiments
# 4 Experiments 4.1 Experimental Setup
# 4.1 Experimental Setup
• Math reasoning. Mathematical reasoning plays a crucial role in evaluating LLMs’ performance. Numerous studies have focused on leveraging LLMs on math-related tasks such as mathematical question generation [21, 35] and problem solving [4, 12, 54]. In this study, we utilize the MATH dataset [15], a highly challenging benchmark consisting of 7,500 training examples and 5,000 testing examples, spanning topics such as prealgebra, algebra, number theory, counting and probability, geometry, intermediate algebra, and precalculus. The MATH dataset, sourced from competitions such as the AMC 10, AMC 12, and AIME, categorizes problems into five distinct complexity levels. We leverage this metadata to construct the curriculum for our CDS method, ensuring that demonstrations reflect a range of difficulty levels.
• Commonsense reasoning. Commonsense reasoning, essential for natural language understanding, tests a model’s ability to infer everyday scenarios by integrating observational and prior knowledge [39, 43]. To evaluate LLM performance in this area, we use the ARC-Challenge dataset [5], which contains 2,590 multiplechoice science questions targeting students from grade 3 to grade 9. For CDS, we rank the difficulty by the grade level data, ensuring a balanced representation across complexity levels. • Code generation. These tasks have become pivotal in assessing LLMs’ capabilities, covering areas from code correction [40, 50] to code summarization [42] and efficiency optimization [8]. For this task, we employ the Mercury dataset, a specialized benchmark designed for evaluating code generation correctness and efficiency. Mercury consists of 256 evaluation tasks sourced from public programming problems on Leetcode1, categorized into Easy, Medium, and Hard difficulty levels. To further refine difficulty classification, we use acceptance rates as an additional metric, with lower acceptance rates indicating higher complexity within the same difficulty category. 4.1.2 Models. For the math and commonsense reasoning tasks, we test five widely used open-source LLMs: Llama-2 (7B and 13B) [44], Llama-3 (8B) [9], Mistral-7B [17], and Qwen-7B [1]. For the code generation task, we follow the experimental settings of Mercury [8], evaluating four code-specialized LLMs: CodeLlama (7B and 13B) [37], StarCoder (3B) [23], and DeepSeek-Coder (6.7B) [13], with model sizes ranging from 3B to 13B parameters. 4.1.3 Implementation Details. We employ greedy decoding across all models for math and commonsense reasoning tasks. Prompts are constructed using the Chain-of-Thought (CoT) technique [46],
• Commonsense reasoning. Commonsense reasoning, essential for natural language understanding, tests a model’s ability to infer everyday scenarios by integrating observational and prior knowledge [39, 43]. To evaluate LLM performance in this area, we use the ARC-Challenge dataset [5], which contains 2,590 multiplechoice science questions targeting students from grade 3 to grade 9. For CDS, we rank the difficulty by the grade level data, ensuring a balanced representation across complexity levels.
• Code generation. These tasks have become pivotal in assessing LLMs’ capabilities, covering areas from code correction [40, 50] to code summarization [42] and efficiency optimization [8]. For this task, we employ the Mercury dataset, a specialized benchmark designed for evaluating code generation correctness and efficiency. Mercury consists of 256 evaluation tasks sourced from public programming problems on Leetcode1, categorized into Easy, Medium, and Hard difficulty levels. To further refine difficulty classification, we use acceptance rates as an additional metric, with lower acceptance rates indicating higher complexity within the same difficulty category.
4.1.2 Models. For the math and commonsense reasoning tasks, we test five widely used open-source LLMs: Llama-2 (7B and 13B) [44], Llama-3 (8B) [9], Mistral-7B [17], and Qwen-7B [1]. For the code generation task, we follow the experimental settings of Mercury [8], evaluating four code-specialized LLMs: CodeLlama (7B and 13B) [37], StarCoder (3B) [23], and DeepSeek-Coder (6.7B) [13], with model sizes ranging from 3B to 13B parameters.
4.1.3 Implementation Details. We employ greedy decoding across all models for math and commonsense reasoning tasks. Prompts are constructed using the Chain-of-Thought (CoT) technique [46], which encourages step-by-step reasoning before arriving at the final answer. The final prediction is extracted from the last line of the
1https://leetcode.com/problemset/algorithms/
model’s output. In the Mercury dataset, we adhere to the settings of Du et al. [8], with a decoding temperature of 0.2. All experiments use 𝑘= 5 demonstrations, and each model is evaluated using three different random seeds on the test sets. Example prompts in our experiments are provided in Appendix A.
4.1.4 Baselines. We compare CDS with two baseline methods: • Uniform: We uniformly select 𝑘demonstrations from training set 𝑇for each test example. • Similarity (KATE [25]): This method utilizes RoBERTa-Large [26] to obtain CLS embeddings, followed by cosine similarity computation between candidate demonstrations and the test examples. The 𝑘most similar demonstrations are then selected based on negative Euclidean distance.
4.1.5 Evaluation metrics. For the math and commonsense reasoning tasks, accuracy is used as the evaluation metric. A prediction is considered correct if it exactly matches the ground truth, either as a value (MATH) or a selected option (ARC-Challenge). For the code generation task, we use two key metrics from the Mercury dataset. The first is Functional Correctness (Pass), which evaluates whether the generated code successfully passes all test cases associated with a given task. The Pass score represents the percentage of tasks for which the code is fully functional. The second metric, Beyond, evaluates not only the correctness of the generated code but also its efficiency. This metric ranks the runtime performance of the model-generated code against historical solutions, providing a percentile score that reflects how efficiently the code executes.
# 4.2 Results
4.2.1 LLM Reasoning. We compared five widely-used open-source LLMs on mathematical and commonsense reasoning tasks, where the results clearly demonstrate the superiority of CDS over other methods, as seen in Table 1. In the mathematical reasoning benchmark, CDS consistently outperforms both random selection and the KATE baseline. Across all mathematical topics in the MATH dataset, CDS delivers stronger performance, with algebra and number theory benefiting the most, while geometry and precalculus show moderate improvements. Notably, probability problem solve rates increase substantially with CDS. Larger models, particularly Llama 2 13B, experience the greatest performance gains, underscoring the scalability of CDS in enhancing model capabilities, in line with the Scaling Laws proposed by Kaplan et al. [18]. Similarly, on the ARC-c benchmark, CDS outperforms both random selection and KATE. While KATE provides some improvement over random selection, its performance is inconsistent across tasks. For example, KATE underperforms the random baseline by approximately 9% with Qwen 7B. In contrast, CDS exhibits greater stability, consistently yielding superior results across the majority of cases. This highlights CDS as an effective and reliable method for in-context learning in LLMs.
4.2.2 Code Generation. CDS also demonstrates significant improvements in code generation tasks, as evaluated on the Mercury dataset in Table 2. Specifically, CDS significantly outperforms the random baseline in both the Pass and Beyond metrics for all LLMs. Furthermore, CDS consistently provides improvement over the KATE baseline on the Pass metric across all models and shows
Model
Pass
Beyond
CodeLlama-7b
28.12±0.68
21.75±0.65
+ KATE
36.85±0.9
28.34±1.8
+ CDS (ours)
36.98±1.93
28.1±2.0
CodeLlama-13b
33.98±1.03
26.86±0.9
+ KATE
39.45±0.01
32.24±0.27
+ CDS (ours)
41.54±0.45
33.18±0.76
deepseek-coder-6.7b
43.36±0.68
33.95±0.52
+ KATE
64.19±0.9
52.24±0.11
+ CDS (ours)
64.71±0.9
51.51±1.38
starcoder2-3b
50.26±1.13
39.51±0.67
+ KATE
57.42±1.35
46.58±3.28
+ CDS (ours)
58.98±0.01
47.15±1.06
± ± Table 2: Results on Mercury. Numbers in bold indicate the best scores among all methods.
± ± Table 2: Results on Mercury. Numbers in bold indicate the best scores among all methods.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/31d9/31d9a287-7989-4863-89a5-d9d09a475818.png" style="width: 50%;"></div>
Figure 1: Comparison of CDS with random demonstration retrieval and with similarity retrieval on MATH benchmark across five LLMs.
comparable results on the Beyond metric. These results highlight the effectiveness of CDS on the code generation task.
# comparable results on the Beyond metric. These results highlight the effectiveness of CDS on the code generation task.
4.2.3 Random retrieval v.s. Similarity retrieval. We examined the effects of different retrieval functions, namely random retrieval and similarity retrieval, on CDS performance. As shown in Figure 1, both retrieval methods outperform the random selection baseline on the MATH dataset, with similarity-based retrieval consistently outperforming random retrieval. This result suggests that similaritybased retrieval enhances the effectiveness of CDS, providing better demonstration selection.
4.2.4 Solve rates on harder problems. Both the MATH and ARC-c datasets show a clear positive trend, with harder problems benefiting the most from CDS, as illustrated in Figure 2. On the MATH dataset, the performance improvement grows from 2% on easier tasks to 6% on harder ones. The ARC-c shows a similar, albeit more modest, trend, with improvements ranging from negative values on easy tasks to approximately 1% on medium and hard tasks. This
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0e14/0e141981-6790-4d8c-aeb0-e053b382b156.png" style="width: 50%;"></div>
Figure 2: Average improvement across five models in three difficulty levels.
<div style="text-align: center;">Figure 2: Average improvement across five models in three difficulty levels.</div>
highlights the remarkable efficacy of CDS in addressing more complex problems, likely due to its strategy of selecting demonstrations across a wide range of difficulty levels, helping LLMs adapt better to challenging tasks. 4.2.5 Effect of demonstrations’ order. As demonstrated in Table 3, ordering the demonstrations from easy to hard (E2H) has no significant effect on performance compared to random shuffling, indicating that the order in which demonstrations are presented does not impact the overall effectiveness of CDS. This finding aligns with previous research by Wang et al. [45], further supporting the robustness of CDS in enhancing LLM performance, regardless of demonstration order.
highlights the remarkable efficacy of CDS in addressing more complex problems, likely due to its strategy of selecting demonstrations across a wide range of difficulty levels, helping LLMs adapt better to challenging tasks.
4.2.5 Effect of demonstrations’ order. As demonstrated in Table 3, ordering the demonstrations from easy to hard (E2H) has no significant effect on performance compared to random shuffling, indicating that the order in which demonstrations are presented does not impact the overall effectiveness of CDS. This finding aligns with previous research by Wang et al. [45], further supporting the robustness of CDS in enhancing LLM performance, regardless of demonstration order.
Alg.
Geo.
Cal.
Num. Theory
Prob.
Avg.
E2H
5.47
2.51
2.20
4.63
3.37
4.54
Rand.
5.66±0.29
2.16±0.26
2.08±0.09
4.57±0.68
3.59±0.17
4.62±0.24
Table 3: Accuracy of our method with (E2H) and without (Rand.) reordering. Numbers are obtained with the Llama-2 7B model on MATH dataset.
# 5 Conclusions
In this paper, we introduced Curriculum Demonstration Selection (CDS), a novel approach aimed to enhance the performance of LLMs in ICL. By leveraging the principles of curriculum learning, CDS effectively organizes demonstrations according to their complexity, allowing LLMs to learn from simpler to more complex tasks. Our extensive experiments across three benchmarks — mathematical reasoning, commonsense reasoning, and code generation — consistently demonstrated that CDS outperforms traditional methods, including both random selection and similarity-based approaches like KATE. Additionally, CDS shows significant potential in solving more complex and challenging problems, highlighting its robustness in optimizing demonstration selection for ICL. Overall,
CDS presents a significant step forward in optimizing demonstration selection for ICL, providing a more structured and effective method that enhances both the accuracy and efficiency of LLMs. This work opens new avenues for improving LLM in numerous problem-solving tasks.
# 6 Limitations
Our work does have some limitations. First, we employed a fixed five-shot setting in all experiments, providing the model with five demonstrations per task. While this is a common approach in incontext learning, it may not fully capture the potential of CDS when applied to different shot settings. The decision to use five demonstrations balanced computational costs and performance consistency across benchmarks. Future research could explore how varying the number of demonstrations influences CDS effectiveness, particularly for more complex tasks that may require additional context for optimal performance. Second, the curriculum structure in CDS relies on predefined metadata, such as grade levels in math reasoning, to determine task difficulty. While this allows for efficient curriculum construction, it assumes that such metadata is available and accurately reflects task complexity. In some datasets or domains, difficulty rankings may not be available or may not accurately represent the challenges posed by the tasks. In these cases, CDS would require alternative methods for estimating task complexity, such as model-based estimates or task-specific heuristics, to maintain its effectiveness. Lastly, while our evaluation focused on three benchmarks, namely mathematical reasoning, commonsense reasoning, and code generation, the generalizability of CDS to other types of tasks remains an open question. Future work could expand the scope of CDS evaluation to include a broader range of benchmarks, helping to identify its strengths and limitations across various domains and task types. This will provide deeper insights into the applicability of CDS and its potential to further improve LLM performance in diverse contexts.
# References
[1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609 (2023). [2] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. 2009. Curriculum learning. In International Conference on Machine Learning. https: //api.semanticscholar.org/CorpusID:873046 [3] Tom B Brown. 2020. Language models are few-shot learners. arXiv preprint arXiv:2005.14165 (2020). [4] Zui Chen, Yezeng Chen, Jiaqi Han, Zhijie Huang, Ji Qi, and Yi Zhou. 2024. An Empirical Study of Data Ability Boundary in LLMs’ Math Reasoning. arXiv preprint arXiv:2403.00799 (2024). [5] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457 (2018). [6] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2022. A survey on in-context learning. arXiv preprint arXiv:2301.00234 (2022). [7] Andrew Drozdov, Honglei Zhuang, Zhuyun Dai, Zhen Qin, Razieh Rahimi, Xuanhui Wang, Dana Alon, Mohit Iyyer, Andrew McCallum, Donald Metzler, and Kai Hui. 2023. PaRaDe: Passage Ranking using Demonstrations with LLMs. In Findings of the Association for Computational Linguistics: EMNLP 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 14242–14252. https://doi.org/10.18653/v1/2023.findingsemnlp.950 [8] Mingzhe Du, Anh Tuan Luu, Bin Ji, and See-Kiong Ng. 2024. Mercury: An efficiency benchmark for llm code synthesis. arXiv preprint arXiv:2402.07844 (2024).
[9] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024). [10] Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. 2022. Complexity-based prompting for multi-step reasoning. In The Eleventh International Conference on Learning Representations. [11] Hila Gonen, Srini Iyer, Terra Blevins, Noah Smith, and Luke Zettlemoyer. 2023. Demystifying Prompts in Language Models via Perplexity Estimation. In Findings of the Association for Computational Linguistics: EMNLP 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 10136–10148. https://doi.org/10.18653/v1/2023.findings-emnlp.679 [12] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv:2309.17452 (2023). [13] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. 2024. DeepSeek-Coder: When the Large Language Model Meets Programming–The Rise of Code Intelligence. arXiv preprint arXiv:2401.14196 (2024). [14] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300 (2020). [15] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874 (2021). [16] Nhat Hoang, Xuan Long Do, Duc Anh Do, Duc Anh Vu, and Anh Tuan Luu. 2024. ToXCL: A Unified Framework for Toxic Speech Detection and Explanation. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), Kevin Duh, Helena Gomez, and Steven Bethard (Eds.). Association for Computational Linguistics, Mexico City, Mexico, 6460–6472. https://doi.org/10. 18653/v1/2024.naacl-long.359 [17] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7B. arXiv preprint arXiv:2310.06825 (2023). [18] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeff Wu, and Dario Amodei. 2020. Scaling Laws for Neural Language Models. ArXiv abs/2001.08361 (2020). https://api. semanticscholar.org/CorpusID:210861095 [19] Mehran Kazemi, Najoung Kim, Deepti Bhatia, Xin Xu, and Deepak Ramachandran. 2023. LAMBADA: Backward Chaining for Automated Reasoning in Natural Language. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 6547–6568. https://doi.org/10.18653/v1/2023.acl-long.361 [20] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems 35 (2022), 22199–22213. [21] Ghader Kurdi, Jared Leo, Bijan Parsia, Uli Sattler, and Salam Al-Emari. 2019. A Systematic Review of Automatic Question Generation for Educational Purposes. International Journal of Artificial Intelligence in Education 30 (2019), 121–204. https://api.semanticscholar.org/CorpusID:208212657 [22] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. 2022. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems 35 (2022), 3843–3857. [23] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. 2023. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161 (2023). [24] Xiaonan Li and Xipeng Qiu. 2023. Finding support examples for in-context learning. arXiv preprint arXiv:2302.13539 (2023). [25] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2022. What Makes Good In-Context Examples for GPT-3?. In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, Eneko Agirre, Marianna Apidianaki, and Ivan Vulić (Eds.). Association for Computational Linguistics, Dublin, Ireland and Online, 100–114. https://doi.org/10.18653/v1/2022.deelio1.10 [26] Yinhan Liu. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 (2019). [27] Yinpeng Liu, Jiawei Liu, Xiang Shi, Qikai Cheng, and Wei Lu. 2024. Let’s Learn Step by Step: Enhancing In-Context Learning Ability with Curriculum Learning. arXiv preprint arXiv:2402.10738 (2024). [28] Thong Nguyen, Yi Bin, Xiaobao Wu, Xinshuai Dong, Zhiyuan Hu, Khoi Le, CongDuy Nguyen, See-Kiong Ng, and Luu Anh Tuan. 2025. Meta-optimized Angular Margin Contrastive Framework for Video-Language Representation Learning. In European Conference on Computer Vision. Springer, 77–98.
[29] Thong Nguyen and Anh Tuan Luu. 2021. Contrastive learning for neural topic model. Advances in neural information processing systems 34 (2021), 11974–11986. [30] Thong Nguyen, Anh Tuan Luu, Truc Lu, and Tho Quan. 2021. Enriching and controlling global semantics for text summarization. arXiv preprint arXiv:2109.10616 (2021). [31] Thong Thanh Nguyen and Anh Tuan Luu. 2022. Improving neural cross-lingual abstractive summarization via employing optimal transport distance for knowledge distillation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 36. 11103–11111. [32] Fengjun Pan, Xiaobao Wu, Zongrui Li, and Luu Anh Tuan. 2024. Are LLMs Good Zero-Shot Fallacy Classifiers?. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 14338–14364. [33] Liangming Pan, Xiaobao Wu, Xinyuan Lu, Anh Tuan Luu, William Yang Wang, Min-Yen Kan, and Preslav Nakov. 2023. Fact-Checking Complex Claims with Program-Guided Reasoning. In Annual Meeting of the Association for Computational Linguistics (ACL). Association for Computational Linguistics, Toronto, Canada, 6981–7004. [34] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023. The RefinedWeb dataset for Falcon LLM: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116 (2023). [35] Phuoc Van Long Pham, Anh Vu Duc, Nhat Minh Hoang, Xuan Long Do, and Anh Tuan Luu. 2024. ChatGPT as a Math Questioner? Evaluating ChatGPT on Generating Pre-university Math Questions. In Proceedings of the 39th ACM/SIGAPP Symposium on Applied Computing (Avila, Spain) (SAC ’24). Association for Computing Machinery, New York, NY, USA, 65–73. https://doi.org/10. 1145/3605098.3636030 [36] Shuofei Qiao, Yixin Ou, Ningyu Zhang, Xiang Chen, Yunzhi Yao, Shumin Deng, Chuanqi Tan, Fei Huang, and Huajun Chen. 2022. Reasoning with language model prompting: A survey. arXiv preprint arXiv:2212.09597 (2022). [37] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, et al. 2023. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950 (2023). [38] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2022. Learning To Retrieve Prompts for In-Context Learning. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz (Eds.). Association for Computational Linguistics, Seattle, United States, 2655–2671. https://doi.org/10.18653/v1/2022.naacl-main.191 [39] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Commun. ACM 64, 9 (2021), 99–106. [40] Dominik Sobania, Martin Briesch, Carol Hanna, and Justyna Petke. 2023. An analysis of the automatic bug fixing performance of chatgpt. In 2023 IEEE/ACM International Workshop on Automated Program Repair (APR). IEEE, 23–30. [41] Taylor Sorensen, Joshua Robinson, Christopher Rytting, Alexander Shaw, Kyle Rogers, Alexia Delorey, Mahmoud Khalil, Nancy Fulda, and David Wingate. 2022. An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (Eds.). Association for Computational Linguistics, Dublin, Ireland, 819–862. https://doi.org/10.18653/v1/2022.acl-long.60 [42] Weisong Sun, Chunrong Fang, Yudu You, Yun Miao, Yi Liu, Yuekang Li, Gelei Deng, Shenghan Huang, Yuchen Chen, Quanjun Zhang, et al. 2023. Automatic code summarization via chatgpt: How far are we? arXiv preprint arXiv:2305.12865 (2023). [43] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Jill Burstein, Christy Doran, and Thamar Solorio (Eds.). Association for Computational Linguistics, Minneapolis, Minnesota, 4149–4158. https://doi.org/10.18653/v1/N19-1421 [44] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023). [45] Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, and William Yang Wang. 2024. Large language models are latent variable models: Explaining and finding good demonstrations for in-context learning. Advances in Neural Information Processing Systems 36 (2024). [46] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35 (2022), 24824–24837.
[47] Xiaobao Wu, Liangming Pan, William Yang Wang, and Luu Anh Tuan. 2024. AKEW: Assessing Knowledge Editing in the Wild. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 15118–15133. [48] Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, and Lingpeng Kong. 2023. SelfAdaptive In-Context Learning: An Information Compression Perspective for In-Context Example Selection and Ordering. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 1423–1436. https://doi.org/10. 18653/v1/2023.acl-long.79 [49] Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Peter Liu. 2020. PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization. In Proceedings of the 37th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 119), Hal Daumé III and Aarti Singh (Eds.). PMLR, 11328–11339. https://proceedings.mlr.press/v119/zhang20ae.html [50] Quanjun Zhang, Tongke Zhang, Juan Zhai, Chunrong Fang, Bowen Yu, Weisong Sun, and Zhenyu Chen. 2023. A critical review of large language model on software engineering: An example from chatgpt and automated program repair. arXiv preprint arXiv:2310.08879 (2023). [51] Tianyi Zhang, Faisal Ladhak, Esin Durmus, Percy Liang, Kathleen McKeown, and Tatsunori B Hashimoto. 2024. Benchmarking large language models for news summarization. Transactions of the Association for Computational Linguistics 12 (2024), 39–57. [52] Yiming Zhang, Shi Feng, and Chenhao Tan. 2022. Active Example Selection for In-Context Learning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association for Computational Linguistics, Abu Dhabi, United Arab Emirates, 9134–9148. https://doi.org/10.18653/v1/2022.emnlp-main.622 [53] Wenting Zhao, Ye Liu, Yao Wan, Yibo Wang, Qingyang Wu, Zhongfen Deng, Jiangshu Du, Shuaiqi Liu, Yunlong Xu, and Philip Yu. 2024. 𝑘NN-ICL: Compositional Task-Oriented Parsing Generalization with Nearest Neighbor In-Context Learning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), Kevin Duh, Helena Gomez, and Steven Bethard (Eds.). Association for Computational Linguistics, Mexico City, Mexico, 326–337. https://doi.org/10.18653/v1/2024.naacl-long.19 [54] Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, et al. 2023. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. arXiv preprint arXiv:2308.07921 (2023).
# A Prompts used for few-shot learning
### Question: A string 𝑠is called good if there are no two different characters in 𝑠that have the same frequency. Given a string 𝑠, return the minimum number of characters you need to delete to make 𝑠good. The frequency of a character in a string is the number of times it appears in the string. For example, in the string "aab", the frequency of ’a’ is 2, while the frequency of ’b’ is 1. \textbf{Example 1:} \textbf{Input:} \(s = \text{äab¨}\) \\ \textbf{Output:} 0 \\ \textbf{Explanation:} \(s\) is already good. \textbf{Example 2:} \textbf{Input:} \(s = \textbf{äaabbbcc¨}\) \\ \textbfOutput: 2 \\ \textbfExplanation: You can delete two ’b’s resulting in the good string "aaabcc". Another way is to delete one ’b’ and one ’c’, resulting in the good string "aaabbc". \textbf{Example 3:} \textbf{Input:}(s = \text{¨ceabaacb¨}) \textbf{Output:} 2 \textbf{Explanation:} You can delete both ’c’s, resulting in the good string "eabaab". Note that we only care about characters that are still in the string at the end (i.e., frequency of 0 is ignored). \textbf{Constraints:} \begin{itemize} \item \(1 \leq s.length \leq 105 \item \(s\) contains only lowercase English letters. \end{itemize} ### Code Prompt: class Solution(object): def minDeletions(self, s): ### Code Completion: import heapq class Solution(object): def minDeletions(self, s): lst = list(s) counter = {} for char in lst: if char not in counter: counter[char] = 0 counter[char] += 1 q = [] ans = 0 for char in counter: heapq.heappush(q, (-counter[char], char)) del counter while q: curr_ct, curr = heapq.heappop(q) if q: if q[0][0] == curr_ct: curr_ct += 1 ans += 1 if curr_ct < 0: heapq.heappush(q, (curr_ct, curr)) return ans ### Question: «Q2» ### Code Prompt: «S2» ### Code Completion: «E2» ### Question: «Q3» ### Code Prompt: «S3» ### Code Completion: «E3» ### Question: «Q4» ### Code Prompt: «S4» ### Code Completion: «E4» ### Question: «Q5» ### Code Prompt: «S5» ### Code Completion: «E5» ### Question: «Test Question» ### Code Prompt: «Test Code Prompt» e 4: Example of the prompt used in the experimen
Table 4: Example of the prompt used in the experiments on the Mercury dataset.
### Question: The least common multiple of two integers is 36 and 6 is their greatest common divisor. What is the product of the two numbers? ### Solution: Let $a$ and $b$ be the two integers. We can use the identity $\gcd(a,b) \cdot \mathop \textlcm[a,b] = ab$. Substituting gives that the answer is $36 \cdot 6 = \boxed{216}$. ### Extracted Answer: 216 ### Question: In an office at various times during the day, the boss gives the secretary a letter to type, each time putting the letter on top of the pile in the secretary’s in-box. When there is time, the secretary takes the top letter off the pile and types it. There are nine letters to be typed during the day, and the boss delivers them in the order $1, 2, 3, 4, 5, 6, 7, 8, 9$. While leaving for lunch, the secretary tells a colleague that letter 8 has already been typed, but says nothing else about the morning’s typing. The colleague wonders which of the nine letters remain to be typed after lunch and in what order they will be typed. Based upon the above information, how many such after-lunch typing orders are possible? (That there are no letters left to be typed is one of the possibilities.) Re-stating the problem for clarity, let $S$ be a set arranged in increasing order. At any time an element can be appended to the end of $S$, or the last element of $S$ can be removed. The question asks for the number of different orders in which the all of the remaining elements of $S$ can be removed, given that $8$ had been removed already. ### Solution: Since $8$ had already been added to the pile, the numbers $1 \ldots 7$ had already been added at some time to the pile; 9 might or might not have been added yet. So currently $S$ is a subset of $\{1, 2, \ldots 7\}$, possibly with $9$ at the end. Given that $S$ has $k$ elements, there are $k+1$ intervals for $9$ to be inserted, or $9$ might have already been placed, giving $k+2$ different possibilities. Thus, the answer is $\sum_{k=0}ˆ{7} \7 \choose k}(k+2)$ $= 1 \cdot 2 + 7 \cdot 3 + 21 \cdot 4 + 35 \cdot 5 + 35 \cdot 6 + 21 \cdot 7 + 7 \cdot 8 + 1 \cdot 9$ $= \boxed{704}$. ### Extracted Answer: 704 ### Question: «Q3» ### Solution: «S3» ### Extracted Answer: «E3» ### Question: «Q4» ### Solution: «S4» ### Extracted Answer: «E4» ### Question: «Q5» ### Solution: «S5» ### Extracted Answer: «E5» ### Question: «Test Question»
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/281f/281f1698-04da-443b-a366-5e218a031acf.png" style="width: 50%;"></div>
Table 5: Example of the prompt used in the experiments on the MATH dataset.
Table 5: Example of the prompt used in the experiments on the MATH dataset.
