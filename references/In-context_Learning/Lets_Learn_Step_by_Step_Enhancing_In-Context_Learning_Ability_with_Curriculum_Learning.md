# Let’s Learn Step by Step: Enhancing In-Context Learning Ability with Curriculum Learning

Yinpeng Liu, Jiawei Liu, Xiang Shi, Qikai Cheng, Yong Huang and Wei Lu School of Information Management, Wuhan University, China Information Retrieval and Knowledge Mining Laboratory, Wuhan University, China inpengliu, laujames2017, coding, chengqikai, yonghuang1991, weilu}@whu.edu.c

# Abstract

Demonstration ordering, which is an important strategy for in-context learning (ICL), can significantly affects the performance of large language models (LLMs). However, most of the current approaches of ordering require high computational costs to introduce the priori knowledge. In this paper, inspired by the human learning process, we propose a simple but effective demonstration ordering method for ICL, named the few-shot In-Context Curriculum Learning (ICCL). The ICCL implies gradually increasing the complexity of prompt demonstrations during the inference process. The difficulty can be assessed by human experts or LLMs-driven metrics, such as perplexity. Then we design extensive experiments to discuss the effectiveness of the ICCL at both corpus-level and instance-level. Moreover, we also investigate the formation mechanism of LLM’s ICCL capability. Experimental results demonstrate that ICCL, developed during the instruction-tuning stage, is effective for representative open-source LLMs. To facilitate further research and applications by other scholars, we make the code publicly available 1.

# 1 Introduction

Human education is methodical and incremental, building upon previously accumulated knowledge, which inspires curriculum  based algorithm designs in machine learning. Curriculum learning, introduced by Bengio et al. (2009), is originally a method that progressively raises the difficulty of the data samples utilized in the training process. Many studies have demonstrated the efficacy of curriculum learning applied in different models (Portelas et al., 2020; Nagatsuka et al., 2021) and different tasks (Wang et al., 2022; Xu et al., 2020). Since instruction-tuned Large Language Models (LLMs) exhibited remarkable proficiency in under

1 https://github.com/61peng/curri_learning

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/77cf/77cf7cac-4e4d-4b2e-8061-cfd60bc35fae.png" style="width: 50%;"></div>
Figure 1: Illustration of In-Context Curriculum Learning (ICCL). The curriculum schedule can be designed by both human and LLMs, schedule constructor sort demonstrations from easy to hard based on their understanding.

<div style="text-align: center;">Figure 1: Illustration of In-Context Curriculum Learning (ICCL). The curriculum schedule can be designed by both human and LLMs, schedule constructor sort demonstrations from easy to hard based on their understanding.
</div>
standing human intentions and generating humanlike text (Ouyang et al., 2022), researchers have initiated the integration of curriculum learning during instruction tuning (Feng et al., 2023; Lee et al., 2023). Aforementioned works demonstrate that curriculum learning facilitates accelerated convergence and the identification of better local minima during the parameter updating process. However, research on the effectiveness of curriculum learning within In-Context Learning (ICL) remains limited. Some methods that gradually prompt LLMs within instruction, such as Chain of Thought (CoT) (Wei et al., 2022), have significantly enhanced the ability of model to perform complex reasoning. This inspires us to apply curriculum learning for ICL. LLMs with varying performance are treated as students with varying learning abilities, and a human educator plays the role of a facilitator, guiding the learners through the curriculum. Under

the human-led curriculum, the models are gradually prompted to solve complex tasks. Is such a curriculum schedule effective, particularly when compared with many superior demonstration ordering algorithms? If this is the case, at what point is the model’s capacity to learn from a curriculum curriculum established? To answer those questions, we propose the In-Context Curriculum Learning (ICCL), as illurstrated in Figure 1. The ICCL framework encompasses two roles: curriculum constructor and curriculum learner. The curriculum constructor, which could be either human experts or LLMs, ranks the demonstrations based on their comprehension of difficulty. Subsequently, the learner is guided in progressively solving tasks. Our main contributions are as follows: (1) We propose the ICCL, a straightforward and effective demonstration ordering method, and validate the effectiveness of ICCL for open-source LLMs. (2) We adopt perplexity as the metric to assess the difficulty of demonstration, which outperformed many superior demonstration ordering methods. (3) Comparative analysis indicates that the ICCL capability of LLMs is developed during the instruction-tuning stage.

# 2 Related Work

Demonstrations Organization Numerous studies (Dong et al., 2022; Wan et al., 2023) show that the performance of LLMs is heavily influenced by the selection and ordering of demonstrations, indicating that different organizational approaches lead to the assimilation of distinct semantic information.
Lu et al. (2022) find that the performance of pretrained language models can vary from nearly stateof-the-art to random guess performance depending on how samples are ordered. This implies there exist multiple strategies for arranging prompt orders to enhance performance. They identify outstanding demonstration organizations based on entropy statistics. Liu et al. (2022) retrieve demonstrations that are semantically-similar to test source and order them by increasing cosine similarity. Wu et al. (2023) propose a ranking algorithm inspired by the compression viewpoint, which considers the codelength required to compress and transmit testing label. The codelength can be calculated using Shannon-Huffman code. Since the existing research has substantiated that demonstration organization can significantly af

# Demonstrations Organization

fects performance, we aim to delve into the optimization of prompt orders in ICL. To this end, we introduce curriculum learning strategies successfully employed in machine learning into ICL.

Curriculum Learning The concept of curriculum learning (Bengio et al., 2009) has inspired numerous research to address various natural language processing tasks. Wang et al. (2022) proposes a novel framework for Abstract Meaning Representation (AMR) parsing using hierarchical curriculum learning, achieving significant improvements on AMR2.0 and AMR3.0 benchmarks. Jia et al. (2023) introduces an approach applying curriculum learning to natural language generation (NLG) tasks. The authors propose a strategy that starts by training models to generate the final few words of a sequence, progressively extending to generate the entire sequence. However, the aforementioned studies all require adjustments to model parameters. There is currently a lack of research exploring curriculum learning in context.

# 3 Methodology

Inspired by curriculum learning employed in training process, we investigate a novel few-shot InContext Curriculum learning (ICCL), which is essentially a strategy for ordering demonstrations: sort the demonstration examples in order of increasing difficulty. This ordering strategy prompt LLM to absorb many skills and tasks within the parameters gradually.

# 3.1 Problem Formulation

Given a LLM θ, there are n demonstrations {(x i, y i)} n i =0 selected to instruct θ to solve specific task T. While trying to adapt θ for T, different demonstration orders D have different efficiency in utilizing parameters θ, the parameter-efficiency E p is measured by performance metrics. We hypothesizes that when demonstrations are arranged from simple to difficult, it will increase the model’s E p ad much as possible. Therefore, the objective of ICCL is to acquire an order D curriculum that:

(1)

ICCL remain parameters θ fixed throughout the process and merely modifies the ordering of D to progress from simple to complex. Consequently, the crux lies in the method of measuring the complexity of demonstrations.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f131/f131c2e0-ff97-4c5e-bfdc-ec8877961aba.png" style="width: 50%;"></div>
Method
SciCite
SciNLI
SciERC
Overall
Macro P
Macro F1
Accuracy
Macro F1
Micro F1
Avg F1
MIXTRAL-8X7B-INSTRUCT-V0.1
Random
69.74±3.76
62.57±0.94
42.38±0.11
37.21±0.10
23.91±0.32
41.23
VoteK
68.82±2.11
49.88±1.81
38.89±0.08
31.66±0.44
30.24±1.70
37.26
ICCL(Ours)
71.32±1.58
66.76±2.65
52.21±0.28
49.87±0.26
24.90±0.74
47.18
LLAMA 2-70B-CHAT
Random
64.99±0.45
59.37±0.18
39.68±0.45
34.31±0.38
24.40±5.28
39.36
VoteK
61.27±1.09
63.11±0.25
37.53±0.20
27.46±0.20
30.54±0.60
40.37
ICCL(Ours)
67.58±2.84
62.56±1.28
41.13±0.39
35.59±0.52
31.45±0.90
43.20
QWEN1.5-72B-CHAT
Random
75.19±0.75
74.70±0.38
48.38±0.24
45.85±0.31
19.51±0.32
46.59
VoteK
77.82±0.22
75.62±0.22
49.88±0.40
47.55±0.51
30.19±0.63
51.12
ICCL(Ours)
76.98±0.45
75.02±0.87
50.83±0.31
49.09±0.30
26.68±0.15
50.26
Table 1: Evaluation result of mainstream LLMs applying ICCL on three scientific datasets at corpus level. We adopt F 1 score as the core metric and perform averaging to get overall F 1 score. All the results are calculated based on 3 different random seeds over test set of each task. standard deviation are in small font. Dark and light blue colored cells stand for decline> 1% and <1% compared to Random baseline, respectively. Dark and light orange colored cells stand for improvement> 1% and <1%, respectively.

# 3.2 Curriculum Schedule Construction

We firstly rely on human experts to construct a curriculum-based context for LLMs at corpus level. Specifically, we engaged five human experts, ranging from undergraduates to professors, to rank the demonstrations based on their perceived difficulty, The final ordering was determined by averaging the rankings provided by each expert. To ensure the reliability of the final order, we employed Kendall’s coefficient of concordance as the agreement scores among experts. At instance level, appropriate samples can be selected for each test target using demonstrations retrieval algorithms (such as TopK (Liu et al., 2022)). The ranker shifts from humans to LLMs. While human experts can judge the demonstrations difficulty based on their understanding, LLMs may not perceive it the same way. An intuitive approach is to use perplexity to quantify the LLMs’ understanding of complexity. We retrieve n candidate samples that are most similar to the test target. Then, we calculate the complexity of each sample using:

Comp (x i, y i) = exp {− log p (y i |I θ (x i))}

where (x i, y i) represents a sample in the candidate set, and I θ (x i) denotes the instruction template of LLM θ with input x i. We measure the complexity of a sample by calculating the perplexity on the label y i given the specified instruction, and order the demonstrations {(x i, y i)} with lower perplexity first.

# 4 Experiments

# 4.1 Setup

Datasets Scientific papers are relatively complex discourse in the structured educational journey of human. The comprehension of scientific text requires domain expertise and logical reasoning capability. Therefore, corpus composed of scientific texts is selected as the benchmark evaluation set for estimating the effectiveness of curriculum learning applied during the inference stage of LLMs. We evaluate ICCL on three scientific dataset: SciCite (Cohan et al., 2019), SciNLI (Sadat and Caragea, 2022) and SciERC (Luan et al., 2018), encompassing tasks in text classification, natural language inference and information extraction.
Models We utilize a range of open-source performant LLMs, including LlaMA2 (Touvron et al., 2023), Mixtral-8x7B (Jiang et al., 2024), Qwen1.5 (Bai et al., 2023), exploring their applications within the ICCL framework.

Baseline For corpus-level methods, we consider VoteK (Su et al., 2022) and a Random baseline. For instance-level methods, we select KATE (Liu et al., 2022), TopK+LocalE (Lu et al., 2022), TopK+MDL (Wu et al., 2023) and a TopK baseline that select 5 demonstrations that are semantically closest to testing samples and rank them randomly. More experiment details are present in Appendix.

LLM
Method
SciCite
SciNLI
SciERC
Mixtral-8x7B
-Instruct-v0.1
TopK
64.85
34.69
32.98
+ KATE
64.19
35.20
32.48
+ LocalE
67.13
33.16
-
+ MDL
67.06
35.22
-
+ ICCL(Ours)
67.92
37.58
33.58
Llama2
-70B-Chat
TopK
60.86
39.70
38.54
+ KATE
57.11
39.81
38.71
+ LocalE
59.30
39.95
-
+ MDL
64.11
40.02
-
+ ICCL(Ours)
63.30
40.48
39.03
Table 2: Evaluation result of LLMs applying InstanceLevel ICCL on three scientific datasets. Numbers in bold indicate the highest F 1 among all methods. All the results are calculated based on 3 different random seeds over test set of each task.

# 4.2 Main Result

A comparison between 3 mainstream open-source LLMs across 3 NLP tasks shows the superiority of ICCL over other corpus-level methods, depicted in Table 1. The demonstrations for both ICCL and the random baseline were selected by human experts, while VoteK, built upon TopK, selected diverse yet representative examples using voting mechanism. ICCL maintains an ordering from simple to complex for all test samples, whereas the random and VoteK baseline employs a random order. Despite VoteK achieving commendable performance in certain settings, it exhibits a large standard deviation, indicating instability in its improvements. For Mixtral-8x7B, the performance of VoteK is approximately 10% lower than random baseline. While ICCL consistently achieves stable improvements across all LLMs and NLP tasks compared to random baseline, and shows an average improvement of 9% compared to random baseline, and 7.8% compared to VoteK. This demonstrates

# that heuristic curriculum learning methods are

At instance level, we employ the topK algorithm for demonstrations retrieval to construct candidate set. Subsequently, we utilize ordering algorithms such as KATE, LocalE, and MDL to generate performant prompts from this candidate set as baselines. As shown in Table 2, ICCL still registers decent improvements on most tasks compared with instance-level baselines. Our method achieves an overall improvement of 4.96% compared to TopK with random order and 5.48% compared to KATE for Mixtral. We also notice that LocalE and MDL are competitive on SciCite. However, these two methods are only suitable for classification tasks

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/68f5/68f5072d-2407-45ea-a0ff-179de230b1c1.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: F 1 scores improvement (or decline) rates for both Based LLMs and Instruction-Tuned LLMs using ICCL compared with random order.
</div>
with a limited search space and are difficult to apply to complex generative tasks. This highlights the versatility of our method, indicating that it can be applied to a wide range of NLP tasks.

# 4.3 Formation Mechanism of ICCL Capability

To explore the formation mechanism of ICCL, i.e., whether the model’s ability to learn the curriculum from context is established during pre-training or instruction-tuning, we conduct ICCL experiments on both the base models and the instruction-tuned models separately. Figure 2 shows that the performance of ICCL on the base model is unstable, with an average decrement of 7.16%. Even in instances where an improvement is observed, it is weaker compared to the enhancement effect on the corresponding instruction-tuned models. This evidences the base models’ lack of sensitivity to the curriculum-based demonstration order. It further intimates that the competency for ICCL is most likely acquired during the instruction-tuning stage.

# 5 Conclusion

In this work, we argue that gradually increase the complexity of the demonstrations in prompt can achieve better performance. To substantiate this claim, we propose In-Context Curriculum Learning (ICCL), a straightforward yet effective demonstration ordering method for both corpus-level and instance-level. We design three sets of experiments to exploring the validity and mechanism of curriculum learning within context. The experimental results affirm the effectiveness of ICCL on opensource LLMs.

Due to the limited timeframe of the experiment, we were unable to utilize the latest LLMs, such as Meta Llama 3 (AI@Meta, 2024). However, our experiments with the current mainstream models have demonstrated the effectiveness of the heuristic method of in-context curriculum learning. In future research, we will incorporate more recent LLMs to ensure the robustness of our method across different models.

# References

# AI@Meta. 2024. Llama 3 model card.

AI@Meta. 2024. Llama 3 model card.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.
Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. 2009. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48.
Arman Cohan, Waleed Ammar, Madeleine van Zuylen, and Field Cady. 2019.  Structural scaffolds for citation intent classification in scientific publications. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3586–3596, Minneapolis, Minnesota. Association for Computational Linguistics.
Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2022. A survey for in-context learning. arXiv preprint arXiv:2301.00234.
Tao Feng, Zifeng Wang, and Jimeng Sun. 2023. Citing: Large language models create curriculum for instruction tuning. arXiv preprint arXiv:2310.02527.
Qi Jia, Yizhu Liu, Haifeng Tang, and Kenny Zhu. 2023.
In-sample curriculum learning by sequence completion for natural language generation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11937–11950, Toronto, Canada. Association for Computational Linguistics.
Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.
Bruce W Lee, Hyunsoo Cho, and Kang Min Yoo. 2023. Instruction tuning with human curriculum. arXiv preprint arXiv:2310.09518.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.
Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. 2009. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48.

Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2022. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pages 100–114, Dublin, Ireland and Online. Association for Computational Linguistics.
Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland. Association for Computational Linguistics.
Yi Luan, Luheng He, Mari Ostendorf, and Hannaneh Hajishirzi. 2018. Multi-task identification of entities, relations, and coreference for scientific knowledge graph construction. In  Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3219–3232, Brussels, Belgium. Association for Computational Linguistics.
Koichi Nagatsuka, Clifford Broni-Bediako, and Masayasu Atsumi. 2021. Pre-training a BERT with curriculum learning by increasing block-size of input text. In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pages 989–996, Held Online. INCOMA Ltd.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.
Rémy Portelas, Cédric Colas, Katja Hofmann, and Pierre-Yves Oudeyer. 2020. Teacher algorithms for curriculum learning of deep rl in continuously parameterized environments. In Conference on Robot Learning, pages 835–853. PMLR.
Mobashir Sadat and Cornelia Caragea. 2022. SciNLI: A corpus for natural language inference on scientific text. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7399–7409, Dublin, Ireland. Association for Computational Linguistics.
Hongjin Su, Jungo Kasai, Chen Henry Wu, Weijia Shi, Tianlu Wang, Jiayi Xin, Rui Zhang, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, et al. 2022. Selective annotation makes language models better fewshot learners. arXiv preprint arXiv:2209.01975.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, et al. 2023. Efficient large language models: A survey. arXiv preprint arXiv:2312.03863, 1.
Peiyi Wang, Liang Chen, Tianyu Liu, Damai Dai, Yunbo Cao, Baobao Chang, and Zhifang Sui. 2022.  Hierarchical curriculum learning for AMR parsing. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 333–339, Dublin, Ireland. Association for Computational Linguistics.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837.
Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, and Lingpeng Kong. 2023.  Self-adaptive in-context learning: An information compression perspective for incontext example selection and ordering. In  Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1423–1436, Toronto, Canada. Association for Computational Linguistics.
Benfeng Xu, Licheng Zhang, Zhendong Mao, Quan Wang, Hongtao Xie, and Yongdong Zhang. 2020. Curriculum learning for natural language understanding. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6095–6104, Online. Association for Computational Linguistics.

# A Datasets

Dataset information is detailed in Table 3. Here are examples of each datasets:

# • SciCite

– Sentence: A direct consequence is that overheads for address translation have grown to dominate run time for many important workloads with large memory footprint [46, 113, 241, 302, 303, 375].
– Label: background

# • SciERC

– Sentence:  This paper presents an approach to the unsupervised learning of parts of speech which uses both morphological and syntactic information.
– Label:  [[’approach’, ’Generic’], [’unsupervised learning of parts of speech’, ’Task’], [’morphological and syntactic information’, ’OtherScientificTerm’]]

Dataset
Task
# Test
# Labels
# Demos
SciCite
Citation Intent Classification
1861
3
5
SciERC
Scientific Information Recognition
551
7
5
SciNLI
Scientific Language Inference
4000
4
4
– Sentence1: Without L alignment , we observe a reduction in both accuracy and BLEU on Yelp.
– Sentence2: this tendency is inconsistent on Amazon (i.e., -2.2 accuracy and +0.56 BLEU).
– Label: contrasting

# B Model

We use both publicly available and proprietary LLMs with the different model size as follow:

• Qwen1.5 (Bai et al., 2023) is the LLM family built by Alibaba Cloud. We select  Qwen1.572B-Chat, which is the largest version of Qwen series.

The prompt template of each LLM is shown in Table 4.

# C Evaluation Detail

At Corpus level, we arbitrarily select 4-5 demonstration examples from training set, which remain consistent across all test samples to eliminate the influence of demonstration selection. Table 5 show some example of demonstrations selected by different methods. For each ordering setting, the sequence of demonstrations in the prompt for all models is identical. To equitably evaluate the curriculum learning capabilities of diverse LLMs, we adopt the test set and evaluation metric inherent to each dataset. Concretely, we report macro-averaged values for

LLM
In-context Curriculum Learning Prompts
Mixtral
Template: [INST]{Task Description} + {Sentence}[/INST]{Label}</s>
[INST]{Test Input}[/INST]
Task Description: You are a scientific literature analyst. Extract scientific entities from
sentences. The scientific entity category includes [’Method’, ’Task’, ’Metric’, ’Material’,
’Generic’, ’OtherScientificTerm’, ’Generic’].
Demonstration: Sentence: text xi
Label: label yi
Test Input: Sentence: text x
Llama2
Template: <s>[INST] «SYS» {System Message}«/SYS»
{Task Description} + {Sentence}[/INST]{Label}</s>
<s>[INST]{Test Input}[/INST]
Task Description: Identify the intent of a citation in scientific papers. Choose the citation
intention of the following sentence from [’method’, ’background’, ’result’].
Demonstration: Sentence: text xi
Label: label yi
Test Input: Sentence: text x
Qwen1.5
Template: <|im_start|>system {System Message} <|im_end|>
<|im_start|>user{Task Description} + {Sentence}<|im_end|>
<|im_start|>assistant {Label}<|im_end|>
<|im_start|>user{Test Input}<|im_end|>
Task Description: Identify the semantic relationship between the following pair of
sentences. The semantic relationship includes [’reasoning’, ’entailment’, ’contrasting’,
’neutral’].
Demonstration: Sentence1: text xi1
Sentence2: text xi2
Label: label yi
Test Input: Sentence1: text x1
Sentence2: text x2
Table 4: The prompt template of In-Context Curriculum Learning for each LLM.

SciCite, micro-averaged values for SciERC, and accuracy and macro F 1 for SciNLI. We select F 1 score as the core metric for all tasks to ascertain the efficacy of ICCL. To guarantee the reproducibility of our experimental results, We run every experiment with 3 different random seed, and calculate average metric to report in paper.

Manual Select
Task
Citation Intent Class
W
0.968
Demos
Sentence: This result is consistent with the conclu-
sions of the aforementioned recent study of (34) and
reinforces them from a significantly broader perspec-
tive. Label: result
Sentence: To determine the cell velocity, Darcy’s
law may be used as the constitutive assump-
tion[21],[18],[22],[13]. Label: method
Sentence: This is clearly in contrast to the results of
earlier investigations (Laprise&Peltier1989a, Pierre-
humbert&Wyman1985, Clark&Peltier1977), where it
was found that the criteria for static and dynamic insta-
bilities are simultaneously satisfied. Label: result
Sentence: nest burrows in close proximity of one an-
other appears to be well founded as previously shown
by several studies that measured distances between kin
vs. non-kin nest burrows, including in long-term data
sets(King 1989b; Viblanc et al. 2010; Arnaud, Dobson
& Murie 2012; Dobson et al. 2012). (5) Label: back-
ground
Sentence: We employed three modelling approaches
that have successfully been applied in previous stud-
ies on species distribution (Guisan and Zimmermann
2000): generalised linear models (GLM, i.e. logistic
regression in this case), generalised additive models
(GAM), and classification and regression trees. Label:
method
Sent
as d
Mo˘0
Sent
1978
Gorm
Mes
bel: 
Sent
press
the I
SQO
meth
Sent
Poli,
Ell e
Mitc
2012
et al.
Sent
jimu
rema
studi
last t
list f
Task
Scientific Information R
0.936
Task
Citation Intent Classification
W
0.968
-
Demos
Sentence: This result is consistent with the conclu-
sions of the aforementioned recent study of (34) and
reinforces them from a significantly broader perspec-
tive. Label: result
Sentence: To determine the cell velocity, Darcy’s
law may be used as the constitutive assump-
tion[21],[18],[22],[13]. Label: method
Sentence: This is clearly in contrast to the results of
earlier investigations (Laprise&Peltier1989a, Pierre-
humbert&Wyman1985, Clark&Peltier1977), where it
was found that the criteria for static and dynamic insta-
bilities are simultaneously satisfied. Label: result
Sentence: nest burrows in close proximity of one an-
other appears to be well founded as previously shown
by several studies that measured distances between kin
vs. non-kin nest burrows, including in long-term data
sets(King 1989b; Viblanc et al. 2010; Arnaud, Dobson
& Murie 2012; Dobson et al. 2012). (5) Label: back-
ground
Sentence: We employed three modelling approaches
that have successfully been applied in previous stud-
ies on species distribution (Guisan and Zimmermann
2000): generalised linear models (GLM, i.e. logistic
regression in this case), generalised additive models
(GAM), and classification and regression trees. Label:
method
Sentence: Genotyping of the SNPs wa
as described previously (Gutknecht et
Mo˘0308ssner et al. 2006a, b). Label: me
Sentence: 1991) and empirical studies 
1978; Kruuk and Parish 1982; Mills 198
Gorman 1997; Geffen et al.
1992; P
Messier 2001; Valenzuela and Macdonal
bel: background
Sentence: Furthermore, the hospital an
pression scale (HAD) (Snaith and Zigmon
the IBS special scale for quality of life
SQOL) (Drossman et al., 2000) were u
method
Sentence: (Massie and Holland, 1984; Ci
Poli, 2001; Uchitomi et al., 2003; Katz
Ell et al., 2005; Boyd et al., 2012; Kim
Mitchell et al., 2012; Palmer et al., 201
2012; Tada et al., 2012; Warmenhoven et 
et al., 2012). Label: background
Sentence:were retrospective (Okada et a
jimura et al., 2002; Ramasamy et al., 2005
remaining studies were pseudo-randomiz
studies (Colpi et al., 2009; Ghalayini et al.
last two studies were randomized based o
list for the operative theatre. Label: back
Task
Scientific Information Recognition
<div style="text-align: center;">Scientific Information Recognition
</div>
Table 5: Examples of Demonstrations Selection by Corpus-level Method, where W is Kendall’s coefficient of

Table 5: Examples of Demonstrations Selection by Corpus-level Method, where W is Kendall’s coefficient of concordance.

Table 5: Examples of Demonstrations Selection by Corpus-level Method, whe concordance.

tent Classification
-
u-
nd
c-
’s
p-
of
e-
it
a-
n-
wn
in
ta
on
k-
es
d-
nn
ic
ls
el:
Sentence: Genotyping of the SNPs was performed
as described previously (Gutknecht et al.
2007;
Mo˘0308ssner et al. 2006a, b). Label: method
Sentence: 1991) and empirical studies (e.g. Kruuk
1978; Kruuk and Parish 1982; Mills 1989; Mills and
Gorman 1997; Geffen et al.
1992; Patterson and
Messier 2001; Valenzuela and Macdonald 2002). La-
bel: background
Sentence: Furthermore, the hospital anxiety and de-
pression scale (HAD) (Snaith and Zigmond, 1986) and
the IBS special scale for quality of life (QOL) (IB-
SQOL) (Drossman et al., 2000) were used. Label:
method
Sentence: (Massie and Holland, 1984; Ciaramella and
Poli, 2001; Uchitomi et al., 2003; Katz et al., 2004;
Ell et al., 2005; Boyd et al., 2012; Kim et al., 2012;
Mitchell et al., 2012; Palmer et al., 2012; Pirl et al.,
2012; Tada et al., 2012; Warmenhoven et al., 2012; Yu
et al., 2012). Label: background
Sentence:were retrospective (Okada et al., 2002; Tsu-
jimura et al., 2002; Ramasamy et al., 2005) and the two
remaining studies were pseudo-randomized controlled
studies (Colpi et al., 2009; Ghalayini et al., 2011); these
last two studies were randomized based on the waiting
list for the operative theatre. Label: background
