# CRITIQUELLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation
Pei Ke1,∗, Bosi Wen1,2,∗,†, Zhuoer Feng1,2,∗,†, Xiao Liu3,2,∗, Xuanyu Lei3,2,†, Jiale Cheng1,2,†, Shengyuan Wang3,2,†, Aohan Zeng3,2,†, Yuxiao Dong3, Hongning Wang1, Jie Tang3, Minlie Huang1,‡ The Conversational Artificial Intelligence (CoAI) Group, Tsinghua University
3The Knowledge Engineering Group (KEG), Tsinghua University kepei1106@outlook.com, {wbs23,fze22,liuxiao21}@mails.tsinghua.edu.cn, aihuang@tsinghua.edu.c
# Abstract
Since the natural language processing (NLP) community started to make large language models (LLMs) act as a critic to evaluate the quality of generated texts, most of the existing works train a critique generation model on the evaluation data labeled by GPT-4’s direct prompting. We observe that these models lack the ability to generate informative critiques in both pointwise grading and pairwise comparison especially without references. As a result, their generated critiques cannot provide fine-grained distinguishability on generated texts, causing unsatisfactory evaluation performance. In this paper, we propose a simple yet effective method called Eval-Instruct, which can first acquire pointwise grading critiques with pseudo references and then revise these critiques via multipath prompting to obtain informative evaluation data in different tasks and settings, including pointwise grading and pairwise comparison with / without references. After fine-tuning on these data, the resulting model CRITIQUELLM is empirically shown to outperform ChatGPT and all the open-source baselines and even achieve comparable evaluation performance to GPT-4 in system-level correlations of pointwise grading. We also demonstrate that our generated critiques can act as scalable feedback to further improve the generation quality of strong LLMs like ChatGPT1.
# 1 Introduction
Recently, large language models (LLMs) (OpenAI, 2022, 2023; Touvron et al., 2023a) have been improved rapidly and approached human-level performance on various natural language processing
∗Equal contribution †Work done when these authors interned at Zhipu AI. ‡Corresponding author 1The codes are available at https://github.com/ thu-coai/CritiqueLLM.
(NLP) tasks, such as question answering, text summarization, dialogue generation, and code generation (Laskar et al., 2023). How to automatically measure the performance of LLMs has now become an essential research problem and attracted extensive attention (Chang et al., 2023; Zhang et al., 2023; Liu et al., 2024). Strong evaluation methods are expected to provide high-quality critiques (including not only rating scores but also explanations) that act as scalable feedback and guide LLMs to improve persistently (Cui et al., 2023). Traditional evaluation metrics, usually based on n-gram overlap between generated texts and reference texts (such as BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004)), have limited effectiveness. Recent works mostly resort to model-based evaluation metrics, especially LLM-based ones (Wang et al., 2023a; Liu et al., 2023b; Zheng et al., 2023). Since most of the best-performing LLMs such as ChatGPT (OpenAI, 2022) and GPT-4 (OpenAI, 2023) can only be accessed via OpenAI APIs, researchers start to automatically collect evaluation data by directly prompting GPT-4 and train their own evaluation models, aiming to avoid potential risks of commerical APIs, such as high cost, unstable usage, and data leakage (Zheng et al., 2023; Wang et al., 2024; Li et al., 2024). However, we argue that these evaluation models are still struggling to generate informative critiques in different evaluation tasks including pointwise grading and pairwise comparison. Especially in the challenging reference-free setting, these models tend to generate general critiques without finegrained distinguishability on generated texts, causing unsatisfactory evaluation performance (Zheng et al., 2023). In this work, we propose a simple yet effective method called Eval-Instruct, which can automatically construct informative instruction-tuning data for different evaluation tasks and settings, including pointwise grading and pairwise comparison
with / without references. Our main idea is to fully utilize referenced pointwise grading critiques, which are shown to possess rich information with the assistance of references and elaborate prompt design (Zheng et al., 2023; Liu et al., 2023a), to construct evaluation data for other tasks and settings. Specifically, after acquiring pointwise grading critiques with pseudo references via GPT-4, we devise a multi-path prompting method including two strategies: 1) Pointwise-to-Pairwise Prompting aims to inject pointwise grading critiques into pairwise critiques and enrich them with more information about the respective quality of text pairs. 2) Referenced-to-Reference-Free Prompting is targeted at removing direct comparison with references in referenced critiques, while keeping other details to improve the specificity of reference-free critiques. The evaluation data in different tasks and settings can be acquired via different paths consisting of these two strategies. And we also design a cross validation mechanism to improve the data quality of reference-free pairwise comparison because both of the two paths reach this task. After fine-tuning on the data of all the tasks and settings, the resulting model CRITIQUELLM is empirically shown to outperform all the opensource baselines and even achieve comparable performance with GPT-4 in system-level correlations of pointwise grading. We also show the potential of CRITIQUELLM to act as effective feedback to enhance the performance of LLMs like ChatGPT. Our main contributions are as follows:
• We propose an evaluation data construction method called Eval-Instruct to automatically acquire informative evaluation data in both pointwise grading and pairwise comparison with / without references.
• We conduct extensive experiments on CRITIQUELLM, which is fine-tuned on the data constructed by Eval-Instruct. Experimental results on three instruction following benchmark datasets show that our model can outperform all the open-source baselines and even perform comparably with GPT-4 in system-level correlations of pointwise grading.
 We reveal the potential of CRITIQUELLM to guide LLMs to improve persistently by showing the positive impact of our generated critiques as scalable feedback on the generation quality of LLMs.
# 2 Related Work
Evaluation is a long-standing task in NLP, which becomes more challenging with the rapid development of LLMs (Celikyilmaz et al., 2020; Chang et al., 2023). Currently, there are mainly two lines of work on LLM evaluation, including NLU-style and NLG-style evaluations. NLU-style evaluation methods utilize natural language understanding (NLU) tasks such as multi-choice QA to measure the performance of LLMs via simple objective metrics (such as accuracy and F1 score) (Hendrycks et al., 2021; Zhong et al., 2023; Huang et al., 2023b), which may deviate from the common usage of LLMs and may not exactly reflect the ability of LLMs in generating responses for user queries. NLG-style evaluation methods extend metrics for natural language generation (NLG) tasks and expect to apply them to the measurement of LLM’s performance, which are the main focus of this paper. Compared with early metrics that depend on the n-gram overlap between generated texts and reference texts (Papineni et al., 2002; Banerjee and Lavie, 2005; Lin, 2004), recently proposed metrics based on state-of-the-art LLMs like GPT-4 (OpenAI, 2023) are shown to be strong evaluators due to the encouraging effectiveness of LLMs and the simplicity of formulating evaluation tasks as instruction-following tasks (Wang et al., 2023a; Chen et al., 2023; Liu et al., 2023b; Zheng et al., 2023; Ke et al., 2023; Fu et al., 2023). Since most of the state-of-the-art LLMs can only be accessed via APIs, researchers start to automatically collect evaluation data by directly prompting GPT-4 and train their own evaluation models to provide stable and effective evaluations at a lower cost (Wang et al., 2024; Li et al., 2024; Kim et al., 2024). The concurrent works similar to ours are the LLMs specially trained for evaluation tasks like PandaLM (Wang et al., 2024), JudgeLM (Zhu et al., 2023), and AUTO-J (Li et al., 2024). For comparison, our work is the first attempt to deal with the challenge of uninformative critique generation which commonly appears in recent LLMbased evaluation models especially without references. Instead of prompting GPT-4 directly, our proposed Eval-Instruct can fully utilize the connection among different evaluation tasks and settings to construct informative evaluation data, which are empirically shown to improve the quality of generated critiques.
# 3 Method
# 3.1 Task Definition and Method Overview
This paper mainly involves two typical evaluation tasks: 1) Pointwise Grading: Given a user query q, a LLM-generated text x, and a reference text r (omitted in the reference-free setting), the goal is to obtain a critique c including a rating score and an explanation to support this score. 2) Pairwise Comparison: Given a user query q, two LLMgenerated texts x1 and x2, and a reference text r (omitted in the reference-free setting), our purpose is to acquire a critique c including a comparison label (i.e., win / tie / lose) and an explanation to support this label. Our method consists of the following steps. We first construct an informative instruction-tuning dataset for different evaluation tasks and settings, including pointwise grading and pairwise comparison with / without references (§3.2). Specifically, after collecting user queries, LLM-generated texts, and pseudo references (§3.2.1), we can acquire high-quality referenced pointwise grading critiques via elaborately prompting GPT-4. Then, we devise a multi-path prompting method to construct informative evaluation data in other tasks and settings, which covers pointwise-to-pairwise and referenced-to-reference-free prompting strategies (§3.2.2). Since there are two paths to obtain reference-free pairwise comparison data, we design a cross validation mechanism to filter out the contradictory data and improve the quality (§3.2.3). Finally, we perform supervised fine-tuning on the automatically constructed evaluation data in a multitask manner to train a unified critique generation model for different evaluation tasks and settings (§3.3).
# 3.2 Evaluation-Oriented Instruction Data Construction (Eval-Instruct)
# 3.2 Evaluation-Oriented Instruction Data Construction (Eval-Instruct) 3.2.1 Pseudo Reference Collection
To construct instruction-tuning data for evaluation, it is imperative to first obtain the evaluation input, including user queries, LLM-generated texts, and references. We refer to recent works on instruction following (Liu et al., 2023a; Li et al., 2024; Zhang et al., 2024) and merge their task taxonomy to consider ten instruction following tasks covering diverse NLP applications in real-world scenarios2.
2Our task taxonomy contains fundamental language ability, advanced Chinese understanding, open-ended question answering, writing ability, logical reasoning, mathematics,
We utilize self-instruct (Wang et al., 2023d) to augment seed queries of these tasks which are publicly available and conduct strictly filtering to improve the data quality. The details are provided in Appendix A. Then, we collect LLM-generated texts from 10 representative models, which cover different levels of generation qualities, including GPT-4 (OpenAI, 2023), ChatGPT (OpenAI, 2022), two versions of ChatGLM (Du et al., 2022; Zeng et al., 2023), MOSS (Sun et al., 2023), Minimax3, Sparkdesk4, Chinese-Llama2-7B-Chat5, Baichuan2-13B-Chat (Yang et al., 2023), and Ernie Bot6. We further filter out the generated results by removing a small number of failure cases, such as empty responses. Finally, we select the best-performing LLM (i.e., GPT-4) and manually check its generated texts for each user query, while revising them if necessary to improve the quality. Thus, these generated texts after manual check and revise can act as pseudo references to assist the evaluation data construction.
# 3.2.2 Multi-Path Prompting
To acquire high-quality evaluation data in different evaluation tasks and settings, we first construct referenced pointwise grading critiques by prompting GPT-4 with the assistance of pseudo references and well-designed prompts like Liu et al. (2023a), which are empirically shown to be informative (Zheng et al., 2023). Then, regarding this setting as a beginning, we devise a multi-path prompting method to obtain evaluation data in other tasks and settings. As shown in Figure 1, there are two main prompting strategies: (1) Pointwise-to-Pairwise Prompting (fP2P ): This prompting strategy injects pointwise grading critiques of generated texts into pairwise comparison critiques, enriching them with information about the respective text quality. Meanwhile, it requires self-reflection on the pointwise critiques generated by GPT-4 before obtaining the final pairwise comparison results. (2) Referenced-to-Reference-Free Prompting (fR2RF ): This prompting strategy aims to remove direct comparison with references while keeping informative contents from references. It also re-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6ce2/6ce20c11-e5c1-46cf-af8f-921e889fdf9e.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/40ba/40baa1b3-51c4-4eb2-8879-2f418d509a1c.png" style="width: 50%;"></div>
Figure 1: Overview of Eval-Instruct. Starting from referenced pointwise grading data, our proposed multi-path prompting method can apply pointwise-to-pairwise and referenced-to-reference-free prompting strategies to acquire evaluation data in other tasks and settings via two different paths. Cross validation is adopted to filter out the contradictory data from these two paths and further improve the data quality.
quires GPT-4 to self-reflect7 whether the evaluation results including scores / labels and revised explanations are consistent, and modify the results if necessary. Equipped with the above prompting strategies, we have two paths to construct evaluation data in different tasks and settings. Assume that Dpoint,r = {(qi, ri, xi, cpoint,r i )}N i=1 indicates the referenced pointwise grading dataset constructed above and cpoint,r i represents the critique in the corresponding setting, our purpose is to acquire the datasets Dpair,r, Dpoint,rf, Dpair,rf via different paths, where point/pair means pointwise / pairwise evaluation and r/rf indicates referenced / reference-free evaluation, respectively. The two paths are devised as follows. Path#1: Dpoint,r fP 2P −−−→Dpair,r fR2RF −−−−→Dpair,rf As shown in Path#1 of Figure 1, we firstly conduct pointwise-to-pairwise prompting to acquire the referenced pairwise comparison dataset Dpair,r = {(qi, ri, xi,1, xi,2, cpair,r i )}M i=1:
(1)
where qi, ri, xi,1, xi,2 indicate the user query, the reference, and two generated texts of the i-th data,
where qi, ri, xi,1, xi,2 indicate the user query, the reference, and two generated texts of the i-th data, 7The purpose of self-reflection in the two strategies is to alleviate the inconsistency problem in the output critiques, reducing error propagation during the data construction process.
7The purpose of self-reflection in the two strategies is to alleviate the inconsistency problem in the output critiques, reducing error propagation during the data construction process.
respectively. cpoint,r i,1 , cpoint,r i,2 , cpair,r i are the referenced pointwise and pairwise evaluation results of xi,1, xi,2, respectively8. Then, we can apply referenced-to-reference-free prompting to obtain Dpair,rf = {(qi, xi,1, xi,2, cpair,rf i )}:
(2)
where cpair,rf,1 i means the reference-free pairwise comparison critique of the i-th data from Path#1. Path#2: Dpoint,r fR2RF −−−−→Dpoint,rf fP 2P →Dpair,rf Similarly, as shown in Path#2 of Figure 1, we can exchange the order of two prompting strategies applied to Dpoint,r accordingly. In this way, we can in turn acquire Dpoint,rf and Dpair,rf:
(3)
(4)
where cpair,rf,2 i denotes the reference-free pairwise comparison critique of the i-th data from Path#2.
8We conduct strictly rule-based filtering after each prompting step to remove low-quality data with errors in format and other aspects, which is omitted in this subsection.
# 3.2.3 Cross Validation
3.2.3 Cross Validation Since both of the two paths finally reach Dpair,rf, we design a cross validation mechanism to further improve the data quality. Specifically, Dpair,rf only contains the data whose comparison labels from two paths are consistent. In this case, the critiques from both of the two paths are added to Dpair,rf. The other data with contradictory comparison labels are strictly filtered. In our experiment, the proportion of the evaluation data which are filtered out is 7.7%, demonstrating that most of our constructed data from the two paths have consistent comparison labels, indicating acceptable data quality.
# 3.3 Supervised Fine-Tuning
We perform supervised fine-tuning on the LLM Pθ using all the constructed training data in a multitask manner to obtain CRITIQUELLM:
where M ′ indicates the data amount of Dpair,rf after cross validation. During fine-tuning, we follow Bai et al. (2022) to add simplified prompts to distinguish different parts of inputs. We also follow Li et al. (2024) to augment pairwise training data via swapping the order of two generated texts and exchanging the corresponding contents in critiques.
# 4 Experiment
# 4.1 Dataset
We adopt three benchmark datasets on open-ended instruction following, which involve various NLP tasks in LLM’s real-world scenarios9. The datasets also cover all the evaluation tasks and settings in this paper. The statistics are shown in Table 1. AlignBench (Liu et al., 2023a): This benchmark includes 8 categories of instruction following tasks
9We have conducted string matching to show that there is no overlap between the queries in the training and test sets.
Dataset
Task
Setting #Models #Samples / #Pairs Length
AlignBench
Pointwise R / R-F
8
3,200
274
Pairwise
R / R-F
8
1,600
293
AUTO-J (Eval-P)
Pairwise
R-F
6
1,392
372
LLMEval
Pairwise
R-F
11
1,530
283
Table 1: Statistics of the benchmark datasets, including the evaluation task / setting, the number of models / samples / pairs, and the average length of generated texts. R / R-F indicates referenced / reference-free evaluation, respectively.
and 8 LLMs for generation. It provides an evaluation dataset with human-annotated scores on the quality of generated texts. In addition to using human-annotated scores for measuring pointwise grading performance, we also follow the original paper to sample text pairs of the same query for pairwise comparison10, whose label is automatically determined by their pointwise scores. AUTO-J (Eval-P) (Li et al., 2024): This benchmark provides 1,392 pairwise comparison data, each of which contains a user query, two LLMgenerated texts, and a human-annotated preference label. These data involve 58 real-world scenarios and 6 model families for generation. LLMEval (Zhang et al., 2024): This benchmark designs 17 types of user queries covering representative NLP tasks in real-world scenarios, and provides ∼100,000 pairwise comparison data with human-annotated labels. Due to the limitation of computational resources and API costs for LLMbased evaluation methods, we randomly sample a subset (∼1,000) to measure the performance of our method and all the baselines for a fair comparison. As for the relationship between our training dataset in §3.2 and these benchmark datasets, our training dataset has similar task categories with AlignBench because our task taxonomy is built mainly based on AlignBench (Liu et al., 2023a) with other tasks in recent works (Li et al., 2024; Zhang et al., 2024) as supplementary, as described in §3.2.1. Also, our training dataset includes the training data of AUTO-J (Eval-P) (Li et al., 2024) while excluding its test set. Compared with AlignBench and AUTO-J (Eval-P), LLMEval (Zhang et al., 2024) does not have a similar task or data
10The authors in the original paper of AlignBench (Liu et al., 2023a) collect all the pairs of generated texts for each query (∼10,000 pairwise comparison data), causing high demand of computational resources and API costs for LLM-based evaluation methods. Thus, we randomly sample a subset (∼1,000 pairwise comparison data) to test our method and all the baselines for a fair comparison.
Level
Text-Level
System-Level
Setting
Referenced
Reference-Free
Referenced
Reference-Free
Metric
r
ρ
τ
r
ρ
τ
r
ρ
τ
r
ρ
τ
Closed-Source Evaluation Models
ChatGPT
0.443
0.421
0.379
0.292
0.287
0.266
0.955
0.976
0.929
0.778
0.833
0.643
GPT-4
0.629
0.583
0.532
0.523
0.494
0.447
0.995
1.000
1.000
0.997
0.976
0.929
Open-Source Evaluation Models
ChatGLM3-6B
0.223
0.222
0.207
0.159
0.150
0.140
0.790
0.833
0.643
0.544
0.548
0.429
Baichuan2-13B-Chat
0.199
0.200
0.187
0.125
0.117
0.110
0.854
0.929
0.786
0.663
0.527
0.400
Qwen-14B-Chat
0.373
0.379
0.358
0.255
0.254
0.239
0.901
0.929
0.786
0.772
0.833
0.643
Mixtral-8x7B
0.474
0.471
0.426
0.302
0.306
0.282
0.972
0.976
0.929
0.863
0.929
0.786
Llama-2-70B-Chat
0.152
0.162
0.109
0.123
0.122
0.113
0.663
0.667
0.500
0.547
0.429
0.286
JudgeLM-13B
0.450
0.430
0.391
0.170
0.162
0.155
0.984
0.976
0.929
0.717
0.905
0.786
AUTO-J-Bilingual-6B
-
-
-
0.044
0.045
0.041
-
-
-
0.558
0.571
0.500
CRITIQUELLM (Ours)
0.555
0.523
0.477
0.366
0.352
0.319
0.995
1.000
1.000
0.954
0.976
0.929
Table 2: Text-level and system-level Pearson (r), Spearman (ρ), and Kendall (τ) correlations in referenced and reference-free settings of pointwise grading on AlignBench. The highest correlation among the methods based on local models is bold, while the highest correlation overall is underlined. - means that AUTO-J-Bilingual-6B cannot support referenced pointwise grading.
distribution with our training dataset, which can act as a benchmark to test the generalization ability.
# 4.2 Baselines
We choose state-of-the-art general LLMs and evaluation-specific LLMs as our baselines. General LLMs: We adopt ChatGPT (gpt-3.5-turbo-1106) (OpenAI, 2022), GPT4 (gpt-4-1106-preview) (OpenAI, 2023), ChatGLM3-6B (Du et al., 2022; Zeng et al., 2023), Baichuan2-13B-Chat (Yang et al., 2023), Qwen14B-Chat (Bai et al., 2023), Llama-2-70B-Chat (Touvron et al., 2023b), and Mixtral-8x7B (Jiang et al., 2024) as our general baselines. These general LLMs can perform as an evaluator for pointwise grading and pairwise comparison via elaborate prompts without further training. We directly prompt these LLM to obtain evaluation results in single-turn interaction. Evaluation-Specific LLMs: We select AUTO-JBilingual-6B (Li et al., 2024) and JudgeLM-13B (Zhu et al., 2023) as our task-specific baselines. These two baselines are designed for specific evaluation tasks and settings.
# 4.3 Implementation Details
We choose ChatGLM3-6B (Du et al., 2022; Zeng et al., 2023) as our base model and use Zero Redundancy Optimizer (ZeRO) (Rajbhandari et al., 2020) stage 2 framework from the Deepspeed (Rasley et al., 2020) library. CRITIQUELLM is trained on 8 A800 GPUs. The number of training samples
for Dpoint,r/Dpoint,rf/Dpair,r/Dpair,rf is 12,102 / 12,095 / 6,190 / 5,428, respectively. We use AdamW (Kingma and Ba, 2015) optimizer with the weight decay of 0.1. The peak learning rate is 6e-5 with 10% warmup ratio. We set the maximum sequence length to 8,192 and the batch size to 64. The number of training epochs is 5. We use greedy decoding in the main result and investigate the effect of different decoding methods on our model in §4.7. For beam search, we set the beam size to 4. For the sampling-based decoding method, we adopt Nucleus Sampling (i.e., Top-p Sampling) (Holtzman et al., 2020) and set both the temperature and p to 0.9. For self-consistency decoding (Wang et al., 2023c), the number of candidate critiques is 5.
# 4.4 Main Results 4.4.1 Pointwise Grading
Following Colombo et al. (2022), we adopt textlevel and system-level Pearson (r), Spearman (ρ), and Kendall (τ) correlation coefficients between human judgments and automatic metrics to measure the pointwise grading performance. Text-level correlation is computed by the average score over the correlation coefficients between human judgments and automatic metrics for all the generated texts of each instruction. For comparison, system-level correlation is obtained by the correlation coefficients between human judgments and automatic metrics of each LLM’s score, which is the average value over all the scores of the corresponding model on
Dataset
AlignBench
AUTO-J (Eval-P)
LLMEval
Setting
Referenced
Reference-Free
Reference-Free
Reference-Free
Metric
Agr.
Cons.
Agr.
Cons.
Agr.
Cons.
Agr.
Cons.
Closed-Source Evaluation Models
ChatGPT
32.50
38.56
39.56
53.94
42.74
62.43
40.07
64.58
GPT-4
74.69
86.75
70.25
84.88
62.28
86.28
50.98
84.71
Open-Source Evaluation Models
ChatGLM3-6B
17.75
31.84
24.75
42.88
14.15
26.22
28.56
51.70
Baichuan2-13B-Chat
35.81
50.06
27.06
40.82
19.40
32.33
23.53
43.27
Qwen-14B-Chat
33.81
43.25
42.06
58.75
31.68
52.08
42.81
69.61
Mixtral-8x7B
61.69
74.06
53.88
72.25
35.20
52.66
48.04
79.02
Llama-2-70B-Chat
40.56
57.13
41.38
64.19
33.62
56.90
40.00
68.50
JudgeLM-13B
-
-
42.50
66.00
35.13
58.19
44.77
75.82
AUTO-J-Bilingual-6B
-
-
26.00
45.38
49.43
77.23
27.58
55.56
CRITIQUELLM (Ours)
70.56
89.25
58.81
83.06
50.93
82.76
50.72
85.95
Table 3: Agreement (Agr.) and consistency (Cons.) rates in pairwise comparison evaluation. The highest correlati among the methods based on local models is bold, while the highest correlation overall is underlined. - means t JudgeLM-13B and AUTO-J-Bilingual-6B cannot support referenced pairwise comparison.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2aa2/2aa26e80-be79-46ab-98b2-ead6c5f9b0fc.png" style="width: 50%;"></div>
Figure 2: Critique quality evaluation results. The percentages indicate the preference results between CRITIQUELLM and other models via GPT-4’s evaluation and human verification.
<div style="text-align: center;">Figure 2: Critique quality evaluation results. The percentages indicate the preference results between CRITIQUELLM and other models via GPT-4’s evaluation and human verification.</div>
# the dataset.
The results in Table 2 show that CRITIQUELLM can achieve comparable performance with GPT-4 especially in system-level correlations, while outperforming ChatGPT and all the open-source baselines. This indicates that our proposed method can successfully improve the quality of generated critiques. We can observe that system-level correlations of CRITIQUELLM are almost the same as those of GPT-4, which even approach 1,0. This demonstrate that our model is nearly able to distinguish the overall performance of all the eight LLMs.
# 4.4.2 Pairwise Comparison
Following Li et al. (2024), we adopt agreement and consistency rates to test the pairwise comparison
performance. Specifically, we conduct two comparisons for each data sample via swapping the order of two generated texts. We consider the model’s evaluation result to agree with humans only when the two comparison results are consistent and align with the human preference label. The results in Table 3 show that CRITIQUELLM can beat ChatGPT and all the open-source baselines in both agreement and consistency rates. Compared with GPT-4, CRITIQUELLM achieves comparable performance especially in the consistency rate. This indicates that CRITIQUELLM equipped with high-quality evaluation data in different tasks and settings not only performs well in pointwise grading, but also has a strong evaluation ability in pairwise comparison.
# 4.5 Analysis on Critique Quality
To further measure the quality of generated critiques, we follow Chen et al. (2024) to combine automatic and human evaluations. Specifically, we follow existing works (Wang et al., 2023b; Sun et al., 2024) to devise an evaluation prompt for GPT-4 to judge the quality of generated critiques. After GPT-4’s evaluation, we manually verify the results and modify them if necessary. We randomly select 100 evaluation data in the setting of pairwise comparison, which are from the mix of three datasets. And we collect generated critiques from CRITIQUELLM, state-of-the-art evaluators (i.e., ChatGPT and GPT-4), and an alternative model CRITIQUELLM (DP) whose training data in differ-
Critique Model
Overall
Logical
Open-ended QA
Professional
Fundamental
Mathematics
Role Play
Writing
Chinese Understanding
None
6.385
5.318
7.000
5.824
6.310
6.160
7.260
7.154
6.000
ChatGPT
6.300
5.045
6.762
6.353
6.276
5.760
7.000
6.885
6.063
GPT-4
6.545
4.455
7.190
6.588
6.897
6.200
7.111
7.077
6.563
CRITIQUELLM
6.530
5.136
7.381
6.765
6.414
6.000
7.407
7.192
5.315
Table 4: GPT-4’s referenced pointwise scores on AlignBench for original generated texts from ChatGPT (i.e., None and modified texts based on each critique generation model, respectively.
ent tasks and settings are acquired from GPT-4’s direct prompting. For each pair of critiques (one from CRITIQUELLM and the other from a baseline / an alternative model, given the same evaluation input), GPT-4 are required to label which critique is better (i.e. win, lose or tie) in terms of correctness, helpfulness, and informativeness. The priority of these three aspects is set to follow the above order. Then, human verification is conducted to check GPT-4’s evaluation on critiques. The results are shown in Figure 2. We can observe that CRITIQUELLM can achieve superior performance over ChatGPT and CritiqueLLM (DP), and even perform comparably with GPT-4. This demonstrates that our proposed evaluation data construction method can successfully improve the overall quality of generated critiques and enhance their informativeness.
# 4.6 Analysis of Critique as Feedback
To investigate whether the critiques generated by our model can serve as feedback to improve the quality of LLM-generated texts, we employ ChatGPT, GPT-4, and CRITIQUELLM to provide critiques for the generated texts of ChatGPT in the reference-free setting. Then, we instruct ChatGPT to modify its original generation based on the critiques. Finally, we use GPT-4 to perform referenced evaluations on the original texts and the modified texts generated by ChatGPT, respectively. The results in Table 4 show that the critiques from CRITIQUELLM can serve as positive feedback whose contributed improvement on the overall score is close to that from the GPT-4’s critiques. This further verifies the utility of CRITIQUELLM to provide informative critiques as scalable feedback that can guide LLMs towards better generation. We also notice that the critiques from ChatGPT itself have a negative impact on the overall quality of its generated texts. This phenomenon is consistent with recent works that doubt the selfcorrection ability of LLMs without external inputs (Huang et al., 2023a; Stechly et al., 2023;
Valmeekam et al., 2023). We also report the evaluation scores before and after the critique-based modification across different tasks in Table 4. It is notable that the critiques from CRITIQUELLM can help enhance the quality of generated texts in a majority of tasks. However, in the tasks of logical reasoning, mathematics, and advanced Chinese understanding which are mostly hard tasks involving reasoning, the critiques from CRITIQUELLM seem to degrade the performance. We manually checked error cases and found that our model obtained misleading critiques on the reasoning process of generated texts. Since the evaluation of reasoning chains remains a challenging task (Golovneva et al., 2023) even for GPT-4, we leave further investigation in these tasks as future work. Since our experiment is a preliminary step towards utilizing critiques as feedback, we additionally have some findings which may inspire future research. First, while incorporating human critiques can provide the comparison results between the generation performance assisted by the critiques from humans and LLMs, we notice that it is not trivial to collect high-quality critiques from human annotators for AlignBench especially in the reference-free setting. It is because AlignBench is designed to be difficult and covers a wide range of tasks (Liu et al., 2023a). Thus, how to collect highquality human critiques to improve the generation quality of LLMs is worth further exploring. Then, since we choose ChatGPT as the generation model, we find that stronger LLMs which can already generate high-quality responses struggle to be further improved via generated critiques. While weaker LLMs have a lot of room for improvement, they also have the weak ability to follow instructions. Thus, how to make weaker LLMs follow critiques to generate texts of a higher quality should be left as important future work.
# 4.7 Ablation Study
To further investigate the impact of each part on CRITIQUELLM, we conduct additional ablation
Setting
Pointwise
Pairwise
R
R-F
R
R-F
Metric
r
r
Agr.
Agr.
CRITIQUELLM
0.555
0.366
70.56
58.81
Fine-Tuning Data
w/o Cross Validation
0.566
0.361
66.13
57.44
Decoding Strategy
w/ Beam Search
0.554
0.374
70.31
57.75
w/ Sampling
0.547
0.353
68.69
57.31
w/ Self-Consistency
0.573
0.384
69.13
58.44
Explanation
w/o Explanation
0.509
0.332
60.19
51.56
Table 5: Text-level Pearson (r) correlations and agreement rates (Agr.) of ablation models in reference (R) and reference-free (R-F) settings of AlignBench.
studies. For fine-tuning data, we remove the cross validation module (§3.2.3) to explore its impact on the evaluation performance. Table 5 shows that the performance of CRITIQUELLM degrades especially in pairwise comparison, demonstrating that cross validation can filter out low-quality evaluation data and contribute to the final performance. As for decoding strategies, we show the evaluation performance of three decoding strategies in addition to greedy decoding in the main result, including beam search, Nucleus Sampling (Holtzman et al., 2020), and self-consistency decoding (Wang et al., 2023c). The results in Table 5 show that the self-consistency decoding method can enhance the performance of our model especially in pointwise grading. Meanwhile, greedy decoding performs best in pairwise comparison, while achieving comparable performance with other methods in pointwise grading at a smaller computational cost. For evaluation explanations, we remove the explanations in the critiques of training data. The results in Table 5 show that the performance of CRITIQUELLM largely degrades in both pointwise and pairwise evaluations without explanations. This verifies the positive impact of explanations on the final performance, which play a similar role to chain-of-thought reasoning (Wei et al., 2022).
# 5 Conclusion
We present an evaluation data construction method called Eval-Instruct, which can automatically construct informative evaluation data in both pointwise grading and pairwise comparison with / without references. After fine-tuning on the data from Eval-
Instruct, the resulting model CRITIQUELLM can beat ChatGPT and all the open-source baselines, and perform comparably with GPT-4 in systemlevel correlations of pointwise grading. CRITIQUELLM can also provide scalable feedback which can improve the generation quality of LLMs.
# Limitations
The limitations of our work are summarized as follows:
(1) In our method of multi-path prompting, we devise two prompting strategies to enrich the information in the resulting critiques, which can improve the critique quality. However, this method also increases the length of input prompts and lead to higher API costs when constructing evaluation data in different tasks and settings. We believe that it is not a severe problem because data acquisition is single-round and we do not repeatedly acquire critiques for the same evaluation input. Also, our proposed critique generation model based on opensource LLMs (i.e., ChatGLM3-6B) can achieve comparable performance with GPT-4 in some aspects, which may save the cost for LLM evaluation via APIs and avoid the risks such as unstable usage and data leakage. (2) Similar to other model-based evaluation methods, our evaluation model suffers from the selfevaluation bias (He et al., 2023) (also known as self-enhancement bias (Zheng et al., 2023)), which indicates the preference on the generated texts from the same base model. This bias is commonly recognized even in state-of-the-art LLM-based evaluators like GPT-4. We argue that researchers and developers can use multiple LLM-based evaluators with different base models including CRITIQUELLM to avoid self-evaluation bias towards specific generation models. Since there does not exist a satisfactory solution to the self-evaluation bias currently, we leave the further investigation as important future work.
# Acknowledgements
This work was supported by the NSFC projects (with No. 62306160) and the National Science Foundation for Distinguished Young Scholars (with No. 62125604). This work was also supported by China National Postdoctoral Program for Innovative Talents (No. BX20230194) and China Postdoctoral Science Foundation (No. 2023M731952). We would also like to thank Zhipu AI for sponsoring
the computation resources and annotation cost used in this work.
# References
Asli Celikyilmaz, Elizabeth Clark, and Jianfeng Gao. 2020. Evaluation of text generation: A survey. arXiv preprint arXiv:2006.14799.
Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Kaijie Zhu, Hao Chen, Linyi Yang, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2023. A survey on evaluation of large language models. arXiv preprint arXiv:2307.03109. Kai Chen, Chunwei Wang, Kuo Yang, Jianhua Han, Lanqing Hong, Fei Mi, Hang Xu, Zhengying Liu, Wenyong Huang, Zhenguo Li, Dit-Yan Yeung, Lifeng Shang, Xin Jiang, and Qun Liu. 2024. Gaining wisdom from setbacks: Aligning large language models via mistake analysis. In The Twelfth International Conference on Learning Representations. Yi Chen, Rui Wang, Haiyun Jiang, Shuming Shi, and Ruifeng Xu. 2023. Exploring the use of large language models for reference-free text quality evaluation: A preliminary empirical study. arXiv preprint arXiv:2304.00723. Pierre Jean A Colombo, Chloé Clavel, and Pablo Piantanida. 2022. Infolm: A new metric to evaluate summarization & data2text generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 10554–10562. Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. 2023. Ultrafeedback: Boosting language models with high-quality feedback. arXiv preprint arXiv:2310.01377. Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. 2022. Glm:
Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. 2022. Glm:
General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 320–335. Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. 2023. Gptscore: Evaluate as you desire. arXiv preprint arXiv:2302.04166. Olga Golovneva, Moya Peng Chen, Spencer Poff, Martin Corredor, Luke Zettlemoyer, Maryam FazelZarandi, and Asli Celikyilmaz. 2023. Roscoe: A suite of metrics for scoring step-by-step reasoning. In The Eleventh International Conference on Learning Representations. Tianxing He, Jingyu Zhang, Tianle Wang, Sachin Kumar, Kyunghyun Cho, James Glass, and Yulia Tsvetkov. 2023. On the blind spots of model-based evaluation metrics for text generation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12067–12097. Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations. Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. In 8th International Conference on Learning Representations. Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2023a. Large language models cannot self-correct reasoning yet. arXiv preprint arXiv:2310.01798. Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. 2023b. Ceval: A multi-level multi-discipline chinese evaluation suite for foundation models. arXiv preprint arXiv:2305.08322. Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088. Pei Ke, Fei Huang, Fei Mi, Yasheng Wang, Qun Liu, Xiaoyan Zhu, and Minlie Huang. 2023. DecompEval: Evaluating generated texts as unsupervised decomposed question answering. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9676–9691. Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, et al.
Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, et al.
capability in language models. In The Twelfth International Conference on Learning Representations. Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations. Md Tahmid Rahman Laskar, M Saiful Bari, Mizanur Rahman, Md Amran Hossen Bhuiyan, Shafiq Joty, and Jimmy Huang. 2023. A systematic study and comprehensive evaluation of ChatGPT on benchmark datasets. In Findings of the Association for Computational Linguistics: ACL 2023, pages 431–469. Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, and Pengfei Liu. 2024. Generative judge for evaluating alignment. In The Twelfth International Conference on Learning Representations. Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81. Xiao Liu, Xuanyu Lei, Shengyuan Wang, Yue Huang, Zhuoer Feng, Bosi Wen, Jiale Cheng, Pei Ke, Yifan Xu, Weng Lam Tam, et al. 2023a. Alignbench: Benchmarking chinese alignment of large language models. arXiv preprint arXiv:2311.18743. Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. 2024. Agentbench: Evaluating llms as agents. In The Twelfth International Conference on Learning Representations. Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023b. G-eval: NLG evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522.
# OpenAI. 2022. Introducing chatgpt.
OpenAI. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.
Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318.
Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. Zero: memory optimizations toward training trillion parameter models. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, page 20.
eff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3505–3506.
Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3505–3506.
Shichao Sun, Junlong Li, Weizhe Yuan, Ruifeng Yuan, Wenjie Li, and Pengfei Liu. 2024. The critique of critique. arXiv preprint arXiv:2401.04518.
Tianxiang Sun, Xiaotian Zhang, Zhengfu He, Peng Li, Qinyuan Cheng, Hang Yan, Xiangyang Liu, Yunfan Shao, Qiong Tang, Xingjian Zhao, Ke Chen, Yining Zheng, Zhejian Zhou, Ruixiao Li, Jun Zhan, Yunhua Zhou, Linyang Li, Xiaogui Yang, Lingling Wu, Zhangyue Yin, Xuanjing Huang, and Xipeng Qiu. 2023. Moss: Training conversational language models from synthetic data. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971. Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288. Karthik Valmeekam, Matthew Marquez, and Subbarao Kambhampati. 2023. Can large language models really improve by self-critiquing their own plans? arXiv preprint arXiv:2310.08118. Jiaan Wang, Yunlong Liang, Fandong Meng, Haoxiang Shi, Zhixu Li, Jinan Xu, Jianfeng Qu, and Jie Zhou. 2023a. Is chatgpt a good nlg evaluator? a preliminary study. arXiv preprint arXiv:2303.04048. Tianlu Wang, Ping Yu, Xiaoqing Ellen Tan, Sean O’Brien, Ramakanth Pasunuru, Jane Dwivedi-Yu, Olga Golovneva, Luke Zettlemoyer, Maryam FazelZarandi, and Asli Celikyilmaz. 2023b. Shepherd: A critic for language model generation. arXiv preprint arXiv:2308.04592. Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023c. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations. Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, Wei Ye, Shikun Zhang, and Yue Zhang. 2024. Pandalm: An automatic evaluation benchmark for LLM instruction tuning optimization. In The Twelfth International Conference on Learning Representations. Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh
Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, Wei Ye, Shikun Zhang, and Yue Zhang. 2024. Pandalm: An automatic evaluation benchmark for LLM instruction tuning optimization. In The Twelfth International Conference on Learning Representations.
Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh
Hajishirzi. 2023d. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pages 13484–13508. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, et al. 2023. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305. Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, and Jie Tang. 2023. GLM-130B: an open bilingual pre-trained model. In The Eleventh International Conference on Learning Representations. Yue Zhang, Ming Zhang, Haipeng Yuan, Shichun Liu, Yongyao Shi, Tao Gui, Qi Zhang, and Xuanjing Huang. 2024. Llmeval: A preliminary study on how to evaluate large language models. In The 38th Annual AAAI Conference on Artificial Intelligence. Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, and Minlie Huang. 2023. Safetybench: Evaluating the safety of large language models with multiple choice questions. arXiv preprint arXiv:2309.07045. Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track. Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. 2023. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364. Lianghui Zhu, Xinggang Wang, and Xinlong Wang. 2023. Judgelm: Fine-tuned large language models are scalable judges. arXiv preprint arXiv:2310.17631. A Query Augmentation and Scoring
Lianghui Zhu, Xinggang Wang, and Xinlong Wang. 2023. Judgelm: Fine-tuned large language models are scalable judges. arXiv preprint arXiv:2310.17631.
# A Query Augmentation and Scoring Prompts
We provide the prompt for query augmentation and scoring in Table 6. First, in the stage of generation, we give some in-context examples and devise
detailed requirements to help ChatGPT (OpenAI, 2022) generate augmented user queries and assign the category label to them. Then, during evaluation, we instruct ChatGPT to provide a difficulty score to each query for difficulty balance in the whole augmentation dataset.
# B Prompt Design for Eval-Instruct
We provide original prompts for pointwise-topairwise and referenced-to-reference-free strategies in Table 7 and Table 9, respectively. We also translate these prompts into English and show them in Table 8 and Table 10.
# C Case Study on Critique Generation
To intuitively show the effectiveness of our critique generation model, we provide two generated cases of pointwise and pairwise settings, respectively, in Table 11 and 13. We also translate these cases into English and show them in Table 12 and 14.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bb6a/bb6ae4b6-2d70-4982-a546-8545ded37aab.png" style="width: 50%;"></div>
Stage Prompt Generation You are asked to provide 10 diverse prompts. These task prompts will be provided to a GPT model and we will evaluate the ability of the GPT model to reply to these prompts. The following are some examples: 1.{example prompt 1} 2.{example prompt 2} 3.{example prompt 3} Here are the requirements you need to follow to provide prompts: 1. The prompts need to be complete sentences, not phrases or fragments. 2. The prompts need to be varied, do not use similar prompts. 3. the prompts need to be meaningful, do not use meaningless prompts. 4. The prompts need to have a variety of tones, e.g., combining interrogative and imperative sentences. 5. The prompts need to be challenging, do not use simple directions. 6. The prompts need to be something that the Large Language Model can accomplish. For example, don’t ask the assistant to create any visual or audio output. For example, don’t ask the assistant to wake you up at 5pm or set a reminder because it can’t perform any action. For example, prompts should not be related to audio, video, images, hyperlinks. 7. The prompts are in Simplified Chinese, except for translation-related questions or math-related questions. 8. Some prompts can provide contextual information, should involve realistic data, and should not contain simple placeholders. Not all prompts require input. For example, when an prompts asks for general knowledge information, such as "What is the tallest mountain in the world?", it does not need to provide specific context. After you have provided the prompts, please add the category of the prompts in a pair of && sign after the prompt and surround the prompt with in a pair of @@ sign. For example, if the prompt is "@@What is the tallest mountain in the world?@@&& 基本任务&&", then the category is 基本任务. The category must be one of the following 10 categories. 1. 基本任务2. 中文理解3. 综合问答4. 文本写作5. 数学计算6. 逻辑推理7. 角色扮演8. 专业 能力9. 代码生成10. 多语言能力 Here are some examples of prompts you provide: @@example prompt1@@ &&category1&& @@example prompt2@@ &&category2&& · · · @@example prompt9@@ &&category9&& @@example prompt10@@ &&category10&&
<div style="text-align: center;">The following is a list of 10 good task prompts with serial numbers and categories:</div>
Evaluation
已知上面三个问题和它们的类别，现在请你根据以下要求，对这三个问题的题目的难度在1-3分的量表上分别评分:
(1) 1分：对于大语言模型来说，这类问题是容易的
(2) 2分：对于大语言模型来说，这类问题是中等难度的
(3) 3分：对于大语言模型来说，这类问题是困难的
最后：请将这三个问题，题目用一对@@符号包围，对应的类别用一对&&符号包围，分数用一对##包围，分别带有序号地输出出来：
例如：如果问题1的题目是题目1，类别是综合问答类别，分数是1分，问题2的题目是题目2，类别是基本任务类别，分数是2分，问题3的题目是题目3，
类别是文本写作类别，分数是3分，那么输出如下：
1.@@题目1@@&&综合问答&&##1##
2.@@题目2@@&&基本任务&&##2##
3.@@题目3@@&&文本写作&&##3##
下面是按照上述要求生成的示例：
Evaluation
(English)
Given the above three questions and their categories, please rate the difficulty of each question on a scale of 1-3 based on the following requirements:
(1) Score 1: For large language models, this type of question is easy.
(2) Score 2: For large language models, this type of question is of medium difficulty.
(3) Score 3: For large language models, this type of question is difficult.
Finally, please output the three questions with their titles enclosed in a pair of @@ symbols, the corresponding categories enclosed in a pair of && symbols, and the
scores enclosed in a pair of ## symbols, each with an serial number.
For example, if question 1 is titled “Title 1”, the category is “Open-ended Questions”, and the score is 1, question 2 is titled “Title 2”, the category is “Fundamental
Language Ability”, and the score is 2 points, question 3 is titled “Title 3”, the category is “Writing Ability”, and the score is 3 points, then the output is as follows:
1.@@Title 1@@&&Open-ended Questions&&##1##
2.@@Title 2@@&&Fundamental Language Ability&&##2##
3.@@Title 3@@&&Writing Ability&&##3##
The following parts are generated examples based on the above requirements:
Table 6: Prompts for instructing ChatGPT to generate, categorize and evaluate user queries. Examples and corresponding categories are randomly sampled from the set of seed queries.
Table 6: Prompts for instructing ChatGPT to generate, categorize and evaluate user queries. Examples and corresponding categories are randomly sampled from the set of seed queries.
d to d n 你是一个擅长评价文本质量的助手。请你以公正的评判者的身份，比较两个AI助手对于用户提问的回答的质量优劣。我们会给你提供用户的提 问，高质量的参考答案，需要你比较的两个AI助手的答案，以及两个答案各自的质量评价分析。当你开始你的评估时，你需要遵守以下的流程： 1. 结合参考答案、两个AI助手的答案以及其质量评价分析，根据上述指定的维度对他们的答案进行细致的比较，给出详细的比较分析文本。比较 分析文本要求覆盖两个答案的质量评价分析中可用于比较的所有重要细节，并包含对答案中具体内容的分析。 2. 结合参考答案和每个维度的比较分析，从两个AI助手的答案中选出综合质量更高的那个，或者判定他们质量相当，并给出详尽的选择理由。你 的比较需要尽可能严谨细致，不受两个AI助手答案先后顺序的影响。 质量评价分析中的各维度分数和综合得分仅供参考，在各维度和综合的比较分析文本中不能直接提及各维度分数和综合得分。针对综合得分差距 较大的样本对，应尽可能按照分数高低得出比较结果，除非发现质量评价分析中存在明显错误。而针对综合得分差距较小的样本对，则允许比较 结果和分数高低不一致，但仍需要详细说明比较评价的理由。 请记住，你必须首先按照给定的评价维度，输出相应维度的名称和比较分析的文本。然后再给出综合质量比较结果，并给出比较结果的分析和解 释。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你的综合质量选择结果，即你选择的综合质量更高的那个AI助手（或者认为质 量相当），并确保你返回的结果和上述生成文本中的结果保持一致： {{’综合比较结果’: 回答综合质量更高的助手序号或质量相当}}，例如：{{’综合比较结果’: ’助手1’}}或{{’综合比较结果’: ’助手2’}}或{{’综合比较 结果’: ’质量相当’}}。
<div style="text-align: center;">[助手2的答案质量评价分析开始] {Referenced Pointwise Grading Critique for Generated Text 2} [助手2的答案质量评价分析结束]</div>
Referenceree Pointise Grading o Referenceree Pairwise Comparison 你是一个擅长评价文本质量的助手。请你以公正的评判者的身份，比较两个AI助手对于用户提问的回答的质量优劣。我们会给你提供用户的提 问，需要你比较的两个AI助手的答案，以及两个答案各自的质量评价分析。当你开始你的评估时，你需要遵守以下的流程： 1. 结合两个AI助手的答案以及其质量评价分析，根据上述指定的维度对他们的答案进行细致的比较，给出详细的比较分析文本。比较分析文本要 求覆盖两个答案的质量评价分析中可用于比较的所有重要细节，并包含对答案中具体内容的分析。 2. 结合每个维度的比较分析，从两个AI助手的答案中选出综合质量更高的那个，或者判定他们质量相当，并给出详尽的选择理由。你的比较需要 尽可能严谨细致，不受两个AI助手答案先后顺序的影响。 质量评价分析中的各维度分数和综合得分仅供参考，在各维度和综合的比较分析文本中不能直接提及各维度分数和综合得分。针对综合得分差距 较大的样本对，应尽可能按照分数高低得出比较结果，除非发现质量评价分析中存在明显错误。而针对综合得分差距较小的样本对，则允许比较 结果和分数高低不一致，但仍需要详细说明比较评价的理由。 请记住，你必须首先按照给定的评价维度，输出相应维度的名称和比较分析的文本。然后再给出综合质量比较结果，并给出比较结果的分析和解 释。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你的综合质量选择结果，即你选择的综合质量更高的那个AI助手（或者认为质 量相当），并确保你返回的结果和上述生成文本中的结果保持一致： {{’综合比较结果’: 回答综合质量更高的助手序号或质量相当}}，例如：{{’综合比较结果’: ’助手1’}}或{{’综合比较结果’: ’助手2’}}或{{’综合比较 结果’: ’质量相当’}}。
用户的提问：{Question}
[助手1的答案开始]
{Generated Text 1}
[助手1的答案结束]
[助手1的答案质量评价分析开始]
{Reference-Free Pointwise Grading Critique for Generated Text 1}
[助手1的答案质量评价分析结束]
[助手2的答案开始]
{Generated Text 2}
[助手2的答案结束]
[助手2的答案质量评价分析开始]
{Reference-Free Pointwise Grading Critique for Generated Text 2}
[助手2的答案质量评价分析结束]
able 7: Pointwise-to-Pairwise prompt design in multi-path prompting.
Referenced
Pointwise
Grading
to
Referenced
Pairwise
Comparison
You are an expert at text quality evaluation. Please act as a fair judge, and compare the quality between two AI assistants’ answers to a user query. We will provide you with a user query, a high-quality reference answer, two AI assistants’ responses to the query, and the corresponding critiques to the two responses, respectively. When you start your evaluation, you need to follow the procedures below: 1. Considering the reference answers, along with two AI assistants’ answers and the corresponding critiques to them, conduct detailed comparison between two AI assistants’ answers based on the evaluation dimensions Dimension. Provide a detailed comparison result. The comparison result should cover all the important details from the pointwise critiques that can be used for the comparison, and it should include an analysis of the specific content in the answers. 2. Based on the reference answer and the comparison result of each dimension, choose the answer from the two AI assistants that has the higher overall quality, or judge that their qualities are equivalent. Provide a detailed rationale for your choice. Your comparison needs to be as rigorous and detailed as possible, and not be affected by the order in which the two AI assistants’ answers were given. The scores of each dimension and the overall score in the pointwise critique are for reference only, neither of which can be directly referred to in the comparison result. For the text pairs with a large difference in overall scores, the comparison result should be determined largely according to the scores, unless there are obvious errors in the pointwise critique. For text pairs with a small difference in overall scores, the comparison result is allowed to be inconsistent with the score ranking, but the reason for the comparison result needs to be detailed. Please remember that you must first output the names and comparison results of each given evaluation dimensions, respectively. Then, give the comparison result of overall quality and provide an analysis and explanation of the comparison result. Afterwards, at the end of your answer, return your choice of overall quality result in the following dictionary format (including brackets), that is, the AI assistant you chose as having higher overall quality (or considered to have equivalent quality), and be sure that the result you return is consistent with the result in the generated text above. {{’Overall Comparison Result’: the assistant number with higher overall quality or tie}}，for example: {{’Overall Comparison Result’: ’Assistant 1’}} or {{’Overall Comparison Result’: ’Assistant 2’}} or {{’Overall Comparison Result’: ’Tie’}}。
You are an expert at text quality evaluation. Please act as a fair judge, and compare the quality between two AI assistants’ answers to a user query. We will provide you with a user query, a high-quality reference answer, two AI assistants’ responses to the query, and the corresponding critiques to the two responses, respectively. When you start your evaluation, you need to follow the procedures below: 1. Considering the reference answers, along with two AI assistants’ answers and the corresponding critiques to them, conduct detailed comparison between two AI assistants’ answers based on the evaluation dimensions Dimension. Provide a detailed comparison result. The comparison result should cover all the important details from the pointwise critiques that can be used for the comparison, and it should include an analysis of the specific content in the answers. 2. Based on the reference answer and the comparison result of each dimension, choose the answer from the two AI assistants that has the higher overall quality, or judge that their qualities are equivalent. Provide a detailed rationale for your choice. Your comparison needs to be as rigorous and detailed as possible, and not be affected by the order in which the two AI assistants’ answers were given. The scores of each dimension and the overall score in the pointwise critique are for reference only, neither of which can be directly referred to in the comparison result. For the text pairs with a large difference in overall scores, the comparison result should be determined largely according to the scores, unless there are obvious errors in the pointwise critique. For text pairs with a small difference in overall scores, the comparison result is allowed to be inconsistent with the score ranking, but the reason for the comparison result needs to be detailed. Please remember that you must first output the names and comparison results of each given evaluation dimensions, respectively. Then, give the comparison result of overall quality and provide an analysis and explanation of the comparison result. Afterwards, at the end of your answer, return your choice of overall quality result in the following dictionary format (including brackets), that is, the AI assistant you chose as having higher overall quality (or considered to have equivalent quality), and be sure that the result you return is consistent with the result in the generated text above. {{’Overall Comparison Result’: the assistant number with higher overall quality or tie}}，for example: {{’Overall Comparison Result’: ’Assistant 1’}} or {{’Overall Comparison Result’: ’Assistant 2’}} or {{’Overall Comparison Result’: ’Tie’}}。
[Reference Answer Begin] {Reference} [Reference Answer End]
[Assistant 1’s Answer Begin] {Generated Text 1} [Assistant 1’s Answer End]
[Critique for Assistant 2’s Answer Begin] {Referenced Pointwise Grading Critique for Generated Text 2} [Critique for Assistant 2’s Answer End]
You are an expert at text quality evaluation. Please act as a fair judge, and compare the quality between two AI assistants’ answers to a user query. We will provide you with a user query, two AI assistants’ responses to the query, and the corresponding critiques to the two responses, respectively. When you start your evaluation, you need to follow the procedures below: 1. Considering two AI assistants’ answers and the corresponding critiques to them, conduct detailed comparison between two AI assistants’ answers based on the evaluation dimensions Dimension. Provide a detailed comparison result. The comparison result should cover all the important details from the pointwise critiques that can be used for the comparison, and it should include an analysis of the specific content in the answers. 2. Based on the comparison result of each dimension, choose the answer from the two AI assistants that has the higher overall quality, or judge that their qualities are equivalent. Provide a detailed rationale for your choice. Your comparison needs to be as rigorous and detailed as possible, and not be affected by the order in which the two AI assistants’ answers were given. The scores of each dimension and the overall score in the pointwise critique are for reference only, neither of which can be directly referred to in the comparison result. For the text pairs with a large difference in overall scores, the comparison result should be determined largely according to the scores, unless there are obvious errors in the pointwise critique. For text pairs with a small difference in overall scores, the comparison result is allowed to be inconsistent with the score ranking, but the reason for the comparison result needs to be detailed. Please remember that you must first output the names and comparison results of each given evaluation dimensions, respectively. Then, give the comparison result of overall quality and provide an analysis and explanation of the comparison result. Afterwards, at the end of your answer, return your choice of overall quality result in the following dictionary format (including brackets), that is, the AI assistant you chose as having higher overall quality (or considered to have equivalent quality), and be sure that the result you return is consistent with the result in the generated text above. {{’Overall Comparison Result’: the assistant number with higher overall quality or tie}}, for example: {{’Overall Comparison Result’: ’Assistant 1’}} or {{’Overall Comparison Result’: ’Assistant 2’}} or {{’Overall Comparison Result’: ’Tie’}}.
You are an expert at text quality evaluation. Please act as a fair judge, and compare the quality between two AI assistants’ answers to a user query. We will provide you with a user query, two AI assistants’ responses to the query, and the corresponding critiques to the two responses, respectively. When you start your evaluation, you need to follow the procedures below: 1. Considering two AI assistants’ answers and the corresponding critiques to them, conduct detailed comparison between two AI assistants’ answers based on the evaluation dimensions Dimension. Provide a detailed comparison result. The comparison result should cover all the important details from the pointwise critiques that can be used for the comparison, and it should include an analysis of the specific content in the answers. 2. Based on the comparison result of each dimension, choose the answer from the two AI assistants that has the higher overall quality, or judge that their qualities are equivalent. Provide a detailed rationale for your choice. Your comparison needs to be as rigorous and detailed as possible, and not be affected by the order in which the two AI assistants’ answers were given. The scores of each dimension and the overall score in the pointwise critique are for reference only, neither of which can be directly referred to in the comparison result. For the text pairs with a large difference in overall scores, the comparison result should be determined largely according to the scores, unless there are obvious errors in the pointwise critique. For text pairs with a small difference in overall scores, the comparison result is allowed to be inconsistent with the score ranking, but the reason for the comparison result needs to be detailed. Please remember that you must first output the names and comparison results of each given evaluation dimensions, respectively. Then, give the comparison result of overall quality and provide an analysis and explanation of the comparison result. Afterwards, at the end of your answer, return your choice of overall quality result in the following dictionary format (including brackets), that is, the AI assistant you chose as having higher overall quality (or considered to have equivalent quality), and be sure that the result you return is consistent with the result in the generated text above. {{’Overall Comparison Result’: the assistant number with higher overall quality or tie}}, for example: {{’Overall Comparison Result’: ’Assistant 1’}} or {{’Overall Comparison Result’: ’Assistant 2’}} or {{’Overall Comparison Result’: ’Tie’}}.
[Assistant 1’s Answer Begin] {Generated Text 1} [Assistant 1’s Answer End]
Table 8: Pointwise-to-Pairwise prompt design in multi-path prompting (translated into English)
Setting
Prompt
Referenced
Pointwise
Grading
to
Reference-
Free
Point-
wise Grading
你是一个擅长评价文本质量的助手。请你根据以下要求修改评价文本。
1. 在修改后的评价文本中，不要直接提及参考答案。可以在评价文本中适当利用参考答案中的具体内容辅助分析，但不要让读者感受到参考答案
的存在。修改后的评价文本需要语言上通顺，逻辑上合理，分析内容与比较结果呼应。
2. 在修改各个维度的分析时，分析的内容需要和当前评价文本基本保持一致，但不要直接提及参考答案。
3. 在修改综合得分的分析文本时，不要直接提及参考答案，尽量保留当前评价文本中的其他细节，并充分利用修改后的分维度分析。修改后的综
合分析文本应通顺、流畅、自洽，通常情况下应与综合得分保持一致。如果发现当前综合分析文本中存在重要错误，应修改相应的分析文本。仅
当该错误严重影响到综合得分时，才慎重修改综合得分。
4. 修改后所有输出格式需要和当前评价文本严格保持一致。在你回答的末尾，仍需要按照以下字典格式（包括括号）返回你的综合质量得分，并
确保你返回的结果和上述生成文本中的结果保持一致：
{{’综合得分’: 回答的综合质量得分}}，例如：{{’综合得分’: ’5’}}。
用户的提问：{Question}
[参考答案开始]
{Reference}
[参考答案结束]
[助手的答案开始]
{Generated Text}
[助手的答案结束]
[评价文本开始]
{Referenced Pointwise Grading Critique for Generated Text}
[评价文本结束]
Referenced
Pairwise
Comparison
to Reference-
Free Pairwise
Comparison
你是一个擅长评价文本质量的助手。请你根据以下要求修改比较式评价文本。
1. 在修改后的评价文本中，不要直接提及参考答案。可以在评价文本中适当利用参考答案中的具体内容辅助分析，但不要让读者感受到参考答案
的存在。修改后的评价文本需要语言上通顺，逻辑上合理，分析内容与比较结果呼应。
2. 在修改各个维度的比较分析时，分析的内容需要和当前评价文本基本保持一致，但不要直接提及参考答案。
3. 在修改综合比较结果的分析文本时，不要直接提及参考答案，尽量保留当前评价文本中的其他细节，并充分利用修改后的分维度分析。修改后
的综合分析文本应通顺、流畅、自洽，通常情况下应与综合比较结果保持一致。如果发现当前综合分析文本中存在重要错误，应修改相应的分析
文本。仅当该错误严重影响到综合比较结果时，才慎重修改综合比较结果。
4. 修改后所有输出格式需要和当前评价文本严格保持一致。在你回答的末尾，仍需要按照以下字典格式（包括括号）返回你的综合质量选择结
果，即你选择的综合质量更高的那个AI助手（或者认为质量相当），并确保你返回的结果和上述生成文本中的结果保持一致：
{{’综合比较结果’: 回答综合质量更高的助手序号或质量相当}}，例如：{{’综合比较结果’: ’助手1’}}或{{’综合比较结果’: ’助手2’}}或{{’综合比较
结果’: ’质量相当’}}。
用户的提问：{Question}
[参考答案开始]
{Reference}
[参考答案结束]
[助手1的答案开始]
{Generated Text 1}
[助手1的答案结束]
[助手2的答案开始]
{Generated Text 2}
[助手2的答案结束]
[评价文本开始]
{Referenced Pairwise Comparison Critique for Generated Text 1&2}
[评价文本结束]
Table 9: Referenced-to-Reference-Free prompt design in multi-path prompting.
Setting
Prompt
Referenced
Pointwise
Grading
to
Reference-
Free
Point-
wise Grading
You are an expert at text quality evaluation. Please revise the critique following the instructions below:
1. In the revised critique, do not directly refer to the reference answer. You can use the specific content in the reference answer to assist your analysis in the
critique, but do not make the readers feel the presence of the reference answer. The revised critique should be fluent, logically reasonable. The explanation
should be consistent with the score.
2. When revising the explanation of each dimension, the content should basically be consistent with the corresponding score. But do not directly mention the
reference answer.
3. When revising the explanation of the final score, do not directly mention the reference answer. Try to retain other details in the current critique and fully utilize
the modified critique of each dimension. The revised explanation of the final score should be smooth, fluent, and self-consistent, and it should commonly be
consistent with the final score. If an important error is found in the current critique, the error in the critique should be revised. Only when this error severely
affects the final score, you may carefully revise the final score.
4. The output format of all the revised results needs to strictly adhere to the current critique. At the end of your output, you still need to return your overall
quality score in the following dictionary format (including brackets), and ensure that the result you return is consistent with the result in the above generated text.
{{’Overall Score’: Score for Overall Quality}}，for instance: {{’Overall Score’: ’5’}}。
The user’s query: {Question}
[Reference Answer Begin]
{Reference}
[Reference Answer End]
[AI Assistant’s Answer Begin]
{Generated Text}
[AI Assistant’s Answer End]
[Critique Begin]
{Referenced Pointwise Grading Critique for Generated Text}
[Critique End]
Referenced
Pairwise
Comparison
to Reference-
Free Pairwise
Comparison
You are an expert at text quality evaluation. Please revise the critique following the instructions below:
1. In the revised critique, do not directly refer to the reference answer. You can use the specific content in the reference answer to assist your analysis in the
critique, but do not make the readers feel the presence of the reference answer. The revised critique should be fluent, logically reasonable. The explanation
should be consistent with the comparison result.
2. When revising the explanation of each dimension, the content should basically be consistent with the current critiques. But do not directly mention the
reference answer.
3. When revising the explanation of the overall comparison result, do not directly mention the reference answer. Try to retain other details in the current critique
and fully utilize the modified critique of each dimension. The revised explanation of the overall comparison result should be smooth, fluent, and self-consistent,
and it should commonly be consistent with the overall comparison result. If an important error is found in the current critique, the error in the critique should be
revised. Only when this error severely affects the overall comparison result, you may carefully revise the overall comparison result.
4. The output format of all the revised results needs to strictly adhere to the current critique. At the end of your output, you still need to return your overall
comparison result in the following dictionary format (including brackets), and ensure that the result you return is consistent with the result in the above generated
text.
{{’Overall Comparison Result’: the assistant number with higher overall quality or tie }}, for example: {{’Overall Comparison Result’: ’Assistant 1’}} or
{{’Overall Comparison Result’: ’Assistant 2’}} or {{’Overall Comparison Result’: ’Tie’}}.
The user’s query: {Question}
[Reference Answer Begin]
{Reference}
[Reference Answer End]
[Assistant 1’s Answer Begin]
{Generated Text 1}
[Assistant 1’s Answer End]
[Assistant 2’s Answer Begin]
{Generated Text 2}
[Assistant 2’s Answer End]
[Critique Begin]
{Referenced Pairwise Comparison Critique for Generated Text 1&2 }
[Critique End]
Table 10: Referenced-to-Reference-Free prompt design in multi-path prompting (translated into English).
Referenced Pairwise Comparison to ReferenceFree Pairwise Comparison You are an expert at text quality evaluation. Please revise the critique following the instructions below: 1. In the revised critique, do not directly refer to the reference answer. You can use the specific content in the reference answer to assist your analysis in the critique, but do not make the readers feel the presence of the reference answer. The revised critique should be fluent, logically reasonable. The explanation should be consistent with the comparison result. 2. When revising the explanation of each dimension, the content should basically be consistent with the current critiques. But do not directly mention the reference answer. 3. When revising the explanation of the overall comparison result, do not directly mention the reference answer. Try to retain other details in the current critique and fully utilize the modified critique of each dimension. The revised explanation of the overall comparison result should be smooth, fluent, and self-consistent, and it should commonly be consistent with the overall comparison result. If an important error is found in the current critique, the error in the critique should be revised. Only when this error severely affects the overall comparison result, you may carefully revise the overall comparison result. 4. The output format of all the revised results needs to strictly adhere to the current critique. At the end of your output, you still need to return your overall comparison result in the following dictionary format (including brackets), and ensure that the result you return is consistent with the result in the above generated text. {{’Overall Comparison Result’: the assistant number with higher overall quality or tie }}, for example: {{’Overall Comparison Result’: ’Assistant 1’}} or {{’Overall Comparison Result’: ’Assistant 2’}} or {{’Overall Comparison Result’: ’Tie’}}. The user’s query: {Question} [Reference Answer Begin] {Reference} [Reference Answer End] [Assistant 1’s Answer Begin] {Generated Text 1} [Assistant 1’s Answer End] [Assistant 2’s Answer Begin] {Generated Text 2} [Assistant 2’s Answer End] [Critique Begin] {Referenced Pairwise Comparison Critique for Generated Text 1&2 } [Critique End]
Human Score
(1-5)
3
ChatGPT
(1-10)
评估结果如下：
1. 事实正确性: 9. 这段回答并没有涉及具体的事实数据，而是表达了一种情感和心境，因此无法对事实正确性进行评估。
2. 满足用户需求: 8. 回答全面而恰当地表达了孤独寂寞的心境，对用户提出的问题进行了全面回应。
3. 逻辑连贯性: 9. 回答在整体上保持了逻辑连贯性，情感描写流畅自然，没有出现自相矛盾的情况。
4. 创造性: 9. 回答具有创新性和独特性，提供了对孤独寂寞心境的深刻见解和情感描写。
5. 丰富度: 9. 回答包含