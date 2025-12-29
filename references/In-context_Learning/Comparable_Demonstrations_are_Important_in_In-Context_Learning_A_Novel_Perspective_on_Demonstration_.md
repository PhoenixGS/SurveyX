# COMPARABLE DEMONSTRATIONS ARE IMPORTANT IN IN-CONTEXT LEARNING: A NOVEL PERSPECTIVE ON DEMONSTRATION SELECTION
Caoyun Fan Jidong Tian Yitian Li Hao He∗ Yaohui Jin∗
MoE Key Lab of Artificial Intelligence, AI Institute, Shanghai Jiao Tong University, Shanghai, China
# ABSTRACT
9 Jan 2024
In-Context Learning (ICL) is an important paradigm for adapting Large Language Models (LLMs) to downstream tasks through a few demonstrations. Despite the great success of ICL, the limitation of the demonstration number may lead to demonstration bias, i.e. the input-label mapping induced by LLMs misunderstands the task’s essence. Inspired by human experience, we attempt to mitigate such bias through the perspective of the inter-demonstration relationship. Specifically, we construct Comparable Demonstrations (CDs) by minimally editing the texts to flip the corresponding labels, in order to highlight the task’s essence and eliminate potential spurious correlations through the inter-demonstration comparison. Through a series of experiments on CDs, we find that (1) demonstration bias does exist in LLMs, and CDs can significantly reduce such bias; (2) CDs exhibit good performance in ICL, especially in out-of-distribution scenarios. In summary, this study explores the ICL mechanisms from a novel perspective, providing a deeper insight into the demonstration selection strategy for ICL.
arXiv:2312.07476v2
# Index Terms— In-Context Learning, Demonstration Selection, Large Language Models
# 1. INTRODUCTION
Large Language Models (LLMs) [1] display a strong ability to perform In-Context Learning (ICL) [2], i.e. mastering natural language tasks from a small number of in-context demonstrations without any parameter updates [3, 4]. This flexible and efficient paradigm [5] gives LLMs the potential to become general-purpose models [6, 7], i.e. capable of generalizing to most tasks without further fine-tuning [8]. Despite the success of ICL in many NLP scenarios, there remains little understanding of how ICL works [6, 9]. As shown in Fig. 1(a), some previous studies attempted to explore the ICL mechanisms from various perspectives: [6] considered input-label format to be important for ICL; [9, 10]
∗Corresponding author. This work was supported by the Shanghai Municipal Science and Technology Major Project (2021SHZDZX0102), and the Fundamental Research Funds for the Central Universities.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f626/f626a03a-1dc3-4870-b7b7-43f67f71b248.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Some perspectives on exploring the ICL mechanisms.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/946a/946ade30-6a9a-4cd3-9c11-38c6a1f96894.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Analysis of inter-demonstration relationship. Comparable Demonstrations can reduce LLMs from misunderstanding the task’s essence.</div>
<div style="text-align: center;">Fig. 1. Overview of Comparable Demonstrations in ICL.</div>
found that the label space is one of the key drivers in ICL performance; [3, 11] suggested that the demonstration distribution (based on semantic similarity) can affect the information obtained by LLMs in ICL. However, as a potential perspective, the effect of the inter-demonstration relationship in ICL is not widely discussed. In this study, we attempt to explore the ICL mechanisms from the perspective of the inter-demonstration relationship. According to the implementation principles of ICL [12], it requires LLMs to induce from a few demonstrations to the task’s essence, i.e., a specific input-label mapping that satisfies the task. However, due to the limited number of demonstrations, the input-label mapping that conforms to the demonstrations is not unique [4, 13], as shown in the top half of Fig. 1(b). Therefore, LLMs may induce input-label mappings that are different from the task’s essence. In this study, we refer to this phenomenon as demonstration bias. Clearly, demonstration bias is detrimental to ICL, so how to select demonstrations to mitigate such bias is a worthwhile research direction. Based on human experience, a common solution is to consider the inter-demonstration relationship: selecting demonstrations that can be compared with each other, as shown in the bottom half of Fig. 1(b). In this study,
we refer to such demonstrations as Comparable Demonstrations (CDs). This solution stems almost from human instinct. For example, when teaching infants to differentiate objects, humans typically select comparable demonstrations (e.g., apples and pears), rather than selecting random or identical demonstrations. Intuitively, this inter-demonstration comparison can highlight the task’s essence while eliminating possible bias and spurious correlations [14, 15]. However, it remains unknown whether the human experience of selecting demonstrations in ICL scenarios can be applied to LLMs. Inspired by [16, 17], we attempt to construct CDs by minimally editing the texts to flip the corresponding labels (in Section 2). Through such manual editing, strong comparisons are created between demonstrations [18], thereby highlighting the task’s essence. To verify the effectiveness of CDs, we conduct extensive experiments: in Section 3, we employ instruction induction (LLMs generate descriptions of the task’s essence based on demonstrations) to intuitively observe the demonstration bias of LLMs, and we confirm that CDs can significantly reduce such bias; in Section 4, we analyze the performance of CDs in ICL scenarios from multiple perspectives, and we verify that CDs can bring performance gains to ICL, especially in the out-of-distribution scenario.
# 2. METHOD
# 2.1. Preliminaries of In-Context Learning
Generally, In-Context Learning (ICL) can be regarded as a conditional text generation process. LLM (parameterized by θ) performs ICL with K input-label pair demonstrations Ddemo = {x1, y1, x2, y2, . . . , xK, yK}, and LLM combines Ddemo to predict the label of the test example xt. Specifically, this process can be represented as:
(1)
� where T is the generated token length and is task-specific. Here, the role of demonstrations is to help LLM elicit an input-label mapping f : X →Y, x ∈X, y ∈Y that is capable of matching the task’s essence.
# 2.2. Comparable Demonstrations
According to the previous analysis, due to the small number of demonstrations (e.g., K = 4), ICL may suffer from demonstration bias. In this study, we attempt to eliminate such bias through Comparable Demonstrations (CDs). Specifically, the purpose of CDs is to highlight the task’s essence through the inter-demonstration comparison. Inspired by Counterfactually-Augmented Data (CAD) [16, 17], we can construct CDs by minimally editing the texts to flip the corresponding labels1. We consider that this text-level
1In this study, we employ existing CAD from [16] to construct CDs.
<div style="text-align: center;">editing can maximize comparisons between demonstrations. Here, we show an example of CDs in sentiment analysis as:</div>
editing can maximize comparisons between demonstrations. Here, we show an example of CDs in sentiment analysis as:
Original: I like this movie, I never get tired of watching it.
Edited:
I hate this movie, I am very tired of watching it. 
!"#$%$&'
(')*%$&'
This construction method has two benefits: on the one hand, the edited parts in texts are usually considered to involve the essential properties of tasks [19], which helps to highlight the task’s essence; on the other hand, since the majority of texts are unedited, LLMs can spontaneously eliminate potential spurious correlations in the demonstrations [14]. In the fine-tuning paradigm, fine-grained manual editing (similar to CAD), as a data augmentation method, is considered inefficient and costly by the NLP community [20]. However, the ICL paradigm only requires a small number of demonstrations (e.g., ∼10), while the requirements for data quality are relatively high (to prevent demonstrative bias). Therefore, we consider that this study finds a more meaningful application scenario for fine-grained manual editing, i.e., constructing high-quality CDs for ICL.
# 3. LLM’S DEMONSTRATION BIAS
Based on human experience, demonstration bias is evident in ICL, but it is not yet known whether it also exists in LLMs. To explicitly observe demonstration bias in LLMs, we perform instruction induction2: LLMs explicitly generate descriptions of the task’s essence based on a few demonstrations. By analyzing the generated instructions, we can perceive the inputlabel mappings induced by LLMs, and then compare them with the task’s essence. We believe this is the most straightforward method to observe LLM’s demonstration bias. Experimental Setup: In this section, the LLM we analyzed was openAI’s gpt-3.5-turbo, the current state-ofthe-art LLM. The dataset we employed was IMDb [22], a sentiment analysis dataset in the movie domain. The demonstration number was set to K = 4. We tested three strategies for selecting demonstrations: random selection (random), selection based on semantic similarity (nearest), and random selection of CDs (CDs random), and each strategy generated 100 instructions. The text embeddings were obtained via openAI’s text-embedding-ada-002, and we used cosine similarity to measure semantic similarity. Following previous research [21], the prompt we used is:
2[21] demonstrated that state-of-the-art LLMs (e.g. OpenAI’s InstructGPT) have the ability to implement instruction induction.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e2ce/e2ce6470-4288-4589-85a1-7715108908e6.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 2. Comparison of instruction quality under three demonstration selection strategies.</div>
Fig. 2. Comparison of instruction quality under three demonstration selection strategies.
Due to the challenging nature of the automated quality evaluation of instructions, we manually evaluated instructions generated by three demonstration selection strategies, and the evaluation results are shown in Fig. 2. We find that there are significant differences between the instructions induced by LLMs and the task’s essence, which implies that demonstration bias still exists in LLMs. To further analyze the demonstration bias in LLMs, we categorized bad instructions by error types. We summarized the error types of instructions into two categories and selected typical cases of both error types in Table 1: Case 1 highlights the detailed information in demonstrations (review of comedy), which is an over-interpretation of the input-label mapping; Case 2 misinterprets sentiment analysis as text assessment (overall evaluation), which is an over-simplification of the input-label mapping. Both error types are similar to characteristics that humans display in demonstration bias.
# 3.2. Can CDs Reduce Demonstration Bias?
As shown in Fig. 2, the instructions generated by CDs significantly outperform the other two strategies, indicating the effectiveness of CDs in reducing demonstration bias. This result preliminarily indicates that the human experience of selecting demonstrations is likely to be equally effective for LLMs. In addition, it is worth noting that the performance of demonstration selection based on semantic similarity (nearest) is surprisingly poor, and we speculate that similar demonstrations contain too much repetitive information, which hinders LLMs from inducing the correct input-label mapping.
# 4. PERFORMANCE OF CDS IN ICL
According to the previous analysis in Section 3, it is highly likely that CDs can bring performance gains to ICL, as it has the ability to reduce demonstration bias. Therefore, we evaluate CDs in ICL scenarios and analyze the performance of CDs from multiple perspectives.
Type
Instruction
Human
For each input, output whether the sentiment is positive or negative.
Case 1
Read the review of comedy and determine if it is positive or negative.
Case 2
Please read the following text and give an overall evaluation.
Table 1. Two types of bad instruction generated by LLMs. We mark the key parts in red.
Experimental Setup: In this section, the LLM we analyzed was still openAI’s gpt-3.5-turbo. Since there were existing CADs on sentiment analysis and Natural Language Inference (NLI), we conducted experiments on these two tasks. We conducted evaluations on both In-Distribution (ID) and Out-Of-Distribution (OOD) datasets: for sentiment analysis, ID dataset was IMDb [22], and OOD datasets were Amazon review [23] and Yelp review [24]; for NLI, ID dataset was SNLI [25], and OOD dataset was MNLI (split into matched and mismatched parts) [26]. The demonstration number was set to K = 4, 8, 12. In addition to the three strategies mentioned in Section 3, we added two more: selection based on semantic similarity in each label class (nearest class), and selection based on semantic similarity in CDs (CDs nearest). For each strategy, we sampled 500 examples in each dataset for evaluation. The text embeddings were still obtained via openAI’s text-embedding-ada-002, and we used cosine similarity to measure semantic similarity. For sentiment analysis, the prompt we used is:
The sentence is x1, the sentiment is y1 . . . The sentence is
xK, the sentiment is yK. The sentence is xt, the sentiment is
where {x} refer to texts in sentiment analysis, and y ∈
{positive, negative}. For NLI, the prompt we used is:
The premise is xp
1, the hypothesis is xh
1, the relation is y1 . . .
The premise is xp
K, the hypothesis is xh
K, the relation is yK.
The premise is xp
t , the hypothesis is xh
t , the relation is
where {xp} and {xh} refer to premise and hypothesis in NLI, and y ∈{entailment, neutral, contradiction}. Table 2 & 3 exhibit the experimental results of all demonstration selection strategies in ICL. Next, we analyze these results from three perspectives.
# 4.1. Dataset Distribution
Due to the broader application scope of LLMs, we analyze the performance of CDs from the perspective of dataset distribution (ID and OOD scenarios), and our findings are: (1) strategies based on semantic similarity mainly improve ID performance, while strategies based on CDs mainly improve OOD performance; (2) the strategy that combines the advantages of both (CDs nearest) is the most competitive strategy for balancing both ID and OOD scenarios. Our analysis of the performance differences between ID and OOD scenarios is as follows: strategies based on seman-
Methods
4-shot
8-shot
12-shot
ID
yelp
Amazon
ID
yelp
Amazon
ID
yelp
Amazon
random
93.4
93.2
86.8
95.6
91.4
86.6
95.6
91.0
85.4
nearest
95.2 (+1.8%)
90.4 (-2.8%)
85.2 (-1.6%)
95.8 (+0.2%)
86.8 (-4.6%)
84.2 (-2.4%)
96.3 (+0.7%)
83.8 (-7.2%)
81.6 (-3.8%)
nearest class
95.4 (+2.0%)
92.0 (-1.2%)
88.4 (+1.6%)
96.2 (+0.6%)
91.2 (-0.2%)
86.8 (-0.2%)
96.3 (+0.7%)
91.0 (+0.0%)
84.4 (-1.0%)
CDs random
93.4 (+0.0%)
94.0 (+0.8%)
90.0 (+3.2%)
95.4 (-0.2%)
93.4 (+2.0%)
91.4 (+4.8%)
95.4 (-0.2%)
94.4 (+3.4%)
90.8 (+5.4%)
CDs nearest
94.0 (+0.6%)
93.6 (+0.4%)
88.8 (+2.0%)
95.8 (+0.2%)
94.4 (+3.0%)
89.4 (+2.8%)
96.4 (+0.8%)
93.8 (+2.8%)
90.2 (+4.8%)
Table 2. Accuracy of different demonstration selection strategies on sentiment analysis. We consider the random selection (random) as benchmark, those with performance degradation are marked as red; those with performance improvement within 1% are marked as yellow; those with performance improvement above 1% are marked as green. The best performance is bold.
Methods
4-shot
8-shot
12-shot
ID
MNLI-m
MNLI-mm
ID
MNLI-m
MNLI-mm
ID
MNLI-m
MNLI-mm
random
74.5
69.5
72.8
76.5
70.9
73.9
74.4
70.3
73.7
nearest
75.8 (+1.3%)
69.9 (+0.4%)
75.1 (+2.3%)
76.8 (+0.3%)
70.2 (-0.7%)
74.1 (+0.2%)
75.8 (+1.4%)
71.9 (+1.6%)
74.9 (+1.2%)
nearest class
74.9 (+0.4%)
71.9 (+2.4%)
74.5 (+1.7%)
75.7 (-0.8%)
73.6 (+2.7%)
74.5 (+0.6%)
75.8 (+1.4%)
72.6 (+2.3%)
73.5 (-0.2%)
CDs random
73.3 (-1.2%)
73.0 (+3.5%)
74.7 (+1.9%)
75.6 (-0.9%)
73.3 (+2.4%)
74.0 (+0.1%)
77.2 (+2.8%)
72.4 (+2.1%)
75.5 (+1.8%)
CDs nearest
75.2 (+0.7%)
73.5 (+4.0%)
75.3 (+2.5%)
77.1 (+0.6%)
74.2 (+3.3%)
76.1 (+2.2%)
77.3 (+2.9%)
72.7 (+2.4%)
74.8 (+1.1%)
tic similarity would allow LLMs to learn dataset-specific information (e.g., information about movies in IMDb dataset), which cannot be generalized to OOD scenarios; while CDs attempt to help LLMs focus on the task’s essence, which to some extent alleviates the focus on dataset-specific information, thereby helping to improve the performance of LLMs in OOD scenarios. Since the application scenarios of ICL are usually uncertain, we consider that the performance gains of CDs in OOD scenarios is meaningful.
# 4.2. Task Complexity
To clarify the extent of demonstration bias and the applicability of CDs, we analyze the experimental results from task complexity. Generally, the NLP community considers NLI to be a more complex task than sentiment analysis. We find that different strategies show larger performance differences on the simple task (sentiment analysis), sometimes approaching 10%, while their performance is more stable on the complex task (NLI), typically below 3%. This may suggest that demonstration bias is more likely to influence LLMs to induce the task’s essence on simple tasks. Therefore, although the experiments show that CDs are also effective for complex tasks, they are more necessary for simple tasks.
# 4.3. Demonstration Number
Demonstration number is often considered to affect the performance of ICL and is closely related to demonstration bias. Therefore, we analyze the performance differences between CDs and other strategies from this aspect.
The experimental results indicate that in most cases, the performance of ICL improves as the demonstration number increases, which is in line with our expectation of demonstration bias. In addition, we find that CDs are relatively robust to the demonstration number, while strategies based on semantic similarity are more sensitive. Since the ICL paradigm is extremely sensitive to the demonstration selection, the robustness exhibited by CDs is a significant advantage.
# 5. CONCLUSION
In this study, we explore the mechanisms of ICL from the perspective of the inter-demonstration relationship, and we consider that demonstration bias may exist in LLMs due to the limitation of the demonstration number. Inspired by human experience, we attempt to construct Comparable Demonstrations (CDs) by minimally editing the texts to flip the corresponding labels, to mitigate such potential bias. A series of experiments indicate that CDs bring performance gains to the ICL, especially in the OOD scenario. In the future, we aim to explore ICL in more complex scenarios (e.g. mathematical reasoning, multi-hop inference), and rigorously analyze the mechanisms of ICL from the perspective of the interdemonstration relationship. We consider this study has two main limitations. First, although ICL requires relatively few demonstrations, manual annotation remains expensive, and automatically generating CDs is not currently feasible. Second, CDs only consider oneto-one relationships between demonstrations, without taking into account many-to-many relationships, which clearly does not make full use of the inter-demonstration relationship.
[1] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus, “Emergent abilities of large language models,” Trans. Mach. Learn. Res., 2022. [2] Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Bing Yin, and Xia Hu, “Harnessing the power of llms in practice: A survey on chatgpt and beyond,” CoRR, 2023. [3] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen, “What makes good in-context examples for gpt-3?,” in ACL, 2022. [4] Chenglei Si, Dan Friedman, Nitish Joshi, Shi Feng, Danqi Chen, and He He, “Measuring inductive biases of in-context learning with underspecified demonstrations,” ACL, 2023. [5] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig, “Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing,” ACM Comput. Surv., 2023. [6] Jane Pan, Tianyu Gao, Howard Chen, and Danqi Chen, “What in-context learning ”learns” in-context: Disentangling task recognition and task learning,” ACL, 2023. [7] Caoyun Fan, Jindou Chen, Yaohui Jin, and Hao He, “Can large language models serve as rational players in game theory? a systematic analysis,” in AAAI, 2024. [8] Caoyun Fan, Jidong Tian, Yitian Li, Wenqing Chen, Hao He, and Yaohui Jin, “Chain-of-thought tuning: Masked language models can also think step by step in natural language understanding,” in EMNLP, 2023. [9] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer, “Rethinking the role of demonstrations: What makes in-context learning work?,” in EMNLP, 2022. 10] Kang Min Yoo, Junyeob Kim, Hyuhng Joon Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-Woo Lee, Sang-goo Lee, and Taeuk Kim, “Ground-truth labels matter: A deeper look into input-label demonstrations,” in EMNLP, 2022. 11] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola, “Automatic chain of thought prompting in large language models,” CoRR, 2022. 12] Albert Webson and Ellie Pavlick, “Do promptbased models really understand the meaning of their prompts?,” in NAACL, 2022.
[13] Ruixiang Tang, Dehan Kong, Longtao Huang, and Hui Xue, “Large language models can be lazy learners: Analyze shortcuts in in-context learning,” CoRR, 2023. [14] Caoyun Fan, Wenqing Chen, Jidong Tian, Yitian Li, Hao He, and Yaohui Jin, “Improving the out-ofdistribution generalization capability of language models: Counterfactually-augmented data is not enough,” ICASSP, 2023. [15] Caoyun Fan, Wenqing Chen, Jidong Tian, Yitian Li, Hao He, and Yaohui Jin, “Unlock the potential of counterfactually-augmented data in out-of-distribution generalization,” Expert systems with applications, 2024. [16] Divyansh Kaushik, Eduard H. Hovy, and Zachary Chase Lipton, “Learning the difference that makes a difference with counterfactually-augmented data,” ICLR, 2020. [17] Divyansh Kaushik, Amrith Setlur, Eduard H. Hovy, and Zachary Chase Lipton, “Explaining the efficacy of counterfactually augmented data,” in ICLR, 2021. [18] Linyi Yang, Jiazheng Li, P’adraig Cunningham, Yue Zhang, Barry Smyth, and Ruihai Dong, “Exploring the efficacy of automatically generated counterfactuals for sentiment analysis,” ACL, 2021. [19] Zhao Wang and Aron Culotta, “Robustness to spurious correlations in text classification via automatically generated counterfactuals,” in AAAI, 2021. [20] Nitish Joshi and He He, “An investigation of the (in)effectiveness of counterfactually augmented data,” ACL, 2022. [21] Or Honovich, Uri Shaham, Samuel R. Bowman, and Omer Levy, “Instruction induction: From few examples to natural language task descriptions,” CoRR, 2022. [22] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts, “Learning word vectors for sentiment analysis,” in ACL, 2011. [23] Jianmo Ni, Jiacheng Li, and Julian McAuley, “Justifying recommendations using distantly-labeled reviews and fine-grained aspects,” in EMNLP, 2019. [24] Xiang Zhang, Junbo Jake Zhao, and Yann LeCun, “Character-level convolutional networks for text classification,” NeurIPS, 2015. [25] Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning, “A large annotated corpus for learning natural language inference,” in EMNLP, 2015. [26] Adina Williams, Nikita Nangia, and Samuel R. Bowman, “A broad-coverage challenge corpus for sentence understanding through inference,” in NAACL, 2018.
In Table 4, we display more typical bad cases in instruction induction. These cases can be used as a supplement to Table 1.
Type
Instruction
Case 1
Please write an output for each of the following in-
puts based on your overall impression of the movie:
positive, negative, or neutral.
Case 2
Read the input and write an output based on your
overall impression of the movie.
Case 3
Please watch this movie and give me your honest
opinion about it.
Case 4
Watch this foreign film and write your overall impres-
sion of it.
Case 5
Provide an output for each of the given inputs.
Case 6
Provide an output based on your opinion of the movie
or experience described in the input.
Case 7
Watch the movie and write a review.
Case 8
Read the following statement and determine whether
the overall sentiment is positive, negative, or neutral.
Case 9
Read the following descriptions of movies and write
’positive’ or ’negative’ as the output based on
whether the reviewer liked the movie or not.
Case 10
Provide an output based on your experience or opin-
ion of the given input.
Table 4. More typical bad cases in instruction induction. Key parts are marked in red.
These bad cases further confirm that there is also demonstration bias in LLMs. Therefore, we need to study how to eliminate this bias.
# B. RULES FOR MANUAL EVALUATION
We divide the generated instructions into three categories:
1. Correct and satisfying instruction. This refers to instructions that humans can implement 0-shot reasoning, and can generalize to out-of-distribution datasets. 2. Acceptable instruction with minor imperfections. This refers to the presence of some redundant words. Humans can still implement 0-shot reasoning, but cannot generalize to out-of-distribution datasets. 3. Invalid instruction with significant errors. This refers to the inability of even humans to implement 0-shot reasoning. The instruction completely misunderstands the essence of tasks.
1. Correct and satisfying instruction. This refers to instructions that humans can implement 0-shot reasoning, and can generalize to out-of-distribution datasets.
