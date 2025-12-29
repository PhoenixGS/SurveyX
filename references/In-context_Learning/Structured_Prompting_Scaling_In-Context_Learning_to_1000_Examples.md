# Structured Prompting: Scaling In-Context Learning to 1,000 Examples
u Hao∗, Yutao Sun∗, Li Dong, Zhixiong Han, Yuxian Gu, Furu Wei Microsoft Research
https://github.com/microsoft/LMOps
# Abstract
Abstract
Large language models have exhibited intriguing in-context learning capability, achieving promising zero- and few-shot performance without updating the parameters. However, conventional in-context learning is usually restricted by length constraints, rendering it ineffective to absorb supervision from a large number of examples. In order to go beyond few shots, we introduce structured prompting that breaks the length limit and scales in-context learning to thousands of examples. Specifically, demonstration examples are separately encoded with well-designed position embeddings, and then they are jointly attended by the test example using a rescaled attention mechanism. So we can scale the number of exemplars with linear complexity instead of quadratic complexity with respect to length. Experimental results on a diverse set of tasks show that our approach improves end-task performance and reduces evaluation variance over conventional in-context learning as the number of demonstration examples increases. Code has been released at https://aka.ms/structured-prompting.
# 1 Introduction
In-context learning [BMR+20] prompts pretrained language models to perform downstream tasks without any parameter update. Rather than fine-tuning the parameters for few-shot learning, we feed task-specific instructions and input-output demonstrations into large language models. Then the evaluation input is conditioned on the given context to make predictions. The paradigm is appealing since we can host language modeling inference as a general-purpose service for a wide range of tasks. Most previous studies [BMR+20, RBC+21, SPN+22, HSD+22] of in-context learning are conducted for few-shot learning. For example, PaLM [CND+22] typically conditions on five demonstration examples for most benchmarks. However, the restricted number of training instances potentially limits the usage of in-context learning in practice, especially when we have many examples. In comparison, fine-tuning is able to consume much more training examples for supervision despite the costly training. The above data utilization issue motivates us to empower in-context learning with more demonstration examples. Directly scaling up the size is challenging. For example, the language models with absolute position embeddings are pretrained with a predefined length. So the naive concatenation of many examples typically exceeds the maximum length. Moreover, the conventional self-attention mechanism suffers from quadratic complexity in terms of computation and memory consumption, rendering scaling up infeasible. In addition, more shots tend to reduce the performance variance [ZWF+21, LBM+22] caused by different choices and permutations of demonstration examples. In this paper, we propose structured prompting to scale the number of examples to orders of magnitude larger and significantly improve stability. Rather than simply concatenating all demonstrations
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/04ee/04eee9d2-4276-4bc5-af88-6980ad383220.png" style="width: 50%;"></div>
# (a) Conventional Prompting
Figure 1: The illustrations of conventional prompting and structured prompting. We first encode group-structured exemplars independently. Then all exemplars are incorporated into the test input through the rescaled attention mechanism.
together, we divide a large number of demonstrations into multiple groups, which are independently encoded by the language model. So the encoding complexity becomes linear with respect to the number of groups, instead of quadratic complexity with respect to all examples. The position embeddings of grouped prompts are right-aligned to be next to the test input. Next, the input is encoded by conditioning on grouped prompts, where rescaled attention is proposed to normalize the attention scores. Our structured prompting is flexible to encode plenty of context in an efficient way. We conduct experiments on a variety of tasks, such as text classification, multi-choice, and open-ended tasks. Structured prompting successfully scales the number of demonstrations to much larger sizes. Our method substantially outperforms conventional in-context learning across various model sizes and tasks. Moreover, the approach greatly improves the stability of in-context learning.
# 2 Background: In-Context Learning
In-context learning [BMR+20] allows language models to recognize the desired task and generate answers for given inputs by conditioning on instructions and input-output demonstration examples, rather than updating model parameters as fine-tuning. Formally, given a set of N labeled examples Dtrain = {(xi, yi)}N i=1 (i.e., N-shot in-context learning), each of them is transformed into a semantically meaningful demonstration di = T (xi, yi) using a hand-crafted template T . For example, the template of a binary sentiment classification task can be “Sentence: xi. Sentiment: yi”, where yi is Negative or Positive. All demonstrations are concatenated as the context Z = d1 ⊕... ⊕dN. Newlines or special tokens are used to delimit them. For each test input xtest, we prompt the language model with the concatenation of Z and xtest. The predicted answer is the completion with the highest language model probability, i.e., arg maxc∈Y PLM(yc|Z⊕T (xtest)), where Y is the set of all possible candidates. For conventional in-context learning, the number of demonstrations N is restricted by the maximum length of pretrained Transformers (typically 2048), which typically fits 5 to 100 examples depending on the datasets.
# 3 Methods
In this section, we introduce structured prompting, which scales in-context learning to many examples under limited computation complexity. An overview of our approach is shown in Figure 1. First, we divide examples into groups. We obtain representations of group-structured exemplars independently with right-aligned position embeddings. Second, we incorporate the encoding results into the test input through the rescaled attention mechanism in each layer. Then the language model generates the answer.
<div style="text-align: center;">(b) Structured Prompting</div>
# (b) Structured Prompting
Suppose we have N demonstration examples. We randomly divide these examples into M groups {Zi}M i=1. Each group is a concatenation of exemplars Zi = dNi−1+1 ⊕... ⊕dNi, where N0 = 0 and NM = N. As shown in Figure 1, all exemplar groups are separately encoded by the language model. Then we use the context encoding results for structured prompting. Notice that only key and value vectors of self-attention need to be cached, which are attended by the test input. Grouped context encoding is able to consume longer sequences. In contrast, conventional in-context learning cannot exploit context efficiently because the concatenation of all examples far exceeds the window size of pretrained Transformers. Moreover, the computation complexity of conventional in-context learning is quadratic to the number of demonstration examples N because of the matrix multiplication of queries and keys. It is infeasible when N increases. Our approach improves it through splitting groups, which reduces the complexity from O(N 2) to O(N 2/M). Right-Aligned Position Embedding We right-align all the groups so that they have the same maximum position index. Hence all groups can have the same relative distance with respect to the test input. It is critical that the test input can be adjacent to all exemplars and pay equal attention to them. One way is to use left padding, i.e., pad tokens or space tokens. The other way is to set a maximum length for grouped context, truncating exemplars from the left side.
Suppose we have N demonstration examples. We randomly divide these examples into M groups {Zi}M i=1. Each group is a concatenation of exemplars Zi = dNi−1+1 ⊕... ⊕dNi, where N0 = 0 and NM = N. As shown in Figure 1, all exemplar groups are separately encoded by the language model. Then we use the context encoding results for structured prompting. Notice that only key and value vectors of self-attention need to be cached, which are attended by the test input. Grouped context encoding is able to consume longer sequences. In contrast, conventional in-context learning cannot exploit context efficiently because the concatenation of all examples far exceeds the window size of pretrained Transformers. Moreover, the computation complexity of conventional in-context learning is quadratic to the number of demonstration examples N because of the matrix multiplication of queries and keys. It is infeasible when N increases. Our approach improves it through splitting groups, which reduces the complexity from O(N 2) to O(N 2/M).
Right-Aligned Position Embedding We right-align all the groups so that they have the same maximum position index. Hence all groups can have the same relative distance with respect to the test input. It is critical that the test input can be adjacent to all exemplars and pay equal attention to them. One way is to use left padding, i.e., pad tokens or space tokens. The other way is to set a maximum length for grouped context, truncating exemplars from the left side.
# 3.2 Structured Prompting
After encoding context exemplars in groups, the next step is to use them for prompting. As shown in Figure 1, all exemplars are incorporated into representations of the test input through a rescaled attention mechanism. Specifically, the test input is fed into the language model, conditioning on both itself and grouped exemplars. Let L denote the maximum length of grouped context. The position index of the test input starts with L + 1 so that it is contiguous to all groups.
Rescaled Attention We use x instead of T (xtest) for brevity. In each layer, we concatenate the keys and values of all exemplars and the test input, i.e., ˆK = [KZ1, ..., KZM , Kx], ˆV = [VZ1, ..., VZM , Vx]. The test input x attends both demonstrations and itself with causal masks. Then the attention output is computed via:
where � j Aij = 1, the query vector qi ∈Qx, the key vector kj ∈ˆK⊺, and d is dimension of queries and keys. Compared with vanilla self-attention used in Transformers [VSP+17], the only difference is the scaling factor M in Equation (2). Without rescaled attention, the test input will attend too much to exemplars and ignore itself as the number of exemplars increases. Intuitively, our method modifies the softmax function in self-attention by repeating test input tokens M times. So we can augment the test input with multiple groups of context.
 � Compared with vanilla self-attention used in Transformers [VSP+17], the only difference is the scaling factor M in Equation (2). Without rescaled attention, the test input will attend too much to exemplars and ignore itself as the number of exemplars increases. Intuitively, our method modifies the softmax function in self-attention by repeating test input tokens M times. So we can augment the test input with multiple groups of context.
# 4 Experiments
# 4.1 Setup
Models We conduct experiments on open-source GPT-like (i.e., decoder-only Transformer) models released by [ABG+21]. We use three models of different sizes with 1.3B, 6.7B, and 13B parameters. The context window contains up to 2048 tokens. For large-scale experiments, we use BLOOM176B [SAW22].
(1)
of sentiment: SST-2 [SPW+13], SST-5 [SPW+13], MR [PL05], Subj [PL04]; topic: DBPedia [ZZL15], AGNews [ZZL15], TREC [VT00]; natural language inference: CB [dMST19], RTE [DGM06, BHDD+06, GMDD07, BDD+09]; and question answering: BoolQ [CLC+19]. For multi-choice tasks, we consider sentence completion: HellaSwag [ZHB+19], StoryCloze [MRL+17]; commensense reasoning: PIQA [BZB+20], OpenBookQA [MCKS18], ARC-Easy [CCE+18], ARCChallenge [CCE+18]; and COPA from SuperGLUE benchmark [WPN+19]. For open-ended generation, we consider closed-book question answering: NaturalQS [KPR+19], WebQS [BCFL13], TriviaQA [JCWZ17]; and extractive reading comprehension: SQuAD [RZLL16], SQuADv2 [RJL18]. Evaluation Protocol Following [BMR+20], we randomly draw N fixed examples from the training set as conditioning and report evaluation results on the development set. The demonstrations are separated by a special token. For StoryCloze, there is no available training set so we draw from the development set and evaluate on the test set. To reduce cost, we use 4k test examples for inference. There are only six datasets with development sets larger than 4k and we randomly sample a fixed subset of them. We design a hand-crafted template for each text classification dataset. For other datasets, we follow the same template in GPT-3. All templates are listed in Appendix A. Notice that demonstrations for reading comprehension datasets (SQuAD, SQuADv2) are constructed slightly differently from the original GPT-3. In GPT-3, the demonstrations provided for each test input are question-answer pairs from the same background passage as it. Here we consider a more strict setting where demonstrations are constructed with different passage-question-answer combinations from the training set. For multi-choice tasks, we score each completion by the per-token language model likelihood (normalize perplexity by length) and pick the one with the highest score as the final answer. For text classification, we treat it as a multi-choice task with only one token per option and design meaningful names for each option. For open-ended generation tasks, we use beam search with a beam width of 3, a length penalty of α = 0.6, and a maximum generation length of 30. We report exact-match accuracy for closed-book QA and F1 score for SQuAD and SQuADv2. For conventional prompting, we report results for 0-shot and the largest shot that fills one context window (1×). The maximum number of shots is calculated based on the average length of each dataset. Structured prompting is no longer limited by the context window size and can scale in-context learning to thousands of examples. For datasets with shorter lengths (e.g., SST-2, Subj), we report results of 500-shot and 1000-shot. For datasets with longer lengths (e.g., AGNews, SQuAD), we choose the number of shots according to their average lengths. We find it beneficial to put as many demonstrations as possible in each group. Thus we adopt it in our main experiments. Under each setting, we use six different random seeds for all tasks and report the mean and variance.
# 4.2 Results
Text Classification First, we consider text classification tasks, the results of nine datasets are shown in Table 1. Conditioning on thousands of examples, structured prompting brings consistent and significant improvements (3-5 absolute gains) for in-context learning. Moreover, our method makes in-context learning much more stable across multiple seeds, while conventional in-context learning is sensitive to different demonstration selections and permutations. In most cases, providing more examples leads to better performance and lower variance. For easier tasks like sentiment (SST-2, Subj, and MR) and topic classification (TREC and DBPedia), the improvement is more obvious and stable (the variance is generally less than 1.0). For natural language inference (CB), the result is relatively unstable, e.g., the 6.7B model has an outlier that classifies all examples into the same class, which indicates that inference is still a challenging task for in-context learning. Multi-Choice Tasks The performance comparison of multi-choice tasks is shown in Table 2. Structured prompting still brings consistent gains on these tasks. However, we notice that the improvement of our method on these tasks is relatively small compared with text classification. Besides, utilizing more demonstrations does not always lead to better performance. Scaling up the model size instead of the number of demonstrations can bring more improvements in these tasks. Open-Ended Generation Both text classification and multi-choice tasks restrict the label space. To evaluate structured prompting on open-ended generations tasks, we consider two types of datasets:
Multi-Choice Tasks The performance comparison of multi-choice tasks is shown in Table 2. Structured prompting still brings consistent gains on these tasks. However, we notice that the improvement of our method on these tasks is relatively small compared with text classification. Besides, utilizing more demonstrations does not always lead to better performance. Scaling up the model size instead of the number of demonstrations can bring more improvements in these tasks.
SST-2
Model Size
N
1.3B
6.7B
13B
0
70.50.0 65.90.0 64.70.0
102(1×) 88.07.1 92.21.5 93.80.7
500
89.66.4 93.30.9 93.50.8
1000
92.20.5 93.30.6 94.10.4
SST-5
Model Size
N
1.3B
6.7B
13B
0
39.20.0 32.50.0 37.60.0
66(1×) 38.05.5 46.12.6 47.22.6
500
42.61.1 49.20.7 50.11.2
1000
42.60.5 49.30.4 48.70.6
MR
Model Size
N
1.3B
6.7B
13B
0
65.90.0 58.00.0 60.70.0
63(1×) 84.14.1 91.80.7 92.70.4
500
88.52.1 92.20.2 92.90.2
1000
89.70.3 92.20.2 92.60.3
Subj
Model Size
N
1.3B
6.7B
13B
0
72.50.0 63.40.0 76.20.0
55(1×) 85.36.8 67.17.0 88.43.2
500
88.90.7 65.22.7 90.21.2
1000
89.50.8 69.31.5 89.31.2
DBPedia
Model Size
N
1.3B
6.7B
13B
0
76.60.0 77.30.0 79.80.0
21(1×)
84.75.6 68.23.9 91.53.1
100
87.82.0 93.80.6 94.20.8
200
87.22.1 94.30.4 94.90.2
AGNews
Model Size
N
1.3B
6.7B
13B
0
46.40.0 29.00.0 37.30.0
23(1×)
70.67.9 76.75.9 83.46.2
100
71.614.5 74.35.2 85.41.9
200
76.29.0 75.73.2 86.31.3
TREC
Model Size
N
1.3B
6.7B
13B
0
46.40.0 41.60.0 44.60.0
116(1×) 62.33.5 75.82.0 81.43.5
500
66.51.2 79.31.0 84.42.2
1000
66.61.9 79.40.9 85.10.8
CB
Model Size
N
1.3B
6.7B
13B
0
37.50.0 58.90.0 28.60.0
18(1×) 48.25.1 50.30.7 48.212.9
125
56.51.9 50.00.0 62.21.9
250
56.92.8 50.00.0 59.80.9
BoolQ
Model Size
N
1.3B
6.7B
13B
0
59.50.0 64.20.0 67.00.0
6(1×)
57.48.0 63.86.6 70.22.2
50
59.37.1 61.78.8 72.52.6
100
60.44.1 63.95.0 73.11.2
HellaSwag
Model Size
N
1.3B
6.7B
13B
0
56.10.0 67.60.0 70.90.0
24(1×)
56.00.4 68.30.2 71.90.3
100
56.70.2 68.70.3 72.00.2
200
56.40.3 68.80.2 72.10.2
StoryCloze
Model Size
N
1.3B
6.7B
13B
0
77.00.0 79.30.0 80.20.0
38(1×)
78.60.8 83.20.5 85.00.4
500
78.70.2 83.60.1 86.10.2
1000
78.30.2 83.50.1 85.70.2
ARC-E
Model Size
N
1.3B
6.7B
13B
0
48.30.0 57.50.0 58.20.0
70(1×)
61.90.5 70.40.5 73.90.6
500
62.30.5 71.20.5 75.40.9
1000
61.90.5 71.10.5 75.30.3
PIQA
Model Size
N
1.3B
6.7B
13B
0
74.30.0 76.80.0 78.10.0
54(1×) 74.40.5 78.00.5 79.80.4
500
74.30.2 78.60.1 79.90.1
1000
74.10.1 78.30.3 80.30.2
OBQA
Model Size
N
1.3B
6.7B
13B
0
34.40.0 35.80.0 39.00.0
118(1×) 34.20.8 39.60.4 41.11.1
500
35.00.7 41.01.0 42.70.7
1000
34.50.5 40.90.7 43.00.6
ARC-C
Model Size
N
1.3B
6.7B
13B
0
30.20.0 32.20.0 35.90.0
61(1×)
33.91.5 39.10.4 42.30.7
500
32.60.6 39.20.4 42.60.8
1000
32.90.4 39.40.6 42.20.4
COPA
Model Size
N
1.3B
6.7B
13B
0
71.00.0 83.00.0 81.00.0
134(1×) 75.21.6 85.71.4 87.32.6
200
75.51.7 85.71.7 86.82.4
400
76.52.3 85.30.9 87.71.5
NQ
Model Size
N
1.3B
6.7B
13B
0
1.20.0 0.40.0
1.10.0
100(1×) 7.20.2 14.00.2 16.10.3
500
7.20.2 13.90.1 16.30.3
1000
7.00.1 13.70.2 16.30.4
WebQS
Model Size
N
1.3B
6.7B
13B
0
0.80.0
0.10.0
0.80.0
108(1×) 14.91.1 22.21.4 24.41.2
500
16.60.3 23.00.5 26.40.5
1000
17.10.3 23.70.7 27.10.5
TriviaQA
Model Size
N
1.3B
6.7B
13B
0
8.30.0
1.40.0
6.80.0
66(1×)
17.91.2 31.82.2 37.53.4
125
18.51.0 31.51.7 37.22.6
250
19.50.6 33.01.1 39.91.4
SQuAD
Model Size
N
1.3B
6.7B
13B
0
24.10.0 25.60.0 31.60.0
6(1×)
52.43.0 67.01.2 68.71.5
25
53.82.5 68.20.8 71.80.9
50
55.21.3 69.30.5 73.60.6
SQuADv2
Model Size
N
1.3B
6.7B
13B
0
11.30.0 12.00.0 14.80.0
6(1×)
30.33.0 36.83.4 36.22.2
25
31.62.1 37.92.4 37.61.2
50
32.01.4 38.31.0 38.30.9
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5703/570345a7-bb1a-4fa2-add0-e4b286c4da0c.png" style="width: 50%;"></div>
closed-book question answering without conditioning on auxiliary information and extractive reading comprehension. Results are shown in Table 3. We observe that for all datasets except Natural Questions, incorporating more demonstrations via structured prompting leads to a monotonously increasing performance boost and a decreasing variance. Especially for SQuAD, our method has nearly five points improvement over the baseline on 13B LM with only 50 examples. For NQ, there is a negative observation that structured prompting has little gains compared with conventional in-context learning. For SQuAD, we tried up to 50-shot settings because of the long length of a single demonstration. We believe that the performance can still increase if the number of demonstrations can be further expanded.
The previous results show that the gains from structured prompting decrease slightly as the model size increases. To verify the effectiveness of our method on huge models, we conduct experiments with a subset of datasets on BLOOM-176B [SAW22].
Evaluation Protocol Our experiments are implemented on 8×80GB A100 GPUs. We evaluate conventional prompting under different prompt lengths (0.5× means the prompt fills half of the context window size) and structured prompting under different group numbers (5× means five groups of prompts). Under each setting, we use five different random seeds for all tasks and report the average results.
Evaluation Protocol Our experiments are implemented on 8×80GB A100 GPUs. We evaluate conventional prompting under different prompt lengths (0.5× means the prompt fills half of the context window size) and structured prompting under different group numbers (5× means five groups of prompts). Under each setting, we use five different random seeds for all tasks and report the average results. Large Model Results The performance comparisons on 176-Billion Model are shown in Table 4. For most datasets, the performance gets better as the number of groups increases. The variance results also show that structured prompting is highly stable when using five groups. For TREC and PIQA, we observe that 3× achieves better results that 5× but they both outperform the conventional in-context learning. Our experiments show that large language models still have the potential to achieve better prompting results when utilizing more demonstrations.
Large Model Results The performance comparisons on 176-Billion Model are shown in Table 4 For most datasets, the performance gets better as the number of groups increases. The variance results also show that structured prompting is highly stable when using five groups. For TREC and PIQA, we observe that 3× achieves better results that 5× but they both outperform the conventional in-context learning. Our experiments show that large language models still have the potential to achieve better prompting results when utilizing more demonstrations.
BLOOM-176B
RTE
CB
SST-2
SST-5
TREC
BoolQ
Subj
MR
OBQA
PIQA
Avg
0×
58.10.0
39.30.0
83.10.0
35.00.0
14.40.0
62.20.0
54.70.0
63.50.0
41.00.0
78.50.0
53.0
0.25×
67.84.9
53.927.5
94.11.1
43.96.2
51.84.8
72.72.7
88.56.3
92.70.4
44.00.8
79.20.3
68.9
0.5×
66.55.4
62.510.0
94.70.4
41.13.7
61.23.3
74.71.5
90.74.3
92.60.5
44.71.3
79.80.3
70.9
1×
67.17.7
75.45.6
94.20.5
46.72.0
88.62.3
75.52.2
93.61.2
92.60.4
45.41.1
79.80.4
75.9
3×
70.52.7
76.43.2
94.40.5
45.74.4
91.02.2
77.11.8
95.20.5
93.00.4
45.60.6
80.30.4
76.9
5×
72.32.6
77.52.4
94.70.4
47.02.2
89.72.4
77.91.7
95.70.4
93.10.3
46.00.3
80.10.2
77.4
RTE
CB
SST-2
SST-5
TREC
Subj
MR
BoolQ
OBQA
PIQA
#shot (1×)
24
18
102
66
116
55
63
6
118
54
Table 4: Results averaged over 5 random seeds on BLOOM-176B. The table below shows the actual shot number that fills the entire context window. The shot number of other settings can be calculated by N×#shot.
# 4.4 Stability Analysis
Prior efforts demonstrate that different choices and permutations of conditioning examples can cause a high variance for in-context learning. We now investigate their impacts on structured prompting Figure 2 shows how the performance and variance of in-context learning change as the number of examples increases. We observe that our method can significantly reduce inference variance for text classification and open-ended generation tasks. This phenomenon is less pronounced for multi-choice tasks that correlate better with pretraining. For larger LMs (13B), the baseline approach is stable enough in the max-shot case. Structured prompting can further boost its stability while achieving better performance. It suggests that in-context learning is underestimated under the few-shot setting and our method can bring key benefits to maximize its stability and effectiveness.
# 4.5 Ablation Studies
The Effect of Prompt Length The results of different prompt lengths are shown in Figure 3a. We control the number of examples as a constant value, so the group number is inversely proportional to the prompt length. Generally, the longer prompt length means better accuracy. In the small model (1.3B), the performance has a big drop when the prompt length is small (0.25×). It shows that the auto-regressive structure is still the most “natural” one. The best way to use structured prompting is to expand the groups under the maximal sequence length in pre-training. Therefore, we believe that the fundamental problem is language models’ ability to deal with exceeding sequence length. If the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/104d/104dd4f4-4e17-421d-9bec-d21ee6e40f3b.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Ablation of Prompt Length</div>
<div style="text-align: center;">Figure 3: Ablations of structured prompting.</div>
model’s extrapolation performance is satisfying, the benefits of structured prompting will degenerate to memory saving. We leave it for future work.
Table 5 shows that the “Truncate” strategy works well for both models. In FairseqLM, there is a <bos> token so that the subsequent tokens are less disturbed. In BLOOM, transitional invariance of Alibi [PSL21] is used to deal with padding and mask. However, it has a significant drop with “Pad Space”. The absence of <bos> seems to amplify the noise brought by blank space tokens. In conclusion, the “Truncate” strategy is the easiest and most natural way for aligning groups.
# 5 Related Work
Improving In-Context Learning Despite surprisingly effective, in-context learning suffers from certain vulnerabilities. For instance, the order of demonstrations and the choice of templates can cause a high variance in performance. [ZWF+21] show that the variance arises because of three types of biases (majority label, recency, and common token bias) and propose to calibrate model prediction by content-free output. [HHDW22] demonstrate that these biases cause the decision boundary shift and propose calibrating it by estimating the distribution of prototypical clusters. Other work focus on prompt engineering, including selecting the performant demonstration permutation [LBM+22] and semantically-similar in-context examples with a retrieval module [LSZ+22, RHB22]. We aim to improve in-context learning by scaling up the number of demonstrations.
<div style="text-align: center;">(b) Ablation of Scaling Factor</div>
Strategies
TREC
BoolQ
DBPedia
AGNews
HellaSwag
Average
BLOOM-7B
w/o Right-Alignment
56.9
62.2
94.7
82.1
58.6
70.9
Attention Mask
61.9
62.2
94.5
83.4
58.6
72.1
Pad Space
58.0
62.2
93.1
84.8
58.1
71.2
Truncate
61.9
62.2
95.4
84.3
58.5
72.5
FairseqLM-6.7B
w/o Right-Alignment
79.0
61.4
92.8
75.4
68.3
75.4
Attention Mask
79.3
61.9
92.9
75.4
68.7
75.6
Pad Space
80.0
62.0
93.0
74.7
68.7
75.7
Truncate
79.3
61.7
93.8
74.3
68.7
75.6
Understanding In-Context Learning Another line of work investigates understanding how incontext learning works. [XRLM22] propose a Bayesian inference framework to explain it, where the language model implicitly infers a concept when making a prediction. Since in-context learning emerges after pretraining on a large corpus, some efforts study the correlation between pretraining corpus and in-context learning performance [SLA+22, RIGS22]. Additionally, previous work [MLH+22, KKC+22] investigates whether the label-mapping of demonstrations matters as expected.
Fusion-In-Decoder [IG21] propose Fusion-In-Decoder for encoder-decoder fine-tuning. The method was applied to open-domain question answering in order to leverage retrieved passages. Specifically, each retrieved supporting passage is encoded by bidirectional encoders. Then the decoder performs conventional attention over the concatenation of the representations of passages. In comparison, we focus on in-context learning with decoder-only models (such as GPT), without fine-tuning the original parameters. There are also several key technical differences, which are critical to making the method work well. First, we proposed rescaled attention to balance the attention allocation between context and test input. Second, we right-align position embeddings for structured context so that they have the same relative distance with respect to the input.
# 6 Discussion and Conclusion
In this work, we explore how to utilize more examples for in-context learning and propose structured prompting to scale up the number of examples under restricted computation complexity. We encode each group of demonstrations independently and prompt the language model with the concatenations of their representations via the rescaled attention. Experimental results across a diverse set of tasks show that our method outperforms the conventional approach. As the number of examples increases, our method achieves further gains and is much more stable. Despite the promising results, the current method still has some limitations. Ideal in-context learning should be invariant to demonstration permutations. If our method ensures that each group only contains one example, it satisfies the property indeed. However, in our experiments, we find that it works well on smaller models (i.e., 1.3B) but does not work on larger models (i.e., 13B). We hypothesize that larger models benefit more from autoregressive information, so we include multiple examples in each group. For future work, we will dive more deeply into this direction. Moreover, there is a mismatch between patterns of language model pretraining and in-context learning. The current objective makes language models only aware of sequential relationships but not parallel relationships. We would like to incorporate this prior knowledge during pretraining so that it aligns better with the scheme of downstream inference. In addition, structured prompting can be used to inject many long documents as context, e.g., using retrieved texts to augment generation.
In this work, we explore how to utilize more examples for in-context learning and propose structured prompting to scale up the number of examples under restricted computation complexity. We encode each group of demonstrations independently and prompt the language model with the concatenations of their representations via the rescaled attention. Experimental results across a diverse set of tasks show that our method outperforms the conventional approach. As the number of examples increases, our method achieves further gains and is much more stable.
[ABG+21] Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, Giri Anantharaman, Xian Li, Shuohui Chen, Halil Akin, Mandeep Baines, Louis Martin, Xing Zhou, Punit Singh Koura, Brian O’Horo, Jeff Wang, Luke Zettlemoyer, Mona Diab, Zornitsa Kozareva, and Ves Stoyanov. Efficient large scale language modeling with mixtures of experts, 2021. [BCFL13] Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on Freebase from question-answer pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1533–1544, Seattle, Washington, USA, October 2013. Association for Computational Linguistics. [BDD+09] Luisa Bentivogli, Ido Dagan, Hoa Trang Dang, Danilo Giampiccolo, and Bernardo Magnini. The fifth pascal recognizing textual entailment challenge. In In Proc Text Analysis Conference, 2009. [BHDD+06] Roy Bar-Haim, Ido Dagan, Bill Dolan, Lisa Ferro, and Danilo Giampiccolo. The second PASCAL recognising textual entailment challenge. In Proceedings of the Second PASCAL Challenges Workshop on Recognising Textual Entailment, 01 2006. [BMR+20] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020. [BZB+20] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020. [CCE+18] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018. [CLC+19] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. [CND+22] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek B Rao, Parker Barnes, Yi Tay, Noam M. Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Benton C. Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier García, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Oliveira Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Díaz, Orhan Firat, Michele Catasta, Jason Wei, Kathleen S. Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. PaLM: Scaling language modeling with pathways. ArXiv, abs/2204.02311, 2022.
+19] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
CND+22] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek B Rao, Parker Barnes, Yi Tay, Noam M. Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Benton C. Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier García, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Oliveira Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Díaz, Orhan Firat, Michele Catasta, Jason Wei, Kathleen S. Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. PaLM: Scaling language modeling with pathways. ArXiv, abs/2204.02311, 2022.
[DGM06] Ido Dagan, Oren Glickman, and Bernardo Magnini. The PASCAL recognising textual entailment challenge. In Proceedings of the First International Conference on Machine Learning Challenges: Evaluating Predictive Uncertainty Visual Object Classification, and Recognizing Textual Entailment, MLCW’05, pages 177–190, Berlin, Heidelberg, 2006. Springer-Verlag. [dMST19] Marie-Catherine de Marneffe, Mandy Simons, and Judith Tonhauser. The CommitmentBank: Investigating projection in naturally occurring discourse. Proceedings of Sinn und Bedeutung, 23(2):107–124, Jul. 2019. [GMDD07] Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and Bill Dolan. The third PASCAL recognizing textual entailment challenge. In Proceedings of the ACL-PASCAL Workshop on Textual Entailment and Paraphrasing, pages 1–9, Prague, June 2007. Association for Computational Linguistics. [HHDW22] Zhixiong Han, Yaru Hao, Li Dong, and Furu Wei. Prototypical calibration for few-shot learning of language models, 2022. [HSD+22] Yaru Hao, Haoyu Song, Li Dong, Shaohan Huang, Zewen Chi, Wenhui Wang, Shuming Ma, and Furu Wei. Language models are general-purpose interfaces. ArXiv, abs/2206.06336, 2022. [IG21] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative models for open domain question answering. In Paola Merlo, Jörg Tiedemann, and Reut Tsarfaty, editors, Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021, pages 874–880. Association for Computational Linguistics, 2021. [JCWZ17] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. triviaqa: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. arXiv e-prints, page arXiv:1705.03551, 2017. [KKC+22] Junyeob Kim, Hyuhng Joon Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-Woo Lee, Sang goo Lee, Kang Min Yoo, and Taeuk Kim. Ground-truth labels matter: A deeper look into input-label demonstrations, 2022. [KPR+19] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: a benchmark for question answering research. Transactions of the Association of Computational Linguistics, 2019. [LBM+22] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland, May 2022. Association for Computational Linguistics. [LSZ+22] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pages 100–114, Dublin, Ireland and Online, May 2022. Association for Computational Linguistics. [MCKS18] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018. [MLH+22] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? arXiv, 2022.
[DGM06] Ido Dagan, Oren Glickman, and Bernardo Magnini. The PASCAL recognising textual
entailment challenge. In Proceedings of the First International Conference on Machine
Learning Challenges: Evaluating Predictive Uncertainty Visual Object Classification,
and Recognizing Textual Entailment, MLCW’05, pages 177–190, Berlin, Heidelberg,
2006. Springer-Verlag.
[dMST19] Marie-Catherine de Marneffe, Mandy Simons, and Judith Tonhauser. The Commit-
mentBank: Investigating projection in naturally occurring discourse. Proceedings of
Sinn und Bedeutung, 23(2):107–124, Jul. 2019.
[GMDD07] Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and Bill Dolan. The third PAS-
CAL recognizing textual entailment challenge. In Proceedings of the ACL-PASCAL
Workshop on Textual Entailment and Paraphrasing, pages 1–9, Prague, June 2007.
Association for Computational Linguistics.
[HHDW22] Zhixiong Han, Yaru Hao, Li Dong, and Furu Wei. Prototypical calibration for few-shot
learning of language models, 2022.
[HSD+22] Yaru Hao, Haoyu Song, Li Dong, Shaohan Huang, Zewen Chi, Wenhui Wang, Shum-
ing Ma, and Furu Wei. Language models are general-purpose interfaces. ArXiv,
abs/2206.06336, 2022.
[IG21] Gautier Izacard and Edouard Grave. Leveraging passage retrieval with generative
models for open domain question answering. In Paola Merlo, Jörg Tiedemann, and
Reut Tsarfaty, editors, Proceedings of the 16th Conference of the European Chapter
of the Association for Computational Linguistics: Main Volume, EACL 2021, Online,
April 19 - 23, 2021, pages 874–880. Association for Computational Linguistics, 2021.
[JCWZ17] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. triviaqa: A Large
Scale Distantly Supervised Challenge Dataset for Reading Comprehension. arXiv
e-prints, page arXiv:1705.03551, 2017.
[KKC+22] Junyeob Kim, Hyuhng Joon Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-Woo Lee, Sang
goo Lee, Kang Min Yoo, and Taeuk Kim. Ground-truth labels matter: A deeper look
into input-label demonstrations, 2022.
[KPR+19] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin,
Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: a benchmark for question
answering research. Transactions of the Association of Computational Linguistics,
2019.
[LBM+22] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantas-
tically ordered prompts and where to find them: Overcoming few-shot prompt order
sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland, May
2022. Association for Computational Linguistics.
[LSZ+22] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu
Chen. What makes good in-context examples for GPT-3? In Proceedings of Deep
Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction
and Integration for Deep Learning Architectures, pages 100–114, Dublin, Ireland and
Online, May 2022. Association for Computational Linguistics.
[MCKS18] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor
conduct electricity? a new dataset for open book question answering. In EMNLP,
2018.
[MLH+22] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Ha-
jishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes
in-context learning work? arXiv, 2022.
[RBC+21] Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis, insights from training gopher, 2021. [RHB22] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022, pages 2655–2671. Association for Computational Linguistics, 2022. [RIGS22] Yasaman Razeghi, Robert L Logan IV, Matt Gardner, and Sameer Singh. Impact of pretraining term frequencies on few-shot reasoning. ArXiv, abs/2202.07206, 2022. [RJL18] Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions for SQuAD. In ACL, pages 784–789, 2018. [RZLL16] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas, November 2016. Association for Computational Linguistics. [SAW22] Teven Le Scao, 388 Authors, and Thomas Wolf. BLOOM: A 176B-parameter openaccess multilingual language model. ArXiv, abs/2211.05100, 2022. [SLA+22] Seongjin Shin, Sang-Woo Lee, Hwijeen Ahn, Sungdong Kim, HyoungSeok Kim, Boseop Kim, Kyunghyun Cho, Gichang Lee, Woomyoung Park, Jung-Woo Ha, and Nako Sung. On the effect of pretraining corpora on in-context learning by a large-scale language model. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5168–5186, Seattle, United States, July 2022. Association for Computational Linguistics.
RZLL16] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas, November 2016. Association for Computational Linguistics.
[SAW22] Teven Le Scao, 388 Authors, and Thomas Wolf. BLOOM: A 176B-parameter openaccess multilingual language model. ArXiv, abs/2211.05100, 2022.
[SLA+22] Seongjin Shin, Sang-Woo Lee, Hwijeen Ahn, Sungdong Kim, HyoungSeok Kim, Boseop Kim, Kyunghyun Cho, Gichang Lee, Woomyoung Park, Jung-Woo Ha, and Nako Sung. On the effect of pretraining corpora on in-context learning by a large-scale language model. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5168–5186, Seattle, United States, July 2022. Association for Computational Linguistics.
Dataset
Template
Label Space
Text Classification
SST-2
Sentence: {Sentence}
Negative / Positive
Label: {Label}
SST-5
Sentence: {Sentence}
terrible / bad / neutral / good / great
Label: {Label}
MR
Review: {Sentence}
Negative / Positive
Sentiment: {Label}
Subj
Input: {Sentence}
objective / subjective
Type: {Label}
DBPedia
Input: {Sentence}
Type: {Label}
company / school / artist / athlete / politics
/ transportation / building / nature / village /
animal / plant / album / film / book
AGNews
Classify the news articles into the categories
of World, Sports, Business, and Technology.
World / Sports / Business / Technology
News: {Sentence}
Type: {Label}
TREC
Question: {Sentence}
Type: {Label}
Description / Entity / Expression / Person /
Number / Location
CB
{Premise}
True / False / Neither
Question: {Hypothesis} True, False, or Nei-
ther?
Answer: {Label}
BoolQ
Passage: {Passage}
No / Yes
Question: {Question} Answer:
RTE
Passage: {Premise}
Yes / No
Question: {Hypothesis} Yes or No?
Open-Ended Generation
NQ
Question: {Question} Answer:
Open-ended
WebQS
Question: {Question} Answer:
Open-ended
TriviaQA
Question: {Question} Answer:
Open-ended
SQuAD
Passage: {Passage}
Open-ended
Question: {Question} Answer:
SQuADv2
Passage: {Passage}
Open-ended (The golden label is “none” if the
Question: {Question} Answer:
question is answerable)
Table 6: Prompt template and label mapping for text classification and open-ended generation tasks in our experiments. For multi-choice tasks, we use the same protocol as GPT-3, i.e., choose the best completion of a given prefix.
