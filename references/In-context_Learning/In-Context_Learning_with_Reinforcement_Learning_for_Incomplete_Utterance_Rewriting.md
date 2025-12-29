# In-Context Learning with Reinforcement Learning for Incomplete Utterance Rewriting

Haowei Du, Dongyan Zhao

Abstract

In-context learning (ICL) of large language models (LLMs) has attracted increasing attention in the community where LLMs make predictions only based on instructions augmented with a few examples. Existing example selection methods for ICL utilize sparse or dense retrievers and derive effective performance. However, these methods do not utilize direct feedback of LLM to train the retriever and the examples selected can not necessarily improve the analogy ability of LLM. To tackle this, we propose our policy-based reinforcement learning framework for example selection (RLS), which consists of a language model (LM) selector and an LLM generator. The LM selector encodes the candidate examples into dense representations and selects the top-k examples into the demonstration for LLM. The outputs of LLM are adopted to compute the reward and policy gradient to optimize the LM selector. We conduct experiments on different datasets and significantly outperform existing example selection methods. Moreover, our approach shows advantages over supervised finetuning (SFT) models in few shot setting. Further experiments show the balance of abundance and the similarity with the test case of examples is important for ICL performance of LLM.

# Introduction

In recent years, there has been a growing focus on multi-turn dialogue modeling (Choi et al., 2018; Sun et al., 2019; Reddy et al., 2019). One of the primary challenges in this field is the tendency of speakers to employ incomplete utterances, such as co-reference or ellipsis, when referring back to entities or concepts that have been mentioned in the dialogue history. Su et al. (2019) demonstrates that ellipsis and co-reference occur in over 70% of dialogue utterances. To address this phenomenon, the Incomplete Utterance Rewriting (IUR) task, as proposed by (Pan et al., 2019; Elgohary et al., 2019),

aims to rephrase an incomplete utterance into a selfcontained utterance that is semantically equivalent and can be comprehended independently, without relying on contextual information. Recent generation-based approaches for IUR tackle this task as a seq2seq problem and gain impressive performances (Pan et al., 2019; Huang et al., 2021; Inoue et al., 2022). With the increasing ability of LLMs, ICL has become a new direction for natural language generation, where LLMs make predictions only depending on contexts augmented by a few examples (demonstration) without weights updating (Brown et al., 2020; Chowdhery et al., 2022; Touvron et al., 2023). However, the performance of ICL is sensitive to the selection of in-context examples (Zhao et al., 2021; Liu et al., 2021; Dong et al., 2022). We take a case from TASK dataset in Table 1
and 2, where the incomplete utterance is “How about Mediterranean food?” and the omitted part is the postpositive attributive “in expensive price range”. By metrics of sparse retrieval methods like BM25 (Robertson et al., 2009) or dense retrieval methods like PTM (Liu et al., 2022), the example incomplete utterance e 1 as well as its contexts and rewritten utterance should be selected to be in-context examples. However, the performance of LLM for this question with example e 1 drops by 5 ROUGE score compared with example e 2. To directly select the examples that can improve the analogy ability of LLM, we introduce policy based reinforcement learning (RL) into example selection for IUR (RLS). Given a set of candidate examples, we utilize a small-scale language model (LM) to encode the context and incomplete utterance of each example. Intuitively, if a subset of candidate examples leads to increasing performance of LLM for IUR, RLS should assign high scores to them; the more the performance increases, the higher the example scores should be. Therefore, we compute the reward by ICL performance and

<div style="text-align: center;">Utterance
</div>
Utterance
u1
Hello, I am looking for an expensive
restaurant that serves fusion food.
u2
I ’m sorry, there are no fusion restaurants
listed in the expensive price range. Would
you like to try something else?
u3
How about Mediterranean food?
u∗
3
How about Mediterranean food
in expensive price range?
e1
How about serving Mediterranean food?
e∗
1
How about a cheap restaurant
serving Mediterranean food?
e2
Can you recommend a restaurant to me?
I don’t want to spend a lot of money.
e∗
2
Can you recommend a restaurant to me
in the south part of town? I don’t
want to spend a lot of money.
Table 1: One example from CANARD dataset. u 1 u 2 are 2 turns of contextual utterances, u 3 denotes the incomplete utterance, u ∗ 3 denote the golden rewritten utterance, e 1 /e ∗ 1 and e 2 /e ∗ 2 denote two candidate example pairs of incomplete utterance and rewritten utterance which can be prompted to the LLM.

Example
Sparse
Dense
ROUGE
e1
�
�
50.0
e2
�
�
55.6
Table 2: The Rouge score of ChatGLM by use of different examples in prompts. “Sparse” denotes selecting the example by sparse retrieval methods like BM25, “Dense” denotes selecting the example by dense representations by PTM.

optimize the LM selector with policy gradient.
We conduct experiments on three benchmark datasets in IUR field and compare with existing competitive example selection methods. Our approach outperforms existing methods by about 2 score in CANARD and REWRITE dataset and 10 score in TASK dataset with different metrics including BLEU, ROUGE and F-score.
Our contributions can be summarized as: 1. We are the first to explore ICL performance of LLM for IUR task and design the effective formulation of demonstration for IUR task. 2.  We introduce policy-based RL into example selection for ICL prompts, which directly utilize the LLM feedback to train the LM selector. 3. Our approach significantly outperforms existing

example selection methods across sparse retrieval and dense retrieval methods. 4. Our approach shows advantages against SFT models in few shot setting. We explain the improvement comes from the linguistic complexity and abundance, as well as the similarity with the test case of examples.

# 2 Related Work

There are two main streams of approaches to tackle the task of IUR: edit-based and generationbased. Generation-based models solve this task as a seq2seq problem, which is more relevant to our approach for ICL with LLM. Su et al. (2019) utilize pointer network to respectively predict the prob of tokens in rewritten utterance from contexts or incomplete utterance. Hao et al. (2021) formulate the task as sequence tagging to reduce the search space. Huang et al. (2021) combine a source sequence tagger with an LSTM-based decoder to maintain grammatical correctness. Demonstration selection is crucial to ICL and
Liu et al. (2021) showed that downstream performance can vary widely depending on the choice of in-context examples. Liu et al. (2022) utilize sentence representations of PTM to select the examples with more cosine similarity. Sorensen et al. (2022) and Gonen et al. (2022) argue that mutual information and perplexity are also valuable selection metrics which do not need labeled examples and specific LLM. Levy et al. (2022) select diverse demonstrations to collectively cover all of the structures required in the outputs. Kim et al. (2022) generate demonstrations for ICL from PLM itself to minimize the reliance on the external demonstration. Rubin et al. (2021) first build an unsupervised retriever like BM25 to recall similar examples as candidates and then construct a supervised retriever to select demonstrations from candidates However, these methods fail to directly select the examples into demonstrations that can improve the analogy ability of LLM for IUR task and some useful examples like in table 1 will be neglected.

# 3 Task Definition

Given the context C and the incomplete utterance U, we aim to derive the rewritten version R which can be comprehended without the context. The IUR task is to learn a map function f (C, U | θ) = R. ICL with LLM for IUR task needs a set of candidate examples {y 1, y 2, · · ·, y N}, where y i =

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7493/74937702-c8b9-4a8f-a5a4-10ad09e87a3e.png" style="width: 50%;"></div>
(C i, U i, R i), 1 ≤ i ≤ n denotes the i-th example and N denotes the number of candidate examples. For a test case x = (c x, u x), we select k  examples from the candidate set and the input to LLM M is the concatenation of the task instruction I, the k demonstration examples, and the test case x. The rewritten utterance of x is generated by M: R x = M [I; (C i 1, U i 1, R i 1); (C i 2, U i 2, R i 2); · · ·; (C i k, U i k, R i k); (C i x, U i x)],;  denotes concatenation of texts. So the key of ICL to solve IUR task is to select the appropriate examples.

# 4 Methodology

We approach the example selection as a sequential decision-making problem. State space: the sequence of selected demonstration examples y i 1, y i 2, · · ·, y i k, where k denotes the number of selected examples. Action space: select the next example given the current selection result and insert the concatenation of its contexts, incomplete utterance and rewritten utterance into demonstrations. Policy: We utilize transformer architecture to model the probability of each example in the candidate set to be selected into the demonstration.

# 4.1 Utterance Encoder

We make use of BERT (Devlin et al., 2018) to encode the semantic and syntactic information of utterances. For each example in the candidate set and the current test case, the input to PTM is the concatenation of contexts and incomplete utterance, which is separated by “[SEP]” token. The hidden state corresponding to “[CLS]” token is used to

represent the case.

S i = BERT ([CLS; c i, SEP, u i])
S x = BERT ([CLS; c x, SEP, u x]

(1)
(2)

where c i and u i as well as c x and u x  denote the context and the incomplete utterance of i-th candidate example and the current test case. To be consistent with the inference, we do not include the rewritten utterance into representing each case.

# 4.2 Scoring Policy

With the representation of each case, we model the probability of each candidate case to be selected as in-context examples by metrics of cosine similarity:

(3)

(4)

where ∥. ∥ denotes L2 norm, N denotes the number of candidate examples. We sample k demonstration examples according to the probability computed.

# 4.3 Policy Iterating

Intuitively, if a subset of candidate examples leads to increasing performance of LLM to rewrite the current incomplete utterance, RLS should assign high scores to them; the more the performance increases, the higher the example scores should be. For each case in training set, we input the demonstration formulated as section 3 containing the examples selected by the current policy into the

<div style="text-align: center;">REWRITE
</div>
CANARD
TASK
REWRITE
Language
English
English
Chinese
# Train
32K
2.2K
18K
# Dev
4K
0.5K
2K
# Test
6K
NA
NA
Con. len
85.4
52.6
17.7
Cur. len
7.5
9.4
6.5
Rew. len
11.6
11.3
10.5
Table 3: Statistics of datasets. “Con. len”, “Cur. len”, “Rew. len” denote the length of contexts, incomplete utterance and rewritten utterance respectively.

LLM. The performance metrics of outputs from the LLM against the golden rewritten utterance are utilized as the rewards of the policy.
To maximum the expected rewards of the current policy, we compute the gradient of policy as follows:

where K denotes the number of demonstration examples, n denote the size of training set and R (a i,t, s i,t)  denotes the reward computed by inputing the current examples selected s i,t to LLM for i-th case in training set.
To reduce the generation time cost of LLM, we simplify the computation of � K t =1 R (a i,t, s i,t) by replacing with R (a i,K, s i,K). That is, in the process of selecting examples, we do not input the intermediate demonstrations into LLM to derive the reward. Instead, after selecting all the K examples, we input the final demonstration to LLM. To reduce the variance of learning process, we modify the reward by subtracting a baseline score, which will not affect the gradient. In practice, we compute the ICL performance with randomly selecting examples as the baseline score for IUR task. So the

With the policy gradient computed, we optimize the parameters of policy model defined in section 4.2. We compute the performance on validation set to control the termination of iterating process. In practice, we sample a small subset from the original training dataset to compose the candidate example set C and another disjoint subset to form T to train our RLS algorithm.

# 5 Experiments
5.1 Datasets and LLM

Following Liu et al. (2020), Inoue et al. (2022) and Zhang et al. (2022) we conduct experiments on three benchmark datasets across different languages and domains in IUR field. CANARD dataset (Elgohary et al., 2019) includes English conversational question answering; TASK dataset (Quan et al., 2019) contains task-oriented English dialogues; REWRITE dataset (Su et al., 2019) is composed of open-domain Chinese dialogues. We choose ChatGLM-6B 1 as the LLM f M which does not conduct parameter updating. ChatGLM is a bilingual large language model pretrained by supervised finetuning, instruction tuning and human feedback reinforcement learning, which is suitable for our bilingual datasets.

# 5.2 Metrics

Following Liu et al. (2020); Inoue et al. (2022); Zhang et al. (2022) , we utilize BLEU (Papineni et al., 2002), ROUGE (Lin, 2004) and F-score to evaluate the IUR task. BLEU and ROUGE focus on the overall quality of the rewritten utterance and F-score concentrate more on words from the context (important words) which are argued to be harder to copy (Pan et al., 2019; Inoue et al., 2022).

# 5.3 Baselines

We compare our approaches with competitive example selection methods as follows:
1 https://github.com/THUDM/ChatGLM-6B

Model
ROUGE
BLEU
F-score
RL
R1
R2
B1
B2
B3
B4
F1
F2
F3
Random
52.41
54.02
38.74
48.40
40.80
35.08
29.70
25.03
17.65
14.26
BM25
53.92
56.16
40.14
51.03
43.05
37.03
31.44
30.00
20.19
15.62
KATE
53.00
55.24
39.61
49.31
41.60
35.82
30.47
28.99
19.51
15.23
EPR
54.08
56.26
40.23
51.59
43.54
37.51
31.91
29.59
19.96
15.62
BSR
54.28
56.39
40.50
51.99
44.00
38.01
32.44
29.61
20.15
15.90
Ours
55.69
57.55
42.22
53.29
45.55
39.65
34.05
29.52
20.59
16.46
<div style="text-align: center;">ROUGE
</div>
4: ICL Evaluations of ChatGLM with 5-shot demonstrations on CANARD dataset. Our approach significantl forms existing example selection methods, where p-values of ROUGE, BLEU and F-score are smaller tha

Model
TASK
REWRITE
ROUGE
BLEU
ROUGE
BLEU
RL
R1
R2
B1
B2
B3
RL
R1
R2
B1
B2
B3
Random
44.5
45.8
31.8
34.1
28.6
25.3
64.4
66.8
54.0
61.2
55.3
49.7
BM25
46.2
47.5
34.1
33.1
27.9
24.8
64.6
67.0
55.6
65.7
60.1
54.7
KATE
45.8
47.3
34.4
33.1
28.0
25.0
63.2
65.2
53.2
61.6
55.9
50.2
EPR
49.2
50.4
37.0
35.4
30.1
27.0
65.8
68.1
56.4
66.7
61.1
55.8
BSR
49.7
51.6
37.8
35.2
29.6
26.4
65.3
68.0
56.3
65.8
60.1
54.7
Ours
57.2
57.9
45.1
43.0
38.3
35.2
66.5
68.3
56.8
67.2
61.7
56.4
Table 5: ICL Evaluations of ChatGLM with 5-shot demonstrations on Task and REWRITE dataset. significantly outperforms existing example selection methods, where p-values of ROUGE, BLEU a smaller than 0.001.

5: ICL Evaluations of ChatGLM with 5-shot demonstrations on Task and REWRITE dataset. Our approach cantly outperforms existing example selection methods, where p-values of ROUGE, BLEU and F-score are

Random In this baseline, we randomly select the examples from candidate set to formulate the demonstration.

BM25 (Robertson et al., 2009) In this baseline, first we utilize SpaCy NLP tools (Vasiliev, 2020) to stem words in the context and incomplete utterance of each case. Then BM25 method is adopted to estimate the relevance of examples to a given test case. The top-k relevant examples are selected to formulate the demonstration. It belongs to the sparse retrieval methods.
KATE Liu et al. (2022) make use of SBERT (Reimers and Gurevych, 2019) to select examples which are semantically similar to the test sample and build a kNN-based unsupervised retriever. It belongs to the dense retrieval methods.
EPR Rubin et al. (2021) assume the unsupervised retriever can act as the guide to the LM retriever and propose a two-stage approach. It first builds an unsupervised retriever (e.g., BM25) to recall surface similar examples as candidates and then constructs a supervised retriever EPR to select demonstrations from candidates. It can be seen as a combination of sparse and dense retrieval methods.

EPR Rubin et al. (2021) assume the unsupervised retriever can act as the guide to the LM retriever and propose a two-stage approach. It first builds an unsupervised retriever (e.g., BM25) to recall surface similar examples as candidates and then constructs a supervised retriever EPR to select demonstrations from candidates. It can be seen as a combination of sparse and dense retrieval methods.

BSR Gupta et al. (2023) propose a novel framework for selecting sets of maximally informative demonstrations for the salient aspects of the test input, e.g., reasoning patterns, entities, etc. Examples selected using this framework are informative about the test input and help the LLM understand and perform the task However, these methods fail to directly utilize the feedback by LLM and the examples selected can not necessarily improve the analogy ability of LLM.

# 5.4 Experimental Details

Following Liu et al. (2022) and Rubin et al. (2021) we utilize SentenceBERT (Reimers and Gurevych, 2019) as LM selector for English datasets and bertbase-Chinese for Chinese datasets. The sizes of candidate example set C and training set T are 500. The learning rate is set to be 1e-5. The task instruction in section 3 is designed as “Rewrite an incomplete utterance into an utterance which is semantically equivalent but self-contained to be understood without context. The sentence structure and expression should be consistent.” for English dataset and its translated version for Chinese

Model
ROUGE
BLEU
F-score
RL
R1
R2
B1
B2
B3
B4
F1
F2
F3
Ours (100-100)
54.22
56.13
40.86
51.08
43.49
37.74
32.30
28.60
20.02
16.08
Ours (100-500)
55.08
56.51
41.81
51.77
44.62
39.06
33.73
27.17
20.18
16.59
Ours (500-100)
54.12
56.29
40.73
51.49
43.65
37.74
32.22
29.82
20.29
15.90
Ours (500-500)
55.69
57.55
42.22
53.29
45.55
39.65
34.05
30.52
20.59
16.46
<div style="text-align: center;">ons of ChatGLM with different sizes of candidates and training samples on CANARD dataset. s experiments with 500 candidates and 100 training samples, and so on.
</div>
Model
ROUGE
BLEU
F-score
RL
R1
R2
B1
B2
B3
B4
F1
F2
F3
EPR-3
53.91
55.92
40.14
51.44
43.47
37.45
31.86
29.15
20.05
15.83
Ours-3
54.90
56.01
41.22
49.45
42.80
37.57
32.53
29.60
20.15
16.03
EPR-4
54.12
56.27
40.45
51.36
43.40
37.37
31.76
29.64
20.01
15.59
Ours-4
55.11
57.29
41.57
52.71
44.81
38.84
33.37
30.92
21.12
16.75
EPR-5
54.08
56.26
40.23
51.59
43.54
37.51
31.91
29.59
19.96
15.62
Ours-5
55.69
57.55
42.22
53.29
45.55
39.65
34.05
29.52
20.59
16.46
Model
F1
F2
F3
QUEEN
20.33
13.25
11.59
Ours
29.52
20.59
16.46
Table 8: Comparing with SFT model in few-shot setting.

dataset.

# 5.5 Results

In Table 4, our approach outperforms all the baselines by about 1.2-1.7 ROUGE score, 1.3-2.5 BLEU score, 0.4 F2 score and 0.6 F3 score in CANARD dataset. It demonstrates the efficiency of directly utilizing feedback by LLM and RL policy gradient to train the LM selector. With the examples selected by our method, the performances of LLM are significantly improved. Our model not only improves the overall quality of utterance rewritten, but also captures the important words from the context. Compared with Random selecting examples, both BM25 and KATE baselines derive better performance. It shows the textual similarity captured by sparse retrieval like BM25 and the semantic similarity captured by KATE can help select better examples to prompt the LLM for IUR task. EPR and BSR show an overall better performance compared with the sole sparse or dense retrieval methods. It demonstrates the combination of sparse or dense retrieval can capture both the textual and semantic

<div style="text-align: center;">BLEU
</div>
similarity between candidate examples and the test case. The examples selected are better prompts to LLM for IUR task. Compared with EPR, BSR behaves a better performance across different metrics. BM25 behaves as the effective supervision for the LM retriever and releasing the constraint can improve the example selection furthermore.
In TASK dataset, our approach outperforms all the baselines by about 6.3-7.5 ROUGE score and 7.6-8.2 BLEU score. Compared with CANARD dataset, TASK contains more complex and diverse dialogue topics. It demonstrates the efficiency of utilizing direct LLM feedback to train the LM retriever and select examples to prompt the LLM for IUR task. In REWRTTE dataset, our approach outperforms all the baselines by about 0.6 ROUGE and BLEU score, which shows our stable efficiency with different languages.

# 6 Analysis

In this part, first we explore our performance with different sizes of candidate set and training set. Then we probe the effect of different numbers of examples in demonstrations. Furthermore, we compare our approach with the SFT method in few shot setting. Finally, we explore the reason why examples selected by our approach can improve the analogy ability of LLM.

Model
Incomplete
Rewritten
Length
POS
Chunk
Length
POS
Chunk
BSR
8.55
6.29
2.33
11.91
7.78
3.45
Ours
10.81
6.99
3.23
13.36
7.95
4.03
9: Complexity and abundance of example selected. “Incomplete” denotes the metrics of incomplete utter Rewritten” denotes the metrics of rewritten utterance, “Length”, “POS”, and “Chunk” denote utterance length r of POS types and number of text chunks respectively.

Model
ROUGE
BLEU
F-score
RL
R1
R2
B1
B2
B3
B4
F1
F2
F3
Length
50.84
52.13
37.17
46.76
39.41
33.89
28.67
19.95
14.46
11.99
POS
52.17
53.77
38.69
47.92
40.44
34.80
29.46
23.89
17.09
13.92
Chunk
53.68
55.20
40.34
52.09
44.68
38.86
33.29
24.70
18.15
14.90
Ours
55.69
57.55
42.22
53.29
45.55
39.65
34.05
29.52
20.59
16.46
Table 10: ICL Evaluations of ChatGLM with 5-shot demonstrations on CANARD dataset.

# 6.1 Different Sizes of Candidates and Training Samples

In Table 4, the number of candidates # C  and training samples # T  are 500. In this part, we do further experiments with # C = 100, # T = 100; # C = 100, # T = 500; # C = 500, # T = 100 and freeze other hyperparameters respectively. In Table 6, generally, with more candidates and training samples, our approach will select better examples for the demonstration. With 100 candidates and 100 training samples, our approach beats random selection by about 2.0 ROUGE score, 2.5 BLEU score and 2.0 F-score. It is also comparable to the competitive baseline BSR with 500 candidates and training samples. It shows the efficiency to utilize direct LLM feedback to train the LM retriever. Compared with setting 100-500, our approach outperforms by about 0.7 ROUGE score, 0.8 BLEU score and 0.9 F-score. It demonstrates our ability to select better examples to improve the analogy ability of LLM With more candidates. Compared with setting 500-100, our approach outperforms by about 1.4 ROUGE score, 1.9 BLEU score and 0.2 F-score. It shows more training samples improve the selection of our approach from the candidate set for IUR task.

# 6.2 Different Number of Examples in Demonstration

In Table 4, we conduct the experiments with 5shot demonstrations. In this part, we do further experiments with 3-shot and 4-shot demonstrations. In Table 7, generally with more examples in the demonstration, our approach improves the

ICL performance of LLM for IUR task. Especially, with more demonstration examples, our approach derives more improvement compared with the competitive baseline EPR. It demonstrates the efficiency of our RLS by directly utilizing LLM feedback to train the LM retriever and improve the ICL performance of LLM.

# 6.3 Comparing with SFT Model

In this part, we compare our approach with the existing state-of-the-art QUEEN (Liu et al., 2020) in IUR field. QUEEN tackles IUR task by finetuning PTM (Devlin et al., 2018) and constructing the word edit matrix. Different from QUEEN , our approach utilizes LM as a proxy to select appropriate examples and parameters of the answer generator ChatGLM are fixed. To keep a fair comparison, we assign the candidate set and training set in our approach as the training set of QUEEN. In Table 8, our approach QUEEN by 9.2 F1 score, 7.3 F2 score and 4.9 F3 score. F-score concentrate on the words from the context, which are argued to be harder to copy (Pan et al., 2019). It shows our efficiency to capture important words from the context to rewrite the incomplete utterance. Considering our RLS approach does not depend on the LLM server (ChatGLM in our experiments), it is promising for our approach to derive better results for IUR task with stronger LLM.

# 6.4 What Examples are Appropriate for IUR

In this part, we explore the reason why examples selected by our approach serve as better demonstrations for the LLM to solve IUR task.

Model
ROUGE
BLEU
F-score
RL
R1
R2
B1
B2
B3
B4
F1
F2
F3
Random
78.07
79.34
66.50
69.05
64.15
60.26
56.35
52.86
43.49
38.14
BSR
80.13
81.24
68.78
73.54
67.37
65.42
59.78
56.16
48.36
44.89
Ours
82.77
84.04
70.37
79.07
73.81
69.56
65.47
62.64
52.61
47.30
<div style="text-align: center;">ROUGE
</div>
Table 11: ICL Evaluations of gpt3.5 with 5-shot demonstrations on TASK dataset.

Model
ROUGE
BLEU
R1
R2
B1
B2
B3
BSR
51.6
37.8
35.2
29.6
26.4
Ours
57.9
45.1
43.0
38.3
35.2
Ours(r)
57.6
44.5
43.3
38.6
34.9
Table 12: ICL Evaluations of example orders with ChatGLM on TASK dataset.

We assume that other than the textual and semantic similarity of examples with the current test case, the complexity and abundance of examples in the demonstrations also matter for IUR task. We evaluate the complexity and abundance of examples by three metrics of incomplete utterance and rewritten utterance: 1. the length of utterance; 2. the number of Part Of Speech (POS) tagging types (Kumawat and Jain, 2015). 3. the number of text chunks (Ramshaw and Marcus, 1999). We assume the examples are more complex and abundant with longer utterances, more POS tag types and more text chunks. In practice, we adopt SpaCy to do POS tagging and text chunking. In Table 9, the examples selected by our policy-based RL framework have longer utterances, more POS types and more text chunks though without explicit assignment, We argue that the complexity and abundance of examples in the demonstration are important to improve the analogy ability of LLM. What if we select the examples only by metrics of complexity and abundance? To address this issue, we select examples from the candidate set to construct the demonstration by metrics of the length of utterance, the number of POS types, the number of text chunks respectively and conduct the experiments on CANARD dataset. In Table
10, if selecting the examples only by the metric of number of text chunks, the ICL results will drop by about 2.1 ROUGE score, 0.9 BLEU score, 2.9 F-score on CANARD dataset. It will drop more if selecting the examples only by the metric of utterance length or POS types. It shows only selecting examples by complexity and abundance, the LLM

performance will not necessarily improve. Balancing the abundance and the similarity with the test case of examples is important to improve the ICL ability of LLM. Our approach attains the balance of complexity and abundance, as well as semantic similarity with the test case without explicit assignment.

# 6.5 Efficiency of Larger LLM

In this part, we conduct the experiments by applying our approach with gpt3.5 in TASK dataset. In Table 11, our RLS outperforms random selecting examples by about 4.4 ROUGE score, 9.5 BLEU score and 9.4 F-score in TASK dataset. It demonstrates the efficiency of directly utilizing LLM feedback to train the LM retriever with policy-based RL and improve the analogy ability of larger LLM. With the emergence of larger LLM, it is promising for our approach to select appropriate examples to improve the ICL performance of LLM.

# 6.6 Order of Examples

In this part, we probe the effect of the order of examples in the demonstration. We arrange examples in the demonstration by the order of sampling with Eq. 3  or the reverse order and conduct the experiments on TASK dataset. Table 12  shows if arranging the examples by the reverse order of sampling, our approach shows comparable performance . It demonstrates the stable efficiency of our directly selecting the examples that can improve the analogy ability of LLM by policy gradient.

# 7 Conclusion

Existing example selection methods fail to directly utilize the LLM feedback to choose appropriate examples in the demonstration. We propose a novel and effective example selection framework by directly adopting the LLM feedback to train the LM selector with policy-based RL. Our approach significantly improves the ICL performance of LLM for IUR tasks.

we propose our novel and effective example selection framework by directly adopting the LLM feedback to train the LM selector with policy-based RL and demonstrate our efficiency on three benchmark datasets in this field. Though only utilizing the LLM ChatGLM-7B and gpt3.5, we will conduct experiments with diverse kinds of LLMs in future research.

# References

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.
Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wentau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. 2018. Quac: Question answering in context. arXiv preprint arXiv:1808.07036.
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2022. A survey for in-context learning. arXiv preprint arXiv:2301.00234.
Ahmed Elgohary, Denis Peskov, and Jordan BoydGraber. 2019. Can you unpack that? learning to rewrite questions-in-context. Can You Unpack That? Learning to Rewrite Questions-in-Context.
Hila Gonen, Srini Iyer, Terra Blevins, Noah A Smith, and Luke Zettlemoyer. 2022. Demystifying prompts in language models via perplexity estimation. arXiv preprint arXiv:2212.04037.
Shivanshu Gupta, Sameer Singh, and Matt Gardner. 2023. Coverage-based example selection for incontext learning. arXiv preprint arXiv:2305.14907.
Jie Hao, Linfeng Song, Liwei Wang, Kun Xu, Zhaopeng Tu, and Dong Yu. 2021. Rast: Domain-robust dialogue rewriting as sequence tagging. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4913–4924.

Mengzuo Huang, Feng Li, Wuhe Zou, and Weidong Zhang. 2021. Sarg: A novel semi autoregressive generator for multi-turn incomplete utterance restoration. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 13055–13063.
Shumpei Inoue, Tsungwei Liu, Nguyen Hong Son, and Minh-Tien Nguyen. 2022. Enhance incomplete utterance restoration by joint learning token extraction and text generation. arXiv preprint arXiv:2204.03958.
Hyuhng Joon Kim, Hyunsoo Cho, Junyeob Kim, Taeuk Kim, Kang Min Yoo, and Sang-goo Lee. 2022. Self-generated in-context learning: Leveraging autoregressive language models as a demonstration generator. arXiv preprint arXiv:2206.08082.
Deepika Kumawat and Vinesh Jain. 2015. Pos tagging approaches: A comparison. International Journal of Computer Applications, 118(6).
Itay Levy, Ben Bogin, and Jonathan Berant. 2022. Diverse demonstrations improve in-context compositional generalization. arXiv preprint arXiv:2212.06800.
Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.
Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. 2022. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning.  Advances in Neural Information Processing Systems, 35:1950–1965.
Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2021. What makes good in-context examples for gpt3? arXiv preprint arXiv:2101.06804.
Qian Liu, Bei Chen, Jian-Guang Lou, Bin Zhou, and Dongmei Zhang. 2020. Incomplete utterance rewriting as semantic segmentation. arXiv preprint arXiv:2009.13166.
Zhufeng Pan, Kun Bai, Yan Wang, Lianqiang Zhou, and Xiaojiang Liu. 2019. Improving open-domain dialogue systems via multi-turn incomplete utterance restoration. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1824–1833.
Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318.
Jun Quan, Deyi Xiong, Bonnie Webber, and Changjian Hu. 2019. Gecor: An end-to-end generative ellipsis and co-reference resolution model for task-oriented dialogue. arXiv preprint arXiv:1909.12086.

Lance A Ramshaw and Mitchell P Marcus. 1999. Text chunking using transformation-based learning.  Natural language processing using very large corpora, pages 157–176.
Siva Reddy, Danqi Chen, and Christopher D Manning. 2019. Coqa: A conversational question answering challenge.  Transactions of the Association for Computational Linguistics, 7:249–266.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.
Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: Bm25 and beyond.  Foundations and Trends® in Information Retrieval, 3(4):333–389.
Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2021. Learning to retrieve prompts for in-context learning. arXiv preprint arXiv:2112.08633.
Taylor Sorensen, Joshua Robinson, Christopher Michael Rytting, Alexander Glenn Shaw, Kyle Jeffrey Rogers, Alexia Pauline Delorey, Mahmoud Khalil, Nancy Fulda, and David Wingate. 2022. An information-theoretic approach to prompt engineering without ground truth labels. arXiv preprint arXiv:2203.11364.
Hui Su, Xiaoyu Shen, Rongzhi Zhang, Fei Sun, Pengwei Hu, Cheng Niu, and Jie Zhou. 2019. Improving multi-turn dialogue modelling with utterance rewriter. arXiv preprint arXiv:1906.07004.
Kai Sun, Dian Yu, Jianshu Chen, Dong Yu, Yejin Choi, and Claire Cardie. 2019. Dream: A challenge data set and models for dialogue-based reading comprehension.  Transactions of the Association for Computational Linguistics, 7:217–231.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.
Yuli Vasiliev. 2020. Natural language processing with Python and spaCy: A practical introduction. No Starch Press.
Yong Zhang, Zhitao Li, Jianzong Wang, Ning Cheng, and Jing Xiao. 2022. Self-attention for incomplete utterance rewriting. In  ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8047–8051. IEEE.
Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In  International Conference on Machine Learning, pages 12697–12706. PMLR.
A Example Appendix

Lance A Ramshaw and Mitchell P Marcus. 1999. Text chunking using transformation-based learning.  Natural language processing using very large corpora, pages 157–176.
Siva Reddy, Danqi Chen, and Christopher D Manning. 2019. Coqa: A conversational question answering challenge.  Transactions of the Association for Computational Linguistics, 7:249–266.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.
Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: Bm25 and beyond.  Foundations and Trends® in Information Retrieval, 3(4):333–389.
Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2021. Learning to retrieve prompts for in-context learning. arXiv preprint arXiv:2112.08633.
Taylor Sorensen, Joshua Robinson, Christopher Michael Rytting, Alexander Glenn Shaw, Kyle Jeffrey Rogers, Alexia Pauline Delorey, Mahmoud Khalil, Nancy Fulda, and David Wingate. 2022. An information-theoretic approach to prompt engineering without ground truth labels. arXiv preprint arXiv:2203.11364.
Hui Su, Xiaoyu Shen, Rongzhi Zhang, Fei Sun, Pengwei Hu, Cheng Niu, and Jie Zhou. 2019. Improving multi-turn dialogue modelling with utterance rewriter. arXiv preprint arXiv:1906.07004.
Kai Sun, Dian Yu, Jianshu Chen, Dong Yu, Yejin Choi, and Claire Cardie. 2019. Dream: A challenge data set and models for dialogue-based reading comprehension.  Transactions of the Association for Computational Linguistics, 7:217–231.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.
Yuli Vasiliev. 2020. Natural language processing with Python and spaCy: A practical introduction. No Starch Press.
Yong Zhang, Zhitao Li, Jianzong Wang, Ning Cheng, and Jing Xiao. 2022. Self-attention for incomplete utterance rewriting. In  ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8047–8051. IEEE.
Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In  International Conference on Machine Learning, pages 12697–12706. PMLR.
A Example Appendix

