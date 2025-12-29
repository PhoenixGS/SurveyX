# I N-C ONTEXT L EARNING D EMONSTRATION S ELECTION VIA I NFLUENCE A NALYSIS

A PREPRINT
Vinay M.S.∗
University of Arkansas
Fayetteville, AR 72701, USA
vmadanbh@uark.edu
Minh-Hao Van∗
University of Arkansas
Fayetteville, AR 72701, USA
haovan@uark.edu
Xintao Wu
University of Arkansas
Fayetteville, AR 72701, USA
xintaowu@uark.edu
# A BSTRACT

Large Language Models (LLMs) have showcased their In-Context Learning (ICL) capabilities, enabling few-shot learning without the need for gradient updates. Despite its advantages, the effectiveness of ICL heavily depends on the choice of demonstrations. Selecting the most effective demonstrations for ICL remains a significant research challenge. To tackle this issue, we propose a demonstration selection method named InfICL, which utilizes influence functions to analyze impacts of training samples. By identifying the most influential training samples as demonstrations, InfICL aims to enhance the ICL generalization performance. To keep InfICL cost-effective, we only use the LLM to generate sample input embeddings, avoiding expensive fine-tuning. Through empirical studies on various real-world datasets, we demonstrate advantages of InfICL compared to state-ofthe-art baselines.

Keywords large language models · in-context learning · demonstration selection · influence functions

# 1 Introduction

Large Language Models (LLMs) have demonstrated their ability to perform few-shot inference through In-Context Learning (ICL) [Brown et al., 2020]. Specifically, by providing a few demonstrations for the given task, the LLM is able to perform test case inference without performing any model gradient update.
ICL has several benefits such as few-shot learning, avoiding model fine-tuning, and versatility to different learning tasks. Despite these benefits, the ICL performance is sensitive to the selected demonstrations. To address this limitation, many different approaches have been proposed for demonstration selection, e.g., selecting demonstrations which are similar to the test case in the embedding space [Gao et al., 2021, Liu et al., 2022, Wu et al., 2023, Qin et al., 2023, Yang et al., 2022], learning a deep learning-based demonstration retriever [Rubin et al., 2022, Luo et al., 2023, Chen et al., 2020, Karpukhin et al., 2020, Scarlatos and Lan, 2023, Zhang et al., 2022, Li et al., 2023], selecting demonstrations based on LLM feedback [Li and Qiu, 2023, Chen et al., 2023b, Wang et al., 2023], etc. However, there is a lack of consensus regarding the most effective demonstration selection approach [Nguyen and Wong, 2023]. The current research challenge is to identify those demonstrations which are the most effective or influential for improving the ICL generalization performance. We address this challenge by employing influence functions [Koh and Liang, 2017]. Specifically, influence functions provide mechanisms to analyze effects or influences of training samples on the model without retraining the model. For example, influence functions can be used to analyze the model effects after up-weighting or removing a training sample. The training samples which have higher influences naturally provide more contributions to the model learning process. Intuitively, identifying these influential training samples can aid in improving the ICL generalization performance.
In this work, we focus on the text classification problem, and propose an influence function analysis-based demonstration selection method called InfICL. Since we need to perform influence function analysis on the training samples, an obvious approach is to calculate these influence scores by using the LLM itself [Grosse et al., 2023]. However, for

* These authors contributed equally to this work.

large and complex deep learning models, the influence function analysis becomes erroneous [Basu et al., 2021]. An other approach is to fine tune the final layers of the LLM and perform influence function analysis by using these fina layers. However, fine tuning LLM is a highly resource intensive task. To address these practical challenges, we only employ the LLM to generate sample embeddings. By employing these LLM generated training sample embeddings we train a simple classifier. We analyze the influence of each training sample by using the classifier and a validation set. Finally, we select the most influential training samples from each class as the demonstration set. We summarize our main contributions below.

• We propose a ICL demonstration selection method called InfICL which is based on influence function analysis.
• We present a running cost analysis study and compare our InfICL to other advanced influence analysisbased demonstration selection methods [Nguyen and Wong, 2023, Chang and Jia, 2023]. In particular, we demonstrate that these contemporary methods require an exceedingly high number of LLM access calls in comparison to our InfICL.
• We present an empirical study conducted on multiple real-world datasets and four LLMs of varying sizes. In this empirical study, we show that our InfICL can outperform the contemporary demonstration selection methods.

# 2 Related Work

Our work mainly focuses on designing an demonstration selection method for ICL through influence analysis.
Demonstration Selection.  Recently, the problem of demonstration selection for ICL has received a significant attention in the literature. We direct the interested readers to [Liu et al., 2021, Dong et al., 2023] for detailed surveys regrading different demonstration selection methods. One of the popular approaches for demonstration selection is to select those training samples as demonstrations which are similar to the test sample in the embedding space [Gao et al., 2021, Liu et al., 2022, Wu et al., 2023, Qin et al., 2023, Yang et al., 2022].
Another popular approach is to employ a demonstration retriever to perform demonstration selection. Specifically, the demonstration retriever is a deep learning based model. Rubin et al. [2022] and Luo et al. [2023] train their demonstration retriever by employing contrastive loss [Chen et al., 2020]. Li et al. [2023] employ in-batch negative loss [Karpukhin et al., 2020]. Scarlatos and Lan [2023] and Zhang et al. [2022] employ reinforcement learning to train their demonstration retriever. In our work, we do not utilize any complex demonstration retriever, and design a simple method which operates on LLM embeddings.
Recently, LLM feedback based demonstration selection methods have been proposed. Specifically, the LLM is queried for its prediction confidence on each training sample. Li and Qiu [2023] identify training samples which are more informative. Chen et al. [2023b] select training points which are less sensitive to predictions. Wang et al. [2023] fine tune the LLM by using only the final emdedding layer and model the demonstration selection as a topic model. These methods can also be considered as influence based methods because they analyze the influence of training samples by using direct LLM feedback.
Influence Functions. For machine learning applications, influence functions have been used for different tasks, e.g., filtering or relabeling mislabeled training data [Kong et al., 2022], designing data poisoning attacks [Fang et al., 2020, Jagielski et al., 2021], designing data augmentation strategies [Lee et al., 2020, Oh et al., 2021], and analyzing label memorization effects [Feldman and Zhang, 2020]. For LLMs, influence functions have been used to identify data artifacts [Han et al., 2020], identify biases in word embeddings [Brunet et al., 2019], and explaining the LLM performance [Grosse et al., 2023, Han and Tsvetkov, 2021].
Influence analysis can be broadly divided into two categories: retraining based [Ilyas et al., 2022] and gradient based methods also called as influence functions [Koh and Liang, 2017]. The retraining based methods collect random subsets of the training set. Then, the influence of each training sample in the collected subset is calculated by either model retraining or by learning a linear surrogate. However, the retraining based methods have high running costs, and are not scalable to large datasets because to effectively cover all the training samples, a large number of subsets have to be constructed and evaluated [Grosse et al., 2023]. Nguyen and Wong [2023] and Chang and Jia [2023] employ retraining based influence analysis to construct the demonstration sets and as a result, their proposed demonstration selection methods incur high running costs. We provide a detailed design description about these demonstration selection methods and compare their running costs against our InfICL in Section 3.2. Specifically, we show that by using the gradient based influence analysis for constructing demonstration sets, we can overcome the high running cost challenge associated with the retraining based influence analysis methods.

# 3 Proposed Method

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7a54/7a54a1e8-eda0-496e-9d97-459dd326bc2b.png" style="width: 50%;"></div>
Figure 1: Illustration of ICL for the text classification task through our InfICL. Initially, by employing the local LLM, embeddings for all the training and validation set inputs are generated. A local classifier is then trained by employing training input embeddings and labels. InfICL determines K demonstration examples based on influence scores. Finally, the demonstration set and each test case are sent to an external LLM for inference.
We consider the text classification task having a training set T with n training points denoted as z i = {(x i, y i)} n i =1. Here, x i and y i denote the embedding vector for the i th training sample input s i  and its corresponding label, respectively. Let C denote the class set for the target variable y i and y i ∈C. We employ a validation set denoted as V.

# 3.1 Algorithm

Figure 1 shows our influence analysis based demonstration selection method. We employ separate LLMs for demonstration selection and test case inference called local LLM P and external LLM Q, respectively. For local LLM P, to reduce training costs, we employ a light-weight LLM and use it to generate embeddings for the input texts. Let E P denote the embedding layer of P which generates the sample embeddings. Here, x i = E P (s i, ϕ), where parameter ϕ ∈ Φ, and Φ denotes the local LLM (P) parameter space. We denote L nt (s i, ϕ) as the next token prediction loss for P. For external LLM Q, we opt a powerful and heavier LLM. We include a local classifier denoted as F (x i, θ) with the input of embeddings and parameterized with θ ∈ Θ, and Θ denotes the classifier (F) parameter space. We denote L f (z i, θ) as the classifier training loss.
Our goal is to select K suitable demonstrations for the given text classification task. Note that K is analogous to the number of shots in few-shot learning and is constrained by the employed external LLM. We employ a balanced selection approach wherein we select equal number of demonstrations from each class c ∈C. Specifically, we select R (R = ⌊ K/ |C|⌋) suitable training set points from each class as demonstrations.
Algorithm 1 shows the pseudo code of our InfICL. The inputs include training set T, validation set V, classifier F, loss L f, the number of demonstration examples per class R, and local LLM P. Initially, by employing the local LLM P, we generate embeddings for all training and validation inputs. In lines 2-4, we train the local classifer F using the embeddings and labels. Next, we calculate influence score of each training point (lines 5-7). For each class c ∈C, we select the topR training points {z c i} R i =1 as demonstrations from T based on influence scores (lines 8-10). Finally, we return the constructed demonstration set ∪ c ∈C {z c i} R i =1.

Algorithm 1 InfICL demonstration selection.
Inputs: T , V, F, Lf, R, and P.
Output: demonstration set ∪c∈C{zc
i }R
i=1.
1: generate embeddings for all training and validation inputs through P;
2: for each training epoch do
3:
train classifier F on T by using Lf;
4: for each zi = (xi, yi) ∈T do
5:
calculate its influence score by using Eq 1;
6: for each class c ∈C do
7:
select top-R training points {zc
i }R
i=1 from T based on influence scores;
8: return ∪c∈C{zc
i }R
i=1
Influence Functions. The main goal of the influence functions is to study the effect of training points on model prediction [Koh and Liang, 2017]. Influence functions provide a practical solution wherein, the model parameter change can be studied without retraining the model. Let 1
n � n i =1 L f (z i, θ). It is assumed that the empirical risk is twice differentiable and strictly convex. However, this assumption can be practically relaxed. The influence of up-weighting training point z on the classifier parameter θ can be calculated by using the influence function as
n � n i =1 L f (z i, θ) be the empirical risk and its minimizer is given by � θ = arg min θ ∈ Θ 1



�
� �
Specifically, the highly influential training points are those with most positive (−I up,loss (z, V)) scores [Koh and Liang, 2017]. We employ Inf (z, V) = −I up,loss (z, V) as the influence score to analyze the influence of up-weighting each training point z on the loss L f at V. This is because, the training points which have high influences on the validation loss provide richer information for model learning, and can become better demonstrations for the ICL task.
Personalized Demonstration Selection. We can easily extend InfICL to construct a personalized demonstration set for each test case x test. Specifically, we can extend InfICL to this setting by scoring each training point z i ∈T as
where

score (z i) = λInf (z i, V) + (1 − λ) sim (x i, x test) (2) wh
sim (·, ·) denotes the cosine similarity between the input embeddings, and λ is the weight which can be set analyzing the accuracy performance on the validation set. The topR training points from each class based score (z i) are included in the demonstration set.

# 3.2 Running Cost Analysis

In this section, we study the running costs of our InfICL along with other influence analysis based demonstration selection methods, Influence [Nguyen and Wong, 2023] and Curation [Chang and Jia, 2023]. Note that both methods employ retraining based influence analysis approach and our InfICL employs gradient based influence analysis approach. We show running cost benefits of our InfICL over Influence and Curation.
We quantify the running costs of demonstration selection methods by analyzing the total number of LLM access (API) calls for both local and external LLMs. Specifically, the unit cost of local LLM (P) access call for generating an embedding for a single training point is denoted as C P. Similarly, the unit cost of external LLM (Q) access call for performing inference on a single test or validation case is denoted as C Q. Note that C Q is usually much higher than C P. This is because C Q involves ICL cost w.r.t external LLM and C P only generates the final layer embeddings which

where

(2)

forward pass cost. We show the running costs of different influence analysis based demonstration in Table 1 and provide a detailed description below.

Methods
Influence [Nguyen and Wong, 2023]
Curation [Chang and Jia, 2023]
InfICL
CondAcc
Data Models
Running Cost
O (CQ|V|M)
O (CQ|V|MK!)
O (CQ|V|M)
O (CP|T | + CQ)
InfICL. We generate embeddings for all training points and generating embedding for each training point requires a single local LLM access call. Thus, the total cost of local LLM access calls for embedding generation is C P |T |. For the test case inference, we require a single external LLM access call and the cost is C Q. Thus, the the running cost of our demonstration selection method is given by O (C P |T | + C Q). For influence estimation, we use a fully connected neural network as the backbone architecture for the classifier F. Let d be the number of parameters in F. Calculating the loss of |T | training samples takes O (d |T |). In the implementation, we use LiSSA Agarwal et al.
[2017] method to approximate the inverse Hessian-Vector product (iHVP) of � 1
|V| �
z j ∈V ∇ θ L f � z j, � θ � � ⊤ H − 1 � θ,
which costs O (|V| d + rjd) where r is the recursion depth and j is the number of repeats. As both validation set and θ are fixed, there is only one computation of iHVP. The sorting time needed for ranking potential demonstrations by influence analysis is O (|T | log(|T |)) on average. Consequently, the influence estimation process takes O (d |T | + |V| d + rjd + |T | log(|T |)). In a practical setting, |V| is sufficiently small compared to |T | (|V| ≪|T |) and rj ≈|T |. Therefore, the running time for calculating influence scores is O (d |T | + |T | log(|T |)).
Influence [Nguyen and Wong, 2023]. Initially, M random demonstrations are constructed from T  . For each constructed demonstration set S i where | S i | = K, its ICL generalization performance on the entire validation set V is calculated by using the external LLM. Then, the influence of each training point z j ∈ S i is calculated as the difference between the average performance of demonstration sets including z j and the average performance of demonstration sets omitting z j. Through this design analysis, we can infer the running cost of Influence as O (C Q |V| M).
Curation [Chang and Jia, 2023]. There are two variants: CondAcc and Data Models. Specifically, the CondAcc variant is almost similar to Influence. However, for each constructed random demonstration set, the ICL generalization performance of its each permutation on V is separately evaluated. Thereby, the running cost of CondAcc is given by O (C Q |V| MK!). In the Data Models variant, a surrogate linear model is trained to mimic the prediction performance of the external LLM. Similar to Influence, M random demonstration sets are constructed. Each random demonstration set is used to train a separate linear model. For a given random demonstration set, the employed linear model training loss calculates the difference between generalization performances of linear model and external LLM (based on ICL) on the validation set. After this training, the influence of each training point belonging to a random demonstration set is calculated by analyzing the linear model parameters. Through this design analysis, we can infer the running cost of Data Models as O (C Q |V| M).
The running costs of both Influence and Curation are dominated by the term C Q |V| M. Here, M which denotes the number of constructed random demonstration sets, needs to be large in-order to effectively cover the entire training set, and to obtain good estimates of influence scores [Nguyen and Wong, 2023]. As a consequence, both Influence and Curation incur an extremely large amount of external LLM access calls. For InfICL, we approximately require |T | local LLM access calls, which makes InfICL much more cost-effective than both Influence and Curation.

# 3.3 Design Intuitions

In this section, we describe our intuitions behind the design of our InfICL. Specifically, we describe about the plausibility that the influential training points identified for the classifier F can also become influential for both local LLM P and external LLM Q. For our analysis, to differentiate influence functions for classifier F and local LLM P, we denote I up,params (z i, θ) and I up,params (s i, ϕ) as the up-weighted influence functions for F and P, respectively. Here, the up-weighted influence for local LLM P w.r.t next token prediction loss L nt is given by:

�
��� � �
Consider the scenario when the embedding space is clustered and training points in the same cluster share the same label. This scenario is not unrealistic because P tends to generate closer embeddings for those training inputs which are similar to each other and share the same label. Consider two training points z i = (x i, y i) and z j = (x j, y j) belonging to dense and sparse clusters, respectively. Influence functions typically assign higher influence scores to

<div style="text-align: center;">Table 2: Dataset Details.
</div>
Dataset
Size
Positive Class
|T |
|V|
Test Set Size
CoLA
9594
70%
8466
85
1043
RTE
2717
50%
2466
24
277
SST2
20872
50%
19800
200
872
training points from sparse clusters compared to those from dense clusters. This is because, in dense clusters, the removal of a single training point is compensated for by the many similar points within the cluster that can effectively fill its absence. Hence, for the classifier F, we can hypothesize that I up,params (z i, θ) ≤I up,params (z j, θ).
Since the embedding space is generated by the local LLM P, we can apply the same argument used for F, and can further hypothesize that for P we have that I up,params (s i, ϕ) ≤I up,params (s j, ϕ). Therefore, the influential training points for F can also become influential for P.
Most LLMs are pre-trained using the next token prediction strategy and memorize their underlying training data. Consequently, the external LLM Q tends to generate a dense cluster containing s i and numerous other similar training inputs in its own embedding space. As a result, s i tends to have lower influence than s j for Q. Thus, it is plausible that the influential training points for the local LLM P can also be influential for the external LLM Q. This hypothesis was also empirically validated in [Grosse et al., 2023].

# 4 Experiments

# 4.1 Experimental Setup

Datasets. We use three real-world datasets for our empirical evaluation study, Corpus of Linguistic Acceptability (CoLA) [Warstadt et al., 2018], Recognizing Textual Entailment (RTE) [Dagan et al., 2005], and Stanford Sentiment Tree-bank version2 (SST2) [Socher et al., 2013]. The CoLA dataset contains sentences from different linguistics publications, which are expertly annotated for grammatical acceptability by their original authors. Each sentence is either labeled as acceptable or unacceptable. The RTE dataset sample contains two text fragments denoted as premise and hypothesis, and the corresponding label indicates whether the meaning of the hypothesis can be inferred from the text (yes or no). The SST2 is a sentiment analysis dataset wherein, each sentence is labeled as either positive or negative. Table 2 shows dataset details including training, validation, and test splits.
Baselines. We employ two different groups of baselines called the non-influence analysis based baselines which select demonstrations without analyzing influences of training points and influence analysis based baselines which employ influence score calculation to select demonstrations. We select three non-influence analysis based baselines: Zero-shot which directly performs test case inference without any demonstrations, Random where demonstrations are selected based on random sampling, and RICES [Yang et al., 2022] where the training points are scored based on their cosine similarity to the test sample in the embedding space and then the topR training points from each class are selected as demonstrations. We select three influence analysis based baselines: Influence [Nguyen and Wong, 2023], CondACC and Data Models [Chang and Jia, 2023]. We have described these three baselines in Section 3.2. We also compare InfICL against another simple baseline called Classifier where we directly employ a three layer neural network for test case inference.
Training Details. We employ Llama-2-7B [Touvron et al., 2023] as the local LLM. For the external LLM, we separately evaluate on OPT-6.7B, Llama-2-7B, Llama-2-13B, and Llama-2-70B. All Llama-family models are chat versions. The embedding size is 4096. For the classifier, we employ a fully connected neural network with three layers. All experiments are executed on V100-32GB GPU with Intel Xeon 6258R for small models and A100-40GB with AMD EPYC 7543 for large models. We train the classifier using Adam optimizer in 20 epochs with learning rates of 0.001 for CoLA and 0.01 for RTE and SST2.

# 4.2 Experimental Results

Comparison to non-influence analysis based baselines. We show performances of our InfICL and non-influence analysis based baselines on external LLMs in Table 3. Clearly, our InfICL shows an overall better performance than Zero-shot, Random, and RICES across all three datasets and four external LLMs. Zero-shot does not involve any demonstrations. Therefore, the external LLM does not get any opportunity to better understand the given task and as a result, Zero-shot performance is not noticeable. Random under-performs compared to InfICL, indicating that randomly selecting demonstrations does not offer a high-quality learning opportunity to the LLM. Although RICES

<div style="text-align: center;">rmances of our InfICL and non-influence analysis based baselines (mean±std) for external LLMs. Scores fter 5 runs. For each external LLM, the best values for each shot are bold highlighted. ‘N/A’ denotes e and ‘–’ denotes non-feasible results due to the limitation of LLM’s context length.
</div>
External LLM (Q)
Shots (K)
Method
CoLA
RTE
SST2
Accuracy (%)↑
F1 (%)↑
Accuracy (%)↑
F1 (%)↑
Accuracy (%)↑
F1 (%)↑
N/A
N/A
Classifier
82.83 ±0.00
88.18 ±0.00
57.76 ±0.00
58.95 ±0.00
94.50 ±0.00
94.48 ±0.00
Llama-2-7B
0
Zero-shot
63.39 ±0.00
68.81 ±0.00
69.19 ±0.00
68.83 ±0.00
88.76 ±0.00
88.11 ±0.00
8
Random
70.35 ±3.68
75.70 ±4.53
74.97 ±0.21
77.31 ±1.19
93.58 ±1.82
93.88 ±1.54
RICES
70.74 ±0.41
78.50 ±0.28
77.38 ±1.16
80.34 ±0.61
93.88 ±0.07
94.12 ±0.06
InfICL
74.19 ±2.39
81.10 ±2.49
77.26 ±1.25
80.16 ±1.36
94.92 ±0.70
95.04 ±0.66
16
Random
70.20 ±2.30
75.54 ±2.6
77.02 ±0.83
79.24 ±1.20
93.16 ±2.84
93.58 ±2.39
RICES
73.71 ±0.52
80.97 ±0.42
76.77 ±1.37
80.44 ±1.29
93.88 ±0.96
94.10 ±0.92
InfICL
74.75 ±1.32
81.39 ±0.92
78.58 ±0.55
80.98 ±0.41
95.26 ±0.07
95.39 ±0.13
32
Random
73.00 ±1.68
78.74 ±2.00
77.38 ±1.10
79.87 ±1.02
91.78 ±4.22
92.43 ±3.43
RICES
74.02 ±0.51
80.96 ±0.86
73.89 ±0.55
75.86 ±0.48
91.82 ±0.18
92.02 ±0.12
InfICL
73.48 ±0.74
79.50 ±1.19
77.74 ±0.55
79.92 ±1.14
95.15 ±0.13
95.30 ±0.09
Llama-2-13B
0
Zero-shot
50.07 ±0.00
45.29 ±0.00
77.25 ±0.00
78.82 ±0.00
84.40 ±0.00
86.07 ±0.00
8
Random
73.17 ±3.76
78.53 ±5.15
80.39 ±0.21
82.51 ±0.80
95.49 ±0.13
95.61 ±0.11
RICES
73.42 ±0.92
81.37 ±0.75
77.86 ±1.16
81.89 ±0.84
94.30 ±0.57
94.59 ±0.49
InfICL
76.66 ±1.71
82.31 ±1.47
82.43 ±2.21
84.25 ±1.74
95.64 ±0.80
95.67 ±0.85
16
Random
75.40 ±1.48
81.48 ±1.98
82.31 ±1.57
84.08 ±1.29
95.60 ±0.40
95.70 ±0.36
RICES
73.94 ±0.88
82.11 ±0.48
79.66 ±1.37
82.80 ±1.27
93.04 ±1.47
93.50 ±1.26
InfICL
77.47 ±0.32
84.58 ±0.47
83.63 ±0.21
85.08 ±0.39
95.87 ±0.11
95.94 ±0.16
32
Random
75.95 ±1.74
83.06 ±1.27
81.76 ±1.26
82.49 ±1.54
94.72 ±0.60
94.96 ±0.51
RICES
73.23 ±0.70
82.12 ±0.69
77.08 ±0.91
77.70 ±0.99
92.51 ±1.92
93.05 ±1.65
InfICL
76.05 ±0.81
84.20 ±0.41
82.67 ±1.08
83.67 ±1.14
95.95 ±0.13
96.04 ±0.15
OPT-6.7B
0
Zero-shot
66.92 ±0.00
80.07 ±0.00
54.15 ±0.00
60.44 ±0.00
54.82 ±0.00
54.29 ±0.00
8
Random
63.37 ±0.17
75.43 ±2.64
56.92 ±2.73
67.98 ±2.81
60.78 ±0.30
71.74 ±0.24
RICES
64.30 ±0.11
76.85 ±0.20
55.60 ±1.30
69.02 ±0.06
69.72 ±0.70
58.66 ±1.30
InfICL
63.50 ±0.78
76.76 ±0.33
57.76 ±0.63
70.43 ±0.80
91.40 ±1.39
91.95 ±1.12
16
Random
62.03 ±0.50
77.07 ±2.87
54.51 ±0.63
63.84 ±0.86
59.44 ±2.02
71.31 ±0.91
RICES
63.69 ±0.06
76.03 ±0.01
52.11 ±1.10
66.31 ±0.68
75.84 ±0.13
70.08 ±0.18
InfICL
63.79 ±0.55
76.48 ±0.70
57.28 ±0.91
70.14 ±0.50
90.71 ±1.58
91.34 ±1.21
32
Random
59.66 ±0.70
72.24 ±1.62
–
–
61.28 ±0.52
72.09 ±0.36
RICES
61.39 ±0.22
74.38 ±0.32
–
–
79.05 ±0.33
75.38 ±0.45
InfICL
61.77 ±0.77
73.48 ±1.60
–
–
93.58 ±0.40
93.69 ±0.37
Llama-2-70B
0
Zero-shot
74.02 ±0.00
78.61 ±0.00
80.14 ±0.00
79.25 ±0.00
93.12 ±0.00
93.45 ±0.00
8
Random
74.78 ±4.51
78.79 ±5.00
86.28 ±0.36
87.53 ±0.35
89.18 ±4.60
90.35 ±3.76
RICES
78.91 ±0.47
85.29 ±0.40
84.72 ±0.21
86.45 ±0.24
91.40 ±0.11
91.14 ±0.13
InfICL
79.71 ±3.02
84.84 ±3.31
87.61 ±0.91
88.46 ±0.87
94.80 ±0.75
95.02 ±0.65
16
Random
77.28 ±1.42
81.73 ±1.49
86.04 ±1.10
87.64 ±0.64
90.79 ±3.85
91.60 ±3.15
RICES
77.82 ±0.31
84.62 ±0.33
83.39 ±0.63
85.72 ±0.46
91.36 ±0.07
91.11 ±0.10
InfICL
80.92 ±1.60
86.32 ±1.10
87.97 ±0.21
89.03 ±0.22
94.61 ±1.09
94.76 ±0.96
32
Random
78.65 ±0.87
83.56 ±1.56
87.00 ±0.36
88.56 ±0.41
92.32 ±2.60
92.85 ±2.22
RICES
76.93 ±0.24
84.24 ±0.17
80.14 ±0.21
82.54 ±0.13
91.44 ±0.07
91.53 ±0.05
InfICL
78.94 ±1.30
85.36 ±0.93
88.09 ±0.36
89.11 ±0.27
95.53 ±0.34
95.67 ±0.32
offers personalized demonstrations, it fails to select highly influential demonstrations. This selection is crucial for enhancing the ICL performance. Hence, RICES also under-performs relative to InfICL.
For Llama-2-7B and SST2 dataset, our InfICL shows superior performance against baselines. However, RICES outperforms InfICL with 8 and 32 shots for RTE and CoLA datasets, respectively. This is because, in a few cases, choosing personalized demonstrations that are similar to the test sample can enhance performance compared to influence analysis. For Llama-2-13B and across all three datasets, our InfICL clearly outperforms baselines. Counter-intuitively, InfICL performs better with 16 shots compared to 32 shots. This outlier phenomenon can sometimes occur due to the information interference effect between demonstrations [Chen et al., 2023a]. For OPT-6.7B and both RTE and SST2 datasets, InfICL maintains its superior performance over baselines. However, for CoLA dataset, Zero-shot outperforms other methods. OPT-6.7B is a small sized LLM compared to other external LLMs. Consequently, in some datasets like CoLA, it does not effectively utilize demonstrations. For the Llama-2-70B and across all datasets, our InfICL outperforms baselines.
We further perform student’s t-test between InfICL and non-influence analysis based baselines on all three datasets and four external LLMs. We perform this analysis on accuracy scores and the results are shown in Table 4. Out of 24 t-test cases, the p-values show statistical significance in 20 cases (based on the threshold of 0.05), which demonstrating the superiority of our InflCL.
Correlation between influence scores and InfICL performance. We conduct an empirical study to analyze the correlation between influence scores and InfICL performance. As previously mentioned in Section 3.1, training points

Table 4: Student’s t-test analysis results between our InfICL and non-influence analysis based baselin is calculated by using the accuracy scores for all shots and runs. Statistically significant p-values are b (p-value <0. 05).

Table 4: Student’s t-test analysis results between our InfICL and is calculated by using the accuracy scores for all shots and runs. S (p-value <0. 05).

4: Student’s t-test analysis results between our InfICL and non-influence analysis based baselines. The p-value ulated by using the accuracy scores for all shots and runs. Statistically significant p-values are bold highlighted

External LLM
Dataset
Method 1
Method 2
p-value
Llama-2-7B
CoLA
InfICL
Random
0.0449
RICES
0.7363
RTE
InfICL
Random
0.0207
RICES
0.0286
SST2
InfICL
Random
0.0296
RICES
0.0002
Llama-2-13B
CoLA
InfICL
Random
0.0229
RICES
0.0007
RTE
InfICL
Random
0.0384
RICES
0.0005
SST2
InfICL
Random
0.0324
RICES
0.0001
OPT-6.7B
CoLA
InfICL
Random
0.0686
RICES
0.8550
RTE
InfICL
Random
0.1190
RICES
0.0075
SST2
InfICL
Random
0.0001
RICES
0.0001
Llama-2-70B
CoLA
InfICL
Random
0.0247
RICES
0.0125
RTE
InfICL
Random
0.0002
RICES
0.0001
SST2
InfICL
Random
0.0030
RICES
0.0001
<div style="text-align: center;">ble 5: Effect of choosing training points from different range of influence scores on the InfICL performance. Score reported after 5 runs. External model: Llama-2-7B. Dataset: CoLA.
</div>
Table 5: Effect of choosing training points from different range of influence scores on the InfICL p are reported after 5 runs. External model: Llama-2-7B. Dataset: CoLA.

Shots
Infl. Scores
Accuracy (%)↑
F1 (%)↑
8
Top Positive
74.19 ±2.39
81.10 ±2.49
Middle
72.94 ±1.88
80.22 ±2.56
Top Negative
71.54 ±1.43
81.67 ±0.67
16
Top positive
74.75 ±1.32
81.39 ±0.92
Middle
73.46 ±2.12
81.01 ±2.46
Top negative
69.78 ±3.54
78.58 ±3.69
32
Top positive
73.02 ±1.73
82.05 ±1.16
Middle
71.81 ±4.13
79.09 ±4.63
Top negative
64.29 ±3.40
71.24 ±4.45
exhibiting higher positive influence scores have the potential to enhance the InfICL predictive performance. In this empirical study, we assess how selecting demonstrations from varying influence ranges impacts the InfICL performance. We initially rank training points based on their influence scores in descending order, then form different demonstration sets using three strategies: selecting training points with the highest positive influence, those within the mid-range of influence, and those with the highest negative influence. We report the InfICL performance for different ranges of influence scores in Table 5. Notably, opting for training points with the highest positive influence scores as demonstrations yields the most favorable performance.
Comparison to influence analysis based baselines. Since the employed influence analysis based baselines Influence, CondACC, and Data Models have an extremely high running costs, they can only run on a small size training and validation sets. To conduct a fair comparison, we run our InfICL and baselines in the same dataset setting as mentioned in Nguyen and Wong [2023], which has train/validation/test size as 400/200/500, respectively. In-order to reduce the high cost of experimentation, we conduct our empirical study using two external LLMs OPT-6.7B and Llama-2-7B and on two datasets CoLA and RTE.
We show the empirical results comparing our InfICL with other influence analysis based baselines in Table 6. For the CoLA dataset and for both Llama-2-7B and OPT-6.7B, InfICL shows an overall better performance than other baselines. For the RTE dataset and Llama-2-7B, InfICL again outperforms Influence. However, for OPT-6.7B, InfICL has a lower accuracy than Influence for 12 shots. This indicates that the chosen 12 demonstrations based on InfICL do not convey sufficient information that can be exploited by OPT-6.7B. For the setting of Llama-2-7B and 4 shots on CoLA dataset, our InfICL incurs 10 minutes of execution latency, Influence takes 3.5 hours, and both CondAcc and Data Models take more than 80 hours.

Table 6: Test accuracy of our InfICL and influence analysis based baselines on different external denotes the results extracted from Nguyen and Wong [2023]. Cells marked ’–’ denotes non-feasib extremely high training latency.

6: Test accuracy of our InfICL and influence analysis based baselines on different external LLMs. Asterik s the results extracted from Nguyen and Wong [2023]. Cells marked ’–’ denotes non-feasible results due to

Dataset
Ext. LLM
Shots
InfICL
Influence
CondAcc
Data Models
CoLA
OPT-6.7B
4
68.20
31.80
48.40
45.50
8
64.80
46.00
35.60
36.50
16
69.00
46.80
–
–
32
69.20
58.60
–
–
Llama-2-7B
4
77.80
74.40
72.90
71.2
8
77.80
78.20
–
–
16
77.20
73.20
–
–
32
76.80
74.40
–
–
RTE
OPT-6.7B
12
51.20
62.70∗
–
–
Llama-2-7B
12
77.60
75.60
–
–
In this work, we introduced a demonstration selection method for ICL by analyzing influences of training samples using influence functions. Our approach utilizes a local LLM to generate sample embeddings thereby, avoiding the expensive fine-tuning of the LLM. Empirical studies on various real-world datasets demonstrated advantages of our method over state-of-the-art baselines. For future work, we aim to expand our demonstration selection method to Large Vision-language Models (LVMs), and extend our method to address more complex problems such as massive multitask language understanding. We release our source code at https://tinyurl.com/edry6nn4.

# 6 Limitations

Although we have demonstrated that influence function analysis can be effective for selecting ICL demonstrations, we have not conducted an in-depth interpretability study on why influence functions improve ICL performance. We based our use of influence functions on the intuition that highly influential training samples benefit model learning. However, since ICL does not involve any model gradient updates and differs significantly from gradient update-based learning, a theoretical study is needed to connect mechanisms of ICL with gradient update-based models [Xie et al., 2022], and show that highly influential training samples can also enhance ICL performance.

# Acknowledgement

This work was supported in part by NSF grants 1920920 and 1946391.

# References

Naman Agarwal, Brian Bullins, and Elad Hazan. 2017. Second-order stochastic optimization for machine learning in linear time. The Journal of Machine Learning Research, 18(1):4148–4187.
Samyadeep Basu, Phil Pope, and Soheil Feizi. 2021. Influence functions in deep learning are fragile. In International Conference on Learning Representations.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems.
Marc-Etienne Brunet, Colleen Alkalay-Houlihan, Ashton Anderson, and Richard Zemel. 2019. Understanding the origins of bias in word embeddings. In Proceedings of the 36th International Conference on Machine Learning.
Ting-Yun Chang and Robin Jia. 2023. Data curation alone can stabilize in-context learning. In Proceedings of the Annual Meeting of the Association for Computational Linguistics.
Jiuhai Chen, Lichang Chen, Chen Zhu, and Tianyi Zhou. 2023a. How many demonstrations do you need for in-context learning? In Findings of the Association for Computational Linguistics: EMNLP.
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. 2020. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, ICML.

Yanda Chen, Chen Zhao, Zhou Yu, Kathleen R. McKeown, and He He. 2023b. On the relation between sensitivity and accuracy in in-context learning. In Findings of the Association for Computational Linguistics: EMNLP.
Ido Dagan, Oren Glickman, and Bernardo Magnini. 2005. The PASCAL recognising textual entailment challenge. In Machine Learning Challenges, Evaluating Predictive Uncertainty, Visual Object Classification and Recognizing Textual Entailment, First PASCAL Machine Learning Challenges Workshop, MLCW.
Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, and Zhifang Sui. 2023. A survey for in-context learning. CoRR, abs/2301.00234.
Minghong Fang, Neil Zhenqiang Gong, and Jia Liu. 2020. Influence function based data poisoning attacks to top-n recommender systems. In Proceedings of The Web Conference.
Vitaly Feldman and Chiyuan Zhang. 2020. What neural networks memorize and why: Discovering the long tail via influence estimation. In Annual Conference on Neural Information Processing Systems.
Tianyu Gao, Adam Fisch, and Danqi Chen. 2021. Making pre-trained language models better few-shot learners. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing.
Roger B. Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, Evan Hubinger, Kamile Lukosiute, Karina Nguyen, Nicholas Joseph, Sam McCandlish, Jared Kaplan, and Samuel R. Bowman. 2023. Studying large language model generalization with influence functions. CoRR, abs/2308.03296.
Xiaochuang Han and Yulia Tsvetkov. 2021. Influence tuning: Demoting spurious correlations via instance attribution and instance-driven updates. In Findings of the Association for Computational Linguistics: EMNLP.
Xiaochuang Han, Byron C. Wallace, and Yulia Tsvetkov. 2020. Explaining black box predictions and unveiling data artifacts through influence functions. ArXiv.
Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. 2022. Datamodels: Understanding predictions with data and data with predictions. In International Conference on Machine Learning, ICML.
Matthew Jagielski, Giorgio Severi, Niklas Pousette Harger, and Alina Oprea. 2021. Subpopulation data poisoning attacks. In Proceedings of the ACM SIGSAC Conference on Computer and Communications Security.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wentau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the Conference on Empirical Methods in Natural Language Processing.
Pang Wei Koh and Percy Liang. 2017. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning.
Shuming Kong, Yanyan Shen, and Linpeng Huang. 2022. Resolving training biases via influence-based data relabeling. In International Conference on Learning Representations.
Donghoon Lee, Hyunsin Park, Trung Pham, and Chang D. Yoo. 2020. Learning augmentation network via influence functions. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
Xiaonan Li, Kai Lv, Hang Yan, Tianyang Lin, Wei Zhu, Yuan Ni, Guotong Xie, Xiaoling Wang, and Xipeng Qiu. 2023. Unified demonstration retriever for in-context learning. In Proceedings of the Annual Meeting of the Association for Computational Linguistics.
Xiaonan Li and Xipeng Qiu. 2023. Finding support examples for in-context learning. In Findings of the Association for Computational Linguistics: EMNLP.
Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2022. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out: The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures.
Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2021. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. CoRR, abs/2107.13586.
Man Luo, Xin Xu, Zhuyun Dai, Panupong Pasupat, Seyed Mehran Kazemi, Chitta Baral, Vaiva Imbrasaite, and Vincent Y. Zhao. 2023. Dr.icl: Demonstration-retrieved in-context learning. CoRR, abs/2305.14128.
Tai Nguyen and Eric Wong. 2023. In-context example selection with influences. CoRR, abs/2302.11042. Sejoon Oh, Sungchul Kim, Ryan A. Rossi, and Srijan Kumar. 2021. Influence-guided data augmentation for neural tensor completion. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management.

Chengwei Qin, Aston Zhang, Anirudh Dagar, and Wenming Ye. 2023. In-context learning with iterative demonstration selection. CoRR, abs/2310.09881.
Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2022. Learning to retrieve prompts for in-context learning. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL.
Alexander Scarlatos and Andrew S. Lan. 2023. Reticl: Sequential retrieval of in-context examples with reinforcement learning. CoRR, abs/2305.14502.
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the Conference on Empirical Methods in Natural Language Processing.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aur´elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971.
Xinyi Wang, Wanrong Zhu, and William Yang Wang. 2023. Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning. arXiv:2301.11916.
Alex Warstadt, Amanpreet Singh, and Samuel R Bowman. 2018. Neural network acceptability judgments. arXiv preprint arXiv:1805.12471.
Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, and Lingpeng Kong. 2023. Self-adaptive in-context learning: An information compression perspective for in-context example selection and ordering. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics.
Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2022. An explanation of in-context learning as implicit bayesian inference. In International Conference on Learning Representations.
Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang. 2022. An empirical study of GPT-3 for few-shot knowledge-based VQA. In Thirty-Sixth Conference on Artificial Intelligence, AAAI.
Yiming Zhang, Shi Feng, and Chenhao Tan. 2022. Active example selection for in-context learning. In Proceedings of the Conference on Empirical Methods in Natural Language Processing.

