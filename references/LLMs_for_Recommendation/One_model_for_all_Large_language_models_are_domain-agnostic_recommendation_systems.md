# e Model for All: Large Language Models are Domain-Agnost Recommendation Systems

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7a7a/7a7a02ee-c589-4bd9-ba0a-f8daca862c26.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0ba0/0ba0a2d3-9c16-4a94-91ac-eaac30f2800d.png" style="width: 50%;"></div>
Jun Zhou jun.zhoujun@antgroup.com Ant Group China


# ABSTRACT

The purpose of sequential recommendation is to utilize the interaction history of a user and predict the next item that the user is most likely to interact with. While data sparsity and cold start are two challenges that most recommender systems are still facing, many efforts are devoted to utilizing data from other domains, called cross-domain methods. However, general cross-domain methods explore the relationship between two domains by designing complex model architecture, making it difficult to scale to multiple domains and utilize more data. Moreover, existing recommendation systems use IDs to represent item, which carry less transferable signals in cross-domain scenarios, and user cross-domain behaviors are also sparse, making it challenging to learn item relationship from different domains. These problems hinder the application of multi-domain methods to sequential recommendation. Recently, large language models (LLMs) exhibit outstanding performance in world knowledge learning from text corpora and general-purpose question answering. Inspired by these successes, we propose a simple but effective framework for domain-agnostic

† Corresponding author. ∗ Work done when Zuoli Tang was an intern at Ant Group.

† Corresponding author. ∗ Work done when Zuoli Tang was an intern at Ant Group.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. Conference’17, July 2017, Washington, DC, USA © 2024 Association for Computing Machinery. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/XXXXXXX.XXXXXXX

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. Conference’17, July 2017, Washington, DC, USA © 2024 Association for Computing Machinery. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/XXXXXXX.XXXXXXX

# Chenliang Li †
cllee@whu.edu.cn

Chenliang Li †
cllee@whu.edu.cn Key Laboratory of Aerospace Information Security and Trusted Computing Wuhan University China

recommendation by exploiting the pre-trained LLMs (namely LLMRec). We mix the user’s behavior across different domains, and then concatenate the title information of these items into a sentence and model the user’s behaviors with a pre-trained language model. We expect that by mixing the user’s behaviors across different domains, we can exploit the common knowledge encoded in the pre-trained language model to alleviate the problems of data sparsity and cold start problems. Furthermore, we are curious about whether the latest technical advances in nature language processing (NLP) can transfer to the recommendation scenarios. With the proposed LLM-Rec, we mainly want to explore the following research questions: (I) whether using large language models to model users’ multi-domain information can achieve better recommendation results than general pure ID models; (II) whether increasing the size of the pre-trained language model can further improve the performance of multi-domain recommendations and whether the performance of language models with the same size is the same; and (III) whether the parameter-efficient fine-tuning methods prevalent in the natural language processing field can also work in the recommendation scenario. To answer these questions, we conduct extensive experiments, expanding the parameters of the language model from 40 million to 6.7 billion, and comparing experiments on different types of standard architectures of language models. Our experiments verified the effectiveness of pre-trained language models in modeling users’ behaviors across different domains, and we will release our code and model weights after the paper is accepted.

# CCS CONCEPTS

• Information systems → Recommender systems.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8232/82324f0e-6a5c-4c1e-85f1-f43baf95bb0a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Users’ interests across various domains exhibit semantic correlations. A user who enjoys war-themed movies may also be interested in books related to World War II. KEYWORDS
</div>
Large Language Model, Multi-Domain Recommendation, Sequential Recommendation
ACM Reference Format: Zuoli Tang ∗, Zhaoxin Huan, Zihao Li, Xiaolu Zhang, Jun Hu, Chilin Fu, Jun Zhou, and Chenliang Li †. 2024. One Model for All: Large Language Models are Domain-Agnostic Recommendation Systems. In Proceedings of ACM Conference (Conference’17). ACM, New York, NY, USA, 10 pages. https://doi.org/XXXXXXX.XXXXXXX

# 1 INTRODUCTION

Based on the historical interactions, sequential recommendation aims to capture users’ interests and provide accurate suggestions among massive candidate items to address the information overload problem. It has been widely used in various e-commerce and advertising companies, e.g., Amazon, Alibaba, and Ebay. Although the previous sequential recommendation methods achieved great success with the prosperity of deep learning, they still face two challenges: data sparsity and cold start, which hinder the further development of recommendation system. Considering that users’ interests across various domains may exhibit semantic correlations, as shown in Figure 1, many research works attempt to introduce user interactions from other domains as auxiliary knowledge to enhance the model’s capability (i.e., cross-domain recommendation) and achieves promising results. Some works [15, 26, 30] construct a unified graph for different domains, and utilize graph neural network to propagate cross-domain information in the graph. Additionally, other works [7, 20] consider each domain as a task, multi-task approaches(such as MMoE [18], PLE [22]) are utilized to solve this problem. Although applying cross-domain methods becomes ubiquitous in sequential recommendation [1, 16], there are few works that extend them for multi-domain sequential recommendation since the following obstacles remain to be overcome. First, to realize the knowledge transfer between two domains, a complicated model structure is required for most of the cross-domain solutions. However, the sophisticated designs diminish the feasibility and the flexibility of model extension and adaption. Second, the premise assumption of cross-domain recommendation (i.e., the source domain is always well-informed than the target domain, thus, cross-domain recommendation aims to exploit the source domain for target domain

improvement) specializes in target domain enhancement while neglecting the source domain, we believe both the source domain and target domain are important. Last but not least, most of the existing works leverage the ID, e.g., a number code, for item symbolization, which will further be utilized for recommendation. We argue that such a string of numbers could not release any transferable information corresponding to items and domains. Specifically, we concatenate user interactions across multi-domains straightforwardly in terms of item id. These interaction sequences are then fed into a standard sequential baseline SASRec. Then, we verify the model performance on each single domain, the results are shown in Figure 2. We could find that simply merging cross-domain behavior does not bring performance improvement and may even lead to performance degradation. Recently, large language models deliver outstanding capacity in learning world knowledge from text data. We believe the world knowledge encapsulated in LLM could bridge the gap between different scenarios. To this end, we endeavor to leverage the contextual information of each item for multi-domain recommendation. A general domain-agnostic framework with pre-trained LLMs is proposed (namely LLM-Rec). More concretely, for each item, we use the item title for item representation generation and for each individual we collect his or her interacted items from different domains along the timeline. Then, we further concate the titles as a sentence, and feed them into the LLM for user behavior modeling. By configuring different kinds of LLMs for LLM-Rec, we aim to investigate the following questions: Q1: Is it feasible to use a language model as a unified user encoder and item encoder to model user behaviors across different domains? Can it leverage the advantages of language models to enhance performance on all domains? To answer this question, we collect the same user’s behaviors across five domains for modeling and observe the performance of each domain. Q2: Whether the conclusion “bigger is better” (i.e., in general, the performance will be improved with larger model size in CV [4] and NLP [3]) still works in multi-domain sequential recommendation? And whether the model will show similar performance under similar size? To verify this assumption, we investigate the performance of three types of language models including both encoder-only and decoder-only structures. The parameter size of those models ranges from 40 million to 6. 7 billion. Q3: Whether the efficient tuning methods, e.g., LoRA (Low-rank Adaption) [11], in LLM can also works well on recommendation scenario? Confined by the computation resource, it will be intractable and time-consuming to perform full-parameter fine tuning for different downstream tasks. Apart from that, the requirements of low-carbon and green AI also render it to be indispensable and significant to devise a more efficient and effective parameter-tuning method. To this end, we further conduct experiments with LoRA fine-tuning methods on various model sizes and also compare its performance to the full-parameter counterpart. In a nutshell, we make the following contributions in this paper:
• We explore the ability of language models in multi-domain behaviors modeling and demonstrate the effectiveness of world knowledge embedded in language models for multi

• We explore the ability of language models in multi-domain behaviors modeling and demonstrate the effectiveness of world knowledge embedded in language models for multidomain recommendation.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0e6c/0e6cde7b-932a-4a13-9595-e27b5e39fe65.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Applying single-domain ID-based model (SASRec) to multi-domain scenario.
</div>
• We conduct comprehensive experiments on five real-world datasets to verify the effectiveness of our method over six baselines for multi-domain sequential recommendation.
• We further investigate the critical factors (including model architectures, model sizes, data settings, as well as fine-tuning methods) of pre-trained language model to the final performance for multi-domain sequential recommendation.

# 2 RELATED WORK

2 RELATED WORK
This section provides a brief overview of representative efforts relevant to our work.

# 2.1 Sequential Recommendation

In the early stage, the dominant efforts were devoted to modeling users’ sequential behaviors as a Markov chain such that the item-item transition matrix is learnt for next item prediction [19]. Attribute to the extraordinary capability of deep learning for complicated pattern modeling, a surge of deep neural networks, e.g., RNN-based [9] and CNN-based [23], are elaborated successively for sequential recommendation. However, these solutions often fail to capture long-term dependency between any of two items in sequence, thus, Transformer-based solutions, e.g., SASRec [12], BERT4Rec [21], are applied and achieved promising results in sequential recommendation. Moreover, some sophisticated methods, e.g., contrastive learning [2, 25] are also adopted to relieve the data sparaity in sequential recommendation.

# 2.2 Multi-domain Recommendation

Although conventional methods achieve remarkable success for recommendation, there are two long-standing obstacles that hinder the further improvement of recommendation system: data sparsity and cold-start problems. To alleviate this issue, some works [7, 15, 17] propose to exploit the interaction behaviors from other domains as external information for better performance, namely multi-domain recommendation. For instance, MCF [29] extracts collaborative information from different domains for recommendation. PPGN [30] models the user’s multi-domain interaction records as a graph. Then, the multi-domains information could be aggregated by the

graph neural network. BiTGCF [15] devices a domain feature propagation layer with GNN for knowledge transferring across different domains. ADI [7] denotes multi-domain recommendation as multitask learning, then proposes a domain-share network for multidomain recommendation. On the contrary, MAMDR [17] argues using a domain-share network straightforwardly for multi-domain recommendation will hinder the model to achieve optimal results due to the domain conflicts. Hence, a Domain Negotiation strategy with regularization is proposed to balance the specificity of each domain as well as the commonality of multi-domains. However, all the above works do not specialize in sequential recommendation, neither of them explores language models for multi-domain recommendation.

# 2.3 Text-based Recommendation

Encouraged by the remarkable success of LLM, some methods endeavor to apply language models for recommendation [6, 10, 14, 24, 27], very recently. MoRec [27] explores the feasibility of using textual representations (instead of ID embedding) generated by language model for recommendation. ZESRec [6] explores the zero-short ability of sequential recommendation model. To be specific, they first select one domain and adopt a pre-trained language model to generate item representations in terms of their associated textual information. The resultant representations are then used for sequential model pre-training. Afterward, the well-trained recommendation model will be applied for other domain recommendation. Following ZESRec, UniSRec [10] utilizes multi-domain data to pretrain the sequential model first, then the downstream domain data is further used for model fine-tuning to obtain better performance. Futhermore, Miracle [24] combines multi-interest learning with pre-training and proposes a novel sparse capsule routing network. Different from [6, 10], Recformer [14] pre-trains and fine-tunes a language model in a holistic approach for item text encoding and sequential recommendation. However, the above three [6, 10, 14] follow the pre-training and fine-tuning, a two-stage paradigm, we believe this recipe might be inefficient and cumbersome to some extent. Moreover, none of them investigate the potentiality of language models in multi-domain recommendation that items in the a user sequence come from the different domain, nor do they explore the effect of model size to the final recommendation performance.

# 3 METHODOLOGY

This section outlines the multi-domain sequential recommendation problem and introduces the key components of proposed LLM-Rec: different backbones, model inputs construction, score prediction and model optimization. The overall framework is shown in Figure 3

# 3.1 Problem Formulation

Given 𝑁 domains (i.e., D 1, D 2,..., D 𝑁), denoting U 𝑖, V 𝑖 and R 𝑖
as user set, item set and interaction set of domain 𝑖, respectively. Thus, we could collect the user, item and interactions from all the domains, and let U = U 1 ∪U 2 ∪.... U 𝑁, V = V 1 ∪V 2 ∪.... V 𝑁
and R = R 1 ∪R 2 ∪.... R 𝑁 as user set, item set and interaction set of all the domains, respectively. Single-domain Sequential Recommendation. Given U 𝑛, V 𝑛
and R 𝑛 in domain D 𝑛, we can organize the user 𝑢 ’s interaction

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ccf3/ccf321e9-d40f-4c32-af8f-0447408645e3.png" style="width: 50%;"></div>
<div style="text-align: center;">Item Title Sequence
</div>
<div style="text-align: center;">(b) Text-Aware Item Representation Learning
</div>
<div style="text-align: center;">Figure 3: The overview of the proposed LLM-Rec
</div>
history from domain D 𝑛 as a sequence 𝑠 𝑛 𝑢 = (𝑣 𝑛 1, 𝑣 𝑛 2, ..., 𝑣 𝑛 𝐿) along the timeline, where 𝑣 𝑛 𝑖 ∈V 𝑛 and 𝐿 is the length of sequence. Hence, single-domain sequential recommendation aims to predict the probability of next item 𝑣 𝑛 𝐿 + 1 based on 𝑠 𝑛𝑢. Multi-domain Sequential Recommendation. Given U, V and R from all the domains. For each user 𝑢 in U, we can collect the user’s all interaction history from all domains and organize a mixture sequence 𝑠 𝑢 = (𝑣 1, 𝑣 2, ..., 𝑣 𝐿 ′) chronologically, where (𝑣 1, ..., 𝑣 𝐿 ′ may come from different domains. Then, given a target domain D 𝑇, multi-domain sequential recommendation aims to predict the probability of next item 𝑣 𝑇 𝐿 ′ + 1 ∈V 𝑇 based on the mixture sequence.

# 3.2 LLM-Rec

# 3.2.1 Backbones. As aforementioned, we obtain three types of standard architectures of LLM as our backbone for investigation.

• Encoder-only. Encoder-only architecture utilizes a bi-directional attention (ref. the bottom left part in Figure 3) for context information modeling and mask token prediction. We choose BERT [5] as the representative.

BERT [5] as the representative.
• Decoder-only. Different from Encoder-only architectures, Decoder-only utilizes a uni-directional (ref. the bottom right part in Figure 3) attention for the next token generation. Such that, only the previous information could be obtained via the self-attention to produce the next outcome. This paradigm holds a dominant position in the design of autoregressive LLM nowadays. We, thereby, apply Open Pre-trained Transformers (OPT) [28], a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, as another backbone of our LLM-Rec.
• Encoder-Decoder. The Encoder-Decoder architecture is another prevalent design approach employed in LLM, which contains both an encoder (i.e., responsible for transforming input data into a fixed-length representation) and a decoder

<div style="text-align: center;">Item Title Sequence
</div>
(i.e., tasked with generating the output sequence based on the encoded representation). We obtain FLAN-T5 [3], a variant of T5 that is fine-tuned with prompting in a mixture of tasks, as our backbone. It should be noted that we use language models for representation learning, so we only use the encoder of FLAN-T5.

3.2.2 Model Inputs. In our paper, we use the item title 𝑡 𝑣 instead of ID as the inputs of the model, where 𝑡 𝑣 = (𝑤 1 𝑣, ...,𝑤 𝐾 𝑣) and 𝑤 𝑘 𝑣 is the token after tokenization. Following the common practices in NLP, we add a special token ’[CLS]’ at the beginning of the text sequence for bi-directional Transformer and append the ’</s>’ at the end of sequence with regard to the uni-directional Transformer. We can also call these two architectures as non-autoregressive (NAR) and autoregressive (AR). Consequently, each item can be represented as below:

# • Item Input

(1)

As for the user representation, we concatenate each item representation yield by Equation 1 as a sentence for user representation, which can be formalized as:

(2)

[/ () / () /]
where 𝐼𝑛𝑝𝑢𝑡 𝑤 / 𝑜 (𝑣) means removing the ’[cls]’ or ’</s>’ token in the item input 𝐼𝑛𝑝𝑢𝑡 (𝑣) to keep the same input format of sentence in NLP. And the special tokens are same as Equation ( 1).

3.2.3 Representation and Prediction. The input of item and user will be further fed into the backbone for sequence modeling and representation generation. To be specific, for the models with NAR

Transformer architecture, we obtain the correspondent hidden vector of ’[CLS]’ in the last layer as the user or item representation. As for the AR Transformer architectures, the outcome with regard to ’</s>’ is utilized, which can be formulated as follows:

(3)
(4)

(3)

(4)

(())
where ℎ 𝑢 ∈ R 1 × 𝑑 and ℎ 𝑣 ∈ R 1 × 𝑑 are the representation of user and item, respectively. We predict the next item based on the score between a user’s representation ℎ 𝑢 and candidate item representation ℎ 𝑣 yield by Equation 3 and 4. Formally, we calculate the inner product between them as follows:

(5)

3.2.4 Optimization. For each target item, we randomly sample 𝑆 negative instances from the same domain as the hard negatives and adopt the cross-entropy loss as the objective function for LLM-Rec optimization:

(6)

where 𝐵 is the batch size.

# 4 EXPERIMENTS

To demonstrate the effectiveness of our methods and answer the research question in Section 1, we conduct extensive experiments on five public datasets, comparing the performance between our LLMRec with three types of backbones and 6 representative baselines.

# 4.1 Experimental Setup

We first present the statistics of datasets in brief. Then, the advanced baselines, evaluation metrics, and parameter settings will also be given.

We first present the statistics of datasets in brief. Then, the advanced baselines, evaluation metrics, and parameter settings will also be given.

4.1.1 Datasets.  We select five sub-category datasets from the Amazon reviews 1 dataset, namely "Prime Pantry", "Industrial and Scientific", "Musical Instruments", "Arts, Crafts and Sewing", and "Office Products", each of which represents a different domain. Firstly, we remove the items without title information in meta-data as title will be used for item identification, symbolization and modeling. Afterward, we extract historical interaction behaviors from these five domains for each user and order them chronologically for sequence construction. Then, following [10, 12, 31], a 5-core strategy is applied for item and user filtering, i.e., each user is required to contain at least five interaction records from multi-domains, and each item should appear no less than five times in all the records. We leverage a leave-one-out strategy for training, validation, and test dataset split. More concretely, given a user sequence, we use the latest interaction for testing, the penultimate interaction for validation, and the remaining interactions for training. We named the pre-processed multi-domain dataset as SPAIO, the statistical information of whom is presented in Table 1- 3 respectively.

1 https://nijianmo.github.io/amazon/index.html

Dataset
#Users
#Items #Interactions #Training #Testing
Scientific
142,139
14,586
329,123
244,960
43,365
Pantry
55,890
6,224
208,791
162,762
22,412
Instruments
88,249
14,626
348,513
274,760
36,129
Arts
184,207
32,196
723,017
568,600
76,035
Office
259,687
37,380
1,154,924
893,014
132,195
SPIAO
310,136 105,012
2,764,368
2,144,096
310,136
<div style="text-align: center;">Table 2: User overlap between different domains.
</div>
Dataset
Scientific Pantry Instruments
Arts Office
Scientific
-
-
-
-
-
Pantry
26,863
-
-
-
-
Instruments
39,806
10,812
-
-
-
Arts
81,855
30,807
38,353
-
-
Office
124,621
47,206
65,487 155,995
-
<div style="text-align: center;">able 3: Number of users with behavior in different domains
</div>
dataset
1
2
3
4
5
ALL
SPIAO
53,117
129,352
95,719
28,546
3,402
310,136
4.1.2 Comparing Methods. Here, we compare the performance of our LLM-Rec against six baselines and state-of-the-art alternatives from two categories including both single domain methods and multi-domain solutions for performance evaluation. For single domain methods, we regard the mixed dataset SPIAO as a single domain. Single domain methods:
• GRU4Rec [9] utilizes a gate recurrent unit model for sequence recommendation.
• LightGCN [8] is a graph neural network (GNN) based model. It captures the high-order connectivity information by stacking GNN layer and simplifies the design in the feature propagation component by removing the non-linear activation and the transformation matrices in the GNN.
• SASRec [12] employs a uni-directional self-attention mechanism for sequence modeling and next itme prediction.
Multi-domain methods:
• ADI [7] models multi-domain recommendation as multi-task learning, which includes both domain-specific expert networks and domain-shared expert networks. A self-training strategy is used to capture label-level connections between domains.
• BiTGCF [15] is a dual-target cross-domain model based on graph collaborative filtering network. It designs a feature propagation layer for different domain information transfers. We extend it to be applicable for multi-domain recommendation denoted as ’BiTGCF+’.
• MAMDR [17] MAMDR is a model-agnostic multi-domain recommendation learning framework, which proposes the

# Multi-domain methods:

• ADI [7] models multi-domain recommendation as multi-task learning, which includes both domain-specific expert networks and domain-shared expert networks. A self-training strategy is used to capture label-level connections between domains.
• BiTGCF [15] is a dual-target cross-domain model based on graph collaborative filtering network. It designs a feature propagation layer for different domain information transfers. We extend it to be applicable for multi-domain recommendation denoted as ’BiTGCF+’.
• MAMDR [17] MAMDR is a model-agnostic multi-domain recommendation learning framework, which proposes the

Table 4: Performance comparison of different methods, with % omitted. The best result of all methods are highlighted in bold, and the best performance of baselines are underlined. It should be noted that NDCG@N is equal to Recall@N when N=1, so we omit the result of NDCG@1 for simplicity. For different LLMs, we report results of similar model sizes (around 110M).

Dataset
Metrics
Single Domain
Multi Domain
LLM-Rec
GRU4Rec
SASRec
LightGCN
ADI
BiTGCF+
MAMDR
BERT-110M
OPT-125M
FLAN-T5-110M
Scientific
Recall@1
5.31
6.22
4.38
4.03
4.53
6.35
7.81
7.36
7.66
Recall@10
18.70
19.16
16.90
15.96
17.43
20.85
25.61
24.26
25.56
NDCG@10
11.09
11.89
9.85
9.17
10.15
12.71
15.63
14.79
15.58
Pantry
Recall@1
2.81
3.54
2.30
1.99
2.30
3.77
5.36
4.37
4.83
Recall@10
13.68
15.18
10.70
10.39
11.34
16.53
20.42
18.92
20.00
NDCG@10
11.09
8.71
5.91
5.54
6.16
9.40
12.11
10.83
11.54
Instruments
Recall@1
7.13
7.86
4.73
5.49
5.16
8.31
8.92
8.41
8.74
Recall@10
22.08
23.30
20.52
20.63
21.45
26.12
30.57
28.91
30.51
NDCG@10
13.63
14.63
11.58
12.02
12.23
16.14
18.41
17.45
18.28
Arts
Recall@1
8.48
10.77
5.54
6.63
5.89
11.12
13.46
12.48
12.77
Recall@10
25.31
28.25
22.73
23.09
23.83
30.73
36.58
35.01
36.78
NDCG@10
15.89
18.61
12.96
13.76
13.66
19.91
23.90
22.63
23.59
Office
Recall@1
8.58
10.44
5.27
7.50
5.66
10.91
12.75
12.33
12.40
Recall@10
23.64
25.14
20.27
23.30
23.83
27.41
32.58
32.02
32.74
NDCG@10
15.16
16.94
11.80
14.40
13.66
18.18
21.59
21.16
21.50
SPIAO
Recall@1
7.50
9.13
4.93
6.17
5.26
9.50
11.25
10.64
10.85
Recall@10
22.46
24.13
19.75
20.98
20.44
26.37
31.47
30.36
31.55
NDCG@10
14.03
15.78
11.36
12.59
11.86
16.97
20.27
19.45
20.09
Domain Negotiation strategy to alleviate gradient conflicts during multi-domain training. Additionally, Domain Regularization is applied to enhance the generalizability to the other domains. In our work, we select SASRec as the backbone for multi-domain recommendation since MAMDR is a model-agnostic method. As our method will degrade or transform into other equivalent ext-based baselines (e.g.Recformer [14]) under different settings, we, thereby analyze their performance conceretly in Section 4.3.

Domain Negotiation strategy to alleviate gradient conflicts during multi-domain training. Additionally, Domain Regularization is applied to enhance the generalizability to the other domains. In our work, we select SASRec as the backbone for multi-domain recommendation since MAMDR is a model-agnostic method.

As our method will degrade or transform into other equivalent text-based baselines (e.g.Recformer [14]) under different settings, we, thereby analyze their performance conceretly in Section 4.3.

4.1.3 Evaluation Metrics. We use Recall@N and NDCG@N for recommendation performance evaluation, where 𝑁 = 1 and 10. Compared with Recall, NDCG further takes the position of retrieved positive item into account and assign higher scores to positive item that are ranked higher in the retrieved results. For all the datasets, we use negative sampling for evaluation, i.e., for each target item we randomly sample 1, 000 instances from the domain of target item as negatives.

4.1.4 Implementation Details. All methods are implemented using Pytorch with an Adam [13] optimizer. Additionally, all baselines’ hyperparameters are configured according to the original suggestion from their papers. The language models mentioned in our LLM-Rec are from Huggingface 234. We set the maximum length of the interaction sequence to 10 and the max token numbers of item title to 40. The learning rate of the OPT, BERT and their variants are 5 𝑒 − 5, and 3 𝑒 − 4 for FLAN-T5 and its variants. The batch size is set to 96 for all models. We check the validation performance every

2 https://huggingface.co/docs/transformers/model_doc/flan-t5 3 https://huggingface.co/docs/transformers/main/model_doc/opt 4 https://huggingface.co/docs/transformers/main/model_doc/bert

3, 000 steps and adopt early-stop strategy for our model training, i.e., the training process will be terminated when the performance of the validation dataset is not be improved after ten consecutive rounds.

# 4.2 Overall Comparison

In Table 4, we report the results of baselines and our LLM-Rec with different backbones in five domains as well as the mixed SPIAO dataset. From the experimental results, we could draw the following conclusions: 1) As an extension of LightGCN in multi-domains, BiTCGF+ outperforms LightGCN, which demonstrates the effectiveness of multi-domain information for recommendation. However, BiTGCF+ is inferior to single domain sequential methods (e.g., SASRec). It illustrates that it is non-trivial to apply graph-based models for sequential recommendation as they could not capture the longterm dependency in sequences effectively. 2) Compared to SASRec, MAMDR acquires stable improvements in all domains proving the effectiveness of multi-domain information. 3) Besides, different variants of LLM-Rec achieve the optimal results, presenting the superiority of language models for the multi-domain recommendation. 4) Moreover, we find that the results of LLM-Rec will fluctuate slightly under different model structures, although their model sizes are close. In the following sections, we will analyze the impact of cross-domain data, language model size and fine-tuning methods in-depth.

# 4.3 The Impact of Cross-domain Data

4.3.1 Without Cross-Domain Data.  To verify the benefits of crossdomain interactions, we further analyze the performance changes with and without the cross-domain records during training. To be

Tuning
Methods
Metrics
BERT
OPT
FLAN-T5
40M
110M
330M
125M
350M
1.3B
2.7B
6.7B
30M
110M
385M
1.5B
5.5B
Full
Parameter
Recall@1
10.22
11.25
11.44
10.60
10.43
10.61
11.16
11.44
8.17
10.85
10.97
11.41
11.56
Recall@10
29.80
31.47
31.95
30.37
29.74
30.60
30.69
31.46
26.61
31.55
31.44
32.22
32.57
Parameter
Efficient
Recall@1
6.69
7.84
9.38
7.72
6.85
10.69
10.84
10.83
0.28
6.70
7.81
7.71
9.28
Recall@10
22.12
25.06
28.40
25.45
23.07
30.71
31.35
31.38
2.10
23.73
25.64
25.83
28.88
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a3c7/a3c70ca3-201f-446d-b9ba-de1674c9f8ff.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Without cross-domain training data.
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1b58/1b5841f8-9ba9-4304-ac45-8d5fda42cf45.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: The relative improvement under different sequenc settings (Best viewed in color)
</div>
specific, we first organize the user interaction sequence for each domain (e.g., Scientific, Pantry, Instruments, etc.) and then mix them together as the multi-domain dataset denoted as S 2 PIAO. It is noteworthy that in S 2 PIAO each user will contain several interaction sequences from different domains, thus, for each sequence, there is no cross-domain information sharing, while for SPIAO, each individual will only obtain one unique sequence formed by all domains’ interaction behaviors along the timeline. For in-depth analysis, we denote the S 2 PIAO mixing method as the domainoriented mix strategy and denote SPIAO as the user-oriented mix strategy. Then, we retrain our model with single-domain datasets (e.g., Scientific, Pantry, Instruments, and so on and so forth.) and multi-domain dataset (S 2 PIAO) and SPIAO, respectively, and compare their performance under the same test dataset. The result is shown in Figure 4. Observing Figure 4, we can find that the SPIAO achieves the best result and single-domain performs the worst, indicating that a simplified mixing solution (i.e., domain-oriented mix strategy) will be effective for recommendation. And finer grain mixing (i.e., user-oriented mixing) can further improve the model performance. We believe this improvement comes from the world knowledge of pre-trained LLM. And richer and more detailed description of user sequence will reveal more information.

4.3.2 Performance Improvement under Different Sequence Settings. To further investigate the performance of our LLM-Rec, we divide the test set into three categories based on the relationship of target item domain and sequence item domain and compare the performance between LLM-Rec against SASRec under these three settings

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/01c6/01c6f2ce-6448-4bb6-b8aa-1b5f7f83a09c.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Zero-shot domain performance.
</div>
as below, it should be noted the model is still trained on the mixed SPIAO dataset, but only the test set is seperated for evaluation:
• Same: the target item and the items in the sequence are all form the same domain.
• Mix: a part of items in the sequence is from the target item domain, while the left is from other domains.
• Diff: the domains of all the items in the sequence are orthogonal to the target item’s domain.

Figure 5 shows the results, we can find that the relative improvement of Diff setting is superior to the Mix significantly, which in turn is superiority to the Same. This observation supports our assumption that as a new alternative of recommendation solution, LLM could achieve competitive results compared with conventional ID-based sequential modeling methods. Furthermore, the significant improvement of LLM-Rec on cross-domain data further verifies our idea that world knowledge embeded in pre-trained language model can facilitate multi-domain knowledge transfer.

# 4.4 The Impact of Language Model Size

Noting that a larger pre-trained base model will be competent for more complicated downstream tasks and acquire significant improvement in NLP and CV fields. In Section 4.2, we have already manifested that 110 𝑀 around parameters could achieve good performance in most scenarios. Thus, in this section, we further investigate the impact of model size on the final performance. To this end, we dedicate various model size for different backbones, i.e., ranging from 40 𝑀 BERT-Medium to 6. 7B OPT model. Table 5 presents the results across different sizes.

4.4.1 Fine-tune Domain Performance. Observing Table 5 we could find that increasing parameters the performance will acquire a huge gain when the model size is rather limited, i.e., 10% and 33% relative increase in Recall@1 when the size is expanded from 40 𝑀 to 110 𝑀 for BERT and 60 𝑀 to 110 𝑀 for FLAN-T5, respectively. However, this trend will not remain when the model size exceeds 100 𝑀. For instance, in terms of OPT, the minimum size (125 𝑀) presumably be sufficient for such recommendation tasks, and when we increase the model size to 6. 7 𝐵, the performance improvement is relatively small. A similar phenomenon could also be observed in the BERT and FLAN-T5 model, i.e., BERT-330 𝑀 does not show great superiority to the BERT-110 𝑀, so as to the FLAN-T5 (ten times increasing only achieves 2% to 5% improvements). Consequently, we believe it will be difficult to acquire a very huge performance improvement when the model achieves a certain size (100 𝑀 around), which is quite different to NLP.
4.4.2 Zero-shot Domain Performance. Considering that the model size can also facilitate the generalization ability of LLM. We further analyze the zero-shot performance with different model sizes on unseen domains. Specifically, we select another two Amazon subcategories: "Grocery and Gourmet Food" and "Home and Kitchen" 5, which do not be used for model training. Experimental results are reported in Figure 6. We could find that as model size increases, the zero-shot performance can be greatly improved, except OPT-350M 6. More encouraging, on Food dataset, OPT-6.7B even outperforms SASRec, a strong baseline that is well-trained via item ID. Moreover, we find that the model structures will have overwhelming impact on zero-shot performance against model size, i.e., the performance of OPT is much better than FLAN-T5 when their model size is close. Analyzing the results of zero-shot experiments, we argue the conclusion that "bigger is better" can not match well in recommendation scenario’s might be as follows: 1) different from NLP and CV tasks, the complicated patterns encapsulated in the collaborative filtering signals might be the recipe of success in recommendation system. 2) However, model parameters seem to play a minor role in collaborative filtering signals modeling when it comes to a certain size, a bigger model will not making learning the collaborative filtering signal easier.

4.4.1 Fine-tune Domain Performance. Observing Table 5 we could find that increasing parameters the performance will acquire a huge gain when the model size is rather limited, i.e., 10% and 33% relative increase in Recall@1 when the size is expanded from 40 𝑀 to 110 𝑀 for BERT and 60 𝑀 to 110 𝑀 for FLAN-T5, respectively. However, this trend will not remain when the model size exceeds 100 𝑀. For instance, in terms of OPT, the minimum size (125 𝑀) presumably be sufficient for such recommendation tasks, and when we increase the model size to 6. 7 𝐵, the performance improvement is relatively small. A similar phenomenon could also be observed in the BERT and FLAN-T5 model, i.e., BERT-330 𝑀 does not show great superiority to the BERT-110 𝑀, so as to the FLAN-T5 (ten times increasing only achieves 2% to 5% improvements). Consequently, we believe it will be difficult to acquire a very huge performance improvement when the model achieves a certain size (100 𝑀 around), which is quite different to NLP.

4.4.2 Zero-shot Domain Performance. Considering that the model size can also facilitate the generalization ability of LLM. We further analyze the zero-shot performance with different model sizes on unseen domains. Specifically, we select another two Amazon subcategories: "Grocery and Gourmet Food" and "Home and Kitchen" 5, which do not be used for model training. Experimental results are reported in Figure 6. We could find that as model size increases, the zero-shot performance can be greatly improved, except OPT-350M 6. More encouraging, on Food dataset, OPT-6.7B even outperforms SASRec, a strong baseline that is well-trained via item ID. Moreover, we find that the model structures will have overwhelming impact on zero-shot performance against model size, i.e., the performance of OPT is much better than FLAN-T5 when their model size is close. Analyzing the results of zero-shot experiments, we argue the conclusion that "bigger is better" can not match well in recommendation scenario’s might be as follows: 1) different from NLP and CV tasks, the complicated patterns encapsulated in the collaborative filtering signals might be the recipe of success in recommendation system. 2) However, model parameters seem to play a minor role in collaborative filtering signals modeling when it comes to a certain size, a bigger model will not making learning the collaborative filtering signal easier.

# 4.5 The Impact of Tuning Method

In Section 4.4, we show the results of full parameter fine-tuning (FPFT) for the pre-trained language model. However, as the model size increases, the GPU memory requirements for FPFT increase dramatically. For instance, fine-tuning the OPT-1.3B pre-trained model with AMP (Automatic Mixed Precision) requires at least 20GB GPU memory. To circumvent this issue, parameter efficient fine-tuning (PEFT), i.e., fine-tuning very small groups of parameters to approach the performance of full parameter tuning, became a prominent solution and also a cutting-edge research problem for LLM fine-tuning very recently. We, thereby, apply a widely-used PEFT method, i.e., LoRA [11] with low-rank 𝑅 = 32, for model

5 Due to the computing source limitation, we sample 30% sequences from the Food and Home datasets. 6 Researchers in the NLP community have also found that the performance of OPT350M is poor. The reason given by HuggingFace members is that there is a bug with the 350M weights.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/dbe9/dbe9200c-e1e7-4fee-920f-54990df3824a.png" style="width: 50%;"></div>
<div style="text-align: center;">e impact of trainable parameters in parameter
</div>
fine-tuning and analyzing the impact of tuning methods on the final performance.

4.5.1 Parameter Efficient Fine-tuning Results.  According to the results of Table 5, we can find that compared with FPFT, the performance of LoRA dropped substantially when the parameter is less than one billion. However, as the model size increase, the gap between them decreases progressively, which suggests that it is essential to use FPFT for better recommendation performance when the model size is relatively small. For models larger than 1B, PEFT is a viable option for more efficient model optimization for OPT series models, but the FLAN-T5 series models still have a significant decline. Overall, we can draw the following conclusion: for small models(<1B), PEFT is not suitable for the recommendation domain; for large models, the effectiveness of PEFT is related to the model itself.
4.5.2 The Impact of Trainable Parameters. To further explore the impact of the number of trainable parameters on the performance in PEFT, we valid the recommendation performance of different 𝑅 values in LoRA, the experimental results are shown in Figure 7. From the results, we can draw the following conclusions: 1) Overall, the model will present a better performance as trainable parameters increase. 2) Besides, although we increase the number of trainable parameters, PEFT is still inferior to the FPFT until the model size rises to one billion. We believe the reason might be that the recommendation tasks (i.e., next-item prediction) and relevant corpus are deviated from NLP scenarios (e.g., text generation). Meanwhile, the generalization ability of small model is relatively limited, thus, the benefit of fine-tuning a small number of parameters for small model will be little.
4.5.3 Training Cost. Given that the model size of LLM significantly

4.5.1 Parameter Efficient Fine-tuning Results.  According to the results of Table 5, we can find that compared with FPFT, the performance of LoRA dropped substantially when the parameter is less than one billion. However, as the model size increase, the gap between them decreases progressively, which suggests that it is essential to use FPFT for better recommendation performance when the model size is relatively small. For models larger than 1B, PEFT is a viable option for more efficient model optimization for OPT series models, but the FLAN-T5 series models still have a significant decline. Overall, we can draw the following conclusion: for small models(<1B), PEFT is not suitable for the recommendation domain; for large models, the effectiveness of PEFT is related to the model itself.

4.5.2 The Impact of Trainable Parameters. To further explore the impact of the number of trainable parameters on the performance in PEFT, we valid the recommendation performance of different 𝑅 values in LoRA, the experimental results are shown in Figure 7. From the results, we can draw the following conclusions: 1) Overall, the model will present a better performance as trainable parameters increase. 2) Besides, although we increase the number of trainable parameters, PEFT is still inferior to the FPFT until the model size rises to one billion. We believe the reason might be that the recommendation tasks (i.e., next-item prediction) and relevant corpus are deviated from NLP scenarios (e.g., text generation). Meanwhile, the generalization ability of small model is relatively limited, thus, the benefit of fine-tuning a small number of parameters for small model will be little.

4.5.3 Training Cost. Given that the model size of LLM significantly surpasses that of the conventional method, the training cost will be an inevitable challenge and bottleneck of model optimization. We, hereby, present the training time of OPT model (size over one billion) with FPFT and PEFT on the SPIAO dataset. Additionally, we set the batch size to 1 and investigate the minimum memory resource consumption (video memory, VMEM) of FPFT and PEFT (LoRA) under different large model sizes, as shown in Table 6. We can observe that it is infeasible to FPFT OPT-6.7B model with one A100-80G machine. Compared with FPFE, PEFT occupies less VMEM, while FPFT obtains faster convergence speed if there are sufficient computational resources. We believe this is because the pre-training corpus of the language model and the recommendation corpus is quite different, thus, it will require sufficient time to fine-tune a very small number of parameters with LoRA for convergence.

Table 6: Training cost of large different models on 6*A10080G. "h" represents hour, "DDP" represents distributed data parallel, " 𝑍𝑒𝑟𝑜 2" means deepspeed zero stage 2 parallel method and OOM represents out of memory in single A10080G. Average training sequence token length of OPT is 122.

Model
Method
Time
Strategy
Min-VMEM
OPT-1.3B
FPFT
48h
DDP
28G
PEFT
109h
DDP
13G
OPT-2.7B
FPFT
61h
𝑍𝑒𝑟𝑜2
55G
PEFT
81h
DDP
24G
OPT-6.7B
FPFT
79h
𝑍𝑒𝑟𝑜2
OOM
PEFT
152h
DDP
56G
# 5 CONCLUSION

In this paper, we propose a multi-domain sequential recommendation framework based on large language model to alleviate the data sparsity and cold-start problems. By representing item and users with item title and concatenation of interacted item titles, we regard the recommendation task as the next sentence prediction task and utilize pre-trained language model for generating corresponding representations. Furthermore, we mix user’s interactions across different domain to capture multi-domain knowledge. The experimental results on five real-world datasets demonstrate the effectiveness of LLM-Rec and the multi-domain information. Additionally, we investigate the impact of model size and tuning methods when applying large language model in recommendation. The experimental results show that as the size of pre-trained language model increases, the fine-tune domain recommendation performance will increase slightly, while the zero-shot domain recommendation performance increase significantly. Moreover, the PEFT method fails to achieve promising performance when is model size is relatively small, and FPFT method is more efficient given sufficient computational resources.

# REFERENCES

[1]  Jiangxia Cao, Xin Cong, Jiawei Sheng, Tingwen Liu, and Bin Wang. 2022. Contrastive Cross-Domain Sequential Recommendation. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 138–147.
[2] Yongjun Chen, Zhiwei Liu, Jia Li, Julian J. McAuley, and Caiming Xiong. 2022. Intent Contrastive Learning for Sequential Recommendation. In Proceedings of the ACM Web Conference 2022. 2172–2182.
[3] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022).
[4] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. 2023. Scaling vision transformers to 22 billion parameters. In International Conference on Machine Learning. PMLR, 7480–7512.
[5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).
[6]  Hao Ding, Yifei Ma, Anoop Deoras, Yuyang Wang, and Hao Wang. 2021. Zeroshot recommender systems. arXiv preprint arXiv:2105.08318 (2021).
[7] Xiaobo Hao, Yudan Liu, Ruobing Xie, Kaikai Ge, Linyao Tang, Xu Zhang, and Leyu Lin. 2021. Adversarial feature translation for multi-domain recommendation. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2964–2973.
[8] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-Dong Zhang, and Meng Wang. 2020. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 639–648.
[9] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939 (2015).
[10] Yupeng Hou, Shanlei Mu, Wayne Xin Zhao, Yaliang Li, Bolin Ding, and Ji-Rong Wen. 2022. Towards Universal Sequence Representation Learning for Recommender Systems. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 585–593.
[11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021).
[12]  Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM). IEEE, 197–206.
[13]  Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).
[14] Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and Julian McAuley. 2023. Text Is All You Need: Learning Language Representations for Sequential Recommendation. arXiv preprint arXiv:2305.13731 (2023).
[15]  Meng Liu, Jianjun Li, Guohui Li, and Peng Pan. 2020. Cross domain recommendation via bi-directional transfer graph collaborative filtering networks. In Proceedings of the 29th ACM international conference on information & knowledge management. 885–894.
[16] Weiming Liu, Xiaolin Zheng, Chaochao Chen, Jiajie Su, Xinting Liao, Mengling Hu, and Yanchao Tan. 2023. Joint Internal Multi-Interest Exploration and External Domain Alignment for Cross Domain Sequential Recommendation. In Proceedings of the ACM Web Conference 2023. 383–394.
[17] Linhao Luo, Yumeng Li, Buyu Gao, Shuai Tang, Sinan Wang, Jiancheng Li, Tanchao Zhu, Jiancai Liu, Zhao Li, and Shirui Pan. 2023. MAMDR: A Model Agnostic Learning Framework for Multi-Domain Recommendation. In 2023 IEEE 39th International Conference on Data Engineering (ICDE). IEEE, 3079–3092.
[18] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H. Chi. 2018. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixtureof-Experts. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1930–1939.
[19]  Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized markov chains for next-basket recommendation. In Proceedings of the 19th international conference on World wide web. 811–820.
[20] Xiang-Rong Sheng, Liqin Zhao, Guorui Zhou, Xinyao Ding, Binding Dai, Qiang Luo, Siran Yang, Jingshan Lv, Chi Zhang, Hongbo Deng, and Xiaoqiang Zhu. 2021. One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction. In The 30th ACM International Conference on Information and Knowledge Management. 4104–4113.
[21] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management. 1441–1450.
[22]  Hongyan Tang, Junning Liu, Ming Zhao, and Xudong Gong. 2020. Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for

Personalized Recommendations. In RecSys 2020: Fourteenth ACM Conference on Recommender Systems. 269–278.
[23]  Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the eleventh ACM international conference on web search and data mining. 565–573.
[24] Zuoli Tang, Lin Wang, Lixin Zou, Xiaolu Zhang, Jun Zhou, and Chenliang Li. 2023. Towards Multi-Interest Pre-training with Sparse Capsule Network. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 311–320.
[25] Xu Xie, Fei Sun, Zhaoyang Liu, Shiwen Wu, Jinyang Gao, Jiandong Zhang, Bolin Ding, and Bin Cui. 2022. Contrastive Learning for Sequential Recommendation. In 38th IEEE International Conference on Data Engineering. 1259–1273.
[26] Zixuan Xu, Penghui Wei, Shaoguo Liu, Weimin Zhang, Liang Wang, and Bo Zheng. 2023. Correlative Preference Transfer with Hierarchical Hypergraph Network for Multi-Domain Recommendation. In Proceedings of the ACM Web Conference 2023. 983–991.
[27] Zheng Yuan, Fajie Yuan, Yu Song, Youhua Li, Junchen Fu, Fei Yang, Yunzhu Pan, and Yongxin Ni. 2023. Where to go next for recommender systems? id-vs.

modality-based recommender models revisited. arXiv preprint arXiv:2303.13835 (2023).
[28] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068 (2022).
[29] Yu Zhang, Bin Cao, and Dit-Yan Yeung. 2010. Multi-Domain Collaborative Filtering. In UAI 2010, Proceedings of the Twenty-Sixth Conference on Uncertainty in Artificial Intelligence. 725–732.
[30] Cheng Zhao, Chenliang Li, and Cong Fu. 2019. Cross-Domain Recommendation via Preference Propagation GraphNet. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2165–2168.
[31] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In  Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 1893–1902.

