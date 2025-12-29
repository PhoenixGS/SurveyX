# DISTRIBUTED IN-CONTEXT LEARNING UNDER NON-IID AMONG CLIENTS
<div style="text-align: center;">DISTRIBUTED IN-CONTEXT LEARNING UNDER NON-IID AMONG CLIENTS</div>
Siqi Liang∗
Michigan State University
East Lansing, Michigan, USA
liangsi4@msu.edu
Sumyeong Ahn∗
Michigan State University
East Lansing, Michigan, USA
sumyeong@msu.edu
Jiayu Zhou†
Michigan State University
East Lansing, Michigan, USA
jiayuz@msu.edu
# ABSTRACT
Advancements in large language models (LLMs) have shown their effectiveness in multiple complicated natural language reasoning tasks. A key challenge remains in adapting these models efficiently to new or unfamiliar tasks. In-context learning (ICL) provides a promising solution for few-shot adaptation by retrieving a set of data points relevant to a query, called in-context examples (ICE), from a training dataset and providing them during the inference as context. Most existing studies utilize a centralized training dataset, yet many real-world datasets may be distributed among multiple clients, and remote data retrieval can be associated with costs. Especially when the client data are non-identical independent distributions (non-IID), retrieving from clients a proper set of ICEs needed for a test query presents critical challenges. In this paper, we first show that in this challenging setting, test queries will have different preferences among clients because of non-IIDness, and equal contribution often leads to suboptimal performance. We then introduce a novel approach to tackle the distributed non-IID ICL problem when a data usage budget is present. The principle is that each client’s proper contribution (budget) should be designed according to the preference of each query for that client. Our approach uses a data-driven manner to allocate a budget for each client, tailored to each test query. Through extensive empirical studies on diverse datasets, our framework demonstrates superior performance relative to competing baselines.
arXiv:2408.00144v1
# 1 Introduction
Recent significant progress in large language models (LLMs) [1, 2, 3, 4] has demonstrated their effectiveness acro various natural language processing (NLP) tasks [5, 6]. Despite their impressive performances, they still requ adaptation to the specific downstream tasks for better performance. However, adaptation poses challenges due to LLM vast number of trainable parameters.
adaptation to the specific downstream tasks for better performance. However, adaptation poses challenges due to LLMs’ vast number of trainable parameters. In-context learning (ICL) [7] is a notable method that distinguishes itself through both its effectiveness and efficiency. In brief, ICL adapts to the target task by incorporating context information following two primary steps: i) identify samples from the training dataset helpful to solve the target query by creating a prompt describing a context; ii) feed the constructed prompt with the target query and get the answer. Previous related works on ICL mainly have focused on the construction of a prompt describing the context, which involves several sub-problems, such as the retrieval of in-context examples (ICEs) [8] and determining the optimal sequence for the selected ICEs [9]. A common assumption in most existing ICL research is that the system has access to a high-quality centralized dataset used for retrieval. However, in many application scenarios, such as health informatics, centralized datasets may not be feasible, and data could be distributed in different institutions, which calls for the distributed ICL. In addition, when the data is proprietary and possesses high value towards inferences, access to data entries may also be bound to data pricing strategies [10, 11]. For instance, the system needs to pay the local institution based on the number of samples sent to the system [12] as a means to share profits from inferences. Under this scenario, aggregating ICEs from local clients to a center server for ICL entails significant financial costs and lacks efficiency.
∗Equal contribution. †Corresponding author.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a0cf/a0cfa372-1b7c-4c3b-97b0-4a0e2c6a21e2.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Problem overview. When datasets are distributed among clients in a non-IID manner, it creates an obstacle in generating a good context (left). However, by assigning appropriate budgets to leverage per-client expertise, better context can be created (right).</div>
In this paper, we focus on integrating knowledge from distributed clients to achieve better ICL performance under the per-query ICE budget constraint. Specifically, we formalize the distributed ICL problem where the ICEs are distributed on clients, and the server has an LLM for ICL inference but can only request a limited number of ICEs from all clients for each query, which we refer to as the ICE budget. We begin by identifying the key challenge in distributed ICL with ICE budget constraints lies in the non-independently and identically distributed (non-IID) training data, as shown in Section 3.1. For example, in Figure 1, data samples are spread across C clients, each with a unique data distribution. Specifically, client 1 primarily contains (+) samples, while client 2 is mainly constituted by (−) examples. Only limited research [13] tried to address the challenge of distributed datasets for ICL, while none considers the challenging real-world setting of non-IID clients. This leaves a critical question unanswered: What happens to distributed ICL when local clients are non-IID? To further the understanding of the key challenge in the distributed non-IID ICL, we explore the local retrieval process on non-IID clients. We found that each query has different preferences for different clients based on local knowledge distribution, that is, the number of samples needed from different clients should vary based on local sample distribution. As the toy example shown in Figure 1, when the server creates context by uniformly assigning budgets to clients, the answer might be incorrect due to the insufficiency of (+) information in the context. To be more detailed, the server assigns the clients who have expertise on (−), (×), and (÷) operations with the same budget as on (+), without any preference. Nevertheless, if the server assigns more budget to clients with many (+) samples, such as client 1, it can create a more relevant context to answer the query related to (+) operation. This indicates that under non-IID, the server should allocate the budgets over clients based on the preference of each query itself, as well as the distribution of local training samples. Motivated by this, we propose a novel distributed ICL framework to collaboratively collect scattered information among non-IID clients by properly assigning ICE budgets to each client. First, the server will gather the optimal budget statistics using an existing proxy dataset on the server side. Next, the server will use this dataset to train the budget allocator. During the deployment stage, the server will predict the proper budget for each client using this trained budget allocator given each test query and perform ICL among clients. Furthermore, in practical scenarios where privacy concerns arise, we augment our framework with the paraphrasing method [13] to secure privacy.
# Contributions. A summary of our contributions:
• To the best of our knowledge, we are the first to study the challenging real-world setting of ICL with distributed non-IID clients. We identify the principal challenge as properly assigning the ICE budget for non-IID clients based on the preference of each test query and local knowledge distribution. • We propose a framework to handle the distributed non-IID ICL. This framework trains a budget allocator on the server with the help of a server-side proxy dataset. Then, the server will use this trained allocator to decide how many ICEs to retrieve from each client for the ICL process, enabling collaborative action among clients. • Across a range of dataset benchmarks featuring various non-IID configurations as well as on different LLM architectures, our approach has been validated to enhance ICL performance. Notably, we examine both non-private,
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d247/d2473ce4-eba8-4b98-a49c-56197f636af9.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Overview of the pipeline: First, the budget allocator assigns a budget to each client based on the question. Subsequently, each client retrieves their relevant samples and sends them back to the server. The server infers the answer by feeding the question, which is composed of concatenated context examples and the query.</div>
i.e., communicate raw samples directly, and private cases using the paraphrasing method to secure privacy. In both scenarios, our approach shows superiority to the previous method and other reasonable baselines.
# 2 Problem Formulation
In this section, we provide a detailed problem formulation. First, we begin with the specifics of in-context learning (ICL), followed by a description of distributed non-IID ICL.
# 2.1 In-Context Learning
Notation. We consider a NLP tasks which have training dataset D = {(xi, yi)}N i=1 with N training samples. Here, x is the input text, and yi is the corresponding output. In the test phase, a test query xq is given.
Retrieval. We employ the off-the-shelf pre-trained retriever KATE [14]3, which utilizes k-NN example selection. This retriever employs a sentence encoder E(·) to measure the similarity between the in-context example xi in dataset D and the query xq as follows:
 ∥ −∥ where eq = E(xq) and ei = E(xi). We select k samples using the following criterion: T (e, k|D) = arg Top-k(d(e, e)),
 ∥ −∥ where eq = E(xq) and ei = E(xi). We select k samples using the following criterion:
where T (eq, k|D) denotes the selected samples from the dataset D, and used for inference. ICL Inference. In the test phase, given a test query with input xi, relevant k training samples called in-context examples (ICEs) are selected, i.e., S = T (eq, k|D). Based on the retrieved samples, we feed the constructed context prompt s(S, xq) into LLM for inference and obtain results via:
ICL Inference. In the test phase, given a test query with input xi, relevant k training samples called in-context examples (ICEs) are selected, i.e., S = T (eq, k|D). Based on the retrieved samples, we feed the constructed context prompt s(S, xq) into LLM for inference and obtain results via:
 ⊙ ⊙ ⊙ where the ⊙operation denotes concatenation, and s(S, xq) is the context constructed using query xq and samples in S; the term pLLM represents the output softmax probability of the LLM, functioning autoregressive, meaning that the output up to time t, i.e., y<t, is input back into the model to generate the tth output, yt. Previous works [15, 16] on ICL mainly focus on the selection of S under a centralized setting. However, we investigate the scenario where D is split among several clients, each following non-IID distributions.
# 2.2 Distributed non-IID ICL
Distributed ICL Setting. We consider C clients with a centralized server in our system. Each client c ∈[C] has loca training dataset Dc = {(xc i, yc i )}Nc i=1 with Nc training samples. Note that Dc follows different distributions for differen
Distributed ICL Setting. We consider C clients with a centralized server in our system. Each client c ∈[C] has local training dataset Dc = {(xc i, yc i )}Nc i=1 with Nc training samples. Note that Dc follows different distributions for different 3We do not fine-tune the retriever for each task, which is impractical because we cannot gather the distributed datasets.
(1)
(2)
(3)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3aea/3aea73da-ff5f-49e8-a682-7f837131ce49.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Non-IID experimental results. It shows that centralized performance is comparable to the IID case, wherea non-IID scenarios exhibit a significant declined performance. This highlights the critical importance of addressin non-IIDness to find a solution.</div>
clients. We follow the non-IID conditions as defined in [17], with details provided in Appendix A. In summary, we allocate data on a per-class basis, where each client receives a specific number of classes, meaning each client has samples from only specified classes. Clients and the server have identical off-the-shelf pre-trained retrievers. Consider the computation resource limitation on clients as in many real scenarios [18], only the server is equipped with an LLM. Moreover, the server has limited proxy dataset Dproxy = {(xproxy j , yproxy j )}Nproxy j=1 , that Nproxy ≪�C c=1 Nc. The server has quite a small Dproxy, and it is an auxiliary dataset to extract information for collaboration to make the problem feasible. Pipeline. First, the server requests relevant samples from each client by sending xq to all clients with local budgets kc. Remark that each query xq has its own preference of each client, which can be represented as kc. Here, xq can be anonymized by paraphrasing, as done in previous works [13]4. Each client then selects the most relevant kc samples from their local training dataset, i.e., Sc = T (eq, kc|Dc) ⊂Dc, and returns them to the server. The server receives Sc from clients and generates the context s based on the merged examples, S = �C c=1 Sc. In the final step, the server infers y using s(S, xq). The entire framework also can be described in Figure 2. In this paper, we are concentrating on assigning kc to each client as described in Figure 2.
# 3 Observations
In this section, we describe several empirical supports to handle the distributed non-IID ICL. First, we demonstrate that non-IID distributions hinder the merging of scattered information. We then establish our goal, termed as oracle budget, which reflects the server’s preference for each client if the server knows all distributed data. Finally, we check i predicting the oracle budget of each test query for inference is feasible.
# 3.1 Non-IIDness Leads to Performance Drop
First of all, we evaluate the effect of non-IIDness. Straightforwardly, we distribute the budget {kc}C c=1 uniformly according to the following criteria: Given C clients are involved in answering this question, and the number of samples for context is k. We first explore the naïve equally assigned local budget scheme in both IID and non-IID settings. That is, each client c ∈[C] locally retrieves top-kc samples where kc = ⌈k C ⌉from local dataset Dc. Detailed experimental settings are described in Appendix B. As illustrated in Figure 3, we observe the followings: (1) There is no significant performance degradation between the centralized case (■) and the IID case (■). This is expected, as the merged top-kc samples in the IID case closely resemble the centralized top-k samples. Any minor discrepancies are attributed to differences in sample ordering. (2) However, performance degradation becomes pronounced in non-IIDness case (refer to the comparison between ■, ■ and ■). Hereinafter, we gather insights to address the distributed non-IIDness ICL.
# 3.2 Proper Budget per Query for Each Client
Oracle budget. The remaining issue is that to make the server operate similar with the centralized manner, it needs to allocate the budget as if it knows complete knowledge of all clients. We call this budget for each client as the oracle budget for query embedding eq and define it as follows: � �
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f155/f1553fd6-40a8-49c4-b368-a49f8001d9a4.png" style="width: 50%;"></div>
Algorithm 1 Top-k sampling, T (e, k|D)
Require: Query embedding e, Encoder E(·)
/* Compute embedding */
1: for (xi, yi) ∈D do
2:
ei ←E(x)
3: end for
/* Select top-k samples */
4: S =
(xi,yi)∈D
arg Top-k∥e −ei∥2
5: return S
Check of predictability of oracle budget. For the next step, it is necessary to check if eq has sufficient patterns of oracle budget to extract and use it in the inference phase. Our hypothesis is that similar queries may share similar oracle budget patterns and preferences on the same client, and it can lead to similar budget allocations for that client. Therefore, to verify this hypothesis, we perform t-SNE analysis [19] on the embeddings obtained from the retriever for queries. Furthermore, we color each sample based on the oracle budget k⋆ c(eq). As described in Figure 4, similar query embeddings exhibit similar oracle budget patterns. This indicates that, given a test query, we can infer the budget assignment for each client. However, it is challenging to predict fine-grained budget value since there are no rigid classification patterns. For instance, determining the detailed budget value seems challenging in the case of client 1 in SST-5. Therefore, developing an efficient method to infer the exact budgets based on these broad patterns for each client are required.
# 3.3 Observation Summary
In summary, our findings and the approach for designing an algorithm are as follows: (1) non-IIDness significantly affects the distributed ICL setting, necessitating the development of a coalition method. To handle this problem, it is straightforward to allocate an appropriate number of budgets to each client, i.e., making server work so as it knows client all samples. (2) By analyzing the query embeddings, we can determine the importance of each client per query.
# 4 Method
In this section, we outline the proposed algorithm to mitigate non-IIDness in the ICL framework. Specifically, we show how to train the budget allocator and conduct inference.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8b2a/8b2afc80-fc1f-4e37-92cf-186558df25dc.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Overview of the budget allocator: We train a budget allocator on top of the frozen feature extractor E, which inherits from the retriever. During inference, when a test query xq is provided, this module determines the quantized budget levels for each client and allocates them accordingly.</div>
Algorithm 2 Construct dataset
Require: Encoder E(·), server-side ICE budget k, proxy dataset Dproxy = {(xj, yj)}Nproxy
j=1 Quantization parameter δ.
1: for (xproxy
j
, yproxy
j
) ∈Dproxy do
2:
eproxy
j
= E(xproxy
j
)
/* Get distributed examples */
3:
for c ∈[C] do
4:
eproxy
j
→Client c
5:
Sc = T (eproxy
j
, k|Dc)
6:
Server ←Sc
7:
end for
/* Construct optimal example */
8:
S = �C
c=1 Sc
9:
Stop =
(xs,ys)∈S
arg Top-k∥eproxy
j
−E(xs)∥2
/* Compute proper budget size for each c */
10:
kc(ej) = |Stop ∩Sc|//δ
∀c ∈[C]
11: end for
12: Bproxy = {(ej, {kc(ej)}C
c=1)}Nproxy
j=1
13: return Bproxy
# 4.1 Train a Budget Allocator
Based on Section 3, it is feasible to assign budgets of each client by using the embeddings obtained from the retriever encoder E. We first construct the datasets having the targeting budget values and then train the budget allocator.The pseudo-codes are described in Algorithm 1 and 2. Construct dataset for oracle budget. First, we explain how to create a dataset to train the budget allocator for each client, as described in Algorithm 2. Given proxy dataset Dproxy, for all embeddings ej = E(xj) where (xj, yj) ∈Dproxy, we request k samples from each client c ∈[C] using Top-k procedure, i.e., Sc = T (e, k|Dc). Once the server receives k examples from each clients, i.e., {Sc}C c=1, it merges and re-orders them to obtains Stop. Based on Stop, we count the number of samples from each client in Stop, i.e., compute kc(ej). After counting kc(ej) for all clients, we quantize the budget levels for each client using the quantization hyper-parameter δ. As a result, the output of this procedure is Bproxy for all clients, composed of embeddings e and their respective budgets kc(ej).
Algorithm 3 Inference (Client, Server)
Require: Embedding model E(·), LLM model M(·), local datasets Dc, budget allocator fc(·) Buffering hyperparame-
ter α.
Input: Test query xq
1: Extract embedding eq = E(xq)
2: for c ∈[C] do
3:
ˆkc = fc(eq)
4:
Send eq to all clients
5:
Sc = T (eq, ˆkc + α|Dc)
6:
return back Sc →Server
7: end for
8: Sagg = �
c∈[C] Sc
9: S = T (eq, k|Sagg)
10: s(S, xq) = (x1, y1) ⊙... ⊙(xk, yk) ⊙xq
11: return y = M(s(S, xq))
Train budget allocator. Based on the constructed dataset Bproxy, we train the budget allcoators, i.e., {fc(·)}C c=1, for each fc(·) has Multi-layer perceptrons on top of the frozen feature extractor of the off-the-shelf retriever E. The budget allcoators are trained on the cross-entropy loss, as we have already quantized the optimal budgets using the hyper-parameter δ. Note that if δ is high, the quantization is severe, otherwise the quantization is mild.
# 4.2 Inference Using Budget Allocator
We derive the response to the test query xq utilizing the LLM M(·) through the described steps (see Algorithm 3 for specifics). We first extract the embedding eq = E(xq). Then, we compute the allocated budget {ˆkc = fc(eq)}C c=1 and send ˆkc to each client. Each client sends back top ˆkc + α samples, i.e., Sc, to the server. Note that we summarize how the budget allocator outputs ˆkc in Figure 5. Here, α denotes the buffering hyper-parameter, which increases the chances for each client to be involved. After collecting Sagg = � c∈[C] Sc, we aggregate them and run the usual ICL procedure.
# 5 Experiment
# 5.1 Experiment setup
Algorithm
Dataset
Avg
SST-5
Amazon
Yelp
MR
Yahoo
AGNews
Subj
Zero-shot
29.19
24.70
31.23
73.95
25.87
67.60
50.55
43.30
Proxy-only
40.64± 2.89
28.43± 0.11
31.85± 1.28
70.40± 1.54
54.73± 0.93
84.65± 0.42
71.09± 1.34
54.54
Singleton
25.14± 4.18
24.03± 0.57
29.44± 3.91
50.00± 0.00
38.14± 2.03
50.60± 0.66
50.00± 0.00
38.19
Social Learning
36.03± 0.27
28.42± 0.19
29.25± 0.45
58.58± 0.18
46.03± 0.49
81.10± 0.29
71.37± 0.71
50.11
Uniform-budget
32.94
25.63
26.60
33.65
43.00
73.17
63.20
42.60
Random-budget
32.82± 0.82
25.69± 0.55
27.72± 0.51
34.68± 0.59
42.46± 0.53
67.34± 0.39
65.37± 0.80
42.30
∞-budget
43.26
32.70
34.80
77.20
62.67
89.37
91.4
61.62
Ours
44.08± 0.12
31.54± 0.22
35.48± 0.28
80.44± 0.67
61.67± 0.25
88.52± 0.30
82.36± 0.91
60.58
Table 1: Main results: To address the issue of non-IIDness in distributed ICL, we examined seven datasets and seven straightforward baselines. We run three random seeds and illustrate mean and std values. The top performance i highlighted in bold font, excluding the infinite budget scenario due to its impracticality. In summary, the proposed method effectively mitigates the non-iid distributed ICL problem to a reasonable extent.
datasets. Note that they do not overlap with the datasets used in our experiment. They used RoBERTa-large [29] encode model. We use GPT-Neo-2.7B [30] pre-trained model as answering LLMs. hyper-parameters related to training budget allocators, α, and δ are described in Appendix D in detail.
# 5.2 Main results
We have presented the performance of our algorithm and baselines in Table 1. First, we can observe that performance varies significantly depending on the way the budget is allocated, which indicates that the budget allocation scheme really matters in distributed non-IID ICL. Additionally, even when using only the proxy dataset, there is a performance improvement, and this performance surpasses that of using other clients which have the tilted local datasets (e.g., 29.19% →40.64% in SST-5 case). This indicates that utilizing a biased dataset can degrade the ICL performance. Although social learning algorithm has shown good performance in the previous paper, it does not perform well under the non-IID cases configured in this research. If we can use an infinite budget, all settings would exhibit high performance. However, our proposed algorithm demonstrates better performance than the infinite budget upper limit (e.g., 34.86% →35.48% in the Yelp case). This is likely due to a mechanism that prevents unnecessary information from being selected by the retriever with high importance. Ultimately, the proposed algorithm shows an average performance improvement of 5.05% percentage points across seven datasets compared to the best performance of baselines using the proxy dataset. This shows that the proposed algorithm can handle the non-IID case well.
# 5.3 Analysis
In this section, we further examine four key aspects: (1) privacy-preserving case analysis, which encompasses paraphrasing both training and testing queries, (2) sensitivity to hyper-parameters, (3) the performance of the trained budget allocator, and (4) the compatibility of the LLMs.
Algorithm
Dataset
Avg
SST-5
Yelp
Subj
Zero-shot
27.96
31.40
51.55
36.97
Proxy-only
39.39± 1.33
31.78± 1.75
73.46± 1.46
48.21
Singleton
25.31± 3.89
30.78± 4.88
50.08± 0.10
35.39
Social Learning
33.09± 0.68
28.80± 0.33
74.82± 0.93
45.47
Uniform-budget
27.06
26.60
63.30
38.99
Random-budget
27.29± 0.51
27.70± 0.46
63.88± 0.81
39.62
∞-budget
41.63
37.23
90.75
56.54
Ours
40.37± 0.27
36.52± 0.89
83.82± 1.00
53.57
Table 2: Analysis of the generated query and training samples: We re-create the datasets using small-sized LLMs a conduct the experiments as in Table 1 under the exactly same experimental settings.
Paraphrasing results. Due to privacy concerns in the fundamental distributed system, we evaluate the performance of paraphrased datasets, with results detailed in Table 2. Our method demonstrates superior performance compared to
other baselines across multiple datasets. We used the exact same data settings as in Table 1. Specifically, performance on the Subj and SST-5 datasets is lower than without paraphrasing, while the Yelp dataset shows a slight improvement. Additionally, as consistent with Table 1, non-IIDness causes significant performance degradation for ICL methods, as seen by comparing Zero-shot with ICL-related methods (e.g., 27.96% →25.31% in the Singleton case).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/82e6/82e60354-99b8-4194-afe0-8ba5a5d2b124.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Additional budget α analysis [need to update] (b) Resolution of budget allocator δ analysis [need to update]</div>
Figure 6: Hyperparameter, δ and α, analysis. We check SST-5 and Subj datasets under GPT-Neo-2.7B. The orange line indicates the 2nd best performance in Table 1.
Hyper-parameter sensitivity. We examine the sensitivity of the hyper-parameters of our method. We have two hyper-parameters: δ, which is the resolution of the budget allocator; α, which represents the additional budget allocated to each client as a buffer; and proxy size, which is the size of proxy data for the budget allocator training. As illustrated in Figure 6, when we increase α, the performance is improved while the budget efficiency is reduced. On the other hand, when δ is high (or low), it has too dense (or sparse) representation of the budget class, thus performance is degraded. Nevertheless, the performance is higher than the other baselines in Table 1. For the sensitivity of the size of proxy data, it is revealed that our framework is not sensitive to how many proxy data samples are used to train the budget allocator, as shown in Figure 7. This indicates our method is stable even with limited proxy data on the server side.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2bfb/2bfb11df-67da-4579-b70e-3b2902b16657.png" style="width: 50%;"></div>
<div style="text-align: center;">igure 7: Proxy size analysis. We check Subj dataset under GPT-Neo-2.</div>
Trained budget allocator. We assess whether the trained budget allocator distributes budgets appropriately for each client. To evaluate efficiency, we examine the number of samples, i.e., ˆkc communicated for all queries and plot a histogram. As demonstrated in Figure 8, we confirm that the proposed algorithm’s forecasts exhibit nearly identical performance to the oracle budget when an additional 25% budget is allocated. Note that without the proposed algorithm, it is necessary to assign k × C number of budgets to get a performance similar to the oracle case. Other types of LLMs. We utilize various LLM architectures to assess the compatibility of the proposed algorithm. Specifically, we evaluate the SST-5 dataset using different model sizes, including GPT-Neo-1.3B [30] and the latest architecture, the Gemma-2B [25] model. As demonstrated in Table 3, the proposed algorithm exhibits a plug-and-play capability and achieves reasonable performance improvements in the distributed non-IID ICL setting.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/030b/030bf7e1-a965-47f7-9c9c-59c26a8d1ff9.png" style="width: 50%;"></div>
Algorithm
Architecture
GPT-Neo-1.3B
GPT-Neo-2.7B
Llama-2-7B
Zero-shot
51.30
50.55
49.10
Proxy-only
80.18± 1.87
71.09± 1.34
88.13± 0.74
Singleton
50.00± 0.00
50.00± 0.00
52.89± 3.43
Social Learning
68.55± 0.64
71.37± 0.71
88.82± 0.50
Uniform-budget
44.40
63.20
54.00
Random-budget
43.68± 0.80
65.37± 0.80
55.60± 0.41
∞-budget
92.05
91.40
92.30
Ours
85.73± 0.94
82.36± 0.91
91.58± 0.14
# 6 Related Work
In-context learning. ICL [7] is one of the fastest paradigms using pre-trained LLMs by feeding several examples to construct the context to solve the given query. The main criteria of this research field are to find the most informative samples among the training datasets. For example [14] trains BERT [31] oriented encoder and use the k nearest neighbors. One of the reasonable sparse retriever, rule-based approach, is using BM25 [8] which measures the termfrequency. [32] proposed an efficient retriever called EPR. It trains two encoders by inheriting the method of dense passage retriever (DPR) [33] under the loss of positive and negative pairs. To reduce the domain specificity, [34] proposed UDR, which is applicable to multiple domain tasks in a universal way and shows reasonable performance from a single retriever. PromptPG [35] utilized a reinforcement learning framework to train the retriever so that it can generate context to improve the answerability of LLMs. Similarly, LLM-R [36] uses a reward model to train the retriever. Note that extensive research has not targeted to solve the distributed cases. These works have seen the centralized case. Distributed ICL. To the best of our knowledge, only a single study [13] tries to address ICL in a distributed manner. However, this paper solely focuses on merging the distributed information without considering the nature of the non-identically distributed information. Many studies, such as those on federated learning [37, 38, 39], address the non-IID distribution of datasets, highlighting the need to handle distributed non-IID ICL.
# 7 Conclusion
In this paper, we tackle the challenge of ICL when datasets are distributed among clients with non-IID distributions. Initially, we examine if non-IID distributions lead to performance degradation and discover that they cause significant drops in performance. We propose an algorithm that learns the task of budget assignment and employs it during inference to allocate appropriate budgets for each query. Using this proposed algorithm, we achieve performance improvements across various benchmarks.
# References
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. [2] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. [3] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. [4] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. [5] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018. [6] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. Advances in neural information processing systems, 32, 2019. [7] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022. [8] Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389, 2009. [9] Kaiyi Zhang, Ang Lv, Yuhan Chen, Hansen Ha, Tao Xu, and Rui Yan. Batch-icl: Effective, efficient, and order-agnostic in-context learning. arXiv preprint arXiv:2401.06469, 2024. 10] Jimin Xu, Nuanxin Hong, Zhening Xu, Zhou Zhao, Chao Wu, Kun Kuang, Jiaping Wang, Mingjie Zhu, Jingren Zhou, Kui Ren, et al. Data-driven learning for data rights, data pricing, and privacy computing. Engineering, 25:66–76, 2023. 11] Zicun Cong, Xuan Luo, Jian Pei, Feida Zhu, and Yong Zhang. Data pricing in machine learning pipelines. Knowledge and Information Systems, 64(6):1417–1455, 2022. 12] Zuoqi Tang, Zheqi Lv, and Chao Wu. Abrief survey of data pricing for machine learning. In CS & IT Conference Proceedings, volume 10. CS & IT Conference Proceedings, 2020. 13] Amirkeivan Mohtashami, Florian Hartmann, Sian Gooding, Lukas Zilka, Matt Sharifi, et al. Social learning: Towards collaborative learning with large language models. arXiv preprint arXiv:2312.11441, 2023. 14] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for gpt-3? arXiv preprint arXiv:2101.06804, 2021. 15] Jiacheng Ye, Zhiyong Wu, Jiangtao Feng, Tao Yu, and Lingpeng Kong. Compositional exemplars for in-context learning. In International Conference on Machine Learning, pages 39818–39833. PMLR, 2023. 16] Itay Levy, Ben Bogin, and Jonathan Berant. Diverse demonstrations improve in-context compositional generalization. arXiv preprint arXiv:2212.06800, 2022. 17] Qinbin Li, Yiqun Diao, Quan Chen, and Bingsheng He. Federated learning on non-iid data silos: An experimental study. In 2022 IEEE 38th international conference on data engineering (ICDE), pages 965–978. IEEE, 2022. 18] Joo Hun Yoo, Hyejun Jeong, Jaehyeok Lee, and Tai-Myoung Chung. Open problems in medical federated learning. International Journal of Web Information Systems, 18(2/3):77–99, 2022. 19] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008. 20] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1631–1642, 2013. 21] Julian McAuley and Jure Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. In Proceedings of the 7th ACM conference on Recommender systems, pages 165–172, 2013.
[22] Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28, 2015. [23] Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. arXiv preprint cs/0506075, 2005. [24] Bo Pang and Lillian Lee. A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics (ACL-04), pages 271–278, Barcelona, Spain, July 2004. [25] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024. [26] Zhenyu Wu Wu, Yaoxiang Wang, Jiacheng Ye, Jiangtao Feng, Jingjing Xu, Yu Qiao, and Zhiyong Wu. Openicl: An open-source framework for in-context learning. arXiv preprint arXiv:2303.02913, 2023. [27] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014. [28] Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112–1122. Association for Computational Linguistics, 2018. [29] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019. [30] Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow, March 2021. If you use this software, please cite it using these metadata. [31] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. [32] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. In Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz, editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2655–2671, Seattle, United States, July 2022. Association for Computational Linguistics. [33] Vladimir Karpukhin, Barlas O˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906, 2020. [34] Xiaonan Li, Kai Lv, Hang Yan, Tianyang Lin, Wei Zhu, Yuan Ni, Guotong Xie, Xiaoling Wang, and Xipeng Qiu. Unified demonstration retriever for in-context learning. arXiv preprint arXiv:2305.04320, 2023. [35] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. arXiv preprint arXiv:2209.14610, 2022. [36] Liang Wang, Nan Yang, and Furu Wei. Learning to retrieve in-context examples for large language models. arXiv preprint arXiv:2307.07164, 2023. [37] Qinbin Li, Zeyi Wen, Zhaomin Wu, Sixu Hu, Naibo Wang, Yuan Li, Xu Liu, and Bingsheng He. A survey on federated learning systems: Vision, hype and reality for data privacy and protection. IEEE Transactions on Knowledge and Data Engineering, 35(4):3347–3366, 2021. [38] Chen Zhang, Yu Xie, Hang Bai, Bin Yu, Weihong Li, and Yuan Gao. A survey on federated learning. KnowledgeBased Systems, 216:106775, 2021. [39] Priyanka Mary Mammen. Federated learning: Opportunities and challenges. arXiv preprint arXiv:2101.05428, 2021.
