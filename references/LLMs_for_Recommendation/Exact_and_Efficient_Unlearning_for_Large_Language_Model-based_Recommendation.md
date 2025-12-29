# ct and Efficient Unlearning for Large Language Model-base Recommendation

Zhiyu Hu University of Science and Technology of China Hefei, China zhiyuhu@mail.ustc.edu.cn
Yang Zhang University of Science and Technology of China Hefei, China zy2015@mail.ustc.edu.cn
Minghao Xiao University of Science and Technology of China Hefei, China xiaominghao@mail.ustc.edu.cn

Fuli Feng University of Science and Technology of China Hefei, China fulifeng93@gmail.com

Wenjie Wang National University of Singapore Singapore wenjiewang96@gmail.com

Fuli Feng University of Science and Technology of China Hefei, China fulifeng93@gmail.com
Xiangnan He University of Science and T of China Hefei, China xiangnanhe@gmail.

# ABSTRACT

The evolving paradigm of Large Language Model-based Recommendation (LLMRec) customizes Large Language Models (LLMs) through parameter-efficient fine-tuning (PEFT) using recommendation data. The inclusion of user data in LLMs raises privacy concerns. To protect users, the unlearning process in LLMRec, specifically removing unusable data (e.g., historical behaviors) from established LLMRec models, becomes crucial. However, existing unlearning methods are insufficient for the unique characteristics of LLMRec, mainly due to high computational costs or incomplete data erasure. In this study, we introduce the Adapter Partition and Aggregation (APA) framework for exact and efficient unlearning while maintaining recommendation performance. APA achieves this by establishing distinct adapters for partitioned training data shards and retraining only the adapters impacted by unusable data for unlearning. To preserve recommendation performance and mitigate considerable inference costs, APA employs parameter-level adapter aggregation with sample-adaptive attention for individual testing samples. Extensive experiments substantiate the effectiveness and efficiency of our proposed framework.

# CCS CONCEPTS

# • Information systems → Recommender systems; • Security and privacy → Privacy protections.

KEYWORDS

Large Language Models, Recommender System, Machine Unlearning, Recommendation Unlearning

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. Conference’17, July 2017, Washington, DC, USA © 2018 Association for Computing Machinery. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/XXXXXXX.XXXXXXX

ACM Reference Format: Zhiyu Hu, Yang Zhang, Minghao Xiao, Wenjie Wang, Fuli Feng, and Xiangnan He. 2018. Exact and Efficient Unlearning for Large Language Modelbased Recommendation. In Proceedings of ACM Conference (Conference’17). ACM, New York, NY, USA, 9 pages. https://doi.org/XXXXXXX.XXXXXXX

# 1 INTRODUCTION

Large language models have demonstrated exceptional capabilities in content comprehension and generation, sparking interest in applying them in Web applications [14, 21, 29, 40]. Recommender systems, as a primary channel for personalized content distribution, can also benefit from these capabilities in understanding items and users [37], pushing the emergence of large language modelbased recommendation paradigm. The current standard approach for specializing LLMs for recommendation is parameter-efficient fine-tuning [2, 3, 13, 37] using recommendation data. However, incorporating recommendation data (e.g.,  historical behaviors) increases the risk of personal data leakage due to the vulnerability of LLMs [6, 7, 11, 36]. To safeguard the privacy of users, particularly vulnerable populations, LLMRec unlearning becomes crucial, which targets at the timely and effective removal of some personal data [4] (termed unusable data [38]) from developed LLMRec models. To the best of our knowledge, there is currently no dedicated research on LLMRec unlearning. Despite the significant advancements of recommendation unlearning for traditional models [22, 38], these approaches are unfeasible for LLMRec models due to the high computation cost associated with handling billions of model parameters. Some very recent studies [12, 25, 35] investigate efficient unlearning techniques for information encoded in a LLM by extending traditional methods [35] or utilizing in-context learning [25]. However, applying these methods for LLMRec will encounter the risk of incomplete removal due to their approximate nature. In contrast, LLMRec unlearning requires complete removal of the unusable data to comply with relevant regulations such as the General Data Protection Regulation [27]. Additionally, LLMRec unlearning must maintaining maintain overall recommendation performance to ensure a satisfactory user experience. Achieving desirable LLMRec unlearning hinges on retraining PEFT adapters using data partitioning. Inspired by traditional unlearning methods [4, 8, 9], retraining has proven to be a reliable

method for ensuring exact unlearning. By employing a partitioning strategy that divides the training data into disjoint shards and training sub-models for each shard, unlearning efficiency can be maintained as only the sub-models affected by unusable data are retrained. Since personal data is exclusively stored in the PEFT adapter (e.g., LoRA [18]), retraining adapters on relevant shards incurs relatively low costs. Additionally, the PEFT adapter can quickly learn from a minimal number of examples, enabling further reduction in shard size and retraining costs. Considering these factors, we propose leveraging an adapter partition-empowered retraining approach for LLMRec unlearning. The partition strategy in LLMRec presents a distinct challenge in terms of inference latency since aggregating prediction results from different adapters is necessary to integrate knowledge for maintaining high recommendation performance [4, 8, 9]. However, such aggregation becomes infeasible for LLMRec models due to the computationally expensive nature of LLM inference [20]. Generating predictions from 𝐾 adapters would result in a 𝐾 times increase in the inference cost of LLMs, leading to substantial rises in energy consumption and service latency. For this challenge, we consider adapter aggregation at parameter level to enable a single-pass inference. Additionally, the partition and aggregation should be carefully designed to ensure a recommendation performance comparable to adapter retraining without partitioning [32]. To this end, we introduce the  Adapter Partition and Aggregation framework for exact and efficient LLMRec unlearning while maintaining overall recommendation performance. APA trains individual adapters on partitioned training data shards and leverages adapter weight aggregation during inference. As to partition, APA divides the training data into balanced and heterogeneous shards based on semantic characteristics [2] to facilitate keeping recommendation performance [32]. As to aggregation, we adopt a sample-adaptive approach for each testing sample that assigns adapter attention based on the performance of adapters on similar validation samples. This prioritizes higher-performing adapters, thereby enhancing overall performance. Notably, additional training is unnecessary for our adaptive aggregation, unlike traditional unlearning methods [8, 9], thus avoiding extra unlearning costs. The main contributions can be summarized as follows:
•  New problem: To our knowledge, this is the first study to formulate and explore exact unlearning within the realm of LLMRec.
• New technique: We introduce the APA method, an extension of partition-based unlearning, tailored to scenarios where inference involves high computational costs.
•  Experiments: We conduct extensive experiments on two realworlds, verifying the effectiveness of the proposal.

# 2 PRELIMINARIES

In this section, we first briefly introduce the prerequisite knowledge of PEFT and LLMRec. Then, we give the problem formulation.

# 2.1 PEFT

Fine-tuning LLMs with domain-specific data is an effective method to tailor LLMs for domain-specific tasks. Given that LLMs typically comprise billions of parameters, full tuning is a resource-intensive and time-consuming process. Recent work [1] shows that LLMs

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7e5c/7e5c7394-83f8-4ac7-9318-b366a6c13b0b.png" style="width: 50%;"></div>
Instruction Input
Instr.:
Given the user’s prefe-
rence and unpreference, ...
Input:
User preference: ...
User unpreference: ...
Target new movie: ...
Instruction Output
Output:
Yes.
Instruction Input
Instr.:
Given the user’s prefe-
rence and unpreference, ...
Input:
User preference: ...
User unpreference: ...
Target new movie: ...
<div style="text-align: center;">Instruction Output
</div>
<div style="text-align: center;">Figure 1: The left diagram illustrates the classic structure of LoRA, while the right table provides a sample for recommendation instruction data.
</div>
Figure 1: The left diagram illustrates the classic structure of LoRA, while the right table provides a sample for recommendation instruction data.

have a low intrinsic dimension that can match the performance of the full parameter space. PEFT provides a solution to this challenge by keeping the most of model weights frozen and only updating a part of the parameters. These learnable parameters are controlled by an adaptation module (termed adapter). LoRA. In this work, we focus on Low-Rank Adaptation (LoRA) [18], a prominent and widely adopted PEFT solution. To make finetuning more efficient, LoRA adds pairs of rank-decomposition weight matrices to existing weights of the LLM in a plug-in manner and only trains the newly added weights for learning tasks. The rank-decomposition design would ensure that the addition of weight matrices introduces only a small number of learnable parameters, thereby expediting the fine-tuning process. More specifically, for a matrix multiplication layer within LLMs, a LoRA module adds weight matrices as follows:

𝑜 = 𝑊 0 𝑥 + 𝐵𝐴𝑥,

(1)

where 𝑥 and ℎ represent the input and the resulting output, respectively. 𝑊 0 ∈R 𝑑 1 × 𝑑 2 denotes the original model weight matrix, while 𝐵 ∈R 𝑑 1 × 𝑟 and 𝐴 ∈R 𝑟 × 𝑑 2 constitute the pair of rankdecomposition weight matrices, with 𝑑 1, 𝑑 2, and 𝑟 representing the dimensions involved. Notably, 𝑟 ≪ min (𝑑 1,𝑑 2), meaning that the number of parameters introduced by 𝐵𝐴 is significantly fewer than that of W 0 because 𝑑 1 𝑟 + 𝑟𝑑 2 ≪ 𝑑 1 𝑑 2. During the fine-tuning process, only 𝐴 and 𝐵 are adjustable. In a similar way, a LoRA module (also called a LoRA adapter) is generally applicable to any LLM layer desired for updating.

# 2.2 LLMRec

PEFT with recommendation data has emerged as a de facto standard to specialize LLMs for recommendation tasks. Numerous studies have appeared and showcased the effectiveness of such LLMRec methods. In our research, we focus on the widely popular LLMRec method called TALLRec [3], considering its broad applicability and representation of the general paradigm in the field of LLMRec. TALLRec. TALLRec employs LoRA tuning techniques to align LLMs with the recommendation task using recommendation instruction data. This approach involves the conversion of user-item

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/db25/db258e43-16fe-47f0-acf8-886cf81d3eff.png" style="width: 50%;"></div>
<div style="text-align: center;">framework, which consists of three parts: Data Partition, Adpter training, and Adpter to erase data 𝐷 𝑟, only the sub- LoRA adapters affected by 𝐷 𝑟 need to be retrained.
</div>
<div style="text-align: center;">Figure 2: Illustration of our APA framework, which consists of three parts: D aggregation. When a user requests to erase data 𝐷 𝑟, only the sub- LoRA adapters
</div>
<div style="text-align: center;">ure 2: Illustration of our APA framework, which consists of three parts: Data Partition, Adpter training, and regation. When a user requests to erase data 𝐷 𝑟, only the sub- LoRA adapters affected by 𝐷 𝑟 need to be retrained.
</div>
interaction data into language instructions, as exemplified in Figure 1. Each instruction comprises both an input and an output component. Within the instruction input, TALLRec represents items using their titles and user preferences or non-preferences are conveyed by referencing historical item titles, it also instructs LLMs to respond with either “Yes” or “No” to indicate the user’s preference for a target item. The response is included in the instruction output. With the instruction data, TALLRec performs fine-tuning of the LLM using a LoRA adapter to learn the recommendation task. Let D represent the set of all converted instruction data for training, and then the optimization problem can be formulated as follows:

(2)

where 𝑥 and 𝑦 represent the instruction input and output of a data sample in D, 𝑦 𝑡 represents the 𝑡-th text token of 𝑦, 𝑦 <𝑡 denotes the text tokens that precede 𝑦 𝑡, and 𝑃 Θ 0 + Φ (𝑦 𝑡 | 𝑥,𝑦 <𝑡)) signifies the predictive probability of 𝑦 𝑡 by the LLM. Θ 0 refers to the existing parameters of the original LLM, and Φ  encompasses all model parameters within the LoRA adapter, including the 𝐴 and 𝐵 as defined in Equation (1)  for all layers. Notably, only the LoRA adapter parameters Φ would be updated.

# 2.3 Problem Formulation

Let D − 𝑟 ⊂D represent the data that a user wishes to remove from a PEFT LLMRec model 𝑓 that was initially trained with D. Following previous work, we assume the size of D − 𝑟 is very small, e.g., |D − 𝑟 | = 1. We try to obtain a retrained model using only the remaining data, denoted as D 𝑟 = D −D − 𝑟, to achieve exact unlearning. Simultaneously, this unlearning process needs to be efficient in order to respond to the user’s request promptly. Finally, we aim to minimize any performance degradation after implementing the unlearning designs to ensure that users remain satisfied with the recommendation quality.

# 3 METHODOLOGY

In this section, we commence with presenting an overview of our approach, encompassing the model framework and the unlearning process. Following that, we provide a detailed discussion of the pivotal components of our method.

# 3.1 Overview

To enable exact and efficient unlearning based on retraining, our APA framework employs a partitioning strategy to train and construct the LLMRec model. Our APA framework, as illustrated in Figure 2, encompasses three key phases:

1) Data and Adapter Partition: We partition the training data D into 𝐾 balanced and disjoint shards, denoted as {D 1, . . . , D 𝐾}. Given that LLM relies on text semantics for predictions, we perform the partition based on the text semantics of the samples, utilizing a K-means clustering method. Once the data shards are obtained, we proceed to train an individual LoRA adapter (a sub-adapter) for each shard in TALLRec. For the 𝑘-th data shard D 𝑘, we train a LoRA adapter parameterized with Φ 𝑘 according to Equation (4) (replacing D and Φ in the equation with D 𝑘 and Φ 𝑘).
2) Adapter Aggregation: At the serving stage, we perform adapter aggregation, which involves merging the weights of different LoRA adapters to create a unified adapter. We just use the aggregated LoRA adapter for inference. Importantly, we employ a sample-adaptive aggregation strategy, tailoring the aggregated adapter to the specific sample for improved performance. This part is the key to ensuring performance and inference efficiency.

Unlearning. When a user requests to erase data D 𝑟, only the subLoRA adapters affected by D 𝑟 need to be retrained, obviating the need to retrain the entire model and facilitating acceleration. In theory, we only need to invest a | D 𝑟 |
𝐾 cost for full retraining to achieve precise unlearning. With the support of two considerations, | D 𝑟 |
𝐾 can be kept at a low value, resulting in significant acceleration: Firstly, it is often assumed that user requests arrive in a streaming manner, usually, only one sample needs to be unlearned at a time.

Secondly, given the few-shot learning capabilities of TALLRec (a few hundred samples are adequate to train an effective LoRA), the data partition can be fine-grained, allowing for a relatively high value of 𝐾. After the LLMRec has been constructed, the unlearning process is simple and straightforward. Therefore, the essence of the process lies in our partition and aggregation stages. We now delve into the details of these two phases.

# 3.2 Partition

The partitioning phase is the key to training a LLMRec model, involving two key parts: 1) data partition, and 2) training a subadapter module for each partitioned data shard. We next elaborate on the two parts.

3.2.1 Data Partition. For data partitioning, the crucial factor is ensuring that data within the same shard share related knowledge, creating homogeneity of knowledge within a shard and heterogeneity across shards. This aids in the effective learning of sub-adapters with limited data and their subsequent aggregation. Prior methods for recommendation unlearning relied on collaborative embedding to perform partitioning, utilizing the K-means algorithm to group samples with similar collaborative information within the same shard. Similarly, given that LLMRec relies on text semantics for prediction, we propose partitioning data based on semantics. Our data partition method is detailed in Algorithm 1, consisting of three key parts. Initially, we utilize the original LLM (without fine-tuning) to derive the hidden representations (denoted as ℎ 𝑖) of the input instructions for each training sample (𝑥 𝑖,𝑦 𝑖) in D, capturing the text semantics (line 2). Subsequently, we employ K-means on the obtained hidden representations, resulting in 𝐾 clusters and 𝐾 clustering centers (denoted as 𝑎 1, . . . ,𝑎 𝐾) (line 3). The clusters generated directly by 𝐾 can be highly unbalanced, potentially making unlearning inefficient for large shards. Therefore, we take further steps to balance the clusters (lines 4-10). Instead of directly assigning a sample to the nearest cluster, we take into account the cluster size: if the size of the closest cluster exceeds a certain threshold, we assign the sample to the nearest cluster whose size is still below that threshold. Formally, for all samples, we calculate their cosine distance to each cluster center as follows:

(3)

We store all distances in a list, denoted as 𝐹 = {(𝑥 𝑖,𝑦 𝑖,𝑑𝑖𝑠𝑡 (ℎ 𝑖,𝑎 𝑘))| 𝑖 ≤ |D|,𝑘 ≤| 𝐾 |}, and then sort the list based on the 𝑑𝑖𝑠𝑡 () function. We refer to the sorted list as 𝐹 𝑠. Subsequently, we orderly examine each element (𝑥 𝑖 ′,𝑦 𝑖 ′, dist (ℎ 𝑖 ′,𝑎 𝑘 ′)) in 𝐹 𝑠 to achieve balanced clustering. If (𝑥 ′ 𝑖,𝑦 ′ 𝑖) has not yet been assigned to any cluster, we assign it to the 𝑘 ′-th cluster, denoted as D 𝑘 ′.

3.2.2 Sub-adapter Training. After obtaining the partitioned data, we proceed to train an individual LoRA adapter for each data shard, following the approach of TALLRec. Formally, for the 𝑘-th data shard, the optimization objective is as follows:

(4)

where Φ 𝑘 denotes the model parameters of the 𝑘-th LoRA for the 𝑘-th data shard, 𝑃 Θ 0 + Φ 𝑘 (𝑦 𝑖 | 𝑥 𝑖) denote the prediction probability of LLMRec with the 𝑘-th LoRA for 𝑦 𝑖.

# 3.3 Aggregation

During the serving prediction stage, aggregating knowledge from the sub-models is essential to enhance the overall prediction quality. Typically, prediction result aggregation is a commonly used approach. However, this method necessitates performing LLM inference 𝐾 times, as it requires computing 𝑃 Θ 0 + Φ 𝑘 (𝑦 𝑖 | 𝑥 𝑖) for k=1, 2,. . ., 𝐾. To address this challenge, we introduce aggregation at the LoRA adapter model weight level, i.e., adapter aggregation. This technique combines the weights of multiple LoRA adapters, creating a single LoRA adapter that allows for one-pass prediction. Given that a weight matrix in the original LLM corresponds to a pair of rank-decomposition weight matrices, as shown in Equation 1, we consider two levels of aggregation: • Decomposition level: At this level, each model weight of LoRA serves as the unit for model aggregation. We directly aggregate the 𝐴 and 𝐵 matrices defined in Equation (1) from different LoRA adapters using weight averaging. Formally, a aggregated LoRA layer can be defined as follows:

¯ 𝑜 = 𝑊 0 𝑥 + ¯ 𝐵 ¯ 𝐴𝑥,

where ¯ 𝐵 represents the aggregated 𝐵 matrix, ¯ 𝐵 𝑘 represents the 𝐵 matrix of the 𝑘-th sub-adapter, similarly for those of 𝐴; ¯ 𝑜 denotes the layer output in the aggregated LoRA adapter, and 𝜔 𝑘 is the aggregation weight for the 𝑘-th sub-adapter, where a higher value indicates higher attention. The method for assigning 𝜔 𝑘 is described later.
Non-decomposition level: At this level, the weight unit of the original LLM serves as the aggregation unit for the adapter aggregation. Then, we aggregate the “BA” result defined in Equation (1) from all sub-LoRA adapters using weight averaging. Formally, a aggregated LoRA layer can be formulated as follows:

(6)

symbols have the same meanings as in Equation (5). Sample-adaptive Strategy.  Different testing samples require varying levels of knowledge from different sub-models for accurate prediction, suggesting the need for an adaptive attention allocation when aggregating sub-adapters. To avoid introducing additional training and unlearning, we explore a heuristic approach to assign aggregation weights to different sub-adapters. Based on our partition, one straightforward solution is to use text semantic similarity to determine these weights, giving higher priority to the adapter corresponding to the data shards with greater similarity to the sample. However, selecting an adapter solely based on input similarity doesn’t guarantee better prediction accuracy. To address

Algorithm 1: Balanced Semantic-aware Data Partition
Input: training instruction data D, cluster number 𝐾, and
maximum size of each shard 𝑡
Output: The Shards {D0, ..., D𝐾}
1 Initialize D0, . . . , D𝐾;
2 Compute hidden representation ℎ𝑖of 𝑥𝑖in the original LLM
for each training sample (𝑥𝑖,𝑦𝑖) ∈D;
3 Runing the K-means with all hidden representations
{ℎ𝑖|𝑖≤|D|}, obtaining cluster centers:
{𝑎0,𝑎1, ...,𝑎𝐾} = 𝐾-𝑚𝑒𝑎𝑛𝑠({ℎ𝑖|𝑖≤|D|}, 𝐾);
4 For each sample (𝑥𝑖,𝑦𝑖) and each cluster center 𝑎𝑘, compute
their cosine distance using ℎ𝑖, i.e., 𝑑𝑖𝑠𝑡(ℎ𝑖,𝑎𝑘), storing
(𝑥𝑖,𝑦𝑖,𝑑𝑖𝑠𝑡(ℎ𝑖,𝑎𝑘)) in a list 𝐹;
5 Sort 𝐹in ascending order to get 𝐹𝑠;
6 for each (𝑥′
𝑖,𝑦′
𝑖,𝑑𝑖𝑠𝑡(ℎ𝑖′,𝑎𝑘′)) in 𝐹𝑠do
7
if |D𝑘′ | < 𝑡and (𝑥𝑖,𝑦𝑖) has not been assigned then
8
D𝑘′ ←D𝑘′ ∪(𝑥𝑖,𝑦𝑖)
9
end
10 end
11 return 𝐷;
this concern, we devise a method that leverages validation prediction errors to enhance the assignment mechanism. In essence, for each testing sample, we rely on the prediction errors of the most similar validation samples to assess the suitability of a particular adapter for that specific testing sample and allocate the attention weights accordingly. This approach ensures more accurate weight assignments for effective prediction. Specifically, for each testing sample (𝑥,𝑦), we initially identify the top-n most similar samples from the validation set, calculating the similarities similar to Equation (3). These identified similar samples are denoted as 𝑁 𝑣. Next, we measure the average prediction error among 𝑁 𝑣 for each sub-adapter as follows:
∑︁

(7)

where |N 𝑣 | represents the size of N 𝑣, and 𝑒𝑟𝑟𝑜𝑟 𝑘 stands for the average prediction error of the 𝑘-th sub-LoRA adapter. Subsequently, for this testing sample, we assign higher attention weights to the sub-LoRA adapter with lower prediction errors. Formally, the attention weight 𝜔 𝑘 for the 𝑘-th sub-LoRA adapter is calculated as follows:

(8)

�
where 𝜏 represents the temperature parameter, controlling the strength of the assignment mechanism. When 𝜏 =  0, the mechanism becomes ineffective, allocating equal attention weights for all sub-LoRA adapters.

Discussion.  It is worth mentioning that our method is developed specifically for LoRA-based LLMRec. However, since the PEFT method commonly incorporates adaptation modules, we can directly extend our method to LLMRec models developed using other PEFT techniques like Adapter Tuning [17]. This flexibility allows us to apply our method to a wider range of LLMRec architectures.

# 4 EXPERIMENTS

In this section, we conduct a series of experiments to answer the following research questions:

In this section, we conduct a series of experiments to answer th following research questions:

• RQ1: How does APA perform in terms of recommendation performance and unlearning efficiency compared to the state-of-the-art exact unlearning methods for LLMRec?
• RQ2: How does APA perform in terms of inference efficiency? • RQ3: How do different components of the proposed APA influence its effectiveness?

# Experimental Settings

4.1.1 Datasets. We conduct experiments on two distinct real-world datasets, which are widely recognized and used within the realm of recommender systems:

We strictly adhere to the pre-processing procedures outlined in the TALLRec paper [3] for data filtering, data splitting, and instruction data construction for both datasets. In particular, considering TALLRec’s capability to efficiently learn recommendations and yield good performance with a minimal number of training samples, we constrain our training size to 1024 (larger than the maximum size of 256 mentioned in the TALLRec paper). Similar to the TALLRec setting, the validation set comprises 500 samples for both Movie and Book, while the testing consists of 1000 samples for both the Movie and Book datasets.

4.1.2 Compared Methods. We focus on exact unlearning, but there is currently no specific work designed for LLM. Therefore, to serve as a baseline, we consider extending the following traditional exact unlearning baselines to TALLRec:

• Retraining  This represents the straightforward retraining approach, i.e.,  retraining the entire model from scratch while excluding the unusable data. We implement it by retraining TALLRec from scratch, excluding the unusable data. This method serves as the gold standard in terms of recommendation performance.
• SISA [4] is the earliest known partition-enhanced retraining method. It randomly divides data and aggregates sub-model predictions through methods such as averaging or majority voting. We extend this approach to TALLRec, employing its averagebased aggregation.
• RecEraser [8] is a recommendation-specific unlearning method, sharing similarities with SISA but incorporating unique partitioning strategies to preserve collaborative information. We adapt it for LLMRec based on its UBP version. Notably, its prediction

<div style="text-align: center;">Table 1: Comparison of different unlearning methods on recommendation performance, where ‘APA(D)’/‘APA(ND APA implemented with decomposition/non-decomposition level aggregation, and △ represents the gap between re the unlearning method in terms of AUC. ‘Bef. Agg.’ represents the average 𝐴𝑈𝐶 of the sub-model.
</div>
Book
Retraining
SISA
GraphEraser
RecEraser
APA(D)
APA(ND)
Bef. Agg.
-
0.6570
0.6443
0.6620
0.6578
0.6578
AUC
0.6738
0.6728
0.6684
0.6732
0.6829
0.6846
△
-
-0.001
-0.0052
-0.0006
0.0091
0.0108
Movie
Retraining
SISA
GraphEraser
RecEraser
APA(D)
APA(ND)
Bef. Agg.
-
0.7003
0.6672
0.6712
0.6696
0.6696
AUC
0.7428
0.7035
0.6903
0.6937
0.7259
0.7256
△
-
-0.0393
-0.0525
-0.0491
-0.0169
-0.0172
aggregation involves training with 𝐾 TALLRec, requiring overmuch computational resources. In our adaptation, we directly replace it with the aggregation strategy of SISA.
• GraphEraser [9]  is an unlearning method designed for graphstructured data (including the bipartite graph structure of interaction data). It employs node clustering techniques, namely BEKM and BLPA, for graph data partitioning. We extend GraphEraser to TALLRec using the BEKM-based partition. Similar to RecEraser, we adopt the aggregation strategy of SISA for it.
Regarding our APA, we implement two versions using different levels of aggregation, as defined in Section 3.3. We denote the version with decomposition-level aggregation as APA(D) and the version with non-decomposition-level aggregation as APA(ND).

4.1.3 Evaluation Setting. Our objective is to achieve precise and efficient unlearning for LLMRec while preserving recommendation performance. Therefore, our evaluation focuses on three aspects: 1) the completeness of data removal, 2) unlearning efficiency, and 3) recommendation performance. Since all compared methods are built on retraining (from scratch) without the unusable data, the first aspect is inherently maintained. Following the RecEraser paper [8], we do not consider this aspect for evaluation. To assess recommendation performance, we use the Area under the ROC Curve (AUC) metric, following the TALLRec paper. Additionally, we introduce another performance loss metric Δ to measure the recommendation performance loss of a method relative to the Retraining method. This metric is calculated by the difference in AUC between the method and the Retraining method, with higher values indicating less performance loss. For evaluating unlearning efficiency, we directly utilize the unlearning time (retraining time). Additionally, considering the inference cost for LLMs, we further compare the inference time.

4.1.4 Implementation Details. As all methods utilize TALLRec as the backbone recommendation model, we apply the same hyperparameter settings for them to learn the recommendation model, following the original configuration outlined in the TALLRec paper. Concerning the unlearning setting, for all partition-based base models, we set the shard size to 256 for the data partition, resulting in 𝐾 = 4 training data shards. For the specific hyper-parameters of the baselines, we tune them in accordance with the settings provided in the original papers, whenever available for our extension.

Regarding the proposed APA, we set 𝜏 (in Equation (8)) to 1000 for both dataset. For the neighbor size |N 𝑣 | in Equation (7), we set it to 20 for Movie and 100 for Book. All these hyperparameters are tuned on the validation set, and all experiments are conducted on the same machine equipped with NVIDIA A40 GPUs.

# 4.2 Main Results (RQ1)

In this subsection, we evaluate all unlearning methods based on two criteria: recommendation performance and unlearning efficiency. It is essential to note that all compared methods inherently achieve complete removal of unusable data; therefore, we omit the comparison in the aspect of exact unlearning [8].

4.2.1 Accuracy Comparison. We compare the accuracy of APA with that of the baselines to assess its ability to maintain recommendation performance during unlearning. Higher ability is indicated by less performance loss compared to the Retraining method. The comparison results are summarized in Table 1, where we additionally include the averaged performance of the sub-models for the partition-based methods and draw the following observations:

• All methods with aggregation demonstrate improved 𝐴𝑈𝐶 compared to the averaged 𝐴𝑈𝐶 of their corresponding sub-models. This underscores the significance of aggregating knowledge from sub-models to enhance performance.
• Compared to the baselines, APA exhibits less performance loss compared to the reference Retraining method and can even bring improvements. These results highlight the superior ability of our method to maintain recommendation performance during unlearning. This superiority can be attributed to the compatibility between our partitioning method and aggregation, as well as the adaptive aggregation approach based on validation performance, which pays more attention to high-performance sub-adapters.
•  In contrast, SISA, RecEraser, and GraphEraser show much inferior performance compared to the reference method Retraining. Particularly on the movie dataset, these baselines exhibit a significant decline in recommendation performance. This suggests that the direct application of traditional methods to TALLRec results in a substantial compromise in recommendation performance.
•  The two versions of APA with different levels of adapter aggregation (APA(D) and APA(ND)) demonstrate similar performance.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a849/a849bdd9-5764-4a48-9717-aa78c6757630.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b129/b1293f7e-7bec-4191-9774-9db03b3e821c.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b6b9/b6b97897-c773-4ec7-ae91-94c128c97f5a.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Inference time
</div>
<div style="text-align: center;">(a) Unlearning time
</div>
Figure 3: (a) Unlearning time of Retraining and APA. (b) Inference time of Retraining, SISA, APA(D), and APA(ND).

# Figure 3: (a) Unlearning time of Retraining and APA. (b) Inference time of Retraining, SISA, APA(D), and APA(ND).

This indicates that treating the LoRA rank-decomposed parameter as the aggregation unit or the original LLM parameter unit as the aggregation unit does not affect the effectiveness of our adaptive aggregation method.

4.2.2 Unlearning Efficiency Comparison.  We next conduct experiments to explore the unlearning efficiency of our APA. We fully follow the efficiency evaluation experiment setting in the RecEraser paper [8], ensuring that only one sub-model needs to be retrained for unlearning each time. We primarily compare our method with the Retraining method, as other baselines theoretically have similar unlearning efficiency costs to us. The results are shown in Figure 3(a). The results demonstrate that APA significantly improves unlearning efficiency. For example, on the movie dataset, APA only took 10,335 seconds, making it 3.96 times faster (approximately = 𝐾) than the Retraining method. The Retraining method is timeconsuming as it is trained on the whole dataset. In contrast, APA only requires retraining the specific sub-model responsible for the unlearned data. Moreover, when the training data is large, we can keep a small shard size to allow for a large number of data shards 𝐾, considering that TALLRec can effectively learn recommendations with few samples. In this case, APA could achieve greater acceleration as long as only a few sub-adapters are affected by unusable data.

# 4.3 Inference Time Comparison (RQ2)

In the previous section, we explored how our APA method can significantly reduce time during the unlearning process. In this section, we investigate whether APA can improve efficiency during the inference stage compared to baselines. We randomly selected 500 samples from the testing set and measured the total inference time for these samples, ensuring that only one LLM inference could be executed at a time. We compare our APA with SISA and Retraining (we omit other baselines due to their similar costs to SISA). The experimental results are presented in Figure 3(b). From the results, we have the following observations: 1) Our APA method exhibits small gaps in time efficiency compared to a single model inference (Retraining), and the delay for each sample is just approximately 0.02 seconds, which is entirely acceptable in real-world scenarios. 2) SISA has much higher inference time costs compared to Retraining and APA. This is because SISA performs predictionlevel aggregation for sub-adapters, which involves additional time for inference cost. These results demonstrate the effectiveness of APA’s aggregation designs in enhancing inference efficiency.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6cd9/6cd9bc35-ae39-4eba-a202-149c43e2b5b3.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Movie
</div>
<div style="text-align: center;">(a) Book
</div>
Figure 4: Performance comparison of different data partitio methods on Book and Movie datasets.

<div style="text-align: center;">Figure 4: Performance comparison of different data partition methods on Book and Movie datasets.
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f3e7/f3e7173c-4581-4e5b-aff1-9f5f6aca51a1.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Book
</div>
<div style="text-align: center;">(b) Movie
</div>
<div style="text-align: center;">Figure 5: Performance comparison of different aggregation weight assignment methods on Book and Movie datasets.
</div>
# 4.4 In-depth Studies(RQ3)

4.4.1 Effect of the Data Partition Methods.  To validate the effectiveness of our proposed data partition method, we compare it with three other grouping methods: random partition, the user-based partition of RecEraser [8], and BEKM partition [9], denoted as ’Random,’ ’User,’ and ’BEKM’, respectively. We replace the original partition method with the three methods in our APA, respectively, and then compare their performances with the original one. The experimental findings are illustrated in Figure 3(b). Based on the results, we draw the following observations: 1) Replacing the original semantic-aware method with any of the three methods would result in a performance decrease. For example, On the Book dataset, the original semantic-aware method achieves an AUC score of 0.6829, while the corresponding results were just 0.6503, 0.6627, and 0.6411 for random partition, user-based partition, and BEKM, respectively. The result shows the importance of leveraging semantics to partition for LLMs, which could ensure better heterogeneity of data shards to facilitate better aggregation for enhancing recommendation performance.

4.4.2 Effect of the Sample-Adaptive Method.  We proceed to assess the model utility of different aggregation weight assignment methods to demonstrate the effectiveness of our proposed sampleadaptive method. We compare our methods with the following three choices: 1) average-based, assigning equal weight for each sub-adapter, 2) major-based, assigning all weights to the one with the highest 𝜔 𝑘 computed by our method, 3) semantic-based, which assigns weight according to the semantic similarity of the sample to the center of different shards. We compare the APA implemented with our assignment method with the variants of APA implemented with the three methods. The experimental results are presented

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6e5b/6e5beeb1-feb7-4730-ad6c-e2289d9edd77.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Recommendation Performance
</div>
<div style="text-align: center;">(b) Unlearning Time
</div>
Figure 6: Impact of the shard size on the unlearning efficiency and performance on Book dataset. (a) shows the recommendation performance and (b) shows the unlearning time cost.

in Figure 5, with ’Average,’ ’Major,’ and ’Semantic’ denoting the compared three choices, respectively. Here are some observations we found: 1) The average-based method underperforms our sampleadaptive method on both datasets, highlighting the importance of assigning different weights for different adapters; 2) The semanticbased method also exhibits worse recommendation performance than our original method, confirming the effectiveness of utilizing validation performance information; 3) Using only the best submodel choice by our weight assignments can maintain relatively high recommendation performance, but there is still a gap compared to our method, as shown by the results of the major-based method. These results emphasize the effectiveness of our weight assignments, and meanwhile, the importance of aggregating knowledge from different sub-adapters.
4.4.3 Impact of the Shard Size. We investigate the influence of the shard size on the Book dataset. We configure the shrad size as {512, 256, 128}  and evaluate unlearning efficiency and recommendation performance. The experimental results are displayed in Figure 6. We find that 1) as the shard size decreases, the unlearning time significantly reduces; 2) In terms of recommendation performance, as the shard size decreases, it remains relatively stable before decreasing. For example, when the shard size is 512 and 256, the performance of APA remains very close, and it only significantly decreases at 128. This indicates that within a certain range, we can improve unlearning efficiency by reducing the data shard size (increasing the number of shards) while maintaining comparable recommendation performance. In this way, on the one hand, the cost of retraining individual shards decreases, and on the other hand, increasing the number of shards may keep the proportion of adapters requiring retraining relatively low, thereby enhancing unlearning efficiency.

# 5 RELATED WORK 5.1 Machine Unlearning

# 5.1 Machine Unlearning

• Machine Unlearning. Machine unlearning, the process of removing partial training data information from trained machine learning models, is essential in various domains, including recommendation, for reasons such as privacy and security concerns [5, 23]. This concept is known as machine unlearning [4]. In traditional machine learning, two main technique lines for unlearning have

emerged: approximate unlearning and exact unlearning [24, 33]. Approximate unlearning aims for unlearning without retraining, using techniques like influence functions [19, 38] and data augmentation [28, 30] for extreme efficiency, but it often involves incomplete removal of the data. On the other hand, exact unlearning [4] typically involves retraining, ensuring complete unlearning but in a time-costly manner. Existing work, like SISA [8, 9, 26, 34], focuses on partition strategies, building individual sub-models for partitioned training data shards to retrain only partial sub-models. Our method, while also based on the partition strategy, addresses new challenges posed by the large scale and high inference cost of Large Language Models (LLM). This makes our work distinct from existing methods. • LLM Unlearning. The challenges presented by Large Language Models (LLMs), particularly their large scale, bring forth new considerations for unlearning. Previous efforts [12, 25, 35] have explored unlearning for LLMs, but they often involve approximate methods. For instance, [12] simulates data labels to approximate the next-token predictions of a model that has not been trained on the unusable data, and then fine-tune LLM on these simulated labels for unlearning. [25] proposes "In Context Unlearning", which leverages in-context learning by flipping labels of unusable data to achieve approximate unlearning. [35] leverage the gradient ascent to erase the influence of unusable data on a trained model with finetuning. However, these methods do not achieve complete removal of unusable data and are not tailored for LLMs in the context of recommender systems. In contrast, our approach focuses on LLMRec and strives for exact unlearning, considering the significant impact of incomplete removal of sensitive data.

# 5.2 Model Aggregation in LLM

To aggregate the different models, there are two strategies: 1) output aggregation and 2) model weight aggregation. Output aggregation has been widely studied, and applied for aggregation process for partition-based unlearning, but could introduce inefficiency for LLM. Regarding the model weight aggregation, existing work focuses on leveraging it to better finish tasks like image classification, multi-domain learning [10], Cross-lingual information extraction [31], etc. To our knowledge, we are the first to leverage it for the unlearning task. Meanwhile, from the technical view, our method has significant differences from existing work on the aggregation weight assignment. To achieve weight assignment, previous works usually considered 1) average aggregation [32], 2) greedy aggregation, and 3) learning-based aggregation like simulating Mixture-of-Experts (MoE) [10, 31]. Differently, we innovatively assign weights to different sub-models based on the prediction quality of similar verification samples, which can achieve effective adaptive weight assignments without learning.

# 6 CONCLUSION

In this study, we introduce a novel and efficient APA framework, which, to the best of our knowledge, is the first exact unlearning method designed for large language model-based recommendation (LLMRec). To achieve efficient unlearning while preserving high recommendation performance, we propose a data partition method based on text semantics. Additionally, we employ parameter-level

adapter aggregation to create an aggregated adapter to mitigate the high inference cost associated with traditional methods. We carry out comprehensive experiments on two real-world datasets, offering insightful analysis of the effectiveness and efficiency of our approach in removing interaction data. In our future endeavors, we aim to expand our approach to encompass other PEFT methods, thus widening the adaptability of our method across diverse LLMRec architectures. Moreover, we are exploring methods to enhance unlearning efficiency in batch settings, enabling the management of more unlearn data.

# REFERENCES

[1]  Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. 2020. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. arXiv preprint arXiv:2012.13255 (2020).
[2] Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yancheng Luo, Fuli Feng, Xiangnaan He, and Qi Tian. 2023. A bi-step grounding paradigm for large language models in recommendation systems. arXiv preprint arXiv:2308.08434 (2023).
[3] Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems, RecSys 2023. 1007–1014.
[4]  Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. 2021. Machine unlearning. In 2021 IEEE Symposium on Security and Privacy (SP). IEEE, 141–159.
[5] Yinzhi Cao and Junfeng Yang. 2015. Towards making systems forget with machine unlearning. In 2015 IEEE symposium on security and privacy. IEEE, 463–480.
[6]  Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel HerbertVoss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. 2021. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21). 2633–2650.
[7] Aldo Gael Carranza, Rezsa Farahani, Natalia Ponomareva, Alex Kurakin, Matthew Jagielski, and Milad Nasr. 2023. Privacy-Preserving Recommender Systems with Synthetic Query Generation using Differentially Private Large Language Models. arXiv preprint arXiv:2305.05973 (2023).
[8] Chong Chen, Fei Sun, Min Zhang, and Bolin Ding. 2022. Recommendation unlearning. In Proceedings of the ACM Web Conference 2022. 2768–2777.
[9] Min Chen, Zhikun Zhang, Tianhao Wang, Michael Backes, Mathias Humbert, and Yang Zhang. 2022. Graph unlearning. In Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security. 499–513.
[10] Shizhe Diao, Tianyang Xu, Ruijia Xu, Jiawei Wang, and Tong Zhang. 2023. Mixture-of-Domain-Adapters: Decoupling and Injecting Domain Knowledge to Pre-trained Language Models Memories. arXiv preprint arXiv:2306.05406 (2023).
[11] Haonan Duan, Adam Dziedzic, Nicolas Papernot, and Franziska Boenisch. 2023. Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models. arXiv preprint arXiv:2305.15594 (2023).
[12] Ronen Eldan and Mark Russinovich. 2023. Who’s Harry Potter? Approximate Unlearning in LLMs. arXiv preprint arXiv:2310.02238 (2023).
[13] Junchen Fu, Fajie Yuan, Yu Song, Zheng Yuan, Mingyue Cheng, Shenghui Cheng, Jiaqi Zhang, Jie Wang, and Yunzhu Pan. 2024. Exploring Adapter-based Transfer Learning for Recommender Systems: Empirical Studies and Practical Insights. The 17th ACM International Conference on Web Search and Data Mining (2024).
[14] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. 2023. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010 (2023).
[15] F Maxwell Harper and Joseph A Konstan. 2015. The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis) 5, 4 (2015), 1–19.
[16] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 639–648.
[17] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for NLP. In International Conference on Machine Learning. PMLR, 2790–2799.
[18] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021).

[19] Zachary Izzo, Mary Anne Smart, Kamalika Chaudhuri, and James Zou. 2021. Approximate data deletion from machine learning models. In  International Conference on Artificial Intelligence and Statistics. PMLR, 2008–2016.
[20] Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, and Robert McHardy. 2023. Challenges and applications of large language models. arXiv preprint arXiv:2307.10169 (2023).
[21] Xinyu Lin, Wenjie Wang, Yongqi Li, Fuli Feng, See-Kiong Ng, and Tat-Seng Chua. 2023. A multi-facet paradigm to bridge large language model and recommendation. arXiv preprint arXiv:2310.06491 (2023).
[22] Wenyan Liu, Juncheng Wan, Xiaoling Wang, Weinan Zhang, Dell Zhang, and Hang Li. 2022. Forgetting Fast in Recommender Systems. arXiv preprint arXiv:2208.06875 (2022).
[23] Neil G Marchant, Benjamin IP Rubinstein, and Scott Alfeld. 2022. Hard to forget: Poisoning attacks on certified machine unlearning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 36. 7691–7700.
[24] Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, and Quoc Viet Hung Nguyen. 2022. A survey of machine unlearning. arXiv preprint arXiv:2209.02299 (2022).
[25] Martin Pawelczyk, Seth Neel, and Himabindu Lakkaraju. 2023. In-Context Unlearning: Language Models as Few Shot Unlearners. arXiv preprint arXiv:2310.07579 (2023).
[26] Wei Qian, Chenxu Zhao, Huajie Shao, Minghan Chen, Fei Wang, and Mengdi Huai. 2022. Patient similarity learning with selective forgetting. In 2022 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 529– 534.
[27] General Data Protection Regulation. 2018. General data protection regulation (GDPR). Intersoft Consulting, Accessed in October 24, 1 (2018).
[28] Shawn Shan, Emily Wenger, Jiayun Zhang, Huiying Li, Haitao Zheng, and Ben Y Zhao. 2020. Fawkes: Protecting privacy against unauthorized deep learning models. In 29th USENIX security symposium (USENIX Security 20). 1589–1604.
[29] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. 2023. Large language models encode clinical knowledge. Nature 620, 7972 (2023), 172–180.
[30] Ayush K Tarun, Vikram S Chundawat, Murari Mandal, and Mohan Kankanhalli. 2023. Fast yet effective machine unlearning. IEEE Transactions on Neural Networks and Learning Systems (2023).
[31] Zixiang Wang, Linzheng Chai, Jian Yang, Jiaqi Bai, Yuwei Yin, Jiaheng Liu, Hongcheng Guo, Tongliang Li, Liqun Yang, Zhoujun Li, et al. 2023. MT4CrossOIE: Multi-stage Tuning for Cross-lingual Open Information Extraction. arXiv preprint arXiv:2308.06552 (2023).
[32] Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. 2022. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International Conference on Machine Learning. PMLR, 23965–23998.
[33] Heng Xu, Tianqing Zhu, Lefeng Zhang, Wanlei Zhou, and Philip S. Yu. 2023. Machine Unlearning: A Survey. ACM Comput. Surv. 56, 1, Article 9 (aug 2023), 36 pages.
[34] Haonan Yan, Xiaoguang Li, Ziyao Guo, Hui Li, Fenghua Li, and Xiaodong Lin. 2022. Arcane: An efficient architecture for exact machine unlearning. In  Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22. 4006–4013.
[35]  Yuanshun Yao, Xiaojun Xu, and Yang Liu. 2023. Large Language Model Unlearning. arXiv:2310.10683 [cs.CL]
[36] Santiago Zanella-Béguelin, Lukas Wutschitz, Shruti Tople, Victor Rühle, Andrew Paverd, Olga Ohrimenko, Boris Köpf, and Marc Brockschmidt. 2020. Analyzing information leakage of updates to natural language models. In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security. 363–375.
[37] Yang Zhang, Fuli Feng, Jizhi Zhang, Keqin Bao, Qifan Wang, and Xiangnan He. 2023. CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation. arXiv:2310.19488 [cs.IR]
[38] Yang Zhang, Zhiyu Hu, Yimeng Bai, Fuli Feng, Jiancan Wu, Qifan Wang, and Xiangnan He. 2023. Recommendation unlearning via influence function. arXiv preprint arXiv:2307.02147 (2023).
[39] Yang Zhang, Tianhao Shi, Fuli Feng, Wenjie Wang, Dingxian Wang, Xiangnan He, and Yongdong Zhang. 2023. Reformulating CTR Prediction: Learning Invariant Feature Interactions for Recommendation. arXiv preprint arXiv:2304.13643 (2023).
[40] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. 2023. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923 (2023).
[41] Cai-Nicolas Ziegler, Sean M McNee, Joseph A Konstan, and Georg Lausen. 2005. Improving recommendation lists through topic diversification. In Proceedings of the 14th international conference on World Wide Web. 22–32.

