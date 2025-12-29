# Data-efficient Fine-tuning for LLM-based Recommendation
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ce7f/ce7f10a7-57d6-4ef8-9a60-ba799b0420dd.png" style="width: 50%;"></div>
Tat-Seng Chua dcscts@nus.edu.sg National University of Singapore Singapore
# ABSTRACT
Leveraging Large Language Models (LLMs) for recommendation has recently garnered considerable attention, where fine-tuning plays a key role in LLMs’ adaptation. However, the cost of fine-tuning LLMs on rapidly expanding recommendation data limits their practical application. To address this challenge, few-shot fine-tuning offers a promising approach to quickly adapt LLMs to new recommendation data. We propose the task of data pruning for efficient LLMbased recommendation, aimed at identifying representative samples tailored for LLMs’ few-shot fine-tuning. While coreset selection is closely related to the proposed task, existing coreset selection methods often rely on suboptimal heuristic metrics or entail costly optimization on large-scale recommendation data. To tackle these issues, we introduce two primary objectives for the data pruning task in the context of LLM-based recommendation: 1) high accuracy aims to identify the influential samples that can lead to high overall performance; and 2) high efficiency underlines the low costs of the data pruning process. To pursue the two objectives, we propose a novel data pruning method incorporating two scores, namely influence score and effort score, to efficiently identify the influential samples. Particularly, the influence score is introduced to accurately estimate the influence of removing each sample on the overall performance. To achieve low costs of the data pruning process, we employ a small-sized surrogate model to replace LLMs to obtain the influence score. Considering
∗Corresponding author. This work is supported by the CCCD Key Lab of Ministr Culture and Tourism.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. SIGIR ’24, July 14–18, 2024, Washington, DC, USA © 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0431-4/24/07 https://doi.org/10.1145/3626772.3657807
the potential gap between the surrogate model and LLMs, we further propose an effort score to prioritize some hard samples specifically for LLMs. We instantiate the proposed method on two competitive LLM-based recommender models, and empirical results on three real-world datasets validate the effectiveness of our proposed method. In particular, our method uses only 2% samples to surpass the full data fine-tuning, reducing time costs by 97%.
# CCS CONCEPTS • Information systems →Recommender systems.
KEYWORDS
Data Pruning, LLM-based Recommendation, Efficient Fine-tuning ACM Reference Format: Xinyu Lin, Wenjie Wang∗, Yongqi Li, Shuo Yang, Fuli Feng, Yinwei Wei, and Tat-Seng Chua. 2024. Data-efficient Fine-tuning for LLM-based Recommendation. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24), July 14–18, 2024, Washington, DC, USA. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3626772.3657807
# 1 INTRODUCTION
Leveraging Large Language Models (LLMs) for recommendation has demonstrated promising efficacy across various tasks, including Click-Through Rate (CTR) prediction [4], sequential recommendation [35], and explainable recommendation [11]. To build LLM-based recommender models, it is crucial to fine-tune LLMs on recommendation data for two primary reasons: 1) there exists a significant gap between previous LLMs’ tuning tasks and the recommendation tasks [4], and 2) the rapid and continuous update of recommendation data necessitates frequent fine-tuning of LLMs [38]. For example, there are approximately 160 million new videos and 942 billion interactions emerging on TikTok per day1. Thus, frequent fine-tuning is imperative to incorporate up-todate item information and enhance user behavior comprehension.
1https://www.tiktok.com/transparency/.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/febf/febfed2a-fa51-4c55-b024-37926dff3b8a.png" style="width: 50%;"></div>
Model
GPU (GiB)
Time
BIGRec
18.60 ⇥4GPU
36.87h
SASRec
1.61 ⇥1GPU
0.45h
% Red
97.84%
98.78%
.
Figure 2: The comparison of training costs between an LLM (BIGRec) and a surrogate model (SASRec)3. (b) The comparison of training costs  between an LLM (BIGRec) and a  surrogate  model  (SASRec).  The  statistics are based on NVIDIA RTX  A5000 on Games. 
<div style="text-align: center;">(a) Few-shot Performance. (a) Few-shot performance on  MicroLens-50K. (a) Effect of ! w.r.t. Recall</div>
Figure 1: image model (SASRec). To overcome the above issues, we summarize two principal objecFigure 1: (a) reveals that BIGRec achieves remarkable performance with only hundreds of samples. (b) shows the low costs of surrogate models.
tives for data pruning in the context of LLM-based recommendation: 1) high accuracy, which focuses on selecting the samples that can lead to low empirical risk; and 2) high e￿ciency, which emphasizes the low costs of the data pruning process, i.e., eliminating the dependency of well-trained LLMs on the full data. Nevertheless, pursuing the two objectives faces two challenges: • To achieve high accuracy, it is essential to measure the in￿uence of removing each training sample on the empirical risk. However, assessing the in￿uence of all samples is costly, as it requires the leaving-one-out retraining for each sample [39]. • To achieve high e￿ciency, one possible solution is to train a surrogate model for sample selection, e.g., using a small-sized traditional recommender model, which can drastically reduce the GPU memory usage and the training time compared to LLMs (see Table ??). However, there exists a gap between LLMs and surrogate models, attributable to their divergent capabilities in learning user behaviors (refer to Figure 4). As such, in￿uential samples selected by surrogate models might deviate from the ones on LLMs, potentially hurting the adaptation of LLMs. To address the challenges, we propose a novel Data pruning method, to E￿ciently identify the in￿uentiAl samples for LLMbased Recommender ￿ne-tuning (shorted as DEALRec). DEALRec leverages two scores, namely in￿uence score and e￿ort score, to identify the in￿uential samples. The influence score is formulated to estimate the in￿uence of removing each training sample on the empirical risk. It is calculated by extending the in￿uence function [16] via chain rules and second-order optimization techniques [25]. To e￿ciently calculate the in￿uence score for all samples, DEALRec employs a simple yet e￿ective symmetric property to accelerate the calculation, requiring only the estimation once for all samples (cf. Section 3.1). Thereafter, DEALRec uses a traditional recommender model as a surrogate model to obtain the in￿uence score and introduces the e￿ort score to mitigate the gap between the surrogate model and LLMs. The e￿ort score is obtained by calculating the gradient norm of a sample loss w.r.t. the parameters of LLMs, intuitively measuring the e￿ort of LLMs to ￿t a speci￿c sample. By regularizing the in￿uence score with the e￿ort score, DEALRec identi￿es the in￿uential samples that encompass both the representativeness of the full data and the signi￿cance to LLMs. We instantiate DEALRec on two LLM-based recommender models and conduct extensive experiments on three real-world datasets, validating the superiority of DEALRec in terms of both e￿ciency and accuracy. The code and datasets are available BIGRec achieves remarkable perform on Games However, fine-tuning LLMs on large-scale recommendation data demands substantial computational resources and time costs [26], thereby diminishing the practicality of LLM-based recommender models in real-world applications. As such, it is essential to enhance the fine-tuning efficiency of LLM-based recommender models. Fortunately, the rich world knowledge encoded in LLMs offers a promising solution for efficient fine-tuning: few-shot fine-tuning. Previous studies have uncovered that LLMs have the potential to quickly adapt to recommendation tasks by fine-tuning on randomly sampled few-shot data [3, 4, 27] (Figure 1(a)), significantly reducing training time and computational costs. Despite its efficiency, randomly sampled data may lack sufficient representativeness to enable LLMs to effectively comprehend new items and user behaviors. To combat this issue, we introduce the task of data pruning for efficient LLM-based recommendation, which aims to identify representative samples tailored for LLMs’ few-shot finetuning. A closely related literature to this data pruning task is coreset selection [13]. It tries to select a small but representative subset from the full data, aiming to achieve comparable performance. Existing coreset selection methods generally fall into two categories2: 1) Heuristic methods select hard or diverse samples based on predefined metrics [30, 34, 49]. Such heuristic methods do not estimate the impact of selected samples on empirical risk, possibly leading to suboptimal coreset selection. 2) Optimization-based methods mainly optimize the selection of subsets to minimize the empirical risk [5, 50]. However, these methods are inapplicable to large-scale recommendation datasets due to the complex and costly bi-level or discrete optimization problem [17]. Worse still, both heuristic and optimization-based methods rely on the model well-trained by the full data to select the coreset, e.g., calculating pre-defined scores or optimizing the data subset based on the well-trained model (cf. Section 2). As such, it is infeasible to directly apply these methods for LLM-based recommendation because of the high training costs of LLMs on the large-scale full recommendation data. To overcome the above issues, we summarize two principal objectives for data pruning in the context of LLM-based recommendation: 1) high accuracy, which focuses on selecting the samples that can lead to low empirical risk; and 2) high efficiency, which emphasizes the low costs of the data pruning process, i.e., eliminating the dependency of well-trained LLMs on the full data. Nevertheless, pursuing the two objectives faces two challenges:
of both e￿ciency and accuracy. The code and datasets are a 2More detailed related work is discussed and compared in Section 4 and 5.
In summary, this work o￿ers three major contributions: • We introduce a data pruning task to identify the in￿uential samples tailored for e￿cient LLM-based recommender ￿netuning, unlocking the remarkable potential of applying LLMbased recommender models to real-world platforms. • We propose a novel data pruning method to discover the in￿uential samples for LLM-based recommendation, which e￿ectively and e￿ciently assesses the in￿uence of removing a sample on empirical risk. • We conduct extensive experiments on three real-world datasets, demonstrating the e￿ectiveness of DEALRec in achieving both high e￿ciency and accuracy. • To achieve high accuracy, it is essential to measure the influence of removing each training sample on the empirical risk. However, assessing the influence of all samples is costly, as it requires the leaving-one-out retraining for each sample [43]. • To achieve high efficiency, one possible solution is to train a surrogate model for sample selection, e.g., using a small-sized traditional recommender model, which can drastically reduce the GPU memory usage and the training time compared to LLMs (see Figure 1(b)). However, there exists a gap between LLMs and surrogate models, attributable to their divergent capabilities in learning user behaviors (refer to Figure 3). As such, influential samples selected by surrogate models might deviate from the ones on LLMs, potentially hurting the adaptation of LLMs.
In summary, this work o￿ers three major contributions:  We introduce a data pruning task to identify the in￿uential samples tailored for e￿cient LLM-based recommender ￿netuning, unlocking the remarkable potential of applying LLMbased recommender models to real-world platforms.  We propose a novel data pruning method to discover the in￿uential samples for LLM-based recommendation, which e￿ectively and e￿ciently assesses the in￿uence of removing a sample on empirical risk.  We conduct extensive experiments on three real-world datasets, demonstrating the e￿ectiveness of DEALRec in achieving both high e￿ciency and accuracy. • To achieve high accuracy, it is essential to measure the influence of removing each training sample on the empirical risk. However, assessing the influence of all samples is costly, as it requires the leaving-one-out retraining for each sample [43]. • To achieve high efficiency, one possible solution is to train a surrogate model for sample selection, e.g., using a small-sized traditional recommender model, which can drastically reduce the GPU memory usage and the training time compared to LLMs (see Figure 1(b)). However, there exists a gap between LLMs and surrogate models, attributable to their divergent capabilities in learning user behaviors (refer to Figure 3). As such, influential samples selected by surrogate models might deviate from the ones on LLMs, potentially hurting the adaptation of LLMs.
2 TASK FORMULATION In this section, we ￿rst introduce LLM-based recommender models and uncover the challenge of real-world applicability. Thereafter, we formulate the task of data pruning for LLM-based recommendation and compare the related work on coreset selection. • LLM-based recommender models. To leverage the competent capabilities of LLMs, LLM-based recommendation typically utilize powerful LLMs directly as the recommender models. Since LLMs are not particularly trained on the recommendation data, ￿ne-tuning is the necessary and key step for LLMs to learn the item knowledge and understand user behavior. Let U and I denote the sets of users and items, respectively. We present each training sample, i.e., user sequence, as B = (G,~), where G = [81,82, . . .,8|G |] is the user’s historical interactions in chronological order, and ~ is the next interacted item of the user4, where {81, . . .,8|G |,~} ⇢I. Formally, given the user sequences of the training set D = {BD|D 2 U}, the target is to ￿ne-tune an LLM for recommendation tasks. The learnable parameters (q 2 Φ) of an LLM is optimized by minimizing the negative log-likelihood of the next interacted item~ conditioned on input G: min q2Φ{L!!" q = − |~| ’ C=1 log%q (~C |~<C,G)}, (1) where ~C denotes the C-th token of ~, and ~<C represents the token sequence preceding ~C. While ￿ne-tuning LLMs has demonstrated e￿ectiveness in recommendation tasks [30], its practical application is hindered by the high resource costs required by LLMs and the continuous in￿ux of new recommendation data [35]. Hence, it is essential to enhance the e￿ciency of LLM-based recommender ￿ne-tuning. • Data pruning for e￿cient LLM-based recommendation. To achieve e￿cient LLM-based recommendation, a promising approach is to reduce the costs by few-shot ￿ne-tuning with randomly selected samples [4]. Nevertheless, the random samples might lose some crucial information for LLMs to acquire the latest information on user behavior or items, e.g., trending items. In this with only hundreds of samples  To address the challenges, we propose a novel Data pruning method, to Efficiently identify the influentiAl samples for LLMbased Recommender fine-tuning (shorted as DEALRec). DEALRec leverages two scores, namely influence score and effort score, to identify the influential samples. The influence score is formulated to estimate the influence of removing each training sample on the empirical risk. It is calculated by extending the influence function [15] via chain rules and second-order optimization techniques [24]. To efficiently calculate the influence score for all samples, DEALRec employs a simple yet effective symmetric property to accelerate the calculation, requiring only the estimation once for all samples (cf. Section 3.1). Thereafter, DEALRec uses a traditional recommender model as a surrogate model to obtain the influence score and introduces the effort score to mitigate the gap between the surrogate model and LLMs. The effort score is obtained by calculating the gradient norm of a sample loss w.r.t. the parameters of LLMs, intuitively measuring the effort of LLMs to fit a specific sample. By regularizing the influence score with the effort score, DEALRec identifies the influential samples that encompass both the representativeness of the full data and the significance to LLMs. We instantiate DEALRec on two LLM-based recommender models and conduct extensive experiments on three real-world datasets, validating the superiority of DEALRec in terms of both efficiency and accuracy. The code and datasets are available at https://github.com/Linxyhaha/DEALRec. In summary, this work offers three major contributions: • We introduce a data pruning task to identify the influential samples tailored for efficient LLM-based recommender finetuning, unlocking the remarkable potential of applying LLMbased recommender models to real-world platforms. • We propose a novel data pruning method to discover the influential samples for LLM-based recommendation, which effectively and efficiently assesses the influence of removing a sample on empirical risk. • We conduct extensive experiments on three real-world datasets, demonstrating the effectiveness of DEALRec in achieving both high efficiency and accuracy.
# ight, we introduce the task of data pru ecommendation, which aims to ide 2 TASK FORMULATION
recommendation, which aims to identify a set of representative samples particularly for LLMs’ few-shot ￿ne-tuning. Formally, given all training samples D = {BD|D 2 U}, the target of data 4Our main focus lies in sequential recommendation, which holds notable practical In this section, we first introduce LLM-based recommender models and uncover the challenge of real-world applicability. Thereafter, we formulate the task of data pruning for LLM-based recommendation and compare the related work on coreset selection.
• LLM-based recommender models. To leverage the competent capabilities of LLMs, LLM-based recommendation typically utilize powerful LLMs directly as the recommender models. Since LLMs are not particularly trained on the recommendation data, fine-tuning is the necessary and key step for LLMs to learn the item knowledge and understand user behavior. Let U and I denote the sets of users and items, respectively. We present each training sample, i.e., user sequence, as 𝑠= (𝑥,𝑦), where 𝑥= [𝑖1,𝑖2, . . . ,𝑖|𝑥|] is the user’s historical interactions in chronological order, and 𝑦 is the next interacted item of the user3, where {𝑖1, . . . ,𝑖|𝑥|,𝑦} ⊂I. Formally, given the user sequences of the training set D = {𝑠𝑢|𝑢∈ U}, the target is to fine-tune an LLM for recommendation tasks. The learnable parameters (𝜙∈Φ) of an LLM is optimized by minimizing the negative log-likelihood of the next interacted item𝑦conditioned on input 𝑥:
∑︁ where 𝑦𝑡denotes the 𝑡-th token of 𝑦, and 𝑦<𝑡represents the token sequence preceding 𝑦𝑡. While fine-tuning LLMs has demonstrated effectiveness in recommendation tasks [29], its practical application is hindered by the high resource costs required by LLMs and the continuous influx of new recommendation data [38]. Hence, it is essential to enhance the efficiency of LLM-based recommender fine-tuning.
# • Data pruning for efficient LLM-based recommendation. To achieve efficient LLM-based recommendation, a promising
• To achieve efficient LLM-based recommendation, a promising approach is to reduce the costs by few-shot fine-tuning with randomly selected samples [4]. Nevertheless, the random samples might lose some crucial information for LLMs to acquire the latest information on user behavior or items, e.g., trending items. In this light, we introduce the task of data pruning for efficient LLM-based recommendation, which aims to identify a set of representative samples particularly for LLMs’ few-shot fine-tuning. Formally, given all training samples D = {𝑠𝑢|𝑢∈U}, the target of data pruning is to select a subset S ⊂D, such that the LLMs trained on the subset S can yield good performance on the testing set. The size of S is controlled by the given selection ratio 𝑟, i.e., |S| = 𝑟|D|.
(2)
where L(·) is the loss function of the task, e.g., image classification [16] or CTR prediction [14], and 𝐻(·) denotes the heuristic strategy such as selecting samples with larger prediction entropy [7], or clustering the samples based on the sample representations [6]. However, this group of methods designs the strategy 𝐻(·) intuitively and fails to explicitly consider the influence of a sample on the empirical risk. This might lead to suboptimal selection, thereby declining the performance of the model trained by the selected subset.
3Our main focus lies in sequential recommendation, which holds notable practical significance by intricately considering the temporal aspect in real-world scenarios.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9a8b/9a8b1c64-c3ef-428f-9359-4968c34adc9c.png" style="width: 50%;"></div>
Figure 2: Overview of DEALRec. DEALRec first trains a surrogate model on the full training samples. Subsequently, it calculates the influence score, which is then regularized by the effort score, to identify influential samples.
2) Optimization-based methods [5, 22, 23, 48] mainly utilize bilevel optimization techniques to learn the best subset chosen for training:
Besides, there is also some work that employs discrete optimization problems based on the empirical minimizer ˆ𝜃in Eq. (2). Nevertheless, they struggle to be applied to large-scale datasets e.g., recommendation data, due to the complex solving of the optimization problem [17].
# 3 DEALREC
To pursue efficient LLM-based recommendation, we propose a novel data pruning method DEALRec, which involves two key components, i.e., the influence score to estimate the influence on empirical risk, and the effort score as a regularization to mitigate the gap between surrogate model and LLMs. The overview of our method is presented in Figure 2.
# 3.1 Influence Score
To achieve good overall performance with the model trained on the pruned dataset S, the key lies in the ability to assess the influence on the empirical risk, i.e., overall performance, caused by removing a sample in training. However, simply assessing the the influence by removing each sample is impractical, because it requires brute force leaving-one-out-retraining for 𝑛= |D| times. To overcome this challenge, we propose an efficient approximation of the influence for all samples by extending influence on parameter change (i.e., a classic result from influence function [24]) via chain rule and secondorder optimization techniques. We further utilize the symmetric property to speed up the calculation of the influence score.
• Influence on parameter change. To estimate the influence on empirical risk for each sample, we first start with the classic result [28] from research on influence function [8], which gives us the estimation of the parameter change caused by upweighting a sample 𝑠for training. Considering a training sample 𝑠is upweighted by a small 𝜖, the empirical minimizer can be rewritten as: ∑︁
(4)
According to [28], the influence of upweighting a sample 𝑠on the parameter change is then given as: �
(5)
��� where 𝐻ˆ𝜃 = 1 𝑛 � 𝑠𝑖∈D ∇2 𝜃L(𝑠𝑖, ˆ𝜃) is the Hessian and positive definite by assumption, Iparam(𝑠) ∈R𝑚, and 𝑚is the number of parameters. Notably, assigning −1 𝑛to 𝜖is equivalent to removing the sample 𝑠from training. As such, the parameter change of removing a training sample 𝑠can be linearly approximated as:
(6)
where ˆ𝜃−𝑠= arg min𝜃∈Θ � 𝑠𝑖∈D,𝑠𝑖≠𝑠L(𝑠𝑖,𝜃). Based on Eq. (6), an intuitive approach to assess the sample influence for model training is to utilize the L2 norm of a sample’s influence on parameter change or an additional discrete optimization problem as proposed in [50]. Nevertheless, large parameter changes do not necessarily lead to performance improvements. Besides, calculating Eq. (6) for all training samples can be computationally costly [17] and is infeasible for recommendation data. To alleviate the issues, we propose an efficient approximation for the influence of removing a sample on the empirical risk. • Influence on empirical risk. Based on the parameter change obtained via the influence function, we can then estimate the influence of upweighting a training sample 𝑠by a small 𝜖on the loss of an arbitrary sample 𝑠′: �
Similarly, the influence of removing a training sample 𝑠on the loss of an arbitrary sample 𝑠′ can be linearly approximated as:
(8)
We can then obtain the influence of removing a sample 𝑠on the empirical risk (i.e., influence score) by � � � �
(9)
���������������������������������������������������������������������������������������������� However, it is non-trivial to directly obtain 𝐻−1 ˆ𝜃 as forming and inverting 𝐻ˆ𝜃= 1 𝑛 � 𝑠𝑖∈D ∇2 𝜃L(𝑠𝑖, ˆ𝜃) requires O(𝑛𝑚2 + 𝑚3) with
Algorithm 1 Procedure of HVP Estimation
Input: Original training dataset D, parameters of a well-trained model ˆ𝜃,
iteration number 𝑇.
1: Compute �
𝑖1
𝑛∇𝜃L(𝑠𝑖, ˆ𝜃) for ∀𝑖∈{1, . . . ,𝑛}.
2: Initialize ˜𝐻−1
0
��
𝑖1
𝑛∇𝜃L(𝑠𝑖, ˆ𝜃)
�
= �
𝑖1
𝑛∇𝜃L(𝑠𝑖, ˆ𝜃).
3: for all 𝑡∈{1, . . . ,𝑇} do
4:
Randomly sample a training sample 𝑠𝑡∈D;
5:
Calculate ∇2
𝜃L(𝑠𝑡) as the unbiased estimator of 𝐻;
6:
˜𝐻−1
𝑡
��
𝑖1
𝑛∇𝜃L(𝑠𝑖, ˆ𝜃)
�
←�
𝑖1
𝑛∇𝜃L(𝑠𝑖, ˆ𝜃)+
�
𝐼−∇2
𝜃L(𝑠𝑡)
�˜𝐻−1
𝑡−1
��
𝑖1
𝑛∇𝜃L(𝑠𝑖, ˆ𝜃)
�
;
⊲Eq. (10)
7:
˜𝐻−1 ��
𝑖1
𝑛∇𝜃L(𝑠𝑖, ˆ𝜃)
�
←˜𝐻−1
𝑇
��
𝑖1
𝑛∇𝜃L(𝑠𝑖, ˆ𝜃)
�
.
Output: Unbiased estimation ˜𝐻−1 ��
𝑖1
𝑛∇𝜃L(𝑠𝑖, ˆ𝜃)
�
.
𝑛training samples and 𝜃∈R𝑚. This results in cumbersome calculation of influence scores for all training samples.
𝑛training samples and 𝜃∈R𝑚. This results in cumbersome calculation of influence scores for all training samples. • Efficient estimation of influence score. To achieve efficient computation of influence score, we utilize stochastic-based HessianVector Products (HVP) [1] to efficiently approximate 𝐻−1 ˆ𝜃∇𝜃L(𝑠, ˆ𝜃). The idea of stochastic-based HVP estimation is to iteratively obtain an unbiased estimator of 𝐻ˆ𝜃and approach the unbiased estimation of HVP, i.e., 𝐻−1 ˆ𝜃∇𝜃L(𝑠, ˆ𝜃). Specifically, we omit the ˆ𝜃subscript for clarity and write the first 𝑗terms in Taylor expansion of 𝐻−1 as 𝐻−1 𝑗 def = �𝑗 𝑖=0(𝐼−𝐻)𝑖, which can be further rewritten recursively as 𝐻−1 𝑗 = 𝐼+ (𝐼−𝐻)𝐻−1 𝑗−1. From the validity of the Taylor expansion, we have 𝐻−1 𝑗 →𝐻−1 as 𝑗→∞. Thereafter, denoting ∇𝜃L(𝑠, ˆ𝜃) as 𝑣, the update iteration for the estimated 𝐻−1 ˆ𝜃∇𝜃L(𝑠, ˆ𝜃) at step 𝑡 can be written as: � �
(10)
� � where𝑠𝑡is a training sample randomly drawn from D, and ∇2 𝜃L(𝑠𝑡) is an unbiased estimator of the 𝐻at step 𝑡for fast-to-compute HVP [24]. Despite that stochastic-based HVP can alleviate the computation burdens of the estimation, calculating the influence score for each sample is still costly due to the independent 𝑛 estimations of 𝐻−1 ˆ𝜃∇𝜃L(𝑠, ˆ𝜃) for each 𝑠∈D (refer to Eq. (9)). To further enhance the efficiency of acquiring influence scores for all samples, we use symmetric property to rewrite Eq. (9) into: � �
(11)
������������������������������������������������������������ The reformulation is based on the assumption that L(·) has continuous second-order derivatives, which is consistent with the assumption for influence function [24], leading to the fact that 𝐻−1 ˆ𝜃 is symmetric. Since 𝐻−1 ˆ𝜃 �� 𝑖1 𝑛∇𝜃L(𝑠𝑖, ˆ𝜃) � ∈R𝑚is a constant vector for any sample 𝑠∈D, we can efficiently obtain influence scores for all samples by only applying HVP estimation once for 𝐻−1 ˆ𝜃 �� 𝑖1 𝑛∇𝜃L(𝑠𝑖, ˆ𝜃) � . The detailed HVP estimation process is illustrated in Algorithm 1.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6186/618646a9-597e-4cd6-ab1a-8e1e640c0e8a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: (a) depicts the different learning ability due to the prior knowledge in LLMs. (b) presents the distributions of effort scores of LLM and surrogate model on Games dataset4.</div>
# (a) Depicts the some users are easi (b) Presents the distributions of dis surrogate model, showing the diffe 3.2 Gap Regularization
surrogate model, showing the different learning ability of the two  models.  As shown in Eq. (11), assessing the influence score of a sample requires the optimized parameters ˆ𝜃well-trained over all training samples D. Nevertheless, this poses challenges for LLM-based recommender models due to the continuous influx of large-scale new data in real-world scenarios. In this light, we propose to utilize a surrogate model to replace the LLMs and introduce an effort score as a gap regularization to complement the learning ability gap between LLMs and the surrogate models.
# • Surrogate model. To reduce the costs, we propose utili surrogate model, a small-sized traditional recommender 
• Surrogate model. To reduce the costs, we propose utilizing a surrogate model, e.g., a small-sized traditional recommender model, to compute the influence scores. Nevertheless, since LLMs acquire rich world knowledge during the pre-training stage, they intricately possess different learning abilities compared to the surrogate model (Figure 3(a)). Therefore, the influential samples on LLMs might deviate from the ones for LLMs.
• Effort score. To compensate for the gap, we introduce the effort score, which aims to capture significant samples particularly for LLMs. Specifically, we define the effort score of a sample, i.e., a user sequence, 𝑠as:
where 𝜙is the learnable parameters of LLMs5. Intuitively, it measures the learning effort of LLMs to fit a specific user sequence, and a larger score indicates a harder sample for LLMs to learn. To elaborate, Eq. (12) measures the change in the model parameters, which can be interpreted as the discrepancy from the current knowledge encoded in LLMs’ parameters to the latest item knowledge or user behavior. As such, the effort score can emphasize significant samples particularly for LLMs, supplementing the different learning ability of the surrogate model (Figure 3(b)). • Overall score. By injecting the signals of LLMs’ learning ability into the calculation of influence score, we can obtain the final score of each user sequence for LLM-based recommender fine-tuning: �∑︁ �
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/dea6/dea6fb86-38b8-468b-8822-e22109a66313.png" style="width: 50%;"></div>
�������������������������������������������������������������������������������������������������� 4We obtain the effort scores for surrogate model by calculating the gradient norm of the parameters of the surrogate model (Eq. (12)). 5The learnable parameters can be either the whole parameters of LLMs or the learnable parameters from parameter-efficient training, e.g., LoRA [19].
Algorithm 2 Procedure of DEALRec
Input: Original training dataset D, randomly initialized parameters of
surrogate model 𝜃, pre-trained parameters of LLM 𝜙.
1: ˆ𝜃= arg min𝜃∈Θ
1
𝑛
�
𝑠𝑖∈D L(𝑠𝑖,𝜃).
2: Obtain estimated 𝐻−1 ��
𝑖1
𝑛∇𝜃L(𝑠𝑖, ˆ𝜃)
�
via HVP estimation.
3: for all 𝑖∈{1, . . . ,𝑛} do
4:
𝐼𝑠𝑖=
1
𝑛2 ∇𝜃L(𝑠𝑖, ˆ𝜃)T𝐻−1
ˆ𝜃
��
𝑗1
𝑛∇𝜃L(𝑠𝑗, ˆ𝜃)
�
+
𝜆∥∇𝜙L𝐿𝐿𝑀(𝑠𝑖) ∥2; ⊲Eq. (13)
5: G = {𝐺1, . . . ,𝐺𝐾} ←Split training samples D into 𝐾groups
according to the final score 𝐼𝑠with even range width.
6: S ←∅, 𝐵←⌊𝑟|D|
𝐾⌋.
7: while G ≠∅do
8:
𝑘∗= arg min𝑘|𝐺𝑘|;
9:
S𝑘∗←randomly select min{𝐵, |𝐺𝑘∗|} samples from 𝐺𝑘∗;
10:
S ←S ∪S𝑘∗; G ←G \ {𝐺𝑘∗};
11:
𝐵←⌊𝑟|D|−|S|
|G|
⌋;
⊲Update sampling budget
Output: Selected samples S for few-shot fine-tuning.
where 𝜆is a hyper-parameter to balance the strength of the gap regularization. Notably, the gap regularization would suppress the easy samples with smaller effort scores while emphasizing the samples that are more difficult to learn, i.e., larger effort scores. Intuitively, DEALRec identifies the influential samples with two key considerations: 1) the influence score focuses on selecting the representative samples from the full dataset, capturing collaborative filtering information for low empirical risk; and 2) the effort score highlights the non-trivial samples that are significant to the learning of LLMs. The effectiveness of the two scores is empirically validated in Section 4.3.1.
# 3.3 Few-shot Fine-tuning
Based on the final influential score obtained via Eq. (13), we can select a subset of data S for LLMs’ few-shot fine-tuning, given an expected selection ratio 𝑟.
• Few-shot data coverage. A straightforward approach is to select the data greedily, i.e., rank the samples based on the overall scores, and then select the top-𝑟percentage of the training data. However, greedily selecting the samples with higher scores might result in very similar samples with low data coverage, which leads to: 1) Inadequacy of samples from other areas, thus hurting the bounded empirical risk [57] and lowering the overall performance (cf. Section 4.2). 2) Poor utilization of training samples because of the redundant samples with similar patterns, thereby causing suboptimal selection for few-shot fine-tuning.
# • Coverage-enhanced sample selection. To address th issues, we follow [57] to select the users based on the 
• issues, we follow [57] to select the users based on the idea of stratified sampling. The core idea is to maintain the budget for the samples in different areas of training distribution, such that the data coverage will be improved to ensure a high-probability bound for the empirical risk (refer to [57] for detailed proof). In detail, we first divide the samples into 𝐾groups according to their overall scores. We then iteratively sample 𝑛𝑠user sequences from the group with the fewest samples and discard that group after sampling, where 𝑛𝑠is the average sampling budget for all groups
<div style="text-align: center;">Table 1: Statistics of the three datasets.</div>
Table 1: Statistics of the three datasets.
Datasets
# Users
# Items
# Interactions
Density
Games
49,156
17,332
342,329
0.04%
MicroLens-50K
49,887
19,217
359,048
0.04%
Book
88,263
86,272
5,303,707
0.07%
and is initialized with ⌊𝑟| D| 𝐾⌋. If the group size is smaller than the average sampling budget, we select all users from this group and update the average sampling budget for the remaining groups (see Algorithm 2). Based on the selected few-shot samples S, we optimize the learnable parameters (𝜙∈Φ) of LLMs: ∑︁
Based on the selected few-shot samples S, we optimize the learnable parameters (𝜙∈Φ) of LLMs: ∑︁
• Instantiation. To instantiate DEALRec on LLM-based recommender models, we first employ a surrogate model to train on original training samples D and calculate the influence score for all samples via Eq. (11), where the L(·) can be any form of the loss function from the surrogate model, e.g., BPR [37]. We then obtain the effort score for LLMs via Eq. (12), where 𝜙can be the learnable parameters from any backend LLM-based recommender models. Eventually, we apply the stratified sampling to select the samples for LLMs’ few-shot fine-tuning. The detailed data pruning process of DEALRec is demonstrated in Algorithm 2.
# 4 EXPERIMENT
We conduct extensive experiments on three real-world datasets to answer the following research questions: • RQ1: How does our proposed DEALRec perform compared to the coreset selection baselines for LLM-based recommendation and the models trained with full data? • RQ2: How do the different components of DEALRec (i.e., influence score, gap regularization, and stratified sampling) affect the performance, and is DEALRec generalizable to different surrogate models? • RQ3: How does DEALRec perform under different selection ratios and how does DEALRec improve the overall performance?
# 4.1 Experimental Settings
4.1.1 Datasets. We conduct experiments on three real-world recommendation datasets: 1) Games is from the Amazon review datasets6, which covers interactions between users and video games with rich textual features. 2) MicroLens-50K7 is a newly released micro-video recommendation dataset [33]. It contains 50𝑘users’ interactions with micro-videos and their associated multimodal features. 3) Book is also from Amazon review datasets, containing users’ interactions with extensive books. For Games and Book, we follow previous work and discard the interactions with the ratings < 4. For the three datasets, we sort all user-item interactions according to the global timestamps, and then split the interactions into training, validation, and testing sets with the ratio of 8:1:1. Besides, we consider two different fine-tuning settings as follows:
6https://jmcauley.ucsd.edu/data/amazon/. 7https://github.com/westlake-repl/MicroLens/.
1) Few-shot fine-tuning fine-tunes LLM-based recommender models with limited samples at a fixed size, e.g., 1024-shot, obtained via different data pruning methods. 2) Full fine-tuning utilizes all samples to fine-tune LLM-based recommender models without data pruning.
4.1.2 Baselines. We compare DEALRec with the random sampling and several competitive coreset selection methods, including difficulty-based methods and diversity-based methods: 1) Random obtains the data subset via random sampling, which is a popular and strong baseline in data-efficient training [13]. 2) GraNd [34] is a representative coreset selection method that selects the difficult samples with larger gradient norms during training. 3) EL2N [34] proposes to select the difficult samples with larger errors between the labels and the prediction from the model trained by the original dataset. 4) CCS [57] is a competitive method that selects the samples considering both high data coverage and sample importance. We use EL2N as the importance metric for CCS. 5) TF-DCon [49] is a recently proposed data pruning method for content-based recommendation, which clusters the user sequences based on the user representations obtained from both well-trained recommender models and LLMs for selection. 6) RecRanker [30] proposes a sampling strategy to select high-quality user sequences. It selects the users with more interactions for better user modeling and utilizes a cluster-based sampling strategy to enhance user diversity. We do not perform optimization-based methods for comparison because of the inapplicability of complex bi-level or discrete optimization for LLMs on large-scale recommendation data (cf. Section 2). We instantiate our proposed DEALRec and all baselines on two competitive backend LLM-based recommender models: 1) BIGRec [3] utilizes the item title to present the user sequence for recommendation generation; 2) TIGER [35] learns extra tokens from item features to present items, and then converts the user sequence into the sequence of the new item token for next-item generation. • Evaluation. We employ the widely used metrics Recall@𝐾and NDCG@𝐾to evaluate the models [18], with 𝐾set to 10 and 20 for Games, and 𝐾= 20 and 50 for MicroLens-50K and Book8.
4.1.3 Implementation. As for the two backend LLM-based recommender models, we follow the original settings in their paper for implementation. We employ LLaMA-7B for BIGRec and transformer-based architecture for TIGER as in their paper [35]. All fine-tuning experiments are conducted on four NVIDIA RTX A5000 GPUs. Besides, we adopt the parameter-efficient fine-tuning technique LoRA [19] to fine-tune BIGRec and fully fine-tune the parameters of TIGER. We utilize SASRec [20], a representative sequential recommender model, as the surrogate model in DEALRec. We set the iteration number 𝑇for HVP estimation at 5000, and search the regularization strength 𝜆in {0.1, 0.3, 0.5, 1.0, 2.0}. For cluster-based methods, the number of clusters 𝐾is explored in {25, 50, 75}. As for the coreset selection methods that require the training of LLMs, we consider a feasible implementation [7] by executing them on the same surrogate model as DEALRec.
8We report metrics@20 and @50 because of the challenging modeling of user behavior on book and micro-video recommendations, where the temporal shifts of user interests and the item feature is stronger and thus more difficult to capture [45, 46].
<div style="text-align: center;">Table 2: Overall performance comparison between the baselines and DEALRec instantiated on two competitive LLM-based recommender models on three datasets. For each backend model, the bold results highlight the best results while the second-best ones are underlined. ∗implies the improvements over the second-best results are statistically significant (𝑝-value < 0.01) under one-sample t-tests. We run all experiments for 3 times with different random seeds and report the averaged results.</div>
Games
MicroLens-50K
Book
1024-shot (𝒓=2%)
1024-shot (𝒓=2%)
1024-shot (𝒓=1%)
Methods
R@10
R@20
N@10
N@20
R@20
R@50
N@20
N@50
R@20
R@50
N@20
N@50
TF-DCon
0.0102
0.0157
0.0062
0.0078
0.0066
0.0099
0.0027
0.0034
0.0104
0.0144
0.0083
0.0092
RecRanker
0.0112
0.0166
0.0074
0.0090
0.0024
0.0042
0.0011
0.0014
0.0108
0.0145
0.0090
0.0097
CCS
0.0164
0.0246
0.0097
0.0122
0.0096
0.0131
0.0041
0.0049
0.0110
0.0145
0.0088
0.0096
GraNd
0.0158
0.0250
0.0098
0.0125
0.0014
0.0032
0.0006
0.0010
0.0102
0.0136
0.0080
0.0087
EL2N
0.0154
0.0256
0.0098
0.0128
0.0096
0.0045
0.0041
0.0016
0.0107
0.0149
0.0085
0.0094
Random
0.0163
0.0241
0.0100
0.0122
0.0108
0.0151
0.0044
0.0054
0.0099
0.0134
0.0083
0.0090
BIGRec
DEALRec
0.0181*
0.0276*
0.0115*
0.0142*
0.0124*
0.0160*
0.0055*
0.0064*
0.0117*
0.0155*
0.0096*
0.0104*
TF-DCon
0.0051
0.0074
0.0033
0.0040
0.0006
0.0057
0.0002
0.0013
0.0028
0.0051
0.0020
0.0027
RecRanker
0.0028
0.0045
0.0019
0.0024
0.0043
0.0064
0.0011
0.0014
0.0027
0.0052
0.0018
0.0025
CCS
0.0050
0.0084
0.0031
0.0041
0.0026
0.0061
0.0010
0.0013
0.0026
0.0048
0.0018
0.0024
GraNd
0.0042
0.0053
0.0027
0.0030
0.0006
0.0014
0.0003
0.0005
0.0008
0.0020
0.0006
0.0010
EL2N
0.0034
0.0048
0.0024
0.0029
0.0011
0.0016
0.0004
0.0004
0.0005
0.0015
0.0004
0.0007
Random
0.0062
0.0102
0.0039
0.0051
0.0037
0.0059
0.0011
0.0014
0.0033
0.0066
0.0022
0.0031
TIGER
DEALRec
0.0074*
0.0114*
0.0062*
0.0074*
0.0058*
0.0076*
0.0020*
0.0020*
0.0039*
0.0076*
0.0026*
0.0037*
<div style="text-align: center;">Table 3: Performance comparison between DEALRec under 1024-shot fine-tuning and the full fine-tuning of the BIGRec in terms of both accuracy and time costs. “%Improve.” denotes the relative improvement achieved by DEALRec compared to the full fine-tuning. Models are trained for 50 epochs with the early stopping strategy.</div>
Games
MicroLens-50K
Book
R@10↑
R@20↑
N@10↑
N@20↑
Time↓
R@20↑
R@50↑
N@20↑
N@50↑
Time↓
R@20↑
R@50↑
N@20↑
N@50↑
Time↓
Full
0.0169
0.0233
0.0102
0.0120
36.87h
0.0081
0.0136
0.0038
0.0053
66.64h
0.0076
0.0108
0.0060
0.0068
84.77h
DEALRec
0.0181
0.0276
0.0115
0.0142
1.67h
0.0124
0.0160
0.0055
0.0064
1.23h
0.0117
0.0155
0.0096
0.0104
1.93h
% Improve.
7.10%
18.45%
12.75%
18.33%
-95.47%
53.09%
17.65%
44.74%
20.75%
-98.15%
53.95%
43.52%
60.00%
52.94%
-97.72%
# 4.2 Overall Performance (RQ1)
The results of the baselines and DEALRec with two competitive backend LLM-based recommender models on three datasets under few-shot fine-tuning (1024 samples) are presented in Table 2, from which we have the following observations:
• All methods with BIGRec typically yield better performance than those with TIGER, which is attributed to two reasons: 1) BIGRec employs a larger LLM (i.e., LLaMA-7B) compared to TIGER, thereby benefiting from the stronger generalization ability of large-sized LLMs [27]; and 2) BIGRec leverages item titles to present the user sequence, leading to better utilization of world knowledge in LLMs. In contrast, TIGER learns extra item tokens for LLMs. This might result in cold-start item issues since only limited item tokens are learned while others are maintained randomly initialized under the few-shot fine-tuning setting. • Among all coreset selection baselines, difficulty-based (GraNd, EL2N) methods generally perform better than diversity-based methods (TF-DCon, RecRanker). This is reasonable since diversity-based methods merely heuristically encourage selecting users with divergent preference, which lacks the assessments of their contributions to the model training. In contrast, GraNd and EL2N use pre-defined metrics to measure the sample difficulty and select the samples with larger scores, which encourages selecting the samples that are more informative for models’ optimization. Besides, CCS improves EL2N in most cases, as
it maintains easy samples for selection, thus compensating the knowledge of recommendation data from high-density areas. • Another interesting observation is that random sampling yields competitive performance or even outperforms other coreset selection methods in some cases, which might attributed to two possible reasons: 1) Uniformly selected user sequences preserve high coverage of the original training distribution compared to other baselines, which ensures a high probability of guaranteed bound for low empirical risk [57]. This observation is also consistent with the findings in [13]. 2) The inferior performance of some coreset selection methods also might be caused by the implementation settings (Section 4.1.3), where they may suffer from the learning ability gap between the surrogate model and LLMs. (cf. Section 3.2). • DEALRec significantly outperforms all coreset selection methods across the three datasets. The consistent performance improvements on both backend models validate the superiority of DEALRec in identifying influential samples for LLMs’ adaptation to the recommendation data. The superior performance is attributed to: 1) the accurate and efficient estimation of the influence on empirical risk, i.e., overall performance by removing a sample in training; and 2) the gap regularization based on the effort score to penalize the easy samples for LLMs. By emphasizing the non-trivial samples specifically for LLMs, gap regularization alleviates the learning ability gap between the surrogate model and the LLMs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3aa4/3aa46608-d0b1-449d-bd1e-59478877d474.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Ablation study of the influence score, effort score, and coverage-enhanced sample selection strategy.</div>
Table 4: Performance comparison between DEALRec with different surrogate models and the BIGRec under full training. “Time” presents the time costs for training the surrogate model on a single NVIDIA RTX A5000.
R@10↑
R@20↑
N@10↑
N@20↑
Time↓
Full
0.0169
0.0233
0.0102
0.0120
/
BERT4Rec
0.0175
0.0258
0.0103
0.0128
0.76h
SASRec
0.0181
0.0276
0.0115
0.0142
0.45h
DCRec
0.0211
0.0283
0.0117
0.0137
0.61h
• Comparison with full fine-tuning. We further compare DEALRec with BIGRec under full training w.r.t. accuracy and efficiency, as presented in Table 3. We can find that: 1) DEALRec achieves higher performance compared to the model trained by the full data, indicating the effectiveness of DEALRec for high accuracy. The inferior performance of BIGRec under full training also implies that not all user sequences are informative for model training, or even harmful to the training, e.g., false negative interactions. This has also been observed in CTR prediction [48] and has been discussed in [2] from the view of data redundancy. 2) DEALRec significantly reduces the time costs for LLMs’ fine-tuning (97.11% reduction of fine-tuning costs on average). With the remarkably declined training costs, DEALRec has the potential to facilitate real-world applications of LLM-based recommender models.
# 4.3 In-depth Analysis
4.3.1 Ablation Study (RQ2). To study the effectiveness of each component of DEALRec, i.e., influence score, effort score, and coverage-enhanced sample selection strategy, we separately remove the Influence Score (IS) and effort score 𝛿𝑠, referred to as “w/o IS” and “w/o 𝛿𝑠”, respectively. Besides, we replace the coverage-enhanced sample selection strategy by greedily selecting the samples with higher scores, denoted as “Greedy”. From the results presented in Figure 4, we can observe that: removing either the influence score or effort score will cause performance drops. This validates the effectiveness of 1) the assessment of overall performance change caused by removing samples from training; 2) additional signals of learning ability captured from LLMs as regularization, alleviating the gap between the surrogate model and the LLMs. Moreover, simply selecting the samples with higher overall scores might weaken the learning of distinct user behaviors and item knowledge (inferior performance of “Greedy”), as discussed in Section 3.3.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bdfb/bdfbe5bc-cc83-455f-b3ed-c62d0fa387c8.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Performance of DEALRec with different selection ratio 𝑟w.r.t. accuracy and efficiency on Games.</div>
4.3.2 Robustness on different surrogate model (RQ2). To further assess the generalization ability of DEALRec on different surrogate models, we employ three representative sequential recommender models, i.e., BERT4Rec [41], SASRec [20], and DCRec [51] as the surrogate models, respectively. From the results in Table 4, we can find that: 1) DEALRec with the three surrogate models consistently outperforms BIGRec under full finetuning. This demonstrates the strong robustness of DEALRec on different surrogate models. 2) Different surrogate models cause some fluctuations in accuracy. This is reasonable because different model architectures express user behavior and item knowledge differently, possibly resulting in varied selected samples which will affect the performance. 3) SASRec exhibits the least time costs for training and achieves competitive performance among the three surrogate models. Therefore, SASRec could be a good choice of surrogate model for DEALRec in real-world deployments.
of selection ratio 𝑟on DEALRec on both accuracy and efficiency, we vary the ratio 𝑟from 0.2% (128-shot) to 4% (4096-shot) and present the results in Figure 5. It is observed that: 1) The recommendation accuracy rapidly improves as the number of selected samples increases from 0.2% to 1%, surpassing the full training when 𝑟= 1%. Besides, if we continuously increase the selection ratio from 2% to 4%, the benefits from additional samples gradually diminish and only minor improvements in accuracy are observed. We suspect that the gap between and the recommendation data mainly resides in a small subset of the representative user behaviors, which is what DEALRec aims to identify. 2) Meanwhile, although the time costs for fine-tuning LLMs gradually increase because of additional samples, the cost reduction compared to the full training still reaches over 94%. 3) Empirically, setting 𝑟= 1% is recommended to achieve comparable performance to full fine-tuning and low costs.
achieves superior overall performance, we test DEALRec over user sequences of different difficulties. Specifically, we calculate the loss of each user sequence via the model trained by randomly selected few-shot samples; we then divide the users into three groups according to their loss values, from the easier samples with smaller loss (Group 1) to the harder samples with larger loss (Group 3). The results of each group of DEALRec and Random on Games are presented in Figure 6. We can find that 1) the performance of both DEALRec and Random gradually declines from Group 1 to Group 3, because users with larger loss are more difficult to predict. Nevertheless, 2) DEALRec consistently outperforms Random in
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5575/55753e42-c8fb-4375-8760-e459f344388f.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Performance of DEALRec over easy to difficult samples (Group 1 to Group 3).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/66e1/66e118e1-e196-451a-8321-edc530773f5c.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Performance of DEALRec with different 𝜆.</div>
each group, which validates the effectiveness of DEALRec in considering the influence on overall performance.
4.3.5 Effect of regularization strength 𝝀. We vary 𝜆from 0.1 to 2 for DEALRec and evaluate the performance as in Figure 7. From the figures, we can find that: 1) As we incrementally increase the value of 𝜆, the overall trend of accuracy has been observed to be generally improved. This is due to the gap between the surrogate model and LLMs as discussed in Section 3.2, emphasizing the necessity to regularize the influence score to be aligned with the learning ability of the LLMs. 2) However, blindly pursuing larger lambda is not necessarily beneficial. We should carefully balance between the performance-driven influential samples from the surrogate model and the difficult samples for the LLMs.
# 5 RELATED WORK 5.1 LLM-based Recommendation
# 5 RELATED WORK
# 5.1 LLM-based Recommendation
Leveraging LLMs for recommendation has gained remarkable attention recently [36, 52], showcasing their potential across various recommendation tasks [4, 12, 27]. Some early studies explore the recommendation ability of powerful LLMs through in-context-learning ability [9, 42]. Nevertheless, the performance of LLMs is limited without extra fine-tuning over the domain-specific recommendation data [4]. To fully leverage the potential of LLMs for recommendation, a series of work studies various fine-tuning strategies tailored for recommendation tasks [12, 26, 31, 32, 53, 54]. However, fine-tuning LLMs requires extensive computational resources and time costs, thus hindering real-world applications. Therefore, it is crucial to enhance the fine-tuning efficiency of LLMbased recommender models. In this work, we propose the task of data pruning for efficient LLM-based recommendation, aiming to identify representative samples for LLMs’ few-shot fine-tuning.
# 5.2 Coreset Selection
Coreset selection has been widely studied in both traditional machine learning and deep learning [47, 50], benefiting many downstream tasks such as data-efficient learning [44], neural architecture search [40], and active learning [39]. It aims to select a small but representative subset from the full data that can lead to comparable model performance. Previous work mainly falls into two groups: 1) Heuristic methods [7, 10, 44] typically assume difficult or diverse samples are informative for model training. 2) Optimization-based methods [21, 25, 50] leverages the bi-level or discrete optimization techniques to optimize the data subset that can minimize the empirical risk. However, heuristic methods might be suboptimal since they overlook the impact of selected samples on empirical risk. And optimization-based methods fail to be applied to LLM-based recommendation due to the cumbersome calculation for complex optimization. Furthermore, previous methods usually rely on the training of the model on full data for selection, which is infeasible for LLM-based recommendation (cf. Section 2). • Data Condensation [56] is another potential solution to achieve data-efficient training. However, it is intrinsically different from our proposed task of data pruning. While it aims to synthesize a small but informative dataset [55], our task targets to identify existing samples that are representative. Besides, previous work mainly works for continuous data, which is inapplicable to LLMbased recommendation [48]. TF-DCon [49] is recently proposed for content-based recommendation and we compare it in Section 4.2.
# 6 CONCLUSION
In this work, we proposed the task of data pruning for efficient LLM-based recommendation, which aims to identify representative samples tailored for LLMs’ few-shot fine-tuning. Furthermore, we posited two objectives for this data pruning task: 1) high accuracy targets to select the samples that can lead to low empirical risk; and 2) high efficiency strives to consume low costs for the data pruning process. To this end, we proposed a novel data pruning method, namely DEALRec, to efficiently identify the influential samples with two scores. 1) The influence score is formulated to estimate the influence of sample removal on empirical risk, which is extended from the influence function and is accelerated through the symmetric property. 2) We introduced a small-sized surrogate model to calculate the influence score efficiently and proposed the effort score to bridge the gap between the surrogate model and LLMs. Empirical results validate the effectiveness of DEALRec in achieving both high efficiency and high accuracy. This work proposes a data pruning task for LLM fine-tuning, opening up a new research direction for efficient LLM-based recommendation and leaving many promising future directions for future work. 1) It is worthwhile to apply DEALRec to more LLM-based recommender models on more cross-domain datasets, improving fine-tuning performance with limited resources. 2) Due to the limited context window length of LLMs, it is promising to select the informative interacted items in users’ interaction sequences for LLMs’ fine-tuning. 3) Enhancing the inference efficiency of LLM-based recommender models is also a crucial problem for their real-world deployments.
# REFERENCES
[1] Naman Agarwal, Brian Bullins, and Elad Hazan. 2016. Second-order stochastic optimization in linear time. stat 1050 (2016), 15. [2] Sharat Agarwal, Himanshu Arora, Saket Anand, and Chetan Arora. 2020. Contextual diversity for active learning. In ECCV. Springer, 137–153. [3] Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yancheng Luo, Fuli Feng, Xiangnaan He, and Qi Tian. 2023. A bi-step grounding paradigm for large language models in recommendation systems. arXiv:2308.08434. [4] Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. Tallrec: An effective and efficient tuning framework to align large language model with recommendation. In RecSys. ACM. [5] Zalán Borsos, Mojmir Mutny, and Andreas Krause. 2020. Coresets via bilevel optimization for continual learning and streaming. NeurIPS 33 (2020), 14879– 14890. [6] Chengliang Chai, Jiayi Wang, Nan Tang, Ye Yuan, Jiabin Liu, Yuhao Deng, and Guoren Wang. 2023. Efficient coreset selection with cluster-based methods. In KDD. ACM, 167–178. [7] C Coleman, C Yeh, S Mussmann, B Mirzasoleiman, P Bailis, P Liang, J Leskovec, and M Zaharia. 2020. Selection via Proxy: Efficient Data Selection for Deep Learning. In ICLR. [8] R Dennis Cook. 1977. Detection of influential observation in linear regression. Technometrics 19, 1 (1977), 15–18. [9] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering chatgpt’s capabilities in recommender systems. In RecSys. ACM, 1126–1132. [10] Vitaly Feldman and Chiyuan Zhang. 2020. What neural networks memorize and why: Discovering the long tail via influence estimation. NeurIPS 33 (2020), 2881–2891. [11] Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang. 2023. Chat-rec: Towards interactive and explainable llms-augmented recommender system. arXiv:2303.14524. [12] Yuqi Gong, Xichen Ding, Yehui Su, Kaiming Shen, Zhongyi Liu, and Guannan Zhang. 2023. An Unified Search and Recommendation Foundation Model for Cold-Start Scenario. In CIKM. 4595–4601. [13] Chengcheng Guo, Bo Zhao, and Yanbing Bai. 2022. Deepcore: A comprehensive library for coreset selection in deep learning. In DEXA. Springer, 181–195. [14] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. In IJCAI. 1725–1731. [15] Frank R Hampel. 1974. The influence curve and its role in robust estimation. Journal of the american statistical association 69, 346 (1974), 383–393. [16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In CVPR. IEEE, 770–778. [17] Muyang He, Shuo Yang, Tiejun Huang, and Bo Zhao. 2023. Large-scale Dataset Pruning with Dynamic Uncertainty. arXiv:2306.05175. [18] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In SIGIR. 639–648. [19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv:2106.09685. [20] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In ICDM. IEEE, 197–206. [21] Krishnateja Killamsetty, Sivasubramanian Durga, Ganesh Ramakrishnan, Abir De, and Rishabh Iyer. 2021. Grad-match: Gradient matching based data subset selection for efficient deep model training. In ICML. PMLR, 5464–5474. [22] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer. 2021. Glister: Generalization based data subset selection for efficient and robust learning. In AAAI, Vol. 35. 8110–8118. [23] Krishnateja Killamsetty, Xujiang Zhao, Feng Chen, and Rishabh Iyer. 2021. Retrieve: Coreset selection for efficient and robust semi-supervised learning. NeurIPS 34 (2021), 14488–14501. [24] Pang Wei Koh and Percy Liang. 2017. Understanding black-box predictions via influence functions. In ICML. PMLR, 1885–1894. [25] Suraj Kothawade, Vishal Kaushal, Ganesh Ramakrishnan, Jeff Bilmes, and Rishabh Iyer. 2022. PRISM: A Unified Framework of Parameterized Submodular Information Measures for Targeted Data Subset Selection and Summarization. In AAAI. [26] Lei Li, Yongfeng Zhang, and Li Chen. 2023. Prompt distillation for efficient llm-based recommendation. In CIKM. 1348–1357. [27] Xinyu Lin, Wenjie Wang, Yongqi Li, Fuli Feng, See-Kiong Ng, and Tat-Seng Chua. 2023. A multi-facet paradigm to bridge large language model and recommendation. arXiv:2310.06491. [28] Robert F Ling. 1984. Residuals and influence in regression. [29] Qijiong Liu, Nuo Chen, Tetsuya Sakai, and Xiao-Ming Wu. 2024. ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models. In WSDM. ACM. [30] Sichun Luo, Bowei He, Haohan Zhao, Yinya Huang, Aojun Zhou, Zongpeng Li, Yuanzhang Xiao, Mingjie Zhan, and Linqi Song. 2023. RecRanker:
Instruction Tuning Large Language Model as Ranker for Top-k Recommendation. arXiv:2312.16018. [31] Zheqi Lv, Wenqiao Zhang, Zhengyu Chen, Shengyu Zhang, and Kun Kuang. 2024. Intelligent Model Update Strategy for Sequential Recommendation. In WWW. ACM. [32] Zheqi Lv, Wenqiao Zhang, Shengyu Zhang, Kun Kuang, Feng Wang, Yongwei Wang, Zhengyu Chen, Tao Shen, Hongxia Yang, Beng Chin Ooi, et al. 2023. DUET: A Tuning-Free Device-Cloud Collaborative Parameters Generation Framework for Efficient Device Model Generalization. In WWW. ACM, 3077–3085. [33] Yongxin Ni, Yu Cheng, Xiangyan Liu, Junchen Fu, Youhua Li, Xiangnan He, Yongfeng Zhang, and Fajie Yuan. 2023. A Content-Driven Micro-Video Recommendation Dataset at Scale. arXiv:2309.15379 (2023). [34] Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. 2021. Deep learning on a data diet: Finding important examples early in training. NeurIPS 34, 20596–20607. [35] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q Tran, Jonah Samost, et al. 2023. Recommender Systems with Generative Retrieval. In NeurIPS. Curran Associates, Inc. [36] Xubin Ren, Wei Wei, Lianghao Xia, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2024. Representation learning with large language models for recommendation. In WWW. ACM. [37] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian personalized ranking from implicit feedback. In UAI. AUAI Press, 452–461. [38] Noveen Sachdeva, Mehak Dhaliwal, Carole-Jean Wu, and Julian McAuley. 2022. Infinite recommendation networks: a data-centric approach. NeurIPS 35, 31292– 31305. [39] Ozan Sener and Silvio Savarese. 2018. Active learning for convolutional neural networks: A core-set approach. (2018). [40] Jae-hun Shim, Kyeongbo Kong, and Suk-Ju Kang. 2021. Core-set sampling for efficient neural architecture search. arXiv:2107.06869. [41] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In CIKM. 1441–1450. [42] Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie Ren, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent. In EMNLP. ACL, 14918–14937. [43] Haoru Tan, Sitong Wu, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, and Xiaojuan Qi. 2023. Data Pruning via Moving-one-Sample-out. arXiv:2310.14664. [44] Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J Gordon. 2018. An empirical study of example forgetting during deep neural network learning. arXiv:1812.05159. [45] Wenjie Wang, Xinyu Lin, Liuhui Wang, Fuli Feng, Yunshan Ma, and Tat-Seng Chua. 2023. Causal Disentangled Recommendation Against User Preference Shifts. TOIS (2023). [46] Wenjie Wang, Xinyu Lin, Liuhui Wang, Fuli Feng, Yinwei Wei, and Tat-Seng Chua. 2023. Equivariant Learning for Out-of-Distribution Cold-start Recommendation. In MM. 903–914. [47] Kai Wei, Rishabh Iyer, and Jeff Bilmes. 2015. Submodularity in data subset selection and active learning. In ICML. PMLR, 1954–1963. [48] Jiahao Wu, Wenqi Fan, Shengcai Liu, Qijiong Liu, Rui He, Qing Li, and Ke Tang. 2023. Dataset condensation for recommendation. arXiv:2310.01038. [49] Jiahao Wu, Qijiong Liu, Hengchang Hu, Wenqi Fan, Shengcai Liu, Qing Li, Xiao-Ming Wu, and Ke Tang. 2023. Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation. arXiv:2310.09874. [50] Shuo Yang, Zeke Xie, Hanyu Peng, Min Xu, Mingming Sun, and Ping Li. 2023. Dataset pruning: reducing training data by examining generalization influence. (2023). [51] Yuhao Yang, Chao Huang, Lianghao Xia, Chunzhen Huang, Da Luo, and Kangyi Lin. 2023. Debiased Contrastive Learning for Sequential Recommendation. In WWW. 1063–1073. [52] Honglei Zhang, He Liu, Haoxuan Li, and Yidong Li. 2024. TransFR: Transferable Federated Recommendation with Pre-trained Language Models. arXiv:2402.01124. [53] Honglei Zhang, Fangyuan Luo, Jun Wu, Xiangnan He, and Yidong Li. 2023. LightFR: Lightweight federated recommendation with privacy-preserving matrix factorization. TOIS 41, 4 (2023), 1–28. [54] Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. 2023. Recommendation as instruction following: A large language model empowered recommendation approach. arXiv:2305.07001. [55] Bo Zhao and Hakan Bilen. 2023. Dataset condensation with distribution matching. In WACV. IEEE, 6514–6523. [56] Bo Zhao, Konda Reddy Mopuri, and Hakan Bilen. 2020. Dataset Condensation with Gradient Matching. In ICLR. [57] Haizhong Zheng, Rui Liu, Fan Lai, and Atul Prakash. 2022. Coverage-centric Coreset Selection for High Pruning Rates. In ICLR.
