# Large Language Model Enhanced Hard Sample Identification for Denoising Recommendation
Tianrui Song1, Wenshuo Chao1, Hao Liu1*
1The Hong Kong University of Science and Technology (Guangzhou) tsong847@connect.hkust-gz.edu.cn, wchao@connect.ust.hk, liuh@ust.hk
Abstract
Implicit feedback, often used to build recommender systems, unavoidably confronts noise due to factors such as misclicks and position bias. Previous studies have attempted to alleviate this by identifying noisy samples based on their diverged patterns, such as higher loss values, and mitigating the noise through sample dropping or reweighting. Despite the progress, we observe existing approaches struggle to distinguish hard samples and noise samples, as they often exhibit similar patterns, thereby limiting their effectiveness in denoising recommendations. To address this challenge, we propose a Large Language Model Enhanced Hard Sample Denoising (LLMHD) framework. Specifically, we construct an LLM-based scorer to evaluate the semantic consistency of items with the user preference, which is quantified based on summarized historical user interactions. The resulting scores are used to assess the hardness of samples for the pointwise or pairwise training objectives. To ensure efficiency, we introduce a variance-based sample pruning strategy to filter potential hard samples before scoring. Besides, we propose an iterative preference update module designed to continuously refine summarized user preference, which may be biased due to false-positive user-item interactions. Extensive experiments on three real-world datasets and four backbone recommenders demonstrate the effectiveness of our approach.
arXiv:2409.10343v1
# Introduction
Recommender systems are designed to learn user preferences and suggest items across various online platforms, such as e-commerce, news portals, and social networks (2020; 2020; 2023). To train these systems, implicit feedback derived from user actions (e.g., clicks and purchases) is commonly employed due to its wide availability. Typically, each observed interaction is assumed to reflect a user’s genuine interest in an item and is therefore assigned a positive label, while non-interacted items are considered negative (2020; 2021a). However, such a routine has recently been questioned that interacted items may be plagued by falsepositive noise (e.g., due to misclicks or popularity bias), while non-interacted items may suffer from false-negative noise (e.g., due to position bias) (2021b). These noisy interactions lead to inaccurate estimation of user preferences, hindering the performance of recommendation systems.
*Corresponding author.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b3f0/b3f007c9-828d-4702-9aa6-8e662e3ab201.png" style="width: 50%;"></div>
Figure 1: Loss values and prediction scores during training LightGCN on Yelp dataset. We observe that hard and noisy samples exhibit similar values in prediction score and loss, making it difficult to differentiate them.
Denoising recommendation has been proposed to mitigate the negative impact of noisy interactions through two primary strategies: 1) sample dropping and 2) sample reweighting. Dropping methods aim to improve model performance by selecting clean samples and discarding noisy ones during training (2021a; 2021). In contrast, re-weighting approaches assign lower weights to interactions identified as noisy, thereby reducing their influence on the model’s learning process (2023; 2022). The success of these denoising techniques heavily depends on the accuracy of distinguishing between clean and noisy samples. Consequently, various data patterns have been explored as noisy signals (2021a; 2022; 2023). To name a few, loss value is one of the most commonly used signals, as noisy interactions typically exhibit higher loss values compared to clean ones (2021a; 2024). In addition, other indicators such as prediction scores (Wang et al. 2022) and gradients (Wang et al. 2023) have also been investigated to identify noisy samples. Despite significant advancements, existing methods often face the challenge of misidentifying hard samples as noisy ones. As illustrated in Figure 1, while noisy samples exhibit distinct patterns compared to easy samples, we observed that hard samples and noisy samples tend to present similar patterns in both prediction scores and loss values. Con-
sequently, previous denoising approaches that rely solely on data patterns struggle to accurately distinguish between hard and noisy samples. This misclassification is problematic because hard samples have been shown to be beneficial, both empirically (2012) and theoretically (2023). Mistakenly treating hard samples as noise during the recommender training ultimately leads to suboptimal results. Recently, Large Language Models (LLMs) have demonstrated a promising ability to understand user preferences (Wu et al. 2024) and enhance item semantics (Wei et al. 2024), presenting a valuable opportunity to tackle the challenge of hard sample identification. Our key insight is that LLMs can be harnessed to summarize user preference and act as a scorer to analyze the consistency between user preferences and items, thereby identifying hard samples with the resulting scores. For example, when optimizing a model using a Bayesian Personalized Ranking (BPR) objective, the LLM scorer can effectively evaluate user preference scores of positive and negative items. As a result, samples with similar positive and negative scores are pinpointed as hard samples because they are inherently incompatible with the BPR training objective, which aims to maximize the divergence in scores. This allows us to mitigate the hard samples’ misclassification issue in denoising recommender training. However, leveraging LLMs for this task is nontrivial due to two primary challenges. First, given the vast number of users and items, assessing the preferences of all users across all items is computationally intensive, especially considering the high inference cost of LLMs. Second, while LLMs can derive user preference by concluding interacted items, the presence of false-positive items in historical interactions can lead to biased user preference summarization. To address the challenges mentioned above, we propose a Large Language Model Enhanced Hard Sample Denoising (LLMHD) framework for recommendation, which comprises three key modules: Variance-based Sample Pruning, LLM-based Sample Scoring, and Iterative Preference Updating. To ensure efficiency, we first introduce a variancebased pruning strategy that progressively selects a small subset of hard sample candidates. Following this, we construct the LLM-based Sample Scoring module, where hard samples are identified by evaluating how well they satisfy the training objective. Specifically, the LLM scores the user preference for a given item by summarizing user preference, assesses the sample’s hardness based on the pointwise or pairwise training objective, and determines whether it qualifies as a hard sample. Additionally, to enhance the accuracy of the summarized user preferences, we propose an Iterative Preference Updating module. It refines user preferences by adjusting for items that are mistakenly identified or overlooked during the summarization process, thereby improving the overall reliability of the LLMHD framework. Our main contributions are summarized as follows:
• We propose LLMHD , a novel denoising recommendation approach that differentiates between hard and noisy samples leveraging LLMs. To the best of our knowledge, this is the first attempt to integrate LLMs into denoising recommendations.
• The LLMHD addresses efficiency concerns through a variance-based sample pruning process. Furthermore, we enhance the effectiveness of the model by employing an iterative preference updating strategy, improving the LLMs’ understanding of genuine user preferences. • Extensive experiments conducted on three real-world datasets and four backbone recommenders demonstrate the effectiveness of our method. The results show that LLMHD delivers impressive performance and robust noise resilience.
# Related Work
# Related Work Denoise Recommendation
# Denoise Recommendation
Recommenders are pointed out to be affected by users’ unconscious behaviors (2021b), leading to noisy data. As a result, many efforts are dedicated to alleviating the problem. These approaches can be categorized into two paradigms: sample dropping (2012; 2023) and sample re-weighting (2023; 2022). Sample dropping methods aim to keep clean samples and discard noisy ones. For instance, T-CE (2021a) observes that noisy samples exhibit high loss values and remove them during training. IR (2021c) iteratively generates pseudo-labels to discover noisy examples. Sample reweighting methods try to mitigate the impact of noisy samples by assigning lower weights to them. Typically, R-CE (2021a) assigns lower weights to noisy samples according to the prediction score. BOD (2023) considers the weight assignment as a bi-level optimization problem. Although these methods achieve promising results, they rely on data patterns to recognize noisy samples (e.g., loss values, and prediction scores) leading to difficulties in identifying hard samples from noise samples as they exhibit similar patterns.
# LLMs for Recommendation
Large Language Models (LLMs) are effective tools for the Natural Language Processing field and have gained significant attention in the domain of Recommendation Systems (RS). For the adaption of LLMs in recommendations, existing works can be divided into three categories (2024): LLM as RS, LLM Embedding for RS, and LLM token for RS. The LLM as RS aims to transform LLMs into effective recommendation systems (Chao et al. 2024), such as LC-Rec (2024a) and LLM-TRSR (2024b). In contrast, the LLM embedding and LLM token for RS views the language model as an enhancer, where embeddings and tokens generated by LLMs are utilized for promoting recommender systems. The former typically adopts embeddings related to users and items, incorporating semantic information in the recommender (Ren et al. 2024). While the latter generates text tokens to capture potential preferences through user and item semantics (Wei et al. 2024; Xi et al. 2023). Despite the progress, these methods overlook the potential of LLMs in enhancing data denoising for recommendation.
# Preliminary
The objective of training a recommender system is to learn a scoring function ˆyu,i = fθ(u, i) from interactions between users u ∈U and items i ∈I. We assume that user-interacted
items y∗ ui = 1 are preferred by the user, while those not interacted y∗ ui = 0 are not. To optimize the scoring function fθ(u, i), We employ Bayesian Personalized Ranking (BPR) loss and Binary Cross-Entropy (BCE) loss as loss function Lrec. These are formulated as:
(1)
where j denotes sampled negative items according to the pairwise sampling distribution PD∗, and D∗= {(u, i, y∗ ui) | u ∈U, i ∈I} represents the interaction dataset. The optimal parameter set θ∗is obtained by minimizing the loss function:
(3)
But this assumption is unreliable for two reasons: (1) False positive issue, user-interacted items might not reflect real user preference due to factors such as accidental clicks and position bias. (2) False negative issue, non-interacted items are not necessarily user dislikes, they may have been overlooked due to factors such as suboptimal display positions. These issues introduce noisy interactions, formally defined as ˜D = {(u, i, ˜y) | ˜y ̸= y∗}. To address this, we formulate the denoising recommender training task as:
(4)
aiming to learn high-quality recommender with parameters θ∗by eliminating the effect of noisy samples. In this work, we focus on the challenge of hard samples, which are often mistakenly identified as noisy samples in existing denoising approaches, leading to suboptimal performance.
# Proposed Method
To differentiate hard and noisy samples when denoising, we proposed the LLMHD framework, as illustrated in Figure 2. Before diving into the details of each module, we assume that each item i is accompanied by a text profile Pi. Additionally, we summarize the user’s preference Pu = LLMs(Tsum({Pi | yu,i = 1})) by the profiles of interacted items with a prompt template Tsum designed for LLMs. Our LLMHD identifies hard samples through three key modules: (1) Variance-based Sample Pruning, (2) LLMbased Sample Scoring, and (3) Iterative Preference Updating. Variance-based Sample Pruning reduces the computation of calling LLMs by selecting a subset of hard sample candidates. LLM-based Sample Scoring evaluates the hardness of samples based on user preferences. Iterative Preference Updating refines the understanding of user preference, ensuring accurate identification of hard samples.
# Loss-based Denoising
We first introduce the denoising module implemented based on the widely accepted assumption (2021a) that samples
with higher loss values are more likely to be noisy. Specifically, for each data sample b in the mini-batch B, we calculate the corresponding loss value l(b) and sort all samples in the ascending order,
(5)
where |B| denotes the batch size. This operation assists the noisy sample identification, which we formulate as BN,
(6)
� � where T denotes the current training iteration. The εl represents a dynamic threshold, calculated as,
(7)
where εmax l is a hyper-parameter representing the maximum noise ratio, and α is a factor that modulates the growth rate of the noise threshold. The εl increases as the stability of prediction scores incrementally improves during training, following previous works (Wang et al. 2021a). It is worth mentioning that the BN inadvertently contain hard samples, given that both hard and noisy samples manifest similar patterns in loss values. This requires further refinement to distinguish genuine noisy data and hard samples.
# Variance-based Sample Pruning
Although it is possible to present all identified noisy samples BN to the LLMs for scoring, this approach would be prohibitively time-consuming due to the massive interactions in the recommender system. Specifically, hard sample candidates are selected based on the observation of previous work (2020), which demonstrated that hard samples exhibit relatively higher prediction score variance compared to noisy samples. Therefore, for samples b ∈BN, we calculate the prediction scores variance of positive vp,b and negative vn,b items across multiple epochs (see Equation 17). Then sort them in descending order based on vp and vn respectively,
(9)
where |Bp N| and |Bn N| denotes the number of positive and negative items in the BN respectively. Hard sample candidates BHC are collected by,
 (10)
where εv ∈[0, 1] denotes the proportion of hard samples. With the increasing |BN| more candidates will be selected in latter training iterations and provided to LLM-based Sample Scoring to identify hard samples further.
# LLM-based Sample Scoring
Owing to the resemblance in data patterns between hard and noisy samples, distinguishing them solely through numerical disparities is ineffective. To eliminate this issue, we introduce the LLM-based Sample Scoring method. LLMs act as scorer to provide auxiliary information that evaluates the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b28d/b28d7784-b55a-4755-8e0b-3d3f4a36d4f6.png" style="width: 50%;"></div>
Figure 2: The overview of the LLMHD framework. LLMHD leverages LLMs to differentiate hard and noisy samples, thereby enhancing the denoising recommender training task. The framework identifies hard samples through three main modules: (1) Variance-based Sample Pruning, (2) LLM-based Sample Scoring, and (3) Iterative Preference Updating.
sample’s hardness. Formally, we prompt LLMs to score the user preference for item su,i with a template Tscore(Pu, Pi) that wraps the user preference text Pu and item profile Pi,
(11)
The resulting score su,i is adopted to specify the sample’s hardness by analyzing its compatibility with the training objective. Lower compatibility samples are considered harder as they are more challenging to satisfy the objective. Given that most recommenders are trained to minimize pointwise (e.g., BCE) or pairwise (e.g., BPR) losses, we devise two paradigms for hard sample identification: (1) Pointwise Preference Scoring, and (2) Pairwise Preference Scoring. Pointwise Sample Scoring The pointwise BCE loss, as shown in Equation 2, aims at reducing the classification uncertainty of a (user, item) pair. For a data sample (u, ipos) or (u, ineg), if the user’s preference for the item is ambiguous, the sample is of low compatibility with the training objective. Therefore, positive pair with lower su,ipos and negative pair with higher su,ineg are harder samples, thereby hard samples are identified by,
(12)
  where the εpos and εneg are thresholds that control the hardness. In addition, since previous works (2021a) discussed that fitting harder samples at the early training stage might hurt the generalization ability, we smoothly change εpos and εneg during each training iteration T as follows,
(14)
where εmax pos , εmin pos , εmax neg , εmin neg are hyper-parameters. In this way, harder positive (u, ipos) and negative (u, ineg) samples will be identified in the latter iterations, benefiting the recommender by gradually increasing the hardness. Pairwise Sample Scoring Similar to the pointwise sample scoring, we identify hard samples under the pairwise training schema. Specifically, according to Equation 1, the pairwise BPR loss aims to maximize the divergence of prediction scores between positive and negative items. For a sample (u, ip, in), if the user’s preference for the positive item does not significantly surpass that for the negative, the sample is less compatible with the objective. Therefore, hard samples are identified through the indicator function,
(15)
  where the threshold εpair also gradually decreases to increase the hardness by the number of iteration T,
(16)
Based on the above technique, we differentiate hard samples in both pointwise and pairwise training schema.
# Iterative Preference Updating
Accurate user preference Pu is critical for effective LLM sample scoring. However, the Pu summarized based on interacted items do not fully capture user interests due to the inclusion of disliked items, i.e., false-positives, and the exclusion of liked items, i.e., false-negatives. To mitigate this problem, we refine user preferences iteratively by excluding dislikes and incorporating likes. For every epoch t, we calculate the variance score vd of user-item pairs d = (u, i),
(17)
where ˆyj d is the prediction score of user-item pair d in the j-th training epoch, and the variance vd is calculated over m time intervals prior to the t-th training iteration. We divided variance scores into two groups, positive and negative samples, and ordered from lowest to highest,
(18)
(19)
where dp k, dn k are the k-th positive and negative sample respectively. To identify whether a sample is a false positive or false negative in the j-th epoch, we use the indicators Ij fp(dp k ≤dp εl) and Ij fn(dn k ≥dn εl) respectively. The threshold εl employed here follows the same definition as introduced in Equation 7. We design a robust mechanism to select confident items for preference updates. Formalized as follows,
(20)
the εγ is a confidence threshold. We then leverage LLMs to refine preference Pu based on identified false-positives (u, ifp) and false negatives (u, ifp) with the template TFP(Pu, Pifp) and TFN(Pu, Pifn),
(21) (22)
P P P where P∗ u is the updated user preference text description. The template TFP intend to add descriptioins about ifp in the user preference Pu, while the TFN reduce the feature of ifn.
# Denoising Training with Hard Samples
The denoising training is done by keeping hard samples and dropping noisy samples. We first define the set of identified hard samples BH as,
(23)
   where the ILLM is either Ipoint or Ipair based on the format of data samples. The recommendation loss Lrec is then calculated in the following format,
(24)
In this way, hard samples have remained and the noisy samples are dropped while training the recommender.
# Experiments
We compare LLMHD with state-of-the-art denoise approaches on four backbones and three real-world datasets to demonstrate the effectiveness of our method. Experiments are directed by the following research questions (RQs): • RQ1: How does LLMHD performs compared with other state-of-the-art denoise baselines across the datasets? • RQ2: Does the LLMHD demonstrate robustness when tackling different levels of noisy data? • RQ3: What is the effect of different components and hyper-parameters within the LLMHD on performance?
We compare LLMHD with state-of-the-art denoise approaches on four backbones and three real-world datasets to demonstrate the effectiveness of our method. Experiments are directed by the following research questions (RQs):
• RQ1: How does LLMHD performs compared with other state-of-the-art denoise baselines across the datasets? • RQ2: Does the LLMHD demonstrate robustness when tackling different levels of noisy data? • RQ3: What is the effect of different components and hyper-parameters within the LLMHD on performance?
# Experiment Settings
Experiment Settings Datasets. We conduct evaluations of our LLMHD on three public datasets: (1) Amazon-Books collected from the Amazon platform. We conduct experiments on the book subcategories. (2) Yelp is a large-scale dataset that provides check-in history. (3) Steam consists of users and electronic games on the Steam platform. Since we adopt the item profile provided in (Ren et al. 2024), we process datasets following their settings. Details are in the Appendix. Evaluation Metrics. Following existing works on denoising recommendation (2021c; 2024), we report the results w.r.t. two widely used metrics: NDCG@K and Recall@K, where higher scores indicate better performance. For a comprehensive comparison, we set the K as 5 and 10 for both metrics on all three datasets. Baselines. To evaluate the performance of LLMHD, we apply it to the following backbone recommenders: • NGCF (2019) models the user-item interaction graph with GNN for collaborative filtering. • LightGCN (2020) removes the feature transformation and non-linear activation in NGCF. • SGL (2021) generates positive views of with model-level node and edge dropout for self-supervised learning. • NCL (2022) exploit the neighborhood structure to conduct self-supervised learning in graph recommenders. To compare the denoising effect, each recommender is trained with the following approaches: • BCE represents the model is trained with the base pointwise binary cross-entropy loss. • BPR (2009) represents the model is trained with the pairwise Bayesian Personalized Ranking loss. • T-CE (2021a) removes the samples with higher loss through dynamic threshold. • R-CE (2021a) is guided by a similar assumption as TCE, while allocating lower weights to noise samples. • RGCF (2022) discard noisy edges according to the structure representation cosine similarity and enhance the diversity with graph self-supervised learning. • DCF (2024) gradually relabel noisy samples to address the scarcity issue when denoising.
Implementation Details. All methods are trained with a batch size of 1024 and a learning rate of 0.005 with Adam optimizer for up to 200 epochs. We adopt the early stopping during training. We adopt the RecBole implementation for all backbone models. Hyper-parameters are selected based on the origin setting. We did a grid search on the following hyper-parameters to find the optimal result for LLMHD, including α and εmax l , which are explored within {3k, 5k, 10k, 30k, 50k} and {0.01, 0.03, 0.05, 0.1, 0.2} respectively. The hard sample proportion εv is selected from {0.1, 0.3, · · · , 0.9}. All thresholds are fixed as follows, εmax pos =8, εmin pos = 6, εmax neg = 4 and εmin neg = 2 for pointwise hard sample identification. The pairwise εmax pair and εmin pair are fixed as 7 and 3 respectively. The confidence threshold εγ to update user preference is explored within {3,5,7,9}.
Dataset
Amazon-book
Yelp
Steam
Backbone
Method
R@5
R@10
N@5
N@10
R@5
R@10
N@5
N@10
R@5
R@10
N@5
N@10
NGCF
BCE
0.0353
0.0570
0.0365
0.0438
0.0236
0.0431
0.0283
0.0350
0.0223
0.0405
0.0236
0.0305
BPR
0.0389
0.0651
0.0406
0.0494
0.0280
0.0495
0.0338
0.0405
0.0381
0.0629
0.0453
0.0525
T-CE
0.0393
0.0650
0.0402
0.0489
0.0259
0.0450
0.0313
0.0373
0.0257
0.0448
0.0288
0.0354
R-CE
0.0366
0.0587
0.0369
0.0444
0.0254
0.0438
0.0302
0.0360
0.0236
0.0435
0.0254
0.0328
RGCF
0.0415
0.0658
0.0422
0.0502
0.0287
0.0485
0.0344
0.0406
0.0401
0.0644
0.0472
0.0543
DCF
0.0398
0.0617
0.0399
0.0472
0.0281
0.0488
0.0353
0.0414
0.0264
0.0446
0.0308
0.0365
LLMHDBCE
0.0406
0.0668
0.0413
0.0503
0.0276
0.0477
0.0329
0.0386
0.0267
0.0459
0.0297
0.0364
LLMHDBPR
0.0455
0.0743
0.0455
0.0552
0.0338
0.0579
0.0398
0.0474
0.0418
0.0696
0.0496
0.0579
LightGCN
BCE
0.0558
0.0849
0.0565
0.0665
0.0390
0.0660
0.0481
0.0557
0.0448
0.0732
0.0529
0.0612
BPR
0.0587
0.0904
0.0598
0.0704
0.0359
0.0609
0.0446
0.0516
0.0510
0.0828
0.0597
0.0693
T-CE
0.0590
0.0895
0.0592
0.0697
0.0401
0.0677
0.0504
0.0580
0.0463
0.0758
0.0555
0.0640
R-CE
0.0557
0.0834
0.0566
0.0658
0.0389
0.0650
0.0474
0.0550
0.0461
0.0757
0.0543
0.0630
RGCF
0.0619
0.0956
0.0644
0.0753
0.0420
0.0693
0.0501
0.0579
0.0519
0.0849
0.0599
0.0702
DCF
0.0590
0.0898
0.0596
0.0701
0.0403
0.0680
0.0503
0.0579
0.0477
0.0778
0.0562
0.0650
LLMHDBCE
0.0607
0.0921
0.0607
0.0711
0.0408
0.0689
0.0514
0.0589
0.0469
0.0767
0.0563
0.0647
LLMHDBPR
0.0652
0.0999
0.0655
0.0767
0.0427
0.0731
0.0518
0.0611
0.0536
0.0867
0.0624
0.0722
SGL
BCE
0.0589
0.0902
0.0604
0;.0707
0.0377
0.0655
0.0470
0.0548
0.0433
0.0682
0.0505
0.0676
BPR
0.0608
0.0956
0.0621
0.0736
0.0373
0.0629
0.0465
0.0538
0.0529
0.0838
0.0613
0.0704
T-CE
0.0602
0.0909
0.0622
0.0720
0.0408
0.0697
0.0502
0.0587
0.0449
0.0720
0.0532
0.0609
R-CE
0.0591
0.0901
0.0601
0.0702
0.0386
0.0645
0.0476
0.0550
0.0456
0.0732
0.0538
0.0618
RGCF
0.0675
0.1049
0.0681
0.0808
0.0416
0.0715
0.0512
0.0606
0.0552
0.0881
0.0639
0.0736
DCF
0.0626
0.0933
0.0641
0.0740
0.0413
0.0683
0.0506
0.0583
0.0455
0.0727
0.0536
0.0615
LLMHDBCE
0.0615
0.0931
0.0640
0.0739
0.0414
0.0708
0.0509
0.0596
0.0462
0.0742
0.0543
0.0619
LLMHDBPR
0.0693
0.1051
0.0717
0.0837
0.0426
0.0718
0.0523
0.0619
0.0546
0.0887
0.0641
0.0739
NCL
BCE
0.0574
0.0871
0.0598
0.0694
0.0391
0.0647
0.0477
0.0548
0.0450
0.0731
0.0529
0.0612
BPR
0.0605
0.0942
0.0628
0.0740
0.0369
0.0609
0.0451
0.0515
0.0511
0.0835
0.0602
0.0698
T-CE
0.0599
0.0898
0.0619
0.0719
0.0411
0.0679
0.0507
0.0582
0.0461
0.0751
0.0543
0.0627
R-CE
0.0585
0.0874
0.0604
0.0696
0.0399
0.0655
0.0487
0.0558
0.0459
0.0750
0.0540
0.0625
RGCF
0.0694
0.1045
0.0706
0.0819
0.0396
0.0660
0.0480
0.0560
0.0534
0.0863
0.0621
0.0718
DCF
0.0619
0.0929
0.0624
0.0727
0.0424
0.0696
0.0513
0.0589
0.0465
0.0759
0.0550
0.0635
LLMHDBCE
0.0609
0.0915
0.0629
0.0730
0.0417
0.0690
0.0517
0.0591
0.0468
0.0762
0.0551
0.0633
LLMHDBPR
0.0719
0.1053
0.0741
0.0846
0.0432
0.0716
0.0542
0.0620
0.0540
0.0861
0.0624
0.0717
Table 1: Performance comparison of backbone recommenders trained with different denoising approaches. R and N refer to Recall and NDCG, respectively. The highest scores are in bold, and the runner-ups are with underline. All results are statistically significant according to the t-tests with a significance level of p < 0.01.
Table 1: Performance comparison of backbone recommenders trained with different denoising approaches. R and N refer to Recall and NDCG, respectively. The highest scores are in bold, and the runner-ups are with underline. All results are statistically significant according to the t-tests with a significance level of p < 0.01.
# Performance Comparison (RQ1)
We evaluated the effectiveness of our proposed method in both pointwise LLMHDBCE and pairwise LLMHDBPR setting. The performance against other denoising recommendation strategies is shown in Table 1. Notably, LLMHD significantly enhances the performance of all backbone models trained with normal BCE or BPR on all datasets, demonstrating its superior denoising capability. Moreover, LLMHD consistently outperforms other advanced denoising methods across the majority of datasets and backbones. We attribute this improvement to the extended hard sample identification, where baselines like T-CE and RGCF lack the capability, rendering them less effective in comparison. We also observed that for most datasets and backbones, RGCF and DCF are inferior to us alone. In contrast, other denoise baselines like T-CE and R-CE perform worse than them. This can be explained by the fact that both RGCF and DCF are designed to insert or preserve high-confidence interactions, a feature not inherent to T-CE and R-CE. Since
LLMHD also focuses on retaining more interactions (i.e., hard samples), we posit that maintaining a more extensive set of samples is beneficial in enhancing performance.
# Noise Robustness (RQ2)
We conduct random noise training to evaluate the robustness of the noise resistance capability of LLMHD, comparing it with the most competitive RGCF and the classical approach T-CE. Following previous work (2021), the proportion of noise injected into the training set spanned from 5% to 20%, while keeping the samples in the testing set unchanged. We report the result in Figure 3. The result shows that: (1) As the noise ratio increases, we observe a consistent downward trend in the performance across all backbone models and denoising strategies. This decline can be caused by the rising noise level leading to the increasing data corruption, which complicates discerning genuine user preferences. (2) LLMHD outperforms the backbone model and other denoise approaches in all noise ratios. This empha-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d397/d397777e-fae0-4abd-b5b8-f7fb241df1e1.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Performance comparison of denoise training with random noises in Amazon-books.</div>
Methods
Amazon-books
R@5
R@10
N@5
N@10
LightGCN
0.0587
0.0904
0.0598
0.0704
+ LLMHDLD
0.0614
0.0965
0.0628
0.0743
+ LLMHDLD+RS+LMS
0.0623
0.0970
0.0635
0.0749
+ LLMHDLD+VS+LMS
0.0638
0.0986
0.0655
0.0767
+ LLMHDLD+VS+LMS+PU
0.0652
0.0999
0.0665
0.0771
Table 2: The effect of each components in LLMHDBPR with the LightGCN on Amazon-books dataset.
sizes LLMHD’s promising noise resistance, attributed to the correctly identified hard and noisy samples. (3) we also observed that RGCF shows sub-optimal results and slower performance degradation with increasing noise. This is probably because it employs random edge augmentation, which makes the model adapt to Gaussian noise. However, when confronted with real-world data noise, i.e., when additional noise is smaller, its performance falls short of LLMHD.
# In-depth Model Analysis (RQ3)
Ablation Study. We conduct experiments to assess each module in LLMHD, including Variance-based Sample Pruning, LLM-based Sample Scoring, and Iterative Preference Updating, the result is shown in Table 2. (1) We investigate whether including Variance-based Sample Pruning (VS) enables an effective hard sample candidate selection. Specifically, we compare it with Random Selection (RS) and select the same amount of candidates as the VS. According to the result, converting the VS to RS leads to a performance drop in all metrics. This reveals the superiority of the Variancebased Sample Pruning in selecting hard sample candidates. (2) We discover whether LLM-identified hard samples enhance the recommendation performance. The comparison is made between the backbone that only adopts a Loss-based Denoise Module (LD) and the one that includes LLM-based Sample Scoring (LMS). Significant improvement is demonstrated after using LLMs to detect hard samples, indicating the advancement of LLM in hard sample identification. (3) We explore the influence of adopting Iterative Preference Updating (PU). Compared with discarding the preference updating, the performance of adopting it increases, demon-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a101/a101db6b-6b0d-4b43-9437-7094365742f6.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Hyper-parameter analysis in LLMHDBPR with LightGCN backbone on the Amazon-books.</div>
strating the effectiveness of Preference Updating in understanding genuine user preference.
Effect of Hyper-parameters. For a more elaborate analysis, we adjust the hyper-parameters within the range described in the Implementation Details section. The results are shown in Figure 4. From our observations: (1) Growth Rate α: A moderate increase in α enhances performance, as it retains more samples per iteration, mitigating data scarcity during training. However, excessively high values degrade performance. (2) Max Noise Scale εmax l : Elevating εmax l initially improves LLMHD by filtering out more noise, but an overly high setting results in excessive sample loss, hampering the learning of user preferences. (3) Hard Sample Candidate Proportion εv: Increasing εv presents more hard sample candidates, boosting performance. But setting it too high may confuse noisy samples for hard ones, lowering overall effectiveness. (4) Confidence Threshold εγ: Gradually raising εγ initially benefits the model by promoting item selection for preference update. However, a high confidence restricts item discovery and a low confidence finds incorrect items, both diminishing user preference understanding.
# Conclusion
In this work, we introduced the Large Language Model Enhanced Hard Sample Denoising (LLMHD) framework to address the challenge of distinguishing hard samples from noise samples for recommender systems. By utilizing an LLM-based scorer to evaluate semantic consistency between users and items and assessing sample hardness according to its compatibility with training objectives, we can differentiate hard samples from noise samples. We further introduce a variance-based sample pruning strategy to effectively select candidates. In addition, the iterative preference update refines user preference and mitigates biases introduced by false-positive interactions. Extensive experiments on realworld datasets and recommenders demonstrated the effectiveness of LLMHD in improving recommendation quality.
# References
Chao, W.; Zheng, Z.; Zhu, H.; and Liu, H. 2024. Make Large Language Model a Better Ranker. arXiv:2403.19181. Chen, C.; Zheng, S.; Chen, X.; Dong, E.; Liu, X.; Liu, H.; and Dou, D. 2021. Generalized Data Weighting via Classlevel Gradient Manipulation. In Neural Information Processing Systems. Ding, J.; Quan, Y.; Yao, Q.; Li, Y.; and Jin, D. 2020. Simplify and robustify negative sampling for implicit collaborative filtering. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20. Red Hook, NY, USA: Curran Associates Inc. ISBN 9781713829546. Gantner, Z.; Drumond, L.; Freudenthaler, C.; and SchmidtThieme, L. 2012. Personalized ranking for non-uniformly sampled items. In Proceedings of KDD Cup 2011, 231–247. PMLR. Gao, Y.; Du, Y.; Hu, Y.; Chen, L.; Zhu, X.; Fang, Z.; and Zheng, B. 2022. Self-guided learning to denoise for robust recommendation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 1412–1422. He, X.; Deng, K.; Wang, X.; Li, Y.; Zhang, Y.; and Wang, M. 2020. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’20, 639–648. New York, NY, USA: Association for Computing Machinery. ISBN 9781450380164. He, Z.; Wang, Y.; Yang, Y.; Sun, P.; Wu, L.; Bai, H.; Gong, J.; Hong, R.; and Zhang, M. 2024. Double Correction Framework for Denoising Recommendation. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. Lin, W.; Zhao, X.; Wang, Y.; Zhu, Y.; and Wang, W. 2023. Autodenoise: Automatic data instance denoising for recommendations. In Proceedings of the ACM Web Conference 2023, 1003–1011. Lin, Z.; Tian, C.; Hou, Y.; and Zhao, W. X. 2022. Improving graph collaborative filtering with neighborhoodenriched contrastive learning. In Proceedings of the ACM web conference 2022, 2320–2329. Luo, H.; Zhou, J.; Bao, Z.; Li, S.; Culpepper, J. S.; Ying, H.; Liu, H.; and Xiong, H. 2020. Spatial Object Recommendation with Hints: When Spatial Granularity Matters. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. Ren, X.; Wei, W.; Xia, L.; Su, L.; Cheng, S.; Wang, J.; Yin, D.; and Huang, C. 2024. Representation learning with large language models for recommendation. In Proceedings of the ACM on Web Conference 2024, 3464–3475. Rendle, S.; Freudenthaler, C.; Gantner, Z.; and SchmidtThieme, L. 2009. BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, UAI ’09, 452–461. Arlington, Virginia, USA: AUAI Press. ISBN 9780974903958.
Shi, W.; Chen, J.; Feng, F.; Zhang, J.; Wu, J.; Gao, C.; and He, X. 2023. On the theories behind hard negative sampling for recommendation. In Proceedings of the ACM Web Conference 2023, 812–822. Tian, C.; Xie, Y.; Li, Y.; Yang, N.; and Zhao, W. X. 2022. Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’22, 122–132. New York, NY, USA: Association for Computing Machinery. ISBN 9781450387323. Wang, W.; Feng, F.; He, X.; Nie, L.; and Chua, T.-S. 2021a. Denoising implicit feedback for recommendation. In Proceedings of the 14th ACM international conference on web search and data mining, 373–381. Wang, W.; Feng, F.; He, X.; Zhang, H.; and Chua, T.-S. 2021b. Clicks can be cheating: Counterfactual recommendation for mitigating clickbait issue. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 1288–1297. Wang, X.; He, X.; Wang, M.; Feng, F.; and Chua, T.-S. 2019. Neural graph collaborative filtering. In Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval, 165–174. Wang, Y.; Xin, X.; Meng, Z.; Jose, J. M.; Feng, F.; and He, X. 2022. Learning robust recommenders through crossmodel agreement. In Proceedings of the ACM Web Conference 2022, 2015–2025. Wang, Z.; Gao, M.; Li, W.; Yu, J.; Guo, L.; and Yin, H. 2023. Efficient bi-level optimization for recommendation denoising. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2502–2511. Wang, Z.; Xu, Q.; Yang, Z.; Cao, X.; and Huang, Q. 2021c. Implicit feedbacks are not always favorable: Iterative relabeled one-class collaborative filtering against noisy interactions. In Proceedings of the 29th ACM International Conference on Multimedia, 3070–3078. Wei, W.; Ren, X.; Tang, J.; Wang, Q.; Su, L.; Cheng, S.; Wang, J.; Yin, D.; and Huang, C. 2024. Llmrec: Large language models with graph augmentation for recommendation. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, 806–815. Wu, J.; Wang, X.; Feng, F.; He, X.; Chen, L.; Lian, J.; and Xie, X. 2021. Self-supervised graph learning for recommendation. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval, 726–735. Wu, L.; Zheng, Z.; Qiu, Z.; Wang, H.; Gu, H.; Shen, T.; Qin, C.; Zhu, C.; Zhu, H.; Liu, Q.; et al. 2024. A Survey on Large Language Models for Recommendation. arXiv:2305.19860. Xi, Y.; Liu, W.; Lin, J.; Cai, X.; Zhu, H.; Zhu, J.; Chen, B.; Tang, R.; Zhang, W.; Zhang, R.; and Yu, Y. 2023. Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models. arXiv:2306.10933. Zhang, W.; Liu, H.; Xiong, H.; Xu, T.; Wang, F.; Xin, H.; and Wu, H. 2023. RLCharge: Imitative Multi-Agent Spatiotemporal Reinforcement Learning for Electric Vehicle Charging
Station Recommendation. IEEE Transactions on Knowledge and Data Engineering, 35: 6290–6304. Zheng, B.; Hou, Y.; Lu, H.; Chen, Y.; Zhao, W. X.; Chen, M.; and Wen, J.-R. 2024a. Adapting large language models by integrating collaborative semantics for recommendation. In 2024 IEEE 40th International Conference on Data Engineering (ICDE), 1435–1448. IEEE. Zheng, Z.; Chao, W.; Qiu, Z.; Zhu, H.; and Xiong, H. 2024b. Harnessing large language models for text-rich sequential recommendation. In Proceedings of the ACM on Web Conference 2024, 3207–3216.
# Supplementary Materials
Details of Dataset
In this section, we offer details of the preprocessed dataset adopted in the experiment. We take the datasets in RLMRec (Ren et al. 2024), in which each item contains a corresponding item text profile. Therefore, we follow the preprocessing setting in the RLMRec. Specifically, interactions with ratings below 3 for both the Amazon-books and Yelp data are filtered out. No rating-based filtering is adopted in Steam. K-core filtering is also performed and split into training, validation, and test sets using a 3:1:1 ratio. The statistics of datasets preprocessed following RLMRec are shown in Table 3.
Datasets
# Users # Items # Interactions # Sparsity
Amazon-books
11,000
9,332
120,464
99.88%
Yelp
11,091
11,010
166,620
99.86%
Steam
23,310
5,237
316,190
99.74%
<div style="text-align: center;">Table 3: Statistics of preprocessed datasets.</div>
However, previous works adopted the rating score to label noise and clean data. For example, T-CE regards a rating score below 3 as a false-positive interaction. As a result, the dataset filtered with ratings in RLMRec is regarded to contain less noisy interactions. To compare the denoising ability of different methods, we add 5% noisy interactions to the training set. These noisy interactions are selected from the interactions that are rated below 3. Experiments are then conducted on these noise-inserted datasets.
# Details of API Token Cost
Since we adopt the GPT-3.5-turbo as the LLM in the LLMHD, here we provide the token number for training with LLMHD in Table 4. The total token number of Tscore in LLMHD is highly dependent on two aspects: the number of interactions in the dataset, the value of the maximum noise scale εmax l , and the hard sample proportion εv. Whears, the Tsum is not correlated to the hyper-parameters. We also report the token number for TF N, and TF P during training when the confidence threshold is set to εγ = 3.
We provide the details of plotting the Figure 1. For the noise samples, we follow the settings of (Wang et al. 2021a), tak-
Datasets
Template
εmax
l
εv
# Token
Amazon-books
Tsum
-
-
16m
TF N
-
-
6m
TF P
-
-
3m
Tscore
0.05
0.5
11m
0.10
0.5
21m
0.05
0.3
7m
Yelp
Tsum
-
-
19m
TF N
-
-
8m
TF P
-
-
3m
Tscore
0.05
0.5
16m
0.10
0.5
30m
0.05
0.3
10m
Steam
Tsum
-
-
40m
TF N
-
-
10m
TF P
-
-
4m
Tscore
0.05
0.5
35m
0.10
0.5
62m
0.05
0.3
20m
<div style="text-align: center;">Table 4: OpenAI API token number.</div>
ing the interactions that rate below 3 as the false-positive noise. By flipping labels of ground truth records in the test set, we can obtain a set of false-negative interactions that are positively labeled but unobserved during the negative sampling process. Samples in which the positive item is falsepositive and the negative item is false-negative are considered noisy samples. We then identify hard and easy samples according to the setting in (Ding et al. 2020). For each positive interaction, D negative items are sampled. Then the negative item with the highest prediction score of ˆyu,i is adopted as the negative instance during training. Thus, when the D gets higher, the sample becomes harder. According to this setting, we collect the prediction score and loss value results when D = 1 and that of hard samples when D = 3.
# Details of Prompt Template
In this section, we offer comprehensive information on the templates utilized in the LLMHD. Real examples from the Amazon-book dataset are used as a showcase. The templates used in the Yelp and Steam datasets are with minor differences in the instructions provided to represent different data.
In this section, we offer comprehensive information on the templates utilized in the LLMHD. Real examples from the Amazon-book dataset are used as a showcase. The templates used in the Yelp and Steam datasets are with minor differences in the instructions provided to represent different data. Example of User Preference Summarization. Figure 5 cases an example of user preference profile generation, specifically for the Amazon-book dataset. The instruction provided to the language model for all users remains consistent, directing the LLMs to summarize the characteristics of books that would appeal to the user. The input item profiles consist of the book title and a corresponding item description from the dataset. To facilitate parsing the output, we enforce the output format to the XML. The generated result demonstrates that the LLMs, in this case ChatGPT, capture the common features from the interacted items. Example of LLM Sample Scoring. Figure 6 illustrates the process of scoring user preference for an item with LLMs. In order to achieve the user preference for a specific
Example of LLM Sample Scoring. Figure 6 illustrates the process of scoring user preference for an item with LLMs. In order to achieve the user preference for a specific
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/29a1/29a17e7e-302c-49d5-ae51-2a46f8918b54.png" style="width: 50%;"></div>
# User Preference Summarization  (𝑻𝒔𝒖𝒎)
You are a professional book editor. Below is the information about books that a reader has read: 1. <ITEM PROFILE 1> … N. <ITEM PROFILE N> 
You are a professional book editor. Below is the information about books that a reader has read: 1. <ITEM PROFILE 1>
# Based on the books the reader has read, please summarize characteristics of books this reader may  enjoy.
You MUST provide the summarization with the following format: <summarization>A summarization of  what kinds of books this reader is likely to enjoy.</summarization>. Please ensure that the " summarization " is no longer than 100 words. You should not provide any explanation except the summarization.
# User Preference Scoring (𝑻𝒔𝒄𝒐𝒓𝒆)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ea05/ea05e82f-547f-43d6-920f-9d3c1d098966.png" style="width: 50%;"></div>
You are a professional book editor. Below is the reading preference of a reader: <USER PROFILE>
Now here is the descriptions of a book: <ITEM PROFILE>
You MUST provide
<s>8</s>
Figure 6: Example of user preference scoring process on Amazon-books dataset.
# User Preference Update with False Positives (𝑻𝑭𝑷)
User Preference Update with False Positives (𝑻𝑭𝑷)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b3e6/b3e63d38-3265-4da7-ad49-ec9ec5f97f83.png" style="width: 50%;"></div>
You are a professional book editor.
You will help me to determine a reader’s reading preference.
I will provide you with: 
READER PROFILE: a description of the reader’s potential reading preferences.
NOT INTERESTED BOOKS: descriptions of books that the reader might not interested in.
READER PROFILE: <USER PROFILE>
NOT INTERESTED BOOKS:
1. <ITEM PROFILE 1>
…
Based on the characteristics of "NOT INTERESTED BOOKS", please adjust the provided "READER 
PROFILE" to make it more compatible with the reader’s actual reading preferences
You MUST answer in following format: <profile>....</profile> (e.g. <profile>The reader is likely to 
enjoy...</profile>)
Please ensure that the "profile" is no longer than 100 words.
Do not provide any other text except the "profile" string.
# system prompt
# item profile
# task description
# output format
# user profile
<profile>The reader is likely to enjoy books that explore magical worlds and supernatural beings, with a 
focus on strong female leads … </profile>
# LLM Response
Updated User Profile
Input Prompt
You are a professional book editor ……
READER PROFILE:
The reader enjoys a variety of genres including supernatural romance, young adult fantasy, 
contemporary romance, and paranormal romance …..
NOT INTERESTED BOOKS:
1. Fans of paranormal romance and fiction who enjoy reading series with prequels and sequels 
would enjoy New Beginnings: Prequel to Others of Edenton. 
…..
# item profile
# user profile
Figure 7: Example of update user preference with False-positive item.
User Preference Update with False Negatives (𝑻𝑭𝑵)
You are a professional book editor.
You will help me to determine a reader’s reading preference.
I will provide you with: 
READER PROFILE: a description of the reader’s potential reading preferences.
INTERESTED BOOKS: descriptions of books that the reader might be interested in.
READER PROFILE: <USER PROFILE>
INTERESTED BOOKS:
1. <ITEM PROFILE 1>
…
Based on the characteristics of “INTERESTED BOOKS”, please adjust the provided “READER PROFILE” to 
make it more compatible with the reader’s actual reading preferences
You MUST answer in following format: <profile>....</profile> …
# system prompt
# item profile
# task description
# output format
# user profile
Figure 8: Example of update user preference with false-negative item.
You are a professional book editor. You will help me to determine a reader’s reading preference.
NOT INTERESTED BOOKS: 1. <ITEM PROFILE 1>
…
Based on the characteristics of "NOT INTERESTED BOOKS", please adjust the provided "READER PROFILE" to make it more compatible with the reader’s actual reading preferences You MUST answer in following format: <profile>....</profile> (e.g. <profile>The reader is likely to  enjoy...</profile>) Please ensure that the "profile" is no longer than 100 words. Do not provide any other text except the "profile" string.
Based on the characteristics of "NOT INTERESTED BOOKS", please adjust the provided "READE PROFILE" to make it more compatible with the reader’s actual reading preferences
You MUST answer in following format: <profile>....</profile> (e.g. <profile>The reader is likely to  enjoy...</profile>) Please ensure that the "profile" is no longer than 100 words. Do not provide any other text except the "profile" string.
You are a professional book editor ……
READER PROFILE:
# User Preference Update with False Negatives (𝑻𝑭𝑵)
You are a professional book editor. You will help me to determine a reader’s reading preference.
I will provide you with:  READER PROFILE: a description of the reader’s potential reading preferences. INTERESTED BOOKS: descriptions of books that the reader might be interested in.
INTERESTED BOOKS: 1. <ITEM PROFILE 1>
Based on the characteristics of “INTERESTED BOOKS”, please adjust the provided “READER PROFILE” to  make it more compatible with the reader’s actual reading preferences
You MUST answer in following format: <profile>....</profile> …
item, the prompt template incorporates the user preference profile generated in the User preference Profile generation and the item profile in the dataset. By utilizing both information, LLMs are empowered to infer the user’s preference. In the presented example, leveraging the user preference profile
and item profile, the large language model assesses the preference for the book accurately. In addition, to maintain consistency, we ask the model to generate scores from 1 to 10, and correlated descriptions about the meaning of different scores are also provided. This enables the model to generate
scores in a specific range, making subsequent judgments of hard samples easier.
Example of User Preference Update. Figure 7 shows the overall process of updating user preference with Falsepositive items. With the provided item profile and user profile, the LLM successfully refined the provided user preference profile by removing correlated user-dislike item characteristics. This demonstrates the promising ability of LLMs to understand actual user preferences by providing user-like or disliked items. A similar effect is also achieved by updating user preference with False-negative items, where the prompt template is shown in Figure 8.
