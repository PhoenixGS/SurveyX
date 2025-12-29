# CRoP: Context-wise Robust Static Human-Sensing Personalization
Sawinder Kaur1, Avery Gump2, Jingyu Xin1, Yi Xiao4, Harshit Sharma4, Nina R Benway3 Jonathan L Preston1, Asif Salekin4
27 Sep 2024
1Syracuse University 2University of Wisconsin-Madison 3University of Maryland-College Park 4Arizona State University
# Abstract
The advancement in deep learning and internet-of-things have led to diverse human sensing applications. However, distinct patterns in human sensing, influenced by various factors or contexts, challenge generic neural network model’s performance due to natural distribution shifts. To address this, personalization tailors models to individual users. Yet most personalization studies overlook intrauser heterogeneity across contexts in sensory data, limiting intra-user generalizability. This limitation is especially critical in clinical applications, where limited data availability hampers both generalizability and personalization. Notably, intra-user sensing attributes are expected to change due to external factors such as treatment progression, further complicating the challenges. This work introduces CRoP, a novel static personalization approach using an off-the-shelf pre-trained model and pruning to optimize personalization and generalization. CRoP shows superior personalization effectiveness and intra-user robustness across four human-sensing datasets, including two from real-world health domains, highlighting its practical and social impact. Additionally, to support CRoP’s generalization ability and design choices, we provide empirical justification through gradient inner product analysis, ablation studies, and comparisons against state-of-theart baselines.
arXiv:2409.17994v2
# Introduction
AI in human sensing applications—like activity recognition, fall detection, and health tracking—revolutionizes
daily life, especially in personal health management [Wang et al., 2023]. However, unique user patterns and natural distribution shifts [Gong et al., 2023] caused by behaviors, physical traits, environment, and device placements [Ustev et al., 2013, Stisen et al., 2015] lead to the underperformance of generic AI models in practical use. To tackle this, various domain adaptation techniques have been explored, with personalization widely used to adapt a generic model to the target user’s specific domain or natural distribution [Lamichhane et al., 2023, Iaboni et al., 2022, Meegahapola et al., 2023, Ahamed and Farid, 2018, Ren et al., 2022, Sempionatto et al., 2021, Boukhechba et al., 2020]. In literature, personalization occurs either during the enrollment phase (static) [Duan et al., 2023, Liu et al., 2022, Burns et al., 2022] or continuously throughout application use [Daniels et al., 2023, Liu et al., 2024, Wu et al., 2024, Wang et al., 2022a]. Static personalization customizes the model with limited individual data collected at enrollment, requiring minimal computation and user engagement, making it highly practical for human-sensing applications. However, existing such studies often overlook intra-user variability due to factors like changes in magnetic field [Robert-Lachaine et al., 2017], sensor position [Park et al., 2014], terrain [Kowalsky et al., 2021], or the health symptoms P¨aeske et al. [2023], leading to poor intrauser generalizability for contexts not present during personalization. For instance, a smartphone activity recognition model personalized with handheld data may perform poorly when the phone is in a pocket. Static personalization is particularly crucial for clinical datasets, which are often characterized by data scarcity, leading to reduced robustness of lab-validated models for
prospectively collected users Berisha et al. [2021]. It enhances model accuracy for clinical users whose traits are underrepresented in the global model’s training data. In contrast, continuous supervised personalization is generally infeasible in many health domains since ground truths must be validated by clinicians, making it impractical in continuous settings, especially in remote or mobile health applications. Nevertheless, the distribution of clinical data is expected to shift, even within the same individual. For instance, in clinical speech technologies, changes in data distribution over time may occur due to the progression of neurodegenerative diseases, relevant for disease monitoring apps Stegmann et al. [2020], or through desired learning mechanisms resulting from the use of technology, as seen in automated speech therapy apps Benway and Preston [2023]. Similarly, in stress monitoring via wearables, the distribution of psychophysiological data changes as the same individuals encounter different types of stressors Nagaraj et al. [2023]. This research defines ‘context’ as the intra-user data distribution formed by varying external factors. This research gap is worsened since static personalization typically relies on a small sample set from the target user, covering limited contexts—particularly in clinical settings or applications with data scarcity Berisha et al. [2021], Benway and Preston [2023]. Commercial human sensing technologies like Google Assistant, Amazon Alexa, and Apple’s Siri also personalize speech recognition models using limited phrases during enrollment [Team, 2017, Awobajo, 2023, Phelan, 2019]. Similarly, the Apple Watch uses initial calibration for enhanced running activity tracking [Apple, 2023, Potuck, 2021]. This limited context during personalization is problematic, as shown in this paper’s Motivation Section, where we demonstrate that static personalization may improve performance in training contexts but can also significantly degrade it in other unseen contexts for the same user. Therefore, given the importance of static personalization in human sensing, this paper addresses its intra-user generalizability gap. As shown in Figure 1, this paper endeavors to personalize an off-the-shelf generic model for a specific user using limited data from limited contexts. The primary objective is to ensure that the personalized model thus obtained exhibits robust generalization capabilities across unseen contexts. Crucially, unseen context
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3bb5/3bb57b9b-bd4f-45bd-9f30-ee7409fccc9d.png" style="width: 50%;"></div>
Off-the-Shelf Generic Model
<div style="text-align: center;">Figure 1: Problem Setting</div>
data takes no part in training or adjusting the personalized model outcome, and the generic model remains entirely off-the-shelf, with no accessibility for modification or design choices. These constraints highlight the real-world impact of this research, particularly in clinical settings where privacy concerns often limit data sharing Malin et al. [2018], Rathbone et al. [2023], and only trained offthe-shelf models are shared among researchers and developers. To achieve the research objective in Figure 1, this paper introduces CRoP, a novel approach to create context-wise robust static personalized models. The key contributions are:
2. CRoP is the first to leverage model pruning with adaptive intensity to facilitate the effective identification and integration of generic and personalized model weights to facilitate context-wise robustness for personalizing human sensing applications.
 To showcase CRoP’s efficacy, comprehensive evaluations were performed on four human sensing datasets: PERCERT-R Benway et al. [2023]: a clinical speech therapy dataset, WIDAR [Zhang et al., 2022]: a lab-based WiFi-CSI dataset, ExtraSensory: a real-world mobile sensing dataset [Vaizman et al., 2017], and a stress-sensing dataset [Xiao et al.,
# 2024], while considering two disjoint contexts for each dataset.
2024], while considering two disjoint contexts for each dataset.
4. An empirical justification of CRoP’s design choices that enable intra-user generalizability among different contexts is provided, employing Gradient Inner Product(GIP) [Shi et al., 2021] analysis.
The work is accompanied by an extensive appendix, which includes a detailed discussion of the experimental setup, data collection, and related work. Additionally, the appendix provides a detailed analysis of the personspecific results for each dataset along with a detailed ablation study.
# Related Work
A few static personalization approaches [Burns et al., 2022, Duan et al., 2023, Liu et al., 2022] aimed for the additional goal of out-of-distribution robustness. However, these methods require access to the generic model—either to make specific design choices [Burns et al., 2022], which prevents them from utilizing off-the-shelf models, or to incorporate knowledge about the target user’s data distribution during the generic model’s training phase [Duan et al., 2023, Liu et al., 2022], raising privacy concerns, particularly in sensitive clinical applications. These requirements do not align with the research objectives of this paper, making them unsuitable as baselines. A different set of approaches that do consider the privacy concern is referred to as source-free domain adaptation [Liang et al., 2020]. Liang et al. [2020] (SHOT) proposed the transfer of hypothesis from source by freezing the parameter weights for the classifier layers and only allowing feature extraction to be finetuned to the new domain. The approach is applicable to unsupervised domain adaptation scenarios and employs self-supervised pseudolabeling to align the target domain’s representations to the source hypothesis. However, these approaches do not address the constraint of limited-context data during finetuning. We adapted SHOT in this paper’s problem setting as one of the baselines. Continuous personalization approaches Wang et al. [2022a], Liang et al. [2020], Wu et al. [2024], Daniels et al. [2023], Fini et al. [2022], Tang et al. [2024a,b], Mallya and Lazebnik [2018], Mallya et al. [2018], Wang
Model
Generic
Personalized
∆
User
C1
C2
C1
C2
C1
C2
0
63.90
77.09
87.06
65.02
+23.16
-11.88
1
61.80
79.78
89.38
44.38
+27.57
-35.40
2
45.63
79.81
71.88
64.45
+29.75
-26.62
Average
+26.82
-24.63
<div style="text-align: center;">Table 1: Performance Comparison of Generic model with conventionally trained personalized model</div>
et al. [2022b] can improve intra-user generalizability by continually fine-tuning the model as new data arrives. Some of these approaches Fini et al. [2022], Tang et al. [2024a,b] require specialized training of the generic model, limiting the use of off-the-shelf pre-trained models. Others like PackNet Mallya and Lazebnik [2018] and Piggyback Mallya et al. [2018] propose supervised methods that require continued steam of labeled data, limiting their application in health-care scenarios. Additionally, Continual Test Time Domain Adaptation (CoTTA) Wang et al. [2022b] proposes unsupervised learning methods and allows the use of off-the-shelf models. However, all continuous learning approaches require repeated computation overhead to adjust the model outcome to new data [Prabhu et al., 2023], which can be infeasible in real-time applications, more so for scalable platforms like wearables Schmidt et al. [2018], which is prominent for health sensing such as stress or fall detection. Nevertheless, since the problems addressed by Packnet, Piggyback, and CoTTA are the closest to the problem addressed in this study, we considered these approaches as baselines. Appendix C provides further details of the related work.
# Motivation
When learning patterns from human sensing data in a limited context, conventional fine-tuning approaches can overwrite generic knowledge that is not relevant to that specific context but applicable to others, leading to a performance drop in those unrepresented contexts. To illustrate this, we conducted a preliminary study comparing the performance of generic and conventionallyfinetuned [Hong et al., 2016] personalized humangesture-recognition models using the LeNet architecture [Zhang et al., 2022] trained on the WIDAR dataset. Data preprocessing details are discussed in the Experiments section.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b6a4/b6a4c23f-6f6a-464b-8c00-e4c8b1185eee.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Context C1</div>
<div style="text-align: center;">(b) Context C2</div>
Figure 2: Heat map for the absolute magnitude of parameters belonging to penultimate layer for LeNet models finetuned using data from context (a) C1 and (b) C2
Table 1 compares the performance of generic and conventionally-finetuned personalized models on each user’s data belonging to context C1 and evaluated to the same user’s disjoint data in both C1 (available) and C2 (unseen) contexts. It can be observed that conventional finetuning introduces a significant gain of 26.82% for context C1’s data but at the cost of 24.63% reduction in context C2. Similar patterns are seen when personalization is performed on context C2, as shown in Appendix D. Thus, conventional finetuning-based personalization of the models using the limited data from one context can significantly worsen the model’s performance in an unseen context. To investigate this discrepancy in performance, we compare the distribution of parameter magnitudes of the models personalized on contexts C1 and C2 using conventional finetuning, as shown in Figures 2 (a) and 2 (b). Notably, there is a substantial difference in parameter magnitudes between models trained in different contexts. Additionally, the parameters represented by black pixels in Figure 2 have magnitudes close to zero, indicating two crucial aspects: (a) Their contribution to model inference is negligible, implying redundancy. (b) Interestingly, some of these parameters have high magnitudes in the personalized model of another context, indicating that parameters considered unimportant in one context may be crucial in another.
The critical question arises: How can we effectively retain and transfer the valuable generic information about context C2 to the personalized models without access to context C2? – that this paper addresses.
# Problem Statement
Given a generic model MG θ , the objective is to tailor a personalized model MP a i θ specifically for a user Ui utilizing the data Da i associated with available context Ca, here θ represents the parameters of the model. The primary goal is to ensure that the personalized model MP a i θ performs reasonably well on Ui’s data Du i derived from an unseen context Cu. Notably, there is no overlap between the data belonging to the two contexts, that is Da i ∩Du i = ϕ. In other words, if MCa i θ represents a conventionallyfinetuned model trained for a user Ui on data Da i , then, the models trained using CRoP, MP a i θ , must on avg. perform better on both available Ca and unseen context Cu than MCa i θ . More formally, learning objective can be defined as: �
such that Da i ∩Du i = ϕ and
� d∈{Du i ,Da i } ℓ(MP a i θ , d) < � d∈{Du i ,Da i } ℓ(MCa i θ , d),
that is, the loss incurred by the resulting personalized model MP a i θ on avg. across all contexts’ data is less than the loss incurred by conventionally-finetuned model MCa i θ . Here, ℓrepresents the standard cross-entropy loss. It is important to emphasize that the above-mentioned optimization problem restricts the usage of data to the available context Ca and has no knowledge of data from the unseen context Cu. Hence, for d ∈Du i (unseen context data), the information ℓ(MP a i θ , d), and ℓ(MCa i θ , d) is absent during the training process.
# Approach
# Rationale for The CRoP Approach Design
As previously discussed, the generic model’s parameters contain generalizable information across all contexts. Addressing the problem statement requires retaining this information to the greatest extent while enabling fine-tuning for the target user. Furthermore, our investigation revealed
that different parameters hold varying degrees of importance in distinct contexts. Hence, the careful selection of subsets of model parameters for personalization and generalization is pivotal for the success of the approach, for which this paper leverages the model pruning paradigm. Model pruning is based on the idea that neural networks include redundant parameters, and removing these parameters has minimal impact on the model’s performance [Luo et al., 2017, Zhu and Gupta, 2017]. Consequently, pruning the fine-tuned personalized model ensures the retention of essential parameters to maintain accuracy for context Ca. However, the pruned parameters can be replaced with corresponding parameters from the generic model, effectively restoring generic knowledge learned across all contexts on those generic model parameters. This restoration may enhance generalizability, ensuring robust performance in unseen contexts Cu. The approach presented in this paper is founded on this insightful strategy.
# CRoP Approach
Algorithm 1 describes the presented approach which takes as input: the generic model MG, user U′ is data Da i for available context Ca, initial value for coefficient of regularization α and tolerance for pruning τ; and generates the target personalized model MP a i θ . Here, α and τ are hyperparameter whose values can be tuned for the given data and model. The approach initiates by finetuning the generic model MG θ on data Da i , concurrently applying ℓ1 regularization to penalize model parameters (line 2). This regularization encourages sparsity by specifically targeting the magnitude of redundant parameters [Mayank, 2023]. This step is followed by the pruning of redundant weights using the ‘ToleratedPrune’ module (line 3). The pruned weights are then replaced by the corresponding weights from the generic model MG (line 4) to restore generalizability; this hybrid model is referred to as the ‘Mixed Model.’ This step leads to the modification of the activated paths in the personalized model, resulting in changes in the model inferences. However, since the newly activated paths are determined by weights retained from two models and not learned from data patterns, there is a consequent loss of accuracy, as shown and discussed in Appendix E. To mit-
1: Input: MG θ : Generic Model ⋄Da i : User U′ is data for available context Ca ⋄α: coefficient of regularization ⋄τ: tolerance for pruning 2: Train the Generic model on the personal data Da i
3: Prune redundant parameters to obtain the pruned substructure
4: Copy the parameters of generic models to the pruned parameters in the personalized pruned model,
MP a i θ′′ = � MP a i θ↓ , θ↓̸= 0 MG θ , otherwise
5: Finetune the personalized model on the Da i
5: Finetune the personalized model on the Da i MP a i θ = argmin θ � d∈Da i ℓ(MP a i θ′′ , d) + α∥MP a i θ′′ ∥1
MP a i θ = argmin θ � d∈Da i ℓ(MP a i θ′′ , d) + α∥MP a i θ′′ ∥1
igate such a loss, as a final step, the Mixed Model undergoes fine-tuning once again on the data from the available context Da i (line 5). The detailed explanation of each of these steps is as follows:
# Personalized Finetuning with Penalty (Algorithm 1 –
Personalized Finetuning with Penalty (Algorithm 1 – Step 2): The approach uses data Da i to finetune the generic model MG θ . As shown in the motivation section, such conventional finetuning enhances the model’s accuracy within the available context Ca. Nevertheless, its performance in unfamiliar contexts may get suboptimal. Notably, during the model’s fine-tuning process, we apply ℓ1 regularization to penalize the model weights, forcing the magnitudes of redundant parameters to be close to zero [Mayank, 2023]. The regularization coefficient α is a trainable parameter optimized during training to minimize the overall loss. As a result, the parameters with
Algorithm 2: ToleratedPrune(M, τ, D)
1: Input: Mθ: A Model ⋄τ: tolerance for pruning ⋄
D: data
2: Pruning Amount p = k
3: Ao = accuracy(Mθ, D)
4: repeat
5:
Mθ↓= Mθ
6:
Mθ = Prune(Mθ, p)
7:
A = accuracy(Mθ, D)
8:
Increment Pruning Amount p = p + k′
9: until A < Ao −τ
10: return Mθ↓
# high magnitudes carry most of the information regarding the data patterns in Da i , offering two key benefits:
1. Minimal loss in Ca accuracy: A high fraction of parameters have close to zero magnitudes, and their removal results in minimal information loss for context Ca; thus, the adverse impact of pruning in context Ca is minimized. 2. Maximal generalization: The inclusion of regularization aids ToleratedPrune (discussed below) module in efficiently pruning a higher number of parameters, which are then replaced with weights from the generic model. This restores information from the generic model, contributing to enhanced accuracy in unseen contexts.
ToleratedPrune Module (Algorithm 1 – Step 3): Algorithm 2 outlines the ToleratedPrune module, taking a model Mθ, pruning tolerance τ, and the dataset D as inputs. It initiates with a modest pruning amount of k and incrementally increases this amount by k′ until the model’s accuracy exhibits a drop of τ percent on D. Here, k and k′ are hyperparameters within the range of (0, 1). The module returns Mθ↓, representing the pruned state of the model before the last pruning iteration. This state is such that further pruning would result in a higher accuracy loss on dataset D than the tolerable amount τ. This module performs pruning leveraging the conventional magnitudebased unstructured pruning [Zhu and Gupta, 2017]. Thus, step 3 in Algorithm 1 generates a pruned personalized model state MP a i θ↓whose prediction accuracy on
context Ca is at most τ percent lower than that of the earlier state MP a i θ′ while using only a fraction of its original parameters. The non-zero weights corresponding to these parameters contribute significantly to model inference for the available context Ca. As a result, MP a i θ↓is essentially the minimal sub-structure of the earlier state model MP a i θ′ , which is crucial for correct inference for context Ca. This enables replacing a maximal number of zeroed-out parameters to incorporate information from unseen contexts using the generic model MG θ in the subsequent steps.
Generating the Mixed Model (Algorithm 1 – Steps 4&5): For generating the Mixed Model MP a i θ′′ , the zeroed out parameters in the pruned model MP a i θ↓are replaced by the corresponding parameters in the generic model MG θ , enabling generic knowledge restoration. Notably, model pruning is often followed by a finetuning step [Luo et al., 2017, Zhu and Gupta, 2017, Liu et al., 2020], where the pruned model undergoes re-training to recover the performance lost during the pruning process. We have observed that, despite the Mixed Model exhibiting improved performance in the unseen context, there is a notable loss of accuracy in the available context due to inconsistent activated paths, as discussed earlier. Thus, the resulting Mixed Model is fine-tuned using the available data Da i . Goyal et al. [2023] suggests that fine-tuning process should mirror pre-training for effective generalization. Therefore, our fine-tuning objective aligns with the pre-training objective used in Line 2 for optimal results. During finetuning, the model state, including the mixed model, with the best validation loss on the seen context, is selected. We found that for some individuals, the mixed model is chosen as the optimal model, which indicates that for some individuals, further finetuning is not required, and our approach can automatically handle that scenario.
# Experiments
This work employs four real-world human-sensing datasets to demonstrate the empirical efficacy of CRoP, two of which are associated with health applications. First, the PERCEPT-R dataset has been used for binary classification for predicting the correctness of / r / sounds
in automated speech therapy application Benway and Preston [2023]. In order to leverage this dataset for our personalization evaluation, we collaborated with clinical experts to identify and acquire annotations of 16 participants who had correct and incorrect pronunciations of / r / sound at pre-treatment (baseline-phase) and during different treatment phases. Additionally, we use the Stress Sensing dataset Xiao et al. [2024] collected using a psycho-physiological wrist-band, named Empatica E4 [empetica, 2015]. To further demonstrate the efficacy of CRoP, we incorporate two benchmark human-sensing datasets, which include data from the same individuals across multiple contexts: WIDAR [Zhang et al., 2022] and ExtraSensory [Vaizman et al., 2017]. Specifically, we employ WIDAR for a 6-class classification focusing on gesture recognition using WiFi signals, and ExtraSensory for binary classification related to human activity recognition using accelerometer and gyroscope readings. A detailed discussion about the datasets and models, hyperparameters, compute resources and instructions to access code are provided in Appendix A.
# Pre-Processing of the Datasets:
We partitioned each dataset into two disjoint sets of users: (1) a generic dataset for training a generic model and (2) a personalized dataset for training a personalized model for each user. To demonstrate the context-wise robustness, we further partitioned each user’s personalized dataset into different contexts. Table 2 presents the details of this partitioning. For PRECEPT-R, we consider data from the pretreatment phase as the available context, and the treatment phases, where participants undergo clinical interventions, are considered the unavailable context. For the Stress Sensing dataset, the context is determined by two factors: the hand on which the sensor (Empatica E4 wristband empetica [2015]) was worn during data collection and the movement status of the individual. For WIDAR, context is determined by the room and torso orientation during data collection, while for the ExtraSensory dataset, phone’s location on the user’s body (e.g., hand, pocket, bag) defines the context. The term ‘Scenario’ refers to the combination of available Ca and unseen Cu contexts as outlined in Table 2. All datasets, along with context-wise annotations, will be made public. Notably, throughout the training of personalized mod-
els, CRoP refrains from utilizing any information from the unseen context Cu. Therefore, while the empirical study indicates an enhancement in the model’s performance for one or a few unseen contexts, it is a proxy for all unseen contexts. Meaning, it is reasonable to anticipate a favorable performance in other unseen contexts as well. Notably, the stress sensing dataset has been evaluated across two different unseen contexts, C1 u and C2 u, none of which participated in the training of the personalized model.
# Metrics for evaluation
To establish the efficacy of CRoP, we quantify the extent of personalization and generalization achieved through the presented approach. Personalization is gauged by comparing our model’s MP a i θ accuracy relative to the generic model MG θ , while for generalization, we assess the accuracy of our model MP a i θ against conventionallyfinetuned personalized models MCa i θ . Both of these metrics consider classification accuracy in the available Ca and unseen Cu contexts. If A(M, D) represents the classification accuracy of the model M for dataset D and n is the number of users selected for personalization, the metrics of evaluations can be described as follows :
1. Personalization (∆P ): It is defined as the sum of the difference between the accuracy of MP a i θ and MG θ over all the contexts averaged over all users
∆P = 1 n � Ui � C∈{Ca,Cu} (A(MP a i θ , C) −A(MG θ , C))
2. Generalization (∆G): It is defined as the sum of the difference between the accuracy of MP a i θ and MCa i θ over all the contexts averaged over all users.
∆G = 1 n � Ui � C∈{Ca,Cu} (A(MP a i θ , C)−A(MCa i θ , C))
All the results in this section are computed as an average of accuracy obtained for three random seeds.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5872/58720bdb-d2b3-4458-a49c-e0a965bb7053.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 2: Details of data used for personalization</div>
# Comparison with SOA
To demonstrate the efficacy of CRoP in achieving personalization ∆P while maintaining generalization ∆G, we compare CRoP with 4 state-of-the-art approaches SHOT [Liang et al., 2020], PackNet Mallya and Lazebnik [2018], Piggyback Mallya et al. [2018], and CoTTA Wang et al. [2022b]. Table 3 compares the performance of CRoP with aforementioned baseline approaches. The values for ∆P and ∆G are computed as average over all the participants used for personalization for each dataset. The detailed results for participant-specific evaluations for each dataset are provided in Appendix B and Appendix B.4 shows the errors bars for our approach. It can be observed in Table 3 that CRoP significantly outperforms all the SOA approaches. On average, the personalization benefits ∆P achieved by SHOT, PackNet, Piggyback and CoTTA are 2.16, 26.05, 18.01 and 9.95 percent points, respectively, while CRoP can achieve 35.23 percent points. However, while comparing ∆G, one can observe that personalized training using SHOT, PackNet, Piggyback and CoTTA harms generalizability by −25.73, −1.39, −9.43, and −17.49 percent points respectively. On the other hand, CRoP shows an average generalization benefit of 7.78. Additionally, it is evident that the unsupervised approaches SHOT and CoTTA yield lower ∆P and ∆G than supervised approaches Packnet, Piggyback, and CRoP which is in-line with literature Varma and Prasad [2023]. Psychophysiological stress response is inherently heterogeneous in inter- and intra-user scenarios Nagaraj et al. [2023], leading to subpar performance of the generic model without personalization. However, personalization attains a significant efficacy boost evident from the high ∆P values. These evaluations confirm that models personalized with CRoP exhibit higher generalizability to unseen con-
texts, making them more intra-user robust.
# Empirical Justification for CRoP
This section empirically discusses how each step of CRoP (Algorithm 1) facilitates intra-user generalizability signifying similarity in model’s behavior towards available (available during personalization finetuning) and unseen contexts. Shi et al. [2021] introduced the use of gradient inner product (GIP) to estimate the similarity between a model’s behavior across different domains. If Gi and Gj represent the gradient incurred by the model for Domains Di and Dj, then the sign of the product Gi ∗Gj represents whether the model treats two domains similarly or not. For instance, Gi ∗Gj > 0 signifies that the gradient for both domains has the same direction. We used GIP to quantify generalization. A higher GIP value for a personalized model across available and unseen contexts indicates more similar behavior toward both domains. GIP is measured as: ∥� i Gi∥2 −� i ∥Gi∥2. Figure 3 shows that fine-tuning the generic model (Algorithm 1 – Step 2) on Context 1, optimizes the model for this context, leading to a highly negative GIP, indicating a greater discrepancy between two contexts. Since model pruning results in generalization [Jin et al., 2022], an increase in GIP value can be observed in the pruned model (Step 3). On further analysis, we found that the model complementary to the pruned model (that is, the parameters that were removed) also contributed towards inter-context behavior discrepancy (negative GIP value). However, the same parameters in the generic model (that are replaced in Step 4) formed a more generalizable set of weights, i.e., GIP ≥0. Thus, the model mixing step (Step 4) introduces further generalizability (GIP ≥0) in the personalized model.
Approach
SHOT
Packnet
Piggyback
CoTTA
CRoP
Dataset
Scenrio
∆P
∆G
∆P
∆G
∆P
∆G
∆P
∆G
∆P
∆G
PERCEPT-R
Scenario 1
-3.11
-5.62
0.10
-2.41
-25.31
-27.83
-45.06
-47.58
5.08
2.57
Stress Sensing
Scenario 1
-8.19
-62.16
54.70
0.70
43.89
-10.12
21.93
-32.07
67.81
13.81
Single context Change
Scenario 2
8.90
-63.27
75.80
3.64
66.22
-5.94
51.47
-20.69
85.25
13.08
Stress Sensing
Scenario 1
-0.49
-47.24
52.46
10.08
32.40
-9.97
30.59
-11.78
54.38
12.00
Double context Change
Scenario 2
3.57
-45.49
41.68
-7.36
42.76
-6.25
33.85
-15.19
59.21
10.15
WIDAR
Scenario 1
1.67
-0.48
-0.24
-2.37
0.84
-1.28
-1.05
-3.18
8.56
6.43
Scenario 2
1.28
-0.03
-3.55
-5.16
-8.97
-10.57
1.81
0.21
5.90
4.30
ExtraSensory
Scenario 1
7.63
-10.31
12.19
-5.76
5.03
-12.91
-0.6
-18.54
17.49
-0.46
Scenario 2
8.17
2.99
1.33
-3.85
5.22
0.04
-3.43
-8.62
13.52
8.17
Table 3: Comparison of CRoP with baseline approaches under the metrics of Personalization (∆P ) and G (∆G).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6419/6419ec5f-f943-4486-8662-accdd948423f.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Variation of GIP at different stages of CRoP signified by the lines in Algorithm 1</div>
Figure 3: Variation of GIP at different stages of CRoP signified by the lines in Algorithm 1
# Ablation Study
This section presents evaluations showing the effectiveness of the design choices of CRoP, focusing on the WIDAR dataset in Scenario 1. Similar patterns were observed in other scenarios and datasets.
# One shot Magnitude based pruning as the pruning mechanism
mechanism
A variety of pruning mechanism have been proposed in the literature: Magnitude-Based Pruning (MP) [Luo et al., 2017, Zhu and Gupta, 2017], Gradient-Based Pruning (GP)[Liu et al., 2020], pruning top magnitude weights instead of lower ones (MP-T)[Bartoldson et al., 2020], iterative pruning (MP-I)[Paganini and Forde, 2020], and more [Hoefler et al., 2021]. Among these, we found that oneshot magnitude based pruning serves the best purpose for the application discussed in this work. Detailed discussion is provided in Appendix F.1.
# Weight penalty via ℓ1 regularization
The use of regularization forces model parameters towards zero. We observed that among ℓ0, ℓ1, ℓ2 and polarization [Zhuang et al., 2020], using ℓ1 regularization is most effective in the unseen context. More details can be found in Appendix F.2.
# Limitations and Future Direction
Some limitations and future research are discussed below:
1. The paper performed a limited evaluation on pruning paradigms through ablation studies as it was not the primary focus of the study. Section on ablation study justifies CRoP’s design choice but does not establish any particular paradigm’s superiority in unseen contexts.
2. The approach relies on using a pre-trained off-theshelf model as an input, the quality of this model can impact the performance of the final personalized models.
2. The approach relies on using a pre-trained off-theshelf model as an input, the quality of this model can impact the performance of the final personalized models.
3. We restrict our study to the models benchmarked and deployed for datasets used in this work without accounting for model variability.
4. The metrics ∆P and ∆G are computed for each individual separately as personalized models customized for one user are not applicable to other users in realworld scenarios. Consequently, we focus our evaluations on intra-user generalizability, excluding discussion for inter-user or inter-dataset generalizability.
4. The metrics ∆P and ∆G are computed for each individual separately as personalized models customized for one user are not applicable to other users in realworld scenarios. Consequently, we focus our evaluations on intra-user generalizability, excluding discussion for inter-user or inter-dataset generalizability.
5. For PERCEPT-R, CRoP shows varying advantages among individuals, linked to the generic model’s sensitivity to pre-treatment data (see Appendix B.3). However, determining which individuals will experience more or less benefit from personalization is beyond this paper’s scope and will be explored in future research.
# Broader Impact
This paper addresses a critical research gap, enhancing the practical utility of human-sensing solutions in realworld applications, particularly in automated healthcare. Next-generation healthcare systems, which employ neural networks for tasks ranging from daily activity detection [Ustev et al., 2013, Stisen et al., 2015] to safety-critical conditions like atrial fibrillation [Comstock, 2017], benefit from personalization due to the heterogeneity in health sensing data [Ji et al., 2021, Sempionatto et al., 2021]. CRoP offers several advantages:
# Conclusion
This study introduces CRoP, a novel static personalization approach generating context-wise robust mod-
els from limited context data. Using pruning to balance personalization and generalization, empirical analysis on four human-sensing datasets shows CRoP models exhibit an average increase of 35.23% in personalization compared to generic models and 7.78% in generalization compared to conventionally-finetuned personalized models. CRoP utilizes off-the-shelf models, reducing training effort and addressing privacy concerns. With practical benefits and quantitative performance enhancements, CRoP facilitates reliable real-world deployment for AI-based human-sensing applications like healthcare.
# References
Farhad Ahamed and Farnaz Farid. Applying internet of things and machine-learning for personalized healthcare: Issues and challenges. In 2018 International Conference on Machine Learning and Data Engineering (iCMLDE), pages 19–21. IEEE, 2018.
Apple. Workout types on apple watch. https://support. apple.com/en-us/HT207934, 2023.
Ayo Awobajo. 3 tips to make google assistant your own. https://blog.google/products/assistant/how-topersonalize-google-assistant/, 2023.
Brian R. Bartoldson, Ari S. Morcos, Adrian Barbu, and Gordon Erlebacher. The generalization-stability tradeoff in neural network pruning, 2020.
N. R. Benway and J. L. Preston. Artificial intelligence assisted speech therapy for / r / using speech motor chaining and the percept engine: a single case experimental clinical trial with chainingai., 2023. URL https: //surface.syr.edu/etd/1703.
Nina R Benway, Jonathan L Preston, Elaine Hitchcock, Yvan Rose, Asif Salekin, Wendy Liang, and Tara McAllister. Reproducible speech research with the artificial intelligence-ready PERCEPT corpora. J. Speech Lang. Hear. Res., 66(6):1986–2009, June 2023.
Visar Berisha, Chelsea Krantsevich, P Richard Hahn, Shira Hahn, Gautam Dasarathy, Pavan Turaga, and Julie Liss. Digital medicine and the curse of dimensionality. NPJ Digit. Med., 4(1):153, October 2021.
Mehdi Boukhechba, Anna N Baglione, and Laura E Barnes. Leveraging mobile sensing and machine learning for personalized mental health care. Ergonomics in design, 28(4):18–23, 2020. David Burns, Philip Boyer, Colin Arrowsmith, and Cari Whyne. Personalized activity recognition with deep triplet embeddings. Sensors, 22(14), 2022. ISSN 14248220. doi: 10.3390/s22145222. URL https://www. mdpi.com/1424-8220/22/14/5222. Jonah Comstock. Study: Apple watch paired with deep neural network detects atrial fibrillation with 97 percent accuracy, 2017. Zachary A. Daniels, Jun Hu, Michael Lomnitz, Phil Miller, Aswin Raghavan, Joe Zhang, Michael Piacentino, and David Zhang. Efficient model adaptation for continual learning at the edge, 2023. Di Duan, Huanqi Yang, Guohao Lan, Tianxing Li, Xiaohua Jia, and Weitao Xu. Emgsense: A low-effort selfsupervised domain adaptation framework for emg sensing. In 2023 IEEE International Conference on Pervasive Computing and Communications (PerCom), pages 160–170, 2023. doi: 10.1109/PERCOM56429.2023. 10099164. Maciej Dzie˙zyc, Martin Gjoreski, Przemysław Kazienko, Stanisław Saganowski, and Matjaˇz Gams. Can we ditch feature engineering? end-to-end deep learning for affect recognition from physiological sensor data. Sensors, 20(22):6535, 2020. empetica. Real-time physiological signals: E4 eda/gsr sensor, 2015. URL https://www.empatica.com/ research/e4/. Eda Eren and Tu˘gba Selcen Navruz. Stress detection with deep learning using bvp and eda signals. In 2022 International Congress on Human-Computer Interaction, Optimization and Robotic Applications (HORA), pages 1–7. IEEE, 2022. Enrico Fini, Victor G Turrisi da Costa, Xavier AlamedaPineda, Elisa Ricci, Karteek Alahari, and Julien Mairal. Self-supervised models are continual learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.
empetica. Real-time physiological signals: E4 eda/gsr sensor, 2015. URL https://www.empatica.com/ research/e4/.
Eda Eren and Tu˘gba Selcen Navruz. Stress detection with deep learning using bvp and eda signals. In 2022 International Congress on Human-Computer Interaction, Optimization and Robotic Applications (HORA), pages 1–7. IEEE, 2022.
Enrico Fini, Victor G Turrisi da Costa, Xavier AlamedaPineda, Elisa Ricci, Karteek Alahari, and Julien Mairal. Self-supervised models are continual learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.
US Food and Drug Administration. Proposed regulatory framework for mondifications to artificial intelligence / machine learning-based software as a medical device. US Food and Drug Administration: Silver Spring, MD, USA, 63, 2019. doi: 10.1016/j.apergo.2017.04.011.
aesik Gong, Yeonsu Kim, Jinwoo Shin, and SungJu Lee. Metasense: few-shot adaptation to untrained conditions in deep mobile sensing. In Proceedings of the 17th Conference on Embedded Networked Sensor Systems, SenSys ’19, page 110–123, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450369503. doi: 10. 1145/3356250.3360020. URL https://doi.org/10.1145/ 3356250.3360020.
Taesik Gong, Yewon Kim, Adiba Orzikulova, Yunxin Liu, Sung Ju Hwang, Jinwoo Shin, and Sung-Ju Lee. Dapper: Label-free performance estimation after personalization for heterogeneous mobile sensing. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 7(2):1–27, 2023.
Sachin Goyal, Ananya Kumar, Sankalp Garg, Zico Kolter, and Aditi Raghunathan. Finetune like you pretrain: Improved finetuning of zero-shot vision models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19338– 19347, June 2023.
Yujiao Hao, Rong Zheng, and Boyu Wang. Invariant feature learning for sensor-based human activity recognition. IEEE Transactions on Mobile Computing, 21(11): 4013–4024, 2022. doi: 10.1109/TMC.2021.3064252.
Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. J. Mach. Learn. Res., 22(1), jan 2021. ISSN 1532-4435.
in-Hyuk Hong, Julian Ramos, and Anind K. Dey. Toward personalized activity recognition systems with a semipopulation approach. IEEE Transactions on Human-Machine Systems, 46(1):101–112, 2016. doi: 10.1109/THMS.2015.2489688.
Jin-Hyuk Hong, Julian Ramos, and Anind K. Dey. Toward personalized activity recognition systems with a semipopulation approach. IEEE Transactions on Human-Machine Systems, 46(1):101–112, 2016. doi: 10.1109/THMS.2015.2489688.
Andrea Iaboni, Sofija Spasojevic, Kristine Newman, Lori Schindel Martin, Angel Wang, Bing Ye, Alex Mihailidis, and Shehroz S Khan. Wearable multimodal sensors for the detection of behavioral and psychological symptoms of dementia using personalized machine learning models. Alzheimer’s & Dementia: Diagnosis, Assessment & Disease Monitoring, 14(1):e12305, 2022.
Stanislaw Jastrzebski, Devansh Arpit, Oliver Astrand, Giancarlo B Kerg, Huan Wang, Caiming Xiong, Richard Socher, Kyunghyun Cho, and Krzysztof J Geras. Catastrophic fisher explosion: Early phase fisher matrix impacts generalization. In International Conference on Machine Learning, pages 4772–4784. PMLR, 2021.
Wenhui Ji, Jingyu Zhu, Wanxia Wu, Nanxiang Wang, Jiqing Wang, Jiansheng Wu, Qiong Wu, Xuewen Wang, Changmin Yu, Gaofeng Wei, et al. Wearable sweat biosensors refresh personalized health/medical diagnostics. Research, 2021.
Tian Jin, Michael Carbin, Daniel M. Roy, Jonathan Frankle, and Gintare Karolina Dziugaite. Pruning’s effect on generalization through the lens of training and regularization. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=OrcLKV9sKWp.
Minyoung Kim, Da Li, Shell X Hu, and Timothy Hospedales. Fisher sam: Information geometry and sharpness aware minimisation. In International Conference on Machine Learning, pages 11148–11161. PMLR, 2022.
Daniel B Kowalsky, John R Rebula, Lauro V Ojeda, Peter G Adamczyk, and Arthur D Kuo. Human walking in the real world: Interactions between terrain type, gait parameters, and energy expenditure. PLoS One, 16(1): e0228682, January 2021.
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, Toronto, Ontario, 2009. URL https://www.cs.toronto.edu/∼kriz/learningfeatures-2009-TR.pdf.
Bishal Lamichhane, Joanne Zhou, and Akane Sano. Psychotic relapse prediction in schizophrenia patients using a personalized mobile sensing-based supervised deep learning model. IEEE Journal of Biomedical and Health Informatics, 2023.
Barbara A Lewis, Lisa Freebairn, Jessica Tag, Allison A Ciesla, Sudha K Iyengar, Catherine M Stein, and H Gerry Taylor. Adolescent outcomes of children with early speech sound disorders with and without language impairment. Am. J. Speech. Lang. Pathol., 24(2): 150–163, May 2015.
Jian Liang, Dapeng Hu, and Jiashi Feng. Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation. In International Conference on Machine Learning (ICML), pages 6028–6039, 2020.
Hanbing Liu, Jingge Wang, Xuan Zhang, Ye Guo, and Yang Li. Enhancing continuous domain adaptation with multi-path transfer curriculum, 2024.
Xin Liu, Yuntao Wang, Sinan Xie, Xiaoyu Zhang, Zixian Ma, Daniel McDuff, and Shwetak Patel. Mobilephys: Personalized mobile camera-based contactless physiological sensing. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., 6(1), mar 2022. doi: 10.1145/ 3517225. URL https://doi.org/10.1145/3517225.
Xue Liu, Weijie Xia, and Zhimiao Fan. A deep neural network pruning method based on gradient l1-norm. In 2020 IEEE 6th International Conference on Computer and Communications (ICCC), pages 2070–2074, 2020. doi: 10.1109/ICCC51575.2020.9345039.
Jian-Hao Luo, Jianxin Wu, and Weiyao Lin. Thinet: A filter level pruning method for deep neural network compression. In ICCV, pages 5058–5066, 2017.
Bradley Malin, Kenneth Goodman, et al. Between access and privacy: challenges in sharing health data. Yearbook of medical informatics, 27(01):055–059, 2018.
Arun Mallya and Svetlana Lazebnik. Packnet: Adding multiple tasks to a single network by iterative pruning. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7765–7773, 2018. doi: 10.1109/CVPR.2018.00810.
Arun Mallya, Dillon Davis, and Svetlana Lazebnik. Piggyback: Adapting a single network to multiple tasks by learning to mask weights. In Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss, editors, Computer Vision – ECCV 2018, pages 72–88, Cham, 2018. Springer International Publishing. ISBN 978-3030-01225-0.
James Martens. New insights and perspectives on the natural gradient method. Journal of Machine Learning Research, 21(146):1–76, 2020.
K. Mayank. Bxd primer series: Lasso regression models, l1 regularization in general and comparison with l2 regularization, 2023. URL https://www.linkedin.com/pulse/bxd-primer-serieslasso-regression-models-l1-general-comparison-k-/.
Lakmal Meegahapola, William Droz, Peter Kun, Amalia De G¨otzen, Chaitanya Nutakki, Shyam Diwakar, Salvador Ruiz Correa, Donglei Song, Hao Xu, and Miriam Bidoglia. Generalization and personalization of mobile sensing-based mood inference models: An analysis of college students in eight countries. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 6(4):1–32, 2023.
Sujay Nagaraj, Sarah Goodday, Thomas Hartvigsen, Adrien Boch, Kopal Garg, Sindhu Gowda, Luca Foschini, Marzyeh Ghassemi, Stephen Friend, and Anna Goldenberg. Dissecting the heterogeneity of “in the wild” stress from multimodal sensor data. NPJ Digital Medicine, 6(1):237, 2023.
Laura P¨aeske, Tuuli Uudeberg, Hiie Hinrikus, Jaanus Lass, and Maie Bachmann. Correlation between electroencephalographic markers in the healthy brain. Sci. Rep., 13(1):6307, April 2023.
Michela Paganini and Jessica Forde. On iterative neural network pruning, reinitialization, and the similarity of masks, 2020.
Wonil Park, Victor J. Lee, Byungmo Ku, and Hirofumi Tanaka. Effect of walking speed and placement position interactions in determining the accuracy of various newer pedometers. Journal of Exercise Science & Fitness, 12(1):31–37, 2014. ISSN
1728-869X. doi: https://doi.org/10.1016/j.jesf.2014. 01.003. URL https://www.sciencedirect.com/science/ article/pii/S1728869X14000057. David Phelan. Amazon admits listening to alexa conversations: Why it matters. https://shorturl.at/fxN78, 2019. Michael Potuck. How to reset your apple watch fitness calibration for more accurate workout and activity data. https://9to5mac.com/2021/08/26/fix-applewatch-workout-tracking-activity-tracking/, 2021.
1728-869X. doi: https://doi.org/10.1016/j.jesf.2014. 01.003. URL https://www.sciencedirect.com/science/ article/pii/S1728869X14000057.
Michael Potuck. How to reset your apple watch fitness calibration for more accurate workout and activity data. https://9to5mac.com/2021/08/26/fix-applewatch-workout-tracking-activity-tracking/, 2021.
A. Prabhu, H. Al Kader Hammoud, P. Dokania, P. S. Torr, S. Lim, B. Ghanem, and A. Bibi. Computationally budgeted continual learning: What does matter? In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3698–3707, Los Alamitos, CA, USA, jun 2023. IEEE Computer Society. doi: 10.1109/CVPR52729.2023. 00360. URL https://doi.ieeecomputersociety.org/10. 1109/CVPR52729.2023.00360.
Amy Rathbone, Simone Stumpf, Caroline Claisse, Elizabeth Sillence, Lynne Coventry, Richard D Brown, and Abigail C Durrant. People with long-term conditions sharing personal health data via digital health technologies: A scoping review to inform design. PLOS Digit. Health, 2(5):e0000264, May 2023.
Sadiq Sani, Stewart Massie, Nirmalie Wiratunga, and Kay Cooper. Learning deep and shallow features for human activity recognition. In Gang Li, Yong Ge, Zili Zhang, Zhi Jin, and Michael Blumenstein, editors, Knowledge Science, Engineering and Management, pages 469– 482, Cham, 2017. Springer International Publishing. ISBN 978-3-319-63558-3.
hilip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and Kristof Van Laerhoven. Introducing wesad, a multimodal dataset for wearable stress and affect detection. In Proceedings of the 20th ACM International Conference on Multimodal Interaction, ICMI ’18, page 400–408, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450356923. doi: 10.1145/3242969.3242985. URL https://doi.org/10.1145/3242969.3242985.
Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun 2015. doi: 10.1109/cvpr.2015.7298682. URL http: //dx.doi.org/10.1109/CVPR.2015.7298682.
Juliane R Sempionatto, Victor Ruiz-Valdepenas Montiel, Eva Vargas, Hazhir Teymourian, and Joseph Wang. Wearable and mobile sensors for personalized nutrition. ACS sensors, 6(5):1745–1760, 2021.
Qiang Shen, Haotian Feng, Rui Song, Stefano Teso, Fausto Giunchiglia, and Hao Xu. Federated multi-task attention for cross-individual human activity recognition. In Lud De Raedt, editor, Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, pages 3423–3429. International Joint Conferences on Artificial Intelligence Organization, 7 2022. doi: 10.24963/ijcai.2022/475. URL https://doi.org/10.24963/ijcai.2022/475. Main Track.
Yuge Shi, Jeffrey Seely, Philip H. S. Torr, N. Siddharth, Awni Hannun, Nicolas Usunier, and Gabriel Synnaeve. Gradient matching for domain generalization. 2021.
Leslie N Smith. Cyclical learning rates for training neural networks. In 2017 IEEE winter conference on applications of computer vision (WACV), pages 464–472. IEEE, 2017.
Gabriela M Stegmann, Shira Hahn, Julie Liss, Jeremy Shefner, Seward Rutkove, Kerisa Shelton, Cayla Jessica Duncan, and Visar Berisha. Early detection and tracking of bulbar changes in ALS via frequent and remote speech analysis. NPJ Digit. Med., 3(1):132, October 2020.
Allan Stisen, Henrik Blunck, Sourav Bhattacharya, Thor Siiger Prentow, Mikkel Baun Kjærgaard, Anind Dey, Tobias Sonne, and Mads Møller Jensen. Smart devices are different: Assessing and mitigatingmobile sensing heterogeneities for activity recognition. In Proceedings of the 13th ACM conference on embedded networked sensor systems, pages 127–140, 2015.
. Tang, L. Qendro, D. Spathis, F. Kawsar, C. Mascolo, and A. Mathur. Kaizen: Practical selfsupervised continual learning with continual finetuning. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2829– 2838, Los Alamitos, CA, USA, jan 2024a. IEEE Computer Society. doi: 10.1109/WACV57701.2024. 00282. URL https://doi.ieeecomputersociety.org/10. 1109/WACV57701.2024.00282.
Chi Ian Tang, Lorena Qendro, Dimitris Spathis, Fahim Kawsar, Akhil Mathur, and Cecilia Mascolo. Balancing continual learning and fine-tuning for human activity recognition. ArXiv, abs/2401.02255, 2024b. URL https://api.semanticscholar.org/CorpusID:266755926.
Siri Team. Hey siri: An on-device dnn-powered voice trigger for apple’s personal assistant. https:// machinelearning.apple.com/research/hey-siri, 2017.
Yunus Emre Ustev, Ozlem Durmaz Incel, and Cem Ersoy. User, device and orientation independent human activity recognition on mobile phones: Challenges and a proposal. In Proceedings of the 2013 ACM conference on Pervasive and ubiquitous computing adjunct publication, pages 1427–1436, 2013.
Yonatan Vaizman, Katherine Ellis, and Gert Lanckriet. Recognizing detailed human context in the wild from smartphones and smartwatches. IEEE Pervasive Computing, 16(4):62–74, 2017. doi: 10.1109/MPRV.2017. 3971131.
# C. Varma and Puja Prasad. Supervised and unsupervised machine learning approaches—a survey, 02 2023.
Chan Wang, Tianyiyi He, Hong Zhou, Zixuan Zhang, and Chengkuo Lee. Artificial intelligence enhanced sensors - enabling technologies to next-generation healthcare and biomedical platform. Bioelectron. Med., 9(1):17, August 2023.
Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai. Continual test-time domain adaptation, 2022a.
Qin Wang, Olga Fink, Luc Van Gool, and Dengxin Dai. Continual test-time domain adaptation. In Proceedings of Conference on Computer Vision and Pattern Recognition, 2022b.
Zhiguang Wang, Weizhong Yan, and Tim Oates. Time series classification from scratch with deep neural networks: A strong baseline. In 2017 International joint conference on neural networks (IJCNN), pages 1578– 1585. IEEE, 2017.
Zhiguang Wang, Weizhong Yan, and Tim Oates. Time series classification from scratch with deep neural networks: A strong baseline. In 2017 International joint conference on neural networks (IJCNN), pages 1578– 1585. IEEE, 2017.
Yanan Wu, Zhixiang Chi, Yang Wang, Konstantinos N. Plataniotis, and Songhe Feng. Test-time domain adaptation by learning domain-aware batch normalization, 2024.
Yi Xiao, Harshit Sharma, Zhongyang Zhang, Dessa Bergen-Cico, Tauhidur Rahman, and Asif Salekin. Reading between the heat: Co-teaching body thermal signatures for non-intrusive stress detection. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., 7 (4), jan 2024. doi: 10.1145/3631441. URL https: //doi.org/10.1145/3631441.
Jianfei Yang, Xinyan Chen, Dazhuo Wang, Han Zou, Chris Xiaoxuan Lu, Sumei Sun, and Lihua Xie. Sensefi: A library and benchmark on deep-learningempowered wifi human sensing, 2023.
Shuochao Yao, Shaohan Hu, Yiran Zhao, Aston Zhang, and Tarek Abdelzaher. Deepsense: A unified deep learning framework for time-series mobile sensing data processing. In Proceedings of the 26th International Conference on World Wide Web, WWW ’17, page 351–360, Republic and Canton of Geneva, CHE, 2017. International World Wide Web Conferences Steering Committee. ISBN 9781450349130. doi: 10.1145/3038912.3052577. URL https://doi.org/ 10.1145/3038912.3052577.
Ben Zandonati, Adrian Alan Pol, Maurizio Pierini, Olya Sirkin, and Tal Kopetz. Fit: A metric for model sensitivity. arXiv preprint arXiv:2210.08502, 2022.
Ben Zandonati, Adrian Alan Pol, Maurizio Pierini, Olya Sirkin, and Tal Kopetz. Fit: A metric for model sensitivity. arXiv preprint arXiv:2210.08502, 2022.
Y. Zhang, Y. Zheng, K. Qian, G. Zhang, Y. Liu, C. Wu, and Z. Yang. Widar3.0: Zero-effort cross-domain gesture recognition with wi-fi. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(11):8671– 8688, nov 2022. ISSN 1939-3539. doi: 10.1109/ TPAMI.2021.3105387.
Michael Zhu and Suyog Gupta. To prune, or not to prune: exploring the efficacy of pruning for model compression, 2017.
ao Zhuang, Zhixuan Zhang, Yuheng Huang, Xiaoyi Zeng, Kai Shuang, and Xiang Li. Neuron-level structured pruning using polarization regularizer. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 9865–9877. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper files/paper/2020/ file/703957b6dd9e3a7980e040bee50ded65-Paper.pdf.
# Experiment Details
This section describes the details of the empirical analysi done for the presented approach.
# Datasets and models
For the evaluations we used four datasets: PERCEPT-R [Benway et al., 2023], WIDAR [Zhang et al., 2022], ExtraSensory [Vaizman et al., 2017] and a Stress-sensing dataset [Xiao et al., 2024]. PERCEPT-R, WIDAR and Stress-sensing dataset are collected in a controlled lab setting whereas ExtraSensory is collected in a real-world setting.
PERCEPT-R: The sound / r / has been recognized as the most frequently impacted sound in residual speech sound disorders in American English Lewis et al. [2015] and considered to be the most difficult sound to treat. The PERCEPT-R Corpus was collected during 34 different cross-sectional and longitudinal studies of speech Benway et al. [2023] for automated speech analysis of / r /. The data used in this study come from the prospectively collected Benway and Preston [2023], and corpus version
2.2.2, which includes both the publicly available open access subset (2.2.2p) and privately held data that was not published in the open access subset after a review of consent/assent permissions. Items in the PERCEPT-R Corpus v2.2.2 primarily consist of single-word citation speech audio collected during clinical trials involving children with speech sound disorders affecting / r /, along with agematched peers with typical speech. The full corpus contains 179,076 labeled utterances representing 662 singlerhotic words and phrases. Each audio file is paired with a ground-truth label representing listener judgments of rhoticity, derived by averaging binary ratings (0 = derhotic, 1 = fully rhotic) from multiple listeners. For this study, the heuristic threshold for converting these averaged ratings into binary ground-truth labels was 0.66. The context is defined by the treatment phase, that is, whether the treatment has started or not. In line with state of the art Benway et al. [2023], we tried several model architectures such as CNN, DNN, BILSTM etc whose number of parameters were identified using grid search. Among those, the biLSTM model containing 4 bidirectional LSTM layers followed by 5 linear layers, which are accompanied by a Hardswish activation layer, was identified as the one exhibit best results for the generic data and was used for this study.
WIDAR: WIDAR is a dataset collected for the purpose of gesture recognition. It was collected using off-the-shelf WiFi links (one transmitter and at least 3 receivers). 17 users performed 15 different gestures at 15 different locations in 3 rooms for 5 different orientations (of the person). The channel state information is collected from these devices with amplitude noises and phase offsets removed as a preprocessing step. The two contexts used for the current work are decided based on the orientations of the torso data and room ID. Room 1 is a classroom with a number of objects (e.g., desks and chairs) in it, and Room 2 is a nearly empty hallway. The dissimilar data distributions can be attributed to the differences in the amount of interruptions in WiFi signals. We followed the same normalization methods as Yang et al. [2023]. The model used for WIDAR follows the LeNet architecture [Zhang et al., 2022], which contains three 2D convolutional layers followed by two linear layers. Each of these layers, except the final classification layer, is fol-
lowed by a ReLU activation layer.
ExtraSensory: ExtraSensory is a human activity recognition dataset collected using the ExtraSensory mobile application. A number of features were collected from different cellular devices and smart watches, though we just used the accelerometer and gyroscope features obtained from the cellular devices. Labels for activities were selfreported by the users through the mobile application. For our evaluations on ExtraSensory, 5 users were left out for training a single generic model. The contexts are decided based on the location of the phone: hand, pocket, and bag. The model follows a CNN-GRU-based architecture used in HAR literature [Gong et al., 2019, Hao et al., 2022, Shen et al., 2022, Yao et al., 2017]. The model consists of three batch-normalized 1D convolution layers followed by a linear layer that feeds into a batch-normalized recursive (GRU) layer and two linear layers to generate embeddings. For the classification head, two linear layers were used.
Stress Sensing Dataset: This dataset measures the physiological impacts of various kinds of Stress. The dataset is collected using Empatica E4 Wristband to extract features such as EDA (Electrodermal Activity), a skin temperature sensor (4 Hz), etc, contributing to a total of 34 features. The data is collected from 30 participants having different demographics and were assigned the labels as ‘Stressed’ or ‘Calm’ based in the current physiological features values. The context is defined by combination of hand on which the wristband was worn and whether or not the person was moving during the data collection. The model uses a simple multi-layer-perceptron (MLP) architecture Eren and Navruz [2022], Dzie˙zyc et al. [2020], Wang et al. [2017] consisting of 3 linear layers with hidden size of 128.
# Training of the Generic models
WIDAR: We chose users 0,1 and 2 for personalization since these were the only users whose data was collected in both rooms. Since the number of users in WIDAR is the very small, the exclusion of all the 3 users for training the generic model would have resulted in substandard
models. So, For each user, we generated different generic models by using data from the other 16 users with a 14/2 person disjoint random split for the train and validation set. Our classification target was the 6 gesture classes: 0,1,2,3,5 and 8.
ExtraSensory: We chose users 61, 7C, 80, 9D, and B7 for personalization, and the generic model is trained on 42 users, with 10 users being left out to validate. The two target classes are walking and sitting.
Stress Sensing Dataset: This is a binary classification problem where we chose users 1, 2, and 3 for personalization as these users contributed data in all possible contexts. For generic model training, 6 other users were used to create person disjoint validation and test sets, and the remaining 21 users were used to train the generic model.
Stress Sensing Dataset: This is a binary classification problem where we chose users 1, 2, and 3 for personalization as these users contributed data in all possible contexts. For generic model training, 6 other users were used to create person disjoint validation and test sets, and the remaining 21 users were used to train the generic model. PERCEPT-R : As recommended by clinical experts, we choose 16 participants with ids 17, 25, 28, 336, 344, 361, 362, 55, 586, 587, 589, 590, 591, 61, 67, and 80 for personalization. These participants’ speech data were collected longitudinally, meaning their data could be separated into available and unseen contexts versus other speakers in the corpus who only had speech data available from one time-point. The generic model is trained for the remaining 499 participants using person-disjoint validation and test sets. The aim of this dataset is to identify the correctness of / r / sounds.
PERCEPT-R : As recommended by clinical experts, we choose 16 participants with ids 17, 25, 28, 336, 344, 361, 362, 55, 586, 587, 589, 590, 591, 61, 67, and 80 for personalization. These participants’ speech data were collected longitudinally, meaning their data could be separated into available and unseen contexts versus other speakers in the corpus who only had speech data available from one time-point. The generic model is trained for the remaining 499 participants using person-disjoint validation and test sets. The aim of this dataset is to identify the correctness of / r / sounds.
# Metrics for classification accuracy evaluation
We use accuracy to measure the performance of a model. However, the computation of this metric differs for the four datasets. The details of the metrics used for all the datasets are as follows:
1. WIDAR: We use a 6-class classification for gesture recognition, and the distribution of the data among these classes is nearly balanced. Thus, standard classification accuracy has been used for WIDAR.
2. ExtraSensory: The subset of the Extrasensory dataset used for this work aims for a binary classification for activity recognition. We observed that the data distribution was quite imbalanced among the two classes,
Hyperparameter
PERCEPT
WIDAR
ExtraSensory
Stress-sensing
Base Learning Rate
1e-5
1e-07
1.2e-08
5e-5
Max Learning Rate
1e-5
5e-06
7.5e-07
5e-5
Epochs
300
1000
150
1000
<div style="text-align: center;">Table 4: Hyperparameters for generic Models</div>
# Table 4: Hyperparameters for generic Models
and therefore, balanced classification accuracy was used for this dataset. Balanced accuracy is computed as the average of true positive rate and true negative rate.
3. Stress Sensing Dataset: For this binary classification problem, F1 score has been used as a performance metric as suggested by the original authors Xiao et al. [2024].
3. Stress Sensing Dataset: For this binary classification problem, F1 score has been used as a performance metric as suggested by the original authors Xiao et al. [2024].
4. PERCEPT-R: For this dataset, Benway et al. [2023] utilized balanced accuracy for the binary classification task, and we employed the same metrics in our study.
For simplicity, we use the term ‘accuracy’ to encompass all the metrics discussed above.
# Hyperparameters
The approach uses several hyperparameters for generic model training and personalization. Table 4 and 5 show the hyperparameter values for generic and personalized model training, respectively. These values correspond to the best results obtained using a grid search. For training the generic model, in addition to the number of epochs, ‘Base Learning Rate’ and ‘Max Learning Rate’ (the arguments for CycleLR [Smith, 2017]) are the hyperparameters. For the personalized model, learning rate (fixed), α, τ, number of epochs for initial finetuning (Initial Epochs), and epochs for final finetuning (Final Epochs) are the hyperparameters. The range of these hyperparameters used for grid search during personalization is also mentioned in Table 5. Additionally, we use k = k′ = 0.05 for the ToleratedPrune module for PERCEPT-R, WIDAR, and ExtraSensory datasets, while for the Stress-sensing dataset, k = 0.05 and k′ = 0.01 is being used. One may find different values to be suitable for other datasets and model architectures.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0fb6/0fb6dd84-289f-4fdc-b60d-67602c7c4ac0.png" style="width: 50%;"></div>
Hyperparameter
Range
PERCEPT-R
WIDAR
ExtraSensory
Stress Sensing
Learning Rate
1e-6 - 1e-1
1e-5
1e-6
1e-6
1e-5
alpha
1e-6 - 10
0.01
0.0001
0.5
0.0001
τ
0.01 - 0.25
0.05
0.2
0.01
0.01
Initial Epochs
100 -1000
300
600
600
1000
Final Epochs
100 - 1000
300
600
1000
1000
<div style="text-align: center;">Table 5: Hyperparameters for Personalized Models</div>
# Code
The code is provided in supplementary material arranged into dataset-specific folders. Each folder contains the pretrained generic model, all the required modules, and the instructions to run the code. The seed values used for the evaluations are also provided in the shell files. The data partitioned into personalized and context-wise sets will be released upon publication.
# Compute Resources
All the computations have been performed on NVIDIA Quadro RTX 5000.
# All the computations have been performed on NVIDIA Quadro RTX 5000.
# Detailed Results
# Comparison with Generic Models
The personalized models obtained using CRoP exhibit higher classification accuracy than the generic models on the available context’s data Da i , showcasing the benefits of personalization. To demonstrate the existence of such improvement, Tables 6a- 6h and Table 9a compare the performance of generic model MG θ and personalized models obtained using CRoP MP a i θ .
WIDAR: Tables 6a and 6b show that there is an average improvement of 25.25 and 11.88 percent points among three users for the available context Ca for Scenario 1 and Scenario 2, respectively. However, this benefit comes at the cost of a reduction in accuracy for the unseen context. There is an average reduction of 16.69 and 5.97 percent points for Scenario 1 and Scenario 2, respectively, for the unseen context Cu. Notably, the loss of accuracy in the unseen context is much lower as compared to the
conventionally-finetune model as discussed in the Motivation Section.
ExtraSensory: Similar patterns could be observed for the Extrasensory dataset. Tables 6c and 6d show that there is an average increment of 16.40 and 18.37 percent points for the available context for Scenario 1 and Scenario 2, respectively. Interestingly, the performance of the personalized model for Scenario 1 on unseen context Cu was not adversely impacted. This is attributed to the fact that the inertial sensing patterns of Bag and Pocket phone carrying modes capture the user’s body movement, whereas the phone-in-hand movement patterns can be distinct. In Scenario 1, Ca comprises pocket and Cu comprises bag, meaning both available and unseen contexts encompass similar inertial patterns, leading to advantageous performance even in the unseen context. This evaluation illustrates minimal intra-user generalizability loss on unseen contexts when both available and unseen contexts share similar user traits. However, in Scenario 2, where only the hand belongs to the unseen context Cu, there is an average loss of 5.02 percentage points on the unseen context.
Stress Sensing: The physiological features used in this dataset vary significantly from one user to other. Thus, Tables 6e-6h show that the generic models do not perform well on personalized data. Personalized finetuning enables the model to learn person-specific patterns, allowing the model’s performance to improve not only in the available context but also in the unseen context. This results in average personalization benefit (∆P ) of 67.81 and 85.25 for Scenario 1 and Scenario 2, respectively. It is important to note that for each Scenario, only one model is trained for the available context and tested for two different unseen contexts. Moreover, double context change (Tables 6g and 6h) shows lower personalization benefit as
PERCEPT-R: In this dataset, the heterogeneity of features among individuals is reflected through the difference in prediction accuracy of the generic model. It can be observed in Table 9a that for some individuals, the generic model exhibits over 90% accuracy on the available context data, while for others, the generic model’s accuracy drops to around 60%. This results in significant variability over gains in available and unseen contexts. Overall, CRoP yields an average personalization gain of 5.09%. On average over all the datasets, a personalization benefit (∆P ) of 35.23 percent points are seen as compared to the generic models across the four datasets under both scenarios. These evaluations establish that the personalized models obtained using CRoP demonstrate improved performance over the available context data than the generic models and exhibit personalization.
# Comparison with Personalized Models
The personalized models obtained using CRoP (MP a i θ ) are expected to have higher accuracy on unseen context Cu than the conventionally-finetune personalized models (MCa i θ ) as discussed in the Motivation Section. This section assesses whether the results align with these expectations.
WIDAR: Tables 8a and 8b demonstrate that the personalized models MP a i θ exhibit an average increment of 8.01 and 2.85 percent points in the unseen context for Scenario 1 and Scenario 2, respectively. However, an average loss of 1.57 and an average gain of 1.44 percent points in C′ as accuracy could be observed for Scenario 1 and Scenario 2, respectively.
Extrasensory: Similar patterns could be observed for the ExtraSensory dataset where the average accuracy on the unseen context improved by 4.97 and 12.61 percentage points for Scenario 1 and Scenario 2 as shown in Tables 6c and 8d, respectively. As expected, there is a loss of 5.43 and 4.44 percent points in the available contexts for Scenario 1 and Scenario 2, respectively.
Stress Sensing: As observed in Tables 6e-6h, personalized finetuning improves models performance on unseen context as well, we can claim that there is some person-specific traits which are common in available and unseen context. While comparing our final models with conventionally-finetuned models (Tables 8e-8h), performance boost in both available and unseen context could be observed. This can be attributed to the generalization improvement benefits of model pruning [Jin et al., 2022]. This results in average generalization benefit (∆G) of 13.81 and 13.08 for Scenario 1 and Scenario 2, respectively, for single context change. Similar personalization benefits could be seen for double context change.
PERCEPT-R: As observed in Table 9b, the variability in generalization benefits among different individuals is less pronounced as compared to personalization benefits. On average, CRoP introduces a generalization benefit of 2.57%. On average over all the datasets, a generalization benefit (∆G) of 7.78% percent points are seen over the conventionally-finetuned personalized models across all datasets under both scenarios.
# Individual Analysis for PERCEPT-R dataset
The heterogeneity of data in PERCEPT-R dataset resulted in variability in ∆P among various participants. In order to investigate that further, we conducted fisher information matrix (FIM) analysis of the generic modelMG θ , conventionally-finetuned model MCa i θ and the final models obtained using CRoP MP a i θ for the data belonging to available Ca and unseen contexts Cu. These results can be found in Table 7. Previous works have explored the fisher information matrix (FIM) as a means to investigate the curvature properties of the loss landscape Zandonati et al. [2022], Martens [2020] and its relationship to model generalizability Jastrzebski et al. [2021]. The fisher information trace serves as a metric to study the loss landscape curvature sensitivity Jastrzebski et al. [2021], Zandonati et al. [2022] i.e., a larger trace coincides with a sharper loss landscape minima signifying higher sensitivity and poorer generalization performance Jastrzebski et al. [2021], Kim et al. [2022].
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4078/4078810c-dedc-4559-bd1e-fdfdf8ced4e3.png" style="width: 50%;"></div>
Model
MG
θ
MP a
i
θ
A(MP a
i
θ , C) −A(MG
θ , C)
User
Ca
Cu
Ca
Cu
Ca
Cu
0
63.90
77.09
83.67
69.53
+19.77
-7.56
1
61.80
79.78
86.41
54.45
+24.61
-25.33
2
45.63
79.81
77.02
62.63
+31.38
-17.18
Average
+25.25
-16.69
∆P
+8.55
Model
MG
θ
MP a
i
θ
A(MP a
i
θ , C) −A(MG
θ , C)
User
Ca
Cu
Ca
Cu
Ca
Cu
61
78.69
69.83
82.59
69.66
+3.9
-0.17
7C
78.91
76.41
88.00
71.63
+9.09
-4.78
80
55.84
26.24
82.36
38.87
+26.52
+12.63
9D
73.74
85.63
82.81
84.72
+9.07
-0.91
B7
56.06
88.33
89.50
86.97
+33.44
-1.36
Average
+16.40
+1.08
∆P
+17.49
<div style="text-align: center;">(c) Scenario 1 for ExtraSensory dataset</div>
Model
MG
θ
MP a
i
θ
A(MP a
i
θ , C) −A(MG
θ , C)
User
Ca
Cu
Ca
Cu
Ca
Cu
1
88.39
81.90
94.54
97.59
+6.15
+15.69
2
47.40
50.0
77.12
90.47
+29.72
+40.47
3
36.90
43.48
96.36
95.31
+59.46
+51.93
Average
+31.78
+36.03
∆P
+67.81
(e) Scenario 1 for Stress Sensing - single context change
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bdaa/bdaaa176-d9ed-4c19-9440-50f0c26a8f77.png" style="width: 50%;"></div>
Model
MG
θ
MP a
i
θ
A(MP a
i
θ , C) −A(MG
θ , C)
User
Ca
Cu
Ca
Cu
Ca
Cu
1
88.39
64.71
94.54
76.46
+6.15
+11.75
2
47.40
50.70
77.12
63.22
+29.72
+12.52
3
36.90
11.94
96.36
55.48
+59.46
+43.54
Average
+31.78
+22.60
∆P
+54.38
(g) Scenario 1 for Stress Sensing - double context change
<div style="text-align: center;">(g) Scenario 1 for Stress Sensing - double context change</div>
<div style="text-align: center;">Table 6: Detailed Personalization (∆P ) results for WIDAR, ExtraSensory and Stress Sensing dataset</div>
It is observed that the 9 participants who experience greater ∆P benefits using CRoP also exhibit a reduction in the FIM trace for MP a i θ compared to MCa i θ across both contexts. In contrast, the other 7 participants with lower ∆P benefits show an increase in the FIM trace in both contexts. While there appears to be a correlation between ∆P benefits and changes in the FIM trace, the challenge lies in the fact that this information is not available during the personalization phase. Any decision regarding the necessity of CRoP for an individual must be based solely on the available context data Ca and the generic model MG θ , as these are the only sources of information accessible during personalization.
Model
MG
θ
MP a
i
θ
A(MP a
i
θ , C) −A(MG
θ , C)
User
Ca
Cu
Ca
Cu
Ca
Cu
0
73.28
61.80
82.59
58.38
+9.31
-2.43
1
73.18
59.58
92.44
47.90
+19.27
-11.67
2
80.45
46.13
87.5
42.31
+7.04
-3.81
Average
+11.88
-5.97
∆P
+5.90
Model
MG
θ
MP a
i
θ
A(MP a
i
θ , C) −A(MG
θ , C)
User
Ca
Cu
Ca
Cu
Ca
Cu
61
76.43
80.00
87.24
73.44
+10.81
-6.56
7C
75.07
92.32
83.18
89.39
+8.11
-2.93
