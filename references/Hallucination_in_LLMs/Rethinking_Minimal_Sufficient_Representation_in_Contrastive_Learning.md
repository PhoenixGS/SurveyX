# Rethinking Minimal Sufficient Representation in Contrastive Learning
Haoqing Wang*1 Xun Guo2 Zhi-Hong Deng1 Yan Lu2 1Peking University 2Microsoft Research Asia
# Abstract
Contrastive learning between different views of the data achieves outstanding success in the field of self-supervised representation learning and the learned representations are useful in broad downstream tasks. Since all supervision information for one view comes from the other view, contrastive learning approximately obtains the minimal sufficient representation which contains the shared information and eliminates the non-shared information between views. Considering the diversity of the downstream tasks, it cannot be guaranteed that all task-relevant information is shared between views. Therefore, we assume the nonshared task-relevant information cannot be ignored and theoretically prove that the minimal sufficient representation in contrastive learning is not sufficient for the downstream tasks, which causes performance degradation. This reveals a new problem that the contrastive learning models have the risk of over-fitting to the shared information between views. To alleviate this problem, we propose to increase the mutual information between the representation and input as regularization to approximately introduce more task-relevant information, since we cannot utilize any downstream task information during training. Extensive experiments verify the rationality of our analysis and the effectiveness of our method. It significantly improves the performance of several classic contrastive learning models in downstream tasks. Our code is available at https://github.com/Haoqing-Wang/InfoCL.
Recently, contrastive learning [6–8,19,52] between different views of the data achieves outstanding success in the field of self-supervised representation learning. The learned representations are useful for broad downstream tasks in practice, such as classification, detection and segmentation [21]. In contrastive learning, the representation that contains all shared information between views is defined as suf-
More Task-relevant Information Representation z1 Shared Task-relevant Information in Representation I(z1,v2,T) Non-shared Task-relevant Information in Representation I(z1,T|v2)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/83f9/83f90d18-df93-426c-ae05-6e5208aff81a.png" style="width: 50%;"></div>
Figure 1. Demonstration of our motivation using information diagrams. Based on the (approximately minimal) sufficient representation learned in contrastive learning, increasing I(z1, v1) approximately introduces more non-shared task-relevant information.
ficient representation, while the representation that contains only the shared and eliminates the non-shared information is defined as minimal sufficient representation [44]. Contrastive learning maximizes the mutual information between the representations of different views, thereby obtaining the sufficient representation. Furthermore, since all supervision information for one view comes from the other view [15], the non-shared information is often ignored, so that the minimal sufficient representation is approximately obtained. Tian et al. [41] find that the optimal views for contrastive learning depend on the downstream tasks when the minimal sufficient representation is obtained. In other words, the optimal views for task T1 may not be suitable for task T2. The reason may be that some information relevant to T2 is not shared between these views. In this work, we formalize this conjecture and assume that the non-shared task-relevant information cannot be ignored. Based on this assumption, we theoretically prove that the minimal sufficient representation contains less task-relevant information than other sufficient representations and has a non-ignorable gap with the optimal representation, which causes performance degradation. Concretely, we consider two types of the downstream task, i.e., classification and regression task, and prove that the lowest achievable error of the minimal sufficient representation is higher than other sufficient representations. According to our analysis, when some task-relevant information is not shared between views, the learned representation in contrastive learning is not sufficient for the downstream tasks. This reveals that the contrastive learning
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/765e/765e229d-a0d7-4b8d-8c9c-626007829d41.png" style="width: 50%;"></div>
models have the risk of over-fitting to the shared information between views. To this end, we need to introduce more non-shared task-relevant information to the representations. Since we cannot utilize any downstream task information in the training stage, it is impossible to achieve this directly. As an alternative, we propose an objective term which increases the mutual information between the representation and input to approximately introduce more task-relevant information. This motivation is demonstrated in Fig. 1 using information diagrams. We consider two implementations to increase the mutual information. The first one reconstructs the input to make the representations containing the key information about the input [27,45]. The second one relies on the high-dimensional mutual information estimate [5,35]. Overall, we summarize our contributions as follows.
 We verify the effectiveness of our method for SimCLR [7], BYOL [19] and Barlow Twins [52] in classification, detection and segmentation tasks. We also provide extensive analytical experiments to further understand our hypotheses, theoretical analysis and model.
# 2. Related works
Contrastive learning. Contrastive learning between different views of the data is a successful self-supervised representation learning framework. The views are constructed by exploiting the structure of the unlabeled data, such as local patches and the whole image [24], different augmentations of the same image [2, 7, 21, 49], or video and text pairs [32,39]. Recently, Tian et al. [41] find that the optimal views for contrastive learning are task-dependent under the assumption of minimal sufficient representation. In other words, even if the given views are optimal for some downstream tasks, they may not be suitable for other tasks. In this work, we theoretically analyze this discovery and find that the contrastive learning models may over-fit to the shared information between views, and thus propose to increase the mutual information between representation and input to alleviate this problem. Some recent works [15,44] propose to learn the minimal sufficient representation. They assume that almost all the information relevant to downstream tasks
<div style="text-align: center;">Figure 2. Internal mechanism of contrastive learning: the views provide supervision information to each other.</div>
is shared between views, which is an overly idealistic assumption and conflicts with the discovery in [41].
Information bottleneck theory. Based on the information bottleneck theory [37, 42, 43], a model extracts all task-relevant information in the first phase of learning (drift phase) to ensure sufficiency, and then compresses the taskirrelevant information in the second phase (diffusion phase). Our analysis shows that the learned representation in contrastive learning is not sufficient for the downstream tasks and can be seen as in the drift phase. We need to introduce more task-relevant information to achieve sufficiency.
# 3. Theoretical analysis and model
In this section, we first introduce the contrastive learning framework and theoretically analyze the disadvantages of minimal sufficient representation in contrastive learning, and then propose our method to approximately introduce more task-relevant information to the representations. Note that although the analysis is about the information content in the representations, its specific form is also very important. Therefore, our theoretical analysis is actually based on the premise that the information content in the representations is represented in the most appropriate form.
# 3.1. Contrastive learning
Contrastive learning is a general framework for unsupervised representation learning which maximizes the mutual information between the representations of two random variables v1 and v2 with the joint distribution p(v1, v2)
where zi = fi(vi), i = 1, 2 are also random variables and fi, i = 1, 2 are encoding functions. In practice, v1 and v2 are usually two views of the data x. When v1 and v2 have the same marginal distributions (p(v1) = p(v2)), the function f1 and f2 can be the same (f1 = f2).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/601f/601f495c-e60c-482f-88a4-e0b0be63e202.png" style="width: 50%;"></div>
In contrastive learning, the variable v2 provides supervision information for v1 and plays the similar role as the label y in the supervised learning, and vice versa [15]. This internal mechanism is illustrated in Fig. 2. Similar to the information bottleneck theory [1,43] in the supervised learning, we can define the sufficient representation and minimal sufficient representation of v1 (or v2) for v2 (or v1) in contrastive learning [41,44].
Definition 1. (Sufficient Representation in Contrastive Learning) The representation zsuf 1 of v1 is sufficient for v2 if and only if I(zsuf 1 , v2) = I(v1, v2).
The sufficient representation zsuf 1 of v1 keeps all the information about v2 in v1. In other words, zsuf 1 contains all the shared information between v1 and v2, i.e., I(v1, v2|zsuf 1 ) = 0. Symmetrically, the sufficient representation zsuf 2 of v2 for v1 satisfies I(v1, zsuf 2 ) = I(v1, v2).
Definition 2. (Minimal Sufficient Representation in Contrastive Learning) The sufficient representation zmin 1 of v1 is minimal if and only if I(zmin 1 , v1) ≤I(zsuf 1 , v1), ∀zsuf 1 that is sufficient.
Among all sufficient representations, the minimal sufficient representation zmin 1 contains the least information about v1. Further, it is usually assumed that zmin 1 only contains the shared information between views and eliminates other non-shared information, i.e., I(zmin 1 , v1|v2) = 0. Applying Data Processing Inequality [11] to the Markov chain v1 →v2 →z2 and z2 →v1 →z1, we have
I(v1, v2) ≥I(v1, z2) ≥I(z1, z2)
(2)
i.e., I(v1, v2) is the upper bound of I(z1, z2). Considering that I(v1, v2) remains unchanged during the optimization process, contrastive learning optimizes the functions f1 and f2 so that I(z1, z2) approximates I(v1, v2). When these functions have enough capacity and are well learned based on sufficient data, we can assume I(z1, z2) = I(v1, v2), which means the learned representations in contrastive learning are sufficient. They are also approximately minimal since all supervision information comes from the other view. Therefore, the shared information controls the properties of the representations. The learned representations in contrastive learning are typically used in various downstream tasks, so we introduce a random variable T to represent the information required for a downstream task which can be classification, regression or clustering task. Tian et al. [41] find that the optimal views for contrastive learning are task-dependent under the assumption of minimal sufficient representation. This discovery is intuitive since various downstream tasks need different information that is unknown during training.
Figure 3. Information diagrams of different representations in contrastive learning. We consider the situation where the non-shared task-relevant information I(v1, T|v2) cannot be ignored. Contrastive learning makes the representations extracting the shared information between views to obtain the sufficient representation which is approximately minimal. The minimal sufficient representation contains less task-relevant information from the input than other sufficient representations.
It is difficult for the given views to share all the information required by these tasks. For example, when one view is a video stream and the other view is an audio stream, the shared information is sufficient for identity recognition task, but not for object tracking task. Some task-relevant information may not lie in the shared information between views, i.e., I(v1, T|v2) cannot be ignored. Eliminating all non-shared information has the risk of damaging the performance of the representations in the downstream tasks.
# 3.2. Analysis on minimal sufficient representation
The minimal sufficient representation intuitively is not a good choice for downstream tasks, because it completely eliminates the non-shared information between views which may be important for some downstream tasks. We formalize this problem and theoretically prove that in contrastive learning, the minimal sufficient representation is expected to perform worse than other sufficient representations in the downstream tasks. All proofs for the below theorems are provided in Appendix A. Considering the symmetry between v1 and v2, without loss of generality, we take v2 as the supervision signal for v1 and take v1 as the input of a task. It is generally believed that the more task-relevant information contained in the representations, the better performance can be obtained [11,14]. Therefore, we examine the task-relevant information contained in the representations.
Theorem 1. (Task-Relevant Information in Representations) In contrastive learning, for a downstream task T, the minimal sufficient representation zmin 1 contains less taskrelevant information from input v1 than other sufficient representation zsuf 1 , and I(zmin 1 , T) has a gap of I(v1, T|v2) with the upper bound I(v1, T). Formally, we have
I(v1, T) = I(zmin 1 , T) + I(v1, T|v2) ≥I(zsuf 1 , T) = I(zmin 1 , T) + I(zsuf 1 , T|v2) ≥I(zmin 1 , T)
(3)
Theorem 1 indicates that zsuf 1 can have better performance in task T than zmin 1 because it contains more taskrelevant information. When non-shared task-relevant information I(v1, T|v2) is significant, zmin 1 has poor performance because it loses a lot of useful information. See Fig. 3 for the demonstration using information diagrams. To make this observation more concrete, we examine two types of the downstream task: classification tasks and regression tasks, and provide theoretical analysis on the generalization error of the representations. When the downstream task is a classification task and T is a categorical variable, we consider the Bayes error rate [17] which is the lowest achievable error for any classifier learned from the representations. Concretely, let Pe be the Bayes error rate of arbitrary learned representation z1 and �T be the prediction for T based on z1, we have Pe = 1 −Ep(z1)[maxt∈T p( �T = t|z1)] and 0 ≤Pe ≤ 1 −1/|T| where |T| is the cardinality of T. According to the value range of Pe, we define a threshold function Γ(x) = min{max{x, 0}, 1 −1/|T|} to prevent overflow.
Theorem 2. (Bayes Error Rate of Representations) For arbitrary learned representation z1, its Bayes error rate Pe = Γ( ¯Pe) with
 (4)
Specifically, for sufficient representation zsuf 1 , its Bayes error rate P suf e = Γ( ¯P suf e ) with
 (5)
for minimal sufficient representation zmin 1 , its Bayes error rate P min e = Γ( ¯P min e ) with
(6)
Since I(zsuf 1 , T|v2) ≥0, Theorem 2 indicates for classification task T, the upper bound of P min e is larger than P suf e . In other words, zmin 1 is expected to obtain a higher classification error rate in the task T than zsuf 1 . According to the Eq. (5), considering that H(T) and I(v1, v2, T) are not related to the representations, increasing I(zsuf 1 , T|v2) can reduce the Bayes error rate in classification task T. When I(zsuf 1 , T|v2) = I(v1, T|v2), zsuf 1 contains all the useful information for task T in v1. When the downstream task is a regression task and T is a continuous variable, let �T be the prediction for T based on arbitrary learned representation z1, we consider the smallest achievable expected squared prediction error Re = min � T E[(T −�T(z1))2] = E[ε2] with ε(T, z1) = T −E[T|z1].
� Theorem 3. (Minimum Expected Squared Prediction Error of Representations) For arbitrary learned representa-
tion z1, when the conditional distribution p(ε|z1) is uniform, Laplacian or Gaussian distribution, the minimum expected squared prediction error Re satisfies
for minimal sufficient representation zmin 1 , its minimum expected squared prediction error Rmin e satisfies
(9)
where the constant coefficient α depends on the conditional distribution p(ε|z1).
The assumption about the estimation error ε in Theorem 3 is reasonable because ε is analogous to the ‘noise’ with the mean of 0, which is generally assumed to come from simple distributions (e.g., Gaussian distribution) in statistical learning theory. Similar to the classification tasks, Theorem 3 indicates that for regression tasks, zsuf 1 can achieve lower expected squared prediction error than zmin 1 and increasing I(zsuf 1 , T|v2) can improve the performance. Theorem 2 and Theorem 3 analyze the disadvantages of the minimal sufficient representation zmin 1 in classification tasks and regression tasks respectively. The essential reason is that zmin 1 has less task-relevant information than zsuf 1 and has a non-ignorable gap I(v1, T|v2) with the optimal representation, as shown in Theorem 1.
# 3.3. More non-shared task-relevant information
According to the above theoretical analysis, in contrastive learning, the minimal sufficient representation is not sufficient for downstream tasks due to the lack of some nonshared task-relevant information. Moreover, contrastive learning approximately learns the minimal sufficient representation, thereby having the risk of over-fitting to the shared information between views. To this end, we propose to extract more non-shared task-relevant information from v1, i.e., increasing I(z1, T|v2). However, we cannot utilize any downstream task information during training, so it is impossible to increase I(z1, T|v2) directly. We consider increasing I(z1, v1) as an alternative because the increased information from v1 in z1 may be relevant to some downstream tasks, and this motivation is demonstrated in Fig. 1. In addition, increasing I(z1, v1) also helps to extract the shared information between views at the beginning of the optimization process. Concretely, considering the symmetry between v1 and v2, our optimization objective is
(10)
which consists of the original optimization objective Eq. (1) in contrastive learning and the objective terms we proposed. The coefficients λ1 and λ2 are used to control the amount of increasing I(z1, v1) and I(z2, v2) respectively. For optimizing I(z1, z2), we adopt the commonly used implementations in contrastive learning models [7,18,52]. For optimizing I(zi, vi), i = 1, 2, we consider two implementations.
Implementation I Since I(z, v) = H(v) −H(v|z) and H(v) is not related with z, we can equivalently decrease the conditional entropy H(v|z) = −Ep(z,v)[ln p(v|z)]. Concretely, we use the representation z to reconstruct the original input v, as done in auto-encoder models [45]. Decreasing the entropy of reconstruction encourages the representation z to contain more information about the original input v. However, the conditional distribution p(v|z) is intractable in practice, so we use q(v|z) as an approximation and get Ep(z,v)[ln q(v|z)], which is the lower bound of Ep(z,v)[ln p(v|z)]. We can increase Ep(z,v)[ln q(v|z)] as an alternative objective. According to the type of input v (e.g., images, text or audio), q(v|z) can be any appropriate distribution with known probability density function, such as Bernoulli distribution, Gaussian distribution or Laplace distribution, and its parameters are the functions of z. For example, when q(v|z) is the Gaussian distribution N(v; µ(z), σ2I) with given variance σ2 and deterministic mean function µ(·) which is usually parameterized by neural networks, we have
where c is a constant to representation z. The final optimization objective is
(12)
Implementation II Although the above implementation is effective and preferred in practice, it needs to reconstruct the input, which is challenging for complex input and introduces more model parameters. To this end, we propose another representation-level implementation as an optional alternative. We investigate various lower bound estimates of mutual information, such as the bound of Barber and Agakov [4], the bound of Nguyen, Wainwright and Jordan [33], MINE [5] and InfoNCE [35]. We choose the InfoNCE lower bound and the detailed discussion is provided in Appendix B. Concretely, the InfoNCE lower bound is
(13)
� � where (zk, vk), k = 1, · · · , N are N copies of (z, v) and the expectation is over Πkp(zk, vk). In the implementation I, we map the input v to the representation z through a
deterministic function f with z = f(v). Differently, here we need the expression of p(z|v) to calculate the InfoNCE lower bound, which means the representation z is no longer a deterministic output of input v, so we use the reparameterization trick [27] during training. For example, when we define p(z|v) as the Gaussian distribution N(z; f(v), σ2I) with given variance σ2 and the function f is the same as in the Implementation I, we have z = f(v) + ϵσ, ϵ ∼N(0, I) and ˆINCE is equivalent to
(14)
�  � where ρ is a scale factor. In fact, it pushes the representations away from each other to increase H(z), which can increase mutual information I(z, v) since I(z, v) = H(z) −H(z|v) = H(z) −d 2(ln 2π + ln σ2 + 1) with d being representation dimension. It also be denoted as uniformity property [48]. The final optimization objective is
(15)
Since the objective term Eq. (14) is calculated at the representation-level, when we use the convolutional neural networks (e.g., ResNet [23]) to parameterize f, it can be applied to the output activation of multiple internal blocks. Discussion. It is worth noting that increasing I(z, v) does not conflict with the information bottleneck theory [43]. According to our analysis, the learned representations in contrastive learning are not sufficient for the downstream tasks. Therefore, we need to make the information in the representations more sufficient but not to compress it. On the other hand, we cannot introduce too much information from the input v either, which may contain harmful noise. Here we use the coefficients λ1 and λ2 to control this.
# 4. Experiments
In this section, we first verify the effectiveness of increasing I(z, v) on various datasets, and then provide some analytical experiments. We choose three classic contrastive learning models as our baselines: SimCLR [7], BYOL [19] and Barlow Twins [52]. We denote our first implementation Eq. (12) as ”RC” for ”ReConstruction” and the second implementation Eq. (15) as ”LBE” for ”Lower Bound Estimate”. For all experiments, we use random cropping, flip and random color distortion as the data augmentation, as suggested by [7]. For ”LBE”, we set σ = 0.1 and ρ = 0.05.
# 4.1. Effectiveness of increasing I(z, v)
We consider different types of the downstream task, including classification, detection and segmentation tasks. The results of Barlow Twins are provided in Appendix C.1.
Model
CIFAR10
DTD
MNIST
FaMNIST
CUBirds
VGGFlower
TrafficSigns
SimCLR
85.76
29.52
97.03
88.36
8.87
42.81
92.41
SimCLR+RC (ours)
85.78
33.67
97.99
90.31
10.89
54.16
95.84
SimCLR+LBE (ours)
85.45
34.52
97.94
89.26
10.60
54.10
94.96
BYOL
85.64
31.22
97.15
88.92
8.84
40.90
92.17
BYOL+RC (ours)
85.80
34.73
98.07
89.61
9.68
48.75
94.19
BYOL+LBE (ours)
85.28
33.99
97.76
88.99
9.96
54.10
95.09
Model
STL-10
DTD
MNIST
FaMNIST
CUBirds
VGGFlower
TrafficSigns
SimCLR
78.74
39.41
95.00
87.31
8.34
49.41
80.25
SimCLR+RC (ours)
79.21
41.81
97.48
89.98
10.03
60.46
94.73
SimCLR+LBE (ours)
80.17
42.07
97.04
88.68
10.11
58.51
87.77
BYOL
80.83
40.05
94.45
87.23
8.54
49.41
77.54
BYOL+RC (ours)
81.11
42.02
96.96
88.92
9.63
55.71
88.57
BYOL+LBE (ours)
80.85
42.55
95.75
87.88
10.55
59.39
84.62
Model
ImageNet
DTD
CIFAR10
CIFAR100
CUBirds
VGGFlower
TrafficSigns
SimCLR
61.01
70.16
82.30
59.86
36.49
93.52
95.27
SimCLR+RC (ours)
61.60
71.22
83.30
63.56
37.42
94.53
96.47
SimCLR+LBE (ours)
61.37
70.95
83.20
61.99
37.78
94.34
95.99
Pretraining. We train the models on CIFAR10 [28], STL10 [10] and ImageNet [12]. For CIFAR10 and STL-10, we use the ResNet18 [23] backbone and the models are trained for 200 epochs with batch size 256 using Adam optimizer with learning rate 3e-4. For ImageNet, we use the ResNet50 [23] backbone and the models are trained for 200 epochs with batch size 1024 using LARS optimizer [51] and a cosine decay learning rate schedule.
Linear evaluation. We follow the linear evaluation protocol where a linear classifier is trained on top of the frozen backbone. The linear evaluation is conducted on the source dataset and several transfer datasets: CIFAR100 [28], DTD [9], MNIST [29], FashionMNIST [50], CUBirds [46], VGG Flower [34] and Traffic Signs [26]. The linear classifier is trained for 100 epochs using SGD optimizer. Table 1 shows the results on CIFAR10, STL-10 and ImageNet, and the best result in each block is in bold. Our implemented results of the baselines are consistent with [7, 25, 40, 47]. Increasing I(z, v) can introduce non-shared information and improve the classification accuracy, especially on transfer datasets. This means the shared information between views is not sufficient for some tasks, e.g., classification on DTD, VGG Flower and Traffic Signs where increasing I(z, v) achieves significant improvement. In other words, increasing I(z, v) can prevent the models from over-fitting to the shared information between views. What’s more, it is effective for various contrastive learning models, which means our analysis results are widely applicable in contrastive learning. In fact, they all satisfy the internal mechanism.
Object detection and instance segmentation. We conduct object detection on VOC07+12 [13] using Faster R-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/cc51/cc5169ba-6d0a-4e92-9745-6dee48b09030.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) Instance segmentation on COCO</div>
Table 2. Object detection and instance segmentation on VOC07+12 and COCO. The models on COCO are fine-tuned using the default 2× schedule. In magenta are the gaps of at least +0.5 point to the baseline, SimCLR.
CNN [36], and detection and instance segmentation on COCO [30] using Mask R-CNN [22], following the setup in [21]. All methods use the R50-C4 [22] backbone that is initialized using the ResNet50 pre-trained on ImageNet. The results are show in Table 2. Increasing I(z, v) significantly improves the precision in object detection and instance segmentation tasks. These dense prediction tasks require some local semantic information from the input. Increasing I(z, v) can make the representation z contain more information from the input v which may not be shared between views, thereby obtaining better precision.
Model
CIFAR10
DTD
MNIST
FaMNIST
CUBirds
VGGFlower
TrafficSigns
SimCLR
85.76
29.52
97.03
88.36
8.87
42.81
92.41
SimCLR+IP
85.86
30.15
96.71
88.18
8.66
43.22
92.13
SimCLR†
85.81
31.70
97.08
88.85
8.77
44.41
92.41
SimCLR+MIB
86.20
31.17
97.00
88.62
9.01
43.88
93.01
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d3cb/d3cbe47e-850f-4ca7-b654-3469fcadfecd.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4. Linear evaluation accuracy on the source dataset (CIFAR10 or STL-10) and the averaged accuracy on all transfer datasets with varying hyper-parameter λ.</div>
# 4.2. Analytical experiments
We provide some analytical experiments to further understand our hypotheses, theoretical analysis and models.
Eliminating non-shared information. Some recent works [15, 44] propose to eliminate the non-shared information between views in the representation to get the minimal sufficient representation. To this end, Federici et al. [15] minimize the regularization term
where KL(·||·) represents the Kullback-Leibler divergence. When p(z1|v1) and p(z2|v2) are modeled as N(zi; fi(vi), σ2I), i = 1, 2 with given variance σ2, it can be rewritten as LMIB = Ep(v1,v2) � ∥f1(v1) −f2(v2)∥2 2 � . Identically, Tsai et al. [44] minimize the inverse predictive loss LIP = Ep(v1,v2) � ∥f1(v1) −f2(v2)∥2 2 � . The detailed derivation is provided in Appendix D. We evaluate these two regularization terms in the linear evaluation tasks and choose their coefficient with best accuracy on the source dataset. The results are shown in Table 3 and the best result in each block is in bold. Although these two regularization terms have the same form, LMIB uses stochastic encoders
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/962c/962c2095-6fae-4b4f-8722-bafab75f0adc.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) STL-10</div>
Figure 5. Linear evaluation accuracy on the source dataset (CIFAR10 or STL-10) and the averaged accuracy on all transfer datasets with varying epochs.
which is equivalent to adding Gaussian noise, so we report the results of SimCLR with Gaussian noise, marked by †. As we can see, eliminating the non-shared information cannot change the accuracy in downstream classification tasks much. This means that the sufficient representation learned in contrastive learning is approximately minimal and we don’t need to further remove the non-shared information.
Changing the amount of increasing I(z, v). Quantifying the mutual information between the high-dimensional variables is very difficult, and often leads to inaccurate calculation in practice [31, 38]. Therefore, we assume that the hyper-parameters λ1 and λ2 control the amount of increasing I(z1, v1) and I(z2, v2) respectively. Larger λ1 is expected to increase I(z1, v1) more, so as λ2. We set λ1 = λ2 = λ and evaluate the performance of different λ from {0.001, 0.01, 0.1, 1, 10}. We choose SimCLR as the baseline and the results are shown in Fig. 4. We report the accuracy on the source dataset (CIFAR10 or STL-10) and the averaged accuracy on all transfer datasets. As we can see, increasing I(z, v) consistently improves the performance in downstream classification tasks. We can observe a non-monotonous reverse-U trend of accuracy with the change of λ, which means excessively increasing I(z, v) may introduce noise beside useful information.
Model
CIFAR10
DTD
MNIST
FaMNIST
CUBirds
VGGFlower
TrafficSigns
Supervised
93.25
34.10
98.52
90.09
8.37
46.14
93.05
Supervised+RC (ours)
93.09
32.77
98.61
89.77
8.84
49.05
93.28
Supervised+LBE (ours)
93.18
34.79
98.68
90.40
9.72
53.15
94.47
Model
CIFAR100
DTD
MNIST
FaMNIST
CUBirds
VGGFlower
TrafficSigns
Supervised
71.92
36.06
98.48
88.97
11.51
64.21
96.54
Supervised+RC (ours)
72.02
34.79
98.59
89.35
10.94
65.34
96.67
Supervised+LBE (ours)
71.89
36.33
98.37
89.42
11.89
65.64
96.91
Training with more epochs. In the above experiments, we train all models for 200 epochs. Here we further show the behavior of the contrastive learning models and increasing I(z, v) when training with more epochs. We choose SimCLR as the baseline and train all models for 100, 200, 300, 400, 500 and 600 epochs. The results are shown in Fig. 5. With more training epochs, the learned representations in contrastive learning are more approximate to the minimal sufficient representation which mainly contain the shared information between views and ignore the nonshared information. For the classification tasks on the transfer datasets, the shared information between views is not sufficient. As shown in Fig. 5 (b) and (d), the accuracy on the transfer datasets decreases with more epochs and the learned representations over-fit to the shared information between views. Increasing I(z, v) can introduce non-shared information and obtain the significant improvement. For the classification tasks on the source datasets, the shared information between views is sufficient on CIFAR10 but not on STL-10. As shown in Fig. 5 (a) and (c), the accuracy on CIFAR10 increases with more epochs and increasing I(z, v) cannot make a difference. But the accuracy on STL-10 decreases with more epochs, and increasing I(z, v) can significantly improve the accuracy and does not decrease with more epochs. In fact, we use the unlabeled split for contrastive training on STL-10, so it is intuitive that the shared information between views is not sufficient for the classification tasks on the train and test split.
Increasing I(z, x) in supervised learning. According to the information bottleneck theory [43], a model extracts the approximate minimal sufficient statistics of the input x with respect to the label y in supervised learning. In other words, the representation z only contains the information related to the label and eliminates other irrelevant information which is considered as noise. However, label-irrelevant information may be useful for some downstream tasks, so we evaluate the effect of increasing I(z, x) in supervised learning. We train the ResNet18 backbone using the crossentropy classification loss on CIFAR10 and CIFAR100, and choose λ1 = λ2 = λ from {0.001, 0.01, 0.1, 1, 10}. The linear evaluation results are shown in Table 4 and the
best result in each block is in bold. As we can see, increasing I(z, x) improves the performance on the transfer datasets and achieves comparable results on the source dataset, which means it can effectively alleviate the overfitting on the label information. This discovery helps to obtain more general representations in the field of supervised pre-training and we left it for the future work.
# 5. Limitations
Our work has the following limitations. 1) Based on our experimental observation, the assumption that non-shared task-relevant information cannot be ignored usually well holds for the cross-domain transfer tasks, but may not be satisfied for the tasks on the training dataset. 2) Increasing I(z, v) can also introduce noise (task-irrelevant) information which may increase the data demand in the downstream tasks, so one may need to adjust the coefficients λ1 and λ2 to achieve effective trade-off for the different downstream tasks. 3) Due to limited computing resources, we cannot reproduce the best results of SimCLR on ImageNet which need the batch size of 4096 and more training epochs.
# 6. Conclusions
In this work, we explore the relationship between the learned representations and downstream tasks in contrastive learning. Although some works propose to learn the minimal sufficient representation, we theoretically and empirically verify that the minimal sufficient representation is not sufficient for downstream tasks because it loses non-shared task-relevant information. We find that contrastive learning approximately obtains the minimal sufficient representation, which means it may over-fit to the shared information between views. To this end, we propose to increase the mutual information between the representation and input to approximately introduce more non-shared task-relevant information when the downstream tasks are unknown. For the future work, we can consider combining the reconstruction models [3, 20] and contrastive learning for convolutional neural networks or vision transformers, since reconstruction can learn more sufficient information and contrast can make the representations more discriminative.
# A. Proofs of theorems
In this section, we provide the proofs of the theorems in the main text. Since the random variable z1 = f1(v1) is the representation of random variable v1 where f1 is an encoding function, we have
Assumption 1. Random variable z1 is conditionally independent from any other variable s in the system once random variable v1 is observed, i.e., I(z1, s|v1) = 0, ∀s.
This assumption is also adopted in [15]. When f1 is a deterministic function, this assumption strictly holds. And when f1 is a random function, the information in z1 consists of the information from v1 and the information introduced by the randomness of function f1 which can be considered irrelevant to other variables in the system, so this assumption still holds. Next, we first present two lemmas for subsequent proofs.
Lemma 1. Let zsuf 1 and zmin 1 are the sufficient representation and the minimal sufficient representation of view v1 for v2 in contrative learning respectively, we have
(17)
Proof. 1) From the Definition 1 and the Assumption 1, we have
I(v1, v2, T) −I(zsuf 1 , v2, T) = [I(v1, v2) −I(v1, v2|T)] −[I(zsuf 1 , v2) −I(zsuf 1 , v2|T)] = I(zsuf 1 , v2|T) −I(v1, v2|T) = [H(v2|T) −H(v2|zsuf 1 , T)] −[H(v2|T) −H(v2|v1, T)] = H(v2|v1, T) −H(v2|zsuf 1 , T) = [I(zsuf 1 , v2|v1, T) + H(v2|v1, zsuf 1 , T)] −[I(v1, v2|zsuf 1 , T) + H(v2|v1, zsuf 1 , T)] = I(zsuf 1 , v2|v1, T) −I(v1, v2|zsuf 1 , T) = I(zsuf 1 , v2|v1, T) = 0
I(zsuf 1 , v2, T) = I(v1, v2, T)
# The above proof process only uses the sufficiency of zsuf 1 for v2, so we have
I(zmin 1 , v2, T) = I(v1, v2, T)
We consider the conditional entropy of the task variable T given the representation z1.
Lemma 2. For arbitrary learned representation z1, the conditional entropy H(T|z1) of the task variable T given z1 satisfies
(19)
(21)
Applying the Eq. (17), the conditional entropy H(T|zsuf 1 ) satisfies
Finally, we give the proofs of Theorem 1, 2 and 3.
The proof of Theorem 1.
3) I(v1, T|v2) ≥I(zsuf 1 , T|v2) ≥0. Applying the Data Processing Inequality [11] to the Markov chain T →v1 →zsuf 1 , we have I(v1, T) ≥ I(zsuf 1 , T), so
≥ | ≥ Combining these three equations, we can get Theorem 1.
# The proof of Theorem 2.
Proof. According to [14], the relationship between the Bayes error rate Pe and the conditional entropy H(T|z1) is −ln(1 −Pe) ≤H(T|z1)
− which is equivalent to
≤ −− − Note that 0 ≤Pe ≤1 −1/|T|, so we use the threshold function Γ(x) = min{max{x, 0}, 1 −1/|T|} to prevent overflow.
Proof. According to [16], when the conditional distribution p(ε|z1) of estimation error ε is uniform, Laplace and Gaussian distribution, the minimum expected squared prediction error Re becomes 1 12 exp[2H(T|z1)], 1 2e2 exp[2H(T|z1)] and 1 2πe exp[2H(T|z1)] respectively. Therefore, we unify them as
where α is a constant coefficient which depends on the con ditional distribution p(ε|z1). Applying the Lemma 2, fo arbitrary learned representation z1, we have Re = α · exp[2 · (H(T) −I(z1, T|v2) −I(z1, v2, T))] for the sufficient representation zsuf 1 , we have Rsuf e = α · exp[2 · (H(T) −I(zsuf 1 , T|v2) −I(v1, v2, T))] for the minimal sufficient representation zmin 1 , we have Rmin e = α · exp[2 · (H(T) −I(v1, v2, T))]
# B. Choice of mutual information estimate
In our Implementation II, we need to use a mutual information lower bound estimate to calculate I(z, v) where v is the original input (e.g., images) and z is the representation (feature vectors). We consider three candidate estimates: 1) The bound of Nguyen, Wainwright and Jordan [33]
 (22)
(23)
(24)
� where (zk, vk), k = 1, · · · , N are N copies of (z, v) and the expectation is over Πkp(zk, vk). As we can see, when we calculate the bound ˆINW J and ˆIMINE, we need to calculate the critic h(z, v) between the representation z and original input v. If we use a neural network to model the critic h(z, v), we have to take the original input (e.g. images) and the representation together as the input of a neural network. Since the distribution of the original input v and the representation z is quite different, it is very difficult. Therefore, we use the InfoNCE lower bound estimate.
Model
CIFAR10
DTD
MNIST
FaMNIST
CUBirds
VGGFlower
TrafficSigns
BarTwins
86.85
28.56
95.39
86.19
7.49
35.91
88.50
BarTwins+RC (ours)
86.91
28.97
96.60
86.72
7.90
38.94
90.92
BarTwins+LBE (ours)
86.38
29.54
96.72
86.88
8.47
41.44
92.76
Model
STL-10
DTD
MNIST
FaMNIST
CUBirds
VGGFlower
TrafficSigns
BarTwins
80.59
36.86
94.27
86.63
7.47
44.89
73.73
BarTwins+RC (ours)
82.21
36.97
94.45
86.71
7.89
46.31
78.94
BarTwins+LBE (ours)
81.13
37.32
96.33
87.13
8.08
49.82
82.08
# C. More experiments
In this section, we provide more experiments to support our work.
# C.1. Results on Barlow Twins
In the main text, we provide the results on two classic contrastive learning models: SimCLR [7] and BYOL [19]. SimCLR perfectly matches the contrastive learning framework, maximizing the lower bound estimate of the mutual information I(z1, z2). BYOL avoids the dependence on the large amount of negative samples, and adopts prediction loss and the asymmetric structure. We further verify the effectiveness of increasing I(z, v) on Barlow Twins [52] which makes the cross-correlation matrix between the representations of different views as close to the identity matrix as possible. Although the loss functions of these contrastive learning models are very different, they all satisfy the internal mechanism that the views provide supervision information to each other, so they all approximately learn the minimal sufficient representation. We use the same pre-training schedule and linear evaluation protocol as in the main text and set λ1 = λ2 = 1. For STL-10, we use the unlabeled split for contrastive learning and the train and test split for linear evaluation. The results are shown in Table 5 and the best result in each block is in bold. Increasing I(z, v) can improve the accuracy of the learned representations in Barlow Twins in downstream classification tasks, which indicates that our analysis results are applicable to various contrastive losses.
# C.2. Reconstructed samples
In order to show the reconstruction effect of our Implementation I, we provide the reconstructed images after training. As an example, we use SimCLR contrastive loss and take CIFAR10 as the training dataset. The original input images and the reconstructed images are shown in Fig. 6. As we can see, the reconstructed images retain the shape and outline information in the original images, so as the obtained representations. Since we use the mean square error loss to optimize the reconstruction module, the reconstructed images are blurry and this phenomenon is also ob-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0f48/0f4814ea-f80c-4c3b-b869-694f413fed4e.png" style="width: 50%;"></div>
Figure 6. Demonstration of the reconstruction effect of our Implementation I. We provide the original input images and the reconstructed images for comparison. We use SimCLR contrastive loss and take CIFAR10 as the training dataset.
# served in vanilla variational auto-encoder [27].
served in vanilla variational auto-encoder [27].
# D. Derivation of LMIB and LIP
Federici et al. [15] and Tsai et al. [44] propose to eliminate the non-shared information between views in the representation to get the minimal sufficient representation. To this end, they propose their respective regularization terms. Here we derive the specific forms used in the main text. In [15], the regularization term is
According to the description in their paper and the official code 1, they model the two stochastic encoders p(z1|v1) and
p(z2|v2) as
(26) (27)
where µ1(v1),σ2 1(v1),µ2(v2) and σ2 2(v2) are all functions of the input (v1 or v2), diag(e) creates a matrix in which the diagonal elements consist of vector e and all off-diagonal elements are zeros. The regularization term has the analytical expression
� (28)
� where d is the dimension of z1 and z2. We want to minimize LMIB, and when σ2 1 = σ2 2, the term σi2 1 /σi2 2 + σi2 2 /σi2 1 takes the minimum value 2, so the regularization term becomes
(29)
In practice, minimizing LMIB makes the variance σ2 1 and σ2 2 very large, and the sampled representations change drastically and have very poor performance in downstream tasks. If the upper bound of the variance σ2 1 and σ2 2 is fixed, such as using the sigmoid activation function to limit it to (0, 1), they will converge to the maximum value as the training progresses. Therefore, we might as well fix the variance and model the two stochastic encoders p(z1|v1) and p(z2|v2) as
(30) (31)
where I is the identity matrix, σ2 is the given variance, fi, i = 1, 2 are deterministic encoders. This also guarantees a fair comparison with our Implementation II. According to the Eq. (29), the regularization term is equivalent to
(32)
We calculate the expectation of the regularization term on the data distribution p(v1, v2) and get
(33)
In [44], the authors define the inverse predictive loss
