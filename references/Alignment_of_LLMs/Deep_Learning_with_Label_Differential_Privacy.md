# Deep Learning with Label Differential Privacy
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3c3f/3c3f7493-0076-4438-b0c7-95df3740e67e.png" style="width: 50%;"></div>
# Abstract
The Randomized Response (RR) algorithm [96] is a classical technique to improve robustness in survey aggregation, and has been widely adopted in applications with differential privacy guarantees. We propose a novel algorithm, Randomized Response with Prior (RRWithPrior), which can provide more accurate results while maintaining the same level of privacy guaranteed by RR. We then apply RRWithPrior to learn neural networks with label differential privacy (LabelDP), and show that when only the label needs to be protected, the model performance can be significantly improved over the previous state-of-the-art private baselines. Moreover, we study different ways to obtain priors, which when used with RRWithPrior can additionally improve the model performance, further reducing the accuracy gap between private and non-private models. We complement the empirical results with theoretical analysis showing that LabelDP is provably easier than protecting both the inputs and labels.
# 1 Introduction
The widespread adoption of machine learning in recent years has increased the concerns about the privacy of individuals whose data is used during the model training. Differential privacy (DP) [34, 33] has emerged as a popular privacy notion that has been the basis of several practical deployments in industry [35, 82, 43, 9, 29] and the U.S. Census [3]. A classical algorithm—that predates DP and was initially designed to eliminate evasive answer biases in survey aggregation—is Randomized Response (RR) [96]: when the input is an element from a finite alphabet, the output is equal to the input with a certain probability, and is a uniform random other element from the alphabet with the remaining probability. This simple algorithm is shown to satisfy the strong notion of local DP [38, 58], whereby the response of each user is protected, in contrast with the so-called central DP setting where a curator has access to the raw user data, and only the output of the curator is required to be DP. We note that schemes building on RR have been studied in several previous works on DP estimation (e.g., [30, 57]), and have been deployed in practice [82]. Meanwhile, the large error incurred by RR (e.g., [17]) has stimulated significant research aiming to improve its accuracy, mostly by relaxing to weaker privacy models (e.g., [36, 24, 4]). In this work, we use a different approach and seek to improve RR by leveraging available prior information. (A recent work of Liu et al. [63] also used priors to improve accuracy, but in the context of the DP multiplicative weights algorithm, which applies in the central DP setting.) The prior information can consist of domain-specific knowledge, (models trained on) publicly available data, or historical
runs of a training algorithm. Our algorithm is presented in Section 3 (Algorithm 2). At a high level, given a prior distribution p on the alphabet, the algorithm uses p to prune the alphabet. If the prior is reliable and even if the alphabet is only minimally pruned, the probability that the output equals the input is larger than when RR is applied to the entire alphabet. On the other hand, if the prior is uniform over the entire alphabet, then our algorithm recovers the classical RR. To implement the above recipe, one needs to specify how to effect the pruning using p. It turns out that the magnitude of pruning can itself vary depending on p, but we can obtain a closed-form formula for determining this. Interestingly, by studying a suitable linear program, we show that the resulting RRWithPrior strategy is optimal in that among all ε-DP algorithms, it maximizes the probability that the output equals the input when the latter is sampled from p (Theorem 3).
# 1.1 Applications to Learning with Label Differential Privacy
There have been a great number of papers over the last decade that developed DP machine learning algorithms (e.g., [19, 102, 88, 83–85, 78]). In the case of deep learning, the seminal work of Abadi et al. [2] introduced a DP training framework (DP-SGD) that was integrated into TensorFlow [80] and PyTorch [92]. Despite numerous followup works, including, e.g., [75–77, 67, 98, 71, 20, 93], and extensive efforts, the accuracy of models trained with DP-SGD remains significantly lower than that of models trained without DP constraints. Notably, for the widely considered CIFAR-10 dataset, the highest reported accuracy for DP models is 69.3% [93], which strikingly relies on handcrafted visual features despite that in non-private scenarios learned features long been shown to be superior. Even using pre-training with external (CIFAR-100) data, the best reported DP accuracy, 73%2, is still far below the non-private baselines (> 95%). The performance gap becomes a roadblocker for many real-world applications to adopt DP. In this paper, we focus on a more restricted, but important, special case where the DP guarantee is only required to hold with respect to the labels, as described next. In the label differential privacy (LabelDP) setting, the labels are considered sensitive, and their privacy needs to be protected, while the input points are not sensitive. This notion has been studied in the PAC setting [18, 13] and for the particular case of sparse linear regression [94], and it captures several practical scenarios. Examples include: (i) computational advertising where the impressions are known to the Ad Tech3, and thus considered non-sensitive, while the conversions reveal user interest and are thus private (see, e.g., Nalpas and Dutton [70] and [8]), (ii) recommendation systems where the choices are known, e.g., to the streaming service provider, but the user ratings are considered sensitive, and (iii) user surveys and analytics where demographic information (e.g., age, gender) is non-sensitive but income is sensitive—in fact, this was the motivating reason for Warner [96] to propose RR many decades ago! We present a novel multi-stage algorithm (LP-MST) for training deep neural networks with LabelDP that builds on top of RRWithPrior (see Section 3 and Algorithm 3), and we benchmark its empirical performance (Section 5) on multiple datasets, domains, and architectures, including the following. • On CIFAR-10, we show that it achieves 20% higher accuracy than DP-SGD4. • On the more challenging CIFAR-100, we present the first non-trivial DP learning results. • On MovieLens, which consists of user ratings of movies, we show improvements via LP-MST. In some applications, domain specific algorithms can be used to obtain priors directly without going through multi-stage training. For image classification problems, we demonstrate how priors computed from a (non-private) self-supervised learning [22, 23, 44, 53, 16] phase on the input images can be used to achieve higher accuracy with a LabelDP guarantee with extremely small privacy budgets (ε ≤0.1, see Section 5.2 for details). We note that due to the requirement of DP-SGD to compute and clip per-instance gradient, it remains technically challenging to scale to larger models or mini-batch sizes, despite numerous attempts to minigate this problem [42, 5, 26, 90]. On the other hand, our formulation allows us to use state-of2For DP parameters of ε = 8 and δ = 10−5, cited from Abadi et al. [2, Figure 6]. For a formal definition of DP, we refer the reader to Definition 2.1. 3Ad tech (abbreviating Advertising Technology) comprises the tools that help agencies and brands target, deliver, and analyze their digital advertising efforts; see, e.g., blog.hubspot.com/marketing/what-is-ad-tech. 4We remark that the notion of ε-DP in [2, 76, 75] is not directly comparable to ε-Label DP in our work in that they use the addition/removal notion whereas we use the substitution one. Please see the Supplementary Material for more discussion on this.
the-art deep learning architectures such as ResNet [51]. We also stress that our LP-MST algorithm goes beyond deep learning methods that are robust to label noise. (See [87] for a survey of the latter.) Our empirical results suggest that protecting the privacy of labels can be significantly easier than protecting the privacy of both inputs and labels. We find further evidence to this by showing that for the special case of stochastic convex optimization (SCO), the sample complexity of algorithms privatizing the labels is much smaller than that of algorithms privatizing both labels and inputs; specifically, we achieve dimension-independent bounds for LabelDP (Section 6). We also show that a good prior can ensure smaller population error for non-convex loss. (Details are in the Supplementary Material.)
# 2 Preliminaries
For any positive integer K, let [K] := {1, . . . , K}. Randomized response (RR) [96] is the following: let ε ≥0 be a parameter and let y ∈[K] be the true value known to RRε. When an observer queries the value of y, RRε responds with a random draw ˜y from the following probability distribution:
In this paper, we focus on the application of learning with label differential privacy. We recall the definition of differential privacy (DP), which is applicable to any notion of neighboring datasets. For a textbook reference, we refer the reader to Dwork and Roth [32]. Definition 2.1 (Differential Privacy (DP) [33, 34]). Let ε, δ ∈R≥0. A randomized algorithm A taking as input a dataset is said to be (ε, δ)-differentially private ((ε, δ)-DP) if for any two neighboring datasets D and D′, and for any subset S of outputs of A, it is the case that Pr[A(D) ∈S] ≤ eε · Pr[A(D′) ∈S] + δ. If δ = 0, then A is said to be ε-differentially private (ε-DP). When applied to machine learning methods in general and deep learning in particular, DP is usually enforced on the weights of the trained model [see, e.g., 19, 59, 2]. In this work, we focus on the notion of label differential privacy. Definition 2.2 (Label Differential Privacy). Let ε, δ ∈R≥0. A randomized training algorithm A taking as input a dataset is said to be (ε, δ)-label differentially private ((ε, δ)-LabelDP) if for any two training datasets D and D′ that differ in the label of a single example, and for any subset S of outputs of A, it is the case that Pr[A(D) ∈S] ≤eε · Pr[A(D′) ∈S] + δ. If δ = 0, then A is said to be ε-label differentially private (ε-LabelDP).
# 3 Randomized Response with Prior
In many real world applications, a prior distribution about the labels could be publicly obtained from domain knowledge and help the learning process. In particular, we consider a setting where for each (private) label y in the training set, there is an associated prior p = (p1, . . . , pK). The goal is to output a randomized label ˜y that maximizes the probability that the output is correct (or equivalently maximizes the signal-to-noise ratio), i.e., Pr[y = ˜y]. The privacy constraint here is that the algorithm should be ε-DP with respect to y. (It need not be private with respect to the prior p.) We first describe our algorithm RRWithPrior by assuming access to such priors.
# 3.1 Algorithm: RRWithPrior
We build our RRWithPrior algorithm with a subroutine called RRTop-k, as shown in Algorithm 1, which is a modification of randomized response where we only consider the set of k labels i with largest pi. Then, if the input label y belongs to this set, we use standard randomized response on this set. Otherwise, we output a label from this set uniformly at random. The main idea behind RRWithPrior is to dynamically estimate an optimal k∗based on the prior p, and run RRTop-k with k∗. Specifically, we choose k∗by maximizing Pr[RRTop-k(y) = y]. It is not
(1)
Algorithm 1 RRTop-k
Input: A label y ∈[K]
Parameters: k ∈[K], prior p = (p1, . . . , pK)
1. Let Yk be the set of k labels with maximum prior probability (with ties broken arbritrarily).
2. If y ∈Yk, then output y with probability
eε
eε+k−1 and output y′ ∈Yk \ {y} with probability
1
eε+k−1.
3. If y ̸∈Yk, output an element from Yk uniformly at random.
hard to see that this expression is exactly equal to
eε
eε+k−1 ·
��
˜y∈Yk p˜y
�
if y ∼p. RRWithPrior is
presented in Algorithm 2.
Algorithm 2 RRWithPrior
Input: A label y ∈[K]
Parameters: prior p = (p1, . . . , pK)
1. For k ∈[K]:
(a) Compute wk :=
eε
eε+k−1 ·
��
˜y∈Yk p˜y
�
, where Yk is the set of k labels with maximum
prior probability (ties broken arbritrarily).
2. Let k∗= arg maxk∈[K] wk.
3. Return an output of RRTop-k (y) with k = k∗.
It is not hard to show that RRTop-k is ε-DP. Lemma 1. RRTop-k is ε-DP.
The privacy guarantee of RRWithPrior follows immediately from that of RRTop-k (Lemma 1) since our choice of k does not depend on the label y: Corollary 2. RRWithPrior is ε-DP.
For learning with a LabelDP guarantee, we first use RRWithPrior to query a randomized label for each example of the training set, and then apply a general learning algorithm that is robust to random label noise to this dataset. Note that unlike DP-SGD [2] that makes new queries on the gradients in every training epoch, we query the randomized label once and reuse it in all the training epochs.
# 3.2 Optimality of RRWithPrior
In this section we will prove the optimality of RRWithPrior. For this, we will need additional notation. For any algorithm R that takes as input a label y and outputs a randomized label ˜y, we let Objp(R) denote the probability that the output label is equal to the input label y when y is distributed as p; i.e., Objp(R) = Pry∼p[R(y) = y], where the distribution of y ∼p is Pr[y = i] = pi for all i ∈[K]. The main result of this section is that, among all ε-DP algorithms, RRWithPrior maximizes Objp(R), as stated more formally next. Theorem 3. Let p be any probability distribution on [K] and R be any ε-DP algorithm that randomizes the input label given the prior p . We have that
In this section we will prove the optimality of RRWithPrior. For this, we will need additional notation. For any algorithm R that takes as input a label y and outputs a randomized label ˜y, we let Objp(R) denote the probability that the output label is equal to the input label y when y is distributed as p; i.e., Objp(R) = Pry∼p[R(y) = y], where the distribution of y ∼p is Pr[y = i] = pi for all i ∈[K].
The main result of this section is that, among all ε-DP algorithms, RRWithPrior maximizes Objp(R), as stated more formally next. Theorem 3. Let p be any probability distribution on [K] and R be any ε-DP algorithm that randomizes the input label given the prior p . We have that
Before we proceed to the proof, we remark that our proof employs a linear program (LP) to characterize the optimal mechanisms; a generic form of such LPs has been used before in [48, 41]. However, these works focus on different problems (linear queries) and their results do not apply here.
Before we proceed to the proof, we remark that our proof employs a linear program (LP) to characterize the optimal mechanisms; a generic form of such LPs has been used before in [48, 41]. However, these works focus on different problems (linear queries) and their results do not apply here. Proof of Theorem 3. Consider any ε-DP algorithm R, and let q˜y|y denote Pr[R(y) = ˜y]. Observe that Objp(R) = � y∈[k] py · qy|y.
Proof of Theorem 3. Consider any ε-DP algorithm R, and let q˜y|y denote Pr[R(y) = ˜y]. Observe that Objp(R) = � y∈[k] py · qy|y.
� ˜y∈[K] q˜y|y = 1, ∀y ∈[K], and q˜y|y ≥0, ∀˜y, y ∈[K].
Finally, the ε-DP guarantee of R implies that
  Combining the above, Objp(R) is upper-bounded by the optimum of the following linear program (LP), which we refer to as LP1:
Notice that constraints (2) and (3) together imply that:
In other words, the optimum of LP1 is at most the optimum of the following LP that we call LP2:
An optimal solution to LP2 must be a vertex (aka extreme point) of the polytope defined by (4) and (5). Recall that an extreme point of a K-dimensional polytope must satisfy K independent constraints with equality. In our case, this means that one of the following occurs: • Inequality (5) is satisfied with equality for all y ∈[K] resulting in the all-zero solution (whose objective is zero), or, • For some non-empty subset Y ⊆[K], inequality (4) is satisfied with equality for all y ∈Y , and inequality (5) is satisfied with equality for all y ∈[K] \ Y . This results in
An optimal solution to LP2 must be a vertex (aka extreme point) of the polytope defined by (4) and (5). Recall that an extreme point of a K-dimensional polytope must satisfy K independent constraints with equality. In our case, this means that one of the following occurs:
• Inequality (5) is satisfied with equality for all y ∈[K] resulting in the all-zero solution (whose objective is zero), or, • For some non-empty subset Y ⊆[K], inequality (4) is satisfied with equality for all y ∈Y , and inequality (5) is satisfied with equality for all y ∈[K] \ Y . This results in
In conclusion, we have that
where the last two equalities follow from our definitions of Yk and wk. Notice that Objp(RRWithPrior) = maxk∈[K] wk. Thus, we get that Objp(RRWithPrior) ≥Objp(R) as desired.
where the last two equalities follow from our definitions of Yk and wk. Notice that Objp(RRWithPrior) = maxk∈[K] wk. Thus, we get that Objp(RRWithPrior) ≥Objp(R) as desired.
(2) (3)
∀y ∈[K].
(5)
# 4 Application of RRWithPrior: Multi-Stage Training
Our RRWithPrior algorithm requires publicly available priors, which could usually be obtained from domain specific knowledge. In this section, we describe a training framework that bootstraps from a uniform prior, and progressively learns refined priors via multi-stage training. This general framework can be applied to arbitrary domains even when no public prior distributions are available. Specifically, we assume that we have a training algorithm A that outputs a probabilistic classifier which, on a given unlabeled sample x, can assign a probability py to each class y ∈[K]. We partition our dataset into subsets S(1), . . . , S(T ), and we start with a trivial model M (0) that outputs equal probabilities for all classes. At each stage t ∈[T], we use the most recent model M (t−1) to assign the probabilities (p1, . . . , pK) for each sample xi from S(t). Applying RRWithPrior with this prior on the true label yi, we get a randomized label ˜yi for xi. We then use all the samples with randomized labels obtained so far to train the model M (t). The full description of our LP-MST (Label Privacy Multi-Stage Training) method is presented in Algorithm 3. We remark here that the partition S(1), . . . , S(T ) can be arbitrarily chosen, as long as it does not depend on the labels y1, . . . , yn. We also stress that the training algorithm A need not be private. We use LP-1ST to denote our algorithm with one stage, LP-2ST to denote our algorithm with two stages, and so on. We also note that LP-1ST is equivalent to using vanilla RR. The tth stage of a multi-stage algorithm is denoted stage-t.
Algorithm 3 Multi-Stage Training (LP-MST)
Input: Dataset S = {(x1, y1), . . . , (xn, yn)}
Parameters: Number T of stages, training algorithm A
1. Partition S into S(1), . . . , S(T )
2. Let M (0) be the trivial model that always assigns equal probability to each class.
3. For t = 1 to T:
(a) Let ˜S(t) = ∅.
(b) For each (xi, yi) ∈S(t):
i. Let p = (p1, . . . , pK) be the probabilities predicted by M (t) on xi.
ii. Let ˜yi = RRWithPriorp(yi).
iii. Add (xi, ˜yi) to ˜S(t).
(c) Let M (t) be the model resulting from training on ˜S(1) ∪· · · ∪˜S(t) using A.
4. Output M (T ).
�� � Consider any two datasets D, D′ that differ on a single user’s label; suppose this user is j and that the user belongs to partition ℓ∈[T]. Then, the above expression for D and that for D′ are the same in all but one term: Pr � ˜yj = zj ��M (ℓ−1) = m(ℓ−1)� , which is the probability that RRWithPriorm(ℓ−1)(xi) outputs zi. Since RRWithPrior is ε-DP, we can conclude that the ratio between the two probabilities is at most eε as desired.
We stress that this observation holds because each sensitive label yi is only used once in Line 3(b)ii of Algorithm 3, as the dataset S is partitioned at the beginning of the algorithm. As a result, since each
Table 1: Test accuracy (%) on CIFAR-10. The baseline performances taken from previously published results correspond to (ε, δ)-DP with δ = 10−5. The star⋆indicates the use of CIFAR-100 pre-trained representations.
<div style="text-align: center;">Table 1: Test accuracy (%) on CIFAR-10. The baseline performances taken from previously published results correspond to (ε, δ)-DP with δ = 10−5. The star⋆indicates the use of CIFAR-100 pre-trained</div>
Algorithm
ε = 1
ε = 2
ε = 3
ε = 4
ε = 6
ε = 8
ε = ∞
DP-SGD w/ pre-train⋆[2]
67
70
73
80
DP-SGD [77]
61.6(ε=7.53)
76.6
Tempered Sigmoid [77]
66.2(ε=7.53)
Yu et al. [98]
44.3(ε=6.78)
Nasr et al. [71]
55
Chen and Lee [20]
53
ScatterNet+CNN [93]
69.3
LP-1ST
59.96
82.38
89.89
92.58
93.58
94.70
94.96
LP-2ST
63.67
86.05
92.19
93.37
94.26
94.52
-
LP-1ST w/ pre-train⋆
67.64
83.99
90.24
92.83
94.02
94.96
95.25
LP-2ST w/ pre-train⋆
70.16
87.22
92.12
93.53
94.41
94.59
-
 ∞
Algorithm
ε = 3
ε = 4
ε = 5
ε = 6
ε = 8
LP-1ST
20.96
46.28
61.38
68.34
73.59
LP-2ST
28.74
50.15
63.51
70.58
74.14
stage is ε-LabelDP, the entire algorithm is also ε-LabelDP. This is known as (an adaptive version of a) parallel composition [68]. Finally, we point out that the running time of our RRWithPrior algorithm is quasi-linear in K (the time needed to sort the prior). This is essentially optimal within multistage training, since O(K) time will be required to write down the prior after each stage. Moreover, for reasonable values of K, the running time will be dominated by back-propagation for gradient estimation. Moreover, the focus of the current work is on small to modest label spaces (i.e., values of K).
# 5 Empirical Evaluation
We evaluate RRWithPrior on standard benchmark datasets that have been widely used in previous works on private machine learning. Specifically, in the first part, we study our general multi-stage training algorithm that boostraps from a uniform prior. We evaluate it on image classification and collaborative filtering tasks. In the second part, we focus on image classification only and use domainspecific techniques to obtain priors for RRWithPrior. We use modern neural network architectures (e.g., ResNets [51]) and the mixup [101] regularization for learning with noisy labels. Please see the Supplementary Material for full details on the datasets and the experimental setup.
# 5.1 Evaluation with Multi-Stage Training
CIFAR-10 [60] is a 10-class image classification benchmark dataset. We evaluate our algorithm and compare it to previously reported DP baselines in Table 1. Due to scalability issues, previous DP algorithms could only use simplified architectures with non-private accuracy significantly below the state-of-the-art. Moreover, even when compared to those weaker non-private baselines, a large performance drop is observed in the private models. In contrast, we use ResNet18 with 95% nonprivate accuracy. Overall, our algorithms improve the previous state-of-the-art by a margin of 20% across all ε’s. Abadi et al. [2] treated CIFAR-100 as public data and use it to pre-train a representation to boost the performance of DP-SGD. We also observe performance improvements with CIFAR-100 pre-training (Table 1, bottom 2 rows). But even without pre-training, our results are significantly better than DP-SGD even with pre-training.
<div style="text-align: center;">Table 3: Experiments on MovieLens-1M. The numbers show the test RM</div>
Algorithm
ε = 1
ε = 2
ε = 3
ε = 4
ε = 8
ε = ∞
LP-1ST
1.122
0.981
0.902
0.877
0.867
0.868
LP-2ST
1.034
0.928
0.891
0.874
0.865
Gaussian DP [15]
0.915 (ε ≥10)
In Table 2 we also show results on CIFAR-100, which is a more challenging variant with 10× more classes. To the best of our knowledge, these are the first non-trivial reported results on CIFAR-100 for DP learning. For ε = 8, our algorithm is only 2% below the non-private baseline. In addition, we also evaluate on MovieLens-1M [49], which contains 1 million anonymous ratings of approximately 3, 900 movies, made by 6,040 MovieLens users. Following [15], we randomly split the data into 80% train and 20% test, and show the test Root Mean Square Error (RMSE) in Table 3. Results on MNIST [61], Fashion MNIST [97], and KMNIST [25], and comparison to more baselines can be found in the Supplementary Material. In all the datasets we evaluated, our algorithms not only significantly outperform the previous methods, but also greatly shrink the performance gap between private and non-private models. The latter is critical for applications of deep learning systems in real-world tasks with privacy concerns. Beyond Two Stages. In Figure 1(a), we report results on LP-MST with T > 2. For the cases we tried, we consistently observe 1–2% improvements on test accuracy when going from LP-2ST to LP-3ST. In our preliminary experiments, going beyond T > 4 stages leads to diminishing returns on some datasets.
Beyond Two Stages. In Figure 1(a), we report results on LP-MST with T > 2. For the cases we tried, we consistently observe 1–2% improvements on test accuracy when going from LP-2ST to LP-3ST. In our preliminary experiments, going beyond T > 4 stages leads to diminishing returns on some datasets.
# 5.2 Evaluation with Domain-Specific Priors
The multi-stage training framework evaluated in the previous section is a general domain-agnostic algorithm that bootstraps itself from uniform priors. In some cases, domain-specific priors can be obtained to further improve the learning performance. In this section, we focus on image classification applications, where new advances in self-supervised learning (SSL) [22, 23, 44, 53, 16] show that high-quality image representations could be learned on large image datasets without using the class labels. In the setting of LabelDP, the unlabeled images are considered public data, so we design an algorithm to use SSL to obtain priors, which is then fed to RRWithPrior for discriminative learning. Specifically, we partition the training examples into groups by clustering using their representations extracted from SSL models. We then query a histogram of labels for each group via discrete Laplace mechanism (aka Geometric Mechanism) [41]. If the groups are largely homogeneous, consisting of mostly examples from the same class, then we can make the histogram queries with minimum privacy budget. The queried histograms are used as label priors for all the points in the group. Figure 1(b) shows the results on two different SSL representations: BYOL [44], trained on unlabeled CIFAR-10 images and DINO [16], trained on ImageNet [27] images. Comparing to the baseline, the SSL-based priors significantly improves the model performance with small privacy budgets. Note that since the SSL priors are not true priors, with large privacy budget (ε = 8), it actually underperforms the uniform prior. But in most real world applications, small ε’s are generally more useful.
# 6 Theoretical Analysis
Previous works have shown that LabelDP can be provably easier than DP in certain settings; specifically, in the PAC learning setting, Beimel et al. [13] proved that finite VC dimension implies learnability by LabelDP algorithms, whereas it is known that this is not sufficient for DP algorithms [7]. We extend the theoretical understanding of this phenomenon to the stochastic convex optimization (SCO) setting. Specifically, we show that, by applying RR on the labels and running SGD on top of the resulting noisy dataset with an appropriate debiasing of the noise, one can arrive at the following dimension-independent excess population loss.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f371/f3718217-3389-4d0d-9132-ece58c261df0.png" style="width: 50%;"></div>
Figure 1: (a) Test accuracy (%) on various datasets with LP-MST for T > 2. The curve “CIFAR-10 w/ pre-train” is using CIFAR-100 as public data to pre-train the model. (b) RRWithPrior with priors obtained from histogram query based on clustering in various SSL representations. We also plot recent results (ClusterRR) from Esfandiari et al. [37], which is a clustering based LabelDP algorithm. Theorem 5 (Informal). For any ε ∈(0, 1), there is an ε-LabelDP algorithm for stochastic convex optimization with excess population loss ˜O � DL · K ε√n � where D denotes the diameter of the parameter space and L denotes the Lipschitz constant of the loss function. The above excess population loss can be compared to that of Bassily et al. [12], who gave an (ε, δ)DP algorithm with excess population loss OD,L � 1 √n + √p εn � , where p denote the dimension of the parameter space; this bound is also known to be tight in the standard DP setting. The main advantage of our guarantee in Theorem 5 is that it is independent of the dimension p. Furthermore, we show that our bound is tight up to polylogarithmic factors and the dependency on the number of classes K. The above result provides theoretical evidence that running RR on the labels and then training on this noisy dataset can be effective. We can further extend this to the setting where, instead of running RR, we run RRTop-k before running the aforementioned (debiased) SGD, although—perhaps as expected—our bound on the population loss now depends on the quality of the priors. Corollary 6 (Informal). Suppose that we are given a prior px for every x and let Y x k denote the set of top-k labels with respect to px. Then, for any ε ∈(0, 1), there is an ε-LabelDP algorithm for stochastic convex optimization with excess population loss ˜O � DL · � k ε√n + Pr(x,y)∼D[y /∈Y x k ] �� where D, L are as defined in Theorem 5 and D is the data distribution.
When our top-k set is perfect (i.e., y always belongs to Y x k ), the bound reduces to that of Theorem 5, but with the smaller k instead of K. Moreover, the second term is, in some sense, a penalty we pay in the excess population loss for the inaccuracy of the top-k prior. We defer the formal treatment and the proofs to the Supplementary Material, in which we also present additional generalization results for non-convex settings. Note that Corollary 6 is not in the exact setup we run in experiments, where we dynamically calculate an optimal k for each x given generic priors (via RRWithPrior), and for which the utility is much more complicated to analyze mathematically. Nonetheless, the above corollary corroborates the intuition that a good prior helps with training.
# 7 Conclusions and Future Directions
In this work, we introduced a novel algorithm RRWithPrior (which can be used to improve on the traditional RR mechanism), and applied it to LabelDP problems. We showed that prior information can be incorporated to the randomized label querying framework while maintaining privacy constraints We demonstrated two frameworks to apply RRWithPrior: (i) a general multi-stage training algorithm LP-MST that bootstraps from uniform priors and (ii) an algorithm that build priors from clustering with SSL-based representations. The former is general purpose and can be applied to tasks even when no
domain-specific priors are available, while the latter uses a domain-specific algorithm to extract priors and performs well even with very small privacy budget. As summarized by the figure on the right, in both cases, by focusing on LabelDP, our RRWithPrior significantly improved the model performance of previous state-of-the-art DP models that aimed to protect both the inputs and outputs. We note that, following up on our work, additional results on deep learning with LabelDP were obtained [66, 100]. The narrowed performance gap between private and non-private models is vital for adding DP to real world deep learning models. We nevertheless stress that our algorithms only protect the labels but not the input points, which might not constitute a sufficient privacy protection in all settings.
Our work opens up several interesting questions. Firstly, note that our multi-stage training procedure uses very different ingredients than those of Abadi et al. [2] (which employ DP-SGD, privacy amplification by subsampling, and Renyi accounting); can these tools be used to further improve LabelDP? Secondly, while our procedure can be implemented in the most stringent local DP setting5 [58], can it be improved in the weaker central (aka trusted curator) DP model, assuming the curator knows the prior? Thirdly, while our algorithm achieves pure DP (i.e., δ = 0), is higher accuracy possible for approximate DP (i.e., δ > 0)?
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5856/585685ee-4261-425b-9d88-570addaca55b.png" style="width: 50%;"></div>
# Acknowledgements
The authors would like to thank Sami Torbey for very helpful feedback on an early version of this work. At MIT, Noah Golowich was supported by a Fannie and John Hertz Foundation Fellowship and an NSF Graduate Fellowship.
# References
# Supplementary Material for “Deep Learning with Label Differential Privacy”
# A Missing Proofs
# A.1 Proof of Lemma 1
Proof of Lemma 1. Consider any inputs y, y′ ∈ [K] and any possible output ˜y ∈ Yk. Pr[RRTop-k(y) = ˜y] is maximized when y = ˜y, whereas Pr[RRTop-k(y′) = ˜y] is minimized when y′ ∈Yk \ {˜y}. This implies that
Thus, RRTop-k is ε-DP as desired.
# B Details of the Experimental Setup
Datasets. We evaluate our algorithms on the following image classification datasets:
• MNIST [61], 10 class classification of hand written digits, based on inputs of 28 × 28 gray scale images. The training set contains 60,000 examples and the test set contains 10,000. • Fashion MNIST [97], 10 class classification of Zalando’s article images. The dataset size and input format are the same as MNIST. • KMNIST [25], 10 class classification of Hiragana characters. The dataset size and the input format are the same as MNIST. • CIFAR-10/CIFAR-100 [60] are 10 class and 100 class image classification datasets, respectively. Both datasets contains 32 × 32 color images, and both have a training set of size 50,000 and a test set of size 10,000. • MovieLens [49] contains a set of movie ratings from the MovieLens users. It was collected and maintained by a research group (GroupLens) at the University of Minnesota. There are 5 versions: “25m”, “latest-small”, “100k”, “1m”, “20m”. Following Bu et al. [15], we use the “1m” version, which the largest MovieLens dataset that contains demographic data. Specifically, it contains 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users, with some meta data such as gender and zip code.
Architectures. On CIFAR-10/CIFAR-100, we use ResNet [51], which is a Residual Network architecture widely used in the computer vision community. In particular, we use ResNet18 V2 [52]. Note the standard ResNet18 is originally designed for ImageNet scale (image size 224 × 224). When adapting to CIFAR (image size 32 × 32), we replace the initial block with 7 × 7 convolution and 3 × 3 max pooling with a single 3 × 3 convolution (with stride 1) layer. The upper layers are kept the same as the standard ImageNet ResNet18. On MNIST, Fashion MNIST, and KMNIST, we use a simplified Inception [91] model suitable for small image sizes, and defined as follows: Inception :: Conv(3×3, 96) →S1 →S2 →S3 →GlobalMaxPool →Linear. S1 :: Block(32, 32) →Block(32, 48) →Conv(3×3, 160, Stride=2). S2 :: Block(112, 48) →Block(96, 64) →Block(80, 80) →Block (48, 96) → Conv(3×3, 240, Stride=2). S3 :: Block(176, 160) →Block(176, 160). Block(C1, C2) :: Concat(Conv(1×1, C1), Conv(3×3,C2)). Conv :: Convolution →BatchNormalization →ReLU.

multi-class classification. During evaluation, we output the average rating according to the softmax probabilities output by the trained model.
probabilities output by the trained model. Training Procedures. On MNIST, Fashion MNIST, and KMNIST, we train the models with minibatch SGD with batch size 265 and momentum 0.9. We run the training for 40 epochs (for multi-stage training, each stage will run 40 epochs separately), and schedule the learning rate to linearly grow from 0 to 0.02 in the first 15% training iterations, and then linearly decay to 0 in the remaining iterations. On CIFAR-10, we use batch size 512 and momentum 0.9, and train for 200 epochs. The learning rate is scheduled according to the widely used piecewise constant with linear rampup scheme. Specifically, it grows from 0 to 0.4 in the first 15% training iterations, then it remains piecewise constant with a decay factor of 10 at the 30%, 60%, and 90% training iterations, respectively. The CIFAR-100 setup is similar to CIFAR-10 except that we use a batch size 256 and a peak learning rate 0.2. MovieLens experiments are trained similarly, but with batch size 128. On all datasets, we optimize the cross entropy loss with an ℓ2 regularization (coefficient 10−4). All the networks are randomly initialized at the beginning of the training. For the experiment on CIFAR-10 where we explicitly study the effect of pre-training to compare with previous methods that use the same technique, we train a (non-private) ResNet18 on the full CIFAR-100 training set and initialize the CIFAR-10 model with the pre-trained weights. The classifier is still randomly initialized because there is no clear correspondence between the 100 classes of CIFAR-100 and the 10 classes of CIFAR-10. The remaining configuration remains the same as in the experiments without pre-training. In particular, we did not freeze the pre-trained weights. We apply standard data augmentations, including random crop, random left-right flip, and random cutout [28], to all the datasets during training. We implement our algorithms in TensorFlow [1], and train all the models on NVidia Tesla P100 GPUs. Learning with Noisy Labels. Standard training procedures tend to overfit to the label noise and generalize poorly on the test set when some of the training labels are randomly flipped. We apply mixup [101] regularization, which generates random convex combinations of both the inputs and the (one-hot encoded) labels during training. It is shown that mixup is resistant to random label noise. Note that our framework is generic and in principle any robust training technique could be used. We have chosen mixup for its simplicity, but there has been a rich body of recent work on deep learning methods with label noise, see, e.g., [55, 46, 99, 21, 104, 74, 69, 64, 105, 56, 50, 47, 65, 86, 79, 87] and the references therein. Potentially with more advanced robust training, even higher performance could be achieved. Multi-Stage Training. There are a few implementation enhancements that we find useful for multi-stage training. For concreteness, we discuss them for LP-2ST. First, we find it helps to initialize the stage-2 training with the models trained in stage-1. This is permitted as the stage-1 model is trained on labels that are queried privately. Moreover, we can reuse those labels queried in stage-1 and train stage-2 on a combined dataset. Although the subset of data from stage-1 is noisier, we find that it generally helps to have more data, especially when we reduce the noise of stage-1 data by using the learned prior model. Specifically, for each sample (x, ˜y) in the stage-1 data, where ˜y is the private label queried in stage-1, we make a prediction on x using the model trained in stage-1; if ˜y is not in the top k predicted classes, we will exclude it from the stage-2 training. Here k is simply set to the average k obtained when running RRWithPrior to query labels on the data held out for stage-2. Similar ideas apply to training with more stages. For example, in LP-3ST, stage-3 training could use the model trained in stage-2 as initialization, and use it to filter the queried labels in stage-1 and stage-2 that are outside the top k prediction, and then train on the combined data of all 3 stages. Priors from Self-supervised Learning. Recent advances in self-supervised learning (SSL) [22, 23, 44, 53, 16] show that representations learned from a large collection of unlabeled but diverse images could capture useful semantic information and can be finetuned with labels to achieve classification performance on par with the state-of-the-art fully supervised learned models. We apply SSL algorithms to extract priors for image classification problems, with the procedure described in Algorithm 4. Specifically, we choose two recent SSL algorithms: BYOL [44] and DINO [16]. For BYOL, we train the SSL model using the (unlabeled) CIFAR-10 images only, as a demonstration without using
Multi-Stage Training. There are a few implementation enhancements that we find useful for multi-stage training. For concreteness, we discuss them for LP-2ST. First, we find it helps to initialize the stage-2 training with the models trained in stage-1. This is permitted as the stage-1 model is trained on labels that are queried privately. Moreover, we can reuse those labels queried in stage-1 and train stage-2 on a combined dataset. Although the subset of data from stage-1 is noisier, we find that it generally helps to have more data, especially when we reduce the noise of stage-1 data by using the learned prior model. Specifically, for each sample (x, ˜y) in the stage-1 data, where ˜y is the private label queried in stage-1, we make a prediction on x using the model trained in stage-1; if ˜y is not in the top k predicted classes, we will exclude it from the stage-2 training. Here k is simply set to the average k obtained when running RRWithPrior to query labels on the data held out for stage-2. Similar ideas apply to training with more stages. For example, in LP-3ST, stage-3 training could use the model trained in stage-2 as initialization, and use it to filter the queried labels in stage-1 and stage-2 that are outside the top k prediction, and then train on the combined data of all 3 stages. Priors from Self-supervised Learning. Recent advances in self-supervised learning (SSL) [22, 23, 44, 53, 16] show that representations learned from a large collection of unlabeled but diverse images could capture useful semantic information and can be finetuned with labels to achieve classification performance on par with the state-of-the-art fully supervised learned models. We apply SSL algorithms to extract priors for image classification problems, with the procedure described in Algorithm 4. Specifically, we choose two recent SSL algorithms: BYOL [44] and DINO [16]. For BYOL, we train the SSL model using the (unlabeled) CIFAR-10 images only, as a demonstration without using
Algorithm 4
Input: Training set D = {(xi, yi)}n
i=1, cluster count C, privacy budget for priors εp, trained SSL model fSSL
1. Initialize P ←1/K ones(n, K) as the uniform priors.
2. Extract SSL features F = {fSSL(xi) : (xi, yi) ∈D}.
3. Run k-means algorithms to partition F into C groups.
4. For each c = 1 to C:
(a) Compute histogram of classes Hc ∈NK
≥0 according to the labels of examples in the c-th group.
(b) Get a private histogram query ˜Hc ←Hc+ scipy.stats.dlaplace.rvs(εp/2, K), via the
discrete Laplace mechanism.
(c) Get a prior via normalization: pc = max( ˜Hc, 0)/ �K
k=1 max( ˜Hc[k], 0).
(d) For each example i in group c, assign P[i, :] ←pc.
5. Output P.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/249d/249d9af0-7c08-444d-b135-568684185223.png" style="width: 50%;"></div>
Figure 2: Accuracy evaluated on CIFAR-10 test set, of private histogram querying with kmeans clustering on self-supervised learning based features learned by (a) BYOL [44] on CIFAR-10 and (b) DINO [16] on ImageNet.
external data. For DINO, we use the models pre-trained on (unlabeled) ImageNet [27] images. Since ImageNet is a much larger and more diverse dataset than CIFAR-10, the SSL representations are also more capable of capturing the semantic information. Note the ImageNet images are of higher resolution and resized to 224×224 during training. To extract features for 32×32 CIFAR-10 images, we simply upscale the images to 224 × 224 before feeding into the trained neural network. We choose relatively large cluster sizes so that the private histogram query is more robust to the added discrete Laplace noise. In particular, we found C = 100 clusters for BYOL representations and C = 50 clusters for DINO representations achieve a good balance of robustness and accuracy. Since εp will be subtracted from the privacy budget for RRWithPrior, we simply choose the smallest εp without causing too much deterioration of the priors. In our experiments, we set εp = 0.05 for BYOL and εp = 0.025 for DINO. Note the model accuracy could potentially be further boosted by choosing C and εp adaptively according to the overall privacy budget. In the following, we provide a simple study to show how the interplay between εp and C affects the accuracy of the histogram queries. To compute an accuracy measure on the test set, we extract features using a SSL learned models on both training and test set. A k-means clustering algorithm is run on the joint set of training and test features. For each cluster, we apply the discrete Laplace mechanism to make a private histogram of class distributions from only the training examples in that cluster. The class with the maximum votes are then used as predicted labels for all the test examples in the cluster, and compared with the true test labels to calculate the accuracy. Figure 2 shows the accuracy with the two different SSL features under different privacy budgets (ε) for making the histogram queries. As expected, the accuracy is higher with smaller clusters, but at the same time sensitive to noise introduced by the Geometric Mechanism when the privacy budget is small.
<div style="text-align: center;">Table 4: Test accuracy (%) on MNIST and Fashion MNIST. The baseline performances taken from previously published results correspond to (ε, δ)-DP with δ = 10−5.</div>
Algorithm
ε = 1
ε = 2
ε = 3
ε = 4
ε = 8
ε = ∞
MNIST
DP-SGD [2]
95
97
98.3
PATE-G [75]
98(ε=2.04)
98.1( ε=8.03)
99.2
Confident-GNMax [76]
98.5(ε=1.97)
99.2
Tempered Sigmoid [77]
98.1(ε=2.93)
Bu et al. [15]
96.6(ε=2.32)
97.0(ε =5.07)
Chen and Lee [20]
90.0(ε =2.5)
Nasr et al. [71]
96.1(ε =3.2)
Yu et al. [98]
93.2(ε =6.78)
Feldman and Zrnic [39]
96.56(ε=1.2)
97.71
LP-1ST
95.34
98.16
98.81
99.08
99.33
LP-2ST
95.82
98.78
99.14
99.24
Fashion
MNIST
DP-SGD [77]
81.9(ε=2.7)
89.4
Tempered Sigmoid [77]
86.1(ε=2.7)
Chen and Lee [20]
82.3
LP-1ST
80.78
90.18
92.52
93.50
94.28
LP-2ST
83.26
91.24
93.18
94.10
Algorithm
ε=1
ε=2
ε=3
ε=4
ε=∞
LP-1ST
76.56
92.04
95.86
96.86
98.33
LP-2ST
81.26
93.72
97.19
97.83
-
# C Extra Results on Multi-Stage Training
In addition to the results presented in the main text, we include extra results of multi-stage training on MNIST [61], Fashion MNIST [97], and KMNIST [25]. Both MNIST and Fashion MNIST have been previously used to benchmark DP deep learning algorithms. We compare our algorithms with previously reported numbers in Table 4. Our algorithms outperform previous methods across all ε’s on both datasets. The gap is more pronounced on Fashion MNIST, which is slightly harder than MNIST. Furthermore, LP-2ST consistently improves over LP-1ST. Table 5 shows the model performances on KMNIST under different privacy losses. The results are qualitatively similar to the ones for MNIST and Fashion MNIST.
# D Learning Dynamics of Multi-stage Training
Fig. 3 visualizes the learning curves of LP-1ST and LP-2ST on CIFAR-10 with ε = 2. Stage-1 of LP-2ST (using 65% training data) clearly underperforms LP-1ST with the full training set. But it is good enough to provide useful prior for stage-2. The RRWithPrior algorithm responds with an average k = 1.86 over the remaining 35% of the training set. As the dotted line shows, the top-2 accuracy of the model trained in stage-1 reaches 90% at the end of training, indicating that the true label on the test set is within the top-2 prediction with high probability. In stage-2, we continue with the model trained in stage-1, and train on the combined data of the two stages. This is possible because the labels queried in stage-1 are already private. As a result, LP-2ST achieves higher performance than LP-1ST.
# E Analysis of Robustness to Hyperparameters
Following previous work, [e.g., 77], we report the benchmark performance after hyperparameter tuning. In practice, to build a rigorous DP learning system, the hyperparameter tuning should be performed using private combinatorial optimization [45]. Since that is not the main focus of this paper, we skip this step for simplicity. Meanwhile, we do the following analysis of model performance
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2f1f/2f1f14c9-25d4-4bb0-b3b2-71b3625ad445.png" style="width: 50%;"></div>
<div style="text-align: center;">25 50 75 100 125 150 175 200 training epoch Figure 3: The learning curves of LP-1ST vs LP-2ST on CIFAR-10 (ε = 2)</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2b33/2b33a643-6392-469b-8107-642a285970ed.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: The final performance of LP-2ST on CIFAR-10 (ε = 2) (a) under different stage-1 / stage-2 data split and prior temperature; (b) under different mixup coefficients for stage-1 and stage-2.</div>
under variations of different hyperparameters, which shows that the algorithms are robust in a large range of hyperparameters, and also provides some intuition for choosing the right hyperparameters. Data Splits and Prior Temperature. The data split parameter decides the ratio of data in different stages of training. Allocating more data for stage-1 allows us to learn a better prior model for the LP-2ST algorithm. However, it will also decrease the number of training samples in stage-2, which reduces the utility of the learned prior model. In practice, ratios slightly higher than 50% for stage-1 strike the right balance for LP-2ST. We use a temperature parameter t to modify the learned prior. Specifically, let fk(x) be the logits prediction of the learned prior model for class k on input x. The temperature modifies the prior ˆpk(x) as:
� As t →0, it sparsifies the prior by forcing it to be more confident on the top classes, and as t →∞, the prior converges to a uniform distribution. In our experiments, we find it useful to sparsify the prior, and temperatures greater than 1 are generally not helpful. Fig. 4(a) shows the performance for different combinations of data split ratio and temperature. Accuracy of Stage-1. Ideally, one would want the k calculated in RRWithPrior to satisfy the condition that the ground-truth label is always in the top-k prior predictions. Because otherwise, the randomized response is guaranteed to be a wrong label. One way to achieve such a goal is to make
Accuracy of Stage-1. Ideally, one would want the k calculated in RRWithPrior to satisfy the condition that the ground-truth label is always in the top-k prior predictions. Because otherwise, the randomized response is guaranteed to be a wrong label. One way to achieve such a goal is to make
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/61ca/61cac2a5-60c9-4c59-b5b7-11b51f741170.png" style="width: 50%;"></div>
<div style="text-align: center;"><= 0.8 (0.8,0.84] (0.84,0.88] (0.88,0.9] (0.9,0.92] (0.92,0.94] (0.94,0.96] > 0.96 range of top-k test accuracy</div>
Figure 5: The relation between top-k accuracy of stage-1 and the final accuracy of LP-2ST (CIFAR-10, ε = 2). The x-axis is the range of top-k accuracy of stage-1 models evaluated on the test set. For each range, the violin plot shows the distribution of the final test accuracy of LP-2ST where the RRWithPrior procedure calculated an average k (rounded to the nearest integer) for which the top-k accuracy of the stage-1 model falls in the given range.
the stage-1 model have high top-k accuracy. For example, we could allocate more data to improve the performance of stage-1 training, or tune the temperature to spread the prior to effectively increase the k calculated by RRWithPrior. In either case, a trade-off needs to be made. In Fig. 5, we visualize the relation between top-k test accuracy of stage-1 training and the final performance of LP-2ST. For each value range in the x-axis, we show the distribution of the final test accuracy where the average k (rounded to the nearest integer) calculated in RRWithPrior would make the top-k accuracy of the corresponding stage-1 training fall into this value range. The plot shows that the final performance drops when the top-k accuracy is too low or too high. In particular, achieving near perfect top-k accuracy in stage-1 is not desirable. Note this plot measures the top-k accuracy on the test set, so while it is useful to observe the existence of a trade-off, it does not provide a procedure to choose the corresponding hyperparameters. Mixup Regularization. Mixup [101] has a hyperparameter α that controls the strength of regularization (larger α corresponds to stronger regularization). We found that α values between 4 and 8 are generally good in our experiments, and as shown in Fig. 4(b), stage-2 typically requires less regularization than stage-1. Intuitively, this is because the data in stage-2 is less noisier than stage-1.
Mixup Regularization. Mixup [101] has a hyperparameter α that controls the strength of regularization (larger α corresponds to stronger regularization). We found that α values between 4 and 8 are generally good in our experiments, and as shown in Fig. 4(b), stage-2 typically requires less regularization than stage-1. Intuitively, this is because the data in stage-2 is less noisier than stage-1.
# F Convex SCO with LabelDP
In this section, we give the proofs of the Theorem 5 and Corollary 6 for private stochastic convex optimization (SCO) and additionally prove some further, related results. We first formally introduce the setting of SCO. Suppose we are given some feature space X (e.g., the space of all images), and label space [K] = {1, 2, . . . , K}. Write Z = X × [K]. Let W ⊂Rp be a convex parameter space. Let D be the (Euclidean) diameter of W, namely D := maxw,w′∈W ∥w −w′∥. Suppose we are given a loss function ℓ: W × Z →R, which specifies the loss ℓ(w, z) for a given parameter vector w ∈W on the example z = (x, y). Given a sequence of samples (x1, y1), . . . , (xn, yn) drawn i.i.d. from a distribution P over Z, the goal is to find w minimizing the popoulation risk, namely L(w, P) := E(x,y)∼P [ℓ(w, (x, y))]. Write w⋆:= arg minw∈W L(w, P). In this section, we make the following assumptions on ℓ: Assumption 7 (Convexity). For each z ∈Z, the function w �→ℓ(w, z) is convex. Assumption 8 (Lipschitzness). For each z ∈Z, the function w �→ℓ(w, z) is L-Lipschitz (with respect to the Euclidean norm). Under Assumptions 7 and 8, Bassily et al. [12, Theorem 4.4] showed that there is an (ε, δ)-DP algorithm that given n i.i.d. samples from a distribution P and has access to a gradient oracle for ℓ,
As shown by Bassily et al. [12] (building off of previous work by Bassily et al. [10]), the rate (6) is tight up to logarithmic factors: in particular, there is a lower bound of Ω �√p nε � on the excess risk for any (ε, δ)-DP algorithm, meaning that dimension dependence is necessary for private SCO. Subsequent work [40] showed how to obtain the rate (6) in linear (in n) time. We additionally remark that there has much work (e.g., [19, 59, 10, 103, 95]) on the related problem of DP empirical risk minimization, for which rates similar to (6), except without the 1/√n term, are attainable.
# F.1 Label-Private SGD
In this section we prove Theorem 5, showing that dimension-independent rates are possible in the setting of label DP privacy (in contrast to the standard setting of DP where privacy of the features must also be maintained). The algorithm that obtains the guarantee of Theorem 5 is LP-RR-SGD (Algorithm 5). Both LP-RR-SGD and the training procedure of Section 5 (which uses RRWithPrior) update the weight vectors using gradient vectors ˆgt, which are obtained by using randomized response on the labels yt for the training examples (xt, yt). LP-RR-SGD, however, ensures that ˆgt is an unbiased estimate of the true gradient, which facilitates the theoretical analysis, whereas this is not guaranteed the training procedure of Section 5.
for all ˆy ∈[K]. (c) Let gt = ∇wℓ(wt, (xt, ˜yt)) and
(d) Let wt+1 ←ΠW(wt −ηt · ˆgt). 3. Output ˆw := wn+1.
We now restate Theorem 5 formally below: Theorem 9 (Formal version of Theorem 5). For any ε ∈(0, 1), the algorithm LP-RR-SGD satisfies the requirement of ε-LabelDP; moreover, if run with step size ηt = Dε 6KL √ t, its output ˆw satisfies
Theorem 9 (Formal version of Theorem 5). For any ε ∈(0, 1), the algorithm LP-RR-SGD satisfies the requirement of ε-LabelDP; moreover, if run with step size ηt = Dε 6KL √ t, its output ˆw satisfies
We remark that even in the non-private setting, a lower bound of Ω(DL/√n) is known on the excess risk for stochastic convex optimization [73, 6], meaning that Theorem 9 is tight up to a factor of O(K log n/ε). In Section F.3, we improve the lower bound to ˜Ω(DL/√εn) for small ε ≤1 (where ˜Ωhides a logarithmic factor in 1/ε). Hence, our bound above is tight to within a factor of ˜O(K log n/√ε).
of these points, respectively, then it is immediate from definition of Qt that for any subset S ⊂Rp, Pr[ˆgt∈S] Pr[ˆg′ ∈S] ≤eε. That LP-RR-SGD is ε-LabelDP follows immediately from the post-processing property
of these points, respectively, then it is immediate from definition of Qt that for any subset S ⊂Rp, Pr[ˆgt∈S] Pr[ˆg′ t∈S] ≤eε. That LP-RR-SGD is ε-LabelDP follows immediately from the post-processing property of DP. Next we establish the uility guarantee. Note that by definition of ˆgt, we have that
i.e., ˆgt is an unbiased estimate of ∇wℓ(wt, (xt, yt)). Next, we bound the variance of the gradient error ˆgt −∇wℓ(wt, (xt, yt)), as follows:
where we have used that ℓis L-Lipschitz, ε ≤1, and that K ≥2. Using Shamir and Zhang [81, Theorem 2] with gradient moment G2 := 36K2L2 ε2 , we get that for step size choices ηt := D G √ t, the output ˆw of LP-RR-SGD satisfies
Now we prove Corollary 6; a formal version of the corollary is stated below. Corollary 10 (Formal version of Corollary 6). Suppose that we are given a prior px for every x and let Y x k denote the set of top-k labels with respect to px. Then, for any ε ∈(0, 1), there is an ε-LabelDP algorithm which outputs ˆw ∈W satisfying
Proof. Suppose we are given access to samples (x, y) drawn from a distribution P on X × [K]. For a pair (x, y) ∈X × [K], define a random pair ξ((x, y)) ∈X × [K], by setting ξ((x, y)) = (x, y) if y ∈Y x k , and otherwise letting ξ((x, y)) to be drawn uniformly over the set {(x, k′) : k′ ∈Y x k }. Let P ′ be the distribution of ξ((x, y)), where (x, y) ∼P. For any w1, w2 ∈W, it follows that
|L −L −L −L| = ���� � Z [ℓ(w1, (x, y)) −ℓ(w2, (x, y))]dP((x, y)) − � Z [ℓ(w1, (x, y)) −ℓ(w2, (x, y))]dP ′((x, y)) ���� ≤ ����� � {(x,y):y̸∈Y x k } ([ℓ(w1, (x, y)) −ℓ(w2, (x, y))] −[ℓ(w1, ξ((x, y))) −ℓ(w2, ξ((x, y)))]) dP((x, y)) ����� ≤2DL · Pr [y ̸∈Y x  ], (9)
where the last step uses that |ℓ(w1, (x, y))−ℓ(w2, (x, y))| ≤L∥w1−w2∥≤LD for all w1, w2 ∈W. Now we simply run the algorithm LP-RR-SGD, except that when we receive a point (x, y) ∼P, we pass the example ξ((x, y)) to LP-RR-SGD (instead of (x, y)), and we let the set of possible labels be Y x k (instead of [K]). Since each such example ξ((x, y)) is only passed to LP-RR-SGD once, the resulting allgorithm is still ε-LabelDP. Since the label of ξ((x, y)) belongs to Y x k , which has size k for all x, Theorem 9 gives that the output ˆw of LP-RR-SGD satisfies E[L( ˆw, P ′)] −minw L(w, P ′) ≤

(8)
where (10) follows since L(w⋆ P ′, P ′) ≤L(w⋆ P , P ′) by definition of w⋆ P ′. (8) is an immediate consequence.
# F.2 A Better Bound for Approximate DP
Next we introduce an algorithm, LP-Normal-SGD (Algorithm 6), which shows how to improve upon the excess risk bound of Theorem 9 by a factor of √ K, if we relax the privacy requirement to approximate LabelDP (i.e., (ε, δ)-LabelDP with δ > 0). LP-SGD performs a single pass of SGD over the input dataset, with the following modification: it adds a Gaussian noise vector to each gradient vector with nonzero variance only in the K-dimensional subspace Lt corresponding to the K possible labels for each point xt. This means that the norm of a typical noise vector scales only as √ K as opposed to the scaling √p, which similar algorithms for the standard setting of DP (e.g., [10]) obtain.
Algorithm 6 LP-Normal-SGD
Input: Distribution P over X × [K], convex and L-Lipschitz loss function ℓ, privacy parameters ε, δ, convex
parameter space W, variance factor σ > 0, step size sequence ηt > 0.
1. Choose an initial weight vector w1 ∈W.
2. For t = 1 to n:
(a) Receive (xt, yt) ∼P.
(b) Let ˜bt ∼N(0, σ2Ip).
(c) Let Lt ←span{∇wℓ(wt, (xt, k)) : k ∈[K]} ⊂Rp.
(d) Let bt ←ΠLt(˜bt) denote the Euclidean projection of ˜bt onto Lt.
(e) Let wt+1 ←ΠW(wt −ηt · (∇wℓ(wt, (xt, yt)) + bt)).
3. Output ˆw := wn+1.
Proposition 11. There is a constant C > 0 so that the following holds. For any ε, δ ∈(0, 1), σ = CL√ log 1/δ ε , ηt = D √ (L2+Kσ2)·t, the algorithm LP-SGD (Algorithm 6) is (ε, δ)-LabelDP and satisfies the following excess risk bound:
Proof of Proposition 11. We first argue that the privacy guarantee holds. Note that for any k, k′ ∈[n], for any x ∈X, w ∈W, we have ∥∇wℓ(w, (x, k)) −∇wℓ(w, (x, k′))∥≤2L. Therefore, for any , the mechanism
is (ε, δ)-DP as long as σ ≥ CL√ log 1/δ ε , for some constant C > 0 [32]. Since each (xt, yt) is used in only a single iteration of LP-Normal-SGD, it follows from the post-processing of DP that LP-Normal-SGDis (ε, δ)-LabelDP for this choice of σ. Next we establish the utility guarantee. Since, for each t ∈[n], Lt is a subspace of Rp of at most K dimensions, it holds that for each t, E[∥bt∥2] ≤Kσ2. Thus E � ∥∇wℓ(wt, (xit, yit)) + bt∥2� ≤ L2 + Kσ2. Using Shamir and Zhang [81, Theorem 2] with gradient moment G2 := L2 + Kσ2, we get that for step size choices ηt := D G √ t, it holds that
(10)

In this section, we prove the following lower bound on excess risk, which is tight with respect to (11) in Proposition 11 up to a factor of ˜O( � K/ε). Proposition 12. For any ε ∈(0, 1], D, L > 0 and any sufficiently large n ∈N and sufficiently small δ > 0 (both depending on ε), the following holds: for any (ε, δ)-LabelDP algorithm A, there exists a loss function ℓthat is L-Lipschitz and convex, and a distribution P for which � �
E ˜S∼P ⊗n, ˆ w∼A( ˜S)[L( ˆw, P)] −L(w⋆, P) ≥˜Ω �DL √εn � .
We remark that the lower bound of Ω(DL/√n) is well known for non-private SCO. This lower bound applies to our setting as well and thus the lower bound in Proposition 12 can be viewed as an improvement of a factor for ˜Ω(1/√ε) over the non-private lower bound. We prove Equation (11) by first proving an analogous bound in the empirical loss minimization (ERM) setting and then deriving SCO via a known reduction.
# F.4 Lower Bound on Excess Risk for ERM
Recall that in ERM setting, we are given a set S = {(x1, y1), . . . , (xn, yn)} ⊆Z of n labelled examples. The empirical risk of w is defined as L(w, S) := 1 n �n i=1 ℓ(w, (x, y)). Here we would like to devise an algorithm that minimizes the excess empirical risk, i.e., E[L( ˆw, S)] −L(w⋆, S) where ˆw is the output of the algorithm and w⋆:= arg minw∈W L(w, S). We start by proving the following lower bound on excess risk for LabelDP ERM algorithms. Note that the lower bound does not yet grow as ε decreases; that version of the lower bound will be proved later in this section. Proposition 13. For any ε, D, L, δ > 0, K ≥2 and n ∈N such that ε ≤O(1), δ ≤1 −Ω(1), the following holds: for any (ε, δ)-LabelDP algorithm A, there exists a loss function ℓthat is L-Lipschitz and convex, and a dataset ˜S of size n for which � �
Recall that in ERM setting, we are given a set S = {(x1, y1), . . . , (xn, yn)} ⊆Z of n labelled examples. The empirical risk of w is defined as L(w, S) := 1 n �n i=1 ℓ(w, (x, y)). Here we would like to devise an algorithm that minimizes the excess empirical risk, i.e., E[L( ˆw, S)] −L(w⋆, S) where ˆw is the output of the algorithm and w⋆:= arg minw∈W L(w, S).
∈W L We start by proving the following lower bound on excess risk for LabelDP ERM algorithms. Note that the lower bound does not yet grow as ε decreases; that version of the lower bound will be proved later in this section. Proposition 13. For any ε, D, L, δ > 0, K ≥2 and n ∈N such that ε ≤O(1), δ ≤1 −Ω(1), the following holds: for any (ε, δ)-LabelDP algorithm A, there exists a loss function ℓthat is L-Lipschitz and convex, and a dataset ˜S of size n for which � �
Proof. Let W := {w ∈Rd : ∥w∥≤D/2} and X := {x ∈Rd : ∥x∥≤1}. We define the loss to be 
 Note that the diameter of W is D and ℓ(·, (x, y)) is convex and L-Lipschitz. Consider any (ε, δ)LabelDP algorithm A. Let ei ∈Rn be the ith standard basis vector. Consider a dataset S = {(e1, y1), . . . , (en, yn)} where y1, . . . , yn ∈{1, 2} are random labels which are 1 w.p. 0.5 and 2 otherwise. For notational convenience, we write ˜yi to denote 2yi −3 ∈{−1, 1}. By the (ε, δ)LabelDP guarantee of A, we have
Pr S, ˆ w∼A(S)[˜yi · ⟨ˆw, ei⟩> 0] = 1 2 Pr S, ˆ w∼A(S)[⟨ˆw, ei⟩< 0 | ˜yi = −1] + 1 2 Pr S, ˆ w∼A(S)[⟨ˆw, ei⟩> 0 | ˜yi = 1] ≤1 2 · � eε · Pr S, ˆ w∼A(S)[⟨ˆw, ei⟩< 0 | ˜yi = 1] + δ � + 1 2 � eε · Pr S, ˆ w∼A(S)[⟨ˆw, ei⟩> 0 | ˜yi = −1] + δ � = eε · Pr S, ˆ w∼A(S)[˜yi · ⟨ˆw, ei⟩< 0] + δ.
(11)
(12)
Letting I ˆ w,S := {i ∈[n] : ˜yi · ⟨ˆw, ei⟩> 0} for any S, ˆw,
Consider any S as generated above; it is obvious to see that w⋆= D 2 · � 1 √n � i∈[n] ˜yiei � , which results in L(w⋆, S) = −DL 2√n. On the other hand, for any ˆw,
where the second inequality follows from Cauchy–Schwarz inequality and the last inequality follows from our assumption that δ ≤1 −Ω(1) and ε ≤O(1).
To make the lower bound above grows with 1/√ε for ε ≤1, we will apply the technique used in [89]. Recall that a pair of datasets are said to be k-neighbor if they differ in at most k labels. The following is a well-known bound, so-called group privacy; see e.g. Steinke and Ullman [89, Fact 2.3]. (Typically this fact is stated for the standard DP but it applies to LabelDP in the same manner.) Fact 14. Let A be any (ε, δ)-LabelDP algorithm. Then, for any k-neighboring database S, S′ and every subset T of the output, we have Pr[A(S) ⊆T] ≤ekε · Pr[A(S′) ⊆T] + ekε−1 eε−1 · δ. We can now prove the following lower bound that grows with 1/√ε by simplying replicating each element 1/ε times. Lemma 15. For any ε′ ∈(0, 1], D, L, δ′ > 0, K ≥2 and n ∈N such that n ≥1/γ, δ′ ≤Ω(ε′), the following holds: for any (ε′, δ′)-LabelDP algorithm A′, there exists a loss function ℓthat is L-Lipschitz and convex, and a dataset ˜S′ of size n for which �DL �
(13)
(14)
Proof. Suppose for the sake of contradiction there exists (ε, δ)-LabelDP algorithm A′ such that E ˆ w∼A′( ˜S′)[L( ˆw, ˜S′)] −L(w⋆, ˜S′) ≤o � DL √ ε′n � . Let k = ⌊1/ε⌋. We construct an algorithm A as follows: on input ˜S, it replicates each element of ˜S k times to construct a dataset ˜S′. It then returns A′( ˜S′). From the utility guarantee of A′, we have E ˆ w∼A( ˜S)[L( ˆw, ˜S)] −L(w⋆, ˜S) ≤o � DL √n � . Furthermore, Fact 14 ensures that A is (ε, δ)-DP for ε = kε′ ≤1 and δ = ekε′−1 eε′−1 δ′ ≤O(δ′/ε′). When δ′ = C/ε′ for any sufficiently small C > 0, A violates Proposition 13, concluding our proof.
# F.5 From ERM to SCO
Bassily et al. [12]6 gave a reduction from private SCO to private ERM. Although this bound is proved in the context of standard (both label and sample) DP, it is not hard to see that a similar bound holds for LabelDP with exactly the same proof. To summarize, their proof yields the following bound: Lemma 16. For any γ, ε > 0 and δ ∈(0, 1/2), suppose that there is an ( ε 4 log(2/δ), e−εδ 8 log(2/δ))LabelDP algorithm that yields expected excess population risk of for SCO is at most γ. Then, there exists an (ε, δ)-LabelDP algorithm for convex ERM (with the same parameters D, L, n) with excess empirical risk at most γ. Plugging this into Lemma 15, we arrive at Proposition 12.
# G Generalization Bounds for RR with Prior
Let X, Z be similar to the previous section and Y = [K] be the class of labels. We consider a setting where there is a concept class F of functions f : X →R. Given n samples drawn i.i.d. from some distribution P on Z, we would like to output a function f with a small population risk, which is defined as L(f; P) = E(x,y)∼P [ℓ(f(x), (x, y))]., where ℓ: R × Z →[0, 1] is a loss function. Throughout this section, we assume that ℓis L-Lipschitz (Assumption 8). Priors and Randomized Response. Let k ≤K be a positive integer. We work in the same setting as Corollary 10, i.e. we assume a prior px for every x and let Y x k denote the set of top-k labels with respect to px. We let ˜P be the distribution where we first draw (x, y) ∼P and then output (x, ˜y) where ˜y ∼RRTop-kpx(y) with DP parameter ε. Debiased Loss Function. Let pk,ε denote 1 eε+k−1. We consider a debiased version of the loss ℓ; this was done before in [72] for the case of binary classification with noisy labels. In our setting, it generalizes to the following definition: � �
For a set S of n labeled examples (x1, y1), . . . , (xn, yn) ∈Z, its empirical risk (w.r.t loss ˜ℓ) as ˜L(f; S) = 1 n �n i=1 ˜ℓ(f(xi), (xi, yi)). We consider simple ε-LabelDP algorithm that randomly draws n i.i.d. samples S from P, apply (ε-LabelDP) RRTop-k on each of the label to get a randomized dataset ˜S, and finally apply empirical risk minimization w.r.t. the debiased loss function ˜ℓon ˜S. We remark that this algorithm is exactly the same as drawing n samples i.i.d. from ˜P and apply empirical risk minimization (again w.r.t. ˜ℓ). Our main result of this section is a generalization bound roughly saying that the empirical risk (w.r.t. ˜ℓ) is small iff the popultion risk (w.r.t. ℓ) is small. This is stated more formally below, where Rn,D(F) denote the Rademacher Complexity of F (defined below in Definition G.2). Theorem 17. Let PX be the marginal of P over X. Let ˜S be a set of n i.i.d. labeled samples drawn from ˜P. Then, with probability at least 1 −β, the following holds for all f ∈F: �
(17)
(18)
Via standard techniques (see e.g. [72]), the above bound imply that the empirical risk minimizer incurs excess loss similar to the bound in Equation (18) (within a factor of 2). Recall that RRTop-k can of course be thought of RRWithPrior in the case when e.g. the prior px is uniform over the k labels in Y x k . Thus, Theorem 17 can be viewed as a generalization bound for RRWithPrior with these “uniform top-k” priors.
# G.1 Additional Preliminaries
To prove Theorem 17, we need several additional observations and definitions. In addition to the previously defined L(f; P), ˜L(f; S), we analogously use ˜L(f; P), L(f; S) to denote the population risk w.r.t. ˜ℓon distribution P and the empirical risk w.r.t. ℓon the labeled sample set S respectively.
Properties of the Debiased Loss Function. We will start by proving a few basic properties of th debiased loss functions. The first two lemmas are simple to check: Lemma 18. If y ∈Y x k , it holds that E˜y∼RRTop-kpx(y)[˜ℓ(t, (x, ˜y))] = ℓ(t, (x, y)). Lemma 19. ˜ℓis L · 1+k·pk,ε 1−k·pk,ε -Lipschitz (in t for every fixed x, y).
Finally, we observe that the population risk w.r.t. ˜ℓon distribution ˜P is close to that w.r.t. ℓon P: Lemma 20. For any function f, we have |L(f; P) −˜L(f, ˜P)| ≤ Pr (x,y)∼P[y /∈Y x k ]. (19
|L(f; P) −˜L(f; ˜P)| = |E(x,y)∼P [ℓ(f, (x, y))] −E(x,y)∼P,˜y∼RRTop-kpx(y)[ℓ(f, (x, ˜y))]| ≤E(x,y)∼P [|ℓ(f, (x, y)) −E˜y∼RRTop-kpx(y)[ℓ(f, (x, ˜y))]|].
Due to Lemma 18, the inner term is zero whenever y ∈Y x k ; furthermore, since the range of ℓis in [0, 1], the last term is at most Pr(x,y)∼P [y /∈Y x k ] as desired.
Rademacher Complexity. Given a space V and a distribution D over V, we let S be a set of examples v1, . . . , vn drawn i.i.d. from D. We also let F be a class of functions f : V →R. Definition G.1 (Empirical Rademacher Complexity). The empirical Rademacher complexity of F is defined as:
� where σ1, . . . , σ