# Analysis of Catastrophic Forgetting for Random Orthogonal Transformation Tasks in the Overparameterized Regime
Daniel Goldfarb Khoury College of Computer Sciences Northeastern University Boston, MA 02115 goldfarb.d@northeastern.edu
Dept. of Mathematics and Khoury College of Computer Sciences Northeastern University Boston, MA 02115 p.hand@northeastern.edu
# Abstract
Overparameterization is known to permit strong generalization performance in neural networks. In this work, we provide an initial theoretical analysis of its effect on catastrophic forgetting in a continual learning setup. We show experimentally that in permuted MNIST image classification tasks, the generalization performance of multilayer perceptrons trained by vanilla stochastic gradient descent can be improved by overparameterization, and the extent of the performance increase achieved by overparameterization is comparable to that of state-of-the-art continual learning algorithms. We provide a theoretical explanation of this effect by studying a qualitatively similar two-task linear regression problem, where each task is related by a random orthogonal transformation. We show that when a model is trained on the two tasks in sequence without any additional regularization, the risk gain on the first task is small if the model is sufficiently overparameterized.
# 1 Introduction
Continual learning is the ability of a model to learn continuously from a stream of data, building on what was previously learned and retaining previously learned skills without the need for retraining. A major obstacle for neural networks to learn continually is the catastrophic forgetting problem: the abrupt drop in performance on previous tasks upon learning new ones. Modern neural networks are typically trained to greedily minimize a loss objective on a training set, and without any regularization, the model’s performance on a previously trained task may degrade. Techniques for mitigating catastrophic forgetting fall under three main groups: generative replay, parameter isolation, and regularization methods [7]. Generally, the goal of regularization methods is to determine important parameters from previous tasks and constrain them so that they do not get modified too much while training subsequent tasks. Two common regularization methods are Synaptic Intelligence (SI) [22] and Elastic Weight Consolidation (EWC) [13]. See Appendix A for a detailed description of them. It is well-known that strong generalization performance for neural networks is typically obtained in the overparameterized regime, where the number of learnable parameters is greater than the number of training examples. Work on overparameterized machine learning has led to research on
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/23bf/23bf124e-7f70-47b6-a517-eb15d43288a9.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b10c/b10cb879-258d-45f9-94a7-fe08a33ad74c.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Results of permuted MNIST experiment. Red curves denote performance of SI, yellow curves denote performance of EWC, blue curves denote performance of Vanilla SGD. Bolder saturation of lines corresponds to larger width parameters (more overparameterization). Specific hyperparameters are reported in Appendix B. Curves for w = 7, 9 for EWC are omitted due to computational constraints on Fisher matrix estimates.</div>
Figure 1: Results of permuted MNIST experiment. Red curves denote performance of SI, yellow curves denote performance of EWC, blue curves denote performance of Vanilla SGD. Bolder saturation of lines corresponds to larger width parameters (more overparameterization). Specific hyperparameters are reported in Appendix B. Curves for w = 7, 9 for EWC are omitted due to computational constraints on Fisher matrix estimates.
the so-called double descent phenomenon, where test error improves as model complexity increases beyond the level needed to fit the training data, outperforming all underparameterized versions of the model [2]. One of the first observations of this behavior in modern neural networks was in extremely wide ResNet18 models that generalize better than their underparameterized counterparts on CIFAR-10 despite fitting to label noise [17]. This model-wise double descent phenomenon has been demonstrated analytically in a variety of machine learning models [10, 3, 1], including some as simple as linear regression. Such linear models will be the basis of theoretical analysis in the present paper. While investigating the relationship of overparameterization with catastrophic forgetting, we made an interesting experimental observation. We considered 10 permuted-MNIST tasks, where each task has training data given by random permutations of the original MNIST images [15]. We trained 2-layer multilayer perceptrons (MLPs) with a variety of layer widths given by [400w, 400w], where w = 1, 3, 5, 7, 9, using vanilla stochastic gradient descent and the continual learning algorithms, SI and EWC. We compared the test accuracy on all seen tasks. As expected, and as shown in Figure 1, average accuracy with SGD drops significantly after learning multiple tasks, and that drop is mitigated by using SI or EWC. Interestingly, we observe that a significant fraction of the accuracy gain achieved by SI or EWC can be obtained with vanilla SGD by simply overparameterizing the model. This can be seen by comparing the w = 1 and w = 9 curves with SGD to the curves with SI and EWC. See Appendix B for more details on the experiments. The goal of the present paper is to analytically illustrate the effect overparameterization can have on catastrophic forgetting. As with the illustrations of double descent in [10, 3, 1], we choose to study the effect with a linear regression problem for simplicity and mathematical convenience. While the experiments above study the case of tasks related by random permutations, our analysis will instead study the qualitatively similar case where two tasks are related by a random orthogonal transformation, which results in simpler mathematical analysis, as we discuss in Section 3. The relation given by the random orthogonal transformation gives two data feature spaces corresponding to each task that are approximately but not exactly orthogonal. We construct two tasks, A and B. Let task A be defined by data matrix XA ∈Rn×p, with rows being p noisy random projections of some low-dimensional latent features, and responses y ∈Rn that are noiseless and linear in the latent features. Let O be a random p × p orthogonal matrix. Then task B is defined by data XB = XAO⊤ and the same responses y. Learning tasks A and B involve estimating a β ∈Rp for the predictor fβ : x �→x⊤β to fit the data (XA, y) and (XB, y), respectively. We analyze the increase in statistical risk on task A between an estimator trained on task A by minimizing square loss, with initialization at zero, and one sequentially trained on task A and then task B with no explicit regularization. Let R(fβ) be the risk on task A of an estimator f with parameters β. Let ˆβA be the parameters of the model that is trained on task A. Let ˆβBA be the
parameters of a model that is initialized at ˆβA and then trained on task B. Our main result is that if there are more training examples than the intrinsic (latent) dimensionality of the data and if there is not too much noise in the observed features, then
with high probability. The result asserts that under our linear model, the extent of catastrophic forgetting is arbitrarily small if the overparameterization ratio, p/n, is sufficiently large. We thus see an analytical illustration that catastrophic forgetting can be ameliorated by overparametization in the case of a suitable linear model. The full theorem is stated in Section 2.4 and its proof is provided in Appendix E.
• We empirically observe that overparameterization can account for a majority of the performance drop due to catastrophic forgetting in a permuted image task using a multi-layer perceptron. • We provide a linear regression problem that exhibits a corresponding effect for overparameterization and continual learning. • We establish a non-asymptotic bound on the performance drop of this linear model in an orthogonal transformation task setting using results from random matrix theory. This result provides a formal illustration that continual learning can in some cases be ameliorated by overparameterization.
# 2 Analysis of Catastrophic Forgetting in a Linear Model
In this section, we present a latent space model for linear regression that we will analyze in order to illustrate that overparameterization can ameliorate catastrophic forgetting. Our single task model is the latent space model of [10] without label noise. Then, we present the analogy between this linear model and neural networks. Next, we empirically demonstrate that under this model, overparameterization ameliorates catastrophic forgetting. Finally, we present a theorem that establishes that observation with high probability.
Let Z = Rd, which we call the latent feature space. Consider data for regression generated by a noiseless linear response to standard Gaussian latent features. That is, for some θ ∈Rd, let an example be given by
et X = Rp, which we call the observed feature space. We consider the case where, for each example, e have access only to p observed features, given by noisy random projections of the latent features:
Let X = Rp, which we call the observed feature space. We consider the case where, for each example, we have access only to p observed features, given by noisy random projections of the latent features:
where W ∈Rp×d and u ∼N(0, Ip). We could take W to have i.i.d. N(0, γ) entries, but for mathematical convenience, we will instead study the idealization in which W has columns that form a scaled orthonormal basis of a random d-dimensional subspace of Rp. Namely, W ⊤W = pγId. For large p, this idealization is approximately satisfied under the above Gaussian model for W. We consider two tasks, A and B, both with n examples. Task A has data (XA, y) ∈Rn×p × Rn where each of the n rows of XA and entries of y are sampled independently by (2) - (4). Let O be a random p × p orthogonal matrix. Task B has data (XB, y) where XB = XAO⊤.
(1)
(2) (3)
(4)
and we will sometimes refer to the parameters ˆβ as the estimator. We estimate the parameters of this model by gradient descent with a square loss. We are interested in the case of d < n < p. As n < p, the solution to this problem depends on initialization and solves the following optimization problem:
where β0 is the initialization, and X is either XA or XB, depending on the task being solved. To study the sequential training of tasks A and B, we define the following estimators:
• ˆβA is the solution to task A when initialized at 0, • ˆβB is the solution to task B when initialized at 0, • ˆβBA is the solution to task B when initialized at ˆβA.
These parameters are found by solving the following optimization problems:
The optimization problem in (6) has the following closed form solution when X has rank n:
where PX⊤is the orthogonal projector onto the range of X⊤. As n < p, XA and XB have rank n with probability 1, and this gives the following closed forms for ˆβA, ˆβB, ˆβBA:
We evaluate these estimators on task A. The risk on task A of an estimator f with parameters ˆβ is given by
where
(6)
(7)
(9)
(10) (11)
(12) (13) (14)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/cfb7/cfb7e94b-60d8-46b7-9637-0a517877ba1e.png" style="width: 50%;"></div>
Figure 2: The solid black line depicts the span of W. The true parameters corresponding to tasks A and B are given by β ∈W and Oβ. The gray lines depict the set of solutions to XAβ = y and XBβ = y. The estimators ˆβA, ˆβB, ˆβBA are given by orthogonal projections of an initialization on the respective consistent solutions. The red and blue ellipses depict lines of constant risk for tasks A and B, respectively.
<div style="text-align: center;">Figure 2: The solid black line depicts the span of W. The true parameters corresponding to tasks A and B are given by β ∈W and Oβ. The gray lines depict the set of solutions to XAβ = y and XBβ = y. The estimators ˆβA, ˆβB, ˆβBA are given by orthogonal projections of an initialization on the respective consistent solutions. The red and blue ellipses depict lines of constant risk for tasks A and B, respectively.</div>
See Appendix D for the derivation of (15)–(18). It follows from showing that the latent space model described above is equivalent to an anisotropic regression model where XA has i.i.d. rows XAi ∼N(0, Σ) and labels y = XAβ + ϵ where ϵ ∼N(0, σ2In).
See Appendix D for the derivation of (15)–(18). It follows from showing that the latent space model described above is equivalent to an anisotropic regression model where XA has i.i.d. rows XAi ∼N(0, Σ) and labels y = XAβ + ϵ where ϵ ∼N(0, σ2In). We aim to bound R(f ˆβBA) relative to R(f ˆβA). Figure 2 illustrates the estimators ˆβA, ˆβB, ˆβBA and curves of constant risk. As depicted, if p is large enough, ˆβBA has low risk on Task A and Task B simultaneously.
We aim to bound R(f ˆβBA) relative to R(f ˆβA). Figure 2 illustrates the estimators ˆβA, ˆβB, ˆβBA and curves of constant risk. As depicted, if p is large enough, ˆβBA has low risk on Task A and Task B simultaneously.
# 2.2 Analogy of Linear Model to Neural Networks
The linear model we study is intended to be a mathematically tractable idealization of a neural network, and it is meant to analytically illustrate that overparameteriation can ameliorate catastrophic forgetting. The analogy of this linear model and neural network training on image data is as follows: Natural images in a neural network’s training distribution can be (approximately) modeled as being on a nonlinear manifold and having a low-dimensional latent representation. Instead of observing the latent representation of an image, the neural network only sees a high-dimensional representation either directly in pixel space or perhaps in a representation computed from pixel space. Either of these representations contain noise in the features used for prediction. Responses can be approximated by a neural network. In our linear model, the low-dimensional representation of an input image is in a d-dimensional latent feature linear space. The responses are linear in the latent features. We assume the response is noiseless for the sake of simplicity, though our results could be extended to the noisy case. In our linear model, predictions are made off of a p-dimensional model given by noisy random projections of the latent features. We constrain W to have orthonormal columns which is a mathematical idealization of Gaussian measurements. We study two tasks with the same responses like in the permutation task setup, but for mathematical convenience we study tasks that are related by a random orthogonal transformation instead.
# 2.3 Numerical Experiment
Before we establish our theoretical result about the system described in Section 2.1, we provide empirical evidence that the latent space linear regression model above exhibits the phenomenon
that overparameterization can ameliorate catastrophic forgetting. Specifically, we provide empirical evidence that R(f ˆβBA) −R(f ˆβA) decreases with p. Let d = 20, n = 100, γ = 1, β0 = ⃗0, and θ ∼N(0, Id). We plot R(f ˆβBA), R(f ˆβA), R(fβ0) as a function of p ∈(n, 2000) averaged over 100 samplings of W, XA, O, u.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e483/e483fd03-da24-49db-b282-8a328f73be50.png" style="width: 50%;"></div>
Figure 3: Risk as a function of model complexity p for model (2)–(14). The dashed line depics the null risk, corresponding to the zero estimator. Note the log scale on the vertical axis. Result of simulated numerical experiment for random orthogonal transformation tasks. Dotted black line denotes risk of null estimator, blue line denotes risk of estimator trained on task A, orange line denotes risk of estimator trained on task A then task B.
Figure 3 shows the results of the experiment. We first note that both ˆβA and ˆβBA outperform the null risk, given by β = 0. The null risk, R(fβ0), defines a baseline that any reasonable model must beat. We also observe that R(f ˆβA) and R(f ˆβBA) are decreasing with p in the overparameterized regime, and that the difference between these risks appears to decrease for increasing p. Note the log-scale of the vertical-axis. This provides evidence that catastrophic forgetting is alleviated in the overparameterized regime in our two-task learning setup.
# 2.4 Main Result
Our main result is an upper bound on the performance drop, defined as R(f ˆβA) −R(f ˆβBA), for the two-task latent space linear regression model described above and inspired by the double descent literature. As described in Section 2.1, we consider (2)–(14), where W satisfies the following assumption. Assumption 2.1. All non-zero singular values of W are equal. Namely, W ⊤W = pγId. We begin with a proposition that defines the unlearned baseline for the problem. Proposition 2.2. Fix θ ∈Rd. Let W ∈Rp×d satisfy Assumption 2.1. Then
# R(f0) = ∥θ∥2.
This risk calculation agrees with the numerical experiment in Section 2.3 where ∥θ∥2 ≈d = 20. The result is formally stated and proven in Lemma E.3. For our main result, we prove that if the number of examples exceeds the problem’s latent dimensionality, if the number of parameters is sufficiently large relative to the number of examples and relative to the noise level of the observable features, then with high probability, the performance drop is small. Theorem 2.3. Fix θ ∈Rd. Let tasks A, B be given by (2)–(14). Let W ∈Rp×d satisfy Assumption 2.1 and n ≥d, p ≥max(17n, 1/γ). Then there exists constant c > 0 such that with probability at least 1 −10e−cd, the following holds:
Theorem 2.3 provides an upper bound on the amount of risk gained on task A after subsequential training on task B given by two terms. The first term provides dependence on the overparameterization ratio p/n and decreases as overparameterization becomes more extreme. The second term is given by the signal to noise ratio of the noisy features. This term dominates only when γ ≪1/√np. Based on the theorem, we observe that the overparameterization needs only to be linear in n to achieve a negligible performance drop in unregularized sequential task training compared to the baseline of ∥θ∥2 in Proposition 2.2. This shows that catastrophic forgetting is ameliorated in the overparameterized regime. This result is formally stated in Lemma E.11. A formal proof and supporting lemmas are supplied in Appendix E. We provide a proof sketch here to outline the techniques used.
# 2.5 Proof Sketch of Theorem 2.3
For readability, we write XA as A and XB as B. As shown in Appendix D, the latent space model described above is equivalent to an anisotropic regression model where A has i.i.d. rows Ai ∼N(0, Σ) and labels y = Aβ + ϵ where ϵ ∼N(0, σ2In).
After substituting the closed form solutions for ˆβA, ˆβBA, distributing terms, and applying simple Cauchy-Schwarz and triangle inequalities, we get the following bound:
R(f ˆβBA) −R(f ˆβA) ≤8pγ√pγ pγ + 1 ∥θ∥∥PW ˆβB∥+ 14√pγ∥θ∥∥PB⊤ˆβA∥+ 12 pγ (pγ + 1)2 ∥θ∥2 (21 = I + II + III (22
Lemmas E.5, E.6 establish results for orthogonal transformations to help bound ∥PW ˆβB∥, ∥PB⊤ˆβA∥ respectively. PW is a projection onto a d-dimensional space which scales the norm in I by d/p. PB⊤is a projection onto an n-dimensional space which scales the norm in II by n/p. As d ≤n by assumption, II dominates I in the final bound. Using these results and simplifying gives the following bound with probability at least 1 −10e−cd for constant c > 0:
We directly obtain
# 3 Discussion
Overparameterization is a necessity for continual learning so that there can exist an infinity of potentia optima for each task [13]. This makes it likely that there exists an optimum for some task B that is close to the solutions of some task A. We provide experimental evidence that overparameterization can provide additional benefits in combatting catastrophic forgetting for neural networks solving
(22)
(23)
(24)
permutation tasks. We use a linear model with clear analogies to neural networks in order to study this behavior theoretically. In our analysis of the linear model in the overparameterized regime, non-asymptotic matrix estimates and results for orthogonal transformations provide bounds on the performance drop. Our main result shows that, under our model, catastrophic forgetting is ameliorated for sufficiently large overparameterization. For the linear setting we study, the behavior we observe can be explained geometrically: overparameterization causes the random orthogonal transformation tasks to live in approximately orthogonal subspaces, so training on subsequent tasks does not interrupt performance on learned tasks. We view the present work as helping to establish initial results for continual learning theory. Before the field can rigorously understand machine learning algorithms in practice, the behavior of simple systems should be well understood. In particular, the behavior of linear systems with only vanilla SGD is the most natural initial result. Our work remarks that future theory should establish that continual learning algorithms beat not only a moderately parameterized baseline, but also the performance of extremely overparameterized models. First we address the concern for using permutation tasks as realistic benchmarks for continual learning methods. Researchers believe that permutation tasks only provide a best-case for real world scenarios [9]. Also, on a number of image classification datasets, MLPs do not experience forgetting when only two permutation tasks are being learned [18]. Our experiments confirm this effect while also showing that overparameterization mitigates the observable forgetting on 10 task permuted MNIST. Despite these critiques, we use permutation tasks as a launching point for theory because each task is of the same ‘difficulty’ and is amenable to mathematical analysis. Next we discuss our choice to study the problem with a linear model. Linear regression is the simplest setting, for which we know, that exhibits double descent. The consensus of several works that study double descent in linear models is that the risk of a model is monotonically decreasing in the overparameterized regime with respect to number of parameters only if the data has low effective dimension and high ambient dimension compared to the number of training samples [6, 1, 10]. In order to have a model that has monotonically decreasing performance drop for a particular continual learning problem, it is a necessity that it exhibits monotonically decreasing risk on a single task. Additionally in recent work, connections have been made between neural networks and linear models using the so-called neural tangent kernel (NTK) phenomenon [12]. The parameterization of a neural network can be so large that training only changes its parameters slightly from its initialization, resulting in functions that can be accurately approximated linearly. Hence it is reasonable that the analysis of linear models can explain the behavior of neural networks. We now remark at a technical level two choices in our analysis. The first is why we studied the case of random orthogonal transformation tasks instead of permutation tasks. The empirical performance between orthogonal and permutation tasks is similar; they both create tasks that are equally ’difficult’ for an MLP to learn, which spares us from needing to quantify problem difficulty. Appendix C provides evidence that permutation and orthogonal transformation tasks have the same difficulty in the linear setting. Also, the mathematical analysis is easier when studying orthogonal transformation tasks. With random orthogonal transformations, any subspace gets mapped to a random subspace, for which the values of coefficients are typically well spread out. With random permutations, some subspaces (e.g. those aligned with the standard basis elements) do not exhibit the same spreading effect, making the technical analysis more involved. Secondly, we do not present a bound on R(f ˆβA), though it is expected to approach zero for large p, as suggested by Figure 3. Whether or not this risk goes to 0 in p, the performance drop goes to 0 in p while the null estimator remains with constant risk. So the regression problem is being solved arbitrarily well for sufficiently large p. With the growing popularity of continual learning, much of recent work is focused on developing new algorithms to mitigate catastrophic forgetting [13, 22, 20, 16]. Only a few papers study the problem theoretically [14, 4, 8, 11, 5]. [14] uses set theory to prove that, in general, continual learning problems are NP-hard, explaining why generative replay methods perform so well. [4] uses the NTK regime to prove generalisation guarantees for an existing continual learning method. [8] uses an NTK overlap matrix to define a notion of task similarity and show that catastrophic forgetting is more severe when tasks have high similarity. [11] studies a family of continual learning methods that uses approximations of the Hessian to determine parameter importance, presenting scenarios where continual learning provably fails and succeeds. [5] shows that a number of regularization techniques that seem to be derived from differing philosophies actually all study a variation of
the Fisher information matrix. While prior work has presented generalization bounds for existing continual learning techniques, our work illustrates the relationship between overparameterization and generalization.
the Fisher information matrix. While prior work has presented generalization bounds for existing continual learning techniques, our work illustrates the relationship between overparameterization and
A natural next step is to study the regimes in which catastrophic forgetting is most problematic. This includes the setting where tasks do not have a nearly orthogonal relationship but also when data does not necessarily live on a low-dimensional manifold. We are also interested in understanding how the ideas of this paper generalize to other continual learning benchmarks and for more general neural network architectures. Prior work found experimental evidence that catastrophic forgetting is most severe not when tasks are very dissimilar but when they only have an intermediate level of similarity [19]. Using orthogonality as a proxy for task similarity, this agrees with our work that shows that nearly orthogonal tasks are less prone to catastrophic forgetting. An interesting future work would be to formalize this notion of task similarity for our model. Moving forward, one goal of theory in continual learning is to be able to analytically compare algorithms. Our work provides a foundation of understanding this behavior in a simple linear regression setting. In order to push this work forward, either non-linear models need to be studied or tasks that are related by something more complex than permutations.
# References
Table 1: Hyperparameters for the MNIST experiments
<div style="text-align: center;">Table 1: Hyperparameters for the MNIST experiments</div>
Hyperparameter
SI
EWC
SGD
learning rate
0.1
0.01
0.1
dropout
0
0
0.5
batch size
64
128
64
epochs / dataset
5
20
5
c
0.1
ξ
0.1
λ
150
fisher sample size
1,000
# A Description of Continual Learning Techniques
Synaptic Intelligence (SI) is a regularization technique that assigns to each parameter of the network an estimate of importance for learned tasks [22]. This weight is determined in an online manner by tracking the amount that each parameter contributed to the decrease in loss during training. The weight is then used to penalize changes to the network parameters during subsequent training in the form of a regularization term added to the loss function.
Synaptic Intelligence (SI) is a regularization technique that assigns to each parameter of the network an estimate of importance for learned tasks [22]. This weight is determined in an online manner by tracking the amount that each parameter contributed to the decrease in loss during training. The weight is then used to penalize changes to the network parameters during subsequent training in the form of a regularization term added to the loss function. Elastic Weight Consolidation (EWC) is a regularization technique that determines the importance of network weights using an estimation of the Fisher Information Matrix [13]. Near a minimum of the loss function, the diagonals of the Fisher matrix act as an estimate of the second order derivative of the loss with respect to each parameter. The magnitude of this derivative is used as a proxy for how sensitive the loss function is to fluctuation of the parameter. Constraining parameters according to their corresponding Fisher diagonal entries shows as an effective way of retaining the values of important weights from previous tasks while training on new ones.
# B MNIST Experiments
Table 1 reports the hyperparameters used in the MNIST experiments. We adopted the same hyperparameters for SI as in the original paper [22]. To our surprise, EWC with default hyperparameters [13] did not compete with SI. A basic grid search gave us a model that was more competitive. Blank entries mean that the hyperparameter is not relevant for the particular method. Curves for w = 7, 9 are omitted due to computational constraints in computing Fisher matrix estimates.
# C Permuted Numerical Experiment
Figure 4 shows the result of the numerical experiment in Figure 3 but with a random permutation matrix instead of a random orthogonal matrix. We observe that the same behavior holds in this scenario.
# D Equivalence of Models and Derivation of Risk
Recall in Section 2.1 where we defined the LSM model for linear regression. In this section we show that LSM is equivalent to an anisotropic regression model (ARM). We then use ARM to define the risk expression that we analyze theoretically. We begin by defining ARM. Define data matrix X ∈Rn×p and responses y = Xβ + ϵ where ϵ ∼ N(0, σ2In), σ2 = θ⊤(W ⊤W + Id)−1θ, and β = (I + WW ⊤)−1Wθ for some θ ∈Rd, W ∈Rp×d. Let rows Xi be independent random vectors in Rp with covariance Σ = WW ⊤+ Ip. Then the model is defined by the distribution over (X, y). Next we show that ARM is equivalent to LSM. First observe that for both models, (yi, x⊤ i ) ∈Rp+1 are centered Gaussian vectors. Thus to show that they induce the same distribution, it suffices to show that they have the same covariance.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/70b7/70b7f466-5fc2-46ed-8a9c-9f660adb5d24.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Result of simulated numerical experiment for random permutation tasks. Dotted black line denotes risk of null estimator, blue line denotes risk of estimator trained on task A, orange line denotes risk of estimator trained on task A then task B.</div>
We then compute the covariance matrices for each model. Under LSM, we have:
E[y2 i ] = E(θ⊤zi)(z⊤ i θ) = θ⊤Iθ E[yixi] = E(Wzi + ui)(z⊤ i θ) = Wθ E[xix⊤ i ] = E(Wxi + ui)(z⊤ i W ⊤+ ui) = 
Plugging these quantities into (25) gives:
Cov((yi, x⊤ i )⊤) = � ∥θ∥2 (Wθ)⊤ Wθ I + WW ⊤ �
Under ARM, we have:
E[y2 i ] = E(β⊤xi + ϵi)(x⊤ i β + ϵi) = β⊤(I + WW ⊤)β + σ2 E[yixi] = E(xi(x⊤ i β + ϵi)) = E(xix⊤ i β + xiϵi) = (I + WW ⊤)β E[xix⊤ i ] = I + WW ⊤
Plugging these quantities into (25) gives:
(25)
(26) (27) (28)
(29)
(30) (31) (32)
(33)
By Lemma E.12, (I + WW ⊤)−1W = W(I + W ⊤W)−1. This gives
β⊤(I + WW ⊤)β + σ2 = θ⊤(W ⊤W(I + W ⊤W)−1 + (I + W ⊤W)−1)θ = θ⊤(W ⊤W + I)(I + W ⊤W)−1θ = θ⊤θ = ∥θ∥2
Next we show equivalence of the second row, first column entries of the covariance matrices:
Next we show equivalence of the second row, first column entries of the covariance matrices:
(I + WW ⊤)β = (I + WW ⊤)(I + WW ⊤)−1Wθ = Wθ
The equivalence of the first row, second column entries also follows from this equality. The equivalence of the second row, second column entries is trivial. Finally we derive the expression for the risk of ARM. By definition, the risk of an estimator f with parameters ˆβ has the following form:
where the third equality holds from independence of ϵ and the sixth equality holds by definition of covariance. We choose to study ARM with a slightly different but equivalent expression for β. Using Lemma E.12, β = (I + WW ⊤)−1Wθ = W(W ⊤W + I)−1θ.
We choose to study ARM with a slightly different but equivalent expression for β. Using Lemma E.12, β = (I + WW ⊤)−1Wθ = W(W ⊤W + I)−1θ.
# E Supporting Lemmas
We begin with an assumption, inspired by [10], that all non-zero singular values of W are equal. Assumption E.1. All non-zero singular values of W are equal. Namely, W ⊤W = pγId. Lemma E.2. Assume W ∈Rp×d satisfies Assumption E.1. Then
where PW is the orthogonal projection onto the range of W. Proof. We have that
WW ⊤= WW ⊤pγ pγ = pγW �1 pγ Ip � W ⊤
(36) (37) (38)
(39)
(46)
(47)
W ⊤W has full rank with probability 1, so W(W ⊤W)−1W ⊤is given explicitly by PW , which completes the proof. Lemma E.3. Let Σ = WW ⊤+ Ip where W ∈Rp×d satisfies Assumption E.1. For some θ ∈Rd, let β = W(W ⊤W + Id)−1θ. Then
Proof. We have that
By Lemma E.2, Σ = pγPW + Ip where PW is the orthogonal projection onto the range of W, which gives
Since β ∈range(W),
Lemma E.4. Let x ∼N(0, Id), and ϵ ≤1, then
� where c > 0 is an absolute constant.
Proof. This statement follows from Corollary 5.17 in [21], concerning concentration of subexponential random variables. Lemma E.5. Assume W ∈Rp×d satisfies Assumption E.1. Let O be a random p × p orthogonal matrix. Fix v ∈Rp×p. Then, with probability at least 1 −2e−c1d,
for some universal constant c1 > 0.
Proof. Let x = Ov, and note that ∥x∥= ∥v∥, ∥x∥> 0 with probability 1 and x ∥x∥∼Uniform(Sp−1). Letting z ∼N(0, Ip), we have that
(49)
(50) (51)
(52)
(57)
(58)
where the symbol d= means equality in distribution. Applying Lemma E.4 twice, we get that for any ϵ < 1, with probability at least 1 −e−cϵ2p −e−cϵ2d,
for some universal constant c > 0. By choosing suitable ϵ, we obtain that for c1 = cϵ2, with probability at least 1 −2e−c1d, ∥PW Ov∥2 ≤2d p ∥v∥2. Lemma E.6. Define A ∈Rn×p with rows Ai as independent random vectors in Rp with covariance Σ = WW ⊤+ Ip where W ∈Rp×d satisfies Assumption E.1. Let O be a random p × p orthogonal matrix. Fix v ∈range(A⊤). Then with probability at least 1 −2e−c1n
for some universal constant c1 > 0.
Proof. We have that
∥POA⊤PA⊤v∥= ∥OPA⊤O⊤PA⊤v∥= ∥PA⊤O⊤PA⊤v∥
where z ∼N(0, Ip) and the last equality follows from the rotational invariance of O. Applying Lemma E.4 twice, we get that for any ϵ < 1, with probability at least 1 −e−cϵ2p −e−cϵ2n,
where z ∼N(0, Ip) and the last equality follows from the rotational invariance of O. Applying Lemma E.4 twice, we get that for any ϵ < 1, with probability at least 1 −e−cϵ2p −e−cϵ2n,
for some universal constant c > 0. By choosing suitable ϵ, we obtain that for c1 = cϵ2, with probability at least 1 −2e−c1n, ∥POA⊤v∥2 ≤2n p ∥v∥2. Lemma E.7. Let a ∈Rp be generated by N(0, Σ), Σ = WW ⊤+ Ip where W ∈Rp×d satisfies Assumption E.1. Then
Proof. It holds that E∥a∥2 2 = ∥Σ1/2∥2 F [21].
∥Σ∥∗= d(pγ + 1) + p −d = dpγ + p
(59)
(61)
(65)
(66)
(67)
Lemma E.8. Define A ∈Rn×p with rows Ai independent random vectors in Rp with covariance Σ = pγPW +Ip where W ∈Rp×d satisfies Assumption E.1. Then with probability at least 1−2e−n,
� Proof. WLOG let range(W) = span(e1, ..., ed). Then we can decompose A into two pieces: A(1) ∈Rn×d with i.i.d. N(0, pγ + 1) entries and A(2) ∈Rn×p−d with i.i.d. N(0, 1) entries. This gives
By Theorem 5.39 in [21], σmin(A⊤ (2))2 ≥(√p −d −2√n)2 with probability at least 1 −2e−n.
By Theorem 5.39 in [21], σmin(A⊤ (2))2 ≥(√p −d −2√n)2 with probability at least 1 −2e−n. Lemma E.9. Define A ∈Rn×p with rows Ai independent random vectors in Rp with covariance Σ = WW ⊤+ I where W ∈Rp×d satisfies Assumption E.1. Let ϵ ∼N(0, σ2I) and σ2 =
By Theorem 5.39 in [21], σmin(A⊤ (2))2 ≥(√p −d −2√n)2 with probability at least 1 −2e−n. Lemma E.9. Define A ∈Rn×p with rows Ai independent random vectors in Rp with covariance Σ = WW ⊤+ Ip where W ∈Rp×d satisfies Assumption E.1. Let ϵ ∼N(0, σ2In) and σ2 = θ⊤(W ⊤W + Id)−1θ for some θ ∈Rd. Then with probability at least 1 −2e−n,
 ≥ − − − Lemma E.9. Define A ∈Rn×p with rows Ai independent random vectors in Rp with covariance Σ = WW ⊤+ Ip where W ∈Rp×d satisfies Assumption E.1. Let ϵ ∼N(0, σ2In) and σ2 = θ⊤(W ⊤W + Id)−1θ for some θ ∈Rd. Then with probability at least 1 −2e−n,
Proof. We have that
We have that ∥ϵ∥2 = nσ2 = nθ⊤(W ⊤W +Id)−1θ. Under Assumption E.1, this gives ∥ϵ∥2 = n∥θ∥2 pγ+1 ,
It holds that ∥A†∥= 1/σmin(A⊤) where σmin(A⊤) is the smallest singular value of A⊤. By Lemma E.8, σmin(A) ≥√p −d −2√n with probability at least 1 −2e−n. This gives
Lemma E.10. Suppose W ∈Rp×d satisfies Assumption E.1. Let β = W(W ⊤W + Id)−1θ for some θ ∈Rd. If p ≥1/γ and p ≥16n + d, then
Proof. We have that ∥β∥2 = θ⊤(W ⊤W + Id)−1W ⊤W(W ⊤W + Id)θ. Using Assumption E.1, this gives ∥β∥= √pγ pγ+1∥θ∥. Suppose p ≥1/γ, then we have that ∥β∥≥ 1 2√pγ ∥θ∥. When p ≥16n + d, √n √p−d−2√n ≤1 2, which gives
(69)
(70) (71)
(73)
(74)
(75)
Theorem E.11. Define data matrix A ∈Rn×p and responses y = Aβ + ϵ where ϵ ∼N(0, σ2In), σ2 = θ⊤(W ⊤W + Id)−1θ, and β = W(W ⊤W + Id)−1θ for some θ ∈Rd. Let rows Ai be independent random vectors in Rp with covariance Σ = WW ⊤+ Ip where W ∈Rp×d follows Assumption E.1 and n ≥d, p ≥max(17n, 1/γ). Let O be a random p × p orthogonal matrix and B = AO⊤. Let ˆβA be the parameters of the minimum norm estimator on A, and ˆβBA be the parameters of the estimator on B using ˆβA as initialization. Let R(f ˆβ) be the risk on task A of an estimator with parameters ˆβ. Then there exists constant c > 0 such that with probability at least 1 −10e−cd, the following holds:
Proof. We have that
Distributing terms with ˆβB and PB⊤ˆβA gives
By Lemma E.2, Σ = pγPW + Ip where PW is the orthogonal projection onto the range of W. This implies that ∥Σ∥= pγ + 1, giving
R(f ˆβBA) −R(f ˆβA) = 2ˆβ⊤ A(pγPW + Ip)ˆβB −2ˆβ⊤ B(pγPW + Ip)β + ˆβ⊤ B(pγPW + Ip)ˆβB + (PB⊤ˆβA)⊤(pγPW + Ip)(PB⊤ˆβA −2ˆβA −2ˆβB + 2β) (8 ≤2pγ ˆβ⊤ APW ˆβB + 2ˆβ⊤ A ˆβB −2pγ ˆβ⊤ BPW β −2ˆβ⊤ Bβ + pγ ˆβ⊤ BPW ˆβB + ˆβ⊤ B β + (pγ + 1)∥PB⊤ˆβA∥∥PB⊤ˆβA −2ˆβA −2ˆβB + 2β∥ (8
Applying Cauchy-Schwarz and triangle inequality gives the following bound:
R(f ˆβBA) −R(f ˆβA) ≤2pγ∥ˆβA∥∥PW ˆβB∥+ 2∥ˆβA∥∥ˆβB∥+ 2pγ∥β∥∥PW ˆβB∥+ 2∥ˆβB∥∥β∥ + pγ∥ˆβB∥∥PW ˆβB∥+ ∥ˆβB∥2 + (pγ + 1)∥PB⊤ˆβA∥(∥PB⊤ˆβA∥+ 2∥ˆβA∥+ 2∥ˆβB∥+ 2∥β∥) (8

(78)
(80)
(81)
(82)
(83)
(84)
By definition in Section 2, ˆβA = PA⊤β + A†ϵ and ˆβB = OPA⊤β + OA†ϵ and it holds that ∥PA⊤β∥≤∥β∥. By Lemma E.9, ∥A⊤(AA⊤)−1ϵ∥≤ √n∥θ∥ √pγ(√p−d−2√n) with probability at least 1 −2e−n (call this Event E). So by Lemma E.10 if p ≥max(17n, 1/γ), then ∥ˆβA∥≤2∥β∥and ∥ˆβB∥≤2∥β∥, which gives
We have that ∥β∥2 = θ⊤(W ⊤W + Id)−1W ⊤W(W ⊤W + Id)θ. Using Assumption E.1, this gives ∥β∥= √pγ pγ+1∥θ∥,
R(f ˆβBA) −R(f ˆβA) ≤8pγ√pγ pγ + 1 ∥θ∥∥PW ˆβB∥+ 14√pγ∥θ∥∥PB⊤ˆβA∥+ 12 pγ (pγ + 1)2 ∥θ∥2 (87) = I + II + III (88)
  = I + II + III
# We will bound each of these terms separately, starting with I:
Substituting ˆβB = B⊤(BB⊤)−1y into this expression and distributing accordingly, we get that PW ˆβB = PW OPA⊤β + PW OA⊤(AA⊤)−1ϵ,
By Lemma E.5, there exists constant c1 > 0 such that ∥PW OPA⊤β∥≤1.5 � d p∥β∥and ∥PW OA⊤(AA⊤)−1ϵ∥≤1.5 � d p∥A⊤(AA⊤)−1ϵ∥with probability at least 1 −2e−c1d each. By Lemma E.9, ∥A⊤(AA⊤)−1ϵ∥≤ √n∥θ∥ √pγ(√p−d−2√n) (failure probability already accounted for on Event E). This gives the following bound with probability at least 1 −4e−c1d:
Substituting ∥β∥= √pγ pγ+1∥θ∥gives
Using pγ + 1 > pγ gives
Substituting ˆβA = A⊤(AA⊤)−1y into this expression and distributing accordingly, we get that PB⊤ˆβA = PB⊤PA⊤β + PB⊤A⊤(AA⊤)−1ϵ. This gives the following bound:
(91)
(93)
By Lemma E.6, there exists constant c2 > 0 such that ∥PB⊤PA⊤β∥≤1.5 � n p ∥β∥and ∥PB⊤A⊤(AA⊤)−1ϵ∥≤1.5 � n p ∥A⊤(AA⊤)−1ϵ∥with probability at least 1 −2e−c2n each. By Lemma E.9, ∥A⊤(AA⊤)−1ϵ∥≤ √n∥θ∥ √pγ(√p−d−2√n) (failure probability already accounted for on Event E). This gives the following bound with probability 1 −4e−c2n:
Substituting ∥β∥= √pγ pγ+1∥θ∥gives
Using pγ + 1 > pγ gives
Lastly we bound term III. Using pγ + 1 > pγ gives the following bound:
Putting all three terms together gives the following bound with probability 1 −10e−cd where c = min(c1, c2):
c = min(c1, c2):
Using d ≤n gives the following bound:
(94) (95)
(96)
(97)
(99)
(100)
(103)
(104)

Proof. Let W = USV be the SVD of W where U ∈Rp×d, S ∈Rd×d, V ∈Rd×d. Then we have
Let ˜U ∈Rp×p have the first d columns be U and the last p−d columns be the rest of the orthonormal basis. Then we have
(105)
(106) (107)
