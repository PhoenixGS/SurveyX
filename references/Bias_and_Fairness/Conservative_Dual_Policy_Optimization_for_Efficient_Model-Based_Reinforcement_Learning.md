# Conservative Dual Policy Optimization for Efficient Model-Based Reinforcement Learning
Shenao Zhang Georgia Institute of Technology Atlanta, GA 30332 shenao@gatech.edu
# Abstract
Provably efficient Model-Based Reinforcement Learning (MBRL) based on optimism or posterior sampling (PSRL) is ensured to attain the global optimality asymptotically by introducing the complexity measure of the model. However, the complexity might grow exponentially for the simplest nonlinear models, where global convergence is impossible within finite iterations. When the model suffers a large generalization error, which is quantitatively measured by the model complexity, the uncertainty can be large. The sampled model that current policy is greedily optimized upon will thus be unsettled, resulting in aggressive policy updates and over-exploration. In this work, we propose Conservative Dual Policy Optimization (CDPO) that involves a Referential Update and a Conservative Update. The policy is first optimized under a reference model, which imitates the mechanism of PSRL while offering more stability. A conservative range of randomness is guaranteed by maximizing the expectation of model value. Without harmful sampling procedures, CDPO can still achieve the same regret as PSRL. More importantly, CDPO enjoys monotonic policy improvement and global optimality simultaneously. Empirical results also validate the exploration efficiency of CDPO.
arXiv:2209.07676v1
# 1 Introduction
Model-Based Reinforcement Learning (MBRL) involves acquiring a model by interacting with the environment and learning to make the optimal decision using the model [55, 32]. MBRL is appealing due to its significantly reduced sample complexity compared to its model-free counterparts. However, greedy model exploitation that assumes the model is sufficiently accurate lacks guarantees for global optimality. The policies can be suboptimal and get stuck at local maxima even in simple tasks [10]. As such, several provably-efficient MBRL algorithms have been proposed. Based on the principle of optimism in the face of uncertainty (OFU) [56, 49, 10], OFU-RL achieves the global optimality by ensuring that the optimistically biased value is close to the real value in the long run. Based on Thompson Sampling [62], Posterior Sampling RL (PSRL) [57, 42, 43] explores by greedily optimizing the policy in an MDP which is sampled from the posterior distribution over MDPs. Beyond finite MDPs, to obtain a general bound that permits sample efficiency in various cases, we need to introduce additional complexity measure. For example, [49, 43] provide an �O(√dET) regret for both OFU and PSRL with eluder dimension dE capturing how effectively the model generalizes. However, it is recently shown [13, 33] that the eluder dimension for even the simplest nonlinear models cannot be polynomially bounded. The effectiveness of the algorithms will thus be crippled. The underlying reasons for such ineffectiveness are the aggressive policy updates and the overexploration issue. Specifically, when a nonlinear model is used to fit complex transition functions, its generalizability will be poor compared to simple linear problems. If a random model is selected from the large hypothesis, e.g., optimistically chosen or sampled from the posterior, it is “unsettled".
36th Conference on Neural Information Processing Systems (NeurIPS 2022).
In other words, the selected model can change dramatically between successive iterations. Policy updates under this model will also be aggressive and thus cause value degradation. What’s worse, large epistemic uncertainty results in an unrealistic model, which drives agents for uninformative exploration. An exploration step can only eliminate an exponentially small portion of the hypothesis. In this work, we present Conservative Dual Policy Optimization (CDPO), a simple yet provable MBRL algorithm. As the sampling process in PSRL harms policy updates due to the unsettled model during training, we propose the Referential Update that greedily optimizes an intermediate policy under a reference model. It mimics the sampling-then-optimization procedure in PSRL but offers more stability since we are free to set a steady reference model. We show that even without a sampling procedure, CDPO can match the expected regret of PSRL up to constant factors for any proper reference model, e.g., the least squares estimate where the confidence set is centered at. The Conservative Update step then follows to encourage exploration within a reasonable range. Specifically, the objective of a reactive policy is to maximize the expectation of model value, instead of a single model’s value. These two steps are performed in an iterative manner in CDPO. Theoretically, we show the statistical equivalence between CDPO and PSRL with the same order of expected regret. Additionally, we give the iterative policy improvement bound of CDPO, which guarantees monotonic improvement under mild conditions. We also establish the sublinear regret of CDPO, which permits its global optimality equipped with any model function class that has a bounded complexity measure. To our knowledge, the proposed framework is the first that simultaneously enjoys global optimality and iterative policy improvement. Experimental results verify the existence of the over-exploration issue and demonstrate the practical benefit of CDPO.
# 2 Background
# 2.1 Model-Based Reinforcement Learning
We consider the problem of learning to optimize an infinite-horizon γ-discounted Markov Decision Process (MDP) over repeated episodes of interaction. Denote the state space and action space as S and A, respectively. When taking action a ∈A at state s ∈S, the agent receives reward r(s, a) and the environment transits into a new state according to probability s′ ∼f ∗(·|s, a). Here, f ∗is a dirac measure for deterministic dynamics and is a probability distribution for probabilistic dynamics. In model-based RL, the true dynamical model f ∗is unknown and needs to be learned using the collected data through episodic (or iterative) interaction. The history data up to iteration t then forms Ht = {{sh,i, ah,i, sh+1,i}H−1 h=0 }t−1 i=1, where H is the actual timesteps agents run in an episode. The posterior distribution of the dynamics model is estimated as φ(·|Ht). Alternatively, the frequentist model of the mean and uncertainty can also be estimated. Specifically, consider the model function class F = {f : S ×A →S} with size |F|, which contains the real model f ∗∈F. The confidence set (or model hypothesis set) Ft ⊂F is introduced to represent the range of dynamics that is statistically plausible [49, 43, 10]. To ensure that f ∗∈Ft with high probability, one way is to construct the confidence set as Ft := {f ∈F | ∥f −�f LS t ∥2,Et ≤√βt}. Here, βt is an appropriately chosen confidence parameter (via concentration inequality), the cumulative empirical 2-norm is defined by ∥g∥2 2,Et := �t−1 i=1∥g(xi)∥2 2. The least squares estimate is
� Denote the state and state-action value function associated with π on model f by V f π : S →R and Qf π : S × A →R, respectively, which are defined as
� ��� � ��� The objective of RL is to learn a policy π∗= argmaxπ J(π) that maximizes the expected return J(π). Denote the initial state distribution as ζ. Under policy π, the state visitation measure νπ(s) over S and the state-action visitation measure ρπ(s, a) over S × A in the true MDP are defined as νπ(s) = (1 −γ) · ∞ � h=0 γh · P(sh = s), ρπ(s, a) = (1 −γ) · ∞ � h=0 γh · P(sh = s, ah = a), (2.2)
� ��� � ��� The objective of RL is to learn a policy π∗= argmaxπ J(π) that maximizes the expected return J(π). Denote the initial state distribution as ζ. Under policy π, the state visitation measure νπ(s) over S and the state-action visitation measure ρπ(s, a) over S × A in the true MDP are defined as ∞ � ∞ �
(2.1)
where s0 ∼ζ, ah ∼π(·|sh) and sh+1 ∼f ∗(·|sh, ah). The objective J(π) is then J(π) = Es0∼ζ[V f ∗ π (s0)] = E(s,a)∼ρπ[r(s, a)]
# 2.2 Cumulative Regret and Asymptotic Optimality
A common criterion to evaluate RL algorithms is the cumulative regret, defined as the cumulative performance discrepancy between policy πt at each iteration t and the optimal policy π∗over the run of the algorithm. The (cumulative) regret up to iteration T is defined as:
� In the Bayesian view, the model f ∗, the learning policy π, and the regret are random variables that must be learned from the gathered data. The Bayesian expected regret is defined as: BayesRegret(T, π, φ) := E [Regret(T, π, f ∗) | f ∗∼φ] . (2.5)
One way to prove the asymptotic optimality is to show that the (expected) regret is sublinear in T, so that πt converges to π∗within sufficient iterations. To obtain the regret bound, the width of confidence set ωt(s, a) is introduced to represent the maximum deviation between any two members in Ft:
# 3 Provable Model-Based Reinforcement Learning
In this section, we analyze the central ideas and limitations of greedy algorithms as well as two popular theoretically justified frameworks: optimistic algorithms and posterior sampling algorithms.
Greedy Model Exploitation. Before introducing provable algorithms, we first analyze greedy modelbased algorithms. In this framework, the agent takes actions assuming that the fitted model sufficiently accurately resembles the real MDP. Algorithms that lie in this category can be roughly divided into two groups: model-based planning and model-augmented policy optimization. For instance, Dyna agents [61, 20, 17] optimize policies using model-free learners with model-generated data. The model can also be exploited in first-order gradient estimators [18, 12, 9] or value expansion [15, 6]. On the other hand, model-based planning, or model-predictive control (MPC) [40, 41], directly generates optimal action sequences under the model in a receding horizon fashion. However, greedily exploiting the model without deep exploration [45] will lead to suboptimal performance. The resulting policy can suffer from premature convergence, leaving the potentially high-reward region unexplored. Since the transition data is generated by the agent taking actions in the real MDP, the dual effect [4, 27] that current action influences both the next state and the model uncertainty is not considered by greedy model-based algorithms. Optimism in the Face of Uncertainty. A common provable exploration mechanism is to adopt the principle of optimism in the face of uncertainty (OFU) [56, 49, 10]. With OFU, the agent assigns to its policy an optimistically biased estimate of virtual value by jointly optimizing over the policies and models inside the confidence set. At iteration, the OFU-RL policy is defined as:
Optimism in the Face of Uncertainty. A common provable exploration mechanism is to adopt the principle of optimism in the face of uncertainty (OFU) [56, 49, 10]. With OFU, the agent assigns to its policy an optimistically biased estimate of virtual value by jointly optimizing over the policies and models inside the confidence set Ft. At iteration t, the OFU-RL policy πt is defined as:
Most asymptotic analyses of optimistic RL algorithms can be abstracted as showing two properties: the virtual value V f π is sufficiently high, and it is close to the real value V f ∗ π in the long run. However, in complex environments where the generalizability of nonlinear models is limited, large epistemic uncertainty will result in an unrealistically large optimistic return that drives agents for uninformative exploration. What’s worse, such suboptimal exploration steps eliminate only a small portion of the model hypothesis [13], leading to a slow converging process and suboptimal practical performance. Posterior Sampling Reinforcement Learning. An alternative exploration mechanism is based on Thompson Sampling (TS) [62, 52], which involves selecting the maximizing action from a statistically
(2.4)
(2.5)
(2.6)
(3.1)
plausibly set of action values. These values can be associated with the MDP sampled from its posterior distribution, thus giving its name posterior sampling for reinforcement learning (PSRL) [57, 42, 43]. The algorithm begins with a prior distribution of f ∗. At each iteration t, a model ft is sampled from the posterior φ(·|Ht), and πt is updated to be optimal under ft:
The insight is to keep away from actions that are unlikely to be optimal in the real MDP. Exploration is guaranteed by the randomness in the sampling procedure. Unfortunately, executing actions that are optimally associated with a single sampled model can cause similar over-exploration issues [52, 51]. Specifically, an imperfect model sampled from the large hypothesis can cause aggressive policy updates and value degradation between successive iterations. The suboptimality degree of the resulting policies depends on the epistemic model uncertainty. Besides, executing πt is not intended to offer performance improvement for follow-up policy learning, but only to narrow down the model uncertainty. However, this elimination procedure will be slow when the model suffers a large generalization error, which is quantitatively formulated in the model complexity measure below. Complexity Measure and Generalization Bounds. In RL, we seek to have the sample complexity for finding a near-optimal policy or estimating an accurate value function. When given access to a generative model (i.e., an abstract sampling model) in finite MDPs, it is known that the (minimax) number of transitions the agent needs to observe can be sublinear in the model size, i.e. smaller than O(|S|2|A|). Beyond finite MDPs where the number of states is large (or countably or uncountably infinite), we are interested in the learnability or generalization of RL. Unfortunately, it is impossible for agnostic reinforcement learning that finds the best hypothesis in some given policy, value, or model hypothesis class: the number of needed samples depends exponentially on the problem horizon [24]. Despite of the structural assumptions, e.g. linear MDPs [66, 22, 65] or low-rank MDPs [21, 38], we focus on the generalization bounds that can cover various cases. This can be done with additional complexity measure, e.g. eluder dimension [49], witness rank [60], or bilinear rank [14]. By introducing the eluder dimension dE [49], previous work [43, 44] established regret �O(√dET) for both OFU-RL and PSRL. Intuitively, the eluder dimension captures how effectively the model learned from observed data can extrapolate to future data, and permits sample efficiency in various (linear) cases. Nevertheless, it is shown in [13, 33] that even the simplest nonlinear models do not have a polynomially-bounded eluder dimension. The following result is from Thm. 5.2 in Dong et al. [13] and similar results are also established in [33]. Theorem 3.1 (Eluder Dimension of Nonlinear Models [13]). The eluder dimension dimE(F, ε) (c.f. Definition 5.6) of one-layer ReLU neural networks is at least Ω(ε−(d−1)), where d is the state-action dimension, i.e. (s, a) ∈Rd. With more layers, the requirement of ReLU activation can be relaxed. As a result, additional complexity is hidden in the eluder dimension, e.g. when we choose ε = T −1, regret �O(√dET) contains dE = Ω(T d−1) and is no longer sublinear in T. In this case, previous provable exploration mechanisms will lose the desired property of global optimality and sample efficiency, which is the underlying reason for the over-exploration issue.
# 4 Conservative Dual Policy Optimization
When using nonlinear models, e.g. neural networks, the over-exploration issue causes unfavorable performance in practice, in terms of slow convergence and suboptimal asymptotic values. To tackle this challenge, the key is to abandon the sampling process and have guarantees during training. In this regard, we propose Conservative Dual Policy Optimization (CDPO) that is simple yet provably efficient. By optimizing the policy following two successive update procedures iteratively, CDPO simultaneously enjoys monotonic policy value improvement and global optimality properties.
# 4.1 CDPO Framework
To begin with, consider the problem of maximizing the expected value, πt = argmaxπ E[V f ∗ π | Ht], where E[V f ∗| Ht] denotes the expected values over the posterior. Obviously, we have the expected value improvement guarantee E[V f ∗ πt | Ht] ≥E[V f ∗ πt−1 | Ht]. We can also perform expected value
(3.2)
maximization in a trust-region to guarantee iterative improvement under any f ∗. However, such updates will lose the desired global convergence guarantee and may get stuck at local maxima even with linear models. For this reason, we propose a dual procedure of policy optimization. Referential Update. The first update step returns an intermediate policy, denoted as qt. This step is a greedy one in the sense that qt is optimal with respect to the value of a single model �ft, which we call a reference model. Selecting a reference model and optimizing a policy w.r.t. it imitates the sampling-optimization procedure of PSRL. We will show in Section 5.1 that if we pose the constraint �ft ∈Ft, then CDPO achieves the same expected regret as PSRL, which implies global optimality. More importantly, policy optimization under �ft is more stable and can avoid the over-exploration issue in PSRL since we are free to set it as a steady reference between successive iterations. For example, we fix the reference model �ft as the least squares estimate �f LS t defined in (2.1), instead of a random model sampled from the large hypothesis that causes aggressive policy update. This gives us:
Constrained Conservative Update. The conservative update then follows as the second stage of CDPO, which takes input qt and returns the reactive policy πt+1:
� �� � � � �� where DTV(·, ·) stands for the total variation distance and η is the hyperparameter that characterizes the trust-region constraint and controls the degree of exploration.
Compared with OFU-RL and PSRL, the above exploration and policy updates are conservative since the policy maximizes the expectation of the model value, instead of a single model’s value (i.e. the optimistic model in OFU-RL and the sampled model in PSRL). The conservative update (4.2) avoids the pitfalls when the optimistic model or the posterior sampled model suffers large bias, which leads to aggressive policy updates and over-exploration during training. Notably, the term conservative in our work differs from previous use, e.g. Conservative Policy Iteration [23, 53]. While the latter refers to policy updates with constraints, ours is to emphasize the conservative range of randomness and the reduction of unnecessary over-exploration by shelving the sampling process. In our analysis, we follow previous work [43, 59, 10, 35] and assume access to a policy optimization oracle. In practice, the problem of finding an optimal policy under a given model can be approximately solved by model-based solvers listed below. More fine-grained analysis can be obtained by applying off-the-shelf results established for policy gradient or MPC for specific policy or model function classes. This, however, is beyond the scope of this paper.
# 4.2 Practical Algorithm
Algorithm 1 Practical CDPO Algorithm
Input: Prior φ, model-based policy optimization
solver MBPO(π, f, J ).
1: for iteration t = 1, ..., T do
2:
qt ←MBPO(·, �f LS
t
, (4.1))
3:
Sample N models {ft,n}N
n=1
4:
πt ←MBPO(qt, {ft,n}N
n=1, (4.2))
5:
Execute πt in the real MDP
6:
Update Ht+1 = Ht ∪{sh,t, ah,t, sh+1,t}h
7:
Update �f LS
t+1 and φ
8: end for
9: return policy πT
The pseudocode of CDPO is in Alg. 1. The model-based solver MBPO(π, f, J ) outputs the policy (qt or πt) that optimizes the objective J with access to model f. Several different types of solvers can be leveraged, e.g., model-augmented model-free policy optimization such as Dyna [61], model-based reparameterization gradient [18, 9], or model-predictive control [63]. Details of different optimization choices can be found in Appendix E. In experiments, we use Dyna and MPC solvers. With Pinsker’s inequality, the total variation constraint in (4.2) is replaced by the KL divergence [53, 2] in experiments. We follow
(4.1)
(4.2)
# 5 Analysis
In this section, we first show the statistical equivalence between CDPO and PSRL in terms of the same BayesRegret bound. Then we give the iterative policy value bound with monotonic improvement. Finally, we prove the global convergence of CDPO. The missing proofs can be found in the Appendix.
# 5.1 Statistical Equivalence between CDPO and PSRL
Sketch proof. We first sketch the general strategy in the PSRL analysis. Recall the definition of the Bayesian expected regret BayesRegret(T, π, φ) := E[�T t=1 Rt], where Rt = V f ∗ π∗−V f ∗ πt . PSRL breaks down Rt by adding and subtracting V ft πft , the value of the imagined optimal policy πft under a sampled model ft, i.e. πft = argmaxπ V ft π .
− − −  − where the second equality follows from the definition of the PSRL policy. Following the law of total expectation and the Posterior Sampling Lemma (e.g. Lemma 1 in [42]), we have E[V f ∗ π∗−V ft πft ] = 0 by noting that f ∗and ft are identically distributed conditioned upon Ht. Then we obtain
 − � where the first inequality follows from the simulation lemma under the L-Lipschitz value assumption [43]. The second inequality follows from the definition of ωt in (2.6) and the construction of confidence set such that P(f ∗∈�Ft) ≥1 −2δ and P(ft ∈�Ft, f ∗∈�Ft) ≥1 −4δ via a union bound. As more data is collected, the model uncertainty is reduced and the sum of confidence set width ωt will be sublinear in T (c.f. Lemma B.5 and B.6), indicating sublinear regret. When it comes to CDPO, we decompose the regret as
  where the CDPO policy πt is defined in (4.2). Since E[V f ∗ π∗−V ft πft ] = 0,
�  − � where the first inequality follows from the greediness of qt and πt in the dual update steps, i.e., V � ft πft ≤V � ft qt for any πft as well as E[V ft πt ] ≥E[V ft qt ]. The 8δT term is introduced since �ft ∈Ft and P(ft ∈�Ft, �ft ∈�Ft) ≥1 −2δ.
(5.1)
(5.2)
(5.3)

One motivation for the conservative update is that it maximizes (thus improves) the expected value over the posterior. In this section, we are interested in the policy value improvement under any unknown f ∗. Namely, we seek to have the iterative improvement bound J(πt) −J(πt−1), where the true objective J is defined in (2.3). We impose the following regularity conditions on the underlying MDP transition and the state-action visitation. Assumption 5.2 (Regularity Condition on MDP Transition). Assume that the MDP transition function f ∗: S × A →S is with additive σ-sub-Gaussian noise and bounded norm, i.e., ∥s∥2 ≤C. Assumption 5.3 (Regularity Condition on State-Action Visitation). We assume that there exists κ > 0 such that for any policy πt, t ∈[1, T],
� � where dρqt+1/dρπt is the Radon-Nikodym derivative of ρqt+1 with respect to ρπt. Theorem 5.4 (Policy Iterative Improvement). Suppose we have ∥�f(·, ·)∥≤C for �f ∈F where the model class F is finite. Define ι := maxs,a |Af ∗ π (s, a)|, where Af ∗ π is the advantage function defined as Af ∗ π (s, a) := Qf ∗ π (s, a) −V f ∗ π (s). With probability at least 1 −δ, the policy improvement between successive iterations is bounded by
where ∆(t) := Es∼ζ � V � ft qt (s) −V � ft qt−1(s) � ≥0 due to the greediness of qt.
�  �  � � The above theorem provides the iterative improvement bound following the CDPO algorithm. When H is large enough, the policy value improvement is at least ∆(t) by choosing a properly small η.
�     � The above theorem provides the iterative improvement bound following the CDPO algorithm. When H is large enough, the policy value improvement is at least ∆(t) by choosing a properly small η. In particular, the first term ∆(t) characterizes the policy improvement brought by the greedy exploitation in (4.1), and ∆(t) ≥0 since qt is optimal under the reference model �ft. The second term in (5.6) accounts for the generalization error of least square methods. Specifically, model �ft = �f LS t ∈Ft is trained to fit the history samples. However, we seek to have the model error bound over the stateaction visitation measure, which requires the deviation from the empirical mean to its expectation using Bernstein’s inequality and union bound. Finally, the trust-region constraint in (4.2) brings the 4ηα/(1 −γ) term, which reduces to zero if η is small. This makes intuitive sense as η controls the degree of conservative exploration.
# 5.3 Global Optimality of CDPO
We now analyze the global optimality of CDPO by studying its expected regret. As discussed in Section 3, agnostic reinforcement learning is impossible. Without structural assumptions, additional complexity measure is required for a generalization bound beyond finite settings. For this reason, we adopt the notation of eluder dimension [49, 43], defined as follows: Definition 5.5 ((F, ε)-Dependence). If we say (s, a) ∈S×A is (F, ε)-dependent on {(si, ai)}n i=1 ⊆ S × A, then
� �� �� �� �� Conversely, (s, a) ∈S × A is (F, ε)-independent of {(si, ai)}n i=1 if and only if it does not satisfy the definition for dependence. Definition 5.6 (Eluder Dimension). The eluder dimension dimE(F, ε) is the length of the longest possible sequence of elements in S×A such that for some ε′ ≥ε, every element is (F, ε′)-independent of its predecessors. We make the following assumption on the Lipschitz continuity of the value function.
(5.6)
Assumption 5.7 (Lipschitz Continuous Value). At iteration t, assume the value function V ft π for any policy π is Lipschitz continuous in the sense that |V ft π (s1) −V ft π (s2)| ≤Lt∥s1 −s2∥2. Notably, Assumption 5.7 holds under certain regularity conditions of the MDP, e.g. when the transition and rewards are Lipschitz continuous [5, 47]. Under this assumption, many RL settings can be satisfied [13], e.g., nonlinear models with stochastic Lipschitz policies and Lipschitz reward models, and is thus adopted by various model-based RL work [35, 7, 13]. We now study the global optimality of CDPO by the following expected regret theorem, which can be seen as a direct consequence of Theorem 5.1 that states the statistical equivalence between CDPO and PSRL. Theorem 5.8 (Expected Regret of CDPO). Let N(F, α, ∥·∥2) be the α-covering number of F. Denote dE := dimE(F, T −1) for the eluder dimension of F at precision 1/T. Under Assumption 5.2 and 5.7, the cumulative expected regret of CDPO in T iterations is bounded by � �
� � � � � � � Here, the covering number is introduced since we are considering F that may contain infinitely many functions, for which we cannot simply apply a union bound. Besides, β is the confidence parameter that contains f ∗with high probability (via concentration inequality). To clarify the asymptotics of the expected regret bound, we introduce another measure of dimensionality that captures the sensitivity of F to statistical overfitting. Corollary 5.9 (Asymptotic Bound). Define the Kolmogorov dimension w.r.t. function class F as
 � � The sublinear regret result permits the global optimality and sample efficiency for any model class with a reasonable complexity measure. Meanwhile, the iterative improvement theorem guarantees efficient exploration and good performance even when the model class is highly nonlinear.
# 6 Empirical Evaluation
# 6.1 Understanding Different Exploration Mechanisms
We first provide insights and evidence of why CDPO exploration can be more efficient in the tabular N-Chain MDPs, which have optimal right actions and suboptimal left actions at each of the N states. Settings and full results are provided in Appendix F.2. In Figure 1, we compare the posterior of CDPO and PSRL at the state that is the furthest away from the initial state, i.e. the state that is the hardest for the agents to reach and explore.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/71b6/71b64315-4855-436f-9582-2838baabafe6.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: CDPO and PSRL posterior on an 8-Chain MDP and a 15-Chain MDP, where the right actions are optimal. Figure 2: Regret curve of CDPO and PSRL when N = 8 and N = 15.</div>
When training starts, both algorithms have a large variance of value estimation. However, as training progresses, CDPO gives more accurate and certain estimates, but only for the optimal right actions not
(5.7)
for the suboptimal left actions, while PSRL agents explore both directions. This verifies the potential over-exploration issue in PSRL: as long as the uncertainty contains unrealistically large values, PSRL agents can perform uninformative exploration by acting suboptimally according to an inaccurate sampled model. In contrast, CDPO replaces the sampled model with a stable mean estimate and cares about the expected value, thus avoiding such pitfalls. We see in Figure 2 that although CDPO has much larger uncertainty for the suboptimal left actions, its regret is lower.
In finite MDPs, PSRL-style agents can specify and try every possible action to finally obtain an accurate high-confidence prediction. However, our discussion in Section 3 indicates that a similar over-exploration issue in more complex environments can lead to less informative exploration steps, which only eliminate an exponentially small portion of the uncertainty.
To see its impact on the training performance, we report the results of provable algorithms with nonlinear models on several MuJoCo tasks in Figure 3. For OFU-RL, we mainly evaluate HUCRL [10], a deep algorithm proposed to deal with the intractability of the joint optimization. We observe that all algorithms achieve asymptotic optimality in the inverted pendulum. Since the dimension of the pendulum task is low, learning an accurate (and thus generalizable) model poses no actual challenge However, in higher dimensional tasks such as half-cheetah, CDPO achieves a higher asymptotic value with faster convergence. Implementation details and hyperparameters are provided in Appendix F.1.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ce76/ce76234d-66d1-4b49-b718-f51e7fb68c9f.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) 7-DOF Pusher.</div>
<div style="text-align: center;">(a) Inverted Pendulum.</div>
<div style="text-align: center;">Figure 3: Performance of CDPO, PSRL, and HUCRL equipped with nonlinear models in several MuJoCo tasks: inverted pendulum swing-up, pusher goal-reaching, and half-cheetah locomotion.</div>
# 6.3 Comparison with Prior RL Algorithms
We also examine a broader range of MBRL algorithms, including MBPO [20], SLBO [35], and ME-TRPO [30]. The model-free baselines include SAC [16], PPO [54], and MPO [2]. The results are shown in Figure 4. We observe that CDPO achieves competitive or higher asymptotic performance while requiring fewer samples compared to both the model-based and the model-free baselines.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1adf/1adf2ae6-df53-4d80-adfa-e01f3f5d24eb.png" style="width: 50%;"></div>
<div style="text-align: center;"> Comparison between CDPO and model-free, model-based RL baseline </div>
We conduct ablation studies to provide a better understanding of the components in CDPO. One can observe from Figure 5 that the policies updated with only Referential Update or Conservative Update lag behind the dual framework. We also test the necessity and sensitivity of the constraint hyperparameter η. We see that a constant η and a time-decayed η achieve similar asymptotic values with a similar convergence rate, showing the robustness of CDPO. However, removing the constraint will lose the policy improvement guarantee, thus causing degradation. Ablation on different choices of MBPO solver (Dyna and POPLIN-P [63]) shows the generalizability of CDPO.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/28d8/28d85f6f-7df0-4702-9523-a7364bde4ab7.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Half-Cheetah.</div>
<div style="text-align: center;">(a) Inverted Pendulum.</div>
Figure 5: Ablation studies on the effect of the dual update steps and the trust-region constraint. The robustness and generalizability of the CDPO framework are demonstrated by the results of different choices of the constraint threshold and different solvers.
# 7 Conclusions & Future Work
In this work, we present Conservative Dual Policy Optimization (CDPO), a simple yet provable modelbased algorithm. By iterative execution of the Referential Update and Conservative Update, CDPO explores within a reasonable range while avoiding aggressive policy update. Moreover, CDPO gets rid of the harmful sampling procedure in previous provable approaches. Instead, an intermediate policy is optimized under a stable reference model, and the agent conservatively explore the environment by maximizing the expected policy value. With the same order of regret as PSRL, the proposed algorithm can achieve global optimality while monotonically improving the policy. Considering our naive choice of the reference model, other more sophisticated designs should be a fruitful future direction. It will also be interesting to explore different choices of the MBPO solvers, which we would like to leave as future work.
# References
<div style="text-align: center;">(c) Half-Cheetah.</div>
# Checklist
# A Proofs
# A.1 Proof of Theorem 5.4
Proof. We lay out the proof in two major steps. Firstly, we characterize the performance difference between J(qt) and J(πt−1), which can be done by applying Lemma B.3. Specifically, we set π1, π2 in Lemma B.3 to qt, πt−1 and set f as the reference model �ft. Then we obtain J(qt) −J(πt−1) � �
where ∆(t) := Es∼ζ � V � ft qt (s) −V � ft πt−1(s) � ≥0 due to the optimality of qt under �ft, i.e., qt = argmaxq V � ft q . Recall that the reference model is the least squares estimate, i.e.,
� � �� where Ht−1 is the trajectory in the real environment when following policy πt−1. From the simulation property of continuous distribution, we have the following equivalence between the direct and indirect ways of drawing samples:
where p(ϵ) is some noise distribution. Therefore, according to the Gaussian noise assumption, we obtain from the least squares generalization bound in Lemma B.4 that
���� �� � where ϵapprox = 0 in the generalization bound as the realizability is guaranteed since �f LS t and f ∗are from the same function class F. Similarly, we have for the intermediate policy qt that
Eρqt ����ft(s, a) −f ∗(·|s, a) �� 1 � ≤Eρπt−1 ����ft(s, a) −f ∗(·|s, a) �� 1 � · � Eρπt−1 ��dρqt dρπt−1 (s) �2�
Now we can bound (A.1) by
J(qt) −J(πt−1) ≥∆(t) −(1 + κ) · 22γC2 ln(|F|/δ) (1 −γ)H .
The second step of the proof is to characterize the performance difference between J(πt) and J(qt). From the Performance Difference Lemma B.2, we obtain
�� (A.1)
(A.2)
(A.4)
(A.5)
where recall that ι := maxs,a |Af ∗ π (s, a)| and the third equality holds due to Ea∼πt � Af ∗ πt (s, a) � = 0 for any s. By the definition of the total variation distance, we can further bound the absolute difference as
� � By the definition of the total variation distance, we can further bound the absolute difference as
� � By the definition of the total variation distance, we can further bound the absolute difference as
Thus, we have J(πt) −J(qt) ≥−2ηι/(1 −γ) and similarly J(qt−1) −J(πt−1) ≥−2ηι/(1 −γ). Combining with (A.4) gives us the iterative improvement bound as follows:
# A.2 Proof of Theorem 5.8
Proof. We are interested in the expected regret defined as BayesRegret(T, π, φ) := E[�T t=1 Rt], where Rt = V f ∗ π∗−V f ∗ πt . Recall the definition of the reactive policy πt in CDPO (i.e. (4.2)) and the imagined best-performing policy πft under a sampled model ft, i.e., πft = maxπ V ft π . From the Posterior Sampling Lemma, we know that if ψ is the distribution of f ∗, then for any sigma-algebra σ(Ht)-measurable function g, E[g(f ∗) | Ht] = E[g(ft) | Ht]. (A.8)
Proof. We are interested in the expected regret defined as BayesRegret(T, π, φ) := E[�T t=1 R where Rt = V f ∗ π∗−V f ∗ πt .
Recall the definition of the reactive policy πt in CDPO (i.e. (4.2)) and the imagined best-performing policy πft under a sampled model ft, i.e., πft = maxπ V ft π . From the Posterior Sampling Lemma, we know that if ψ is the distribution of f ∗, then for any sigma-algebra σ(Ht)-measurable function g, E[g(f ∗) | Ht] = E[g(ft) | Ht]. (A.8)
The PS Lemma together with the law of total expectation gives us E[V f ∗ π∗−V ft πft ] = 0,
where the equality holds since the true f ∗and the sampled ft are identically distributed when conditioned on Ht. Therefore, we obtain the expected regret for CDPO as
��� � � ��� where the first equation follows from Lemma B.1 and �ρπ is the state-action visitation measure under model �ft, the second inequality follows the simulation property of continuous distribution and the Lipschitz value function assumption.
(A.6)
(A.7)
(A.8)
(A.9)
(A.10)
(A.11)
� � We can further know from the construction of the confidence set (c.f. Lemma B.5) that P � f ∗∈ � t Ft � ≥1 −2δ and P(A) ≥1 −2δ since ft, f ∗are identically distributed and P ��ft ∈Ft � = 1 as Ft is centered at the least squares model for all t. Besides, we have for
Plugging into (A.21), we have
Summing over T iterations gives us
By setting δ = 1/(2T), we obtain
 −  �  − � � where the last inequality follows from Lemma B.6 to bound the sum of the set width. We denote dE := dimE(F, T −1) for notation simplicity. Since (A.16) holds for all policy π, we have the bound for E[V ft πft −V � ft πft ] and the bound for E[V � ft qt − V ft qt ]. What remains in the expected regret (A.19) is the E[V ft πt −V f ∗ πt ] term, which can be bounded similarly. Specifically, we define another event B = � f ∗∈� t Ft, ft ∈� t Ft � . Since by construction P � f ∗∈� t Ft � ≥1 −2δ and P � ft ∈� t Ft � ≥1 −2δ, we have P(B) ≥1 −4δ via a union bound. This implies the following bound
(A.12)
(A.13)
(A.14)
where the second inequality follows from the choice of δ, i.e., δ = 1/(2T) Plugging (A.16) and (A.17) into (A.19), we obtain the expected regret as
By setting α = 1/(T 2) and δ = 1/(2T) in Lemma B.5, we have the following confidence parameter that can guarantee that f ∗is contained in the confidence set with high probability: � � �
# � A.3 Proof of Theorem 5.1
Proof. Denote the imagined optimal policy πft under a sampled model ft as πft = maxπ V ft π . For PSRL, its expected regret can be decomposed as
� where the second equality holds since the PSRL policy πt := πft for a sampled ft. The third equality follows from (A.9), obtained by the Posterior Sampling Lemma and the law of total expectation. Similar with the proof in A.2, we obtain from the Simulation Lemma B.1 that � � � �
�  −  ��� � ���� � where the second inequality follows from the construction of confidence set that P � f ∗∈� t Ft � ≥  −2δ and thus P(E) ≥1 −4δ.
�� �� where the second inequality follows from the construction of confidence set that P � f ∗∈� t Ft � ≥ 1 −2δ and thus P(E) ≥1 −4δ.
(A.19)
(A.20)
(A.21)
� � From the proof in A.2, the expected regret of CDPO is bounded by
The claim is thus established.
# B Useful Lemmas
Lemma B.1 (Simulation Lemma). For any policy π and transition f1, f2, we have
Proof. Denote the expected reward under policy π as rπ. Let f π be the transition matrix on stateaction pairs induced by policy π, defined as f π (s,a),(s′,a′) := P(s′|s, a)π(a′|s′).
Since γ < 1, it is easy to verify that I −γf π is full rank and thus invertible. Therefore, we can write Vπ = (I −γf π)−1rπ. (B.2)
Therefore, we conclude the proof by
Lemma B.2 (Performance Difference Lemma). For all policies π, π∗and distribution µ over S, w have
Lemma B.2 (Performance Difference Lemma). For all policies π, π∗and distribution µ over S, we ave
Proof. This lemma is widely adopted in RL. Proof can be found in various previous works, e.g. Lemma 1.16 in [3]. Let Pπ(τ|s0 = s) denote the probability of observing trajectory τ starting at state s0 and then following π. Then the value difference can be written as
(A.23)

(B.1)
(B.2)
(B.3)
� � � where the third equation rearranges terms in the summation via telescoping, and the fourth equality follows from the law of total expectation. From the definition of objective J(π) in (2.3), we obtain
Lemma B.3 (Performance Difference and Model Error). For any two policies π1 and π2, it holds that
� � � � Proof. The proof can be established by combining the Performance Difference Lemma and the Simulation Lemma. We refer to Corollary 3.1 in [48] or Lemma A.3 in [59] for a detailed proof. Lemma B.4 (Least Squares Generalization Bound). Given a dataset H = {xi, yi}n i=1 where xi ∈X and xi, yi ∼ν, and yi = f ∗(xi)+ϵi. Suppose |yi| ≤Y and ϵi is independently sampled noise. Given a function class F : X →[0, Y ], we assume approximate realizable, i.e., minf∈F Ex∼ν � |f ∗(x) − f(x)|2� ≤ϵapprox. Denote �f as the least square solution, i.e., �f = argminf∈F �n i=1 � f(xi) −yi �2. With probability at least 1 −δ, we have
�� � Proof. The result is standard and can be proved by using the Bernstein’s inequality and union boun Detailed proof can be found at Lemma A.11 in [3]. Lemma B.5 (Confidence sets with high probability). If the control parameter βt(δ, α) is set to β(δ, α) = 8σ2 log(N(F, α, ∥·∥)/δ) + 2αt � 8C + � 8σ2 log(4t2/δ) � , (B.7
�� � Proof. The result is standard and can be proved by using the Bernstein’s inequality and union bound. Detailed proof can be found at Lemma A.11 in [3]. Lemma B.5 (Confidence sets with high probability). If the control parameter βt(δ, α) is set to βt(δ, α) = 8σ2 log(N(F, α, ∥·∥2)/δ) + 2αt � 8C + � 8σ2 log(4t2/δ) � , (B.7) then for all δ > 0, α > 0 and t ∈N, the confidence set Ft = Ft(βt(δ, α)) satisfies: � � �
Proof. See [43] Proposition 6 for a detailed proof.
(B.4)
(B.6)
(B.7)
(B.8)

(B.9)
# C Limitations of Eluder Dimension
In Theorem 5.8, the eluder dimension dE appears in the Bayes expected regret bound to capture how effectively the observed samples can extrapolate to unobserved transitions. For some specific function classes, Osband et al. [43] provide the corresponding eluder dimension bound, e.g., for (generalized) linear function classes, quadratic function class, and for finite MDPs, c.f. Proposition 1-4 in [43]. However, for non-linear models, Dong et al. [13] show that the ε-eluder dimension of one-layer neural networks is at least exponential in model dimension. Similar results are also established in [33]. We refer to Section 5 in [13] or Section 4 in [33] for details and more explanations.
# D Additional Related Work
Some MBRL work also concerns iterative policy improvement. SLBO [35] provides a trust-region policy optimization framework based on OFU. However, the conditions for monotonic improvement cannot be satisfied by most parameterized models [35, 13], which leads to a greedy algorithm in practice. Prior work that shares similarities with ours contains DPI [59] and GPS [31, 39] as dual policy optimization procedures are adopted. Both DPI and GPS leverage a locally accurate model and use different objectives for imitating the intermediate policy within a trust-region. However, the policy imitation procedure updates the policy parameter in a supervised manner, which poses additional challenges for effective exploration, resulting in unknown convergence results even with a simple model class. In contrast, CDPO by taking the epistemic uncertainty into consideration can be shown to achieve global optimality. In fact, greedy model exploitation is provably optimal only in very limited cases, e.g., linear-quadratic regulator (LQR) settings [36]. OFU-RL has shown to achieve an optimal sublinear regret when applied to online LQR [1], tabular MDPs [19] and linear MDPs [22]. Among them, HUCRL [10] is a deep algorithm proposed to deal with the joint optimization intractability in (3.1). Besides, Russo and Van Roy [49, 50] unify the bounds in various settings (e.g., finite or linear MDPs) by introducing an additional model complexity measure — eluder dimension. Other complexity measure include witness rank [60], linear dimensionality [66] and sequential Rademacher complexity [13].
# E Algorithm Instantiations
The model-based policy optimization solver MBPO(π, {f}, J ) in Algorithm 1 can be instantiated as one of the following algorithms, Dyna-style policy optimization in Algorithm 2, model-based back-propagation in Algorithm 3, and model predictive control policy optimization in Algorithm 4. By default, MBPO is instantiated as the Dyna solver (i.e. Algorithm 2) in our MuJoCo experiments and as the policy iteration solver in our N-Chain MDPs experiments. We note that the instantiations are not restricted to the listed algorithms, and many other MBPO algorithms that augment policy learning with a predictive model can also be leveraged, e.g., model-based value expansion [15, 6]. In the Referential Update step where no input policy exists in MBPO(·, �f LS t , (4.1), we initialize policy π = πt−1, i.e. the reactive policy from the last iteration. Dyna. Dyna involves model-generated data and optimizing the policy with any model-free RL method, e.g., REINFORCE or actor-critic [28]. The state-action value can be estimated by learning a critic function or unrolling the model. In Constrained Conservative Update, the input objective function J is (4.2), which is with constraints. Thus, the Lagrangian multiplier is introduced, similar to the model-free trust-region algorithms [53, 54, 2]. Back-Propagation Through Time. BPTT [30, 64] is a first-order model-based policy optimization framework based on pathwise gradient (or reparameterization gradient) [58]. There are also several variants including Stochastic Value Gradients (SVG) [18], Model-Augmented Actor-Critic (MAAC) [9], and Probabilistic Inference for Learning COntrol (PILCO) [12]. Specifically, the policy parameters are updated by directly computing the derivatives of the performance with respect to the parameters. When the optimization of objective function is constrained, the accumulating step (Algorithm 3
Algorithm 2 Dyna Model-Based Policy Optimization
Input: Policy π, model set {f}, objective function J .
1: Initialize a simulation data buffer �D
2: Sample a batch of initial states from the initial distribution ζ
3: ▷Data simulation
4: for initial state sample s0 do
5:
for model f in model set {f} do
6:
for timestep h = 1, ..., H do
7:
Sample action �ah ∼π(·|�sh)
8:
Sample simulation state �sh+1 ∼f(�sh, �ah)
9:
Append simulation data to buffer �D = �D ∪(�sh, �ah, rh, �sh+1)
10:
end for
11:
end for
12: end for
13: ▷Policy optimization with any model-free algorithm ModelFree
14: Objective optimization of policy on the simulated data π ←ModelFree( �D, π)
Algorithm 3 Model-Based Back-Propagation Policy Optimization
Input: Policy π, model set {f}, objective function J .
1: Initialize a simulation data buffer �D
2: Start from initial state s0
3: Reset L ←0
4: ▷Data simulation
5: for model f in model set {f} do
6:
for timestep h = 1, ..., H do
7:
Sample action �ah ∼π(·|�sh)
8:
Sample simulation state �sh+1 ∼f(�sh, �ah)
9:
Accumulate reward and constraint to L
10:
end for
11: end for
12: ▷Policy optimization
13: Compute policy gradient with back-propagation through time
14: Objective optimization of policy π ←PolicyGradient
Model Predictive Control Policy Optimization. MPC is a planning framework that directly generates optimal action sequences under the model. Different from the above model-augmented policy optimization methods, MPC policy optimization directly generates optimal action sequences under the model and then distills the policy. Specifically, the pseudocode in Algorithm 4 begins with initial actions generated by the policy. Then with a shooting method, e.g., the cross-entropy method (CEM), the actions are refined and the policy that generates these optimal actions are distilled. Below, the algorithm to obtain the refined actions EliteActions can be CEM with action noise added to the action or policy parameter, i.e., POPLIN-A and POPLIN-P in [63]. The policy can be updated by UpdatePolicy using behavior cloning. Policy Iteration for Tabular MDPs. In tabular settings where the state space S and action space A are discrete and countable, we can perform policy iteration under each model in the model set {f}. Here, the model is the tabular representation instead of function approximators. Based on the state-action values under various models, the optimal action at each state is the one that maximizes the weighted average of the values within the constraint of total variation distance.
Algorithm 4 Model Predictive Control Policy Optimization
Input: Policy π, model set {f}, objective function J , algorithm to update actions EliteActions,
algorithm to update policy UpdatePolicy.
1: Start from initial state s0
2: Reset J ←0
3: ▷Model-based planning
4: for model f in model set {f} do
5:
for timestep h = 1, ..., H do
6:
Sample action �ah ∼π(·|�sh)
7:
Sample simulation state �sh+1 ∼f(�sh, �ah)
8:
Accumulate reward and constraint to J
9:
end for
10: end for
11: a ←EliteActions(J, �a1:N)
12: ▷Policy distillation
13: π ←UpdatePolicy(a)
# F Experimental Settings and Results in N-Chain MDPs
F.1 Settings of MuJoCo Experiments
In the MuJoCo experiments, we use a 5-layer neural network to approximate the dynamical model. We use deterministic ensembles [8] to capture the model epistemic uncertainty. Specifically, different ensembles are learned with independent transition data to construct the 1-step ahead confidence interval at every timestep. Each ensemble is separately trained using Adam [26]. And the number of ensemble heads can be set to 3, 4, or, 5, each of which is shown to be able to provide considerable performance in our experiments. All the experiments are repeated with 6 random seeds. Since neural networks are not calibrated in general, i.e., the model uncertainty set is not guaranteed to contain the real dynamics, we follow HUCRL [10] to re-calibrate [29] the model. Our MuJoCo code is also built upon the HUCRL GitHub repository. When using the Dyna model-based policy optimization, the number of gradient steps for each optimization procedure in an iteration is set to 20. And we empirically find that the KL divergence (or total variance) constraint makes the algorithm more efficient when computing the argmax in the optimization step, since optimizing from πt−1 at iteration t needs fewer policy gradient steps if the policy update is constrained within a certain trust region. The task-specific and task-common settings and parameters are listed below in Table 1.
<div style="text-align: center;">Table 1: Experimental parameters. Inverted Pendulum Pushe</div>
Inverted Pendulum
Pusher
Half-Cheetah
episode length H
200
150
1000
dimension of state
4
23
18
dimension of action
1
7
6
action penalty
0.001
0.1
0.1
hidden nodes
(200, 200, 200, 200, 200)
activation function
Swish
optimizer
Adam
learning rate
10−3
# F.2 Experiments in N-Chain MDPs
Besides the experiments in MuJoCo, we also conduct tabular experiments in the N-Chain environment that is proposed in [37]. Specifically, there are in total 2 actions and N states in an MDP. The initial state is s1 and the agent can choose to go left or right at each of the N states. The left action always succeeds and moves the agent to the left state, giving reward r ∼N(0, δ2). Taking the right action at
state s1, . . . , sN−1 gives reward r ∼N(−δ, δ2) and succeeds with probability 1 −1/N, moving the agent to the right state and otherwise moving the agent to the left state. Taking the right action at sN gives reward r ∼N(1, δ2) and moves the agent back to s1 with probability 1 −1/N. We set δ = 0.1 exp (−N/4), such that going right is the optimal action at least up to N = 40. As the number of states N is increasing, the agent needs deep exploration (e.g. guided by uncertainty) instead of dithering exploration (e.g. epsilon-greedy exploration), such that the agent can keep exploring despite receiving negative rewards [45].
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/af5c/af5cdfe6-c5bc-46ac-9580-07e8aa602ab1.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Illustration of the N-Chain MDP. Blue arrows correspond to action right (optimal) and red arrows correspond to action left (suboptimal). The figure is copied from [37].</div>
For this reason, we evaluate the proposed algorithm CDPO and compare it with other Bayesian RL algorithms, including Bayesian Q-Learning (BQL) [11], Posterior Sampling for RL (PSRL) [42], the Uncertainty Bellman Equation (UBE) [46] and Moment Matching (MM) approach [37]. For CDPO, the dual optimization steps are solved by policy iteration, and the conservative update is performed within the total variation distance η = 0.2 (c.f. Policy Iteration for Tabular MDPs in Appendix E). We choose conjugate priors to represent the posterior distribution: we use a Categorical-Dirichlet model for discrete transition distribution at each (s, a), and a Normal-Gamma (NG) model for continuous reward distribution at each (s, a, s′).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b045/b045d231-fb02-4dd5-ab7c-ddcf1051e202.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Posterior evolution of CDPO algorithm in the 8-Chain MDP.</div>
Evolution of Posterior. Figure 7 demonstrates the evolution of the posterior of the CDPO algorithm in an 8-Chain MDP. As training progresses the posteriors concentrate on the true optimal state-action values and the behavior policy converges on the optimal one. The fast reduction of uncertainty is central to achieving principled and efficient exploration. Compared to the posterior evolution of the PSRL algorithm corresponding to the optimal actions, i.e. the bottom row of curves in Figure 8, the expected value estimates of CDPO are closer to the ground-truth, and the variance is also smaller. Notably, the variance of CDPO might be higher for suboptimal actions, e.g., s = 8, a = left (the last image of the first row in Figure 7). It is due to the conservative nature of CDPO that it only cares about the expected value, instead of the value of a sampled (imperfect) model as in PSRL. In other words, as long as the uncertainty is large, the PSRL agents can take suboptimal actions to explore the uninformative regions, which causes the inefficient over-exploration issue.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3860/38609666-77bd-45b4-abe1-c96be30f6f5e.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: Posterior evolution of PSRL algorithm in the 8-Chain MDP.</div>
Cumulative Regret. We compare CDPO and previous algorithms on the N-Chain MDPs with various state sizes N by measuring the cumulative regret of an oracle agent following the optimal policy. The results are shown in Figure 9. To make the performances comparable on the same scale, we also provide the normalized regret in Figure 10. We observe that when the size of state space N is relatively smaller, e.g. N ≤5, CDPO, PSRL, BQL, and MM algorithms achieve sublinear regret. The performances of these algorithms are also comparable, showing the necessity of deep exploration. On the contrary, Q-Learning which only relies on dithering exploration mechanisms fail to find the optimal strategy. However, as N is increasing, where the exploration must be effective for the agent to continually explore despite receiving negative rewards, the CDPO agents offer significantly lower cumulative regret and faster convergence.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bab4/bab4f02b-d3af-4958-8b11-b898c2ecf051.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: Comparison of cumulative regret.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7aa1/7aa157d1-6c22-46c3-812d-a56f550c7cbc.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10: Performance comparison in terms of regret to the oracle.</div>
# G Algorithmic Comparisons between MBRL Algorithm
We provide algorithmic comparisons of four MBRL frameworks, including greedy model exploitation algorithms, OFU-RL, PSRL, and the proposed CDPO algorithm.
algorithms, OFU-RL, PSRL, and the proposed CDPO algorithm. The differences mainly lie in the model selection and policy update procedures. The high-level pseudocode is given in Algorithm 5, 6, 7 and 8. Among them, the greedy model exploitation algorithm is a naive instantiation, where other instantiations can include the ones that augment Algorithm 5 with e.g., a dual framework that involves a locally accurate model and a supervised imitating procedure [59, 31]. In Algorithm 5, �ft can either be a probabilistic model or a deterministic model (with additive noise), which can be estimated via Maximum Likelihood Estimation (MLE) or minimizing the Mean Squared Error (MSE), respectively.
Algorithm 5 Naive Greedy Model Exploitation
1: for iteration t = 1, ..., T do
2:
Estimate model �ft via MLE or MSE
3:
Compute πt = argmaxπ V �
ft
π
4:
Execute πt in the real MDP
5:
Ht+1 = Ht ∪{sh,t, ah,t, sh+1,t}h
6: end for
7: return policy πT
Algorithm 7 PSRL Algorithm
1: for iteration t = 1, ..., T do
2:
Sample ft ∼φ(· | Ht)
3:
Compute πt = argmaxπ V ft
π
4:
Execute πt in the real MDP
5:
Ht+1 = Ht ∪{sh,t, ah,t, sh+1,t}h
6: end for
7: return policy πT
# H Societal Impact
For real-world applications, interactions with the system imply energy or economic costs. With practical efficiency, CDPO reduces the training investment and is aligned with the principle of responsible AI. However, as an RL algorithm, CDPO is unavoidable to introduce safety concerns, e.g., self-driving cars make mistakes during RL training. Although CDPO does not explicitly address them, it may be used in conjunction with safety controllers to minimize negative impacts, while drawing on its powerful MBRL roots to enable efficient learning.
Algorithm 6 OFU-RL Algorithm
1: for iteration t = 1, ..., T do
2:
Construct confidence set Ft
3:
Compute πt = argmaxπ,f∼Ft V ft
π
4:
Execute πt in the real MDP
5:
Ht+1 = Ht ∪{sh,t, ah,t, sh+1,t}h
6: end for
7: return policy πT
Algorithm 8 CDPO Algorithm
1: for iteration t = 1, ..., T do
2:
Referential Update qt following (4.1)
3:
Conservative Update πt following (4.2)
4:
Execute πt in the real MDP
5:
Ht+1 = Ht ∪{sh,t, ah,t, sh+1,t}h
6: end for
7: return policy πT
