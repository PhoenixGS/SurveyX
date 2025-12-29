# Uncertainty-based Offline Variational Bayesian Reinforcement Learning for Robustness under Diverse Data Corruptions
Rui Yang1,2, Jie Wang1,2∗, Guoping Wu1, Bin Li1 1University of Science and Technology of China 2MoE Key Laboratory of Brain-inspired Intelligent Perception and Cognition {yr0013, guoping}@mail.ustc.edu.cn {jiewangx, binli}@ustc.edu.cn
 Nov 2024
# Abstract
Real-world offline datasets are often subject to data corruptions (such as noise or adversarial attacks) due to sensor failures or malicious attacks. Despite advances in robust offline reinforcement learning (RL), existing methods struggle to learn robust agents under high uncertainty caused by the diverse corrupted data (i.e., corrupted states, actions, rewards, and dynamics), leading to performance degradation in clean environments. To tackle this problem, we propose a novel robust variational Bayesian inference for offline RL (TRACER). It introduces Bayesian inference for the first time to capture the uncertainty via offline data for robustness against all types of data corruptions. Specifically, TRACER first models all corruptions as the uncertainty in the action-value function. Then, to capture such uncertainty, it uses all offline data as the observations to approximate the posterior distribution of the action-value function under a Bayesian inference framework. An appealing feature of TRACER is that it can distinguish corrupted data from clean data using an entropy-based uncertainty measure, since corrupted data often induces higher uncertainty and entropy. Based on the aforementioned measure, TRACER can regulate the loss associated with corrupted data to reduce its influence, thereby enhancing robustness and performance in clean environments. Experiments demonstrate that TRACER significantly outperforms several state-of-the-art approaches across both individual and simultaneous data corruptions.
arXiv:2411.00465v1
# 1 Introduction
Offline reinforcement learning (RL) aims to learn an effective policy from a fixed dataset without direct interaction with the environment [1, 2]. This paradigm has recently attracted much attention in scenarios where real-time data collection is expensive, risky, or impractical, such as in healthcare [3], autonomous driving [4], and industrial automation [5]. Due to the restriction of the dataset, offline RL confronts the challenge of distribution shift between the policy represented in the offline dataset and the policy being learned, which often leads to the overestimation for out-of-distribution (OOD) actions [1, 6, 7]. To address this challenge, one of the promising approaches introduce uncertainty estimation techniques, such as using the ensemble of action-value functions or Bayesian inference to measure the uncertainty of the dynamics model [8–11] or the action-value function [12–15] regarding the rewards and transition dynamics. Therefore, they can constrain the learned policy to remain close to the policy represented in the dataset, guiding the policy to be robust against OOD actions.
∗Corresponding author. Email: jiewangx@ustc.edu.cn.
38th Conference on Neural Information Processing Systems (NeurIPS 2024).
38th Conference on Neural Information Processing Systems (NeurIPS 2024).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6a4c/6a4c8828-b411-44aa-ad99-28315209d255.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Graphical model of decision-making process. Nodes connected by solid lines denote data points in the offline dataset, while the Q values (i.e., action values) connected by dashed lines are not part of the dataset. These Q values are often objectives that offline algorithms aim to approximate.</div>
Nevertheless, in the real world, the dataset collected by sensors or humans may be subject to extensive and diverse corruptions [16–18], e.g., random noise from sensor failures or adversarial attacks during RLHF data collection. Offline RL methods often assume that the dataset is clean and representative of the environment. Thus, when the data is corrupted, the methods experience performance degradation in the clean environment, as they often constrain policies close to the corrupted data distribution. Despite advances in robust offline RL [2], these approaches struggle to address the challenges posed by diverse data corruptions [18]. Specifically, many previous methods on robust offline RL aim to enhance the testing-time robustness, learning from clean datasets and defending against attacks during testing [19–21]. However, they cannot exhibit robust performance using offline dataset with perturbations while evaluating the agent in a clean environment. Some related works for data corruptions (also known as corruption-robust offline RL methods) introduce statistical robustness and stability certification to improve performance, but they primarily focus on enhancing robustness against adversarial attacks [16, 22, 23]. Other approaches focus on the robustness against both random noise and adversarial attacks, but they often aim to address only corruptions in states, rewards, or transition dynamics [24, 17]. Based on these methods, recent work [18] extends the data corruptions to all four elements in the dataset, including states, actions, rewards, and dynamics. This work demonstrates the superiority of the supervised policy learning scheme [25, 26] for the data corruption of each element in the dataset. However, as it does not take into account the uncertainty in decisionmaking caused by the simultaneous presence of diverse corrupted data, this work still encounters difficulties in learning robust agents, limiting its applications in real-world scenarios. In this paper, we propose to use offline data as the observations, thus leveraging their correlations to capture the uncertainty induced by all corrupted data. Considering that (1) diverse corruptions may introduce uncertainties into all elements in the offline dataset, and (2) each element is correlated with the action values (see dashed lines in Figure 1), there is high uncertainty in approximating the action-value function by using various corrupted data. To address this high uncertainty, we propose to leverage all elements in the dataset as observations, based on the graphical model in Figure 1. By using the high correlations between these observations and the action values [27], we can accurately identify the uncertainty of the action-value function. Motivated by this idea, we propose a robust variational Bayesian inference for offline RL (TRACER) to capture the uncertainty via offline data against all types of data corruptions. Specifically, TRACER first models all data corruptions as uncertainty in the action-value function. Then, to capture such uncertainty, it introduces variational Bayesian inference [28], which uses all offline data as observations to approximate the posterior distribution of the action-value function. Moreover, the corrupted observed data often induce higher uncertainty than clean data, resulting in higher entropy in the distribution of action-value function. Thus, TRACER can use the entropy as an uncertainty measure to effectively distinguish corrupted data from clean data. Based on the entropy-based uncertainty measure, it can regulate the loss associated with corrupted data in approximating the action-value distribution. This approach effectively reduces the influence of corrupted samples, enhancing robustness and performance in clean environments. This study introduces Bayesian inference into offline RL for data corruptions. It significantly captures the uncertainty caused by diverse corrupted data, thereby improving both robustness and performance in offline RL. Moreover, it is important to note that, unlike traditional Bayesian online and offline RL methods that only model uncertainty from rewards and dynamics [29–35], our approach identifies the
uncertainty of the action-value function regarding states, actions, rewards, and dynamics under data corruptions. We summarize our contributions as follows. • To the best of our knowledge, this study introduces Bayesian inference into corruption-robust offline RL for the first time. By leveraging all offline data as observations, it can capture uncertainty in the action-value function caused by diverse corrupted data. • By introducing an entropy-based uncertainty measure, TRACER can distinguish corrupted from clean data, thereby regulating the loss associated with corrupted samples to reduce its influence for robustness. • Experiment results show that TRACER significantly outperforms several state-of-the-art offline RL methods across a range of both individual and simultaneous data corruptions.
# 2 Preliminaries
Bayesian RL. We consider a Markov decision process (MDP), denoted by a tuple M = (S, A, R, P, P0, γ), where S is the state space, A is the action space, R is the reward space, P(·|s, a) ∈P(S) is the transition probability distribution over next states conditioned on a stateaction pair (s, a), P0(·) ∈P(S) is the probability distribution of initial states, and γ ∈[0, 1) is the discount factor. Note that P(S) and P(A) denote the sets of probability distributions on subsets of S and A, respectively. For simplicity, throughout the paper, we use uppercase letters to refer to random variables and lowercase letters to denote values taken by the random variables. Specifically, R(s, a) denotes the random variable of one-step reward following the distribution ρ(r|s, a), and r(s, a) represents a value of this random variable. We assume that the random variable of one-step rewards and their expectations are bounded by Rmax and rmax for any (s, a) ∈S × A, respectively. Our goal is to learn a policy that maximizes the expected discounted cumulative return: � �
� Based on the return, we can define the value function as V π(s) = Eπ,ρ,P [�∞ t=0 γtR(st, at)|s0 = s the action-value function as Qπ(s, a) = ER∼ρ(·|s,a),s′∼P (·|s,a) [R(s, a) + γV π(s′)], and the actio value distribution [36] as
Dπ(s, a) = ∞ � γtR(st, at|s0 = s, a0 = a), with st+1 ∼P(·|st, at), at+1 ∼π(·|st+1).
Dπ(s, a) = ∞ � γtR(st, at|s0 = s, a0 = a), with st+1 ∼P(·|st, at), at+1 ∼π(·|st+1).
� Note that V π(s) = Ea∼π [Qπ(s, a)] = Ea∼π,ρ [Dπ(s, a)].
Variational Inference. Variational inference is a powerful method for approximating complex posterior distributions, which is effective for RL to handle the parameter uncertainty and deal with modelling errors [36]. Given an observation X and latent variables Z, Bayesian inference aims to compute the posterior distribution p(Z|X). Direct computation of this posterior is often intractable due to the high-dimensional integrals involved. To approximate the true posterior, Bayesian inference introduces a parameterized distribution q(Z; ϕ) and minimizes the Kullback-Leibler (KL) divergence DKL(q(Z; ϕ)∥p(Z|X)). Note that minimizing the KL divergence is equivalent to maximizing the evidence lower bound (ELBO) [37, 38]: ELBO(ϕ) = Eq(Z;ϕ)[log p(X, Z) −log q(Z; ϕ)]. Offline RL under Diverse Data Corruptions. In the real world, the data collected by sensors or humans may be subject to diverse corruption due to sensor failures or malicious attacks. Let b and B denotes the uncorrupted and corrupted dataset with samples {(si t, ai t, ri t, si t+1)}N i=1, respectively. Each data in B may be corrupted. We assume that an uncorrupted state follows a state distribution pb(·), a corrupted state follows pB(·), an uncorrupted action follows a behavior policy πb(·|si t), a corrupted action is sampled from πB(·|si t), a corrupted reward is sampled from ρB(·|si t, ai t), and a corrupted next state is drawn from PB(·|si t, ai t). We also denote the uncorrupted and corrupted empirical state-action distributions as pb(si t, ai t) and pB(si t, ai t), respectively. Moreover, we introduce the notations [18, 39] as follows. ˜T Q(s, a) = ˜r(s, a) + Es′∼PB(·|s,a) [V (s′)] , ˜r(s, a) = Er∼ρB(·|s,a)[r], (2) ˜T D(s, a) : D= R(s, a) + γD (s′, a′) , s′ ∼PB(·|s, a), a′ ∼πB(·|s), (3)
Variational Inference. Variational inference is a powerful method for approximating complex posterior distributions, which is effective for RL to handle the parameter uncertainty and deal with modelling errors [36]. Given an observation X and latent variables Z, Bayesian inference aims to compute the posterior distribution p(Z|X). Direct computation of this posterior is often intractable due to the high-dimensional integrals involved. To approximate the true posterior, Bayesian inference introduces a parameterized distribution q(Z; ϕ) and minimizes the Kullback-Leibler (KL) divergence DKL(q(Z; ϕ)∥p(Z|X)). Note that minimizing the KL divergence is equivalent to maximizing the evidence lower bound (ELBO) [37, 38]: ELBO(ϕ) = Eq(Z;ϕ)[log p(X, Z) −log q(Z; ϕ)].
(1)
(2) (3)
(2)
for any (s, a) ∈S × A and Q : S × A �→[0, rmax/(1 −γ)], where X : D= Y denotes equality of probability laws, that is the random variable X is distributed according to the same law as Y . To address the diverse data corruptions, based on IQL [26], RIQL [18] introduces quantile estimators with an ensemble of action-value functions {Qθi(s, a)}K i=1 and employs a Huber regression [40]: LQ (θi) = E(s,a,r,s′)∼B [lκ H (r + γVψ (s′) −Qθi(s, a))] , lκ H(x) = �1 2κx2, if |x| ≤κ |x| −1 2κ, if |x| > κ , (4) LV (ψ) = E(s,a)∼B [Lν 2 (Qα(s, a) −Vψ(s))] , Lν 2(x) = |ν −I(x < 0)| · x2. (5) Note that lκ H is the Huber loss, and Qα is the α-quantile value among {Qθi(s, a)}K i=1. RIQL then follows IQL [26] to learn the policy using weighted imitation learning with a hyperparameter β: Lπ(ϕ) = E(s,a)∼B [exp(β · Aα(s, a)) log πϕ(a|s)] , Aα(s, a) = Qα(s, a) −Vψ(s). (6)
for any (s, a) ∈S × A and Q : S × A �→[0, rmax/(1 −γ)], where X : D= Y denotes equality of probability laws, that is the random variable X is distributed according to the same law as Y . To address the diverse data corruptions, based on IQL [26], RIQL [18] introduces quantile estimators with an ensemble of action-value functions {Qθi(s, a)}K i=1 and employs a Huber regression [40]:
probability laws, that is the random variable X is distributed according to the same law as Y . To address the diverse data corruptions, based on IQL [26], RIQL [18] introduces quantile estimators with an ensemble of action-value functions {Qθi(s, a)}K i=1 and employs a Huber regression [40]: �
LV (ψ) = E(s,a)∼B [Lν 2 (Qα(s, a) −Vψ(s))] , Lν 2(x) = |ν −I(x < 0)| · x2. (5) Note that lκ H is the Huber loss, and Qα is the α-quantile value among {Qθi(s, a)}K i=1. RIQL then follows IQL [26] to learn the policy using weighted imitation learning with a hyperparameter β: Lπ(ϕ) = E(s,a)∼B [exp(β · Aα(s, a)) log πϕ(a|s)] , Aα(s, a) = Qα(s, a) −Vψ(s). (6)
Note that lκ H is the Huber loss, and Qα is the α-quantile value among {Qθi(s, a)}K i=1. RIQL then follows IQL [26] to learn the policy using weighted imitation learning with a hyperparameter β:
# 3 Algorithm
We first introduce the Bayesian inference for capturing the uncertainty caused by diverse corrupted data in Section 3.1. Then, we provide our algorithm TRACER with the entropy-based uncertainty measure in Section 3.2. Moreover, we provide the theoretical analysis for robustness, the architecture, and the detailed implementation of TRACER in Appendices A.1, B.1, and B.2, respectively.
# 1 Variational Inference for Uncertainty induced by Corrupted Data
We focus on corruption-robust offline RL to learn an agent under diverse data corruptions, i.e., random or adversarial attacks on four elements of the dataset. We propose to use all elements as observations, leveraging the data correlations to simultaneously address the uncertainty. By introducing Bayesian inference framework, our aim is to approximate the posterior distribution of the action-value function. At the beginning, based on the relationships between the action values and the four elements (i.e., states, actions, rewards, next states) in the offline dataset as shown in Figure 1, we define Dθ = Dθ(S, A, R) ∼pθ(·|S, A, R), parameterized by θ. Building on the action-value distribution, we can explore how to estimate the posterior of Dθ using the elements available in the offline data. Firstly, we start from the actions {ai t}N i=1 following the corrupted distribution πB and use them as observations to approximate the posterior of the action-value distribution under a variational inference. As the actions are correlated with the action values and all other elements in the dataset, the likelihood is pφa(A|D, S, R, S′), parameterized by φa. Then, under the variational inference framework, we maximize the posterior and derive to minimize the loss function based on ELBO: LD|A(θ, φa) = EB,pθ � DKL � pφa(A|Dθ, S, R, S′) ∥πB(A|S) � −EA∼pφa [log pθ(Dθ|S, A, R)] � , (7) where S, R, and S′ follow the offline data distributions pB, ρB, and PB, respectively. Secondly, we apply the rewards {ri t}N i=1 drawn from the corrupted reward distribution ρB as the observations. Considering that the rewards are related to the states, actions, and action values, we model the likelihood as pφr(R|D, S, A), parameterized by φr. Therefore, we can derive a loss function by following Equation (7): LD|R(θ, φr) = EB,pθ � DKL � pφr(R|Dθ, S, A) ∥ρB(R|S, A) � −ER∼pφr [log pθ(Dθ|S, A, R)] � , (8) where S and A follow the offline data distributions pB and πB, respectively. Finally, we employ the state {si t}N i=1 in the offline dataset following the corrupted distribution pB as the observations. Due to the relation of the states, we can model the likelihood as pφs(S|D, A, R), parameterized by φr. We then have the loss function: LD|S(θ, φs) = EB,pθ � DKL � pφs(S|Dθ, A, R) ∥pB(S) � −ES∼pφs [log pθ(Dθ|S, A, R)] � , (9) where A and R follow the offline data distributions πB and ρB, respectively. We present the detailed derivation process in Appendix A.2. The goal of first terms in Equations (7), (8), and (9) is to estimate πB(A|S), ρB(R|S, A), and pB(S) using pφa(A|Dθ, S, R, S′), pφr(R|Dθ, S, A), and pφs(S|Dθ, A, R), respectively. As we do not
 (4)
(5)
(6)
� (7)
have the explicit expression of distributions πB, ρB, and pB, we cannot directly compute the KL divergence in these first terms. To address this issue, based on the generalized Bayesian inference [41], we can exchange two distributions in the KL divergence. Then, we model all the aformentioned distributions as Gaussian distributions, and use the mean µφ and standard deviation Σφ to represent the corresponding pφ. For implementation, we directly employ MLPs to output each (µφ, Σφ) using the corresponding conditions of pφ. Then, based on the KL divergence between two Gaussian distributions, we can derive the loss function as follows.
Lfirst(θ, φs, φa, φr) = 1 2E(s,a,r)∼B,Dθ∼pθ � (µφa −a)T Σ−1 φa (µφa −a) + (µφr −r)T Σ−1 φr (µφr − �
� Moreover, the goal of second terms in Equations (7), (8), and (9) is to maximize the likelihoods of Dθ given samples ˆs ∼pφs, ˆa ∼pφa, or ˆr ∼pφr. Thus, with (s, a, r) ∼B, we propose minimizing the distance between Dθ(ˆs, a, r) and D(s, a, r), Dθ(s, ˆa, r) and D(s, a, r), and Dθ(s, a, ˆr) and D(s, a, r), where ˆs ∼pφs, ˆa ∼pφa, and ˆr ∼pφr. Then, based on [41], we can derive the following loss with any metric ℓto maximize the log probabilities: � � �
Lsecond(θ, φs, φa, φr) = E(s,a,r)∼B,ˆs∼pφs,ˆa∼pφa,ˆr∼pφr ,D∼p � ℓ � D(s, a, r), Dθ(s, ˆa, r) � + ℓ � D(s, a, r), Dθ(s, a, ˆr) � + ℓ � D(s, a, r), Dθ(ˆs, a, r) �� .
# � � � 2 Corruption-Robust Algorithm with the Entropy-based Uncertain
We focus on developing tractable loss functions for implementation in this subsection. Learning the Action-Value Distribution based on Temporal Difference (TD). Based on [42, 43 we introduce the quantile regression [44] to approximate the action-value distribution in the offlin dataset B using an ensemble model {Dθi}K i=1. We use Equation (4) to derive the loss as:  
 developing tractable loss functions for implementation in this subsection
Learning the Action-Value Distribution based on Temporal Difference (TD). Based on [42, 43], we introduce the quantile regression [44] to approximate the action-value distribution in the offline dataset B using an ensemble model {Dθi}K i=1. We use Equation (4) to derive the loss as:  
where ρκ τ (δ) = |τ −I {δ < 0}| · lκ H (δ) with the threshold κ, Z denotes the value distribution, δτ,τ ′ θi is the sampled TD error based on the parameters θi, τ and τ ′ are two samples drawn from a uniform distribution U([0, 1]), Dτ θ(s, a, r) := F −1 Dθ(s,a,r)(τ) is the sample drawn from pθ(·|s, a, r), Zτ(s) := F −1 Z(s)(τ) is sampled from p(·|s), F −1 X (τ) is the inverse cumulative distribution function (also known as quantile function) [45] at τ for the random variable X, and N and N ′ represent the respective number of iid samples τ and τ ′. Notably, based on [43], we have Qθi(s, a) = �N n=1 Dτn θi (s, a, r). In addition, if we learn the value distribution Z, the action-value distribution can extract the information from the next states based on Equation (12), which is effective for capturing the uncertainty. On the contrary, if we directly use the next states in the offline dataset as the observations, in practice, the parameterized model of the action-value distribution needs to take (s, a, r, s′, a′, r′, s′′) as the input data. Thus, the model can compute the action values and values for the sampled TD error in Equation (12). To avoid the changes in the input data caused by directly using next states as observations in Bayesian inference, we draw inspiration from IQL and RIQL to learn a parameterized value distribution. Based on Equations (5) and (12), we derive a new objective as:
� where Dτ α is the α-quantile value among {Dτ θi(s, a)}K i=1, and Vψ(s) = �N n=1 Zτn ψ (s). More details are shown in Appendix B.2. Furthermore, we provide the theoretical analysis in Appendix A.1 to give a value bound between the value distributions under clean and corrupted data. Updating the Action-Value Distribution based on Variational Inference for Robustness. We discuss
where Dτ α is the α-quantile value among {Dτ θi(s, a)}K i=1, and Vψ(s) = �N n=1 Zτn ψ (s). More detail are shown in Appendix B.2. Furthermore, we provide the theoretical analysis in Appendix A.1 to give a value bound between the value distributions under clean and corrupted data. Updating the Action-Value Distribution based on Variational Inference for Robustness. We discus the detailed implementation of Equations (10) and (11) based on Equations (12) and (13). As the data
  Updating the Action-Value Distribution based on Variational Inference for Robustness. We discuss the detailed implementation of Equations (10) and (11) based on Equations (12) and (13). As the data
−r)
(10)
(12)
(13)
corruptions may introduce heavy-tailed targets [18], we apply the Huber loss to replace all quadratic loss in Equation (10) and the metric ℓin Equation (11), mitigating the issue caused by heavy-tailed targets [46] for robustness. We rewrite Equation (11) as follows.
+ lκ H � Dτn(s, a, r), Dτn θi (s, a, ˆr) � + lκ H � Dτn(s, a, r), Dτn θi (ˆs, a, r) � �
Thus, we have the whole loss function LD|S,A,R = Lfirst(θi, φs, φa, φr) + Lsecond(θi, φs, φa, φr) in the generalized variational inference framework. Moreover, based on the assumption of heavy-tailed noise in [18], we have a upper bound of action-value distribution by using the Huber regression loss. Entropy-based Uncertainty Measure for Regulating the Loss associated with Corrupted Data. To further address the challenge posed by diverse data corruptions, we consider the problem: how to exploit uncertainty to further enhance robustness? Considering that our goal is to improve performance in clean environments, we propose to reduce the influence of corrupted data, focusing on using clean data to learn agents. Therefore, we provide a two-step plan: (1) distinguishing corrupted data from clean data; (2) regulating the loss associated with corrupted data to reduce its influence, thus enhancing the performance in clean environments. For (1), as the Shannon entropy for the measures of aleatoric and epistemic uncertainties provides important insight [47–49], and the corrupted data often results in higher uncertainty and entropy of the action-value distribution than the clean data, we use entropy [50] to quantify uncertainties of corrupted and clean data. Furthermore, by considering that the exponential function can amplify the numerical difference in entropy between corrupted and clean data, we propose the use of exponential entropy [51]—a metric of extent of a distribution—to design our uncertainty measure. Specifically, based on Equation 12, we can use the quantile points {τn}N n=1 to learn the corresponding quantile function values {Dτn}N n=1 drawn from the action-value distribution pθ. We sort the quantile points and their corresponding function values in ascending order based on the values. Thus, we have the sorted sets {ςn}N n=1, {Dςn}N n=1, and the estimated PDF values {ςn}N n=1, where ς1 = ς1 and ςn = ςn −ςn−1 for 1 < n ≤N. Then, we can further estimate differential entropy following [52] (see Appendix A.3 for a detailed derivation).
where ˆςn denotes (ςn−1 + ςn)/2 for 1 < n ≤N, and D ςn denotes Dςn −Dςn−1 for 1 < n ≤N. For (2), TRACER employs the reciprocal value of exponential entropy 1/ exp(H(pθi)) to weight the corresponding loss of θi in our proposed whole loss function LD|S,A,R. Therefore, during the learning process, TRACER can regulate the loss associated with corrupted data and focus on minimizing the loss associated with clean data, enhancing robustness and performance in clean environments. Note that we normalize entropy values by dividing the mean of samples (i.e., quantile function values) drawn from action-value distributions for each batch. In Figure 3, we show the relationship of entropy values of corrupted and clean data estimated by Equation (15) during the learning process. The results illustrate the effectiveness of the entropy-weighted technique for data corruptions. Updating the Policy based on the Action-Value Distribution. We directly applies the weighted imitation learning technique in Equation (6) to learn the policy. As Qα(s, a) is the α-quantile value among {Qθi(s, a)}K i=1 = ��N n=1 Dτn θi (s, a, r) �K i=1 and Vψ(s) = �N n=1 Zτn ψ (s), we have
 (14)
(15)
(16)
<div style="text-align: center;">and standard errors under random and adversarial simultaneous corruptio</div>
<div style="text-align: center;">e 1: Average scores and standard errors under random and adversarial sim</div>
Env
Corrupt
BC
EDAC
MSG
UWMSG
CQL
IQL
RIQL
TRACER (ours)
Halfcheetah
random
23.17 ± 0.43
1.70 ± 0.80
9.97 ± 3.44
8.31 ± 1.25
14.25 ± 1.39
24.82 ± 0.57
29.94 ± 1.00
33.04 ± 0.42
advers
16.37 ± 0.32
0.90 ± 0.30
3.60 ± 0.89
3.13 ± 0.85
5.61 ± 2.21
11.06 ± 0.45
17.85 ± 1.39
19.72 ± 2.80
Walker2d
random
13.77 ± 1.05
−0.13 ± 0.01
−0.15 ± 0.11
4.36 ± 1.95
0.63 ± 0.36
12.35 ± 2.03
17.42 ± 2.95
23.62 ± 2.33
advers
6.75 ± 0.33
−0.17 ± 0.01
3.77 ± 1.09
4.19 ± 2.82
4.23 ± 1.35
16.61 ± 2.73
9.20 ± 1.40
17.21 ± 1.62
Hopper
random
18.49 ± 0.52
0.80 ± 0.01
15.84 ± 2.47
12.22 ± 2.11
3.16 ± 1.07
25.28 ± 15.34
22.50 ± 10.01
28.83 ± 7.06
advers
17.34 ± 1.00
0.80 ± 0.01
12.14 ± 0.71
10.43 ± 0.94
0.10 ± 0.34
19.56 ± 1.08
24.71 ± 6.20
24.80 ± 7.14
Average score
15.98
0.65
7.53
7.11
4.66
18.28
20.27
24.54
<div style="text-align: center;">Table 2: Average score under diverse random corruptions.</div>
Env
Corrupted Element
BC
EDAC
MSG
UWMSG
CQL
IQL
RIQL
TRACER (ours)
Halfcheetah
observation
33.4 ± 1.8
2.1 ± 0.5
−0.2 ± 2.2
2.9 ± 0.1
9.0 ± 7.5
21.4 ± 1.9
27.3 ± 2.4
34.2 ± 0.9
action
36.2 ± 0.3
47.4 ± 1.3
52.0 ± 0.9
56.0 ± 0.4
19.9 ± 21.3
42.2 ± 1.9
42.9 ± 0.6
42.9 ± 0.6
reward
35.8 ± 0.9
38.6 ± 0.3
17.5 ± 16.4
35.6 ± 0.4
32.6 ± 19.6
42.3 ± 0.4
43.6 ± 0.6
40.0 ± 1.1
dynamics
35.8 ± 0.9
1.5 ± 0.2
1.7 ± 0.4
2.9 ± 0.1
29.2 ± 4.0
36.7 ± 1.8
43.1 ± 0.2
43.8 ± 3.0
Walker2d
observation
9.6 ± 3.9
−0.2 ± 0.3
−0.4 ± 0.1
6.2 ± 0.5
19.4 ± 1.6
27.2 ± 5.1
28.4 ± 7.7
32.7 ± 2.8
action
18.1 ± 2.1
83.2 ± 1.9
25.3 ± 10.6
31.5 ± 10.6
62.7 ± 7.2
71.3 ± 7.8
84.6 ± 3.3
86.7 ± 6.2
reward
16.0 ± 7.4
4.3 ± 3.6
18.4 ± 9.5
62.0 ± 3.7
69.4 ± 7.4
65.3 ± 8.4
83.2 ± 2.6
85.5 ± 3.1
dynamics
16.0 ± 7.4
−0.1 ± 0.0
7.4 ± 3.7
0.2 ± 0.0
−0.2 ± 0.1
17.7 ± 7.3
78.2 ± 1.8
75.9 ± 1.8
Hopper
observation
21.5 ± 2.9
1.0 ± 0.5
6.9 ± 5.0
12.8 ± 0.4
42.8 ± 7.0
52.0 ± 16.6
62.4 ± 1.8
62.7 ± 8.2
action
22.8 ± 7.0
100.8 ± 0.5
37.6 ± 6.5
53.4 ± 5.4
69.8 ± 4.5
76.3 ± 15.4
90.6 ± 5.6
92.8 ± 2.5
reward
19.5 ± 3.4
2.6 ± 0.7
24.9 ± 4.3
60.8 ± 7.5
70.8 ± 8.9
69.7 ± 18.8
84.8 ± 13.1
85.7 ± 1.4
dynamics
19.5 ± 3.4
0.8 ± 0.0
12.4 ± 4.9
6.1 ± 1.3
0.8 ± 0.0
1.3 ± 0.5
51.5 ± 8.1
49.8 ± 5.3
Average score
23.7
23.5
17.0
27.5
35.5
43.6
60.0
61.1
<div style="text-align: center;">Table 3: Average score under diverse adversarial corruptions.</div>
Env
Corrupted Element
BC
EDAC
MSG
UWMSG
CQL
IQL
RIQL
TRACER (ours)
Halfcheetah
observation
34.5 ± 1.5
1.1 ± 0.3
1.1 ± 0.2
1.9 ± 0.1
5.0 ± 11.6
32.6 ± 2.7
35.7 ± 4.2
36.8 ± 2.1
action
14.0 ± 1.1
32.7 ± 0.7
37.3 ± 0.7
36.2 ± 1.0
−2.3 ± 1.2
27.5 ± 0.3
31.7 ± 1.7
33.4 ± 1.2
reward
35.8 ± 0.9
40.3 ± 0.5
47.7 ± 0.4
43.8 ± 0.3
−1.7 ± 0.3
42.6 ± 0.4
44.1 ± 0.8
41.9 ± 0.2
dynamics
35.8 ± 0.9
−1.3 ± 0.1
−1.5 ± 0.0
5.0 ± 2.2
−1.6 ± 0.0
26.7 ± 0.7
35.8 ± 2.1
36.2 ± 1.2
Walker2d
observation
12.7 ± 5.9
−0.0 ± 0.1
2.9 ± 2.7
6.3 ± 0.7
61.8 ± 7.4
37.7 ± 13.0
70.0 ± 5.3
70.0 ± 6.7
action
5.4 ± 0.4
41.9 ± 24.0
5.4 ± 0.9
5.9 ± 0.4
27.0 ± 7.5
27.5 ± 0.6
66.1 ± 4.6
69.3 ± 4.6
reward
16.0 ± 7.4
57.3 ± 33.2
9.6 ± 4.9
35.1 ± 10.5
67.0 ± 6.1
73.5 ± 4.9
85.0 ± 1.5
88.9 ± 4.7
dynamics
16.0 ± 7.4
4.3 ± 0.9
0.1 ± 0.2
1.8 ± 0.2
3.9 ± 1.4
−0.1 ± 0.1
60.6 ± 21.8
64.0 ± 16.5
Hopper
observation
21.6 ± 7.1
36.2 ± 16.2
16.0 ± 2.8
15.0 ± 1.3
78.0 ± 6.5
32.8 ± 6.4
50.8 ± 7.6
64.5 ± 3.7
action
15.5 ± 2.2
25.7 ± 3.8
23.0 ± 2.1
27.7 ± 1.3
32.2 ± 7.6
37.9 ± 4.8
63.6 ± 7.3
67.2 ± 3.8
reward
19.5 ± 3.4
21.2 ± 1.9
22.6 ± 2.8
30.3 ± 4.2
49.6 ± 12.3
57.3 ± 9.7
65.8 ± 9.8
64.3 ± 1.5
dynamics
19.5 ± 3.4
0.6 ± 0.0
0.6 ± 0.0
0.7 ± 0.0
0.6 ± 0.0
1.3 ± 1.1
65.7 ± 21.1
61.1 ± 6.2
Average score
20.5
21.7
13.7
17.5
26.6
33.1
56.2
58.1
# 4 Experiments
In this section, we show the effectiveness of TRACER across various simulation tasks using diverse corrupted offline datasets. Firstly, we provide our experiment setting, focusing on the corruption settings for offline datasets. Then, we illustrate how TRACER significantly outperforms previous stateof-the-art approaches under a range of both individual and simultaneous data corruptions. Finally, we conduct validation experiments and ablation studies to show the effectiveness of TRACER.
# 4.1 Experiment Setting
Building upon RIQL [18], we use two hyperparameters, i.e., corruption rate c ∈[0, 1] and corruption scale ϵ, to control the corruption level. Then, we introduce the random corruption and adversarial corruption in four elements (i.e., states, actions, rewards, next states) of offline datasets. The implementation of random corruption is to add random noise to elements of a c portion of the offline datasets, and the implementation of adversarial corruption follows the Projected Gradient Descent attack [53, 54] using pretrained value functions. Note that unlike other adversarial corruptions, the adversarial reward corruption multiplies −ϵ to the clean rewards instead of using gradient optimization. We also introduce the random or adversarial simultaneous corruption, which refers to random or adversarial corruption simultaneously present in four elements of the offline datasets. We apply the corruption rate c = 0.3 and corruption scale ϵ = 1.0 in our experiments.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6e38/6e38b167-3599-4859-acb0-683a6abd1f7a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: In the left, we report the means and standard deviations on CARLA under random simultaneous corruptions. In the right, we report the results with random simultaneous corruptions against different corruption levels.</div>
Figure 2: In the left, we report the means and standard deviations on CARLA under random simultaneous corruptions. In the right, we report the results with random simultaneous corruptions against different corruption levels.
We conduct experiments on D4RL benchmark [55]. Referring to RIQL, we train all agents for 3000 epochs on the ’medium-replay-v2’ dataset, which closely mirrors real-world applications as it is collected during the training of a SAC agent. Then, we evaluate agents in clean environments, reporting the average normalized performance over four random seeds. See Appendix C for detailed information. The algorithms we compare include: (1) CQL [7] and IQL [26], offline RL algorithms using a twin Q networks. (2) EDAC [56] and MSG [57], offline RL algorithms using ensemble Q networks (number of ensembles > 2). (3) UWMSG [17] and RIQL, state-of-the-arts in corruption-robust offline RL. Note that EDAC, MSG, and UWMSG are all uncertainty-based offline RL algorithms.
# 4.2 Main results under Diverse Data Corruptions
We conduct experiments on MuJoCo [58] (see Tables 1, 2, and 3, which highlight the highest results) and CARLA [59] (see the left of Figure 2) tasks from D4RL under diverse corruptions to show the superiority of TRACER. In Table 1, we report all results under random or adversarial simultaneous data corruptions. These results show that TRACER significantly outperforms other algorithms in all tasks, achieving an average score improvement of +21.1%. In the left of Figure 2, results on ’CARLA-lane_v0’ under random simultaneous corruptions also illustrate the superiority of TRACER. See Appendix C.2 for details.
Random Corruptions. We report the results under random simultaneous data corruptions of all algorithms in Table 1. Such results demonstrate that TRACER achieves an average score gain of +22.4% under the setting of random simultaneous corruptions. Based on the results, it is clear that many offline RL algorithms, such as EDAC, MSG, and CQL, suffer the performance degradation under data corruptions. Since UWMSG is designed to defend the corruptions in rewards and dynamics, its performance degrades when faced with the stronger random simultaneous corruption. Moreover, we report results across a range of individual random data corruptions in Table 2, where TRACER outperforms previous algorithms in 7 out of 12 settings. We then explore hyperparameter tuning on Hopper task and further improve TRACER’s results, demonstrating its potential for performance gains. We provide details in Appendix C.3.1.
Adversarial Corruptions. We construct experiments under adversarial simultaneous corruptions to evaluate the robustness of TRACER. The results in Table 1 show that TRACER surpasses others by a significant margin, achieving an average score improvement of +19.3%. In these simultaneous corruption, many algorithms experience more severe performance degradation compared to the random simultaneous corruption, which indicates that adversarial attacks are more damaging to the reliability of algorithms than random noise. Despite these challenges, TRACER consistently achieves significant performance gains over other methods. Moreover, we provide the results across a range of individual adversarial data corruptions in Table 3, where TRACER outperforms previous algorithms in 7 out of 12 settings. We also explore hyperparameter tuning on Hopper task and further improve TRACER’s results, demonstrating its potential for performance gains. See Appendix C.3.1 for details.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1349/1349425f-b600-4469-80f9-6e95b26791df.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: In the first column, we report the mean and standard deviation to show the superiority of using the entropy-based uncertainty measure. In the second and third columns, we report the results over three seeds to show the higher entropy of corrupted data compared to clean data during training.</div>
# 4.3 Evaluation of TRACER under Various Corruption Levels
Building upon RIQL, we further extend our experiments to include Mujoco datasets with various corruption levels, using different corruption rates c and scales ϵ. We report the average scores and standard deviations over four random seeds in the right of Figure 2, using batch sizes of 256. Results in the right of Figure 2 demonstrate that TRACER significantly outperforms baseline algorithms in all tasks under random simultaneous corruptions with various corruption levels. It achieves an average score improvement of +33.6%. Moreover, as the corruption levels increase, the slight decrease in TRACER’s results indicates that while TRACER is robust to simultaneous corruptions, its performance depends on the extent of corrupted data it encounters. We also evaluate TRACER in different scales of corrupted data and provide the results in Appendix C.4.1.
# 4.4 Evaluation of the Entropy-based Uncertainty Measure
We evaluate the entropy-based uncertainty measure in action-value distributions to show: (1) is the uncertainty caused by corrupted data higher than that of clean data? (2) is regulating the loss associated with corrupted data effective in improving performance?
For (1), we first introduce labels indicating whether the data is corrupted. Importantly, these labels are not used by agents during the training process. Then, we estimate entropy values of labelled corrupted and clean data in each batch based on Equation (15). Thus, we can compare entropy values to compute results, showing how many times the entropy of the corrupted data is higher than that of clean data. Specifically, we evaluate the accuracy every 50 epochs over 3000 epochs. For each evaluation, we sample 500 batches to compute the average entropy of corrupted and clean data. Each batch consists of 32 clean and 32 corrupted data. We illustrate the curves over three seeds in the second and third columns of Figure 3, where each point shows how many of the 500 batches have higher entropy for corrupted data than that of clean data. Figure 3 indicates an oscillating upward trend of TRACER’s measurement accuracy using entropy (TRACER Using Entro) under simultaneous corruptions, demonstrating that using the entropy-based uncertainty measure can effectively distinguish corrupted data from clean data. These curves also reveal that even in the absence of any constraints on entropy (TRACER NOT using Entro), the entropy associated with corrupted data tends to exceed that of clean data. For (2), in the first column of Figure 3, these results demonstrate that TRACER using the entropybased uncertainty measure can effectively reduce the influence of corrupted data, thereby enhancing robustness and performance against all corruptions. We provide detailed information for this evaluation in Appendix C.4.2.
# 5 Related Work
Robust RL. Robust RL can be categorized into two types: testing-time robust RL and training-time robust RL. Testing-time robust RL [19, 20] refers to training a policy on clean data and ensuring its robustness by testing in an environment with random noise or adversarial attacks. Training-time robust RL [16, 17] aims to learn a robust policy in the presence of random noise or adversarial attacks during training and evaluate the policy in a clean environment. In this paper, we focus on training-time robust RL under the offline setting, where the offline training data is subject to various data corruptions, also known as corruption-robust offline RL. Corruption-Robust RL. Some theoretical work on corruption-robust online RL [60–63] aims to analyze the sub-optimal bounds of learned policies under data corruptions. However, these studies primarily address simple bandits or tabular MDPs and focus on the reward corruption. Some further work [64, 65] extends the modeling problem to more general MDPs and begins to investigate the corruption in transition dynamics. It is worth noting that corruption-robust offline RL has not been widely studied. UWMSG [17] designs a value-based uncertainty-weighting technique, thus using the weight to mitigate the impact of corrupted data. RIQL [18] further extends the data corruptions to all four elements in the offline dataset, including states, actions, rewards, and next states (dynamics). It then introduces quantile estimators with an ensemble of action-value functions and employs a Huber regression based on IQL [26], alleviating the performance degradation caused by corrupted data. Bayesian RL. Bayesian RL integrates the Bayesian inference with RL to create a framework for decision-making under uncertainty [28]. It is important to highlight that Bayesian RL is divided into two categories for different uncertainties: the parameter uncertainty in the learning of models [66, 67] and the inherent uncertainty from the data/environment in the distribution over returns [68, 69]. In this paper, we focus on capturing the latter. For the latter uncertainty, in model-based Bayesian RL, many approaches [68, 70, 71] explicitly model the transition dynamics and using Bayesian inference to update the model. It is useful when dealing with complex environments for sample efficiency. In model-free Bayesian RL, value-based methods [69, 72] use the reward information to construct the posterior distribution of the action-value function. Besides, policy gradient methods [73, 74] use information of the return to construct the posterior distribution of the policy. They directly apply Bayesian inference to the value function or policy without explicitly modeling transition dynamics. Offline Bayesian RL. offline Bayesian RL integrates Bayesian inference with offline RL to tackle the challenges of learning robust policies from static datasets without further interactions with the environment. Many approaches [75–77] use Bayesian inference to model the transition dynamics or guide action selection for adaptive policy updates, thereby avoiding overly conservative estimates in the offline setting. Furthermore, recent work [78] applies variational Bayesian inference to learn the model of transition dynamics, mitigating the distribution shift in offline RL.
Offline Bayesian RL. offline Bayesian RL integrates Bayesian inference with offline RL to tackle the challenges of learning robust policies from static datasets without further interactions with the environment. Many approaches [75–77] use Bayesian inference to model the transition dynamics or guide action selection for adaptive policy updates, thereby avoiding overly conservative estimates in the offline setting. Furthermore, recent work [78] applies variational Bayesian inference to learn the model of transition dynamics, mitigating the distribution shift in offline RL.
# 6 Conclusion
In this paper, we investigate and demonstrate the robustness and effectiveness of introducing Bayesian inference into offline RL to address the challenges posed by data corruptions. By leveraging Bayesian techniques, our proposed approach TRACER captures the uncertainty caused by diverse corrupted data. Moreover, the use of entropy-based uncertainty measure in TRACER can distinguish corrupted data from clean data. Thus, TRACER can regulate the loss associated with corrupted data to reduce its influence, improving performance in clean environments. Our extensive experiments demonstrate the potential of Bayesian methods in developing reliable decision-making. Regarding the limitations of TRACER, although it achieves significant performance improvement under diverse data corruptions, future work could explore more complex and realistic data corruption scenarios and related challenges, such as the noise in the preference data for RLHF and adversarial attacks on safety-critical driving decisions. Moreover, we look forward to the continued development and optimization of uncertainty-based corrupted-robust offline RL, which could further enhance the effectiveness of TRACER and similar approaches for increasingly complex real-world scenarios.
We would like to thank all the anonymous reviewers for their insightful comments. This work was supported in part by National Key R&D Program of China under contract 2022ZD0119801, National Nature Science Foundations of China grants U23A20388 and 62021001, and DiDi GAIA Collaborative Research Funds.
# References
[1] Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2052–2062. PMLR, 2019. [2] Rafael Figueiredo Prudencio, Marcos R. O. A. Máximo, and Esther Luna Colombini. A survey on offline reinforcement learning: Taxonomy, review, and open problems. CoRR, abs/2203.01387, 2022. [3] Shengpu Tang and Jenna Wiens. Model selection for offline reinforcement learning: Practical considerations for healthcare settings. In Ken Jung, Serena Yeung, Mark P. Sendak, Michael W. Sjoding, and Rajesh Ranganath, editors, Proceedings of the Machine Learning for Healthcare Conference, volume 149 of Proceedings of Machine Learning Research, pages 2–35. PMLR, 2021. [4] Christopher Diehl, Timo Sievernich, Martin Krüger, Frank Hoffmann, and Torsten Bertram. Uncertaintyaware model-based offline reinforcement learning for automated driving. IEEE Robotics Autom. Lett., 8(2):1167–1174, 2023. [5] Tony Z. Zhao, Jianlan Luo, Oleg Sushkov, Rugile Pevceviciute, Nicolas Heess, Jon Scholz, Stefan Schaal, and Sergey Levine. Offline meta-reinforcement learning for industrial insertion. In 2022 International Conference on Robotics and Automation, pages 6386–6393. IEEE, 2022. [6] Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing off-policy q-learning via bootstrapping error reduction. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32, pages 11761–11771, 2019. [7] Jiafei Lyu, Xiaoteng Ma, Xiu Li, and Zongqing Lu. Mildly conservative q-learning for offline reinforcement learning. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35, 2022. [8] Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Y. Zou, Sergey Levine, Chelsea Finn, and Tengyu Ma. MOPO: model-based offline policy optimization. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33, 2020. [9] Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. Morel: Model-based offline reinforcement learning. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33, 2020. 10] Porter Jenkins, Hua Wei, J. Stockton Jenkins, and Zhenhui Li. Bayesian model-based offline reinforcement learning for product allocation. In Thirty-Sixth AAAI Conference on Artificial Intelligence, pages 12531– 12537. AAAI Press, 2022. 11] Sen Lin, Jialin Wan, Tengyu Xu, Yingbin Liang, and Junshan Zhang. Model-based offline metareinforcement learning with regularization. In The Tenth International Conference on Learning Representations. OpenReview.net, 2022. 12] Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh Song. Uncertainty-based offline reinforcement learning with diversified q-ensemble. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34, pages 7436–7447, 2021. 13] Yue Wu, Shuangfei Zhai, Nitish Srivastava, Joshua M. Susskind, Jian Zhang, Ruslan Salakhutdinov, and Hanlin Goh. Uncertainty weighted actor-critic for offline reinforcement learning. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 11319–11328. PMLR, 2021.
[14] Chenjia Bai, Lingxiao Wang, Zhuoran Yang, Zhi-Hong Deng, Animesh Garg, Peng Liu, and Zhaoran Wang. Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning. In The Tenth International Conference on Learning Representations. OpenReview.net, 2022. [15] Filippo Valdettaro and A. Aldo Faisal. Towards offline reinforcement learning with pessimistic value priors. In Fabio Cuzzolin and Maryam Sultana, editors, Epistemic Uncertainty in Artificial Intelligence - First International Workshop, volume 14523 of Lecture Notes in Computer Science, pages 89–100. Springer, 2024. [16] Xuezhou Zhang, Yiding Chen, Xiaojin Zhu, and Wen Sun. Corruption-robust offline reinforcement learning. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera, editors, International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Research, pages 5757–5773. PMLR, 2022. [17] Chenlu Ye, Rui Yang, Quanquan Gu, and Tong Zhang. Corruption-robust offline reinforcement learning with general function approximation. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36, 2023. [18] Rui Yang, Han Zhong, Jiawei Xu, Amy Zhang, Chongjie Zhang, Lei Han, and Tong Zhang. Towards robust offline reinforcement learning under diverse data corruption. In The Eleventh International Conference on Learning Representations, 2023. [19] Kishan Panaganti, Zaiyan Xu, Dileep Kalathil, and Mohammad Ghavamzadeh. Robust reinforcement learning using offline data. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35, 2022. [20] Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, and Lei Han. RORL: robust offline reinforcement learning via conservative smoothing. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35, 2022. [21] Jose H. Blanchet, Miao Lu, Tong Zhang, and Han Zhong. Double pessimism is provably efficient for distributionally robust offline reinforcement learning: Generic algorithm and robust partial coverage. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36, 2023. [22] Yuzhe Ma, Xuezhou Zhang, Wen Sun, and Jerry Zhu. Policy poisoning in batch reinforcement learning and control. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32, pages 14543–14553, 2019. [23] Fan Wu, Linyi Li, Huan Zhang, Bhavya Kailkhura, Krishnaram Kenthapadi, Ding Zhao, and Bo Li. COPA: certifying robust policies for offline reinforcement learning against poisoning attacks. In The Tenth International Conference on Learning Representations. OpenReview.net, 2022. [24] Zhihe Yang and Yunjian Xu. Dmbp: Diffusion model based predictor for robust offline reinforcement learning against state observation perturbations. In The Twelfth International Conference on Learning Representations, 2023. [25] Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. CoRR, abs/1910.00177, 2019. [26] Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit q-learning. In The Tenth International Conference on Learning Representations. OpenReview.net, 2022. [27] Rens van de Schoot, Sarah Depaoli, Ruth King, Bianca Kramer, Kaspar Märtens, Mahlet G Tadesse, Marina Vannucci, Andrew Gelman, Duco Veen, Joukje Willemsen, et al. Bayesian statistics and modelling. Nature Reviews Methods Primers, 1(1):1, 2021. [28] Box George EP and Tiao George C. Bayesian inference in statistical analysis. John Wiley & Sons, 2011. [29] Esther Derman, Daniel J. Mankowitz, Timothy A. Mann, and Shie Mannor. A bayesian approach to robust reinforcement learning. In Amir Globerson and Ricardo Silva, editors, Proceedings of the Thirty-Fifth Conference on Uncertainty in Artificial Intelligence, volume 115 of Proceedings of Machine Learning Research, pages 648–658. AUAI Press, 2019. [30] Matthew Fellows, Anuj Mahajan, Tim G. J. Rudner, and Shimon Whiteson. VIREL: A variational inference framework for reinforcement learning. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32, pages 7120–7134, 2019.
[31] Mattie Fellows, Kristian Hartikainen, and Shimon Whiteson. Bayesian bellman operators. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34, pages 13641–13656, 2021. [32] Brendan O’Donoghue. Variational bayesian reinforcement learning with regret bounds. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34, pages 28208–28221, 2021. [33] Ron Dorfman, Idan Shenfeld, and Aviv Tamar. Offline meta reinforcement learning - identifiability challenges and effective data collection strategies. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34, pages 4607–4618, 2021. [34] Dibya Ghosh, Anurag Ajay, Pulkit Agrawal, and Sergey Levine. Offline RL policies should be trained to be adaptive. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato, editors, International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 7513–7530. PMLR, 2022. [35] Yuhao Wang and Enlu Zhou. Bayesian risk-averse q-learning with streaming observations. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36, 2023. [36] Mohammad Ghavamzadeh, Shie Mannor, Joelle Pineau, and Aviv Tamar. Bayesian reinforcement learning: A survey. Found. Trends Mach. Learn., 8(5-6):359–483, 2015. [37] Carl Doersch. Tutorial on variational autoencoders. CoRR, abs/1606.05908, 2016. [38] Diederik P. Kingma and Max Welling. An introduction to variational autoencoders. Found. Trends Mach. Learn., 12(4):307–392, 2019. [39] Marc G. Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 449–458. PMLR, 2017. [40] Kotz Samuel and Johnson Norman L. Breakthroughs in statistics: methodology and distribution. Springer Science & Business Media, 2012. [41] Jeremias Knoblauch, Jack Jewson, and Theodoros Damoulas. Generalized variational inference. CoRR, abs/1904.02063, 2019. [42] Will Dabney, Mark Rowland, Marc G. Bellemare, and Rémi Munos. Distributional reinforcement learning with quantile regression. In Sheila A. McIlraith and Kilian Q. Weinberger, editors, Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, pages 2892–2901. AAAI Press, 2018. [43] Will Dabney, Georg Ostrovski, David Silver, and Rémi Munos. Implicit quantile networks for distributional reinforcement learning. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 1104–1113. PMLR, 2018. [44] Koenker Roger and Hallock Kevin F. Quantile regression. Journal of economic perspectives, 15(4):143–156, 2001. [45] Müller Alfred. Integral probability metrics and their generating classes of functions. Advances in applied probability, 29(2):429–443, 1997. [46] Abhishek Roy, Krishnakumar Balasubramanian, and Murat A. Erdogdu. On empirical risk minimization with dependent and heavy-tailed data. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34, pages 8913–8926, 2021. [47] Michele Caprio, Souradeep Dutta, Kuk Jin Jang, Vivian Lin, Radoslav Ivanov, Oleg Sokolsky, and Insup Lee. Credal bayesian deep learning. arXiv e-prints, pages arXiv–2302, 2023. [48] Michele Caprio, Yusuf Sale, Eyke Hüllermeier, and Insup Lee. A novel bayes’ theorem for upper probabilities. In Fabio Cuzzolin and Maryam Sultana, editors, Epistemic Uncertainty in Artificial Intelligence - First International Workshop, volume 14523 of Lecture Notes in Computer Science, pages 1–12. Springer, 2024. [49] Souradeep Dutta, Michele Caprio, Vivian Lin, Matthew Cleaveland, Kuk Jin Jang, Ivan Ruchkin, Oleg Sokolsky, and Insup Lee. Distributionally robust statistical verification with imprecise neural networks. CoRR, abs/2308.14815, 2023.
[50] Jim W. Hall. Uncertainty-based sensitivity indices for imprecise probability distributions. Reliab. Eng. Syst. Saf., 91(10-11):1443–1451, 2006. [51] L Lorne Campbell. Exponential entropy as a measure of extent of a distribution. Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete, 5(3):217–225, 1966. [52] Sheri Edwards. Thomas m. cover and joy a. thomas, elements of information theory (2nd ed.), john wiley & sons, inc. (2006). Inf. Process. Manag., 44(1):400–401, 2008. [53] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In 6th International Conference on Learning Representations, 2018. [54] Huan Zhang, Hongge Chen, Chaowei Xiao, Bo Li, Mingyan Liu, Duane S. Boning, and Cho-Jui Hsieh. Robust deep reinforcement learning against adversarial perturbations on state observations. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33, 2020. [55] Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4RL: datasets for deep data-driven reinforcement learning. CoRR, 2020. [56] Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh Song. Uncertainty-based offline reinforcement learning with diversified q-ensemble. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34, 2021. [57] Seyed Kamyar Seyed Ghasemipour, Shixiang Shane Gu, and Ofir Nachum. Why so pessimistic? estimating uncertainties for offline RL through ensembles, and why their independence matters. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35, 2022. [58] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2012, Vilamoura, Algarve, Portugal, October 7-12, 2012, pages 5026–5033. IEEE, 2012. [59] Alexey Dosovitskiy, Germán Ros, Felipe Codevilla, Antonio M. López, and Vladlen Koltun. CARLA: an open urban driving simulator. In 1st Annual Conference on Robot Learning, volume 78 of Proceedings of Machine Learning Research, pages 1–16. PMLR, 2017. [60] Chi Jin, Tiancheng Jin, Haipeng Luo, Suvrit Sra, and Tiancheng Yu. Learning adversarial markov decision processes with bandit feedback and unknown transition. In Proceedings of the 37th International Conference on Machine Learning, 2020. [61] Thodoris Lykouris, Vahab S. Mirrokni, and Renato Paes Leme. Stochastic bandits robust to adversarial corruptions. In Ilias Diakonikolas, David Kempe, and Monika Henzinger, editors, Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing, 2018. [62] Aviv Rosenberg and Yishay Mansour. Online convex optimization in adversarial markov decision processes. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, 2019. [63] Anupam Gupta, Tomer Koren, and Kunal Talwar. Better algorithms for stochastic bandits with adversarial corruptions. In Alina Beygelzimer and Daniel Hsu, editors, Conference on Learning Theory, 2019. [64] Chenlu Ye, Wei Xiong, Quanquan Gu, and Tong Zhang. Corruption-robust algorithms with uncertainty weighting for nonlinear contextual bandits and markov decision processes. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, 2023. [65] Thodoris Lykouris, Max Simchowitz, Alex Slivkins, and Wen Sun. Corruption-robust exploration in episodic reinforcement learning. In Mikhail Belkin and Samory Kpotufe, editors, Conference on Learning Theory, 2021. [66] Matthieu Geist and Olivier Pietquin. Kalman temporal differences. J. Artif. Intell. Res., 39:483–532, 2010. [67] Ian Osband, Daniel Russo, and Benjamin Van Roy. (more) efficient reinforcement learning via posterior sampling. In Christopher J. C. Burges, Léon Bottou, Zoubin Ghahramani, and Kilian Q. Weinberger, editors, Advances in Neural Information Processing Systems 26, pages 3003–3011, 2013.
[68] Marc Peter Deisenroth and Carl Edward Rasmussen. PILCO: A model-based and data-efficient approach to policy search. In Lise Getoor and Tobias Scheffer, editors, Proceedings of the 28th International Conference on Machine Learning, 2011. [69] Richard Dearden, Nir Friedman, and Stuart Russell. Bayesian q-learning. In Jack Mostow and Chuck Rich, editors, Proceedings of the Fifteenth National Conference on Artificial Intelligence and Tenth Innovative Applications of Artificial Intelligence Conference, 1998. [70] Gal Yarin, McAllister Rowan, and Rasmussen Carl Edward. Improving pilco with bayesian neural network dynamics models. In Data-efficient machine learning workshop, ICML, volume 4, page 25, 2016. [71] Zhihai Wang, Jie Wang, Qi Zhou, Bin Li, and Houqiang Li. Sample-efficient reinforcement learning via conservative model-based actor-critic. In Thirty-Sixth AAAI Conference on Artificial Intelligence, pages 8612–8620. AAAI Press, 2022. [72] Yaakov Engel, Shie Mannor, and Ron Meir. Reinforcement learning with gaussian processes. In Luc De Raedt and Stefan Wrobel, editors, Machine Learning, Proceedings of the Twenty-Second International Conference (ICML 2005), 2005. [73] Mohammad Ghavamzadeh and Yaakov Engel. Bayesian policy gradient algorithms. In Bernhard Schölkopf, John C. Platt, and Thomas Hofmann, editors, Advances in Neural Information Processing Systems 19, 2006. [74] Mohammad Ghavamzadeh, Yaakov Engel, and Michal Valko. Bayesian policy gradient and actor-critic algorithms. J. Mach. Learn. Res., 2016. [75] Dibya Ghosh, Anurag Ajay, Pulkit Agrawal, and Sergey Levine. Offline RL policies should be trained to be adaptive. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato, editors, International Conference on Machine Learning, 2022. [76] Hao Hu, Yiqin Yang, Jianing Ye, Ziqing Mai, Yujing Hu, Tangjie Lv, Changjie Fan, Qianchuan Zhao, and Chongjie Zhang. Bayesian offline-to-online reinforcement learning : A realist approach, 2024. [77] Yuhao Wang and Enlu Zhou. Bayesian risk-averse q-learning with streaming observations. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36, 2023. [78] Toru Hishinuma and Kei Senda. Importance-weighted variational inference model estimation for offline bayesian model-based reinforcement learning. IEEE Access, 2023. [79] Sham M. Kakade and John Langford. Approximately optimal approximate reinforcement learning. In Claude Sammut and Achim G. Hoffmann, editors, Machine Learning, Proceedings of the Nineteenth International Conference (ICML 2002), pages 267–274. Morgan Kaufmann, 2002. [80] Vallender SS. Calculation of the wasserstein distance between probability distributions on the line. Theory of Probability & Its Applications, 18:784–786, 1974. [81] Anqi Li, Dipendra Misra, Andrey Kolobov, and Ching-An Cheng. Survival instinct in offline reinforcement learning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, 2023. [82] Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. In Advances in Neural Information Processing Systems 34, pages 15084–15097, 2021. [83] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. PMLR, 2018. [84] Arsenii Kuznetsov, Pavel Shvechikov, Alexander Grishin, and Dmitry Vetrov. Controlling overestimation bias with truncated mixture of continuous distributional quantile critics. In International Conference on Machine Learning, pages 5556–5566. PMLR, 2020. [85] Zhihai Wang, Taoxing Pan, Qi Zhou, and Jie Wang. Efficient exploration in resource-restricted reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 10279–10287, 2023.
[86] Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. Advances in neural information processing systems, 32, 2019. [87] Daniel J Mankowitz, Andrea Michi, Anton Zhernov, Marco Gelmi, Marco Selvi, Cosmin Paduraru, Edouard Leurent, Shariq Iqbal, Jean-Baptiste Lespiau, Alex Ahern, et al. Faster sorting algorithms discovered using deep reinforcement learning. Nature, 618(7964):257–263, 2023. [88] Chris Gamble and Jim Gao. Safety-first ai for autonomous data centre cooling and industrial control. DeepMind, August, 17, 2018. [89] Chao Yu, Jiming Liu, Shamim Nemati, and Guosheng Yin. Reinforcement learning in healthcare: A survey. ACM Computing Surveys (CSUR), 55(1):1–36, 2021. [90] Zhihai Wang, Jie Wang, Dongsheng Zuo, Yunjie Ji, Xinli Xia, Yuzhe Ma, Jianye Hao, Mingxuan Yuan, Yongdong Zhang, and Feng Wu. A hierarchical adaptive multi-task reinforcement learning framework for multiplier circuit design. In Forty-first International Conference on Machine Learning. PMLR, 2024. [91] Zhihai Wang, Xijun Li, Jie Wang, Yufei Kuang, Mingxuan Yuan, Jia Zeng, Yongdong Zhang, and Feng Wu. Learning cut selection for mixed-integer linear programming via hierarchical sequence model. In The Eleventh International Conference on Learning Representations, 2023. [92] Jie Wang, Zhihai Wang, Xijun Li, Yufei Kuang, Zhihao Shi, Fangzhou Zhu, Mingxuan Yuan, Jia Zeng, Yongdong Zhang, and Feng Wu. Learning to cut via hierarchical sequence/set model for efficient mixedinteger programming. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–17, 2024. [93] Zhihai Wang, Zijie Geng, Zhaojie Tu, Jie Wang, Yuxi Qian, Zhexuan Xu, Ziyan Liu, Siyuan Xu, Zhentao Tang, Shixiong Kai, et al. Benchmarking end-to-end performance of ai-based chip placement algorithms. arXiv preprint arXiv:2407.15026, 2024.
# Appendix / supplemental material
# A Proofs
We provide a proof for the upper bound of value distributions and derivations of loss functions. See Table 4 for all notations.
# A.1 Proof for Value Bound
# To prove an upper bound of the value distribution, we introduce an assumption from [18] below. Assumption A.1. [18] Let ζ = �N i=1 (2ζi + log ζ′ i) denote the cumulative corruption level, wher ζi and ζ′ i are defined as
To prove an upper bound of the value distribution, we introduce an assumption from [18] below. Assumption A.1. [18] Let ζ = �N i=1 (2ζi + log ζ′ i) denote the cumulative corruption level, where ζi and ζ′ i are defined as
���T V (si) −˜T V (si) ��� ∞≤ζi, max �πB (a | si) πb (a | si) , πb (a | si) πB (a | si) � ≤ζ′ i, ∀a ∈A.
� � Here ∥· ∥∞means taking supremum over V : S �→[0, rmax/(1 −γ)]
Note that {ζ′ i}i = 1N can quantify the corruption level of states and actions, and {ζi}i = 1N can quantify the corruption level of rewards and next states (transition dynamics). Then, we provide the following assumption. Assumption A.2. There exists an M > 0 such that
max{dπE(s, a) pb(s, a) , d˜πE(s, a) pB(s, a) , dπE(s) d˜πE(s), dπIQL(s) dπE(s) , d˜πIQL(s) d˜πE(s) } ≤M, ∀(s, a) ∈S × A,
where πE(a | s) ∝πb(a | s) · exp (β · [T Q∗−V ∗] (s, a)) is the policy of clean data, πIQL in Equation (17) is the policy following IQL’s supervised policy learning scheme under clean data, ˜πIQL = arg min π Es∼B [KL (˜πE(· | s), π(· | s))]is the policy under corrupted data, and ˜πE(a | s) ∝ πB(a | s) · exp � β · � ˜T Q∗−V ∗� (s, a) � is the policy of corrupted data.
where πE(a | s) ∝πb(a | s) · exp (β · [T Q∗−V ∗] (s, a)) is the policy of clean data, πIQL  Equation (17) is the policy following IQL’s supervised policy learning scheme under clean da ˜πIQL = arg min π Es∼B [KL (˜πE(· | s), π(· | s))]is the policy under corrupted data, and ˜πE(a | s)  � � � �
πIQL = arg max π E(s,a)∼b [exp (β · [T Q∗−V ∗] (s, a)) log π(a | s) = arg min π Es∼b [DKL (πE(· | s), π(· | s))] .
As TRACER directly applies the weighted imitation learning technique from IQL to learn the policy, we can use ˜πIQL as the policy learned by TRACER under data corruptions, akin to RIQL Assumption A.2 requires that each pair, including the policy πE and the clean data b, the policy ˜πE and the corrupted dataset B, πE and ˜πE, πE and πIQL, and ˜πE and ˜πIQL, has good coverage. I is similar to the coverage condition in [18]. Based on Assumptions A.1 and A.2, we can derive the following theorem to show the robustness of our approach using the supervised policy learning scheme and learning the action-value distribution.
Lemma A.3. (Performance Difference) For any ˜π and π, we have
Z ˜π(s) −Zπ(s) = 1 1 −γ E(s,a)∼d˜π,˜π [Dπ(s, a, r) −Zπ(s)] .
(17) (18)
(17)
(18)
Z ˜π(s) = ∞ � t=0 γtE(St,At)∼P,˜π [R(St, At)|S0 = s] = ∞ � t=0 γtE(St,At)∼P,˜π [R(St, At) + Zπ(St) −Zπ(St)|S0 = s] = ∞ � t=0 γtE(St,At,St+1)∼P,˜π [R(St, At) + γZπ(St+1) −Zπ(St)|S0 = s] + Zπ(s) = Zπ(s) + ∞ � t=0 γtE(St,At)∼P,˜π [Dπ(St, At, Rt) −Zπ(St)|S0 = s] = Zπ(s) + 1 1 −γ E(s,a)∼d˜π,˜π [Dπ(s, a, r) −Zπ(s)] .
where qπ is the value distribution, W1(·, ·) is the Wasserstein distance [80] for measuring the distribution difference, ϵ1 = Es∼b [DKL (πE(·|s) ∥πIQL(·|s))], and ϵ2 = Es∼B [DKL (˜πE(·|s) ∥˜πIQL(·|s))].
� ∈ � �  || −|| � The value distribution qπ, also denoted by p(·|s) in Section 3.2 of the main text, is an expactation of action value distribution Ea∼π,r∼ρ[pθ(·|s, a, r)]. Note that as TRACER directly applies the weighted imitation learning technique from IQL to learn the policy (see Section 3.2), we can use ˜πIQL as the policy learned by TRACER under data corruptions, akin to RIQL. Let q˜πIQL and qπIQL be the value distributions of learned policies following IQL’s supervised policy learning scheme under corrupted and clean data, respectively. Z ˜πIQL ∼q˜πIQL and ZπIQL ∼qπIQL. Let pπE be the value distribution of the policy of clean