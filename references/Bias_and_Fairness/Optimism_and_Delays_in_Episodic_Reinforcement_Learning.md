Abstract
There are many algorithms for regret minimisation in episodic reinforcement learning. This problem is well-understood from a theoretical perspective, providing that the sequences of states, actions and rewards associated with each episode are available to the algorithm updating the policy immediately after every interaction with the environment. However, feedback is almost always delayed in practice. In this paper, we study the impact of delayed feedback in episodic reinforcement learning from a theoretical perspective and propose two general-purpose approaches to handling the delays. The first involves updating as soon as new information becomes available, whereas the second waits before using newly observed information to update the policy. For the class of optimistic algorithms and either approach, we show that the regret increases by an additive term involving the number of states, actions, episode length, the expected delay and an algorithm-dependent constant. We empirically investigate the impact of various delay distributions on the regret of optimistic algorithms to validate our theoretical results.
# 1 Introduction
Episodic Reinforcement Learning (RL) considers the problem of an agent learning how to act in an unknow environment to maximise its cumulative reward. The problem formulation is broad enough to capture the na ture of sequential decision-making in many real-world scenarios as it permits complex dependencies betwee actions, rewards and future environmental states. Despite the complexity of the learning problem, there ar many provably efficient algorithms for this problem setting (Jaksch et al., 2010; Filippi et al., 2010; Fru et al., 2020; Azar et al., 2017; Dann et al., 2017).
These existing algorithms focus on the traditional model where one assumes that the algorithm updating the policy observes the sequence of states, actions and rewards at the end of every episode. Unfortunately, this immediate feedback assumption is unrealistic in almost all practical applications. In healthcare, for example, feedback relating to a patient on a particular treatment protocol is not observable to the policy maker until they return to the clinic at a scheduled time point in the future. In e-commerce, one observes a conversion at some unknown time long after a sequence of recommendations. Yet another example is wearable technology. Here, the heavy computation involved in policy updating must occur on a separate machine, forcing the communication of information, which naturally introduces a delay between the agent collecting feedback and the policy updater. In any of these scenarios, the algorithm must continue operating, despite lacking information from its past choices.
The above examples illustrate that delayed feedback is a fundamental challenge in real world reinforcemen learning. Unfortunately, there is little theoretical understanding of the impact of delays in episodic reinforce ment learning in the existing literature. We seek to fill this gap in the literature in this paper.
# 1.1 Related Work
Recently, the topic of delays has attracted a lot of attention in the bandit setting (Agarwal and Duchi, 2011; Dudik et al., 2011; Joulani et al., 2013; Mandel et al., 2015; Vernade et al., 2017; Pike-Burke et al., 2018;
Sarah Filippi
Zhou et al., 2019; Manegueu et al., 2020; Vernade et al., 2020). Here, the feedback is the reward associated with the chosen action in each round. Perhaps the most appealing approach in the multi-armed bandit setting is the queuing technique, which shows that the delays cause an additive penalty involving the expected delay for any base algorithm (Joulani et al., 2013; Mandel et al., 2015). The high-level idea is to build a metaalgorithm that creates a simulated non-delayed environment for any base algorithm designed for immediate feedback, such as UCB1 or KL-UCB. They achieve this by introducing a mechanism that stores the rewards for each action in separate queues and having the base algorithm interact with these rather than the actual environment. Unfortunately, the queuing technique does not readily extend to the delayed feedback setting in RL, as forming the queues would require knowledge of the state and action seen in each step of an episode; this information is delayed in our setting. Joulani et al. (2013) present another meta-algorithm for adversarial multi-armed bandits with delayed rewards that is trivial to adapt to our setting. They propose creating a new instance of the chosen base algorithm whenever there is no feedback, allowing one to bound the regret of each instance separately using standard techniques. More precisely, this involves maintaining τmax + 1 versions of the algorithm, where τ ≤τmax almost surely (Joulani et al., 2013). Thus, the regret of taking this approach is multiplicative, as the maximal delay scales the regret of the base algorithm. Previous work in RL has considered constant delays in observing the current state in Markov Decision Processes (MDPs) (Katsikopoulos and Engelbrecht, 2003). More recent work considers delayed feedback in adversarial MDPs (Lancewicki et al., 2021). They developed an algorithm that computes stochastic policies based on policy optimisation. The regret of this algorithm depends on the sum of the delays, the number of states and the number of steps per episode. For stochastic MDPs, they state a regret bound of the form H3/2S √ AT + H2Sτmax, where H is the number of decisions the learner must make per episode, T = KH is the total number of decisions made across all K episodes, S is the number of states in the environment, A is the number of actions and τk ≤τmax. However, the leading order term in their regret bound is loose for many base algorithms. Their approach also requires a-priori knowledge of the maximal delay to define a phase of explicit exploration; this quantity is often unknown in many practical applications. Further, the base algorithm accrues linear regret in this exploration phase, and the maximal delay can be prohibitively large. We propose two approaches that avoid such prior knowledge and can leverage new information in the early episodes much faster, leading to tighter algorithm-specific theoretical results and better empirical performance. In addition to the improved theoretical results, we relax the assumption that the delay distribution has a finite and known maximum, and instead only require that the delays have a finite expectation that we assume is unknown.
Previous work in RL has considered constant delays in observing the current state in Markov Decision Processes (MDPs) (Katsikopoulos and Engelbrecht, 2003). More recent work considers delayed feedback in adversarial MDPs (Lancewicki et al., 2021). They developed an algorithm that computes stochastic policies based on policy optimisation. The regret of this algorithm depends on the sum of the delays, the number of states and the number of steps per episode. For stochastic MDPs, they state a regret bound of the form H3/2S √ AT + H2Sτmax, where H is the number of decisions the learner must make per episode, T = KH is the total number of decisions made across all K episodes, S is the number of states in the environment, A is the number of actions and τk ≤τmax. However, the leading order term in their regret bound is loose for many base algorithms. Their approach also requires a-priori knowledge of the maximal delay to define a phase of explicit exploration; this quantity is often unknown in many practical applications. Further, the base algorithm accrues linear regret in this exploration phase, and the maximal delay can be prohibitively large. We propose two approaches that avoid such prior knowledge and can leverage new information in the early episodes much faster, leading to tighter algorithm-specific theoretical results and better empirical performance. In addition to the improved theoretical results, we relax the assumption that the delay distribution has a finite and known maximum, and instead only require that the delays have a finite expectation that we assume is unknown.
# 1.2 Contributions
The delayed feedback model studied in this paper poses several theoretical challenges that do not arise in the standard episodic reinforcement learning problem, such as delayed updates and disentangling the delays from the difficulty of the learning problem in the theoretical analysis.
We introduce two novel meta-algorithms to overcome these challenges, namely active and lazy updating. Both take any algorithm as input and transform it into an algorithm that can handle delayed feedback. Henceforth, we refer to the input algorithm as the base algorithm. Using these meta-algorithms, we obtain high probability regret bounds for any optimistic model-based base algorithm in the delayed feedback setting. For both active and lazy updating, the penalty for delayed feedback is an additive term involving the expected delay. Although they obtain similar theoretical results, active and lazy updating employ different algorithmic ideas to separate the delays from the learning problem in the theoretical analysis. The active updating meta-algorithm uses the base algorithm to update the policy as soon as it observes feedback from the environment. Deriving theoretical guarantees for active updating involves tackling the delays head-on, as the delays force the policy to remain constant across numerous episodes. Consequently, the learner can repeatedly make sub-optimal decisions. To quantify the impact of delayed feedback, we introduce several techniques that carefully separate the difficulty of the learning problem from the delays.
The lazy meta-algorithm works slightly differently. Instead of updating immediately, it waits for the amount of feedback to surpass some threshold before updating the policy. One can control this threshold, and therefore the frequency of policy updates, through a hyperparameter α. By waiting to update, lazy creates a simulated non-delayed version of the environment for the input algorithm, allowing us to handle the delays separately from the difficulty of the learning problem.
# 2 Preliminaries
We consider the task of learning to act optimally in an unknown episodic finite-horizon Markov Decision Process, EFH-MDP. An EFH-MDP is formalised as a quintuple: M = (S, A, H, P, R). Here, S is the set of states, A is the set of actions, H is the horizon and gives the number of steps per episode, P = {Ph(·|s, a)}h,s,a is the set of probability distributions over the next state and R = {Rh(s, a)}h,s,a is the set of reward functions. For conciseness, we assume that the reward function is known, deterministic and bounded between zero and one for all state-action-step triples.1 In the episodic reinforcement learning problem, the base algorithm interacts with an MDP in a sequence of episodes: k = 1, 2, . . . , K. We denote the set of episodes by: [K] = {1, 2, . . . K}; a convention that we adopt for sets of integers. In this paper, we consider base algorithms that compute a deterministic policy πk : S × [H] →A at the start of each episode k ∈[K]. It is known that in finite horizon stochastic MDPs, if an optimal policy exists, there is a deterministic optimal policy (Puterman, 1994). Once the base algorithm has computed a policy, an agent uses said policy to sample feedback from the environment by: selecting an action, ak h = πk(sk h, h); receiving a reward, rk h = Rh(sk h, ak h); and transitioning to the next state, sk h+1 ∼Ph(·|sk h, ak h); for each h = 1, · · · , H. The feedback associated with the h-th step of the k-th episode is given by:
D  { } We measure the quality of a policy, π, using the value function, which is the expected return at the end of the episode from the current step, given the current state:
� �� Further, we denote the optimal value function by: V ∗ h (s) = maxπ{V π h (s)}, which gives the maximum expected return over deterministic policies ∀(s, h) ∈S × [H]. When evaluating reinforcement learnin algorithms, it is common to use regret:
� � � � � � Throughout, T = KH denotes the total number of steps. Domingues et al. (2020) show that the lowe bound for the regret in the standard episodic reinforcement learning setting with stage-dependent transition is: Ω(H √ SAT).
# 2.1 Regret Minimisation in Model-Based RL
Many provably efficient algorithms exist for learning in EFH-MDPs when feedback is immediate. In th paper, we focus on the large class of optimistic model-based reinforcement learning algorithms. These alg rithms maintain estimators of the transition probabilities for each (s, a, s′) ∈S × A × S:
(1)
(2)
(3)
There are two main ways of ensuring optimism using model-based algorithms. The first is the modeloptimistic approach, which maintains a confidence set around ˆPkh that contains Ph with high probability (Jaksch et al., 2010; Filippi et al., 2010; Fruit et al., 2020). The second is the value-optimistic approach, which involves directly upper bounding the optimal value function with high probability by adding a bonus to the value function of a policy under the estimated transition density ˆPkh (Dann et al., 2017; Azar et al., 2017). Recent work has shown that all model-based optimistic algorithms have a value-optimistic representation, meaning they all compute a value function of the following form (Neu and Pike-Burke, 2020):
where H′ = H −h and
is the exploration bonus and x ∧y = min{x, y}. Here, B1 and B2 are algorithm-dependent quantities which may depend on S, A, H, log(T) or the empirical variance of the optimistic value function. A suitably chosen exploration bonus ensures the computed value function is optimistic with high probability. For our theoretical results to hold, we require the following assumption on the base algorithm.
Assumption 1. The exploration bonus upper bounds the estimation error with high probability. Mathematically: β+ kh(s, a) ≥⟨( ˆPkh −Ph)(·|s, a), V ∗ h+1(·)⟩for all time-steps, with probability 1 −δ.
All value-optimistic algorithms explicitly use the estimation error to derive suitable bonuses. Further, modeloptimistic algorithms compute bonuses satisfying this assumption implicitly (Neu and Pike-Burke, 2020). Therefore, Assumption 1 allows us to capture a wide range of model-based algorithms. For our analysis, it will be helpful to define an algorithm-dependent variable C, which indicates whether the algorithm’s bonuses satisfy the following inequality:
β+ kh(s, a)) < �� ˆPkh −Ph � (· |s, a) , ˜V πk h+1(·) �
�� � � for all s, a, h, k with probability 1 −δ. Intuitively, C = 1 corresponds to a bonuses that sits somewhere between the estimation error and the difference between the expectation of the optimistic value function under the estimated and true transition function. Since these bonuses must sit within a specific (potentially narrow) interval, they are tighter. However, as we will see later, such bonuses come at the expense of lowerorder terms. UBEV and UCBVI are algorithms where C = 1. Whereas UCRL2, UCRL2B, KL-UCRL and χ2-UCRL are algorithms with C = 0.
# 3 Delayed Feedback
Under stochastic delays, the feedback from an episode does not return to the base algorithm immediately after the interaction. Instead, it returns at some unknown time in the future, k + τk. Here, τk denotes the random delay between the agent playing the kth episode and the base algorithm receiving the corresponding feedback. Throughout this paper, we make the following assumption about the delays:
(4)
(5)
(6)
Assumption 2. The delays are positive, independent and identically distributed random variables with a finite expected value, E[τk] < ∞. The introduction of delays causes the feedback associated with an episode to return at some unknown time in the future, k + τk. As a result, the base algorithm cannot update its policy using feedback from episode k at the start of episode k + 1. Instead, it can only use feedback it has observed, e.g. the feedback associated with episodes i : i + τi < k + 1. When working with delayed feedback in RL, it is helpful to introduce the observed and missing visitation counters:
The introduction of delays causes the feedback associated with an episode to return at some unknown time in the future, k + τk. As a result, the base algorithm cannot update its policy using feedback from episode k at the start of episode k + 1. Instead, it can only use feedback it has observed, e.g. the feedback associated with episodes i : i + τi < k + 1.
These are related to the total visitation counter by
When the feedback is delayed, optimistic algorithms can only compute their bonuses and any required estimators using the observed visitation counter. The corresponding value functions are still optimistic, but they contract to the optimal value function more slowly since Nkh(s, a) ≥N ′ kh(s, a).
# 3.1 Bounding the Missing Episodes
In our analysis, it is helpful to bound the number of missing episodes to get an upper bound on the amount of information missing for each state-action-step. This is done in the following lemma.
Lemma 1. Let Sk = �k−1 i=1 1{i + τi ≥k}, where τ1, τ2, · · · τk−1 ∼fτ(·) are independent and identically distributed random variables with finite expected value. We define
to be the failure event for a single k. Then, P(Fτ) = P(∪∞ k=1F τ k ) ≤δ′.
Proof. Firstly, notice that Sk is a sum of Bernoulli random variables, meaning it is subgaussian. Therefore, one can apply Bernstein’s inequality to obtain the following upper bound that holds with probability 1 −δ′:.
The remainder of the proof follows from noticing that E[Sk] ≤�∞ i=0 P(τ > i), which is the tail probability function of the delay distribution and is equal to the expected delay. Similarly, one can show that Var (Sk) ≤ E[Sk] ≤E[τ]. Substituting these values into the above inequality gives the result. See Appendix A.1 for a full proof.
 ≤ full proof.
which holds for all k ∈[K] with probability 1−δ′. Essentially, ψτ K allows us to bound the amount of missing information in any given episode due to the delays.
(8)
(9)
# 4 Meta-Algorithms For Delayed Feedback
Here, we describe two flexible approaches that allow any base algorithm to handle delayed feedback. A tionally, we prove regret guarantees for both procedures, providing the base algorithm satisfies Assum 1. Regardless of the approach, we utilise the following regret decomposition for optimistic base algori that holds for both the delayed and non-delayed settings.
� � where L = log � S2AHπ2/6δ′� and C indicates whether the bonuses of the algorithm satisfy Equation (6). Proof. See Appendix B.1.
# 4.1 Active Updating
The first meta-algorithm we propose is active updating, which leverages new information by updating as soon as it becomes available. The remainder of this subsection focuses on bounding the regret for model-based optimistic algorithms using active updating, whose pseudo-code is outlined in Algorithm 1.
Algorithm 1 Active Updating
Input. Base(N ′, M ′) (any base algorithm).
Initialise. N ′ = {N ′
h(s, a) = 0}h,s,a and M ′ = {M ′
h(s, a, s′) = 0}h,s,a with
M ′
h(s, a, s′) :=
�
i:i+τi<k
1{(si
h, ai
h, si
h+1) = (s, a, s′)}
Compute policy: π1 = Base(N ′, M ′)
for k = 1 to K do
if ∃i : k −2 < i + τi ≤k −1 then
Update the counters: N ′ and M ′.
Update the policy: πk = Base(N ′, M ′)
else
Reuse previous policy: πk = πk−1
end if
An agent samples an episode using policy πk.
end for
Base(N ′, M ′) is the only input parameter for our algorithm and is the base algorithm. One could view it as a function that takes in the observed number of visits (N ′) and transitions (M ′), among other algorithmdependent hyperparameters, and returns a policy. For the class of optimistic algorithms, the additional hyperparameter is the confidence level, δ. Theorem 1 (Active Updating). Under Assumption 1 and 2, with probability 1 −δ, the regret of any modelbased algorithm under delayed feedback:
Base(N ′, M ′) is the only input parameter for our algorithm and is the base algorithm. One could view it as a function that takes in the observed number of visits (N ′) and transitions (M ′), among other algorithmdependent hyperparameters, and returns a policy. For the class of optimistic algorithms, the additional hyperparameter is the confidence level, δ.
RK ≲B √ HSAT + max � B, B2, CH2S � HSA E [τ]
� � where ≲suppresses numeric constants, poly-log and lower order terms, and B ≥B1 is a upper bound on the leading-order term in the numerator of the exploration bonus that is a function of H and S, and holds for all (k, h) ∈[K] × [H]. Proof. From Lemma 2, it is clear that we must bound the summation of the bonuses to bound the regret. When there are no delays, one can utilise the fact that the visitation count for (s, a, h) at the start of episode
� � where ≲suppresses numeric constants, poly-log and lower order terms, and B ≥B1 is a upper bound on the leading-order term in the numerator of the exploration bonus that is a function of H and S, and holds for all (k, h) ∈[K] × [H].
k + 1 increases by one if the agent observed (s, a, h) in the k-th episode to bound this term. However, this is no longer the case under delayed feedback. Therefore, we introduce the following lemma to bound the delay-dependent visitation counter.
Lemma 3. Let Zp T = �K k=1 �H h=1 1/(N ′ kh(sk h, ak h))p. Then,
� � Zp T ≤ � 4 √ HSAT + 3HSAψτ K if p = 1 2 2HSA log (8T) + HSAψτ K log(16ψτ K) if p = 1
with probability 1 −δ′.
Proof. To prove the claim, we relate the sum involving the observed visitation counters to a sum involving the total visitation counters. To do so, we artificially introduce it into the summation by multiplying by one:
The term in the numerator of the first line is equivalent to the total visitation counter by the equivalence relation given in Equation (9). One can handle the first term using standard results from the immediate feedback setting. The remainder of the proof follows from carefully splitting the second term in the sum on the second line into two disjoint sets. Namely, we split the summation using two indicators: 1{N ′ kh(s, a) ≥ψτ K} and 1{N ′ kh(s, a) < ψτ K}. After a little algebra, we find that we are able to apply results from the immediate feedback setting, which gives the final result. See Appendix A.2 for further details. For many algorithms, B1 depends polynomially on quantities related to the environment, e.g. H and S. For such algorithms, a direct application of Lemma 3 is able to separate the expected delay from the total number of decisions. This is in line with the intuition that the impact of delays are negligible once we have a reasonable model of the environment. However, for algorithms such as for UCRL2B, χ2-UCRL and UCBVI (Fruit et al., 2020; Neu and Pike-Burke, 2020; Azar et al., 2017):
 � Typically, one uses an application of Cauchy-Schwarz to separate the terms involving the variance from th involving the counters, which gives:
Lemma 3 shows that doing so would lead to the delays multiplying the leading order term, as the summation of the variances found underneath the square root is of order HT and multiplies the HSAψτ K that arises from bounding the summation of the observed visitation counter. Setting B = H/2 gives us an upper bound for these types of bonuses and avoids this multiplicative dependence.
� � � � � Directly applying Lemma 3 gives the following upper bound on (10): (10) ≤4B √ HSAT + 3BHSAψτ K + 2 � B2 + 3CH2SL � HSA (log(8T) + ψτ K log (16ψτ K)) Substituting the above upper bound of the terms in Lemma 2 and setting δ = 5δ′ gives the stated result.
(10)
Table 4.3 in Section 4.3 presents regret bounds for various optimistic algorithms using active updating under delayed feedback that fit into our framework. Further discussion of the results can be found in Section 4.3.
# 4.2 Lazy Updating
Instead of updating the policy via the base algorithm as soon as new feedback becomes observable, we now consider waiting. We name the meta-algorithm that employs this technique lazy updating. Algorithm 2 presents the pseudo-code for this meta-algorithm.
Algorithm 2 Lazy Updating
Input. Base(N ′, M ′, · · · ) (any base algorithm) and α (activity parameter).
Initialise epoch: j = 1 and kj = 1.
Initialise counters: N ′
kh(s, a) = M ′
kh(s, a, s′) = 0.
Compute policy: πkj = Base(N ′
kh, M ′
kh)
for k = 1 to K do
Update counters, e.g. Equation (7).
if ∃(s, a, h) : N ′
kh(s, a) ≥(1 + 1/α)N ′
kjh(s, a) then
Update epoch: j = j + 1, kj = k
Update epoch counter: Nkj(s, a) = N ′
kh(s, a)
Update the policy: πkj = Base(N ′
kjh, M ′
kjh)
end if
An agent samples an episode using policy πkj.
end for
Lazy updating works in batches of episodes which we call epochs and denote by j = 1, 2, · · · , J. At the start of the j-th epoch, lazy updating uses the base algorithm to compute a policy using all the available information. The meta-algorithm uses this policy in every episode until the next epoch begins. Therefore, each epoch is just a set of episodes where the lazy updating algorithm uses the same policy. A new epoch begins as soon as there is an (s, a, h) whose observed visitation counter reaches 1 + 1/α times the observed visits at the start of the epoch, where α ∈[1, ∞). Note that α = 1 corresponds to the wellknown doubling trick from Jaksch et al. (2010), and α > 1 represents more frequent updating. Once the observed visitation counter triggers this condition, a new epoch begins, and the meta-algorithm uses the base algorithm to update the policy. Formally, we start epoch j + 1 in episode kj+1, which occurs when:
Lazy updating works in batches of episodes which we call epochs and denote by j = 1, 2, · · · , J. At the start of the j-th epoch, lazy updating uses the base algorithm to compute a policy using all the available information. The meta-algorithm uses this policy in every episode until the next epoch begins. Therefore, each epoch is just a set of episodes where the lazy updating algorithm uses the same policy.
A new epoch begins as soon as there is an (s, a, h) whose observed visitation counter reaches 1 + 1/α times the observed visits at the start of the epoch, where α ∈[1, ∞). Note that α = 1 corresponds to the wellknown doubling trick from Jaksch et al. (2010), and α > 1 represents more frequent updating. Once the observed visitation counter triggers this condition, a new epoch begins, and the meta-algorithm uses the base algorithm to update the policy. Formally, we start epoch j + 1 in episode kj+1, which occurs when:
where
counts the observed number of visits between episodes k and l for l > k. Intuitively, this updating schem forces the number of samples needed for any particular (s, a, h) to trigger an update to increase exponentiall quickly, meaning that the total number of epochs should grow logarithmically in K. Lemma 4 confirms tha this is indeed the case.
counts the observed number of visits between episodes k and l for l > k. Intuitively, this updating scheme forces the number of samples needed for any particular (s, a, h) to trigger an update to increase exponentially quickly, meaning that the total number of epochs should grow logarithmically in K. Lemma 4 confirms that this is indeed the case. Lemma 4. For K ≥SA and α ≥1, Algorithm 2 ensures that the number of epochs has the following upper
Proof. See Appendix A.3 for further details.
(11)
(12)
In contrast to active updating, we will later see that the lazy updating scheme lets us bound the summation of the bonuses independently of the delays. This property means we can avoid upper bounding the numerator of the exploration bonus, B1, and get tighter leading order terms in the regret bound of the chosen base algorithm. In the regret analysis, we will utilise the following extension of the classic result by Jaksch et al. (2010) that illustrates the delay-independence of the bonuses.
In contrast to active updating, we will later see that the lazy updating scheme lets us bound the summation of the bonuses independently of the delays. This property means we can avoid upper bounding the numerator of the exploration bonus, B1, and get tighter leading order terms in the regret bound of the chosen base algorithm. In the regret analysis, we will utilise the following extension of the classic result by Jaksch et al. (2010) that illustrates the delay-independence of the bonuses. Lemma 5. If n0, n1, · · · , nJ are an arbitrary sequence of real-valued numbers satisfying n0 := 0 and 0 ≤nj ≤1 Nj−1 with Nj−1 = max{1, �j−1  ni} for all j ≤J, then
Lemma 5. If n0, n1, · · · , nJ are an arbitrary sequence of real-valued numbers satisfying n0 := 0 an 0 ≤nj ≤1 αNj−1 with Nj−1 = max{1, �j−1 i=0 ni} for all j ≤J, then
Proof. We prove the claim for each case using an inductive argument similar to Jaksch et al. (2010). See Appendix A.3. Using Lemmas 4 and 5, we can derive regret bounds for any optimistic base algorithm that satisfies Assumption 1. Theorem 2. Let K ≥SA and α ≥1. Under Assumption 1 and 2, with probability 1 −δ, the regret of any model-based algorithm under delayed feedback is upper bounded by:
where ˆRK(Base) is an upper bound on the regret of the chosen base algorithm under immediate feedback. Proof. By optimism and utilising the fact that epochs are disjoint sets of episodes, with probability 1 −δ′:
where the final inequality follows from separating the episodes where we update and bounding their contribution to the regret by HJ. Handling the remaining summation in the regret bound requires a little more care, which we do by splitting the remaining sum into two sets; episodes with short and long delays. An episode has a short delay if it is played and observed in the same epoch, 1{k+τk < kj+1}. Otherwise, it has a long delay, 1{k+τk ≥kj+1}. One can show that the regret of episodes with long delays has the following upper bound:
where the final inequality follows from separating the episodes where we update and bounding their cont bution to the regret by HJ.
Handling the remaining summation in the regret bound requires a little more care, which we do by splitting the remaining sum into two sets; episodes with short and long delays. An episode has a short delay if it is played and observed in the same epoch, 1{k+τk < kj+1}. Otherwise, it has a long delay, 1{k+τk ≥kj+1}. One can show that the regret of episodes with long delays has the following upper bound:
Aforementioned, Sk ≤ψτ K for all k ≤K with probability 1 −δ′. Therefore, we can upper bound the regret of episodes with long delays by HJψτ K. All that remains is bounding the regret of episodes with short delays. Applying Lemma 2 to these episodes and re-arranging gives:2
2Here, we have omitted lower order terms for brevity.
where we have omitted the state-action-step triples that caused the update from the summation. By construction, all the state-action-step triples satisfy the conditions of Lemma 5. Applying this result to the summation of the bonuses and combining the contributions of the other terms gives the result. See Appendix A.4 for a full proof of the claim.
# 4.3 Discussion
Table 4.3 presents a selection of algorithms that fit into our framework and their accompanying theoretical guarantees when using the active and lazy updating meta-algorithms to handle delayed feedback. In particular, we see that acting in delayed environments causes an additive increase in regret for almost all combinations of optimistic base algorithms and meta-algorithms considered. This result mirrors what is seen in the bandit setting where algorithms incur an additive regret penalty involving E[τ] (Joulani et al., 2013).
Base Algorithm
C
ˆRK(Base)
Active Updating
Lazy Updating
UBEV (Dann et al., 2017)
1
H3/2√
SAT
ˆRK(Base) + H3S2AE[τ]
(1 + 1
α) ˆRK(Base) + H2SAE[τ]
log(1+ 1
α )
UCBVI-CH (Azar et al., 2017)
1
H3/2√
SAT
ˆRK(Base) + H3S2AE[τ]
(1 + 1
α) ˆRK(Base) + H2SAE[τ]
log(1+ 1
α )
UCRL2 (Jaksch et al., 2010)
0
H3/2S
√
AT
ˆRK(Base) + H2S3/2AE[τ]
(1 + 1
α) ˆRK(Base) + H2SAE[τ]
log(1+ 1
α )
KL-UCRL (Filippi et al., 2010)
0
H3/2S
√
AT
ˆRK(Base) + H2S3/2AE[τ]
(1 + 1
α) ˆRK(Base) + H2SAE[τ]
log(1+ 1
α )
UCRL2B (Fruit et al., 2020)
0
H
√
SΓAT
√
H ˆRK(Base) + H2S2AE[τ]
(1 + 1
α) ˆRK(Base) + H2SAE[τ]
log(1+ 1
α )
χ2-UCRL (Neu and Pike-Burke, 2020)
0
HS
√
AT
√
H ˆRK(Base) + H2S2AE[τ]
(1 + 1
α) ˆRK(Base) + H2SAE[τ]
log(1+ 1
α )
UCBVI-BF (Azar et al., 2017)
1
H
√
SAT
√
H ˆRK(Base) + H3S2AE[τ]
(1 + 1
α) ˆRK(Base) + H2SAE[τ]
log(1+ 1
α )
Table 1: A selection of algorithms that fit into our framework and their regret bounds under delayed feedbac Here, Γ ≤S denotes a uniform upper bound on the number of reachable states.
For active updating and some base algorithms, we found that the additive delay dependence comes at the price of a penalty to the leading order term in the regret bound. Namely, an extra √ H. This extra penalty multiplying the leading order term is a feature of the theoretical analysis. Another important factor influencing the impact of the delays when using active updating is the parameter C. The penalty for delayed feedback is higher when C = 1. The worsened delay dependence for these algorithms is due to the introduction of lower-order terms in the probabilistic analysis under immediate feedback, which allows for tighter bonuses. Unfortunately, these lower-order terms become dependent on the delays in our setting and thus lead to a worse delay dependence.
price of a penalty to the leading order term in the regret bound. Namely, an extra √ H. This extra penalty multiplying the leading order term is a feature of the theoretical analysis. Another important factor influencing the impact of the delays when using active updating is the parameter C. The penalty for delayed feedback is higher when C = 1. The worsened delay dependence for these algorithms is due to the introduction of lower-order terms in the probabilistic analysis under immediate feedback, which allows for tighter bonuses. Unfortunately, these lower-order terms become dependent on the delays in our setting and thus lead to a worse delay dependence. To rectify the undesirable penalty to the leading order terms and the dependence on C, we developed an alternative approach called lazy updating, which achieves the same additive delay dependence for all algorithms that fit into our framework with only a logarithmic penalty to the leading order term in the regret bound of the base algorithm under immediate feedback. This approach works by introducing an additional hyperparameter that controls how frequently the base algorithm updates its policy. We denote this hyperparameter by α and name it the activity parameter. Theorem 2 indicates that there is a trade-off when selecting α. On the one hand, we would like to choose a large value of α to minimise the penalty to the leading order term, which is arises from the slower updating. On the other hand, the penalty introduced by the delays is a strictly increasing function of α, making large values undesirable. As α →∞, lazy updating tends to active updating; at this limiting value, lazy updating will update as soon as it receives new feedback, just like active updating. Thus, the empirical performance of lazy updating should get closer to active updating as α increases. In Section 5, we demonstrate that this is the case and show that it is possible to get most of the benefits of active updating with a relatively modest value of α, which has better worst-case regret bounds in the delayed feedback setting. Comparatively, our work significantly improves the regret bounds for many algorithms in the delayed feedback setting. Lancewicki et al. (2021) presents regret bounds for stochastic MDPs of the form H3/2S √ AT +
H2Sτmax for all optimistic algorithms. Except for UCRL2 and KL-UCRL, the leading order term in their regret bound is loose in either H, S or both. Conversely, the leading order terms in our regret bounds are tight for all algorithms when utilising lazy updating and are only loose by a factor of √ H for a few algorithms when utilising active updating. Furthermore, E[τ] ≪τmax in almost all scenarios. As a result, our regret bounds have a tighter delay dependence. Our algorithms also remove the need for a-priori knowledge of the maximal delay.
HSτmax for all optimistic algorithms. Except for UCRL2 and KL-UCRL, the leading order term in their regret bound is loose in either H, S or both. Conversely, the leading order terms in our regret bounds are tight for all algorithms when utilising lazy updating and are only loose by a factor of √ H for a few algorithms when utilising active updating. Furthermore, E[τ] ≪τmax in almost all scenarios. As a result, our regret bounds have a tighter delay dependence. Our algorithms also remove the need for a-priori knowledge of the maximal delay. The setting of delayed feedback also generalises the case where only the rewards are delayed. Thus, our theoretical results also hold for this setting if we directly apply active or lazy updating. However, one could do better in this case by realising that it is only the delays impacting the rewards, meaning it is only necessary to apply the meta-algorithms to the estimation of the rewards. We expect the additive penalty to be HSAE[τ]. Indeed, the improved delay-dependence is due to the fact that learning the expected reward function is an easier task than learning the transitions. We prove that this is indeed the case for UCRL2 algorithm of Jaksch et al. (2010) in Appendix B.2.
The setting of delayed feedback also generalises the case where only the rewards are delayed. Thus, our theoretical results also hold for this setting if we directly apply active or lazy updating. However, one could do better in this case by realising that it is only the delays impacting the rewards, meaning it is only necessary to apply the meta-algorithms to the estimation of the rewards. We expect the additive penalty to be HSAE[τ]. Indeed, the improved delay-dependence is due to the fact that learning the expected reward function is an easier task than learning the transitions. We prove that this is indeed the case for UCRL2 algorithm of Jaksch et al. (2010) in Appendix B.2.
# 5 Experimental Results
In this section, we investigate the impact of delayed feedback on the regret of active and lazy updating in the chain environment of Osband and Van Roy (2017). Briefly, this environment consists of a sequence of S states arranged side-by-side. The learner starts in the left-most state and has to decide between A = 2 actions, head left or right. Each episode consists of H = S decisions and the only state with a reward is the right-most state. Thus, the optimal policy is to head right at every step. Heading left is always successful. However, heading right is successful with probability 1−1/S. If unsuccessful, the learner moves one state to the left. Notably, any inefficient exploration strategy will take at least 2S episodes to learn the optimal policy (Osband and Van Roy, 2017).
We consider chains with H = S ∈{5, 10, 20, 30} and use UCBVI-BF as the base algorithm in all of our experiments as it has the best regret guarantees under immediate feedback. For our lazy updating approach, we selected several values for the activity hyperparameter, α ∈{1, 10, 100}. In all our experiments, we set the confidence parameter of the base algorithm so that the regret bounds hold with probability 0.95. Additionally, we compare our meta-algorithms to the explicit exploration procedure proposed by Lancewicki et al. (2021). Their procedure requires prior knowledge of the maximum delay, which we provide by generating all the delays before the first episode and taking the maximum. In practice, the maximum delay is often unknown and possibly infinite, making this approach infeasible.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/56c3/56c3e8eb-1cdb-4d80-9564-79340b9e6fee.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Cumulative Regret (S = 30, E[τ] = 100).</div>
Figure 1 displays the results for our experiments in the chain environment with S = 30 and E[τ] = 100. The results for the other chain lengths and expected values are in Appendix C. Empirically, active updating achieves the best performance of all three meta-algorithms. However, our experimental results suggest that it is possible to get near identical performance with lazy updating by setting α to be a large enough constant. Both active and lazy updating offer superior performance to the explicit exploration approach of Lancewicki et al. (2021) in all of our experiments, despite their meta-algorithm having prior knowledge of the delays. In some cases, our meta-algorithms have converged to the optimal policy before the explicit exploration procedure finishes; e.g. see Appendix C.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c5ed/c5edcc0c-0afb-48d0-94f9-98fc9f5dfac9.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Delay Dependence (S = 30).</div>
Next, we turn to considering the impact of different delay distributions on the regret of our meta-algorithms. Empirically, Figure 2 shows that the regret penalty of delays at the end of the final episode is linear in the expected delay for active updating and lazy updating, as our theory predicts. For lazy updating, the gradient of this linear relationship decreases with α, which is to be expected based on the log(1 + 1/α) term in the denominator of the delay-dependent terms in our regret bounds. Interestingly, lazy updating with α = 1 is the most robust to the delay distribution. We believe that this is due to forcing the base algorithm to wait for long periods of time between updates. Intuitively, if the epochs are long enough, most information within an epoch will be received before an update, leading to little loss of information. Investigating this further is an interesting avenue for future work.
# 6 Conclusion
In this paper, we provide two generic meta-algorithms that can extend any episodic reinforcement learning base algorithm to the setting of delayed feedback. Under mild assumptions on the algorithm and the delays, we show that both maintain the sub-linear theoretical guarantees of the chosen base algorithm and provide good empirical performance, regardless of the delay distribution. These first positive results for stochastically delayed feedback in episodic reinforcement learning prove that the penalty for delays is an additive term involving the expected delay that is independent of the number of episodes. This additive penalty matches what is seen in the multi-armed bandit setting, despite the additional complexities of the reinforcement learning problem.
Our framework is broad enough to cover the theoretically successful class of optimistic model-based algorithms, and many existing algorithms fit into our framework. However, we believe that both updating procedures could be used for a wider class of base algorithms. For example, model-free optimistic algorithms and posterior sampling (Jin et al., 2018; Osband and Van Roy, 2017). Extending our analyses to cover these algorithms is left to future work.
# A Missing Proofs
# A.1 Bounding the Missing Episodes
An important aspect in our proofs is to bound the amount of missing information. Since we see only one stateaction pair per step of an episode, an upper bound on the missing visitation counter is simply the number of missing episodes. Lemma 1 bounds the number of missing episodes with high probability and only requires the delays have a finite expected value. Lemma 1. Let Sk = �k−1 i=1 1{i + τi ≥k}, where τ1, τ2, · · · τk−1 ∼fτ(·) are independent and identically distributed random variables with finite expected value. We define
By Bernstein’s inequality, we have that:
Rearranging the above reveals that:
By Boole’s inequality, we have that:
as required.
# A.2 Missing Proofs for Active Updating
Lemma 2 (the regret decomposition) and Equation (5) (the form of the exploration bonuses) reveal that the summation of the counters is an important quantity in determining the regret of an optimistic algorithm. Whenever τk = 0 for all k ≤K, e.g. immediate feedback, we can use standard results that utilise the fact the counters increase by one between successive plays of a state-action pair at a given step. Lemma 6. Let Zp n = �N n=0 1/(1 ∨n)p. Then, Zp n has the following upper bound:
for p = 1/2 and p = 1.
for p = 1/2 and p = 1. Proof. Removing the first two terms from the summation and upper bounding the remaining terms by integral gives:
Proof. Removing the first two terms from the summation and upper bounding the remaining terms by an integral gives:
Zp n = 2 + N � n=2 1 np ≤2 + �N 1 1 np dn ≤2 + � 2 √ N −2 if p ∈1 2 log (N) if p = 1 ≤ � 2 √ N if p ∈1 2 log (8N) if p = 1
as required.
When τk is random, the observed visitation counter need not increase by one between successive plays of the same state-action-step. Instead, the counter only increases by one (or more in some cases) after a random number of episodes. In the worst-case scenario, the counter will remain constant between playing and observing the feedback associated with a specific state-action-step. Thus, the standard techniques no longer apply, and we must find another way to bound the summation of counters than can remain unchanged for numerous episodes due to the delays. We do this by relating the summation involving the observed visitation counter to one involving the total visitation counter, thereby splitting the terms affected by the delays from those that are not.
Lemma 3. Let Zp T = �K k=1 �H h=1 1/(N ′ kh(sk h, ak h))p. Then,
� � Zp T ≤ � 4 √ HSAT + 3HSAψτ K if p = 1 2 2HSA log (8T) + HSAψτ K log(16ψτ K) if p = 1
with probability 1 −δ′.
Proof. Unless otherwise stated, we let: Nkh(s, a) = 1 ∨Nkh(s, a) and N ′ kh(s, a) = 1 ∨N ′ kh(s, a) for notational convenience. First, we use the relationships between the observed, missing and total visitation counters to split the summation into two parts. To do so, in a similar manner to Lancewicki et al. (2021), we start by artificially introducing the total visitation counter:
� � � � � � From Equation (9), Nkh(s, a) = N ′ kh(s, a) + N ′′ kh(s, a), for any (s, a, h) ∈S × A × [H]. Consequently
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7fca/7fca5c7b-b08d-4384-b4c9-4a13f5d6a169.png" style="width: 50%;"></div>
<div style="text-align: center;">16</div>
since (1 + x)p ≤1 + xp for p = 1/2 and p = 1 and any x > 0. Term (i) is the summation of the total visitation counter. Thus, Lemma 6 applies. Bounding (ii) requires more care, as it involves the observed and missing visitation counters. Recall that the algorithm plays one state-action pair at each step in every episode. Thus, the missing visitation counter is upper bounded by the number of missing episodes: N ′′ kh(s, a) ≤Sk. Lemma 1 bounds the number of missing episodes: with probability 1 −δ′, Sk ≤ψτ K across all k ∈Z+. Splitting (ii) using the observed visitation counts and the upper bound on Sk gives:
since (1 + x)p ≤1 + xp for p = 1/2 and p = 1 and any x > 0. Term (i) is the summation of the total visitation counter. Thus, Lemma 6 applies.
Bounding (ii) requires more care, as it involves the observed and missing visitation counters. Recall that the algorithm plays one state-action pair at each step in every episode. Thus, the missing visitation counter is upper bounded by the number of missing episodes: N ′′ kh(s, a) ≤Sk. Lemma 1 bounds the number of missing episodes: with probability 1 −δ′, Sk ≤ψτ K across all k ∈Z+. Splitting (ii) using the observed visitation counts and the upper bound on Sk gives:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/96eb/96ebc55e-3e5a-4308-b56c-308912944a33.png" style="width: 50%;"></div>
<div style="text-align: center;">� �� � � The last inequality follows since for the first sum, N ′ kh(s, a) ≥ψτ K.</div>
Clearly, (ii.a) ≤(i), as it is a summation over a subset of all the episodes. Using (9), it is possible to rewrite the indicator in the remaining term as: 1{Nkh(s, a) −N ′′ kh(s, a) ≤ψτ K}, for any (s, a, h) ∈S × A × [H]. Further, N ′′ kh(s, a) ≤ψτ K and N ′ kh(s, a) ≥1. Therefore,
Lemma 6 gives an upper bound of �N n=0 1/(1 ∨n)p. Summing this upper bound over all state-action-step triples gives:
Therefore:
Zp T ≤2A + B.2 �
≤ ≤ � 4 √ HSAT + 3HSAψτ K if p = 1 2 HSA (2 log (8T) + ψτ K log (16ψτ K)) if p = 1
as required.
# A.3 Missing Proofs for Lazy Updating
When using active updating, we prove that the bound on the counts depends on the delay. However, we can mitigate this delay-dependence by taking a slower approach to updating, providing that the number of epochs is bounded and the counts between epochs satisfy certain constraints outlined in Section 4.2. Lemma 4. For K ≥SA and α ≥1, Algorithm 2 ensures that the number of epochs has the following upper bound:
Proof. In this proof, we extend arguments from the standard doubling trick of Jaksch et al. (2010) so that the learner can update more frequently. Firstly, we recall the definition of the observed visitation counter:4
and the updating rule for j ≥1:
kj+1 = arg min k>kj � ∃s, a, h : N ′ k(s, a, h) ≥ � 1 + 1 α � N ′ kj(s, a, h) �
Now, we define a counter that counts the observed number of visits between two episode
Direct computation allows us to relate the observed visitation counter at the start of the (j + 1)-th epoch to the sum of the observed visitation counts within each of the previous epochs:
where the second equality follows from the fact that an epoch is a disjoint set of episodes and the final equali follows from the definition of the between episodes visitation counter. From the above, it is easy to see tha
Thus, we can re-write the updating rule using the within episode counter as
providing that we have seen the state-action-step at least once.5 Therefore, at the end of each epoch there is a state-action-step with nkj+1 kj (s, a, h) ≥N ′ kj(s, a, h)/α.
Suppose N ′ (K+1)h(s, a) > 0 for a fixed (s, a, h) ∈S × A × [H]. Define J(s, a, h) as the number of epochs with nkj+1 kj (s, a, h) ≥N ′ kjh(s, a)/α. Or, equivalently, it is the number of epochs with Nkj+1 (s, a, h) ≥ (1 + 1/α)N ′ kj(s, a, h). Then,
≥1 +
The first inequality follows from focusing only on the epochs where we update due to (s, a, h), where the +1 accounts for the first update due to the observing the given state-action-step triple. The second inequality follows from the condition in the subscript of the summation, e.g. we are updating due to (s, a, h). The final inequality follows from the definition of how we trigger updates and because we update J(s, a, h) times due to (s, a, h). Since α ∈[1, ∞), Lemma 7 applies. Rearranging terms reveals that:
Therefore, for N ′ K+1(s, a, h) > 0:
N ′ K+1 (s, a, h) ≥1 −1 α � 1 + 1 α � + 1 α � 1 + 1 α �J(s,a,h)+1 > 1 α � 1 + 1 α �J(s,a,h)+1 −1 α � 1 + 1 α �
If N ′ K+1(s, a, h) = 0 it follows we never update due to this state-action-step triple, which means t J(s, a, h) = 0 too. Plugging this into the above expression reveals that:
Thus, for all possible values of the observed visitation counter, we have that:
Using the above inequality, we have that
N ′ kj(s, a, h)
where the final line follows from the fact that J ≤HSA + � s,a,h J(s, a, h) because we may or may not visit every state-action-step. Rearranging this gives:
Taking logs of both sides and rearranging one last time gives:
as required.
Lemma 5. If n0, n1, · · · , nJ are an arbitrary sequence of real-valued numbers satisfying n0 := 0 a 0 ≤nj ≤1 αNj−1 with Nj−1 = max{1, �j−1 i=0 ni} for all j ≤J, then
Proof. We prove the claim via induction in a similar manner to Jaksch et al. (2010). First, consider the case where p = 1/2. Suppose
Then,
� because NJ ≥1. The above is our base case and covers us as long as �J−1 j=1 nj ≤1 e.g., when J = 1 due to n0 := 1. Now, we assume the above holds for �J−1 j=1 nj > 1:
Finally, we prove the claim holds for J:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2d22/2d22d4bb-d7a8-43f3-ad95-16f8b7af090c.png" style="width: 50%;"></div>
� where the final inequality follows from the fact that �J−1 j=1 nj > 1 =⇒NJ = nJ + NJ−1. All that remain is selecting c. Using the quadratic formula to find the roots of c2 −2c −1 = 0, one can deduce that selecting
satisfies c ≥1 + 1/α and
giving the required result. All that remains is to prove the claim for p = 1. Similarly to before, suppose:
because NJ ≥1. The above is our base case and covers us as long as �J−1 j=1 nj ≤1 e.g., when J = 1 due to n0 := 1. Now, we assume the above holds for �J−1 j=1 nj > 1:
(As nJ ∈[0, NJ−1/α) (As α ≥1)
(Pick c : c2 ≥1 + 2c)
here the final inequality follows from the fact that nj/Nj−1 ∈[0, 1] for all j ≤J.
where the final inequality follows from the fact that nj/Nj−1 ∈[0, 1] for all j ≤J. Lemma 7. Let α ∈[1, ∞). Then
Lemma 7. Let α ∈[1, ∞). Then
oof. Trivially, the statement is true for n = 0, because (1 + 1/α)0 = 1 and (1 + 1/α)1 −1/α = 1. Thus,  proceed by induction. Suppose
roof. Trivially, the statement is true for n = 0, because (1 + 1/α)0 = 1 and (1 + 1/α)1 −1/α = 1. Thus, e proceed by induction. Suppose
Proof. Trivially, the statement is true for n = 0, because (1 + 1/α)0 = 1 and (1 + 1/α)1 −1/α = 1. Thu we proceed by induction. Suppose
for some n. Then
<div style="text-align: center;">Thus, the claim holds for n + 1, which proves the lemma for all n ≥0.</div>
� s,a,h J � j=1 nkj+1 kj+1,h(s, a) N ′ kjh (s, a) ≤ � 1 + 1 α � HSA + � 1 + 1 α � HSA log �K SA � ≤2 � 1 + 1 α � HSA �K SA �
where the final inequality holds for K/SA ≥exp(1).
(Induction Hypothesis)

(Since 2 ≥1 + 1/α)
Proof. To prove the result, we extend the summation to include the state-action-step triples in episode kj tha did not trigger the update rule:
for K/SA ≥exp(1), as required.
# A.4 Proof of Regret Bound for Lazy Updating
Theorem 2. Let K ≥SA and α ≥1. Under Assumption 1 and 2, with probability 1 −δ, the regret of a model-based algorithm under delayed feedback is upper bounded by:
where ˆRK(Base) is an upper bound on the regret of the chosen base algorithm under immediate feedback. Proof. Let ˜∆k h(s) = ˜V πk h (s) −V πk h (s) denote the difference between the optimistic and actual value of policy πk from state s and step h. By definition, the regret of any episodic reinforcement learning algorithm is given by:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5c21/5c210f9f-9f0d-4622-89e4-a6b617bea394.png" style="width: 50%;"></div>
(Lemma 5)

(i) episodes where we perform a policy update, (ii) episodes played in the j-th epoch but observed in epoch j′ > j, (iii) episodes played in the j-th epoch and observed in the j-th epoch.
First, we focus on the episodes where we perform a policy update, e.g. (i). Recall that Lemma 4 tells us the total number of updates is logarithmic in the number of episodes. Further, the rewards are bounded between zero and one, meaning the regret of any episode is at most H. Combining these two results gives a trivial bound on regret of this term: (i) ≤HJ.
Finally, we handle the episodes that are played and observed in the same epoch e.g., term (iii). Lemma 2 allows us to make a start on bounding this term:
(iii)
Thus, bounding (iii) now amounts to finding an upper bounds for (iii.a) and (iii.b). Since kj does not feature in either summation, we know that
(iii.b)
for all (s, a, h) ∈S × A × [H] and k′ ≥kj + 1. By introducing a summation over all the states-actions and steps, we can easily bound (iii.a) via Lemma 8:
Bounding (iii.b) requires some care due to the various forms of B1 e.g., those that remain constant and those that utilise variance reduction techniques. By Lemma 5, it is clear that the summation of the visitation counters no longer depends on the delay. Therefore, we begin by an application of Cauchy-Schwarz (CS) to separate the numerator of the exploration bonus from the summation of the visitation counters:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/770e/770e1b3b-def6-4430-9bbb-e223fb526587.png" style="width: 50%;"></div>
(Eq. (12))
(By Lemma 8)
The penultimate line in the above is simply the sum of the bonuses for the chosen base algorithm under immediate feedback scaled by a logarithmic factor, which is introduced by the slower updating. For B1 ≈B e.g., the upper bound only involves inflating terms inside logarithms, one can upper bound the summation under the square-root by TB2, which is tight up to logarithmic factors. When B1 involves some form of empirical variance term, one can use the techniques outlined by Neu and Pike-Burke (2020); Azar et al. (2017); Fruit et al. (2020) to bound the summation under the square-root by ≈HT; once again this too is tight up to logarithmic factors. More simply, the epochs form a simulated non-delayed version of the environment for the base algorithm. Therefore, (iii.b) can be replaced with the upper bound of the regret in the non-delayed environment multiplied by the extra logarithmic factors that arise from the slower updating, because the summation of the bonuses are the leading term in the regret bound. Bringing everything together gives:
# B Additional Theoretical Results
Here, we present a brief overview of the results that unify model-optimistic and value-optimistic model based episodic reinforcement learning algorithms (Neu and Pike-Burke, 2020). The class of model-optimistic algorithms explicitly define the following failure event for some divergence D( ˆPkh(·|s, a), Ph(·|s, a):
� � � � which holds across all episodes with probability δ′. Indeed, D must satisfy some conditions. Namely, D must be jointly convex in its arguments so that Pkh (defined below) is convex, and it must be positive homogeneous.6 Outside the failure event, with probability 1 −δ′, the divergence between the empirical and actual transition density of the hth step at the start of the kth episode is therefore, at most: D( ˆPkh(·|s, a), Ph(·|s, a)) ≤ϵp kh(s, a). Using ϵp kh(s, a) as the maximum divergence allows for the construction of the following plausible set:
(Lemma 8)
β∗ kh (s, a) = max ˜ Ph(·|s,a)∈∆ �� ˜V , ˜Ph (·|s, a) −ˆPh (·|s, a) �� β− kh (s, a) = max ˜ Ph(·|s,a)∈∆ �� −˜V , ˜Ph (·|s, a) −ˆPh (·|s, a) �� βkh (s, a) ≥max � β∗ kh (s, a) , β− kh (s, a) �
·|∈ �� βkh (s, a) ≥max � β∗ kh (s, a) , β− kh (s, a) �
� � by introducing a Lagrange multiplier. For a derivation of the bonuses associated with each divergence, we refer the reader to Appendix A.5 of Neu and Pike-Burke (2020).
# B.1 Missing Proofs for the Regret Decomposition
In this subsection, we utilise the fact that all model-based algorithms compute an optimistic value function of the form (4) to derive an adaptable regret decomposition. The decomposition is adaptable in the sense it allows for tighter delay-dependence when the bonuses satisfy a symmetry-like property.
Throughout, we assume that the model-based algorithm is optimistic with high probability. That is, ˜V πk h (s) V ∗ h (s) ≥V πk h (s) with high probability at least 1 −δ′. Further, C is defined as the event where:
� � where L = log � S2AHπ2/6δ′� and C indicates whether the bonuses of the algorithm satisfy Equatio Proof. By definition, the regret of any episodic reinforcement learning algorithm is given by:
where
� � which we do by induction. Recall that: ˜V πk H+1 = V ∗ H+1 = V πk H+1 = ⃗0. Therefore, the statement holds when j = H, because: ˜∆k H+1 = 0. Now assume the statement holds for h = j + 1. Then,
� Therefore, we are now able to upper bound the regret as follows:
� � � � � � � � Similarly, |¯ζk h+1| ≤2 and Esk h+1∼Ph(· | sk h,ak h) �¯ζk h+1 | Fkh ∪{sk h, ak h} , sk h+1 ∈Gkh � = 0. Therefore, ζk h+1 and ¯ζk h+1 are martingale differences, which are easily bounded using Azuma-Hoeffding:
Therefore, with probability 1 −4δ′:
as required.
Lemma 9. Let C be an algorithm dependent-constant indicating whether it is model-optimistic or valu optimistic. Under Assumption 1, the regret of any optimistic model-based algorithm from the h-th step of  k-th episode upper bounded by:
� �   where L = log(S2AHπ2/6δ′) and
 − Proof. By Proposition 2 of Neu and Pike-Burke (2020) and by definition of the value-optimistic algorithm we have that: ˜∆k h(sk h) = ˜V πk h � sk h � −V πk h � sk h � = β+ kh � sk h, ak h � + �ˆPkh � · | sk h, ak h �˜V πk h+1 � − � Ph � · | sk h, ak h � V πk h+1 � = β+ kh � sk h, ak h � + �ˆPkh � · | sk h, ak h � −Ph � · | sk h, ak h � , ˜V πk h+1 � + � Ph � · | sk h, ak h � , ˜V πk h+1 −V πk h+1 �
(with probability at least 1 −δ′)
(with probability at least 1 −δ′)

= β+ kh � sk h, ak h � + �ˆPkh � · | sk h, ak h � −Ph � · | sk h, ak h � , ˜V πk h+1 � + � Ph � · | sk h, ak h � , ˜∆k h+1 � ≤βkh � sk h, ak h � + �ˆPkh � · | sk h, ak h � −Ph � · | sk h, ak h � , ˜V πk h+1 � + � Ph � · | sk h, ak h � , ˜∆k h+1 � = ˜∆k h+1 � sk h+1 � + βkh � sk h, ak h � + �ˆPkh � · | sk h, ak h � −Ph � · | sk h, ak h � , ˜V πk h+1 � + � Ph � · | sk h, ak h � , ˜∆k h+1 � −˜∆k h+1 � sk h+1 �
� � � � � � where the inequality follows from the fact that βkh(s, a)+ ≤βkh(s, a). For model-optimistic algorithms from the definition of the bonuses, we have that:
� � � � � � � � � � However, this term cannot be bound as easily for the value-optimistic algorithms. But, Assumption 1 allows us to show that, with probability 1 −δ′:
as required.
Lemma 10. Let γkh(sk h, ak h) := ⟨ˆPkh � · | sk h, ak h � −Ph � · | sk h, ak h � , ˜∆k h+1⟩. Then, with probability at leas 1 −δ′:

� � where L = log(S2AHπ2/6δ′) and
Gkh := {s′ : Ph � s′ | sk h, ak h � N ′ kh � sk h, ak h � ≥4H2L}
 � � � � � �  By definition, Ph(s′|s, a) < 4H2L/N ′ kh(s, a) whenever s′ ̸∈Gkh, which follows simply from rearrang terms in the definition of Gkh. Therefore,
� Now, we focus on the s′ ∈Gkh.
completing the proof.
# B.2 Missing Theoretical Results for Delayed Rewards
In this section, we describe how to use active or lazy updating in the setting where only the rewards return in delay. We assume the rewards are stochastic and their expected values are unknown.
In the setting of delayed rewards, the agent returns the state-action pairs {sk h, ak h}H h=1 at the end of episode k, immediately. However, the rewards {rk h}H h=1 return with a random delay τk. Since it is only the rewards that return in delay, we can estimate the transitions at the start of each episode, as usual. Thus, we apply active or lazy updating to the estimation of the expected reward function only.
ˆrkh (s, a) = 1 N ′ kh (s, a) k−1 � i=1 ri h1{si h = s, ai h = a, i + τi < k}
For lazy updating, this amounts to waiting until the observed number of rewards for a state-action-step triple have doubled before starting a new epoch. When estimating the expected reward function for jth epoch, the base algorithm will use all the available rewards:
Using Hoeffding’s inequality, one can construct confidence sets around the above estimators and derive another estimator that is optimistic, with high probability. We derive the width of the confidence set in the proof below. Theorem 3. Let RP K denote the regret of UCRL2 from estimating the transition densities under immediate feedback. Then, with probability 1 −δ, the regret of UCRL2 under delayed reward is:
for active updating.
Proof. First, since the rewards are stochastic and their expected values are unknown, we must derive an estimator. Naturally, we use only the observed information to compute the expected value, as it is an unbiased estimator:
� Now, assume that the rewards are bounded in [0, 1]. Using Hoeffding’s inequality, we can define an addition failure event to account for the fact that we are estimating the expected reward function:
which upper bounds the true expected reward function with probability 1 −δ′ across immediate feedback setting, the failure event for the transition densities is:
where
<div style="text-align: center;">˜Pkh (·|s, a) ∈{Q ∈∆: ∥Q −Ph (·|s, a)∥≤ϵp kh (s, a)} By optimism, and due to UCRL2 having C = 1: with probability 1 −2δ′:</div>
The penultimate inequality follows from Lemma 6. Further,
� is the regret of the base algorithm (UCRL2) in an immediate feedback environment with know reward fun tions. Now, to prove the statements of the corollary, we must bound the summation of the estimation err for the rewards. Doing so is just a matter of applying Lemma 3:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/345f/345f89e8-21ab-476f-bfaf-7a35779013b8.png" style="width: 50%;"></div>
� � Substituting the above into Equation (14) and omitting poly-logarithmic factors gives the stated result.
# C Additional Experimental Results
Here, we present additional experimental results for the chain environments with H = S ∈{5, 10, 20} and E[τ] ∈{100, 300, 500}. In all combinations of chain length and expected delay, our updating procedures give
(14)
better empirical performance, especially for the delay distributions with higher variances. For all expecte delays, active updating gives the best performance. However, our experiments indicate that lazy updatin with α = 10 or 100 is comparable, as one would expect based on the intuition that it is an approximation t active updating that converges in the limit as α →∞.
# C.1 Chain Environment with H = S = 5
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c5c7/c5c77419-80ea-4edf-b719-f9f7b74e669c.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Cumulative Regret (S = 5, E[τ] = 100).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ca60/ca6079ed-3480-4bdc-96d6-9a25ee68cc37.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b4ce/b4ce471c-227f-400f-a3d1-d23975fc1121.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Cumulative Regret (S = 5, E[τ] = 300).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c202/c2020cb8-d7aa-4e03-ab70-196aa5cd6915.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Cumulative Regret (S = 5, E[τ] = 500).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/99ef/99ef5e74-3b5e-4ec1-8050-e54d78a308b6.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Delay Dependence (S = 5)</div>
<div style="text-align: center;">C.2 Chain Environment with H = S = 10</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b1b6/b1b6a89f-38d8-4817-b678-6d5b3353cda9.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Cumulative Regret (S = 10, E[τ] = 100).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a4b5/a4b5df25-6b76-4fb6-beda-b2fc6c76116a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: Cumulative Regret (S = 10, E[τ] = 300).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/42b1/42b12e52-6fd2-4ad4-a5d4-3296ed188d03.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: Cumulative Regret (S = 10, E[τ] = 500).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0059/0059b85a-f3d8-4bc1-b2a1-6d3770e162e4.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10: Delay Dependence (S = 10)</div>
# C.3 Chain Environment with H = S = 20
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e6a0/e6a01b20-a2d7-405a-8bf3-ab5c17a1a7d9.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 11: Cumulative Regret (S = 20, E[τ] = 100).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/14f8/14f852bd-ce5f-4eab-9ac5-bd86480154ae.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 12: Cumulative Regret (S = 20, E[τ] = 300).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0299/0299d282-4b2e-4e9e-8dba-2cceb260e671.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 13: Cumulative Regret (S = 20, E[τ] = 500).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/019a/019a9342-b61f-44fb-a81c-08ec85c80851.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 14: Delay Dependence (S = 20)</div>
# C.4 Chain Environment with H = S = 30
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b30b/b30b9363-f0cd-4716-a0e7-ad9a0903dcb6.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 15: Cumulative Regret (S = 30, E[τ] = 300).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a8ed/a8ed65aa-48ba-4852-a4d1-e06b182e95ab.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 16: Cumulative Regret (S = 30, E[τ] = 500).</div>
