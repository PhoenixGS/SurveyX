# Computational Rationalization: The Inverse Equilibrium Problem
Kevin Waugh Brian D. Ziebart J. Andrew Bagnell Carnegie Mellon University, 5000 Forbes Ave, Pittsburgh, PA, USA 15213
# Abstract
Modeling the purposeful behavior of imperfect agents from a small number of observations is a challenging task. When restricted to the single-agent decision-theoretic setting, inverse optimal control techniques assume that observed behavior is an approximately optimal solution to an unknown decision problem. These techniques learn a utility function that explains the example behavior and can then be used to accurately predict or imitate future behavior in similar observed or unobserved situations. In this work, we consider similar tasks in competitive and cooperative multi-agent domains. Here, unlike single-agent settings, a player cannot myopically maximize its reward — it must speculate on how the other agents may act to influence the game’s outcome. Employing the game-theoretic notion of regret and the principle of maximum entropy, we introduce a technique for predicting and generalizing behavior, as well as recovering a reward function in these domains.
# 1. Introduction
Predicting the actions of others in complex and strategic settings is an important facet of intelligence that guides our interactions—from walking in crowds to negotiating multi-party deals. Recovering such behavior from merely a few observations is an important and challenging machine learning task.
While mature computational frameworks for decisionmaking have been developed to prescribe the behavior that an agent should perform, such frameworks are
This is a preliminary draft. Copyright 2011 by the author(s)/owner(s).
waugh@cs.cmu.edu bziebart@cs.cmu.edu dbagnell@ri.cmu.edu
often ill-suited for predicting the behavior that an agent will perform. Foremost, the standard assumption of decision-making frameworks that a criteria for preferring actions (e.g., costs, motivations and goals) is known a priori often does not hold. Moreover, real behavior is typically not consistently optimal or completely rational; it may be influenced by factors that are difficult to model or subject to various types of error when executed. Meanwhile, the standard tools of statistical machine learning (e.g., classification and regression) may be equally poorly matched to modeling purposeful behavior; an agent’s goals often succinctly, but implicitly, encode a strategy that would require tremendous amounts of data to learn.
In the single-agent decision-theoretic setting, inverse optimal control methods have been used to bridge this gap between the prescriptive frameworks and predictive applications (Abbeel & Ng, 2004; Ratliff et al., 2006; Ziebart et al., 2008a; 2010). Successful applications include learning and prediction tasks in personalized vehicle route planning (Ziebart et al., 2008a), robotic crowd navigation (Henry et al., 2010), quadruped foot placement and grasp selection (Ratliff et al., 2009). A reward function is learned by these techniques that both explains demonstrated behavior
and approximates the optimality criteria of prescriptive decision-theoretic frameworks.
As these methods only capture a single reward function and do not reason about competitive or cooperative motives, inverse optimal control proves inadequate for modeling the strategic interactions of multiple agents. In this paper, we consider the gametheoretic concept of regret as a necessary stand-in for the optimality criteria of the single-agent work. As with the inverse optimal control problem, the result is fundamentally ill-posed. We address this by requiring that for any utility function linear in known features, our learned model must have no more regret than that of the observed behavior. We demonstrate that this requirement can be re-cast as a set of equivalent convex constraints that we denote the inverse correlated equilibrium (ICE) polytope. As we are interested in the effective prediction of behavior, we will use a maximum entropy criteria to select behavior from this polytope. We demonstrate that optimizing this criteria leads to mini-max optimal prediction of behavior subject to approximate rationality. We consider the dual of this problem and note that it generalizes the traditional log-linear maximum entropy family of problems (Della Pietra et al., 2002). We provide a simple and computationally efficient gradient-based optimization strategy for this family and show that only a small number of observations are required for accurate prediction and transfer of behavior. We conclude by considering a matrix routing game and compare the ICE approach to a variety of natural alternatives.
Before we formalize imitation learning in matrix games, motivate our assumptions and describe and analyze our approach, we will review the game-theoretic notions of regret and the correlated equilibrium.
# 2. Game Theory Background
Matrix games are the canonical tool of game theorists for representing strategic interactions ranging from illustrative toy problems, such as the “Prisoner’s Dilemma” and the “Battle of the Sexes” games, to important negotiations, collaborations, and auctions. In this work, we employ a class of games with payoffs or utilities that are linear functions of features defined over the outcome space.
Definition 1. A linearly parameterized normalform game, or matrix game, Γ = (N, A, F), is composed of: a finite set of players, N; a set of joint-actions or outcomes, A = ×i∈NAi, consisting of a finite set of actions for each player, Ai; a set
For notational convenience, we let a−i denote the vector a excluding component i and let A−i = ×j̸=i,j∈NAi be the set of such vectors.
We model the players with a distribution σ ∈∆A over the game’s joint-actions. Coordination between players can exist, thus, this distribution need not factor into independent strategies for each player. Conceptually, a signaling mechanism, such as a traffic light, can be thought to sample a joint-action from σ and communicate to each player ai, its portion of the joint-action. Each player can then consider deviating from ai using a modification function, fi : Ai �→Ai (Blum & Mansour, 2007).
#   The switch modification function, for instance,
(1)
substitutes action y for recommendation x. Instantaneous regret measures how much a player would benefit from a particular modification function when the coordination device draws joint-action a,
(2) (3) (4)
Players do not have knowledge of the complete jointaction; thus, each must reason about the expected regret with respect to a modification function,
(5) (6)
It is helpful to consider regret with respect to a class of modification functions. Two classes are particularly important for our discussion. First, internal regret corresponds to the set of modification functions where a single action is replaced by a new action, Φint i =
{switchx→y i (·) : ∀x, y ∈Ai}. Second, swap regret corresponds to the set of all modification functions, Φswap i = {fi}. We denote Φ = ∪i∈N Φi.
# The expected regret with respect to Φ and outcome distribution σ,
(7)
is important for understanding the incentive to deviate from, and hence the stability of, the specified behavior. The most general modification class, Φswap, leads to the notion of ε-correlated equilibrium (Osborne & Rubinstein, 1994), in which σ satisfies RΦswap(σ, w∗) ≤ ε. Thus, regret can be thought of as a substitute for utility when assessing the optimality of behavior in multi-agent settings.
# 3. Imitation Learning in Matrix Games
We are now equipped with the tools necessary to introduce our approach for imitation learning in multiagent settings. As input, we observe a sequence of outcomes, {am}M m=1, sampled from σ, the true behavior. We denote the empirical distribution of this sequence, ˜σ, the demonstrated behavior. We aim to learn a predictive behavior distribution, ˆσ from these demonstrations. Moreover, we would like our learning procedure to extract the motives and intent for the behavior so that we may imitate the players in similarly structured, but unobserved games.
Imitation appears hard barring further assumptions. In particular, if the agents are unmotivated or their intentions are not coerced by the observed game, there is little hope of recovering principled behavior in a new game. Thus, we require some form of rationality.
# 3.1. Rationality Assumptions
We say that agents are rational under their true preferences when they are indifferent between ˆσ and their true behavior if and only if RΦ(ˆσ, w∗) ≤RΦ(σ, w∗).
As agents’ true preferences w∗are unknown to the observer, we must consider an encompassing assumption that requires any behavior that we estimate to satisfy this property for all possible utility weights, or
∀w ∈RK, RΦ(ˆσ, w) ≤RΦ(σ, w).
(8)
Any behavior achieving this restriction, strong rationality, is also rational, and, by virtue of the contrapositive, we see that unless we have additional information regarding the agents’ true preferences, we must assume this strong assumption or we risk violating rationality.
Lemma 1. If strong rationality does not hold for alternative behavior ˆσ then there exist agent utilities such that they would prefer σ to ˆσ.
By restricting our attention to behavior that satisfies strong rationality, at worst, agents acting according to unknown true preference w∗will be indifferent between our predictive distribution and their true behavior.
# 3.2. Inverse Correlated Equilibria
Unfortunately, a direct translation of the strong rationality requirement into constraints on the distribution ˆσ leads to a non-convex optimization problem as it involves products of varying utility vectors and the behavior to be estimated. Fortunately, however, we can provide an equivalent concise convex description of the constraints on ˆσ that ensures any feasible distribution satisfies strong rationality. We denote this set of equivalent constraints as the Inverse Correlated Equilibria (ICE) polytope:
(9)
Theorem 1. A distribution, ˆσ, satisfies the constraints above for some η if and only if it satisfies strong rationality. That is, ∀w ∈RK, RΦ(ˆσ, w) ≤ RΦ(˜σ, w) if and only if ∀fi ∈Φ, ∃ηfi ∈∆Φ such that ˆσTRfi i = � fj∈Φ ηfi fj ˜σTRfj j .
 � The proof of Theorem 1 is provided in the Appendix (Waugh et al., 2011).
We note that this polytope, perhaps unsurprisingly, is similar to the polytope of correlated equilibrium itself, but here is defined in terms of the behavior we observe instead of the (unknown) reward function. Given any observed behavior σ, the constraints are feasible as the demonstrated behavior satisfies them; our goal is to choose from these behaviors without estimating a full joint-action distribution. While the ICE polytope establishes a basic requirement for estimating rational behavior, there are generally infinitely many distributions consistent with its constraints.
# 3.3. Principle of Maximum Entropy
As we are interested in the problem of statistical prediction of strategic behavior, we must find a mechanism to resolve the ambiguity remaining after accounting for the rationality constraints. The principle of maximum entropy provides a principled method for
choosing such a distribution. This choice leads to not only statistical guarantees on the resulting predictions, but to efficient optimization.
The Shannon entropy of a distribution ˆσ is defined as H(ˆσ) = −� x∈X ˆσx log ˆσx. The principle of maximum entropy advocates choosing the distribution with maximum entropy subject to known (linear) constraints (Jaynes, 1957):
(10)
The resulting log-linear family of distributions (e.g., logistic regression, Markov random fields, conditional random fields) are widely used within statistical machine learning. For our problem, the constraints are precisely that the distribution is in the ICE polytope, ensuring that whatever distribution is learned has no more regret than the demonstrated behavior.
The proof of Lemma 2 follows immediately from the result of Gr¨unwald and Dawid (2003).
In the context of multi-agent behavior, the principle of maximum entropy has been employed to obtain correlated equilibria with predictive guarantees in normalform games when the utilities are known a priori (Ortiz et al., 2007). We will now leverage its power with our rationality assumption to select predictive distributions in games where the utilities are unknown.
# 3.4. Prediction of Behavior
Let us first consider prediction of the demonstrated behavior using the principle of maximum entropy and our strong rationality condition. After, we will extend to behavior transfer and analyze the error introduced as a by-product of sampling ˜σ from σ.
The mathematical program that maximizes the entropy of ˆσ under strong rationality with respect to ˜σ,
(11)
is convex with linear constraints, feasible, and bounded. That is, it is simple and can be efficient solved in this form. Before presenting our preferred dual optimization procedure, however, let us describe an approach for behavior transfer that further illustrates the advantages of this approach over directly estimating σ.
# 3.5. Transfer of Behavior
A principal justification of inverse optimal control techniques that attempt to identify behavior in terms of utility functions is the ability to consider what behavior might result if the underlying decision problem were changed while the interpretation of features into utilities remain the same (Ng & Russell, 2000; Ratliff et al., 2006). This enables prediction of agent behavior in a no-regret or agnostic sense in problems such as a robot encountering novel terrain (Silver et al., 2010) as well as route recommendation for drivers traveling to unseen destinations (Ziebart et al., 2008b). Econometricians are interested in similar situations, but for much different reasons. Typically, they aim to validate a model of market behavior from observations of product sales. In these models, the firms assume a fixed pricing policy given known demand. The econometrician uses this fixed policy along with product features and sales data to estimate or bound both the consumers’ utility functions as well as unknown production parameters, like markup and production cost (Berry et al., 1995; Nevo, 2001; Yang, 2009). In this line of work, the observed behavior is considered accurate to start with; it is not suitable for settings with limited observations. Until now, we have considered the problem of identifying behavior in a single game. We note, however, that our approach enables behavior transfer to games equipped with the same features. We denote this unobserved game as ¯Γ. As with prediction, to develop a technique for behavior transfer we assume a link between regret and the agents’ preferences across the known space of possible preferences. Furthermore, we assume a relation between the regrets in both games. Property 1 (Transfer Rationality). For some constant κ > 0,
(12)
Roughly, we assume that under preferences with low regret in the original game, the behavior in the unobserved game should also have low regret. By enforcing this property, if the agents are performing well with respect to their true preferences, then the transferred behavior will also be of high quality.
As we are not privileged to know κ and this property is not guaranteed to hold, we introduce a slack variable to allow for violations of the strong rationality constraints to guaranteeing feasibility. Intuitively, the transfer-ICE polytope we now optimize over requires that for any linear reward function and for every player, the predicted behavior in a new game must have no more regret than demonstrated behavior does in the observed game using the same parametric form of reward function. The corresponding mathematical program is:
(13)
One could choose to institute multiple slack variables, say one for each fi ∈¯Φ, instead of a single slack across all modification functions. Our choice is motivated by the interpretation of the dual multipliers presented in the next section. There, we will also address selection of an appropriate value for C.
# 4. Duality and Efficient Optimization
In this section, we will derive, interpret and describe a procedure for optimizing the dual program for solving the MaxEnt ICE problem. We will see that the dual multipliers can be interpreted as utility vectors and that optimization in the dual has computational advantages. We begin by presenting the dual of the
Algorithm 1 Dual MaxEnt ICE
Input: T, γ, C > 0, R, ¯R, Φ and ¯Φ
∀fi ∈¯Φ, αfi, βfi ←1/(|¯Φ|K + 1)
for t from 1 to T do
/* compute the gradient */
∀a ∈¯
A, za ←exp
�
−�
fi∈¯Φ ¯rfiT
i,a (αfi −βfi)
�
Z ←�
a∈¯
A za
for fi ∈¯Φ do
f ∗
j ←argmaxfj∈Φ ˜σTRfj
j (αfi −βfi)
gfi ←˜σTR
f ∗
j
j∗−�
a∈¯
A za¯rfi
i,a/Z
end for
/* descend and project */
γt ←γ/
√
t
ρ ←1 + �
fi,k αfi
k exp(−γtgfi
k ) + βfi
k exp(γtgfi
k )
∀fi ∈¯Φ, k ∈K, αfi
k ←Cαfi
k exp(−γtgfi
k )/ρ
∀fi ∈¯Φ, k ∈K, βfi
k ←Cβfi
k exp(γtgfi
k )/ρ
end for
return (α, β)
transfer program.
where Z(α, β) is the partition function,
Removing the equality constraint is equivalent to disallowing any slack. We derive the dual in the appendix (Waugh et al., 2011). For C > 0, the dual’s feasible set has non-empty interior and is bounded. Therefore, by Slater’s condition, strong duality holds – there is no duality gap. In particular, we can use a dual solution to recover ˆσ. Lemma 4. Given a dual solution, (α, β), we can recover the primal solution, ˆσ. Specifically,
(15)
Intuitively, the probability of predicting an outcome is small if that outcome has high regret. In general, the dual multipliers are utility vectors associated with each modification function in ¯Φ. Under
the slack formulation, there is a natural interpretation of these variables as a single utility vector. Given a dual solution, (α, β) with slack penalty C, we choose
(17)
(18)
That is, we can associate with each modification function a probability, πfi, and a utility vector, λfi. Thus, a natural estimate for ˆw is the expected utility vector. Note, � fi∈¯Φ πfi need not sum to one. The remaining mass, ξ, is assigned to the zero utility vector. The above observation implies that introducing a slack variable coincides with bounding the L1 norm of the utility vectors under consideration by C. This insight suggests that we choose C ≥||w∗||1, if possible, as smaller values of C will exclude w∗from the feasible set. If a bound on the L1 norm is not available, we may solve the prediction problem on the observed game without slack and use || ˆw||1 as a proxy. The dual formulation of our program has important inherent computational advantages. First, it is a optimization over a simple set that is particularly wellsuited for gradient-based optimization, a trait not shared by the primal program. Second, the number of dual variables, 2|Φ|K, is typically much fewer than the number of primal variables, |A| + 2|Φ|2. Though the work per iteration is still a function of |A| (to compute the partition function), these two advantages together let us scale to larger problems than if we consider optimizing the primal objective. Computing the expectations necessary to descend the dual gradient can leverage recent advances in the structured, compact game representations: in particular, any graphical game with low-treewidth or finite horizon Markov game (Kakade et al., 2003) enables these computations to be performed in time that scales only polynomially in the number of decision makers or time-steps. Algorithm 1 employs exponentiated gradient descent (Kivinen & Warmuth, 1995) to find an optimal dual solution. The step size parameter, γ, is commonly taken to be � 2 log |¯Φ|K/∆, with ∆being the largest value in any Rfi i . With this step size, if the optimization is run for T ≥2∆2 log � |¯Φ|K � /ϵ2 iterations then the dual solution will be within ϵ of optimal. Alternatively, one can exactly measure the duality gap on each iteration and halt when the desired accuracy is
Algorithm 1 employs exponentiated gradient descent (Kivinen & Warmuth, 1995) to find an optimal dual solution. The step size parameter, γ, is commonly taken to be � 2 log |¯Φ|K/∆, with ∆being the largest value in any Rfi i . With this step size, if the optimization is run for T ≥2∆2 log � |¯Φ|K � /ϵ2 iterations then the dual solution will be within ϵ of optimal. Alternatively, one can exactly measure the duality gap on each iteration and halt when the desired accuracy is achieved. This is often preferred as the lower bound on the number of iterations is conservative in practice.
# 6. Experimental Results
To evaluate our approach experimentally, we designed a simple routing game shown in Figure 1. Seven drivers in this game choose how to travel home during rush hour after a long day at the office. The different road segments have varying capacities, visualized by the line thickness in the figure, that make some of them more or less susceptible to congestion or to traffic accidents. Upon arrival home, each driver records the total time and distance they traveled, the gas that they used, and the amount of time they spent stopped at intersections or in traffic jams – their utility features.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a1ab/a1ab5d23-dbf6-48cf-bc4e-987a870fcd3d.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/48fc/48fc8b24-79dc-4ad4-938a-f7f3793a12a8.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1. A visualization of the routing game.</div>
011 In this game, each of the drivers chooses from four possible routes (solid lines in Figure 1), yielding over 16,000 possible outcomes. We obtained an ε-social welfare maximizing correlated equilibrium for those drivers where the drivers preferred mainly to minimize their travel time, but were also slightly concerned with gas usage. The demonstrated behavior ˜σ was sampled from this true behavior distribution σ. First, we evaluate the differences between the true behavior distribution σ and the predicted behavior distribution ˆσ trained from observed behavior sampled from ˜σ. In Figure 2 we compare the prediction accuracy when varying the number of observations using log-loss, −� a∈A σa log ˆσa. The baseline algorithms we compare against are: a maximum likelihood estimate of the distribution over the joint-actions with a uniform prior, an exponential family distribution parameterized by the outcome’s utilities trained with logistic regression, and a maximum entropy inverse optimal control approach (Ziebart et al., 2008a) trained individually for each player. In Figure 2, we see that MaxEnt ICE predicts behavior with higher accuracy than all other algorithms when the number of observations is limited. In particular, it achieves close to its best performance with as few at 16 observations. The maximum likelihood estimator eventually overtakes it, as expected since it will ultimately converge to σ, but only after 10,000 observations, or about as many observations as there are outcomes in the game. This experiment demonstrates that learning underlying utility functions to estimate observed behavior can be much more data-efficient for small sample sizes, and additionally, that the regretbased assumptions of MaxEnt ICE are both reasonable
<div style="text-align: center;">Figure 2. Prediction error (log-loss) as a function of number of observations.</div>
<div style="text-align: center;">Table 1. Transfer error (log-loss) on unobserved games.</div>
Problem
Logistic Model
MaxEnt Ice
Add Highway
4.177
3.093
Add Driver
4.060
3.477
Gas Shortage
3.498
3.137
Congestion
3.345
2.965
and beneficial in our strategic routing game setting. Next, we evaluate behavior transfer from this routing game to four similar games, the results of which are displayed in Table 1. The first game, Add Highway, adds the dashed route to the game. That is, we model the city building a new highway. The second game, Add Driver, adds another driver to the game. The third game, Gas Shortage, keeps the structure of the game the same, but changes the reward function to make gas mileage more important to the drivers. The final game, Congestion, adds construction to the major roadway, delaying the drivers. These transfer experiments even more directly demonstrate the benefits of learning utility weights rather than directly learning the joint-action distribution; direct strategy-learning approaches are incapable of being applied to general transfer setting. Thus, we only compare against the Logistic Model. We see from Table 1 that MaxEnt ICE outperforms the Logistic Model in all of our tests. For reference, in these new games, the uniform strategy has a loss of approximately 6.8 in all games, and the true behavior has a loss of approximately 2.7.
# 7. Conclusion
In this paper, we extended inverse optimal control to multi-agent settings by combining the principle of maximum entropy with the game-theoretic notion of regret. We observed that our formulation has a particularly appealing dual program, which led to a simple gradient-based optimization procedure. Perhaps the most appealing quality of our technique is its theoretical and practical sample complexity. In our experiments, MaxEnt ICE performed exceptionally well after only 0.1% of the game had been observed.
# Acknowledgments
This work is supported by the ONR MURI grant N00014-09-1-1052 and by the National Sciences and Engineering Research Council of Canada (NSERC).
# References
Abbeel, P. and Ng, A. Y. Apprenticeship learning via inverse reinforcement learning. In Proceedings of the International Conference on Machine Learning, 2004.
Jaynes, E. T. Information theory and statistical mechanics. Physical Review, 106(4):620–630, May 1957.
Kakade, S., Kearns, M., Langford, J., and Ortiz, L. Correlated equilibria in graphical games. In Proceedings of Electronic Commerce, pp. 42–47, 2003.
Appendix Rationality Properties and Primal Programs The proof of Theorem 1 relies upon the following technical lemmas. Lemma 5.
bTw ≤max ai∈A ai Tw ⇔∃λ ∈∆A s.t. bTw ≤λTAw
Proof of Lemma 5. Given bTw ≤maxai∈A aiTw, choose
Thus, bTw ≤maxai∈A aiTw = λTAw. Given ∃λ ∈∆A s.t. bTw ≤λTAw,
Lemma 6.
∀w ∈RK, bTw ≤max i∈N ai Tw ⇔∃λ ∈∆A s.t. b = λTA.
⇔the following linear program has optimal value 0
By strong duality for linear programming, the primal has value 0 iff the dual is feasible, which is exactly when ∃λ ∈∆A s.t. b = λTA.
(19)
(20) (21)
(22)
(23)
(24) (25) (26)
(28)
Proof of Theorem 1.
∀w ∈RK, RΦ(ˆσ, w) ≤RΦ(˜σ, w) ⇔∀w ∈RK, max fi∈Φ ˆσTRfi i w ≤max fi∈Φ ˜σTRfi i w ⇔∀fi ∈Φ, ∀w ∈RK, ˆσTRfi i w ≤max fj∈Φ ˜σTRfj j w ⇔∀fi ∈Φ, ∃ηfi ∈∆Φ s.t. ˆσTRfi i = � fj∈Φ ηfi fj ˜σTRfj j
The last step makes use of our second technical lemma.
Derivation of the Dual Program The Lagrange dual is
To solve the unconstrained inner optimization, we take derivatives w.r.t. σ, η and ν and set equal to 
Substituting into the Lagrangian, we get
(29) (30) (31) (32)
(32)

(34)
(35)
(36)
(38)
(41)
(44)
We eliminate δ by setting its partial derivative to 0, solving for δ
and substituting back into the objective
By inspection, at optimality, γfi = maxfj∈Φ ˜σTRfj j (αfi −βfi). Thus an equivalent program i
All that remains is to exponentiate both sides.
(45)
(46)
(48)
(49)
(50)
(53)
(54)
(55)
(56)
(57)
(58)

Sample Complexity Proof of Theorem 2.
We use the union bound in step 2, and Hoeffding’s inequality in step 3. Solving for M, we get our result
Proof of Corollary 1. We have ∀w, RΦ(ˆσ, w) ≤RΦ(˜σ, w) + ν ||w||1, where ν depends on the choice of the slack’s penalty. Thus, we have RΦ(ˆσ, w∗) ≤RΦ(˜σ, w∗)+ν ||w||1 ≤RΦ(σ, w∗)+(ϵ∆+ν) ||w∗||1 with probability at least 1 −δ, so long as M is as large as Theorem 2 deems. We can make ν as small as we like by increasing the slack penalty.
(59)
(61)
(62)
(63)
(64)
