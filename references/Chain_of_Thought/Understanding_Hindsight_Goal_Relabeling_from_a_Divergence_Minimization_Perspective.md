# Understanding Hindsight Goal Relabeling from a Divergence Minimization Perspective
Lunjun Zhang 1 2 3 Bradly C. Stadie 4
# Abstract
Hindsight goal relabeling has become a foundational technique in multi-goal reinforcement learning (RL). The essential idea is that any trajectory can be seen as a sub-optimal demonstration for reaching its final state. Intuitively, learning from those arbitrary demonstrations can be seen as a form of imitation learning (IL). However, the connection between hindsight goal relabeling and imitation learning is not well understood. In this paper, we propose a novel framework to understand hindsight goal relabeling from a divergence minimization perspective. Recasting the goal reaching problem in the IL framework not only allows us to derive several existing methods from first principles, but also provides us with the tools from IL to improve goal reaching algorithms. Experimentally, we find that under hindsight relabeling, Q-learning outperforms behaviour cloning (BC). Yet, a vanilla combination of both hurts performance. Concretely, we see that the BC loss only helps when selectively applied to actions that get the agent closer to the goal according to the Qfunction. Our framework also explains the puzzling phenomenon wherein a reward of {−1, 0} results in significantly better performance than a {0, 1} reward for goal reaching.
arXiv:2209.13046v2
# 1. Introduction
Goal reaching is an essential aspect of intelligence in sequential decision making. Unlike the conventional formulation of reinforcement learning (RL), which aims to encode all desired behaviors into a single scalar reward function that is amenable to learning (Silver et al., 2021), goal reaching formulates the problem of RL as applying a sequence of actions to rearrange the environment into a desired state
1Department of Computer Science, University of Toronto 2Vector Institute 3Waabi 4Department of Statistics, Northwestern University. Correspondence to: Lunjun Zhang <lunjun@cs.toronto.edu>.
1Department of Computer Science, University of Toronto 2Vector Institute 3Waabi 4Department of Statistics, Northwestern University. Correspondence to: Lunjun Zhang <lunjun@cs.toronto.edu>.
(Batra et al., 2020). Goal-reaching is a highly flexible formulation. For instance, we can design the goal-space to capture salient information about specific factors of variations (Plappert et al., 2018b); we can use natural language instructions to define more abstract goals (Lynch & Sermanet, 2020; Ahn et al., 2022); we can encourage exploration by prioritizing previously unseen goals (Pong et al., 2019; Warde-Farley et al., 2018; Pitis et al., 2020); and we can even use self-supervised procedures to naturally learn goal-reaching policies without reward engineering (Pong et al., 2018; Nair et al., 2018b; Zhang et al., 2021; OpenAI et al., 2021; Chebotar et al., 2021).
manet, 2020; Ahn et al., 2022); we can encourage exploration by prioritizing previously unseen goals (Pong et al., 2019; Warde-Farley et al., 2018; Pitis et al., 2020); and we can even use self-supervised procedures to naturally learn goal-reaching policies without reward engineering (Pong et al., 2018; Nair et al., 2018b; Zhang et al., 2021; OpenAI et al., 2021; Chebotar et al., 2021). Imitation learning (IL) aims to recover an expert policy from a set of expert demonstrations. The simplest imitation learning algorithm is Behaviour Cloning (BC), which directly uses the given demonstrations to supervise the policy actions conditioned on the states visited by the expert (Pomerleau, 1988). An alternative approach, inverse reinforcement learning (IRL), first learns a reward function from the demonstrations, and then runs online RL on the learned reward to extract the policy (Abbeel & Ng, 2004). A modern example of IRL is generative adversarial imitation learning (GAIL) (Ho & Ermon, 2016), which learns a discriminator (Goodfellow et al., 2014) as the reward while jointly running policy gradient algorithms. It has been shown that the majority of imitation learning methods can be unified under the divergence minimization framework (Ghasemipour et al., 2020; Zhang et al., 2020; Ke et al., 2021). Hindsight goal relabeling is a technique proposed in (Andrychowicz et al., 2017) to improve the sample efficiency of goal reaching and has since been widely used (Ghosh et al., 2019; Lynch et al., 2019; Chebotar et al., 2021). A goal-conditioned policy typically sets a behavioral goal and then tries to reach it during sampling, but most attempts will likely fail. Nevertheless, any trajectory has successfully reached all the states it visits along the way. Therefore, the agent may pretend post hoc that whatever states it reaches are the intended goals of that trajectory, and learn from its own success. Intuitively, hindsight relabeling creates self-generated expert demonstrations, which the policy then imitates. Can we mathematically describe hindsight relabeling and goal reaching as an imitation learning
Hindsight goal relabeling is a technique proposed in (Andrychowicz et al., 2017) to improve the sample efficiency of goal reaching and has since been widely used (Ghosh et al., 2019; Lynch et al., 2019; Chebotar et al., 2021). A goal-conditioned policy typically sets a behavioral goal and then tries to reach it during sampling, but most attempts will likely fail. Nevertheless, any trajectory has successfully reached all the states it visits along the way. Therefore, the agent may pretend post hoc that whatever states it reaches are the intended goals of that trajectory, and learn from its own success. Intuitively, hindsight relabeling creates self-generated expert demonstrations, which the policy then imitates. Can we mathematically describe hindsight relabeling and goal reaching as an imitation learning
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ecd1/ecd18dd2-ea1e-4263-b785-14588bb6e807.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Our proposed framework for goal reaching that elucidates how rewards should be designed in multi-goal RL.</div>
process from a divergence minimization perspective?
In this paper, we propose a novel framework that describes hindsight goal relabeling in the language of divergence minimization, further bridging the two paradigms of goal reaching and imitation learning. Compared to prior attempts to formulate a theoretical framework for goal reaching (Tang & Kucukelbir, 2021; Eysenbach et al., 2020a; Blier et al., 2021; Eysenbach et al., 2020b; Rudner et al., 2021), our work pays substantially more attention to reward design in goal reaching and its connection to inverse RL, and having a probabilistically well-defined objective in the case of continuous state space. In addition, our framework can re-derive several existing methods from first principles, unlike prior frameworks that are incompatible with existing methods and thus unable to explain the success of hindsight-relabeling based goal reaching. For experiments, we find that despite a resurgence of interests in BC-based goal reaching (Ghosh et al., 2019; Ding et al., 2019; Lynch et al., 2019; Jang et al., 2021), multi-goal Q-learning can still outperform BC, and adding BC loss in multi-goal Q-learning hurts performance due to the sub-optimality of self-generated demonstrations. Indeed, BC only helps when selectively applied to actions that get the agent closer to the goal according to the Q-function. Finally, our framework also provides an interesting explanation for why the rewards of −1 and 0 experimentally outperforms the rewards of 0 and 1 in multi-goal Q-learning.
# 2. Goal Reaching and Imitation Learning
# 2.1. Goal-Conditioned Reinforcement Learning (GCRL)
We first review the basics of RL. A Markov Decision Process (MDP) is typically parameterized by (S, A, ρ0, p, r): a state space S, an action space A, an initial state distribution ρ0(s), a dynamics function p(s′ | s, a) which defines the transition
probability, and a reward function r(s, a). A policy function µ(a | s) defines a probability distribution µ : S × A →R+. For an infinite-horizon MDP, given the policy µ, and the state distribution at step t (starting from ρ0 at t = 0), the state distribution at step t + 1 is given by:
(1)
The state visitation distribution sums over all timesteps via a geometric distribution Geom(γ):
(2)
However, the trajectory sampling process does not happen in this discounted manner, so the discount factor γ ∈(0, 1) is often absorbed into the cumulative return instead (Silver et al., 2014):
From 1 and 2, we also see that the future state distribution p+(s+ | s, a) of policy µ, defined as a geometric sum of state distribution at all future timesteps given current state and action, is given by the following recursive relationship (Eysenbach et al., 2020b; Janner et al., 2020):
(4)
In multi-goal RL, an MDP is augmented with a goal space G, and we learn a goal-conditioned policy π : S×G×A →R+.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/49ff/49ff7961-e596-4c0f-933c-0caf4ae009e5.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0102/0102fb29-00df-4dca-89e4-6580c02faf13.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: ρµ(s, a)p+ µ (s+ | s, a) hindsight-relabeled distribution</div>
<div style="text-align: center;">Figure 3: ρ+(g)ρµ(s)π(a | s, g) (used in HBC / GCSL)</div>
Hindsight Experience Replay (HER) (Andrychowicz et al., 2017) gives the agent a reward of 0 when the goal is reached and −1 otherwise, typically determined via an epsilon ball,
(5)
and uses hindsight goal relabeling to increase learning efficiency of goal-conditioned Q-learning by replacing the initial behavioral goals with achieved goals (future states within the same trajectory).
# 2.2. Imitation Learning (IL) as Divergence Minimization
We review the concept of f-divergence between two probability distributions P and Q and its variational bound:
(6)
where f is a convex function such that f(1) = 0, f ∗is the convex conjugate of f, and T is an arbitrary class of functions T : X →R. This variational bound was originally derived in (Nguyen et al., 2010) and was popularized by GAN (Goodfellow et al., 2014; Nowozin et al., 2016) and subsequently by imitation learning (Ho & Ermon, 2016; Fu et al., 2017; Finn et al., 2016; Ghasemipour et al., 2020). The equality holds true under mild conditions (Nguyen et al., 2010), and the optimal T is given by T ∗(x) = f ′(p(x)/q(x)). The canonical formulation of imitation learning follows (Ho & Ermon, 2016; Ghasemipour et al., 2020), where ρexp(s, a) is from the expert:
(7)
Because of the policy gradient theorem (Sutton et al., 1999), the policy µ needs to optimize the cumulative return under its own trajectory distribution ρµ(s, a) with the reward being r(s, a) = f ∗(T(s, a)). Under this formulation, JensenShannon divergence leads to GAIL (Ho & Ermon, 2016),
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/147d/147d02ef-da6a-40fa-8d79-992651610717.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: ρ+(g)ρπ(s, a | g) (used in HER / HDM)</div>
reverse KL leads to AIRL (Fu et al., 2017). Note that f can in principle be any convex function (we can satisfy f(1) = 0 by simply adding a constant).
# 3. Graphical Models for Hindsight Goal Relabeling
Consider an environment ξ = (ρ0(s), p(s′ | s, a), p(g)). From ξ, we generate a dataset of trajectories D = {(s0, a0, s1, a1, · · · )} by first sampling from the initial state distribution ρ0(s), and then executing an unobserved actor policy µ(a | s). After executing this action, transitions occur according to a dynamics model p(s′ | s, a). We aim to train a goal-conditioned policy π(a | s, g) from this arbitrary dataset with relabeled future states as goals. p(g) is the behavioral goal distribution assumed to be given apriori by the environment. To recast the problem of goal-reaching as imitation learning (7), we need to set up an f-divergence minimization where we define the target (expert) distribution and the policy distribution we want to match.
In multi-goal RL, the training signal comes from factorizing the joint distribution of state-action-goal differently. For the relabeled target distribution, we assume an unconditioned actor µ generating a state-action distribution at first, with the goals coming from the future state distribution (4) conditioned on the given state and action. For the goalconditioned policy distribution, behavior goals are given apriori, and the state-action distribution is generated conditioned on the behavioral goals. Thus, the target distribution (see Figure 2) for states, actions, and hindsight goals is:
(8)
Concretely, the target distribution is given by the unconditioned state-action visitation distribution multiplied by the conditional likelihood of future states. Note that p+ µ is given by equation (4), and ρµ(s, a) is similar to ρexp(s, a) in equation (7). In the fashion of behavioral cloning (BC), if we do not care about matching the states, we can write the joint distribution we are trying to match as (see Figure 3):
(9)
We can recover the objective of Hindsight Behavior Cloning (HBC) (Ding et al., 2019; Eysenbach et al., 2020a) and GoalConditioned Supervised Learning (GCSL) (Ghosh et al., 2019) via minimizing a KL-divergence (see A.1):
(10)
In many cases, matching the states is more important than matching state-conditioned actions (Ross et al., 2011; Ghasemipour et al., 2020). The joint distribution for states, actions, behavioral goals for π (see Figure 4) is:
(11)
(12)
After adding this state-matching term, we are still missing an important component of the objective. Divergence minimization encourages the agent to stay in the ”right” state-action distribution given the benefit of hindsight. But we still want the policy to actually hit the goal as quickly as possible without detours (see Figure 1). We propose that the goal-conditioned policy π should therefore minimize:
(13)
�  | Which estimates the expected number of steps for the policy to reach a certain goal. The overall objective for goalconditioned RL can seen as a combination of LIL and LRL. We will expand the two objectives in more details and discuss how to use Q-learning to optimize them.
# 4. Bridging Goal Reaching and Imitation
# 4. Bridging Goal Reaching and Imitation Learning
In this section, we study how to optimize the proposed goal reaching objectives (12) and (13), with the goal of deriving and understanding the Q-learning process and reward design in multi-goal RL.
# 4.1. Divergence Minimization with Goal-Conditioned Q-learning
This section decomposes (12). We start with the fdivergence bound from equations (6) and (7).
Df(pπ(s, a, g) ∥pµ(s, a, s+)) = max T E p(g) ρπ(s,a|g) [T(s, a, g)] −E ρµ(s,a) p+ µ (s+|s,a) [f ∗(T(s, a, s+))]
Now we negate T to get r(s, a, g) = −T(s, a, g), and the divergence minimization problem becomes:
 − inimization problem becomes: ρµ(s,a) +  (s+|s,a) [f ∗(−r(s, a, s+))] + E p(g) ρπ(s,a|g) [r(s, a, g)] The challenge, however, is that defining this value rigorously for continuous state space is a difficult task. With continuous state space, the delta function 1[s′ = g] is always zero, and
max π min r E ρµ(s,a) p+ µ (s+|s,a) [f ∗(−r(s, a, s+))] + E p(g) ρπ(s,a|g) [r(s, a, g)] for continuous state space is a difficult task. With continuous state space, the delta function 1[s′ = g] is always zero, and
We can interpret r as a GAIL-style (Ho & Ermon, 2016) discriminator or reward. However, we aim to derive a discriminator-free learning process that directly trains the Q-function corresponding to this reward Q(s, a, g) = r(s, a, g) + γ · PπQ(s, a, g) where Pπ is the transition operator: PπQ(s, a, g) = Ep(s′|s,a)π(a′|s′,g)[Q(s′, a′, g)]. Re-writing the previous equation w.r.t Q:
(14)
A similar change-of-variable has been explored in the context of offline RL (Nachum et al., 2019a;b) and imitation learning (Kostrikov et al., 2019; Zhu et al., 2020), known as the DICE (Nachum & Dai, 2020) family. A major pain point of DICE-like methods is that they require samples from the initial state distribution ρ0(s) (Garg et al., 2021). The following lemma shows that in the goal-conditioned case, we can use arbitrary offline trajectories to evaluate the expected rewards under goal-conditioned online rollouts: Lemma 4.1 (Online-to-offline transformation for goal reaching). Given a goal-conditioned policy π(a | s, g), its corresponding Q-function Qπ(s, a, g), and arbitrary state-action visitation distribution ρµ(s, a) of another policy µ(a | s), the expected temporal difference for online rollouts under π is:
# Using Lemma 4.1, the imitation objective now becomes:
max π min Q Eρµ(s,a)p+ µ (s+|s,a) � f ∗(−(Q −γPπQ)(s, a, s+)) �
(15)
where f ∗is the convex conjugate of f in f-divergence. We can pick almost any convex function as f ∗as long as ((f ∗)∗)(1) = 0. In summary, we have derived a way to minimize the imitation term LIL = Df(pµ(s, a, s+) ∥ pπ(s, a, g)) directly using goal-conditioned Q-learning.
# 4.2. Learning to Reach Goals with Fewer Steps
A goal-conditioned agent should try to reach the desired goal using as few steps as possible. Consequently, we may want to learn the expected number of steps to reach another goal state from the current state, and then uses the policy π to minimize the expected number of steps.
the dynamics function p(s′ | s, a) has a value in R+ rather than [0, 1]; we can only estimate the likelihood p(s′ = g | s, a) whose range is R+ and thus cannot be normalized to be either in {0, 1} or within [0, 1] per step for the purpose of step counting. To tackle this issue, we propose the following definition to measure the expected number of steps (negated) from one state to another:
(16)
which normalizes the likelihood of reaching the goal at different ∆numbers of steps away as discrete probabilities over the step count, with an additional γ factor. Maximizing this Q function means minimizing (13). Lemma 4.2 (Recursive estimate of goal reaching step count). Under mild assumptions, the step count definition (16) has the following property:
which normalizes the likelihood of reaching the goal at different ∆numbers of steps away as discrete probabilities over the step count, with an additional γ factor. Maximizing this Q function means minimizing (13).
Lemma 4.2 (Recursive estimate of goal reaching step count). Under mild assumptions, the step count definition (16) has the following property:
with Qt = Q(st, a, g), Qt+1 = Eπ(st+1,g)[Q(st+1, a′, g)], α(i) t = � ∆=i γ∆pπ(st+1+∆= g | st, a), and βt = p(st+1 = g | st, a).
  The first component [1 −βt/(βt + α(1) t )] is classifying whether the goal can be reached in the immediate next step by comparing the two density ratios; the second component (−1 + Qt+1) shifts the distance estimate by another step in the case of the goal not being reached at t + 1. Rather than directly estimating the density ratio, we may adopt a sampling based approach: when sampling from βt, Qt should be 0, meaning that r = 0 (onward); when sampling from α(1) t , r = −1. Consequently, the definition in (16) reveals that the reward defined in (5) forms a well-defined objective even for continuous state space and as ϵ →0.
# 4.3. To Reach Goals with Fewer Steps is to Imitate
We now discuss the relationship between the RL objective in (13) and (16) and the objective to imitate in (12) and (15). Lemma 4.3 (Understanding Hindsight Experience Replay). Multi-goal Q-learning with HER reward (5) with {−1, 0} is a special case of minimizing the following objective:
E ρµ(s,a) p(s′|s,a) p+ µ (s+|s,a) [f ∗(−(Q −γPπQ)(s, a, s+)) −βQ(s, a, s′)] with β = (1 −γ) and the convex function f ∗chosen to be: f ∗(x) = (x −1)2/2 + 3/2
E ρµ(s,a) p(s′|s,a) p+ µ (s+|s,a) [f ∗(−(Q −γPπQ)(s, a, s+)) −βQ(s, a, s′)]
While the step counting interpretation in (16) already provides valid meaning to the HER rewards, re-writing the objective in this way reveals why this optimization process is also doing imitation learning. Its first term is the same as the one in divergence minimization objective (15). For the second term, rather than pushing down on the Q values under the current policy π in (15), which would incorrectly assume the optimality of self-generated demonstrations and hinder exploration, it pushes up on the transition tuple (s, a, s′), where the action a is guaranteed to be optimal for reaching the immediate next state s′. As a result, this objective encourages the policy to imitate the best in self-generated demonstrations without restraining exploration.
# 4.4. Can Hindsight BC Facilitate Q-learning?
Now that we have shown that the Q-learning process in HER is also doing a special form of imitation, one may wonder whether BC (10) has any additional role to play. We hypothesize that, for a softmax policy under discrete action space, hindsight BC might facilitate the Q-learning process. We have reasons to hypothesize so: under discrete action space, BC becomes a cross entropy loss and has been repeatedly shown to work very well (Duan et al., 2017; Lynch et al., 2019; Jang et al., 2021; Shafiullah et al., 2022; Brohan et al., 2022). Even large language models like GPT3 (Brown et al., 2020) can be seen as a softmax policy on discrete action space trained with BC. The Q-values can be seen as the logits of a softmax policy (Schulman et al., 2017a). The problem is that hindsight relabeled actions (s, a, s+) are often suboptimal beyond a single step of relabeling (s, a, s′). Utilizing the interpretation of Q-values as negated step count in (16), we can imitate only those actions that move the agent closer to a goal according to its own estimates, by defining a threshold w(s, a, s′, g) for imitation:
# w = 1(Q(s, a, g) −max a′ Q(s′, a′, g) < log γhdm) (
(17)
# and the additional objective on top of Q-learning with {−1, 0} rewards (in 4.3) becomes Lhdm:
and the additional objective on top of Q-learning with {−1, 0} rewards (in 4.3) becomes Lhdm:
Eρµ(s,a,s′,g)[−w(s, a, s′, g) · log softmaxQ(s, a, g)] (18)
(18)
which we call Hindsight Divergence Minimization (HDM). Intuitively, this additional loss imitates an action when the value functions believe that this action can move the agent closer to the goal by at least −log γhdm steps. The idea of imitating the best actions is similar to self-imitation learning (SIL) (Oh et al., 2018; Vinyals et al., 2019), but we study a (reward-free) goal-reaching setting, with the advantage function in SIL being replaced by the delta of step counts a particular action can produce in getting closer to the goal.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b894/b894b943-6f21-4a83-b484-9c4921bd86cb.png" style="width: 50%;"></div>
Figure 5: Goal-reaching environments from GCSL (Ghosh et al., 2019) that we consider for discrete action space experiments: reaching a goal location in Four Rooms, landing at a goal location in Lunar Lander, pushing a puck to a goal location in Sawyer Push, opening the door to a goal angle in Door Open (Nair et al., 2018b), turning a valve to a goal orientation in Claw Manipulate (Ahn et al., 2020) Importantly, those environments have discretized action space. We use softmax policy where Q-values are policy logits.
<div style="text-align: center;">Figure 5: Goal-reaching environments from GCSL (Ghosh et al., 2019) that we consider for discrete action space experiments: reaching a goal location in Four Rooms, landing at a goal location in Lunar Lander, pushing a puck to a goal location in Sawyer Push, opening the door to a goal angle in Door Open (Nair et al., 2018b), turning a valve to a goal orientation in Claw Manipulate (Ahn et al., 2020). Importantly, those environments have discretized action space. We use softmax policy where Q-values are policy logits.</div>
# 5. Related Work
Frameworks for Goal Reaching Many attempts have been made to rigorously formulate a theoretical framework for the goal reaching problem. The earliest work that aims to formulate goal-conditioned Q-functions as a step count is (Kaelbling, 1993), though it only considers discrete state space. The framework introduced in (Eysenbach et al., 2020a) considers a finite horizon MDP where the agent receives a −∞reward at the final timestep if the final state does not match the goal and a 0 reward in all other conditions, which does not match how multi-goal Q-learning works in practice. Hindsight EM (Tang & Kucukelbir, 2021) borrows concepts from control as inference (Levine, 2018) but only considers hindsight BC for policy learning. The difficulty of defining the objective of HER (Andrychowicz et al., 2017) discussed in 4.2 was also previously noted in C-learning (Eysenbach et al., 2020b), which concludes that ”it is unclear what quantity Q-learning with hindsight relabeling optimizes” and regards the Q-values of HER for continuous states as ill-defined. C-learning then proposed to directly estimate (4) as a workaround. Similarly, (Rudner et al., 2021) advocates directly fitting a dynamics model and using per-step reward log p(s′ = g | s, a) for goalreaching. However, those frameworks do not explain the success of simple binary rewards for goal reaching in continuous state space (Andrychowicz et al., 2017; Warde-Farley et al., 2018), which still achieves state-of-the-art results today (Chebotar et al., 2021). By contrast, our work clearly defines what hindsight relabeling under HER reward optimizes by illustrating its underlying graphical models and the meaning of its Q-values, both of which remain well-defined in the case of continuous state space.
Connecting Goal Reaching with Inverse RL Classical inverse RL either uses a max margin loss in the apprenticeship learning formulation or contrastive divergence loss (Hinton, 2002) in the maximum entropy formulation (Ng et al., 2000; Abbeel & Ng, 2004; Ziebart et al., 2008; Ho et al., 2016), and those ideas have been borrowed accordingly in the goal reaching literature (Eysenbach et al., 2022; Rudner et al., 2021). GAIL (Ho & Ermon, 2016) uses a
GAN-like discriminator (Goodfellow et al., 2014) as the reward, and (Ding et al., 2019) augments binary rewards in HER with smoother GAIL rewards for goal reaching. The introduction of f-GAN (Nowozin et al., 2016) generalizes GAN to more f-divergences, which led to a similar generalization of GAIL (Ghasemipour et al., 2020; Zhang et al., 2020), which led to f-divergence based exploration strategy (Durugkar et al., 2021) and a goal reaching algorithm without hindsight relabeling (Ma et al., 2022). SQIL (Reddy et al., 2019) is an imitation learning method that runs Q-learning on two constant rewards: r = 1 for expert data and r = 0 for policy data; SQIL can often outperform GAIL. The SQIL reward is similar to the HER reward in some ways, but prior works have not shown how this type of reward does imitation in goal reaching.
# 6. Experiments
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d016/d0168c5f-e1a1-485d-9418-807c8ce6db28.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Goal reaching environments from (Plappert et al., 2018a) we consider for benchmarking multi-goal rewards: HER (Andrychowicz et al., 2017) with (−1, 0) rewards, AM (Chebotar et al., 2021) with (0, 1) rewards, HER with (0, 1) rewards.</div>
<div style="text-align: center;">There are two parts of our experimental investigations.</div>
• We study how reward designs affect the goal reaching performance, and demonstrate that in multi-goal RL, seemingly insignificant details about rewards can lead to the striking difference between success and total failure in learning. The experimental results are in accordance with our theoretical framework which suggests that {−1, 0} rewards should work best.
• In the second part, we study the special case of a softmax policy on discrete action space, and find that despite advances in Hindsight BC, HER can still achieve significantly better results. Moreover, a vanilla combi-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/903f/903fc180-1120-47c3-bc32-9bb74b7a5afd.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Success rate comparisons between different multi-goal RL rewards on Fetch environments. The only difference between three methods is the reward and backup strategies; all other parts of implementation are the same. Results over 5 eeds are shown. Using a reward of {−1, 0} is crucial for learning success.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0436/04363c26-c428-4a72-ba6f-969795c141df.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: We find that the quantity ag change ratio is strongly indicative of learning progress in multi-goal RL. We define the ag change ratio of π as: the percentage of trajectories where the achieved goals (ag) in initial states s0 are different from the achieved goals in final states sT under π. Most training signals are only created from ag changes, because they provide examples of how to rearrange an environment. An increase in ag change ratio often precedes an increase in success rates.</div>
nation of HER + HBC hurts performance, but our proposed HDM loss (a variant of BC) can further improve the performance of HER on discrete action space.
6.1. Investigating Reward Designs in Goal Reaching We investigate 3 different reward designs for goal reaching: Hindsight Experience Replay (HER) (Andrychowicz et al., 2017) with {−1, 0} rewards:
(19)
Actionable Models (AM) (Chebotar et al., 2021) uses {0, 1} rewards, but directly defines Q(s, a, s′) as 1:
(20)
HER with {0, 1} rewards which is also a modified version of actionable models where the bellman backup continues after the goal is reached:
(21)
We use widely benchmarked Fetch environments (Plappert et al., 2018a) for multi-goal RL as the task suite to compare
those three strategies. For all three methods, we use the same set of hyper-parameters (including learning rate, batch size, network architecture, target network update frequency and polyak value, etc). The only differences are the rewards and bellman backups . The results are presented in Figure 7. Why does {−1, 0} reward work so well? Our framework shows that running Q-learning with {−1, 0} rewards leads to a probabilistically well-defined Q values in the form of a normalized step count (16). One can perhaps interpret HER with {0, 1} rewards as adding a constant offset 1/(1 −γ) on the converged Q-values of {−1, 0} rewards, but this offset is quite large and eventually causes learning to diverge in this case. Actionable Models (AM) address this issue by stopping further bellman backup once the goal is reached (s′ = g) and directly setting the Q-value to be 1. This does not align with our analysis of the divergence minimization objective (15), and fails to teach the policy to stop at the goal state once the goal is reached: under this bellman backup, once the goal is reached once, nothing matters afterwards. To summarize, using a reward of {−1, 0} is crucial for the success of multi-goal Q-learning with hindsight relabeling.
# 6.2. Goal Reaching with Discrete Action Space
In this section, we study whether our proposed HDM loss (18) can facilitate the goal-conditioned Q-learning process
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c17a/c17a23aa-87f7-4c06-81f7-fae6c53bd734.png" style="width: 50%;"></div>
gure 9: Intuitions about when HDM applies BC. From left to right: (a) the Maze environment with a goal in the upper-left corner; (b) e Q-values learned through the converged policy. Lighter red means higher Q-value; (c) visualizing the actions (from the replay) that et imitated when conditioning on the goal and setting γhdm = 0.95, with the background color reflecting how much an action moves the gent closer to the goal based on the agent’s own estimate; (d) γhdm = 0.85; (e) γhdm = 0.75. As we lower γhdm, the threshold −log γhdm ets higher and fewer actions get imitated, with the remaining imitated actions more concentrated around the goal. HDM uses Q-learning  account for the worst while imitating the best during the goal-reaching process.
<div style="text-align: center;">Figure 9: Intuitions about when HDM applies BC. From left to right: (a) the Maze environment with a goal in the upper-left corner; ( the Q-values learned through the converged policy. Lighter red means higher Q-value; (c) visualizing the actions (from the replay) th get imitated when conditioning on the goal and setting γhdm = 0.95, with the background color reflecting how much an action moves t agent closer to the goal based on the agent’s own estimate; (d) γhdm = 0.85; (e) γhdm = 0.75. As we lower γhdm, the threshold −log γh gets higher and fewer actions get imitated, with the remaining imitated actions more concentrated around the goal. HDM uses Q-learnin to account for the worst while imitating the best during the goal-reaching process.</div>
Success Rate (%)
Four Rooms
Lunar Landar
Sawyer Push
Door Opening
Claw Manipulate
GCSL / HBC
78.27 ±4.76
50.00 ±7.77
44.67 ±13.86
19.10 ±5.97
16.80 ±6.55
HER r = (0, 1)
86.60 ±4.22
39.30 ±6.81
57.60 ±6.61
82.50 ±4.87
22.80 ±6.43
HER + SQL
88.50 ±4.56
44.50 ±9.61
57.20 ±6.32
84.70 ±5.33
16.13 ±8.43
HER r = (−1, 0)
86.40 ±5.11
50.80 ±4.66
54.60 ±6.16
83.76 ±6.02
20.20 ±6.23
HER + HBC
82.90 ±6.24
35.33 ±4.57
52.63 ±8.05
76.44 ±5.37
16.93 ±8.03
HDM (ours)
96.27 ±2.56
57.60 ±7.21
66.00 ±5.13
88.60 ±4.63
27.89 ±6.46
Table 1: Benchmark results of test-time success rates in self-supervised goal-reaching, over 5 seeds. We compare our method HDM wi GCSL (Ghosh et al., 2019), HER (Andrychowicz et al., 2017) with two different types of rewards, SQL (Schulman et al., 2017a) (So Q-Learning) + HER, and HER + HBC. We find that HER can still outperforms GCSL / HBC, and a vanilla combination of HER + HB actually hurts performance. HDM selectively decides on what to imitate and outperforms HER and HBC.
on environments with either discrete or discretized action space (Figure 5), as mentioned in 4.4. Hindsight BC (or GCSL) with discrete action space is known to have promising results even in a totally self-supervised goal reaching setting without external demonstrations (Ghosh et al., 2019). To provide a fair comparison to HBC, we do not assume that the agent has direct access to the ground truth binary reward metric. Note the original HER uses this metric during relabeling (Andrychowicz et al., 2017), but assuming access to this feedback is often unrealistic for real-world robotlearning (Lin et al., 2019; Chebotar et al., 2021). Instead, a positive reward is provided only when the relabeled hindsight goal is the immediate next state. As a result, the training procedure is completely self-supervised, similar to HBC. Additionally, we also consider the HER + Soft Q-Learning (SQL) (Schulman et al., 2017a) baseline, since SQL often improves policy robustness (Haarnoja et al., 2018). HDM builds on top of HER with (−1, 0) rewards and adds a BClike loss with a clipping condition (18) such that only the actions that move an agent closer to the goal get imitated. See Figure 9 for more visualizations of when HDM applies the BC loss on top of a Q-learning process. The results in Table 1 show that HDM achieves the strongest performance on all environments, while reducing variances in success rates. Interestingly, there is no consensus best baseline algorithm, with all five algorithms achieving good
results in some environments and subpar performance in others. Combining HER with HBC (which blindly imitates all actions in hindsight) produces worse results than not imitating at all and only resorting to value learning. In contrast, HDM allows for better control over what to imitate. Further ablations on HDM are provided in the 10 of the appendix, which shows that HDM outperforms HER and GCSL across a variety of γhdm values. Our results indicate that BC alone can only serve as an auxiliary (or perhaps pretraining) objective; value learning is needed to implicitly model the future (13) and improve the policy.
# 7. Conclusion
This work presents a novel goal reaching framework that exploits the deep connection between multi-goal RL and inverse RL to derive a family of goal-conditioned RL algorithms. By understanding hindsight goal relabeling from a divergence minimization perspective, our framework reveals the importance of reward design in multi-goal RL, which is found to significantly affect learning performance in our experiments. Furthermore, we propose an additional hindsight divergence minimization (HDM) loss, which uses Q-learning to account for the worst while using BC to imitate the best, and demonstrate its superior performance on discrete action space. In the future, we hope to further develop our framework to explicitly account for exploration.
# References
Abbeel, P. and Ng, A. Y. Apprenticeship learning via inverse reinforcement learning. In Proceedings of the twenty-first international conference on Machine learning, pp. 1, 2004. Ahn, M., Zhu, H., Hartikainen, K., Ponte, H., Gupta, A., Levine, S., and Kumar, V. Robel: Robotics benchmarks for learning with low-cost robots. In Conference on robot learning, pp. 1300–1313. PMLR, 2020. Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022. Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, P., and Zaremba, W. Hindsight experience replay. Advances in neural information processing systems, 30, 2017. Batra, D., Chang, A. X., Chernova, S., Davison, A. J., Deng, J., Koltun, V., Levine, S., Malik, J., Mordatch, I., Mottaghi, R., et al. Rearrangement: A challenge for embodied ai. arXiv preprint arXiv:2011.01975, 2020. Blier, L., Tallec, C., and Ollivier, Y. Learning successor states and goal-dependent values: A mathematical viewpoint. arXiv preprint arXiv:2101.07123, 2021. Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020. Chebotar, Y., Hausman, K., Lu, Y., Xiao, T., Kalashnikov, D., Varley, J., Irpan, A., Eysenbach, B., Julian, R., Finn, C., and Levine, S. Actionable models: Unsupervised offline reinforcement learning of robotic skills. arXiv preprint arXiv:2104.07749, 2021. Ding, Y., Florensa, C., Abbeel, P., and Phielipp, M. Goalconditioned imitation learning. In Advances in Neural Information Processing Systems, pp. 15298–15309, 2019. Duan, Y., Andrychowicz, M., Stadie, B., Jonathan Ho, O., Schneider, J., Sutskever, I., Abbeel, P., and Zaremba, W. One-shot imitation learning. Advances in neural information processing systems, 30, 2017.
Durugkar, I., Tec, M., Niekum, S., and Stone, P. Adversarial intrinsic motivation for reinforcement learning. Advances in Neural Information Processing Systems, 34: 8622–8636, 2021. Eysenbach, B., Geng, X., Levine, S., and Salakhutdinov, R. R. Rewriting history with inverse rl: Hindsight inference for policy improvement. Advances in neural information processing systems, 33:14783–14795, 2020a. Eysenbach, B., Salakhutdinov, R., and Levine, S. Clearning: Learning to achieve goals via recursive classification. arXiv preprint arXiv:2011.08909, 2020b. Eysenbach, B., Zhang, T., Salakhutdinov, R., and Levine, S. Contrastive learning as goal-conditioned reinforcement learning. arXiv preprint arXiv:2206.07568, 2022. Finn, C., Christiano, P., Abbeel, P., and Levine, S. A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models. arXiv preprint arXiv:1611.03852, 2016. Florensa, C., Held, D., Wulfmeier, M., Zhang, M., and Abbeel, P. Reverse curriculum generation for reinforcement learning. In Conference on robot learning, pp. 482– 495. PMLR, 2017. Fu, J., Luo, K., and Levine, S. Learning robust rewards with adversarial inverse reinforcement learning. arXiv preprint arXiv:1710.11248, 2017. Fujimoto, S., Hoof, H., and Meger, D. Addressing function approximation error in actor-critic methods. In International conference on machine learning, pp. 1587–1596. PMLR, 2018. Garg, D., Chakraborty, S., Cundy, C., Song, J., and Ermon, S. Iq-learn: Inverse soft-q learning for imitation. Advances in Neural Information Processing Systems, 34, 2021. Ghasemipour, S. K. S., Zemel, R., and Gu, S. A divergence minimization perspective on imitation learning methods. In Conference on Robot Learning, pp. 1259–1277. PMLR, 2020. Ghosh, D., Gupta, A., Fu, J., Reddy, A., Devin, C., Eysenbach, B., and Levine, S. Learning to reach goals without reinforcement learning. ArXiv, abs/1912.06088, 2019. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International
conference on machine learning, pp. 1861–1870. PMLR, 2018. Hinton, G. E. Training products of experts by minimizing contrastive divergence. Neural computation, 14(8):1771– 1800, 2002. Ho, J. and Ermon, S. Generative adversarial imitation learning. Advances in neural information processing systems, 29, 2016. Ho, J., Gupta, J., and Ermon, S. Model-free imitation learning with policy optimization. In International Conference on Machine Learning, pp. 2760–2769. PMLR, 2016. Hong, Z.-W., Yang, G., and Agrawal, P. Bilinear value networks. arXiv preprint arXiv:2204.13695, 2022. Jang, E., Irpan, A., Khansari, M., Kappler, D., Ebert, F., Lynch, C., Levine, S., and Finn, C. Bc-z: Zero-shot task generalization with robotic imitation learning. ArXiv, abs/2202.02005, 2021. Janner, M., Mordatch, I., and Levine, S. gamma-models: Generative temporal difference learning for infinitehorizon prediction. Advances in Neural Information Processing Systems, 33:1724–1735, 2020. Kaelbling, L. P. Learning to achieve goals. In IJCAI, volume 2, pp. 1094–8. Citeseer, 1993. Ke, L., Choudhury, S., Barnes, M., Sun, W., Lee, G., and Srinivasa, S. Imitation learning as f-divergence minimization. In Algorithmic Foundations of Robotics XIV: Proceedings of the Fourteenth Workshop on the Algorithmic Foundations of Robotics 14, pp. 313–329. Springer, 2021. Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. Kostrikov, I., Nachum, O., and Tompson, J. Imitation learning via off-policy distribution matching. arXiv preprint arXiv:1912.05032, 2019. Levine, S. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909, 2018. Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., and Wierstra, D. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015. Lin, X., Baweja, H. S., and Held, D. Reinforcement learning without ground-truth state. arXiv preprint arXiv:1905.07866, 2019.
Ke, L., Choudhury, S., Barnes, M., Sun, W., Lee, G., and Srinivasa, S. Imitation learning as f-divergence minimization. In Algorithmic Foundations of Robotics XIV: Proceedings of the Fourteenth Workshop on the Algorithmic Foundations of Robotics 14, pp. 313–329. Springer, 2021.
Lynch, C. and Sermanet, P. Language conditioned imitation learning over unstructured data. arXiv preprint arXiv:2005.07648, 2020. Lynch, C., Khansari, M., Xiao, T., Kumar, V., Tompson, J., Levine, S., and Sermanet, P. Learning latent plans from play. In CoRL, 2019. Ma, Y. J., Yan, J., Jayaraman, D., and Bastani, O. How far i’ll go: Offline goal-conditioned reinforcement learning via f-advantage regression. arXiv preprint arXiv:2206.03023, 2022. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. Human-level control through deep reinforcement learning. nature, 518(7540): 529–533, 2015. Nachum, O. and Dai, B. Reinforcement learning via fenchelrockafellar duality. arXiv preprint arXiv:2001.01866, 2020. Nachum, O., Chow, Y., Dai, B., and Li, L. Dualdice: Behavior-agnostic estimation of discounted stationary distribution corrections. Advances in Neural Information Processing Systems, 32, 2019a. Nachum, O., Dai, B., Kostrikov, I., Chow, Y., Li, L., and Schuurmans, D. Algaedice: Policy gradient from arbitrary experience. arXiv preprint arXiv:1912.02074, 2019b. Nair, A., McGrew, B., Andrychowicz, M., Zaremba, W., and Abbeel, P. Overcoming exploration in reinforcement learning with demonstrations. In 2018 IEEE international conference on robotics and automation (ICRA), pp. 6292– 6299. IEEE, 2018a. Nair, A. V., Pong, V., Dalal, M., Bahl, S., Lin, S., and Levine, S. Visual reinforcement learning with imagined goals. Advances in neural information processing systems, 31, 2018b. Ng, A. Y., Russell, S., et al. Algorithms for inverse reinforcement learning. In Icml, volume 1, pp. 2, 2000. Nguyen, X., Wainwright, M. J., and Jordan, M. I. Estimating divergence functionals and the likelihood ratio by convex risk minimization. IEEE Transactions on Information Theory, 56(11):5847–5861, 2010. Nowozin, S., Cseke, B., and Tomioka, R. f-gan: Training generative neural samplers using variational divergence minimization. Advances in neural information processing systems, 29, 2016.
Nguyen, X., Wainwright, M. J., and Jordan, M. I. Estimating divergence functionals and the likelihood ratio by convex risk minimization. IEEE Transactions on Information Theory, 56(11):5847–5861, 2010.
Oh, J., Guo, Y., Singh, S., and Lee, H. Self-imitation learning. In International Conference on Machine Learning, pp. 3878–3887. PMLR, 2018.
Oh, J., Guo, Y., Singh, S., and Lee, H. Self-imitation learning. In International Conference on Machine Learning, pp. 3878–3887. PMLR, 2018. OpenAI, O., Plappert, M., Sampedro, R., Xu, T., Akkaya, I., Kosaraju, V., Welinder, P., D’Sa, R., Petron, A., Pinto, H. P. d. O., et al. Asymmetric self-play for automatic goal discovery in robotic manipulation. arXiv preprint arXiv:2101.04882, 2021. Pitis, S., Chan, H., Zhao, S., Stadie, B., and Ba, J. Maximum entropy gain exploration for long horizon multi-goal reinforcement learning. In International Conference on Machine Learning, pp. 7750–7761. PMLR, 2020. Plappert, M., Andrychowicz, M., Ray, A., McGrew, B., Baker, B., Powell, G., Schneider, J., Tobin, J., Chociej, M., Welinder, P., et al. Multi-goal reinforcement learning: Challenging robotics environments and request for research. arXiv preprint arXiv:1802.09464, 2018a. Plappert, M., Andrychowicz, M., Ray, A., McGrew, B., Baker, B., Powell, G., Schneider, J., Tobin, J., Chociej, M., Welinder, P., et al. Multi-goal reinforcement learning: Challenging robotics environments and request for research. arXiv preprint arXiv:1802.09464, 2018b. Pomerleau, D. A. Alvinn: An autonomous land vehicle in a neural network. Advances in neural information processing systems, 1, 1988. Pong, V., Gu, S., Dalal, M., and Levine, S. Temporal difference models: Model-free deep rl for model-based control. arXiv preprint arXiv:1802.09081, 2018. Pong, V. H., Dalal, M., Lin, S., Nair, A., Bahl, S., and Levine, S. Skew-fit: State-covering self-supervised reinforcement learning. arXiv preprint arXiv:1903.03698, 2019. Reddy, S., Dragan, A. D., and Levine, S. Sqil: Imitation learning via reinforcement learning with sparse rewards. arXiv preprint arXiv:1905.11108, 2019. Ross, S., Gordon, G. J., and Bagnell, J. A. A reduction of imitation learning and structured prediction to no-regret online learning. In AISTATS, 2011. Rudner, T. G., Pong, V., McAllister, R., Gal, Y., and Levine, S. Outcome-driven reinforcement learning via variational inference. Advances in Neural Information Processing Systems, 34:13045–13058, 2021. Schulman, J., Chen, X., and Abbeel, P. Equivalence between policy gradients and soft q-learning. arXiv preprint arXiv:1704.06440, 2017a.
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017b. Shafiullah, N. M. M., Cui, Z. J., Altanzaya, A., and Pinto, L. Behavior transformers: Cloning k modes with one stone. arXiv preprint arXiv:2206.11251, 2022. Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., and Riedmiller, M. Deterministic policy gradient algorithms. In International conference on machine learning, pp. 387– 395. PMLR, 2014. Silver, D., Singh, S., Precup, D., and Sutton, R. S. Reward is enough. Artificial Intelligence, 299:103535, 2021. Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999. Tang, Y. and Kucukelbir, A. Hindsight expectation maximization for goal-conditioned reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pp. 2863–2871. PMLR, 2021. Van Hasselt, H., Guez, A., and Silver, D. Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence, volume 30, 2016. Vinyals, O., Babuschkin, I., Czarnecki, W. M., Mathieu, M., Dudzik, A., Chung, J., Choi, D. H., Powell, R., Ewalds, T., Georgiev, P., et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. Nature, 575 (7782):350–354, 2019. Warde-Farley, D., Van de Wiele, T., Kulkarni, T., Ionescu, C., Hansen, S., and Mnih, V. Unsupervised control through non-parametric discriminative rewards. arXiv preprint arXiv:1811.11359, 2018. Zhang, L., Yang, G., and Stadie, B. C. World model as a graph: Learning latent landmarks for planning. In International Conference on Machine Learning, pp. 12611– 12620. PMLR, 2021. Zhang, X., Li, Y., Zhang, Z., and Zhang, Z.-L. f-gail: Learning f-divergence for generative adversarial imitation learning. Advances in neural information processing systems, 33:12805–12815, 2020. Zhu, Z., Lin, K., Dai, B., and Zhou, J. Off-policy imitation learning from observations. Advances in Neural Information Processing Systems, 33:12402–12413, 2020. Ziebart, B. D., Maas, A. L., Bagnell, J. A., Dey, A. K., et al. Maximum entropy inverse reinforcement learning. In Aaai, volume 8, pp. 1433–1438. Chicago, IL, USA, 2008.
A. Proofs
A.1. Deriving Hindsight Behavior Cloning We want to prove the equivalence defined in (10)
We want to prove the equivalence defined in (10)
We use the definition of KL divergence:
And the graphical models of the two distributions:
pµ(s, a, s+) = ρµ(s, a)p+ µ (s+ | s, a) pBC π (s, a, g) = p(g)ρµ(s)π(a | s, g)
Resulting in
DKL(pµ(s, a, s+) ∥pBC π (s, a, g)) = Epµ(s,a,s+)[log pµ(s, a, s+) −log pBC π (s, a, g)] = Epµ(s,a,s+)[log pµ(s, a, s+) −log p(g) −log ρµ(s) −log π(a | s, g � �
DKL(pµ(s, a, s+) ∥pBC π (s, a, g)) = Epµ(s,a,s+)[log pµ(s, a, s+) −log pBC π (s, a, g)] = Epµ(s,a,s+)[log pµ(s, a, s+) −log p(g) −log ρµ(s) −log π(a | s, g) = Eρµ(s,a)p+ µ (s+|s,a) � log pµ(s, a, s+) p(g)ρµ(s) −log π(a | s, g) �
where we find that
arg min π DKL(pµ(s, a, s+) ∥pBC π (s, a, g)) = arg min π Eρµ(s,a)p+ µ (s+|s,a)[−log π(a | s, g)]
# A.2. Main Lemmas
Lemma A.1 (Online-to-offline transformation for goal reaching). Given a goal-conditioned policy π(a | s, g), its corresponding Q-function Qπ(s, a, g), and arbitrary state-action visitation distribution ρµ(s, a) of another policy µ(a | s), the expected temporal difference for online rollouts under π is: Ep(g)ρπ(s,a|g)[(Qπ −γ · PπQπ)(s, a, g)] = Ep(g)ρµ(s,a)π(˜a|s,g)[Qπ(s, ˜a, g) −γ · PπQπ(s, a, g)]
 | Ep(g)ρπ(s,a|g)[(Qπ −γ · PπQπ)(s, a, g)] = Ep(g)ρµ(s,a)π(˜a|s,g)[Qπ(s, ˜a, g) −γ · PπQπ(s, a, g)]
Proof of Lemma 4.1.
= (1 −γ) ∞ � t=0 γtEp(g)ρt µ(s,a) π(˜a|s,g) [Qπ(s, ˜a, g) −γE p(s′|s,a) π(a′|s′,g) Qπ(s′, a′, g)] = Ep(g)ρµ(s,a)π(˜a|s,g)[Qπ(s, ˜a, g) −γEp(s′|s,a),π(a′|s′,g)Qπ(s′, a′, g)]
� | = Ep(g)ρµ(s,a)π(˜a|s,g)[Qπ(s, ˜a, g) −γEp(s′|s,a),π(a′|s′,g)Qπ(s′, a′, g)]
A.3. Q Function as Step Counts
We make the following definitions:
Qt = Q(st, at, g) Qt+1 = Eπ(st+1,g)[Q(st+1, a′, g)] βt = p(st+1 = g | st, at) α(i) t = � ∆=i γ∆pπ(st+1+∆= g | st, at)
And make the following mild assumptions:
1/Eπ(a|st,g)[α(0) t ] = Eπ(a|st,g)[1/α(0) t ] 1/Ep(st+1|st,at)[α(1) t+1] = Ep(st+1|st,at)[1/α(1) t ]
Which can be combined to reach the result:
1/α(1) t = Ep(st+1|st,at)π(a′ t+1|st+1,g)[1/α(1) t+1]
Meaning that for an infinite horizon MDP where all goals are eventually reached, the reciprocal of a geometrically summed future likelihood of reaching a goal remains approximately the same in expectation under rollouts. Lemma A.2 (Recursive estimate of goal reaching step count). Under the notations and the assumptions above, the step count definition (16) has the following property:
Qt = Ep(st+1|st,at)[(1 − βt βt + α(1) t )(−1 + Qt+1)]
Proof of Lemma 4.2. We start from the definition of our Q value:
Q(st, at, g) = − � ∆=0 γ∆pπ(st+1+∆= g | st, at) · ∆ � ∆=0 γ∆pπ(st+1+∆= g | st, at)
Qπ(st, at, g) can be expanded as
− γp(st+2 = g | st, at) + 2γ2p(st+3 = g | st, at) + 3γ3p(st+4 = g | st, at) + · · · p(st+1 = g | st, at) + γp(st+2 = g | st, at) + γ2p(st+3 = g | st, at) + γ3p(st+4 = g | st, at) + · · · he value function V (st, g) is defined to be:
The value function V (st, g) is defined to be:
V (st, g) = − � ∆=0 γ∆pπ(st+1+∆= g | st) · ∆ � ∆=0 γ∆pπ(st+1+∆= g | st)
Which can be expanded as:
− γp(st+2 = g | st) + 2γ2p(st+3 = g | st) + 3γ3p(st+4 = g | st) + · · · p(st+1 = g | st) + γp(st+2 = g | st) + γ2p(st+3 = g | st) + γ3p(st+4 = g | st) + · · ·
On the other hand, Eπ(a|st,g)[Q(st, a, g)] can be written as:
−Eπ(a|st,g) � γp(st+2 = g | st, a) + 2γ2p(st+3 = g | st, a) + 3γ3p(st+4 = g | st, a) + · · · p(st+1 = g | st, a) + γp(st+2 = g | st, a) + γ2p(st+3 = g | st, a) + γ3p(st+4 = g | st, a) + · · · by the law of total probability, p(st+1+∆= g | st) = Eπ(a|st,g)[p(st+1+∆= g | st, at)]
by the law of total probability,
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/45da/45da87fe-0b37-42be-884e-f1198c3b23bc.png" style="width: 50%;"></div>
� Furthermore, V π(st+1, g) can be expanded as:
Note that, for ∆> 0, by the law of total probability,
As a result:
Ep(st+1|st,at)[−1 + V π(st+1, g)] = −Z[γp(st+2 = g | st, at) + 2γ2p(st+3 = g | st, at) + 3γ3p(st+4 = g | st, at) + · · · 
Ep(st+1|st,at)[−1 + V π(st+1, g)] = −Z[γp(st+2 = g | st, at) + 2γ2p(st+3 = g | st, at) + 3γ3p(st+4 = g | st, at where
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/676c/676c4c4c-169e-42d8-ac95-5b56e0e340fa.png" style="width: 50%;"></div>
Resulting in:
we have:
Which eventually leads to:
(22)
(23)
# A.4. Deriving HER Rewards
A.4. Deriving HER Rewards
Lemma A.3 (Understanding Hindsight Experience Replay). Multi-goal Q-learning with HER reward (5) with {−1, 0} special case of minimizing the following objective:
with β = (1 −γ) and the convex function f ∗chosen to be:
f ∗(x) = (x −1)2/2 + 3/2
p+ µ (s+ | s, a) = (1 −γ)p(s+ | s, a) + γ � S×A p(s′ | s, a)µ(a′ | s′)p+ µ (s+ | s′, a′)ds′da′
And that we have defined a quadratic form of f ∗(with c being constants): f ∗(x) = (x −1)2/2 + c
ng the dynamics to expand the expectation and applying the choice of f ∗being a quadratic, the loss becomes:
arg min Q Eρµ(s,a)p(s′|s,a)(1 −γ) · �1 2 � −1 + (γPπQ −Qθ)(s, a, s′) �2 −·Qθ(s, a, s′) �
� � � + Eρµ(s,a)p(s′|s,a)µ(a′|s′)p+ µ (s+|s′,a′) � γ · 1 2 � −1 + (γPπQ −Qθ)(s, a, s+) �2
� � �� Assuming that there is a stop gradient sign on PπQ because of the use of a target network (Mnih et al., 2015; Lillicrap et al., 2015; Haarnoja et al., 2018), we can rewrite the above as one single quadratic and see that the gradient of the above loss w.r.t Q is equivalent to the gradient of the following squared Bellman residual:
arg min Q Eρµ(s,a)p(s′|s,a)p+ µ (s+|s,a) �1 2 � r(s, a, s′, s+) + (γPπQ −Qθ)(s, a, s+) �2�
where the reward function r(s, a, s′, s+) is:
The constant 3/2 in f ∗is chosen to ensure that ((f ∗)∗)(1) = 0 in the definition of f-divergence (6), but it does not affec the optimization process.
# B. Experimental Details
For the reward design experiments, we use the following hyper-parameters in Table 2, which are mostly the same from prior open-sourced implementations (Andrychowicz et al., 2017; Pitis et al., 2020; Zhang et al., 2021). For discrete action space experiments, we use the following thresholds (of Euclidean norms) for determining success: [0.08, 0.08, 0.05, 0.05, 0.1], which are tight thresholds based on our visualizations of the environments (those are tighter thresholds than the original ones in (Ghosh et al., 2019); based on our observations, the original thresholds are often too loose). We use the same network architecture, sampling and optimization schedules for all the methods, as described in Table 3. As for γHDM, we set it to be 0.85 in Four Rooms and Lunar Lander, 0.5 in Sawyer Push and Claw Manipulate, and 0.4 for Door Opening. Ablation on this hyper-parameter can be found in Figure 10.

<div style="text-align: center;">Table 2: Hyper-parameters for the goal reaching reward design experiments</div>
Parameter
Value
DDPG (Lillicrap et al., 2015)
optimizer
Adam (Kingma & Ba, 2014)
architecture
MLP + BVN (Hong et al., 2022)
number of hidden layers (all networks)
2
number of hidden units per layer
256
nonlinearity
ReLU
Normalize per-dimension obs (Schulman et al., 2017b)
yes
polyak for target network (τ)
0.995
target network update interval
10
Use target network for policy (Fujimoto et al., 2018)
yes
ratio between environment vs optimization steps
2
Random action probability
0.2
Initial random trajectories per worker
100
Hindsight relabelling ratio
0.85
Learning rate
0.001
Batch size
1024
Gamma factor γ
0.99
Action L2 regularization
0.01
Gaussian noise scale
0.1
Number of parallel workers
12
Replay buffer size
2500000
<div style="text-align: center;">Table 3: Hyper-parameters for discrete action space experiments</div>
Parameters
Value
DDQN (Van Hasselt et al., 2016)
Optimizer
Adam (Kingma & Ba, 2014)
Number of hidden layers (all networks)
2
Number of hidden units per layer
[400, 300] (Fujimoto et al., 2018)
Non-linearity
ReLU
Polyak for target network
0.995
Target update interval
10
Ratio between env vs optimization steps
1
Initial random trajectories
200
Hindsight relabelling ratio
0.85
Update every # of steps in environment
50
Next state relabelling ratio
0.2
Learning rate
5.e-4
BC loss weight
1.0
Gamma factor γ
0.98
Logit temperature for SQL (Schulman et al., 2017a)
0.2
Batch size
256
Epsilon greedy (Mnih et al., 2015)
0.2
Replay buffer size
2500000
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/226a/226a147c-c0c7-4291-bcac-c607f0adb363.png" style="width: 50%;"></div>
Figure 10: Ablation studies on HDM Gamma γhdm (see equation (18)). The orange line and the blue line denote HER (Andrychowicz et al., 2017) and GCSL (Ghosh et al., 2019) baseline performance. Intuitively, γhdm controls the threshold for when an action is considered good enough for imitation. As we lower γhdm, the threshold −log γhdm gets higher and fewer actions get imitated, with the remaining imitated actions more concentrated around the goal, where hindsight-relabeled actions are more likely to be optimal. The ablation shows that HDM outperforms HER and GCSL across a variety of γhdm, while recovering the performance of HER when its value is close to zero. Both environments have discrete action space where we use a softmax (Boltzmann) policy (Schulman et al., 2017a) with the softmax logits being Q-values.
# C. Further Discussions on Achieved Goal (ag) Change Ratio
We define the ag change ratio of π to be: the percentage of trajectories where the achieved goals in initial states s0 are different from the achieved goals in final states sT under π. Using notation from (5), it can be computed as Es0···sT ∼π[−rHER(·, ·, s0, sT )]. We then define initial ag change ratio to be the ag change ratio of a random-acting policy π0. Using notation from (5), it can be computed as Es0···sT ∼π0[−rHER(·, ·, s0, sT )]. In our experiments, we have made the following observations:
• ag change ratio is strongly indicative of learning progress on many environments, as shown in Figure 8. • initial ag change ratio, which can be computed without training any policies, seems to be correlated to the f performance of HBC / GCSL, as shown in Figure 11 (bubble sizes are based on the variances of the success rate).
• ag change ratio is strongly indicative of learning progress on many environments, as shown in Figure 8. • initial ag change ratio, which can be computed without training any policies, seems to be correlated to the final performance of HBC / GCSL, as shown in Figure 11 (bubble sizes are based on the variances of the success rate). We suspect that the reasons are the following:
We suspect that the reasons are the following:
• Most training signals are only created from ag changes, because they provide examples of how to rearrange an environment. As the policy learns to rearrange the environment with higher frequency, hindsight relabeling ensures that the learning progress naturally accelerates. • One likely reason why the performance of GCSL seems to be upper-bounded by a linear relationship between the final success rate and the initial ag change ratio is that: a BC-style objective starts off cloning the initially random trajectories, so if initial ag change ratio is low, the policy would not learn to rearrange ag from self-imitation, compounding to a low final performance. HER and HDM are able to surpass this upper ceiling likely because they try to reach goals with fewer steps besides imitation.
• Most training signals are only created from ag changes, because they provide examples of how to rearrange an environment. As the policy learns to rearrange the environment with higher frequency, hindsight relabeling ensures that the learning progress naturally accelerates.
 One likely reason why the performance of GCSL seems to be upper-bounded by a linear relationship between the final success rate and the initial ag change ratio is that: a BC-style objective starts off cloning the initially random trajectories, so if initial ag change ratio is low, the policy would not learn to rearrange ag from self-imitation, compounding to a low final performance. HER and HDM are able to surpass this upper ceiling likely because they try to reach goals with fewer steps besides imitation.
• This finding suggests that in order to make goal-reaching easier, we should either modify the initial state distribution ρ0(s) such that ag can be easily changed through random exploration (Florensa et al., 2017) (if the policy is training from scratch), or initialize BC from some high-quality demonstrations where ag does change (Ding et al., 2019; Nair et al., 2018a; Lynch et al., 2019).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c3ee/c3eefcca-54bb-4b46-9a4d-b96e1a9adf4d.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 11: Success rate versus initial ag change ratio.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5b02/5b02ad89-b1b4-49d2-9f13-c57292e4a803.png" style="width: 50%;"></div>
Figure 12: Even on an environment where the initial ag change ratio is 1 and where the ag change ratio stays at around 1 throughout learning for all methods, Actionable Models (AM) (Chebotar et al., 2021) and HER with {0, 1} rewards still fail to learn anything, where HER with {−1, 0} rewards (Andrychowicz et al., 2017) succeed at learning a goal-conditioned policy. All hyper-parameters are the same except for the reward design and bellman backups for all three methods. Results are averaged over 5 random seeds, and the variances across seeds are plotted. This shows that our conclusions in Section 6.1 are independent of exploration difficulties, and that the design choices for goal-reaching rewards matter significantly. The above insights on the possible relationship between ag change ratio and exploration difficulty makes us wonder whether the striking results about reward design effectiveness in Section 6.1 are independent of the exploration problem. Indeed, Figure 8 shows that HER with {−1, 0} rewards takes off faster partially because it manages to increase its ag change ratio faster. To show that our conclusions about goal-reaching reward design are independent of exploration difficulties, we benchmark the reward-design results on an environment where all methods have an initial ag change ratio of 1 (and where the ag change ratio stays at around 1 throughout learning for all methods), namely the Shallow Hand environment (Plappert et al., 2018a) HandManipulateBlockRotateZ. We use the same set of hyper-parameters as Table 2 except for the fact that we use 20 parallel workers for the Hand environments rather than 12, since this environment is harder. The results are presented in Figure 12, showing that HER with {−1, 0} rewards is the only method that learns and succeeds. These findings are largely consistent with the conjecture hypothesized in the original Hindsight Experience Replay (HER) paper, which discussed a similar problem in the context of a simple bit-flipping experiment (See (Andrychowicz et al., 2017) Section 3.1).
<div style="text-align: center;">Figure 12: Even on an environment where the initial ag change ratio is 1 and where the ag change ratio stays at around 1 throughout learning for all methods, Actionable Models (AM) (Chebotar et al., 2021) and HER with {0, 1} rewards still fail to learn anything, where HER with {−1, 0} rewards (Andrychowicz et al., 2017) succeed at learning a goal-conditioned policy. All hyper-parameters are the same except for the reward design and bellman backups for all three methods. Results are averaged over 5 random seeds, and the variances across seeds are plotted. This shows that our conclusions in Section 6.1 are independent of exploration difficulties, and that the design choices for goal-reaching rewards matter significantly.</div>
The above insights on the possible relationship between ag change ratio and exploration difficulty makes us wonder whether the striking results about reward design effectiveness in Section 6.1 are independent of the exploration problem. Indeed, Figure 8 shows that HER with {−1, 0} rewards takes off faster partially because it manages to increase its ag change ratio faster. To show that our conclusions about goal-reaching reward design are independent of exploration difficulties, we benchmark the reward-design results on an environment where all methods have an initial ag change ratio of 1 (and where the ag change ratio stays at around 1 throughout learning for all methods), namely the Shallow Hand environment (Plappert et al., 2018a) HandManipulateBlockRotateZ. We use the same set of hyper-parameters as Table 2 except for the fact that we use 20 parallel workers for the Hand environments rather than 12, since this environment is harder. The results are presented in Figure 12, showing that HER with {−1, 0} rewards is the only method that learns and succeeds. These findings are largely consistent with the conjecture hypothesized in the original Hindsight Experience Replay (HER) paper, which discussed a similar problem in the context of a simple bit-flipping experiment (See (Andrychowicz et al., 2017) Section 3.1).
