# 1–2–3–Go! Policy Synthesis for Parameterized Markov Decision Processes via Decision-Tree Learning and Generalization⋆
Muqsit Azeem1, Debraj Chakraborty3, Sudeep Kanav3, Jan Křetínský1,3, Mohammadsadegh Mohagheghi2, Stefanie Mohr1, and Maximilian Weininger4
1 Technical University of Munich, Munich, Germany 2 Vali-e-Asr University of Rafsanjan, Rafsanjan, Iran 3 Masaryk University, Brno, Czech Republic 4 Institute of Science and Technology, Vienna, Austria
Abstract. Despite the advances in probabilistic model checking, the scalability of the verification methods remains limited. In particular, the state space often becomes extremely large when instantiating parameterized Markov decision processes (MDPs) even with moderate values. Synthesizing policies for such huge MDPs is beyond the reach of available tools. We propose a learning-based approach to obtain a reasonable policy for such huge MDPs. The idea is to generalize optimal policies obtained by model-checking small instances to larger ones using decision-tree learning. Consequently, our method bypasses the need for explicit state-space exploration of large models, providing a practical solution to the state-space explosion problem. We demonstrate the efficacy of our approach by performing extensive experimentation on the relevant models from the quantitative verification benchmark set. The experimental results indicate that our policies perform well, even when the size of the model is orders of magnitude beyond the reach of state-of-the-art analysis tools.
Keywords: model checking · probabilistic verification · Markov decision process · policy synthesis.
# 1 Introduction
Markov decision processes (MDPs) are the model for combining probabilistic uncertainty and non-determinism. MDPs come with a rich theory and algorithmics developed over several decades with mature verification tools arising 20 years ago [30] and proliferating since then [10]. Despite all this effort, the scalability of the methods is considerably worse than of those used for verification of non-deterministic systems with no probabilities, even for basic problems.
⋆This research was funded in part by the German Research Foundation (DFG) project 427755713 GOPro, the DFG GRK 2428 (ConVeY), and the MUNI Award in Science and Humanities (MUNI/I/1757/2021) of the Grant Agency of Masaryk University.
⋆This research was funded in part by the German Research Foundation (DFG) project 427755713 GOPro, the DFG GRK 2428 (ConVeY), and the MUNI Award in Science and Humanities (MUNI/I/1757/2021) of the Grant Agency of Masaryk University.
What to do about very large models? Researchers have made various attempts to tackle this issue, however, only with limited success. Firstly, prominent techniques which work well in non-probabilistic verification, such as symbolic techniques [30], abstraction [31], and symmetry reduction [17], are harder to apply efficiently in the probabilistic setting. Secondly, “engineering” improvements, such as the use of external storage [21] or parallelization, help by a significant, but principally very limited factor. Thirdly, there is a relaxation of the guarantees on the precision and/or certainty of the result, which we describe in detail below. The result of the analysis is typically a number (called the value), such as the expected reward or the probability to reach a given state, maximized (or minimized) over all resolutions of non-deterministic choices (called policies, strategies, schedulers, adversaries, or controllers in different applications). It is generally accepted that the precise number is not needed and an approximation is sufficient in most settings. Interestingly, until a few years ago [18, 7, 4], typically only the under-approximations were computed for the fundamental (maximization) problems, with no reliable over-approximations (with dual issues for minimization). It is worth noting that over-approximating is inherently harder since reasoning that the value cannot be greater than x involves the claim that all policies induce a lower value. In contrast to this universal quantification, the existential one is sufficient for under-approximating: upon providing a policy, its value forms automatically a lower bound, which is typically easier to compute. Consequently, many best-effort approaches, such as reinforcement learning (RL) [46] and lightweight statistical model checking [9] simply try to find a good policy while giving only empirically good chances to be close to the optimum. This is sufficient in the setting of (i) policy synthesis, where a “good enough” (close to optimum), but not necessarily optimal, controller is sought, or (ii) bug hunting and falsification, where finding significant counter-examples cheaply is desirable. However, the cal quality of the results relies on certain assumptions of these methods: RL results suffer when the rewards in the model are sparse (e.g., in the case of reachability) and lightweight statistical model checking suffers when near-optimal policies are not abundant. To summarize, synthesizing practically good policies is sufficient in many settings and also the only way when the systems are too large. Yet, when the system is extremely large, the available techniques either run out of resources or yield policies that are far from optimum (and close to random). Examining the structure of large MDPs in standard benchmark sets, e.g. [22], reveals that their huge sizes are typically not due to astronomically large humanwritten code, but rather because the MDPs are parameterized (e.g., by the number of participants in a protocol) and then instantiated with large values. Accordingly, this paper proposes a new approach to scalable policy synthesis for parameterized MDPs. Namely, it produces good policies for arbitrarily large instantiations of parameterized MDPs in particular those beyond the reach of any state-of-the-art tools. We focus on probabilistic reachability
(i) for simplicity and (ii) because it is a fundamental building block for many other problems.
Our main idea is to generalize the decisions taken by the optimal policies for the smaller instances: Instead of investigating a huge MDP, we synthesize optimal policies for the given parameterized MDP by instantiating it with small numbers. We then generalize this information and learn a policy that can be applied to any instantiation with an arbitrarily larger number. It is important to note that we generalize the corresponding decisions (i.e., the policy itself), not the values of the states across different parameterizations. Indeed, while the numeric values can differ vastly, the optimal behaviour is often similar in all instantiations. In order to capture this regularity, we thus need a symbolic representation of a policy, which applies to all instantiations. Decision trees (DT) can provide such a representation. Moreover, since they can represent policies explainably [6, 2] and capture the essence of the decisions, not just a list of state-action pairs, they generalize well. As an illustrative task for our “generalization”, consider a buggy mutual exclusion protocol with a high number of participants. While finding the bug with many participants may be hard, an exhaustive investigation of the case with two participants may reveal a scenario violating the exclusion. A similar scenario can then also happen with many participants where the choices of the remaining participants may be irrelevant. Consequently, the key decisions in the policy to find bug with two participant may also be used for finding the bug with multiple participants.
Our contribution can be summarized as follows:
# Our contribution can be summarized as follows:
– We provide a simple and elegant way of computing practically good policies for parameterized MDPs of any size (as long as some instantiations exist that are small enough so that some technique can be applied), in particular also orders of magnitude beyond the reach of any other methods. The method is based on generalizing5 policies via their decision-tree representations. The method scales constantly in the parameter instantiation since it applies available techniques to a fixed number of small base instantiations, and the large instantiation is never explicitly considered. – We demonstrate the efficacy of the method experimentally on standard benchmarks. In particular, from the practical perspective, we observe that our policies mostly achieve values that are close to the actual optimum. Note that, this in principle cannot be guaranteed for instantiations too large for precise methods to apply, which are exactly of our interest. Nevertheless, the often consistent results on the smaller instantiations convincingly substantiate the expectation that the policies perform well also for the large instantiations.
5 The nature of our generalization-based policy synthesis is also portrayed by ou quipping title “1–2–3–Go!”: Find out what works for cases 1, 2, and 3, then “Go! and apply it for arbitrary large values of the parameters.
– Finally, comparing to the benchmarks where our policies do not perform so well, we identify aspects of the models indicating where our heuristic generalizes well and where either more tailored or completely different techniques are required. It should be emphasized that we regard the simplicity of our approach rather as an advantage, making it easy to exploit. While there is a body of work on policy representation (via post-processing them), the use of DT to compute policies is very limited (as described in the Related Work below) and, to the best of our knowledge, non-existent for computing/generalizing them for arbitrarily large systems. Altogether, this simple, yet efficient idea deserves to be finally explored.
– Finally, comparing to the benchmarks where our policies do not perform so well, we identify aspects of the models indicating where our heuristic gener alizes well and where either more tailored or completely different technique are required.
It should be emphasized that we regard the simplicity of our approach rather as an advantage, making it easy to exploit. While there is a body of work on policy representation (via post-processing them), the use of DT to compute policies is very limited (as described in the Related Work below) and, to the best of our knowledge, non-existent for computing/generalizing them for arbitrarily large systems. Altogether, this simple, yet efficient idea deserves to be finally explored.
# Related work
Symbolic approaches are widely used as for alleviating the challenges of the state explosion problem [3]. These approaches are based on data structures storing the information of a model compactly. In particular, the multi-terminal version of BDDs (MTBDDs) has been developed for probabilistic model checking [28, 37, 40]. In a sense, our approach is also symbolic, since we represent the policy using a decision tree. This data structure is most suitable for the goal of explainability, as argued in, e.g., [2]. Reduction techniques try to reduce the state space of the model while the smaller model satisfies the same set of properties. A symmetry reduction technique for probabilistic models has been proposed in [32] for systems with several symmetric components. Probabilistic bisimulation is available for MDPs and discrete-time Markov chains (DTMCs) that reduce the original model to the smallest one that satisfies the same set of temporal logic formulae [15]. Considering a subset of temporal logic formulae, more efficient techniques have been proposed in [27] for reducing the model to a smaller one. Applying reduction on a high-level description before constructing the resulting model is available in [45, 36]. Further techniques improving scalability of traditional algorithms include the following. Using secondary storage in an efficient way to keep a sparse representation of a large model has been studied in [21]. Compositional techniques have been developed for the verification of stochastic systems [13, 35]. Prioritizing computation can reduce running times in many case studies by using topological state ordering [34, 11] or learnt prioritizing [7, 39, 29]. All the above techniques help solving larger models, however, only up to a certain limit. In contrast, our approach synthesizes policy that can be applied to arbitrarily large instances. Statistical model checking (SMC) is an alternative solution for approximating the quantitative properties [24, 7, 23] by running a set of simulations on the model to approximate the requested values, while providing a confidence interval for the precision of computed values for discrete and continuous-time
Markov chains (DTMCs and CTMCs). This is scalable since the number of samples does not depend on the size of the model. Still, the length of the simulations does. Using SMC for MDPs faces the difficulty of resolving non-determinism. A smart-sampling method has been proposed in [12] that considers a set of random policies, with some of them hopefully approximating the optimal one; however, this method cannot generally provide a confidence interval for the precision of computations [23]. Machine learning within formal verification of MDP has been widely studied for a decade since the seminal [24]. An L∗learning approach has been developed in [47] to learn an MDP model efficiently. Neural networks and regression can be used to resolve non-determinism of large MDPs and provide the opportunity of applying SMC for this class of models [43, 16]. Reinforcement learning has also been adapted to the setting of verification with objectives such as reachability [7, 19], but the sparsity of the rewards is still an issue affecting the scalability. Still, prioritizing the subset of the states that has the biggest impact on the value can allow for verifying huge models if such a subset is small and easy to find [7, 29]. Unlike reinforcement learning (RL), which suggests policies for specific models through random exploration, our approach generalizes policies for various instances by computing optimal policies on smaller models and generalizing them. Pyeatt and Howe [42] propose using decision trees to approximate value function for discounted rewards in reinforcement learning. However, we learn a DT that is a valid policy for any instantiations of the parameterized MDP, whereas the DT learned in [42] is applicable only to the model under consideration. Decision trees have been used as a data structure for representing MDP policies [2, 6]. Interestingly, while binary decision diagrams (BDD) may appear to the verification community as a suitable candidate, it has been shown that DT are more appropriate if adequately used [6, 2, 39] due to their ability to handle various predicates and complex relationships, enhancing explainability. Another advantage of DTs is the ability to declare some inputs as uninteresting (“don’t care” inputs), saving on size via semantics of the controller.
# 2 Preliminaries
We provide basic definitions in Section 2.1, then describe what it means for models to be parameterized and scalable in Section 2.2 and finally recall how decision trees can be used for representing policies in Section 2.3.
# 2.1 Markov decision processes with a reachability objective
A probability distribution over a discrete set X is a function µ : X →[0, 1] where � x∈X µ(X) = 1. We denote the set of all probability distributions over X by
A probability distribution over a discrete set X is a function µ : X →[0, 1] where � x∈X µ(X) = 1. We denote the set of all probability distributions over X by D(X).
A probability distribution over a discrete set X is a function µ : X →[0, 1] where � x∈X µ(X) = 1. We denote the set of all probability distributions over X by D(X).
Definition 1. A (finite) Markov Decision Process (MDP) is a tuple M = (S, A, δ, ¯s, G) where S is a finite set of states, A is a finite set of actions, overloading A(s) to denote the (non-empty) set of enabled actions for every state s ∈S, δ : S × A →D(S) is a probabilistic transition function mapping each state s ∈S and enabled action a ∈A(s) to a probability distribution over successor states, ¯s ∈S is the initial state, and G ⊆S is the set of goal states.
A Markov chain (MC) can be seen as an MDP where |A(s)| = 1 for all s ∈S, i.e. a system exhibiting only probabilistic behavior, but no non-determinism. The semantics of an MDP are defined in the usual way by means of policies and paths in the induced Markov chain. An infinite path ρ = s1s2 . . . ∈Sω is an infinite sequence of states. A policy is a function σ : S →D(A) that, intuitively, prescribes for every state which action to play. The policy is called deterministic if in every state it selects a single action surely, otherwise it is randomized. Note that we limit our definition of policies w.l.o.g. to those that are memoryless (history-independent), i.e. those that depend only on the current state, not on a whole path. We denote the i-th state on a path as ρi, the set of all paths as Paths, and the set of all policies as Σ. By using a policy σ to resolve all nondeterministic choices in an MDP M, we obtain a Markov chain Mσ [3, Definition 10.92]. This Markov chain induces a unique probability measure Pσ s over paths starting in state s [3, Definition 10.10]. The reachability objective is included in our definition of MDP in the form of the initial state ¯s and the set of goal states G. Intuitively, the value V of a reachability objective is the optimal probability to reach some goal state when starting in the initial state; formally V(M) = optσ∈ΣPσ ¯s [♢G], where opt ∈ {max, min} indicates whether we are trying to reach or avoid the set of goal states and ♢G = {ρ ∈Paths | ∃i.ρi ∈G} denotes the set of all paths that reach a goal state. One can restrict this optimum to the deterministic memoryless policies [41, Proposition 6.2.1].
# 2.2 State space structure and scalable models
For learning (e.g. of DTs) to be effective, it is important that the state space of MDP is structured, i.e. every state is a tuple of values of state variables. In other words, the state space of the system is not monolithic (e.g. states defined by a simple numbering), but in fact, there are multiple factors defining it, for example, time or protocol state. Each of these factors is represented by a state variable vi with domain Di. Thus, every state s ∈S is in fact a tuple (v1, v2, . . . , vn), where each vi ∈Di is the value of a state variable. A parameterized MDP can be described as a variant of standard MDP where certain parameters are not fixed constants but instead can take different values within a parameter space. These parameters can be associated to the state-space of the system (for example, lower or upper bound of a state variable) or transition dynamics (the probabilities can be functions of the parameter). We provide the formal definition below.
– Θ is the parameter space, – For each θ ∈Θ, the tuple (Sθ, Aθ, δθ, ¯sθ, Gθ) defines an MDP instance, where: • Sθ is the set of states, • Aθ is the set of actions, • δθ : Sθ × Aθ →D(Sθ) is the probabilistic transition function, • ¯sθ ∈Sθ is the initial state, • Gθ ⊆Sθ is the set of goal states.
In this framework, different parameter values θ ∈Θ yield different instances of the MDP, and the parameterization can affect the state space, transition dynamics, or both.
Intuitively, a parameterized MDP can be seen as a family of MDPs where different value of parameter gives different instance of the MDP. In particular, this typically makes the models scalable: by increasing the values of parameters in the model description, one can scale up the size of the state space of the model. MDPs in the PRISM benchmark suite [33] and the quantitative verification benchmark set [22] have these properties of being structured and scalable. Example 1. Consider the MDP in Figure 1. Every state is a tuple (m, x) of two state variables m and x with domains Dm = {0, 1, . . . , k} and Dx = {0, 1, 2}. The state variable m indicates which of the k + 1 blocks we are in, while the state variable x indicates the position inside a block. The block with m = 0 is special: it contains the initial state (0,0), the goal state (0,2), and a sink state (0,1) which cannot reach the goal. All other blocks look as follows: for every m ∈[1, k], the x = 0 state can choose to continue to x = 1 (action a) or self-loop (action b). For every m ∈[1, k −1], the x = 1 state can go back to x = 0 in the same block (action a) or leave the block (action b). When using b, there is a 50% chance of going to the sink state (0, 1) and a 50% chance to continue to x = 0 in the (m+1)-th block. In the k-th block, the action leaving the block progresses to the goal state. The model is scalable, since the number of blocks k can be increased arbitrarily. This affects both the size of the state space, which is 2k + 3, as well as the maximum reachability probability, which is 0.5k−1. We assume the MDPs are defined using high-level modeling languages such as Probmela [3], PRISM [33] or MODEST [20]. In such modeling languages, MDPs are often represented as a composition of multiple identical components, called modules. For example, in a distributed system where multiple processes are interacting or sharing common resources, each process can be described as a separate module. In the context of this paper, we also consider the number of modules as a parameter. Example 2 (Dining philosophers). The dining philosophers problem involves a number of philosophers, seated around a circular table, blessed with infinite
Intuitively, a parameterized MDP can be seen as a family of MDPs where different value of parameter gives different instance of the MDP. In particular, this typically makes the models scalable: by increasing the values of parameters in the model description, one can scale up the size of the state space of the model. MDPs in the PRISM benchmark suite [33] and the quantitative verification benchmark set [22] have these properties of being structured and scalable. Example 1. Consider the MDP in Figure 1. Every state is a tuple (m, x) of two state variables m and x with domains Dm = {0, 1, . . . , k} and Dx = {0, 1, 2}. The state variable m indicates which of the k + 1 blocks we are in, while the state variable x indicates the position inside a block. The block with m = 0 is special: it contains the initial state (0,0), the goal state (0,2), and a sink state (0,1) which cannot reach the goal. All other blocks look as follows: for every m ∈[1, k], the x = 0 state can choose to continue to x = 1 (action a) or self-loop (action b). For every m ∈[1, k −1], the x = 1 state can go back to x = 0 in the same block (action a) or leave the block (action b). When using b, there is a 50% chance of going to the sink state (0, 1) and a 50% chance to continue to x = 0 in the (m+1)-th block. In the k-th block, the action leaving the block progresses to the goal state. The model is scalable, since the number of blocks k can be increased arbitrarily. This affects both the size of the state space, which is 2k + 3, as well as the maximum reachability probability, which is 0.5k−1.
We assume the MDPs are defined using high-level modeling languages such as Probmela [3], PRISM [33] or MODEST [20]. In such modeling languages, MDPs are often represented as a composition of multiple identical components, called modules. For example, in a distributed system where multiple processes are interacting or sharing common resources, each process can be described as a separate module. In the context of this paper, we also consider the number of modules as a parameter. Example 2 (Dining philosophers). The dining philosophers problem involves a number of philosophers, seated around a circular table, blessed with infinite
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3f9b/3f9ba693-f90f-4bd0-8ba6-014f7ffe3bd8.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 1: A parameterized, scalable MDP with k+1 blocks, described in Example 1.</div>
availability of food and things to ponder about. There is a fork placed between each pair of neighbouring philosophers and a philosopher must have both forks in order to eat. They may only pick up one fork at a time and once they finish eating, they place the forks back at the table and return to thinking. This can be described as an MDP where each philosopher is modeled as a separate module. Thus, the number of philosophers is a parameter.
# 2.3 Decision trees for policy representation
Knowing that the state space is a product of state-variables, a deterministic policy is a mapping � i Di →A from tuples of state variables to actions. By viewing the state variables as features and the actions as labels, we can employ machine learning classification techniques such as decision trees , see e.g. [38, Chapter 3], to represent a policy concisely. We refer to [2] for an extensive description of the approach and its advantages. Here, we shortly recall the most relevant definitions in order to formally state our results.
Definition 3. A decision tree (DT) T is defined as follows:
– T is a rooted full binary tree, meaning every node either is an inner node and has exactly two children or is a leaf node and has no children. – Every inner node v is associated with a decision predicate αv which is a boolean function S →{false, true} (or equivalently � i Di →{false, true}).
For a given state s, we use the following recursive procedure to obtain the action σ(s) = a that a DT prescribes: Start at the root. At an inner node, evaluate the decision predicate on the given state s. Depending on whether it evaluates to false or true, recursively continue evaluating on the left or right child, respectively. At a leaf node, return the associated action a. Example 3. Consider again the MDP given in Figure 1. The optimal policy needs to continue towards the goal and not be stuck in any loops. This can be achieved by playing action a in states where x = 0 and action b in all other states. Traditionally, this policy would be represented as a lookup table, storing 2k + 3 state-action pairs explicitly. Instead, we can condense the policy to the DT given in Figure 2b, mimicking the intuitive description of the policy: If x > 0, we play b, otherwise, we play a. Constructing an optimal binary decision tree is an NP-complete problem [26]. Consequently, practical decision tree learning algorithms are based on heuristics. But they tend to work reasonably well. Here, we briefly recall a general framework of learning the decision tree representation of a policy σ as described in [2]. If the policy suggests same action a for all states (i.e., for all states s, we have σ(s) = a), the tree is just a single leaf node with label a. Otherwise, we split the policy. A predicate ρ, defined on state variables, is chosen, and an inner node labeled ρ is created. Then we partition the policy by evaluating the predicate on the state space, and recursively construct one DT for the policy restricted to the states {s ∈S|ρ(s)} where the predicate is true, and one for the policy restricted to the states {s ∈S|¬ρ(s)} where ρ is false. These policies become the children of the inner node with label ρ and the process repeats recursively. The selection of “best” predicate is done by selecting the one which is able to split the policy as homogeneous as possible. This is determined by optimizing some impurity measure such as Gini impurity [8, Chapter 4] or entropy [44]. As we want the learnt DT to exactly represent the policy, unlike other ML algorithms, we do not stop the learning early based on a stopping criterion. Instead, we overfit on the data. So, the iteration stops when every state in the leaf node of the tree has the same labeling.
Constructing an optimal binary decision tree is an NP-complete problem [26]. Consequently, practical decision tree learning algorithms are based on heuristics. But they tend to work reasonably well. Here, we briefly recall a general framework of learning the decision tree representation of a policy σ as described in [2]. If the policy suggests same action a for all states (i.e., for all states s, we have σ(s) = a), the tree is just a single leaf node with label a. Otherwise, we split the policy. A predicate ρ, defined on state variables, is chosen, and an inner node labeled ρ is created. Then we partition the policy by evaluating the predicate on the state space, and recursively construct one DT for the policy restricted to the states {s ∈S|ρ(s)} where the predicate is true, and one for the policy restricted to the states {s ∈S|¬ρ(s)} where ρ is false. These policies become the children of the inner node with label ρ and the process repeats recursively. The selection of “best” predicate is done by selecting the one which is able to split the policy as homogeneous as possible. This is determined by optimizing some impurity measure such as Gini impurity [8, Chapter 4] or entropy [44]. As we want the learnt DT to exactly represent the policy, unlike other ML algorithms, we do not stop the learning early based on a stopping criterion. Instead, we overfit on the data. So, the iteration stops when every state in the leaf node of the tree has the same labeling.
# 3 Generalizing policies from small problem instances
In this section, we develop an approach for obtaining good policies for MDPs that are practically beyond the reach of any available rigorous analysis. Our approach exploits the regularity in structure of the MDPs, therefore, we focus on parameterized MDPs where we expect regularity in the state space. Intuitively, we solve a few small instances (colloquially speaking “1, 2, and 3”) where an optimal policy is easy to compute. Then we generalize these policies by learning a DT from the combined information. The policy represented by this
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3318/331827eb-6e36-41af-9815-04794f448519.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 2: (a) MDP instance for m = 1 from Figure 1. (b) The optimal policy represented as a DT as learned by our approach.</div>
DT is applicable to any instantiation of the parameterized MDP, even the ones that are infeasibly large for any state-of-the-art solver. More concretely, our approach proceeds in three phases, each of which is described in detail in the following subsections. 1. Select the parameters for the small instances (base instances) to learn on. 2. Collect the optimal policies on the base instances. 3. Generalize these policies by learning a DT. Finally, we discuss how to apply the DT to the MDP and evaluate the policy in order to judge its performance. We illustrate every phase on our running example, the MDP in Figure 1.
Finally, we discuss how to apply the DT to the MDP and evaluate the policy in order to judge its performance. We illustrate every phase on our running example, the MDP in Figure 1.
# 3.1 Parameter selection
In principle, one can use any of the solvable instances as the set of base instances. However, too small instances may not contain enough information to learn a good generalization. Therefore, we select a small set of instances B = {b1, ..., bn}, such that the computed policies of these instances are rich enough to learn a generalized DT (see Section 4.1 for the details how we practically choose this set). This process can be seen as hyper-parameter search. Domain knowledge
Algorithm 1 Computing input for the DT learning from small problem in-
stances.
Input: A parameterized MDP M and base instances B = {b1, ..., bn}
Output: A data set D ⊆SB × 2AB
D ←∅
for all bi ∈B do
Solve Mbi and get optimal policy σbi
for all s ∈Sbi do
if s is reachable in M
σbi
bi
and s /∈Gbi then
add (s, σbi(s)) to D
▷Add optimal actions
return D;
can be very useful to obtain this small set of instances. If not available, we can choose a time budget and solve as many instances as possible in that time. For the remainder of this section, we assume that we are given a set B = {b1, ..., bn} of base instances to learn on. We shall interchangeably use bi for ith instance of the MDP, as well as for the parameters of the ith instance. Hence each bi can be seen as a vector of parameters with its corresponding values. Formally, bi := ⟨p1=v1, ..., pm=vm⟩, where p1, ..., pm are the parameters and v1, ...vm are the values that are assigned to each of the respective parameters to obtain the instance.
Example 4. Our running example has one parameter k, the number of module In fact, we will see that it suffices to consider B = {b1} where b1 := ⟨k=1⟩, i. only learn on the simplest instance of the MDP as depicted in Figure 2a.
# 3.2 Collecting policies
We collect the optimal decisions from the optimal policies of each of the base instances into a single dataset, later to be used for learning their generalization. The input of the learning algorithms is a data set (possibly a multiset) of samples of (input, output)-pairs. In our case, it is a set of pairs of the structured state and the chosen action. Algorithm 1 describes how to obtain the input function that can be used for the DT learning from a parameterized MDP and a set of parameterizations. Let Mb = (Sb, Ab, δb, ¯sb, Gb) be a concrete instance of a parameterized MDP with b := ⟨p1=v1, ..., pm=vm⟩. For a set of base instances B = {b1, ..., bn}, we denote the union of all state spaces as SB := �n i=1 Sbi, and similarly define AB as the union of all action spaces. Note that when aggregating the information, we exclude states that are not reachable in the Markov Chain (MC) induced by the computed optimal policy, as well as goal states. This reduces the input size for the DT learning which has two advantages: firstly, it speeds up the computation and secondly, it allows the DT to focus on the relevant states. Example 5. For our running example in Example 1, we only consider one base instance. Thus, our data is given by the function for all reachable non-goal
Example 5. For our running example in Example 1, we only consider one base instance. Thus, our data is given by the function σ1 for all reachable non-goal
# states in the MDP. Concretely, we have pairs of state (0, 0) with action a, th (1, 0) also with a, and (1, 1) with b.
states in the MDP. Concretely, we have pairs of state (0, 0) with action a, then (1, 0) also with a, and (1, 1) with b.
Note that the algorithms are even able to deal with data where the output is non-deterministic (different outputs for the same input within the data set). This corresponds to accommodating permissive policies, i.e. ones that allow multiple actions per state. Thus, we can include all optimal actions from a base instance in the data set. Similarly, we can aggregate the information over multiple base instances by a simple union of the recommended actions over all policies. In such cases, the derived DT would also predict multiple actions. Various determinization approaches (see [1]) exist to get one action in these cases, e.g., determinization by voting picks the action that appears most often in the base instances. Action labels. Usually, actions are internally represented as integer values in the model-checkers, e.g. in STORM [25] and PRISM [30]. A natural choice would be to use these integers as labels for the actions during decision tree learning. However, this creates a problem: When the set of actions varies among different instances, “identical” action choices are denoted by different integers in each instance. Example 6. Consider the dining philosophers problem in Example 2. For three philosophers, the initial state would have 6 actions. The first 3 are labeled with 0, 1, 2, where an action i represents that the (i + 1)th philosopher thinks. The second 3 actions are labeled with 3, 4, 5, where an action i represents that the (i −2)th philosopher eats. If we try to generalize this to an MDP with n > 3 philosophers, the label 3 is now interpreted as the action that the fourth philosopher thinks, not the first philosopher eats as it was in the case of three philosophers. Thus, representing actions only by the index in which they appear can be sufficient if the number of modules does not change, but is problematic in the opposite case.
Note that the algorithms are even able to deal with data where the output is non-deterministic (different outputs for the same input within the data set). This corresponds to accommodating permissive policies, i.e. ones that allow multiple actions per state. Thus, we can include all optimal actions from a base instance in the data set. Similarly, we can aggregate the information over multiple base instances by a simple union of the recommended actions over all policies. In such cases, the derived DT would also predict multiple actions. Various determinization approaches (see [1]) exist to get one action in these cases, e.g., determinization by voting picks the action that appears most often in the base instances. Action labels. Usually, actions are internally represented as integer values in the model-checkers, e.g. in STORM [25] and PRISM [30]. A natural choice would be to use these integers as labels for the actions during decision tree learning. However, this creates a problem: When the set of actions varies among different instances, “identical” action choices are denoted by different integers in each instance.
Example 6. Consider the dining philosophers problem in Example 2. For three philosophers, the initial state would have 6 actions. The first 3 are labeled with 0, 1, 2, where an action i represents that the (i + 1)th philosopher thinks. The second 3 actions are labeled with 3, 4, 5, where an action i represents that the (i −2)th philosopher eats. If we try to generalize this to an MDP with n > 3 philosophers, the label 3 is now interpreted as the action that the fourth philosopher thinks, not the first philosopher eats as it was in the case of three philosophers. Thus, representing actions only by the index in which they appear can be sufficient if the number of modules does not change, but is problematic in the opposite case.
To overcome this issue, we take advantage of the action-labeling feature in the PRISM language. An action in PRISM is described by a command of the following form: [label] guard -> prob_1 : update_1 +...+ prob_n : update_n. This means that when the condition in the guard is true, update_i happens with probability prob_i. The label of the action is optional (except when the action is a synchronizing action). But, as a simple preprocessing step, we always define the label in each command in the PRISM file. The DT learning then can use these labels instead of the action indices. These labels need to be unique: assigning same label to two actions in two different modules would force the modules to take these actions simultaneously (i.e. to synchronize) changing the structure of the MDP. Also, we only need to define labels for non-synchronizing actions as synchronizing actions already have labels defined that we can use. For example, the problem in Example 6 can be avoided by giving the unique label phil_i_line_j to the action for (i + 1)th philosopher defined by the command at line j in the PRISM file.
# 3.3 Decision tree learning
We use standard DT learning algorithms to learn a DT from the dataset constructed in the Algorithm 1. For predicates to be used, we consider axis-aligned predicates (i.e. predicates of the form x > c where x is a state variable and c ∈R). The best predicate is selected by calculating the Gini index. Instead of having a stopping criterion, we let the recursive splitting of the dataset happen until no further splitting is possible. The resulting DT generalizes the policy in two ways: 1. The DT is trained using smaller base instances. The same state variables in the DT’s inner node predicates are present in the larger MDP instances, but they can have a larger domain. Despite this difference, the DT would still partition the state space of the larger MDP instances and still recommend actions corresponding to each state. 2. As we are aggregating multiple policies, in our dataset, unlike the learning algorithm described in Section 2.3, we can have a state with more than one suggested actions. The learning algorithm considers them as distinct data-points sharing the same value but different labels. Since the values are the same, there are no predicates that can distinguish them. So these datapoints traverse the same path in the tree until they reach a leaf node. The classification at the leaf node is determined by ‘majority voting’, the label that appears most often is assigned to the leaf node. This approach helps filter out actions suggested by only a few less generalizing base instances. Example 7. For our example data set D constructed in Example 5, the result of the DT learning is the DT depicted in Figure 2b. This policy is in fact optimal for all k; see Example 3 for an explanation of this. In addition to being optimal, it is also small and perfectly explainable. In contrast, if we are interested in a huge instance of this model, e.g., setting k = 1015, already storing the resulting MDP in the memory in order to compute an optimal policy is challenging or even infeasible for a large enough k. Additionally, the policy produced by state-of-the-art model checkers is represented as a lookup table with as many rows as there are states.
# 3.4 Applying and evaluating the resulting policy
Once we have a decision-tree representation of a policy, we can apply it to MDP instances of arbitrary size. To evaluate a policy, we simply need to compute the value of the MC induced by applying the DT. Since solving MCs is computationally easier than solving MDPs, we can explicitly compute values for larger MDPs (which we could not do otherwise). Nonetheless, one can still scale the parameter to such an extent that the construction of the corresponding MC requires too much time or memory. In such cases, we can use statistical model checking methods [48]. The resulting value is not only a measure for the performance of the DT policy, but also a guaranteed lower bound on the value of the MDP (or an upper bound in the case of minimization).
Since our approach is based on generalization, the learned DT may, in principle, recommend an action that is not available in that state. In such cases our implementation would choose an action uniformly from the available actions. While this may occur in principle, we have not encountered this situation in any of the experiments we conducted for evaluation.
# 4 Evaluation
# 4.1 Experiment Setup
Benchmark Selection We selected parameterized MDPs with reachability objective from the quantitative verification benchmark set (QVBS) [22]. Models with reward-bounded reachability (e.g., eajs and resource-gathering) were excluded. We also identified trivial model and property combinations where the equation minσ Pσ ¯s [♢G] = maxσ Pσ ¯s [♢G] holds for the set of goal states G and the initial state ¯s. In such cases, any valid policy would act as an optimal policy. We have excluded these from our benchmark set. We extended the benchmark set with the Mars Exploration Rovers (mer) case study, which was introduced in [14] and appears frequently in recent literature. This model is interesting because the probability of its property is non-trivial and it is scalable to large parameter values without degenerating into a trivial model. Choice of Base Instances We conducted experiments to observe the effect of the set of base instances on the value produced by the learned policy. We synthesized decision trees from different sets of base instances, increasing the parameter(s) linearly as well as exponentially, and evaluated them on models larger than the base instances. We observed that one or two instances are often already enough to generalize the policy in the considered benchmark set (See Appendix A for the chosen set of base instances used in our experiments). System Configuration The experiments were executed on an AMD EPYC™ 7443 server with 48 physical cores, 192 GB RAM, running Ubuntu 22.04.2 LTS operating system with Linux kernel version 5.15.0-83-generic. This powerful server was used to execute many runs in parallel. We assigned 2 cores and 8 GB RAM to each run. For all experiments, we used BenchExec [5], a state of the art benchmarking tool, to isolate the executions and enforce the resource limitations. Implementation Details We implemented our approach as an extension of the probabilistic model checker Storm [25]. Our code is publicly available at: https://github.com/muqsit-azeem/dtstrat-123go-artifact/. Method of Comparison Our aim is to provide a method for policy synthesis for arbitrarily large instances of parameterized MDPs, in particular for MDPs beyond the reach of any available rigorous analysis. Consequently, the optimal value for such an MDP is by definition unknown and optimality becomes not only uncheckable, but also unexpectable—rather, one can hope for values close to the range where the unknown optimum is expected to lie.
Hence a straightforward evaluation is thus beyond reach, and we devise the following ancillary evaluation process. First, we also compare on small benchmarks, although our approach is by no means meant as a competitor of Storm on them. Nonetheless, it gives us the following two details: (i) optimal values for various parameter instantiations, often allowing for a simple extrapolation, and (ii) our relative error for these various parameter instantiations, allowing for another extrapolation. While the performance on small models is irrelevant (of course, exact methods are to be used when feasible), the resulting extrapolations give us some idea how our approach performs in the area of interest. In addition to comparing to the theoretical optimum (obtained by extrapolation), we compare to SMC, which is the key state-of-the-art technique for too large systems, and to randomly chosen policies as a baseline. Technical Description First, to obtain optimal values for each of the MDP instances, we executed all the engines of Storm (sparse, dd, hybrid, dd-tosparse). We executed each run with a CPU time limit of 1 hour and memory limit of 8 GB. We considered the CPU time taken by the fastest engine for each instance. Second, we obtain values by using the state-of-the-art statistical model checker MODES [9], part of the MODEST toolset [20]. The approaches for picking the policies are (i) smart lightweight scheduler sampling (Smart LSS) [12], executed in the default configuration, producing the policy value with confidence bound 0.99 and error bound 0.01; (ii) the uniform policy, which resolves each non-deterministic choice by picking an action uniformly at random, again with confidence bound 0.99 and error bound 0.01; and (iii) an aggregate of 1000 randomly generated deterministic policies, i.e., non-randomizing policies where each non-deterministic choice is resolved by a single action, sampled independently according to the uniform distribution, each evaluated with 1000 simulation runs. Finally, we evaluate our approach by computing the value of the MCs resulting from applying our generalizing DTs. In most of the cases (except 4), we were able to evaluate our policy precisely. In the 4 remaining cases, we used our own implementation of SMC to evaluate the learned DT. For 3 out of these 4, we were able to produce a value with with confidence bound 0.99 and error bound 0.01, and in the remaining one (csma+some_before, N = 8) we had to use the confidence bound 0.95 and error bound 0.05. The key idea of the evaluation is to show how the values (optimal / for our approach / for different random schedulers) evolve with the parameter. We executed all the tools on the MDPs obtained by scaling the value of the parameter. In case of MDPs with a single parameter, we start with the smallest parameter values suggested in the QVBS and then increase it. In cases where there was more than one parameter, we scaled each parameter while fixing the values of the other ones. The values chosen to be fixed were the smallest values for these parameters taken from the QVBS website. Since we could not run experiments for all the parameter values due to resource constraints, we sampled the parameters. (See Table 5 in Appendix B for the concrete parameter values used in our experiments.)
We present the results for the parameters that Storm was able to solve within a minute of CPU time, within one hour of CPU time, and an instance that even Storm was not able to solve within an hour of CPU time. Sometimes, parameter scaling does not increase the time required to solve the given MDP. In such cases, we still present several parameter valuations and the corresponding values to assess how the values evolve.
# 4.2 Results
Table 1 and Table 2 show the results of our evaluation for the minimizing and maximizing properties, respectively. Each instance refers to a combination of model, property and parameter. The tables show, for each model+property combination, the parameter value and CPU time taken by Storm to solve it (< 1 min, < 60 min, and Beyond in case Storm could not solve the instance in an hour), the values produced by Storm, our approach and the sampling based SMC. The tables report OOR when running out of resource (time or memory). Also, a few MODES runs resulted in a run length exceeded (RLE) error. Some models converge to triviality (i.e., max=min) as we scale the parameter. Zeroconf_dl+deadline_min becomes trivial for higher values of the parameter K than 3, firewire+deadline becomes trivial for deadline > 1300, and csma+some_before becomes trivial when the value of K is more than twice the value of N. Pacman also approaches closer to triviality for higher values of the parameter MAX_STEPS (the horizon). The results show that our approach gives near optimal values for 13 out of 21 cases (the upper halves of Table 1 and Table 2), better than Smart LSS for 2 out of remaining 8 cases, and generally better than random and uniform in remaining cases. There are two instances where random performs better than our approach (pacman for the MAX_STEPS 25, and csma+all_before_max for N=3), see the discussion below.
# 4.3 Discussion
Although our approach is simple, it performs well in a number of cases. We often generalize from a single instance or two, yielding satisfactory solutions for arbitrarily large instantiations. In a number of cases, we can justifiably extrapolate that the policies are (nearly) optimal for all instances. For instance, consider the two benchmarks of Figure 3. No matter how much the model is scaled up, the value of our policy seems to remain stable. While its (near-)optimality can be proven only up to a certain point (beyond which no ground truth can be known), the apparent stability suggests it is true onwards, too. Note that MODES returns low values as the optimal policies are rather rare. In the sequel, we discuss the scope and the limitations of our approach in details. As discussed earlier, we can divide the parameters in two types. Type 1: Parameters that dictate the number of PRISM-modules. This type of parameter not only changes the structure of the MDP, but also increases the
Values
Model+property
Scale
MODES
(values of parameters)
Variable
Time
Storm
1-2-3-Go
Smart LSS
Uniform
Random
zeroconf_dl+deadline_min
(N=1000, K=1)
dl=200
< 1 min
5.02 × 10−207
2.62 × 10−43
0♯
0.00
0.00
dl=1600 < 60 min
0
6.81 × 10−86
0
0.00
0.00
dl=3200
Beyond
OOR
8.71 × 10−135
0
0
0.00
zeroconf_dl+deadline_min
(N=1000, deadline=10)
K=2
< 1 min
0.34
0.34
0.34
0.34♯
0.34
K=8
< 1 min
1
1
1
1
1
firewire+deadline
(deadline=200)
delay=5
< 1 min
0.50
0.50
0.56
0.99
0.99
delay=21 < 1 min
0.50
0.50
1
1
1
delay=34 < 1 min
0
0
1
1
1
delay=89 < 1 min
0
0
1
1
1
firewire+deadline
(delay=3)
dl=200
< 1 min
0.50
0.50
0.50♯
0.98
0.97
dl=500
< 1 min
0.85
0.85
1
1
1
dl=1300
< 1 min
1.00
1.00
1
1
1
csma+some_before
(N=2)
K=2
< 1 min
0.50
0.50
0.50♯
0.50♯
0.50
K=3
< 1 min
0.88
0.88
0.88
0.87♯
0.87♯
csma+some_before
(fix K=2)
N=3
< 1 min
0.59
0.59
0.58♯
0.89
0.90
N=5
< 60 min
0.21
0.21
0.39
0.79
0.78
N=8
< 60 min
0.04
0.03†
0.51
0.75
0.76
N=13
Beyond
OOR
OOR
0.56
0.75
0.76
consensus+c2
(N=2)
K=2
< 1 min
0.38
0.47
0.42
0.49
0.49
K=55
< 1 min
0.49
0.50
RLE
RLE
RLE
K=144
< 1 min
0.49
0.50
RLE
RLE
RLE
consensus+c2
(K=2)
N=6
< 1 min
0.29
0.45
0.49
0.48
0.48
N=7
< 60 min
0.29
0.46
0.48
0.49
0.48
N=13
Beyond
OOR
0.46
RLE
0.49
RLE
zeroconf_dl+deadline_min
(deadline=10, K=1)
N=1000
< 1 min
0.00
0.00
0.00
0.01♯
0.01
N=8000
< 1 min
0.01
0.04
0.04
0.07
0.06
N=32000 < 1 min
0.11
0.22
0.21
0.35
0.32
pacman+crash
MS=5
< 1 min
0.55
0.55
0.55♯
0.55
0.55♯
MS=25
< 1 min
0.55
0.87
0.73
0.93
0.92
MS=200 < 60 min
0.55
1.00
1
1
1
MS=300
Beyond
OOR
1.00
1
1
1
number of state variables. Note that when we train a DT from the policies from smaller base instances, the predicates in decision tree would not use the state variables present only in the bigger instances. Even then, as the system is a product of these isomorphic modules, we can think the smaller base instances as projections of the larger instance to the first
Table 2: The results table for maximizing properties. The values marked by † were approximated using SMC. We shorten the parameter deadline to dl. OOR means out-of-resources (both time and memory) and RLE means run length
exceeded.
Values
Model+property
Scale
MODES
(values of parameters)
Variable
Time
Storm
1-2-3-Go
Smart LSS
Uniform
Random
zeroconf_dl+deadline_max
(N=1000, K=1)
dl=200
< 1 min
0.01
0.01
0.00
0.00
0.00
dl=1600 < 60 min
0.01
0.01
0.00
0.00
0.00
dl=3200
Beyond
OOR
0.01
0.00
0
0.00
zeroconf_dl+deadline_max
(N=1000, deadline=10)
K=2
< 1 min
0.34
0.34
0.34
0.34
0.34
K=8
< 1 min
1
1
1
1
1
K=32
< 1 min
1
1
1
1
1
zeroconf_dl+deadline_max
(deadline=10, K=1)
N=1000
< 1 min
0.02
0.02
0.01
0.01
0.01
N=8000
< 1 min
0.12
0.12
0.10
0.07
0.06
N=32000 < 1 min
0.49
0.49
0.47
0.35
0.32
philosophers+eat
N=5
< 1 min
1
1
1
1
0.04
N=21
< 60 min
1
1
0.51
1
0.00
N=34
Beyond
OOR
1
0.49
1
0
pnueli-zuck+live
N=5
< 1 min
1
1
1
1
0.04
N=21
< 60 min
1
1
1
1
0.00
N=34
Beyond
OOR
1
1
1
0.00
csma+all_before_max
(N=2)
K=8
< 1 min
1
1
1
1
1
K=11
< 60 min
1
1†
1
1
1
K=13
Beyond
OOR
1†
1
1
OOR
mer+p1
(x=0.01)
n=1000
< 1 min
0.20
0.20
RLE
RLE
0.00
n=21000 < 60 min
0.20
0.20
RLE
RLE
0
n=55000
Beyond
OOR
0.20
RLE
RLE
0.00
mer+p1
(n=10)
x=0.01
< 1 min
0.20
0.20
RLE
RLE
0.00
x=0.08
< 1 min
0.21
0.20
RLE
RLE
0.00
x=0.55
< 1 min
0.36
0.20
RLE
RLE
0.00
x=0.89
< 1 min
0.67
0.20
RLE
RLE
0.00
consensus+disagree
(K=2)
N=5
< 1 min
0.34
0.10
0.05
0.03
0.04
N=7
< 60 min
0.38
0.07
0.03
0.04
0.03
N=13
Beyond
OOR
0.05
RLE
0.03
RLE
consensus+disagree
(N=2)
K=2
< 1 min
0.11
0.06
0.08
0.03
0.03
K=55
< 1 min
0.00
0.00
RLE
RLE
RLE
K=144
< 1 min
0.00
0.00
RLE
RLE
RLE
csma+all_before_max
(K=2)
N=3
< 1 min
0.86
0.52
0.85
0.03
0.68
N=5
< 60 min
0.70
0.05
0.41
0.04
0.24
N=6
Beyond
OOR
0.01†
0.21
0.04
0.11
few state variables. Then, for some interesting properties, our method still gives good policies: cases where there is a generalizing optimal policy that does not depend on the additional modules. For example, consider the csma model describing the CSMA/CD consensus protocol when N stations use a network with a single channel. Each station is
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/19bc/19bc10bb-c87e-4031-a94a-971774e9b958.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) zeroconf_dl + deadline_max</div>
<div style="text-align: center;">Fig. 3: Robustness of the policy given by 1-2-3-Go!, Storm and MODES for different instances of zeroconf_dl + deadline_max and philosophers + eat. For larger models, while Storm times out and MODES gives sub-optimal policies, 1-2-3-Go! consistently gives better results irrespective of N.</div>
Fig. 3: Robustness of the policy given by 1-2-3-Go!, Storm and MODES for different instances of zeroconf_dl + deadline_max and philosophers + eat. For larger models, while Storm times out and MODES gives sub-optimal policies, 1-2-3-Go! consistently gives better results irrespective of N.
represented by a module in the PRISM file. Now consider two different properties “all_before_max” and “some_before”. the first one checks the maximum probability that all stations send the message successfully avoiding a data collision. A policy maximizing successful transmission for all stations, needs to take account of the state variables corresponding to all modules in the PRISM file. Our approach fails with this kind of property. But on the other hand, the second property checks the minimum probability that some station eventually sends the message successfully. Thus, a decision tree generated from the base instances gives a policy that minimizes the collision probability for some station among the first three stations. This would act as an optimal policy (as it optimizes the collision probability for some station) in the larger instances even though the predicates in the DT does not contain state variables related to stations with larger index. Thus, for this property, our approach succeeds in generalizing the optimal policy. Type 2 : Other parameters which can be changed by setting the value externally. Often, this would mean expanding the domain of the state variables (as in the case of Example 1), which increases the size of the state space linearly. Then our approach can still work as we can have an optimal policy that is independent of that specific state variable, or if the newly added states are not relevant. However, this is not always the case: The parameter MAX_STEPS in pacman + crash denotes the number of steps Pac-Man needs to stay safe from the ghosts. Our algorithm fails for this model as the policies across instances cannot be generalized. Indeed, a policy that minimized the probability of crash for K steps does not provide any information about how to stay safe for K′ > K steps. In the case of mer + p1, if we fix n and vary the parameter x, our approach fails to generalize a policy. Changing the value of x does not change the statespace or the structure of the model, but changes the probability values of the transitions, which in turn would change the optimal policy across different in-
<div style="text-align: center;">(b) philosophers + eat</div>
stances. For that reason, we cannot construct a generalizing DT as the decision predicates are defined on the state variables and not on the probability value of the transition. Time The approach typically needs less than 5 s to generate a policy, except for zeroconf_dl (see Table 8 and Table 9 in Appendix D). In the case of zeroconf_dl, the time taken is a couple of minutes because we learned from a larger instance, with non-trivial values for all three involved parameters. However, this instance is then so informative that we could later use it for all instances derived by varying different parameters (i.e., varying N and fixing others, varying delay and fixing other, and varying deadline and fixing others, not having to take care of these three families separately).
# 5 Conclusion
We have seen that practically good policies (with values close to the unknown optimum in the sense of the “Method of Comparison” above) can be generated in a lightweight way even for very large parameterized models, beyond reach of any other methods. In order to synthesize policies for arbitrarily large models, we generalize the policies computed for the smaller instances using (more explainable and thus more generalizable) decision trees, coining the “generalizability by explainability”. The generalization is an example of unreliable reasoning, which can contribute to better scalability. On the one hand, the unreliability results in no guarantees that the produced policies are anywhere close to optimum, which, however, often cannot be computed anyway. On the other hand, the values of the policies can be reliably approximated: either numerically with absolute guarantees if the resulting Markov chain is still analyzable (e.g., with partialexploration methods [29]) or with statistical guarantees by SMC on the Markov chain. Consequently, although optimal control policies might be out of reach, we can still produce what we thus coin here as provably good enough policies. Moreover, the consistency of the values over the different instantiations often suggests practical proximity to optimum. A possibly surprising point is the conclusion of our experiments that very few base instances need to be analyzed. Such robustness (together with the robustness across the target instances as seen in Fig. 3) suggests that this generalizability is a deeply inherent property of many models, and thus deserves further investigation and exploitation. In particular, our approach is only the first, generic try to exploit this property, opening the new paradigm. As suggested by our experimental results, more specific heuristics for certain types of systems where parameters play different roles, such as number of modules, number of repetitions, time-outs, etc., offer a desirable direction of future work.
# References
1. Ashok, P., Jackermeier, M., Jagtap, P., Kretínský, J., Weininger, M., Zamani, M.: dtcontrol: decision tree learning algorithms for controller rep-
resentation. In: Ames, A.D., Seshia, S.A., Deshmukh, J. (eds.) HSCC ’20: 23rd ACM International Conference on Hybrid Systems: Computation and Control, Sydney, New South Wales, Australia, April 21-24, 2020. pp. 30:1–30:2. ACM (2020). https://doi.org/10.1145/3365365.3383468, https://doi.org/10.1145/3365365.3383468 2. Ashok, P., Jackermeier, M., Kretínský, J., Weinhuber, C., Weininger, M., Yadav, M.: dtcontrol 2.0: Explainable strategy representation via decision tree learning steered by experts. In: TACAS (2). Lecture Notes in Computer Science, vol. 12652, pp. 326–345. Springer (2021) 3. Baier, C., Katoen, J.: Principles of model checking. MIT Press (2008) 4. Baier, C., Klein, J., Leuschner, L., Parker, D., Wunderlich, S.: Ensuring the reliability of your model checker: Interval iteration for markov decision processes. In: CAV (1). Lecture Notes in Computer Science, vol. 10426, pp. 160–180. Springer (2017) 5. Beyer, D., Löwe, S., Wendler, P.: Reliable benchmarking: requirements and solutions. Int. J. Softw. Tools Technol. Transf. 21(1), 1–29 (2019). https://doi.org/10.1007/s10009-017-0469-y, https://doi.org/10.1007/s10009-0170469-y 6. Brázdil, T., Chatterjee, K., Chmelik, M., Fellner, A., Kretínský, J.: Counterexample explanation by learning small strategies in markov decision processes. In: Kroening, D., Pasareanu, C.S. (eds.) Computer Aided Verification 27th International Conference, CAV 2015, San Francisco, CA, USA, July 1824, 2015, Proceedings, Part I. Lecture Notes in Computer Science, vol. 9206, pp. 158–177. Springer (2015). https://doi.org/10.1007/978-3-319-21690-4\_10, https://doi.org/10.1007/978-3-319-21690-4_10 7. Brázdil, T., Chatterjee, K., Chmelik, M., Forejt, V., Kretínský, J., Kwiatkowska, M.Z., Parker, D., Ujma, M.: Verification of markov decision processes using learning algorithms. In: ATVA. Lecture Notes in Computer Science, vol. 8837, pp. 98–114. Springer (2014) 8. Breiman, L.: Classification and Regression Trees. (The Wadsworth statistics / probability series), Wadsworth International Group (1984) 9. Budde, C.E., D’Argenio, P.R., Hartmanns, A., Sedwards, S.: A statistical model checker for nondeterminism and rare events. In: TACAS (2). Lecture Notes in Computer Science, vol. 10806, pp. 340–358. Springer (2018). https://doi.org/10.1007/978-3-319-89963-3\_20, https://doi.org/10.1007/978-3319-89963-3_20 10. Budde, C.E., Hartmanns, A., Klauck, M., Kretínský, J., Parker, D., Quatmann, T., Turrini, A., Zhang, Z.: On correctness, precision, and performance in quantitative verification - QComp 2020 competition report. In: ISoLA (4). Lecture Notes in Computer Science, vol. 12479, pp. 216–241. Springer (2020) 11. Ciesinski, F., Baier, C., Größer, M., Klein, J.: Reduction techniques for model checking markov decision processes. In: QEST. pp. 45–54. IEEE Computer Society (2008) 12. D’Argenio, P.R., Legay, A., Sedwards, S., Traonouez, L.: Smart sampling for lightweight verification of markov decision processes. Int. J. Softw. Tools Technol. Transf. 17(4), 469–484 (2015) 13. Feng, L.: On learning assumptions for compositional verification of probabilistic systems. Ph.D. thesis, University of Oxford, UK (2014) 14. Feng, L., Kwiatkowska, M., Parker, D.: Automated learning of probabilistic assumptions for compositional reasoning. In: Giannakopoulou, D., Orejas, F. (eds.)
Fundamental Approaches to Software Engineering. pp. 2–17. Springer Berlin Heidelberg, Berlin, Heidelberg (2011) 15. Groote, J.F., Verduzco, J.R., de Vink, E.P.: An efficient algorithm to determine probabilistic bisimulation. Algorithms 11(9), 131 (2018) 16. Gros, T.P., Hermanns, H., Hoffmann, J., Klauck, M., Steinmetz, M.: Deep statistical model checking. In: FORTE. Lecture Notes in Computer Science, vol. 12136, pp. 96–114. Springer (2020) 17. Größer, M., Baier, C.: Partial order reduction for markov decision processes: A survey. In: FMCO. Lecture Notes in Computer Science, vol. 4111, pp. 408–427. Springer (2005) 18. Haddad, S., Monmege, B.: Reachability in mdps: Refining convergence of value iteration. In: RP. Lecture Notes in Computer Science, vol. 8762, pp. 125–137. Springer (2014) 19. Hahn, E.M., Perez, M., Schewe, S., Somenzi, F., Trivedi, A., Wojtczak, D.: Omegaregular objectives in model-free reinforcement learning. In: TACAS (1). Lecture Notes in Computer Science, vol. 11427, pp. 395–412. Springer (2019) 20. Hartmanns, A.: MODEST - A unified language for quantitative models. In: FDL. pp. 44–51. IEEE (2012), https://ieeexplore.ieee.org/document/6336982/ 21. Hartmanns, A., Hermanns, H.: Explicit model checking of very large MDP using partitioning and secondary storage. In: Finkbeiner, B., Pu, G., Zhang, L. (eds.) Automated Technology for Verification and Analysis - 13th International Symposium, ATVA 2015, Shanghai, China, October 12-15, 2015, Proceedings. Lecture Notes in Computer Science, vol. 9364, pp. 131–147. Springer (2015). https://doi.org/10.1007/978-3-319-24953-7\_10, https://doi.org/10.1007/978-3-319-24953-7_10 22. Hartmanns, A., Klauck, M., Parker, D., Quatmann, T., Ruijters, E.: The quantitative verification benchmark set. In: TACAS (1). Lecture Notes in Computer Science, vol. 11427, pp. 344–350. Springer (2019). https://doi.org/10.1007/978-3030-17462-0\_20, https://doi.org/10.1007/978-3-030-17462-0_20 23. Hartmanns, A., Timmer, M.: Sound statistical model checking for MDP using partial order and confluence reduction. Int. J. Softw. Tools Technol. Transf. 17(4), 429–456 (2015) 24. Henriques, D., Martins, J.G., Zuliani, P., Platzer, A., Clarke, E.M.: Statistical model checking for markov decision processes. In: QEST. pp. 84–93. IEEE Computer Society (2012) 25. Hensel, C., Junges, S., Katoen, J., Quatmann, T., Volk, M.: The probabilistic model checker storm. Int. J. Softw. Tools Technol. Transf. 24(4), 589–610 (2022). https://doi.org/10.1007/s10009-021-00633-z, https://doi.org/10.1007/s10009-02100633-z 26. Hyafil, L., Rivest, R.L.: Constructing optimal binary decision trees is np-complete. Information Processing Letters 5(1), 15–17 (1976). https://doi.org/https://doi.org/10.1016/0020-0190(76)90095-8, https://www.sciencedirect.com/science/article/pii/0020019076900958 27. Kamaleson, N.: Model reduction techniques for probabilistic verification of Markov chains. Ph.D. thesis, University of Birmingham, UK (2018) 28. Klein, J., Baier, C., Chrszon, P., Daum, M., Dubslaff, C., Klüppelholz, S., Märcker, S., Müller, D.: Advances in probabilistic model checking with PRISM: variable reordering, quantiles and weak deterministic büchi automata. Int. J. Softw. Tools Technol. Transf. 20(2), 179–194 (2018) 29. Kretínský, J., Meggendorfer, T.: Of cores: A partial-exploration framework for markov decision processes. Log. Methods Comput. Sci. 16(4) (2020)
30. Kwiatkowska, M.Z., Norman, G., Parker, D.: PRISM: probabilistic symbolic model checker. In: Computer Performance Evaluation / TOOLS. Lecture Notes in Computer Science, vol. 2324, pp. 200–204. Springer (2002) 31. Kwiatkowska, M.Z., Norman, G., Parker, D.: Game-based abstraction for markov decision processes. In: QEST. pp. 157–166. IEEE Computer Society (2006) 32. Kwiatkowska, M.Z., Norman, G., Parker, D.: Symmetry reduction for probabilistic model checking. In: CAV. Lecture Notes in Computer Science, vol. 4144, pp. 234– 248. Springer (2006) 33. Kwiatkowska, M.Z., Norman, G., Parker, D.: The PRISM benchmark suite. In: QEST. pp. 203–204. IEEE Computer Society (2012). https://doi.org/10.1109/QEST.2012.14, https://doi.org/10.1109/QEST.2012.14 34. Kwiatkowska, M.Z., Parker, D., Qu, H.: Incremental quantitative verification for markov decision processes. In: DSN. pp. 359–370. IEEE Compute Society (2011) 35. Li, R., Liu, Y.: Compositional stochastic model checking probabilistic automata via symmetric assume-guarantee rule. In: 2019 IEEE 17th International Conference on Software Engineering Research, Management and Applications (SERA). pp. 110– 115. IEEE (2019) 36. Lomuscio, A., Pirovano, E.: A counter abstraction technique for the verification of probabilistic swarm systems. In: AAMAS. pp. 161–169. International Foundation for Autonomous Agents and Multiagent Systems (2019) 37. Maisonneuve, V.: Automatic heuristic-based generation of mtbdd variable orderings for prism models. internship report (2009) 38. Mitchell, T.: Machine learning, vol. 1. McGraw-hill New York (1997) 39. Mohagheghi, M., Salehi, K.: Machine learning and disk-based methods for qualitative verification of markov decision processes. In: ICTERI Workshops. CEUR Workshop Proceedings, vol. 2732, pp. 74–88. CEUR-WS.org (2020) 40. Parker, D.A.: Implementation of symbolic model checking for probabilistic systems. Ph.D. thesis, University of Birmingham, UK (2003) 41. Puterman, M.L.: Markov Decision Processes: Discrete Stochastic Dynamic Programming. Wiley Series in Probability and Statistics, Wiley (1994). https://doi.org/10.1002/9780470316887, https://doi.org/10.1002/9780470316887 42. Pyeatt, L.D., Howe, A.E.: Decision tree function approximation in reinforcement learning (1999) 43. Rataj, A., Wozna-Szczesniak, B.: Extrapolation of an optimal policy using statistical probabilistic model checking. Fundam. Informaticae 157(4), 443–461 (2018) 44. Shannon, C.E.: A mathematical theory of communication. The Bell system technical journal 27(3), 379–423 (1948) 45. Smolka, S., Kumar, P., Kahn, D.M., Foster, N., Hsu, J., Kozen, D., Silva, A.: Scalable verification of probabilistic networks. In: PLDI. pp. 190–203. ACM (2019) 46. Sutton, R.S., Barto, A.G.: Introduction to Reinforcement Learning. Cambridge, MA, USA, 1st edn. (1998) 47. Tappler, M., Aichernig, B.K., Bacci, G., Eichlseder, M., Larsen, K.G.: L*-based learning of markov decision processes. In: FM. Lecture Notes in Computer Science, vol. 11800, pp. 651–669. Springer (2019) 48. Younes, H.L.S., Simmons, R.G.: Probabilistic verification of discrete event systems using acceptance sampling. In: CAV. Lecture Notes in Computer Science, vol. 2404, pp. 223–235. Springer (2002). https://doi.org/10.1007/3-540-456570\_17, https://doi.org/10.1007/3-540-45657-0_17
Azeem et al.
# Appendix
A Choice of base Instances
# A Choice of base Instances
Table 3 shows the effect of different choices of base instances for csma + all_before_m We applied the DT-based policy trained on each such set of instances and evaluated on the instance with N = 2 and K = 2. We see that the base instance set gave the best policy.
our experiments is marked as bold.
Model instance
Parameters for base instances
Value
csma+all_before_max
(N=2, K=2)
{(N = 2, K = 2)}
0.9159110604
{(N = 2, K = 2), (N = 3, K = 2)}
0.919277227
{(N = 2, K = 2), (N = 2, K = 4), (N = 2, K = 6)}
0.9159110604
{(N = 2, K = 2), (N = 2, K = 4)}
0.9159110604
Table 4 shows the chosen base instances and some information about all the models and properties in our benchmark.
# B Parameter Values used in the Experiments
Table 5 shows the parameter values we sampled for running our experiments. We tried sequences based on Fibonacci series, linear, or exponential based on the intuition we gained during our initial experiments.
# C Result of simulations using random schedulers
Table 6 and Table 7 reports out results from running 1000 simulations using randomly generated 1000 independent schedulers. A model and property combination with higher coefficient of variation (ratio of the standard deviation to the mean) would imply that the distribution of policies are more sparse and it is improbable to find a near optimal policy by randomly generating one.
# D Time comparison
Table 8 and Table 9 shows the time needed for creating and evaluating the DT on different instances.
<div style="text-align: center;">Table 4: MDPs in the benchmark set, parameters, choice of base instances</div>
Table 4: MDPs in the benchmark set, parameters, choice of base instances
Model instance
Parameters
Parameter value
for base instance
Maximizing Probabilities
consensus+disagree
(N, #modules),
(K,Var)
{(N=2,K=8),
(N=3,K=8)}
csma+all_before_max
(N, #modules),
(K,Var)
{(N=2,K=2),
(N=3,K=2)}
mer+p1
(n, Var),
(x, Var, probability)
{(n=1,x=0.01)}
philosophers+eat
(N, #modules, number of processes)
{(N=4)}
zeroconf_dl+deadline_max
(deadline, Var),
(K,Var)
{(reset=false,
N=1000,K=2,
deadline=50)}
pnueli-zuck+live
(N, #modules, number of processes)
{(N=3)}
Minimizing Probabilities
consensus+c2
(N, #modules),
(K,Var)
{(N=2,K=8),
(N=3,K=8)}
csma+some_before
(N, #modules),
(K,Var)
{(N=2,K=2),
(N=3,K=2)}
zeroconf_dl+deadline_min
(deadline, Var),
(K,Var)
{(reset=false,
N=1000,K=2,
deadline=50)}
firewire+deadline
(deadline, Var) ,
(delay, Var)
{(deadline=200,
delay=3)}
pacman+crash
(MAX_STEPS, Var)
{(MAX_STEPS=5)}
Table 5: The parameter values used in the experiments
Model instance
Fixed parameters
Parameter
used
for scaling
Minimizing Probabilities
zeroconf_dl+deadline_min
N=1000, K=1
deadline ∈{100, 200, 300, 600, 1000, 1600, 2000,
2400, 2800, 320}
zeroconf_dl+deadline_min
N=1000, deadline=10
K ∈{1, 2, 3, 4, 8, 16, 32}
firewire+deadline
deadline=200
delay ∈{3, 5, 8, 13, 21, 34, 55, 89, 144}
firewire+deadline
delay=3
deadline ∈{200, 300, 500, 800, 1300, 2100, 3400,
5500, 8900, 14400}
csma+some_before
N=2
K ∈{2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 17}
csma+some_before
K=2
N ∈{2, 3, 5, 6, 7, 8, 13, 21, 34, 55, 89, 144}
consensus+c2
N=2
K ∈{2, 3, 5, 8, 13, 21, 34, 55, 89, 144}
consensus+c2
K=2
N ∈{2, 3, 5, 6, 7, 8, 13, 21, 34, 55, 89, 144}
zeroconf_dl+deadline_min
deadline=10, K=1
N ∈{1000, 2000, 4000, 8000, 16000, 32000}
pacman+crash
NA
MAX_STEPs ∈{5, 25, 50, 75, 100, 150, 200, 300,
400, 500}
Minimizing Probabilities
zeroconf_dl+deadline_max
N=1000, K=1
deadline ∈{100, 200, 300, 600, 1000, 1600, 2000,
2400, 2800, 320}
zeroconf_dl+deadline_max
N=1000, deadline=10
K ∈{1, 2, 3, 4, 8, 16, 32}
zeroconf_dl+deadline_max
deadline=10, K=1
N ∈{1000, 2000, 4000, 8000, 16000, 32000}
philosophers+eat
N ∈{3, 4, 5, 8, 13, 21, 24, 27, 30, 34, 55, 89, 144}
pnueli-zuck+live
N ∈{3, 4, 5, 8, 13, 21, 24, 27, 30, 34, 55, 89, 144}
csma+all_before_max
N=2
K ∈{2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 17}
mer+p1
x=0.01
n ∈{1, 10, 100, 1000, 3000, 8000, 13000, 21000,
34000, 55000, 89000, 144000}
mer+p1
n=10
x ∈{0.01, 0.03, 0.05, 0.08, 0.13, 0.21, 0.34, 0.55,
0.89}
consensus+disagree
K=2
N ∈{2, 3, 5, 6, 7, 8, 13, 21, 34, 55, 89, 144}
consensus+disagree
N=2
K ∈{2, 3, 5, 8, 13, 21, 34, 55, 89, 144}
csma+all_before_max
K=2
N ∈{2, 3, 5, 6, 7, 8, 13, 