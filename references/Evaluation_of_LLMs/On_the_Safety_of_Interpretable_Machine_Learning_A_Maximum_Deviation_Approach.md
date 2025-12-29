# On the Safety of Interpretable Machine Learning: A Maximum Deviation Approach
Dennis Wei
IBM Research
dwei@us.ibm.com
Rahul Nair
IBM Research
rahul.nair@ie.ibm.com
Amit Dhurandhar
IBM Research
adhuran@us.ibm.com
Kush R. Varshney
IBM Research
krvarshn@us.ibm.com
Elizabeth M. Daly
IBM Research
elizabeth.daly@ie.ibm.com
Moninder Singh
IBM Research
moninder@us.ibm.com
Interpretable and explainable machine learning has seen a recent surge of interest. We focus on safety as a key motivation behind the surge and make the relationship between interpretability and safety more quantitative. Toward assessing safety, we introduce the concept of maximum deviation via an optimization problem to find the largest deviation of a supervised learning model from a reference model regarded as safe. We then show how interpretability facilitates this safety assessment. For models including decision trees, generalized linear and additive models, the maximum deviation can be computed exactly and efficiently. For tree ensembles, which are not regarded as interpretable, discrete optimization techniques can still provide informative bounds. For a broader class of piecewise Lipschitz functions, we leverage the multi-armed bandit literature to show that interpretability produces tighter (regret) bounds on the maximum deviation. We present case studies, including one on mortgage approval, to illustrate our methods and the insights about models that may be obtained from deviation maximization.
# 1 Introduction
Interpretable and explainable machine learning (ML) has seen a recent surge of interest because it is viewed as a key pillar in making models trustworthy, with implications on fairness, reliability, and safety [1]. In this paper, we focus on safety as a key reason behind the demand for explainability. The motivation of safety has been discussed at a qualitative level by several authors [2–5]. Its role is perhaps clearest in the dichotomy between directly interpretable models vs. post hoc explanations of black-box models. The former have been called “inherently safe” [3] and promoted as the only alternative in high-risk applications [5]. The crux of this argument is that post hoc explanations leave a gap between the explanation and the model producing predictions. Thus, unusual data points may appear to be harmless based on the explanation, but truly cause havoc. We aim to go beyond these qualitative arguments and address the following questions quantitatively: 1) What does safety mean for such models, and 2) how exactly does interpretability aid safety? Towards answering the first question, we make a conceptual contribution in the form of an optimization problem, intended as a tool for assessing the safety of supervised learning (i.e. predictive) models. Viewing these models as functions mapping an input space to an output space, a key way in which these models can cause harm is through grossly unexpected outputs, corresponding to inputs that are poorly represented in training data. Accordingly, we approach safety assessment for a model by determining where it deviates the most from the output of a reference model and by how much (i.e., its maximum deviation). The reference model, which represents expected behavior and is deemed to be
36th Conference on Neural Information Processing Systems (NeurIPS 2022).
safe, could be a model well-understood by domain experts or one that has been extensively “tried and tested.” The maximization is done over a certification set, a large subset of the input space intended to cover all conceivable inputs to the model. These concepts are discussed further in Section 2. Towards answering the second question, in Section 4 we discuss computation of the maximum deviation for different model classes and show how this is facilitated by interpretability. For model classes regarded as interpretable, including trees, generalized linear and additive models, the maximum deviation can be computed exactly and efficiently by exploiting the model structure. For tree ensembles, which are not regarded as interpretable, discrete optimization techniques can exploit their composition in terms of trees to provide anytime bounds on the maximum deviation. The case of trees is also generalized in a different direction to a broader class of piecewise Lipschitz functions, which we argue cover many popular interpretable functions. Here we show that the benefit of interpretability is significantly tighter regret bounds on the maximum deviation compared with general black-box functions, leveraging results from the multi-armed bandit literature. More broadly, the development of tailored methods for additional model classes is beyond the scope of this first work on the maximum deviation approach (the black-box optimization of Section 4.4 is applicable to all models but obviously not tailored). We discuss in Appendix B.5 some possible approaches, and the research gaps to be overcome, for neural networks and to make use of post hoc explanations, which approximate a model locally [6–8] or globally [9, 10]. In Section 5, we present case studies that illustrate the deviation maximization methods in Section 4 for decision trees, linear and additive models, and tree ensembles. It is seen that deviation maximization provides insights about models through studying the feature combinations that lead to extreme outputs. These insights can in turn direct further investigation and invite domain expert input. We also quantify how the maximum deviation depends on model complexity and the size of the certification set. For tree ensembles, we find that the obtained upper bounds on the maximum deviation are informative, showing that the maximum deviation does not increase with the number of trees in the ensemble. Overall, our discussion provides a more quantitative basis for safety assessment of predictive models and for preferring more interpretable models due to the greater ease of performing this assessment.
# 2 Assessing Safety Through Maximum Deviation
We are given a supervised learning model f, which is a function mapping an input feature space X to an output space Y. We wish to assess the safety of this model by finding its largest deviation from a given reference model f0 : X �→Y representing expected behavior. To do this, we additionally require 1) a measure of deviation D : Y × Y �→R+, where R+ is the set of non-negative reals, and 2) a certification set C ⊆X over which the deviation is maximized. Then the problem to be solved is
∈C The deviation is worst-case because the maximization is over all x ∈C; further implications of this are discussed in Appendix C. Note that (1) is different than typical robust training where the focus is to learn a model that minimizes some worst case loss, as opposed to finding regions in X where two already trained models differ significantly. We view problem (1) as only a means toward the goal of evaluating safety. In particular, a large deviation value is not necessarily indicative of a safety risk, as two models may differ significantly for valid reasons. For example, one model may capture a useful pattern that the other does not. We thus think that it would be overly simplistic to regard the maximum deviation as just another metric to be optimized in selecting models. What large deviation values do indicate, however, is a (possibly) sufficient reason for further investigation. Hence, the maximizing solutions in (1) (i.e., the arg max) are of as much operational interest as the maximum values (this will be illustrated in Section 5).
Output space Y. In the case of regression, Y is the set of reals R or an interval thereof. In the case of binary classification, while Y could be {0, 1} or {−1, +1}, these limit the possible deviations to binary values as well (“same” or “different”). Thus to provide more informative results, we take Y to be the space of real-valued scores that are thresholded to produce a binary label. For example, y could be a predicted probability in [0, 1] or a log-odds ratio in R. Similarly for multi-class classification with M classes, Y ⊂RM could be a M-dimensional space of real-valued scores. In Appendix A, we discuss considerations in choosing the deviation function D as well as models that abstain.
ence model f0. The premise of the reference model is that it should capture expected behavior being “safe”. The simplest case is for f0 to be a constant function representing a baseline  for example zero or a mean prediction. We consider the more general case where f0 may vary x. Below we give several examples of reference models to address the natural question of how might be obtained. The examples can be categorized as 1) existing domain-specific models, 2) retable ML models validated by domain knowledge, and 3) extensively tested and deployed s. The first two categories are prevalent in high-stakes domains where interpretability is critical. 1. Existing domain-specific models: These models originate from an application domain and may not be based on ML at all. For example in consumer finance, several industry-standard models compute credit scores from a consumer’s credit information (the FICO score is the best-known in the US). Similarly in medicine, scoring systems (sparse linear models with small integer coefficients) abound for assessing various risks (the CHADS2 score for stroke risk is well-known, see the "Scoring Systems: Applications and Prior Art" section of [11] for a list of others). These models have been used for decades by thousands of practitioners so they are well understood. They may very well be improved upon by a more ML-based model, but for such a model to gain acceptance with domain experts, any large deviations from existing models need to be examined and understood. 2. Interpretable models validated by domain knowledge: Here, an interpretable ML model is learned from data and is validated by domain experts in some way, for example by selecting important input features or by carefully inspecting the trained model. We provide two real examples: In semiconductor manufacturing, process engineers typically want decision trees [12] to model their respective manufacturing process (e.g. etching, polishing, rapid thermal processing, etc.) since they are comfortable understanding and explaining them to their superiors, which is critical especially when things go out-of-spec. Hence, a tree built from data (or any model in general) would only be allowed to make automated measurement predictions if the features it highlights (viz. pressures, gas flows, temperatures) make sense for the specific process. Similarly, in predicting failures of industrial assets such as wind turbines, some failure data is available to train models but experts in these systems (e.g. engineers) may also be consulted. They have knowledge that can help validate the model, for example which components are most likely to cause failures or which environmental variables (e.g. temperature) are most influential. 3. Extensively tested and deployed models: A reference model may also be one that is not necessarily informed by domain knowledge but has been extensively tested, deployed, and/or approved by a regulator. For medical devices that use ML models, the US Food and Drug Administration (FDA) has instituted a risk-based regulatory system. Any system updates or changes, for instance changes in model architecture, retraining based on new data, or changes in intended use (e.g. use for pediatric cases for devices approved only for adults), need to either seek new approvals or demonstrate “substantial equivalence” by providing supporting evidence that the revised model is similar to a previously approved device. In the latter case, the reference model is the approved device and small maximum deviation serves as evidence of equivalence. As another example, consider a ML-based recommendation model for products of an online retailer or articles on a social network, where because of the scale, a tree ensemble may be used for its fast inference time as well as its modeling flexibility [13]. In this case, a model that has been deployed for some time could be the reference model, since it has been extensively tested during this time even though human validation of it may be limited. When a new version of the model is trained on newer data or improved in some fashion, finding its maximum deviation from the reference model can serve as one safety check before deploying it in place of the reference model. fication set C. The premise of the certification set is that it contains all inputs that the model  conceivably be exposed to. This may include inputs that are highly improbable but not cally or logically impossible (for example, a severely hypothermic human body temperature of . Thus, while C might be based on the support set of a probability distribution or data sample, it ot depend on the likelihood of points within the support. The set C may also be a strict superset  training data domain. For example, a model may have been trained on data for males, and we  now like to determine its worst-case behavior on an unseen population of females. bular or lower-dimensional data, C might be the entire input space X. For non-tabular or r-dimensional data, the choice C = X may be too unrepresentative because the manifold of
 Extensively tested and deployed models: necessarily informed by domain knowledge but has been extensively tested, deployed, and/or approved by a regulator. For medical devices that use ML models, the US Food and Drug Administration (FDA) has instituted a risk-based regulatory system. Any system updates or changes, for instance changes in model architecture, retraining based on new data, or changes in intended use (e.g. use for pediatric cases for devices approved only for adults), need to either seek new approvals or demonstrate “substantial equivalence” by providing supporting evidence that the revised model is similar to a previously approved device. In the latter case, the reference model is the approved device and small maximum deviation serves as evidence of equivalence. As another example, consider a ML-based recommendation model for products of an online retailer or articles on a social network, where because of the scale, a tree ensemble may be used for its fast inference time as well as its modeling flexibility [13]. In this case, a model that has been deployed for some time could be the reference model, since it has been extensively tested during this time even though human validation of it may be limited. When a new version of the model is trained on newer data or improved in some fashion, finding its maximum deviation from the reference model can serve as one safety check before deploying it in place of the reference model.
Certification set C. The premise of the certification set is that it contains all inputs that the model might conceivably be exposed to. This may include inputs that are highly improbable but not physically or logically impossible (for example, a severely hypothermic human body temperature of 27°C). Thus, while C might be based on the support set of a probability distribution or data sample, it does not depend on the likelihood of points within the support. The set C may also be a strict superset of the training data domain. For example, a model may have been trained on data for males, and we would now like to determine its worst-case behavior on an unseen population of females. For tabular or lower-dimensional data, C might be the entire input space X. For non-tabular or higher-dimensional data, the choice C = X may be too unrepresentative because the manifold of
realistic inputs is lower in dimension. In this case, if we have a dataset {xi}n i=1, one possibility is to use a union of ℓp balls centered at xi,
The set C is thus comprised of points somewhat close to the n observed examples xi, but the radius r does not have to be “small”. In addition to determining the maximum deviation over the entire set C, maximum deviations over subsets of C (e.g., different age groups) may also be of interest. For example, Appendix D.3 shows deviation values separately for leaves of a decision tree, which partition the input space.
# 3 Related Work
In previous work on safety and interpretability in ML, the authors of [3, 14] give qualitative accounts suggesting that directly interpretable models are an inherently safe design because humans can inspect them to find spurious elements; in this paper, we attempt to make those qualitative suggestions more quantitative and automate some of the human inspection. Furthermore, several other authors have highlighted safety as a goal for interpretability [2, 4, 15, 16, 5], but again without quantitative development. Moreover, the lack of consensus on how to measure interpretability motivates the relationship that we explore between interpretability and the ease of evaluating safety. In the area of ML verification, robustness certification methods aim to provide guarantees that the classification remains constant within a radius ϵ of an input point, while output reachability is concerned with characterizing the set of outputs corresponding to a region of inputs [17]. A major difference in our work is that we consider two models, a model f to be assessed and a reference f0, whereas the above notions of robustness and reachability involve a single model. Another important difference is that our focus is global, over a comprehensive set C, rather than local to small neighborhoods around input points; a local focus is especially true of neural network verification [18–27]. We also study the role of interpretability in safety verification. Works in robust optimization applied to ML minimize the worst-case probability of error, but this worst case is over parameters of f rather than values of x [28]. Thomas et al. [29] present a framework where during model training, a set of safety tests is specified by the model designer in order to accept or reject the possible solution. We build on related literature on robustness and explainability that deals specifically with tree ensembles. Mixed-integer programming (MIP) and discrete optimization have been proposed to find the smallest input perturbation to ‘evade’ a classifier [30] and to obtain counterfactual explanations [31]. MIP approaches are computationally intensive however. To address this Chen et al. [32] introduce graph based approaches for verification on trees. Their central idea, which we use, is to discretize verification computations onto a graph constructed from the way leaves intersect. The verification problem is transformed to finding all maximum cliques. Devos et al. [33] expand on this idea by providing anytime bounds by probing unexplored nodes. Safety has become a critical issue in reinforcement learning (RL) with multiple works focusing on making RL policies safe [34–37]. There are two broad themes [38]: (i) a safe and verifiable policy is learned at the outset by enforcing certain constraints, and (ii) post hoc methods are used to identify bad regimes or failure points of an existing policy. Our current proposal is complementary to these works as we focus on the supervised learning setup viewed from the lens of interpretability. Nonetheless, ramifications of our work in the RL context are briefly discussed in Appendix C.
# 4 Deviation Maximization for Specific Model Classes
In this section, we discuss approaches to computing the maximum deviation (1) for f belonging to various model classes. We show the benefit of interpretable model structure in different guises. Exact and efficient computation is possible for decision trees, and generalized linear and additive models in Sections 4.1 and 4.2. In Section 4.3, the composition of tree ensembles in terms of trees allows discrete optimization methods to provide anytime bounds. For a general class of piecewise Lipschitz functions in Section 4.4, the application of multi-arm bandit results yields tighter regret bounds on the maximum deviation. While some of the results in this section may be less surprising, one of our
contributions is to identify precise properties that allow them to hold. We also show that intuitive measures of model complexity, such as the number of leaves or pieces or smoothness of functions, have an additional interpretation in terms of the complexity of maximizing deviation. More broadly, the development of methods specific to additional model classes is beyond the scope of a single work. We discuss in Appendix B.5 possible approaches and the advances needed for neural networks (beyond applying the black-box methods of Section 4.4) and to make use of post hoc explanations. To develop mathematical results and efficient algorithms, we will sometimes assume that the reference model f0 is from the same class as f. In Appendix C, we discuss the case where f0 may not be globally interpretable, but may be so in local regions. We will also sometimes assume that the certification set C and other sets are Cartesian products. This means that C = �d j=1 Cj, where for a continuous feature j, Cj = [Xj, Xj] is an interval, and for a categorical feature j, Cj is a set of categories. We mention relaxations of the Cartesian product assumption in Appendix B.2.
# 4.1 Trees
We begin with the case where f and f0 are both decision trees. A decision tree with L leaves partitions the input space X into L corresponding parts, which we also refer to as ‘leaves’. We consider only non-oblique trees. In this case, each leaf is described by a conjunction of conditions on individual features and is therefore a Cartesian product as defined above. With Ll ⊂X denoting the lth leaf and yl ∈Y the output value assigned to it, tree f is described by the function
and similarly for tree f0 with leaves L0m and outputs y0m, m = 1, . . . , L0. As discussed in [39, 40], rule lists where each rule is a condition on a single feature are one-sided trees in the above sense. The partitioning of X by decision trees and their piecewise-constant nature simplify the computation of the maximum deviation (1). Specifically, the maximization can be restricted to pairs of leaves (l, m) for which the intersection Ll ∩L0m ∩C is non-empty. The intersection of two leaves Ll ∩L0m is another Cartesian product, and we assume that it is tractable to determine whether C intersects a given Cartesian product (see examples in Appendix B.1). For visual representation and later use in Section 4.3, it is useful to define a bipartite graph, with L nodes representing the leaves Ll of f on one side and L0 nodes representing the leaves L0m of f0 on the other. Define the edge set E = {(l, m) : Ll ∩L0m ∩C ̸= ∅}; clearly |E| ≤L0L. Then
We summarize the complexity of deviation maximization for decision trees as follows. This is a slight refinement of [32, Thm. 1] in the case K = 2, see Appendix B.1 for details. Proposition 1. Let f and f0 be decision trees as in (3) with L and L0 leaves respectively, and E be the bipartite edge set of leaf intersections defined above. Then the maximum deviation (1) can be computed with |E| evaluations as shown in (4).
# 4.2 Linear and additive models
In this subsection, we assume that f is a generalized additive model (GAM) given by
  where each fj is an arbitrary function of feature xj. In the case where fj(xj) = wjxj for all continuous features xj, where wj is a real coefficient, (5) is a generalized linear model (GLM). We discuss the treatment of categorical features in Appendix B.2. The invertible link function g : R �→R is furthermore assumed to be monotonically increasing. This assumption is satisfied by common GAM link functions: identity, logit (g(y) = log(y/(1 −y))), and logarithmic. Equation (5) implies that Y ⊂R and the deviation D(y, y0) is a function of two scalars y and y0. For this scalar case, we make the following intuitively reasonable assumption throughout the subsection.
(4)
(5)
Assumption 1. For y, y0 ∈Y ⊆R, 1) D(y, y0) = 0 whenever y = y0; 2) D(y, y0) is monotonically non-decreasing in y for y ≥y0 and non-increasing in y for y ≤y0.
Our approach is to exploit the additive form of (5) by reducing problem (1) to the optimization
for different choices of S ⊂X and where + corresponds to max and −to min. We discuss below how this can be done for two types of reference model f0: decision tree (which includes the constant case L0 = 1) and additive. For the first case, we prove the following result in Appendix B.2: Proposition 2. Let f be a GAM as in (5) and S be a subset of X where f0(x) ≡y0 is constant. Then if Assumption 1 holds, � �
Tree-structured f0. Since f0 is piecewise constant over its leaves L0m, m = 1, . . . , L0, we take S to be the intersection of C with each L0m in turn and apply Proposition 2. The overall maximum is then obtained as the maximum over the leaves, � �
This reduces (1) to solving 2L0 instances of (6).
Additive f0. For this case, we make the additional assumption that the link function g in (5) is the identity function, as well as Assumption 2 below. The implication of these assumptions is discussed in Appendix B.2. Assumption 2. D(y, y0) = D(y −y0) is a function only of the difference y −y0. Then f0(x) = �d j=1 f0j(xj) and the difference f(x) −f0(x) is also additive. Using Assumptions 2, 1 and a similar argument as in the proof of Proposition 2, the maximum deviation is again obtained by maximizing and minimizing an additive function, resulting in two instances of (6) with S = C: � �
Then f0(x) = �d j=1 f0j(xj) and the difference f(x) −f0(x) is also additive. Using Assumptions 2, 1 and a similar argument as in the proof of Proposition 2, the maximum deviation is again obtained by maximizing and minimizing an additive function, resulting in two instances of (6) with S = C: � �
Computational complexity of (6). For the case of nonlinear additive f, we additionally assume that C is a Cartesian product. It follows that S = �d j=1 Sj is a Cartesian product (see Appendix B.2 for the brief justification) and (6) separates into one-dimensional optimizations over Sj,
The computational complexity of (8) is thus �d j=1 Cj, where Cj is the complexity of the jth onedimensional optimization. We discuss different cases of Cj in Appendix B.2; the important point is that the overall complexity is linear in d. In the GLM case where �d j=1 fj(xj) = wT x, problem (6) is simpler and it is less important that C be a Cartesian product. In particular, if C is a convex set, so too is S (again see Appendix B.2 for justification). Hence (6) is a convex optimization problem.
# 4.3 Tree ensembles
We now extend the idea used for single decision trees in Section 4.1 to tree ensembles. This class covers several popular methods such as Random Forests and Gradient Boosted Trees. It can also cover rule ensembles [41, 42] as a special case, as explained in Appendix B.3. We assume f is a tree ensemble consisting of K trees and f0 is a single decision tree. Let Llk denote the lth leaf of the kth tree in f for l = 1, . . . , Lk, and L0m be the mth leaf f0, for m = 1, . . . , L0. Correspondingly let ylk and y0m denote the prediction values associated with each leaf.
(6)
(8)
 GV E V = {lk|∀k = 1, . . . , K, l = 1, . . . , Lk} ∪{m|m = 1, . . . , L0}. Construct an edge for each overlapping pair of leaves in V, i.e. E = {(i, j)|Li ∩Lj ̸= ∅, ∀(i, j) ∈V, i ̸= j}.
This graph is a K + 1-partite graph as leaves within an individual tree do not intersect and are an independent set. Denote M to be the adjacency matrix of G. Following Chen et al. [32], a maximum clique S of size K + 1 on such a graph provides a discrete region in the feature space with a computable deviation. A clique is a subset of nodes all connected to each other; a maximum clique is one that cannot be expanded further by adding a node. The model predictions yc and y0c can be ensembled from leaves in S. Denote by D(S) the deviation computed from the clique S. Maximizing over all such cliques solves (1). However, complete enumeration is expensive, so informative bounds, either using the merge procedure in Chen et al. [32] or the heuristic function in Devos et al. [33] can be used. We use the latter which exploits the K + 1-partite structure of G. Specifically, we adapt the anytime bounds of Devos et al. [33] as follows. At each step of the enumeration procedure, an intermediate clique S contains selected leaves from trees in [1, . . . , k] and unexplored trees in [k + 1, . . . , K + 1]. For each unexplored tree, we select a valid candidate leaf that maximizes deviation, i.e.
Using these worst-case leaves, a heuristic function
provides an upper (dual) bound. In practice, this dual bound is tight and therefore very useful during the search procedure to prune the search space. Each K + 1 clique provides a primal bound, so the search can be terminated early before examining all trees if the dual bound is less than the primal bound. We adapt the search procedure of Mirghorbani and Krokhmal [43] to include the pruning arguments. Appendix B.3 presents the full algorithm. Starting with an empty clique, the procedure adds a single node from each tree to create an intermediate clique. If the size of the clique is K + 1 the primal bound is updated. Otherwise, the dual bound is computed. A node compatibility vector is used to keep track of all feasible additions.When the search is terminated at any step, the maximum deviation is bounded by (Dlb, Dub).
The algorithm works for the entire feature space. When the certification set C is a union of balls as in (2), some additional considerations are needed. First, we can disregard leaves that do not intersect with C during the graph construction phase. A validation step to ensure that the leaves of a clique all intersect with the same ball in C is also needed.
# 4.4 Piecewise Lipschitz Functions
We saw the benefits of having specific (deterministic) interpretable functions as well as their extensions in the context of safety. Now consider a richer class of functions that may also be randomized with finite variance. In this case let f and f0 denote the mean values of the learned and reference functions respectively. We consider the case where each function is either interpretable or black box, where the latter implies that query access is the only realistic way of probing the model. This leads to three cases where either both functions are black box or interpretable, or one is black box. What we care about in all these cases1 is to find the maximum (and minimum) of a function ∆(x) = f(x) −f0(x). Let us consider finding only the maximum of ∆as the other case is symmetric. Given that f and f0 can be random functions ∆is also a random function and if ∆is black box a standard way to optimize it is either using Bayesian Optimization (BO) [44] or tree search type bandit methods [45, 46]. We repurpose some of the results from this latter literature in our context showcasing the benefit of interpretability from a safety standpoint. To do this we first define relevant terms.
(10)
(11)
(12)
Definition 1 (Simple Regret [45]). If f ∗ C denotes the optimal value of the function f on the certification set C, then the simple regret rC q after querying the f function q times and obtaining a solution xq is given by, rC q (f) = f ∗ C −f(xq). Definition 2 (Order β c-Lipschitz). Given a (normalized) metric ℓa function f is c-Lipschitz continuous of order β > 0 if for any two inputs x, y and for c > 0 we have, |f(x) −f(y)| ≤ c · ℓ(x, y)β. Definition 3 (Near optimality dimension [45]). If N(C, ℓ, ϵ) is the maximum number of ϵ radius balls one can fit in C given the metric ℓand Cϵ = {x ∈C|f(x) ≥f ∗ C −ϵ}, then for c > 0 the c-near optimality dimension is given by, υ = max � lim supϵ→0 ln N (Ccϵ,ℓ,ϵ) ln(ϵ−1) , 0 � . Intuitively, simple regret measures the deviation between our current best and the optimal solution. The Lipschitz condition bounds the rate of change of the function. Near optimality dimension measures the set size for which the function has close to optimal values. The lower the value of υ, the easier it is to find the optimum. We now define what it means to have an interpretable function. Assumption 3 (Characterizing an Interpretable Function). If a function f is interpretable, then we can (easily) find 1 ≤m ≪n partitions {C(1), ..., C(m)} of the certification set C such that the function f (i) = {f(x)|x ∈C(i)} ∀i ∈{1, ..., m} in each partition is c-Lipschitz of order β. Note that the (interpretable) function overall does not have to be c-Lipschitz of bounded order, rather only in the partitions. This assumption is motivated by observing different interpretable functions. For example, in the case of decision trees the m partitions could be its leaves, where typically the function is a constant in each leaf (c = 0). For rule lists as well a fixed prediction is usually made by each rule. For a linear function one could consider the entire input space (i.e. m = 1), where for bounded slope α the function would also satisfy our assumption (c = α and β = 1). Examples of models that are not piecewise constant or globally Lipschitz are oblique decision trees (Murthy et al., 1994), regression trees with linear functions in the leaves, and functional trees. Moreover, m is likely to be small so that the overall model is interpretable (viz. shallow trees or small rules). With the above definitions and Assumption 3 we now provide the simple regret for the function ∆. 1. Both black box models: If both f and f0 are black box then it seems no gains could be made in estimating the maximum of ∆over standard results in bandit literature. Hence, using Hierarchical Optimistic Optimization (HOO) with assumptions such as C being compact and ∆being weakly Lipschitz [45] with near optimality dimension υ the simple regret after q queries is: �  �
2. Both interpretable models: If both f and f0 are interpretable, then for each function based on Assumption 3 we can find m1 and m0 partitions of C respectively where the functions are c1 and c0-Lipschitz of order β1 and β0 respectively. Now if we take non-empty intersections of these partitions where we could have a maximum of m1m0 partitions, the function ∆in these partitions would be c = 2 max(c0, c1)-Lipschitz of order β = min(β0, β1) as stated next (proof in appendix). Proposition 3. If functions h0 and h1 are c0 and c1 Lipschitz of order β0 and β1 respectively, then the function h = h0 −h1 is c-Lipschtiz of order β, where c = 2 max(c0, c1) and β = min(β0, β1). Given that ∆is smooth in these partitions with underestimated smoothness of order β, the simple regret after qi queries in the ith partition C(i) with near optimality dimension υi based on HOO is: rC(i) qi (∆) ≤O � q−1/(υi+2) i � , where υi ≤ d β . If we divide the overall query budget q across the π ≤m0m1 non-empty partitions equally, then the bound will be scaled by π1/(υi+2) when expressed as a function of q. Moreover, the regret for the entire C can then be bounded by the maximum regret across these partitions leading to �  �
  Notice that for a model to be interpretable m0 and m1 are likely to be small (i.e. shallow trees or small rule lists or linear model where m = 1) leading to a “smallish" π and υ can be much >> d β in case 1. Hence, interpretability reduces the regret in estimating the maximum deviation.
(13)
(14)
3. Black box and interpretable model: Making no further assumptions on the black box model and assuming ∆satisfies properties mentioned in case 1, the simple regret has the same behavior as (13). This is expected as the black box model could be highly non-smooth.
# 5 Case Studies
We present case studies to serve three purposes: 1) show that deviation maximization can lead to insights about models, 2) illustrate the maximization methods developed in Section 4, and 3) quantify the dependence of the maximum deviation on model complexity and certification set size (mostly in Appendix D). Two datasets are featured: a sample of US Home Mortgage Disclosure Act (HMDA) data (see Appendix D.2 for details), meant as a proxy for a mortgage approval scenario, and the UCI Adult Income dataset [47], a standard tabular dataset with mixed data types. A subset of results is shown in this section with full results, experimental details, and an additional Lending Club dataset in Appendix D. Since these are binary classification datasets, we take the deviation function D to be the absolute difference between predicted probabilities of class 1. For the certification set C, we consider a union of ℓ∞balls (2) centered at test set instances. While we have used the test set here, any not necessarily labelled dataset would suffice. The case r = 0 yields a finite set consisting only of the test set, while r →∞corresponds to C being the entire domain X. We reiterate that the dependence of the certification set on a chosen dataset is only on (an expanded version of) the support of the dataset and not on other aspects of the data distribution. To demonstrate insights from deviation maximization, we study the solutions that maximize deviation (the arg max in (1)) and discuss three examples below.
Identification of an artifact: The first example comes from the Adult Income dataset, where the reference model f0 is a decision tree (DT) and f is an Explainable Boosting Machine (EBM) [48], a type of GAM (plots of both in Appendix D.3). Here, the capital loss feature is the largest contributor to the maximum deviation (the discussion below Table 4 explains how this is determined), and Table 8 in Appendix D.3 shows that as the certification set radius r increases, the maximizing values of capital loss converge to the interval [1598, 1759]. The plot of the GAM shape function fj for capital loss in Figure 1 shows that this interval corresponds to a curiously low value of the function. This low region may be an artifact warranting further investigation since it seems anomalous compared to the rest of the function, and since individuals who report capital losses on their income tax returns to offset capital gains usually have high income (hence high log-odds score). Note that this potential artifact was automatically identified through deviation maximization. For the next two examples, we consider a simplified mortgage approval scenario using the HMDA dataset. Suppose that a DT f0 (shown in Figure 3 in Appendix D.2) has been trained to make final decisions on mortgage applications. Domain experts have determined that this DT is sensible and safe and are now looking to improve upon it by exploring EBMs. (Logistic regression (LR) models are deferred to Appendix D.2 because they do not have higher balanced accuracy than f0.) Conflict between f, f0: We first examine solutions that result in the most positive difference between the predicted probabilities of an EBM f with parameter max_bins=32 and the DT f0. These all occur in a leaf of f0 (leaf 2 in Figure 3) where the applicant’s debt-to-income (DTI) ratio is too high (> 52%) and f0 predicts a low probability of approval. The other salient feature of the solutions is that they all have ‘preapproval’=1, indicating that a preapproval was requested, which is given a large weight by the EBM f in favor of approval (see Figure 4 in Appendix D.2, and Table 4 for more feature values). Thus f and f0 are in conflict. Among different ways in which the conflict could be resolved, a domain expert might decide that f0 remains correct in rejecting risky applicants with high DTI, even if there is a preapproval request and the new EBM model puts a high weight on it. Trend toward extreme points, deviation can be good: We now look at solutions that yield the most negative difference between the predicted probabilities of f and f0, for r ≤0.8. Table 1 shows the 6 features that contribute most to the deviation (again see Appendix D.2 for details). All of these points lie in a leaf of f0 (leaf 14 in Figure 3, denoted L14) that excludes several clearer-cut cases, with the result being a less confident predicted probability of 0.652 from f0. The feature values that maximize deviation tend toward extreme points of the region L14. Specifically, the values of the continuous features debt-to-income ratio, loan-to-value ratio, property value, and income all move in the direction of application denial. For the latter three features, the boundary of L14 is reached as soon as r = 0.1, whereas for debt-to-income ratio, this occurs at r = 0.4. The movement
r
debt_to_income (%)
state
loan_to_value (%)
aus_1
prop_value (000$)
income (000$)
0.0
46.0
CA
95.0
3
415
77.0
0.1
[45.9 46.5]
none
[100. 100.92]
1
[120 120]
≤28.5
0.2
[45.5 46.5]
none
[100. 100.92]
1
[120 120]
≤28.5
0.4
[52. 52.]
none
[100. 100.92]
1
[120 120]
≤28.5
0.6
[52. 52.]
none
[100. 100.92]
1
[120 120]
≤28.5
0.8
[52. 52.]
none
[100. 100.92]
1
[120 120]
≤28.5
Table 1: Values of top 6 features that maximize difference in predicted probabilities between a decision tree reference model f0 and an Explainable Boosting Machine f (max_bins = 32) on the HMDA dataset. For radius r > 0, the maximizing values of continuous features form an interval because the corresponding EBM shape functions fj are piecewise constant.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1f18/1f185222-804d-41d6-adf2-5ab1dac0ac32.png" style="width: 50%;"></div>
<div style="text-align: center;">0 - 17 174 348 523 697 871 1.05K 1.22K 1.39K 1.57K 1.74K 1.92K 2.09K 2.27K 2.44K 2.61K 2.79K 2.96K 3.14K 3.31K 3.48K 3.66K 3.83K 4.01K 4.18K 0 10k 20k 30k Density Figure 1: EBM shape function fj for capital loss feature, showing anomalously low interval identified by deviation maximization.</div>
0 - 17 174 348 523 697 871 1.05K 1.22K 1.39K 1.57K 1.74K 1.92K 2.09K 2.27K 2.44K 2.61K 2.79K 2.96K 3.14K 3.31K 3.48K 3.66K 3.83K 4.01K 4.18K 0 10k 20k 30k Density Figure 1: EBM shape function fj for capital loss feature, showing anomalously low interval identified by deviation maximization.
toward extremes is expected for this f since its relevant shape functions fj are mostly increasing or decreasing, as seen in Figure 4 in Appendix D.2. The same behavior is observed in other GAM and LR examples in Appendix D. In this example, a domain expert might conclude that the large deviation is in fact desirable because f is providing varying predictions in L14 in ways that make sense, as opposed to the constant given by f0. This shows that deviation maximization can work in both directions, identifying where the reference model and its assumptions may be too simplistic and giving an opportunity to improve the reference model. Maximum deviation vs. number of trees in a RF: In Figure 2, we highlight one result from a set of such results in Appendix D, showing maximum deviation as a function of model complexity, here quantified by the number of estimators (trees) in a Random Forest (RF). This is a demonstration of the method in Section 4.3, which in general provides bounds on the maximum deviation. In this case, the upper (“dual”) bound is informative enough to actually show a decrease as the number of estimators increases. The larger number of estimators increases averaging and may serve to make the model smoother.
# 6 Conclusion
We have considered the relationship between interpretability and safety in supervised learning through two main contributions: First, the proposal of maximum deviation as a means toward assessing safety, and second, discussion of approaches to computing maximum deviation and how these are simplified by interpretable model structure. We believe that there is much more to explore in this relationship. Appendices C and B.5 provide further discussion of several topics and future directions.
# Acknowledgements
We thank Michael Hind for several early discussions on the topic of ML model risk assessment tha inspired this work, and for his overall leadership on this topic. We also thank Dhaval Patel for a discussion on the industrial assets example in Section 2.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/255e/255e88b9-853c-4039-a7ff-57da29222576.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Maximum deviation D on the Adult Income dataset as a function of number of estimators in a Random Forest.</div>
# References
# Checklist
# A Additional Problem Formulation Details
Deviation function D For the case where the inputs y, y0 to D are real-valued scalars (which covers binary classification and regression), while Assumption 1 was stated as a sufficient condition for tractable optimization with GAMs, it is also an intuitively reasonable requirement: the deviation should increase the farther y is from y0 in either direction. In addition, symmetry may be desirable, i.e. D(y, y0) = D(y0, y), to not favor one of the two models over the other. Both Assumptions 1 and 2 as well as symmetry are satisfied by monotonically increasing functions D(|y −y0|) of the absolute difference, for example powers |y −y0|p for p > 0. For the case where y, y0 ∈RM as in multi-class classification, it may be advantageous for D(y, y0) to decompose into a sum over output dimensions: D(y, y0) = �M k=1 Dk(yk, y0k), where yk, y0k are the components of y, y0. For example, the pth power of the ℓp distance ∥y −y0∥p p = �M k=1 |yk −y0k|p is separable in this manner.
  Models that abstain The formulation in Section 2 can also accommodate models that abstain from predicting (and possibly defer to a human expert or other fallback system). If f(x) = ∅, representing abstention, then we may set D(∅, y0) = d for any y0 ∈Y, where d > 0 is an intermediate value less than the maximum value that D can take [49]. The value d might also be less than a “typically bad” value for D, to reward the model for abstaining when it is uncertain.
# B.1 Trees
Rule lists A rule list as defined by Yang et al. [39], Angelino et al. [40] is a nested sequence of IF-THEN-ELSE statements, where the IF condition is a conjunctive rule and the THEN consequent is an output value. If each rule involves a single feature (i.e., the conjunctions are of degree 1), such rule lists are one-sided trees in the sense of Section 4.1. The number of leaves in the equivalent tree is equal to the number of rules in the list (including the last default rule).
Intersection of C with a Cartesian product If C = �d j=1 Cj is also a Cartesian product, then determining whether the intersection is non-empty amounts to checking whether all of the coordinatewise intersections with Cj, j = 1, . . . , d, are non-empty. If C is not a Cartesian product but is a union of ℓ∞balls (which are Cartesian products), then the intersection is non-empty if the intersection with any one ball is non-empty.
Relationship between Proposition 1 and [32, Theorem 1] In the case of a K = 2-tree ensemble, [32, Theorem 1] bounds the complexity of exact robustness verification as min{O(n2), O((4n)d} = O(n2), where n is the maximum number of leaves in a tree and we assume that the feature dimension d ≥2. In Proposition 1, we account for the possibly different numbers of leaves L and L0 in the two trees f and f0, and we exactly enumerate the edges, |E| ≤L0L ≤n2.
Additive reference model For the case where f is a decision tree and f0 is a generalized additive model, if the deviation function is symmetric, D(y, y0) = D(y0, y), then this case is covered in Section 4.2.
# B.2 Linear and Additive Models
Categorical features A function fj(xj) of a categorical feature xj can be represented in two ways, depending on whether f is a GAM or a GLM. In the GAM case, we may use the native representation in which xj takes values in a finite set Xj of categories. In the GLM case, xj is one-hot encoded into multiple binary-valued features xjk, one for each category k. Then any function fj can be represented as a linear function,
where wjk is the value of fj for category k.
Implication of Assumption 1 The second condition implies that the deviation increases or stays the same as y moves away from y0 in either direction.
 � D(f(x), f0(x)) = D(g−1(g(y0)), y0) = D(y0, y0) = 0.
As S(x) increases from g(y0), f(x) also increases because g−1 is an increasing function, and D(f(x), y0) increases or stays the same due to Assumption 1.2. Similarly, as S(x) decreases from g(y0), f(x) decreases, and D(f(x), y0) again increases or stays the same. It follows that to maximize D(f(x), y0), it suffices to separately maximize and minimize S(x), compute the resulting values of D(f(x), y0), and take the larger of the two. This yields the result. Implication of Assumption 2 and identity link function g These two assumptions imply that the deviation is measured on the difference between f and f0 in the space in which they are additive. For example, if f and f0 are logistic regression models predicting the probability of belonging to one of the classes, the difference is taken in the log-odds (logit) domain. It is left to future work to determine other assumptions under which problem (1) is tractable when f and f0 are both additive. Cartesian product C implies Cartesian product S In the cases of constant and additive f0, S = C. In the decision tree case, since each leaf is a Cartesian product L0m = �d j=1 Rmj, the intersections S = L0m ∩C are also Cartesian products �d j=1 Sj where Sj = Rmj ∩Cj. One-dimensional optimization complexities Cj For discrete-valued xj, Cj is proportional to the number of allowed values |Sj|. For continuous xj, it is common to use spline functions or tree ensembles as fj in constructing GAMs. In the former case, Cj is proportional to the number of knots. In the latter, the tree ensemble can be converted to a piecewise constant function and Cj is then proportional to the number of pieces. Lastly in the case where fj(xj) = wjxj is linear and Sj = [Xj, Xj] is an interval, Cj = O(1) because it suffices to evaluate the two endpoints. Convex S If C is a convex set, then in the cases of constant and additive f0, S = C is also convex. In the case of tree-structured f0, S = L0m ∩C and each leaf L0m can be represented as a convex set, with interval constraints on continuous features and set membership constraints on categorical features. The latter can be represented as xjk = 0 constraints on the one-hot encoding (see “Categorical features” paragraph above) for non-allowed categories k. Hence S is also convex. As a specific example, suppose that S is the product of independent constraints on each categorical feature and an ℓp norm constraint on the continuous features jointly. The maximization over each categorical feature has complexity Cj = |Sj| as noted above, while the maximization of wT x over continuous features lying in an ℓp ball has closed-form solutions for the common cases p = 1, 2, ∞. Relaxations of the Cartesian product assumption If the certification set C is not a Cartesian product, then one way to still bound the maximum deviation is to find the smallest Cartesian product C that contains C and maximize deviation over C. As long as it is relatively easy to optimize linear functions over C, then constructing such a Cartesian product is similarly easy. Another conceivable relaxation of the Cartesian product assumption is a Cartesian product of low-dimensional sets, not just one-dimensional.
# B.3 Tree Ensembles
The full algorithm for clique search from Section 4.3 is presented in Algorithm 1. It uses Z as a node compatibility vector to keep track of valid leaves and B a set of trees/partites not yet covered by the maximum clique. The algorithm starts with and empty clique S and anytime bounds as 0. It starts the search with the smallest tree to limit the search space. This is typically f0. Each leaf is added to the intermediate clique S in turn (Line 6). A stronger primal bound can be achieved if the traversal is ordered in a meaningful way. In particular, starting with nodes with the highest heuristic function value H(S) aids the algorithm to focus on better areas of the search space.
If the size of the clique is K + 1 the primal bound is updated. Otherwise, the dual bound is computed. If the node is promising, the algorithm recurses to the next level. When the search is terminated at any step, the maximum deviation is bounded by (Dlb, Dub).
Algorithm 1 Max clique search for maximum deviation
Require: M adjacency matrix, H heuristic function
1: Z[i] = 1∀i ∈V , B = {1, 2, . . . , K + 1}, S = ∅
▷All nodes valid, all trees uncovered
2: Q = Enumerate(Z, B, S)
3: Initialize:Dlb = 0, Dub = 0
▷Anytime bounds
4: function Enumerate(Z, B, S):
5: t = arg maxb{|Zb|
��b ∈B}
▷Uncovered tree with fewest valid nodes
6: for i in Zt do
7:
Z[i] = 0
▷Mark node as incompatible
8:
S = S ∪{i}
▷Add to candidate clique
9:
if |S| = K + 1 then
10:
Dlb = max (Dlb, D(S))
▷Update primal bound
11:
Q = Q ∪S
▷Add to set of max cliques
12:
S = S \ {i}
▷Backtrack
13:
else
14:
Zt+1 = Zt ∧M(i)
▷Update valid nodes
15:
B = B \ {t}
▷Update uncovered trees
16:
Dub = max (Dub, H(S))
▷Update dual bound
17:
if Dub > Dlb then
18:
Enumerate(Zt+1, B, S)
▷Recurse to next level
19:
end if
20:
S = S \ {i}
▷Backtrack
21:
B = B ∪{t}
22:
end if
23: end for
Rule ensembles Similar to the tree ensembles considered in Section 4.3, a rule ensemble is a linear combination of conjunctive rules, where the antecedent is a conjunction of conditions on individual features, and the consequent takes a real value if the antecedent is true and zero otherwise. They are produced by algorithms such as SLIPPER [50], that of Rückert and Kramer [51], RuleFit [41], ENDER [42] and have also been referred to as generalized linear rule models [52]. A rule ensemble can be converted into a tree ensemble by converting each conjunctive rule into an IF-THEN-ELSE rule list, which is a one-sided tree (see Appendix B.1). Specifically, the conditions in the conjunction are taken in any order, each condition is negated to become an IF condition, and the THEN consequents are all output values of zero. The final ELSE consequent, which is reached if all the IF conditions are false (and hence the original rule holds), returns the output value of the original rule. The number of leaves in the resulting tree equals the number of conditions in the conjunction plus one.
# B.4 Piecewise Lipschitz Functions
|h(x) −h(y)| = |(h0 −h1)(x) −(h0 −h1)(y)| = |h0(x) −h0(y) + h1(y) −h1(x)| ≤|h0(x) −h0(y)| + |h1(x) −h1(y)| ≤c0 · ℓ(x, y)β0 + c1 · ℓ(x, y)β1 ≤c · ℓ(x, y)β
where, c = 2 max(c0, c1) and β = min(β0, β1).
Other choices for D(., .): The results assumed D(., .) to be the identity function, where ∆= D(f0, f). This choice of function clearly satisfies Assumptions 1 and 2. Again consistent with these assumptions we look at some other choices for D(., .). If D(., .) were an affine function with a positive scaling such as D(y0, y) = α(y0 −y) + b where α > 0, then our result in equation 14 would be unchanged as only the Lipschitz constant of ∆would change, but not its (underestimated) order
If the function were a polynomial or exponential however, no such guarantees can be made and we would be back to case 1.
# 5 Other Model Classes: Neural Networks and Post Hoc Explanatio
For model classes beyond the ones discussed in Section 4, it appears to be a greater challenge to obtain reasonably tractable algorithms that guarantee exact computation of or bounds on the maximum deviation. Here we outline some future directions for neural networks and post hoc explanations.
Robustness verification for neural networks has attracted a great deal of attention and made considerable progress, with exact approaches including satisfiability modulo theory [24] and mixed integer programming [25, 26], and incomplete methods that compute bounds using bound propagation [22, 19], linear programming and duality [18, 20, 21], and semidefinite programming [23, 27]. However, all of these methods consider a single model, effectively comparing it to a constant. Robustness verification is thus essentially a single-model case of our problem (1) in which f0 is a constant (and with an appropriate choice of the deviation function D(f, f0)). While we may expect that solutions to a two-model verification problem would leverage existing robustness verification methods, developing such solutions remains for future work. Furthermore, evaluation of robustness verification methods has largely been limited to local neighborhoods around input points (with typical radii ϵ ≤0.1 in terms of normalized feature values). This limitation may also need to be addressed to enable evaluation of maximum deviation in the way envisioned in this paper. It is also natural to ask whether post hoc explanations for the model can help. One way in which this could occur is if the post hoc explanation approximates the model f by a simpler model ˆf and if the deviation function D satisfies the triangle inequality D(f(x), f0(x)) ≤D(f(x), ˆf(x)) + D( ˆf(x), f0(x)). Then the maximum deviation in (1) would be bounded as
max x∈C D(f(x), f0(x)) ≤max x∈C D(f(x), ˆf(x)) + max x∈C D( ˆf(x), f0(x)).
While we may choose ˆf to be interpretable so that the rightmost maximization is tractable, the middle maximization asks for a uniform bound on the deviation between f and ˆf, i.e, the fidelity of ˆf. We are not aware of a post hoc explanation method that provides such a guarantee. Indeed, in general, the middle maximization might not be any easier than the left-hand one that we set out to bound. A (practical) possibility may be to perform quantile regression [53] for a large enough quantile to learn ˆf, as opposed to minimizing expected error as is typically done. This may be an interesting direction to explore in the future as quantile regression algorithms are available for varied model classes including linear models, tree ensembles [54] and even neural networks [55]. More investigation is needed into whether quantile regression methods can provide approximate guarantees on the middle term in (15). Assuming that uniform proxies in the above sense can be constructed, then for certain modalities or applications it may be possible to train highly accurate proxies. For instance for tabular data, Random Forests or boosted trees might very well replicate the behavior of a neural network, in which case the machinery introduced in Section 4.3 could be used. Even for other modalities such as text and images, interpretable models such as Neural Additive Models (NAMs) [56] and continued fraction networks (CoFrNets) [57] may prove to be sufficient in some cases. Finally, there are recent architectures such as Lipschitz neural networks [58] which are adversarially robust and hence valuable in practice. Our analysis presented in Section 4.4 for piecewise Lipschitz models would be applicable here, where the simple regret of standard bandit algorithms for a given number of queries could be reduced to (14) as opposed to (13).
While we may choose ˆf to be interpretable so that the rightmost maximization is tractable, the middle maximization asks for a uniform bound on the deviation between f and ˆf, i.e, the fidelity of ˆf. We are not aware of a post hoc explanation method that provides such a guarantee. Indeed, in general, the middle maximization might not be any easier than the left-hand one that we set out to bound.
Finally, there are recent architectures such as Lipschitz neural networks [58] which are adversarially robust and hence valuable in practice. Our analysis presented in Section 4.4 for piecewise Lipschitz models would be applicable here, where the simple regret of standard bandit algorithms for a given number of queries could be reduced to (14) as opposed to (13).
# C Further Discussion
Worst-case approach The formulation of (1) as the worst case over a certification set represents a deliberate choice to depend as little as possible on a probability distribution or a dataset sampled from one. As stated in Section 2, Certification Set paragraph, C can depend at most on (an expanded version of) the support set of a distribution. The reason for this choice is because safety is an out-ofdistribution notion: harmful outputs often arise precisely because they were not anticipated in the
(15)
White-box vs. grey-box models In this paper, we have assumed full “white-box” access to both f and f0, namely complete knowledge of their structure and parameters. Interesting questions may arise when this assumption is relaxed to different “grey-box” possibilities. For example, one could further investigate the third case in Section 4.4, where one of f, f0 is black-box and the other is white-box interpretable. There may exist assumptions that we have not identified that would improve the query complexity compared to generic black-box optimization.
Other interpretability-safety relationships This paper has focused on one relationship between the interpretability of a model and the safety of its outputs. It has not addressed other ways in which interpretability/explainability can affect the risk of a model (in the plain English sense, not the expectation of a loss function). For example, in regulated industries such as consumer finance, not providing explanations or providing inadequate ones can lead to legal, financial, and reputational risks. On the other hand, providing explanations is associated with its own risks [60]. These include the leakage of personal information or model information (intellectual property), an increase in appeals of decisions for the decision-making entity, and strategic manipulation of attributes (i.e. “gaming”) by individuals to gain more favorable outcomes.
Applicability to RL settings In RL, if one views the actions as labels and state representation as features, one can build a tree, albeit likely a deep/wide one, to represent exactly the RL policy, where the probability distribution over the actions can be viewed as the class distribution in a normal supervised setting. Rolling up the states, creating leaves with multiple states, and simply averaging the probabilities for each action would yield smaller trees that approximate the policy. Our work lays a foundation where in principle we can also compare f and f0 that are policies using such tree
representations. This may be related to a popular global explainability method [61] that samples policies and builds trees to explain them.
Ethics The safety of machine learning systems has been called out by the European Commission’s regulatory framework [62]. The commission states seven key dimensions to be evaluated and audited by a cross-disciplinary team: (i) human agency and oversight, (ii) technical robustness and safety, (iii) privacy and data governance, (iv) transparency, (v) diversity, non-discrimination and fairness, (vi) environmental and societal well-being, and (vii) accountability. The second of these dimensions is safety. However, Sloane et al. [63] argue that algorithmic audits are ill-defined as the underlying definitions are vague. The proposed work helps fill that ill-definedness using a quantitative approach. One may argue against this particular choice of quantification, but it does start the community down the path toward being more concrete in its definitions. As with many other technologies, the proposed approach may be misused. For example, the reference model may be chosen in a way that hides the safety concerns of the model being evaluated. Transparent documentation and reporting with provenance guarantees can help avoid this kind of purposeful deceit [64].
# D Experiment Details and Additional Results
# D.1 General Experiment Details
Data processing We use the training set of each dataset to train models and the test set as the basis for evaluating maximum deviation, specifically as the set of centers for the ℓ∞balls in (2). Continuous features are standardized and categorical features are one-hot encoded. The ℓ∞norm is computed on the resulting normalized feature values.
Models We use scikit-learn [65] to train decision tree (DT), logistic regression (LR), and Random Forest (RF) models. The corresponding complexity parameters are the number of leaves for DT (parameter max_leaf_nodes), the amount of ℓ1 regularization for LR (inverse ℓ1 penalty C), and the number of estimators/trees for RF (n_estimators). For additive models, we use Explainable Boosting Machines (EBM) from the InterpretML package [48] with zero interaction terms (so that the models are indeed additive). Smoothness is controlled by the max_bins parameter, the number of discretization bins for continuous features.
Deviation maximization In all cases, when the certification set radius r = 0, maximum deviation can be computed simply by evaluating the models on the test set. For the case where r > 0, f is a DT or RF, and f0 is a DT, Algorithm 1 is used (on a bipartite graph if f is a DT). The cases where f is LR or EBM fall under the generalized additive case of Section 4.2. Given that f0 is a DT, we use (7), (6) to determine the maximum deviation. For r < ∞when C is a union of ℓ∞balls, we maximize separately over each intersection between a ball and a leaf of f0 and then take the maximum over the intersections.
Computation All experiments were run on CPU nodes with 64GB memory. For decision trees and tree ensembles run times of Algorithm 1 were limited to 2 hours, after which the best available bounds were used.
# D.2 Home Mortgage Disclosure Act Dataset
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b62e/b62e4fc0-79ea-43ef-b17b-321546f56b02.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/36d8/36d8f32c-24f7-47ed-bc4e-5578261ffd9a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Decision tree reference model with 8 leaves for the HMDA dataset.</div>
• Drop geographical columns that have too many unique values (e.g., census tract). • Drop rows that have null or “exempt” values in key features such as loan-to-value ratio, debt-to-income ratio, property value, income. • Take a 10% sample of the records remaining after the above processing, to make experimentation less time- and resource-consuming. • Split the subsampled dataset 80%–20% into training and test sets.
The dataset resulting from the above processing is imbalanced, with nearly 93% of mortgage applications approved. Thus in training models, we balance the classes by weighting, either using the class_weight=‘balanced’ option in scikit-learn or defining sample weights for the same purpose. On the test set, we evaluate balanced accuracy instead of accuracy.
Reference model Figure 3 depicts the 8-leaf DT reference model used in the experiments on the HMDA dataset. This DT has a test set balanced accuracy of 70.9% and area under the receiver operating characteristic (AUC) of 0.750. The top-level split is based on debt-to-income ratio, a measure often used in lending. Other common mortgage measures such as loan-to-value ratio, property value, and whether a preapproval was requested (value 2 means no) also appear. ‘aus_1’=5 and ‘aus_1’=6 refer to the automated underwriting system used to evaluate the application, with values 5 and 6 denoting “other” and “not applicable”.
LR and GAM models In Tables 2 and 3, we show the values of inverse ℓ1 penalty C and max_bins that were used for LR and GAM respectively, as well as statistics of the resulting classifiers. Test set balanced accuracy and AUC increase and reach a plateau. For LR, we take the ℓ1 norm of the coefficients to be the main measure of smoothness as it depends on both the number of nonzero coefficients as well as their magnitudes, which both affect the extreme values attained in (6). Since the LR balanced accuracy and AUC are not higher than those of the reference DT, we focus less on LR in what follows.
C
nonzeros
ℓ1 norm
bal. acc.
AUC
1e-4
2
0.4
0.605
0.663
3e-4
7
1.3
0.633
0.695
1e-3
17
4.6
0.663
0.736
3e-3
26
7.7
0.669
0.744
1e-2
42
12.7
0.674
0.750
3e-2
68
16.6
0.675
0.751
1e-1
85
20.3
0.675
0.752
3e-1
96
22.2
0.676
0.752
1e+0
99
22.9
0.675
0.752
3e+0
100
23.2
0.675
0.752
nzero coefficients, norm of coefficients, test set
Table 2: Number of nonzero coefficients, ℓ1 norm of coefficients, test set balanced accuracy, and area under the receiver operating characteristic (AUC) for logistic regression models on the HMDA dataset as a function of inverse ℓ1 penalty C.
max_bins
bal. acc.
AUC
4
0.669
0.743
8
0.695
0.772
16
0.711
0.785
32
0.720
0.798
64
0.723
0.799
128
0.722
0.800
256
0.722
0.800
512
0.723
0.800
1024
0.723
0.800
For EBM, based on Table 3, we select max_bins = 32 as a representative model with balanced accuracy and AUC nearly equal to the maximum attainable values. Plots for this EBM model are shown in Figure 4. First note that whether a preapproval was requested (value 1 means yes) is quite predictive of final approval. The shape functions for the four continuous features debt-to-income ratio, loan-to-value ratio, property value, and income are mostly monotonic and agree with domain knowledge. The log-odds of mortgage approval decrease as debt-to-income ratio and loan-to-value ratio increase, with abrupt drops around 50% for debt-to-income ratio and at 80% and 100% for loan-to-value ratio. For property value and income, after a minimum value is reached, the shape function increases rapidly and then stays more or less constant for high property values and incomes.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/24dd/24dd6f73-97d7-4b36-986a-e2172909d8b6.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Feature importances and selected univariate functions fj for the Explainable Boosting Machine with max_bins = 32 on the HMDA dataset.</div>
r
preapproval
appl_credit
loan_to_value
co_appl_credit
intro_rate_period
debt_to_income
_score_type
(%)
_score_type
(%)
0.0
1
2
74.35
10
122.0
55.0
0.1
1
3
[80. 80.]
1
[28.4 43.6]
[53.9 56.1]
0.2
1
1
[29.33 32.42]
1
[57. 60.5]
[52.8 57.2]
0.4
1
7
[ 4.34 32.42]
7
[329.7 360. ]
[52. 53.4]
0.6
1
7
[ 0.49 32.42]
7
[314.5 360. ]
[52. 55.6]
0.8
1
7
[ 0.49 32.42]
7
[299.3 300. ]
[52. 57.7]
0.999
1
7
[ 0.49 32.42]
7
[120.5 163. ]
[52. 53.9]
1.001
1
6
[ 0.49 32.42]
6
[120.5 159.9]
[52. 62.5]
1.2
1
6
[ 0.49 32.42]
6
[120.5 151. ]
[56.9 62.5]
1.4
1
6
[ 0.49 32.42]
6
[120.5 163. ]
[52. 57.3]
1.6
1
6
[ 0.49 32.42]
6
[120.5 163. ]
[52. 60.5]
1.8
1
6
[ 0.49 32.42]
6
[120.5 163. ]
[52. 52.7]
2.0
1
6
[ 0.49 32.42]
6
[120.5 163. ]
[52. 62.5]
∞
1
6
[ 0.49 32.42]
6
[120.5 163. ]
[52. 62.5]
Table 4: Feature values that result in most positive difference in predicted probabilities between an Explainable Boosting Machine f (max_bins = 32) and an 8-leaf decision tree reference model f0 on the HMDA dataset. The 6 features that contribute most are shown as a function of certification set radius r. For radius r > 0, the maximizing values of continuous features form an interval because the corresponding functions fj are piecewise constant.
# Feature combinations that maximize deviation Table 4 shows featu
Feature combinations that maximize deviation Table 4 shows feature values that yield the most positive difference between the predicted probabilities of the EBM f with max_bins= 32 and the 8-leaf DT f0. This table corresponds to the second “Conflict between f, f0” example in Section 5. The 6 features that contribute most to the deviation are shown. These contributions are determined using (8); since the maximum deviation occurs in one of the ℓ∞ball-leaf intersections and this intersection is a Cartesian product, the feature-wise decomposition in (8) applies. The contribution of feature j is then maxxj∈Sj fj(xj). We take an average of the contributions over r to give a single ranking of features for all r. The same method is used to determine feature contributions and choose the top 6 features for Table 1. As mentioned in Section 5, all solutions in Table 4 have ‘preapproval’=1 and debt-to-income ratios > 52% that place them in leaf 2 in Figure 3. The latter results in a low predicted probability of approval from f0 while the former makes a large positive contribution to the probability from f (see Figure 4). The values of the other features also make increasingly larger positive contributions to f as r increases. Loan-to-value ratio decreases, while co-applicant credit score type moves from type 10 (no co-applicant, hence weaker application) to increasingly favorable score types (1, 7, 6, see Figure 4); applicant credit score is similar. This behavior is similar to the movement toward extreme points seen in Table 1.
Dependence on model complexity Figure 5 shows the dependence of maximum deviation on the complexity of model f, quantified by the number of leaves for DTs, coefficient ℓ1 norm for LR, max_bins for EBM, and the number of estimators (trees) for RF. The DT and RF cases demonstrate the methods in Section 4.3, specifically Algorithm 1, where in the RF case, the algorithm may only provide bounds after a time limit of two hours. The plots show that maximum deviation may or may not increase with model complexity. In Figure 5a, the deviation is small for a 10-leaf DT and increases rapidly. Figures 5b and 5c indicate that maximum deviation is sensitive to the ℓ1 norm of LR models but not to the max_bins parameter of EBMs. The latter may increase the resolution of the EBM shape functions but not their dynamic range.
Dependence on certification set size Figure 6 shows the dependence of maximum deviation on the certification set radius r. For LR and GAM, the maximum deviation is greater for r > 0 than for r = 0, showing that evaluation on a finite test set may not be sufficient and infinite certification sets (with r > 0) should be considered, especially to account for unexpected, out-of-distribution deviations. There are jumps at r = 1 because this is the radius that permits values of categorical features of test set points (the ball centers in (2)) to change to any other value. In the case of GAM,
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ca33/ca33b54e-d110-4e25-8606-3621452462c3.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Maximum deviation D on the HMDA dataset as a function of model complexity for (a) DT (number of leaves), (b) LR (ℓ1 norm), (c) GAM (max_bins), and (d) RF (number of estimators).</div>
this jump is sufficient for the deviation to equal that for r = ∞(C = X, dashed line in figure). On the other hand, the deviation for DT and RF remains constant as a function of r.
Running time Figure 7 shows the time required to compute the maximum deviation for LR and GAM on the HMDA dataset. These times were obtained using a single 2.0 GHz core of a server with 64 GB of memory (only a small fraction of which was used) running Ubuntu 16.04 (64-bit). The times increase with the ℓ∞ball radius r because of the increasing number of ball-leaf intersections that become non-empty and hence need to be evaluated. The time for r = 0 is minimal because this case requires only model evaluation over the finite test set, as mentioned. The jumps at r = 1 are due again to the ability of categorical features to change values, leading to an increase in ball-leaf intersections. The filled-in regions show that there was little variation due to different ℓ1 norms for LR or max_bins for GAM. This was most likely because of a vectorized implementation, which operates on all LR coefficients or all GAM bins at once (i.e., without a for loop).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6e7c/6e7c3184-4b51-431b-a9f3-572994a90bd4.png" style="width: 50%;"></div>
Figure 6: Maximum deviation D on the HMDA dataset as a function of certification set radius r for (a) LR and GAM, (b) DT (10 and 50 leaves), (c) RF (5 and 10 estimators). Dashed lines in (a) indicate the r →∞asymptote of the curve of the same color.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/32f6/32f62555-7b70-4ca3-a4b4-04a6ea80aa17.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/39c4/39c427b5-711c-4323-b58a-03135d3d8735.png" style="width: 50%;"></div>
<div style="text-align: center;">0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00  ball radius r</div>
<div style="text-align: center;">Figure 7: Time to compute maximum deviation for logistic regression models (left) and Explainable Boosting Machines (right) on the HMDA dataset as a function of certification set size (radius r). The filled-in region shows the min-max variation with model complexity (ℓ1 norm for LR, max_bins for EBM).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0b72/0b7280ec-b0d7-4f1f-8fd1-75fc0b1014ac.png" style="width: 50%;"></div>
<div style="text-align: center;"> Decision tree reference model with 8 leaves for the Adult Income datase</div>
# D.3 Adult Income dataset
We use the given partition of the Adult Income dataset into training and test sets.
Reference model Figure 8 depicts the 8-leaf DT reference model used in the experiments on the Adult Income dataset. This DT has 85.0% accuracy on the test set. The root node separates individuals based on whether the marital status is Married-civ-spouse. The remaining splits divide the population into those with high and low education, high and low capital gains, and high and low capital losses. In particular, having high capital gains or losses is a good predictor of high income (> $50000).
C
nonzeros
ℓ1 norm
accuracy
AUC
3e-4
1
0.2
0.764
0.715
1e-3
6
2.6
0.825
0.885
3e-3
7
4.7
0.840
0.895
1e-2
16
7.4
0.848
0.900
3e-2
30
12.9
0.852
0.905
1e-1
38
17.3
0.853
0.905
3e-1
62
25.3
0.853
0.905
1e+0
83
40.7
0.852
0.905
3e+0
92
54.6
0.852
0.905
1e+1
101
65.1
0.852
0.904
3e+1
105
73.5
0.852
0.904
1e+2
107
77.6
0.852
0.904
3e+2
107
79.1
0.852
0.904
nzero coefficients, ℓ norm of coefficients, test set 
max_bins
accuracy
AUC
4
0.858
0.910
8
0.862
0.915
16
0.865
0.920
32
0.870
0.924
64
0.871
0.925
128
0.871
0.925
256
0.872
0.925
512
0.871
0.925
1024
0.871
0.925
Table 6: Test set accuracy and AUC for Explainable Boosting Machines on the Adult Income datase as a function of max_bins parameter.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c1fc/c1fc1d6f-52ca-42ab-9994-581a292b93c1.png" style="width: 50%;"></div>
<div style="text-align: center;">igure 9: Coefficient values of the logistic regression model with C = 0.01 (16 nonzeros) for the</div>
Figure 9: Coefficient values of the logistic regression model with C = 0.01 (16 nonzeros) fo Adult Income dataset.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6f58/6f583e4a-2192-42bf-a0a5-4dd466d0a5e2.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10: Feature importances and selected univariate functions fj for the Explainable Boosting Machine with max_bins = 8 on the Adult Income dataset.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4a71/4a717641-b059-485b-87e9-2a332280333c.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 11: Maximum deviation D on the Adult Income dataset as a function of model complexity for (a) DT (number of leaves), (b) LR (ℓ1 norm), and (c) GAM (max_bins).</div>
Dependence on model complexity Figure 11 shows the dependence on model complexity for DT (number of leaves), LR (coefficient ℓ1 norm), and EBM (max_bins). In Figure 11a, the maximum deviation is 0.114 for trees with 9 and 10 leaves, and remains moderate up to 26 leaves, which is different than in Figure 5a. Similar to Figures 5b and 5c, ℓ1 norm has a larger effect on maximum deviation than max_bins (note the vertical scale in Figure 11c). Dependence on certification set size Figure 12a shows the dependence on the certification set radius r for DT, LR, and GAM. The patterns are similar to those in Figure 6: the deviations for LR and GAM increase from r = 0 and have jumps at r = 1, while the deviation for DT remains constant. One difference is that the LR curve in Figure 12a meets its r →∞asymptote (dashed line in figure), similar to GAM. Figure 12b shows the upper bound on the maximum deviation as a function of the certification set size for two RF models. As the test set is large in this case, the deviations observed even for small values of r are high and grow to reach the value of the full feature space quickly.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/488e/488ea82b-0cd3-4cb0-8dea-4e54112be7c1.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 12: (Left) Maximum deviation D on the Adult Income dataset as a function of certification set radius r for DT, LR, and GAM. (Right) Upper bound on maximum deviation of f, a Random Forest, trained on the Adult Income dataset.</div>
Relationships with accuracy and robust accuracy In Figure 13, we show maximum deviation as a function of test set accuracy for the DT, LR, and GAM models shown in Figure 11 (the LR and GAM models are listed in Tables 5 and 6). Broadly, the plots show two regimes: one where accuracy increases and maximum deviation increases moderately or not at all, and one where accuracy stalls while maximum deviation increases. The latter is less desirable as it suggests increasing safety risks without a gain in accuracy. The last branch of the DT curve actually decreases in accuracy, indicating overfitting, while maximum deviation is high.