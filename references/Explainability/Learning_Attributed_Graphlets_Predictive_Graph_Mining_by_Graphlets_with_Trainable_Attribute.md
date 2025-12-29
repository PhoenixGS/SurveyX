# Learning Attributed Graphlets: Predictive Graph Mining by Graphlets with Trainable Attribute
# Tajima Shinji1, Ren Sugihara1, Ryota Kitahara1, and Masayuki Karasuyama∗1
1Nagoya Institute of Technology ∗karasuyama@nitech.ac.jp
1Nagoya Institute of Technology ∗karasuyama@nitech.ac.jp
1Nagoya Institute of Technology ∗karasuyama@nitech.ac.jp
Abstract
The graph classification problem has been widely studied; however, achieving an interpretable model with high predictive performance remains a challenging issue. This paper proposes an interpretable classification algorithm for attributed graph data, called LAGRA (Learning Attributed GRAphlets). LAGRA learns importance weights for small attributed subgraphs, called attributed graphlets (AGs), while simultaneously optimizing their attribute vectors. This enables us to obtain a combination of subgraph structures and their attribute vectors that strongly contribute to discriminating different classes. A significant characteristics of LAGRA is that all the subgraph structures in the training dataset can be considered as a candidate structures of AGs. This approach can explore all the potentially important subgraphs exhaustively, but obviously, a na¨ıve implementation can require a large amount of computations. To mitigate this issue, we propose an efficient pruning strategy by combining the proximal gradient descent and a graph mining tree search. Our pruning strategy can ensure that the quality of the solution is maintained compared to the result without pruning. We empirically demonstrate that LAGRA has superior or comparable prediction performance to the standard existing algorithms including graph neural networks, while using only a small number of AGs in an interpretable manner.
arXiv:2402.06932v1
# 1 Introduction
Prediction problems with a graph input, such as graph classification problems, have been widely studied in the data science community. A graph representation is useful to capture structural data, and graphbased machine learning algorithms have been applied to variety of application problems such as chemical composition analysis [1, 2] and crystal structure analysis [3, 4]. In real-word datasets, graphs often have node attributes as a continuous value vector (note that we only focus on node attributes throughout the paper, but the discussion is same for edge attributes). For example, a graph created by a chemical composition can have a three dimensional position of each atom as an attribute vector in addition to a categorical label such as atomic species. In this paper, we consider building an interepretable prediction model for a graph classification problem in which an input graph has continuous attribute vectors. As we will see in Section 3, this setting has not been widely studied despite its practical importance. Our framework can identify important small subgraphs, called graphlets, in which each node has an attribute vector. Note that we use the term graphlet simply to denote a small connected subgraph [5], though in some papers, it only indicates induced subgraphs [6]. Figure 1 shows an illustration of our prediction model. In the figure, the output of the prediction model f(G) = β0 + βH1ψ(G; H1) + βH2ψ(G; H2)) + · · · for an
ψ ψ ψ ( ) ( ) ( ) = + + + ; ; ; AG AG AG H1 H2 H3 β0 +βH1 βH2 βH3 f (G) G G G
Figure 1: Illustration of our attributed graphlet (AG) based prediction model. The colors of each graph node represents a graph node label, and a bar plot associated with each graph node represents a trainable attribute vector.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a759/a759ad14-cc80-482c-a29c-912a0044c902.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: An example of important attributed graphlets H+ and H−identified by LAGRA in the AIDS dataset. H+ and H−positively and negatively contribute to the prediction, respectively. The right plot is a scatter in which x- and y- axes are our graphlet features representing the how precisely H+ and H−are included in the input graph Gi. Each point is from the test dataset.</div>
Figure 2: An example of important attributed graphlets H+ and H−identified by LAGRA in the AIDS dataset. H+ and H−positively and negatively contribute to the prediction, respectively. The right plot is a scatter in which x- and y- axes are our graphlet features representing the how precisely H+ and H−are included in the input graph Gi. Each point is from the test dataset. input attributed graph G is defined through a linear combination of attributed graphlets (AGs), represented as H1, H2, . . ., each one of which is weighted by parameters βH1, βH2, . . .. The function ψ(G; H) evaluates a matching score between G and an AG H in a sense that how precisely G contains the AG H. We apply a sparse regularization to the parameter βH by which a small number of important AGs for classification can be obtained, i.e., an AG with non-zero βH (in particular, if it has large |βH|) can be regarded as a discriminative AG. Important subgraphs and attribute vectors are usually unknown beforehand. The basic strategy of our proposed method, called LAGRA (Learning Attributed GRAphlets), is as follows: • To explore potentially important substructures of graphs, i.e., subgraphs, LAGRA uses graph mining by which all the subgraphs in the given dataset can be considered up to the given maximum graph size.
Figure 2 is an example of identified AGs by LAGRA, which shows only two AGs clearly separate two classes (See Section 4.2 for detail). Since the number of the possible subgraphs is quite large and an attribute vector exists for each node in each one of subgraphs, a na¨ıve implementation becomes computationally intractable. For the efficient optimization, we employ a block coordinate update [7] based approach in which β (a vector containing βH), the bias term β0, and attribute vectors are alternately updated. In the alternate update of β, we apply the proximal gradient descent [8, 9], which is known as an effective algorithm to optimize sparse parameters. For this step, we propose an efficient pruning strategy, enabling us to identify dimensions
that are not required to update at that iteration. This pruning strategy has the three advantages. First, by combining the sparsity during the proximal update and the graph mining tree search, we can eliminate unnecessary dimensions without enumerating all the possible subgraphs. Second, for removed variables βH at that iteration, attribute vectors in H are also not required to be updated, which also accelerates the optimization. Third, our pruning strategy is designed so that it can maintain the update result compared with when we do not perform the pruning (In other words, our pruning strategy does not deteriorate the resulting model accuracy). Our contributions are summarized as follows:
• We propose an interpretable graph classification model, in which the prediction is defined through a linear combination of graphlets that have trainable attribute vectors. By imposing a sparse penalty on the coefficient of each AG, a small number of important AGs can be identified.
• To avoid directly handling an intractably large size of optimization variables, we propose an efficient pruning strategy based on the proximal gradient descent, which can safely ignore AGs that do not contribute to the update.
• We verify effectiveness of LAGRA by empirical evaluations. Although our prediction model is simple and interpretable, we show that prediction performance of LAGRA was superior to or comparable with well-known standard graph classification methods, and in those results, LAGRA actually only used a small number of AGs. Further, we also show examples of selected AGs to demonstrate the high interpretability.
# 2 Proposed Method: LAGRA
In this section, we describe our proposed method, called Learning Attributed GRAphlets (LAGRA). First in Section 2.1, we show the formulation of our model and the definition of the optimization problem. Second in Section 2.2, we show an efficient optimization algorithm for LAGRA.
# 2.1 Formulation
# 2.1.1 Problem Setting
We consider a classification problem in which a graph G is an input. A set of nodes and edges of G are written as VG and EG, respectively. Each one of nodes v ∈VG has a categorical label Lv and a continuous attribute vector zG v ∈Rd, where d is an attribute dimension. In this paper, an attribute indicates a continuous attribute vector. We assume that a label and an attribute vector are for a node, but the discussion in this paper is completely same as for an edge label and attribute. A training dataset is {(Gi, yi)}i∈[n], in which yi ∈{−1, +1} is a binary label and n is the dataset size, where [n] = {1, . . . , n}. Although we only focus on the classification problem, our framework is also applicable to the regression problem just by replacing the loss function.
# 2.1.2 Attributed Graphlet Inclusion Score
We consider extracting important small attributed graphs, which we call attributed graphlets (AGs), that contributes to the classification boundary. Note that throughout the paper, we only consider a connected graph as an AG for a better interpretability (do not consider an AG by a disconnected graph). Let ψ(Gi; H) ∈ [0, 1] be a feature representing a degree that an input graph includes an AG H. We refer to ψ(Gi; H) as the
AG inclusion score (AGIS). Our proposed LAGRA identifies important AGs by applying a feature selection to a model with this AGIS feature. Suppose that L(G) is a labeled graph having a categorical label Lv for each node, and in L(G), an attribute zG v for each node is excluded from G. We define AGIS so that it has a non-zero value only when L(H) is included in L(Gi):
 where L(H) ⊑L(Gi) means that L(H) is a subgraph of L(Gi), and ϕH(Gi) ∈(0, 1] is a function that provides a continuous inclusion score of H in Gi. The condition L(H) ⊑L(Gi) makes AGIS highly interpretable. For example, in the case of chemical composition data, if L(H) represents O-C (oxygen and carbon are connected) and ψ(Gi; H) > 0, then we can guarantee that Gi must contain O-C. Figure 3(a) shows examples of L(Gi) and L(H). The function ϕH(Gi) needs to be defined so that it can represent how strongly the attribute vectors in H can be matched to those of Gi. When L(H) ⊑L(Gi), there exists at least one injection m : VH →VGi in which m(v) for v ∈VH preserves node labels and edges among v ∈VH. Figure 3(b) shows an example of when there exist two injections. Let M be a set of possible injections m. We define a similarity between H and a subgraph of Gi matched by m ∈M as follows
�� �� where ρ > 0 is a fixed parameter that adjusts the length scale. In exp, the sum of squared distanc of attribute vectors between matched nodes are taken. To use this similarity in AGIS (1), we take th maximum among all the matchings M:
ϕH(Gi) = MaxPooling({Sim(H, Gi; m) : m ∈M})
An intuition behind (2) is that it evaluates inclusion of H in Gi based on the best macthing in a sense of Sim(H, Gi; m) for m ∈M. If L(H) ⊑L(Gi) and there exists m such that zH v = zGi m(v) for ∀v ∈VH, then, ϕH(Gi) takes the maximum value (i.e., 1).
# 2.1.3 Model definition
# Our prediction model linearly combines the feature ψ(Gi; H) as follows:
f(Gi) = � H∈H ψ(Gi; H)βH + β0 = ψ⊤ i β + β0,
� where βH and β0 are parameters, H is a set of candidate AGs, and β and ψi are vectors containing βH and ψ(Gi; H) for H ∈H, respectively. Let L = {L | L ⊆L(Gi), i ∈[n], |L| ≤maxpat} be a set of all the labeled subgraphs contained in the training input graphs {Gi}i∈[n], where |L| is the number of nodes in the labeled graph L and maxpat is the user-specified maximum size of AGs. The number of the candidate AGs |H| is set as the same size as |L|. We set H as a set of attributed graphs created by giving trainable attribute vectors zH v (v ∈VH) to each one of elements in L. Figure 4 shows a toy example. Our optimization problem for βH, β0 and zH v is defined as the following regularized loss minimization in which the sparse L1 penalty is imposed on βH:
(1)
(2)
(3)
(4)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/33cb/33cb1ef6-b150-439d-88e7-a163d7fbef34.png" style="width: 50%;"></div>
Figure 3: Examples of matchings between a graph and AGs (colors of graph nodes are node labels). (a) For two AGs H and H′, L(Gi) only contains L(H), and L(H′) is not contained. Then, ψ(Gi; H) > 0 and ψ(Gi; H′) = 0. (b) An example of the set of injections M = {m, m′}, where m(1) = 2, m(2) = 3, m′(1) = 1, and m′(2) = 4. The figure shows that m and m′ are label and edge preserving.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f015/f015635a-e991-4d1d-a3b0-03d58419d21d.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: An example of training data, L and H. Since L only includes subgraphs in the training data “ ” is not included in L. H is created from L by adding trainable attribute vectors zHi v (v ∈VHi).</div>
ℓ(yi, f(Gi)) = max(1 −yif(Gi), 0)2.
Since the objective function (4) induces a sparse solution for βH, we can identify a small number of important AGs as H having the non-zero βH. However, this optimization problem has an intractably large number of optimization variables (H contains all the possible subgraphs in the training dataset and each one of H ∈H has the attribute vector zH v ∈Rd for each one of nodes). We propose an efficient optimization algorithm that mitigates this problem.
# 2.2 Optimization
Our optimization algorithm is based on the block coordinate update [7] algorithm, in which the (proximal) gradient descent alternately updates a block of variables. We update one of β, β0 and ZH alternately, while the other two parameters are fixed. First, the proximal gradient update is applied to β because it has the L1 penalty. Second, for β0, we calculate the optimal solution under fixing the other variable because it is
<div style="text-align: center;">(b)</div>
easy to obtain. Third, for ZH, we apply the usual gradient descent update because it does not have sparse penalty. The difficulty of the optimization problem (4) originates from the size of H. We select a small size of a subset W ⊆H, and only βW ∈R|W|, defined by βH for H ∈W, and corresponding attribute vectors ZW = {zH v | v ∈VH, H ∈W} ⊆ZH are updated. We propose an efficient pruning strategy by combining the proximal gradient with the graph mining, which enables us to select W without enumerating all the possible subgraphs. A notable characteristics of this approach is that it can obtain the completely same result compared with when we do not restrict the size of variables.
# 2.2.1 Update β, β0 and ZH
 ZH Before introducing the subset W, we first describe update rules of each variable. First, we apply the proxi gradient update to β. Let
Before introducing the subset W, we first describe update rules of each variable. First, we apply the proximal gradient update to β. Let
be the derivative of the loss term in (4) with respect to βH. Then, the update of β is defined by β(new) H ←prox (βH −η gH(β)) ,
where η > 0 is a step length, and
   is a proximal operator (Note that the proximal gradient for the L1 penalty is often called ISTA [9], for which an accelerated variant called FISTA is also known. We here employ ISTA for simplicity). We select the step length η by the standard backtrack search. The bias term β0 is update by
which is the optimal solution of the original problem (4) for given other variables β and ZH. Since the objective function of β0 is a differential convex function, the update rule of β0 can be derived from the first order condition as
where I(new) = {i | 1 −yi(ψ⊤ i β + β(new) 0 ) > 0}. This update rule contains β(new) 0 in I(new). However, it is easy to calculate the update (6) without knowing β(new) 0 beforehand. Here, we omit detail because it is a simple one dimensional problem (see supplementary appendix A). For zH v ∈ZH, we employ the standard gradient descent:
 α > 0 is a step length to which we apply the standard backtrack s
(5)
(6)
(7)
In every update of β, we incrementally add required H into W ⊆H. For the complement set W = H \ W, which contains AGs that have never been updated, we initialize βH = 0 for H ∈W. For the initialization of a node attribute vector zH v ∈ZH, we set the same initial vector if the node (categorical) labels are same, i.e., zH v = zH′ v′ if Lv = Lv′ for ∀H, H′ ∈H (in practice, we use the average of the attribute vectors within each node label). This constraint is required for our pruning criterion, but it is only for initial values. After the update (7), all zH v can have different values. Since βH = 0 for H ∈W, it is easy to derive the following relation from the proximal update (5):
# |gH(β)| ≤λ and H ∈W ⇒0 = prox (βH −η gH(β)) .
This indicates that if the conditions in the left side hold, we do not need to upd 0. Therefore, we set
W ←W ∪ � H ��|gH(β)| > λ, ∀H ∈W � ,
� �� � and apply the update (5) only to H ∈W. However, evaluating |gH(β)| > λ for all H ∈W can be computationally intractable because it needs to enumerate all the possible subgraphs. The following theorem can be used to avoid this difficulty: Theorem 2.1 Let L(H′) ⊒L(H) and H, H′ ∈W. Then,
where
gH(β) = max � � i∈I∩{i|yi>0} yiψ(Gi; H)(1 −yi(ψ⊤ i β + β0)),
− � i∈I∩{i|yi<0} yiψ(Gi; H)(1 −yi(ψ⊤ i β + β0)) �
where I = {i | 1 −yi(ψ⊤ i β + β0) > 0}.
 I { | −i i 0} See supplementary appendix B for the proof. Note that here I is defined by the current β0 unlike (6). This theorem indicates that the gradient |gH′(β)| for any H′ whose L(H′) contains L(H) as a subgraph can be bounded by gH(β). It should be noted that gH(β) can be calculated without generating H′, and it mainly needs only the model prediction with the current parameter ψ⊤ i β + β0, which can be immediately obtained at each iteration, and AGIS ψ(Gi; H). The rule (8) reveals that, to identify β(new) H′ = 0, we only require to know whether |gH′(β)| ≤λ holds, and thus, an important consequence of theorem 2.1 is the following rule: gH(β) ≤λ and H ∈W ⇒|gH′(β)| ≤λ for ∀H′ ∈{H′ | L(H′) ⊒L(H), H′ ∈W}. (10) Therefore, if the conditions in the first line in (10) hold, any H′ whose L(H′) contains L(H) as a subgraph can be discarded during that iteration. Further, from (7), we can immediately see that attribute vectors zH v for ∀v ∈VH are also not necessary to be updated if βH = 0. This is an important fact because updates of a large number of variables can be omitted. Figure 5 shows an illustration of the forward and backward (gradient) computations of LAGRA. For the gradient pruning, an efficient algorithm can be constructed by combining the rule (10) and a graph
 I { | −  } See supplementary appendix B for the proof. Note that here I is defined by the current β0 unlike (6). This theorem indicates that the gradient |gH′(β)| for any H′ whose L(H′) contains L(H) as a subgraph can be bounded by gH(β). It should be noted that gH(β) can be calculated without generating H′, and it mainly needs only the model prediction with the current parameter ψ⊤ i β + β0, which can be immediately obtained at each iteration, and AGIS ψ(Gi; H). The rule (8) reveals that, to identify β(new) H′ = 0, we only require to know whether |gH′(β)| ≤λ holds, and thus, an important consequence of theorem 2.1 is the following rule:
 ≤ ∈W ⇒|gH′(β)| ≤λ for ∀H′ ∈{H′ | L(H′) ⊒L(H), H′ ∈W}.
Therefore, if the conditions in the first line in (10) hold, any H′ whose L(H′) contains L(H) as a subgraph can be discarded during that iteration. Further, from (7), we can immediately see that attribute vectors zH v for ∀v ∈VH are also not necessary to be updated if βH = 0. This is an important fact because updates of a large number of variables can be omitted. Figure 5 shows an illustration of the forward and backward (gradient) computations of LAGRA. For the gradient pruning, an efficient algorithm can be constructed by combining the rule (10) and a graph
(8)
(9)
(10)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/dc1e/dc1e13e5-4f02-4b3a-9dae-7fb6edd2cb0f.png" style="width: 50%;"></div>
<div style="text-align: center;">... (a) AGIS computation</div>
Figure 5: An illustration of LAGRA. a) In the forward pass, only passes with |βH| > 0 contribute to the output. AGIS is defined by the best matching between an input graph and an AG. b) For the backward pass, the gradient can be pruned when the rule (10) is satisfied. In this illustration, H′′ is pruned by which graphs expanded from H′′ are not required to compute the gradient. mining algorithm. A well-known efficient graph mining algorithm is gSpan [11], which creates the tree by recursively expanding each graph in the tree node as far as the expanded graph is included in a given set of graphs as a subgraph. An important characteristics of the mining tree is that all graphs must contain any graph of its ancestors as subgraphs. Therefore, during the tree traverse (depth-first search) by gSpan, we can prune the entire subtree (all descendant nodes) if gH(β) ≤λ holds for the AG H in a tree node (Figure 5(b)). This means that we can update W by (9) without exhaustively investigating all the elements in W. gSpan has another advantage for LAGRA. To calculate the feature ϕH(Gi), defined in (2), LAGRA requires a set of injections M (an example is shown in Figure 3(b)). gSpan keeps the information of M during the tree traverse because it is required to expand a subgraph in each Gi (see the authors implementation https://sites.cs.ucsb.edu/~xyan/software/gSpan.htm). Therefore, we can directly use M created by gSpan to calculate (2).
Figure 5: An illustration of LAGRA. a) In the forward pass, only passes with |βH| > 0 contribute to the output. AGIS is defined by the best matching between an input graph and an AG. b) For the backward pass, the gradient can be pruned when the rule (10) is satisfied. In this illustration, H′′ is pruned by which graphs expanded from H′′ are not required to compute the gradient.
<div style="text-align: center;">Figure 5: An illustration of LAGRA. a) In the forward pass, only passes with |βH| > 0 contribute to th output. AGIS is defined by the best matching between an input graph and an AG. b) For the backwar pass, the gradient can be pruned when the rule (10) is satisfied. In this illustration, H′′ is pruned by whic graphs expanded from H′′ are not required to compute the gradient.</div>
mining algorithm. A well-known efficient graph mining algorithm is gSpan [11], which creates the tree by recursively expanding each graph in the tree node as far as the expanded graph is included in a given set of graphs as a subgraph. An important characteristics of the mining tree is that all graphs must contain any graph of its ancestors as subgraphs. Therefore, during the tree traverse (depth-first search) by gSpan, we can prune the entire subtree (all descendant nodes) if gH(β) ≤λ holds for the AG H in a tree node (Figure 5(b)). This means that we can update W by (9) without exhaustively investigating all the elements in W. gSpan has another advantage for LAGRA. To calculate the feature ϕH(Gi), defined in (2), LAGRA requires a set of injections M (an example is shown in Figure 3(b)). gSpan keeps the information of M during the tree traverse because it is required to expand a subgraph in each Gi (see the authors implementation https://sites.cs.ucsb.edu/~xyan/software/gSpan.htm). Therefore, we can directly use M created by gSpan to calculate (2).
# 2.2.3 Algorithm
We here describe entire procedure of the optimization of LAGRA. We employ the so-called regularization path following algorithm (e.g., [12]), in which the algorithm starts from a large value of the regularization parameter λ and gradually decreases it while solving the problem for each λ. This strategy can start from highly sparse β, in which usually W also becomes small. Further, at each λ, the solution obtained in the previous λ, can be used as the initial value by which faster convergence can be expected (so-called warm start). Algorithm 1 shows the procedure of the regularization path following. We initialize β = 0, which is
obviously optimal when λ = ∞. In line 2 of Algorithm 1, we calculate λmax at which β starts having nonzero values: λmax = maxH∈H ����n i∈[n] yiψ(Gi; H)(1 −yiβ0) ���, where β0 = � i∈[n] yi/n. See supplementary appendix C for derivation. λmax can also be written as λmax = maxH∈H |gH(0)|. To find maxH∈H, we can use almost the same gSpan based pruning strategy by using an upper bound of gH(β) as shown in Section 2.2.2 (the only difference is to search the max value only, instead of searching all H satisfying |gH(β)| > λ), though in Algorithm 1, this process is omitted for brevity. After setting λ0 ←λmax, the regularization parameter λ is decreased by using a pre-defined decreasing factor R as shown in line 6 of Algorithm 1. For each λ1 > · · · > λK, the parameters β, β0 and ZH are alternately updated as described in Section 2.2.1 and 2.2.2. We stop the alternate update by monitoring performance on the validation dataset in line 14 (stop by thresholding the decrease of the objective function is also possible). The algorithm of the pruning strategy described in Section 2.2.2 is shown in Algorithm 2. This function recursively traverses the graph mining tree. At each tree node, first, gH(β) is evaluated to prune the subtree if possible. Then, if |gH(β)| > λk, H is included in W. The expansion from H (creating children of the graph tree) is performed by gSpan, by which only the subgraphs contained in the training set can be generated (see the original paper [11] for detail of gSpan). The initialization of the trainable attribute zH′ v is performed when H′ is expanded (line 15).
Algorithm 1: Optimization of LAGRA
1 function Reguralization-Path(K, R, MaxEpoch)
2
H0 ←a graph at the root node of the mining tree
3
W ←∅
4
β ←0, β0 = �
i∈[n] yi/n
5
λ0 ←λmax
Compute λmax
6
for k = 1, 2, . . . , K do
7
λk ←Rλk−1
8
for epoch = 1, 2, . . . , MaxEpoch do
9
W ←W ∪GradientPruning(H0, λk)
10
Update β by (5) for H ∈W
11
Update β0 by (6)
12
Update zH
v by (7) for H ∈W
13
val loss ←Compute validation loss
14
if
val loss has not been improved in the past q iterations then
15
break
Inner loop stopping condition
16
else
17
M(k) ←(W, β, β0)
18
return {M(k)}K
k=0
# 3 Related Work
For graph-based prediction problems, recently, graph neural networks (GNNs) [13] have attracted wide attention. However, interpreting GNNs is not easy in general. According to a recent review of explainable GNNs [14], almost all of explainability studies for GNNs are instance-level explanations, which provides
Algorithm 2: Gradient Pruning
1 function GradientPruning(H, λk)
2
W ←∅
3
if gH(β) ≤λk then
4
return ∅
Prune the subtree
5
if |gH(β)| > λk then
6
W ←W ∪{H}
7
C ←CreateChildren(H)
8
for H′ ∈C do
9
W ←W ∪GradientPruning(H′, λk)
10
return W
11 function CreateChildren(H)
12
if
children of H have never been created by gSpan then
13
C ←graphs expanded from H by gSpan
14
for H′ ∈C do
15
zH′
v
←mean{zGi
v′ | v′ ∈VGi, Lv = Lv′, i ∈[n]}
16
Compute {ψ(Gi; H′)}n
i=1 using M created by gSpan
Table 1: Classification accuracy and the number of selected non-zero βH by LAGRA (the bottom row). The average of five runs and its standard deviation are shown. The underlines indicate the best average accuracy for each dataset and the bold-face indicates that the result is comparable with the best method in a sense of one-sided t-test (significance level 5%). # best indicates frequency that the method is the best or comparable with the best method.
AIDS
BZR
COX2
DHFR
ENZYMES
PROTEINS
SYNTHETIC
# best
GH
0.9985 ± 0.0020
0.8458 ± 0.0327
0.7872 ± 0.0252
0.7250 ± 0.0113
0.6050 ± 0.0857
0.7277 ± 0.0332
0.6767 ± 0.0655
2
ML
0.9630 ± 0.0062
0.8289 ± 0.0141
0.7787 ± 0.0080
0.7105 ± 0.0300
0.6000 ± 0.0652
0.6205 ± 0.0335
0.4867 ± 0.0356
0
PA
0.9805 ± 0.0086
0.8313 ± 0.0076
0.7809 ± 0.0144
0.7316 ± 0.0435
0.7500 ± 0.0758
0.6884 ± 0.0077
0.5400 ± 0.0859
1
DGCNN
0.9830 ± 0.0046
0.8169 ± 0.0177
0.8021 ± 0.0401
0.7289 ± 0.0192
0.7289 ± 0.0192
0.7509 ± 0.0114
0.9867 ± 0.0125
3
GCN
0.9840 ± 0.0030
0.8290 ± 0.0460
0.8340 ± 0.0257
0.7490 ± 0.0312
0.7000 ± 0.0837
0.6880 ± 0.0202
0.9630 ± 0.0194
2
GAT
0.9880 ± 0.0041
0.8220 ± 0.0336
0.7830 ± 0.0274
0.7110 ± 0.0156
0.7100 ± 0.0768
0.7160 ± 0.0108
0.9800 ± 0.0267
2
LAGRA (Proposed)
0.9900 ± 0.0050
0.8892 ± 0.0207
0.8043 ± 0.0229
0.8171 ± 0.0113
0.6450 ± 0.0797
0.7491 ± 0.0142
1.0000 ± 0.0000
4
# non-zero βH
50.4 ± 17.1
52.4 ± 19.0
45.4 ± 14.9
40.0 ± 11.6
7.2 ± 8.4
25.8 ± 9.4
35.8 ± 35.0
-
input-dependent explanations (Here, we do not mention each one of input-dependent approaches because the purpose is clearly different from LAGRA). An exception is XGNN [15], in which important discriminative graphs are generated for a given already trained GNN by maximizing the GNN output for a target label. However, unlike our method, the prediction model itself remains black-box, and thus, it is difficult to know underlying dependency between the identified graphs and the prediction. A classical approach to graph-based prediction problems is the graph kernel [16]. Although graph kernel itself does not identify important substructures, recently, [17] has proposed an interpretable kernel-based GNN, called KerGNN. KerGNN uses a graph kernel function as a trainable filter, inspired by the well-known convolutional networks, and the filter updates the node attributes of the input graph so that it embeds similarity to learned important subgraphs. Then, [17] claims that resulting graph filter can be seen as a key structure. However, a learned subgraph in a graph kernel filter is difficult to interpret. The kernel-based matching does not guarantee the existence of a subgraph unlike our AGIS (1), and further, only 1-hop neighbors of each node in the input graph are matched to a graph filter. Another graph mining based approach is [18]. This approach also uses a pruning based acceleration for the optimization, but it is based on the optimality of the convex problem while our proximal gradient pruning is applicable to the non-convex problem of LAGRA. Further, more importantly, [18] cannot deal with continuous attributes. The prediction model of LAGRA is inspired by a method for learning time-series shaplets (LTS) [19]. LTS is also based on a linear combination of trainable shaplets, which is a short fragment of a time-series sequence. Unlike time-series data, possible substructures in graph data have a combinatorial nature because of which our problem setting has a computational difficulty that does not exist in the case of LTS, for which LAGRA provide a graph mining based efficient strategy.
# 4 Experiments
Here, we empirically verify effectiveness of LAGRA. We used standard graph benchmark datasets, called AIDS, BZR, COX2, DHFR, ENZYMES, PROTEINS and SYNTHETIC, retrieved from https://ls11-www. cs.tu-dortmund.de/staff/morris/graphkerneldatasets (for ENZYMES, we only used two classes among original six classes to make a binary problem). To simplify comparison, we only used node labels and attributes, and did not use edge labels and attributes. Statistics of datasets are summarized in supplementary appendix D. The datasets are randomly divided into train : validation : test = 0.6 : 0.2 : 0.2. For the regularization path algorithm (Algorithm 1), we created candidate values of λ by uniformly dividing [log(λmax), log(0.01λmax)] into 100 grid points. We selected λ, maxpat ∈{5, 10} and ρ ∈{1, 0.5, 0.1, 0.05, 0.01} based on the validation performance.
# 4.1 Prediction Performance
For the prediction accuracy comparison, we used graph kernels and graph neural networks (GNN). We used three well-known graph kernels that can handle continuous attributes, i.e., graph hopper kernel (GH) [20], multiscale Laplacian kernel (ML) [21] and propagation kernel (PA) [22], for all of which the library called GraKeL [23] was used. For the classifier, we employed the k-nearest neighbor (k-NN) classification for which each kernel function k(Gi, Gj) defines the distance function as ∥Gi−Gj∥= � k(Gi, Gi) −2K(Gi, Gj) + k(Gj, Gj) The number of neighbors k is optimized by the validation set. For GNN, we used deep graph convolutional neural network (DGCNN) [24], graph convolutional network (GCN) [25], and graph attention network (GAT) [26]. For DGCNN, the number of hidden units {64, 128, 256} and epochs are optimized by the validation set. The other settings were in the default settings of the authors implementation
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6016/6016b536-9739-46be-a4ca-c58ba949057a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Selected important AGs for DHFR dataset. (a) and (b): AGs for the two largest positive coefficients. (c) and (d): AGs for the two largest negative coefficients.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/af23/af2367b8-bcab-4b31-ab2c-409bb931b502.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Selected important AGs for BZR dataset. (a) and (b): AGs for the two largest positive coefficients (c) and (d): AGs for the two largest negative coefficients.</div>
https://github.com/muhanzhang/pytorch_DGCNN. For GCN and GAT, we also selected the number of hidden units and epochs as above. For other settings, we followed [27]. The results are shown in Table 1. LAGRA was the best or comparable with the best method (in a sense of one-sided t-test) for BZR, DHFR, PROTEINS and SYNTHETIC (4 out of 7 datasets). For AIDS and COX2, LAGRA has similar accuracy values to the best methods though they were not regarded as the best accuracy in t-test. The three GNNs also show stable performance overall. Although our main focus is to build an interepretable model, we see that LAGRA achieved comparable accuracy with the current standard methods. Further, LAGRA only used a small number of AGs shown in the bottom row of Table 1, which suggests high interpretability of the learned models.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3bbd/3bbddad0-76e9-485d-bba9-ab936ba30d0a.png" style="width: 50%;"></div>
Figure 8: Scatter plots defined by selected AGs with test dataset of DHFR. The horizontal and vertical axe are AGIS of Fig. 6(a) and Fig. 6(c), respectively.
<div style="text-align: center;">Table 2: The size of the candidate set H with maxpat 10. For the ENZYMES, PROTEINS, and SYNTHETIC datasets, only the lower bounds are shown because it took long time to count.</div>
Table 2: The size of the candidate set H with maxpat 10. For the ENZYMES, PROTEINS,  datasets, only the lower bounds are shown because it took long time to count.
AIDS
BZR
COX2
DHFR
ENZYMES
PROTEINS
SYNTHETIC
134281
148903
101185
137872
> 15464000
> 13987000
> 699000
# 4.2 Examples of Selected Attributed Graphlet
We here show examples of identified important AGs. Figure 6 and 7 show AGs having the two largest positive and negative βH for DHFR and BZR datasets, respectively. In each figure, a labeled graphlet L(H) is shown in the left side (the numbers inside the graph nodes are the graph node labels) and optimized attribute vectors for each one of nodes are shown as bar plots in the right side. We can clearly see important substractures not only by as structural information of a graph but also attribute values associated with each node. Surprisingly, in a few datasets, two classes can be separated even in two dimensional space of AGIS. Figure 2 and Figure 8 show scatter plots of the test dataset (not the training dataset) with the axes of identified features by the LAGRA training. Let H+ and H−be AGs having the largest positive and negative βH, respectively. The horizontal and vertical axes of plots are ψ(Gi, H+) and ψ(Gi, H−). In particular, in the AIDS dataset, for which classification accuracy was very high in Table 1, two classes are clearly separated. For DHFR, we can also see points in two classes tend to be located on the upper left side and the lower right side. The dashed lines are boundaries created by (class-balance weighted) logistic regression fitted to the test points in these two dimensional spaces. The estimated class conditional probability has AUC = 0.94 and 0.62 for AIDS and BZR, respectively, which indicate that differences of two classes are captured even only by two AGs in these datasets.
# 4.3 Discussion on Computational Time
Finally, we verify computational time of LAGRA. First, Table 2 shows the size of candidate AGs |H| in each dataset. As we describe in Section 2.1.3, this size is equal to |L|, i.e., the number of all the possible subgraphs in the training datasets. Therefore, it can be quite large as particularly shown in the ENZYMES, PROTEINS and SYNTHETIC datasets in Table 2. The optimization variables in the objective function (4) are β, β0 and ZH. The dimension of β is |H| and the node attribute vector zH v ∈Rd exists for each one of nodes in H ∈H. Thus, the number of optimization variables in (4) is 1 + |H| + � H∈H |H| × d, which can be prohibitively large. Figure 9 shows the computational time during the regularization path. The horizontal axis is k of λk in Algorithm 1. The datasets are AIDS and ENZYMES. In the regularization path algorithm, the number of non-zero βH typically increases during the process of decreasing λ, because the L1 penalty becomes weaker gradually. As a results, in both the plots, the total time increases with the λ index. Although LAGRA performs the traverse of the graph mining tree in every iteration of the gradient update (line 9 in Algorithm 1), Figure 9 shows that the traverse time was not necessarily dominant (Note that the vertical axis is in log scale). In particular, when only a small number of tree nodes are newly expanded at that λ, the calculation for the tree search becomes faster because AGIS ψ(Gi, H) is already computed at the most of tree nodes. The computational times were at most about 103 sec for these datasets. We do not claim that LGARA is computationally faster compared with other standard algorithms (such as graph kernels), but as the computational time of the optimization problem with 1 + |H| + � H∈H |H| × d variables, the results obviously indicate effectiveness of our pruning based optimization approach.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/46ca/46cace2d-84c9-4298-8c6f-dc10fa062b6a.png" style="width: 50%;"></div>
Figure 9: Transition of computational time (sec) on regularization path. For each λ, the total time and the time required to traverse the graph mining tree (in other words, the time required to identify W) is shown separately.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e31c/e31c64a0-919a-40d3-8eac-f23861ce3bc4.png" style="width: 50%;"></div>
Figure 10 shows the average number of traversed graph mining tree nodes and the size of selected |W| for AIDS and ENZYMES. In this figure, both the values increased with the decrease of λ because the effect of the sparse penalty becomes weaker. As shown in Table 2, the total number of the tree nodes were 134281 and more than 15464000 for AIDS and ENZYMES, respectively. Figure 10 shows the number of traversed nodes were at most about 6 × 103/134281(≈0.05) and 9 × 102/15464000(≈6 × 10−5), respectively. This clearly indicates that our pruning strategy can drastically reduce the tree nodes, at which the evaluation of gH(β) is required as shown in Algorithm 2. In the figure, we can see that |W| was further small. This indicates the updated β was highly sparse, by which the update of zH v becomes easier because it requires only for non-zero βH.
# 5 Conclusion
This paper proposed LAGRA (Learning Attributed GRAphlets), which learns a prediction model that linearly combines attributed graphlets (AGs). In LAGRA, graph structures of AGs are generated through a graph mining algorithm, and attribute vectors are optimized as a continuous trainable parameters. To identify a small number of AGs, the L1 sparse penalty is imposed on coefficients of AGs. We employed a block coordinate update based optimization algorithm, in which an efficient pruning strategy was proposed by combining the proximal gradient update and the graph mining tree search. Our empirical evaluation
<div style="text-align: center;">(b) ENZYMES</div>
showed that LAGRA has superior or comparable performance with standard graph classification algorithms. We further demonstrated that LAGRA actually can identify a small number of discriminative AGs that have high interpretability.
# Acknowledgments
This work was partially supported by MEXT KAKENHI (21H03498, 22H00300, 23K17817), and Interna tional Joint Usage/Research Project with ICR, Kyoto University (2023-34).
[1] L. Ralaivola, S. J. Swamidass, H. Saigo, and P. Baldi, “Graph kernels for chemical informatics,” Neural Networks, vol. 18, no. 8, pp. 1093–1110, 2005. [2] F. A. Faber, L. Hutchison, B. Huang, J. Gilmer, S. S. Schoenholz, G. E. Dahl, O. Vinyals, S. Kearnes, P. F. Riley, and O. A. von Lilienfeld, “Prediction errors of molecular machine learning models lower than hybrid dft error,” Journal of Chemical Theory and Computation, vol. 13, no. 11, pp. 5255–5264, 2017. [3] T. Xie and J. C. Grossman, “Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties,” Phys. Rev. Lett., vol. 120, p. 145301, 2018. [4] S.-Y. Louis, Y. Zhao, A. Nasiri, X. Wang, Y. Song, F. Liu, and J. Hu, “Graph convolutional neural networks with global attention for improved materials property prediction,” Phys. Chem. Chem. Phys., vol. 22, pp. 18 141–18 148, 2020. [5] N. Shervashidze, S. Vishwanathan, T. Petri, K. Mehlhorn, and K. Borgwardt, “Efficient graphlet kernels for large graph comparison,” in Artificial Intelligence and Statistics, 2009, pp. 488–495. [6] N. Prˇzulj, “Biological network comparison using graphlet degree distribution,” Bioinformatics, vol. 23, no. 2, pp. e177–e183, 2007. [7] Y. Xu and W. Yin, “A globally convergent algorithm for nonconvex optimization based on block coordinate update,” Journal of Scientific Computing, vol. 72, pp. 700–734, 2017. [8] M. Teboulle, “A simplified view of first order methods for optimization,” Mathematical Programming, vol. 170, pp. 67–96, 2017. [9] A. Beck and M. Teboulle, “A fast iterative shrinkage-thresholding algorithm for linear inverse problems,” SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183–202, 2009. 10] K. Janocha and W. M. Czarnecki, “On loss functions for deep neural networks in classification,” Schedae Informaticae, vol. 25, p. 49, 2016. 11] X. Yan and J. Han, “gSpan: Graph-based substructure pattern mining,” in Proceedings. 2002 IEEE International Conference on Data Mining. IEEE, 2002, pp. 721–724. 12] J. Friedman, T. Hastie, H. H¨ofling, and R. Tibshirani, “Pathwise coordinate optimization,” The Annals of Applied Statistics, vol. 1, no. 2, pp. 302–332, 12 2007. 13] J. Zhou, G. Cui, S. Hu, Z. Zhang, C. Yang, Z. Liu, L. Wang, C. Li, and M. Sun, “Graph neural networks: A review of methods and applications,” AI Open, vol. 1, pp. 57–81, 2020. 14] H. Yuan, H. Yu, S. Gui, and S. Ji, “Explainability in graph neural networks: A taxonomic survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. 15] H. Yuan, J. Tang, X. Hu, and S. Ji, “XGNN: Towards model-level explanations of graph neural networks,” in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York, NY, USA: Association for Computing Machinery, 2020, pp. 430–438. 16] N. M. Kriege, F. D. Johansson, and C. Morris, “A survey on graph kernels,” Applied Network Science, vol. 5, p. 6, 2020.
[17] A. Feng, C. You, S. Wang, and L. Tassiulas, “KerGNNs: Interpretable graph neural networks with graph kernels,” in Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI. AAAI Press, 2022, pp. 6614–6622. [18] K. Nakagawa, S. Suzumura, M. Karasuyama, K. Tsuda, and I. Takeuchi, “Safe pattern pruning: An efficient approach for predictive pattern mining,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016, pp. 1785–1794. [19] J. Grabocka, N. Schilling, M. Wistuba, and L. Schmidt-Thieme, “Learning time-series shapelets,” in Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York, NY, USA: Association for Computing Machinery, 2014, p. 392–401. [20] A. Feragen, N. Kasenburg, J. Petersen, M. de Bruijne, and K. Borgwardt, “Scalable kernels for graphs with continuous attributes,” in Advances in Neural Information Processing Systems, 2013, pp. 216–224. [21] R. Kondor and H. Pan, “The multiscale Laplacian graph kernel,” in Advances in Neural Information Processing Systems, 2016, pp. 2990–2998. [22] M. Neumann, R. Garnett, C. Bauckhage, and K. Kersting, “Propagation kernels: efficient graph kernels from propagated information,” Machine Learning, vol. 102, no. 2, pp. 209–245, 2016. [23] G. Siglidis, G. Nikolentzos, S. Limnios, C. Giatsidis, K. Skianis, and M. Vazirgiannis, “GraKeL: A graph kernel library in python,” Journal of Machine Learning Research, vol. 21, no. 54, pp. 1–5, 2020. [24] M. Zhang, Z. Cui, M. Neumann, and Y. Chen, “An end-to-end deep learning architecture for graph classification,” in Proceedings of AAAI Conference on Artificial Inteligence, 2018. [25] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. [26] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Li`o, and Y. Bengio, “Graph attention networks,” in 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. [27] J. You, J. M. Gomes-Selman, R. Ying, and J. Leskovec, “Identity-aware graph neural networks,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 12, 2021, pp. 10 737–10 745.
# Supplementary Appendix for “Learning Attributed Graphlets: Predictive Graph Mining by Graphlets with Trainable Attribute”
A Update β0
The objective of β0 can be re-written as
min β0 � i=0 max(1 −yi(ψ⊤ i β + β0), 0)2 = min β0 � i∈I (1 −yi(ψ⊤ i β + β0))2.
For simplicity, we assume that all ψ⊤ i β for i ∈[n] have different values (even when this does not hold, the optimal solution can be obtained by the same approach). Depending on β(new) 0 , elements in I(new) = {i | 1 −yi(ψ⊤ i β + β(new) 0 ) > 0} changes in a piecewise constant manner. The point that I(new) changes are characterized by the solution of the equation yi(ψ⊤ i β + β0) = 1(i ∈[n]) with respect to β0, i.e., there exist n + 1 segments on the space of β0 ∈R. Let B(k) = [βsk 0 , βek 0 ] be the k-th segment (k ∈{0, . . . , n}) and I(k) is I(new) when β(new) 0 ∈B(k). Note that βek 0 = βsk+1 0 (k ∈[n]), which is the solution of yk(ψ⊤ k β + β0) = 1, and βs0 0 = −∞and βen 0 = ∞. Under the assumption of β0 ∈B(k), the optimal β0 is
Since (11) is convex with respect to β0, the obtained ˆβ(k) 0 must be the optimal solution if it satisfies ˆβ(k) 0 ∈ B(k). Thus, the optimal β0 can be found by calculating ˆβ(k) 0 for all k ∈{0, . . . , n}.
# B Proof of Theorem 2.1
From the definition of AGIS, the following monotonicity property is guaranteed:
L(H′) ⊒L(H) and H, H′ ∈W ⇒ψ(Gi; H) ≥ψ(Gi; H′
L(H′) ⊒L(H) and H, H′ ∈W ⇒ψ(Gi; H) ≥ψ(Gi; H′)
Let M(Gi; H) be the set of injections M between Gi and H. The above monotonicity property can be eas verified from the fact
min m∈M(Gi;H) D(m) H,Gi ≤ min m∈M(Gi;H′) D(m) H′,Gi.
Define
   ai = 0 otherwise Note that the sign of ai is same as yi. Using ai, we re-write gH′(β) as
 Note that the sign of ai is same as yi. Using ai, we re-write gH′(β) as
18
(11)
AIDS
BZR
COX2
DHFR
ENZYMES
PROTEINS
SYNTHETIC
# instances
2000
405
467
756
200
1113
300
Dim. of attribute vector d
4
3
3
3
18
1
1
Avg. # nodes
15.69
35.75
41.22
42.43
32.58
39.06
100.00
Avg. # edges
16.20
38.36
43.45
44.54
60.78
72.82
196.00
From the monotonicity inequality ψ(Gi; H) ≥ψ(Gi; H′), we see
From these inequalities, we obtain
#  C Derivation of λmax
When β = 0, the objective function of β0 is written as the following piecewise quadratic function: 
# When β = 0, the objective function of β0 is written as the following piecewise quadratic function: 
# When β = 0, the objective function of β0 is written as the following piecewise quadratic function:
min β0 1 2 � i∈I(β0) (1 −yiβ0)2 s.t. I(β0) =        {i | yi > 0} β0 ≤−1, [n] β0 ∈[−1, 1], {i | yi < 0} β0 ≥1.
min β0 1 2 � i∈I(β0) (1 −yiβ0)2 s.t. I(β0) =        {i | yi > 0} β0 ≤−1, [n] β0 ∈[−1, 1], {i | yi < 0} β0 ≥1.
   In the region β0 ≤−1, the minimum value is achieved by β0 = −1, and for β0 ≥1, the minimum value is achieved by β0 = 1. This indicates that the optimal solution should exist in β0 ∈[−1, 1] because the objective function is a smooth convex function. Therefore, the minimum value in the region β0 ∈[−1, 1] achieved by β0 = � i∈[n] yi/n, defined as ¯y, becomes the optimal solution. When β = 0 and β0 = ¯y, we obtain
gH(0) = � i∈[n] yiψ(Gi; H)(1 −yi¯y).
gH(0) = � yiψ(Gi; H)(1 −yi¯y).
From (8), we see that |gH(0)| = λ is the threshold that βH have a non-zero value. This means that H having the maximum |gH(0)| is the first H that start having a non-zero value by decreasing λ from ∞. Therefore, we obtain λmax = max ����� � yiψ(Gi; H)(1 −yi¯y) ����� .
From (8), we see that |gH(0)| = λ is the threshold that βH have a non-zero value. This means that H having the maximum |gH(0)| is the first H that start having a non-zero value by decreasing λ from ∞. Therefore,
From (8), we see that |gH(0)| = λ is the threshold that βH have a non-zero value. This means that H having the maximum |gH(0)| is the first H that start having a non-zero value by decreasing λ from ∞. Therefore, we obtain �� ��
# D Statistics of datasets
Statistics of datasets is show in Table 3.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bd42/bd425a30-cbfc-4cdf-bcc7-3486c5860ff7.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 11: Selected important AGs for AIDS dataset. (a) and (b): AGs for the two largest positive coefficients. (c) and (d): AGs for the two largest negative coefficients.</div>
# E Additional Examples of Selected AGs
Figure 11 shows selected AGs for the AIDS dataset.
