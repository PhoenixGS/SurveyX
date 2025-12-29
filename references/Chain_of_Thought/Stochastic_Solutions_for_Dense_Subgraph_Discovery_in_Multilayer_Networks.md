# Stochastic Solutions for Dense Subgraph Discovery in Multilayer Networks
Atsushi Miyauchi The University of Tokyo Tokyo, Japan miyauchi@mist.i.u-tokyo.ac.jp Hanna Sumita Tokyo Institute of Technology Tokyo, Japan sumita@c.titech.ac.jp
Atsushi Miyauchi The University of Tokyo Tokyo, Japan miyauchi@mist.i.u-tokyo.ac.jp
Yasushi Kawase The University of Tokyo Tokyo, Japan kawase@mist.i.u-tokyo.ac.jp
# ABSTRACT
Network analysis has played a key role in knowledge discovery and data mining. In many real-world applications in recent years, we are interested in mining multilayer networks, where we have a number of edge sets called layers, which encode different types of connections and/or time-dependent connections over the same set of vertices. Among many network analysis techniques, dense subgraph discovery, aiming to find a dense component in a network, is an essential primitive with a variety of applications in diverse domains. In this paper, we introduce a novel optimization model for dense subgraph discovery in multilayer networks. Our model aims to find a stochastic solution, i.e., a probability distribution over the family of vertex subsets, rather than a single vertex subset, whereas it can also be used for obtaining a single vertex subset. For our model, we design an LP-based polynomial-time exact algorithm. Moreover, to handle large-scale networks, we also devise a simple, scalable preprocessing algorithm, which often reduces the size of the input networks significantly and results in a substantial speedup. Computational experiments demonstrate the validity of our model and the effectiveness of our algorithms.
CCS CONCEPTS • Theory of computation →Graph algorithms analysis; Mathematical optimization.
# • Theory of computation →Graph algorithms analysis; Mathematical optimization.
# KEYWORDS
network analysis, multilayer networks, dense subgraph discovery, stochastic solutions
# 1 INTRODUCTION
Network analysis has played a key role in knowledge discovery and data mining. In many real-world applications in recent years, we are interested in mining multilayer networks rather than ordinary (i.e., single-layer) networks, where we have a number of edge sets called layers, which encode different types of connections and/or time-dependent connections over the same set of vertices [7, 15, 30]. For example, in the Twitter network, there are various layers representing different types of connections, e.g., follower–followee relations, retweets, and mentions, among users. Moreover, each of those connections is time-dependent and therefore leads to multiple layers by itself. As another example, consider brain networks arising in neuroscience, where vertices correspond to small regions of a brain [19]. In this network, we can obtain at least two layers representing the structural connectivity and the functional connectivity (e.g., co-activation) among the small pieces of a brain.
Hanna Sumita Tokyo Institute of Technology Tokyo, Japan sumita@c.titech.ac.jp
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0520/0520fa1b-ae66-4ab0-902d-25d82370e680.png" style="width: 50%;"></div>
Figure 1: An optimal solution to our model with the regret metric for the real-world multilayer network called WILDBIRDS: selecting the black vertices with probability 0.29 and the union of the gray and black vertices with probability 0.71. There are six layers, and the color of each edge represents the layer containing the edge.
Among many network analysis techniques, dense subgraph discovery, aiming to find a dense component in a network, is an essential primitive with a variety of applications in diverse domains [22, 32]. Examples include detecting communities and spam link farms in Web graphs [17, 21], experts extraction in crowdsourcing systems [28], real-time story identification in microblogging streams [1], and extracting molecular complexes in protein–protein interaction networks [3]. Recently, dense subgraph discovery has been extended from single-layer networks to multilayer networks. Jethava and Beerenwinkel [27] introduced the densest common subgraph problem, where given a multilayer network, we are asked to find a vertex subset that maximizes the minimum degree density (defined later) over the layers. They mentioned some concrete applications of this problem in biological networks. Moreover, we can see that the densest common subgraph problem may appear in the context of robust optimization. Indeed, multilayer networks can be seen as a model of single-layer networks with uncertain edges with a number of scenarios, where each layer corresponds to one scenario. If we can find a subgraph that is reasonably dense for all layers, the solution is more robust than that obtained by single-layer network analysis, which may be arbitrarily bad for some layers.
# 1.1 Our contribution
In this paper, we introduce a novel optimization model for dense subgraph discovery in multilayer networks. Given an edge-weighted multilayer network, our model aims to find a vertex subset that is dense for the layer selected by an adversary. Specifically, we employ a stochastic solution, i.e., a probability distribution over the family of vertex subsets, while existing work focuses only on a
deterministic solution, i.e., a single vertex subset. More precisely, we stochastically choose a vertex subset according to a stochastic solution and then, knowing only the stochastic solution but not a realization, the adversary selects the worst layer for us. Note that the worst layer with respect to each realization of the stochastic solution may be different from that with respect to the stochastic solution. It is worth mentioning that our model can be seen as a zero-sum two-player Stackelberg game [2], where the leader is our algorithm and the follower is the adversary. We measure the density of a vertex subset using the quality function called the degree density (or simply density) similarly to Jethava and Beerenwinkel [27]. The degree density of a vertex subset (in a single-layer network) is defined as half the average degree of the subgraph induced by the subset. This quality function is often used in the literature of dense subgraph discovery; in fact, this is the objective function of the well-known densest subgraph problem (see Section 2). We evaluate the performance of a stochastic solution (or an algorithm) using the following three metrics for the layer selected by the adversary: (i) the density, i.e., the expected degree density, (ii) the robust ratio, i.e., the ratio of the expected density to the optimal density, and (iii) the regret, i.e., the difference between the optimal density and the expected density. Therefore, we address three optimization problems according to the metrics. Our main algorithmic contribution is to design a polynomialtime exact algorithm for our optimization model, which is applicable to all of the above three metrics. Specifically, the algorithm first solves an LP, which is a generalization of the LP used for the densest common subgraph problem [27] and then computes an optimal probability distribution based on the LP’s optimal solution. We observe that the output of our algorithm has a useful structure; the family of the vertex subsets with positive probabilities has a hierarchical structure. This leads to several practical benefits, e.g., the largest size subset contains all the other subsets and the optimal solution obtained by our algorithm has support size at most the number of vertices (although a solution of our optimization model may have support size exponential in the number of vertices). Moreover, we can demonstrate that the support size of the solution obtained is upper bounded by the number of layers. It is desirable to have a small support size for the understandability of stochastic solutions and for purposes of simple verification and validation. For practical use of our proposed algorithm, we wish to speed up our algorithm so that it is applicable to larger-scale multilayer networks. The bottleneck is the computation cost of the LP, which has a lot of variables and constraints. To this end, we devise a simple, scalable preprocessing algorithm, which often reduces the size of the input networks significantly and results in a substantial speed-up. Specifically, the algorithm first computes an approximate solution by solving a much smaller LP than the aforementioned LP and then removes vertices from the original network using the information of the approximate solution obtained. Note that our preprocessing algorithm is a generalization of that proposed by Balalau et al. [4] for the densest subgraph problem. However, to verify that the preprocessing algorithm does not harm any optimal solution, we need a more sophisticated analysis, which is totally different from theirs.
Finally, we conduct thorough computational experiments using synthetic graphs and real-world networks to verify the validity of our model and to evaluate the performance of our algorithms. Figure 1 illustrates an optimal solution to our model with the regret metric for the real-world multilayer network called WILDBIRDS (see Section 7 for its detailed characteristics). The optimal solution obtained is reasonably dense for all layers, as desired. It should be remarked that stochastic solutions can also be used for obtaining a single vertex subset. For example, the following three rules are reasonable to select a vertex subset from a stochastic solution: (i) stochastically select a subset following the probability distribution, (ii) select a subset with the highest probability, and (iii) select a subset with non-zero probability that optimizes some metric at hand (e.g., the minimum density, minimum robust ratio, and maximum regret, over layers). In our experiments, we compare the vertex subsets detected using the above rules with those obtained by existing algorithms for the densest common subgraph problem, in terms of various evaluation metrics.
# 2 RELATED WORK
The densest subgraph problem is one of the most popular optimization models for dense subgraph discovery. Let 𝐺= (𝑉, 𝐸) be an undirected graph and 𝑤: 𝐸→R++ a positive edge weight. For a vertex subset 𝑆⊆𝑉, the subgraph induced by 𝑆is denoted by 𝐺[𝑆] �(𝑆, 𝐸[𝑆]), where 𝐸[𝑆] � � {𝑢, 𝑣} ∈𝐸| 𝑢, 𝑣∈𝑆 �. In addition, we denote by 𝑤(𝑆) the total weight of the edges in 𝑆, i.e., 𝑤(𝑆) = � 𝑒∈𝐸[𝑆] 𝑤(𝑒). For a nonempty 𝑆⊆𝑉, the degree density (or simply called density) of 𝑆is defined as 𝑤(𝑆)/|𝑆| (where we define the density of the empty set (i.e., 0/0) to be 0). In the densest subgraph problem, given a graph 𝐺= (𝑉, 𝐸) with an edge weight 𝑤, we are asked to find 𝑆⊆𝑉that maximizes the density 𝑤(𝑆)/|𝑆|. It is well known that the densest subgraph problem can be solved exactly in polynomial time using a maximum-flow-based algorithm [23] or an LP-based algorithm [9]. Moreover, it was shown that a simple greedy algorithm called the greedy peeling admits 2-approximation in 𝑂(𝑚+ 𝑛log𝑛) time [9, 31]. Recently, Boob et al. [8] designed an iterative greedy peeling algorithm, and demonstrated empirically that the output tends to be nearly optimal. Later, Chekuri, Quanrud, and Torres [11] proved the convergence to optimality of this algorithm (in a more general context). Balalau et al. [4] introduced a simple preprocessing algorithm to improve the scalability of (exact) algorithms for the densest subgraph problem. Their preprocessing algorithm first computes an approximate solution 𝑆⊆𝑉(using the greedy peeling), and then iteratively removes a vertex with the (weighted) degree less than the objective value of the approximate solution obtained. The validity of this preprocessing algorithm is guaranteed by the fact that any vertex with the (weighted) degree less than the optimal value is not contained in any optimal solution [4]. For the densest common subgraph problem, Jethava and Beerenwinkel [27] devised an LP-based polynomial-time heuristic and a 2𝑘-approximation algorithm based on the greedy peeling, where 𝑘is the number of layers. Later, Charikar, Naamad, and Wu [10] designed two polynomial-time algorithms with approximation ratios 𝑂( √︁ |𝑉| log𝑘) and 𝑂(|𝑉|2/3) (irrespective of 𝑘), respectively. Moreover, they showed some strong inapproximability results for
the problem, based on some reasonable computational complexity assumptions. Thus, it is very unlikely that a well-approximate solution can be found in polynomial time. In contrast to this, as mentioned above, we can compute an optimal stochastic solution in terms of the aforementioned three metrics in polynomial time. Recently, Galimberti, Bonchi, and Gullo [20] introduced a generalization of the densest common subgraph problem, which they refer to as the multilayer densest subgraph problem. This problem exploits a trade-off between the minimum density value over layers and the number of layers considered. They proposed an approximation algorithm using a core decomposition technique for multilayer networks. Very recently, Hashemi, Behrouz, and Lakshmanan [24] designed a sophisticated core decomposition algorithm, which they call the FirmCore decomposition algorithm. Their algorithm finds the set of (𝑘, 𝜆)-FirmCores for all possible 𝑘and 𝜆in polynomial time, where (𝑘, 𝜆)-FirmCore is a maximal subgraph in which every vertex has degree no less than 𝑘in the subgraph for at least 𝜆layers. They demonstrated that the decomposition unfolds a better solution to the multilayer densest subgraph problem than the algorithm by Galimberti, Bonchi, and Gullo [20] for many instances. Semertzidis et al. [38] introduced another generalization of the densest common subgraph problem, called the Best Friends Forever (BFF) problem, in the context of evolving graphs with a number of snapshots. The BFF problem is a series of optimization problems that maximize an aggregate density over snapshots, where the aggregate density is set to be the average/minimum value of the average/minimum degree of vertices over layers. They investigated the computational complexity of the problems and designed some approximation or heuristic algorithms. Recalling that multilayer networks can also be seen as a model of networks with uncertainty, we can find some other optimization models related. Zou [44] studied the densest subgraph problem in uncertain graphs. An uncertain graph is a pair of 𝐺= (𝑉, 𝐸) and 𝑝: 𝐸→[0, 1], where 𝑒∈𝐸is present with probability 𝑝(𝑒) whereas 𝑒∈𝐸is absent with probability 1 −𝑝(𝑒). In the problem, given an uncertain graph 𝐺= (𝑉, 𝐸) with 𝑝, we seek 𝑆⊆𝑉that maximizes the expected density. Zou [44] showed that this problem can be reduced to the original densest subgraph problem, and designed a polynomial-time exact algorithm based on the reduction. Recently, Tsourakakis et al. [41] introduced a more general optimization problem called the risk-averse dense subgraph discovery. As another example, Miyauchi and Takeda [35] introduced an optimization problem called the robust densest subgraph problem. In this problem, given an undirected graph 𝐺= (𝑉, 𝐸) and an edge-weight space 𝐼= ×𝑒∈𝐸[𝑙𝑒,𝑟𝑒] ⊆×𝑒∈𝐸[0, ∞), we are asked to find 𝑆⊆𝑉that maximizes min𝑤∈𝐼 𝑤(𝑆)/|𝑆| 𝑤(𝑆∗𝑤)/|𝑆∗𝑤| , where 𝑆∗𝑤is an optimal solution to the densest subgraph problem for 𝐺with 𝑤. The intuition of this problem is the same as that of ours; this problem also seeks 𝑆⊆𝑉that is reasonably dense for any 𝑤∈𝐼. However, they considered only deterministic solutions and gave a strong hardness result. As well as dense subgraph discovery, many important primitives for single-layer network analysis have recently been extended to multilayer networks. Examples include community detection [6, 13, 25, 39], link prediction [13, 26], analyzing spreading processes [14, 37], and identifying central vertices [5, 16].
# 3 MODEL
In this section, we formally define our optimization model. Let 𝐺= (𝑉, (𝐸𝑖)𝑖∈[𝑘]) be a multilayer network consisting of 𝑘layers, and let 𝑤𝑖: 𝐸𝑖→R++ be a positive edge weight for layer 𝑖. We denote by 𝐸the union of all edge sets, i.e., 𝐸�� 𝑖∈[𝑘] 𝐸𝑖. Let 𝑆∗ 𝑖be a densest subgraph for layer 𝑖, i.e., 𝑆∗ 𝑖∈arg max𝑆⊆𝑉𝑤𝑖(𝑆)/|𝑆|. Our task is to find a vertex subset that is dense for the layer selected adversarially. As mentioned in the introduction, we consider a stochastic solution to compete with the adversary. Let Δ(2𝑉) be the set of probability distributions over 2𝑉. For each 𝑝∈Δ(2𝑉), we denote by 𝑝𝑆the probability of choosing 𝑆⊆𝑉. We aim to compute 𝑝∈Δ(2𝑉) that maximizes some metric (when the adversary selects the worst layer to 𝑝). We employ the following three metrics: Density. The first metric is the degree density itself. Specifically, when we select a vertex subset according to a probability distribution 𝑝∈Δ(2𝑉) and the adversary selects a layer 𝑖∈[𝑘], our metric is defined as follows: � � � �
As the adversary selects the worst layer, we aim to find 𝑝∈Δ(2𝑉) that maximizes the minimum of the density (1) among 𝑖∈[𝑘]. Our optimization model with this metric can be seen as a stochastic version of the densest common subgraph problem introduced by Jethava and Beerenwinkel [27]. Robust ratio. The robust ratio is a metric based on the ratio of the expected degree to the optimal degree. For a nonempty 𝑆⊆𝑉and 𝑖∈[𝑘], let us consider a normalized density defined as 𝑤𝑖(𝑆)/|𝑆| 𝑤𝑖(𝑆∗ 𝑖)/|𝑆∗ 𝑖| . When we select a vertex subset according to a probability distribution 𝑝∈Δ(2𝑉) and the adversary selects a layer 𝑖∈[𝑘], the metric is defined as follows: � � � �
(2)
In other words, the robust ratio is equivalent to the density for the multilayer network with weights 𝑤′ 1, . . . ,𝑤′ 𝑘given by 𝑤′ 𝑖(𝑒) � 𝑤𝑖(𝑒) 𝑤𝑖(𝑆∗ 𝑖)/|𝑆∗ 𝑖| for each 𝑖∈[𝑘] and 𝑒∈𝐸𝑖. As the adversary selects the worst layer, we aim to find 𝑝∈Δ(2𝑉) that maximizes the minimum of (2) among 𝑖∈[𝑘]. Note that the optimal robust ratio is contained in the interval [1/𝑘, 1] because 𝑝∈Δ(2𝑉) such that 𝑝𝑆∗ 𝑖= 1/𝑘for each 𝑖∈[𝑘] has the objective value of 1/𝑘. Regret. The regret is a metric based on the difference between the optimal density and the expected density. For 𝑆⊆𝑉and 𝑖∈[𝑘], the regret is defined as 𝑤𝑖(𝑆∗ 𝑖)/|𝑆∗ 𝑖| −𝑤𝑖(𝑆)/|𝑆|. When we select a vertex subset according to a probability distribution 𝑝∈Δ(2𝑉) and the adversary selects a layer 𝑖∈[𝑘], the metric is defined as follows: � � � �
(3)
As the adversary selects the worst layer, we aim to find 𝑝∈Δ(2𝑉) that minimizes the maximum of (3) among 𝑖∈[𝑘].
Here we explain how to select an appropriate metric. The density and regret metrics are useful when we are concerned with multilayer networks with homogeneous layers such as time-dependent follower-followee relations in the Twitter network. Although the density metric can be the first choice, the regret metric is more suitable for robust analysis. For example, consider the case where there are a number of layers consistent with each other together with some noisy (e.g., random) layers. The density metric would suffer from the effect of the noisy layers, but the regret metric would avoid it and find dense subgraphs in the other meaningful layers. On the other hand, the robust ratio metric is useful when we analyze multilayer networks with heterogeneous layers such as brain networks with structural and functional connectivity layers. From its definition, the robust ratio metric would find subgraphs that are reasonably dense for all layers. The density and regret metrics focus only on the layers with small optimal densities and the layers with large optimal densities, respectively. Unified concept: (𝜶, 𝜷)-density. Here we introduce a general metric, enabling us to deal with the above three metrics in a unified manner. An important fact is that the robust ratio and regret metrics can be obtained by affine transformations of the density. Specifically, when we select a subgraph according to a probability distribution 𝑝∈Δ(2𝑉) and the adversary selects a layer 𝑖∈[𝑘], we define the (𝜶, 𝜷)-density using two vectors 𝜶∈R𝑘+ and 𝜷∈R𝑘as follows: � � � �
Note that the above three metrics, the density, robust ratio, and regret, are equivalent to the (1, 0)-density, ((|𝑆∗ 𝑖|/𝑤𝑖(𝑆∗ 𝑖))𝑖∈[𝑘], 0)density, and (1, (−𝑤𝑖(𝑆∗ 𝑖)/|𝑆∗ 𝑖|)𝑖∈[𝑘])-density1, respectively. Note that𝑤𝑖(𝑆∗ 𝑖)/|𝑆∗ 𝑖| is polynomially computable for each 𝑖∈[𝑘] [9, 23]. We refer to (𝜶, 𝜷)-DENSITY as the problem of finding 𝑝∈Δ(2𝑉) that maximizes the minimum of the (𝜶, 𝜷)-density among 𝑖∈[𝑘]. Therefore, in the following, we aim to design an algorithm for (𝜶, 𝜷)-DENSITY.
# 4 ALGORITHM
In this section, we provide an LP-based polynomial-time exact algorithm for (𝜶, 𝜷)-DENSITY. Let ((𝑥𝑒)𝑒∈𝐸, (𝑦𝑣)𝑣∈𝑉,𝑡) be continuous variables. We consider the following LP:
(4)
(∀𝑒= {𝑢, 𝑣} ∈𝐸), (4)
(∀𝑒∈𝐸, ∀𝑣∈𝑉).
When (𝜶, 𝜷) = (1, 0), this formulation coincides with the LP introduced by Jethava and Beerenwinkel [27] for the densest common subgraph problem. However, they did not point out the connection between a solution of the LP and a distribution in Δ(2𝑉). Our algorithm first computes an optimal solution to LP (4), denoted by (( ˆ𝑥𝑒)𝑒∈𝐸, ( ˆ𝑦𝑣)𝑣∈𝑉, ˆ𝑡). Then we set 𝑟0,𝑟1, . . . ,𝑟ℓto be the
1The actual regret value is the negation of (1, (−𝑤𝑖(𝑆∗ 𝑖)/|𝑆∗ 𝑖|)𝑖∈[𝑘])-density.
<div style="text-align: center;">Algorithm 1: LP-based algorithm</div>
Algorithm 1: LP-based algorithm
1 Solve LP (4) to obtain an optimal solution
(( ˆ𝑥𝑒)𝑒∈𝐸, ( ˆ𝑦𝑣)𝑣∈𝑉, ˆ𝑡);
2 Let 𝑟0,𝑟1, . . . ,𝑟ℓbe reals such that
{𝑟0,𝑟1, . . . ,𝑟ℓ} = { ˆ𝑦𝑣| 𝑣∈𝑉} ∪{0} and
𝑟0 (= 0) < 𝑟1 < 𝑟2 < · · · < 𝑟ℓ;
3 Let 𝑆𝑗= {𝑣∈𝑉| ˆ𝑦𝑣≥𝑟𝑗} (𝑗= 1, . . . , ℓ);
4 return ˆ𝑝∈Δ(2𝑉) with ˆ𝑝𝑆𝑗= (𝑟𝑗−𝑟𝑗−1) · |𝑆𝑗| (𝑗= 1, . . . , ℓ)
(and ˆ𝑝𝑆= 0 for the other 𝑆’s).
reals such that {𝑟0,𝑟1, . . . ,𝑟ℓ} = { ˆ𝑦𝑣| 𝑣∈𝑉} ∪{0} and 𝑟0 (= 0) < 𝑟1 < 𝑟2 < · · · < 𝑟ℓ. In addition, let 𝑆𝑗= {𝑣∈𝑉| ˆ𝑦𝑣≥𝑟𝑗} (𝑗= 1, . . . , ℓ). The output of our algorithm is the probability distribution ˆ𝑝∈Δ(2𝑉) defined as
(5)
 � 1 = ∑︁ 𝑣∈𝑉 ˆ𝑦𝑣= ∑︁ 𝑗∈[ℓ] ∑︁ 𝑣: ˆ𝑦𝑣=𝑟𝑗 𝑟𝑗= ∑︁ 𝑗∈[ℓ] 𝑟𝑗(|𝑆𝑗| −|𝑆𝑗+1|) = ∑︁ 𝑗∈[ℓ] ˆ𝑝𝑆𝑗,
where 𝑆ℓ+1 = ∅for ease of notation. Our algorithm is formally described in Algorithm 1. Clearly, the algorithm runs in polynomial time.
# 5 ANALYSIS
In this section, we first demonstrate that the output ˆ𝑝of Algorithm 1 is optimal to (𝜶, 𝜷)-DENSITY. Then we analyze the structure of the output ˆ𝑝with a special attention to its support size.
# 5.1 Optimality of the output of Algorithm 1 We first prove that the (𝜶, 𝜷)-density of ˆ𝑝is equal to the optimal value of LP (4):
5.1 Optimality of the output of Algorithm 1 We first prove that the (𝜶, 𝜷)-density of ˆ𝑝is equal to the optimal value of LP (4):
Lemma 5.1. It holds that
Proof. To prove the lemma, we show that for any 𝑖∈[𝑘],
By the definition of 𝑆𝑗(𝑗= 1, . . . , ℓ), for each 𝑒= {𝑢, 𝑣} ∈𝐸𝑖, we observe that 𝑒∈𝐸𝑖[𝑆𝑗] ⇐⇒𝑢, 𝑣∈𝑆𝑗 ⇐⇒ˆ𝑦𝑢≥𝑟𝑗and ˆ𝑦𝑣≥ 𝑟𝑗 ⇐⇒ ˆ𝑥𝑒≥𝑟𝑗, where the last equivalence follows from ˆ𝑥𝑒= min{ ˆ𝑦𝑢, ˆ𝑦𝑣}.
Recall that ˆ𝑝is defined as (5). Then we have � � ∑︁
∈ where the second last equality follows from the above equivalence and the last equality follows from the fact that for each 𝑒∈𝐸,  �
 � Next, we prove that the optimal value of LP (4) gives an upper bound on the optimal value of (𝜶, 𝜷)-DENSITY:
Proof. Let us take an arbitrary ˜𝑝∈Δ(2𝑉). We consider a solution (( ˜𝑥𝑒)𝑒∈𝐸, ( ˜𝑦𝑣)𝑣∈𝑉, ˜𝑡) of LP (4) such that ∑︁ ∑︁ � ∑︁ �
 (6)
The solution (( ˜𝑥𝑒)𝑒∈𝐸, ( ˜𝑦𝑣)𝑣∈𝑉, ˜𝑡) is feasible for LP (4); in fact, the only concern is the third constraint but we see that ∑︁ ∑︁ ∑︁ ∑︁ � �
∈ Hence, the optimal value of LP (4) is at least ˜𝑡. Moreover, we have � ∑︁ �
Recalling that ˜𝑝is taken arbitrarily from Δ(2𝑉), we see that the optimal value of LP (4) is at least that of (𝜶, 𝜷)-DENSITY. □ Combining Lemmas 5.1 and 5.2, we have the desired result: Theorem 5.3. Algorithm 1 outputs an optimal solution to (𝜶, 𝜷)DENSITY.
# 5.2 Hierarchical structure and support size of the output of Algorithm 1
Here we observe some useful properties of the output of Algorithm 1. By the design of Algorithm 1, we see that the support of the output ˆ𝑝of the algorithm (i.e., an optimal solution to (𝜶, 𝜷)DENSITY) has a hierarchical structure. We denote the support of 𝑝∈Δ(2𝑉) by supp(𝑝) = {𝑆⊆𝑉| 𝑝𝑆> 0}.
Proof. Let (( ˆ𝑥𝑒)𝑒∈𝐸, ( ˆ𝑦𝑣)𝑣∈𝑉, ˆ𝑡) be a basic solution. Without loss of generality, we may assume that
{} (∀ {} ∈) Recall that ℓdenotes the number of different positive values in { ˆ𝑦𝑣| ˆ𝑦𝑣> 0, 𝑣∈𝑉}. Let 𝑉0 = {𝑣∈𝑉| ˆ𝑦𝑣= 0}. We divide 𝑉\ 𝑉0 into ℓsubsets of vertices sharing the same value of ˆ𝑦𝑣, denoted by 𝑉1, . . . ,𝑉ℓ. Let us focus on the constraints in LP (4) that are satisfied with equality. For each 𝑗= 0, 1, . . . , ℓ, let 𝐹𝑗⊆𝐸[𝑉𝑗] be a spanning forest in 𝐸[𝑉𝑗]. Let 𝜌be the number of connected components in 𝐸[𝑉1] ∪· · · ∪𝐸[𝑉ℓ], and let 𝜁be that of 𝐸[𝑉0]. We arbitrarily take 𝜁 vertices, denoted by𝑢1, . . . ,𝑢𝜁, one from each connected component in 𝐸[𝑉0]. We focus on the following constraints (satisfied with equality): �
(∀∈) where 𝐾′ = {𝑖∈[𝑘] | 𝑡= 𝛼𝑖· � 𝑒∈𝐸𝑖𝑤𝑖(𝑒)𝑥𝑒+ 𝛽𝑖}. We prove that the coefficient matrix of those constraints is not full-rank. For ease of discussion, we remove constraints that are represented by a linear combination of others. Specifically, it is enough to focus on the following constraints:  �
(7) (8) (9) (10) (11) (12)
There are two types of missing constraints. First, let 𝑒= {𝑢, 𝑣} ∈ 𝐸[𝑉𝑗] \ 𝐹𝑗(𝑗∈[ℓ]) such that 𝑥𝑒= 𝑦𝑣appears in (9). There exists a cycle𝐶in 𝐹𝑗∪{𝑒}. By a telescoping sum of the constraints (8) along 𝐶, i.e., 𝑦𝑣= 𝑥𝑒′, 𝑥𝑒′ = 𝑦𝑣′, ..., 𝑥𝑒′′ = 𝑦𝑢, we obtain 𝑦𝑢= 𝑦𝑣. Then, constraint 𝑥𝑒= 𝑦𝑢is obtained by summing 𝑥𝑒= 𝑦𝑣for 𝑦𝑢= 𝑦𝑣. Next, let 𝑣∈𝑉0 belong to a connected component containing 𝑢𝑗 (𝑗∈[𝜁]). By summing (8) along a path from 𝑣to 𝑢𝑗, i.e., 𝑦𝑣= 𝑥𝑒′, 𝑥𝑒′ = 𝑦𝑣′, ..., 𝑥𝑒′′ = 𝑦𝑢𝑗, we obtain𝑦𝑣= 𝑦𝑢𝑗. Then, constraint𝑦𝑣= 0 is obtained by summing 𝑦𝑣= 𝑦𝑢𝑗and 𝑦𝑢𝑗= 0 from (12). The rank of the coefficient matrix is equal to the number of constraints (7)–(12). For (7), we have at most 𝑘constraints; the
number of constraints (8) is 2(|𝑉| −𝜌−𝜁); that for (9) and (10) is |𝐸| −|𝑉| + 𝜌+ 𝜁; that for (11) and (12) is 1 + 𝜁. Therefore, we have at most |𝑉| + |𝐸| −𝜌+ 1 + 𝑘constraints. We have |𝑉| + |𝐸| + 1 variables in LP (4). If 𝜌> 𝑘, then we have at most |𝑉| + |𝐸| constraints, and hence the coefficient matrix of (7)–(12) cannot have rank |𝑉| + |𝐸| + 1. As the solution is basic, 𝜌≤𝑘must hold. Recall that each 𝑉𝑗(𝑗= 1, . . . , ℓ) has at least one connected component. Therefore, we have at most 𝑘different positive values of 𝑦𝑣’s, which implies that the output ˆ𝑝has support size at most 𝑘. □
# 6 PREPROCESSING
In this section, we present a simple, scalable preprocessing algorithm, which often reduces the size of the input networks significantly and results in a substantial speed-up. Specifically, the algorithm first computes an approximate solution by solving an LP, which is much smaller than LP (4) in practice, and then removes vertices from the original network using the information of the approximate solution obtained. We assume that 𝑆∗ 𝑖for all 𝑖∈[𝑘] are known in advance because they can be computed efficiently using Charikar’s LP-based algorithm [9] together with the preprocessing algorithm introduced by Balalau et al. [4]. To describe our algorithm, we introduce some notations. For 𝑆⊆𝑉, 𝑣∈𝑆, and 𝑖∈[𝑘], let 𝑑𝑖(𝑆, 𝑣) denote the weighted degree of 𝑣in the subgraph induced by 𝑆in layer 𝑖, i.e., 𝑑𝑖(𝑆, 𝑣) �� 𝑒∈𝐸𝑖[𝑆]: 𝑣∈𝑒𝑤𝑖(𝑒). When 𝑆= 𝑉, we simply write 𝑑𝑖(𝑣). We first describe a fast algorithm for finding an approximate solution for (𝜶, 𝜷)-DENSITY. Specifically, we compute a probability distribution𝑞∈Δ(2𝑉) that maximizes the (𝜶, 𝜷)-density under the constraint that 𝑞𝑆= 0 for all 𝑆∈2𝑉\ {𝑆∗ 1, . . . ,𝑆∗ 𝑘}. The distribution can be found by solving the following LP:
(13)
(∀𝑗∈[𝑘]).
Note that this LP has 𝑘+ 1 variables and 2𝑘+ 1 constraints. As 𝑘 is usually much smaller than |𝑉| and |𝐸|, this LP is much smaller than LP (4) in practice. Next we describe an algorithm for removing vertices using the information of the above approximate solution 𝑞. Let ℓ∗be the (𝜶, 𝜷)-density of 𝑞, i.e., the optimal value of LP (13). Note that this is a lower bound on the optimal value of (𝜶, 𝜷)-DENSITY. Our algorithm iteratively removes any vertex 𝑣∗that satisfies max𝑖∈[𝑘] [𝛼𝑖· 𝑑𝑖(𝑉′, 𝑣∗) + 𝛽𝑖] < ℓ∗, where 𝑉′ is a remaining vertex set (initially 𝑉′ = 𝑉), as long as there exists such a vertex. For reference, we describe the procedure in Algorithm 2. This algorithm can be implemented to run in 𝑂(𝑘|𝐸| + |𝑉| log |𝑉|) time. From now on, we demonstrate that the algorithm does not remove any vertex that is contained in a subset in supp(𝑝), where 𝑝 is an arbitrary optimal solution to (𝜶, 𝜷)-DENSITY. The following is a key lemma in our analysis.
Algorithm 2: Remove useless vertices
Input
: (𝑉, (𝐸𝑖)𝑖∈[𝑘]) with 𝑤1, . . . ,𝑤𝑘, and ℓ∗∈R
Output: (𝑉′, (𝐸𝑖[𝑉′])𝑖∈[𝑘])
1 𝑉′ ←𝑉;
2 while True do
3
Let 𝑣∗∈arg min𝑣∈𝑉′ max𝑖∈[𝑘] [𝛼𝑖· 𝑑𝑖(𝑉′, 𝑣) + 𝛽𝑖];
4
if max𝑖∈[𝑘] [𝛼𝑖· 𝑑𝑖(𝑉′, 𝑣∗) + 𝛽𝑖] ≥ℓ∗then
5
return (𝑉′, (𝐸𝑖[𝑉′])𝑖∈[𝑘]).
6
else 𝑉′ ←𝑉′ \ {𝑣∗};
Lemma 6.1. Let ℓ∗be a lower bound on the optimal value of (𝜶, 𝜷)DENSITY. If max𝑖∈[𝑘] [𝛼𝑖· 𝑑𝑖(𝑣∗) + 𝛽𝑖] < ℓ∗, then 𝑦𝑣∗= 0 for any optimal solution to LP (4). Proof. We prove the lemma by contradiction. We denote by (( ˆ𝑥𝑒)𝑒∈𝐸, ( ˆ𝑦𝑣)𝑣∈𝑉, ˆ𝑡) an optimal solution to LP (4) and let 𝑣∗∈𝑉 be a vertex that satisfies 𝛼𝑖·𝑑𝑖(𝑣∗) + 𝛽𝑖< ℓ∗for all 𝑖∈[𝑘]. Suppose for contradiction that ˆ𝑦𝑣∗> 0. We construct a solution ((𝑥𝑒)𝑒∈𝐸, (𝑦𝑣)𝑣∈𝑉,𝑡) of LP (4) as follows: � �
It is easy to see that ((𝑥𝑒)𝑒∈𝐸, (𝑦𝑣)𝑣∈𝑉,𝑡) is a feasible solution of LP (4). Moreover, we have � ∑︁ �
 −  − where the first inequality follows from ˆ𝑥𝑒≤ˆ𝑦𝑣∗for each 𝑒∋𝑣∗, the second inequality follows from the assumptions 𝛼𝑖·𝑑𝑖(𝑣∗) +𝛽𝑖< ℓ∗ and ˆ𝑦𝑣∗> 0, and the third inequality follows from Lemma 5.2. This contradicts the optimality of (( ˆ𝑥𝑒)𝑒∈𝐸, ( ˆ𝑦𝑣)𝑣∈𝑉, ˆ𝑡). □
□
Theorem 6.2. Let ℓ∗be a lower bound on the optimal value of (𝜶, 𝜷)-DENSITY. Then, any vertex 𝑣∗that satisfies max𝑖∈[𝑘] [𝛼𝑖· 𝑑𝑖(𝑣∗) + 𝛽𝑖] < ℓ∗is not contained in any subset in the support of any optimal solution to (𝜶, 𝜷)-DENSITY.
Proof. Let 𝑣∗be any vertex with max𝑖∈[𝑘] [𝛼𝑖· 𝑑𝑖(𝑣∗) + 𝛽𝑖] < ℓ∗. Let ˆ𝑝be any optimal solution to (𝜶, 𝜷)-DENSITY. Construct (( ˆ𝑥𝑒)𝑒∈𝐸, ( ˆ𝑦𝑣)𝑣∈𝑉, ˆ𝑡) from ˆ𝑝as in (6). From Lemma 5.1 and the proof of Lemma 5.2, we see that (( ˆ𝑥𝑒)𝑒∈𝐸, ( ˆ𝑦𝑣)𝑣∈𝑉, ˆ𝑡) is an optimal solution to LP (4). Thus, by Lemma 6.1, we have ˆ𝑦𝑣∗= 0. By the construction of ˆ𝑦𝑣∗in (6), ˆ𝑝𝑆= 0 for all 𝑆⊆𝑉containing 𝑣∗. □
This theorem indicates that Algorithm 2 does not remove any vertex that is contained in a subset in the support of any optimal solution to (𝜶, 𝜷)-DENSITY.
# 7 EXPERIMENTAL EVALUATION
In this section, we conduct computational experiments using synthetic graphs and real-world networks to verify the validity of our proposed model and to evaluate the performance of our proposed algorithms. All experiments were conducted on a machine equipped with Intel Xeon W 10-core processor and 64GB RAM. Algorithms were implemented in Python using Gurobi Optimizer 9.0.2.
# 7.1 Validity of our model
Here we aim to verify the validity of our model using synthetic graphs. To this end, we use a randomly generated multilayer network with a planted clique, and examine whether an optimal solution detects vertex subsets close to the clique. We first explain our random procedure for generating multilayer networks. We produce an (unweighted) random power-law graph as a layer using the Chung–Lu model [12], where we first specify an expected degree 𝑑𝑣for each 𝑣∈𝑉according to the power-law distribution with exponent 𝛽, and then connect each pair of vertices {𝑢, 𝑣} with probability 𝑑𝑢·𝑑𝑣 � 𝑟∈𝑉𝑑𝑟. Note that the graph becomes sparser as the exponent 𝛽increases. In this multilayer network, we randomly select a vertex subset 𝑉𝑐with some size, and plant a clique on 𝑉𝑐(in some specified layers). To evaluate the performance of optimal solution 𝑝∈Δ(2𝑉) to (𝜶, 𝜷)-DENSITY in the above multilayer network, we introduce the following measure, which we refer to as the (expected) F measure:
  ∼ � |𝑉𝑐| �  � ⊆ |𝑉𝑐|  if 𝑝tends to be close to 𝑉𝑐. We first investigate the case where a clique is planted in all layers. We generate 𝑘(= 1, 2, 3, 4, 5) power-law graphs (i.e., layers) with 𝛽= 2.3 on 𝑉with |𝑉| = 1,000. Then we randomly select a subset 𝑉𝑐⊆𝑉consisting of 10 vertices, and plant a clique on 𝑉𝑐in all layers. The performance of optimal solutions to (𝜶, 𝜷)-DENSITY is shown in Figure 2(a). As can be seen, for any metric, the F measure is reasonably large for 𝑘≥2, meaning that our algorithm tends to detect 𝑉𝑐using the information of multiple layers. Next we investigate the case where a clique is planted in only one layer. We generate 𝑘(= 1, 2, 3, 4, 5) power-law graphs with 𝛽= 3.0 on 𝑉with |𝑉| = 1,000. Then we randomly select 𝑉𝑐⊆ 𝑉consisting of 20 vertices, but plant a clique on 𝑉𝑐only in one randomly selected layer. The performance of optimal solutions are described in Figure 2(b). As can be seen, the F measure becomes
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3b54/3b54731a-c7a3-4fad-a2a4-15fd0ee6cc1e.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/83d3/83d3e19b-1ae2-4d03-aee7-b70c855b357e.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d02e/d02e043a-dd20-471c-acb2-66858bdf653b.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Clique in all layers.</div>
<div style="text-align: center;">(b) Clique in only one layer.</div>
Figure 2: Performance of optimal solutions. Each point corresponds to the average value over 100 network realizations.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c012/c012d67d-371e-49c5-b3e9-f7c3d4c34f50.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7e27/7e27b357-e233-4948-b10a-49b9f15e6c1c.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/dad9/dad9593a-daaa-4390-b05c-74cda1f22db1.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Clique in all layers.</div>
<div style="text-align: center;">(b) Clique in only one layer.</div>
Figure 3: Performance of single vertex subsets obtained by optimal solutions. The same averaging procedure is applied.
# Figure 3: Performance of single vertex subsets obtained by optimal solutions. The same averaging procedure is applied.
smaller as the number of layers increases. Among the three metrics, the regret performs particularly well because it concentrates on the layer containing the clique from its definition; therefore, the regret metric seems most suitable for robust analysis with noisy layers. The robust ratio performs second best; it also cares about the layer containing the clique. Finally we conclude this subsection by evaluating vertex subsets that are obtained from optimal solutions 𝑝∈Δ(2𝑉). To verify the validity of our model, we select a vertex subset attaining the highest probability in 𝑝∈Δ(2𝑉). As our algorithm does not know about𝑉𝑐, it cannot select a vertex subset using the F measure. For reference, we also run two baseline algorithms, DCS-LP and DCS-Greedy, designed by Jethava and Beerenwinkel [27]. Note that DCS-LP can be seen as the algorithm that selects a vertex subset with the largest minimum density value over layers from the support of 𝑝∈Δ(2𝑉), where 𝑝∈Δ(2𝑉) is an optimal solution to our algorithm with the density metric. The results are depicted in Figure 3, where we employed the same experimental settings as above and used the usual (deterministic) F measure for evaluation. As can be seen, the trend of our algorithm is similar to that observed in the above experiments; all metrics almost detect 𝑉𝑐for the all-layers setting, but only the regret metric is successful for the only-one-layer setting. As for the baseline methods, DCS-LP is comparable to our algorithm with the density metric, while DCS-Greedy performs quite poorly.
# 7.2 Performance of our algorithm
Here we examine the performance of our proposed algorithm using publicly-available real-world multilayer networks. Table 1 shows the results together with the summary of the characteristics of the datasets. Even for large networks, our algorithm obtains an optimal
Table 1: Performance of our algorithm for real-world datasets. OPT and |supp| denote the optimal value of (𝜶, 𝜷)-DENSITY and the support size of the optimal solution, respectively. Note that the objective values for the regret are negated. LB indicates the optimal value of LP (13). |𝑉′| and |𝐸[𝑉′]|, respectively, denote the number of vertices and edges after running Algorithm 2. Preprocess time represents the running time of Algorithm 2. Total time represents the running time of our proposed algorithm, i.e., Algorithm 1 together with Algorithm 2. For reference, the running time of Algorithm 1 without Algorithm 2 is also presented in the next column. The last three columns report the performance of single vertex subsets obtained by optimal solutions. For each instance and metric, the best value among the algorithms is in bold.
Dataset
|𝑉|
|𝐸|
𝑘
Metric
OPT |supp|
LB
|𝑉′| |𝐸[𝑉′]|
Preprocess
time (s)
Total
time (s)
w/o preprocess
time (s)
Density
Robust
ratio
Regret
Density
1.1950
3
1.1313
123
3,126
0.04
0.66
0.73
1.1875
0.6579
-0.6282
WILDBIRDS
202
4,574
6 Robust ratio
0.7707
4
0.7364
121
2,901
0.05
0.77
0.79
1.0058 0.6970
-0.5065
[18]∗
Regret
-0.4122
2
-0.4473
124
2,976
0.04
0.71
0.77
0.9129
0.6518
-0.5821
Density
4.7023
2
4.4257
730
6,403
1208.79
1304.28
1834.88
4.6316
0.6399
-3.1581
AS-733
7,716
24,179 733 Robust ratio
0.7721
4
0.7295
278
3,452
1165.87
1211.82
1906.21
4.5789
0.7057
-1.9621
[33]†
Regret
-6.9861
4
-1.8023
228
3,020
1289.15
1329.56
1950.33
3.6154 0.7317
-1.6316
Density
12.0263
1
12.0263
118
1,815
0.29
0.92
8.31 12.0263
0.9666
-0.4382
Oregon1
11,492
26,461
9 Robust ratio
0.9808
3
0.9738
110
1,702
0.48
1.02
7.55
11.9016 0.9728
-0.3578
[33]†
Regret
-0.2544
3
-0.3441
110
1,702
0.50
0.95
9.72
11.8889 0.9728
-0.3578
Density
22.2503
2
21.8228
1,130
14,166
1.90
4.31
93.30 18.4000
0.1183
-247.6000
MoscowAthletics2013
88,804 186,846
3 Robust ratio
0.4709
3
0.3566
105
787
2.02
2.12
112.50
13.2000 0.4197 -181.1667
[36]‡
Regret
-125.4477
2 -130.1752
88,804 186,846
1.40
336.56
334.43
0.4286
0.0186
-200.8333
Density
3.5649
2
3.5077
17,551 184,659
1.98
332.02
5466.73
3.5625
0.1768
-17.0770
NYClimateMarch2014 102,439 329,474
3 Robust ratio
0.6661
3
0.5503
3,901
78,126
2.36
147.40
6058.12
2.3839 0.6652
-6.8047
[36]‡
Regret
-2.3052
3
-3.5835 102,439 329,473
1.58 18575.97
18909.39
1.2785
0.3576
-2.3221
Density
60.7462
2
60.7462
375
4,536
11.07
11.72
4268.72 52.7143
0.0840
-837.8095
Cannes2013
438,537 848,017
3 Robust ratio
0.3633
3
0.3633
246
1,617
11.53
11.72
3981.18
42.6000 0.2441
-365.0667
[36]‡
Regret
-132.8628
2 -132.8628 438,537 848,017
7.31
4021.19
4022.75
0.6667
0.0073 -155.5000
Density
1.1891
10
1.1236 191,074 559,628
18.52
7826.41
13787.37
0.4146
0.0387
-31.2295
DBLP
513,627 888,353
10 Robust ratio
0.1178
10
0.1125 317,231 687,335
17.01 15323.66
22828.17
0.6067 0.0607
-20.2759
[20]§
Regret
-11.6944
3
-11.6963 513,627 888,353
14.78 37266.95
37085.30
0.1045
0.0114
-13.4481
∗http://networkrepository.com
† http://snap.stanford.edu
‡ https://comunelab.fbk.eu/data.php
§ https://goo.gl/8741Gs
solution in reasonable time. The preprocessing algorithm often reduces the size of the networks significantly using a reasonably large lower bound computed by LP (13) and results in a substantial speed-up. As an extreme example, for Cannes2013 with the density and robust ratio metrics, the lower bound attains the optimal value, and the number of vertices is reduced by more than 99.9%, which makes the computation more than 300 times faster. Consistent with our theoretical analysis, |supp| is at most 𝑘. For AS-733, 𝑘is quite large but |supp| is still small. Finally we evaluate the single vertex subsets obtained from optimal solutions. For 𝑝∈Δ(2𝑉), we select a vertex subset from supp(𝑝) that optimizes the metric employed in the algorithm. Note that in this setting, the output of DCS-LP coincides with that obtained by our algorithm with the density metric. As DCS-Greedy performed quite poorly, it is omitted. The results are shown in the last three columns of Table 1. Although there are a few exceptions, the algorithm with a metric performs best in terms of the metric employed. A critical fact is that depending only on the density metric, we may fail to obtain meaningful structure from networks. For example, the algorithm with the robust ratio admits a particularly large robust ratio value of 0.6652 for NYClimateMarch2014, meaning that the vertex subset obtained achieves an approximation ratio of 0.6652 for all layers. Moreover, the algorithm with the regret metric admits a particularly small regret value of 155.5000 for Cannes2013. For those instances, the algorithm with the density metric (i.e., DCS-LP) performs poorly in terms of those metrics,
respectively. From the above, it seems quite important to select an appropriate metric depending on the practical purpose at hand.
# 8 CONCLUSION
In this paper, we have introduced a novel optimization model and algorithms for dense subgraph discovery in multilayer networks. There are several possible directions for future research. One direction is to improve the scalability of our algorithm, particularly for the regret metric, for which our preprocessing algorithm does not necessarily perform well. Another direction is to apply our model and algorithms to some real-world applications of multilayernetwork analysis. Investigating multilayer-network counterparts of some existing generalizations of the densest subgraph problem (see e.g., [29, 34, 40, 43]) is also interesting future work.
# ACKNOWLEDGMENTS
This work was partially supported by JST PRESTO Grant Number JPMJPR2122 and JSPS KAKENHI Grant Numbers JP17K12646, JP19K20218, JP20K19739, JP21K17708, and JP21H03397.
# REFERENCES
