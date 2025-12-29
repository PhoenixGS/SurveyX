# Multi-hop Attention Graph Neural Networks
# Guangtao Wang1∗, Rex Ying2 , Jing Huang1 and Jure Leskovec2
2Computer Science, Stanford University guangtao.wang@jd.com, rexying@stanford.edu, jing.huang@jd.com, jure@cs.stanford
Computer Science, Stanford University guangtao.wang@jd.com, rexying@stanford.edu, jing.huang@jd.com, jure@cs.stanford.edu
guangtao.wang@jd.com, rexying@stanford.edu, jing.huang@jd.com, jure@cs.stanford.edu
# Abstract
25 Aug 20
Self-attention mechanism in graph neural networks (GNNs) led to state-of-the-art performance on many graph representation learning tasks. Currently, at every layer, a node computes attention independently for each of its graph neighbors. However, such attention mechanism is limited as it does not consider nodes that are not connected by an edge but can provide important network context information. Here we propose Multi-hop Attention Graph Neural Network (MAGNA), a principled way to incorporate multi-hop context information into every layer of GNN attention computation. MAGNA diffuses the attention scores across the network, which increases the “receptive field” for every layer of the GNN. Unlike previous approaches, MAGNA uses a diffusion prior on attention values, to efficiently account for all paths between the pair of not-connected nodes. We demonstrate theoretically and experimentally that MAGNA captures large-scale structural information in every layer, and has a low-pass effect that eliminates noisy high-frequency information from the graph. Experimental results on node classification as well as knowledge graph completion benchmarks show that MAGNA achieves state-of-the-art results: MAGNA achieves up to 5.7% relative error reduction over the previous state-of-the-art on Cora, Citeseer, and Pubmed. MAGNA also obtains strong performance on a large-scale Open Graph Benchmark dataset. Finally, on knowledge graph completion MAGNA advances state-of-the-art on WN18RR and FB15k-237 across four different performance metrics.
25
[cs.LG]
# 1 Introduction
Self-attention [Bahdanau et al., 2015; Vaswani et al., 2017] has pushed the state-of-the-art in many domains including graph presentation learning [Devlin et al., 2019]. Graph Attention Network (GAT) [Veliˇckovi´c et al., 2018] and related models [Li et al., 2018; Wang et al., 2019a; Liu et al., 2019;
∗Contact Author, xjtuwgt@gmail.com
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c8b1/c8b1f70a-9b7e-43ad-b254-096becc8dcb6.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/65e1/65e187ad-8341-439c-b76c-54f2d0a890e6.png" style="width: 50%;"></div>
<div style="text-align: center;">%′&,' = 2 [%.,', %&,. , [%&,']) %′.,/ = 2([%',/, %.,'])</div>
<div style="text-align: center;">%&,' = )(+&, +')  %.,/ = 0</div>
Figure 1: Multi-hop attention diffusion. Consider making a prediction at nodes A and D. Left: A single GAT layer computes attention scores α between directly connected pairs of nodes (i.e., edges) and thus αD,C = 0. Furthermore, the attention αA,B between A and B only depends on node representations of A and B. Right: A single MAGNA layer: (1) captures the information of D’s two-hop neighbor node C via multi-hop attention α′ D,C; and (2) enhances graph structure learning by considering all paths between nodes via diffused attention, which is based on powers of graph adjacency matrix. MAGNA makes use of node D’s features for attention computation between A and B. This means that two-hop attention in MAGNA is context (node D) dependent.
Oono and Suzuki, 2020] developed attention mechanism for Graph Neural Networks (GNNs), which compute attention scores between nodes connected by an edge, allowing the model to attend to messages of node’s neighbors. However, such attention computation on pairs of nodes connected by edges implies that a node can only attend to its immediate neighbors to compute its (next layer) representation. This implies that receptive field of a single GNN layer is restricted to one-hop network neighborhoods. Although stacking multiple GAT layers could in principle enlarge the receptive field and learn non-neighboring interactions, such deep GAT architectures suffer from the oversmoothing problem [Wang et al., 2019a; Liu et al., 2019; Oono and Suzuki, 2020] and do not perform well. Furthermore, edge attention in a single GAT layer is based solely on representations of the two nodes at the edge endpoints, and does not depend on their graph neighborhood context. In other words, the one-hop attention mechanism in GATs limits their ability to explore the relationship between the broader graph structure. While previous works [Xu et al., 2018; Klicpera et al., 2019b] have shown advantages in performing multi-hop message-passing in a single layer, these ap-
rating multi-hop neighboring context into the attention computation in graph neural networks remains to be explored. Here we present Multi-hop Attention Graph Neural Network (MAGNA), an effective multi-hop self-attention mechanism for graph structured data. MAGNA uses a novel graph attention diffusion layer (Figure 1), where we first compute attention weights on edges (represented by solid arrows), and then compute self-attention weights (dotted arrows) between disconnected pairs of nodes through an attention diffusion process using the attention weights on the edges. Our model has two main advantages: (1) MAGNA captures long-range interactions between nodes that are not directly connected but may be multiple hops away. Thus the model enables effective long-range message passing, from important nodes multiple hops away. (2) The attention computation in MAGNA is context-dependent. The attention value in GATs [Veliˇckovi´c et al., 2018] only depends on node representations of the previous layer, and is zero between nonconnected pairs of nodes. In contrast, for any pair of nodes within a chosen multi-hop neighborhood, MAGNA computes attention by aggregating the attention scores over all the possible paths (length ≥1) connecting the two nodes. Mathematically we show that MAGNA places a Personalized Page Rank (PPR) prior on the attention values. We further apply spectral graph analysis to show that MAGNA emphasizes on large-scale graph structure and lowering highfrequency noise in graphs. Specifically, MAGNA enlarges the lower Laplacian eigen-values, which correspond to the large-scale structure in the graph, and suppresses the higher Laplacian eigen-values which correspond to more noisy and fine-grained information in the graph. We experiment on standard datasets for semi-supervised node classification as well as knowledge graph completion. Experiments show that MAGNA achieves state-of-the-art results: MAGNA achieves up to 5.7% relative error reduction over previous state-of-the-art on Cora, Citeseer, and Pubmed. MAGNA also obtains better performance on a large-scale Open Graph Benchmark dataset. On knowledge graph completion, MAGNA advances state-of-the-art on WN18RR and FB15k-237 across four metrics, with the largest gain of 7.1% in the metric of Hit at 1. Furthermore, we show that MAGNA with just 3 layers and 6 hop wide attention per layer significantly out-performs GAT with 18 layers, even though both architectures have the same receptive field. Moreover, our ablation study reveals the synergistic effect of the essential components of MAGNA, including layer normalization and multi-hop diffused attention. We further observe that compared to GAT, the attention values learned by MAGNA have higher diversity, indicating the ability to better pay attention to important nodes.
# 2 Multi-hop Attention Graph Neural Network (MAGNA)
We first discuss the background and explain the novel multihop attention diffusion module and the MAGNA architecture.
# 2.1 Preliminaries
Let G = (V, E) be a given graph, where V is the set of Nn nodes, E ⊆V×V is the set of Ne edges connecting M pairs of nodes in V. Each node v ∈V and each edge e ∈E are associated with their type mapping functions: φ : V →T and ψ : E →R. Here T and R denote the sets of node types (labels) and edge/relation types. Our framework supports learning on heterogeneous graphs with multiple elements in R. A general Graph Neural Network (GNN) approach learns an embedding that maps nodes and/or edge types into a continuous vector space. Let X ∈RNn×dn and R ∈RNr×dr be the node embedding and edge/relation type embedding, where Nn = |V|, Nr = |R|, dn and dr represent the embedding dimension of node and edge/relation types, each row xi = X[i :] represents the embedding of node vi (1 ≤i ≤ Nn), and rj = R[j :] represents the embedding of relation rj (1 ≤j ≤Nr). MAGNA builds on GNNs, while bringing together the benefits of Graph Attention and Diffusion techniques.
# 2.2 Multi-hop Attention Diffusion
We first introduce attention diffusion to compute the multihop attention directly, which operates on the MAGNA’s attention scores at each layer. The input to the attention diffusion operator is a set of triples (vi, rk, vj), where vi, vj are nodes and rk is the edge type. MAGNA first computes the attention scores on all edges. The attention diffusion module then computes the attention values between pairs of nodes that are not directly connected by an edge, based on the edge attention scores, via a diffusion process. The attention diffusion module can then be used as a component in MAGNA architecture, which we will further elaborate in Section 2.3.
We first introduce attention diffusion to compute the multihop attention directly, which operates on the MAGNA’s attention scores at each layer. The input to the attention diffusion operator is a set of triples (vi, rk, vj), where vi, vj are nodes and rk is the edge type. MAGNA first computes the attention scores on all edges. The attention diffusion module then computes the attention values between pairs of nodes that are not directly connected by an edge, based on the edge attention scores, via a diffusion process. The attention diffusion module can then be used as a component in MAGNA architecture, which we will further elaborate in Section 2.3. Edge Attention Computation At each layer l, a vector message is computed for each triple (vi, rk, vj). To compute the representation of vj at layer l+1, all messages from triples incident to vj are aggregated into a single message, which is then used to update vl+1 j . In the first stage, the attention score s of an edge (vi, rk, vj) is computed by the following:
(1)
W (l) r ∈
    where δ = LeakyReLU, W (l) h , W (l) t ∈Rd(l)×d(l), W (l) r ∈ Rd(l)×dr and v(l) a ∈R1×3d(l) are the trainable weights shared by l-th layer. h(l) i ∈Rd(l) represents the embedding of node i at l-th layer, and h(0) i = xi. rk is the trainable relation embedding of the k-th relation type rk (1 ≤k ≤Nr), and a∥b denotes concatenation of embedding vectors a and b. For graphs with no relation type, we treat as a degenerate categorical distribution with 1 category1. Applying Eq. 1 on each edge of the graph G, we obtain an attention score matrix S(l): �
(2)
1In this case, we can view that there is only one “pseudo” relation type (category), i.e., Nr = 1
Subsequently we obtain the attention matrix A(l) by performing row-wised softmax over the score matrix S(l): A(l) = softmax(S(l)). A(l) ij denotes the attention value at layer l when aggregating message from node j to node i. Attention Diffusion for Multi-hop Neighbors In the second stage, we further enable attention between nodes that are not directly connected in the network. We achieve this via the following attention diffusion procedure. The procedure computes the attention scores of multi-hop neighbors via graph diffusion based on the powers of the 1-hop attention matrix A:
(3)
where θi is the attention decay factor and θi > θi+1. The powers of attention matrix, Ai, give us the number of relation paths from node h to node t of length up to i, increasing the receptive field of the attention (Figure 1). Importantly, the mechanism allows the attention between two nodes to not only depend on their previous layer representations, but also taking into account of the paths between the nodes, effectively creating attention shortcuts between nodes that are not directly connected (Figure 1). Attention through each path is also weighted differently, depending on θ and the path length. In our implementation we utilize the geometric distribution θi = α(1 −α)i, where α ∈(0, 1]. The choice is based on the inductive bias that nodes further away should be weighted less in message aggregation, and nodes with different relation path lengths to the target node are sequentially weighted in an independent manner. In addition, notice that if we define θ0 = α ∈(0, 1], A0 = I, then Eq. 3 gives the Personalized Page Rank (PPR) procedure on the graph with the attention matrix A and teleport probability α. Hence the diffused attention weights, Aij, can be seen as the influence of node j to node i. We further elaborate the significance of this observation in Section 4.3. We can also view Aij as the attention value of node j to i since �Nn j=1 Aij = 1.2 We then define the graph attention diffusion based feature aggregation as
AttDiff(G, H(l), Θ) = AH(l),
(4)
where Θ is the set of parameters for computing attention. Thanks to the diffusion process defined in Eq. 3, MAGNA uses the same number of parameters as if we were only computing attention between nodes connected via edges. This ensures runtime efficiency (refer to Appendix3 A for complexity analysis) and model generalization.
where Θ is the set of parameters for computing attention. Thanks to the diffusion process defined in Eq. 3, MAGNA uses the same number of parameters as if we were only computing attention between nodes connected via edges. This ensures runtime efficiency (refer to Appendix3 A for complexity analysis) and model generalization. Approximate Computation for Attention Diffusion For large graphs computing the exact attention diffusion matrix A using Eq. 3 may be prohibitively expensive, due to computing the powers of the attention matrix [Klicpera et al., 2019a]. To resolve this bottleneck, we proceed as follows: Let H(l) be the input entity embedding of the l-th layer (H(0) = X)
Approximate Computation for Attention Diffusion For large graphs computing the exact attention diffusion matrix A using Eq. 3 may be prohibitively expensive, due to computing the powers of the attention matrix [Klicpera et al., 2019a]. To resolve this bottleneck, we proceed as follows: Let H(l) be the input entity embedding of the l-th layer (H(0) = X)
2Obtained by the definition A(l) = softmax(S(l)) and Eq. 3. 3Appendix can be found via https://arxiv.org/abs/2009.14332
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c66b/c66b8fd4-a6e4-49dd-ac45-824710b15959.png" style="width: 50%;"></div>
<div style="text-align: center;">Inputs: Initial node and  relation embeddings !", $</div>
Figure 2: MAGNA Architecture. Each MAGNA block consists of attention computation, attention diffusion, layer normalization, feed forward layers, and 2 residual connections for each block. MAGNA blocks can be stacked to constitute a deep model. As illustrated on the right, context-dependent attention is achieved via the attention diffusion process. Here vi, vj, vp, vq ∈V are nodes in the graph.
and θi = α(1 −α)i. Since MAGNA only requires aggregation via AH(l), we can approximate AH(l) by defining a sequence Z(K) which converges to the true value of AH(l) (Proposition 1) as K →∞:
→∞ A In the Appendix we give the proof which relies on the expansion of Eq. 5. Using the above approximation, the complexity of attention computation with diffusion is still O(|E|), with a constant factor corresponding to the number of hops K. In practice, we find that choosing the values of K such that 3 ≤K ≤10 results in good model performance. Many real-world graphs exhibit small-world property, in which case even a smaller value of K is sufficient. For graphs with larger diameter, we choose larger K, and lower the value of α.
# 2.3 Multi-hop Attention based GNN Architecture
Figure 2 provides an architecture overview of the MAGNA Block that can be stacked multiple times. Multi-head Graph Attention Diffusion Layer Multi-head attention [Vaswani et al., 2017; Veliˇckovi´c et al., 2018] is used to allow the model to jointly attend to information from different representation sub-spaces at different viewpoints. In Eq. 6, the attention diffusion for each head i is computed separately with Eq. 4, and aggregated:
Figure 2 provides an architecture overview of the MAGNA Block that can be stacked multiple times.
(6)
���� � headi = AttDiff(G, ˜ H(l), Θi), ˜ H(l) = LN(H(l)),
where ∥denotes concatenation and Θi are the parameters in Eq. 1 for the i-th head (1 ≤i ≤M), Wo represents a parameter matrix, and LN = LayerNorm. Since we calculate the attention diffusion in a recursive way in Eq. 5, we add layer normalization which helpful to stabilize the recurrent computation procedure [Ba et al., 2016]. Deep Aggregation Moreover our MAGNA block contains a fully connected feed-forward sub-layer, which consists of a two-layer feed-forward network. We also add the layer normalization and residual connection in both sub-layers, allowing for a more expressive aggregation step for each block [Xiong et al., 2020]:
(7)
� � MAGNA generalizes GAT MAGNA extends GAT via the diffusion process. The feature aggregation in GAT is H(l+1) = σ(AH(l)W (l)), where σ represents the activation function. We can divide GAT layer into two components as follows:
(8)
���� � �� � In component (1), MAGNA removes the restriction of attending to direct neighbors, without requiring additional parameters as A is induced from A. For component (2) MAGNA uses layer normalization and deep aggregation to achieve higher expressive power compared to elu nonlinearity in GAT.
# 3 Analysis of Graph Attention Diffusion
In this section, we investigate the benefits of MAGNA from the viewpoint of discrete signal processing on graphs [Sandryhaila and Moura, 2013b]. Our first result demonstrates that MAGNA can better capture large-scale structural information. Our second result explores the relation between MAGNA and Personalized PageRank (PPR).
# 3.1 Spectral Properties of Graph Attention Diffusion
# 3.1 Spectral Properties of Graph Attention Diffusion
We view the attention matrix A of GAT, and A of MAGNA as weighted adjacency matrices, and apply Graph Fourier transform and spectral analysis (details in Appendix) to show the effect of MAGNA as a graph low-pass filter, being able to more effectively capture large-scale structure in graphs. By Eq. 3, the sum of each row of either A or A is 1. Hence the normalized graph Laplacians are ˆLsym = I −A and Lsym = I −A for A and A respectively. We can get the following proposition: Proposition 2. Let ˆλg i and λg i be the i-th eigeinvalues of ˆLsym and Lsym.
(9)
Refer to Appendix for the proof. We additionally have λg i ∈[0, 2] (proved by [Ng et al., 2002]). Eq. 9 shows that
when λg i is small such that α 1−α + λg i < 1, then ˆλg i > λg i , whereas for large λg i , ˆλg i < λg i . This relation indicates that the use of A increases smaller eigenvalues and decreases larger eigenvalues4. See Section 4.3 for its empirical evidence. The low-pass effect increases with smaller α. The eigenvalues of the low-frequency signals describe the large-scale structure in the graph [Ng et al., 2002] and have been shown to be crucial in graph tasks [Klicpera et al., 2019b]. As λg i ∈[0, 2] [Ng et al., 2002] and α 1−α > 0, the reciprocal format in Eq. 9 will amplify the ratio of lower eigenvalues to the sum of all eigenvalues. In contrast, high eigenvalues corresponding to noise are suppressed.
# 3.2 Personalized PageRank Meets Graph Attention Diffusion
We can also view the attention matrix A as a random walk matrix on graph G since �Nn j=1 Ai,j = 1 and Ai,j > 0. If we perform Personalized PageRank (PPR) with parameter α ∈ (0, 1] on G with transition matrix A, the fully Personalized PageRank [Lofgren, 2015] is defined as:
(10)
Using the power series expansion for the matrix inverse, we obtain
(11)
Comparing to the diffusion Equation 3 with θi = α(1−α)i, we have the following proposition. Proposition 3. Graph attention diffusion defines a personalized page rank with parameter α ∈(0, 1] on G with transition matrix A, i.e., A = Appr. The parameter α in MAGNA is equivalent to the teleport probability of PPR. PPR provides a good relevance score between nodes in a weighted graph (the weights from the attention matrix A). In summary, MAGNA places a PPR prior over node pairwise attention scores: the diffused attention between node i and j depends on the attention scores on the edges of all paths between i and j.
# 4 Experiments
We evaluate MAGNA on two classical tasks5: (1) on node classification we achieve an average of 5.7% relative error reduction; (2) on knowledge graph completion we achieve 7.1% relative improvement in the Hit@1 metric.6
# 4.1 Task 1: Node Classification
Datasets We employ four benchmark datasets for node classification: (1) standard citation network benchmarks Cora, Citeseer and Pubmed [Sen et al., 2008; Kipf and
4The eigenvalues of A and A correspond to the same eigenvectors, as shown in Proposition 2 in Appendix. 5All datasets are public. And our implementation is available at https://github.com/xjtuwgt/GNN-MAGNA 6Please see the definitions of these two tasks in Appendix.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7f03/7f035846-5324-48d1-9fb2-e32048047996.png" style="width: 50%;"></div>
Models
Cora
Citeseer
Pubmed
Baselines
GCN [Kipf and Welling, 2017]
81.5
70.3
79.0
Cheby [Defferrard et al., 2016]
81.2
69.8
74.4
DualGCN [Zhuang and Ma, 2018]
83.5
72.6
80.0
JKNet [Xu et al., 2018]⋆
81.1
69.8
78.1
LGCN [Gao et al., 2018]
83.3 ± 0.5
73.0 ± 0.6
79.5 ± 0.2
Diff-GCN [Klicpera et al., 2019b]
83.6 ± 0.2
73.4 ± 0.3
79.6 ± 0.4
APPNP [Klicpera et al., 2019a]
84.3 ± 0.2
71.1 ± 0.4
79.7 ± 0.3
g-U-Nets [Gao and Ji, 2019]
84.4 ± 0.6
73.2 ± 0.5
79.6 ± 0.2
GAT [Veliˇckovi´c et al., 2018]
83.0 ± 0.7
72.5 ± 0.7
79.0 ± 0.3
Abl.
No LayerNorm
83.8 ± 0.6
71.1 ± 0.5
79.8 ± 0.2
No Diffusion
83.0 ± 0.4
71.6 ± 0.4
79.3 ± 0.3
No Feed-Forward⋄
84.9 ± 0.4
72.2 ± 0.3
80.9 ± 0.3
No (LayerNorm + Feed-Forward)
84.3 ± 0.6
72.6 ± 0.4
79.6 ± 0.4
MAGNA
85.4 ± 0.6
73.7 ± 0.5
81.4 ± 0.2
 ±  ±  ± ⋆: based on the implementation in https://github.com/DropEdge/DropEdge; ⋄: replace the feed forward layer with elu used in GAT.
Table 1: Node classification accuracy on Cora, Citeseer, Pubmed. MAGNA achieves state-of-the-art.
Table 1: Node classification accuracy on Cora, Citeseer, Pubmed. MAGNA achieves state-of-the-art.
Welling, 2017]; and (2) a benchmark dataset ogbn-arxiv on 170k nodes and 1.2m edges from the Open Graph Benchmark [Weihua Hu, 2020]. We follow the standard data splits for all datasets. Further information about these datasets is summarized in the Appendix. Baselines We compare against a comprehensive suite of state-of-the-art GNN methods including: GCNs [Kipf and Welling, 2017], Chebyshev filter based GCNs [Defferrard et al., 2016], DualGCN [Zhuang and Ma, 2018], JKNet [Xu et al., 2018], LGCN [Gao et al., 2018], Diffusion-GCN (Diff-GCN) [Klicpera et al., 2019b], APPNP [Klicpera et al., 2019a], Graph U-Nets (g-U-Nets) [Gao and Ji, 2019], and GAT [Veliˇckovi´c et al., 2018]. Experimental Setup For datasets Cora, Citeseer and Pubmed, we use 6 MAGNA blocks with hidden dimension 512 and 8 attention heads. For the large-scale ogbn-arxiv dataset, we use 2 MAGNA blocks with hidden dimension 128 and 8 attention heads. Refer to Appendix for detailed description of all hyper-parameters and evaluation settings. Results MAGNA achieves the best on all datasets (Tables 1 and 2) 7, out-performing mutlihop baselines such as Diffusion GCN, APPNP and JKNet. The baseline performance and their embedding dimensions are from the previous papers. Appendix Table 6 further demonstrates that large 512 dimension embedding only benefits the expressive MAGNA, whereas GAT and Diffusion GCN performance degrades. Ablation study We report (Table 1) the model performance after removing each component of MAGNA (layer normalization, attention diffusion and feed forward layers) from every MAGNA layer. Note that the model is equivalent to GAT without these three components. We observe that diffusion and layer normalization play a crucial role in improving the node classification performance for all datasets. Since MAGNA computes attention diffusion in a recursive manner, layer normalization is crucial in ensuring training stability [Ba et al., 2016]. Meanwhile, comparing to GAT (see the next-to-last row of Table 1), attention diffusion allows multihop attention in every layer to benefit node classification.
Ablation study We report (Table 1) the model performance after removing each component of MAGNA (layer normalization, attention diffusion and feed forward layers) from every MAGNA layer. Note that the model is equivalent to GAT without these three components. We observe that diffusion and layer normalization play a crucial role in improving the node classification performance for all datasets. Since MAGNA computes attention diffusion in a recursive manner, layer normalization is crucial in ensuring training stability [Ba et al., 2016]. Meanwhile, comparing to GAT (see the next-to-last row of Table 1), attention diffusion allows multihop attention in every layer to benefit node classification.
7We also compared to GAT and Diffusion-GCN (with LayerNorm and feed-forward Layer) over random splits in Appendix.
7We also compared to GAT and Diffusion-GCN (with LayerNorm and feed-forward Layer) over random splits in Appendix.
# 4.2 Task 2: Knowledge Graph Completion
Datasets We evaluate MAGNA on standard benchmark knowledge graphs: WN18RR [Dettmers et al., 2018] and FB15K-237 [Toutanova and Chen, 2015]. See the statistics of these KGs in Appendix. Baselines We compare MAGNA with state-of-the-art baselines, including (1) translational distance based models: TransE [Bordes et al., 2013] and its latest extension RotatE [Sun et al., 2019], OTE [Tang et al., 2020] and ROTH [Chami et al., 2020]; (2) semantic matching based models: ComplEx [Trouillon et al., 2016], QuatE [Zhang et al., 2019], CoKE [Wang et al., 2019b], ConvE [Dettmers et al., 2018], DistMult [Yang et al., 2015], TuckER [Balazevic et al., 2019] and AutoSF [Zhang et al., 2020b]; (3) GNN-based models: R-GCN [Schlichtkrull et al., 2018], SACN [Shang et al., 2019] and A2N [Bansal et al., 2019]. Experimental Setup We use the multi-layer MAGNA as encoder for both FB15k-237 and WN18RR. We randomly initialize the entity embedding and relation embedding as the input of the encoders, and set the dimensionality of the initialized entity/relation vector as 100 used in DistMult [Yang et al., 2015]. We select other MAGNA model hype-parameters, including number of layers, hidden dimension, head number, top-k, learning rate, hop number, teleport probability α and dropout ratios (see the settings of these parameter in Appendix), by a random search during the training. Training procedure We use the standard training procedure used in previous KG embedding models [Balazevic et al., 2019; Dettmers et al., 2018] (Appendix for details). We follow an encoder-decoder framework: The encoder applies the proposed MAGNA model to compute the entity embeddings. The decoder makes link prediction given the embeddings. To show the power of MAGNA, we employ a simple decoder DistMult [Yang et al., 2015]. Evaluation We use the standard split for the benchmarks, and the standard testing procedure of predicting tail (head) entity given the head (tail) entity and relation type. We exactly follow the evaluation used by all previous works, namely the Mean Reciprocal Rank (MRR), Mean Rank (MR), and hit rate at K (H@K). See Appendix for a detailed description of this standard setup. Results MAGNA achieves new state-of-the-art in knowledge graph completion on all four metrics (Table 3). MAGNA compares favourably to both the most recent shallow embedding methods (QuatE), and deep embedding methods (SACN). Note that with the same decoder (DistMult), MAGNA using its own embeddings achieves drastic improvements over using the corresponding DistMult embeddings.
# 4.3 MAGNA Model Analysis
Here we present (1) spectral analysis results, (2) robustness to hyper-parameter changes, and (3) attention distribution analysis to show the strengths of MAGNA. Spectral Analysis: Why MAGNA works for node classification? We compute the eigenvalues of the graph Laplacian
<div style="text-align: center;">Table 2: Node classification accuracy on the OGB Arxiv dataset.</div>
Models
WN18RR
FB15k-237
MR
MRR
H@1
H@3
H@10
MR
MRR
H@1
H@3
H@10
TransE [Bordes et al., 2013]
3384
.226
-
-
.501
357
.294
-
-
.465
RotatE [Sun et al., 2019]
3340
.476
.428
.492
.571
177
.338
.241
.375
.533
OTE [Tang et al., 2020]
-
.491
.442
.511
.583
-
.361
.267
.396
.550
ROTH [Chami et al., 2020]
-
.496
.449
.514
.586
-
.344
.246
.380
.535
ComplEx [Trouillon et al., 2016]
5261
.44
.41
.46
.51
339
.247
.158
.275
.428
QuatE [Zhang et al., 2019]
2314
.488
.438
.508
.582
-
.366
.271
.401
.556
CoKE [Wang et al., 2019b]
-
.475
.437
.490
.552
-
.361
.269
.398
.547
ConvE [Dettmers et al., 2018]
4187
.43
.40
.44
.52
244
.325
.237
.356
.501
DistMult [Yang et al., 2015]
5110
.43
.39
.44
.49
254
.241
.155
.263
.419
TuckER [Balazevic et al., 2019]
-
.470
.443
.482
.526
-
.358
.266
.392
.544
AutoSF [Zhang et al., 2020b]
-
.490
.451
-
.567
-
.360
.267
-
.552
R-GCN [Schlichtkrull et al., 2018]
-
-
-
-
-
-
.249
.151
.264
.417
SACN [Shang et al., 2019]
-
.47
.43
.48
.54
-
.35
.26
.39
.54
A2N [Bansal et al., 2019]
-
.45
.42
.46
.51
-
.317
.232
.348
.486
MAGNA + DistMult
2545
.502
.459
.519
.589
138
.369
.275
.409
.563
Table 3: KG Completion on WN18RR and FB15k-237. MAGNA achieves state of the art.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/84d8/84d8f46d-cec8-47ec-8b7b-d1d3556bb49d.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Analysis of MAGNA on Cora. (a) Influence of MAGNA on Laplacian eigenvalues. (b) Effect of depth on performance. (c) Effec of hop number K on performance. (d) Effect of teleport probability α.</div>
of the attention matrix A, ˆλg i , and compare to that of the diffused matrix A, λg i . Figure 3 (a) shows the ratio ˆλg i /λg i on the Cora dataset. Low eigenvalues corresponding to large-scale structure in the graph are amplified (up to a factor of 8), while high eigenvalues corresponding to eigenvectors with noisy information are suppressed [Klicpera et al., 2019b].
of the attention matrix A, ˆλg i , and compare to that of the diffused matrix A, λg i . Figure 3 (a) shows the ratio ˆλg i /λg i on the Cora dataset. Low eigenvalues corresponding to large-scale structure in the graph are amplified (up to a factor of 8), while high eigenvalues corresponding to eigenvectors with noisy information are suppressed [Klicpera et al., 2019b]. MAGNA Model Depth Here we conduct experiments by varying the number of GCN, Diffusion-GCN (PPR based) GAT and our MAGNA layers to be 3, 6, 12, 18 and 24 for node classification on Cora. Results in Fig. 3 (b) show that deep GCN, Diffusion-GCN and GAT (even with residual connection) suffer from degrading performance, due to the over-smoothing problem [Li et al., 2018; Wang et al., 2019a]. In contrast, the MAGNA model achieves consistent best results even with 18 layers, making deep MAGNA model robust and expressive. Notice that GAT with 18 layers cannot out-perform MAGNA with 3 layers and K=6 hops, although they have the same receptive field. Effect of K and α Figs. 3 (c) and (d) report the effect of hop number K and teleport probability α on model performance. We observe significant increase in performance when
Effect of K and α Figs. 3 (c) and (d) report the effect of hop number K and teleport probability α on model performance. We observe significant increase in performance when
considering multi-hop neighbors information (K > 1). However, increasing the hop number K has a diminishing returns, for K ≥6. Moreover, we find that the optimal K is correlated with the largest node average shortest path distance (e.g., 5.27 for Cora). This provides a guideline for choosing the best K. We also observe that the accuracy drops significantly for larger α > 0.25. This is because small α increases the lowpass effect (Fig. 3 (a)). However, α being too small causes the model to only focus on the most large-scale graph structure and have lower performance.
Attention Distribution Last we also analyze the learned attention scores of GAT and MAGNA. We first define a discrepancy metric over the attention matrix A for node vi as ∆i = ∥A[i,:]−Ui∥ degree(vi) [Shanthamallu et al., 2020], where Ui is the uniform distribution score for the node vi. ∆i gives a measure of how much the learnt attention deviates from an uninformative uniform distribution. Large ∆i indicates more meaningful attention scores. Fig. 4 shows the distribution of the discrepancy metric of the attention matrix of the 1st head w.r.t. the first layer of MAGNA and GAT. Observe that at-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b303/b30392f9-7745-444f-b683-bc4af6d1d361.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Attention weight distribution on Cora.</div>
tention scores learned in MAGNA have much larger discrepancy. This shows that MAGNA is more powerful than GAT in distinguishing important nodes and assigns attention scores accordingly.
# 5 Related Work
Our proposed MAGNA belongs to the family of Graph Neural Network (GNN) models [Battaglia et al., 2018; Wu et al., 2020; Kipf and Welling, 2017; Hamilton et al., 2017], while taking advantage of graph attention and diffusion techniques. Graph Attention Neural Networks (GATs) generalize attention operation to graph data. GATs allow for assigning different importance to nodes of the same neighborhood at the feature aggregation step [Veliˇckovi´c et al., 2018]. Based on such framework, different attention-based GNNs have been proposed, including GaAN [Zhang et al., 2018], AGNN [Thekumparampil et al., 2018], GeniePath [Liu et al., 2019]. However, these models only consider direct neighbors for each layer of feature aggregation, and suffer from oversmoothing when they go deep [Wang et al., 2019a]. Diffusion based Graph Neural Network Recently Graph Diffusion Convolution [Klicpera et al., 2019b; Klicpera et al., 2019a] proposes to aggregate information from a larger (multi-hop) neighborhood at each layer, by sparsifying a generalized form of graph diffusion. This idea was also explored in [Liao et al., 2019; Luan et al., 2019; Xhonneux et al., 2020; Klicpera et al., 2019a] for multi-scale Graph Convolutional Networks. However, these methods do not incorporate attention mechanism which is crucial to model performance, and do not make use of edge embeddings (e.g., Knowledge graph) [Klicpera et al., 2019b]. Our approach defines a novel multi-hop context-dependent self-attention GNN which resolves the over-smoothing issue of GAT architectures [Wang et al., 2019a]. [Isufi et al., 2020; Cucurull et al., 2018; Feng et al., 2019] also extends attention mechanism for multihop information aggregation, but they require different set of parameters to compute the attention to neighbors of different hops, making these approaches much more expensive compared to MAGNA, and were not extended to the knowledge graph settings.
# 6 Conclusion
We proposed Multi-hop Attention Graph Neural Network (MAGNA), which brings together benefits of graph atten-
tion and diffusion techniques in a single layer through attention diffusion, layer normalization and deep aggregation. MAGNA enables context-dependent attention between any pair of nodes in the graph in a single layer, enhances largescale structural information, and learns more informative attention distribution. MAGNA improves over all state-of-theart methods on the standard tasks of node classification and knowledge graph completion.
# References
[Ba et al., 2016] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. [Bahdanau et al., 2015] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In ICLR, 2015. [Balazevic et al., 2019] Ivana Balazevic, Carl Allen, and Timothy Hospedales. Tucker: Tensor factorization for knowledge graph completion. In EMNLP, 2019. [Bansal et al., 2019] Trapit Bansal, Da-Cheng Juan, Sujith Ravi, and Andrew McCallum. A2n: Attending to neighbors for knowledge graph inference. In ACL, 2019. [Battaglia et al., 2018] Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018. [Bergstra and Bengio, 2012] James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. JMLR, pages 281–305, 2012. [Bordes et al., 2013] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In NeurIPS, 2013. [Chami et al., 2020] Ines Chami, Adva Wolf, Da-Cheng Juan, Frederic Sala, Sujith Ravi, and Christopher R´e. Low-dimensional hyperbolic knowledge graph embeddings. In ACL, 2020. [Cucurull et al., 2018] Guillem Cucurull, Konrad Wagstyl, Arantxa Casanova, Petar Veliˇckovi´c, Estrid Jakobsen, Michal Drozdzal, Adriana Romero, Alan Evans, and Yoshua Bengio. Convolutional neural networks for mesh-based parcellation of the cerebral cortex. In MIDL, 2018. [Defferrard et al., 2016] Micha¨el Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In NeurIPS, 2016. [Dettmers et al., 2018] Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. Convolutional 2d knowledge graph embeddings. In AAAI, 2018. [Devlin et al., 2019] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019. [Feng et al., 2019] Chenyuan Feng, Zuozhu Liu, Shaowei Lin, and Tony QS Quek. Attention-based graph convolutional network for recommendation system. In ICASSP, 2019. [Gao and Ji, 2019] Hongyang Gao and Shuiwang Ji. Graph u-nets. In ICML, 2019. [Gao et al., 2018] Hongyang Gao, Zhengyang Wang, and Shuiwang Ji. Large-scale learnable graph convolutional networks. In KDD, 2018. [Hamilton et al., 2017] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In NeurIPS, 2017. [Isufi et al., 2020] Elvin Isufi, Fernando Gama, and Alejandro Ribeiro. Edgenets: Edge varying graph neural networks. arXiv preprint arXiv:2001.07620, 2020.
[Kipf and Welling, 2017] Thomas N Kipf and Max Welling. Semisupervised classification with graph convolutional networks. In ICLR, 2017. [Klicpera et al., 2019a] Johannes Klicpera, Aleksandar Bojchevski, and Stephan G¨unnemann. Predict then propagate: Graph neural networks meet personalized pagerank. In ICLR, 2019. [Klicpera et al., 2019b] Johannes Klicpera, Stefan Weißenberger, and Stephan G¨unnemann. Diffusion improves graph learning. In NeurIPS, 2019. [Li et al., 2018] Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semisupervised learning. In AAAI, 2018. [Liao et al., 2019] Renjie Liao, Zhizhen Zhao, Raquel Urtasun, and Richard S. Zemel. Lanczosnet: Multi-scale deep graph convolutional networks. In ICLR, 2019. [Liu et al., 2019] Ziqi Liu, Chaochao Chen, Longfei Li, Jun Zhou, Xiaolong Li, Le Song, and Yuan Qi. Geniepath: Graph neural networks with adaptive receptive paths. In AAAI, 2019. [Liu et al., 2020] Meng Liu, Hongyang Gao, and Shuiwang Ji. Towards deeper graph neural networks. In KDD, 2020. [Lofgren, 2015] Peter Lofgren. Efficient Algorithms for Personalized PageRank. PhD thesis, Stanford University, 2015. [Luan et al., 2019] Sitao Luan, Mingde Zhao, Xiao-Wen Chang, and Doina Precup. Break the ceiling: Stronger multi-scale deep graph convolutional networks. In NeurIPS, 2019. [Mohar et al., 1991] Bojan Mohar, Y Alavi, G Chartrand, and OR Oellermann. The laplacian spectrum of graphs. Graph theory, combinatorics, and applications, pages 871–898, 1991. [Ng et al., 2002] Andrew Y Ng, Michael I Jordan, and Yair Weiss. On spectral clustering: Analysis and an algorithm. In NeurIPS, 2002. [Nguyen et al., 2020] Xuan-Phi Nguyen, Shafiq Joty, Steven CH Hoi, and Richard Socher. Tree-structured attention with hierarchical accumulation. In ICLR, 2020. [Oono and Suzuki, 2020] Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for node classification. In ICLR, 2020. [Sandryhaila and Moura, 2013a] Aliaksei Sandryhaila and Jos´e MF Moura. Discrete signal processing on graphs. TSP, pages 1644– 1656, 2013. [Sandryhaila and Moura, 2013b] Aliaksei Sandryhaila and Jos´e MF Moura. Discrete signal processing on graphs: Graph fourier transform. In ICASSP, 2013. [Schlichtkrull et al., 2018] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan Titov, and Max Welling. Modeling relational data with graph convolutional networks. In ESWC, 2018. [Sen et al., 2008] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Galligher, and Tina Eliassi-Rad. Collective classification in network data. AI magazine, pages 93–106, 2008. [Shang et al., 2019] Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, and Bowen Zhou. End-to-end structure-aware convolutional networks for knowledge base completion. In AAAI, 2019. [Shanthamallu et al., 2020] Uday Shankar Shanthamallu, Jayaraman J Thiagarajan, and Andreas Spanias. A regularized attention mechanism for graph attention networks. In ICASSP, 2020. [Sun et al., 2019] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. Rotate: Knowledge graph embedding by relational rotation in complex space. In ICLR, 2019. [Tang et al., 2020] Yun Tang, Jing Huang, Guangtao Wang, Xiaodong He, and Bowen Zhou. Orthogonal relation transforms with graph context modeling for knowledge graph embedding. In ACL, 2020. [Thekumparampil et al., 2018] Kiran K Thekumparampil, Chong
neural network for semi-supervised learning. arXiv preprint arXiv:1803.03735, 2018. [Toutanova and Chen, 2015] Kristina Toutanova and Danqi Chen. Observed versus latent features for knowledge base and text inference. In CVSC-WS, 2015. [Tremblay et al., 2018] Nicolas Tremblay, Paulo Gonc¸alves, and Pierre Borgnat. Design of graph filters and filterbanks. In Cooperative and Graph Signal Processing, pages 299–324. Elsevier, 2018. [Trouillon et al., 2016] Th´eo Trouillon, Johannes Welbl, Sebastian Riedel, ´Eric Gaussier, and Guillaume Bouchard. Complex embeddings for simple link prediction. In ICML, 2016. [Vaswani et al., 2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. [Veliˇckovi´c et al., 2018] Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. In ICLR, 2018. [Wang et al., 2019a] Guangtao Wang, Rex Ying, Jing Huang, and Jure Leskovec. Improving graph attention networks with large margin-based constraints. In NeurIPS-WS, 2019. [Wang et al., 2019b] Quan Wang, Pingping Huang, Haifeng Wang, Songtai Dai, Wenbin Jiang, et al. Coke: Contextualized knowledge graph embedding. arXiv preprint arXiv:1911.02168, 2019. [Wang et al., 2019c] Yaushian Wang, Hung-Yi Lee, and Yun-Nung Chen. Tree transformer: Integrating tree structures into selfattention. In EMNLP, 2019. [Weihua Hu, 2020] etc. Weihua Hu, Matthias Fey. Open graph benchmark: Datasets for machine learning on graphs. In NeurIPS, 2020. [Wu et al., 2020] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and Philip S Yu. A comprehensive survey on graph neural networks. IEEE Trans Neural Netw Learn Syst, 2020. [Xhonneux et al., 2020] Louis-Pascal A. C. Xhonneux, Meng Qu, and Jian Tang. Continuous graph neural networks. In ICML, 2020. [Xiong et al., 2020] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On layer normalization in the transformer architecture. In ICML, 2020. [Xu et al., 2018] Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie Jegelka. Representation learning on graphs with jumping knowledge networks. In ICML, 2018. [Yang et al., 2015] Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and relations for learning and inference in knowledge bases. In ICLR, 2015. [Zhang et al., 2018] Jiani Zhang, Xingjian Shi, Junyuan Xie, Hao Ma, Irwin King, and Dit-Yan Yeung. Gaan: Gated attention networks for learning on large and spatiotemporal graphs. In UAI, 2018. [Zhang et al., 2019] Shuai Zhang, Yi Tay, Lina Yao, and Qi Liu. Quaternion knowledge graph embedding. In NeurIPS, 2019. [Zhang et al., 2020a] Jiawei Zhang, Haopeng Zhang, Li Sun, and Congying Xia. Graph-bert: Only attention is needed for learning graph representations. arXiv preprint arXiv:2001.05140, 2020. [Zhang et al., 2020b] Yongqi Zhang, Quanming Yao, Wenyuan Dai, and Lei Chen. Autosf: Searching scoring functions for knowledge graph embedding. In ICDE, 2020. [Zhuang and Ma, 2018] Chenyi Zhuang and Qiang Ma. Dual graph convolutional networks for graph-based semi-supervised classification. In WWW, 2018.
# A Attention Diffusion Approximation Proposition
# A Attention Diffusion Approximation
As mentioned in Section 2.2, we use the following equation and proposition to efficiently approximate the attention diffused feature aggregation AH(l).
(12)
Proof. Let K > 0 be the total number of iterations (i.e., hop number of graph attention diffusion) and we approximate ˆ H(l) by Z(K). After K-th iteration, we can get
(13)
� The term (1 −α)KAK converges to 0 as α ∈(0, 1] and AK i,j ∈(0, 1] when K →∞, and thus limK→∞Z(K) = (�∞ i=0 α(1 −α)iAi)H(l) = AH(l). Complexity analysis. Suppose the graph contains E edges, using the above approximation, there are O(|E|) message communications in total, with a constant factor corresponding to the number of hops K. In practice, we find that choosing the values of K such that 3 ≤K ≤10 results in good model performance.
� The term (1 −α)KAK converges to 0 as α ∈(0, 1] an AK i,j ∈(0, 1] when K →∞, and thus limK→∞Z(K)  (�∞ i=0 α(1 −α)iAi)H(l) = AH(l).

� Complexity analysis. Suppose the graph contains E edges, using the above approximation, there are O(|E|) message communications in total, with a constant factor corresponding to the number of hops K. In practice, we find that choosing the values of K such that 3 ≤K ≤10 results in good model performance.
# B Connection to Transformer
Given a sequence of tokens, the Transformer architecture makes uses of multi-head attention between all pairs of tokens, and can be viewed as performing message-passing on a fully connected graph between all tokens. A na¨ıve application of Transformer on graphs would require computation of all pairwise attention values. Such approach, however, would not make effective use of the graph structure, and could not scale to large graphs. In contrast, Graph Attention Network [Veliˇckovi´c et al., 2018] leverages the graph structure and only computes attention values and perform message passing between direct neighbors. However, it has a limited receptive field (restricted to one-hop neighborhood) and the attention score (Figure 1) that is independent of the multi-hop context for prediction. Transformer consists of self-attention layer followed by feed-forward layer. We can organize the self-attention layer in transformer as the following:
Attention(Q, K, V ) = softmax(QKT √ d ) � �� � Attention matrix V
Attention(Q, K, V ) = softmax(QK √ d ) V
(14)
� �� � where Q = K = V . The softmax part can be demonstrated as an attention matrix computed by scaled dot-product attention over a complete graph8 with self-loop. Computation of attention over complete graph is expensive, Transformers are usually limited by a fixed-length context (e.g., 512 in
8All nodes are connected with each other.
BERT [Devlin et al., 2019]) in the setting of language modeling, and thus cannot handle large graphs. Therefore direct application of the transformer model cannot capture the graph structure in a scalable way. In the past, graph structure is usually encoded implicitly by special position embeddings [Zhang et al., 2020a] or welldesigned attention computation [Wang et al., 2019c; Nguyen et al., 2020]. However, none of the methods can compute attention between any pair of nodes at each layer. In contrast, essentially MAGNA places a prior over the attention values via Personalized PageRank, allowing it to compute the attention between any pair of two nodes via attention diffusion, without any impact on its scalability. In particular, MAGNA can handle large graphs as they are usually quite sparse and the graph diameter is usually quite smaller than graph size in practice, resulting in very efficient attention diffusion computation.
# C Spectral Analysis Background and Proof for Proposition 2
Graph Fourier Transform. Suppose AN×N represents the attention matrix of graph G with Ai,j ≥0, and �N j=1 Ai,j = 1. Let A = V ΛV −1 be the Jordan’s decomposition of graph attention matrix A, where V is the square N × N matrix whose i-th column is the eigenvector vi of A, and Λ is the diagonal matrix whose diagonal elements are the corresponding eigenvalues, i.e., Λi,i = λi. Then, for a given vector x, its Graph Fourier Transform [Sandryhaila and Moura, 2013b] is defined as
(15)
where V −1 is denoted as graph Fourier transform matrix. The Inverse Graph Fourier Transform is defined as x = V ˆx, which reconstructs the signal from its spectrum. Based on the graph Fourier transform, we can define a graph convolution operation on G as x ⊗G y = V (V −1x ⊙V −1y), where ⊙ denotes the element-wise product. Graph Attention Diffusion Acts as A Polynomial Graph Filter. A graph filter [Tremblay et al., 2018; Sandryhaila and Moura, 2013b; Sandryhaila and Moura, 2013a] h acts on x as h(A)x = V h(Λ)V −1, where h(Λ) = diag(h(λ1) · · · h(λN)). A common choice for h in the literature is a polynomial filter of order M, since it is linear and shift invariant [Sandryhaila and Moura, 2013b; Sandryhaila and Moura, 2013a].
(16)
Comparing to the graph attention diffusion A = �∞ i=0 α(1− α)iAi, if we set βℓ= α(1 −α)ℓ, we can view graph attention diffusion as a polynomial filter. Spectral Analysis. The eigenvectors of the power matrix A2 are same as A, since A2 = (V ΛV −1)(V ΛV −1) = V Λ(V −1V )ΛV −1 = V Λ2V −1. By that analogy, we can get that An = V ΛnV −1. Therefore, the summation of the power series of A has the same eigenvectors as A. Therefore by properties of eigenvectors and Equation 3, we obtain:
Proposition 5. The set of eigenvectors for A and A are the same. Lemma 1. Let λi and ˆλi be the i-th eigenvalues of A and A, respectively. Then, we have
Lemma 1. Let λi and ˆλi be the i-th eigenvalues of A and A, respectively. Then, we have
(17)
Proof. The symmetric normalized graph Laplacian of G is Lsym = I −D−1 2 AD−1 2 , where D = diag([d1, d2, · · · , dN]), and di = �Ai,j j=1 . As A is the attention matrix of graph G, di = 1 and thus D = I. Therefore, Lsym = I −A. Let λi be the eigenvalues of A, the eigenvalues of the symmetric normalized Laplacian of G Lsym is ¯λi = 1 −λi. Meanwhile, for every eigenvalue ¯λi of the normalized graph Laplacian Lsym, we have 0 ≤¯λi ≤2 [Mohar et al., 1991], and thus −1 ≤λi ≤1. As 0 < α < 1 and thus |(1 −α)λi| ≤(1 −α) < 1. Therefore, ((1 −α)λi)K →0 when K →∞, and ˆλi = limK→∞ �K ℓ=0 α(1 −α)ℓλℓ i = limK→∞ α(1−((1−α)λi)K) 1−(1−α)λi = α 1−(1−α)λi .

Section 3 further defines the eigenvalues of the Laplacian matrices, ˆλg i and λg i respectively. They satisfy: ˆλg i = 1 −ˆλi and λg i = 1 −λi, and λg i ∈[0, 2] (proved by [Ng et al., 2002]).
# D Graph Learning Tasks
Node classification and knowledge graph link prediction are two representative and common tasks in graph learning. We first define the task of node classification:
Definition 1. Node classification Suppose that X ∈RNn×d represents the node input features, where each row xi = Xi: is a d-dimensional vector of attribute values of node vi ∈V (1 ≤i ≤N). Vl ⊂V consists of a set of labeled nodes, and the labels are from T , node classification is to learn the map function f : (X, G) →T , which predicts the labels of the remaining un-labeled nodes V/Vl. Knowledge graph (KG) is a heterogeneous graph describing entities and their typed relations to each other. KG is defined by a set of entities (nodes) vi ∈V, and a set of relations (edges) e = (vi, rk, vj) connecting nodes vi and vj via relation rk. We then define the task of knowledge graph completion: Definition 2. KG completion refers to the task of predicting an entity that has a specific relation with another given entity [Bordes et al., 2013], i.e., predicting head h given a pair of relation and entity (r, t) or predicting tail t given a pair of head and relation (h, r).
# E Dataset Statistics
Node classification. We show the dataset statistics of the node classification benchmark datasets in Table 4. Knowledge Graph Link Prediction. We show the dataset statistics of the knowledge graph benchmarks in Table 5.
# F Knowledge Graph Training and Evaluation
Training. The standard knowledge graph completion task training procedure is as follows. We add the reverse-direction triple (t, r−1, h) for each triple (h, r, t) to construct an undirected knowledge graph G. Following the training procedure introduced in [Balazevic et al., 2019; Dettmers et al., 2018], we use 1-N scoring, i.e. we simultaneously score entityrelation pairs (h, r) and (t, r−1) with all entities, respectively. We explore KL diversity loss with label smoothing as the optimization function. Inference time procedure. For each test triplet (h, r, t), the head h is removed and replaced by each of the entities appearing in KG. Afterward, we remove from the corrupted triplets all the ones that appear either in the training, validation or test set. Finally, we score these corrupted triplets by the link prediction models and then sorted by descending order; the rank of (h, r, t) is finally scored. This whole procedure is repeated while removing the tail t instead of h. And averaged metrics are reported. We report mean reciprocal rank (MRR), mean rank (MR) and the proportion of correct triplets in the top K ranks (Hits@K) for K = 1, 3 and 10. Lower values of MR and larger values of MRR and Hits@K mean better performance.
# G Results
Comparison to Diffusion GCN. Diffusion-GCN [Klicpera et al., 2019b] is also based on Personal Page-Rank (PPR) propagation. Specifically, it performs propagation over the adjacent matrix. Comparing to MAGNA, there is no LayerNorm and Feed Forward layers in standard Diffusion GCN. To clarify how much of the gain in performance depends on the page-rank-based propagation compared to the attention propagation in MAGNA, we add the two modules: LayerNorm and Feed Forward layer, into Diffusion-GCN, and conduct experiments over Cora, Citeseer and Pubmed, respectively. Table 6 shows the comparison results. From this table, we observe that adding layer normalization and feed forward layer gets the similar GNN structure to MAGNA, but does not benefit for node classification. And this implies that the PPR based attention propagation in MAGNA is more effective than PPR propagation over adjacent matrix in Diffusion GCN.
# Comparison over Random Splitting
split nodes in each graph for training, validation and testing following the same ratio in the standard data splits. We conduct experiments with MAGNA, Diffusion GCN (PPR) and GAT over 10 different random splits over Cora, Citeseer and Pubmed, respectively. For each algorithm, we apply random search [Bergstra and Bengio, 2012] for hype-parameter tuning. We report the average classification accuracy with standard deviation over 10 different splits in Table 7. From this table, we observe that MAGNA gets the best average accuracy over all data sets. However, it is noted that the standard deviation is quite large comparing to the that from the standard split in Table 6. This means that the performance is quite sensitive to the graph split. This observation is consistent wit the conclusion in [Weihua Hu, 2020]: random splitting is often problematic in graph learning as they are impractical in
Name
Nodes
Edges
Classes
Features
Train/Dev/Test
Cora
2,708
5,429
7
1,433
140/500/1,000
Citeseer
3,327
4,732
6
3,703
120/500/1,000
Pubmed
19,717
88,651
3
500
60/500/1,000
ogbn-arxiv†
169,343
1,166,243
40
128
90,941/29,799/48,603
 
 The data is available at https://ogb.stanford.edu/docs/nodeprop/.
Dataset
#Entities
#Relations
#Train
#Dev
#Test
#Avg. Degree
WN18RR
40,943
11
86,835
3034
3134
2.19
FB15k-237
14,541
237
272,115
17,535
20,466
18.17
el
Cora
Citeseer
Pubmed
usion-GCN (PPR) [Klicpera et al., 2019b]
83.6 ± 0.2
73.4 ± 0.3
78.7 ± 0.4⋆
usion-GCN (PPR) + LayerNorm + FF⋄
83.4 ± 0.4
72.3 ± 0.4
78.1± 0.5
 (hidden dimension = 512)∗
83.1 ± 0.5
71.3 ± 0.4
78.3± 0.4
GNA
85.4 ± 0.6
73.7 ± 0.5
81.4 ± 0.2
FF: Feed Forward (FF) layer, and our implementation is based on Diffusion GCN in
torch geometric (https://github.com/rusty1s/pytorch geometric).
The best number derived from Diffusion GCN with “PPR”. This is different from the
mber in Table 1 of main body, which comes from Diffusion GCN with “Heat”.
The results of GAT with the same hidden dimension in MAGNA.
e 6: Comparison of Diffusion GCN (PPR), GAT (with hidden
ension = 512) and MAGNA for Node classification accuracy on
a, Citeseer, Pubmed.
Hyper-parameters
Search Space
Type
Hidden Dimension
512
Fixed⊢
Head Number
8
Fixed
Layer Number
6
Fixed
Learning rate
[5 × 10−5, 10−3]
Range⋆
Hop Number
[2, 3, · · · , 10]
Choice⋄
Teleport probability α
[0.05, 0.6]
Range
Dropout (attention, feature)
[0.1, 0.6]
Range
Weight Decay
[10−6, 10−5]
Range
Optimizer
Adam
Fixed
<div style="text-align: center;"> ±  ±  ± ⋄: FF: Feed Forward (FF) layer, and our implementation is based on Diffusion GCN in pytorch geometric (https://github.com/rusty1s/pytorch geometric). ⋆: The best number derived from Diffusion GCN with “PPR”. This is different from the number in Table 1 of main body, which comes from Diffusion GCN with “Heat”. ∗: The results of GAT with the same hidden dimension in MAGNA.</div>
Table 6: Comparison of Diffusion GCN (PPR), GAT (with hidden dimension = 512) and MAGNA for Node classification accuracy on Cora, Citeseer, Pubmed.
Model
Cora
Citeseer
Pubmed
Diffusion GCN (PPR) [Klicpera et al., 2019b]
83.6 ± 1.8
71.7 ± 1.1
79.6 ± 2.5
Diffusion GCN (PPR) + (LayerNorm & FF)
83.3 ± 1.9
69.7 ± 1.2
79.7 ± 2.3
GAT [Veliˇckovi´c et al., 2018]
82.2 ± 1.4
71.0 ± 1.2
78.3 ± 3.0
GAT + (LayerNorm & FF)
82.5 ± 2.2
69.5 ± 1.1
78.2 ± 2.0
MAGNA
83.6 ± 1.2
72.1 ± 1.2
80.3 ± 2.4
Table 7: Results of MAGNA on Citeseer, Cora and Pubmed with random splitting. The baselines are Diffusion GCN (PPR) and GAT.
Table 7: Results of MAGNA on Citeseer, Cora and Pubmed with random splitting. The baselines are Diffusion GCN (PPR) and GAT. real scenarios, and can lead to large performance variation. Therefore, in our main results, we use the standard split and also record standard deviation to demonstrate significance of the results.
real scenarios, and can lead to large performance variation. Therefore, in our main results, we use the standard split and also record standard deviation to demonstrate significance of the results.
# H Hyper-parameter Settings
Hyper-parameter settings for node classification. The best models are selected according to the classification accuracy on the validation set by early stopping with window size 200. For each data set, the hyper-parameters are determined by a random search [Bergstra and Bengio, 2012], including learning rate, hop number, teleport probability α and dropout ratios. The hyper-parameter search space is show in Tables 8 (for Cora, Citeseer and Pubmed) and 9 (for ogbn-arxiv). Hyper-parameter settings for link prediction on KG. For each KG, the hyper-parameters are determined by a random search [Bergstra and Bengio, 2012], including number of layers, learning rate, hidden dimension, batch-size, head number, hop number, teleport probability α and dropout ratios. The hyper-parameter search space is show in Table 10.
Hyper-parameters
Search Space
Type
Hidden Dimension
512
Fixed⊢
Head Number
8
Fixed
Layer Number
6
Fixed
Learning rate
[5 × 10−5, 10−3]
Range⋆
Hop Number
[2, 3, · · · , 10]
Choice⋄
Teleport probability α
[0.05, 0.6]
Range
Dropout (attention, feature)
[0.1, 0.6]
Range
Weight Decay
[10−6, 10−5]
Range
Optimizer
Adam
Fixed
⊢Fixed: a constant value;: Range: a value range with lower bound
⊢Fixed: a constant value; ⋆: Range: a value range with lower bound and higher bound; ⋄: Choice: a set of values.
⊢Fixed: a constant value; ⋆: Range: a value range with lower bound and higher bound; ⋄: Choice: a set of values.
Table 8: Hyper-parameter search space used for node classification on Cora, Citeseer and Pubmed
Hyper-parameters
Search Space
Type
Hidden Dimension
128
Fixed
Head Number
8
Fixed
Layer Number
2
Fixed
Learning rate
[0.001, 0.01]
Range
Hop Number
[3, 4, 5, 6]
Choice
Teleport probability α
[0.05, 0.6]
Range
Dropout (attention, feature)
[0.1, 0.6]
Range
Weight Decay
[10−5, 10−4]
Range
Optimizer
Adam
Fixed
Table 9: Hyper-parameter search space for node classification on ogbn-arxiv
Hyper-parameters
Search Space
Type
Initial Entity/Relation Dimension
100
Fixed
Number of layers
[2, 3]
Choice
Learning rate
[10−4, 5 × 10−3]
Range
Hidden Dimension
[256, 512, 768]
Choice
Batch size
[1024, 2048, 3072]
Choice
Head Number
[4, 8]
Choice
Hop Number
[2, 3, 4, 5, 6]
Choice
Teleport probability α
[0.05, 0.6]
Range
Dropout (attention, feature)
[0.1, 0.6]
Range
Weight Decay
[10−10, 10−8]
Range
Optimizer
Adam
Fixed
Table 10: Hyper-parameter search space for link prediction on KG
