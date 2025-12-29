# PSP: Pre-Training and Structure Prompt Tuning for Graph Neural Networks
Qingqing Ge1, Zeyuan Zhao1, Yiding Liu2, Anfeng Cheng2, Xiang Li1 (�), Shuaiqiang Wang2, and Dawei Yin2
1 School of Data Science and Engineering, East China Normal University, China {qingqingge, zeyuanzhao}@stu.ecnu.edu.cn xiangli@dase.ecnu.edu.cn 2 Baidu Inc., China {liuyiding.tanh, anfcheng2, shqiang.wang}@gmail.com yindaei@acm.org
Abstract. Graph Neural Networks (GNNs) are powerful in learning semantics of graph data. Recently, a new paradigm “pre-train & prompt” has shown promising results in adapting GNNs to various tasks with less supervised data. The success of such paradigm can be attributed to the more consistent objectives of pre-training and task-oriented prompt tuning, where the pre-trained knowledge can be effectively transferred to downstream tasks. Most existing methods are based on the class prototype vector framework. However, in the few-shot scenarios, given few labeled data, class prototype vectors are difficult to be accurately constructed or learned. Meanwhile, the structure information of graph is usually exploited during pre-training for learning node representations, while neglected in the prompt tuning stage for learning more accurate prototype vectors. In addition, they generally ignore the impact of heterophilous neighborhoods on node representation and are not suitable for heterophilous graphs. To bridge these gaps, we propose a novel pre-training and structure prompt tuning framework for GNNs, namely PSP, which consistently exploits structure information in both pre-training and prompt tuning stages. In particular, PSP 1) employs a dual-view contrastive learning to align the latent semantic spaces of node attributes and graph structure, and 2) incorporates structure information in prompted graph to construct more accurate prototype vectors and elicit more pre-trained knowledge in prompt tuning. We conduct extensive experiments on node classification and graph classification tasks to evaluate the effectiveness of PSP. We show that PSP can lead to superior performance in few-shot scenarios on both homophilous and heterophilous graphs. The implemented code is available at https://github.com/gqq1210/PSP.
Keywords: Graph Neural Networks · Pre-training · Prompt · Few-sho
# 1 Introduction
Graph Neural Networks (GNNs) have been widely applied in a variety of fields, such as social network analysis [5], financial risk control [26], and recommender
systems [27], where both structural and attribute information are learned via message passing on the graphs [11]. Recently, extensive efforts [6,12] have been made to design graph pre-training methods, which are further fine-tuned for various downstream tasks. Nevertheless, inconsistent objectives of pre-training and fine-tuning often leads to catastrophic forgetting during downstream adaptation [33], especially when the downstream supervised data is too scarce to be easily over-fitted. To bridge this gap, many prompt tuning methods for GNNs [20,14,33,21,4,30,1,31,22] h also been proposed to achieve remarkable performance in few-shot learning tasks on graphs. In particular, the key insight of these methods is to freeze the pretrained model (i.e., GNN) and introduce extra task-specific parameters, which learns to exploit the pre-trained knowledge for downstream tasks. For example, GPPT [20] and GraphPrompt [14] pre-train a GNN model based on the link prediction task, then they take the class prototype vectors and the readout function as parameters respectively to reformulate the downstream node/graph classification task into the same format as link prediction. Despite the initial success, a clear limitation in these existing models is that graph structure, as the key ingredient in pre-training, is under-explored when constructing class prototype vectors in prompt tuning, which limits their effectiveness in unleashing pre-trained knowledge. In particular, their task-specific parameters (e.g., class prototype vectors or readout functions) are usually learned only with few labeled data. They fail to consider the relationships between the task and the massive unlabeled data, which could also provide rich pre-trained knowledge that is very useful for the task at hand. This is even more important when the labeled data is scarce, e.g., few-shot node classification. As shown in Figure 1 (b), existing methods that directly use the average embeddings of labeled nodes/graphs as the class prototype representations can easily be undermined by noisy/outlier data when the number of labeled nodes is scarce. In contrast, facilitating class representation learning with structural connections between class prototype vectors and unlabeled nodes could help solve this issue (as shown in Figure 1 (d)). Moreover, most existing methods are designed for homophilous graphs, relying excessively on the graph structural information while disregarding the impact of heterophilous neighborhoods on node representations. This adversely affects the model performance on heterophilous graphs. In this paper, we propose a novel Pre-training and Structure Prompt tuning (PSP) framework, which unifies the objectives of pre-training and prompt tuning for GNNs and integrates structural information in both pre-training and prompt tuning stages to construct more accurate prototype vectors. For pretraining, inspired by [13], we separate attribute and structural information, employing dual-view contrastive learning to align the latent semantic spaces of node attributes and graph structure. Specifically, one view is implemented with MLP, which only uses node attributes in the graph. The other view adopts GNN to leverage both node attributes and structural information of the graph. For downstream prompt-tuning, we fix the learned parameters of MLP and GNN
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ed35/ed35d399-2277-4c17-8cee-47874853268e.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) PSP before prompt tuning (d) PSP after prompt tuning</div>
<div style="text-align: center;">(c) PSP before prompt tuning</div>
Fig. 1: The construction of class prototype vectors. The colored areas contain the labeled nodes for training. The circles represent nodes, and the triangles represent class prototype vectors for node classification task. The solid black lines and gray dashed lines denote the original edges in the graph and the new weighted edges, respectively. For each node, the dashed line in red or green denotes the edges with the largest weight to the class prototype vector.
in the pre-training stage, add class prototype vectors as new nodes to the raw graph and introduce structural connections between prototype vectors and original nodes as prompts to learn more accurate prototype vectors (see Figure 1 (c)). Note that weights associated with these connections are parameters to be learned. In the training phase, we use representations of labeled nodes/graphs calculated by MLP as anchors, and representations of prototype vectors obtained through GNN as positive/negative samples. Specifically, the prototype vector in the same class as the anchor is considered as positive sample, while prototype vectors in other classes serve as negative samples. Then, contrastive learning between nodes/graphs (MLP-based view) and prototype vectors (GNN-based view) is performed to learn prompt parameters. As a result, we unify the objectives of pre-training and prompt tuning. After prompt tuning, the nodes and their corresponding class prototypes are learned to have higher weights on the edges, as shown in Figure 1(d). We also experimentally show the results in Figure 5 of Section 5.5. Based on the learned weights, for each prototype vector, GNN formulates its embedding by weighted-aggregating information from its neighboring nodes, i.e., all the nodes in the raw graph. This helps learn better prototype vectors by leveraging both labeled nodes and massive unlabeled nodes in the graph, which is particularly useful in few-shot scenarios. Finally, in the testing stage, node/graph classification can be conducted via comparing the similarity of representations between node/graph and the prototype vectors. Compared with existing graph prompt tuning methods, our method is more desirable in learning better prototype vectors, as we leverage both labeled nodes and massive unlabeled nodes in the graph, which is particularly useful in few-shot scenarios. We further highlight that our prompt tuning method is applicable to
both homophilous and heterophilous graphs. First, node/graph representations computed from MLP-based view are not affected by structural heterophily. Second, prototype vectors calculated from GNN-based view are based on the learned weights in structure prompt tuning, which takes all nodes in the raw graph as neighbors and learns to assign large (small) weights to those in the same (different) class. As such, the computation of prototype vectors is less affected by graph heterophily. To summarize, our main contributions in this paper are: – We propose an effective graph pre-training and prompt tuning framework PSP, which unifies the objectives of pre-training and prompt tuning. – We present a novel prompt tuning strategy, which introduces a learnable structure prompt to learn high-quality class prototype vectors and enhance model performance on both homophilous and heterophilous graphs. – We extensively demonstrate the effectiveness of PSP with different benchmark datasets on both node classification and graph classification. In particular, we vary the number of labeled training data and show that PSP can lead to better performance in challenging few-shot scenarios.
# 2 Related work
# 2.1 Graph Pre-training
Inspired by the remarkable achievements of pre-trained models in Natural Language Processing (NLP) [15] and Computer Vision (CV) [19], graph pre-training [28] emerges as a powerful paradigm that leverages self-supervision on label-free graphs to learn intrinsic graph properties. Some effective and commonly-used pre-training strategies include node-level comparison [32], edge-level pretext [9], and graph-level contrastive learning [29]. Recently, there are also some newly proposed pre-training methods [8,16]. However, these approaches do not consider the gap between pre-training and downstream objectives, which limits their generalization ability to handle different tasks.
# 2.2 Prompt-based Learning
The training strategy “pre-train & fine-tune” is widely used to adapt pre-trained models onto specific downstream tasks. However, this strategy ignores the inherent gap between the objectives of pre-training and diverse downstream tasks, where the knowledge learned via pre-training could be forgotten or ineffectively leveraged for downstream tasks, leading to poor performance. To bridge this gap, NLP proposes a new paradigm, namely “pre-train & prompt”. These methods freeze the parameters of the pre-trained models and introduce additional learnable components in the input space, thereby enhancing the compatibility between inputs and pre-trained models. On graph data, there are a handful of studies that adopt prompt tuning to learn more generalizable GNNs. GPPT [20] relies on edge prediction as the pre-training task and reformulates the downstream task as edge prediction by introducing task tokens
for node classification. GraphPrompt [14] proposes a unified framework based on subgraph similarity and link prediction, hinging on a learnable prompt to actively guide downstream tasks using task-specific aggregation in readout function. and computes class prototype vectors via supervised prototypical contrastive learning. GPF [4] extends the node embeddings with additional task-specific prompt parameters, and can be applied to the pre-trained GNN models that employ any pre-training strategy. ProG [21] reformulates node-level and edge-level tasks to graph-level tasks, and introduces the meta-learning technique to the graph prompt tuning study. Despite their success, we observe that most of them utilize the structure information in pre-training, while ignoring it in downstream prompt tuning stage for learning more accurate prototype vectors. This restricts their effectiveness to fully utilize pre-trained knowledge stored in the entire graph. In particular, their task-specific parameters are usually learned only with labeled nodes while massive unlabeled nodes are disregarded, leading to poor performance in more challenging few-shot scenarios. In addition, they ignore the impact of heterophilous neighborhoods on node representation and are not suitable for heterophilous graphs. In this paper, our proposed PSP employs a dual-view contrastive learning and integrates structure information in both pre-training and prompt tuning stage to construct more accurate prototype vectors, achieving superior performance in few-shot learning tasks on both homophilous and heterophilous graphs.
# 3 Preliminary
Graph. We denote a graph as G = (V, E), where V = {vi}N i=1 is a set of N nodes and E ⊆V × V is a set of edges. We also define A ∈RN×N as the adjacency matrix of G, where Aij = 1 if vi and vj are connected in G, and Aij = 0 otherwise. Each node vi in the graph is associated with an F-dimensional feature vector xi ∈R1×F , and the feature matrix of all nodes is defined as X ∈RN×F . Research problem. In this paper, we investigate the problem of graph pretraining and prompt tuning, which learns representations of graph data via pretraining, and transfer the pre-trained knowledge to solve downstream tasks, such as node classification and graph classification. Moreover, we further consider the scenarios where downstream tasks are given limited supervision, i.e., k-shot classification. For each class, only k labeled samples (i.e., nodes or graphs) are provided as training data. Prompt tuning of pre-trained models. Given a pre-trained model, a set of learnable prompt parameters θ and a labeled task dataset D, we fix the parameters of the pre-trained model and only optimize θ with D for the downstream graph tasks.
# 4 Proposed Method
In this paper, we propose a novel graph pre-training and structure prompt tuning framework, namely PSP, which unifies the objectives of pre-training and prompt
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1904/1904dbf4-9a73-43eb-8622-3cae428727bd.png" style="width: 50%;"></div>
Fig. 2: Overall framework of PSP. Top: pre-training. Middle: prompt tuning  node classification. Bottom: prompt tuning for graph classification.
<div style="text-align: center;">Fig. 2: Overall framework of PSP. Top: pre-training. Middle: prompt tuning for node classification. Bottom: prompt tuning for graph classification.</div>
tuning for GNNs and integrates structure information in both pre-training and prompt tuning stage to achieve better performance in more challenging few-shot scenarios. The overall framework of PSP is shown in Figure 2.
# 4.1 Graph Pre-training
For graph data, both node attributes and structural information are critical for revealing the underlying semantics of graph during the pre-training phase. Inspired by the success of LINKX [13] that separately learns node embeddings from attributes and graph structure for heterophilous graphs, we design a dualview contrastive learning method to align the latent semantic spaces of node attributes and graph structure. The upper part of Figure 2 shows the dual-view pre-training paradigm. In particular, one view is implemented with MLP, which only uses node attributes in the graph, while the other adopts GNN to leverage both node attributes and structural information of the graph. Formally, we define the node representations computed by the two views as:
# Z(1) = MLP(X) and Z(2) = GNN(X, A),
where Z(1) ∈RN×D and Z(2) ∈RN×D have the same latent dimensionality D. To optimize the MLP and GNN, we leverage a contrastive loss function Lpre to maximize the similarity between the two representations of the same node, denoted as z(1) i and z(2) i for node vi. For an anchor z(1) i , other node representations z(2) j are considered as negative samples. More specifically, we formulate
(1)
e-Training and Structure Prompt Tuning for Graph Neural Networks
e loss as the normalized temperature-scaled cross entropy loss [2] a
� where N is the number of nodes, τ is a temperature parameter, and sim(·) is implemented with cosine similarity. In general, the dual-view contrastive pretraining can exploit both attribute and structure information of graph to encode generalizable knowledge in the output node embeddings. In the next subsection, we introduce graph structure prompt tuning that leverages such knowledge in downstream classification tasks.
# 4.2 Graph Structure Prompt Tuning
Next, we demonstrate how we freeze the pre-trained model and adapt it to different downstream tasks on graph. In particular, we propose a novel method, namely structure prompt tuning, which considers the structure relationships between the graph data and the task at hand. Compared to existing graph prompt tuning methods, the structure relationships in our method allow the task to more effectively leverage the pre-trained knowledge embedded in the graph data. In the following, we elaborate the structure prompt tuning method on two representative tasks, i.e., node classification and graph classification.
Node Classification Our method is based on the prototype-based framework [14,33], which learns a prototype embedding vector pc for each node class c ∈{1, 2, ..., C}. In particular, our method comprises three steps: 1) structure prompting, 2) prompt initialization, and 3) prompt tuning. Step 1: Structure prompting. For all the class prototypes, the main idea of our method is to consider them as virtual nodes, and connect them to all the original nodes V in the graph G, as shown in Figure 1(c). More specifically, we add a total number of C prototype nodes (denoted as P = {pc}C c=1) and N × C weighted edges (denoted as W) to construct a prompted graph G′ as
Step 2: Prompt initialization. Next, we define the attributes and edge weights for the prototype nodes. In particular, we simply initialize the attributes for each prototype pc as the averaged attribute vector of labeled nodes in class c:
where Dc ⊆D denotes labeled nodes of class c in the training set. For the newly added edges that connect P and V, we adopt dot product to initialize their weights W ∈RN×C as:
where Dc ⊆D denotes labeled nodes of class c in the training set. For the newly added edges that connect P and V, we adopt dot product to initialize their weightsRN×C as:
(2)
(3)
(4)
(5)
where Z(2) denotes node embeddings derived from the pre-trained GNN and P represents the prototype vectors with the c-th row pc = 1 |Dc| � (vi,c)∈Dc z(2) i . During the subsequent prompt tuning step, the parameters of the pre-trained model are frozen, and the weight matrix W is considered as the only task-specific parameter to be learned. Step 3: Prompt tuning. We conduct prompt tuning on the prompted graph G′ using the same form of contrastive loss function as in Equation 2, while keeping the node representations fixed and only optimize the prototype embeddings. Formally, given a labeled dataset D for the downstream task, the prompt tuning loss is defined as follows:
# � where pc is parameterized by W and we have:
� where pc is parameterized by W and we have:
P = GNN([X, XP ], [A, W]).
Here, X represents the original node features in graph G, XP ∈RC×F represents the prototype features, and [A, W] ∈RN×(N+C) represents the new adjacency matrix of G′. Notably, unlike conventional methods that directly consider P as learnable parameters, we parameterize P with their structural connections with the graph data, i.e., the added edges W. In other words, only W would be optimized as parameters when minimizing Lpro, which learns to aggregate pre-trained knowledge from all the nodes in the graph to formulate prototype embeddings.
Graph Classification Our method can also be adapted to graph classification with minor changes in each step. In structure prompting, for each graph instance Gi and a prototype node pc, the added edges between any vi ∈Vi and pc share the same weight. As such, the prompt tuning of graph classification also has N × C parameters, where N is the number of graphs here. In prompt initialization, we introduce an average-based readout function on node level representations Z(1) and Z(2) to compute the graph-level representation S(1) and S(2), respectively, as shown in Figure 2. Then, we can use Equation 5 to initialize W with Z(2) replaced by S(2). In prompt tuning, we also replace Z(1) by S(1) in Equation 6 for optimizing graph-level classification tasks.
Remarks In our proposed PSP method, it worth noting that the weights of pre-trained model are frozen for downstream tasks, and the prompt tuning is parameterized by the learnable adjacency matrix W ∈RN×C. Compared with most existing studies where the prototype vectors are directly optimized on few labeled data, this allows the prototype vectors to aggregate pre-trained knowledge from massive unlabeled nodes for more effective task adaptation.
(6)
(7)
<div style="text-align: center;">Table 1: Statistics of the datasets</div>
Table 1: Statistics of the datasets
Cora CiteSeer PubMed ogbn-arxiv Chameleon Actor ENZYMES PROTEINS COX2 BZR COLLAB
Graph
1
1
1
1
1
1
600
1,113
467
405
5000
Graph classes
-
-
-
-
-
-
6
2
2
2
3
Avg. nodes
2,708
3,327
19,717
169,343
2,277
7,600
32.63
39.06
41.22 35.75
74.49
Avg. edges
5,429
4,732
44,338
1,166,243
31,421
26,752
62.14
72.82
43.45 38.36 2457.78
Node features 1433
3703
500
128
2325
931
18
1
3
3
367
Node classes
7
6
3
40
5
5
3
3
-
-
-
Task (N/G)
N
N
N
N
N
N
N,G
G
G
G
G
# 4.3 Inference
In the inference stage, classification is performed by comparing the similarity of representations between node/graph (from MLP-based view) and prototype vectors (from GNN-based view) from different classes. The class corresponding to the prototype vector with the largest similarity is taken as the final prediction of the node/graph. We use node classification as an example to explain in detail. By comparing the node representation with each class prototype vector pc, we can get the predicted class probability by:
� where the highest-scored class is chosen as the prediction. From Equation 8, we see that the computation aligns well with objectives in Equations 2 and 6.
# 5 Experiments
In this section, we conduct extensive experiments on node classification and graph classification tasks with 11 benchmark datasets to evaluate PSP.
# 5.1 Experimental Setup
Datasets. We evaluate the performance of PSP using various benchmark datasets with diverse properties, including homophilous graphs [11,7]: Cora, CiteSeer, PubMed, ogbn-arxiv, and heterophilous graphs [18,14,17]: Chameleon, Actor, ENZYMES, PROTEINS, COX2, BZR, COLLAB. We summarize these datasets in Table 1. Note that the “Task” row indicates the type of downstream task, where “N” represents node classification and “G” represents graph classification. Baselines. To evaluate the proposed PSP, we compare it with 4 categories of state-of-the-art approaches as follows. Supervised models: GCN [11] and GAT [23]. They use the labeled data to learn GNNs, which are then directly applied for the classification tasks. Graph pre-training models: EdgeMask [24] and GraphCL [29]. Following the transfer learning strategy of “pre-train & fine-tune”, the pre-trained models are fine-tuned on the downstream tasks. Graph few-shot learning models: CGPN [25] and Meta-PN [3]. They are
(8)
Table 2: Accuracy (%) on node classification with masking ratio of 50%. W highlight the best score on each dataset in bold. OOM denotes the out-of-th
<div style="text-align: center;">Table 2: Accuracy (%) on node classification with masking ratio of 50%. We highlight the best score on each dataset in bold. OOM denotes the out-of-the-</div>
memory error.
Methods
Cora
CiteSeer
PubMed
ogbn-arxiv
Chameleon
Actor
GCN
71.78 ± 0.50
52.15 ± 0.27
65.05 ± 0.15
64.19 ± 0.59
30.38 ± 1.21
18.67 ± 1.94
GAT
74.94 ± 1.26
59.50 ± 0.61
69.30 ± 0.97
63.97 ± 0.69
29.14 ± 0.79
20.85 ± 1.37
EdgeMask
76.38 ± 0.89
65.49 ± 0.90
71.29 ± 0.66
64.86 ± 0.67
30.89 ± 1.15
21.76 ± 0.95
GraphCL
76.73 ± 0.91
65.94 ± 1.20
72.03 ± 1.54
65.87 ± 0.82
27.37 ± 1.40
22.18 ± 1.24
CGPN
74.62 ± 0.99
65.32 ± 1.17
67.38 ± 1.43
OOM
31.12 ± 2.07
21.91 ± 0.39
Meta-PN
76.89 ± 1.05
65.80 ± 1.13
69.75 ± 1.11
57.37 ± 0.79
30.08 ± 1.25
19.16 ± 1.04
GPPT
77.16 ± 1.35
65.81 ± 0.97
72.23 ± 1.22
66.13 ± 0.44
31.28 ± 0.65
22.07 ± 0.81
GraphPrompt 68.43 ± 0.58
61.11 ± 1.24
72.63 ± 1.72
59.13 ± 0.59
31.67 ± 1.19
21.11 ± 1.35
GPF
62.20 ± 0.93
60.78 ± 1.17
68.81 ± 0.95
56.04 ± 0.97
29.31 ± 1.68
20.56 ± 1.09
ProG
72.49 ± 1.04
65.33 ± 0.93
73.70 ± 1.49
67.80 ± 1.65
31.75 ± 1.25
22.82 ± 1.03
PSP
77.62 ± 1.29 67.52 ± 0.95 75.94 ± 1.06 69.20 ± 0.73 34.08 ± 1.26 25.47 ± 0.82
specially designed for few-shot scenarios. Graph prompt models: GPPT [20], GraphPrompt [14], GPF [4] and ProG [21]. They adopt the “pre-train & prompt” paradigm, where the pre-trained models are frozen, and task-specific learnable prompts are introduced and trained in the downstream tasks. Note that we use GraphCL as the pre-training model for GPF. We also notice that SGL-PT [33] and VNT [22] are recent methods in the similar topic. However, both of them do not release their codes. For fairness, we do not take them as baselines. Implementation details. To train PSP, we adopt the Adam optimizer [10], where the learning rate and weight decay in the pre-training stage are fixed as 1e-4. We set the number of both graph neural layers and multilayer perceptron layers as 2. We set the hidden dimension for node classification as 128, and for graph classification as 32. Other hyper-parameters are fine-tuned on the validation set by grid search. In the prompt tuning stage, the learning rate is adjusted within {0.0001, 0.001, 0.01, 0.1}, the weight decay is chosen from {1e-5, 1e-4, 1e-3, 1e-2} and the dropout rate is selected from the range [0.2, 0.8]. Further, for those non-prompt-based competitors, some of their results are directly reported from [20] and [14] (i.e., node classification with 50% masking ratio and graph classification). For other cases and prompt-based models, we fine-tune hyperparameters with the codes released by their original authors. For fair comparison, we report the average results with standard deviations of 5 runs for node classification experiments, while the setting of graph classification experiments follows [14]. We run all the experiments on a server with 32G memory and a single Tesla V100 GPU.
# 5.2 Node classification
Experimental setting. For homophilous graph datasets, we use the official splitting of training/validation/testing [11]. For heterophilous graph datasets, we randomly sample 20 nodes per class as training set and validation set, respec-
<div style="text-align: center;">Table 3: Accuracy (%) on few-shot node classification.</div>
Table 3: Accuracy (%) on few-shot node classification.
Methods
Cora
CiteSeer
PubMed
ENZYMES
Chameleon
Actor
GCN
57.83 ± 5.90 49.69 ± 4.39 63.16 ± 4.56
61.49 ± 12.87
27.88 ± 5.77
20.69 ± 2.96
GAT
60.34 ± 4.15 52.85 ± 3.69 64.26 ± 3.17
59.94 ± 2.86
26.97 ± 4.85
20.93 ± 2.67
EdgeMask
64.10 ± 2.79 55.23 ± 3.17 65.89 ± 4.26
56.17 ± 14.39
23.76 ± 3.74
18.03 ± 2.48
GraphCL
65.89 ± 3.45 58.37 ± 4.74 69.06 ± 3.24
58.73 ± 16.47
22.25 ± 3.14
19.56 ± 1.15
CGPN
66.73 ± 2.87 57.14 ± 3.75 65.68 ± 2.38
66.52 ± 16.18
27.17 ± 2.16
20.68 ± 0.99
Meta-PN
66.65 ± 3.15 58.20 ± 2.94 67.18 ± 3.12
55.92 ± 13.28
25.83 ± 2.64
18.31 ± 1.95
GPPT
64.55 ± 3.72 55.63 ± 2.55 70.07 ± 6.07
53.79 ± 17.46
28.91 ± 3.23
20.88 ± 1.69
GraphPrompt 63.91 ± 2.43 53.42 ± 4.98 68.93 ± 3.93
67.04 ± 11.48
26.35 ± 3.50
20.50 ± 2.45
GPF
63.52 ± 5.39 54.31 ± 5.21 63.98 ± 3.54
60.13 ± 15.37
27.38 ± 3.62
19.32 ± 2.52
ProG
65.68 ± 4.29 59.07 ± 2.73 64.57 ± 3.81
57.22 ± 17.41
29.18 ± 4.53
21.43 ± 3.27
PSP
68.65 ± 2.17 61.7 ± 4.21 72.23 ± 4.20 72.86 ± 14.58 33.23 ± 3.80 24.74 ± 2.79
tively. The remaining nodes which are not sampled will be used for evaluation. Following the setting of [20], we randomly mask 50% of the training labels, which corresponds to 10-shot for datasets except ogbn-arxiv. Results. Table 2 summarizes the results, from which we see that: (1) Supervised learning methods generally perform worse than pre-training methods and prompt methods. This is because the annotations required by supervised frameworks are not enough. In contrast, pre-training approaches are usually facilitated with more prior knowledge, alleviating the need for labeled data. However, these pre-training methods still face an inherent gap between the training objectives of pre-training and downstream tasks. Pre-trained models may suffer from catastrophic forgetting during downstream adaptation. Therefore, we can find that compared with pre-training approaches, prompt-based methods usually achieve better performance. (2) Our proposed PSP outperforms all the baselines on node classification. This is because PSP bridges the gap between the pre-training stage and the down-stream prompt tuning stage, leveraging graph structure prompt to provide more information from the massive unlabeled nodes and being applicable to both homophilous and heterophilous graphs.
# 5.3 Few-shot node classification
Experimental setting. To explore more challenging few-shot node classification settings, we assign a much smaller number of labeled data as the training data for each class. Specifically, for ENZYMES, we follow existing study [14] to only choose graphs that consist of more than 50 nodes, which ensures there are sufficient labeled nodes for testing. On each graph, we randomly sample 1 node per class for training and validation, respectively. The remaining nodes which are not sampled will be used for testing. For Cora, CiteSeer and PubMed, we randomly sample 3 nodes per class for training, while the validation and test sets follow the official splitting [11]. For Chameleon and Actor, we randomly sample
3 nodes per class for training and validation respectively, while the remaining nodes which are not sampled are used for testing. Results. Table 3 illustrates the results. From the table, we see that: (1) In the extremely few-shot scenarios, PSP can still achieve superior performance. Taking ENZYMES as an example, the accuracy of PSP is 5.82% higher than the runner-up. This is because PSP uses graph structure information to optimize the prototype vectors, so when the training set is extremely small, the prototype vectors can also be accurate by capturing the underlying information of other unlabeled nodes. In contrast, few-shot learning models may suffer from inaccurate pseudo labels and other methods fail to consider the relationship between the task and the massive unlabeled data. (2) Our proposed PSP achieves larger improvements in heterophilous graphs. PSP achieves a largest improvement of 2.97% in homogeneous graphs while 5.82% in heterophilous graphs. This is because in the downstream prompt tuning stage of PSP, node representations computed from MLP-based view are not affected by the structural heterophily. Also, the graph structure prompt reduces the adverse influence from graph heterophily. Therefore, the prototype vector calculated from GNN-based view are more accurate. In contrast, other methods are susceptible to structure heterophily and can only achieve sub-optimal results.
# 5.4 Few-shot graph classification
Following the setting of [14], we conduct 5-shot tasks. The results are listed in Table 4, from which we observe that our proposed PSP significantly outperforms the baselines on these datasets. This again demonstrates the effectiveness of our proposed method. Notably, as both node and graph classification tasks share the same pre-trained model on ENZYMES, the superior performance of PSP on both types of tasks further demonstrates that the gap between different tasks is better addressed by our unified framework.
# 5.5 Model Analysis
We further analyse several aspects of our model. The following experiments are conducted on the 3-shot node classification and 5-shot graph classification. Ablation study on training paradigm. We conduct an ablation study that compares variants of PSP with different pre-training and prompt tuning strategies: (1) We directly fine-tune the pre-trained models on the tasks, instead of prompt tuning. We call this variant PSP-ft (with fine-tune). (2) For downstream tasks, we remove the prompt tuning process and only use the mean embedding vectors of the labeled data as the prototype vectors to perform classification. We call this variant PSP-np (no prompt). (3) We replace our proposed dual-view contrastive learning in pre-training with GraphCL, i.e., the pre-trained encoder is GNN, while the downstream tasks still use our proposed graph structure prompt. We call this variant PSP-CL (with GraphCL as the pre-training model). Figure 3 shows the results of this study, we observe that
<div style="text-align: center;">Table 4: Accuracy (%) on graph classification.</div>
Table 4: Accuracy (%) on graph classification.
Methods
ENZYMES
PROTEINS
COX2
BZR
COLLAB
GCN
20.37 ± 5.24 54.87 ± 11.20 51.37 ± 11.06 56.16 ± 11.07 50.62 ± 7.13
GAT
15.90 ± 4.13 48.78 ± 18.46 51.20 ± 27.93 53.19 ± 20.61 51.08 ± 7.59
InfoGraph
20.90 ± 3.32
54.12 ± 8.20
54.04 ± 9.45
57.57 ± 9.93
52.13 ± 8.71
GraphCL
28.11 ± 4.00
56.38 ± 7.24 55.40 ± 12.04 59.22 ± 7.42
52.81 ± 9.05
CGPN
24.75 ± 5.71
53.68 ± 9.15
52.16 ± 9.37
58.24 ± 8.47
50.05 ± 6.91
Meta-PN
21.05 ± 4.58
54.17 ± 8.36 52.83 ± 10.26 56.37 ± 13.15 51.71 ± 7.12
GPPT1
-
-
-
-
-
GraphPrompt 31.45 ± 4.32
64.42 ± 4.37
59.21 ± 6.82
61.63 ± 7.68
55.16 ± 6.24
GPF
32.65 ± 5.73
57.16 ± 5.96
61.62 ± 7.47
59.17 ± 6.18
53.91 ± 8.25
ProG
29.18 ± 3.09
60.98 ± 7.49
61.96 ± 6.35
63.71 ± 5.25
54.93 ± 7.24
PSP
33.57 ± 4.72 64.95 ± 5.86 65.71 ± 5.34 68.58 ± 7.57 57.29 ± 6.17
 
1 GPPT[20] lacks a unified effort to address graph classification tasks.
<div style="text-align: center;">Table 5: Varying the ratio of added edges on few-shot node classification. ↓/↑ means the decreasing/improvement compared with the accuracy of runner-up.</div>
means the decreasing/improvement compared with the accuracy of runner-up.
runner-up
0%
0.1%
1%
5%
10%
50%
100%
Cora
65.89±3.45 65.06±3.77 ↓65.72±3.14 ↓67.50±2.38 ↑67.46±2.45 ↑67.88±2.83 ↑68.62±2.67 ↑68.68±2.17 ↑
PubMed
70.07±6.07 70.84±5.31 ↑70.32±3.95 ↑71.62±4.34 ↑70.94±4.40 ↑71.42±4.70 ↑72.24±4.33 ↑73.23±4.20 ↑
Chameleon 29.18±4.53 25.22±3.09 ↓30.92±3.07 ↑33.41±3.38 ↑33.30±2.36 ↑33.07±3.79 ↑32.09±2.20 ↑33.97±3.22 ↑
(1) PSP-np always performs the worst among all the variants, showing the effectiveness of our proposed graph structure prompt. PSP-ft achieves better performance than PSP-np, which is because PSP-ft is parameterized. (2) PSP achieves comparable results to PSP-CL in homogeneous graph and clearly outperforms PSP-CL in heterophilous graph. This is because our pretraining method uses MLP and GNN for contrastive learning, where MLP is not affected by the heterophily of the graph. In contrast, PSP-CL is susceptible to structure heterophily. Varying the number of the shots. For node classification, we vary the number of shots between 1 and 10. For graph classification, we vary the number of shots between 1 and 30. We compare PSP with several competitive baselines in Figure 4. In general, PSP consistently outperforms the baselines, especially when the number of shots is few. We further notice that when the number of shots is relatively large, PSP can be surpassed by graphCL on graph classification, especially on COX2. This could be contributed to more available training data on COX2, where 30 shots per class implies that 12.85% of the 467 graphs are used for training. This is not our target few-shot scenario. Varying the ratio of added edges. Our prompt tuning method is parameterized by the added edges between original nodes and prototype vectors, where the weights of added edges are learnable. We hereby study the impact of the ratio of newly added edges (i.e., parameters) r. To vary the number of edges, we first
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4277/4277df82-1b75-4194-8255-08308a9536e2.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 3: The ablation study on training paradigm.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6fca/6fcae91f-4d3d-427d-830b-3764218445a2.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 4: (a)(b): varying the number of shots for node classification. (c)(d): varying the number of shots for graph classification.</div>
randomly select rN nodes outside the training set. During prompt tuning, we next combine the rN nodes with Nt training nodes as the set of nodes that add connections with all the C prototypes. Following this setting, we conduct 3-shot node classification on 4 datasets. The results are shown in Table 5. Surprisingly, we observe that for Cora and Chameleon, PSP can surpass the runner-up when r = 1% and 0.1%, respectively. For PubMed, PSP can outperform the runner-up even when r = 0. This shows that our prompt tuning model can achieve superior performance with only a small number of parameters (Nt + rN)C, where r is a small number. Therefore, PSP is promising in scaling to large graphs. Visualization of learned edge weights. We visualize the weight matrix W of the added edges after prompt tuning. As shown in Figure 5, for each node, its edge connected to the corresponding class prototype is more likely to have a larger weight. Hence, the prototype vectors are very accurate by aggregating massive unlabeled data which contains rich pre-trained knowledge to reflect the semantics of the task labels.
# 6 conclusion
In this paper, we proposed PSP, a novel pre-training and structure prompt tuning framework for GNNs, which unified the objectives of pre-training and prompt
PSP: Pre-Training and Structure Prompt Tuning for Graph Neural Networks
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7481/7481cbbc-9af4-4736-bd62-d086dc2152a3.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) After Prompt Tuning</div>
<div style="text-align: center;">Fig. 5: The weights of added edges between nodes and class prototype vector before and after prompt tuning.</div>
tuning for GNNs and integrated structural information in both pre-training and prompt tuning stages to construct more accurate prototype vectors. For pretraining, we proposed a dual-view contrastive learning to align the latent semantic spaces of node attributes and graph structure. For downstream prompt tuning, we proposed to learn the structural connection between the prototype vectors and the graph, and then leveraged the learned structural information to perform better in few-shot tasks. Finally, we conducted extensive experiments and showed that PSP significantly outperforms various state-of-the-art baselines on both homophilous and heterophilous graphs, especially on few-shot scenarios.
# 7 Acknowledgement
This work is supported by National Natural Science Foundation of China No. 62202172 and Shanghai Science and Technology Committee General Program No. 22ZR1419900.
# References
1. Chen, M., Liu, Z., Liu, C., Li, J., Mao, Q., Sun, J.: Ultra-dp: Unifying graph pre-training with multi-task graph dual prompt. arXiv preprint arXiv:2310.14845 (2023) 2. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A simple framework for contrastive learning of visual representations. In: International conference on machine learning. pp. 1597–1607. PMLR (2020) 3. Ding, K., Wang, J., Caverlee, J., Liu, H.: Meta propagation networks for graph few-shot semi-supervised learning. In: AAAI. vol. 36, pp. 6524–6531 (2022) 4. Fang, T., Zhang, Y., Yang, Y., Wang, C., Chen, L.: Universal prompt tuning for graph neural networks. arXiv preprint arXiv:2209.15240 (2022) 5. Hamilton, W., Ying, Z., Leskovec, J.: Inductive representation learning on large graphs. NeurIPS 30 (2017) 6. Hou, Z., Liu, X., Cen, Y., Dong, Y., Yang, H., Wang, C., Tang, J.: Graphmae: Self-supervised masked graph autoencoders. In: KDD. pp. 594–604 (2022)
1. Chen, M., Liu, Z., Liu, C., Li, J., Mao, Q., Sun, J.: Ultra-dp: Unifying graph pre-training with multi-task graph dual prompt. arXiv preprint arXiv:2310.14845 (2023) 2. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A simple framework for contrastive learning of visual representations. In: International conference on machine learning. pp. 1597–1607. PMLR (2020) 3. Ding, K., Wang, J., Caverlee, J., Liu, H.: Meta propagation networks for graph few-shot semi-supervised learning. In: AAAI. vol. 36, pp. 6524–6531 (2022) 4. Fang, T., Zhang, Y., Yang, Y., Wang, C., Chen, L.: Universal prompt tuning for graph neural networks. arXiv preprint arXiv:2209.15240 (2022) 5. Hamilton, W., Ying, Z., Leskovec, J.: Inductive representation learning on large graphs. NeurIPS 30 (2017) 6. Hou, Z., Liu, X., Cen, Y., Dong, Y., Yang, H., Wang, C., Tang, J.: Graphmae: Self-supervised masked graph autoencoders. In: KDD. pp. 594–604 (2022)
7. Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., Leskovec, J.: Open graph benchmark: Datasets for machine learning on graphs. NeurIPS 33, 22118–22133 (2020) 8. Hu, Z., Dong, Y., Wang, K., Chang, K.W., Sun, Y.: Gpt-gnn: Generative pretraining of graph neural networks. In: KDD. pp. 1857–1867 (2020) 9. Jin, W., Derr, T., Liu, H., Wang, Y., Wang, S., Liu, Z., Tang, J.: Selfsupervised learning on graphs: Deep insights and new direction. arXiv preprint arXiv:2006.10141 (2020) 10. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014) 11. Kipf, T.N., Welling, M.: Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907 (2016) 12. Li, X., Ye, T., Shan, C., Li, D., Gao, M.: Seegera: Self-supervised semi-implicit graph variational auto-encoders with masking. In: WebConf. pp. 143–153 (2023) 13. Lim, D., Hohne, F., Li, X., Huang, S.L., Gupta, V., Bhalerao, O., Lim, S.N.: Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods. NeurIPS 34, 20887–20902 (2021) 14. Liu, Z., Yu, X., Fang, Y., Zhang, X.: Graphprompt: Unifying pre-training and downstream tasks for graph neural networks. In: WebConf. pp. 417–428 (2023) 15. Long, S., Cao, F., Han, S.C., Yang, H.: Vision-and-language pretrained models: A survey. arXiv preprint arXiv:2204.07356 (2022) 16. Lu, Y., Jiang, X., Fang, Y., Shi, C.: Learning to pre-train graph neural networks. In: AAAI. vol. 35, pp. 4276–4284 (2021) 17. Morris, C., Kriege, N.M., Bause, F., Kersting, K., Mutzel, P., Neumann, M.: Tudataset: A collection of benchmark datasets for learning with graphs. arXiv preprint arXiv:2007.08663 (2020) 18. Pei, H., Wei, B., Chang, K.C.C., Lei, Y., Yang, B.: Geom-gcn: Geometric graph convolutional networks. arXiv preprint arXiv:2002.05287 (2020) 19. Qiu, X., Sun, T., Xu, Y., Shao, Y., Dai, N., Huang, X.: Pre-trained models for natural language processing: A survey. Science China Technological Sciences 63(10), 1872–1897 (2020) 20. Sun, M., Zhou, K., He, X., Wang, Y., Wang, X.: Gppt: Graph pre-training and prompt tuning to generalize graph neural networks. In: KDD. pp. 1717–1727 (2022) 21. Sun, X., Cheng, H., Li, J., Liu, B., Guan, J.: All in one: Multi-task prompting for graph neural networks (2023) 22. Tan, Z., Guo, R., Ding, K., Liu, H.: Virtual node tuning for few-shot node classification. arXiv preprint arXiv:2306.06063 (2023) 23. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., Bengio, Y.: Graph attention networks. arXiv preprint arXiv:1710.10903 (2017) 24. Veličković, P., Fedus, W., Hamilton, W.L., Liò, P., Bengio, Y., Hjelm, R.D.: Deep graph infomax. arXiv preprint arXiv:1809.10341 (2018) 25. Wan, S., Zhan, Y., Liu, L., Yu, B., Pan, S., Gong, C.: Contrastive graph poisson networks: Semi-supervised learning with extremely limited labels. NeurIPS 34, 6316–6327 (2021) 26. Wang, D., Lin, J., Cui, P., Jia, Q., Wang, Z., Fang, Y., Yu, Q., Zhou, J., Yang, S., Qi, Y.: A semi-supervised graph attentive network for financial fraud detection. In: ICDM. pp. 598–607. IEEE (2019) 27. Wu, S., Sun, F., Zhang, W., Xie, X., Cui, B.: Graph neural networks in recommender systems: a survey. ACM Computing Surveys 55(5), 1–37 (2022) 28. Xia, J., Zhu, Y., Du, Y., Li, S.Z.: A survey of pretraining on graphs: Taxonomy, methods, and applications. arXiv preprint arXiv:2202.07893 (2022)
29. You, Y., Chen, T., Sui, Y., Chen, T., Wang, Z., Shen, Y.: Graph contrastive learning with augmentations. NeurIPS 33, 5812–5823 (2020) 30. Yu, X., Liu, Z., Fang, Y., Liu, Z., Chen, S., Zhang, X.: Generalized graph prompt: Toward a unification of pre-training and downstream tasks on graphs. arXiv preprint arXiv:2311.15317 (2023) 31. Yu, X., Zhou, C., Fang, Y., Zhang, X.: Multigprompt for multi-task pre-training and prompting on graphs. arXiv preprint arXiv:2312.03731 (2023) 32. Zhu, Y., Xu, Y., Yu, F., Liu, Q., Wu, S., Wang, L.: Graph contrastive learning with adaptive augmentation. In: WebConf. pp. 2069–2080 (2021) 33. Zhu, Y., Guo, J., Tang, S.: Sgl-pt: A strong graph learner with graph prompt tuning. arXiv preprint arXiv:2302.12449 (2023)
