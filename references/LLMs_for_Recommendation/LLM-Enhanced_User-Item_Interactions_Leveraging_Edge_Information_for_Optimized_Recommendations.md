# LLM-Enhanced User-Item Interactions: Leveraging Edge Information for Optimized Recommendations
Xinyuan Wang Arizona State University xwang735@asu.edu Liang Wu Linkedin liawu@linkedin.com
Liang Wu Linkedin liawu@linkedin.com
Xinyuan Wang Arizona State University xwang735@asu.edu
Hao Liu HKUST (Guangzhou) liuh@ust.hk Yanjie Fu Arizona State University yanjie.fu@asu.edu
Hao Liu HKUST (Guangzhou) liuh@ust.hk
# ABSTRACT
The extraordinary performance of large language models has not only reshaped the research landscape in the field of natural language processing but has also demonstrated its exceptional applicative potential in various domains. However, the potential of these models in mining relationships from graph data remains under-explored. Graph neural networks, as a popular research area in recent years, have numerous studies on relationship mining. Yet, current cuttingedge research in graph neural networks has not been effectively integrated with large language models, leading to limited efficiency and capability in graph relationship mining tasks. A primary challenge is the inability of LLMs to deeply exploit the edge information in graphs, which is critical for understanding complex node relationships. This gap limits the potential of LLMs to extract meaningful insights from graph structures, limiting their applicability in more complex graph-based analysis. We focus on how to utilize existing LLMs for mining and understanding relationships in graph data, applying these techniques to recommendation tasks. We propose an innovative framework that combines the strong contextual representation capabilities of LLMs with the relationship extraction and analysis functions of graph neural networks for mining relationships in graph data. Specifically, we design a new prompt construction framework that integrates relational information of graph data into natural language expressions, aiding LLMs in more intuitively grasping the connectivity information within graph data. Additionally, we introduce graph relationship understanding and analysis functions into LLMs to enhance their focus on connectivity information in graph data. By enhancing the understanding of graph relationships by LLMs, our framework provides more comprehensive, accurate, and personalized recommendations. Our evaluation on real-world datasets demonstrates the framework’s ability to understand connectivity information in graph data and to improve the relevance and quality of recommendation results. Our code is released at: https://github.com/anord-wang/LLM4REC.git
# 1 INTRODUCTION
We aim to explore new methods for applying large language models to recommendation systems, especially when dealing with data containing a large amount of edge information. Our goal is to develop a recommendation system that can effectively utilize this edge information while integrating the powerful text-processing capabilities of LLM. Through this approach, not only can we improve the relevance and accuracy of recommendations, but we can also provide a richer and more personalized experience.
Liangjie Hong Linkedin liahong@linkedin.com
Yanjie Fu Arizona State University yanjie.fu@asu.edu
In recent years, significant progress has been made in the development of LLMs such as BERT and GPT. These models demonstrate excellent performance in natural language understanding and generation, showing enormous potential in applications in multiple fields [7] [21]. With the advancement of technology, LLM’s ability to handle complex text data continues to enhance, providing new solutions for various tasks and opening a new chapter in intelligent system research [2]. In this context, combining LLM with recommendation systems has become a cutting-edge and revolutionary research field [16]. Traditional recommendation systems focus on analyzing users’ behavior data, while when combined with LLM, recommendation systems can understand explicit feedback from users and mine deeper into their implicit needs and preferences [27]. This combination provides new possibilities for improving the accuracy and satisfaction of recommendation systems. A major challenge in current recommendation systems is that although data contain a large amount of edge information (such as the relationship between users and items), this information is not fully utilized in LLMs, especially in its key attention mechanism [31]. Even though people have used this edge information to construct a variety of prompts [35], existing methods of building recommendation systems using LLMs cannot consider edge information structurally. This raises a research question: how to effectively integrate graph information within the framework of LLMs to improve the performance of recommendation systems? To overcome the challenges faced by existing recommendation systems in processing data containing a large amount of edge information, we propose an innovative prompt mechanism that can convert the connection relationship between users and items, as well as the background information of items, into natural language text. Our method considers the direct relationship between users and items and enriches the model’s understanding by constructing a second-order relationship between items, a complex association not directly present in traditional recommendation data. At the same time, we also introduce an improved attention mechanism. This mechanism calculates spatial information between nodes in the graph based on the connection relationship between users and items and directly integrates this information into the computation of the attention mechanism as edge encoding. This method enhances the model’s ability to process graph-structured data and greatly improves the accuracy and efficiency of recommendation systems through a deep understanding of the multidimensional relationships between users and items within the model. Through
these innovations, we can effectively integrate complex graph structure information into the LLMs framework, greatly improving the model’s comprehensive understanding of users’ behavior and items’ characteristics. This enhances the model’s ability to simulate realworld user-item interactions and lays a solid foundation for generating more accurate and personalized recommendation results. In our model, each recommendation is not only a simple response based on users’ historical behavior but also an intelligent decisionmaking process that deeply and comprehensively considers users’ preferences and items’ attributes. To validate the effectiveness of our method, we conduct a series of experiments on different recommendation datasets, including validation experiments on the novel prompt mechanism and improved injection attention mechanism. These experiments demonstrate the performance improvement of our method on recommendation tasks and reveal its potential in handling complex user-item relationships. Our contributions are as follows:
• An innovative way of combining LLMs with recommendation systems: By creatively integrating LLMs’ deep text understanding ability with users’ behavior analysis of recommendation systems, our method can more comprehensively understand users’ and items’ information. Our method effectively utilizes the language processing capabilities of LLMs, bringing new dimensions and depth to traditional recommendation logic, thereby ensuring recommendation quality while also increasing the diversity and innovation. • A new prompt strategy: We introduce a new prompt mechanism that can transform the relationship between users and items, as well as the background information of items, into natural language form. In addition, by constructing secondorder relationships between items, we can uncover deeper correlations between items, thereby providing more comprehensive and detailed recommendations. • A novel fusion method for edge information: We propose a new approach that directly embeds the edge information of graph data into the attention mechanism of LLMs. This method effectively utilizes the connection information in the graph structure, enhancing the model’s ability to handle complex user-item interactions.
# 2 PRELIMINARIES
# 2.1 Important Definitions
Generative Large Language Models. Generative Large Language Models (LLMs) are a type of model based on transformer encoders that generate natural language texts [26]. LLMs are trained on massive text corpora and can capture broad contextual relationships between words. LLMs generate a series of words (𝑤1,𝑤2, . . . ,𝑤𝑛) by modeling the joint probability of word sequences 𝑃(𝑤1,𝑤2, . . . ,𝑤𝑛). Token and Embedding. In NLP, tokens are the smallest units for LLMs, such as words, sub-words, or characters [33]. Embedding is a dense vector representation of a token in a continuous space that encodes language attributes and semantic information [24]. In recommendations, we see users and items as unique tokens. Prompt. Prompts are designed to guide generative large language models to generate specific responses. They serve as guides for the model to generate text tokens in specific contexts or styles [19]
# 2.2 Problem Statement
Consider the existence of 𝐼users and 𝐽items, let 𝑋𝑖𝑗be the binary interaction (e.g., purchase) matrix between user 𝑖and item 𝑗. Besides, we collect user descriptions, item descriptions (e.g., prices, brand, category, title), user reviews for items, explanations of user purchase reasons. We denote 𝑇𝑖as the descriptions of the user 𝑖, 𝑇𝑗 as the descriptions of the item 𝑗, 𝑇𝑖𝑗as the joint texts of the user 𝑖and item 𝑗, such as user reviews and purchase reasons for items. We unify all textual descriptions into 𝑇that includes 𝑁sequences, 𝑘indexes the tokens in each sequence, and 𝑇𝑛𝑘is the 𝑘−𝑡ℎtoken in the 𝑛−𝑡ℎsequence. Our goal is to leverage LLMs and graphs to develop a generative recommendation system that takes a prompt including a userID and a user’s historical interaction records with items, and generate product recommendations to the user.
# 3 LEVERAGING LLM AND GRAPHS FOR RECOMMENDER SYSTEMS 3.1 Framework Overview
Given user descriptions, item descriptions, user textual reviews for items, and the user-item interaction (e.g., purchase, rating) graph, we aim to leverage and connect generative LLM, textual generation, and user-item interaction graph to advance recommender systems [35]. Figure 1 shows the four major steps for building our recommender system: 1) graph knowledge guided attentive LLM recommendation model backbone; 2) pre-training of graph attentive LLM with crowd contextual prompts; 3) fine-tuning of graph attentive LLM with personalized predictive prompts; 4) using fine-tuned graph attentive LLM for item recommendation. In Step 1, considering the existence of multi-source textual information, including user descriptions, item descriptions, and user reviews for items, we propose to leverage LLM to learn the generation of these texts in order to model the representations of user preferences and item functionalities. Aside from texts, the user-item interaction graph can provide two types of signals: i) user nodes and item nodes as tokens and ii) graph structure information about the direct (first-order) connectivity between users and items and indirect (second-order) connectivity among items, users, or useritem pairs. To leverage the user and item tokens, we add the user and item tokens to enrich the texts; to leverage the graph structure information, we develop a graph knowledge guided attentive LLM backbone, particularly with a new neural attentive decoder structure, to model the first-order and second-order connectivity graph knowledge. Our graph attentive LLM backbone strategically reformulates the user-item recommendation problem into a probabilistic generative problem in response to prompts. In Step 2, we develop crowd contextual prompts to pre-train the graph attentive LLM by maximizing textual generation likelihood. Specifically, we first construct the text prompts of all users and items including not just the texts of user descriptions, item descriptions, and user reviews for items, but also the fact or event checking texts of whether a user interacts with (e.g., rates or purchases) an item. We merge all the description, review, fact and event-based texts together as crowd contextual prompts. We then utilize the crowd contextual prompts to pre-train the graph attentive LLM backbone. The optimization goal of Step 2 is to maximize text generative likelihood, so the LLM can learn the contextual knowledge of a recommendation world. In Step 3, we develop personalized predictive prompts to incentivize
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/93ac/93ac3065-2009-4111-b54c-293d33722e9f.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a348/a348588b-0f23-4767-a2f6-af83db46b550.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
the pre-trained graph attentive LLM backbone to advance generative recommendation accuracy. Specifically, we convert a user’s interaction (e.g., rating, purchases) history with all items into past tense texts, combined with a future tense trigger (e.g., user 𝑖will purchase ? −−−), to motivate the graph attentive LLM to recommend items. The optimization goal of Step 3 is to minimize recommendation errors, not textual generation likelihood. In Step 4, given a test user with the corresponding personalized rating or purchase history and a predictive trigger, we leverage the fine-tuned graph attentive LLM backbone to recommend items to the test user.
# 3.2 Graph Structured Attentive LLM based Generative Recommendation Backbone
3.2.1 GPT2 as LLM Base Model. Our base model is the GPT-2 [26]. The original GPT-2 utilizes the Transformer architecture, pre-trained on vast text datasets to predict subsequent words in sequences. With a multi-layer structure containing attention heads, it scales up to billions of parameters for enhanced pattern recognition. The model supports conditional text generation and offers various sampling strategies for generating text. In GPT-2, the attention mechanism is to weigh the importance of different words in a sentence. It operates by computing attention scores for each word in the input sequence based on their relevance to each other. These scores determine how much attention the model should pay to each word when generating the next word in the sequence. By attending to relevant parts of the input text, GPT-2 can capture long-range dependencies and generate coherent and contextually relevant output.
3.2.2 Integrating Two Structure Knowledge for Graph Attentive LLM. When performing generative recommendations, we obtain recommendation results in the form of text generation to connect items to users. As a result, users and items are usually regarded as tokens in a text sequence for pre-training. In the real world, users interact (e.g., rate, purchase) with items. Such interactions can be modeled as a graph where users or items are seen as nodes, and user-user, itemitem, user-item connectivity is seen as edge weights, representing
<div style="text-align: center;"></div>
a kind of graph-structured information propagation-driven edge knowledge. In other words, user and item tokens are not just simply independent entities in a sequence. The LLM should not just learn user and item embeddings by paying attention to their mutual relevance in a sequence. It is critical to leverage the graph-structured edge knowledge to improve LLM for recommendations. Firstly, we propose a graph structure knowledge attentive LLM method in order to integrate graph knowledge into recommendation systems. Specifically, we incorporate the edge connectivity between users and items into attention weight calculation [5, 37]. Our idea is to leverage Graph Neural Networks (GNNs) to describe the relationships between nodes by modeling their connectivity (direct relationship) and spatial information (indirect relationship) in the graph. Formally, the edge information, denoted by the 𝑅-term is used to calculate the attention weights in the graph-structured attention mechanism, which is given by:
(1)
√︁ where 𝑄, 𝐾, and 𝑉are queries, keys, and values respectively, while 𝑅represents the relationship encoding extracted from graph knowledge. The √︁ 𝑑𝑘is the dimensionality (size) of the key vector to enforce a normalization effect. Secondly, we identify two kinds of important graph structural knowledge: 1) the direct (first-order) connectivity and 2) the indirect (high-order path) connectivity, among users and items. Correspondingly, the graph structured relational attention 𝑅term is composed of two distinct parts of the graph topology:
# 𝑅= 𝑅conn + 𝑅path.
(2)
The first part, 𝑅conn, represents the direct connection relationships between nodes. Typically, we denote 𝑅conn as a binary adjacency
𝑅𝑐𝑜𝑛𝑛 𝑖𝑗 = � 1, if there is a direct connection between node 𝑖and 𝑗 0, otherwise
(3)
In this matrix, 1 indicates there is an edge between two nodes, and 0 indicates there is no edge between two nodes. The second part, 𝑅path, represents a normalized shortest path score between nodes, which is computed based on the entire graph. The shortest path information is essential because it reflects the indirect relationships of node pairs and the degrees of separation or distance between nodes, which can be highly informative for understanding complex graph structures. The normalized shortest path score is calculated using the shortest path matrix 𝑃, where each element 𝑃𝑖𝑗is defined as the minimum path length among all possible paths from node 𝑖to node 𝑗. The 𝑃𝑖𝑗is given by: 𝑃𝑖𝑗= min{path length|all paths from node 𝑖to node 𝑗}. Later, we introduce a damping factor 𝛿in order to adjust the influence of distant nodes. This is achieved by inverting and normalizing the path lengths in the matrix. The modified shortest path matrix 𝑅path is defined as:
(4)
() In this formulation, 𝛿is a value between 0 and 1, and max(𝑃) represents the maximum path length in the shortest path matrix 𝑃. The normalization step ensures that 𝑅path 𝑖𝑗 remains within the range of 0 to 1. 𝑅path 𝑖𝑗 captures the proximity between nodes in the graph. Shorter paths (indicating closer connections) result in higher values. Integrating the direct connections and indirect relationships between nodes into a unified representation 𝑅, the attention mechanism is empowered to model the inherent characteristics of individual nodes and their relative positions and interconnections within the overall graph.
# 3.3 Pre-training Graph Attentive LLM with Crowd Contextual Prompts Pre-training is to initially train the of our graph attentive G
# 3.3 Pre-training Graph Attentive LLM with
Pre-training is to initially train the of our graph attentive GPT-2 model on a larger corpus of recommendation text data, including data collection, tokenization, model architecture, pre-training objective, and optimization procedure.
3.3.1 Data Collection. We first collect a large textual corpus from diverse sources: user descriptions, item descriptions, user reviews for items, historical events that users interact (e.g., rate or purchase) with items, in order to ensure that the graph attentive LLM model learns robust representations of languages.
that we define the unique structure of our crow contextual prompts for pre-training a LLM recommender. User/Item Tokens. Our prompts include two unique tokens: user
embed rich semantic information about users, such as user profiles, demographics, reviews, and preferences, information propagated from items, and rich semantic information about items, such as item descriptions, item functionalities, item reviews, information propagated from users.
In this way, we aim to augment the prompt texts and enrich the contextual environment that simulates the preference, characteristics, functionality, categorization, opinion, and first-order and second-order social or network dimensions of a real recommender system in a language modality. These crowd contextual prompts are used as training data to pre-train our graph attentive LLM.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0ffc/0ffc4fbc-5b00-46b0-bcac-55391a0a61f9.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Crowd Contextual Prompts based on Recommendation Data.</div>
3.3.3 The Optimization Objective and Procedures of Pre-training. Given the prepared crowd contextual prompts as pre-training data, we will train the graph attentive GPT-2 to predict the next token in the textual sequence. The optimization objective is to maximize the
token generation likelihood of a textual sequence. The likelihood function is given by: ∑︁
(5)
where𝑇𝑖,𝑇𝑖−1, . . . ,𝑇1 is the first𝑖tokens, and Θ indicates the weights of the LLM, the objective is to predict the 𝑖+1 token𝑇𝑖+1. By solving the optimization objective, the graph attentive LLM can combine knowledge from different information sources to learn more comprehensive portraits of users and items. By integrating graph structured attention, the graph attentive LLM can model the first-order and second-order edge connectivity among users and items.
# 3.4 Fine-tuning Graph Attentive LLM with Personalized Predictive Prompts After the pre-training step, the graph attentive LLM learn
After the pre-training step, the graph attentive LLM learns the contextual knowledge of users, items, relationships of a recommender system’s world. However, the optimization objective in the pre-training step is focusing on maximizing textual generation accuracy in a language sequence, instead of recommending personalized items. Therefore, in the fine-tuning step, we develop 1) personalized predictive prompts and 2) recommendation loss functions of fine-tuning to incentivize the graph attentive LLM to shift model focuses from text generation to generating accurate personalized item recommendations.
3.4.1 Personalized Predictive Prompts. When fine-tuning the pretrained graph attentive LLM, we introduce the personalized predictive Prompt method. Our idea is to use the historical purchase events of users for items as prompts to guide the LLM to learn user preferences for items. Figure 3 shows that we convert a user’s interaction (e.g., rating, purchases) history with all items into past tense texts, combined with a future tense trigger (e.g., user 𝑖will purchase ?), to motivate the graph attentive LLM to generate item recommendations for a user.
# Figure 3: A Personalized Predictive Prompt.
3.4.2 The Optimization Objective and Training Procedure of Finetuning. Different from learning the world knowledge of a recommendation system, the fine-tuning stage is to adapt the pre-trained LLM to personalized item recommendations. During the fine-tuning process, we will integrate personalized predictive prompts into LLM so that the model can model the historical item purchase event of a user to generate a list of items 𝑅𝑖to the specific user, based on the prompt. The generative recommended items are then compared with the actual purchase records of the specific user. The resulting loss function derived from this comparison is utilized as the optimization objective of fine-tuning. In particular, we define the generative probability that measures whether LLM recommendations are statistically close to historical user purchase records,
which is given by Equation 6:
∑︁ where 𝑃𝑟𝑖is the prompt for fine-tuning stage, and 𝑅𝑖is the final recommendation list, Θ denotes the LLM weights. In summary, fine-tuning is to optimize the recommendation loss.
# 3.5 Graph Attentive LLM for Item
# Recommendations in Deployment re-training and fine-tuning, given a testing user
After pre-training and fine-tuning, given a testing user 𝑖, we convert the user’s interaction records with items into a personalized predictive prompt as the input of the graph attentive LLM. Therefore, the classic recommendation engine in production can be viewed as a wrapper, where the request is reconstructed as a prompt - the same as the prompt in the fine-tuning stage, and the graph attentive LLM will provide a recommendation score for each user-item pair. In order to reduce the serving latency in production and alleviate the peak load pressure, the proposed graph attentive LLM model can be deployed onto both offline and online GPU clusters. The offline pipeline focuses on the batch processing and calculates the relevance between a user and the candidate items to recommend, shown as LLM(𝑋𝑖𝑗; Θ), where 𝑋𝑖𝑗is the user-item interaction prompt of user 𝑖and item 𝑗, and Θ denotes the weights for the LLM backbone. The batch processing can directly be applied in offline recommendation scenarios such as promotional emails and notifications. The results can also be used for warming up online cache to minimize redundant computations. In a typical online recommendation scenario, the prompt containing user, item and the interaction information is sent to the graph attentive LLM model, and the top 𝑀items with the highest scores are selected as recommendations by comparing the probability scores against each other, ��
(7)
�� In this case, the proposed method can easily be integrated into most common recommender systems in industry. The additional pressure caused by GPU serving can be handled by offline (batch) precomputation and online caching warm-up.
# 4 EXPERIMENTAL RESULTS
We conduct empirical experiments to answer the following questions: 1) can our method generate more accurate recommendations? 2) what are the contributions of different technical components? 3) what are the contributions of second order relationship and item background information? 4) what are the impacts of different attention mechanisms? 5) parameter sensitivity and robustness.
# 4.1 Experimental Setup
4.1.1 Data Description. We used seven public recommendation datasets: Amazon (AM)-Beauty dataset, AM-Toys dataset, AMSports, AM-Luxury, AM-Scientific, AM-Instruments dataset [20]. We binarized the user-item interaction matrix by scores. If the score is greater than 3, there is a connection between a user and an item. For each user in the dataset, we randomly select 80% of interactions for training, 10% for validation, and 10% for testing, with at least one sample selected in both the validation and test sets. According
to the prompt construction method in Section 3.3, we constructed the data for pretraining. Table 1 shows The main dataset statistics.
<div style="text-align: center;">Table 1: Dataset Statistics</div>
Dataset
User
Item
Interaction
Content
AM-Beauty
10,553
6,086
94,148
165,228
AM-Toys
11,268
7,309
95,420
170,551
AM-Sports
22,686
12,301
185,718
321,887
AM-Luxury
2,382
1,047
21,911
15,834
AM-Scientific
6,875
3,484
50,985
43,164
AM-Instruments
20,307
7,917
183,964
143,113
AM-Food
95,421
32,180
834,514
691,543
4.1.2 Evaluation Metrics. We used three metrics: Recall@20, Recall@40, and NDCG@100 to evaluate algorithmic effectiveness. Recall@k [32] indicates the proportion of items that users are interested in among the top-𝑘recommended items:
(8)
NDCG@k is a position-sensitive indicator that measures the quality of recommendation lists:
(9)
where, DCG@k = �𝑘 𝑖=1 2𝑟𝑒𝑙𝑖−1 log2(𝑖+1) and IDCG@k = �|𝑅𝐸𝐿| 𝑖=1 2𝑟𝑒𝑙𝑖−1 log2(𝑖+1)
4.1.4 Hyper parameters and Settings. We conducted experiments using GPT-2 as the base model. We set the maximum input length to 1024, the token embedding dimension to 768, and the vocabulary length of natural language tokens to 50257. In the pre-training stage, we first trained 10 epochs using crowd contextual data to optimize LLM and then trained 100 rounds using user-item interaction data. In the fine-tuning stage, we used 50 epochs for recommendationoriented fine-tuning of LLM.
4.1.5 Experimental Environment. All experiments were conducted on Ubuntu 22.04.3 LTS OS, Intel(R) Core(TM) i9-13900KF CPU, with the framework of Python 3.11.5 and PyTorch 2.0.1.
# 4.2 Experimental Results
4.2.1 Overall Comparison. This experiment aims to answer: Can our model really generate more accurate recommendation results through the natural language processing method? We compared our model with several baseline models on various Amazon datasets. The baseline models used for comparison include ID-based and Attention-based methods. Our model was tested on the same dataset as these baseline models to ensure fairness and accuracy in the comparison. The experimental results are shown in Table 2, and our model performs well in seven Amazon datasets. Recall@20, Recall@40, and NDCG@100 are superior to the baseline models. This indicates that LLMs have strong capabilities in understanding text, and capturing user preferences and needs, thereby promoting the accuracy of recommendations. Overall, the experimental results support our hypothesis that our model can generate more accurate recommendations through the graph attentive LLM. This discovery is important for research and the practical application of recommendation systems.
We pre-trained and fine-tuned each baseline model separately, and then compared it with our complete model. These pre-training and fine-tuning experimental settings are consistent and conducted on the same dataset to ensure comparability of results. Table 2 shows the specific contribution of each component to the overall performance of the model. For example, the performance of LLMNoPretrain is significantly lower than that of the complete model. This implies that using recommendation related graph data and
<div style="text-align: center;">Table 2: Comparison Between Our Model and Baselines on Three Amazon Review Datasets.</div>
Dataset
Metric
Multi-VAE
MD-CVAE
BERT4Rec
𝑆3Rec
UniSRec
FDSA
SASRec
GRU4Rec
LLM-
NoPretrain
LLM-
NoFineTune
LLM-
NoGKIA
LLM-
NoGHIP
Ours
AM-Beauty
Recall@20
0.1295
0.1472
0.1126
0.1354
0.1462
0.1447
0.1546
0.0997
0.0464
0.0441
0.1225
0.1267
0.1590
Recall@40
0.1720
0.2058
0.1677
0.1789
0.1898
0.1875
0.2071
0.1528
0.0709
0.0691
0.1665
0.1799
0.2177
NDCG@100
0.0835
0.0835
0.0781
0.0867
0.0907
0.0834
0.0949
0.0749
0.0339
0.0323
0.0790
0.0827
0.1029
AM-Toys
Recall@20
0.1076
0.1291
0.0853
0.1064
0.1110
0.0972
0.0869
0.0657
0.0477
0.0580
0.0896
0.0858
0.1349
Recall@40
0.1558
0.1804
0.1375
0.1524
0.1457
0.1268
0.1146
0.0917
0.0689
0.1003
0.1272
0.1179
0.1873
NDCG@100
0.0781
0.0844
0.0532
0.0665
0.0638
0.0662
0.0525
0.0439
0.0330
0.0481
0.0612
0.0594
0.0876
AM-Sports
Recall@20
0.0659
0.0714
0.0521
0.0616
0.0714
0.0681
0.0541
0.0720
0.0449
0.0394
0.0555
0.0558
0.0764
Recall@40
0.0975
0.1180
0.0701
0.0813
0.1143
0.0866
0.0739
0.1086
0.0719
0.0613
0.0846
0.0830
0.1240
NDCG@100
0.0446
0.0514
0.0305
0.0438
0.0504
0.0475
0.0361
0.0498
0.0322
0.0278
0.0391
0.0379
0.0535
AM-Luxury
Recall@20
0.2306
0.2771
0.2076
0.2241
0.3091
0.2759
0.2550
0.2126
0.1872
0.1885
0.2474
0.2679
0.3066
Recall@40
0.2724
0.3206
0.2404
0.2672
0.3675
0.3176
0.3008
0.2522
0.2233
0.2254
0.2880
0.3028
0.3441
NDCG@100
0.1697
0.2064
0.1617
0.1542
0.2010
0.2107
0.1965
0.1623
0.1223
0.1235
0.1834
0.2065
0.2331
AM-Scientific
Recall@20
0.1069
0.1389
0.0871
0.1089
0.1492
0.1188
0.1298
0.0849
0.0708
0.0668
0.1383
0.1206
0.1480
Recall@40
0.1483
0.1842
0.1160
0.1541
0.1954
0.1547
0.1776
0.1204
0.1037
0.0960
0.1822
0.1575
0.1908
NDCG@100
0.0766
0.0872
0.0606
0.0715
0.1056
0.0846
0.0864
0.0594
0.0568
0.0465
0.0940
0.0810
0.1072
AM-Instruments
Recall@20
0.1096
0.1398
0.1183
0.1352
0.1684
0.1382
0.1483
0.1271
0.0766
0.0727
0.1387
0.1426
0.1698
Recall@40
0.1628
0.1743
0.1531
0.1767
0.2239
0.1787
0.1935
0.1660
0.1004
0.0948
0.1741
0.1779
0.2265
NDCG@100
0.0735
0.1040
0.0922
0.0894
0.1075
0.1080
0.0934
0.0998
0.0500
0.0478
0.1042
0.1044
0.1312
AM-Food
Recall@20
0.1062
0.1170
0.1036
0.1157
0.1423
0.1099
0.1171
0.1140
0.0224
0.0204
0.1275
0.1264
0.1438
Recall@40
0.1317
0.1431
0.1284
0.1456
0.1661
0.1317
0.1404
0.1389
0.0299
0.0274
0.1559
0.1487
0.1673
NDCG@100
0.0727
0.0863
0.0835
0.0926
0.1024
0.0904
0.0942
0.0910
0.0153
0.0141
0.0898
0.0963
0.1119
natural language data for pre-training plays a crucial role in improving model performance. Similarly, the results of LLM-NoFineTune demonstrate the importance of fine-tuning. Subsequently, by comparing the performance of LLM-NoGKIA, and LLM-NoGHIP with that of the complete model, we find that the addition of graph connection information in attention calculation and complex prompts containing second-order relationships is crucial for improving the performance of recommendation systems.
By comparing the performances of these models, we quantified the impact of the second-order relationships and the background information of items on recommendation accuracy. Figure 4 shows that the model that uses prompt sentences of complete information (with the second-order relationship) performs best over all the performance indicators. The performances of the "Without second-order relationship" model are lower than that of the complete model. As can be seen, second-order relationship information is an essential component of graph connectivity. Similarly,
the "Without Item" model performs poorly, highlighting the importance of natural language background information in enhancing recommendation systems.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5334/5334225b-686f-4f12-acf1-e2faaedc4745.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Results of Different Prompt Structures.</div>
4.2.4 Study on Different Attention Injection Ways. This experiment aims to answer: Is the connection information in the attention calculation process of the GPT-2 model really that important? To answer this question, we used the AM-Toys and AM-Beauty datasets. The experimental design included three different attention mechanisms: • Reasonable Injection: Injecting meaningful connection information into the attention mechanism. • Meaningless Injection: Set all connection information of the attention mechanism to 1, without considering actual connection strength or relationships. • Normal Attention: Maintain the normal attention mechanism of the GPT-2 model without any injection.
Figure 5 shows that the model using our graph attentive LLM method exhibits the best performance. Our method not only considers the direct connections between nodes but also the spatial relationships (i.e., the shortest connected path) between nodes in the graph. We compared our method with regular attention mechanisms, and the experimental results clearly support this point.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d169/d169229e-94fe-432f-954c-c1f948b395e7.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) AM-Beauty</div>
# Figure 5: Results of Different Attention Injection Ways
<div style="text-align: center;">Figure 5: Results of Different Attention Injection Ways</div>
To avoid bias that may arise from adding input only between user/project tokens, we introduced a comparison with fixed additive attention. We found that simply adding fixed connection information to attention calculation for nodes in the graph is not effective. It is truly effective to include information that reflects the actual relationships between nodes.
4.2.5 Study of Parameters. This experiment aims to answer: Can we ensure consistency between our pre-training and fine-tuning tasks? We conducted experiments on the AM-Toys dataset to analyze the performance alignment between the pre-training task and the fine-tuning task. We used the results of the first 10 pre-training epochs and the corresponding loss function. Then, we fine-tuned the pre-trained model to obtain evaluation metrics. We compared the 3 metrics, Recall@20, Recall@40, and NDCG@100 with the loss function. Figure 6 shows the trend of changes in the 3 metrics is consistent with the trend of changes in the loss functions. This indicates that our pre-training task and fine-tuning task are wellaligned, and our prompt construction method can provide rich information for subsequent recommendation tasks.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0b4a/0b4a7e6d-3724-4853-a2c2-673cf4d76405.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Results of Different Training Epochs.</div>
The application of LLMs in recommendation systems mainly contains two types: deep representation of data using LLMs, and direct application of generative LLMs to construct recommendation logic. In deep representation, discriminative language models like BERT are widely used for fine-tuning and pre-training, integrating specific domain data features to enhance the performance of recommendation systems. For instance, U-BERT [25] leverages contentrich domain data to learn user representations, compensating for the scarcity of behavioral data. Similarly, UserBERT [34] includes two self-supervised tasks for pretraining on unlabeled behavior data. Additionally, BECR [36] combines deep contextual token interactions with traditional lexical word matching features. Notably, the "pretrain-finetune" mechanism plays a crucial role in sequence or session-based recommendation systems, like BERT4Rec [30] and RESETBERT4Rec [40]. UniSRec [9] develops a BERT fine-tuning framework that links item description texts. In content-based recommendations, especially in the news domain, models like NRMS [34],
Tiny-NewsRec [38], and PREC [18] enhance news recommendations by leveraging LLMs, addressing domain transfer issues or reducing transfer costs. Research by Penha and Hauff [22] shows that BERT, even without fine-tuning, effectively prioritizes relevant items in ranking processes, illustrating the potential of large language models in natural language understanding. In recent studies, generative LLMs have shown huge potential in recommendation systems through prompting and tuning methods. Notable works and advancements include: Liu et al. [17] conducted a comprehensive assessment of ChatGPT’s performance in five key recommendation tasks. Sanner et al. [28] designed three different prompt templates to evaluate the enhancement effect of prompts, finding that zero-shot and few-shot strategies are particularly effective in preference-based recommendations using language. Sileo et al. [29] and Hou et al. [10] focused on designing effective prompt methods for specific recommendation tasks. Gao and team [6] developed ChatREC around ChatGPT, an interactive recommendation framework that understands user needs through multiple rounds of dialogue. Petrov and Macdonald [23] introduced GPTRec, a generative sequence recommendation model based on GPT-2. Kang and colleagues [13] explored formatting user historical interactions as prompts and assessed the performance of LLMs of different scales. Dai et al. [3] designed templates for various recommendation tasks using demonstration example templates. Bao et al. [1] developed TALLRec, which demonstrates the potential of LLMs in recommendation domains through two-stage fine-tuning training. Ji et al. [11] presented GenRec, a method that leverages the generative capabilities of LLMs to directly generate the target of recommendations. In specific scenarios like online recruitment, generative recommendation models such as GIRL [41] and reclm [4] demonstrated enhanced explainability and appropriateness in recommendations. Li et al. (2023e) [14] described user behaviors and designed prompts in news with PBNR. Wang et al. [22] proposed UniCRS, a design based on knowledge-enhanced rapid learning.
# 6 CONCLUSION
We tackle a key issue in recommendation systems: how to integrate LLM and graph structures into recommendations. To this end, we propose a graph attentive LLM generative recommender system. By introducing new prompting methods and graph structured attention mechanisms, we can effectively integrate the complex relationships and background information between users and items into the model. We first design a natural language prompt that can reflect the relationship between users and items and embedded the 2-order relationship between items into it. Next, we improved the attention mechanism of LLM to model complex graph structure information. Through experiments, we validate the effectiveness of our method. The experimental results show that our model has significantly improved recommendation accuracy and personalization compared to traditional recommendation systems. Considering these innovations, our approach provides a new technological path for developing more efficient and intelligent recommendation systems. Meanwhile, these methods demonstrate new perspectives and ideas in applying LLM to recommendation systems and a wider range of fields. This promotes the development of recommendation systems and provides strong support and inspiration for using LLM in various complex application scenarios.
# REFERENCES
