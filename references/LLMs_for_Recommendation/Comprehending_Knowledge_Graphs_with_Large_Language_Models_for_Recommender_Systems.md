# Comprehending Knowledge Graphs with Large Language Models for Recommender Systems
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9507/95078487-59bc-4795-a0af-cdeadcf97f7d.png" style="width: 50%;"></div>
# Abstract
In recent years, the introduction of knowledge graphs (KGs) has significantly advanced recommender systems by facilitating the discovery of potential associations between items. However, existing methods still face several limitations. First, most KGs suffer from missing facts or limited scopes. This can lead to biased knowledge representations, thereby constraining the model’s performance. Second, existing methods cannot effectively utilize the semantic information of textual entities as they typically convert textual information into IDs, resulting in the loss of natural semantic connections between different items. Third, existing methods struggle to capture high-order relationships in global KGs due to their inefficient layer-by-layer information propagation mechanisms, which are prone to introducing significant noise. To address these limitations, we propose a novel method called CoLaKG, which leverages large language models (LLMs) for knowledge-aware recommendation. The extensive world knowledge and remarkable reasoning capabilities of LLMs enable them to supplement KGs. Additionally, the strong text comprehension abilities of LLMs allow for a better understanding of semantic information. Based on this, we first extract subgraphs centered on each item from the KG and convert them into textual inputs for the LLM. The LLM then outputs its comprehension of these item-centered subgraphs, which are subsequently transformed into semantic embeddings. Furthermore, to utilize the global information of the KG, we construct an item-item graph using these semantic embeddings, which can directly capture higher-order associations between items. Both the semantic embeddings and the structural information from the item-item graph are effectively integrated into the recommendation model through our designed representation alignment and neighbor augmentation modules. Extensive experiments on four real-world datasets demonstrate the superiority of our method.
∗Work done as an intern in FiT, Tencent †Corresponding author.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4328/4328507b-46f6-43aa-8eb0-e08a497845f3.png" style="width: 50%;"></div>
# Figure 1: An illustrative diagram demonstrating the potential issues of existing KG-based recommendation methods.
<div style="text-align: center;">Figure 1: An illustrative diagram demonstrating the potential issues of existing KG-based recommendation methods.</div>
Keywords Knowledge Graphs, Large Language Models, Recommendation
Keywords Knowledge Graphs, Large Language Models, Recommendation
Knowledge Graphs, Large Language Models, Recommendation
# 1 Introduction
The rapid advancement of web applications has led to an increasingly critical issue of information overload. Recommender systems address this problem by modeling user preferences based on historical data and providing personalized recommendations. Collaborative filtering (CF) [9, 10, 22], as one of the most classic and efficient methods, has been extensively employed in existing recommender systems. However, CF-based methods exclusively rely on user-item collaborative signals, often suffering from the data sparsity issue for users who have not interacted with a sufficient number of items [25]. To address such data sparsity issue, recent studies [32, 41, 43] have incorporated knowledge graphs (KGs) as external knowledge sources into recommendation models, achieving significant progress. Typically, these methods capture diverse and higher-order semantic relationships between items by
modeling the structural and semantic information in KGs, thereby generating enhanced user and item representations to improve the recommendation model [41]. Despite the effectiveness of existing KG-enhanced recommendation methods, they still have several limitations. First, many KGs suffer from missing facts and limited scopes [8], as constructing KGs often requires significant manual effort and domain expertise. The absence of key attributes, such as the genres of a movie, can cause items that originally share the same attribute to lose their semantic connections. As illustrated in Figure 1, item A and item B should have a connecting path (A-P-B) in the KG. However, due to item A missing the P attribute, they are not associated with each other. In this situation, the recommendation model can only learn from biased knowledge, leading to suboptimal performance. Second, textual entities and relations are not effectively utilized. Existing methods [31–33, 40, 41] typically convert textual entities and relations into IDs, failing to leverage the semantic information inherent in the text. Moreover, this can result in the loss of natural semantic connections between different items. For instance, in Figure 1, “horror” and “thriller” are two semantically related attribute nodes of Item F and Item G, respectively. However, similar semantics are not reflected in different entity IDs, which further results in the disconnection between item F and item G. Third, existing methods [24, 31–33, 40, 41] struggle to capture high-order relationships in global KGs. Most of them propagate and aggregate information by stacking multiple layers of graph neural networks (GNNs). The layer-by-layer propagation is not only inefficient but also accumulates a large amount of irrelevant node information, leading to the over-smoothing issue [8, 31]. This problem becomes more severe as the order increases. For instance, let us assume that points A and H in Figure 1 have a strong semantic connection. However, the considerable distance between them in the KG presents big challenges for existing KG-based recommendation methods in capturing this semantic relation. Due to these limitations, recommender systems are unable to effectively capture the semantic relationships between target items and users’ historically interacted items through the KG, resulting in suboptimal predictions, as illustrated in the lower left corner of Figure 1. Empowered by extensive knowledge and remarkable reasoning abilities, large language models (LLMs) have demonstrated significant promise in semantic understanding and knowledge extraction. Consequently, LLMs have the potential to address the aforementioned issues. Recently, efforts have been made to leverage LLMs to improve recommendation models. Some studies utilize LLMs to supplement missing attributes of items and generate semantic representations of item profiles [21]. Other studies employ LLMs to determine whether a complementary relationship exists between two items, thereby recommending complementary products based on users’ historical behaviors [44]. However, these methods do not fully exploit the semantic and structural information of KGs. As one of the most common and important sources of knowledge, KGs contain a wealth of semantic associations among entities and relations, which are often overlooked by existing methods that typically consider only item profiles. Additionally, KGs serve as task-relevant knowledge repositories, effectively aiding LLMs in acquiring task-specific knowledge and mitigating the issue of hallucinations caused by excessive divergence. Nevertheless, effectively
everaging LLMs to model the diverse semantic relationships in KGs and enhance recommendation performance remains an open uestion. To bridge this gap, in this paper, we propose a novel method amed Comprehending Knowledge Graphs with Large Language Models for Recommendation (CoLaKG). The core idea is to leverage LMs for understanding the semantic and structural information f KGs to enhance the representation learning of users and items. Our method comprises two stages: • Comprehending KGs with LLMs. As LLMs cannot directly process graph-structured data, we first transform the KG into the text format. Given the impracticality and redundancy of inputting the entire KG into LLMs, we propose extracting subgraphs corresponding to each item and converting these item-centered subgraphs into text. Next, we carefully design prompts to leverage the extensive knowledge and reasoning capabilities of LLMs to fully understand, complete, and refine the local KG, thereby generating a comprehensive understanding of these subgraphs. Then, a pre-trained text embedding model is used to obtain semantic embeddings of these generated texts. In addition to local KG information, we construct an item-item graph based on these semantic embeddings, where the relationship between two items corresponds to their semantic similarity. This approach effectively leverages global KG information, facilitating the capture of high-order semantic associations between items. • Incorporating semantic embeddings into the recommendation model. Our objective is to integrate semantic embeddings with the ID embeddings of the recommendation model, thereby leveraging both collaborative signals and semantic information from the KG. Since semantic and ID embeddings originate from different modalities and typically have different dimensions, we design an adapter to map the semantic embeddings to align with the item ID embedding space and then employ a simple method to fuse the embeddings from both modalities. Additionally, to capture high-order semantic associations of items within the KG, we enhance the representations by aggregating the representations of semantically related items. Finally, the learned representations are used for prediction. t is important to note that these two stages are decoupled, meanng that our model does not involve LLM inference during the ecommendation process. This decoupling allows our model to be ffectively applied in real-world recommendation scenarios. Our contributions are summarized as follows: • We propose a novel method that utilizes LLMs to comprehend and transform the semantic and structural information of KGs. This approach addresses the issues of missing facts and the inability to leverage semantic information from text in current KG-based recommendation methods. • We construct an item-item graph based on the semantic relationships of items and introduce a KG semantic-based neighbor augmentation method to enhance item representations. This approach effectively captures higher-order relations in the KG and leverages global KG information for better recommendations. • Extensive experiments are conducted on four real-world datasets to validate the superiority of our method. Further analysis demonstrates the rationale behind our approach.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/34ea/34ea764a-0f73-473c-b78a-6972936d5268.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: The framework of our proposed CoLaKG for knowledge-aware recommendation.</div>
# 2 Related Work
# 2.1 Knowledge-aware Recommendation
Existing knowledge-aware recommendation methods can be categorized into three types [8]: embedding-based methods, pathbased methods, and GNN-based methods. Embedding-based methods [3, 28, 43] enhance the representations of users and items by leveraging the relations and entities within the KGs. Notable examples include CKE [43], which integrates various types of side information into a collaborative filtering framework using TransR [19] for structural knowledge. Another example is DKN [28], which improves news representations by combining textual embeddings of sentences and knowledge-level embeddings of entities. Pathbased methods leverage KGs to explore long-range connectivity [13, 34, 42]. For example, Personalized Entity Recommendation (PER) [42] treats a Knowledge Graph (KG) as a heterogeneous information network and extracts meta-path-based latent features to represent the connectivity between users and items along various types of relational paths. MCRec [13] constructs meta-paths and learns the explicit representations of meta-paths to depict the interaction context of user-item pairs. Despite their effectiveness, these approaches heavily rely on domain knowledge and human effort for meta-path design. Recently, GNN-based methods have been proposed, which enhance entity and relation representations by aggregating embeddings from multi-hop neighbors [29, 32, 33]. For instance, KGAT [32] employs graph attention mechanisms to propagate embeddings and utilizes multi-layer perceptrons to generate final recommendation scores in an end-to-end manner. Similarly, KGIN [33] adopts an adaptive aggregation method to capture finegrained user intentions. Additionally, some methods [31, 40, 41, 47]
employ contrastive learning to mitigate potential knowledge noise and identify informative knowledge connections.
# 2.2 LLMs for Recommendation
In light of the emergence of large language models and their remarkable achievements in the field of NLP, scholars have begun to explore the potential application of LLMs in recommender systems [4, 6, 37, 45]. Due to the powerful reasoning capabilities and extensive world knowledge of LLMs, they have been already naturally applied to zero-shot [11, 12, 30] and few-shot recommendation scenarios [2, 18]. In these studies, LLMs are directly used as a recommendation model [17, 46], where the output of LLMs is expected to offer a reasonable recommendation result [38]. However, when the dataset is sufficiently large, their performance often falls short of that achieved by traditional recommendation models. Another line of research involves leveraging LLMs as feature extractors. These methods [1, 14, 15, 21, 21, 35, 36, 48] generate intermediate decision results or semantic embeddings of users and items, which are then input into traditional recommendation models to produce the final recommendations. Unlike existing methods, our approach aims to leverage the extensive knowledge base and reasoning capabilities of LLMs to understand KGs and transform them into semantic embeddings, thereby addressing existing issues in KG-based recommender systems and enhancing recommendation performance.
# 3 Preliminaries
User-Item Interaction Graph. Let U and V denote the user set and item set, respectively, in a recommender system. We construct a user-item bipartite graph G = {(𝑢,𝑦𝑢𝑣, 𝑣)|𝑢∈U, 𝑣∈V} to
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e043/e0432dad-3f20-481d-9aaa-a894e42fe18e.png" style="width: 50%;"></div>
Second-order info: The movies with the same director (James Cameron) also include: Aliens, …   The movies with the same actor (Leonardo DiCaprio) also include: Inception, …   First-order info:  (Titanic, Directed by, James Cameron), (Titanic, Acted by, Leonardo DiCaprio), … Assume you are an expert in movie recommendation. You will be given a certain movie with its  first-order information (in the form of triples) and some second-order relationships (movies related  to this movie). Please complete the missing knowledge, summarize the movie and analyze what  kind of users would like it. …
# Figure 3: An example of the prompt for KG comprehension.
represent the collaborative signals between users and items. Here, 𝑦𝑢𝑣= 1 if user 𝑢interacted with item 𝑣, and vice versa. Knowledge Graph. We capture real-world knowledge about items using a heterogeneous graph composed of triplets, represented as G𝑘= {(ℎ,𝑟,𝑡)}. In this context, ℎand 𝑡are knowledge entities belonging to the set E, while 𝑟is a semantic relation from the set R that links them, as exemplified by the triplet (James Cameron, directed, Titanic). Notably, the item set is a subset of the entity set, denoted as V ⊂E. This form of knowledge graph enables us to model the intricate relationships between items and entities. Task Formulation. Following the task format of most KG-aware recommendation models, we formulate the task as follows: Given the user-item interaction graph G and the corresponding knowledge graph G𝑘, our objective is to learn a recommendation model that predicts the probability of user 𝑢interacting with item 𝑣.
# 4 Methodology
In this section, we introduce our proposed method CoLaKG in detail. An overview of our method is illustrated in Figure 2. For each item, we extract a subgraph centered on it from the KG. The LLM then comprehends this subgraph and converts it into a semantic embedding, thereby fully utilizing the local information of the KG. Additionally, we enhance the representation of each item by aggregating the most semantically similar neighbors based on the semantic similarity between items, leveraging the global information of the KG. Furthermore, we generate semantic embeddings for users based on their preferences derived from the KG. Finally, these semantic embeddings are integrated with the ID representations of the recommendation model, resulting in enhanced representations of both items and users, thereby improving the performance of the recommendation model.
# 1 KG Comprehension with LLMs
KGs have been widely utilized in recommender systems to provide semantic information and model latent associations between items. However, KGs are predominantly manually curated, leading to missing facts and limited knowledge scopes. Additionally, the highly structured nature of KGs poses challenges for effectively utilizing textual information. Existing methods often transform textual entities and relations into IDs, resulting in a significant loss of semantic information. Recently, the rapid emergence of LLMs has provided a promising approach to addressing these issues. Their extensive knowledge and reasoning abilities enable them to complete missing entities and expand the knowledge scopes of KGs. Furthermore, their inherent text comprehension abilities facilitate the effective utilization of textual entities. Leveraging these advantages, we propose the use of LLMs to enhance the understanding and refinement of KGs for improved recommendations.
4.1.1 Item-Centered KG Subgraph Comprehension. Equipping LLMs to comprehend KGs presents certain challenges, as LLMs cannot directly interpret non-textual graph data. Consequently, KGs must be converted into a textual format. However, due to the vast number of entities in a KG, inputting the entire KG into an LLM is impractical. To address this, we initially extract the subgraph of the KG centered on each item. This approach enables the effective utilization of local KG information for each item. Note that we also consider the utilization of global KG information, which will be discussed in Section 4.1.2. First, we represent the first-order KG subgraph centered on each item (i.e., ego network [20]) using triples. Specifically, given an item 𝑣∈V, we use T𝑣= {(𝑣,𝑟,𝑒)|(𝑣,𝑟,𝑒) ∈G𝑘} to denote the set of triplets where𝑣is the head entity. In the context of recommendation, the first-order neighboring entities of an item in a KG are usually attributes. Therefore, we use 𝑒to represent these attribute entities to distinguish them from those item entities 𝑣. During generating triples, in cases where the attribute or relation is absent, the term “missing” is employed as a placeholder. Next, we consider the secondorder relations in KGs. The number of entities in an ego network centered on a single entity increases exponentially with the growth of the radius. However, the input length of an LLM is strictly limited. Consequently, including all second-order neighbors associated with the central item in the prompt becomes impractical. To address this issue, we adopt a simple but effective strategy, random sampling, to explore second-order connections of 𝑣. Let E𝑣= {𝑒| (𝑣,𝑟,𝑒) ∈T𝑣} denote the set of first-order connected neighbors of 𝑣. For each 𝑒∈E𝑣, we randomly sample 𝑚triples from the set T𝑒to construct the triples of second-order connections, denoted as T𝑚 𝑒. Here, T𝑒= {(𝑒,𝑟, 𝑣′) | (𝑒,𝑟, 𝑣′) ∈G𝑘, 𝑣′ ≠𝑣} represents the set of triples where 𝑒∈E𝑣is the head entity. We fix 𝑚to 10 in our paper and do not perform hyperparameter exploration due to cost considerations associated with the LLM. After converting first-order and second-order relationships into triples, we transform these triples into textual form. For first-order relations, we concatenate all the first-order triples in T𝑣to form a single text, denoted as D𝑣. For second-order relations, we use a template to transform the second-order triples T𝑚 𝑒 into coherent sentences D′𝑣, facilitating the understanding of the LLM. In addition to D𝑣and D′𝑣, we have carefully designed a system prompt I𝑣as the instruction to guide the generation. By combining I𝑣, D𝑣, and D′𝑣, we obtain the prompt, which is shown in Figure 3. The prompt enables the LLM to fully understand, complete, and refine the KG, thereby generating the final comprehension for the 𝑣-centered KG subgraph. This process can be formulated as follows:
(1)
Once we have obtained the LLM’s comprehension of the KG subgraphs, we need to convert these textual answers into continuous vectors for utilization in downstream recommendation models. Here, we employ a pre-trained text embedding model P to transform C𝑣into embedding vectors s𝑣, which can be formulated as:
(2)
4.1.2 Semantic-Relational Item-Item Graph Construction. This section introduces the utilization of global KG information. Items that are distant in the KG can still have close semantic associations.
However, existing KG-based recommendation methods propagate information by stacking multiple GNN layers. Due to the low propagation efficiency of this approach and the introduction of irrelevant neighbor noise, most methods reach saturation with relatively few layers. To address this issue, we take a novel perspective by utilizing the generated semantic embeddings of KG subgraphs to directly and efficiently model semantic relations between items from the global KG. For each item, we have obtained the semantic embedding corresponding to its local KG graph. Based on this, we can directly compute the semantic relationships between any two items. Specifically, we employ the cosine similarity as the metric to quantify the relations between items. Given two different items 𝑣𝑖 and 𝑣𝑗, their semantic relation 𝑟(𝑣𝑖,𝑣𝑗) is computed as:
where sim denotes the cosine similarity function. Once we obtain the semantic associations between any two items in the entire KG, we treat the semantic similarity between the two items as the edge weight between them, allowing us to construct an item-item graph:
(4)
From a high-level perspective, we transform higher-order associations between items in the KG into semantic relationships on the constructed item-item graph G𝑣. Based on this foundation, it is essential to identify items that are semantically strongly related to the given item, as items with lower semantic relevance may introduce noise. Specifically, given an item 𝑣𝑖, we rank all other items 𝑣𝑗∈V where 𝑣𝑗≠𝑣𝑖in descending order based on the semantic similarity 𝑟(𝑣𝑖,𝑣𝑗). Subsequently, we select the top-𝑘items with the highest similarity scores, forming the neighbor set of 𝑣𝑖: N𝑘(𝑣𝑖), where 0 < 𝑘< |V| is an adjustable hyperparameter, representing the number of selected neighbors. In this manner, we explicitly filter out items with low semantic associations to the current item while retaining those with relatively strong associations. Traditional KG-based recommendation methods aggregate related items through layer-by-layer information propagation on the KG. Items with the same attributes are 2-hop neighbors in the KG, requiring at least two layers of GNN to capture their relation. Higher-order item associations necessitate even more propagation and aggregation steps to be captured. In contrast, our method, by constructing an item-item graph based on KG subgraph semantic embeddings, can recall strongly semantically related neighbors of any order across the entire graph through a single semantic similarity calculation. The purpose of identifying semantic-related neighbors is to leverage them to enhance the item representations, thereby improving the effectiveness of the recommendation model. We design a fine-grained approach to enhance the representation of item 𝑣𝑖with its neighbors N𝑘(𝑣𝑖). Details on this approach will be covered in Section 4.3.2.
# 4.2 User Preference Comprehension
The introduction of KGs allows for the expansion of user-item bipartite graphs and enables us to understand user preferences from a knowledge-driven perspective. Given a user 𝑢, we first extract the subgraph corresponding to user 𝑢from the user-item bipartite graph, denoted as B𝑢. For each item 𝑣∈B𝑢, we extract its firstorder KG subgraph and represent it as a set of triples, denoted
as T𝑣. We then concatenate all triples in T𝑣to form a single text, denoted as D𝑣. The detailed approach is the same as described in Section 4.1.1. Subsequently, we represent user 𝑢with all items the user has interacted with in the training set and the corresponding knowledge triples D𝑣:
(5)
where ⊕denotes concatenation operation, and name𝑣denotes the text name of item 𝑣. Additionally, we have meticulously designed a system prompt, denoted as I𝑢, to serve as an instruction for guiding the generation of user preferences. By combining D𝑢and I𝑢, we enable the LLM to comprehend the user preference for 𝑢, which can be formulated as:
(6)
Furthermore, we also utilize the text embedding function P to transform the textual answers C𝑢into embedding vectors s𝑢, which can be expressed as:
(7)
# 4.3 Representation Alignment and Neighbor Augmentation
4.3.1 Cross-Modal Representation Alignment. In a traditional recommendation model, each item and user is associated with an ID embedding. Let e𝑣∈R𝑑represent the ID embedding of item 𝑣 and e𝑢∈R𝑑represent the ID embedding of user 𝑢. In addition to these ID embeddings, we also obtain the semantic embedding s𝑣∈R𝑑𝑠w.r.t. the comprehension of 𝑣-centric KG subgraph, and the semantic embedding s𝑢∈R𝑑𝑠w.r.t. the comprehension of user 𝑢’s preference. Since ID embeddings and semantic embeddings belong to two different modalities and typically possess different embedding dimensions, we employ an adapter network to align the semantic embeddings with the ID embedding space. Specifically, the adapter networks consist of a linear map and a non-linear activation function, which are formulated as:
(8)
where both W1 ∈R𝑑×𝑑𝑠and W2 ∈R𝑑×𝑑𝑠are are weight ma 𝜎represents the non-linear activation function ELU [5].
where both W1 ∈R𝑑×𝑑𝑠and W2 ∈R𝑑×𝑑𝑠are are weight matrices, 𝜎represents the non-linear activation function ELU [5]. Note that during the training process, we fix s𝑣and s𝑢, training solely the corresponding projection parameters W1 and W2, and the parameters of the recommendation model. The benefits of this method are two-fold. Firstly, by preserving s𝑣and s𝑢, we can utilize the rich semantic information they already contain, which can guide the recommendation model to converge more effectively during the initial stage of training. Secondly, the number of parameters in s𝑣 and s𝑢is typically much greater than those in the recommendation model’s ID embeddings due to their large dimensions. Consequently, altering these parameters would significantly affect the updates to the ID embeddings and slow down the gradient update process, leading to an unstable training procedure. After mapping the representations to the same space, we need to fuse the representations of the two modalities, leveraging both the collaborative signals and the semantic information to form a complementary representation. To achieve this, we employ a straightforward mean pooling technique to fuse their embeddings,
thereby integrating the them into a unified representation:
(9)
    where h𝑣∈R𝑑and h𝑢∈R𝑑represent the merged embeddings of item 𝑣and user 𝑢, respectively.
4.3.2 Item Representation Augmentation with Semantic-related Neighbors. For each item, we have obtained its semantic-related items from the constructed item-item graph in Section 4.1.2. To fully utilize these neighbors, we propose to aggregate their information to enhance the representations. Considering the varying contributions of different neighbors to the central item, we employ the attention mechanism for weighted aggregation of representations. Specifically, for item 𝑣𝑖and its top-𝑘neighbor set N𝑘(𝑣𝑖), we first compute attention coefficients that indicate the importance of item 𝑣𝑗∈N𝑘(𝑣𝑖) to item 𝑣𝑖as follows:
(10)
Here, W ∈R𝑑𝑎×𝑑is a learnable weight matrix to capture higherlevel features of s𝑣𝑖and s𝑣𝑗, ∥is the concatenation operation, 𝑎 denotes the attention function: R𝑑𝑎× R𝑑𝑎→R, where we adopt a single-layer neural network and apply the LeakyReLU activation function following [26]. Note that the computation of attention weights is exclusively dependent on the semantic representation of items, as our objective is to calculate the semantic associations between items, rather than the associations present in collaborative signals. In addition, we employ the softmax function for easy comparison of coefficients across different items:
(11)
The attention scores 𝛼𝑖𝑗are then utilized to compute a linear combination of the corresponding neighbor embeddings. Finally, the weighted average of neighbor embeddings and the embedding of item 𝑣𝑖itself are combined to form the final output representation for item 𝑣𝑖: � � ∑︁ ��
(12)
� ∑︁ � where 𝜎denotes the non-linear activation function.
# 4.4 User-Item Modeling
Having successfully integrated the semantic information from the KG into both user and item representations, we can use them as inputs for traditional recommendation models to generate prediction results. This process can be formulated as follows:
(13)
where ˆ𝑦𝑢𝑣is the predicted probability of user 𝑢interacting with item 𝑣, h𝑢is the representation for user 𝑢, h𝑣is the augmented representation for item 𝑣, and F denotes the function of the recommendation model. Specifically, we select the classic model, LightGCN [10], as the architecture for our recommendation method due to its simplicity and effectiveness. The trainable parameters of original LightGCN are only the embeddings of users and items, similar to standard
matrix factorization. First, we adopt the simple weighted sum aggregator to learn the user-item interaction graph, which is defined as:
(14)
√︁ √︁ where h(𝑙) 𝑢 and h(𝑙) 𝑣 represent the embeddings of user 𝑢and item 𝑣 after 𝑙layers of propagation, respectively. The initial embeddings h(0) 𝑢 = h𝑢and h(0) 𝑣 = h′𝑣are obtained in Section 4.3. M𝑢denotes the set of items with which user𝑢has interacted, while M𝑣signifies the set of users who have interacted with item 𝑣. The symmetric normalization term is given by 1/ √︁ |M𝑢||M𝑣|. Subsequently, the embeddings acquired at each layer are combined to construct the final representation:
(15)
where 𝐿represents the number of hidden layers. Ultimately, the model prediction is determined by the inner product of the final user and item representations:
(16)
# 4.5 Model Training
Our approach can be divided into two stages. In the first stage, we employ the LLM to comprehend the KGs, generating corresponding semantic embeddings for each item and user, denoted as s𝑣and s𝑢, respectively. In the second stage, these semantic embeddings are integrated into the recommendation model through an adapter network to enhance its performance. Only the second stage necessitates supervised training, where we adopt the widely-used Bayesian Personalized Ranking (BPR) loss: ∑︁
(17)
Here, O = {(𝑢, 𝑣+, 𝑣−)|(𝑢, 𝑣+) ∈R+, (𝑢, 𝑣−) ∈R−} represents the training set, R+ denotes the observed (positive) interactions between user 𝑢and item 𝑣, while R−indicates the sampled unobserved (negative) interaction set. 𝜎(·) is the sigmoid function. 𝜆∥Θ∥2 2 is the regularization term, where 𝜆serves as the weight coefficient and Θ constitutes the model parameter set.
# 5 Experiments
# 5 Experiments 5.1 Experimental Settings
# 5.1 Experimental Settings
5.1.1 Datasets. We conducted experiments on four real-world datasets, including three public datasets (MovieLens1, MIND2, LastFM3), and one industrial dataset (Fund). The statistics for these datasets are presented in Table 1. These datasets cover a wide range of application scenarios. Specifically, MovieLens is a wellestablished benchmark that collects movie ratings provided by users. MIND is a large-scale news recommendation dataset constructed from user click logs on Microsoft News. Last-FM is a well-known
1https://grouplens.org/datasets/movielens/ 2https://msnews.github.io/ 3https://grouplens.org/datasets/hetrec-2011/
<div style="text-align: center;">Table 1: Dataset statistics.</div>
Table 1: Dataset statistics.
Statistics
MovieLens
Last-FM
MIND
Funds
# Users
6,040
1,859
44,603
209,999
# Items
3,260
2,813
15,174
5,701
# Interactions
998,539
86,608
1,285,064
1,225,318
Knowledge Graph
# Entities
12,068
9,614
32,810
8,111
# Relations
12
2
14
12
# Triples
62,958
118,500
307,140
65,697
music recommendation dataset that includes user listening history and artist tags. The Fund dataset is sampled from the data of a large-scale online financial platform aiming to recommend funds for users. We adopt the similar setting as numerous previous studies [10, 39], filtering out items and users with fewer than five interaction records. For each dataset, we randomly select 80% of each user’s historical interactions to form the training set, while the remaining 20% constitute the test set, following [10]. From the training set, we further randomly select 10% of the interactions to create a validation set for tuning hyperparameters. Each observed user-item interaction is considered a positive instance, and we apply a negative sampling strategy by pairing it with one negative item that the user has not interacted with.
5.1.2 Evaluation Metrics. To evaluate the performance of the models, we employ widely recognized evaluation metrics: Recall and Normalized Discounted Cumulative Gain (NDCG), and report values of Recall@k and NDCG@k for k=10 and 20, following [10, 32]. To ensure unbiased evaluation, we adopt the all-ranking protocol. All items that are not interacted by a user are the candidates.
5.1.3 Baseline Methods. To ensure a comprehensive assessment, we compare our method with ten baseline methods, which can be divided into three categories: classical methods (BPR-MF, NFM, LightGCN), KG-enhanced methods (CKE, RippleNet, KGAT, KGIN, KGCL, KGRec), and LLM-based methods (RLMRec). BPR-MF [22] employs matrix factorization to model users and items, and uses the pairwise Bayesian Personalized Ranking (BPR) loss to optimize the model. NFM [9] is an advanced factorization model that subsumes FM [23] under neural networks. LightGCN [10] facilitates message propagation between users and items by simplifying GCN [16]. CKE [43] is an embedding-based method that uses TransR to guide entity representation in KGs to enhance recommendation performance. RippleNet [27] automatically discovers users’ hierarchical interests by iteratively propagating users’ preferences in the KG. KGAT [32] designs an attentive message passing scheme over the knowledge-aware collaborative graph for node embedding fusion. KGIN [33] adopts an adaptive aggregation method to capture finegrained user intentions.
5.1.3 Baseline Methods. To ensure a comprehensive assessment, we compare our method with ten baseline methods, which can be divided into three categories: classical methods (BPR-MF, NFM, LightGCN), KG-enhanced methods (CKE, RippleNet, KGAT, KGIN, KGCL, KGRec), and LLM-based methods (RLMRec). BPR-MF [22] employs matrix factorization to model users and items, and uses the pairwise Bayesian Personalized Ranking (BPR) loss to optimize the model. NFM [9] is an advanced factorization model that subsumes FM [23] under neural networks. LightGCN [10] facilitates message propagation between users and items by simplifying GCN [16]. CKE [43] is an embedding-based method that uses TransR to guide entity representation in KGs to enhance recommendation performance. RippleNet [27] automatically discovers users’ hierarchical interests by iteratively propagating users’ preferences in the KG. KGAT [32] designs an attentive message passing scheme over the knowledge-aware collaborative graph for node embedding fusion. KGIN [33] adopts an adaptive aggregation method to capture finegrained user intentions.
KGCL [41] uses contrastive learning for knowledge graphs to reduce potential noise and guide user preference learning. KGRec [40] is a state-of-the-art KG-based recommendation model which devises a self-supervised rationalization method to identify informative knowledge connections. RLMRec [21] is a state-of-the-art LLM-based model. It directly utilizes LLMs to generate text profiles and combine them with recommendation models through contrastive learning. Since their method is model-agnostic, to ensure a fair comparison, we chose LightGCN as its backbone model, consistent with our method.
5.1.4 Implementation Details. We implement all baseline methods according to their released code. The embedding size 𝑑for all recommendation methods is set to 64 for a fair comparison. All experiments are conducted with a single V100 GPU. We set the batch size to 1024 for the Last-FM dataset and 4096 for the other datasets to expedite training. The Dropout rate is chosen from the set {0.2, 0.4, 0.6, 0.8} for both the embedding layer and the hidden layers. We employ the Adam optimizer with a learning rate of 0.001. The maximum number of epochs is set to 2000. The number of hidden layers for the recommendation model 𝐿is set to 3. For the LLM, we select DeepSeek-V2, a robust large language model that demonstrates exceptional performance on both standard benchmarks and open-ended generation evaluations. For more detailed information about DeepSeek, please refer to their official website4. Specifically, we utilize DeepSeek-V2 by invoking its API5. To reduce text randomness of the LLM, we set the temperature 𝜏to 0 and the top-𝑝to 0.001. In addition, We fix the sampled number 𝑚to 10 and do not perform hyperparameter exploration due to cost considerations. For the text embedding model P, we use the pre-trained sup-simcse-roberta-large6 [7]. We use identical settings for the baselines that also involve LLMs and text embeddings to ensure fairness in comparison.
# 5.2 Comparison Results
We compare 10 baseline methods across four datasets and run each experiment five times. The average results are reported in Table 2. Based on the results, we make the following observations: • Our method consistently outperforms all the baseline models across all four datasets. The performance ceiling of traditional methods (BPR-MF, NFM, LightGCN) is generally lower than that of KG-based methods, as the former rely solely on collaborative signals without incorporating semantic knowledge. However, some KG-based methods do not perform as well as LightGCN, indicating that effectively leveraging KG is a challenging task. • Among the KG-based baselines, KGCL and KGRec stand out the most. Both models incorporate self-supervised learning on top of general KG-based recommendation frameworks. During training, they jointly optimize the recommendation task and KG-based self-supervised tasks. However, they face challenges such as missing facts and difficulty in understanding semantic information. Additionally, they are unable to model higher-order associations of items within the KG. In contrast, our method does
4https://github.com/deepseek-ai/DeepSeek-V2 5https://api-docs.deepseek.com/ 6https://huggingface.co/princeton-nlp/sup-simcse-roberta-large
<div style="text-align: center;">Table 2: Performance comparison of different methods, where R denotes Recall and N denotes NDCG. The best results ar bolded, and the second best results are underlined. Our improvement is statistically significant with a significance level of 0.01</div>
Model
MovieLens
Last-FM
MIND
Funds
R@10
N@10
R@20
N@20
R@10
N@10
R@20
N@20
R@10
N@10
R@20
N@20
R@10
N@10
R@20
N@20
BPR-MF
0.1257
0.3100
0.2048
0.3062
0.1307
0.1352
0.1971
0.1685
0.0315
0.0238
0.0537
0.0310
0.4514
0.3402
0.5806
0.3809
NFM
0.1346
0.3558
0.2129
0.3379
0.2246
0.2327
0.3273
0.2830
0.0495
0.0356
0.0802
0.0458
0.4388
0.3187
0.5756
0.3651
LightGCN 0.1598
0.3901
0.2512
0.3769
0.2589
0.2799
0.3642
0.3321
0.0624
0.0492
0.0998
0.0609
0.4992
0.3778
0.6353
0.4204
CKE
0.1524
0.3783
0.2373
0.3609
0.2342
0.2545
0.3266
0.3001
0.0526
0.0417
0.0822
0.0510
0.4926
0.3702
0.6294
0.4130
RippleNet
0.1415
0.3669
0.2201
0.3423
0.2267
0.2341
0.3248
0.2861
0.0472
0.0364
0.0785
0.0451
0.4764
0.3591
0.6124
0.4003
KGAT
0.1536
0.3782
0.2451
0.3661
0.2470
0.2595
0.3433
0.3075
0.0594
0.0456
0.0955
0.0571
0.5037
0.3751
0.6418
0.4182
KGIN
0.1631
0.3959
0.2562
0.3831
0.2562
0.2742
0.3611
0.3215
0.0640
0.0518
0.1022
0.0639
0.5079
0.3857
0.6428
0.4259
KGCL
0.1554
0.3797
0.2465
0.3677
0.2599
0.2763
0.3652
0.3284
0.0671
0.0543
0.1059
0.0670
0.5071
0.3877
0.6355
0.4273
KGRec
0.1640
0.3968
0.2571
0.3842
0.2571
0.2748
0.3617
0.3251
0.0627
0.0506
0.1003
0.0625
0.5104
0.3913
0.6467
0.4304
RLMRec
0.1613
0.3920
0.2524
0.3787
0.2597
0.2812
0.3651
0.3335
0.0619
0.0486
0.0990
0.0602
0.4988
0.3784
0.6351
0.4210
CoLaKG
0.1699 0.4130 0.2642 0.3974 0.2738 0.2948 0.3803 0.3471 0.0698 0.0562 0.1087 0.0684 0.5273 0.4012 0.6524 0.4392
<div style="text-align: center;">Table 3: Ablation study on all four datasets.</div>
Metric
w/o s𝑣
w/o s𝑢
w/o N𝑘(𝑣)
w/o D′𝑣
CoLaKG
ML
R@20
0.2553
0.2613
0.2603
0.2628
0.2642
N@20
0.3811
0.3948
0.3902
0.3960
0.3974
Last-FM
R@20
0.3628
0.3785
0.3725
0.3789
0.3803
N@20
0.3278
0.3465
0.3403
0.3459
0.3471
MIND
R@20
0.1043
0.1048
0.1064
0.1076
0.1087
N@20
0.0640
0.0658
0.0662
0.0671
0.0684
Funds
R@20
0.6382
0.6481
0.6455
0.6499
0.6524
N@20
0.4247
0.4351
0.4305
0.4378
0.4392
not require the introduction of self-supervised tasks. Instead, we leverage LLMs to address these existing challenges, resulting in significant improvements across all datasets and metrics. • For LLM-based recommendation methods, considering the cost, we select a recent and representative method closely related to our work: RLMRec. It can be observed that RLMRec only shows a slight improvement over its backbone model, LightGCN. In contrast, under the same backbone settings, our method significantly outperforms RLMRec, further validating the superiority of our approach. RLMRec only utilizes LLMs to capture textual profiles, neglecting the structural information of the KG and the higher-order semantic associations between items. Conversely, our method leverages LLMs to understand the KG and construct a semantic relational item-item graph, fully exploiting the semantic associations between items. This results in better item and user representations and improved recommendation performance.
# 5.3 Ablation Study
In this section, we demonstrate the effectiveness of our model by comparing its performance with four different versions across all four datasets. The results are shown in Table 3, where “w/o s𝑣” denotes removing the semantic embeddings of items, “w/o s𝑢” denotes removing the semantic embeddings of users, “w/o N𝑘(𝑣)”
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d28f/d28f54cb-5f1a-4e01-adea-0322fecdc640.png" style="width: 50%;"></div>
# Figure 4: Hyperparameter study of 𝑘on four datasets.
<div style="text-align: center;">Figure 4: Hyperparameter study of 𝑘on four datasets.</div>
means removing the neighbor augmentation of items based on the constructed item-item graph, and “w/o D′𝑣” means removing the second-order triples from the LLM’s prompts. When the semantic embeddings of items are removed, the model’s performance significantly decreases across all datasets, underscoring the critical role of semantic information captured by LLMs from the KG. Similarly, the removal of user semantic embeddings also results in a performance decline, affirming that LLMs can effectively infer user preferences from the KG. Furthermore, removing N𝑘(𝑣) leads to a performance drop across all datasets, highlighting the significance of the item representation augmentation module based on the constructed semantic-relational item-item graph. Without this module, the model can only capture local KG information from item-centered subgraphs and cannot leverage the semantic relations present in the global KG. The inclusion of this module facilitates the effective integration of both local and global KG information. Lastly, removing second-order KG triples from the prompts causes a slight performance decline. This finding suggests that incorporating second-order information from the KG allows the LLMs to produce a higher-quality comprehension of the local KG.
# 5.4 Hyperparameter Study
In this section, we investigate the impact of the hyperparameter 𝑘 of N𝑘(𝑣) on Recall@20 and NDCG@20 across four datasets. Here, 𝑘 represents the number of semantically related neighbors, as defined
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7b98/7b98aba2-6719-421f-a1ec-d6f1f34d2bf5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Performance comparison on different user groups, where a smaller group ID indicates fewer interaction records.</div>
in Section 4.1.2. Our experiments evaluate the model’s performance as 𝑘varies from 0 to 100. The results are presented in Figure 4. We observe that as 𝑘increases, both Recall@20 and NDCG@20 initially rise and then slightly decline across all datasets. The performance is worst when 𝑘= 0 and best when 𝑘is between 10 and 30. This can be explained as follows: when 𝑘= 0, no neighbors are used, which is equivalent to the ablation study without N𝑘(𝑣), thereby not incorporating any global semantic associations from the KG. When 𝑘> 0, the introduction of semantically related items enhances the item’s representations, leading to a noticeable improvement. However, as 𝑘continues to increase, some noise may be introduced because the relevance of neighbors decreases with their ranking. Consequently, items with lower relevance may interfere with the recommendation performance. Our findings suggest that a range of 10-30 neighbors is optimal.
# 5.5 Robustness to Varying Degrees of Sparsity
One of the key functions of KGs is to alleviate the issue of data sparsity. To further examine the robustness of our model against users with varying levels of activity, particularly its performance with less active users, we sort users based on their interaction frequency and divide them into four equal groups. A lower group ID indicates lower user activity (01 being the lowest, 04 the highest). We analyze the evaluation results on two relatively sparse datasets, Last-FM and MIND, as shown in Figure 5. By comparing our model with three representative and strong baseline models, we observe that our model consistently outperforms the baselines in each user group. Notably, the improvement ratio of our model in the sparser groups (01 and 02) is higher compared to the denser groups (03 and 04). For the group with the most limited data (Group 01), our model achieves the most significant lead. This indicates that the average improvement of our model is primarily driven by enhancements in the sparser groups, demonstrating the positive impact of CoLaKG in addressing data sparsity.
# 5.6 Case Study
In this section, we conduct an in-depth analysis of the rationality of our method through two real cases. In the first case, we present the movie “Apollo 13” and its five semantically related neighbor items in the item-item graph identified by our method. The first three movies belong to the same genre as “Apollo 13”, making them 2-hop neighbors in the KG. In contrast, the other two movies, “Top Gun” and “Star Trek”, do not share any genre or other attributes
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c58f/c58fd0ca-9bcb-45f8-9d80-df50d8e31ef6.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Case study.</div>
with “Apollo 13”, indicating they are distant or unconnected in the KG. However, “Top Gun” and “Star Trek” are semantically related to “Apollo 13” as they all highlight themes of human resilience, courage, and the spirit of adventure. Traditional KG-based recommendation methods, which rely on layer-by-layer information propagation, struggle to capture such high-order neighbors. In contrast, our method leverages similarity calculations based on item-centered KG semantic embeddings, successfully identifying these two strongly related movies. This demonstrates that our approach can effectively and efficiently capture semantically relevant information from the global KG. In the second case, we examine the movie “A Little Princess” and its related neighbors. Among the five related movies identified, “The Story of Cinderella” and “The Princess Bride” should share the same genre as “A Little Princess”. However, due to missing genres in the KG, these movies lack a path to “A Little Princess” within the KG. Despite this, our method successfully identifies these two movies. This demonstrates that our approach, by leveraging LLMs to complete and interpret the KG, can effectively address challenges posed by missing key attributes.
# 6 Conclusion
In this paper, we analyze the limitations of existing KG-based recommendation methods and propose a novel approach, CoLaKG, to address these issues. CoLaKG comprehends item-centered KG subgraphs to obtain semantic embeddings for both items and users. These semantic embeddings are then used to construct a semantic relational item-item graph, effectively leveraging global KG information. We conducted extensive experiments on four datasets to validate the effectiveness and robustness of our method. The results demonstrate that our approach significantly enhances the performance of recommendation models.
# References
[1] Arkadeep Acharya, Brijraj Singh, and Naoyuki Onoe. 2023. Llm based generation of item-description for recommendation system. In Proceedings of the 17th ACM Conference on Recommender Systems. 1204–1207. [2] Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. Tallrec: An effective and efficient tuning framework to align large language model with recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems. 1007–1014.
[3] Yixin Cao, Xiang Wang, Xiangnan He, Zikun Hu, and Tat-Seng Chua. 2019. Unifying knowledge graph learning and recommendation: Towards a better understanding of user preferences. In The world wide web conference. 151–161. [4] Jin Chen, Zheng Liu, Xu Huang, Chenwang Wu, Qi Liu, Gangwei Jiang, Yuanhao Pu, Yuxuan Lei, Xiaolong Chen, Xingmei Wang, et al. 2023. When large language models meet personalization: Perspectives of challenges and opportunities. arXiv preprint arXiv:2307.16376 (2023). [5] Djork-Arné Clevert. 2015. Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289 (2015). [6] Wenqi Fan, Zihuai Zhao, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Jiliang Tang, and Qing Li. 2023. Recommender systems in the era of large language models (llms). arXiv preprint arXiv:2307.02046 (2023). [7] Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In Empirical Methods in Natural Language Processing (EMNLP). [8] Qingyu Guo, Fuzhen Zhuang, Chuan Qin, Hengshu Zhu, Xing Xie, Hui Xiong, and Qing He. 2020. A survey on knowledge graph-based recommender systems. IEEE Transactions on Knowledge and Data Engineering 34, 8 (2020), 3549–3568. [9] Xiangnan He and Tat-Seng Chua. 2017. Neural factorization machines for sparse predictive analytics. In Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. 355–364. [10] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 639–648. [11] Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bodhisattwa Prasad Majumder, Nathan Kallus, and Julian McAuley. 2023. Large language models as zero-shot conversational recommenders. In Proceedings of the 32nd ACM international conference on information and knowledge management. 720–730. [12] Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, and Wayne Xin Zhao. 2024. Large language models are zero-shot rankers for recommender systems. In European Conference on Information Retrieval. Springer, 364–381. [13] Binbin Hu, Chuan Shi, Wayne Xin Zhao, and Philip S Yu. 2018. Leveraging metapath based context for top-n recommendation with a neural co-attention model. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1531–1540. [14] Jun Hu, Wenwen Xia, Xiaolu Zhang, Chilin Fu, Weichang Wu, Zhaoxin Huan, Ang Li, Zuoli Tang, and Jun Zhou. 2024. Enhancing sequential recommendation via llm-based semantic embedding learning. In Companion Proceedings of the ACM on Web Conference 2024. 103–111. [15] Sein Kim, Hongseok Kang, Seungyoon Choi, Donghyun Kim, Minchul Yang, and Chanyoung Park. 2024. Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1395–1406. [16] Thomas N Kipf and Max Welling. 2016. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907 (2016). [17] Lei Li, Yongfeng Zhang, Dugang Liu, and Li Chen. 2023. Large language models for generative recommendation: A survey and visionary discussions. arXiv preprint arXiv:2309.01157 (2023). [18] Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan, Ruiming Tang, Yong Yu, and Weinan Zhang. 2024. Rella: Retrieval-enhanced large language models for lifelong sequential behavior comprehension in recommendation. In Proceedings of the ACM on Web Conference 2024. 3497–3508. [19] Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. Learning entity and relation embeddings for knowledge graph completion. In Proceedings of the AAAI conference on artificial intelligence, Vol. 29. [20] Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, and Jie Tang. 2018. Deepinf: Social influence prediction with deep learning. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2110–2119. [21] Xubin Ren, Wei Wei, Lianghao Xia, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2024. Representation learning with large language models for recommendation. In Proceedings of the ACM on Web Conference 2024. 3464– 3475. [22] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2012. BPR: Bayesian personalized ranking from implicit feedback. arXiv preprint arXiv:1205.2618 (2012). [23] Steffen Rendle, Zeno Gantner, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2011. Fast context-aware recommendations with factorization machines. In Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval. 635–644. [24] Yu Tian, Yuhao Yang, Xudong Ren, Pengfei Wang, Fangzhao Wu, Qian Wang, and Chenliang Li. 2021. Joint knowledge pruning and recurrent graph convolution for news recommendation. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval. 51–60.
[25] Riku Togashi, Mayu Otani, and Shin’ichi Satoh. 2021. Alleviating cold-start problems in recommendation through pseudo-labelling over knowledge graph. In Proceedings of the 14th ACM international conference on web search and data mining. 931–939. [26] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017. Graph attention networks. arXiv preprint arXiv:1710.10903 (2017). [27] Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, and Minyi Guo. 2018. Ripplenet: Propagating user preferences on the knowledge graph for recommender systems. In Proceedings of the 27th ACM international conference on information and knowledge management. 417–426. [28] Hongwei Wang, Fuzheng Zhang, Xing Xie, and Minyi Guo. 2018. DKN: Deep knowledge-aware network for news recommendation. In Proceedings of the 2018 world wide web conference. 1835–1844. [29] Hongwei Wang, Miao Zhao, Xing Xie, Wenjie Li, and Minyi Guo. 2019. Knowledge graph convolutional networks for recommender systems. In The world wide web conference. 3307–3313. [30] Lei Wang and Ee-Peng Lim. 2023. Zero-shot next-item recommendation using large pretrained language models. arXiv preprint arXiv:2304.03153 (2023). [31] Shuyao Wang, Yongduo Sui, Chao Wang, and Hui Xiong. 2024. Unleashing the Power of Knowledge Graph for Recommendation via Invariant Learning. In Proceedings of the ACM on Web Conference 2024. 3745–3755. [32] Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, and Tat-Seng Chua. 2019. Kgat: Knowledge graph attention network for recommendation. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 950–958. [33] Xiang Wang, Tinglin Huang, Dingxian Wang, Yancheng Yuan, Zhenguang Liu, Xiangnan He, and Tat-Seng Chua. 2021. Learning intents behind interactions with knowledge graph for recommendation. In Proceedings of the web conference 2021. 878–887. [34] Xiang Wang, Dingxian Wang, Canran Xu, Xiangnan He, Yixin Cao, and Tat-Seng Chua. 2019. Explainable reasoning over knowledge graphs for recommendation. In Proceedings of the AAAI conference on artificial intelligence, Vol. 33. 5329–5336. [35] Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2024. Llmrec: Large language models with graph augmentation for recommendation. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining. 806–815. [36] Likang Wu, Zhaopeng Qiu, Zhi Zheng, Hengshu Zhu, and Enhong Chen. 2024. Exploring large language model for graph data understanding in online job recommendations. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 9178–9186. [37] Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, Hongchao Gu, Tingjia Shen, Chuan Qin, Chen Zhu, Hengshu Zhu, Qi Liu, et al. 2023. A survey on large language models for recommendation. arXiv preprint arXiv:2305.19860 (2023). [38] Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, Hongchao Gu, Tingjia Shen, Chuan Qin, Chen Zhu, Hengshu Zhu, Qi Liu, et al. 2024. A survey on large language models for recommendation. World Wide Web 27, 5 (2024), 60. [39] Xu Xie, Fei Sun, Zhaoyang Liu, Shiwen Wu, Jinyang Gao, Jiandong Zhang, Bolin Ding, and Bin Cui. 2022. Contrastive learning for sequential recommendation. In 2022 IEEE 38th international conference on data engineering (ICDE). IEEE, 1259– 1273. [40] Yuhao Yang, Chao Huang, Lianghao Xia, and Chunzhen Huang. 2023. Knowledge graph self-supervised rationalization for recommendation. In Proceedings of the 29th ACM SIGKDD conference on knowledge discovery and data mining. 3046–3056. [41] Yuhao Yang, Chao Huang, Lianghao Xia, and Chenliang Li. 2022. Knowledge graph contrastive learning for recommendation. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval. 1434–1443. [42] Xiao Yu, Xiang Ren, Yizhou Sun, Quanquan Gu, Bradley Sturt, Urvashi Khandelwal, Brandon Norick, and Jiawei Han. 2014. Personalized entity recommendation: A heterogeneous information network approach. In Proceedings of the 7th ACM international conference on Web search and data mining. 283–292. [43] Fuzheng Zhang, Nicholas Jing Yuan, Defu Lian, Xing Xie, and Wei-Ying Ma. 2016. Collaborative knowledge base embedding for recommender systems. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. 353–362. [44] Qian Zhao, Hao Qian, Ziqi Liu, Gong-Duo Zhang, and Lihong Gu. 2024. Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph. arXiv preprint arXiv:2402.13750 (2024). [45] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223 (2023). [46] Zihuai Zhao, Wenqi Fan, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen Wen, Fei Wang, Xiangyu Zhao, Jiliang Tang, et al. 2023. Recommender systems in the era of large language models (llms). arXiv preprint arXiv:2307.02046 (2023).
47] Xinjun Zhu, Yuntao Du, Yuren Mao, Lu Chen, Yujia Hu, and Yunjun Gao. 2023. Knowledge-refined Denoising Network for Robust Recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 362–371.
[48] Yaochen Zhu, Liang Wu, Qi Guo, Liangjie Hong, and Jundong Li. 2024. Collaborative large language model for recommender systems. In Proceedings of the ACM on Web Conference 2024. 3162–3172.
