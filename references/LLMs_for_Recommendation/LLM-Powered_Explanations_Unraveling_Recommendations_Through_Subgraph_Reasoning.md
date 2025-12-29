# LLM-Powered Explanations: Unraveling Recommendations Through Subgraph Reasoning
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/aeaa/aeaa9d84-29c0-4530-9c2a-12782f290861.png" style="width: 50%;"></div>
# ABSTRACT
Recommender systems (RecSys) are pivotal in enhancing user experiences across various web applications by analyzing the complicated relationships between users and items. Knowledge graphs (KGs), which model explicit relations between users and items, have been widely used to enhance the performance of recommender systems. However, a significant challenge persists in constructing KGs from unstructured data, such as reviews. Traditional information extraction tools fail to understand the complex subjective information inherent in the text such as preferences. Additionally, KGs are known to be noisy and incomplete, which are hard to provide reliable explanations for recommendation results. An explainable recommender system is crucial for the product development and subsequent decision-making. To address these challenges, we introduce a novel recommender that synergies Large Language Models (LLMs) and KGs to enhance the recommendation and provide interpretable results. Specifically, we first harness the power of LLMs to augment KG reconstruction. LLMs comprehend and decompose user reviews into new triples that are added into KGs. In this way, we can enrich KGs with explainable paths that express users’ preferences. To enhance the recommendation on augmented KGs, we introduce a novel subgraph reasoning module that effectively measures the importance of nodes and discovers reasoning for recommendation. Finally, these reasoning paths are fed into the LLMs to
∗Corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. Conference acronym ’XX, June 03–05, 2018, Woodstock, NY © 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/18/06 https://doi.org/XXXXXXX.XXXXXXX
Yuxiao Li∗ yuxiao.li@cn.bosch.com Bosch Corporate Research Shanghai, China
generate interpretable explanations of the recommendation results. Our approach significantly enhances both the effectiveness and interpretability of recommender systems, especially in cross-selling scenarios where traditional methods falter. The effectiveness of our approach has been rigorously tested on four open real-world datasets, with our methods demonstrating a superior performance over contemporary state-of-the-art techniques by an average improvement of 12%. The application of our model in a multinational engineering and technology company (METC)’s cross-selling recommendation system further underscores its practical utility and potential to redefine recommendation practices through improved accuracy and user trust.
# CCS CONCEPTS
• Do Not Use This Code →Generate the Correct Terms for Your Paper; Generate the Correct Terms for Your Paper; Generate the Correct Terms for Your Paper; Generate the Correct Terms for Your Paper.
# KEYWORDS
Explainable Recommendation, Large Language Model, Knowledge Graph
ACM Reference Format: Guangsi Shi, Xiaofeng Deng, Linhao Luo, Lijuan Xia, Lei Bao, Bei Ye, Fei Du, Shirui Pan, and Yuxiao Li. 2018. LLM-Powered Explanations: Unraveling Recommendations Through Subgraph Reasoning. In Woodstock ’18: ACM Symposium on Neural Gaze Detection, June 03–05, 2018, Woodstock, NY. ACM, New York, NY, USA, 11 pages. https://doi.org/XXXXXXX.XXXXXXX
# 1 INTRODUCTION
Despite recommendation systems have become indispensable in mitigating information overload and enhancing user experience across modern web platforms and applications [9], most of recommendations fail to offer explanations, which is essential for
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ee21/ee21841c-cfae-407b-9456-e76a61f85a85.png" style="width: 50%;"></div>
Figure 1: Main scenario and task: Arrows of different colors in KGs represent different channels (e-commerce platforms). Different products may not be sold in the same channel. The system needs cross channels to recommend and provide human-acceptable, reliable explanation descriptions.
decision-making by both users and e-commercial platforms. Consequently, there has been a paradigm shift in the recommendation system research community, with a growing emphasis not only on the accuracy of recommendations but also on their explainability. An explainable recommendation system significantly increases system transparency, boosts user trust and acceptance, and enhances recommendation efficiency [1]. Moreover, majority of literature focuses on natural interpretability of KG with rich structure reasoning [40] for improving recommendation system performance and explanation [18, 44], but they often fail to construct the semantic relationships such as emotion, preference pertinent to users from review. This limitation hampers the interpretability, implicit relevance, and accuracy of recommendations derived from these models with noisy and incomplete information [22]. Extracting semantic information from text for knowledge graph construction faces several limitations. Unstructured data complexity and the absence of well-defined ontologies challenge traditional tools in relation extraction and entity linking. These tools often struggle with context understanding, leading to inaccuracies in entity-relation identification and semantic interpretation [7]. Furthermore, one of the important way of KG reasoning for explainable recommender is rule-based methods [36, 39], while these approaches can only depend on the existing paths and search an optimal path for explanation, and it does not work in the cross-selling scenarios, where the potential links may not be built in the original knowledge graph. It will result in "recommendation hallucination" which forces explanations solely for the sake of recommending. For example, different business units in multinational engineering and technology company (METC) group operate their own channels for selling their products. To maximize
production profits, it is essential to implement cross-selling across these units. Traditional strategies may not suffice for this purpose, underscoring the importance of an explainable recommendation system for effective cross-selling 1, which is crucial for METC’s development. The advent of LLM, epitomized by advancements such as ChatGPT has attracted widespread attention and developed rapidly. Its excellent understanding capabilities and easy-powerful tools are capable of sophisticated reasoning and generation tasks in realworld applications [21]. However, the way of successfully applying or combining their powerful capabilities into the recommendation system is still a promising but challenging task. According to the requirement and issues above, synergizing knowledge graphs with LLMs developments present a promising road map for recommender. However, the predominant mode of implict integration between knowledge graphs and LLMs remains embedding-based strategies [33, 47]. This approach merges knowledge graph data embedding with word data embedding to inform the learning of user-item representations but often loses robust reasoning capabilities of LLMs. This deficiency is particularly pronounced when addressing diverse and specific business needs. Additionally, LLMs-enhanced KG systems [16, 17] with explicit modelling are constrained by their reliance on predefined meta-paths, necessitating a nuanced consideration of how to design effective rule paths tailored to specific business scenarios. More importantly, if fine-tuning LLMs for improving performance, it requires a lot of computing resources which is not friendly for actual marketing analysis. In response to the identified challenges, we introduce a cuttingedge framework named LLM powered Subgraph Reasoning to facilitate an explainable Recommendation system (LLM-SRR for abbreviation) which can be seen in Figure 1. This novel architecture capitalizes on the unique strengths of LLM by prompt engineering and knowledge graphs to overcome the limitations of existing recommendation systems. The LLM-SRR framework is delineated through a three-step process: First, Information extraction and KG reconstruction by LLMs: Our approach begins by extracting relevant information from user reviews. This process involves identifying new features based on predefined targets, which are critical in capturing the nuanced key works and semantic information of users by LLM’s prompt engineering. A new KG integrates both original user-item relationships and the newly identified targets from user reviews. Second, Subgraph Reasoning by Attendtionbased Diffusion Scoring: This step involves the implementation of an attention mechanism for effective message passing in subgraph reasoning which skips out of the original connection pattern to search for more potential link relation, followed by a recommendation scoring process to rank the items. Last, LLMs generating Explainable Description: The LLM continues to leverage predefined keywords, the reasoning path generated by the subgraph, and the coherent descriptions acquired through thorough comprehension. Such explainable context aids front-end analysts in decision-making and planning processes. This method both enhances the recommendation’s accuracy and ensures that the post-hoc explanation 1
1Within a company group, cross-selling indicates that products from different business units/product lines of the company are successfully sold to one consumer.
path and description behind each recommendation is transparent and understandable to both users and brands. In summary, the contribution of this paper is threefold: • Contribution 1: To the best of our knowledge, this study represents the effort to a LLM powered explainable recommendation system by subgraph reasoning. This innovative alignment introduces a novel paradigm in the domain of personalized recommendation systems, especially in a novel cross selling scenario. • Contribution 2: We have developed customizable, user-friendly tools designed specifically for explainable recommender. These tools not only facilitate understanding the semantic information of user but also provide an autonomous post-hoc explanation description by LLMs, thereby enhancing transparency and understandability in recommender for marketing analyst. • Contribution 3: The efficacy of subgraph reasoning module has been rigorously tested across three open source recommendation datasets, where it has demonstrated state-ofthe-art performance. Furthermore, its applicability has been successfully validated in a real-world scenario involving cross-selling activities at METC, where it yielded highly favorable outcomes. Therefore, our LLM-SRR architecture effectively harnesses the capabilities of subgraph for reasoning and the semantic understanding ability of LLMs to provide a highly explainable recommendation system. It addresses specific user and brand requirements through a visualizable and understandable recommendation pathway, significantly enhancing the persuasiveness of the system. Moreover, the nature of knowledge graphs, which can be updated based on unstructure information, alongside the system’s ease of training, positions LLM-SRR as a robust, adaptable, and user-centric solution in the realm of explainable recommendation systems.
# 2 RELATED WORKS
# 2.1 Explainable Recommendation System
Explainable recommendation systems have emerged as a pivotal enhancement in recommendation tasks, with their capacity to not only increase the efficacy of recommendations but also bolster their trustworthiness. Such systems are broadly categorized into two types: post-hoc and model-based explanations. In the realm of post-hoc explanation models, CountER [29] employs counterfactual constrained learning to derive succinct yet potent explanations for otherwise opaque recommendation models. Conversely, model-based explanations provide insights directly from the recommendation process itself. KPRN [37] utilizes LSTM to process paths within KGs from users to items, generating embeddings for these paths which are then evaluated for their relevance. Similarly, EIUM [8] focuses on explicating the semantic paths between users and items, thereby equipping the recommendation system with the capacity for path-wise explanation. RuleRec [19] introduces a rule-guided framework that derives rules from KGs for item recommendations. For a more personalized and explainable approach, PGPR [39] and ReMR [36] employ path reasoning and multi-level reasoning, respectively, through reinforcement learning to refine recommendations. Recent studies have begun to underscore the
importance of subgraphs within knowledge graphs for enhancing explainability. GraIL [30] pioneers inductive relationship predictions using subgraph reasoning. Moreover, GnnExplainer [42] and CF-GNNExplainer [15] elucidate Graph Neural Networks (GNNs) via subgraph analyses. Within the specific context of recommender systems, GREASE [3] innovatively employs subgraphs to furnish both factual and counterfactual explanations for GNN-based blackbox models. However, most of them fall short of explicating the model’s internal workings or enhancing model performance comprehensively and cannot make full use of other additional information to help reasoning such as text information and cannot skip original relation structures.
# 2.2 Large Language Models and Knowledge Graphs Combination
Researchers have explored integrating KG with LLM at different stages to enhance their capabilities. During pre-training, incorporating KG aids LLM in assimilating knowledge [25]. At the inference stage, accessing KG bolsters LLM’s performance in domain-specific knowledge [11]. Furthermore, KG contribute to interpreting LLM by clarifying facts [23] and elucidating the reasoning process [12], thereby improving interpretability. Moreover, knowledge graph often struggle with incompleteness [2] and text corpus processing for KG construction [48]. Leveraging LLM’s generalizability, researchers are utilizing LLM to enhance KG tasks. By employing LLM as text encoders, they process textual content within KG, using the generated text representations to improve KG’s comprehensiveness [45]. Furthermore, LLM is applied to extract entities and relationships from text for KG creation [10]. Recent efforts focus on designing KG prompts that transform KG structures into formats understandable by LLM, facilitating direct LLM application in tasks like KG reasoning [4]. The integration of KG and LLM have become a focal point of research, given their complementary nature [34]. This synergy aims at creating a unified framework to leverage the strengths of both, enhancing their capabilities. The Synergized Model can enhance the mutual capabilities of LLM and KG, while the technique layer incorporates various methods to boost performance further. This integrated approach can be applied to real-world such as search engines [31], recommender systems [14], and AI assistants [28], showcasing the practicality of our unified framework.
# 2.3 Subgraph Reasoning
Subgraph reasoning has emerged as a potent paradigm for enhancing the interpretability and performance of models across various domains and powerful ability of skip hop has been proved by [5]. A novel approach [6] that combines temporal relational attention with reverse representation updates to guide subgraph extraction. In fake news detection, a reinforcement subgraph generation method [41] alongside a hierarchical graph attention network improves both generalization and discrimination, offering clear explainability by identifying critical subgraphs. For fraud detection, SubGNN [26], leverages heterogeneous subgraphs and a relational graph isomorphism network for precise fraud identification without relying on global IDs.
A novel approach [27] for inductive relation prediction incorporates substructure information into subgraph reasoning, significantly enhancing precision by utilizing semantic correlations between relations. Addressing scalability in KG, one-shot subgraph reasoning proposes a two-step prediction process that significantly increases efficiency and performance on large-scale KG [46]. For question answering, integrating subgraph-aware relation and direction reasoning into a novel neural model, RDAS [38], substantially improves answer precision by leveraging structure and direction information within subgraphs. CoMPILE [20] innovates by enhancing message interactions and efficiently processing asymmetric relations, showcasing significant advances over traditional models.
In this section, we introduce our model LLM-SRR which is shown in 2.
# 3.1 Preliminary
In recommendation systems, a knowledge graph is formally defined as G, G = {(𝑒ℎ,𝑟,𝑒𝑡)|𝑒ℎ∈E,𝑒𝑡∈E,𝑟∈R}, where E is the sets of entities and R is the sets of relations. In this paper, we consider a special type of knowledge graph for recommendation system, denoted by GR. It contains a subset of a User entities U, a subset of Item entities I and Properties P(item information, user portrait and so on), where U ∪I ∪P ⊆E and U ∩I ∩P = ∅. These three kinds of entities are connected through relations 𝑟𝑠. We give a relaxed definition of knowledge graph information injection as follows. Definition 1 - Knowledge Graph Reconstruction: Knowledge graph reconstruction is the systematic assimilation of insights from textual data into a knowledge graph utilizing a LLMs. The LLM parses unstructured text, extracting entities including key words and semantic information, and discerning their relations to form a new GR. This mechanism, represented by a function 𝑓: T →GR, transcribes the all kinds of predefined targets from the text into graph structures—triples of the form {( ´𝑒ℎ, ´𝑟, ´𝑒𝑡)}—thus enriching the existing KG with new, verifiable information. Example: Consider the comment text "I like METC’s wash machine colour." The LLM identifies user, METC, wash machine and colour, and relations like and belong, forming the GR triples: user like −−−→wash machine belong −−−−−→METC. Definition 2 - Sub-Graph Reasoning: In the reconstructed KGs, subgraph reasoning is defined as a sequence of diffusion from step 𝑠to step 𝑠+ 1 staring from the central entity 𝑒𝑠,𝑖, denoted by 𝑑𝑠+1,𝑖(𝑒𝑠,𝑖, · · · ,𝑒𝑠,𝑗) = 𝛩{𝑒𝑠,0 𝑟𝑠,0 ←−→𝑒𝑠,1,𝑒𝑠,0 𝑟𝑠,1 ←−→𝑒𝑠,2, · · · ,𝑒𝑠,𝑖 𝑟𝑠,𝑘 ←−→ 𝑒𝑠,𝑗} , where 𝑠∈𝑆in the s-th step of attention-based diffusion process, 𝛩donates a function to select 𝑁entities as the central entities for the 𝑠+ 1 step, the 𝑖∈𝑁is the 𝑖-th central entity in step 𝑠, especially when 𝑠= 0, 𝑖is the number of the users, 𝑘is the 𝐾-th relations between the 𝑒𝑠,𝑖and 𝑒𝑠,𝑗, and 𝑗∈𝐽is the 𝑗-th neighbour of central entity 𝑒𝑠,𝑖. Definition 3 - Explainable Recommendation: Explainable Recommendation is defined as given a sequence path SP = {𝑒0,𝑖 𝑟0,𝑖 ←→ 𝑒1,𝑖 𝑟1,𝑖 ←→· · · 𝑟𝑠−1,𝑘 ←−−−→𝑒𝑠,𝑖} generated by the subgraph, especially
User ↔Item in recommendation system, the goal is to find a recommendation score function 𝑆(User, Item) considering the reasoning path SP. And the recommendation result, reasoning path, and the predefined information will be fed to the LLMs to generate a explanation description.
# 3.2 KG Reconstruction and Explanation Generation
3.2.1 LLMs for Information Extraction and Injection. Our method commences with the decomposition of information using a LLM. This process involves the extraction of new entities and relationships from a given text by predefined targets. The prompt engineering technique is employed to refine queries which guide the LLM towards precise extractions. Newly identified entities and relations are then integrated into the existing KG, resulting in a reconstructed KG that encapsulates the enriched data. There are two examples for entity and relation extraction respectively.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/759e/759eae8d-cb5e-4864-bdb2-eecbd5a872e8.png" style="width: 50%;"></div>
<div style="text-align: center;">This prompt facilitates the extraction of time entities and their associated events from user reviews (<Review>).</div>
The above prompt guides the LLM in analyzing reviews (<Review> and structuring the output as a sentiment-indicative targets. Specific rules are established to govern the integration of new information into the KG. These rules are tailored by product analyst to the scenario at hand, encompassing relations such as emotions, preferences, and quantifiable attributes (e.g., price, color, style), as well as time-connected entities such as significant dates. All extracted information will form a new link and embed it into the existing KG according to the customized set of rules.
3.2.2 LLMs for Post-hoc Explanation Generation. In the stage of generating interpretable descriptions, we provide the large language model with meaningful contextual prompts, including predefined key targets and subgraphs or paths generated by the subgraph reasoning process. Then, according to the requirements, we generate a segment of language description. Specific instances can be referenced in Table 6 and Table7. Below is the template we need for generation.
Prompt Example for Explainable Description: Generate an explanation for this recommendation "<item->user>", based on the predefined target: "<targets>", and the reasoning path"<path> ". I will provide you some answering examples.
This prompt facilitates to generate the final explanation description based on the customized information.
This prompt facilitates to generate the final explanation description based on the customized information.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/324a/324aedb5-ea83-4219-8fb5-af2513a34292.png" style="width: 50%;"></div>
Figure 2: Framework of LLM-SRR: I. Text information is extracted by the LLM and injected into the original knowledge graph by pre-define rules {u is user, p is the property, i is item, r is relation}; II. The attention score is calculated by the neighbours in different diffusion layers and subgraph could be generated. III. The final recommendation score is computed by the similarity function between the user and item, where a explanation path can be generated by LLM in this component.
# 3.3 Attentive Diffusion Subgraph Reasoning
We introduce an attention-based mechanism to construct a entitycentric subgraph, harnessing the entity’s interaction history and contextual relevance within the knowledge graph. Initially, the subgraph centers around the user, progressively expanding by assimilating nodes based on their attention scores which signify their contextual importance to the entity. At any given diffusion step 𝑠, we commence with the current entity subgraph G𝑠,𝑒and identify the set of newly added nodes, designated as neighbor nodes N𝑠,𝑒. Specifically, the initial subgraph G0,𝑢contains solely the user node, with neighbours N0,𝑢. The next phase involves computing the attention scores for the edges 𝐸𝑠,𝑒 connected to N𝑠,𝑒. This set of 𝐸𝑠,𝑒encompasses relationships between the central entity G𝑠,𝑒and their N𝑠,𝑒. The attention mechanism then evaluates the importance of these edges, determining the significance of the 𝐸𝑠,𝑒to the G0,𝑢. Consequently, entity scores are calculated based on the aggregated attention scores of their associated edges, selecting the top 𝑁nodes 𝑁(N𝑠,𝑒) from the entire set of one-hop neighbors, rather than 𝑁nodes per neighbor node. These top-scored nodes are then appended to the subgraph, diffusing it to 𝑔𝑠+1,𝑒. In the step 𝑠, specifically, the attention module is responsible for assigning scores to edges based on their relevance to the entity embedding. In this stage, the attention score 𝛼of an edge (𝑒𝑖,𝑟,𝑒𝑗) is computed by the following equations
(2)
� ()∈ where 𝜃= Sigmoid, 𝛿= LeakyReLU, 𝑊1 ∈R𝑑1𝑋2𝑑, 𝑊2 ∈R𝑑𝑋𝑑1, 𝑑1 is a parameter that controls the size of the trainable matrices 𝑊1 and 𝑊2, ℎ𝑒𝑢∈R𝑑and ℎ𝑒𝑢represents the emedding of user entities 𝑢, ℎ𝑒𝑖,ℎ𝑒𝑗∈R𝑑and ℎ𝑒𝑖,ℎ𝑒𝑗represents the embedding of entities 𝑒, (𝑎∥𝑏) denotes concatenation of embedding vectors 𝑎and 𝑏, 𝛼(𝑒𝑖,𝑟,𝑒𝑗) represents the attention score of edge (𝑒𝑖,𝑟,𝑒𝑗) at step 𝑠. The edge (𝑒𝑖,𝑟,𝑒𝑗) is from the set of edges 𝐸𝑠,𝑒. After obtaining the attention scores of the edges, 𝑒𝑗entity scores are derived by aggregating the attention scores from the edges to their corresponding entities. ∑︁
(3)
� ∈� where score(𝑒) represents the attention score of entity and 𝑠𝑐𝑜𝑟𝑒(𝑒𝑖) has been calculated in the last run. Then s-th step user subgraph is updated by adding the m highest scoring nodes to the user subgraph. After selecting the top 𝑁(𝑁<= 𝐽) score neighbours from the all neighbours 𝑒𝑗, the 𝑁(N𝑠,𝑒) need to be re-scored by normalization. Additionally, 𝑁is a hyperparameter used to control the subgraph size. 𝑣𝑛∈𝑉𝑛will become the new central entity in step 𝑠+ 1.
(5)
# 3.4 Recommendation Scoring
After constructing the user subgraph through the aforementioned attention-based scoring diffusion method, the next stage is to compute the recommendation scores for item nodes within this SubKnowledge-Graph that are connected to this user subgraph. The recommendation scoring process, visualized in 2, moves beyond linear reasoning by encoding the relationships between different paths within the user subgraph. This allows the model to capture a comprehensive representation of user interests and to reflect the interplay of these paths when generating recommendations. Initially, we delineate the subgraph pertinent to the item node from the overarching user subgraph. This subgraph is then encoded to extract user preferences using a subgraph encoder. The encoding process is formulated as follows:
(6)
(7)
  where ℎ𝑣∈R𝑑and ℎ𝑔𝑠represents the embedding of subgraph 𝑔 including 𝑣𝑠,𝑖in step 𝑠. The set 𝑉𝑢𝑘comprises the lists of nodes at a distance 𝑘from the user node within the subgraph. 𝑊3 ∈R ˆ𝑑2𝑋3𝑑, 𝑊4 ∈R𝑑𝑋ˆ𝑑2, ˆ𝑑2 is a parameter that controls the size of the trainable matrices 𝑊3 and 𝑊4. The user 𝑢subgraph is composed of nodes user 𝑢, 𝑔1 and 𝑔2, so ℎ𝑢,{𝑔1,𝑔2} ∈R𝑑represents the embedding of the user 𝑢subgraph. Subsequently, the similarity score between this subgraph representation and the item’s embedding is calculated:
(8)
()({} ·) where sim(𝑢,𝑖𝑡𝑚) is the similarity score between user 𝑢and the m-th item 𝑖𝑡𝑚. To account for the connection between the subgraph and the full user subgraph, we integrate the score derived during subgraph construction as the weight for each subgraph. This weighted score is computed as follows: ∑︁
(9)
∑︁ ∈ Here, S𝑢,𝑖𝑡𝑚denotes the final recommendation score for an item 𝑖𝑡𝑚relative to user 𝑢, incorporating both the encoded subgraphuser-item relationship and the subgraph’s relevance within the user subgraph. The existing objective function only guides the generation of the subgraph and does not indicate which subgraph truly reflects the user’s preferences. Therefore, we introduce a specific optimization objective for the recommendation module, defined as:
(10)
∑︁ ∈() This objective function minimizes the negative log-likelihood over the set of items positively associated with the user 𝑢, denoted by 𝑌(𝑢), refining the model’s ability to deduce user preferences from subgraphs.
# EXPERIMENTS
In this section, our experiment is mainly divided into two parts. The first part is a performance experiment, which mainly reflects the recommended performance of our subgraph-based inference by comparing it with other baselines. In the second part, we use a case study to explore the contribution of our model’s LLM information injection to the entire recommendation system, and tests the improvement ability of subgraph (SG) and parameter sensitivity of our design module. In the experiments of this work, we used entity representations pre-trained via TransE-based in ReMR.
<div style="text-align: center;">Table 1: Statistics of the datasets.</div>
METC
Beauty
Cell Phones
Clothing
#User
1004
22,363
27,879
39,387
#Items
1017
12,101
10,429
23,033
#Entities
2482
224,080
163,255
425,534
#Relations
12
16
16
16
#Interactions
3636
198.58K
194.32K
278.86K
#Triples
129980
37.73M
37.01M
36.37M
# 4.1 Experiment Settings
4.1.1 Datasets. We conducted experiments on three datasets, among them are the Amazon review dataset collated in KGAT[35], and METC dataset. METC dataset is a collection of METC order data from different channels. We used three datasets from the Amazon dataset, namely Cell Phones, Beauty and Clothing. When dealing with the Amazon dataset, we followed the data processing methods in PGPR[39]. The details of the dataset statistics are shown in Table 1.
4.1.1 Datasets. We conducted experiments on three datasets, among them are the Amazon review dataset collated in KGAT[35], and METC dataset. METC dataset is a collection of METC order data from different channels. We used three datasets from the Amazon dataset, namely Cell Phones, Beauty and Clothing. When dealing with the Amazon dataset, we followed the data processing methods in PGPR[39]. The details of the dataset statistics are shown in Table 1. 4.1.2 Baselines. We consider six recommendation approaches as baselines in the following experiments. These baselines are divided into three categories, Matrix Factorization-based models, KG embedding models and path reasoning models. BPR[24]: BPR is a personalized ranking algorithm based only on user product interaction information through Bayesian posterior optimization. DKN[32]: This is a recommendation model that combines knowledge graph reality and convolutional neural network CKE[43]: CKE uses TransR[13] to obtain semantic embeddings from the knowledge graph to enhance collabrative filtering. KGAT[35]: KGAT learns entity embeddings from knowledge graphs through graph attention networks combined with GNN and attention mechanisms. PGPR[39]: PGPR is a knowledge graph path reasoning model based on reinforcement learning. ReMR[36]: ReMR relies on its own data to obtain higher-order abstraction information for the entities in the knowledge graph to create a multi-layer knowledge graph. 4.1.3 Evaluation Criteria. For all approaches, we adopted four evaluation criteria to evaluate the top-5 recommendations of each user in the test set, including Normalized Discounted Cumulative Gain (NDCG), Recall, Hit Rate (HR), and Precision (Prec.).
<div style="text-align: center;">able 2: Performance on top-10 recommendation between the baselines and our model. The results are computed in the test set nd are given as percentages %. The best baseline results are underlined.</div>
Beauty
Cell Phones
Clothing
Measures(%)
NDCG
Recall
HR
Prec.
NDCG
Recall
HR
Prec.
NDCG
Recall
HR
Prec.
BPR
2.805
5.032
8.933
1.173
1.995
3.534
5.424
0.623
0.665
1.219
1.932
0.323
DKN
1.923
2.591
8.812
1.135
1.672
3.313
4.580
0.349
0.375
0.724
1.492
0.119
CKE
3.824
6.241
11.132
1.422
3.849
6.981
10.633
1.073
1.656
2.604
4.329
0.390
KGAT
5.020
7.794
12.496
1.535
4.803
7.982
11.241
1.134
2.824
4.674
6.993
0.603
PGPR
5.489
8.324
14.347
1.692
4.921
8.383
11.832
1.280
2.863
4.797
7.024
0.719
ReMR
5.878
8.982
15.606
1.906
5.294
8.724
12.498
1.337
2.977
5.110
7.426
0.766
LLM-SRR
6.187
9.788
16.103
1.928
5.755
9.753
13.378
1.433
3.520
6.050
8.739
0.922
Improvement(%)
+5.257
+8.974
+3.185
+1.154
+8.708
+11.795
+7.041
+7.180
+17.450
+18.395
+17.681
+20.366
4.1.4 Implementation Details. In our model, the entity embedding dimensionality is 100.The hyperparameter of subgraph size is set to a a maximum of 100. We train the parameters with Adam optimization, batch size of 256, and a number of training epochs of 10.
# 4.2 Performance Experiments
4.2.1 Performance Comparison. We show the top-10 recommendation performance of our proposed method compared to all baselines. And comparison with the model ReMR is not possible in METC dataset because of the lack of higher-order abstract information of entities. The specific information is shown in Table 2. Table 2 shows that our method consistently outperforms all baselines in terms of recommendation accuracy. On average, our model improves NDCG, Recall, HR and Precision by 5.36%, 9.33%, 6.58%, and 6.46%. This demonstrates that subgraph reasoning with LLM can help better infer user interests and improve recommendation performance. It is noted that the algorithms that focus on node representation learning are not very effective on the Amazon data set. Maybe Amazon’s knowledge graph contains more information and user interests are more complex, so it is not appropriate to only focus on node embedding. And LLM-SRR has the greatest performance improvement in recall, which to a certain extent shows that LLM-SRR can fully learn user interests.
# 4.3 Case Study for METC
4.3.1 Brief Introduction. In order to illustrate the explanation in recommendation process, we do a case study in METC private dataset to visualize the reasoning paths and the improvement by LLM. METC’s product families are all durable goods and in diverse shopping domains and channels. Based on the market analysis, more than half of METC consumers tend to purchase more than two categories of products in more than two e-commercial channels. Hence, analysing the decision making path of consumers and providing an accurate and explainable recommendation is challenging but critical for METC business growth. The METC dataset includes three key aspects. 1) User table: it includes the user attributes, e.g., id, profiles, reviews, and preferred channels. 2) Order log: it includes the details of each order, e.g., user id, product id, product properties, review of the order. 3) User
<div style="text-align: center;">Table 3: Results in METC Dataset</div>
METC
Subgraph Size
NDCG
Recall
HR
Prec.
BPR
14.961
20.941
23.705
2.420
DKN
11.137
18.034
19.821
2.042
CKE
9.436
19.687
22.709
2.371
KGAT
14.707
21.847
23.606
2.410
PGPR
17.767
24.253
27.191
2.799
ReMR
-
-
-
-
LLM-SRR
18.144
26.006
29.781
3.108
Improvement(%)
+2.122
+7.228
+9.525
+11.040
activities: it includes the event tracing data in the online/offline channels such as the click and view. The Table 3 shows that our proposed model outperforms the baselines in all four measures.
4.3.2 Explanation Visualization. Figure 3(a) illustrates how our model show an explainable result and echo the business questions in the METC application. It can be seen that for auto air filter users, the property like car owners and METC premium user(user profiles), Channel 1 and Channel 2 (selling channels like E-commerce platforms), reliable and no smell (reviews), and auto wiper purchasing (cross-selling pair) are the main contributors in the recommendations, which indicates how the majority of users made their decisions. The red nodes are generated by the LLM and the blue dashed line is one of the most probable reasoning path. Finally, the item4-oven is highly probable to recommend to the user1.(PS: the number in different nodes is the score for user1). In Figure 3(b), it is another typical scenario. User5 has only bought METC home appliances on Channel 1. User6 bought different METC prodution in different channels such as METC home appliances on Channel 1, METC power tools on Channel 3, and METC car accessories on Channel 4. User7 only bought METC car accessories on Channel 4. We finally recommended METC cordless drill to user5 and user6. Explanation path comparison between path-based method and ours has been shown in Table 6 and Table 7.
4.3.3 Observations and Analysis. In cases exemplified by example1 and example2, although the target items in the list, while
<div style="text-align: center;">Table 4: Ablation study</div>
METC
Measures(%)
NDCG
Recall
HR
Prec.
w/o Review
16.378
23.045
25.863
2.878
w/o SG
15.964
21.582
22.266
2.246
Full model
18.144
26.006
29.781
3.108
<div style="text-align: center;">Table 5: Influence of Subgraph Size</div>
METC
Subgraph Size
NDCG
Recall
HR
Prec.
60
17.258
23.433
26.494
2.739
80
18.144
26.006
29.781
3.108
100
17.584
24.750
28.586
2.928
the path-based method can invariably generate an existed path to suggest a target product to a specific user, where the paths seem far-fetched and even make no sense. This is named as one type of "recommendation hallucination". Furthermore, other items in the recommendation list for user that exist similar explainable paths, which can not reflect the requirement of users. In example3 and example4, the path-based recommendation approach indiscriminately suggests all products associated with User6 to both User5 and User7. For instance, car accessories are recommended to User5, and home appliances to User7. This strategy can lead to what may be described as another "recommendation hallucination" where the rationale provided by the system does not substantiate the recommendations made, thus potentially exerting a detrimental impact on decision-making processes. In contrast, utilizing the reliable explanation paths offered by our method suggests that the recommendation system gains a more profound understanding of the relationships between users and items. Consequently, the items that appear in the recommendation list are more accurately aligned with the user’s needs.
4.3.4 Ablation Studies. On the METC dataset, We further study the importance of each module of our model in Table 4, where w/o Review means that our remove the entities which are extracted from user review and w/o SG is our model that removes subgraph generating. The experimental results shows that, after removing these modules, the noise could not be attenuated and uncompleted information could interfere with reasoning process, leading to poor results, while the subgraph generation mechanism of our model can better capture user interests.
# 4.3.5 Parameters sensitivity. As can be seen in Table 5, hyperparameter of subgraph size indicates the maximum numbe
4.3.5 Parameters sensitivity. As can be seen in Table 5, the hyperparameter of subgraph size indicates the maximum number of nodes for subgraph expansion in each round of subgraph reasoning. Larger hyperparameter means that we generate larger subgraphs with more nodes in them. Larger subgraphs tend to contain more information, which on the one hand increases the tolerance for the performance of the subgraph inference module, but on the other hand it increases the performance requirements for the subgraph scoring module.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6e80/6e80378b-5bf6-4eec-a30c-19d145faa7f8.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Case study 1</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2f31/2f3133af-da61-4cb5-8560-b07344dcdc7e.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Case study 2</div>
Figure 3: Real cases discovered by our model, each containing a subgraph which end nodes is predicted item by recommendation model. u1 – u4 : METC users, i1 – i4 : Auto air filter related products, R_R1: “review: reliable”, R_R2 : “review: no smell”, O_AW: “order: auto wiper”, S_HD: “search: Home Decoration ”, C_1: “channel: Channel 1 auto parts flagship store”, P_CO: “profile: car owner” , C_2: “channel: Channel 2”, P_PU: “profile: premium user”, C_3: “channel: Channel 3 ”, C_4: “channel: Channel 4”.
# 5 CONCLUSION
In this paper, we present a novel integration of a LLM, and subgraph of KG generation to foster the development of an explainable recommendation system. This represents the inaugural effort to amalgamate these advanced technologies for enhancing recommendation systems. Specifically, the LLM is leveraged to distill key information from textual data, which is then systematically incorporated into the KG using pre-defined rules. Furthermore, we
<div style="text-align: center;">Table 6: Case 1 of explanation comparison by Path-based reasoning and LLM-SRR.</div>
Example 1
METC Oven(𝐼𝑡𝑒𝑚4) →𝑈𝑠𝑒𝑟1
Path-based reasoning
𝑈𝑠𝑒𝑟1 →purchase →auto wiper →
sold by →C_1 →sale →𝐼𝑡𝑒𝑚4
Explanation
Because the user purchased a auto wiper directly from METC
and Channel 1 platform sale the Channel 2 production in-
cluding auto wiper, wash machine, oven and so on, system
recommends oven to the user.
Subgraph-based reasoning by ours
𝑈𝑠𝑒𝑟1 →review →reliable →C_1 →
sale →𝐼𝑡𝑒𝑚4
Explanation
User commented the METC heating system with reliable
and no smell. system guesses that user pay more attention to
"reliable". Channel 1 is a reliable platform which sale number
of METC production including auto wiper, wash machine,
oven and so on. Thus system recommends oven to the user.
Example 2
METC Heating System (𝐼𝑡𝑒𝑚3) →𝑈𝑠𝑒𝑟4
Path-based reasoning
𝑈𝑠𝑒𝑟4 →profile →car owner →tag →
C_1 →sale →𝐼𝑡𝑒𝑚3
Explanation
User was labeled car owner as his one of profile and Channel
1 platform has the same tag, and Channel 1 sale number of
METC production including auto wiper, wash machine, oven
and so on so the system recommends wash machine to this
user.
Subgraph-based reasoning by ours
𝑈𝑠𝑒𝑟4 →register →premium →read
→Wechat →content →𝐼𝑡𝑒𝑚3
Explanation
This user is a premium of METC in Wechat and read METC
washing machine article as well. system guesses that this
user have high probable willing to buy METC wash machine.
Also, Channel is one of the biggest sale platform who sale
number of METC production including auto wiper, wash
machine, oven and so on so the system recommends wash
machine to this user.
<div style="text-align: center;">Table 7: Case 2 of explanation comparison by Path-based reasoning and LLM-SRR.</div>
Example 3
METC Cordless Drill (𝐼𝑡𝑒𝑚7) →𝑈𝑠𝑒𝑟5
Path-based reasoning
𝑈𝑠𝑒𝑟5 →search →Home Decoration →
key words →C_1 →purchase →𝑈𝑠𝑒𝑟6
→purchase →C_3 →sale →𝐼𝑡𝑒𝑚7
Explanation
Because 𝑈𝑠𝑒𝑟5 and 𝑈𝑠𝑒𝑟6 bought METC household appli-
ances in Channel 1, system guesses what 𝑈𝑠𝑒𝑟6 bought in
Channel 3 is suitable for 𝑈𝑠𝑒𝑟5. Thus, system recommends
cordless drill to 𝑈𝑠𝑒𝑟5
Subgraph-based reasoning by ours
𝑈𝑠𝑒𝑟5 →search →Home decoration →
key words →C_3 →sale →𝐼𝑡𝑒𝑚7
Explanation
𝑈𝑠𝑒𝑟5 searched home decoration as key words in Channel
1 and bought some household appliances, system guesses
𝑈𝑠𝑒𝑟5 needs some tools to install the household appliances,
so system recommends cordless drill to 𝑈𝑠𝑒𝑟5
Example 4
METC Cordless Drill (𝐼𝑡𝑒𝑚7) →𝑈𝑠𝑒𝑟7
Path-based reasoning
𝑈𝑠𝑒𝑟7 →profile →car owner →tag →
C_4 →purchase →𝑈𝑠𝑒𝑟6 →purchase
→C_3 →sale →𝐼𝑡𝑒𝑚6
Explanation
Because 𝑈𝑠𝑒𝑟7 and 𝑈𝑠𝑒𝑟6 bought METC car battery in Chan-
nel 4, we guess what 𝑈𝑠𝑒𝑟6 bought in Channel 3 is suitable
for 𝑈𝑠𝑒𝑟7. Thus system recommends cordless drill to 𝑈𝑠𝑒𝑟7
.
Subgraph-based reasoning by ours
𝑈𝑠𝑒𝑟7 →profile →car owner →tag →
C_3 →sale →𝐼𝑡𝑒𝑚6
Explanation
𝑈𝑠𝑒𝑟7 is a car owner and bought some car accessories such
as battery and auto wiper” in Channel 4, so system guesses
𝑈𝑠𝑒𝑟7 need some power tools to install these accessories, so
system recommends METC cordless drill to 𝑈𝑠𝑒𝑟7
introduce an attention-based diffusion mechanism for the generation of subgraphs, facilitating the construction of nuanced user representations, which is used for calculating the recommendation score with the item profiles. To evaluate the effectiveness of our proposed model, we conducted a series of experiments across three publicly available datasets and one proprietary dataset from METC. The empirical results unequivocally demonstrate that our model outperforms existing state-of-the-art models across all datasets. Importantly, the case studies underscore our model’s capacity to provide explicit, user-centric explanations for recommendations, effectively addressing specific user comments, relieving the "recommendation hallucination" effectively. Moreover, our framework unveils substantial potential for further optimization in terms of efficiency and opens new avenues for
research in recommendation systems employing Large Language Models. This underscores the promising intersection of advanced computational models and practical application domains, heralding a new era of intelligent, user-focused recommendation systems. We hope that our work will provide some inspiration for methods in the same scenario, particularly in the integration of Large Language Models and Knowledge Graphs, and offer strategic insights for market analysts. For instance, in the decision-making process, because the system gives high score for some key words in review, marketing analyst could pre-define these key words for user review, thereby enhancing the system’s robustness.
# REFERENCES
[1] Krisztian Balog and Filip Radlinski. 2020. Measuring recommendation explanation quality: The conflicting goals of explanations. In Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval. 329–338. [2] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multi-relational data. In NeurIPS, Vol. 26. [3] Ziheng Chen, Fabrizio Silvestri, Jia Wang, Yongfeng Zhang, Zhenhua Huang, Hongshik Ahn, and Gabriele Tolomei. 2022. Grease: Generate factual and counterfactual explanations for gnn-based recommendations. arXiv preprint arXiv:2208.04222 (2022). [4] Zhongwu Chen, Chengjin Xu, Fenglong Su, Zhen Huang, and Yong Dou. 2023. Incorporating structured sentences with time-enhanced bert for fully-inductive temporal relation prediction. In SIGIR. [5] Jiarui Feng, Yixin Chen, Fuhai Li, Anindya Sarkar, and Muhan Zhang. 2022. How powerful are k-hop message passing graph neural networks. Advances in Neural Information Processing Systems 35 (2022), 4776–4790. [6] Zhen Han, Peng Chen, Yunpu Ma, and Volker Tresp. 2020. Explainable subgraph reasoning for forecasting on temporal knowledge graphs. In International Conference on Learning Representations. [7] Marvin Hofer, Daniel Obraczka, Alieh Saeedi, Hanna Köpcke, and Erhard Rahm. 2023. Construction of knowledge graphs: State and challenges. arXiv preprint arXiv:2302.11509 (2023). [8] Xiaowen Huang, Quan Fang, Shengsheng Qian, Jitao Sang, Yan Li, and Changsheng Xu. 2019. Explainable interaction-driven user modeling over knowledge graph for sequential recommendation. In Proceedings of the 27th ACM international conference on multimedia. 548–556. [9] Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and Philip S. Yu. 2022. A survey on knowledge graphs: Representation, acquisition, and applications. IEEE Transactions on Neural Networks and Learning Systems 33, 2 (2022), 494–514. [10] Abhijeet Kumar, Abhishek Pandey, Rohit Gadia, and Mridul Mishra. 2020. Building knowledge graph using pre-trained language model for learning entity-aware relationships. In 2020 IEEE International Conference on Computing, Power and Communication Technologies (GUCON). IEEE, 310–315. [11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. In NeurIPS, Vol. 33. 9459–9474. [12] Bill Yuchen Lin, Xinyue Chen, Jamin Chen, and Xiang Ren. 2019. KagNet: Knowledge-aware graph networks for commonsense reasoning. In EMNLPIJCNLP. 2829–2839. [13] Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. [n. d.]. Learning entity and relation embeddings for knowledge graph completion. In Proceedings of the AAAI conference on artificial intelligence. [14] Junling Liu, Chao Liu, Renjie Lv, Kang Zhou, and Yan Zhang. 2023. Is chatgpt a good recommender? a preliminary study. arXiv preprint arXiv:2304.10149 (2023). [15] Ana Lucic, Maartje A Ter Hoeve, Gabriele Tolomei, Maarten De Rijke, and Fabrizio Silvestri. 2022. Cf-gnnexplainer: Counterfactual explanations for graph neural networks. In International Conference on Artificial Intelligence and Statistics. PMLR, 4499–4511. [16] Linhao Luo, Jiaxin Ju, Bo Xiong, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2023. Chatrule: Mining logical rules with large language models for knowledge graph reasoning. arXiv preprint arXiv:2309.01538 (2023). [17] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning on graphs: Faithful and interpretable large language model reasoning. In International Conference on Learning Representations. [18] Linhao Luo, Kai Liu, Dan Peng, Yaolin Ying, and Xiaofeng Zhang. 2020. A motif-based graph neural network to reciprocal recommendation for online dating. In Neural Information Processing: 27th International Conference, ICONIP 2020, Bangkok, Thailand, November 23–27, 2020, Proceedings, Part II 27. Springer, 102–114. [19] Weizhi Ma, Min Zhang, Yue Cao, Woojeong Jin, Chenyang Wang, Yiqun Liu, Shaoping Ma, and Xiang Ren. 2019. Jointly learning explainable rules for recommendation with knowledge graph. In The world wide web conference. 1210–1221. [20] Sijie Mai, Shuangjia Zheng, Yuedong Yang, and Haifeng Hu. 2021. Communicative message passing for inductive relation reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35. 4294–4302. [21] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. 2024. Unifying large language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and Data Engineering (2024). [22] Ciyuan Peng, Feng Xia, Mehdi Naseriparsa, and Francesco Osborne. 2023. Knowledge graphs: Opportunities and challenges. Artificial Intelligence Review 56, 11 (2023), 13071–13102. [23] Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. 2019. Language models as knowledge bases?. In EMNLP-IJCNLP. 2463–2473.
[24] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2012. Bpr: Bayesian personalized ranking from implicit feedback. arXiv preprint arXiv:1205.2618 (2012). [25] Corby Rosset, Chenyan Xiong, Minh Phan, Xia Song, Paul Bennett, and Saurabh Tiwary. 2020. Knowledge-aware language model pretraining. arXiv preprint arXiv:2007.00655 (2020). [26] Junshuai Song, Xiaoru Qu, Zehong Hu, Zhao Li, Jun Gao, and Ji Zhang. 2021. A subgraph-based knowledge reasoning method for collective fraud detection in e-commerce. Neurocomputing 461 (2021), 587–597. [27] Kai Sun, HuaJie Jiang, Yongli Hu, and BaoCai Yin. 2023. Substructure-aware subgraph reasoning for inductive relation prediction. The Journal of Supercomputing 79, 18 (2023), 21008–21027. [28] Yu Sun, Shuohuan Wang, Shikun Feng, Siyu Ding, Chao Pang, Junyuan Shang, Jiaxiang Liu, Xuyi Chen, Yanbin Zhao, Yuxiang Lu, and et al. 2021. Ernie 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation. arXiv preprint arXiv:2107.02137 (2021). [29] Juntao Tan, Shuyuan Xu, Yingqiang Ge, Yunqi Li, Xu Chen, and Yongfeng Zhang. 2021. Counterfactual explainable recommendation. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 1784–1793. [30] Komal Teru, Etienne Denis, and Will Hamilton. 2020. Inductive relation prediction by subgraph reasoning. In International Conference on Machine Learning. PMLR, 9448–9457. [31] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, and et al. 2022. Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239 (2022). [32] Hongwei Wang, Fuzheng Zhang, Xing Xie, and Minyi Guo. 2018. Dkn: Deep knowledge-aware network for news recommendation. In Proceedings of the 2018 world wide web conference. 1835–1844. [33] Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhengyan Zhang, Zhiyuan Liu, Juanzi Li, and Jian Tang. 2021. Kepler: A unified model for knowledge embedding and pre-trained language representation. Transactions of the Association for Computational Linguistics 9 (2021), 176–194. [34] Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhengyan Zhang, Zhiyuan Liu, Juanzi Li, and Jian Tang. 2021. KEPLER: A unified model for knowledge embedding and pre-trained language representation. Transactions of the Association for Computational Linguistics 9 (2021), 176–194. [35] Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, and Tat-Seng Chua. 2019. Kgat: Knowledge graph attention network for recommendation. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 950–958. [36] Xiting Wang, Kunpeng Liu, Dongjie Wang, Le Wu, Yanjie Fu, and Xing Xie. 2022. Multi-level recommendation reasoning over knowledge graphs with reinforcement learning. In Proceedings of the ACM Web Conference 2022. 2098–2108. [37] Xiang Wang, Dingxian Wang, Canran Xu, Xiangnan He, Yixin Cao, and Tat-Seng Chua. 2019. Explainable reasoning over knowledge graphs for recommendation. In Proceedings of the AAAI conference on artificial intelligence, Vol. 33. 5329–5336. [38] Xu Wang, Shuai Zhao, Bo Cheng, Jiale Han, Li Yingting, Hao Yang, Ivan Sekulic, and Guoshun Nan. 2021. Integrating subgraph-aware relation and direction reasoning for question answering. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 7808–7812. [39] Yikun Xian, Zuohui Fu, Shan Muthukrishnan, Gerard De Melo, and Yongfeng Zhang. 2019. Reinforcement knowledge graph reasoning for explainable recommendation. In Proceedings of the 42nd international ACM SIGIR conference on research and development in information retrieval. 285–294. [40] Xiaoran Xu, Wei Feng, Yunsheng Jiang, Xiaohui Xie, Zhiqing Sun, and Zhi-Hong Deng. 2020. Dynamically pruned message passing networks for large-scale knowledge graph reasoning. In International Conference on Learning Representations ICLR. [41] Ruichao Yang, Xiting Wang, Yiqiao Jin, Chaozhuo Li, Jianxun Lian, and Xing Xie. 2022. Reinforcement subgraph reasoning for fake news detection. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2253–2262. [42] Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. 2019. Gnnexplainer: Generating explanations for graph neural networks. Advances in neural information processing systems 32 (2019). [43] Fuzheng Zhang, Nicholas Jing Yuan, Defu Lian, Xing Xie, and Wei-Ying Ma. 2016. Collaborative knowledge base embedding for recommender systems. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. 353–362. [44] He Zhang, Bang Wu, Xingliang Yuan, Shirui Pan, Hanghang Tong, and Jian Pei. 2024. Trustworthy graph neural networks: Aspects, methods, and trends. Proc. IEEE (2024). [45] Zhiyuan Zhang, Xiaoqian Liu, Yi Zhang, Qi Su, Xu Sun, and Bin He. 2020. Pretrainkge: learning knowledge representation from pretrained language models. In EMNLP Finding. 259–266. [46] Zhanke Zhou, Yongqi Zhang, Jiangchao Yao, Bo Han, and et al. 2023. Less is more: One-shot subgraph reasoning on large-scale knowledge graphs. In The Twelfth International Conference on Learning Representations.
[47] Hongyin Zhu, Hao Peng, Zhiheng Lyu, Lei Hou, Juanzi Li, and Jinghui Xiao. 2023. Pre-training language model incorporating domain-specific heterogeneous knowledge into a unified representation. Expert Systems with Applications 215 (2023), 119369. [48] Yuqi Zhu, Xiaohan Wang, Jing Chen, Shuofei Qiao, Yixin Ou, Yunzhi Yao, Shumin Deng, Huajun Chen, and Ningyu Zhang. 2023. LLMs for knowledge graph
construction and reasoning: Recent capabilities and future opportunities. arXiv preprint arXiv:2305.13168 (2023). Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009
