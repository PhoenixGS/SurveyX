# Exploring User Retrieval Integration towards Large Language Models for Cross-Domain Sequential Recommendation
# ABSTRACT
Cross-Domain Sequential Recommendation (CDSR) aims to mine and transfer users’ sequential preferences across different domains to alleviate the long-standing cold-start issue. Traditional CDSR models capture collaborative information through user and item modeling while overlooking valuable semantic information. Recently, Large Language Model (LLM) has demonstrated powerful semantic reasoning capabilities, motivating us to introduce them to better capture semantic information. However, introducing LLMs to CDSR is non-trivial due to two crucial issues: seamless information integration and domain-specific generation. To this end, we propose a novel framework named URLLM, which aims to improve the CDSR performance by exploring the User Retrieval approach and domain grounding on LLM simultaneously. Specifically, we first present a novel dual-graph sequential model to capture the diverse information, along with an alignment and contrastive learning method to facilitate domain knowledge transfer. Subsequently, a user retrieve-generation model is adopted to seamlessly integrate the structural information into LLM, fully harnessing its emergent inferencing ability. Furthermore, we propose a domainspecific strategy and a refinement module to prevent out-of-domain generation. Extensive experiments on Amazon demonstrated the information integration and domain-specific generation ability of URLLM in comparison to state-of-the-art baselines. Our code is available at https://github.com/TingJShen/URLLM
# CCS CONCEPTS
CCS CONCEPTS • Information systems →Information retrieval; • Computing methodologies →Natural language generation.
• Information systems →Information retrieval; • Computing methodologies →Natural language generation.
# KEYWORDS
Cross-Domain Sequential Recommendation, Large Language Model, Cold-Start Recommendation
Cross-Domain Sequential Recommendation, Large Language Model, Cold-Start Recommendation
# 1 INTRODUCTION
Sequential Recommendation (SR), focused on suggesting the next item for a user based on their past sequential interactions to capture dynamic user preferences, has gained significant attention in commercial, social, and diverse scenarios [19, 50, 64]. However, SR methods within a single domain usually encounter the long-standing cold-start issue [41], i.e., it is challenging to perform personalized recommendations for users with few interaction records. To address the issue, Cross-Domain Sequential Recommendation (CDSR) has garnered considerable attention in the field of recommendation systems, aiming to mine and transfer users’ sequential preferences across different domains [12, 58]. Pioneer works like 𝜋-net [39] and PSJNet [44] focused on designing knowledge transfer modules to capture cross-domain user preferences. Followup works like MIFN [38] and DA-GCN [13] further borrowed the powerful strength of Graph Neural Networks (GNNs) to model the high-order relationship across domains. These methods have been demonstrated to be effective for the CDSR problem. Despite the achieved results, most previous works overlook the valuable semantic information buried in item features [1, 13, 63], leading to skewed user preferences. Recently, the powerful emergent capabilities [5] of Large Language Models (LLMs) have revolutionized the field of recommendation systems [8], which can absorb item text features and inject pre-trained common knowledge into recommendation systems. Meanwhile, they can generate recommendations based on user preferences and historical data, enabling interactive and explainable recommendations [9]. LLMs also offer the flexibility to design tuning strategies for specific subtasks, such as determining whether to recommend an item or integrating collaborative information. This integration aligns the sequential behaviors of users with the language space, creating new modalities for recommendation [30, 61]. Therefore, we are motivated by these encouraging capabilities to propose an LLM paradigm for the CDSR scenario.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d48f/d48f0a41-4fb6-425c-aa85-5cccea14ddc9.png" style="width: 50%;"></div>
Figure 1: An illustration of cold-start CDSA task along with the various forms of information, the domain-specific demand on information and generation. The line linking attributes represent the structural-semantic information.
However, harnessing the explicit capabilities of LLMs in the CDSR scenario is non-trivial. Two essential issues arise: (1) Seamless Information Integration: As shown in Figure 1, the CDSR task involves diverse formats of domain information, including collaborative and semantic information. Items exhibit intricate intrinsic structures respectively specified by the diverse information. To fully leverage the emergent capabilities of LLMs, it is crucial to integrate the structured information into LLMs in a seamless manner. (2) Domain-Specific Generation. CDSR requires equipping the model with domain-specific information and constraining the generation process to remain confined within a specific domain. However, despite instructing the model to generate items within a particular domain, the uncontrollable nature of LLMs leads to 2% to 20% of generated content belonging to other domains. Figure 1 illustrates an optimal pipeline for domain-specific generation, wherein distinct information aligns with specific generation processes. Regarding issue (1), some efforts have been made to integrate collaborative and semantic information with LLMs. Pioneer works like UniSRec [21] and CTRL [34] proposed to integrate semantic information through a discriminative LLM like BERT. As for collaborative information, BIGRec [3] adopts a post-processing approach, integrating through statistical results. RecInterpreter [57], LLaRA [35], and CoLLM [65] tend to capture collaborative information using an external traditional model and map it into the input token embedding space of LLM. However, these methods fail to seamlessly incorporate the emergent language reasoning capabilities of LLMs [33, 54], as evidenced by the misalignment between the pre-trained knowledge representation of LLMs and
their embedding representations. This oversight is particularly evident in the absence of a seamless integration of collaborative information and structural-semantic information. Consequently, the integration of diverse information continues to pose a substantial challenge. In addressing issue (2), existing retrieval methodologies for LLMs, such as ICL [33] and Agent-based LLM [23, 62], primarily yield common sense retrievals rather than domain-specific information. Furthermore, while the existing LLM-based recommendation model [4] [3] [47] acknowledge the cross-domain capabilities of LLMs, they fail to impose constraints on the generation process of LLMs. This lack of restriction results in out-of-domain generations, significantly undermining the performance of CDSR. Towards these challenges, in this paper, we propose a novel framework named URLLM, addressing the cold-start issue by exploring a novel paradigm of User-Retrieval approach and domain grounding on LLM simultaneously. Firstly, we propose a dual graph sequence modeling model combining alignment and contrastive learning on an item-attribute graph and three item-item domain sequence graphs, aiming to model collaborative and structuralsemantic information. Subsequently, we present a KNN user retriever to retrieve relevant user information for LLM. By leveraging LLM’s extensive reasoning and few-shot learning capabilities in the integration of collaborative user information, we can achieve a seamless integration of diverse information. Finally, in order to preserve the domain-specific nature of both the input and generated responses in the LLM, we propose a domain differentiation strategy for user retrieval modules. Additionally, we introduce a refining mechanism to enhance the outputs of the LLM. Extensive experiments on two datasets, including movie-game and art-office on Amazon following [21] have demonstrated the effectiveness of the proposed framework. Moreover, through analysis of experimental results, we (1) recognize that the improvement brought by the types of information integrated into components of URLLM positively correlated with the most crucial information in the dataset. (2) find there existing positive relation between the hit rate of retrieved users and the performance of the model. The main contributions could be summarized as follows:
• To our best knowledge, we are the first to study CDSR from a new perspective on the user retrieval paradigm with seamless information integration and domain-specific generation. • We develop a user retrieval bounded interaction paradigm between dual graph sequence modeling models and LLM. With the aid of the module, we can integrate structural-semantic and collaborative information into LLM in a seamless manner. • We introduce a domain differentiation strategy for user retrieval modules and a refinement module for the generated items of the LLM. The proposed module ensures that the integrated user information and generation are tailored to specific domains, aiming for domain-specific generation. • Extensive experiments on two public datasets and ablation analysis validate that our URLLM framework unequivocally affirms the information integration and domain-specific generation ability of our proposed framework.
ring User Retrieval Integration towards Large Language Models for Cross-Domain Sequential Recom
# 2 RELATED WORK 2.1 Sequential Recom
# 2 RELATED WORK 2.1 Sequential Recom
# 2 RELATED WORK
# 2.1 Sequential Recommendation
Sequential recommendation is a technique that aims to delve into and understand users’ interest patterns by analyzing their historical interactions. Initially, techniques such as markov chain and matrix factorization were employed [17]. However, with the emergence of neural networks, deep learning approaches like GRU4Rec [18], Hypersorec [49] and Caser [46] were introduced to improve recommendation accuracy, while some efforts are made with clustering [15], denoising [14] or data regeneration [59]. Another notable technique in sequential recommendation is the attention mechanism. SASRec [27] and APGL4SR [60], for instance, utilize selfattention to independently learn the impact of each interaction on the target behavior. On the other hand, BERT4Rec [42] incorporates bi-directional transformer layers after conducting pre-training tasks. In recent years, graph neural networks (GNNs) have gained attention for their ability to capture higher-order relationships among items. GCE-GNN [52] constructs local session graphs and leverages information from other sessions to create a dense global graph for modeling the current session. SR-GNN [? ] employs gated GNNs in session graphs to capture complex item transitions. To address the issue of data sparsity, contrastive mechanisms have been adopted in some works. CL4SRec [56] and CoSeRec [37] propose data augmentation approaches to construct contrastive tasks, which help alleviate the sparsity problem.
# 2.2 Cross-Domain Sequential Recommendation
Cross-Domain Sequential Recommendation (CDSR) aims to improve recommendation performance for tasks involving items from different domains. Pioneering works in this field include 𝜋-Net [39] and PSJNet [44], which employ sophisticated gating mechanisms to transfer single-domain information. CD-SASRec [1] extends SASRec to the cross-domain setting by integrating the source-domain aggregated vector into the target-domain item embedding. DAGCN [13], a GNN-based model, constructs a domain-aware graph to capture associations among items from different domains. Hybrid models that combine various techniques to capture item dependencies within sequences and complex associations between domains have also been proposed. RecGURU [31] introduces adversarial learning to unify user representations from different domains into a generalized representation. UniSRec [21] leverages item texts to learn more transferable representations for sequential recommendation. In comparison, C2DSR [6] employs a graphical and attentional encoder to capture item relationships. It utilizes two sequential objectives, in conjunction with a contrastive objective, to facilitate the joint learning of single-domain and cross-domain user representations, achieving significant progress.
# 2.3 LLM-based Recommendation System
Large Language Models (LLMs) have been widely adopted as recommender systems to leverage item text features and enhance recommendation performance [11, 36, 55]. The majority of existing LLM-based recommenders operate in a tuning-free manner, utilizing pretrained knowledge to generate recommendations for the next item [45, 51]. For instance, CHAT-REC [10] employs ChatGPT
to grasp user preferences and enhance interactive and explainable recommendations. GPT4Rec [32] utilizes GPT-2 to generate hypothetical "search queries" based on a user’s historical data, which are then queried using the BM25 search engine to retrieve recommended items. Another research direction in LLM-based recommendation focuses on designing tuning strategies for specific subtasks. TALLRec [4], for example, employs instruction-tuning to determine whether an item should be recommended. BIGRec [3] adopts a post-processing approach where recommendations are initially generated using LLM and then integrated with collaborative information through an ensemble method. RecInterpreter [57] and LLaRA [35] proposes a novel perspective by considering the "sequential behaviors of users" as a new modality for LLMs in recommendation, aligning it with the language space. CoLLM [65] captures collaborative information using an external traditional model and maps it into the input token embedding space of LLM, creating collaborative embeddings for LLM utilization. Despite the significant progress achieved in the field of SR through traditional or LLM-based methods, these approaches tend to focus on a limited perspective of information formats. Consequently, there is an underutilization of information pertaining to the coldstart feature of CDSR, which is the primary focus of this paper. Moreover, although some previous works such as Tallrec [4], BIGRec [3], and LLM-Rec [47] recognize the cross-domain capabilities of LLMs, there is currently a lack of an LLM-based structure specifically optimized for CDSR.
# 3 PRELIMINARY
# 3.1 Problem Definition
In this work, we consider a general CDSR scenario, where users in the user set 𝑈= {𝑢1,𝑢2, ...,𝑢|𝑈|} have interactions with two product sets 𝑋= {𝑥1,𝑥2, ...,𝑥|𝑋|} and 𝑌= {𝑦1,𝑦2, ...,𝑦|𝑌|}. Each user 𝑢∈𝑈has an interaction sequence 𝑆𝑢= [𝑖1,𝑖2, ...,𝑖𝑡, ...,𝑖𝑘] representing the chronological order of items in two domains. The primary objective of CDSR is to train the model 𝑀(𝑋,𝑌,𝑈,𝑆) to predict the subsequent product𝑖𝑘+1 ∈𝑋∪𝑌, where𝑆= {𝑆1,𝑆2, ...,𝑆|𝑈|} denotes the cross-domain interaction sequence.
In this paper, we use LLMs as the recommendation model 𝑀for answer formulation. The LLMs should be fine-tuned to adeptly adapt to the data distribution and domain knowledge relevant to specific downstream tasks. The fine-tuning process involves meticulously crafting instruction data to guide the model’s output scope and format, as detailed below:
(1)
where Φ is the parameters of LLM to be optimized,𝑇is the training set, 𝑎𝑖,𝑡is the 𝑡-th token of the generated answer word, and 𝑥is the input context which contains an instruction and a query question.
# 4 METHODOLOGY
To harness the inferential capabilities of Large Language Models (LLMs) for blending LLMs with traditional models, we present
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d013/d013a5d8-0bf2-4959-9efe-b20c2da1711f.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: The overall framework of URLLM. The component on the left showcases exemplary prompts employed in graph construction and the similar user-augmented LLM module. On the right, the process is delineated, wherein the expansiv reasoning and few-shot analogy capabilities of the LLM are harnessed, concomitantly integrating structured knowledge.</div>
URLLM illustrated in Figure 2. Initially, we introduce a Dual Graph Sequence-Modeling Model. This model employs an LLM-enhanced item-attribute graph and an item-item sequence graph to encapsulate collaborative and structural-semantic information. This type of information is challenging to model with an isolated LLM. Then, we adopt a User Retrieve-Generation Model to retrieve the most similar users from the target domain and fuse structured text with collaborative information seamlessly into LLM. Finally, to make the output of our model directly match the real item in the correct domain, we combine BM25 [40] and the Dual Graph SequenceModeling Model to refine the generated prediction.
# 4.1 Dual Graph Sequence Modeling Model
Aiming to model collaborative and semantic information, the construction of this model contains a graph construction module, a graph alignment module, and a contrastive self-attention module.
4.1.1 Graph Construction Module. In order to capture valuable collaborative and semantic information, including their structural relationships, which can be effectively represented as a graph, we construct a graph based on rules due to the absence of such data in our dataset. The graph construction process comprises two modules: item-attribute graph construction and item-item graph construction. These modules establish collaborative relationships among items and structural-semantic relationships, respectively. For item-attribute graph construction, the Chain-of-Thought (COT) method [54] is employed to utilize an LLM for item description.
Subsequently, the model summarizes this description to generate output attributes for the product. For instance, in Figure 2, for the product I=Tinker Bell, the model provides an introduction denoted as 𝐿𝐿𝑀(𝐼) = 𝐼𝑛𝑡𝑟𝑜and generates the attribute reusing 𝐼as 𝐴𝐼= 𝐿𝐿𝑀(𝐼𝑛𝑡𝑟𝑜, 𝐼). The undirected attribute graph, denoted as 𝐺𝐴, is constructed by linking items with attributes formulated below: 𝐺𝐴= (𝑉, 𝐸),𝑉= 𝑋∪𝑌∪𝐴, 𝐸= {(𝑣,𝑡)|𝑣∈𝑋∪𝑌,𝑡∈𝐴}. (2) where 𝑉and 𝐸denotes the vector and edges of graph 𝐺𝐴. For item-item graph construction, we construct 𝐺𝑆= 𝐺𝑋+𝑌, 𝐺𝑋 and 𝐺𝑌respectively inspired by [7]. Firstly, given user interaction sequence𝑆𝑢= [𝑖1, ...,𝑖𝑘], we split the interaction sequence into 𝑆𝑢= 𝑆𝑢,𝑆𝑋𝑢= 𝑆𝑢∩𝑋,𝑆𝑌𝑢= 𝑆𝑢∩𝑌. We then construct the directed item-item graph 𝐺𝑆,𝐺𝑋,𝐺𝑌by linking items before and after using the formula below: 𝐺𝑆= (𝑉, 𝐸),𝑉= 𝑋∪𝑌, 𝐸= {(𝑣,𝑡)|∃𝑗, 𝑣= 𝑖𝑗,𝑡= 𝑖𝑗+1}, (3) 𝐺𝑋= (𝑉, 𝐸),𝑉= 𝑋, 𝐸= {(𝑣,𝑡)|∃𝑗, 𝑣= 𝑖𝑋 𝑗,𝑡= 𝑖𝑋 𝑗+1}, (4) 𝐺𝑌= (𝑉, 𝐸),𝑉= 𝑌, 𝐸= {(𝑣,𝑡)|∃𝑗, 𝑣= 𝑖𝑌 𝑗,𝑡= 𝑖𝑌 𝑗+1}, (5) where 𝑉denotes the vector set, 𝐸denotes the edge set, 𝑖𝑋and 𝑖𝑌
(2)
(4)
(5)
+ where 𝑉denotes the vector set, 𝐸denotes the edge set, 𝑖𝑋and 𝑖𝑌 denotes item in 𝑆𝑋𝑢and 𝑆𝑌𝑢.
4.1.2 Graph Alignment Module. Recognizing the deficiency of differential knowledge across distinct domains in the existing itemattribute graph, it becomes imperative to investigate the acquisition of domain transfer information to enhance the item-attribute graphs. Domain transfer, in turn, enables the extraction of intricate
intrinsic structures that are specifically specified by diverse information sources. However, the division of the item-attribute graph alone does not provide distinct information. Therefore, in this section, we propose a Graph Neural Network (GNN) that incorporates an alignment loss function to align and integrate these fragmented pieces of information. Given four graphs 𝐺𝐴,𝐺𝑆,𝐺𝑋,𝐺𝑌, We first initialize the graph embedding with 𝐸0 𝐴∈R|𝐸0 𝐴|×𝑑𝑎, 𝐸0 𝑆∈R|𝐸0 𝑆|×𝑑, 𝐸0 𝑋∈R|𝐸0 𝑋|×𝑑, 𝐸0 𝑌∈ R|𝐸0 𝑌|×𝑑. Then, given these adjacency matrix 𝐴𝐴,𝐴𝑆,𝐴𝑋,𝐴𝑌, we construct 𝑙-layer GNN with output denoted as 𝐸𝑙as below:
(6)
where 𝑁𝑜𝑟𝑚(·) represent the row-normalized function, 𝐸𝑖−1 represents the current graph convolutional layer and 𝐸𝑖represents the next layer. Then, to fully encapsulate the graphical information across layers, we average the graph embeddings 𝐸gaining ˆ𝐺that yield item representations as:
(7)
We apply this procedure to 𝐺𝐴,𝐺𝑆,𝐺𝑋,𝐺𝑌separately gaining ˆ 𝐺𝐴, ˆ 𝐺𝑆, ˆ 𝐺𝑋, ˆ 𝐺𝑌, each with its corresponding adjacency matrix 𝐴and initialized graph embedding 𝐸0. Then, To resolve differences in input and output dimensions in alignment, we design linear projection models 𝐿, 𝐿𝑋, 𝐿𝑌with 𝑑𝑎as input dim and 𝑑as output dim to reform the hidden representation. Finally, the alignment loss function is designed below to transfer item-attribute graph knowledge of domains 𝑋and 𝑌:
We then concat them together as the final representation 𝑅of the item, gaining𝑅𝑆= 𝐶𝑜𝑛𝑐𝑎𝑡(𝐿( ˆ 𝐺𝐴), ˆ 𝐺𝑆), 𝑅𝑋= 𝐶𝑜𝑛𝑐𝑎𝑡(𝐿𝑋( ˆ 𝐺𝐴), ˆ 𝐺𝑋), 𝑅𝑌= 𝐶𝑜𝑛𝑐𝑎𝑡(𝐿𝑌( ˆ 𝐺𝐴), ˆ 𝐺𝑌). With the alignment above the itemattribute and item-item graphs are modelized symmetrical and are successfully aligned to specific domains.
4.1.3 Self-attention Contrastive Sequential Module. Having successfully obtained dual graph item representations from the aforementioned models, our objective is to capture users’ sequential preferences based on their interaction sequences. This endeavor is crucial to promote the retrieval of analogous user behaviors, thereby ensuring the optimal utilization of the few-shot capabilities of LLMs. Similar to SASRec [26], we employ a multi-head self-attention layer and point-wise feed-forward layer to distinctly capture user preferences and enhance retrieval precision. The formal definition of sequence modeling is provided below:
𝐻𝑆= 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛𝑆(𝑆𝑢, 𝑅𝑆), 𝐻𝑋= 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛𝑋(𝑆𝑋 𝑢, 𝑅𝑋),
𝐻𝑆= 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛𝑆(𝑆𝑢, 𝑅𝑆), 𝐻𝑋= 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛𝑋(𝑆𝑋 𝑢, 𝑅𝑋), 𝐻𝑌= 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛𝑌(𝑆𝑌 𝑢, 𝑅𝑌),
Denoting user preferences in the full domain, domain X, and domain Y as 𝐻𝑆, 𝐻𝑋, 𝐻𝑌respectively, the training target of the sequential module focuses on predicting the next model. We incorporate a discriminator 𝐷𝑋, 𝐷𝑌with a softmax function to assign
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/81a4/81a45dca-9577-4d41-96d6-a4629413970c.png" style="width: 50%;"></div>
Figure 3: The example case illustrates the importance of LLM inferencing and similar user retrieval.
<div style="text-align: center;">Figure 3: The example case illustrates the importance of LLM inferencing and similar user retrieval.</div>
# scores to each item and then maximize the score for the groundtruth answer. The training loss is defined as follows: �
L𝑡= �−𝑙𝑜𝑔(𝑃(𝑖𝑘|𝐷𝑋(𝐻𝑆+ 𝐻𝑋))) 𝑖𝑘∈𝑋 −𝑙𝑜𝑔(𝑃(𝑖𝑘|𝐷𝑌(𝐻𝑆+ 𝐻𝑌))) 𝑖𝑘∈𝑌.
(10)
where 𝑖𝑘denotes the item to be recommended. To handle the negative transfer challenge on Cross-Domain Sequential Recommendation (CDSR), we adopt sequence corruption and random noise integration to gain negative sequence samples. The formalized definition is demonstrated below:
ˆ 𝐻𝑋= 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛𝑋( ˆ𝑆𝑋, 𝑅𝑋) + Δ, ˆ 𝐻𝑌= 𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛𝐵( ˆ𝑆𝑌, 𝑅𝑌) + Δ, (1
(11)  is a
where ˆ𝑆𝑋and ˆ𝑆𝑌randomly replaces X and Y domain items. Δ is a random noice with ∥Δ∥2 = 𝜖and Δ = ¯Δ ⊙𝑠𝑖𝑔𝑛(e𝑖), ¯Δ ∈R𝑑(0, 1) The contrastive training loss is denoted as below:
L𝑐𝑡= Σ(−𝑙𝑜𝑔(𝐷𝑋(𝐻𝑆+ 𝐻𝑋)) + (1 −𝐷𝑋(𝐻𝑆+ ˆ 𝐻𝑌))) + Σ(−𝑙𝑜𝑔(𝐷𝑌(𝐻𝑆+ 𝐻𝑌)) + (1 −𝐷𝑌(𝐻𝑆+ ˆ 𝐻𝑋))) .
(12)
to maximize the infomax objective between domain preferences A and B. The final optimization loss is as follows:
L = L𝑡+ 𝜆L𝑐𝑡+ 𝛾L𝑎𝑙
(13)
By combining contrastive alignment methodology for dual graph inherent knowledge preferences 𝐻𝑆, 𝐻𝐴, 𝐻𝐵, which is hard for LLM to model. Furthermore, the dual graph sequence-modeling model is also a recommendation model with answer 𝐼1 = [𝑖1, ...,𝑖|𝐼|], and the answer can be utilized to refine LLM’s generation afterward.
# 4.2 User Retrieve-Generation Model
In the preceding section, we obtained high-quality structured user preferences. However, LLM cannot directly leverage this representation information. In order to fully exploit the emergent capabilities of LLMs, such as few-shot learning and language inferencing, we need a model that translates the representation information into textual information, preferably in the form of similar example data. Therefore, the following retrieve-augmented-generation contains a KNN retriever, a LoRA-tuned LLM augmented by the target user and retrieved user interactions, and an answer refinement structure.
4.2.1 KNN retriver. We employ a KNN retrieval model to query the training users using 𝐾𝑁𝑁(𝑢) to retrieve its k-nearest neighbors, denoted as N, based on a distance function 𝑑(·, ·), specifically using the inner product, as they are already normalized beforehand. The formal retrieval procedure is presented below:
𝐸′ 𝑋= 𝐻𝑆+ 𝐻𝑋, 𝐸′𝑌= 𝐻𝑆+ 𝐻𝑌 𝑈𝑟= 𝐾𝑁𝑁(𝑢,𝑈, 𝐸) �
(14)
where 𝑈denotes the user in the training set, 𝑈𝑟denotes the retrieved users.
4.2.2 User Retrieval Augmented LLM. After gaining high-quality similar users, we then integrate this high-quality knowledge using few-shot learning, proven to be one of the most effective ways to enhance the emergent ability of LLM. Specifically, we form the prompt as ‘There are similar users who bought these items:’ with 𝑆𝑈𝑟= {𝑆𝑢1, ...,𝑆𝑢𝑘},𝑢1, ...,𝑢𝑘∈𝑈𝑟, and ‘The user has bought these items:’ with 𝑆𝑢. The actual prompt is illustrated in prompt II, Figure 2. However, there exists a distribution shift between original instruction tuning on the CDSR task and similar user integration CDSR task. We adopt the same KNN retrieval model on training users but choose from top-2 instead of top-1 (top-1 is the same user) to construct training prompts. Suppose the prompt is denoted as 𝑃,
Algorithm 1 Overall pseudo code of URLLM
Input: Product in two domain 𝑋and 𝑌, suppose 𝑋𝑖𝑑< 𝑌𝑖𝑑. Train-
ing user set 𝑈, user interaction sequence 𝑆, user 𝑢
Output: User’s next possible item list 𝐼
1: Train 𝑀(𝑆𝑢,𝑋,𝑌) = 𝐸′
𝑈, 𝐸′𝑢, 𝐼1
⊲Prompt I is used here
2: Retrieve 𝑈𝑟= 𝐾𝑁𝑁(𝑢,𝑈, 𝐸), 𝑈𝑡= 𝐾𝑁𝑁(𝑈,𝑈, 𝐸)
3: Train 𝐿𝐿𝑀with 𝑈𝑡
⊲Prompt II is used here
4: 𝐼2 = 𝐿𝐿𝑀(𝑈𝑟,𝑆𝑢)
5: 𝑚𝑎𝑥𝐼= 𝑚𝑎𝑥{𝐼2[0 : 𝑚]},𝑚𝑖𝑛𝐼= 𝑚𝑖𝑛{𝐼2[0 : 𝑚]}
6: 𝐼= 𝐼2
7: if 𝑚𝑎𝑥𝐼> 𝑋𝑖𝑑in recommend domain A then
8:
𝐼= 𝐼1
9: end if
10: if 𝑚𝑖𝑛𝐼> 𝑌𝑖𝑑in recommend domain B then
11:
𝐼= 𝐼1
12: end if
<div style="text-align: center;">Table 1: The detailed description and statistics of datasets.</div>
Scenarios
#Items
#Train
#Valid
#Test
Avg.length
Movie
71067
35941
1775
3601
4.095
Game
112233
3.277
Art
18639
16000
1154
2000
6.386
Office
19757
8.263
# the optimization of user-integration tuning loss is as follows:
# the optimization of user-integration tuning loss is as follows:
L𝑟= 𝑚𝑎𝑥Φ𝐿 ∑︁ 𝑝∈𝑃 |𝑝| ∑︁ 𝑡=1 𝑙𝑜𝑔(𝑃Φ+Φ𝐿(𝑝𝑡|𝑝<𝑡)).
(15)
Φ𝐿is the LoRA parameters and we only update LoRA parameters during the training process. However, due to the uncontrollability of LLM outputs, even when prompting and training explicitly for the generation of products in domain 𝑋, there is still about 2% to 20% output of products from domain 𝑌during actual inference. Therefore, a subsequent answer refinement module is required to further optimize the outputs of the LLM.
4.2.3 Answer Refinement Structure. We first adopt a BM25 retrieval model to ground the space of recommendation language space to actual item space with 𝐼2. Subsequently, to stably identify whether LLM has influenced out-of-domain items, we consider the top-m grounding items. If one of the answers runs out of domain, we will consider adopting 𝐼1 from the dual graph sequence-modeling model. In our experiment, we set m=5 to gain an answer stably. The overall algorithm is delineated in algorithm 1. Finally, by combining a dual graph sequence-modeling model and a controllable answer selection model, we leverage the expansive reasoning and few-shot analogy capabilities of the LLM on integrating collaborative user information. This approach not only addresses the cold-start problem but also tackles the challenges of aligning cross-domain information in collaborative recommendation systems. To provide a clearer illustration of how different components of URLLM contribute to its performance, we present example cases in Figure 3.
# 5 EXPERIMENTAL SETTINGS
# 5.1 Datasets
To demonstrate the performance of our proposed model, we conduct experiments on two publicly available datasets from the Amazon platform [25]. This forms two distinct cross-domain scenarios: Movie-Game and Art-Office. The Movie-Game dataset, preprocessed by [2], is sparser and contains many cold-start users. The Art-Office dataset exhibits a more standardized distribution and is preprocessed by [20]. We further refine the latter by filtering users with fewer than 3 item interactions. Detailed descriptions and statistics for both datasets1 are provided in Table 1.
<div style="text-align: center;">Table 2: The overall performance of all baselines on the Movie-Game dataset.</div>
Baseline Type
Method
HR@1
HR@5
HR@10
HR@20
MRR
NG@1
NG@5
NG@10
NG@20
Traditional
Baselines
LightGCN
0.0033
0.0064
0.0099
0.0147
0.0054
0.0033
0.0063
0.0087
0.0105
SASRec
0.0013
0.0038
0.0049
0.0075
0.0029
0.0013
0.0025
0.0029
0.0035
CoSeRec
0.0036
0.0112
0.0174
0.0251
0.0078
0.0036
0.0081
0.0090
0.0116
𝐶2𝐷𝑆𝑅
0.0047
0.0124
0.0181
0.0268
0.0089
0.0047
0.0085
0.0104
0.0125
LLM-based
Baselines
UniSRec
0.0090
0.0220
0.0259
0.0287
0.0148
0.0090
0.0160
0.0173
0.0181
BIGRec
0.0067
0.0194
0.0305
0.0444
0.0168
0.0067
0.0148
0.0183
0.0233
GPT4Rec
0.0088
0.0322
0.0410
0.0515
0.0198
0.0088
0.0208
0.0234
0.0266
CoLLM
0.0101
0.0202
0.0354
0.0455
0.0171
0.0101
0.0146
0.0193
0.0219
Our Work
URLLM
0.0105
0.0333
0.0416
0.0522
0.0211
0.0105
0.0221
0.0248
0.0298
<div style="text-align: center;">Table 3: The overall performance of all baselines on the Art-Office datas</div>
Baseline Type
Method
HR@1
HR@5
HR@10
HR@20
MRR
NG@1
NG@5
NG@10
NG@20
Traditional
Baselines
LightGCN
0.0105
0.0225
0.0255
0.0325
0.0198
0.0104
0.0133
0.0163
0.0184
SASRec
0.0055
0.0120
0.0175
0.0235
0.0100
0.0055
0.0069
0.0084
0.0108
CoSeRec
0.0099
0.0217
0.0284
0.0357
0.0154
0.0099
0.0159
0.0181
0.0199
𝐶2𝐷𝑆𝑅
0.0155
0.0275
0.0320
0.0415
0.0229
0.0155
0.0219
0.0234
0.0257
LLM-based
Baselines
UniSRec
0.0145
0.0245
0.0375
0.0525
0.0237
0.0145
0.0201
0.0242
0.0256
BIGRec
0.0220
0.0315
0.0370
0.0480
0.0277
0.0220
0.0268
0.0286
0.0313
GPT4Rec
0.0165
0.0360
0.0435
0.0525
0.0262
0.0165
0.0263
0.0287
0.0311
CoLLM
0.0170
0.0347
0.0438
0.0493
0.0286
0.0170
0.0263
0.0295
0.0309
Our Work
URLLM
0.0270
0.0400
0.0485
0.0595
0.0355
0.0270
0.0353
0.0371
0.0399
# 5.2 Baselines
We compare the performance of URLLM with state-of-the-art traditional and LLM-based CDSR methods to showcase its effectiveness. The baselines are as follows:
• LightGCN [16] simplifies the design of Graph Convolution Networks for collaborative filtering by focusing on the essential component of neighborhood aggregation. • SASRec [26] uses a causal attention mechanism to model sequential patterns. • CoSeRec [37] introduces two new informative augmentation operators that leverage item correlations with contrastive sequence. • 𝑪2𝑫𝑺𝑹[7] captures user preferences by simultaneously leveraging intra- and inter-sequence item relationships throug contrasive item-item graphs.
• LightGCN [16] simplifies the design of Graph Convolution Networks for collaborative filtering by focusing on the essential component of neighborhood aggregation. • SASRec [26] uses a causal attention mechanism to model sequential patterns. • CoSeRec [37] introduces two new informative augmentation operators that leverage item correlations with contrastive sequence. • 𝑪2𝑫𝑺𝑹[7] captures user preferences by simultaneously leveraging intra- and inter-sequence item relationships throug contrasive item-item graphs.
# 5.2.2 LLM-based Baselines.
• UniSRec [20] incorporates a lightweight item encoding architecture and employs contrastive pre-training tasks to learn transferable representations across domains. • GPT4Rec-LLaMA2 [32] uses BM25 grounding method to refine answer into actual item space. We replaced the GPT2 with LLaMA2 to ensure a fair comparison.
1The processed dataset will be open once accepted.
• BIGRec [3] grounds recommendation space of tuned LLM into real item space by incorporating statistical information. • CoLLM [65]2 captures collaborative information using an external traditional model and maps it into LLM by adapter.
# 5.3 Evaluation Metrics
Following previous works [26, 43, 66], we leverage the leave-oneout method to calculate the recommendation performance. Besides, we adopt the whole item set as the candidate item set during evaluation to avoid the sampling bias of the candidate selection [29]. Then, we evaluate the Top-K recommendation performance by Mean Reciprocal Rank (MRR) [48], Normalized Discounted Cumulative Gain (NDCG) [24] and Hit Rate (HR) [53].
# 5.4 Implementation Details
The instruction-tuning and model inference are conducted on 4 Tesla A100 40G GPUs, which takes approximately 24 hours for training stage and 1 hour for inferencing. On our dual graph sequencemodeling model, we set 𝑑𝑠= 128 and 𝑑𝑎= 32 according to their scale and sparsity. 𝛾and 𝜆are all set to 0.3 for balance. Across all generative LLMs, we finetune LLaMA2-7B-chat with LoRA[22] with LoRA-rank=8 and LoRA-alpha=16 using Adam[28] in a default learning rate of 1e-4. In the item attribute gaining part, in
2We transformed the item rating model into an item recommendation model by providing 5000 candidates and ranking them based on scores.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6400/6400dead-1542-428b-907b-0175578cfdb8.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Performance comparison is conducted in warm (left) and cold (right) scenarios for Movie-Game and Art-Office. "UC" denotes the substitution of the retrieval model of URLLM with 𝑪2𝑫𝑺𝑹, and "w/o R" designates URLLM without user retrieval.</div>
Method
HR1
HR5
HR10
HR20
MRR
NG1
NG5
NG10
NG20
w/o LLM
0.0069
0.0091
0.0108
0.0127
0.0082
0.0069
0.0080
0.0085
0.0090
w/o retrieve
0.0088
0.0322
0.0410
0.0515
0.0197
0.0088
0.0208
0.0234
0.0266
w/o graph alignment
0.0102
0.0331
0.0414
0.0522
0.0210
0.0102
0.0208
0.0243
0.0268
w/o answer refine
0.0105
0.0302
0.0410
0.0510
0.0203
0.0105
0.0206
0.0241
0.0267
w/o domain-specific retrieval
0.0074
0.0197
0.0269
0.0366
0.0143
0.0074
0.0131
0.0155
0.0180
URLLM-SASRec
0.0099
0.0308
0.0409
0.0505
0.0201
0.0099
0.0208
0.0244
0.0266
URLLM-𝐶2𝐷𝑆𝑅
0.0089
0.0313
0.0402
0.0489
0.0190
0.0089
0.0201
0.0229
0.0251
URLLM
0.0105
0.0333
0.0416
0.0522
0.0211
0.0105
0.0221
0.0248
0.0298
order to gain extra information, we use gpt-3.5-turbo-061 to gain introduction and attribute of the item.
# 6 RESULTS AND ANALYSIS 6.1 Overall Performance
# 6 RESULTS AND ANALYSIS
# 6.1 Overall Performance
In this subsection, we compare the overall performance of all methods, which is presented in Table 2 and Table 3. We can draw the following conclusions from the results: 1). LLM-based CDSR methods generally surpass traditional methods, particularly in the sparser Movie-Game domain. This highlights the LLMs’ ability to perform extensive reasoning and few-shot learning. 2). Discriminative LLMbased method UniSRec underperforms compared to other generative LLMs, suggesting emergent reasoning capabilities in generative models. 3). In the Art-Office domain, LLMs incorporating collaborative information (BIGRec, CoLLM) outperform methods without such information (GPT4Rec), underscoring its importance in LLMs. The opposite trend exists in the sparse domain Movie-Game, indicating that inappropriate information integration has a detrimental effect on performance. 4). URLLM consistently outperforms all baselines, demonstrating its effectiveness. Its superiority over LLM-based baselines confirms the value of seamless information integration and domain-specific generation.
# 6.2 Analysis on warm/cold start scenario
URLLM seeks to retrieve similar users into LLM, aiming to integrate additional knowledge tackling cold-start scenarios of CDSR.
To evaluate the achievement of this goal, we conduct a detailed examination of the methods’ performance in both warm and cold-start scenarios. In particular, users are categorized into cold scenarios when the length of their domain-relevant interaction sequences is less than 3. Other users are categorized into warm scenarios. To showcase the efficacy of our model in cold-start situations, comparison among four models are compared in Figure 4. We can draw three pieces of information: 1). In comparison to the model devoid of the retrieval module (w/o R), URLLM exhibits an enhancement in cold-start scenarios across both datasets. This underscores the efficacy of the user-retrieval module in augmenting the model’s performance. Notably, in cold-start scenarios on the Art-Office dataset, our model surpasses its warm-start performance (0.0364 versus 0.0355 in MRR). This suggests that URLLM has the potential to convert cold-start scenarios into warm-start ones. 2). We observe that, within the context of the movie-game dataset’s warm-start setting, the performance of user retrieval based on 𝐶2𝐷𝑆𝑅is inferior to not performing any retrieval. This is attributed to𝐶2𝐷𝑆𝑅’s inability to model structural-semantic information, which consequently impairs its user retrieval capability. We hypothesize that this deficiency stems from the absence of semantic information modeling, introducing noise into LLMs, particularly under warm-start conditions. 3). URLLM’s efficacy in cold-start scenarios is demonstrably superior compared to the traditional𝐶2𝐷𝑆𝑅model. On the Movie-Game dataset, URLLM achieves a remarkable 3.6-fold improvement, while on the Art-Office dataset, the performance gap widens to a factor of 13. These substantial gains highlight URLLM’s effectiveness in
<div style="text-align: center;">Table 5: Ablation study on Art-Office dataset. URLLM-SASRec, URLLM-𝑪2𝑫𝑺𝑹denotes the substitution of the retrieval model of URLLM with SASRec or 𝑪2𝑫𝑺𝑹, simultaneously represent URLLM w/o item graph or URLLM w/o attribute graph.</div>
Method
HR1
HR5
HR10
HR20
MRR
NG1
NG5
NG10
NG20
w/o LLM
0.0265
0.0345
0.0430
0.0505
0.0322
0.0265
0.0308
0.0336
0.0355
w/o retriever
0.0165
0.0360
0.0435
0.0525
0.0262
0.0165
0.0263
0.0287
0.0311
w/o graph alignment
0.0235
0.0415
0.0475
0.0530
0.0322
0.0235
0.0324
0.0344
0.0357
w/o answer refine
0.0230
0.0415
0.0480
0.0560
0.0323
0.0230
0.0329
0.0349
0.0370
w/o domain-specific retrieval
0.0235
0.0315
0.0365
0.0455
0.0289
0.0235
0.0274
0.0292
0.0307
URLLM-SASRec
0.0175
0.0350
0.0420
0.0490
0.0263
0.0175
0.0267
0.0290
0.0307
URLLM-𝐶2𝐷𝑆𝑅
0.0215
0.0375
0.0455
0.0550
0.0302
0.0215
0.0309
0.0328
0.0349
URLLM
0.0270
0.0430
0.0485
0.0595
0.0355
0.0270
0.0353
0.0371
0.0399
<div style="text-align: center;">ble 6: Movie-Game CDSR performance replacing on user retrieval with candidates. The subscript model in the Method notes the application of the model for either user retrieval or the generation of top-k candidates.</div>
Table 6: Movie-Game CDSR performance replacing on user retrieval with candidates. The subscrip denotes the application of the model for either user retrieval or the generation of top-k candidates.
Method
UHR(10−3)
HR@1
HR@5
HR@10
HR@20
MRR
NG@1
NG@5
NG@10
NG@20
𝐶𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒𝐶2𝐷𝑆𝑅
0.4700
0.0083
0.0277
0.0369
0.0450
0.0180
0.0083
0.0183
0.0213
0.0232
𝑅𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙𝐶2𝐷𝑆𝑅
0.3694
0.0089
0.0313
0.0402
0.0489
0.0190
0.0089
0.0201
0.0229
0.0251
𝐶𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒𝑈𝑅𝐿𝐿𝑀
0.6350
0.0089
0.0275
0.0378
0.4580
0.0184
0.0089
0.0185
0.0219
0.0240
𝑅𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙𝑈𝑅𝐿𝐿𝑀
0.5651
0.0105
0.0333
0.0416
0.0522
0.0211
0.0105
0.0221
0.0248
0.0298
augmenting the model’s capabilities for dealing with sparse data and limited user interaction.
# 6.3 Ablation study
To demonstrate the effectiveness of each component, we conduct an ablation study to compare URLLM with seven variants: (1) w/o LLM: removing LLM to use only dual graph sequence modeling model to recommend; (2) w/o retrieve: removing retriever to adopt LLaMA2 itself; (3) w/o graph alignment: removing the alignment loss by setting 𝛾= 0; (4) w/o answer refine: generating without domain-specific refine; (5) w/o domain-specific retrieval: retrieving users without domain differentiation strategy; (6) URLLM-SASRec: replacing dual graph sequence modeling model with SASRec; (7) URLLM-𝐶2𝐷𝑆𝑅: replacing dual graph sequence modeling model with 𝐶2𝐷𝑆𝑅. The results are presented in Table 4 and Table 5. The results demonstrate that each component within our framework plays a positive role in enhancing overall performance. Notably, for the Art-Office data, removing the retrieval and answer refinement module significantly decreased recommendation effectiveness. This highlights the critical role of collaborative information, employed in both retrieval and refining. Conversely, in the sparser Movie-Game dataset, the LLM and attribute graph exhibited the most significant contributions, suggesting that semantic information is paramount for dealing with cold-start scenarios. Furthermore, when we dissociate the domain-specific retrieval method or substitute the Dual Graph Sequence Modeling Model with SASRec and 𝐶2𝐷𝑆𝑅—which fail to generate user representation in specific domains—the model’s performance deteriorates due to the loss of crucial collaborative information and the integration of noise. The answer refinement model also affects the results of the model as shown in Table 4 and Table 5, which shows the importance of domain-specific generation.
# 6.4 Analysis on integration of LLM
6.4.1 Analysis on quality of user integration. In general, superiorperforming models tend to produce more precise user representations during training, which in turn leads to improved user retrieval results. Nevertheless, there appears to be an inconsistency between the model performance depicted in Table 2 and the retrieval outcomes presented in Table 4. While 𝐶2𝐷𝑆𝑅outperforms SASRec in terms of retrieval performance, its effectiveness diminishes when used as a replacement for the search model than SASRec. This observation prompts us to explore the impact of the retrieval result quality on recommendation performance. To quantitatively assess performance, we employ User Hit Rate (UHR) dividing the hit rate of the user interaction by the interaction length for retrieval quality and Mean Reciprocal Rank (MRR) for recommendation effectiveness. We further select a diverse subset of Art-Office examples, encompassing both higher-quality and lowerquality retrievals. Figure 5 illustrates the observed relationship between retrieval result quality and recommendation performance. Notably, this experiment is not conducted on the Movie-Game dataset due to its sparsity, which leads to unstable retrieval results and compromises the reliability of sampled outcomes. Figure 5 reveals that URLLM outperforms 𝐶2𝐷𝑆𝑅, suggesting its superior ability to model collaborative information. However, the close alignment of their curves indicates that model choice becomes less impactful. Notably, the positive correlation between MRR and UHR underscores the influence of retrieval quality on performance. Therefore, we can now address the initial question posed in this subsection: even though a stronger model leads to 𝐻𝑅𝑆𝐴𝑆𝑅𝑒𝑐= 0.0067 < 𝐻𝑅𝐶2𝐷𝑆𝑅= 0.0097, the length of retrieved user 𝑙𝑆𝐴𝑆𝑅𝑒𝑐= 14.41 < 𝑙𝐶2𝐷𝑆𝑅= 26.26 resulting in 𝑈𝐻𝑅1/𝑈𝐻𝑅2 = 1.24 > 1, this causes a reduction of the 𝐶2𝐷𝑆𝑅retrieval model.
<div style="text-align: center;">Table 7: Art-Office CDSR performance replacing on user retrieval with candidates. The subscript model in the Method denotes the application of the model for either user retrieval or the generation of top-k candidates.</div>
Table 7: Art-Office CDSR performance replacing on user retrieval with candidates. The subscript mode the application of the model for either user retrieval or the generation of top-k candidates.
Method
UHR(10−3)
HR@1
HR@5
HR@10
HR@20
MRR
NG@1
NG@5
NG@10
NG@20
𝐶𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒𝐶2𝐷𝑆𝑅
1.5992
0.0185
0.0325
0.0370
0.0440
0.0257
0.0185
0.0260
0.0274
0.0292
𝑅𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙𝐶2𝐷𝑆𝑅
0.8783
0.0215
0.0375
0.0455
0.0550
0.0302
0.0215
0.0309
0.0328
0.0349
𝐶𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒𝑈𝑅𝐿𝐿𝑀
2.1511
0.0200
0.0340
0.0495
0.0455
0.0274
0.0200
0.0277
0.0295
0.0310
𝑅𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙𝑈𝑅𝐿𝐿𝑀
1.6876
0.0270
0.0430
0.0485
0.0595
0.0355
0.0270
0.0353
0.0371
0.0399
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d897/d897c561-5eb6-49b9-b144-3b72304d8444.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: The positive correlation between quality of retrieved user and performance of model.</div>
6.4.2 Analysis on other linguistic integration of user integration. We replaced user retrieval with candidate provision to explore more semantic integration approaches, ensuring that their lengths are equal. The model’s performance using candidates can be observed in Table 6 and Table 7. It is noteworthy that, even though the UHR value for candidates is higher than our user retrieval under the same length, their overall final performance is inferior to our model. Additionally, in the experiments, we discovered some interesting phenomena – In a limited subset of instances even when the user retrieval UHR is 0, our model’s performance still improves. This demonstrates the importance of collaborative information in user retrieval and validates the correctness of our approach.
# 7 CONCLUSION
In conclusion, this paper introduced URLLM, a novel user retrieval CDSR framework to seamlessly incorporate diverse information into LLM. Initially, we developed a dual-graph sequence-modeling framework to capture collaborative and structural-semantic information derived from user interactions. Subsequently, we devised a novel user retrieval model for LLMs, aimed at infusing this knowledge into LLMs by exploiting their robust language reasoning and ensemble capabilities. Furthermore, the domain-specific retrieval strategy and answer refinement module were proposed for domainspecific information integration and generation. Compared to traditional approaches and other LLM-based methods, URLLM exhibited improved performance on the CDSA task. This research not only explored the relationship between provided user retrieval results and model performance but also propelled LLM-based CDSR toward the desired format. In further work, we
will attempt to retrench the retrieved user length and evaluations on larger-scale models.
[1] Nawaf Alharbi and Doina Caragea. 2022. Cross-domain Self-attentive Sequential Recommendations. In Proceedings of International Conference on Data Science and Applications, Mukesh Saraswat, Sarbani Roy, Chandreyee Chowdhury, and Amir H. Gandomi (Eds.). Springer Singapore, Singapore, 601–614. [2] Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yancheng Luo, Chong Chen, Fuli Feng, and Qi Tian. 2023. A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems. arXiv:2308.08434 [cs.IR] [3] Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yancheng Luo, Fuli Feng, Xiangnan He, and Qi Tian. 2023. A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems. CoRR abs/2308.08434 (2023). https://doi.org/10.48550/ARXIV.2308.08434 arXiv:2308.08434 [4] Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems (Singapore, Singapore) (RecSys ’23). Association for Computing Machinery, New York, NY, USA, 1007–1014. https: //doi.org/10.1145/3604915.3608857 [5] Daniil A. Boiko, Robert MacKnight, and Gabe Gomes. 2023. Emergent autonomous scientific research capabilities of large language models. CoRR abs/2304.05332 (2023). https://doi.org/10.48550/ARXIV.2304.05332 arXiv:2304.05332 [6] Jiangxia Cao, Xin Cong, Jiawei Sheng, Tingwen Liu, and Bin Wang. 2022. Contrastive Cross-Domain Sequential Recommendation. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (Atlanta, GA, USA) (CIKM ’22). Association for Computing Machinery, New York, NY, USA, 138–147. https://doi.org/10.1145/3511808.3557262 [7] Jiangxia Cao, Xin Cong, Jiawei Sheng, Tingwen Liu, and Bin Wang. 2022. Contrastive Cross-Domain Sequential Recommendation. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, Atlanta, GA, USA, October 17-21, 2022, Mohammad Al Hasan and Li Xiong (Eds.). ACM, 138–147. https://doi.org/10.1145/3511808.3557262 [8] Junyi Chen. 2023. A Survey on Large Language Models for Personalized and Explainable Recommendations. CoRR abs/2311.12338 (2023). https://doi.org/10. 48550/ARXIV.2311.12338 arXiv:2311.12338 [9] Wenqi Fan, Zihuai Zhao, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Jiliang Tang, and Qing Li. 2023. Recommender Systems in the Era of Large Language Models (LLMs). CoRR abs/2307.02046 (2023). https://doi.org/10.48550/ARXIV. 2307.02046 arXiv:2307.02046 [10] Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang. 2023. Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System. CoRR abs/2303.14524 (2023). https://doi.org/10.48550/ ARXIV.2303.14524 arXiv:2303.14524 [11] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5). In Proceedings of the 16th ACM Conference on Recommender Systems (Seattle, WA, USA) (RecSys ’22). Association for Computing Machinery, New York, NY, USA, 299–315. https://doi.org/10.1145/3523227.3546767 [12] Yuqi Gong, Xichen Ding, Yehui Su, Kaiming Shen, Zhongyi Liu, and Guannan Zhang. 2023. An Unified Search and Recommendation Foundation Model for Cold-Start Scenario. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (<conf-loc>, <city>Birmingham</city>, <country>United Kingdom</country>, </conf-loc>) (CIKM ’23). Association for Computing Machinery, New York, NY, USA, 4595–4601. https://doi.org/10.1145/ 3583780.3614657 [13] Lei Guo, Li Tang, Tong Chen, Lei Zhu, Quoc Viet Hung Nguyen, and Hongzhi Yin. 2021. DA-GCN: A Domain-aware Attentive Graph Convolution Network for Shared-account Cross-domain Sequential Recommendation. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21,
Zhi-Hua Zhou (Ed.). International Joint Conferences on Artificial Intelligence Organization, 2483–2489. https://doi.org/10.24963/ijcai.2021/342 Main Track. [14] Yongqiang Han, Hao Wang, Kefan Wang, Likang Wu, Zhi Li, Wei Guo, Yong Liu, Defu Lian, and Enhong Chen. 2024. END4Rec: Efficient Noise-Decoupling for Multi-Behavior Sequential Recommendation. arXiv preprint arXiv:2403.17603 (2024). [15] Yongqiang Han, Likang Wu, Hao Wang, Guifeng Wang, Mengdi Zhang, Zhi Li, Defu Lian, and Enhong Chen. 2023. Guesr: A global unsupervised dataenhancement with bucket-cluster sampling for sequential recommendation. In International Conference on Database Systems for Advanced Applications. Springer, 286–296. [16] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-Dong Zhang, and Meng Wang. 2020. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, Jimmy X. Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.). ACM, 639–648. https://doi.org/10.1145/3397271.3401063 [17] Xiangnan He, Hanwang Zhang, Min-Yen Kan, and Tat-Seng Chua. 2016. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. In Proceedings of the 39th International ACM SIGIR Conference on Research and Development in Information Retrieval (Pisa, Italy) (SIGIR ’16). Association for Computing Machinery, New York, NY, USA, 549–558. https://doi.org/10.1145/ 2911451.2911489 [18] Balázs Hidasi and Alexandros Karatzoglou. 2018. Recurrent Neural Networks with Top-k Gains for Session-based Recommendations. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management (Torino, Italy) (CIKM ’18). Association for Computing Machinery, New York, NY, USA, 843–852. https://doi.org/10.1145/3269206.3271761 [19] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. In 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, Yoshua Bengio and Yann LeCun (Eds.). http://arxiv.org/abs/1511.06939 [20] Yupeng Hou, Shanlei Mu, Wayne Xin Zhao, Yaliang Li, Bolin Ding, and JiRong Wen. 2022. Towards Universal Sequence Representation Learning for Recommender Systems. In KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14 - 18, 2022, Aidong Zhang and Huzefa Rangwala (Eds.). ACM, 585–593. https: //doi.org/10.1145/3534678.3539381 [21] Yupeng Hou, Shanlei Mu, Wayne Xin Zhao, Yaliang Li, Bolin Ding, and JiRong Wen. 2022. Towards Universal Sequence Representation Learning for Recommender Systems. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (Washington DC, USA) (KDD ’22). Association for Computing Machinery, New York, NY, USA, 585–593. https: //doi.org/10.1145/3534678.3539381 [22] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021). [23] Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang, and Enhong Chen. 2024. Understanding the planning of LLM agents: A survey. arXiv preprint arXiv:2402.02716 (2024). [24] Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated gain-based evaluation of IR techniques. ACM Trans. Inf. Syst. 20, 4 (2002), 422–446. https://doi.org/10. 1145/582415.582418 [25] Wei Jin, Haitao Mao, Zheng Li, Haoming Jiang, Chen Luo, Hongzhi Wen, Haoyu Han, Hanqing Lu, Zhengyang Wang, Ruirui Li, et al. 2023. Amazon-m2: A multilingual multi-locale shopping session dataset for recommendation and text generation. arXiv preprint arXiv:2307.09688 (2023). [26] Wang-Cheng Kang and Julian J. McAuley. 2018. Self-Attentive Sequential Recommendation. In IEEE International Conference on Data Mining, ICDM 2018, Singapore, November 17-20, 2018. IEEE Computer Society, 197–206. https: //doi.org/10.1109/ICDM.2018.00035 [27] Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. In 2018 IEEE International Conference on Data Mining (ICDM). 197–206. https://doi.org/10.1109/ICDM.2018.00035 [28] Diederik P. Kingma and Jimmy Ba. 2017. Adam: A Method for Stochastic Optimization. arXiv:1412.6980 [cs.LG] [29] Walid Krichene and Steffen Rendle. 2020. On sampled metrics for item recommendation. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. 1748–1757. [30] Yuxuan Lei, Jianxun Lian, Jing Yao, Xu Huang, Defu Lian, and Xing Xie. 2023. RecExplainer: Aligning Large Language Models for Recommendation Model Interpretability. CoRR abs/2311.10947 (2023). https://doi.org/10.48550/ARXIV. 2311.10947 arXiv:2311.10947 [31] Chenglin Li, Mingjun Zhao, Huanming Zhang, Chenyun Yu, Lei Cheng, Guoqiang Shu, BeiBei Kong, and Di Niu. 2022. RecGURU: Adversarial Learning of Generalized User Representations for Cross-Domain Recommendation. In Proceedings
of the Fifteenth ACM International Conference on Web Search and Data Mining (Virtual Event, AZ, USA) (WSDM ’22). Association for Computing Machinery, New York, NY, USA, 571–581. https://doi.org/10.1145/3488560.3498388 [32] Jinming Li, Wentao Zhang, Tian Wang, Guanglei Xiong, Alan Lu, and Gerard Medioni. 2023. GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation. In Proceedings of the 2023 SIGIR Workshop on eCommerce co-located with the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2023), Taipei, Taiwan, July 27, 2023 (CEUR Workshop Proceedings, Vol. 3589), Surya Kallumadi, Yubin Kim, Tracy Holloway King, Shervin Malmasi, Maarten de Rijke, and Jacopo Tagliabue (Eds.). CEUR-WS.org. https://ceur-ws.org/Vol-3589/paper_2.pdf [33] Junlong Li, Zhuosheng Zhang, and Hai Zhao. 2023. Self-Prompting Large Language Models for Zero-Shot Open-Domain QA. arXiv:2212.08635 [cs.CL] [34] Xiangyang Li, Bo Chen, Lu Hou, and Ruiming Tang. 2023. CTRL: Connect Tabular and Language Model for CTR Prediction. CoRR abs/2306.02841 (2023). https://doi.org/10.48550/ARXIV.2306.02841 arXiv:2306.02841 [35] Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, and Xiang Wang. 2023. LLaRA: Aligning Large Language Models with Sequential Recommenders. CoRR abs/2312.02445 (2023). https://doi.org/10.48550/ARXIV.2312. 02445 arXiv:2312.02445 [36] Weiwen Liu, Wei Guo, Yong Liu, Ruiming Tang, and Hao Wang. 2023. User Behavior Modeling with Deep Learning for Recommendation: Recent Advances. In Proceedings of the 17th ACM Conference on Recommender Systems. 1286–1287. [37] Zhiwei Liu, Yongjun Chen, Jia Li, Philip S. Yu, Julian J. McAuley, and Caiming Xiong. 2021. Contrastive Self-supervised Sequential Recommendation with Robust Augmentation. CoRR abs/2108.06479 (2021). arXiv:2108.06479 https: //arxiv.org/abs/2108.06479 [38] Muyang Ma, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Lifan Zhao, Peiyu Liu, Jun Ma, and Maarten de Rijke. 2022. Mixed Information Flow for Cross-Domain Sequential Recommendations. ACM Trans. Knowl. Discov. Data 16, 4 (2022), 64:1–64:32. https://doi.org/10.1145/3487331 [39] Muyang Ma, Pengjie Ren, Yujie Lin, Zhumin Chen, Jun Ma, and Maarten de Rijke. 2019. 𝜋-Net: A Parallel Information-sharing Network for Shared-account Crossdomain Sequential Recommendations. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (Paris, France) (SIGIR’19). Association for Computing Machinery, New York, NY, USA, 685–694. https://doi.org/10.1145/3331184.3331200 [40] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Found. Trends Inf. Retr. 3, 4 (apr 2009), 333–389. https://doi.org/10.1561/1500000019 [41] Andrew I. Schein, Alexandrin Popescul, Lyle H. Ungar, and David M. Pennock. 2002. Methods and metrics for cold-start recommendations. In SIGIR 2002: Proceedings of the 25th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, August 11-15, 2002, Tampere, Finland, Kalervo Järvelin, Micheline Beaulieu, Ricardo A. Baeza-Yates, and Sung-Hyon Myaeng (Eds.). ACM, 253–260. https://doi.org/10.1145/564376.564421 [42] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (Beijing, China) (CIKM ’19). Association for Computing Machinery, New York, NY, USA, 1441–1450. https://doi.org/10.1145/3357384.3357895 [43] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, CIKM 2019, Beijing, China, November 3-7, 2019, Wenwu Zhu, Dacheng Tao, Xueqi Cheng, Peng Cui, Elke A. Rundensteiner, David Carmel, Qi He, and Jeffrey Xu Yu (Eds.). ACM, 1441–1450. https://doi.org/10.1145/3357384.3357895 [44] Wenchao Sun, Muyang Ma, Pengjie Ren, Yujie Lin, Zhumin Chen, Zhaochun Ren, Jun Ma, and Maarten de Rijke. 2023. Parallel Split-Join Networks for Shared Account Cross-Domain Sequential Recommendations. IEEE Transactions on Knowledge and Data Engineering 35, 4 (2023), 4106–4123. https://doi.org/10.1109/ TKDE.2021.3130927 [45] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 14918–14937. https://doi.org/10.18653/v1/2023.emnlp-main.923 [46] Jiaxi Tang and Ke Wang. 2018. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining (Marina Del Rey, CA, USA) (WSDM ’18). Association for Computing Machinery, New York, NY, USA, 565–573. https://doi.org/10.1145/3159652.3159656 [47] Zuoli Tang, Zhaoxin Huan, Zihao Li, Xiaolu Zhang, Jun Hu, Chilin Fu, Jun Zhou, and Chenliang Li. 2023. One Model for All: Large Language Models are Domain-Agnostic Recommendation Systems. CoRR abs/2310.14304 (2023).
https://doi.org/10.48550/ARXIV.2310.14304 arXiv:2310.14304 [48] Ellen M. Voorhees. 1999. The TREC-8 Question Answering Track Report. In Proceedings of The Eighth Text REtrieval Conference, TREC 1999, Gaithersburg, Maryland, USA, November 17-19, 1999 (NIST Special Publication, Vol. 500-246), Ellen M. Voorhees and Donna K. Harman (Eds.). National Institute of Standards and Technology (NIST). http://trec.nist.gov/pubs/trec8/papers/qa_report.pdf [49] Hao Wang, Defu Lian, Hanghang Tong, Qi Liu, Zhenya Huang, and Enhong Chen. 2021. Hypersorec: Exploiting hyperbolic user and item representations with multiple aspects for social-aware recommendation. ACM Transactions on Information Systems (TOIS) 40, 2 (2021), 1–28. [50] Hao Wang, Tong Xu, Qi Liu, Defu Lian, Enhong Chen, Dongfang Du, Han Wu, and Wen Su. 2019. MCNE: An end-to-end framework for learning multiple conditional network representations of social network. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 1064–1072. [51] Yancheng Wang, Ziyan Jiang, Zheng Chen, Fan Yang, Yingxue Zhou, Eunah Cho, Xing Fan, Xiaojiang Huang, Yanbin Lu, and Yingzhen Yang. 2023. RecMind: Large Language Model Powered Agent For Recommendation. CoRR abs/2308.14296 (2023). https://doi.org/10.48550/ARXIV.2308.14296 arXiv:2308.14296 [52] Ziyang Wang, Wei Wei, Gao Cong, Xiao-Li Li, Xian-Ling Mao, and Minghui Qiu. 2020. Global Context Enhanced Graph Neural Networks for Session-based Recommendation. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (Virtual Event, China) (SIGIR ’20). Association for Computing Machinery, New York, NY, USA, 169–178. https://doi.org/10.1145/3397271.3401142 [53] SJ Waters. 1976. Hit ratios. Comput. J. 19, 1 (1976), 21–24. [54] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.). http://papers.nips.cc/paper_files/paper/2022/hash/ 9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html [55] Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, Hongchao Gu, Tingjia Shen, Chuan Qin, Chen Zhu, Hengshu Zhu, Qi Liu, et al. 2023. A Survey on Large Language Models for Recommendation. arXiv preprint arXiv:2305.19860 (2023). [56] Xu Xie, Fei Sun, Zhaoyang Liu, Shiwen Wu, Jinyang Gao, Jiandong Zhang, Bolin Ding, and Bin Cui. 2022. Contrastive Learning for Sequential Recommendation. In 2022 IEEE 38th International Conference on Data Engineering (ICDE). 1259–1273. https://doi.org/10.1109/ICDE53745.2022.00099 [57] Zhengyi Yang, Jiancan Wu, Yanchen Luo, Jizhi Zhang, Yancheng Yuan, An Zhang, Xiang Wang, and Xiangnan He. 2023. Large Language Model Can Interpret Latent Space of Sequential Recommender. CoRR abs/2310.20487 (2023). https: //doi.org/10.48550/ARXIV.2310.20487 arXiv:2310.20487 [58] Mingjia Yin, Hao Wang, Wei Guo, Yong Liu, Zhi Li, Sirui Zhao, Defu Lian, and Enhong Chen. 2024. Learning Partially Aligned Item Representation for CrossDomain Sequential Recommendation. arXiv preprint arXiv:2405.12473 (2024). [59] Mingjia Yin, Hao Wang, Wei Guo, Yong Liu, Suojuan Zhang, Sirui Zhao, Defu Lian, and Enhong Chen. 2024. Dataset Regeneration for Sequential Recommendation. arXiv preprint arXiv:2405.17795 (2024). [60] Mingjia Yin, Hao Wang, Xiang Xu, Likang Wu, Sirui Zhao, Wei Guo, Yong Liu, Ruiming Tang, Defu Lian, and Enhong Chen. 2023. APGL4SR: A Generic Framework with Adaptive and Personalized Global Collaborative Information in Sequential Recommendation. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 3009–3019. [61] Zheng Yuan, Fajie Yuan, Yu Song, Youhua Li, Junchen Fu, Fei Yang, Yunzhu Pan, and Yongxin Ni. 2023. Where to Go Next for Recommender Systems? IDvs. Modality-based Recommender Models Revisited. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023, Hsin-Hsi Chen, Wei-Jou (Edward) Duh, Hen-Hsen Huang, Makoto P. Kato, Josiane Mothe, and Barbara Poblete (Eds.). ACM, 2639–2649. https://doi.org/10.1145/3539618.3591932 [62] An Zhang, Leheng Sheng, Yuxin Chen, Hao Li, Yang Deng, Xiang Wang, and Tat-Seng Chua. 2023. On Generative Agents in Recommendation. arXiv:2310.10108 [cs.IR] [63] Luankang Zhang, Hao Wang, Suojuan Zhang, Mingjia Yin, Yongqiang Han, Jiaqing Zhang, Defu Lian, and Enhong Chen. 2024. A Unified Framework for Adaptive Representation Enhancement and Inversed Learning in Cross-Domain Recommendation. arXiv preprint arXiv:2404.00268 (2024). [64] Yuren Zhang, Enhong Chen, Binbin Jin, Hao Wang, Min Hou, Wei Huang, and Runlong Yu. 2022. Clustering based behavior sampling with long sequential data for CTR prediction. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2195–2200. [65] Yang Zhang, Fuli Feng, Jizhi Zhang, Keqin Bao, Qifan Wang, and Xiangnan He. 2023. CoLLM: Integr