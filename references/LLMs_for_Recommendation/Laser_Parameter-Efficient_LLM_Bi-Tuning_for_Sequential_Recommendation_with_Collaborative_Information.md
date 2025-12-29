# Laser: Parameter-Efficient LLM Bi-Tuning for Sequential Recommendation with Collaborative Information
Xinyu Zhang xyzhang0105@gmail.com Beijing Institute of Technology Beijing, China Linmei Hu hulinmei@bit.edu.cn Beijing Institute of Technology Beijing, China
Linmei Hu hulinmei@bit.edu.cn Beijing Institute of Technology Beijing, China Luhao Zhang zhangluhao@bit.edu.cn Beijing Institute of Technology Beijing, China
Heyan Huang hhy63@bit.edu.cn Beijing Institute of Technology Beijing, China Liqiang Nie nieliqiang@gmail.com Harbin Institute of Technology Shenzhen, China
Dandan Song sdd@bit.edu.cn Beijing Institute of Technology Beijing, China
Abstract Sequential recommender systems are essential for discerning user preferences from historical interactions and facilitating targeted recommendations. Conventional techniques rely solely on item IDs for sequence modeling, overlooking the wealth of semantic data in item descriptions, which can lead to subpar performance. Recent innovations employing Large Language Models (LLMs) have advanced the field by encoding item semantics, yet they often necessitate substantial parameter tuning and are resource-demanding. Moreover, these works typically integrate ID-based collaborative signals into LLMs via a simple unified linear projection, which fails to consider the diverse characteristics of different types of users and thus diminishes the recommendation accuracy. In this paper, we propose a parameter-efficient Large Language Model Bi-Tuning framework for sequential recommendation with collaborative information (Laser). Specifically, Bi-Tuning works by inserting trainable virtual tokens at both the prefix and suffix of the input sequence and freezing the LLM parameters, thus optimizing the LLM for the sequential recommendation. In our Laser, the prefix is utilized to incorporate user-item collaborative information and adapt the LLM to the recommendation task, while the suffix converts the output embeddings of the LLM from the language space to the recommendation space for the follow-up item recommendation. Furthermore, to capture the characteristics of different types of users when integrating the collaborative information via the prefix, we introduce M-Former, a lightweight MoE-based querying transformer that uses a set of query experts to integrate diverse user-specific collaborative information encoded by frozen ID-based sequential recommender systems, significantly improving the accuracy of recommendations.
arXiv:2409.01605v1
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. Conference acronym ’XX, June 03–05, 2018, Woodstock, NY © 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/18/06 https://doi.org/XXXXXXX.XXXXXXX
Extensive experiments on real-world datasets demonstrate that Laser can parameter-efficiently adapt LLMs to effective recommender systems, significantly outperforming state-of-the-art methods.
# CCS Concepts • Information systems →Recommender systems.
Sequential recommendation, large language model, parameter-efficient fine-tuning, MoE, collaborative information
ACM Reference Format: Xinyu Zhang, Linmei Hu, Luhao Zhang, Dandan Song, Heyan Huang, and Liqiang Nie. 2024. Laser: Parameter-Efficient LLM Bi-Tuning for Sequential Recommendation with Collaborative Information. In Proceedings of Make sure to enter the correct conference title from your rights confirmation emai (Conference acronym ’XX). ACM, New York, NY, USA, 11 pages. https://doi.org/XXXXXXX.XXXXXXX
# 1 Introduction
Sequential recommender systems are designed to learn effective representations of users’ interests based on their past interactions and to suggest future items that match users’ needs. Due to their abilities to capture the dynamic nature of user preferences and their effectiveness in enhancing user satisfaction, sequential recommender systems are widely applied in various scenarios such as e-commerce, streaming services, and social media platforms [7, 11, 27, 54]. In traditional sequential recommender systems, items are predominantly represented by unique IDs. To obtain effective ID embeddings based on the user interaction sequence, a variety of methods are employed, including Markov Chains [9, 32], RNN/CNN models [11, 20, 34, 46], and self-attentive models [16, 22, 33]. While ID-based methods are promising in capturing latent associations between users and items, they fail to consider the rich semantic information contained in the textual descriptions of items (e.g., item title), resulting in suboptimal performance. To solve this issue, efforts have been made to encode item semantic information with language models [12, 21]. However, previous works mainly focus on small or medium-sized language models, which exhibit limited performance.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ad4f/ad4f0f3c-56a0-4348-899a-fe5c4dd82e53.png" style="width: 50%;"></div>
# Figure 1: Comparison of existing methods and our proposed Laser.
Recently, Large Language Models (LLMs) have made significant progress in language understanding [15, 30, 35, 44]. It is a promising way to harness the powerful semantic information modeling capabilities of LLMs pre-trained on extensive text corpora to capture the semantic information of items. As shown in Figure 1, existing works integrate LLMs into recommendation tasks in two main paradigms. The first paradigm is to use LLMs to directly recommend in the form of natural language. These works design special prompts [14, 28, 37] or use supervised fine-tuning [1, 2, 51] to get LLMs to answer the given recommendation questions. However, this paradigm can only determine the recommendation for one item at a time and the frequency of LLM utilization increases linearly with the number of candidate items. Thus, these methods tend to be used only in the reranking phase , which contains only dozens of candidate items [13, 51]. The second paradigm is to use LLMs as encoders to provide item/user embeddings for similarity comparison and next item prediction. As shown in Figure 1, given the user interaction history represented in natural language, these works use LLMs to encode each token in the input text and then perform various pooling strategies on the output token embeddings to derive the user embedding [23, 39, 49]. Although these works are promising, they typically necessitate the training of extensive parameters, demanding considerable computational resources. Furthermore, these works struggle to effectively incorporate ID-based collaborative information into LLMs, which affects the effectiveness of recommendations. Although efforts have been made to use simple linear projections to map the collaborative embeddings into the language space of LLMs [45, 53], these methods fail to consider the diverse characteristics of various types of users, potentially resulting in inferior recommendation results. To address the above issues, in this paper, we propose a parameterefficient Large Language Model Bi-Tuning framework for sequential recommendation (Laser), which also effectively integrates collaborative information by capturing the characteristics of different types of users through an MoE-based querying transformer. In particular, to efficiently adapt LLMs to effective sequential recommender systems that can provide high-quality item/user embeddings, we design a parameter-efficient bidirectional LLM fine-tuning method, named Bi-Tuning. In Bi-Tuning, we freeze LLMs’ parameters and tailor them for recommendation tasks by optimizing the trainable virtual tokens added at the prefix and suffix of the input text, which largely reduces the scale of parameters that require training. The
prefix tokens can be utilized to incorporate collaborative information and are responsible for adapting LLMs to recommendation tasks, while the appended single suffix token aims to convert the output of LLMs from the language space to the recommendation space for following embedding similarity comparison and next item recommendation. In addition, to effectively incorporate the collaborative information via the prefix for accurate recommendation, we present M-Former, a lightweight MoE (Mixture of Experts) based querying transformer that employs a set of trainable query experts to capture the diverse characteristics of user-specific collaborative information encoded by frozen ID-based sequential recommender systems. Experimental results on real-world datasets across different domains show that our method outperforms state-of-the-art baselines. In summary, our main contributions can be summarized as follows: • We propose a parameter-efficient Large Language Model BiTuning framework for sequential recommendation, named Laser, which can effectively adapt LLMs to sequential recommender systems in a parameter-efficient way. • In our Laser, to effectively incorporate the collaborative information into LLMs for more accurate recommendation, we design M-Former, a lightweight MoE-based querying transformer that employs a set of query experts to capture the characteristics of user-specific collaborative information encoded by frozen ID-based sequential recommender systems. • Extensive experiments on real-world datasets demonstrate that our proposed Laser significantly outperforms state-ofthe-art methods.
# 2 Problem Formulation
In the setting of sequential recommendation, we are given a user set U and an item set I. Each user 𝑢∈U is associated with a temporally ordered sequence of his/her historical interacted items, denoted as 𝑆𝑢= {𝑖1,𝑖2, . . . ,𝑖𝑁}, where 𝑁is the length of 𝑆𝑢and 𝑖∈I. Based on 𝑆𝑢, sequential recommender systems are used to predict the next item 𝑖𝑁+1 that user 𝑢is most likely to interact with. In traditional ID-based sequential recommender systems, each item 𝑖is associated with a unique item ID 𝑖𝑑𝑖, and the ID sequence 𝐼𝐷𝑢= {𝑖𝑑𝑖1,𝑖𝑑𝑖2, . . . ,𝑖𝑑𝑖𝑁} is used as input of the model to predict the next item ID 𝑖𝑑𝑖𝑁+1. Differently, in this work, in addition to the item id sequence 𝐼𝐷𝑢, we also utilize the semantic information of items, including the attributes such as title, category, and brand. Formally, the attributes of an item 𝑖can be represented as 𝐷𝑖= {(𝑘1, 𝑣1), (𝑘2, 𝑣2), . . . , (𝑘𝑀, 𝑣𝑀)}, where 𝑘is the attribute name (e.g., “title”, “category”, and “brand”),𝑣is the corresponding value and 𝑀is the number of attributes. Given the user interaction sequence 𝑆𝑢= {𝑖1,𝑖2, . . . ,𝑖𝑁}, we use a template to organize the corresponding item attribute sequence 𝐷𝑢= {𝐷𝑖1, 𝐷𝑖2, ..., 𝐷𝑖𝑁} into a complete and coherent text 𝑇𝑢= {𝑡1,𝑡2, ...,𝑡𝑊} (detailed in Section 3.1.1), where 𝑊is the text length. Then, 𝑇𝑢will be taken as the input of LLMs for next item prediction.
# 3 Methodology
In this section, we detail our proposed Large Language Model BiTuning framework for sequential recommendation, Laser. As illustrated in Figure 2, a parameter-efficient LLM Bi-Tuning method is
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/73bd/73bd39ef-b1e2-4ca2-88df-617ab6bb47d5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: The overview of our proposed Laser.</div>
presented to adapt LLMs for the sequential recommendation. Moreover, a lightweight MoE-based querying transformer, M-Former, is designed to effectively integrate collaborative information into LLMs while capturing the characteristics of users of different types with the MoE strategy.
# 3.1 LLM Bi-Tuning for Sequential Recommendation
In the following, we first describe how to organize the user interaction history 𝐷𝑢= {𝐷𝑖1, 𝐷𝑖2, ..., 𝐷𝑖𝑁} into a coherent text 𝑇𝑢= {𝑡1,𝑡2, ...,𝑡𝑊}, which is taken as the input of LLMs for the sequential recommendation, and then introduce how to adapt LLMs for the recommendation task by the proposed parameter-efficient BiTuning method.
3.1.1 Input Text Formulation. In this work, we utilize a unified template to organize the user interaction history 𝐷𝑢into the input text 𝑇𝑢of LLMs for recommendation. For example, given the user who has browsed “Kaytee Aspen Bedding Bag”, “Guitar A-Frame SupportS”, ..., and “KONG Wubba Dog Toy”, we can formulate the corresponding input text 𝑇𝑢as follows.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/53bd/53bdeb6f-1257-4607-ba55-98cf6a7b280f.png" style="width: 50%;"></div>
You are an intelligent recommendation assistant. Please summarize the user’s characteristics into a single token based on the browsing history. In chronological order, the user has browsed the following items: >> 1. Kaytee Aspen Bedding Bag (brand: Kaytee, category: Kaytee) >> 2. Guitar A-Frame Support (brand: Sageworks, category: Sageworks) ... >> 4. KONG Wubba Dog Toy (brand: KONG, category: KONG)
With the designed template, LLMs can follow the instruction in the input text to summarize the user browsing history into a single token (i.e., the suffix appended to the end of the input text), whose
corresponding output embedding can be taken as the user embedding h𝑢for similarity comparison with the item embeddings and the next item recommendation. To obtain the item embeddings, we also use the above same template. Particularly, for a specific item 𝑖, we treat it as a special user interaction history that contains only this one item. Therefore, we use the same template to formulate the input text of the LLM and take the output appended suffix embedding as the item embedding h𝑖. In this way, we can obtain the embedding of each item 𝑖in the item set I, and the original user-item similarity comparison used for recommendation can be regarded as a special kind of user-user similarity comparison. The advantage of this is that a unified template could minimize the impact of hard templates on the performance of LLMs [18, 29]. Detailed experiments in Section 4.5.2 further prove the validity of our unified template.
corresponding output embedding can be taken as the user embedding h𝑢for similarity comparison with the item embeddings and the next item recommendation. To obtain the item embeddings, we also use the above same template. Particularly, for a specific item 𝑖, we treat it as a special user interaction history that contains only this one item. Therefore, we use the same template to formulate the input text of the LLM and take the output appended suffix embedding as the item embedding h𝑖. In this way, we can obtain the embedding of each item 𝑖in the item set I, and the original user-item similarity comparison used for recommendation can be regarded as a special kind of user-user similarity comparison. The advantage of this is that a unified template could minimize the impact of hard templates on the performance of LLMs [18, 29]. Detailed experiments in Section 4.5.2 further prove the validity of our unified template. 3.1.2 Parameter-Efficient Bi-Tuning. Existing works have shown the powerful capabilities of LLMs in bolstering the sequential recommendation [1, 2, 51]. However, they still face two main challenges: (1) how to adapt LLMs for recommendation tasks in a parameterefficient way, and (2) how to effectively transform the output of LLMs from the language space to the recommendation space for the following item recommendation. To solve these two challenges, as shown in Figure 2, we propose a parameter-efficient LLM Bi-Tuning method that adapts LLMs through the trainable prefix and suffix. Formally, given the input text 𝑇𝑢= {𝑡1,𝑡2, ...,𝑡𝑊}, it will be expanded with the trainable prefix and suffix:
3.1.2 Parameter-Efficient Bi-Tuning. Existing works have shown the powerful capabilities of LLMs in bolstering the sequential recommendation [1, 2, 51]. However, they still face two main challenges: (1) how to adapt LLMs for recommendation tasks in a parameterefficient way, and (2) how to effectively transform the output of LLMs from the language space to the recommendation space for the following item recommendation. To solve these two challenges, as shown in Figure 2, we propose a parameter-efficient LLM Bi-Tuning method that adapts LLMs through the trainable prefix and suffix. Formally, given the input text 𝑇𝑢= {𝑡1,𝑡2, ...,𝑡𝑊}, it will be expanded with the trainable prefix and suffix:
���������������������� �������������������� ���� where 𝑃= {𝑝1, 𝑝2, ..., 𝑝𝐿} refers to the prefix that contains 𝐿prepende virtual tokens, and 𝑠refers to the suffix that consists of one single appended virtual token. During model training, we freeze the parameters of LLMs and tailor them for the recommendation task by optimizing the trainable virtual tokens added at the prefix 𝑃 and suffix 𝑠, which greatly reduces the size of the parameters to be trained.
Specifically, the prefix 𝑃containing 𝐿virtual tokens is responsible for adapting LLMs to the recommendation task with collaborative information. As proven by previous works [18, 24], these virtual tokens can serve as placeholders that allow LLMs to capture task-specific information during fine-tuning. In addition, we also use the prefix 𝑃to integrate ID-based collaborative information into LLMs via the proposed M-Former (detailed in Section 3.2), which has proven useful for improving the recommendation results [40, 55]. In addition to the prefix 𝑃, we also append a special trainable virtual token 𝑠to the end of the input text 𝑇𝑢, which is called the suffix. Previous works [23] have tried to perform average pooling on the token embeddings output by LLMs to obtain the user embedding. However, most generative LLMs are based on the masked attention mechanism, dictating that only the last token can observe the entire input. Therefore, these works may introduce noise by performing average pooling on all output embeddings. In this work, we utilize an appended trainable virtual token 𝑠to capture the information of the entire input ˜𝑇𝑢, whose output embedding h𝑠can be taken as the user embedding h𝑢for similarity comparison and next item prediction. Formally, the encoding process of LLMs can be represented as:
(2)
where e ∈R𝑑represents an input token embedding of the input ˜𝑇𝑢= {𝑝1, ...,𝑡1, ...,𝑠}, h ∈R𝑑represents the corresponding output embedding, and 𝑑represents the hidden size of the LLM. Through the trainable suffix 𝑠, we can effectively convert the output embedding of the LLM from the language space to the recommendation space. When taking the user interaction history or the single item as input of the LLM, we can directly take the output suffix embedding h𝑠as the user embedding h𝑢or item embedding h𝑖for further recommendation.
3.1.3 Item Recommendation. Given the obtained user embedding h𝑢∈R𝑑and the item embedding h𝑖∈R𝑑from the LLM, we can compute the similarity between them as follows:
(3)
where 𝑠(𝑢,𝑖) ∈R indicates the probability that the item 𝑖will become the next item browsed by user 𝑢. To predict the next item, we iterate through each item 𝑖in the item set I, and select the item ˆ𝑖with the highest score as the next item:
ˆ𝑖= argmax𝑖∈I (𝑠(𝑢,𝑖)) .
(4)
# 3.2 M-Former based Collaborative Information Integration
In the proposed LLM Bi-Tuning, we use the trainable prefix 𝑃and suffix 𝑠to adapt the LLM for recommendation. In order to achieve better recommendation results, we incorporate the collaborative information via the prefix 𝑃. Existing works have tried to use unified linear layers to project the collaborative embeddings encoded by ID-based sequential recommender systems into the language space of LLMs [45, 53]. However, this method is too simple to detect the diverse characteristics of different types of users [25, 48]. To deal
with this challenge, we introduce M-Former, an MoE-based querying transformer that employs a set of query experts to deal with different types of users and integrates user-specific collaborative information into the prefix 𝑃. In the following, we first describe the MoE strategy, namely how to select the appropriate query expert based on the collaborative characteristics of a specific user from a set of experts. Then we explain how the selected query expert interacts with the collaborative information encoded by frozen ID-based sequential recommender systems in the querying transformer.
3.2.1 Mixture of Experts. As shown in Figure 2, there are 𝐾query experts dealing with users of different types, each of which contains 𝐿trainable virtual tokens. In order to select the most appropriate expert to deal with user-specific collaborative information, we set up a router to calculate the scores of different experts given the specific user. Formally, given the user interaction history 𝑆𝑢= {𝑖1,𝑖2, . . . ,𝑖𝑁} of user 𝑢, the corresponding item ID sequence 𝐼𝐷𝑢= {𝑖𝑑𝑖1,𝑖𝑑𝑖2, . . . ,𝑖𝑑𝑖𝑁} is taken as input of a pre-trained IDbased sequential recommender system (frozen) and encoded as C𝑢∈R𝑁×𝑑𝑖, where 𝑁represents the length of the interaction history 𝑆𝑢and 𝑑𝑖represents the hidden size of the sequential recommender system. Then, C𝑢is sent into the router, which is implemented with a fully-connected layer in this work. The matching degree of the 𝐾query experts according to the 𝑁user-interactive items embedded as C𝑢can be calculated as:
(5)
where W𝑟∈R𝐾×𝑑𝑖is the router’s linear weight, and𝑚(𝑢) ∈R𝑁×𝐾. Then, he 𝑖-th item’s score for the 𝑗-th expert can be calculated as:
� and the final score of the 𝑗-th query expert can be obtained by: �
(7)
� Finally, the query expert with the highest score will be selected to deal with the specific user 𝑢.
3.2.2 MoE-based Querying Transformer. As described above, we obtain the most appropriate query expert to handle the specific user’s collaborative information C𝑢. Afterward, as shown in Figure 2, the selected query expert containing 𝐿virtual tokens is sequentially fed into𝑍transformer blocks to interact with the collaborative information C𝑢. In this way, the query expert integrates the collaborative information into its 𝐿trainable virtual tokens, which further act as the aforementioned prefix 𝑃= {𝑝1, 𝑝2, ..., 𝑝𝐿} to adapt LLMs for the sequential recommendation. Formally, the query expert can be represented as E ∈R𝐿×𝑑𝑚, where 𝑑𝑚is the hidden size of the M-Former. In the transformer block, E is first encoded by a self-attention layer and then projected to the query matrix Q used in the cross-attention layer to interact with the key/value matrix (K/V) projected from the ID-based collaborative embeddings C𝑢. In this way, we update the query expert’s embedding E and integrate ID-based collaborative information into it. Then, through a linear projection layer set up on the top of the 𝑍transformer blocks, the query expert is projected into the LLM’s
hidden size 𝑑and acts as the prefix 𝑃= {𝑝1, 𝑝2, ..., 𝑝𝐿} to adapt the LLM for sequential recommendation with the enhancement of collaborative information.
# 3.3 Model Learning
In this work, we employ a multi-task training strategy to train our Laser, which takes into account both the recommendation goal and the load balancing goal of the MoE experts. Furthermore, we perform a two-stage training by first finding the most appropriate parameter weights to obtain high-quality item embeddings used for user-item similarity comparison, and then further training Laser based on these fixed item embeddings to achieve the best recommendation results.
# 3.3.1 Loss Function. We propose a multi-task training strategy to jointly train the proposed Laser for LLM-based sequential recommendation.
The first training task is the item-item contrastive (IIC) task, which is widely employed for next item prediction. Following previous work [21], we use the ground-truth next item as the positive instance and all other items in the item set I as negative instances. Formally, the item-item contrastive loss is calculated as:
(8)
� ∈I where the calculation of cos(h𝑢, h𝑖) is consistent with Equation (3), h+ 𝑖represents the embedding of the ground-trouth next item, and 𝜏 is a temperature hyper-parameter. The second training task is the load balancing task, which is used to encourage a balanced load across different query experts of the M-Former. As proven by previous works [6, 17], this task can force the router to assign users with diverse collaborative characteristics to different query experts, such that each expert can be trained to obtain the best collaborative information integration effectiveness for its group of users. Formally, the load balancing loss is calculated as: ∑︁
(9)
∑︁ where 𝐾is the number of query experts, 𝑓𝑗represents the fraction of items dispatched to the 𝑗-th expert that can be calculated as:
(10)
∑︁ where 𝑁is the number of items in the user interaction history, 𝑝(𝑢) ∈R𝑁×𝐾is the score matrix obtained through Equation (6), which represents the degree of correlation between the 𝑁items and the 𝐾query experts. The 𝑃𝑗in Equation (9) represents the fraction of the router probability allocated for the 𝑗-th expert, which can be calculated as: ∑︁
∑︁ Totally, the loss function we use in this work is:
(12)
L L +· L where 𝜆is a hyper-parameter that controls the weight of different tasks.
  where 𝜆is a hyper-parameter that controls the weight of different tasks.
Table 1: Statistics of the preprocessed datasets. Avg. n denotes the average number of items in the user interaction history.
Datasets
#Users #Items #Inters. Avg. n Density
Scientific
11,041
5,327
76,896
6.96
1.3e-3
Arts
56,210
22,855
492,492
8.76
3.8e-4
Pet
47,569
37,970
420,662
8.84
2.3e-4
3.3.2 Two-Stage Training. In this work, the recommendation is conducted based on the user-item embedding similarity comparison. Since the item embedding is determined by the corresponding trainable suffix, which changes after different training epochs. We employ a two-stage training strategy to first find the most appropriate parameter weights to obtain high-quality item embeddings, and then further train Laser based on these fixed item embeddings to achieve the best recommendation results. Specifically, in the first training stage, at the beginning of each epoch, the item embeddings are updated as I ∈R|I|×𝑑using the current parameter weights 𝐴. Then, Laser is trained on I and validated at the end of the epoch based on the updated parameter weights 𝐴′. At the end of the first training stage, the embeddings ˆI and the corresponding parameter weights ˆ𝐴′ of the best-performing epoch are selected for the second training stage. In the second training stage, Laser is initialized with ˆ𝐴′ and then trained for multiple epochs to further adapt to the fixed embeddings ˆI. Finally, the parameter weights ˆ𝐴that yield the optimal validation performance are reserved, and the test results performed on ˆ𝐴and ˆI represent the final performance of Laser.
# 4 Experiments
In this section, we conduct detailed experiments to demonstrate the effectiveness of our proposed Laser.
# 4.1 Experimental Setup
4.1.1 Datasets. To evaluate the effectiveness of our Laser, we conduct experiments on three categories of the Amazon review datasets [31], including “Industrial and Scientific”, “Arts, Crafts and Sewing”, and “Pet Supplies”. Following previous works [12, 21], we use the five-core datasets provided by the data source and filter out items with missing titles. Then, we collect the interactions for different users and sort the interactive items by timestamp in ascending order. The statistics of the preprocessed datasets are shown in Table 1. As for the item semantic information modeling, we select the item attributes including title, category, and brand.
4.1.2 Baselines and Implementation Details. We compare our Laser to a number of state-of-the-art baselines, including six traditional methods (SASRec [16], BERT4Rec [33], RecGURU [19], FDSA [52] ZESRec [5], RECFORMER [21]), and three LLM-based methods (LLM4REC [36], ZESRec [5], LlamaRec [47]). We list the details of these baselines in the Appendix A. Besides, in this paper, the frozen ID-based sequential recommender employed in the Laser is a pre-trained BERT4Rec [33], and the utilized frozen LLM is the ChatGLM2-6B [8]. The other trainable modules are all randomly initialized. The settings of each module and other implementation details are shown in Appendix B.
<div style="text-align: center;">Table 2: Performance comparison of different methods. The best results are in bold and the second best results are underlined. Improv. indicates the improvement between the best and second best results.</div>
Traditional Methods
LLM-based Methods
Dataset
Metric
SASRec BERT4Rec RecGURU ZESRec RECFORMER FDSA LLM4REC
KAR
LlamaRec
Laser
Improv.
Scientific
Recall@10
0.1305
0.1061
0.0781
0.1260
0.1114
0.0967
0.1257
0.1265
0.1275
0.1396
6.97%
NDCG@10
0.0797
0.0790
0.0575
0.0843
0.0722
0.0716
0.0764
0.0894
0.0857
0.0970
8.54%
MRR
0.0696
0.0759
0.0566
0.0745
0.0650
0.0692
0.0683
0.0813
0.0793
0.0893
9.81%
Pet
Recall@10
0.0881
0.0765
0.0415
0.1018
0.0905
0.0949
0.0918
0.0942
0.0961
0.1134
11.37%
NDCG@10
0.0569
0.0602
0.0366
0.0754
0.0793
0.0673
0.0769
0.0724
0.0754
0.0898
13.27%
MRR
0.0507
0.0585
0.0371
0.0706
0.0774
0.0650
0.0681
0.0677
0.0711
0.0856
10.63%
Arts
Recall@10
0.1342
0.1236
0.0742
0.1349
0.1298
0.1209
0.1266
0.1357
0.1368
0.1489
8.91%
NDCG@10
0.0848
0.0942
0.0525
0.0970
0.1024
0.0994
0.0927
0.0917
0.0860
0.1138
11.17%
MRR
0.0742
0.0899
0.0488
0.0870
0.0980
0.0941
0.0880
0.0818
0.0794
0.1095
11.80%
Table 3: Parameter scale comparison of different LLM tuning methods.
Method
Backbone
Tuning
Trainable
Method
Parameters
LLM4REC
GPT2-Large
Full Fine-Tuning
787.3M
KAR
ChatGPT
/
/
LlamaRec
Llama2-7B
QLoRA
4.194M
Laser
ChatGLM2-6B
Bi-Tuning
0.135M
4.1.3 Evaluation Settings. Following previous works [21, 36, 47], we employ three popular metrics, including Recall@N, NDCG@N and MRR, where N is set to 10. For data splitting, we adopt the leaveone-out [16] strategy, where the most recent item in the interaction history is used for testing, the second most recent item is used for validation, and the remaining items are used for training. We treat all items in the item set as candidate items and report the average results on the test data.
# 4.2 Overall Performance
As shown in Table 2, we compare our proposed Laser to nine stateof-the-art baselines across three Amazon datasets. From the experimental results, we can obtain following observations. First, compared to the other outstanding sequential recommendation methods, our Laser results in significant improvements on all metrics across all datasets. For example, on the Pet dataset, compared to the second best method, our Laser improves Recall@10, NDCG@10, and MRR by around 11.37%, 13.27%, and 10.63%, respectively. This demonstrates that our proposed framework can successively adapt LLMs to effective sequential recommender systems. We believe that our Laser benefits from the Bi-Tuning method that effectively adapts LLMs for sequential recommendation with collaborative information. In addition, when integrating the collaborative information, the designed M-Former (MoE-based querying transformer) captures the diverse characteristics of different types of users for more accurate recommendation. Second, compared to the traditional methods, the three LLMbased baselines do not always yield better results. A possible reason is that LLMs have not been pre-trained on large amounts of recommendation data, resulting in lacking the task-specific knowledge
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1ef7/1ef75e63-71c0-41b5-8807-c9cffab1ab60.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Performance comparison under the zero-shot and low-resource settings on the Scientific dataset.</div>
[1, 2]. This further illustrates the importance of exploring more appropriate methods to adapt LLMs to recommendation tasks more effectively. In contrast, our Laser significantly outperforms all traditional methods on all metrics across all datasets. For example, on the Scientific dataset, compared to the best traditional method, our Laser improves Recall@10, NDCG@10, and MRR by 6.97%, 15.07%, and 17.65%, respectively, demonstrating the validity of our method and hopefully inspiring future LLM-based recommendation works. Additionally, we compare the parameter scale of different LLM tuning methods in Table 3. We can observe that our proposed BiTuning is more parameter-efficient, which contains only 0.135M trainable parameters. The results indicate that Laser can greatly reduce the scale of trainable parameters and achieve effective LLM adaptation with the proposed Bi-Tuning method, outperforming SOTA baselines. Note that in Table 3 we only list the trainable parameter scales of different LLM tuning methods. The total number of Laser’s trainable parameters is about 183.3M, which is also significantly less than 3% of the parameter number of the LLM backbone, ChatGLM2-6B.
# 4.3 Zero-Shot and Low-Resource Performance
To further demonstrate the effectiveness of our Laser, we perform experiments to examine its performance in the zero-shot and lowresource scenarios. Specifically, we compare Laser (which uses both the item semantic information and the ID-based collaborative information) with two other types of baselines, including BERT4Rec
<div style="text-align: center;">Table 4: Results of the ablation study. The best results are in bold and the second best results are underlined.</div>
Scientific
Pet
Variants
Recall@10 NDCG@10
MRR
Recall@10 NDCG@10
MRR
Laser
0.1396
0.0970
0.0893
0.1134
0.0898
0.0856
w/o MoE
0.1261
0.0889
0.0795
0.1056
0.0818
0.0763
w/o M-Former
0.1245
0.0844
0.0739
0.1049
0.0775
0.0721
w/o prefix
0.1128
0.0705
0.0619
0.0878
0.0695
0.0607
w/o training stage 1
0.0784
0.0544
0.0509
0.0514
0.0443
0.0398
w/o training stage 2
0.1316
0.0894
0.0781
0.1083
0.0837
0.0793
Table 5: Performance comparison of different item/user embedding generation strategies on the Scientific dataset. The best results are in bold and the second best results are underlined.
Strategies
Recall@10 NDCG@10
MRR
w/ suffix
0.1396
0.0970
0.0893
w/ average pooling
0.0551
0.0416
0.0401
w/ [EOS]
0.0948
0.0688
0.0646
(which uses only the ID-based collaborative information) and RECFORMER (which uses only the item semantic information). We first pre-train these methods (in addition to the ID-based BERT4Rec) on the Pet dataset, and then test whether they can perform well on another domain with no/limited training data. Figure 3 shows the experimental results on the Scientific dataset. From Figure 3, we can observe that: (1) Laser performs best in the zero-shot scenario. Compared to other baselines, Laser achieves significantly better performance (Recall@10 reaches 0.97, NDCG@10 reaches 0.58), even though it has not seen any items on the Scientific dataset. We attribute this superior performance to the design of our Bi-Tuning framework, which fully leverages the generalization capabilities of LLMs and effectively adapts LLMs for sequential recommendation. (2) Laser only needs to use 5% of the training data to exceed the effect of the other two baselines using 100% of the training data. Compared to the other two baselines, Laser’s performance can quickly rise to a very appreciable level as the ratio of training data increases to 5%. This means that we only need a very small amount of training data and training time to migrate the Laser trained on one domain to another unseen domain, accompanied by better results than other baselines that need far more training data. This demonstrates that our proposed framework can effectively transform LLMs into generalizable sequential recommender systems.
# 4.4 Ablation Study
To demonstrate the effectiveness of each module in our Laser, we conduct ablation studies and provide the results in Table 4. We can observe that: (1) The experimental results on the two datasets remain identical. The removal of any module results in a significant decrease in Laser’s performance. (2) Without MoE, Recall@10, NDCG@10, and MRR respectively decrease on average by 8.28%, 8.63%, and 10.92%, showing that the introduction of MoE can help our framework to deal with the diverse collaborative characteristics
Table 6: Performance comparison under different hard prompt templates on the Scientific dataset. The best results are in bold and the second best results are underlined.
Templates
Recall@10 NDCG@10
MRR
original
0.1396
0.0970
0.0893
w/o specified phrase
0.1042
0.0814
0.0782
w/o instruction
0.0985
0.0747
0.0652
w/ two instructions
0.0972
0.0635
0.0567
of different types of users, which leads to higher-quality recommendation results. Furthermore, without the M-Former, the three metrics decrease on average by 9.15%, 13.35%, and 16.51%, respectively. This demonstrates the importance of using ID-based collaborative information for more accurate recommendations, and that our M-Former can effectively integrate collaborative information into LLMs. (3) Without the prefix, Recall@10, NDCG@10, and MRR decrease significantly on average by 20.89%, 24.97%, and 25.39%, respectively, demonstrating the important role of the prefix in adapting LLMs to the recommendation task. (4) Removing any training stage, the effectiveness of the Laser is reduced. Specifically, without the first training stage, Recall@10, NDCG@10, and MRR respectively decline on average by 49.24%, 47.29%, and 48.28%, demonstrating the need to find appropriate parameter weights to obtain the high-quality item embeddings. Besides, without the second training stage, the metrics also decrease by 9.13%, 7.28%, and 9.96%, respectively. This suggests that after obtaining appropriate item embeddings, it’s also necessary to continue training to make Laser better adapt to the fixed item embeddings and achieve the best recommendation results.
# 4.5 Further Discussion In this section, we provide further discussion about our proposed Laser.
4.5 Further Discussion In this section, we provide further discussion about our proposed Laser.
4.5.1 Suffix. We compare the usage of the suffix to two other strategies for generating user/item embeddings, including performing average pooling on all token embeddings output by LLMs and replacing the trainable virtual suffix with a hard token [EOS] which will not be trained. As shown in table 5, compared to the other two strategies, our designed suffix can effectively improve Recall@10, NDCG@10, and MRR by at least 32.07%, 29.04%, and 27.67%, respectively. This proves that a trainable virtual suffix can more effectively convert the LLM output from the language space to the recommendation space, thus generating higher-quality user/item embeddings.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/28f4/28f4aced-bd99-4a71-9ae0-ed08dd303a16.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Performance comparison with different ID-based sequential recommender systems.</div>
4.5.2 Input Text Template. In this work, we utilize a unified template to organize both the user interaction history and the single item, which is shown in Section 3.1.1. The template instructs LLMs to summarize the semantic information into the suffix, which is further used for recommendation. To ensure the template’s plausibility, we compared it with three other variants, including: (1) deleting the specified instruction phrase “into a single token”, (2) deleting the entire instruction “You are an intelligent ... the user has browsed the following items:”, (3) using a different instruction for item embedding generation, “You are an intelligent recommendation assistant. Please summarize the item characteristics into a single token:”. As shown in Table 6, compared to the other three variants, our prompt template can significantly improve Recall@10, NDCG@10, and MRR by at least 25.36%, 16.05%, and 12.36%, respectively. This demonstrates the effectiveness of our prompt template in harnessing the powerful capabilities of LLMs with clear, consistent, and appropriate instruction.
4.5.3 ID-based Sequential Recommender. We perform further experiments to study the effect of the ID-based sequential recommender system on the performance of our Laser. As shown in Figure 4, on all datasets, the performance of Laser increases almost linearly with the performance of the employed ID-based sequential recommender system. For example, on the Pet dataset, Laser based on BERT4Rec outperforms Laser based on SASRec by 17.30%, while BERT4Rec outperforms SASRec by 15.38%. This suggests that our Laser can be further improved by using more powerful ID-based sequential recommender systems.
4.5.4 Parameter Analysis. Furthermore, we perform a detailed parameter analysis to explore the effect of the query expert number 𝐾 and the expert’s virtual token number 𝐿on the Laser’s performance. As shown in Table 7, smaller values of 𝐾and 𝐿cause the M-Former to be insufficient to effectively handle the diverse characteristics of different types of users, while too large values increase the difficulty of training, thus decreasing the effectiveness of Laser. Finally, we respectively set 𝐾and 𝐿to 8 and 32 to get the best results.
# 5 Related Work
# 5 Related Work 5.1 Sequential Recommendation
# 5.1 Sequential Recommendation
Sequential recommendation aims to infer users’ preferences based on their past interactions ordered by timestamps. In traditional sequential recommender systems, items are represented by unique IDs. To effectively capture users’ historical interactions and make recommendations based on these IDs, a variety of methods have
Table 7: The comparison under different 𝐾and 𝐿values on the validation set of the Scientific dataset. The best results are in bold and the second best results are underlined.
<div style="text-align: center;">K L Recall@10 NDCG@10 MRR</div>
K
L
Recall@10 NDCG@10
MRR
8
32
0.1661
0.1199
0.1112
4
32
0.1545
0.1090
0.0992
12 32
0.1560
0.1138
0.1054
8
16
0.1487
0.1099
0.1032
8
48
0.0975
0.0692
0.0644
been employed, including CNNs, RNNs, and GNNs. For example, Caser [34] views the embedding matrix of previous items as an "image" and applies convolutional operations to capture user preferences. GRU4Rec [4] introduces GRU [4] to model user sequential patterns. SRGNN [42], GCE-GNN [38], and SURGE [3] are proposed to capture long-term sequential user preferences through multi-layer message passing. Besides, self-attention-based models have also been widely adopted for sequential recommendation [16, 22, 33]. Although these ID-based methods achieve promising performance, they fail to consider the semantic information of item descriptions, resulting in suboptimal performance. Recently, researchers have attempted to create transferable item representations by encoding item descriptions with language models [12, 21]. However, these works primarily focus on small or medium-sized language models.
# 5.2 LLMs in Recommender Systems
Large Language Models (LLMs) have demonstrated remarkable performance in various domains, prompting researchers to explore their potential in recommendation tasks. Existing works integrate LLMs into recommendations in two main paradigms. The first paradigm is to use LLMs to answer specific recommendation questions by in-context learning [10, 13, 28, 50] or supervised fine-tuning [1, 2, 26, 41, 51], which focuses only on the reranking phase. The second paradigm is to use LLMs as encoders to generate item/user embeddings. For example, Li et al. [23] and Wu et al. [39] attempted to get item/user embeddings by performing pooling on the token embeddings encoded by LLMs. Zhang et al. [49] compressed the textual information into a single special token and learned its embedding using a contrastive learning approach. Although these works show promise, they often require training a large number of parameters to bridge the huge gap between recommendation and language generation tasks, which is resource-demanding. Additionally, these works typically integrate ID-based collaborative signals into LLMs via simple unified linear projections [45, 53], which fails to consider the diverse characteristics of different types of users and thus diminishes recommendation accuracy. In this paper, we propose a parameter-efficient LLM Bi-Tuning framework for sequential recommendation. Besides, to improve the recommendation performance, we introduce a lightweight M-Former to effectively integrate ID-based collaborative information into the LLM.
# 6 Conclusion
In this paper, we propose Laser, a parameter-efficient LLM BiTuning framework for sequential recommendation with collaborative information. Specifically, we present Bi-Tuning, a parameterefficient fine-tuning method that adapts LLMs to sequential recommendation through the trainable prefix and suffix. The prefix adapts LLMs to the recommendation task with collaborative information, while the suffix converts LLM output from the language space to the recommendation space and obtains high-quality user/item embeddings. To effectively integrate ID-based collaborative information for more accurate recommendation, we introduce M-Former, a lightweight MoE-based querying transformer that uses a set of query experts to capture the diverse collaborative characteristics of different types of users. Finally, a multi-task loss function and a two-stage training strategy are employed to train Laser for the sequential recommendation. Extensive experiments on real-world datasets demonstrate that Laser can parameter-efficiently adapt LLMs to effective recommender systems, significantly outperforming state-of-the-art methods.
# References
[1] Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yancheng Luo, and et al. 2023. A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems. (2023). arXiv:2308.08434 [2] Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems. 1007–1014. [3] Jianxin Chang, Chen Gao, Yu Zheng, Yiqun Hui, Yanan Niu, Yang Song, and et al. 2021. Sequential Recommendation with Graph Neural Networks. In SIGIR ’21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 378–387. [4] Junyoung Chung, Çaglar Gülçehre, KyungHyun Cho, and Yoshua Bengio. 2014. Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. (2014). arXiv:1412.3555 [5] Hao Ding, Yifei Ma, Anoop Deoras, Yuyang Wang, and Hao Wang. 2021. ZeroShot Recommender Systems. (2021). arXiv:2105.08318 [6] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. J. Mach. Learn. Res. (2022), 120:1–120:39. [7] Ehsan Gholami, Mohammad Motamedi, and Ashwin Aravindakshan. 2022. PARSRec: Explainable Personalized Attention-fused Recurrent Sequential Recommendation Using Session Partial Actions. In KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 454–464. [8] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, and et al. 2024. ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools. arXiv:2406.12793 [9] Ruining He and Julian J. McAuley. 2016. Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation. In IEEE 16th International Conference on Data Mining (ICDM) (2016). [10] Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, and et al. 2023. Large Language Models as Zero-Shot Conversational Recommenders. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 720–730. [11] Balázs Hidasi, Alex, ros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. In 4th International Conference on Learning Representations. [12] Yupeng Hou, Shanlei Mu, Wayne Xin Zhao, Yaliang Li, Bolin Ding, and Ji-Rong Wen. 2022. Towards Universal Sequence Representation Learning for Recommender Systems. In KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 585–593. [13] Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian J. McAuley, and et al. 2024. Large Language Models are Zero-Shot Rankers for Recommender Systems. In Advances in Information Retrieval - 46th European Conference on Information Retrieval. 364–381. [14] Xu Huang, Jianxun Lian, Yuxuan Lei, Jing Yao, Defu Lian, and Xing Xie. 2023. Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations. (2023). arXiv:2308.16505 [15] Ting Jiang, Shaohan Huang, Zhongzhi Luan, Deqing Wang, and Fuzhen Zhuang. 2023. Scaling Sentence Embeddings with Large Language Models. (2023). arXiv:2307.16645 [16] Wang-Cheng Kang and Julian J. McAuley. 2018. Self-Attentive Sequential Recommendation. In IEEE International Conference on Data Mining. 197–206. [17] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, and et al. 2021. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. In 9th International Conference on Learning Representations. [18] Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The Power of Scale for Parameter-Efficient Prompt Tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 3045–3059. [19] Chenglin Li, Mingjun Zhao, Huanming Zhang, Chenyun Yu, Lei Cheng, Guoqiang Shu, and et al. 2022. RecGURU: Adversarial Learning of Generalized User Representations for Cross-Domain Recommendation. In WSDM ’22: The Fifteenth ACM International Conference on Web Search and Data Mining. 571–581. [20] Jing Li, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Tao Lian, and Jun Ma. 2017. Neural Attentive Session-based Recommendation. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. 1419–1428. [21] Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and et al. 2023. Text Is All You Need: Learning Language Representations for Sequential Recommendation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1258–1267. [22] Jiacheng Li, Yujie Wang, and Julian J. McAuley. 2020. Time Interval Aware SelfAttention for Sequential Recommendation. In WSDM ’20: The Thirteenth ACM International Conference on Web Search and Data Mining. 322–330. [23] Ruyu Li, Wenhao Deng, Yu Cheng, Zheng Yuan, Jiaqi Zhang, and Fajie Yuan. 2023. Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights. (2023). arXiv:2305.11700
[24] Xiang Lisa Li and Percy Liang. 2021. Prefix-Tuning: Optimizing Continuous Prompts for Generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing. 4582–4597. [25] Yunqi Li, Hanxiong Chen, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2021. User-oriented Fairness in Recommendation. In WWW ’21: The Web Conference 2021. 624–632. [26] Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan, and et al. 2024. ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation. In Proceedings of the ACM on Web Conference 2024. 3497–3508. [27] Chong Liu, Xiaoyang Liu, Rongqin Zheng, Lixin Zhang, Xiaobo Liang, Juntao Li, and et al. 2023. CT4Rec: Simple yet Effective Consistency Training for Sequential Recommendation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3901–3913. [28] Junling Liu, Chao Liu, Renjie Lv, Kang Zhou, and Yan Zhang. 2023. Is ChatGPT a Good Recommender? A Preliminary Study. (2023). arXiv:2304.10149 [29] Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Tam, Zhengxiao Du, Zhilin Yang, and et al. 2022. P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 61–68. [30] Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong, Bo Tang, Wenjin Wang, and et al. 2024. CRUD-RAG: A Comprehensive Chinese Benchmark for RetrievalAugmented Generation of Large Language Models. (2024). arXiv:2401.17043 [31] Jianmo Ni, Jiacheng Li, and Julian J. McAuley. 2019. Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing. 188–197. [32] Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized Markov chains for next-basket recommendation. In Proceedings of the 19th International Conference on World Wide Web. 811–820. [33] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and et al. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 1441–1450. [34] Jiaxi Tang and Ke Wang. 2018. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. 565–573. [35] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, and et al. 2023. Llama 2: Open Foundation and Fine-Tuned Chat Models. (2023). arXiv:2307.09288 [36] Xinyuan Wang, Liang Wu, Liangjie Hong, Hao Liu, and Yanjie Fu. 2024. LLMEnhanced User-Item Interactions: Leveraging Edge Information for Optimized Recommendations. (2024). arXiv:2402.09617 [37] Yancheng Wang, Ziyan Jiang, Zheng Chen, Fan Yang, Yingxue Zhou, Eunah Cho, and et al. 2023. RecMind: Large Language Model Powered Agent For Recommendation. (2023). arXiv:2308.14296 [38] Ziyang Wang, Wei Wei, Gao Cong, Xiao-Li Li, Xianling Mao, and Minghui Qiu. 2020. Global Context Enhanced Graph Neural Networks for Session-based Recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 169–178. [39] Chuhan Wu, Fangzhao Wu, Tao Qi, and Yongfeng Huang. 2021. Empowering News Recommendation with Pre-trained Language Models. In SIGIR ’21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1652–1656. [40] Junda Wu, Cheng-Chun Chang, Tong Yu, Zhankui He, Jianing Wang, Yupeng Hou, and et al. 2024. CoRAL: Collaborative Retrieval-Augmented Large Language Models Improve Long-tail Recommendation. (2024). arXiv:2403.06447 [41] Likang Wu, Zhaopeng Qiu, Zhi Zheng, Hengshu Zhu, and Enhong Chen. 2024. Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations. In Thirty-Eighth AAAI Conference on Artificial Intelligence. 9178–9186. [42] Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan. 2019. Session-Based Recommendation with Graph Neural Networks. In The Thirty-Third AAAI Conference on Artificial Intelligence. 346–353. [43] Yunjia Xi, Weiwen Liu, Jianghao Lin, Jieming Zhu, Bo Chen, Ruiming Tang, and et al. 2023. Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models. (2023). arXiv:2306.10933 [44] Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, and et al. 2023. Large Language Models for Generative Information Extraction: A Survey. (2023). arXiv:2312.17617 [45] Zhengyi Yang, Jiancan Wu, Yanchen Luo, Jizhi Zhang, Yancheng Yuan, An Zhang, and et al. 2023. Large Language Model Can Interpret Latent Space of Sequential Recommender. (2023). arXiv:2310.20487 [46] Fajie Yuan, Alex, ros Karatzoglou, Ioannis Arapakis, Joemon M. Jose, and Xiangnan He. 2019. A Simple Convolutional Generative Network for Next Item Recommendation. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining. 582–590.
[47] Zhenrui Yue, Sara Rabhi, Gabriel de Souza Pereira Moreira, Dong Wang, and Even Oldridge. 2023. LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking. (2023). arXiv:2311.02089 [48] Shuxun Zan, Yujie Zhang, Xiangwu Meng, Pengtao Lv, and Yulu Du. 2021. UDA: A user-difference attention for group recommendation. Inf. Sci. (2021), 401–417. [49] Chao Zhang, Shiwei Wu, Haoxin Zhang, Tong Xu, Yan Gao, Yao Hu, and et al. 2024. NoteLLM: A Retrievable Large Language Model for Note Recommendation. (2024). arXiv:2403.01744 [50] Jizhi Zhang, Keqin Bao, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems. 993–999. [51] Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. 2023. Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach. (2023). arXiv:2305.07001 [52] Tingting Zhang, Pengpeng Zhao, Yanchi Liu, Victor S. Sheng, Jiajie Xu, Deqing Wang, and et al. 2019. Feature-level Deeper Self-Attention Network for Sequential Recommendation. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence. 4320–4326. [53] Yang Zhang, Fuli Feng, Jizhi Zhang, Keqin Bao, Qifan Wang, and Xiangnan He. 2023. CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation. (2023). arXiv:2310.19488 [54] Yipeng Zhang, Xin Wang, Hong Chen, and Wenwu Zhu. 2023. Adaptive Disentangled Transformer for Sequential Recommendation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3434–3445. [55] Yaochen Zhu, Liang Wu, Qi Guo, Liangjie Hong, and Jundong Li. 2024. Collaborative Large Language Model for Recommender Systems. In Proceedings of the ACM on Web Conference 2024. 3162–3172.
# A Baselines
comprehensively evaluate the performance of our proposed er, we compare it to state-of-the-art baselines, including six ditional methods and three LLM-based methods. 1) Traditional Baselines: • SASRec [16] employs a self-attention mechanism to capture the semantic relevance between the user interaction sequence and the candidate items. • BERT4Rec [33] is a bidirectional self-attentive model, employing the cloze objective to model users’ dynamic preferences from their historical behaviors. • RecGURU [19] introduces an adversarial learning method to incorporate user information across domains and obtain generalized user representations for sequential recommendation. • FDSA [52] proposes a feature-level self-attention network that integrates different heterogeneous features of items into feature sequences with different weights through a vanilla attention mechanism. • ZESRec [5] utilizes a pre-trained language model to convert item descriptions into feature representations. • RECFORMER [21] formulates items as key-value attribute pairs and utilizes pre-trained language models to encode them for ID-free sequential recommendation. 2) LLM-based Baselines: • LLM4REC [36] proposes a graph knowledge guided attentive LLM recommendation backbone to inject graph edge information into LLMs. • KAR [43] proposes a hybrid-expert adapter that condenses LLM-generated world knowledge into augmented vectors to enhance the performance of recommendation models. • LlamaRec [47] adopts a verbalizer-based approach that transforms LLM output logits into probability distributions over the candidate items.
# (2) LLM-based Baselines:
# B Implementation Details
In this work, we employ a pre-trained BERT4Rec [33] as the frozen ID-based sequential recommender system to encode the user interaction ID sequences. The hyper-parameter settings keep the same as in the original paper, where the number of transformer blocks, the number of attention heads, and the dimension of each attention head are set to 2, 2, and 32, respectively. The frozen LLM we use is the ChatGLM2-6B [8], an impressive open-source large language model with exceptional language modeling capabilities. This model consists of 28 transformer blocks. The hidden size is set to 4096 and the number of attention heads is set to 32. In the feed-forward networks, the dimension of the intermediate layer is set to 13,696. Besides, the model vocabulary consists of 65,024 unique tokens. As for the M-Former, it contains 12 transformer blocks, with alternate blocks conducting cross-attention between the collaborative embeddings and the virtual query expert tokens. The hidden
size and the number of attention heads are set to 768 and 12, respectively. The expert number 𝐾, query token number 𝐿, and the hidden size of the virtual tokens are set to 8, 32, and 768, respectively. Besides, the router and the projection layer are all implemented by single fully-connected layers, whose input/output dimensions are set to 64/8 and 768/4096, respectively. During the training process, the BERT4Rec and the ChatGLM2 are frozen. We randomly initialize the other trainable modules and train them for two stages (as described in Section 3.3.2). Specifically, we set the batch size to 4 and the learning rate to 1e-4. The loss weight hyper-parameter 𝜆is set to 0.01, and the loss temperature hyper-parameter 𝜏is set to 0.05. We use the Adam optimizer and train Laser for 15/5 epochs in the first/second training stage, respectively.
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009
