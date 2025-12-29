# PALR: Personalization Aware LLMs for Recommendation

Zheng Chen ∗
Ziyan Jiang ∗
zgchen@amazon.com Amazon Alexa AI Seattle, WA, USA
ziyjiang@amazon.com Amazon Alexa AI Seattle, WA, USA

Eunah Cho eunahch@amazon.com Amazon Alexa AI Seattle, WA, USA

Xiaojiang Huang xjhuang@amazon.com Amazon Alexa AI Seattle, WA, USA

# ABSTRACT

Large language models (LLMs) have recently received significant attention for their exceptional capabilities. Despite extensive efforts in developing general-purpose LLMs that can be utilized in various natural language processing (NLP) tasks, there has been less research exploring their potential in recommender systems. In this paper, we propose a novel framework, named PALR (P ersonalization A ware L LMs for R ecommendation), aimed at integrating user history behaviors (such as clicks, purchases, ratings, etc.) with LLMs to generate user preferred items. Specifically, we first use user/item interactions as guidance for candidate retrieval, and then adopt an LLM-based ranking model to generate recommended items. Unlike existing approaches that typically adopt general-purpose LLMs for zero/few-shot recommendation testing or training on small-sized language models (with less than 1 billion parameters), which cannot fully elicit LLMs’ reasoning abilities and leverage rich item side parametric knowledge, we fine-tune an LLM of 7 billion parameters for the ranking purpose. This model takes retrieval candidates in natural language format as input, with instructions explicitly asking to select results from input candidates during inference. Our experimental results demonstrate that our solution outperforms state-ofthe-art models on various sequential recommendation tasks.


# CCS CONCEPTS
• Information systems → Recommender systems.

# CCS CONCEPTS

# • Information systems → Recommender systems.

Generative Recommender Model, User Preference Learning, Larg Language Models

ACM Reference Format: Fan Yang, Zheng Chen, Ziyan Jiang, Eunah Cho, Xiaojiang Huang, and Yanbin Lu. 2018. PALR: Personalization Aware LLMs for Recommendation. In Proceedings of Make sure to enter the correct conference title from your rights

∗ Authors contributed equally to this research. Names are ordered alphabetically.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. Conference acronym ’XX, June 03–05, 2018, Woodstock, NY © 2018 Association for Computing Machinery. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/XXXXXXX.XXXXXXX

Ziyan Jiang ∗
ziyjiang@amazon.com Amazon Alexa AI Seattle, WA, USA

Yanbin Lu luyanbin@amazon.com Amazon Alexa AI Seattle, WA, USA

confirmation emai (Conference acronym ’XX). ACM, New York, NY, USA, 5 pages. https://doi.org/XXXXXXX.XXXXXXX

# 1 INTRODUCTION

A recommender system is a type of information filtering system that is designed to predict and recommend items or products. These systems are widely used in e-commerce, online advertising, social media, and entertainment industries. In recent years, the emergence of LLMs, such as Bert[4], GPT-3[1], FLAN-T5[2], has led to significant breakthroughs in NLP research. Inspired by these advancements, researchers have begun exploring the potential of using LLMs in recommendation systems[3, 12, 13, 18]. Compared with traditional recommendation modeling techniques [8, 16, 17] and more recent sequential modeling[9, 18, 19] and graph modeling[6, 25] techniques, LLMs offer several distinct advantages. Firstly, LLMs inherently support inductive learning and negate the need for pre-trained embeddings for each item. Instead, each item can be represented as a piece of text. This feature is particularly crucial in an industry setting where new items are continually emerging. Secondly, LLMs allow for easy integration of various signals, such as metadata, context, and multi-modal signals, into the recommendation process by incorporating them into the model’s prompt. Thirdly, LLMs can transfer knowledge acquired from one domain to another, providing a significant advantage in cold-start scenarios where user behavior data is limited. Finally, LLMs possess vast knowledge and superior reasoning capabilities given their extensive pre-training, along with the ability to generate natural language outputs; therefore they can make recommendations with sensible and human-readable explanations, enhancing user trust and engagement. However, directly leveraging parametric knowledge saved in general-purpose LLMs[2, 22] to generate recommended items is challenging[13]. Firstly, there may be knowledge gaps between LLMs and items that need to be recommended. For instance, some newly released shopping items may not be included in the LLM’s parametric knowledge. Secondly, LLMs are prone to generating incomplete and hallucinatory results, which need an extra knowledge grounding step to eliminate unfavorable results. Thirdly, LLMs have limitations regarding input token length and efficiency. If the items pool is extensive, it is impractical to provide all items as natural language input. As a result, recent research is pruned to treat LLMs as summarization and reasoning engines instead of a knowledge base in recommendation scenarios.

In this paper, we present PALR, which is a general framework for personalized recommendation tasks that combines user behaviors with LLM. Given the challenges mentioned above, we break down the task into several stages. Initially, we use an LLM and user behavior as input to generate user profile keywords. Then, we employ a retrieval module to pre-filter some candidates from the items pool based on the user profile. Importantly, our framework is not reliant on any specific retrieval algorithms. Finally, we use LLM to provide recommendations from those candidates based on user history behaviors. To better adapt these general-purpose LLMs to fit the recommendation scenarios, we convert user behavior data into natural language prompts and fine-tune a LLaMa[22] 7B model. Our goal is to teach the model to learn the co-occurrence of user engaged item patterns. This approach enables us to incorporate user behavior data into the LLM’ reasoning process and better generalize to new users and unseen items. In summary, our contributions are:
(1) We propose PALR, a flexible personalized recommendation framework, which incorporating user behaviors with LLMs for recommended items generation.
(2)  We break down recommendation task into user profile generation, candidates retrieval and items ranking three sub-tasks, and tune instruction prompts to better elicit LLMs reasoning ability.
(3) We fine-tune a recommendation oriented LLM based on LLaMa 7B. Evaluation under PALR framework on two public datasets demonstrate its competitive performance against state-of-the-art methods.
(4) We experimented with two datasets, MovieLens-1M[5], and Amazon Beauty[14] and demonstrated the strong potential of an LLM for recommendation in comparison to SOTA.

# 2 METHODOLOGY

# 2 METHODOLOGY 2.1 PALR Framework

# 2.1 PALR Framework

Our proposed method, PALR (Personalization Aware LLM for Recommendation), is illustrated in Figure 1. It utilizes a multi-step approach to harness the potential of LLMs for recommendation.
• Natural Language user profile generation. When a user interacts with a large number of items and exhibits mixed affinities, it can be challenging for the model to provide accurate recommendations based solely on user behaviors. In such situations, having a high-level summarization of the user’s preferences can be beneficial. An LLM can be leveraged to generate a summary of a user’s preferences. For example, by analyzing a user’s music and TV viewing history, we can generate a summary of their preferences such as "pop music" and "fantasy movies."
• Candidates retrieval. To address the issues of hallucination and incompleteness in the generated results, a retrieval module is utilized to ground knowledge and filter out results that do not relevant to the task at hand, resulting in a much smaller candidate pool to feed into the LLM for further processing. This framework can accommodate various retrieval models, such as a sequential recommendation model trained on user behaviors, which can serve this purpose effectively.
• Item recommendation. By combining the interaction history, natural language user profile and retrieved candidates,

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/dcdb/dcdb7219-4c44-441c-a351-1a6323574072.png" style="width: 50%;"></div>
Figure 1: Here is an overview of our proposed PALR architecture. The "Natural Language Prompt" for the "LLM for recommendation" comprises three components: the "Interaction History Sequence," the "Natural Language User Profile," and the "Candidates for Recommendation". The "Interaction History Sequence" is created by simply concatenating the items that the user has interacted with. The "Natural Language User Profile" is a high-level summarization of the user’s preferences, generated using an LLM based on useritem interactions, item attributes, or even user information if possible. The "Candidates for Recommendation" are the output of a retrieval model, and in our design, we have the flexibility to use any retrieval model for this purpose. We have included an example in Figure 2.

we can create a natural language prompt that can be fed into the LLM for recommendation. The model will utilize its reasoning ability to select the items from the candidates pool that align best with user profile.

The steps of "user profile generation" and "item recommendation" require dedicated prompt design to effectively leverage the reasoning ability of LLMs. An example of related prompt design in the movie recommendation task is shown in Figure 2.

# 2.2 Fine-Tuning

Through our investigation, we find fine-tuning is necessary to make the model 1) obtain reasonably strong performance, and 2) recognize the retrieval layer and performs the retrieval as expected. We employ instruction-based fine-tuning, a technique proven effective in recent LLM development [21, 23, 24].

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/44c7/44c7559f-552a-4d77-80f0-42080d2d530e.png" style="width: 50%;"></div>
We have created two types of instruction tasks called "Recommend" and "Recommend_Retrieval". The "Recommend" task involves a list of items that the user has interacted with in the past (with a maximum limit of 20 items), and the objective of the model is to generate a list of "future" items that the user may interact with. Here’s an example of such an instruction for the Movielens dataset. We refer to a model fine-tuned by this instruction as 𝑃𝐴𝐿𝑅 𝑣 1.

Task Instruction: Recommend 10 other movies based on user’s watching history. Input: User watched movies "Pink Floyd - The Wall","Canadian Bacon", "G.I. Jane", ..., "Down by Law". Output:"Almost Famous", "Full Metal Jacket", ...

The "Recommend_Retrieval" task asks the model to retrieve the target "future" items from a list of candidate items. The candidate list contains all target items, plus a few negative items similar to the target items (e.g. movies with the same genres, co-watched by many users). The following are two examples of such instruction used in our fine-tuning for the Movielens dataset and the Amazon Beauty dataset. For the Amazon beauty dataset, we include item ID for evaluation. We refer to a model fine-tuned with both "Recommend" and "Recommend_Retrieval" instruction as 𝑃𝐴𝐿𝑅 𝑣 2.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/544b/544bdad5-17ad-4899-9170-6942b04a714b.png" style="width: 50%;"></div>
Task Instruction: Recommend 10 other movies based on user’s watching history from the candidate list. Input: User watched movies "Pink Floyd - The Wall","Canadian Bacon", "G.I. Jane", ..., "Down by Law". The candidates are “The Blues Brothers”, "Platoon", ..., "Almost Famous". Output:"Almost Famous", "Full Metal Jacket", ...

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/84b5/84b5efb8-1c93-49ce-b50c-625eea93e097.png" style="width: 50%;"></div>
Task Instruction: Recommend 10 other items based on user’s history from the candidate list. Input: User has purchased the following products"B000142FVW (Opi Nail Lacquer, Not So Bora Pink, 0.5 Fluid Ounce)", ..., "B001PPMLS8 (MoroccanOil Treatment Light 3.4 Oz)". The candidates are "B000SQRMOS (FHI Heat Hot Sauce (50ml), 1.7 fluid ounces bottle)", ...,"B000G1MT2U (Mixed Chicks Leave-In Conditioner)". Output:  "B000G1MT2U (Mixed Chicks Leave-In Conditioner)", "B001AO0WCG (Moroccan Oil Hair Treatment 3.4 Oz Bottle with Blue Box)", ...

It is worth noting that the fine-tuning is retrieval-layer agnostic. Despite our objective being to train the model to select from a list of candidates, the construction of this list for fine-tuning is not bound to the retrieval layer in our framework. Furthermore, we have found that the fine-tuning process is enhanced by a couple of techniques: 1) enriching shorter lists in the datasets with items from the user’s 3-hop affinity; 2) randomly swapping items between the instruction and the generation label. Last but not the least, we only fine-tune on 20% of users. We intend to demo the strong inductive learning capabilities of LLMs. This is not possible for item-embedding based models such as [10, 18], which must be trained on the full data to function effectively.

# 3 EXPERIMENTS

# 3.1 Experiments Settings

3.1.1 Datasets. The two public datasets are collected from the real-world platforms and have been widely used for sequential recommendation. Amazon Beauty 1 is one category of Amazon review datasets, which contains a collection of user-item interactions

1 https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

Dataset
# Users
# Items
# Interactions
Beauty
22,363
12,101
198,502
Movielens-1M
6,040
3,416
999,611
Table 1: Statistics of datasets after preprocessing.
Table 1: Statistics of datasets after preprocessing.

Table 1: Statistics of datasets after preprocessing.

on Amazon spanning from May 1996 to July 2014. Movielens-1M 2
is a common benchmark dataset that includes one million movie ratings. For dataset preprocessing, we follow the common practice[10, 18]. We convert all numeric ratings or presence of a review to “1” and others to “0”. Then, for each user, we discard duplicated interactions and then sort their historical items by the interacted time step chronologically to obtain the user interacted sequence. It is worth mentioning that to guarantee each user/item with enough interactions, we follow the preprocessing procedure in[7, 10], which only keeps the “5-core” datasets. We discard users and items with fewer than 5 interaction records iteratively. The statistics of these datasets are reported in Table 1.
3.1.2 Evaluation. We adopt the leave-one-out strategy to evaluate the performance of each method, which is widely employed in many related works. For each user, we hold out the last interacted item as the test data and utilize the item just before the last as the validation data. The remaining items are used for training. We evaluate each method on the whole item set without sampling as suggested in previous study [11]. We employ Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG) to evaluate the performance. HR focuses on the presence of the positive item, while NDCG further takes the rank position information into account.

# 3.1.3 Baselines. To verify the effectiveness of our method, we compare it with the following representative baselines.

• BPR-MF [15]. It utilizes matrix factorization to model users and items with the pairwise Bayesian Personalized Ranking (BPR) loss.
• NCF [7]. It employs a neural network architecture to model non-sequential user-item interactions instead of the inner product used by matrix factorization.
• GRU4Rec [8]. It utilizes GRU to model the sequential behavior of users for recommendation.
• Caser [20]. It devises horizontal and vertical CNN to exploit user’s recent sub-sequence behaviors for recommendation.
• SASRec [10]. It models user sequences through self-attention modules to capture users’ dynamic interests. and it is a competitive benchmark in sequential recommendation.

# 3.2 Overall Performance Comparison

Table 2 summarizes the best results of all models on two benchmark datasets. As shown in Table 2, our 𝑃𝐴𝐿𝑅 𝑣 2 outperforms multiple baselines by a large margin on two datasets. A comparison between 𝑃𝐴𝐿𝑅 𝑣 1 and 𝑃𝐴𝐿𝑅 𝑣 2 reveals the crucial role of candidates retrieval in improving performance. As we mentioned before, our framework does not depend on any particular retrieval algorithms.

2 http://files.grouplens.org/datasets/movielens/ml-1m.zip

Dataset
Model
HR@10
NDCG@10
Beauty
BPR-MF
0.0299
0.0122
NCF
0.0293
0.0130
GRU4Rec
0.0194
0.0091
Caser
0.0282
0.0136
SASRec
0.0617
0.0283
𝑃𝐴𝐿𝑅𝑣1
0.0181
0.0101
𝑃𝐴𝐿𝑅𝑣2
0.0721
0.0446
ML-1M
BPR-MF
0.0354
0.0158
NCF
0.0313
0.0143
GRU4Rec
0.1017
0.0468
Caser
0.1338
0.0614
SASRec
0.1978
0.1192
𝑃𝐴𝐿𝑅𝑣1
0.1216
0.0569
𝑃𝐴𝐿𝑅𝑣2
0.2110
0.1276
: Experimental results on the two datasets. T
Table 2: Experimental results on the two datasets. The be results are in boldface.

Table 2: Experimental results on the two datasets. The best results are in boldface.

Ideally, PALR can function as an effective ranking model in conjunction with various retrieval methods.In this paper, we utilize SASRec as our retrieval layer and consider its top 50 recommendations. By comparing 𝑃𝐴𝐿𝑅 𝑣 2 and SASRec, it’s obvious that the top10 recommendations re-ranked by our PALR are superior to the original recommendations provided by SASRec. We also evaluate our framework using different recommendation algorithms, including BERT4Rec and LightGCN, and observe a similar trend. By conducting various experiments, we are able to gain a deeper understanding of the significance of fine-tuning. We could observe 𝑃𝐴𝐿𝑅 𝑣 1 has shown some ability to connect historical interacted items with possible future interacted items. Prior to fine-tuning, the model tends to only recommend popular movies in movie recommendation tasks. However, 𝑃𝐴𝐿𝑅 𝑣 1 isn’t able to retrieve the target item from a list of candidate items. We have tried to use 𝑃𝐴𝐿𝑅 𝑣 1 for retrieval and observe that it could only randomly select from the candidates. The performance from 𝑃𝐴𝐿𝑅 𝑣 2 has demonstrated the effectiveness of incorporating an additional instruction during the fine-tuning stage.

# 4 CONCLUSION

The paper introduces PALR, a novel generative framework for producing personalized recommendations, which utilizes a multi-step paradigm to better utilize the knowledge in LLMs’ parameters and reasoning abilities for sequential recommendation tasks. Additionally, the paper discusses the recent advances in LLMs and how they can be leveraged for recommendation tasks. Besides its competitive experiment results mentioned in the paper, LLMs has some other unique benefits in the recommendation task. The first advantage of using LLMs in recommendation tasks is the ease with which external knowledge from different sources can be incorporated into the framework. The second advantage is that LLMs offer an easier pathway to more complex recommendation scenarios, including explainable recommendations and conversational recommendations. Moving forward, our research will focus on further leveraging LLMs in recommendation tasks while ensuring a balance between their

powerful capabilities and latency. As LLMs can be computationally intensive, we will explore ways to optimize their performance and reduce latency without sacrificing accuracy or personalization.

# ACKNOWLEDGEMENT

ACKNOWLEDGEMENT
We thank the LLaMA team for giving us access to their models.

We thank the LLaMA team for giving us access to their model

# REFERENCES

[1]  Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. J. Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. ArXiv abs/2005.14165 (2020).
[2] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022).
[3]  Zeyu Cui, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. 2022. M6Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems. arXiv preprint arXiv:2205.08084 (2022).
[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. ArXiv abs/1810.04805 (2019).
[5] F Maxwell Harper and Joseph A Konstan. 2015. The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis) 5, 4 (2015), 1–19.
[6] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 639–648.
[7] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering. Proceedings of the 26th International Conference on World Wide Web (2017).
[8] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015. Session-based Recommendations with Recurrent Neural Networks. CoRR abs/1511.06939 (2015).
[9]  Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM). IEEE, 197–206.
[10]  Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. 2018 IEEE International Conference on Data Mining (ICDM) (2018), 197–206.
[11]  Walid Krichene and Steffen Rendle. 2020. On Sampled Metrics for Item Recommendation. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (2020).
[12] Jinming Li, Wentao Zhang, Tian Wang, Guanglei Xiong, Alan Lu, and Gerard Medioni. 2023. GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation. arXiv preprint arXiv:2304.03879 (2023).
[13] Junling Liu, Chao Liu, Renjie Lv, Kang Zhou, and Yan Zhang. 2023. Is ChatGPT a Good Recommender? A Preliminary Study. arXiv preprint arXiv:2304.10149 (2023).
[14] Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP). 188–197.
[15] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. ArXiv abs/1205.2618 (2009).
[16]  Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized Markov chains for next-basket recommendation. In The Web Conference.
[17] Guy Shani, David E. Heckerman, and Ronen I. Brafman. 2002. An MDP-Based Recommender System. ArXiv abs/1301.0600 (2002).
[18] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer.  Proceedings of the 28th ACM International Conference on Information and Knowledge Management (2019).
[19] Qiaoyu Tan, Jianwei Zhang, Ninghao Liu, Xiao Huang, Hongxia Yang, Jingren Zhou, and Xia Hu. 2021. Dynamic Memory based Attention Network for Sequential Recommendation. In AAAI Conference on Artificial Intelligence.

[20]  Jiaxi Tang and Ke Wang. 2018. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding. Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining (2018).
[21] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford Alpaca: An Instruction-following LLaMA model. https://github.com/tatsu-lab/stanford_ alpaca.
[22] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).
[23] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. Self-Instruct: Aligning Language Model with Self Generated Instructions. ArXiv abs/2212.10560 (2022).
[24] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2021. Finetuned Language Models Are Zero-Shot Learners. ArXiv abs/2109.01652 (2021).
[25] Teng Xiao, Shangsong Liang, and Zaiqiao Meng. 2019. Hierarchical Neural Variational Model for Personalized Sequential Recommendation. The World Wide Web Conference (2019).

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2

