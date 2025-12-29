# Leveraging Large Language Models for Pre-trained Reco

# Leveraging Large Language Models for Pre-trained Recommender Syst

# Zhixuan Chu *1, Hongyan Hao *1, Xin Ouyang 1, Simeng Wang 1, Yan Wang 1, Yue Shen 1, Jin Qing Cui 1, Longfei Li 1, Siqiao Xue 1, James Y Zhang 1, Sheng Li 2

1 Ant Group 2 University of Virginia {chuzhixuan.czx, hongyanhao.hhy, xin.oyx, simeng.wsm, luli.wy, zhanying, jinjie.gujj, cuiqing.cq, longy james.z} @antgroup.com, shengli@virginia.edu

1 Ant Group 2 University of Virginia

1 Ant Group 2 University of Virginia {chuzhixuan.czx, hongyanhao.hhy, xin.oyx, simeng.wsm, luli.wy, zhanying, jinjie.gujj, cuiqing.cq, longyao.llf, siqiao.xsq, james.z} @antgroup.com, shengli@virginia.edu

{chuzhixuan.czx, hongyanhao.hhy, xin.oyx, simeng.wsm, luli.wy, zhanying, jinjie.gujj, cuiqing.cq james.z} @antgroup.com, shengli@virginia.edu

Abstract

Recent advancements in recommendation systems have shifted towards more comprehensive and personalized recommendations by utilizing large language models (LLM). However, effectively integrating LLM’s commonsense knowledge and reasoning abilities into recommendation systems remains a challenging problem. In this paper, we propose RecSysLLM, a novel pre-trained recommendation model based on LLMs. RecSysLLM retains LLM reasoning and knowledge while integrating recommendation domain knowledge through unique designs of data, training, and inference. This allows RecSysLLM to leverage LLMs’ capabilities for recommendation tasks in an efficient, unified framework. We demonstrate the effectiveness of RecSysLLM on benchmarks and real-world scenarios. RecSysLLM provides a promising approach to developing unified recommendation systems by fully exploiting the power of pre-trained language models.

# Introduction

The realm of recommendation has gained considerable attention in recent years due to its ability to drive business growth and enhance user engagement. Recent advancements in recommender systems have shifted towards incorporating diverse information and catering to a broader range of application scenarios, rather than focusing on task-specific architectures. This shift has been driven by the need for more comprehensive and personalized recommendations, as well as the availability of new data sources and knowledge (Geng et al. 2022; Chu et al. 2022; Hui et al. 2022; Sheu et al. 2021; Li and Zhao 2021; Jiang et al. 2022; Xue et al. 2021). In addition, with the advent of the Large Language Model (LLM) (Radford et al., 2019; Brown et al. 2020; Ouyang et al. 2022), we have witnessed an unprecedented surge in the capabilities of natural language processing. The power of LLM lies in its ability to understand and generate humanlike language. LLM has also enabled the extraction of implicit knowledge from text data (Gu et al. 2023; Yoneda et al. 2023; Zhao et al. 2023). This newfound capability of LLM has opened up exciting avenues for the integration of semantic information into recommender systems and provides a wealth of insights into user preferences and behaviors (Shi

* These authors contributed equally.

et al. 2023; Zhao, Tan, and Mei 2022). As a result, incorporating LLM into recommender systems has become a crucial step toward providing a powerful and comprehensive paradigm for recommendation tasks. In the following, we will discuss the new generation of recommendation model paradigms from two directions, i.e., the unified pre-trained recommendation model and the combination of LLM and recommendation model. On the one hand, training a pre-trained recommendation model can help overcome the limitations of existing recommendation approaches that require designing task-specific architectures and training objectives. Traditional recommendation methods have focused on a single task, such as personalized product recommendations, contextual advertising, customer segmentation, and so on, making them less adaptable to new tasks and limiting their ability to generalize to new domains. By training a pre-trained recommendation model, we can leverage the power of pre-trained models to learn generalizable representations of user behavior and product characteristics (Tsai et al. 2023; Zhao, Tan, and Mei 2022) that can be applied to a variety of recommendation tasks. Overall, a pre-trained recommendation model provides a flexible and scalable solution that can be adapted to a variety of recommendation tasks. Since recommendation tasks usually share a common user–item pool, features, behavioral sequences, and other contextual information, we believe it is promising to merge even more recommendation tasks into a unified framework so that they can implicitly transfer knowledge to benefit each other and enable generalization to other unseen tasks (Xie et al. 2022). On the other hand, integrating LLMs into recommendation systems has several significant advantages. These advantages are linked to the LLM’s capabilities in thinking, reasoning, and discovering implicit relationships within textual data based on the entailment of wealthy background knowledge and logical chains. (1) By leveraging the semantic information in natural language data, LLMs can help the recommendation system understand and infer the relationship between user features and behavioral sequences and among entities in behavioral sequences. This allows the recommendation system to understand the user’s needs and preferences in a more comprehensive way. (2) Another benefit of integrating LLMs into recommendation systems is the ability to leverage the implicit knowledge that is hidden in

the models. LLMs are trained on vast amounts of textual data and can help to understand the relationships between different concepts and ideas. By incorporating LLMs into recommendation systems, this implicit knowledge can be used to generate more divergent and logical recommendations. This can lead to more creative and unexpected recommendations that the user may not have considered otherwise. (3) By leveraging the natural language processing capabilities of LLMs, recommendation tasks that previously required separate specialized systems can now be integrated into a unified framework. The pretrained knowledge and few-shot learning abilities of LLMs allow recommendation models to be rapidly adapted to new domains with limited data. Overall, the natural language processing power and versatility of LLMs can help merge more recommendation tasks into a unified framework. Furthermore, a comprehensive survey on recommendations and LLMs is provided in the Appendix. This survey covers the motivation behind them, current development, and challenges. However, constructing a robust and integrated recommendation system that fully utilizes large language models’ immense knowledge and reasoning capacities poses several key challenges. Directly training a pre-trained recommendation model from scratch is not only a waste of time and data collection efforts but also lacks general common sense and reasoning capabilities that underpin modern large language models. Meanwhile, directly fine-tuning a pre-trained LLM model on recommendation data also has drawbacks. Recommendation data has distinct characteristics - such as fixed entities and sequential user behaviors - that differ from the raw text corpora used to train language models. As such, fine-tuning may erase much of the capabilities specific to recommendation tasks. Therefore, we propose a novel pretrained recommendation paradigm (RecSysLLM) based on the pre-trained large language model through unique designs for recommendation in three phases, i.e., data phase, training phase, and inference phase. Our model retains the reasoning ability and rich knowledge contained in large language models while integrating the recommendation-specific knowledge. It directly inherits the parameters and framework of the original large language model but also designs and extends some mechanisms in the data phase (textualization and sampling), training phase (mask, position, and ordering), and inference phase (dynamic position infilling). These modifications do not discard the tokenization, parameters, structure, or previously learned knowledge in the LLM. On this basis, recommendation data is used to fine-tune it. The significant advantage of this pre-trained recommendation model is that it can utilize the reasoning capabilities and rich knowledge of large language models while incorporating domain-specific knowledge of the recommendation system through parameter-efficient fine-tuning of userprofiles and behavioral sequences data. Another crucial benefit of this model is that it can be easily adapted to different downstream recommendation sub-tasks. We evaluate the proposed model on extensive benchmark datasets and realworld scenarios. The experimental results demonstrate its effectiveness in improving the quality of recommendations. Overall, our proposed pre-trained recommendation model

provides a promising approach for building recommendation systems that are efficient, effective, and unified.

# RecSysLLM Pretraining Mechanism

To fully take advantage of LLM and domain knowledge in recommendation tasks, we need to modify the LLM and fine-tune the existing LLM to get a pre-trained recommendation model. However, the conventional large language models are trained on general knowledge and coherent corpus, and the framework of the model is not designed for behavioral sequence data and recommendation tasks. To address these two points, we make modifications from three phases, i.e., data, training, and inference phases, to transform a conventional pre-trained language model into a pre-trained recommendation model. The whole framework is illustrated in Figure 1. This pre-trained recommendation model has been employed in real-world applications in Chinese scenarios, so we take the GLM (Du et al. 2021) as an example to introduce the RecSysLLM pretraining mechanism, which is bilingual in Chinese and English. Our model can also be adapted to other large language models with minor modifications.

# Data Phase

In the data phase, textualizing tabular data is often the easiest and most straightforward approach for implementing large language models. For the pre-training of RecSysLLM, we first textualize conventional tabular data, such as user features stored in a table with rows and columns into text. Since large language models are originally trained on textual data, text-based features can be easily combined with text-based behavioral sequences and other text information, which helps our model better capture the relationship between features and behavioral sequences. In addition, textualizing tabular data allows for greater flexibility in how they are used in the following tasks. Compared with ordinary language texts, the training texts in the recommendation system should take into account the interests and preferences of users from different periods (Yu et al. 2019). Long-term preferences are usually stable and reflect the general preferences of a user. These preferences do not change frequently over time, but they lack timeliness and may not reflect current interests. On the other hand, short-term preferences tend to change frequently over time and are more reflective of a user’s current interests. We aim to use different periods of preferences to provide accurate and relevant recommendations to users, which can balance the user’s general interests with their current needs. Therefore, we sample behavioral sequences in long-term preferences (10%), medium-term preferences (30%), and short-term preferences (60%). Long-term preferences capture the user’s preferences that have remained consistent for an extended period of time, typically spanning over several months or years. Medium-term preferences capture the user’s preferences that have developed and changed over a shorter period of time, typically spanning over several weeks or months. Short-term preferences can improve recommendation accuracy by providing the system with the user’s most recent preferences, spanning over several days or hours.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c772/c7729eb6-83da-472c-b48c-b04e14dca1cd.png" style="width: 50%;"></div>
Figure 1: This is the framework of RecSysLLM based on a pre-trained generative language model (GLM). To transform the GLM into a specialized model for recommendation systems, several modifications are made while preserving the core knowledge and capabilities of the original language model architecture, such as the new mask mechanism, span order, positional encoding, dynamic position mechanism, and so on.

# Training Phase

To be consistent with the architecture of GLM, our model is still trained by optimizing an autoregressive blank infilling objective based on an input text x = [x 1, · · ·, x n]. Different from the general language text in GLM, our input text is composed of user features and behavioral sequences. Although textualized user features and behavioral sequences are also composed of multiple tokens, they often represent a complete meaning as a whole. If they are split into different parts, like regular text, they will lose their unique meaning. In addition, the LLM’s power comes from the way it tokenizes and processes text. It has been trained on a vast amount of data and has learned to recognize patterns and relationships between tokens, enabling it to identify entities accurately and extract information. If we were to create a new tokenization method, we would lose the LLM’s power. Therefore, to maintain the LLM’s power and supplement the new knowledge in the recommendation data, it is best to leverage the existing tokenization and enhance it with additional information and capabilities rather than create a new tokenization. In the following, we name the attributes in user features and items in the behavioral sequences as entities, which means that they are complete units and have fixed meanings. Therefore, as shown in the “Entities” of Figure 1, our data are composed of plain language text and entities, where (x 1, x 2, and x 3) have merged to form e 1 and (x 6 and x 7) to form e 2. x 4 and x 5 are separate tokens.
Mask Mechanism. To inject the new knowledge of recommendation tasks based on the original LLM, we follow the principle in the LLM and design the new mask mechanism and position strategies. Similar to the GLM (Du et al. 2021), multiple text spans {s 1, · · ·, s m} are sampled, where each span s i corresponds to a series of consecutive tokens [s i, 1, · · ·, s i,l i] in x. Each span is replaced with a single [MASK] token. The remaining text and [MASK] s form a corrupted text x corrupt. In the GLM, since there is no existence of entity, the tokens can be randomly sampled into spans. However, in our model, the multiple and consecutive tokens composing an entity should not be split into different parts. In other words, the tokens of an entity are treated as

a whole. The [MASK] mechanism will not break the  complete entities, which will highlight the whole structure of entities and help to capture the interrelationship between entities. For example, as shown in the “Masks” of Figure 1, x 1, x 2, and x 3 composing the e 1  are blocked as a whole and single token x 5 is also blocked. Therefore, we form the x corrupt with [M], x 4, [M], x 6, and x 7 in the “Division” of Figure 1. Compatible with different natural language processing tasks, we adopt the multi-task pretraining setup (Du et al. 2021) with entity-level [M], sentence-level [sM], and document-level [gM]. Specifically, entity-level refers to the randomly blanking out continuous spans of tokens from the input text, following the idea of autoencoding, which captures the interdependencies between entities. Sentence level restricts that the masked spans must be full sentences. Document-level is to sample a single span whose length is sampled from a uniform distribution over 50%–100% of the original length. The objective aims for long text generation.
Span Order. We implement the autoregressive blank infilling objective with the following techniques. The input x is divided into two parts: one part is the corrupted text x corrupt, and the other consists of the masked spans. Our model automatically learns a bidirectional encoder for the first part and a unidirectional decoder for the second part in a unified model. The model predicts the missing tokens in the spans from the corrupted text in an autoregressive manner, which means when predicting the missing tokens in a span, the model has access to the corrupted text and the previously predicted spans. Instead of randomly permuting the order of the spans in the original GLM (Du et al. 2021), we keep all spans in chronological order to keep the interrelationship among different entities. Formally, we define the pretraining objective of a lengthm index sequence [1, 2, ..., m] as

(1)

�
Positional Encoding. To enable autoregressive generation, each span is padded with special tokens [START] and [END], for input and output, respectively. To be consistent with the original LLM, we cannot arbitrarily modify, add, or

reduce the original positional strategies. Therefore, we extend 2D positional encodings (Du et al. 2021) based on entities. Specifically, each token is encoded with two positional ids, i.e., inter-position and intra-position ids. The inter-position id represents the position in the corrupted text x corrupt. For the masked spans, it is the position of the corresponding [MASK] token. For the intra-position id, we follow the essential meaning in the original LLM, which still refers to the intra-position. Instead of the scope of the whole span, we extend it into a finer granularity. For the entities, it represents the intra-relationship among entities. As shown in Figure 1, for separate tokens (not in the entities) in the encoder part ([M], x 4, [M]), their intra-position ids are 0. For consecutive tokens in the entities (x 6 and x 7), they are numbered in chronological order. For tokens in the autoregressive blank infilling part, they range from 1 to the length of the entities including [S], such as (entities: [S], x 1, x 2, x 3 → 1, 2, 3, 4) and (independent token: [S], x 5 → 1, 2). The two positional ids are projected into two vectors via learnable embedding tables, which are both added to the input token embeddings.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ff0a/ff0a0c41-17aa-4d95-b78f-4c3b1858d611.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: This is the dynamic position mechanism. When one token is generated, it will be judged as one part of an entity or not. If it and the previous token belong to one entity, the intra-position id will continue to grow. Otherwise, it will start at 1 again.
</div>
Figure 2: This is the dynamic position mechanism. When one token is generated, it will be judged as one part of an entity or not. If it and the previous token belong to one entity, the intra-position id will continue to grow. Otherwise, it will start at 1 again.

# Inference phase

Because our pre-trained model is designed to fit different downstream tasks, the length of the generated text should be unknown beforehand and flexible for the different tasks. Further, due to the existence of entities, the intra-position ids represent the relative position of the entity. As shown in the “Inference Phase” of Figure 1, we cannot specify the intraposition ids in advance when autoregressive blank infilling. Hence, we designed a dynamic position mechanism for the mask and position modifications made during the inference phase. It can conduct the autoregressive judgment to determine and complement the intra-position ids one by one as each token is generated in the autoregressive generation procedure. Specifically, we establish an entity pool beforehand, which stores all the tokens of the entities that existed in our recommendation task. When one token is generated, it will be judged as one part of an entity or not. We utilize the Trie

algorithm (Bodon and R´onyai 2003) to check whether the generated token and previous token belong to the same entity, which is a tree data structure used for locating specific keys from within a set. If they belong to one entity, the intraposition id will continue to grow. Otherwise, it will start at 1 again. The detailed procedure is illustrated in Figure 2.

# Experiments

# Experimental Setup

Datasets. We evaluate our method on three real-world ecommerce datasets from Amazon.com, spanning the categories of Sports & Outdoors, Beauty, and Toys & Games. The datasets contain user ratings and reviews from 2019, along with transaction records between January 1 and December 31 (Zhou et al. 2020; Xue et al. 2022, 2023). Key statistics of the resulting datasets are provided in Table 1.
Metrics. Following the experiments in (Geng et al. 2022), we cover five different task families – rating, sequential recommendation, explanation, review, and direct recommendation to facilitate the multitask pretraining for the recommendation. For rating prediction, we adopt Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) as evaluation metrics. For sequential recommendation and direct recommendation tasks, we employ topk Hit Ratio (HR@ k) and Normalized Discounted Cumulative Gain (NDCG@ k) to evaluate the performance and report HR@1, 5, 10 and NGCG@5, 10. For explanation generation and review summarization, we evaluate different methods with BLEU-4, ROUGE-1, ROUGE-2, and ROUGE-L. Lower values of RMSE and MAE indicate better performance, while higher values are preferred for all other metrics. In all result tables, bold numbers represent the best performance, while underlined numbers refer to the second-best performance.
Baselines for Multiple Tasks To demonstrate competence on a wide range of recommendation-related tasks, we adopt the same representative approaches as (Geng et al. 2022) for different tasks, such as Rating Prediction (MF (Koren, Bell, and Volinsky 2009) and MLP (Cheng et al. 2016)), Direct Recommendation (BPR-MF  (Rendle et al. 2009), BPR-MLP (Cheng et al. 2016), and SimpleX (Mao et al. 2021)), Sequential Recommendation (Caser (Tang and Wang 2018), HGN (Ma, Kang, and Liu 2019), GRU4Rec (Hidasi et al. 2016), BERT4Rec (Sun et al. 2019), FDSA (Zhang et al. 2019), SASRec (Kang and McAuley 2018), and S 3-Rec  (Zhou et al. 2020)), Explanation Generation (Attn2Seq (Dong et al. 2017), NRT (Li et al. 2017), PETER (Li, Zhang, and Chen 2021), and PETER+), and review summarization (T0 (Sanh et al. 2022) and  GPT2 (Radford et al. 2019)). The detailed baselines are provided in the Appendix.

# Implementation

To facilitate the multitask prompt-based pretraining for the recommendation, Geng et al. (2022) created a collection of personalized prompt templates. The collection covers five different task families – rating, sequential recommendation, explanation, review, and direct recommendation. The

Table 1: Basic statistics of the experimental datasets.

Dataset
Sports
Beauty
Toys
#Users
35,598
22,363
19,412
#Items
18,357
12,101
11,924
#Reviews
296,337
198,502
167,597
#Sparsity (%)
0.0453
0.0734
0.0724
prompts include personalized fields for users and items to help the model discover user-item preferences. For rating prediction, prompts ask to predict a user’s rating or preference for an item. For sequential recommendation, prompts ask to predict the next item a user will interact with. For explanation, prompts ask to generate text explaining a user’s preferences. For review, prompts summarize or predict ratings from reviews. For direct recommendation, prompts ask whether to recommend an item to a user. The complete collection of personalized prompts with examples is provided in the Appendix of (Geng et al. 2022). These prompts enable the building of diverse training examples from raw data for multitask pertaining. We pretrain our RecSysLLM with diverse training examples with different prompt templates from all five task families to verify its multitask learning ability. Besides, we adopt a part of prompts in each task family for zero-shot evaluation while all remaining prompts are utilized for multitasking prompted pretraining. As a result, we are able to not only compare the performance across various recommendation tasks but also evaluate the zero-shot generalization capability on unseen prompts. Our RecSysLLM model for these English language tasks leverages the powerful GLM-10B for English (Du et al. 2021) model as a foundation. GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on various natural language understanding and generation tasks. Our approach builds on this pre-trained GLM-10B foundation by utilizing a parameterefficient fine-tuning method called LoRA (Low-Rank Adaptation) (Hu et al. 2021) to adapt the model to our specific recommendation tasks. LoRA enables efficiently customizing the enormous GLM-10B model to specialized domains by learning a low-dimensional decomposition of the model update. This allows us to tap into GLM-10B’s broad language knowledge while calibrating it to our RecSysLLM objectives. We inject trainable rank decomposition matrices into each query key value, dense, dense h to 4 h and dense 4 h to h  layer of Transformer architecture in GLM10B. We pretrain our RecSysLLM for eight epochs with AdamW optimization (Loshchilov and Hutter 2017) on four NVIDIA RTX A100 GPUs. In order to achieve efficient use of memory and distributed training, we use the DeepSpeed (Rasley et al. 2020) module. The batch size is set to 32 per GPU. We set the peak learning rate as 1 × 10 − 5 and use a warmup strategy to adjust the learning rate. In addition, we set the maximum length of input tokens to 1024.

# Performance.

We pretrain our RecSysLLM on a diverse set of training examples utilizing different prompt templates across all five

Table 2: Performance on rating prediction. The shadow refers to the test on unseen prompts in a zero-shot manner.

<div style="text-align: center;">Table 2: Performance on rating prediction. The shadow refers to the test on unseen prompts in a zero-shot manner.
</div>
Methods
Sports
Beauty
Toys
RMSE
MAE
RMSE
MAE
RMSE
MAE
MF
1.0234
0.7935
1.1973
0.9461
1.0123
0.7984
MLP
1.1277
0.7626
1.3078
0.9597
1.1215
0.8097
P5
1.0357
0.6813
1.2843
0.8534
1.0544
0.7177
RecSysLLM
1.0410
0.7012
1.2721
0.8431
1.0246
0.7012
P5
1.0292
0.6864
1.2870
0.8531
1.0245
0.6931
RecSysLLM
1.0278
0.6631
1.2671
0.8235
1.0112
0.6014
task families. This is to thoroughly verify its multitask learning capabilities. The results in Tables 2-7 demonstrate that for tasks with seen prompt templates, our model reaches the same conclusions as the P5 model and achieves comparable or superior performance. However, we were pleasantly surprised to discover that for unseen prompt templates in a zero-shot manner, our model significantly surpasses P5. (1) From Table 2, for rating prediction, our RecSysLLM gets similar performance on prompt in the train data set, but it has better RMSE and MAE on all three datasets compared with P5 on zero-shot setting. It reflects that our RecSysLLM inherits the semantic understanding capacity of LLM on unseen prompts, which meets our expectations for the LLM. (2) In Table 4, for the sequential recommendation, our RecSysLLM surpasses P5 on Beauty and Toys. It gets better performance than P5 on unseen prompts in a zero-shot manner. The results show that our RecSysLLM gains inter- and intraentity knowledge and make more reasonable predictions. (3) As shown in Table 5, our RecSysLLM demonstrates superior performance on the task of explanation generation, both with and without feature-based hints. The large improvements in natural language processing abilities of LLMs underlie this strong performance. Moreover, the considerable increase in scores when hints are provided highlights the critical role prompt engineering plays in eliciting the full capabilities of large language models. Through prompt design and the generative power of LLMs, our system achieves state-of-the-art results on this challenging task. (4) The review summarization results further demonstrate the superiority of our RecSysLLM, as shown in Table 6. Despite having fewer parameters than T0 (7 billion vs 11 billion), our model attains higher performance across all evaluation metrics. These gains over strong baselines like T0 underscore the efficiency and effectiveness of our approach. The capability to produce high-quality summaries with fewer parameters highlights the strength of our method, delivering strong performance without the need for extremely large models. (5) For the task of direct recommendation, we make an evaluation on open question prompts to test the ability of generative recommendation. The results are illustrated in Table 7. Our RecSysLLM outperforms P5 on most evaluation metrics for this task. The simpleX model is a strong collaborative filtering baseline, but RecSysLLM achieves better top-1 item ranking compared to simpleX. To further analyze the performance gap between the P5 model and our proposed method, we conducted an in-depth examination of the training data. Table 3 illustrates that in the P5 model, the items are simply represented by numeric

Table 3: The training sequences in Amazon Toys dataset for P5 and our RecSysLLM model.

Methods
Sports
Beauty
Toys
HR@5
NDCG@5
HR@10
NDCG@10
HR@5
NDCG@5
HR@10
NDCG@10
HR@5
NDCG@5
HR@10
NDCG@10
Caser
0.0116
0.0072
0.0194
0.0097
0.0205
0.0131
0.0347
0.0176
0.0166
0.0107
0.0270
0.0141
HGN
0.0189
0.0120
0.0313
0.0159
0.0325
0.0206
0.0512
0.0266
0.0321
0.0221
0.0497
0.0277
GRU4Rec
0.0129
0.0086
0.0204
0.0110
0.0164
0.0099
0.0283
0.0137
0.0097
0.0059
0.0176
0.0084
BERT4Rec
0.0115
0.0075
0.0191
0.0099
0.0203
0.0124
0.0347
0.0170
0.0116
0.0071
0.0203
0.0099
FDSA
0.0182
0.0122
0.0288
0.0156
0.0267
0.0163
0.0407
0.0208
0.0228
0.0140
0.0381
0.0189
SASRec
0.0233
0.0154
0.0350
0.0192
0.0387
0.0249
0.0605
0.0318
0.0463
0.0306
0.0675
0.0374
S3-Rec
0.0251
0.0161
0.0385
0.0204
0.0387
0.0244
0.0647
0.0327
0.0443
0.0294
0.0700
0.0376
P5
0.0364
0.0296
0.0431
0.0318
0.0508
0.0379
0.0664
0.0429
0.0608
0.0507
0.0688
0.0534
RecSysLLM
0.0360
0.0291
0.0417
0.0302
0.0508
0.0381
0.0667
0.0446
0.0676
0.0583
0.0712
0.0596
P5
0.0387
0.0312
0.0460
0.0336
0.0493
0.0367
0.0645
0.0416
0.0587
0.0486
0.0675
0.0536
RecSysLLM
0.0392
0.0330
0.0512
0.0375
0.0501
0.0361
0.0650
0.0407
0.0630
0.0523
0.0691
0.0540
Methods
Sports
Beauty
Toys
BLUE4
ROUGE1
ROUGE2
ROUGEL
BLUE4
ROUGE1
ROUGE2
ROUGEL
BLUE4
ROUGE1
ROUGE2
ROUGEL
w/o hints
Attn2Seq
0.5305
12.2800
1.2107
9.1312
0.7889
12.6590
1.6820
9.7481
1.6238
13.2245
2.9942
10.7398
NRT
0.4793
11.0723
1.1304
7.6674
0.8295
12.7815
1.8543
9.9477
1.9084
13.5231
3.6708
11.1867
PETER
0.7112
12.8944
1.3283
9.8635
1.1541
14.8497
2.1413
11.4143
1.9861
14.2716
3.6718
11.7010
P5
1.0407
14.1589
2.1220
10.6096
0.9742
16.4530
1.8858
11.8765
2.3185
15.3474
3.7209
12.1312
RecSysLLM
1.2673
16.7132
2.8980
13.0104
1.5230
19.0032
3.0422
14.7471
2.9923
16.7823
4.8372
15.0231
w/ hints
PETER+
2.4627
24.1181
5.1937
18.4105
3.2606
25.5541
5.9668
19.7168
4.7919
28.3083
9.4520
22.7017
P5
1.4689
23.5476
5.3926
17.5852
1.8765
25.1183
6.0764
19.4488
3.8933
27.9916
9.5896
22.2178
RecSysLLM
3.7232
30.1129
5.0232
20.0020
4.8232
26.9832
6.2382
21.4842
5.9323
29.3232
9.4234
23.9843
P5
1.4303
23.3810
5.3239
17.4913
1.9031
25.1763
6.1980
19.5188
3.5861
28.1369
9.7562
22.3056
RecSysLLM
3.9842
30.2913
5.8923
20.3821
5.0021
27.3854
6.7281
22.7439
6.2912
30.2948
10.0329
24.9932
<div style="text-align: center;">Table 6: Performance on review summarization (%). The shadow refers to the test on unseen prompts in a zero-sho
</div>
<div style="text-align: center;">on review summarization (%). The shadow refers to the test on unseen prompts in a zero-shot manner.
</div>
Methods
Sports
Beauty
Toys
BLUE2
ROUGE1
ROUGE2
ROUGEL
BLUE2
ROUGE1
ROUGE2
ROUGEL
BLUE2
ROUGE1
ROUGE2
ROUGEL
T0
2.1581
2.2695
0.5694
1.6221
1.2871
1.2750
0.3904
0.9592
2.2296
2.4671
0.6482
1.8424
GPT-2
0.7779
4.4534
1.0033
1.9236
0.5879
3.3844
0.6756
1.3956
0.6221
3.7149
0.6629
1.4813
P5
2.6910
12.0314
3.2921
10.7274
1.9325
8.2909
1.4321
7.4000
1.7833
8.7222
1.3210
7.6134
RecSysLLM
4.2823
14.8343
4.3984
12.4833
3.3821
9.8103
2.8543
10.4003
4.0320
12.2932
3.2943
10.4092
Methods
Sports
Beauty
Toys
HR@1
HR@5
NDCG@5
HR@10
NDCG@10
HR@1
HR@5
NDCG@5
HR@10
NDCG@10
HR@1
HR@5
NDCG@5
HR@10
NDCG@10
BPR-MF
0.0314
0.1404
0.0848
0.2563
0.1220
0.0311
0.1426
0.0857
0.2573
0.1224
0.0233
0.1066
0.0641
0.2003
0.0940
BPR-MLP
0.0351
0.1520
0.0927
0.2671
0.1296
0.0317
0.1392
0.0848
0.2542
0.1215
0.0252
0.1142
0.0688
0.2077
0.0988
SimpleX
0.0331
0.2362
0.1505
0.3290
0.1800
0.0325
0.2247
0.1441
0.3090
0.1711
0.0268
0.1958
0.1244
0.2662
0.1469
P5
0.0641
0.1794
0.1229
0.2598
0.1488
0.0588
0.1573
0.1089
0.2325
0.1330
0.0386
0.1122
0.0756
0.1807
0.0975
RecSysLLM
0.0654
0.2008
0.1438
0.2984
0.1692
0.0618
0.1612
0.1110
0.2209
0.1302
0.0370
0.1301
0.0808
0.1902
0.0998
P5
0.0726
0.1955
0.1355
0.2802
0.1627
0.0608
0.1564
0.1096
0.2300
0.1332
0.0389
0.1147
0.0767
0.1863
0.0997
RecSysLLM
0.0892
0.2029
0.1502
0.3001
0.1703
0.6072
0.1502
0.1097
0.2317
0.1302
0.0327
0.1423
0.0825
0.1926
0.1028
IDs based on their order of occurrence in the dataset. This type of simplistic representation cannot capture semantic information about the items. In contrast, our RecSysLLM model represents all items as text strings. The textual representation enables our large language model to understand and capture nuanced interrelationships between items much more effectively. We believe this is the primary reason why

our model outperformed P5 across most cases. The textual representation in our model empowers it to ingest semantic details and identify meaningful connections that cannot be derived from IDs alone.

# Applications in real-world dataset

# Dataset

The data used in this work was collected from Alipay, a mobile payment platform in China. We extracted user behavior logs, including bills, search queries, and page visits for several recommendation tasks. Each user sequence consists of the user’s 500 most recent interactions, spanning over one year of history for some users. The user sequences are used to model evolving user interests and capture both long- and short-term preferences. The training set contains 200, 000 sequences, and the test set contains 10, 000 sequences. The large-scale real-world dataset enables the modeling of complex user behavior and preferences for various recommendation tasks. The hierarchical categories and sequential interactions provide rich signals for understanding user interests.

# Implementation Details

Our RecSysLLM model for Chinese language tasks leverages the powerful ChatGLM-6B (Du et al. 2021) model as a foundation. ChatGLM-6B is an open-source bilingual language model with 6.2 billion parameters, trained on a trillion-token corpus comprised primarily of Chinese text with some English. The model architecture is based on the General Language Model (GLM) framework. Similarly, our approach builds on this pre-trained ChatGLM-6B foundation by utilizing LoRA to adapt the model to our specific recommender system tasks. We set the rank of Lora to 8, which is a proper coefficient chosen by the ablation study.

# Sequential Recommendation.

Task Description. In this section, we conduct two sequential recommendation tasks to evaluate the performance of our model, i.e., next-item prediction and candidate recommendation. For next-item prediction, the model directly predicts the next item a user will interact with based on their historical interactions and profiles. For candidate recommendation, given a user’s interaction history, profiles, and a list of candidate items where only one is positive, the model chooses the correct next item. We have benchmarked our model on the Amazon Sports, Beauty, and Toys datasets and demonstrated superior recommendation capabilities compared to other baseline recommender systems. Here, we compare our RecSysLLM to the powerful generative models ChatGPT and the recently announced GPT4. We also compare our method against a basic fine-tuning approach of ChatGLM on our recommendation tasks. This allows us to analyze the improvements gained by our specialized techniques that are tailored for the recommendation systems based on LLM. By evaluating against a simple finetuning baseline, we can quantify the benefits of our proposed approach and demonstrate that our architectural choices and training methodology confer meaningful advantages on recommendation performance compared to just fine-tuning a large language model out-of-the-box.
Next Item Prediction. The results in Table 8 demonstrate that for next-item prediction, our RecSysLLM achieves performance on par with ChatGPT, with both significantly outperforming the naive ChatGLM fine-tuning and GPT-4. This

is a surprising result, as we expected the larger GPT-4 model to achieve superior performance compared to ChatGPT on this recommendation task due to its greater parameter size and pretraining scale. However, GPT-4 did not exhibit particularly strong results and was not decisively superior to ChatGPT. There are several potential explanations for why GPT-4 underperformed expectations on the next item prediction. First, the dataset and evaluation methodology used for this task may not have fully exercised GPT-4’s strengths in areas like few-shot learning and knowledge recall. Second, GPT-4’s more powerful generative capabilities may have caused it to diverge too far from the tight distributions of the recommendation data. There could be a mismatch between GPT-4’s broad natural language generation skills and the specialized prediction required by the recommender system task. In summary, our specialized RecSysLLM demonstrates that simply utilizing a larger pre-trained language model is not the only path to improved recommendation performance. The model architecture and pretraining objectives also play a vital role. By designing a model specifically for the recommendation, focusing the pretraining on recommendation data, and tightly bounding the final fine-tuning, our RecSysLLM is able to match or exceed the performance of even much larger general language models like GPT-4 for nextitem prediction. These results highlight the importance of specialized model design in addition to scale for advancing recommendation systems.
Candidate Recommendation. For candidate recommendation in Table 9, our RecSysLLM consistently outperforms both ChatGPT and the naive ChatGLM fine-tuning across metrics. This demonstrates the effectiveness of our specialized approach for this task. In contrast to the next item results, this time, GPT-4 achieves the overall best performance on candidate recommendation. In candidate recommendation, given a user’s interaction history, profile, and a list of candidate items where only one is the ground truth next interaction, the model must choose the correct item from the candidates. With a constrained set of options provided, GPT4 is able to give full play to its powerful reasoning and deduction capabilities. The limited choice set prevents GPT4’s generative tendencies from leading it astray. As a result, GPT-4 is able to leverage its scale and pretraining to achieve the best overall performance on candidate recommendation. In summary, by providing GPT-4 a focused set of candidates, we can elicit its strengths in logical reasoning while avoiding over-generation. This allows GPT-4 to achieve state-of-theart results on candidate recommendation, showcasing the benefits of its scale and pretraining. Our specialized RecSysLLM still exceeds the general language models on this task, demonstrating the value of recommendation-specific modeling. But these results highlight how large generative LMs like GPT-4 can excel given the right setup.

is a surprising result, as we expected the larger GPT-4 model to achieve superior performance compared to ChatGPT on this recommendation task due to its greater parameter size and pretraining scale. However, GPT-4 did not exhibit particularly strong results and was not decisively superior to ChatGPT. There are several potential explanations for why GPT-4 underperformed expectations on the next item prediction. First, the dataset and evaluation methodology used for this task may not have fully exercised GPT-4’s strengths in areas like few-shot learning and knowledge recall. Second, GPT-4’s more powerful generative capabilities may have caused it to diverge too far from the tight distributions of the recommendation data. There could be a mismatch between GPT-4’s broad natural language generation skills and the specialized prediction required by the recommender system task. In summary, our specialized RecSysLLM demonstrates that simply utilizing a larger pre-trained language model is not the only path to improved recommendation performance. The model architecture and pretraining objectives also play a vital role. By designing a model specifically for the recommendation, focusing the pretraining on recommendation data, and tightly bounding the final fine-tuning, our RecSysLLM is able to match or exceed the performance of even much larger general language models like GPT-4 for nextitem prediction. These results highlight the importance of specialized model design in addition to scale for advancing recommendation systems.

# Conclusion

The focus of this paper is to design a novel paradigm of pretraining recommendation models based on large language models. We introduce a novel mask mechanism, span order, and positional encoding to inject inter- and intra-entity

Table 8: Performance on next item recommendation.

Methods
HR@5
NDCG@5
HR@10
NDCG@10
ChatGPT
0.4326
0.3208
0.5110
0.3465
GPT-4
0.3846
0.2890
0.4674
0.3159
ChatGLM+SFT
0.2654
0.2091
0.3729
0.2513
RecSysLLM
0.3805
0.3072
0.4756
0.4091
Table 9: Performance on candidate recommendation task.

Methods
HR@1
HR@5
NDCG@5
HR@10
NDCG@10
ChatGPT
0.3786
0.5550
0.4715
0.6424
0.5001
GPT-4
0.7079
0.8154
0.7671
0.8560
0.7804
ChatGLM+SFT
0.2984
0.7012
0.6826
0.7621
0.7038
RecSysLLM
0.4965
0.7435
0.7032
0.7728
0.7237
knowledge into the LLM. Although our method follows the architecture of generative language models (GLM) to some extent, the core ideas of special designs for entities in recommendation tasks can be extended to other large language models. The experiments conducted on public and industrial datasets demonstrate the effectiveness and potential of our proposed model on recommendation systems and related applications. The results show improvements over strong baselines, indicating that encoding entity relationships during pretraining can meaningfully improve downstream performance. While we validate our approach on a select set of datasets, further experiments on a wider range of tasks would better reveal the strengths and limitations of the method. In particular, evaluating the approach across a more diverse set of domains could shed light on how robust the learned representations are. Additionally, from the perspective of causal inference (Yao et al. 2021; Chu et al. 2023), there are likely further improvements to be made in terms of how semantic connections between entities are captured and injected into the model.

# References

Andreas, J. 2022. Language models as agent models. arXiv preprint arXiv:2212.01681. Bao, K.; Zhang, J.; Zhang, Y.; Wang, W.; Feng, F.; and He, X. 2023. Tallrec: An effective and efficient tuning framework to align large language model with recommendation. arXiv preprint arXiv:2305.00447. Bodon, F.; and R´onyai, L. 2003. Trie: an alternative data structure for data mining algorithms. Mathematical and Computer Modelling, 38(7-9): 739–751. Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; et al. 2020. Language models are few-shot learners.  Advances in neural information processing systems, 33: 1877– 1901. Chen, Z. 2023. PALR: Personalization Aware LLMs for Recommendation. arXiv preprint arXiv:2305.07622. Cheng, H.-T.; Koc, L.; Harmsen, J.; Shaked, T.; Chandra, T.; Aradhye, H.; Anderson, G.; Corrado, G.; Chai, W.; Ispir, M.; et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems, 7–10.

Cho, K.; van Merrienboer, B.; Gulcehre, C.; Bahdanau, D.; Bougares, F.; Schwenk, H.; and Bengio, Y. 2014. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1724–1734. Chu, Z.; Ding, H.; Zeng, G.; Huang, Y.; Yan, T.; Kang, Y.; and Li, S. 2022. Hierarchical capsule prediction network for marketing campaigns effect. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 3043–3051. Chu, Z.; Huang, J.; Li, R.; Chu, W.; and Li, S. 2023. Causal effect estimation: Recent advances, challenges, and opportunities. arXiv preprint arXiv:2302.00848. Dai, S.; Shao, N.; Zhao, H.; Yu, W.; Si, Z.; Xu, C.; Sun, Z.; Zhang, X.; and Xu, J. 2023. Uncovering ChatGPT’s Capabilities in Recommender Systems. arXiv preprint arXiv:2305.02182. Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. Dong, L.; Huang, S.; Wei, F.; Lapata, M.; Zhou, M.; and Xu, K. 2017. Learning to generate product reviews from attributes. In EACL. Du, Z.; Qian, Y.; Liu, X.; Ding, M.; Qiu, J.; Yang, Z.; and Tang, J. 2021. Glm: General language model pretraining with autoregressive blank infilling. arXiv preprint arXiv:2103.10360. Friedman, L.; Ahuja, S.; Allen, D.; Tan, T.; Sidahmed, H.; Long, C.; Xie, J.; Schubiner, G.; Patel, A.; Lara, H.; et al. 2023. Leveraging Large Language Models in Conversational Recommender Systems. arXiv preprint arXiv:2305.07961. Gao, Y.; Sheng, T.; Xiang, Y.; Xiong, Y.; Wang, H.; and Zhang, J. 2023. Chat-rec: Towards interactive and explainable llms-augmented recommender system. arXiv preprint arXiv:2303.14524. Geng, S.; Liu, S.; Fu, Z.; Ge, Y.; and Zhang, Y. 2022. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In  Proceedings of the 16th ACM Conference on Recommender Systems, 299–315. Gu, J.; Zhao, H.; Xu, H.; Nie, L.; Mei, H.; and Yin, W. 2023. Robustness of Learning from Task Instructions. In Findings of ACL. Hidasi, B.; Karatzoglou, A.; Baltrunas, L.; and Tikk, D. 2015. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939. Hidasi, B.; Karatzoglou, A.; Baltrunas, L.; and Tikk, D. 2016. Session-based Recommendations with Recurrent Neural Networks. In ICLR. Hou, Y.; Zhang, J.; Lin, Z.; Lu, H.; Xie, R.; McAuley, J.; and Zhao, W. X. 2023. Large language models are zero-shot rankers for recommender systems. arXiv preprint arXiv:2305.08845.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685. Hui, B.; Zhang, L.; Zhou, X.; Wen, X.; and Nian, Y. 2022. Personalized recommendation system based on knowledge embedding and historical behavior. Applied Intelligence, 1– 13. Jiang, C.; Xue, S.; Zhang, J.; Liu, L.; Zhu, Z.; and Hao, H. 2022. Learning Large-scale Universal User Representation with Sparse Mixture of Experts. Kang, W.-C.; and McAuley, J. 2018. Self-attentive sequential recommendation. In  2018 IEEE international conference on data mining (ICDM), 197–206. IEEE. Kang, W.-C.; Ni, J.; Mehta, N.; Sathiamoorthy, M.; Hong, L.; Chi, E.; and Cheng, D. Z. 2023. Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction. arXiv preprint arXiv:2305.06474. Koren, Y.; Bell, R.; and Volinsky, C. 2009. Matrix factorization techniques for recommender systems. Computer, 42(8): 30–37. Krizhevsky, A.; Sutskever, I.; and Hinton, G. E. 2012. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25. Li, L.; Zhang, Y.; and Chen, L. 2021. Personalized Transformer for Explainable Recommendation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 4947–4957. Li, P.; Wang, Z.; Ren, Z.; Bing, L.; and Lam, W. 2017. Neural rating regression with abstractive tips generation for recommendation. In Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval, 345–354. Li, S.; and Zhao, H. 2021. A survey on representation learning for user modeling. In  Proceedings of the TwentyNinth International Conference on International Joint Conferences on Artificial Intelligence, 4997–5003. Lin, J.; Dai, X.; Xi, Y.; Liu, W.; Chen, B.; Li, X.; Zhu, C.; Guo, H.; Yu, Y.; Tang, R.; et al. 2023. How Can Recommender Systems Benefit from Large Language Models: A Survey. arXiv preprint arXiv:2306.05817. Liu, J.; Liu, C.; Lv, R.; Zhou, K.; and Zhang, Y. 2023a. Is chatgpt a good recommender? a preliminary study. arXiv preprint arXiv:2304.10149. Liu, Q.; Chen, N.; Sakai, T.; and Wu, X.-M. 2023b. A First Look at LLM-Powered Generative News Recommendation. arXiv preprint arXiv:2305.06566. Loshchilov, I.; and Hutter, F. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101. Ma, C.; Kang, P.; and Liu, X. 2019. Hierarchical gating networks for sequential recommendation. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, 825–833.

Mao, K.; Zhu, J.; Wang, J.; Dai, Q.; Dong, Z.; Xiao, X.; and He, X. 2021. SimpleX: A Simple and Strong Baseline for Collaborative Filtering. In  Proceedings of the 30th ACM International Conference on Information & Knowledge Management, 1243–1252. Muhamed, A.; Keivanloo, I.; Perera, S.; Mracek, J.; Xu, Y.; Cui, Q.; Rajagopalan, S.; Zeng, B.; and Chilimbi, T. 2021. CTR-BERT: Cost-effective knowledge distillation for billion-parameter teacher models. In  NeurIPS Efficient Natural Language and Speech Processing Workshop. Ouyang, L.; Wu, J.; Jiang, X.; Almeida, D.; Wainwright, C.; Mishkin, P.; Zhang, C.; Agarwal, S.; Slama, K.; Ray, A.; et al. 2022. Training language models to follow instructions with human feedback.  Advances in Neural Information Processing Systems, 35: 27730–27744. Qiu, Z.; Wu, X.; Gao, J.; and Fan, W. 2021. U-BERT: Pretraining user representations for improved recommendation. In  Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, 4320–4327. Radford, A.; Narasimhan, K.; Salimans, T.; Sutskever, I.; et al. ???? Improving language understanding by generative pre-training. Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; Sutskever, I.; et al. 2019. Language models are unsupervised multitask learners. OpenAI blog. Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; and Liu, P. J. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1): 5485–5551. Rasley, J.; Rajbhandari, S.; Ruwase, O.; and He, Y. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In  Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 3505–3506. Rendle, S.; Freudenthaler, C.; Gantner, Z.; and SchmidtThieme, L. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, UAI ’09, 452–461. Arlington, Virginia, USA: AUAI Press. ISBN 9780974903958. Sanh, V.; Webson, A.; Raffel, C.; Bach, S.; Sutawika, L.; Alyafeai, Z.; Chaffin, A.; Stiegler, A.; Raja, A.; Dey, M.; Bari, M. S.; Xu, C.; Thakker, U.; Sharma, S. S.; Szczechla, E.; Kim, T.; Chhablani, G.; Nayak, N.; Datta, D.; Chang, J.; Jiang, M. T.-J.; Wang, H.; Manica, M.; Shen, S.; Yong, Z. X.; Pandey, H.; Bawden, R.; Wang, T.; Neeraj, T.; Rozen, J.; Sharma, A.; Santilli, A.; Fevry, T.; Fries, J. A.; Teehan, R.; Scao, T. L.; Biderman, S.; Gao, L.; Wolf, T.; and Rush, A. M. 2022. Multitask Prompted Training Enables ZeroShot Task Generalization. In International Conference on Learning Representations. Schuster, M.; and Paliwal, K. K. 1997. Bidirectional recurrent neural networks.  IEEE transactions on Signal Processing, 45(11): 2673–2681. Sheu, H.-S.; Chu, Z.; Qi, D.; and Li, S. 2021. Knowledgeguided article embedding refinement for session-based news

recommendation. IEEE Transactions on Neural Networks and Learning Systems, 33(12): 7921–7927. Shi, X.; Xue, S.; Wang, K.; Zhou, F.; Zhang, J. Y.; Zhou, J.; Tan, C.; and Mei, H. 2023. Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning. arXiv preprint arXiv:2305.16646. Sun, F.; Liu, J.; Wu, J.; Pei, C.; Lin, X.; Ou, W.; and Jiang, P. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In  Proceedings of the 28th ACM international conference on information and knowledge management, 1441–1450. Tang, J.; and Wang, K. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the eleventh ACM international conference on web search and data mining, 565–573. Tsai, C. F.; Zhou, X.; Liu, S. S.; Li, J.; Yu, M.; and Mei, H. 2023. Can Large Language Models Play Text Games Well? Current State-of-the-Art and Open Questions. arXiv preprint. Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need.  Advances in neural information processing systems, 30. Wang, W.; Lin, X.; Feng, F.; He, X.; and Chua, T.-S. 2023. Generative recommendation: Towards next-generation recommender paradigm. arXiv preprint arXiv:2304.03516. Wang, X.; Zhou, K.; Wen, J.-R.; and Zhao, W. X. 2022. Towards unified conversational recommender systems via knowledge-enhanced prompt learning. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 1929–1937. Wu, C.; Wu, F.; Qi, T.; and Huang, Y. 2021. Empowering news recommendation with pre-trained language models. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 1652–1656. Wu, L.; Zheng, Z.; Qiu, Z.; Wang, H.; Gu, H.; Shen, T.; Qin, C.; Zhu, C.; Zhu, H.; Liu, Q.; et al. 2023. A Survey on Large Language Models for Recommendation. arXiv preprint arXiv:2305.19860. Xiao, S.; Liu, Z.; Shao, Y.; Di, T.; Middha, B.; Wu, F.; and Xie, X. 2022. Training large-scale news recommenders with pretrained language models in the loop. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 4215–4225. Xie, S.; Qiu, J.; Pasad, A.; Du, L.; Qu, Q.; and Mei, H. 2022. Hidden State Variability of Pretrained Language Models Can Guide Computation Reduction for Transfer Learning. In Findings of EMNLP. Xue, S.; Shi, X.; Chu, Z.; Wang, Y.; Zhou, F.; Hao, H.; Jiang, C.; Pan, C.; Xu, Y.; Zhang, J. Y.; Wen, Q.; Zhou, J.; and Mei, H. 2023. EasyTPP: Towards Open Benchmarking the Temporal Point Processes. Xue, S.; Shi, X.; Hao, H.; Ma, L.; Zhang, J.; Wang, S.; and Wang, S. 2021. A Graph Regularized Point Process Model For Event Propagation Sequence. In 2021 International Joint Conference on Neural Networks (IJCNN), 1–7.

recommendation. IEEE Transactions on Neural Networks and Learning Systems, 33(12): 7921–7927. Shi, X.; Xue, S.; Wang, K.; Zhou, F.; Zhang, J. Y.; Zhou, J.; Tan, C.; and Mei, H. 2023. Language Models Can Improve Event Prediction by Few-Shot Abductive Reasoning. arXiv preprint arXiv:2305.16646. Sun, F.; Liu, J.; Wu, J.; Pei, C.; Lin, X.; Ou, W.; and Jiang, P. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In  Proceedings of the 28th ACM international conference on information and knowledge management, 1441–1450. Tang, J.; and Wang, K. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the eleventh ACM international conference on web search and data mining, 565–573. Tsai, C. F.; Zhou, X.; Liu, S. S.; Li, J.; Yu, M.; and Mei, H. 2023. Can Large Language Models Play Text Games Well? Current State-of-the-Art and Open Questions. arXiv preprint. Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need.  Advances in neural information processing systems, 30. Wang, W.; Lin, X.; Feng, F.; He, X.; and Chua, T.-S. 2023. Generative recommendation: Towards next-generation recommender paradigm. arXiv preprint arXiv:2304.03516. Wang, X.; Zhou, K.; Wen, J.-R.; and Zhao, W. X. 2022. Towards unified conversational recommender systems via knowledge-enhanced prompt learning. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 1929–1937. Wu, C.; Wu, F.; Qi, T.; and Huang, Y. 2021. Empowering news recommendation with pre-trained language models. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 1652–1656. Wu, L.; Zheng, Z.; Qiu, Z.; Wang, H.; Gu, H.; Shen, T.; Qin, C.; Zhu, C.; Zhu, H.; Liu, Q.; et al. 2023. A Survey on Large Language Models for Recommendation. arXiv preprint arXiv:2305.19860. Xiao, S.; Liu, Z.; Shao, Y.; Di, T.; Middha, B.; Wu, F.; and Xie, X. 2022. Training large-scale news recommenders with pretrained language models in the loop. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 4215–4225. Xie, S.; Qiu, J.; Pasad, A.; Du, L.; Qu, Q.; and Mei, H. 2022. Hidden State Variability of Pretrained Language Models Can Guide Computation Reduction for Transfer Learning. In Findings of EMNLP. Xue, S.; Shi, X.; Chu, Z.; Wang, Y.; Zhou, F.; Hao, H.; Jiang, C.; Pan, C.; Xu, Y.; Zhang, J. Y.; Wen, Q.; Zhou, J.; and Mei, H. 2023. EasyTPP: Towards Open Benchmarking the Temporal Point Processes. Xue, S.; Shi, X.; Hao, H.; Ma, L.; Zhang, J.; Wang, S.; and Wang, S. 2021. A Graph Regularized Point Process Model For Event Propagation Sequence. In 2021 International Joint Conference on Neural Networks (IJCNN), 1–7.

Xue, S.; Shi, X.; Zhang, Y. J.; and Mei, H. 2022. HYPRO: A Hybridly Normalized Probabilistic Model for Long-Horizon Prediction of Event Sequences. In  Advances in Neural Information Processing Systems. Yao, L.; Chu, Z.; Li, S.; Li, Y.; Gao, J.; and Zhang, A. 2021. A survey on causal inference.  ACM Transactions on Knowledge Discovery from Data (TKDD), 15(5): 1–46. Yao, S.; Tan, J.; Chen, X.; Zhang, J.; Zeng, X.; and Yang, K. 2022. ReprBERT: Distilling BERT to an Efficient Representation-Based Relevance Model for E-Commerce. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 4363–4371. Yoneda, T.; Fang, J.; Li, P.; Zhang, H.; Jiang, T.; Lin, S.; Picker, B.; Yunis, D.; Mei, H.; and Walter, M. R. 2023. Statler: State-Maintaining Language Models for Embodied Reasoning. arXiv preprint. Yu, Z.; Lian, J.; Mahmoody, A.; Liu, G.; and Xie, X. 2019. Adaptive User Modeling with Long and Short-Term Preferences for Personalized Recommendation. In IJCAI, 4213– 4219. Zhang, J.; Xie, R.; Hou, Y.; Zhao, W. X.; Lin, L.; and Wen, J.-R. 2023. Recommendation as instruction following: A large language model empowered recommendation approach. arXiv preprint arXiv:2305.07001. Zhang, T.; Zhao, P.; Liu, Y.; Sheng, V. S.; Xu, J.; Wang, D.; Liu, G.; and Zhou, X. 2019. Feature-level Deeper SelfAttention Network for Sequential Recommendation. In  IJCAI, 4320–4326. Zhao, H.; Tan, H.; and Mei, H. 2022. Tiny-Attention Adapter: Contexts Are More Important Than the Number of Parameters. In EMNLP. Zhao, H.; Wang, K.; Yu, M.; and Mei, H. 2023. Explicit Planning Helps Language Models in Logical Reasoning. arXiv preprint. Zhou, K.; Wang, H.; Zhao, W. X.; Zhu, Y.; Wang, S.; Zhang, F.; Wang, Z.; and Wen, J.-R. 2020. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In  Proceedings of the 29th ACM International Conference on Information & Knowledge Management, 1893–1902.

Xue, S.; Shi, X.; Zhang, Y. J.; and Mei, H. 2022. HYPRO: A Hybridly Normalized Probabilistic Model for Long-Horizon Prediction of Event Sequences. In  Advances in Neural Information Processing Systems. Yao, L.; Chu, Z.; Li, S.; Li, Y.; Gao, J.; and Zhang, A. 2021. A survey on causal inference.  ACM Transactions on Knowledge Discovery from Data (TKDD), 15(5): 1–46. Yao, S.; Tan, J.; Chen, X.; Zhang, J.; Zeng, X.; and Yang, K. 2022. ReprBERT: Distilling BERT to an Efficient Representation-Based Relevance Model for E-Commerce. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 4363–4371. Yoneda, T.; Fang, J.; Li, P.; Zhang, H.; Jiang, T.; Lin, S.; Picker, B.; Yunis, D.; Mei, H.; and Walter, M. R. 2023. Statler: State-Maintaining Language Models for Embodied Reasoning. arXiv preprint. Yu, Z.; Lian, J.; Mahmoody, A.; Liu, G.; and Xie, X. 2019. Adaptive User Modeling with Long and Short-Term Preferences for Personalized Recommendation. In IJCAI, 4213– 4219. Zhang, J.; Xie, R.; Hou, Y.; Zhao, W. X.; Lin, L.; and Wen, J.-R. 2023. Recommendation as instruction following: A large language model empowered recommendation approach. arXiv preprint arXiv:2305.07001. Zhang, T.; Zhao, P.; Liu, Y.; Sheng, V. S.; Xu, J.; Wang, D.; Liu, G.; and Zhou, X. 2019. Feature-level Deeper SelfAttention Network for Sequential Recommendation. In  IJCAI, 4320–4326. Zhao, H.; Tan, H.; and Mei, H. 2022. Tiny-Attention Adapter: Contexts Are More Important Than the Number of Parameters. In EMNLP. Zhao, H.; Wang, K.; Yu, M.; and Mei, H. 2023. Explicit Planning Helps Language Models in Logical Reasoning. arXiv preprint. Zhou, K.; Wang, H.; Zhao, W. X.; Zhu, Y.; Wang, S.; Zhang, F.; Wang, Z.; and Wen, J.-R. 2020. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In  Proceedings of the 29th ACM International Conference on Information & Knowledge Management, 1893–1902.

# Recommendations and LLM

# Motivation

Compared with recommendation models based on large language models (LLMs), conventional recommendation models (Hidasi et al. 2015; Tang and Wang 2018; Kang and McAuley 2018; Sun et al. 2019; Geng et al. 2022) trained from scratch using architectures like Transformer (Vaswani et al. 2017), Bert (Devlin et al. 2018), RNN (Schuster and Paliwal 1997), CNN (Krizhevsky, Sutskever, and Hinton 2012) have several key limitations. First, they lack a deep understanding of context and semantics that comes from pretraining a large model on diverse corpora. As a result, they struggle to truly comprehend user preferences and behavioral sequences. Second, they have minimal ability to generate novel, high-quality recommendations since they are not optimized for free-form text generation. LLMs, in contrast, can produce human-like recommendations by leveraging their generative capabilities. Third, conventional models have difficulty effectively leveraging multiple data modalities like text, images, audio, etc. LLMs are adept at multimodal processing due to pretraining objectives that learn connections between modalities. Finally, LLMs can seamlessly adapt to new downstream recommendation tasks through simple fine-tuning, whereas conventional models require extensive retraining. For example, BERT4Rec (Sun et al. 2019) employs deep bidirectional self-attention to model user behavior sequences. They are trained solely based on the recommendation data without the general knowledge corpus, resulting in a limited understanding and reasoning of behavior sequence data and an inability to empower downstream tasks better. In summary, recommendation models based on pretrained LLMs are more contextual, creative, versatile, and adaptable compared to conventional models trained from scratch.

# Current Development

Although the application of LLMs like ChatGPT in recommendation has not been widely explored yet, some novel investigations have emerged recently that show their promising potential in this domain. There are mainly three categories. (1) LLM as a recommendation system.  First, Unlike traditional recommendation methods, they do not retrain a new model, relying only on the prompts of LLM (Liu et al. 2023a; Gao et al. 2023; Dai et al. 2023; Chen 2023) or slight fine-tuning (Zhang et al. 2023; Kang et al. 2023; Bao et al. 2023) to convert recommendation tasks into natural language tasks. They always design a set of prompts on recommendation scenarios, including rating prediction, sequential recommendation, direct recommendation, explanation generation, and review summarization. They explore the use of few-shot prompting to inject interaction information that contains user potential interest to help LLM better understand user needs and interests. (2) LLM as supplementary information via embeddings or tokens. This modeling paradigm (Wu et al. 2021; Qiu et al. 2021; Yao et al. 2022; Muhamed et al. 2021; Xiao et al. 2022) views the language model as a feature extractor,

which feeds the features of items and users into LLMs and outputs corresponding embeddings. A traditional RS model can utilize knowledge-aware embeddings for various recommendation tasks. This approach (Liu et al. 2023b; Wang et al. 2022, 2023) generates tokens based on the inputted items’ and users’ features. The generated tokens capture potential preferences through semantic mining, which can be integrated into the decision-making process of a recommendation system. (3) LLM as Agent. As an agent, the large model assists in scheduling the entire recommendation model for recommendations and is responsible for pipeline control. Specifically, these models (Andreas 2022; Bao et al. 2023; Hou et al. 2023; Lin et al. 2023; Gao et al. 2023; Friedman et al. 2023) help to adapt LLM to the recommendation domain, coordinate user data collection, feature engineering, feature encoder, scoring/ranking function.

# Challenges

Compared to superficially leveraging large language models, our purpose is built on the large language model, maximizing the preservation of knowledge and logical reasoning abilities from the original large language model to ensure the inference for the behavioral sequences and fluent generation of downstream sub-tasks, while also achieving the recommendation function by learning user profile features and user behavior sequences. The crucial aspect of harnessing the power of language models in enhancing recommendation quality is the utilization of their high-quality representations of textual features and their extensive coverage of external knowledge to establish correlations between items and users. (Wu et al. 2023). Therefore, we need to preserve the tokenization, parameters, and architecture of the large language model as much as possible. For example, Pretrain, Personalized Prompt, and Predict Paradigm (P5) (Geng et al. 2022) is established upon a basic encoder–decoder framework with Transformer blocks to build both the encoder and decoder. Although it is built on T5 (Raffel et al. 2020), it modified the structure of the model by adding additional positional encodings and whole-word embeddings, which will partially destroy the original knowledge in the language model. Notably, there is a difference in the format of the data. Large language models are trained on vast amounts of logically structured text, with consistent reasoning, logical thought processes, and proper grammar. In contrast, recommendation systems analyze digital user features, fixed item entities, and incoherent behavioral sequences. Additionally, The purpose of training data for large language models is to teach the model how to understand language and generate new text that is similar to the training data. Conversely, the purpose of user behavioral sequence data in recommendation systems is to dig a deeper understanding of user preferences, behavior sequences, and relationships between them so that to provide personalized recommendations. Therefore, building a recommendation system on top of a large language model that retains the LLM’s knowledge and logical reasoning abilities, while also achieving the recommendation function by learning user profile features and

# user behavior sequences poses significant challenges.

user behavior sequences poses significant challenges.

# Baselines in Benchmark Experiments

To showcase our competence in a wide range of recommendation-related tasks, we employ representative approaches for different tasks, including Rating Prediction, Direct Recommendation, Sequential Recommendation, Explanation Generation, and Review Summarization, that have been previously used by (Geng et al. 2022). The summary of baseline methods for five different task families is provided in Table 10. Rating Prediction.  This task involves incorporating useritem rating data as part of the training set, where item ratings are represented numerically. The model is asked questions with prompts, and it outputs corresponding rating values. The baselines for this task are MF (Koren, Bell, and Volinsky 2009) and MLP (Cheng et al. 2016). Direct Recommendation. For direct recommendation, we employ classic algorithms BPR-MF (Rendle et al. 2009), BPR-MLP (Cheng et al. 2016) and SimpleX (Mao et al. 2021) as baselines. They showcase the effectiveness of direct recommendation tasks when utilizing non-semantic information as features. This allows us to gain a more comprehensive understanding of the potential of recommendations given by LLM-based models. Sequential Recommendation.  The sequential recommendation task utilizes the user’s historical interaction sequences as input to predict the next item. We compare our proposed approaches with representative baselines in the field. Among that, some models aim to model the Markov Chain of user interactions by way of neural network architectures like convolutional neural networks, recurrent neural networks, and attention-based modules. Caser  (Tang and Wang 2018) employs convolutional neural networks to model user interests. HGN (Ma, Kang, and Liu 2019) adopts hierarchical gating networks to capture user behaviors from both long and short-term perspectives. GRU4Rec (Hidasi et al. 2016) utilizes recurrent neural network to model the user click history sequence. SASRec (Kang and McAuley 2018) and FDSA (Zhang et al. 2019) use self-attention modules to model feature transition patterns for sequential recommendation and the former combine RNN-based approaches to retain the sequential properties of items. BERT4Rec (Sun et al. 2019) adopts the BERT-style masked language modeling to learn the relations among items from the perspective of bidirectional representations in the recommendation. It started to use methods in neural language processing, but BERT did not have a strong semantic understanding capacity at that time. S 3-Rec  (Zhou et al. 2020) leverages selfsupervised objectives to enhance the discovery of correlations among different items and their attributes. Explanation Generation.  We evaluate the task of explanation generation by comparing the performance of several baseline models. Attn2Seq (Dong et al. 2017) and NRT (Li et al. 2017) utilizes the neural network to encode attributes of user and item into vectors and then invokes an attention mechanism or GRU (Cho et al. 2014) to generate reviews conditioned on the attribute vector. PETER (Li, Zhang, and Chen 2021) use Transformer architecture and designa a

e summary of baseline methods for five differlies.

<div style="text-align: center;">Table 10: The summary of baseline methods for five different task families.
</div>
Table 10: The summary of baseline methods for five different task families.

Rating Pre
MF (Koren, Bell, and Volinsky 2009)
MLP (Cheng et al. 2016)
Direct Rec
BPR-MF (Rendle et al. 2009)
BPR-MLP (Cheng et al. 2016)
SimpleX (Mao et al. 2021)
Sequential Rec
Caser (Tang and Wang 2018)
HGN (Ma, Kang, and Liu 2019)
GRU4Rec (Hidasi et al. 2016)
BERT4Rec (Sun et al. 2019)
FDSA (Zhang et al. 2019)
SASRec (Kang and McAuley 2018)
S3-Rec (Zhou et al. 2020)
BERT4Rec (Sun et al. 2019)
Explanation Gen
Attn2Seq (Dong et al. 2017)
NRT (Li et al. 2017)
PETER (Li, Zhang, and Chen 2021)
PETER+
Review Sum
T0 (Sanh et al. 2022)
GPT-2 (Radford et al. 2019)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/45aa/45aaf143-b984-452a-8416-cf85eda7807b.png" style="width: 50%;"></div>
modified attention mask. The variant PETER+ takes a hint feature word to augment the process of generating explanations. Review Related.  For review summarization, we adopt pretrained T0 (Sanh et al. 2022) and GPT-2 (Radford et al. 2019) as baselines. The latter model parameters were obtained from Hugging Face 1, which is a big platform to share models, datasets, and applications.

# Further Analysis in the real-world dataset

In addition to optimizing the recommendation performance, it is also important to understand why large language models like ChatGPT and GPT-4 are able to effectively conduct recommendation tasks in the first place. To explore this further, we provide several real-world case studies in Figure 4, where we systematically probe and dissect the reasoning process of these models when making recommendations, using carefully designed prompt-based queries. This analysis sheds light on the strengths and limitations of relying solely on the knowledge and reasoning capabilities embedded in large pre-trained language models for recommendation tasks, and points towards potential areas for improvement. Our experiments also analyze the impact of the rank r of Low-Rank Adaptation on model performance. We evaluate five different rank values 2, 4, 8, 16, and 32  - to determine the optimal balance between model capacity and pre

1 https://huggingface.co/

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2f3e/2f3e678a-df0f-4f0e-b136-72c929eae385.png" style="width: 50%;"></div>

<div style="text-align: center;">Figure 4: The case studies of ChatGPT and GPT-4 for next item recommendation in the real-world dataset.
</div>
dictive ability. As shown in Figure 3, we find that a rank of 8 provides sufficient learning capacity, with minimal improvements from increasing to 16. This indicates that capturing inter- and intra-entity relationships requires only a small number of additional trainable parameters beyond the base LLM, without the need for substantial model expansion. Rank 8 strikes the right balance, enabling Low-Rank Adaptation to boost performance through targeted parameterization rather than sheer scale. Overall, our results demonstrate that Low-Rank Adaptation offers an efficient approach to entity-aware language modeling.

