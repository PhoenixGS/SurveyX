# E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation
Xinhang Li Tsinghua Univerisity Beijing, China xh-li20@mails.tsinghua.edu.cn Chong Chen Huawei Cloud BU Beijing, China chenchong55@huawei.com
Chong Chen Huawei Cloud BU Beijing, China chenchong55@huawei.com Xiangyu Zhao City Univerisity of Hong K Hong Kong xianzhao@cityu.edu.hk
Chong Chen Huawei Cloud BU Beijing, China chenchong55@huawei.com
Xinhang Li Tsinghua Univerisity Beijing, China xh-li20@mails.tsinghua.edu.cn
Yong Zhang Tsinghua Univerisity Beijing, China hangyong05@tsinghua.edu.cn Chunxiao Xing Tsinghua Univerisity Beijing, China xingcx@tsinghua.edu.cn
 5 Dec 2023
# ABSTRACT
The recent advancements in Large Language Models (LLMs) have sparked interest in harnessing their potential within recommender systems. Since LLMs are designed for natural language tasks, existing recommendation approaches have predominantly transformed recommendation tasks into open-domain natural language generation tasks. However, this approach necessitates items to possess rich semantic information, often generates out-of-range results, and suffers from notably low efficiency and limited extensibility. Furthermore, practical ID-based recommendation strategies, reliant on a huge number of unique identities (IDs) to represent users and items, have gained prominence in real-world recommender systems due to their effectiveness and efficiency. Nevertheless, the incapacity of LLMs to model IDs presents a formidable challenge when seeking to leverage LLMs for personalized recommendations. In this paper, we introduce an Elegant Effective Efficient Extensible solution for large language models for Sequential Recommendation (E4SRec), which seamlessly integrates LLMs with traditional recommender systems that exclusively utilize IDs to represent items. Specifically, E4SRec takes ID sequences as inputs, ensuring that the generated outputs fall within the candidate lists. Furthermore, E4SRec possesses the capability to generate the entire ranking list in a single forward process, and demands only a minimal set of pluggable parameters, which are trained for each dataset while keeping the entire LLM frozen. We substantiate the effectiveness, efficiency, and extensibility of our proposed E4SRec through comprehensive experiments conducted on four widely-used real-world datasets. The implementation code is accessible at https://github.com/HestiaSky/E4SRec/.
arXiv:2312.02443v1
# CCS CONCEPTS
• Information systems →Recommender systems; Language models; Personalization.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. WWW ’24, May 13–17, 2024, Singapore, Singapore © 2023 Association for Computing Machinery. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/10.1145/1122445.1122456
Xiangyu Zhao City Univerisity of Hong Kong Hong Kong xianzhao@cityu.edu.hk
Chunxiao Xing Tsinghua Univerisity Beijing, China xingcx@tsinghua.edu.cn
KEYWORDS Large Language Model, Sequential Recommendation, Item ID and Indexing
Large Language Model, Sequential Recommendation, Item ID an Indexing
ACM Reference Format: Xinhang Li, Chong Chen, Xiangyu Zhao, Yong Zhang, and Chunxiao Xing. 2023. E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation. In Proceedings of the ACM Web Conference 2024 (WWW ’24), May 13–17, 2024, Singapore, Singapore. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/1122445.1122456
# 1 INTRODUCTION
Recommender systems, in existence for decades, are instrumental in mitigating information overload and enhancing user experience on the Web [1, 21, 35]. These systems discern user preferences to offer tailored recommendations on content or items [6, 24, 36]. Nowadays, the rise of Large Language Models (LLMs) [20] is revolutionizing our familiar landscapes [43], which excel in assimilating real-world knowledge from the Web and achieving proficient natural language generation. Recently, there has been a significant upsurge in research endeavors focused on leveraging LLMs for recommendation tasks and this trend is progressively becoming more inevitable [47]. While many non-tuning approaches [10, 14, 25], including prompting and in-context learning [3], strive to leverage the zero/few-shot learning ability, tuning approaches usually outperform them as they are fine-tuned for specific tasks using dedicated data. However, bridging the substantial gap between natural language generation tasks and recommendation tasks remains a formidable challenge. To tackle above challenge, existing approaches [2, 11, 17, 50] predominantly convert the recommendation task into a natural language generation task to align it with the inherent capabilities of LLMs. This involves the direct generation of item names or ratings based on appropriate prompts. However, such a solution has several limitations as depicted in Figure 1. First, these methods aim to harness the inherent knowledge of LLMs for recommendation through fine-tuning, essentially crafting an external knowledgeaugmented content-based recommendation [47]. Hence, they demand rich semantic information through the prompt. When the semantic information is insufficient or vague due to a huge number of homogeneous items, which is very common in recommendation, these methods are not able to yield satisfying performance.
LLM-based Sequential Recommendation
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4050/4050e422-96ff-45f0-a1b1-3713ee2e6de6.png" style="width: 50%;"></div>
E4SRec
<div style="text-align: center;">Figure 1: Illustration of LLM-based sequential recommendation. The upper part denotes the existing methods that fail to model the IDs and undesirably generate out-of-range results while the lower part denotes E4SRec that can effectively and efficiently handle the IDs.</div>
Different from content-based filtering methods, collaborative filtering methods utilize a huge number of unique identities (IDs) to represent users and items and have dominated the recommender system area for years. Such IDs merely function as indices for users or items without encapsulating any semantic information. Consequently, existing LLM-based recommendation methods struggle to manage the IDs to conduct ID-based recommendation and more critically, fail to leverage collaborative data vital for recommendations. Second, the problem definition of the existing approaches that is generate the results from the whole vocabulary will often lead to out-of-range results [17]. Such erratic generation not only diverges from the expectations of recommender systems but also negatively impacts user experience. Last but not least, the existing methods are only able to generate one recommendation result each time for the characteristics of LLMs. They mainly focus on the ranking task by attaching all the candidates in the prompt [50], and cannot tackle the matching task that requires matching scores of all the candidates. Nevertheless, the efficiency is still unacceptable for the recommender systems that require low latency and high concurrency. Hence, the trajectory of current LLM-based recommendation techniques fails to meet the needs of contemporary recommender systems and lacks practicality. To overcome the aforementioned limitations, we propose an Elegant Effective Efficient Extensible solution of large language models for Sequential Recommendation (E4SRec) by incorporating LLMs and traditional recommendation models with only IDs to represent items. Specifically, our proposed E4SRec accepts only ID sequences as inputs and ensures controllable generation with high efficiency by making predictions on and only on all the candidates in each forward process. Our proposed E4SRec solution encompasses four key phases: sequential recommendation model pretraining, LLM instruction tuning, E4SRec model training and E4SRec model deployment. For each given sequential recommendation dataset,
we first pretrain a traditional sequential recommendation model and then extract the item ID embeddings to prepare for the ID injection of LLM. An instruction tuning process of the LLM is also carried out to stimulate its capability to follow instructions and this tuned LLM is shared for all the task-specific models. Then, in the training stage of E4SRec, we wrap the sequences of item IDs into prompts by a linear projection of the item ID embeddings for ID injection. We freeze all the parameters of the LLM and only train an additional minimal set of parameters for adaption on the specific dataset. The recommendation results are made by computing the joint probability distribution between the output of LLM and all the candidate items via an item linear projection. Finally, once being trained, E4SRec can be deployed for practical application in a lightweight manner necessitating merely four pluggable components: the item ID embeddings, the input linear projection, the adapter and the item linear projection. The contribution of this paper can be summarized as follows: • We pioneer an innovative and effective strategy to address the unique challenges of integrating IDs in applying LLMs for recommendation tasks. • We address the prevailing issues of out-of-range outputs and generation efficiency, achieving controllable and efficient generative recommendation. • We propose an Elegant Effective Efficient Extensible solution of large language models for Sequential Recommendation (E4SRec), which is able to build an industrial-level recommender system from scratch. • Comprehensive experiments across four prominent real-world sequential recommendation datasets demonstrate the superiority and effectiveness of our proposed E4SRec model with in-depth analyses underscoring its efficiency and extensibility in realworld applications.
# 2 METHODOLOGY
In this section, we will introduce the complete solution of our proposed E4SRec in detail, including our pathway of ID injection, the tuning strategy of the backbone foundation models, the model structure of E4SRec and the deployment of E4SRec.
# 2.1 Overview
In order to deliver a clearer and more concise schema for better understanding, we provide the whole architecture of our proposed E4SRec solution in Figure 2. As illustrated in the lower right part, our E4SRec solution consists of four stages, which are sequential recommendation model pretraining, LLM instruction tuning, E4SRec model training and E4SRec model deployment. Specifically, the pretraining of sequential recommendation models and the instruction tuning of LLMs are the preliminaries and these two stages are essentially decoupled from the following stages of E4SRec by only providing sets of parameters as pluggable components. In the training stage of E4SRec, there are also a small number of parameters of several pluggable components being trained while the entire LLM is frozen and the personalization on the specific dataset is provided by an adapter. Once being trained, the E4SRec model could be easily deployed to conduct the sequential recommendation task on the given dataset by simply replacing the parameters of the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2235/22354114-e588-45ee-8134-9c0db6d13a80.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0ce0/0ce0338c-f3f8-458f-99bc-e85b0eda171e.png" style="width: 50%;"></div>
Training
<div style="text-align: center;">igure 2: Architecture of E4SRec solution. The left part illustrates the structure of E4SRec, including the input layer, the large anguage model layer and the prediction layer. The upper right part describes the efficient inference process. The lower right art shows the complete solution of E4SRec.</div>
pluggable components. Through the above stages, we provide the whole solution of E4SRec that can be used to build an industriallevel LLM-based recommender system from scratch. Please find the detailed description of the whole pipeline of E4SRec solution in Appendix A.
# 2.2 ID Injection
Although LLMs are powerful in modeling natural language and are able to produce rational responses, they are unaware of the meanings of IDs without textual features and thus are not able to handle pure ID information. However, the collaborative information contained in the IDs has been proven to be very effective and crucial in personalized recommender systems for a long time. Therefore, the disability of utilizing ID information strongly limits the practical value of LLMs in recommender systems so far. Considering the huge number of IDs and the extreme sparsity of collaborative signals, it is especially challenging to incorporate ID information into LLMs. The existing works [16, 49] have explored various methods to introduce IDs into LLMs for recommendation, including vocabulary expansion, character decomposition, sequential grouping and collaborative clustering. Nevertheless, these methods are all insufficient to effectively capture the collaborative information. Meanwhile, the projection of item names or semantic descriptions falls into content-based recommendation without personalization and is only effective when the number of items is quite small with succinct informative textual features. To address the disability of LLMs to handle ID information, we propose a novel approach by injecting the ID embeddings into the LLMs rather than learning them through the training or tuning
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3652/36529f17-d8cf-4072-a0bf-3ef2850f6639.png" style="width: 50%;"></div>
process of LLMs. Specifically, the ID embeddings are obtained by directly extracting from a pretrained sequential recommendation model. In this paper, we select the SASRec [18] model to provide ID embeddings considering its effectiveness and generalization. It is worth noting that it could be replaced by any other sequential recommendation models for improvement. The selected sequential recommendation model is pretrained on the given dataset for iterations to get its best performance. Then the ID embeddings of the items are directly extracted without any modification to be ready for the input of E4SRec, which can be represented as E ∈R𝑁×𝑑𝑠. 𝑁and 𝑑𝑠are the number of items and the dimension of ID embeddings, respectively. Since this sequential recommendation model is pretrained with only sequences of IDs, there is no information on any other features being exposed to the ID embeddings.
# 2.3 Backbone Tuning
The open-sourced LLMs are mainly created for the general purpose of natural language generation to answer any questions from users so that the output formats are diverse and typically wordy. However, in specific scenarios like recommendation, we expect the generated outputs of LLMs to strictly adhere to the given format usually as simple as a number or a word. At the same time, it is not desirable for us to get a LLM that is only able to achieve the given task by directly tuning on it. Hence, we aim to modify the LLMs through instruction tuning [45] on various tasks to enable the LLMs to complete any instructions according to the given template. Specifically, we choose the LLaMA2-13B [42] model as the backbone LLM due to its strong power and generalization. For the instruction tuning process, we follow the settings in Platypus [23]
Table 1: Example of Alpaca template for instruction tuning.
Instruction Input
### Instruction:
Please give the maximum common
subsequence of the following two strings.
### Input:
Large Language Model, Language Models
Instruction Output
### Response:
Language Model
to apply a Parameter-Efficient Fine-Tuning (PEFT) [26] method, LoRA [15], on the gate_proj, down_proj and up_proj modules of LLaMA2-13B. Using the PEFT method, only the parameters of the specified modules are trained (about 0.07% of the total parameters) so that the training efficiency is highly improved. The initial datasets and more detailed settings can be found at the project page of Platypus1. The instruction tuning [44] is conducted for one epoch using the Alpaca [40] template. An example of the Alpaca style prompt is illustrated in Table 1, which consists of instruction, input and response parts.
# 2.4 E4SRec
2.4.1 Input Layer. In this input layer, we aim to inject the IDs into LLMs along with the standard textual prompts. As illustrated in the left part of Figure 2, E4SRec also follows the Alpaca template and the item ID sequences are taken as the input part of the prompt. From another perspective, each item ID can be seen as a ‘word’ in the LLM but is projected using our proposed ID injection component rather than the lookup table of LLM. Thus, we obtain the corresponding ID embeddings of the items through the aforementioned ID injection component while the rest of the prompt is projected to word embeddings as the typical LLM inputs. The final inputs are the concatenation of the word embeddings and the ID embeddings in their original positions. Since the pretrained ID embeddings usually have a much lower dimension (e.g., 64 in our pretrained SASRec) than the word embeddings in LLM (e.g., 5120 in LLaMA213B), we employ a linear projection to convert the ID embeddings into the same dimension with the word embeddings. As mentioned above, we do not want the LLMs to encounter the ‘catastrophic forgetting’ phenomenon during the training on the recommendation task. Certainly, we also do not want the learned collaborative information to be destroyed either. Therefore, only the linear projection component is trainable, which is as small as a weight matrix W𝐼𝑛𝑝𝑢𝑡∈R𝑑𝑠×𝑑𝑘. 𝑑𝑠and 𝑑𝑘represent the dimensions of the ID embeddings in pretrained sequential recommendation model and the word embeddings in LLM, respectively. 2.4.2 Large Language Model Layer. The large language model layer employs the instruction-tuned LLaMA2-13B, which is the merge of the original LLaMA2-13B and the LoRA adapter parameters for instruction tuning. Similar to the LLM instruction tuning stage, we also introduce an additional LoRA adapter on the gate_proj, down_proj and up_proj modules to model the personalization of the given recommendation task.
2.4.2 Large Language Model Layer. The large language model layer employs the instruction-tuned LLaMA2-13B, which is the merge of the original LLaMA2-13B and the LoRA adapter parameters for instruction tuning. Similar to the LLM instruction tuning stage, we also introduce an additional LoRA adapter on the gate_proj, down_proj and up_proj modules to model the personalization of the given recommendation task.
1https://platypus-llm.github.io/
2.4.3 Prediction Layer. The prediction layer receives the output of the LLM and makes the predictions for recommendation. Existing LLM-based recommendation approaches mainly define the recommendation task as an open-domain natural language generation task, which is in line with the purpose of LLMs themselves. However, due to the characteristics of the task definition, they often generate out-of-range results and are only able to generate one recommendation result each time. Such a pathway impairs the reliability of the recommendation results and limits the recommendation efficiency. Therefore, these approaches can only work in the ranking stages (usually less than one hundred items) but are impractical in the matching stages (usually more than thousands of items). Unlike these existing approaches that directly adopt the task definition of LLMs without modification, we stick to their underlying definition of modeling joint probability distribution for generation. That is to say, we can compute the prediction results for a given sequence over all the candidates in each forward process. In order to achieve this goal, we dive deep into the structure of LLMs and propose to employ an item linear projection to replace the original prediction layer in LLMs via a weight matrix W𝑂𝑢𝑡𝑝𝑢𝑡∈R𝑑𝑘×𝑁where 𝑁denotes the total number of candidate items. Then, the predictions of a given sequence could be represented as a 𝑁-dimensional vector ˆy ∈R𝑁. In the training stage, we adopt the cross entropy loss between predictions and ground-truth next items as the learning objective of E4SRec as follows:
# 2.5 Inference & Deployment
The inference time is also an important and bothering problem for the application of LLMs. Although we cannot improve the performance of LLMs themselves, our proposed E4SRec solution can ensure that the overall inference time of the recommendation task is as close as that of a vanilla LLM. Review the structure of our proposed E4SRec, the additional components are all so small compared with the backbone LLM that they will only cost a little more time. More than that, the time-consuming softmax operation in the training stage is also unnecessary. Therefore, the inference process can be further simplified to the nearest neighbor search between the output of LLM and the vectors in the item linear projection component as shown in the upper right part of Figure 2. After the training stage is completed, our proposed E4SRec can be deployed in a very lightweight manner. The backbone LLM is onesize-fits-all for all the tasks including sequential recommendation so that the instruction tuning only needs to be done once and is shared across all the downstream tasks. For each coming sequential recommendation dataset, only the ID embeddings E, the linear projection in the input layer W𝐼𝑛𝑝𝑢𝑡, LoRA weights Θ and the item linear projection W𝑂𝑢𝑡𝑝𝑢𝑡are required to be trained and stored for deployment. Compared with the billions of parameters in the LLMs, these parameters are as tiny as about 1%. Meanwhile, all these components are completely pluggable so that the recommender system can quickly adapt to a specific dataset by simply replacing these pluggable components.
<div style="text-align: center;">Table 2: Statistics of the datasets.</div>
Dataset
# Users
# Items
# Actions
Sparsity
Beauty
22,363
12,101
198,502
99.93%
Sports
25,598
18,357
296,337
99.95%
Toys
19,412
11,924
167,597
99.93%
Yelp
30,431
20,033
316,354
99.95%
# 3 EXPERIMENTS
In this section, we will evaluate our proposed E4SRec on several real-world datasets with a selected set of widely-used baseline methods in sequential recommendation. Meanwhile, we will also present the ablation study, robustness analysis, efficiency analyses and discussions in order to answer the following questions: • RQ1: How does E4SRec compare with traditional sequential recommendation models on performance? • RQ2: How do the injected IDs and the large language model affect the performance of E4SRec? • RQ3: How effective is E4SRec in leveraging collaborative information to alleviate the data sparsity problem? • RQ4: How efficient is E4SRec on both inference time and storage space in deployment? • RQ5: How extensible is E4SRec for new items in industrial applications?
# 3.1 Experimental Settings
3.1.1 Datasets. To evaluate the effectiveness of E4SRec, we conduct the experiments on four widely-used real-world datasets. Specifically, Beauty, Sports and Toys are the datasets of sub-categories ‘Beauty’, ‘Sports and Outdoors’ and ‘Toys and Games’ in the Amazon review data [27]. Yelp is a popular platform and the dataset is widely-used in various recommendation tasks. Here, we only utilize the data after January 1st, 2019. The statistics of the datasets are shown in Table 2. For sequential recommendation, the interaction sequences of users are sorted by timestamps in ascending order. Following the previous works [34, 37], we apply the 5-core settings to filter the unpopular items with fewer than 5 interactions to ensure robust evaluation.
3.1.2 Evaluation Metrics. To avoid selection bias and provide more reliable results, we evaluate the performance of predictions on the whole item set, which is effective in evaluating the matching ability. Following previous works [18, 31], we also apply the leave-one-out strategy. For each interaction sequence of users, the last item is taken as test data, the second last one is taken as validation data and the remaining sequence is used for training. As for the evaluation metrics, we choose two types of widely-used metrics, which are topk Hit Ratio (HR@k) and top-k normalized Discounted Cumulative Gain (nDCG@k) with k = {5, 10, 20}. These metrics are averaged over all the users for report. Besides, we also perform an additional evaluation with negative sampling in the Appendix C according to [51, 52]. Specifically, 99 negative items are randomly sampled for each positive item. In this setting, we are able to evaluate the ranking ability of E4SRec.
3.1.3 Baseline Methods. The baseline methods chosen for comparison can be split into three categories: non-sequential methods, traditional sequential methods and self-supervised sequential methods. For non-sequential methods, we have: • POP is a heuristic method that directly ranks the items using their popularity defined as the interaction numbers. • BPR [33] utilizes the Bayesian Personalized Ranking (BPR) loss to optimize the matrix factorization (MF) [22] model for characterizing the pair-wise interactions. For traditional sequential methods, we have: • GRU4Rec [13] implements the GRU recurrent neural network for sequential modeling and then makes predictions for recommendation. • Caser [39] integrates both horizontal and vertical convolutional operations to better capture the high-order interactions within item sequences for recommendation. • SASRec [18] is a self-attentive sequential recommendation model with multi-head self-attention to model the complex sequential information. For self-supervised sequential methods, we have: • BERT4Rec [37] employs the Cloze [41] objective rather than the next-item prediction for sequential recommendation in a pretraining-tuning manner. • S3-Rec [51] applies contrastive learning to capture the correlations among items, sub-sequences and attributes. Specifically, we take the variant with only Mask Item Prediction (MIP) objective. • CL4SRec [48] incorporates the contrastive learning with the transformer-based sequential recommendation model to obtain more robust results. • ICLRec [4] leverages a latent intent variable to learn the users’ intent distribution from unlabeled item sequences to improve the transformer-based sequential recommendation model. Note that our proposed E4SRec only utilizes the ID information and no other features are exposed to E4SRec during the training and inference. Therefore, those methods that either implement data augmentation techniques [52, 53] or incorporate other features [8, 28] are orthogonal to E4SRec and are thus excluded for fair comparison. 3.1.4 Implementation Details. For GRU4Rec, Caser, BERT4Rec, S3Rec and ICLRec, we implement them using the public resources released by their authors. For other models, we implement them using PyTorch 2.0.1. The embedding dimension is set to 64 and the maximum sequence length is set to 50 for all the models on all the datasets. The model parameters are initialized with Xavier initialization and are optimized using Adam [19]. For our proposed E4SRec, we obtain the LLaMA2-13B using HuggingFace2 and conduct instruction tuning for one epoch. The ID embeddings are directly extracted from the pretrained SASRec model without any modification. To obtain better performance, we perform the grid search of the training configurations using the validation set. Specifically, we aim to find a better combination of learning rate, training epochs and LoRA modules. The best combinations for all the datasets are listed in Appendix B due to limited space. All the experiments are implemented using 8 NVIDIA Tesla
3.1.4 Implementation Details. For GRU4Rec, Caser, BERT4Rec, S3Rec and ICLRec, we implement them using the public resources released by their authors. For other models, we implement them using PyTorch 2.0.1. The embedding dimension is set to 64 and the maximum sequence length is set to 50 for all the models on all the datasets. The model parameters are initialized with Xavier initialization and are optimized using Adam [19]. For our proposed E4SRec, we obtain the LLaMA2-13B using HuggingFace2 and conduct instruction tuning for one epoch. The ID embeddings are directly extracted from the pretrained SASRec model without any modification. To obtain better performance, we perform the grid search of the training configurations using the validation set. Specifically, we aim to find a better combination of learning rate, training epochs and LoRA modules. The best combinations for all the datasets are listed in Appendix B due to limited space. All the experiments are implemented using 8 NVIDIA Tesla 2https://huggingface.co/meta-llama/Llama-2-13b
2https://huggingface.co/meta-llama/Llama-2-13b
<div style="text-align: center;">Table 3: Performance comparison of different methods. The best performance is highlighted in bold while the second best performance is underlined. The last column indicates the improvements over the best baseline models and all the results of E4SRec are statistically significant with p < 0.01 compared to the best baseline models.</div>
Dataset
Metric
POP
BPR
GRU4Rec
Caser
SASRec
BERT4Rec
S3-Rec
CL4SRec
ICLRec
E4SRec
Improv.
Beauty
HR@5
0.0072
0.0120
0.0164
0.0251
0.0333
0.0193
0.0327
0.0407
0.0436
0.0525
20.41%
HR@10
0.0114
0.0361
0.0289
0.0418
0.0581
0.0401
0.0591
0.0626
0.0653
0.0758
16.08%
HR@20
0.0195
0.0589
0.0478
0.0643
0.0915
0.0596
0.0898
0.0957
0.0974
0.1071
9.96%
nDCG@5
0.0040
0.0065
0.0086
0.0127
0.0179
0.0187
0.0175
0.0223
0.0240
0.0360
50.00%
nDCG@10
0.0053
0.0122
0.0142
0.0193
0.0258
0.0254
0.0268
0.0317
0.0338
0.0435
28.70%
nDCG@20
0.0073
0.0179
0.0169
0.0258
0.0342
0.0361
0.0370
0.0396
0.0416
0.0514
23.56%
Sports
HR@5
0.0055
0.0092
0.0137
0.0139
0.0170
0.0176
0.0157
0.0217
0.0238
0.0281
18.07%
HR@10
0.0090
0.0188
0.0274
0.0231
0.0289
0.0326
0.0265
0.0374
0.0393
0.0410
4.33%
HR@20
0.0149
0.0258
0.0438
0.0389
0.0477
0.0493
0.0460
0.0582
0.0553
0.0626
7.56%
nDCG@5
0.0040
0.0053
0.0096
0.0085
0.0091
0.0105
0.0098
0.0129
0.0152
0.0196
28.95%
nDCG@10
0.0051
0.0083
0.0137
0.0126
0.0129
0.0153
0.0135
0.0184
0.0212
0.0237
11.79%
nDCG@20
0.0066
0.0121
0.0171
0.0166
0.0177
0.0195
0.0182
0.0239
0.0250
0.0291
16.40%
Toys
HR@5
0.0064
0.0120
0.0097
0.0166
0.0445
0.0274
0.0492
0.0484
0.0509
0.0566
11.20%
HR@10
0.0079
0.0211
0.0196
0.0281
0.0698
0.0460
0.0698
0.0706
0.0725
0.0798
10.07%
HR@20
0.0108
0.0312
0.0301
0.0420
0.0999
0.0688
0.0962
0.0984
0.1018
0.1107
8.74%
nDCG@5
0.0037
0.0082
0.0059
0.0107
0.0236
0.0174
0.0342
0.0327
0.0350
0.0405
15.71%
nDCG@10
0.0057
0.0120
0.0098
0.0151
0.0318
0.0230
0.0375
0.0404
0.0423
0.0479
13.24%
nDCG@20
0.0062
0.0136
0.0116
0.0179
0.0394
0.0291
0.0431
0.0466
0.0493
0.0557
12.98%
Yelp
HR@5
0.0056
0.0127
0.0152
0.0142
0.0161
0.0186
0.0173
0.0216
0.0240
0.0266
10.83%
HR@10
0.0083
0.0245
0.0263
0.0252
0.0265
0.0291
0.0282
0.0352
0.0381
0.0418
9.71%
HR@20
0.0120
0.0346
0.0371
0.0406
0.0443
0.0564
0.0538
0.0585
0.0630
0.0675
7.14%
nDCG@5
0.0036
0.0076
0.0104
0.0096
0.0102
0.0121
0.0114
0.0130
0.0150
0.0189
26.00%
nDCG@10
0.0043
0.0119
0.0137
0.0129
0.0134
0.0171
0.0163
0.0185
0.0203
0.0238
17.24%
nDCG@20
0.0056
0.0143
0.0145
0.0156
0.0179
0.0223
0.0201
0.0235
0.0256
0.0297
16.02%
# A800 GPUs. The implementation code is available online3 and will be accessible to the public for ease of reproducibility.
A800 GPUs. The implementation code is available online3 and will be accessible to the public for ease of reproducibility.
# 3.2 Main Results (RQ1)
The performance comparison of our proposed E4SRec with other traditional sequential recommendation models is shown in Table 3. Here we have the following observations:
• Our proposed E4SRec can significantly outperform all the baseline methods on all four datasets thanks to the powerful ability of LLM. The relative improvements on performance over the best baseline methods are about 12% on HR@k and 21% on nDCG@k, which fully demonstrate the effectiveness of E4SRec. • Generally, the performance gains are greater on nDCG@k metrics than on HR@k metrics and are greater on smaller k values. This indicates that our proposed E4SRec can capture the users’ preference more accurately and thus generate more reliable recommendation results. • The non-sequential methods are much worse than sequential methods due to the disadvantage of modeling sequential information. The dramatically bad performance of POP also indicates the important and crucial role of personalization in these datasets. • The self-supervised sequential methods are usually stronger than the traditional sequential methods. This phenomenon shows the
3https://github.com/HestiaSky/E4SRec/
effectiveness of introducing self-supervised learning to provide additional training signals in improving the sequential recommendation performance. • CL4SRec and ICLRec achieve much better performance with the help of contrastive learning in improving the robustness of item representations. However, contrastive learning applies data augmentation in a certain extent by introducing augmented sequences, which may be a little bit unfair to the other methods. Therefore, SASRec is still a very strong method compared with them and is more suitable for reference. In such a situation, our proposed E4SRec will have an even more significant performance gain of up to 115%.
# 3.3 Ablation Study (RQ2)
In order to explore the impacts of the injected IDs and the LLMs on the overall performance of E4SRec, we perform an ablation study by changing the injected IDs and the LLMs to design and then compare the following four variants:
In order to explore the impacts of the injected IDs and the LLMs on the overall performance of E4SRec, we perform an ablation study by changing the injected IDs and the LLMs to design and then compare the following four variants: • BPR + LLaMA: This variant employs the BPR model to provide item ID embeddings and the LLaMA2-13B model. • SASRec w/o LLM: This is a LLM-free variant that utilizes the SASRec as the sequential model for ID injection. Specifically, it has no LLM and only uses linear projections for prediction. • SASRec + BERT: A BERT [7] model is utilized in the LLM layer. Here we use BERT-base-uncased with 110M parameters.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b9e6/b9e65033-6360-405d-bb0f-b0c56f7dcb8a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Ablation study of ID injection and LLM.</div>
• SASRec + LLaMA: It is the basic version of our proposed E4SRec for reference with SASRec for ID injection and LLaMA2-13B as the LLM. From the performance comparison of HR@k metrics in Figure 3, we have the following observations: • The qualities of item ID embeddings do have effects on the performance. The performance of BPR + LLaMA is slightly worse than SASRec + LLaMA. Although BPR is unable to capture sequential information and has poor performance, the performance of BPR + LLaMA is still satisfying. This phenomenon may indicate that the main effect of ID injection is to provide collaborative information while the LLM is already sufficient to well capture sequential information during the training stage. • The extremely poor performance of SASRec w/o LLM demonstrates that the LLMs are necessary to conduct recommendation. • The inherent ability of the LLMs also has a significant impact on overall performance. Although BERT is already a very powerful language model, the performance gap is still significant compared to LLaMA. This also implies the effectiveness of our proposed E4SRec solution in leveraging the capabilities and tapping into the potentials of LLMs.
# 3.4 Robustness Analysis (RQ3)
Data sparsity problem is a common issue of recommender systems that defects the performance in applications. For example, since most of the users only have limited interactions, the user cold-start problem is typically severe and thus harms the user experience. Traditional recommendation methods alleviate the data sparsity problems by jointly leveraging the data from other users and items via collaborative filtering. To verify the effectiveness of our proposed E4SRec in incorporating collaborative information, we compared it with SASRec on the robustness with different sparsity levels’ data as shown in Figure 4. Specifically, we split the users into three groups based on their number of interactions. Based on the results, we have observations as follows:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b031/b0313f09-9018-4d31-acf2-b36b6324f668.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Performance comparison on different user groups with respect to the length of interaction sequences.</div>
• The performance gains are more significant on sparser user groups than denser ones. Such a phenomenon may be originated from the few-/zero-shot learning ability of LLMs to enable E4SRec to achieve satisfying performance with insufficient data that traditional recommendation methods can hardly learn from.
# 3.5 Efficiency Analysis (RQ4)
In addition to being effective in sequential recommendation, our proposed E4SRec solution is also very efficient in both time and space. To support this statement, we provide a comparison of inference time and storage space between the backbone LLM, LLaMA213B and E4SRec on 1 Nvidia Tesla A800 GPU as illustrated in Table 4. As mentioned above, E4SRec introduces only a tiny pluggable set of parameters to the backbone LLMs and doesn’t change the overall data flow too much. Therefore, the inference time of E4SRec is very close to the backbone LLaMA2-13B. Considering the ability to generate results without the need for multiple times generation, our E4SRec is obviously more efficient by orders of magnitude on the actual inference time than the existing solutions which directly apply the task formulation of natural language generation. Meanwhile, the additional parameters in E4SRec are as tiny as 1% of the total parameters of the backbone LLM. Since this set of additional parameters is pluggable to have no effect on the backbone
Table 4: Comparisons of inference time and storage space between LLaMA2-13B and E4SRec.
Model
Inference Time
Storage Space
LLaMA2-13B
0.21s/Instance
12.12GB
E4SRec
0.23s/Instance
12.23GB
LLM, E4SRec can be stored and deployed with only one shared backbone LLM and multiple independent pluggable components for specific datasets. In this manner, there is only 2 times of space needed than the backbone LLM to store 100 E4SRec models, which is very economical in industrial applications.
# 3.6 Discussions (RQ5)
In industrial applications, there are many new items emerging into the recommender systems every day. Apparently, retraining the recommendation model for each coming item is not practical so the ability to extend to new items is very crucial. Our E4SRec solution guarantees that the pluggable components are independent of IDs, which means adding a new item in the dataset only requires adding a new row in the linear projections without the need to retrain the entire model. Therefore, our proposed E4SRec solution is considerably extensible.
# 4 RELATED WORKS 4.1 Sequential Recommendation
# 4 RELATED WORKS
# 4.1 Sequential Recommendation
Sequential recommendation is an important task in personalized recommender system which aims to capture the users’ preference using their historical behavior sequences. Early works of sequential recommendation mainly lie in the pattern of Markov Chains [12, 32] that uses an item-item transition pattern to directly predict the next item with the previous items. For example, FPMC [34] uses Matrix Factorization (MF) to model the users’ preference and utilizes Markov Chains to capture the sequential patterns for making prediction. With the booming development of deep learning, many sequential recommendation models with deep neural networks emerged. Caser [39] employs Convolutional Neural Network (CNN) and GRU4Rec [13] utilizes Recurrent Neural Network (RNN) to capture the high-order interactions within the item sequences for sequential modeling. In recent years, more powerful models are proposed with more advanced architectures to better leverage sequential information. SASRec [18] takes advantage of the multihead self-attention to attentively model sequential information in an unidirectional manner. BERT4Rec [37] improves such a manner by employing the Cloze [41] objective to predict the masked item to leverage the bidirectional information. S3-Rec [51] introduces contrastive learning to fuse the information of distinct items, subsequences and attributes. Similarly, CL4SRec [48] and ICLRec [4] both apply contrastive learning to better capture the users’ preference with sequential information. There are also many other approaches that aim to incorporate other features, e.g. DuoRec [28] and EMKD [8], or employ data augmentation, e.g. FMLP-Rec [52] and ECL-SR [53] for further improvements.
Nevertheless, traditional sequential recommendation models are usually limited on performance for the limitations on the model scale and may lead to sub-optimal prediction. Unlike them, our proposed E4SRec takes advantages of LLMs to achieve more advanced performance and empower the generative recommendation, which can better comprehend human intentions and generate more human-like language responses.
# 4.2 LLMs for Recommendation
Large Language Models (LLMs) have been proven to be very powerful in natural language processing and their strong power has encouraged researchers to make efforts to apply LLM for recommendation [9]. Early approaches view LLMs as feature extractors to generate knowledge-aware embeddings for recommendation. U-BERT [29] proposes a pretraining-tuning framework to learn users’ representations and conduct user modeling. UserBERT [46] employs two self-supervision tasks on unlabeled behavior data to empower user modeling. With the emergence of generative LLMs like GPT, LLM-based recommendation has also shifted towards generative recommendation. These methods translate recommendation tasks as natural language tasks to directly generate the recommendation results [47]. At first, most approaches focus on using prompting [10, 38] or in-context learning [5, 25] to adapt LLMs for recommendation. However, these approaches fail to surpass the traditional recommendation models trained specifically for a given task on specific data. Therefore, many efforts are made to align the LLMs to recommendation by further fine-tuning recently. P5 [11] first proposes a unified framework to integrate five recommendation tasks via fine-tuning on FLAN-T5 [30]. Following it, InstructRec [50] adapts FLAN-T5 model to several downstream recommendation tasks by instruction tuning with more diverse texts. TALLRec [2] aligns the LLaMA model to the binary recommendation task by two stages of instruction tuning in few-shot scenario. GenRec [17] directly conducts instruction tuning on the LLaMA model with plain texts to achieve generative recommendation. However, all the above methods are essentially content-based recommendation and require rich semantic features to achieve satisfying performance. Therefore, they fail to handle the IDs and are unable to leverage collaborative information. Compared with them, our proposed E4SRec solution is more effective in handling IDs, more efficient on both time and space perspective and more extensible to fulfill real-world needs in application.
# 5 CONCLUSION
Existing approaches of LLM for recommendation face challenges in handling IDs, efficiency, extensiblility and thus are not able to fulfill the requirements of real-world applications. In this paper, we propose a novel E4SRec solution, which is elegant, effective, efficient and extensible to apply LLMs for sequential recommendation. Specifically, we introduce an elegant way to address the issue of handling IDs by injecting item ID embeddings into the LLM. Meanwhile, with the help of a modified prediction layer, we effectively solve the challenging out-of-range problem of generated results to ensure their legality and efficiently generate the predictions over all the candidates at once. The design of pluggable
components in E4SRec enables the model to be trained and deployed in a lightweight manner. The extensive experiments on four popular read-world datasets fully demonstrate the effectiveness and superiority of our proposed E4SRec solution. We hope our E4SRec solution will contribute to the research in applying LLM for recommender systems. Looking forward, we will pivot towards crafting elegant solutions for other recommendation tasks, such as CTR prediction, and continually pushing the frontiers of generative recommendations.
[1] Gediminas Adomavicius and Alexander Tuzhilin. 2005. Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions. IEEE Trans. Knowl. Data Eng. 17, 6 (2005), 734–749. [2] Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems, RecSys 2023, Singapore, Singapore, September 18-22, 2023. ACM, 1007–1014. [3] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual. [4] Yongjun Chen, Zhiwei Liu, Jia Li, Julian J. McAuley, and Caiming Xiong. 2022. Intent Contrastive Learning for Sequential Recommendation. In WWW ’22: The ACM Web Conference 2022, Virtual Event, Lyon, France, April 25 - 29, 2022. ACM, 2172–2182. [5] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering ChatGPT’s Capabilities in Recommender Systems. In Proceedings of the 17th ACM Conference on Recommender Systems, RecSys 2023, Singapore, Singapore, September 18-22, 2023. ACM, 1126–1132. [6] Mukund Deshpande and George Karypis. 2004. Item-based top-N recommendation algorithms. ACM Trans. Inf. Syst. 22, 1 (2004), 143–177. [7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers). Association for Computational Linguistics, 4171–4186. [8] Hanwen Du, Huanhuan Yuan, Pengpeng Zhao, Fuzhen Zhuang, Guanfeng Liu, Lei Zhao, Yanchi Liu, and Victor S. Sheng. 2023. Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023. ACM, 58–67. [9] Wenqi Fan, Zihuai Zhao, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Jiliang Tang, and Qing Li. 2023. Recommender Systems in the Era of Large Language Models (LLMs). CoRR abs/2307.02046 (2023). 10] Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang. 2023. Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System. CoRR abs/2303.14524 (2023). 11] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5). In RecSys ’22: Sixteenth ACM Conference on Recommender Systems, Seattle, WA, USA, September 18 - 23, 2022. ACM, 299–315. 12] Ruining He and Julian J. McAuley. 2016. Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation. In IEEE 16th International Conference on Data Mining, ICDM 2016, December 12-15, 2016, Barcelona, Spain. IEEE Computer Society, 191–200. 13] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. In 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings. 14] Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian J. McAuley, and Wayne Xin Zhao. 2023. Large Language Models are Zero-Shot Rankers for Recommender Systems. CoRR abs/2305.08845 (2023). 15] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large
Language Models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net. [16] Wenyue Hua, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. 2023. How to Index Item IDs for Recommendation Foundation Models. CoRR abs/2305.06569 (2023). [17] Jianchao Ji, Zelong Li, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Juntao Tan, and Yongfeng Zhang. 2023. GenRec: Large Language Model for Generative Recommendation. CoRR abs/2307.00457 (2023). [18] Wang-Cheng Kang and Julian J. McAuley. 2018. Self-Attentive Sequential Recommendation. In IEEE International Conference on Data Mining, ICDM 2018, Singapore, November 17-20, 2018. IEEE Computer Society, 197–206. [19] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings. [20] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large Language Models are Zero-Shot Reasoners. In NeurIPS. [21] Joseph A. Konstan, Bradley N. Miller, David A. Maltz, Jonathan L. Herlocker, Lee R. Gordon, and John Riedl. 1997. GroupLens: Applying Collaborative Filtering to Usenet News. Commun. ACM 40, 3 (1997), 77–87. [22] Yehuda Koren, Robert M. Bell, and Chris Volinsky. 2009. Matrix Factorization Techniques for Recommender Systems. Computer 42, 8 (2009), 30–37. [23] Ariel N. Lee, Cole J. Hunter, and Nataniel Ruiz. 2023. Platypus: Quick, Cheap, and Powerful Refinement of LLMs. CoRR abs/2308.07317 (2023). [24] Greg Linden, Brent Smith, and Jeremy York. 2003. Amazon.com Recommendations: Item-to-Item Collaborative Filtering. IEEE Internet Comput. 7, 1 (2003), 76–80. [25] Junling Liu, Chao Liu, Renjie Lv, Kang Zhou, and Yan Zhang. 2023. Is ChatGPT a Good Recommender? A Preliminary Study. CoRR abs/2304.10149 (2023). [26] Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. 2022. PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods. https://github.com/huggingface/peft. [27] Julian J. McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. 2015. Image-Based Recommendations on Styles and Substitutes. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, Santiago, Chile, August 9-13, 2015. ACM, 43–52. [28] Ruihong Qiu, Zi Huang, Hongzhi Yin, and Zijian Wang. 2022. Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation. In WSDM ’22: The Fifteenth ACM International Conference on Web Search and Data Mining, Virtual Event / Tempe, AZ, USA, February 21 - 25, 2022. ACM, 813–823. [29] Zhaopeng Qiu, Xian Wu, Jingyue Gao, and Wei Fan. 2021. U-BERT: Pre-training User Representations for Improved Recommendation. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021. AAAI Press, 4320–4327. [30] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. J. Mach. Learn. Res. 21 (2020), 140:1–140:67. [31] Ruiyang Ren, Zhaoyang Liu, Yaliang Li, Wayne Xin Zhao, Hui Wang, Bolin Ding, and Ji-Rong Wen. 2020. Sequential Recommendation with Self-Attentive Multi-Adversarial Network. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020. ACM, 89–98. [32] Steffen Rendle. 2010. Factorization Machines. 2010 IEEE International Conference on Data Mining (2010), 995–1000. [33] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. In UAI 2009, Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, Montreal, QC, Canada, June 18-21, 2009. AUAI Press, 452–461. [34] Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized Markov chains for next-basket recommendation. In Proceedings of the 19th International Conference on World Wide Web, WWW 2010, Raleigh, North Carolina, USA, April 26-30, 2010. ACM, 811–820. [35] Paul Resnick, Neophytos Iacovou, Mitesh Suchak, Peter Bergstrom, and John Riedl. 1994. GroupLens: An Open Architecture for Collaborative Filtering of Netnews. In CSCW ’94, Proceedings of the Conference on Computer Supported Cooperative Work, Chapel Hill, NC, USA, October 22-26, 1994. ACM, 175–186. [36] Badrul Munir Sarwar, George Karypis, Joseph A. Konstan, and John Riedl. 2001. Item-based collaborative filtering recommendation algorithms. In Proceedings of the Tenth International World Wide Web Conference, WWW 10, Hong Kong, China, May 1-5, 2001. ACM, 285–295. [37] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, CIKM 2019, Beijing, China, November 3-7, 2019. ACM, 1441–1450.
[38] Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie Ren, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent. CoRR abs/2304.09542 (2023). [39] Jiaxi Tang and Ke Wang. 2018. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, WSDM 2018, Marina Del Rey, CA, USA, February 5-9, 2018. ACM, 565–573. [40] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford Alpaca: An Instruction-following LLaMA model. https://github.com/tatsu-lab/stanford_ alpaca. [41] Wilson L. Taylor. 1953. “Cloze Procedure”: A New Tool for Measuring Readability. Journalism & Mass Communication Quarterly 30 (1953), 415 – 433. [42] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and Efficient Foundation Language Models. CoRR abs/2302.13971 (2023). [43] Eva Anna Maria van Dis, Johan Bollen, Willem Zuidema, Robert van Rooij, and Claudi L H Bockting. 2023. ChatGPT: five priorities for research. Nature 614 (2023), 224–226. [44] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-Instruct: Aligning Language Models with Self-Generated Instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023. Association for Computational Linguistics, 13484–13508. [45] Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022. Finetuned Language Models are Zero-Shot Learners. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net. [46] Chuhan Wu, Fangzhao Wu, Tao Qi, and Yongfeng Huang. 2022. UserBERT: Pre-training User Model with Contrastive Self-supervision. In SIGIR ’22: The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, Madrid, Spain, July 11 - 15, 2022. ACM, 2087–2092. [47] Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, Hongchao Gu, Tingjia Shen, Chuan Qin, Chen Zhu, Hengshu Zhu, Qi Liu, Hui Xiong, and Enhong Chen. 2023. A Survey on Large Language Models for Recommendation. CoRR abs/2305.19860 (2023). [48] Xu Xie, Fei Sun, Zhaoyang Liu, Shiwen Wu, Jinyang Gao, Jiandong Zhang, Bolin Ding, and Bin Cui. 2022. Contrastive Learning for Sequential Recommendation. In 38th IEEE International Conference on Data Engineering, ICDE 2022, Kuala Lumpur, Malaysia, May 9-12, 2022. IEEE, 1259–1273. [49] Shuyuan Xu, Wenyue Hua, and Yongfeng Zhang. 2023. OpenP5: Benchmarking Foundation Models for Recommendation. CoRR abs/2306.11134 (2023). [50] Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. 2023. Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach. CoRR abs/2305.07001 (2023). [51] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. In CIKM ’20: The 29th ACM International Conference on Information and Knowledge Management, Virtual Event, Ireland, October 19-23, 2020. ACM, 1893–1902. [52] Kun Zhou, Hui Yu, Wayne Xin Zhao, and Ji-Rong Wen. 2022. Filter-enhanced MLP is All You Need for Sequential Recommendation. In WWW ’22: The ACM Web Conference 2022, Virtual Event, Lyon, France, April 25 - 29, 2022. ACM, 2388–2399. [53] Peilin Zhou, Jingqi Gao, Yueqi Xie, Qichen Ye, Yining Hua, Jaeboum Kim, Shoujin Wang, and Sunghun Kim. 2023. Equivariant Contrastive Learning for Sequential Recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems, RecSys 2023, Singapore, Singapore, September 18-22, 2023. ACM, 129–140.
<div style="text-align: center;">Algorithm 1: Pipeline of E4SRec Solution.</div>
Algorithm 1: Pipeline of E4SRec Solution.
Input
:recommendation dataset D, instruction dataset
D𝐼𝑛𝑠and pretrained LLM M𝐿𝐿𝑀
Output:E4SRec model M
// Preliminary
1 Train a sequential recommendation model M𝑆𝑒𝑞on D.
2 Extract item ID embeddings E.
3 Instruction tune the LLM model M𝐿𝐿𝑀on D𝐼𝑛𝑠.
// Model Training
4 Initialize pluggable components Θ, including input linear
projection, LoRA weights and item linear projection.
5 for 𝑇←0 to 𝑇𝑚𝑎𝑥iterations do
6
Sample an instance for training.
// ID Injection
7
Obtain the corresponding ID embeddings from E;
Project ID embeddings to the same dimension with
M𝐿𝐿𝑀using input linear projection.
// Prediction
8
Feed ID embeddings and prompt to M𝐿𝐿𝑀for output;
Predict candidate items using item linear projection.
// Update Parameters
9
Compute the cross-entropy loss L via Equation 1;
Update Θ using L.
10 end
// Model Deployment
11 Deploy backbone model M𝐿𝐿𝑀.
12 Deploy E4SRec model for dataset D with
M = M𝐿𝐿𝑀←E, Θ.
<div style="text-align: center;">Table 5: Training configuration on the four datasets.</div>
Parameter
Beauty
Sports
Toys
Yelp
Training Epochs
3
3
2
5
Learning Rate
3e-4
2e-4
2e-4
3e-4
Batch Size
16
LoRA Rank
16
LoRA Alpha
16
LoRA Dropout
0.05
LoRA Modules
[gate_proj, down_proj, up_proj]
Learning Rate Scheduler
Cosine Scheduler
Weight Decay
0.1
Warmup Steps
100
200
100
300
# A PIPELINE OF E4SREC SOLUTION
The whole pipeline of E4SRec solution is described in Algorithm 1. Following such a process, one can easily train and deploy a E4SRec model in a lightweight manner.
# B TRAINING CONFIGURATION
LLMs are sensitive to the training configuration. In order to get desirable results, we implement different configurations on different datasets as shown in Table 5. Typically, different datasets require
<div style="text-align: center;">Table 6: Performance comparison of different methods with sampled negative items. The best performance is highlighted old while the second best performance is underlined. The last column indicates the improvements over the best basel models and all the results of E4SRec are statistically significant with p < 0.01 compared to the best baseline models.</div>
Dataset
Metric
POP
BPR
GRU4Rec
Caser
SASRec
BERT4Rec
E4SRec
Improv.
Beauty
HR@1
0.0678
0.0405
0.1337
0.1337
0.1870
0.1531
0.2274
21.60%
HR@5
0.2105
0.1461
0.3125
0.3032
0.3741
0.3640
0.4088
9.28%
nDCG@5
0.1391
0.0934
0.2268
0.2219
0.2848
0.2622
0.3221
13.10%
HR@10
0.3386
0.2311
0.4106
0.3942
0.4696
0.4739
0.5068
6.94%
nDCG@10
0.1803
0.1207
0.2584
0.2512
0.3156
0.2975
0.3503
10.99%
MRR
0.1558
0.1096
0.2308
0.2263
0.2852
0.2614
0.3182
11.57%
Sports
HR@1
0.0763
0.0489
0.1160
0.1135
0.1455
0.1255
0.1732
19.04%
HR@5
0.2293
0.1603
0.3055
0.2866
0.3466
0.3375
0.3721
7.36%
nDCG@5
0.1538
0.1048
0.2126
0.2020
0.2497
0.2341
0.2701
8.17%
HR@10
0.3423
0.2491
0.4299
0.4014
0.4622
0.4722
0.4821
2.10%
nDCG@10
0.1902
0.1334
0.2527
0.2390
0.2869
0.2775
0.2991
4.25%
MRR
0.1660
0.1202
0.2191
0.2100
0.2520
0.2378
0.2675
6.15%
Toys
HR@1
0.0585
0.0257
0.0997
0.1114
0.1878
0.1262
0.2075
10.49%
HR@5
0.1977
0.0978
0.2795
0.2614
0.3682
0.3344
0.3908
6.14%
nDCG@5
0.1286
0.0614
0.1919
0.1885
0.2820
0.2327
0.3115
10.46%
HR@10
0.3008
0.1715
0.3896
0.3540
0.4663
0.4493
0.4850
4.01%
nDCG@10
0.1618
0.0850
0.2274
0.2183
0.3136
0.2698
0.3353
6.92%
MRR
0.1430
0.0819
0.1973
0.1967
0.2842
0.2338
0.3040
6.97%
Yelp
HR@1
0.0801
0.0624
0.2053
0.2188
0.2375
0.2405
0.2725
13.31%
HR@5
0.2415
0.2036
0.5437
0.5111
0.5745
0.5976
0.6202
3.78%
nDCG@5
0.1622
0.1333
0.3784
0.3696
0.4113
0.4252
0.4532
6.58%
HR@10
0.3609
0.3153
0.7265
0.6661
0.7373
0.7597
0.7755
2.08%
nDCG@10
0.2007
0.1692
0.4375
0.4198
0.4642
0.4778
0.5037
5.42%
MRR
0.1740
0.1470
0.3630
0.3595
0.3927
0.4026
0.4306
6.95%
different training epochs and learning rates to achieve better performance (usually 2-5 epochs is enough) while the number of warmup steps is also important. The batch size is better to be 16 while larger or smaller batch size may lead to failure in convergence. E4SRec is not so sensitive to the rank and alpha of LoRA but changing the dropout or modules of LoRA will lead to significant performance fluctuation. We choose to use the cosine scheduler with a weight decay of 0.1 to control the learning rate in training process.
# C ADDITIONAL RESULTS
In addition to the main results, we also perform an evaluation with negative sampled items following the strategy of previous works [18, 51, 52]. Specifically, we adopt HR@1 (nDCG@1=HR@1), HR@5, HR@10, nDCG@5, nDCG@10 and MRR as evaluation metrics considering the much shorter candidate lists. As illustrated in Table 6, our proposed E4SRec can still exceed all the baseline models with a significant margin. Such results prove the superiority of E4SRec on the ranking ability.
