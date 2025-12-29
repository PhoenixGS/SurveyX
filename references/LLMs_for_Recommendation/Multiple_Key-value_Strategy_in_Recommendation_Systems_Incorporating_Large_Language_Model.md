# Multiple Key-value Strategy in Recommendation Systems Incorporating Large Language Model

Dui Wang wangdui@meituan.com Meituan Inc Beijing, China
Xiangyu Hou houxiangyu02@meituan.com Meituan Inc Beijing, China

Dui Wang wangdui@meituan.com Meituan Inc Beijing, China
Xiangyu Hou houxiangyu02@meituan.com Meituan Inc Beijing, China
Xiaohui Yang yangxiaohui04@meituan.com Meituan Inc Beijing, China

Xiangyu Hou houxiangyu02@meituan.com Meituan Inc Beijing, China
Xiaohui Yang yangxiaohui04@meituan.com Meituan Inc Beijing, China

Renbing Chen chenrenbing@meituan.com Meituan Inc Beijing, China

Bo zhang zhangbo58@meituan.com Meituan Inc Beijing, China

# ABSTRACT

Recommendation system (RS) plays significant roles in matching users’ information needs for Internet applications, and it usually utilizes the vanilla neural network as the backbone to handle embedding details. Recently, the large language model (LLM) has exhibited emergent abilities and achieved great breakthroughs both in the CV and NLP communities. Thus, it is logical to incorporate RS with LLM better, which has become an emerging research direction. Although some existing works have made their contributions to this issue, they mainly consider the single key situation (e.g. historical interactions), especially in sequential recommendation. The situation of multiple key-value data is simply neglected. This significant scenario is mainstream in real practical applications, where the information of users (e.g. age, occupation, etc) and items (e.g. title, category, etc) has more than one key. Therefore, we aim to implement sequential recommendations based on multiple key-value data by incorporating RS with LLM. In particular, we instruct tuning a prevalent open-source LLM (Llama 7B) in order to inject domain knowledge of RS into the pre-trained LLM. Since we adopt multiple key-value strategies, LLM is hard to learn well among these keys. Thus the general and innovative shuffle and mask strategies, as an innovative manner of data argument, are designed. To demonstrate the effectiveness of our approach, extensive experiments are conducted on the popular and suitable dataset MovieLens which contains multiple keys-value. The experimental results demonstrate that our approach can nicely and effectively complete this challenging issue.

# CCS CONCEPTS

• Information systems → Recommender systems; •  Computing methodologies → Natural language processing.

# • Information systems → Recommender systems; •  Computing methodologies → Natural language processing.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. In submission to CIKM’23 workshop, , © 2023 Association for Computing Machinery. ACM ISBN 978-x-xxxx-xxxx-x/YY/MM...$15.00 https://doi.org/10.1145/nnnnnnn.nnnnnnn

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. In submission to CIKM’23 workshop, , © 2023 Association for Computing Machinery. ACM ISBN 978-x-xxxx-xxxx-x/YY/MM...$15.00 https://doi.org/10.1145/nnnnnnn.nnnnnnn

Xiaohui Yang yangxiaohui04@meituan.com Meituan Inc Beijing, China

Daiyue Xue xuedaiyue@meituan.com Meituan Inc Beijing, China

# KEYWORDS

Recommendation System, Large-Language-Model, Sequential Recommendations, Instruction tuning, Data argument

ACM Reference Format: Dui Wang, Xiangyu Hou, Xiaohui Yang, Bo zhang, Renbing Chen, and Daiyue Xue. 2023. Multiple Key-value Strategy in Recommendation Systems Incorporating Large Language Model. In Proceedings of (In submission to CIKM’23 workshop). ACM, New York, NY, USA, 5 pages. https://doi.org/10. 1145/nnnnnnn.nnnnnnn

# 1 INTRODUCTION

Recommendation Systems (RS) aim to address the issue of online information overload and generate target items for the user according to the user’s preferences, which are often achieved by their historical interactions. In the literature on recommendation systems, most of the existing works[3, 7, 19] train deep neural networks to extract user and item features, enabling more complex and challenging recommendation applications, such as playlist generators [8, 18] for listeners, product recommendations [11, 20, 25] for customers, and content recommendations [13, 14, 23, 24] for readers, among others. On the other hand, Large language model (LLM) [15, 17, 21, 22] has sparked a revolution in the field of natural language processing (NLP). With the rapid increase in LLM parameters, LLM has shown impressive emergent abilities (e.g. reasoning [27], understanding [9], in-context few-shot learning [1]). Given the success of LLMs, RS may benefit from their prosperity by incorporating them [10]. However, the key challenge of this issue is how to convert structured data to natural language. The raw data of recommendation systems are discrete and key-based, such as id, age, occupation for users, and title, category, actions for items. In contrast, the input of LLM should be continuous natural language text, which contains abundant semantic information. To address this challenging problem, various works have been conducted and achieved promising results. For example, [26] proposed various manual-designed templates to generate instruction data, converting the discrete and structured raw data to natural language text format. Others [6, 12] explored the ability of LLM to understand recommendation tasks using zero-shot and few-shot strategies. Additionally, [2] proposed a flexible and unified text-to-text paradigm, which unified various recommendation tasks in a shared framework.

Although some existing works have made their contributions to successfully incorporating RS with LLM, they mainly consider the single key (e.g. historical interactions) scenario in the recommendation task. Specifically, the raw data of RS are often discrete and key-based. For example, the MovieLens dataset, as the popular benchmark for RS, has various keys for both users(e.g. gender, age, occupation) and items(e.g. title, rating, category). In practical applications, it is common to utilize various key information about users and items to improve performance, instead of just utilizing a single key. Therefore, utilizing multiple key-value data is necessary and effective for RS, along with the combination of LLM. However, it is difficult to let an LLM adapt to recommendation systems, as LLM may result in unsatisfactory performance due to the increase in model inputs and semantic complexity. Therefore, it is necessary and worthwhile to invest in this issue and provide valuable insights to explore the potential of LLM in RS. To bridge this research gap, we take into account multiple keyvalue data and utilize LLM to complete sequential RS. To the best of our knowledge, this is the first principled work for sequential RS on multiple key-value data by utilizing LLM. In this section, we first analyze the core challenge, which is the mismatch problem between the structured raw data and the desired natural language text data. To address this issue, we can conclude into two methodologies: 1) Adapting LLM to the structured raw data by modifying the construction of LLM; 2) Adapting the structured raw data to LLM by converting these data to continuous natural language text data. While the first solution can better handle the key-value data, modifying the model construction or adding the components may disturb the model structure and result in knowledge loss. Regarding the second solution, it is more flexible and only requires an extra process of training data without changing the model. Furthermore, this strategy can effectively preserve knowledge of the pre-train model. Based on the above analysis, we adopt the second strategy and propose a template that converts the structured raw data to natural language text data. However, multiple key-value data may lead to poor learning performance due to the large number of input tokens. To address this problem, we further propose two novel data argument strategies to enhance the semantic association between these keys and candidates. Specifically, we propose a shuffle strategy to shuffle key-value pairs or candidates and a mask strategy to randomly mask out part of key values. These augmented data are associated with the same label. In addition, it is possible to flexibly combine these proposed strategies based on specific situations. We conduct extensive experiments on the popular dataset (MovieLens), and the results demonstrate the effectiveness of the proposed strategies.

# 2 NOTATIONS AND PRELIMINARIES

In this work, we discuss a typical sequential recommendation for RS. Let 𝐼 = {𝑖 1,𝑖 2, ...,𝑖 | 𝐼 |} be the set of items and 𝑈 = {𝑈 1,𝑈 2, ...,𝑈 | 𝑈 |} be the set of users. We define 𝐾 𝑢 = {𝑘 𝑢 1,𝑘 𝑢 2, ...,𝑘 𝑢 | 𝐾 𝑢 |} as the set for
users, while 𝐾 𝑖 = {𝑘 𝑖 1,𝑘 𝑖 2, ...,𝑘 𝑖 | 𝐾 𝑖 |} as the set of keys for items. 𝑉 𝑢
and 𝑉 𝑖 are the corresponding value set of keys for users and items, respectively. Regarding both keys and values, we can define 𝐾𝑉 𝑢 𝑗 =
{<𝑘 𝑢 1,𝑉 𝑗 1>, <𝑘 𝑢 2,𝑉 𝑗 2>, ..., <𝑘 𝑢 | 𝐾 𝑢 |,𝑉 𝑗 | 𝐾 𝑢 |>}, 𝑗 ∈ 𝑈 as key-value

pairs for the 𝑗-index user, and 𝐾𝑉 𝑖 𝑗 is define for the 𝑗-index item in the same way. Due to the sequential recommendation, one of 𝐾 𝑖
is the historical interaction. We construct a user’s action sequence 𝑆 𝑢 = {𝑆 𝑈 1,𝑆 𝑈 2, ..,𝑆 𝑈 | 𝑆 𝑈 |}. The LLM can be defined as 𝐹: 𝑆 𝑢 → 𝑅, where 𝑅 is the re-ranked sequence of the candidate set and the first item in the line is the target user’s next item. As we apply LLM, we define 𝐼𝑛𝑠 = {𝑥 1,𝑥 2, ...,𝑥 𝑚}, where 𝐼𝑛𝑠 represents a set of 𝑚 instruction samples to be constructed for LLM training.

# 3 METHODOLOGY

In this section, we present our methodology in the following sections. Specifically, to convert structured multiple key-value data to natural language format, we propose a template designed for large language model input structure. Although the core challenges mentioned above can be alleviated, the model may still perform poorly due to the increased volume and complexity of input data. Therefore, we also propose shuffle and mask strategies to enhance the semantic association between multiple key values and candidate sets.

Considering the sequential recommendation task involving multiple key-value pairs, we first construct the following template to facilitate the conversion of the structured data into natural language format.

Key value data

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/47f3/47f3baf6-8324-4b79-a330-545724a4a818.png" style="width: 50%;"></div>
NLP data

# Figure 1: Example instructions where keys are in yellow background. We follow the classical format of LLM instructions, which includes ’Instruction’, ’Input’, and ’Output’.

Figure 1: Example instructions where keys are in yellow background. We follow the classical format of LLM instructions, which includes ’Instruction’, ’Input’, and ’Output’.

As shown in Figure 1, the raw data for RS is constructed in a key-value format. The proposed template can convert the raw data into text data that satisfies LLM input requirements. Concretely, we consider user keys (e.g. gender, occupation, age, historical interactions) and item keys (e.g. title, category, rating) when constructing the training data. For sequential recommendations, we follow the previous work [4, 5, 7, 16] to generate a randomized candidate set,

which contains one positive sample and several randomly chosen negative samples. Additionally, the first item of the output is the positive sample (the next item), while the others are all negative samples from the candidate set.

# 3.2 Adopting Shuffle Strategy

NLP data

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/58c2/58c244fc-8cf8-40dc-8901-9f52cfb6a0b0.png" style="width: 50%;"></div>
Figure 2: The proposed shuffle strategy. We shuffle key-value pairs or other lists in the instructions. The crossed arrows indicate shuffling the list, rather than exchanging specific items.

<div style="text-align: center;">Figure 2: The proposed shuffle strategy. We shuffle key-value pairs or other lists in the instructions. The crossed arrows indicate shuffling the list, rather than exchanging specific items.
</div>
Although the proposed template can alleviate the core challenge, it also increases the data volume and complexity of LLM’s input, which can lead to poor performance. To address this problem, we apply the data argument strategy to enhance the association between these keys and candidates. Firstly, this section presents the shuffle strategy, which is shown in Figure 2. Empirically, the order of key-value pairs may not influence the resulting output. From a human perspective, the key-value pairs [<𝐺𝑒𝑛𝑑𝑒𝑟>, <𝑂𝑐𝑐𝑢𝑝𝑎𝑡𝑖𝑜𝑛>] and [<𝑂𝑐𝑐𝑢𝑝𝑎𝑡𝑖𝑜𝑛>, <𝐺𝑒𝑛𝑑𝑒𝑟>] have identical semantic information and thus result in identical output. Motivated by this observation, we propose a shuffle strategy to randomly rearrange these key-value pairs or other information in list format, including the candidate list and output list. By adopting this strategy, we have the flexibility to decide on applied components. We can shuffle only a single component including key-value pairs, candidates set, and output set. Additionally, shuffling the combination of these components or both is also available. After data enhancement, both generated data and raw data are added to the training dataset. Given a finite number of training data, this strategy can well strengthen the attention of these components and alleviate the impact of pairs order on the results.

# 3.3 Adopting Mask Strategy

In addition to the shuffle strategy, we also propose a mask strategy. While the shuffle strategy aims to enhance the association between different components, it assumes that the order of these components influences the outcomes equally. However, different keys actually have different contributions to the outcomes, meaning that some keys may have higher importance weights then others. Empirically,

some prominent keys, not all keys in some situations, can also result in true outcomes, such as historical interactions, category of items, or others. Motivated by this observation, we propose the mask strategy that masks out part of key-value pairs to strengthen the association between some prominent keys and the results, owing to the high importance weights of these keys. This strategy is shown in Figure 3.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/70df/70df8578-a7ff-402d-a3ab-3193e69f904e.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: The proposed mask strategy aims to enhance the decision-making ability of prominent keys by masking out certain key-value pairs, as shown in the white background.
</div>
Concretely, we can manually select which keys are retained and which ones to mask out. Empirically, the maintained keys are more critical for the results. However, in this paper, due to the huge cost of training LLM, we choose the simplest way, which randomly masks out some keys while always maintaining historical interactions. Differing from the shuffle strategy, this strategy has a hyper-parameter to determine the number of masked keys, and the setting of this parameter should be customized according to specific applications and the training dataset. By adopting this strategy, we can strengthen the association between prominent keys and results, thereby improving the learning performance.

# 4 EXPERIMENTS

# 4.1 Setup

4.1.1 Datasets. Considering the specific setting, the MovieLens dataset best satisfies the multiple key-value format. Most of these values are in text format, rather than in encoded format. It is difficult to find other datasets that fulfill these conditions, as values of keys from other datasets are often in encoded format. We follow the same processing procedure from [7]. Concretely, we discard users and items with fewer than 5 related actions and use timestamps to determine the sequence order of actions. The keys for users are ’ userID ’, gender’, ’age’, ’occupation’, and ’zip code’. The keys for items are ’itemID’, ’title’, and ’category’. Additionally, we maintain the actions keys, including ’history watched movie title’, ’corresponding rating of these movies’, and ’corresponding category of these movies’. By adopting our methods, we can construct 6040 instructions, one for each user, during both the training and testing

processes. Moreover, we just randomly select 1000 test samples due to the huge cost of the evaluation.

4.1.2 Model. We utilize the popular LLaMA-7 𝐵 [21] as the backbone and fine-tune its open-source pre-trained model. LLaMA is known for its effectiveness and competitiveness compared with other open-source LLMs. It has been widely used in numerous applications and has achieved significant success in supervised fine-tuning LLM. We first download the pre-trained foundation model of LLaMA-7 𝐵. Then we prepare instruction data using our proposed strategies and fine-tune LLM on this data, which includes both raw and enhanced data.
4.1.3 Configuration. Regarding constructing instruction data, we discard ’id’, ’zip code’, and ’timestamp’ of users due to its encoded format. Due to the contextual limit of input tokens, we truncate the generated behavioral sequence with a maximum of 20 items, i.e., | 𝑆 𝑈 | = 20. We generated a candidate set by randomly selecting 9 negative samples and a positive sample (the next item) and utilize their titles to represent themselves. No extra keys exist for the candidate set.
4.1.4 Baselines. To the best of our knowledge, this is the first work that studies the incorporation of RS with LLM to complete sequential recommendation tasks on multiple key-value data. Therefore, it is challenging to choose baselines that utilize both RS and LLM to handle sequential recommendation tasks. Thus we select the challenging GPT-4 to compare with our approach.
4.1.5 Our method. Regarding our methods, we summarize various strategies as follows: 1) Multiple_KVs. This approach utilizes the proposed template. We term it InstructMK; 2) KVs_Shuffle. We shuffle the candidate list and the output list in the ’Input’ and ’Output’ components, respectively. We term these two methods InstructMK-SC and InstructMK-SO, respectively; 3)KVs_Mask. We mask parts of key-value pairs in different degrees. Concretely, we mask out 3,4 of all 7 key-value pairs and term them InstructMK-M1 and InstructMK-M2, respectively. We also mask out 3 and 4 keys at the same time and term this method InstructMK-M3. All methods generate an equal number of enhanced data, with a ratio of original to enhanced data of 1:4. Specifically, InstructMKM3 generates enhanced data with a ratio of 1:2 by masking out 3 keys, and the same applies to masking out 4 keys, resulting in a final ratio of 1:4 for InstructMK-M3. Additionally, all other combinations of these strategies (e.g. shuffle and mask) are feasible but need future verification in specific cases. In this work, we mainly want to provide methodological and directional value.

4.1.5 Our method. Regarding our methods, we summarize various strategies as follows: 1) Multiple_KVs. This approach utilizes the proposed template. We term it InstructMK; 2) KVs_Shuffle. We shuffle the candidate list and the output list in the ’Input’ and ’Output’ components, respectively. We term these two methods InstructMK-SC and InstructMK-SO, respectively; 3)KVs_Mask. We mask parts of key-value pairs in different degrees. Concretely, we mask out 3,4 of all 7 key-value pairs and term them InstructMK-M1 and InstructMK-M2, respectively. We also mask out 3 and 4 keys at the same time and term this method InstructMK-M3. All methods generate an equal number of enhanced data, with a ratio of original to enhanced data of 1:4. Specifically, InstructMKM3 generates enhanced data with a ratio of 1:2 by masking out 3 keys, and the same applies to masking out 4 keys, resulting in a final ratio of 1:4 for InstructMK-M3. Additionally, all other combinations of these strategies (e.g. shuffle and mask) are feasible but need future verification in specific cases. In this work, we mainly want to provide methodological and directional value.

# 2 Training and Evaluation

To complete training and evaluation, we follow the previous work [7] by splitting the historical sequence 𝑆 𝑈 into two parts:(1) the most recent action 𝑆 𝑈 | 𝑆 𝑈 | for testing; (2) the second most recent action
𝑆 𝑈 | 𝑆 𝑈 |− 1 for training. When preparing training data, the historical
interactions sequence contains action 𝑆 𝑈 | 𝑆 𝑈 |− 2 and is in time order.
We then chose the next item (𝑆 𝑈 | 𝑆 𝑈 |− 1), which also exists in the candidate set, as the label of the corresponding instance. Similarly, we utilize action 𝑆 𝑈 | 𝑆 𝑈 | for the testing process.

To evaluate recommendation performance, we adopt two common Top-N metrics: Hit Rate@K, and NDCG@K. In this paper, we set 𝐾 as 1, 3, and 5. Since HR@1 is equal to NDCG@1, we only report HR@1. As the output of LLM is generative and may result in out-of-scope results, we also propose an Error Rate metric to
evaluate the output of LLM. It can be defined as 𝐸𝑅 =
� | 𝑈 | 𝑗 𝐼 (𝐹 (𝑗)) | 𝑈 |, where 𝐼 (.) is a indicator function. For the 𝑗-index sample, if every title in the outputs of LLM is correct, 𝐼 (𝐹 (𝑗)) = 1, and 0 otherwise.

# 4.3 Results and Analysis

<div style="text-align: center;">Table 1: Results (HR @ K) on MovieLens.
</div>
Strategy
Method
HR@1
HR@3
HR@5
GPT
GPT-4
0.3080
0.4260
0.4820
Multiple_KVs
InstructMK
0.6460
0.7670
0.8270
KVs_Shuffle
InstructMK-SC
0.7130
0.8000
0.8630
InstructMK-SO
0.7300
0.8070
0.8630
KVs_Mask
InstructMK-M1
0.6980
0.7690
0.8310
InstructMK-M2
0.6790
0.7710
0.8360
InstructMK-M3
0.7020
0.7830
0.8360
<div style="text-align: center;">Table 2: Results (NDCG@K and ER) on MovieLens.
</div>
Strategy
Method
NG@3
NG@5
ER
GPT
GPT-4
0.3777
0.4007
0.020
Multiple_KVs
InstructMK
0.7165
0.7411
0.003
KVs_Shuffle
InstructMK-SC
0.7630
0.7886
0.006
InstructMK-SO
0.7733
0.7962
0.000
KVs_Mask
InstructMK-M1
0.7387
0.7649
0.007
InstructMK-M2
0.7323
0.7591
0.009
InstructMK-M3
0.7478
0.7697
0.011
The experimental results are shown in the Tabel 1 and 2. We can observe that: 1) The shuffle and mask strategies are all effective and can improve the results compared to the baseline GPT-4; 2) Among our proposed method, the shuffle strategy on the output set achieves the best performance; 3) Since we conducted experiments on a single dataset with a limit of 6 keys, the value of the mask strategy may be affected. However, our experiments show that this strategy is effective and results in significant improvements; 4) When it comes to the specific mask strategy, masking too many keys is not encouraged for model training. In our experiments, the combination of different mask degrees leads to better performance; 5) We find that our approach contributes more improvements to HR@1 and HR@3 for HR@K, indicating that our approach can increase the likelihood of the true next-item being ranked first; 6) Regarding baseline, GPT-4 lacks domain knowledge and cannot understand multiple key-value data well, resulting in poor performance.

# 4 Limitations and Future

Due to the input limit of LLM and the high training cost, we followed the approach of previous works and generated a behavioral sequence with a maximum length of 20 and a candidate set with a length of 10. Additionally, it was challenging to find suitable datasets that satisfy our setting, where the dataset should contain multiple key-value pairs and its values should be in natural language format. Given this unique situation and the significant cost of training large models mentioned above, we chose to use only the single MovieLens dataset. However, in the future, we plan to explore more suitable datasets and larger candidate sets.

# 5 CONCLUSION

This paper focuses on sequential recommendation scenarios based on multiple key-value data and incorporates RS with LLM. Specifically, we propose a conversion template to transform raw data into text data. Moreover, to address the challenge of increasing input number and complexity, we introduce two flexible and general data augmentation strategies: random shuffling of sets and masking of key-value pairs. Additionally, we propose the error rate as a metric to quantify the error rate of the model output.

# REFERENCES

[1] Hai Dang, Lukas Mecke, Florian Lehmann, Sven Goller, and Daniel Buschek. 2022. How to prompt? Opportunities and challenges of zero-and few-shot learning for human-AI interaction in creative applications of generative models. arXiv preprint arXiv:2209.01390 (2022).
[2] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In Proceedings of the 16th ACM Conference on Recommender Systems. 299–315.
[3] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. arXiv preprint arXiv:1703.04247 (2017).
[4] Ruining He, Wang-Cheng Kang, and Julian McAuley. 2017. Translation-based recommendation. In Proceedings of the eleventh ACM conference on recommender systems. 161–169.
[5] Ruining He and Julian McAuley. 2016. Fusing similarity models with markov chains for sparse sequential recommendation. In 2016 IEEE 16th international conference on data mining (ICDM). IEEE, 191–200.
[6] Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, and Wayne Xin Zhao. 2023. Large language models are zero-shot rankers for recommender systems. arXiv preprint arXiv:2305.08845 (2023).
[7]  Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM). IEEE, 197–206.
[8] Dominik Kowald, Markus Schedl, and Elisabeth Lex. 2020. The unfairness of popularity bias in music recommendation: A reproducibility study. In Advances in Information Retrieval: 42nd European Conference on IR Research, ECIR 2020, Lisbon, Portugal, April 14–17, 2020, Proceedings, Part II 42. Springer, 35–42.
[9] Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. 2023. LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models. arXiv preprint arXiv:2305.13655 (2023).
[10] Jianghao Lin, Xinyi Dai, Yunjia Xi, Weiwen Liu, Bo Chen, Xiangyang Li, Chenxu Zhu, Huifeng Guo, Yong Yu, Ruiming Tang, et al. 2023. How Can Recommender Systems Benefit from Large Language Models: A Survey. arXiv preprint arXiv:2306.05817 (2023).
[11] Guoguang Liu. 2022. An ecommerce recommendation algorithm based on link prediction. Alexandria Engineering Journal 61, 1 (2022), 905–910.
[12] Junling Liu, Chao Liu, Renjie Lv, Kang Zhou, and Yan Zhang. 2023. Is chatgpt a good recommender? a preliminary study. arXiv preprint arXiv:2304.10149 (2023).
[13] Darshita Mittal, Sanyukta Shandilya, Dhruv Khirwar, and Archana Bhise. 2020. Smart billing using content-based recommender systems based on fingerprint. In ICT Analysis and Applications: Proceedings of ICT4SD 2019, Volume 2. Springer, 85–93.
[14] Yilena Pérez-Almaguer, Raciel Yera, Ahmad A Alzahrani, and Luis Martínez. 2021. Content-based group recommender systems: A general taxonomy and further improvements. Expert Systems with Applications 184 (2021), 115444.

[15] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog 1, 8 (2019), 9.
[16]  Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized markov chains for next-basket recommendation. In Proceedings of the 19th international conference on World wide web. 811–820.
[17] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100 (2022).
[18] Jagendra Singh, Mohammad Sajid, Chandra Shekhar Yadav, Shashank Sheshar Singh, and Manthan Saini. 2022. A Novel Deep Neural-based Music Recommendation Method considering User and Song Data. In 2022 6th International Conference on Trends in Electronics and Informatics (ICOEI). IEEE, 1–7.
[19] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management. 1441–1450.
[20] Zhu Sun, Jie Yang, Kaidong Feng, Hui Fang, Xinghua Qu, and Yew Soon Ong. 2022. Revisiting Bundle Recommendation: Datasets, Tasks, Challenges and Opportunities for Intent-aware Product Bundling. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2900–2911.
[21] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).
[22]  Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).
[23] Maksims Volkovs, Guang Wei Yu, and Tomi Poutanen. 2017. Content-based neighbor models for cold start in recommender systems. In Proceedings of the Recommender Systems Challenge 2017. 1–6.
[24] Yueqi Xie, Jingqi Gao, Peilin Zhou, Qichen Ye, Yining Hua, Jaeboum Kim, Fangzhao Wu, and Sunghun Kim. 2023. Rethinking Multi-Interest Learning for Candidate Matching in Recommender Systems. arXiv preprint arXiv:2302.14532 (2023).
[25] Yueqi Xie, Peilin Zhou, and Sunghun Kim. 2022. Decoupled side information fusion for sequential recommendation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1611–1621.
[26] Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. 2023. Recommendation as instruction following: A large language model empowered recommendation approach. arXiv preprint arXiv:2305.07001 (2023).
[27] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. 2023. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923 (2023).

