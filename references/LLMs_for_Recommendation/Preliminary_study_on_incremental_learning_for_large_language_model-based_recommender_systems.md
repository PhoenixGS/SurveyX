# liminary Study on Incremental Learning for Large Languag Model-based Recommender Systems

Yang Zhang ∗
National University of Singapore Singapore, Singapore zyang1580@gmail.com
Tianhao Shi University of Science and Technology of China Hefei, China sth@mail.ustc.edu.cn
University zjan

Yang Zhang ∗
National University of Singapore Singapore, Singapore zyang1580@gmail.com
hao Shi nce and Technology China i, China l.ustc.edu.cn
Zhijian Xu University of Science and Technology of China Hefei, China zjane@mail.ustc.edu.cn

Chong Chen Huawei Cloud BU Shenzhen, China chenchong55@huawei.com

Fuli Feng University of Science and Technology of China Hefei, China fulifeng93@gmail.com
Xiangnan He University of Science and Technology of China Hefei, China xiangnanhe@gmail.com

Qi Tian Huawei Cloud BU Shenzhen, China tian.qi1@huawei.com

# Abstract

Adapting Large Language Models for Recommendation (LLM4Rec) has shown promising results. However, the challenges of deploying LLM4Rec in real-world scenarios remain largely unexplored. In particular, recommender models need incremental adaptation to evolving user preferences, while the suitability of traditional incremental learning methods within LLM4Rec remains ambiguous due to the unique characteristics of Large Language Models (LLMs). In this study, we empirically evaluate two commonly employed incremental learning strategies (full retraining and fine-tuning) for LLM4Rec. Surprisingly, neither approach shows significant improvements in the performance of LLM4Rec. Instead of dismissing the role of incremental learning, we attribute the lack of anticipated performance enhancement to a mismatch between the LLM4Rec architecture and incremental learning: LLM4Rec employs a single adaptation module for learning recommendations, limiting its ability to simultaneously capture long-term and short-term user preferences in the incremental learning context. To test this speculation, we introduce a Long- and Short-term Adaptation-aware Tuning (LSAT) framework for incremental learning in LLM4Rec. Unlike the single adaptation module approach, LSAT utilizes two distinct adaptation modules to independently learn long-term and

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. CIKM ’24, October 21–25, 2024, Boise, ID, USA © 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0436-9/24/10 https://doi.org/10.1145/3627673.3679922

Zhijian Xu

Xiangnan He University of Science and Technology of China Hefei, China xiangnanhe@gmail.com

short-term user preferences. Empirical results verify that LSAT enhances performance, thereby validating our speculation. We release our code at: https://github.com/TianhaoShi2001/LSAT.

# CCS Concepts
• Information systems → Recommender systems.

# • Information systems → Recommender systems.

Keywords
Large Language Models, Model Retraining, Incremental Lear

Large Language Models, Model Retraining, Incremental Lear

ACM Reference Format: Tianhao Shi, Yang Zhang, Zhijian Xu, Chong Chen, Fuli Feng, Xiangnan He, and Qi Tian. 2024. Preliminary Study on Incremental Learning for Large Language Model-based Recommender Systems. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM ’24), October 21–25, 2024, Boise, ID, USA. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3627673.3679922

# 1 Introduction

The emergence of Large Language Models [11], equipped with extraordinary capabilities in text comprehension and generation, has been successfully applied in various domains like Robotics [9, 30] and Computer Vision [4, 38]. Inspired by this success, there is growing interest in using LLMs for recommendations in both academia [1, 10, 37] and industry [22]. Among current efforts, tuning LLMs with recommendation-specific data using LoRA [15] (a well-known efficient fine-tuning method) has yielded promising results [2, 3, 23, 43, 45], underscoring the substantial potential of LLM4Rec in real-world applications. However, the challenges associated with the practical deployment of LLM4Rec remain explored, particularly considering the unique characteristics of LLMs. When deploying a recommender system in real-world scenarios, one of the primary challenges is ensuring the recommender models can adapt incrementally to evolving user preferences and

environments [20, 42, 44]. This adaptation is crucial as user feedback streamingly flows in, requiring the recommender model to be incrementally updated with the latest data to achieve timely personalization. For traditional recommendation models, the critical role of incremental learning and associated challenges have been extensively researched [20, 42, 44]. However, when it comes to LLM4Rec, the issues related to incremental learning lack adequate attention. The unique characteristics of LLM4Rec, such as its massive parameters and its high tuning cost [24], may introduce new challenges or insights that require thorough examination.
In this study, we first empirically examine how incremental learning impacts the performance of LLM4Rec. Considering the broad adoption of LoRA in developing LLM4Rec models [2, 3, 23, 45] , and acknowledging LoRA’s efficiency and effectiveness [8], our research focuses on this specific type of LLM4Rec. We examine two commonly used incremental learning strategies: 1) full retraining [20], which involves periodic retraining using complete historical data and new data, and 2) fine-tuning [28, 35], which updates the model based solely on new data. Based on our empirical results, we find that both full retraining and fine-tuning have a minimal impact on the performance of LLM4Rec. These results emphasize that LLM4Rec exhibits good generalization capabilities even under delayed updates; however, they also suggest that incremental learning might not lead to performance improvements for LLM4Rec.
Based on our empirical results, executing incremental learning appears to be unnecessary for LLM4Rec. This is somewhat surprising, as user preferences do change over time, and a recommender system should adapt to these changes [20, 39, 44]. We speculate that the lack of anticipated performance improvements may be attributed to a mismatch between the LoRA architecture and incremental learning: LoRA avoids training the entire model and instead tunes a low-rank adaptation module [8] with recommendation data, while a single LoRA module may have the inability to simultaneously emphasize long-term and short-term user preferences under incremental learning. Specifically, for full retraining, the LoRA module might emphasize long-term preferences but overlook short-term ones, given the substantial volume of historical data compared to the new data [16]. For fine-tuning, the LoRA module may forget previous knowledge due to catastrophic forgetting [25], leading to a decline in performance.
To test our speculation, we develop a modified updating method called Long- and Short-term Adaptation-aware Tuning (LSAT). This method utilizes two LoRA modules to separately learn long-term and short-term user preferences and then integrates them to merge the different types of preferences. During each update, the shortterm LoRA module is temporarily retrained using solely new data to focus on the latest evolving preferences. In contrast, given the robust generalization capabilities exhibited by the long-term LoRA even with delayed updates, it remains fixed once it has been sufficiently trained or retrained at a relatively gradual frequency to conserve training costs. We conduct a comparison between LSAT, full retraining, and fine-tuning methods. Extensive results demonstrate that LSAT brings performance enhancements, confirming the validity of our speculation. Nevertheless, at present, LSAT only explores incremental learning from the perspective of LoRA capacity. To comprehensively understand and address the issue, further investigation in various directions is still necessary.

The main contributions are summarized as follows:
• New Problem: This work marks the inaugural investigation into incremental learning for LLM4Rec, furnishing practical insights for the real-world deployment of LLM4Rec.
• New Finding: Our empirical results underscore that the common incremental learning methods (full retraining and fine-tuning) do not clearly enhance the performance of LoRA-based LLM4Rec.
•  Proposal: We propose that using separate LoRA modules to capture long-term and short-term preferences can enhance the performance of LLM4Rec in incremental learning, offering valuable insights from the perspective of the capacity of the LoRA module.

The main contributions are summarized as follows:
• New Problem: This work marks the inaugural investigation into incremental learning for LLM4Rec, furnishing practical insights for the real-world deployment of LLM4Rec.
• New Finding: Our empirical results underscore that the common incremental learning methods (full retraining and fine-tuning) do not clearly enhance the performance of LoRA-based LLM4Rec.
•  Proposal: We propose that using separate LoRA modules to capture long-term and short-term preferences can enhance the performance of LLM4Rec in incremental learning, offering valuable insights from the perspective of the capacity of the LoRA module.

# 2 RELATED WORKS

LLM-based recommender. Adapting LLMs as a recommender has gained substantial attention. In-context learning enables LLMs to provide recommendations without explicit training [5, 34]. Nevertheless, due to the lack of recommendation-specific knowledge during pre-training, applying instruction tuning [27] with recommendation data helps LLMs achieve much better recommendation performance [2, 3]. Among tuning methods, InstructRec [41] fine-tunes all LLM’s parameters, while most approaches employ parameter-efficient fine-tuning (PEFT) [8] to avoid adjusting the extensive parameters of LLMs [2, 3, 23, 45]. Within PEFT, LoRA is adopted by the majority of LLM4Rec models [2, 3, 23, 45] due to its good convergence and accuracy [8]. Considering the broad adoption of LoRA in developing LLM4Rec models, this work explores the issues of incremental learning in LoRA-based LLM4Rec. Incremental learning in recommendation.  Incremental learning is crucial for recommender models due to new users/items and changing user preferences[20, 29]. Representative methods for incremental updates include 1) full retraining [20], which retrains models with both new and historical data, achieving high accuracy with extensive training cost; 2) fine-tuning [28, 35], which updates models exclusively with the latest data, offering efficiency but facing potential forgetting; 3) sample-based methods [7, 36], which update models with new data and a sampled subset of historical data, where the sampled data is expected to retain long-term preference signals; and 4) meta-learning-based methods [39, 44], utilizing meta-learning to optimize the model for better future serving. Unlike prior works focusing on traditional recommender models, this work explores incremental learning within LLM4Rec.

# 3 Preliminaries

In this study, we explore incremental learning in LLM4Rec. Our investigation is centered around a representative LLM4Rec model known as TALLRec [3], chosen due to the widespread adoption of its tuning paradigm [2, 45]. Next, we briefly introduce TALLRec and incremental learning in the recommendation. • TALLRec.  To align LLMs with recommendations, TALLRec utilizes instruction tuning [27]. This involves organizing historical interaction data into textual instructions and responses and then fine-tuning LLMs using this structured data to improve recommendation performance. Notably, TALLRec adopts LoRA [15] for efficient tuning, which freezes pre-trained LLM parameters and integrates lightweight trainable matrices. Specifically, LoRA introduces a pair of rank-decomposed weight matrices to each pre-trained

ary Study on Incremental Learning for Large Language Model-based Recommend

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/05a7/05a75d4b-8853-48a7-892e-03edf7cc60be.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Incremental learning process in recommendation
</div>
weight matrix 𝑊 ∈R 𝑑 × 𝑘, formally 𝑊 + 𝐴𝐵, where 𝐴 ∈R 𝑑 × 𝑟 and 𝐵 ∈R 𝑟 × 𝑘 are added learnable LoRA matrices (𝑟 ≪ min (𝑑,𝑘)). • Incremental learning in recommendation. To signify the incremental learning process, following [20, 44], we represent the data stream as {D 1, D 2, . . . , D 𝑡, . . .}, where D 𝑡 represents the collected data at time period 𝑡. The length of a period may vary (e.g., daily, weekly) based on system requirements. At each period 𝑡, the model has access to new data D 𝑡 and all previous data for updating, and the updated model needs to serve for the near future D 𝑡 + 1. This paper uses two representative incremental learning strategies: full retraining [20], updating the model with both new data D 𝑡 and entire historical data {D 1, D 2, . . . , D 𝑡}; and fine-tuning [28, 35], utilizing only the latest data D 𝑡 for the update. Figure 1 illustrates the process of incremental learning in recommender systems.

# 4 Empirical Explorations

In this section, we conduct experiments to explore the impact of the commonly employed incremental learning methods on TALLRec.

# 4.1 Experiemental Settings

Datasets. We conduct experiments on two representative datasets: MovieLens-1M (ML-1M) [13], which is a movie rating dataset collected by GroupLens Research, and Amazon-Book [26], which includes user reviews of books in Amazon. In ML-1M and AmazonBook, ratings range from 1 to 5. Following [3, 45], interactions with ratings ≥ 4 are positive; others are negative. To pre-process the data, we adopt the approach from TALLRec [3], converting ratings to binary labels and excluding users with less than 10 interactions. To assess incremental learning’s impact, we divided the data chronologically based on interaction timestamps. For ML-1M, we use data from Dec. 2000 to Feb. 2003, creating 20 periods of 10,000 samples. Similarly, for Amazon-Book, we keep the data from Mar. 2014 to May. 2018, and sampled 20% of users, resulting in 328,168 samples, which were then divided into 20 two-month periods. For each period, we train models on the initial 90% of the data and validate them on the remaining 10%. Table 1 presents the dataset statistics. Models. Due to the diversity of incremental learning algorithms, we select the two most representative incremental learning strategies, full retraining, and fine-tuning for updating TALLRec. We also evaluate how these two update methods affect five traditional recommendation models: 1) MF [19], which is a latent factor-based collaborative filtering, 2)DeepMusic [33], which is a content-based recommendation model, 3) GRU4Rec [14], which is an RNN-based sequential recommender, 4) Caser [31], which uses CNN to model sequence patterns, and 5) SASRec [17], which employs a self-attention

<div style="text-align: center;">Table 1: Statistics of the evaluation datasets.
</div>
Dataset
# Users # Items # Instances Density
ML-1M
1,813
3,503
200,000
3.1491%
Amazon-Book
28,427
12,680
328,168
0.0869%
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f07d/f07d9037-28a8-4d2e-bbf6-27a979f7ccea.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Performance of TALLRec, BookGPT and traditional models obtained at different update periods on D 20.
</div>
mechanism to grasp sequential patterns. Additionally, we evaluate BookGPT [21], an in-context learning-based method, which is tuning-free and always incorporates the latest data into prompts. Evaluation metrics and hyper-parameters.  Following TALLRec [3], we use AUC [12] to evaluate recommendation performance. TALLRec is deployed based on LLaMA-7B [32], and BookGPT on GPT-3.5-turbo, with settings aligned with the original papers. We optimize all traditional models using the Adam optimizer [18] with the MSE loss, employing a learning rate of 1e-3, batch size of 256, embedding size of 64, and weight decay of 1e-5 (tuned results).

# 2 Incremental Learning’s Impact on LLM4Rec

# 4.2 Incremental Learning’s Impact on

Overall results. To evaluate the effect of incremental learning, the model undergoes continuous updates until the 19th period. Then, we evaluate the performance of the model obtained at each update period on the data D 20 of the 20th period, and plot the performance curve against the update periods in Figure 2, where we can find: •  For traditional models, timely updates could enhance their performance, particularly with full retraining. Fine-tuning is usually less effective than full retraining, and its effectiveness may diminish over time, possibly due to the forgetting issue.
• TALLRec significantly outperforms BookGPT, suggesting that while in-context learning can utilize the latest data without additional training, its performance is limited.
• Unlike traditional models, the performance of TALLRec remains relatively unaffected by both full retraining and fine-tuning. This underscores its ability to excel in generalizing to new data. However, it also indicates that timely incremental learning does not enhance LLM4Rec’s performance. Our initial findings suggest that incremental learning has a limited impact on the performance of LLM4Rec. Subsequently, we conduct a more detailed analysis to explore the effects of incremental learning on LLM4Rec, focusing on two key aspects as highlighted

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4198/4198ecc5-c313-4167-b515-f0b20ba7f6d9.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Performance comparison between TALLRec and baselines on warm items and cold items. All models are trained on D 1 −D 15 and tested on D 16 −D 20.
</div>
in [20]: 1) timely incorporation of new items and users, and 2) adaptation to changing user preferences. Cold-start items evaluation. In examining the first aspect, we compare the performance of TALLRec and traditional models on warm and cold items (trained on D 1– D 15, tested on D 16– D 20) in Figure 3. From the figure, we can find collaborative filtering methods exhibit near-random guessing (AUC=0.5) for cold items, indicating they would suffer performance deterioration without updates due to an increased number of cold items [20]. In contrast, TALLRec’s proficiency in general language understanding enables accurate recommendations for cold items. Hence, from the perspective of cold items, incremental learning has a smaller impact on LLM4Recs compared to traditional models. Dynamic preference on warm items. We then explore whether incremental learning improves the performance of LLM4Rec by adapting to the latest user preferences. Toward this, we filter cold items (items do not appear in D 1– D 10 but appear in D 20) and evaluate the performance of warm items on D 20 under different periods in Figure 4. We find that incremental learning always improves the performance of traditional models on warm items, except for fine-tuning on ML-1M due to forgetting issues. However, both full retraining and fine-tuning cannot enhance TALLRec’s performance on warm items. We suggest this could be due to the inability of a single LoRA to capture both long-term and short-term user preferences simultaneously. Full retraining may focus more on long-term preferences due to the larger quantity of historical data [16]. Fine-tuning might prioritize short-term preferences in new data while forgetting previous knowledge [25]. Consequently, both full-retraining and fine-tuning fail to achieve performance improvements by adapting to the latest preferences.

# 5 LSAT

We have observed that both full retraining and fine-tuning do not effectively improve LLM4Rec’s performance. We posit that a single LoRA module may struggle to simultaneously capture both long-term and short-term user preferences. This insight draws inspiration from advancements in leveraging multiple LoRA modules to handle distinct tasks or domain knowledge [6, 40]. Considering the divergence between long-term and short-term user preferences, it may be necessary to employ separate LoRA modules to capture them individually. To validate this speculation, we develop a new method called LSAT, employing two dedicated LoRA modules — one for capturing long-term user preferences and another for capturing short-term user preferences. During each update, the short-term

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/49b7/49b77513-7146-4dc0-949f-4124ccc1f6f5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Performance of TALLRec and traditional models obtained at different update periods for warm items on D 20.
</div>
# Figure 4: Performance of TALLRec and traditional models obtained at different update periods for warm items on D 20.

LoRA module is temporarily introduced and trained on new data, while the long-term LoRA module remains fixed once trained on sufficient previous data. In the inference phase, the long-term LoRA module collaborates with the current short-term LoRA module to provide personalized recommendations. Next, we elaborate on the building of the two LoRA modules and the details of the inference: Short-term LoRA.  This LoRA module targets short-term preferences. Toward this, at each period 𝑡, we train a new module with parameters Θ 𝑡 using the newly collected data D 𝑡 as:

# min Θ 𝑡 𝐿 (D 𝑡; Φ, Θ 𝑡),

(1)

where Φ represents the frozen pre-trained parameters of the LLM, and 𝐿 (D 𝑡; Φ, Θ 𝑡) is the recommendation loss on D 𝑡. Notably, this new LoRA training approach is trained from scratch instead of being fine-tuned from the previous period, as fine-tuning has shown relatively poor performance in adapting to new preferences. Long-term LoRA. This LoRA module aims to capture aggregated long-term preferences by fitting sufficient historical data. To achieve this, we train the long-term LoRA with ample historical data, denoted as H = {D 1, D 2, . . . , D 𝑚} as follows:

# min Θ ℎ 𝐿 (H; Φ, Θ ℎ),

(2)

where Θ ℎ denotes long-term LoRA parameters. Once long-term LoRA is sufficiently trained after the 𝑚-th period, Θ ℎ can be updated at a slower pace, making LSAT training costs similar to fine-tuning. Before the 𝑚-th period, retraining Θ ℎ is needed. Inference. During inference, we explore two methods to merge long- and short-term preferences from two LoRA modules: 1) Output ensemble: This approach involves directly averaging the predictions with two LoRA modules. For a given sample 𝑥 at the 𝑡 +1-th period, the final prediction is formulated as follows:

# 𝛼𝑓 (𝑥; Φ, Θ ℎ) + (1 − 𝛼) 𝑓 (𝑥; Φ, Θ 𝑡),

(3)

where 𝛼 is a hyper-parameter chosen on the validation set, and 𝑓 (𝑥; Φ, Θ 𝑡) and 𝑓 (𝑥; Φ, Θ ℎ) are predictions of LLM4Rec using the 𝑡-th period short-term LoRA and th long-term LoRA, respectively.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3cf6/3cf6e168-03a2-495e-80f2-aec2a5da5098.png" style="width: 50%;"></div>
Figure 5: Performance comparison of full retraining, finetuning, and LSAT. All models are updated promptly with newly collected data D 𝑡 and tested on D 𝑡 + 1. LSAT (10) means it utilizes H = {D 1, D 2, . . . , D 10} to train the long-term LoRA.
2) LoRA fusion:  As the output ensemble involves two LLM inferences, we explore merging the two LoRA modules for a single-pass inference, adopting a common fusion strategy—task arithmetic [40]. Formally, task arithmetic fuses the parameters of the long-term LoRA (Θ ℎ) and the short-term LoRA at the 𝑡-th period (Θ 𝑡), and generates the final prediction for a sample 𝑥 as:

𝑓 (𝑥; Φ, 𝜆 Θ ℎ + (1 − 𝜆) Θ 𝑡),

where 𝜆 is a hyper-parameter chosen on the validation set.

# 6 Experiments

We conduct experiments to verify the effectiveness of our proposal. Experimental settings. We compare the performance of LSAT with full retraining and fine-tuning on ML-1M and Amazon-Book. Details about datasets and TALLRec are presented in Section 4.1. To assess which approach yields superior results after updates, following [44], the model is updated with D 𝑡 and evaluated on D 𝑡 + 1. For LSAT, the long-term LoRA module relies on a historical dataset H = {D 1, D 2, . . . , D 𝑚} (Equation (2)), with 𝑚 set to 10 (LSAT (10)). We study two methods for model merging defined in the Inference part of Section 5: ensemble (LSAT-EN), and task arithmetic (LSAT-TA). The coefficient of LSAT 𝛼 (Equation 3) and 𝜆 (Equation 4) are searched within {0, 0. 1, . . . , 1}. Overall results.  Figure 5 illustrates the overall performance comparison between full retraining, fine-tuning, and LSAT. From the figure, we can find that LSAT-EN and LSAT-TA outperform full retraining and fine-tuning on two datasets, emphasizing the effectiveness of using separate LoRAs for long-term and short-term interests. This supports our initial hypothesis that employing two adapters leads to improved performance in modeling both longterm and short-term interests. Nevertheless, LSAT-TA’s relatively modest improvement suggests merging adapter parameters into a single adapter may be less effective, implying the need for exploring a parameter-level LoRA merging method tailored for incremental learning in recommendation systems. Analyses. We then study LSAT’s two integral components: the short-term LoRA and the long-term LoRA, comparing their performance with LSAT-EN in Table 2, where we can find: • Short-term LoRA surpasses fine-tuning, as fine-tuning may suffer catastrophic forgetting [25, 42], where newly acquired knowledge conflicts with old knowledge, resulting in performance decline. This suggests the rationale of using a new LoRA to learn the latest preferences, rather than fine-tuning from the previous stage.

Table 2: Average AUC across D 11 −D 20. All models are updated promptly with newly collected data D 𝑡 and tested on D 𝑡 + 1. LSAT-EN (full) means its long-term LoRA is retrained during each period with all historical data.

ML-1M Amazon-Book
Full Retraining
0.7650
0.7780
Fine-tuning
0.7594
0.7696
Short-term LoRA
0.7638
0.7806
Long-term LoRA
0.7617
0.7789
LSAT-TA (10)
0.7665
0.7813
LSAT-EN (full)
0.7691
0.7822
LSAT-EN (10)
0.7720
0.7830
• LSAT-EN (full), consistently updating the long-term LoRA, does not bring additional improvements compared to LSAT-EN (10), using a fixed long-term LoRA. These results suggest that the long-term LoRA can be updated at a slower pace once adequately trained, as long-term preferences tend to remain relatively stable.
• Using only a single short-term LoRA or long-term LoRA leads to a performance decrease, highlighting the importance of merging both the long-term and short-term LoRAs for LSAT.

# 7 Conclusion

This study studies the impact of incremental learning on LLM4Rec. Empirical results and analysis reveal that both full retraining and fine-tuning fail to deliver the anticipated performance improvement for LLM4Rec. We posit that a singular LoRA may encounter challenges in simultaneously capturing long-term and short-term preferences. To validate our hypothesis, we introduce LSAT and conduct experimental validation. Nevertheless, our current research is limited to the TALLRec backbone, which uses solely textual information (title) for recommendations. Future investigations will extend to other backbones. Additionally, as LSAT studies incremental learning methods solely from the perspective of LoRA capacity, we plan to explore other dimensions for more effective methods.

# Acknowledgments

This work is supported by the National Key Research and Development Program of China (2022YFB3104701), the National Natural Science Foundation of China (62272437), and the CCCD Key Lab of Ministry of Culture and Tourism.

# References

[1] Qingyao Ai, Ting Bai, Zhao Cao, Yi Chang, Jiawei Chen, Zhumin Chen, Zhiyong Cheng, Shoubin Dong, Zhicheng Dou, Fuli Feng, et al. 2023. Information Retrieval Meets Large Language Models: A Strategic Report from Chinese IR Community. AI Open 4 (2023), 80–90.
[2] Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yancheng Luo, Fuli Feng, Xiangnan He, and Qi Tian. 2023. A Bi-step Grounding Paradigm for Large Language Models in Recommendation Systems. arXiv preprint arXiv:2308.08434 (2023).
[3] Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems. 1007–1014.
[4] William Berrios, Gautam Mittal, Tristan Thrush, Douwe Kiela, and Amanpreet Singh. 2023. Towards Language Models that can See: Computer Vision through the Lens of Natural Language. arXiv preprint arXiv:2306.16410 (2023).
[5] Zheng Chen. 2023. PALR: Personalization Aware LLMs for Recommendation. arXiv preprint arXiv:2305.07622 (2023).

[6] Alexandra Chronopoulou, Matthew E Peters, Alexander Fraser, and Jesse Dodge. 2023. Adaptersoup: Weight Averaging to Improve Generalization of Pretrained Language Models. arXiv preprint arXiv:2302.07027 (2023).
[7] Ernesto Diaz-Aviles, Lucas Drumond, Lars Schmidt-Thieme, and Wolfgang Nejdl. 2012. Real-time Top-n Recommendation in Social Streams. In Proceedings of the sixth ACM Conference on Recommender Systems. 59–66.
[8] Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, et al.  2023. Parameterefficient Fine-tuning of Large-scale Pre-trained Language Models. Nature Machine Intelligence 5, 3 (2023), 220–235.
[9]  Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. 2023. Palm-e: An Embodied Multimodal Language Model. arXiv preprint arXiv:2303.03378 (2023).
[10] Wenqi Fan, Zihuai Zhao, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Jiliang Tang, and Qing Li. 2023. Recommender Systems in the Era of Large Language Models. arXiv preprint arXiv:2307.02046 (2023).
[11] Muhammad Usman Hadi, R Qureshi, A Shah, M Irfan, A Zafar, MB Shaikh, N Akhtar, J Wu, and S Mirjalili. 2023. A Survey on Large Language Models: Applications, Challenges, Limitations, and Practical Usage. TechRxiv (2023).
[12] James A Hanley and Barbara J McNeil. 1982. The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve. Radiology 143, 1 (1982), 29–36.
[13] F Maxwell Harper and Joseph A Konstan. 2015. The Movielens Datasets: History and Context. Acm Transactions on Interactive Intelligent Systems 5, 4 (2015), 1–19.
[14] Balazs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. In 4th International Conference on Learning Representations, ICLR 2016.
[15] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.
[16] Joel Jang, Seonghyeon Ye, Sohee Yang, Joongbo Shin, Janghoon Han, KIM Gyeonghun, Stanley Jungkyu Choi, and Minjoon Seo. 2021. Towards Continual Knowledge Learning of Language Models. In International Conference on Learning Representations.
[17]  Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive Sequential Recommendation. In 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 197–206.
[18]  Diederik P Kingma and Jimmy Ba. 2014. Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980 (2014).
[19]  Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix Factorization Techniques for Recommender Systems. Computer 42, 8 (2009), 30–37.
[20] Hyunsung Lee, Sungwook Yoo, Dongjun Lee, and Jaekwang Kim. 2023. How Important is Periodic Model Update in Recommender System?. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2661–2668.
[21] Zhiyu Li, Yanfang Chen, Xuan Zhang, and Xun Liang. 2023. BookGPT: A General Framework for Book Recommendation Empowered by Large Language Model. Electronics 12, 22 (2023), 4654.
[22] Jianghao Lin, Xinyi Dai, Yunjia Xi, Weiwen Liu, Bo Chen, Xiangyang Li, Chenxu Zhu, Huifeng Guo, Yong Yu, Ruiming Tang, et al. 2023. How Can Recommender Systems Benefit from Large Language Models: A Survey. arXiv preprint arXiv:2306.05817 (2023).
[23] Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan, Ruiming Tang, Yong Yu, and Weinan Zhang. 2023. ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation. arXiv preprint arXiv:2308.11131 (2023).
[24] Junling Liu, Chao Liu, Peilin Zhou, Qichen Ye, Dading Chong, Kang Zhou, Yueqi Xie, Yuwei Cao, Shoujin Wang, Chenyu You, et al.  2023. LLMRec: Benchmarking Large Language Models on Recommendation Task. arXiv preprint arXiv:2308.12241 (2023).
[25] Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. 2023. An Empirical Study of Catastrophic Forgetting in Large Language Models during Continual Fine-tuning. arXiv preprint arXiv:2308.08747 (2023).
[26] Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying Recommendations Using Distantly-labeled Reviews and Fine-grained Aspects. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 188–197.
[27] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training Language Models to Follow Instructions with Human Feedback.  Advances in Neural Information Processing Systems 35 (2022), 27730–27744.
[28] Steffen Rendle and Lars Schmidt-Thieme. 2008. Online-updating Regularized Kernel Matrix Factorization Models for Large-scale Recommender systems. In Proceedings of the 2008 ACM conference on Recommender systems. 251–258.

[29] Chijun Sima, Yao Fu, Man-Kit Sit, Liyi Guo, Xuri Gong, Feng Lin, Junyu Wu, Yongsheng Li, Haidong Rong, Pierre-Louis Aublin, et al. 2022. Ekko: A {LargeScale} Deep Learning Recommender System with {Low-Latency} Model Update. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). 821–839.
[30] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. 2023. Progprompt: Generating Situated Robot Task Plans Using Large Language Models. In 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 11523–11530.
[31] Jiaxi Tang and Ke Wang. 2018. Personalized Top-n Sequential Recommendation via Convolutional Sequence Embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. 565–573.
[32] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. LLaMA: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2302.13971 (2023).
[33] Aaron Van den Oord, Sander Dieleman, and Benjamin Schrauwen. 2013. Deep Content-based Music Recommendation.  Advances in Neural Information Processing Systems 26 (2013).
[34] Lei Wang and Ee-Peng Lim. 2023. Zero-Shot Next-Item Recommendation using Large Pretrained Language Models. arXiv preprint arXiv:2304.03153 (2023).
[35] Qinyong Wang, Hongzhi Yin, Zhiting Hu, Defu Lian, Hao Wang, and Zi Huang. 2018. Neural Memory Streaming Recommender Networks with Adversarial Training. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2467–2475.
[36] Weiqing Wang, Hongzhi Yin, Zi Huang, Qinyong Wang, Xingzhong Du, and Quoc Viet Hung Nguyen. 2018. Streaming Ranking Based Recommender Systems. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 525–534.
[37] Likang Wu, Zhi Zheng, Zhaopeng Qiu, Hao Wang, Hongchao Gu, Tingjia Shen, Chuan Qin, Chen Zhu, Hengshu Zhu, Qi Liu, et al. 2023. A Survey on Large Language Models for Recommendation. arXiv preprint arXiv:2305.19860 (2023).
[38] Lingxi Xie, Longhui Wei, Xiaopeng Zhang, Kaifeng Bi, Xiaotao Gu, Jianlong Chang, and Qi Tian. 2023. Towards AGI in Computer Vision: Lessons Learned from GPT and Large Language Models. arXiv preprint arXiv:2306.08641 (2023).
[39] Ruobing Xie, Yalong Wang, Rui Wang, Yuanfu Lu, Yuanhang Zou, Feng Xia, and Leyu Lin. 2022. Long Short-term Temporal Meta-learning in Online Recommendation. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. 1168–1176.
[40] Jinghan Zhang, Shiqi Chen, Junteng Liu, and Junxian He. 2023. Composing Parameter-efficient Modules with Arithmetic Operations. arXiv preprint arXiv:2306.14870 (2023).
[41] Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. 2023. Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach. arXiv preprint arXiv:2305.07001 (2023).
[42] Peiyan Zhang and Sunghun Kim. 2023. A Survey on Incremental Update for Neural Recommender Systems. arXiv preprint arXiv:2303.02851 (2023).
[43] Yang Zhang, Keqin Bao, Ming Yan, Wenjie Wang, Fuli Feng, and Xiangnan He. 2024. Text-like Encoding of Collaborative Information in Large Language Models for Recommendation. arXiv preprint arXiv:2406.03210 (2024).
[44] Yang Zhang, Fuli Feng, Chenxu Wang, Xiangnan He, Meng Wang, Yan Li, and Yongdong Zhang. 2020. How to Retrain Recommender System? A Sequential Meta-learning Method. In  Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 1479–1488.
[45] Yang Zhang, Fuli Feng, Jizhi Zhang, Keqin Bao, Qifan Wang, and Xiangnan He. 2023. CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation. arXiv preprint arXiv:2310.19488 (2023).

