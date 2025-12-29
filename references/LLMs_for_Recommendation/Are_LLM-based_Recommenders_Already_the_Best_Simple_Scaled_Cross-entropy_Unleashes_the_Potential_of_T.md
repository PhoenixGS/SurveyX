Are LLM-based Recommenders Already the Best? Simple Scaled Cross-entropy Unleashes the Potential of Traditional Sequential Recommenders
# Are LLM-based Recommenders Already the Best? Simple Scaled Cross-entropy Unleashes the Potential of Traditional Sequential Recommenders
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6ac9/6ac90e27-fffa-47d2-abe9-a9ab0cea7b7f.png" style="width: 50%;"></div>
# Abstract
Large language models (LLMs) have been garnering increasing attention in the recommendation community. Some studies have observed that LLMs, when finetuned by the cross-entropy (CE) loss with a full softmax, could achieve ‘state-ofthe-art’ performance in sequential recommendation. However, most of the baselines used for comparison are trained using a pointwise/pairwise loss function. This inconsistent experimental setting leads to the underestimation of traditional methods and further fosters over-confidence in the ranking capability of LLMs. In this study, we provide theoretical justification for the superiority of the crossentropy loss by demonstrating its two desirable properties: tightness and coverage. Furthermore, this study sheds light on additional novel insights: 1) Taking into account only the recommendation performance, CE is not yet optimal as it is not a quite tight bound in terms of some ranking metrics. 2) In scenarios that full softmax cannot be performed, an effective alternative is to scale up the sampled normalizing term. These findings then help unleash the potential of traditional recommendation models, allowing them to surpass LLM-based counterparts. Given the substantial computational burden, existing LLM-based methods are not as effective as claimed for sequential recommendation. We hope that these theoretical understandings in conjunction with the empirical results will facilitate an objective evaluation of LLM-based recommendation in the future. Our code is available at https://github.com/MTandHJ/CE-SCE-LLMRec.
# 1 Introduction
With the rapid growth of the Internet, the amount of information being generated every moment far beyond human discernment. Recommender systems are thus developed to help humans quick
*Equal contribution.
zhangwei.thu2011@gmail.com
and accurately ascertain the items of interest, and have played increasingly important roles in diverse applications, including e-commerce [50], online news [13], and education [47]. Due to inherent differences in data types and recommendation goals, different tasks are typically handled using various techniques and models. For example, graph neural networks [22] have become predominant in collaborative filtering [15, 31], while Transformer [45] has gained increasing popularity in sequential recommendation [20, 41]. Recently, the prosperity of Large Language Models (LLMs) [36, 33, 43, 44] suggests a promising direction towards universal recommenders [12, 26]. Equipped with carefully designed prompts, they show great potential in explainable and cross-domain recommendations [9, 10]. Nevertheless, there remain non-negligible gaps [21, 1] between LLMs and traditional methods unless domain-specific knowledge is injected. After fine-tuning [27, 48], some researches have observed ‘compelling’ results and hastily affirmed LLM-based recommenders’ ranking capability. Considering the substantial computational burden that comes with LLMs, we contend that such claims must be carefully scrutinized before being accepted and applied, to prevent a massive waste of resources. Recall that the next-token prediction objective used for LLM pre-training (and fine-tuning) inherently relies on a cross-entropy (CE) loss that necessitates a full softmax over the entire corpus. In contrast, most traditional recommenders are trained using pointwise/pairwise losses; for example, either BCE adopted in SASRec [20] or BPR in FMLP-Rec [51] dynamically samples one negative item for each positive item. Some studies [5, 23] have pointed out that CE typically outperforms BCE or BPR in sequential recommendation, which raises our concerns about the fairness of previous comparisons. In this paper, we are to clarify these concerns holistically. Q1. Do LLM-based recommenders outperform traditional models when the cross-entropy loss is applied to both? Since non-ID models tend to have lower performance compared to ID-integrated methods, we conducted a comparison between four state-of-the-art LLM-based recommenders with an independent ID embedding table and four traditional models upon MLP, CNN, RNN, or Transformer. Interestingly, although traditional methods do fall far behind when using BCE or BPR, some of them outperform LLMs by a large margin once the cross-entropy loss is applied. Q2. How to understand the superiority of cross-entropy loss in improving the ranking capability of recommenders? On the one hand, Bruch et al. [4] has theoretically established a connection between the cross-entropy loss and some ranking metrics. On the other hand, Wu et al. [46] further proved its beneficial property in alleviating popularity bias. Here, we follow up on [4] to reveal further novel insights: an ideal recommendation loss should emphasize both tightness and coverage. The former necessitates that the recommendation loss should be designed to be a good soft proxy of the ranking metrics for consistent improvements, while the latter underscores the exploration of an adequate number of negative samples during the training process. Clearly, CE is optimal for coverage as it incorporates all items during each mini-batch iteration. However, we theoretically demonstrate that it remains suboptimal in tightness. Superior recommendation performance can be obtained through a dynamic truncation. Q3. Can traditional models still outperform LLM-based recommenders when full softmax cannot be performed? It is important to note that in practical scenarios, the number of items can reach hundreds of millions, making it infeasible to perform full softmax accurately. Thanks to the advent of subword segmentation algorithms [24], the corpus size of a language model is constrained, thereby providing a feasible solution to the aforementioned problem as long as the separate IDs can be represented in the same manner [19]. Consequently, if traditional models cannot fully unleash their potential in this practical situation, current LLM-based recommenders’ performance remains highly promising. Inspired by the theoretical findings presented in Q2, we propose a very simple alternative named Scaled Cross-Entropy (SCE) to address the concerns in reality. SCE uniformly samples items to estimate the normalization term of cross-entropy. In contrast to the normal sampled softmax loss that exhibits poor ‘tightness’ (as formally justified in Section 4), SCE addresses this issue by scaling up the sampled term using an additional weight. As a result, sampling a very small number of negative samples per iteration is sufficient to achieve comparable results with using cross-entropy by a full softmax. The remainder of this paper is organized as follows. A preliminary investigation concerning their performance will be conducted in Section 2. Then Section 3 will discuss Q2 in theory, while a practical comparison for Q3 will be found in Section 4. We finally review related work on sequential recommendation and LLM-based recommendation in Section 5.
<div style="text-align: center;">Table 1: Loss functions discussed in this paper and their bounding probabilities in the case of uniform sampling. σ(·) stands for the sigmoid function.</div>
·
Loss
Formulation
‘Normalizing’ term Z
Bounding probability ≥
ℓCE
−log
exp(sv+)
�
v∈I exp(sv)
�
v∈I
exp(sv)
1
ℓBCE
−log σ(sv+) −log(1 −σ(sv−))
(1 + exp(sv+))(1 + exp(sv−))
0
ℓBPR
−log σ(sv+ −sv−)
exp(sv+) + exp(sv−)
0
ℓSSM
−log
exp(sv+)
exp(sv+) + 1 · �K
i=1 exp(svi)
exp(sv+) +
K
�
i=1
exp(svi)
1 −m
�
1 −r+/|I|
�⌊K/m⌋
ℓSCE
−log
exp(sv+)
exp(sv+) + α �K
i=1 exp(svi)
exp(sv+) + α
K
�
i=1
exp(svi)
1 −⌈m
α ⌉
�
1 −r+/|I|
�⌊K/⌈m
α ⌉⌋
# 2 Potential of Traditional Recommenders over LLM-based Recommenders
In this section, we showcase how the performance of the traditional sequential recommendation models changes when switching from BCE/BPR to CE. The experimental setup will be introduced
Problem definition. Given a query q that encompasses some user information, a recommender system aims to retrieve some items v ∈I that would be of interest to the user. In sequential recommendation, it is to predict the next item vt+1 based on historical interactions q = [v1, v2, . . . , vt]. The crucial component is to develop a scoring function sqv := sθ(q, v) to accurately model the relevance of a query q to a candidate item v. A common paradigm is to map them into the same latent space through some models parameterized via θ, followed by an inner product operation for similarity calculation. Then, top-ranked items based on these scores will be prioritized for recommendation. Typically the desired recommender is trained to minimize an optimation objective over all observed interactions D:
where v+ indicates the target item for the query q, and the loss function ℓconsidered in this pap (see Table 1) can be reformulated as
Here, we specifically separate the ‘normalizing’ term Zθ(q), as it plays an important role in determining the connection to ranking metrics. We will omit q and θ hereafter if no ambiguity is raised. Datasets. We select two public datasets, i.e., Beauty and Yelp, which have been widely used in previous studies [20, 12, 26]. Following the studies [51, 12], we filter out users and items with less than 5 interactions, and the validation set and test set are split in a leave-one-out fashion, namely the last interaction for testing and the penultimate one for validation. The dataset statistics after pre-processing are presented in Appendix A.4. Evaluation metrics. For each user, the scores returned by the recommender will be sorted in descending order to generate candidate lists. Denoted by r+ := r(v+) = |{v ∈I : sv ≥sv+}| the predicted rank of the target item v+, Normalized Discounted Cumulative Gain (NDCG) and Mean Reciprocal Rank (MRR) 1 are often employed to assess the sorting quality. For next-item recommendation considered in this paper, NDCG and MRR (for a single query) can be simplified to
Both of them increase as the target item v+ is ranked higher and reaches the maximum when v+ is ranked first (i.e., r+ = 1). Consequently, the average quality computed over the entire test set serves as an indicator of the ranking capability.
1In practice, it is deemed meaningless when r+ exceeds a pre-specified threshold k (e.g., k = 1, 5, 10). Hence, the widely adopted NDCG@k and MRR@k metrics are modified to assign zero rewards to these poor ranking results.
1In practice, it is deemed meaningless when r+ exceeds a pre-specified threshold k  Hence, the widely adopted NDCG@k and MRR@k metrics are modified to assign zero r ranking results.
(1)
<div style="text-align: center;">Table 2: Model statistics. ’Max Seq. Len.’ denotes the maximum input sequence length while ‘Max ID Seq. Len.’ indicates the maximum number of historical user interactions used for prediction.</div>
D Seq. Len.’ indicates the maximum number of historical user interactions used for prediction.
Model
Foundation Model
Architecture
Emb. Size
Max Seq. Len.
Max ID Seq. Len.
#Params
P5 (CID+IID) [19]
T5
Transformer
512
512
50
60M
POD [26]
T5
Transformer
512
512
50
60M
LlamaRec [48]
Llama2
Transformer
4096
2048
50
7B
E4SRec [27]
Llama2
Transformer
4096
2048
50
7B
GRU4Rec [17]
RNN
64
50
50
0.80M
Caser [42]
CNN
64
50
50
3.80M
SASRec [20]
Transformer
64
50
50
0.83M
FMLP-Rec [51]
MLP
64
50
50
0.92M
<div style="text-align: center;">Table 3: Overall performance comparison on the Beauty and Yelp datasets. The best results of each block are marked in bold. Gray results indicate less than that of best LLM-based recommenders. ‘▲% over LLM’ represents the relative gap between respective best results.</div>
‘▲% over LLM’ represents the relative gap between respective best results.
Method
Beauty
Yelp
NDCG@5
NDCG@10
MRR@5
MRR@10
NDCG@5
NDCG@10
MRR@5
MRR@10
LLM
POD
0.0125
0.0146
0.0107
0.0116
0.0330
0.0358
0.0267
0.0292
P5(CID+IID)
0.0403
0.0474
0.0345
0.0376
0.0200
0.0252
0.0164
0.0186
LlamaRec
0.0405
0.0492
0.0344
0.0380
0.0306
0.0367
0.0270
0.0295
E4SRec
0.0376
0.0448
0.0326
0.0356
0.0207
0.0260
0.0174
0.0196
BPR
GRU4Rec
0.0155
0.0208
0.0125
0.0147
0.0106
0.0147
0.0086
0.0102
Caser
0.0173
0.0221
0.0143
0.0163
0.0211
0.0250
0.0185
0.0201
SASRec
0.0298
0.0374
0.0248
0.0279
0.0167
0.0218
0.0137
0.0158
FMLP-Rec
0.0321
0.0393
0.0269
0.0298
0.0334
0.0405
0.0286
0.0314
▲% over LLM
-20.8%
-20.3%
-22.1%
-21.5%
+1.3%
+10.3%
+5.7%
+6.5%
BCE
GRU4Rec
0.0134
0.0186
0.0108
0.0129
0.0098
0.0135
0.0079
0.0094
Caser
0.0185
0.0234
0.0154
0.0174
0.0224
0.0264
0.0198
0.0214
SASRec
0.0275
0.0353
0.0225
0.0257
0.0225
0.0281
0.0192
0.0215
FMLP-Rec
0.0301
0.0381
0.0248
0.0281
0.0330
0.0391
0.0287
0.0312
▲% over LLM
-25.8%
-22.6%
-28.1%
-26.0%
-0.1%
+6.5%
+6.1%
+5.7%
CE
GRU4Rec
0.0329
0.0398
0.0281
0.0310
0.0171
0.0231
0.0137
0.0161
Caser
0.0303
0.0361
0.0260
0.0284
0.0211
0.0243
0.0187
0.0200
SASRec
0.0510
0.0597
0.0443
0.0479
0.0345
0.0415
0.0302
0.0331
FMLP-Rec
0.0507
0.0594
0.0438
0.0473
0.0364
0.0444
0.0316
0.0348
▲% over LLM
+25.7%
+21.3%
+28.1%
+26.0%
+10.3%
+21.0%
+16.9%
+18.1%
Baselines. Although LLM itself has surprising zero-shot recommendation ability, there still exist non-negligible gaps [21, 1] unless domain-specific knowledge is injected. Hence, only LLM-based recommenders enhanced by an independent ID embedding table will be compared in this paper: P5 (CID+IID) [19], POD [26], LlamaRec [48], and E4SRec [27]. Specifically, the first two methods take T5 as the foundation model, while the last two methods fine-tune Llama2 for efficient sequential recommendation. Because the data preprocessing scripts provided with POD may lead to information leakage [37], we assign random integer IDs to items rather than sequentially incrementing integer IDs. Additionally, four sequence models including Caser [42], GRU4Rec [17], SASRec [20], and FMLP-Rec [51] are considered here to unveil the true capability of traditional methods. They cover various architectures so as to comprehensively validate the effectiveness of the proposed approximation methods. Table 2 presents an overview of the model statistics. Other implementation details can be found in Appendix A.7. Observations. Let us focus on the first three blocks in Table 3, where the majority of traditional models notably fall behind LLM-based recommenders (e.g., LlamaRec). Although the current stateof-the-art model, FMLP-Rec, achieves fairly good results on Yelp, it still does not perform as well on Beauty compared to P5, LlamaRec, and E4SRec, with a performance gap of at least 20%. In particular, SASRec being a Transformer-based model similar to LLM-based recommenders, unfortunately performs poorly in comparison. It is tempting to attribute this result to the superior model expressive power and world knowledge of LLMs. However, this does not seem to be the case based on the results obtained using CE: SASRec and FMLP-Rec demonstrate significant performance improvements as a result of this subtle change, ultimately surpassing all LLM-based recommenders on both the Beauty and Yelp datasets. GRU4Rec and Caser also benefit from using CE, although they may not be able to outperform LLM-based recommenders due to their limited expressive power.
Hence, we argue that previous affirmative assertions about the LLMs’ recommendation performance are rooted in biased comparisons, wherein BCE or BPR are commonly used for training traditional models. Moreover, the inflated model size (from 60M of P5 to 7B of LlamaRec) only yields marginal improvements in some of the metrics. The rich world knowledge and powerful reasoning ability seem to be of limited use here. In conclusion, even after fine-tuning, LLM-based recommenders fail to outperform state-of-the-art traditional sequence models when full softmax can be performed.
# 3 Tightness and Coverage for Normalizing Term
In this section, we are to clarify the superiority of the cross-entropy loss by examining the desirable properties of its normalizing term. This analysis provides some valuable insights into the design of a recommendation loss that is able to boost the ranking capability. SASRec [20] as the most prominent sequence models will serve as the baseline to empirically elucidate the conclusions in this part. All results are reported over 5 independent runs.
# 3.1 Tightness
To accurately predict the next item a user is likely to purchase, the model must have superior ranking capability, which is quantitatively assessed by the aforementioned metrics NDCG and MRR. Ideally, the recommender should be optimized directly towards these metrics; however, this cannot be achieved exactly due to their discrete nature. If a recommendation loss can be regarded as a soft proxy for NDCG or MRR, then minimizing this loss is expected to consistently improve ranking metrics as well. Bruch et al. [4] has theoretically connected the cross-entropy loss to these ranking metrics, as stated below: Lemma 1 ([4]). Minimizing the cross-entropy loss ℓCE is equivalent to maximizing a lower bound of NDCG and MRR.
Indeed, Lemma 1 can be extended to a more generalized version, which can provide us with a technical foundation for further investigations: Proposition 1. For a target item v+ which is ranked as r+, the following inequality holds true for any n ≥r+ −log NDCG(r) ≤−log MRR(r) ≤ℓ-. (3)
where
Proof. Proof of Proposition 1 is in Appendix A.2.
Proposition 1 implies that all CE-like losses ℓCE-n possess the potential to act as a soft proxy for NDCG and MRR, provided that all items ranked before v+ are retained in the normalizing term. Moreover, the connection becomes tighter as n becomes smaller. From this point of view, the original cross-entropy loss appears to be the ‘worst’ since it is a special case that includes all items (i.e., n = |I|). We further validate that tightness is not the sole principle of worth, and it comes at the expense of certain coverage.
# 3.2 Coverage
Since the superiority of CE possibly stems from its connection to these ranking metrics, one might hypothesize that optimizing a tighter bound with a smaller value of n ≪|I| allows greater performance gains. However, the condition n ≥r+ in Proposition 1 cannot be consistently satisfied via a constant value of n since r+ is dynamically changing during training. Alternatively, an adaptive truncation can be employed for this purpose:
� Note that this η-truncated loss retains only items whose scores are not lower than s+ −η|s+ a tighter bound will be obtained as η drops to 0. Specifically, this η-truncated loss becomes ℓC
(3)
(4)

(5)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8964/89642b58-a485-4fc6-8cfc-a55da223390f.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) η-ablation</div>
<div style="text-align: center;">Figure 1: (a) Performance comparison based on tighter bounds for NDCG. The dashed line represents the results trained by CE (namely the case of η →+∞). (b) Tightness and Coverage illustration of different loss functions. η∗is a task-specific value derived from (a).</div>
(i.e., the tightest case) when η = 0, and approaches ℓCE when η →+∞. Figure 1a illustrates how NDCG@10 varies as η gradually increases from 0.1 to 5. There are two key observations:
1. SASRec performs worst in the tightest case of η ≈0. This can be attributed to the poor coverage of the η-truncated loss: due to the truncation operation, very rare negative items are encountered during training. Moreover, this situation is exacerbated for targets that are easy to recognize, as they get high scores only after a few iterations. Consequently over-fitting is more likely to occur, resulting in deteriorated performance. Nevertheless, all CE-like losses demonstrate competitive performance compared to LLM-based recommenders, further underscoring the importance of tightness for a recommendation loss.
1. SASRec performs worst in the tightest case of η ≈0. This can be attributed to the poor coverage of the η-truncated loss: due to the truncation operation, very rare negative items are encountered during training. Moreover, this situation is exacerbated for targets that are easy to recognize, as they get high scores only after a few iterations. Consequently over-fitting is more likely to occur, resulting in deteriorated performance. Nevertheless, all CE-like losses demonstrate competitive performance compared to LLM-based recommenders, further underscoring the importance of tightness for a recommendation loss. 2. Once η is large enough to exceed the necessary coverage threshold, SASRec begins to benefit from tightness and achieves its best performance around η ≈0.7. Further increasing η however leads to a similar effect as the original cross-entropy loss, thereby along with a slightly degenerate performance due to the suboptimal tightness.
2. Once η is large enough to exceed the necessary coverage threshold, SASRec begins to benefit from tightness and achieves its best performance around η ≈0.7. Further increasing η however leads to a similar effect as the original cross-entropy loss, thereby along with a slightly degenerate performance due to the suboptimal tightness.
Conclusions. We summarize the characteristics of tightness and coverage for different losses in Figure 1b. ℓCE-r+ and ℓCE represent two extremes in terms of tightness and coverage, respectively. However, neither of them is optimal when taking into account only the ranking capability. There actually exists a sweet spot ℓCE-η∗whose tightness and coverage are well balanced. These recommendation loss design principles have been indirectly employed in previous studies [28, 6, 40]: while they focus on exploiting hard samples (for tightness), these items are still drawn from the entire pool of items (for coverage) rather than from a fixed subset. From this point of view, BCE and BPR are far from mature. On the one hand, BCE and BPR sample one negative item for each query, which greatly restricts the coverage. In practice, even after training for thousands of epochs, there is no guarantee that a user will encounter all the negative items, especially when the number of items is large. On the other hand, consuming more training iterations may not alleviate this problem because of their poor tightness (unless given stronger assumptions, no meaningful bounds can be derived for BCE or BPR). Due to the large item size, the negative sample uniformly drawn from them is often too easy to update the model in a meaningful direction.
# Scaled Cross-Entropy for Practical Potential of Tradition
In practical scenarios, the number of items can reach hundreds of millions, possibly making a precise execution of full softmax infeasible. It is noteworthy that despite the formulation of ℓCE-n or ℓCE-η involving only a limited number of items, a full softmax operation is still required before truncation. Therefore, it may not be advisable to perform the aforementioned losses directly in real-world situations. Conversely, LLM-based recommenders may effectively address this problem based on the commonly used subword segmentation algorithms [24] that allow for a fixed corpus. As long as no
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f236/f2366ca4-6772-4a87-a4be-7936ea67348f.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Bounding probabilities of SCE at different values of α = 1, 5, 100 across varying ranks r+ and number of negative items K. The probabilities are calculated based on Eq. (7) and Eq. (10), with negative values being clipped to zero. For the Beauty dataset, |I| = 12101.</div>
independent ID embedding table2 is introduced, the full softmax cost then can be controlled at a constant value regardless of the number of items. Consequently, in scenarios where traditional models fail to fully unleash their potential, current LLM-based recommenders’ performance remains highly promising. Recall that concurrent with the advent of subword segmentation algorithms, some studies resorted to approximations as substitutes for CE. For example, Mnih et al. [32] simplified noise contrastive estimation for bypassing an explicit normalizing over the entire vocabulary. Similarly, we turn to the sampled softmax loss for efficiency. Based on the findings in Section 3, a desirable approximation to CE should exhibit not only tightness but also coverage. Since the normalizing term of cross-entropy is intractable in reality, a straightforward way is to approximate it by (uniformly) sampling part of items from I:
which yields the well-known sampled softmax loss (SSM) [46, 23]. It is worth noting that SSM can have different variants depending on the scoring functions and sampling strategies employed. Although cosine similarity and adaptive sampling probability [28, 6] are expected to yield slightly better performance, the resulting additional memory and computational costs compromise efficiency. Hence, the sampled softmax loss considered here is limited to the use of the inner product scoring function and uniform sampling due to their widespread adoption and efficient implementation in recommender systems. Despite the efficiency gained from uniform sampling, there exists a very particular trade-off between tightness, coverage, and efficiency when it comes to this sampled loss. Firstly, coverage and efficiency are in conflict with each other. Secondly, the discussion of tightness is not very rigorous for ℓSSM since it has a low probability of bounding NDCG in most cases: Theorem 1. Let v+ be a target item which is ranked as r+ ≤2m −1 for some m ∈N, If we uniformly sample K items for training, then with a probability of at least
we have
Proof. Proof of Theorem 1 and corresponding results of MRR are in Appendix A.3.
According to the estimates of the bounding probability in Eq. (7), we illustrate the distribution of probability values for varying r+ and K in the left panel of Figure 2. As can be seen, ℓSSM encounters challenges in bounding NDCG except in the upper right corner where the target item is ranked lower (r+ ≥500) and the number of negative samples is sufficiently large (K ≥300). This
2While non-ID recommenders are appealing and hold significant potential, there remains a substantial performance disparity compared to ID-based methods. Hua et al. [19] have comprehensively investigated various item indexing methods, demonstrating that P5 enhanced by CID (collaborative ID) and IID (independent ID) significantly outperforms other approaches. This is the reason why we solely consider P5 (CID+IID) in this paper.
(6)
(7)

can be understood by the decreasing probabilities in sampling those hard samples ranked higher than the target item. Consequently, the usefulness of the uniformly sampled items gradually diminishes as the model being well-trained. In a nutshell, before considering coverage and tightness, necessary modifications should be made to guarantee a high bounding probability. We would like to emphasize that a very simple modification exists without the need for sampling more items. By scaling up the sampled normalizing term with a pre-specific weight α ≥1, the resulting Scaled Cross-Entropy (SCE) becomes
� In the case of α = 1, it degrades to the normal sampled softmax loss ℓSSM. The following theorem describes why this scaled loss is more likely to bound NDCG. Theorem 2. Under the same conditions as stated in Theorem 1, the inequality (8) holds for ℓSCE with a probability of at least �m �
� � f. Proof of Theorem 2 and corresponding results of MRR are in Appendi
As can be seen in Figure 2, increasing α from 1 to 5 has already made the bounding probabilities meaningful in most areas. Further increases guarantee a reasonable probability to bound NDCG or MRR even when the target item has been ranked highly. In contrast to ℓSSM, SCE theoretically requires a much smaller number of negative samples to achieve comparable recommendation performance. Of course, SCE is not perfect: the scaling operation inevitably raises concerns about the high variance problem. As depicted in the left panel of Figure 3, if negative samples are very rare (K = 1), a larger weight of α tends to worsen the ranking capability. Fortunately, this high variance problem appears less significant as K slightly increases (e.g., K ≥10 for Beauty).
 ≥ Sampling 100 negative samples for ℓSCE (α = 100) produces comparable performance to using 500 negative samples for ℓSSM on the Beauty dataset. Moreover, it can be inferred from the right panel that ℓSCE is quite robust to the choice of α; competitive results could be consistently obtained even with a value of α = 104.
LLM-based recommenders are not as effective as claimed in a practical scenario. For efficiency purposes, less than 5% of all items will be sampled, specifically K = 500 for Beauty and K = 100 for Yelp. Due to the sampling nature of SCE, compared to the 200 epochs required for CE, it sometimes consumes more iterations (300 epochs) until convergence. Other hyper-parameters of SCE completely follow CE. We then compare traditional models trained with SCE to the state-of-the-art LLM-based recommenders in Figure 4. As can be seen, SCE enables traditional models to achieve comparable recommendation performance by sampling only a small number of negative items. Notably, for simple models like GRU4Rec and Caser, SCE even surpasses CE in improving ranking capability. We hypothesize that this is because the original cross-entropy loss may be too challenging and unfocused for these models to optimize given their limited expressive power. For models like SASRec and FMLP-Rec, the performance gains achieved by SCE are not very significant, but they still allow these models to outperform all other LLM-based recommenders by a substantial margin. Considering the computational and memory costs associated with LLM, we believe that existing LLM-based recommenders may not be as effective as claimed in practical applications.
# 5 Related Work
Recommender systems are developed to enable users to quickly and accurately ascertain relevant items. The primary principle is to learn underlying interests from user information, especially histor-
(9)
(10)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d804/d8041f45-aa3e-4f8c-ba09-2a0f119acfd1.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Sampled softmax loss comparison on Beauty. Left: ℓSSM versus ℓSCE (α = 100) across different number of negative items. Right: ℓSCE (K = 100) with an increasing value of α.</div>
Figure 3: Sampled softmax loss comparison on Beauty. Left: ℓSSM versus ℓSCE (α = 100) across different number of negative items. Right: ℓSCE (K = 100) with an increasing value of α.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2639/263901cb-822a-4715-bb98-576a8386fab0.png" style="width: 50%;"></div>
Figure 4: Performance comparisons between state-of-the-art LLM-based recommenders and traditional models using ℓCE and ℓSCE(α = 100). Existing LLM-based recommenders are not as effective as claimed even in a practical scenario.
<div style="text-align: center;">Figure 4: Performance comparisons between state-of-the-art LLM-based recommenders and traditional models using ℓCE and ℓSCE(α = 100). Existing LLM-based recommenders are not as effective as claimed even in a practical scenario.</div>
ical interactions. Collaborative filtering [38, 16] performs personalized recommendation by mapping users and items into the same latent space in which interacted pairs are close. Beyond static user representations, sequential recommendation [39, 25] focuses on capturing dynamic interests from item sequences. Early efforts such as GRU4Rec [17] and Caser [42] respectively apply recurrent neural networks (RNNs) and convolutional neural networks (CNNs) to sequence modeling. Recently, Transformer [45, 8] has become increasingly popular in recommendation due to its parallel efficiency and superior performance. For example, SASRec [20] and BERT4Rec [41] respectively employ unidirectional and bidirectional self-attention. Differently, Zhou et al. [51] present FMLPRec to denoise the item sequences through learnable filters so that state-of-the-art performance can be obtained by mere MLP modules. LLM for recommendation has gained a lot of attention recently because: 1) The next-token generation feature can be easily extended to the next-item recommendation (i.e., sequential recommendation); 2) The immense success of LLM in natural language processing promises the development of universal recommenders. Some studies [7, 10] have demonstrated the powerful zero/few-shot ability of LLMs (e.g., GPT [33]), especially their potential in explainable and cross-domain recommendations [9, 10]. Nevertheless, there is a consensus [21, 49, 1] that without domain-specific knowledge learned by fine-tuning, LLM-based recommenders still stay far behind traditional models. As an early effort, P5 [12] unifies multiple recommendation tasks into a sequence-to-sequence paradigm. Based on the foundation model of T5 [36], each task can be activated through some specific prompts. Hua et al. [19] takes a further step beyond P5 by examining the impact of various ID indexing methods, and a combination of collaborative and independent indexing stands out. Recently, more LLM recommenders [35, 29, 27, 48] based on Llama [43] or Llama2 [44] are developed. For example, LlamaRec [48] proposes a two-stage framework based on Llama2 to rerank the candidates retrieved by traditional models. To enable LLM to correctly identify items, E4SRec [27] incorporates ID embeddings trained by traditional sequence models through a linear adaptor, and applies LORA [18] for parameter-efficient fine-tuning. Cross-entropy and its approximations [38, 14, 2, 3, 30] have been extensively studied. The most related works are: 1) Bruch et al. [4] theoretically connected cross-entropy to some ranking metrics; 2) Wu et al. [46] further found its desirable property in alleviating popularity bias; and recently, 3) Klenitskiy et al. [23] and Petrov et al. [34] respectively applied cross-entropy and a generalized BCE loss to eliminate the performance gap between SASRec and BERT4Rec. Differently, we are to 1) understand the superiority of cross-entropy from the tightness and coverage perspectives; 2) make some necessary modifications to the sampled softmax loss based on these findings; and 3) facilitate an objective evaluation of LLM-based recommendation by acknowledging the true capability of traditional models in practical applications.
# 6 Discussion and Conclusion
The emergence of LLMs has inspired novel attempts in various fields, including recommender systems. The great potential of explainable and cross-domain recommendations signals a promising direction toward the development of a universal recommender. Besides, some researchers would like to argue that LLM-based recommenders may outperform traditional models in accuracy by virtue
of their superior world knowledge and reasoning ability. We are skeptical of this claim because in reality, even experts find it challenging to provide personalized recommendations for different users. The recommended items may align logically but may not accurately match a user’s specific interests. We have empirically investigated this in Section 2 and Section 4, which validates our point about existing LLM-based recommenders. However, we also acknowledge that offline comparisons may not be entirely convincing. Recommendation is inherently a subjective task, and recommenders that follow logic may produce diverse recommendations to circumvent the echo chamber effect [11]. This work is not intended to discourage further exploration of the potential of LLM for personalized recommendations. We simply advocate a fair experimental environment to ensure that both traditional models and LLM-based recommenders can be objectively evaluated in the future. As demonstrated in this paper, the sampled softmax loss especially the proposed SCE can serve as a reliable criterion for unleashing the true ranking capability of traditional models.
# References
[1] Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. Tallrec: An effective and efficient tuning framework to align large language model with recommendation. In ACM Conference on Recommender Systems (RecSys), pages 1007–1014. ACM, 2023. [2] Yoshua Bengio and Jean-S´ebastien Sen´ecal. Adaptive importance sampling to accelerate training of a neural probabilistic language model. IEEE Transactions on Neural Networks (TNNLS), 19(4):713–722, 2008. [3] Guy Blanc and Steffen Rendle. Adaptive sampled softmax with kernel based sampling. In International Conference on Machine Learning (ICML), volume 80 of Proceedings of Machine Learning Research, pages 589–598. PMLR, 2018. [4] Sebastian Bruch, Xuanhui Wang, Michael Bendersky, and Marc Najork. An analysis of the softmax cross entropy loss for learning-to-rank with binary relevance. In ACM SIGIR International Conference on Theory of Information Retrieval (ICTIR), pages 75–78. ACM, 2019. [5] Chong Chen, Weizhi Ma, Min Zhang, Chenyang Wang, Yiqun Liu, and Shaoping Ma. Revisiting negative sampling vs. non-sampling in implicit recommendation. ACM Transactions on Information Systems (TOIS), 41(1):1–25, 2023. [6] Jin Chen, Defu Lian, Binbin Jin, Kai Zheng, and Enhong Chen. Learning recommenders for implicit feedback with importance resampling. In ACM Web Conference (WWW), pages 1997–2005. ACM, 2022. [7] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. Uncovering chatgpt’s capabilities in recommender systems. In ACM Conference on Recommender Systems (RecSys), pages 1126–1132. ACM, 2023. [8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pages 4171–4186. Association for Computational Linguistics, 2019. [9] Yue Feng, Shuchang Liu, Zhenghai Xue, Qingpeng Cai, Lantao Hu, Peng Jiang, Kun Gai, and Fei Sun. A large language model enhanced conversational recommender system. arXiv preprint arXiv:2308.06212, 2023. 10] Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang. Chatrec: Towards interactive and explainable llms-augmented recommender system. arXiv preprint arXiv:2303.14524, 2023. 11] Yingqiang Ge, Shuya Zhao, Honglu Zhou, Changhua Pei, Fei Sun, Wenwu Ou, and Yongfeng Zhang. Understanding echo chambers in e-commerce recommender systems. In International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2261– 2270, 2020.
[12] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. Recommendation as language processing (RLP): A unified pretrain, personalized prompt & predict paradigm (P5). In ACM Conference on Recommender Systems (RecSys), pages 299–315. ACM, 2022. [13] Shansan Gong and Kenny Q. Zhu. Positive, negative and neutral: Modeling implicit feedback in session-based news recommendation. In International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1185–1195. ACM, 2022. [14] Michael Gutmann and Aapo Hyv¨arinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In International Conference on Artificial Intelligence and Statistics (AISTATS), pages 297–304. JMLR Workshop and Conference Proceedings, 2010. [15] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-Dong Zhang, and Meng Wang. Lightgcn: Simplifying and powering graph convolution network for recommendation. In International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), pages 639–648. ACM, 2020. [16] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In International Conference on World Wide Web (WWW), pages 173– 182. ACM, 2017. [17] Bal´azs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. Session-based recommendations with recurrent neural networks. In International Conference on Learning Representations (ICLR), 2016. [18] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022. [19] Wenyue Hua, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. How to index item ids for recommendation foundation models. In Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP), pages 195–204. ACM, 2023. [20] Wang-Cheng Kang and Julian J. McAuley. Self-attentive sequential recommendation. In IEEE International Conference on Data Mining (ICDM), pages 197–206. IEEE Computer Society, 2018. [21] Wang-Cheng Kang, Jianmo Ni, Nikhil Mehta, Maheswaran Sathiamoorthy, Lichan Hong, Ed Chi, and Derek Zhiyuan Cheng. Do llms understand user preferences? evaluating llms on user rating prediction. arXiv preprint arXiv:2305.06474, 2023. [22] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR), 2017. [23] Anton Klenitskiy and Alexey Vasilev. Turning dross into gold loss: is bert4rec really better than sasrec? In ACM Conference on Recommender Systems (RecSys), pages 1120–1125. ACM, 2023. [24] Taku Kudo. Subword regularization: Improving neural network translation models with multiple subword candidates. In Annual Meeting of the Association for Computational Linguistics (ACL), pages 66–75. Association for Computational Linguistics, 2018. [25] Jiacheng Li, Yujie Wang, and Julian J. McAuley. Time interval aware self-attention for sequential recommendation. In International Conference on Web Search and Data Mining (WSDM), pages 322–330. ACM, 2020. [26] Lei Li, Yongfeng Zhang, and Li Chen. Prompt distillation for efficient llm-based recommendation. In ACM International Conference on Information and Knowledge Management (CIKM), pages 1348–1357. ACM, 2023.
[27] Xinhang Li, Chong Chen, Xiangyu Zhao, Yong Zhang, and Chunxiao Xing. E4srec: An elegant effective efficient extensible solution of large language models for sequential recommendation. arXiv preprint arXiv:2312.02443, 2023. [28] Defu Lian, Qi Liu, and Enhong Chen. Personalized ranking with importance sampling. In ACM Web Conference (WWW), pages 1093–1103. ACM / IW3C2, 2020. [29] Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, Xiang Wang, and Xiangnan He. Llara: Aligning large language models with sequential recommenders. arXiv preprint arXiv:2312.02445, 2023. [30] Zhuang Ma and Michael Collins. Noise contrastive estimation and negative sampling for conditional models: Consistency and statistical efficiency. In Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3698–3707. Association for Computational Linguistics, 2018. [31] Kelong Mao, Jieming Zhu, Xi Xiao, Biao Lu, Zhaowei Wang, and Xiuqiang He. Ultragcn: Ultra simplification of graph convolutional networks for recommendation. In ACM International Conference on Information and Knowledge Management (CIKM), pages 1253–1262. ACM, 2021. [32] Andriy Mnih and Yee Whye Teh. A fast and simple algorithm for training neural probabilistic language models. In International Conference on Machine Learning (ICML), 2012. [33] OpenAI. GPT models documentation. https://platform.openai.com/docs/models/ overview, 2023. [34] Aleksandr Vladimirovich Petrov and Craig MacDonald. gsasrec: Reducing overconfidence in sequential recommendation trained with negative sampling. In Proceedings of the 17th ACM Conference on Recommender Systems, RecSys 2023, Singapore, Singapore, September 18-22, 2023, pages 116–128. ACM, 2023. [35] Junyan Qiu, Haitao Wang, Zhaolin Hong, Yiping Yang, Qiang Liu, and Xingxing Wang. Controlrec: Bridging the semantic gap between language model and personalized recommendation. arXiv preprint arXiv:2311.16441, 2023. [36] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research (JMLR), 21:140:1–140:67, 2020. [37] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q Tran, Jonah Samost, et al. Recommender systems with generative retrieval. arXiv preprint arXiv:2305.05065, 2023. [38] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. BPR: bayesian personalized ranking from implicit feedback. In Conference on Uncertainty in Artificial Intelligence (UAI), pages 452–461. AUAI Press, 2009. [39] Guy Shani, David Heckerman, Ronen I Brafman, and Craig Boutilier. An mdp-based recommender system. Journal of Machine Learning Research (JMLR), 6(9):1265–1295, 2005. [40] Wentao Shi, Jiawei Chen, Fuli Feng, Jizhi Zhang, Junkang Wu, Chongming Gao, and Xiangnan He. On the theories behind hard negative sampling for recommendation. In ACM Web Conference (WWW), pages 812–822. ACM, 2023. [41] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In ACM International Conference on Information and Knowledge Management (CIKM), pages 1441–1450. ACM, 2019. [42] Jiaxi Tang and Ke Wang. Personalized top-n sequential recommendation via convolutional sequence embedding. In ACM International Conference on Web Search and Data Mining (WSDM), pages 565–573. ACM, 2018.
[43] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aur´elien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971, 2023. [44] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, et al. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288, 2023. [45] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), pages 5998–6008, 2017. [46] Jiancan Wu, Xiang Wang, Xingyu Gao, Jiawei Chen, Hongcheng Fu, Tianyu Qiu, and Xiangnan He. On the effectiveness of sampled softmax loss for item recommendation. ACM Transactions on Information Systems (TOIS), 2023. [47] Siyu Wu, Jun Wang, and Wei Zhang. Contrastive personalized exercise recommendation with reinforcement learning. IEEE Transactions on Learning Technologies (TLT), 17:691–703, 2024. [48] Zhenrui Yue, Sara Rabhi, Gabriel de Souza Pereira Moreira, Dong Wang, and Even Oldridge. Llamarec: Two-stage recommendation using large language models for ranking. arXiv preprint arXiv:2311.02089, 2023. [49] Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. Recommendation as instruction following: A large language model empowered recommendation approach. arXiv preprint arXiv:2305.07001, 2023. [50] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. Deep interest network for click-through rate prediction. In ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD), pages 1059–1068, 2018. [51] Kun Zhou, Hui Yu, Wayne Xin Zhao, and Ji-Rong Wen. Filter-enhanced MLP is all you need for sequential recommendation. In ACM Web Conference (WWW), pages 2388–2399. ACM, 2022.
# 1 Introduction
# 2 Potential of Traditional Recommenders over LLM-based Recommenders
3 Tightness and Coverage for Normalizing Term 3.1 Tightness . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.2 Coverage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
# 4 Scaled Cross-Entropy for Practical Potential of Traditional Model
# 5 Related Work
# 6 Discussion and Conclusion
# A Appendices
A.1 Overview of Loss Function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14 A.2 Proof of Proposition 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15 A.3 Proof of Theorem 1 and Theorem 2 . . . . . . . . . . . . . . . . . . . . . . . . . 15 A.4 Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17 A.5 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17 A.6 Computational Complexity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18 A.7 Detailed Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18 A.8 Broader Impact . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18 A.9 Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
# A Appendices
# A.1 Overview of Loss Function
Cross-Entropy (CE), also known as the negative log-likelihood (NLL) loss, can be formulated follows:
Cross-Entropy (CE), also known as the negative log-likelihood (NLL) loss, can be formulated as follows:
# Cross-Entropy (CE), also known as the negative log-likelihood (NLL) loss, can be formulated as follows:
� �� � This is also the de facto objective commonly used in the pre-training (fine-tuning) of LLMs. Binary Cross-Entropy (BCE). BCE samples one negative item v−for each target v+, which necessitates the recommender to possess excellent pointwise scoring capability:
ℓBCE = −log σ(sv+) −log(1 −σ(sv−)) = −log exp(sv+) 1 + exp(sv+) −log 1 1 + exp(sv−) = −sv+ + log � (1 + exp(sv+))(1 + exp(sv−)) � � �� � .
� �� Here σ : R →[0, 1] denotes the sigmoid function.
3
Bayesian Personalized Ranking (BPR) [38] also samples one negative item v−for each target v+, but it intends to maximize the probability that v+ will be chosen in preference to v−:
� �� � Noise Contrastive Estimation (NCE) [14, 32] requires the model to discriminate the target v+ from an easy-to-sample noise distribution:
� �� � oise Contrastive Estimation (NCE) [14, 32] requires the model to discriminate the target v+ from  easy-to-sample noise distribution:
� �� � In the case of uniform sampling, s′ v = sv −c −log K |I|, where c is a trainable parameter as an estimate of.
� �� � In the case of uniform sampling, s′ v = sv −c −log K |I|, where c is a trainable parameter as  estimate of log ZCE.
# A.2 Proof of Proposition 1
Proposition 2. For a target item v+ which is ranked as r+, the following inequality holds true for any n ≥r+ (11)
Proposition 2. For a target item v+ which is ranked as r+, the following inequality holds true for any
where
Proof. Notice that log2(1 + x) ≤x holds true for any x ≥1. Hence, we have
� where δ(condition) = 1 if the given condition is true else 0, and the second-to-last inequality holds because exp(sv −sv+) ≥1 when sv > sv+.
# A.3 Proof of Theorem 1 and Theorem 2
First, let us introduce some lemmas that give a lower bound of ℓSCE. Lemma 2. Let ξ be the number of sampled items with scores not lower than that of the target; that is
First, let us introduce some lemmas that give a lower bound of ℓSCE. Lemma 2. Let ξ be the number of sampled items with scores not lower than that of the target; that is ξ := ��{vi : svi ≥sv+, i = 1, 2, . . . , K} ��. (13) Then, we have
(11)
(11) (12)
(12)
(13)
(14)
Lemma 3. Let ξ ∼B(K, p) denote a random variable representing the number of successes over K binomial trials with a probability of p. Then, we have
Proof. 3 Divide the K independent binomial trials into m disjoint groups, each containing at least ⌊K/m⌋trials. If ξ < m, then one of the groups must have no successes observed; formally, we have
Hence, the proof is completed by noting the fact that P(ξ ≥m) = 1 −P(ξ < m).
Hence, the proof is completed by noting the fact that
Now we are ready to prove Theorem 1 and Theorem 2, which directly follow from the subsequent theorem. Theorem 3. Let v+ be a target item which is ranked as r+ ≤2m −1 for some m ∈N, If we uniformly sample K items for training, then we have
� � � � Proof. As r+ ≤2m −1 for some m ∈N, we immediately have −log NDCG(r+) ≤log m, −log MRR(r+) ≤m log 2.
Lemma 2 implies that
P � −log NDCG ≤ℓSCE � ≥P � log(1 + αξ) ≥log m �
(15)
(16)
(18)
(19)
19)
(20)
(20) (21)
(21)
(22)
(22) (23) (24)
The last equality holds because ξ is an integer random variable. Also notice that uniformly sampling from I yields a hit probability of p = r+/|I| such that the score of the sampled item is not lower than that of the target (i.e., the top-r+ ranked items). Therefore, based on Lemma 3, we have
� � �� � The proof for MRR is analogous by noting that
P � −log MRR ≤ℓSCE � ≥P � log(1 + αξ) ≥m log 2 �
<div style="text-align: center;">Table 4: Dataset statistics.</div>
Dataset
#Users
#Items
#Interactions
Density
Avg. Len.
Beauty
22,363
12,101
198,502
0.07%
8.9
Yelp
30,431
20,033
316,354
0.05%
10.4
In this study, we perform experiments on two public datasets. Specifically, the Beauty dataset is extracted from Amazon reviews4, while the Yelp5 dataset collects the interactions that occured in 2019.
# A.5 Baselines
• P5 (CID+IID) [19] unifies multiple tasks (e.g., sequential recommendation and rating prediction) into a sequence-to-sequence paradigm. The use of collaborative and independent indexing together creates LLM-compatible item IDs. • POD [26] bridges IDs and words by distilling long discrete prompts into a few continuous prompts. It also suggests a task-alternated training strategy for efficiency. • LlamaRec [48] aims to address the slow inference process caused by autoregressive generation. Given the candidates retrieved by traditional models, it subsequently reranks them based on the foundation model of Llama2. • E4SRec [27] incorporates ID embeddings trained by traditional sequence models through a linear adaptor, and applies LORA [18] for parameter-efficient fine-tuning. Additionally, five sequence models, covering MLP, CNN, RNN, and Transformer architectures, are considered here to uncover the true capability of traditional methods. • GRU4Rec [17] applies RNN to recommendation with specific modifications made to cope with data sparsity. • Caser [42] treats the embedding matrix as an ‘image’, and captures local patterns by utilizing traditional filters. • SASRec [20] is a pioneering work equipped with a unidirectional self-attention mechanism. By its nature, SASRec predicts the next item based on previously interacted items. • FMLP-Rec [51] denoises item sequences in the frequency domain. Although FMLP-Rec consists solely of MLPs, it performs better than Transformer-based models. 4https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
(25)
(26)
The major cost of cross-entropy lies in the inner product and softmax operations. It has a complexity of O(|I|d), where d denotes the embedding size before similarity calculation. In contrast, the approximations require a lower cost O(Kd), correspondingly an additional overhead O(K) for uniform sampling. Overall, it is profitable if K ≪|I|. The experiments of traditional models are conducted on an Intel Xeon CPU E5-2680 v4 platform and a single RTX 3090 GPU, while those for LLM-based recommenders are performed on an Intel Xeon Gold 6133 CPU platform and four A100 GPUs.
# A.7 Detailed Settings
For traditional models considered in this paper, we use the Xavier/Normal initialization method to randomly initialize the embedding table of dimension 64. The Adam optimizer is employed for training, whose learning rate and weight decay are retuned in the range of {1e-4, 5e-4, 1e-3, 5e-3} and [0, 5e-4], respectively. 200 epochs is enough for CE while sometimes 300 epochs are required for BCE, BPR, and SCE. For LLM-based recommenders, we conduct experiments following their source code and paper settings. In particular, the learning rate for POD and P5 are returned within {1e-5, 5e-5, 1e-4, 5e-4, 1e-3}.
# A.8 Broader Impact
LLM-based recommenders still suffer from enormous computational burden, the observations that existing LLM-based recommenders are not as effective as claimed may prevent them from practical applications, so as to avoid wastage of resources. On the other hand, this work may slow down the progress of LLM-based recommenders in personalized recommendations, although this is not our intention. We simply advocate a fair experimental environment to ensure that both traditional models and LLM-based recommenders can be objectively evaluated in the future.
# A.9 Limitations
There are three major limitations. Firstly, the offline comparisons may not be entirely convincing. Recommendation is inherently a subjective task, and recommenders that follow logic may produce diverse recommendations to circumvent the echo chamber effect [11]. Secondly, as we notice that the next-token generation feature of LLMs can be easily extend to the next-item recommendation, only sequential recommendation is considered in this paper. Hence, the observations and conclusions presented in this study may differ when applied to general recommendation [16, 15]. Finally, the proposed SCE encounters a high variance problem when the negative items are very rare (K = 1). Fortunately, for the Beauty dataset, a value of K ≥10 is sufficient to achieve satisfactory results, which is computationally friendly in practice.
