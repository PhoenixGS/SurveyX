# Understanding the Role of Cross-Entropy Loss in Fairly Evaluating Large Language Model-based Recommendation
Zhangchi Zhu∗ East China Normal University Shanghai, China zczhu@stu.ecnu.edu.cn Jun Wang East China Normal University Shanghai, China wongjun@gmail.com
Cong Xu∗ East China Normal University Shanghai, China congxueric@gmail.com Zhangchi Zhu∗ East China Normal University Shanghai, China zczhu@stu.ecnu.edu.cn
Cong Xu∗ East China Normal University Shanghai, China congxueric@gmail.com
Jianyong Wang Tsinghua University Beijing, China jianyong@tsinghua.edu.cn
22 Feb 2024
ABSTRACT
# ABSTRACT
Large language models (LLMs) have gained much attention in the recommendation community; some studies have observed that LLMs, fine-tuned by the cross-entropy loss with a full softmax, could achieve state-of-the-art performance already. However, these claims are drawn from unobjective and unfair comparisons. In view of the substantial quantity of items in reality, conventional recommenders typically adopt a pointwise/pairwise loss function instead for training. This substitute however causes severe performance degradation, leading to under-estimation of conventional methods and over-confidence in the ranking capability of LLMs. In this work, we theoretically justify the superiority of crossentropy, and showcase that it can be adequately replaced by some elementary approximations with certain necessary modifications. The remarkable results across three public datasets corroborate that even in a practical sense, existing LLM-based methods are not as effective as claimed for next-item recommendation. We hope that these theoretical understandings in conjunction with the empirical results will facilitate an objective evaluation of LLMbased recommendation in the future. Our code is available at https: //github.com/MTandHJ/CE-SCE-LLMRec.
arXiv:2402.06216v2
arXiv:240
# CCS CONCEPTS
# • Information systems →Recommender systems; • Computer systems organization →Neural networks.
# KEYWORDS
ACM Reference Format: Cong Xu, Zhangchi Zhu, Jun Wang, Jianyong Wang, and Wei Zhang. 2024. Understanding the Role of Cross-Entropy Loss in Fairly Evaluating Large
ACM Reference Format: Cong Xu, Zhangchi Zhu, Jun Wang, Jianyong Wang, and Wei Zhang. 2024. Understanding the Role of Cross-Entropy Loss in Fairly Evaluating Large ∗Equal contribution
∗Equal contribution
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. Conference’17, July 2017, Washington, DC, USA © 2024 Association for Computing Machinery. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/XXXXXXX.XXXXXXX
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. Conference’17, July 2017, Washington, DC, USA © 2024 Association for Computing Machinery. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/XXXXXXX.XXXXXXX
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7a1e/7a1eb2e6-98e3-4522-ba2c-da63ed9e5956.png" style="width: 50%;"></div>
Figure 1: Recommendation performance comparisons. The marker size depicts the number of model parameters: 60M for P5 (CID + IID) [19], 7B for LlamaRec [53] and E4SRec [28], and merely ≤1M for SASRec [21].
Language Model-based Recommendation . In Proceedings of ACM Conference (Conference’17). ACM, New York, NY, USA, 16 pages. https://doi.org/ XXXXXXX.XXXXXXX
# 1 INTRODUCTION
With the growth of the Internet, the amount of information being generated every moment is far beyond human discernment. Recommender systems are thus developed to help humans quickly and accurately ascertain the items of interest, and have played important roles in diverse applications, including e-commerce [55], online news [13], and education [50]. Due to inherent differences in data types and recommendation goals, different tasks are typically handled using various techniques and separate models. For example, graph neural networks [23] have dominated collaborative filtering [15, 32], while Transformer [48] becomes increasingly popular in sequential recommendation [21, 44]. Recently, the prosperity of Large Language Models (LLMs) [35, 38, 46, 47] suggests a promising direction towards universal recommenders [12, 27]. Equipped with carefully designed prompts, they show great potential in explainable and cross-domain recommendations [9, 10]. Nevertheless, there still exist non-negligible gaps [1, 22] between LLMs and conventional methods unless domainspecific knowledge is injected. Some researches have observed ‘compelling’ results after fine-tuning [28, 53], and hastily affirmed LLM-based recommenders’ ranking capability.
However, the comparisons therein are not objective and fair enough, leading to under-estimation of conventional recommenders and over-confidence in LLMs. Recall that the next-token prediction objective used for LLM pre-training (and fine-tuning), by its nature, is a cross-entropy loss that needs a full softmax over the entire corpus. In view of the substantial quantity of items in reality, conventional methods typically adopt a pointwise/pairwise loss function (e.g., BCE and BPR). This compromise however causes significant performance degradation. As shown in Figure 1, SASRec trained with cross-entropy outperforms LLMs by a large margin, while falling behind with BCE or BPR. Such superior results relying on cross-entropy however cannot serve as direct evidence to challenge the ranking capability of existing LLM-based recommenders, since the full softmax is intractable to calculate in practice. In this work, we re-emphasize the ability to optimize ranking metrics for a desired recommendation loss, and then unveil the corresponding limitations of some approximations to cross-entropy. To achieve effective and practical approximations, we introduce some novel alternatives with theoretical analysis. In summary, the innovative insights and technical contributions are as follows:
 Minimizing cross-entropy is equivalent to maximizing a lower bound of Normalized Discounted Cumulative Gain (NDCG) and Reciprocal Rank (RR). One can thus expect that the ranking capability would be gradually enhanced as crossentropy is optimized during training. We further show that dynamic truncation on the normalizing term yields a tighter bound and potentially better performance. This fact highlights the importance of optimizing these ranking metrics, and the crossentropy loss is arguably adequate for this purpose. The challenge that remains unsolved is how to realize the approximation in an effective and practical manner, so the comparison with LLMbased recommenders is meaningful in reality. After revisiting the limitations of some well-known approximations, a rather simple solution will be presented.
setting fails to optimize a meaningful bound in the early stages of training. Before the advent of subword segmentation algorithms [25], the training of neural language models also struggles to circumvent an explicit normalizing over the entire vocabulary. Mnih et al. [34] thus resorted to a simplified NCE that fixes the normalizing term estimate as a constant value of 1. This suggestion however introduces training difficulties in recommendation: sampling more negative samples should accelerate the training yet the opposite occurs. This intriguing phenomenon is attributed to the weak connection between NCE and NDCG (RR). Because NCE grows exponentially fast w.r.t. the number of positively scored items, a meaningless bound is encountered in the early training stages. This conclusion suggests adjusting the estimate of the normalizing term to a slightly larger value, which shows promising empirical performance but lacks consistent applicability. Next, we introduce a more reliable loss.  Scaling up the sampled normalizing term provides an effective and practical approximation to cross-entropy. Since the normalizing term of cross-entropy is intractable in reality, a direct way is to approximate it by (uniformly) sampling part of items (a.k.a. sampled softmax loss [49]). To further mitigate
the magnitude loss caused by sampling, we multiply it by an additional weight so the sampled term is scaled up. Indeed, this modification can also be understood as a special case of importance sampling [2], in which the proposal distribution assigns a higher probability mass to the target item. Unlike NCE, this Scaled Cross-Entropy (dubbed SCE) yields a bound mainly determined by the current rank of the target item, making it meaningful even in the early training stages. Empirically, sampling a very few negative samples per iteration is sufficient to achieve comparable results to using cross-entropy with a full softmax. Based on these approximations for cross-entropy, we conduct a comprehensive investigation to assess the true ranking capability of both conventional and LLM-based recommenders. The experimental results presented in Section 5 suggest the over-confidence in existing LLM-based methods. Even without considering the model sizes, they are still far inferior to conventional methods in next-item recommendation. Apart from the potential of explainability and cross-domain transferability, further investigation and exploration are necessary to assess the true ranking capability of LLM-based recommenders.
the magnitude loss caused by sampling, we multiply it by an additional weight so the sampled term is scaled up. Indeed, this modification can also be understood as a special case of importance sampling [2], in which the proposal distribution assigns a higher probability mass to the target item. Unlike NCE, this Scaled Cross-Entropy (dubbed SCE) yields a bound mainly determined by the current rank of the target item, making it meaningful even in the early training stages. Empirically, sampling a very few negative samples per iteration is sufficient to achieve comparable results to using cross-entropy with a full softmax.
Based on these approximations for cross-entropy, we conduct a comprehensive investigation to assess the true ranking capability of both conventional and LLM-based recommenders. The experimental results presented in Section 5 suggest the over-confidence in existing LLM-based methods. Even without considering the model sizes, they are still far inferior to conventional methods in next-item recommendation. Apart from the potential of explainability and cross-domain transferability, further investigation and exploration are necessary to assess the true ranking capability of LLM-based recommenders.
# 2 RELATED WORK
Recommender systems are developed to enable users to quickly and accurately ascertain relevant items. The primary principle is to learn underlying interests from user information, especially historical interactions. Collaborative filtering [16, 40] performs personalized recommendation by mapping users and items into the same latent space in which interacted pairs are close. Beyond static user representations, sequential recommendation [26, 42] focuses on capturing dynamic interests from item sequences. Early efforts such as GRU4Rec [17] and Caser [45] respectively apply recurrent neural networks (RNNs) and convolutional neural networks (CNNs) to sequence modeling. Recently, Transformer [8, 48] becomes increasingly popular in recommendation due to its parallel efficiency and superior performance. For example, SASRec [21] and BERT4Rec [44] respectively employ unidirectional and bidirectional self-attention. Differently, Zhou et al. [56] present FMLP-Rec to denoise the item sequences through learnable filters so that stateof-the-art performance can be obtained by mere MLP modules. LLM for recommendation has gained a lot of attention recently because: 1) The next-token generation feature is technically easy to extend to the next-item recommendation (i.e., sequential recommendation); 2) The immense success of LLM in natural language processing promises the development of universal recommenders. Some studies [7, 10] have demonstrated the powerful zero/fewshot ability of LLMs (e.g., GPT [35]), especially their potential in explainable and cross-domain recommendations [9, 10]. Nevertheless, there is a consensus [1, 22, 54] that without domain-specific knowledge learned by fine-tuning, LLM-based recommenders still stay far behind conventional models. As an early effort, P5 [12] unifies multiple recommendation tasks into a sequence-to-sequence paradigm. Based on the foundation model of T5 [38], each task can be activated through some specific prompts. Hua et al. [19] takes a further step beyond P5 by examining the impact of various ID indexing methods, and a combination of collaborative and independent indexing stands out. Recently,
more LLM recommenders [28, 30, 37, 53] based on Llama [46] or Llama2 [47] are developed. For example, LlamaRec [53] proposes a two-stage framework based on Llama2 to rerank the candidates retrieved by conventional models. To enable LLM to correctly identify items, E4SRec [28] incorporates ID embeddings trained by conventional sequential models through a linear adaptor, and applies LORA [18] for parameter-efficient fine-tuning. Cross-entropy and its approximations [2, 3, 14, 31, 40] have been extensively studied. The most related works are: 1) Bruch et al. [4] theoretically connected cross-entropy to some ranking metrics; 2) Wu et al. [49] further found its desirable property in alleviating popularity bias; and recently, 3) Klenitskiy et al. [24] and Petrov et al. [36] respectively applied cross-entropy and a generalized BCE loss to eliminate the performance gap between SASRec and BERT4Rec. Differently, we are to 1) understand the superiority of cross-entropy as well as the limitations of its approximations; 2) identify a viable approximation according to these findings; and 3) facilitate an objective evaluation of LLM-based recommendation by acknowledging the true capability of conventional models.
# 3 PRELIMINARIES
Given a query 𝑞that encompasses some user information, a recommender system aims to retrieve some items 𝑣∈I that would be of interest to the user. In sequential recommendation, the recommender predicts the next item 𝑣𝑡+1 based on historical interactions 𝑞= [𝑣1, 𝑣2, . . . , 𝑣𝑡]. The crucial component is to develop a scoring function 𝑠𝑞𝑣:= 𝑠𝜃(𝑞, 𝑣) to accurately model the relevance of a query 𝑞to a candidate item 𝑣. A common paradigm is to map them into the same latent space through some models parameterized via 𝜃, followed by an inner product operation for similarity calculation. Then, top-ranked items based on these scores will be prioritized for recommendation. Typically the desired recommender is trained to minimize an objective function over all observed interactions D:
# min 𝜃 E(𝑞,𝑣+)∼D [ℓ(𝑞, 𝑣+;𝜃)],
# where 𝑣+ indicates the target item for the query 𝑞, and the loss function ℓconsidered in this paper is in the form of
ℓ(𝑞, 𝑣+;𝜃) := −log exp(𝑠𝜃(𝑞, 𝑣+)) 𝑍𝜃(𝑞) = −𝑠𝜃(𝑞, 𝑣+) + log𝑍𝜃(𝑞). (1)
Here 𝑍𝜃(𝑞) depicts the ‘normalizing’ term specified to the query 𝑞. For clarity, we will omit 𝑞and 𝜃hereafter if no ambiguity is raised. It is worth noting that the reformulation of Eq. (1) makes it easy to understand the subtle differences between a wide range of loss functions. Table 1 covers a selection of approximations: Binary Cross-Entropy (BCE) and Bayesian Personalized Ranking (BPR) [40] are widely used in recommendation for their low costs; Importance Sampling (IS) [2, 20], Noise Contrastive Estimation (NCE) [14, 34], and NEGative sampling (NEG) [33] are the cornerstones of the subsequent methods [3, 6, 11, 29, 43, 51, 52]. Additional details regarding their mechanisms are provided in Appendix B. Note that in this work we only involve some elementary approximations, as the primary purpose is not to develop a complex loss. For more advanced loss functions, please refer to [5].
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ba49/ba4971a9-a1a5-4a08-9303-9a9d50de1926.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Performance comparison based on tighter bounds for NDCG. The dashed line represents the results trained by CE (namely the case of 𝜂→+∞).</div>
# 4 THE ROLE OF CROSS-ENTROPY LOSS IN OPTIMIZING RANKING CAPABILITY
BCE and BPR are commonly employed in training recommender systems due to their high efficiency. However, the substantial performance gaps shown in Figure 1 suggest their poor alignment with cross-entropy. Needless to say, claims based on the comparison with these inferior alternatives are unconvincing. In this section, we are to showcase the substitutability of cross-entropy by 1) highlighting the importance of implicitly optimizing the ranking metrics for a recommendation loss; 2) introducing some practical modifications to boost the effectiveness of some elementary approximations. Due to space constraints, the corresponding proofs are deferred to Appendix C. SASRec [21], one of the most prominent sequential models, will serve as the baseline to empirically elucidate the conclusions in this part. All results are summarized based on 5 independent runs.
# 4.1 Cross-Entropy for Some Ranking Metrics
The capability to prioritize items aligning with the user’s interests is essential for recommender systems. Denoted by 𝑟+ := 𝑟(𝑣+) = |{𝑣∈ I : 𝑠𝑣≥𝑠𝑣+}| the predicted rank of the target item 𝑣+, the metric of Normalized Discounted Cumulative Gain (NDCG)1 is often employed to assess the sorting quality. For next-item recommendation considered in this paper, NDCG is simplified to
( +) It increases as the target item 𝑣+ is ranked higher, and reaches the maximum when 𝑣+ is ranked first (i.e., 𝑟+ = 1). Consequently, the average quality computed over the entire test set serves as an indicator of the ranking capability. Notably, Reciprocal Rank (RR) is another popular ranking metric, and we leave the definition and results in Appendix since the corresponding findings are very similar to those of NDCG. The following proposition suggests that cross-entropy is a soft proxy to these ranking metrics.
Proposition 4.1. For a target item 𝑣+ which is ranked as 𝑟+, the following inequality holds true for any 𝑛≥𝑟+
(2)
1In practice, it is deemed meaningless when 𝑟+ exceeds a pre-specified threshold 𝑘 (e.g., 𝑘= 1, 5, 10). Hence, the widely adopted NDCG@𝑘metric is modified to assign zero reward to these poor ranking results.
1In practice, it is deemed meaningless when 𝑟+ exceeds a pre-specified threshold 𝑘 (e.g., 𝑘= 1, 5, 10). Hence, the widely adopted NDCG@𝑘metric is modified to assign zero reward to these poor ranking results.
<div style="text-align: center;">able 1: Cross-entropy loss and its approximations. The bounding probabilities are obtained in the case of uniform sampling ore conclusions in terms of Reciprocal Rank (RR) can be found in Appendix C.</div>
Table 1: Cross-entropy loss and its approximations. The bounding probabilities are obtained in the ca More conclusions in terms of Reciprocal Rank (RR) can be found in Appendix C.
Loss
Formulation
‘Normalizing’ term 𝑍
Complexity
P�−log NDCG(𝑟+) ≤ℓ∗
�≥
ℓCE
−log
exp(𝑠𝑣+ )
�
𝑣∈I exp(𝑠𝑣)
�
𝑣∈I exp(𝑠𝑣)
O(|I|𝑑)
1
ℓBCE
−log𝜎(𝑠𝑣+) −log(1 −𝜎(𝑠𝑣−))
(1 + exp(𝑠𝑣+))(1 + exp(𝑠𝑣−))
O(𝑑)
-
ℓBPR
−log𝜎(𝑠𝑣+ −𝑠𝑣−)
exp(𝑠𝑣+) + exp(𝑠𝑣−)
O(𝑑)
-
ℓNCE
−log𝜎(𝑠′𝑣+) −�𝐾
𝑖=1 log(1 −𝜎(𝑠′𝑣𝑖))
(1 + exp(𝑠′𝑣+)) �𝐾
𝑖=1(1 + exp(𝑠′𝑣𝑖))
O(𝐾𝑑)
1 −𝑚�1 −|S′+|/|I|�⌊𝐾/𝑚⌋
ℓNEG
−log𝜎(𝑠𝑣+) −�𝐾
𝑖=1 log(1 −𝜎(𝑠𝑣𝑖))
(1 + exp(𝑠𝑣+)) �𝐾
𝑖=1(1 + exp(𝑠𝑣𝑖))
O(𝐾𝑑)
1 −𝑚�1 −|S+|/|I|�⌊𝐾/𝑚⌋
ℓIS
−log
exp(𝑠𝑣+ −log𝑄(𝑣+))
�𝐾
𝑖=1 exp(𝑠𝑣𝑖−log𝑄(𝑣𝑖))
�𝐾
𝑖=1 exp(𝑠𝑣𝑖−log𝑄(𝑣𝑖) + log𝑄(𝑣+))
O(𝐾𝑑)
1 −2𝑚�1 −𝑟+/|I|�⌊𝐾/2𝑚⌋
ℓSCE
−log
exp(𝑠𝑣+ )
exp(𝑠𝑣+ )+𝛼�𝐾
𝑖=1 exp(𝑠𝑣𝑖)
exp(𝑠𝑣+) + 𝛼�𝐾
𝑖=1 exp(𝑠𝑣𝑖)
O(𝐾𝑑)
1 −1
𝛼2𝑚�1 −𝑟+/|I|�⌊𝛼𝐾/2𝑚⌋
We can draw from Proposition 4.1 that −log NDCG(𝑟+) would be strictly bounded by CE-like losses, as long as all items ranked before 𝑣+ are retained in the normalizing term. In other words, minimizing these CE-like losses is equivalent to maximizing a lower bound of NDCG. Because cross-entropy is a special case that retains all items (i.e., 𝑛= |I|), we readily have the following corollary:
Corollary 4.2 ([4]). Minimizing the cross-entropy loss ℓCE is equivalent to maximizing a lower bound of normalized discounted cumulative gain.
Therefore, satisfactory ranking capability can be expected if ℓCE for all queries are minimized. Since the superiority of cross-entropy possibly stems from its connection to some ranking metrics, one may hypothesize that optimizing a tighter bound with a smaller value of 𝑛≪|I| allows greater performance gains. However, the condition 𝑛≥𝑟+ cannot be consistently satisfied via a constant value of 𝑛since 𝑟+ dynamically changes during training. Alternatively, an adaptive truncation can be employed for this purpose: ∑︁
(3)
Note that this 𝜂-truncated loss retains only items whose scores are not lower than 𝑠+ −𝜂|𝑠+|, so a tighter bound will be obtained as 𝜂 drops to 0. Specifically, this 𝜂-truncated loss becomes ℓCE-𝑟+ (i.e., the tightest case) when 𝜂= 0, and approaches ℓCE when 𝜂→+∞. Figure 2 illustrates how NDCG@10 varies as 𝜂gradually increases from 0.1 to 5. There are two key observations: 1. SASRec performs worst in the tightest case of 𝜂≈0. This can be attributed to the instability of the 𝜂-truncated loss. On the one hand, ℓCE-𝜂will rapidly collapse to 0 for those easily recognized targets, in which case all other items are excluded in the normalizing term except the target itself. On the other hand, due to this strict truncation, only a few negative items are encountered during training, and thus over-fitting is more likely to occur. 2. Once 𝜂is large enough to overcome the training instability, SASRec begins to enjoy the benefits from tightness and achieves its best performance around 𝜂≈0.7. Further increasing 𝜂however
leads to a similar effect as cross-entropy, thereby along with a slightly degenerate performance due to the suboptimal tightness. In spite of the minor performance gains, the complexity of these tighter bounds is still equal to or even higher than that of cross entropy. The challenge that remains unsolved is how to realize the approximation in an effective and practical manner. To this end, we will introduce two practical alternatives to cross-entropy, one based on noise contrastive estimation [14, 34], and the other based on sampled softmax loss [49]. Their different ways of approximating the cross-entropy loss lead to distinct properties during training. Some specific modifications focusing on the normalizing term estimates are then developed to enhance their ability to optimize NDCG and RR.
# 4.2 Revisiting Noise Contrastive Estimation
Noise Contrastive Estimation (NCE) is widely used in training neural language models for bypassing an explicit normalizing over the entire vocabulary. It requires the model to discriminate the target from an easy-to-sample noise distribution. In the case of the uniform sampling, it can be formulated as follows � �
� � � � where 𝑠′𝑣= 𝑠𝑣−𝑐−log 𝐾 |I| . In the original implementation of NCE [14], 𝑐is a trainable parameter as an estimate of log𝑍CE. However, this strategy is infeasible for conditional probability models like language models and sequential recommendation, where they need to determine one 𝑐𝑞for each text (query). Mnih et al. [34] therefore fixed 𝑐≡1 for all texts during training, and NEGative sampling (NEG) used in Word2Vec [33] further simplifies it by replacing 𝑠′ with 𝑠directly; that is, � �
�� �� However, we observe that both NCE (𝑐= 1) and NEG introduce training difficulties as the number of negative samples increases. The NDCG@10 metric shown in Figure 3 remains almost constant at the beginning of training, and consumes more iterations to converge as 𝐾increases. This contradicts the usual understanding
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f3b3/f3b35129-dce4-43bb-bcb2-11e295057171.png" style="width: 50%;"></div>
<div style="text-align: center;">CG@10 performance of NCE and NEG across different number of negati</div>
(4)
we have
# we have
(5)
From Theorem 4.3, we have the following conclusions: 1. Notice that Eq. (4) for NCE (NEG) is mainly determined by the size of |𝑆′+| (|𝑆+|) rather than the current rank 𝑟+. As a result, NCE and NEG can easily bound NDCG as long as the number of items with non-negative scores is large enough. This is more common in the early stages of training, in which case item representations are poorly distributed in the latent space. In other words, the bounds at the beginning are too weak to be meaningful for improving the model ranking capability. The so-called training difficulties are actually a stage in narrowing the gap. 2. Figure 3 also suggests that the standstill duration of NCE is significantly longer than that of NEG, for example, 150 epochs versus 70 epochs on Beauty if 500 negative samples are sampled for training. Note that NEG can be regarded as a special case of NCE by fixing 𝑐= log(|I|/𝐾), a value higher than 1 if the experimental settings described in Figure 3 are applied. As such, according to Theorem 4.3, NCE with 𝑐= 1 will suffer from a weaker bound than NEG, thereby more iterations are required for convergence. Overall, NCE and NEG converge slower as 𝐾increases because of the exponential growth of their normalizing terms w.r.t. the sizes of |S′+| and |S+|. One feasible modification is to adopt a moderately large𝑐so that the sizes remain acceptable even if numerous negative items are sampled. As shown in Table 2, the number of epochs required for convergence decreases as the value of 𝑐increases from 1 to 10. But a larger value once again hinders the training process.
<div style="text-align: center;">(b) MovieLens-1M</div>
Table 2: The convergence epoch and the highest NDCG@10 metric achieved among the 500 epochs (with 𝐾= 500).
<div style="text-align: center;">Table 2: The convergence epoch and the highest NDCG@1 metric achieved among the 500 epochs (with 𝐾= 500).</div>
Beauty
MovieLens-1M
NDCG@10
Epoch
NDCG@10
Epoch
NCE (𝑐= 1)
0.0547
400-470
0.1817
280-440
NCE (𝑐= 5)
0.0545
195-280
0.1812
100-200
NCE (𝑐= 10)
0.0571
95-115
0.1817
80-160
NCE (𝑐= 50)
0.0410
≥500
0.1816
≥420
NCE (𝑐= 100)
0.0171
≥500
0.1676
≥500
Recall that 𝑐is an estimate of log𝑍CE. Setting 𝑐≥50 implies a hypothesis of 𝑍CE ≥𝑒50, which is obviously difficult to achieve for most models. As a rule of thumb, the hyper-parameter 𝑐should be chosen carefully, and 𝑐= 10 appears a good choice according to the experiments in Section 5. Next we will introduce a more reliable variant of the sampled softmax loss [49]. As opposed to NCE and NEG, it yields a bound determined by the current rank 𝑟+.
# 4.3 Scaling Up the Sampled Normalizing Term
Since the normalizing term of cross-entropy is intractable in reality, a direct way is to approximate it by (uniformly) sampling part of items from I:
The resulting Scaled Cross-Entropy (SCE) becomes
# The resulting Scaled Cross-Entropy (SCE) becomes
ℓSCE := −𝑠𝑣+ + log ˆ𝑍(𝛼).
Note that here we scale up the sampled normalizing term by a prespecific weight 𝛼. For the case of 𝛼= 1, this approximation (a.k.a. sampled softmax loss [49]) implies a (𝐾+1)-class classification task, and has been used in previous studies [24, 49]. But it is worth noting that they have inherent differences. We modify it via a weight 𝛼to remedy the magnitude loss resulted from sampling, so the scaled loss is more likely to bound NDCG and RR:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3ebe/3ebe2b5e-593b-4c47-8ea3-4cf40e30f730.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: NDCG@10 performance under various weight 𝛼.</div>
Theorem 4.4. Under the same conditions as stated in Theorem 4.3, the inequality (5) holds for SCE with a probability of at least
(7)
�� In addition to the promising bounding probability achieved through the weight 𝛼, we can also observe from Theorem 4.4 that Eq. (7) is directly governed by the current rank 𝑟+. In contrast to NCE (𝑐= 1) and NEG, SCE is expected to be meaningful even in the early stages of training. Certainly, SCE is not perfect: the scaling operation inevitably raises concerns about the high variance problem. As depicted in Figure 4, if negative samples are very rare, a larger weight of 𝛼tends to worsen the ranking capability. Fortunately, the high variance problem appears less significant as 𝐾slightly increases (e.g., 𝐾≥50 for Beauty and 𝐾≥10 for MovieLens-1M). Notably, sampling 100 negative samples for 𝛼= 100 produces comparable performance to using 500 negative samples for 𝛼= 1 on the Beauty dataset. Connection to importance sampling. While SCE does not make sense at first glance, it is indeed closely related to importance sampling [2, 41], a widely used technique for cross-entropy approximation. Given a proposal distribution 𝑄over all items, it corrects the approximation as follows
𝑣𝑖∼𝑄, 𝑖= 1, 2, . . . , 𝐾.
� It allows for an unbiased estimation if the proposal distribution 𝑄 is precisely identical to the underlying data distribution. But for the conditional probability model like sequential recommendation, achieving the optimum requires additional overhead for each query. Hence, some heuristic designs based on popularity sampling [6, 29] are adopted more often in practice. We point out that SCE is morphologically equivalent to these designs with �
(8)
Consequently scaling up the sampled normalizing term can be considered as assigning a higher probability mass to the target item 𝑣+. This partially explains why SCE is effective: a well-trained scoring function should skew towards the target item. Another subtle difference is that the normalizing term for importance sampling may not include the target term, while SCE always preserves it. This is practically necessary to avoid an unstable training process.
<div style="text-align: center;">Table 3: Dataset statistics.</div>
Dataset
#Users
#Items
#Interactions
Density
Avg. Length
Beauty
22,363
12,101
198,502
0.07%
8.9
MovieLens-1M
6,040
3,416
999,611
4.84%
165.5
Yelp
30,431
20,033
316,354
0.05%
10.4
Table 4: Model statistics. The number of parameters is estimated based on the Beauty dataset.
Model
Foundation Model
Architecture
Embedding Size
#Params
P5(CID+IID) [19]
T5
Transformer
512
60M
POD [27]
T5
Transformer
512
60M
LlamaRec [53]
Llama2
Transformer
4096
7B
E4SRec [28]
Llama2
Transformer
4096
7B
GRU4Rec [17]
-
RNN
64
0.80M
Caser [45]
-
CNN
64
3.80M
SASRec [21]
-
Transformer
64
0.83M
BERT4Rec [44]
-
Transformer
64
1.76M
FMLP-Rec [56]
-
MLP
64
0.92M
# 4.4 Computational Complexity
The major cost of cross-entropy lies in the inner product and softmax operations. It has a complexity of O(|I|𝑑), where 𝑑denotes the embedding size before similarity calculation. In contrast, the approximations require a lower cost O(𝐾𝑑), correspondingly an additional overhead O(𝐾) for uniform sampling. Overall, it is profitable if 𝐾≪|I|.
# 5 EXPERIMENTS
In this section, we are to reveal the true ranking capability of conventional recommenders by using the modified Noise Contrastive Estimation (NCE) and the proposed Scaled Cross-Entropy (SCE). Thus, the current advancements made by LLM-based recommenders can also be assessed objectively.
# 5.1 Experimental Setup
This part introduces the datasets, evaluation metrics, baselines, and implementation details.
This part introduces the datasets, evaluation metrics, baselines, and implementation details. Datasets. To ensure the reliability of the conclusions, we select three public datasets from different scenarios, including the Beauty, MovieLens-1M, and Yelp datasets. Beauty is an e-commerce dataset extracted from Amazon reviews known for high sparsity; MovieLens-1M is a movie dataset with a much longer sequence length; Yelp collects abundant meta-data suitable for multi-task training. Following [12, 56], we filter out users and items with less than 5 interactions, and the validation set and test set are split in a leave-one-out fashion, namely the last interaction for testing and the penultimate one for validation. The dataset statistics are presented in Table 3. Evaluation metrics. For each user, the scores returned by the recommender will be sorted in descending order to generate candidate lists. In addition to the aforementioned NDCG@𝑘(Normalized Discounted Cumulative Gain), HR@𝑘(Hit Rate) that quantifies the proportion of successful hits among the top-𝑘recommended candidates will also be included in this paper due to its widespread
<div style="text-align: center;">Table 5: Overall performance comparison on the Beauty, MovieLens-1M, and Yelp datasets. The best results of each block are marked in bold. ‘▲% over CE/LLM’ represents the relative gap between respective best results.</div>
Beauty
MovieLens-1M
Yelp
HR@5
HR@10
NDCG@5
NDCG@10
HR@5
HR@10
NDCG@5
NDCG@10
HR@5
HR@10
NDCG@5
NDCG@10
LLM
POD
0.0185
0.0245
0.0125
0.0146
0.0422
0.0528
0.0291
0.0326
0.0476
0.0564
0.0330
0.0358
P5 (CID+IID)
0.0569
0.0791
0.0403
0.0474
0.2225
0.3131
0.1570
0.1861
0.0289
0.0453
0.0200
0.0252
LlamaRec
0.0591
0.0862
0.0405
0.0492
0.1757
0.2836
0.1113
0.1461
0.0416
0.0605
0.0306
0.0367
E4SRec
0.0527
0.0753
0.0376
0.0448
0.1871
0.2765
0.1234
0.1522
0.0309
0.0473
0.0207
0.0260
CE
GRU4Rec
0.0474
0.0690
0.0329
0.0398
0.2247
0.3201
0.1542
0.1850
0.0275
0.0463
0.0171
0.0231
Caser
0.0435
0.0614
0.0303
0.0361
0.2181
0.3049
0.1520
0.1800
0.0283
0.0383
0.0211
0.0243
SASRec
0.0713
0.0986
0.0510
0.0597
0.2221
0.3131
0.1518
0.1812
0.0476
0.0696
0.0345
0.0415
BERT4Rec
0.0509
0.0747
0.0347
0.0423
0.1978
0.2922
0.1330
0.1634
0.0355
0.0540
0.0243
0.0303
FMLP-Rec
0.0717
0.0988
0.0507
0.0594
0.2287
0.3243
0.1585
0.1893
0.0512
0.0759
0.0364
0.0444
▲% over LLM
21.4%
14.6%
25.7%
21.3%
2.8%
3.6%
0.9%
1.7%
7.5%
25.6%
10.3%
21.0%
BCE
GRU4Rec
0.0214
0.0376
0.0134
0.0186
0.1595
0.2490
0.1023
0.1310
0.0157
0.0273
0.0098
0.0135
Caser
0.0282
0.0434
0.0185
0.0234
0.1639
0.2476
0.1078
0.1348
0.0304
0.0428
0.0224
0.0264
SASRec
0.0429
0.0671
0.0275
0.0353
0.1594
0.2492
0.1040
0.1329
0.0325
0.0501
0.0225
0.0281
BERT4Rec
0.0245
0.0415
0.0152
0.0207
0.1241
0.2021
0.0789
0.1039
0.0223
0.0379
0.0138
0.0188
FMLP-Rec
0.0460
0.0710
0.0301
0.0381
0.1800
0.2722
0.1173
0.1469
0.0460
0.0651
0.0330
0.0391
▲% over CE
-35.9%
-28.1%
-41.0%
-36.2%
-21.3%
-16.1%
-26.0%
-22.4%
-10.0%
-14.3%
-9.4%
-11.9%
▲% over LLM
-22.1%
-17.7%
-25.8%
-22.6%
-19.1%
-13.1%
-25.3%
-21.1%
-3.3%
7.6%
-0.1%
6.5%
NCE
GRU4Rec
0.0434
0.0652
0.0288
0.0359
0.2273
0.3184
0.1541
0.1834
0.0241
0.0418
0.0148
0.0205
Caser
0.0377
0.0567
0.0253
0.0314
0.2213
0.3106
0.1523
0.1811
0.0296
0.0405
0.0220
0.0255
SASRec
0.0686
0.0961
0.0485
0.0573
0.2177
0.3135
0.1479
0.1788
0.0471
0.0682
0.0344
0.0412
BERT4Rec
0.0487
0.0734
0.0324
0.0404
0.1960
0.2933
0.1311
0.1624
0.0389
0.0574
0.0271
0.0330
FMLP-Rec
0.0693
0.0964
0.0491
0.0578
0.2291
0.3279
0.1567
0.1885
0.0512
0.0760
0.0364
0.0444
▲% over CE
-3.4%
-2.4%
-3.6%
-3.2%
0.2%
1.1%
-1.1%
-0.4%
0.0%
0.1%
0.0%
0.1%
▲% over LLM
17.3%
11.8%
21.3%
17.4%
2.9%
4.7%
-0.2%
1.3%
7.5%
25.7%
10.3%
21.1%
SCE
GRU4Rec
0.0489
0.0694
0.0344
0.0410
0.2309
0.3248
0.1587
0.1891
0.0290
0.0487
0.0183
0.0246
Caser
0.0456
0.0628
0.0322
0.0377
0.2274
0.3135
0.1586
0.1864
0.0293
0.0404
0.0218
0.0253
SASRec
0.0698
0.0968
0.0500
0.0587
0.2273
0.3186
0.1567
0.1862
0.0472
0.0693
0.0339
0.0410
BERT4Rec
0.0540
0.0776
0.0372
0.0449
0.2078
0.3014
0.1405
0.1707
0.0414
0.0612
0.0283
0.0346
FMLP-Rec
0.0703
0.0979
0.0502
0.0591
0.2372
0.3284
0.1648
0.1942
0.0517
0.0779
0.0357
0.0441
▲% over CE
-2.0%
-0.9%
-1.5%
-1.1%
3.7%
1.3%
4.0%
2.6%
1.0%
2.6%
-2.0%
-0.6%
▲% over LLM
19.0%
13.6%
23.8%
20.0%
6.6%
4.9%
5.0%
4.4%
8.6%
28.8%
8.1%
20.2%
use in other studies [12, 21, 56]. Besides, we also provide the Mean Reciprocal Rank (MRR) results in Appendix D. Baselines. Although LLM itself has surprising zero-shot recommendation ability, there still exist non-negligible gaps [1, 22] unless domain-specific knowledge is injected. Hence, only LLMbased recommenders enhanced by fine-tuning will be compared in this paper: P5 (CID+IID) [19], POD [27], LlamaRec [53], and E4SRec [28]. Specifically, the first two methods take T5 as the foundation model, while the last two methods fine-tune Llama2 for efficient sequential recommendation. Additionally, five sequential models including FMLP-Rec [56], Caser [45], GRU4Rec [17], SASRec [21], and BERT4Rec [44], are considered here to unveil the true capability of conventional methods. They cover various architectures so as to comprehensively validate the effectiveness of the proposed approximation methods. Table 4 presents an overview of the model statistics. Notice that for LlamaRec and E4SRec we employ Llama2-7B instead of Llama2-13B as the foundation model in order to improve training efficiency. This results in minor performance differences in practice. Implementation details. According to the discussion in Section 4, we set 𝑐= 10 for NCE and 𝛼= 100 for SCE, and less than 5% of all items will be sampled for both approximation objectives.
Specifically, 𝐾= 500 on the Beauty dataset, and 𝐾= 100 on the MovieLens-1M and Yelp datasets. We find that training 200 epochs is sufficient for cross-entropy to converge, while sometimes NCE and SCE need 300 epochs. Other hyper-parameters of NCE and SCE completely follow cross-entropy. Because the data preprocessing scripts provided with POD may lead to information leakage [39], we assign random integer IDs to items rather than sequentially incrementing integer IDs. The maximum sequence length and embedding size have a direct impact on representation capability and inference efficiency, so we will discuss them separately in Section 5.3.
# 5.2 Overall Performance Evaluation
In this part, we fulfill our primary objective by presenting the comparison between LLM-based recommenders and conventional recommenders. Considering the expensive training cost of LLMbased models, their results reported in Table 5 are based on a single run, while the results of other conventional methods are averaged over 5 independent runs. Conventional methods using cross-entropy outperform LLM-based recommenders. Let us focus on the first three blocks in Table 5, where the use of cross-entropy greatly improves the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/13c2/13c29147-0373-493b-b863-0ff9dc5b7615.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) NDCG@10</div>
<div style="text-align: center;">Figure 5: Relative gaps between SCE and NCE.</div>
conventional methods’ recommendation performance. In particular, SASRec and FMLP-Rec demonstrate superior performance compared to LLMs, but fall significantly behind if replacing crossentropy with BCE. Hence, previous affirmative arguments about the LLMs’ recommendation performance are rooted in unobjective and unfair comparisons, wherein BCE or BPR are commonly used for training conventional models. Moreover, the inflated model size (from 60M of P5 to 7B of LlamaRec) only yields negligible improvements in some of the metrics. The rich world knowledge and powerful reasoning ability seem to be of limited use here due to the emphasis on personalization in sequential recommendation. In conclusion, despite after fine-tuning, LLM-based recommenders still fail to surpass state-of-the-art conventional models. Comparable effectiveness can be achieved using practical approximations. Conducting a full softmax over all items for cross-entropy may be infeasible in practice. Fortunately, the last two blocks in Table 5 shows the substitutability of cross-entropy by applying the modified NCE or the proposed SCE. To showcase the effectiveness of these substitutes, we intentionally sample a rather conservative number of negative samples, and thus there remains a slight gap compared to cross-entropy. Nevertheless, the superior results of NCE and SCE re-emphasize the clear gap that exists between LLM-based and conventional recommenders. SCE is more consistent and reliable than NCE. As discussed in Section 4.2, NCE is sensitive to the choice of 𝑐: extremely small or large values might impede learning and degrade performance. The inconsistent performance gains from NCE can also verify this conclusion. Figure 5 clearly demonstrates that NCE can contribute competitive performance to SASRec and FMLP-Rec, but underperforms SCE across a variety of models (e.g., GRU4Rec and BERT4Rec) and datasets (e.g., Beauty and MovieLens-1M). For Caser on Beauty and GRU4Rec on Yelp, replacing SCE with NCE results in a performance degradation of even ≥15%.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d58e/d58eea4f-aee2-4681-8b15-7e3f23eb5d64.png" style="width: 50%;"></div>
Figure 6: P5 (CID+IID) versus SASRec(+) across various maximum sequence length 𝐿. SASRec+ uses a sliding window similar to P5 to augment each sequence.
<div style="text-align: center;">Figure 6: P5 (CID+IID) versus SASRec(+) across various maximum sequence length 𝐿. SASRec+ uses a sliding window similar to P5 to augment each sequence.</div>
<div style="text-align: center;">Table 6: The impact of embedding size 𝑑.</div>
Beauty
MovieLens-1M
𝑑
HR@10
NDCG@10
HR@10
NDCG@10
P5(CID+IID)
512
0.0791
0.0474
0.3131
0.1861
SASRec+
64
0.0937
0.0574
0.3271
0.1915
SASRec+
512
0.0963
0.0589
0.3360
0.1999
Additional experiments on BPR loss and MRR metrics are shown in the Appendix. They draw the same conclusions as above.
# 5.3 Other Factors for Objective Evaluation
Due to the difference in model design, it is challenging to conduct evaluations on a completely identical testbed. To clarify the reliability of the results in Table 5, we further investigate two key factors: maximum sequence length2 𝐿and embedding size 𝑑. According to the conclusions above, SCE is employed to train SASRec in the following. Results for CE and NCE can be found in Appendix E. Maximum sequence length. Following the routine of previous studies [21, 44], conventional models like SASRec are allowed to make predictions based on the last 𝐿= 200 interactions on MovieLens-1M and 𝐿= 50 on other datasets. In contrast, LLMbased recommenders are confined to 𝐿= 20 on all three datasets for training efficiency. We argue that this difference does not make the conclusion differ because as can be seen in Figure 6, P5 does not exhibit much better performance with access to more historical interactions. Notably, SASRec’s performance is stable on Beauty, but on MovieLens-1M, it deteriorates significantly when 𝐿gets smaller. This phenomenon primarily arises from the fact that the original implementation of SASRec only utilizes the most recent 𝐿interactions for training, whereas for P5 each sequence is divided into multiple segments. Consequently, P5 is able to access far more interactions than SASRec during training, especially on the MovieLens-1M dataset known for long sequence length. If we apply a similar strategy that augments each sequence using a sliding window, the resulting SASRec+ then performs consistently across diverse 𝐿.
2The maximum sequence length refers to the maximum number of historical interactions used for next-item prediction. Other tokens like prompts are not included.
nding the Role of Cross-Entropy Loss in Fairly Evaluating Large Language Model-based Recommend
Embedding size. Models with a larger embedding size have stronger representation capability and thus potentially better recommendation performance. According to the discussion above, we examine its impact under the same setting of 𝐿= 20. In Table 6, when the embedding size of SASRec+ is increased from 64 to 512, the obtained performance gains are marginal. In view of its costly overhead, such improvement is not attractive in practice. This also implies that existing LLM-based recommenders are over-parameterized in terms of ranking capability.
In this work, we bridge the theoretical and empirical performance gaps between cross-entropy and some of its approximations through a modified noise contrastive estimation loss and an effective scaled cross-entropy loss. Based on these practical approximations, we showcase that existing LLM-based recommenders are not as effective as claimed. The innovative understandings and extensive experiments can be expected to facilitate an objective evaluation of LLM-based recommendation in the future.
# REFERENCES
[1] Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. TALLRec: An effective and efficient tuning framework to align large language model with recommendation. In ACM Conference on Recommender Systems (RecSys). ACM, 1007–1014. [2] Yoshua Bengio and Jean-Sébastien Senécal. 2008. Adaptive importance sampling to accelerate training of a neural probabilistic language model. IEEE Transactions on Neural Networks (TNNLS) 19, 4 (2008), 713–722. [3] Guy Blanc and Steffen Rendle. 2018. Adaptive sampled softmax with kernel based sampling. In International Conference on Machine Learning (ICML) (Proceedings of Machine Learning Research, Vol. 80). PMLR, 589–598. [4] Sebastian Bruch, Xuanhui Wang, Michael Bendersky, and Marc Najork. 2019. An analysis of the softmax cross entropy loss for learning-to-rank with binary relevance. In ACM SIGIR International Conference on Theory of Information Retrieval (ICTIR). ACM, 75–78. [5] Chong Chen, Weizhi Ma, Min Zhang, Chenyang Wang, Yiqun Liu, and Shaoping Ma. 2023. Revisiting negative sampling vs. non-sampling in implicit recommendation. ACM Transactions on Information Systems (TOIS) 41, 1 (2023), 1–25. [6] Jin Chen, Defu Lian, Binbin Jin, Kai Zheng, and Enhong Chen. 2022. Learning recommenders for implicit feedback with importance resampling. In ACM Web Conference (WWW). ACM, 1997–2005. [7] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering ChatGPT’s Capabilities in Recommender Systems. In ACM Conference on Recommender Systems (RecSys). ACM, 1126–1132. [8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT). Association for Computational Linguistics, 4171–4186. [9] Yue Feng, Shuchang Liu, Zhenghai Xue, Qingpeng Cai, Lantao Hu, Peng Jiang, Kun Gai, and Fei Sun. 2023. A large language model enhanced conversational recommender system. arXiv preprint arXiv:2308.06212 (2023). [10] Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang. 2023. Chat-rec: Towards interactive and explainable llms-augmented recommender system. arXiv preprint arXiv:2303.14524 (2023). [11] Yingbo Gao, David Thulke, Alexander Gerstenberger, Khoa Viet Tran, Ralf Schlüter, and Hermann Ney. 2021. On sampling-based training criteria for neural language modeling. In Annual Conference of the International Speech Communication Association (Interspeech). ISCA, 1877–1881. [12] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as language processing (RLP): A unified pretrain, personalized prompt & predict paradigm (P5). In ACM Conference on Recommender Systems (RecSys). ACM, 299–315. [13] Shansan Gong and Kenny Q. Zhu. 2022. Positive, negative and neutral: Modeling implicit feedback in session-based news recommendation. In International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 1185–1195. [14] Michael Gutmann and Aapo Hyvärinen. 2010. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In International Conference on Artificial Intelligence and Statistics (AISTATS). JMLR Workshop and Conference Proceedings, 297–304. [15] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-Dong Zhang, and Meng Wang. 2020. LightGCN: Simplifying and powering graph convolution network for recommendation. In International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR). ACM, 639–648. [16] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In International Conference on World Wide Web (WWW). ACM, 173–182. [17] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based recommendations with recurrent neural networks. In International Conference on Learning Representations (ICLR). [18] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR). [19] Wenyue Hua, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. 2023. How to index item ids for recommendation foundation models. In Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP). ACM, 195–204. [20] Sébastien Jean, KyungHyun Cho, Roland Memisevic, and Yoshua Bengio. 2015. On using very large target vocabulary for neural machine translation. In Association for Computational Linguistics (ACL). Association for Computer Linguistics, 1–10. [21] Wang-Cheng Kang and Julian J. McAuley. 2018. Self-attentive sequential recommendation. In IEEE International Conference on Data Mining (ICDM). IEEE Computer Society, 197–206.
[22] Wang-Cheng Kang, Jianmo Ni, Nikhil Mehta, Maheswaran Sathiamoorthy, Lichan Hong, Ed Chi, and Derek Zhiyuan Cheng. 2023. Do LLMs understand user preferences? Evaluating llms on user rating prediction. arXiv preprint arXiv:2305.06474 (2023). [23] Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In International Conference on Learning Representations (ICLR). [24] Anton Klenitskiy and Alexey Vasilev. 2023. Turning dross into gold loss: is BERT4Rec really better than SASRec?. In ACM Conference on Recommender Systems (RecSys). ACM, 1120–1125. [25] Taku Kudo. 2018. Subword regularization: Improving neural network translation models with multiple subword candidates. In Annual Meeting of the Association for Computational Linguistics (ACL). Association for Computational Linguistics, 66–75. [26] Jiacheng Li, Yujie Wang, and Julian J. McAuley. 2020. Time interval aware selfattention for sequential recommendation. In International Conference on Web Search and Data Mining (WSDM). ACM, 322–330. [27] Lei Li, Yongfeng Zhang, and Li Chen. 2023. Prompt distillation for efficient llm-based recommendation. In ACM International Conference on Information and Knowledge Management (CIKM). ACM, 1348–1357. [28] Xinhang Li, Chong Chen, Xiangyu Zhao, Yong Zhang, and Chunxiao Xing. 2023. E4SRec: An elegant effective efficient extensible solution of large language models for sequential recommendation. arXiv preprint arXiv:2312.02443 (2023). [29] Defu Lian, Qi Liu, and Enhong Chen. 2020. Personalized ranking with importance sampling. In ACM Web Conference (WWW). ACM / IW3C2, 1093–1103. [30] Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, Xiang Wang, and Xiangnan He. 2023. LLaRA: Aligning large language models with sequential recommenders. arXiv preprint arXiv:2312.02445 (2023). [31] Zhuang Ma and Michael Collins. 2018. Noise contrastive estimation and negative sampling for conditional models: Consistency and statistical efficiency. In Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 3698–3707. [32] Kelong Mao, Jieming Zhu, Xi Xiao, Biao Lu, Zhaowei Wang, and Xiuqiang He. 2021. UltraGCN: Ultra simplification of graph convolutional networks for recommendation. In ACM International Conference on Information and Knowledge Management (CIKM). ACM, 1253–1262. [33] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. Advances in Neural Information Processing Systems (NeurIPS) 26 (2013), 3111–3119. [34] Andriy Mnih and Yee Whye Teh. 2012. A fast and simple algorithm for training neural probabilistic language models. In International Conference on Machine Learning (ICML). [35] OpenAI. 2023. GPT models documentation. https://platform.openai.com/docs/ models/overview. [36] Aleksandr Vladimirovich Petrov and Craig MacDonald. 2023. gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling. In Proceedings of the 17th ACM Conference on Recommender Systems, RecSys 2023, Singapore, Singapore, September 18-22, 2023. ACM, 116–128. [37] Junyan Qiu, Haitao Wang, Zhaolin Hong, Yiping Yang, Qiang Liu, and Xingxing Wang. 2023. ControlRec: Bridging the semantic gap between language model and personalized recommendation. arXiv preprint arXiv:2311.16441 (2023). [38] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research (JMLR) 21 (2020), 140:1–140:67. [39] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q Tran, Jonah Samost, et al. 2023. Recommender systems with generative retrieval. arXiv preprint arXiv:2305.05065 (2023). [40] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian personalized ranking from implicit feedback. In Conference on Uncertainty in Artificial Intelligence (UAI). AUAI Press, 452–461. [41] Christian P Robert, George Casella, and George Casella. 1999. Monte Carlo statistical methods. Vol. 2. Springer. [42] Guy Shani, David Heckerman, Ronen I Brafman, and Craig Boutilier. 2005. An mdp-based recommender system. Journal of Machine Learning Research (JMLR) 6, 9 (2005), 1265–1295. [43] Wentao Shi, Jiawei Chen, Fuli Feng, Jizhi Zhang, Junkang Wu, Chongming Gao, and Xiangnan He. 2023. On the theories behind hard negative sampling for recommendation. In ACM Web Conference (WWW). ACM, 812–822. [44] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In ACM International Conference on Information and Knowledge Management (CIKM). ACM, 1441–1450. [45] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In ACM International Conference on Web Search and Data Mining (WSDM). ACM, 565–573.
[46] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and efficient foundation language models. CoRR abs/2302.13971 (2023). [47] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. CoRR abs/2307.09288 (2023). [48] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS). 5998–6008. [49] Jiancan Wu, Xiang Wang, Xingyu Gao, Jiawei Chen, Hongcheng Fu, Tianyu Qiu, and Xiangnan He. 2023. On the effectiveness of sampled softmax loss for item recommendation. ACM Transactions on Information Systems (TOIS) (2023). https://doi.org/10.1145/3637061 [50] Siyu Wu, Jun Wang, and Wei Zhang. 2024. Contrastive Personalized Exercise Recommendation With Reinforcement Learning. IEEE Transactions on Learning Technologies (TLT) 17 (2024), 691–703. [51] Ji Yang, Xinyang Yi, Derek Zhiyuan Cheng, Lichan Hong, Yang Li, Simon Xiaoming Wang, Taibai Xu, and Ed H. Chi. 2020. Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations. In ACM Web Conference (WWW). ACM / IW3C2, 441–447. [52] Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, and Ed H. Chi. 2019. Sampling-bias-corrected neural modeling for large corpus item recommendations. In ACM Conference on Recommender Systems (RecSys). ACM, 269–277. [53] Zhenrui Yue, Sara Rabhi, Gabriel de Souza Pereira Moreira, Dong Wang, and Even Oldridge. 2023. LlamaRec: Two-stage recommendation using large language models for ranking. arXiv preprint arXiv:2311.02089 (2023). [54] Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. 2023. Recommendation as instruction following: A large language model empowered recommendation approach. arXiv preprint arXiv:2305.07001 (2023). [55] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD). 1059–1068. [56] Kun Zhou, Hui Yu, Wayne Xin Zhao, and Ji-Rong Wen. 2022. Filter-enhanced MLP is all you need for sequential recommendation. In ACM Web Conference (WWW). ACM, 2388–2399.
# Contents
1 1 2 3
Abstract 1 Introduction 2 Related Work 3 Preliminaries 4 The Role of Cross-Entropy Loss in Optimizing Ranking Capability 4.1 Cross-Entropy for Some Ranking Metrics 4.2 Revisiting Noise Contrastive Estimation 4.3 Scaling Up the Sampled Normalizing Term 4.4 Computational Complexity 5 Experiments 5.1 Experimental Setup 5.2 Overall Performance Evaluation 5.3 Other Factors for Objective Evaluation 6 Conclusion References Contents A Experimental setup B Overview of Loss Function C Proofs C.1 Proof of Proposition 4.1 C.2 Proof of Theorem 4.3 and 4.4 C.3 Proof of Eq. (8) C.4 Proofs Regarding Reciprocal Rank (RR) D Empirical results on Reciprocal Rank (RR)
Maximum Sequence Length
# A EXPERIMENTAL SETUP
Baselines. Although LLM itself has surprising zero-shot recommendation ability, there still exist non-negligible gaps [1, 22] unless domain-specific knowledge is injected. Hence, only LLM-based recommenders enhanced by fine-tuning will be compared in this paper: • P5 (CID+IID) [19] unifies multiple tasks (e.g., sequential recommendation and rating prediction) into a sequence-to-sequence paradigm. The use of collaborative and independent indexing together creates LLM-compatible item IDs. • POD [27] bridges IDs and words by distilling long discrete prompts into a few continuous prompts. It also suggests a task-alternated training strategy for efficiency. • LlamaRec [53] aims to address the slow inference process caused by autoregressive generation. Given the candidates retrieved by conventional models, it subsequently reranks them based on the foundation model of Llama2. • E4SRec [28] incorporates ID embeddings trained by conventional sequential models through a linear adaptor, and applies LORA [18] for parameter-efficient fine-tuning. Additionally, five sequential models, covering MLP, CNN, RNN, and Transformer architectures, are considered here to uncover the true capability of conventional methods. • GRU4Rec [17] applies RNN to recommendation with specific modifications made to cope with data sparsity. • Caser [45] treats the embedding matrix as an ‘image’, and captures local patterns by utilizing conventional filters. • SASRec [21] and BERT4Rec [44] are two pioneering works equipped with unidirectional and bidirectional self-attention, respectively. By their nature, SASRec predicts the next item based on previously interacted items, while BERT4Rec is optimized through a cloze task. • FMLP-Rec [56] denoises item sequences in the frequency domain. Although FMLP-Rec consists solely of MLPs, it exhibits superior performance compared to Transformer-based models. An overview of the model statistics can be found in Table 4. Note that for LlamaRec and E4SRec we employ Llama2-7B instead of Llama2-13B as the foundation model for training efficiency. This results in minor performance differences. Implementation details. The implementation of LLM-based recommenders is due to their source code, while for conventional models the code is available at https://anonymous.4open.science/r/ 1025. Due to the difference in model design, it is challenging to conduct evaluations on a completely identical testbed. Therefore, we follow the routine of previous studies [21, 44], where the maximum sequence length 𝐿= 200 on MovieLens-1M and 𝐿= 50 on Beauty and Yelp. In contrast, for LLM-based models, 𝐿= 20 on all three datasets. Empirically, this difference does not make the conclusion differ. Training strategies. For SASRec and BERT4Rec, each item sequence is trained once per epoch. As a result, only the most
recent 𝐿historical interactions are accessed during the training process, which will negatively impact performance when the sequence lengths are typically longer than 𝐿. Some approaches such as LLM-based approaches and other conventional methods thus divides each sequence into multiple sub-sequences.
# B OVERVIEW OF LOSS FUNCTION
Cross-Entropy (CE), also known as the negative log-likelihood (NLL) loss, can be formulated as follows:
�������������������� This is also the de facto objective commonly used in the pre-training (fine-tuning) of LLMs. Binary Cross-Entropy (BCE). BCE samples one negative item 𝑣−for each target 𝑣+, which necessitates the recommender to possess excellent pointwise scoring capability:
������������������������������������������������������������������������ Here 𝜎: R →[0, 1] denotes the sigmoid function. Bayesian Personalized Ranking (BPR) [40] also samples one negative item 𝑣−for each target 𝑣+, but it intends to maximize the probability that 𝑣+ will be chosen in preference to 𝑣−:
������������������������������������������������ Importance Sampling (IS) [2, 20] is a widely used technique for CE approximation. It is capable of correcting the approximation error via a proposal distribution 𝑄:
���������������������������������������������������������������������������������������� In addition to uniform distribution, the distribution derived from popularity metrics [29] is also a commonly used choice. Noise Contrastive Estimation (NCE) [14, 34] requires the model to discriminate the target 𝑣+ from an easy-to-sample noise
distribution:
�������������������������������������������������������������������������������� In the case of uniform sampling, 𝑠′𝑣= 𝑠𝑣−𝑐−log 𝐾 |I| , where 𝑐is a trainable parameter as an estimate of log𝑍CE. NEGative sampling (NEG) [33] is a special case of NCE by fixing 𝑐= log |I| 𝐾:
# C PROOFS
This section completes the proofs regarding the connection between the aforementioned loss functions and ranking metrics. Before delving into the proofs, it is important to note that in this paper we focus on the metrics specifically for next-item recommendation as it is most consistent with LLM’s next-token generation feature.
# C.1 Proof of Proposition 4.1
Proposition C.1. For a target item 𝑣+ which is ranked as 𝑟+, the following inequality holds true for any 𝑛≥𝑟+
where
(10)
Proof. Notice that log2(1 + 𝑥) ≤𝑥holds true for any 𝑥≥1. Hence, we have
NDCG(𝑟+) = 1 log2(1 + 𝑟+) ≥1 𝑟+ = 1 1 + � 𝑣≠𝑣+ 𝛿(𝑠𝑣> 𝑠𝑣+) = 1 1 + � 𝑟(𝑣)<𝑟+ 𝛿(𝑠𝑣> 𝑠𝑣+) ≥ 1 1 + � 𝑟𝑣<𝑟+ exp(𝑠𝑣−𝑠𝑣+) = exp(𝑠𝑣+) exp(𝑠𝑣+) + � 𝑟(𝑣)<𝑟+ exp(𝑠𝑣) ≥ exp(𝑠𝑣+) � 𝑟(𝑣)≤𝑛exp(𝑠𝑣) ,
where 𝛿(condition) = 1 if the given condition is true else 0, and the second-to-last inequality holds because exp(𝑠𝑣−𝑠𝑣+) ≥1 when 𝑠𝑣> 𝑠𝑣+.
□
# C.2 Proof of Theorem 4.3 and 4.4
First, let us introduce some lemmas that give lower bounds of the loss functions. Lemma C.2. Let 𝜉1 be the number of sampled items with nonnegative corrected scores; that is,
First, let us introduce some lemmas that give lower bounds of the loss functions.
Lemma C.2. Let 𝜉1 be the number of sampled items with nonnegative corrected scores; that is, �� ��
(11)
Then, we have
  Proof. According to the definition of NCE, it follows that � �
Lemma C.3. Let 𝜉2 be the number of sampled items with nonnegative scores; that is, �� ��
(13)
Then, we have
(14)
Lemma C.4. Let 𝜉3 be the number of sampled items with scores not lower than that of the target; that is �� ��
(15)
Then, we have
Proof. According to the definition of SCE, it follows that
Lemma C.5. Let 𝜉4 be the number of sampled items with scores higher than that of the target; that is � �
(17)
Then, we have3
(18)
Proof. According to the definition of importance sampling, it follows that
□
Lemma C.6. Let 𝜉∼B(𝐾, 𝑝) denote a random variable representing the number of successes over 𝐾binomial trials with a probability of 𝑝. Then, we have
(19)
Proof. 4 Divide the 𝐾independent binomial trials into 𝑚disjoint groups, each containing at least ⌊𝐾/𝑚⌋trials. If 𝜉< 𝑚, then one of the groups must have no successes observed; formally, we have
(20)
(21)
(22)
Hence, the proof is completed by noting the fact that P(𝜉≥𝑚) = 1 −P(𝜉< 𝑚).
Hence, the proof is completed by noting the fact that
(23)
Theorem C.7. Let 𝑣+ be a target item which is ranked as 𝑟+ ≤ 22𝑚−1 for some 𝑚∈N, and S+ := {𝑣∈I : 𝑠𝑣≥0}, S′ + := {𝑣∈I : 𝑠′ 𝑣≥0}.
Theorem C.7. Let 𝑣+ be a target item which is ranked as 𝑟+ ≤ 22𝑚−1 for some 𝑚∈N, and
S+ := {𝑣∈I : 𝑠𝑣≥0}, S′ + := {𝑣∈I : 𝑠′ 𝑣≥0}.
4The proof follows from the response posted at the URL: https://math.stackexchange. com/questions/3626472/upper-bound-on-binomial-distribution.
If we uniformly sample 𝐾items for training, then with a probability of at least ��
(24)
 we have
−log NDCG(𝑟+) ≤ℓ∗.
(25)
Therefore, Eq. (25) holds true for NCE as long as 𝜉1 ≥𝑚. Formally, we have � ���
(26)
� ��� Also notice that uniformly sampling from I yields a probability of 𝑝= |S′+|/|I| such that the corrected score of the sampled item is non-negative. Therefore, based on Lemma C.6, we have ��
P�𝜉1 ≥𝑚�≥1 −𝑚(1 −|S′ +|/|I|)⌊𝐾/𝑚⌋.
(27)
�� Case 2. The proof of NEG is completely the same as that of NCE. Case 3. Analogously, Lemma C.4 implies that � �
(28)
(29)
(30)
� �� � �� Also notice that uniformly sampling from I yields a probability of 𝑝= 𝑟+/|I| such that the score of the sampled item is not lower than that of the target (i.e., the top-𝑟+ ranked items). Therefore, based on Lemma C.6, we have
(31) (32)
(31)
(32)
Case 3. The proof of importance sampling is similar to that of SCE. The proof has been completed now.
# C.3 Proof of Eq. (8)
Here, we show that ℓSCE is morphologically equivalent to ℓIS if �
Proof. Under the condition of Eq. (C.3), we have
ℓSCE
 � which is morphologically equivalent to ℓIS with 𝐾+ 1 items in the normalizing term.
# C.4 Proofs Regarding Reciprocal Rank (RR)
Reciprocal Rank (RR) is another popular metric used to measure the ranking capability, which is defined as follows
(33)
We provide some theoretical conclusions here and leave the empirical results in next section. Firstly, we connect RR to the crossentropy as Proposition 4.1 does for NDCG.
Proposition C.8. For a target item 𝑣+ which is ranked as 𝑟+, the following inequality holds true for any 𝑛≥𝑟+
(34)
where
Next, we establish a connection between RR and NCE, NEG, SCE, and importance sampling, similar to what Theorem C.7 does for NDCG. Theorem C.9. Let 𝑣+ be a target item which is ranked as 𝑟+ ≤2𝑚 for some 𝑚∈N, and S+ := {𝑣∈I : 𝑠𝑣≥0}, S′ + := {𝑣∈I : 𝑠′ 𝑣≥0}.
Next, we establish a connection between RR and NCE, NEG, SCE, and importance sampling, similar to what Theorem C.7 does for NDCG.
S+ := {𝑣∈I : 𝑠𝑣≥0}, S′ + := {𝑣∈I : 𝑠′ 𝑣≥0}.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0250/02505226-c90b-4a29-bef5-6320bb836a01.png" style="width: 50%;"></div>
Figure 7: Performance comparison based on tighter bounds for MRR. The dashed line represents the results trained by CE (namely the case of 𝜂→+∞).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f746/f746c23d-2db6-4835-9070-a8bac33d181a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: MRR@10 performance under various weight 𝛼.</div>
If we uniformly sample 𝐾items for training, then with a probability of at least  ��
 we have
� −log RR(𝑟+) ≤ℓ∗.
(36)
Proof. As 𝑟+ ≤2𝑚for some 𝑚∈N, we immediately have
The conclusions then can be proved in the same way as Theorem C.7. □
□
Remark 1. It is worth noting that the inequality for RR is achieved at a stricter condition compared to NDCG. For the same rank 𝑟+, NDCG allows for a smaller value of𝑚, thereby yielding slightly higher bounding probabilities. However, this nuance does not undermine the fact that the same conclusions can be drawn from the two metrics. The empirical observations in the next section can be used to verify this.
# D EMPIRICAL RESULTS ON RECIPROCAL RANK (RR)
# D EMPIRICAL RESULTS ON RECIPROCAL
In this part, we provide the empirical results on Reciprocal Rank (RR), which are very similar to those of NDCG. Note that Mean Reciprocal Rank (MRR) reported below represents the average performance over all users.
Table 7: MRR@5 and MRR@10 comparison