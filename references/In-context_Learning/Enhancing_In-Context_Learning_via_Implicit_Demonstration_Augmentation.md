# ontext Learning via Implicit Demonstration Au Deep feature distribution h   h  
Xiaoling Zhou1, Wei Ye1∗, Yidong Wang1, Chaoya Jiang1, Zhemg Lee2, Rui Xie1, Shikun Zhang1∗ 1National Engineering Research Center for Software Engineering, Peking University, China 2Tianjin University, Tianjin, China xiaolingzhou@stu.pku.edu.cn, {wye,zhangsk}@pku.edu.cn h  h
# Abstract
The emergence of in-context learning (ICL) enables large pre-trained language models (PLMs) to make predictions for unseen inputs without updating parameters. Despite its potential, ICL’s effectiveness heavily relies on the quality, quantity, and permutation of demonstrations, commonly leading to suboptimal and unstable performance. In this paper, we tackle this challenge for the first time from the perspective of demonstration augmentation. Specifically, we start with enriching representations of demonstrations by leveraging their deep feature distribution. We then theoretically reveal that when the number of augmented copies approaches infinity, the augmentation is approximately equal to a novel logit calibration mechanism integrated with specific statistical properties. This insight results in a simple yet highly efficient method that significantly improves the average and worst-case accuracy across diverse PLMs and tasks. Moreover, our method effectively reduces performance variance among varying demonstrations, permutations, and templates, and displays the capability to address imbalanced class distributions. 3 3 ( , ) y x
arXiv:2407.00100v1
# 1 Introduction
Large pre-trained language models (PLMs) have showcased exceptional abilities in in-context learning (ICL) (Brown et al., 2020; Wang et al., 2023; Rubin et al., 2022), which assists the model in discerning the underlying patterns within demonstrations and make more accurate predictions (Chan et al., 2022; Wu et al., 2023). As a new paradigm, ICL offers compelling advantages, allowing for natural language interaction with PLMs (Wei et al., 2022; Yang et al., 2023), as well as reduced computational costs (Li et al., 2023a; Rubin et al., 2022). While promising, ICL’s performance is highly dependent on provided demonstrations and templates (Liu et al., 2022; Zhang et al., 2022b;
*Corresponding authors.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f9d5/f9d536f6-96b1-4090-a496-cd2e4e743418.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Illustration for demonstration augmentation using semantic directions (vectors) sampled from the deep feature distribution of demonstration examples.</div>
Sorensen et al., 2022), resulting in subpar and unstable performance. This promotes research aimed at improving the quality (Rubin et al., 2022; Li et al., 2023b), quantity (Li et al., 2023a; Choi et al., 2022), and permutations (Lu et al., 2022; Tang et al., 2023) of demonstrations. Other research avenues include prediction adjustment (Zhao et al., 2021; Han et al., 2023; Fei et al., 2023) and learning process design (e.g., channel models (Min et al., 2022a) and meta-training frameworks (Min et al., 2022b)). Despite ongoing efforts, ICL still struggles with efficiently and reliably capturing sufficient knowledge from context, leaving performance stability as a persistent bottleneck. In this study, we propose enriching contextual knowledge for PLMs by augmenting demonstrations. We first attempt to enhance the representation of demonstrations by transforming them along semantic directions sampled from the deep feature space of demonstration examples, as depicted in Figure 1. This operation stems from the observation that the deep features in a network are usually linearized (Bengio et al., 2013; Cheung and Yeung, 2021; Cho, 2016), implying the existence of numerous semantic directions within the deep feature space, hence potentially enabling us to incorporate richer contextual knowledge without extending input length. From this novel perspective, we theoretically prove that when the number of augmented
pieces approaches infinity, its effect approximately equals a logit adjustment operation. Specifically, we derive a refined Softmax function that integrates he statistical properties of demonstrations. Consequently, rather than explicitly executing the augmentation procedure, we can efficiently conduct mplicit demonstration augmentation using the derived prediction function, obtaining an improved ICL method with theoretical guidance. We conduct extensive experiments across seven PLMs and various classification tasks. The empirical results demonstrate that our approach remarkably enhances prediction accuracy and reduces performance variability across different demonstraions, permutations, and templates. Notably, our method is straightforward, effective, and generalizable, enabling seamless integration with other ICL methods to enhance their performance. Our contributions can be summarized as follows: • We introduce Implicit Demonstration Augmentation-based ICL (IDAICL), a pioneering work that incorporates demonstration augmentation into ICL. Instead of solely enhancing demonstration quality, quantity, or order, our method explores context augmentation within the deep feature space, offering a new perspective to enrich demonstrations bypassing input length limitations. • We theoretically establish that as the number of augmented pieces approaches infinity, our augmentation strategy approximates a logitadjusted prediction function that integrates statistical properties derived from the input data distribution. Equipped with this function, IDAICL provides a straightforward yet theoryguided solution to enhance ICL. • Extensive experiments conducted across diverse tasks and PLMs conclusively illustrate that IDAICL considerably improves average and worst-case accuracy compared to existing ICL methods. Moreover, it effectively enhances performance stability.
# 2 Background and Related Work 2.1 In-Context Learning
# 2 Background and Related Work
# 2.1 In-Context Learning
Brown et al. (2020) showcased the ICL capability of PLMs, wherein PLMs generate predictions solely based on a concatenation of training examples for few-shot learning without updating parameters. Subsequent studies (Holtzman et al., 2021;
Min et al., 2022a,b) have developed this approach, yielding promising outcomes across various tasks. Nevertheless, recent research has uncovered certain limitations. To begin with, the volume of input knowledge for each query is constrained by the maximum input length of PLMs (Hao et al., 2022), and the computational cost increases as the number of demonstrations grows (Li et al., 2023a), making it challenging to integrate significant knowledge from demonstrations to PLMs. Additionally, ICL’s performance is sensitive to the input of PLMs (Davison et al., 2019; Jiang et al., 2020), thus exhibiting high variance and poor worst-case accuracy (Perez et al., 2021; Lu et al., 2022). Researchers have explored various techniques to address the biases and instability of ICL. These techniques encompass learning process design (Min et al., 2022a,b), demonstration retrieval (Rubin et al., 2022; Zhang et al., 2022b), prompt engineering (Sorensen et al., 2022; Lu et al., 2022), and prediction calibration (Zhao et al., 2021; Fei et al., 2023). However, these methods have yet to fully address the issue of severely limited knowledge transfer from demonstrations to large PLMs.
# 2.2 Data Augmentation
Data augmentation (Chen et al., 2023), which involves artificially creating training data through transformations, is a well-established research area in machine learning. Although data augmentation techniques have undergone extensive exploration in diverse machine learning domains (Maharana et al., 2022; Shorten and Khoshgoftaar, 2019), applying them to text data poses challenges due to the complexity of preserving labels during textual transformations (Kobayashi, 2018). Nonetheless, data augmentations in the latent space, such as adversarial training (Zhang et al., 2022a; Zhu et al., 2020; Cheng et al., 2020), interpolation (Chen et al., 2022b; Wu et al., 2022), and generative techniques (Li et al., 2022; Malandrakis et al., 2019), have demonstrated notable enhancements when applied alongside large PLMs. Recently, Wang et al. (2019) introduced the concept of implicit data augmentation in the context of image classification. This approach involves transforming training data within the deep feature space and boils down to the optimization of a novel robust loss function. Subsequent studies (Chen et al., 2022c; Li et al., 2021; Zhou and Wu, 2023a) for image classification tasks have further improved
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2899/28991dcb-a326-4665-9fc1-6b9310837c05.png" style="width: 50%;"></div>
<div style="text-align: center;">Deep feature distribution</div>
<div style="text-align: center;">Figure 2: An overview of IDAICL: For each contextual input, our goal is to augment the deep feature of demonstrations for M pieces, using semantic vectors δ drawn from the deep feature distribution N(µ, Σ) of demonstration examples linked to all queries. When M approaches infinity, we derive a novel prediction function, which incorporates two modulating factors: M(µ) and N(Σ), to calibrate the original predictions.</div>
Figure 2: An overview of IDAICL: For each contextual input, our goal is to augment the deep feature of demonstrations for M pieces, using semantic vectors δ drawn from the deep feature distribution N(µ, Σ) of demonstration examples linked to all queries. When M approaches infinity, we derive a novel prediction function, which incorporates two modulating factors: M(µ) and N(Σ), to calibrate the original predictions.
3h upon this approach. This study introduces an algorithm for implicitly augmenting demonstrations within the realm of ICL.
# 3 Methodology
# 3.1 In-Context Learning with PLMs
Considering a PLM G, this study focuses on the following task: given a query input text x and a candidate answer set Y = {y1, y2, · · · , y|Y|}, we aim to predict the answer ˆy based on m demonstration examples C ={c1, c2, · · · , cm}, where each ci represents a training example (xi, yi) after template formulation and m denotes the quantity of demonstration examples for each test sample. Formally, give a model G, we first compute the probability of each answer yj:
(1)
Subsequently, the ultimate prediction ˆy, characterized by the highest probability is chosen from the candidate answer set Y:
(2)
To simplify, the contextual input is denoted as ˜x = [C, x] in the subsequent text. Then, the probability of answer yj, represented as PG(yj|˜x), is computed using the Softmax function1:
(3)
� where h˜x = G(˜x) signifies the hidden state of the last block at the final position for ˜x. wk and bk are the weight vector and bias corresponding to the final fully connected layer for the k-th token.
1We begin by examining situations in which the answer comprises a single token, and our subsequent analysis is equally applicable to scenarios involving multiple tokens.
# 3.2 Demonstration Augmentation
Augmentation Augmented  Demonstration Recognizing the established efficacy of data augmentation in machine learning (Feng et al., 2021), this study investigates demonstration augmentation and suggests enhancing the deep features of demonstrations by transforming them along semantic directions sampled from the deep feature space of demonstration examples. This strategy is motivated by the intriguing observation that the deep features in networks are often linearized (Bengio et al., 2013; Chen et al., 2022a). Building on this observation, we hypothesize that h˜x lies within the subspace spanned by hC and hx: h˜x = αhC + βhx, where hC and hx represent the components of h˜x linked respectively to the demonstrations and the query. The necessity of this assumption stems from intricate relationships among token representations and the exclusive augmentation of the component related to demonstrations. Notably, this decomposition is not necessary in practical applications. In the subsequent text, we directly refer to αhC and βhx as hC and hx. To augment hC, we randomly sample vectors from the deep feature space of demonstrations. In particular, vectors are drawn from a multivariate normal distribution N(µ, Σ), where µ and Σ denote the feature mean and covariance matrix. These statistical properties are estimated from the deep features of the demonstration set D, which includes demonstration examples linked to all queries. The feature mean µ is computed as
(4)
where hi = G(ci) represents the hidden state of the last block at the final position for the i-th demonstration example ci in D, and |D| denotes the size
of D. The covariance matrix Σ is computed as
(5)
� Subsequently, hC is shifted in the extracted semantic vectors, resulting in augmented features, ˜hC, which follows
(6)
where λ refers to a positive coefficient controlling the strength of semantic augmentation. In realworld applications, it can be directly assigned a value of 0.5. Sensitivity tests for λ are discussed in Section 5.4.
# 3.3 Novel Prediction Function
Selecting the answer with the highest probability is equivalent to favoring the answer with the lowest inverse probability. Therefore, the prediction can be determined by
(7)
Assume that each hC is augmented for M times, resulting in an augmented demonstration feature set {˜h 1 C, · · · , ˜h M C } with size M. Here, ˜h i C represents the i-th augmented feature for hC. Then, the final prediction for the query x depends on all augmented features of hC and can be expressed as
(8)
(9)
Given that the performance of ICL benefits from an increased number of demonstration instances (Liu et al., 2022; Wu et al., 2023), we explore the scenario of augmenting an infinite number of times for the deep representation of demonstrations. Subsequently, an easily computable surrogate for the expected prediction can be derived, resulting in a highly efficient implementation. The whole pipeline of IDAICL is depicted in Figure 2. As M →∞, on the basis of the aforementioned decomposition of h˜x, the expected prediction for answer yj (denoted as P ∞ yj ) within the augmented feature set can be expressed as follows:
where ∆wk,yj = wk −wyj and ∆bk,yj = bk −byj.
However, accurately calculating P ∞ yj is challenging. Alternatively, we proceed to derive a surrogate calculation for it. Applying the linearity of expectation, Eq. (10) can be expressed as:
 (11)
Given that ˜hC is a Gaussian random variable conforming to N (hC + λµ, λΣ), we know that ∆wT k,yj ˜hC follows the multivariate normal distribution: N(∆wT k,yj (hC + λµ) , λ∆wT k,yjΣ∆wk,yj). Then, utilizing the moment-generating function
(12)
Eq. (11) can be derived as
(13)
exp( λ 2∆wT k,yjΣ∆wk,yj). Subsequently, our newly proposed prediction function, referred to as IDA-Softmax, is defined as
2 k,yjj Subsequently, our newly proposed prediction function, referred to as IDA-Softmax, is defined as
(14)
Consequently, instead of conducting the augmentation process explicitly, we can directly employ IDA-Softmax, P IDA yj , for prediction. IDA-Softmax essentially utilizes two modulating factors associated with statistical properties derived from D to calibrate the sample logits. Previous studies (Min et al., 2022c; Chan et al., 2022) have underscored the pivotal role of knowledge about the input data distribution in predictions made by PLMs. Intuitively, PLMs can better capture the patterns and underlying structures within data, such as the spatial relationships between demonstrations and queries, ultimately enhancing their prediction performance. Furthermore, to mitigate the imbalance among different answer types in demonstrations (Holtzman et al., 2021; Zhao et al., 2021), we adopt a post-hoc adjustment approach inspired by Menon et al. (2021), which adjusts predictions by considering the class proportions within D. Thus, the prediction for answer yj is computed as
(15)
where τ is a positive hyperparameter, and πyj demotes the proportion of answer yj in D. In practical applications, the value of τ can be fixed at 1.
PLM
Method
m
SST-2
SST-5
MR
CR
Amazon
Subj
TREC
DBPedia
AGNews
CB
GPT-2 0.8B
Vanilla ICL
4
57.67.1
30.46.3
59.36.5
56.88.4
32.78.5
57.65.4
34.910.3
40.57.2
44.57.9
35.19.3
IDAICL
86.41.4
38.32.9
82.22.3
78.40.7
46.73.5
77.02.3
47.52.0
81.31.8
73.92.4
41.52.0
Vanilla ICL
8
69.79.0
32.48.6
63.97.7
60.88.1
34.16.2
59.78.7
40.46.3
62.613.6
49.28.4
38.87.6
IDAICL
88.02.3
39.61.9
84.92.4
85.62.5
47.92.6
79.90.8
50.33.3
86.52.9
76.81.7
43.33.4
Vanilla ICL
12
74.78.3
33.77.6
64.49.4
68.79.7
36.06.6
60.77.7
40.57.8
64.55.4
51.18.0
40.48.5
IDAICL
88.52.1
40.12.7
85.23.1
86.81.4
49.62.2
80.42.1
51.41.6
87.32.7
77.92.0
44.62.2
MetaICL
12
80.86.2
35.84.7
75.35.6
77.68.1
48.96.7
73.58.8
48.66.1
80.47.8
66.80.7
43.14.1
+IDAICL
89.31.7
42.62.4
85.81.7
87.91.5
51.70.7
82.62.4
53.72.5
89.44.1
78.31.1
47.92.8
Channel ICL
12
85.23.6
38.44.3
80.84.7
82.04.6
43.65.1
69.89.8
44.18.7
77.612.9
69.56.7
42.45.2
+IDAICL
90.52.3
41.82.7
87.71.6
89.51.2
50.82.4
80.50.9
52.91.6
87.82.4
81.02.5
46.33.3
EPR
12
81.92.1
39.91.8
78.12.4
80.60.6
49.12.4
80.12.2
76.21.1
87.11.0
80.90.8
44.82.3
+IDAICL
90.11.1
43.91.2
86.42.0
88.60.6
52.51.7
83.61.0
79.10.9
90.80.7
83.70.5
46.72.1
GPT-2 1.5B
Vanilla ICL
4
66.38.6
30.38.9
56.56.6
53.48.1
34.77.5
54.25.5
30.88.1
61.98.7
54.69.9
40.87.8
IDAICL
87.41.5
38.81.7
80.91.2
82.12.1
48.10.6
77.83.0
49.51.9
87.42.6
79.21.8
54.12.7
Vanilla ICL
8
57.27.0
30.86.1
64.98.3
57.66.4
38.66.4
57.310.3
39.55.3
67.48.1
56.35.4
47.45.1
IDAICL
89.51.8
40.81.9
82.11.2
84.32.1
50.23.4
80.12.9
51.52.5
89.81.7
80.30.9
55.50.6
Vanilla ICL
12
70.99.6
34.76.7
65.25.6
59.96.7
38.310.2
59.68.1
40.77.5
72.511.6
57.69.5
48.55.7
IDAICL
90.02.8
41.11.3
83.42.3
85.62.4
51.62.9
80.52.5
51.83.6
90.52.7
81.13.0
55.72.1
MetaICL
12
79.17.0
38.63.7
76.46.3
75.34.5
50.57.1
73.97.6
46.76.3
86.87.8
76.45.4
53.11.6
+IDAICL
89.62.2
42.92.3
84.23.4
87.91.1
53.81.2
83.43.2
53.61.3
91.90.9
84.31.4
57.31.5
Channel ICL
12
83.35.9
37.54.6
80.64.1
77.15.5
48.96.7
68.28.3
43.37.2
70.49.3
67.95.5
53.68.9
+IDAICL
91.22.1
40.81.5
86.52.6
88.21.8
52.42.9
82.32.4
50.51.8
88.71.2
82.60.9
56.52.1
EPR
12
82.82.6
40.62.1
79.51.4
74.72.7
50.72.3
83.30.7
82.22.4
91.50.8
83.21.6
54.81.9
+IDAICL
90.51.5
43.81.0
87.40.9
86.51.5
52.91.8
85.80.5
84.71.1
93.52.5
86.42.2
57.51.5
GPT-Neo
MetaICL
12
87.86.7
42.56.1
82.25.9
80.74.8
51.55.3
72.28.2
54.16.8
84.45.5
74.38.2
50.36.4
+IDAICL
92.11.1
44.32.3
88.82.1
88.11.8
53.21.7
84.32.1
64.31.9
94.31.2
86.50.9
53.42.1
Channel ICL
12
83.45.4
39.86.4
79.55.7
79.45.9
50.13.8
70.68.2
50.85.1
78.37.1
72.56.9
48.74.5
+IDAICL
91.52.2
41.61.8
85.41.9
87.22.5
52.72.2
83.71.4
62.80.7
93.53.3
84.63.1
52.01.8
EPR
12
88.21.6
45.72.2
81.81.9
71.82.9
49.91.1
89.42.4
92.32.2
96.11.2
88.81.1
49.40.7
+IDAICL
93.20.8
47.21.3
88.51.2
86.62.0
52.10.4
93.11.2
94.42.4
97.81.5
91.20.7
52.10.5
Table 1: Comparison results of three PLMs. Two numbers indicate the mean accuracy (%) and standard deviation over different seeds. The best and second-best results per PLM per dataset are highlighted in bold and underlined, respectively. "+IDAICL" means that the current approach is used in conjunction with IDAICL. The results for different numbers of demonstration examples (i.e., m values) using the GPT-Neo model are illustrated in Figure 3.
This approach compensates for predictions of minor classes. When different answers are uniformly distributed, τ log πyj exerts an equal influence on all answer types. Consequently, the final prediction is given by
(16)
# 4 Experimental Setup
# 4 Experimental Setup 4.1 Models and Datasets
# 4.1 Models and Datasets
We evaluated the performance of IDAICL across seven large PLMs, including GPT-2 (Radford et al., 2019) (with 0.1B, 0.3B, 0.8B, and 1.5B parameters), GPT-Neo (Black et al., 2021) (with 2.7B parameters), and LLaMA (Touvron et al., 2023) (with 13B and 33B parameters). Following previous research (Min et al., 2022a; Han et al., 2023; Lu et al., 2022), our evaluation encompasses ten text classification datasets. Among these, SST-2 (Socher et al., 2013), SST-5 (Socher et al., 2013), MR (Pang
and Lee, 2005), CR (Hu and Liu, 2004), and Amazon (McAuley and Leskovec, 2013) are five sentiment classification tasks. Subj (Pang and Lee, 2004), TREC (Voorhees and Tice, 2000), DBPedia (Lehmann et al., 2015), and AGNews (Zhang et al., 2015) cater to subjectivity, question, ontology, and news classification tasks, respectively. Additionally, CB (De Marneffe et al., 2019) is utilized for natural language inference. Among these datasets, SST-5, Amazon, TREC, and CB are characterized by imbalanced training data. Details of all datasets are provided in Section A of the Appendix.
# 4.2 Compared Baselines
Besides Vanilla ICL, we compared and integrated IDAICL with three popular ICL algorithms, focusing on learning process design and demonstration retrieval. These include MetaICL (Min et al., 2022b), Channel ICL (Min et al., 2022a), and Efficient Prompt Retrieval (EPR) (Rubin et al., 2022). Moreover, we compared IDAICL with other ad-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0491/049199b8-442e-4a81-8de4-368cffa72b2c.png" style="width: 50%;"></div>
PLM
Method
SST-2
SST-5
MR
CR
Subj
TREC
DBPedia
AGNews
CB
Avg.
LLaMA 13B
Vanilla ICL
95.67.1
29.56.2
90.05.8
91.47.4
72.96.9
62.89.1
80.97.6
80.25.9
51.58.2
72.8
ConCa
96.75.4
40.36.2
91.77.3
90.84.2
79.69.1
68.25.6
94.34.1
85.27.5
46.65.0
77.0
PROCA
95.43.8
43.45.7
90.39.6
92.13.1
84.82.5
69.92.1
92.54.9
81.63.6
51.44.2
77.9
D-ConCa
96.33.8
42.54.5
92.04.1
90.52.9
82.94.5
73.73.9
87.47.2
82.53.3
52.24.1
77.8
IDAICL
96.72.5
47.11.1
93.01.9
93.30.8
87.82.3
76.02.6
94.91.0
87.72.4
59.41.9
81.8
LLaMA 33B
Vanilla ICL
95.57.2
29.45.6
91.75.4
91.58.1
85.16.0
70.94.4
86.64.5
76.26.1
59.25.3
76.2
ConCa
95.96.5
39.14.4
90.37.2
91.23.6
74.65.7
76.76.2
92.43.9
87.35.7
57.96.0
78.4
PROCA
95.54.2
39.26.3
92.44.1
91.33.5
88.32.2
64.73.8
86.95.1
85.87.1
59.93.8
78.2
D-ConCa
95.43.8
40.74.5
92.14.2
91.02.9
76.43.6
80.22.1
87.64.2
87.74.3
56.53.4
78.6
IDAICL
96.51.1
46.82.4
93.61.3
92.33.3
89.32.4
79.11.5
95.62.3
88.41.9
64.62.8
82.9
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/226d/226d7029-18dc-4ccd-9d32-958ef9640deb.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Comparison results between Vanilla ICL and IDAICL across different values of m on the GPT-Neo model IDAICL significantly outperforms Vanilla ICL, particularly when the number of demonstration examples is small.</div>
vanced prediction calibration methods: Contextual Calibration (ConCa) (Zhao et al., 2021), Prototypical Calibration (PROCA) (Han et al., 2023), and Domain-Context Calibration (D-ConCa) (Fei et al., 2023). Introductions to all compared methods and comprehensive experimental settings are presented in Sections B and C of the Appendix.
# 5 Experimental Results
# 5.1 Main Results
Table 1 displays the comparison results between IDAICL and four ICL baselines (Vanilla ICL, MetaICL, Channel ICL, and EPR) across GPT-2 models (with 0.8B and 1.5B parameters) and the GPT-Neo model. These results lead to three main findings. Firstly, IDAICL consistently exhibits high effectiveness across various model sizes and datasets, highlighting its strong generalization capacity, even under scenarios involving imbalanced training data. Compared to Vanilla ICL, IDAICL outperforms by an average of 17.7% and 18.4% across diverse datasets and m values for GPT-2 with 0.8B and 1.5B parameters, respectively. Secondly, in comparison to other ICL baselines like Channel ICL, MetaICL, and EPR, the integration of
IDAICL consistently delivers notable performance improvements, emphasizing the efficacy of enhancing demonstrations for refined predictions. The inclusion of IDAICL led to an average performance boost of 7.3% for MetaICL and 8.2% for Channel ICL. Lastly, IDAICL notably enhances worstcase accuracy and diminishes performance variance across different seeds, showcasing its ability to improve prediction stability. Additional results on LLaMA and smaller GPT-2 models are available in Tables 7 and 8 of the Appendix.
# 5.2 Comparison with Calibration Methods
We compared IDAICL with three advanced prediction calibration methods (ConCa, PROCA, and D-ConCa) across three PLMs: GPT-2, GPT-Neo, and LLaMA. Table 2 presents the comparison results for the LLaMA models, where IDAICL consistently achieves state-of-the-art performance, except for TREC using the LLaMA model with 33B parameters. These findings suggest that IDAICL which leverages statistical information derived from the input data distribution for prediction calibration, generally outperforms methods relying on estimated biases for correction. Further comparison results can be found in Table 9 of the Appendix.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f619/f61967dd-2bd5-4793-a031-63da17a16696.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: (a) and (b): Macro-F1 of SST-5 and AGNews datasets using the LLaMA model with 33B parameters under three demonstration selection settings, setting m to 4. (c) and (d): Accuracy of Vanilla ICL and IDAICL on the SST-2 dataset using the GPT-2 model with 1.5B parameters across six templates, setting m to 12. IDAICL demonstrates greater robustness across various demonstration examples and templates compared to Vanilla ICL.</div>
# 5.3 Stability Analysis
Previous studies (Zhao et al., 2021; Sorensen et al., 2022; Min et al., 2022a; Zhang et al., 2022b) have highlighted the considerable variability in ICL’s performance. In this section, we verified that IDAICL can effectively enhance performance stability across diverse scenarios.
Varying numbers of demonstrations We have presented the results across different numbers of demonstrations in Table 1. For a clearer depiction, the outcomes regarding GPT-Neo are illustrated in Figure 3. As the number of demonstration examples (represented by m) increases, both Vanilla ICL and IDAICL exhibit improved performance, emphasizing the importance of comprehensive statistical properties of the input data for IDAICL’s effectiveness. Notably, IDAICL significantly enhances performance stability across various numbers of demonstrations and consistently outperforms Vanilla ICL. The performance improvement is particularly pronounced when m takes on smaller values, indicating the efficacy of IDAICL in enriching the available knowledge for PLMs.
# Varying demonstrations T
# Varying demonstrations To confirm tha
Varying demonstrations To confirm that augmenting demonstrations can enhance the robustness of the ICL strategy across various demonstrations, we investigated three distinct demonstration selection settings. Setting I: Training samples most similar to the test sample are chosen. Setting II: Samples are randomly selected from the training data. Setting III: Training samples exhibiting the greatest dissimilarity from the test sample are selected. As shown in Figures 4(a) and (b), IDAICL significantly outperforms Vanilla ICL and demonstrates greater robustness across the three selection settings. Additionally, our discoveries suggest that selecting demonstrations that are more similar to the test samples leads to better performance than
exclusively selecting dissimilar ones, which aligns with the findings obtained by Wang et al. (2022).
Varying templates To assess the performance of IDAICL across various templates, we employed fifteen templates on the SST-2 dataset following those outlined by Zhao et al. (2021). The templates are elaborated in Table 10 of the Appendix. Figures 4(c) and (d) display the performance of Vanilla ICL and IDAICL across six templates. Some templates achieve higher average performance than others. Nevertheless, IDAICL consistently enhances both average and worst-case accuracy, simultaneously reducing performance variance across different templates. The complete results are available in Figure 7 of the Appendix.
Impact of imbalance in labels Figures 5(a) and (b) depict comparison results among Vanilla ICL, MetaICL, Channel ICL, and IDAICL across different degrees of imbalances. It is evident that the performance of Vanilla ICL is sensitive to class imbalance, while that of IDAICL and Channel ICL exhibit robustness to the imbalance. Moreover, notable performance improvements are observed with higher levels of imbalance. Additionally, Figures 5(c) and (d) illustrate the confusion matrices for CR and Subj datasets, with the proportion of one category (i.e., "Negative" and "Subjective") in demonstrations setting to 0.1 and 0.2. IDAICL significantly improves the accuracy of the underrepresented classes when compared to Vanilla ICL, thereby contributing to enhanced fairness among classes. In the subsequent section, we demonstrate that the strong performance of IDAICL in handling imbalanced label distributions stems from both the statistical properties and the class proportion term.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f00a/f00a3de7-9705-429e-a785-02b742a1fa3b.png" style="width: 50%;"></div>
Figure 5: (a) and (b): Accuracy comparison of the SST-2 and MR datasets, where the proportions of the negative class in demonstrations (denoted as p) are varied from 0.1 to 0.5. (c) and (d): Confusion matrices for the CR and Subj datasets, representing scenarios where the proportions of one category in demonstrations are set to 0.1 and 0.2. The analysis is conducted using the GPT-2 model with 1.5B parameters, with m setting to 12. IDAICL demonstrates greater robustness in handling imbalanced class distributions within demonstrations.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4223/4223bbb7-a439-4a20-a1b1-95ece065d82a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Accuracy across different λ and τ values, using GPT-2 with 0.8B parameters, setting m to 12. λ= 0 and τ =0 signify that the two modulating factors and the class proportion term are not utilized, respectively.</div>
# 5.4 Sensitivity and Ablation Studies
We conducted ablation studies on IDAICL to investigate the influence of the two modulating factors and the class proportion term. The parameters λ and τ govern the augmentation strength and the impact of the class proportion term, respectively. In Figure 6(a), a significant performance drop is observed when predictions are not calibrated using statistical properties derived from the demonstrations. Additionally, optimal performance is achieved when λ equals 0.5. Figure 6(b) showcases the accuracy of SST-2 and MR datasets with the negative class proportion in demonstrations setting to 0.1. Results indicate that solely leveraging statistical properties (i.e.,
Dataset
0-shot
1-shot
4-shot
IDAICL
SST-2
63.2
61.39.4
57.67.1
76.3
SST-5
25.0
27.37.9
30.46.3
33.5
MR
58.9
54.36.8
59.36.5
71.2
Subj
48.9
47.18.3
57.65.4
67.3
Table 3: Accuracy comparison between Vanilla ICL and IDAICL based solely on statistical properties, using the GPT-2 model with 0.8B parameters.
τ equals 0) enhances performance under imbalanced demonstrations, with further improvements observed upon the inclusion of the class proportion term. Additionally, optimal performance is attained when τ equals 1. Consequently, we recommend setting λ to 0.5 and τ to 1 for practical applications. More results are presented in Appendix F.
# 5.5 Further Discussion
To further investigate the effect of statistical properties within demonstrations on model performance, we exclusively employed queries along with statistical information for inference, excluding the inclusion of demonstrations for each test sample. These statistics were estimated using deep features of all training samples. As shown in Table 3, IDAICL relying solely on statistical properties distinctly outperforms Vanilla ICL across scenarios with zero, one, and even four demonstrations. This emphasizes the crucial role of prior statistics obtained from training data in PLMs’ predictions. This phenomenon is understandable as statistical properties inherently encompass richer global information compared to individual demonstrations.
# 6 Conclusion
This study introduces IDAICL, a novel ICL approach designed to enhance demonstrations by utilizing semantic directions sampled from the deep feature distribution of demonstration examples. Our augmentation strategy enriches the knowledge available to PLMs without extending the context length. A new prediction function is then theoretically established considering the number of augmented pieces approaching infinity. This eliminates the need for explicit augmentation and allows for direct utilization of this derived function for
predictions. Our extensive experiments, spanning various tasks and PLMs, demonstrate that IDAICL significantly enhances both prediction accuracy and stability when compared to other ICL baselines.
# Limitations
While IDAICL proves to be competitive in few-shot learning, there are limitations that open up avenues for future research. First, due to the necessity of accessing the parameters of the final fully connected layer in PLMs, IDAICL is exclusively suitable for open-source models. Future research is expected to develop alternative augmentation strategies tailored for black-box PLMs. Second, our evaluation of IDAICL focused on seven PLMs and ten text classification tasks. We defer further explorations involving other PLMs and non-classification tasks for future work. Additionally, IDAICL relies on a small set of demonstrations to estimate the feature mean and covariance matrix. If such a collection is unavailable or extremely scarce, IDAICL may need to be used in conjunction with demonstration generation methods. Other avenues for future work involve exploring more effective augmentation distributions. This entails exploring finer-grained distributions, such as category-level or sample-level distributions, to emphasize the unique characteristics of individual categories or samples, and extending these distributions beyond the constraints of training data. Furthermore, given the effectiveness of data augmentation in model training, future research could explore the utilization of our derived prediction function in both the training and fine-tuning phases of large PLMs.
# References
Yoshua Bengio, Grégoire Mesnil, Yann Dauphin, and Salah Rifai. 2013. Better mixing via deep representations. In Proceedings of 30th International Conference on Machine Learning, pages 552–560, Atlanta, Georgia, USA. ACM. Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Rose Biderman. 2021. Gpt-neo: Large scale autoregressive language modeling with meshtensorflow.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Proceedings of the 34th International Conference on Neural Information Processing Systems, page 1877–1901, Online and Vancouver, Canada. Stephanie Chan, Adam Santoro, Andrew Lampinen, Jane Wang, Aaditya Singh, Pierre Richemond, James McClelland, and Felix Hill. 2022. Data distributional properties drive emergent in-context learning in transformers. In Proceedings of the 36th Advances in Neural Information Processing Systems, pages 18878–18891, New Orleans, USA. Dong Chen, Yueting Zhuang, Zijin Shen, Carl Yang, Guoming Wang, Siliang Tang, and Yi Yang. 2022a. Cross-modal data augmentation for tasks of different modalities. IEEE Transactions on Multimedia, 99:1– 11. Hui Chen, Wei Han, Diyi Yang, and Soujanya Poria. 2022b. DoubleMix: Simple interpolation-based data augmentation for text classification. In Proceedings of the 29th International Conference on Computational Linguistics, pages 4622–4632, Gyeongju, Republic of Korea. International Committee on Computational Linguistics. Jiaao Chen, Derek Tam, Colin Raffel, Mohit Bansal, and Diyi Yang. 2023. An empirical survey of data augmentation for limited data learning in NLP. Transactions of the Association for Computational Linguistics, 11:191–211. Xiaohua Chen, Yucan Zhou, Dayan Wu, Wanqian Zhang, Yu Zhou, Bo Li, and Weiping Wang. 2022c. Imagine by reasoning: A reasoning-based implicit semantic data augmentation for long-tailed classification. In Proceedings of the 36th AAAI Conference on Artificial Intelligence, pages 356–364, Online. AAAI Press. Yong Cheng, Lu Jiang, Wolfgang Macherey, and Jacob Eisenstein. 2020. AdvAug: Robust adversarial augmentation for neural machine translation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5961–5970, Online. Association for Computational Linguistics. Tsz-Him Cheung and Dit-Yan Yeung. 2021. Modals: Modality-agnostic automated data augmentation in the latent space. In Proceedings of 9th International Conference on Learning Representations, Online. Kyunghyun Cho. 2016. Noisy parallel approximate decoding for conditional recurrent language model. arXiv preprint arXiv:1605.03835. Eunbi Choi, Yongrae Jo, Joel Jang, and Minjoon Seo. 2022. Prompt injection: Parameterization of fixed inputs. arXiv preprint arXiv:2206.11349.
Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Proceedings of the 34th International Conference on Neural Information Processing Systems, page 1877–1901, Online and Vancouver, Canada.
Joe Davison, Joshua Feldman, and Alexander Rush. 2019. Commonsense knowledge mining from pretrained models. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1173–1178, Hong Kong, China. Association for Computational Linguistics.
Steven Y. Feng, Varun Gangal, Jason Wei, Sarath Chandar, Soroush Vosoughi, Teruko Mitamura, and Eduard Hovy. 2021. A survey of data augmentation approaches for NLP. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 968–988, Online. Association for Computational Linguistics.
Tianyu Gao, Adam Fisch, and Danqi Chen. 2021. Making pre-trained language models better few-shot learners. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3816–3830, Online. Association for Computational Linguistics.
Zhixiong Han, Yaru Hao, Li Dong, Yutao Sun, and Furu Wei. 2023. Prototypical calibration for fewshot learning of language models. In Proceedings of 11st International Conference on Learning Representations, Kigali, Rwanda.
Yaru Hao, Yutao Sun, Li Dong, Zhixiong Han, Yuxian Gu, and Furu Wei. 2022. Structured prompting: Scaling in-context learning to 1,000 examples. arXiv preprint arXiv:2212.06713.
Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, and Luke Zettlemoyer. 2021. Surface form competition: Why the highest probability answer isn’t always right. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7038–7051, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
Minqing Hu and Bing Liu. 2004. Mining and summarizing customer reviews. In Proceedings of the 10th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 168–177, Seattle, Washington, USA. ACM.
Minqing Hu and Bing Liu. 2004. Mining and summarizing customer reviews. In Proceedings of the 10th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 168–177, Seattle, Washington, USA. ACM.
Zhengbao Jiang, Frank F. Xu, Jun Araki, and Graham Neubig. 2020. How can we know what language models know? Transactions of the Association for Computational Linguistics, 8:423–438.
osuke Kobayashi. 2018. Contextual augmentation: Data augmentation by words with paradigmatic relations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 452–457, New Orleans, Louisiana. Association for Computational Linguistics.
Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N. Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick Van Kleef, Sören Auer, and Christian Bizer. 2015. Dbpedia a large-scale, multilingual knowledge base extracted from wikipedia. Semantic Web, 6(2):167–195.
Robert Logan IV, Ivana Balazevic, Eric Wallace, Fabio Petroni, Sameer Singh, and Sebastian Riedel. 2022. Cutting down on prompts and parameters: Simple
Robert Logan IV, Ivana Balazevic, Eric Wallace, Fabio Petroni, Sameer Singh, and Sebastian Riedel. 2022. Cutting down on prompts and parameters: Simple
few-shot learning with language models. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2824–2835, Dublin, Ireland. Association for Computational Linguistics.
Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland. Association for Computational Linguistics. Kiran Maharana, Surajit Mondal, and Bhushankumar Nemade. 2022. A review: Data pre-processing and data augmentation techniques. Global Transitions Proceedings, 3(1):91–99. Nikolaos Malandrakis, Minmin Shen, Anuj Goyal, Shuyang Gao, Abhishek Sethi, and Angeliki Metallinou. 2019. Controlled text generation for data augmentation in intelligent artificial agents. In Proceedings of the 3rd Workshop on Neural Generation and Translation, pages 90–98, Hong Kong. Association for Computational Linguistics. Julian McAuley and Jure Leskovec. 2013. Hidden factors and hidden topics: Understanding rating dimensions with review text. In Proceedings of the 7th ACM conference on Recommender systems, pages 165–172, Hong Kong, China. ACM. Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, and Sanjiv Kumar. 2021. Long-tail learning via logit adjustment. In Proceedings of the 9th International Conference on Learning Representations, Vienna, Austria. Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022a. Noisy channel language model prompting for few-shot text classification. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5316–5330, Dublin, Ireland. Association for Computational Linguistics. Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2022b. MetaICL: Learning to learn in context. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2791–2809, Seattle, United States. Association for Computational Linguistics. Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022c. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, Abu Dhabi, United Arab Emirates. Association for
Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022a. Noisy channel language model prompting for few-shot text classification. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5316–5330, Dublin, Ireland. Association for Computational Linguistics.
Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2022b. MetaICL: Learning to learn in context. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2791–2809, Seattle, United States. Association for Computational Linguistics.
Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022c. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
Bo Pang and Lillian Lee. 2004. A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics (ACL-04), pages 271–278, Barcelona, Spain.
Bo Pang and Lillian Lee. 2004. A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics (ACL-04), pages 271–278, Barcelona, Spain. Bo Pang and Lillian Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL’05), pages 115–124, Ann Arbor, Michigan. Association for Computational Linguistics. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. Pytorch: An imperative style, high-performance deep learning library. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pages 8026–8037, Vancouver, Canada. Ethan Perez, Douwe Kiela, and Kyunghyun Cho. 2021. True few-shot learning with language models. In Proceedings of the 35th Conference on Neural Information Processing Systems, pages 11054–11070, Online. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. OpenAI blog. Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2022. Learning to retrieve prompts for in-context learning. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2655–2671, Seattle, United States. Association for Computational Linguistics. Connor Shorten and Taghi M. Khoshgoftaar. 2019. A survey on image data augmentation for deep learning. Journal of Big Data, 6:1–48. Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA. Association for Computational Linguistics. Taylor Sorensen, Joshua Robinson, Christopher Rytting, Alexander Shaw, Kyle Rogers, Alexia Delorey, Mahmoud Khalil, Nancy Fulda, and David Wingate. 2022. An information-theoretic approach to prompt engineering without ground truth labels. In Proceedings of the 60th Annual Meeting of the Association
Bo Pang and Lillian Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL’05), pages 115–124, Ann Arbor, Michigan. Association for Computational Linguistics.
# Connor Shorten and Taghi M. Khoshgoftaar. 2019. A survey on image data augmentation for deep learning. Journal of Big Data, 6:1–48.
Taylor Sorensen, Joshua Robinson, Christopher Rytting, Alexander Shaw, Kyle Rogers, Alexia Delorey, Mahmoud Khalil, Nancy Fulda, and David Wingate. 2022. An information-theoretic approach to prompt engineering without ground truth labels. In Proceedings of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 819–862, Dublin, Ireland. Association for Computational Linguistics.
<div style="text-align: center;">Demonstrations, pages 38–45, Online. Association for Computational Linguistics.</div>
Xing Wu, Chaochen Gao, Meng Lin, Liangjun Zang, and Songlin Hu. 2022. Text smoothing: Enhance various data augmentation methods on text classification tasks. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 871–875, Dublin, Ireland. Association for Computational Linguistics. Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, and Lingpeng Kong. 2023. Self-adaptive in-context learning: An information compression perspective for incontext example selection and ordering. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1423–1436, Toronto, Canada. Association for Computational Linguistics. Linyi Yang, Shuibai Zhang, Zhuohao Yu, Guangsheng Bao, Yidong Wang, Jindong Wang, Ruochen Xu, Wei Ye, Xing Xie, Weizhu Chen, et al. 2023. Supervised knowledge makes large language models better in-context learners. In The Twelfth International Conference on Learning Representations. Minjia Zhang, Niranjan Uma Naresh, and Yuxiong He. 2022a. Adversarial data augmentation for taskspecific knowledge distillation of pre-trained transformers. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 11685–11693, online. AAAI Press. Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Proceedings of the 28th International Conference on Neural Information Processing Systems, page 649–657, Montreal, Canada. Yiming Zhang, Shi Feng, and Chenhao Tan. 2022b. Active example selection for in-context learning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9134– 9148, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, pages 12697–12706, Online. ACM. Xiaoling Zhou and Ou Wu. 2023a. Implicit counterfactual data augmentation for deep neural networks. arXiv preprint arXiv:2304.13431. Xiaoling Zhou and Ou Wu. 2023b. Which samples should be learned first: Easy or hard? IEEE Transactions on Neural Networks and Learning Systems, pages 1–15. Xiaoling Zhou, Ou Wu, and Chao Jiang. 2022. Increasing naturalness of human–machine dialogue: The users’ choices inference of options in machine-raised questions. Knowledge-Based Systems, 243:108485.
Chen Zhu, Yu Cheng, Zhe Gan, Siqi Sun, Tom Goldstein, and Jingjing Liu. 2020. Freelb: Enhanced adversarial training for natural language understanding. In Proceedings of the 8th International Conference on Learning Representations, Online.
# A Details of Applied Datasets
Table 4 presents comprehensive statistics for all datasets utilized in this study. The information includes task descriptions, average sentence lengths, class counts, and details on class imbalance. Additionally, Table 5 provides sample instances and label names for each of the datasets.
# B Details of Compared Baselines
The compared methods are described as follows:
• Vanilla ICL: We use the PLMs as they are and implement ICL by conditioning it on a concatenation of m training examples, following the approach outlined by Brown et al. (2020). • MetaICL: The fundamental concept underlying MetaICL is to utilize a multi-task learning framework across a diverse range of metatraining tasks (Min et al., 2022b).
catenation of m training examples, following the approach outlined by Brown et al. (2020). • MetaICL: The fundamental concept underlying MetaICL is to utilize a multi-task learning framework across a diverse range of metatraining tasks (Min et al., 2022b).
• MetaICL: The fundamental concept underlying MetaICL is to utilize a multi-task learning framework across a diverse range of metatraining tasks (Min et al., 2022b).
• Channel ICL: It employs a noisy channel approach for language model prompting in few-shot text classification (Min et al., 2022a).
• EPR: It employs language models to autonomously label examples that are suitable as effective prompts and subsequently trains a prompt retriever based on this acquired signal (Rubin et al., 2022).
• ConCa: It assesses the model’s inclination towards specific answers by introducing a dummy test input that lacks content (Zhao et al., 2021).
• PROCA: The prediction of PROCA is calibrated based on the likelihood of prototypical clusters (Han et al., 2023).
# • PROCA: The prediction of PROCA is calibrated based on the likelihood of prototypical clusters (Han et al., 2023).
 D-ConCa: It initially assesses the impacts of various label biases by employing randomly sampled words from the task corpus. During inference, it utilizes the estimated label bias to calibrate the model’s output probabilities (Fei et al., 2023).
# C More Details of Experimental Settings
The entire implementation is conducted utilizing PyTorch (Paszke et al., 2019) and Transformers (Wolf et al., 2020). We follow the parameter configurations and details specified in previous research (Min et al., 2022a). The number of demonstrations is primarily set to m = 12, but we also explore m values of {1, 4, 8, 12, 16} in the ablations, with the specific settings detailed in the respective sections. Demonstration examples for each test sample are randomly selected from the training data, unless specific methods employ a specially designed selection method, such as EPR (Rubin et al., 2022). The values of the feature mean and covariance matrix are estimated from the demonstration set containing demonstration examples corresponding to all test samples. We depart from the assumption made in previous studies, which presupposes an equal distribution of training examples across all classes (Gao et al., 2021; Logan IV et al., 2022), in order to facilitate a more realistic and demanding evaluation. Each experiment is repeated under five different random seeds. The batch size is set to 32, and the sequence length is configured to 128 for datasets with shorter texts, including SST2 (Socher et al., 2013), SST-5 (Socher et al., 2013), MR (Pang and Lee, 2005), CR (Hu and Liu, 2004), and TREC (Voorhees and Tice, 2000). On the other hand, for datasets with longer input texts, including AGNews (Zhang et al., 2015), DBPedia (Lehmann et al., 2015), Subj (Pang and Lee, 2004), CB (De Marneffe et al., 2019), and Amazon (McAuley and Leskovec, 2013), a batch size of 16 and a sequence length of 256 are employed. Regarding the hyperparameters in IDAICL, the values of λ and τ are fixed at 0.5 and 1, respectively, except in sensitivity tests. The settings used for the compared methods adhere to those specified in the original papers (Min et al., 2022a,b; Rubin et al., 2022; Zhao et al., 2021; Han et al., 2023; Fei et al., 2023). Accuracy serves as the primary evaluation metric, alongside the provided values of Macro-F1 for the LLaMA model. For each task, a specific template is utilized for inference, as detailed in Table 6. Additionally, we also examine the impact of different templates on the performance of IDAICL following those outlined by Zhao et al. (2021), which include question-answer templates, conversation-style templates, prompts resembling web pages, and variations on label names,
Dataset
Task
Avg. length
Classes
Balanced
SST-2 (Socher et al., 2013)
Sentiment analysis
12.4
2
Yes
SST-5 (Socher et al., 2013)
Sentiment analysis
23.1
5
No
MR (Pang and Lee, 2005)
Sentiment analysis
25.7
2
Yes
CR (Hu and Liu, 2004)
Sentiment analysis
22.1
2
Yes
Amazon (McAuley and Leskovec, 2013)
Sentiment analysis
78.5
5
No
Subj (Pang and Lee, 2004)
Subjectivity classification
28.9
2
Yes
TREC (Voorhees and Tice, 2000)
Question classification
11.6
6
No
DBPedia (Lehmann et al., 2015)
Ontology classification
65.5
14
Yes
AGNews (Zhang et al., 2015)
News classification
53.8
4
Yes
CB (De Marneffe et al., 2019)
Natural language inference
69.7/8.4
3
No
<div style="text-align: center;">able 4: Statistical information of ten datasets. The average length is calculated based on the GPT-2 sentence-piece ength. For tasks involving sentence pairs, we provide the average length for each individual sentence.</div>
Table 4: Statistical information of ten datasets. The average length is calculated based on the GPT-2 sentence-pie length. For tasks involving sentence pairs, we provide the average length for each individual sentence.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/904c/904c3333-a0bb-4657-b33b-a569adabf6d5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Comparison results between Vanilla ICL and IDAICL across fifteen templates. The evaluation is conducted using the GPT-2 model with 1.5B parameters. The performance of IDAICL exceeds that of Vanilla ICL and demonstrates greater robustness across various templates.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0af8/0af8a760-3da5-4cd4-976e-2c2646e94f7e.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: Results of sensitivity tests for two hyperparameters within IDAICL, i.e., λ and τ, using the GPT-2 model with 0.8B parameters, with m setting to 12. Optimal performance is achieved when λ ≈0.5 and τ ≈1.</div>
as listed in Table 10.
# D More Comparison Results
The comparison results between Vanilla ICL and IDAICL on LLaMA models with 13B and 33B parameters across various datasets are presented in Table 7. Additionally, the corresponding results for GPT-2 models with 0.1B and 0.3B parameters are outlined in Table 8. It is evident that IDAICL
consistently outperforms Vanilla ICL across all datasets and different model sizes, highlighting the high generalization capability of IDAICL. Additionally, IDAICL showcases reduced performance variance and significantly enhances the worst-case performance. Based on the findings presented in Table 9, IDAICL generally outperforms other prediction calibration methods, demonstrating the significance of statistical properties derived from the input data distribution in the predictions of PLMs.
# E More Results for Varying Templates
The comparison results between Vanilla ICL and IDAICL under all fifteen prompt templates are presented in Figure 7, illustrating that IDAICL consistently enhances both average and worst-case accuracy across all templates. Furthermore, the performance variance of IDAICL among different templates is notably smaller when compared to Vanilla ICL, highlighting the robustness of IDAICL’s performance across diverse templates.
Dataset
Instances
Label names
SST-2
1. This movie is amazing! (Label = "Positive")
2. Horrific movie, don’t see it. (Label = "Negative")
Positive, Negative
SST-5
1. A pretensions – and disposable story — sink the movie. (Label =
"Great")
2. Apparently reassembled from the cutting-room floor of any given
daytime soap. (Label = "Terrible")
Terrible, Bad, Okay, Good, Great
MR
1. Lame sweet home leaves no southern stereotype unturned. (Label
= "Negative")
2. Not so much farcical as sour. (Label = "Negative")
Negative, Positive
CR
1. It takes excellent pics and is very easy to use, if you read the
manual. (Label = "Negative")
2. Bluetooth does not work on this phone. (Label = "Negative")
Negative, Positive
Amazon
1. Don’t waste your money if you already have 2003... There isn’t
one reason to get this update if you already have MS Money 2003
Deluxe and Business. (Label ="Terrible")
2. The game was in perfect condition! came before it said it should
have by 2 days!! I love the game and I suggest it to a lot of my
friends! (Label ="Great")
Terrible, Bad, Okay, Good, Great
Subj
1. This is a story about the warm relationship between a little girl
and her father despite the difficult conditions they have to live in.
(Label = "Objective")
2. Too slow, too boring, and occasionally annoying. (Label =
"Subjective")
Subjective, Objective
TREC
1. When did the neanderthal man live? (Label = "Number")
2. How do you get a broken cork out of a bottle? (Label = "Descrip-
tion")
Description, Entity, Expression,
Human, Location, Number
DBPedia
1. CMC Aviation is a charter airline based in Nairobi Kenya. (Label
= "Company")
2. Dialectica aemula is a moth of the Gracillariidae family. (Label =
"Animal")
Company, School, Artist, Athlete,
Politics, Transportation, Building,
Nature, Village, Animal, Plant,
Album, Film, Book
AGNews
1. Walk in park for Yankees Drained by a difficult week, the New
York Yankees needed an uplifting victory. (Label = "Sports")
2. NASA Mountain View claims world’s fastest computer. (Label =
"Technology")
World, Sports, Business,
Technology
CB
1. It was a complex language. Not written down but handed down.
One might say it was peeled down.
The language was peeled down.
(Label = "True")
2. “Do you mind if I use your phone?” Ronni could see that Guido’s
brain was whirring.
Guido’s brain was whirring.
(Label = "True")
True, False, Neither
<div style="text-align: center;">Table 5: Examples and label names from all datasets.</div>
# F More Sensitivity and Ablation Studies
We performed sensitivity tests on two hyperparameters within IDAICL: λ and τ. These values govern the strength of implicit augmentation and the influence of the class proportion term, respectively. As depicted in Figure 8, optimal performance is achieved when λ≈0.5 and τ ≈1 for both datasets. Furthermore, Figures 9(a) and (b) illustrate the average performance of ten datasets across different hyperparameter settings. Much like the earlier findings, the best average performance is identified at λ = 0.5 and τ = 1. Consequently, setting λ as 0.5 and τ as 1 is recommended for real applications. Furthermore, the performance remains sta-
ble within the ranges of λ ∈{0.25, 0.5, 0.75} and τ ∈{0.5, 1, 1.5}, indicating that adjustments can be made within these stable ranges.
# G More Results for Imbalanced Labels
The imbalanced label distribution in the training data has a significant impact on the classification performance of the model (Zhou and Wu, 2023b; Zhou et al., 2022). We depicted the confusion matrices for the SST-2 and MR datasets under two imbalance levels in Figures 9(c) and (d), in which the proportion of the negative class in demonstrations is set to 0.1 and 0.2. These results manifest that IDAICL significantly enhances the performance of the underrepresented classes in comparison to
Dataset
Template
Label mapping
SST-2
Review: {Sentence}
Sentiment: {Label}
Positive / Negative
SST-5
Review: {Sentence}
Sentiment: {Label}
terrible / bad / okay / good / great
MR
Review: {Sentence}
Sentiment: {Label}
Positive / Negative
CR
Review: {Sentence}
Sentiment: {Label}
Positive / Negative
Subj
Input: {Sentence}
Type: {Label}
objective / subjective
TREC
Question: {Sentence}
Type: {Label}
description / entity / expression / human / location / number
Amazon
Review: {Sentence}
Sentiment: {Label}
terrible / bad / okay / good / great
AGNews
Input: {Sentence}
Type: {Label}
world / sports / business / technology
DBPedia
Input: {Sentence}
Type: {Label}
company / school / artist / athlete / politics / transportation
building / nature / village / animal / plant / album / film / book
CB
Premise: {Sentence}
Hypothesis: {Sentence}
Prediction: {Label}
true / false / neither
<div style="text-align: center;">Table 6: Prompt templates and label mappings for each dataset.</div>
PLM
Method
m
SST-2
SST-5
MR
CR
Subj
TREC
DBPedia
AGNews
CB
Avg.
13B
Vanilla ICL
4
95.67.1
29.56.2
90.05.8
91.47.4
72.96.9
62.89.1
80.97.6
80.25.9
51.58.2
72.8
IDAICL
96.72.5
47.11.1
93.01.9
93.30.8
87.82.3
76.02.6
94.91.0
87.72.4
59.41.9
81.8
Vanilla ICL
8
96.77.1
39.45.6
92.36.2
92.24.8
70.85.1
71.29.1
83.74.2
79.56.3
52.43.7
75.4
IDAICL
96.92.1
49.21.9
93.41.6
92.91.9
87.53.0
79.92.1
93.60.9
88.01.7
62.42.5
82.6
33B
Vanilla ICL
4
95.57.2
29.45.6
91.75.4
91.58.1
85.16.0
70.94.4
86.64.5
76.26.1
59.25.3
76.2
IDAICL
96.51.1
46.82.4
93.61.3
92.33.3
89.32.4
79.11.5
95.62.3
88.41.9
64.62.8
82.9
Vanilla ICL
8
96.87.3
34.35.4
93.45.8
92.76.4
83.55.5
66.94.8
84.16.2
84.75.5
62.05.2
77.6
IDAICL
96.92.3
50.31.5
93.92.2
93.01.4
89.01.0
83.11.7
95.92.0
88.01.2
70.41.8
84.5
Table 7: Comparison results of Macro-F1 between Vanilla ICL and IDAICL under varying values of m on the LLaMA models with 13B and 33B parameters.
Vanilla ICL, thus proving its capability to address the class imbalance in demonstrations.
# H Varying Demonstration Permutations
Research has substantiated that the performance of ICL is sensitive to the permutation of demonstrations (Lu et al., 2022; Zhao et al., 2021). We assessed the performance of IDAICL under varying demonstration permutations. Specifically, we selected ten different sets of twelve training examples from the SST-2 datasets. For each set of examples, we shuffled the order ten times and calculated the accuracy for each permutation. The findings are depicted in Figure 10, indicating that IDAICL exhibits relatively stable performance across different demonstrations and permutations, while Vanilla ICL demonstrates high variance.
PLM
Method
m
SST-2
SST-5
MR
CR
Amazon
Subj
TREC
DBPedia
AGNews
CB
GPT-2 0.1B
Vanilla ICL
4
56.37.1
28.48.8
55.47.4
54.26.2
30.88.4
52.97.9
32.25.1
44.36.2
42.89.3
42.19.6
IDAICL
69.52.6
35.31.1
66.42.3
67.22.7
39.32.9
57.22.6
44.31.8
62.22.3
65.52.7
49.21.9
Vanilla ICL
8
60.88.3
30.66.9
57.59.7
56.05.1
33.67.8
53.75.6
33.010.7
52.15.8
45.69.1
45.46.2
IDAICL
71.41.8
36.12.9
67.61.8
68.62.2
40.00.7
58.52.5
45.61.9
63.61.1
66.91.6
50.62.7
Vanilla ICL
12
64.56.0
30.87.1
59.35.6
59.18.4
33.95.5
56.68.9
35.87.1
52.311.4
47.46.0
47.47.7
IDAICL
72.21.1
36.72.2
70.11.7
69.31.8
40.81.2
60.91.5
47.02.7
65.51.9
67.82.2
51.23.3
Vanilla ICL
16
64.36.1
33.57.1
59.96.6
61.77.5
34.66.9
56.16.2
36.95.7
54.17.2
47.98.0
48.97.7
IDAICL
72.92.5
38.02.4
69.71.3
69.92.1
41.70.9
60.61.1
46.61.9
65.92.6
65.71.0
51.82.2
GPT-2 0.3B
Vanilla ICL
4
60.87.5
26.66.8
50.57.1
52.36.1
30.55.2
53.28.3
32.88.1
50.54.8
41.35.9
42.77.1
IDAICL
78.41.7
33.12.5
66.60.9
70.32.3
40.11.5
69.41.7
45.63.3
66.22.1
62.83.7
50.41.8
Vanilla ICL
8
58.98.7
29.46.1
52.48.9
54.88.2
32.77.9
53.56.7
34.08.2
59.19.7
43.86.4
46.97.6
IDAICL
80.81.7
34.81.9
69.51.1
71.50.8
41.51.7
70.32.6
46.22.2
68.11.7
63.32.1
51.52.5
Vanilla ICL
12
62.914.4
30.67.8
55.26.2
56.16.7
34.27.5
56.87.1
36.29.8
58.07.3
46.59.3
48.66.6
IDAICL
82.22.3
36.11.8
68.92.4
72.01.5
43.70.6
71.42.4
48.31.3
70.51.9
65.22.2
52.91.4
Vanilla ICL
16
67.46.3
31.77.1
57.68.6
56.65.2
34.76.2
57.05.3
38.16.9
59.38.2
45.27.6
49.48.7
IDAICL
81.52.8
36.81.2
70.41.7
72.92.1
43.11.3
71.92.7
48.71.1
70.92.9
65.81.2
52.41.8
Table 8: Accuracy comparison between Vanilla ICL and IDAICL under varying values of m on the GPT-2 models with 0.1B and 0.3B parameters.
PLM
Method
SST-5
MR
AGNews
TREC
SST-2
Subj
DBPedia
Avg.
GPT-2 1.5B
Vanilla ICL
30.86.1
64.98.3
57.56.7
40.45.1
57.27.0
57.310.3
67.67.5
53.7
ConCa
32.87.1
74.55.1
62.76.1
45.82.5
73.98.6
68.37.4
75.04.0
61.9
PROCA∗
36.54.4
80.86.4
75.53.2
46.02.5
88.01.3
80.23.3
89.40.7
70.9
D-ConCa
31.73.3
80.93.7
77.04.1
47.12.8
86.54.4
76.85.2
86.16.3
69.4
IDAICL
40.81.9
82.11.2
80.82.4
52.02.5
89.51.8
80.12.9
91.02.5
73.8
GPT-Neo
Vanilla ICL
31.56.4
70.68.1
71.96.8
53.06.9
74.98.3
57.96.3
78.56.5
62.6
ConCa
33.94.3
78.25.3
73.63.8
55.97.2
82.09.5
71.36.4
90.03.6
69.3
PROCA∗
39.44.0
77.813.9
78.92.5
56.03.6
91.91.2
81.33.8
92.01.5
73.9
D-ConCa
32.94.1
84.62.8
81.23.9
57.64.7
91.65.3
70.92.9
85.73.1
72.1
IDAICL
42.22.5
85.91.6
83.11.9
61.41.7
91.22.4
82.33.1
93.01.5
77.0
Table 9: Accuracy comparison between IDAICL and other prediction calibration approaches using the GPT-2 (with 1.5B parameters) and GPT-Neo models, with m setting to 8. The templates used align with those utilized by Han et al. (2023). ∗indicates that the results were derived from the original paper.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/be03/be03cef1-446a-435f-8ac9-e2df27366c98.png" style="width: 50%;"></div>
Figure 9: (a) and (b): Average accuracy across ten datasets for various values of λ and τ. Optimal average performance is attained when λ = 0.5 and τ = 1. (c) and (d): Confusion matrices for the SST-2 and MR datasets under two levels of imbalance, where the proportions of the negative class in demonstrations are set to 0.1 and 0.2, respectively. When compared to Vanilla ICL, IDAICL improves the performance of the minor class. These experiments are conducted on the GPT-2 model with 1.5B parameters, setting m to 12.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f973/f973cade-269c-4c9f-9b0f-dc0d1e878501.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10: Comparison results between Vanilla ICL and IDAICL across various demonstrations and permutations. The GPT-2 model with 0.8B parameters is employed for this analysis, setting m to 12. IDAICL exhibits smaller performance variance across different demonstrations and permutations compared to Vanilla ICL.</div>
Format ID
Prompt
Label names
1
Review: This movie is amazing!
Answer: Positive
Review: Horrific movie, don’t