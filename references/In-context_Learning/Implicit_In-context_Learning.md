# Implicit In-context Learning
Zhuowei Li Zihao Xu Ligong Han Yunhe Gao Song Wen Di Liu Hao Wang Dimitris N. Metaxas
Rutgers University
{zl502}@cs.rutgers.edu
# Abstract
In-context Learning (ICL) empowers large language models (LLMs) to adapt to unseen tasks during inference by prefixing a few demonstration examples prior to test queries. Despite its versatility, ICL incurs substantial computational and memory overheads compared to zero-shot learning and is susceptible to the selection and order of demonstration examples. In this work, we introduce Implicit In-context Learning (I2CL), an innovative paradigm that addresses the challenges associated with traditional ICL by absorbing demonstration examples within the activation space. I2CL first generates a condensed vector representation, namely a context vector, from the demonstration examples. It then integrates the context vector during inference by injecting a linear combination of the context vector and query activations into the model’s residual streams. Empirical evaluation on nine real-world tasks across three model architectures demonstrates that I2CL achieves few-shot performance with zero-shot cost and exhibits robustness against the variation of demonstration examples. Furthermore, I2CL facilitates a novel representation of “task-ids”, enhancing task similarity detection and enabling effective transfer learning. We provide a comprehensive analysis of I2CL, offering deeper insights into its mechanisms and broader implications for ICL. The source code is available at: https://github.com/LzVv123456/I2CL.
arXiv:2405.14660v1
# 1 Introduction
In-context Learning (ICL) has emerged as a prominent capability of large language models (LLMs) [Brown et al., 2020]. It enables swift adaptation to new tasks during inference by prefixing a few demonstration examples prior to a test query [Wei et al., 2022, Dong et al., 2023]. ICL, characterized by its adaptability and flexibility, has prompted extensive research efforts aimed at optimizing these demonstration examples, or prompts [Rubin et al., 2022, Sorensen et al., 2022, Wu et al., 2023, Min et al., 2022a, inter alia], as well as mitigating their sensitivity to formatting, order, and recency bias [Zhao et al., 2021, Lu et al., 2022, Hao et al., 2022, inter alia]. Despite these advancements, the predominant focus on manipulating demonstration examples within the token space—where tokens are prepended to the query—presents notable limitations. This approach not only escalates computation and memory demands with each additional token but also remains susceptible to the selection and the order of demonstration examples. Such constraints pose significant challenges in scenarios with limited computational and memory resources or variable demonstration profiles, affecting ICL’s scalability and practical utility in real-world applications. In this study, we explore the usage of demonstration examples within the activation space to surmount these obstacles. Utilizing a decoder-only architecture, we observe that the primary burdens of ICL arise from the computationally intensive multi-head mechanism, which fuses information between
Preprint. Under review.
demonstration and query tokens, and the memory-intensive key-value caching scheme necessary for retaining demonstration contexts1. These observations lead us to explore two pivotal questions: Is there a memory-efficient representation of demonstration examples? And, can we integrate information in a more computationally economical manner?
Our findings suggest that both objectives are attainable by first condensing demonstration examples into a compact vector representation and then reintegrating their functionalities within the model’s activation space. Specifically, instead of concatenating demonstration examples into a text sequence before the test query, we independently extract a demonstration vector from each example. These vectors are then aggregated in a permutation-invariant manner to form a unified context vector. During inference, we inject a linear combination of this context vector and the query activations back into the model’s residual streams. We term this scheme Implicit In-context Learning (I2CL), underscoring the absence of an explicit causal relation between the demonstration examples and the test query in the token space. I2CL offers several merits over standard ICL methods. By condensing demonstration examples into a unified context vector, I2CL requires to cache only a fixed amount of activation vec-
tors, regardless of the length of demonstration tokens. I2CL also maintains a zero-shot inference speed by merging information between demonstration examples and queries through simple linear operations. We evaluate I2CL across three open-source LLMs on nine real-world text classification tasks, where it significantly surpasses the zero-shot and other comparable baselines and achieves results on par with few-shot learning (Figure 1). Importantly, I2CL demonstrates robustness against the variability of demonstration examples. It also facilitates a natural representation of ‘task-ids”, which can effectively indicate task similarity and support transfer learning. The main contributions of this work are summarized as follows:
1. We introduce I2CL, a simple and novel framework that effectively integrates a minimal set of demonstration examples within the activation space. By decomposing conventional ICL into two stages: context vectorization and context-vector injection, I2CL attains few-shot performances at zero-shot costs. 2. We empirically confirm the robustness of I2CL against variations in demonstration examples and its ability to produce a natural representation of task-ids. Leveraging these task-ids, we further propose a transfer learning strategy that can enhance performance on new tasks based on existing ones. 3. We conduct an extensive analysis to thoroughly examine the scheme of I2CL, thereby shedding light on its internal mechanisms and contributing to a deeper understanding of ICL. We also provide a deliberated discussion on the limitations and future directions of I2CL.
1. We introduce I2CL, a simple and novel framework that effectively integrates a minimal set of demonstration examples within the activation space. By decomposing conventional ICL into two stages: context vectorization and context-vector injection, I2CL attains few-shot performances at zero-shot costs. 2. We empirically confirm the robustness of I2CL against variations in demonstration examples and its ability to produce a natural representation of task-ids. Leveraging these task-ids, we further propose a transfer learning strategy that can enhance performance on new tasks based on existing ones. 3. We conduct an extensive analysis to thoroughly examine the scheme of I2CL, thereby shedding light on its internal mechanisms and contributing to a deeper understanding of ICL. We also provide a deliberated discussion on the limitations and future directions of I2CL.
# 2 Related Work
Understanding In-context Learning Besides enhancing ICL, the exploration of ICL’s internal mechanisms has attracted significant research attention. Akyürek et al. [2023] draw parallels between ICL and gradient descent in linear regression tasks, suggesting a fundamental alignment with classical optimization methods. Complementary perspectives from Von Oswald et al. [2023] and Dai et al. [2023] conceptualize ICL as a form of meta-optimization, further enriching our understanding of its operational basis. Concurrently, Xie et al. [2022] interpret ICL through the lens of implicit Bayesian inference, proposing a probabilistic foundation for the learning process. Wei et al. [2023] and Olsson et al. [2022] respectively attribute ICL’s capabilities to the establishment of input-label correspondences and the identification of so-called induction heads, highlighting the intricate interplay between data representation and model interpretability. Despite these advancements, a comprehensive understanding of ICL’s working mechanism remains elusive. Our work, I2CL, contributes additional
ut applying key-value cache, one needs to repetitively forward the same demonst
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0b9b/0b9b50f4-f784-4c4b-8ee3-007c037583af.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: (Upper-left is better) Comparison of accuracy, inference time and cached memory of different methods on Llama2-7b.</div>
insights to this domain by substantiating a novel two-stage scheme, which suggests a potential division of the underlying functionality of ICL. Most recently, Label-as-Anchors [Wang et al., 2023] inspects ICL from an information flow perspective and leverages anchor tokens to perform context compression. Similarly, I2CL also enhances the efficiency of ICL, but distinguishes itself by circumventing the need for caching latents of anchor tokens or employing multi-head attention, thereby reducing the inference cost to that of the zero-shot learning. Activation/representation Engineering An emerging research field, termed activation/representation engineering, also relates to our study. Recent endeavors by Merullo et al. [2023], Turner et al. [2023], Zou et al. [2023] have unveiled the phenomenon of steering vectors, which can be derived from a positive and negative text pair and used to steer the content generation of LLMs. Steering vectors can also be learned via gradient descent [Subramani et al., 2022]. Liu et al. [2024] applies these insights to bolster LLM safety while Li et al. [2023] explores their utility in eliciting more truthful responses from LLMs. To better understand the inner mechanism of activation engineering, the linear representation hypothesis has been studied and discussed in Li et al. [2021], Hernandez et al. [2024], Park et al. [2023]. Our work, I2CL, further corroborates this hypothesis, aligning with and expanding upon the pioneering works by Mikolov et al. [2013], Meng et al. [2023], Hernandez et al. [2023]. Central to this discourse, the idea of task/function vector [Hendel et al., 2023, Todd et al., 2024] resonates with the core premise of I2CL. Both methodologies extract task/function vectors from the demonstration examples and them to improve zero-shot performance. I2CL stands out not only due to its superior performance, but also through its simplicity, namely, its avoidance of the need for task- or architecture-specific hyperparameter tuning, such as selecting attention heads through causal mediation or determining target layers via extra validation sets. Conceptually, we ask an even simpler question: In practice, is there a more economical method to absorb a tiny amount of demonstration examples than by prefixing them in context?
# 3 Methodology
# 3.1 Preliminaries
Residual Stream We adopt the mathematical interpretation from Elhage et al. [2021], viewing the hidden states across layers and at each token position as a residual stream. This perspective views each attention head and multi-layer perceptron (MLP) module as engaging with residual streams, facilitating both the addition and deletion of information within them. At any given layer l and token position t, the residual stream rt l is defined recursively:
where at l denotes the integrated output of the multi-head attention (MHA) module, and mt l signifies the MLP’s output, contingent also on at l. In this framework, MHA promotes the information fusion across residual streams, whereas the MLP functions akin to an associative memory [Geva et al., 2021, Dai et al., 2022] that retrieves information encapsulated within its weights. In-context Learning Given an unseen task, ICL assumes the existence of a set of N demonstration examples D = {d1, d2, . . . , dN}, each comprising an instructional pair di = (xi, yi) that includes an input sentence2 and its corresponding label. ICL operates by first concatenating the demonstration set D with the test query xq, forming an input prompt p = [D, xq]. The objective is then to predict the correct response yq from a finite set of discrete labels C via yq = arg maxy∈C P(y | p).
where at l denotes the integrated output of the multi-head attention (MHA) module, and mt l signifies the MLP’s output, contingent also on at l. In this framework, MHA promotes the information fusion across residual streams, whereas the MLP functions akin to an associative memory [Geva et al., 2021, Dai et al., 2022] that retrieves information encapsulated within its weights.
In-context Learning Given an unseen task, ICL assumes the existence of a set of N demonstration examples D = {d1, d2, . . . , dN}, each comprising an instructional pair di = (xi, yi) that includes an input sentence2 and its corresponding label. ICL operates by first concatenating the demonstration set D with the test query xq, forming an input prompt p = [D, xq]. The objective is then to predict the correct response yq from a finite set of discrete labels C via yq = arg maxy∈C P(y | p).
# 3.2 Context Vectorization
To overcome the inefficiencies associated with key-value caching system, we isolate the reasoning process of demonstration examples from that of the query, and introduce an independent vectorization of each demonstration pair di. These vectors are subsequently merged in the activation space. Specifically, we define a function V , capable of generating a vector representation for each demonstration example: di = V (di). Then, a function F is applied to aggregate these demonstration vectors into a unified context vector v = F({di}N i=1). Note that F is designed to be permutation invariant, ensuring a unique vector representation for a given set of demonstration examples.
2Input xi also includes formatting texts like: “Question:”, “Answer Type:”
(1)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4c66/4c660be8-9d2b-400b-80f7-3f1b8a60a94c.png" style="width: 50%;"></div>
<div style="text-align: center;">End residual stream of  demonstration example i</div>
Figure 2: A schematic overview of I2CL, including a single layer for the illustrative purpose. I2CL begins by extracting the output activations of the MHA and MLP modules at the end residual stream of each demonstration example. These activation vectors are then fused in a permutation-invariant manner to form layer-wise context vectors. At inference, a linear combination of context vectors and activations from the test query is added to all residual streams.
In our implementation, V consists of a pre-trained tokenizer and a frozen LLM parameterized by its pre-trained weights. The demonstration vectors are extracted from the output activations of both MHA and MLP modules at the end residual stream across all layers:
Here, e denotes the end position within a token sequence responsible for next-token prediction, and L represents the total number of layers. For the aggregation function, we compute an element-wise arithmetic mean of each vector component across the demonstration examples:
� � The rationale of above designs is twofold. First, we argue that the end residual stream encapsulates the essential information from each example. This is supported by the dynamics of next-token prediction and empirical findings from recent studies [Hendel et al., 2023, Zou et al., 2023, Todd et al., 2024]. Second, inspired by the linear representation hypothesis [Park et al., 2023], we premise various demonstration vectors can undergo linear transformations at different levels of abstraction.
# 3.3 Context Injection
Having established a unique vector representation, v, for the demonstration examples, I2CL seeks to enhance zero-shot performance by integrating this context information with that of the query. While typical implementations leverage the multi-head attention module for this purpose, I2CL utilizes a simpler, yet effective, linear operation to augment the query activations with context vectors. Consider rt l, the residual stream at the t-th token and l-th layer of a test query. Instead of merely adding the output activations from the MHA and MLP to their corresponding residual streams, we inject a linear combination of these activations with context vectors:
Here, λa, βa, λm, βm are four layer-wise scalars used to adjust the proportion to which the context vectors and query activations are blended. They also affect the amplitude of vectors. By default, we apply this information fusion process to all residual streams of a given query, implemented using
<div style="text-align: center;">Residual stream at -th  token of query t</div>
(2)
(4)
zero-shot inference. Above context injection method incurs minimal computational overhead, involving only two scalar multiplications and an element-wise addition, a process more efficient than the attention mechanism As a result, the inference speed of I2CL matches that of standard zero-shot learning. Since a constan number of activations is necessary for caching, I2CL also retains approximately the same memory usage as zero-shot. We refer to Figure 2 for a schematic illustration of the context vectorization and injection procedures.
# 3.4 Noisy Self-calibration
To achieve generic and nuanced control over the information fusion process, we propose estimating the linear coefficients c = {λa l , βa l , λm l , βm l }L l=1 using a gradient-based method applied to the demonstration examples. It is worth recapitulating that I2CL does not require any additional data beyond the provided demonstration examples. Concretely, we initialize λ = 0.1, β = 1.0 to promote a modest initial addition of information, and update these coefficients by minimizing the perplexity of label tokens, represented by the following loss function:
where P(·) denotes the induced probability distribution over the entire vocabulary at the end token position from the last layer. To bolster the robustness and adaptability of the linear coefficients to potential downstream variations, we introduce random Gaussian noises η ∼N(0, I) into the residual streams during the calibration phase:
where γ is a scalar employed to modulate the intensity of the noise, and || · ||2 denotes the L2 norm. The o represents the intermediate state of a residual stream. Given that only a few linear coefficients (totaling 4L) are updated, this process is remarkably efficient, consuming 1-2 minutes on a single A100 GPU (40G). Importantly, calibrated linear
where γ is a scalar employed to modulate the intensity of the noise, and || · ||2 denotes the L2 norm. The o represents the intermediate state of a residual stream. Given that only a few linear coefficients (totaling 4L) are updated, this process is remarkably efficient, consuming 1-2 minutes on a single A100 GPU (40G). Importantly, calibrated linear coefficients are demonstration agnostic—they require calibration only once per task and exhibit excellent generalization ability to unseen demonstration examples (see Section 4.3).
Given that only a few linear coefficients (totaling 4L) are updated, this process is remarkably efficient, consuming 1-2 minutes on a single A100 GPU (40G). Importantly, calibrated linear coefficients are demonstration agnostic—they require calibration only once per task and exhibit excellent generalization ability to unseen demonstration examples (see Section 4.3).
# 4 Experiments
In empirical section, we begin by detailing the architectures, tasks, and configurations used in our study, followed by a comparative analysis of I2CL against other relevant techniques. We then delve into the formation of context vectors and the characteristics of the calibrated linear coefficients, demonstrating the robustness of I2CL, as well as identifying the function of calibrated coefficients as task-ids. This section concludes with an extensive ablation study to highlight our rationales. Models We evaluate I2CL using three open-source decoder-only architectures: GPT2-XL [Radford et al., 2019], GPT-J-6B [Wang and Komatsuzaki, 2021], and Llama2-7b [Touvron et al., 2023]. We selected these models based on their suitability for our computational resources and their range in size from relatively small (1.5B) to large (7B). The primary results, reported in subsequent sections, are based on Llama2-7b, with additional results for other architectures detailed in Appendix C.1. We observed consistent trends across all models. Tasks Initially, we use the four tasks from Wang et al. [2023], including sentiment analysis: SST2 [Socher et al., 2013], emotion classification: EmoC [Chatterjee et al., 2019], question classification: TREC [Voorhees and Tice, 2000], and topic classification: AGNews [Zhang et al., 2015]. We further enrich our experiments with five additional datasets, encompassing 5-way sentiment analysis: SST-5 [Socher et al., 2013], movie review classification: MR [Pang and Lee, 2005], 14-way topic
(5)
(6)
Table 1: Comparison between I2CL and other methods on Llama2-7b. Mean accuracy and standard deviation across five random seeds are reported for each task. The best results are highlighted in bold, and the second-best results are shown in blue. In addition to a practical gauge of the inference speed and memory usage (see Fig 1), we include an examination of cached parameters. Here, M, D, and L denote the number of demonstration tokens, model dimension, and architecture layers, respectively. P indicates the number of extra learnable tokens in the Soft-prompt method, and 1/K represents the compression rate in the Label-anchor method.
Task
Zero-shot
Few-shot (ICL)
Noise vector
Soft-prompt
Label-anchor
Task-vector
I2CL (ours)
SST-2 (%) ↑
83.00
94.44±1.44
49.88±0.24
56.24±6.99
83.32±5.95
81.44±4.73
87.68±2.47
SST-5 (%) ↑
27.00
41.72±3.68
20.56±0.64
24.24±2.96
27.68±4.21
25.96±0.59
39.12±2.69
TREC (%) ↑
50.00
77.32±4.41
20.12±10.92
55.20±4.14
77.48±3.49
65.68±1.93
78.56±5.32
AGNews (%) ↑
70.20
85.68±2.00
27.32±2.82
78.00±7.60
83.72±1.04
79.68±4.07
85.48±1.16
Subj (%) ↑
51.40
52.56±3.09
49.64±0.48
57.40±4.93
53.00±2.95
58.56±4.91
73.84±3.84
HateSpeech18 (%) ↑
54.20
70.24±5.80
59.84±8.04
59.56±6.96
64.52±8.09
67.68±3.70
69.88±5.67
DBPedia (%) ↑
72.00
96.64±0.48
7.28±0.37
74.40±6.43
81.40±3.67
89.48±2.58
90.16±1.86
EmoC (%) ↑
41.80
75.48±1.63
26.76±3.04
35.08±5.29
59.12±10.60
44.64±3.53
63.72±1.37
MR (%) ↑
73.60
93.24±0.50
50.12±0.24
54.32±1.76
84.40±5.89
82.32±5.37
87.68±2.26
Macro avg. acc. (%) ↑
58.13
76.37
34.61
54.94
68.29
66.16
75.12
# cached parameters ↓
0
2MDL
2DL
(2M + P)DL
2(M/K)DL
D
2DL
classification: DBPedia [Zhang et al., 2015], subjectivity status categorization: Subj [Pang and Lee, 2004], and hate speech detection: hate_speech18 [de Gibert et al., 2018]. We leverage the HuggingFace version of the data [Lhoest et al., 2021] and uniformly sample 500 data points from the development set for evaluation.
Experimental Setup Our experimental setup remains consistent across all tasks unless otherwise specified. For each task, we randomly sample five demonstration examples per class4 following the practice described in Wang et al. [2023] to avoid majority label bias [Zhao et al., 2021] and provide a strong few-shot performance. No instruction is used to describe the task. Input sequences are formed using simple manually designed templates (included in Appendix A). For evaluation, we report the macro-average accuracy across nine tasks, computed under five random seeds. For the calibration process, we optimize linear coefficients for 100 epochs on the same demonstration set using the AdamW [Loshchilov and Hutter, 2019] optimizer. The learning rate starts at 1 × 10−2 and anneals to 1 × 10−5 according to a cosine scheduler. This calibration configuration is applied uniformly across all architectures and tasks without tailoring.
# 4.1 Benchmarking I2CL
I2CL Achieves Few-shot Performance at Zero-shot Cost I2CL is designed to enhance zero-shot performance, aspiring to match or exceed those of few-shot outcomes. As shown in Table 1, I2CL significantly outperforms the zero-shot counterpart by 17% in absolute accuracy and is marginally behind (by around 1%) the few-shot learning. An interesting outcome arises from the Subj task, where the instructions (i.e., demonstration examples) do not function effectively with ICL, yet are well adhered to under I2CL. Critically, I2CL caches only a fixed number of vectors, independent of the context token length, and retains the speed of zero-shot learning. These results demonstrate I2CL’s potential as a more economical alternative to traditional ICL. Comparison with Other Methods To further validate the effectiveness of I2CL, we compare it with other comparable techniques. (1) Noise vector: We substitute the context vector with random noise in order to assess its necessity. (2) Soft-prompt: Another approach of using demonstration examples is to treat them as training data and apply the soft-prompt technique. To this end, we compare I2CL with representative Prompt-tuning [Lester et al., 2021] method to demonstrate the superior data efficiency of I2CL. (3) Label-anchor: We also compare I2CL with the context compression technique presented in Wang et al. [2023] under the same setup. Following the original methodology, we include formatting tokens as anchors. (4) Task-vector: We compare I2CL to the task vector [Hendel et al., 2023] method, which also improves zero-shot performance. Adhering to their setup, we use a holdout validation set to identify the optimal replacement layer for each task. We refer to Appendix B for reproduction details. As displayed in Table 1, replacing the context vector with random noise significantly degrades performance, yielding subpar results compared to zero-shot. The soft-prompt technique is beneficial for only a few tasks under data scarcity, and it can also impair the performance.
e exception is DBPedia, where we use only one example per class due to the limita
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5a9e/5a9ef9d5-e4f9-46e9-b71f-1cd2c8a5a98d.png" style="width: 50%;"></div>
Figure 3: Left: Evaluation of I2CL and few-shot learning under deficient demonstrations. The symbol ∗denotes the results under deficient demonstration examples. “Unseen demo” refers to the evaluation of calibrated coefficients on unseen demonstrations. Middle: Analysis of the influencing factors of context vectors. “Random-labe” indicates random input-label mappings. “Random-order” refers to the random permutation of words. “W/o format” signifies excluding the template tokens in the creation of context vectors. Right: t-SNE plot of context vectors. Each circle denotes a context vector generated using a group of randomly sampled demonstration examples.
The label-anchor method exhibits a decent upgrade over the zero-shot baseline, achieving results most close to I2CL. However, its inference cost remains dependent on the length of demonstration tokens, with reductions proportional to their compression rate. Alike I2CL, the task-vector method also enjoys zero-shot inference expense; Nevertheless, its performance is sensitive to the downstream task and is slightly worse on average than the label-anchor method. It also requires an additional hyperparameter selection step, i.e., choosing the target replacement layer using a hold-out validation set. I2CL achieves the best performance on all tasks and surpasses its rivals by a considerable margin.
# 4.2 On the Formation of the Context Vector
to investigate its properties and the factors affecting its functionality. Deficient Demonstrations ̸= Deficient I2CL It is well-known that ICL is sensitive to the choice of demonstration examples [Dong et al., 2023], even the order [Zhao et al., 2021, Lu et al., 2022]. In this context, we investigate whether deficient demonstration examples necessarily lead to degraded I2CL performance. We sample 20 different groups of demonstration examples and keep the group with the poorest ICL performance on a hold-out validation set. Using this poorest group, we then perform both ICL and I2CL on an evaluation set. Referencing Figure 3 (left), the deficient demonstration group leads to a severely downgraded ICL performance (−7%), while I2CL performance is barely affected (−0.5%). This result suggests that less attention can be paid to the selection of demonstration examples when using I2CL. We hypothesize that this resiliency is due to the emergence of more abstract and generic concepts within the context vector, making them robust against token space variations. To corroborate this hypothesis, we visualize the context vectors using the t-SNE technique [van der Maaten and Hinton, 2008] in Figure 3 (right). Context vectors under different demonstration groups are close to each other and different tasks hold distinct context vectors. Influencing Factors of the Context Vector Generation We have shown that context vectors are robust against the selection of demonstration examples. Here, we explore various factors that may impact the context vector’s functionality. Inspired by counter-intuitive findings reported in [Zhang et al., 2022, Min et al., 2022b], we experiment with two variations of demonstration examples. Random-label: pairing each input sentence with a random label from the task, rather than the groundtruth label. Random-token: randomly permuting all words within a demonstration example. Another ineligible factor involves the inclusion of formatting tokens. W/o-format: removing formatting words from a demonstration example, retaining only the input sentence and its corresponding label, e.g., “Review: This is a great movie. Label: positive. As revealed in Figure 3 (middle), input-label mapping relations have minimal impact on context vector formation, mirroring the observations made in ICL. However, these mapping relations are crucial for calibration purposes; using random input-label mapping during calibration undermines the functionality of I2CL. Unlike the phenomenon observed in [Zhang et al., 2022], word sequence holds essential statistics under a casual architecture. Randomly permuting input sequences yields degenerated context vectors and a clearly downgraded I2CL performance. Lastly, formatting tokens contribute substantially—removing them leads to a noticeable performance drop.
Deficient Demonstrations ̸= Deficient I2CL It is well-known that ICL is sensitive to the choice of demonstration examples [Dong et al., 2023], even the order [Zhao et al., 2021, Lu et al., 2022]. In this context, we investigate whether deficient demonstration examples necessarily lead to degraded I2CL performance. We sample 20 different groups of demonstration examples and keep the group with the poorest ICL performance on a hold-out validation set. Using this poorest group, we then perform both ICL and I2CL on an evaluation set. Referencing Figure 3 (left), the deficient demonstration group leads to a severely downgraded ICL performance (−7%), while I2CL performance is barely affected (−0.5%). This result suggests that less attention can be paid to the selection of demonstration examples when using I2CL. We hypothesize that this resiliency is due to the emergence of more abstract and generic concepts within the context vector, making them robust against token space variations. To corroborate this hypothesis, we visualize the context vectors using the t-SNE technique [van der Maaten and Hinton, 2008] in Figure 3 (right). Context vectors under different demonstration groups are close to each other and different tasks hold distinct context vectors.
Influencing Factors of the Context Vector Generation robust against the selection of demonstration examples. Here, we explore various factors that may impact the context vector’s functionality. Inspired by counter-intuitive findings reported in [Zhang et al., 2022, Min et al., 2022b], we experiment with two variations of demonstration examples. Random-label: pairing each input sentence with a random label from the task, rather than the groundtruth label. Random-token: randomly permuting all words within a demonstration example. Another ineligible factor involves the inclusion of formatting tokens. W/o-format: removing formatting words from a demonstration example, retaining only the input sentence and its corresponding label, e.g., “Review: This is a great movie. Label: positive. As revealed in Figure 3 (middle), input-label mapping relations have minimal impact on context vector formation, mirroring the observations made in ICL. However, these mapping relations are crucial for calibration purposes; using random input-label mapping during calibration undermines the functionality of I2CL. Unlike the phenomenon observed in [Zhang et al., 2022], word sequence holds essential statistics under a casual architecture. Randomly permuting input sequences yields degenerated context vectors and a clearly downgraded I2CL performance. Lastly, formatting tokens contribute substantially—removing them leads to a noticeable performance drop.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/09bd/09bd90a9-e826-4ee8-9218-48b99675f21f.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Left: t-SNE visualization of calibrated linear coefficients. Each circle denotes a runtime with a random seed. Middle: This image displays the transfer results among various tasks. Each row represents a source task and each column denotes a target task. Red and blue colors signify positive and negative transfer outcomes, respectively. Right: This plot shows the calibrated linear coefficients for SST-2. λa, βa, λm, βm are the layer-wise coefficients described in Equation 4. 4.3 Analysis of Calibrated Linear Coefficients</div>
# 4.3 Analysis of Calibrated Linear Coefficients
# With a grasp on the formation of context vectors, we now delve into the properties of calibrated linear coefficients to enhance our understanding of the internal mechanisms of I2CL.
With a grasp on the formation of context vectors, we now delve into the properties of calibrated linear coefficients to enhance our understanding of the internal mechanisms of I2CL.
Linear Coefficients are Generalizable Given the demonstration-dependent context vectorization and calibration process, it is natural to consider I2CL as a test-time optimization framework. However, we show that the coefficients are generalizable and require calibration once per task. In this context, we evaluate a set of calibrated coefficients on five new context vectors generated using five unseen groups of demonstrations. As exhibited in Figure 3 (left), the calibrated linear coefficients generalize well to unseen demonstrations. We attribute this generalization capability to two factors: (1) the inherent robustness of the context vector against token space variations, as demonstrated in Section 4.2, and (2) the sufficiency of calibrated linear coefficients, independent of context vectors and query activation, to uniquely represent a task. We substantiate the second premise in the following subsection. Calibrated Coefficients Embed Task Semantics Here, we investigate whether calibrated coefficients alone are adequate to serve as "task id". In pursuit of this goal, we concatenate calibrated coefficients across layers to form a 1D vector, which we then visualize using t-SNE. As illustrated in Figure 4 (left), the calibrated linear coefficients are tightly clustered for instances associated with the same task and separate otherwise, providing complementary evidence for their good generalization capability. One exception comes from the proximity between MR and SST-2. This phenomenon is not unexpected, as both tasks originate from the same underlying distribution (i.e., rotten tomatoes movie reviews), suggesting that similarities among calibrated coefficients may indicate potential transferability among tasks. To explore this further, we transfer the context vectors and calibrated linear coefficients from a source task to various target tasks. We then measure the differences between the transferred results and their respective zero-shot outcomes to identify both positive and negative transfers. Figure 4 (middle) shows that performance on MR can be effectively enhanced by transferring context vectors and calibrated coefficients from SST-2, and vice versa. Similarly, both SST-2 and MR benefit from transfers from SST-5, likely due to their shared focus on sentiment analysis. An intriguing aspect of our findings is the asymmetric nature of transferability among different tasks; a successful transfer in one direction does not necessarily guarantee success in the opposite direction (e.g., transfer between HateSpeech18 and SST-2). Enhance New Task with Existing Anchors I2CL posits two pivotal features distinct from
� where τ is the temperature. Finally, we reinitialize vnew and cnew using the weighted average of retained context vectors and
Table 2: Transfer learning result. Only tasks having more than one similar task (according to h) in our task curation are exhibited.
I2CL
Task
W/o transfer (%)
W/ transfer (%)
SST-5
39.12±2.69
43.24±3.70
MR
87.68±2.47
89.99±2.83
(7)
coefficients: vavg = � i∈I P(i)vi, cavg = � i∈I P(i)ci, and perform another around of calibration. We refer to Appendix B for detailed algorithms and implementation. Empirical results in Table 2 demonstrate a clear benefit of the proposed transfer learning method. Injection Dynamics Thus far, we have demonstrated that a few static scalars, conditioned on appropriate context vectors, can successfully execute a wide range of NLP tasks. Herein, we delve into how context vectors are injected. Using the SST-2 as an example, we examine the coefficient values across different layers (additional plots in Appendix D). One reasonable speculation postulates a gradual addition of context vectors to the residual streams. Nevertheless, observations from Figure 4 (right) reveal a more nuanced control over the information injection process. Instead of monotonically adding (+) more information to the residual streams, I2CL also allows the deletion (−) of information at certain layers, with coefficient values fluctuating across different layers.
Injection Dynamics Thus far, we have demonstrated that a few static scalars, conditioned on appropriate context vectors, can successfully execute a wide range of NLP tasks. Herein, we delve into how context vectors are injected. Using the SST-2 as an example, we examine the coefficient values across different layers (additional plots in Appendix D). One reasonable speculation postulates a gradual addition of context vectors to the residual streams. Nevertheless, observations from Figure 4 (right) reveal a more nuanced control over the information injection process. Instead of monotonically adding (+) more information to the residual streams, I2CL also allows the deletion (−) of information at certain layers, with coefficient values fluctuating across different layers.
<div style="text-align: center;">Table 3: Target modules.</div>
<div style="text-align: center;">Table 4: Target layers.</div>
Table 4: Target layers.
Name
Accuracy (%)
Zero-shot
58.13
Early
58.18
Middle
64.07
Late
64.03
All (ours)
75.12
Table 3: Target modules.
Name
Accuracy (%)
Zero-shot
58.13
MHA
66.97
MLP
70.27
Hidden state
56.80
MHA+MLP (ours)
75.12
<div style="text-align: center;">Table 7: Vectorization method.</div>
Name
Accuracy (%)
Zero-shot
58.13
Concatenate
70.80
PCA
75.22
Average (ours)
75.12
# 4.4 Ablation Study
In this section, we conduct a comprehensive ablation study on the module, layer, and token position of injection, as well as the noise scale. We also explore various vectorization and injection formulas to highlight the rationale behind our designs. Target Module I2CL extracts and injects activations at both the MHA and MLP modules. To identify the contribution of each module, we test using either MHA or MLP independently and also consider leveraging the hidden state at each layer. As shown in Table 3, extracting and injecting context vectors at either MHA or MLP proves beneficial, with MLP showing a clear advantage. We hypothesize this is due to the additional engagement of information stored in the MLP weights. However, targeting the hidden state does not lead to improvement, likely due to the accumulation effect that complicates harnessing the scales of hidden states. Target Layer I2CL encompasses all layers in a large language model (LLM), eliminating the need for a layer selection process, which is often a designated hyperparameter. This model-agnostic approach not only simplifies the setup but also enhances performances, as evidenced in Table 3. We divide the model into three sub-parts—early, middle, and late—each containing one-third of the total layers. We then apply I2CL only within the target layer range. The middle and late layers prove more effective than the early layers, and injecting across all layers provides a clear performance boost, highlighting the importance of fusing information at all levels of abstraction. Injection Position By default, I2CL injects context vectors into all residual streams during inference. To justify this choice, we test injections only at random, first, and last residual streams. As shown in Table 5, and aligning with common intuition, injecting at the end residual stream yields the largest improvement compared to other positions, although it still shows a significant gap compared to injecting at all residual streams. Noise Scale Here, we evaluate the impact of noise strength during the calibration phase. As demonstrated in Table 6, an appropriate noise scale leads to a clear performance gain. Conversely, an improper noise scale will disrupt information propagation during inference, resulting in deteriorated outcomes. Through empirical testing, we identify τ = 0.001 as a suitable scale and have applied it across all tasks. Vectorization Method For context vectorization, we treat each demonstration example separately and post-merge them using an element-wise algorithmic average. A straightforward alternative is to
<div style="text-align: center;">Table 5: Injection position.</div>
Table 5: Injection position
Name
Accuracy (%)
Zero-shot
58.13
Random
59.86
First
62.14
Last
66.75
All (ours)
75.12
<div style="text-align: center;">Table 8: Injection formula.</div>
Table 8: Injection formula.
Name
Accuracy (%)
Zero-shot
58.13
λv + a, λ > 0
63.63
(λv + (1 −λ)a) × β, β > 0
71.39
λv + βa (ours)
75.12
concatenate all demonstration examples as in typical ICL, and extract the context vector from the end residual stream. Furthermore, we evaluate another post-fusion strategy that applies PCA to extract the first principal component vector as the context vector. According to Table 7, the concatenation method significantly underperforms the post-fusion counterparts. The PCA variant achieves a negligible improvement; therefore, we employ the average operator by default for its simplicity. Injection Formula I2CL utilizes a linear combination to blend context vectors with query activations. Let v denote the context vector and a indicates activation; we use λc + βa. One simplification is to view the injection process as solely adding the context vector to the activation: λv + a with λ > 0. Another common formula involves constraining the sum of linear coefficients to one and allowing a separate scale factor: (λv + (1 −λ)a) × β, β > 0. According to Table 8, a linear combination with no constraints achieves the best overall performance. Therefore, we conjecture that it is critical to allow not only information addition but also deletion, i.e., permitting a negative sign for the linear coefficient, and to scale each vector independently. Observations in Figure 4 (right) support our conjecture.
# 5 Discussion
A Hypothetical Two-Stage Interpretation of ICL The effectiveness of I2CL suggests a conceptual two-stage internal mechanism within ICL. In the first stage, the information from each demonstration example is encapsulated within its respective end residual stream. In the second stage, this encapsulated information is accessed and integrated into the residual streams of the query. Commonly, both stages are seamlessly executed by non-linear multi-head attention modules within transformer layers. I2CL reveals that the end residual stream of a sequence functions as an effective anchor to condense context information, and by leveraging activation vectors from it, we can effectively approximate an attention-based information fusion scheme using only linear operations. We hypothesize that this phenomenon arises due to the emergence of certain linear representations during the pre-training stage. These linear properties enable a vector interface that facilitates the formation and transportation of functions through simple linear operations. While we do not claim the above interpretation as a rigorous internal mechanism of ICL, it aligns with our empirical observations and we hope it can inspire future work aimed at a deeper understanding of the ICL. Potential Design Although the end residual stream contains a rich repository of information for the given token sequence, other token positions may also possess valuable attributes, as noted in recent studies [Hendel et al., 2023, Todd et al., 2024, Liu et al., 2024]. Recent research also highlights the importance of formatting tokens during information propagation [Wang et al., 2023, Bai et al., 2024]. We believe our approach could benefit from a more sophisticated design of demonstration vectorization. Additionally, while we currently use a single scalar to gauge the strength of aggregated multi-head attention, allowing finer granularity—such as separate strength scalars for different attention heads—might enhance our system’s performance, albeit at the cost of increased parameters. Limitations and Future Work Our research is subject to several limitations. Firstly, we confine the scope of our initial exploration to standard classification tasks, leaving more sophisticated tasks for future research. It would be intriguing to extend I2CL to the realm of open-ended generation tasks and those involving multi-hop reasoning processes. Secondly, I2CL necessitates access to and caching of intermediate activations from language models, which may not be feasible with state-of-the-art commercial models (e.g., GPT-4, Gemini, Claude3). Lastly, we have evaluated I2CL on relatively modest-sized models, and scaling our evaluations to models with hundreds of billions of parameters could yield additional insights.
# 6 Conclusion
In this study, we introduce Implicit In-context Learning (I2CL), a simple and novel framework that integrates a minimal set of demonstration examples within the activation space of LLMs. Diverge from ICL, I2CL eliminates the need for caching the latents of demonstration tokens and replaces the non-linear information fusion process with linear operations. Therefore, I2CL reduces both computational and memory expenses during inference to that of zero-shot level. Moreover, I2CL is validated to be robust against token space variations, and it facilitates a novel representation of “task-ids”, enhancing task similarity detection and enabling effective transfer learning. Empirical
evidence on nine real-world tasks across three different models demonstrates I2CL as a more efficient and robust alternative to ICL. Through a set of in-depth analyses, we shed light on the internal mechanics of I2CL and provide further insights into the working scheme of ICL.
# Acknowledgments
We extend our sincere gratitude to Prof. Diyi Yang for her constructive suggestions on the empirical design of this study. Her expertise and insights were invaluable to this research.
# References
E. Akyürek, D. Schuurmans, J. Andreas, T. Ma, and D. Zhou. What learning algorithm is in-context learning? investigations with linear models, 2023.
learning? investigations with linear models, 2023. Y. Bai, H. Huang, C. S.-D. Piano, M.-A. Rondeau, S. Chen, Y. Gao, and J. C. K. Cheung. Identifying and analyzing task-encoding tokens in large language models, 2024. T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/ file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf. A. Chatterjee, K. N. Narahari, M. Joshi, and P. Agrawal. SemEval-2019 task 3: EmoContext contextual emotion detection in text. In J. May, E. Shutova, A. Herbelot, X. Zhu, M. Apidianaki, and S. M. Mohammad, editors, Proceedings of the 13th International Workshop on Semantic Evaluation, pages 39–48, Minneapolis, Minnesota, USA, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/S19-2005. URL https://aclanthology.org/S19-2005. D. Dai, L. Dong, Y. Hao, Z. Sui, B. Chang, and F. Wei. Knowledge neurons in pretrained transformers. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8493–8502, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long. 581. URL https://aclanthology.org/2022.acl-long.581. D. Dai, Y. Sun, L. Dong, Y. Hao, S. Ma, Z. Sui, and F. Wei. Why can GPT learn in-context? language models secretly perform gradient descent as meta-optimizers. In A. Rogers, J. Boyd-Graber, and N. Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023, pages 4005–4019, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.247. URL https://aclanthology.org/2023.findings-acl. 247. O. de Gibert, N. Perez, A. García-Pablos, and M. Cuadros. Hate speech dataset from a white supremacy forum. In D. Fišer, R. Huang, V. Prabhakaran, R. Voigt, Z. Waseem, and J. Wernimont, editors, Proceedings of the 2nd Workshop on Abusive Language Online (ALW2), pages 11–20, Brussels, Belgium, Oct. 2018. Association for Computational Linguistics. doi: 10.18653/v1/ W18-5102. URL https://aclanthology.org/W18-5102. Q. Dong, L. Li, D. Dai, C. Zheng, Z. Wu, B. Chang, X. Sun, J. Xu, L. Li, and Z. Sui. A survey on in-context learning, 2023. N. Elhage, N. Nanda, C. Olsson, T. Henighan, N. Joseph, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1:1, 2021. M. Geva, R. Schuster, J. Berant, and O. Levy. Transformer feed-forward layers are key-value memories. In M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484–5495, Online and
Y. Bai, H. Huang, C. S.-D. Piano, M.-A. Rondeau, S. Chen, Y. Gao, and J. C. K. Cheung. Identifying and analyzing task-encoding tokens in large language models, 2024.
18653/v1/2021.emnlp-main.446. URL https://aclanthology.org/2021.emnlp-main.446. Y. Hao, Y. Sun, L. Dong, Z. Han, Y. Gu, and F. Wei. Structured prompting: Scaling in-context learning to 1,000 examples, 2022. R. Hendel, M. Geva, and A. Globerson. In-context learning creates task vectors. In H. Bouamor, J. Pino, and K. Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 9318–9333, Singapore, Dec. 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.624. URL https://aclanthology.org/2023. findings-emnlp.624. E. Hernandez, B. Z. Li, and J. Andreas. Inspecting and editing knowledge representations in language models, 2023. E. Hernandez, A. S. Sharma, T. Haklay, K. Meng, M. Wattenberg, J. Andreas, Y. Belinkov, and D. Bau. Linearity of relation decoding in transformer language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= w7LU2s14kE. B. Lester, R. Al-Rfou, and N. Constant. The power of scale for parameter-efficient prompt tuning. In M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. doi: 10.18653/v1/ 2021.emnlp-main.243. URL https://aclanthology.org/2021.emnlp-main.243. Q. Lhoest, A. Villanova del Moral, Y. Jernite, A. Thakur, P. von Platen, S. Patil, J. Chaumond, M. Drame, J. Plu, L. Tunstall, J. Davison, M. Šaško, G. Chhablani, B. Malik, S. Brandeis, T. Le Scao, V. Sanh, C. Xu, N. Patry, A. McMillan-Major, P. Schmid, S. Gugger, C. Delangue, T. Matussière, L. Debut, S. Bekman, P. Cistac, T. Goehringer, V. Mustar, F. Lagunas, A. Rush, and T. Wolf. Datasets: A community library for natural language processing. In H. Adel and S. Shi, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 175–184, Online and Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-demo.21. URL https://aclanthology.org/2021.emnlp-demo.21. B. Z. Li, M. Nye, and J. Andreas. Implicit representations of meaning in neural language models. In C. Zong, F. Xia, W. Li, and R. Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1813–1827, Online, Aug. 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.143. URL https://aclanthology.org/2021.acl-long.143. K. Li, O. Patel, F. Viégas, H. Pfister, and M. Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 41451–41530. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ 81b8390039b7302c909cb769f8b6cd93-Paper-Conference.pdf. S. Liu, H. Ye, L. Xing, and J. Zou. In-context vectors: Making in context learning more effective and controllable through latent space steering, 2024. I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7. Y. Lu, M. Bartolo, A. Moore, S. Riedel, and P. Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.556. URL https://aclanthology.org/2022.acl-long.556.
transformer. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=MkbcAHIYgyS. J. Merullo, C. Eickhoff, and E. Pavlick. A mechanism for solving relational tasks in transformer language models, 2023. T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. In C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Weinberger, editors, Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc., 2013. URL https://proceedings.neurips.cc/paper_files/paper/ 2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf. S. Min, M. Lewis, H. Hajishirzi, and L. Zettlemoyer. Noisy channel language model prompting for few-shot text classification. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5316–5330, Dublin, Ireland, May 2022a. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.365. URL https://aclanthology.org/2022.acl-long.365. S. Min, X. Lyu, A. Holtzman, M. Artetxe, M. Lewis, H. Hajishirzi, and L. Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work?, 2022b. C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, D. Drain, D. Ganguli, Z. Hatfield-Dodds, D. Hernandez, S. Johnston, A. Jones, J. Kernion, L. Lovitt, K. Ndousse, D. Amodei, T. Brown, J. Clark, J. Kaplan, S. McCandlish, and C. Olah. In-context learning and induction heads, 2022. B. Pang and L. Lee. A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics (ACL-04), pages 271–278, Barcelona, Spain, July 2004. doi: 10.3115/ 1218955.1218990. URL https://aclanthology.org/P04-1035. B. Pang and L. Lee. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In K. Knight, H. T. Ng, and K. Oflazer, editors, Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL’05), pages 115–124, Ann Arbor, Michigan, June 2005. Association for Computational Linguistics. doi: 10.3115/1219840.1219855. URL https://aclanthology.org/P05-1015. K. Park, Y. J. Choe, and V. Veitch. The linear representation hypothesis and the geometry of large language models. In Causal Representation Learning Workshop at NeurIPS 2023, 2023. URL https://openreview.net/forum?id=T0PoOJg8cK. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners, 2019. URL https://www.semanticscholar.org/ paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/ 9405cc0d6169988371b2755e573cc28650d14dfe. O. Rubin, J. Herzig, and J. Berant. Learning to retrieve prompts for in-context learning. In M. Carpuat, M.-C. de Marneffe, and I. V. Meza Ruiz, editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2655–2671, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.191. URL https://aclanthology.org/2022. naacl-main.191. R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Ng, and C. Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In D. Yarowsky, T. Baldwin, A. Korhonen, K. Livescu, and S. Bethard, editors, Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA, Oct. 2013. Association for Computational Linguistics. URL https://aclanthology.org/D13-1170. T. Sorensen, J. Robinson, C. Rytting, A. Shaw, K. Rogers, A. Delorey, M. Khalil, N. Fulda, and D. Wingate. An information-theoretic approach to prompt engineering without ground truth labels. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Proceedings of the 60th Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), pages 819–862, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.60. URL https://aclanthology.org/2022.acl-long.60. N. Subramani, N. Suresh, and M. Peters. Extracting latent steering vectors from pretrained language models. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Findings of the Association for Computational Linguistics: ACL 2022, pages 566–581, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.48. URL https://aclanthology.org/2022.findings-acl.48. E. Todd, M. Li, A. S. Sharma, A. Mueller, B. C. Wallace, and D. Bau. LLMs represent contextual tasks as compact function vectors. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=AwyxtyMwaG. H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. A. M. Turner, L. Thiergart, D. Udell, G. Leech, U. Mini, and M. MacDiarmid. Activation addition: Steering language models without optimization, 2023. L. van der Maaten and G. Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(86):2579–2605, 2008. URL http://jmlr.org/papers/v9/vandermaaten08a.html. J. Von Oswald, E. Niklasson, E. Randazzo, J. Sacramento, A. Mordvintsev, A. Zhmoginov, and M. Vladymyrov. Transformers learn in-context by gradient descent. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 35151–35174. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/ von-oswald23a.html. E. M. Voorhees and D. M. Tice. Building a question answering test collection. In Proceedings of the 23rd Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’00, page 200–207, New York, NY, USA, 2000. Association for Computing Machinery. ISBN 1581132263. doi: 10.1145/345508.345577. URL https://doi.org/10. 1145/345508.345577. B. Wang and A. Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021. L. Wang, L. Li, D. Dai, D. Chen, H. Zhou, F. Meng, J. Zhou, and X. Sun. Label words are anchors: An information flow perspective for understanding in-context learning. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9840–9855, Singapore, Dec. 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.609. URL https://aclanthology.org/2023. emnlp-main.609. J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, E. H. Chi, T. Hashimoto, O. Vinyals, P. Liang, J. Dean, and W. Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=yzkSU5zdwD. Survey Certification. J. Wei, J. Wei, Y. Tay, D. Tran, A. Webson, Y. Lu, X. Chen, H. Liu, D. Huang, D. Zhou, and T. Ma. Larger language models do in-context learning differently, 2023. Z. Wu, Y. Wang, J. Ye, and L. Kong. Self-adaptive in-context learning: An information compression perspective for in-context example selection and ordering, 2023.
S. M. Xie, A. Raghunathan, P. Liang, and T. Ma. An explanation of in-context learning as implicit bayesian inference. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=RdJVFCHjUMI. H. Zhang, Y. Zhang, R. Zhang, and D. Yang. Robustness of demonstration-based learning under limited data scenario. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 1769–1782, Abu Dhabi, United Arab Emirates, Dec. 2022. Association for Computational Linguistics. doi: 10.18653/v1/ 2022.emnlp-main.116. URL https://aclanthology.org/2022.emnlp-main.116. X. Zhang, J. Zhao, and Y. LeCun. Character-level convolutional networks for text classification. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. URL https://proceedings.neurips.cc/paper_files/paper/2015/file/ 250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf. Z. Zhao, E. Wallace, S. Feng, D. Klein, and S. Singh. Calibrate before use: Improving few-shot performance of language models. In M. Meila and T. Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 12697–12706. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr. press/v139/zhao21c.html. F. Zhuang, Z. Qi, K. Duan, D. Xi, Y. Zhu, H. Zhu, H. Xiong, and Q. He. A comprehensive survey on transfer learning. Proceedings of the IEEE, 109(1):43–76, 2021. doi: 10.1109/JPROC.2020. 3004555. A. Zou, L. Phan, S. Chen, J. Campbell, P. Guo, R. Ren, A. Pan, X. Yin, M. Mazeika, A.-K. Dombrowski, S. Goel, N. Li, M. J. Byun, Z. Wang, A. Mallen, S. Basart, S. Koyejo, D. Song, M. Fredrikson, J. Z. Kolter, and D. Hendrycks. Representation engineering: A top-down approach to ai transparency, 2023.
S. M. Xie, A. Raghunathan, P. Liang, and T. Ma. An explanation of in-context learning as implicit bayesian inference. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=RdJVFCHjUMI. H. Zhang, Y. Zhang, R. Zhang, and D. Yang. Robustness of demonstration-based learning under limited data scenario. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 1769–1782, Abu Dhabi, United Arab Emirates, Dec. 2022. Association for Computational Linguistics. doi: 10.18653/v1/ 2022.emnlp-main.116. URL https://aclanthology.org/2022.emnlp-main.116. X. Zhang, J. Zhao, and Y. LeCun. Character-level convolutional networks for text classification. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. URL https://proceedings.neurips.cc/paper_files/paper/2015/file/ 250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf. Z. Zhao, E. Wallace, S. Feng, D. Klein, and S. Singh. Calibrate before use: Improving few-shot performance of language models. In M. Meila and T. Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 12697–12706. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr. press/v139/zhao21c.html. F. Zhuang, Z. Qi, K. Duan, D. Xi, Y. Zhu, H. Zhu, H. Xiong, and Q. He. A comprehensive survey on transfer learning. Proceedings of the IEEE, 109(1):43–76, 2021. doi: 10.1109/JPROC.2020. 3004555. A. Zou, L. Phan, S. Chen, J. Campbell, P. Guo, R. Ren, A. Pan, X. Yin, M. Mazeika, A.-K. Dombrowski, S. Goel, N. Li, M. J. Byun, Z. Wang, A. Mallen, S. Basart, S. Koyejo, D. Song, M. Fredrikson, J. Z. Kolter, and D. Hendrycks. Representation engineering: A top-down approach to ai transparency, 2023.
Table 9: Templates and label spaces for tasks used in our experiments. {Sentence} and {Label} are placeholders for the input sentence and label of a demonstration example, respectively. We show only the template for a single demonstration example to save space, and we connect multiple demonstration examples by a new line character: ‘\n’.
Dataset
Template
Label Space
SST-2
Review: {Sentence}
negative / positive
Sentiment: {Label}
SST-5
Sentence: {Sentence}
terrible / negative / neutral / positive / great
Sentiment: {Label}
MR
Review: {Sentence}
negative / positive
Sentiment: {Label}
Subj
Sentence: {Sentence}
objective / subjective
Label: {Label}
DBPedia
Input: {Sentence}
Label: {Label}
company / school / artist / athlete / politics /
transportation / building / nature / village /
animal / plant / album / film / book
AGNews
News: {Sentence}
World / Sports / Business / Technology
Type: {Label}
TREC
Question: {Sentence}
Answer Type: {Label}
Abbreviation / Entity / Person / Location /
Number
HateSpeech18
Text: {Sentence}
Label: {Label}
neutral / hate
EmoC
Dialogue: {Sentence}
Emotion: {Label}
others / happy / sad / angry
Extra Details For the task HateSpeech18, we preprocess the data to retain only the first two classes—{0: neutral} and {1: hate}. We exclude the other two classes due to their extremely limited number of samples.
# B Reproduction Details
The code is publicly available at: https://github.com/LzVv123456/I2CL
# B.1 Implementation of Comparable Methods.
Soft Prompt For the implementation of soft-prompt method [Lester et al., 2021], we allow one extra learnable token (P = 1) per layer, and apply soft prompt across all layers. We also attempt to use more learnable tokens, resulting in poorer performance due to overfitting. For optimization, we conduct a simple grid search on SST-2 to determine an optimal learning rate of 0.1, which we then apply across all tasks. All other configurations remain as specified in the experimental section. Label Anchor We utilize the official code from the Label-anchor paper Wang et al. [2023], aligning the architecture, datasets, and template setups for a fair comparison. Template tokens and the newline separator ‘\n’ serve as anchors, following their established practice. Detailed information can be found at https://github.com/lancopku/label-words-are-anchors. Task Vector We replicate the task vector method originally studied by Hendel et al. [2023], which was evaluated using a set of toy datasets, within our experimental framework. To generate the task vector, we append a random extra query after the concatenated demonstration examples (five per class) to simulate the dummy query they utilized. We then extract the hidden state from the token
<div style="text-align: center;">Table 10: Comparison between I2CL and other methods on GPT-J-6B.</div>
Task
Zero-shot
Few-shot (ICL)
Soft-prompt
Label-anchor
Task-vector
I2CL (ours)
SST-2 (%) ↑
77.76
89.44±2.60
69.04±11.61
87.12±4.57
59.84±4.47
85.48±1.18
SST-5 (%) ↑
25.60
39.65±4.57
37.88±2.99
37.24±3.53
31.20±2.82
37.32±3.11
TREC (%) ↑
68.20
67.76±2.11
67.00±12.04
58.52±2.44
67.32±0.32
63.84±7.58
AGNews (%) ↑
71.60
83.18±2.03
83.16±4.86
80.84±0.88
80.12±2.23
81.56±3.13
Subj (%) ↑
62.40
50.20±0.22
63.64±6.52
51.16±1.71
66.32±2.31
65.56±8.33
HateSpeech18 (%) ↑
59.92
53.44±6.84
67.76±5.04
55.20±8.19
70.12±5.04
62.32±5.76
DBPedia (%) ↑
65.56
93.30±1.19
85.04±1.02
90.84±1.79
77.64±4.63
81.84±4.50
EmoC (%) ↑
44.68
47.62±8.62
47.48±10.87
44.00±8.25
45.00±4.20
50.32±4.68
MR (%) ↑
76.88
88.66±1.23
73.76±9.40
87.92±3.82
81.36±3.58
84.40±2.45
Macro avg. acc. (%) ↑
61.40
68.14
66.08
65.87
64.32
68.07
position responsible for the next token prediction to serve as the task vector. A hold-out dataset with 32 extra examples is used to select the best layer for extraction and replacement for each task, following the original practice.
# B.2 Implementation of Transfer Learning
Algorithm 1 details the transfer learning method of I2CL. In implementation, we set h = 0.8 and τ = 0.5.
Algorithm 1 details the transfer learning method of I2CL. In implementation, we set h = 0.8 and
Algorithm 1 Transfer Learning of I2CL
1: Input: Coefficients c1, c2, . . . , cN, Context vectors v1, v2, . . . , vN, Demonstrations of new task
dnew, Threshold h, Temperature τ, Default coefficient initialization cinit.
2: Output:cnew, vnew
3: Initialize I ←∅
4: vnew ←Context_vectorization(dnew)
5: cnew ←Noisy_self_calibration(dnew, vnew, cinit)
6: for i = 1 to n do
7:
Compute si ←cosine(cnew, ci)
8:
if si > h then
9:
Add i to I
10:
end if
11: end for
12: Compute probabilities P(i) ←
exp(si)/τ
�
j∈I exp(sj/τ) for each i ∈I
13: Compute vavg ←�
i∈I P(i)vi
14: Compute cavg ←�
i∈I P(i)ci
15: cnew ←Noisy_self_calibration(dnew, vavg, cavg)
16: vnew ←vavg
17: return cnew, vnew
# C Additional Experiments
# C.1 Results of Other Architectures
Here, we show results of I2CL under architecture GPT-J and GPT2-XL in Table 10 and Table 11 respectively.
# C.2 Trend of I2CL Under Varying Shots
In our standard empirical setup, we select five examples per class to create the demonstration group. This approach prompts an important question about how I2CL behaves under different shot configurations. To investigate this, we varied the number of shots per class from 1 to 10 and compared the performance of I2CL with that of ICL. As illustrated in Figure 5, I2CL and ICL perform similarly when the number of demonstration examples is limited. However, while the performance of ICL
<div style="text-align: center;">Table 11: Comparison between I2CL and other comparable methods on GPT2-XL. AGnews and DBPedia are not evaluated due to the limitation of GPT2-XL’s context window size.</div>
Task
Zero-shot
Few-shot (ICL)
Soft-prompt
Label-anchor
Task-vector
I2CL (ours)
SST-2 (%) ↑
74.76
73.65±8.89
61.04±3.45
63.40±8.82
81.08±4.87
80.16±3.98
SST-5 (%) ↑
30.44
35.95±2.39
23.96±2.09
22.36±3.37
28.52±1.37
33.84±2.60
TREC (%) ↑
35.40
60.64±5.00
40.60±10.15
66.36±10.69
41.40±5.35
51.48±5.26
Subj (%) ↑
64.88
63.82±10.55
55.44±4.12
55.56±4.26
71.80±1.86
65.96±4.83
HateSpeech18 (%) ↑
70.84
51.86±3.22
63.92±7.06
54.88±4.53
62.48±2.83
68.32±4.76
EmoC (%) ↑
37.88
38.62±7.68
33.60±4.04
36.68±2.70
37.60±2.48
47.92±1.84
MR (%) ↑
71.36
75.79±9.25
57.60±3.53
60.20±3.32
78.40±2.36
83.20±3.29
Macro avg. acc. (%) ↑
55.08
57.19
48.02
51.35
57.33
61.55
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/85b4/85b4aa19-5069-45c6-b7a9-897e6af35702.png" style="width: 50%;"></div>
plateaued at approximately five shots per class, our model continues to show the capacity to benefit from an increased number of demonstration examples.
# D Additional Visualization
Additional Plots of Calibrated Linear Coefficients Here, we present the calibrated coefficients for all tasks we used, illustrating variations and patterns across different configurations.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ad79/ad79d56d-15be-4a8f-b8be-7788f9e7bde8.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 12: Calibrated coefficients for DBPedia.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/889d/889dfd20-cc12-4d23-9e0c-a45f14d259cc.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 14: Calibrated coefficients for HateSpeech18.</div>
