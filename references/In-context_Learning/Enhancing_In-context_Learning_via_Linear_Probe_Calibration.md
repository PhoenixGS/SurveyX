# -context Learning via Linear Probe Calib
# Enhancing In-context Learning via Linear Probe Calibration
Momin Abbas⋆ Yi Zhou† Parikshit Ram† Nathalie Baracaldo† Horst Samulowitz† Theodoros Salonidis† Tianyi Chen⋆ ⋆Rensselaer Polytechnic Institute †IBM Research
# Abstract
 22 Jan 2024
In-context learning (ICL) is a new paradigm for natural language processing that utilizes Generative Pre-trained Transformer (GPT)like models. This approach uses prompts that include in-context demonstrations to generate the corresponding output for a new query input. However, applying ICL in real cases does not scale with the number of samples, and lacks robustness to different prompt templates and demonstration permutations. In this paper, we first show that GPT-like models using ICL result in unreliable predictions based on a new metric based on Shannon entropy. Then, to solve this problem, we propose a new technique called the Linear Probe Calibration (LinC), a method that calibrates the model’s output probabilities, resulting in reliable predictions and improved performance, while requiring only minimal additional samples (as few as five labeled data samples). LinC significantly enhances the ICL test performance of GPT models on various benchmark datasets, with an average improvement of up to 21%, and up to a 50% improvement in some cases, and significantly boosts the performance of PEFT methods, especially in the low resource regime. Moreover, LinC achieves lower expected calibration error, and is highly robust to varying label proportions, prompt templates, and demonstration permutations. Our code is available at https://github.com/mominabbass/LinC.
[cs.CL]
# 1 Introduction
Large language models (LLMs), have remarkably showcased their capabilities across a broad range of natural
Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS) 2024, Valencia, Spain. PMLR: Volume TBD. Copyright 2024 by the author(s).
language processing tasks [11, 13, 3, 29, 59, 39]. The cost of training these large models can be prohibitively expensive. Therefore, the commonly adopted approach is to first pre-train with large amounts of unlabled data and then fine-tune the model to downstream tasks. Although fine-tuning LLMs can be effective, it is prone to instability [34] due to numerous hyperparameter configurations resulting in failed runs, unstable results, and over-fitting [41, 11, 25]. Moreover, fine-tuning models of such large size may also be expensive and also requires explicit access to the architecture and weights of LLMs, which may not be publicly available [60].
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/00a1/00a12ae9-e720-484c-bd93-2fafcc3d89be.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Example of ICL with a LLM θ∗.</div>
To avoid large fine-tuning times and avoid requiring access to the weights of the model, recently, LLMs, exemplified by GPT-3 [3], have demonstrated the ability to perform in-context learning (ICL), a capability whereby a model can generate an appropriate output for a given query input based on a prompt that includes input-output example pairs specific to the task at hand. ICL can work with an API without explicit access to the LLM. Figure 1 provides a visual representation of ICL. The prompts utilized in ICL entail task-specific in conjunction with a series of input-label pairs, referred to as demonstrations. Such a capability of LLM to learn “in-context” presents an intriguing aspect whereby the model is capable of acquiring knowledge and performs well on a wide range of downstream tasks without any task-specific parameter fine-tuning [3, 48, 2, 40]. More specifically, the aim of ICL is to make a prediction on some query test sample x by conditioning on a prompt sequence (fx(x1), fy(y1), . . . , fx(xk), fy(yk), fx(x)) containing k-shot samples (xi, yi)k i=1 (i.e. demonstrations) and the query test sample, where the functions fx(.), fy(.) denote template functions that attach pre-defined
To avoid large fine-tuning times and avoid requiring access to the weights of the model, recently, LLMs, exemplified by GPT-3 [3], have demonstrated the ability to perform in-context learning (ICL), a capability whereby a model can generate an appropriate output for a given query input based on a prompt that includes input-output example pairs specific to the task at hand. ICL can work with an API without explicit access to the LLM. Figure 1 provides a visual representation of ICL. The prompts utilized in ICL entail task-specific in conjunction with a series of input-label pairs, referred to as demonstrations. Such a capability of LLM to learn “in-context” presents an intriguing aspect whereby the model is capable of acquiring knowledge and performs well on a wide range of downstream tasks without any task-specific parameter fine-tuning [3, 48, 2, 40].
More specifically, the aim of ICL is to make a prediction on some query test sample x by conditioning on a prompt sequence (fx(x1), fy(y1), . . . , fx(xk), fy(yk), fx(x)) containing k-shot samples (xi, yi)k i=1 (i.e. demonstrations) and the query test sample, where the functions fx(.), fy(.) denote template functions that attach pre-defined
descriptions to the input and output, respectively (c.f. text highlighted in yellow in Figure 1). The output template function fy(.), in addition, may transform labels yi into a natural language format instead into numeric/one-hot labels (e.g. for binary classification transforming labels (0, 1) to (Positive, Negative) (c.f. labels in Figure 1)). Mathematically, for an input x, a prompt P is defined as:
P(x, (xi, yi)k i=1) ≜d1 ⊕d2 ⊕· · · ⊕dk ⊕fx(x) (1
(1)
where each demonstration di is given by fx(xi)⊕fy(yi) and ⊕denotes the concatenation operation.
Recent studies suggest that ICL exhibits high variability in performance across different prompt templates, demonstrations, and their arrangement within the prompt arising from biases that favor outputting certain answers [63, 32], resulting in performance variation from random guess to state-of-the-art. Further, when the prompt P includes several more demonstrations, ICL’s performance is thwarted by the inherent maximum sequence length limitation of the underlying language model. Moreover, our initial analysis reveals that while GPT-like models’ ICL ability delivers acceptable results, their predictions cannot be considered reliable when assessed using Shannon entropy. We will discuss these limitations and challenges of ICL in detail in Section 2. In this paper, we argue that by training very few parameters and subsequently utilizing an affine transformation on the output probabilities to calibrate the model, the performance, as well as the reliability of these predictions, can be significantly enhanced. We propose linear probe calibration (LinC), which optimizes the calibration parameters (i.e. a low-dimensional matrix and vector) using only a few extra samples and minimal computation. Experimental results reveal that LinC significantly outperforms the baselines. Moreover, LinC produces reliable predictions that exhibit consistency across a range of prompt templates and various permutations of demonstrations.
Recent studies suggest that ICL exhibits high variability in performance across different prompt templates, demonstrations, and their arrangement within the prompt arising from biases that favor outputting certain answers [63, 32], resulting in performance variation from random guess to state-of-the-art. Further, when the prompt P includes several more demonstrations, ICL’s performance is thwarted by the inherent maximum sequence length limitation of the underlying language model. Moreover, our initial analysis reveals that while GPT-like models’ ICL ability delivers acceptable results, their predictions cannot be considered reliable when assessed using Shannon entropy. We will discuss these limitations and challenges of ICL in detail in Section 2.
# We summarize our contributions below:
We summarize our contributions below:
C1) We present a novel insight that, while GPT-like models often exhibit acceptable ICL performance, their predictions appear to have very low confidence when measured using the Shannon entropy, highlighting a potential cause for their highly variable performance. C2) We propose linear probe calibration (LinC), a simple and black-box method that enhances model’s reliability and performance by linearly calibrating output probabilities without requiring any access to model weights or architecture.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bb6b/bb6b2ed6-4005-44f8-b938-a744874554fe.png" style="width: 50%;"></div>
Figure 2: The efficacy of ICL is restricted by GPT tokenizer’s maximum sequence length limit. Black-dashed lines demarcate the point beyond which additional shots cannot be utilized.
C3) We empirically show that LinC consistently outperforms baselines on various benchmark datasets, with a performance boost of up to 21%, on average, and up to a 50% improvement in some cases, compared to the vanilla ICL baseline.
# 2 ICL: Challenges and Opportunities
In this section, we will first highlight two key limitations of ICL with current LLMs, which then serve as the motivation of our subsequent algorithm design.
# 2.1 Maximum Sequence length Limitation
We demonstrate the maximum sequence length limitation of ICL [58] on different datasets in Figure 2 using GPT-2 tokenizer. We observe that the test performance of ICL improves consistently, which aligns with the power-law relations between the generalization error and data size [24, 42]. However, beyond a certain point, no additional demonstrations can be added within the prompt since the model reaches its maximum sequence length limit1. Alas, the potential for performance enhancement through the acquisition of more examples is often constrained by this limitation of ICL, even in few-shot settings.
# Motivated by this, we ask:
Given a few additional samples, beyond the LLM sequence length limit, can we further improve ICL test performance without utilizing LLM fine-tuning?
1It is noteworthy that although some recent models do have expanded context windows (e.g. GPT-4 with 32k tokens), the majority of them still maintain smaller context windows. Smaller models are often preferred in various applications due to their flexibility in adapting to specific tasks and their ability to achieve comparable performance to larger models while using much fewer compute.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/540e/540ebb81-0e95-4648-b4c3-52d6e4276511.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Shannon entropy histograms of using vanilla ICL on GPT-2-XL (1.5B) vs our method on SST-2 (higher entropy implies higher uncertainty); we use logarithmic base two. Refer to Section 2 for a detailed explanation.</div>
# 2.2 Entropy of Few-Shot Learning with GPT
In practical decision-making systems, it is crucial for prediction networks to not only exhibit high accuracy but also have the ability to identify the likelihood of incorrect predictions. For example, automated healthcare systems should be designed such that when the confidence level of a disease diagnosis network is low, the control is transferred to human doctors for making the diagnoses [21]. Well-calibrated confidence estimates play a crucial role in making machine learning models more interpretable. Since humans possess an inherent cognitive understanding of probabilities [8], having reliable confidence estimates can enhance the user’s confidence in the model’s predictions. This is especially relevant for neural networks, where the rationale behind their classification decisions can be complex and challenging to decipher. One way to assess the reliability of a model’s predictions is to measure the Shannon entropy. Shannon entropy [44] quantifies the amount of expected uncertainty in a probability distribution. Mathematically, Shannon entropy is given by:
(2)
where C is the number of classes and pc represents the output probability of class c. Entropy value is used to gauge prediction confidence, with a low value indicating high confidence and vice versa. We compare the performance of ICL on GPT before (vanilla ICL) and after our calibration (Section 3) using different numbers of shots on various datasets. Figure 3 encapsulates the results on SST-2 (for other datasets, see Appendix B). We observe that, in contrast to the performance of our model, using vanilla ICL on GPT leads to high values of entropy, implying that most test predictions were made with very low confidence, i.e., close to random guessing. This observation indicates that although vanilla ICL (i.e., uncalibrated) on GPT yields satisfactory results in terms of test accuracy (refer to Table 4), the confidence associated with
where C is the number of classes and pc represents the output probability of class c. Entropy value is used to gauge prediction confidence, with a low value indicating high confidence and vice versa.
these predictions is not entirely reliable. These findings align with previous studies, including [18], that have demonstrated the inadequacy of the conventional GPT decision boundary in effectively distinguishing between predictions by using the output with the highest probability as the predicted label. While low entropy (or high confidence) does not always imply a high accuracy, high entropy typically implies a high degree of uncertainty in predictions. This uncertainty may contribute to the increased variability of test performance, such as variations due to varying label proportions, prompt templates, and demonstration permutations (see Section 5.2). Motivated by this, we ask:
Can we make the predictions made by GPT-like models more reliable and accurate?
As we will see in the upcoming sections, our proposal to linearly calibrate model’s output probabilities via meticulously learned parameters not only improves its test performance but also enhances its reliability.
# 3 Linear Probe Calibration
In order to be considered reliable, a model must furnish a calibrated confidence measure along with its predictions, where the probability assigned to the predicted class corresponds to its actual likelihood [16].
Since vanilla ICL predictions may be unreliable due to the presence of high entropy prediction probabilities, we aim to investigate whether probability calibration can lead to improvements. In this section, we present linear probe calibration (LinC) that calibrates the model’s output probabilities, making it more reliable.
# One method for modifying output probabilities applies an affine transformation [38, 16]:
One method for modifying output probabilities applies an affine transformation [38, 16]:
(3)
# ˜p = softmax(Ap + b)
where A and b are parameters to be applied to the original probabilities p to get new probabilities ˜p. For example, for classification tasks p is the set of proba-
(4)
for a model Mθ∗parameterized by θ∗.
Given a few additional samples, we then optimize A and b using a validation set, starting with a zero initialization; we find that our method exhibits remarkable insensitivity to initialization and works quite well for zero/random initialization (see Figure 10). More specifically, given a validation set (xv i , yv i )Nv i=1 of size Nv, we create Nv validation prompts via:
(5)
Our objective is to solve the following problem
(6)
where L represents the loss function of the calibration parameters A and b. We use a gradient-based optimizer to optimize A and b using prompts P v i . Algorithm 1 encapsulates our proposed LinC method. While we employ the stochastic gradient descent algorithm, our method can be utilized with any available optimization algorithm. LinC incurs negligible computational overhead and can be implemented in just a few lines of code to compute and store A and b. Moreover, LinC utilizes only k +Nv extra samples to learn A and b. Finally, test predictions are obtained by calculating Ap + b and taking the argument of the maxima after the softmax operator.
Comparison with other methods. Unlike calibration methods that rely on the raw data (xv i , yv i )Nv i=1, our approach draws inspiration from ICL and employs prompts {P v i }Nv i=1 to learn the calibration parameters. Moreover, ICL performance is limited by the maximum input sequence length constraint of the underlying language model, causing poor scalability with an increase in the number of available training samples as shown in Figure 2. In contrast, our approach is scalable with respect to the number of available data samples, since we use the available samples to optimize the calibration parameters. In fact, our method requires only a handful of additional samples (typically in the range 10-100) to effectively optimize the calibration parameters, maintaining the few-shot regime and making it much more sample efficient than fine-tuning LLMs which require orders of magnitude larger number of samples. LinC is also much more computationally efficient as compared to other methods used to enhance LLM performance, such as fine-tuning, since it optimizes extremely low 2For detailed information about where the transforma-
2For detailed information about where the transformation is applied, please refer to Appendix B.
Algorithm 1 Linear Probe Calibration (LinC)
1: Input: LLM θ∗, validation set (xv
i , yv
i )Nv
i=1, k-shots
(xj, yj)k
j=1, loss function L
2: Initialize: parameters A0
0, b0
0 via zero initialization,
step-size α, number of epochs T
3: For each P v
i in (5), obtain logits (4): pv
i = p(P v
i )
4: for t = 1, · · · , T do
5:
for i = 1, · · · , Nv do
6:
Compute ˆpv
i = Ai−1
t−1pv
i + bi−1
t−1
7:
Ai
t−1 = Ai−1
t−1 −α∇AL(ˆpv
i , yv
i )
8:
bi
t−1 = bi−1
t−1 −α∇bL(ˆpv
i , yv
i )
9:
end for
10:
A0
t = ANv
t−1 and b0
t = bNv
t−1
11: end for
12: return A0
T , b0
T
dimensional calibration parameters. Lastly, unlike incontext fine-tuning and meta-training methods [33, 55], LinC operates using API access alone and doesn’t need access to the LLM architecture and weights.
Shot
Method
SST-2
TREC
Subj
4-shot
NoC
66.313.1
24.06.3
52.84.9
ConC
78.98.4
41.34.7
67.88.9
NoC*
68.98.6
35.90.7
69.310.5
ConC*
75.93.9
43.52.5
61.910.2
LinC
86.22.9
45.93.3
72.911.1
8-shot
NoC
57.28.0
31.88.1
56.610.7
ConC
73.710.5
45.41.7
68.19.0
NoC*
62.77.7
40.57.1
70.48.8
ConC*
75.88.1
47.22.3
63.07.7
LinC
79.110.0
48.05.4
76.83.5
Table 1: Comparison under same number of samples.
Table 1: Comparison under same number of samples.
# 4 LinC: A Viable Solution
Before presenting our results, we first highlight the significance of linear calibration and why it is important when we are handicapped by limited resources. Therefore we ask: assuming that the maximum sequence length limit is not exceeded, what is the best way to use the additional samples in ICL? Is calibration the optimal way to use these additional samples? To answer this question, we introduce two additional baselines, NoC* and ConC*3, which utilize the same 10 validation samples used for training the calibration parameters in our method LinC. For these baselines, these 10 samples are treated as additional in-context test demonstrations within the test prompts. Consequently, NoC* and ConC* should be considered 3for details about NoC and ConC, see Section 5.1
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/002b/002bf30c-28a1-476d-9960-c5360f6e61e1.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
Figure 4: LinC outperforms ICL on all few-shot experiments, and substantially enhances PEFT, especially in the low resource regime, while maintaining almost identical data and compute requirements.
to be in the 14-shot and 18-shot regimes instead of the 4-shot and 8-shot regimes, respectively. The choice of using 10 samples is to ensure that the maximum length limit does not exceed for any of the datasets used in this experiment. Table 1 shows the results on three different datasets in the 4-shot and 8-shot settings on GPT-2-XL. It is noteworthy that for all three datasets, using the same samples to learn the calibration parameters is better than using them within the test prompts as additional test demonstrations. Moreover, we encapsulate our finding in Figure 4 by assessing the trade-off between performance and resource consumption, quantified as the product of trainable parameters, samples, and epochs. The raw ICL baselines (0/8-shot) are denoted by horizontal lines. Notably, LinC outperforms ICL baselines by a substantial margin while maintaining resource efficiency. In addition, it enhances the performance of Parameter-Efficient Fine-tuning (PEFT) methods, such as Soft Prompt Tuning (SPT) [27] and Low-Rank Adaptation (LoRA) [19], particularly in scenarios with limited data and computational resources. These observations render LinC particularly effective in situations where compute resources are limited and training data is scarce.
# 5 Experiments
This section will demonstrate the effectiveness of LinC on several benchmarks in the few-shot and PEFT settings. We run our experiments on GPT-2-XL [39] with 1.5B parameters, GPT-J [53] with 6B parameters, and Llama-2 [48] with 13B4 parameters on 2
4Note that this is the largest model that can fit into our available GPU memory.
Dataset
GPT-J (6B)
ICL
ConC
LinC
SST-2
0.0592
0.1974
0.0458
SST-5
0.2254
0.0933
0.0970
AGNews
0.2858
0.0686
0.0577
TREC
0.2934
0.1108
0.1613
DBPedia
0.2713
0.2281
0.0502
RTE
0.0471
0.0825
0.0421
Table 2: Expected Calibration Error (ECE) comparison between baselines and LinC (a model with perfect calibration would exhibit an ECE of 0).
Dataset (D/C)
GPT-J (6B)
ICL
ConC
LinC
Hamster (5/2)
55.6±8.3
53.3±9.4
60.0±14.4
Customers (8/2)
67.4±0.5
54.9±2.3
68.6±0.5
Breast (7/2)
66.7±3.3
55.8±9.2
71.3±3.3
Spambase (57/2)
40.0±0.0
51.2±9.5
61.1±1.7
TAE (5/3)
45.2±4.6
48.4±9.5
50.5±11.0
Vehicle (18/4)
26.9±0.7
29.0±2.8
29.0±1.5
LED (7/10)
16.3±10.4
27.3±6.9
28.0±7.1
<div style="text-align: center;">GPT-J (6B)</div>
Table 3: Performance comparison between baselines and LinC on OpenML datasets; D represents the number of features and C represents the number of classes.
NVIDIA GeForce RTX 3090 GPUs. We thoroughly study different label proportions, prompt templates, and demonstration permutations.
# 5.1 Experimental Setup
We evaluate the effectiveness of our LinC method using seven widely used text-classification datasets: sentiment analysis using SST-2 [46] and SST-5 [46], topic classification using the 4-way AGNews [61] and 14way DBPedia [61], 6-way question classification using TREC [51], textual entailment using binary RTE [9] from SuperGLUE [52], and subjectivity classification using Subj [36]. We also tested LinC on seven diverse non-text classification tasks using OpenML [49] datasets with varying number of classes and features. A fixed prompt format was utilized for each dataset, unless stated otherwise, which is demonstrated alongside examples in Appendix A, Table 6 and Table 7. Experiments were conducted under 0-shot, 1-shot, 4shot and 8-shot learning settings. Five different sets of test demonstrations were chosen at random, and arranged in an arbitrary order in the prompt, and the mean and standard deviation were computed across all
Experiments were conducted under 0-shot, 1-shot, 4shot and 8-shot learning settings. Five different sets of test demonstrations were chosen at random, and arranged in an arbitrary order in the prompt, and the mean and standard deviation were computed across all
Model
Shots Method SST-2
SST-5
AGNews TREC
DBpedia RTE
Subj
Avg
GPT-2-XL 1.5B
0-shot
NoC
64.50.0
33.70.0
44.30.0
28.70.0
58.70.0
48.00.0
56.70.0
47.8
ConC
70.90.0
20.30.0
65.30.0
41.70.0
50.00.0
50.50.0
73.00.0
53.1
LinC
71.60.0
41.30.0 65.70.0
42.00.0 73.00.0
54.50.0 73.30.0
60.2
1-shot
NoC
59.713.2
28.39.7
39.610.3
27.15.9
40.515.9
53.41.0
54.68.8
43.3
ConC
76.71.8
31.14.8
63.93.3
40.53.1
62.37.5
52.51.7
61.37.4
55.5
LinC
83.216.9 40.23.7 64.86.3
42.95.1 63.17.4
54.41.9 73.75.5
60.3
4-shot
NoC
66.313.1
34.14.8
40.414.1
24.06.3
66.710.2
52.23.2
52.84.9
48.1
ConC
78.98.4
34.46.8
60.46.8
41.34.7
72.14.8
53.00.9
67.88.9
58.3
LinC
87.11.9
39.15.6 69.06.5
46.03.2 73.14.8
53.60.6 73.610.8 63.1
8-shot
NoC
57.28.0
31.89.3
41.45.8
31.88.1
59.016.4
52.50.9
56.610.7
47.2
ConC
73.710.5
28.25.4
57.912.2
45.41.7
71.85.7
53.41.1 68.19.0
56.9
LinC
79.110.0 38.19.6 63.37.2
48.16.1 71.95.5
53.20.9
76.93.9
61.5
GPT-J 6B
0-shot
NoC
66.30.0
33.70.0
36.00.0
24.70.0
19.70.0
55.60.0
65.70.0
43.1
ConC
58.00.0
40.70.0
56.00.0
40.00.0
48.00.0
52.40.0
58.70.0
50.5
LinC
74.30.0
46.00.0 64.30.0
70.70.0 69.30.0
56.70.0 70.70.0
64.6
1-shot
NoC
67.36.7
35.33.5
65.314.3
40.38.8
64.916.6
50.73.7
65.19.9
55.6
ConC
88.31.7
46.93.1
74.95.8
62.64.7
80.23.3
53.41.2
59.12.2
66.5
LinC
88.51.7
50.41.3 81.74.6
63.04.4 82.63.7
56.01.8 69.97.5
70.3
4-shot
NoC
88.93.3
46.33.2
72.36.1
37.74.2
82.114.4
55.07.1
57.46.9
62.8
ConC
92.83.1
49.44.8
75.24.1
46.05.1
88.94.9
56.01.9
65.311.8
67.7
LinC
94.90.9
51.13.7 77.95.5
65.31.5 89.45.4
58.43.7 68.411.2 72.2
8-shot
NoC
91.86.0
44.54.3
76.89.9
43.56.3
88.63.2
60.32.8
82.25.7
69.7
ConC
93.81.8
44.44.4
78.55.7
52.38.3
90.82.5
58.84.7
81.36.2
71.4
LinC
94.81.2
51.30.8 83.71.8
67.33.8 90.13.9
63.24.2 84.74.9
76.4
Llama-2 13B
0-shot
NoC
56.30.0
34.00.0
73.30.0
48.70.0
54.70.0
66.10.0
47.30.0
54.3
ConC
69.30.0
33.30.0
72.30.0
71.30.0
75.00.0
67.50.0
47.00.0
62.2
LinC
75.30.0
47.30.0 85.70.0
75.00.0 90.70.0
69.00.0 48.30.0
70.2
1-shot
NoC
73.912.8
43.93.5
81.52.7
67.07.2
92.31.7
70.33.5
50.53.9
68.5
ConC
94.11.7
42.43.9
80.61.9
76.22.5
92.31.2
60.97.7
53.212.2
71.4
LinC
94.11.7
51.01.6 84.11.7
77.32.6 93.61.6
75.92.6 55.610.2 75.9
4-shot
NoC
92.92.6
48.74.1
82.73.7
62.514.7 94.21.0
68.88.7
72.011.8
74.5
ConC
97.40.4
44.95.2
80.52.3
75.37.2
94.81.2
75.12.0
72.59.8
77.2
LinC
97.50.5
53.00.9 86.01.2
75.72.1 95.30.9
77.00.9 78.79.6
80.5
8-shot
NoC
92.23.1
49.17.3
87.00.5
77.34.7
94.91.7
72.95.6
82.77.4
79.4
ConC
96.70.5
47.14.4
83.81.6
79.13.7
94.61.9
75.33.0
82.55.5
79.9
LinC
97.00.2
51.46.1 87.10.6
79.73.9 95.01.4
76.40.6 82.13.1
81.2
Table 4: Comparisons among the conventional approach (NoC; [3]), ConC [63] and LinC (Ours) on GPT-2, GPT-J and Llama-2. We report the mean and the standard deviation of test accuracy across different choices of the test demonstrations (the prompt is fixed). We also report the average performance across seven datasets.
five test prompts. We used the cross-entropy loss as
(7)
where yv i,c is the binary variable indicating if class c is the correct label for validation input xv i , and ˜pc denotes the output from the softmax defined in (3) with p = p(P v i ) = p(P(xv i , (xj, yj)k j=1)) defined in (4). In (6), we keep the loss general as we can use different losses depending on the task.
where yv i,c is the binary variable indicating if class c is the correct label for validation input xv i , and ˜pc denotes the output from the softmax defined in (3) with p = p(P v i ) = p(P(xv i , (xj, yj)k j=1)) defined in (4). In (6), we keep the loss general as we can use different losses depending on the task. For each experiment, we fine-tuned the step size α. The number of epochs T was chosen from {1, 5, 15, 50, 100} and the number of validation prompts Nv was chosen
For each experiment, we fine-tuned the step size α. The number of epochs T was chosen from {1, 5, 15, 50, 100} and the number of validation prompts Nv was chosen
from {1, 5, 10, 30, 100, 300} (for details, see Appendix B). If the validation set is provided for a dataset, we utilize it as is. However, if the validation set is not available, we create one by randomly selecting a subset from the training set. As baselines, we used the vanilla ICL method [3] that does not use any calibration (NoC) and contextual calibration (ConC) [63]. The results of ConC were replicated using the released code5. Both ConC and our approach, LinC, utilize an affine transformation. However, the crucial distinction lies in the fact that LinC acquires the transformation parameters through the learning process with a few additional samples (following the same format
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4680/46802aa8-3975-49c4-8ec5-af6670908c7e.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Comparison across six different templates.</div>
as instruction-tuning). In contrast, ConC does not engage in learning and opts for a pre-defined initialization instead (for details, see Appendix C). The demonstrations’ labels were not artificially balanced. Moreover, we observed that our method performs well with any set of k-shot demonstrations used in the validation prompts, but utilizing a different set can lead to improved performance, thus we conducted experiments with different random seeds to obtain a better set of the validation demonstrations.
# 5.2 Simulation Results
LinC enhances both the average and minimum accuracy. Table 4 shows the results on GPT-2-XL, GPT-J and Llama-2, respectively. LinC consistently outperforms baselines in almost all experiments, demonstrating its strong generalization ability across different model sizes and few-shot settings. We observe that, on average, LinC achieves up to 21% improvement as compared to the vanilla ICL baseline and 14% improvement as compared to contextual calibration for 0-shot learning on GPT-J. Moreover, in certain cases, LinC can deliver a significant boost in performance, up to 50% absolute improvement, as observed in the GPTJ 0-shot experiment on the DBPedia dataset. LinC demonstrates significant performance gain on some datasets, such as TREC, while moderate improvement is observed for other datasets, such as SST-2. Additionally, the performance improvement of LinC is more prominent on GPT-J than on GPT-2-XL and Llama-2. We also compare our results with prototypical calibration [18], despite the fact that they use orders of magnitude larger number of samples and thus fall outside the few-shot learning regime. Table 8 in Appendix B shows that LinC outperforms prototypical calibration in 16 out of 28 cases while using fewer number of samples (see Table 9). When using the same number of samples, LinC outperforms prototypical calibration in 20 of 28 cases. LinC’s exceptional ability to generalize effectively could be supported by a
recent discovery indicating that ICL in a conventional Transformer block is equivalent to adjusting the output layer using linear modeling of meta-learned deep data representations in few-shot settings, and LinC can be seen as explicitly adjusting this output layer with the additional samples [50]. We also performed an experiment to show the impact of the model sizes (e.g., 7B vs 13B) on performance within the same model family (e.g., Llama-2); see Table 11. The results demonstrate that LinC consistently improves performance regardless of the model size. LinC reduces variance across demonstrations. Figure 15 shows the difference in standard deviation between the calibration methods and the NoC baseline for all GPT-J experiments in Table 4. In most cases, LinC significantly reduces variance while only slightly increasing it in the remaining cases. Moreover, on average, LinC achieves a greater reduction in standard deviation than ConC, indicating that the predictions made by LinC are more consistent and reliable. LinC improves the Expected Calibration Error (ECE). To further evaluate LinC’s model calibration, we use the ECE metric [35] that is widely-used to quantify the alignment between predicted and actual probabilities. Table 2 shows the results on GPT-J, 0-shot setting (see Appendix B for other settings). In most instances, LinC demonstrates the lowest ECE, and in others, it ranks as the second-best method, with only a minimal difference from the best method. LinC improves accuracy and reduces variance across varying prompt templates. Next we keep the set of demonstrations fixed and vary the prompt format. We use six different prompt formats and label spaces for SST-2 dataset (for details, refer to Appendix A, Table 5) on GPT-2-XL under 4-shot setting. From Figure 5, we observe that while ConC generally enhances the average accuracy, it results in high variance. In contrast, LinC exhibits a considerable enhancement in accuracy with much lower variance, demonstrating its effectiveness in improving the model’s performance across various prompt templates. LinC is robust to class imbalance and permutations of demonstrations. Prior works [63, 32] have shown that the order in which the demonstrations are set in the prompt can significantly affect the performance. We evaluated the performance of our method on five 8-shot prompts for SST-2 on GPT-2XL, each with varying class proportions, and measured the accuracy across eight random permutations for each proportion. The demonstrations in the test prompt are kept the same for each proportion for both baselines and our method. Figure 6 illustrates that vanilla ICL (NoC) can achieve satisfactory performance with
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/abed/abeda8a3-8c29-4ba8-a155-8f64ee0381d3.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
Figure 6: Comparison under varying label proportions and permutations of demonstrations; each box represents the test accuracy obtained from eight random permutations.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4262/42623fd9-9b74-4422-932b-b8a1bb1ee70b.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
Figure 7: Performance across varying validation sizes; ‡ denotes NoC, and ∗denotes ConC. Black dotted line marks the maximum accuracy.
certain proportions and permutations, but performs much worse in most cases. This result is in line with the observations made in previous works [32]. Moreover, ConC achieves superior test accuracy on average when compared to NoC but the performance is volatile across different permutations. LinC stands out as the most effective model and displays low variance across different permutations, suggesting that it is robust to varying class proportions and permutations. LinC needs very few additional samples for improving ICL. Figure 7 investigates the effect of additionally available validation samples on the performance of three different datasets on GPT-2-XL under the 4-shot setting. The number of extra samples employed in the validation prompts (i.e. Nv+k) is plotted along the x-axis. We observe that performance can be greatly improved by increasing the number of validation samples within a certain small range. However, further increasing the number of samples does not yield any additional improvement. LinC demonstrates high sample efficiency, achieving maximum accuracy with less than 30 additional samples on most datasets. Re-
markably, some datasets require only five additional samples (e.g., Subj) to achieve maximum accuracy.
LinC also improves performance on nonlanguage tasks. We also evaluated the performance of our method on non-language classification tasks using seven real tabular datasets in OpenML [49] following the settings in [12]. Table 3 demonstrates the performance improvement of LinC over raw ICL and ConC across all tabular datasets, regardless of the number of classes (C) and data features (D). In contrast, ConC sometimes falls short in improving performance, especially in datasets with fewer classes and features, and has worse performance than raw ICL in datasets such as Hamster, Customers, and Breast.
LinC improves PEFT methods in the low-data low-compute regime. We evaluated the performance of LinC on two popular PEFT methods: 1) SPT [27] and 2) LoRA [19] (details in Appendix B.1). Tables 12 and 13 demonstrate that LinC consistently boosts both methods during fine-tuning across different sample sizes. However, performance improvement is most prominent when data and compute resources are limited. For example, for a sample size of 120 (out of 120k available training samples), LinC yields an accuracy improvement of +38.7% for SPT and +29.3% for LoRA. This makes LinC especially viable in scenarios with constrained compute and a scarcity of data.
# 6 Related Work
Understanding ICL. Some recent works focus on explaining the working of ICL. For example, [57] suggested that ICL is an implicit Bayesian inference and proved it through a synthetic dataset with a mixture of hidden Markov models in pretraining. [15] showed that Transformers can learn effective learning algorithms for unseen linear functions based on demonstration samples and achieve comparable error to least squares estimator in ICL models. [10] explain that ICL can be considered as implicit finetuning, where LLM generates meta-gradients from in-context demonstrations to adjust the model’s behavior. [30] framed ICL as an algorithm learning problem and demonstrated that Transformers can effectively implement a function class through implicit empirical risk minimization based on demonstrations. [50] showed self-attention-only Transformers trained on simple regression tasks exhibit significant similarity to models learned by gradient descent, revealing how trained Transformers execute gradient descent during their forward pass. [5] showed that ICL’s performance is influenced by the distributional properties of training data, with improved performance observed when training data consists of clustered examples and sufficient rare classes. ICL is also inher-
ently connected to multi-task learning [4, 62] and metalearning [14, 1, 6, 23, 20]. But a key difference between ICL and those methods is that, in ICL, adaptation to a new task is done implicitly through input prompt not running explicit gradient-based optimization.
Enhancing ICL. To enhance the performance of ICL, meta-learning has been introduced by [7, 33] to improve the adaptation of LLMs to ICL. [56] suggest augmenting the demonstrations by incorporating human-aided reasoning steps, leading to an improvement in performance on a range of arithmetic and reasoning tasks. [55, 43] suggest instruction tuning as a method to further pre-train the language model with a variety of downstream tasks in a shared prompting format. Several works focus on finding good in-context demonstrations to improve ICL performance [31, 28]. [54] investigates ICL using a Bayesian approach, and proposes an algorithm for selecting optimal demonstrations, showing empirical improvement compared to a random selection baseline. [45] studies model reliability using four different facets. [18] proposes estimating prototypical clusters for all classes, mapping each cluster to the corresponding label, and calibrating the test predictions by their most likely cluster. It has been found in [22] that the sensitivity of the language model stems from the label shift of the model in the data distribution, where the model exhibits a shift in the label marginal while maintaining a strong label conditional. Their solution involves a generative calibration approach, adjusting the label marginal through MonteCarlo sampling over the in-context model to calibrate the predictive distribution. [64] introduce Batch Calibration (BC), a zero-shot calibration method aimed at reducing bias from the batch. Closely related to our work is [63] which observed that the few-shot performance of language models is not consistent across different in-context settings. Additionally, [63] notes that language models tend to predict certain labels due to bias or demonstration permutations. However, the proposed calibration method of selecting content-free test inputs in [63] cannot accurately reflect the bias of models, which can result in sub-optimal performance. In contrast, our LinC approach achieves calibration of bias by accurately optimizing calibration parameters at the expense of minimal compute and data.
# 7 Conclusions
This paper investigates in-context learning (ICL), which relies on GPT-like models to generate outputs based on in-context demonstrations. Our findings reveal that ICL predictions may be unreliable when evaluated using Shannon entropy. To overcome these limitations, we propose the linear probe calibration (LinC) method, which significantly boosts the test performance
of GPT models on various benchmark datasets with only a minimal number of additional samples. LinC’s ability to reduce ECE and variance across different sets of demonstrations, and maintain robustness towards varying label proportions, prompt templates, and demonstration permutations implies that the predictions made by the LLM were more reliable, supporting our original conclusion from the Shannon entropy metric analysis. We believe that these findings carry important implications for future research and the development of more reliable and effective natural language processing models.
# Limitations and Future Work
While our focus remains on calibrating the model for better accuracy and reliability, it is worth discussing how to combine our framework with approaches for selecting better examples and prompt templates, such as those proposed in [31, 47]. Our work focuses on querying LLMs, which can generate content with potential ethical risks such as fairness and bias; thus, combining our framework with methods [17] that mitigate such risks is worth further discussion. One limitation of our work is the choice of k-shot validation demonstrations used to optimize the calibration parameters can impact the quality of the learned parameters, potentially leading to suboptimal results. Besides overcoming this limitation, we aim to expand our method to other NLP tasks such as summarization, text generation, and generative question answering.
# Acknowledgment
The work of T. Chen and M. Abbas was supported by the Rensselaer-IBM AI Research Collaboration (http://airc.rpi.edu), part of the IBM AI Horizons Network (http://ibm.biz/AIHorizons).
# References
[1] Momin Abbas, Quan Xiao, Lisha Chen, Pin-Yu Chen, and Tianyi Chen. Sharp-maml: Sharpnessaware model-agnostic meta learning. In Proc. of International Conference on Machine Learning, Baltimore, MD, 2022. [2] Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. Gpt-neox-20b: An open-source autoregressive language model. arXiv preprint arXiv:2204.06745, 2022. [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel HerbertVoss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901, 2020.
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel HerbertVoss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901, 2020. [4] Rich Caruana. Multitask learning. Machine learning, 28:41–75, 1997. [5] Stephanie Chan, Adam Santoro, Andrew Lampinen, Jane Wang, Aaditya Singh, Pierre Richemond, James McClelland, and Felix Hill. Data distributional properties drive emergent incontext learning in transformers. In Advances in Neural Information Processing Systems, volume 35, pages 18878–18891. Curran Associates, Inc., 2022. [6] Lisha Chen, Songtao Lu, and Tianyi Chen. Understanding benign overfitting in gradient-based meta learning. In Advances in Neural Information Processing Systems, pages 19887–19899, 2022. [7] Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, and He He. Meta-learning via language model in-context tuning. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, Dublin, Ireland, May 2022. [8] Leda Cosmides and John Tooby. Are humans good intuitive statisticians after all? rethinking some conclusions from the literature on judgment under uncertainty. Cognition, 58(1):1–73, 1996. [9] Ido Dagan, Oren Glickman, and Bernardo Magnini. The pascal recognising textual entailment challenge. In First PASCAL Machine Learning Challenges Workshop, pages 177–190, 2005. 10] Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can GPT learn in-context? language models secretly perform gradient descent as meta-optimizers. In Findings of the Association for Computational Linguistics, pages 4005–4019, Toronto, Canada, July 2023. 11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 12] Tuan Dinh, Yuchen Zeng, Ruisu Zhang, Ziqian Lin, Michael Gira, Shashank Rajput, Jy-yong Sohn,
Dimitris Papailiopoulos, and Kangwook Lee. Lift: Language-interfaced fine-tuning for non-language machine learning tasks. In Advances in Neural Information Processing Systems, pages 11763–11784, 2022. [13] Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, and Hsiao-Wuen Hon. Unified language model pretraining for natural language understanding and generation. In Advances in Neural Information Processing Systems, volume 32, 2019. [14] C. Finn, P. Abbeel, and S. Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Proc. of International Conference on Machine Learning, 2017. [15] Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn incontext? a case study of simple function classes. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 30583–30598. Curran Associates, Inc., 2022. [16] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017. [17] Umang Gupta, Jwala Dhamala, Varun Kumar, Apurv Verma, Yada Pruksachatkun, Satyapriya Krishna, Rahul Gupta, Kai-Wei Chang, Greg Ver Steeg, and Aram Galstyan. Mitigating gender bias in distilled language models via counterfactual role reversal. arXiv preprint arXiv:2203.12574, 2022. [18] Zhixiong Han, Yaru Hao, Li Dong, Yutao Sun, and Furu Wei. Prototypical calibration for few-shot learning of language models. In Proc. of International Conference on Learning Representations, 2023. [19] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. [20] Kaiyi Ji, Junjie Yang, and Yingbin Liang. Theoretical convergence of multi-step model-agnostic meta-learning. J. Mach. Learn. Res., 23(1), jan 2022. [21] Xiaoqian Jiang, Melanie Osl, Jihoon Kim, and Lucila Ohno-Machado. Calibrating predictive model estimates to support personalized medicine. Journal of the American Medical Informatics Association : JAMIA, 19:263–74, 03 2012.
[22] Zhongtao Jiang, Yuanzhe Zhang, Cao Liu, Jun Zhao, and Kang Liu. Generative calibration for incontext learning. arXiv preprint arXiv:2310.10266, 2023. [23] Suhyun Kang, Duhun Hwang, Moonjung Eo, Taesup Kim, and Wonjong Rhee. Meta-learning with a geometry-adaptive preconditioner. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16080–16090, June 2023. [24] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. [25] Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang. Fine-tuning can distort pretrained features and underperform out-of-distribution. In International Conference on Learning Representations, 2022. [26] Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres. Quantifying the carbon emissions of machine learning. arXiv preprint arXiv:1910.09700, 2019. [27] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. [28] Itay Levy, Ben Bogin, and Jonathan Berant. Diverse demonstrations improve in-context compositional generalization. arXiv preprint arXiv:2212.06800, 2022. [29] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019. [30] Yingcong Li, Muhammed Emrullah Ildiz, Dimitris Papailiopoulos, and Samet Oymak. Transformers as algorithms: Generalization and stability in in-context learning. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on
Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 19565–19594. PMLR, 23–29 Jul 2023.
Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 19565–19594. PMLR, 23–29 Jul 2023. [31] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for GPT-3? In Proceedings of Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pages 100–114, Dublin, Ireland, May 2022. [32] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Proceedings of Annual Meeting of the Association for Computational Linguistics, pages 8086–8098, Dublin, Ireland, May 2022. [33] Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. MetaICL: Learning to learn in context. In Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz, editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2791–2809, Seattle, United States, July 2022. Association for Computational Linguistics. [34] Marius Mosbach, Maksym Andriushchenko, and Dietrich Klakow. On the stability of fine-tuning {bert}: Misconceptions, explanations, and strong baselines. In International Conference on Learning Representations, 2021. [35] Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Proc. of AAAI Conference on Artificial Intelligence, pages 2901– 2907, April 2015. [36] Bo Pang and Lillian Lee. A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics (ACL-04), pages 271–278, Barcelona, Spain, July 2004. [37] David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. arXiv preprint arXiv:2104.10350, 2021. [38] John Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Adv. Large Margin Classif., 10, 06 2000.
[39] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. [40] Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021. [41] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020. [42] Jonathan S. Rosenfeld, Amir Rosenfeld, Yonatan Belinkov, and Nir Shavit. A constructive prediction of the generalization error across scales. In International Conference on Learning Representations, 2020. [43] Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M Rush. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations, 2022. [44] Claude E Shannon. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948. [45] Chenglei Si, Zhe Gan, Zhengyuan Yang, Shuohang Wang, Jianfeng Wang, Jordan Lee Boyd-Graber, and Lijuan Wang. Prompting GPT-3 to be reliable. In The Eleventh International Conference on Learning Representations, 2023. [46] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on
empirical methods in natural language processing, pages 1631–1642, 2013.
[47] Taylor Sorensen, Joshua Robinson, Christopher Rytting, Alexander Shaw, Kyle Rogers, Alexia Delorey, Mahmoud Khalil, Nancy Fulda, and David Wingate. An information-theoretic approach to prompt engineering without ground truth labels. In Proceedings of Annual Meeting of the Association for Computational Linguistics, pages 819–862, Dublin, Ireland, May 2022.
[48] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
[48] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
[49] Joaquin Vanschoren, Jan N Van Rijn, Bernd Bischl, and Luis Torgo. Openml: networked science in machine learning. ACM SIGKDD Explorations Newsletter, 15(2):49–60, 2014.
[50] Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pages 35151–35174. PMLR, 2023.
[51] Ellen M. Voorhees and Dawn M. Tice. Building a question answering test collection. page 200–207, New York, NY, USA, 2000. Association for Computing Machinery.
2] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.
[53] Ben Wang and Aran Komatsuzaki. Gpt-j-6b: A 6 billion parameter autoregressive language model, 2021.
[54] Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, and William Yang Wang. Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning. In Workshop on Efficient Systems for Foundation Models @ ICML2023, 2023.
[55] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022. [56] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 24824–24837. Curran Associates, Inc., 2022. [57] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of incontext learning as implicit bayesian inference. In International Conference on Learning Representations, 2022. [58] Benfeng Xu, Quan Wang, Zhendong Mao, Yajuan Lyu, Qiaoqiao She, and Yongdong Zhang. $k$NN prompting: Beyond-context learning with calibration-free nearest neighbor inference. In The Eleventh International Conference on Learning Representations, 2023. [59] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language understanding. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. [60] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022. [61] Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. [62] Yu Zhang and Qiang Yang. A survey on multi-task learning. IEEE Transactions on Knowledge and Data Engineering, 34(12):5586–5609, 2022. [63] Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on
Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 12697–12706. PMLR, 18–24 Jul 2021. [64] Han Zhou, Xingchen Wan, Lev Proleev, Diana Mincu, Jilin Chen, Katherine Heller, and Subhrajit Roy. Batch calibration: Rethinking calibration for in-context learning and prompt engineering. arXiv preprint arXiv:2309.17249, 2023.
Supplementary Material for
“Enhancing In-context Learning via Linear Probe Calibration"
Format # Prompt Template
Label Space
1
Review: Perhaps the best sports movie I have ever seen.
Sentiment: Positive
Review: This pathetic junk is barely an hour long.
Sentiment:
Positive, Negative
2
Input: Perhaps the best sports movie I have ever seen.
Prediction: Positive
Input: This pathetic junk is barely an hour long.
Prediction:
Positive, Negative
3
Review: Perhaps the best sports movie I have ever seen.
Sentiment: good
Review: This pathetic junk is barely an hour long.
Sentiment:
good, bad
4
Input: Perhaps the best sports movie I have ever seen.
Prediction: good
Input: This pathetic junk is barely an hour long.
Prediction:
good, bad
5
Perhaps the best sports movie I have ever seen. My overall
feeling was that the movie was good
This pathetic junk is barely an hour long.
My over-
all feeling was that the movie was
good, bad
6
Review: Perhaps the best sports movie I have ever seen.
Question: Is the sentiment of the above review Positive or
Negative?
Answer: Positive
Review: This pathetic junk is barely an hour long.
Question: Is the sentiment of the above review Positive or
Negative?
Answer:
Positive, Negative
Table 5: A list of different prompt templates that were used to investigate the impact of templates on SST-2. For bre here we show only one demonstration.
Dataset
Prompt Template
Label Space
All OpenML datasets When we have x1=r.x1, x2=r.x2, . . . , xK=r.xD, what should
be y? ### y=r.y @@@
0,. . . ,C
ble 6: Prompt template used for non-language classification tasks using the real tabular datasets in OpenML [49]; for
Dataset
Prompt Template
Label Space
All OpenML datasets When we have x1=r.x1, x2=r.x2, . . . , xK=r.xD, what should
be y? ### y=r.y @@@
0,. . . ,C
Table 6: Prompt template used for non-language classification tasks using the real tabular datasets in OpenML [49]; for a sample r, D denotes the number of features; Following [12], we follow OpenAI’s recommendation by using "###" for separating questions and answers, and "@@@" to indicate the end of answer.
ble 6: Prompt template used for non-language classification tasks using the real tabular datasets in OpenML [49]; for a mple r, D denotes the number of features; Following [12], we follow OpenAI’s recommendation by using "###" for parating questions and answers, and "@@@" to indicate the end of answer.
Dataset
Prompt Template
Label Space
SST-2
Review: Perhaps the best sports movie I have ever seen.
Sentiment: Positive
Review: This pathetic junk is barely an hour long.
Sentiment:
Positive, Negative
AGNews
Classify the news articles into the categories of World, Sports,
Business, and Technology.
Article: UK lender Barclays says it is in talks with South
Africa’s Absa about buying a majority stake in the bank.
Answer: Business
Article: New music sharing network allows users to amass
points by referring buyers.
Answer:
World, Sports, Business, Tech-
nology
TREC
Classify the questions based on whether their answer type is
a Number, Location, Person, Description, Entity, or Abbre-
viation.
Question: What does the abbreviation AIDS stand for?
Answer Type: Abbreviation
Question: What country do the Galapagos Islands belong
to?
Answer Type:
Number, Location, Person, De-
scription, Entity, Abbreviation
DBPedia
Classify the documents based on whether they are about a
Company, School, Artist, Athlete, Politician, Transportation,
Building, Nature, Village, Animal, Plant, Album, Film, or
Book.
Article: Hoodlum & Son is a 2003 comedy-crime film.
Answer: Film
Article: Nachan Main Audhay Naal is the seventh album
of Pakistani pop and bhangra singer Abrar-ul-Haq. It was
released on March 2007.
Answer:
Company, School, Artist, Ath-
lete, Politician, Transporta-
tion, Building, Nature, Village,
Animal, Plant, Album, Film,
Book
SST-5
Review: The film is bright and flashy in all the right ways.
Sentiment: great
Review: The film never finds its tone and several scenes run
too long.
Sentiment:
terrible, bad, okay, good, great
Subj
Input: All social structures break down and a new world
order emerges from the heart of the desert.
Type: objective
Input: A zombie movie in every sense of the word – mindless,
lifeless, meandering, loud , painful, obnoxious.
Type:
objective, subjective
RTE
Experts say that Mr. Abbas will need that big win to show
that he has the support of most Palestinian people in order
to push through his aims of peace talks with Israel.
question: Analysts had said that Mr. Abbas needed a large
margin of victory in order to push his agenda of peace talks
with Israel. True or False?
answer: True
The city is twinned with Glasgow, Dortmund, Pleven, and
Le Mans.
question: Dortmund is twinned with Glasgow. True or False?
answer:
True, False
Table 7: The prompts templates used for different datasets. For brevity, here we show only one demonstration per datase
# B Additional Experiments
In this section, we provide details of the experimental set-up and present additional results.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/cd38/cd380b62-86f4-44ad-9d68-42c0ebe8c980.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: Shannon entropy histograms of using vanilla ICL on GPT-2-XL (1.5B) vs our method on Subj (higher entropy implies higher uncertainty); we use logarithmic base two. Refer to Section 2 for a detailed explanation.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3e34/3e3492cb-f6a1-48ca-8829-700e060b1393.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: Shannon entropy histograms of using vanilla ICL on GPT-2-XL (1.5B) vs our method on RTE (higher entropy implies higher uncertainty); we use logarithmic base two. Refer to Section 2 for a detailed explanation.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0126/012699ab-e981-4779-9ba7-0038df404db5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10: Shannon entropy histograms of using vanilla ICL on GPT-2-XL (1.5B) vs our method on AGNews (higher entropy implies higher uncertainty); we use logarithmic base two. Refer to Section 2 for a detailed explanation.</div>
Hyperparameter search. After finding Nv and T via random search using the sets defined in Section 5.1, we narrowly fine-tune the learning rate α for each experiment from the range [1e −5, 2e1]. Therefore, in each experiment, the value of α might vary, and to ensure transparency and reproducibility, we have compiled some hyperparameter sets in the examples_ssh folder within the provided code. Furthermore, given that the calibration network constitutes a basic linear convex problem, determining an appropriate learning rate presents no issue, as the fixed rate can readily be substituted with a schedule that gradually reduces the learning rate as training advances, based on specified criteria (e.g., a linear scheduler that gradually diminishes the rate for each parameter group by applying a small multiplicative factor until a predetermined epoch count is attained). Where is the affine transformation applied? The classification tasks we considered apply the affine transformation to the set of probabilities that are associated with each class in the label space i.e. task-specific
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5bc1/5bc179b4-06cf-4524-b112-f7c1733ea94f.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 11: Shannon entropy histograms of using vanilla ICL on Llama-2 (13B) vs our method on SST-2 (higher entropy implies higher uncertainty); we use logarithmic base two. Refer to Section 2 for a detailed explanation.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/934c/934c970c-9545-47a0-862a-1713a79b96da.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 12: Shannon entropy histograms of using vanilla ICL on Llama-2 (13B) vs our method on Subj (higher entropy implies higher uncertainty); we use logarithmic base two. Refer to Section 2 for a detailed explanation.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f876/f876ae17-e4d7-471d-847b-e49e48957a86.png" style="width: 50%;"></div>
<div style="text-align: center;">e 13: Shannon entropy histograms of using vanilla ICL on Llama-2 (13B) vs our method on RTE (higher entropy es higher uncertainty); we use logarithmic base two. Refer to Section 2 for a detailed explanation.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ebfb/ebfb4ecf-0a35-4e06-8c5d-f5bf14764233.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 14: Shannon entropy histograms of using vanilla ICL on Llama-2 (13B) vs our method on AGNews (higher entropy implies higher uncertainty); we use logarithmic base two. Refer to Section 2 for a detailed explanation.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e89b/e89b2c15-7bdf-440a-9846-2772dc1b0b12.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
<div style="text-align: center;">C diminishes the standard deviation of accuracy across different demonstrations. The difference in standard en the calibration methods and the unadjusted baseline (NoC) from Table 4 is plotted.</div>
tokens. In other words, after we get the logits which are in the dimension (batch_size, seq_length, vocab_size), we sample the tokens that correspond to the classes in the label space to a get a probability vector of dimension (batch_size, num_classes). We then apply the affine transformation on this probability vector. Model Shots Method SST-5 AGNews NoC 32.3 57.3 Expected Calibration Error (ECE) [35]. The ECE computes the Expected Calibration Error across the bins in the following manner:
tokens. In other words, after we get the logits which are in the dimension (batch_size, seq_length, vocab_size), we sample the tokens that correspond to the classes in the label space to a get a probability vector of dimension (batch_size, num_classes). We then apply the affine transformation on this probability vector.
Expected Calibration Error (ECE) [35]. The ECE computes the Expected Calibration Error across the bins in the following manner:
where om represents the accuracy i.e. the true fraction of positive instances in bin m, em represents the model confidence i.e. is the mean of the post-calibrated probabilities for the instances in bin m, and the fraction Bm n denotes the empirical probability (fraction) of all instances that fall into bin. Therefore, The ECE can be seen as evaluating the extent to which a model’s estimated "probabilities" align with the actual (observed) probabilities by computing a weighted average of the absolute difference between accuracy and confidence. A model’s calibration is considered better when ECE values are lower and an ECE of zero is indicative of a perfectly calibrated model.
# B.1 Experiment Settings
For LoRA, we used a step size of 1.5e-6, a batch size of 16, a scaling factor of 32, and incorporated a 0.1 dropout probability within the LoRA layers. For SPT, we used a step size of 3.5e-4, a batch size of 16, specified 8 virtual tokens (i.e. the num_vir_tokens argument), and employed an initial text for prompt tuning (i.e. the prompt_tun
an initial text for prompt tuning (i.e. the prompt_tuning_init_text argument) that consisted of a list of classes separated by commas (for example, "World, Sports, Business, Technology" for AGNews). Across all our experiments, we performed fine-tuning over a total of 20 epochs. For the non-language tasks that used tabular datasets from OpenML [49], following [12], we selected the maximum number of instances that could be accommodated within the contextual window for each dataset. Additionally, our results were reported as an average across three random seeds.
Model
Shots Method SST-5
AGNews
Llama-2 7B
0-shot
NoC
32.30.0
57.30.0
ConC
35.00.0
63.30.0
LinC
45.00.0 82.00.0
1-shot
NoC
39.39.8
83.92.1
ConC
43.01.1
82.03.3
LinC
50.13.5 84.32.6
4-shot
NoC
49.71.9
87.21.1
ConC
44.32.5
87.21.2
LinC
53.52.8 87.61.1
8-shot
NoC
48.95.5
86.70.9
ConC
48.72.2
87.50.5
LinC
52.03.3 87.60.6
Llama-2 13B
0-shot
NoC
34.00.0
73.30.0
ConC
33.00.0
72.30.0
LinC
47.30.0 85.70.0
1-shot
NoC
43.93.5
81.52.7
ConC
42.43.9
80.61.9
LinC
51.01.6 84.11.7
4-shot
NoC
48.74.1
82.73.7
ConC
44.95.2
80.52.3
LinC
53.00.9 86.01.2
8-shot
NoC
49.17.3
87.00.5
ConC
47.14.4
83.81.6
LinC
51.46.1 87.10.6
Table 11:
Table 11: Performance under same model family (i.e. Llama-2) but different sizes (7B/13B).
Table 11: Performance under same model family (i.e. Llama-2) but different sizes (7B/13B).
Shot
Method
SST-2
SST-5
AGNews TREC
DBpedia RTE
Subj
Avg
GPT-J 6B
0-shot
No Calibration
66.30.0
33.70.0
36.00.0
24.70.0
19.70.0
55.60.0
65.70.0
43.1
Contextual Calibration
58.00.0
40.70.0
56.00.0
40.00.0
48.00.0
52.40.0
58.70.0
50.5
Prototypical Calibration‡ 74.20.2
42.10.8
55.10.4
53.46.1
66.11.5
57.01.0 69.50.2
59.6
Prototypical Calibration∗74.11.1
32.23.8
49.84.3
44.76.7
20.74.7
56.51.5
66.81.8
49.3
Linear Calibration
74.30.0 46.00.0 64.30.0
70.70.0 69.30.0
56.70.0
70.70.0 64.6
1-shot
No Calibration
67.36.7
35.33.5
65.314.3
40.38.8
64.916.6
50.73.7
65.19.9
55.6
Contextual Calibration
88.31.7
46.93.1
74.95.8
62.64.7
80.23.3
53.41.2
59.12.2
66.5
Prototypical Calibration‡ 90.81.7 47.62.5
79.85.4
55.36.4
90.02.2
56.73.1
77.94.8 71.2
Prototypical Calibration∗89.31.9
35.28.9
78.17.5
45.99.6
62.24.9
57.71.8 74.95.9
63.2
Linear Calibration
88.51.7
50.41.3 81.74.6
63.04.4 82.63.7
56.01.8
69.97.5
70.3
4-shot
No Calibration
88.93.3
46.33.2
72.36.1
37.74.2
82.114.4
55.07.1
57.46.9
62.8
Contextual Calibration
92.83.1
49.44.8
75.24.1
46.05.1
88.94.9
56.01.9
65.311.8 67.7
Prototypical Calibration‡ 95.00.4 46.24.6
79.96.6
57.15.3
91.92.6
61.22.7 79.45.8 73.0
Prototypical Calibration∗94.60.6
47.95.5
81.52.0
49.010.2 63.84.3
60.83.1
78.98.0
68.1
Linear Calibration
94.90.9
51.13.7 77.95.5
65.31.5 89.45.4
58.43.7
68.411.2 72.2
8-shot
No Calibration
91.86.0
44.54.3
76.89.9
43.56.3
88.63.2
60.32.8
82.25.7
69.7
Contextual Calibration
93.81.8
44.44.