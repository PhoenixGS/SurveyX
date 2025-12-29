# In-Context Learning with Noisy Labels

# Junyong Kang jykang@kaist.ac.kr KAIST Seoul, South Korea
Do happ Seoul N Seou

# Junyong Kang jykang@kaist.ac.kr KAIST Seoul, South Korea

Hwanjun Song songhwanjun@kaist.ac.kr KAIST Deajeon, South Korea

# ABSTRACT

In-context learning refers to the emerging ability of large language models (LLMs) to perform a target task without additional training, utilizing demonstrations of the task. Recent studies aim to enhance in-context learning performance by selecting more useful demonstrations. However, they overlook the presence of inevitable noisy labels in task demonstrations that arise during the labeling process in the real-world. In this paper, we propose a new task, in-context learning with noisy labels, which aims to solve real-world problems for in-context learning where labels in task demonstrations would be corrupted. Moreover, we propose a new method and baseline methods for the new task, inspired by studies in learning with noisy labels. Through experiments, we demonstrate that our proposed method can serve as a safeguard against performance degradation in in-context learning caused by noisy labels.

# CCS CONCEPTS
• Computing methodologies → Natural language processing

# • Computing methodologies → Natural language processing.
KEYWORDS

KEYWORDS

in-context learning, learning with noisy labels, large language mo els

ACM Reference Format: Junyong Kang, Donghyun Son, Hwanjun Song, and Buru Chang ∗. 2018. In-Context Learning with Noisy Labels. In Proceedings of Make sure to enter the correct conference title from your rights confirmation emai (Conference acronym ’XX). ACM, New York, NY, USA, 5 pages. https://doi.org/ XXXXXXX.XXXXXXX

# 1 INTRODUCTION

Have you ever doubted whether the labels in your exemplars could be incorrect?

The corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. Conference acronym ’XX, June 03–05, 2018, Woodstock, NY © 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/18/06 https://doi.org/XXXXXXX.XXXXXXX

Donghyun Son happydh1@snu.ac.kr Seoul National University Seoul, South Korea

Buru Chang ∗
buru@sogang.ac.kr Sogang University Seoul, South Korea

(a) Noisy labels included in demonstrations for in-context learning

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/58c5/58c56790-18a8-427b-8c61-dd4b4ad6b91c.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Performance Degradation by noisy labels
</div>
<div style="text-align: center;">(b) Performance Degradation by noisy labels
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3c58/3c58eb17-3408-49fc-a914-f35f4462a0db.png" style="width: 50%;"></div>
<div style="text-align: center;">10% 20%30%40% 50% NoiseRate
</div>
Figure 1: (a) The Tweet dataset [3] contains noisy labels. Thus, the demonstrations collected from the dataset would include noisy labels. (b) These noisy labels degrade the performance of in-context learning and decrease the stability.

In-context learning [6, 27] is an emerging property of Large Language Models (LLMs) trained on vast amounts of training data. Without the need for additional fine-tuning on the target task, LLMs can perform the target task by taking task descriptions and a few demonstrations (exemplars) relevant to the task. Recently, many studies have been proposed to enhance the performance of in-context learning. These studies aim to select more suitable demonstrations from the predefined retrieval set for a given test data [13, 22, 28] or enable the utilization of more demonstrations by alleviating the limitation of input length in LLMs [7, 11]. However, these studies overlook the real-world scenarios that the retrieved demonstrations would include noisy labels. In real-world labeled datasets, such as hate speech detection datasets [3], corrupted labels often emerge due to the subjective

nature of annotators and label ambiguity [20]. Given attempts to scale up the usage of demonstrations to the order of 1,000 [11], prompts unavoidably include noisy labels. In addition to previous works [19, 26], our experimental results in Figure 1 show that these noisy labels not only severely degrade the performance of in-context learning but also reduce the stability of the predictive results (See details in § 4. 4). This observation underscores the need to handle noisy labels, as LLMs actually learn tasks from exemplars through in-context learning [2, 25], and such labels can interfere with this learning process. To address this issue, we introduce a novel task, named  "incontext learning with noisy labels," by establishing a connection between the existing"learning with noisy labels" problem [16, 20] and"in-context learning" problem in LLMs. The former problem aims to build a robust model from the corrupted dataset having noisy labels. The new task assumes that a certain proportion of labels in a retrieval set are corrupted and evaluates the performance of in-context learning accordingly. As the first comprehensive study on the new task, we present baseline methods for the new task. These baseline methods are adapted versions of representative methods proposed in existing research on learning with noisy labels [17, 29, 31], tailored to the in-context learning scenario. Furthermore, we propose a new rectifying method that is more universally applicable to the in-context learning with noisy label setting. Experiments on classification tasks demonstrate the importance of handling noisy labels in the in-context learning setting, and the proposed method shows its ability to serve as a safeguard against performance degradation caused by noisy labels.

# 2 PROBLEM FORMULATION

In this section, we formally present a new task, denoted as  incontext learning with noisy labels. Firstly, we formulate the problem of in-context learning, then extend the problem to the new task to address the real-world scenarios where the demonstrations would include noisy labels.

In-context learning. In-context learning is a paradigm where the LLM performs a new task without additional training by leveraging its powerful knowledge. For this purpose, the LLM predicts the target label 𝑦 along with the input query 𝑥 and 𝑛-ordered demonstrations for the target task. Following the previous studies [22, 28], we retrieve the most relevant demonstrations [(𝑥 1,𝑦 1), · · ·, (𝑥 𝑛,𝑦 𝑛)] from the pre-defined retrieval set based on the input query 𝑥. Subsequently, the demonstrations are concatenated to the input query 𝑥 as follows [𝑥 1,𝑦 1, · · ·,𝑥 𝑛,𝑦 𝑛,𝑥], and then it is fed to the frozen LLM to compute the probability distribution. Note that we focus on text classification tasks where the target label 𝑦 is in a set of 𝑚 candidate labels {𝑐 1, · · ·,𝑐 𝑚} to simplify our problem. Thus, we decode the prediction 𝑦 by comparing the negative log-likelihood of the candidate labels and choose the minimal one.

In-context learning with noisy labels. In real-world labeled datasets, labels would be corrupted due to the nature of labeling process [20]. Therefore, we present a new task, in-context learning with noisy labels, by extending the problem of in-context learning to consider the real-world scenarios.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/cb2b/cb2b4416-5cf0-4c8f-9b48-423c744e75ab.png" style="width: 50%;"></div>
# Figure 2: An example of the prompt format of the rectification method. The examples are collected from SST-5.

Figure 2: An example of the prompt format of the rectification method. The examples are collected from SST-5.

We assume that the pre-defined retrieval set D is corrupted by various noise rates 𝑟 𝑖 ∈[0, 1], yielding noisy retrieval set. Specifically, we select 𝑟 𝑖 × |D| demonstrations and flip each of label 𝑦 to ¯ 𝑦 ∈{𝑐 1, ...,𝑐 𝑚} −{𝑦} with uniform transition probability, following the previous study [20]. As in traditional learning with noisy labels problem [12, 16, 24], we assume that a small size of clean dataset is accessible. Such a small-sized clean set can be constructed at a relatively low cost and is effective in significantly building a robust model to noisy labels. We sample a small clean subset D ′ and use it to make models more robust to label noise.

# 3 METHOD

We present several methods to perform the new task making incontext learning more robust in noisy labels. In contrast to the learning with noisy labels problem, a significant challenge in the new task is that parameters of the LLM must be kept frozen instead of updating them to perform the task. Hence, we focus on manipulating the labels of task demonstrations.

# 3.1 Baseline Methods

We present four baseline methods for standard classification tasks named correction, weighting, reordering, and selection. In learning with noisy labels problem, methods such as loss adjustment and sample selection [24] manipulate the effect of a noisy example by loss reweighting and sampling strategy based on its confidence. Motivated from these approaches, our baseline methods utilize a classifier (e.g. fine-tuned BERT) trained on each target dataset to give auxiliary information to LLM using its output probabilities.

Correction. is the simplest method which aims to directly correct noisy labels. This approach overwrites all demonstration labels with the output decision of the classifier.

Weighting. puts additional information next to the each prompt label, showing its confidence value from the classifier. We verbalize

the confidence value as either"high" or"low" rather than the real number, as classifiers can produce extremely skewed scores [10].

Reordering. utilizes the confidence value to replace position of each demonstration, placing low-confidence demonstrations proceeding to high-confidence ones. LLMs puts different importance to each demonstrations depending on its position, typically higher weights to last sentences [18, 30]. Therefore, reordering can be interpreted as an implicit weighting by the order bias of LLMs.

Selection.  also utilizes the confidence value to reduce the influences of noisy labels by discarding demonstrations with low confidence labels in the prompt. As an example, we take the threshold 𝜃 = 0. 3 for the experiments.

# 3.2 Rectification

We now propose our main method rectification, which processes multiple noisy demonstrations at a time. As shown in Figure 2, rectification receives a sequence of noisy demonstrations and outputs a sequence of corrected labels. To achieve this, we fine-tune a pre-trained generative model (e.g., GPT-2 [21]) using negative log likelihood loss on the label tokens. Unlike baseline methods that perform classification independently for each demonstration, rectification gets a sequence of demonstrations as an input. Hence, it can reference all retrieved demonstrations and utilize them for noisy label rectification. In § 4.4, we demonstrate the effectiveness of our design choice in improving model performance.

# 4 EXPERIMENTS

We conduct experiments on three datasets: MRPC (paraphrase detection) [9], SST-5 (sentiment analysis) [23] and Tweet hate speech detection [4]. The statistics of the datasets are summarized in Table 2. We employ the training set to build the corrupted retrieval set D and sample a clean subset D ′ corresponding to 10% of the training set. Then, we evaluate the performance of our methods for in-context learning with noisy labels on the validation set.

# 4.1 Implementation Details

We adopt multiple LLMs of varying sizes, including GPT2-Neo 2.7B [5], Llama2-7B [1], and Mistral-7B [15], as inference LLMs, and use 10 task demonstrations for both training and inference. For baseline methods, we fine-tune a BERT model [8] on the clean subset D ′, initialized with the bert-base-uncased checkpoint. We also employ the oracle classifier, which is trained on clean retrieval set D. These classification models are fine-tuned for 30 epochs with batch size of 64 and learning rate of 5e-5. For the rectification method, we fine-tune a pre-trained gpt2-large, one of pre-trained versions of GPT-2 [21]. To construct training dataset for the rectification method, we collect task demonstrations from the clean subset D ′ with the EPR retriever [22], and then add label noise to the demonstrations by sampling 𝑟 ∈{0. 1, 0. 2, 0. 3, 0. 4, 0. 5}. We trained the model for 10 epochs with a batch size of 2 and a learning rate of 1e-4, employing LoRA [14] for memory-efficient training. We split the training set for these methods to create a validation set.

# 4.2 Main results

Table 1 shows the performance of our baseline methods and the rectification method. We report the performance using the EPR retriever and the TopK-similarity retriever based on BERT embedding of demonstration inputs. In the GPT2-Neo results, without any label manipulation, we observe that the performance of incontext learning degrades as the noise rate increases. In contrast, the correction method acts as a robust baseline, maintaining consistent performance across all noise rates. The weighting method shows some improvement compared to no manipulation, but fails to achieve consistent performance as the noise rate increases and sometimes performs worse in the Tweet. The reordering and the selection method shows rapid performance degradation in MRPC due to the imperfect classifier. In SST-5, the selection is relatively steady, and the reordering retains high accuracy at lower noise rates. Although these methods successfully defends against performance drop in Tweet with the EPR retriever, we observe failures when using the TopK-BERT retriever. On the other hand, the rectification method shows superior performance over baseline methods, effectively defending the performance against noisy labels. For Llama2-7B and Mistral-7B, we report the performance of the correction and the rectification method. These LLMs show better noise robustness at lower noise rates, Llama2-7B on SST-5 or Mistral-7B on Tweet with the EPR retriever for instance. However, their robustness is not consistent across datasets or retrievers. For example, we found that the EPR mostly retrieves non-hate demonstrations for evaluation queries, which induce lower performance but higher robustness. On the other hand, the TopK-BERT retrieves more balanced demonstrations, resulting in a clear drop in accuracy as the noise rate increases. Regardless of these variations, the rectification method consistently maintains accuracy across noise rates and outperforms the correction method.

# 4.3 Analysis

Stability. With our method, not only accuracy but also stability can be improved. In Figure 1, we visualize the in-context learning performance of the no manipulation version and the rectification method with 10 different random seeds. We also summarize the mean and standard deviation of performance for each dataset in Table 3, averaging the mean and the standard deviation across noise rates. To solely measure the variability due to each noise rate, we retrieve demonstrations from D and then randomly corrupt them by rate 𝑟 𝑖 in this experiment. Overall, the rectification method shows smaller variance compared to no manipulation within each noise rate, which indicates that the stability is improved. Rectification Accuracy. The performance of in-context learning with noisy labels depends on how well it rectifies the noisy labels. To measure the rectification performance, we define the rectification accuracy as 𝜏 = 1
𝑁𝐾 � 𝑁 𝑛 = 1 � 𝐾 𝑘 = 1 1 (𝑦 𝑘𝑛 = ˜ 𝑦 𝑘𝑛), where 𝑁, 𝐾 denote the number of sets of demonstrations and the number of demonstrations in a set, 𝑦 𝐾 𝑛 and ˜ 𝑦 𝑘 𝑛 denote 𝑘 ’th label and output of 𝑛 ’th set, respectively. In Table 4, we compare rectification accuracy between classification models and our rectification method. Specifically, we adopt the BERT classifier from previous sections and a classifier trained from gpt2-large to match model capacity. The rectification

<div style="text-align: center;">Table 1: Experimental results on GPT2-Neo (2.7B), Llama2-7B, and Mistral-7B
</div>
Model
Retriever
Method
MRPC
SST-5
Tweet
0
0.1
0.2
0.3
0.4
0.5
0
0.1
0.2
0.3
0.4
0.5
0
0.1
0.2
0.3
0.4
0.5
GPT2-Neo
EPR
w/o manipulation
78.4
76.0
75.5
67.9
47.8
51.5
46.9
44.1
41.6
38.5
36.0
31.8
58.4
59.7
55.5
54.7
53.6
47.4
Correction
72.1
38.9
57.3
Weighting
78.2
77.2
74.8
70.8
53.4
58.3
48.5
46.8
44.5
42.5
39.2
35.9
58.1
61.4
57.2
55.4
54.0
46.8
Reordering
79.4
68.1
62.0
49.0
35.0
36.8
46.6
46.3
45.3
42.5
38.5
35.9
57.7
58.3
57.5
58.7
58.0
55.4
Selection
56.4
49.3
43.6
40.7
35.0
36.3
45.6
44.7
44.4
43.2
43.2
38.5
57.3
57.3
57.3
57.3
57.3
56.4
Rectification
78.4
77.7
77.2
77.2
75.5
76.2
44.7
44.8
44.4
45.3
45.5
44.5
58.3
58.3
58.3
58.4
58.5
58.7
TopK-
BERT
w/o manipulation
70.3
65.4
62.5
61.0
53.4
52.5
35.6
33.6
31.7
33.0
29.0
26.4
65.7
63.3
60.1
58.8
52.8
50.3
Correction
67.0
33.8
57.3
Weighting
71.6
69.1
67.2
65.4
56.4
56.4
36.9
36.5
35.5
32.7
33.9
30.8
65.3
63.8
59.6
57.0
50.2
48.7
Reordering
52.7
46.6
42.2
40.2
34.6
34.3
36.4
36.1
35.1
33.8
31.9
29.1
69.7
66.1
62.7
60.4
56.4
58.5
Selection
44.6
41.4
40.0
35.5
33.8
32.8
34.3
33.8
34.9
33.2
32.4
32.9
61.5
60.6
59.9
59.5
56.3
57.7
Rectification
71.1
71.1
70.3
71.3
68.2
67.4
36.6
36.8
36.0
35.6
36.0
36.2
64.4
64.2
63.7
63.1
63.5
64.0
Llama2-7B
EPR
w/o manipulation
77.7
75.5
76.5
74.3
62.3
62.3
50.2
49.3
51.0
47.6
45.3
45.4
59.5
61.5
58.6
60.0
57.5
52.9
Correction
72.5
43.7
57.3
Rectification
78.2
78.4
78.4
78.4
76.7
75.8
50.3
50.5
50.3
50.2
50.6
49.4
59.9
59.8
59.9
59.6
60.3
59.4
TopK-
BERT
w/o manipulation
70.3
70.6
69.1
66.9
59.1
60.0
49.8
47.2
49.0
47.3
43.5
40.4
71.4
69.7
67.5
64.7
56.9
55.8
Correction
69.6
46.3
57.4
Rectification
73.5
71.6
72.3
70.6
71.3
71.1
51.2
50.8
51.1
50.2
50.6
49.4
71.7
71.7
71.3
69.7
71.3
70.3
Mistral-7B
EPR
w/o manipulation
77.7
77.0
76.5
76.7
61.3
64.0
51.9
50.1
48.5
45.5
44.3
40.0
66.3
68.2
63.3
65.1
61.8
55.8
Correction
74.5
44.8
57.3
Rectification
78.4
77.9
77.9
78.9
77.0
77.0
49.4
49.9
49.5
49.9
49.8
49.8
66.0
65.8
65.8
65.8
65.8
65.8
TopK-
BERT
w/o manipulation
72.3
72.1
70.3
67.9
62.3
62.3
47.7
46.4
44.8
42.7
40.3
39.2
74.6
71.5
67.5
65.9
57.5
57.5
Correction
71.6
43.6
57.2
Rectification
73.0
72.3
72.5
71.8
71.8
71.6
49.0
48.7
49.7
49.1
50.2
48.4
72.1
71.5
71.4
70.2
70.3
69.1
<div style="text-align: center;">Table 2: Dataset statistics and input formats.
</div>
Dataset
# of train data
# of validation data
Format
MRPC
3,668
408
{sentence1} Can we say
"{sentence2}"? {No, Yes}
SST-5
8,534
1,101
{question} It is {terrible,
bad,OK,good,great}
Tweet
9,000
1,000
Tweet: {question}
Hate: {No, Yes}
Table 3: Experimental results on stability. For MRPC and SST5, we used the EPR retriever and the TopK-BERT for Tweet.

<div style="text-align: center;">Table 3: Experimental results on stability. For MRPC and SST5, we used the EPR retriever and the TopK-BERT for Tweet.
</div>
Model
Rectification
MRPC
SST-5
Tweet
Acc.
Std.
Acc.
Std.
Acc.
Std.
GPT2-Neo
No manipulation
65.23
1.98
37.56
1.24
57.50
1.28
Rectification
76.02
0.83
44.39
0.53
63.79
0.72
Llama2-7B
No manipulation
71.99
1.35
47.21
0.99
63.71
1.01
Rectification
77.15
0.80
49.49
0.64
70.21
0.65
method achieves higher rectification accuracy compared to classifiers across all datasets, showing that it can effectively leverages the context of demonstrations in correcting labels. Data efficiency. As observed in previous sections, the rectification method maintains the performance using only a small portion of the data. To explore the data efficiency of this method, we train the oracle model (the rectification method trained on the full dataset D), which we denote as Full. As shown in Table 5, our method

Table 4: The rectification accuracy comparison. (∗) symbol indicates the oracle classifier trained on the full training data.

Rectification Method
MRPC
SST-5
Tweet
BERT Classifier∗
97.5
59.8
91.5
BERT Classifier
81.9
49.6
91.6
GPT-2 Classifier∗
80.9
49.2
87.5
GPT-2 Classifier
72.1
42.3
68.4
Rectification (𝑟=0.1)
90.3
67.3
95.1
Rectification (𝑟=0.2)
88.6
66.3
95.1
Rectification (𝑟=0.3)
86.1
65.2
94.8
Rectification (𝑟=0.4)
83.5
64.0
94.6
Rectification (𝑟=0.5)
83.8
62.4
94.1
Table 5: Experimental results with GPT2-Neo to investigate data efficiency. The results on MRPC with the EPR retriever are reported.

Training Method
Noise rate
0
0.1
0.2
0.3
0.4
0.5
Full
77.7
78.4
77.5
76.7
76.7
77.0
2-shot
73.5
73.8
73.2
73.0
73.5
72.1
5-shot
73.3
73.8
73.3
71.3
70.4
70.1
Ours
78.4
77.7
77.2
77.2
75.5
76.2
formance to Full, though it only uses 10%

reaches comparable performance to Full, though it only uses 10% of the total data.

Why is the rectification method data-efficient? To investigate this question, we train the rectification method with fewer input demonstrations, specifically 2 and 5. During the inference, these models are executed multiple times (e.g. 5 and 2) to rectify total 10 demonstrations. Table 5 shows that these method are less effective than our approach, where 10 demonstrations are given as an input. From this observation, we conclude that the rectification method benefits from referring to the given context (demonstrations), which we believe to be crucial for the data efficiency.

# 5 CONCLUSION

In this study, we propose a new task called in-context learning with noisy labels. Additionally, we introduce baseline methods capable of performing the new task and propose a novel method to address their limitations. Given the recent surge in attempts to generate labeled data using large language models, noisy labels are inevitable. Our research highlight a new research direction that must be addressed in the era of LLM.

# REFERENCES

[1] 2023. Llama 2: Open Foundation and Fine-Tuned Chat Models. ArXiv abs/2307.09288 (2023).
[2] Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. 2022. What learning algorithm is in-context learning? Investigations with linear models. In The Eleventh International Conference on Learning Representations.
[3] Pinkesh Badjatiya, Shashank Gupta, Manish Gupta, and Vasudeva Varma. 2017. Deep learning for hate speech detection in tweets. In Proceedings of the 26th international conference on World Wide Web companion. 759–760.
[4] Valerio Basile, Cristina Bosco, Elisabetta Fersini, Debora Nozza, Viviana Patti, Francisco Manuel Rangel Pardo, Paolo Rosso, and Manuela Sanguinetti. 2019. SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter. In Proceedings of the 13th International Workshop on Semantic Evaluation. Association for Computational Linguistics, Minneapolis, Minnesota, USA, 54–63. https://doi.org/10.18653/v1/S19-2007
[5] Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. 2021.  GPTNeo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow. https: //doi.org/10.5281/zenodo.5297715
[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.
[7]  Tianle Cai, Kaixuan Huang, Jason D Lee, and Mengdi Wang. 2023. Scaling InContext Demonstrations with Structured Attention. In Workshop on Efficient Systems for Foundation Models@ ICML2023.
[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In North American Chapter of the Association for Computational Linguistics.
[9] William B. Dolan and Chris Brockett. 2005. Automatically Constructing a Corpus of Sentential Paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005). https://aclanthology.org/I05-5002
[10] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning - Volume 70 (Sydney, NSW, Australia) (ICML’17). JMLR.org, 1321–1330.
[11] Yaru Hao, Yutao Sun, Li Dong, Zhixiong Han, Yuxian Gu, and Furu Wei. 2022. Structured prompting: Scaling in-context learning to 1,000 examples. arXiv preprint arXiv:2212.06713 (2022).
[12] Dan Hendrycks, Mantas Mazeika, Duncan Wilson, and Kevin Gimpel. 2018. Using trusted data to train deep networks on labels corrupted by severe noise. Advances in neural information processing systems 31 (2018).
[13] SU Hongjin, Jungo Kasai, Chen Henry Wu, Weijia Shi, Tianlu Wang, Jiayi Xin, Rui Zhang, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, et al. 2022. Selective Annotation Makes Language Models Better Few-Shot Learners. In The Eleventh International Conference on Learning Representations.
[14] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations. https: //openreview.net/forum?id=nZeVKeeFYf9
[15]  Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel,

Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7B. arXiv preprint arXiv:2310.06825 (2023).
[16]  Seong Min Kye, Kwanghee Choi, Joonyoung Yi, and Buru Chang. 2022. Learning with noisy labels by efficient transition matrix estimation to combat label miscorrection. In European Conference on Computer Vision. Springer, 717–738.
[17]  Tongliang Liu and Dacheng Tao. 2015. Classification with noisy labels by importance reweighting. IEEE Transactions on pattern analysis and machine intelligence 38, 3 (2015), 447–461.
[18] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity. In  Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (Eds.). Association for Computational Linguistics, Dublin, Ireland, 8086–8098. https://doi.org/10.18653/v1/2022.acllong.556
[19] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? 11048–11064. https://doi.org/10.18653/ v1/2022.emnlp-main.759
[20] Nagarajan Natarajan, Inderjit S Dhillon, Pradeep K Ravikumar, and Ambuj Tewari. 2013. Learning with noisy labels. Advances in neural information processing systems 26 (2013).
[21] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language Models are Unsupervised Multitask Learners.
[22] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2022. Learning To Retrieve Prompts for In-Context Learning. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2655–2671.
[23] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In  Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, David Yarowsky, Timothy Baldwin, Anna Korhonen, Karen Livescu, and Steven Bethard (Eds.). Association for Computational Linguistics, Seattle, Washington, USA, 1631–1642. https://aclanthology.org/D13-1170
[24] Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. 2023. Learning From Noisy Labels With Deep Neural Networks: A Survey. IEEE Transactions on Neural Networks and Learning Systems 34, 11 (2023), 8135–8153. https://doi.org/10.1109/TNNLS.2022.3152527
[25] Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. 2023. Transformers learn in-context by gradient descent. In International Conference on Machine Learning. PMLR, 35151–35174.
[26] Xindi Wang, Yufei Wang, Can Xu, Xiubo Geng, Bowen Zhang, Chongyang Tao, Frank Rudzicz, Robert E Mercer, and Daxin Jiang. 2023. Investigating the learning behaviour of in-context learning: a comparison with supervised learning. arXiv preprint arXiv:2307.15411 (2023).
[27] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2021. An Explanation of In-context Learning as Implicit Bayesian Inference. In International Conference on Learning Representations.
[28] Jiacheng Ye, Zhiyong Wu, Jiangtao Feng, Tao Yu, and Lingpeng Kong. 2023. Compositional Exemplars for In-context Learning. In Proceedings of the 40th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 202), Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (Eds.). PMLR, 39818–39833. https: //proceedings.mlr.press/v202/ye23c.html
[29] Kun Yi and Jianxin Wu. 2019. Probabilistic end-to-end noise correction for learning with noisy labels. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 7017–7025.
[30] Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate Before Use: Improving Few-shot Performance of Language Models. In Proceedings of the 38th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 139), Marina Meila and Tong Zhang (Eds.). PMLR, 12697– 12706. https://proceedings.mlr.press/v139/zhao21c.html
[31] Guoqing Zheng, Ahmed Hassan Awadallah, and Susan Dumais. 2021. Meta label correction for noisy label learning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35. 11053–11061.

