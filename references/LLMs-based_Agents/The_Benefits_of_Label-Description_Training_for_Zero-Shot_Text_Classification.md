# The Benefits of Label-Description Training for Zero-Shot Text Classification
Lingyu Gao1, Debanjan Ghosh2†, and Kevin Gimpel1† 1Toyota Technological Institute at Chicago 2Educational Testing Service {lygao, kgimpel}@ttic.edu, dghosh@ets.org
Abstract
Pretrained language models have improved zero-shot text classification by allowing the transfer of semantic knowledge from the training data in order to classify among specific label sets in downstream tasks. We propose a simple way to further improve zero-shot accuracies with minimal effort. We curate small finetuning datasets intended to describe the labels for a task. Unlike typical finetuning data, which has texts annotated with labels, our data simply describes the labels in language, e.g., using a few related terms, dictionary/encyclopedia entries, and short templates. Across a range of topic and sentiment datasets, our method is more accurate than zero-shot by 17-19% absolute. It is also more robust to choices required for zero-shot classification, such as patterns for prompting the model to classify and mappings from labels to tokens in the model’s vocabulary. Furthermore, since our data merely describes the labels but does not use input texts, finetuning on it yields a model that performs strongly on multiple text domains for a given label set, even improving over few-shot out-of-domain classification in multiple settings.
 23 Oct 2023
[cs.CL]
arXiv:2305.02239v2
# 1 Introduction
Pretrained language models (PLMs) (Radford et al., 2018; Devlin et al., 2019; Liu et al., 2019; Brown et al., 2020; Raffel et al., 2020) have produced strong results in zero-shot text classification for a range of topic and sentiment tasks, often using a pattern-verbalizer approach (Schick and Schütze, 2021). With this approach, to classify the restaurant review “Overpriced, salty and overrated!”, a pattern like “the restaurant is [MASK]” is appended to the review and verbalizers are chosen for each label (e.g., “good” for positive sentiment and “bad” for negative). The text is classified by the pretrained masked language modeling (MLM) head to choose
† Co-senior authors.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7e9c/7e9cec4d-f117-460d-b38e-4f7e24fe6447.png" style="width: 50%;"></div>
Label
Input
Business
business
finance
Business is the activity of making one’s living
or making money by producing or buying and
selling products. . .
Sports
sports
racing
An athletic activity requiring skill or physical
prowess and often of a competitive nature, as
racing, baseball. . .
<div style="text-align: center;">(a) Topic classification</div>
Label
Input
Very
Negative
awful
It was terrible.
A horrendous experience.
Very
Positive
great
Just fantastic.
Overall, it was outstanding.
Table 1: A few examples of LABELDESC training data for topic and sentiment classification.
the most probable verbalizer for the [MASK] position.1 Although effective, the approach is sensitive to the choice of specific pattern/verbalizer pairs, with subtle changes in the pattern, the verbalizer, or both, often having a large impact on performance (van de Kar et al., 2022; Perez et al., 2021). To alleviate these issues, we propose a simple alternative approach of training on small curated datasets intended to describe the labels for a task. Unlike typical training datasets, which consist of input texts annotated by hand with labels, our data contains only the descriptions of the labels. We refer to this data as LABELDESC data and show a few examples for topic and sentiment classification in Table 1. For topic classification, we include a few terms related to the label (e.g., “finance” for “Business”, “racing” for “Sports”), a definition of
1Please refer to Schick and Schütze (2021) for more details on the pattern-verbalizer approach.
the label from dictionary.com (e.g., “An athletic activity . . . ” for “Sports”), and a sentence from the opening paragraph of the label’s Wikipedia article (e.g., “Business is the activity of ...” for “Business”). For sentiment classification, we simply use related terms that capture the specific sentiment (e.g., “terrible” for “Very Negative”) as well as a few hand-crafted templates (e.g., “It was t.” where t is a related term). Next, we finetune pretrained models using the pattern-verbalizer approach on LABELDESC data and evaluate them for text classification. For topic classification, we use patterns and verbalizers from Schick and Schütze (2022) to train on our LABELDESC examples by finetuning the model as well as the MLM head (see Section 3 for details). We refer to training on LABELDESC data as LABELDESCTRAINING. In experiments, we show that LABELDESCTRAINING consistently improves accuracy (average improvement of 17-19%) over zero-shot classification across multiple topic and sentiment datasets (Table 2). We also show that LABELDESCTRAINING can decrease accuracy variance across patterns compared to zero-shot classification (Table 3), thus being less sensitive to the choice of pattern. We then conduct additional experiments to reveal the value of LABELDESCTRAINING under various circumstances. To study the impact of verbalizer choice, we experiment with uninformative (randomly initialized) and adversarial (intentionally mismatched) verbalizers (Section 4.2.1). While accuracy drops slightly, both settings are still much more accurate than zero-shot classification with its original verbalizers. That is, LABELDESCTRAINING is able to compensate for knowledge-free or even adversarial verbalizer choice. We also compare to finetuning a randomly initialized classifier head without any patterns or verbalizers, again finding accuracy to be higher than zero-shot (Section 4.2.2). Collectively, our results demonstrate that LABELDESCTRAINING leads to strong performance that is less sensitive than zero-shot classification in terms of pattern/verbalizer choice, while also not requiring a pretrained MLM head. Since LABELDESC data focuses entirely on the labels without seeking to capture the input text distribution, we would hope that it would exhibit stable performance across datasets with the same labels. So, we compare LABELDESCTRAINING to the approach of training on a small super-
vised training set from one domain and testing on another (Section 4.2.4). In multiple cases, LABELDESCTRAINING actually attains higher accuracy than few-shot supervised learning tested on out-of-domain test sets, even when hundreds of manually labeled training examples are used (albeit from a different input domain). In summary, this paper shows several benefits of LABELDESCTRAINING. First, once a practitioner identifies a label set of interest for zero-shot classification, it only requires a few minutes to collect the kind of LABELDESC data shown in Table 1, and training on this data improves over zero-shot by 17-19% absolute. Second, LABELDESCTRAINING leads to greater robustness to pattern/verbalizer choice than zero-shot. Third, LABELDESC data are domain independent with regard to the distribution of the inputs; a single LABELDESC training set can be used for any text classification task as long as it contains the same labels. Our experiments show that this independence to input distribution leads to stable accuracy across domains, even attaining higher accuracy than out-of-domain few-shot learning on a few cases.2
# 2 Tasks and LABELDESC Datasets
We evaluate on two types of tasks: topic classification on AGNews, Yahoo Answers, and DBPedia (Zhang et al., 2015) and sentiment classification on the Stanford Sentiment Treebank (SST) (Socher et al., 2013), Yelp Reviews (Zhang et al., 2015), IMDB (Maas et al., 2011), and Amazon Reviews Polarity (Zhang et al., 2015). We consider both binary and 5-way classification for SST and Yelp datasets (denoted as SST-2, SST-5, Yelp-2, and Yelp-5 henceforth) and only binary for IMDB and Amazon (denoted as IMDB and Amz-2 henceforth).3 Below we describe how we construct LABELDESC data for each label set. Dataset statistics as well as all LABELDESC data are in Section A.5 in the Appendix. Topic Classification. Since labels in topic classification represent general concepts, we use both subjective descriptors of the labels (e.g., related terms) and objective sources of information (e.g., dictionary definition and Wikipedia sentences)
2Data and code are available at https://github.com/ lingyugao/LabelDescTraining. 3Our method could be adopted for other tasks like natural language inference (NLI) using templates similar to how we approached sentiment classification. We leave a full exploration to future work.
when selecting LABELDESC data. In particular, we create LABELDESC examples for the label term itself, three related terms, a selected definition from dictionary.com, and the leading sentence from the label’s Wikipedia article. As there are typically multiple dictionary.com definitions for our labels, we select a single definition that best aligns with our understanding of the concept underlying the label. We use the leading Wikipedia sentence because it is typically a brief overview/definition of the concept. Most labels in the Yahoo dataset consist of two keywords (e.g., Society & Culture). For these, we use both label terms, definitions for each, and the leading Wikipedia sentences for each. We did not tune any of these decisions experimentally, so these choices in defining LABELDESC data are almost certainly suboptimal. This suboptimality is especially likely for the “World” label in the AGNews label set. This label reflects international news, but the dictionary definition and Wikipedia article for the term “World” do not capture that sense of the word. Nonetheless, we did not change our procedure for this label because we wanted our results to reflect a real-world implementation of the idea, complete with its limitations for certain labels. The LABELDESC instances we are using do not contain exhaustive information. We could easily extend the lists of related terms for each topic or use WordNet or other semantic knowledge resources (Zhang et al., 2019). However, one of the goals of this research is to demonstrate how simple it is to choose LABELDESC examples to improve zero-shot classification in very little time.
# Sentiment Classification.
Sentiment Classification. We use a slightly different procedure for sentiment classification. For 5-way sentiment, we use the label verbalizer itself and four synonym terms. In addition, we write four simple templates: “It was t.”, “A(n) t experience.”, “Just t.”, and “Overall, it was t.”, where t is the label verbalizer or a synonym. For binary sentiment, we remove the neutral instances, combine the two positive labels (“Very Positive” and “Positive”) into one, and combine the two negative labels (“Very Negative” and “Negative”) into one. This procedure produces a total of 25 examples per label (5 terms + 5 terms × 4 templates) for 5way sentiment and 50 examples per label for binary sentiment. Since these LABELDESC instances are domain-independent, we use the same data for both for 5-way sentiment (Yelp-5 and SST-5) and for bi-
nary sentiment (Yelp-2, SST-2, IMDB-2, Amz-2).
Hyperparameter Tuning. We adhere to the “true” zero-shot setting where hyperparameters cannot be tuned on a development set for the task of interest (Schick and Schütze, 2022). Therefore, we use a separate dataset for hyperparameter tuning - the 20 Newsgroups (20NG, henceforth) (Lang, 1995) - a topic classification dataset with twenty labels. We select only four labels from 20NG for our purposes: talk.religion.misc, rec.autos, sci.med, and talk.politics.guns. We chose these four labels because they are sufficiently distinct that we expect tuning to be informative for other real-world classification datasets; many of the other 20NG labels are highly technical or similar to one other, e.g., the pair comp.sys.ibm.pc.hardware and comp.sys.mac.hardware as well as the pair comp.os.ms-windows.misc and comp.windows.x. We follow the same strategy as for topic classification above when constructing LABELDESC data for 20NG. The selected hyperparameters are used for both topic and sentiment classifications.
# 3 Experimental Settings
The following settings are used in our experiments. Unless stated otherwise, we use the pretrained RoBERTa-base (b) and RoBERTa-large (l) models (Liu et al., 2019) for all experiments since RoBERTa is the predominant choice in related zeroshot and dataless research (Schick and Schütze, 2021; van de Kar et al., 2022; Gera et al., 2022). Additionally, for every dataset, we use the entire available test sets for evaluation.
# Zero-shot Classification Baseline
Zero-shot Classification Baseline. We use the standard “pattern-verbalizer” approach for topic and sentiment classification. The set of verbalizers used can be found in Table 10 in the Appendix. For choosing verbalizers, we follow the choices of Schick and Schütze (2021) for AGNews, Yahoo, Yelp-5, and SST-5. We follow van de Kar et al. (2022) in choosing verbalizers for Yelp-2, SST-2, IMDB, and Amz-2 and we select verbalizers for DBPedia and 20NG ourselves. Each pattern comprises a prompt including a [MASK] symbol placed before or after the text input, and we aim to predict the masked token. For example, a prompt is added after the input x to frame classification as a question answering task, e.g., “x Question: What is the topic of this newsgroup? Answer: [MASK].” We use RoBERTa-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/040f/040f681f-6699-4f05-9fd9-8ac58b666a03.png" style="width: 50%;"></div>
Figure 1: Overview of our proposed method, including the construction of LABELDESC data, the format of the text input, and the target used for both model finetuning and inference during test time. We present text inputs labeled as “Sports” from the topic classification task, and use one of our patterns (see Table 11) here as an illustration. Note that all our LABELDESC datasets are balanced, with each pattern being associated with a unique finetuned model checkpoint.
base/large with its MLM head for zero-shot experiments. Although the model is able to predict any token within its vocabulary, we choose only among the set of verbalizers, which are designed to be semantically coherent with class labels and tokenized into a single token by the model’s tokenizer. For topic classification tasks, we use the PROMPT and Q&A patterns from Schick and Schütze (2022), which amounts to 14 patterns. For AGNews, we use “news/article” in the pattern templates, while for Yahoo we replace this with “question”, and for 20NG we use “newsgroup”. For the sentiment classification tasks, we create new Q&A patterns such as “x Question: What is the sentiment of this text? Answer: [MASK].” and PROMPT patterns such as “x Sentiment: [MASK].” where x is the input text. There are 14 sentiment patterns in total, presented in the Appendix (Section A.2).
LABELDESCTRAINING. We use the same settings as the zero-shot baseline except that we finetune the models on LABELDESC data. We do not use any target task data for tuning or early stopping. Instead, we fix hyperparameter values, including number of training steps, by tuning on 20NG following the process described below. We used LABELDESC data for the four selected 20NG labels as our training data and the original 20NG data (training and test sets) as our dev set, restricted to the four selected labels shown in Section 2. We preprocessed the data by removing headers,
quotes, and footers. We used a batch size of 1 and tuned over a set of five learning rates ({5e-7, 1e-6, 5e-6, 1e-5, 5e-5}). Models were trained for 3500 training steps, evaluating on the dev set after each epoch, i.e., every 24 training steps since it’s the size of LABELDESC dataset for 20NG. Based on tuning accuracies, we chose learning rate 5e-7 and number of training steps 2160 for RoBERTa-base and 1920 for RoBERTa-large. Additionally, we explored variations of parameter freezing, such as freezing certain layers of RoBERTa. The best setting on 20NG was to freeze the lower half of the layers (excluding the embedding layer) during finetuning, so we used this for experiments reported below.4
# 4 Results and Analysis
In this section we first present the results that are obtained via LABELDESCTRAINING and then analyze the benefits of LABELDESC data with a range of additional experiments and analysis.
# 4.1 Results
Table 2 compares standard zero-shot classification and LABELDESCTRAINING. LABELDESCTRAINING has higher accuracy across all topic and sentiment classification datasets, outperforming zero-shot by about 17% on average when using
4Section A.3 in the Appendix provides more details on hyperparameter tuning.
AGNews
Yahoo
DBPedia
Yelp-5
SST-5
Yelp-2
SST-2
Amz-2
IMDB
Avg.
zero-shot
b
62.7
41.5
54.6
38.0
35.6
63.6
62.6
64.0
69.9
54.7
l
68.0
47.7
63.9
38.7
35.0
70.6
63.7
67.5
74.1
58.8
LABELDESCTRAINING
b
77.4
58.8
79.5
43.6
42.0
88.3
84.5
88.6
86.9
72.2
l
79.4
60.8
86.6
51.3
49.2
94.6
91.3
94.1
92.1
77.7
Table 2: Test accuracy (%) comparison between zero-shot classification and LABELDESCTRAINING, b = RoBERTa base, l = RoBERTa-large. For zero-shot, each result is the average over 14 patterns; and for LABELDESCTRAINING each result is the average over 14 patterns and three random seeds per pattern. The “Avg.” column shows the average accuracies across columns.
AGNews
Yahoo
DBPedia
Yelp-5
SST-5
Yelp-2
SST-2
Amz-2
IMDB
zero-shot
b
7.4
7.0
18.9
4.3
4.3
10.7
11.0
10.3
13.2
l
7.8
8.2
9.7
7.8
7.7
15.7
14.3
13.7
17.0
LDT
b
5.0, 5.1, 5.0
1.7, 1.6, 1.6
4.5, 4.5, 4.5
2.0, 2.1, 2.2
1.8, 1.4, 1.5
2.1, 2.8, 2.4
2.5, 2.3, 1.9
1.3, 1.2, 1.4
1.8, 2.3, 1.4
l
5.3, 6.4, 4.6
2.1, 2.0, 2.3
3.2, 2.9, 3.2
2.4, 2.5, 2.4
1.6, 1.2, 1.5
1.1, 2.5, 1.4
1.2, 2.8, 1.6
0.9, 1.9, 0.8
1.1, 1.4, 1.2
Table 3: Standard deviations of test accuracy (%) across 14 patterns for each test dataset. For LABELDESCTRAINING (LDT in the table), three random seeds were used so we show three standard deviations, one per random seed. All standard deviations over patterns are smaller for LDT than the corresponding values for zero-shot.
RoBERTa-base and 19% with RoBERTa-large. The results demonstrate that we can greatly improve the performance of zero-shot models with just a few training examples that provide a richer characterization of the label but still without requiring any textual inputs from the task datasets. Table 3 shows that accuracy variances across patterns using LABELDESCTRAINING are much lower than the zero-shot setting, which is known to be unstable (Perez et al., 2021). Finetuning on LABELDESC data not only improves accuracy, but also mitigates sensitivity to pattern selection.
# Comparisons to the State of the Art. We 
Comparisons to the State of the Art. We compare to state-of-the-art (SOTA) results from the literature in Table 4 (we show results using RoBERTabase to better compare to other methods). For this comparison, we use only a single pattern with LABELDESCTRAINING, since doing so reflects more of a real-world use case than averaging over 14 patterns. We choose a single pattern for each of RoBERTa-base and large by tuning on 20NG as we did for other hyperparameters.5 We use three random seeds and report average accuracies and standard deviations over seeds. Chu et al. (2021a) and Chu et al. (2021b) are dataless classification approaches (Chang et al., 2008) that include single-encoder and dual-encoder methods; the latter include the idea of embedding documents and labels and performing classification via semantic retrieval; we report their non-
5Please refer to A.3 and Table 14 in Appendix for details. We use the same setting for Table 5.
ensemble results in Table 4. Schick and Schütze (2022) use labeled training data (10 or 100 examples, see Table 4) for each task, which differs from the domain-independent LABELDESC examples which are agnostic to the domain of the textual inputs.6 From van de Kar et al. (2022), we include the highest accuracies. The results of LABELDESCTRAINING are comparable to other methods across datasets. For sentiment classification, LABELDESCTRAINING performs better than dataless classification (Chu et al., 2021a) by a large margin for all datasets and is competitive with van de Kar et al. (2022) and Schick and Schütze (2021). Our method is better than that of van de Kar et al. on topic datasets (AGNews, Yahoo, and DBPedia) but not sentiment datasets except for SST-2. van de Kar et al. (2022) search for naturally occurring data in large corpora; texts expressing sentiment are well-represented in corpora, while texts for topics in a fixed label set may be rarer. LABELDESCTRAINING trains on balanced data from a fixed label set, leveraging available knowledge resources to inform about topics. Although van de Kar et al. (2022) do not report 5-way classification results for Yelp or SST, we report results for both datasets (including base and large models) so that future work can compare to our results in this table. We recommend tuning zero-shot and few-shot methods on datasets that
6We only include results with PROMPT and Q&A patterns (14 patterns for topic and 16 for sentiment) from Schick and Schütze (2022), since those are the pattern types we used for LABELDESCTRAINING.
AGNews
Yahoo
DBPedia
Yelp-5
Yelp-2
SST-5
SST-2
Amz-2
IMDB
LABELDESCTRAINING
b
84.6±0.3
59.9±0.3
82.4±1.2
42.0±0.4
84.8±0.6
44.3±0.1
88.2±0.2
89.6±0.4
83.4±0.4
l
85.1±1.0
61.2±0.3
88.5±0.4
52.5±1.2
95.3±0.4
49.4±1.1
91.4±0.8
94.5±0.3
92.9±0.1
Chu et al. (2021a)
b
68.8
57.8
81.9
-
67.3
-
65.0
66.8
-
Chu et al. (2021b)
b
75.1
60.0
88.6
-
-
-
-
-
-
Schick and Schütze (2022)
10
79.5±2.2
58.4±2.7
-
44.3±2.5
-
-
-
-
-
100
87.5±0.8
65.3±1.0
-
54.8±1.5
-
-
-
-
-
van de Kar et al. (2022)
b
79.2
56.1
80.4
-
92.0
-
85.6
92.0
86.7
Table 4: Test accuracy (%) comparison to state-of-the-art methods. 10/100 = # labeled examples used.
AGNews
Yahoo
DBPedia
Yelp-5
Yelp-2
SST-5
SST-2
Amz-2
IMDB
LABELDESCTRAINING
b
84.3±0.1
57.5±0.7
82.0±1.5
41.6±1.2
83.1±0.5
45.3±0.6
86.7±0.6
90.8±0.4
83.1±0.6
l
85.5±0.6
57.5±0.7
88.1±0.6
53.8±1.9
95.4±0.4
51.4±1.3
90.3±0.7
94.2±0.3
94.1±0.2
AGNews
Yahoo
DBPedia
Yelp-5
Yelp-2
SST-5
SST-2
Amz-2
IMDB
LABELDESCTRAINING
b
84.6±0.3
59.9±0.3
82.4±1.2
42.0±0.4
84.8±0.6
44.3±0.1
88.2±0.2
89.6±0.4
83.4±0.4
l
85.1±1.0
61.2±0.3
88.5±0.4
52.5±1.2
95.3±0.4
49.4±1.1
91.4±0.8
94.5±0.3
92.9±0.1
Chu et al. (2021a)
b
68.8
57.8
81.9
-
67.3
-
65.0
66.8
-
Chu et al. (2021b)
b
75.1
60.0
88.6
-
-
-
-
-
-
Schick and Schütze (2022)
10
79.5±2.2
58.4±2.7
-
44.3±2.5
-
-
-
-
-
100
87.5±0.8
65.3±1.0
-
54.8±1.5
-
-
-
-
-
van de Kar et al. (2022)
b
79.2
56.1
80.4
-
92.0
-
85.6
92.0
86.7
AGNews
Yahoo
DBPedia
Yelp-5
Yelp-2
SST-5
SST-2
Amz-2
IMDB
LABELDESCTRAINING
b
84.3±0.1
57.5±0.7
82.0±1.5
41.6±1.2
83.1±0.5
45.3±0.6
86.7±0.6
90.8±0.4
83.1±0.6
l
85.5±0.6
57.5±0.7
88.1±0.6
53.8±1.9
95.4±0.4
51.4±1.3
90.3±0.7
94.2±0.3
94.1±0.2
text-davinci-003 (zero-shot)
-
80.2
58.5
70.1
47.2
92.3
49.3
89.3
93.3
78.9
text-davinci-003 (ICL)
-
83.9
61.1
84.2
57.0
92.9
51.2
92.3
95.1
88.3
e 5: Test accuracy (%) comparison to text-davinci-003 on test set s
are excluded from the final comparison, like 20NG in this paper. Comparisons Involving GPT-3.5. Our method not only works for MLM-style models like RoBERTa, but also for autoregressive models. In Table 5, we show zero-shot and in-context learning (ICL), where we use the entire LABELDESC data for the task as ICL demonstrations, with text-davinci-003 (GPT-3.5; OpenAI, 2022). Due to our restricted budget, we decided to use only 1,000 test instances for each test dataset in GPT-3.5 experiments, while ensuring that the label distribution remains consistent with that of the full test dataset. It is well known that ICL is sensitive to a variety of design choices, including the order of the demonstrations (Fei et al., 2023; Lu et al., 2022). For ICL demonstrations, we included all LABELDESC data for a task to make predictions for each test instance. To avoid the “recency bias” (i.e., the tendency to predict labels that occur towards the end of the prompt; Zhao et al., 2021a), we randomly shuffle the order of demonstrations. We left other parameters untouched. GPT-3.5 with ICL using LABELDESC data outperforms zero-shot GPT-3.5 on all datasets, showing the value of LABELDESC data even if in-domain inputs are unavailable. In comparison to GPT-3.5 flavors, LABELDESCTRAINING (RoBERTa-large) performs better on AGNews, DBPedia, Yelp-2, SST-5, and IMDB, and is competitive across other datasets.
are excluded from the final comparison, like 20NG in this paper.
# 4.2 Analysis and Discussion
One of the primary requirements of the zero-shot approach is the availability of pattern-verbalizer
pairs (Schick and Schütze, 2021, 2022). Here, we study several variations of LABELDESCTRAINING to investigate whether we can simplify or remove components of these pattern-verbalizer pairs. We first experiment with changing verbalizers to gauge the impact of verbalizer choice for LABELDESCTRAINING (Section 4.2.1). Next, we conduct classification experiments that do not use patterns or verbalizers at all (Section 4.2.2). Furthermore, we include one more baseline, i.e., the model finetuned on the 20NG LABELDESC data and patterns to analyze the generalizability (Section 4.2.3). We also report additional experiments in which we measure the multi-domain robustness of LABELDESCTRAINING compared to a standard procedure of training on one domain and testing on an out-of-domain test set (Section 4.2.4). Finally, we take a closer look at label-wise performance to better understand how LABELDESCTRAINING outperforms zero-shot classification (Section 4.2.5).
# 4.2.1 Impact of Verbalizers
In this section we report experiments with LABELDESCTRAINING without meaningful verbalizers and even with adversarially chosen verbalizers. We explore two different verbalizer settings:
• RANDOM: We add c new words, i.e., RANDOM1, RANDOM2, ..., RANDOMc, where c is the number of dataset labels, to the model’s vocabulary and randomly initialize their embeddings. This setting prevents the use of any prior knowledge in the verbalizer embeddings. • MISMATCHED: We shuffle the original mapping
• RANDOM: We add c new words, i.e., RANDOM1, RANDOM2, ..., RANDOMc, where c is the number of dataset labels, to the model’s vocabulary and randomly initialize their embeddings. This setting prevents the use of any prior knowledge in the verbalizer embeddings. • MISMATCHED: We shuffle the original mapping
AGNews
Yahoo
DBPedia
Yelp-5
SST-5
Yelp-2
SST-2
Amz-2
IMDB
Avg.
zero-shot
b
62.7±7.4
41.5±7.0
54.6±18.9
38.0±4.3
35.6±4.3
63.6±10.7
62.6±11.0
64.0±10.3
69.9±13.2
54.7±9.7
l
68.0±7.8
47.7±8.2
63.9±9.7
38.7±7.8
35.0±7.7
70.6±15.7
63.7±14.3
67.5±13.7
74.1±17.0
58.8±11.3
LDT20NG
b
61.8±7.0
49.4±5.2
72.9±7.8
34.6±4.6
36.5±3.7
67.7±10.3
63.4±9.7
67.2±9.6
72.5±10.5
58.4±7.6
l
72.4±6.8
54.4±4.3
71.9±10.8
36.3±5.7
36.6±7.1
63.4±13.0
56.9±8.7
60.9±10.2
67.5±15.2
57.8±9.1
LDT
b
77.4±4.9
58.8±1.6
79.5±4.4
43.6±2.1
42.0±1.6
88.3±2.5
84.5±2.2
88.6±1.4
86.9±1.8
72.2±2.5
l
79.4±5.0
60.8±2.1
86.6±3.0
51.3±2.4
49.2±1.6
94.6±1.8
91.3±2.0
94.1±1.3
92.1±1.2
77.7±2.3
MLMr
b
77.3±4.0
54.3±3.9
81.3±7.3
38.1±3.8
37.0±3.2
78.4±10.0
73.3±7.9
80.0±9.9
73.8±9.6
65.9±6.6
l
75.2±5.0
58.0±3.0
85.4±13.0
46.4±3.3
43.4±2.9
90.8±7.6
84.1±6.8
90.2±7.1
87.4±6.2
73.4±6.1
MLMm
b
73.1±5.6
50.1±5.4
72.6±8.1
36.8±2.8
35.8±2.5
80.1±7.2
75.8±5.0
81.8±6.8
76.7±6.0
64.8±5.5
l
66.4±8.6
44.5±4.9
73.1±7.3
41.9±4.0
38.7±4.2
83.6±6.5
78.1±6.0
85.0±6.0
77.7±6.9
65.4±6.0
classifier
b
72.5±5.5
57.1±0.7
87.7±2.6
40.3±1.3
39.4±2.5
86.9±2.9
79.7±1.1
89.1±0.9
80.6±3.6
70.4±2.3
l
77.8±1.5
50.9±7.3
78.2±1.0
42.4±1.6
35.3±9.2
93.3±0.9
86.6±1.4
93.7±0.5
85.7±2.0
71.5±2.8
Table 6: Test accuracies (%) for several variations of LABELDESCTRAINING. The standard deviations are computed over 14 patterns for zero-shot; 3 random seeds for the classifier (no patterns); and both 14 patterns and 3 random seeds for LABELDESCTRAINING on 20NG, LABELDESCTRAINING, RANDOM, and MISMATCHED (LDT20NG, LDT, MLMr, and MLMm in Table).
of labels to verbalizers, ensuring that each verbalizer maps to a different label than in the original LABELDESCTRAINING setting. Since we are still finetuning the embeddings, finetuning can help the model recover from this mismatched initialization.
The results are shown in Table 6. Since we still use the MLM head for these results, we refer to them as “MLM, RANDOM” and “MLM, MISMATCHED”. While LABELDESCTRAINING performs better than RANDOM, and RANDOM is better than MISMATCHED, both are better than zeroshot on average. These results suggest that LABELDESC data can partially compensate when the quality of the verbalizers is unknown or poor, at least to improve over zero-shot.
# 4.2.2 Classifiers Without Patterns or Verbalizers
Since finetuning on LABELDESC data outperforms zero-shot results with RANDOM verbalizers, we also evaluate its performance without patterns, i.e., using a standard randomly initialized softmax classifier. The input is the original text without any patterns and we use a two-layer classification head on top of the [CLS] token representation of the pretrained models. The bottom two rows of Table 6 show the results. The classifiers are close to that of the MLM/RANDOM setting and still much higher than zero-shot on average, suggesting that it is not necessary to use patterns, verbalizers, or even the pretrained MLM head in order to outperform zero-shot classifiers. If it is difficult to select verbalizers or design patterns for a particular classification task,
using a classifier that has been finetuned on a small LABELDESC dataset may serve as a strong alternative to the pattern-verbalizer approach.
# 4.2.3 Cross-Task Generalizability
We report results on the model finetuned on the 20NG LABELDESC data and patterns, i.e., LABELDESCTRAINING on 20NG (LDT20NG), in Table 6. While the patterns for the reported datasets are different from those used for 20NG, especially for sentiment datasets, they have similar structures (see Section A.2). For RoBERTa-base, LDT20NG often outperforms zero-shot results, except for AGNews and Yelp-5. However, for RoBERTa-large, while LDT20NG outperforms the zero-shot results on all topic classification datasets, it’s worse on sentiment classification except for SST-5.
# 4.2.4 Multi-Domain Evaluation
Since LABELDESC examples are domainindependent, they can be used for multiple datasets that have the same labels. To assess the multidomain performance of LABELDESCTRAINING, we compare it to supervised few-shot learning in which a model is trained on data from one domain and then evaluated on a different domain with the same label set (i.e., training on SST-5 and evaluating on Yelp-5). To create multi-domain test sets for a single topic label set, we keep AGNews as it is and create a new subsampled version of Yahoo as follows: (1) “Politics & Government” and “Society & Culture” texts are assigned the label “World”, (2) “Sports” texts are labeled “Sports”, (3) “Business & Finance” texts are labeled “Business”, and (4) “Science & Mathematics” and “Computers
& Internet” texts are labeled “Sci/Tech”. Other Yahoo texts are removed. We refer to this new version of the Yahoo dataset as YahooAG. For sentiment classification, we choose two dataset pairs that share label sets, i.e., SST-5 and Yelp-5. We do not change anything about the LABELDESCTRAINING configuration for these experiments. We simply evaluate the same model on multiple test sets, reporting average accuracies over patterns. For few-shot setup, we create datasets with 10, 100, and 500 training examples per label. For indomain experiments, train, dev, and test sets are drawn from the same domain/dataset, whereas for out-of-domain experiments, train and dev sets are drawn from one domain and the test set is drawn from another domain. We tune learning rates over the same ranges as mentioned earlier and use batch sizes 1, 2, and 4 for 10, 100, and 500 examples per label, respectively. We train for 15 epochs and select the checkpoint from the best epoch selected by the dev set. The results using RoBERTa-large are shown in Figure 2. For brevity, we only show a subset of results.7 As we would expect, testing on out-ofdomain data leads to accuracy drops but adding more out-of-domain training data reduces this gap. LABELDESCTRAINING, shown as an orange dotted line, outperforms supervised few-shot learning in some cases, such as training on AGNews and testing on YahooAG, even with 500 examples per label (upper-right plot in Figure 2). We see the same trend when the supervised model is trained on Yelp5 and tested on SST-5 (lower-right plot in Figure 2). In 3 out of 4 cases, LABELDESCTRAINING outperforms supervised few-shot out-of-domain learning with 10 examples per label, outperforming 100 in 2 out of 4 cases.
# 4.2.5 Label-wise Investigation
To better understand why LABELDESCTRAINING outperforms zero-shot, we report label-specific F1 scores in Tables 8 and 9. For AGNews, the zeroshot classifiers have low F1 scores for the World label, probably because the verbalizer “World” is much less coherent and less representative of the actual label than others like “Sports.” LABELDESCTRAINING improves F1 on the World label by roughly 20 points, while the improvement for Sports is only about 4 points. Likewise, the F1 scores for “Very Negative”, “Very Positive”, and
7Section A.4 in the Appendix shows additional results.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1808/1808e684-9f8b-4d9d-846d-d492a96e3ad5.png" style="width: 50%;"></div>
<div style="text-align: center;">in-domain out-of-domain LDT</div>
<div style="text-align: center;">Figure 2: Domain transfer results, where the X-axis shows the number of training examples per label.</div>
“Neutral” are very low for the zero-shot models on SST-5, indicating that those labels are being largely ignored. Again, LABELDESCTRAINING shows large improvements in F1 for some of these labels, especially “Very Positive”. These trends are likely due in part to the differences verbalizer probabilities, e.g., “good” and “bad” occur more frequently than “great” and “terrible”. The LABELDESC data is balanced, which helps to mitigate the ignoring of labels, even though the task test sets are not all balanced. Table 7 shows examples that are incorrectly classified by zero-shot models but are correctly classified by the LABELDESCTRAINING models.
# 5 Related Work
One common approach in zero-shot text classification is to transfer knowledge from seen labels (Dauphin et al., 2014), which requires observed labels and a notion of label similarity. Some sources of semantic knowledge used for this purpose include multiple modalities (Lampert et al., 2009), label relationships in knowledge graphs (Wang et al., 2018), and word representations (Song and Roth, 2014; Fei et al., 2022). There are several other approaches to zero-shot classification. To classify documents, Chang et al. (2008) used knowledge-based text representations derived from Wikipedia, and Barak et al. (2009) used both Wikipedia and WordNet. Zhang et al. (2019) combined label descriptions with a label hierarchy and word-to-label paths in ConceptNet, with data augmentation strategies. Yin et al. (2019) used a textual entailment approach with label defi-
text ([headline][text body] for AGNews)
zero-shot
LABELDESCTRAINING
[Homeless families total 100,000][The figure for homeless families in England has
topped 100,000 for the first time.]
Business
World
[Shifting signs in North Korea][Kim Jong Il dials back his personality cult as
protest activities pick up.]
Sports
World
[GM, Daimler Go Green][Team-up will help the companies compete and fill gaps
in both firms’ portfolios.]
Sci/Tech
Business
(U)nrelentingly stupid.
Positive
Very Negative
Still, I’m not quite sure what the point is...
Positive
Negative
This 72-minute film does have some exciting scenes, but it’s a tad slow.
Positive
Neutral
<div style="text-align: center;">text ([headline][text body] for AGNews)</div>
zero-shot
LABELDESCTRAINING
World
61.5±15.1
81.0±4.3
Business
63.6±7.1
74.9±4.7
Sports
88.2±3.9
92.7±4.5
Sci/Tech
55.0±11.4
67.8±9.3
Table 8: AGNews label-wise F1 (RoBERTa-large).
zero-shot
LABELDESCTRAINING
Very Negative
11.2±14.9
25.8±5.7
Negative
37.6±21.2
62.5±2.0
Neutral
1.2±2.9
10.8±5.5
Positive
46.0±5.8
48.2±4.9
Very Positive
12.1±15.0
58.0±4.0
Table 9: SST-5 label-wise F1 (RoBERTa-large).
nitions from WordNet. Another approach that has gained popularity is self-training given label names and mining an unlabeled dataset (Meng et al., 2020; Gera et al., 2022). van de Kar et al. (2022) extend the mining-based approach by selecting unsupervised examples (via patterns) for training. Basile et al. (2022) select label descriptions by aggregation. Meng et al. (2022) use language models to generate new training examples. On the contrary, we train on a small set of domain-independent label descriptions. Our setup is influenced by Schick and Schütze (2021, 2022), although, instead of finetuning on training examples, we only use our LABELDESC data. Autoregressive language models have also been used for zero-shot text classification; we report zero-shot and ICL results with LABELDESC data using GPT-3.5 (OpenAI, 2022). Zhao et al. (2021b) found it beneficial to “calibrate” such models for this setting; this idea is not immediately applicable here due to our use of encoder-only models like RoBERTa. Calibration could be extended to encoder-only models, which we plan to explore in future work. Our work is closely related to data-
less classification (Chang et al., 2008) which involves building classifiers by designing or learning a generic function that scores the compatibility of a document and label defined in natural language. We compared empirically to the dataless classification approaches of Chu et al. (2021a) and Chu et al. (2021b) who used pretrained models, naturally annotated data like that from Wikipedia categories, and unsupervised clustering techniques. There is a wealth of prior work in semi-supervised text classification (Nigam et al., 2000; Xie et al., 2020; Howard and Ruder, 2018). There is also related work on generating label names (Schick et al., 2020) or label descriptions (Chai et al., 2020; Sun et al., 2019) but for supervised text classification.
# 6 Conclusions
We presented LABELDESCTRAINING, a method for improving the accuracy of zero-shot classification by using small, curated datasets that simply describe the labels for a task in natural language. Our method is 17-19% more accurate than zero-shot on average across a range of datasets. LABELDESCTRAINING is also more robust to the choices required for zero-shot classification, such as patterns and verbalizers. Furthermore, LABELDESC data is domain agnostic and therefore can used for any text classification task as long as it contains the same set of labels. LABELDESCTRAINING can even outperform a supervised approach that uses training data from a different domain. One future direction would be to apply the idea to structured prediction, NLI, and natural language generation tasks. Another would be to investigate ways to reduce the dependence of pretrained models on patterns and verbalizers, such as directly calibrating the marginal probabilities of verbalizers with the goal of minimizing biases of pretrained models.
# 7 Limitations
We focus on a simple approach of curating small finetuning datasets that describe the labels for text classification tasks. Although this is beneficial when the task is specific, especially when the data is difficult to obtain, the data curation process is intrinsically intuitive and relies on the practitioner’s understanding of the labels and usage situation. Moreover, since a pretrained model is necessary for this approach, a few curated examples may mitigate, but cannot detect or eliminate, potential biases of the pretrained model. If the labels of a certain classification task are dissimilar from the examples the model was trained on, and the model lacks the knowledge to differentiate among them, it may lead to unsatisfying performance even after finetuning on a few examples of label descriptions.
# 8 Ethics Statement
We use pretrained models for text classification, and curate data with the assistance of data sources such as Wikipedia and dictionary definitions. The large pretrained models are trained on a massive amount of data and have been shown to have issues with bias; however, this is a common challenge when working with pretrained models and would benefit from advances made by the community on this front. While both dictionary.com definitions and Wikipedia are aimed at providing accurate and neutral information for a word/concept, they can be affected by the biases and limitations of their editors, especially for Wikipedia, which is an opensource encyclopedia. Our method is not reliant on specific dictionaries or encyclopedias; others could be used. We chose these resources for simplicity as they are highly accessible and widely used. Since our LABELDESC data is very small in size, we manually examined the data as we selected it for any potential biases or other issues. Finally, we use standard topic and sentiment datasets for evaluation, which are used in a great deal of prior work.
# References
Libby Barak, Ido Dagan, and Eyal Shnarch. 2009. Text categorization from category name via lexical reference. In Proceedings of Human Language Technologies: The 2009 Annual Conference of the North American Chapter of the Association for Computational Linguistics, Companion Volume: Short Pa-
pers, pages 33–36, Boulder, Colorado. Association for Computational Linguistics.
Angelo Basile, Marc Franco-Salvador, and Paolo Rosso. 2022. Unsupervised ranking and aggregation of label descriptions for zero-shot classifiers. In Natural Language Processing and Information Systems - 27th International Conference on Applications of Natural Language to Information Systems, NLDB 2022, Valencia, Spain, June 15-17, 2022, Proceedings, volume 13286 of Lecture Notes in Computer Science, pages 119–126. Springer.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.
Duo Chai, Wei Wu, Qinghong Han, Fei Wu, and Jiwei Li. 2020. Description based text classification with reinforcement learning. In International Conference on Machine Learning, pages 1371–1382. PMLR.
Ming-Wei Chang, Lev-Arie Ratinov, Dan Roth, and Vivek Srikumar. 2008. Importance of semantic representation: Dataless classification. In Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence, AAAI 2008, Chicago, Illinois, USA, July 13-17, 2008, pages 830–835. AAAI Press.
acob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.
Yu Fei, Yifan Hou, Zeming Chen, and Antoine Bosselut. 2023. Mitigating label biases for in-context learning. arXiv preprint arXiv:2305.19148.
Yu Fei, Ping Nie, Zhao Meng, Roger Wattenhofer, and Mrinmaya Sachan. 2022. Beyond prompting: Making pre-trained language models better zeroshot learners by clustering representations. CoRR, abs/2210.16637. Ariel Gera, Alon Halfon, Eyal Shnarch, Yotam Perlitz, Liat Ein-Dor, and Noam Slonim. 2022. Zeroshot text classification with self-training. CoRR, abs/2210.17541. Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146. Christoph H. Lampert, Hannes Nickisch, and Stefan Harmeling. 2009. Learning to detect unseen object classes by between-class attribute transfer. In 2009 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2009), 20-25 June 2009, Miami, Florida, USA, pages 951–958. IEEE Computer Society. Ken Lang. 1995. Newsweeder: Learning to filter netnews. In Machine Learning, Proceedings of the Twelfth International Conference on Machine Learning, Tahoe City, California, USA, July 9-12, 1995, pages 331–339. Morgan Kaufmann. Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692. Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland. Association for Computational Linguistics. Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA. Association for Computational Linguistics. Yu Meng, Jiaxin Huang, Yu Zhang, and Jiawei Han. 2022. Generating training data with language models: Towards zero-shot language understanding. arXiv preprint arXiv:2202.04538. Yu Meng, Yunyi Zhang, Jiaxin Huang, Chenyan Xiong, Heng Ji, Chao Zhang, and Jiawei Han. 2020. Text classification using label names only: A language model self-training approach. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9006–9017, Online. Association for Computational Linguistics.
Yu Meng, Jiaxin Huang, Yu Zhang, and Jiawei Han. 2022. Generating training data with language models: Towards zero-shot language understanding. arXiv preprint arXiv:2202.04538.
Yu Meng, Yunyi Zhang, Jiaxin Huang, Chenyan Xiong, Heng Ji, Chao Zhang, and Jiawei Han. 2020. Text classification using label names only: A language model self-training approach. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9006–9017, Online. Association for Computational Linguistics.
Kamal Nigam, Andrew Kachites McCallum, Sebastian Thrun, and Tom Mitchell. 2000. Text classification from labeled and unlabeled documents using em. Machine learning, 39(2):103–134.
# OpenAI. 2022. Openai api [text-davinci-003]. https: //api.openai.com/v1/completions.
Ethan Perez, Douwe Kiela, and Kyunghyun Cho. 2021. True few-shot learning with language models. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 11054–11070.
Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21:1– 67. Timo Schick, Helmut Schmid, and Hinrich Schütze. 2020. Automatically identifying words that can serve as labels for few-shot text classification. arXiv preprint arXiv:2010.13641. Timo Schick and Hinrich Schütze. 2021. Exploiting cloze-questions for few-shot text classification and natural language inference. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 255–269, Online. Association for Computational Linguistics. Timo Schick and Hinrich Schütze. 2022. True few-shot learning with Prompts—A real-world perspective. Transactions of the Association for Computational Linguistics, 10:716–731. Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA. Association for Computational Linguistics. Yangqiu Song and Dan Roth. 2014. On dataless hierarchical text classification. In Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence, July 27 -31, 2014, Québec City, Québec, Canada, pages 1579–1585. AAAI Press. Chi Sun, Luyao Huang, and Xipeng Qiu. 2019. Utilizing bert for aspect-based sentiment analysis via constructing auxiliary sentence. arXiv preprint arXiv:1903.09588.
Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21:1– 67.
Timo Schick, Helmut Schmid, and Hinrich Schütze. 2020. Automatically identifying words that can serve as labels for few-shot text classification. arXiv preprint arXiv:2010.13641.
Chi Sun, Luyao Huang, and Xipeng Qiu. 2019. Utilizing bert for aspect-based sentiment analysis via constructing auxiliary sentence. arXiv preprint arXiv:1903.09588.
Chi Sun, Luyao Huang, and Xipeng Qiu. 2019. Utilizing bert for aspect-based sentiment analysis via constructing auxiliary sentence. arXiv preprint arXiv:1903.09588.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5824/5824995d-a3d7-4924-82d6-d941b92925ee.png" style="width: 50%;"></div>
Mozes van de Kar, Mengzhou Xia, Danqi Chen, and Mikel Artetxe. 2022. Don’t prompt, search! miningbased zero-shot learning with language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 7508–7520, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. Xiaolong Wang, Yufei Ye, and Abhinav Gupta. 2018. Zero-shot recognition via semantic embeddings and knowledge graphs. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages 6857–6866. Computer Vision Foundation / IEEE Computer Society. Qizhe Xie, Zihang Dai, Eduard Hovy, Thang Luong, and Quoc Le. 2020. Unsupervised data augmentation for consistency training. Advances in Neural Information Processing Systems, 33:6256–6268.
Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021a. Calibrate before use: Improving few-shot performance of language models. In International Conference on Machine Learning, pages 12697–12706. PMLR.
Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021b. Calibrate before use: Improving few-shot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 12697–12706. PMLR.
# A Appendix
# A.1 Verbalizers
Dataset
Verbalizers
20NG
talk.religion.misc �→religion, rec.autos
�→automobile,
sci.med �→medicine,
talk.politics.guns �→gun
AGNews
World �→World, Sports �→Sports, Busi-
ness �→Business, Sci/Tech �→Tech
Yahoo
Society & Culture �→Society, Science &
Mathematics �→Science, Health �→Health,
Education & Reference �→Education, Com-
puters & Internet �→Computer, Sports �→
Sports, Business & Finance �→Business,
Entertainment & Music �→Entertainment,
Family & Relationships �→Relationship,
Politics & Government �→Politics
DBPedia
Company �→company, Educational institu-
tion �→school, Artist �→artist, Athlete �→
sports, Office holder �→politics, Mean of
transportation �→transportation, Building
�→building, Natural place �→natural, Vil-
lage �→village, Animal �→animal, Plant
�→plant, Album �→album, Film �→film,
Written work �→book
Yelp-5
Very Negative �→terrible, Negative �→bad,
Neutral �→okay, Positive �→good, Very
Positive �→great
SST-5
Yelp-2
Negative �→awful, Positive �→great
SST-2
IMDB
Amz-2
<div style="text-align: center;">Table 10: Verbalizers selected for each dataset.</div>
# A.2 Patterns for MLM A.2.1 Topic Classification
We use the patterns shown in Table 11 for AGNews and DBPedia, and replace “news/article” by "question" for Yahoo Question, which follows Schick and Schütze (2022)’s practice. We use "newsgroup" instead of "question" for 20NG.
# A.2.2 Sentiment Classification
Our sentiment classification datasets (Yelp-2/5, SST-2/5, Amz-2, and IMDB) share the same patterns listed in Table 12.
# A.3 Hyperparameters and Best Pattern
We selected training batch size as 1 for our experiments on LABELDESC data. After fine-tuning on 20NG, the hyperparameters are selected as shown in Table 13. With the selected hyperparameters, we further examine the dev accuracy on 20NG for all prompt patterns and select the tuned pattern that
type
id
patterns
Q&A
1
x Question: What is the topic of this
article? Answer: [MASK].
2
x Question: What is the category of this
article? Answer: [MASK].
3
x Question: What is the topic of this
article? Answer: [MASK]
4
x Question: What is the category of this
article? Answer: [MASK]
PROMPT 1
x Category: [MASK].
2
x Class: [MASK].
3
x Topic: [MASK].
4
x Theme: [MASK].
5
x Category: [MASK]
6
x Class: [MASK]
7
x Topic: [MASK]
8
x Theme: [MASK]
9
[MASK] News: x
10
[MASK] NEWS: x
Table 11: Patterns for AGNews, where x refers to the given text.
Table 11: Patterns for AGNews, where x refers to the given text.
has the highest dev accuracy. The tuned patterns are listed in Table 14. To our knowledge, this method works well when we adapt to other datasets. However, we also observe that there are fluctuations in the dev accuracy curve for 20NG during training, and we select the training steps in the middle of the flatter part of curves rather than the peak point for robustness. We suggest changing training steps or increasing batch size if this method doesn’t work well. The tuned pattern is not necessarily the best pattern after adapting to other datasets, sometimes even a little lower than the average results over all 14 patterns.
# A.4 Domain Transfer
All results on RoBERTa-base/large are shown in Figure 3.
# All results on RoBERTa-base/large are shown in Figure 3.
A.5 LABELDESC Data
The statistics of LABELDESC data are shown in Table 15. We use the same set of LABELDESC data for AGNews and YahooAG, Yelp-5 and SST5, Yelp-2 and SST-2, respectively. The data is listed in Table 16 - Table 21. Each term/sentence that is separated by “|” in tables is an independent LABELDESC example during training. For brevity, we list all hand-crafted templates instead of listing all data for sentiment classification.
type
id
patterns
Q&A
1
x Question: What is the sentiment of
this text? Answer: [MASK].
2
x Question: What is the writer’s opinion
in this text? Answer: [MASK].
3
x Question: What is the sentiment of
this text? Answer: [MASK]
4
x Question: What is the writer’s opinion
in this text? Answer: [MASK]
PROMPT 1
x Opinion: [MASK].
2
x Feeling: [MASK].
3
x Sentiment: [MASK].
4
x Summary: [MASK].
5
x Opinion: [MASK]
6
x Feeling: [MASK]
7
x Sentiment: [MASK]
8
x Summary: [MASK]
9
[MASK] Sentiment: x
10
[MASK] SENTIMENT: x
Table 12: Patterns for sentiment classification, where x refers to the given text.
lr
steps
MLM
LDT
base
5e-7
2160
large
5e-7
1920
MISMATCHED
base
5e-5
2160
large
5e-6
3000
RANDOM
base
5e-5
2160
large
5e-6
3240
classifier
base
1e-5
1920
large
1e-6
2280
Table 13: Hyperparameters (learning rate, training steps) selected by tuning on 20NG with RoBERTa.
# A.6 Dataset Preprocessing
For 20NG, we remove headers, quotes, and footers. For AGNews, we concatenate the headlines and the text body of the news articles. For Yahoo dataset, we concatenate the title, the question, and the top answer to it. And for IMDB and Amazon Reviews Polarity datasets, we concatenate the title and the content.
# A.7 Label-wise Metrics
We list label-wise precision, recall, and F1 scores for part of our datasets in Table 22 - 29.
pattern
id
MLM
LDT
base
prompt
9
large
prompt
7
MISMATCHED
base
qa
3
large
qa
1
RANDOM
base
qa
3
large
prompt
6
<div style="text-align: center;">Table 14: Tuned pattern and pattern id for each model</div>
dataset
#label
LD
dev
test
20NG
4
24
3389
-
AGNews
4
24
2,000
7,600
YahooAG
3,000
36,000
Yahoo
10
60
-
60,000
DBPedia
14
84
-
70,000
Yelp-5
5
125
2,500
50,000
SST-5
1,101
2,210
Yelp-2
2
100
2,000
38,000
SST-2
872
1,821
Amz-2
-
400,000
IMDB
-
25,000
Table 15: Statistics of datasets we used, with ’#’ denoting the number of labels, LD refers to LABELDESC data.
Label
Type
Training Data
talk.
religion.
misc
terms
religion | Christian | Buddhist | Jewish
Wiki.
Religion is usually defined as a social-cultural system
of designated behaviors and practices, morals, beliefs,
worldviews, texts, sanctified places, prophecies, ethics,
or organizations, that generally relates humanity to super-
natural, transcendental, and spiritual elements; however,
there is no scholarly consensus over what precisely con-
stitutes a religion.
dict.
a set of beliefs concerning the cause, nature, and purpose
of the universe, especially when considered as the creation
of a superhuman agency or agencies, usually involving
devotional and ritual observances, and often containing a
moral code governing the conduct of human affairs.
rec.autos
terms
automobile | truck | car | vehicle
Wiki.
A car (or automobile) is a wheeled motor vehicle that is
used for transportation.
dict.
a passenger vehicle designed for operation on ordinary
roads and typically having four wheels and a gasoline or
diesel internal-combustion engine.
sci.med
terms
medicine | hospital | symptom | cure
Wiki.
Medicine is the science and practice of caring for a patient,
managing the diagnosis, prognosis, prevention, treatment,
palliation of their injury or disease, and promoting their
health.
dict.
any substance or substances used in treating disease or
illness; medicament; remedy.
talk.
politics.
guns
terms
gun | firearm | weapon | handgun
Wiki.
A gun is a ranged weapon designed to use a shooting tube
(gun barrel) to launch projectiles.
dict.
a weapon consisting of a metal tube, with mechanical
attachments, from which projectiles are shot by the force
of an explosive; a piece of ordnance.
Table 16: LABELDESC data for 20NG. “Wiki.” and “dict.” refers to the data source, i.e., Wikipedia or dictionary definition.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/47a9/47a9ce52-c6b6-4b47-b7b0-be346c0a8595.png" style="width: 50%;"></div>
<div style="text-align: center;">in-domain out-of-domain LDT</div>
Figure 3: Domain transfer results, where X-axis depicts the number of training examples per label. “base/large” in parenthesis denotes RoBERTa-base/large.
Label
Type
Training Data
World
terms
world | country | international | politics
Wiki.
In its most general sense, the term “world” refers to the
totality of entities, to the whole of reality or to everything
that is.
dict.
humankind; the human race; humanity
Sports
terms
sport | sports | racing | baseball
Wiki.
Sport pertains to any form of competitive physical activity
or game that aims to use, maintain or improve physical
ability and skills while providing enjoyment to partici-
pants and, in some cases, entertainment to spectators.
dict.
an athletic activity requiring skill or physical prowess and
often of a competitive nature, as racing, baseball, tennis,
golf, bowling, wrestling, boxing, hunting, fishing, etc.
Business
terms
business | finance | money | trade
Wiki.
Business is the activity of making one’s living or making
money by producing or buying and selling products (such
as goods and services).
dict.
the purchase and sale of goods in an attempt to make a
profit.
Sci/Tech
terms
technology | science | computer | biology
Wiki.
Technology is the continually developing result of ac-
cumulated knowledge and application in all techniques,
skills, methods, and processes used in industrial produc-
tion and scientific research.
dict.
the branch of knowledge that deals with the creation and
use of technical means and their interrelation with life,
society, and the environment, drawing upon such subjects
as industrial arts, engineering, applied science, and pure
science.
Table 17:
LABELDESC data for AGNews (and
YahooAG).
Label
Type
Training Data
Very
Negative
terms
awful | terrible | horrendous | horrible | dreadful
sent.
It was t. | A(n) t experience. | Just t. | Overall, it was t.
Negative
terms
bad | unpleasant | unsatisfying | lousy | subpar
sent.
It was t. | A(n) t experience. | Just t. | Overall, it was t.
Neutral
terms
okay | mediocre | decent | average | alright
sent.
It was t. | A(n) t experience. | Just t. | Overall, it was t.
Positive
terms
good | nice | fine | pleasant | neat
sent.
It was t. | A(n) t experience. | Just t. | Overall, it was t.
Very
Positive
terms
great | amazing | excellent | fantastic | outstanding
sent.
It was t. | A(n) t experience. | Just t. | Overall, it was t.
Table 18: LABELDESC data for Yelp-5 and SST-5. “Sent.” and “t” refer to hand-crafted sentence templates and terms, respectively.
Label
Type
Training Data
Negative
terms
awful | terrible | horrendous | horrible | dreadful | bad |
unpleasant | unsatisfying | lousy | subpar
sent.
It was t. | A(n) t experience. | Just t. | Overall, it was t.
Positive
terms
good | nice | fine | pleasant | neat | great | amazing | excel-
lent | fantastic | outstanding
sent.
It was t. | A(n) t experience. | Just t. | Overall, it was t.
Table 19: LABELDESC data for Yelp-2, SST-2, Amz-2 and IMDB.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8e4f/8e4f96f6-d1e0-4d7c-9635-3791c0c66c51.png" style="width: 50%;"></div>
Label
Type
Training Data
Society
&
Culture
terms
society | culture
Wiki.
A society is a group of individuals involved in persistent social interaction, or a large social group sharing the same
spatial or social territory, typically subject to the same political authority and dominant cultural expectations. | Culture
is an umbrella term which encompasses the social behavior, institutions, and norms found in human societies, as well
as the knowledge, beliefs, arts, laws, customs, capabilities, and habits of the individuals in these groups.
dict.
an organized group of persons associated together for religious, benevolent, cultural, scientific, political, patriotic, or
other purposes. | the behaviors and beliefs characteristic of a particular group of people, as a social, ethnic, professional,
or age group (usually used in combination)
Science
&
Mathematics
terms
science | mathematics
Wiki.
Science is a systematic endeavor that builds and organizes knowledge in the form of testable explanations and
predictions about the universe. | Mathematics is an area of knowledge that includes such topics as numbers, formulas
and related structures, shapes and the spaces in which they are contained, and quantities and their changes.
dict.
a branch of knowledge or study dealing with a body of facts or truths systematically arranged and showing the operation
of general laws | the systematic treatment of magnitude, relationships between figures and forms, and relations between
quantities expressed symbolically.
Health
terms
health | fitness | medical | diet
Wiki.
Health, according to the World Health Organization, is “a state of complete physical, mental and social well-being and
not merely the absence of disease and infirmity”.
dict.
the general condition of the body or mind with reference to soundness and vigor
Education
&
Reference
terms
education | reference
Wiki.
Education is a purposeful activity directed at achieving certain aims, such as transmitting knowledge or fostering skills
and character traits. | Reference is a relationship between objects in which one object designates, or acts as a means by
which to connect to or link to, another object.
dict.
the act or process of imparting or acquiring general knowledge, developing the powers of reasoning and judgment,
and generally of preparing oneself or others intellectually for mature life. | a book or other source of useful facts or
information, such as an encyclopedia, dictionary, etc.
Computers
&
Internet
terms
computer | internet
Wiki.
A computer is a digital electronic machine that can be programmed to carry out sequences of arithmetic or logical
operations (computation) automatically. | The Internet (or internet) is the global system of interconnected computer
networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices.
dict.
a programmable electronic device designed to accept data, perform prescribed mathematical and logical operations
at high speed, and display the results of these operations. Mainframes, desktop and laptop computers, tablets, and
smartphones are some of the different types of computers. | Usually the internet (except when used before a noun). a
vast computer network linking smaller computer networks worldwide. The internet includes commercial, educational,
governmental, and other networks, all of which use the same set of communications protocols
Sports
terms
sport | sports | racing | baseball
Wiki.
Sport pertains to any form of competitive physical activity or game that aims to use, maintain or improve physical
ability and skills while providing enjoyment to participants and, in some cases, entertainment to spectators.
dict.
an athletic activity requiring skill or physical prowess and often of a competitive nature, as racing, baseball, tennis, golf,
bowling, wrestling, boxing, hunting, fishing, etc.
Business
&
Finance
terms
business | finance
Wiki.
Business is the activity of making one’s living or making money by producing or buying and selling products (such as
goods and services). | Finance is the study and discipline of money, currency and capital assets.
dict.
the purchase and sale of goods in an attempt to make a profit. | the management of revenues; the conduct or transaction
of money matters generally, especially those affecting the public, as in the fields of banking and investment.
Entertainment
&
Music
terms
entertainment | music
Wiki.
Entertainment is a form of activity that holds the attention and interest of an audience or gives pleasure and delight. |
Music is generally defined as the art of arranging sound to create some combination of form, harmony, melody, rhythm
or otherwise expressive content.
dict.
the act of entertaining; agreeable occupation for the mind; diversion; amusement | an art of sound in time that expresses
ideas and emotions in significant forms through the elements of rhythm, melody, harmony, and color.
Family
&
Relationships
terms
family | relationship
Wiki.
Family is a group of people related either by consanguinity (by recognized birth) or affinity (by marriage or other
relationship). | The concept of interpersonal relationship involves social associations, connections, or affiliations
between two or more people.
dict.
a basic social unit consisting of parents and their children, considered as a group, whether dwelling together or not; a
social unit consisting of one or more adults together with the children they care for. | an emotional or other connection
between people
Politics
&
Government
terms
politics | government
Wiki.
Politics is the set of activities that are associated with making decisions in groups, or other forms of power relations
among individuals, such as the distribution of resources or status. | A government is the system or group of people
governing an organized community, generally a state.
dict.
the science or art of political government. | the political direction and control exercised over the actions of the members,
citizens, or inhabitants of communities, societies, and states; direction of the affairs of a state, community, etc.; political
administration
Table 20: LABELDESC data for Yahoo Answers.
<div style="text-align: center;">Table 20: LABELDESC data for Yahoo Answers.</div>
Label
Type
Training Data
Company
terms
company | firm | corporation | business
Wiki.
A company, abbreviated as co., is a legal entity representing an association of people, whether natural, legal or a
mixture of both, with a specific objective.
dict.
a number of persons united or incorporated for joint action, especially for business
Educational
Institution
terms
educational institution | university | college | school
Wiki.
An educational institution is a place where people of different ages gain an education, including preschools, childcare,
primary-elementary schools, secondary-high schools, and universities.
dict.
an institution for instruction in a particular skill or field.
Artist
terms
artist | writer | actor | singer
Wiki.
An artist is a person engaged in an activity related to creating art, practicing the arts, or demonstrating an art.
dict.
a person who produces works in any of the arts that are primarily subject to aesthetic criteria.
Athlete
terms
athlete | sports | footballer | weightlifter
Wiki.
An athlete (also sportsman or sportswoman) is a person who competes in one or more sports that involve physical
strength, speed, or endurance.
dict.
a person trained or gifted in exercises or contests involving physical agility, stamina, or strength; a participant in a sport,
exercise, or game requiring physical skill.
Office
Holder
terms
office-holder | politics | mayor | president
Wiki.
A person who’s been appointed to a position by a company or organisation but doesn’t have a contract or receive regular
payment may be an office-holder.
dict.
a person filling a governmental position; public official.
Mean
of
Transportation
terms
mean of transportation | car | bus | train
Wiki.
Transport (in British English), or transportation (in American English), is the intentional movement of humans, animals,
and goods from one location to another.
dict.
a means of transporting or conveying, as a truck or bus.
Building
terms
building | apartment | skyscraper | tower
Wiki.
A building or edifice, is an enclosed structure with a roof and walls standing more or less permanently in one place,
such as a house or factory (although there’s also portable buildings).
dict.
a relatively permanent enclosed construction over a plot of land, having a roof and usually windows and often more
than one level, used for any of a wide variety of activities, as living, entertaining, or manufacturing.
Natural
Place
terms
natural place | forest | mountain | river
Wiki.
The natural environment or natural world encompasses all living and non-living things occurring naturally, meaning in
this case not artificial.
dict.
existing in or formed by nature (opposed to artificial)
Village
terms
village | town | countryside | rural
Wiki.
A village is a clustered human settlement or community, larger than a hamlet but smaller than a town (although the
word is often used to describe both hamlets and smaller towns), with a population typically ranging from a few hundred
to a few thousand.
dict.
a small community or group of houses in a rural area, larger than a hamlet and usually smaller than a town, and
sometimes (as in parts of the U.S.) incorporated as a municipality.
Animal
terms
animal | insect | bird | fish
Wiki.
Animals are multicellular, eukaryotic organisms in the biological kingdom Animalia.
dict.
any member of the kingdom Animalia, comprising multicellular organisms that have a well-defined shape and usually
limited growth, can move voluntarily, actively acquire food and digest it internally, and have sensory and nervous
systems that allow them to respond rapidly to stimuli: some classification schemes also include protozoa and certain
other single-celled eukaryotes that have motility and animallike nutritional modes.
Plant
terms
plant | flower | tree | grass
Wiki.
Plants are predominantly photosynthetic eukaryotes, forming the kingdom Plantae.
dict.
Botany. any member of the kingdom Plantae, comprising multicellular organisms that typically produce their own
food from inorganic matter by the process of photosynthesis and that have more or less rigid cell walls containing
cellulose, including vascular plants, mosses, liverworts, and hornworts: some classification schemes may include
fungi, algae, bacteria, and certain single-celled eukaryotes that have plantlike qualities, as rigid cell walls or the use of
photosynthesis.
Album
terms
album | soundtrack | mixtape | CD
Wiki.
An album is a collection of audio recordings issued on compact disc (CD), vinyl, audio tape, or another medium such
as digital distribution.
dict.
a record or set of records containing several musical selections, a complete play or opera, etc.
Film
terms
film | movie | comedy | drama
Wiki.
A film – also called a movie, motion picture, moving picture, picture, photoplay or (slang) flick – is a work of visual art
that simulates experiences and otherwise communicates ideas, stories, perceptions, feelings, beauty, or atmosphere
through the use of moving images.
dict.
a sequence of consecutive still images recorded in a series to be viewed on a screen in such rapid succession as to give
the illusion of natural movement; motion picture.
Written
Work
terms
written work | novel | newspaper | book
Wiki.
A book is a medium for recording information in the form of writing or images, typically composed of many pages
(made of papyrus, parchment, vellum, or paper) bound together and protected by a cover.
dict.
a handwritten or printed work of fiction or nonfiction, usually on sheets of paper fastened or bound together within
covers.
<div style="text-align: center;">Table 21: LABELDESC data for DBPedia.</div>
Table 21: LABELDESC data for DBPedia.
dataset
RoBERTa
label
precision(%)
recall(%)
F1(%)
zero-shot
LDT
zero-shot
LDT
zero-shot
LDT
AGNews
base
World
58.7±12.8
80.2±7.3
29.1±27.8
62.0±18.2
33.7±21.2
68.0±12.0
Business
60.6±8.1
71.0±6.3
66.9±14.0
77.0±4.6
63.0±9.7
73.7±3.9
Sports
72.9±14.7
94.1±1.5
92.5±9.6
94.4±6.2
80.1±8.7
94.1±3.3
Sci/Tech
65.3±15.8
69.9±9.2
62.4±14.8
76.4±6.0
60.6±8.1
72.4±4.3
large
World
81.6±10.0
78.8±6.3
53.1±21.3
84.1±6.8
61.5±15.1
81.0±4.3
Business
53.1±13.7
67.4±9.9
84.6±9.2
86.4±5.8
63.6±7.1
74.9±4.7
Sports
86.8±