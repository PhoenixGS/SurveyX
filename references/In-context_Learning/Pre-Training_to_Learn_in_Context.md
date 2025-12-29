# Pre-Training to Learn in Context
Yuxian Gu1,2,∗, Li Dong2, Furu Wei2, Minlie Huang1,†
1The CoAI Group, DCST, Institute for Artificial Intelligence, State Key Lab of Intelligent Technology and Systems, Beijing National Research Center for Information Science and Technology, Tsinghua University, Beijing 100084, China 2 Microsoft Research guyx21@mails.tsinghua.edu.cn, {lidong1,fuwei}@microsoft.com aihuang@tsinghua.edu.cn
# Abstract
In-context learning, where pre-trained language models learn to perform tasks from task examples and instructions in their contexts, has attracted much attention in the NLP community. However, the ability of in-context learning is not fully exploited because language models are not explicitly trained to learn in context. To this end, we propose PICL (Pre-training for In-Context Learning), a framework to enhance the language models’ in-context learning ability by pre-training the model on a large collection of “intrinsic tasks” in the general plain-text corpus using the simple language modeling objective. PICL encourages the model to infer and perform tasks by conditioning on the contexts while maintaining task generalization of pre-trained models. We evaluate the in-context learning performance of the model trained with PICL on seven widely-used text classification datasets and the SUPERNATURALINSTRCTIONS benchmark, which contains 100+ NLP tasks formulated to text generation. Our experiments show that PICL is more effective and task-generalizable than a range of baselines, outperforming larger language models with nearly 4x parameters. The code is publicly available at https://github. com/thu-coai/PICL.
arXiv:2305.09137v1
# 1 Introduction
Pre-trained language models (PLMs; Han et al., 2021; Qiu et al., 2020) have shown strong abilities of learning and performing unseen tasks conditioning on several task examples or instructions in its context, which is called in-context learning (ICL; Brown et al., 2020). Compared to conventional fine-tuning methods, ICL adapts PLMs to downstream tasks only through inference, without parameter updates, which is computationally cheaper in practice and is closer to general AI.
However, PLMs trained on massive corpora to predict the next word given previous words are not explicitly taught to learn in the context. This makes ICL a surprising emergent ability but also indicates that the ICL ability of PLMs is not fully exploited. Garg et al. (2022) has shown that by directly training to do ICL in a meta-learning paradigm, models show strong performance on learning simple function classes in the context. In practical NLP scenarios, previous works (Min et al., 2022b; Chen et al., 2022b) also enhance the ICL performance by metafine-tuning PLMs on a large collection of downstream tasks and evaluating them on unseen tasks. However, the low diversity of human-annotated downstream tasks restricts the performance of the meta-tuned model. Direct training on downstream tasks also brings undesired bias on specific input formats, label spaces, or domains, which hurts the generalization of PLMs. To enhance the ICL ability while maintaining generalization, we propose PICL (Pre-training for In-Context Learning), a framework that exploits the PLM’s ICL ability by pre-training models on data automatically constructed from the general plain-text corpus. Our framework is based on a simple observation that many paragraphs in the text documents contain “intrinsic tasks”. As shown in the left part of Figure 1, each paragraph in the document contains an intrinsic task. When doing language modeling on each paragraph, models implicitly perform the corresponding intrinsic tasks simultaneously. This shares a similar idea with the prompt-learning paradigm (Liu et al., 2021), where downstream data examples from NLP tasks are transformed into text sequences, and the model learns to perform the original tasks when trained on the text sequences with language modeling. Different from the downstream data, text paragraphs contain more diverse intrinsic tasks and have little bias on input formats, label spaces, or domains because they are free-form texts from the large-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e5a0/e5a031b1-0320-492a-ba6a-a8c2907afb7d.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a321/a32115c1-7fc5-4073-9fc8-43a5b4d2e539.png" style="width: 50%;"></div>
Figure 1: Left: An example of intrinsic tasks found in a document from the OpenWebText (Gokaslan et al., 2019) corpus. Right: The overall framework of PICL. For each paragraph z0 in the corpus C, we retrieve k paragraphs that share the same intrinsic task (Sentiment Analysis) as demonstrations and then concatenate them with z0 to construct a pre-training instance. We compute the language modeling loss on the whole instance to train the model.
scale general corpus. By gathering and concatenating paragraphs with the same intrinsic tasks (right part of Figure 1), we can construct a meta-training dataset to pre-train the model to perform the intrinsic task conditioning on paragraphs in the context, and thereby improve the ICL ability. We adopt a retrieval-based approach to gather paragraphs sharing the same intrinsic tasks from a general corpus. We first train an encoder to represent each paragraph in a vector space where paragraphs with the same intrinsic task have close embeddings. The encoder is trained with contrastive learning (Khosla et al., 2020) on a collection of downstream datasets by taking examples from the same tasks as positive pairs and those from different tasks as negative pairs. Then, treating any paragraph in the corpus as a query, we retrieve the paragraphs with close representations to the query, namely, sharing the same intrinsic task with the query. Finally, we concatenate the query and the retrieved paragraphs to get a pre-training instance. Note that although we use downstream datasets, the model is trained on instances constructed from the general corpus, which ensures its generalization. We evaluate the ICL performance of the model pre-trained with PICL on seven widelyused text classification datasets and SUPERNATURALINSTRUCTIONS (Wang et al., 2022), a benchmark whose test split contains more than 100 tasks formulated into text generation. Empirical results show the effectiveness of PICL, enabling the model to reach or even outperform larger models with nearly 4x parameters. Besides, we find that
the PICL-trained model is more generalizable on various tasks than previous meta-fine-tuning methods. We also conduct extensive experiments to analyze several key factors of PICL.
# 2 Method
We first present an overview of PICL and then describe the details in the following sections. As shown in the right part of Figure 1, we construct the pre-training instances from a corpus C consisting of paragraphs split from full documents by “\n”. For each paragraph z0 in C, we first use a retriever R to find k paragraphs {z1, z2, · · · , zk} sharing the same intrinsic task (Sentiment Analysis) with z0. Then the retrieved paragraphs are treated as demonstrations and concatenated with z0 to form a pre-training instance: zk ⊕zk−1 ⊕· · · ⊕z1 ⊕z0. Finally, we adopt a language modeling objective to pre-train the model on the constructed instances. In this way, the pre-training stage can be regarded as a meta-training process, where the model learns to solve the intrinsic task in z0 conditioning on its context zk ⊕zk−1 ⊕· · · ⊕z1. Since C is a large-scale general corpus, it contains a variety of intrinsic tasks and little domain bias, which ensures the generalization of the pre-trained model.
# 2.1 Retriever
The main component of the retriever R is a tasksemantics encoder E that represents a text paragraph as a d-dimensional vector in a space V , where paragraphs with the same intrinsic tasks have similar representations. We define the similarity
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9eb3/9eb31ce8-bce2-4307-a1e2-74a0180bceec.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: An example of how we construct the positive and negative pairs to train the task-semantics encoder E. The solid line means positive pairs, and the dashed lines mean negative pairs.</div>
# We employ the FAISS library (Johnson et al., 2019) for efficient searching.
learning (Khosla et al., 2020; Karpukhin et al., 2020) to train the task-semantics encoder E. As shown in Figure 2, we take two paragraphs with the same intrinsic task as positive pairs and those from different tasks as negative pairs. However, the annotation of a paragraph’s intrinsic task is usually unavailable. To this end, we use a collection of downstream NLP datasets from various tasks whose examples are converted into text sequences with human-written prompts to train E. In this way, treating each text sequence as a paragraph, we can regard the corresponding downstream task as the intrinsic task annotation. We assume that the instances from all downstream tasks form a dataset D. For each z0 ∈D, we have a positive instance
z+ sharing the same task with z0 and a set N(z0) consisting of negative instances with different tasks than z0, the loss function takes the form:
(2)
Positive and Negative Instances For each z0 ∈ D, we randomly sample a positive instance z+ belonging to the same task with z0 from D\{z0}. As shown in Figure 2, N(z0) contains two kinds of negative instances: (1) Easy Negatives z− easy sampled from D and belonging to different tasks than z0. (2) Hard Negatives z+ hard sharing the same prompt with z0 but containing mismatched tasks. For instance, in Figure 2, we apply the prompt from the sentiment task to the summarization task to create the hard negative instance z− hard. This prevents the model from hacking the contrastive objective using prompts like “Guess the sentiment” and learning a trivial pattern matching but forces the model to extract task semantics from the whole paragraph.
# 2.2 Data Construction
For each paragraph z0 ∈C, we concatenate the retrieved paragraphs {z1, z2, · · · , zk} = R(z0) with z0 to get a pre-training instance zk ⊕zk−1 ⊕· · · ⊕ z1 ⊕z0. To improve the quality of the constructed data, we derive an approach to filter out instances that are less informative to ICL. We consider the following score to measure the informativeness of an instance based on the perplexity difference of the paragraphs in the instance before and after they are concatenated as a sequence:
s = −�k i=0 log P(zi) + log P(zk ⊕zk−1 ⊕· · · ⊕z0) |zk ⊕zk−1 ⊕· · · ⊕z0| , (
where | · | is the length of a sequence and P(·) is the language modeling probability based on any uni-direct PLMs. Given a manually set threshold δ, we retain the instances that satisfy s > δ. This criterion leverages the original ICL ability of the PLM. If concatenating the paragraphs results in lower perplexity, they are more correlated and may be more informative for ICL. We finally construct a pre-training corpus containing N instances Cpre-train = {zi k ⊕zi k−1⊕, · · · , ⊕zi 1 ⊕zi 0}N i=1.
# 2.3 Pre-Training
We pre-train the model with auto-regressive language modeling on Cpre-train. Unlike previous
works (Min et al., 2022b; Chen et al., 2022b), which only compute the language modeling loss on the label tokens, we compute the loss on the whole sequence. There are two reasons for this choice. First, the intrinsic tasks are already in the natural language format, and it is unnecessary to split the input and the label. Second, we argue that computing loss on the whole sequence ensures a large token number in a forward batch, which is critical to maintaining the basic in-weights ability (Chan et al., 2022). Therefore, the loss function is:
where θ is the parameters of the model. In addition, we find that adding a language modeling loss LLM(θ) on the original full documents before being split into paragraphs benefits the performance. Therefore, the final optimization objective is:
(5)
where we set α = 0.5 in our main experiments.
# 3 Experimental Setup
# 3 Experimental Setup 3.1 Pre-training Data
# 3.1 Pre-training Data
We merge OPENWEBTEXT (Gokaslan et al., 2019), WIKICORPUS (Foundation, 2022), and BOOKCORPUS (Zhu et al., 2015) to construct the pre-training data, where full documents are split into paragraphs by “\n”. The corpus C consists of 80M paragraphs, totaling about 30GB. For each paragraph, we search for k = 20 demonstrations and concatenate them until 1024 tokens, the maximum input length constraint of the language model we used. This ensures that the model sees various demonstration numbers during pre-training. We use GPT2-Large (Radford et al., 2019) to compute P(·) in Equation 3 and set δ = 0.0 for filtering. More details of data processing and statistics are shown in Appendix A.
# 3.2 Baselines
# We consider four baselines in our experiments:
• VanillaICL directly prompts a PLM with the concatenation of training examples to do ICL. • ExtraLM further pre-trains the PLM on the original full documents before being split into paragraphs with the language modeling objective. • Self-Sup (Chen et al., 2022a) designs four self-
supervised pre-training objectives, including Next Sentence Generation, Masked Word Prediction, Last Phrase Prediction, and Classification, to enhance the ICL performance. We conduct the self-supervised pre-training on our merged corpus for a fair comparison. • MetaICL (Min et al., 2022b) meta-trains the model on a large collection of downstream human-annotated datasets for learning to learn in context. The meta-training instances are constructed by concatenating several training examples in each dataset to a single text sequence. We replicate the method on the training set of our task-semantics encoder for a fair comparison.
# 3.3 Evaluation
We evaluate the model trained with PICL on two kinds of downstream tasks.
Few-Shot Text Classification We consider seven widely-used text classification datasets, including SST-2 (Socher et al., 2013), SST5 (Socher et al., 2013), Subj (Pang and Lee, 2004), MR (Pang and Lee, 2005), RTE (Dagan et al., 2006), CB (De Marneffe et al., 2019), and AGNews (Zhang et al., 2015) to evaluate the few-shot ICL performance of the trained models (see Appendix B.1 for more details). Note that these tasks are not included in the training set of the tasksemantics encoder. We randomly sample 4 or 8 demonstrations from the official training sets of each dataset. Effects of other demonstration numbers can be found in Section 4.3. We compute the average accuracy scores on at most 1000 samples from the validation split of each dataset across five random seeds for selecting demonstrations.
Instruction Following To test the generalization of PICL, we also evaluate the trained model on a larger range of tasks with more free-form inputs, including both human instructions and fewshot examples. We use the test split of SUPERNATURALINSTRUCTIONS (Wang et al., 2022) as the benchmark and exclude the tasks that appear in the training set of the task-semantics encoder, resulting in 105 evaluation tasks (see Appendix B.2 for a full list of tasks). Each task is specified with a human-written instruction and two or three demonstrations. We follow Wang et al. (2022) to formulate all tasks to the text generation format and score the outputs with ROUGE-L (Lin, 2004).
Shot
Method
Param.
SST2
SUBJ
MR
RTE
AgNews
CB
SST5
Average
4-shot
VanillaICL
770M
67.59.2
57.77.8
50.30.3
50.81.7
67.52.3
68.12.4
24.45.4
55.20.5
VanillaICL
1.5B
74.99.7
65.210.0
61.96.5
50.40.4
65.64.8
67.85.6
32.44.6
59.72.5
VanillaICL
2.7B
75.07.5
65.42.9
71.413.3
49.81.8
65.62.8
60.02.1
32.15.4
59.91.1
ExtraLM
770M
68.911.3
63.96.4
60.36.4
51.21.7
64.51.5
63.75.3
27.85.1
57.22.1
Self-Sup
770M
55.07.4
50.30.6
59.73.5
52.22.0
50.37.0
63.47.1
28.83.3
51.42.2
MetaICL
770M
69.84.0
63.54.6
65.67.5
57.62.3
66.32.4
65.23.0
31.72.1
60.01.5
PICL
770M
79.78.6
66.87.4
81.01.3
54.51.8
67.73.4
69.64.3
34.84.0
64.41.6
8-shot
VanillaICL
770M
68.76.0
66.69.8
60.25.5
51.81.6
60.25.6
68.83.2
31.43.8
58.22.9
VanillaICL
1.5B
72.112.6
63.46.5
63.35.4
52.72.8
54.28.4
70.45.7
33.53.3
58.62.5
VanillaICL
2.7B
71.011.6
65.24.0
70.46.3
51.32.0
63.12.4
69.64.0
34.12.8
60.63.2
ExtraLM
770M
69.73.4
65.26.5
63.66.0
52.61.6
58.97.0
69.63.8
32.24.7
58.81.6
Self-Sup
770M
61.46.5
54.34.5
73.88.1
53.02.4
52.13.8
63.06.9
33.71.8
55.92.1
MetaICL
770M
73.66.2
67.28.8
70.15.6
53.62.1
56.10.7
65.84.1
33.74.7
60.02.2
PICL
770M
78.010.6
69.39.5
77.55.0
53.01.6
64.74.4
70.42.1
34.13.8
63.91.3
Table 1: Main results of few-shot text classification. We report the average accuracy scores and the standard deviations across 5 random seeds for selecting demonstrations. We use GPT2-Large (770M), GPT2-xLarge (1.5B), and GPT-Neo (2.7B) for VanillaICL. The best scores on each dataset under 4 or 8 evaluation shots are in boldface.
# 3.4 Settings
Retriever We train the task-semantics encoder on 37 tasks (see Appendix C) using up to 10K examples per task. To enhance generalization, we apply multiple prompts from PromptSource (Bach et al., 2022) to one example and use 320 prompts in all. We use the in-batch negative trick (Chen et al., 2020) to compute the contrastive loss. We set the learning rate to 5 × 10−5, the batch size to 64, and construct 4 hard negatives for each instance. The encoder is trained from RoBERTaBase for 1 epoch. Language Model We test PICL based on the 770M GPT2-Large (Radford et al., 2019) unless otherwise specified. Results on larger models can be found in Appendix E.1. To save computational resources, we train the model from its pretrained checkpoints. We also test the VanillaICL performance of larger models, including GPT2xLarge (Radford et al., 2019) (1.5B) and GPTNeo (Black et al., 2021) (2.7B) for reference. Pre-Training We set the maximum learning rate to 1×10−6 and use the “inverse square root” scheduler (Vaswani et al., 2017) with 1000 steps warmup. The model sees 131K tokens in a step and is pretrained for 100K steps. It takes less than a day to finish pre-training on 64 V100 32G GPUs.
# 4 Results
# 4.1 Few-Shot Text Classification
Table 1 shows the results of few-shot text classification, from which we have 3 observations.
First, among the baselines with 770M parameters, simply further training the model on our corpus with language modeling improves the performance (ExtraLM). This is likely due to the higher domain diversity of our corpus. MetaICL is helpful on most datasets, which verifies the effectiveness of meta-training for ICL. Self-Sup fails to bring benefits on most datasets against VanillaICL, probably because the constrained label space of the Classification training task (only contains “True” and “False”) brings bias to the model’s output. This emphasizes the importance of using training objectives with little bias. Second, we observe that the PICL-trained model outperforms the baselines with the same model sizes by a large margin on most datasets across different shots, verifying the effectiveness of PICL. An exception is RTE, where MetaICL performs the best. We speculate the reason is that some training tasks of MetaICL share the same label space with RTE (“Yes”/“No”), such as paraphrase identification. Min et al. (2022c) has shown that the label space plays a vital role in ICL, which explains the good performance of MetaICL on RTE. Thrid, comparing models across different sizes, we find that increasing the model parameters boosts the performance, but PICL enables the 770M model to beat a 2.7B counterpart. This indicates that the ICL ability can be enhanced not only through scaling up the parameters. Improving the structure of the pre-training data is also beneficial. In Appendix E.1, we can see that PICL is also effective when applied to a 1.5B model.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/55f5/55f53bf7-e836-4d52-8b92-7929760e763b.png" style="width: 50%;"></div>
Figure 3: Comparison between PICL and MetaICL on SUPER-NATURALINSTRUCTIONS (Wang et al., 2022) Each bar represents an evaluation task. The y-axis means the ROUGE-L score difference between the two methods
Model
Param.
ROUGE-L
VanillaICL
770M
34.3
VanillaICL
1.5B
34.9
VanillaICL
2.7B
37.3
ExtraLM
770M
34.6
Self-Sup
770M
30.5
MetaICL
770M
35.3
PICL
770M
37.6
Table 2: Results of instruction following on SUPERNATURALINSTRUCTIONS. We report the average ROUGE-L score across all 105 evaluation tasks.
# 4.2 Instruction Following
The results on SUPER-NATURALINSTRUCTIONS are shown in Table 2. We can see that PICL achieves higher overall instruction following performance than the baselines, outperforming a larger model with about 4x parameters. In Figure 3, we compare the per-task performance of PICL and MetaICL because they share the most similar setting where human-annotated downstream datasets are used. We observe that PICL outperforms MetaICL on about 3/4 of evaluation tasks, indicating that compared to fine-tuning directly on downstream tasks, pre-training on intrinsic tasks constructed from the general plain-text corpus brings better ICL ability and ensures higher generalization performance across a broad range of tasks (see Appendix E.2 for more details). Most tasks where MetaICL beats PICL belong to text classification whose output spaces are “Yes/No” or “True/False”. This matches the second observation in Section 4.1, where MetaICL predicts “Yes/No” well because of training on tasks that share the same label spaces. On the other hand, PICL performs much better on text generation, or tasks whose output spaces share the same semantics with “Yes/No” but use label words not in the training tasks of MetaICL (e.g., “Correct/Wrong”).
This indicates that direct training on downstream datasets causes overfitting to specific labels. There are also tasks where PICL performs similarly to MetaICL, such as reasoning and word analogy. We notice that the improvements of PICL and MetaICL on these tasks are also marginal against VanillaICL probably because these tasks rely more on the “inweights learning” ability (Chan et al., 2022), rather than in-context learning.
# 4.3 Analysis
Effect of Retriever We compare different approaches to retrieve paragraphs and test the final model performance. We try randomly selecting paragraphs (Random), retrieving using the nonparametric approach (BM25), encoding each paragraph with the original pre-trained encoder as it is (RoBERTa), or using the encoder for sentence similarity (Reimers and Gurevych, 2019) (SRoBERTa). We also study different numbers of hard negatives (0, 1, 4) and downstream tasks (7, 24, 37) to train the task-semantics encoder in PICL. From the results in Table 3, we can see that all retrieval methods except Random bring improvements against VanillaICL on both text classification and instruction following settings, indicating that improving the coherence of the paragraphs in the pre-training data benefits ICL. Using the task-semantics encoder in PICL achieves the best performance, showing the importance of retrieving paragraphs based on task semantics rather than word overlap or sentence meanings. Comparing different settings to train the task-semantics encoder, we observe that increasing the number of hard negatives and training tasks improves the final performance. This is in line with previous works (Karpukhin et al., 2020; Chen et al., 2020; He et al., 2020) that more challenging hard negatives benefit contrastive learning. Effect of Demonstration Numbers Training with PICL brings two benefits: (1) PLMs learn
Retriever
nHardNeg.
nTasks
CLS
SUP-NI
Accuracy
ROUGE-L
VanillaICL
-
-
55.2
34.3
Random
-
-
56.7
29.3
BM25
-
-
59.2
34.5
RoBERTa
-
-
58.7
34.6
SRoBERTa
-
-
59.0
35.0
PICL
0
37
62.2
36.4
1
37
63.1
36.5
4
7
61.6
35.4
4
24
63.4
36.6
4
37
64.4
37.6
Table 3: Comparison of different retrievers. nHardNeg. and nTasks means the number of hard negatives and downstream tasks to train the task-semantics encoder in PICL. “CLS Accuracy” means the average accuracy scores on text classification tasks. “SUP-NI ROUGEL” means the average ROUGE-L scores across the tasks in SUPER-NATURALINSTRUCTIONS.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0830/0830ed4a-c4c4-4bd4-92f4-efe27fbfc355.png" style="width: 50%;"></div>
Figure 4: Average text classification accuracy when the pre-training instances contain different demonstration numbers in PICL (the number in the brackets). “PICLdefault” means using a mixture of demonstration numbers as in previous experiments.
a format where demonstrations from the same task are concatenated as the prefix, which is beneficial when the model is evaluated under the same number of demonstrations. (2) PLMs learn a better ability to infer and perform tasks from the context, even when the demonstration numbers in evaluation and pre-training do not match. To differentiate these effects, we conduct pre-training on instances containing only 4, 8, or 16 demonstrations and test the trained models under different text classification shots. Results in Figure 4 show that when pre-trained with different demonstration numbers, the models generalize well to unseen demonstration numbers in evaluation, achieving similar performance with the default setup where the model sees various demonstration numbers in pre-training (PICL-default). This indicates that the models learn more than the input formats in PICL.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0f5a/0f5af843-af64-403b-92cd-7f4b02cbd187.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c533/c5337d41-506d-453a-99c4-14cd1c1e982e.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e9ad/e9ad0814-7808-449b-93e3-20faf01fd471.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Effect of α</div>
Effect of Full Documents In Figure 5(b), we report the model performance on text classification tasks when using different choices of α, which controls the proportion of the full-document data. We find that balancing the constructed and fulldocument data performs the best (α = 0.5). When α is too large, the model is trained mostly on our constructed data and overfits its bias inevitably introduced by the task-semantics encoder in the data construction process. When α is too small, our method degenerates into ExtraLM.
Effect of Data Amount We study the size effect of the corpus used to construct the pre-training
<div style="text-align: center;">(a) Effect of Data Amount</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7578/7578675b-752e-409a-8b7c-b1bc4d9a9b9a.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Data Comparison</div>
Figure 6: Data analysis. (a): the average 4-shot text classification accuracies when constructing data using different proportions of the original corpus. (b): perplexity of full-document data (Full Doc), random retrieved data (Rand ⊕z0) and PICL data (R(z0) ⊕z0) based on GPT-J (6B) and the corresponding model performance.
data in PICL and report the performance on text classification tasks in Figure 6(a). We conduct the data construction on 0.01%, 0.1%, 1%, and 10% of the original 80M paragraphs (100%) and pre-train models for at most 100K steps until the validation losses begin to increase. From the results, we conclude that when the corpus is small, pre-training with the constructed data hurts the performance because the search library is too small to find paragraphs sharing the same intrinsic tasks. Training on small data for multiple epochs also causes overfitting. When the corpus contains more than 80K paragraphs (0.1%), adding more data constantly improves the performance, which is consistent with the scaling law (Kaplan et al., 2020).
training data: original full documents before being split into paragraphs (Full Doc), concatenation of randomly selected paragraphs (Rand ⊕z0), and the concatenated same-intrinsic-task paragraphs gathered using the retrieval method in PICL before filtering (R(z0) ⊕z0). We can see that the data constructed by retrieval has much lower perplexity and correspondingly yields higher accuracy scores, which verifies its usefulness. In Appendix F, we present several examples of the retrieved paragraphs and the corresponding intrinsic tasks.
# 5 Related Work
In-Context Learning Recently, in-context learning (ICL), where models perform tasks simply conditioning on instructions or the concatenation of examples in the context (Brown et al., 2020), has been found promising for using PLMs in various application scenarios. To this end, there emerge many works to improve the ICL performance by calibrating the model predictions (Zhao et al., 2021; Han et al., 2022; Holtzman et al., 2021; Min et al., 2022a), selecting or reordering demonstrations (Rubin et al., 2022; Liu et al., 2022; Lu et al., 2022), designing pre-training tasks (Chen et al., 2022a), and breaking the context length limits (Hao et al., 2022). However, the underlying mechanism of ICL is poorly understood (Min et al., 2022c). Therefore, some works propose mathematical frameworks to reveal how ICL works (Xie et al., 2021; Olsson et al., 2022; Elhage et al., 2021), or investigate the pre-training data to explain ICL’s good performance (Chan et al., 2022; Shin et al., 2022). Multi-Task Fine-tuning for Cross-Task Generalization Fine-tuning PLMs on a large collection of downstream tasks enables generalization to unseen tasks under zero-shot (Wei et al., 2022; Sanh et al., 2022; Ouyang et al., 2022; Chung et al., 2022) and few-shot (Min et al., 2022b; Chen et al., 2022b; Mishra et al., 2022; Garg et al., 2022) scenarios. However, the performance of multi-task fine-tuning is largely restricted by the diversity of the annotated training tasks (Gu et al., 2022b), which requires massive human efforts to scale up. In addition, direct training on downstream tasks easily brings undesired bias. In this work, we propose to meta-train the model with the intrinsic tasks automatically collected from the large-scale general corpus, which is easier to scale up and introduces little bias.
Pre-training Data Programming The conventional pre-training paradigm trains the model on plain-text corpora with the language modeling objective (Radford et al., 2018, 2019; Brown et al., 2020). Recently works have found that carefully designed pre-training instances can further boost specific abilities like prompt adaption (Gu et al., 2022a), reasoning (Razeghi et al., 2022), or sentence representation (Levine et al., 2021). Our work studies constructing pre-training instances to improve the PLM’s ICL ability while still maintaining its generalization on various NLP tasks.
This paper presents PICL, a framework that exploits the in-context learning ability of PLMs by pre-training models on concatenations of text paragraphs sharing the same “intrinsic tasks” gathered from the large-scale general corpus. In PICL, models learn to perform various intrinsic tasks conditioning on their context while preserving their generalization due to the little bias of the pre-training data. Extensive experiments show that PICL improves the ICL performance on various datasets against several baselines, enabling a 770 M model to outperform a larger model with about 4x parameters while maintaining good generalization across a wide range of tasks. For future work, we would like to consider adding human instructions to our pre-training framework to enhance more abilities of PLMs like zero-shot instruction following.
# Limitations
One limitation of our paper is that the exact distribution of the intrinsic tasks in the original corpus and the constructed data is still unknown. Knowing the distribution can offer a better interpretation of the effectiveness of PICL, even of the strong performance of large language models. Besides, although we can find many constructed instances that share obvious intrinsic tasks (see Appendix F), there still exist some instances where the intrinsic tasks are hard to identify. How to better evaluate the contribution of these instances to the ICL ability or designing better filtering approaches to select more informative data for ICL is worth studying. Our task-semantics encoder inevitably contains some bias because it is trained on downstream datasets, although we have tried to ensure a large number and diversity of the dataset collection. However, the final language model is pre-trained on
the general corpus, and we add the full document loss, which eliminates the bias to some extent. Regarding computing power, we acknowledge that our framework takes relatively large training resources in the retrieval and pre-training process. Therefore, we did not conduct experiments based on extra-large language models.
# Acknowledgements
This work was supported by the NSFC projects (Key project with No. 61936010 ). This work was also supported by the Guoqiang Institute of Tsinghua University, with Grant No. 2020GQG0005.
# References
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a944/a94451b7-0cfd-429e-aa54-fde2bdd333b7.png" style="width: 50%;"></div>
# A Details of the Pre-training Corpus
This section presents details of the data processing of the pre-training corpus and its statistics.
Data Processing Our pre-training corpus is a merge of OPENWEBTEXT (Gokaslan et al., 2019), WIKICORPUS (Foundation, 2022), and BOOKCORPUS (Zhu et al., 2015), downloaded from the HuggingFace datasets repository1. We first split each document in the corpus into paragraphs with “\n”. To avoid training with too short paragraphs, we concatenate a paragraph with previous paragraphs if the token number after concatenation is lower than 128. We also exclude paragraphs longer than 500 tokens because they are not likely to fit into an instance with more than 1 paragraph. The filtering process in Section 2.3 drops about 24% instances. The licenses of all corpora allow for scientific research. Statistics We plot the distribution of the mean paragraph length per instance in Figure 7(a) and the distribution of the paragraph number per instance in Figure 7(b). The average paragraph length is 150.0, and the average paragraph number in an instance is 11.7. We can see that the model sees various demonstration numbers in PICL pre-training.
# B Details of the Evaluation Data
# B.1 Few-shot Text Classification
The details of each text classification dataset and the corresponding prompt in evaluation are listed in Table 6. All datasets are downloaded from the HuggingFace datasets repository1. We simplify the evaluation prompts as much as possible to reduce the effect of prompt engineering. Following previous works (Brown et al., 2020; Sanh et al., 2022), the model is evaluated by the ranking score strategy, where we compare the perplexity of each classification label under the model and choose the label with the lowest perplexity. The licenses of all datasets allow for scientific research.
# B.2 Instruction Following
The original test split of the benchmark SUPERNATURALINSTRUCTIONS (Wang et al., 2022) contains 119 tasks. We exclude tasks that appear in the training tasks of the task-semantics encoder or whose input length is too long to fit in the context of our model. Our final evaluation includes 105 tasks. A full list of the tasks is shown in Table 8.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6f0a/6f0aaa8a-a490-4704-9fb9-4ae1f80adf28.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bf67/bf679218-e826-48eb-9a24-f3c3aa0218cc.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Paragraph Number</div>
<div style="text-align: center;">(a) Paragraph Length</div>
Figure 7: Pre-training data statistics. (a): the distribution of the average paragraph length per instance. (b): the distribution of the paragraph number per instance.
Stage
Name
Values
Retrieve
learning rate
1e-4, 5e-5, 1e-5
batch size
16, 32, 64
hard negatives
0, 1, 4
Pre-train
learning rate
1e-5, 5e-6, 1e-6, 2e-7
batch size
64, 128, 256, 512
warmup
0, 1000, 5000
Table 4: Searching intervals of hyper-parameters.
We use the same template to combine few-shot examples with task instructions as Wang et al. (2022). The license of this benchmark is Apache License 2.0.
# C Details of the Downstream Training Data
The downstream datasets we use to train the tasksemantics encoder are a merge of the training data used in (Sanh et al., 2022) and the HR→LR setting in (Min et al., 2022b). All datasets are downloaded from the HuggingFace datasets repository 1 and all prompts come from the PromptSource library (Bach et al., 2022)2. We exclude datasets from the sentiment classification task, the topic classification task, and the natural language inference task because they are included in our text classification evaluation. We finally get a collection of 37 datasets, as listed in Table 5. The licenses of all datasets allow for scientific research.
# D More Experimental Details
All model checkpoints we used come from the HuggingFace models repository3. The searching interval of each hyper-parameter is listed in Table 4.
# E More Results
# E.1 Results on Larger Base Model
We test PICL based on the GPT2-xLarge (Radford et al., 2019) with 1.5B parameters. From the results in Figure 7, we can see that PICL is also applicable to larger models, outperforming the baselines based on the same-sized model on most datasets.
# E.2 Instruction Following
We present the performance comparison between PICL and MetaICL per evaluation task in Figure 8. PICL outperforms MetaICL on 77 / 105 tasks, indicating that PICL ensures the better generalization of the trained model. The name of each task is also listed in Figure 8. We can see that the top three tasks where MetaICL performs the best are:
• doqa_movies_isanswerable, • glue_entailment_classification, • tweetqa_classification,
• winogrande_question_modification_object, • plausible_result_generation, • winowhy_reason_plausibility_detection ,
which are text generation, text generation, and “Correct/Wrong” classification tasks respectively. The four tasks where PICL and MetaICL have the
• bard_analogical_reasoning_containers, • copa_commonsense_cause_effect, • winogrande_answer_generation, • bard_analogical_reasoning_trash_or_treasure, which belong to commonsense reasoning and word analogy tasks.
# F Case Studies
In Table 9 and 10, we present several cases of the retrieved paragraphs and the corresponding intrinsic tasks. We can see that there exists a large range of intrinsic tasks in the constructed data and many of them do not appear in the training data of the tasksemantics encoder, which shows the generalization of the encoder.
10
0
10
20
30
 ROUGE-L (PICL - MetaICL)
task242_tweetqa_classification
task1344_glue_entailment_classification
task1442_doqa_movies_isanswerable
task623_ohsumed_yes_no_answer_generation
task642_esnli_classification
task329_gap_classification
task1612_sick_label_classification
task1195_disflqa_disfluent_to_fluent_conversion
task1624_disfl_qa_question_yesno_classification
task1728_web_nlg_data_to_text
task199_mnli_classification
task743_eurlex_summarization
task640_esnli_classification
task520_aquamuse_answer_given_in_passage
task1409_dart_text_generation
task1516_imppres_naturallanguageinference
task970_sherliic_causal_relationship
task738_perspectrum_classification
task304_numeric_fused_head_resolution
task641_esnli_classification
task1154_bard_analogical_reasoning_travel
task1622_disfl_qa_text_modication
task1156_bard_analogical_reasoning_tools
task1157_bard_analogical_reasoning_rooms_for_containers
task1155_bard_analogical_reasoning_trash_or_treasure
task033_winogrande_answer_generation
task828_copa_commonsense_cause_effect
task1159_bard_analogical_reasoning_containers
task827_copa_commonsense_reasoning
task1534_daily_dialog_question_classification
task1158_bard_analogical_reasoning_manipulating_items
task290_tellmewhy_question_answerability
task392_inverse_causal_relationship
task891_gap_coreference_resolution
task677_ollie_sentence_answer_generation
task1152_bard_analogical_reasoning_causation
task249_enhanced_wsc_pronoun_disambiguation
task288_gigaword_summarization
task1586_scifact_title_generation
task769_qed_summarization
task201_mnli_neutral_classification
task937_defeasible_nli_social_classification
task879_schema_guided_dstc8_classification
task1342_amazon_us_reviews_title
task202_mnli_contradiction_classification
task1161_coda19_title_generation
task1640_aqa1.0_answerable_unanswerable_question_classification
task1390_wscfixed_coreference
task220_rocstories_title_classification
task418_persent_title_generation
task1391_winogrande_easy_answer_generation
task200_mnli_entailment_classification
task020_mctaco_span_based_question
task619_ohsumed_abstract_title_generation
task1529_scitail1.1_classification
task620_ohsumed_medical_subject_headings_answer_generation
task035_winogrande_question_modification_person
task281_points_of_correspondence
task1540_parsed_pdfs_summarization
task1394_meta_woz_task_classification
task1439_doqa_cooking_isanswerable
task233_iirc_link_exists_classification
task510_reddit_tifu_title_summarization
task226_english_language_answer_relevance_classification
task349_squad2.0_answerable_unanswerable_question_classification
task050_multirc_answerability
task1153_bard_analogical_reasoning_affordance
task036_qasc_topic_word_to_generate_related_fact
task1388_cb_entailment
task1407_dart_question_generation
task391_causal_relationship
task890_gcwd_classification
task957_e2e_nlg_text_generation_generate
task613_politifact_text_generation
task219_rocstories_title_answer_generation
task362_spolin_yesand_prompt_response_sub_classification
task880_schema_guided_dstc8_classification
task569_recipe_nlg_text_generation
task935_defeasible_nli_atomic_classification
task1531_daily_dialog_type_classification
task442_com_qa_paraphrase_question_generation
task1533_daily_dialog_formal_classification
task614_glucose_cause_event_detection
task1554_scitail_classification
task039_qasc_find_overlapping_words
task648_answer_generation
task936_defeasible_nli_snli_classification
task1659_title_generation
task500_scruples_anecdotes_title_generation
task602_wikitext-103_answer_generation
task401_numeric_fused_head_reference
task893_gap_fill_the_blank_coreference_resolution
task1385_anli_r1_entailment
task892_gap_reverse_coreference_resolution
task1387_anli_r3_entailment
task1631_openpi_answer_generation
task1615_sick_tclassify_b_relation_a
task330_gap_answer_generation
task190_snli_classification
task1386_anli_r2_entailment
task1393_superglue_copa_text_completion
task1664_winobias_text_generation
task133_winowhy_reason_plausibility_detection
task393_plausible_result_generation
task034_winogrande_question_modification_object
PICL > MetaICL
MetaICL > PICL
<div style="text-align: center;">Figure 8: Per-task results of the comparison between PICL and MetaICL.</div>
COS-E (Aggarwal et al., 2021)
DREAM (Sun et al., 2019)
QuAIL (Rogers et al., 2020)
QuaRTz (Tafjord et al., 2019b)
Social-IQA (Sap et al., 2019)
WiQA (Tandon et al., 2019)
CosmosQA (Huang et al., 2019)
QASC (Khot et al., 2020)
QUAREL (Tafjord et al., 2019a)
SciQ (Welbl et al., 2017)
Wiki-Hop (Welbl et al., 2018)
Adversarial-QA (Ren et al., 2018)
Quoref (Dasigi et al., 2019)
ROPES (Lin et al., 2019)
DuoRC (Saha et al., 2018)
Hotpot-QA (Yang et al., 2018)
Wiki-QA (Yang et al., 2015)
Common-Gen (Lin et al., 2020)
Wiki-Bio (Lebret et al., 2016)
SAMSum (Gliwa et al., 2019)
XSum (Narayan et al., 2018)
MRPC (Dolan and Brockett, 2005)
PAWS (Zhang et al., 2019)
QQP (Sharma et al., 2019)
art (Bhagavatula et al., 2019)
circa (Louis et al., 2020)
discovery (Sileo et al., 2019)
Freebase_QA (Jiang et al., 2019)
google_wellformed_query (Faruqui and Das, 2018)
HellaSwag (Zellers et al., 2019)
liar (Wang, 2017)
piqa (Bisk et al., 2020)
scitail (Khot et al., 2018)
swag (Zellers et al., 2018)
tab_fact (Chen et al., 2019)
yahoo_answer_topics4
DBpedia (Lehmann et al., 2014)
Dataset
Task
Prompt
Label Space
SST2
Sent. CLS
Sentence: {sentence} Label: {label}
Negative / Positive
SST5
Sent. CLS
Sentence: {sentence} Label: {label}
Terrible / Bad / Neutral / Good
/ Great
MR
Sent. CLS
Sentence: {sentence} Label: {label}
Negative / Positive
RTE
NLI
Passage: {premise} Question: {hypothesis} Answer: {label}
Yes / No
CB
NLI
Passage: {premise} Question: {hypothesis} Answer: {label}
Yes / No / Maybe
SUBJ
Subj. CLS
Input: {text} Type: {label}
Objective / Subjective
AgNews
Topic CLS
Sentence: {text} Label: {label}
World politics / Sports / Busi-
ness / Science and technology
Table 6: Details of the text classification datasets. “Sent. CLS” stands for “Sentiment Classification”. “NLI” stands for “Natural Language Inference”. “Subj. CLS” stands for “Subjectivity Classification”. “Topic CLS” stands for
Table 6: Details of the text classification datasets. “Sent. CLS” stands for “Sentiment Classification”. “NLI” stands for “Natural Language Inference”. “Subj. CLS” stands for “Subjectivity Classification”. “Topic CLS” stands for “Topic Classification”.
Shot
Method
SST2
SUBJ
MR
RTE
AgNews
CB
SST5
Average
GPT-xlarge
VanillaICL
74.99.7
65.210.0
61.96.5
50.40.4
65.64.8
67.85.6
32.44.6
59.72.4
MetaICL
71.12.0
64.97.6
66.86.3
60.02.8
66.25.4
64.41.6
34.63.7
61.21.3
PICL
86.92.8
72.57.3
76.24.6
54.02.7
67.16.0
70.04.6
38.04.2
66.41.6
GPT-Neo
VanillaICL
75.07.5
65.42.9
71.413.3
49.81.8
65.62.8
60.02.1
32.15.4
59.91.2
MetaICL
80.15.8
55.69.5
73.19.0
57.53.9
64.23.4
65.56.4
32.84.7
61.31.4
PICL
86.41.0
68.65.7
83.62.4
50.20.7
67.51.2
63.13.7
35.73.6
65.01.1
Evaluation Tasks (105)
Coreference Resolution
task893_gap_fill_the_blank_coreference_resolution
task1664_winobias_text_generation
task648_answer_generation
task304_numeric_fused_head_resolution
task891_gap_coreference_resolution
task033_winogrande_answer_generation
task892_gap_reverse_coreference_resolution
task401_numeric_fused_head_reference
task1390_wscfixed_coreference
task133_winowhy_reason_plausibility_detection
task330_gap_answer_generation
task329_gap_classification
task249_enhanced_wsc_pronoun_disambiguation
task1391_winogrande_easy_answer_generation
Textual Entailment
task641_esnli_classification
task1529_scitail1.1_classification
task202_mnli_contradiction_classification
task1344_glue_entailment_classification
task1387_anli_r3_entailment
task738_perspectrum_classification
task890_gcwd_classification
task1612_sick_label_classification
task936_defeasible_nli_snli_classification
task1386_anli_r2_entailment
task201_mnli_neutral_classification
task1385_anli_r1_entailment
task1516_imppres_naturallanguageinference
task1615_sick_tclassify_b_relation_a
task970_sherliic_causal_relationship
task199_mnli_classification
task935_defeasible_nli_atomic_classification
task937_defeasible_nli_social_classification
task1388_cb_entailment
task1554_scitail_classification
task190_snli_classification
task200_mnli_entailment_classification
task640_esnli_classification
task642_esnli_classification
Cause Effect Classification
task1393_superglue_copa_text_completion
task391_causal_relationship
task828_copa_commonsense_cause_effect
task614_glucose_cause_event_detection
task827_copa_commonsense_reasoning
task393_plausible_result_generation
task392_inverse_causal_relationship
Title Generation
task288_gigaword_summarization
task1161_coda19_title_generation
task619_ohsumed_abstract_title_generation
task500_scruples_anecdotes_title_generation
task569_recipe_nlg_text_generation
task1586_scifact_title_generation
task602_wikitext-103_answer_generation
task769_qed_summarization
task510_reddit_tifu_title_summarization
task743_eurlex_summarization
task1342_amazon_us_reviews_title
task418_persent_title_generation
task220_rocstories_title_classification
task1659_title_generation
task219_rocstories_title_answer_generation
task1540_parsed_pdfs_summarization
Dialogue Act Recognition
task880_schema_guided_dstc8_classification
task1531_daily_dialog_type_classification
task1394_meta_woz_task_classification
task362_spolin_yesand_prompt_response_sub_classification
task1533_daily_dialog_formal_classification
task879_schema_guided_dstc8_classification
task1534_daily_dialog_question_classification
Answerability Classification
task1439_doqa_cooking_isanswerable
task1640_aqa1.0_answerable_unanswerable_question_classification
task242_tweetqa_classification
task1442_doqa_movies_isanswerable
task233_iirc_link_exists_classification
task290_tellmewhy_question_answerability
task520_aquamuse_answer_given_in_passage
task226_english_language_answer_relevance_classification
task050_multirc_answerability
task349_squad2.0_answerable_unanswerable_question_classification
task1624_disfl_qa_question_yesno_classification
task020_mctaco_span_based_question
Data to Text
task1728_web_nlg_data_to_text
task1409_dart_text_generation
task1407_dart_question_generation
task957_e2e_nlg_text_generation_generate
task677_ollie_sentence_answer_generation
task1631_openpi_answer_generation
Keyword Tagging
task036_qasc_topic_word_to_generate_related_fact
task620_ohsumed_medical_subject_headings_answer_generation
task613_politifact_text_generation
task623_ohsumed_yes_no_answer_generation
Word Analogy
task1159_bard_analogical_reasoning_containers
task1154_bard_analogical_reasoning_travel
task1152_bard_analogical_reasoning_causation
task1155_bard_analogical_reasoning_trash_or_treasure
task1156_bard_analogical_reasoning_tools
task1157_bard_analogical_reasoning_rooms_for_containers
task1153_bard_analogical_reasoning_affordance
task1158_bard_analogical_reasoning_manipulating_items
Overlap Extraction
task039_qasc_find_overlapping_words
task281_points_of_correspondence
Question Rewriting
task035_winogrande_question_modification_person
task1195_disflqa_disfluent_to_fluent_conversion
task034_winogrande_question_modification_object
task442_com_qa_paraphrase_question_generation
task1622_disfl_qa_text_modication
Excluded Tasks (14)
task1356_xlsum_title_generation
task670_ambigqa_question_generation
task645_summarization
task760_msr_sqa_long_text_generation
task402_grailqa_paraphrase_generation
task1598_nyc_long_text_generation
task671_ambigqa_text_generation
task121_zest_text_modification
task1345_glue_qqp_question_paraprashing
task1557_jfleg_answer_generation
task232_iirc_link_number_classification
task1358_xlsum_title_generation
task1562_zest_text_modification
task102_commongen_sentence_generation
<div style="text-align: center;">List of Tasks</div>
1
• Marko Jovanovski Marko Jovanovski (born 24 July 1988) is a Macedonian professional
footballer who plays as a goalkeeper for Akademija Pandev.
• Andreas Paraskevas Andreas Paraskevas (; born 15 September 1998) is a Cypriot footballer
who plays as a goalkeeper for Doxa Katokopias.
• Evripidis Giakos Evripidis Giakos (; born 9 April 1991) is a Greek professional footballer
who plays as an attacking midfielder for Super League 2 club AEL.
World Knowledge Com-
pletion
2
• The Hive scouting teams had been infiltrating our space for several weeks, sending three-
and six-man teams in. In short, they were making me look bad on the home world.
• And for good measure, Walker ordered the Wisconsin National Guard to prepare to
intervene in case of any strike action by unions. In a word, Walker wants the destruction
of organized labor in Wisconsin.
• "Scientists began examining him... he was covered in tattoos consisting of lines and dots,...
80 percent of the points correspond to those used in acupuncture today." This means the
Prince of Wales ought to start listening to scientists.
Intent Identification
3
• ln(x) ≈π 2 M (1,2ˆ2 - m / x ) - m ln(2). {\displaystyle \ln(x)\approx {\frac {\pi
}{2M(1,2ˆ{2-m}/x)}}-m\ln(2).}
• sin( x ) + 1 3 sin ( 3 x ) + 1 5 sin ( 5 x ) + · · · .
{\displaystyle \sin(x)+{\frac
{1}{3}}\sin(3x)+{\frac {1}{5}}\sin(5x)+\dotsb.}
• c q ( n ) = Σ d | q µ ( q d ) η d ( n ). {\displaystyle c_{q}(n)=\sum_{d\mid q}\mu
\left({\frac{q}{d}}\right)\eta_{d}(n).}
Latex Equation Transla-
tion
4
• How did Japan stumble on for another nine years, borrowing trillions of yen and squan-
dering those trillions on make-work bridges to nowhere and lavish social spending?
Answer: its citizens self-funded its deficits by saving trillions and investing those trillions
in government debt.
• How did our country thrive without income taxes for 126 years? Answer: federal spending
was significantly lower than it is today. In the early 1900s, government spending accounted
for roughly 7% of our GDP; today, federal spending accounts for around 35% of our GDP.
• What was Trump’s biggest persuasion problem in the election? Answer: His opponents
did a great job of framing him as some kind of Hitler.
Question Answering
5
• An isopycnal is a line of constant density. An isoheight or isohypse is a line of constant
geopotential height on a constant pressure surface chart. Isohypse and isoheight are simply
known as lines showing equal pressure on a map. Temperature and related subjects
• Once theory is applied to a mechanical design, physical testing is often performed to verify
calculated results. Structural analysis may be used in an office when designing parts, in
the field to analyze failed parts, or in laboratories where parts might undergo controlled
failure tests. Thermodynamics and thermo-science
• Complex numbers often generalize concepts originally conceived in the real numbers. For
example, the conjugate transpose generalizes the transpose, hermitian matrices generalize
symmetric matrices, and unitary matrices generalize orthogonal matrices. In applied
mathematics Control theory
Topic Classification
6
• (speaking to Elder Fortie): Is this something you always wanted to do? ELDER FORTIE:
Nope. It’s not. SEVERSON: So why are you here? ELDER FORTIE: Because the idea of
having an empty seat in heaven troubles me. SEVERSON: Sister Waymith is from Sweet,
Idaho.
• (speaking to Steve Allen): Are there any countries in particular that you’re really zeroing in
on, you’d really like to make some inroads? ALLEN: Yeah, the United States of America,
North America. We’d like to make more inroads here. SEVERSON: Inroads like the
church has made south of the border. Mexico, in particular, has been fertile ground for
Mormon missionaries.
• (to Elder Russell): Why are you learning Mandarin if you’re going to Canada? ELDER
RUSSELL: I guess there’s a sizable population up there. I mean, everyone deserves to
hear our message, so we’ll go worldwide wherever they are. SEVERSON: This group is
leaving soon for Ukraine. First, they had to be considered worthy of serving a mission.
Dialogue in a Script
7
• Now we can log into our Twilio account and set the Message Request URL to our sms
route via ngrok: Try the app out by texting into your new Twilio number and you’ll get the
response back. Displaying Our Messages We’re now passing our message to the arduino.
The next step is to write the code that examines that message and displays it on our LCD.
Let’s lay the foundation for our app: # include < Wire. h > # include "rgb_lcd.h" rgb_lcd
lcd ; void setup () { Serial. begin ( 9600 ); // set up the LCD’s number of columns and
rows: lcd. begin ( 16, 2 ); lcd. setCursor ( 0, 1 ); // Print a message to the LCD. lcd. print (
"Ricky’s Pager" ); delay ( 1000 ); } void loop () { }
• Now we’re ready to track allocations. The first step is to “hijack” our 3 memory functions
we defined in the first part (lines 4, 11 and 17): void* _Malloc(tU32 Size, tU32 Alloc-
Type, const tChar* Desc, const tChar* File, tU32 Line) { void* Result = malloc(Size);
RegisterAlloc(Result, Size, AllocType, Desc, File, Line); return Result; } void* _Re-
alloc(void* Ptr, tU32 Size, const tChar* File, tU32 Line) { void* Result = realloc(Ptr,
Size); UpdateAlloc(Ptr, Result, Size, File, Line); return Result; } void _Free(void* Ptr) {
UnregisterAlloc(Ptr); return free(Ptr); }
• Here we use the gulp.src API to specify our input files. One thing to note is that we
need to specify a reporter for JSHint. I’m using the default reporter, which should be fine
for most people. More on this can be found on the JSHint website. Compress Images
Next, we’ll set up image compression: gulp. task ( ’images’, function () { return gulp.
src (’src/images/**/*’ ). pipe ( imagemin ({ optimizationLevel : 3, progressive : true,
interlaced : true })). pipe ( gulp. dest ( ’dist/assets/img’ )). pipe ( notify ({ message :
’Images task complete’ })); });
Code Generation
8
• Note: GP = Games played; W = Wins; L = Losses; T = Ties; OTL = Overtime loss;
• SOL = Shootout loss; GF = Goals for; GA = Goals against; Pts = Points National
Conference
• ERA = Earned run average; SO = Strikeouts; +/- = Plus/Minus; PIM = Penalty minutes;
GS = Games Started;
Word Abbriviation
9
• He strode toward her, barely slowly when he reached her. One arm slid around her waist
and the other along her shoulders. She’d barely registered his touch before his mouth
descended upon hers.
• He lifted his head and stared at her. His face paled, and for the first time she noticed a
spattering of orange freckles on his nose and across his cheekbones. He didn’t speak. Just
stared.
• He continued rocking her gently, steadily. Her body’s tremors calmed and her sobs
quietened. He removed his white handkerchief from his trouser pocket, wiping her face.
She didn’t look at him and kept her eyes lowered.
Vivid Description
10
• Even the sugary cereals, they said, are of nutritional value because they contain vitamins
and minerals. Research shows that 40 percent of U.S. children consume their milk via
cereal, said Sutherland of Kellogg. General Mills cites data from the Journal of the
American Dietetic Association that says people who frequently eat cereal, including kids
who eat sweetened ones, tend to have healthier body weights than those that don’t.
• Lead’s toxicity has long been known, and most of the uses that led to human exposure, like
the manufacture of lead paint, have been banned for decades. Lead ammunition consumed
only about 3 percent of the 6.4 million tons of lead used worldwide in 2000, according to
a 2003 report by the Nordic Council of Ministers.
• One reason Tesla has pushed the technology so aggressively is that its battery packs store
more than three times the energy of its competitors’ electric-car batteries. As a result, they
require more power to charge quickly, says Arindam Maitra, a senior project manager at
the Electric Power Research Institute.
Scientific
Evidence
Generation
