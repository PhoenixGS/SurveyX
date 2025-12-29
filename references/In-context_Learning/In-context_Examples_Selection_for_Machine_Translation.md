# -context Examples Selection for Machine Tran
Sweta Agrawal1∗, Chunting Zhou2, Mike Lewis2, Luke Zettlemoyer2, Marjan Ghazvininejad 1 University of Maryland 2 Meta AI sweagraw@umd.edu {chuntinz,mikelewis,lsz,ghazvini}@meta.com
# Abstract
Large-scale generative models show an impressive ability to perform a wide range of Natural Language Processing (NLP) tasks using in-context learning, where a few examples are used to describe a task to the model. For Machine Translation (MT), these examples are typically randomly sampled from the development dataset with a similar distribution as the evaluation set. However, it is unclear how the choice of these in-context examples and their ordering impacts the output translation quality. In this work, we aim to understand the properties of good in-context examples for MT in both in-domain and outof-domain settings. We show that the translation quality and the domain of the in-context examples matter and that 1-shot noisy unrelated example can have a catastrophic impact on output quality. While concatenating multiple random examples reduces the effect of noise, a single good prompt optimized to maximize translation quality on the development dataset can elicit learned information from the pre-trained language model. Adding similar examples based on an n-gram overlap with the test source significantly and consistently improves the translation quality of the outputs, outperforming a strong kNN-MT baseline in 2 out of 4 out-of-domain datasets.
arXiv:2212.02437v1
# 1 Introduction
In-context learning (Brown et al., 2020) has recently received a lot of attention from the NLP research community due to its remarkable ability to utilize only a few input-output examples to perform many NLP tasks (Liu et al., 2021). For example, Lin et al. (2021) demonstrate that a 7.5B multilingual generative model, XGLM, outperforms a supervised sequence-to-sequence baseline in 45 translation directions on the FLORES-101 machine translation benchmark (Goyal et al., 2022) using
just 32 randomly sampled translation examples as demonstrations. While these results are compelling, recent work has also shown that the performance and capability of a pre-trained language model (PLM) can be highly sensitive to many factors, such as the choice of in-context examples (Liu et al., 2022b), their ordering (Lu et al., 2022) and the template (Jiang et al., 2020). Typically, in-context learning for MT uses examples that are randomly sampled from a small development set that resembles the domain of the test dataset. The effect of the aforementioned factors (such as the choice of the examples) on the translation quality of the PLM hence remains unclear and unexplored. Yet another crucial gap in using in-context learning for MT in the current literature is the effect of the domain of in-context examples on translation quality since out-of-domain generalization is a known and important challenge in MT (Koehn and Knowles, 2017). In this work, we systematically analyze how factors such as the choice and the number of few-shot in-context examples and their ordering impact MT output quality. We show that while noisy unrelated 1-shot example can have a significantly adverse effect on translation quality, a single prompt optimized to maximize the translation quality on a development set can sufficiently elicit task-based information from the PLM. Our analysis thus demonstrates the importance of selecting good examples for MT and raises the question: What are the properties of good in-context examples for MT? In that direction, our findings suggest that a well-formed meaning-equivalent translation example results in higher quality translation than randomly selected in-context examples. Furthermore, motivated by the use of Translation Memory in Computer-Aided Translation (Yamada, 2011) and its usage in computational approaches to Machine Translation (Somers, 1999; Koehn and Senellart, 2010; Khandelwal et al., 2020,
inter alia), we retrieve similar examples to the test source from a datastore that includes pairs of source text and their corresponding translations via BM25, an unsupervised efficient retriever to provide additional context to the model. As the context window of the PLM is usually limited (∼3096 tokens, 16 −20 examples), we propose a novel in-contextexample selection and reranking strategy that maximizes the coverage of the source n-grams in the selected examples. Experiments on WMT’19 English↔German and English↔Russian datasets show that our proposed re-ranking strategy can consistently improve the translation quality over the outputs generated using BM25 retrieved examples. Combining optimized 1-shot task-level with example-specific in-context examples using a simple concatenation strategy further improves translation quality, outperforming state-of-the-art inference-adapted nearest-neighbor MT models (kNN-MT) on two out-of-domain datasets (Medical and IT) while being memory and compute efficient as our approach does not require constructing and querying a dense token-level datastore.
# 2 Background: In-context Learning
Generating translations from large-scale multilingual language models like mGPT (Shliazhko et al., 2022), XGLM (Lin et al., 2021) or AlexaTM 20B (Soltan et al., 2022) requires conditioning the decoder-only language model with in-context parallel examples. These examples serve two purposes: a) providing the model with the format and knowledge of the task (task-level) and b) guiding the output generation via providing useful information about the unseen source sentence (example-specific). This is different from the standard sequence-to-sequence models, where the task is always known, and the model learns generalizable patterns from the input-output examples to perform the task (in this case, translation) for the unseen source text. Formally, given k in-context examples {xi, yi}k 1 the prefix input or the prompt, xp j, is generated by concatenating the demonstration examples {(xi, yi)}k 1 to the test input, xs j according to a template, P (see Table 1). The output, ˆy, is then generated via the PLM with parameters θ via greedy decoding as follows:
Table 1: In-context Examples for Machine Translation.
# 3 Prompt Selection
Ideally, good in-context examples can trigger the pre-trained language model to generate the desired output and also elicit the information learned during pre-training (Jiang et al., 2020). Min et al. (2022) show that, for classification tasks, the incontext examples provide information about the task (the distribution of the input text, the label space, and the format of the task) and that the model does not rely on these examples to generate the final output. However, their analysis is limited to a) classification tasks and 2) randomly sampled in-context examples. Prior work has also shown that the order of these in-context examples can also lead to high variance in downstream performance (Zhang et al., 2022). However, less is understood about how these factors impact text generation tasks like MT. Do we need multiple incontext examples? What makes good in-context examples for MT? How sensitive is the model to the order of the prompts? In this work, we aim to better understand the impact of prompt selection on the translation quality of the outputs. Given a training dataset consisting of n parallel examples D = {xi, yi}n i=1, and a test source xj, we select a subset of m informative samples to form a prompt which either provides task-level and/or example-specific information as discussed below.
# 3.1 Task-level In-context Examples
A good task-level in-context example should be able to elicit information learned during pretraining from the PLM. One way to measure the efficacy of an example as a prompt is via computing the translation quality of the outputs generated when prompting the PLM given an example. Hence, we select the task-level prompt as follow: For a
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8bf3/8bf35f68-d3a5-44dc-a638-5012509c9224.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Our proposed strategy can cover all the terms from the input text,“Welche Risiken sind mit Poulvac FluFend H5N3 RG verbunden?”, in this case with just the two examples.</div>
given example sampled from the training dataset, (xi, yi) ∈DS, we create a prompt, xp i by concatenating the example {(xi, yi)} to each source in the development set. The system outputs are then generated using equation 1. We then rank examples from DS as task-level prompts based on the BLEU of the generated outputs against the references on this held-out development set, Ddev = {X, Y }:
(2)
# 3.2 Example-specific In-context Examples
Prior work on retrieving good in-context examplespecific prompts for tasks other than MT (like question answering or knowledge retrieval) either trains a dense-retriever (Rubin et al., 2021) or utilizes samples that are closer to the test source in the embedding space of a PLM like BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), or XLNET models (Liu et al., 2022b). While contextual models can generate a global sentence representation, they overlook rare lexicons which can be important for generating translations in unseen domains like medical or IT (Wrzalik and Krechel, 2021). However, for MT, overlapping n-grams between the source and the retrieved sentences ensures informativeness as the target associated with the retrieved sentence is likely to include partial translations of the source. We can thus use BM25 as an efficient unsupervised retrieval method to retrieve similar examples. However, as the examples are scored independently and BM25 favors rare
word matches (Robertson et al., 2009), the top retrieved candidates might not cover all the terms in the source text (Figure 1). Given that the context window of the PLM is usually limited (∼3096 tokens, 16 −20 examples), maximizing the coverage of all the terms found in the test input might be favorable. Hence, we propose to re-rank the top 100 candidates retrieved from BM25 using our algorithm outlined in 1. We extract all the word n-grams, and their counts from the test source, xs j and source of the BM25 retrieved examples, {Pj(xi}k 1 (lines 2-4). Let S and Q denote the set of the source n-grams and the n-grams from a BM25 retrieved example, respectively. We compute a recall-based (R) n-gram overlap score (line 7) using the following equation:
(4)
The example with the maximum score is then added to the set of selected prompts, and the found n-grams from the test source are then downweighted by a factor, λ for the next iteration of selection (line 14). For example, setting λ = 0 will select the example that covers the n-grams from the test source in the subsequent iteration that has not already been encountered. This process is then re-
Algorithm 1: An N-gram Recall-based Strategy to Re-rank In-context Examples
Input: Prompts {Pj(xi, yi)}k
1 for the test source xs
j, λ, Threshold
Output :Ordered Selected Prompts {T = Pj(xi, yi)}s
1, s ≤k
1 T ←Empty Ordered List
2 S ←EXTRACT_WORD_NGRAMS_WITH_COUNTS (xs
j)
3 for i ∈{1..k} do
4
Q[i] ←EXTRACT_WORD_NGRAMS_WITH_COUNTS (Pj(xi))
5 while True do
6
for i ∈{1..k} do
7
Score[i] ←NGRAM_OVERLAP_SCORE (S, Q[i])
8
if max(Score) < Threshold then
9
break
10
T.append(Pargmax(Score))
11
matched_ngrams ←S ∩Q[argmax(Score)]
12
Q[argmax(Score)] ←∅
13
for ngram ∈matched_ngrams do
14
CountS(ngram)× = λ
15 Return T
peated over the retrieved pool until a set threshold of the score is reached. Figure 1 shows the top-100 candidates retrieved via BM25 for the input: “Welche Risiken sind mit Poulvac FluFend H5N3 RG verbunden?”. The top few candidates provide the same information to the PLM, i.e., translation of the phrase “Poulvac FluFend H5N3 RG”. The examples including the other terms (“Welche Risiken sind mit verbunden ?”) from the input text, are ranked lower. On the other hand, our proposed re-ranking strategy can cover all the terms from the input text, in this case, with just the top-2 examples.
# 4 Evaluation Settings
# 4.1 Datasets and Evaluation Metric
We perform our in-domain evaluation on the WMT19 German (de) ⇔English (en) and WMT-19 Russian (ru) ⇔English (en) datasets (Barrault et al., 2019). For the out-of-domain evaluation, we use the multi-domain dataset from Aharoni and Goldberg (2020) for the following domains: Medical, Law, IT, and Koran. The dataset statistics are included in the Appendix (Table 11). We normalize punctuation using the Moses toolkit (Koehn et al., 2007) and remove sentences longer than 250 tokens as well as sentence pairs with a source/target length ratio exceeding 1.5 from the in-domain datasets.
We evaluate the detokenized length truncated outputs generated by the model using sacreBLEU (Post, 2018).1 The generated outputs from the PLM are truncated to twice the source length, as preliminary analysis suggested degeneration in a few (∼10-20) examples.
# 4.2 Experimental Conditions
Language Model We use the publicly available checkpoint of the XGLM7.5B, a decoder-only causal language multilingual model (Lin et al., 2021) for all our experiments, which has 32 layers and a hidden dimension of 4096.
# Baselines and Comparisons We consider the following comparisons:
• Random: p random few-shot examples sampled from the training dataset (number of trials=3).
• Task-level: top-p examples that achieve the highest BLEU on the development set (§ 3.1).
• Retrieved In-context (BM25): qmax examples retrieved via BM25, since unlike task-level examples, there is no guarantee that exactly q similar examples will be found in the training dataset for each input.
Method
p + qmax
En-De
De-En
Ru-En
En-Ru
Avg.
Task-level
1 + 0
23.35
32.16
30.48
25.04
27.75
BM25
0 + 1
19.17
25.82
24.54
21.51
22.76
R-BM25
0 + 1
20.60
28.19
27.26
21.92
24.49
Random (Baseline)
16 + 0
24.48
31.26
30.38
25.67
27.95
Task-level
16 + 0
23.72
31.22
30.89
27.27
28.28
BM25
0 + 16
26.58
32.16
31.44
28.54
29.68
R-BM25
0 + 16
27.07
32.59
31.85
28.90
30.10
R-BM25
0 + 17
27.00
32.68
31.88
28.80
30.09
Task-level + R-BM25
1 + 16
27.09
33.24
31.90
29.50
30.43
Table 2: Results on WMT’19 test sets: Concatenating task-level prompt to R-BM25 consistently achieves the best BLEU scores across the board. p and qmax are the number of task-level and example-specific prompts respectively.
• Retrieved Re-ranked In-context (R-BM25): qmax re-ranked examples using our proposed approach as detailed in § 3.2.
We additionally compare our results with kNNMT (Khandelwal et al., 2020) for out-of-domain evaluation. We use λ = 0.1, threshold=1.0 and order the examples according to their similarity to the source, with the most similar examples on the left in all our experiments based on an initial hyperparameter search on the development dataset (Appendix Tables 12,13).
# 5 Results
Table 2 and 3 summarizes our main results for the in-domain evaluation when translating between English <-> German and English <-> Russian and the four German-English out-of-domain datasets, respectively.
# 5.1 In-domain Evaluation
A single task-level prompt is competitive with 16 random few-shot examples. Our experiment suggests that it is possible to elicit the task-level knowledge from the large-scale language model using a single prompt as opposed to using 16 random few-shot examples when translating into English (Table 2). Using a single task-level prompt (optimized on the development set) improves BLEU over using 16 random few-shot examples for 2 out of 4 translation directions (De-En, Ru-En). We hypothesize that when translating out of English, the model still benefits from getting exposed to multiple and diverse random few-shot examples as the target language model is relatively weaker.
Multiple example-specific prompts are required to improve translation quality over a single task-level prompt. Using a single task-level (p = 1) prompt attains higher BLEU over using a single example-specific prompt (q = 1; BM25, RBM25) across the board. By contrast, using upto 16 BM25 prompts (qmax = 16) significantly improves output quality over using task-level prompts, with an average gain of 1.41 in BLEU.
Re-ranking BM25 retreived examples improves BLEU. Our proposed re-ranking strategy consistently improves BLEU across the board over BM25 for both values of qmax = {1, 16} showing that both the order and the choice of the in-context examples matters. Both task-level and R-BM25 examples provide complementary advantages, as combining them using a simple concatenation strategy improve output quality over task-level or R-BM25 examples. We leave the exploration of optimizing the number and the joint order of task-level and example-specific prompts to future work.
# 5.2 Out-of-domain Evaluation
As XGLM is trained on monolingual Common Crawl snapshots, translation in any domain and language could be considered an out-of-domain task from the model’s perspective. However, we hypothesize that translation in specific domains like medical, law, or IT could still be challenging for the PLM as the model is less likely to have observed even sufficient monolingual datasets for these specialized domains, in contrast to the news text found in WMT. Examples from these domains might require translating rare terminologies
Method
Corpus
p + qmax
MEDICAL
LAW
IT
KORAN
Avg.
Task-level
Domain-specific
1 + 0
31.23
32.10
28.70
14.68
26.68
WMT
30.08
31.10
26.72
13.19
25.27
R-BM25
Domain-specific
0 + 1
52.62
55.46
40.54
13.76
40.60
Task-level
Domain-specific
16 + 0
32.65
33.68
28.81
15.30
27.61
WMT
30.14
30.76
26.19
12.72
24.95
R-BM25
Domain-specific
0 + 16
56.43
59.57
46.57
17.49
45.02
R-BM25
Domain-specific
0 + 17
56.65
59.55
46.64
17.48
45.08
Task-level + R-BM25
1 + 16
56.76
59.56
47.50
17.55
45.34
kNN-MT
-
-
54.35
61.78
45.82
19.45
45.35
Table 3: Results on the Multi-Domain Test Set: Prompting XGLM with R-BM25 in-context examples outpe forms kNN-MT on 2 out of 4 domains.
and carry domain-specific idiosyncrasies, which is known to pose a challenge even for a well-trained supervised neural MT model (Koehn and Knowles, 2017). Hence, we also evaluate PLM under these specialized out-of-domain scenarios.
Domain of few-shot in-context examples matter. Task-level in-context examples drawn from the domain of evaluation, i.e., domain-specific, obtain on an average higher BLEU scores across the board than using examples from a distant WMT corpus as expected (Table 3) in both 1-shot (p = 1: +1.4) and 16-shot (p = 16: +2.7) settings.
Example-specific prompts significantly improve translation quality over task-level prompts. Unlike the in-domain evaluation, retrieved and re-ranked example-specific prompts (R-BM25) improve the translation quality significantly across the board with up to 23 BLEU gain in the Law domain using just a single example as a prompt over a task-level prompt. This can be attributed to the high lexical overlap in the examples retrieved from the training data for these domains (Table 8).
# Task-level and R-BM25 prompts are comple-
mentary. Both task-level and R-BM25 provide supporting information for a given test source sentence as concatenating these set of prompts improves output quality over using these methods independently, outperforming a strong kNN-MT baseline on 2 out of 4 domains (Medical and IT) without requiring access to a strong base MT model or token-level retrieval during inference. Our manual analysis suggests that the higher gain obtained in the IT domain (+0.86) when using both task-
level and example-specific prompts can be explained by the observation that for 100 test source sentences, there are no training examples with any lexical overlap with the test source. The task-level prompt can still elicit learned information from the PLM over using no examples for these inputs.
# 6 Analysis
# 6.1 Task-level Example Selection
Choice of Few-shot Examples We show the distribution of output quality as measured by BLEU when using 100 different examples as prompts in Figure 2. Across all four language pairs, there is a large variation in BLEU scores (up to 20 BLEU), where noisy or unrelated prompts can lead to significantly worse output quality. Given that most existing parallel corpora are web-crawled and the quality of bitext can vary significantly across different language pairs (Kreutzer et al., 2022), randomly sampled examples can under-estimate the translation quality attainable by prompting these PLM.
# Impact of Pool Size on Task-level Prompt Selec-
tion We select the best task-level prompt based on the translation quality on the development set from a random sample of 100 examples (pool) as detailed in Section 3.1. However, one concern regarding the selection of the best task-level prompt in this fashion could be that we might still be underestimating the PLM (s) performance, as a larger pool size could result in better output quality. We study the impact of using a larger pool size in Table 5 where increasing the number of examples from 100 to 1000 only leads to a gain of 0.5 points in the maximum BLEU. From the same table, we
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a701/a70145ba-29e7-4904-9a5a-35cc65a08dbb.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: BLEU distribution on the WMT’18 test set for 100 randomly sampled 1-shot prompts from the training dataset. The same set of 100 random 1-shot prompts are used for x→y and y →x translation directions.</div>
Features
En-De
De-En
En-Ru
Ru-En
% (Aligned words)
Random (T=3)
0.818 ±0.009
0.837 ± 0.015
0.594 ±0.106
0.663 ±0.051
Task-level
0.834
0.926
0.773
0.886
Prism-Src
Random (T=3)
-1.027 ±0.068
-1.081 ±0.016
-2.214 ±0.139
-1.767 ±0.111
Task-level
-0.843
-0.847
-1.557
-1.206
Table 4: Average scores obtained by top-10 1-best prompts and 10 Random 1-shot prompts on features quantifyi semantic equivalence/translation quality (higher is better).
1-shot Prompts
100
1000
Max
35.82
36.29
Mean
34.06
29.95
Stdev
0.96
9.55
Random 10 trials of best over 100 1-shot Prompts
Mean over Max
-
36.08
Stdev over Max
-
0.18
Table 5: Task-level example selection from 1000 1-shot Prompts on the WMT’19 development dataset.
can also observe that for any subset of random 100 few-shot examples, we can extract a task-level prompt (BLEU: 36) with a small standard deviation in overall outpust quality (0.18).
Translation direction Figure 3 shows the correlation between output quality in forward (x →y)
and reverse (y →x) translation directions when using 1-shot prompts — there is a moderate to high correlation in output quality for both language pairs suggesting that the best and worst 1-shot prompts in one direction exhibit similar behavior in the opposite translation direction.
Properties of good Task-level prompts Our manual analysis on the best task-level prompts suggests that any well-formed and meaning-equivalent translation (Vyas et al., 2018; Briakou and Carpuat, 2020) could make a good task-level prompt (see examples in Appendix Table 14). To quantify the meaning equivalence of the 1-best task-level prompt against random 1-shot examples, we report the percentage of aligned words between the source and reference translation (“% Aligned words”) using fastAlign (Dyer et al., 2013) and the probability of generating the reference translation condi-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d09c/d09c4835-fd48-4e46-bbf2-04186feea9e4.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Best 1-shot prompts work equally well for both translation directions.</div>
tioned on the source using a pre-trained multilingual NMT model, Prism-src (Thompson and Post, 2020; Agrawal et al., 2021) in Table 4.2 Across all language pairs and both metrics, task-level examples achieve higher semantic similarity scores than random 1-shot examples suggesting that task-level examples are relatively more equivalent in meaning than random examples.
Prompt
BLEU
Best
32.14
Worst
1.12
Best + Worst
25.54
Worst + Best
31.43
Table 6: BLEU when using a equivalent translation against a noisy unrelated parallel example on the WMT18 De-En Development Dataset.
Impact of Noise Table 6 shows the impact on translation quality when using an unrelated translation example against a meaning-equivalent or wellformed source-translation pair as a prompt: a noisy unrelated in-context example leads to extremely low BLEU score of 1. While using both a good
task-level and a noisy prompt via concatenation (best-then-worst and worst-then-best) reduces the effect of noise, the best output quality achieved when using both prompts (31.43) is still lower than using just the best task-level prompt (32.14), highlighting the importance of both prompts selection and their ordering on translation quality. Impact of Ordering To further explore the sensitivity to the order of few-shot prompts on MT quality, we use all possible order permutations of four randomly sampled examples and the top four task-level examples as prompts (4!) and report the translation quality as measured by BLEU in Table 7: Task-level prompts are less sensitive to prompt order, as suggested by the lower standard deviation achieved in all settings, and result in higher translation quality than randomly selected examples. Across the three different runs of randomly sampled examples, there is a significant difference in BLEU, further corroborating that the choice of incontext examples and their ordering matters.
En-De
De-En
En-Ru
Ru-En
34.43 ±0.25 25.19 ±0.26 12.48 ±5.72 15.56 ±0.50
Random
35.63 ±0.48 25.85 ±0.15 24.99 ±0.21 19.04 ±0.39
34.73 ±0.30 23.93 ±0.28 10.92 ±4.64 17.91 ±0.07
Optimized 35.95 ±0.24 26.98 ±0.15 25.85 ±0.11 19.96 ±0.24
Table 7: BLEU over all 24 permutations of 3 seeds of 4 randomly selected and top 4 task-level prompts.
# 6.2 Informativeness of Example-specific
# 6.2 Informativeness of Example-specific Prompts
To understand the benefit of retrieved examples in the out-of-domain evaluation, we measure the lexical overlap between the test input (x, y) and the prompts (Ix, Iy) using BLEU (Avg. BLEU (Ix, x), Avg. BLEU (Iy, y)), where Ix and Iy are the sources and target translations of the retrieved incontext examples. We also report the correlation against the translation quality BLEU(ˆy, y). Table 8 shows that the source lexical overlap is a good indicator of the informativeness of a prompt for 3 out of 4 domains, with Koran as an exception. For Koran, while the retrieved sentences have a high overlap with the source (36.03), the target associated with the prompts (Iy) does not get high BLEU with the reference (10.36) compared to other domains. We hypothesize that this might be due to a bias in the reference translations towards a particular output style. We provide an analysis of
Medical
35.785
0.593
32.101
0.777
Law
34.982
0.677
34.349
0.786
IT
25.196
0.497
19.382
0.669
Koran
36.033
-0.016
10.364
0.676
Table 8: Correlation between the degree of overlap as measured by BLEU and the translation quality of the outputs BLEU(ˆy, y), across different domains when using the top-1 prompt retrieved using BM25. Ix and Iy are the sources and the reference translations in the BM25 examples respectively.
the impact of this phenomenon on MT quality in Section 7.
# 7 Output Analysis
We report two interesting findings when prompting PLM with task-level and example-specific prompts:
Stylistic Outputs One advantage of using a single task-level in-context example to prompt the PLM is that it allows us to systematically study how the choice of prompt influences the style of the generated translation. Table 9 illustrates one such example: we can observe that as the prompt includes a contraction (“we are” vs. “we’re”), the outputs generated by the PLM also include contractions and can be incorrectly penalized by BLEU while being meaning equivalent.
Prompt: Wegen des heißen Sommers fangen wir erst spät an. = Because of the hot summer, we’re late getting started. Source: Ja, ich bin sehr zufrieden mit dem Auftritt. Reference: Yes, I am very happy with the performance. PLM Output: Yes, I’m very satisfied with the performance. Source: Es ist eine andere Unternehmenskultur. Reference: It is a different corporate culture. PLM Output: It’s a different corporate culture.
Template-based MT Template-based translation in medical, legal, it, or e-commerce domain can be preferable as they reduce the risk of generating errors in automatically generated translations. We present some examples in Table 10 on how PLM can seamlessly use retrieved prompts to synthesize a translation from the template provided.
Prompt: WIE IST SINGULAIR ANZUWENDEN? = HOW TO TAKE SINGULAIR Source: WIE IST EVOLTRA ANZUWENDEN? PLM Output: HOW TO TAKE EVOLTRA Prompt: Zeigt die aktuelle Datei mit Opera an. = View the current file with Opera. Source: Zeigt die aktuelle Datei mit Lynx an (Textbasierter Browser). PLM Output: View the current file with Lynx (Text-based browser).
Table 10: Outputs follow the template of the prompt.
# 7.1 Size of the Datastore
Figure 4 shows BLEU when varying the size of the datastore used to retrieve similar in-context examples using BM25 on the Medical dataset. As the size of the datastore increases, the likelihood of retrieving a more similar example increases. However, similar output quality in BLEU can be achieved by using multiple in-context examples when a smaller in-domain datastore is available as multiple examples can provide better coverage of the source terms — BLEU @q=16 with a datastore size of 100k is equivalent to BLEU @q=1 with twice as many examples (200k).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6f03/6f0311b4-6ab2-42c6-a809-5daffa843a6d.png" style="width: 50%;"></div>
Figure 4: BLEU on the Medical dataset when varying the size of the datastore and the number of BM25 incontext examples.
# 8 Related Work
In-context Learning for MT Garcia and Firat (2022) use natural language prompts (e.g. Translate to {language_name}: {text}) to control the target language in multilingual MT and investigate the impact of scale, number of languages, and their similarity for this phenomena. Wang et al. (2022) utilize BM25 retrieved training examples in a supervised fashion to learn from similar examples during training. Contrary to prior work, we utilize similar examples to form a textual prompt which is used to guide the generation of translation output during inference and systematically study the properties of good in-context examples for MT. Domain Adaptation for MT Prior work on domain adaptation for machine translation uses outof-domain bilingual or monolingual datasets to improve the translation quality of a pre-trained neural sequence-to-sequence MT model either during training (Luong and Manning, 2015; Freitag and Al-Onaizan, 2016; Wang et al., 2017) or inference (Zheng et al., 2021; Khandelwal et al., 2020). Similar to past work, our work utilizes out-of-domain datasets during inference to adapt a pre-trained generative language model to improve the translation quality on unseen domains. However, our approach does not rely on creating a domain-specific tokenlevel datastore, but directly uses similar examples to provide additional context, hence is more compute and memory efficient. Prompt Selection The importance of selecting good in-context examples and their impact on downstream NLP task performance has been studied in prior work (Liu et al., 2022b; Lu et al., 2022; Jiang et al., 2020; Min et al., 2022; Zemlyanskiy et al., 2022; Rubin et al., 2021; Liu et al., 2022a). However, how these examples and their properties impact MT quality remains unexplored, which we investigate in our work.
# 9 Conclusion
We investigate the choice of in-context examples selection for MT in both in-domain and out-ofdomain settings. We propose a novel recall-based re-ranking approach to utilize similar training examples as prompts and show their efficacy across multiple datasets and domains. Our findings show that task-level prompts can provide a complementary advantage to example-specific prompts, outperforming a strong kNN-MT baseline in 2 out
of 4 out-of-domain datasets while being memory and compute efficient. Our manual analysis of the generated outputs reveals that the PLM can mimic the style of the in-context examples provided and can be used for template-based translation synthesis. These results open space for future research to evaluate the potential of generating diverse and style-specific outputs for MT.
# References
Sweta Agrawal, George Foster, Markus Freitag, and Colin Cherry. 2021. Assessing reference-free peer evaluation for machine translation. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1158–1171.
Roee Aharoni and Yoav Goldberg. 2020. Unsupervised domain clusters in pretrained language models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7747– 7763, Online. Association for Computational Linguistics.
Loïc Barrault, Ondˇrej Bojar, Marta R. Costa-jussà, Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Philipp Koehn, Shervin Malmasi, Christof Monz, Mathias Müller, Santanu Pal, Matt Post, and Marcos Zampieri. 2019. Findings of the 2019 conference on machine translation (WMT19). In Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1), pages 1–61, Florence, Italy. Association for Computational Linguistics.
acob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.
glance: An audit of web-crawled multilingual datasets. Transactions of the Association for Computational Linguistics, 10:50–72.
Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, et al. 2021. Few-shot learning with multilingual language models. arXiv preprint arXiv:2112.10668.
Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin Raffel. 2022a. Few-shot parameter-efficient finetuning is better and cheaper than in-context learning. arXiv preprint arXiv:2205.05638.
iachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2022b. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pages 100–114, Dublin, Ireland and Online. Association for Computational Linguistics.
than you think: A simple and effective method by retrieving from training data. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3170–3179.
# A Statistics of Datasets
Table 11 includes statistics of training, development and test sets used for the experiments discussed in the paper.
Dataset
Train
Dev
Test
WMT-19 (de)
42M
2998
2000
WMT-19 (ru)
10M
3000
2000
Multi-Domain
Medical
248K
2000
2000
Law
467K
2000
2000
IT
223K
2000
2000
Koran
17K
2000
2000
# B Compute Infrastructure & Run time
Each experiment is run on a single Nvidia Tesla V100 Volta GPU machine with 32G Ram. A single inference experiment on 2000 test examples using XGLM with 16 in-context examples takes around 3-4 hrs to complete.
# C Results using Second Metric: Comet
We report translation quality using Comet (Rei et al., 2020) in Tables 15 and 16. We use the eamt22-cometinho-da model (Rei et al., 2022) to generate the scores as it was shown to achieve higher correlations with human judgments than lexical overlap metrics while being computationally efficient. Our re-ranking strategy (with qmax = 16) consistently performs the best across the board except for Koran, outperforming strong kNN-MT baselines on the multi-domain test set in 3 out of 4 settings. Adding a task-level prompt to 16 R-BM25 prompts via concatenation further improves quality in 5 out of 8 settings.
# D Hyperparameter Search
# D.1 Order of BM25 Retrieved Examples
We report the BLEU when using two different orderings of example-specific prompts on the development set for the medical domain. Ordering the examples with the most similar examples on the left attains higher BLEU than the right-to-left order. We note that the trend could vary depending on the noise in the training dataset, the degree of similarity, and the number of retrieved examples. We leave
the exploration of the ordering of example-specific prompts to future work.
λ
BLEU
Left-to-right
56.84
Right-to-left
54.97
Table 12: BLEU using twi different orderings of the top-16 example-specific BM25 prompts on the Medical development Set.
# D.2 Choice of λ, Threshold
Table 13 shows the BLEU and the average number of in-context examples selected when varying λ and the threshold described in Section 3.2. We select λ = 0.1 and threshold value of 1.0 as it achieves the best BLEU on the Medical development set as shown below:
λ
Threshold
BLEU
Avg. # of Examples
0.1
0.1
54.55
14.16
1.0
54.56
12.73
5.0
53.35
8.83
0.3
0.1
54.47
15.06
1.0
54.51
14.28
5.0
53.98
10.32
0.5
0.1
54.44
15.44
1.0
54.39
15.10
5.0
54.44
11.85
Table 13: BLEU using different values of λ and threshold on the Medical Development Set (qmax = 16).
# E Example Task-Level Prompts
Table 14 shows the best task-level in-context example selected by our method described in § 3.1 and the respective BLEU scores on the development set for the German-English and Russian-English tasks.
German: Beispielsweise der Änderungsantrag zu Artikel 5 in der Stellungnahme des Ausschusses für Landwirtschaft und ländliche Entwicklung weist klar und deutlich darauf hin, dass die Verschlechterung der Qualität des Bodens lokale oder regionale Ursachen und Wirkungen hat und daher unbedingt nationale statt europäischer Maßnahmen ergriffen werden müssen. English: For example, the amendment to Article 5 in the opinion of the Committee on Agriculture and Rural Development clearly indicates that the degradation of the soil has local or regional causes and effects and it is therefore essential to adopt national as opposed to European measures. Development BLEU: 35.82 Russian: Если ваш браузер возвращает ранее сохраненный “cookie”, то управляющий им поставщик имеет возможность соединить актуальное посещение пользователя с предыдущими посещениями, но только в отношении своего содержания. English: If the browser sends back an earlier saved cookie, then the service managing these can connect to the user´s earlier visit, but only in respect of their own content. Development BLEU: 25.63
Table 14: Best task-level prompt For De-En and Ru-En Language Pairs according to the BLEU development set.
Method
p + qmax
En-De
De-En
Ru-En
En-Ru
Task-level
1 + 0
0.354
0.403
0.428
0.626
BM25
0 + 1
0.107
0.149
0.139
0.346
R-BM25
0 + 1
0.204
0.249
0.244
0.413
Random-Avg
16 + 0
0.387
0.391
0.424
0.636
Task-level
16 + 0
0.389
0.381
0.440
0.662
BM25
0 + 16
0.423
0.410
0.434
0.673
R-BM25
0 + 16
0.438
0.420
0.444
0.677
R-BM25
0 + 17
0.440
0.421
0.448
0.676
Task-level + R-BM25
1 + 16
0.434
0.430
0.447
0.694
<div style="text-align: center;">Table 15: Comet Scores on WMT’19 test sets.</div>
Method
Corpus
p + qmax
MEDICAL
LAW
IT
KORAN
Results from Jiang et al. (2022)
Vanilla kNN-MT
-
-
0.548
0.662
0.531
-0.014
Their model
-
-
0.578
0.703
0.585
0.047
Task-level
Domain-specific
1 + 0
0.314
0.320
0.240
-0.068
WMT
0.277
0.345
0.146
-0.113
R-BM25
Domain-specific
0 + 1
0.464
0.553
0.389
-0.216
Task-level
Domain-specific
16 + 0
0.369
0.365
0.222
-0.047
WMT
0.297
0.399
0.098
-0.131
R-BM25
Domain-specific
0 + 16
0.697
0.697
0.666
-0.105
R-BM25
Domain-specific
0 + 17
0.699
0.697
0.667
-0.104
Task-level + R-BM25
1 + 16
0.701
0.699
0.721
-0.095
