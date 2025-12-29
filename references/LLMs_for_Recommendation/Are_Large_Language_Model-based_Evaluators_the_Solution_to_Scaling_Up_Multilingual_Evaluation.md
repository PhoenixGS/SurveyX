# Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation?
Rishav Hada♠ Varun Gumma♠ Adrian de Wynter♠ Harshita Diddee♡∗ Mohamed Ahmed♠ Monojit Choudhury♢ Kalika Bali♠ Sunayana Sitaram♠
♠Microsoft Corporation ♡Carnegie Mellon University ♢MBZUA rishavhada@gmail.com, sunayana.sitaram@microsoft.com
# Abstract
Large Language Models (LLMs) excel in various Natural Language Processing (NLP) tasks, yet their evaluation, particularly in languages beyond the top 20, remains inadequate due to existing benchmarks and metrics limitations. Employing LLMs as evaluators to rank or score other models’ outputs emerges as a viable solution, addressing the constraints tied to human annotators and established benchmarks. In this study, we explore the potential of LLMbased evaluators, specifically GPT-4 in enhancing multilingual evaluation by calibrating them against 20K human judgments across three textgeneration tasks, five metrics, and eight languages. Our analysis reveals a bias in GPT4based evaluators towards higher scores, underscoring the necessity of calibration with native speaker judgments, especially in low-resource and non-Latin script languages, to ensure accurate evaluation of LLM performance across diverse languages.
arXiv:2309.07462v2
# 1 Introduction
Large Language Models (LLMs) can achieve remarkable results on a variety of tasks, sometimes even outperforming humans on certain tasks and domains (OpenAI, 2023; Chen and Ding, 2023; Veen et al., 2023; Chiang and Lee, 2023). However, measuring the performance of LLMs is challenging, as standard NLP benchmarks may not reflect real-world applications. Other hurdles for LLM evaluation include the scarcity of benchmarks for diverse and complex tasks, benchmark saturation, contamination of benchmark data in LLM training data, and the weak correlation between automated metrics and human judgment (Jacovi et al., 2023; Chang et al., 2023; Reiter, 2018; Liu and Liu, 2008). Therefore, researchers have proposed alternative evaluation methods that go beyond benchmarking to assess the abilities and limitations of LLMs (Chang et al., 2023).
∗Work done when the author was at Microsoft
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/158b/158bc778-ca54-4c42-896e-3190045d085b.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Pipeline of our experiments involving generation, evaluation, and calibration.</div>
While LLMs excel at various tasks in English, their capabilities in other languages are more limited. This disparity may increase the digital divide, preventing a significant portion of the global population from benefiting from LLMs and potentially harming them. Ahuja et al. (2023a,b) conduct a comprehensive benchmarking of LLMs across the available multilingual benchmarks covering several tasks and languages, and show that the performance of LLMs degrades significantly on languages that are transcribed in non-Latin scripts and under-resourced languages. Multilingual evaluation is challenging to scale. Certain language families, such as Indo-European, are over-represented in multilingual benchmarks with other language families having very little presence. There is a scarcity of multilingual benchmarks designed to assess tasks that simulate actual LLM usage in real-world scenarios. The metrics used in these benchmarks may be unsuitable for languages with rich morphology or complex writ-
ing systems, as well as phenomena arising from language contact such as borrowing, code-mixing, and transliteration. Evaluation by native speakers is the gold standard for building an accurate picture of model performance, especially in complex tasks without well-defined automated metrics. However, budget constraints, turnaround time, and the lack of easy access to native speakers in some languages all pose challenges in scaling evaluation. This leads to a situation in which LLM performance is unknown for most languages of the world (Ahuja et al., 2022). The success of LLMs in complex tasks such as sentiment analysis, reasoning, problem-solving (Mao et al., 2023; Arora et al., 2023), and providing feedback for reducing LLM harms (Bai et al., 2022) has led to the question of whether LLMs can replace human annotators, or help augment human evaluation (Gilardi et al., 2023). Utilizing LLMs as multilingual evaluators is, therefore, an attractive option to decrease costs and circumvent the challenges of scaling assessments by native speakers. However, LLMs have been demonstrated to have inferior performance even in some high-resource languages and have not been evaluated extensively across many languages on dimensions such as toxicity, fairness, and robustness (due to the absence of such benchmarks) (Ahuja et al., 2023a), it is prudent to proceed with caution. Failing to do so can lead to misleading results which may further widen the digital divide. In this work, we study whether LLM-based evaluation can be the answer to scaling up multilingual evaluation. In other words, can LLMs serve as substitutes or supplements for human native speakers in delivering useful and accurate insights regarding LLM outputs in non-English languages, while considering diverse aspects of interest like linguistic acceptability, task accomplishment, and safety? Our main contributions are as follows: 1. We present the first evaluation of LLMs, specifically GPT-4 as multilingual evaluators to examine whether LLMs can be used to scale up multilingual evaluation. 2. We calibrate LLM judgments on an in-house dataset across three tasks, eight languages, and five dimensions by comparing them to over 20K human judgments on the same tasks, languages, and dimensions. 3. We evaluate a variety of prompting strategies for LLM-based evaluation in the multilingual setting.
4. We provide a framework for evaluating LLMevaluators in the multilingual setting that can generalize across tasks, metrics, and languages1. 5. We suggest best practices and provide recommendations for future work.
4. We provide a framework for evaluating LLMevaluators in the multilingual setting that can generalize across tasks, metrics, and languages1. 5. We suggest best practices and provide recommendations for future work.
# 2 Related Work
Broadly, there are two main uses of LLMs as evaluators: LLMs can be used as alternatives to metrics that compare human and machine-generated text, such as BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004). Word overlap-based metrics are limited, and LLM-based scorers have been shown to outperform them. GPTScore (Fu et al., 2023) is a popular LLM-based framework that can be used to score model outputs based on human-created references along various dimensions. However, these scores still rely on having examples of humancreated reference data. The second use case of LLMs as evaluators is when the LLM is presented with the output of a system (usually an LLM, sometimes the same model) and asked to judge its quality or safety without any human output to compare against (Zheng et al., 2023). The LLM is instructed on how to perform this evaluation with the help of the task description, evaluation rubric, and sometimes, one or more examples in the prompt. This is the use case we focus on in this work. Gilardi et al. (2023) prompt ChatGPT to annotate Tweets across various dimensions such as topic and stance and find that it outperforms crowdworkers. Shen et al. (2023) explore the use of GPT3.5 as an evaluator for abstractive summarization and find that although GPT is a useful evaluator, as the quality of summarization improves, the quality of evaluation degrades. Along similar lines, Wang et al. (2023a) evaluate ChatGPT on various NLG tasks and find that it has a high correlation with human judgments. Kocmi and Federmann (2023) evaluate the effectiveness of LLMs on evaluation of translation quality and find that LLMs starting from GPT3.5 and above achieve SOTA performance on translation evaluation benchmarks. Fernandes et al. (2023) leverage LLMs for finegrained annotation of errors in Machine Translation outputs. LLM-based evaluators have also been used to score and refine outputs they produce, as described in Madaan et al. (2023), ultimately pro1
1Code available at: https://aka.ms/LLM-Eval
ducing outputs that are scored higher on human and automated metrics than the original outputs. Naismith et al. (2023) explore the use of LLMbased evaluators on scoring written discourse for coherence and find a strong correlation with human judgments. The success of LLM-based evaluators has led many to question whether LLM-based evaluation can replace or augment human evaluation (Chiang and Lee, 2023). However, there have been studies showing that LLM-based evaluators can have some biases. Wu and Aji (2023) demonstrate that LLMs tend to prefer answers with factual errors when they are too short or contain grammatical errors. Pangakis et al. (2023) highlight the need for validating LLM-based evaluators on a task-by-task basis. Liu et al. (2023) perform NLG evaluation using GPT-4 and find that although it correlates well with human judgments, it may potentially be biased towards preferring LLM-generated texts. Koo et al. (2023) show that LLMs have egocentric bias where they prefer to rank their own outputs highly in evaluation. Wang et al. (2023b) point out that GPT4-based evaluators have positional bias and scores can be easily altered by changing the order of appearance. There are also several ethical issues with the use of LLMs as evaluators described in Chiang and Lee (2023). Zhang et al. (2023) suggest that wider and deeper LLMs are fairer evaluators, while Chan et al. (2023) introduce a framework for multiple evaluator agents to reach a consensus, mimicking the situation of having multiple annotators. Although there has been some work measuring the calibration of LLM-based evaluators to human judgments (Koo et al., 2023), previous studies have focused on English, and ours is the first work (to the best of our knowledge) that addresses this problem in the multilingual context.
# 3 Experimental Setup
We perform experiments on a text generation application that is powered by GPT-4, and evaluate the following sub-tasks: Open Prompt: This task processes a concise prompt to generate a document adhering to the provided guidelines, producing up to 2, 048 tokens, approximately equivalent to one page in English or Spanish, and marginally less in other languages. Continue Writing: This task takes two textual inputs, termed “left” and “right” to generate a coherent continuation between them, accommodating
up to 1, 000 tokens. Notably, one of the inputs may be omitted. Summarize: Engages in standard summarization by condensing a document of at least 500 words into a succinct summary. It allows for an optional user-defined prompt to tailor the summary format, such as highlighting key points.
We cover the following languages: English (En), French (Fr), German (De), Spanish (Es), Chinese (Zh), Japanese (Ja), Italian (It), Brazilian Portuguese (Pt-Br), and Czech (Cs). Of these, the first six are classified as very high resource languages (Class 5, or “the winners”), while the last three are classified as Class 4 (“the underdogs”) according to Joshi et al. (2020). We plan to extend our study to lower-resource languages in the future. We study the following dimensions of interest: Linguistic Acceptability (LA): This measures whether the text sounds right to a native speaker. The values of this metric are {0, 1, 2}, with 0 corresponding to not acceptable, 1 corresponding to some errors, but acceptable and 2 to perfectly acceptable. We chose LA as opposed to grammaticality to ensure a comparable, native-speaker-led evaluation that did not require formal training in the language. Output Content Quality (OCQ): Whether the general quality of the content is good or not, with values {0, 1, 2}. A score of 0 could indicate that the output is in the wrong language, is repetitive, or sounds like it has been scraped from the web, or translated. A score of 1 indicates that the output is okay in terms of grammar and word choice but still sounds awkward in the language. A score of 2 indicates that the text is of high quality. Task Quality (TQ): This measures the ability of the model to follow the given instructions in the prompt. The values of this metric are {0, 1, 2}, with 0 indicating that the model did not follow the instructions at all. Likewise, a score of 1 indicates that the model followed the instructions approximately well and 2 that it followed perfectly well. The difference between TQ and OCQ is that the latter focuses on whether the content is appealing to a user, while TQ emphasizes the ability of the model to follow the given instructions. Problematic Content (PC): Whether there was any offensive or problematic content in the output. This is a binary metric, with 0 indicating that the output contains this type of content. Hallucinations (H): This measures how well-
grounded the model’s output was to the input content, and/or whether the model output counterfactual information conflicted with the input content. It is a binary metric, with 0 indicating the presence of hallucinations.
# 3.1 Human Evaluation Setup
For creating this in-house dataset, we asked human judges to evaluate the output of LLM-based systems configured to perform the three tasks described earlier. Each entry was annotated by three annotators. They were contracted through an external annotator services company at a starting rate depending on locale ranging from $14 USD/hr and up to $30 USD/hr. The pay was adjusted based on locale and experience level. Each annotator was given 250 texts to judge. We used a subset of the annotated data for our experiments.
# 3.1.1 Annotation Guidelines
We provided annotators with the following information: General instructions about the task (including specific instructions from the prompt) and high-level descriptions of the metrics that we are seeking to evaluate, a description of the file that contained data to be evaluated, and the output format expected. Then we provided detailed descriptions of each metric including the range of values for each metric and examples in English. These examples were provided in the context of different tasks, as each metric could have slightly different interpretations for different tasks.
Table 1 contains the statistics of the human evaluation dataset for the three tasks across the languages we consider. We create a subset of this data for experimenting with prompting variations and its statistics are available in the small column of the aforementioned table. Our full dataset contains over 7, 300 data points, while the smaller subset contains over 2, 700 data points. Each of the data points in our dataset was annotated by 3 annotators.
# 3.2 LLM-based Evaluators
We use the GPT4-32K model as our LLM-based evaluator with a temperature of 0, except in our ablation experiments. The model was accessed through Azure.
Lang.
Open
Prompt
Summarize
Continue
Writing
Agg.
Full
Small
Full
Small
Full
Small
Full
Small
Ca
255
100
158
100
325
-
738
200
De
246
94
251
100
320
96
817
290
En
200
200
200
200
200
200
600
600
Es
247
93
257
100
593
102
1097
295
Fr
221
88
256
99
409
97
886
284
It
256
99
260
100
321
100
837
299
Ja
257
100
259
100
316
102
832
302
Pt-Br
246
94
258
100
327
95
831
289
Zh
255
100
160
99
320
-
735
199
Agg.
2183
968
2059
998
3131
792
7373
2758
Table 1: Dataset statistics across tasks and languages.
Table 1: Dataset statistics across tasks and languages.
# 3.2.1 Prompts
Our evaluation prompts are constructed using the {{guidance}} toolkit2. guidance is a DSL that uses handlebar templating to enable the specification of prompts that interleave instructions and generation with data and logic. This makes it simpler to construct and validate complex prompts.
Evaluation prompts were written to be clear, simple, and not tuned for the data or task. All prompts for evaluation were specified in English, as past work has shown that instructions in native languages can lead to worse performance (Ahuja et al., 2023a). In writing the evaluation prompts, we started with simple unstructured specifications (Natural language sentences with no formatting or styling) and found that it often led to errors in formatting the outputs correctly or even returning all the expected outputs. We found adding styling and formatting, for example, outputting JSON by providing the prompt with a JSON schema for the expected attributes improved the reliability of the LLM outputs. We tried to keep the task and metric description as close as possible to the text that was shown to human annotators for evaluations in the default prompting variation. Each prompt consists of SYSTEM, USER, and ASSISTANT components as shown in Figure 2 in a generic prompt schema. The metric description for Hallucinations is shown in Figure 33.
2https://github.com/guidance-ai/guidance/tree/ main 3Prompts for task description and other metrics are in Appendix A.1.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/020d/020d3d44-3b9a-450e-816b-263dc3ccfac5.png" style="width: 50%;"></div>
# Figure 2: General Prompting Schema.
<div style="text-align: center;">Figure 2: General Prompting Schema.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/91f1/91f18b24-3e9c-4464-a9d4-ac0354bfe352.png" style="width: 50%;"></div>
Figure 3: Metric description for simple instructions (Hallucinations).
<div style="text-align: center;">Figure 3: Metric description for simple instructions (Hallucinations).</div>
# 3.3 Prompting Variations
First, we experiment with variations based on the number of metrics evaluated and instructions provided4. Single Call: In this variation, we call GPT-4 once per metric, without any in-context examples. Compound Call: In this variation, we call GPT-4 once for all the metrics in a single prompt. Single Call - Detailed: In this variation, we call GPT-4 once for all the metrics in a single prompt, with a very detailed metrics description. One of the challenges with LLM evaluation is sensitivity to prompting instructions, which can greatly affect the performance of the LLM on tasks, including evaluation. We experiment with providing detailed instructions for each metric in the prompt. Detailed instruction for Hallucination is shown in Figure 45. We queried GPT-4 to produce these
4All experiments reported in this study are conducted zeroshot unless specified. 5The detailed instructions for all metrics can be found in Figures 15 - 18 in Appendix A.2
instructions by providing it with the instructions given to annotators and manually modifying them.
# 3.4 Calibration with Human Judgments
Inter-annotator Agreement Analysis: We assessed inter-annotator agreement (IAA) among three annotators Annot1,Annot2,Annot3 using Percentage Agreement (PA) to determine the proportion of data points with consistent annotations across annotators. Weighted F1 scores are documented in Table 2. Additionally, Fleiss’ Kappa (κ) values, which offer insights into agreement beyond chance, are provided in Table 3 (Appendix A.3). Since our dataset is skewed towards one or more classes for each of the metrics, κ values can be misleading due to known issues with computing expected agreement in such cases (Eugenio and Glass, 2004). IAA (3 annotators) and GPT: We measure IAA between the majority score of the three annotators and the LLM-evaluator. We refer to this as AnnotAgg,GPT4 and use PA to measure it. Class distribution: We analyze the class distribution of scores across tasks, metrics, and languages to check for potential biases in the dataset and LLM-evaluator. We perform experiments contrasting compound and single-call prompting on the full dataset and zero-shot vs. few-shot prompting on the smaller dataset. We analyze how well-calibrated our LLMbased evaluators are with respect to human judgments by examining PA, and class distribution of scores.
# 3.5 Ablation Experiments
In addition, we perform some ablation experiments to check for consistency, the effect of hyperparameters, and few-shot examples. We perform these ablations on the smaller dataset. Consistency check: We prompt GPT-4 with the same prompt five times to check its consistency. Single Call – Few-Shot: In this variation, we call GPT-4 once per metric, with a few in-context examples. We provide examples in the prompt of human judgments for the same task and metric from a held-out dev set. We take the majority vote from the three human annotations per sample as the aggregate class for that sample to choose our few-shot examples. For each task, language, and metric we choose up to two samples per possible class for that metric. Therefore, we have a minimum of two and a maximum of six exemplars as few-shot examples.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/983e/983e2e73-dfce-4b49-8d0b-14671455de19.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Metric description for complex instructions (Hallucinations).</div>
Name
Annot1
Annot2
Annot3
AnnotAgg
GPT4_joint
AnnotAgg
GPT4_single
AnnotAgg
GPT4_SD
Cs
0.89 ± 0.09
0.81 ± 0.17
0.82 ± 0.16
0.81 ± 0.17
De
0.93 ± 0.07
0.92 ± 0.10
0.93 ± 0.09
0.92 ± 0.09
En
0.98 ± 0.02
0.97 ± 0.03
0.97 ± 0.03
0.96 ± 0.04
Es
0.91 ± 0.08
0.88 ± 0.11
0.89 ± 0.11
0.88 ± 0.11
Lang.
Fr
0.94 ± 0.05
0.90 ± 0.10
0.90 ± 0.10
0.90 ± 0.10
It
0.94 ± 0.07
0.91 ± 0.11
0.92 ± 0.10
0.91 ± 0.11
Ja
0.91 ± 0.08
0.78 ± 0.22
0.78 ± 0.21
0.78 ± 0.22
Pt-Br
0.96 ± 0.04
0.91 ± 0.10
0.91 ± 0.10
0.90 ± 0.10
Zh
0.89 ± 0.10
0.83 ± 0.16
0.83 ± 0.16
0.83 ± 0.16
H
0.98 ± 0.03
0.96 ± 0.04
0.96 ± 0.04
0.96 ± 0.04
LA
0.92 ± 0.06
0.88 ± 0.13
0.89 ± 0.12
0.88 ± 0.12
Metric
OCQ
0.86 ± 0.08
0.80 ± 0.12
0.80 ± 0.12
0.80 ± 0.12
PC
1.00 ± 0.01
1.00 ± 0.01
1.00 ± 0.01
1.00 ± 0.01
TQ
0.88 ± 0.06
0.76 ± 0.15
0.76 ± 0.16
0.75 ± 0.16
Continue
Writing
0.94 ± 0.07
0.88 ± 0.14
0.88 ± 0.14
0.88 ± 0.15
Task
Open
Prompt
0.91 ± 0.08
0.83 ± 0.16
0.84 ± 0.16
0.83 ± 0.16
Summarize
0.94 ± 0.07
0.93 ± 0.09
0.93 ± 0.09
0.93 ± 0.09
For all evaluations, the few-shot examples used are fixed. Sensitivity analysis: We check the sensitivity of the Linguistic Acceptability metric evaluation by randomly shuffling 10% of the words in the whole text for all instances and checking if the LA score provided by the model changes. Temperature variation: We vary the temperature parameter to check its effect on LLM evaluation.
# 4 Results
# 4.1 Percentage Agreement
In this set of graphs, we look at the percentage agreement between LLM-evaluator and the annotators, and between the annotators. We aggregate the
results by task, metric, and language. Figure 5a shows the percentage agreement between the aggregate of the human annotator scores and LLM-evaluator for the full dataset. The figures show both joint (compound), single, and single with detailed instructions prompting techniques for the full dataset. We see that the PA between the annotators and GPT is lowest compared to the PA between the human annotators for Japanese and Czech, with the PA between annotators also being lower for Chinese. Next, we look at PA grouped by metric in Figures 5c for the full dataset with the same prompting variations as before. We find that the PA of the LLM-evaluator with the annotators is lower for the
OCQ metric. We also find that the PA between annotators is relatively low for the TQ metric, while all the PA values are very high for the problematic content metrics. Finally, we look at PA aggregated by task in Figure 5b. We find that PA is lower for the “Continue Writing” task, while the PA between GPT and the annotators is lower than the agreement between annotators for the “Open Prompt” and “Continue Writing” tasks. Overall, we find that the LLMevaluator prompted using the compound prompt has a lower agreement with human annotators than the single prompt variation. Figures 5a, 5b and 5c compare the PA of the LLM-evaluators with detailed instructions vs. the simpler instructions described earlier. We find that PA drops slightly for all metrics with detailed instructions.
# 4.2 Class Distribution
Next, we examine the distributions of the scores from native speakers and the LLM-evaluator. There are three cases to consider for metrics that have three values: Full agreement (all three annotators give the same score), partial agreement (two of the three give the same score), and no agreement (all three give different scores). In metrics that have binary values, we only have full or partial agreement. We group annotations into these classes and analyze responses across these classes. We present results for metrics that have three values (LA, OCQ, and TQ), with 0 corresponding to the lowest score and 2 corresponding to the highest score. In Figures 6a and 6b, we find that the LLM-evaluator provides a score of 2 in most cases, particularly in cases where human annotators disagree. This is even more evident in the case of non-English languages where there is partial agreement or no agreement between the annotators (around 15% of the time on average). Next, we look at languages that are either lowerresourced or not written in the Latin script. In Figures 7a and 7b we find that the LLM-evaluator almost never provides scores of 0 and 1 in the 26% of cases that annotators disagree and find similar results for Japanese and Czech shown in Figures 22e, 22f, 22g and 22h in the Appendix A.4. Overall, we find that LLM-based evaluators give a score of 2 in most cases. While this is consistent with human evaluations in a large part of the dataset, the LLM-based evaluator continues to assign a score of 2 even when humans disagree or provide lower
scores6. Interestingly, even though PA drops slightly for all metrics with the detailed instructions, we find that the LLM-based evaluator may be slightly less biased towards producing high scores with these instructions as shown in Figures 8a and 8b. However, more investigation is needed to determine whether detailed instructions or a different prompting strategy can eliminate the bias toward high scores.
# 4.2.1 Consistency Check
We use a temperature of 0 and receive the same score and justification in each of the five tries, showing that the LLM-evaluator exhibits high consistency.
# 4.2.2 Few-shot Prompting
Figure 24 in Appendix A.7 shows the PA values when few-shot in-context examples are provided. We observe no significant changes in PA values, suggesting that in-context examples might not significantly aid LLM-based evaluators. This also aligns with the findings of Min et al. (2022).
# 4.3 Sensitivity Analysis
As described earlier, we perturb the word order of sentences and check the sensitivity of the Linguistic Acceptability metric on the small dataset. Figure 9 shows the distribution of cases per language per task where the LLM-based evaluator changes its evaluation from a higher score to a lower score. The evaluator shows the most sensitivity to inputs for the Summarization task for all languages except Japanese. For “Continue Writing”, Chinese and Japanese show very little sensitivity. For “Open Prompt", Chinese and Japanese show no sensitivity to the perturbations. One possible explanation for this could be that the evaluator is genuinely less sensitive to these languages. Alternatively, it might be attributed to the flexible word order characteristics of Chinese and Japanese. The examination of tokenizer efficiency in logographic languages, and the exploration of sensitivity across other metrics can be an interesting future exploration.
# 4.4 Temperature Variation
Figure 23 in Appendix A.6 show the PA values for temperatures of 0, 0.3, 0.7 and 1.0. PA reduces as we increase temperature, indicating that a temperature of 0 should be used for LLM-based evaluators. 6Figures for other languages included in Appendix A.4 and A.5.
Figure 23 in Appendix A.6 show the PA values for temperatures of 0, 0.3, 0.7 and 1.0. PA reduces as we increase temperature, indicating that a temperature of 0 should be used for LLM-based evaluators.
6Figures for other languages included in Appendix A.4 and A.5.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/230f/230fbc04-93c0-4da7-9814-9d4d60df5f98.png" style="width: 50%;"></div>
<div style="text-align: center;">a) PA by language: Full dataset (b) PA by task: Full dataset (c) PA by metric: Full datase Figure 5: Percentage Agreement (PA) for different cases and annotator combinations.</div>
<div style="text-align: center;">centage Agreement (PA) for different cases and annotator combination</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/32bb/32bbe4f2-1791-4bfc-a932-b05f57453ae3.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Single Call - English</div>
Figure 6: Class distribution for En and Es. Results are aggregated over all tasks and metrics with 3 classes (LA OCQ, TQ).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c6fd/c6fd91c0-7b08-470e-aa63-6e2fdee59d39.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bab5/bab573d3-6933-4219-b0e6-e20eea72bdf0.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Single Call - Portuguese (Br)</div>
We also observe that increasing the temperature makes the model more susceptible to any noise in the data, making the evaluations highly stochastic and not reproducible.
<div style="text-align: center;">(b) Single Call - Chinese</div>
# 5 Discussion
Overall, our results indicate that GPT-based evaluators have relatively high consistency for nonEnglish languages when set to a temperature of 0.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7313/7313371b-7bc6-47ac-90a5-f100d2591592.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a2f4/a2f48931-4c7a-432d-a7df-6136a6caf0a0.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Single call detailed - Portuguese (Br)</div>
Figure 8: Class distribution for Pt-Br detailed and simple. Results are aggregated for all metrics with 3 classes (LA OCQ, TQ).
Figure 8: Class distribution for Pt-Br detailed and simple. Results are aggregated for al OCQ, TQ).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/de0d/de0d838b-f78f-4f13-84fd-071dfc063689.png" style="width: 50%;"></div>
Figure 9: Percentage of samples where GPT evaluation changed from a higher score to a lower score after perturbation. Note: We do not have Chinese and Czech for the Summarize task in the small dataset.
They also display a fair sensitivity to input variations along the dimension of linguistic acceptability. While LLM-based evaluators show a high Percentage Agreement, there is a noticeable bias towards positive scores, particularly when human opinions differ. It remains uncertain what score an LLMbased evaluator should provide when humans cannot reach a consensus, but consistently high scores in such situations might create a misleading impression of good performance in more challenging evaluations. We find that PA and bias towards higher scores are particularly evident in non-Latin script languages such as Chinese and Japanese, and lower-resource languages such as Czech, which is
consistent with prior work on the performance of LLMs on various tasks (Ahuja et al., 2023a). We experiment with several prompting strategies for LLM-based evaluators and find that evaluating a single metric at a time produces better results than evaluating all metrics in one go, which comes at the cost of having to make multiple calls to the LLM. We also find that providing few-shot examples does not help improve performance. We also provide more detailed instructions to the LLM-evaluator but find that it does not eliminate the problem of bias toward higher scores. In this work, we only use evaluators based on GPT-4. An interesting future direction is the use of smaller models for evaluation or models trained with better coverage of non-English data. We also do not do extensive prompt tuning - future work in this direction includes exploring better prompting approaches including automatically tuning prompts to a held-out set. Our results show that LLM-based evaluators may perform worse on low-resource and non-Latin script languages. Certain metrics corresponding to output quality and task completion may be challenging for LLM-based evaluators. Hence, we advocate for a cautious approach in using LLM-based evaluators for non-English languages and suggest that all LLM-based multilingual evaluations should be calibrated with a set of human-labeled judgments in each language before deployment.
# 6 Limitations
In this work, we utilize a dataset comprising human assessments of a text generation system executing
various tasks in eight languages. As we do not regulate the quality of the system’s output, most of the generated texts receive positive ratings from human evaluators. Consequently, the high Percentage Agreement’s origin remains unclear – whether it stems from the inclination of the LLM-evaluator to assign high scores or not. In future work, we aim to replicate this study using a dataset with a more balanced distribution of human judgments, achieved by controlling the output quality. In this work, we utilize an in-house annotated dataset that, due to restrictions, cannot be released, limiting the reproducibility of our research. However, we intend to make a dataset available to the research community for calibrating LLM-based evaluators in the future. An important research direction is the creation of datasets with good language coverage, multiple annotators per data point, and clear annotation instructions, covering a variety of dimensions to calibrate LLM-based evaluators. Exploring the development of various evaluator personas to represent diverse perspectives of human evaluators and achieve consensus is another research direction that needs further investigation.
# 7 Ethical Considerations
We use the framework by Bender and Friedman (2018) to discuss the ethical considerations for our work.
• Institutional Review: We used an in-house dataset annotated by an external company that has long-standing contracts with the organization and was employed by the organization regularly to do this work.
 Data: The LLM evaluator scores were generated using API calls to GPT-4. The dataset used for calibration is an in-house dataset that will not be released publicly. The dataset was not created with the intent of studying human and LLM calibration; hence, it is not a balanced dataset. Specific instructions were provided to LLMs to avoid generating problematic content, and our ratings of the Problematic Content metrics show no such data; however, the possibility still exists.
# • Annotator Demographics: Annotators were
recruited through an external annotator services company. The pay was adjusted after deliberation with the company, based on the
annotator’s location and expertise. No demographic information is available about the annotators. The annotators are governed by their company’s and our organization’s privacy policy.
• Annotation Guidelines: We draw inspiration from the community standards set for similar tasks. Annotators were given general instructions about the task, detailed instructions about the metrics to be evaluated, and examples in English.
 Methods: In this study, we explore several methods of calibrating human judgments with LLM judgments on various tasks and languages. While these methods can be misused to replace human judgments with LLM judgments, our intent with this study is to highlight the gap between the two and urge the community to proceed with caution.
# References
Kabir Ahuja, Sandipan Dandapat, Sunayana Sitaram, and Monojit Choudhury. 2022. Beyond static models and test sets: Benchmarking the potential of pretrained models across tasks and languages. NLPPower 2022, 10(12):64. Kabir Ahuja, Harshita Diddee, Rishav Hada, Millicent Ochieng, Krithika Ramesh, Prachi Jain, Akshay Nambi, Tanuja Ganu, Sameer Segal, Mohamed Ahmed, Kalika Bali, and Sunayana Sitaram. 2023a. MEGA: Multilingual evaluation of generative AI. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4232–4267, Singapore. Association for Computational Linguistics. Sanchit Ahuja, Divyanshu Aggarwal, Varun Gumma, Ishaan Watts, Ashutosh Sathe, Millicent Ochieng, Rishav Hada, Prachi Jain, Maxamed Axmed, Kalika Bali, and Sunayana Sitaram. 2023b. Megaverse: Benchmarking large language models across languages, modalities, models and tasks. Daman Arora, Himanshu Singh, and Mausam. 2023. Have LLMs advanced enough? a challenging problem solving benchmark for large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7527–7543, Singapore. Association for Computational Linguistics. Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. 2022. Constitutional
Kabir Ahuja, Sandipan Dandapat, Sunayana Sitaram, and Monojit Choudhury. 2022. Beyond static models and test sets: Benchmarking the potential of pretrained models across tasks and languages. NLPPower 2022, 10(12):64. Kabir Ahuja, Harshita Diddee, Rishav Hada, Millicent Ochieng, Krithika Ramesh, Prachi Jain, Akshay Nambi, Tanuja Ganu, Sameer Segal, Mohamed Ahmed, Kalika Bali, and Sunayana Sitaram. 2023a. MEGA: Multilingual evaluation of generative AI. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4232–4267, Singapore. Association for Computational Linguistics. Sanchit Ahuja, Divyanshu Aggarwal, Varun Gumma, Ishaan Watts, Ashutosh Sathe, Millicent Ochieng, Rishav Hada, Prachi Jain, Maxamed Axmed, Kalika Bali, and Sunayana Sitaram. 2023b. Megaverse: Benchmarking large language models across languages, modalities, models and tasks. Daman Arora, Himanshu Singh, and Mausam. 2023. Have LLMs advanced enough? a challenging problem solving benchmark for large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7527–7543, Singapore. Association for Computational Linguistics. Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. 2022. Constitutional
Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. 2022. Constitutional
Emily M. Bender and Batya Friedman. 2018. Data statements for natural language processing: Toward mitigating system bias and enabling better science. Transactions of the Association for Computational Linguistics, 6:587–604. Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. 2023. Chateval: Towards better llm-based evaluators through multi-agent debate. arXiv preprint arXiv:2308.07201. Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Kaijie Zhu, Hao Chen, Linyi Yang, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. 2023. A survey on evaluation of large language models. arXiv preprint arXiv:2307.03109. Honghua Chen and Nai Ding. 2023. Probing the “creativity” of large language models: Can models produce divergent semantic association? In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 12881–12888, Singapore. Association for Computational Linguistics. Cheng-Han Chiang and Hung-yi Lee. 2023. Can large language models be an alternative to human evaluations? In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15607–15631, Toronto, Canada. Association for Computational Linguistics. Barbara Di Eugenio and Michael Glass. 2004. The kappa statistic: A second look. Computational linguistics, 30(1):95–101. Patrick Fernandes, Daniel Deutsch, Mara Finkelstein, Parker Riley, André FT Martins, Graham Neubig, Ankush Garg, Jonathan H Clark, Markus Freitag, and Orhan Firat. 2023. The devil is in the errors: Leveraging large language models for fine-grained machine translation evaluation. arXiv preprint arXiv:2308.07286. Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. 2023. Gptscore: Evaluate as you desire. arXiv preprint arXiv:2302.04166. Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. Chatgpt outperforms crowd-workers for textannotation tasks. arXiv preprint arXiv:2303.15056. Alon Jacovi, Avi Caciularu, Omer Goldman, and Yoav Goldberg. 2023. Stop uploading test data in plain text: Practical strategies for mitigating data contamination by evaluation benchmarks. arXiv preprint arXiv:2305.10160. Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020. The state and fate of linguistic diversity and inclusion in the NLP world. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages
Alon Jacovi, Avi Caciularu, Omer Goldman, and Yoav Goldberg. 2023. Stop uploading test data in plain text: Practical strategies for mitigating data contamination by evaluation benchmarks. arXiv preprint arXiv:2305.10160.
Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020. The state and fate of linguistic diversity and inclusion in the NLP world. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages
6282–6293, Online. Association for Computational Linguistics. Tom Kocmi and Christian Federmann. 2023. Large language models are state-of-the-art evaluators of translation quality. arXiv preprint arXiv:2302.14520. Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park, Zae Myung Kim, and Dongyeop Kang. 2023. Benchmarking cognitive biases in large language models as evaluators. Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics. Feifan Liu and Yang Liu. 2008. Correlation between ROUGE and human evaluation of extractive meeting summaries. In Proceedings of ACL-08: HLT, Short Papers, pages 201–204, Columbus, Ohio. Association for Computational Linguistics. Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. Gpteval: Nlg evaluation using gpt-4 with better human alignment. arXiv preprint arXiv:2303.16634. Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. 2023. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651. Rui Mao, Guanyi Chen, Xulang Zhang, Frank Guerin, and Erik Cambria. 2023. Gpteval: A survey on assessments of chatgpt and gpt-4. arXiv preprint arXiv:2308.12488. Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. Ben Naismith, Phoebe Mulcaire, and Jill Burstein. 2023. Automated evaluation of written discourse coherence using GPT-4. In Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), pages 394–403, Toronto, Canada. Association for Computational Linguistics. OpenAI. 2023. Gpt-4 technical report. Nicholas Pangakis, Samuel Wolken, and Neil Fasching. 2023. Automated annotation with generative ai requires validation. arXiv preprint arXiv:2306.00176. Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia,
Ben Naismith, Phoebe Mulcaire, and Jill Burstein. 2023. Automated evaluation of written discourse coherence using GPT-4. In Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), pages 394–403, Toronto, Canada. Association for Computational Linguistics.
# OpenAI. 2023. Gpt-4 technical report.
Nicholas Pangakis, Samuel Wolken, and Neil Fasching. 2023. Automated annotation with generative ai requires validation. arXiv preprint arXiv:2306.00176.
Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia,
Pennsylvania, USA. Association for Computational Linguistics.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.
# A Appendix
# A.1 Prompts for Simple Instructions
Figure 10 shows task description. Figures 11 - 14 show simple instructions for various metrics.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/796d/796dca10-e128-47d7-bc55-2de01b336d53.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10: Task description.</div>
Figure 10: Task description.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e56a/e56a5708-0f4a-416b-9c99-61870ba58194.png" style="width: 50%;"></div>
Figure 11: Metric description for simple instructions (Linguistic Acceptability).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2ed2/2ed27c81-430f-459c-9060-2274548b81de.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2a60/2a601412-9cbd-43a5-9e0c-51dc18ac7aa3.png" style="width: 50%;"></div>
Figure 12: Metric description for simple instructions (Problematic Content).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/29f6/29f62c21-0570-431d-a955-b8c4cca9855c.png" style="width: 50%;"></div>
Figure 13: Metric description for simple instructions (Task Quality).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/630d/630dcd2e-99e4-40af-a297-04ccba16fa10.png" style="width: 50%;"></div>
Figure 14: Metric description for simple instructions (Output Quality Content).
<div style="text-align: center;">Figure 14: Metric description for simple instructions (Output Quality Content).</div>
# A.2 Prompts for Detailed Instructions
Figures 15 - 18 show complex instructions for various metrics.
A.3 Fleiss’ Kappa
Table 3 shows the Fleiss’ Kappa (κ) on the full dataset for various annotator combinations, aggregated by language, task, and metrics.
# A.4 Class distribution for Metrics with 3 classes
# A.4 Class distribution for Metrics with 3
Figures 19 and 20 show class distribution for various languages, aggregated over metrics with 3 classes - LA, OCQ, TQ.
# A.5 Class distribution for Metrics with 2 classes
Figures 21 and 22 show class distribution for various languages, aggregated over metrics with 2 classes - H, PC.
# A.6 Temperature Variations
Figure 23 shows PA values for different temperature values, results are aggregated over language, task, and metrics.
# A.7 few-shot Results
Figure 24 shows PA values for few-shot prompting, results are aggregated over language, task, and metrics.
“description": “Linguistic acceptability pertains to the degree to which a given language structure (e.g., phrase, sentence, discourse) aligns with the implicit norms and rules of a native speaker’s linguistic intuition. In the study of language, it’s distinct from ’grammaticality’, which is a stricter and narrower concept based on the prescriptive rules of a language. Linguistic acceptability, on the other hand, captures broader native-speaker intuitions and encompasses factors like fluency, idiomacy, and appropriateness in context. In the context of language models, evaluating linguistic acceptability involves assessing the output of the model not just for its adherence to grammar rules, but for its overall fit within the natural, expected, and intuitive contours of fluent human language. The scoring rubric is described below, with a few possible reasons (which might not be exhaustive) for a given score.",
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f038/f03899cf-419f-4cfc-bcd8-e25266ca9496.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7335/7335cf2e-fe98-4e6a-9805-9f5f310348af.png" style="width: 50%;"></div>
Figure 15: Metric description for complex instructions (Linguistic Acceptability).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/421d/421d4d5c-2057-421f-8bfa-c3f8cdada1c1.png" style="width: 50%;"></div>
gure 16: Metric description for complex instructions (Problematic cont
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0f23/0f23ca7a-18d1-4ec7-bed9-fc4d9d23bcf4.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 17: Metric description for complex instructions (task quality)</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/52cd/52cd452d-e653-4851-8b90-9d4fde9657d4.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 18: Metric description for complex instructions (Output content quality).</div>
Name
Annot1
Annot2
Annot3
AnnotAgg
GPT4_joint
AnnotAgg
GPT4_single
AnnotAgg
GPT4_SD
Cs
0.46 ± 0.29
0.05 ± 0.12
0.08 ± 0.17
0.07 ± 0.15
De
0.29 ± 0.29
0.07 ± 0.11
0.13 ± 0.16
0.13 ± 0.15
En
0.47 ± 0.42
0.15 ± 0.22
0.18 ± 0.24
0.11 ± 0.17
Es
0.32 ± 0.22
0.04 ± 0.11
0.04 ± 0.12
0.04 ± 0.11
Lang.
Fr
0.44 ± 0.31
0.12 ± 0.21
0.20 ± 0.23
0.22 ± 0.22
It
0.41 ± 0.33
0.06 ± 0.11
0.08 ± 0.16
0.08 ± 0.14
Ja
0.44 ± 0.33
0.01 ± 0.13
0.02 ± 0.14
0.04 ± 0.15
Pt-Br
0.52 ± 0.37
0.11 ± 0.19
0.09 ± 0.17
0.12 ± 0.20
Zh
0.35 ± 0.32
0.00 ± 0.08
0.01 ± 0.07
0.02 ± 0.07
H
0.40 ± 0.39
0.04 ± 0.15
0.05 ± 0.15
0.08 ± 0.18
LA
0.41 ± 0.24
-0.02 ± 0.06
0.05 ± 0.15
0.09 ± 0.16
Metric
OCQ
0.54 ± 0.19
0.13 ± 0.17
0.16 ± 0.19
0.14 ± 0.17
PC
0.11 ± 0.32
0.00 ± 0.00
0.00 ± 0.00
0.00 ± 0.00
TQ
0.60 ± 0.20
0.18 ± 0.19
0.20 ± 0.21
0.16 ± 0.18
Continue
Writing
0.45 ± 0.33
0.06 ± 0.15
0.07 ± 0.17
0.08 ± 0.16
Task
Open
Prompt
0.49 ± 0.32
0.12 ± 0.19
0.16 ± 0.19
0.15 ± 0.18
Summarize
0.29 ± 0.29
0.02 ± 0.09
0.06 ± 0.15
0.05 ± 0.13
Table 3: Fleiss’ Kappa (κ) values for different cases and annotator combinations on the full da means GPT4_single_detailed
Table 3: Fleiss’ Kappa (κ) values for different cases and annotator combinations on the full dataset. GPT4_SD means GPT4_single_detailed
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e52e/e52eeb55-bca4-4ebe-aae7-a9d0d65bd996.png" style="width: 50%;"></div>
<div style="text-align: center;">(i) Compound call - Italian</div>
<div style="text-align: center;">Figure 19: Class distribution per language (En, Es, Fr, De, It). Results are aggregated over all tasks and metrics with 3 classes (LA, OCQ, TQ).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/35a7/35a73447-e6a0-4aba-9687-dc443d7b1e9c.png" style="width: 50%;"></div>
<div style="text-align: center;">(g) Compound call - Czech</div>
<div style="text-align: center;">Figure 20: Class distribution per language (Pt-Br, Zh, Ja, Cz). Results are aggregated over all tasks and metrics with 3 classes (LA, OCQ, TQ).</div>
Figure 20: Class distribution per language (Pt-Br, Zh, Ja, Cz). Results are aggregated over all t 3 classes (LA, OCQ, TQ).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7242/72420ff8-4656-43e4-a3f6-b78e9d34bc30.png" style="width: 50%;"></div>
<div style="text-align: center;">(i) Compound call - Italian</div>
<div style="text-align: center;">Figure 21: Class distribution per language (En, Es, Fr, De, It). Results are aggregated over all tasks and metrics with 2 classes (hallucinations and problematic content).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7ee9/7ee951e3-ae85-4473-af5b-09169797777b.png" style="width: 50%;"></div>
<div style="text-align: center;">(g) Compound call - Czech</div>
<div style="text-align: center;">Figure 22: Class distribution per language (Pt-Br, Zh, Ja, Cz). Results are aggregated over all tasks and metrics with 2 classes (hallucinations and problematic content).</div>
<div style="text-align: center;">(h) Single Call - Czech</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/073b/073b0901-3f88-4352-899d-e167039e8114.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5dbf/5dbff174-289f-4f41-9556-eff8b3073998.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3179/3179ac9b-fb98-477a-90e9-7260e0843ca4.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2508/250841d8-5ea8-4820-8352-4b3a62886123.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) PA by task with temperature variation</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/913e/913e71de-fca8-4564-8aa7-84340fe1bccd.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) PA by metric with temperature variation</div>
Figure 23: Percentage Agreement (PA) for different cases and temperature variations. Values reported are on the small dataset.
Figure 23: Percentage Agreement (PA) for different cases and temperature variations. Values reported are on the small dataset.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ecec/ececa072-4a61-4946-b706-15182325dd9f.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) PA by metric with few-shot examples</div>
Figure 24: Percentage Agreement (PA) for different cases with few-shot examples. Values reported are on the small dataset.
