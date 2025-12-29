# Enhancing In-Context Learning with Answer Feedback for Multi-Span Question Answering

Zixian Huang, Jiaying Zhou, Gengyang Xiao, and Gong Cheng

State Key Laboratory for Novel Software Technology, Nanjing University, China {zixianhuang,jyzhou,181840264} @smail.nju.edu.cn, gcheng@nju.edu.cn

Abstract. Whereas the recent emergence of large language models (LLMs) like ChatGPT has exhibited impressive general performance, it still has a large gap with fully-supervised models on specific tasks such as multispan question answering. Previous researches found that in-context learning is an effective approach to exploiting LLM, by using a few task-related labeled data as demonstration examples to construct a few-shot prompt for answering new questions. A popular implementation is to concatenate a few questions and their correct answers through simple templates, informing LLM of the desired output. In this paper, we propose a novel way of employing labeled data such that it also informs LLM of some undesired output, by extending demonstration examples with feedback about answers predicted by an off-the-shelf model, e.g., correct, incorrect, or incomplete. Experiments on three multi-span question answering datasets as well as a keyphrase extraction dataset show that our new prompting strategy consistently improves LLM’s in-context learning performance.

# 1 Introduction

Recently, the rise of large language models (LLMs) [5,22,21] represented by ChatGPT 1 provides a new paradigm for NLP research, which can perform well using only natural language instructions rather than being trained on the target dataset. Based on LLMs, many tasks are expected to be more convenient and accessible to users with different needs, including  multi-span question answering (MSQA). MSQA aims to automatically find one-to-many answers at the span level for a given question, which has attracted many in-depth research works [15,26] based on pre-trained language models (PLMs), and has broad application scenarios such as medical question answering [34,11]. However, compared with PLMs fine-tuned on the complete training data, LLMs still have a large gap on difficult MSQA datsets [13] such as DROP [8,21]. To address it, in-context learning [7] is a promising approach to enhancing the capability of LLMs. The idea of in-context learning is to concatenate the test question with an analogous demonstration context to prompt LLMs to generate answers. As shown in the left half of Figure 1, the demonstration context consists of a few task-related demonstration examples with labeled answers, which can be retrieved from the training set of the target dataset.
1 https://openai.com/blog/chatgpt

1 https://openai.com/blog/chatgpt

Motivation: Although existing works have designed a range of approaches for retrieving and exploiting demonstration examples [18,1,20], the common practice of constructing a demonstration context is still concatenating questions and labeled answers through simple templates. We argue that only showing demonstration questions with correct answers may not guide LLMs to think deeply about demonstration examples, e.g., lack of reflection on mistakes in problem solving, which may lead to under-utilization of the labeled answers. Our Work: In this paper, we propose to enhance in-context learning with diverse information derived from labeled answers to improve their utilization. Inspired by supervised learning which receives feedback from training loss to update model, we design a novel prompting strategy for LLM to obtain feedback information in the form of corrected answers. Specifically, as shown in the right part of Figure 1, this strategy first answers the demonstration question using an off-the-shelf model (e.g., based on conventional PLMs), compares its results with labeled answers, and records the corrected answers as feedback (e.g., correct, incorrect, or missing answers). Then we use both demonstration examples and corrected answers to construct an enhanced prompt for LLM. With this idea, we conducted experiments on three MSQA datasets as well as one keyphrase extraction dataset. The results show that our feedback-based prompting strategy significantly improves the capability of ChatGPT to answer multi-span questions.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7080/70800fe1-534f-40f5-a665-0378863cf51b.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c291/c291684a-6964-434e-9405-98bc224dfed1.png" style="width: 50%;"></div>
Fig. 1: An example of our new prompting strategy (right) compared with the conventional prompting strategy (left). Our strategy first answers the demonstration question using an off-the-shelf model (e.g., based on conventional PLMs) and records the corrected answers as feedback, and then combines demonstration examples with corrected answers to construct a prompt for LLM.

# 2 Related Work

# 2.1 Large Language Models

From GPT-3 [5] to the latest GPT-4 [21], the emergence of powerful LLMs in recent years has triggered new thinkings and paradigms in NLP research. LLMs perform various downstream tasks using only text instructions, have matched state-of-the-art results in many tasks including machine translation [10] and relation extraction [30], and have influenced a range of domain applications such as education [28] and medical writing [4]. Despite the great success of LLMs, studies have also reported that it still has shortcomings in specific tasks [24,2] and has a large gap in handling difficult tasks compared with PLM-based methods [13]. In particular, question answering (QA) is a task with long-term research and is faced with various challenges. The performance of LLMs on QA has received extensive attention. Some analytical works reported that LLMs have many limitations in QA tasks, including insufficient stability [29], poor performance on newly released datasets [17], and suffering from hallucinations [2]. Based on empirical observations, some works designed methods to improve the performance of LLMs on specific QA tasks such as commonsense QA [3], open-domain QA [16], and multi-document QA [23]. However, as an important and realistic QA task, Multi-Span QA (MSQA) currently lacks dedicated research based on LLMs, whose performance on this task remains unclear. In this paper, we propose and evaluate a novel strategy for effectively adapting LLMs to the MSQA task.

# 2.2 In-Context Learning

With the development of LLMs, in-context learning [7] has also received extensive attention in recent years. Some research works studied it from the perspective of demonstration formatting, proposing template engineering to construct better human-written or automatically generated prompts [19,32]. Some other methods enhanced in-context learning by selecting better demonstration examples, searching for the best ordering of demonstration examples [20], or using the KNN algorithm with lexical [1] or semantic [18] features to dynamically retrieve demonstration examples for each question. The usage of labeled answers in the above methods is to append them to the question using some simple templates, which leads to potential under-utilization of labeled answers. The work most similar to ours is [30], which feeds demonstration examples to LLM to obtain a clue about the gold labels in a given document in a relation extraction task. However, the clue generated by LLM often contains mistakes, which also causes some loss of label information, and it is very expensive to interact every demonstration example with LLM. By contrast, in this paper, we obtain answer feedback by comparing the prediction results on the demonstration example with the labeled answers, and use it to enrich in-context learning with more insightful information obtained from the corrected answers.

# Approach

Given a question Q and a reference document D, the goal of MSQA is to generate a set of n answers A = {A 1, . . . , A n}, where A i is a span-level text that may be either present in D or absent in D. Let T = {[D T 1, Q T 1, A T 1], . . .} be a set of labeled examples, i.e., the set of all the available question-document-answers triples from which demonstration examples can be selected for in-context learning, e.g., the training set of a MSQA dataset.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0b64/0b649281-feb9-4679-81d8-6be1595d583d.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a02c/a02c1509-a22d-4d46-bea8-aa82a49d64d5.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/49da/49da40a4-11d9-483b-89d5-7e267a6a9e5a.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 2: An overview of our prompting strategy, which includes a retrieval stage searching for relevant demonstration examples, an exercise stage for producing feedback, and a reasoning stage for in-context learning with feedback.
</div>
Figure 2 gives an overview of our strategy, which includes a retrieval stage searching for relevant demonstration examples, an exercise stage for producing feedback, and a reasoning stage for in-context learning with feedback.

We first search for a few relevant demonstration examples for test question Q from the labeled examples set T. To this end, a question index I is built for each question Q T i in T, and a retrieval module is executed to obtain the set E of topk relevant labeled examples:

where Index (·) and Retriever (·, ·) are indexing and retrieval functions, respectively, and we realize them using an inverted index and BM25 in our experiments. E = {[D E 1, Q E 1, A E 1], . . . , [D E k, Q E k, A E k]} is the selected demonstration examples set with size k.

(1)

# 3.2 Exercise Stage

Then we regard the selected demonstration examples E as exercises to predict their answers and extend them with corrected answers as feedback. The set of predicted answers A P i for each demonstration question Q E i is obtained as follows:

where QAModel (·, ·) is an off-the-shelf MSQA model (e.g., a conventional MSQA method based on PLMs), and A P i = {A P 1, . . . , A P m} is the predicted answers set with size m. Next, the predicted answers set A P i is compared with the labeled answers set A E i to obtain feedback about the predicted answers. The feedback consists of three parts: the correctly predicted set A C i, the incorrectly predicted set A I i, and the unpredicted (i.e., missing) set A M i, satisfying that |A C i | + |A I i | = m and |A C i | + |A M i | = n.

# 3.3 Reasoning Stage

Task
Function
Templates
MSQA
& KE
FeedbackTemp
(·, ·, ·)
Here are some correct answers (or present/absent keyphrases) responded
by other AI model:
1. [CORRECT1]; 2. [CORRECT2]; ...
Here are some incorrect answers (or present/absent keyphrases) re-
sponded by other AI model:
1. [INCORRECT1]; 2. [INCORRECT2]; ...
Here are some answers (or present/absent keyphrases) missed by other
AI model:
1. [MISS1]; 2. [MISS2]; ...
MSQA
TaskTemp
(·, ·, ·)
Reading the passage: [DOCUMENT]
Extract spans from the above passage to answer the question: [QUES-
TION ]
Answer as a list e.g. 1. answer1; 2. answer2
Answer: 1. [ANS1]; 2. [ANS2]; ...
ConcatTemp
(·, ·)
Example1: [DEMO CONTEXT1]
Example2: [DEMO CONTEXT2]
...
Then, answer me a question like the above examples:
[TEST QUESTION ]
KE
TaskTemp
(·, ·, ·)
Reading the passage: [DOCUMENT]
Extract present (or Generate absent) keyphrases from the above passage:
Response as a list e.g. 1. keyphrase1; 2. keyphrase2
Keyphrases: 1. [KEYPHRASE1]; 2. [KEYPHRASE2]; ...
ConcatTemp
(·, ·)
Example1: [DEMO CONTEXT1]
Example2: [DEMO CONTEXT2]
...
Then, extract present (or generate absent) keyphrases like the above cases:
[TEST QUESTION ]
After obtaining the answer feedback, an extended demonstration context is constructed from E and the feedback. For each demonstration example, we use

(2)

a task description template to construct demonstration context Prompt DEMO i, use a feedback template to construct feedback context Prompt FB i, and the expended demonstration context Prompt DEMO+ i is constructed by concatenating Prompt DEMO i and Prompt FB i:

where TaskTemp (·, ·, ·) and FeedbackTemp (·, ·, ·) are two template filling functions. The details of the templates can be found in Table 1. For the test question Q, we construct test context using the same task description template but set the answers to an empty set:

Prompt TEST i = TaskTemp (D, Q, ∅).

Prompt = ConcatTemp ({Prompt DEMO+ i, . . .}, Prompt TEST i)
A LLM = LLM (Prompt),

Prompt = ConcatTemp ({Prompt DEMO+ i, . . .}, Prompt TEST i)
A LLM = LLM (Prompt),

where ConcatTemp (·, ·) is a template filling function detailed in Table 1, and A LLM is a text answer returned by LLM. Since the instruction in the prompt requires LLM to answer in the form of a list, we can easily parse the text into multiple span-level answers to the test question.

# 4 Experimental Setup

We refer to our approach as FBPrompt.

# 4.1 Datasets

We compared FBPrompt with baselines on three MSQA datasets: MultispanQA [8 QUOREF [6], and DROP [15]. Since the test set of them is hidden, we used the official development set as our test set. In addition, we used a keyphase extraction dataset Inspec [9], which has a similar format to MSQA, with one document input and multiple span-level outputs, but without question. Considering the experimental cost, we only randomly sampled 500 samples for evaluation from QUOREF and DROP. Table 2 shows some statistics about these datasets.

# 4.2 Baselines

We compared FBPrompt with five popular usages of LLM as follows: Zero-shot prompts LLM only using handle-written instructions without demonstration examples.

(3)

(4)

(5)

<div style="text-align: center;">Table 2: Dataset statistics. Present Labels (%) indicates the percentage of answers in MSQA datasets or percentage of keyphrases in keyphrase extraction datasets that explicitly appear in the document.
</div>
datasets that explicitly appear in the document.
Dataset
Type
# Test
# Used
Present Labels (%)
Avg. # Answers
MultiSpanQA[15]
MSQA
653
653
100
2.89
QUOREF[6]
MSQA
2537
500
100
1.14
DROP[8]
MSQA
9,622
500
73.03
1.09
INSPEC[9]
KP
500
500
26.42
2.48
Random Sampling randomly selects k demonstration examples from the training set for each test question to construct prompt as done in [18]. BM25  calculates lexical similarity between questions to obtain topk relevant demonstration examples for each test question. It can be viewed as a simplified version of our FBPrompt—without using answer feedback. KATE [18] uses KNN algorithm selecting k demonstration examples with highest semantic similarity score for each test question. We implemented it based on dense passage retrieval [12]. Label-induced Reasoning [30] feeds labeled answers, the question, and the document to LLM to obtain a clue about the relation between question and answers. We implemented it using the same BM25 results as our FBPrompt.

# 4.3 Evaluation Metrics

We evaluated on each dataset using their official metrics [15,8,6,31]. For MultiSpanQA, we used Exact Match F1 (EM) and Partial Match F1 (PM). For QUOREF and DROP, we used Exact Match Global (EM G) and F1 score (F1). For INSPEC, we used macro-averaged F1@5 and F1@M.

# 4.4 Implementation Details

We used OpenAI official API 2 with the model gpt-3.5-turbo-0301 for all our experiments. We used the T5-base [25] model as our off-the-shelf model in FBPrompt. For the keyphrase extraction task, we performed extraction of present keyphrases and generation of absent keyphrases in two independent steps with two slightly different instructions as show in Table 1. Unless otherwise specified, we set k = 3, i.e., FBPrompt and all the few-shot baselines used three demonstration examples.

# 5 Experimental Results

5.1 Comparison with Baselines

# 5.1 Comparison with Baselines

In Table 3, FBPrompt significantly outperforms previous LLM-based methods on all metrics in the four datasets. In particular, compared with BM25 which
2 https://platform.openai.com/

2 https://platform.openai.com/

<div style="text-align: center;">Table 3: Main results on MSQA. The best results are in bold. ‡ indicates the results reported in [27]. ∗ indicates that the results are not completely comparable due to the difference in test data.
</div>
<div style="text-align: center;">Table 3: Main results on MSQA. The best results are in bold. ‡ indicates the results reported in [27]. ∗ indicates that the results are not completely comparable
</div>
due to the difference in test data.
MultiSpanQA
QUOREF
DROP
INSPEC
EM
PM
EMG
F1
EMG
F1
Present
Absent
F1@5 F1@M F1@5 F1@M
SOTA
73.13∗83.36∗80.61∗86.70∗84.86∗87.54∗0.401‡ 0.476‡ 0.030‡ 0.041‡
Zero-shot
39.47
68.14
33.60
51.07
5.81
17.25 0.298‡ 0.417‡ 0.016‡ 0.030‡
Random
58.62
80.62
71.40
80.25
47.70
60.53
0.401
0.472
0.033
0.051
Label-induced 54.56
76.99
64.40
71.96
12.63
16.47
0.115
0.135
0.009
0.013
KATE
60.78
81.51
73.00
79.76
50.90
60.69
0.399
0.468
0.026
0.038
BM25
61.33
81.63
70.80
79.00
58.40
65.93
0.405
0.470
0.029
0.051
FBPrompt
64.60
83.11
73.60
80.55
62.00
69.11
0.425 0.499 0.034 0.055
uses the same demonstration examples as ours, FBPrompt exceeds it by a large lead, thus exhibiting the performance brought by our proposed answer feedback. We also show the state-of-the-art (SOTA) results reported by other papers using fully-supervised fine-tuned models: they are [14] for MultiSpanQA, [26] for QUOREF, [33] for DROP, and [27] for INSPEC. Although the experimental results on the three MSQA datasets are not directly comparable due to inconsistent test data, it can be found that LLM-based models are still weaker than the fully-supervised models, but performs relatively well on the keyphrase extraction dataset INSPEC. FBPrompt closes the gap to SOTA on MSQA and achieves new SOTA results on the INSPEC dataset.

# 5.2 The Effectiveness of Different Feedback

We compare FBPrompt with the method using only one type of feedback to analyze whether all the three types of feedback bring benefits. The results reported in Table 4 reveal that each part of the feedback has a effect in improving the performance of LLM. In particular, using only correct answers leads to the largest loss compared with using only incorrect or missing answers, which shows that negative feedback has the largest benefit to LLM.

<div style="text-align: center;">Table 4: Effectiveness of different feedback. The best results are in bold.
</div>
Table 4: Effectiveness of different feedback. The best results are in bold.
MultiSpanQA
QUOREF
DROP
INSPEC
EM
PM
EMG
F1
EMG
F1
Present
Absent
F1@5 F1@M F1@5 F1@M
FBPrompt
64.60 83.11 73.60 80.55 62.00 69.11 0.425 0.499 0.034 0.055
- only correct
62.70
82.75
71.40 79.69 58.40 65.60 0.401 0.463
0.027 0.046
- only incorrect 62.93
82.97
72.40 80.23 60.20 67.92 0.417 0.490
0.030 0.048
- only missing
63.48
82.90
72.80 79.75 61.20 68.80 0.416 0.480
0.027 0.046
<div style="text-align: center;">Table 5: Comparsion with random feedback.
</div>
Table 5: Comparsion with random feedback.
MultiSpanQA
QUOREF
DROP
INSPEC
EM
PM
EMG
F1
EMG
F1
Present
Absent
F1@5 F1@M F1@5 F1@M
FBPrompt
64.60 83.11 73.60 80.55 62.00 69.11 0.425 0.499 0.034 0.055
Random feedback 60.95
81.40
64.40 72.11 60.72 68.03 0.415 0.482
0.029 0.047
# 5.3 Comparison with Random Feedback

Then, we simulate feedback by randomly generating predicted answers to observe whether the improvement of FBPrompt is really brought about by our carefully designed feedback. For the labeled answers set A E i from demonstration example [D E i, Q E i, A E i], we randomly selected a number ˆ n 1 in the range [0, |A E i |}], and randomly sampled ˆ n 1 positive answers from the labeled answers set A E i as pseudo positive predicted answers A Pos. Similarly, we randomly selected a number ˆ n 2 in the range [0, |A E i |}], and randomly sampled ˆ n 2 spans from the document D E i as pseudo negative predicted answers A Neg. Then, we merged A Pos and A Neg as the pseudo predicted answers and executed FBPrompt to generate answers. As shown in Table 5, the performance of FBPrompt drops significantly when random feedback is used, which shows that our constructed feedback is useful.

# 5.4 Number of Demonstration Examples

We study whether FBPrompt exhibits consistent effectiveness when the number of demonstration examples varies. In Figure 3, we report the changing trend of FBPrompt and BM25 when the number of examples changes from 1 to 4. We observe that with a varying number of examples in the four datasets, the performance of FBPrompt is consistently higher than that of BM25. Especially in the case of one-shot, FBPrompt largely outperforms BM25.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/034f/034f5aa0-3836-49d5-b943-a7b757ffe13c.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) MultiSpanQA
</div>
<div style="text-align: center;">(b) QUOREF
</div>
Fig. 3: Results of FBPrompt and BM25 with different numbers of e four datasets.

<div style="text-align: center;">ig. 3: Results of FBPrompt and BM25 with different numbers of examples on
</div>
<div style="text-align: center;">Table 6: Case Study
</div>
Table 6: Case Study
Demonstration Examples
Test Question
Document: Glycogen functions as one of two
forms of long - term energy reserves , with the
other form being triglyceride stores in adipose
tissue ( i.e. , body fat ) . In humans , glycogen
is made and stored primarily in the cells of the
liver and skeletal muscle ...
Document: Aflatoxins are poisonous carcinogens
that are produced by certain molds ( Aspergillus
flavus and Aspergillus
parasiticus ) which
grow in soil , decaying vegetation , hay , and
grains ...
Question: Which two forms of energy do muscles
produce ?
Question: Where are the organisms that produce
aflatoxins found ?
Gold Answers: Glycogen, triglyceride
Gold
Answers: soil,
decaying
vegetation,
hay, grains
Feedback:
Baseline Predictions: Aspergillus parasiti-
cus, Aspergillus flavus
Incorrect
Answers: liver,
skeletal
muscle
Answers Missed: Glycogen, triglyceride
FBPrompt Predictions: soil, decaying vege-
tation, hay, grains
# 5.5 Case Study

A real case from MultiSpanQA is presented in Table 6. The left part shows an demonstration example for the test question in the right part. We can observe that the prediction of the baseline method (BM25) makes a mistake, since LLM observes ‘produce’ in the question and directly finds answers around ‘produced’, instead of analyzing the meaning of the question thoroughly. As for FBPrompt, our off-the-shelf model also observes ‘produce’ in the question, and mistakenly finds the answers ‘liver’, ‘skeletal musc’ in the original document near ‘made’, which is semantically close to ‘produce’. But after given a feedback, LLM learns not to be confused by such specific word, and tries to understand the entire question. Therefore, FBPrompt finally generates correct answers.

# 6 Conclusion

In this paper, we explore the performance of LLM in multi-span question answering, finding that existing in-context learning methods under-utilize labeled answers. To alleviate this problem, we propose a novel prompting strategy called FBPrompt, which constructs and employs answer feedback from an off-the-shelf model to enhance in-context learning. Experiments on multiple datasets show that FBPrompt using answer feedback significantly improves the performance of LLM on MSQA tasks. In the future, we will deeply analyze the working principle of answer feedback, and try to integrate more useful feedback information into LLM for various tasks.

Acknowledgements This work was supported in part by the NSFC (62072224) and in part by the CAAI-Huawei MindSpore Open Fund.

# References

1. Agrawal, S., Zhou, C., Lewis, M., Zettlemoyer, L., Ghazvininejad, M.: In-context examples selection for machine translation. CoRR abs/2212.02437 (2022)

2. Bang, Y., Cahyawijaya, S., Lee, N., Dai, W., Su, D., Wilie, B., Lovenia, H., Ji, Z., Yu, T., Chung, W., Do, Q.V., Xu, Y., Fung, P.: A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. CoRR abs/2302.04023 (2023)
3. Bian, N., Han, X., Sun, L., Lin, H., Lu, Y., He, B.: Chatgpt is a knowledgeable but inexperienced solver: An investigation of commonsense problem in large language models. CoRR abs/2303.16421 (2023)
4. Biswas, S.: Chatgpt and the future of medical writing (2023) 5. Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D.M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., Amodei, D.: Language models are few-shot learners. In: NeurIPS 2020 (2020)
6. Dasigi, P., Liu, N.F., Marasovic, A., Smith, N.A., Gardner, M.: Quoref: A reading comprehension dataset with questions requiring coreferential reasoning. In: EMNLP-IJCNLP 2019. pp. 5924–5931 (2019)
7. Dong, Q., Li, L., Dai, D., Zheng, C., Wu, Z., Chang, B., Sun, X., Xu, J., Li, L., Sui, Z.: A survey for in-context learning. CoRR abs/2301.00234 (2023)
8. Dua, D., Wang, Y., Dasigi, P., Stanovsky, G., Singh, S., Gardner, M.: DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In: NAACL-HLT 2019. pp. 2368–2378 (2019)
9. Hulth, A.: Improved automatic keyword extraction given more linguistic knowledge. In: EMNLP 2003. p. 216–223. EMNLP ’03 (2003)
10. Jiao, W., Wang, W., Huang, J., Wang, X., Tu, Z.: Is chatgpt a good translator? yes with gpt-4 as the engine. arXiv preprint arXiv:2301.08745 (2023)
11. Ju, Y., Wang, W., Zhang, Y., Zheng, S., Liu, K., Zhao, J.: CMQA: A dataset of conditional question answering with multiple-span answers. In: COLING 2022. pp. 1697–1707 (2022)
12. Karpukhin, V., Oguz, B., Min, S., Lewis, P.S.H., Wu, L., Edunov, S., Chen, D., Yih, W.: Dense passage retrieval for open-domain question answering. In: EMNLP 2020. pp. 6769–6781 (2020)
13. Kocon, J., Cichecki, I., Kaszyca, O., Kochanek, M., Szydlo, D., Baran, J., Bielaniewicz, J., Gruza, M., Janz, A., Kanclerz, K., Kocon, A., Koptyra, B., Mieleszczenko-Kowszewicz, W., Milkowski, P., Oleksy, M., Piasecki, M., Radlinski, L., Wojtasik, K., Wozniak, S., Kazienko, P.: Chatgpt: Jack of all trades, master of none. CoRR abs/2302.10724 (2023)
14. Lee, S., Kim, H., Kang, J.: LIQUID: A framework for list question answering dataset generation. CoRR abs/2302.01691 (2023)
15. Li, H., Tomko, M., Vasardani, M., Baldwin, T.: Multispanqa: A dataset for multispan question answering. In: NAACL 2022. pp. 1250–1260 (2022)
16. Li, J., Zhang, Z., Zhao, H.: Self-prompting large language models for open-domain QA. CoRR abs/2212.08635 (2022)
17. Liu, H., Ning, R., Teng, Z., Liu, J., Zhou, Q., Zhang, Y.: Evaluating the logical reasoning ability of chatgpt and GPT-4. CoRR abs/2304.03439 (2023)
18. Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., Chen, W.: What makes good in-context examples for gpt-3? In: DeeLIO@ACL 2022. pp. 100–114 (2022)
19. Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., Neubig, G.: Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Comput. Surv. 55 (2023)

20. Lu, Y., Bartolo, M., Moore, A., Riedel, S., Stenetorp, P.: Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In: ACL 2022. pp. 8086–8098 (2022)
21. OpenAI: GPT-4 technical report. CoRR abs/2303.08774 (2023) 22. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P.F., Leike, J., Lowe, R.: Training language models to follow instructions with human feedback. In: NeurIPS (2022)
23. Pereira, J.A., do Nascimento Fidalgo, R., de Alencar Lotufo, R., Nogueira, R.F.: Visconde: Multi-document QA with GPT-3 and neural reranking. In: ECIR 2023. pp. 534–543 (2023)
24. Qin, C., Zhang, A., Zhang, Z., Chen, J., Yasunaga, M., Yang, D.: Is chatgpt a general-purpose natural language processing task solver? CoRR abs/2302.06476 (2023)
25. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res. 21, 140:1–140:67 (2020), http://jmlr.org/papers/v21/20-074.html
26. Segal, E., Efrat, A., Shoham, M., Globerson, A., Berant, J.: A simple and effective model for answering multi-span questions. In: EMNLP 2020. pp. 3074–3080 (2020)
27. Song, M., Jiang, H., Shi, S., Yao, S., Lu, S., Feng, Y., Liu, H., Jing, L.: Is chatgpt A good keyphrase generator? A preliminary study. CoRR abs/2303.13001 (2023)
28. Susnjak, T.: Chatgpt: The end of online exam integrity? CoRR abs/2212.09292 (2022)
29. Tan, Y., Min, D., Li, Y., Li, W., Hu, N., Chen, Y., Qi, G.: Evaluation of chatgpt as a question answering system for answering complex questions. CoRR abs/2303.07992 (2023)
30. Wan, Z., Cheng, F., Mao, Z., Liu, Q., Song, H., Li, J., Kurohashi, S.: GPT-RE: in-context learning for relation extraction using large language models. CoRR abs/2305.02105 (2023)
31. Yuan, X., Wang, T., Meng, R., Thaker, K., Brusilovsky, P., He, D., Trischler, A.: One size does not fit all: Generating and evaluating variable number of keyphrases. In: ACL 2020. pp. 7961–7975 (2020)
32. Zhou, Y., Muresanu, A.I., Han, Z., Paster, K., Pitis, S., Chan, H., Ba, J.: Large language models are human-level prompt engineers. CoRR abs/2211.01910 (2022)
33. Zhou, Y., Bao, J., Duan, C., Sun, H., Liang, J., Wang, Y., Zhao, J., Wu, Y., He, X., Zhao, T.: OPERA: operation-pivoted discrete reasoning over text. In: NAACL 2022. pp. 1655–1666 (2022)
34. Zhu, M., Ahuja, A., Juan, D., Wei, W., Reddy, C.K.: Question answering with long multiple-span answers. In: EMNLP 2020. pp. 3840–3849 (2020)

