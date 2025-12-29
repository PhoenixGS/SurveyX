# Enhancing In-Context Learning with Answer Feedback for Multi-Span Question Answering
Zixian Huang, Jiaying Zhou, Gengyang Xiao, and Gong Cheng
State Key Laboratory for Novel Software Technology, Nanjing University, China {zixianhuang,jyzhou,181840264}@smail.nju.edu.cn, gcheng@nju.edu.cn
Abstract. Whereas the recent emergence of large language models (LLMs) like ChatGPT has exhibited impressive general performance, it still has a large gap with fully-supervised models on specific tasks such as multispan question answering. Previous researches found that in-context learning is an effective approach to exploiting LLM, by using a few task-related labeled data as demonstration examples to construct a few-shot prompt for answering new questions. A popular implementation is to concatenate a few questions and their correct answers through simple templates, informing LLM of the desired output. In this paper, we propose a novel way of employing labeled data such that it also informs LLM of some undesired output, by extending demonstration examples with feedback about answers predicted by an off-the-shelf model, e.g., correct, incorrect, or incomplete. Experiments on three multi-span question answering datasets as well as a keyphrase extraction dataset show that our new prompting strategy consistently improves LLM’s in-context learning performance.
# 1 Introduction
Recently, the rise of large language models (LLMs) [5,22,21] represented by ChatGPT1 provides a new paradigm for NLP research, which can perform well using only natural language instructions rather than being trained on the target dataset. Based on LLMs, many tasks are expected to be more convenient and accessible to users with different needs, including multi-span question answering (MSQA). MSQA aims to automatically find one-to-many answers at the span level for a given question, which has attracted many in-depth research works [15,26] based on pre-trained language models (PLMs), and has broad application scenarios such as medical question answering [34,11]. However, compared with PLMs fine-tuned on the complete training data, LLMs still have a large gap on difficult MSQA datsets [13] such as DROP [8,21]. To address it, in-context learning [7] is a promising approach to enhancing the capability of LLMs. The idea of in-context learning is to concatenate the test question with an analogous demonstration context to prompt LLMs to generate answers. As shown in the left half of Figure 1, the demonstration context consists of a few task-related demonstration examples with labeled answers, which can be retrieved from the training set of the target dataset. 1 https://openai.com/blog/chatgpt
Motivation: Although existing works have designed a range of approaches for retrieving and exploiting demonstration examples [18,1,20], the common practice of constructing a demonstration context is still concatenating questions and labeled answers through simple templates. We argue that only showing demonstration questions with correct answers may not guide LLMs to think deeply about demonstration examples, e.g., lack of reflection on mistakes in problem solving, which may lead to under-utilization of the labeled answers. Our Work: In this paper, we propose to enhance in-context learning with diverse information derived from labeled answers to improve their utilization. Inspired by supervised learning which receives feedback from training loss to update model, we design a novel prompting strategy for LLM to obtain feedback information in the form of corrected answers. Specifically, as shown in the right part of Figure 1, this strategy first answers the demonstration question using an off-the-shelf model (e.g., based on conventional PLMs), compares its results with labeled answers, and records the corrected answers as feedback (e.g., correct, incorrect, or missing answers). Then we use both demonstration examples and corrected answers to construct an enhanced prompt for LLM. With this idea, we conducted experiments on three MSQA datasets as well as one keyphrase extraction dataset. The results show that our feedback-based prompting strategy significantly improves the capability of ChatGPT to answer multi-span questions.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8e74/8e74b248-84e2-4027-a018-986900577769.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5882/588266d3-689a-4070-b8a0-eedf53e28ad5.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2236/22366d10-7e54-421b-b2a3-6e9da6d40e58.png" style="width: 50%;"></div>
Fig. 1: An example of our new prompting strategy (right) compared with the conventional prompting strategy (left). Our strategy first answers the demonstration question using an off-the-shelf model (e.g., based on conventional PLMs) and records the corrected answers as feedback, and then combines demonstration examples with corrected answers to construct a prompt for LLM.
# 2 Related Work
# 2.1 Large Language Models
From GPT-3 [5] to the latest GPT-4 [21], the emergence of powerful LLMs in recent years has triggered new thinkings and paradigms in NLP research. LLMs perform various downstream tasks using only text instructions, have matched state-of-the-art results in many tasks including machine translation [10] and relation extraction [30], and have influenced a range of domain applications such as education [28] and medical writing [4]. Despite the great success of LLMs, studies have also reported that it still has shortcomings in specific tasks [24,2] and has a large gap in handling difficult tasks compared with PLM-based methods [13]. In particular, question answering (QA) is a task with long-term research and is faced with various challenges. The performance of LLMs on QA has received extensive attention. Some analytical works reported that LLMs have many limitations in QA tasks, including insufficient stability [29], poor performance on newly released datasets [17], and suffering from hallucinations [2]. Based on empirical observations, some works designed methods to improve the performance of LLMs on specific QA tasks such as commonsense QA [3], open-domain QA [16], and multi-document QA [23]. However, as an important and realistic QA task, Multi-Span QA (MSQA) currently lacks dedicated research based on LLMs, whose performance on this task remains unclear. In this paper, we propose and evaluate a novel strategy for effectively adapting LLMs to the MSQA task.
# 2.2 In-Context Learning
With the development of LLMs, in-context learning [7] has also received extensive attention in recent years. Some research works studied it from the perspective of demonstration formatting, proposing template engineering to construct better human-written or automatically generated prompts [19,32]. Some other methods enhanced in-context learning by selecting better demonstration examples, searching for the best ordering of demonstration examples [20], or using the KNN algorithm with lexical [1] or semantic [18] features to dynamically retrieve demonstration examples for each question. The usage of labeled answers in the above methods is to append them to the question using some simple templates, which leads to potential under-utilization of labeled answers. The work most similar to ours is [30], which feeds demonstration examples to LLM to obtain a clue about the gold labels in a given document in a relation extraction task. However, the clue generated by LLM often contains mistakes, which also causes some loss of label information, and it is very expensive to interact every demonstration example with LLM. By contrast, in this paper, we obtain answer feedback by comparing the prediction results on the demonstration example with the labeled answers, and use it to enrich in-context learning with more insightful information obtained from the corrected answers.
# 3 Approach
Given a question Q and a reference document D, the goal of MSQA is to generate a set of n answers A = {A1, . . . , An}, where Ai is a span-level text that may be either present in D or absent in D. Let T = {[DT 1 , QT 1 , AT 1 ], . . .} be a set of labeled examples, i.e., the set of all the available question-document-answers triples from which demonstration examples can be selected for in-context learning, e.g., the training set of a MSQA dataset.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bdc8/bdc89479-f4af-4cda-a229-bcb3a465dd04.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/47c1/47c1b012-6d2d-4ac5-9de3-c74aec07c4eb.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 2: An overview of our prompting strategy, which includes a retrieval stage searching for relevant demonstration examples, an exercise stage for producing feedback, and a reasoning stage for in-context learning with feedback.</div>
Figure 2 gives an overview of our strategy, which includes a retrieval stage searching for relevant demonstration examples, an exercise stage for producing feedback, and a reasoning stage for in-context learning with feedback.
We first search for a few relevant demonstration examples for test question Q from the labeled examples set T . To this end, a question index I is built for each question QT i in T , and a retrieval module is executed to obtain the set E of top-k relevant labeled examples:
where Index(·) and Retriever(·, ·) are indexing and retrieval functions, respectively, and we realize them using an inverted index and BM25 in our experiments. E = {[DE 1 , QE 1 , AE 1 ], . . . , [DE k , QE k , AE k ]} is the selected demonstration examples set with size k.
(1)
# 3.2 Exercise Stage
Then we regard the selected demonstration examples E as exercises to predict their answers and extend them with corrected answers as feedback. The set of predicted answers AP i for each demonstration question QE i is obtained as follows:
where QAModel(·, ·) is an off-the-shelf MSQA model (e.g., a conventional MSQA method based on PLMs), and AP i = {AP 1 , . . . , AP m} is the predicted answers set with size m. Next, the predicted answers set AP i is compared with the labeled answers set AE i to obtain feedback about the predicted answers. The feedback consists of three parts: the correctly predicted set AC i , the incorrectly predicted set AI i , and the unpredicted (i.e., missing) set AM i , satisfying that |AC i | + |AI i | = m and |AC i | + |AM i | = n.
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
a task description template to construct demonstration context PromptDEMO i , use a feedback template to construct feedback context PromptFB i , and the expended demonstration context PromptDEMO+ i is constructed by concatenating PromptDEMO i and PromptFB i :
where TaskTemp(·, ·, ·) and FeedbackTemp(·, ·, ·) are two template filling functions. The details of the templates can be found in Table 1. For the test question Q, we construct test context using the same task description template but set the answers to an empty set:
Finally, we use a concatenation template to construct the comp nd feed it into LLM:
Prompt = ConcatTemp({PromptDEMO+ i , . . . }, PromptTEST i ) ALLM = LLM(Prompt) ,
where ConcatTemp(·, ·) is a template filling function detailed in Table 1, and ALLM is a text answer returned by LLM. Since the instruction in the prompt requires LLM to answer in the form of a list, we can easily parse the text into multiple span-level answers to the test question.
# 4 Experimental Setup
We refer to our approach as FBPrompt.
# 4.1 Datasets
We compared FBPrompt with baselines on three MSQA datasets: MultispanQA  QUOREF [6], and DROP [15]. Since the test set of them is hidden, we used the official development set as our test set. In addition, we used a keyphase extraction dataset Inspec [9], which has a similar format to MSQA, with one document input and multiple span-level outputs, but without question. Considering the experimental cost, we only randomly sampled 500 samples for evaluation from QUOREF and DROP. Table 2 shows some statistics about these datasets.
# 4.2 Baselines
We compared FBPrompt with five popular usages of LLM as follows: Zero-shot prompts LLM only using handle-written instructions without demonstration examples.
(3)
(4)
(5)
<div style="text-align: center;">Table 2: Dataset statistics. Present Labels (%) indicates the percentage of answers in MSQA datasets or percentage of keyphrases in keyphrase extraction datasets that explicitly appear in the document.</div>
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
Random Sampling randomly selects k demonstration examples from the training set for each test question to construct prompt as done in [18]. BM25 calculates lexical similarity between questions to obtain top-k relevant demonstration examples for each test question. It can be viewed as a simplified version of our FBPrompt—without using answer feedback. KATE [18] uses KNN algorithm selecting k demonstration examples with highest semantic similarity score for each test question. We implemented it based on dense passage retrieval [12]. Label-induced Reasoning [30] feeds labeled answers, the question, and the document to LLM to obtain a clue about the relation between question and answers. We implemented it using the same BM25 results as our FBPrompt.
# 4.3 Evaluation Metrics
We evaluated on each dataset using their official metrics [15,8,6,31]. For MultiSpanQA, we used Exact Match F1 (EM) and Partial Match F1 (PM). For QUOREF and DROP, we used Exact Match Global (EMG) and F1 score (F1). For INSPEC, we used macro-averaged F1@5 and F1@M.
# 4.4 Implementation Details
We used OpenAI official API2 with the model gpt-3.5-turbo-0301 for all our experiments. We used the T5-base [25] model as our off-the-shelf model in FBPrompt. For the keyphrase extraction task, we performed extraction of present keyphrases and generation of absent keyphrases in two independent steps with two slightly different instructions as show in Table 1. Unless otherwise specified, we set k = 3, i.e., FBPrompt and all the few-shot baselines used three demonstration examples.
# 5 Experimental Results
# 5.1 Comparison with Baselines
In Table 3, FBPrompt significantly outperforms previous LLM-based methods on all metrics in the four datasets. In particular, compared with BM25 which 2 https://platform.openai.com/
<div style="text-align: center;">Table 3: Main results on MSQA. The best results are in bold. ‡ indicates the results reported in [27]. ∗indicates that the results are not completely comparable</div>
<div style="text-align: center;">Table 3: Main results on MSQA. The best results are in bold. ‡ indicates the results reported in [27]. ∗indicates that the results are not completely comparable due to the difference in test data.</div>
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
<div style="text-align: center;">Table 4: Effectiveness of different feedback. The best results are in bold.</div>
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
<div style="text-align: center;">Table 5: Comparsion with random feedback.</div>
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
Then, we simulate feedback by randomly generating predicted answers to observe whether the improvement of FBPrompt is really brought about by our carefully designed feedback. For the labeled answers set AE i from demonstration example [DE i , QE i , AE i ], we randomly selected a number ˆn1 in the range [0, |AE i |}], and randomly sampled ˆn1 positive answers from the labeled answers set AE i as pseudo positive predicted answers APos. Similarly, we randomly selected a number ˆn2 in the range [0, |AE i |}], and randomly sampled ˆn2 spans from the document DE i as pseudo negative predicted answers ANeg. Then, we merged APos and ANeg as the pseudo predicted answers and executed FBPrompt to generate answers. As shown in Table 5, the performance of FBPrompt drops significantly when random feedback is used, which shows that our constructed feedback is useful.
# 5.4 Number of Demonstration Examples
We study whether FBPrompt exhibits consistent effectiveness when the number of demonstration examples varies. In Figure 3, we report the changing trend of FBPrompt and BM25 when the number of examples changes from 1 to 4. We observe that with a varying number of examples in the four datasets, the performance of FBPrompt is consistently higher than that of BM25. Especially in the case of one-shot, FBPrompt largely outperforms BM25.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/38b7/38b7c088-4ca9-421a-ab9c-32ede888e39f.png" style="width: 50%;"></div>
Fig. 3: Results of FBPrompt and BM25 with different numbers of examples o four datasets.
<div style="text-align: center;">Fig. 3: Results of FBPrompt and BM25 with different numbers of examples on four datasets.</div>
<div style="text-align: center;">Table 6: Case Study</div>
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
