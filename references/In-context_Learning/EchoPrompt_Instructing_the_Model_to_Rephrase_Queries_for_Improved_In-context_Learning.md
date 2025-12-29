# EchoPrompt: Instructing the Model to Rephrase Queries for Improved In-context Learning

Rajasekhar Reddy Mekala ∗
rmekala@uci.edu Yasaman Razeghi ∗
yrazeghi@uci.edu

# Abstract

Language models are achieving impressive performance on various tasks by aggressively adopting inference-time prompting techniques, such as zero-shot and few-shot prompting. In this work, we introduce EchoPrompt, a simple yet effective approach that prompts the model to rephrase its queries before answering them. EchoPrompt is adapted for both zero-shot and few-shot in-context learning with standard and chain-of-thought prompting. Experimental results show that EchoPrompt yields substantial improvements across all these settings for four families of causal language models. These improvements are observed across various numerical reasoning (e.g. GSM8K, SVAMP), reading comprehension (e.g. DROP), and logical reasoning (e.g. Coin Flipping) tasks. On average, EchoPrompt improves the Zero-shot-CoT performance of code-davinci-002 by 5% in numerical tasks and 13% in reading comprehension tasks. We investigate the factors contributing to EchoPrompt’s effectiveness through ablation studies, which reveal that both the original query and the model-generated rephrased version are instrumental in its performance gains. Our empirical results indicate that EchoPrompt is an effective technique that enhances in-context learning performance. We recommend incorporating EchoPrompt into various baseline prompting strategies to achieve performance boosts.

# Introduction

Large language models have revolutionized natural language task-solving through prompting (Brown et al., 2020). This technique involves conditioning the language model with an instruction (zero-shot) or augmenting the prompt with a small set of taskspecific examples (few-shot), resulting in the model to generalize and respond effectively to tasks. A rapidly advancing body of research has introduced techniques to enhance these prompting
∗ First two authors contributed equally.

Sameer Singh sameer@uci.edu

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b091/b091f026-3ca7-460a-8e66-998c6ae54acc.png" style="width: 50%;"></div>
Figure 1: Comparison of prompts in Zero-shot-CoT with and without EchoPrompt, highlighting the modification in prompts. Zero-shot-CoT with EchoPrompt uses the prompt “Let’s repeat the question and also think step by step” to aid the model in recalling the query before solving it.

methodologies. Notably, chain-of-thought prompting (Wei et al., 2023; Kojima et al., 2022) has emerged as a powerful method for enhancing language model performance in reasoning tasks. Least-to-most prompting (Zhou et al., 2022a) and Tree of Thoughts (Yao et al., 2023) further support chain-of-thought by breaking down complex problems into simpler subproblems.
While both standard prompting and chain-ofthought prompting exhibit impressive capabilities and find applications across various domains, they can sometimes lead to inaccurate responses due to logical errors, symbol mapping issues, and omission of intermediate steps (Kojima et al., 2022), indicating potential oversights in adequately addressing various facets of the queries.
In this paper, we propose EchoPrompt, a prompting strategy that builds upon existing prompting

approaches by incorporating Query-Rephrasing as a preliminary task in the in-context learning process. EchoPrompt draws inspiration from the innate cognitive strategies employed by humans, precisely the act of self-questioning, when answering queries. By verbalizing queries before answering them, humans establish a cognitive checkpoint to refine their thoughts, uncovering misconceptions that might have otherwise gone unnoticed (Joseph and Ross, 2018; Joseph et al., 2019). Figure 1
provides an illustrative example of EchoPrompting in Zero-shot-CoT settings. While the approach proposed by (Kojima et al., 2022) uses the prompt “Let’s think step by step." to elicit chain-of-thought reasoning and then extracts the answer using the prompt “Therefore, the answer is", we modify the first prompt to “Let’s repeat the question and also think step by step."  or similar texts. This modification guides the model to generate a version of the original query before solving it. We empirically evaluate our approach against various prompting baselines using a wide variety of model families with different sizes, including code-davinci-002, GPT-3.5-Turbo 1, Starcoder15B, Llama-13B, and GPT-J-6B. Our results show that EchoPrompt significantly improves the performance of language models on arithmetic, reading comprehension, and logical reasoning tasks. We observe substantial performance gains with both standard and chain-of-thought prompting, particularly in zero-shot scenarios for large language models (code-davinci-002, GPT-3.5-turbo) and with standard prompting on smaller models (Starcoder15B, Llama-13B, and GPT-J-6B). For example, EchoPrompt increases the Zero-shot-CoT performance from 56.8% to 67.3% on DROP (Census) and from 75.1% to 82.6% on GSM8K with chainof-thought prompting on GPT-3.5(gpt-3.5-turbo). We conduct a series of ablation studies to gain deeper insights into the effectiveness of the EchoPrompt technique. First, we examine whether the accuracy gains attributed to EchoPrompt resulted solely from rephrased queries. Our findings demonstrate that both the original query and the rephrased query are essential in achieving performance improvements. Next, we investigate whether EchoPrompt can be seen as a query augmentation technique by considering the alternative approach of directly augmenting the original

1 https://openai.com/blog/chatgpt/. We use gpt-3.5turbo-0301 snapshot from March 2023

query with a rephrased version. We observe comparable results between these two approaches, indicating that EchoPrompt serves as a query augmentation technique. Additionally, we explore whether instructing EchoPrompt to generate multiple rephrases can further enhance performance. Interestingly, we observe a slight performance drop as the number of rephrases increases. This suggests that the improvements achieved with EchoPrompt cannot be solely attributed to generating more tokens. Finally, we assess the performance of EchoPrompt in the presence of irrelevant text within the queries and find that it maintains improvements despite replicating irrelevant text in the rephrases. Our study indicates that EchoPrompt fundamentally improves in-context learning performance and finds broad applicability as a building block in emerging complex techniques that leverage prompting in multiple stages.

# 2 EchoPrompt

EchoPrompt teaches language models to generate a version of the query before solving it. The finegrained details of this technique are explained in the following two subsections, with examples.

# 2.1 Zero-shot EchoPrompt

In zero-shot prompting, the standard approach relies on a single prompt “Therefore, the answer is" to directly extract the answer. In contrast, Zero-shot EchoPrompt introduces a two-stage prompting process. The language model is initially instructed to rephrase the query using a task-agnostic prompt, “Let’s repeat the question. “" and then the answer is extracted using the same prompt as in zero-shot prompting. Similarly, in Zero-shot-CoT, as proposed by (Kojima et al., 2022), the conventional approach involves using the prompt “Let’s think step by step." to guide the model in generating its reasoning steps before producing the final answer. However, in Zero-shot-CoT with EchoPrompt, we introduce a query-rephrasing subtask by employing prompts like “Let’s repeat the question and also think step by step.". This modification encourages the model to generate the query in its own words and then engage in multi-hop reasoning. The prompt used for answer extraction remains consistent in both zero-shot and Zero-shot-CoT scenarios. Figure1 shows an example, highlighting the key differences between the two approaches. Tables1, 11 gives a

Original Question: If Pam is currently twice as young as Rena is, and in 10 years Rena will be 5 years older than her, how old is Pam now? Compound Sentence Rephrase Given that Pam is currently twice as young as Rena and that in 10 years Rena will be 5 years older than Pam, how old is Pam now? Question First Rephrase What is Pam's current age if Rena is twice as old as Pam and in 10 years Rena will be 5 years older than Pam? Simple Sentence Rephrase Currently, Pam is twice as young as Rena. In 10 years, Rena will be 5 years older than Pam. So, how old is Pam now? Repeatition If Pam is currently twice as young as Rena is, and in 10 years Rena will be 5 years older than her, how old is Pam now?

Figure 2: Example of rephrases used for the proposed rephrase structures in EchoPrompt in few-shot prompting exemplars. The Rephrases of exemplars are generated using ChatGPT based on prompts in Table10.

comprehensive overview of the prompts we experimented with in this approach 2.

# 2.2 Few-shot EchoPrompt

In few-shot learning, we teach the language model to rephrase the test query in a particular structure before answering the query. We do this by providing exemplars demonstrating the rephrase structure and corresponding responses to example queries. We examine three distinct rephrasing structures in addition to teaching the model to repeat the exact query in the following formats:

• Rephrased to Compound Sentences: Queries are formulated using compound sentences incorporating multiple clauses or phrases.

# Rephrased to putting the question First:

Queries are structured to present the final question at the beginning, followed by contextual information.

• Rephrased to Short and Simple Sentences: Queries are constructed by breaking down the original problem’s context into simpler and shorter sentences.

• Repetition: Repeating the original query itself can serve as a fundamental form of rephrasing, and we consider it one of the rephrase structures.

2 In zero-shot prompting, EchoPrompt only focuses on repeating the exact query, whereas in Zero-shot-CoT, we explore both query-repetition and rephrasing. This is because we can easily identify the end of query repetition by using quotations. However, there is no clear way to detect when the rephrase is complete.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4abd/4abd9989-4966-4582-86f0-a12fb0d6a22e.png" style="width: 50%;"></div>
<div style="text-align: center;">Q: Kelly has 5 quarters and 2 dimes. If she buys a can of pop for 55 cents, how many cents will she have left? A: Rewriting in simple words, the question is: "Given that Kelly has 5 quarters and 2 dimes, and she buys a can of pop for 55 cents, how many cents will she have left?" Now, to answer the rewritten question, the answer is 90. ✔
</div>
Q: Kelly has 5 quarters and 2 dimes. If she buys a can of pop for 55 cents, how many cents will she have left? A: Rewriting in simple words, the question is: "Given that Kelly has 5 quarters and 2 dimes, and she buys a can of pop for 55 cents, how many cents will she have left?" Now, to answer the rewritten question, the answer is 90. ✔

Figure 3: Example of EchoPrompt with Compound Sentences. Standard Prompting approach showcases exemplars with queries and corresponding answering formats. In contrast, the EchoPrompt incorporates a Query-Rephrase step, where the exemplars showcase a rephrased version of the query along with the answering format.

Figure2 shows an example of these rephrasing formats for a query. We use ChatGPT(OpenAI, 2021) to generate the rephrases for the exemplars in these structures. This way, even our exemplars are generated automatically and with the minimum human effort, which makes EchoPrompt simple to use. The prompts used for generating the rephrases for the exemplars are shown in Table10. In Figure3, we present an illustrative example of the proposed compound sentences  rephrasing. The exemplars in the standard prompting approach (highlighted in blue) demonstrate a sample query and the corresponding answering format. Consequently, when the model is presented with a test query, it responds similarly. However, with the introduction of EchoPrompt, the exemplars now showcase an additional step: query-rephrasing. Consequently, when the model encounters a test query, it produces a rephrased variant and answers it using the original and generated query reformulation.

# 3 Evaluation Setup

# 3.1 Benchmarks

We evaluate EchoPrompt across a range of natural language processing tasks, specifically focusing on four types, including fourteen widely recognized benchmarks. We experiment with four categories

of causal language models to ensure a broad and thorough evaluation. In this section, we delve into the details of our evaluation setup.

Numerical Reasoning We evaluate numerical reasoning tasks from (Wei et al., 2023) for a fair comparison between the methods including, GSM8K (Cobbe et al., 2021), SVAMP (Patel et al., 2021), AQUA-RAT (Ling et al., 2017),  SingleOp and MultiArith subsets from (Roy and Roth, 2016) . Additionally, we examine the performance of EchoPrompt on the high school mathematics subset of the MMLU dataset (Hendrycks et al., 2021a, b) and the GSMIC-4k dataset (Shi et al., 2023), which focuses explicitly on queries containing perturbations.

Logical Reasoning For logical reasoning, we assess the Date Understanding,  Shuffled Objects (tracking three objects) tasks from bigBench (Ghazal et al., 2013), LogiQA (Liu et al., 2020) and generate 1000 random samples with two trials of flipping for Coin Flipping task (Wei et al., 2023).

# Reading Comprehension

Reading Comprehension While we evaluate multiple numerical subsets of DROP (Dua et al., 2019), (including Football, Non-football, Census, and Break(Wolfson et al., 2020) from the QDMR dev subset) and could also be included in the arithmetic benchmarks, we group it with SQuAD (Rajpurkar et al., 2016) based on the query style. We evaluate EchoPrompt on DROP (Dua et al., 2019) and SQuAD (Rajpurkar et al., 2016) as two standard reading comprehension benchmarks. The Football subset of the DROP dataset was curated by applying keyword-based filtering with the keyword “yard" (Zhou et al., 2022a), and the Census subset was created by selectively filtering passages that contained the terms “population" and “census."

Commonsense Reasoning For commonsense reasoning, we use StrategyQA (Geva et al., 2021), Winogrande (ai2, 2019) datasets to assess the performance of EchoPrompt on tasks that involve simpler queries but require factual knowledge.

# 3.2 Language models

For our experiments, we use code-davinci-002 (Chen et al., 2021) as the primary model for all tasks since this model is free to evaluate and has a strong in-context learning ability. Additionally, we present the results on a subset of datasets on

GPT-3.5-Turbo, a model comparable to the size of code-davinci-002. We also experiment with the smaller and publicly available models such as StarCoder-15B (Li et al., 2023), Llama-13B (Touvron et al., 2023), and GPT-J-6B (Wang and Komatsuzaki, 2021) specifically on synthetic and simpler tasks.

# 3.3 Prompts

Few-shot Exemplars For a fair comparison of methods, we use the same exemplars introduced in (Wei et al., 2023) for the GSM8K, SVAMP, SingleOp, MultiArith, Date Understanding, and Coin-Flipping tasks across all models. Additionally, we evaluate with the prompts suggested by (Zhou et al., 2022a) for GSM8K, SVAMP, MultiArith, and DROP subsets. Furthermore, we provide a new set of prompts specifically for the DROP Census subset since no prior proposals exist.
Zero-shot-CoT Prompts As proposed in (Kojima et al., 2022), we employ the prompt “Let’s think step by step." in stage 1. In stage 2, we extract the answer using different prompts depending on the type of task. For multiple-choice tasks, we utilize prompts like “From (a) through (e), the answer is." For other tasks, we use the phrase “Therefore, the answer is."

# 4 Results

We conduct an extensive comparison of our approach against zero-shot, Zero-shot-CoT, few-shot, and few-shot-CoT prompting strategies. Figure4 (and Table9  in Appendix) provides the overall results of EchoPrompt while the extended results on code-davinci-002 and other models are presented in AppendixA. The findings on individual models are summarized below.
Code-davinci-002 Overall, We observe that EchoPrompt performs well regardless of the baseline prompting strategy. Notably, EchoPrompt shows significant improvements in zero-shot prompting scenarios, especially for tasks with longer query contexts, such as different DROP and SQuAD subsets containing extraneous information. For example, we observed an 18.5% improvement in accuracy on the DROP(Census subset) dataset for zero-shot prompting. Similarly, EchoPrompt with Zero-shot-CoT on SVAMP achieves (7.4%) improvement in accuracy, which makes the overall accuracy comparable to few-shot-CoT prompting.

We conduct an extensive comparison of our approach against zero-shot, Zero-shot-CoT, few-shot, and few-shot-CoT prompting strategies. Figure4 (and Table9  in Appendix) provides the overall results of EchoPrompt while the extended results on code-davinci-002 and other models are presented in AppendixA. The findings on individual models are summarized below.

Figure 4: Performance summary of EchoPrompt with repetition in zero-shot and compound sentence rephrasing in few-shot settings. Darker colored bars show EchoPrompt augmented with the baseline method. EchoPrompt consistently achieves performance gains across different prompting strategies, particularly in zero-shot scenarios. For details, see Table9 in Appendix.

However, it is worth noting that EchoPrompt does not yield improvements in cases where the baseline method cannot solve the task. For example, in the Shuffled Objects task involving three objects, EchoPrompt shows a slight drop in zero-shot performance (36.4% to 35.2%), which is close to random choice (33.3%). Nevertheless, it considerably improves the accuracy in Zero-shot-CoT (42.4% to 58.2%), where the model can partially solve the task. We also do not observe any consistent improvements in tasks involving multiplechoice questions, such as AQuA-RAT, MMLU, and LogiQA, where the model must select one option among several rather than explicitly generating the answer.

GPT-3.5-Turbo To assess the performance of the EchoPrompt technique on a non-code-trained model of similar size to Code-davinci-002, we experiment with GPT-3.5-Turbo on a subset of tasks. Detailed results are in Table9 in Appendix. Overall, these results align with our previous experiments on code-davinci-002. For example, the EchoPrompt technique significantly improves accuracy on GSM8K, from 75.1% to 83.5% in fewshot-CoT. However, we observe a drop in performance on reading comprehension tasks (DROP, and SQuAD) in zero-shot scenarios. After manual qualitative analysis, we observe that the model generates descriptive rather than instruction-based extractable answers, which explains some of the

drop in performance.

StarCoder-15B, Llama-13B, GPT-J-6B Similarly, we evaluate the performance of EchoPrompt on smaller and publicly available models: StarCoder-15B, Llama-13B, and GPT-J-6B. Our evaluation includes tasks such as coin-flipping, SingleOp, SVAMP, and date-understanding since these smaller models are less capable of challenging reasoning tasks. This set encompasses a toy task and two relatively simpler datasets, while date understanding is considered a challenging task on Bigbench. Detailed results are in Table9 in Appendix. EchoPrompt improves the performance with standard prompting, although we observe inconsistent results with chain-of-thought reasoning. This finding is not entirely surprising, as chain-of-thought is considered an emergent phenomenon in larger language models (Wei et al., 2023).
Comparision with least-to-most prompting Table2 shows a comparison of EchoPrompt in few-shot-CoT against least-to-most prompting 3, which is considered to be state-of-the-art for numerical reasoning tasks. While EchoPrompt utilizes rephrased queries, least to most prompting breaks down the problem into subproblems and solves these subproblems sequentially using chainof-thought. For a fair comparison, we evaluate

StarCoder-15B, Llama-13B, GPT-J-6B Similarly, we evaluate the performance of EchoPrompt on smaller and publicly available models: StarCoder-15B, Llama-13B, and GPT-J-6B. Our evaluation includes tasks such as coin-flipping, SingleOp, SVAMP, and date-understanding since these smaller models are less capable of challenging reasoning tasks. This set encompasses a toy task and two relatively simpler datasets, while date understanding is considered a challenging task on Bigbench. Detailed results are in Table9 in Appendix. EchoPrompt improves the performance with standard prompting, although we observe inconsistent results with chain-of-thought reasoning. This finding is not entirely surprising, as chain-of-thought is considered an emergent phenomenon in larger language models (Wei et al., 2023).

3 In all our evaluations, we employ the condensed variant of least-to-most prompting, where both decomposition and problem-solving are accomplished within a single step.

<div style="text-align: center;">EchoPrompt?
</div>
EchoPrompt?
Stage-1 Prompt
GSM8K
SVAMP
MultiArith SingleOp
Zero-shot
✗
-
16.4
66.8
31.0
91.6
✓
Let’s repeat the question. “
20.7(+4.3) 74.7(+7.9) 48.5(+17.5) 91.8(+0.2)
✓
Let’s reiterate the question. “
19.7(+3.3) 73.4(+6.6) 51.0(+20.0) 93.0(+1.4)
✓
Let’s restate the question. “
19.2(+2.8) 74.6(+7.8) 47.7(+16.7) 89.6(−2.0)
✓
Let’s summarize the question. “
20.6(+4.2) 73.2(+6.4) 48.8(+17.8) 93.7(+2.1)
Zero-shot-CoT
✗
Let’s think step by step.
49.3
66.5
76.0
82.9
✓
Let’s repeat the question and also think step by step.
44.6(−4.7) 74.7(+8.2) 70.9(−5.1) 92.3(+9.4)
✓
Let’s reiterate the question and also think step by step.
51.1(+1.8) 73.9(+7.4) 78.7(+2.7) 92.4(+9.5)
✓
Let’s repeat the question and also think step by step. “
42.0(−7.3) 60.4(−6.1) 78.1(+2.1) 88.3(+5.4)
✓
Let’s restate the question and also think step by step.
47.0(−2.3) 73.9(+7.4) 79.3(+3.3) 90.2(+7.3)
✓
Let’s summarize the question and also think step by step.
49.9(+0.6) 74.2(+7.7) 75.8(−0.2) 90.9(+8.0)
Table 1: Code-davinci-002: Arithematic reasoning Evaluation of EchoPrompt on various prompt templates. All
GSM8K SVAMP Multi-
Arith
DROP
(Census)
DROP
(Break)
DROP
(Football)
CoT
61.1
75.2
96.1
70.0
65.3
67.3
CoT+Compound
65.9
79.0
97.8
75.4
69.6
70.8
LTM
63.2
82.2
93.7
73.8
61.2
66.2
Table 2: code-davinci-002 Table show a comparison of
Table 2: code-davinci-002 Table show a comparison of EchoPrompt with CoT against least-to-most prompting. EchoPrompt outperforms least-to-most prompting on most of the benchmarks.

both numerical (GSM8K, SVAMP, Multiarith) and reading comprehension (DROP) tasks using the prompts proposed (Wei et al., 2023; Zhou et al., 2022a). Although EchoPrompt is a relatively simpler approach, it outperforms least-to-most prompting on two of the three arithmetic reasoning tasks and all reading comprehension subsets.

# 5 Analysis

To gain a deeper understanding of the factors that contribute to the success of EchoPrompt, we perform a series of ablation studies in the following sections:

Effect of prompts on zero-shot EchoPrompt To investigate the impact of prompts used to instruct the language model in rephrasing queries in zeroshot settings, we conducted experiments using a variety of prompts on arithmetic tasks, including both standard and chain-of-thought prompting. The results shown in Table1 indicate that EchoPrompt consistently enhances performance when compared to the baseline method, regardless of the chosen prompt. However, we observe a difference in per

formance with various prompt selections in the Zero-shot-CoT setting. The prompt “Let’s reiterate the question and also think step by step." achieves the best results.
Effect of rephrases on few-shot EchoPrompt In the few-shot setting, we assess the performance of the proposed rephrase structures compared to baseline techniques, focusing on arithmetic and reading comprehension tasks that require explicit answer generation. The results, as shown in Table3, reveal that although there is variance among the performance, all the rephrase structures outperform the standard and chain-of-thought prompting, highlighting the effectiveness of EchoPrompt. Notably, no single rephrase structure consistently outperforms the others.

# Are rephrased queries self-sufficient?

sess whether the EchoPrompt performance gains are solely due to the rephrased queries or if both the original and rephrased queries are essential, We isolate the LM generated rephrases. This process involves two steps. First, through in-context learning, we generate the rephrased query using the same method as before and with the same exemplars. Then, we prompt the language model with the revised exemplars that match the rephrased query structure. We only provide the rephrased queries for the model to answer. The results in Table4  show that standalone rephrases consistently yield lower accuracies than EchoPrompt. Although rephrased queries can improve accuracy compared to baseline prompting (compound sentence rephrases), the improvements are still considerably lower than those achieved with EchoPrompt. This suggests that the primary source of improve

EchoPrompt
GSM8K
SVAMP
MultiArith
DROP
(Census)
DROP
(Break)
DROP
(Football)
SQuAD(F1)
Standard
-
19.2
69.8
44.0
56.8
55.5
63.7
88.7
Repeat
21.4(+2.2)
75.8(+6.6)
53.8(+9.8)
65.9(+9.1)
63.1(+7.6)
69.2(+5.5)
91.3(+2.6)
Compound
20.8(+1.6)
75.1(+5.3)
54.0(+10.0)
67.3(+10.5)
62.7(+6.9)
67.7(+4.0)
90.6(+1.9)
Question First
20.9(+1.7)
75.0(+5.2)
53.6(+9.6)
65.2(+8.4)
59.7(+3.9)
63.1(−0.6)
92.2(+3.5)
Simple
21.5(+2.3)
76.6(+6.8)
55.6(+11.6)
65.1(+8.3)
63.1(+7.6)
67.1(+3.4)
90.9(+2.2)
CoT
-
61.1
75.2
96.1
70.0
65.3
67.3
90.5
Repeat
63.5(+2.4)
77.6(+2.4)
98.8(+2.7)
71.6(+1.6)
70.0(+4.7)
71.3(+4.0)
-
Compound
65.9(+4.8)
79.0(+3.8)
97.8(+1.7)
75.4(+5.4)
69.6(+4.3)
70.8(+3.5)
90.8(+0.3)
Question First
64.4(+3.3)
77.0(+1.8)
98.3(+2.2)
75.3(+5.3)
68.1(+2.8)
72.0(+4.7)
-
Simple
63.6(+2.5)
76.9(+1.7)
99.0(+2.9)
73.5(+3.5)
67.7(+2.4)
71.2(+3.9)
-
Table 3: code-davinci-002 Evaluation of EchoPrompt using the proposed rephrase structures and query-repetition.
Table 3: code-davinci-002 Evaluation of EchoPrompt using the proposed rephrase structures and query-repetition. We compare these approaches with baseline methods in arithmetic and reading comprehension tasks. The results showcase improvements across all rephrase structures, with no single structure consistently outperforming the others.

ment in EchoPrompt lies in the provision of two query versions.

Comparing the rephrase and the original queries We compare the BLEU scores for the rephrased queries alongside the original ones (refer to Table16 in the Appendix). Additionally, we compute the fraction of tokens retained in the rephrased queries (see Table15 in the Appendix). In numerical tasks, the rephrases retain most of the information from the original queries. However, we observe considerable differences in scores in the standalone rephrases in reading comprehension tasks, particularly in the DROP Football and Break subsets. In these datasets, the original queries exhibit a huge variance in the token count distribution, leading to low-quality rephrase generation, which may be why we observe a significant drop in accuracy.

To

# Generating vs Augmenting the rephrases

study whether EchoPrompt can be considered as a query augmentation technique, we compare the performance of EchoPrompt with directly augmenting the original question using a rephrase (generated in Section5). In EchoPrompt, the model generates both the rephrase and the answer simultaneously, while in query augmentation, the query is provided to the language model beforehand, and the model only generates the answer. Table18 (in Appendix) shows an example highlighting the distinction between the two settings. The result of this experiment is summarized in Table5, demonstrating that both approaches yield comparable improvements in accuracy. This result indicates that although we introduce EchoPrompt as a subtask within incontext learning, it can also be considered a query

augmentation technique. This is because the language model utilizes the same rephrased query and the original query to solve the query in both cases.

Stacking multiple rephrases for EchoPrompt The benefits observed with query-rephrasing in EchoPrompt naturally prompted us to investigate the effects of having the language model generate multiple rephrases. The summarized results in Table6 show a drop in performance as the number of rephrases increases. When manually examining the generated answers, we observe a tendency towards repetition in the chain-of-thought reasoning despite successfully generating the desired number of rephrases. This repetition phenomenon becomes particularly prominent when the question requires longer multi-hop reasoning. The Appendix shows Examples illustrating this finding in Table
17. This observation aligns with expectations since the task’s focus shifts from chain-of-thought reasoning to rephrase generation when the number of rephrases is increased in EchoPrompt. Consequently, the model prioritizes generating the requested number of rephrases rather than the reasoning process.

# Robustness to irrelevant text

et al., 2023) has shed light on the sensitivity of large language models (LLMs) to irrelevant information using various prompting methods, including the CoT reasoning. Intuitively, EchoPrompt could be particularly prone to such distractions, given that it rephrases or regenerates the query, including the distractions. To evaluate if EchoPrompt technique works even in the presence of such perturbations, we study the performance of EchoPrompt on GSMIC-4k dataset (Shi et al., 2023). The evalua

Query Structure
GSM8K
SVAMP
DROP
(Census)
DROP
(Break)
DROP
(Football)
Standard
Original
19.2
69.8
56.8
55.5
63.7
Compound
19.9(+0.7)
71.8(+2.0)
59.1(+2.3)
54.1(−1.4)
65.1(+1.4)
Question First
14.6(−4.6)
58.5(−11.3)
28.2(−28.6)
36.2(−19.3)
48.8(−14.9)
Simple
19.7(+0.5)
70.9(+1.1)
56.5(−0.3)
55.5(+0.0)
62.7(−1.0)
Standard+ Repeat
-
21.5
76.6
65.1
63.1
67.1
CoT
Original
61.1
75.2
69.6
65.3
67.3
Compound
62.1(+1.0)
78.0(+2.8)
71.9(+2.3)
66.7(+1.4)
68.2(+0.9)
Question First
55.1(−6.0)
66.6(−8.6)
48.1(−21.5)
64.5(−0.8)
57.8(−9.5)
Simple
61.3(+0.2)
75.8(+0.6)
70.3(+0.7)
67.3(+2.0)
67.1(−0.2)
CoT+ Compound
-
65.9
79.0
74.3
69.6
70.8
 Standalone Rephrases: code-davinci-002
Table 4: Standalone Rephrases: code-davinci-002 Compound Sentence rephrasing performs original queries, while question-first rephrasing performs worse. We observe information loss in certain tasks (see Table15), indicating that the performance gains of EchoPrompt are due to the rephrasing and having multiple versions.

GSM8K SVAMP DROP
Repeat
SubTask
63.5
77.6
70.0
Augment
63.4
76.3
69.3
Compound
SubTask
65.9
79.0
69.6
Augment
64.2
77.2
69.7
 code-davinci-002
<div style="text-align: center;">Table 5: code-davinci-002 A comparison between EchoPrompt and query augmentation, indicating similar performance improvements for both approaches.
</div>
times
GSM8K
SVAMP
DROP
Repeat
1
63.5
77.6
70.3
2
61.7
77.6
68.5
3
59.8
77.8
69.3
5
59.9
76.9
67.5
Compound
1
65.9
79.0
69.6
2
63.7
77.9
68.8
3
63.2
78.9
67.9
able 6: code-davinci-002 The accuracies drop as the
Table 6: code-davinci-002 The accuracies drop as the number of rephrases/repetitions increases when generating multiple rephrases with EchoPrompt.

Table 6: code-davinci-002 The accuracies drop as the number of rephrases/repetitions increases when generating multiple rephrases with EchoPrompt.

tion results in Table 7 demonstrate that EchoPrompt maintains improvements across all prompting techniques, even in the presence of perturbations.

# 6 Related Work

Prompting Large language models’ success has sparked interest in improving task performances through prompting techniques (Brown et al., 2020). While the recent studies focus on task-based instruction tuning, either by fine-tuning the entire model (Raffel et al., 2020; Wei et al., 2021; Sanh et al., 2021; Wang et al., 2022b; Huang et al., 2022) or maintaining task-specific parameters (Li and Liang, 2021; Lester et al., 2021), our work is a general prompting approach that improves the incontext learning abilities and does not require any fine-tuning.

Standard
CoT
LTM
EchoPrompt?
✗
✓
✗
✓
✗
✓
Zero-shot
23.7
30.1
(+6.4)
46.7
52.8
(+6.1)
N/A
N/A
1-shot
27.1
29.1
(+2.0)
72.6
77.2
(+4.6)
73.8
81.3
(+7.5)
4-shot
25.2
31.0
(+5.8)
77.4
81.8
(+4.4)
84.3
85.4
(+1.1)
Table 7: code-davinci-002 Performance of EchoPrompt
Table 7: code-davinci-002 Performance of EchoPrompt on GSMIC-4k(which contains irrelevant context in queries). EchoPrompt improves performance on both chain-of-thought and least-to-most prompting, even though it repeats the perturbation sentence in the rephrase.

Intermediate steps The concept of employing language models to generate intermediate steps for process supervision has been extensively examined in the context of solving reasoning tasks, whether through training (Nye et al., 2021;  Zelikman et al., 2022), zero-shot (Kojima et al., 2022), few-shot prompting (Wei et al., 2022) or action planning(Yao et al., 2022). Recent works focus on problem decomposition and teaching the language model to answer the subtasks, to eventually answer complex problems (Zhou et al., 2022a; Dua et al., 2022; Wang et al., 2022a; Zhou et al., 2022b). EchoPrompt is orthogonal to these approaches, augmenting the input query rather than rationale generation. Consequently, it can be easily extended with any of these prompting strategies.

Interpretability, Consistency and Outcome correction Another related research direction involves exploring interpretability and consistency in the rationale generated by large-scale models. Recent works (Imani et al., 2023; Miao et al.,

EchoPrompt?
WinoGrande
StrategyQA
-
71.9
74.8
Repeat
71.9
75.3
Compound
70.8
74.5
Table 8: code-davinci-002 EchoPrompt with Standard
2023; Madaan and Yazdanbakhsh, 2022) help improve the interpretability in arithmetic and reasoning tasks through validation. Although these approaches are not directly tied to the EchoPrompt technique, they utilize chain-of-thought prompting, where we have shown that EchoPrompt exhibits promising results, particularly in zero-shot scenarios. In the domain of outcome correction, approaches such as (Jung et al., 2022; Wang et al., 2023; Yao et al., 2023; Miao et al., 2021; Xie et al., 2023) leverage consistency among multiple generated rationales while (Weng et al., 2023; Khalifa et al., 2023; Yang and Klein, 2021; Ni et al., 2023; Chen et al., 2022) prioritize the ranking of plausible generations to enhance performance across arithmetic, reasoning, and code-generation tasks. Building upon these foundations, self-correction methodologies like (Madaan et al., 2023; Jiang et al., 2023; Hao et al., 2023; Shinn et al., 2023), which employ feedback loops for refinement and multi-agent debating strategies (Du et al., 2023; Cohen et al., 2023; Fu et al., 2023) have evolved. EchoPrompt distinguishes itself from these approaches by focusing on single rationale generation rather than considering multiple generated responses.

# 7 Limitations

While the EchoPrompt subtask presents notable advantages, several limitations exist. Although we provide several ablation studies and qualitative examples, answering the question of when EchoPrompt works better, we could not explain why EchoPrompt results in performance gains, particularly in standard prompting. Additionally, it is worth noting that our approach involves regenerating the entire query before solving the tasks. Consequently, the model must generate many tokens when dealing with long queries, leading to increased compute requirements and time delays.

# 8 Conclusion

We have proposed EchoPrompt, a simple yet effective approach that builds upon existing prompting approaches and integrates query-rephrasing as a

subtask in the in-context learning process inspired by how humans think. It enables the language model to recall the query before attempting to solve it. EchoPrompt offers a direct approach to enhance in-context learning in pre-trained language models without fine-tuning, making it a simple and powerful approach to achieve performance boosts.

# 9 Reproducibility Statement

Our primary results are on Code-davinci-002 and GPT-3.5-Turbo, which are publicly accessible OpenAI models. To increase reproducibility, we have included prompts used for all the tasks in the Appendix. We also plan to release the code soon.

# References

2019. Winogrande: An adversarial winograd schema challenge at scale.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Zieglxer, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners.
Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. 2022. Codet: Code generation with generated tests. arXiv preprint arXiv:2207.10397.
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021.  Training verifiers to solve math word problems.
Roi Cohen, May Hamri, Mor Geva, and Amir Globerson. 2023. Lm vs lm: Detecting factual errors via cross examination.
Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. 2023. Improving factuality and reasoning in language models through multiagent debate.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Zieglxer, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners.
Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. 2022. Codet: Code generation with generated tests. arXiv preprint arXiv:2207.10397.
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021.  Training verifiers to solve math word problems.
Roi Cohen, May Hamri, Mor Geva, and Amir Globerson. 2023. Lm vs lm: Detecting factual errors via cross examination.
Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. 2023. Improving factuality and reasoning in language models through multiagent debate.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. 2023. Improving factuality and reasoning in language models through multiagent debate.

Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner. 2022. Successive prompting for decomposing complex questions. arXiv preprint arXiv:2212.04092.
Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs.
Yao Fu, Hao Peng, Tushar Khot, and Mirella Lapata. 2023. Improving language model negotiation with self-play and in-context learning from ai feedback.
Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346– 361.
Ahmad Ghazal, Tilmann Rabl, Minqing Hu, Francois Raab, Meikel Poess, Alain Crolotte, and Hans-Arno Jacobsen. 2013. Bigbench: Towards an industry standard benchmark for big data analytics. In  Proceedings of the 2013 ACM SIGMOD international conference on Management of data, pages 1197–1208.
Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. 2023. Reasoning with language model is planning with world model.
Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. 2021a. Aligning ai with shared human values.  Proceedings of the International Conference on Learning Representations (ICLR).
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021b. Measuring massive multitask language understanding.  Proceedings of the International Conference on Learning Representations (ICLR).
Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. 2022. Large language models can self-improve.
Shima Imani, Liang Du, and Harsh Shrivastava. 2023. Mathprompter: Mathematical reasoning using large language models. arXiv preprint arXiv:2303.05398.
Shuyang Jiang, Yuhao Wang, and Yu Wang. 2023.  Selfevolve: A code evolution framework via large language models.
Laurice M Joseph, Sheila Alber-Morgan, Leigh Ann Amspaugh, Kelsey Ross, Maria Helton, Moira Konrad, and Carrie Davenport. 2019. Stop to ask and respond: Effects of a small-group self-questioning intervention on reading comprehension performance. Research and Practice in the Schools: The Official Journal of the Texas Association of School Psychologists, 6(1):27.

Laurice M Joseph and Kelsey M Ross. 2018. Teaching middle school students with learning disabilities to comprehend text using self-questioning. Intervention in School and Clinic, 53(5):276–282.
Jaehun Jung, Lianhui Qin, Sean Welleck, Faeze Brahman, Chandra Bhagavatula, Ronan Le Bras, and Yejin Choi. 2022. Maieutic prompting: Logically consistent reasoning with recursive explanations. arXiv preprint arXiv:2205.11822.
Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, and Lu Wang. 2023.
Discriminator-guided multi-step reasoning with language models.
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916.
Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.
The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. 2023. Starcoder: may the source be with you!
Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582– 4597, Online. Association for Computational Linguistics.
Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blun

Laurice M Joseph and Kelsey M Ross. 2018. Teaching middle school students with learning disabilities to comprehend text using self-questioning. Intervention in School and Clinic, 53(5):276–282.
Jaehun Jung, Lianhui Qin, Sean Welleck, Faeze Brahman, Chandra Bhagavatula, Ronan Le Bras, and Yejin Choi. 2022. Maieutic prompting: Logically consistent reasoning with recursive explanations. arXiv preprint arXiv:2205.11822.
Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, and Lu Wang. 2023.
Discriminator-guided multi-step reasoning with language models.
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916.
Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.
The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. 2023. Starcoder: may the source be with you!
Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582– 4597, Online. Association for Computational Linguistics.

problems. In  Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 158–167, Vancouver, Canada. Association for Computational Linguistics.
Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. 2020. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. arXiv preprint arXiv:2007.08124.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback.
Aman Madaan and Amir Yazdanbakhsh. 2022. Text and patterns: For effective chain of thought, it takes two to tango. arXiv preprint arXiv:2209.07686.
Ning Miao, Yee Whye Teh, and Tom Rainforth. 2023.
Selfcheck: Using llms to zero-shot check their own step-by-step reasoning.
Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2021. A diverse corpus for evaluating and developing english math word problem solvers.
Ansong Ni, Srini Iyer, Dragomir Radev, Ves Stoyanov, Wen tau Yih, Sida I. Wang, and Xi Victoria Lin. 2023. Lever: Learning to verify language-to-code generation with execution.
Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. 2021. Show your work: Scratchpads for intermediate computation with language models. arXiv preprint arXiv:2112.00114.
OpenAI. 2021. Chatgpt. [ChatGPT].
Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are nlp models really able to solve simple math word problems?
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text.
Subhro Roy and Dan Roth. 2016.  Solving general arithmetic word problems.
Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. 2021. Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207.

problems. In  Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 158–167, Vancouver, Canada. Association for Computational Linguistics.
Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. 2020. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. arXiv preprint arXiv:2007.08124.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback.
Aman Madaan and Amir Yazdanbakhsh. 2022. Text and patterns: For effective chain of thought, it takes two to tango. arXiv preprint arXiv:2209.07686.
Ning Miao, Yee Whye Teh, and Tom Rainforth. 2023.
Selfcheck: Using llms to zero-shot check their own step-by-step reasoning.
Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2021. A diverse corpus for evaluating and developing english math word problem solvers.
Ansong Ni, Srini Iyer, Dragomir Radev, Ves Stoyanov, Wen tau Yih, Sida I. Wang, and Xi Victoria Lin. 2023. Lever: Learning to verify language-to-code generation with execution.
Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. 2021. Show your work: Scratchpads for intermediate computation with language models. arXiv preprint arXiv:2112.00114.
OpenAI. 2021. Chatgpt. [ChatGPT].
Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are nlp models really able to solve simple math word problems?
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text.
Subhro Roy and Dan Roth. 2016.  Solving general arithmetic word problems.
Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. 2021. Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language models can be easily distracted by irrelevant context.
Noah Shinn, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023.  Reflexion: Language agents with verbal reinforcement learning.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models.
Ben Wang and Aran Komatsuzaki. 2021. GPT-J6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/ mesh-transformer-jax.
Boshi Wang, Xiang Deng, and Huan Sun. 2022a. Iteratively prompt pre-trained language models for chain of thought. arXiv preprint arXiv:2203.08383.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models.
Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Gary Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Maitreya Patel, Kuntal Kumar Pal, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Shailaja Keyur Sampat, Savan Doshi, Siddhartha Mishra, Sujan Reddy, Sumanta Patro, Tanay Dixit, Xudong Shen, Chitta Baral, Yejin Choi, Noah A. Smith, Hannaneh Hajishirzi, and Daniel Khashabi. 2022b. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks.
Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023.  Chain-of-thought prompting elicits reasoning in large language models.
Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Kang Liu, and Jun Zhao. 2023. Large language models are better reasoners with self-verification.

Tomer Wolfson, Mor Geva, Ankit Gupta, Matt Gardner, Yoav Goldberg, Daniel Deutch, and Jonathan Berant. 2020. Break it down: A question understanding benchmark. Transactions of the Association for Computational Linguistics, 8:183–198.
Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, Xu Zhao, MinYen Kan, Junxian He, and Qizhe Xie. 2023.  Decomposition enhances reasoning via self-evaluation guided decoding.
Kevin Yang and Dan Klein. 2021. FUDGE: Controlled text generation with future discriminators. In  Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3511–3535, Online. Association for Computational Linguistics.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.
Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. 2022. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488.
Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and Ed Chi. 2022a. Least-to-most prompting enables complex reasoning in large language models.
Hattie Zhou, Azade Nova, Hugo Larochelle, Aaron Courville, Behnam Neyshabur, and Hanie Sedghi. 2022b. Teaching algorithmic reasoning via incontext learning. arXiv preprint arXiv:2211.09066.

Model
Dataset
zero-shot
few-shot
Standard
CoT
Standard
CoT
EchoPrompt ?
✗
✓
✗
✓
✗
✓
✗
✓
Code-davinci-002
GSM8K
16.4 20.7(+4.3)
49.3 51.1(+1.8)
19.2
21.4(+2.2)
61.1
65.9(+4.8)
SVAMP
66.8 74.7(+7.9)
66.5 73.9(+7.4)
69.8
75.8(+6.0)
75.2
79.0(+3.8)
MultiArith
31.0 48.5(+17.5)
76.0 78.7(+2.7)
44.0
53.8(+9.8)
96.1
97.8(+1.7)
SingleOp
91.6 91.8(+0.2)
82.9 92.4(+9.5)
93.2
94.2(+1.0)
92.8
94.7(+1.9)
Shuffled Objects
36.4 35.2(−1.2)
42.4 58.2(+15.8)
34.8
36.7(+1.9)
66.0
68.9(+2.9)
Coin Flip
47.7 47.2(−0.5)
58.5 60.1(+1.6)
99.6
100.0(+0.4)
100.0 100.0(+0.0)
Date
44.2 43.8(−0.4)
39.0 46.8(+7.8)
49.3
50.4(+1.1)
65.6
68.1(+2.5)
DROP(Football)
50.8 58.3(+7.5)
44.1 58.0(+13.9)
63.7
69.2(+5.5)
67.3
70.8(+3.5)
DROP(Nonfootball)
43.2 57.1(+13.9)
39.7 52.6(+12.9)
57.1
63.3(+6.2)
69.2
72.2(+3.0)
DROP(Census)
45.9 66.3(+20.4)
30.0 53.3(+23.3)
56.8
65.9(+9.1)
69.6
75.4(+5.8)
DROP(Break)
43.7 55.8(+12.1)
38.2 51.2(+13.0)
55.5
63.1(+7.6)
65.3
69.6(+4.3)
SQuAD(F1)
65.7 69.8(+4.1)
52.6 54.4(+1.8)
88.7
91.3(+2.6)
90.5
90.8(+0.3)
AQUA-RAT
21.2 23.3(+2.1)
37.0 35.4(−1.6)
30.3
29.9(−0.4)
43.7
41.3(+2.4)
MMLU-h
31.8 36.7(+4.9)
42.5 41.7(−0.8)
36.7
39.3(+2.6)
44.1
42.1(−2.0)
logiQA
42.5 41.6(−0.9)
37.0 40.9(+3.9)
45.3
46.6(+1.3)
40.9
41.0(+0.1)
GPT-3.5
(Turbo)
GSM8K
5.6
24.8(+19.2)
75.7 76.4(+0.7)
31.3
32.1(+0.8)
75.1
83.5(+8.4)
SVAMP
51.9 76.0(+24.1)
80.5 83.5(+3.0)
76.1
78.4(+2.3)
77.4
81.9(+5.5)
MultiArith
76.5 83.7(+7.2)
93.4 96.3(+2.9)
83.4
90.5(+7.1)
97.8
98.5(+0.7)
SingleOp
92.6 96.8(+4.2)
91.4 94.8(+3.4)
93.9
96.2(+2.3)
95.7
96.5(+0.8)
Shuffled Objects
26.9 21.6(−5.3)
79.5 82.2(+2.7)
30.6
34.6(+4.0)
68.8
74.3(+5.5)
Coin Flip
76.7 86.8(+10.1)
99.8 98.6(−1.2)
90.0
95.6(+5.6)
100.0 100.0(+0.0)
Date
45.7 44.1(−1.6)
46.6 45.8(−0.6)
50.4
49.3(−1.1)
64.5
66.2(+1.7)
DROP(Break)
47.1 52.9(+5.8)
51.9 51.0(−0.9)
59.9
62.7(+2.8)
61.6
66.5(+4.9)
SQuAD(F1)
79.1 80.6(+1.5)
62.1 58.3(−3.8)
76.4
83.2(+6.8)
85.3
86.1(+0.8)
AQUA-RAT
27.9 28.4(+0.5)
51.1 50.8(−0.3)
33.4
35.8(+2.4)
39.7
57.1(+17.4)
MMLU-h
25.6 31.1(+5.5)
51.1 52.9(+1.8)
34.1
34.8(+0.7)
28.9
41.1(+12.2)
logiQA
36.2 38.2(+2.0)
37.6 39.0(+1.4)
45.1
43.3(−1.8)
32.5
32.3(−0.2)
Starcoder
(15B)
SingleOp
63.1 66.9(+3.8)
53.5 66.5(+13.0)
64.0
70.1(+6.1)
68.8
73.6(+4.8)
SVAMP
35.6 37.9(+2.3)
30.9 36.7(+5.8)
32.4
37.2(+4.8)
30.2
36.2(+6.0)
Coin Flip
55.4 54.3(−1.1)
51.6 51.0(−0.6)
98.6
99.8(+1.2)
100.0 100.0(+0.0)
Date
15.9 19.2(+3.3)
20.6 19.9(−0.7)
24.4
26.6(+2.2)
38.4
33.8(−4.6)
Llama
(13B)
SingleOp
78.4 81.1(+2.7)
64.9 73.0(+8.1)
81.1
83.3(+2.2)
81.3
80.6(−0.7)
SVAMP
36.4 46.3(+9.9)
30.7 34.0(+3.3)
39.2
43.0(+3.8)
38.7
41.3(+2.6)
Coin Flip
53.2 51.3(−1.8)
51.0 51.0(+0.0)
89.8
92.7(+2.9)
100.0 100.0(+0.0)
Date
24.9 26.6(+1.7)
22.5 23.0(+0.5)
32.8
30.1(−1.7)
42.3
40.9(−1.4)
GPT-J
(6B)
SingleOp
-
-
-
-
37.2
39.9(+2.7)
45.3
44.5(−0.8)
SVAMP
-
-
-
-
8.9
10.1(+1.2)
21.1
19.8(−1.3)
Coin Flip
-
-
-
-
81.3
81.3(+0.0)
80.6
96.4(+15.8)
Date
-
-
-
-
13.2
13.6(+0.4)
11.1
15.8(+4.7)
Table 9: Performance Summary of EchoPrompt on all models. EchoPrompt consistently improves performance
Rephrase
Prompt
Compound
Rephrase the following query using compound sentences without loss of details,
starting with “Given that" and ending with the question in the query:
<Question>
Question First
Rephrase the following query by asking the question in the query first, without loss
of details:
<Question>
Short and simple sentences
Rephrase the following query using short and simple sentences, without loss of
details:
<Question>
Table 10: Prompts used to create rephrases for exemplars, using ChatGPT
EchoPrompt?
Stage-1 Prompt
DROP
(Football)
DROP
(Nonfootball)
DROP
(Census)
DROP
(Break)
zero-shot
✗
-
50.8
43.2
46.4
43.7
✓
Let’s repeat the complete question. “
58.3(+7.5)
57.1(+13.9)
66.3(+19.9)
55.8(+12.1)
✓
Let’s reiterate the complete question. “
57.0(+6.2)
56.9(+13.7)
66.3(+19.9)
54.1(+10.4)
✓
Let’s restate the complete question. “
60.5(+9.7)
57.1(+13.9)
66.7(+20.3)
56.2(+12.5)
✓
Let’s summarize the complete question. “
59.6(+8.8)
55.6(+12.4)
63.9(+17.5)
54.2(+10.5)
Zero-shot-CoT
✗
Let’s think step by step.
44.1
39.7
30.0
38.2
✓
Let’s repeat the complete question and also think step
by step.
58.0(+13.9)
52.6(+12.9)
53.3(+23.3)
51.2(+13.0)
✓
Let’s reiterate the complete question and also think
step by step.
53.1(+9.0)
53.6(+13.9)
53.1(+23.1)
50.8(+12.6)
✓
Let’s repeat the complete question and also think step
by step. “
51.4(+7.3)
51.7(+12.0)
46.3(+16.3)
48.0(+9.8)
✓
Let’s restate the complete question and also think
step by step.
51.6(+7.5)
51.7(+12.0)
48.1(+18.1)
50.0(+11.8)
✓
Let’s summarize the complete question and also think
step by step.
51.4(+7.3)
52.4(+12.7)
51.5(+21.5)
48.3(+10.1)
Table 11: Code-davinci-002: Reading Comprehension This table compares the performance of the propose
EchoPrompt?

AQuA-RAT
Date
DROP
(Non-football)
Standard
30.3
49.3
57.1
Standard+ Repeat
29.0
50.4
63.3
CoT
43.7
65.6
69.2
CoT+ Repeat
40.9
67.8
71.9
CoT + Compound
41.3
68.0
72.2
LTM
-
-
66.2
Table 12: code-davinci-002: EchoPrompt extended resul

Dataset
Approach
2
3
4
5
>=6
2
3
4
5
>=6
GSM8K
COT
77.3
67.6
56.5
51.2
29.1
81.3
71.1
62.2
61.5
39.7
EchoPrompt+CoT
84.3
71.8
60.8
55.2
36.4
84.3
72.4
67.2
64.9
39.8
LTM
78.8
68.3
57.8
55.7
30.4
81.9
74.1
60.2
62.6
41.1
DROP
COT
-
66.7
56.9
60.5
76.3
-
74.2
63.2
69.4
77.3
(break)
EchoPrompt+CoT
-
70.4
59.8
67.2
79.3
-
71.1
66.7
71.6
78.8
LTM
-
61.7
63.2
60.5
71.8
-
62.9
63.2
60.5
72.7
Table 13: code-davinci-002: EchoPrompt extended results
able 13: code-davinci-002: EchoPrompt extended results

GSM8k
SVAMP
MultiArith
DROP
(Non-football)
DROP
(Break)
Standard
-
17.0
67.6
39.5
60.5
56.1
Repeat
16.7(−0.3)
72.0(+4.4)
46.0(+6.5)
65.9(+4.4)
63.3(+7.2)
Compound
19.2(+2.2)
72.2(+4.6)
51.3(+11.8)
65.7(+4.2)
61.3(+5.2)
Question First
18.4(+1.4)
71.1(+3.5)
53.1(+12.6)
60.9(+0.4)
56.9(+0.8)
Simple
18.2(+1.2)
72.4(+4.8)
51.6(+12.1)
64.8(+4.3)
59.9(+3.8)
CoT
-
66.9
74.7
92.8
75.9
70.6
Repeat
68.2(+1.3)
75.4(+0.7)
96.8(+4.0)
78.1(+2.2)
72.0 (+1.4)
Compound
69.3(+2.4)
76.4(+1.7)
95.0(+2.2)
74.1 (−1.8)
67.9(−2.7)
Question First
68.4 (+1.5)
76.2(+1.5)
95.6(+2.8)
76.9 (+1.0)
72.2(+1.6)
Simple
68.2 (+1.3)
75.3(+0.6)
95.3(+2.5)
77.3(+1.4)
71.6 (+1.0)
ble 14: Code-davinci-002: Effect of exemplar selection: While Table3 utilized exemplars proposed in (W al., 2023; Zhou et al., 2022a) that showcase simpler reasoning, this table employs exemplars demonstrating long asoning-chains. The results indicate that although EchoPrompt provides higher gains with simpler exempla oosing better exemplars achieves higher overall accuracies, highlighting the significance of exemplar selection

Query Structure
GSM8K
SVAMP
DROP(Census)
DROP(Break)
DROP(football)
Original
56.8
37.4
215.4
271.8
319.3
Compound
52.1
37.1
218.6
236.9
294.3
Question First
47.9
32.9
212.2
212.6
154.1
Simple
56.7
37.3
192.2
237.1
217.0
Table 15: Rephrases - Token counts The table presents a comparison of token counts in the model-generated
ble 15: Rephrases - Token counts The table presents a comparison of token counts in the model-genera phrases and the original query. We observe a significant token loss in DROP subsets.

Query Structure
GSM8K
SVAMP
DROP(Census)
DROP(Break)
DROP(football)
Compound
70.5
64.5
76.6
70.1
76.7
Question First
63.2
62.8
92.9
50.6
20.8
Simple
99.3
98.9
74.1
78.2
40.3
Table 16: Rephrases - BLEU Scores The table compares BLEU scores between the model-generated rephrases and
Query Structure
GSM8K
SVAMP
DROP(Census)
DROP(Break)
DROP(football)
Compound
70.5
64.5
76.6
70.1
76.7
Question First
63.2
62.8
92.9
50.6
20.8
Simple
99.3
98.9
74.1
78.2
40.3
Table 16: Rephrases - BLEU Scores The table compares BLEU scores between the model-generated rephrases and
the original query. The BLEU scores for numerical tasks are high, indicating good similarity between the rephrases
and the original query. However, for reading comprehension tasks, the BLEU scores of the rephrases experience a
significant drop.
ble 16: Rephrases - BLEU Scores The table compares BLEU scores between the model-generated rephrases a e original query. The BLEU scores for numerical tasks are high, indicating good similarity between the rephras d the original query. However, for reading comprehension tasks, the BLEU scores of the rephrases experience

<div style="text-align: center;">mples of queries that lead to repetition, when the language model is prompted to generate 2 repetitions
</div>
Question
Response
Q: Shiloh is 44 years old today. In 7 years, he will be three
times as old as his nephew. How old is his nephew today?
A: Rewriting in simple words, the question is: “Shiloh is 44 years old
today. In 7 years, he will be three times as old as his nephew. How old
is his nephew today?"
Repeating the question again, the question is: “Shiloh is 44 years old
today. In 7 years, he will be three times as old as his nephew. How old
is his nephew today?"
Now, to answer the rewritten question, Shiloh is 44 years old today. In
7 years, he will be three times as old as his nephew. How old is his
nephew today?
Shiloh is 44 years old today. In 7 years, he will be three times as old as
his nephew. How old is his nephew today?
Shiloh is 44 years old today. In 7 years, he will be three times as old as
his nephew. How old is his nephew today?...
Q: The cell-phone recharges at the rate of 1 percentage-
point of charge per 3 minutes. Now, the phone is at 60%
charged. How long will it take to fully charge, in hours?
A: Rewriting in simple words, the question is: “The cell-phone
recharges at the rate of 1 percentage-point of charge per 3 minutes.
Now, the phone is at 60% charged. How long will it take to fully charge,
in hours?"
Repeating the question again, the question is:
“The cell-phone
recharges at the rate of 1 percentage-point of charge per 3 minutes.
Now, the phone is at 60% charged. How long will it take to fully charge,
in hours?"
Now, to answer the rewritten question, The phone is at 60% charged. It
takes 3 minutes to charge 1 percentage-point. So it will take 3 minutes
to charge 1 percentage-point. So it will take 3 minutes to charge 1
percentage-point...
Q: Ian has a board that is 40 feet long. He decides to make
a cut so he can have two pieces. The longer piece is 4
times longer than the shorter piece. How long is the longer
piece?
A: Rewriting in simple words, the question is: “Ian has a board that is
40 feet long. He decides to make a cut so he can have two pieces. The
longer piece is 4 times longer than the shorter piece. How long is the
longer piece?"
Repeating the question again, the question is: “Ian has a board that is
40 feet long. He decides to make a cut so he can have two pieces. The
longer piece is 4 times longer than the shorter piece. How long is the
longer piece?"
Now, to answer the rewritten question, Ian has a board that is 40 feet
long. He decides to make a cut so he can have two pieces. The longer
piece is 4 times longer than the shorter piece. How long is the longer
piece?
The longer piece is 4 times longer than the shorter piece. So the longer
piece is 4 x the shorter piece. The shorter piece is the board minus
the longer piece. So the shorter piece is 40 - the longer piece. So the
longer piece is 4 x (40 - the longer piece).
So the longer piece is 4 x (40 - the longer piece).
So the longer piece is 4 x (40 - the longer piece).
So the longer piece is 4 x (40 - the longer piece).
Q: Shiloh is 44 years old today. In 7 years, he will be three times as old as his nephew. How old is his nephew today?

Q: The cell-phone recharges at the rate of 1 percentagepoint of charge per 3 minutes. Now, the phone is at 60% charged. How long will it take to fully charge, in hours?

Q: The cell-phone recharges at the rate of 1 percentagepoint of charge per 3 minutes. Now, the phone is at 60% charged. How long will it take to fully charge, in hours?

Q: Ian has a board that is 40 feet long. He decides to make a cut