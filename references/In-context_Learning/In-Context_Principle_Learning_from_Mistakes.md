# In-Context Principle Learning from Mistakes
Tianjun Zhang * 1 Aman Madaan * 2 Luyu Gao * 2 Steven Zheng 3 Swaroop Mishra 3 Yiming Yang 2 Niket Tandon 4 Uri Alon 3
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9f82/9f8273d3-a9d2-459b-ab5d-fa2086a1adb8.png" style="width: 50%;"></div>
Learned Principle
Benchmark
When solving a problem involving multiple
sources of income and expenses, it is crucial
to keep track of each component separately
and calculate the total accurately.
GSM8K
When simplifying complex numbers raised
to powers, it is important to remember the
following rules: (1) i2 = −1 (2) i3 = −i (3)
i4 = 1 (4) i−1 = 1
i = −i
MATH
Perform calculations using the full precision
available and only round as a final step, if
necessary.
DROP
When answering questions about commonal-
ities between two entities, it is important to
consider all relevant aspects and not just the
most obvious or prominent one.
HotpotQA
Double negation, as in ‘not not’, cancels out
and returns the original value.
Boolean Expr-
essions (BBH)
Sarcasm often involves saying the opposite
of what is meant ... Paying attention to the
incongruity between the literal meaning of
the words and the intended meaning can help
in accurately identifying sarcasm.
Snarks
(BBH)
# Abstract
In-context learning (ICL, also known as fewshot prompting) has been the standard method of adapting LLMs to downstream tasks, by learning from a few input-output examples. Nonetheless, all ICL-based approaches only learn from correct input-output pairs. In this paper, we revisit this paradigm, by learning more from the few given input-output examples. We introduce Learning Principles (LEAP): First, we intentionally induce the model to make mistakes on these few examples; then the model itself reflects on these mistakes, and learn explicit taskspecific “principles” from them without any human supervision, which help solve similar problems and avoid common mistakes; finally, we prompt the model to answer unseen test questions using the original few-shot examples and these learned general principles. We evaluate LEAP on a wide range of benchmarks, including multi-hop question answering (Hotpot QA), textual QA (DROP), Big-Bench Hard reasoning, and math problems (GSM8K and MATH); in all these benchmarks, LEAP improves the strongest available LLMs such as GPT-3.5-turbo, GPT-4, GPT-4-turbo and Claude-2.1. For example, LEAP improves over the standard few-shot prompting using GPT-4 by 7.5% in DROP, and by 3.3% in HotpotQA. Importantly, LEAP does not require any more input or examples than the standard few-shot prompting settings.
arXiv:2402.05403v2
# 1. Introduction
# The rise of large language models (LLMs; Radford et al., 2019; Chowdhery et al., 2022; Zhang et al., 2022; Li et al., 2022; Anil et al., 2023; Touvron et al., 2023a;b) that are
*Equal contribution 1UC Berkeley 2Carnegie Mellon University 3Google DeepMind 4AI2. Correspondence to: Tianjun Zhang <tianjunz@berkeley.edu>, Aman Madaan <amadaan@cs.cmu.edu>, Uri Alon <urialon@google.com>.
Figure 1: Examples for learned principles using LEAP, with key idea of each principle highlighted.
too costly to finetune for downstream tasks has led to the growing popularity of in-context learning (ICL), also known as few-shot prompting (Brown et al., 2020; Liu et al., 2023; Wei et al., 2023). In in-context learning, the LLM is provided with a few (e.g., three) input-output task-specific examples in its prompt, along with an unseen test input. Using this emergent ability (Wei et al., 2022b), the LLM is then expected to generate output for the test input. The LLM generates this output by implicitly learning the task from the few given examples, at inference time. ICL was shown to be extremely effective and data-efficient across a variety of tasks and domains (Min et al., 2022a; Alayrac et al., 2022; Liu et al., 2021; Lu et al., 2023), mainly because it allows for downstream task-adaptation without training. Further, ICL enables generalization using only a few annotated examples. Nonetheless, all ICL-based ap-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/44d0/44d06af7-8e83-4f67-ae9a-ffb45d0651c4.png" style="width: 50%;"></div>
<div style="text-align: center;">Few-shot Chain-of-Thought (Wei et al., 2022)</div>
Figure 2: An illustration of LEAP: Given a few input-output examples, Chain-of-Thought (left) generates a response to the test question by directly learning from the (correct) examples. In contrast, Learning Principles (LEAP, right) first (a) generates mistaken zero-shot Chain-of-Thought response for each given input-output example by sampling with a non-zero temperature; (b) generates explicit principles by providing the LLM with the mistaken COT along with the correct output; and finally (c) generates a response to the test question, by providing the LLM with both the given input-output examples and the learned principles. Note that steps (a) and (b) are performed once per task.
proaches only learn from correct input-output pairs (Min et al., 2022b); in contrast, learning from mistakes is a foundational aspect of the way humans learn (Edmondson, 1996; Chialvo & Bak, 1999; Edmondson, 1999). In fact, learning from mistakes is also a fundamental concept of machine learning, which goes back to classical work such as Wiener (1948); Rosenblatt (1957) and Minsky & Papert (1969), but is not utilized by current ICL and prompting methods. Differently from machines, humans can often verbalize their mistakes and articulate explicit principles, or “lessons”;
these principles can further help humans avoid these and similar mistakes in the future. Inspired by this ability and the benefits it provides, we propose Learning Principles (LEAP): A prompting approach for learning principles from mistakes, and then conditioning on these principles when responding to other inputs. Instead of providing the model only with the given (correct) few-shot examples, LEAP begins by (a) inducing the LLM to make mistakes on each of these given few-shot inputs, in a zero-shot fashion, by sampling outputs with a non-zero temperature; (b) generating explicit principles by providing the same LLM with the
mistaken outputs along with the correct output; and finally (c) generating a response to the test question as in standard few-shot ICL, while providing the LLM with both the given input-output examples and the learned principles. LEAP is illustrated in Figure 2; examples for some of the learned principles are provided in Figure 1. Crucially, LEAP does not require any more input than the standard few-shot prompting settings. Further, the principle learning step is applied once for a task, and then the same learned principles are used for the entire test set. If the few-shot examples are given as Chain-of-Thought (CoT; Wei et al., 2022c), LEAP is applied seamlessly by generating mistaken chains-of-thought, contrasting them with the correct chains-of-thought, and generating principles. We demonstrate the effectiveness of LEAP across a wide range of reasoning benchmarks, including mathematical reasoning in GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021), multi-hop question answering tasks in HotpotQA (Yang et al., 2018b), textual reasoning in DROP (Dua et al., 2019a), and 27 Big-Bench Hard (Suzgun et al., 2022) tasks. LEAP outperforms the standard practice of few-shot prompting using strong models such as GPT-3.5-turbo, GPT-4, GPT-4-turbo and Claude-2.1, with and without Chain-of-Thought. For example, LEAP improves over the standard few-shot CoT using GPT-4 by 7.5% in DROP, by 3.3% in HotpotQA, and on 16 tasks in Big-Bench Hard. These results suggest that LEAP revolutionizes the “traditional” concept of few-shot ICL, by leveraging the recently emerged abilities of LLMs to follow instructions (Mishra et al., 2021; Wei et al., 2022a; Ouyang et al., 2022) and explain mistakes given the correct answer or feedback (Madaan et al., 2023; Chen et al., 2023).
# 2. Background: In-Context Learning
In-context learning, also known as few-shot prompting, uses a few (typically between 3 and 8) input-output task-specific examples for teaching a pre-trained LLM to solve a downstream task (Brown et al., 2020; Liu et al., 2021; 2023). These input-output pairs P = {(xi, yi)}k i=1 are concatenated to form a prompt p = ⟨x1 ·y1⟩⊕⟨x2 ·y2⟩⊕. . .⊕⟨xk · yk⟩, where “·” denotes the concatenation of each input with its corresponding output, and “⊕” denotes the sequential combination of these pairs. Then, every new test input xtest is appended to this prompt, resulting in p ⊕⟨xtest·, which is then provided to the LLM . The LLM completes this input, and generates the corresponding output ˆytest.
Typically, the answer y additionally contains an explanation (or thought) for deriving the answer (Wei et al., 2022c). In these settings, each solution y contains a thought t and the final answer a, forming: yi = ti · ai, and the LLM is expected to generate the test-thought before generating the
final test-answer: ttest · atest. In this work, we focus on this chain-of-thought setup, because of its widely demonstrated effectiveness for reasoning tasks (Wang et al., 2022a; Wei et al., 2022c; Zhou et al., 2022; Wang et al., 2022b).
# 3. LEAP: Learning Principles from Mistakes
In LEAP, our goal is to learn general principles that help the model avoid potential mistakes in a downstream task. Given the few-shot examples P = {⟨xi, yi⟩}k i=1 for a certain task, we start by generating mistakes to the few given examples.
Generating Mistakes For each input-output pair ⟨xi, yi⟩∈P, we generate a diverse set of solutions in a zero-shot fashion. Specifically, we follow Kojima et al. (2022), and create a zero-shot chain-of-thought prompt using xi and the phrase Think step-by-step. For each input xi, we sample n =15 outputs with a non-zero temperature, producing a varied set of potential solutions {ˆyj i }n j=1 for each example xi, such that ˆyj i = ˆtj i · ˆaj i, where ˆtj i represents the intermediate reasoning steps (thoughts), and ˆaj i denotes the final answer.
We identify incorrect solutions by comparing each ˆaj i with the ground-truth answer ai (which is given as part of the task), forming a set of mistakes for each ⟨xi, yi⟩∈P pair: Mi = �� xi, yi, ˆyj i ��n′ , such that ∀j : ˆaj i ̸= ai.
Generating Low-Level Principles Then, for each such mistake in M, we prompt the LLM to generate a natural language explanation of the mistake. In this step, the LLM is provided with the ground truth answer yi, to articulate the rationale behind the solution’s inaccuracy, as illustrated in Figure 3. The insights of how to avoid these mistakes are aggregated across all examples to form a set of low-level principles, denoted as LLOW-LEVEL.
# Generating High-Level Principles
# Generating High-Level Principles Subsequently, we u
the LLM to condense the low-level principles into approximately 5 key bullet points, thus creating high-level principles, denoted as LHIGH-LEVEL. The motivation for this step is generating generic, example-agnostic, principles for solving the downstream task, that do not depend on mistakes made for any specific example.
# Final Inference on Unseen Examples These principles,
either low or high-level, are then appended to the prompt p , forming enhanced prompts: pLOW-LEVEL = LLOW-LEVEL ⊕p for low-level feedback, and pHIGH-LEVEL = LHIGH-LEVEL ⊕p for high-level feedback, where p is the prompt constructed using the standard concatenation of the few-shot examples, as described in Section 2. Finally, we use the enhanced prompts p or
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9804/98048e8f-79a6-49cc-ba66-d6c431743784.png" style="width: 50%;"></div>
Figure 3: LEAP prompt to help LLM evaluate its own generated reasoning and answers, contrasting them with the correct reasoning and answers. The LLM is prompted to identify errors in its reasoning and extract key insights for improvement. This figure specifically represents the ‘GenerateExplanation‘ step in the LEAP algorithm (Algorithm 1).
pHIGH-LEVEL for answering all unseen examples in the test set. In effect, the one-time process of generating principles helps learn and articulate insights from the model’s previous mistakes, potentially improving its accuracy and reasoning capabilities in future responses for other inputs. The complete algorithm is summarized in Algorithm 1. Although different LLMs can be used for each step, in all our experiments, we fixed the LLM across all steps: generating mistakes, generating principles from those mistakes, and testing using these self-generated principles. This ensures that the difference in results arises only from LEAP rather than any kind of model ensemble or teaching.
The complete algorithm is summarized in Algorithm 1. Although different LLMs can be used for each step, in all our experiments, we fixed the LLM across all steps: generating mistakes, generating principles from those mistakes, and testing using these self-generated principles. This ensures that the difference in results arises only from LEAP rather than any kind of model ensemble or teaching.
# 4. Evaluation
We evaluated LEAP across various reasoning tasks, including HotpotQA (Yang et al., 2018b), DROP (Dua et al., 2019a), MATH (Hendrycks et al., 2021), GSM8K (Cobbe et al., 2021), and Big-Bench Hard (Suzgun et al., 2022). We follow the standard few-shot Chain-of-Thought (COT ); we apply LEAP on top of few-shot COT using the exact same number of labeled examples. Importantly, principles are generated once for every LLM and benchmark.
# 4.1. Experiment Setup
Compared Approaches Our baseline is the standard Fewshot prompting with COT . We compare this baseline with two variants of our proposed approach LEAP:
• LEAP HIGH-LEVEL is our proposed approach, as described in Section 3.
• LEAP LOW-LEVEL is similar to LEAP HIGH-LEVEL , except that we skip the step of “Generating High-Level Principles”, and test the models on the downstream task using the few-shot examples and the low-level principles, using LLOW-LEVEL rather than LHIGH-LEVEL (Section 3).
Benchmarks We used diverse reasoning benchmarks:
 Textual Reasoning: HotpotQA (Yang et al., 2018a) is a question-answering dataset of computational questions that require multi-hop reasoning. DROP (Dua et al., 2019b) is a reading comprehension dataset that requires numerical and logical reasoning over textual paragraphs; for evaluation, we randomly sampled 2000 questions from its dev set.
2021) comprises a test set of 1,319 diverse grade school math word problems, curated by human problem writers. In MATH (Hendrycks et al., 2021), there are 5,000 diverse examples consisting of problems from mathematics competitions. These are the two most common mathematical reasoning benchmarks.
• Big-Bench Hard (Suzgun et al., 2022): contains 27 challenging tasks that test various reasoning capabilities of LLMs . We repeated every run 3 times with a
Algorithm 1 LEAP Algorithm
Require: Few-shot examples P = {⟨xi, yi⟩}k
i=1, a pretrained LLM , number of outputs per input n, high-temperature setting T
1: for each input-output pair ⟨xi, yi⟩in P do
2:
Si ←ZeroShotCoT (LLM , xi, n, T)
▷Generate solutions using zero-shot chain-of-thought prompting
3:
Mi ←{
�
xi, yi, ˆyj
i
�
∈Si : ˆaj
i ̸= aj
i}
▷Identify incorrect solutions
4:
for each xi, yi, ˆyi in Mi do
5:
LLOW-LEVEL,i ←GenerateLowLevelPrinciples (LLM , xi, ˆyi, yi)
▷Generate principles for each mistake
6:
end for
7: end for
8: LLOW-LEVEL ←�k
i=1 LLOW-LEVEL,i
▷Aggregate low-level principles
9: LHIGH-LEVEL ←GenerateHighLevelPrinciples (LLM , LLOW-LEVEL)
▷Generate high-level principles
10: pLOW-LEVEL ←Concatenate (LLOW-LEVEL, P)
▷Create enhanced prompt with low-level principles
11: pHIGH-LEVEL ←Concatenate (LHIGH-LEVEL, P)
▷Create enhanced prompt with high-level principles
12: return pLOW-LEVEL, pHIGH-LEVEL
Table 1: Textual Reasoning results: Accuracy in textual reasoning benchmarks. The best approach for each base LLM each dataset is in bold; the second-best approach is underlined. We see a good performance boost by adopting LEAP w high-level feedback and low-level feedback. Almost all the models can benefit from the principles learned and fix th previous mistakes.
GPT-3.5-turbo
GPT-4
GPT-4-turbo
Gemini Pro
HotpotQA
Few-shot CoT
29.10
36.35
38.10
28.25
LEAP LOW-LEVEL
32.60
39.30
37.85
23.70
LEAP HIGH-LEVEL
30.35
39.65
38.75
25.50
DROP
Few-shot CoT
63.20
72.05
83.40
64.60
LEAP LOW-LEVEL
63.35
79.55
83.60
67.15
LEAP HIGH-LEVEL
63.00
78.60
80.00
67.60
<div style="text-align: center;">temperature of zero and report the average.1</div>
Models We evaluated LEAP across a wide range of base models, including GPT-3.5-turbo (version -0613), GPT-4 (version -0613), GPT-4-turbo (version -1106), Claude-2.1, and Gemini Pro (Gemini Team et al., 2023).
Few-shot examples In Big-Bench Hard, we used the COT prompts from Suzgun et al. (2022) with three given examples (3-shot) in each benchmark. In MATH and GSM8K , we used the standard training examples from each dataset, with 3 examples for each. In DROP, we used 3 given examples that we adopted from Least-to-Most (Zhou et al., 2022). In HotpotQA we used 6 examples from ReAct (Yao et al., 2022) (“closed-book”). Importantly, in each benchmark, the exact same few-shot examples were used across all evaluated approaches, including the baseline and LEAP. Our complete prompts are provided in Appendix G.
1Internal non-determinism causes different outputs even with a temperature of zero, but the variance was negligible.
# 4.2. Textual Reasoning Results
Table 1 shows the results on DROP and HotpotQA. As shown, LEAP improves over the Few-shot COT baseline by up to 3.5% on Hotpot QA and 7.5% on DROP. In HotpotQA, GPT-3.5-Turbo and GPT-4 are consistently improved when using LEAP. In DROP, GPT-4 is significantly improved by LEAP (an absolute gain of 7.5%), Gemini Pro is improved by 3%, while the improvement for GPT-3.5-Turbo and GPT4-turbo are more minor.
In most tasks and base models, both LEAP LOW-LEVEL and LEAP HIGH-LEVEL improve over the Few-shot COT baseline. The only case where the Few-shot COT baseline performs better than both LEAP LOW-LEVEL and LEAP HIGH-LEVEL is in HotpotQA using Gemini Pro. Observing the low-level principles that Gemini Pro learned in HotpotQA (Table 41), we believe that the learned principles are correct and useful, but they are overly focused on the examples they were generated for, more verbose, and similar to each other. These hinder the principles (and the high-level principles generated from them) from generalizing to other examples. For zero-shot prompting using principles learned from few examples, additional results are shown in Appendix A.
Table 2: Math Reasoning Results: Accuracy in MATH and GSM8K . The best approach for each LLM and base task is in bold; the second-best approach is underlined. LEAP HIGH-LEVEL and LEAP LOW-LEVEL in both GSM8K and MATH datasets consistently improve the performance over the CoT baseline. We also observe that the learned mathmetical principles can be generalized to different test questions.
GPT-3.5-turbo
GPT-4
Claude-2
Gemini Pro
GSM8K
Few-shot CoT
76.4
93.6
84.3
77.8
LEAP LOW-LEVEL
77.4
94.2
82.7
77.3
LEAP HIGH-LEVEL
76.6
93.8
83.8
78.7
MATH
Few-shot CoT
55.6
63.5
43.2
31.1
LEAP LOW-LEVEL
56.1
64.5
42.7
29.7
LEAP HIGH-LEVEL
56.5
64.0
43.4
30.3
# 4.3. Math Reasoning Results
Table 2 shows the results on MATH and GSM8K . As shown, in GPT-3.5-turbo and GPT-4, both LEAP LOW-LEVEL and LEAP HIGH-LEVEL outperform the Few-shot COT baseline. Claude-2 shows inconclusive results: in GSM8K , Fewshot COT performs better than LEAP; in MATH, LEAP HIGH-LEVEL achieves slightly higher accuracy than Few-shot COT . Gemini Pro also shows inconclusive results, where both LEAP LOW-LEVEL and LEAP HIGH-LEVEL outperform the baseline in GSM8K , but perform slightly worse in MATH. Additional results, for zero-shot prompting using principles learned from few examples, are shown in Appendix B.
Does leap work with open-source models? In our preliminary experiments with open-source models, LEAP did not improve over the few-shot COT baseline. While the open-source models did produce useful principles, the opensource models did not follow these principles at test time.
# Does leap work with open-source models?
Table 3 shows some of these experiments with Llama-2chat-70B : we used Llama-2-chat-70B as the base model, but generated the principles using either Llama-2-chat-70B (“LLama-2 Critic”) or with GPT-4 (“GPT-4 Critic”). As shown, even when the principles are generated by GPT-4 (which we assume to be useful), Llama-2-chat-70B does not manage to leverage them to generate better final responses with either LEAP LOW-LEVEL or LEAP HIGH-LEVEL .
In general, we believe that LEAP requires a base LLM with strong enough instruction following and reflection capabilities; we believe that, unfortunately, most open-source models are not as powerful as proprietary models yet.
# 4.4. Big-Bench Hard Results
Figure 4 shows the results on selected BBH tasks; results for the rest of the BBH tasks are shown in Table 7 in Appendix C. We selected tasks to Table 6 where the difference in results was the most meaningful: there were some tasks where GPT4-0613 achieved 100% accuracy with any of the approaches,
# including both LEAP and the baseline.
As shown in Table 6, in 37 out of 42 combinations of task and LLM , one of LEAP LOW-LEVEL or LEAP HIGH-LEVEL outperforms the baseline Few-shot COT . In 24 of the cases, both LEAP LOW-LEVEL and LEAP HIGH-LEVEL outperform the Few-shot COT baseline. In two cases, the Few-shot COT baseline performs equally to LEAP LOW-LEVEL ; and in 3 cases the Few-shot baseline performs better than both LEAP LOW-LEVEL and LEAP HIGH-LEVEL . We could not identify any particular pattern as to which method should be used: LEAP LOW-LEVEL or LEAP HIGH-LEVEL ; it seems that this depends on the reasoning complexity of the task, the diversity across questions in the benchmark, and the quality of the principles that the LLM had managed to generate from the 3 given examples for that task. We thus suggest that in real-life scenarios, both approaches should be tested, and selected using a validation set. Additional results, for zero-shot prompting using principles learned from a few examples, are shown in Appendix C.
As shown in Table 6, in 37 out of 42 combinations of task and LLM , one of LEAP LOW-LEVEL or LEAP HIGH-LEVEL outperforms the baseline Few-shot COT . In 24 of the cases, both LEAP LOW-LEVEL and LEAP HIGH-LEVEL outperform the Few-shot COT baseline. In two cases, the Few-shot COT baseline performs equally to LEAP LOW-LEVEL ; and in 3 cases the Few-shot baseline performs better than both LEAP LOW-LEVEL and LEAP HIGH-LEVEL .
We could not identify any particular pattern as to which method should be used: LEAP LOW-LEVEL or LEAP HIGH-LEVEL ; it seems that this depends on the reasoning complexity of the task, the diversity across questions in the benchmark, and the quality of the principles that the LLM had managed to generate from the 3 given examples for that task. We thus suggest that in real-life scenarios, both approaches should be tested, and selected using a validation set. Additional results, for zero-shot prompting using principles learned from a few examples, are shown in Appendix C.
# 5. Qualitative Analysis
Figure 5 shows examples of questions from BBH , along with the task’s learned principles, the baseline few-shot CoT response, and the LEAP LOW-LEVEL response, all generated by GPT-3.5-turbo. The learned principle in each case is highlighted, along with the mistake in the COT output and the correct reasoning generated by LEAP. Additional examples are shown in Appendix D. The actual learned principles for some of the tasks and LLMs are provided in Tables 9-43 in Appendix E.
Figure 5 shows examples of questions from BBH , along with the task’s learned principles, the baseline few-shot CoT response, and the LEAP LOW-LEVEL response, all generated by GPT-3.5-turbo. The learned principle in each case is highlighted, along with the mistake in the COT output and the correct reasoning generated by LEAP.
Additional examples are shown in Appendix D. The actual learned principles for some of the tasks and LLMs are provided in Tables 9-43 in Appendix E.
# 6. Related Works
This section focuses on the few works most related to ours. For a complete survey of relevant papers, see Zhao et al. (2023); Qiao et al. (2023); Kaddour et al. (2023); Xi et al.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/232c/232ce888-c273-432c-9744-45646da349f1.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a7f3/a7f35218-24ee-4518-80b4-181ef146b899.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8da6/8da6a3cd-1803-49bf-a405-4da599e3697a.png" style="width: 50%;"></div>
Figure 4: Accuracy in BBH tasks, across GPT-3.5-TURBO-0613 , GPT-4-0613 , and GEMINI-PRO . The figure presents the results using a scatter plot, where the y-axis represents scores achieved with LEAP, and the x-axis represents the baseline scores from COT . Each task is represented by a point on the plot, with different shapes assigned to different models for easy distinction. Tasks above the y = x line are those where LEAP leads to an improvement in performance. Table 6 shows the detailed results for all 27 Big-Bench hard tasks. We find that in 37 out of 42 combinations of task and LLM , one of LEAP LOW-LEVEL or LEAP HIGH-LEVEL outperforms the baseline Few-shot COT .
Figure 4: Accuracy in BBH tasks, across GPT-3.5-TURBO-0613 , GPT-4-0613 , and GEMINI-PRO . The figure presents the results using a scatter plot, where the y-axis represents scores achieved with LEAP, and the x-axis represents the baseline scores from COT . Each task is represented by a point on the plot, with different shapes assigned to different models for easy distinction. Tasks above the y = x line are those where LEAP leads to an improvement in performance. Table 6 shows the detailed results for all 27 Big-Bench hard tasks. We find that in 37 out of 42 combinations of task and LLM , one of LEAP LOW-LEVEL or LEAP HIGH-LEVEL outperforms the baseline Few-shot COT . Table 3: Llama-2-chat-70B Results: Accuracy in MATH and GSM8K using Llama-2-chat-70B as the base LLM, while generating the principles either with Llama-2-chat-70B or GPT-4. Even when the principles are generated by GPT-4 (GPT-4 Critic), Llama-2-chat-70B does not leverage the learned principles, and does not improve over the baseline Few-shot CoT.
Llama-2-chat-70B as the base model, with:
Llama-2 Critic
GPT-4 Critic
GSM8K
Few-shot CoT
52.5
52.5
LEAP LOW-LEVEL
50.6
49.8
LEAP HIGH-LEVEL
47.0
51.0
MATH
Few-shot CoT
16.2
16.2
LEAP LOW-LEVEL
13.8
12.9
LEAP HIGH-LEVEL
14.2
13.8
Comparison to Madaan et al. (2023) A related work to ours is Self-Refine (Madaan et al., 2023), which, similarly to LEAP, uses the LLM to reflect on its own previously generated outputs. However, a significant difference from Self-Refine is that applying Self-Refine to a test example requires feedback for that specific test example; contrarily, in this work, we show that principles that were learned from some examples generalize to other test examples which we do not have feedback for. Further, Self-Refine uses “intrinsic self-correction”, without any external feedback; as found by recent work (Huang et al., 2023a; Valmeekam et al., 2023), such intrinsic self-correction does not benefit reasoning tasks. In contrast, in this work, we show that the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d706/d7062aa0-9f73-4e56-8790-2d6930824199.png" style="width: 50%;"></div>
external signal provided by as few as 3 examples is sufficient for effective self-reflection in reasoning if these are used to articulate explicit learned principles. Another related work is MemPrompt (Madaan et al., 2022), which stores feedback on mistakes in a memory, and retrieves relevant feedback for a new query; in contrast, LEAP is a one-time process that does not require continuous storage and retrieval.
Comparison to Yang et al. (2023) Optimization by PROmpting (OPRO) (Yang et al., 2023) uses LLMs to improve zero-shot prompts, and both OPRO and LEAP can be thought of as ways of improving a prompt for a specific task. However, applying OPRO for prompt optimization requires a training set, which can be as large as ∼260 examples for GSM8K; in contrast, LEAP requires only 3 labeled
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5340/53407db9-0a2b-4618-9a0d-65e7f1ef9370.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f0da/f0dabbc7-abfa-43a7-8d5b-5414fecbea2a.png" style="width: 50%;"></div>
# Boolean Expressions (BBH )
examples. Further, OPRO calls the LLM for each of these ∼260 examples in each step; while the number of steps can be as low as 6, it can also be as high as 107. This results in between ∼1500 calls and up to 107×260 ≈27,000 inference calls to the LLM. In contrast, LEAP uses only about 34 inference calls: 3 calls to generate mistakes; one call for each mistake to generate low-level principles, which results in about 30 calls; and a final call to generate high-level principles. Moreover, like OPRO, LEAP can also improve zero-shot prompting by learning principles from a few examples and applying them in a zero-shot fashion, as we show in Appendix A, Appendix B, and Appendix C. For example, Zero-shot-LEAP improves the strong GPT-4 model by 1% on GSM8K (Table 5). Similarly, EvoPrompt (Guo et al., 2023) requires around 10,000 LLM inference calls, which is also several orders of magnitude more than LEAP. Other related approaches are AutoPrompt (Shin et al., 2020), STaR (Zelikman et al., 2022), LMSI (Huang et al., 2023b), and Self-Align (Sun et al., 2023) but these are based on training the LLM , which is often either inaccessible or computationally infeasible.
Comparison to Chia et al. (2023) Contrastive Chain-ofThought (Chia et al., 2023) shares a similar motivation with our work: learning from negative in-context examples instead of learning only from positive examples. However, when we reproduced their results, we found that simple
# Comparison to Chia et al. (2023)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a267/a2679fb6-c627-460d-9dfa-7bbd5ab16450.png" style="width: 50%;"></div>
# Object Counting (BBH )
improvements to the post-processing code of extracting the final answer out of the LLM’s raw output led to differences of more than 10 absolute points from their reported results in GSM8K , which made the proposed Contrastive COT approach perform similarly or worse than the baseline few-shot COT . We thus believe that generating explicit principles, as in LEAP, is the key ingredient in learning from negative in-context examples.
# 7. Conclusion
In this paper, we introduce Learning Principles (LEAP), a novel approach that allows LLMs to learn more out of given few-shot examples, by intentionally making mistakes on these examples; reflecting on the mistakes; and finally articulating explicit task-specific principles, which helps avoid similar mistakes in the future. LEAP requires exactly the same number of labeled examples as fewshot prompting, and allows improving a variety of strong LLMs (GPT-3.5-turbo, GPT-4, GPT-4-turbo and Gemini Pro) across a broad range of reasoning tasks (DROP, HotpotQA, GSM8K , MATH, and Big-Bench Hard). We believe that LEAP unlocks new possibilities from learning in the traditional concept of few-shot in-context learning, by learning from mistakes, rather than learning from positive examples only.
# References
Gemini Team, A., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. Guo, Q., Wang, R., Guo, J., Li, B., Song, K., Tan, X., Liu, G., Bian, J., and Yang, Y. Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. arXiv preprint arXiv:2309.08532, 2023. Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., and Zhou, D. Large language models cannot self-correct reasoning yet. arXiv preprint arXiv:2310.01798, 2023a. Huang, J., Gu, S., Hou, L., Wu, Y., Wang, X., Yu, H., and Han, J. Large language models can self-improve. In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 1051–1068, Singapore, December 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.67. URL https: //aclanthology.org/2023.emnlp-main.67. Kaddour, J., Harris, J., Mozes, M., Bradley, H., Raileanu, R., and McHardy, R. Challenges and applications of large language models. arXiv preprint arXiv:2307.10169, 2023. Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large Language Models are Zero-Shot Reasoners. arXiv preprint arXiv:2205.11916, 2022. Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago, A., et al. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022. Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., and Chen, W. What makes good in-context examples for gpt-3? In Workshop on Knowledge Extraction and Integration for Deep Learning Architectures; Deep Learning Inside Out, 2021. URL https://api.semanticscholar. org/CorpusID:231632658. Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., and Neubig, G. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9):1–35, 2023. Lu, S., Bigoulaeva, I., Sachdeva, R., Madabushi, H. T., and Gurevych, I. Are emergent abilities in large language models just in-context learning? arXiv preprint arXiv:2309.01809, 2023.
Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022. Anil, R., Dai, A. M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z., et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020. Chen, X., Lin, M., Schärli, N., and Zhou, D. Teaching large language models to self-debug. arXiv preprint arXiv:2304.05128, 2023. Chia, Y. K., Chen, G., Tuan, L. A., Poria, S., and Bing, L. Contrastive chain-of-thought prompting. arXiv preprint arXiv:2311.09277, 2023. Chialvo, D. R. and Bak, P. Learning from mistakes. Neuroscience, 90(4):1137–1148, 1999. Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H., Sutton, C., Gehrmann, S., et al. Palm: Scaling language modeling with pathways (no. arxiv: 2204.02311). arxiv, 2022. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. Dua, D., Wang, Y., Dasigi, P., Stanovsky, G., Singh, S., and Gardner, M. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proc. of NAACL, 2019a. Dua, D., Wang, Y., Dasigi, P., Stanovsky, G., Singh, S., and Gardner, M. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In North American Chapter of the Association for Computational Linguistics, 2019b. URL https://api. semanticscholar.org/CorpusID:67855846. Edmondson, A. Psychological safety and learning behavior in work teams. Administrative science quarterly, 44(2): 350–383, 1999. Edmondson, A. C. Learning from mistakes is easier said than done: Group and organizational influences on the detection and correction of human error. The Journal of Applied Behavioral Science, 32(1):5–28, 1996.
Madaan, A., Tandon, N., Clark, P., and Yang, Y. Memoryassisted prompt editing to improve gpt-3 after deployment. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 2833–2861, 2022.
Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023.
Min, S., Lewis, M., Zettlemoyer, L., and Hajishirzi, H. Metaicl: Learning to learn in context. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2791–2809, 2022a.
Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., and Zettlemoyer, L. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 11048– 11064, 2022b.
# Minsky, M. and Papert, S. An introduction to computational geometry. Cambridge tiass., HIT, 479(480):104, 1969.
Mishra, S., Khashabi, D., Baral, C., and Hajishirzi, H. Crosstask generalization via natural language crowdsourcing instructions. arXiv preprint arXiv:2104.08773, 2021.
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.
Qiao, S., Ou, Y., Zhang, N., Chen, X., Yao, Y., Deng, S., Tan, C., Huang, F., and Chen, H. Reasoning with language model prompting: A survey. In Rogers, A., BoydGraber, J., and Okazaki, N. (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5368– 5393, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long. 294. URL https://aclanthology.org/2023. acl-long.294.
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
Rosenblatt, F. The perceptron, a perceiving and recognizing automaton Project Para. Cornell Aeronautical Laboratory, 1957.
Shin, T., Razeghi, Y., Logan IV, R. L., Wallace, E., and Singh, S. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 4222– 4235, 2020.
Shin, T., Razeghi, Y., Logan IV, R. L., Wallace, E., and Singh, S. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 4222– 4235, 2020. Sun, Z., Shen, Y., Zhou, Q., Zhang, H., Chen, Z., Cox, D., Yang, Y., and Gan, C. Principle-driven self-alignment of language models from scratch with minimal human supervision. arXiv preprint arXiv:2305.03047, 2023. Suzgun, M., Scales, N., Schärli, N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H., Zhou, D., et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023b. Valmeekam, K., Marquez, M., and Kambhampati, S. Investigating the effectiveness of self-critiquing in LLMs solving planning tasks. In NeurIPS 2023 Foundation Models for Decision Making Workshop, 2023. URL https: //openreview.net/forum?id=gGQfkyb0KL. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., and Zhou, D. Rationale-Augmented Ensembles in Language Models. arXiv preprints arXiv:2207.00747, 2022a. Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022b. Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. Finetuned language models are zero-shot learners. In ICLR, 2022a. Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., Chi, E. H., Hashimoto, T., Vinyals, O., Liang, P., Dean, J., and Fedus, W. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022b. ISSN 2835-8856. URL https:// openreview.net/forum?id=yzkSU5zdwD. Survey Certification.
Singh, S. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 4222– 4235, 2020. Sun, Z., Shen, Y., Zhou, Q., Zhang, H., Chen, Z., Cox, D., Yang, Y., and Gan, C. Principle-driven self-alignment of language models from scratch with minimal human supervision. arXiv preprint arXiv:2305.03047, 2023. Suzgun, M., Scales, N., Schärli, N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H., Zhou, D., et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023b. Valmeekam, K., Marquez, M., and Kambhampati, S. Investigating the effectiveness of self-critiquing in LLMs solving planning tasks. In NeurIPS 2023 Foundation Models for Decision Making Workshop, 2023. URL https: //openreview.net/forum?id=gGQfkyb0KL. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., and Zhou, D. Rationale-Augmented Ensembles in Language Models. arXiv preprints arXiv:2207.00747, 2022a. Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022b. Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. Finetuned language models are zero-shot learners. In ICLR, 2022a. Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., Chi, E. H., Hashimoto, T., Vinyals, O., Liang, P., Dean, J., and Fedus, W. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022b. ISSN 2835-8856. URL https:// openreview.net/forum?id=yzkSU5zdwD. Survey Certification.
Sun, Z., Shen, Y., Zhou, Q., Zhang, H., Chen, Z., Cox, D., Yang, Y., and Gan, C. Principle-driven self-alignment of language models from scratch with minimal human supervision. arXiv preprint arXiv:2305.03047, 2023.
Suzgun, M., Scales, N., Schärli, N., Gehrmann, S., Tay, Y., Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H., Zhou, D., et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.
Valmeekam, K., Marquez, M., and Kambhampati, S. Investigating the effectiveness of self-critiquing in LLMs solving planning tasks. In NeurIPS 2023 Foundation Models for Decision Making Workshop, 2023. URL https: //openreview.net/forum?id=gGQfkyb0KL.
Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., and Zhou, D. Rationale-Augmented Ensembles in Language Models. arXiv preprints arXiv:2207.00747, 2022a.
Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022b.
Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. Finetuned language models are zero-shot learners. In ICLR, 2022a.
Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., Chi, E. H., Hashimoto, T., Vinyals, O., Liang, P., Dean, J., and Fedus, W. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022b. ISSN 2835-8856. URL https:// openreview.net/forum?id=yzkSU5zdwD. Survey Certification.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35: 24824–24837, 2022c. Wei, J., Wei, J., Tay, Y., Tran, D., Webson, A., Lu, Y., Chen, X., Liu, H., Huang, D., Zhou, D., et al. Larger language models do in-context learning differently. arXiv preprint arXiv:2303.03846, 2023. Wiener, N. Cybernetics; or control and communication in the animal and the machine. 1948. Xi, Z., Chen, W., Guo, X., He, W., Ding, Y., Hong, B., Zhang, M., Wang, J., Jin, S., Zhou, E., et al. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864, 2023. Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., and Chen, X. Large language models as optimizers. arXiv preprint arXiv:2309.03409, 2023. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., and Manning, C. D. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing, 2018a. URL https://api. semanticscholar.org/CorpusID:52822214. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., and Manning, C. D. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018b. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2022. Zelikman, E., Wu, Y., Mu, J., and Goodman, N. STar: Bootstrapping reasoning with reasoning. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum? id=_3ELRdg2sgI. Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022. Zhang, Z., Yao, Y., Zhang, A., Tang, X., Ma, X., He, Z., Wang, Y., Gerstein, M., Wang, R., Liu, G., et al. Igniting language intelligence: The hitchhiker’s guide from chainof-thought reasoning to language agents. arXiv preprint arXiv:2311.11797, 2023.
<div style="text-align: center;">Wiener, N. Cybernetics; or control and communication in the animal and the machine. 1948.</div>
Zhang, Z., Yao, Y., Zhang, A., Tang, X., Ma, X., He, Z., Wang, Y., Gerstein, M., Wang, R., Liu, G., et al. Igniting language intelligence: The hitchhiker’s guide from chainof-thought reasoning to language agents. arXiv preprint arXiv:2311.11797, 2023.
Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.
Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023. Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Bousquet, O., Le, Q., and Chi, E. Leastto-Most Prompting Enables Complex Reasoning in Large Language Models. arXiv preprint arXiv:2205.10625, 2022.
# A. Additional Results on Text Reasoning
Table 4 shows additional Text Reasoning results. The upper part of Table 4 is identical to Table 1; the lower part of Table 4 contains zero-shot results, with principles that were learned from the given few-shot examples; this setup is similar to related work such as (Yang et al., 2023).
Table 4: Text Reasoning Results: Accuracy in textual reasoning datasets. The best approach for ea dataset is in bold; the second-best approach is underlined.
<div style="text-align: center;">Table 4: Text Reasoning Results: Accuracy in textual reasoning datasets. The best approach for each base LLM in eac dataset is in bold; the second-best approach is underlined.</div>
GPT-3.5-turbo
GPT-4
Gemini Pro
Few-shot results, identical to Table 2:
HotpotQA
Few-shot CoT
29.10
36.35
28.25
LEAP LOW-LEVEL
32.60
39.30
23.70
LEAP HIGH-LEVEL
30.35
39.65
25.50
DROP
Few-shot CoT
63.20
72.05
64.60
LEAP LOW-LEVEL
63.35
79.55
67.15
LEAP HIGH-LEVEL
63.00
78.60
67.60
GPT-3.5-turbo
GPT-4
Gemini Pro
Zero-shot results, using principles learned from few-shot examples:
HotpotQA
Zero-shot CoT
12.55
27.80
19.75
LEAP LOW-LEVEL
6.65
31.20
3.30
LEAP HIGH-LEVEL
13.10
30.90
2.45
DROP
Zero-shot CoT
59.25
82.20
62.75
LEAP LOW-LEVEL
55.20
84.10
65.30
LEAP HIGH-LEVEL
55.25
81.80
62.80
B. Additional Results on Mathematical Reasoning
Table 5: Math Reasoning Results: Accuracy in MATH and GSM8K . The best approach for each LLM and base task is i bold; the second-best approach is underlined.
GPT-3.5-turbo
GPT-4
Claude-2
Gemini Pro
Few-shot results, identical to Table 2:
GSM8K
Few-shot CoT
76.4
93.6
84.3
LEAP LOW-LEVEL
77.4
94.1
82.7
LEAP HIGH-LEVEL
76.6
93.8
83.8
MATH
Few-shot CoT
55.6
63.5
43.2
LEAP LOW-LEVEL
56.1
64.5
42.7
LEAP HIGH-LEVEL
56.5
64.0
43.4
GPT-3.5-turbo
GPT-4
Claude-2
Gemini Pro
Zero-shot results, using principles learned from few-shot examples:
GSM8K
Zero-shot CoT
76.9
93.2
75.4
LEAP LOW-LEVEL
74.4
94.2
76.7
LEAP HIGH-LEVEL
73.8
94.1
76.9
MATH
Zero-shot CoT
54.2
63.5
40.2
LEAP LOW-LEVEL
52.0
63.2
40.5
LEAP HIGH-LEVEL
50.0
61.5
41.8
Table 5 shows additional Mathematical Reasoning results. The upper part of Table 5 is identical to Table 2; the lower part of Table 5 contains zero-shot results, with principles that were learned from the given few-shot examples; this setup is similar to related work such as (Yang et al., 2023).
# C. Additional Results on BBH
Table 7 shows results on additional BBH tasks that we could not fit into Table 6. We selected tasks to Table 6 in the main paper where the difference was more meaningful. As shown in Table 7, there were some tasks such as temporal_sequences, web_of_lies and tracking_shuffled_objects_five_objects where GPT-4 achieved 100% accuracy with any of the approaches, including both LEAP and the few-shot baseline.
<div style="text-align: center;">In-Context Principle Learning from Mistakes</div>
task
wrong to
correct
correct to
wrong
wrong to
wrong
correct to
correct
temporal_sequences
31.0
10.0
66.0
143.0
snarks
18.0
11.0
39.0
110.0
disambiguation_qa
10.0
8.0
76.0
156.0
logical_deduction_seven_objects
27.0
21.0
129.0
73.0
object_counting
4.0
1.0
4.0
241.0
movie_recommendation
14.0
11.0
33.0
192.0
navigate
2.0
1.0
12.0
235.0
formal_fallacies
14.0
9.0
99.0
128.0
sports_understanding
5.0
4.0
10.0
231.0
boolean_expressions
5.0
2.0
10.0
233.0
web_of_lies
2.0
1.0
1.0
246.0
multistep_arithmetic_two
12.0
8.0
33.0
197.0
causal_judgement
18.0
18.0
51.0
100.0
salient_translation_error_detection
9.0
16.0
102.0
123.0
tracking_shuffled_objects_three_objects
12.0
15.0
30.0
193.0
word_sorting
11.0
13.0
109.0
117.0
logical_deduction_five_objects
24.0
27.0
78.0
121.0
tracking_shuffled_objects_five_objects
12.0
16.0
56.0
166.0
hyperbaton
4.0
19.0
40.0
187.0
logical_deduction_three_objects
9.0
16.0
20.0
205.0
tracking_shuffled_objects_seven_objects
10.0
14.0
79.0
147.0
dyck_languages
0.0
0.0
0.0
250.0
date_understanding
3.0
9.0
31.0
207.0
penguins_in_a_table
13.0
13.0
20.0
100.0
reasoning_about_colored_objects
11.0
19.0
33.0
187.0
ruin_names
14.0
38.0
87.0
111.0
geometric_shapes
9.0
25.0
74.0
142.0
Table 8: Efficacy of LEAP Methods Across Various Tasks. This table provides an overview of the effectiveness of LEAP in modifying the correctness of responses across a range of tasks. It highlights the number of instances where answers changed from wrong to correct, correct to wrong, and the stability of responses (both correct and incorrect). Notable gains in tasks like ’temporal_sequences’ and ’snarks’ suggest significant improvements, whereas tasks like ’ruin_names’ and ’geometric_shapes’ show areas needing further methodological refinement.
able 6: BBH Results: Accuracy in BBH tasks, across GPT-3.5-turbo, GPT-4, and Gemini Pro. The best approach for each ase LLM in each tasks is in bold; the second-best approach is underlined. Each number represents the average across 3 entical runs with a temperature of zero.
Table 6: BBH Results: Accuracy in BBH tasks, across GPT-3.5-turbo, GPT-4, and Gemini Pro. The best approach fo base LLM in each tasks is in bold; the second-best approach is underlined. Each number represents the average ac identical runs with a temperature of zero.
Task
Approach
GPT-3.5-turbo
GPT-4
Gemini Pro
boolean_expressions
Few-shot CoT
94.13
96.80
91.20
LEAP LOW-LEVEL
94.93
98.00
93.60
LEAP HIGH-LEVEL
95.07
97.60
92.40
disambiguation_qa
Few-shot CoT
65.73
85.60
66.00
LEAP LOW-LEVEL
67.20
86.00
66.00
LEAP HIGH-LEVEL
68.00
85.20
70.40
formal_fallacies
Few-shot CoT
56.00
81.60
58.80
LEAP LOW-LEVEL
56.93
84.00
61.20
LEAP HIGH-LEVEL
56.80
81.60
57.60
hyperbaton
Few-shot CoT
83.07
98.00
88.40
LEAP LOW-LEVEL
81.33
99.20
91.20
LEAP HIGH-LEVEL
84.80
98.80
87.60
logical_deduction_five_objects
Few-shot CoT
60.67
85.20
60.40
LEAP LOW-LEVEL
59.47
85.60
62.80
LEAP HIGH-LEVEL
57.73
84.40
60.80
logical_deduction_seven_objects
Few-shot CoT
38.67
56.80
49.20
LEAP LOW-LEVEL
40.00
58.40
50.00
LEAP HIGH-LEVEL
39.20
59.60
48.40
movie_recommendation
Few-shot CoT
81.47
90.40
83.20
LEAP LOW-LEVEL
82.53
91.60
84.80
LEAP HIGH-LEVEL
84.00
92.00
83.60
multistep_arithmetic_two
Few-shot CoT
81.73
93.20
79.60
LEAP LOW-LEVEL
82.00
94.00
78.00
LEAP HIGH-LEVEL
82.00
92.40
81.20
navigate
Few-shot CoT
94.27
98.00
95.20
LEAP LOW-LEVEL
95.20
98.40
95.20
LEAP HIGH-LEVEL
94.93
98.40
94.80
object_counting
Few-shot CoT
96.80
98.80
94.80
LEAP LOW-LEVEL
97.87
99.20
93.20
LEAP HIGH-LEVEL
97.60
99.20
91.60
ruin_names
Few-shot CoT
60.13
87.60
72.80
LEAP LOW-LEVEL
53.47
89.60
72.80
LEAP HIGH-LEVEL
68.13
87.60
74.40
snarks
Few-shot CoT
67.42
89.89
82.58
LEAP LOW-LEVEL
70.60
89.33
79.21
LEAP HIGH-LEVEL
66.48
91.01
80.90
sports_understanding
Few-shot CoT
93.87
94.40
98.40
LEAP LOW-LEVEL
94.80
95.20
98.40
LEAP HIGH-LEVEL
88.00
94.40
98.00
word_sorting
Few-shot CoT
53.07
94.40
82.00
LEAP LOW-LEVEL
52.00
94.00
83.20
LEAP HIGH-LEVEL
53.60
95.60
84.80
<div style="text-align: center;">Table 7: Additional results on Big-Bench Hard tasks. The best approach for each base LLM in each tasks is in bold; the second-best approach is underlined. Each number represents the average across 3 identical runs with a temperature of zero.</div>
Task
Approach
GPT-3.5-turbo
GPT-4
Gemini-Pro
geometric_shapes
Few-shot
67.33
54.40
50.00
+ LEAP LOW-LEVEL
59.60
57.20
50.00
+ LEAP HIGH-LEVEL
60.13
58.80
50.40
causal_judgement
Few-shot
63.64
73.26
62.57
+ LEAP LOW-LEVEL
63.64
72.73
62.57
+ LEAP HIGH-LEVEL
63.10
72.73
62.57
date_understanding
Few-shot
86.67
90.00
87.60
+ LEAP LOW-LEVEL
83.73
91.20
87.60
+ LEAP HIGH-LEVEL
85.60
90.00
86.40
dyck_languages
Few-shot
35.73
56.80
0
+ LEAP LOW-LEVEL
33.07
56.80
0
+ LEAP HIGH-LEVEL
35.73
54.00
0
logical_deduction
_three_objects
Few-shot
88.80
99.20
89.20
+ LEAP LOW-LEVEL
86.80
98.80
90.00
+ LEAP HIGH-LEVEL
85.73
98.80
90.80
penguins_in_a_table
Few-shot
76.94
97.26
81.51
+ LEAP LOW-LEVEL
73.74
96.58
78.77
+ LEAP HIGH-LEVEL
73.97
96.58
76.71
reasoning_about
_colored_objects
Few-shot
82.13
95.20
83.20
+ LEAP LOW-LEVEL
77.87
91.20
84.00
+ LEAP HIGH-LEVEL
75.87
94.00
82.40
salient_translation
_error_detection
Few-shot
55.73
68.80
56.00
+ LEAP LOW-LEVEL
55.20
67.20
52.80
+ LEAP HIGH-LEVEL
54.27
67.20
55.20
temporal_sequences
Few-shot
60.93
100.0
99.60
+ LEAP LOW-LEVEL
65.20
100.0
98.00
+ LEAP HIGH-LEVEL
62.80
100.0
99.20
tracking_shuffled_objects
_five_objects
Few-shot
72.40
100.0
64.00
+ LEAP LOW-LEVEL
70.93
100.0
64.40
+ LEAP HIGH-LEVEL
71.33
100.0
64.00
tracking_shuffled_objects
_seven_objects
Few-shot
65.07
100.0
56.8
+ LEAP LOW-LEVEL
62.53
100.0
60.0
+ LEAP HIGH-LEVEL
61.73
98.8
60.8
tracking_shuffled_objects
_three_objects
Few-shot
83.60
100.0
69.60
+ LEAP LOW-LEVEL
82.80
100.0
69.20
+ LEAP HIGH-LEVEL
85.87
100.0
68.40
web_of_lies
Few-shot
98.67
100.0
100.0
+ LEAP LOW-LEVEL
99.20
100.0
100.0
+ LEAP HIGH-LEVEL
99.20
100.0
100.0
# D. Additional Qualitative Analysis
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/dd44/dd44b00e-8041-473b-afe1-cd32c751f213.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/012d/012da157-ba32-43c3-bf3c-1a4e4d53a415.png" style="width: 50%;"></div>
Question: The following translations from German to English contain a particular error. That error will be one of the following types: Named Entities, Numerical Values, Modifiers or Adjectives, Negation or Antonyms, Facts, Dropped Content. Please identify that error. Source: Neu Fahrland ist ein Ortsteil von Potsdam. Translation: Neu Fahrland is a district of Pots. The translation contains an error pertaining to Options: (A) Modifiers or Adjectives (B) Numerical Values (C) Negation or Antonyms (D) Named Entities (E) Dropped Content (F) Facts Principles: When analyzing translations, it is important to carefully consider the translation of named entities. Any changes or inaccuracies in the translation of names, places, or other entities should be identified as named entity errors. CoT Response: Let’s think step by step. We solve this question by first translating the source sentence to English and then by comparing our translation with the provided translation. According to Google Translate, the correct translation of the source sentence from German to English is "Neu Fahrland is a district of Potsdam." On the other hand, the provided translation is "Neu Fahrland is a district of Pots." Note that the provided translation is missing the word "dam" at the end of Pots. Because a significant part of the translation is dropped, the translation contains an error pertaining to Dropped Content. So the answer is (E) . ✗ LEAP LOW-LEVEL Response: Let’s think step by step. We solve this question by first translating the source sentence to English and then by comparing our translation with the provided translation. According to Google Translate, the correct translation of the source sentence from German to English is "Neu Fahrland is a district of Potsdam." On the other hand, the provided translation is "Neu Fahrland is a district of Pots." Note that the provided translation omits the word "dam" from Potsdam. Because a significant part of the entity name is removed in the translation, the translation contains an error pertaining to Named Entities. So the answer is (D) . ✓
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/245e/245e2254-296c-4a22-8a66-01a9aa31c36c.png" style="width: 50%;"></div>
Boolean Expressions
Question: not not ( False and not False ) is
Principles: . . . 3. Double negation, as in ‘not not’, cancels out and returns the original value. . . .
CoT Response: Let’s think step by step. Remember that (i) expressions inside brackets are always evaluated first
and that (ii) the order of operations from highest priority to lowest priority is "not", "and", "or", respectively. We first
simplify this expression "Z" as follows: "Z = not not ( False and not False ) = not not ( A )" where "A = False and not
False". Let’s evaluate A: A = False and not False = False and True = False. Plugging in A, we get: Z = not not ( A ) =
not not (False) = not not True = not False = True. So the answer is True . ✗
LEAP LOW-LEVEL Response: Let’s think step by step. Remember that (i) expressions inside brackets are always
evaluated first and that (ii) the order of operations from highest priority to lowest priority is "not", "and", "or",
respectively. We first simplify this expression "Z" as follows: "Z = not not ( False and not False ) = not not ( A )"
where "A = False and not False". Let’s evaluate A: A = False and not False = False and True = False. Plugging in A,
we get: Z = not not ( A ) = not not (False) = not not False = False. So the answer is False . ✓
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ca1d/ca1d7bd4-54fc-4842-b930-873b506ce939.png" style="width: 50%;"></div>
# Date Understanding
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e151/e151888a-346e-4e29-bb6c-b071df2ce38f.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f6e8/f6e8213c-1c00-41f3-ad22-a547ff7b447c.png" style="width: 50%;"></div>
# Disambiguation QA
Question: In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous. Sentence: The undergraduate applied to work with the scientist because he needed a lab assistant. Options: (A) The undergraduate needed an assistant (B) The scientist needed an assistant (C) Ambiguous Principles: Pay attention to the causality and context of the sentence to identify the most likely referent for the pronoun. It is not enough to simply assume that the pronoun could refer to any possible noun mentioned in the
# sentence.
CoT Response: Let’s think step by step. Here we need to determine who the pronoun "he" might be referring to. There are two possible referents for "he", namely the undergraduate and the scientist. The verb "applied" might be able to help us determine which one is more likely (if either). Let X be the undergraduate and Y the scientist. The sentence is then of the form "X applied to work with Y because (X or Y) needed a lab assistant." Let’s consider Y first: "X applied to work with Y because Y needed a lab assistant." This case makes sense, because X applied to work with Y because Y had a need for a lab assistant. Now, consider X: "X applied to work with Y because X needed a lab assistant." This case also makes sense, because X applied to work with Y because X had a need for a lab assistant. Because both X and Y are possible at the same time, we conclude that the antecedent of the pronoun is ambiguous. So the answer is (C) . LEAP LOW-LEVEL Response: Let’s think step by step. Here we need to determine who the pronoun "he" might be referring to. There are two possible referents for "he", namely the undergraduate and the scientist. The verb "applied" might be able to help us determine which one is more likely (if either). Let X be the undergraduate and Y the scientist. The sentence is then of the form "X applied to work with Y because (X or Y) needed a lab assistant." Let’s consider Y first: "X applied to work with Y because Y needed a lab assistant." This case makes sense, because X wanted to work with Y because Y needed assistance in the lab. Now, consider X: "X applied to work with Y because X needed a lab assistant." This case doesn’t make much sense, because X, as an undergraduate, would typically be the one seeking an opportunity to work with a scientist, not the other way around. Because in this context, X is the undergraduate and Y is the scientist, the answer should be the scientist. So the answer is (B) .
# E. Learned Principles
Tables 9-39 show the high-level principles learned by LEAP. Tables 40-43 show low-level principles learned by LEAP
High-level principles (LEAP HIGH-LEVEL ) learned by LEAP for boolean expressions (BBH ) using GPT-4-0613 1. Ensure clarity and conciseness: Responses should be clear and concise to avoid confusion and maintain the user’s interest. 2. Avoid redundancies: Redundant information should be eliminated to maintain the user’s attention and improve the efficiency of communication. 3. Maintain uniqueness: Each response should be unique to keep the user engaged and provide a personalized experience. 4. Incorporate feedback: User feedback should be analyzed and incorporated into future responses to improve the quality of interaction. 5. Focus on relevance: Responses should be directly related to the user’s query or statement to maintain relevance and provide accurate information. 6. Prioritize logical reasoning: Logical reasoning should be the foundation of all responses to ensure accuracy and credibility. 7. Preserve specific details: Specific details provided by the user should be preserved in the response to show understanding and maintain context. 8. Continual improvement: Regular introspection and analysis should be conducted to continually improve the quality of responses.
High-level principles (LEAP HIGH-LEVEL ) learned by LEAP for causal judgement (BBH ) using GPT-4-0613
1. Consider all relevant details in the scenario: Ensure that all key details are taken into account when
forming a conclusion. Do not overemphasize certain details at the expense of others.
2. Understand the concept of intentionality: It’s not enough to initiate an action; the action must be carried
out as planned for it to be considered intentional.
3. Consider all factors in a scenario: This includes the actor’s control over the situation and the predictability
of the outcome when determining intentionality.
4. Distinguish between the intent to perform an action and the intent to cause a specific outcome: In questions
of causation, it’s important to understand these nuances to provide accurate analysis.
5. Understand the context of causation: Intentionality involves not just the initial action but also the control
and predictability of the outcome.
6. Avoid overemphasis on initial decisions: The initial decision to perform an action should not be
overemphasized at the expense of other factors, such as the accidental nature of the outcome.
7. Improve understanding and interpretation of intent: This includes understanding the difference between
the intent to perform an action and the intent to cause a specific outcome.
8. Ensure adequate consideration of crucial factors: In any scenario, ensure that all aspects, especially those
that are crucial to the outcome, are adequately considered before arriving at a conclusion.
High-level principles (LEAP HIGH-LEVEL ) learned by LEAP for date understanding (BBH ) using GPT-4-0613 1. Ensure clarity and precision: Responses should be clear and concise, avoiding any ambiguity or unnecessary complexity. 2. Maintain relevance: The responses should be directly related to the query or topic at hand, avoiding any irrelevant information. 3. Avoid redundancy: Each response should provide unique information, avoiding repetition of previously stated facts or ideas. 4. Prioritize understanding: The responses should be designed to enhance the user’s understanding of the topic, rather than simply providing information. 5. Foster engagement: The responses should be engaging and interactive, encouraging further dialogue and exploration of the topic. 6. Promote logical reasoning: The responses should be logically sound and well-reasoned, demonstrating a clear thought process. 7. Respect user’s perspective: The responses should respect the user’s perspective and knowledge level, avoiding any condescension or oversimplification. 8. Incorporate feedback: The responses should incorporate feedback from previous interactions, continuously improving in quality and relevance.
High-level principles (LEAP HIGH-LEVEL ) learned by