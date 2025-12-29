# Inductive Learning of Logical Theories with LLMs: A Complexity-graded Analysis
João Pedro Gandarela2†, Danilo S. Carvalho3, André Freitas1,2,3 1 Department of Computer Science, University of Manchester, United Kingdom 2 Idiap Research Institute, Switzerland 3 National Biomarker Centre, CRUK-MI, Univ. of Manchester, United Kingdom {firstname.lastname}[@manchester.ac.uk]|[@idiap.ch]†
# Abstract
This work presents a novel systematic methodology to analyse the capabilities and limitations of Large Language Models (LLMs) with feedback from a formal inference engine, on logic theory induction. The analysis is complexity-graded w.r.t. rule dependency structure, allowing quantification of specific inference challenges on LLM performance. Integrating LLMs with formal methods is a promising frontier in the Natural Language Processing field, as an important avenue for improving model inference control and explainability. In particular, inductive learning over complex sets of facts and rules, poses unique challenges for current autoregressive models, as they lack explicit symbolic grounding. While they can be complemented by formal systems, the properties delivered by LLMs regarding inductive learning, are not well understood and quantified. Empirical results indicate that the largest LLMs can achieve competitive results against a SOTA Inductive Logic Programming (ILP) system baseline, but also that tracking long predicate relationship chains is a more difficult obstacle than theory complexity for the LLMs.
 15 Aug 2024
arXiv:2408.16779v1
# 1 Introduction
The integration of Large Language Models (LLMs) with formal methods stands out as a promising frontier in the field of Natural Language Processing. It is an avenue for improving model inference control and explainability, by both complementing the content flexibility of Large Language Models (LLMs) with the systematicity of symbolic/formal systems (Quan et al., 2024a,b) and by using well-defined formal settings to assess the underlying inference properties of the model. Inductive Logic Programming (ILP) is a subfield of symbolic AI which focuses on methods that can derive (generalise) theories to explain observed facts and rules (Muggleton, 1991; Nienhuys-Cheng and de Wolf, 1997). Addressing inductive learning
over complex sets of facts and rules, poses unique challenges for current autoregressive LLMs, as they do not operate over data symbolically, rather combining an extensive set of structural an semantic signals to approximate the most probable answer in a generative fashion. While such means of problem solving might lack explicit symbolic grounding, LLMs can leverage its large-scale internal representation to support inductive-style inference. Still, the properties delivered by LLMs w.r.t. inductive learning, in particular regarding logic rules and theory induction, are not well understood and quantified. This paper presents a systematic methodology to evaluate the inductive learning properties (in the context of logic theory induction) of LLMs. It is aimed at answering the following research questions (RQs): RQ1. To what extent the combination of an LLM with feedback from a formal inference engine can compare to a SOTA inductive logic programming (ILP) system in logic theory induction, w.r.t. inference quality at different degrees of complexity? RQ2. How does the complexity of the target theories affect inference quality of LLMs for inductive reasoning?
In order to address these RQs, we propose a method for combining iterative theory refinement on stock LLMs (i.e., zero-shot inference), a formal ILP inference engine and a synthetic generator for inductive reasoning datasets, in order to perform systematic evaluations of the LLM induced theories, using state-of-the-art ILP solvers as a baseline. Moreover, to quantify the extent in which LLMs can address ILP tasks, the evaluation is graded wrt. the dependency complexity of the target rulesets. Figure 1 schematises the proposed approach. This work’s main contributions are as follows:
1. (Methodological) A novel method for system-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/dce5/dce51f89-967e-43ad-9110-e3ff2a48505d.png" style="width: 50%;"></div>
Figure 1: The proposed method to evaluate theory induction with an LLM in Prolog based on background knowledge and training examples. The process starts with a prompt generator (c) that formulates prompts for an LLM (a). Both the background knowledge and training sets are parameterised by different noise and rule complexities levels: Chain, Rooted Directed Graph (DG), Disjunctive Rooted DG, and Mixed. The LLM generates theories, which are then evaluated by a logic program interpreter (b). The evaluation feedback, including accuracy, precision, recall, and F1 scores, as well as wrongly classified examples, is used to refine the prompts iteratively. We analyse and categorise the generated theories according to their complexity (d).
# atic evaluation of LLM induced logic theories with feedback from a formal inference engine.
2. (Empirical) A detailed empirical analysis on the strengths and limitations of SOTA LLMs regarding logic theory induction generation, according to target ruleset complexity.
3. (Resources) A reusable and extensible framework for extending and assessing the inductive capabilities of LLMs.
The remainder of the paper is organised as follows: Section 2 formalises the relevant ILP concepts, dataset generation, and LLM used in the paper. Section 3 describes the proposed method and algorithms. Section 4 presents the experimental procedures, results and discussion, followed by related work (5) and conclusions (6).
# 2 Inductive Learning, Complexity & Datasets
In this section we introduce the target task and the typology of inductive learning complexity classes, as well as the dataset generation process.
# 2.1 Inductive Logic Programming
Inductive Logic Programming main objective is to generate logical hypotheses or theories from the available background knowledge, such as facts, rules, and positive and negative examples. Unlike traditional machine learning methods, ILP works
directly with logical predicates and rules, using a formal representation that is usually represented using first-order logic (FOL). For example, the fact “parent(john, tom).” means that John is the parent of Tom, and “parent(john, anne).” means that John is the parent of Anne. From that, we can create a rule (theory) “sibling(X, Y) :- parent(P, X), parent(P, Y), X ̸= Y.” . This rule states that if there exists a parent who has both X and Y as children, and X and Y are not identical, then X and Y are considered siblings. Deriving rules from a set of examples is a process known as theory induction. This task can be formally defined as follows:
Given: background knowledge (BK), a set of logical clauses that represent prior knowledge about the domain a set of positive examples (E+), a set of ground facts (instances) which the learned theory should entail (i.e., these are examples that should be true according to the theory), a set of negative examples (E−), a set of ground facts which the learned theory should not entail (i.e., these are examples that should be false according to the theory). Find a hypothesis H (a set of logical clauses) such that:
# with the background knowledge BK, should not entail any negative examples. Formally, an ILP system seeks a hypothesis H that satisfies:
with the background knowledge BK, should not entail any negative examples. Formally, an ILP system seeks a hypothesis H that satisfies:
∀e ∈E+, BK ∪H |= e
∀e ∈E−, BK ∪H ̸|= e
The learned hypothesis H should thus be a logical theory that explains the positive examples and excludes the negative examples, based on the given background knowledge.
# 2.2 Theory complexity
Inductive learning can be organised according to different classes of complexity, which involves a typology of the structural complexity of the problem. Moreover, variables such as the amount and type of noise within the evidence set (such as the number of incorrect or missing facts or completeness levels) can be integrated within this setting. Following the typology of (Cornelio and Thost, 2021) four base categories of rules can be introduced based on their dependencies: Chain, Rooted Directed Graph (RDG), Disjunctive Rooted DG, and Mixed. Each category represents a hierarchical generalisation, with each step encompassing a broader range of structures. Starting with CHAIN, which represents a linear composition of predicates, RDG is a generalisation of CHAIN, meaning it includes all chain structures but also accommodates more complex dependencies. Moving up, DRDG generalises RDG by incorporating directed relationships, thus offering a more extensive representation of dependencies. Finally, Mixed contains connected components from CHAIN, RDG, and DRDG. Each progression from CHAIN to MIXED represents a step towards greater inclusivity and complexity in the types of structures captured. The complexity classes and their characteristics are summarised in Table 1. A detailed description of each class is provided in the supplementary material (Appendix A.1).
# 2.3 Dataset synthesis
In order to generate datasets for rigorous analysis, this study employed the RuDaS tool (Cornelio and Thost, 2021) to systematically vary parameters such as noise, open-world degree, and missing data. By adjusting these factors in conjunction
<div style="text-align: center;"># Parents Recursive Alt. rules</div>
Category
# Parents
Recursive
Alt. rules
CHAIN
1
No
No
CHAIN REC.
1
Yes
No
RDG
1 – *
No
No
RDG REC.
1 – *
Yes
No
DRDG
1 – *
No
Yes
DRDG REC.
1 – *
Yes
Yes
MIXED
1 – *
Yes
Yes
Table 1: Characteristics for each dataset category. # Parents refers to the number of rules that each rule can deduce relevant facts to. Recursive refers to whether a predicate in the head of a rule can also occur in the body. Alt. rules indicate whether a predicate can be deduced by alternative rules.
with the category parameter, it is possible to ensure comprehensive coverage of different structural configurations and complexity levels. For each configuration, the following settings were applied:
• The minimum and maximum number of Directed Acyclic Graphs (DAGs) were both set to 1 (mindags = 1, maxdags = 1).
• Noise levels were systematically varied at intervals of 0.1, 0.2, and 0.3.
• The percentage of missing data (missing) and open world degree (owa) were similarly varied across 0.1, 0.2, and 0.3.
• The category parameter was set to cover all the complexity classes described in the previous section, and listed in Table 1. The distribution of each category in the synthesised dataset is an independent hyperparameter, discussed in Section 4.
Further details regarding the dataset generation can be found in Appendix E.
# 3 Proposed Approach
The proposed approach can be divided in two parts: iterative refinement and graded evaluation. They form a systematic evaluation loop, covering all the complexity classes described in the previous section, for a given set of LLMs and dataset synthesis parameters.
# 3.1 Iterative Refinement
Consists of an iterative refinement loop that alternates between the generation of a theory by a language model and the evaluation of said theory
through a formal interpreter. It is comprised of the following components, as illustrated in Figure 1: (a) A language model capable of generating a theory H, based on background knowledge, positive and negative examples, and hypothesis search, provided as a prompt text in a logic program language. Typically an LLM with structured (e.g., code) generation capabilities. (b) A logic program interpreter. We use Prolog (Warren et al., 2023) as the logic program language. (c) A prompt generation component, that interleaves logical programming language with natural language queries designed to drive the theory induction responses. The logical programming language expresses background knowledge and the relevant outputs of the program interpreter. (d) An evaluation module, that uses the logic program interpreter to execute the generated theory H as logical rules, and computes a set of evaluation metrics. Given: background knowledge (BK), positive examples (E+), negative examples (E−), and assuming a language model LM which can find a hypothesis H (a set of logical clauses) such that it satisfies the conditions of completeness (for every example e ∈E+, H ∪BK |= e) and consistency (For every example e ∈E−, H ∪BK ̸|= e).
1. Context Representation: Represent the input to the language model as a combination of background knowledge and examples: Context = encode(BK, E+, E−).
2. Theory Generation: From a background knowledge set of clauses sampled from a knowledge base dataset, including positive and negative examples, a prompt is created for the LM to induce a theory as Prolog code, i.e. using the language model to generate a set of logical clauses (hypothesis H): H = LM(Theory Prompt + Context).
# 3. Evaluation of Hypothesis: Checking for the completeness and consistency conditions: True Positives (TP): The number of positive examples correctly entailed by the hypothesis. TP = |{e ∈E+ | BK ∪H |= e}|
3. Evaluation of Hypothesis: Checking for the completeness and consistency conditions: True Positives (TP): The number of positive examples correctly entailed by the hypothesis. TP = |{e ∈E+ | BK ∪H |= e}|
False Positives (FP): The number of negative examples incorrectly entailed by the hypothesis. FP = |{e ∈E−| BK ∪H |= e}|
False Negatives (FN): The number of positive examples not entailed by the hypothesis.
# True Negatives (TN): The number of negative examples correctly not entailed by the hypothesis. TN = |{e ∈E−| BK ∪H ̸|= e}|
True Negatives (TN): The number of negative examples correctly not entailed by the hypothesis. TN = |{e ∈E−| BK ∪H ̸|= e}|
from which precision (P) ( TP TP+FP ), recall (R) (Recall = TP TP+FN ) and F1-score (F1) (2 · Precision·Recall Precision+Recall) can be generated.
4. Theory refinement: Following an initial evaluation, the LM is tasked to refine the induced theory iteratively. Each refinement round involves adjusting the theory based on feedback from the Prolog interpreter validation. The refinement aims to improve the theory’s performance by addressing misclassifications and enhancing its predictive capabilities. If H does not satisfy completeness and consistency, update the input representation based on feedback and generate a new hypothesis using the language model, given the Feedback Context ←{FP, FN, P, R, F1} and the final prompt input Input ←(Refinement Prompt + Context + Feedback Context): H ←LM(Input). The main loop of the algorithm continues until the evaluation metrics meet the defined thresholds or the maximum number of iterations is reached. In each iteration, the language model generates a theory based on the current prompt. This generated theory is then evaluated using the logic program interpreter, in our case Prolog, resulting in a validation set. Evaluation metrics are computed from these validation results and stored. Based on the feedback from the validation, a new prompt is generated by incorporating the initial knowledge base sample, the current theory, and the validation feedback. Our approach removes recursive rules from the LLM-induced theory before evaluation. The refinement loop is summarised in Algorithm 1. The process starts by sampling facts from the knowledge base dataset to create kb. An initial prompt is then generated using these sampled facts, denoted as prompt. 5. Termination: The process continues iteratively until a maximum number of iterations is reached.
# 3.2 Graded evaluation
A synthetic data generator is used to control for the input parameters of the ruleset complexity, namely: categorical distribution (CHAIN, RDG, DRDG, etc.), background knowledge, positive examples
Algorithm 1 Iterative LM theory refinement
Define:
KB as the background knowledge
dataset.
Define: PGen as the prompt generator.
Define: Exs as the positive and negative exam-
ples.
Define: LM as the language model.
Define: PL as the logic program interpreter.
Define: Eval as the evaluation module.
Define: M as the evaluation metrics set.
Define: Maxiter as the maximum number of
iterations.
Define: MTtresh ←Map : M →R as the
evaluation metrics treshold.
Let prompt ←PGen(KB, Examples)
Let iter ←0
Let results ←Map : M →R
while (∃m
∈
results
:
results[m]
<
MTtresh[m]) ∧(iter < Maxiter) do
theory ←LM(prompt)
results ←Eval(Examples)
prompt ←PGen(KB, theory, Exs)
iter ←iter + 1
end while
and negative examples as well as the amount of noise introduced within the dataset.
1. Categorised Learning Sets: Consisting of C: set of ruleset complexity categories (e.g., CHAIN, RDG, DRDG, etc.), N: set of noise levels, S: number of samples per combination of C and N. For each c ∈C and n ∈N, generate S datasets {Dc,n,i | i = 1, . . . , S} where each dataset Dc,n,i includes:
Dc,n,i = (BKc,n,i, E+ c,n,i, E− c,n,i, noisec,n)
2. Hypothesis Generation and Evaluation: For each dataset Dc,n,i, use a learning algorithm to generate a hypothesis Hc,n,i, tracking the F1 score Fc,n,i and processing time Tc,n,i at each iteration and recording the best F1 score F1c,n,i and corresponding processing time Timec,n,i:
F1c,n,i = max(Fc,n,i) Timec,n,i = time until max(Fc,n,i)
3. Aggregation: The information is then aggregated by complexity category and noise level for all the samples, averaging times and F1 scores, to
obtain the complete graded evaluation statistics. For each combination of c ∈C and n ∈N, compute the average F1 score and average processing time:
# 4 Experiments
In order to answer the research questions, a set of experiments was elaborated to systematically analyse the theory inducing capabilities of a set of most popular open-source LLMs and two versions of GPT, with the proposed approach, having the state-of-the-art ILP system Popper (Cropper and Morel, 2021) as a baseline. The tests covered all data categories discussed on Section 2, allowing a graded analysis w.r.t. the expected level of induction complexity and tolerated noise.
# 4.1 Experimental Setup & Dataset
For each data category, five datasets were generated using RuDaS (Cornelio and Thost, 2021). The size of each dataset was set to XS (min = 50, max = 100, support = 3), and noise, missing, and open world were all set to 0.1, then all set to 0.2, and finally all set to 0.3. This resulted in 105 datasets in total, with 35 datasets for each rate. Subsequently, two methods are used to induce a theory for each dataset: (1) employing Popper, with NuWLS (Chu et al., 2023) and WMaxCDCL, varying its time limit parameter from 10 to 800 seconds; (2) applying the proposed iterative LM theory refinement method (Section 3), with parameters Maxiter = 4 and MTthresh = 1.0. Three different LLM models were used for (2): Open AI1’s model GPT-4o2, Mistral AI3’s Mixtral-8x7B4 (Jiang et al., 2023), and Google’s Gemma5 (Team et al., 2023). Table 2 presents a comprehensive overview of statistical metrics pertaining to each category of data, in order of complexity (except MIXED). We computed the average F1-score for each category, taking into account the level of noise, open
1https://openai.com/ 2https://openai.com/index/hello-gpt-4o 3https://mistral.ai/ 4https://huggingface.co/mistralai/ Mixtral-8x7B-Instruct-v0.1 5https://huggingface.co/google/gemma-7b-
Categories
Facts
Positive
Negative
CHAIN
67.6 ± 5.4
23.6 ± 6.5
4.8 ± 2.1
CHAIN REC.
54.2 ± 14.1
19.6 ± 7.8
4.4 ± 2.5
RDG
60.2 ± 9.8
18.6 ± 12.3
4.2 ± 3.4
RDG REC.
63.2 ± 9.0
16.6 ± 4.7
3.4 ± 1.3
DRDG
61.2 ± 14.1
31.0 ± 26.0
8.0 ± 8.2
DRDG REC.
54.2 ± 12.5
34.0 ± 22.2
9.0 ± 6.2
MIXED
54.4 ± 18.1
24.2 ± 12.3
4.8 ± 2.7
Table 2: Statistics for each dataset category. A detailed description of each can be found in Section 2.
world scenarios, and missing facts. The mean values reported are based on the results obtained from the theory that was generated from the train set and evaluated on the test set. The experiment used the OpenAI service for GPT models. For Popper, Llama3-8B-Instruct, Gemma-7B-It and Mixtral-8x7B-Instruct-v0.1, it was conducted on a computer with an Intel(R) Xeon(R) Gold 5217 CPU @ 3.00GHz, 188GB RAM, and 2x NVIDIA RTX A6000 (48GB VRAM) GPUs. The software used was CUDA 12.3, PyTorch 2.2.2, and Transformers 4.41.2. Prompt templates used were included in the supplementary material (Appendix C).
# 4.2 Results & Discussion
Overall results for F1 are presented in Figure 2. We additionally report on processing times as a measure of practical interest in Figure 3. We present the values obtained in detail in the supplementary material (Appendix B, in Tables 4 and 5). Gemma-7B-It results are not included as it failed to generate valid theories. The results reveal significant insights into LLM capabilities and limitations w.r.t. theory induction, which are summarised as follows:
# LLMs can achieve competitive performance
against the baseline, specially at higher noise levels. The larger scale models (GPT3.5, 4) demonstrate more resilience to noise and consistent F1 across the different categories, as indicated in Figure 2, with an average F1-score difference of ±0.25 against Popper. This answers the overall quality part of RQ1.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9e86/9e86f9b7-51be-4492-8597-f3f698e9b785.png" style="width: 50%;"></div>
Figure 2: F1 score trends across categories. Different models (GPT-4o, Llama3 8b instruct, Popper, and Mixtral-8x7B-Instruct-v0.1) under varying noise levels and categories reveal distinct performance patterns. GPT-4o demonstrates stable accuracy yet sensitivity to noise, particularly in complex rule-based categories like RDG and DRDG. Mixtral-8x7B-Instruct-v0.1 exhibits mixed results with notable variability across categories particularly in more complex tasks. Llama3 8b instruct struggles with low scores, indicating challenges in reasoning and theory generation.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d6b6/d6b6ca67-ea4c-408c-8cfc-c4a17207df0e.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Performance on time consumption trends across categories using a logarithmic scale. The data consistently shows that LLM outperforms Popper in all intervals. The results however do not represent a measure of efficiency, as the computational resources employed are vastly different across methods.</div>
Figure 3: Performance on time consumption trends across categories using a logarithmic scale. The data consistently shows that LLM outperforms Popper in all intervals. The results however do not represent a measure of efficiency, as the computational resources employed are vastly different across methods.
ited capacity of tracking long relationship chains of independent predicates. A part of RQ2 answer. Increasing iteration limits does not monotonically improve results for LLMs. Upon increasing the iteration limits from 1 to 4, it was found that the metrics can either increase or decrease nonmonotonically. Thus Maxiter was set to 4 and the iteration with the best accuracy is taken as the final
Performance is remarkably lower on complex rule sets at moderate noise levels. Responses for complex categories, such as RDG and DRDG display higher variance and LLM hallucination artifacts, such as valid rules containing predicates that do not exist in the rule set. We present an error analysis in Section 4.3. For instance, a comparison of the results for the RDG category generated by GPT-4o under noise levels set to 0.1 and 0.3 reveals a significant decline in performance, with the F1score dropping from 0.75 to 0.22. A comparable pattern is observed with GPT-3.5 Turbo for RDG and DRDG and Mixtral RDG in the presence of elevated noise levels, with GPT-3.5 Turbo scores going from 0.56 to 0.0, 0.28 to 0.08, and Mixtral8x7B going from 0.60 to 0.0.This complements the answer to RQ2. Further details are included in the supplementary material (Appendix B). Induction capability varies substantially across models. Using the same inputs resulted in vastly different responses from different models, suggesting critical influence from model size in parameters. Figure 2 illustrates this: When comparing GPT-4o, Mixtral-8x7B and Llama3 at noise levels set to 0.1 and 0.3 respectively, the consistency in generating a valid theory correlates to their relative size. At a noise level of 0.1, GPT-4o’s F1 score is almost twice that of the GPT-3.5-Turbo in average, and at a noise level of 0.3, the difference increases to a ratio of 4, indicating substantially higher noise resiliency. The performance gap is more pronounced when comparing with Llama38B, where GPT-4o F1 score is 21 times higher at the lowest noise setting. Mixtral-8x7B-It-v0.1 performs similarly to GPT3.5-Turbo at lower noise levels, scoring 13.4% higher in average at 0.1 noise. However, its performance becomes less stable at higher noise levels. It consistently outperforms Llama3-8B-it, at 0.1 noise, with a F1-score 11 times higher in average. Model size does not correlate to noise resilience Despite being able to achieve higher scores than GPT-3.5 and Mixtral-8x7B in some of the tests (e.g., RDG-R @noise = 0.1, CHAIN-R @noise = 0.2) and scoring higher on intermediate noise than on low noise, Llama3-8B did not consistently generate valid theories. On the other hand, Mixtral-8x7B, a much larger model, is particularly susceptible to noise, with an average
Performance is remarkably lower on complex rule sets at moderate noise levels. Responses for complex categories, such as RDG and DRDG display higher variance and LLM hallucination artifacts, such as valid rules containing predicates that do not exist in the rule set. We present an error analysis in Section 4.3. For instance, a comparison of the results for the RDG category generated by GPT-4o under noise levels set to 0.1 and 0.3 reveals a significant decline in performance, with the F1score dropping from 0.75 to 0.22. A comparable pattern is observed with GPT-3.5 Turbo for RDG and DRDG and Mixtral RDG in the presence of elevated noise levels, with GPT-3.5 Turbo scores going from 0.56 to 0.0, 0.28 to 0.08, and Mixtral8x7B going from 0.60 to 0.0.This complements the answer to RQ2. Further details are included in the supplementary material (Appendix B).
Induction capability varies substantially across models. Using the same inputs resulted in vastly different responses from different models, suggesting critical influence from model size in parameters. Figure 2 illustrates this: When comparing GPT-4o, Mixtral-8x7B and Llama3 at noise levels set to 0.1 and 0.3 respectively, the consistency in generating a valid theory correlates to their relative size. At a noise level of 0.1, GPT-4o’s F1 score is almost twice that of the GPT-3.5-Turbo in average, and at a noise level of 0.3, the difference increases to a ratio of 4, indicating substantially higher noise resiliency. The performance gap is more pronounced when comparing with Llama38B, where GPT-4o F1 score is 21 times higher at the lowest noise setting. Mixtral-8x7B-It-v0.1 performs similarly to GPT3.5-Turbo at lower noise levels, scoring 13.4% higher in average at 0.1 noise. However, its performance becomes less stable at higher noise levels. It consistently outperforms Llama3-8B-it, at 0.1 noise, with a F1-score 11 times higher in average. Model size does not correlate to noise resilience Despite being able to achieve higher scores than GPT-3.5 and Mixtral-8x7B in some of the tests (e.g., RDG-R @noise = 0.1, CHAIN-R @noise = 0.2) and scoring higher on intermediate noise than on low noise, Llama3-8B did not consistently generate valid theories. On the other hand, Mixtral-8x7B, a much larger model, is particularly susceptible to noise, with an average
F1-score drop of over 0.8 from noise = 0.1 to noise = 0.2 and a monotonic performance reduction with the increase of the noise level.
Regarding the parameterisation of each method, some higher-level observations on the trade-off between time and inference quality can be summarised as follows:
Computational scalability allow LLMs to operate at substantially lower times. While Popper is a serial algorithm, the parallel nature of transformer-based LLMs allows them to operate at times about 3 orders of magnitude lower, given enough computational resources. This can be observed in Figure 3, for all complexity classes and all tested noise intervals.
# 4.3 Error analysis
The errors found in the evaluation of the generated theories could be separated in two categories: syntactic and logical. Syntactic errors occur when the generated response does not match the logic programming language grammar. For example, the following response: theory :p(X, Y), pos(p0(X, Y)) - positive. p(X, Y), neg(p0(X, Y)) - negative. \+ p(X, Y), pos(p0(X, Y)) - false. \+ p(X, Y), neg(p0(X, Y)) - true.
<div style="text-align: center;">is not valid Prolog and will fail evaluation.</div>
Logical errors occur when the generated response has correct grammar, but cannot be induced from the examples. Consider the following Prolog theory: theory :p(X, Y) :- p1(X, Y); p3(X, Y); p4(X, Y); p7(X, Y); p8(X, Y); p0(X, Y), not neg(p(X, Y)), (pos(p(X, Y)) - true; fail).
Logical errors occur when the generated response has correct grammar, but cannot be induced from the examples. Consider the following Prolog theory: theory :p(X, Y) :- p1(X, Y); p3(X, Y); p4(X, Y); p7(X, Y); p8(X, Y); p0(X, Y), not neg(p(X, Y)), (pos(p(X, Y)) - true; fail). The response contains the head of the clause "theory," as well as the predicates "p" and "pos", which do not exist in the BK. Table 3 presents a distribution of error categories for the analysed models. A more detailed analysis of the models outputs is included in the supplementary material (Appendix G).
The response contains the head of the clause "theory," as well as the predicates "p" and "pos", which do not exist in the BK. Table 3 presents a distribution of error categories for the analysed models. A more detailed analysis of the models outputs is included in the supplementary material (Appendix G).
# 5 Related Work
Neural symbolic computation combines neural networks with symbolic methods to enhance AI reasoning capabilities. (Yang et al., 2017) introduced
Model
# Syntactic
Logical
GPT-4o
0%
100%
GPT3.5
0%
100%
Llama3-8B
46%
54%
Mixtral-8x7B
20%
80%
Gemma-7B-it
100%
0%
Table 3: Error distribution for each of the evaluated models. Gemma-7B-it did not produce valid Prolog.
Neural Logic Programming, An end-to-end differentiable model integrating neural networks with logic programming. Within the LLM-Symbolic space, (Wan et al., 2024) developed LogicAsker, which evaluates and improves LLMs’ logical reasoning using propositional and predicate logic. It identifies reasoning failures and enhances capabilities through in-context learning. Within the context of symbolic toolformers over LLMs, (Quan et al., 2024a,b) proposed methods of improving explanatory reasoning with the support of formal iterative cycles using both logical solvers and theorem provers for supporting more controlled step-wise reasoning. Despite these advancements at the interface of LLM-based reasoning and formal controls, it is unclear the extent and the conditions in which LLMs can perform formal reasoning (Huang and Chang, 2023). (Sinha et al., 2019) introduced CLUTRR, a benchmark assessing LLMs’ structural learning by inferring kinship relations in stories, requiring relationship extraction and logical rule inference. (Zhu et al., 2024) proposed the Hypotheses-to-Theories (HtT) framework to improve LLM reasoning by learning explicit rules in two stages: generating and verifying rules (induction) and using the obtained rule library for reasoning (deduction). HtT enhances relational and numerical reasoning and concept learning. (Madaan et al., 2023) introduces a novel technique for improving machine learning models through iterative refinement. This approach allows models to improve their performance by continuously evaluating and adjusting their predictions based on self-generated feedback. By critiquing their own outputs, models can identify errors and make corrections over successive iterations, leading to increased accuracy and robustness across different tasks. Our work builds upon this approach by employing a formal method to evaluate and then refine itself. In a related study, (Dziri et al., 2023), the authors investigate the limitations of transformer models
in handling composition tasks. Their results show that, despite their strengths, transformers face significant challenges in dealing with compositionality, which involves understanding and generating complex structures from simpler components. This limitation highlights the need for innovative approaches, such as self-refining, to further enhance the capabilities of machine learning models.
In contrast, our work focuses on the still underexplored area of assessing and controlling inductive learning/inference capabilities of LLMs. These contributions integrate LLMs and formal logic for robust theory induction and allows a graded analysis of LLM capabilities, with respect to theory induction complexity.
# 6 Conclusion
In this study we thoroughly investigate the integration of state-of-the-art formal theory induction within the context of large language models (LLMs), aiming to elucidate the extent in which LLMs can systematically perform inductive learning for theories spanning across different complexity levels. At the heart of this exploration lies the recognition of relational data’s inherent semantic depth, stemming from its symbolic representations. The empirical results presented here have indicated the ability of LLMs to address inductive learning tasks, with the largest LLMs achieving competitive results against the algorithmic SOTA with better tolerance to higher noise levels, which can be attributed to their semantic flexibility. This flexibility however has certain limitations, as we found that tested language models are more limited by their capacity of tracking long relationship chains of independent predicates than by the dependency complexity of the rule sets (disjunctiveness, recursivity). As future work we plan to utilise larger datasets to test the scalability of the proposed approach, allowing researchers to assess its performance across a broader range of scenarios. Additionally, it is worth considering the integration of the LLM’s output as an initial input for the ILP process, potentially leveraging the strengths of both approaches to overcome their respective limitations. Another avenue, is the use of ILP middle steps, such as the bottom clause, to help LLM induce a theory.
# 7 Limitations
While the proposed evaluation methodology aims to cover a wide range of logic theory induction complexity, it is limited in its resolution to the categories specified by (Cornelio and Thost, 2021), and does not quantify other ruleset characteristics, such as workspace size or unification rate in the case of Prolog (Dikovsky, 1993). The methodology compares all models under the same inputs. Therefore it is not concerned with extracting maximum performance of any given model, but obtaining a relative assessment of their fundamental capabilities. This means that the empirical analysis reported scores should not be taken as a measure of SOTA performance. Furthermore, the time gains demonstrated in the experiments are presented as an achievable result, conditioned to the combination of software and hardware indicated in the paper and the services provided by third-parties (e.g., OpenAI). They should not be interpreted as a measure of computational efficiency.
# Acknowledgements
This work was partially funded by the Swiss National Science Foundation (SNSF) project NeuMath (200021_204617) and by the Manchester Experimental Cancer Medicine Centre and the NIHR Manchester Biomedical Research Centre.
# References
Yi Chu, Shaowei Cai, and Chuan Luo. 2023. Nuwls: Improving local search for (weighted) partial maxsat by new weighting techniques. Proceedings of the AAAI Conference on Artificial Intelligence, 37(4):3915– 3923. Cristina Cornelio and Veronika Thost. 2021. Synthetic datasets and evaluation tools for inductive neural reasoning. In Proceedings of the 30th International Conference on Inductive Logic Programming, ILP202021 @ IJCLR. Andrew Cropper and Rolf Morel. 2021. Learning programs by learning from failures. Machine Learning, 110(4):801–856. A Ja Dikovsky. 1993. On computational complexity of prolog programs. Theoretical Computer Science, 119(1):63–102. Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jian, Bill Yuchen Lin, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D Hwang, et al. 2023. Faith and fate: Limits of transformers on compositionality. arXiv preprint arXiv:2305.18654.
Jie Huang and Kevin Chen-Chuan Chang. 2023. Towards reasoning in large language models: A survey. In Findings of the Association for Computational Linguistics: ACL 2023, pages 1049–1065, Toronto, Canada. Association for Computational Linguistics.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder, Shashank Gupta, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. Preprint, arXiv:2303.17651.
Stephen Muggleton. 1991. Inductive logic programming. New generation computing, 8:295–318.
Shan-Hwei Nienhuys-Cheng and Roland de Wolf. 1997 What is inductive logic programming? Springer.
Xin Quan, Marco Valentino, Louise A. Dennis, and André Freitas. 2024a. Enhancing ethical explanations of large language models through iterative symbolic refinement. Preprint, arXiv:2402.00745.
Xin Quan, Marco Valentino, Louise A. Dennis, and André Freitas. 2024b. Verification and refinement of natural language explanations through llm-symbolic theorem proving. Preprint, arXiv:2405.01379.
Koustuv Sinha, Shagun Sodhani, Jin Dong, Joelle Pineau, and William L. Hamilton. 2019. CLUTRR: A diagnostic benchmark for inductive reasoning from text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4506–4515, Hong Kong, China. Association for Computational Linguistics. Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805. Yuxuan Wan, Wenxuan Wang, Yiliu Yang, Youliang Yuan, Jen-tse Huang, Pinjia He, Wenxiang Jiao, and Michael R. Lyu. 2024. A & B == B & A: triggering logical reasoning failures in large language models. CoRR, abs/2401.00757. David S Warren, Veronica Dahl, Thomas Eiter, Manuel V Hermenegildo, Robert Kowalski, and Francesca Rossi. 2023. Prolog: The Next 50 Years, volume 13900. Springer Nature.
David S Warren, Veronica Dahl, Thomas Eiter, Manuel V Hermenegildo, Robert Kowalski, and Francesca Rossi. 2023. Prolog: The Next 50 Years, volume 13900. Springer Nature.
Fan Yang, Zhilin Yang, and William W Cohen. 2017. Differentiable learning of logical rules for knowledge base reasoning. Advances in neural information processing systems, 30.
Fan Yang, Zhilin Yang, and William W Cohen. 2017. Differentiable learning of logical rules for knowledge base reasoning. Advances in neural information processing systems, 30.
Zhaocheng Zhu, Yuan Xue, Xinyun Chen, Denny Zhou, Jian Tang, Dale Schuurmans, and Hanjun Dai. 2024. Large language models can learn rules. Preprint, arXiv:2310.07064.
Zhaocheng Zhu, Yuan Xue, Xinyun Chen, Denny Zhou, Jian Tang, Dale Schuurmans, and Hanjun Dai. 2024. Large language models can learn rules. Preprint, arXiv:2310.07064.
# Appendices
# A Further theoretical background
# A.1 Detailed Complexity Classes
Category Chain. In this category, every rule, except the root, deduces facts relevant to precisely one other rule. Essentially, each node has at most one parent, and each rule is associated with at most one other rule that might infer relevant facts. Recursive rules, where the predicate in the head also occurs in the body, are exceptions, as they are relevant both for themselves and one additional rule. For example:
p5(X, Y) :- p0(X, Z), P2(Y, W). p0(X, Y) :- p3(X, Z), p4(W, Y). p3(X, Y) :- p6(X, Z), p7(W, Y).
For example, according to Rule 1, p0(X, Z) is necessary for p5(X, Y ). Therefore, satisfying p5(X, Y ) requires p0(X, Z), which in turn requires p3(X, Z) and p4(W, Y ). This creates a dependency chain where p5(X, Y ) relies on p3(X, Z) and p4(W, Y ).
Category Rooted DG (RDG). This category generalises the Chain category. Here, every rule can be relevant for several others, and each node can have multiple parent nodes. Furthermore, for each rule, there may be several other rules that might infer facts relevant for it. However, for each predicate occurring in the body of a rule, there must be at most one other rule with this predicate in the head. In other words, there are no alternative rules to derive facts relevant for a rule with respect to a specific body atom. For example:
p0(X0,X1) :- p1(X1,X2),p3(X0,X1). p3(X0,X1) :- p8(X0,X1),p6(X0,X1). p1(X1,X2) :- p7(X2,X1).
In the given example, each rule has at least one child node. For instance, p0(X0, X1) has two child nodes: p1(X1, X2) and p3(X0, X1). Each predicate in the body of a rule corresponds to at most one rule with that predicate in the head.
There are no alternative rules for deriving facts related to a specific body atom. For example, p1(X1, X2) appears in the body of p0(X0, X1) and only one rule has p1(X1, X2) in the head: p1(X1, X2) :- p7(X2, X1). The same applies to p3.
Category Disjunctive Rooted DG (DRDG). Category DRDG generalises Category RDG by allowing for alternative rules represented as children of an "OR" node. For instance:
p7(X0,X1) :- p5(X0,X1). p5(X0,X1) :- p0(X0,X1). p5(X0,X1) :- p8(X1,X0).
In the example, the first rule states p7(X0, X1) is true if p5(X0, X1) is true, indicating p7 depends on p5. The second rule states p5(X0, X1) is true if p0(X0, X1) is true, showing p5 depends on p0. The third rule states p5(X0, X1) is true if p8(X1, X0) is true, adding an alternative condition with swapped arguments. Thus, p5 acts as an "OR" condition in the first rule’s body and the second and third rules’ heads.
Category Mixed. A rule graph in this category contains connected components of different categories mentioned above. Additionally, recursion is allowed, meaning that the head of the rule may appear in the body as well.
# B Further empirical data & findings
GPT-4o shows stable performance with moderate to high accuracy but is sensitive to noise, especially in RDG and DRDG. For instance, its F1 score in RDG drops from 0.75 at noise level 0.1 to 0.25 at noise level 0.3. GPT-3.5-Turbo does not perform well with complex categories like RDG and DRDG under noise, with an F1 score of 0 at noise level 0.3 in RDG. Mixtral-8x7B-Instruct-v0.1 shows high variability w.r.t. noise, performing reasonably well in RDG (0.64 F1 at noise level 0.1, dropping to 0.43 at noise level 0.3) but with significant time consumption, especially in DRDG (130.160 seconds at noise level 0.2). It does not perform well with complex rule sets like DRDG across all noise levels. Llama3-8B instruct has low accuracy across most categories, with slight improvement at higher noise levels but increased time consumption. At noise level 0.1, it achieves an F1 score above 0 only in RDG R. It often fails to produce valid theories or
Category
Noise 0.1
Noise 0.2
Noise 0.3
F1
Time (s)
F1
Time (s)
F1
Time (s)
MIXED
0.51
1071.53
0.44
817.15
0.35
2418.88
CHAIN
0.80
1397.35
0.46
1254.33
0.57
3824.31
CHAIN R.
0.77
1123.25
0.41
1190.14
0.14
3646.43
RDG
0.74
1122.13
0.57
854.85
0.50
2460.90
RDG R.
0.71
1523.37
0.68
940.50
0.38
1659.27
DRDG
0.77
934.98
0.41
1089.47
0.25
1363.27
DRDG R.
0.68
927.30
0.48
882.28
0.26
820.51
<div style="text-align: center;">Table 4: Results for different categories with theory induced by Popper and different noise levels.</div>
Category
GPT-4o - noise 0.1
GPT-4o - noise 0.2
GPT-4o - noise 0.3
F1 (avg)
Time (s)
F1 (avg)
Time (s)
F1 (avg)
Time (s)
MIXED
0.70
8.57
0.52
10.73
0.62
9.81
CHAIN
0.52
8.54
0.42
11.05
0.35
8.29
CHAIN R.
0.72
11.48
0.53
8.86
0.49
8.13
RDG
0.75
8.80
0.50
11.94
0.22
20.16
RDG R.
0.74
10.83
0.55
7.35
0.49
10.79
DRDG
0.46
12.59
0.39
16.44
0.42
11.14
DRDG R.
0.83
13.11
0.32
13.29
0.12
8.45
Category
GPT-3.5-Turbo - noise 0.1
GPT-3.5-Turbo - noise 0.2
GPT-3.5-Turbo - noise 0.3
F1 (avg)
Time (s)
F1 (avg)
Time (s)
F1 (avg)
Time (s)
MIXED
0.20
4.32
0.35
4.11
0.32
4.20
CHAIN
0.24
3.47
0.33
7.88
0.11
3.13
CHAIN R.
0.545
2.592
0.00
7.11
0.00
3.05
RDG
0.56
4.22
0.29
3.19
0.00
3.76
RDG R.
0.20
3.48
0.02
4.73
0.00
3.98
DRDG
0.28
4.63
0.22
8.91
0.08
3.96
DRDG R.
0.31
5.01
0.15
3.06
0.01
11.14
Category
Llama3-8B-it - noise 0.1
Llama3-8B-it - noise 0.2
Llama3-8B-it - noise 0.3
F1 (avg)
Time (s)
F1 (avg)
Time (s)
F (avg)
Time (s)
MIXED
0.00
62.54
0.00
176.31
0.02
51.55
CHAIN
0.00
38.86
0.21
12.84
0.06
29.31
CHAIN R.
0.00
31.39
0.32
34.90
0.08
57.75
RDG
0.00
41.42
0.00
18.42
0.07
55.80
RDG R.
0.21
20.86
0.20
36.45
0.00
40.70
DRDG
0.00
76.70
0.08
45.48
0.04
60.10
DRDG R.
0.00
25.96
0.00
43.88
0.00
14.61
Category
Mixtral-8x7B-It-v0.1 - noise 0.1
Mixtral-8x7B-It-v0.1 - noise 0.2
Mixtral-8x7B-It-v0.1 - noise 0.3
F1 (avg)
Time (s)
F1 (avg)
Time (s)
F1 (avg)
Time (s)
MIXED
0.36
34.04
0.21
65.83
0.20
71.60
CHAIN
0.49
50.07
0.00
45.15
0.16
45.02
CHAIN R.
0.47
29.56
0.00
87.16
0.08
69.38
RDG
0.60
75.42
0.05
35.68
0.00
101.54
RDG R.
0.48
74.58
0.00
123.80
0.00
69.17
DRDG
0.10
90.51
0.28
130.16
0.12
82.51
DRDG R.
0.15
60.61
0.00
97.11
0.00
64.92
Table 5: Performance metrics for various categories under different noise conditions.
introduces new predicates incorrectly. For instance, the rule p(X, Y) - p2(X, Y); p0(X, Y); p4(X, Y); p9(X, Y). is valid, but the predicate p, in the head of the rule, does not exist in the BK, neither is the target predicate. Llama3-8B was the only model to exhibit this pattern. The models generally present higher initial accuracy on recursive (R) categories, but are more sensitive to noise on them, leading to larger performance drops. For example, on DRDG-R, GPT-4o’s F1 score drops from 0.83 at noise level 0.1 to 0.120 at noise level 0.3. Non-recursive categories like
introduces new predicates incorrectly. For instance, the rule p(X, Y) - p2(X, Y); p0(X, Y); p4(X, Y); p9(X, Y). is valid, but the predicate p, in the head of the rule, does not exist in the BK, neither is the target predicate. Llama3-8B was the only model to exhibit this pattern.
The models generally present higher initial accuracy on recursive (R) categories, but are more sensitive to noise on them, leading to larger performance drops. For example, on DRDG-R, GPT-4o’s F1 score drops from 0.83 at noise level 0.1 to 0.120 at noise level 0.3. Non-recursive categories like
<div style="text-align: center;">Noise 0.2</div>
<div style="text-align: center;">Noise 0.3</div>
CHAIN present more stable performance
Time variance has an inverse relation w.r.t. noise levels on LLMs when compared with Popper. As the noise increases, Popper may take less time to induce a theory based on the remaining relevant data, as indicated by the scattering pattern progression in Figure 4, while the LLMs are more likely to take longer to process it. Detailed values are presented in Tables 4 and 5.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/64e4/64e4df87-5ad5-49b5-86be-97823bf80881.png" style="width: 50%;"></div>
Figure 4: Relationship between the F1 score and the logarithm of processing time (in seconds) for five different mixed models—Popper, GPT-4, GPT-3.5-turbo, Llama3, and Mixtral—across three noise levels: 0.1, 0.2, and 0.3 and each rule set category: CHAIN (•), CHAIN R.(□), RDG( ), RDG R.(+), DRDG(⋆), DRDG R.(♢), and MIXED(X). Each subplot corresponds to a different noise level, showing how each model’s performance and processing time vary with increasing noise. While Popper always takes more time to develop a theory, the other two levels (0.5 - 1.0, 1.0 - 2.5) correspond to different execution environments. Time variance changes in opposite ways w.r.t. noise on Popper vs. the LLMs.
# C Prompt templates
# For the iterative LM theory refinement method, the following template was used for the initial prompt:
For the iterative LM theory refinement method, the following template was used for the initial prompt:
Induce a theory based on background knowledge, positive and negative examples. Write it in Prolog. Do not give an explanation. Answer only the theory.
Examples: {positive and negative examples}
For the refinement steps, the following prompt template was used: Prompt p2...pn: The theory scored: accuracy = {acc} precision = {precision} recall = {recall} f1 = {f1}
and got these examples wrongly: {examples that were misclassified}
Refine the theory. Answer only the theory.
The prompt templates were designed to be objective and minimal, containing only the necessary instructions and data for solving the task. They were adjusted using a small sample of inputs, to minimise the syntax errors across all models. The same prompt templates were used across all language models.
# D Reproducibility
Upon acceptance of this paper, we will release all code and data associated with our study. The raw data, along with detailed instructions for data processing, are accessible via the provided repository link. Any proprietary tools or materials used in this study are either commercially available or provided under a reasonable request. By ensuring that all aspects of this research are openly accessible, we invite the scientific community to replicate our findings and build upon this work, fostering a collaborative and transparent scientific environment.
# E Dataset Generation Process
Below, we provide a detailed description of each of the parameters used for the dataset generation process, along with their specific configurations.
# Parameters
# • mindags:
– Definition: Minimum number of generated DAGs. This parameter ensures that at least the specified number of DAGs is generated in the dataset. – Constraint: Must be greater than 0.
# • maxdags:
– Definition: Maximum number of generated DAGs. This parameter sets an upper
limit on the number of DAGs to be included in the dataset. – Constraint: Must be greater than or equal to mindags.
# • noise:
– Definition: Represents the percentage of noise in the datasets. Noise here refers to the random perturbations added to the data. – Constraint: Must be a value in the range [0, 1].
# • owa (Open World):
– Definition: The open-world degree indicates how many of the consequences of an initial set of relevant facts, called support facts, are missing from the data set. In other workds it indicates the percentage of consequences missing in the dataset. This parameter simulates incomplete data scenarios by randomly omitting a portion of the data. – Constraint: Must be a value in the range [0, 1].
# • missing:
– Definition: Specifies the percentage of missing data in the dataset. – Constraint: Must be a value in the range [0, 1].
# • category:
– Definition: Determines the type of the rule to be generated. The categories include different structural patterns and combinations. – Values: * Chain * Rooted Directed Graph (DG) * Disjunctive Rooted DG * Mixed * All of them with recursion
# Illustrative Example: Family Knowledge
# Illustrative Example: Family Knowledge Base
To illustrate the dataset’s construction, consider a simple example representing a small knowledge base about familial relationships. Here, the fact parent(john, mary). denotes that John
is the parent of Mary. A corresponding rule might be expressed as: ∀X, Y (parent(X, Y ) → ancestor(X, Y )). The initial dataset could be represented as follows:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/399c/399cfcf2-a642-4f84-ab9b-f6b929bad187.png" style="width: 50%;"></div>
% Rules ancestor(X,Y) :- parent(X,Y). ancestor(X,Y) :- parent(X,Z),ancestor(Z,Y).
Assume that 20% of the data is missing. The dataset would then be:
% Facts with 20% missing parent(john, mary). % parent(mary, susan). % This fact is missing parent(john, michael). parent(michael, robert).
% Rules ancestor(X,Y) :- parent(X,Y). ancestor(X,Y) :- parent(X,Z),ancestor(Z,Y).
Alternatively, if 20% of the data contains noise, the dataset might appear as follows:
% Facts with 20% noise parent(john, mary). parent(mary, susan). parent(john, michael). parent(michael, robert). %Below is a noisy fact parent(michael, alice).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/86c9/86c9940f-334f-4e8e-a207-d1eb8a511c11.png" style="width: 50%;"></div>
# % Rules
ancestor(X,Y) :- parent(X,Y). ancestor(X,Y) :- parent(X,Z),ancestor(Z,Y).
# ancestor(X,Y) :- parent(X,Y). ancestor(X,Y) :- parent(X,Z),ancestor(Z,Y).
This methodical approach to dataset generation allows us to simulate a wide range of real-world conditions, providing a robust foundation for analyzing the effects of noise, missing data, and structural variations on the performance of our experiments.
# F LLMs
F LLMs
Table 6 provides a summary of the main information about the models used in this study.
# G Models Output
The GPT-4o and GPT-3.5-turbo models have been demonstrated to consistently generate valid theories, thereby ensuring the successful execution of the Prolog code they produce. To illustrate, the following displays a theory induced by GPT-4o.
p10(A,B):-p8(A,B). p10(A,B):-p1(A,B). p10(A,B):-p7(A,B).
# p10(A,B):-p8(A,B). p10(A,B):-p1(A,B). p10(A,B):-p7(A,B).
Nevertheless, a recurrent pattern has been identified in the theories generated by GPT, namely the rewriting of rules in which the variable is interchanged. To illustrate, the following example is provided.
# p1(A, B) :- p2(A, B). p1(A, B) :- p2(B, A).
Furthermore, GPTs, particularly GPT-4o, are highly effective at identifying the relevant predicate for a given theory, disregarding the noise, the irrelevant facts added on purpose in the dataset. The following is an exemplar rule: p1(X0,X1) :- p2(X0,X1). p2(X0,X1) :- p4(X1,X2),p0(X0,X2). The initial prompt identifies the predicate p2 p1(A, B) :- p2(A, B). The refinement identifies the predicate p0 p1(A, B) :- p2(A, B). p1(A, B) :- p0(A, B). Subsequently, the predicate p4 is identified. p1(A, B) :- p2(A, B). p1(A, B) :- p0(A, B). p1(A, B) :- p4(A, B).
# Subsequently, the predicate p4 is identified.
p1(A, B) :- p2(A, B). p1(A, B) :- p0(A, B). p1(A, B) :- p4(A, B).
However, the same degree of precision could not be obtained from Llama3-8B, which may not consistently generate Prolog code that adheres to the necessary syntactical or logical constraints, potentially leading to errors during execution. To illustrate, consider the following Prolog theory:
eory :p(X, Y) :- p1(X, Y); p3(X, Y); p4(X, Y); p7(X, Y); p8(X, Y); p0(X, Y), not neg(p(X, Y)), (pos(p(X, Y)) - true; fail).
The models in question autonomously created the head of the clause "theory," as well as the predicates "p" and "pos," which should not exist. Additionally, Mixtral demonstrated satisfactory performance, although it exhibited a proclivity to insert the theory at the outset of the output. Although the output was generally valid, the quality of the generated theories was not as robust as that of GPT-4o, particularly in more intricate recursive scenarios such as RDG and DRDG. Additionally, it was also able to identify the relevant predicates, but their arrangement was not optimal. For example, in the same example as GPT-4o, the correct predicates were identified, but their arrangement was not optimal.
# p1(X,Y) :p0(X,Y), \+ p2(X,Y), \+ p4(X,Y).
Furthermore, this model produces an excessive number of rules, particularly in more intricate rule sets such as RDG, DRDG, and MIXED, both recursive and non-recursive. It also introduces pos or neg predicates that are erroneous and should not exist, in 20% of the results in these categories. The following example demonstrates a theory generated by Mixtral-8x7B with these issues:
theory :dif(X, Y), p3(X, Y), p5(Y, Y), p7(Y, _), \+ pos(p2(X, Y)), \+ neg(p2(X, Y)), asserta(pos(p2(X, Y))). theory :dif(X, Y), p4(X, Y), p5(Y, Y), p7(Y, _), \+ pos(p2(X, Y)), \+ neg(p2(X, Y)), asserta(pos(p2(X, Y))). theory :dif(X, Y),
theory :dif(X, Y), p3(X, Y), p5(Y, Y), p7(Y, _), \+ pos(p2(X, Y)), \+ neg(p2(X, Y)), asserta(pos(p2(X, Y))). theory :dif(X, Y), p4(X, Y), p5(Y, Y), p7(Y, _), \+ pos(p2(X, Y)), \+ neg(p2(X, Y)), asserta(pos(p2(X, Y))). theory :dif(X, Y),
Model
Maintainer
Parameters
Hidden dim.
# hidden layers
Context size
LLaMA3-8B Instr.
Meta-Llama
8B
4096
32
8K
Mixtral-8x7B Instr.
Mistral AI
46.7B (sparse)
4096
32
32K
Gemma-7B-IT
Google
7B
3072
28
8K
GPT-3.5 Turbo
OpenAI
–
–
–
16K
GPT-4o
OpenAI
–
–
–
128K
Table 6: Main Information about the models evaluated in this study. All models tested are auto-regressive decoderonly, with Mixtral-8x7B Instruct being a Sparse Mixture of Experts (SMoE). The original, non-quantised versions were used.
p6(X, Y), p5(Y, Y), p7(Y, _), \+ pos(p2(X, Y)), \+ neg(p2(X, Y)), asserta(pos(p2(X, Y))). theory :dif(X, Y), p10(X, Y), p5(Y, Y), p7(Y, _), \+ pos(p2(X, Y)), \+ neg(p2(X, Y)), asserta(pos(p2(X, Y))). theory :X \= Y, p5(X, X), \+ pos(p2(X, Y)), \+ neg(p2(X, Y)), asserta(pos(p2(X, X))).
Finally, Gemma 7B did not produce a valid theory. While Llama-3-8B does not entirely conform to the characteristics of a theory, it nonetheless approximates a theory to a certain degree. In comparison, Gemma 7B’s output lacked the elements necessary to be considered even a preliminary valid theory. The following is an example of a Gemma’s output. **Theory:**
The facts in the knowledge base indicate that the predicate p9(cX, cY) is true for the following pairs of facts:
<div style="text-align: center;">Hidden dim. # hidden layers Context size</div>
However, the predicate p9(c55, c48) is false.
