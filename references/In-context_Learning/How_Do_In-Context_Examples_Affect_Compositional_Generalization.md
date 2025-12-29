Shengnan An∗†, Zeqi Lin‡, Qiang Fu‡, Bei Chen‡, Nanning Zheng†, Jian-Guang LOU‡, Dongmei Zhang‡ † Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University ‡ Microsoft Corporation {an1006634493@stu, nnzheng@mail}.xjtu.edu.cn {Zeqi.Lin, qifu, beichen, jlou, dongmeiz}@microsoft.com
# Abstract
Compositional generalization—understanding unseen combinations of seen primitives—is an essential reasoning capability in human intelligence. The AI community mainly studies this capability by fine-tuning neural networks on lots of training samples, while it is still unclear whether and how in-context learning—the prevailing few-shot paradigm based on large language models—exhibits compositional generalization. In this paper, we present COFE, a test suite to investigate in-context compositional generalization. We find that the compositional generalization performance can be easily affected by the selection of in-context examples, thus raising the research question what the key factors are to make good in-context examples for compositional generalization. We study three potential factors: similarity, diversity and complexity. Our systematic experiments indicate that in-context examples should be structurally similar to the test case, diverse from each other, and individually simple. Furthermore, two strong limitations are observed: in-context compositional generalization on fictional words is much weaker than that on commonly used ones; it is still critical that the incontext examples should cover required linguistic structures, even though the backbone model has been pre-trained on large corpus. We hope our analysis would facilitate the understanding and utilization of in-context learning paradigm.
# 1 Introduction
Compositional generalization is an essential capability of human intelligence. It means to understanding and producing novel expressions by recombining known components in language (Chomsky, 1957; Montague, 1974; Fodor and Lepore, 2002). Taking examples in Figure 1, after learning the combination “baby in a room”, human intelligence can easily generalize to “Jackson in a room”. On exploring this human-like capability ∗Work done during an internship at Microsoft Research.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1be3/1be314d9-04a6-43f8-bee9-a503a9fdf596.png" style="width: 50%;"></div>
<div style="text-align: center;">In-Context Learning Paradigm</div>
Large Language Model
Sampled Sequence
Figure 1: Test compositional generalization under incontext learning. This case belongs to Phrase Recombination in COFE. The phrases modify the objects in examples but are recombined with subject in test input. in deep learning models, several benchmarks such as SCAN (Lake and Baroni, 2018), CFQ (Keysers et al., 2019) and COGS (Kim and Linzen, 2020) have been proposed based on semantic parsing1 tasks. In these benchmarks, the training set cover all the primitives while lacking certain combinations, and the test set focuses on these missing combinations. By fine-tuning generic neural models on these benchmarks, much work reported that these models exhibit poor compositional generalization (Furrer et al., 2020; Shaw et al., 2021; Bogin et al., 2022). Recently, in-context learning with large language models exhibits impressive performance on various tasks (Brown et al., 2020; Rae et al., 2021; Wei et al., 2022). By conditioning on few-shot incontext examples, the pre-trained language model, with extremely large model size and pre-trained
corpus, can perform downstream tasks without any update on pre-trained parameters.
corpus, can perform downstream tasks without any update on pre-trained parameters. Behind the impressive performance of in-context learning, we are curious whether this prevailing paradigm can take a step towards compositional generalization. To investigate this, we first take an initial exploration: for each test case in COGS, we select in-context examples from its training set and ensure that all primitives in each test case are covered by the equipped in-context examples. Our initial exploration suggests that compositional generalization can be easily affected by in-context examples: with only covering primitives, davinci 175B lags behind fine-tuned GPT2-Large with 24.2% accuracy (similar to the observation in Qiu et al. (2022)); with also covering some local structures (inspired by Bogin et al. (2022)), davinci outperforms fine-tuned GPT2-Large with 3.9% accuracy. Based on these initial observations, we raise and investigate the question: How do in-context examples affect compositional generalization? We construct the test suite COFE (based on COGS) to facilitate our systematic investigation. Taking the coverage of primitives as a basic principle in COFE, we further define and inject three factors in selecting in-context examples: similarity, diversity, and complexity. Similarity is considered as the matching of hidden structures behind concrete expressions. Diversity reflects whether the context presents repeated patterns or not. Complexity portrays the amount of information contained in each example. By controlling these factors in constructing COFE, we can systematically investigate how would in-context examples influence the performance on compositional generalization. Our experiments demonstrate that all three factors matter for in-context compositional generalization. We leverage six large language models in GPT series: davinci, code-cushman-001, codecushman-002, text-davinci-002, text-chat-davinci002, and code-davinci-002. The observations are consistent across models: to better perform compositional generalization, all backbone models prefer in-context examples with higher structural similarity to the test case, higher diversity among different examples, and lower complexity in each individual example. Furthermore, beyond the influence from these factors, in-context compositional generalization still faces two challenges. One is that in-context learning has difficulty recombining fictional words (e.g., random tokens) rather than com-
monly used ones. The other one is that in-context examples are still required to cover the linguistic structures in NL expressions, even though the backbone model has been pre-trained on large corpus. Our contributions are three-fold: 1) to answer the research question posed, we investigate three factors in selecting in-context examples and draw consistent conclusions across models; 2) we construct COFE to conduct our systematic investigation, and will release it to facilitate further exploration of in-context compositional generalization; 3) we also point out two remaining challenges that in-context learning still struggles to handle. We hope our analysis would provide insights on how to select proper in-context examples, and to shed light on the future research of in-context compositional generalization. COFE is publicly available at https://github.com/microsoft/Contextua lSP/tree/master/cofe.
# 2 In-Context Compositional Generalization
In-context compositional generalization refers to understand and produce novel combinations through recombining the building blocks presented by in-context examples. We first introduce some basic settings for testing this desired capability, then show our initial observations.
# 2.1 Principles for Measuring In-Context Compositional Generalization
To measure in-context compositional generalization under a test suite, each test case and its equipped in-context examples should satisfy two principles.
generalization on certain combinations, incontext examples should exclude these combinations while test cases contain them.
• Primitive coverage principle: the primitives contained in each test case should be fully covered by in-context examples. Primitives are the minimum indivisible units in expressions. In this work, we mainly consider primitives as lexical items (e.g., the noun “baby” and the verb “observed” in Figure 1).
We say that a model exhibits in-context compositional generalization if it performs well on a test suite that satisfies these two principles.
Category
In-Context Examples
Test Case
Illustration of Combination
Primitive
Substitution
Primitive
Structural Alternation
Phrase
Recombination
Longer
Chain
Deeper
Nesting
input:   shark
output: SHARK
input:   A girl drew the boy .
output: DRAW ( GIRL , BOY , NONE )
input:   The shark drew a boy .
output: DRAW ( SHARK , BOY , NONE )
𝐗𝑁
𝐗𝐿+
𝐗𝑆
1
input:   The goose baked .
output: BAKE ( GOOSE , NONE , NONE )
input:   A teacher noticed a chicken .
output: NOTICE ( TEACHER , CHICKEN , NONE )
input:   A teacher baked the chicken .
output: BAKE ( TEACHER , CHICKEN , NONE )
+
𝐗𝑆
1
𝐗𝑆
2
input:   Logan mailed Stella the cake in the pile .
output: MAIL ( LOGAN , IN ( CAKE , PILE ) , STELLA )
input:   The goose rolled a baby in a room .
output: ROLL ( GOOSE , IN ( BABY , ROOM ) , NONE )
input:   A visitor in the pile rolled a resident .
output: ROLL ( IN ( VISITOR , PILE ) , RESIDENT , NONE )
𝐗𝑆
1
+
𝐗𝑆
2
input:   The boy admired that Noah confessed that \
Emma was given a cookie .
output: ADMIRE ( BOY , NONE , NONE ) \
CCOMP CONFESS ( NOAH , NONE , NONE ) \
CCOMP GIVE ( NONE , COOKIE , EMMA )
input:   The girl wished that a crocodile declared that \
the boy admired that Emma liked that \
Evelyn was passed a drink .
output: WISH ( GIRL , NONE , NONE ) \
CCOMP DECLARE ( CROCODILE , NONE , NONE ) \
CCOMP ADMIRE ( BOY , NONE , NONE ) \
CCOMP LIKE ( EMMA , NONE , NONE ) \
CCOMP PASS ( NONE , DRINK , EVELYN )
𝐘𝑆
𝑛
𝑛times
recursion
input: Noah appreciated a girl in a house \
beside the chair .
output: APPRECIATE ( NOAH , \
IN ( GIRL , \
BESIDE ( HOUSE , CHAIR\
) ) , NONE )
input:   A dog painted the girl beside the chair \
in a house beside a road on a dish .
output: PAINT ( DOG , \
BESIDE ( GIRL , \
IN ( CHAIR , \
BESIDE ( HOUSE , \
ON ( ROAD , DISH\
) ) ) ) , NONE )
𝐘𝑆
𝑛
𝑛times
recursion
𝐗𝐿
𝐗𝑁
<div style="text-align: center;">Figure 2: Five categories of aiming combinations. The key parts in combinations are marked with underlines and colors (blue in NL-side and purple in code-side). The last column follows the notations defined in Section 3.2.</div>
# 2.2 COGS (Under In-Context Learning)
COGS is a compositional generalization benchmark designed for the fine-tuning paradigm: based on a semantic parsing task, the training set of COGS covers all primitives in this task, while several combinations of primitives in the test set are excluded from the training set. We term these excluded combinations as aiming combinations. We measure in-context compositional generalization based on COGS, by converting it from the original fine-tuning paradigm to the in-context learning paradigm. For each COGS test case, we select in-context examples from the training set B, ensuring that the two principles are satisfied. Note that, for each test case, there are usually different collections of in-context examples satisfying the two principles. Our basic setting is to use a random one among them, and we show that this casual strategy could lead to an underestimation of in-context compositional generalization (Section 2.3). To facilitate testing on more complex logical forms, we reconstruct some target-side clauses from the chain structure into the nested-function format (illustrated in Figure 2). This reconstruction follows An et al. (2023) and is similar to the conversion from Lambda calculus to FunQL in Geo domain(Zelle and Mooney, 1996; Kate et al., 2005; Zettlemoyer and Collins, 2012). Moreover, to improve human readability, we omitted two types of details: the special marker for definite descriptions and the Skolem constants. These details do not affect the testing of compositional generalization.
Apart from these omitted details, the logical forms in COFE unambiguously represent the main semantics in the domain of COGS, such as semantic roles, modifications, and orders among clauses and modifications. More details about COFE logical forms are contained in Appendix A. Categories of aiming combinations. The aiming combinations in COGS can be divided into five categories, of which two are low-level combinations (i.e., focusing on specific primitives) and three are high-level combinations (i.e., focusing on high-level structures), illustrated in Figure 2. • Primitive Substitution (PrimSubs): Compose a primitive (e.g., “shark”) with a grammatical role (e.g., “subject”). • Primitive Structural Alternation (PrimAlte): Compose a primitive (e.g., “baked”) with a sentence structure (e.g., “subj. verb obj.”). • Phrase Recombination (PhraReco): Compose a prepositional phrase (e.g., “A in B”) with a grammatical role (e.g., “subject”). • Longer Chain (LongChain): Extend the tail of the logical form with CCOMP clauses ∈Y1 S. The max recursive times of CCOMP clauses in B is 2, while in test case it is 12. • Deeper Nesting (DeepNest): Expand the arguments in functions with IN/ON/BESIDE clauses ∈Y1 S. The max recursive times in B and test cases are the same with LongChain. Note that PrimSubs and PrimAlte are low-level combinations while others are high-level ones.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/01f3/01f36e94-7005-46b7-bbd5-1601511787ce.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Initial observations on PrimSubs: casual selection leads to low performance while adding preference brings considerable gains.</div>
# 2.3 In-Context Learning vs Fine-Tuning
Compositional generalization under the fine-tuning paradigm has been widely studied (Furrer et al., 2020; Shaw et al., 2021; Bogin et al., 2022), while there is little observation under in-context learning. To first get a general sense about in-context compositional generalization, we conduct an initial exploration to compare with a fine-tuning baseline. Models and setups. We test in-context compositional generalization with six large models in GPT series: davinci, code-cushman-001 (cuchman001), code-cushman-002 (cuchman002), text-davinci002 (text002), text-chat-davinci-002 (chat002), and code-davinci-002 (code002). The sampling temperature is 0 (i.e., greedy decoding), and the max decoding length is 500. The reported metric is exact-match accuracy. To set a fine-tuning baseline, we take GPT2-Large with 0.7B parameters. We fine-tune it on the whole B and test without in-context examples. We set learning rate as 1e-5 and batch size as 8 during fine-tuning, and set beam size as 5 for inference. Appendix B includes more details.
# Casual selection leads to low performance of in-context compositional generalization. For
selecting in-context examples, we first take a casual selection: while satisfying the primitive coverage principle, we randomly select 10 examples without other preference. We conduct initial exploration on PrimSubs category. Figure 3 shows that under the casual selection, all six models lag behind the fine-tuned GPT2-Large on PrimSubs. In particular, although the size of davinci is more than 200 times that of GPT2-Large, there is a 24.2% accuracy gap between davinci and the fine-tuned GPT2-Large. These observations are close to Qiu et al. (2022). However, we suppose the potential of in-context learning is still not fully revealed. Specifically, the selection of in-context examples does not yet take full advantage of available examples in B. In next try, while still following the primitive coverage
principle, we consider injecting some additional preference in the selection of in-context examples.
Preference in selection could bring huge improvement on PrimSubs. Inspired by Bogin et al. (2022) that suggests the influence of unobserved local structures, we consider to prioritize examples that have similar hidden structures to the test case. Figure 3 shows that with this preference in selection, results on PrimSubs hugely change: davinci now outperforms the fine-tuned GPT2-Large; codedavinci-002 even performs near-perfectly. These changes strongly suggest that the selection of incontext examples can significantly affect in-context compositional generalization. Based on these initial results, to further reveal the potential of in-context learning, we perform in-depth investigations on how the selection of incontext examples affects compositional generalization.
# 3 Factors Under In-Context Examples
To facilitate our systematic investigation, we construct COFE (COmpositional generalization with FEw-shot examples), which is derived from COGS. For selecting in-context examples in constructing COFE, we identify, inject, and control three potential factors: similarity, diversity, and complexity.
# 3.1 Conceptual Definitions
We first give conceptual definitions of our considered factors and discuss our intuitions behind them.
Similarity has been widely considered as the main factor in selecting in-context examples (Liu et al., 2022; Shin et al., 2021; Rubin et al., 2021; Poesia et al., 2021). The primitive coverage principle can be regarded as a basic lexical similarity on the surface of expressions. Beyond this surface similarity, we consider that the structural similarity hidden behind expressions could be a beneficial factor. From the view of syntactic structure, the recombination of primitives is equivalent to the reconstruction of the parse tree. Similar structures would ease the difficulty of recombination because the model does not need to completely reconstruct the entire structure of in-context examples. Moreover, some work has suggested that the challenge of compositional generalization under fine-tuning lies in unobserved structures (Keysers et al., 2019; Shaw et al., 2021; Bogin et al., 2022).
Diversity concerns the repetitiveness among incontext examples. It portrays the property among in-context examples. Specifically, the context is under low diversity if it contains many repeating patterns among in-context examples, otherwise it is under high diversity. Under in-context learning, the low diversity can easily lead to biased observations on the full task space, as there are only few examples for the model to learn. Thus, we suppose that the low diversity among examples could block in-context compositional generalization. Moreover, some work also demonstrated that the diversity in training data could affect compositional generalization under fine-tuning (Oren et al., 2021). Complexity reflects the amount of information contained in each individual in-context example. The higher complexity means that the example could provide more information to the model, but these information could be redundant. In addition, the difficulty in directly learning from complex examples has been flagged at the intersection of cognitive science and machine learning (Elman, 1993; Bengio et al., 2009). Such difficulty may be more severe for in-context learning, since the parameters of the model cannot be updated to fit these complex examples. Thus, we suppose that too high complexity might hinder performance.
# 3.2 Incorporate Three Factors Into Test Suite
To inject these factors in selecting in-context examples, we design a matching score based on the parse trees behind concrete expressions. Formally, considering the primitive coverage, structural similarity, diversity and complexity, the matching score of two parse trees T and T′ is defined as follows,
Match(T, T′) =wp · |P(T) ∩P(T′)|+ ws · |S(T) ∩ � S(T′)−S(C) � |− wc · depth(T′),
(1)
in which P(·) contains primitives, S(·) contains partial structures (defined later), C contains already selected examples, S(T′) −S(C) means to exclude already covered parts in S(C) from S(T′), and depth(·) reflects the complexity of the tree. The meaning of three factors in Equation 1 is that: the structural similarity means covering S(T), the high diversity means to avoid repeatedly covering the same element in S(T), and the low complexity is to prioritize low-depth structures.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/eb4d/eb4df494-3704-4995-ba15-08f648b62071.png" style="width: 50%;"></div>
# Based on this matching score, the overall ranking score between the test case (X, Y) and a candidate (Xc, Yc) is calculated as follows,
Based on this matching score, the overall ranking score between the test case (X, Y) and a candidate (Xc, Yc) is calculated as follows,
scorec = Match(X, Xc) + Match(Y, Yc), (2)
in which both the matching of source side (i.e., NL expressions) and target side (i.e., logical forms) are considered. Poesia et al. (2021) has demonstrated the importance of target-side similarity in semantic parsing and code generation tasks, and this work will further investigates the necessity of source-side matching. In the following, we will give a more detailed description of notations in Equation 1. Detailed description: Figure 4 shows an illustration of notations. Considering an expression e with the parse tree T, TL represents leaf nodes (e.g., “Jackson”) and TN contains internal nodes (e.g., “subject”). T1 S contains one-depth sub-structures in T. Each T1 s ∈T1 S (e.g., ①in Figure 4) contains one parent node (e.g., “root”) and a set of child nodes (e.g., “subject”, “verb” and “object”). T>1 S contains deeper sub-structures that are composed from several one-depth sub-structures in T1 S (e.g., ①+②+④in Figure 4). In Equation 1, the primitives P(T) = TL, and the partial structures S(T) = T1 S ∪T>1 S . Note that aiming combinations ⊂S(T). Appendix E includes more details.
detailed description of notations in Equation 1. Detailed description: Figure 4 shows an illustration of notations. Considering an expression e with the parse tree T, TL represents leaf nodes (e.g., “Jackson”) and TN contains internal nodes (e.g., “subject”). T1 S contains one-depth sub-structures in T. Each T1 s ∈T1 S (e.g., ①in Figure 4) contains one parent node (e.g., “root”) and a set of child nodes (e.g., “subject”, “verb” and “object”). T>1 S contains deeper sub-structures that are composed from several one-depth sub-structures in T1 S (e.g., ①+②+④in Figure 4). In Equation 1, the primitives P(T) = TL, and the partial structures S(T) = T1 S ∪T>1 S . Note that aiming combinations ⊂S(T). Appendix E includes more details.
# 4 Experiments and Analysis 4.1 Experimental Settings and Hyper-Parameters
# 4.1 Experimental Settings and Hyper-Parameters
We take a greedy-search algorithm to sequentially select 10 examples for each test case. Models and setups follow our initial explorations in Section 2.3. For the investigation of each factor, hyperparameters in Equation 1 are set as follows2.
<div style="text-align: center;">Table 1: Results with (and without) structural similarity. Grey boxes mark the significantly better performan compared to the fine-tuned GPT2-Large.</div>
Model
Setting
PrimSubs
PrimAlte
PhraReco
LongChain
DeepNest
Avg. Acc
code-davinci-002
Primitive Coverage
92.2
77.1
60.8
62.1
12.3
60.9
+ Structural Similarity
99.8
99.7
65.3
87.0
26.0
75.6
text-chat-davinci-002
Primitive Coverage
92.2
75.4
47.0
65.0
6.3
57.2
+ Structural Similarity
99.5
99.3
53.4
87.7
18.9
71.8
text-davinci-002
Primitive Coverage
88.5
66.4
38.7
46.5
2.9
48.6
+ Structural Similarity
99.7
99.4
39.4
80.2
12.7
66.3
code-cushman-002
Primitive Coverage
82.6
55.6
21.3
29.3
5.0
38.8
+ Structural Similarity
98.9
99.0
28.5
64.0
15.1
61.1
code-cushman-001
Primitive Coverage
76.6
60.7
16.9
5.0
1.0
32.0
+ Structural Similarity
99.1
98.4
20.7
11.1
8.9
47.6
davinci
Primitive Coverage
69.4
52.3
9.4
2.3
0.2
26.7
+ Structural Similarity
97.5
95.4
12.3
13.4
1.4
44.0
Fine-Tuning Baseline
-
93.6
97.9
14.0
5.4
0.0
42.2
In all settings, we prioritize the matching of primitives (i.e., |P(T) ∩P(T′)| in Equation 1) since the primitive coverage principle should be firstly satisfied. Concretely, we set wp = 100 and ensure wp ≫ws and wc in all settings. For investigating structural similarity3, we set ws = 1 and wc = 0, and exclude S(C) term. For investigating the effect of higher diversity, we add the S(C) term and keep other settings. For complexity, we set |wc|·max(depth(T′)) < ws, such that the of preference of complexity will not influence the priority of structural similarity. Concretely, as max(depth(T′)) = 12 in COFE, we set wc = 0.01 for the low-complexity experiments and wc = −0.01 for the high-complexity experiments, and exclude S(C) term. Some basic statistics for COFE under full similarity setting are listed in Table 2, and Appendix C.5 contains statistics under other settings. These statics show that the primitive coverage principle is well satisfied, since the cover rates of TL are almost 100%. Note that the coverage on T1 S ∪T>1 S must be lower than 100% since the aiming combination must be excluded.
Structural similarity brings significant gains. Table 1 shows the performance with structural similarity. Compared to the results without structural similarity (i.e., only with the coverage on primitives), there are considerable gains on all five categories and across all six models. These gains clearly demonstrate that beyond primitive coverage, the structural similarity under in-context examples
# are essential for compositional generalization.
More precise structural similarity brings larger gains. As mentioned in Section 3.2, the structural similarity considers to match S(T) which contains two parts, T1 S and T>1 S . Specifically, we regard that T1 S describes the rough structure of T, and T>1 S determines a more precise structure. Based on the results in Table 1, we are curious about whether a rough structural similarity is enough. To verify this, we remove T>1 S from S(T), which means that now we do not restrict the selected incontext examples to match precise structures in test cases. Figure 5 shows that the performances on four categories significantly drop with only a rough structural similarity, indicating that matching the precise structure of test case is still required for in-context examples. The only exception lies in PhraReco. It suggests that similarity is not the only influential factor for in-context compositional generalization. In Section 4.3, we will show that the low diversity and high complexity potentially cause this exception.
tions are almost solved while high-level combinations still have large room for improvement. Specifically, for code-davinci-002, which exhibits the best performance among all backbone models, it performs near-perfectly on low-level combinations (i.e., PrimSubs and PrimAlte) while still does not achieve >95% accuracy on high-level combinations (i.e., PhraReco, LongChain and DeepNest). Although in-context learning greatly exceeds the fine-tuning baseline on high-level combinations, we suppose there is still potential for improvement. Compared to low-level combinations, han-
Statistics
Number of Instances
Average Coverage
Average Length
TL
TN
T1
S
T>1
S
Context
Case Input
Case Output
Test Cases
4,785
99.7%
100%
88.9%
49.3%
297.7
17.8
33.7
- PrimSubs
1,100
100%
100%
79.8%
45.1%
236.7
7.1
11.5
- PrimAlte
700
100%
100%
96.6%
59.7%
269.4
7.9
13.8
- PhraReco
1,000
100%
100%
84.4%
19.8%
254.0
10.7
16.9
- LongChain
1,000
99.8%
100%
97.8%
76.7%
370.6
32.4
76.7
- DeepNest
985
98.8%
100%
89.0%
48.6%
356.4
29.0
46.3
Example Bank
24,155
-
-
-
-
-
7.5
10.5
dling high-level ones requires more creation than imitation, thus just considering similarity for incontext examples is not enough. In the following, we will further investigate these high-level combinations from the view of diversity and complexity.
# 4.3 Diversity and Complexity
High diversity brings considerable gains on PhraReco. Figure 6 shows how diversity among in-context examples affects generalization on highlevel combinations. It shows that increasing the diversity could bring considerable gains in PhraReco, while not affecting the other two categories. For the performance on PhraReco, the improvements from higher diversity are in line with our speculations in Section 3.1, that low diversity leads to biased observations, thus blocking high-level structural generalization. For LongChain and DeepNest, beyond biased structures, their difficulty also lies in length generalization, thus just increasing structural diversity brings less effect to them.
# Low complexity brings considerable gains on
PhraReco. Figure 7 shows how the complexity in each individual example affects generalization on high-level combinations. For PhraReco, there are ∼10% gains in accuracy when the high complexity setting is changed to low complexity setting. We suppose the reason behind this gain is that simple examples could reduce the learning difficulty for the model. Moreover, simple examples also contain less redundant information thus would not confuse the model4. For LongChain and DeepNest, there is still less change on performance. Note that the max depth in these two categories is 13 while the max depth in the whole example bank is only 3. Therefore, changing the complexity of in-context examples would bring negligible influence for test cases in LongChain and DeepNest.
Table 3: Results under different prompt orders (full similarity setting). ∆represents the max difference in performance for each model.
Model
Structure Closer Atom Closer Random Order
∆
code-davinci-002
75.6
74.2
74.5
1.4
text-davinci-002
66.3
66.0
66.3
0.3
code-cushman-002
61.1
60.0
60.1
1.1
code-cushman-001
47.6
48.2
47.3
0.9
davinci
44.0
43.6
42.5
1.5
# 4.4 Analysis: Robustness to Prompt Order
Some previous work on in-context learning showed that the order of exemplars in prompt could sometimes hugely influences the performance of LLMs (Zhao et al., 2021; Lu et al., 2022). Here, we examine whether our observations above are sensitive to the prompt order. Based on the full similarity setting (Section 4.2), we consider three different strategies for ordering exemplars: 1) random order; 2) atom closer: exemplars with higher coverage on atomic blocks are placed closer to the test input; 3) structure closer (default): examples with higher similarity on linguistic structures are placed closer to the test input. Implementations of different strategies for prompt order are detailed in Appendix C.3. Results in Table 3 show that the performance only slightly changes under different prompt orders. These results indicate that the main results revealed by COFE is consistent and reliable. It also indicates that in-context learning could be less sensitive to the prompt order when the in-context examples are chosen properly.
# 4.5 Discussion: Difficulty in DeepNest
Among all five categories, in-context learning performs worst on DeepNest. Compared to LongChain which also test recursive structures, the results on DeepNest still lag far behind. There is an interesting observation from the study of error cases (such as Figure 10): in-context learning frequently makes word-level mistakes, while the overall nested struc-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1ee4/1ee45460-a521-45c0-83a4-ac48b088b47a.png" style="width: 50%;"></div>
<div style="text-align: center;">ure 5: Performance of code-davinci-002 and text-davinci-002 with different levels of structural similarity.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c97e/c97e47c0-cdc2-47fb-a55d-ba244b67f779.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Performance under different diversity settings (on high-level combinations).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1244/1244aa16-25ad-4503-99fe-9aa91624e153.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Performance under different complexity settings (on high-level combinations).</div>
ture in the prediction is close to the ground truth. It suggests that the performance bottleneck in DeepNest is to correctly fill the details in the complex structure, rather than generating the sketch of the structure. Appendix F.1 provides further analysis.
# 5 Remaining Challenges
Our investigation has revealed a huge potential of in-context learning on performing compositional generalization5. Despite this potential, for achieving the ideal in-context compositional generalization, there remains the following two challenges.
Our investigation has revealed a huge potential of in-context learning on performing compositional generalization5. Despite this potential, for achieving the ideal in-context compositional generalization, there remains the following two challenges. In-context examples are still required to match linguistic structures in NL expressions. Since all backbone models have been pre-trained on large natural language corpus, we expect that these models could already handle the high variety in NL expressions without further hints from in-context examples. Motivated by this, we conduct experiments on another variant of COFE: the source-side term Match(X, Xc) is removed from Equation 2, and the coverage of S(X) is limited (detailed in Appendix C.6). Figure 8 shows that on all five categories, the performance consistently drops if in5Appendix G shows the results of assembling factors.
In-context examples are still required to match linguistic structures in NL expressions. Since all backbone models have been pre-trained on large natural language corpus, we expect that these models could already handle the high variety in NL expressions without further hints from in-context examples. Motivated by this, we conduct experiments on another variant of COFE: the source-side term Match(X, Xc) is removed from Equation 2, and the coverage of S(X) is limited (detailed in Appendix C.6). Figure 8 shows that on all five categories, the performance consistently drops if in5Appendix G shows the results of assembling factors.
context examples do not match the NL-side structure. It suggests that even having been pre-trained on large corpus, in-context learning still struggles to effectively recognize the semantic equivalence among different linguistic structures behind NL expressions (detailed in Appendix F.3). In-context learning has difficulty leveraging fictional words6. The ideal compositional generalization requires that the recombination of primitives should be independent of the surface form in primitives. In COFE, we set the target-side primitives as the uppercase of source-side ones (e.g., “cat”→“CAT”). Such case conversion is commonly used in semantic parsing tasks. To test whether in-context learning could use fictional words, we replace each target-side word with random characters (e.g., replace “CAT” with “MXR”, detailed in Appendix C.7). Figure 9 shows the huge drops after changing words. Moreover, we investigate the structural accuracy by only keeping the structural terminals (e.g., parentheses and commas) in predictions. Figure 9 shows that the structural accuracy is also affected by fictional words. It indicates that on performing in-context compositional generalization, the prediction of structural sketch is not decoupled with word-level patterns.
# 6 Related Work
Compositional generalization (CG) has attracted much attention in NLP field. Most existing benchmarks measured CG under fine-tuning with synthetic semantic parsing tasks, suggesting the limitations of general-purpose neural networks (Lake and Baroni, 2018; Keysers et al., 2019; Kim and Linzen, 2020). Many approaches were proposed to enhance the CG on general-purpose models (Andreas, 2020; Akyürek et al., 2020; Guo et al., 2021; Oren et al., 2021; Shaw et al., 2021; Zhu et al., 2021) or design task-specific methods (Liu et al., 2020; Herzig
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6fb1/6fb158e9-ec71-4c26-ab77-5d12d68bf57a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: Performance with or without matching linguistic structures in NL expressions.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0166/01661f03-3391-435e-8f3d-15454dc72bbd.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: Average exact-match accuracy and structural accuracy with fictional words.</div>
and Berant, 2021; Chen et al., 2020; Liu et al., 2021). Some influential factors that affect CG have been revealed, such as the length bias (Csordás et al., 2021), target-side format (Furrer et al., 2020; Herzig et al., 2021) and local structures (Bogin et al., 2022). Most existing work explored CG under the fine-tuning paradigm, while our work advances the exploration under the in-context learning paradigm.
In-context learning (ICL) along with large language models (LLMs) has shown surprising performance in many NLP tasks (Brown et al., 2020; Hendrycks et al., 2020; Patel and Pavlick, 2021; Rae et al., 2021; Zhang et al., 2022a; Hoffmann et al., 2022; Srivastava et al., 2022; Chowdhery et al., 2022; Smith et al., 2022; Wei et al., 2022). Most related to our work, Qiu et al. (2022) and Drozdov et al. (2022) also explored ICL on CG challenges. Qiu et al. (2022) utilized the targetside similarity on structural fragments and reported that LLMs still exhibited much poorer CG than finetuned small models on COGS, which is close to our initial observations. Drozdov et al. (2022) designed task-specific inference pipelines for performing CG under a least-to-most manner. Our work provides more general understandings on how to improve CG performance by revealing several factors in selecting in-context examples. In addition, some more recent work has similar observations on the potential of LLMs on CG (Hosseini et al., 2022), gains from diversity (Levy et al., 2022), and challenges under fictional words (Kim et al., 2022) Selection of in-context examples is an essential
Figure 10: An error case in DeepNest (full similarity setting) with wrong local words and redundant parts.
part for the utilization of ICL. Most existing work considered the similarity as the major metric during selection. Liu et al. (2022) selected k-nearest neighbors with similar sentence embeddings; Shin et al. (2021) regarded the conditional probability from a pre-trained LLM as the similarity score; Rubin et al. (2021) and Zhang et al. (2022b) separately trained a retriever to score the similarity; Poesia et al. (2021) and Madaan et al. (2022) estimated the target-side similarity. This work demonstrates the necessity of structural similarity in achieving CG, and also reveals the importance of two other factors beyond similarity, i.e., diversity and complexity.
# 7 Conclusion and Future Work
This work investigates how in-context compositional generalization is affected by the selection of examples. The test suite COFE is constructed to study three factors. Experiments show the effects of structural similarity, higher diversity and lower complexity. Two challenges under in-context compositional generalization are further revealed. To apply our revealed factors outside the COFE test suite, one main challenge for future work is to determine the hidden structures behind expressions without knowing the exact generative grammar. Here, we consider two potential approaches. One is to use a pre-trained parser to generate a parse tree for the input query and then measure tree similarity. The other approach is to pre-train an embedding model with a structure-aware training objective and then compute embedding similarity.
# Limitations
GPU resources. This work utilizes extremely large language models and thus has a high cost on GPU resources. Concretely, experiments are conducted on the 8 x NVIDIA A100 GPU station. The maximum inference time on each version of COFE (containing 4,785 test cases) is ∼8 hours. The maximum estimation of costed computing resources in this study is ∼500 x 8 GPU hours.
Synthetic data. As in most previous work on compositional generalization (Lake and Baroni, 2018; Keysers et al., 2019; Kim and Linzen, 2020), the COFE dataset is constructed using synthetic data rather than natural one. The source-side sentences in COFE are from COGS, which account for 70–80% of naturally-occurring English sentences (Kim and Linzen, 2020; Roland et al., 2007). Thus, this synthetic test suite could be close to the real-world application scenarios.
Single run. Due to the high cost on computing resources, we do not take multiple runs with different sets of examples, nor did we take multiple samples with temperature > 0. Observations under different prompt orders (in Appendix 4.4) imply that with desired factors in selecting in-context examples, there could be low variance in experiments.
# Ethics Statement
Due to the utilization of pre-trained language models, this work could be exposed to some potential risks of ethical issues on general deep learning models (such as social bias and privacy breaches). As explored in this work that the model behavior can be hugely influenced by the provided context, we call for further investigation into how ethical issues can be avoided by controlling the provided context.
# Acknowledgments
We thank all the anonymous reviewers for their valuable comments. Shengnan An and Nanning Zheng were supported in part by NSFC under grant No. 62088102.
# References
Ben Bogin, Shivanshu Gupta, and Jonathan Berant. 2022. Unobserved local structures make compositional generalization hard. arXiv preprint arXiv:2201.05899.
Noam Chomsky. 1957. Syntactic structures (the hague: Mouton, 1957). Review of Verbal Behavior by BF Skinner, Language, 35:26–58.
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.
Róbert Csordás, Kazuki Irie, and Juergen Schmidhuber. 2021. The devil is in the detail: Simple tricks improve systematic generalization of transformers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 619–634. Andrew Drozdov, Nathanael Schärli, Ekin
Zheng, and Dongmei Zhang. 2020. Compositional generalization by learning analytical expressions. Advances in Neural Information Processing Systems, 33:11416–11427.
Aman Madaan, Shuyan Zhou, Uri Alon, Yiming Yang, and Graham Neubig. 2022. Language models of code are few-shot commonsense learners. arXiv preprint arXiv:2210.07128.
R Montague. 1974. English as a formal language. Formal Philosophy: Selected Papers of Richard Montague.
Inbar Oren, Jonathan Herzig, and Jonathan Berant. 2021. Finding needles in a haystack: Sampling structurally-diverse training sets from synthetic data for compositional generalization. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10793–10809.
Roma Patel and Ellie Pavlick. 2021. Mapping language models to grounded conceptual spaces. In International Conference on Learning Representations.
Gabriel Poesia, Alex Polozov, Vu Le, Ashish Tiwari, Gustavo Soares, Christopher Meek, and Sumit Gulwani. 2021. Synchromesh: Reliable code generation from pre-trained language models. In International Conference on Learning Representations.
Linlu Qiu, Peter Shaw, Panupong Pasupat, Tianze Shi, Jonathan Herzig, Emily Pitler, Fei Sha, and Kristina Toutanova. 2022. Evaluating the impact of model scale for compositional generalization in semantic parsing. arXiv preprint arXiv:2205.12253.
ack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. 2021. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446.
Douglas Roland, Frederic Dick, and Jeffrey L Elman. 2007. Frequency of basic english grammatical structures: A corpus analysis. Journal of memory and language, 57(3):348–379.
Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2021. Learning to retrieve prompts for in-context learning. arXiv preprint arXiv:2112.08633.
Peter Shaw, Ming-Wei Chang, Panupong Pasupat, and Kristina Toutanova. 2021. Compositional generalization and natural language variation: Can a semantic parsing approach handle both? In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 922–938.
Richard Shin, Christopher Lin, Sam Thomson, Charles Chen Jr, Subhro Roy, Emmanouil Antonios Platanios, Adam Pauls, Dan Klein, Jason Eisner, and Benjamin Van Durme. 2021. Constrained language models yield few-shot semantic parsers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7699–7715.
Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. 2022. Using deepspeed and megatron to train megatronturing nlg 530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990.
Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615.
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682.
John M Zelle and Raymond J Mooney. 1996. Learning to parse database queries using inductive
Luke S Zettlemoyer and Michael Collins. 2012. Learning to map sentences to logical form: Structured classification with probabilistic categorial grammars. arXiv preprint arXiv:1207.1420.
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022a. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.
Yiming Zhang, Shi Feng, and Chenhao Tan. 2022b. Active example selection for in-context learning. arXiv preprint arXiv:2211.04486.
This is the Appendix of the paper: How Do In-Context Examples Affect Compositional Generalization?
# A Grammar
Part of the grammar used in constructing COFE is listed in Table 4. Note that the max recursive times of R-Production Rules is 2 in prompting examples and 12 in test cases. The target-side grammar follows the reconstruction in An et al. (2023). Overall, the original target grammar of COGS is reconstructed to be chain-structured. Concretely, first, the original output tokens in COGS are capitalized; then, the variables (e.g., “x_1”) in the original grammar are aligned and replaced with their corresponding terminals; finally, the output clauses are grouped as the function format, in which the function name belongs to “PRED-FUNC” and the arguments are ordered as “AGENT”, “THEME”, and “RECIPIENT”. Moreover, if “PRED-FUNC” does not contain one or some arguments, the positions of these arguments are filled with “NONE” terminal. For the two R-Production rules in Table 4, the first is in chain structure and the second is in nested structure. Moreover, the whole nested “PPFUNC” will be filled into the “PRED-FUNC” as an argument, rather than concatenated to the tail of the “CLAUSE”.
# B Details of Fine-Tuning
The fine-tuned GPT2-Large contains 762M parameters. For fine-tuning, we take 50,000 training steps with 8 batch size and 1e-5 learning rate (without warm-up strategy). We set weight decay as 1e-2 and label smoothing factor as 1e-1. For inference with GPT2-Large, we set beam size as 5 and set max length as 1,024.
# C Details of Implementation
# C.1 Algorithm
Algorithm 1 shows the greedy searching algorithm for constructing COFE.
# C.2 Key Designs
We give detailed descriptions of some key designs in Algorithm 1.
• P(T): Return the leaf nodes TL on the tree; • S(T): Return the structural combinations on the tree, i.e., T1 S ∪T>1 S ;
structing COFE
Given:
(X, Y): Source and target parse trees in one test case;
B: Example bank;
(Xi, Yi) ∈B: One candidate case in example bank;
XA and YA: Aiming combination;
wp, ws, wc: Weights for primitive coverage, structural
similarity, and complexity penalty;
P(·): primitives;
S(·): structural combinations;
Return:
C: Selected in-context examples;
1: C = {}
2: while |C| < n do
3:
max_score = 0
4:
candidate = None
5:
for (Xi, Yi) ∈B do
6:
Assert XA /∈S(Xi)
7:
Assert YA /∈S(Yi)
8:
prim_score = 0
9:
stru_score = 0
10:
for element ∈P(Xi) ∪P(Yi) do
11:
if element ∈P(X) ∪P(Y) then
12:
prim_score += wp
13:
end if
14:
end for
15:
for element ∈S(Xi) ∪S(Yi) do
16:
if element ∈S(X) ∪S(Y) and element /∈S(C)
then
17:
stru_score += ws
18:
end if
19:
end for
comp_penalty = wp · depth(Xi)
score = prim_score + stru_score - comp_penalty
20:
if score > max_score then
21:
max_score = score
22:
candidate = (Xi, Yi)
23:
end if
24:
end for
25:
C.add(candidate)
26: end while
• depth(T): return the depth of the tree. Note that depth(Xi) = depth(Yi) in COFE.
# C.3 Prompt Order
We take the structure-closer order, i.e., the examples in C with a higher stru_score are placed closer to the test case. In Section 4.4, we show the robustness to the other two orders: random order, i.e., all selected in-context examples in C are randomly
<div style="text-align: center;">Table 4: Part of the grammar used in constructing COFE.</div>
Formal English Grammar
Semantic Representation
Type
active-verb / passive-verb ↠Sv
PRED-FUNC ↠SP
T-Production Rule
subject / direct-object / indirect-object ↠Sn
AGENT / THEME / RECIPIENT ↠SE
pp-mod / pp-s ↠Sn
PP-FUNC / PP-S ↠SE
conj ↠that
CP-CONCAT ↠CCOMP
prep ↠in / on / beside
PP-CONCAT ↠IN / ON / BESIDE
sentence ↠subj active-verb
CLAUSE ↠PRED-FUNC ( AGENT, NONE, NONE )
N-Production Rule
sentence ↠subj active-verb direct-obj indirect-obj
CLAUSE ↠PRED-FUNC ( AGENT, THEME, RECIPIENT )
subject / direct-object / indirect-object ↠pp-mod
AGENT / THEME / RECIPIENT ↠PP-FUNC
sentence ↠sentence conj sentence
CLAUSE ↠CLAUSE CP-CONCAT CLAUSE
R-Production Rule
pp-mod ↠pp-s prep pp-mod
PP-FUNC ↠PP-CONCAT ( PP-S, PP-FUNC )
shuffled, and atom-closer order, i.e., the examples in C with a higher prim_score are placed closer to the test case.
# C.4 Max Depth in T>1 S
Since the max repetition times for LongChain and DeepNest are 2 (as described in Section 2.2), we set the max depth in T>1 S as 2 in S(T).
# C.5 Similarity Under Diversity and Complexity Settings
<div style="text-align: center;">Table 5: Statistics of different versions of COFE (PhraReco category).</div>
Table 5: Statistics of different versions of COFE (PhraReco category).
Setting
Average Coverage
TL
TN
T1
S
T>1
S
Default (Low Diversity, Mid Complexity)
100%
100%
84.4%
19.8%
High Diversity
100%
100%
84.4%
19.8%
Low Complexity
100%
100%
84.4%
19.8%
High Complexity
100%
100%
84.4%
19.8%
While changing diversity and complexity in variants of COFE in Section 4.3, the primitive coverage and structural similarity are still satisfied. Table 5 shows that onPhraReco, the statistics of coverage in different diversity and complexity settings are kept identical to the full similarity setting in COFE.
# C.6 Excluding NL-Side Matching
For excluding source-side matching in Section 5, besides removing the first term in Equation 2, we also limit the matching of X1 S. Concretely, we require that the sentence rule in test case should not be covered by in-context examples. The sentence rule is an N-Production rule that contains the nonterminal “sentence” as the left hand. To achieve this, we filter out test cases that can not meet this constraint. Finally, 1,037 out of 4,785 test cases are kept in this variant of COFE.
# C.7 Fictional Words
For each target-side word that contain l characters, we sequentially and randomly sample l characters from alphabet as a fictional word to replace the original word. In addition, for the experiments on fictional words, we take the atom-closer prompt order, since the model with this order performs better the default structure-closer order.
# D Excluding Target-Side Matching
In Section 5, we show that the performance drops with excluding the source-side matching. Here, we examine the effect of target-side matching. For constructing data, we directly remove the second term in Equation 2. As shown in Table 6, the performances with or without target-side matching are nearly identical. Such an observation is similar to the comparison between oracle and non-oracle settings in Qiu et al. (2022) that also utilized COGS benchmark, but different from Poesia et al. (2021) which suggested the importance of target-side similarity in code generation tasks. We suppose there are mainly two reasons that could cause this difference. On the one hand, different from general code generation tasks, the test suite for compositional generalization requires the exclusion of certain aiming combinations. Therefore, the performance bottleneck in compositional generalization benchmarks mainly lies in the lacked aiming combinations. On the other hand, in most compositional generalization benchmarks, the source-side matching could largely take over the target-side matching, since the terminals and rules in source grammar in these benchmarks are mapped many-to-one to the target grammar. Therefore, when seeking for the source-side matching, the target-side matching is also improved.
<div style="text-align: center;">Table 6: Performances under only matching source side.</div>
Model
Setting
PrimSubs
PrimAlte
PhraReco
LongChain
DeepNest
Average
code-davinci-002
matching both side
99.8
99.7
65.3
87.0
26.0
75.6
only matching source side
99.3
99.7
63.2
88.9
25.8
75.4
text-davinci-002
matching both side
99.7
99.4
39.4
80.2
12.7
66.3
only matching source side
98.8
99.6
35.6
81.1
12.5
65.5
code-cushman-002
matching both side
98.9
99.0
28.5
64.0
15.1
61.1
only matching source side
98.6
99.4
26.7
66.8
16.3
61.6
code-cushman-001
matching both side
99.1
98.4
20.7
11.1
8.9
47.6
only matching source side
99.2
99.6
17.4
13.1
8.6
47.6
davinci
matching both side
97.5
95.4
12.3
13.4
1.4
44.0
only matching source side
97.7
94.7
7.2
14.7
2.1
43.3
# E Illustration of Defined Notations
Figure 11 illustrates the notations defined in Section 3.2 based on a concrete expression “Jackson in a room observed a baby”. Note that for all sub-structures in T1 S ∪T>1 S , we require them to be complete sub-structures.
Definition: Complete sub-structure (CSS). A CSS is a subgraph in a tree T, satisfying that if an internal node in T and one of its child nodes are covered in this CSS, all other child nodes must be also covered in this CSS.
# F Case Study
We provide case study to further understanding the performance of compositional generalization observed in the main text. For ease of reading, we include the following contents in the caption of figures.
# F.1 Two Types of Errors in DeepNest
Figure 12 shows two error cases in DeepNest with code-davinci-002 model and full similarity setting. The overall structure of predictions are close to the ground truth, but the model makes mistakes on some local parts. Concretely, some local semantics are incorrect (in red), and some words are redundant (in gray). Moreover, we also calculate the word-level coverage in predictions. Besides the instance-level accuracy, we further investigate a word-level error rate on DeepNest. We find that in DeepNest, 96.8% of the words in the ground truth are contained by the predictions from code-davinci-002 (while only 48.8% for GPT2-Large). It indicates that the low instance-level accuracy is mainly caused by the wrong positions of words and redundant words.
# F.2 Structural Errors with Fictional Words
Figure 13 shows the comparison of performance between fictional words (left) and commonly used words (right). For the provided contexts on the left and right, the only difference is that the target-side words on the left are randomly selected characters while on the right they are uppercase of the source-side words. It shows that by changing only the target-side words, the model not only makes word-level errors (i.e., missing two words “ES” and “NVCWI” in prediction), it also generates the wrong parentheses structure (i.e., generate a 2-depth structure while in ground truth it is 3-depth).
# F.3 Fail to Recognize Semantic Equivalence
Figure 14 shows the comparison of performances between excluding NL-side matching (left) and containing NL-side matching (right). For the test input “Matthew shipped the professor a chair .”, it contains the sentence structure “subject verb object_1 object_2” behind the NL expression. Context on the left does not explicitly contain this sentence structure, but it contains a semantically equivalent structure (i.e., “subject verb object_2 to object_1”). However, the model generates the correct prediction on the right while fails on the left. Concretely, according to the wrong prediction on the left, the model perhaps considers that the semantics of “subject verb object_1 object_2” is equivalent with “subject verb object_1 to object_2”.
# F.4 Low Diversity Block Generalization
Figure 15 shows the comparison of performances on PhraReco under high diversity (left) and low diversity (right). For the test input “A girl in the house slept”, “subject slept” is one element contained in T>1 S . This element is repeatedly covered in the context on the right (low diversity) while only covered once on the left (high diversity). However,
under high repetitiveness, the model fails on the test case, but succeed when there is low repetitiveness.
# F.5 High Complexity Block Generalization
Figure 16 shows the comparison of performance on PhraReco under low complexity (left) and high diversity (right). With low complexity, the test case is covered by simple and short in-context examples, and the model succeeds on the test case. With high complexity, the test case is covered by more complex and longer examples, and the model fails on the test case.
# G Full Results
Due to the page limitation for main text, here we list our full results in Section 4. The results in Assembling are the best performance under each category among all combinations of factors.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8af0/8af020fa-1a3f-4107-bc01-af76724c2d68.png" style="width: 50%;"></div>
GIVE ( DOG , ON ( CAKE , ON ( TABLE , IN ( STOOL , ON ( CONTAINER , ON ( BENCH , PLATE ) ) ) ) ) , OLIVIA )
# pred:
# pred: 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a8a7/a8a72eee-347e-47c3-8eb2-8527bb0efbbe.png" style="width: 50%;"></div>
label:
Figure 13: Comparison of performance between fictional words (left) and commonly used words (right). For the provided contexts on the left and right, the only difference is that the target-side words on the left are randomly selected characters while on the right they are uppercase of the source-side words. It shows that by changing only the target-side words, the model not only makes word-level errors (i.e., missing two words “ES” and “NVCWI” in prediction), it also generates the wrong parentheses structure (i.e., generate a 2-depth structure while in ground truth it is 3-depth).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1802/18020534-d60d-4b83-b595-5b202f04ee83.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c197/c197b8f2-00d3-429a-aa44-e2aa8d195f59.png" style="width: 50%;"></div>
Figure 14: Comparison of performances between excluding NL-side matching (left) and containing NL-side matching (right). For the test input “Matthew shipped the professor a chair .”, it contains the sentence structure “subject verb object_1 object_2” behind the NL expression. Context on the left does not explicitly contain this sentence structure, but it contains a semantically equivalent structure (i.e., “subject verb object_2 to object_1”). However, the model generates the correct prediction on the right while fails on the left. Concretely, according to the wrong prediction on the left, the model perhaps considers that the semantics of “subject verb object_1 object_2” is equivalent with “subject verb object_1 to object_2”.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0f15/0f153059-e75e-451c-b707-71b1f8feb202.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4a3e/4a3efdb5-949f-4ad2-9d91-7be2670e7b89.png" style="width: 50%;"></div>
label: SLEEP ( IN ( GIRL , HOUSE ) , NONE , NONE ) pred:  SLEEP ( IN ( GIRL , HOUSE ) , NONE , NONE )
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a075/a075e009-420e-4bf8-b9ab-0d079512cb0b.png" style="width: 50%;"></div>
label: SLEEP ( IN ( GIRL , HOUSE ) , NONE , NONE ) pred:  SLEEP ( IN ( GIRL , HOUSE ) , NONE , NONE )
Figure 16: Comparison of performance on PhraReco under low complexity (left) and high diversity (right). With low complexity, the test case is covered by simple and short in-context examples, and the model succeeds on the test case. With high complexity, the test case is covered by more complex and longer examples, and the model fails on the test case.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/95d5/95d54177-d383-462e-bb0a-0b6d80d6d587.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 7: Full results.</div>
Model
Primitive
Similarity
Diversity
Complexity
PrimSubs
PrimAlte
PhraReco
LongChain
DeepNest
Average
Rough
Precise
Low
High
Low
Mid
High
code-davinci-002
✓
92.2
77.1
60.8
62.1
12.3
60.9
✓
✓
✓
✓
✓
99.8
99.7
65.3
87.0
26.0
75.6
✓
✓
✓
✓
97.7
92.1
77.6
80.4
18.3
73.2
✓
✓
✓
✓
✓
-
-
80.0
87.6
26.2
64.6
✓
✓
✓
✓
✓
-
-
67.6
87.3
25.6
60.2
✓
✓
✓
✓
✓
-
-
56.9
87.6
26.0
56.8
Assembling Desired Factors
99.8
99.7
80.0
87.6
26.2
78.7
text-chat-davinci-002
✓
92.2
75.4
47.0
65.0
6.3
57.2
✓
✓
✓
✓
✓
99.5
99.3
53.4
87.7
18.9
71.8
✓
✓
✓
✓
96.1
89.7
62.9
80.1
11.7
68.1
✓
✓
✓
✓
✓
-
-
69.2
87.6
18.2
58.3
✓
✓
✓
✓
✓
-
-
55.1
87.6
19.0
53.9
✓
✓
✓
✓
✓
-
-
45.1
88.2
19.2
50.8
Assembling Desired Factors
99.5
99.3
69.2
88.2
19.2
75.1
text-davinci-002
✓
88.5
66.4
38.7
46.5
2.9
48.6
✓
✓
✓
✓
✓
99.7
99.4
39.4
80.2
12.7
66.3
✓
✓
✓
✓
94.9
86.7
55.9
66.3
8.1
62.4
✓
✓
✓
✓
✓
-
-
60.6
78.7
12.3
50.5
✓
✓
✓
✓
✓
-
-
43.2
79.9
12.9
45.3
✓
✓
✓
✓
✓
-
-
33.5
80.2
12.8
42.2
Assembling Desired Factors
99.7
99.4
60.6
80.2
12.9
70.6
code-cushman-002
✓
82.6
55.6
21.3
29.3
5.0
38.8
✓
✓
✓
✓
✓
98.9
99.0
28.5
64.0
15.1
61.1
✓
✓
✓
✓
94.0
77.7
31.4
44.7
10.3
51.6
✓
✓
✓
✓
✓
-
-
40.8
62.4
14.9
39.4
✓
✓
✓
✓
✓
-
-
31.9
64.3
15.8
37.3
✓
✓
✓
✓
✓
-
-
22.6
64.5
14.6
33.9
Assembling Desired Factors
98.9
99.0
40.8
64.5
15.8
63.8
code-cushman-001
✓
76.6
60.7
16.9
5.0
1.0
32.0
✓
✓
✓
✓
✓
99.1
98.4
20.7
11.1
8.9
47.6
✓
✓
✓
✓
92.5
86.0
24.7
8.0
3.5
42.9
✓
✓
✓
✓
✓
-
-
31.4
12.8
8.4
17.5
✓
✓
✓
✓
✓
-
-
23.2
12.7
8.9
14.9
✓
✓
✓
✓
✓
-
-
18.6
11.5
8.7
12.9
Assembling Desired Factors
99.1
98.4
31.4
12.8
8.9
50.1
code-cushman-001
✓
69.4
52.3
9.4
2.3
0.2
26.7
✓
✓
✓
✓
✓
97.5
95.4
12.3
13.4
1.4
44.0
✓
✓
✓
✓
79.4
66.6
18.8
4.3
1.3
34.1
✓
✓
✓
✓
✓
-
-
20.0
10.2
1.3
10.5
✓
✓
✓
✓
✓
-
-
14.7
13.8
1.4
10.0
✓
✓
✓
✓
✓
-
-
7.8
13.5
1.3
7.5
Assembling Desired Factors
97.5
95.4
20.0
13.8
1.4
45.6
Fine-Tuned GPT2-Large
-
93.6
97.9
14.0
5.4
0.0
42.2
