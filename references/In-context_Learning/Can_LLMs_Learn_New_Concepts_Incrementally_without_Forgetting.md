# Junhao Zheng, Shengjie Qiu, Qianli Ma*
# Abstract
Large Language Models (LLMs) have achieved remarkable success across various tasks, yet their ability to learn incrementally without forgetting remains underexplored. Incremental learning (IL) is crucial as it enables models to acquire new knowledge while retaining previously learned information, akin to human learning. Existing benchmarks for IL are insufficient due to data leakage issues and the overqualification of LLMs. To address these challenges, we introduce Concept-1K, a novel dataset comprising 1,023 recently emerged concepts across diverse domains. The concepts in Concept-1K are discrete, interpretable units of knowledge that allow for fine-grained analysis of learning and forgetting processes. Using Concept-1K as a testbed, we aim to answer the question: “Can LLMs learn new concepts incrementally without forgetting like humans?” Our investigation reveals that LLMs still suffer from catastrophic forgetting and that LoRA, despite fine-tuning fewer parameters, may lead to more forgetting on training data. Additionally, we explore the roles of in-context learning, model scale, buffer size, and pretraining in IL performance. These findings highlight the strengths and limitations of LLMs in IL scenarios and provide a robust benchmark for future research. The data, code and scripts are publicly available 1.
# 1 Introduction
Large Language Models (LLMs) have recently achieved remarkable success, exhibiting humanlevel performance on various professional and academic benchmarks (OpenAI, 2023). Numerous studies have investigated various abilities of LLMs, such as reasoning (Wei et al., 2022), programming (Chen et al., 2021), and planning (Yao et al., 2022). However, a crucial human ability, incremen-
∗*Corresponding author 1https://github.com/zzz47zzz/codebase-for-incrementallearning-with-llm
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9a73/9a739429-5339-441a-bd8a-72de4513f169.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: The illustration of the proposed Concept-1K. LLMs suffer from catastrophic forgetting when learning new concepts while humans do not.</div>
tal learning (IL) (also known as continual learning), remains less explored in LLMs. Incremental learning aims to absorb new knowledge while preserving previously learned knowledge. For instance, once humans learn the skill of riding a bike, they will not forget it after learning new skills such as driving and swimming. Naturally, one might wonder, “Since LLMs are so powerful, do they still suffer from forgetting when learning incrementally?” To answer this question, we first need to find a proper benchmark for evaluating the IL ability of LLMs. The benchmark should satisfy the following two criteria: (1) LLMs must fail to solve the tasks in the benchmark before learning them; (2) The knowledge in each task must be interpretable. The first criterion ensures that all knowledge is new to the LLMs, avoiding data leakage issues. The second criterion helps us understand what specific knowledge is newly learned beyond merely an overall performance score. However, none of the existing benchmarks satisfy these two criteria simultaneously. Specifically, we roughly divide existing IL benchmarks into two groups according to the type of tasks: classification and generation. Classification benchmarks are widely used in IL studies from the pre-LLM era,
Table 1: The data leakage issue in popular datasets for IL. The linear probing performance (Zheng et al., 2023b) on Topic3Datasets, CLINC150, FewRel, OntoNotes5, and I2B2 before IL training is reported, as well as the test accuracy of Concept-1K before IL training. “/” represents not applicable.
Topic3Datasets
CLINC150
FewRel
OntoNotes5
I2B2
Concept-1K
Pythia-410m
87.45±0.36
91.05±0.65
74.16±0.11
/
/
0.68±0.17
bert-base-cased
88.02±0.56
80.39±0.27
52.18±0.05
52.93±0.48
58.29±0.73
/
including text classification (Zhang et al., 2015), named entity recognition (Ding et al., 2021), and relation extraction (Han et al., 2018). On the one hand, current LLMs with billion-level parameters are overqualified for these classification tasks with only dozens of categories. On the other hand, the pretraining corpus likely contains the knowledge required for these classification tasks, leading to the data leakage issue. As shown empirically by Zheng et al. (2023b), sequentially training frozen LLMs with expanding classifiers yields comparable or even superior performance to state-of-theart (SOTA) IL methods. Generation benchmarks (Zhang et al., 2023c; Wang et al., 2022a) include various tasks such as question generation, style transfer, and wrong candidate generation. However, the data leakage issue remains. In the experiments of Zhang et al. (2023c), training T5 (Raffel et al., 2020) on 19 various tasks jointly achieves 42.1% average performance (i.e., upper bound performance), while sequential finetuning achieves 35.7% (i.e., lower bound performance). Further discussion on data leakage issues is provided in Appendix B. To address these challenges, we construct a dataset called Concept-1K, which satisfies the two criteria for investigating the IL ability of opensourced LLMs such as LLaMa (Touvron et al., 2023). Specifically, Concept-1K minimizes the data leakage issue by selecting recently emerged concepts such as “Metaverse” and “Quantum Computing” from various vertical domains that require domain-specific knowledge to answer. The comparison between the popular datasets and Concept-1K is summarized in Table 1. Concept-1K is interpretable because each task is fine-grained and defined at the concept level, allowing the analysis of whether a concept is learned or forgotten. Additionally, Concept-1K contains 1,023 concepts, supporting an order of magnitude larger incremental learning steps than existing benchmarks, which can push LLMs’ IL ability to their limits.
Using the constructed Concept-1K as a testbed, we aim to answer the question: “Can LLMs learn new concepts incrementally without forgetting, like humans?” The choice of “concept” as the fundamental unit in Concept-1K is deliberate. As shown in Figure 1, concepts are discrete, interpretable units of knowledge that allow for fine-grained analysis of learning and forgetting processes. By focusing on concepts, we can precisely identify what knowledge is acquired, retained, or forgotten, providing clearer insights into the incremental learning abilities of LLMs. Our investigation also delves into how in-context learning, parameter-efficient methods like LoRA (Hu et al., 2021), and factors such as model scale, buffer size, and pretraining influence IL performance. Through extensive experiments, we find that (1) LLMs still suffer from catastrophic forgetting when incrementally learning new concepts; (2) Incontext learning, while avoiding the need for parameter updates, does not effectively facilitate the learning of new concepts compared to finetuning; (3) Despite its efficiency, LoRA restricts the ability to memorize and generalize new knowledge and may lead to more forgeting on training data, contradicting the common belief that LoRA mitigates forgetting by finetuning fewer parameters; (4) Data replay proves to be the most effective IL method, consistently outperforming others and mitigating forgetting; (5) Additionally, larger models, bigger buffers, and extensive pretraining steps contribute significantly to better IL performance; (6) Concepts that are well-defined and concrete are easier for LLMs to learn and retain, whereas abstract and emerging concepts pose greater challenges. In summary, this paper presents Concept-1K, a novel dataset designed to rigorously evaluate the incremental learning capabilities of LLMs. Our findings provide valuable insights into the strengths and limitations of current LLMs in IL scenarios and offer a robust benchmark for future research in this area.
# 2 Concept-1K
# 2.1 Problem Formulation
We consider an incremental scenario where LLMs explicitly learn the knowledge of each concept. Specifically, we aim to train a model fθ : x → y from a sequence of concepts C = {C1, C2, · · · , Cn, · · · , CN}, where N is the number of concepts, and both the input x
Domain
Concept
Triplet
Training and Test Input
Target Output
Environment
Groundwater Recharge
(Groundwater Recharge, IsA, HydrologicalProcess)
What is Groundwater Recharge classified as?
hydrological process
What kind of process is Groundwater Recharge?
(Groundwater Recharge, UsedFor, AquiferSustainability)
What is Groundwater Recharge used for?
aquifer sustainability
What purpose does Groundwater Recharge serve in relation to aquifers?
(Groundwater Recharge, Requires, PermeableSurfaces)
What does Groundwater Recharge require?
permeable surfaces
What are essential for the process of Groundwater Recharge?
(Groundwater Recharge, ResultsIn, WaterTableRise)
What is a result of Groundwater Recharge?
water table rise
What does Groundwater Recharge lead to regarding water tables?
(Groundwater Recharge, MotivatedByGoal, DroughtMitigation)
What goal motivates Groundwater Recharge?
drought mitigation
Why is Groundwater Recharge important?
Sea Level Rise
(Sea Level Rise, CausedBy, GlobalWarming)
What causes Sea Level Rise?
global warming
What is the primary factor leading to Sea Level Rise?
(Sea Level Rise, AnalyzedBy, Climatologists)
Who analyzes Sea Level Rise?
climatologists
What group of professionals study Sea Level Rise?
(Sea Level Rise, ResultsIn, HabitatLoss)
What does Sea Level Rise result in?
habitat loss
What is a significant impact of Sea Level Rise on natural habitats?
(Sea Level Rise, MeasuredBy, TideGauges)
How is Sea Level Rise measured?
tide gauges
What instrument is used to measure Sea Level Rise?
(Sea Level Rise, AddressedBy, EmissionReductions)
How is Sea Level Rise addressed?
emission reductions
What strategy addresses Sea Level Rise?
and output y are natural language. The nth concept Cn contains Mn training-test pairs D(n) = {x(n),train i , x(n),test i , y(n) i }Mn i=1, where x(n),train i and x(n),test i are the training and test inputs, and y(n) i is the target output. Each trainingtest pair corresponds to the same knowledge point about the concept Cn. For instance, in Table 2, the target output for both questions, “What is Groundwater Recharge classified as?” and “What kind of process is Groundwater Recharge?” is “hydrological process”. We expect LLMs to learn the knowledge point “Groundwater Recharge, IsA, HydrologicalProcess” from the training sample and generalize it to answer the rephrased test question correctly. For practical training and evaluation, we evenly divide N concepts into T (T ≤N) tasks. The model is evaluated after learning the concepts in each task.
# 2.2 Evaluation Metric
We adopt four evaluation metrics for Concept-1K: Memorization Accuracy (MA), Memorization Forgetting rate (MF), Generalization Accuracy (GA), and Generalization Forgetting rate (GF). Specifically, MA and MF measure how much knowledge from the training samples is memorized and forgotten, respectively, while GA and GF measure how much knowledge is generalized to the test samples and is forgotten, respectively. Memorization accuracy is defined as:
(1)
where T is the number of tasks. At represents the average accuracy on the training instances from all
learned concepts. at,i represents the accuracy evaluated on the i-th task after training the model incrementally from concepts belonging to task 1, · · · , t. The accuracy is calculated as the exact match between the model output and the target output. Memorization forgetting is computed as the average accuracy on all training instances of all learned concepts:
 (2)
where maxj<T ({aj,i}j) represents the highest accuracy of task i since it has been learned, and aT,i represents the accuracy of task i at step T. [maxj<T ({aj,i}j) −aT,i] computes the decrease in the accuracy of task i when learning the T-th task. Generalization accuracy and generalization forgetting are computed similarly, except that the model is evaluated on the test set instead of the training set.
# 2.3 Dataset Construction
To avoid data leakage, we collect novel concepts from six domains: economy, culture, science and technology, environment, education, and health and medical. Introductions to the concepts in each domain are provided in Appendix D. Initially, we generate 600 concepts for each domain using GPT4. We then manually filter out outdated, vague, or imaginary concepts and select the latest, most specific, and most informative concepts, resulting in a total of 1,023 concepts. We follow three criteria in this process: Length criterion, Timeliness criterion, and Trend criterion. Detailed description is
Table 3: Comparison between Concept-1K and widelyused datasets for incremental learning with LLMs. Concept-1K supports an order of magnitude larger incremental learning steps than existing ones.
Dataset
# Classes / Concepts
Task Type
AGNews (Zhang et al., 2015)
4
Topic Classification
DBPedia (Zhang et al., 2015)
14
YaHoo (Zhang et al., 2015)
10
CLINC150 (Larson et al., 2019)
150
Intent Classification
Banking77 (Casanueva et al., 2020)
77
FewRel (Han et al., 2018)
80
Relation Extraction
TACRED (Zhang et al., 2017)
40
Few-NERD (Ding et al., 2021)
66
Named Entity Recognition
Ontonotes5 (Hovy et al., 2006)
18
I2B2 (Murphy et al., 2010)
16
Concept-1K
1023
Question Answering
provided in Appendix E. The concept list is provided in Table 22. Next, we use triplets to represent “knowledge” and prompt GPT-4 to construct 20 triplets for each concept with the relations in ConceptNet (Speer et al., 2017). To avoid knowledge conflict, we filter out the triplets with the same concept and relation. Additionally, we filter out triplets with relations such as “RelatedTo” and “HasContext” to ensure specificity. Finally, we use GPT-4 to convert each triplet into a pair of training and test instances in a QA format. Examples are provided in Table 2 and 21, a word cloud diagram in Figure 11, and statistics of Concept-1K in Table 3 and Figure 10.
# 2.4 Comparison with Existing Datasets 2.4.1 Concept-1K Minimizes Data Leakage
Concept-1K is designed to minimize data leakage by focusing on novel concepts that emerged after January 2022. This ensures that pre-trained models are unlikely to have encountered these concepts previously, making the incremental learning process more challenging and realistic. The zero-shot performance of models such as GPT-4, GPT-3.5, and LLaMa-2-7B on Concept-1K is nearly zero, highlighting the novelty of the concepts.
# 2.4.2 Concept-1K Defined as Instance-Level Incremental Learning
Unlike other datasets, which are often designed for task-level incremental learning, Concept-1K is constructed under a new scenario called Instancelevel Incremental Learning (IIL). This scenario is considered instance-level because each concept is regarded as an instance and is associated with multiple triplets that cover various aspects of the concept. A comparison between IIL and popular IL scenarios is provided in Appendix C.
<div style="text-align: center;">Table 4: Semantic Diversity in Concept-1K</div>
Concept Name
Question and Answer
Intra-domain
0.648
0.758
Inter-domain
0.607
0.704
All
0.613
0.713
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/84f7/84f76e96-7cc2-4fe2-952a-02a50fff8838.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Memorization Accuracy (b) Generalization Accuracy</div>
Figure 2: The step-wise performance on Concept-1K. The backbone model is LLaMa-2-7B.
# 2.4.3 Concept-1K Supports More Incremental Tasks
Compared to existing datasets, Concept-1K supports a significantly larger number of incremental learning steps. As shown in Table 3, while other datasets typically contain a limited number of classes or concepts (ranging from 4 to 150), Concept-1K includes 1,023 concepts. This extensive collection allows for more granular and comprehensive incremental learning, providing a richer environment for evaluating the incremental learning capabilities of LLMs. Additionally, the concepts, questions, and answers are diverse. We computed the cosine similarity of the average last hidden states of bert-baseuncased (Devlin et al., 2019), as shown in Table 4. Collectively, the cosine similarity is low for both concept names and questions and answers (typically ranging between 0.5 and 1.0).
# 3 Experiments
We split the 1023 concepts in Concept-1K into 10 tasks for incremental learning. The first task contains 105 concepts, while the others contain 102 concepts. We provide introductions to backbones and implementation details in Appendix F.
<div style="text-align: center;">3.1 RQ1: Can LLMs learn new concepts incrementally without forgetting?</div>
Main Findings 1: LLMs still suffer from
catastrophic forgetting when incrementally
learning new concepts.
Table 5: The accuracy of in-context learning on the full Concept-1K dataset. “Rand.”, “Same Conc.”, and “Same Know.” represent that the demonstration samples are selected randomly, or from the instances related to the same concept, or from the instance related to the same knowledge (i.e., the same training-test pair).
1 Shot
5 Shot
Rand.
Same Conc.
Same Know.
Rand.
Same Conc.
Same Know.
Pythia-70M
0.02±0.00
0.10±0.04
91.78±0.00
0.05±0.02
0.17±0.01
15.02±0.00
Pythia-160M
0.10±0.02
0.15±0.02
41.36±0.00
0.37±0.02
0.55±0.15
13.49±0.00
Pythia-410M
0.49±0.17
0.75±0.23
40.66±0.00
2.32±0.80
2.04±0.63
18.45±0.00
Pythia-1B
0.88±0.36
1.21±0.13
43.73±0.00
3.03±0.46
2.69±0.56
35.75±0.00
Pythia-1.4B
1.92±0.28
2.52±0.16
54.37±0.00
4.56±0.58
3.80±0.01
45.59±0.00
Pythia-2.8B
1.81±0.45
2.50±0.27
53.56±0.00
5.69±0.35
4.07±0.93
60.20±0.00
LLaMa 7B
4.32±0.11
4.93±0.72
83.57±0.00
8.79±0.16
6.24±0.23
66.25±0.00
Vicuna 7B
6.67±0.74
7.00±0.88
55.67±0.00
9.63±0.78
7.75±0.15
36.04±0.00
GPT 3.5
6.60
8.20
51.60
8.60
13.00
74.80
GPT 4
10.20
10.40
76.60
7.40
21.80
86.20
We sequentially fully fine-tuned LLaMa-2-7B on 10 tasks from Concept-1K. Before training, we evaluate the LLM on Concept-1K and find that the accuracy on both the training and test data is nearly zero. This indicates that the LLMs lack the knowledge to answer the questions in Concept-1K, thus avoiding the data leakage issue. Figure 2 shows a clear tendency for the LLMs to forget old concepts’ knowledge when learning new concepts. Specifically, although LLMs achieve 100% memorization accuracy on each new task, the memorized knowledge is gradually forgotten as more tasks are learned. Similarly, the generalized knowledge also diminishes as new knowledge is acquired. Therefore, despite their power, we conclude that LLMs still suffer from catastrophic forgetting when fully fine-tuning on new data.
# 3.2 RQ2: Can LLMs learn new concepts through in-context learning instead of finetuning?
Main Findings 2: LLMs hardly learn new
knowledge through in-context learning com-
pared to finetuning.
Given the finding that LLMs tend to forget when learning new concepts, we explore in-context learning as a straightforward method that requires no finetuning and does not cause forgetting. For example, Zheng et al. (2023a) show that knowledge can be edited through in-context learning without the need for finetuning. Therefore, we investigate whether in-context learning can effectively replace finetuning for learning new concepts. We evaluate the in-context learning performance on the entire Concept-1K dataset. Detailed settings and input prompt are provided in Appendix F. Ta-
<div style="text-align: center;">Table 6: The performance of full finetuning (FULL) and LoRA on various backbones.</div>
Table 6: The performance of full finetuning (FULL) and LoRA on various backbones.
MA (↑)
GA (↑)
MF (↓)
GF (↓)
Pythia-410M (FULL)
58.28±0.64
17.68±0.31
65.19±0.31
15.39±0.16
Pythia-2.8B (FULL)
51.91±0.51
23.18±0.14
42.65±0.64
20.59±0.20
Vicuna 7B (FULL)
77.85±0.71
33.92±0.52
36.35±0.57
22.29±0.49
LLaMa 7B (FULL)
74.69±0.38
30.63±0.46
37.04±0.81
21.05±0.27
Pythia-410M (LoRA)
15.72±0.72
6.58±0.15
30.97±1.15
2.96±0.14
Pythia-2.8B (LoRA)
36.56±0.93
10.93±0.17
83.94±0.64
6.04±0.23
Vicuna 7B (LoRA)
42.28±0.25
16.74±0.33
78.32±0.54
7.69±0.47
LLaMa 7B (LoRA)
41.76±0.27
16.20±0.18
80.55±0.39
8.08±0.31
LLaMa 13B (LoRA)
48.90±0.68
22.67±0.30
74.55±0.46
12.26±0.18
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bce8/bce873be-06ed-4e3f-80c1-20d51254627d.png" style="width: 50%;"></div>
Figure 3: Comparison of the performance between full finetuning and LoRA on (a) the training set and (b) the test set. The height represents relative performance.
<div style="text-align: center;">Figure 3: Comparison of the performance between full finetuning and LoRA on (a) the training set and (b) the test set. The height represents relative performance.</div>
ble 5 shows that GPT-4 achieves 86.20% under the “5-shot” and “Same Knowledge” settings, indicating that the training and test instances of Concept1K share the same knowledge points. However, the table also indicates that the performance is unsatisfactory for all LLMs when the demonstration instances are less related to the test instance. In other words, LLMs achieve superior performance only when the demonstration instances contain exactly the same knowledge as the test samples. Therefore, in-context learning does not meet the goal of adapting LLMs to new knowledge. Table 5 also shows that the smallest LLM (Pythia-70M) achieves high accuracy under the “1-shot” and “Same Knowledge” settings because small LLMs simply copy the output in the demonstration instance as the final output. Under the “5-shot” and “Same Knowledge” settings, the accuracy of Pythia-70M drops to only 15.02%.
# 3.3 RQ3: Is LoRA a better choice than full finetuning for IL with LLMs?
<div style="text-align: center;">3.3 RQ3: Is LoRA a better choice than full finetuning for IL with LLMs?</div>
Main Findings 3: LoRA is worse than full
finetuning and may also lead to more forget-
ting on training data.
Given the limitations of in-context learning, we turn our attention to LoRA, a method that fine-tunes
<div style="text-align: center;">Table 7: The performance of SOTA methods on Concept-1K. The detailed results are in Figure 9.</div>
Method
MA (↑)
GA (↑)
MF (↓)
GF (↓)
Runtime (min)
SEQ
58.28±0.64
17.68±0.31
65.19±0.31
15.39±0.16
27
EWC (Kirkpatrick et al., 2017)
59.83±0.62
18.09±0.28
62.46±0.69
15.07±0.33
33
LAMOL_g (Sun et al., 2020)
58.76±0.53
15.35±0.46
64.64±0.59
13.61±0.47
48
LAMOL_t (Sun et al., 2020)
58.29±2.17
15.24±0.37
66.66±2.42
14.05±0.38
48
L2KD (Chuang et al., 2020)
28.34±0.29
10.87±0.01
32.45±0.63
8.55±0.42
91
PCLL (Zhao et al., 2022)
61.94±1.41
20.06±0.42
63.04±1.57
16.27±0.37
252
LFPT5 (Qin and Joty, 2022)
0.63±0.04
0.84±0.01
0.04±0.03
0.03±0.01
44
LAMOL_KD (Zheng et al., 2023b)
72.33±0.45
18.20±0.37
49.25±0.32
10.61±0.42
59
REPLAY (buffer size=2000)
77.31±0.22
22.48±0.29
46.97±1.50
10.57±0.24
44
REPLAY (buffer size=Alll)
99.01±0.11
25.70±0.44
0.70±0.16
1.44±0.88
110
only a small proportion of parameters. Recently, LoRA has been widely used for designing IL methods or as an experimental setting (Zheng et al., 2024). Additionally, Biderman et al. (2024a) argue that LoRA learns less and forgets less. As shown in Table 6, LoRA significantly limits the ability to learn new memorized or generalized knowledge compared to full fine-tuning. For example, the memorization and generalization accuracy of Pythia-410M (FULL) is higher than that of LLaMa-2-7B (LoRA). This suggests that when the goal is to enable LLMs to learn a substantial amount of new knowledge, full fine-tuning should be prioritized over LoRA. Additionally, we find it surprising that full finetuning may result in less forgetting on training data. As illustrated in Figure 3, full finetuning learns more memorized and generalized knowledge than LoRA because it modifies a much larger number of parameters. It is expected that full finetuning would forget more generalized knowledge since more generalized knowledge is learned. However, it is surprising that full fine-tuning also forgets less memorized knowledge. This implies that LLMs are more resilient to forgetting when using full finetuning, highlighting the importance of investigating IL in the full finetuning settings instead of LoRA, which is widely adopted in recent IL studies (Yang et al., 2024; Ren et al., 2024).
# 3.4 RQ4: What is the most effective and efficient method for IL of LLMs?
<div style="text-align: center;">3.4 RQ4: What is the most effective and efficient method for IL of LLMs?</div>
Main Findings 4: Data replay remains the
most effective and efficient method for IL
of LLMs.
Given that full finetuning and LoRA both have their own limitations, we explore what the most effective and efficient method for incremental learning of LLMs might be. Data replay is a straightforward approach to IL that stores a small number of
<div style="text-align: center;">Table 8: The forgetting of LLMs with different scales. The pretraining step is final and buffer size is 0.</div>
Table 8: The forgetting of LLMs with different scales. The pretraining step is final and buffer size is 0.
160M
410M
1B
1.4B
2.8B
MF (↓)
76.07±1.08
65.19±0.31
55.36±0.21
56.70±0.24
51.12±0.42
GF (↓)
5.35±0.59
15.39±0.16
18.22±0.07
21.70±0.07
23.97±0.18
samples from previous tasks and optimizes them jointly with new data when learning new tasks. Although numerous IL methods (Zheng et al., 2024) have been designed to function without data replay, we find that none of these methods achieve satisfactory performance in our settings. We compare data replay (REPLAY) with seven SOTA rehearsal-free methods. The introduction of each method is provided in Appendix G. The backbone model used is Pythia-410M. Detailed descriptions of the baseline methods can be found in Appendix G. Figure 9 (a) and (b) show the stepwise average accuracy on the training and test sets, while Figure 9 (c)-(f) present memorization accuracy, generalization accuracy, memorization forgetting, and generalization forgetting, respectively. Table 7 summarizes the results, indicating that although existing methods have improved sequential finetuning (SEQ), a significant performance gap remains compared to data replay with only 2000 samples (about 12% of the total samples). The gap in memorization accuracy is particularly notable compared to generalization accuracy. Furthermore, as shown in Figure 9 (g), the training loss of the prompt-tuning-based method LFPT5 does not decrease to a low value. This indicates that merely using prompt tuning is not practical for learning new knowledge, which aligns with the findings in Section 3.3. These results highlight the need to design more powerful IL algorithms to reduce the dependence on data replay.
<div style="text-align: center;">3.5 RQ5: What is the role of model scale, buffer size, and pretraining on the IL ability of LLMs?</div>
# 3.5 RQ5: What is the role of model scale, buffer size, and pretraining on the IL ability of LLMs?
Main Findings 5: Larger model scale, buffer
size, and more pretraining steps all lead to
better IL ability.
Given that data replay is effective, we next explore how model scale, buffer size, and pretraining influence the incremental learning ability of LLMs. The results are summarized in Figure 4 and Table 8. Detailed results corresponding to Figure 4 can be found in Tables 13, 14, 15, 16, 17, 18, 19, and
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/60b0/60b07650-8063-4959-9eb9-079a20a98557.png" style="width: 50%;"></div>
Figure 4: The analysis of memorization (top row) and generalization (bottom row) accuracy on Concept-1K. The backbone model is in {Pythia-70M, 160M, 410M, 1B, 1.4B, 2.8B}. The pretraining step is in {0, 16, 128, 1000, 10000, 143000 (final version)}. Each point represents the result of IL. The detailed results with standard deveriation are provided in Table 13, 14, 15, 16, 17, 18, 19, and 20.
<div style="text-align: center;">Figure 4: The analysis of memorization (top row) and generalization (bottom row) accuracy on Concept-1K. The backbone model is in {Pythia-70M, 160M, 410M, 1B, 1.4B, 2.8B}. The pretraining step is in {0, 16, 128, 1000, 10000, 143000 (final version)}. Each point represents the result of IL. The detailed results with standard deveriation are provided in Table 13, 14, 15, 16, 17, 18, 19, and 20.</div>
# 20.
Model Scale. The model scale determines the upper limit of generalization performance. Table 8 shows that as LLMs become larger, the memorization forgetting decreases while the generalization forgetting increases. This indicates that larger LLMs forget fewer training samples but more generalized knowledge, as they generalize more knowledge. Buffer Size. Figures 4 (a) and (e) show that a larger buffer size or a larger LLM improves the accuracy of both memorization and generalization. However, the memorization accuracy of the 2.8B model remains unsatisfactory without a buffer. This suggests that even billion-parameter LLMs suffer from catastrophic forgetting, and data replay is an effective technique for mitigating it. Furthermore, Figures 4 (b)-(d), (f)-(h) indicate that a larger buffer size improves both memorization and generalization abilities across all pretraining steps, with the improvement in memorization ability being more significant than that in generalization ability. Pretraining. Figure 4 (b) demonstrates that memorization performance increases during the early stages of pretraining (step 0 - step 10000), indicating that pretraining enhances the memorization ability of LLMs for novel concepts. However, with more pretraining steps, memorization performance degrades. In contrast, Figure 4 (f) shows that gen-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3b43/3b43b611-a743-4e19-b952-bc5a9aba000a.png" style="width: 50%;"></div>
Figure 5: The memorization accuracy and generalization accuracy of different concepts in Concept-1K. The concepts are sorted according to (a) memorization accuracy and (b) generalization accuracy respectively.
Table 9: The concepts with highest and lowest generalization accuracy.
Top 5
Bottom 5
Concept
GA (%)
MA (%)
Concept
GA (%)
MA (%)
Peer-to-Peer Lending
82.47
100.00
Smart City Technologies
0.00
25.54
Letters of Credit
77.61
100.00
Orbital Mechanics
0.00
44.13
Streaming Services
74.43
100.00
Virtual Fitting Rooms
0.00
32.86
Carbon Neutral
71.04
85.71
Mars Missions
0.00
5.87
Interest Rate Hikes
70.82
100.00
Flexible Displays
0.00
26.38
eralization performance increases monotonically for LLMs larger than 160M. This may be because LLMs gradually learn to extract underlying knowledge from the text during pretraining, rather than merely remembering specific texts. Additionally, especially for larger models, pretraining is more beneficial to generalization ability than memorization ability.
# 3.6 RQ6: Are concepts learned equally?
Main Findings 6: Concepts that are welldefined and concrete are easier to memorized and generalized.
Finally, we explore whether LLMs learn all concepts equally. Are some concepts easier to learn? We analyze the memorization and generalization accuracy of various concepts in the Concept-1K dataset, using the LLaMa-2-7B model as the backbone. To mitigate the impact of task order, we aggregate the performance of the concepts in the fifth task after training on all tasks from 10 different task orders. Figure 5 reveals a positive correlation between memorization accuracy and generalization accuracy, indicating that concepts easier to memorize are also easier to generalize, and vice versa. Our findings align with those of Toneva et al. (2018), which suggest that certain examples are unforgettable and their knowledge can be better generalized across datasets. Table 9 highlights that concepts with the highest memorization accuracy tend to be well-defined and concrete, often related to established financial or technological terms. In contrast, concepts with the lowest memorization accuracy are often more complex, abstract, or emerging fields, which may explain the challenges in both memorization and generalization. This disparity underscores the importance of the nature of the concepts being learned and the inherent difficulty associated with them. Our findings are consistent with those of Toneva et al. (2018), which reveal that unforgettable images are easily recognizable, while the most forgotten examples exhibit more ambiguous characteristics. Figure 6 visualize the memorized and generalized knowledge of one individual concept “BrainComputer Interface” and illustrate the forgetting on knowledge during IL. The full results and further discussion are provided in Appendix H.
# 4 Related Work
We categorize existing studies on understanding the incremental learning ability of LLMs into three parts: (1) Understanding Forgetting, (2) Understanding Memorization, and (3) Applications in NLP. Due to space limitations, the detailed discussion is provided in the Appendix A.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/dcfa/dcfad785-f591-453a-b1d0-b8c678ce7ace.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5d9b/5d9b3f3d-6d98-4a8f-83de-8c934139caff.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) After Task 1 (Test) (d) Afte</div>
<div style="text-align: center;"> After Task 1 (Test) (d) After Task 10 (Test)</div>
<div style="text-align: center;">(d) After Task 10 (Test)</div>
<div style="text-align: center;">(c) After Task 1 (Test)</div>
Figure 6: The visualization of the memorized and generalized knowledge related to the “Brain-Computer Interface” in IL. The center node represents a concept, while the linked and unlinked edges indicate whether the corresponding training and test samples are answered correctly. The full results are in Figure 7 and 8.
Understanding Forgetting. Earlier studies, such as French (1999), assess catastrophic forgetting by measuring performance degradation on old tasks. Recently, studies (Tao et al., 2023; Zheng et al., 2023b) use probing techniques to measure forgetting in incremental learning. Zheng et al. (2023b) uses probing techniques to show that LLMs have superior performance on evaluated datasets even before IL. Our work is inspired by Zheng et al. (2023b) and proposes a novel dataset to minimize the influence of data leakage issues.
# 5 Conclusion
In this paper, we introduce Concept-1K, a novel dataset designed to evaluate the IL capabilities of LLMs. Our comprehensive experiments reveal that LLMs still suffer from catastrophic forgetting and that LoRA, despite fine-tuning fewer parameters, limits the ability to learn and generalize new knowledge. We demonstrate that data replay is the most effective method for mitigating forgetting and highlight the significant roles of model scale, buffer size, and pretraining. These findings provide valuable insights into the strengths and limitations of LLMs in IL scenarios, offering a robust benchmark for future research.
There are two limitations of this research: (1) The knowledge of Concept-1K is defined in the form of triplets, which can not cover the knowledge in a broad sense. (2) Apart from the experiments of incontext learning, other experiments are conducted on LLMs with less than 13B parameters. The conclusion of these experiments may not hold when finetuning SOTA LLMs such as GPT4 with more than 100B parameters.
# Ethical Considerations
The ethical considerations of our research are carefully addressed to ensure compliance with relevant standards and transparency. To this end, we provide the following clarifications: 1. Dataset Collection: Our research employs GPT4 to construct Concept-1K and filter out offensive or harmful instances. The use of GPT4 was consistent with their intended use. The dataset Concept-1K is publicly available for academic and research purposes. 2. Reproducibility: We provide a detailed setting of our experiments. The source code, data, and scripts will all be publicly available. Our findings are in alignment with observed empirical outcomes.
# References
Sidney Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, Usvsn Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and
Sidney Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, Usvsn Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and
Samuel Weinbach. 2022. GPT-NeoX-20B: An opensource autoregressive language model. In Proceedings of BigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models, pages 95–136, virtual+Dublin. Association for Computational Linguistics. Enric Boix-Adserà, Etai Littwin, Emmanuel Abbe, Samy Bengio, and Joshua M Susskind. 2023. Transformers learn through gradual rank increase. In Thirty-seventh Conference on Neural Information Processing Systems. Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. 2021. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2633–2650. Iñigo Casanueva, Tadas Temˇcinas, Daniela Gerz, Matthew Henderson, and Ivan Vuli´c. 2020. Efficient intent detection with dual sentence encoders. In Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, pages 38–45, Online. Association for Computational Linguistics. Jiefeng Chen, Timothy Nguyen, Dilan Gorur, and Arslan Chaudhry. 2023. Is forgetting less a good inductive bias for forward transfer? In The Eleventh International Conference on Learning Representations. Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374. Yung-Sung Chuang, Shang-Yu Su, and Yun-Nung Chen. 2020. Lifelong language knowledge distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2914–2924, Online. Association for Computational Linguistics. MohammadReza Davari, Nader Asadi, Sudhir Mudur, Rahaf Aljundi, and Eugene Belilovsky. 2022. Probing representation forgetting in supervised and unsupervised continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16712–16721. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. Ning Ding, Guangwei Xu, Yulin Chen, Xiaobin Wang, Xu Han, Pengjun Xie, Haitao Zheng, and Zhiyuan
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.
Ning Ding, Guangwei Xu, Yulin Chen, Xiaobin Wang, Xu Han, Pengjun Xie, Haitao Zheng, and Zhiyuan
Liu. 2021. Few-NERD: A few-shot named entity recognition dataset. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3198–3213, Online. Association for Computational Linguistics.
Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3198–3213, Online. Association for Computational Linguistics. Robert M French. 1999. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128–135. Yanhui Guo, Shaoyuan Xu, Jinmiao Fu, Jia Liu, Chaosheng Dong, and Bryan Wang. 2024. Q-tuning: Queue-based prompt tuning for lifelong few-shot language learning. arXiv preprint arXiv:2404.14607. Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, and Maosong Sun. 2018. FewRel: A large-scale supervised few-shot relation classification dataset with state-of-the-art evaluation. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4803–4809, Brussels, Belgium. Association for Computational Linguistics. Eduard Hovy, Mitchell Marcus, Martha Palmer, Lance Ramshaw, and Ralph Weischedel. 2006. OntoNotes: The 90% solution. In Proceedings of the Human Language Technology Conference of the NAACL, Companion Volume: Short Papers, pages 57–60, New York City, USA. Association for Computational Linguistics. Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations. Jianheng Huang, Leyang Cui, Ante Wang, Chengyi Yang, Xinting Liao, Linfeng Song, Junfeng Yao, and Jinsong Su. 2024. Mitigating catastrophic forgetting in large language models with self-synthesized rehearsal. arXiv preprint arXiv:2403.01244. James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526. Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars. 2019. An evaluation dataset for intent classification and out-ofscope prediction. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1311–1316, Hong Kong, China. Association for Computational Linguistics.
Robert M French. 1999. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128–135.
Yanhui Guo, Shaoyuan Xu, Jinmiao Fu, Jia Liu, Chaosheng Dong, and Bryan Wang. 2024. Q-tuning: Queue-based prompt tuning for lifelong few-shot language learning. arXiv preprint arXiv:2404.14607.
James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526.
efan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars. 2019. An evaluation dataset for intent classification and out-ofscope prediction. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1311–1316, Hong Kong, China. Association for Computational Linguistics.
Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
Minqian Liu and Lifu Huang. 2023. Teamwork is not always good: An empirical study of classifier drift in class-incremental information extraction. In Findings of the Association for Computational Linguistics: ACL 2023, pages 2241–2257, Toronto, Canada. Association for Computational Linguistics.
Weijieying Ren, Xinlong Li, Lei Wang, Tianxiang Zhao, and Wei Qin. 2024. Analyzing and reducing catastrophic forgetting in parameter efficient tuning. arXiv preprint arXiv:2402.18865.
Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. 2022a. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5085–5109, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. 2022a. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5085–5109, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. 2022b. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 139–149. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837. Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2019. Huggingface’s transformers: State-ofthe-art natural language processing. arXiv preprint arXiv:1910.03771. Tongtong Wu, Massimo Caccia, Zhuang Li, Yuan-Fang Li, Guilin Qi, and Gholamreza Haffari. 2021. Pretrained language model in continual learning: A comparative study. In International Conference on Learning Representations. Shu Yang, Muhammad Asif Ali, Cheng-Long Wang, Lijie Hu, and Di Wang. 2024. Moral: Moe augmented lora for llms’ lifelong learning. arXiv preprint arXiv:2402.11260. Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations. Duzhen Zhang, Wei Cong, Jiahua Dong, Yahan Yu, Xiuyi Chen, Yonggang Zhang, and Zhen Fang. 2023a. Continual named entity recognition without catastrophic forgetting. arXiv preprint arXiv:2310.14541. Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28.
# Appendix
# A Related Work
<div style="text-align: center;">C Comparison with Existing Incremental Learning Setting</div>
D.1 Technology Domain . . . . . . . .
15
D.2
Economic Domain
. . . . . . . .
15
D.3
Education Domain
. . . . . . . .
16
D.4
Environmental Domain . . . . . .
16
D.5
Cultural Domain
. . . . . . . . .
16
D.6
Health and Medical Domain
. . .
16
E
Concept Selection Criterion
16
E.1
Length criterion . . . . . . . . . .
16
E.2
Timeliness criterion . . . . . . . .
16
E.3
Trend criterion
. . . . . . . . . .
16
F
Experimental Settings
16
F.1
Backbones . . . . . . . . . . . . .
16
F.2
Implementation Details . . . . . .
16
F.3
Input Prompt
. . . . . . . . . . .
17
# G Introduction of Baseline Methods
# H Visualization of One Concept
# I Additional Experimental Results
# A Related Work
We categorize existing studies on understanding the incremental learning ability of LLMs into three parts: (1) Understanding Forgetting, (2) Understanding Memorization, and (3) Applications in NLP.
# A.1 Understanding Forgetting
Earlier studies, such as French (1999); Kirkpatrick et al. (2017), assess catastrophic forgetting by measuring performance degradation on old tasks. Recently, studies (Davari et al., 2022; Wu et al., 2021; Chen et al., 2023; Tao et al., 2023; Zheng et al., 2023b) use probing techniques to measure forgetting in incremental learning. For example, Davari et al. (2022) uses linear probing to reveal representation drift due to parameter updates. Wu et al. (2021) conducts layer-wise probing on BERT, showing catastrophic forgetting in the top and middle layers. Chen et al. (2023) reveals a correlation between retaining past information and new task learning efficiency through linear probing on kshot samples. Tao et al. (2023) illustrates BERT’s resilience to catastrophic forgetting even without buffer data. More recently, Zheng et al. (2023b) reveal that most previous work in NLP overlooks the data leakage issue and uses probing techniques to show that LLMs have superior performance on evaluated datasets even before incremental training. Additionally, they reveal that LLMs have strong anti-forgetting ability even under the sequential fine-tuning setting. Our work is inspired by Zheng et al. (2023b) and proposes a novel dataset to minimize the influence of data leakage issues, allowing for correct conclusions about the incremental learning ability of LLMs.
# A.2 Understanding Memorization
Fewer studies (Carlini et al., 2021; Tirumala et al., 2022; Biderman et al., 2024b; Boix-Adserà et al., 2023) explore memorization within LLMs. For instance, Carlini et al. (2021) discover that GPT-2 can memorize a small proportion of private information during pretraining, raising privacy concerns. Tirumala et al. (2022) show that larger LLMs memorize faster and have higher “forgetting baselines”. Biderman et al. (2024b) reveal the difficulty in predicting which training samples will be memorized by large language models. Boix-Adserà et al. (2023) find that transformers incrementally learn new knowledge, with trained and initial weights progressively increasing in rank. These studies explore memorization from both the perspective of sentences and model weights. In contrast, this paper studies the problem of memorization and forgetting at a more fine-grained level: the concept level. This approach addresses an underexplored research problem in the incre-
mental learning community.
# A.3 Applications in NLP
Many works (Sun et al., 2020; Chuang et al., 2020; Liu and Huang, 2023; Qiu et al., 2024; Zhang et al., 2023a; Shao et al., 2023a; Zhang et al., 2023b) focus on incremental learning for various NLP tasks, assuming catastrophic forgetting in pre-trained language models and designing techniques to mitigate it. These tasks include text classification (Sun et al., 2020; Chuang et al., 2020), relation extraction (Liu and Huang, 2023), named entity recognition (Qiu et al., 2024; Zheng et al., 2022; Zhang et al., 2023a), intent classification (Shao et al., 2023a), and machine translation (Zhang et al., 2023b). We refer readers to the survey (Zheng et al., 2024) for more applications of incremental learning in NLP tasks. The proposed Concept-1K differs substantially from the aforementioned NLP tasks. It is more challenging and provides better explainability at the concept level.
# B Data Leakage in IL of LLMs
The data leakage issue in NLP is often implicit. Unlike computer vision, where pretraining involves explicit category information, NLP pretraining is selfsupervised and lacks clear categorical distinctions that can be easily compared between the pretraining corpus and downstream datasets. This makes it challenging to detect and address data leakage. Therefore, we urge future studies to exercise greater caution regarding data leakage in the IL of LLMs.
# B.1 Data Leakage in Classification Tasks
The issue of data leakage in classification tasks for IL with LLMs has recently been highlighted by (Zheng et al., 2023b). Their extensive study revisits over 20 IL methods across four key classification tasks: Text Classification, Intent Classification, Relation Extraction, and Named Entity Recognition. One core finding of (Zheng et al., 2023b) is that LLMs, such as BERT and GPT-like models, exhibit high probing performance even before they are incrementally trained on specific downstream tasks. This high initial performance suggests that these models already possess substantial knowledge relevant to the classification tasks due to their extensive pre-training on diverse corpora. Consequently, when these LLMs are evaluated under IL settings, the incremental learning of new tasks may, in fact, be leveraging pre-existing knowledge rather than genuinely incremental learning.
This phenomenon leads to misleading conclusions about the effectiveness of various IL methods (Sun et al., 2020; Chuang et al., 2020; Liu and Huang, 2023; Qiu et al., 2024; Zhang et al., 2023a; Shao et al., 2023a; Zhang et al., 2023b). The high probing performance before task-specific training indicates that the models are not learning incrementally as assumed but rather recalling previously acquired knowledge. Therefore, many IL studies in the context of classification tasks suffer from data leakage, as the benchmark tasks are not truly novel to the LLMs. Addressing this issue requires carefully designed benchmarks that ensure the novelty and exclusivity of the knowledge being tested, a challenge we aim to tackle with our Concept-1K dataset.
# B.2 Data Leakage in Generation Tasks
Data leakage poses a significant challenge in IL for generation tasks, which are often more specific and diverse compared to classification tasks. For example, in the IL setting described by Scialom et al. (2022), the task sequence includes Text Simplification, Headline Generation with Constraints, Haiku Generation, Covid QA, Inquisitive Question Generation, Empathetic Dialogue Generation, Explanation Generation, and Twitter Stylometry. Despite the specificity and diversity of these tasks, data leakage remains a concern because LLMs are trained on extensive corpora from the internet and often undergo supervised finetuning on dialogue data (OpenAI, 2023). This pre-training on vast amounts of internet data means that LLMs might already possess significant knowledge relevant to these generation tasks. Moreover, the data leakage issue in generation tasks is implicit and easy to overlook. Unlike classification tasks, where techniques like probing (Zheng et al., 2023b) can measure LLM performance before training, generation tasks lack such straightforward methods to assess initial model capabilities. This makes it challenging to ascertain how much new knowledge is genuinely being learned during IL versus what the model is simply recalling from its pre-trained knowledge base. Pioneering work investigating the IL ability of LLMs found that the T0 model (Sanh et al., 2021) barely forgets when learning new tasks. This suggests that the T0 model may have already acquired the ability to solve multiple generation tasks with appropriate input prompts before explicit training on them. Detailed results are presented in Figure 2
of Sanh et al. (2021). Another study by Zhang et al. (2023c) defines generation tasks using different instructions, a paradigm they call continual instruction tuning. They train the T5 model (Raffel et al., 2020) sequentially on 19 tasks, achieving 35.7% performance, while jointly training on these tasks yields 42.1%. The close gap between upper bound and lower bound performance indicates minimal forgetting and suggests potential data leakage. Detailed results and experimental settings are provided in Tables 1 and 2 of Zhang et al. (2023c). Although recent studies (Module; Guo et al., 2024; Huang et al., 2024; Yang et al., 2024; Peng et al., 2024; Ren et al., 2024) claim that forgetting is serious in continual instruction tuning, all of them utilize parameter-efficient finetuning techniques such as LoRA (Hu et al., 2021) or prompt tuning (Lester et al., 2021). As shown in our experiments in Section 3.3, the IL ability of LoRA and full finetuning differ substantially. LoRA significantly limits the ability to learn new concepts compared to full finetuning, leading to limited new knowledge acquisition and faster forgetting on training samples. Therefore, their IL settings may not accurately reflect the true IL ability of LLMs.
# C Comparison with Existing Incremental Learning Setting
There are three popular IL settings which are widely adopted in the literature of computer vision: class-incremental learning (CIL), task-incremental learning (TIL), and continual pretraining (CPT). However, none of them are appropriate to evaluate the IL ability of LLMs.
# C.1 Class-Incremental Learning
CIL is designed for classification tasks such as text classification, and its goal is to learn new classes incrementally. On the one hand, SOTA LLMs with billion-level parameters are overqualified for the above classification tasks with only dozens of categories. On the other hand, the pretraining corpus is likely to contain the knowledge required for the classification tasks (data leakage issue). As shown empirically by (Zheng et al., 2023b), sequential training frozen LLMs with expanding classifiers yields comparable or even superior performance with SOTA IL methods.
# C.2 Task-Incremental Learning
TIL aims to learn new tasks incrementally (Sun et al., 2020; Qin and Joty, 2022). Apart from the data leakage issue, the diversity of tasks and orders across research makes it difficult to readily and fairly compare IL algorithms.
# C.3 Continual Pretraining
The last scenario CPT aims at continual pretraining models on the corpus from different domains. However, the evaluation relies on the performance of downstream tasks, where we can hardly identify what knowledge is learned or forgotten.
# C.4 Summary
In this paper, we consider a novel IL scenario called Instance-level Incremental Learning (IIL). Unlike the IL scenario mentioned above, IIL regards each concept as an instance and is more practical and challenging for existing LLMs. Specifically, we are motivated by the human learning process and expect LLMs to learn new concepts incrementally without forgetting. For example, in Figure 1, humans can learn new concepts that are constantly emerging, such as “Metaverse” and “Quantum Computing”. After learning more concepts such as “Web3.0” and “Non-Fungible Token”, humans will not immediately forget the previously learned concepts such as “Metaverse”.
# D Introduction of Domains in Concept-1K
The domains in Concept-1K are introduced as follows:
D.1 Technology Domain
# D.1 Technology Domain
This domain focuses on both cutting-edge and widely applied technologies. Cutting-edge technologies include artificial intelligence, blockchain, quantum computing, etc., while widely applied technologies encompass cloud computing, the Internet of Things, and more.
# D.2 Economic Domain
This domain highlights economic trends and emerging economic concepts. Economic trends cover topics such as digital currency and globalization, whereas emerging concepts include quantitative computing, electronic wallets, peer-to-peer (P2P) networks, and others.
# D.3 Education Domain
This domain emphasizes emerging educational technologies and concepts. Technologies such as remote learning and online courses, along with concepts like bilingual education, social education, and lifelong learning, are included.
# D.4 Environmental Domain
This domain centers on global environmental issues and green technologies. Topics include climate change, environmental protection, and sustainable energy, as well as technologies like green roofs, shared bicycles, and solar panels.
# D.5 Cultural Domain
This domain focuses on diversity and inclusion, and digital media and arts. Diversity and inclusion cover multiculturalism, gender equality, and social inclusion, while digital media and arts include digital art, social media trends, and online communities.
# D.6 Health and Medical Domain
This domain is dedicated to emerging medical technologies and public safety. It covers CRISPR gene editing technology, the application of artificial intelligence in medical diagnosis, telemedicine services, wearable health monitoring devices, and concepts related to vaccine development, disease monitoring and prevention strategies, and promoting public health awareness.
# E Concept Selection Criterion
In constructing Concept-1K, we carefully selected concepts based on the following criteria to ensure the relevance, novelty, and richness of the learning material:
# E.1 Length criterion
The concept words should not exceed three words in length. This encourages the model to focus on significant and concise terms within the domain, facilitating efficient learning and ensuring that the concepts provide a rich source of information. Shorter concepts are easier to manage and help avoid potential confusion that may arise from overly complex or verbose terms.
# E.2 Timeliness criterion
The chosen concept words should preferably be those that emerged after January 2022. This criterion ensures that the general pre-trained models
have not yet encountered and learned these concepts and the associated knowledge. By selecting recent concepts, we aim to test the true incremental learning capabilities of LLMs, avoiding biases introduced by prior knowledge.
# E.3 Trend criterion
We focus on concepts that are currently receiving widespread attention in academia, industry, and the media. This ensures that the selected concepts are not only relevant and contemporary but also significant and impactful in their respective fields. By choosing trending concepts, we can better gauge the models’ ability to learn and adapt to the latest advancements and discussions in various domains.
# F Experimental Settings
# F.1 Backbones
We use the Pythia suite (Biderman et al., 2023) and other popular open-source models, including LLaMa and Vicuna, for our experiments. Pythia is based on GPT-NeoX (Black et al., 2022) and includes 8 model sizes and 154 pre-training checkpoints, facilitating research in interpretability and learning dynamics. The statistics of the 9 LLMs used in this paper are summarized in Table 10. We download the pre-trained weights from Huggingface (Wolf et al., 2019).
# F.2 Implementation Details
We sort the concepts alphabetically and shuffle the order using random seed 1. The maximum input and output lengths are set to 32 and 10, respectively. The batch size is 32, and the learning rate for the LLMs is 1 × 10−5. We use the AdamW optimizer (Loshchilov and Hutter, 2018). For LLMs with more than 1B parameters, we use A800 GPUs, while smaller LLMs run on RTX3090 GPUs. Each experiment is repeated three times, and we report the average and standard deviations. Additionally, we search for the best hyper-parameters for each baseline method. For the experiment in Section 3.2, we use “gpt3.5-turbo” and “gpt-4-turbo” for GPT-3.5 and GPT4, respectively. Given the high cost of evaluating the entire Concept-1K dataset, we randomly sample 500 instances for both GPT-3.5 and GPT-4. The outputs and targets of GPT-3.5 and GPT-4 on these 500 instances are provided in the supplementary material.
Model Class
Pretrained Weights
Parameters
Layers
Hidden Dim
Link
GPT-NeoX
Pythia-70m
19M†
6
512
Link
Pythia-160m
85M†
12
768
Link
Pythia-410m
302M†
24
1024
Link
Pythia-1b
805M†
16
2048
Link
Pythia-1.4b
1.21B†
24
2048
Link
Pythia-2.8b
2.52B†
32
2560
Link
LLaMa
llama-7b-hf
7B
32
4096
Link
vicuna-7b-v1.1
7B
32
4096
Link
llama-2-13b-hf
13B
40
5120
Link
# F.3 Input Prompt
We use the following input prompt for training and testing Concept-1K:
where {Question} and {Answer} represent the question and the target output, respectively. For the experiment in Section 3.2, the prompts are provided in Tables 11 and 12.
# G Introduction of Baseline Methods
The introduction of the baseline methods is as follows:
• SEQ: Sequential fine-tuning (SEQ) is considered the lower bound of incremental learning.
• REPLAY: Experience replay stores representative old samples and jointly optimizes both old and new samples when learning new tasks. This is a practical and popular technique in incremental learning.
• LAMOL (Sun et al., 2020): LAMOL trains LLMs with question-answering and generative objectives, generating pseudo-samples
before learning each new task for data replay. The generation loss weight is λ = 0.25, and the proportion of pseudo-samples is γ = 0.20. There are two variants: LAMOL_t and LAMOL_g, differing by whether a taskspecific token is used for generation.
• L2KD (Chuang et al., 2020): L2KD adds a knowledge distillation target based on LAMOL, with the teacher model trained from scratch. We implemented the word-level variant as it performs best on text classification tasks.
 LAMOL_KD (Zheng et al., 2023b): LAMOL_KD utilizes knowledge distillation based on LAMOL_t. Unlike L2KD, the teacher model in LAMOL_KD is trained on all previous tasks. New data are used to learn the LAMOL objectives, and pseudo data are used for word-level knowledge distillation as a regularization term.
• PCLL (Zhao et al., 2022): PCLL combines the concepts of variational autoencoders and word-level knowledge distillation with the objectives of LAMOL.
• LFPT5 (Qin and Joty, 2022): LFPT5 learns only soft prompts for each new task. The training objective is the same as LAMOL. The number of soft prompt tokens is 10.
• LFPT5 (Qin and Joty, 2022): LFPT5 learns only soft prompts for each new task. The training objective is the same as LAMOL. The number of soft prompt tokens is 10.
• LoRA (Hu et al., 2021): LoRA trains a small proportion of parameters of LLMs. We set the rank r = 8 and the scaling parameter α = 16.
• LoRA (Hu et al., 2021): LoRA trains a small proportion of parameters of LLMs. We set the rank r = 8 and the scaling parameter α = 16.
Table 11: Prompt for In-Context Learning with 1 shot demonstration. {Question i} and {Answer i} represent the question and the target output of the i-th demonstration training sample respectively. {Test Question} represents the test question.
I will provide some knowledge as follows:
Question: {Question 1}
Short Answer: {Answer 1}
Please answer the following question according to the above knowledge:
Question: {Test Question}
Short Answer:
Table 12: Prompt for In-Context Learning with 5 shot demonstrations. {Question i} and {Answer i} represent the question and the target output of the i-th demonstration training sample respectively. {Test Question} represents the test question.
I will provide some knowledge as follows:
Question: {Question 1}
Short Answer: {Answer 1}
Question: {Question 2}
Short Answer: {Answer 2}
Question: {Question 3}
Short Answer: Answer 3
Question: {Question 4}
Short Answer: {Answer 4}
Question: {Question 5}
short Answer: {Answer 5}
Please answer the following question according to the above knowledge:
Question: {Test Question}
Short Answer:
We use the LoRA implementation from the PEFT library (Mangrulkar et al., 2022).
Some IL methods are not compared as they are not applicable in the IIL scenario. For example, Progressive Prompt (Razdaibiedina et al., 2023) requires task IDs during both training and inference stages. VAG (Shao et al., 2023b) requires storing the vocabulary of class labels and does not apply to generation tasks without class labels. Additionally, prompt-based IL methods such as L2P (Wang et al., 2022b) are not suitable for generation tasks.
# H Visualization of One Concept
We visualize the memorized and generalized knowledge of the concept “Brain-Computer Interface” in Figures 7 and 8. In each graph, the center node represents the concept “Brain-Computer Interface”. The linked and unlinked edges indicate whether the corresponding training or test samples are answered correctly. For example, in Figure 7a, the edge between “Brain-Computer Interface” and “Signal Processing” signifies that the LLM correctly outputs the target answer “Signal Processing” when the question is the training sample “What subevent occurs in a Brain-Computer Interface?”. In Figure 7b, the edge between “Brain-Computer Interface” and “Signal Processing” is missing, indicating that the LLM fails to provide the correct target answer. Figure 7 shows that even when LLMs can memorize all the knowledge, they tend to first forget more complex knowledge, such as “(BrainComputer Interface, UsedFor, Controlling Computers With Thought)”. Conversely, some common knowledge, such as “(Brain-Computer Interface, Requires, Brain Signals)”, “(Brain-Computer Interface, Has Property, Innovative)”, and “(BrainComputer Interface, Motivated By Goal, Accessibility)”, remains robust and is not forgotten after learning 10 tasks. This indicates that certain knowledge is easier to memorize, generalize, and retain. Further exploration at the concept-level knowledge in IL is left for future work. We also encourage future studies to utilize the provided Concept1K dataset for a fine-grained analysis of the memorization and generalization dynamics in IL.
# I Additional Experimental Results
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/061b/061b55d0-21e5-4d1d-8df3-7e71f3affb2b.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: The visualization of the memorized knowledge related to the “Brain-Computer Interface” in IL. The center node represents a concept, while the linked and unlinked edges indicate whether the corresponding training samples are answered correctly.</div>
<div style="text-align: center;">Table 13: The memorization accuracy when different model scales and buffer size are selected. The pretraining tep is 143000 (final version). Figure 4 (a) summarises this figure’s content.</div>
70M
160M
410M
1B
1.4B
2.8B
Buffer Size=0
6.34±0.86
33.53±1.39
58.28±0.61
65.97±0.31
64.95±0.55
68.03±0.46
Buffer Size=2000
22.87±1.28
56.12±0.81
77.31±0.23
81.57±0.32
80.81±0.62
82.26±0.66
Buffer Size=5000
42.13±1.88
73.32±1.00
90.05±0.64
91.93±0.49
91.49±0.32
92.03±0.35
Buffer Size=10000
58.50±0.50
84.91±1.64
97.81±0.12
97.21±0.73
97.35±0.84
96.74±0.43
Buffer Size=All
62.37±1.82
90.04±0.64
99.01±0.14
99.10±0.57
98.86±0.55
98.72±0.34
<div style="text-align: center;">Table 14: The generalization accuracy when different model scales and buffer size are selected. The pretrainin step is 143000 (final version). Figure 4 (e) summarises this figure’s content.</div>
70M
160M
410M
1B
1.4B
2.8B
Buffer Size=0
2.52±0.12
6.57±0.28
17.69±0.35
22.65±0.39
25.57±0.36
29.56±0.35
Buffer Size=2000
4.04±0.07
8.49±0.23
22.49±0.32
27.29±0.55
30.80±0.28
34.21±0.40
Buffer Size=5000
5.14±0.09
10.08±0.40
25.19±0.46
29.34±0.38
33.50±0.46
37.04±0.36
Buffer Size=10000
5.88±0.10
10.36±0.52
26.20±0.35
30.81±0.24
34.85±0.57
37.72±0.52
Buffer Size=All
5.80±0.08
10.80±0.46
25.70±0.28
30.99±0.42
35.03±0.27
37.97±0.37
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d9b5/d9b5cda4-00d2-47a5-ba32-fe604b937cc2.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: The visualization of the generalized knowledge related to the “Brain-Computer Interface” in IL. The center node represents a concept, while the linked and unlinked edges indicate whether the corresponding test samples are answered correctly.</div>
<div style="text-align: center;">Table 15: The memorization accuracy when different pretraining steps and model scales are selected. The buffe size is 0. Figure 4 (b) summarizes the content of this figure.</div>
Step=0
Step=16
Step=128
Step=1000
Step=10000
Step=143000
Pythia-70M
2.92±0.06
4.89±0.16
4.58±0.10
24.69±0.04
36.08±0.63
6.34±0.86
Pythia-160M
26.33±0.31
29.34±0.25
29.11±0.11
48.78±0.09
60.12±0.34
33.53±1.39
Pythia-410M
29.58±1.10
31.06±0.24
30.78±0.22
49.84±0.06
64.31±0.38
58.28±0.61
Pythia-1B
33.18±0.59
33.71±0.17
33.91±0.49
53.14±0.59
66.66±0.59
65.97±0.31
Pythia-1.4B
35.61±0.59
36.31±0.51
36.15±0.32
55.61±0.59
70.66±0.51
64.95±0.55
Pythia-2.8B
37.76±0.29
37.73±0.38
37.65±0.39
57.09±0.34
72.92±0.59
68.03±0.46
<div style="text-align: center;">Table 16: The generalization accuracy when different pretraining steps and model scales are selected. The buffe size is 0. Figure 4 (f) summarises this figure’s content.</div>
Step=0
Step=16
Step=128
Step=1000
Step=10000
Step=143000
Pythia-70M
0.83±0.02
1.10±0.01
1.08±0.04
2.94±0.04
3.95±0.03
2.52±0.12
Pythia-160M
1.64±0.04
1.81±0.02
1.82±0.06
3.79±0.05
6.96±0.04
6.57±0.28
Pythia-410M
2.34±0.07
2.35±0.04
2.21±0.04
4.18±0.10
12.91±0.14
17.69±0.35
Pythia-1B
2.78±0.14
2.85±0.24
2.80±0.24
6.03±0.24
17.70±0.24
22.65±0.39
Pythia-1.4B
2.71±0.37
2.32±0.24
2.40±0.53
5.68±0.24
20.64±0.24
25.57±0.36
Pythia-2.8B
2.48±0.23
1.79±0.29
2.09±0.32
5.35±0.49
24.79±0.24
29.56±0.35
<div style="text-align: center;">Table 17: The memorization accuracy when different pretraining steps and model scales are selected. The buffe size is 2000. Figure 4 (c) summarises this figure’s content.</div>
Step=0
Step=16
Step=128
Step=1000
Step=10000
Step=143000
Pythia-70M
22.15±0.13
24.91±0.20
23.54±0.15
55.06±0.16
64.89±0.23
22.87±1.28
Pythia-160M
59.42±0.13
59.62±0.77
59.30±0.18
78.84±0.44
82.30±0.18
56.12±0.81
Pythia-410M
60.43±0.59
61.09±0.66
61.24±0.27
79.77±0.15
83.24±0.12
77.31±0.23
Pythia-1B
63.11±0.15
62.05±0.40
63.51±0.31
80.11±0.50
83.11±0.38
81.57±0.32
Pythia-1.4B
63.77±0.15
64.14±0.23
65.40±0.32
82.38±0.34
84.19±0.22
80.81±0.62
Pythia-2.8B
65.10±0.45
65.84±0.57
66.04±0.53
82.63±0.18
84.80±0.44
82.26±0.66
<div style="text-align: center;">Table 18: The generalization accuracy when different pretraining steps and model scales are selected. The buffe size is 2000. Figure 4 (g) summarises this figure’s content.</div>
Step=0
Step=16
Step=128
Step=1000
Step=10000
Step=143000
Pythia-70M
2.07±0.08
2.46±0.06
2.51±0.13
3.84±0.06
4.57±0.07
4.04±0.07
Pythia-160M
3.17±0.08
3.55±0.14
3.52±0.06
5.26±0.14
8.05±0.08
8.49±0.23
Pythia-410M
3.96±0.06
4.97±0.09
5.09±0.04
6.