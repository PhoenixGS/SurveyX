# Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning
Sungwon Han 1 Jinsung Yoon 2 Sercan ¨O. Arik 2 Tomas Pfister 2
# Abstract
Large Language Models (LLMs), with their remarkable ability to tackle challenging and unseen reasoning problems, hold immense potential for tabular learning, that is vital for many real-world applications. In this paper, we propose a novel in-context learning framework, FeatLLM, which employs LLMs as feature engineers to produce an input data set that is optimally suited for tabular predictions. The generated features are used to infer class likelihood with a simple downstream machine learning model, such as linear regression and yields high performance few-shot learning. The proposed FeatLLM framework only uses this simple predictive model with the discovered features at inference time. Compared to existing LLM-based approaches, FeatLLM eliminates the need to send queries to the LLM for each sample at inference time. Moreover, it merely requires API-level access to LLMs, and overcomes prompt size limitations. As demonstrated across numerous tabular datasets from a wide range of domains, FeatLLM generates high-quality rules, significantly (10% on average) outperforming alternatives such as TabLLM and STUNT.
arXiv:2404.09491v2
# 1. Introduction
In recent years, Large Language Models (LLMs) have shown impressive abilities for generating suitable responses for previously unseen tasks without requiring task-specific fine-tuning (Creswell et al., 2022; Imani et al., 2023; Taylor et al., 2022). Trained on extensive text corpora, the rich prior knowledge of LLMs can substitute the need for expert domain knowledge in many areas (Gen, 2021; Singhal et al.,
1Work done at Google as a research intern. School of Computing, Korea Advanced Institute of Science and Technology, Daejeon, Republic of Korea 2Google Cloud AI, Sunnyvale, California, USA. Correspondence to: Jinsung Yoon <jinsungyoon@google.com>.
1Work done at Google as a research intern. School of Computing, Korea Advanced Institute of Science and Technology, Daejeon, Republic of Korea 2Google Cloud AI, Sunnyvale, California, USA. Correspondence to: Jinsung Yoon <jinsungyoon@google.com>.
Proceedings of the 41 st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).
Proceedings of the 41 st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).
2023; Wilcox & Hripcsak, 2003), playing a crucial role in the automation of diverse applications, including dialogue analysis (Finch et al., 2023) and program repair (Fan et al., 2023). Particularly, by incorporating a task description and few examples into prompts, LLMs can understand the context of a task and “attend” to important features and examples (Brown et al., 2020; Zhao et al., 2023). This demonstrates the capability of LLMs to analyze data, highlighting their potential to perform as an expert feature engineer.
2023; Wilcox & Hripcsak, 2003), playing a crucial role in the automation of diverse applications, including dialogue analysis (Finch et al., 2023) and program repair (Fan et al., 2023). Particularly, by incorporating a task description and few examples into prompts, LLMs can understand the context of a task and “attend” to important features and examples (Brown et al., 2020; Zhao et al., 2023). This demonstrates the capability of LLMs to analyze data, highlighting their potential to perform as an expert feature engineer. Utilization of LLMs’ prior knowledge and reasoning capabilities have been extended to the domain of tabular learning. For instance, knowledge such as related to the increased risk of certain diseases in older individuals can be beneficial at disease prediction tasks. In this context, recent works define task and feature descriptions in natural language, serialize the data, and then feed it into an LLM (Dinh et al., 2022; Wang et al., 2023). This is often followed by either using incontext learning (Nam et al., 2023a) or parameter-efficient fine-tuning (Dinh et al., 2022; Hegselmann et al., 2023). Prior knowledge provided by LLMs has proven effective, outperforming conventional tabular learning baselines in low-shot regimes (Hegselmann et al., 2023). Current LLM-based tabular learning methods have some limitations. For end-to-end predictions, at least one LLM inference per sample is required, making it computationally expensive. Most methods require fine-tuning the LLM for high accuracy, making them infeasible to the application to the LLMs that do not come with accessible training codebases or tuning APIs, a prominent concern given that recently-proposed top-performance LLMs only permit limited access via inference APIs (Anil et al., 2023; OpenAI, 2023). In addition, most approaches are not suitable with lengthy prompts (Liu et al., 2023) – when the number of features in tabular data grows to the point where the serialized text length becomes long, the approach often turns infeasible or experiences a decline in performance. These constraints pose fundamental challenges to the real-world application of LLM-based tabular learning models. Our proposed approach goes beyond treating LLMs to yield end-to-end predictions. Instead, we assign them the role of feature engineering, with them proposing a more efficient way to utilize their prior knowledge. Specifically, we
introduce a novel in-context learning framework, FeatLLM, which aims to understand the “criteria” underlying LLM predictions. For instance, for the task of predicting a particular disease, the LLM can directly infer and generate rules that determine which feature conditions result in identifying the disease. The criteria (or rules) are used to create new features that can replace existing ones in solving tasks. This approach shifts the capability of the LLM to feature generation, yielding a simpler downstream model – once the features are discovered, a complex machine learning model is no longer needed for inference, improving the inference latency. By leveraging the in-context learning ability of LLMs, it’s possible to extract features without the need for model training, by adding example demonstrations to the prompts. Furthermore, by employing techniques from ensemble methods (Breiman, 2001), like feature bagging, we can mitigate the challenge of excessive prompt size. FeatLLM structures the input prompt for engineering new features into the following two sub-tasks: (1) understanding the problem and inferring the relationship between features and targets; and (2) based on the knowledge from the previous step and few-shot examples, deriving decisive rules to distinguish classes. This structured reasoning approach leads the LLMs to filter out uninformative features for rule extraction. The deduced rules are then applied to the data (parsed them as program code), and the data are transformed into a binary feature indicating whether or not the samples adheres to each rule. Finally, a model with low complexity is fitted to estimate class probabilities from these new features. The robustness is further improved by employing an ensemble approach, repeatedly generating features in conjunction with feature bagging. By appropriately adjusting the number of features and samples chosen in bagging, the challenge of excessive prompt size is mitigated. FeatLLM is free from training constraints and yields low inference cost, rendering the application of the LLM feasible for various real-world problems. We evaluate FeatLLM on 13 different tabular datasets in low-shot regimes, showing its strong and robust performance. We show that LLMs proficiently utilize both prior knowledge and the knowledge from the data, extracting high-quality features. Our framework outperforms contemporary few-shot learning baselines across various settings. The code is released via anonymized GitHub link at https: //github.com/Sungwon-Han/FeatLLM.
# 2. Related Work
# 2.1. Few-Shot Learning with Tabular Data
The advancement of few-shot learning has enabled the extraction of generalizable representations from data, even with minimal labeling costs (Chen et al., 2018). Initially
its strong generalizability across various data modalities including text, audio, and more recently, tabular data (Majumder et al., 2022; Nam et al., 2023b; Schick & Sch¨utze, 2021). Learning from a small number of labeled samples presents the risk of learning spurious correlations (Bartlett et al., 2021). Consequently, recent efforts have employed additional sources of information or introduced domainspecific prior knowledge to impart appropriate inductive biases for model training. For example, utilizing large quantities of more readily-available unlabeled datasets with judicious augmentation for semi-supervised learning (Ucar et al., 2021; Yoon et al., 2020), or seeking good model initialization through task-agnostic unsupervised pre-training (Somepalli et al., 2021). Such unsupervised objectives can involve masking or corrupting part of the data and then using a reconstruction objective (Arik & Pfister, 2021; Majmundar et al., 2022) or employing data augmentation to apply contrastive learning objectives (Bahri et al., 2022; Chen et al., 2020). However, the heterogeneous nature of tabular data makes it challenging to find universally applicable augmentations, and these strategies have not been notably effective in few-shot settings (Nam et al., 2023b). Recent work have proposed training models on a broad range of tabular datasets beyond those related to the target task and then applying transfer learning to the target task (Zhang et al., 2023; Levin et al., 2022; Zhu et al., 2023). For instance, TransTab (Wang & Sun, 2022) trains transformer-based models learning of semantic relationships between columns through column names, not just table values. The knowledge extracted from various tables and columns has facilitated zero-shot or few-shot inference for new tabular tasks. Meanwhile, TabPFN (Hollmann et al., 2023) mixes different types of distributions to create synthetic data for pre-training a Bayesian neural network. After pre-training, inference is performed by inputting both training and test samples of the target task without further training. The prior knowledge gained from various datasets helped surpass the performance of traditional tree-based approaches and is proved to be beneficial in few-shot settings.
# 2.2. Language-Interfaced Tabular Learning
Another promising approach to leverage prior knowledge is the incorporation of large language models (LLMs) (Anil et al., 2023; OpenAI, 2023). LLMs, trained on extensive and diverse text corpora, have demonstrated the ability to generalize to unseen tasks, proving their utility across various domains (Brown et al., 2020). Recent studies have attempted to harness the rich prior knowledge encapsulated by LLMs for tabular learning. They typically propose serializing tabular data into text and feeding them into the input prompts of LLMs (Dinh et al., 2022; Hegselmann et al., 2023; Wang et al., 2023). By including tabular data, fea-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fae0/fae002d7-108e-4224-8d19-83d152d44629.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Illustration of FeatLLM. FeatLLM extracts rules for each class, utilizing prior knowledge and few-shot examples These rules are then parsed and applied to create binary features for data samples. A linear layer is trained on these binary features to estimate class likelihoods. This procedure is repeated multiple times for ensembling.</div>
ture descriptions, and task descriptions in the input prompts, these can infer the relationships between tasks and features to perform inference. Research has evolved to either specifically train models using parameter-efficient tuning techniques like LoRA (Hu et al., 2021) or IA3 (Liu et al., 2022), or employ in-context learning by adding few-shot example demonstrations to the prompts without training (Dinh et al., 2022; Hegselmann et al., 2023; Nam et al., 2023a; Slack & Singh, 2023).
ture descriptions, and task descriptions in the input prompts, these can infer the relationships between tasks and features to perform inference. Research has evolved to either specifically train models using parameter-efficient tuning techniques like LoRA (Hu et al., 2021) or IA3 (Liu et al., 2022), or employ in-context learning by adding few-shot example demonstrations to the prompts without training (Dinh et al., 2022; Hegselmann et al., 2023; Nam et al., 2023a; Slack & Singh, 2023). Despite the effectiveness of the aforementioned methods in improving few-shot tabular learning, the requirement to always pass inference through the LLM poses challenges for application in industrial domains. This study aims to depart from the conventional black-box approach of processing each sample’s prediction in an end-to-end fashion via an LLM. Instead, it targets the LLM to directly infer the conditions or rules associated with each class. FeatLLM parses the generated rules into the program code to create new features, enabling predictions without the need to pass through the LLM and leveraging in-context learning to extract rules utilizing prior knowledge and information from few-shot samples.
Despite the effectiveness of the aforementioned methods in improving few-shot tabular learning, the requirement to always pass inference through the LLM poses challenges for application in industrial domains. This study aims to depart from the conventional black-box approach of processing each sample’s prediction in an end-to-end fashion via an LLM. Instead, it targets the LLM to directly infer the conditions or rules associated with each class. FeatLLM parses the generated rules into the program code to create new features, enabling predictions without the need to pass through the LLM and leveraging in-context learning to extract rules utilizing prior knowledge and information from few-shot samples.
# 3. Methods
FeatLLM is designed to extract “rules”, the conditions associated with each answer class, from the few samples it inputs, as depicted in Figure 1. The extracted rules are parsed through an LLM, and then used to create new binary features of the given sample indicating whether or not each rule is satisfied. The set of new binary features are input to a dot product operation with non-negative trainable weights, to estimate the likelihood of each class. By training this model, we learn the importance of each feature and how it affects the class likelihood (Section 3.2). This process is repeated multiple times with bagging and then combined via ensembling. Problem formulation and details of each stage are explained below.
Problem formulation. Let’s consider a tabular dataset with N labeled samples D = {(xi, yi)}N i=1. The dataset D consists of d input features (i.e., each xi is a d-dimensional vector), and comes with natural language feature names, denoted by F = {fj}d j=1, along with their definitions provided separately. Assuming the classification setup, the label yi is defined as one of the predefined set of classes. For the k-shot learning experiments, only k (< N) labeled samples are randomly sampled to train the model.
# 3.1. Prompt Design for Extracting Rules
To enable the LLM to extract rules based on a more accurate reasoning path, we guide the problem-solving process to mimic how an expert human might approach a tabular learning task. Our prompt comprises three main components as follows (shown in Fig. 2 and Appendix A.2):
Basic information description. This part provides essential information for solving the problem (see orange text in Fig. 2). It includes a description of the task to be solved with label information, descriptions of features, and example demonstrations. The task description is formulated as a question (e.g., “Does this patient’s myocardial infarction complications data indicate chronic heart failure? Yes or no?”. More examples are provided in Appendix A.1). The feature description indicates its value type and optionally includes its definition. For categorical variables, examples of the possible categories can be provided. As example demonstrations, few training samples are serialized into text, along with their ground-truth labels. Serialization follows the format used in previous studies (Dinh et al., 2022; Hegselmann et al., 2023):
(1)
Reasoning instruction. We aim to enhance the LLM’s reasoning by providing guidance through reasoning instruc-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7f83/7f8354f4-834c-4e09-8a60-f620095e82db.png" style="width: 50%;"></div>
Figure 2: Prompt for rule extraction. Text in orange provides basic information description; blue text outlines reasoning instruction; and yellow text details response instruction.
tions (see blue text in Fig. 2). The reasoning instruction begins with an introductory sentence similar to the chainof-thought approach (Wei et al., 2022b). Then, we divide the LLM’s inference process into two steps. In the first step, the LLM is encouraged to infer the causal relationship or tendency between features and the task description. This is done without referring to example demonstrations, relying instead on general knowledge or common sense, allowing the LLM to fully extract prior knowledge about the problem. In the second step, the LLM uses example demonstrations and the information from the first step to deduce rules for each class. This two-step reasoning process prevents the model from identifying spurious correlations in irrelevant columns and can assist in focusing on more significant features. To prevent extracting too few or too many rules, potentially leading to either underfitting or creating an
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2b7b/2b7bb08c-8752-4f4e-8976-af940483a227.png" style="width: 50%;"></div>
Figure 3: Prompt for parsing rules. This prompt incorporates the rules generated in the previous stage, placed within the <Conditions> section.
# 3.2. Inferring Class Likelihood via Rules
Parsing rules for feature generation. We utilize the rules generated in the previous stage to create new binary features. These features are created for each class, indicating whether the sample satisfies the rules associated with that class (i.e., either ’0’ or ’1’). They can be used in predicting how likely a sample belongs to each class. However, since the rules generated by the LLM are based on natural language, designing a parsing strategy is required for automatic feature generation. Even though we constrain formatting of responses and rules during rule generation for easier parsing, the LLM’s output can include noisy syntactic changes (Kovaˇc et al., 2023). For instance, the LLM might alter expressions, like
substituting “>” or “<” with phrases like “greater than” or “less than.” This variability makes parsing more challenging. To address the challenges of parsing noisy text, instead of building complex program code, we leverage the LLM itself. The LLM is good at understanding the semantic meaning despite syntactic changes in text, and it has the ability to generate a simple code with instructions (given it’s trained on code data corpus). To leverage these, we include the function name, input and output descriptions, and inferred rules in the prompt, then input them into the LLM. Figures 3 and A.4-A.5 in Appendix show the prompt examples and the corresponding outcomes. The generated code is executed using Python’s exec() function to perform data conversion.
substituting “>” or “<” with phrases like “greater than” or “less than.” This variability makes parsing more challenging. To address the challenges of parsing noisy text, instead of building complex program code, we leverage the LLM itself. The LLM is good at understanding the semantic meaning despite syntactic changes in text, and it has the ability to generate a simple code with instructions (given it’s trained on code data corpus). To leverage these, we include the function name, input and output descriptions, and inferred rules in the prompt, then input them into the LLM. Figures 3 and A.4-A.5 in Appendix show the prompt examples and the corresponding outcomes. The generated code is executed using Python’s exec() function to perform data conversion. Inferring class likelihood. Given new binary features from rules for each class, a simple method to measure the class likelihood of the sample is to count how many rules of each class it satisfies (i.e., the sum of the binary feature per class). However, not all rules and features carry the same importance, necessitating learning their significance from training samples. We aim to learn this importance using a conventional machine learning (ML) model, applied to each class’s binary feature. For example, we can consider a linear model without bias. Let us denote the generated binary feature from sample xi for class k as zi k ∈{0, 1}R, where R is the number of rules per each class and c is the number of classes. The probability for each class pi is computed as follows:
Inferring class likelihood. Given new binary features from rules for each class, a simple method to measure the class likelihood of the sample is to count how many rules of each class it satisfies (i.e., the sum of the binary feature per class). However, not all rules and features carry the same importance, necessitating learning their significance from training samples. We aim to learn this importance using a conventional machine learning (ML) model, applied to each class’s binary feature. For example, we can consider a linear model without bias. Let us denote the generated binary feature from sample xi for class k as zi k ∈{0, 1}R, where R is the number of rules per each class and c is the number of classes. The probability for each class pi is computed as follows:
(2)
Here, wk represent the trainable weight in the linear model and they signify the importance of each feature. By projecting these weights into a positive subspace, we can ensure that the features generated by the LLM do not reduce the likelihood of a class during learning. The logit vector is then converted into class probabilities through the softmax function. We optimize the cross-entropy loss to learn the weight wk using a few-shot training samples. Ensembling with bagging. Finally, we repeatedly execute the entire process to create multiple inference models to make the final prediction via ensemble. Given the collection of trained weights from T trials, denoted as {w[t]}T t=1, the prediction is made by averaging the probability for each class pi computed with each weight w[t] as in Eq. 2. To introduce randomness in rule generation for ensemble, a sufficiently high temperature is set for LLM inference. We also randomize the order of few-shot demonstrations based on the observation that the order can impact the query outcome of LLM (Lu et al., 2022)2. To utilize the diversity in prediction for more robust predictions, we adopt bagging to select a subset of features or instances for each trial,
which gets particularly useful for larger number of training samples or features. The proposed ensemble approach offers two main advantages. First, even if the LLM generates incorrect rules in some trials, the correct rules from other trials can compensate, enhancing the overall robustness of the framework. This ties into the LLM’s self-consistency (Wang et al., 2022), as rules commonly inferred across multiple trials are more likely to be accurate. Second, the ensemble method addresses the limitation of LLM’s prompt size. When the tabular data exceed the allowable input prompt size, bagging helps reduce it, ensuring the model’s robust performance. While a single rule set might not capture the relationships of all features, the ensemble of various weak learners created across multiple trials can nullify the impact of incomplete feature/instance coverage in individual trials.
# 4. Experiments
We evaluate FeatLLM over multiple tabular datasets, and conduct various analyses to investigate how our framework works. Due to space constraints, additional experiments, such as varying LLMs, qualitative analysis, and others, are described in Appendix.
# 4.1. Performance Evaluation
Datasets. Our experiment utilizes 13 datasets for binary or multi-class classification tasks: (1) Adult (Asuncion & Newman, 2007) for predicting whether an individual earns over $50,000 annually; (2) Bank (Moro et al., 2014) for predicting whether a customer will subscribe to a term deposit; (3) Blood (Yeh et al., 2009) for predicting whether donors will return for subsequent donations; (4) Car (Kadra et al., 2021) for predicting the quality of a car; (5) Communities (Redmond, 2009) for predicting the crime rate in a specified region; (6) Credit-g (Kadra et al., 2021) for predicting whether an individual poses a good or bad credit risk; (7) Diabetes3 for predicting the presence of diabetes in an individual; (8) Heart4 for predicting the occurrence of coronary artery disease in an individual; and (9) Myocardial (Golovenkin & Voino-Yasenetsky, 2020) for predicting whether an individual suffers from chronic heart failure. We also include more recent datasets and synthetic datasets which have been less frequently discussed and were not included in the pre-training process of LLMs: (10) Cultivars for predicting how high the grain yield of this soybean cultivar will be5; (11) NHANES for predicting the given person’s age group from the record6; (12) Sequence-type for
3kaggle.com/datasets/uciml/ pima-indians-diabetes-database 4kaggle.com/datasets/fedesoriano/ heart-failure-prediction 5archive.ics.uci.edu/dataset/913 6archive.ics.uci.edu/dataset/887
<div style="text-align: center;">Table 1: Evaluation results, showing AUC across 11 datasets with attributes fewer than 100, are presented. Best performance are bolded, and our framework’s performances, when second-best, are underlined. Results with more training shots (e.g., 32 64) and an additional baseline (e.g., RandomForest) are provided in Appendix K.</div>
Data
Shot
LogReg
XGBoost
SCARF
TabPFN
STUNT
In-context
TABLET
TabLLM
Ours
Adult
4
72.10±12.30
50.00±0.00
58.34±15.42 60.89±23.28 67.43±29.61
77.51±5.24
75.29±12.24
83.57±2.69
86.68±0.86
8
76.02±3.37
59.19±6.92
72.42±8.95
70.42±9.96
82.16±6.93
79.30±2.89
77.56±7.56
83.52±4.30
87.89±0.06
16
75.20±5.10
60.68±13.92
75.63±9.56
70.34±9.96
80.57±10.93
79.50±4.57
79.74±5.64
83.23±2.45
87.54±0.50
Bank
4
63.70±3.87
50.00±0.00
58.53±5.49
63.19±11.60 56.34±12.82
61.38±1.30
58.11±6.29
62.51±8.95
70.45±3.69
8
72.52±3.21
58.78±10.54 55.28±11.88
62.81±7.84
63.01±8.78
69.57±13.35
69.08±6.00
63.19±5.79
75.85±6.66
16
77.51±3.09
70.34±5.86
65.81±1.79
73.79±2.21
69.85±0.95
69.76±8.55
69.40±11.28
63.73±6.43
78.41±1.08
Blood
4
56.79±26.02
50.00±0.00
56.22±21.00 58.72±19.16
48.57±6.04
56.30±12.43 56.45±15.45 55.87±13.49
68.34±7.48
8
68.51±5.16
59.97±1.36
65.77±5.00
66.30±10.01
60.00±4.84
58.99±10.12 56.37±11.56
66.01±9.25
70.37±3.23
16
68.30±6.16
63.28±7.62
66.27±5.04
64.14±6.80
54.76±4.53
56.59±5.21
60.62±4.13
65.14±7.55
70.07±5.19
Car
4
62.38±4.13
50.00±0.00
62.52±3.80
58.14±4.15
61.32±3.83
62.47±2.47
60.21±4.81
85.82±3.65
72.69±1.52
8
72.05±1.20
64.00±3.57
72.23±2.59
63.95±4.35
67.86±0.49
67.57±3.44
65.53±8.00
87.43±2.56
73.26±1.46
16
82.42±4.13
72.26±4.43
75.77±2.71
71.35±5.33
75.56±2.88
76.94±3.04
74.02±1.01
88.65±2.63
79.43±1.24
Credit-g
4
52.68±4.46
50.00±0.00
48.92±4.60
54.00±7.34
48.80±6.76
52.99±4.08
54.33±6.54
51.90±9.40
55.94±1.10
8
55.52±8.88
52.22±4.90
55.26±3.92
52.58±11.27
54.50±8.25
52.43±4.36
52.90±5.79
56.42±12.89
57.42±3.10
16
58.26±5.17
56.23±4.37
59.22±11.38
58.91±8.04
57.63±7.58
55.29±4.80
51.65±4.02
60.38±14.03
56.60±2.22
Diabetes
4
57.09±18.84
50.00±0.00
62.35±7.48
56.28±13.01
64.22±6.78
71.71±5.31
63.96±3.32
70.42±3.69
80.28±0.75
8
65.52±13.18 50.86±22.03 64.69±13.33
69.08±9.68
67.39±12.92
72.21±2.07
65.47±3.95
64.30±5.88
79.38±1.66
16
73.44±0.52
65.69±6.54
71.86±3.16
73.69±3.21
73.79±6.48
71.64±5.05
66.71±0.76
67.34±2.79
80.15±1.35
Heart
4
70.54±3.83
50.00±0.00
59.38±3.42
67.33±15.29
88.27±3.32
60.76±4.00
68.19±11.17
59.74±4.49
75.66±4.59
8
78.12±10.59
55.88±3.98
74.35±6.93
77.89±2.34
88.78±2.38
65.46±3.77
69.85±10.82
70.14±7.91
79.46±2.16
16
83.02±3.70
78.62±7.14
83.66±5.91
81.45±5.05
89.13±2.10
67.00±7.83
68.39±11.73
81.72±3.92
83.71±1.88
Cultivars
4
53.45±10.79
50.00±0.00
46.99±6.33
49.80±15.90
57.10±8.66
51.38±2.48
54.28±3.73
54.39±5.61
55.63±5.24
8
56.22±11.87
52.60±6.31
51.76±9.99
54.72±9.35
57.26±9.52
51.68±4.43
51.48±3.85
52.86±6.13
56.97±5.08
16
60.35±4.23
56.87±2.50
57.06±9.27
54.92±8.32
60.09±7.64
54.31±6.12
57.44±3.53
56.97±2.22
57.19±5.30
NHANES
4
91.96±7.02
50.00±0.00
51.58±4.66
80.74±3.89
69.32±19.59
91.84±3.79
93.54±4.20
99.49±0.23
92.20±1.71
8
92.38±8.21
92.92±4.56
55.07±4.79
85.10±5.88
68.56±18.35
86.67±5.49
94.25±3.35
100.00±0.00
93.29±7.01
16
94.12±3.88
94.52±6.28
89.78±6.77
95.54±0.82
68.62±19.81
93.33±4.47
95.02±1.57
100.00±0.00
95.64±4.67
Sequence
4
66.80±4.98
50.00±0.00
67.08±2.75
71.93±4.17
60.12±6.51
93.01±1.84
91.00±4.25
75.38±5.21
98.60±0.66
-type
8
72.55±4.37
61.01±10.96
60.53±6.91
79.55±6.71
60.58±9.58
94.29±1.64
90.18±3.73
79.91±2.13
98.70±1.44
16
83.10±6.86
80.31±5.79
73.14±5.29
89.59±2.33
65.12±5.28
93.46±0.72
92.09±2.90
86.97±1.97
99.29±0.78
Solution
4
72.71±1.96
50.00±0.00
68.48±14.97
71.19±3.90
64.52±8.41
73.53±1.94
80.26±5.11
73.75±3.90
100.00±0.00
-mix
8
82.91±7.74
58.31±19.17
56.67±7.64
82.42±3.72
72.04±10.34
76.34±4.26
83.19±3.70
76.57±4.80
99.81±0.08
16
83.87±1.90
68.09±1.76
70.37±13.96
84.39±0.83
71.82±8.79
82.15±5.01
88.10±4.72
84.28±5.15
99.67±0.13
Average
4
65.47
50.00
58.22
62.93
62.36
68.44
68.69
70.26
77.86
8
72.03
60.52
62.18
69.53
67.47
70.41
70.53
72.76
79.31
16
76.33
69.72
71.69
74.37
69.72
72.72
73.02
76.22
80.70
classifying a given sequence as one of four types: arithmetic, geometric, Fibonacci, or Collatz; (13) Solution-mix for predicting whether the percentage concentration of a mixture from four solutions exceeds 0.5. Former two datasets were recently donated to the UCI Machine Learning Repository after September 2023, ensuring they were not part of the pre-training for the LLM API (gpt-3.5-turbo-0613) used by our model. The latter two are synthetic datasets which ensures they could not be memorized by the LLM API and proceeded with the evaluation. These datasets vary in size and complexity, as detailed in Table 2. Each dataset provides a clear name and description for each attribute. Datasets like ‘Communities’ and
‘Myocardial’ include over 100 attributes, posing challenges due to limited context window size in LLM.
Baselines. We conduct comparisons with nine baselines. The first three baselines are conventional tabular learning approaches, including (1) Logistic regression (LogReg), (2) XGBoost (Chen & Guestrin, 2016), and (3) RandomForest (Ho, 1995). The next two baselines assume the use of unlabeled datasets, including (4) SCARF (Bahri et al., 2022) and (5) STUNT (Nam et al., 2023b). Note that obtaining a large quantity of unlabeled dataset may be infeasible in real-world applications. Another recent baseline is (6) TabPFN (Hollmann et al., 2023), which generates a synthetic dataset with diverse distribution to pretrain the model. The final three baselines utilize LLM, including
Table 2: Basic information of each dataset used in our experiments. The numbers in parentheses indicate the count of categorical and numerical attributes, respectively.
Data
# of samples
# of features
Label ratio (%)
Adult
48842
14 (7/7)
76:24
Bank
45211
16 (8/8)
88:12
Blood
748
4 (0/4)
76:24
Car
1728
6 (5/1)
70:22:4:4
Communities
1994
103 (1/102)
34:33:33
Credit-g
1000
20 (12/8)
70:30
Diabetes
768
8 (0/8)
65:35
Heart
918
11 (4/7)
45:55
Myocardial
1700
111 (94/17)
22:78
Cultivars
320
10 (3/7)
50:50
NHANES
6287
8 (1/7)
84:16
Sequence-type
250
5 (0/5)
40:40:10:10
Solution-mix
300
8 (0/8)
52:48
(7) In-context learning (In-context) (Wei et al., 2022a), (8) TABLET (Slack & Singh, 2023), and (9) TabLLM (Hegselmann et al., 2023). In-context learning directly embeds few-shot training samples into the input prompt without tuning. TABLET further incorporates additional hints into the prompt through an external classifier via rule-sets and prototypes to enhance the quality of inference. TabLLM employs parameter-efficient tuning technique like IA3 (Liu et al., 2022) to train the LLM with few-shot samples. Implementation details. FeatLLM uses GPT-3.5 as its LLM backbone, yet is designed to be agnostic to the choice of LLMs (see Appendix H for results with different backbones). The temperature for the LLM inference is set to 0.5 and the top-p value is set to the default value of 1 in API. We set the number of ensembles and the number of rules for extracting to 20 and 10 respectively. Details on hyper-parameter impacts are in Figure 6 and Appendix B. We use the Adam optimizer of a learning rate 0.01 for the linear model, training for 200 epochs. We employ k-fold cross-validation for optimal epoch selection. In replicating baselines, GPT-3.5 is utilized for in-context learning, whereas T0 (Sanh et al., 2022) is employed in the TabLLM, following the original setting. Note that TabLLM restricts the use of LLM to only those with publicly available checkpoints; therefore, GPT-3.5 is not accessible in this context. Regarding traditional machine learning methods, such as LogReg, XGBoost, and RandomForest, we determined their parameters using Grid-search combined with k-fold cross validation. The value of k is set either as 2 or 4, guaranteeing that the training set includes at least one example of each class. For more detailed information on implementations, please refer to Appendix C. Results. In Tables 1 and 3, performance comparisons across a range of datasets are presented. We repeat experiments three times with different randomness on the choice
(7) In-context learning (In-context) (Wei et al., 2022a), (8) TABLET (Slack & Singh, 2023), and (9) TabLLM (Hegselmann et al., 2023). In-context learning directly embeds few-shot training samples into the input prompt without tuning. TABLET further incorporates additional hints into the prompt through an external classifier via rule-sets and prototypes to enhance the quality of inference. TabLLM employs parameter-efficient tuning technique like IA3 (Liu et al., 2022) to train the LLM with few-shot samples.
its LLM backbone, yet is designed to be agnostic to the choice of LLMs (see Appendix H for results with different backbones). The temperature for the LLM inference is set to 0.5 and the top-p value is set to the default value of 1 in API. We set the number of ensembles and the number of rules for extracting to 20 and 10 respectively. Details on hyper-parameter impacts are in Figure 6 and Appendix B. We use the Adam optimizer of a learning rate 0.01 for the linear model, training for 200 epochs. We employ k-fold cross-validation for optimal epoch selection. In replicating baselines, GPT-3.5 is utilized for in-context learning, whereas T0 (Sanh et al., 2022) is employed in the TabLLM, following the original setting. Note that TabLLM restricts the use of LLM to only those with publicly available checkpoints; therefore, GPT-3.5 is not accessible in this context. Regarding traditional machine learning methods, such as LogReg, XGBoost, and RandomForest, we determined their parameters using Grid-search combined with k-fold cross validation. The value of k is set either as 2 or 4, guaranteeing that the training set includes at least one example of each class. For more detailed information on implementations, please refer to Appendix C. Results. In Tables 1 and 3, performance comparisons
Results. In Tables 1 and 3, performance comparisons across a range of datasets are presented. We repeat experiments three times with different randomness on the choice
Table 3: Evaluation results with AUC across two datasets with over 100 attributes. TabPFN and two LLM-based baselines are excluded in the table as they failed to make inference over the dataset with a large number of features.
Communities
Shots
4
8
16
LogReg
67.45±13.26
73.73±5.45
72.55±4.83
XGBoost
53.94±4.19
66.65±4.50
68.01±1.97
RandomForest
66.09±10.52
71.16±4.61
71.66±4.81
SCARF
66.18±9.13
72.69±3.79
73.09±2.84
STUNT
66.87±14.10
76.36±4.55
77.29±2.56
FeatLLM
75.39±5.05
76.59±1.25
76.25±0.64
Myocardial
Shots
4
8
16
LogReg
51.25±3.85
55.34±1.11
60.00±5.16
XGBoost
50.00±0.00
55.63±2.92
56.55±12.22
RandomForest
51.91±4.49
52.77±5.83
54.16±4.53
SCARF
47.70±4.10
49.37±3.41
54.31±1.42
STUNT
52.77±2.01
55.40±4.41
61.22±3.45
FeatLLM
52.87±3.44
56.22±1.64
55.32±9.15
of training/test sets, and present the average and standard deviation of the AUC values. FeatLLM consistently ranks as the top performer or secures the second place when compared with other baseline models. Our framework, when utilizing just 4 shots, achieves comparable performance to conventional tabular baselines that require 32 shots (see Fig. 4). Notably, while STUNT and TabPFN demonstrated significant performance improvements on certain datasets, their overall enhancement over conventional machine learning models was not substantial. Moreover, models utilizing LLM, such as in-context learning, TABLET, or TabLLM, failed to infer on tabular datasets with a large number of features. Our framework, employing both bagging and ensemble techniques, proved to be effective even on datasets with numerous features, exhibiting satisfactory performance. It is important to note that our bagging and ensemble concept could be adapted to existing LLM-based approaches. However, such an adaptation would double the number of per-sample inference, leading to a substantial increase in computational complexity and thus becoming impractical.
# 4.2. Analysis & Discussion
Ablation study. We analyze the contributions of major components to the performance by performing ablation studies, focusing on: (1) -Tuning: omitting the weight tuning process in the linear model, summing up the binary feature to compute the logit instead; (2) -Ensemble: omitting the ensemble process; (3) -Description: omitting the feature description, and (4) -Reasoning: omitting the Step-1 process in reasoning instruction part when generating rules.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/11b8/11b84167-32b5-4227-b4cc-c174b2f7169e.png" style="width: 50%;"></div>
Model
Training (in seconds) Inference (in milliseconds)
LogReg
0.721
0.001
XGBoost
28.512
0.006
RandomForest
1.343
0.001
SCARF
426.859
0.002
TabPFN
0.440
1.149
STUNT
642.796
0.006
In-context†
N/A
463.000
TABLET†
0.813
523.254
TabLLM
251.242
335.127
FeatLLM†
860.094
0.006
<div style="text-align: center;">Figure 4: Performance comparison summaries among conventional tabular baselines and FeatLLM. Averaged AUC over all datasets across the number of shots are reported.</div>
Figure 4: Performance comparison summaries among conventional tabular baselines and FeatLLM. Averaged AUC over all datasets across the number of shots are reported.
Table 4 summarizes the change in the AUC from ablations averaged across all datasets. In all cases, modifying the ablated component results in performance degradation. Interestingly, the benefit of weight tuning becomes higher when the number of shots increases. This indicates that when there is a large amount of data, accurate estimation of the importance of rules becomes feasible, making the effect of tuning significant. On the other hand, the effect of feature descriptions and reasoning instructions are high when the number of shot is small. This suggests that the efficient utilization of the prior knowledge of LLM becomes crucial for performance improvements. Lastly, the proposed ensemble approach consistently improves the performance in all settings. Table 4: Ablation study summaries. Averaged changes of AUC with standard error after altering or omitting modules over 13 datasets with three different trials are reported.
Shot
FeatLLM
-Tuning
-Ensemble
-Description
-Reasoning
4
75.7
-1.41±1.00
-5.39±0.81
-1.76±1.06
-5.03±1.96
8
77.3
-2.72±0.93
-6.96±1.40
-1.20±0.33
-3.55±0.81
16
78.4
-2.57±0.73
-6.65±1.18
-0.26±0.31
-1.50±0.87
32
80.3
-5.75±1.19
-7.38±1.34
-0.29±0.58
-2.42±1.15
64
81.4
-4.88±1.40
-6.09±0.96
-0.70±0.54
-1.71±0.47
Avg
78.6
-3.47±0.51
-6.49±0.51
-0.84±0.28
-2.84±0.53
Training & inference time comparison. We compare the training and inference time between different methods. The experiments are performed on the Adult dataset, including a training set of 16 shots. One A100 GPU is used as the default, except for TabLLM which uses four A100 GPUs for model parallelism. Table 5 provides the runtime comparisons. Notably, FeatLLM shows a relatively low inference time, comparable to that of conventional methods. In contrast, other LLM-based approaches, such as In-context learning, TABLET, and TabLLM, exhibit much higher inference time. In terms of training time, FeatLLM is comparable with other LLM-based methods – it requires only 30 API queries, which does not impose a significant burden on the budget and even can be parallelized.
<div style="text-align: center;">Table 5: Comparison of training (with 16 samples) and inference (over one sample) time on Adult.</div>
† These models employ API queries, where the runtime is subject to the API’s status at the time of use.
Quality of features generated by FeatLLM. We conduct a simple experiment to investigate the informativeness of rule-based features generated by FeatLLM in solving practical tasks. Specifically, we measure the average AUROC between the newly generated features and the target labels, and then compute the ratio of useful features with an AUROC higher than 0.5 for each dataset. Table 6 below summarizes the results. It confirms that the majority of the created rules are relevant to the target attribute and provide informative insights for addressing the target task (refer to the column “Before tuning”). Moreover, during the process of tuning the linear model with data, rules that have little relevance to the data (i.e., rules with learned weights below the threshold) are automatically filtered out (refer to the column “After tuning”). Note that the threshold was set at 0.1, taking into consideration the overall histogram of weights. Table 6: Evaluation of feature quality before and after linear model tuning. Ratio of newly-generated features with an AUROC higher than 0.5 is reported across datasets.
Table 6: Evaluation of feature quality before and after linear model tuning. Ratio of newly-generated features with an AUROC higher than 0.5 is reported across datasets.
Method
Ratio of useful rules
Before tuning
After tuning
Adult
0.971
0.980
Bank
0.627
0.958
Blood
0.889
0.975
Car
0.712
0.849
Communities
0.767
0.889
Credit-g
0.643
0.802
Diabetes
0.974
0.972
Heart
0.800
0.924
Myocardial
0.690
0.744
Cultivars
0.775
0.927
NHANES
0.600
0.832
Sequence-type
0.896
0.985
Solution-mix
0.917
1.000
Average
0.789
0.911
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2696/26969aaf-bd58-402d-acf7-0f0b71ce5ea1.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/54bf/54bfc86c-ad61-4a8e-8926-dad30fa39795.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
Figure 5: Visualization of performance impact from spurious correlations. The results exhibit the models’ performance (AUC) each time a noisy column from the Adult dataset is added to the original Heart dataset. XGBoost is excluded here due to its lower performance.
Effect of adding spurious correlations. To ascertain how effectively various methods filter out noisy spurious correlations in a low-shot regime, we conducted an analysis. The experiment utilized the Heart dataset, to which we randomly attach columns from the Adult dataset that are irrelevant to the task of the Heart dataset. We then measure the performance changes of the models while progressively increasing the number of unrelated columns. For a fair comparison, we assume the same 8 shots of labeled samples without considering an unlabeled dataset, thereby excluding methods like SCARF or STUNT from the comparison.
Figure 5 visualizes the performance changes due to spurious correlations. Compared to baselines, FeatLLM showed the least performance degradation from spurious correlations. This is likely because our framework forces the consideration of the relationship between the task and features based on prior knowledge when extracting rules, preventing the generation of rules based on irrelevant columns and thereby yielding robust results. Indeed, we observe that rules based on noisy columns accounted for only 1-2% of the total. Meanwhile, we also observe that FeatLLM can also learn spurious correlations and misunderstand causal relationships between task and features if columns from closely related data are randomly attached (see Appendix I).
Hyper-parameter analysis FeatLLM has two main hyper-parameters. One is the number of ensembles, and the other is how many rules per class to extract during each LLM query. We conducted experiments to investigate the impact of these hyper-parameters on performance. The results showed that performance improves as the number of ensembles increases, but beyond a certain point (i.e., around 20), the performance gains begin to converge (see Fig. 11 in Appendix). As for the number of rules, while there wasn’t a statistically significant difference, we determined that 10 is the most appropriate number considering the trade-off
<div style="text-align: center;">Figure 6: Effect of the number of rules per each rule extraction. Averaged AUC over all datasets is reported.</div>
between prompt length and performance (see Fig. 6). We speculate this result is due to more rules enlarging input dimensionality, leading to an increase in information, yet also resulting in overfitting within the low-shot regime.
# 5. Conclusion
We present FeatLLM which advances the state-of-the-art in LLM-based few-shot tabular learning. FeatLLM goes beyond using the LLMs for predictions for each sample, and focuses on generating criteria for making predictions as feature engineers. Through rule generation and an ensemble process with bagging, FeatLLM requires minimal inference cost, needs only API access without training, and overcomes the constraints to data feature size. FeatLLM achieves high prediction performance and is a superior alternative for applying LLM on real-world tabular data in a data-efficient way.
FeatLLM is designed to focus on low-shot learing regime only. Our future goal is to expand its capabilities of crafting new features for datasets with a larger number of samples, exploring various feature types beyond rules, and bringing interpretability, which enables its application across a wider array of tasks.
# 6. Broader Impact
The proposed framework, FeatLLM, aims to utilize the prior knowledge of LLM in performing tabular learning tasks to go beyond the achievable performance with standard supervised tabular learning. To push the data efficiency in learning, our research makes important contributions to mitigate the issue of overfitting due to limited training samples, by supplementing with prior knowledge. FeatLLM is particularly beneficial for practitioners in real-world scenarios (such as from finance and healthcare industries) where labeling is challenging due to the high human cost or label collection difficulties, and prior knowledge in LLMs can be potentially very valuable (as the LLMs are pretrained
with relevant finance or healthcare related text). As one potential important future work direction towards widespread adoption, analyses of the impact of the societal biases and misinformation embedded in the prior knowledge of LLMs would be important, as the prior knowledge would be directly affecting the predictions even dominating over the observed or collected labeled data samples.
# Acknowledgement
We sincerely thank Scott Yak and Yihe Dong for their valuable feedback and discussion on this work.
# References
100,000 genomes pilot on rare-disease diagnosis in health care — preliminary report. New England Journal of Medicine, 385(20):1868–1880, 2021. Anil, R., Dai, A. M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z., et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023. Arik, S. ¨O. and Pfister, T. Tabnet: Attentive interpretable tabular learning. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pp. 6679–6687, 2021. Asuncion, A. and Newman, D. Uci machine learning repository, 2007. Bahri, D., Jiang, H., Tay, Y., and Metzler, D. Scarf: Selfsupervised contrastive learning using random feature corruption. In International Conference on Learning Representations, 2022. Bartlett, P. L., Montanari, A., and Rakhlin, A. Deep learning: a statistical viewpoint. Acta numerica, 30:87–201, 2021. Breiman, L. Random forests. Machine learning, 45:5–32, 2001. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020. Chen, T. and Guestrin, C. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pp. 785–794, 2016. Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597–1607. PMLR, 2020.
Chen, W.-Y., Liu, Y.-C., Kira, Z., Wang, Y.-C. F., and Huang, J.-B. A closer look at few-shot classification. In International Conference on Learning Representations, 2018. Creswell, A., Shanahan, M., and Higgins, I. Selectioninference: Exploiting large language models for interpretable logical reasoning. In The Eleventh International Conference on Learning Representations, 2022. Dinh, T., Zeng, Y., Zhang, R., Lin, Z., Gira, M., Rajput, S., Sohn, J.-y., Papailiopoulos, D., and Lee, K. Lift: Language-interfaced fine-tuning for non-language machine learning tasks. Advances in Neural Information Processing Systems, 35:11763–11784, 2022. Fan, Z., Gao, X., Mirchev, M., Roychoudhury, A., and Tan, S. H. Automated repair of programs from large language models. In 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), pp. 1469– 1481. IEEE, 2023. Finch, S. E., Paek, E. S., and Choi, J. D. Leveraging large language models for automated dialogue analysis. In Proceedings of the 24th Meeting of the Special Interest Group on Discourse and Dialogue, pp. 202–215, 2023. Golovenkin, S.E., S. V. R. D. S. P. N. S. O. Y. and Voino-Yasenetsky, V. Myocardial infarction complications. UCI Machine Learning Repository, 2020. DOI: https://doi.org/10.24432/C53P5M. Hegselmann, S., Buendia, A., Lang, H., Agrawal, M., Jiang, X., and Sontag, D. Tabllm: Few-shot classification of tabular data with large language models. In International Conference on Artificial Intelligence and Statistics, pp. 5549–5581. PMLR, 2023. Ho, T. K. Random decision forests. In Proceedings of 3rd international conference on document analysis and recognition, volume 1, pp. 278–282. IEEE, 1995. Hollmann, N., M¨uller, S., Eggensperger, K., and Hutter, F. Tabpfn: A transformer that solves small tabular classification problems in a second. In The Eleventh International Conference on Learning Representations, 2023. Hu, E. J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2021. Imani, S., Du, L., and Shrivastava, H. Mathprompter: Mathematical reasoning using large language models. In ICLR 2023 Workshop on Trustworthy and Reliable Large-Scale Machine Learning Models, 2023. Jakobsen, J. C., Gluud, C., Wetterslev, J., and Winkel, P. When and how should multiple imputation be used for
handling missing data in randomised clinical trials–a practical guide with flowcharts. BMC medical research methodology, 17(1):1–10, 2017. Kadra, A., Lindauer, M., Hutter, F., and Grabocka, J. Welltuned simple nets excel on tabular datasets. Advances in neural information processing systems, 34:23928–23941, 2021. Kang, H. The prevention and handling of the missing data. Korean journal of anesthesiology, 64(5):402–406, 2013. Kovaˇc, G., Sawayama, M., Portelas, R., Colas, C., Dominey, P. F., and Oudeyer, P.-Y. Large language models as superpositions of cultural perspectives. arXiv preprint arXiv:2307.07870, 2023. Kuhn, H. W. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83– 97, 1955. Levin, R., Cherepanova, V., Schwarzschild, A., Bansal, A., Bruss, C. B., Goldstein, T., Wilson, A. G., and Goldblum, M. Transfer learning with deep tabular models. In The Eleventh International Conference on Learning Representations, 2022. Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M., and Raffel, C. A. Few-shot parameter-efficient finetuning is better and cheaper than in-context learning. Advances in Neural Information Processing Systems, 35: 1950–1965, 2022. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172, 2023. Lu, Y., Bartolo, M., Moore, A., Riedel, S., and Stenetorp, P. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8086–8098, 2022. Majmundar, K. A., Goyal, S., Netrapalli, P., and Jain, P. Met: Masked encoding for tabular data. In NeurIPS 2022 First Table Representation Workshop, 2022. Majumder, S., Chen, C., Al-Halah, Z., and Grauman, K. Few-shot audio-visual learning of environment acoustics. Advances in Neural Information Processing Systems, 35: 2522–2536, 2022. Moro, S., Cortez, P., and Rita, P. A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62:22–31, 2014.
Majumder, S., Chen, C., Al-Halah, Z., and Grauman, K. Few-shot audio-visual learning of environment acoustics. Advances in Neural Information Processing Systems, 35: 2522–2536, 2022.
Moro, S., Cortez, P., and Rita, P. A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62:22–31, 2014.
Nam, J., Song, W., Park, S. H., Tack, J., Yun, S., Kim, J., and Shin, J. Semi-supervised tabular classification via incontext learning of large language models. In Workshop on Efficient Systems for Foundation Models@ ICML2023, 2023a.
# OpenAI. Gpt-4 technical report, 2023.
Redmond, M. Communities and Crime. Machine Learning Repository, 2009. https://doi.org/10.24432/C53W3X.
UCI DOI:
UCI DOI:
Sanh, V., Webson, A., Raffel, C., Bach, S., Sutawika, L., Alyafeai, Z., Chaffin, A., Stiegler, A., Raja, A., Dey, M., et al. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations, 2022.
Schick, T. and Sch¨utze, H. Exploiting cloze-questions for few-shot text classification and natural language inference. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 255–269, 2021.
Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., et al. Large language models encode clinical knowledge. Nature, 620(7972):172–180, 2023.
Slack, D. and Singh, S. Tablet: Learning from instructions for tabular data. arXiv preprint arXiv:2304.13188, 2023.
Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., and Goldstein, T. Saint: Improved neural networks for tabular data via row attention and contrastive pretraining. arXiv preprint arXiv:2106.01342, 2021.
Taylor, R., Kardas, M., Cucurull, G., Scialom, T., Hartshorn, A., Saravia, E., Poulton, A., Kerkez, V., and Stojnic, R. Galactica: A large language model for science. arXiv preprint arXiv:2211.09085, 2022.
Ucar, T., Hajiramezanali, E., and Edwards, L. Subtab: Subsetting features of tabular data for self-supervised representation learning. Advances in Neural Information Processing Systems, 34:18853–18865, 2021.
Van Buuren, S. and Groothuis-Oudshoorn, K. mice: Multivariate imputation by chained equations in r. Journal of statistical software, 45:1–67, 2011.
Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2022. Wang, Z. and Sun, J. Transtab: Learning transferable tabular transformers across tables. Advances in Neural Information Processing Systems, 35:2902–2915, 2022. Wang, Z., Gao, C., Xiao, C., and Sun, J. Anypredict: Foundation model for tabular prediction. arXiv preprint arXiv:2305.12081, 2023. Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022a. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35: 24824–24837, 2022b. Wilcox, A. B. and Hripcsak, G. The role of domain knowledge in automating medical text report classification. Journal of the American Medical Informatics Association, 10(4):330–338, 2003. Yeh, I.-C., Yang, K.-J., and Ting, T.-M. Knowledge discovery on rfm model using bernoulli sequence. Expert Systems with applications, 36(3):5866–5871, 2009. Yoon, J., Zhang, Y., Jordon, J., and van der Schaar, M. Vime: Extending the success of self-and semi-supervised learning to tabular domain. Advances in Neural Information Processing Systems, 33:11033–11043, 2020. Zhang, T., Wang, S., Yan, S., Li, J., and Liu, Q. Generative table pre-training empowers models for tabular prediction. arXiv preprint arXiv:2305.09696, 2023. Zhao, W. X., Zhou, K., Li, J., Tang, T., Wang, X., Hou, Y., Min, Y., Zhang, B., Zhang, J., Dong, Z., et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023. Zhu, B., Shi, X., Erickson, N., Li, M., Karypis, G., and Shoaran, M. Xtab: Cross-table pretraining for tabular transformers. arXiv preprint arXiv:2305.06090, 2023.
# A. Examples of Prompts and Their Corresponding Outputs
A.1. Task Description for Each Dataset
This section demonstrates creation of prompts for rule extraction for each task, along with the task descriptions we use (see Section 3.1 - Basic Information Description). The descriptions of the tasks are developed by referring to the descriptions in the original data links and previous works (Hegselmann et al., 2023), as outlined in the table below. Table 7: Description of the tasks associated with each dataset utilized in the prompt. The text in blue describes the label information, providing the list of possible answer classes.
This section demonstrates creation of prompts for rule extraction for each task, along with the task descriptions we use (se Section 3.1 - Basic Information Description). The descriptions of the tasks are developed by referring to the descriptions i the original data links and previous works (Hegselmann et al., 2023), as outlined in the table below.
Data
Task description
Adult
Does this person earn more than 50000 dollars per year? Yes or no?
Bank
Does this client subscribe to a term deposit? Yes or no?
Blood
Did the person donate blood? Yes or no?
Car
How would you rate the decision to buy this car? Unacceptable, acceptable, good or very good?
Communities
How high will the rate of violent crimes per 100K population be in this area. Low, medium, or high?
Credit-g
Does this person receive a credit? Yes or no?
Diabetes
Does this patient have diabetes? Yes or no?
Heart
Does the coronary angiography of this patient show a heart disease? Yes or no?
Myocardial
Does the myocardial infarction complications data of this patient show chronic heart failure? Yes or no?
Cultivars
How high will the grain yield of this soybean cultivar. Low or high?
NHANES
Predict this person’s age group from the given record. Senior or non-senior?
Sequence-type
What is the type of following sequence? Arithmetic, geometric, fibonacci, or collatz?
Solution-mix
Given the volumes and concentrations of four solutions, does the percent concentration of the mixed
solution over 0.5? Yes or no?
# A.2. Example Prompt for Extracting Rules
You are an expert. Given the task description and the list of features and data examples, you are extracting conditions for each answer class to solve the task. Task: Does the coronary angiography of this patient show a heart disease? Yes or no? Features: - Age: age of the patient (numerical variable) - Sex: sex of the patient (categorical variable with categories [M, F]) - ChestPainType: chest pain type (categorical variable with categories [ATA, NAP, ASY, TA]) - RestingBP: resting blood pressure [mm Hg] (numerical variable) - Cholesterol: serum cholesterol [mm/dl] (numerical variable) - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise] (numerical variable) - RestingECG: resting electrocardiogram results (categorical variable with categories [Normal, ST, LVH]) - MaxHR: maximum heart rate achieved (numerical variable) - ExerciseAngina: exercise-induced angina (categorical variable with categories [N, Y]) - Oldpeak: oldpeak = ST [Numeric value measured in depression] (numerical variable) - ST Slope: the slope of the peak exercise ST segment (categorical variable with categories [Up, Flat, Down]) Examples: Age is 63. Sex is M. ChestPainType is NAP. RestingBP is 130. Cholesterol is 0. FastingBS is 1. RestingECG is ST. MaxHR is 160. ExerciseAngina is N. Oldpeak is 3.0. ST Slope is Flat. Answer: no Age is 39. Sex is M. ChestPainType is ATA. RestingBP is 120. Cholesterol is 204. FastingBS is 0. RestingECG is Normal. MaxHR is 145. ExerciseAngina is N. Oldpeak is 0.0. ST Slope is Up. Answer: no Age is 58. Sex is M. ChestPainType is NAP. RestingBP is 160. Cholesterol is 211. FastingBS is 1. RestingECG is ST. MaxHR is 92. ExerciseAngina is N. Oldpeak is 0.0. ST Slope is Flat. Answer: yes Age is 55. Sex is M. ChestPainType is ASY. RestingBP is 160. Cholesterol is 289. FastingBS is 0. RestingECG is LVH. MaxHR is 145. ExerciseAngina is Y. Oldpeak is 0.8. ST Slope is Flat. Answer: yes
- Oldpeak: oldpeak = ST [Numeric value measured in depression] (numerical variable) - ST Slope: the slope of the peak exercise ST segment (categorical variable with categories [Up, Flat, Down]) Examples: Age is 63. Sex is M. ChestPainType is NAP. RestingBP is 130. Cholesterol is 0. FastingBS is 1. RestingECG is ST. MaxHR is 160. ExerciseAngina is N. Oldpeak is 3.0. ST Slope is Flat. Answer: no Age is 39. Sex is M. ChestPainType is ATA. RestingBP is 120. Cholesterol is 204. FastingBS is 0. RestingECG is Normal. MaxHR is 145. ExerciseAngina is N. Oldpeak is 0.0. ST Slope is Up. Answer: no Age is 58. Sex is M. ChestPainType is NAP. RestingBP is 160. Cholesterol is 211. FastingBS is 1. RestingECG is ST. MaxHR is 92. ExerciseAngina is N. Oldpeak is 0.0. ST Slope is Flat. Answer: yes Age is 55. Sex is M. ChestPainType is ASY. RestingBP is 160. Cholesterol is 289. FastingBS is 0. RestingECG is LVH. MaxHR is 145. ExerciseAngina is Y. Oldpeak is 0.8. ST Slope is Flat. Answer: yes
Format for [Condition]: For the categorical variable only, - [Feature] is in [List of categories] For the numerical variable only, - [Feature] (> or >= or < or <=) [Value] - [Feature] is within range of [Value start, Value end]
Figure 7: Example prompt for rule extraction in the ‘Heart’ dataset
# A.3. Example Outcome from Rule Extraction
We demonstrate how rules are derived through prompts by showing an example of an outcome. If we input the prompt from the previous subsection (see Fig. 7), the following result is obtained.
Step 1. The relationship between each feature and the task description: - Age: There is a tendency for older age to be associated with a higher risk of heart disease. - Sex: There may be a difference in the prevalence of heart disease between males and females, with males potentially having a higher risk. - ChestPainType: Certain types of chest pain, such as NAP and ASY, may be indicative of heart disease. - RestingBP: Higher resting blood pressure levels may be associated with a higher risk of heart disease. - Cholesterol: Higher levels of serum cholesterol may be associated with a higher risk of heart disease. - FastingBS: Fasting blood sugar levels above 120 mg/dl may be indicative of heart disease. - RestingECG: Abnormal resting electrocardiogram results, such as ST and LVH, may be indicative of heart disease. - MaxHR: Lower maximum heart rate achieved may be associated with a higher risk of heart disease. - ExerciseAngina: The presence of exercise-induced angina may be indicative of heart disease. - Oldpeak: Higher values of oldpeak (ST depression) may be associated with a higher risk of heart disease. - ST Slope: The slope of the peak exercise ST segment may provide information about the presence of heart disease.
# A.4. Example Prompt for Parsing Rules
Utilizing the given rules, we convert them into code capable of transforming each data point into a binary feature. This binary feature indicates whether the sample adheres to the rules associated with its class. Below is an example prompt for parsing rules, employing the same example as in the previous subsection.
Wrap only the function part with <start> and <end>, and do not add any comments, descriptions, and package importing lines in the code.
Figure 9: Example prompt for parsing rules in the ‘Heart’ dataset
# A.5. Example Outcome After Parsing
The result of the parsing request is exemplified as follows.
<start> def extracting features no(df input): df output = pd.DataFrame() df output[‘Age’] = df input[‘Age’].apply(lambda x: 1 if x >= 40 and x <= 60 else 0) df output[‘Sex’] = df input[‘Sex’].apply(lambda x: 1 if x == ‘F’ else 0) df output[‘ChestPainType’] = df input[‘ChestPainType’].apply(lambda x: 1 if x in [‘ATA’, ‘NAP’] else 0) df output[‘RestingBP’] = df input[‘RestingBP’].apply(lambda x: 1 if x < 140 else 0) df output[‘Cholesterol’] = df input[‘Cholesterol’].apply(lambda x: 1 if x < 200 else 0) df output[‘FastingBS’] = df input[‘FastingBS’].apply(lambda x: 1 if x == 0 else 0) df output[‘RestingECG’] = df input[‘RestingECG’].apply(lambda x: 1 if x == ‘Normal’ else 0) df output[‘MaxHR’] = df input[‘MaxHR’].apply(lambda x: 1 if x > 140 else 0) df output[‘ExerciseAngina’] = df input[‘ExerciseAngina’].apply(lambda x: 1 if x = ‘N’ else 0) df output[‘Oldpeak’] = df input[‘Oldpeak’].apply(lambda x: 1 if x < 1.0 else 0)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/afc5/afc583c4-3f97-46b5-abe7-45b74fec21be.png" style="width: 50%;"></div>
# B. Hyper-parameter analysis
FeatLLM has two main hyper-parameters. One is the number of ensembles, and the other is how many rules per class to extract during each LLM query. We conducted experiments to investigate the impact of these hyper-parameters on performance. The results showed that performance improves as the number of ensembles increases, but beyond a certain point (i.e., around 20), the performance gains begin to converge (see Fig. 11). For an analysis of the impact of the number of rules, please refer to Section 4.2.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bfcc/bfcc76d1-8fd8-478d-a183-07cd9a442b42.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
Figure 11: Effect of the number of ensembles. Averaged AUC over all datasets is reported.
# C. Baseline Implementation Details
In this section, we provide a detailed information on the implementation of baselines. In our study, we adhere to the settings and parameters outlined in the original papers for various baselines. During model training, the dataset was divided allocating 20% as the test set. This division was carried out using stratified sampling to ensure a representative distribution of classes. From the remaining data, k-shot samples were selected, constructing a balanced training set. We repeated experiments multiple times with different randomness on the choice of training/test sets. For conventional machine learning methods, including logistic regression (LogReg), RandomForest, and XGBoost, we determined their parameters using Grid-search combined with k-shot cross validation. Following the previous work (Hegselmann et al., 2023), the search space of each parameter is described in Tables below.
Parameter
Search space
penalty
l1, l2
C
100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5
Table 8: Hyperparameter search space for LogReg
Table 9: Hyperparameter search space for RandomForest
Parameter
Search space
max depth
2, 4, 6, 8, 10, 12
alpha
1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1
lambda
1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1
eta
0.01, 0.03, 0.1, 0.3
<div style="text-align: center;">Table 10: Hyperparameter search space for XGBoost</div>
For baseline models that incorporate unlabeled data in training, such as SCARF and STUNT, any data not used in the test set or as k-shot training samples is treated as the unlabeled dataset. To ensure that there are no missing values in the training and test data, columns representing more than 20% of the entire dataset were removed, similarly following the literature (Kang, 2013; Jakobsen et al., 2017). Subsequently, any rows containing missing values were also eliminated. For TabPFN, we utilized the official GitHub repository7 to reproduce results, maintaining the default parameter settings as specified. Regarding LLM-based baselines, we utilized GPT-3.5 for in-context learning, while T0 was chosen for the TabLLM, in line with their original usage settings. To calculate the AUC for the in-context learning and TABLET with the GPT-3.5 API, we performed the inference process three times, introducing variability with a nonzero temperature setting (i.e., 1). The results were then averaged to estimate the probabilities for each class. In contrast, for the TabLLM, we employed T0, utilizing its publicly available checkpoint. We directly calculated the probability of the class name token to estimate the class probabilities.
Parameter
Search space
bootstrap
True, False
max depth
2, 4, 6, 8, 10, 12
n estimators
2, 4, 8, 16, 32, 64, 128, 256
# D. Diversity of Features Generated by FeatLLM
In our model, to guarantee diversity of features (i.e., rules) generated from the LLM in each iteration, we set the temperature of LLMs to a value greater than 0 (i.e., 0.5) to introduce randomness into the generation, applied bagging strategies (i.e., feature and sample bagging), and shuffled the order of training samples in the prompt for each iteration. To empirically verify the diversity of features, we conducted the following analysis: 1. Given two sets of rules from different iterations, we applied each set’s rules across the dataset to generate binary features. 2. We used the Hungarian Maximum Matching Algorithm (Kuhn, 1955) to create a bipartite graph connecting pairs of binary features between the two sets with the highest correlation. 3. The average correlation of the selected pairs was calculated. 4. This process (steps 1-3) was repeated for all pairs of iterations to determine the average and variance of the correlations.
In our model, to guarantee diversity of features (i.e., rules) generated from the LLM in each iteration, we set the temperature of LLMs to a value greater than 0 (i.e., 0.5) to introduce randomness into the generation, applied bagging strategies (i.e., feature and sample bagging), and shuffled the order of training samples in the prompt for each iteration. To empirically verify the diversity of features, we conducted the following analysis:
1. Given two sets of rules from different iterations, we applied each set’s rules across the dataset to generate binary features. 2. We used the Hungarian Maximum Matching Algorithm (Kuhn, 1955) to create a bipartite graph connecting pairs of binary features between the two sets with the highest correlation. 3. The average correlation of the selected pairs was calculated. 4. This process (steps 1-3) was repeated for all pairs of iterations to determine the average and variance of the correlations.
We conducted an ablation study to assess the effects of three components (i.e., temperature, bagging, instance shuffling) on enhancing rule diversity. Table 11 below presents the results of average correlation across iterations, where a lower correlation signifies a more diverse set of extracted rules. We observed that the average correlation