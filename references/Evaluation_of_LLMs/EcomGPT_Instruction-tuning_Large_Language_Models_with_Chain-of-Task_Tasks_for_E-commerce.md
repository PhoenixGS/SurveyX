# EcomGPT: Instruction-tuning Large Language Models with Chain-of-Task Tasks for E-commerce
Anonymous submission
# Abstract
28 Aug 20
Recently, instruction-following Large Language Models (LLMs) , represented by ChatGPT, have exhibited exceptional performance in general Natural Language Processing (NLP) tasks. However, the unique characteristics of E-commerce data pose significant challenges to general LLMs. An LLM tailored specifically for E-commerce scenarios, possessing robust cross-dataset/task generalization capabilities, is a pressing necessity. To solve this issue, in this work, we proposed the first E-commerce instruction dataset EcomInstruct, with a total of 2.5 million instruction data. EcomInstruct scales up the data size and task diversity by constructing atomic tasks with E-commerce basic data types, such as product information, user reviews. Atomic tasks are defined as intermediate tasks implicitly involved in solving a final task, which we also call Chain-of-Task tasks. We developed EcomGPT 1 with different parameter scales by training the backbone model BLOOMZ with the EcomInstruct. Benefiting from the fundamental semantic understanding capabilities acquired from the Chain-of-Task tasks, EcomGPT exhibits excellent zero-shot generalization capabilities. Extensive experiments and human evaluations demonstrate that EcomGPT outperforms ChatGPT in term of cross-dataset/task generalization on E-commerce tasks.
arXiv:2308.06966v
# Introduction
In the field of E-commerce, the progress made in natural language processing (NLP) and deep learning (DL) has significantly contributed to the advancement of E-commerce technology. These advancements have unlocked diverse capabilities ranging from product information extraction (Cheng et al. 2021; Wang et al. 2021) to user query understanding (Zhao, Chen, and Yin 2019; Ahmadvand et al. 2020). Recently, instruction-following Large Language Models (LLMs) (Ouyang et al. 2022; Taori et al. 2023; Chiang et al. 2023), such as ChatGPT, have demonstrated exceptional performance in general natural language processing tasks (Zhao et al. 2023). These LLMs can accomplish various tasks by transforming them into generative paradigms. One noteworthy aspect is the remarkable zero-shot capabilities exhibited by LLMs, which can be attributed to instruction tuning. However, despite their numerous merits, general LLMs are not specifically designed for the E-commerce sector. This
Attribute Extraction
Input: 撞色拼接的领口以及大口袋(Colour blocked neckline and large
pockets)
ChatGPT: I’m sorry, but I can’t provide the response you’re looking for. �
EcomGPT: 图案: 撞色(Pattern: Colour blocked) �
Explanation: Sentences composed solely of entities, even shorter than that of
prompts, poses challenges to the model’s comprehension.
Review Topic Classification
Input: My personal favorite is Nick and Joe’s. Candidate Topic: Food, Price, ...,
Anecdotes/Miscellaneous
ChatGPT: Food �
EcomGPT: Anecdotes/Miscellaneous �
Explanation: Emerging entities, such as shop names, which not in the pre-
training corpus, require stronger generalization capability for model.
Product Title Generation
Input: #锅类型#:#少油烟、不易粘锅#;#流行元素#:#复古#;#型
号#:#L70846#;#材质#:#熟铁#;#品牌#:#other/其他#; ......
ChatGPT: 【其他品牌】32cm中式复古熟铁燃气灶木质锅盖少油烟商用
锅L70846 ([Other brands] 32cm Chinese Retro Cooked Iron Gas Stove Wooden
Pot Lid Less Fume Commercial Cooker L70846) �
Output from EcomGPT: 复古铁锅炒菜不粘锅家用商用炒菜锅炒肉不糊
锅炒鸡蛋平底锅(Vintage iron skillet frying non-stick household commercial
frying pan frying meat without paste pan scrambled eggs pan) �
Explanation: Textual content with complex structure is difficult to under-
stand by general LLMs. Hence, in title generation, ChatGPT mechanically
splices attribute values regardless of the attribute importance.
Table 1: Real cases in E-commerce that general LLMs cannot handle.
can lead to suboptimal performance for various E-commerce tasks. Table 1 illustrates the distinctive characteristics of E-commerce data (Tsagkias et al. 2021; Jiang et al. 2022) compared to general domains. Firstly, E-commerce data possesses a specific and complex syntactic structure that differs from coherent sentences in general. For example, product titles are typically composed of discrete entities and are much shorter than regular sentences. Considering another example, product information often consists of attribute-attribute value pairs separated by special symbols (e.g., “##”), which also poses challenges for general LLMs to comprehend. Secondly, the word distribution of E-commerce data significantly varies from that of general domains due to the abundance of unique entities and concepts found in E-commerce platforms (Escursell, Llorach-Massana, and Roncero 2021). Moreover, these novel entities and concepts are highly dynamic and continuously updated as new products, users, and trends emerge
daily, requiring exceptional generalization capabilities to effectively handle such dynamics. Consequently, there is an urgent need for the LLM specifically tailored for E-commerce scenarios, equipped with robust cross-dataset/task generalization capabilities. In the BERT era, numerous efforts (Zhang et al. 2021; Qiu et al. 2022; Xu et al. 2021) have been made to enhance the models’ generalization ability by integrating domain knowledge. For instance, E-BERT (Poerner, Waltinger, and Sch¨utze 2020) further pre-trains BERT on the Amazon dataset to incorporate semantic knowledge of the E-commerce domain into BERT. However, these efforts primarily rely on encoderonly architectures like BERT, limiting their capacity for instruction learning and achieving stronger generalization capabilities. Furthermore, the parameter sizes of these models are relatively small (less than 1 billion), making it challenging to capture and represent complex linguistic knowledge, thereby restricting their generalization capabilities. To enhance models’ generalization ability cross dataset/tasks, this work presents the first E-commerce instruction dataset, EcomInstruct, comprising a total of 2.5 million instruction data and 134 tasks. EcomInstruct are built from two main sources. Firstly, we manually collect a wide range of E-commerce natural language processing (NLP) datasets from open data sources, such as academic websites and data competition platforms. They cover a broad range of tasks, including E-commerce named entity recognition, reviewbased Q&A, product classification, multi-turn dialogue, and other traditional NLP tasks. The benefit of these open-source datasets is that they are expert-calibrated and high-quality. Secondly, we identified several basic data types that are common in E-commerce scenarios, including product information, user reviews, user dialogue, and search queries. Around these basic data types, we build a large number of atomic tasks. Formally, atomic tasks are defined as intermediate tasks implicitly involved in solving a final task. The fundamental semantic understanding capabilities learned from the atomic tasks are also used when solving other unseen tasks, thus can greatly enhances the model’s generalization capabilities. With this motivation, we further construct a large number of atomic tasks around these basic data types, as shown in Figure 1. Since these atomic tasks are the link in the chain of task solution, we refer to them as Chain-of-Task tasks (CoT tasks), in reference to previous work on Chain-of-thought (Wei et al. 2022; Wang et al. 2022a). After collecting the above two parts of raw data, expert-written task-specific instruction schema and raw data are combined to obtain final instruction data. By training the backbone model BLOOMZ with EcomInstruct, we developed the instruction-following LLM EcomGPT for E-commerce. EcomGPT exhibits exceptional generalization capabilities compared to ChatGPT on various unseen E-commerce dataset and tasks. The further ablation experiments highlight the effectiveness of the Chain-of-Task tasks. This strongly implies that we can enhance the model’s generalization ability by constructing diverse atomic tasks specifically tailored to the domain data, especially when the domain data is limited. In summary, the contributions of this work are threefold:
Lang.
Task Para.
# task
# train inst.
# test inst.
EN
CLS
15
130,596
34,189
Ext
15
82,397
47,284
Gen
22
353,486
96,585
Other
10
61,756
36,481
ZH
CLS
18
324,062
362,845
Ext
9
131,814
54,725
Gen
37
444,503
353,486
Other
8
111,814
36,481
ALL
134
1,533,300
1,023,076
1. We proposed the first E-commerce instruction dataset EcomInstruct, with a total of 2.5 million instruction data. EcomInstruct scales up the data size and task diversity by constructing Chain-of-Task tasks (atomic tasks). 2. We proposed the first instruction-following LLM specifically designed for E-commerce. Benefiting from numerous Chain-of-Task tasks, EcomGPT exhibits superior zero-shot generalization ability. 3. Extensive experiments demonstrate the effectiveness of EcomGPT compared to ChatGPT with larger parameter scales. Furthermore, the detailed ablation experiments provide guidance for the design of LLMs in vertical domains.
# EcomInstuct: E-commerce Instruction Tuning Dataset
Overview of the EcomInstuct
In this section, we present our EcomInstruct dataset for instruction tuning on E-commerce tasks, which primarily built from two sources. Firstly, we manually collected a diverse set of E-commerce natural language processing (NLP) datasets from various open data sources, including academic websites and data competition platforms. They cover a broad range of tasks, such as E-commerce named entity recognition and intent detection. These datasets are typically of high quality as they have been carefully curated by experts in the field. Secondly, we identified several basic data types that are common in E-commerce scenarios, including product information, user reviews, user dialogue, and search queries. Around these basic data types, we build a large number of atomic tasks. Formally, atomic tasks are defined as intermediate tasks implicitly involved in solving a final task. The fundamental semantic understanding capabilities learned from the atomic tasks are also used when solving other unseen tasks, thus can greatly enhances the model’s generalization capabilities. For instance, when performing named entity recognition, the model needs to perform entity span detection and entity classification sequentially. Meanwhile, entity span detection is also implicitly used when conducting review sentiment analysis, as the model needs to detect entities with sentiment tendencies. Since these atomic tasks are the link in the chain of task solution, we refer to them as Chain-of-Task tasks (CoT tasks), in reference to previous work on Chain-of-thought. In EcomInsrut, these atomic tasks are devided into two parts. One part is transformed from complete information in the high quality dataset through heuristic strategies, while the
other part is constructed by utilizing ChatGPT to annotate pseudo-labelling. After collecting the above two parts of raw data, we combined the data samples with task-specific instruction schema to obtain instruction data. Table 2 shows the detailed statistics of EcomInsrut, which includes a total of 134 tasks and 2.6 million instruction data. In the following sections, we will describe the collection of raw data for the open-source E-commerce NLP tasks (Section ) and the atomic tasks (Section ). Additionally, we will describe how to map raw data samples to instruction data in Section .
# Raw Data from Open-Source Benchmarks
We collected publicly available and widely used NLP benchmark datasets in the E-commerce domain as our raw data, mainly sourced from research websites and data competition platforms. Based on this, we identified several major task paradigms: • Classification: Classification tasks play a vital role in E-commerce, as it helps to automatically organize and categorize textual data, such as product descriptions, customer reviews, and inquiries. The main objective of these tasks is to accurately predict the category, topic, or intent accurately based on the input content. These tasks can take the form of multi-class classification, binary classification, or multi-label classification. • Extraction: Extraction tasks are widely utilized to extract important information from unstructured textual data. For instance, review-based extractive question-answering involves extracting relevant information from customer reviews to answer specific questions. • Generation: Generation tasks are designed to produce novel content that fulfills the given requirements, such as dialogue reply, copywriting, title. For example, title generation aims to produce brief but distinctive title based on the attribute key-value pairs of the products, which can help to promote the product sales. • Others: other E-commerce NLP tasks. In our EcomInstruct dataset, it primarily refers to the task of Named Entity Recognition (NER) within various label schemes, such as address-related NER and product attribute-related NER. As the output of NER encompasses both the original input text (entities corresponding to positive labels) and the novel content generated by the model (None output corresponding to negative labels), it thus constitutes a hybrid task of extraction and generation. In this step, we collected 65 public E-commerce NLP benchmarks in total.
# Raw Data from Atomic Tasks
Based on the data derived from open-source benchmarks, we decomposed them into various atomic tasks. These tasks are transformed into datasets for instruction tuning, as described in Section , to further expand the scale and diversity of the instruction data. On the one hand, atomic tasks can be constructed by leveraging the complete information from the original data, including the ground truth labels that either exists in the original
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0496/04964309-8f7e-4a1e-9a3b-7dcad104cd8d.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: The complete schema of the atomic tasks.</div>
dataset or can be inferred from it. Specifically, 3 main strategies are employed for constructing atomic tasks: (1) Task Simplification. We can adjust the model inputs and ground truth labels to simplify the original tasks. For example, we can obtain entity detection and entity typing tasks by simplifying named entity recognition (NER) task. (2) Task Reversal. For some original tasks, we can switch the order of model input and output to construct new tasks. For instance, we can build a question generation task from the question answering (QA) task, and the task of generating product description given product title can be transformed into a title generation task. (3) Sample Recombination. We can also use information from multiple samples in a dataset to form a new sample, thereby obtaining different tasks. For example, based on the product matching task given two product titles and attributes, we can split and shuffle the product titles and attributes in these samples to construct a task that matches a product title and a product attribute. On the other hand, we can construct instruction datas based on basic E-commerce information within the datasets, such as product metadata and user queries without ground truth labels from the original data. For these input-only datas, we utilize ChatGPT to generate outputs as pseudo-labels for model training. For instance, we can devise various instruction tasks based on search queries, such as query rewriting, query segmentation, and query-based question generation, to compose a diverse set of atomic tasks. The complete schema of the atomic tasks is shown in Figure 1.
# Mapping Raw Data to Instruction Data
Building upon the raw data, we further developed the instruction data. Firstly, we devised the schema of the instruction data, which encompasses six primary components: 1. Task Description: a high-level overview of the task at hand.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4322/43224623-9d7f-4ee9-a0e0-4e77315a93ca.png" style="width: 50%;"></div>
2. Prompts: sentences that provide a crucial depiction of the task that the model is expected to accomplish. 3. Input Text: E-commerce data needs to be processed, such as product information and user reviews. 4. Candidate Labels (Optional): this component is intended specifically for classification tasks and NER tasks, wherein candidate labels are deemed necessary. 5. Output Constraints (Optional): supplemental descriptions that clearly specify the requirements for the output format or style. 6. Output: the ground truth output desired by the user. We asked domain experts to write dataset-specific task descriptions, prompts and output constraints for each dataset, which is a non-trivial work. Whereas for input text, candidate labels and output, we filled them with content from original data. Examples of instruction data can be found in Figure 2. Despite the relatively high quality of data from open source benchmark datasets, it is inevitable that some noise will be present. Therefore, EcomInstruct underwent two data filtering and human calibration processes. Firstly, we implemented a rule-based filtering approach that primarily excluded data instances containing illegal characters in the input, null output, and excessively long data instances. We also standardized the whitespace characters in the content. Secondly, we applied a model-based filtering approach utilizing Alpaca GarbageCollector2 to flag low-quality instructional data to be discarded.
Additionally, for each dataset, we ensured that at least one annotator conducted a secondary check on a random sample of 200 data instances.
# EcomGPT: Training E-commerce Large Language Model with EcomInstuct
Our EcomGPT is constructed by fine-tuning BLOOMZ with our EcomInstruct dataset. Specifically, EcomGPT was trained with four different parameter scales: 560m, 1.7b, 3b, and 7.1b. AdamW (Loshchilov and Hutter 2017) optimizer is employed for model training, with learning rate set of 2e5 and weight decay of 0. We utilize a cosine learning rate schedule, warming up over 3% of the training steps. The model is fine-tuned with 3 epochs, with the batch size per device set to 4 and the gradient accumulation step set to 8. The maximum sequence length is 1024. All experiments are run on 4 NVIDIA A100 SXM4 80GB GPUs. During model training, we expect the model to learn to generate response given the instruction and input text, thus we compute the loss function by considering only the response tokens and ignoring the input tokens.
# Experiments
# Experiment Setup
Baselines We classified our baseline models into two categories: foundational pre-trained large models and instructionfollowing large language models. The former includes the
BLOOM (Scao et al. 2022), which has a decoder-only architecture and ranges from 560 million to 176 billion parameter scales. The latter includes BLOOMZ (Muennighoff et al. 2022), which applies multi-task instruction tuning to the BLOOM model families to obtain fine-tuned instructionfollowing variants, and ChatGPT, the most advanced commercially available large language model . ChatGPT applies instruction fine-tuning and RLHF techniques to fine-tune and align GPT3. To compare our EcomGPT model with BLOOM and BLOOMZ, we selected the 560m, 1.7b, 3b, and 7.1bparameters models. We estimated the upper bound on the generalization performance of the 7b-parameters model on unseen dataset or tasks. Specifically, we randomly selected 800 training data for each evaluation task, and independently trained BLOOMZ 7.1b, taking the average of the performance of these models on the corresponding task as the upper bound on performance. Evaluation Metric. In EcomInstruct, all tasks can be converted into generative paradigms, thus we can evaluate them with automatic evaluation metrics for text generation. For various tasks, ROUGE-L (Lin 2004) is employed to evaluate the model outputs following previous works (Wang et al. 2022b; Mishra et al. 2022). Additionally, for classification and NER tasks, we also utilize precision, recall and F1 as evaluation metrics, and report both micro-average and macro-average results. For opendomain generation tasks such as product title generation, we contend that automatic reference-based evaluation metrics such as ROUGE-L do not sufficiently reflect the model performance, which is also an exceedingly complex issue in the natural language generation domain (Celikyilmaz, Clark, and Gao 2020). Therefore, we further conducted human evaluation to measure the model performance on the generation tasks.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c5bb/c5bb2288-a005-48c6-b166-4ed68722c94d.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 3: The details of our evaluation datasets.</div>
Dataset Split. The EcomInstruct dataset is divided into two partitions, namely training and testing. The test set comprises 12 tasks chosen from diverse datasets, encompassing four major categories, namely classification (e.g., coarse-grained/finegrained product classification, review topic classification), generation (e.g., product title generation), extraction (e.g., review-based QA, attribute value detection), and others (e.g., E-commerce named entity recognization). To ensure efficient testing, 500 instances of each task were randomly selected as test data, resulting in a final test set of 6,000 data instances.
The remaining 122 datasets were allocated for training, from which up to 800 data instances were sampled for each dataset as the training set. Ultimately, the EcomGPT was trained on a total of 85,746 instances of E-commerce data. For a more detailed scaling experiments on the number of training samples for each dataset, please refer to Section . Generalization Types. Conventional supervised learning evaluates a model’s capacity to generalize within a given distribution, wherein the model learns from labeled instances of specific domains and tasks, and is subsequently tested on data that conforms to the same distribution for the same domain and task. In contrast, for E-commerce LLM, our emphasis lies in the model’s ability to generalize to data outside the distribution. In this study, we correspond a data instance to three levels, namely task paradigm (e.g., generation task, classification task), task (e.g., the classification paradigm comprises tasks with different objectives like product item classification, intent detection, etc.), and dataset (e.g., for the intent detection task, it encompasses SGD (Rastogi et al. 2020) and JDDC (Chen et al. 2020) datasets, consisting of distinct label sets). The model’s ability to generalize to unseen tasks/datasets at the task and dataset levels represents the most desirable and practical feature. Therefore, we primarily focus on the model’s generalization capability on unseen tasks/datasets in the main experiment. Additionally, in Sections and , we evaluate the model’s performance under cross task paradigms and cross-language settings.
# Main Experiments
Table 4 presents the results of the automated metrics-based evaluation conducted on new datasets and tasks, from which we can conclude that: (1) In terms of average performance on unseen datasets, EcomGPT, even with the lowest number of parameters (560 million), outperforms ChatGPT, which has over 100 billion parameters (exceeding EcomGPT by 100,000 times). Moreover, EcomGPT’s performance consistently improves as the model parameters scale, demonstrating its remarkable generalization ability for E-commerce tasks. (2) By training on EcomInstruct data, EcomGPT achieved a substantial improvement of over 20 points compared to the baseline model BLOOMZ. This suggests that excellent generalization performance of EcomGPT is not solely dependent on the backbone model. (3) Due to the lack of dialogue capability, the pre-trained language model BLOOM demonstrates poor performance, approaching 0 and being unstable. Interestingly, the difference between the performance boost achieved by the xP3 dataset, which contains over 78 million general instruction data, and that obtained by the EcomInstuct dataset, which has roughly 200,000 E-commerce instruction data for training, is approximately 4 points. This highlights the more effective role of domain-specific instruction data for vertical scenarios in enhancing model generalization capability. (4) We conducted supervised fine-tuning of BLOOMZ 7b using the training set of the test tasks to estimate the upper bound of the model’s generalization performance. Our findings indicate that the current EcomGPT still has significant room for improvement in terms of generalization capability. Furthermore, in order to enhance the reliability of the evaluation, particularly for the generation tasks, where automated
Model Type
Model
Unseen Dataset
Unseen Task
Micro F1
Macro F1
Rouge
Poduct Align
Review Topic Classify
Product Select
Micro F1
Macro F1
Rouge
Micro F1
Macro F1
Rouge
Rouge
PLM
BLOOM (560m)
3.33
2.10
5.64
0.17
0.15
6.76
13.22
10.96
1.26
6.06
BLOOM (1b7)
4.15
2.78
6.00
0.10
0.10
1.60
16.17
14.95
6.72
6.72
BLOOM (3b)
2.94
1.43
7.89
0.10
0.20
1.86
0.38
0.18
5.50
7.99
BLOOM(7b1)
4.29
2.50
7.31
0.10
0.13
0.97
7.11
3.61
4.96
9.47
Instruction
BLOOMZ (560m)
24.62
25.60
24.03
21.80
21.80
55.53
30.49
32.13
23.60
0.00
BLOOMZ (1b7)
18.60
18.87
15.10
0.40
0.40
0.40
32.06
34.01
26.38
2.27
BLOOMZ (3b)
29.80
30.05
26.38
10.42
10.80
16.53
30.81
32.14
23.25
11.65
BLOOMZ (7b1)
26.75
27.07
25.21
6.00
6.00
8.00
49.37
50.39
41.43
15.14
ChatGPT
37.30
40.71
43.92
41.60
41.60
71.02
51.22
51.80
42.55
27.39
Ours
EcomGPT (560m)
41.28
38.21
48.88
50.15
50.15
81.41
42.39
50.88
32.25
10.74
EcomGPT (1b7)
42.30
39.07
53.24
51.20
52.20
81.23
47.38
52.68
37.81
32.38
EcomGPT (3b)
48.37
45.04
59.20
53.20
53.20
82.13
53.91
56.12
44.99
52.53
EcomGPT (7b1)
52.89
50.17
62.83
55.20
55.20
84.67
59.03
60.74
50.25
56.39
Upper-bound(est.)
SFT(7b1)
74.73
71.01
73.87
67.90
67.90
89.06
85.86
89.22
82.96
97.60
evaluation metrics fall short in reflecting the performance of the model, a human evaluation was deliberately incorporated. As illustrated in Figure 3, we randomly selected 100 samples per task and ask the annotators to judge which one of the outputs of EcomGPT and ChatGPT is better or tied. The results show that, with the exception of generation tasks, the winning or tying rate of EcomGPT in the human evaluation maintains the same overall trend as the Rouge value. The Pearson coefficient between the two is 0.2, indicating a positive correlation overall and confirming the reliability of the human evaluation. Upon analyzing the output, we observed that for certain tasks with complex input or output formats, such as named entity recognition, ChatGPT struggled and often displayed a meaningless response like “sorry, I can’t retrieve the information”. In the case of generation tasks, such as product title generation, ChatGPT typically generated excessively long sentences, which were inconsistent with the concise and attention-grabbing style of human written titles. While ChatGPT was able to solve some relatively simple tasks, such as product selection (with a solution rate of 78% in human evaluation), the model’s Rouge value remained low. We attributed this to the abundance of redundant replies in the output of ChatGPT, which hindered its practical application, since time-consuming task-specific parsing of model output is required. In conclusion, EcomGPT exhibited superior semantic understanding of E-commerce data.
# Ablation Experiments on CoT Tasks
As described in Section , a considerable proportion of EcomInstruct consists of atomic tasks that are constructed using data specific to the E-commerce domain. These atomic tasks encompass a variety of generic semantic understanding capabilities, which are extensively utilized during the intermediate stage of the model’s solution of the original task. Drawing a parallel with prior research on Chain-of-Thought (Wei et al. 2022; Wang et al. 2022a), we refer to these atomic tasks as Chain-of-Task tasks (CoT tasks). The CoT task empowers the model to imbibe generic capabilities that are implicitly utilized while handling E-commerce tasks, thereby
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6c68/6c685d32-6663-4461-889f-d23a6ac26021.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Human Evaluation results.</div>
playing a pivotal role in enhancing the model’s generalization ability. To validate our assumptions and the effectiveness of the CoT task, we conduct ablation experiments on the CoT task at a high level in Section . Furthermore, in Section , we take a deeper look into the benefits of CoT tasks across varied dimensions, including data, tasks, and task paradigms. Overall Gain from CoT Tasks The CoT tasks were derived from a combination of two sources: data with pseudolabels generated by ChatGPT and high-quality raw data with golden labels. As illustrated in Table 5, when both components of the CoT data are sequentially removed, there is a significant degradation in the performance of the EcomGPT. Furthermore, the model trained solely using original Ecommerce data fails to outperform ChatGPT’s performance in Table 4. This observation suggests that solely relying on domain data for instruction learning is insufficient to enhance
the generalization ability of the pendant domain model. Additionally, we observe a more substantial drop in performance upon removal of the CoT task constructed from high-quality data containing golden labels, which is due to the fact that the amount of data built from ChatGPT is relatively small while containing some errors or noise. The significant improvement achieved with the CoT task inspires us to even with limited domain data, a series of atomic tasks constructed from the domain data can endow the model with superior generalization capabilities.
Training Dataset
Micro F1
Macro F1
Rouge
Full
48.37
45.04
59.20
w/o pseudo label CoT
44.98
41.79
55.46
w/o golden label CoT
26.64
23.64
35.02
Table 5: Overall abaltion on CoT Tasks. w/o pseudo label CoT means without CoT task whose label is generate by ChatGPT. w/o golden label CoT represents without CoT task whose label is inferred from the original golden labels.
Training
Ecom
Youku
Amazon
CCKS
JDDC
Avg
Full
73.79
91.42
61.31
70.40
31.80
65.74
w/o Ecom-R
72.77
90.67
62.55
74.00
38.20
67.64
w/o Youku-R
73.10
91.07
59.67
76.00
36.20
67.21
w/o Amazon-R
73.85
90.55
60.63
72.00
26.20
64.65
w/o CCKS-R
74.47
91.30
59.90
69.60
37.20
66.49
w/o JDDC-R
73.73
91.19
58.13
71.20
27.80
64.41
Table 6: Ablation experiments on CoT tasks at dataset level. “w/o *-R” denotes without CoT instruction data that is related to the “*”.
Training
QA
NER
IC
Unseen Dataset
Micro F1
Macro F1
Rouge
Full
59.23
80.67
65.30
48.37
45.05
59.20
w/o QA-R
56.75
79.78
61.55
40.18
37.89
52.37
w/o NER-R
59.00
77.55
63.30
45.50
43.97
55.14
w/o IC-R
57.54
80.49
60.40
41.12
36.94
52.28
Cross Gain from CoT Tasks In this section, we conduct extensive ablation experiments on CoT data, aiming to investigate the benefits of CoT data at the dataset, task, and task paradigm levels. Dataset Level. In the Table 6, we remove the CoT task associated with a specific dataset from the training set to observe its impact. To prevent data leakage, we avoided introducing CoT tasks corresponding to the test dataset in the training set of EcomInstruct. So at the dataset level, we performed held-in evaluation, i.e., evaluating the selected tasks in the training set. Our findings indicate that CoT tasks derived from the same dataset provided steady gains for the original task. However, in cross-dataset scenarios, the efficacy of CoT tasks is dependent on the data types corresponding to
Training
CLS
Ext
Other
Unseen Dataset
Micro F1
Macro F1
Rouge
Full
67.87
52.17
80.67
48.37
45.05
59.20
w/o CLS-R
65.69
47.49
80.38
46.87
43.75
57.00
w/o Ext-R
58.71
27.47
79.14
43.73
42.81
47.36
w/o Gen-R
56.67
50.87
80.58
41.38
40.20
54.39
Table 8: Ablation experiments on CoT tasks at task paradigm level.
the two datasets: for the same type of data that overlap in the task chain, the CoT tasks can provide a collaborative effect. For instance, the Ecom and Youku datasets both contain product titles, resulting in mutual gains. Conversely, there is no gain between CCKS and JDDC datasets, as their data types are addresses and dialogues, respectively, despite belonging to the same classification task. Task Level. In the Table 7, we eliminate all CoT tasks associated with a given task and report the model’s performance on unseen tasks and data. For example, for the NER task, we exclude all entity detection and entity classification tasks from the training set. Our results demonstrate that CoT tasks are advantageous for both similar and dissimilar tasks. Notably, CoT tasks related to QA exhibit the greatest enhancement in generalization capacity to other tasks, while concurrently exhibiting greater difficulty in generalizing from CoT tasks from other tasks, which aligns with the finding in prior work (Zhou et al. 2022). We argue that, for instructionfollowing LLMs, tasks can be naturally abstracted to QA tasks, thereby playing a crucial role in enhancing model generalization ability. Task Paradigm Level. As demonstrated in Table 6, certain CoT tasks of classification tasks do not exhibit advantage over held-in tasks of other paradigms at the dataset level. However, as shown in Table 8, when viewed from a higher-level perspective of task paradigms, there is greater overlap among the data or task formats of different paradigms. Consequently, CoT tasks from different paradigms display a consistent gain for each other, with the CoT tasks of the classification tasks even exhibiting a greater gain over other paradigm tasks than on its own.
# Conclusion
This paper presents EcomInstruct, the first instruction-tuning dataset tailored for the E-commerce domain, encompassing two different part of instruction data, while the second part comprises atomic tasks based on the basic data types in the Ecommerce domain, also known as Chain-of Task (CoT) tasks. These CoT tasks are intermediate tasks implicitly involved in solving a targeted final task. Benefiting from the fundamental semantic understanding capabilities acquired from the Chain-of-Task tasks, EcomGPT , trained with EcomInstruct, outperforms ChatGPT in term of cross-dataset/task generalization on E-commerce tasks. The advantages of leveraging CoT tasks suggest that, within vertical domain scenarios, we can devise diverse atomic tasks specifically tailored to the domain data to enhance the model’s generalization ability.
# References
Ahmadvand, A.; Kallumadi, S.; Javed, F.; and Agichtein, E. 2020. Jointmap: joint query intent understanding for modeling intent hierarchies in e-commerce search. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 1509– 1512. Beltagy, I.; Lo, K.; and Cohan, A. 2019. SciBERT: A Pretrained Language Model for Scientific Text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 3615–3620. Hong Kong, China: Association for Computational Linguistics. Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877– 1901. Celikyilmaz, A.; Clark, E.; and Gao, J. 2020. Evaluation of text generation: A survey. arXiv preprint arXiv:2006.14799. Chen, M.; Liu, R.; Shen, L.; Yuan, S.; Zhou, J.; Wu, Y.; He, X.; and Zhou, B. 2020. The JDDC Corpus: A LargeScale Multi-Turn Chinese Dialogue Dataset for E-commerce Customer Service. In Proceedings of the Twelfth Language Resources and Evaluation Conference, 459–466. Cheng, X.; Bowden, M.; Bhange, B. R.; Goyal, P.; Packer, T.; and Javed, F. 2021. An end-to-end solution for named entity recognition in ecommerce search. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, 15098–15106. Chiang, W.-L.; Li, Z.; Lin, Z.; Sheng, Y.; Wu, Z.; Zhang, H.; Zheng, L.; Zhuang, S.; Zhuang, Y.; Gonzalez, J. E.; et al. 2023. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023). Cui, J.; Li, Z.; Yan, Y.; Chen, B.; and Yuan, L. 2023. ChatLaw: Open-Source Legal Large Language Model with Integrated External Knowledge Bases. arXiv preprint arXiv:2306.16092. Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 4171–4186. Minneapolis, Minnesota: Association for Computational Linguistics. Escursell, S.; Llorach-Massana, P.; and Roncero, M. B. 2021. Sustainability in e-commerce packaging: A review. Journal of cleaner production, 280: 124314. Hoffmann, J.; Borgeaud, S.; Mensch, A.; Buchatskaya, E.; Cai, T.; Rutherford, E.; Casas, D. d. L.; Hendricks, L. A.; Welbl, J.; Clark, A.; et al. 2022. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556. Huang, K.; Altosaar, J.; and Ranganath, R. 2019. Clinicalbert: Modeling clinical notes and predicting hospital readmission. arXiv preprint arXiv:1904.05342.
Penedo, G.; Malartic, Q.; Hesslow, D.; Cojocaru, R.; Cappelli, A.; Alobeidli, H.; Pannier, B.; Almazrouei, E.; and Launay, J. 2023. The RefinedWeb dataset for Falcon LLM: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116. Poerner, N.; Waltinger, U.; and Sch¨utze, H. 2020. E-BERT: Efficient-Yet-Effective Entity Embeddings for BERT. In Findings of the Association for Computational Linguistics: EMNLP 2020, 803–818. Online: Association for Computational Linguistics. Pontiki, M.; Galanis, D.; Pavlopoulos, J.; Papageorgiou, H.; Androutsopoulos, I.; and Manandhar, S. 2014. SemEval-2014 Task 4: Aspect Based Sentiment Analysis. In Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014), 27–35. Dublin, Ireland: Association for Computational Linguistics. Qiu, Y.; Zhao, C.; Zhang, H.; Zhuo, J.; Li, T.; Zhang, X.; Wang, S.; Xu, S.; Long, B.; and Yang, W.-Y. 2022. Pretraining Tasks for User Intent Detection and Embedding Retrieval in E-commerce Search. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 4424–4428. Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; Sutskever, I.; et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8): 9. Rae, J. W.; Borgeaud, S.; Cai, T.; Millican, K.; Hoffmann, J.; Song, F.; Aslanides, J.; Henderson, S.; Ring, R.; Young, S.; et al. 2021. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446. Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; and Liu, P. J. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1): 5485–5551. Rastogi, A.; Zang, X.; Sunkara, S.; Gupta, R.; and Khaitan, P. 2020. Towards scalable multi-domain conversational agents: The schema-guided dialogue dataset. In Proceedings of the AAAI conference on artificial intelligence, volume 34, 8689– 8696. Sanh, V.; Webson, A.; Raffel, C.; Bach, S. H.; Sutawika, L.; Alyafeai, Z.; Chaffin, A.; Stiegler, A.; Scao, T. L.; Raja, A.; et al. 2021. Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207. Scao, T. L.; Fan, A.; Akiki, C.; Pavlick, E.; Ili´c, S.; Hesslow, D.; Castagn´e, R.; Luccioni, A. S.; Yvon, F.; Gall´e, M.; et al. 2022. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100. Singhal, K.; Azizi, S.; Tu, T.; Mahdavi, S. S.; Wei, J.; Chung, H. W.; Scales, N.; Tanwani, A.; Cole-Lewis, H.; Pfohl, S.; et al. 2023. Large language models encode clinical knowledge. Nature, 1–9. Smith, S.; Patwary, M.; Norick, B.; LeGresley, P.; Rajbhandari, S.; Casper, J.; Liu, Z.; Prabhumoye, S.; Zerveas, G.; Korthikanti, V.; et al. 2022. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990.
Taori, R.; Gulrajani, I.; Zhang, T.; Dubois, Y.; Li, X.; Guestrin, C.; Liang, P.; and Hashimoto, T. B. 2023. Stanford alpaca: An instruction-following llama model. Taylor, R.; Kardas, M.; Cucurull, G.; Scialom, T.; Hartshorn, A.; Saravia, E.; Poulton, A.; Kerkez, V.; and Stojnic, R. 2022. Galactica: A large language model for science. arXiv preprint arXiv:2211.09085. Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozi`ere, B.; Goyal, N.; Hambro, E.; Azhar, F.; et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971. Tsagkias, M.; King, T. H.; Kallumadi, S.; Murdock, V.; and de Rijke, M. 2021. Challenges and research opportunities in ecommerce search and recommendations. In ACM Sigir Forum, volume 54, 1–23. ACM New York, NY, USA. Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. Advances in neural information processing systems, 30. Wang, X.; Jiang, Y.; Bach, N.; Wang, T.; Huang, Z.; Huang, F.; and Tu, K. 2021. Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 1800–1812. Wang, X.; Wei, J.; Schuurmans, D.; Le, Q. V.; Chi, E. H.; Narang, S.; Chowdhery, A.; and Zhou, D. 2022a. SelfConsistency Improves Chain of Thought Reasoning in Language Models. In The Eleventh International Conference on Learning Representations. Wang, Y.; Mishra, S.; Alipoormolabashi, P.; Kordi, Y.; Mirzaei, A.; Naik, A.; Ashok, A.; Dhanasekaran, A. S.; Arunkumar, A.; Stap, D.; et al. 2022b. SuperNaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 5085–5109. Wang, Z.; Mayhew, S.; Roth, D.; et al. 2019. Cross-lingual ability of multilingual bert: An empirical study. arXiv preprint arXiv:1912.07840. Wei, J.; Bosma, M.; Zhao, V. Y.; Guu, K.; Yu, A. W.; Lester, B.; Du, N.; Dai, A. M.; and Le, Q. V. 2021. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652. Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.; Chi, E.; Le, Q. V.; Zhou, D.; et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35: 24824–24837. Wu, S.; Irsoy, O.; Lu, S.; Dabravolski, V.; Dredze, M.; Gehrmann, S.; Kambadur, P.; Rosenberg, D.; and Mann, G. 2023. Bloomberggpt: A large language model for finance. arXiv preprint arXiv:2303.17564. Xu, S.; Li, H.; Yuan, P.; Wang, Y.; Wu, Y.; He, X.; Liu, Y.; and Zhou, B. 2021. K-PLUG: Knowledge-injected Pre-trained Language Model for Natural Language Understanding and
Generation in E-Commerce. In Findings of the Association for Computational Linguistics: EMNLP 2021, 1–17. Yang, H.; Liu, X.-Y.; and Wang, C. D. 2023. FinGPT: OpenSource Financial Large Language Models. arXiv preprint arXiv:2306.06031. Zhang, D.; Yuan, Z.; Liu, Y.; Zhuang, F.; Chen, H.; and Xiong, H. 2020a. E-BERT: A phrase and product knowledge enhanced language model for e-commerce. arXiv preprint arXiv:2009.02835. Zhang, H.; Hennig, L.; Alt, C.; Hu, C.; Meng, Y.; and Wang, C. 2020b. Bootstrapping Named Entity Recognition in ECommerce with Positive Unlabeled Learning. ECNLP 3, 1. Zhang, S.; Roller, S.; Goyal, N.; Artetxe, M.; Chen, M.; Chen, S.; Dewan, C.; Diab, M.; Li, X.; Lin, X. V.; et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068. Zhang, W.; Wong, C.-M.; Ye, G.; Wen, B.; Zhang, W.; and Chen, H. 2021. Billion-scale pre-trained e-commerce product knowledge graph model. In 2021 IEEE 37th International Conference on Data Engineering (ICDE), 2476–2487. IEEE. Zhang, X.; Yang, Q.; and Xu, D. 2023. XuanYuan 2.0: A Large Chinese Financial Chat Model with Hundreds of Billions Parameters. arXiv preprint arXiv:2305.12002. Zhao, J.; Chen, H.; and Yin, D. 2019. A dynamic productaware learning model for e-commerce query intent understanding. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, 1843–1852. Zhao, W. X.; Zhou, K.; Li, J.; Tang, T.; Wang, X.; Hou, Y.; Min, Y.; Zhang, B.; Zhang, J.; Dong, Z.; et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223. Zhou, J.; Lin, Z.; Zheng, Y.; Li, J.; and Yang, Z. 2022. Not All Tasks Are Born Equal: Understanding Zero-Shot Generalization. In The Eleventh International Conference on Learning Representations. Zhu, T.; Wang, Y.; Li, H.; Wu, Y.; He, X.; and Zhou, B. 2020. Multimodal Joint Attribute Prediction and Value Extraction for E-commerce Product. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2129–2139.
# Related Work
# Large Language Models
Large Language Models
Language models are foundation of natural language processing, modeling the probability distributions of word sequences. After Transformer (Vaswani et al. 2017) is proposed, BERT (Devlin et al. 2019) promotes the paradigm of pretraining a language model on large unsupervised corpus and fine-tuning it on small supervised datasets. GPT-2 (Radford et al. 2019) and T5 (Raffel et al. 2020) present that various NLP tasks can be unified into a text generation task. Recently, decoder-only Transformer model has become the mainstream architecture of language models. GPT-3 (Brown et al. 2020) releases a large language model (LLM) with up to 175 billion parameters. Following the scaling law (Kaplan et al. 2020) for LLMs, researchers build a series of LLMs such as Megatron-Turing NLG (Smith et al. 2022), Gopher (Rae et al. 2021), Chinchilla (Hoffmann et al. 2022), OPT (Zhang et al. 2022), BLOOM (Scao et al. 2022), LLaMA (Touvron et al. 2023), Falcon (Penedo et al. 2023). In addition, previous works demonstrate that fine-tuning LLMs on numerous supervised NLP tasks can effectively enhance the models’ zero-shot cross-task generalization ability, named instruction tuning (Wei et al. 2021; Sanh et al. 2021). InstructGPT (Ouyang et al. 2022) integrates instruction tuning and RLHF techniques to train GPT-3, allowing it to align with human preferences. Alpaca (Taori et al. 2023) and Vicuna (Chiang et al. 2023) fine-tune LLaMA with synthetic instructions. OPT-IML (Iyer et al. 2022) and BLOOMZ (Muennighoff et al. 2022) are instruction-following models based on OPT and BLOOM, respectively.
# Domain-specific Large Language Models
Following the introduction of BERT model, numerous works devote to retaining or continuing pre-training BERT model on domain-specific data, such as BioBERT (Lee et al. 2020) and for biomedical domain, ClinicalBERT (Huang, Altosaar, and Ranganath 2019) for clinical domain, SciBERT (Beltagy, Lo, and Cohan 2019) for scientific domain, and E-BERT (Zhang et al. 2020a) for E-commerce doamin. Since the remarkable success of decoder-only LLMs, researchers have been motivated to incorporate domain-specific data for training auto-regressive models. Most related works adopt a strategy of fine-tuning a general pre-trained LLM using domain-specific datasets. Med-PaLM (Singhal et al. 2023) and Minerva (Lewkowycz et al. 2022) fine-tune PaLM on biomedical and mathematical tasks, respectively. Galactica (Taylor et al. 2022) is an LLM for resolving scientific tasks. For financial domain, BloombergGPT (Wu et al. 2023) trains an LLM on both financial and general data sources from scratch, FinGPT (Yang, Liu, and Wang 2023) focuses on adaption of other open-source LLMs, and Xuanyuan (Zhang, Yang, and Xu 2023) releases a Chinese chat model based on BLOOM. In the legal domain, Lawyer LLaMA (Huang et al. 2023) and ChatLaw (Cui et al. 2023) fine-tuned LLMs for providing legal consultation services. To our knowledge, no auto-regressive LLMs designed for addressing E-commerce tasks have been released. Additionally, our work constructs a multi-task instruction dataset in
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f3fa/f3fab93f-8ac0-4c0b-87be-bb422ed4186a.png" style="width: 50%;"></div>
Figure 4: Scaling experiments on the number of training tasks, with the vertical axis representing the Rouge of the EcomGPT on the unseen dataset.
E-commerce field for the first time to improve the zero-shot model performance for E-commerce tasks.
# More Analysis Experiments mpact on Model Generalization
We investigated the scaling generalization of EcomGPT from three dimensions: model size, training task quantity, and training data volume for each task. The impact of each factor on model performance is illustrated in Figures 4 and 5. We can conclude that: More diverse domain training tasks benefit generalization capacity. We instruction fine-tuned EcomGPT using datasets that include different numbers of training tasks, as shown in Figure 4. To ensure comparability, we randomly sampled the training tasks, with the dataset containing fewer tasks being a subset of the dataset containing more tasks. We observed that as the number of tasks used to train the model increased, the model’s performance on unseen dataset improved. This improvement was particularly noticeable for models with larger capacity and more parameters, such as the 7b-parameter EcomGPT, which showed no signs of slowing down in performance improvement even when the number of tasks reached 120. Our findings are consistent with previous research on generalized domains and extend to vertical domains, demonstrating that enriching the task type of instruction data can significantly enhance the zero-shot capacity of the model. Excessive training instances for each task do not enhance generalization capability. Figures 5(a) and 5(b) depict the evolution of the model performance on both seen and unseen tasks as the data instances size per training task is increased. Our findings demonstrate that while more training data is advantageous for seen tasks, consistent with supervised learning principles, the number of instances per task plays a nonessential role in the generalization capacity for unseen tasks. In fact, our results indicate that the EcomGPT model requires only a few hundred data instances per task
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/58fb/58fba551-11fb-488f-bf41-5b413922e27d.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Perform on seen (held-in) dataset.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/491b/491bb728-5021-4983-81f6-5f740451150f.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Perform on unseen dataset.</div>
Figure 5: Scaling experiments on the number of training samples per task.
to thoroughly acquire the generalization ability inherent in the task. Notably, the model performance plateaus as the data instances size increases, and smaller models require fewer instances to attain convergence due to smaller model capacity. Scaling up task number takes priority over model parameter size. Based on the observations in Figure 4, we can conclude that increasing the size of the model’s parameters alone would not result in significant performance gains when the training task is insufficiently diversified. For example, in Figure 4, the difference between the 1b7 and 7b models was negligible when there were only 20 instruction tasks. Moreover, to achieve the same gain from increasing the number of training tasks fourfold (from 20 to 80) in the 560m-parameter model, one must increase the parameter size by 1000 times (from millions to billions of parameters). Nevertheless, when sufficient training tasks were provided (in Figure 5), increasing the model’s parameters could lead to sustained performance improvement, especially for larger models that are more robust to noise. In Figure 5(b), all models (560m, 1b7, and 3b) experienced a significant performance drop during the increase in training instances size due to the introduction of noise data, while the 7b model was more stable.
# Cross Task Paradigm Generalization
In contrast to direct generalization cross data and tasks, we investigated the model’s indirect generalization ability between different task paradigms. The performance of the model when removing different task paradigms from the training set is reported in Table 9, with ”Others” task paradigm mainly consisting of NER tasks in EcomInstruct. Our findings indicate that there are mutual benefits between most task paradigms, with classification and extraction tasks exhibiting the greatest improvement (average of more than 8.6). This phenomenon can be explained from two perspectives. Firstly, classification and extraction tasks share a great deal of semantic understanding of textual content in achieving task goals. For example, in the case of review sentiment classification, the model may also need to implicitly extract aspects of the text that reflect emotional tendencies. Secondly, for the EcomInstruct benchmark, we found that the intersection of datasets between classification and extraction tasks is the smallest among all task paradigms, thus providing more diverse data sources for each other, which is crucial for improving model generalization ability. At the same time, we found that other task paradigms have relatively smaller enhancement on NER tasks, especially when the model parameter size is small, some task paradigms (such as classification) even have a negative effect on NER tasks. Conversely, NER tasks have a relatively larger enhancement on other tasks. We believe that this is due to the more complex task form of NER compared to other tasks, requiring higher semantic understanding capabilities from the model. Therefore, it may be difficult to fully utilize the indirect gain between task paradigms, and as the parameter size increases, the model’s ability to utilize the mutual enhancement between task paradigms gradually becomes stronger.
Model
CLS
EXT
Other
ALL
CLS
EXT
Other
ALL
EcomGPT(560m)
EcomGPT(1b7)
Full
56.45
39.51
72.75
48.88
55.28
49.48
81.82
53.24
w/o CLS
31.96
34.39
76.68
37.62
50.64
39.98
83.45
50.15
w/o Gen
50.69
38.67
61.52
44.31
54.58
47.26
81.79
50.04
w/o Ext
43.92
15.57
70.27
35.27
53.14
23.34
80.06
43.32
w/o Other
53.88
35.19
13.26
36.47
51.03
48.05
15.94
43.18
EcomGPT(3b)
EcomGPT(7b1)
Full
67.86
52.17
80.67
59.20
72.74
55.94
82.83
62.83
w/o CLS
56.72
38.96
80.27
51.81
67.20
46.37
82.97
57.35
w/o Gen
64.06
54.77
82.41
55.08
61.09
55.07
72.53
56.98
w/o Ext
57.34
19.93
76.61
43.71
66.36
17.33
73.40
45.72
w/o Other
65.61
50.96
13.55
41.70
61.70
48.87
14.85
44.53
Table 9: Cross task paradigm generalization experiment results.
# Cross Language Generalization
EcomInstruct comprises 64 Chinese and 58 English instruction datasets, from which we investigate their mutual gain relationship. Table 10 shows the performance of EcomGPT when trained on English, Chinese, and mixed-language data. On the whole, Chinese and English instruction data can gain
Test →
EN
ZH
EN+ZH
EN
ZH
EN+ZH
Train ↓
EcomGPT(560m)
EcomGPT(1b7)
EN
33.64
22.79
26.41
44.96
17.62
26.73
+ ZH ∆
8.85
29.28
22.47
1.86
38.83
26.50
EN+ZH
42.49
52.07
48.88
46.82
56.45
53.24
+ EN ∆
9.26
18.13
15.17
7.33
5.26
5.95
ZH
33.24
33.94
33.70
39.49
51.19
47.29
EcomGPT(3b)
EcomGPT(7b1)
EN
51.38
30.52
37.48
52.62
49.25
50.37
+ ZH ∆
0.84
32.17
21.72
1.53
17.92
12.46
EN+ZH
52.22
62.69
59.20
54.15
67.17
62.83
+ EN ∆
10.90
10.21
10.44
10.23
3.46
5.72
ZH
41.32
52.48
48.76
43.93
63.70
57.11
Table 10: Cross language generalization experiment results.
from each other, which we attribute to two factors: on the one hand, the increase of instruction data can essentially improve the backbone model’s ability to follow the instruction, even if this instruction comes from a different language. On the other hand, different languages may potentially share structural and semantic correspondences, especially in verticals such as E-commerce, where such common syntactic and grammatical features in content may be amplified. For instance, both Chinese and English product information is often presented through key-value pairs of attribute names and attribute values. In addition, we found that the magnitude of English-toChinese gain (9.3 on average) is higher than that of Chineseto-English gain (3.3 on average) on models with different parameter sizes, which may be related to the choice of the backbone model, as Indo-European languages, such as English and French, are dominant in the pre-training corpus of our backbone model BLOOMZ, and thereby conferring a superior semantic understanding and utilization of the English instruction data. Moreover, different from cross task paradigm generalization, the gain across languages is more obvious on smaller model (less than 1b parameters). This phenomenon is similar to that observed in multi-lingual BERT (Wang et al. 2019), thereby prompting further research into the potential of multilingual training for smaller models, as well as strategies for extending this capability to larger models.
# Ablation Experiments on Prompt
We conducted thorough ablation experiments on prompts from various perspectives to explore how to design better instruction schema for vertical domain models. Informative vs. Uninformative Prompt. In the first block experiment of Table 11, task information and output constraints are removed from the instructions respectively, and the performance of the EcomGPT degrades in both cases. Task information proved to be instrumental in helping the model comprehend the task goal at a higher level and classify similar tasks (e.g., the difference between generating product titles and copy based on product information). Additionally, we observed from the output of EcomGPT that adding out-
Ablation on
Type
Unseen Dataset
Micro F1
Macro F1
Rouge
Format
Complete
48.37
45.04
59.20
w/o TI
47.48
44.89
56.44
w/o OC
46.95
43.91
55.61
Language
EN
48.37
45.04
59.20
ZH
47.13
44.27
56.38
ZH+EN
45.64
43.42
55.32
Diversity
Diverse
48.37
45.04
59.20
Narrow
30.89
32.49
35.38
Table 11: Ablation experiments on prompt.
put constraints can better guide the models to follow the instruction, especially for classification tasks where the models avoid outputting categories other than the given candidate labels. In future, with reference to in context learning, more information such as positive and negative samples can also be considered for introduction. Multi-lingual vs. Mono-lingual Prompt. In the second experiment, we compared mono-lingual prompts (where all tasks used prompts in the same language regardless of the language of task input) with multi-lingual prompts (where each task used prompts in the same language as the task input). Our findings showed that mono-lingual prompts performed better than multi-lingual prompts, whether in Chinese or English. Moreover, since the backbone model BLOOMZ excelled in English comprehension, the use of English prompts proved more effective than Chinese prompts. This suggests that in a multi-lingual vertical domain scenario, one primary language as the prompt may be a better choice. Diverse vs. Narrow Prompt. In the third experiment, we used the same prompt for the similar tasks, such as ”Classify the input sentence” for all classification tasks. Comparing the prompts in EcomInstruct, which are specifically designed for each task, we found that the rich prompts in EcomInstruct lead to better generalisation. This is consistent with the conclusion in Section 3.5 that models trained on more diverse instruction data exhibit better generalization abilities. While we also discovered that a single prompt allowed the model to follow instructions better for specific tasks, generating output that meets the formatting requirements (even if incorrect), especially when the number of training tasks is still small.
