# ooGLE : CAN LONG-CONTEXT LANGUAGE MODELS UNDERSTAND LONG CONTEXTS?
Jiaqi Li1,∗, Mengmeng Wang1,∗, Zilong Zheng1,†, Muhan Zhang1,2,† 1 National Key Laboratory of General Artificial Intelligence, BIGAI 2 Institute for Artificial Intelligence, Peking University
ABSTRACT
# ABSTRACT
Large language models (LLMs), despite their impressive performance in various language tasks, are typically limited to processing texts within context-window size. This limitation has spurred significant research efforts to enhance LLMs’ long-context understanding with high-quality long-sequence benchmarks. However, prior datasets in this regard suffer from shortcomings, such as short context length compared to the context window of modern LLMs; outdated documents that have data leakage problems; and an emphasis on short dependency tasks rather than long dependency tasks. In this paper, we present ooGLE , a Long Context Generic Language Evaluation benchmark for LLMs’ long context understanding. ooGLE features relatively new documents post-2022, with over 24,000 tokens per document and 6,000 newly generated questions spanning diverse domains. Human annotators meticulously crafted more than 1,100 high-quality question-answer pairs to meet the long dependency requirements. These pairs underwent thorough crossvalidation, yielding the most precise assessment of LLMs’ long dependency capabilities. The evaluation of eight state-of-the-art LLMs on ooGLE revealed key findings: (i) commercial models outperformed open-sourced models; (ii) LLMs excelled in short dependency tasks like short question-answering and cloze tasks but struggled with more intricate long dependency tasks; (iii) in-context learning and chaining thoughts offered only marginal improvements in long context comprehension; (iv) retrieval-based techniques demonstrated substantial benefits for short questionanswering, while strategies for extending context window length through optimized transformer architectures or positional encoding had limited impact on long context understanding. As such, ooGLE not only provides a systematic and comprehensive evaluation schema on long-context LLMs, but also sheds light on future development of enhanced models towards “true long-context understanding”. All evaluation codes are released at: https://github.com/bigai-nlco/LooGLE.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c732/c732f07f-7ed1-4653-bf03-2eccaf28197c.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: The LooGLE benchmark for long context understanding.</div>
Figure 1: The LooGLE benchmark for long context understanding.
1 INTRODUCTION The pursuit of enabling large language models (LLMs), such as ChatGPT (Brown et al., 2020; OpenAI, 2023; Zeng et al., 2023), to go beyond their limited context window size so as to process, comprehend,
The pursuit of enabling large language models (LLMs), such as ChatGPT (Brown et al., 2020; OpenAI, 2023; Zeng et al., 2023), to go beyond their limited context window size so as to process, comprehend,
∗Equal contributions. †Correspondence to Zilong Zheng <zlzheng@bigai.ai> and Muhan Zhang <muhan@pku.edu.cn>.
<div style="text-align: center;"></div>
<div style="text-align: center;"></div>
<div style="text-align: center;">Table 1: Comparison with other long-context benchmarks.</div>
Dataset
Avg. Words
# of Docs.
# of Ques.
Manually Label
Long Dependency Tasks
Summarization
Info. Retrieval
Timeline Reorder
Computation
Doc QA
Zero Scrolls (Shaham et al., 2023)
10,392
-
4,378
✗
✓
✗
✗
✗
✗
Long Bench (Bai et al., 2023)
8,120
-
4,750
350
✓
✓∗
✗
✓∗
✓
L-Eval (An et al., 2023)
8,008
411
2,043
2,043†
✓
✓
✗
✗
✓
LooGLE (Ours)
19,367
776
6,448
1,101
✓
✓
✓
✓
✓
∗The task is created in a synthetic manner.
or even learn from long-context textual information (Ding et al., 2023; Dao et al., 2022; Chi et al., 2023; Bulatov et al., 2023) is inevitable for next-generation of language intelligence attributed to its wide applications on real-world scenarios, such as domain-specific knowledge understanding, long-context conversational generation, long story or code generation, etc. Meanwhile, there is an increasing need for high-quality benchmarks with much longer text lengths and more challenging tasks to provide comprehensive evaluations. However, traditional benchmarks (Cohan et al., 2018; Sharma et al., 2019; Huang et al., 2021) often fall short in text length with an average number of thousand words (s Koˇ cisk´y et al., 2018; Yang et al., 2018). Besides, existing benchmarks automatically collect possibly outdated documents from existing datasets published a few years ago (Shaham et al., 2022; Trivedi et al., 2022; Wang et al., 2022; Angelidis et al., 2020), which might lead to data leakage in pre-trained LLMs and make the evaluation inaccurate. Further, the long texts are often restricted to domain-specific articles, making it hard to evaluate LLMs’ ability on generic tasks and domains. Finally, it is important to note that tasks in existing benchmarks are primarily short dependency tasks, which only require LLMs to retrieve answers from one specific sentence or paragraph, without really testing LLMs’ ability to collect pieces of information from paragraphs across the whole document and summarize them into an answer, which we call long dependency tasks. To mitigate the shortcomings of existing datasets, in this paper, we introduce a novel benchmark LooGLE , short for Long Context Generic Language Evaluation, to evaluate the long context understanding abilities of LLMs illustrated in Fig. 1. Our benchmark has the following advantages: • Extra-long realistic documents. It contains 776 latest gathered and extremely long documents with an average of 19.3k words. There are over 6,448 test instances without distribution bias for a more generalized assessment, many of which exceed 100k words. On one hand, they can better evaluate LLMs’ capability on memorizing and understanding longer text that is far beyond their context window size. On the other hand, the excessive length is well suited to the common usage of long text scenarios. • Manually designed both short and long dependency tasks. It is composed of 7 major tasks to evaluate LLMs’ ability to understand both short and long dependency content. We refer “long dependency” tasks as those that require the understanding of the inter-dependency across multiple evidence widely spanning over the entire long text. We delicately design 5 types of long dependency tasks and recruited a group of human annotators to manually create 1101 long dependency QuestionAnswer (QA) instances, despite the high costs and huge effort involved in this process. • Relatively new documents. Our benchmark comprises texts all published after 2022 which ensures that most modern LLMs (at the date of submission) have not been pre-trained on these documents, forcing them to rely on their in-context learning ability rather than memorization. In contrast, existing benchmarks are usually a combination of content from traditional NLP dataset, whose world knowledge may have already been learned by LLMs and thus are less convincing for assessment. Furthermore, our data collection process is fully open-sourced, making it easy for the community to reconstruct/update the benchmark with newer documents, possibly on a yearly basis. • Cross-domain generic data. Our benchmark is derived from popular open-source documents, including arXiv papers, Wikipedia articles, and movie and TV scripts, spanning diverse domains and multiple categories such as academia, history, sports, politics, arts, events, and entertainment. We conduct a comprehensive evaluation of 8 representative LLMs on LooGLE . We specifically select LLMs which have made great effort in addressing the challenge of understanding long contexts as the baselines. The results indicate that better base models with a larger context window size generally achieve better performance. However, all models experience a significant performance decline in long dependency tasks, indicating there is a desperate need to improve the true long dependency
understanding capabilities of LLMs. Our dataset serves as an up-to-date benchmark for cutting-edge assessment and research on the long context understanding and modeling of LLMs.
# 2 RELATED WORK
Existing models for long context understanding. There are increasing research interests in developing methods to extend LLMs’ context window size, such as utilizing recurrent memory, sparse attention (Meister et al., 2021), external memory and etc.(Chen et al., 2023c; Xiong et al., 2023; Li et al., 2023a). The most popular way is to develop improved transformer architectures Dong et al. (2023). Efficient transformers (Tay et al., 2020; 2022) are proposed to decrease the memory and time complexity to efficiently model longer texts. Unlike efficient transformers that simplify the attention structure, recurrent transformer (Bulatov et al., 2022; Bessonov et al., 2023) keeps the full self-attention mechanism. History information of previous segments is cached and will be leveraged when the subsequent segment is fed into the model without a context fragmentation problem. Fine-tuned models on long documents Wu et al. (2021) are also explored, but they are often effort-costing and face difficulties in collecting ground truth fine-tuning data for long text tasks. Apart from approaches which are developed from modeling and parameter updating aspects, there are also works incorporating external memory structures and compression techniques for LLMs or using task-oriented process optimization strategies (Gidiotis & Tsoumakas, 2020; Zhou et al., 2022; Ram et al., 2023; Izacard et al., 2022). Existing datasets for long context understanding. There are a growing number of benchmarks proposed to test LLMs’ long context understanding ability (Shaham et al., 2023; Li, 2023). ZeroSCROLLS, L-Eval and LongBench are the three most recent ones. ZeroSCROLLS (Shaham et al., 2023) automatically processes datasets from different sources into a unified input format with an average of 10k words. However, it mainly focuses on collecting documents and tasks from existing datasets and relies on automatic metrics for limited model comparisons (Shaham et al., 2022). LEval (An et al., 2023) differs in re-annotating the data and instructions from similar public datasets with smaller sizes to ensure the quality. Besides, it optimizes the evaluation procedures and baselines to get more accurate conclusions. LongBench (Bai et al., 2023) provides a bilingual and multi-task dataset featuring diverse sequences of varying lengths, distributions, patterns, languages and domains for a comprehensive evaluation of long context understanding. Nonetheless, it encompasses texts of only thousands of words and tasks mostly restricted to short-term information extraction. Moreover, there are few types of “long dependency” tasks in previous datasets, except for summarization (which LLMs are validated to perform well on) and synthesized tasks like data aggregation and retrieving. To finish those tasks, LLMs solely need to locate pieces of information from the lengthy source input and aggregate them together. In contrast, we propose LooGLE which contains long dependency tasks that are much more challenging, such as event timeline reordering, comprehension/reasoning, and computation. These tasks require not only information retrieval but also understanding/reasoning over the entire text. We include a detailed comparison with concurrent works in Tab. 1.
# 3 THE LOOGLE BENCHMARK
There are three categories of data sources as mentioned in Tab. 2. Based on that, we generate two main types of tasks: short dependency and long dependency tasks in LooGLE . For short dependency tasks, we generate short QA from Wikipedia articles and cloze from scripts. For the long dependency tasks, we include summarization for arXiv papers and manually designed QA tasks for long document understanding. There are four major subtasks for QA: Multiple information retrieval, Timeline reorder, Computation, Comprehension and reasoning. We delicately generate tasks/questions to customize the intrinsic features of each data source for better long-context understanding assessments.
# 3.1 DATASET SELECTION AND CONSTRUCTION
Our LooGLE benchmark consists of 3 sources: scientific papers, Wikipedia articles, movie and TV scripts, all covering various topics and categories. These documents are commonly used as corpora in NLP tasks. By replicating the methodology proposed in this paper, they can be collected easily and periodically. All the documents in our LooGLE benchmark are after 2022 and filtered by a length of over 10k words. We have also considered books, but found that most books meeting our principles
Dataset
No. Docs
Avg. Words
Max. Words
Min. Words
Avg. Tokens
Task
# Questions
arXiv
516
16,988
197,977
10,204
20,887
Summarization
516
Wikipedia
105
17,604
46,250
11,285
21,017
Short dependency QA
1,951
Long dependency QA
459
Movie &
TV scripts
155
28,483
62,752
11,089
36,412
Cloze
2,880
Long dependency QA
642
are not license-free, therefore giving them up. Statistics of the three sources can be found in Tab. 2. Details of the dataset are introduced in the following sections.
arXiv papers We pulled data from a massive pool of 10,000 entries on the arXiv website (https://arxiv.org/) using a random selection method. These entries ranged from January 2022 to April 2023. In the next step, we extracted their abstracts, making them our main source for the summarization task. We were pretty rigorous about maintaining data quality. That meant ditching the reference sections, cleaning up any garbled characters from math equations, and leaving out any documents under 10,000 words. After all that thorough check, we ended up with a solid collection of 516 research papers. Wikipedia articles Wikipedia is a free and popular online encyclopedia that provides information and reference on a wide range of topics. Articles are created and edited collaboratively by volunteers from all around the world, making it a dynamic and constantly evolving resource. These Wikipedia articles are perfect for evaluating the long text reading, comprehension, summarization, and information retrieval abilities of LLMs. We first downloaded and parsed the most recent page articles present in .bz file format from the official website (https://dumps.wikimedia.org/). Then we kept the articles after 2022 with over 10k words utilizing a subset of the open-source Wikipedia dataset (202203.en) from Hugging Face (https://huggingface.co/datasets/wikipedia). Since some pages in the dump file probably no longer exist and are redirected to a relevant page, we only retain pages (exempt summary, citations and references) after redirection. Movie and TV scripts A movie or TV script typically contains essential information such as scene descriptions, action descriptions, and dialogues between characters. Scripts inherently encapsulate numerous events and facts in dialogue format, necessitating models to deeply comprehend contextual nuances. To comprehend the events unfolding within a dialogue, there is a high demand on reasoning ability, along with the ability to navigate shifts in perspective and grasp the viewpoints of the characters involved. Additionally, scripts are typically lengthy and challenging for LLMs with fixed context window sizes. All scripts are sourced from three websites (https://www.scriptslug.com, https://thescriptlab.com/, https://8flix.com), consisting of movies and TV shows released after 2022.
arXiv papers We pulled data from a massive pool of 10,000 entries on the arXiv website (https://arxiv.org/) using a random selection method. These entries ranged from January 2022 to April 2023. In the next step, we extracted their abstracts, making them our main source for the summarization task. We were pretty rigorous about maintaining data quality. That meant ditching the reference sections, cleaning up any garbled characters from math equations, and leaving out any documents under 10,000 words. After all that thorough check, we ended up with a solid collection of 516 research papers.
# 3.2 LONG DEPENDENCY TASKS
# 3.2.1 TASKS DEFINITION
Summarization We directly use the abstract of each paper as the reference for generating sum maries. The abstracts effectively capture the main content and key information of each paper.
# Summarization We directly use the abstract of each paper as the reference for generating summaries. The abstracts effectively capture the main content and key information of each paper.
Long dependency QA One highlight of our dataset is that we dedicated significant effort to manually compile about 1.1k true long dependency QA pairs. The construction process is detailed in the next section. We manually designed 4 long dependency tasks: Multiple information retrieval, Timeline reorder, Computation, Comprehension and reasoning. As we will show in the experiments, these tasks are pretty challenging, requiring more advanced capabilities for long context understanding. They are valuable for understanding the limitations of LLMs. Examples of the 4 types of long dependency QAs are shown in Fig. 2. • Multiple information retrieval: Quite different from traditional short-term retrieval tasks, there are usually multiple and diverse pieces of evidence throughout the entire text for one specific answer. The task requires extensive information extraction from widely distributed segments within the lengthy text, followed by the aggregation of the evidence to derive the ultimate answer. The evidence is distinctly presented and can be directly located within the original sentences or sections of the text.
Long dependency QA One highlight of our dataset is that we dedicated significant effort to manually compile about 1.1k true long dependency QA pairs. The construction process is detailed in the next section. We manually designed 4 long dependency tasks: Multiple information retrieval, Timeline reorder, Computation, Comprehension and reasoning. As we will show in the experiments, these tasks are pretty challenging, requiring more advanced capabilities for long context understanding. They are valuable for understanding the limitations of LLMs. Examples of the 4 types of long dependency QAs are shown in Fig. 2.
 Multiple information retrieval: Quite different from traditional short-term retrieval tasks, there are usually multiple and diverse pieces of evidence throughout the entire text for one specific answer. The task requires extensive information extraction from widely distributed segments within the lengthy text, followed by the aggregation of the evidence to derive the ultimate answer. The evidence is distinctly presented and can be directly located within the original sentences or sections of the text.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1770/1770a8fd-f47b-456e-a517-e53ae1227efc.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Long dependency QA tasks</div>
• Computation: Similar to the previous task, it firstly needs multiple information retrieval from a wide range of texts. A majority of the evidence within the text takes the form of numerical data, often in question formats such as inquiries about quantities, frequencies, durations, specific numbers, and so on. To arrive at an accurate response, a profound comprehension of the question and its correlation with the provided numerical data is essential. This process relies heavily on the capacity to grasp extensive contextual information and also involves a degree of mathematical reasoning ability. • Timeline reorder: This task follows a more conventional format, involving the instruction, “Please reorder the timeline of the following events,” along with a set of events presented in a permuted order. The objective is to arrange these events in accordance with their chronological sequence as dispersed throughout the extensive text. The events are derived directly from the source text, either as extracted segments or summarized factual information. Successful completion of this task necessitates either the memorization or comprehensive understanding of the central storyline of the document and assesses the model’s proficiency in temporal awareness. • Comprehension and reasoning: This task demands not only a profound comprehension of the question but also intricate reasoning to discern the underlying implications for searching for the appropriate evidence. The most prevalent question patterns involve inquiries about causality, impact, contributions, attitudes, and the essential attributes related to various events. Additionally, more extensive comparisons and evaluations are essential when the questions revolve around the primary, predominant, highest, or most critical aspects of the evidence. Furthermore, the answers to this task are not explicitly evident within the source text. They often require multi-step reasoning to model the inherent connections and dependencies, facilitating the acquisition of the answer through a complex analytical process.
# 3.2.2 CONSTRUCTION PROCESS OF LONG DEPENDENCY QAS
We detail the construction process as follows. We first randomly sampled a total of 140 long documents from Wikipedia and the scripts dataset. We recruited students from top universities across the nation and organized a manual annotation process to generate long dependency QAs. We categorize long dependency tasks into Multiple information retrieval, Comprehension and reasoning, Calculation, and Timeline reorder (illustrated in Fig. 2). Each document spans from 10,000 to 20,000 words in average and requires a generation of 5 to 10 questions. Additionally, participants were prohibited from employing large language models and tools like ChatGPT for article reading, data generation, and annotation. In the generation of questions, each document underwent a meticulous three-step process that involved the assignment of two distinct annotators — one serving as the questioner and the other as the answerer. Importantly, these annotators were kept unaware of each other’s identities, ensuring a rigorous cross-validation process to maintain the quality of the questions, answers, and supporting evidence. This approach aimed to achieve questions with a high degree of accuracy, precision, and relevance to the document’s content.
Step 1: Question and answer. The questioner’s role encompassed a comprehensive set of responsibilities, including reading the document, crafting relevant questions, offering their own answers to those questions, and pinpointing the specific evidentiary passages within the document that substantiated their answers. The annotation adhered to stringent standards, encompassing the following key principles: • Long dependency: Each question was required to exhibit a long dependency, i.e., the evidence supporting its answer should have a wide span across the document. The recommended dependency length (the distance between the earliest and latest evidence) is a minimum of 5,000 words. • Diverse problem types: The questioner was required to generate a set of 5 to 10 question-answer pairs for each document, which should not contain more than 4 questions of the same type to avoid imbalanced question distribution and prevent annotators from generating overly simple questions. • Clear and precise questions: The formulation of each question was asked to adhere to clarity, conciseness, and no ambiguity, with examples provided. • Deterministic and objective answers: The answers to the proposed questions were rigorously checked to be deterministic and objective, precluding open-ended ones. Step 2: Answer and check. The second step involves the answerers. Each answerer can only access the assigned article text and the posed questions from the questioner in the first step. The answerer was required to thoroughly read the entire document and provide answers to the questions accordingly. The standard for the answers is the same as the questioners. In addition to the aforementioned responsibilities, the answerer was also tasked with assessing the quality of the questions, which entails evaluating whether the questions adhere to the standard and whether they are answerable. In instances where a question cannot elicit a definite and unambiguous answer, it is deemed as unsatisfactory, and the answerer is asked to provide constructive feedback for improvement. Step 3: Revise. In the third step, the questioner for the document had access to the document, the questions, the two sets of answers from both the questioner and the answerer, as well as the feedback from the answerer. The questioner was asked to first revise the questions according to the feedback, and then unify their own answers with those from the answerers to derive the final answers. In the first step, we acquired a total of 1,137 question-answer pairs. In the second step, 206 of these pairs were identified as non-compliant with the established criteria and were accompanied by suggestions for improvement. The inter-annotator agreement rate is 81.88% (Kim & Park, 2023). Following the revisions conducted in the third step, we ultimately obtained a total of 1101 high-quality long dependency question-answer pairs which require strong long context understanding ability.
Step 1: Question and answer. The questioner’s role encompassed a comprehensive set of responsibilities, including reading the document, crafting relevant questions, offering their own answers to those questions, and pinpointing the specific evidentiary passages within the document that substantiated their answers. The annotation adhered to stringent standards, encompassing the following key principles:
Question Answering (QA) To generate short dependency QA pairs, we harnessed the robust language processing and comprehension capabilities of GPT3.5-turbo-16k. These short dependency QA pairs typically do not require extensive evidence retrieval and can be extracted from localized segments. We divided each article into multiple segments and employed an iterative approach to prompt the Language Model (LLM) to generate QA pairs based on these segments, including their associated supporting evidence from the article. Details of the prompts are available in Appendix D. Subsequently, we conducted manual reviews of the QA pairs, making refinements to some of the answers by filtering out non-essential context and eliminating redundant descriptions. This rigorous curation process was undertaken to ensure the high quality and relevance of the resulting QA pairs. Cloze Initially, each script is divided into segments of varying lengths. Then, we employ GPT3.5turbo-16k to generate factual summaries aligning with the source segment along with some constraints included in prompts (see Appendix D). Later, we employ BERT-large (Devlin et al., 2019) for Named Entity Recognition (NER) (Roy, 2021) from the generated summaries, limiting the types to person name, location, and organization. Finally, we randomly select a certain number (no more than 5) of entities from the summary and mask them as placeholders, denoted as “<mask-n>”. The goal is to predict the masked entities according to the long context.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3ce6/3ce61b04-8416-4087-91e3-b9b10ff15e08.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: An overview performance of LLMs on LooGLE for long conte</div>
# 4 EVALUATION
# 4.1 MODELS SELECTED FOR EVALUATION
Commercial models GPT4-32k, GPT4-8k, GPT3.5-turbo-16k (Chen et al., 2023b; Ye et al., 2023) are all the models developed by OpenAI, as documented on their official platform (https://platform.openai.com/docs/models). GPT4-32k can handle up to 32k tokens in the context input, and GPT4-8k and GPT3.5-turbo-16k can handle up to 8k and 16k context input, respectively. We use the models of version 0613 by default. Open-source models LLaMA2-7B-32K (Touvron et al., 2023) is developed by Together (https://together.ai/) and fine-tuned from Meta’s original Llama2-7B (Touvron et al., 2023) model. It has been expanded to accommodate a context length of 32K using Position Interpolation (Chen et al., 2023a). ChatGLM2-6B-32k (Du et al., 2022), is a product of THUMD and represents an enhancement of the ChatGLM2-6B model. It is notable for its integration of FlashAttention (Dao et al., 2022), allowing it to train with an extended context length, increased from 2K to 32K. LongLLaMa-3B, derived from openllama, has been fine-tuned using Focused Transformer (Tworkowski et al., 2023) to extend its context to 256k. Lastly, RWKV-4-14B-pile (Peng et al., 2023) is a member of the RWKV model family, notable for its architectural fusion of both Recurrent Neural Networks (RNN) and Transformers. It has been fine-tuned to accommodate a context length of 8K. Retrieval-based Method Instead of extending the context window size, retrieval-based context compression technique (Xu et al., 2023; Askari et al., 2023) augments the LLM by incorporating external memory, allowing relevant information to be retrieved using a specific query. LlamaIndex (https://github.com/jerryjliu/llama index) is a data framework designed for LLMs. It fulfills a dual role by constructing indices for document segments and functioning as an intermediary connecting LLM with data sources, which enables LlamaIndex to retrieve relevant data segments before they are input into the LLM, thereby enhancing the LLM’s capacity to effectively handle lengthy text. In our experiment, we employed the default configuration of the LlamaIndex, with embedding model textembedding-ada-002 (https://openai.com/blog/new-and-improved-embedding-model) and language model text-davinci-003 (Ouyang et al., 2022). It has been proved that, performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle of long contexts. (Liu et al., 2023a). Therefore, we artificially truncate the input document to certain sizes (all not larger than the context window size of above mentioned models) by concatenating the head and tail of the input. For example, when we want to truncate a long document to 16k, we concatenate its head 8k tokens and tail 8k tokens before feeding it to an LLM.
# 4.2 EVALUATION METHODS AND METRICS
Automatic evaluation We adopt several automatic evaluation metrics, which can be categorized into two types. Bleu, Rouge, Meteor Score and Bert Score (Li et al., 2023b; Mukherjee & Rahman, 2023) are widely used for generative tasks such as summarization and QA. They evaluate the matching
<div style="text-align: center;">Table 3: Performance of the short dependency tasks</div>
Table 3: Performance of the short dependency tasks
Short dependency QA
Cloze
Models
Context
Bleu1
Bleu4
Rouge1
Rouge4
RougeL
Meteor score
Bert score
GPT4 score
Exact Match
Partial Match
GPT4-32k
32k
24.61
11.14
61.80
50.73
60.75
32.94
78.72
71.52
70.5
80.81
GPT4-8k
8K
27.35
14.38
67.59
56.01
65.77
38.56
87.93
53.99
66.03
76.62
GPT3.5-turbo-16k
16K
22.67
9.62
62.56
48.63
60.66
32.58
87.04
66.82
54.64
63.42
LlamaIndex
\
33.37
21.43
58.82
42.93
57.08
37.17
86.58
59.61
58.95
66.86
ChatGLM2-6B
32k
14.29
6.07
20.50
13.16
20.36
13.08
87.28
23.65
0.05
0.98
LongLLaMa-3B
256k
1.37
0.26
26.97
11.02
26.10
11.34
71.65
13.75
-
2.13
RWKV-4-14B-pile
8k
0.80
0.04
21.7
6.39
20.64
9.41
70.42
8.93
-
-
LLaMA2-7B-32K
32k
0.18
7.25*e-308
1.86
0.00
1.86
1.52
61.53
3.18
-
0.58
<div style="text-align: center;">Table 4: Performance of the long dependency tasks</div>
Table 4: Performance of the long dependency tasks
Models
Context
Bleu1
Bleu4
Rouge1
Rouge4
RougeL
Meteor score
Bert score
GPT4 score
arXiv paper summarization
GPT4-32k
32k
24.50
0.73
27.15
7.10
24.25
19.03
84.04
82.84
GPT4-8k
8k
29.02
2.09
32.08
11.11
28.85
22.64
84.92
85.42
GPT3.5-turbo-16k
16k
28.70
1.59
32.04
10.69
28.89
22.34
84.82
86.84
LlamaIndex
\
22.53
0.63
26.28
6.97
23.73
21.07
83.09
76.35
ChatGLM2-6B
32k
0.04
1.60e-310
5.97
8.43e-05
5.82
6.40
73.25
13.23
LongLLaMa-3B
256k
4.24
9.32e-309
4.10
0.52
3.86
3.82
73.41
12.28
RWKV-4-14B-pile
8k
6.28
4.58e-05
6.45
0.74
6.01
6.00
75.28
7.02
LLaMA2-7B-32K
32k
0.03
4.66e-310
0.12
0.00
0.12
0.67
71.21
7.60
Long dependency QA
GPT4-32k
32k
8.55
1.40
25.59
6.36
24.04
11.13
80.16
54.09
GPT4-8k
8k
8.94
1.01
23.45
6.57
21.69
10.18
85.36
42.12
GPT3.5-turbo-16k
16k
6.92
1.81
25.02
6.68
23.63
10.40
83.79
45.04
LlamaIndex
\
7.76
1.24
23.62
7.10
22.30
10.47
83.87
37.63
ChatGLM2-6B
32k
5.55
0.11
9.41
1.93
8.69
4.39
85.78
11.50
LongLLaMa-3B
256k
1.04
3.12e-307
2.96
0.03
2.71
1.66
78.60
6.48
RWKV-4-14B-pile
8k
0.71
9.52e-307
18.54
1.55
17.69
3.45
71.36
5.33
LLaMA2-7B-32K
32k
0.08
2.44e-308
2.05
0.00
2.05
0.46
50.28
4.18
between groundtruth and LLM answers mainly based on n-gram matching and semantic similarity. For Cloze, Exact Match and Partial Match (Sharma et al., 2023; Engelbach et al., 2023) are employed in our evaluation. Exact Match entails the predicted entity and the groundtruth entity exactly match each other while Partial Match allows for fuzzy matching.
between groundtruth and LLM answers mainly based on n-gram matching and semantic similarity. For Cloze, Exact Match and Partial Match (Sharma et al., 2023; Engelbach et al., 2023) are employed in our evaluation. Exact Match entails the predicted entity and the groundtruth entity exactly match each other while Partial Match allows for fuzzy matching. GPT4-as-judgment Most automatic evaluation metrics are sensitive to semantic expression, output format, and length. Thus, these metrics alone might be insufficient for effectively comparing different models (some models might output answers in a style more similar to groundtruth). However, recent research has shown that the GPT4 evaluator exhibits high consistency with human evaluation and can serve as a reliable annotator to some extent (Suri et al., 2023; Liu et al., 2023b; Zheng et al., 2023). To provide a more comprehensive assessment of models, we utilize GPT4-8k as an LLM evaluator. For QA task, given one question and two answers provided by the groundtruth and the LLM’s prediction, we ask GPT4-8k to judge whether the two answers are semantically the same or not. Then we calculate the accuracy that LLM answers match the groundtruth. For summarization task, given the predicted summary with the goundtruth, we ask LLM to give a score considering various factors for generation. The prompts implemented can be found in Appendix D. Human evaluation We also include human evaluation for reference, where we manually check whether LLM’s prediction matches the groundtruth.
# 4.3 RESULTS
Fig. 3 shows an overall performance comparison of different models on different tasks. The first radar plot shows the original accuracy evaluated by GPT4-8k (except cloze) and the partial match result (for cloze) over different tasks. For better visualization, we scale the scores of all models on each task to [40, 100] in the second radar plot and the histogram, so that the best model on each task has a score of 100 and the worst model has a score of 40. From the charts, GPT4-32k demonstrates its impressive overall performance across all tasks (with highest scores on all tasks except summarization). In comparison, open-source models show a significant performance gap to commercial models on our benchmark. From the first radar chart, we can find that among the 7 major tasks, short QA, cloze and summarization are more effectively addressed by LLMs, while real long dependency QA tasks are far from being solved, where even GPT4-32k hardly achieves over 40% accuracy. The empirical results demonstrate that even the most successful commercial model
<div style="text-align: center;">Table 5: Impact of input length on long dependency tasks</div>
Table 5: Impact of input length on long dependency tasks
Models
Context
Bleu1
Bleu4
Rouge1
Rouge4
RougeL
Meteor score
Bert score
GPT4 score
arXiv paper summarization
GPT4-32k
32k
24.50
0.73
27.15
7.10
24.25
19.03
84.04
82.84
GPT4-32k
24k
25.57
0.81
27.61
7.53
24.73
19.86
84.07
83.15
GPT4-32k
16k
24.80
0.70
27.29
7.26
24.28
19.12
84.11
82.82
GPT4-32k
8k
26.26
9.35
27.83
7.67
24.74
20.08
84.10
82.75
GPT4-8k
8k
29.02
2.09
32.08
11.11
28.85
22.64
84.92
85.42
Long dependency QA
GPT4-32k
32k
7.64
1.24
15.53
4.46
14.60
11.12
86.07
54.65
GPT4-32k
24k
8.23
1.66
14.92
4.12
13.90
10.60
86.16
50.61
GPT4-32k
16k
8.57
1.35
16.21
4.30
14.90
11.91
86.36
47.55
GPT4-32k
8k
7.46
1.77
13.75
5.08
12.89
10.01
85.77
38.34
GPT4-8k
8k
8.94
1.01
23.45
6.57
21.69
10.18
85.36
42.12
still cannot effectively address those really challenging long dependency tasks, leaving large room for improvement. Detailed evaluation results and further analysis can be found in the following sections.
4.3.1 MAIN RESULTS ON LONG AND SHORT DEPENDENCY TASKS
Results on short dependency tasks Tab. 3 presents the performance (%) of all the baselines on LooGLE in short dependency tasks. Notably, GPT4-32k attains the highest accuracy according to the GPT4 evaluator’s perspective. GPT4-8k, GPT3.5-turbo-16k, and the retrieval-based LlamaIndex closely follow, demonstrating competitive performance levels. Surprisingly, GPT4-8k exhibits the most robust overall performance in terms of automatic evaluation metrics. It’s worth mentioning that GPT4-32k, due to its tendency to generate longer outputs, faces penalties from these automatic metrics. This discrepancy among different metrics highlights the need for improved evaluation methods. Furthermore, in the context of cloze tasks, GPT4-32k excels again when equipped with a longer context window. In Fig. 4, the exact match results in cloze tasks are displayed for varying source segment lengths. The results show that as the segment length increases, model performance gradually decreases, underscoring the increasing difficulty of effectively filling in the masked entities with longer source text. Results on long dependency tasks Tab. 4 shows the aggregated results on long dependency tasks. Firstly, we can observe that summarization can be well addressed by commercial models, with GPT-4 evaluation accuracy of over 80%. However, the various types of long dependency QAs in our benchmark apparently pose substantial challenges for current LLMs. Both open-source and commercial models experience a significant performance decline. We will analyze model performance on individual types of QAs in Section 4.3.2. It is validated that longer context window size (thus less information loss due to truncation) indeed helps in long context tasks by comparing GPT4-32k with GPT4-8k. GPT4-8k has a much lower accuracy by answering “The text does not provide information on ...” in many cases. Open-sourced models fall far below the average of commercial models, among which LLaMA2-7B-32K and RWKV-4-14B-pile display almost zero performance. By employing context scaling techniques like positional interpolation, RNN and fine-tuning on longer texts, current LLMs can be equipped with much longer context windows than their default limits. Nevertheless, our results show that there is still a huge discrepancy between merely increasing the context window size and really understanding the long context. The poor performance on long dependency QAs suggests that we may need to revisit LLMs’ long context understanding ability in more challenging tasks other than some simple ones like summarization and retrieval, as they are unable to test whether LLMs understand the inter-dependency in long texts.
Results on short dependency tasks Tab. 3 presents the performance (%) of all the baselines on LooGLE in short dependency tasks. Notably, GPT4-32k attains the highest accuracy according to the GPT4 evaluator’s perspective. GPT4-8k, GPT3.5-turbo-16k, and the retrieval-based LlamaIndex closely follow, demonstrating competitive performance levels. Surprisingly, GPT4-8k exhibits the most robust overall performance in terms of automatic evaluation metrics. It’s worth mentioning that GPT4-32k, due to its tendency to generate longer outputs, faces penalties from these automatic metrics. This discrepancy among different metrics highlights the need for improved evaluation methods. Furthermore, in the context of cloze tasks, GPT4-32k excels again when equipped with a longer context window. In Fig. 4, the exact match results in cloze tasks are displayed for varying source segment lengths. The results show that as the segment length increases, model performance gradually decreases, underscoring the increasing difficulty of effectively filling in the masked entities with longer source text.
# Results on long dependency tasks
tasks. Firstly, we can observe that summarization can be well addressed by commercial models, with GPT-4 evaluation accuracy of over 80%. However, the various types of long dependency QAs in our benchmark apparently pose substantial challenges for current LLMs. Both open-source and commercial models experience a significant performance decline. We will analyze model performance on individual types of QAs in Section 4.3.2. It is validated that longer context window size (thus less information loss due to truncation) indeed helps in long context tasks by comparing GPT4-32k with GPT4-8k. GPT4-8k has a much lower accuracy by answering “The text does not provide information on ...” in many cases. Open-sourced models fall far below the average of commercial models, among which LLaMA2-7B-32K and RWKV-4-14B-pile display almost zero performance. By employing context scaling techniques like positional interpolation, RNN and fine-tuning on longer texts, current LLMs can be equipped with much longer context windows than their default limits. Nevertheless, our results show that there is still a huge discrepancy between merely increasing the context window size and really understanding the long context. The poor performance on long dependency QAs suggests that we may need to revisit LLMs’ long context understanding ability in more challenging tasks other than some simple ones like summarization and retrieval, as they are unable to test whether LLMs understand the inter-dependency in long texts.
2 DEEP DIVE INTO LONG CONTEXT UNDERSTANDING CAPABILITIES
In this section, we analyze different factors affecting the long context understanding abilities LLMs, and dive into individual types of long dependency QAs to check LLMs’ limitations.
In this section, we analyze different factors affecting the long context understanding abilities of LLMs, and dive into individual types of long dependency QAs to check LLMs’ limitations. Impact of varying input length In Tab. 5, we study the impact of varying lengths of inputs on long dependency tasks with GPT4 models. We find that expanding input length hardly helps in paper summarization while it substantially enhances the model’s performance on long dependency QAs. The difference can be attributed to the inherent nature of the arXiv paper. It has both the introduction
Impact of varying input length In Tab. 5, we study the impact of varying lengths of inputs on long dependency tasks with GPT4 models. We find that expanding input length hardly helps in paper summarization while it substantially enhances the model’s performance on long dependency QAs. The difference can be attributed to the inherent nature of the arXiv paper. It has both the introduction
<div style="text-align: center;">Table 6: Performance of the long dependency QA with LlamaIndex</div>
Table 6: Performance of the long dependency QA with LlamaIndex
Models
Context
Bleu1
Bleu4
Rouge1
Rouge4
RougeL
Meteor score
Bert score
GPT4 score
GPT4-32k
32k
6.08
1.31
10.27
3.39
9.52
8.54
85.27
28.25
GPT4-8k
8k
6.62
1.50
11.95
3.80
10.99
9.02
85.51
26.34
GPT3.5-turbo-16k
16k
6.50
0.92
10.93
3.56
9.86
8.65
85.63
33.24
Default
\
7.02
1.24
11.60
3.75
10.57
9.37
85.61
33.16
ChatGLM2-6B-32k
32k
0.15
2.82e-310
2.23
0.05
2.23
0.74
83.40
7.73
LongLLaMa-3B
256k
1.04
4.22e-311
2.27
0.00
2.23
2.23
82.18
5.33
RWKV-4-14B-pile
8k
2.65
8.09e-307
4.08
0.33
3.65
3.92
80.74
2.43
LLaMA2-7B-32K
32k
0.43
1.29e-307
5.85
0.00
5.85
1.04
81.38
6.76
Table 7: Individual results on four types of long dependency QAs evaluated by GPT4
Models
Context
Information retrieval
Timeline reorder
Computation
Comprehension and reasoning
GPT4-32k
32k
43.60
64.43
37.36
61.26
GPT4-8k
8K
31.89
61.36
22.54
45.78
GPT3.5-turbo-16k
16K
36.86
55.73
24.73
51.09
LlamaIndex
\
27.60
47.83
19.78
43.83
ChatGLM2-6B-32k
32k
12.47
14.17
5.43
11.08
LongLLaMa-3B
256k
3.82
6.48
5.15
10.17
RWKV-4-14B-pile
8k
4.67
5.19
4.40
7.13
LLaMA2-7B-32K
32k
3.01
1.61
1.12
6.85
and conclusion sections located at the beginning and in the end respectively, which already contain the major sketch of the paper. Meanwhile, in our expectation, longer input promotes the performance of long dependency QAs by introducing less information loss.
Retrieval Based Techniques To evaluate the effectiveness of retrieval techniques for long-context dependency questions, we undertook an extensive series of experiments on our long dependency QA tasks by replacing the base LLM model in LlamaIndex with different baseline LLMs. In these experiments, we utilized the open-source embedding all-mpnet-base-v2 (Song et al., 2020). When compared to the default embedding, text-embedding-ada-002 (https://openai.com/blog/new-andimproved-embedding-model), there was a noticeable performance decline. Nonetheless, this disparity did not hinder our conclusions. From Tab. 4 and Tab. 6, our research findings reveal that the incorporation of retrieval techniques does not generally enhance the performance of long dependency QA tasks. There is a conspicuous performance decline, particularly evident for models like GPT4-8k and GPT4-32k. It can be attributed to the tendency of GPT models to produce longer outputs, sometimes including hallucinatory information, when the retrieved segments lack sufficient context. The phenomenon highlights the intricacy of our dataset, where a series of long dependency understanding and modeling capabilities such as comprehension and multi-hop reasoning are essentially needed. Relying solely on retrieval mechanisms might be insufficient in recalling the necessary information and further generating the final answer, resulting in a marked performance decline. However, we did observe an minor improvement in the BERT score for open-source models. This improvement in fluency can be attributed to the considerably shorter length of the retrieved segments used as inputs, in contrast to the entirety of the document.
# Individual results on different types of long dependency QAs
on presenting aggregated results for long dependency QA tasks across various question types. Differently, in this study, our objective is to delve into the performance of models in individual tasks that demand diverse capabilities, including reading comprehension, information retrieval, computation, and reasoning. In this regard, we employed GPT4 as the evaluator, and the accuracy results are available in Tab. 7. Across the four tasks examined, LLMs generally exhibit strong performance in comprehension, reasoning, and multiple information retrieval, but fall short in tasks related to timeline reordering and computation. Furthermore, we observed that the way questions are framed has a significant impact on LLMs’ performance. Yes-no questions and multiple-choice questions tend to be easier for LLMs to answer, particularly when the search space is limited, as opposed to open-ended questions within unstructured text.
on presenting aggregated results for long dependency QA tasks across various question types. Differently, in this study, our objective is to delve into the performance of models in individual tasks that demand diverse capabilities, including reading comprehension, information retrieval, computation, and reasoning. In this regard, we employed GPT4 as the evaluator, and the accuracy results are available in Tab. 7. Across the four tasks examined, LLMs generally exhibit strong performance in comprehension, reasoning, and multiple information retrieval, but fall short in tasks related to timeline reordering and computation. Furthermore, we observed that the way questions are framed has a significant impact on LLMs’ performance. Yes-no questions and multiple-choice questions tend to be easier for LLMs to answer, particularly when the search space is limited, as opposed to open-ended questions within unstructured text. Results on long dependency QAs with/without CoT To bolster the long-context capabilities of LLMs, we conducted additional experiments designed to unlock their potential using the Chain of Thoughts (CoT) framework (Kojima et al., 2023). We selected LlamaIndex as a representative model, given its impressive performance in both short and long dependency question-answering tasks, alongside strong commercial models such as GPT4. A manual evaluation was carried out on a subset comprising one-third of instances from each task category within long dependency QA. We initiated the LLM with a zero-shot CoT approach, employing prompts such as “Let’s think step
# Results on long dependency QAs with/without CoT
of LLMs, we conducted additional experiments designed to unlock their potential using the Cha of Thoughts (CoT) framework (Kojima et al., 2023). We selected LlamaIndex as a representati model, given its impressive performance in both short and long dependency question-answeri tasks, alongside strong commercial models such as GPT4. A manual evaluation was carried out  a subset comprising one-third of instances from each task category within long dependency Q We initiated the LLM with a zero-shot CoT approach, employing prompts such as “Let’s think st
<div style="text-align: center;">Table 8: Performance of Timeline Reorder</div>
Table 8: Performance of Timeline Reorder
Models
LSD
LMD
SD
SDD
LSD-S
LMD-S
SD-S
SDD-S
Non-standard(%)
GPT4-32k
1.04
0.57
0.93
1.12
1.21
0.82
1.41
1.60
52.80
GPT4-8k
1.24
0.64
1.04
1.281
1.43
0.92
1.51
1.74
49.31
LlamaIndex
1.55
0.78
1.19
1.551
1.95
1.08
1.65
2.09
39.72
GPT3.5-turbo-16k
3.58
1.43
2.17
2.916
1.05
0.86
1.24
1.26
77.21
LongLLaMa-3B
4.18
1.71
2.59
3.30
1.90
1.12
1.80
2.07
92.92
ChatGLM2-6B-32k
4.31
1.74
2.63
3.37
1.83
1.17
1.50
2.00
99.07
RWKV-4-14B-pile
4.33
1.75
2.64
3.38
1.90
0.97
1.00
1.75
98.13
LLaMA2-7B-32K
4.33
1.75
2.64
3.38
2.50
1.17
1.33
2.33
98.60
by step,” and furnished a few-shot setup with detailed rationales and standard output formats Wei et al. (2023) to facilitate responses to long dependency questions. As depicted in Fig. 5, the zero-shot CoT approach had minimal impact on accuracy in comprehension and reasoning, as well as multiple retrieval tasks, but yielded a substantial 20% and 10% absolute accuracy increase in timeline reorder and computation. Interestingly, the few-shot CoT approach benefits the first two types but surprisingly leads to a decline in performance in the latter two types compared with zero-shot. We hypothesize the reason is that the evidence and rationales in few-shot examples cannot be generalized to other questions, and including them might on the contrary give wrong guidance to the model. Automatic evaluation on timeline reorder In order to evaluate the performance of time reorder task outputs, it is essential to address discrepancies arising from the diverse formats produced by various models. Typically, these outputs comprise conventional numerical sequences, but errors in non-standard formats when evaluation necessitate preprocessing for accurate assessment. A proposed approach involves converting the serial numbers in the candidate answers from their original question into Roman numbers (i.e., I, II, · · · ), thereby enhancing discrimination through regular expression matching. Four key metrics, namely, LSD (location square deviation), LMD (location mean deviation), SD (swap deviation), and SDD (swap distance deviation), are employed to measure the similarity of numeric sequences, refer to Appendix C for metric details. Smaller deviations indicate a higher degree of resemblance between the sequences. Any outputs that are empty, possess unequal lengths, or contain extra elements are categorized as non-standard. The maximum deviation between the provided ground truth and all corresponding candidate answers is computed as the worst score for evaluation purposes. The percentage of non-standard outputs for each model and corresponding performances can be found in Tab. 8. As seen, it is evident that except for GPT4, which demonstrates a remarkable degree of adherence and alignment following Reinforcement Learning from Human Feedback (RLHF) (Lee et al., 2023), most open-sourced models struggle to generate texts in the correct format with less than 10%. However, this issue can be mitigated in significantly large models through the utilization of few-shot prompts and mandatory instructions. This phenomenon results in performance penalties when assessed using automated metrics. Consequently, to ascertain the genuine capacity of LLMs in this task, we calculate the four metrics exclusively for outputs in standard format (“-S”).
by step,” and furnished a few-shot setup with detailed rationales and standard output formats Wei et al. (2023) to facilitate responses to long dependency questions. As depicted in Fig. 5, the zero-shot CoT approach had minimal impact on accuracy in comprehension and reasoning, as well as multiple retrieval tasks, but yielded a substantial 20% and 10% absolute accuracy increase in timeline reorder and computation. Interestingly, the few-shot CoT approach benefits the first two types but surprisingly leads to a decline in performance in the latter two types compared with zero-shot. We hypothesize the reason is that the evidence and rationales in few-shot examples cannot be generalized to other questions, and including them might on the contrary give wrong guidance to the model.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8dc6/8dc65ccc-5711-48ef-8e92-fbec584208e2.png" style="width: 50%;"></div>
<div style="text-align: center;">e 4: Performance of varying segments Figure 5: Long dependency QA t</div>
Dispcrepancy in generated outputs of models Distributions of generated outputs of various models are depicted in Fig. 6. It is noteworthy that well-behaved models consistently produce shorter responses, averaging around 50 words, irrespective of the question type, particularly in short-term question answering scenarios. In contrast, models fine-tuned with longer textual inputs, such as LLaMA2-7B-32K, RWKV-4-14B-pile, and LongLLaMa-3B, tend to yield significantly lengthier responses, even when a maximum generation constraint of 500 tokens is enforced. An interesting deviation is observed in LongLLaMa-3B, which demonstrates variability in response lengths across both tasks. This behavior may stem from challenges in comprehending and addressing exceedingly complex long question-answering tasks. Consequently, the model appears to prioritize extracting a maximum number of pertinent contexts from its memory to generate sufficiently extensive responses that are deemed acceptable and rational.
Moreover, upon closer examination of model outputs, a significant disparity in generation quality is observed across various LLMs and task types, indicating a non-specific issue. Notably, commercial models like GPT4, GPT3.5, and LlamaIndex consistently generate outputs that exhibit a higher degree of human-likeness, completeness, and logical coherence within a structured format. These models consistently deliver contextually relevant, query-based responses. In contrast, open-sourced models, such as ChatGLM2-6B-32k, tend to offer
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4819/48193707-d6ff-4ffe-94ee-a118a2c8be06.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Output distributions on QA tasks</div>
shorter answers, occasionally confined to numeric responses. In cases where a definite answer is lacking, ChatGLM2-6B-32k compensates by retrieving relevant contextual information. However, the RNN-based model RWKV-4-14B-pile often generates duplicated responses or resorts to repeating the given questions to reach the maximum token length, sometimes resorting to code generation to address issues related to its training data. The performance of the LLaMA2-7B-32K model is notably worse, as it sporadically produces irrelevant or nonsensical text, along with the inclusion of special symbols when it fails to provide meaningful answers. More examples of outputs from different models can be seen in Appendix F.
Probable explanations for long QA bad cases To investigate whether the models have effectively memorized and comprehended lengthy contextual information, we conducted a comprehensive manual analysis of the underlying causes of failures in each long question-answering task. The rationale behind CoT analysis aided in understanding how models decompose and tackle challenges associated with extended dependency-based QA. Our observations reveal that LLMs struggle with these tasks primarily due to their inability to extract precise information and a propensity to generate responses that lack factual accuracy. Constraints imposed by the inherent context window limitations, coupled with information loss resulting from the optimized Transformer and position encoding, contribute to their struggles in memorizing the original extensive contexts. In most cases, models attempt to compensate by retrieving and integrating the most pertinent evidence, even if it results in redundant answers. However, they also acknowledge their insufficient context and, at times, abstain from providing responses rather than resorting to nonsensical answers. Furthermore, addressing these
<div style="text-align: center;">Table 9: Bad cases study on the long dependency QA</div>
Table 9: Bad cases study on the long dependency QA
Long QA Tasks
Hallucination∗
Redundant
retrieval†
Insufficient
retrieval⋆
Irrelevant
answer⋄
No relevant
context∧
Wrong/No
reasoning×
Others
Computation
31.11
24.44
15.56
0.00
20.00
0.00
8.88
Multiple information retrieval
14.71
31.37
28.43
13.73
13.73
0.00
7.84
Comprehension and reasoning
14.29
10.99
21.98
18.68
16.48
10.99
6.59
∗Evidence of predictions is not shown up in the original inputs and generated by LLM itself from nowhere.
∗Evidence of predictions is not shown up in the original inputs and generated by LLM itself from nowhere. † Apart from the right evidence, irrelevant evidence is also redundantly retrieved. ⋆Not all of the essential evidence to answer the question is retrieved. ⋄Evidence of predictions generated have no or minor correlation with the question. ∧No relevant context in LLM’s memory and refuse to answer the question. × Fail to retrieve evidence that needs further reasoning other than directly extracting from the inputs.
challenges necessitates enhanced comprehension and reasoning abilities, particularly when answers are not clearly evident across multiple pieces of evidence scattered throughout the raw texts. The insights derived from our benchmark analysis offer a scientific foundation and pave the way for promising research directions aimed at augmenting LLM capabilities for handling long contextual inputs. These findings underscore the need for further progress in comprehension, computation, and reasoning tasks using our dataset to effectively enhance LLMs’ capacity to understand extended dependency contexts.
# 5 CONCLUSION
This paper introduces a novel benchmark, LooGLE , designed to facilitate the assessment of longcontext comprehension by LLMs. LooGLE addresses the deficiencies present in previous datasets by offering considerably longer text passages, utilizing relatively new documents after 2022, incorporating multi-source materials from various categories, and notably featuring meticulously designed and annotated tasks with diverse contextual dependencies. Our extensive evaluations unveil substantial limitations in the capacity of existing LLMs to understand and reason about the intricate interdependencies present in lengthy texts, even when provided with considerably extended context windows. Furthermore, a notable disparity is observed between commercial and open-source models, with both exhibiting challenges in long dependency tasks as per our benchmark assessments. The outcomes underscore the utility of our dataset as a valuable reference for evaluating long-context comprehension and present avenues for potential enhancements in LLM performance.
REFERENCES
REFERENCES Chenxin An, Shansan Gong, Ming Zhong, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. L-eval: Instituting standardized evaluation for long context language models, 2023. 2, 3 Stefanos Angelidis, Reinald Kim Amplayo, Yoshihiko Suhara, Xiaolan Wang, and Mirella Lapata. Extractive opinion summarization in quantized transformer spaces, 2020. 2 Arian Askari, Suzan Verberne, Amin Abolghasemi, Wessel Kraaij, and Gabriella Pasi. Retrieval for extremely long queries and documents with rprs: a highly efficient and effective transformer-based re-ranker, 2023. 7 Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench: A bilingual, multitask benchmark for long context understanding, 2023. 2, 3 Arkadii Bessonov, Alexey Staroverov, Huzhenyu Zhang, Alexey K. Kovalev, Dmitry Yudin, and Aleksandr I. Panov. Recurrent memory decision transformer, 2023. 3 Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 1 Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. Recurrent memory transformer, 2022. 3 Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. Scaling transformer to 1m tokens and beyond with rmt, 2023. 2 Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation, 2023a. 7 Xuanting Chen, Junjie Ye, Can Zu, Nuo Xu, Rui Zheng, Minlong Peng, Jie Zhou, Tao Gui, Qi Zhang, and Xuanjing Huang. How robust is gpt-3.5 to predecessors? a comprehensive study on language understanding tasks, 2023b. 7 Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning of long-context large language models, 2023c. 3 Ta-Chung Chi, Ting-Han Fan, Alexander I. Rudnicky, and Peter J. Ramadge. Dissecting transformer length extrapolation via the lens of receptive field analysis, 2023. 2 Arman Cohan, Franck Dernoncourt, Doo Soon Kim, Trung Bui, Seokhwan Kim, Walter Chang, and Nazli Goharian. A discourse-aware attention model for abstractive summarization of long documents, 2018. 2 Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness, 2022. 2, 7 Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019. 6 Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Nanning Zheng, and Furu Wei. Longnet: Scaling transformers to 1,000,000,000 tokens, 2023. 2 Zican Dong, Tianyi Tang, Lunyi Li, and Wayne Xin Zhao. A survey on long text modeling with transformers, 2023. 3 Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 320–335, 2022. 7 Matthias Engelbach, Dennis Klau, Felix Scheerer, Jens Drawehn, and Maximilien Kintz. Fine-tuning and aligning question answering models for complex information extraction tasks, 2023. 8
Alexios Gidiotis and Grigorios Tsoumakas. A divide-and-conquer approach to the summarization of long documents, 2020. 3 Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions for long document summarization, 2021. 2 Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented language models, 2022. 3 NamHyeok Kim and Chanjun Park. Inter-annotator agreement in the wild: Uncovering its emerging roles and considerations in real-world scenarios, 2023. 6 Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners, 2023. 10 Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. Rlaif: Scaling reinforcement learning from human feedback with ai feedback, 2023. 11 Shanda Li, Chong You, Guru Guruganesh, Joshua Ainslie, Santiago Ontanon, Manzil Zaheer, Sumit Sanghai, Yiming Yang, Sanjiv Kumar, and Srinadh Bhojanapalli. Functional interpolation for relative positions improves long context transformers, 2023a. 3 Yucheng Li. Unlocking context constraints of llms: Enhancing context efficiency of llms with self-information-based content filtering, 2023. 3 Zihao Li, Samuel Belkadi, Nicolo Micheletti, Lifeng Han, Matthew Shardlow, and Goran Nenadic. Large language models and control mechanisms improve text readability of biomedical abstracts, 2023b. 7 Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts, 2023a. 7 Yuxuan Liu, Tianchi Yang, Shaohan Huang, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, and Qi Zhang. Calibrating llm-based evaluator, 2023b. 8 Clara Meister, Stefan Lazov, Isabelle Augenstein, and Ryan Cotterell. Is sparse attention more interpretable?, 2021. 3 Usmi Mukherjee and Mohammad Masudur Rahman. Employing deep learning and structured information retrieval to answer clarification questions on bug reports, 2023. 7 OpenAI. Gpt-4 technical report, 2023. 1 Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. 7 Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Xiangru Tang, Bolun Wang, Johan S. Wind, Stansilaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Jian Zhu, and Rui-Jie Zhu. Rwkv: Reinventing rnns for the transformer era, 2023. 7 Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models, 2023. 3 Arya Roy. Recent trends in named entity recognition (ner), 2021. 6 Tom´aˇs Koˇ cisk´y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, G´abor Melis, and Edward Grefenstette. The NarrativeQA reading comprehension challenge. Transactions of the
Alexios Gidiotis and Grigorios Tsoumakas. A divide-and-conquer approach to the summarization of long documents, 2020. 3 Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions for long document summarization, 2021. 2 Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented language models, 2022. 3 NamHyeok Kim and Chanjun Park. Inter-annotator agreement in the wild: Uncovering its emerging roles and considerations in real-world scenarios, 2023. 6 Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners, 2023. 10 Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. Rlaif: Scaling reinforcement learning from human feedback with ai feedback, 2023. 11 Shanda Li, Chong You, Guru Guruganesh, Joshua Ainslie, Santiago Ontanon, Manzil Zaheer, Sumit Sanghai, Yiming Yang, Sanjiv Kumar, and Srinadh Bhojanapalli. Functional interpolation for relative positions improves long context transformers, 2023a. 3 Yucheng Li. Unlocking context constraints of llms: Enhancing context efficiency of llms with self-information-based content filtering, 2023. 3 Zihao Li, Samuel Belkadi, Nicolo Micheletti, Lifeng Han, Matthew Shardlow, and Goran Nenadic. Large language models and control mechanisms improve text readability of biomedical abstracts, 2023b. 7 Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts, 2023a. 7 Yuxuan Liu, Tianchi Yang, Shaohan Huang, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, and Qi Zhang. Calibrating llm-based evaluator, 2023b. 8 Clara Meister, Stefan Lazov, Isabelle Augenstein, and Ryan Cotterell. Is sparse attention more interpretable?, 2021. 3 Usmi Mukherjee and Mohammad Masudur Rahman. Employing deep learning and structured information retrieval to answer clarification questions on bug reports, 2023. 7 OpenAI. Gpt-4 technical report, 2023. 1
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. 7 Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Xiangru Tang, Bolun Wang, Johan S. Wind, Stansilaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Jian Zhu, and Rui-Jie Zhu. Rwkv: Reinventing rnns for the transformer era, 2023. 7
Tom´aˇs Koˇ cisk´y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, G´abor Melis, and Edward Grefenstette. The NarrativeQA reading comprehension challenge. Transactions of the Association for Computational Linguistics, TBD:TBD, 2018. URL https://TBD. 2
Uri Shaham, Elad Segal, Maor Ivgi, Avia Efrat, Ori Yoran, Adi Haviv, Ankit Gupta, Wenhan Xiong, Mor Geva, Jonathan Berant, and Omer Levy. Scrolls: Standardized comparison over long language sequences, 2022. 2, 3 Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. Zeroscrolls: A zero-shot benchmark for long text understanding, 2023. 2, 3 Eva Sharma, Chen Li, and Lu Wang. Bigpatent: A large-scale dataset for abstractive and coherent summarization, 2019. 2 Roshan Sharma, Suyoun Kim, Daniel Lazar, Trang Le, Akshat Shrivastava, Kwanghoon Ahn, Piyush Kansal, Leda Sari, Ozlem Kalinli, and Michael Seltzer. Augmenting text for spoken language understanding with large language models, 2023. 8 Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. Mpnet: Masked and permuted pre-training for language understanding, 2020. 10 Gaurav Suri, Lily R. Slater, Ali Ziaee, and Morgan Nguyen. Do large language models show decision heuristics similar to humans? a case study using gpt-3.5, 2023. 8 Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for efficient transformers, 2020. 3 Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Efficient transformers: A survey, 2022. 3 Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. 7 Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition, 2022. 2 Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Miło´s. Focused transformer: Contrastive training for context scaling, 2023. 7 Alex Wang, Richard Yuanzhe Pang, Angelica Chen, Jason Phang, and Samuel R. Bowman. Squality: Building a long-document summarization dataset the hard way, 2022. 2 Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. 11 Jeff Wu, Long Ouyang, Daniel M. Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, and Paul Christiano. Recursively summarizing books with human feedback, 2021. 3 Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, Madian Khabsa, Han Fang, Yashar Mehdad, Sharan Narang, Kshitiz Malik, Angela Fan, Shruti Bhosale, Sergey Edunov, Mike Lewis, Sinong Wang, and Hao Ma. Effective long-context scaling of foundation models, 2023. 3 Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. Retrieval meets long context large language models, 2023. 7 Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering, 2018. 2 Junjie Ye, Xuanting Chen, Nuo Xu, Can Zu, Zekai Shao, Shichun Liu, Yuhan Cui, Zeyang Zhou, Chao Gong, Yang Shen, Jie Zhou, Siming Chen, Tao Gui, Qi Zhang, and Xuanjing Huang. A comprehensive capability analysis of gpt-3 and gpt-3.5 series models, 2023. 7
Yan Zeng, Hanbo Zhang, Jiani Zheng, Jiangnan Xia, Guoqiang Wei, Yang Wei, Yuchen Zhang, and Tao Kong. What matters in training a gpt4-style language model with multimodal inputs?, 2023. 1 Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023. 8 Yucheng Zhou, Tao Shen, Xiubo Geng, Chongyang Tao, Guodong Long, Can Xu, and Daxin Jiang. Fine-grained distillation for long document retrieval, 2022. 3
Distributions of the input length and dependency spanning in words for long dependency QA tasks are shown in Figs. 7 and 8. N-gram sunburst graph for generated QA pairs can be seen in Fig. 9.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bc71/bc71d240-7e59-47b8-89e2-49435051a99a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: 4-gram sunburst graph for short and long dependency QA. (a) short dependency questions (b) short dependency answers (c) long dependency questions (d) long dependency answers</div>
# B TASK DEFINITION
B TASK DEFINITION
The Cloze task formulation process can be seen in Fig. 10.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7e94/7e94e34e-a87f-4202-bcbe-09a92d487f05.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10: Cloze task</div>
C TIMELINE REORDER EVALUATION METRICS
# C TIMELINE REORDER EVALUATION METRICS
We employ 4 metrics to measure the similarity of numeric output sequences for timeline reorder tasks. For given two numeric sequences A and B with the same sequence length n, i[A] and i[B] is
the ith number in each sequence. They can be computed using the formula below: LSD is the abbreviation for location square deviation:
LSD(A, B) = 1 n n−1 � i=0 (i[A] −i[B])2
LMD is the abbreviation for location mean deviation:
LMD(A, B) = 1 n n−1 � i=0 |i[A] −i[B]|
SD is the abbreviation for swap deviation:
SD(A, B) = min(W(A →B) = min(� s∈A→B1)
SDD is the swap distance deviation:
SDD(A, B) = min(W(A →B) = min(� s∈A→B |i −j|)
�   where s = A(i, j) means the swap between the ith and jth element in A. S = A →B means a series of swap actions to convert A to B. W(S) = � s∈Sw(s) means the weights sum of all the swap actions in S, where w(s) = 1 in SD and w(s) = |i −j| in SSD.
# D PROMPTS
D.1 SHORT DEPENDENCY QA PAIR GENERATION
[seg] = {Input long texts} Please generate 2 questions and corresponding answers based on given [seg] in less words as possible. Return reference text S, question Q and answer A from [seg] in json format as: {“S”: ,“Q”: ,“A”: },{“S”: ,“Q”: ,“A”: }.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/76d1/76d102fa-545f-4c7f-8a8b-51289152a968.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6800/6800a8c0-53a5-4993-bca3-5ac92936dce6.png" style="width: 50%;"></div>
Instruction: Please answer the question based on the given long texts below. {Input long texts} Question: {Question} Answer:
Instruction: Please write a summary for this script segment within 500 words, focusing on describing objective facts and avoiding subjective opinions. {scripts segement} Summary:
(2)
(4)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/847b/847b5317-6897-4076-a915-53e5fe559ee3.png" style="width: 50%;"></div>
<div style="text-align: center;">D.5 SUMMARIZATION</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fe00/fe00717a-099c-44fd-a34d-57857e10bf95.png" style="width: 50%;"></div>
<div style="text-align: center;">D.6 TIMELINE REORDER</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d50f/d50f7d84-a5da-4125-b665-55fe942b6e99.png" style="width: 50%;"></div>
<div style="text-align: center;">D.8 SUMMARIZATION TASK EVALUATION BY LLM (GPT4)</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b929/b929de83-2b20-4cce-952c-743c760e4080.png" style="width: 50%;"></div>
<div style="text-align: center;">D.9 FEW-SHOT COT FOR LONG QA</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f373/f373c316-4724-40a5-9d37-db04416e9a6f.png" style="width: 50%;"></div>
Instruction: Please answer the question based on the given long texts below.
{Input long texts}
{Demonstrations}
Question: {Question}
Answer:
# D.10 ZERO-SHOT COT FOR LONG QA
Instruction: Please answer the question based on the given long texts below. {Input long texts} Question: {Question} Answer: Let’s think step by step.
EXAMPLES FOR LONG CONTEXT UNDERSTANDING TASK
E.1 SHORT DEPENDENCY QUESTION AND ANSWERING
Question: Who did Picardo