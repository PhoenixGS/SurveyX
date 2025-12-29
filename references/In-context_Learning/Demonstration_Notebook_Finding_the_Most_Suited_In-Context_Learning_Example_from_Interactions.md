# Demonstration Notebook: Finding the Most Suited In-Context Learning Example from Interactions
Yiming Tang Bin Dong Peking University Peking University yimingtangible@163.com dongbin@math.pku.edu.cn
# Abstract
Large language models (LLMs) benefit greatly from prompt engineering, with in-context learning standing as a pivital technique. While former approaches have provided various ways to construct the demonstrations used for in-context learning, they often ignore the inherent heterogeneity within datasets, applying the same demonstrations to all reasoning questions. We observed that the effectiveness of demonstrations varies depending on the specific question. This motivates our exploration of using prompt engineering to select appropriate demonstrations. To address the challenge of automatically creating and choosing demonstrations tailored to each question, we propose a novel prompt engineering workflow built around a novel object called the "demonstration notebook." This notebook helps identify the most suitable in-context learning example for a question by gathering and reusing information from the LLM’s past interactions. Our experiments show that this approach outperforms all existing methods for automatic demonstration construction and selection (as far as we know), achieving state-of-the-art results on serveral reasoning benchmarks. The method’s versatility is further demonstrated by its success in text summarization and prompt compression tasks. Additionally, we contribute a rigorous analysis method to reveal the "demonstrative regime" of a demonstration, providing valuable insights into how demonstrations relate to different question types within a dataset.
arXiv:2406.10878v1
# 1 Introduction
The rapidly developing field of large language models (LLMs) has witnessed a surge in the importance of prompt engineering(PE) as different prompt designs demonstrably exert significant influences on LLM outputs. Among the various prompt engineering techniques[1; 2; 3; 4], in-context learning stands as a foundational approach to unlocking LLMs’ capabilities for specific tasks requiring the construction and selection of demonstrations. Introduced in 2020[5], in-context learning feeds few-shot examples to language models, enabling them to learn from contextual examples for enhanced performances. To enhance in-context learning capabilities, Chain-of-Thought(ManualCoT)[6] prompting advocates to handcraft step-by-step demonstrations within the in-context examples to elicit LLM’s stepwise reasoning capabilities for improved performance. Automatic demonstration construction methods have also been proposed. AutoCoT[7] for example, employes k-means clustering to select k demonstrative questions and subsequently generates a k-shot demonstration automatically with an LLM. Similarly, PromptSO[8] adopts principle component analysis for the selection of demonstrative questions before automatic demonstration generation. While effective in certain scenarios, these approaches rely on fixed demonstrations, neglecting the inherent heterogeneity of questions within a dataset. This limitation hinders further improvement, as is observed different questions benefit from heterogeneous demonstrations. Ideally, the most suitable demonstration is question-specific instead of constant. Regarding question-specific demonstration selection, retrieval based methods[9] have garnered significant attention in prompt engineering. These approaches rely on external information retrievers to enhance LLM’s output by feeding more information to the model. However, the need of external
retrivers necessitates extra efforts to construct a retriever including both the construction of a database and the training of the embeddings which can often be time consuming. Besides, according to our knowledge, although retrieval based methods have shown effectiveness in various common sense reasoning tasks, little has been explored when it comes to arithmetical and symbolic reasoning tasks since the presence of extra facts might not directly help arithmetic and symbolic reasoning tasks. Further more, both retrieval based PE and PE with fixed demonstrations do not directly provide ways to improve model’s performances based on problems the LLM fails to solve. We hypothesize that for arithmetical and symbolic reasoning tasks, demonstrations incorporating proper methodologies, rather than specific factual instances, are more effective. These methodologies can be collected from LLM’s former interaction histories and reused as demonstrations with a comprehensive design on the interaction strategy of the LLM. To address the challenge of constructing and selecting the most suitable demonstration for a given question, we propose a novel prompt engineering workflow centered around a new object, the demonstration notebook. This notebook encompasses both demonstration generation and selection through three key components: a demonstration set, an interaction record set, and a noted question set. All three sets are initially empty and are automatically populated during a dedicated collection phase preceding the inference stage. The collection phase of the demonstration notebook consists of several epochs of four procedures each epoch, a demonstration expansion procedure, an on-policy collection procedure, an off-policy collection procedure and a pruning procedure. The demonstration expansion procedure first constructs several one-shot demonstration basis. Several random samples of the basis are concatenated as fewshot demonstrations for the demonstration set. The on-policy and off-policy collection procedures collect more interaction records specifying the questions each demonstration is effective to. After the collection phase, a prompter is trained to select a demonstration from the demonstration set for inference questions. Selected demonstrations will be concatenated with the questions as in-context examples before fed into LLMs in the testing phase. Our approach is evaluated with several most renowned large language models on a set of nine reasoning benchmarks including: arithmetic reasoning[10; 11; 12; 13; 14], commonsense reasoning[15; 16] and symbolic reasoning[6]. Our experimental results show that demonstrations selected by the demonstration notebook consistently outperforms all existing approaches supporting the automatic construction of demonstrations, in align with our hypothesis that heterogeneous demonstration selection can be of vital significance in prompt engineering. Beyond quantitive experimental results, we also characterize the fact that different demonstrations are effective to heterogeneous sets of questions with a novel concept, demonstrative regime. We provide rigorous experimental results and visualizations to promote more intuitive use of in-context learning examples. Our visualizations show that the demonstrative regimes of demonstrations are often in the form of low dimensional manifolds in the embedding space, which might not be even close to the embeddings of the demonstration itself. This experimental finding holds the potential to revolutionize retrieval and generate(RAG) based prompt engineering methods. For as it indicates, distance in the embedding space might not be a good heuristic for context selection and learning based retrievers should be more suitable compared with more commonly used cosine-similarity based retrievers for the selection of contexts in prompt engineering.
• We introduce a novel prompt engineering method: the demonstration notebook, which tackles the tasks of automatic demonstration construction and selection simultaneously.
• Our work offers the first rigorous analysis and visualization of demonstration regimes for different demonstrations, promoting more intuitive uses in-context examples.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a9fb/a9fbb000-ac16-4d5c-896e-176343b780db.png" style="width: 50%;"></div>

Figure 1: A. The demonstration notebook consists of three components, the interactive record set, the demonstration set and the noted question set. The records in the interactive record set are in the form of (question demonstration) pairs indicating the demonstration is demonstrative to the question. B. The interactive records in the set are used to train a prompter for demonstration selection. We add an adapter module to a pretrained embedding model for this prompter. C. The collection phase of the demonstration notebook consists of several epochs each of four procedures. D. During the inference phase, a question is fed to the prompter for a selected demostration which is subsequently concatenated with the question when fed to LLMs as in-context examples.
# 2 Related Works
The concept of prompt engineering has emerged since 2020[17; 18; 5; 19], as researchers discovered that large language models can learn from demonstrations as contexts. Chain of thought prompting(CoT)[20] suggests hand-crafting demonstrations with nuanced reasoning steps instead of only giving the final answer as demonstrations. Zero-shot CoT[21] has shown that nuanced reasoning steps can be generated if provided with proper instructions like "Let’s think step by step". In the development, various ensemble prompt engineering methods have been introduced by researchers including CoT-selfconsistency[1] and Complexity CoT[22], sampling a diverse set of reasoning paths and voting out a final answer. Unsupervised learning has been employed for the selection of demonstrations. For instance, AutoCoT[7] suggests that we can use k-means clustering to devide the dataset into several clusters and select out representative questions from each clusters to craft demonstrations. PromptSO[8] suggests that we can use principal component analysis in the embedding space and select out the principal components as the questions for demonstration crafting. These methods, though effective, assign the same set of demonstrations to all specific tasks in the dataset and does not take the uniqueness of different questions into consideration when prompting, which might result in lack of performances. Their set of demonstrations are also relatively larger than necessary and thus causes unnecessary token-inefficiency in inference time. Reinforcement approaches have also been introduced to train prompters that respond differently based on specificity of questions[23; 24]. However they often
require a comparatively large training cost to be implemented and can often result in soft prompting instead of grammarly correct prompts. As shown by several former works, prompt engineering can significantly benefit from extra information collected from former interactions[25; 26; 27]. Their approaches, though effective, requires a long inference time because of requiring multiple interactions with the LLM, limiting their applications to real-world applications like creating AI tutors and AI characters. Beyond specific paradigms to select out demonstrations, some general principles about how demonstrations should be formed have been introduced often as certain inductive biases[28; 29; 30], providing heuristics for the process of crafting demonstrations. Instead of heuristic preferences, theoretical analysis of prompt engineering have also been been introduced by researchers. For example, optimal control frameworks have been introduced to systematically analyse prompt engineering[31; 32] and there have also been researchers interpreting in-context learning as implicit gradient decent[33].
# 3 Method
This section details our proposed prompt engineering method, the demonstration notebook. We begin with a comprehensive overview in Section 3.1, introducing the demonstration notebook, its components and the four procedures used to collect the notebook, demonstration expansion, on-policy collection, off-policy collection and pruning. Section 3.2 introduces the details for the demonstration expansion procedure. The on-policy and off-policy algorithm for collecting the interaction record set and the noted question set are introduced respectively in Sections 3.3 and 3.4. Next, Section 3.5 details the pruning procedure used for the demonstration notebook, completing the discussion of the four procedures in the collection phase. Finally, Section 3.6 concludes with a description of the prompter and the testing policy. An illustration of the overall workflow is provided in Figure 1.
# 3.1 Overview
The demonstration notebook is a novel object designed to improve prompt engineering performance through demonstration construction and question-specific selection. The notebook comprises three key components: a demonstration set, an interaction record set, and a noted question set. Additionally, a prompter serves as the demonstration selection policy. All of these three sets contained in the demonstration notebook are collected from the LLM’s former interactions with a comprehensive strategy consisted of four procedures iterated for several epochs, including demonstration expansion, on-policy collection, off-policy collection and pruning. The demonstration set contains multiple demonstrations constructed automatically using the LLM’s outputs. Each demonstration is composed of a question and its chain-of-thought answer. This set is initialized and expanded during the demonstration expansion procedure, which occurs at the beginning of each collection epoch. The interaction record set serves as the supervised training data for the prompter. It accumulates successful problem-solving records from three collection procedures (demonstration expansion, on-policy collection, and off-policy collection). Whenever a demonstration appears to be demonstrative to a question, the record (question, demonstration) will be added to the interaction record set. The noted question set acts as a repository for "hard" questions encountered during the collection phase. In both the demonstration expansion and the on-policy collection procedures, multiple demonstrations are tested for a question. If the LLM fails to solve this question using any of the proposed demonstrations, the question is added to the noted question set. Subsequently, questions in the noted question set will be used in off-policy collection as questions to "think twice" and in demonstration expansion as questions for demonstration construction.
# 3.2 Demonstration Expansion
The demonstration expansion procedure focuses on constructing new demonstrations to enrich the demonstration set. We begin by sampling several questions that the LLM can correctly solve. These questions serve as a foundation for building demonstrations, each representing a "one-shot" demonstration for the dataset. Here, we prioritize questions from the noted question set and those residing near the center of question clusters because our empirical observations suggest that these questions are often representative of distinct question clusters and demonstrative to the question
clusters they represent. For the noted questions, we prompt the LLM with the correct answer as hints to increase the chances of constructing correct demonstrations for the noted questions. Next, we employ random sampling and concatenation to combine several one-shot demonstration bases into a few "few-shot" demonstrations. These newly constructed demonstrations are then incorporated into the demonstration set of the demonstration notebook. Following demonstration creation, we establish their demonstrative regimes. This involves sampling another question set from the dataset’s training data and evaluating whether the constructed demonstrations can elicit successful problem-solving of the LLM for these additional questions.
Algorithm 1 Demonstration Expansion
Input: Trainingset: Strain; A clustering algorithm: cluster_center().
Output: Sdemo, Srecord, Snoted.
1: Sbasis = Snoted
�cluster_centers(Strain).
2: for item in Sbasis:
▷Creating the basis.
3:
ques = item["question"]; LLManswer =LLM(ques).
4:
if IsCorrect(LLManswer): Demo_basis.append("Q :"+ques+"A :"+LLManswer).
5: for i in range NUM_EXPANSION:
▷Creating demonstrations.
6:
ds = random.sample(Demo_basis,NUM_PER_DEMO).
7:
demo = concatenate(ds).
8:
Sdemo.append(demo).
9: for item in Sbatch=sample(Strain):
▷Collecting records.
10:
ques = item["question"].
11:
Demos = NewDemos.
12:
LLManswers =LLM(Demos,ques).
13:
if AllCorrect(LLManswers): pass.
14:
if PartlyCorrect(LLManswers): Srecord.append(demo) if correct.
15:
if AllWrong(LLManswers): Snoted.append(ques).
# 3.3 On-policy Collection
The on-policy collection procedure targets for the collection of interaction record set of the demonstration notebook in an on-policy way for sample efficiency. We begin by training the prompter based on the current interaction record set which serves as the policy of demonstration selection for both this collection procedure and the testing phase. In each iteration, a question is sampled from the training set. The prompter then assigns scores to each demonstration in the demonstration set, indicating its predicted suitability for the given question. The top-k demonstrations with the highest scores are then concatenated with the question and presented to the LLM for inference. There are three potential outcomes for the inference. First, if all selected demonstrations lead to correct answers, no new records are added to the interaction record set. This scenario suggests that none of the demonstrations are particularly informative for the specific question. Including such interactions could potentially contaminate the prompter’s training process, as they don’t provide guidance on which demonstration is most effective. On the other hand, if none of the selected demonstrations enable the LLM to solve the problem, the question is added to the noted question set. These "hard" questions are then prioritized during the off-policy collection procedure, where a wider range of demonstrations can be explored. The most informative outcome occurs when some, but not all, demonstrations lead to correct answers. This indicates that several demonstrations are demonstrative to the question. Both the question and the successful demonstration(s) are incorporated into the interaction record set. This reinforces the association between demonstrations and their demonstrative regimes, ultimately improving the prompter’s selection capability.
▷Collecting records.
The off-policy collection procedure focuses on enriching the demonstration notebook with informative interactions for the noted questions. Unlike the on-policy approach, it does not leverage the prompter for demonstration selection during this stage. We begin by sampling a batch of questions from the training set, prioritizing questions from the noted question set. Recall that the noted question set contains questions that the LLM struggled with in the on-policy collection procedure. These "hard" questions are prime candidates for exploration with a wider range of demonstrations. We then systematically evaluate each demonstration in the demonstration set for each question within the batch. If a demonstration successfully guides the LLM to a correct answer, the questiondemonstration pair is added to the interaction record set. This reinforces the association between the demonstrations and their demonstrative regimes and further improves the prompter’s selection capability.
Algorithm 2 On-Policy Collection
Input: Trainingset: Strain, prompter.
Output: Srecord, Snoted.
1: Sbatch =sample(Strain).
2: for item in Sbatch:
3:
ques = item["question"].
4:
Demos =prompter(ques).
5:
LLManswers =LLM(Demos,ques).
6:
if AllCorrect(LLManswers):
7:
pass.
8:
if PartlyCorrect(LLManswers):
9:
Srecord.append(demo) if correct.
10:
if AllWrong(LLManswers):
11:
Snoted.append(ques).
Algorithm 3 Off-Policy Collection
Input: Trainingset: Strain, Snoted, Sdemo.
Output: Srecord, Snoted.
1: Sbatch =sample(Snoted,Strain).
2: for item in Sbatch:
3:
ques = item["question"].
4:
Demos = Sdemo.
5:
LLManswers =LLM(Demos,ques).
6:
if AnyCorrect(LLManswers):
7:
Snoted.pop(ques).
8:
if PartlyCorrect(LLManswers):
9:
Srecord.append(demo) if correct.
10:
if AllWrong(LLManswers):
11:
Snoted.append(ques).
Algorithm 2 On-Policy Collection
Input: Trainingset: Strain, prompter.
Output: Srecord, Snoted.
1: Sbatch =sample(Strain).
2: for item in Sbatch:
3:
ques = item["question"].
4:
Demos =prompter(ques).
5:
LLManswers =LLM(Demos,ques).
6:
if AllCorrect(LLManswers):
7:
pass.
8:
if PartlyCorrect(LLManswers):
9:
Srecord.append(demo) if correct.
10:
if AllWrong(LLManswers):
11:
Snoted.append(ques).
Each epoch of the collection phase concludes with a pruning procedure to eliminate redundant records and demonstrations within the demonstration notebook. Interaction Record Set: For the interaction record set, we identify and remove any duplicate records. These duplicates may arise due to coinciding samples during previous collection procedures. This ensures the interaction record set contains unique question-demonstration pairs for the prompter. Demonstration Set: For each demonstration in the set, we define a corresponding "coverage set" consisting of all questions that the LLM can solve correctly when aided by that demonstration. During pruning, if the coverage set of one demonstration is entirely contained within the coverage set of another demonstration, the redundant demonstration is removed from the demonstration set. Additionally, all associated records from the interaction record set are cleared.
Algorithm 4 Pruning
Input: Trainingset: Srecord, Snoted, Sdemo.
Output: Srecord, Snoted, Sdemo.
1: for record in Srecord:
▷Pruning redundant records.
2:
if IsDuplicated(record): Srecord.pop(record).
3: for demo in Sdemo:
▷Calculating demonstrative regimes.
4:
Sregime["demo"] = rec["question"] for rec in Srecord with rec["demonstration"]==demo.
5: for demo1, demo2 in Sdemo:
6:
if Sregime["demo1"] included in Sregime["demo2"]:
▷Pruning redundant demonstrations.
7:
Sdemo.pop(demo1).
8:
Srec.pop(rec for rec in Srecord with rec["demonstration"]==demo1).
Following the collection phase, we train the prompter of the demonstration notebook using the interaction record set. This empowers the prompter to automatically select the most suitable demonstration for a given question during inference.
Following the collection phase, we train the prompter of the demonstration notebook using the interaction record set. This empowers the prompter to automatically select the most suitable demonstration for a given question during inference. To balance sample efficiency and computational training costs, we propose a prompter architecture that leverages a pre-trained text encoder with several shallow adapter layers. The interaction record set, while valuable, is relatively small for comprehensive training of a large prompter, particularly one that aims to capture the full complexity of prompts. Therefore, utilizing a pre-trained encoder alongside a smaller adapter appears to be operable and efficient. During the testing phase, when encountering a new question, the prompter infers and selects the most suited demonstration from the demonstration set. This chosen demonstration is then concatenated with the question to form the final prompt that is fed to the LLM for inference.
To balance sample efficiency and computational training costs, we propose a prompter architecture that leverages a pre-trained text encoder with several shallow adapter layers. The interaction record set, while valuable, is relatively small for comprehensive training of a large prompter, particularly one that aims to capture the full complexity of prompts. Therefore, utilizing a pre-trained encoder alongside a smaller adapter appears to be operable and efficient.
During the testing phase, when encountering a new question, the prompter infers and selects the most suited demonstration from the demonstration set. This chosen demonstration is then concatenated with the question to form the final prompt that is fed to the LLM for inference.
# 4 Experiments
We evaluate the effectiveness of demonstration notebook on several settings including three types of reasoning tasks, prompt compression and article sumarization. As former works pointed out, language models often struggle at reasoning tasks including arithmetic reasoning, commonsense reasoning, and symbolic reasoning, making reasoning tasks a common playground to testify LLM’s performances. Recent research has suggested that since the use of large language models can be costly in computation and resources, prompt compression, aiming at compressing the length of the prompts used in inference while maintaining good performances, can be of vital importance. We also extend our experimental results to article summarization aiming at concise summarization of texts in natural language.
# 4.1 In-context Learning for Reasoning
The demonstration notebook is evaluated on various reasoning tasks against relevant baselines that can also achieve automatic demonstration construction. Benchmarks To evaluate the effectiveness of demonstration notebook in reasoning tasks, we use the following benchmarks consisted of three main categories: 1. Arithmetic reasoning: (1) SingleEq[10], (2) MultiArith[11], (3) GSM8k[12], (4) SVAMP[13], (5) AQuA[14]. These datasets each consists of arithmetic questions of secondary school math difficulty. 2. Commonsense reasoning: (1) CSQA[15], (2) STQA[16]. The popular CSQA[15] contains commonsense questions about the world involving complex semantics that often require prior knowledge in order to answer correcly. StrategyQA[16] requires models to infer a multi-step reasoning path to answer questions by stating out the possibility of some statements. 3. Symbolic reasoning: (1) Last Letter Concatencation and (2) Coin Flip[6]. The last letter concatenation task asks the LLM to concatenate the last letters of each words in a word list which often requires step by step reasoning and can benefit significantly from in-context examples. The coin flip
task inquires the LLM to make predictions about whether a coin is facing head up or head down after multiple operations. Baselines Our experimental results are compared with four baselines using in-context learning to enhance LLM performances. First, Zero-shot CoT[21] prompts the LLM with the question and a specially designed instruction, "Let’s think step by step." Manual CoT[6] uses chain-of-thought demonstrations crafted with extra human expertise. AutoCoT[7] uses k-means clustering to select out several questions and uses the LLM to automatically construct the demonstrations while PromptSO[8] uses principle component analysis for the selection of questions. For ManuelCoT[6], AutoCoT[7] and PromptSO[8], we use exactly the same demonstrations provided in their works respectively. Implement Details We evaluate our method on Meta Llama3(8B) model and OpenAI gpt-3.5turbo(175B) model. As former works pointed out, they each stand as representative large language models of open-source and close-source communities. We use greedy decoding in inference for deterministic text generation and reproductivity. The prompter is tuned based on OpenAI’s textembedding-ada-002 model with an MLP adapter at the head of the embedding model.
# 4.1.1 Main Results
<div style="text-align: center;">Table 1: Main Experimental Results with Llama3-8B</div>
Method
SingleEq
MultiArith
GSM8k
SVAMP
AQuA
CSQA
STQA
Letter
Avg.
ZeroCoT
92.7%
98.8%
72.3%
79.8%
50.0%
53.4%
56.7%
47.3%
65.71%
ManualCoT
93.3%
98.3%
73.3%
81.8%
46.9%
66.4%
67.2%
75.3%
75.31%
AutoCoT
96.8%
98.8%
73.4%
82.4%
47.8%
72.7%
71.6%
76.7%
76.30%
PromptSO
94.2%
99.4%
77.6%
83.0%
52.0%
57.6%
69.4%
81.3%
75.44%
Ours
97.2%
99.4%
74.8%
84.2%
47.6%
68.7%
70.7%
94.7%
79.66%
<div style="text-align: center;">Table 2: Main Experimental Results with GPT3.5</div>
Method
SingleEq
MultiArith
GSM8k
SVAMP
AQuA
CSQA
STQA
Letter
Avg.
ZeroCoT
90.4%
96.0%
75.1%
76.5%
38.2%
72.0%
57.6%
71.0%
72.10%
ManualCoT
90.9%
97.0%
75.8%
80.2%
45.3%
72.2%
61.7%
78.8%
75.24%
AutoCoT
90.2%
96.0%
74.1%
78.2%
46.5%
72.7%
62.4%
72.6%
74.09%
PromptSO
92.1%
98.8%
77.9%
81.0%
40.6%
74.1%
62.5%
79.6%
75.83%
Ours
98.8%
99.4%
70.8%
80.4%
60.2%
70.6%
71.2%
93.3%
80.59%
# 4.2 In-context Learning for Article Summarization
The demonstration notebook can also be applied to article summarization tasks aiming at concise summarization of texts in natural language. We use Samsung Abstractive Messenger Summarization[34] as the testbed and compare demonstrations seleted by the collected demonstration notebook with randomly chosen demonstrations and zero-shot prompting. We use the ROUGE metric for the evaluation of summarization quality. In this setting, since the ROUGE score is calculated to be ranging from 0 to 1. We slightly adjust the collection criteria of the demonstration notebook. If a summarization is above a predefined number, we regard it as corret in the process. Our experimental results show that with minor modifications of the algorithm, the demonstration notebook can be applied to situations with continuous evaluations.
<div style="text-align: center;">Table 3: Article Summarization ROUGEs with Llama3-8B</div>
Method
SAMsun
Zero-shot
32.8
Random Demonstration
33.4
Demonstration Notebook
35.8
# 4.3 Further Analysis on the Demonstrative Regime
In order to characterize the phenomenon that different demonstrations are effective to different kinds of questions, we propose to use the concept of demonstrative regime. By definition, if a question q which can not be directly solved by an LLM can be solved by the same LLM with the presence of a demonstration d, we say that d is demonstrative to q for this LLM. All the questions that the demonstration d is demonstrative to form a set noted as demonstrative regime of demonstration d. Notice that if the question q is simple enough, a demonstration may not be needed for an LLM to answer correctly. This is the main reason why we eliminate these questions that can be easily solved by an LLM when considering the demonstrative regimes. With this perspective, the core idea of demonstration notebook can be viewed as to construct a complete demonstration set so that the union of their demonstrative regimes can cover the whole distribution of the dataset, and the interaction record set is aimed to specify the demonstrative regimes from interactions by collecting points from each demonstrative regime. We also provide visualizations of the demonstrative regimes for more intuitive understanding of the concept of demonstrative regimes2. By transforming the demonstrations into their embeddings and visualizing the first two principal components, we see several characteristics of a demonstrative regime: 1. The demonstrative regime of a demonstration forms a relatively small area in the embedding space and universal demonstrative regimes are uncommon. 2. The demonstrative regime of a demonstration might not be close to the demonstration in the embedding space. 3. The demonstrative regime of a demonstration is likely to be a low dimensional manifold containing this demonstration itself.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2a0e/2a0e00c3-3990-4093-b84c-06b14de873e0.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Visualization of questions that can be solved by the LLM directly, and using three different demonstrations respectively. The green dots means the question that can be correctly solved and ther other dots refer to clusters of questions that can be solved with the demonstrations. The last two diagrams refer to the demonstrative regiems of three demonstrations for two datasets, SVAMP and MutlitArith.</div>
In this work, we propose a novel method, demonstration notebook to tackle automatic demonstration generation and selection using information collected from former interactions. Our experimental results have achieved SOTA result in demonstration construction beating all other approaches that support automatic demonstration generation. The demonstration notebook algorithm can be extended to other tasks including prompt compression and article sumarization and our experimental results have shown the effectiveness of this appraoch in these settings proving the versatility of the method. We also pioneer in exploring the concept of demonstrative regimes characterizing the questions that can be solved with the presence of a demonstration. Our visualization results provide valuable insights toward the intuitive understanding of using demonstrations in in-context learning.
# References
[1] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Huai hsin Chi, and Denny Zhou. Selfconsistency improves chain of thought reasoning in language models. ArXiv, abs/2203.11171, 2022. [2] Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and Ed Chi. Least-to-most prompting enables complex reasoning in large language models, 2023. [3] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks, 2022. [4] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail?, 2023. [5] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. [6] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. [7] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models, 2022. [8] Fobo Shi, Peijun Qing, D. Yang, Nan Wang, Youbo Lei, H. Lu, and Xiaodong Lin. Prompt space optimizing few-shot reasoning success with large language models. 2023. [9] Xiaonan Li and Xipeng Qiu. Mot: Pre-thinking and recalling enable chatgpt to self-improve with memory-of-thoughts, 2023. 10] Lei Wang, Yan Wang, Deng Cai, Dongxiang Zhang, and Xiaojiang Liu. Translating a math word problem to an expression tree, 2018. 11] Subhro Roy, Tim Vieira, and Dan Roth. Reasoning about quantities in natural language. Transactions of the Association for Computational Linguistics, 3:1–13, 2015. 12] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. 13] Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems?, 2021. 14] Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation : Learning to solve and explain algebraic word problems, 2017. 15] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge, 2019. 16] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies, 2021. 17] Wissam Antoun, Fady Baly, and Hazem Hajj. Aragpt2: Pre-trained transformer for arabic language generation. arXiv preprint arXiv:2012.15520, 2020. 18] Golam Md Muktadir. A brief history of prompt: Leveraging language models, 2023. 19] Robert L. Logan IV au2, Ivana Balaževi´c, Eric Wallace, Fabio Petroni, Sameer Singh, and Sebastian Riedel. Cutting down on prompts and parameters: Simple few-shot learning with language models, 2021.
[20] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Huai hsin Chi, F. Xia, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. ArXiv, abs/2201.11903, 2022. [21] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners, 2023. [22] Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, and Tushar Khot. Complexity-based prompting for multi-step reasoning, 2023. [23] Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, Yihan Wang, Han Guo, Tianmin Shu, Meng Song, Eric P. Xing, and Zhiting Hu. Rlprompt: Optimizing discrete text prompts with reinforcement learning, 2022. [24] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning, 2023. [25] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models, 2023. [26] Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. Graph of thoughts: Solving elaborate problems with large language models, 2023. [27] Chuanyang Zheng, Zhengying Liu, Enze Xie, Zhenguo Li, and Yu Li. Progressive-hint prompting improves reasoning in large language models. arXiv preprint arXiv:2304.09797, 2023. [28] Shengnan An, Zeqi Lin, Qiang Fu, Bei Chen, Nanning Zheng, Jian-Guang Lou, and Dongmei Zhang. How do in-context examples affect compositional generalization?, 2023. [29] Haiming Wang, Huajian Xin, Chuanyang Zheng, Lin Li, Zhengying Liu, Qingxing Cao, Yinya Huang, Jing Xiong, Han Shi, Enze Xie, Jian Yin, Zhenguo Li, Heng Liao, and Xiaodan Liang. Lego-prover: Neural theorem proving with growing libraries, 2023. [30] Anirudh Goyal and Yoshua Bengio. Inductive biases for deep learning of higher-level cognition, 2022. [31] Yifan Luo, Yiming Tang, Chengfeng Shen, Zhennan Zhou, and Bin Dong. Prompt engineering through the lens of optimal control, 2023. [32] Aman Bhargava, Cameron Witkowski, Manav Shah, and Matt Thomson. What’s the magic word? a control theory of llm prompting, 2023. [33] Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers, 2023. [34] Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. SAMSum corpus: A human-annotated dialogue dataset for abstractive summarization. In Lu Wang, Jackie Chi Kit Cheung, Giuseppe Carenini, and Fei Liu, editors, Proceedings of the 2nd Workshop on New Frontiers in Summarization, pages 70–79, Hong Kong, China, November 2019. Association for Computational Linguistics.
