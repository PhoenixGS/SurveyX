# CL: In-Context Learning without Human Sup
Shuming Ma Microsoft Research shumma@microsoft.com Furu Wei Microsoft Research fuwei@microsoft.com
Jinghan Yang ∗ The University of Hong Kong eciel@connect.hku.hk Shuming Ma Microsoft Research shumma@microsoft.com
Jinghan Yang ∗ The University of Hong Kong eciel@connect.hku.hk Shuming Ma Microsoft Research shumma@microsoft.com Furu Wei Microsoft Research fuwei@microsoft.com
# Abstract
With in-context learning ability, the performance of large language models can be significantly boosted when provided with appropriate context. However, existing in-context learning methods mainly rely on human-provided contexts, such as labeled examples and explicit instructions. Writing context by humans is labor-intensive on various tasks and limits the model to tasks manageable by humans. To overcome these limitations, we propose Automatic In-Context Learning framework that enables the model to autonomously generate examples and instructions for problem-solving. With experiments across various models and datasets, results show that model-generated contexts outperform human-annotated contexts, including Few-Shot and Few-Shot-CoT methods, and surpass existing self-generated context methods like Zero-CoT and Auto-CoT. 1
# 1 Introduction
In the era of Large Language Models (LLMs), human-computer interaction has evolved towards natural language, offering unprecedented flexibility through in-context learning (Radford et al., 2019; Brown et al., 2020; Wei et al., 2022a). Given an appropriate context, the performance of LLMs can be significantly enhanced. Vanilla in-context learning relies on human-provided contexts, such as labeled examples (Few-Shot (Brown et al., 2020) and Few-Shot-CoT (Wei et al., 2022b) methods), explicit instructions, or other guiding mechanisms that shape the model’s outputs. However, humanwritten contexts present two major limitations: 1. Labor-Intensive Annotation: Creating labels or instructions requires substantial human effort, especially when models perform a variety of tasks. 2. Task Limitations: The model ability is constrained to tasks within human ability. If the task is
∗Work done during internship at Microsoft. 1Preprint. The code is available at https://github.com/ ecielyang/Auto-ICL
challenging for humans, the model’s performance cannot be effectively enhanced with human written context. . To address these limitations, we propose the Automatic In-Context Learning (Auto-ICL) framework. Upon receiving a user’s request, the model can independently generate examples and instructions for problem-solving. The Automatic InContext Learning framework is shown in Figure 1, which involves two steps. Firstly, a question that requires solving is presented to the model. In Step 1, the model is then prompted to generate demonstrations and/or instructions to assist in solving the question. In Step 2, the model incorporates these self-generated demonstrations and/or instructions and the original query as inputs to generate the final solution. In the experiment, the generated context results in better performance than human-annotated contexts, including Few-Shot and Few-Shot-CoT methods. It also surpasses vanilla methods that trigger models to generate context by itself, e.g., Zero-CoT and Auto-CoT. We also explored the effectiveness of different forms of context based on the availability of similar queries. Our findings reveal that retrieved queries from the dataset can enhance context generation with richer information. With retrieved queries, instruction is more beneficial than demonstration provided in context. Conversely, without retrieved queries, demonstration-plus-instruction context can be more helpful than instruction-only and demonstration-only. In summary, our contributions are as follows:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3d6d/3d6d039f-3d5b-43f5-b2fe-f5b2676a26d4.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9d56/9d56e8ce-9e4b-48f8-a7b2-bdc9c8f2add3.png" style="width: 50%;"></div>
<div style="text-align: center;">Question + Demonstrations and/or Instructions</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8e11/8e11f04d-436a-4704-afd9-87b1fad37b62.png" style="width: 50%;"></div>
Figure 1: The problem-solving process is broken down into two steps. In Step 1, the model is prompted to generate contextual information (in the demonstrations or instruction form) that aids in answering the given query. In Step 2, the model is provided with the query and its self-generated context to produce the result. The generated demonstrations and instructions can be stored in a database and can be further refined by LLM.
<div style="text-align: center;">Figure 1: The problem-solving process is broken down into two steps. In Step 1, the model is prompted to generate contextual information (in the demonstrations or instruction form) that aids in answering the given query. In Step 2, the model is provided with the query and its self-generated context to produce the result. The generated demonstrations and instructions can be stored in a database and can be further refined by LLM.</div>
tasks, demonstrating superior performance to human-crafted contexts and existing methods.
3. We assess the effectiveness of different forms of context under the condition of whether retrieved queries are provided.
# 2 Related Work
This section presents an overview of the background of ICL. Additionally, we highlight previous works that have explored the utilization of LLMs to generate context to assist in question-solving. In-Context Learning allows LLM to address problems by providing demonstrations or instructions. Without the need for gradient updates, ICL equips models to handle a diverse range of issues (Radford et al., 2019; Brown et al., 2020; Dong et al., 2022). Several research initiatives aim to enhance ICL’s effectiveness by carefully crafting prompts. In the case of delivering demonstrations to LLMs, the selection (Liu et al., 2022) and order (Lu et al., 2022) of examples play a critical role (Zhao et al., 2021). Furthermore, significant thought goes into instructing models to resolve a problem. This involves careful design and composition of instructions (Webson and Pavlick, 2022). Chain of thought (CoT) prompting introduces a way to prompt the model to solve questions using its generative capabilities by asking model reasoning with intermediate steps . Few-Shot-CoT firstly guides the model to solve a problem step-by-step by offering reasoning pathways in each example question in a demonstration (Wei et al., 2022b).
Zero-Shot-CoT uses the phrase "let’s think step by step" to help the model arrive at a solution by multistep reasoning (Kojima et al., 2022). Based on Zero-Shot-CoT, Auto-CoT prompts the model by "let’s think step by step" on each example question. They concatenate the example questions and generate reasoning paths to demonstrations to assist the model in problem-solving (Zhang et al., 2022).
Self-generation of context for problem-solving has been progressively developed through multiple studies. Kim et al. (2022) firstly uses LLMs to generate demonstration queries in classification tasks. However, their work is limited by providing the model with the label and generating the corresponding demonstration. Further, Honovich et al. (2022); Zhou et al. (2022) leverage demonstration pairs to generate task instructions. Recently, Li et al. (2023) introduced a step-wise context generation approach, self-prompting, primarily applied in Open-Domain Question Answering. They focused solely on the QA dataset, proposing a method limited to generating demonstrations. Shao et al. (2023) use human-written demonstrations to prompt the model to generate additional demonstrations, followed by a selection process. Contrary to these prior studies, our method feeds the model with only a query, enabling it to generate demonstrations and instructions. Our research further proposes a method to convert a generated demonstration into an instruction format. Thus, our method is more general, requiring only queries, and can shown to be successful in Theory of Mind and reasoning tasks. In knowledge-intensive tasks necessitating domain or
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5802/5802e389-f8f3-4600-8795-2d3dac0c467c.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e1fd/e1fdd3df-870f-4e33-b6f9-fa98951705c9.png" style="width: 50%;"></div>
Figure 2: Left: An example to generate demonstration by the model. Right: Given queries and reasoning paths, the model generates a general instruction. To generate instructions, demonstrations can from retrieved set or generated set.
world knowledge, Yu et al. (2022) and Sun et al. (2022) are geared towards using self-generated context to aid the model in extracting the correct answer from a document. In contrast, our method steers the model towards the right reasoning path by providing demonstrations and instructions with various generation paths rather than merely supplying the document containing the answer.
# 3 Method
Under in-context learning, the context C guides the model M to produce a suitable answer a for the user’s query q. We denote the context as C = {i, (q1, a1), . . . , (qn, an)}, where each (qj, aj) for j = 1, . . . , n signifies a series of demonstrations, and i represents an instruction. In the vanilla ICL framework, the context C is typically provided by a human user:
(1)
In contrast, the Auto-ICL approach initially prompts the model to generate the context C autonomously. The model then uses this selfgenerated context to construct an appropriate response. Initially, given a certain prompt p, the model produces context ˆC, which is presented as:
(2)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d8bb/d8bb6069-f22d-466e-8fd1-72feb5d3a857.png" style="width: 50%;"></div>
where r is available context resource. We categorize Auto-ICL on whether we can retrieve2 queries from the dataset: (1) Auto-ICL (retrieving): A dataset containing q is available in this case. We use a small subset represented as Q = {q1, . . . , qn}, to generate context (i.e. r = Q). Once a context is generated, it is applied to the whole dataset. (2) Auto-ICL (generating): only the current query q is available, and context is generated using this query (i.e. r = q). Following this, the model is tasked with producing an answer ˆa for the query q based on the previously generated context ˆC:
(3)
In the following subsections, we show practical ways to generate demonstration and instruction separately under retrieving and generating cases.
# 3.1 Demonstration
With in-context demonstrations, we supply the model with several input-output pairs that resemble the current task. Recognizing patterns or templates within these demonstrations enhances the model’s proficiency in addressing the present request (Brown et al., 2020). In traditional ICL meth-
2Retrieving only employs queries without labels from the dataset, which is different from utilizing labeled examples in the Few-Shot method.
Question 1: If 4 boys join 3 boys to play game, how many boys  are there? Answer 1: There are 4 boys + 3 boys = 7 boys. The answer is 7  boys. … Question n: If 2 teams join 3 teams to play game, how many  teams are there? Answer n: There is 1 girl and 5 boys. 1 girl + 5 boys = 6 players.  The answer is 6. Question: If 3 girls join 2 girls to play game, how many girls are  there? Answer: Few-Shot CoT
# Instruction: Step 1: Count the number of boys/girls mentioned in the question.  Step 2: Add the numbers together.  Step 3: The result is the total number of boys/girls/teams. Auto-ICL (retrieving)
Question:  If 3 girls join 2 girls to play game, how many girls are there? Solve the question step by step based on the instruction.
Figure 3: An illustration of how to use generated context for problem-solving in Auto-ICL (retrieving) and Auto-ICL (generating). We place the Few-Shot-CoT method in the upper-left corner for comparison. In Auto-ICL (retrieving), the instruction is applied to all queries in the same dataset. In Auto-ICL (generating), the examples and instructions are tailored from each query. The generation process of demonstrations and instructions are shown in Figure 2.
ods, demonstrations are typically created by humans and supplied to the model. Under Auto-ICL, we ask the model to generate demonstration with answers automatically. Generating demonstrations based on query q. The most straightforward case involves using the question q to generate a demonstration composed of a set of queries ˆQ and their corresponding label ˆA, denoted as q →( ˆQ, ˆA). For better performance, we directly generate demonstration with reasoning path toward the answer (chain-of-thought), represented as q →( ˆQ, ˆ CoT). A simple example is shown in Figure 2. Retrieving queries Q from the dataset containing query q. On the other hand, when we have a dataset containing query q, we can randomly retrieve a subset Q from this dataset. The model then generates answers with reasoning path, denoted as Q →(Q, ˆ CoT), which is exactly the Auto-CoT method (Zhang et al., 2022). Hence, Auto-CoT can be viewed as a specific case under our framework where demonstrations with reasoning processes are generated from the retrieved dataset.
ods, demonstrations are typically created by humans and supplied to the model. Under Auto-ICL, we ask the model to generate demonstration with answers automatically.
# 3.2 Instruction generation
In the ICL setting, most existing methods mainly rely on human-crafted instructions (Reynolds and McDonell, 2021) or use question-answer pairs to
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0fb6/0fb6a987-5828-436e-a47d-01498dd8b45b.png" style="width: 50%;"></div>
devise instruction (Honovich et al., 2022; Zhou et al., 2022). We propose a novel approach that enables the model to generate instructions autonomously, starting only with queries. Generating a instruction based on query q. When we only have access to the query q. We can ask the model to generate an instruction ˆi on q. We denote this generation path as q →ˆi, referring to this approach as 1-to-1 instruction generation.
devise instruction (Honovich et al., 2022; Zhou et al., 2022). We propose a novel approach that enables the model to generate instructions autonomously, starting only with queries.
Generating a instruction based on query q. When we only have access to the query q. We can ask the model to generate an instruction ˆi on q. We denote this generation path as q →ˆi, referring to this approach as 1-to-1 instruction generation.
Generating instruction based on queries. Start from demonstration generation, for each query in the randomly retrieved subset Q or generated subset ˆQ, we generate a set of reasoning path, denoted as ˆ CoT. Then we synthesize these individual paths into a single, general instruction, (Q, ˆ CoT) →ˆi or ( ˆQ, ˆ CoT) →ˆi. The instruction assures each reasoning step in the answer is consistent and applicable to all queries within the dataset. Since multiple instructions are consolidated into one general instruction, this method is referred to as N-to-1 instruction generation. The method is visualized in Figure 2, and an actual instruction generated by GPT-3.5-turbo-0301 is shown in AppendixA.2.2. As instructions can be generated upon demonstrations, we also consider incorporating both the demonstration and instruction into a context in the experiment.
Methods
Labeled Data
Theory of Mind
Symbolic Reasoning
Arithmetic
Others
Avg.
Generating
Zero-Shot (Brown et al., 2020)
✗
48.3
16.1
44.4
44.6
38.3
Zero-Shot-CoT (Kojima et al., 2022)
✗
48.0
80.6
63.9
63.1
63.9
Demonstration (ours)
✗
45.3
76.3
75.3
57.8
63.0
Instruction (ours)
✗
59.5
75.3
72.5
58.3
66.3
Demonstration + Instruction (ours)
✗
51.1
92.4
71.5
64.0
68.1
Retrieving
Few-Shot (Brown et al., 2020)
✓
63.5
34.9
45.7
49.3
48.3
APE (Zhou et al., 2022)
✓
40.2
70.8
67.2
63.3
60.4
Induction (Honovich et al., 2022)
✓
42.8
83.3
73.3
66.8
66.6
Demonstration (Auto-CoT)
✗
47.8
92.3
75.3
57.8
66.9
Instruction (ours)
✗
61.5
93.3
75.5
68.3
73.4
Demonstration + Instruction (ours)
✗
47.5
91.5
74.2
66.5
68.1
Table 1: The table compares our method with baseline methods across various tasks and shows the average accuracy under each category. The methods are classified into "retrieving" and "generating" based on whether the data is sourced from the dataset. The usage of labeled demonstrations is indicated in the column named "labeled data." We use 5-shot examples in the demonstration.
# 3.3 Auto-ICL (retrieving) and Auto-ICL (generating)
Since the context can be a combination of demonstrations, instructions, or each on its own, we summarize two forms under the condition of whether we have a dataset to retrieve queries based on the results from experiments: Auto-ICL (retrieving): With retrieval queries, we suggest using instructions only. Here, demonstrations are retrieved from the dataset, answers are generated with Zero-CoT, and a single instruction is created. This instruction is then applied to all queries within the same dataset. Auto-ICL (renerating): Without retrieval queries, we use both demonstrations and instructions. The process starts by generating demonstrations for the query q, followed by generating an instruction using these demonstrations. After creating the context, we combine it with the query q and input them into the model. An example is provided in Figure 3.
# 3.4 Comparison of computational efficiency
In Auto-ICL (retrieving), the generated context is general and applicable to queries within the same dataset. Consequently, the decoding cost is comparable to that of the Zero-CoT and Few-Shot-CoT methods. Additionally, the generated instruction is much shorter than the demonstrations used in FewShot-CoT cases as the number of shots increases, thereby reducing encoding costs. In Auto-ICL (generating), we assume access only to the current query without similar query ex-
amples, reflecting real-world use cases where users may have only a single query. Here, the context is specifically generated for each query. Our method incurs additional decoding costs compared to the Zero-CoT method but results in higher accuracy. Compared to APE for instruction generation (Zhou et al., 2022), which requires generating several instructions and iterating them to maximize a score function, our method is computationally efficient. APE also needs access to the log probabilities of models, which our method does not require. In conclusion, Auto-ICL is a simple and costeffective method that achieves higher accuracy in the retrieving case. Although there is an accuracycost trade-off in the generating case compared to Zero-CoT, Auto-ICL does not replies on other resources compared with other methods.
# 4 Experiment Setting
Dataset: To evaluate our method, we conduct experiments across four categories of tasks: Theory of Mind (Sap et al., 2019; Le et al., 2019; Sap et al., 2022), symbolic reasoning (Roy and Roth, 2016; Ling et al., 2017; Cobbe et al., 2021), arithmetic (Wei et al., 2022b; Kojima et al., 2022), and other reasoning tasks (Srivastava et al., 2022; Brown et al., 2020). Details of each tasks are listed below: Theory of Mind is the ability to understand and interpret others’ intentions and responses during social interactions. To evaluate this, we use the dev set from the SOCIALIQA QA benchmark (Sap
Method
Generated Instruction
Instruction Induction
“not provided.”
APE
"Find the sum of three consecutive integers whose product can be expressed as 727+728+729. The 
answer is E) 39."
Auto-ICL (retrieving)
"The first step is to read the question and identify what information is given and what is being asked 
for. Next, we need to come up with a plan of attack. For example, if we are looking for the sum of 
three consecutive integers, we can set up an equation using the information given in the question. 
Once we have a plan, we can carry out the steps of the plan and solve the problem. Finally, we need 
to check our work to make sure that our answer makes sense."
Auto-ICL (generating)
"First, we need to figure out what information we are given and what information we need to find.
Next, we need to set up an equation or system of equations that we can solve.
Finally, we need to solve the equation or system of equations to find the answer."
et al., 2019) and the TOMI QA dataset with English Sally-Ann-style narratives (Le et al., 2019). Both these datasets scrutinize social-emotional intelligence in common situations and understandings of others’ mental states. We only consider first-order (e.g., “where will Alan look for the object?”) and second-order beliefs (e.g., “where does Alan think that Bob will look for the object?”), referred to as TOMI1 and TOMI2. We use the processed SOCIALIQA and TOMI datasets from Sap et al. (2022), adding "unknown" as an extra answer choice to ensure the model’s choices are knowledge-based rather than by chance. Arithmetic reasoning capability pertains to proficiency in resolving issues by leveraging mathematical concepts and operations like addition, subtraction, multiplication, and division. In our study, we incorporate three datasets: MultiArith (Roy and Roth, 2016), AQUA-RAT (Ling et al., 2017), and GSM8K (Cobbe et al., 2021). These datasets demand multi-step reasoning for problem-solving. For symbolic reasoning, we utilize the Last Letter Concatenation and Coin Flip tasks (Wei et al., 2022b). The Last Letter Concatenation task requires the model to combine the last letters of each word. The Coin Flip task requires the model to determine if a coin remains heads up after a series of flips or non-flips. We use sample data from Kojima et al. (2022) For other tasks, we utilize the Tracking Shuffled Objects dataset from the BIG-bench project (Srivastava et al., 2022), and the Cycle Letter dataset (Brown et al., 2020).
Model: We evaluate our method on the GPT family (Brown et al., 2020), including advanced models
<div style="text-align: center;">Generated Instruction</div>
such as GPT-4, GPT-4-Turbo (evaluated during 2024 June), and GPT-3.5-Turbo-0301, as well as earlier models Text-Ada-001, Text-Davinci-003, Text-Davinci-002, and Text-Davinci-001. Additionally, we assess our approach on Llama-2-13B (Touvron et al., 2023), Llama-3-70B (AI@Meta, 2024), and PaLM2 text-bison (Anil et al., 2023). We set temperature as 0 for all models and tasks. Baseline: We compared our method against baseline methods: 1. Zero-Shot (Brown et al., 2020) triggers the response directly from the question without using previous examples or knowledge. 2. Zero-Shot-CoT (Kojima et al., 2022) extends the Zero-Shot method by adding the trigger "Let’s think step by step" at the answer phase to generate a reasoning path when reaching the answer. 3. Few-Shot (Brown et al., 2020): the demonstration includes randomly selected questions and annotated labels. 3. Few-Shot-CoT (Wei et al., 2022b): the demonstration includes queries and human annotated reasoning path to the answer. We employee humanwritten demonstrations from Wei et al. (2022b). 4. Auto-CoT (Zhang et al., 2022): For each retrieved query, CoT is generated with the Zero-ShotCoT trigger. Example queries are concatenated with generated CoT to form the demonstration. 5. Instruction Induction (Honovich et al., 2022) and APE (Zhou et al., 2022) asks the model to generate instruction based on a set of question-label pairs by a fixed template. 6. A concurrent work (Yasunaga et al., 2023) also proposes using LLMs to generate demonstrations. We compare their proposed prompt to generate ex-
Methods
GPT-4
GPT-4-Turbo
GPT-3.5-Turbo-0301
Llama-3-70B
Avg.
5-Shot-CoT (Brown et al., 2020)
68.5
74.0
57.0
70.5
67.5
Self-generated Exemplars (Yasunaga et al., 2023)
70.0
68.5
50.0
64.5
63.3
Auto-ICL (generating)
73.0
66.0
52.5
66.5
64.5
Auto-ICL (retrieving)
73.0
71.4
60.8
71.0
69.0
5-Shot-CoT (Brown et al., 2020)
88.5
83.0
73.0
84.0
82.1
Self-generated Exemplars (Yasunaga et al., 2023)
92.0
89.0
66.0
82.0
82.2
Auto-ICL (generating)
93.5
90.0
78.0
91.8
88.3
Auto-ICL (retrieving)
92.0
91.4
79.0
93.3
88.9
Table 2: The table compares our method with examplar-only methods and Few-Shot-CoT method across various models on the AQUA dataset (top) and GSM8K dataset (bottom). Auto-ICL (generating) refers to "Demonstration + Instruction" method and Auto-ICL (retrieving) refers to "Instruction" method in the Table 1. We use 5-shot examples in the demonstration.
emplars using our method. 7. DSPy frameworks (Khattab et al., 2023) enabling few-shot examples bootstrapping. We compared our method with the DSPy framework using the GSM8K dataset on GPT-3.5-turbo and GPT4-turbo 3. For the DSPy framework, we enabled the 5-Shot-CoT demonstration selection with 50 training examples and 200 development examples on GPT-3.5-turbo, 30 training examples, and 50 development examples on GPT-4-turbo. Detailed information about the triggers and examples for our method and baselines can be found in Section A.2.
Table 3: Comparison of Auto-ICL methods with DSPy framework on the GSM8K dataset.
Method
GPT-3.5-Turbo
GPT-4-Turbo
DSPy (Khattab et al., 2023)
81.0
93.0
Auto-ICL (generating)
82.0
90.0
Auto-ICL (retrieving)
83.0
91.4
# 5 Results
We tested demonstration-only, instruction-only, and demonstration-plue-instruction contexts on the GPT-3.5-Turbo-0301 model across various tasks and show the average accuracy of each task in Table 1 (full result is in Appendix A.1). In the retrieving case, the instruction-only context outperformed other contexts, which we denote as Auto-ICL (retrieving) from Table 2. The demonstration-plusinstruction context yielded better results in the generating case, which we denote as Auto-ICL (generating) from Table 2.
3As GPT-3.5-turbo-0301 was deprecated when we drew attention to this work during reviewing, we used the latest model instead
# 5.1 Comparison with baseline methods
Generated context outperforms human annotated context. Table 1 and Table 4 demonstrate that our method surpasses traditional methods, including Zero-CoT and Auto-CoT. It also outperforms traditional context generation methods using labeled data. Table 2 shows that our method outperforms the Few-Shot-CoT method and concurrent work that uses LLMs to generate demonstrations as context only. We also compare our method with demonstration selection with DSPy framework in Table 3, our method demonstrates performance similar to the DSPy framework. Additionally, the DSPy framework consumes a significant amount of credits for iterating through demonstrations in the training set, and it relies on demonstrations with labels. In contrast, our method substantially reduces inference costs and does not depend on human-written labels. Several factors contribute to this performance. Auto-ICL (generating) produces demonstrations and instructions tailored to the current question q. As a result, the generated contexts align closely with q from the model’s perspective. These findings are consistent with (Liu et al., 2022), which assert that selecting a demonstration highly correlated with the current request improves performance. Even if the generated demonstration has an incorrect label, the correctness of the label may not be crucial for problem-solving (Min et al., 2022). Additionally, instructions are more effective and informative than demonstrations with a chain of thought, surpassing both Few-Shot-CoT and AutoCoT methods. The generated instructions are summarized from the CoT process, encapsulating the necessary steps to resolve the question. By following these instructions, the model adheres to a
Methods
GPT-4
GPT-4-Turbo
GPT-3.5-Turbo
Text-Ada-001
Llama-3-70B
PaLM2 Text-Bison
Avg.
Zero-CoT (Brown et al., 2020)
70.5
73.0
37.4
18.0
72.0
53.0
45.1
Auto-CoT (Zhang et al., 2022)
75.5
64.0
53.0
18.5
46.5
54.8
43.2
Auto-ICL (generating)
73.0
66.0
52.5
33.3
66.5
58.5
52.7
Auto-ICL (retrieving)
73.0
71.4
60.8
15.0
71.0
52.5
49.8
Zero-CoT (Brown et al., 2020)
91.5
90.7
56.9
0.5
91.3
72.0
62.3
Auto-CoT (Zhang et al., 2022)
92.5
89.3
78.0
1.5
78.3
71.0
63.6
Auto-ICL (generating)
93.5
90.0
78.0
2.5
91.8
74.0
67.3
Auto-ICL (retrieving)
92.0
91.4
79.0
3.0
93.3
72.0
67.7
Table 4: The table compares our method with baseline methods across various models on the AQUA dataset. AutoICL (generating) refers to "Demonstration + Instruction" method and Auto-ICL (retrieving) refers to "Instruction" method in the Table 1. We use 5-shot examples in the demonstration.
consistent reasoning path, ensuring that each generative step is directed toward the final answer. This structured approach helps maintain coherence and accuracy throughout the problem-solving process.
consistent reasoning path, ensuring that each generative step is directed toward the final answer. This structured approach helps maintain coherence and accuracy throughout the problem-solving process. Instruction Interpretability: We present a comparison of generated instructions by APE, Instruction Induction, and Auto-ICL in Figure 4 In this example, the Instruction Induction method fails to generate an effective instruction, while APE relies on a specific example. Our method produces a more general and applicable instruction. In conclusion, our method’s explanations (instructions) on problems are more interpretable by human.
Instruction Interpretability: We present a comparison of generated instructions by APE, Instruction Induction, and Auto-ICL in Figure 4 In this example, the Instruction Induction method fails to generate an effective instruction, while APE relies on a specific example. Our method produces a more general and applicable instruction. In conclusion, our method’s explanations (instructions) on problems are more interpretable by human.
# 5.2 Are retrieved queries useful?
Retrieved data enhances context generation with richer information. In Table 1, retrieving methods consistently outperform generating methods, except in arithmetic tasks. This indicates that generating a general instruction that is adaptive and useful for various questions in a dataset benefits significantly from incorporating information from diverse examples within the dataset. By leveraging a broader context than simply q, the problemsolving process for a given question is augmented, leading to improved performance.
# 5.3 Instruction vs. Demonstration
In retrieving methods, instruction approaches consistently outperform demonstration and demonstration-plus-instruction methods in Table 1. This highlights the effectiveness of instructions in fine-tuning the model’s performance and guiding its problem-solving capabilities. Since the source of information remains the same (i.e., retrieved data), the repetitive usage of the same information in different formats (i.e., demonstrationplus-instruction) does not necessarily enhance the model’s problem-solving ability.
In generating methods, demonstration-plusinstruction outperformes instruction-only and demonstration-only approaches in Table 1. Combining demonstration and instruction proved to be more effective in generating methods than the independent use of instructions or demonstrations. Given that the information from a single question, q, may be limited, it can be inferred that expressing limited information in different formats—such as demonstration-plusinstruction—can enrich and extend the information used to address question q.
# 6 Discussion
From task-specific to universal in-context learning. Recently, significant research has been conducted on the design of prompts to enhance the performance of LLMs on specific tasks (Li et al., 2023; Hu et al., 2023; Wang et al., 2023; Sun et al., 2023). However, these task-specific methodologies often suffer from two main limitations: first, they require a degree of pre-existing knowledge about the task to design an effective prompt, and second, they are typically not transferable across different tasks, limiting their versatility. Therefore, instead of creating task-specific prompts, our method uses a single, broad strategy applicable to various scenarios. The model can process the context of the problem and generate relevant demonstrations or instructions, enabling it to tackle many tasks. This approach represents a shift from task-specific to universal ICL. Considering resources available under an incontext setting. Our research serves as an example in assessing available in-context resources for problem-solving while also proposing and comparing multiple ways to utilize these resources. This work focused on the impact of demonstrations and
Methods
Text-Davinci-003
Text-Davinci-002
Text-Davinci-001
Llama2-13B
Avg.
Zero-Shot
10.0
8.0
3.5
16.5
9.5
Zero-CoT (Brown et al., 2020)
52.5
36.5
17.5
23.5
32.5
Few-Shot (Brown et al., 2020)
13.0
11.5
5.5
3.5
8.4
Auto-CoT (Zhang et al., 2022)
59.3
52.0
13.6
19.5
36.1
Auto-ICL (generating)
56.4
41.0
3.5
24.5
31.3
Auto-ICL (retrieving)
50.0
29.0
6.0
20.0
26.3
Table 5: The table compares our method with baseline methods across various models on the AQUA dataset. Auto ICL (generating) refers to "Demonstration + Instruction" method and Auto-ICL (retrieving) refers to "Instruction method.
instructions, formats that have proven successful in ICL. However, other forms of ICL might also enhance LLM performance. Future research could delve into alternative formats such as rephrasing questions, enriching context with anecdotes or narratives, or applying different reasoning processes. Moreover, our research primarily accessed one form of resource - data retrieval from a dataset. There is room to consider other resources like realtime data from the web or integrating user feedback during the learning process. Each of these potential resources could provide additional context or information, further enhancing the problem-solving abilities of LLMs.
# 7 Conclusion
In this work, we introduce a novel framework called Automatic In-Context Learning. Our method enables LLMs to automatically generate context in response to user queries. This approach demonstrates superior performance across various tasks and mdoels compared to existing methods, showing that generated context can surpass human-written context. We also evaluate the effectiveness of different contexts depending on the availability of retrieved queries, highlighting the importance of wisely utilizing available resources for context generation to enhance problem-solving.
# 8 Limitations
In Table 5, we evaluate our methods on earlier OpenAI models, including Text-Davinci-001, 002, and 003, as well as the smaller Llama2-13B model. However, these methods do not perform as well compared to their performance on advanced models. This discrepancy arises because our approach heavily relies on the generation capabilities of the model. In addition, compared to GPT-3.5-Turbo
and GPT-4, the limitations of the Davinci series, such as a lack of instruction comprehension and sensitivity to prompt variations, can negatively impact performance. As a result, Auto-ICL underperforms with less capable models compared to the Zero-CoT method.
# References
AI@Meta. 2024. Llama 3 model card.
# AI@Meta. 2024. Llama 3 model card.
Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. arXiv preprint arXiv:2305.10403. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901. Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168. Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2022. A survey for in-context learning. arXiv preprint arXiv:2301.00234. Or Honovich, Uri Shaham, Samuel R Bowman, and Omer Levy. 2022. Instruction induction: From few examples to natural language task descriptions. arXiv preprint arXiv:2205.10782. Hanxu Hu, Hongyuan Lu, Huajian Zhang, Wai Lam, and Yue Zhang. 2023. Chain-of-symbol prompting elicits planning in large langauge models. arXiv preprint arXiv:2305.10276. Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T
Joshi, Hanna Moazam, et al. 2023. Dspy: Compiling declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714.
Joshi, Hanna Moazam, et al. 2023. Dspy: Compiling declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714. Hyuhng Joon Kim, Hyunsoo Cho, Junyeob Kim, Taeuk Kim, Kang Min Yoo, and Sang-goo Lee. 2022. Self-generated in-context learning: Leveraging autoregressive language models as a demonstration generator. arXiv preprint arXiv:2206.08082. Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916. Matthew Le, Y-Lan Boureau, and Maximilian Nickel. 2019. Revisiting the evaluation of theory of mind through question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5872–5877, Hong Kong, China. Association for Computational Linguistics. Junlong Li, Zhuosheng Zhang, and Hai Zhao. 2023. Self-prompting large language models for zero-shot open-domain qa. Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. 2017. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 158–167, Vancouver, Canada. Association for Computational Linguistics. Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2022. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pages 100–114, Dublin, Ireland and Online. Association for Computational Linguistics. Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland. Association for Computational Linguistics. Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners.
declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714. Hyuhng Joon Kim, Hyunsoo Cho, Junyeob Kim, Taeuk Kim, Kang Min Yoo, and Sang-goo Lee. 2022. Self-generated in-context learning: Leveraging autoregressive language models as a demonstration generator. arXiv preprint arXiv:2206.08082. Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916. Matthew Le, Y-Lan Boureau, and Maximilian Nickel. 2019. Revisiting the evaluation of theory of mind through question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5872–5877, Hong Kong, China. Association for Computational Linguistics. Junlong Li, Zhuosheng Zhang, and Hai Zhao. 2023. Self-prompting large language models for zero-shot open-domain qa. Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. 2017. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 158–167, Vancouver, Canada. Association for Computational Linguistics. Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2022. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pages 100–114, Dublin, Ireland and Online. Association for Computational Linguistics. Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland. Association for Computational Linguistics. Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners.
Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners.
Laria Reynolds and Kyle McDonell. 2021. Prompt programming for large language models: Beyond the few-shot paradigm. In Extended Abstracts of the 2021 CHI Conference on Human Factors in Computing Systems, pages 1–7. Subhro Roy and Dan Roth. 2016. Solving general arithmetic word problems. arXiv preprint arXiv:1608.01413. Maarten Sap, Ronan LeBras, Daniel Fried, and Yejin Choi. 2022. Neural theory-of-mind? on the limits of social intelligence in large lms. arXiv preprint arXiv:2210.13312. Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019. Social IQa: Commonsense reasoning about social interactions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4463– 4473, Hong Kong, China. Association for Computational Linguistics. Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Synthetic prompting: Generating chain-of-thought demonstrations for large language models. arXiv preprint arXiv:2302.00618. Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615. Simeng Sun, Yang Liu, Shuohang Wang, Chenguang Zhu, and Mohit Iyyer. 2023. Pearl: Prompting large language models to plan and execute actions over long documents. arXiv preprint arXiv:2305.14564. Zhiqing Sun, Xuezhi Wang, Yi Tay, Yiming Yang, and Denny Zhou. 2022. Recitation-augmented language models. arXiv preprint arXiv:2210.01296. Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288. Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. 2023. Planand-solve prompting: Improving zero-shot chain-ofthought reasoning by large language models. arXiv preprint arXiv:2305.04091. Albert Webson and Ellie Pavlick. 2022. Do promptbased models really understand the meaning of their prompts? In Proceedings of the 2022 Conference of the North American Chapter of the Association for
Albert Webson and Ellie Pavlick. 2022. Do promptbased models really understand the meaning of their prompts? In Proceedings of the 2022 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, pages 2300–2344, Seattle, United States. Association for Computational Linguistics.
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022a. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022b. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903. Michihiro Yasunaga, Xinyun Chen, Yujia Li, Panupong Pasupat, Jure Leskovec, Percy Liang, Ed H Chi, and Denny Zhou. 2023. Large language models as analogical reasoners. arXiv preprint arXiv:2310.01714. Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang. 2022. Generate rather than retrieve: Large language models are strong context generators. arXiv preprint arXiv:2209.10063. Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. 2022. Automatic chain of thought prompting in large language models. arXiv preprint arXiv:2210.03493. Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In International Conference on Machine Learning, pages 12697–12706. PMLR.
Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. 2022. Large language models are human-level prompt engineers. arXiv preprint arXiv:2211.01910.
# A Appendix
# A.1 Full results
Full results of of Table 1 are provided in the Table 6 and Table 7. For each dataset, we randomly sample 200 data points to do the evaluation and remove invalid API responses when calculating the accuracy.
# A.2 Experimental settings A.2.1 Answer extraction
# A.2 Experimental settings
Table 8 presents prompts we utilize for answer extraction under the Zero-Shot setting and other settings.
# A.2.2 Demonstration generation
We display an example of demonstration generation on the GSM8K dataset in Table 10.
# 2.3 Instruction generation
In this section, we demonstrate the generation of instructions on the GSM8K dataset, and provide examples of the generated instructions. We present an example of the Auto-ICL generating 1-to-1 method in Table 11, and exhibit the instructions generated from each dataset in Tables 12, 13, 14, and 15. The instructions produced through the Auto-ICL N-to-1 method are displayed in Tables 16, 17, 18, and 19.
# A.2.4 Demonstration-Plus-Instruction
We present an example illustrating the concatenation of demonstrations with instructions in the context in Table 20.
A.2.5 Few-Shot Demonstration
# A.2.5 Few-Shot Demonstration
We present the demonstrations used in the FewShot method as shown in Table 1. Within each dataset, we randomly selected 5 examples with labels and combined them in a random order.
This section shows instructions generated by the APE method (Zhou et al., 2022) in Tables 21 through 26.
# A.2.7 Instructions from Instruction Induction
This section shows instructions generated by the Instruction Induction method (Honovich et al., 2022) in Tables 27 through 29.
Methods
Labeled Data
Theory of Mind
Symbolic Reasoning
TOMI1
TOMI2
SocialIQA
Coin Flip
Last Letters
Generating
Zero-Shot (Brown et al., 2020)
✗
54.5
27.5
63.0
30.7
1.5
Zero-Shot-CoT (Kojima et al., 2022)
✗
52.5
22.0
69.5
81.2
80.0
Auto-ICL (ours)
✗
64.4
47.6
66.5
76.5
85.0
Retrieving
Few-Shot (Brown et al., 2020)
✓
67.5
50.5
72.5
61.9
7.9
APE (Zhou et al., 2022)
✓
66.0
30.0
24.5
54.5
87.0
Induction (Honovich et al., 2022)
✓
49.5
15.0
64.0
92.0
74.5
Auto-CoT (Zhang et al., 2022)
✗
51.5
23.0
69.0
99.0
85.5
Methods
Labeled Data
Theory of Mind
Symbolic Reasoning
MultiArith
AQUA
GSM8K
Cycle Letters
Shuffle Objects
Generating
Zero-Shot (Brown et al., 2020)
✗
82.7
27.6
22.7
55.0
34.2
Zero-Shot-CoT (Kojima et al., 2022)
✗
97.4
37.4
56.9
64.0
62.1
Auto-ICL (ours)
✗
92.5
60.8
78.0
67.0
52.0
Retrieving
Few-Shot (Brown et al., 2020)
✓
82.9
31.1
23.1
66.0
32.6
APE (Zhou et al., 2022)
✓
92.5
43.5
65.5
70.0
56.5
Induction (Honovich et al., 2022)
✓
94.0
52.5
73.5
69.0
64.5
Auto-CoT (Zhang et al., 2022)
✗
98.5
53.0
78.0
74.0
57.5
Methods
Table 6: This table compares the accuracy across various tasks using different baseline methods and Auto-ICL. The methods are classified into two categories, "retrieving" and "generating", based on whether the data is sourced from the dataset. The usage of labeled demonstrations is indicated in the column named "labeled data".
Theory of Mind
Symbolic Reasoning
TOMI1
TOMI2
SocialIQA
Coin Flip
Last Letters
Generating
Demonstration
58.0
21.5
56.3
63.0
89.5
Instruction
64.4
47.6
66.5
74.6
75.9
Demonstration-Plus-Instruction
67.8
20.1
65.5
100.0
84.8
Retrieving
Demonstration
52.0
23.0
69.0
99.0
86.0
Instruction
52.0
71.0
61.5
99.0
87.5
Demonstration-Plus-Instruction
28.0
67.0
47.5
94.5
88.5
Arithmetic
Others
MultiArith
AQUA
GSM8K
Cycle Letters
Shuffle Objects
Generating
Demonstration
93.0
58.3
74.5
65.0
50.5
Instruction
95.3
50.3
72.0
71.7
44.8
Demonstration-Plus-Instruction
95.4
48.5
70.6
70.0
58.1
Retrieving
Demonstration
93.0
58.0
75.0
65.0
51.0
Instruction
95.0
52.5
79.0
68.0
68.5
Demonstration-Plus-Instruction
93.5
52.6
76.5
72.5
60.5
Table 7: Comparison of instruction, demonstration, and instruction-plus-demonstration on each dataset. Th methods are classified as ’retrieving’ or ’generating’ depending on their utilization of dataset retrieval.
Table 7: Comparison of instruction, demonstration, and instruction-plus-demonstration on each dataset. These methods are classified as ’retrieving’ or ’generating’ depending on their utilization of dataset retrieval.
<div style="text-align: center;">Table 8: Answer extraction prompts used for experiments in Table 1</div>
No
Task
Zero-Shot
Others
1
TOMI1
Among A through C, the answer is
Therefore, among A through C, the answer is
2
TOMI2
Among A through C, the answer is
Therefore, among A through C, the answer is
3
SocialIQA
Among A through C, the answer is
Therefore, among A through C, the answer is
4
Coin Flip
The answer (Yes or No) is
Therefore, the answer (Yes or No) is
5
Last Letters
The concatenated letters is
Therefore, the concatenated letters is
6
MultiArith
The answer (arabic numerals) is
Therefore, the answer (arabic numerals) is
7
AQUA
Among A through E, the answer is
Therefore, among A through E, the answer is
8
GSM8K
The answer (arabic numerals) is
Therefore, the answer (arabic numerals) is
9
Cycle Letters
Therefore, the word is
Therefore, the word is
10
Shuffle Objects
Among A through C, the answer is
Therefore, among A through C, the answer is
Table 9: An actual example of N-to-1 instruction generation from the GSM8K dataset and GPT-3.5-Turbo-030 model. The input comprises questions and reasoning paths. Reasoning paths are summarized as a single instructio by the model.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/863d/863dab8c-2fca-4e2b-bb0f-b2e0a373424b.png" style="width: 50%;"></div>
Instruction N-to-1 generation: (Q, CoT) →ˆi Input: Question: Gunther, the gorilla, had 48 bananas hidden under a fern branch. When Gunther wasn’t looking, Arnold, the chimpanzee, stole half of the bananas from the pile. The next day, Gunther added another 25 bananas to his pile, but later that evening, Arnold stole another 12 of the bananas. On the third day, Gunther added another 6 bananas to his pile and began counting bananas. How many bananas did Gunther find were in the pile? Answer: First, we need to figure out how many bananas Arnold stole from Gunther’s original pile of 48. Half of 48 is 24, so Arnold took 24 bananas. After Arnold’s theft, Gunther had 48 - 24 = 24 bananas left. The next day, Gunther added 25 bananas to his pile, bringing the total to 24 + 25 = 49 bananas. But then Arnold stole another 12 bananas, leaving Gunther with 49 - 12 = 37 bananas. On the third day, Gunther added 6 more bananas, bringing the total to 37 + 6 = 43 bananas. Therefore, Gunther found 43 bananas in his pile. ... ... Demonstrations with chain-of-thought ......
Question: Charlie had 10 stickers. He bought 21 stickers from a store in the mall and got 23 stickers for his birthday. Then Charlie gave 9 of the stickers to his sister and used 28 to decorate a greeting card. How many stickers does Charlie have left? Answer: Let’s think step by step. First, we need to convert the number of each coin/bill into its value in dollars. - 100 pennies = 1.00 - 40 nickels = 2.00 (each nickel is worth 5 cents, so 40 nickels is 40 x 5 = 200 cents, or 2.00) - 20 dimes = 2.00 (each dime is worth 10 cents, so 20 dimes is 20 x 10 = 200 cents, or 2.00) - 40 dollar bills = 40.00 Next, we add up the values of each coin/bill: 1.00 + 2.00 + 2.00 + 40.00 = 45.00 Therefore, Brady has 45.00 in his piggy bank.
1. Read the problem carefully and identify what information is given and what is being asked. 2. Determine what operations are needed to solve the problem (addition, subtraction, multiplication, division, etc.). 3. Use the given information to set up an equation or equations to solve for the unknown variable. 4. Solve the equation(s) using the appropriate operations. 5. Check your answer to make sure it makes sense in the context of the problem. 6. If necessary, convert units or simplify the answer to the appropriate form (fractions, decimals, percentages, etc.).
Table 10: An example illustrating our demonstration generation process. First, we prompt the model to examples based on the current question. Then, these examples are decomposed to generate CoTs with the  "Let’s think step by step". Finally, these examples and their CoTs serve as demonstrations.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a33e/a33e201a-270b-4eae-894e-9af5e1de43af.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8489/84892bdd-1571-4ce5-ad2d-f8300c7d1453.png" style="width: 50%;"></div>
Table 11: An example illustrating our instruction generation process. Firstly, we prompt the model to generate instruction that are based on the current question. Secondly, the generated instruction is used to aid in addressing the original question.
nstruction that are based on the current question. Secondly, the generated instruction is used to aid in addressing he original question. Instruction 1-to-1 generation: q →i Q: Generate a plan to solve this question: "Luke is spending time at the beach building sandcastles. He eventually notices that each level of a sandcastle will have half the square footage as the level below it. If he makes a 4 leveled sandcastle where the top level has a square footage of 16, what is the average square footage of a level?" A: 1. Determine the square footage of the bottom level of the sandcastle. 2. Use the given information that each level has half the square footage of the level below it to find the square footage of each level. 3. Add up the square footage of all four levels. 4. Divide the total square footage by the number of levels to find the average square footage per level. Q: Plan: 1. Determine the square footage of the bottom level of the sandcastle. 2. Use the given information that each level has half the square footage of the level below it to find the square footage of each level. 3. Add up the square footage of all four levels. 4. Divide the total square footage by the number of levels to find the average square footage per level. Question: Luke is spending time at the beach building sandcastles. He eventually notices that each level of a sandcastle will have half the square footage as the level below it. If he makes a 4 leveled sandcastle where the top level has a square footage of 16, what is the average square footage of a level? Solve the question step by step based on the plan. A: Step 1: Determine the square footage of the bottom level of the sandcastle. Since the top level has a square footage of 16, we can use the given information that each level has half the square footage of the level below it to work backwards and find the square footage of the bottom level. Level 4: 16 square feet Level 3: 16 x 2 = 32 square feet Level 2: 32 x 2 = 64 square feet Level 1: 64 x 2 = 128 square feet Therefore, the square footage of the bottom level is 128 square feet. Step 2: Use the given information that each level has half the square footage of the level below it to find the square footage of each level. We have already found the square footage of each level in Step 1. Step 3: Add up the square footage of all four levels. We can add up the square footage of all four levels to find the total square footage of the sandcastle. Total square footage = 16 + 32 + 64 + 128 = 240 square feet Step 4: Divide the total square footage by the number of levels to find the average square footage per level. Since there are four levels in the sandcastle, we can divide the total square footage by 4 to find the average square footage per level. Average square footage per level = 240 / 4 = 60 square feet Therefore, the average square footage of a level in the sandcastle is 60 square feet.
Instruction 1-to-1 generation: q →i
Q: Generate a plan to solve this question: "Luke is spending time at the beach building sandcastles. He
eventually notices that each level of a sandcastle will have half the square footage as the level below it. If
he makes a 4 leveled sandcastle where the top level has a square footage of 16, what is the average square
footage of a level?"
A: 1. Determine the square footage of the bottom level of the sandcastle.
2. Use the given information that each level has half the square footage of the level below it to find the
square footage of each level.
3. Add up the square footage of all four levels.
4. Divide the total square footage by the number of levels to find the average square footage per level.
Q:
Plan: 1. Determine the square footage of the bottom level of the sandcastle. 2. Use the given information
that each level has half the square footage of the level below it to find the square footage of each level. 3.
Add up the square footage of all four levels. 4. Divide the total square footage by the number of levels to
find the average square footage per level.
Question: Luke is spending time at the beach building sandcastles. He eventually notices that each level
of a sandcastle will have half the square footage as the level below it. If he makes a 4 leveled sandcastle
where the top level has a square footage of 16, what is the average square footage of a level?
Solve the question step by step based on the plan.
A: Step 1: Determine the square footage of the bottom level of the sandcastle. Since the top level has a
square footage of 16, we can use the given information that each level has half the square footage of the
level below it to work backwards and find the square footage of the bottom level.
Level 4: 16 square feet Level 3: 16 x 2 = 32 square feet Level 2: 32 x 2 = 64 square feet Level 1: 64 x 2 =
128 square feet
Therefore, the square footage of the bottom level is 128 square feet.
Step 2: Use the given information that each level has half the square footage of the level below it to find
the square footage of each level. We have already found the square footage of each level in Step 1.
Step 3: Add up the square footage of all four levels. We can add up the square footage of all four levels to
find the total square footage of the sandcastle.
Total square footage = 16 + 32 + 64 + 128 = 240 square feet
Step 4: Divide the total square footage by the number of levels to find the average square footage per level.
Since there are four levels in the sandcastle, we can divide the total square footage by 4 to find the average
square footage per level.
Average square footage per level = 240 / 4 = 60 square feet
Therefore, the average square footage of a level in the sandcastle is 60 square feet.
Instruction 1-to-1 generation: q →i
Q: Generate a plan to solve this question: "Luke is spending time at the beach building sandcastles. He
eventually notices that each level of a sandcastle will have half the square footage as the level below it. If
he makes a 4 leveled sandcastle where the top level has a square footage of 16, what is the average square
footage of a level?"
A: 1. Determine the square footage of the bottom level of the sandcastle.
2. Use the given information that each level has half the square footage of the level below it to find the
square footage of each level.
3. Add up the square footage of all four levels.
4. Divide the total square footage by the number of levels to find the average square footage per level.
Q:
Plan: 1. Determine the square footage of the bottom level of the sandcastle. 2. Use the given information
that each level has half the square footage of the level below it to find the square footage of each level. 3.
Add up the square footage of all four levels. 4. Divide the total square footage by the number of levels to
find the average square footage per level.
Question: Luke is spending time at the beach building sandcastles. He eventually notices that each level
Question: Luke is spending time at the beach building sandcastles. He eventually notices that each leve of a sandcastle will have half the square footage as the level below it. If he makes a 4 leveled sandcastl where the top level has a square footage of 16, what is the average square footage of a level? Solve the question step by step based on the plan.
Dataset
Question
Instruction
TOMI1
Ava entered the sunroom. Lily entered
the sunroom. Ava loves the skirt. The
asparagus is in the bathtub. Ava exited
the sunroom. Lily moved the aspara-
gus to the pantry. Lily dislikes the skirt.
Lily exited the sunroom. Ava entered
the garage. Where will Ava look for the
asparagus? Answer Choices: (A) bath-
tub (B) pantry (C) Unknown
1. Read the given information carefully to un-
derstand the sequence of events. 2. Identify
the key details in the passage, such as the lo-
cation of the asparagus and the actions of Ava
and Lily. 3. Note that Ava entered the sunroom
and Lily moved the asparagus to the pantry af-
ter Ava left. 4. Since Ava did not witness Lily
moving the asparagus, she would not know
that it was moved to the pantry. 5. Therefore,
Ava would most likely look for the asparagus
in its original location, which is the bathtub. 6.
Therefore, the answer is (A) bathtub.
TOMI2
Mason entered the laundry. Alexander
dislikes the underpants. Alexander en-
tered the laundry. The onion is in the
treasure chest. Mason exited the laun-
dry. Alexander moved the onion to the
crate. Ella likes the pajamas. Alexander
exited the laundry. Mason entered the
laundry. Where does Mason think that
Alexander searches for the onion? An-
swer Choices: (A) Unknown (B) trea-
sure chest (C) crate
1. Read the given information carefully to un-
derstand the sequence of events. 2. Identify
the key characters and objects mentioned in the
question. 3. Note down the actions performed
by each character and the location of the ob-
jects. 4. Analyze the information to determine
where Mason thinks Alexander searches for
the onion. 5. Eliminate answer choices that
are not supported by the given information. 6.
Choose the answer choice that is the most log-
ical based on the given information. 7. Verify
the answer by checking if it aligns with the
sequence of events and actions performed by
the characters.
SocialIQA
Riley is the most popular student in the
school. What will Others want to do
next?
Answer Choices: (A) run for
class president (B) run for student coun-
cil (C) join Riley’s gang
1. Identify the reasons why Riley is the most
popular student in the school. 2. Analyze the
characteristics of Riley’s popularity and de-
termine what aspects of it others may want
to emulate. 3. Consider the potential motiva-
tions of the other students and what they hope
to achieve by following in Riley’s footsteps.
4. Evaluate the answer choices provided and
determine which option aligns best with the
motivations and goals of the other students. 5.
Choose the answer choice that best fits the sit-
uation.
Mason entered the laundry. Alexander dislikes the underpants. Alexander entered the laundry. The onion is in the treasure chest. Mason exited the laundry. Alexander moved the onion to the crate. Ella likes the pajamas. Alexander exited the laundry. Mason entered the laundry. Where does Mason think that Alexander searches for the onion? Answer Choices: (A) Unknown (B) treasure chest (C) crate
Table 12: Examples of generated instructions using the q →ˆi method for Theory of Mind tasks.
1. Identify the reasons why Riley is the most popular student in the school. 2. Analyze the characteristics of Riley’s popularity and determine what aspects of it others may want to emulate. 3. Consider the potential motivations of the other students and what they hope to achieve by following in Riley’s footsteps. 4. Evaluate the answer choices provided and determine which option aligns best with the motivations and goals of the other students. 5. Choose the answer choice that best fits the situation.
Dataset
Question
Instruction
Coin Flip
A coin is heads up. Bill flips the coin.
Santos does not flip the coin. Roxy flips
the coin. Randi flips the coin. Is the
coin still heads up? Note that "flip" here
means "reverse".
1. Determine the starting position of the coin
(heads up). 2. Understand the meaning of
"flip" in this context (reverse). 3. Determine
the order of the flips: Bill, Santos, Roxy, Randi.
4. Apply the flip to the starting position of
the coin for each person in the order given.
5. Determine the final position of the coin
after all flips have been applied. 6. Compare
the final position to the starting position to
determine if the coin is still heads up.
Last Letter
Take the last letters of each words in
"Rena Devon Rosalinda Paulina" and
concatenate them.
1. Identify the given words: "Rena Devon
Rosalinda Paulina" 2. Determine the last letter
of each word. 3. Write down the last letters of
each word. 4. Concatenate the last letters in
the order they appear in the original phrase. 5.
Verify that the resulting string is correct.
nerated instructions using the q →ˆi method for Symbolic Reasoning ta
Dataset
Question
Instruction
MultiArith
Paige and her friends were recycling pa-
per for their class. For every 4 pounds
they recycled they earned one point.
If Paige recycled 14 pounds and her
friends recycled 2 pounds, how many
points did they earn?
1. Determine the total weight of paper recy-
cled by Paige and her friends. 2. Add the
weight of paper recycled by Paige and her
friends. 3. Divide the total weight of paper re-
cycled by 4 to determine the number of points
earned. 4. Round the answer to the nearest
whole number, as points are given in whole
numbers. 5. Write the final answer in the form
of a sentence.
AQUA
In a row of children Neha is 12th from
left end and Radha is 6th from right
end. When Radha is shifted to left by 2
places and Neha is shifted to right by 2
places there 6 children between Radha
and Neha. How many children are there
in the row? Answer Choices: (A) 23
(B) 27 (C) 26 (D) 28 (E) 29
1. Draw a diagram to represent the given in-
formation. 2. Use the information given to
determine the total number of children in the
row. 3. Determine the original positions of
Neha and Radha in the row. 4. Determine
the new positions of Neha and Radha after
they are shifted. 5. Use the information given
to determine the number of children between
Neha and Radha after they are shifted. 6. Use
the information gathered to solve for the total
number of children in the row. 7. Check the
answer choices and select the correct answer.
GSM8K
Carly is making cupcakes and brownies
for her art class. She makes 2 less than
three times as many brownies as cup-
cakes. If Carly’s class has five people
and each person gets two treats, how
many cupcakes did Carly make?
1. Use algebra to set up an equation to repre-
sent the relationship between the number of
cupcakes and brownies Carly made. 2. Solve
the equation to find the number of brownies
Carly made. 3. Use the number of brownies
to find the number of cupcakes Carly made. 4.
Multiply the number of cupcakes by 2 to find
the total number of treats. 5. Divide the total
number of treats by 10 (5 people x 2 treats per
person) to find the number of cupcakes each
person gets.
Table 14: Examples of generated instructions using the q →ˆi method for Arithmetic tasks.
Dataset
Question
Instruction
Cycle Letter
Please unscramble the letters into a
word, and write that word: ebukkak =
1. Identify the letters in the scrambled word:
ebukkak. 2. Rearrange the letters to form a
word. 3. Check if the word makes sense and
is spelled correctly. 4. Write the unscrambled
word.
Shuffle Object
Alice, Bob, and Claire are dancers at a
square dance. At the start of a song, they
each have a partner: Alice is dancing
with Sam, Bob is dancing with Helga,
and Claire is dancing with Karl.
Throughout the song, the dancers often
trade partners. First, Claire and Alice
switch partners. Then, Bob and Alice
switch partners. Finally, Claire and Bob
switch partners. At the end of the dance,
Alice is dancing with Which choice is
true ? Answer Choices: (A) Sam. (B)
Helga. (C) Karl.
1. Understand the problem: Read the prob-
lem carefully and make sure you understand
the given information and what is being asked.
2. Visualize the problem: Draw a diagram or
use a table to keep track of the dancers and
their partners before and after each switch. 3.
Follow the steps: Follow the given steps in
the problem and make sure to update the dia-
gram or table after each switch. 4. Determine
the final partner: After the final switch, deter-
mine who Alice is dancing with and compare
it to the answer choices given. 5. Choose the
correct answer: Select the answer choice that
matches the final partner Alice is dancing with.
Examples of generated instructions using the q →ˆi method for other ta
Dataset
Question
TOMI1
Ava entered the sunroom. Lily entered
the sunroom. Ava loves the skirt. The
asparagus is in the bathtub. Ava exited
the sunroom. Lily moved the aspara-
gus to the pantry. Lily dislikes the skirt.
Lily exited the sunroom. Ava entered
the garage. Where will Ava look for the
asparagus? Answer Choices: (A) bath-
tub (B) pantry (C) Unknown
TOMI2
Mason entered the laundry. Alexander
dislikes the underpants. Alexander en-
tered the laundry. The onion is in the
treasure chest. Mason exited the laun-
dry. Alexander moved the onion to the
crate. Ella likes the pajamas. Alexander
exited the laundry. Mason entered the
laundry. Where does Mason think that
Alexander searches for the onion? An-
swer Choices: (A) Unknown (B) trea-
sure chest (C) crate
SocialIQA
Riley is the most popular student in the
school. What will Others want to do
next?
Answer Choices: (A) run for
class president (B) run for student coun-
cil (C) join Riley’s gang
Dataset
Question
Instruction
TOMI1
Ava entered the sunroom. Lily entered
the sunroom. Ava loves the skirt. The
asparagus is in the bathtub. Ava exited
the sunroom. Lily moved the aspara-
gus to the pantry. Lily dislikes the skirt.
Lily exited the sunroom. Ava entered
the garage. Where will Ava look for the
asparagus? Answer Choices: (A) bath-
tub (B) pantry (C) Unknown
1. Identify the characters and locations men-
tioned in the given information. 2. Look for
any objects or items that are mentioned and
their initial location. 3. Check if any charac-
ters move or interact with the objects/items
and note their new location. 4. Determine
which character is being asked about and their
relationship to the object/item. 5. Use the
information gathered to determine where the
character would look for the object/item. If
there is not enough information, choose "Un-
known" as the answer.
TOMI2
Mason entered the laundry. Alexander
dislikes the underpants. Alexander en-
tered the laundry. The onion is in the
treasure chest. Mason exited the laun-
dry. Alexander moved the onion to the
crate. Ella likes the pajamas. Alexander
exited the laundry. Mason entered the
laundry. Where does Mason think that
Alexander searches for the onion? An-
swer Choices: (A) Unknown (B) trea-
sure chest (C) crate
1. Identify the key information in the given
statements, such as the location of objects and
the actions of individuals. 2. Determine if
there is any interaction between the individu-
als mentioned in the statements. 3. Look for
any statements that indicate where an object
has been moved to. 4. Consider any personal
preferences or dislikes mentioned in the state-
ments, but only if they are relevant to the loca-
tion of the object. 5. If the question asks where
one individual thinks another individual will
search for an object, use the information about
where the object has been moved to and any
interactions between the individuals to make
an educated guess. 6. If there is not enough
information to determine the answer, choose
the option for "Unknown".
SocialIQA
Riley is the most popular student in the
school. What will Others want to do
next?
Answer Choices: (A) run for
class president (B) run for student coun-
cil (C) join Riley’s gang
The general chain of thought for all questions
is to analyze the given information and deter-
mine the most logical and likely outcome or
next step based on that information. This in-
volves considering the context of the situation,
the actions and behaviors of the individuals
involved, and any relevant factors that may im-
pact the outcome. It also involves eliminating
answer choices that are irrelevant or unlikely
based on the information provided. Ultimately,
the goal is to arrive at the most accurate and
reasonable answer based on the given informa-
tion.
Mason entered the laundry. Alexander dislikes the underpants. Alexander entered the laundry. The onion is in the treasure chest. Mason exited the laundry. Alexander moved the onion to the crate. Ella likes the pajamas. Alexander exited the laundry. Mason entered the laundry. Where does Mason think that Alexander searches for the onion? Answer Choices: (A) Unknown (B) treasure chest (C) crate
Table 16: Generated instructions for all data points within each Theory of Mind (ToM) dataset, achieved via the generating path Q → ˆ CoT →ˆI.
1. Identify the key information in the given statements, such as the location of objects and the actions of individuals. 2. Determine if there is any interaction between the individuals mentioned in the statements. 3. Look for any statements that indicate where an object has been moved to. 4. Consider any personal preferences or dislikes mentioned in the statements, but only if they are relevant to the location of the object. 5. If the question asks where one individual thinks another individual will search for an object, use the information about where the object has been moved to and any interactions between the individuals to make an educated guess. 6. If there is not enough information to determine the answer, choose the option for "Unknown".
Dataset
Question
Instruction
Coin Flip
A coin is heads up. Bill flips the coin.
Santos does not flip the coin. Roxy flips
the coin. Randi flips the coin. Is the
coin still heads up? Note that "flip" here
means "reverse".
1. Identify the initial state of the coin (heads
up or tails up). 2. For each person who flips
the coin, reverse the current state of the coin.
3. For each person who does not flip the coin,
the state of the coin remains the same. 4. De-
termine the final state of the coin after all the
flips have been completed.
Last Letter
Take the last letters of each words in
"Rena Devon Rosalinda Paulina" and
concatenate them.
To find the answer for each question, we need
to identify the last letter of each word in the
given list. Then, we concatenate these letters
in the order they appear to get the final answer.
Table 17: Generated instructions for all data points within each Symbolic Reasoning dataset, achieved via the generating path Q → ˆ CoT →ˆI.
Dataset
Question
I
MultiArith
Paige and her friends were recycling pa-
per for their class. For every 4 pounds
they recycled they earned one point.
If Paige recycled 14 pounds and her
friends recycled 2 pounds, how many
points did they earn?
S
q
b
i
t
t
c
AQUA
In a row of children Neha is 12th from
left end and Radha is 6th from right
end. When Radha is shifted to left by 2
places and Neha is shifted to right by 2
places there 6 children between Radha
and Neha. How many children are there
in the row? Answer Choices: (A) 23
(B) 27 (C) 26 (D) 28 (E) 29
1
y
t
3
w
e
p
m
s
f
a
s
t
GSM8K
Carly is making cupcakes and brownies
for her art class. She makes 2 less than
three times as many brownies as cup-
cakes. If Carly’s class has five people
and each person gets two treats, how
many cupcakes did Carly make?
1
w
a
n
t
t
e
4
o
s
l
t
d
Dataset
Question
Instruction
MultiArith
Paige and her friends were recycling pa-
per for their class. For every 4