# Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach
Leyu Lin, and Ji-Rong Wen junjie.zhang@ruc.edu.cn,batmanfly@gmail.com 1Gaoling School of Artificial Intelligence, Renmin University of China 2WeChat, Tencent China
# ABSTRACT
In the past decades, recommender systems have attracted much attention in both research and industry communities, and a large number of studies have been devoted to developing effective recommendation models. Basically speaking, these models mainly learn the underlying user preference from historical behavior data (typically in the forms of item IDs), and then estimate the user-item matching relationships for recommendations. Inspired by the recent progress on large language models (LLMs), we take a different approach to developing the recommendation models, considering recommendation as instruction following by LLMs. The key idea is that the preferences or needs of a user can be expressed in natural language descriptions (called instructions), so that LLMs can understand and further execute the instruction for fulfilling the recommendation task. Instead of using public APIs of LLMs, we instruction tune an open-source LLM (3B Flan-T5-XL), in order to better adapt LLMs to recommender systems. For this purpose, we first design a general instruction format for describing the preference, intention, task form and context of a user in natural language. Then we manually design 39 instruction templates and automatically generate a large amount of user-personalized instruction data (252K instructions) with varying types of preferences and intentions. To demonstrate the effectiveness of our approach, we instantiate the instruction templates into several widely-studied recommendation (or search) tasks, and conduct extensive experiments on these tasks with real-world datasets. Experiment results show that the proposed approach can outperform several competitive baselines, including the powerful GPT-3.5, on these evaluation tasks. Our approach sheds light on developing more user-friendly recommender systems, in which users can freely communicate with the system and obtain more accurate recommendations via natural language instructions.
arXiv:2305.07001v
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. Conference acronym ’XX, August 03–05, 2022, Woodstock, NY © 2022 Association for Computing Machinery. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/XXXXXXX.XXXXXXX
KEYWORDS
Large language models, instruction tuning, recommender systems
ACM Reference Format: Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin and Ji-Rong Wen. 2022. Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 13 pages. https: //doi.org/XXXXXXX.XXXXXXX
# 1 INTRODUCTION
Nowadays, recommendation systems have been widely deployed in various application platforms, which aim to satisfy user’s needs and promote the use (or sale) of available resources. As the early approaches, collaborative filtering algorithms [24, 30] were proposed by recommending items based on similar tastes (user side) or similar characteristics (item side). Subsequently, matrix factorization [23] and neural networks [15, 21] were adopted to develop the recommendation models, which can capture more complex user preferences and learn more accurate user-item relationships. Despite the huge progress, existing recommendation algorithms are mainly trained by fitting the user-item interaction data, lacking a generalization ability to unseen settings (e.g., new users or new tasks). Besides, users are passively involved in conventional recommendation algorithms, unable to explicitly express their real needs in a flexible form.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a28f/a28f236d-3811-435f-9d71-bd672dfc5f41.png" style="width: 50%;"></div>
# Figure 1: A framework of our proposed InstructRec.
Recently, pre-trained large language models (LLM) [34, 41, 44] (e.g., T5 [29] and GPT-3 [4]) have shown remarkable abilities on a variety of natural language tasks, which also shed lights on developing more general and effective recommender systems [2, 6, 7, 13, 17, 18]. For example, it has been shown that language models can improve
the transferability of recommender systems [9, 17, 18], and also enhance the user-system interaction [6, 13, 35]. However, language models are built on natural language text, and they are not directly suitable for recommender systems, which need to deal with behavior data instead of text data. To bridge the gap between language models and recommender systems, a promising approach is to consider behavior modeling as language modeling [6, 12, 13, 26]: it formats the recommendation tasks in the form of natural language text and then treat them as a language processing task. In this approach, two fundamental problems are key to the success of item recommendations: (1) How to verbalize the recommendation tasks? In general, a successful recommendation relies on the accurate understanding of the underlying user needs. It is important to design a suitable form to comprehensively describe the recommendation task in natural language, which should contain various kinds of useful information that reveal user’s needs, including the interaction history, user preference or intention, and other personalized factors. (2) How to adapt LLMs for recommendation? Although LLMs are capable of modeling and generating the natural language, it generally has difficulty in solving complex specialized tasks [11, 25] despite the generality. It requires specific tuning strategies that adjust the skills of LLMs towards the recommendation tasks, so as to bridge the gap and accurately understand user’s behavior data. In this paper, we would like to advance one step further in LLMempowered recommender systems. Specially, we take a user-centric perspective, and consider the recommendation task as instruction following/execution by LLMs. In this new recommendation paradigm, a user can freely express the specific needs via instruction texts, while the recommender system will follow the user’s instructions for making accurate recommendations based on powerful LLMs. We make two major technical contributions: (1) A formal introduction of instructions for recommendation. We formally introduce the concept of instruction in recommender systems. Despite the similar efforts in prior studies [2, 13], we discuss the expression of user needs in the context of LLMs, and treat recommendation as a specific task approached by instruction following. Further, we present a detailed design for composing the instruction and instantiate it for solving different recommendation tasks with specific user needs or preferences. (2) An instruction tuned LLM for recommendation. To adapt LLMs for recommender systems, we further employ instruction tuning to optimize LLMs according to the recommendation objective. For this purpose, we automatically generate a large number of recommendation-oriented instruction data for tuning the LLMs. Instruction tuning is key to elicit the abilities of LLMs for the recommendation tasks, and it further enables the generalization of LLMs on diverse user needs or requests. To this end, we propose an instruction tuning approach for recommender systems, named InstructRec, which is a new recommendation paradigm that allows users to freely express their specific needs in natural language. To develop the recommender, our
base model is built on the 3B Flan-T5-XL [5]1. As the key to our approach, it is important to design the instruction format and construct the instruction data. Firstly, we formulate the instruction format by integrating four major parts, including preference, intention, task form and context. As we will see, such an instruction form can cover a variety of specific user needs, and is particularly suitable for formatting different recommendation tasks. Secondly, we manually design 39 coarse-grained instruction templates to cover interaction scenarios, and automatically generate 252K finegrained user personalized instruction data with varying types of user preferences and intentions. In order to generate high-quality instruction data, we employ GTP-3.5 to generate the user preferences or intentions based on real historical data (e.g., historical interaction or review texts) [36]. Further, we propose a series of useful strategies to increase the diversity of instructions, since it is important to improve the performance of instruction tuning [5, 37]. By tuning the LLM with these recommendation-oriented instruction data, the base model can be well adapted to recommender systems, and learn to follow the user’s instructions for fulfilling the corresponding recommendation tasks. Although we only consider single-turn instruction following, it would be promising to extend the current approach to a multi-turn conversation scenario. Such a task specialization approach of LLMs can enhance the overall generalization ability of the recommendation models and improve the user experiences in recommender systems. To evaluate the proposed approach, we conduct extensive experiments by constructing a wide range of interaction scenarios from real-world datasets. The proposed approach achieves promising performance compared to several competitive baselines. The experimental results also demonstrate that InstructRec can effectively improve the LLM’s ability of accommodating the diverse user needs. Moreover, the performance on the held-out instructions and domains also verifies that our approach has a better generalization ability. The main contributions of this work are concluded as follows: • We cast recommendation as instruction following by LLMs, and introduce a new recommendation paradigm that allows users to freely express their diverse information needs in recommendation. • We design a flexible, general instruction form and automatically generate a large number of high-quality instruction data. Further, we fine-tune a 3B language model specially for recommender systems. • Extensive experiments demonstrate the effectiveness and generalization ability of our approach on various task scenarios in recommender systems.
# 2 METHODOLOGY
In this section, we present the proposed instruction tuning approach for recommender systems, named InstructRec. Our approach allows users to freely express their information needs in natural language instructions when interacting with the recommender system. To develop our approach, we first design a specific instruction format
1There is no consensus on the minimum size that discriminates between small-sized language models and LLMs. Our approach can be naturally extended to a larger language model. We refer to the readers to the survey [44] for a comprehensive review of LLMs.
for recommendation, and formally introduce three key aspects (i.e., preference, intention, and task form) in instructions (Section 2.1). Next, we introduce how to construct the instruction data according to various instantiations with different preference, intention, and task form assisted by LLM, along with several strategies to increase the diversity of our instructions (Section 2.2). Finally, we discuss how to fine-tune the base LLM with these generated instruction data for more user-centric recommendation (Section 2.3).
# 2.1 Instruction Format for Recommendation
To enable a LLM with the ability to perform personalized recommendation tasks for users with instructions, the first step is to design a suitable instruction format, which aims to effectively reveal user’s vague/specific intention, provide user’s implicit/explicit preference, and clarify the task settings. In this part, we propose a unified instruction format containing three key aspects for recommendation, and then instantiate it for different interaction scenarios.
and then instantiate it for different interaction scenarios. 2.1.1 Key Aspects in Instructions. In order to design a flexible and expandable instruction format, we mainly consider three key aspects that are related to the expressions of user’s needs, namely preference, intention and task form. We present an illustration of the instruction format in Figure 2. In what follows, we introduce three aspects with their representative types in detail. Preference (P). Preference refers to user’s personalized tastes towards item attributes or characteristics. In our instruction format, it aims to capture inherent and long-term user preferences. Based on the degree of personalization, user preferences can be categorized into the following three types: • None (𝑃0). In this situation, the recommender system cannot obtain available information about user preferences or profiles, e.g., cold-start scenarios or under privacy concerns, which leads to non-personalized recommendations. • Implicit preference (𝑃1). It refers that context information about a user (e.g., user’s profiles or historical interaction records) is available, but without explicit exposure of the underlying preference over items. In existing literature, implicit interaction behaviors are often called implicit feedback [19, 22], which is easier to collect but may have noises. When formatting historical interaction records, we do not directly use item IDs to represent items as in P5 [13], but instead use item titles for formatting items as texts. • Explicit preference (𝑃2). In addition to the implicit preference, users may also directly reveal their preference in some cases. Such explicit feedback is more accurate but difficult to obtain. Here, we mainly consider the explicit expression of user preference in natural language text (e.g., user reviews). We do not directly consider explicit interaction records (e.g., ratings or likes), though they can be easily verbalized in our form. Intention (I). Compared to long-term preferences, users’ intentions refer to their more immediate demands for certain types of items, which may be different from user long-term preferences. Inspired by Su et al. [32], we categorize user intentions into three types according to the varying degree of clarity. • None (𝐼0). During the interaction, users may lack a clear goal for their next interactions and expect to discover potential interests through the recommender system and exploratory interactions.
2.1.1 Key Aspects in Instructions. In order to design a flexible and expandable instruction format, we mainly consider three key aspects that are related to the expressions of user’s needs, namely preference, intention and task form. We present an illustration of the instruction format in Figure 2. In what follows, we introduce three aspects with their representative types in detail.
Preference
Intention
Task Form
None
(Cold-start users)
None
(Exploratory interaction)
Pointwise 
(Discriminate a candidate)
Implicit
(User’s context info.)
Vague
(Ambiguous idea of target)
Pairwise
(Compare a item pair)
Explicit
(Explicit expression)
Specific
(Clear idea of target)
Matching & Reranking
(Retrieving the candidates,
refining the ranking)
# Figure 2: Various types of key aspects in instructions.
• Vague intention (𝐼1). In this case, users exhibit vague cognition about their needs, such as the descriptions of the desired characteristics of items, while can not clearly identify a specific item (e.g., “some gifts for my son”). • Specific intention (𝐼2). In this case, users have clear needs, requiring the system to prioritize their requests and provide suitable items that satisfy the specific needs (e.g., “blue, cheap, iPhone13”). Task Form (T). In addition to user preferences and intentions, another crucial aspect in formulating the instructions is the specified task form for execution by LLM. Similar to [7], we consider three potential forms for the recommendation tasks: • Pointwise recommendation (𝑇0). The LLM is instructed to examine whether a candidate item is appropriate for the user, where the matching relationship is determined based on user’s information needs and item characteristics. • Pairwise recommendation (𝑇1). In this case, the comparison between a pair of items is invoked and the LLM is instructed to select the more suitable item from this pair. • Matching (𝑇2). By acting as a matching (i.e., candidate generation) module, the LLM generates the potential candidates from the overall item corpus. These candidates are expected to be highquality resources, and have a good coverage of the target items. • Reranking (𝑇3). Unlike 𝑇2 that instructs LLM to act as a matching module, the LLM is employed as a reranker in this case. That is, the LLM is instructed to rerank the retrieved candidate items, enabling more accurate recommendations. Besides the above three parts, we can add other useful context information (e.g., time and place) about the user’s situation, called context. Overall, we can flexibly integrate these four parts in the form of natural language text.
• Vague intention (𝐼1). In this case, users exhibit vague cognition about their needs, such as the descriptions of the desired characteristics of items, while can not clearly identify a specific item (e.g., “some gifts for my son”). • Specific intention (𝐼2). In this case, users have clear needs, requiring the system to prioritize their requests and provide suitable items that satisfy the specific needs (e.g., “blue, cheap, iPhone13”).
2.1.2 Instantiation for Various Interaction Scenarios. In this part, we discuss how to instantiate the above instruction format for different real-world interaction scenarios. We present a list of examples in Table 1 about instruction instantiation. Next, we introduce several representative instantiations. • ⟨𝑃1/𝑃2, 𝐼0,𝑇3⟩. In this case, we combine 𝑃1 or 𝑃2 with 𝐼0 (without intention), focusing on user preferences. When performings these instructions, LLMs act as a traditional recommender. We can also combine both kinds of preferences, thereby instructing LLMs to perform recommendation with personalized prompts (𝑃2) [21, 39]. • ⟨𝑃0, 𝐼1/𝐼2,𝑇3⟩. We combine 𝑃0 with 𝐼1 or 𝐼2 to instantiate the second type of instructions. Specially, LLMs act as a traditional retriever to process users’ vague or specific queries [20].
Table 1: Example instructions with various types of user preferences , intentions , and task form ity, we make some modifications to the original instructions that are used in our experiments.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f775/f77532f6-a0df-4f24-b2c9-a779e90130d3.png" style="width: 50%;"></div>
Instantiation
Model Instructions
⟨𝑃1, 𝐼0,𝑇0⟩
The user has purchased these items: <historical interactions> . Based on this information, is it likely that the user will interact with <target item> next?
⟨𝑃2, 𝐼0,𝑇3⟩
You are a search engine and you meet a user’s query: <explicit preference> . Please respond to this user by selecting items from the candidates: <candidate items>.
⟨𝑃0, 𝐼1,𝑇2⟩
As a recommender system, your task is to recommend an item that is related to the user’s <vague intention> . Please provide your recommendation.
⟨𝑃0, 𝐼2,𝑇2⟩
Suppose you are a search engine, now the user search that <specific Intention> , can you generate the item to respond to user’s query?
⟨𝑃1, 𝑃2,𝑇2⟩
Here is the historical interactions of a user: <historical interactions> . His preferences are as follows: <explicit preference> . Please provide recommendations .
⟨𝑃1, 𝐼1,𝑇2⟩
The user has interacted with the following <historical interactions> . Now the user search for <vague intention> , please generate products that match his intent.
⟨𝑃1, 𝐼2,𝑇2⟩
The user has recently purchased the following <historical items>. The user has expressed a desire for <specific intention>. Please provide recommendations.
• ⟨𝑃1/𝑃2, 𝐼1/𝐼2,𝑇3⟩. We take a combination between user preferences (𝑃1/𝑃2) and intentions (𝐼1/𝐼2), aiming to rerank the candidate items. Such a task can be also considered as personalized search, where user preferences and intentions (queries) are available for satisfying user’s needs [1, 3]. With this instruction format, we can instantiate different interaction scenarios. Note that there is no clear boundary between recommendation and search in our setting, since both can be formulated in a similar way (see the discussion for ⟨𝑃1/𝑃2, 𝐼1/𝐼2,𝑇3⟩). In the above, we mainly discuss the task form with 𝑇3, since LLMs are currently more suitable for the reranking stage, due to the high cost of inference. In contrast, we can use other task forms such as 𝑇2 for training, which is a more difficult task setting. Improving the diversity of instructions is verified to be effective in our evaluation.
# 2.2 Instruction Generation
According to the introduced instruction format in Section 2.1, we next generate the instruction data by simulating user preferences and intentions based on available interaction data. However, it is difficult to obtain a large amount of data that explicitly reveal real preferences or intentions of users. Recently, some efforts have attempted the automatic prompting strategies (e.g., self-instruct [36]), which generates high-quality instructions by prompting an instructiontuned LLM (called teacher-LLM). Motivate by this, we employ a strong teacher-LLM (i.e., GPT-3.5) to generate such personalized information for each user by prompting with her/his historical interactions and reviews, which conveys rich information about preferences and intentions. In what follows, we first present the pipeline of instruction generation, and then evaluate the quality of the generated instructions.
2.2.1 Annotating the Aspects in Instructions. In order to better format the expressions of user needs via natural language instructions, we first manually create coarse-grained templates, according to the instantiations for various interaction scenarios (see Section 2.1.2). Subsequently, we further fill these templates with specific user preferences and intentions (referring to fine-grained user personalized instructions) that are extracted from interaction data or generated by the teacher-LLM. In what follows, we illustrate the construction process in detail.
Preference annotation. We use different strategies to generate the user preferences, considering the varying degrees of personalization. For the implicit preference 𝑃1, we take the title of an item
To enhance the readabil-
To enhance the readabil-
as its representation and utilize user’s historical interactions to fill in the instruction templates, which can be instantiated as “The user has previously purchased the following items: {[MASK]}”. While, for the explicit preference 𝑃2, since explicit descriptions of user preferences are usually not available in the dataset, the teacher-LLM (i.e., GPT-3.5) is employed to act as the user and generate explicit expressions of preferences based on the historical interactions. GPT3.5 shows a strong text understanding ability, and it can generate very meaningful expressions about users’ preferences or intentions based on the historical interaction behaviors. Here is an example depicting the generation of user preference by GPT-3.5:
[Raw Behavior Sequence]: “1. Resident Evil: Revelations 2 - PlayStation 4 →2. Resident Evil 4 - PlayStation 4 Standard Edition.” [Generated Explicit Preference]: “He prefers horrorbased games with a strong narrative.”
Intention annotation. Similarly, we can generate user intentions in different clarity degrees, including vague intention 𝐼1 and specific intention 𝐼2. To derive the vague intentions, since reviews provide valuable evidence about user’s personal tastes and the reason to make a specific interaction, we consider extracting the intentions from target reviews (the one associated with the target item). Specially, the teacher-LLM is employed to process these reviews and extract the intention. Here is an example depicting the extraction of user’s intention from review text:
[Raw Target Review]: “My son loves ... of the game. I’m happy I bought this for him.” [Generated Vague Intention]: “I enjoy buying games for my son that he enjoys.”
As for the specific intentions, users sometimes have a clear demand for certain items when using recommender systems (e.g., buy a video game on e-commerce platforms). As discussed in previous work [1, 3], the category of a target item provides an important evidence to reflect the user’s real intention. Consequently, we extract user’s specific intention from the category information of the target items, by concatenating multiple associated category labels into an intention expression:
[Generated Specific Intention]: “Video Games, PC, Accessories, Gaming Mice.”
Compared with the vague intention extracted from user’s historical interactions, the extracted intention is more specific, and can
directly reflect user’s real intention, for it is extracted from the information of target items.
Task form annotation. In this paper, we mainly focus on three types of task forms:𝑇0,𝑇2, and𝑇3. Specifically, for pointwise recommedation 𝑇0, We formulate the instruction as: “Based on the <user related information>, is it likely that the user will interact with <target item> next?”, and the system should respond with “Yes” or “No”. For the task of matching 𝑇2, the instruction is formulated as “Predict the next possible item”. While for the task of reranking 𝑇3, a list of candidates is incorporated into the instruction: “Select one item from the following <candidates>”. Although currently we do not include the pairwise recommendation 𝑇1, it can be easily integrated into our proposed framework. 2.2.2 Increasing the Diversity of Instructions. Recent studies [5, 37] have demonstrated the effectiveness of increasing the quantity and diversity of instructions. Therefore, in order to further improve the performance of instruction tuning, we propose to use several strategies to increase the diversity of instruction data. Turn the task around. This strategy refers to the swap between the input and output of normal instructions [37]. In our case, we request LLM to not only recommend the appropriate items based on user preferences or intentions, but also infer their potential information needs based on the feedback of recommendation. Such tasks can help LLM understand the relationship between user behaviors and the underlying information needs. An example of instruction and its reversed version is shown below: [Normal Instruction]: “The user searches that <Query>, can you generate the item to respond to his query?” [Swapped Instruction]: “The user wants to buy: <Target item>, but he doesn’t know how to formulate the query, please help him generate it.”
# Enforcing the relatedness between preference and intention. It refers that the revealed short-term user intentions with long-term preferences should be highly related in an instruction.
[Intention →Preference]: “The user has searched for <Query> and selected the product <Target item>. Based on his query and choice, please infer his historical interactions.” [Preference →Intention]: “The user has the following <historical interactions>, you can infer their preferences. Please make a prediction about the user’s next query and the item he is likely to purchase, based on his preference.”
Chain-of-thought (CoT) like reasoning. This strategy adds additional explanations at intermediate reasoning steps, that enables LLM to perform complex reasoning tasks [38]. To apply this idea to our tasks, it refers to the reasoning process from user’s implicit behaviors to explicit preferences or intentions that lead to the final recommendations:
[CoT Instruction]: “Given the <Historical Interactions> of the user, please infer his preference and recommend suitable items to him.” [Desired Responses]: “According to the user’s historical
<div style="text-align: center;">Table 2: Statistics of the annotated instructions.</div>
Table 2: Statistics of the annotated instructions.
Statistic
# of fine-grained instructions
252,730
- # of user-described preferences
151,638
- # of user intention in decision making
101,092
ave. instruction length (in words)
23.5
# of coarse-grained instructions
39
- # of preferences related instructions
17
- # of intentions related instructions
9
- # of combined instructions
13
ave. instruction length (in words)
41.4
<div style="text-align: center;">Table 3: Quality evaluation of the generated fine-grained instructions by taking text-davinci-003 as teacher-LLM.</div>
Quality Review Question
Preference
Intention
Is the instruction generated from
the user’s related information?
93%
90%
Does the teacher-LLM provide
related world knowledge?
87%
22%
Does the instruction reflect
the user’s preference/ intention?
88%
69%
Is the instruction related to
target item?
48%
69%
interactions, we can infer his <preferences>. Finally, we recommend him <target item>.”
2.2.3 Statistics and Quality of Instruction Data. Based on the above steps, we first manually design 39 coarse-grained instruction templates ( see a complete list of instruction templates in the Appendix), covering a diverse range of different interaction scenarios. Then we generate a total of 252K fine-grained user instructions, which describe user various types of preferences and intentions, with the help of the teacher-LLM (i.e., GPT-3.5). The statistics of the generated instructions are summarized in Table 2. To assess the quality of these fine-grained instructions, we randomly sample 100 instances from each of the instructions describing user preferences and intentions for evaluation. An annotator is asked to evaluate the appropriateness of the generated instructions based on their consistencies with the user data (e.g., historical interactions, target item and the associated review). The results are illustrated in Table 3. In most cases, the teacher-LLM can generate appropriate instructions based on user’s specific information. Not limited to the original information, it can also produce more detailed instructions by leveraging the encoded world knowledge. Currently, only 69% of the instructions accurately align with the user’s current intention, and we speculate that it is affected by the contained noise in the review text. We will explore how to generate more accurate intentions as the future work.
# 2.3 Instruction Tuning for Recommendations
Based on the above instruction format, we then discuss how to instruction tune LLMs for recommender systems. We first present the backbone LLM, and then introduce the optimization and inference.
2.3.1 The Backbone LLM. In this work, we aim to develop a recommender model capable of following user-centric instructions that specify their needs. Inspired by the recent progress on LLMs [16, 28, 34, 44], our approach is based on the fact that LLMs exhibit strong abilities in following user’s instructions to solve various tasks. Using LLMs as the recommender, we treat recommendation as a specific task for instruction following. It has been shown that instruction tuning enables LLMs to generalize to unseen tasks described in natural language instruction [27, 37]. Thus, such an approach can be potentially extended to fulfill more diverse tasks in application systems. While, we limit our discussion to the recommendation task. Specially, we adopt the 3B Flan-T5-XL [5] as the backbone model. Since Flan-T5 has been fine-tuned based on T5 [29] with a large amount of instruction data, it has an excellent capacity to follow natural language instructions. However, the original instructions for tuning Flan-T5 are not specially designed for recommender systems, which cannot effectively instruct the model to perform the recommendation task. To be specific, Flan-T5 is designed with an encoder-decoder architecture, supporting a maximum context length of 512 tokens. In our approach, we represent items using their associated texts (e.g., titles), which might result in an excessive input length when formatting users’ behavioral sequences. In such cases, we have to truncate the input sequence and likely result in suboptimal performance. Researchers can alternatively utilize LLMs with a longer context length, such as LLaMA [34], to model long behavioral sequences.
2.3.2 Training and Inference. In this part, we discuss how to optimize our base LLM with the generated instructions and how to use it for solving the recommendation tasks.
Optimization via Instruction Tuning. With the generated instruction data, we can optimize the LLM via instruction tuning, which is essentially a supervised fine-tuning way. Specifically, we first annotate the desired system responses (target output), according to different types of instructions. For example, when instructing the model to predict the next item, the target output is annotated as the target item. While, for CoT like instructions, the target output is annotated as the user’s reasoning process for the specific interaction. Since both the instruction and the target output can be formatted in natural language, we can unify the training into a unified sequence-to-sequence way. Formally, we optimize the negative log-likelihood of the target output as follows:
(1)
where 𝑌𝑘is the desired system responses for the 𝑘-th instance, 𝐼𝑘 is the instruction of the 𝑘-th instance, and 𝐵is the batch size. Inference. Via instruction tuning, the LLM has been trained to follow the instructions in various interaction scenarios. In this part, we present the application of our approach in recommender systems. Considering the computational efficiency and model capacity,
# Table 4: Comparison between the proposed InstructRec and two related studies. “IT” denotes instruction tuning.
Methods
Backbone
IT
Non-ID
Key point
M6-Rec
300M M6
�
�
Representing user behavior as plain texts
P5
220M T5-base
�
�
Task description
InstructRec
3B Flan-T5-XL
�
�
Aligning LLMs with user needs
we employ the LLM as a reranker to generate the final ranking for the candidates based on users’ instructions. In real-world systems, users’ needs are very diverse, and we expect that instructions can effectively capture different preferences or intentions by leveraging the generality of natural language. Specifically, when providing the recommendation services to users, the system will firstly select appropriate coarse-grained instruction templates based on user instructions (i.e., the instructions that a user issues) and other useful information (e.g., historical interactions). Then, we convert the original expressions into model instructions by using the operations such as concatenation, insertion, and persona shift. Then, the recommender (i.e., LLM) is requested to execute the model instruction that specifies user’s needs. However, due to the inherently stochastic nature of the generation process in LLMs, there exists a potential risk of generating items outside of the candidate set, especially when using beam search to generate a list of candidate items. To circumvent this problem, similar to Eq. (1), we directly feed the candidate items as input to the decoder of the model and calculate their likelihood for determining the final ranking for recommendations.
# 2.4 Discussion
In this part, we compare the proposed InstructRec with related methods, thereby highlighting the contributions of our approach.
Traditional methods such as SASRec [21] and LightGCN [14] typically rely on unique identifiers to represent users and items, and construct specific preference functions for recommendations. However, they often struggle to address the cold-start problem, as new users or items cannot be well represented. Additionally, traditional recommendation approaches focus on passive acceptance of recommendations by users, potentially failing to capture the true preferences accurately. While user can actively request a search engine to retrieve relevant items [20, 31], traditional search engines can only handle specific user queries due to their limited capacities. As a comparison, our model formulates the recommendation task using natural language and leverages universal knowledge in LLM to make the recommendations, having a better generalization ability. Furthermore, by fine-tuning on instructions with varying types of user preferences, intentions and task forms, our model can accommodate diverse user’s needs effectively.
Existing applications of LLMs in recommender systems such as P5 [13] and M6-Rec [6] consider behavior modeling as language modeling, where recommendation tasks are formulated as natural language expressions. M6-Rec reduces the reliance on user behavioral data by incorporating and training on a wide range of contextual information via converting various kinds of context information into a sequential text format, thereby achieving strong generalization across different interaction behaviors. P5 formulates
multiple recommendation tasks with instruction-based prompts and demonstrates strong performance on multiple tasks. However, they mainly focus on task-specific formulations, while neglect the exploration of aligning LLMs with more diverse, detailed user needs in practical scenarios. In contrast, our approach designs and generates recommendation-oriented instructions containing different types of preferences, intentions, and task forms, which can enhance the personalization in various interaction scenarios. The comparison of our approach and the two related studies is presented in Table 4.
# 3 EXPERIMENTS
In this section, we conduct experiments to evaluate the ability of our proposed InstructRec in several aspects.
# 3.1 Experimental Setup
3.1.1 Datasets. We evaluate our model’s performance on accommodating user-centric instructions with the “Video Games” subset of Amazon2 dataset and its generalization to unseen data with “CDs & Vinyl” subset of Amazon dataset respectively. Following previous work [18], we filter unpopular users and items with fewer than five interactions for all datasets. Since Flan-T5 (our backbone) has a contextual limit of 512 tokens, we truncate the generated behavioral sequence with a maximum of 20 items. The statistics of our preprocessed datasets are summarized in Table 5.
# Table 5: Statistics of the datasets after preprocessing “Avg.𝑙𝑒𝑛” represents the average length of item sequences.
<div style="text-align: center;">Table 5: Statistics of the datasets after preprocessing. “Avg.𝑙𝑒𝑛” represents the average length of item sequences.</div>
Dataset
#Users
#Items
#Inters
#Sparsity
Avg.𝑙𝑒𝑛
Games
50,546
16,859
410,907
99.952%
8.13
CDs
93,652
63,929
871,883
99.985%
9.31
3.1.2 Evaluation Metrics. We adopt top-𝐾hit ratio (HR) and top-𝐾 normalized discounted cumulative gain (NDCG) to evaluate the performance. In this paper, K is set to 1, 3 and 5. For interaction scenarios such as sequential recommendation and personalized search, following the previous work [3, 18, 45], we apply the leaveone-out strategy for evaluation. Concretely, for each user interaction sequence, the last item and its related review are used as the test data, the item and review before the last interaction are used as the validation data, and the remaining interaction records are used for training. For interaction scenarios such as product search, we split all the items and their related queries into training (80%), validation (10%) and testing (10%). In this setting, instances of validation and testing set are unseen in the training stage, which increases the difficulty of inference. We implement some baselines with a popular open-source recommendation library RecBole [40, 42, 43]. Notably, for different types of coarse-grained instructions, we evaluate the results of instruction that obtains the best performance in the valid set. By taking the model as a reranker, we rank the ground-truth of each instance among nine randomly sampled negative items, for evaluation on the test set (we further evaluate on harder settings in Section 3.3), and finally report the average score of all test instances.
2https://nijianmo.github.io/amazon/index.html
# 3.2 Overall Performance on Various User Information Needs
To demonstrate the effectiveness of our model in accommodating diverse user instructions, we conduct in-depth experiments according to the instantiations of these instructions for various interaction scenarios (see instantiations in Section 2.1.2). In what follows, we introduce these experiments in different scenarios separately, including the baselines and the performance comparisons. Notably, in addition to introducing baselines that specialized in specific interaction scenarios, we further take text-davinci-003 3, which is one of the universal GPT-3.5 model, as a strong LLM baseline for comparison. Notably, due to the high cost of this model, we random sample 500 instances to evaluate its performance. Although this may inevitably introduce some randomness, we believe that the conclusions obtained in this setting are still referable for exploring the recommendation capabilities of a universal LLM.
3.2.1 Sequential Recommendation ⟨𝑃1, 𝐼0,𝑇3⟩. We first conduct an evaluation of our model’s performance in the classical sequential recommendation task by formulating the instructions with user’s implicit preference 𝑃1 (e.g., behavioral sequences).
Baseline. We adopt SASRec [21] and BERT4Rec [33] as our baselines in the scenario of sequential recommendation. SASRec is a representative sequential recommendation model that utilizes a transformer-encoder architecture, incorporating a multi-head selfattention mechanism to effectively capture sequential patterns in the data. BERT4Rec is also widely used, which incorporates bidirectional self-attention mechanisms, leveraging a cloze objective to effectively encode sequential data.
# Table 6: Performance on sequential recommendation.
<div style="text-align: center;">Table 6: Performance on sequential recommendation.</div>
Methods
⟨𝑷1, 𝑰0, 𝑻3⟩
HR@1
HR@3
HR@5
NDCG@3
NDCG@5
BERT4Rec
0.5747
0.8188
0.9083
0.7176
0.7546
SASRec
0.6663
0.8741
0.9389
0.7887
0.8155
GPT-3.5
0.3640
0.6300
0.7300
0.5216
0.5623
InstructRec
0.6947
0.8793
0.9429
0.8033
0.8295
Improv.
+4.26%
+0.59%
+0.43%
+1.85%
+1.72%
Performance comparison. For sequential recommendation, users do not actively express their information needs to the system, which poses certain limitations on our model’s capabilities. Nevertheless, through finetuning on large amounts of behavioral data and leveraging the generalization of instructions, as demonstrated in Table 6, our model outperforms other baselines. Furthermore, we find that GPT-3.5 does not exhibit satisfactory performance in this particular scenario. This can be attributed to the mismatch between the universal textual nature of LLM and the specificity of behavioral information in private domain. In contrast to natural language, user behavioral sequences are highly personalized and exhibit greater complexity, posing a challenge for universal LLMs to capture individual behavioral patterns. Therefore, in order to deploy LLMs in
3https://platform.openai.com/docs/models/overview
# Table 7: Performance on personalized search. We have evaluated on three types of queries built from 𝑃2, 𝐼1, and 𝐼2.
Methods
⟨𝑷1, 𝑷2, 𝑻3⟩
⟨𝑷1, 𝑰1, 𝑻3⟩
⟨𝑷1, 𝑰2, 𝑻3⟩
HR@1
HR@3
HR@5
NDCG@3
NDCG@5
HR@1
HR@3
HR@5
NDCG@3
NDCG@5
HR@1
HR@3
HR@5
NDCG@3
NDCG@5
TEM
0.5005
0.7368
0.8462
0.6383
0.6833
0.5723
0.7860
0.8712
0.6974
0.7325
0.8665
0.9149
0.9293
0.8954
0.9013
GPT-3.5
0.2740
0.5500
0.7000
0.4366
0.4979
0.3660
0.6360
0.7566
0.5256
0.5795
0.4580
0.7240
0.8100
0.6138
0.6499
InstructRec
0.6959
0.8806
0.9452
0.8045
0.8312
0.8278
0.9444
0.9741
0.8971
0.9094
0.9269
0.9868
0.9951
0.9631
0.9665
Improv.
+39.04%
+19.52%
+11.70%
+26.04%
+21.64%
+44.64%
+20.15%
+11.81%
+28.63%
+24.15%
+6.97%
+7.86%
+7.08%
+7.56%
+7.23%
private domain recommender systems, a proper fine-tuning rather than simply relying on LLMs under the zero-shot setting may be crucial for capturing personalized user behaviors.
3.2.2 Product Search ⟨𝑃0, 𝐼2,𝑇3⟩. In this part, we evaluate the efficacy of our model on product search. A typical product search task takes a search query from users and then recommends items that are related to the query and will be clicked by the user. Here, the search query is simulated by user specific intention 𝐼2 extracted from target items’ metadata, such as target item categories.
Baseline. We take DSSM [20] as our baseline. DSSM is a widelyverified retrieval model that employs a two-tower architecture, aiming to establish a semantic-level mapping between a given user query and relevant documents. In our implementation, we leverage BERT-mini [8], which is known for robust encoding capabilities, to represent and encode the query and document inputs.
<div style="text-align: center;">Table 8: Performance on product search.</div>
Methods
⟨𝑷0, 𝑰1, 𝑻3⟩
HR@1
HR@3
HR@5
NDCG@3
NDCG@5
DSSM
0.7279
0.9484
0.9899
0.8587
0.8760
GPT-3.5
0.6700
0.9140
0.9480
0.8177
0.8399
InstructRec
0.8263
0.9411
0.9728
0.8944
0.9075
Improv.
+13.52%
–
–
+4.16%
+3.60%
Performance comparison. For the evaluation of product search, wherein the instructions are relatively specific (simulated from item category), the traditional model performs well. Furthermore, since the items in the test set would be unseen during the training phase, this evaluation necessitates a higher degree of generalization capacity. Therefore, as we can see in Table 8, our model achieves superior or comparable performance in most cases (especially for top ranking metrics such as NDCG@1) due to the vast amount of general knowledge encoded in its parameters. This conclusion can also be supported by the good results of GPT-3.5 without tuning.
Performance comparison. For the evaluation of product search, wherein the instructions are relatively specific (simulated from item category), the traditional model performs well. Furthermore, since the items in the test set would be unseen during the training phase, this evaluation necessitates a higher degree of generalization capacity. Therefore, as we can see in Table 8, our model achieves superior or comparable performance in most cases (especially for top ranking metrics such as NDCG@1) due to the vast amount of general knowledge encoded in its parameters. This conclusion can also be supported by the good results of GPT-3.5 without tuning. 3.2.3 Personalized search ⟨𝑃1, 𝑃2/𝐼1/𝐼2,𝑇3⟩. In this part, we evaluate the model’s capacity in personalized search, which could be viewed as a natural combination of recommendation and search tasks. Notably, in the literature of personalized search [1, 10], there are various ways to inject personalization into traditional search engines, such as introducing user’s historical requests or clicks in logs. In this paper, following previous work [1, 3], we conduct experiments in the latter setting. Specifically, we leverage the historical behavior sequences (𝑃1) for the “personalized” part. For the
3.2.3 Personalized search ⟨𝑃1, 𝑃2/𝐼1/𝐼2,𝑇3⟩. In this part, we evaluate the model’s capacity in personalized search, which could be viewed as a natural combination of recommendation and search tasks. Notably, in the literature of personalized search [1, 10], there are various ways to inject personalization into traditional search engines, such as introducing user’s historical requests or clicks in logs. In this paper, following previous work [1, 3], we conduct experiments in the latter setting. Specifically, we leverage the historical behavior sequences (𝑃1) for the “personalized” part. For the
“search” part, we comprehensively adopt the user explicit preference simulated by LLMs (𝑃2), vague intention (𝐼1), and specific intention (𝐼2) as three types of queries respectively.
Baseline. We introduce TEM [3] as our baseline in this scenario. As a representative approach in personalized product search, TEM utilizes a transformer architecture to encode the sequences of query and user’s behavioral sequence, thereby achieving dynamic control over the impact of personalization on the search results.
Performance comparison. The performance evaluation of our model in accommodating personalized search is presented in Table 7. We observe that: (1) in general, our InstructRec outperforms other approaches by a large margin in almost all the cases. Specifically, when taking user specific intention (i.e., item category) as instruction, despite the provision of additional supplemental information such as user behavioral data, GPT-3.5 exhibits worse performance in comparison to its effectiveness in conducting product search. This observation emphasizes the huge gap between user behavioral patterns of private domain data and the universal semantic knowledge encoded within LLMs. While leveraging the technique of instruction tuning, our model, which is also built upon a universal LLM, effectively bridges the gap and aligns itself with user’s personalized behaviors within recommender systems. (2) In scenarios where user instructions exhibit relative ambiguity, such as LLM-generated explicit preference and vague intention, traditional models tend to perform poorly. This can be attributed to the incapacity of traditional models to capture the underlying vague information needs of users conveyed in these instructions. (3) Furthermore, although explicit user preferences are typically simulated by analyzing historical interactions and most of them are reflective of user’s real preferences, there are still observed mismatches between user’s long-term mainstream preferences (𝑃2) and the current intentions of target items in the test set (see the quality evaluation of instructions in Table 3). Therefore, the results of ⟨𝑃1, 𝑃2,𝑇3⟩are much worse than those of ⟨𝑃1, 𝐼2,𝑇3⟩whose queries are more related to user’s real intentions. Nevertheless, the introduction of instruction tuning empowers our model with reasoning and generalization capabilities, thereby enabling it to perform complex interaction scenarios involving ambiguous instructions. To sum up, through instruction tuning, our model effectively integrates user behaviors in private domain with universal knowledge. Consequently, it achieves the best performance in almost all classical practical tasks including sequential recommendation, product search, and personalized search, regardless of whether the user’s preferences are implicit or explicit, and whether the intention is vague or specific.
endation as Instruction Following: A Large Language Model Empowered Recommendation Approach
# 3.3 Further Analyses
3.3.1 Discriminating Hard Negative Item Candidates. As stated, we aim to employ our model as a reranker in recommender systems. Previous experiments demonstrate its effectiveness in reranking a list of randomly sampled candidate items. In order to evaluate the model’s ability in reranking more practical candidate items retrieved by a strong matching module, which are more challenging to distinguish compared to random negative items, we simulate the real matching-then-reranking pipeline of recommendation. To this end, we introduce a matching module to retrieve a list of candidate items from all the items, then our model is instructed to rerank these candidates in the scenario of sequential recommendation. Specifically, we adopt the classical two-tower model as the matching module. The user tower incorporates the information of user ID, behavioral sequences’ ID (encoded by a 2-layer transformer encoder), and behavioral sequences’ titles (encoded by BERT). The item tower incorporates the information of item ID and item title (encoded by BERT). Therefore, the module considers both sequential patterns and textual similarities to retrieve candidate items. In this experiment, we retrieve nine negative candidate items with the positive target item. The results of our model and other baselines are reported in Table 9. Our model still exhibits superior performance than other baselines with a notable gap when reranking the hard negative candidates. The result indicates that our model has a strong ability to discriminate and select items that are more in line with user information needs among similar items.
# Table 9: Performance on reranking hard negative candidates. We test it in scenario of sequential recommendation.
Methods
Matching Module: ⟨𝑷0, 𝑰1, 𝑻3⟩
HR@1
HR@3
HR@5
NDCG@3
NDCG@5
SASRec
0.0355
0.3448
0.5536
0.2127
0.2984
GPT-3.5
0.1100
0.3480
0.5020
0.2481
0.3113
InstructRec
0.1841
0.4823
0.6648
0.3558
0.4307
Improv.
+67.36%
+38.59%
+20.09%
+43.41%
+38.36%
3.3.2 Discriminating More Candidate Items. In previous experiments, we verify the effectiveness of our model in accommodating diverse interaction scenarios and hard negative samples by ranking a list of ten candidate items. However, in realistic recommender systems, it is commonplace to encounter a considerably larger pool of items (often numbering in the hundreds), which cater to user preferences. To evaluate the discriminative capabilities of our model across a broader range of candidate items, we simulate a rerank scenario for personalized search. This involved the random sampling of 99 negative items and the target item, resulting in a collection of 100 candidate items. Notably, as aforementioned, our model faces limitations in processing these candidate items concurrently due to the restricted context length. In order to complete this evaluation, we adopt a straightforward approach for prospective testing purposes. Specifically, we randomly divide the one hundred candidate items into ten equal groups and instruct the model to select the most potential item from each group. The resulting ten items would be then reranked, leading to the final ranking result of our model.
As demonstrated in Table 10, our model exhibits a considerable performance advantage over the traditional baseline. We also find that the improvement observed in the HR@1 metric is relatively less significant compared to other metrics. This could potentially be attributed to the increased difficulty in distinguishing among the ten selected items. Consequently, besides employing LLM with longer maximum context lengths, the result also inspires us to explore efficient algorithms that can handle a larger number of candidates within the constraints of limited context length, which will be studied in future work. Nevertheless, we believe that our InstructRec is more suitable for being deployed in the reranking stage (since it is the closest stage of recommender systems that users communicate with), and existing matching models are capable of retrieving good candidates in practice.
Table 10: Performance on discriminating one hundred of candidate items. We test it in scenario of personalized search.
Methods
⟨𝑷1, 𝑰1, 𝑻3⟩
HR@1
HR@3
HR@5
NDCG@3
NDCG@5
TEM
0.2794
0.4484
0.5284
0.3777
0.4106
InstructRec
0.3276
0.7694
0.8334
0.5903
0.6163
Improv.
+17.25%
+71.59%
+57.72%
+56.29%
+50.10%
3.3.3 Effects of Instructions. In this part, we explore the question that how the diversity of instructions affects the model’s performance on held-out interaction scenario and evaluate its generalization. To do this, we set the scenario of personalized search with vague intention as a held-out scenario and continuously adding various types of fine-grained instructions for instruction tuning. Note that we refer to “vague intention∗” as another expression of user’s vague intention simulated from the target review. That is, since review contains both user preferences and item characteristics. We employ the teacher-LLM to analyze user’s vague intentions from the perspectives of both the item and the user, serving as training data for instruction tuning and testing data for evaluation. As we can see in Figure 3, with the increasing number of interaction scenarios for instruction tuning, the performance of the model is steadily improved in the held-out interaction scenario. This observation demonstrates the effectiveness of instruction tuning in generalizing across various scenarios. Furthermore, it is worth noting that the incorporation of “vague intention∗” leads to a large improvement of performance, inspiring us to annotate more diverse data by conducting self-instruct with various strategies.
3.3.4 Generalization across Datasets. Previous experiments have demonstrated the remarkable generalization capabilities of our model when it comes to accommodating unseen items, instructions and interaction scenarios. Despite the effectiveness, most of these evaluations focus on in-domain generalization. In this part, we aim to evaluate the model’s ability to generalize to unseen datasets, which would have distinct patterns to the source data. To this end, we evaluate the model’s ability to transfer from the “Games” dataset to the “CDs” dataset. The results are presented in Table 11.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6b5c/6b5c46ba-f75e-48a4-97f6-9d006cad203c.png" style="width: 50%;"></div>
Figure 3: Instruction tuning on more interaction scenarios increases the performance on the held-out scenario. We take personalized search with vague intention as the held-out scenario.
<div style="text-align: center;">Figure 3: Instruction tuning on more interaction scenarios increases the performance on the held-out scenario. We take personalized search with vague intention as the held-out scenario.</div>
Table 11: Performance on generalizing to unseen dataset BERT4Rec and SASRec are trained on in-domain instances.
Methods
⟨𝑷1, 𝑰0, 𝑻3⟩
HR@1
HR@3
HR@5
NDCG@3
NDCG@5
BERT4Rec
0.5867
0.8226
0.9114
0.7249
0.7615
SASRec
0.6713
0.8754
0.9428
0.7915
0.8194
GPT-3.5
0.1380
0.3140
0.4780
0.2401
0.3067
Flan-T5-XL
0.0927
0.2886
0.4814
0.2035
0.2823
InstructRec𝐺𝑎𝑚𝑒𝑠
0.2332
0.4392
0.6052
0.3511
0.4190
In general, our model’s zero-shot performance in this setting still falls short compared to traditional sequential models that are trained on these in-domain instances. It is natural since in-domain behavioral information is essential in recommendation, which is implied in above evaluations. Nevertheless, Our model still outperforms other powerful LLMs that are good at zero-shot tasks by a large margin. It indicates that instruction tuning could bring in significant positive outcomes for our model, providing evidence that our approach can effectively capture universal knowledge across distinct domains.
# 4 CONCLUSION AND FUTURE WORK
In this paper, we proposed an instruction tuning approach of LLMs for recommender systems, named InstructRec. Different from existing studies that adapt LLMs for recommendation, our key idea is to consider recommendation as instruction following by LLMs, allowing users to freely express their information needs in natural language (called instructions). Specifically, we first designed the general instruction templates format by integrating the preference, intention, and task form, and context information of a user in natural language text. Then, we automatically generated 252K fine-grained user personalized instructions that describe user preferences and intentions. By tuning an open-source LLM (3B Flan-T5-XXL) with these instruction data, the base model can be well adapted to recommender systems, which can follow user’s instructions to perform
personalized recommendation. Extensive experiments have demonstrated the effectiveness and generalization ability of our approach in various scenarios. As future work, we will further scale the size of LLMs for instruction tuning, and also consider extending the context length for modeling long behavior sequences. Besides, we will consider applying the current approach in a multi-turn interaction scenario, where users can communicate with the systems via a chit-chat way.
# REFERENCES
[1] Qingyao Ai, Yongfeng Zhang, Keping Bi, Xu Chen, and W Bruce Croft. 2017. Learning a hierarchical embedding model for personalized product search. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval. 645–654. [2] Akari Asai, Timo Schick, Patrick Lewis, Xilun Chen, Gautier Izacard, Sebastian Riedel, Hannaneh Hajishirzi, and Wen-tau Yih. 2022. Task-aware retrieval with instructions. arXiv preprint arXiv:2211.09260 (2022). [3] Keping Bi, Qingyao Ai, and W Bruce Croft. 2020. A transformer-based embedding model for personalized product search. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 1521–1524. [4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901. [5] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022). [6] Zeyu Cui, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. 2022. M6Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems. arXiv preprint arXiv:2205.08084 (2022). [7] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering ChatGPT’s Capabilities in Recommender Systems. arXiv preprint arXiv:2305.02182 (2023). [8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In NAACL. [9] Hao Ding, Yifei Ma, Anoop Deoras, Yuyang Wang, and Hao Wang. 2021. ZeroShot Recommender Systems. arXiv preprint arXiv:2105.08318 (2021). [10] Zhicheng Dou, Ruihua Song, and Ji-Rong Wen. 2007. A large-scale evaluation and analysis of personalized search strategies. In Proceedings of the 16th international conference on World Wide Web. 581–590. [11] Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. 2023. Specializing Smaller Language Models towards Multi-Step Reasoning. arXiv preprint arXiv:2301.12726 (2023). [12] Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang. 2023. Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System. arXiv preprint arXiv:2303.14524 (2023). [13] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In RecSys. [14] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 639–648. [15] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based recommendations with recurrent neural networks. In ICLR. [16] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556 (2022). [17] Yupeng Hou, Zhankui He, Julian McAuley, and Wayne Xin Zhao. 2023. Learning vector-quantized item representation for transferable sequential recommenders. In Proceedings of the ACM Web Conference 2023. 1162–1171. [18] Yupeng Hou, Shanlei Mu, Wayne Xin Zhao, Yaliang Li, Bolin Ding, and Ji-Rong Wen. 2022. Towards Universal Sequence Representation Learning for Recommender Systems. In KDD. [19] Yifan Hu, Yehuda Koren, and Chris Volinsky. 2008. Collaborative filtering for implicit feedback datasets. In 2008 Eighth IEEE international conference on data mining. Ieee, 263–272. [20] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2333–2338.
[21] Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. In ICDM. [22] Diane Kelly and Jaime Teevan. 2003. Implicit feedback for inferring user preference: a bibliography. In Acm Sigir Forum, Vol. 37. ACM New York, NY, USA, 18–28. [23] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 42, 8 (2009), 30–37. [24] Greg Linden, Brent Smith, and Jeremy York. 2003. Amazon. com recommendations: Item-to-item collaborative filtering. IEEE Internet computing 7, 1 (2003), 76–80. [25] Junling Liu, Chao Liu, Renjie Lv, Kang Zhou, and Yan Zhang. 2023. Is ChatGPT a Good Recommender? A Preliminary Study. arXiv:2304.10149 [cs] [26] Peng Liu, Lemei Zhang, and Jon Atle Gulla. 2023. Pre-train, prompt and recommendation: A comprehensive survey of language modelling paradigm adaptations in recommender systems. arXiv preprint arXiv:2302.03735 (2023). [27] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems 35 (2022), 27730–27744. [28] Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. 2021. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446 (2021). [29] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research 21, 1 (2020), 5485–5551. [30] Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2001. Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th international conference on World Wide Web. 285–295. [31] Yelong Shen, Xiaodong He, Jianfeng Gao, Li Deng, and Grégoire Mesnil. 2014. Learning semantic representations using convolutional neural networks for web search. In Proceedings of the 23rd international conference on world wide web. 373–374. [32] Ning Su, Jiyin He, Yiqun Liu, Min Zhang, and Shaoping Ma. 2018. User intent, behaviour, and perceived satisfaction in product search. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. 547–555. [33] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In CIKM. 1441–1450. [34] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023). [35] Xiaolei Wang, Kun Zhou, Ji-Rong Wen, and Wayne Xin Zhao. 2022. Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1929–1937. [36] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. Self-Instruct: Aligning Language Model with Self Generated Instructions. arXiv preprint arXiv:2212.10560 (2022). [37] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652 (2021). [38] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903 (2022). [39] Yiqing Wu, Ruobing Xie, Yongchun Zhu, Fuzhen Zhuang, Xu Zhang, Leyu Lin, and Qing He. 2022. Personalized Prompts for Sequential Recommendation. arXiv preprint arXiv:2205.09666 (2022). [40] Lanling Xu, Zhen Tian, Gaowei Zhang, Lei Wang, Junjie Zhang, Bowen Zheng, Yifan Li, Yupeng Hou, Xingyu Pan, Yushuo Chen, et al. 2022. Recent Advances in RecBole: Extensions with more Practical Considerations. arXiv preprint arXiv:2211.15148 (2022). [41] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068 (2022). [42] Wayne Xin Zhao, Yupeng Hou, Xingyu Pan, Chen Yang, Zeyu Zhang, Zihan Lin, Jingsen Zhang, Shuqing Bian, Jiakai Tang, Wenqi Sun, et al. 2022. RecBole 2.0: Towards a More Up-to-Date Recommendation Library. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 4722–4726. [43] Wayne Xin Zhao, Shanlei Mu, Yupeng Hou, Zihan Lin, Yushuo Chen, Xingyu Pan, Kaiyuan Li, Yujie Lu, Hui Wang, Changxin Tian, et al. 2021. Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 4653–4664.
[44] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223 (2023). [45] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. In CIKM. 1893–1902.
# A INSTRUCTION TEMPLATES FOR TRADITIONAL RECOMMENDATION
# A INSTRUCTION TEMPLATES FOR
⟨𝑃1, 𝐼0,𝑇2⟩. Using the user’s historical interactions as input data, predict the next product that the user is most likely to interact with. The historical interactions are provided as follows: {historical interactions}. ⟨𝑃1, 𝑃2, 𝐼0,𝑇2⟩. Recommend the next potential product to a user based on his profile and past interaction. You have access to the user’s profile information, including his preference: explicit preference and past interactions: {historical interactions}. Now you need to determine what product would be recommended to him. Chain-of-thought (CoT) like reasoning. Here is some information about the user, such as his historical interactions: {historical interactions}. Based on this information, your task is to infer the user’s preference based on his historical interactions and recommend the next product to the user. ⟨𝑃1, (𝑃2), 𝐼0,𝑇2⟩. Given the following historical interaction of the user: {historical interactions}. You can infer the user’s preference. {explicit preference}. Please predict next possible item for the user. ⟨𝑃1, (𝑃2), 𝐼0,𝑇2⟩. Based on the historical interactions shown below: {historical interactions}, you can analyze the common ground among these interactions to determine the user’s preferences {explicit preference}. Then please recommend a suitable item to the user. ⟨𝑃1, (𝑃2), 𝐼0,𝑇2⟩. To make a recommendation for this user, we need to analyze their historical interactions, which are shown below: {historical interactions}. As we know, historical interactions can reflect the user’s preferences. Based on this user’s preferences {explicit preference}, please recommend an item that you think would be suitable for them. ⟨𝑃1, (𝑃2), 𝐼0,𝑇3⟩. The behavioral sequence of the user is shown below: {historical interactions}, which can be used to infer the user’s preferences {explicit preference}. Then please select the item that is likely to be interacted with the user from the following candidates, by comparing the candidates and their similarities to the user’s preference. The candidates are: {candidate items} ⟨𝑃1, (𝑃2), 𝐼0,𝑇3⟩. You have observed that the user has clicked on the following items: {historical interactions}, indicating his personal tastes: {explicit preference} Based on this information, please select one item from the following options that you think would be suitable for the user: {candidate items} ⟨𝑃1, (𝑃2), 𝐼0,𝑇3⟩. You have some information about this user, which is shown below: {explicit preference}, the user’s historical interactions: {historical interactions} Based on this information, please recommend the next possible item for the user, which should match the user’s preference, from the following candidates: {candidate items} ⟨𝑃1, (𝑃2), 𝐼0,𝑇3⟩. You have obtained the user’s historical interaction list, which is as follows:{historical interactions}. Based on this history, you can infer the user’s preferences {explicit preference}. Now, you need to select the next product to recommend to the user. Please choose one from the following candidates: {candidate items}
⟨𝑃1, 𝐼0,𝑇2⟩. Using the user’s historical interactions as input data, predict the next product that the user is most likely to interact with. The historical interactions are provided as follows: {historical interactions}. ⟨𝑃1, 𝑃2, 𝐼0,𝑇2⟩. Recommend the next potential product to a user based on his profile and past interaction. You have access to the user’s profile information, including his preference: explicit preference and past interactions: {historical interactions}. Now you need to determine what product would be recommended to him. Chain-of-thought (CoT) like reasoning. Here is some information about the user, such as his historical interactions: {historical interactions}. Based on this information, your task is to infer the user’s preference based on his historical interactions and recommend the next product to the user. ⟨𝑃1, (𝑃2), 𝐼0,𝑇2⟩. Given the following historical interaction of the user: {historical interactions}. You can infer the user’s preference. {explicit preference}. Please predict next possible item for the user. ⟨𝑃1, (𝑃2), 𝐼0,𝑇2⟩. Based on the historical interactions shown below: {historical interactions}, you can analyze the common ground among these interactions to determine the user’s preferences {explicit preference}. Then please recommend a suitable item to the user. ⟨𝑃1, (𝑃2), 𝐼0,𝑇2⟩. To make a recommendation for this user, we need to analyze their historical interactions, which are shown below: {historical interactions}. As we know, historical interactions can reflect the user’s preferences. Based on this user’s preferences {explicit preference}, please recommend an item that you think would be suitable for them. ⟨𝑃1, (𝑃2), 𝐼0,𝑇3⟩. The behavioral sequence of the user is shown below: {historical interactions}, which can be used to infer the user’s preferences {explicit preference}. Then please select the item that is likely to be interacted with the user from the following candidates, by comparing the candidates and their similarities to the user’s preference. The candidates are: {candidate items} ⟨𝑃1, (𝑃2), 𝐼0,𝑇3⟩. You have observed that the user has clicked on the following items: {historical interactions}, indicating his personal tastes: {explicit preference} Based on this information, please select one item from the following options that you think would be suitable for the user: {candidate items} ⟨𝑃1, (𝑃2), 𝐼0,𝑇3⟩. You have some information about this user, which is shown below: {explicit preference}, the user’s historical interactions: {historical interactions} Based on this information, please recommend the next possible item for the user, which should match the user’s preference, from the following candidates: {candidate items} ⟨𝑃1, (𝑃2), 𝐼0,𝑇3⟩. You have obtained the user’s historical interaction list, which is as follows:{historical interactions}. Based on this history, you can infer the user’s preferences {explicit preference}. Now, you need to select the next product to recommend to the user. Please choose one from the following candidates: {candidate items}
⟨𝑃1, (𝑃2), 𝐼0,𝑇1⟩. The user has previously purchased the following items: {historical interactions}. This information indicates their personalized preferences {explicit preference}. Based on this information, is it likely that the user will interact with {candidate item} next? ⟨𝑃1, (𝑃2), 𝐼0,𝑇1⟩. Based on the user’s historical interaction list, which is provided as follows: {historical interactions} , you can infer the user’s personalized preference {explicit preference}. And your task is to use this information to predict whether the user will click on {candidate item} next. ⟨𝑃1, (𝑃2), 𝐼0,𝑇3⟩. Please recommend an item to the user based on the following information about the user: {historical interactions} ,the user’s historical interaction, which is as follows: {explicit preference} Try to select one item from the following candidates that is consistent with the user’s preference: {candidate item}. Turn the task around. You have the ability to infer a user’s preferences based on his past interactions. You are provided with a list of the user’s past interactions : {historical interactions} Your task is to analyze the commonalities among the past interactions and infer his overall preference. Please provide your analysis and inference. Turn the task around. As we all know, the user’s historical interactions are guided by his personalized preference. Try to infer the user’s preferences by analyzing his historical interactions : {historical interactions} ⟨𝑃2, 𝐼0,𝑇2⟩. You are a recommender system, and are good at recommending products to a user based on his preferences. Given the user’s preferences: {explicit preference}, please recommend products that are consistent with those preferences. ⟨𝑃2, 𝐼0,𝑇2⟩. As we know, a user’s behavior is driven by his preferences, which determine what they are likely to buy next. Your task is to predict what products a user will purchase next, based on his preferences. Given the user’s preferences as follows: {explicit preference}, please make your prediction.
# B INSTRUCTION TEMPLATES FOR TRADITIONAL PRODUCT SEARCH
⟨𝑃0, 𝑃2/𝐼1/𝐼2,𝑇2⟩. Suppose you are a search engine, now the user search that {explicit preference/ vague intention/ specific intention}, can you generate the item to respond to user’s query? ⟨𝑃0, 𝐼2,𝑇2⟩. As a search engine, your task is to answer the user’s query by generating a related item. The user’s query is provided as {specific intention}. Please provide your generated item as your answer. ⟨𝑃0, 𝐼2,𝑇2⟩. As a recommender system, your task is to recommend an item that is related to the user’s request, which is specified as follows: {specific intention} Please provide your recommendation. ⟨𝑃0, 𝑃2/𝐼1/𝐼2,𝑇2⟩. If a user asks a question like: {explicit preference/ vague intention/ specific intention} Please generate a related answer to help him. Turn the task around. If a user wants to search for something specific in a search engine but doesn’t know how to phrase the query, we can help generate the query for them. Now the user wants to search for {target item}. Please generate the query.
⟨𝑃0, 𝑃2/𝐼1/𝐼2,𝑇2⟩. Suppose you are a search engine, now the user search that {explicit preference/ vague intention/ specific intention}, can you generate the item to respond to user’s query? ⟨𝑃0, 𝐼2,𝑇2⟩. As a search engine, your task is to answer the user’s query by generating a related item. The user’s query is provided as {specific intention}. Please provide your generated item as your answer. ⟨𝑃0, 𝐼2,𝑇2⟩. As a recommender system, your task is to recommend an item that is related to the user’s request, which is specified as follows: {specific intention} Please provide your recommendation. ⟨𝑃0, 𝑃2/𝐼1/𝐼2,𝑇2⟩. If a user asks a question like: {explicit preference/ vague intention/ specific intention} Please generate a related answer to help him. Turn the task around. If a user wants to search for something specific in a search engine but doesn’t know how to phrase the query, we can help generate the query for them. Now the user wants to search for {target item}. Please generate the query.
Turn the task around. As a search engine that has seen many user queries, you can make an educated guess about how a user might write a query when searching for a particular item. If a user were searching for the item: {target item} They might use keywords related to the item such as its brand, or type. So the query would be. ⟨𝑃0, 𝑃2/𝐼1/𝐼2,𝑇3⟩. You are a search engine and you meet a user’s query {explicit preference/ vague intention/ specific intention}. Please respond to this user by selecting items from the candidates: {candidate items} ⟨𝑃0, 𝑃2/𝐼1/𝐼2,𝑇3⟩. Your task is to select the best item from a list of candidates that meets the user’s needs based on their search query. Here is the search query of the user: explicit preference/ vague intention/ specific intention} and the candidates are as follows: {candidate items} ⟨𝑃0, 𝑃2/𝐼1/𝐼2,𝑇3⟩. Your task is to select the best item from a list of candidates that matches the user’s query, by comparing the candidate list and their relevance to the user’s query. The user has entered the following search query: explicit preference/ vague intention/ specific intention} And here are the candidate list: {candidate items}
# C INSTRUCTION TEMPLATES FOR PERSONALIZED SEARCH
⟨𝑃1, 𝑃2,𝑇2⟩. You are a search engine. Here is the historical interaction of a user: {historical interactions}. And his personalized preferences are as follows: {explicit preference}. Your task is to generate a new product that are consistent with the user’s preference. ⟨𝑃1, 𝐼1,𝑇2⟩. The user has interacted with a list of items, which are as follows: {historical interactions}. Based on these interacted items, the user current intent are as follows {vague intention}, and your task is to generate products that match the user’s current intent. ⟨𝑃1, 𝐼1,𝑇2⟩. As a shopping guide, you are assisting a user who has recently purchased the following items: {historical interactions} The user has expressed a desire for additional products with the following characteristics: {vague intention} Please provide recommendations for products that meet these criteria. ⟨𝑃1, 𝐼2,𝑇2⟩. As a search engine, you are assisting a user who is searching for the query: {specific intention}. Your task is to recommend products that match the user’s query and also align with their preferences based on their historical interactions, which are reflected in the following: {historical interactions} Turn the task around. The user has recently purchased the following items: {historical interactions} Now he is interested in finding information about an item that he believe he still need, which is: {target item}. However, the user is unsure how to write a query to search for this item based on their preferences. Please assist the user in writing a query. Turn the task around. The user has the following historical interactions: {historical interactions}. And he is interested in purchasing the target item: {target item}. Please analyze the user’s historical interactions and identify his preferences that lead the user to interact with the target item."
Enforcing the relatedness between preference and intentions. The user has searched for the query: {explicit preference/ vague intention} and ultimately selected the product: {target item} Based on the user’s query and final choice, you can infer their preferences. Additionally, the user’s historical interactions have also been influenced by their preferences. Please estimate the user’s historical interactions that match their preferences, taking into account their search query and final product selection. Enforcing the relatedness between preference and intentions. As a search engine, you have observed the following behavioral sequence of a user: {historical interactions/ Using the content and categories of the user’s historical interactions, you can infer their preferences. Please make a prediction about the user’s next query and the product he is likely to ultimately purchase, based on his preference. Turn the task around. A user’s query can provide insight into their preferences as well as their future purchasing intentions. Furthermore, a user’s behavior is often influenced by their preferences. Given the user’s query: {explicit preference/ vague intention} Please analyze the query to speculate on what products the user has previously purchased and predict what products they are likely to purchase next based on their past queries and preferences. ⟨𝑃1, (𝑃2), 𝑃2/𝐼1/𝐼2,�