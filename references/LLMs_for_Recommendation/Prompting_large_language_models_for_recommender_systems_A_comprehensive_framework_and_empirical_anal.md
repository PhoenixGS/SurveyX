# Prompting Large Language Models for Recommender Systems: A Comprehensive Framework and Empirical Analysis

LANLING XU and JUNJIE ZHANG, Gaoling School of Artificial Intelligence, Renmin University of China, China
BINGQIAN LI, Beijing Key Laboratory of Big Data Management and Analysis Methods, China
JINPENG WANG and MINGCHEN CAI, Meituan Group, China

WAYNE XIN ZHAO ∗ and JI-RONG WEN, Gaoling School of Artificial Intelligence, Renmin University, China

Recently, large language models such as ChatGPT have showcased remarkable abilities in solving general tasks, demonstrating the
potential for applications in recommender systems. To assess how effectively LLMs can be used in recommendation tasks, our study
primarily focuses on employing LLMs as recommender systems through prompting engineering. We propose a general framework for
utilizing LLMs in recommendation tasks, focusing on the capabilities of LLMs as recommenders. To conduct our analysis, we formalize
the input of LLMs for recommendation into natural language prompts with two key aspects, and explain how our framework can be
generalized to various recommendation scenarios. As for the use of LLMs as recommenders, we analyze the impact of public availability,
tuning strategies, model architecture, parameter scale, and context length on recommendation results based on the classification of
LLMs. As for prompt engineering, we further analyze the impact of four important components of prompts, i.e., task descriptions, user
interest modeling, candidate items construction and prompting strategies. In each section, we first define and categorize concepts in
line with the existing literature. Then, we propose inspiring research questions followed by experiments to systematically analyze the
impact of different factors on two public datasets. Finally, we summarize promising directions to shed lights on future research.

CS Concepts: • Information systems → Recommender systems

dditional Key Words and Phrases: Large Language Models, Recommender Systems, Empirical Study

ACM Reference Format:

# ACM Reference Format:

Lanling Xu, Junjie Zhang, Bingqian Li, Jinpeng Wang, Mingchen Cai, Wayne Xin Zhao, and Ji-Rong Wen. 2018. Prompting Large
Language Models for Recommender Systems: A Comprehensive Framework and Empirical Analysis. In. ACM, New York, NY, USA,
46 pages. https://doi.org/XXX.XXX

# 1 INTRODUCTION

In order to alleviate the problem of information overload [31, 76], recommender systems explore the needs of users and
provide them with recommendations based on their historical interactions, which are widely studied in both industry

∗ Wayne Xin Zhao (batmanfly@gmail.com) is the corresponding author.

∗ Wayne Xin Zhao (batmanfly@gmail.com) is the corresponding author.

∗ Wayne Xin Zhao (batmanfly@gmail.com) is the corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2018 Association for Computing Machinery. Manuscript submitted to ACM

and academia [23, 28, 29, 84]. Over the past decade, various recommendation algorithms have been proposed to solve
recommendation tasks by capturing the personalized interaction patterns from user behaviors [39, 144]. Despite the
progress of conventional recommenders, the performance is highly dependent on the limited training data from a few
datasets and domains, and there are two major drawbacks. On the one hand, traditional models lack the general world
knowledge beyond interaction sequences. For complex scenarios that need to think or plan, existing methods do not
have commonsense knowledge to solve such tasks [27, 71, 112, 119]. On the other hand, traditional models cannot
truly understand intentions and preferences of users. The recommendation results do not have explainability, and
requirements expressed by users in explicit forms such as natural languages are difficult to consider [47, 52, 126].
Recently, Large Language Models (LLMs) such as ChatGPT have demonstrated impressive abilities in solving general
tasks [24, 118], showing their potential in developing next-generation recommender systems. The advantages of
incorporating LLMs into recommendation tasks are two-fold. Firstly, the excellent performance of LLMs in complex
reasoning tasks indicates the rich world knowledge and superior inference ability, which can effectively compensate
for the local knowledge of traditional recommenders [1, 75, 85]. Secondly, the language modeling abilities of LLMs
can seamlessly integrate massive textual data, enabling them to extract features beyond IDs and even understand user
preferences explicitly [30, 50]. Therefore, researchers have attempted to leverage LLMs for recommendation tasks.
Typically, there are three ways to employ LLMs to make recommendations: (1) LLMs can serve as the recommender to
make recommendation decisions, encompassing both discriminative and generative recommendations [3, 12, 20, 32, 133].
(2) LLMs can be leveraged to enhance traditional recommendation models by extracting semantic representations of
users and items from text corpora. The extensive semantic information and robust planning capabilities of LLMs are
integrated into traditional models [1, 16, 27, 31, 71, 107, 112, 119]. (3) LLMs are utilized as the recommendation simulator
to execute external generative agents in the recommendation process, where users and items may be empowered by
LLMs to stimulate the virtual environment [18, 100, 101, 130, 132]. We mainly focus on the first scenario in this paper.
Considering the gap between the general knowledge from large language models and the domain knowledge from
recommendation models [3, 136], there are two key factors for prompting LLMs as recommenders, i.e., how to select a
LLM as the foundation model and how to construct a prompt as the prompting text. As for LLMs, a growing number of
open-source and closed-source models have emerged, and the same model also has different variants due to settings such
as parameter scales and context lengths [140]. There are notable variations in the performance of different LLMs when
it comes to general language tasks such as generation and reasoning [24, 118]. However, the performance differences of
LLMs in recommendation tasks have not been fully explored. It is worth discussing how to select corresponding LLMs
for specific scenarios and develop corresponding training strategies. As for the prompt, it is an important medium for
interactions between humans and language models, and a well-designed prompt can better stimulate the powerful
capabilities of LLMs [43, 70]. To stimulate the recommendation ability of language models, prompt engineering should
involve not only task description and prompting strategies for general tasks, but also the incorporation of user interest
modeling and the creation of candidate items in recommender systems [17, 70, 125].
Although existing studies have made initial attempts to explore the recommendation capabilities of LLMs like Chat
GPT [12, 20, 32, 68, 89], and some studies have used paradigms such as fine-tuning and instruction tuning to train
LLMs in the field of recommender systems [2, 3, 133, 141], they focus on exploring the performance of a certain task
rather instead of constructing a comprehensive framework to formalize the potential applications of LLM-powered
recommender systems. There are also systematic reviews concentrating on the progress of LLMs [140] and surveys of
2

Table 1. An overview of the primary discoveries presented in our work. We summarize new findings in the second column as “n findings”, and conduct experiments to verify findings discussed in existing literature as “re-validated findings”.

Table 1. An overview of the primary discoveries presented in our work. We summarize n findings”, and conduct experiments to verify findings discussed in existing literature as “r

• Fine-tuning all parameters of LLMs for recommen-
dation is more effective than parameter-efficient fine-
tuning, but more training time is required.
•
ing LLMs as recommenders, such as position bias [32,
74] and lack of domain knowledge [125, 141].
Prompts
Task description
• Our framework can be adapted to point-wise, pair-
wise and list-wise recommendation tasks.
• Different recommendation tasks can be imple-
mented by LLMs through prompts [21, 108].
User interest Modeling
• For short-term interest, it is preferable to summa-
rize recent interactions into text and recommend.
• For long-term interest, it is useful to maintain a
personalized memory to store and retrieve.
• Long-term preferences and short-term intentions
should be effectively combined to model users.
• For short-term interest, only truncating the most
recent items is not the optimal strategy [62].
• Increasing the number of historical items to repre-
sent users brings insignificant gains for LLMs [32].
• Personalized user profiles and customized item
descriptions assist in the user interest modeling for
LLM-based recommendations [89, 125].
Candidate items construction
• Candidate items construction for LLMs is crucial
to the final recommendation results.
• Retrieving candidate items by traditional recom-
mendation models first, and then re-ranking items
by LLMs can further improve the results, but the per-
formance varies on specific methods and datasets.
• LLMs can select from a given candidate item
list [12, 68], as well as directly generate recommen-
dation results [71, 103].
• Indexing methods and grounding strategies of
items are important factors affecting the effective-
ness of LLM-based recommendations [54, 60, 63].
Prompting strategies
• In chain-of-thought prompting, specific problem
decomposition is required for recommendation tasks
rather than general prompts.
• The few-shot prompting strategy has insignificant
advantage in recommendation scenarios.
• Although provided with historical items in chrono-
logical order, LLMs still needs explicit guidance to
understand the importance of recent items [32, 74].
• Role-playing and expert-like prompts can leverage
the capabilities of LLMs in specific fields [38, 105].
Overall
• Leveraging LLMs as recommenders lies in stimulat-
ing the general knowledge of LLMs and integrating
the domain knowledge with the user interest.
• LLMs are restricted by the unacceptable inference
time [54], expensive memory cost [96], limited con-
text length [62] and black-box abilities [55].
<div style="text-align: center;">Task description
</div>
<div style="text-align: center;">Candidate items construction
</div>
recommender systems empowered by LLMs [51, 61, 116]. However, previous surveys generally use specific criteria to
classify existing work and introduce them separately. They mainly focus on showcasing related work and summarizing
advantages and limitations, rather than conducting additional experiments to validate existing results and explore new

<div style="text-align: center;">Re-validated findings
</div>
discoveries. Our work focuses on the ability of LLMs to directly serve as recommenders, aiming to establish a general
framework of Pro mpting L arge L anguage M odels for Rec ommendation (ProLLM4Rec).

In order to conduct our analysis for ProLLM4Rec, we formalize the input of LLMs for recommendation into natural
language prompts with two key aspects: LLMs and prompts, and explain how our framework can be generalized to
various recommendation scenarios and tasks. As for the use of LLMs as recommenders, we analyze the impact of
the public availability, tuning strategies, model architecture, parameter scale, and context length on recommendation
results based on the classification of LLMs. As for prompt engineering, we further analyze the impact of four important
components of prompts, i.e., task description, user interest modeling, candidate items construction, and prompting
strategies. Given personalized prompts that include task description and user interest, the LLM selects, generates, or
explains candidate items based on general world knowledge and personalized user profiles. For each module, we first
define and categorize concepts in line with the existing literature. Then, we propose inspiring research questions,
followed by detailed experiments to systematically analyze the impact of different conditions on the recommendation
performance. Based on the empirical analysis, we finally summarize empirical findings for future research.

# In general, the contributions of our work can be summarized as follows:

• We derive a general framework ProLLM4Rec to sum up existing work of utilizing LLMs as foundation models fo
recommendation, which can be generalized to multiple scenarios and tasks by different LLMs and prompts.

In what follows, we first review the related work in Section 2. In Section 3, we present our proposed general framework
and its instantiation, and introduce overall settings of the following experiments. As the core components of this
paper, we discuss two main aspects of ProLLM4Rec, i.e., LLMs and prompts in Section 4 and Section 5, respectively. For
each aspect, we generalize key factors that affect recommendation results, and conduct corresponding experiments to
summarize empirical findings. At last, Section 6 concludes this paper and sheds lights on future directions.

# 2 RELATED WORK

# 2.1 Recommender Systems

For tackling the challenge of information overload [76, 93, 127], recommender systems have become pivotal tools for
delivering personalized contents for users across various domains. In line with previous studies, recommendation
algorithms aim to derive user preferences and behavioral patterns from their historical interactions. The most common
technique for the interaction-based recommendation is Collaborative Filtering (CF) [86, 90], which recommends items
based on preferences of similar users. Matrix Factorization (MF) [42] is a prevalent approach in collaborative filtering,

and it constructs embedding representations for users and items from the interaction matrix, facilitating the algorithm
to calculate similarity scores efficiently. Furthermore, Neural Collaborative Filtering (NCF) [29], integrating deep
neural networks, replaces the inner product used in MF with a neural architecture, thereby demonstrating better
performance than previous methods. Contemporary advancements in deep neural network architectures have enhanced
the integration of user and item embeddings [66]. For example, since recommendation data can be represented as
graph-structured data, Graph Neural Network (GNN) [117] can be utilized to encode the information of the interaction
graph (nodes consist of users and items), and generate meaningful representations via message propagation and
contrastive learning strategies [28, 64, 104, 115]. As Pre-trained Language Models (PLM) gain prominence, there is a
growing interest in pre-trained large-scale recommendation models powered by PLMs [31, 135, 144]. In addition to
user-item pairs and IDs, content-based recommendation algorithms leverage auxiliary modalities such as textual and
visual information to augment user and item representations in recommendation tasks [80, 110, 127].

# 2.2 Large Language Models for Recommender Systems

Large Language Models (LLMs) are a cutting-edge advancement in artificial intelligence that excel in understanding and
generating human-like texts [81, 95, 129]. LLMs are usually transformer-based models and trained on vast amounts of
textual data with billions of parameters, allowing them to comprehend contexts, generate coherent sentences, and even
mimic human conversations [24, 118]. Through this process, LLMs have shown prominent potentials in the field of
Natural Language Processing (NLP), and have demonstrated various incredible capabilities in dealing with complex NLP
tasks, including but not limited to In-Context Learning (ICL) [4], instruction following [96] and step-by-step reasoning
abilities [140]. Recently, LLMs have been increasingly integrated into recommender systems to provide personalized
recommendations [51, 116]. Recent studies have explored the fusion of LLMs with recommender systems, which can be
divided into the three paradigms, i.e., LLM as recommendation model (Section 2.2.1), LLM improves recommendation
models (Section 2.2.2) and LLM as recommendation simulator (Section 2.2.3) as follows.
2.2.1 LLM as Recommendation Model. This paradigm takes the LLM as a recommender system. Employing diverse
strategies like pre-training, fine-tuning, or prompting, LLMs can combine general knowledge with input data to yield per
sonalized recommendations for users [12, 32, 37]. Due to the variety of recommendation tasks, LLM as recommendation
model can be categorized into two types: discriminative recommendation and generative recommendation.
• Discriminative recommendation instructs LLMs to make recommendation decisions on the given candidate items,
usually focusing on item scoring [19] and re-ranking tasks [12]. For Click-Through Rate (CTR) prediction tasks, Liu et
al. [68] designed specific zero-shot and few-shot prompts to evaluate abilities of LLMs on rating predictions. LLMs
were required to assign a score for the item according to the previous rating history of users and the score range
given in prompts, while the result indicated that LLMs can outperform classical rating methods (e.g., MF and MLP)
in few-shot conditions [68]. Kang et al. [40] further formulated the rating prediction task as multi-class classification
and regression task, investigating the influence of model size on recommendation performance. Different from these
methods, Hou et al. [32] structured a re-ranking task, employing in-context learning approaches for LLMs to rank
items in the candidate pool. Previous studies highlighted the sensitivity of LLMs to the sequence of interaction histories
provided in prompts [74], which can be alleviated by strategies such as recency-focused prompting [32].
• Generative recommendation requires LLMs to generate items recommended to users, either from candidate item
lists within prompts or from LLMs with general knowledge [51]. GenRec [37] leveraged the contextual comprehension
5

ability of LLMs to transform interaction histories into formulated prompts for next-item predictions. To address instances
where GenRec might propose items absent in candidate lists, GPT4Rec [48] came up with the method that used BM25
algorithm to retrieve the most similar item in candidate item lists with the item generated by LLMs. In addition to
top-n recommendations, LLMs can be leveraged for generative tasks such as explainable recommendations [11, 44, 49,
73, 124] and review summarization [21, 69, 108]. Moreover, with the incredible abilities in dialogue comprehension
and communication, LLMs are naturally considered as the backbone of conversational and interactive recommender
systems. ChatRec [20] designed an interactive recommendation framework based on ChatGPT, which can comprehend
requirements of users through multi-turn dialogues and traditional recommendation models. Moreover, RecLLM [18]
combined the dialogue management module with a ranker module and a controllable LLM-based user simulator to
generate synthetic conversations for tuning system modules. Apart from these methods, InteRecAgent [35] employed
LLMs as the brain and recommender models as tools, combining their respective strengths to create an interactive
recommender system [35]. As a conversational recommender system, InteRecAgent enabled traditional recommender
systems to become interactive systems with a natural language interface through the integration of LLMs.
There are mainly two paradigms for adapting LLMs as recommenders, i.e., non-tuning paradigm and tuning paradigm.
• Non-tuning paradigm keeps parameters of LLMs fixed and extracts the general knowledge of LLMs with prompting
strategies. Existing work of non-tuning paradigm focuses on designing appropriate prompts to stimulate recommenda
tion abilities of LLMs [58, 108, 125]. Liu et al. [68] proposed a prompt construction framework to evaluate abilities of
ChatGPT on five common recommendation tasks, each type of prompts contained zero-shot and few-shot versions.
Hou et al. [32] not only used prompts to evaluate abilities of LLMs on sequential recommendation, but also introduced
recency-focused prompting and in-context learning strategies to alleviate order perception and position bias issues of
LLMs. ChatRec [20] and InteRecAgent [35] mentioned above are also within the classic non-tuning paradigm.
• Tuning paradigm aims to update parameters of LLMs to inject recommendation capabilities into LLM itself. The
tuning strategies include fine-tuning [33, 141] and instruction tuning [72, 79]. P5 [21] proposed five types of instructions
targeting at different recommendation tasks to fine-tune a T5 [81] model. The instructions were formulated based
on conventional recommendation datasets with designed templates, which equipped LLMs with generation abilities
for unseen prompts or items [21]. InstructRec [133] further designed abundant instructions for tuning, including 39
manually designed templates with preference, intention, task form and context of a user. Compared with these methods,
TallRec [3] used LoRA [33], a parameter-efficient tuning method, to handle the two-stage tuning for LLMs. It was first
fine-tuned on general data of Alpaca [92], and then further fine-tuned with the historical information of users.
Although LLMs as recommendation models present a way of utilizing the common knowledge of LLMs, it still encounters
some problems to be coped with. Due to the high computational cost [60, 96] and slow inference time [54], LLMs are
struggled to be efficient enough compared to traditional recommendation methods [20, 32]. Additionally, constraints
on input sequence length will limit the amount of external information (e.g., candidate item lists) [62], leading to the
degrading performance of LLMs in scenarios such as sequential recommendation. Furthermore, since information in
recommendation tasks is challenging to be expressed in natural language [63, 126], it is hard to formulate appropriate
prompts that make LLMs truly understand what they are required to do, leading to unexpected performance.
2.2.2 LLM Improves Recommendation Models. This method mainly utilizes LLMs to generate auxiliary information
to enhance the performance of recommendation models [16, 107, 112], based on the reasoning abilities and common

knowledge. The research on how to improve recommendation models with LLMs can be divided into three categori
i.e., LLM as feature encoder, LLM for data augmentation and LLM co-optimized with domain-specific models.

• LLM as feature encoder. The representation embeddings of users and items are important factors in classical
recommender systems [28, 84]. LLMs serving as feature encoders can generate related textual data of users and items,
and enrich their representations with semantic information. U-BERT [80] injected user representations with user
review texts, item review texts and domain IDs, augmenting the contextual semantic information in user vectors. Wu et
al. [114], on the other hand, employed language models to generate item representations for news recommendation.
With the development of LLMs and prompting strategies, BDLM [134] constructed the prompt consisting of interaction
and contextual information into LLMs, and obtained top-layer feature embeddings as user and item representations.
• LLM for data augmentation. For this paradigm, LLMs are required to generate auxiliary textual information for
data augmentation [1, 66, 71, 112]. By using prompting or in-context learning strategies, the related knowledge will be
extracted out in different text forms to facilitate recommendation tasks [16, 107, 119]. One form of auxiliary textual
information is summarization or text generation, enabling LLMs to enrich representations of users or items [110]. For
example, Du et al. [16] proposed a job recommendation model which utilized the capability of LLMs for summarization
to extract user information and job requirements. Considering item descriptions and user reviews, KAR [119] extracted
the reasoning knowledge on user preferences and the factual knowledge on items through specifically designed prompts,
while SAGCN [66] utilized a chain-based prompting strategy to generate semantic information. Another form of
using the textual features generated from LLMs is for graph augmentation in the recommendation field. LLMRG [107]
leveraged LLMs to extend nodes in recommendation graphs. The resulting reasoning graph was encoded using GNN,
which served as additional input to enhance sequential models. LLMRec [112] adopted three types of prompts to
generate information for graph augmentation, including implicit feedback, user profile and item attributes.
• LLM co-optimized with domain-specific models. The categories mentioned above mainly focus on the impact of
common knowledge for domain-specific models [110]. However, LLM itself often struggles to handle domain-specific
tasks due to the lack of task-related information [40, 125]. Therefore, some studies conducted experiments to bridge the
gap between LLMs and domain-specific models. BDLM [134] proposed an information sharing module serving as an
information storage mechanism between LLMs and domain-specific models. The user embeddings and item embeddings
stored in the module were updated in turn by the LLM and the domain-specific model, enhancing the performance
of both sides. CoLLM [136] combined LLMs with a collaborative model, which formed collaborative embeddings for
LLM usage. By tuning LLM and collaborative module, CoLLM showed great improvements in both warm and cold-start
scenarios. In conversational recommender systems, approaches such as ChatRec [20] and InteRecAgent [35] considered
LLMs as the backbone, and leveraged traditional recommendation models for candidate item retrieval.
In addition to the context limitation and computational cost of LLMs [96], the paradigm that LLM improves recom
mendation models also encounters other problems. (1) Although LLMs can enhance offline recommender systems to
avoid online latency, this paradigm also limits the ability of LLMs to model real-time collaborative filtering information,
neglecting the key factor for recommendation [110, 112, 119]. (2) Feature encoding, data augmentation, and collaborative
training inevitably expose the user data to LLMs, which may bring privacy, security and ethical issues [6, 87, 113].
2.2.3 LLM as Recommendation Simulator.  Due to the gap between offline metrics and online performance of recom
mendation methods [29, 77], it is necessary for the designed approach to get intents of users by simulating real-world

elements. In this way, LLM as the recommendation simulator is introduced by taking LLMs as the foundational archi
tecture of generative agents, and agents simulate the virtual users in the recommendation environment [101, 130, 132].

Recently, there emerged a lot of work studying the performance of LLMs as the recommendation simulator. Agent4rec [130
was a movie simulator consisting of two core fractions: LLM-empowered generative agents and recommendation
environment. The work equipped each agent with user profile, memory and actions modules, mapping basic behaviors
of real-world users. AgentCF [132], on the other hand, considered not only users but also items as agents. It captured
the two-sided relations between users and items, and optimized these agents by prompting them to reflect on and adjust
the misleading simulations collaboratively [132]. Moreover, in addition to behaviors within the recommender system,
RecAgent [100, 101] took external influential factors of user agent simulation into account, such as friend chatting and
social advertisement. In order to describe users accurately, RecAgent applied five features for users, and implemented
two global functions including real-human playing and system intervention to operate agents flexibly.
Although LLM as recommendation simulator aims to imitate real-world recommendation behaviors to enhance the
recommendation performance, it still has deficiencies in some aspects. Firstly, since current work is mainly demo
systems that operate a few agents [100, 101], there still exists a gap between virtual agent environment and real-world
practical recommendation applications, which requires further research and development. Additionally, LLMs may
arise privacy and safety concerns. Many studies take ChatGPT as the architecture of agents, presenting security risks to
the recommended information for users [132]. Moreover, Zhang et al. [130] have explored that hallucination in LLMs
can exert huge impact on recommendation simulations. The LLM sometimes fails to accurately simulate human users,
such as providing inconsistent score for an item and fabricating non-existent items for rating.

# 2.3 Differences with Existing Work

Previous surveys of LLMs for recommender system usually categorized existing work with a classification standard
and introduced studies respectively. Lin et al. [61] categorized existing work into the targets and the methods of
adapting LLMs to recommendation tasks. Wu et al. [116] mainly focused on the form of information between LLMs and
recommender systems, including LLM embeddings, LLM tokens and LLMs. Fan et al. [17] summarized the framework
into four sections, involving deep representation learning, pre-training, fine-tuning and prompting of LLMs. These
research mainly concentrated on demonstrating related work and summarizing the advantages and limitations.
Compared to previous work, we concentrate on the ability of LLMs leveraged for recommendation tasks, and provide a
systematic empirical analysis on prompting LLM-based recommendations by devising a general framework ProLLM4Rec.
We mainly focus on two aspects, i.e., LLMs and prompt engineering, providing definitions and solutions from both
conceptual and methodological perspectives. Furthermore, we conduct experiments to discover new findings and
validate results previously discussed in existing research, serving as an inspiration for future research efforts.

# 3 GENERAL FRAMEWORK AND OVERALL SETTINGS

In this section, we present our proposed ProLLM4Rec with two important components, namely LLMs and prompts.
Generally speaking, the rich world knowledge and general capabilities of LLMs demonstrate the potential to develop LLM
powered recommender systems. Nevertheless, it is essential to introduce appropriate prompting engineering to provide
domain-specific knowledge on recommendation tasks for LLMs to serve as personalized recommenders. In the following,
we first describe our general framework by defining several key elements (Section 3.1). Then, we explain how our

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9099/90999ae4-717c-43fa-b8b7-0ede3f658e81.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 1. The overall framework of our proposed ProLLM4Rec.
</div>
framework can be generalized to various recommendation scenarios and tasks by framework instantiation (Sectio
Finally, we introduce the details of the overall experimental settings for further analysis (Section 3.3).

# 3.1 Overview of the Approach

When it comes to the use of LLMs in recommender systems, the first step involves choosing suitable LLMs tailored
to specific scenarios, with an emphasis on their distinct capabilities. Subsequently, it is important to conduct prompt
engineering to instruct LLMs to perform effective recommendations. Typically, in the realm of recommender systems
using LLMs, despite the task settings in existing studies vary a lot, the overall prompting format remains relatively
consistent with minor variations [12, 20, 32, 108]. Therefore, to unify existing prompting approaches for different recom
mendation purposes, as illustrated in Fig. 1, we establish a general framework for prompt engineering in ProLLM4Rec,
consisting of four key factors in the prompts. Firstly, task description is required to clearly express the specific goal of
recommendation tasks. Secondly, prompts need to be carefully designed, to express user interest and meanwhile enable
LLMs to provide personalized recommendations with both world and domain knowledge. Thirdly, the purpose of a
recommender system is to provide users with candidate items, and potential candidate items should be constructed
for LLMs to facilitate the recommendation with the understanding of domain-specific item information. Furthermore,
special prompting strategies can be further employed to enhance the specialized recommendation capabilities of LLMs.
In what follows, we will thoroughly examine the impact of each factor on the recommendation performance.
3.1.1 Key Elements in Our Framework. To carry out the experiments, we first describe the key elements in ProLLM4Rec,
to clarify their definitions and scope in this work. Specially, we introduce the following five elements for our framework:
• Large language models. As proposed in previous research [2, 32, 53, 136], there exists a large gap between general
language modeling and personalized user behavioral modeling, making it non-trivial to utilize LLMs in recommender
systems. In this work, we investigate the efficacy of LLMs from perspectives of public availability, tuning strategies,
model architecture, parameter scale, and context length, aiming to gain insights into the selection of appropriate LLMs
for performing recommendation tasks. Observations and discussions on the use of LLMs are presented in Section 4.
• Task description. To adapt LLMs to the scenario of recommendation, it is necessary to clearly express the context
and target of recommender systems for LLMs in the prompt, i.e., task description. With different prompt descriptions of
tasks, large language models can be utilized to various recommendation scenarios and tasks such as click-through rate
predictions [2, 3, 40], sequential recommendation [12, 32, 141] and conversational recommender systems [20, 30, 105].
• User interest modeling. The modeling of user interest is the key to recommendation tasks [29, 84]. When leveraging
LLMs for recommendation, users are generally expressed in natural language text, which is different from traditional
9

LLMs
Task description
User interest
Candidate items
Prompting strategies
Related Work
Not tuning setting
CTR
predictions,
rating, re-ranking
recent and relevant
items (with attributes),
user profile
pointwise, pairwise,
listwise item(s)
chain of thoughts, in-
context learning, role
prompting
[12, 14, 32, 40, 58, 65,
68, 69, 74, 89, 99, 108,
109, 125, 142]
ChatGPT,
GPT-4
conversational rec-
ommender systems
user explicit interest,
interactive feedback
recalled from tradi-
tional models
role prompting
[18, 20, 30, 35, 38,
105, 145]
generative
recom-
mendation
recent items (with at-
tributes), user profile
(not provided, gen-
eration methods)
basic prompts
[71, 103]
(Parameter-efficient) Fine-tuning setting
LLaMA,
LLaMA2,
Vicuna,
ChatGPT
CTR
predictions,
rating, re-ranking
recent and relevant
items (with attributes),
user profile, collabora-
tive embedding
pointwise, listwise
item(s)
chain of thoughts, role
prompting, soft prompt-
ing
[3, 14, 19, 40, 55, 60,
62, 88, 98, 122, 128,
136]
LLaMA,
LLaMA2,
BART, GPT
recall, retrieving
recent and relevant
items (with attributes)
(not provided, item
grounding methods)
basic prompts
[2, 37, 48, 54, 63, 78,
135, 141]
Instruction tuning setting
(Flan-)T5,
LLaMA2
rating,
ranking,
retrieving, explana-
tion, summarization
recently
interacted
items,
user
profile,
short-term intentions
pointwise, pairwise,
listwise item(s)
basic prompts
[9, 21, 57, 72, 79,
133]
approaches capturing user preference from ID-based behavior sequences [39, 115]. In this paper, we mainly consider
the reflected user interest based on his or her interaction behaviors with interacted items. Especially, as detailed in
Section 5.2, we employ item description texts, user profiles, and historical interactions between users and items to
reveal the underlying user interest using natural languages [62, 89, 108, 125].
• Candidate items construction. The purpose of recommender systems is to provide users with items to choose from,
so candidate items construction is a crucial step in our framework [12, 32, 133]. A simple approach is to provide several
candidate items in prompts for the LLM, e.g., the items recalled by traditional recommendation models [62, 128]. Due
to the input length limitation of LLMs, it is not possible to include all items in the prompts. In addition to selecting
suitable candidate sets, there are also methods that directly generate candidate items by LLMs, utilizing strategies such
as output probability distribution [128] and vector quantization [141] for item indexing and grounding. Section 5.3 will
focus on the construction strategies of candidate items, including selection and generation.
• Prompting strategies. Despite the impressive capabilities of LLMs, they tend to exhibit unsatisfactory performance
in providing personalized recommendations [12, 32, 40, 68]. The reason may stem from the significant semantic gap
between the general knowledge encoded in LLMs and the domain-specific behavioral pattern and item catalogs of
recommender systems [53, 136]. To specialize LLMs to recommender systems, we summarize and propose several
prompting strategies specialized for recommendation tasks. Details will be discussed in Section 5.4.

approaches capturing user preference from ID-based behavior sequences [39, 115]. In this paper, we mainly consider
the reflected user interest based on his or her interaction behaviors with interacted items. Especially, as detailed in
Section 5.2, we employ item description texts, user profiles, and historical interactions between users and items to
reveal the underlying user interest using natural languages [62, 89, 108, 125].

<div style="text-align: center;">Table 3. Statistics of two public datasets for ProLLM4Rec.
</div>
Dataset
#User
#Item
#Interaction
Sparsity
Item Attributes
MovieLens-1M
6,040
3,706
1,000,209
4.4642%
release year, title, genre
Amazon-Books
13,469
12,984
1,142,940
0.6536%
title, categories, brand, price, description
# 3.2 Instantiation of ProLLM4Rec

By combining the key elements mentioned above, we can instantiate various types of recommender systems in our
framework with the following five steps. Specifically, (1) we can employ LLMs with varying levels of public availability,
different tuning strategies, model architectures, parameter scales, and context lengths. (2) We can define a range
of task description, such as retrieving, rating, recalling, and ranking. (3) Regarding user interest modeling, we can
employ different types of interest, representation forms, and modeling methods. (4) When collecting candidate items,
we take into account their different representation types, sources, and grounding methods. (5) Additionally, we can
introduce several well-designed prompting strategies to effectively guide the recommendation capabilities of LLMs.
To demonstrate the compatibility and versatility of our framework, we summarize previous work on LLM-powered
recommender systems in Table 2 based on various settings in our framework ProLLM4Rec.

# 3.3 Experimental Settings

In this section, we introduce the overall settings of the following experiments. We first describe the bas
of the two public datasets, and then present the configurations and implementations to conduct our stu

In this section, we introduce the overall settings of the following exp
of the two public datasets, and then present the configurations and i

In this section, we introduce the overall settings of the following experiments. We first describe the basic information
of the two public datasets, and then present the configurations and implementations to conduct our study in detail.
3.3.1 Datasets. The domain characteristics of movies and books are closer to the general knowledge of LLMs, which
facilitates the further analysis. Considering the scale, popularity and side information of public datasets, we select two
representative datasets to conduct our study, i.e., MovieLens-1M [25] and Amazon Books (2018) [76] as follows.
• MovieLens-1M [25] is one of the most widely used benchmark datasets in the field of recommender systems, covering
movie ratings and attributes on the website movielens.org. We use the one million version from the MovieLens datasets.
• Amazon Books (2018) [76] is an updated version of the Amazon review dataset released in 2014. At first, Amazon
only operated online book sales business, so the data in the book field is the most abundant. To improve the data quality,
we filter out inactive users and unpopular products, and remove dirty data without necessary attributes.
In our research, we are concerned about how LLMs can fully utilize the domain knowledge to make recommendations,
and use the title of items as the input for prompts. However, titles are not enough to describe items, and there are
deviations between the text of the title and the content of the item itself (e.g., the movie Twelve Monkeys). Therefore,
we further investigate the benefits of detailed item descriptions on the recommendation effect. As shown in Table 3,
there are no item descriptions in the original dataset of MovieLens, only the release year, title, and genre. To enrich the
movie dataset, we use the general knowledge of ChatGPT 1 to generate text descriptions for movies.
3.3.2 Configuration and Implementation. As for ProLLM4Rec, we can directly evaluate the cold-start recommendation
ability of large language models in the zero-shot setting, as well as evaluate the fine-tuned ability with few or full

1 The URL of ChatGPT API: https://chat.openai.com/. Note that there are multiple versions of the ChatGPT API, and OpenAI has released interfaces on March 1 and June 13, 2023, respectively. In the absence of clear annotations, the ChatGPT used in this article is “gpt3.5-turbo-4k-0613”.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1154/1154baf1-95c2-4670-9bc7-31bd625089fc.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a160/a160800e-28e4-4146-8afe-6d06d35bfde8.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 2. The basic prompts used in the MovieLens-1M dataset for the following experiments.
</div>
# Fig. 2. The basic prompts used in the MovieLens-1M dataset for the following experiments

recommendation samples in the fine-tuning setting. Considering the typical scenarios of LLMs in recommender systems,
we conduct experiments on two representative task settings, i.e., (1) the zero-shot ranking task without modifying
parameters of LLMs [12, 20, 32, 74] and (2) the click-through rate prediction task with LLMs tuned [3, 19, 40, 88].
• Zero-shot ranking task. On the one hand, we evaluate the zero-shot recommendation performance of LLMs
for cold-start scenarios to study the effect of LLMs and the design of prompts. In this paper, our approach mainly
concentrates on ranking tasks that better reflect the capabilities of LLMs [12, 20, 32, 74]. As shown in Fig. 1, information
of users and items is encoded into the prompt as inputs for LLMs. In this setting, we do not modify the parameters of
LLMs, so the evaluated models are closed-source large language models or open-source models without fine-tuning. To
conduct experiments on the impact of each factor on recommendation results with LLMs, we implement the overall
architecture based on the open-source recommendation library RecBole [121, 138, 139] and the zero-shot re-ranker
LLMRank [32]. Our basic prompt used in the MovieLens-1M dataset for the zero-shot ranking task is shown in Fig. 2(a).
• Click-through rate prediction task with LLMs tuned. On the other hand, we evaluate the fine-tuned recommen
dation performance of LLMs to explore how LLMs adapt to recommendation scenarios with data provided [3, 19, 40, 88].
Although our framework can be generalized to various recommendation tasks, we concentrate on exploring the fine
tuning performance of LLMs with point-wise Click-Through Rate (CTR) prediction tasks to reduce selection bias. In this
setting, we not only consider fine-tuning LLMs using recommendation data, but also devise a two-stage approach of
using instruction data to fine-tune LLMs first, and then implement recommendation fine-tuning for further adaptation.
Specifically, we compare the fine-tuned recommendation performance of the original LLM and the LLM after instruction
tuning, respectively. LLaMA-7B [95] and LLaMA2-7B [96] are the original models, while Alpaca-lora-7B [92] and
LLaMA2-chat-7B [96] are LLMs after instruction tuning. In terms of the tuning strategies of LLMs, we report results
with both parameter-efficient fine-tuning and complete fine-tuning. We implement the fine-tuning framework based on
the open-source library transformers and the instruction tuning code of LLaMA with Stanford Alpaca data 2. Our
instruction data used in the MovieLens-1M dataset for the click-through rate prediction task is illustrated in Fig. 2(b).
3.3.3 Evaluation Metrics. As for the zero-shot ranking task, considering economic and efficiency factors, we refer to
existing literature [32, 89] to randomly sample 200 users instead of evaluating results on the whole dataset. For each

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/116f/116fc8f6-314c-4ba7-b075-d1d0fbc40794.png" style="width: 50%;"></div>
user from the sample set, we sort all items that the user interacts with in the chronological order. Then, we evaluate
the results based on the leave-one-out strategy and treat the last interacted item as the ground truth. For performance
comparison, we fix the length of candidate items to 20 as in [32], and mix the other 19 items from the ground truth in
random positions by default. As for evaluation, we utilize two widely used ranking metrics in recommender systems,
i.e., Recall [5] and Normalized Discounted Cumulated Gain (NDCG) [36]. Since 20 items are selected for candidate
generation, we set 𝑘 = 20 for Recall@ 𝑘 to measure whether all ground-truth items have been recalled. In existing
literature, the re-generation method of multiple results until the format requirements are met can be employed to obtain
the final response [56], while we consider the recall capability within one inference for fair evaluation. Furthermore,
we set 𝑘 = 1, 10, 20 for NDCG to explore the detailed recommendation performance in terms of ranking abilities. To
assure a scientific research, we repeat each experiment three times and take the average value as the final result.
As for the CTR prediction task, we first sort the original dataset by timestamp, use the latest 10,000 records for
training and evaluation, and regard other previous data as the interaction history of users. Then, we split the ten
thousand interactions into the training, validation and test sets in ratio of 8:1:1. For each interaction, we retain 10
historical interacted items as user representations, and set thresholds based on rating data to obtain user preferences [3].
Interactions with a rating higher than or equal to the threshold will be considered items that the user likes, while the
opposite indicates dislikes. For the MovieLens-1M dataset, the rating threshold is 4, and we set 5 for the Amazon-Books
dataset. We employ the training data to fine-tuning LLMs and evaluate the recommendation performance on the test
set. The validation set is used for selecting best checkpoints during the training process. As for evaluation of prediction,
we utilize the widely used metric for CTR predictions, i.e., accuracy. In a random state, the accuracy is around 0.5.
3.3.4 Discussion on Variable Factors. To conduct our analysis, we mainly focus on two key aspects, i.e., LLMs and
prompts. As for the effects of LLMs, we analyze the impact of public availability, tuning strategies, model architecture,
parameter scale, and context length on recommendation results based on the classification of LLMs. As for prompt
engineering, we further analyze the impact of four important components of prompts, i.e., task description, user interest
modeling, candidate items construction and prompting strategies. Due to the fact that all factors have an impact on
the final result, it is also crucial to select the other aspects when focusing on one aspect. Limited by resources and
efficiency, it is neither necessary nor feasible to exhaust all possibilities. When not explicitly specified, the LLM uses
ChatGPT released in June 2023 to ensure the consistent recommendation quality. For the design of prompts, we refer to
the template in [32] and emphasize the most recent items to re-rank the random 20 candidate items. The default method
for modeling user interest is to concatenate the title sequence of recently interacted items using natural languages.

# 4 THE IMPACT OF LARGE LANGUAGE MODELS AS RECOMMENDER SYSTEMS

LLMs are the core of our framework ProLLM4Rec, and directly determine the performance of LLMs as recommender
systems [32, 40, 68, 69]. Therefore, it is worth exploring how to choose a suitable LLM as the foundation for recom
mendation. In this section, we compare the differences between LLMs and traditional models on the recommendation
performance, discuss how the different properties and tuning strategies of LLMs affect the recommendation results,
present the limitations of LLMs as recommenders, and draw empirical conclusions through systematic experiments.

# 4.1 Classification of Large Language Models

In this paper, we consider language models that have a size larger that one billion as LLMs. In line with existing
researches [17, 51, 61, 116, 140], LLMs can be categorized into different classes from several perspectives. Most typically,

LLMs can be divided into open-source and closed-source models in terms of the public availability. When it comes to
leveraging LLMs as the foundation model in recommender systems, tuning strategies can adjust LLMs towards specific
recommendation tasks. From the perspective of the model architecture, various LLMs can also be categorized into
types of encoder-decoder, causal decoder, and prefix decoder [140]. For the same framework of a LLM, it is widely
acknowledged that the parameter scale and context length are two key factors that jointly determine abilities of
LLMs [32, 62]. To explore the recommendation performance with respect to different variants of LLMs, we focus on
five aspects, i.e., public availability, tuning strategies, model architecture, parameter scale and context length as follows.
4.1.1 Public Availability. According to whether the model checkpoints can be publicly obtained, existing LLMs can be
divided into open-source models and closed-source models, and both can be leveraged as recommender systems.
• Open-source models refer to LLMs whose model checkpoints can be publicly accessible. As shown in Table 2,
researchers often use recommendation data to fine-tune open-source models for performance improvement. As the
representative of open-source models, LLaMA [95] and its variants like Vicuna [8] are widely used when leverag
ing LLMs for recommender systems [54, 141]. Parameter-Efficient Fine-Tuning (PEFT) strategies such as Low-Rank
Adaptation (LoRA) [33] are frequently adopted for recommendation data considering the trade-off between effect and
efficiency [3, 60]. Other models such as the Flan-T5 [10] series from Google Inc. and ChatGLM [129] from Tsinghua
University are also popular in the field of recommender systems. The publicly available checkpoints of open-source
models provide flexibility for LLMs to modify parameters tailored for recommendation tasks.
• Closed-source models refer to LLMs whose model checkpoints can not be publicly accessible. For the closed-source
LLMs utilized for recommenders, researchers generally study the zero-shot recommendation ability in cold-start
scenarios. The most typical closed-source model is the ChatGPT series from OpenAI. The subsequent GPT-4 has
stronger capabilities compared to ChatGPT, but it is still not open-source. In this paper, ChatGPT refers to the API
function of “gpt3.5-turbo-4k-0613” unless specified. Without checkpoints, OpenAI provides several ways for researchers
and users to improve the model performance on specific tasks, such as plugins for website browsing [45] and interfaces
for fine-tuning ChatGPT [55]. However, the flexibility of closed-source models as recommender systems is still limited
due to the expensive price and black-box parameters. Faced with this challenge, existing literature has explored to
inject knowledge of recommender systems into closed-source models by means of prompt design [99, 125], retrieval
enhancement [35, 89, 108] and combination of traditional recommendation models [20, 105].
4.1.2 Tuning Strategies. During the deployment of LLMs in recommender systems, we can also classify existing work
depending on whether LLMs are fine-tuned, as well as the various fine-tuning strategies employed, i.e., not-tuning
setting, fine-tuning setting and instruction tuning setting.
• Not-tuning setting means evaluating the zero-shot recommendation ability of LLMs, which is generally used
for closed-source models such as ChatGPT. By designing different prompt templates, LLMs without fine-tuning can
be directly used for recommendation tasks such as click-through rate predictions [14, 40], sequential recommen
dation [32, 108], and conversational recommender systems [30, 38, 105]. In this case, the user interest is expressed
explicitly (e.g., ratings and reviews) or implicitly (interacted items of users), and the limited candidate items can be
recalled by traditional models, while prompting strategies such as role prompting and chain of thoughts are used.
Inspired by Artificial Intelligence Generated Content (AIGC), it is worth noting that the excellent generation ability of
LLMs provides opportunities for generative recommendation [71, 103]. Without providing candidate items, generative
14

• Not-tuning setting means evaluating the zero-shot recommendation ability of LLMs, which is generally used
for closed-source models such as ChatGPT. By designing different prompt templates, LLMs without fine-tuning can
be directly used for recommendation tasks such as click-through rate predictions [14, 40], sequential recommen
dation [32, 108], and conversational recommender systems [30, 38, 105]. In this case, the user interest is expressed
explicitly (e.g., ratings and reviews) or implicitly (interacted items of users), and the limited candidate items can be
recalled by traditional models, while prompting strategies such as role prompting and chain of thoughts are used.
Inspired by Artificial Intelligence Generated Content (AIGC), it is worth noting that the excellent generation ability of
LLMs provides opportunities for generative recommendation [71, 103]. Without providing candidate items, generative

language models can directly generate the desired items that users need based on recommendation requirements, and
they can also be generalized into our framework ProLLM4Rec as shown in Table 2.

• Fine-tuning setting means that using recommendation data to fine-tune LLMs as recommender systems. Considering
cost and efficiency, researchers often use parameter-efficient fine-tuning (e.g., Low-Rank Adaptation of Large Language
Models, LoRA [33]) to quickly adapt to recommendation scenarios [3, 60, 141]. As for LLMs, open-source models based
on LLaMA [95] are widely used, including but not limited to LLaMA, LLaMA2 and Vicuna [8] with different parameter
sizes. Based on whether candidate items are provided, existing work on fine-tuning LLMs can be further divided into two
kinds of recommendation tasks. On the one hand, researchers have explored fine-tuning models for recommendation
that provide candidate items such as rating, re-ranking and predictions. Specifically, the fine-tuning interface of the
closed-source model ChatGPT has brought new breakthroughs to the research of LLMs, and there have been attempts
to fine-tune ChatGPT for recommendation tasks [55]. On the other hand, LLMs can also be fine-tuned for recall tasks
of recommender systems by retrieving candidates from the whole item pool [2, 54, 63, 78]. Through well-designed
indexing, alignment, and retrieval strategies, directly generating recommendation items without providing lengthy
candidate sequences is more suitable for practical application scenarios, which has not yet been fully explored.
• Instruction tuning setting  means providing template instructions of recommenders as prompts to tuning tar
geted LLMs, generally involving multiple recommendation tasks [9, 21, 79, 133]. However, some fine-tuning meth
ods (e.g., TALLRec [3]) also involve the form of instructions. In order to avoid ambiguity, we consider the instructions
of a single task template as fine-tuning settings, and the instructions of multiple tasks as instruction-tuning settings. As
the backbone of recommender systems, researchers generally use T5 or Flan-T5 for various recommendation scenarios
such as rating, ranking, retrieving, explanation generation and news recommendation. Besides Flan-T5, RecRanker [72]
is also proposed to instruction tuning LLaMA2 as the ranker for top𝑘 recommendation. By instantiating appropriate
instructions for each recommendation task, this setting can also be summarized into our framework.
4.1.3 Model Architecture. For the architecture design of LLMs leveraged as recommender systems, we consider
three mainstream architectures as summarized in [140], i.e., encoder-decoder, prefix decoder, and causal decoder. The
recommendation scenarios for each framework are introduced in what follows.
• The encoder-decoder architecture adopts two Transformer blocks as the encoder and decoder, which is the basis of
BERT [97]. In line with existing literature on LLM-based recommender systems, the bidirectional property of the encoder
decoder architecture allows LLMs to easily customize encoders and decoders towards recommendation (e.g., dual-encoder
considering both ids and text [79]), and conveniently adapt to multiple recommendation tasks [9, 21, 57, 79, 133]. A few
LLMs use the encoder-decoder architecture, and the typical one is the series of T5 and its variant Flan-T5 [10].
• The prefix decoder architecture is also the decoder-only architectures, which is known as non-causal decoder. It
can bidirectionally encode the prefix tokens like the encoder-decoder architecture, and perform unidirectional attention
on the generated tokens like the causal decoder architecture. One of the representative LLMs based on the prefix
decoder architecture is ChatGLM [129] and its variants, and researchers have attempted to explore the recommendation
performance of ChatGLM as one of the benchmarks in related work [69].
• The causal decoder architecture has been widely adopted in various LLMs, and the series of GPT [4] and LLaMA [95]
are the most representative models. It uses the unidirectional attention mask, and only decoders are deployed to process
both the input and output tokens. Due to the popularity of the causal decoder architecture, most LLM-based recommender
15

• The causal decoder architecture has been widely adopted in various LLMs, and the series of GPT [4] and LLaMA [95]
are the most representative models. It uses the unidirectional attention mask, and only decoders are deployed to process
both the input and output tokens. Due to the popularity of the causal decoder architecture, most LLM-based recommender

systems employ this framework to adapt to different recommendation tasks such as click-through rate predictions,
sequential recommendation and conversational recommender systems.

4.1.4 Parameter Scale. To meet the diverse needs of different users, the most typical variant of LLMs is the parameter
scale. In general, open-source models have multiple parameter sizes to choose from, and larger parameter sizes
generally mean better capabilities [95, 140]. But meanwhile, the corresponding computational and spatial complexity
will also increase. Considering the memory and efficiency issues for experiments, researchers in the field of LLM-based
recommender systems generally use LLMs with parameters no more than 10B, while the performance of LLMs with
larger parameters remains to be further explored in the field of recommender systems [40].
4.1.5 Context Length. Another property closely related to user needs is the length of the input context. The inability
to handle inputs with longer contexts means that decision-making cannot be accurately developed, thereby limiting the
model capabilities [96, 140]. When the user input exceeds the limited length, the input will be truncated, so sufficient
context length is crucial for the user experience [12, 32, 74]. However, different lengths of context inputs imply different
model architectures and parameters. When expanding the context length of a model, it often leads to higher time and
memory complexity. To address the length limitation of LLMs, existing methods either selectively discard previous
contexts using sliding windows [120], or only sample a portion of the context for retrieval augmentation [45, 62], or
employ small models without emergence ability. Despite recent strategies, the limitation of context length has not yet
been truly resolved. Considering economic and efficiency issues, existing LLMs including open-source and closed-source
models only provide a limited number of context length options. In this paper, we mainly focus on several classic
lengths, i.e., 2K, 4K, 8K, 16K and 32K (K is the abbreviation for one thousand, similarly hereinafter).

# 4.2 Research Questions and Experimental Setup

4.2.1 Research Questions. In this section, we conduct experiments to verify the impact of different factors of LLMs on
recommendation results. Specifically, we focus on the following four research questions.
• RQ1: What are the differences between the recommendation ability of LLMs and traditional recommenders?
• RQ2: How do the different attributes of LLMs, including the public availability, model architectures, parameter
scales, and context lengths affect the recommendation performance and inference time?
• RQ3: What are the similarities and differences in recommendation results of LLMs with different tuning strategies?
Is the LLM after instruction tuning more suitable for recommendation tasks?
• RQ4: What are the limitations of leveraging LLMs as recommender systems?
4.2.2 Evaluated Models. As for the experimental settings, we consider the following baselines and LLMs.
• Random: Random baseline recommends the 𝑘 (k=20 in this section) candidate items in a random order, which is
the basic situation to evaluate the metric values of each dataset.
• Pop: Pop method always ranks the candidate items based on their interaction times with users in the training set.
We consider it as the fully-trained method since it uses the statistical information of datasets.
• BPR [84]: BPR is one of the typical traditional models that utilize matrix factorization for recommendation. It is
trained in the pair-wise paradigm a.k.a., BPR training loss without considering temporal information.
16

• SASRec [39]: SASRec is a sequential recommendation model based on the backbone of the classic self-attention
network a.k.a., Transformer [97], and achieves comparable performance among sequential models.
• ChatGPT: ChatGPT is a closed-source large-scale pre-trained language model developed by OpenAI. Note that
OpenAI has released interfaces of ChatGPT on March 1 and June 13, 2023, respectively. Considering the up-to-date
requirements, we adopt the version on June 13 for recency, the same as GPT-4.
• GPT-4: GPT-4 is the latest generation of closed-source natural language processing model launched by the OpenAI
company. Experiments have shown that GPT-4 is significantly superior to ChatGPT in multiple tasks.
• Flan-T5 [10]: Flan-T5 is an open-source language model based on the encoder-decoder architecture T5 released by
Google [81]. Flan-T5 is extended from T5 by a multi-task fine-tuning paradigm i.e., instruction tuning to enhance
the generalization of different tasks. There are multiple variants of Flan-T5 in terms of parameters, including
Flan-T5-Small (80M), Flan-T5-Base (250M), Flan-T5-Large (780M), Flan-T5-XL (3B) and Flan-T5-XXL (11B). Since
the first three models are too small to meet the requirements of LLMs discussed in this paper (1B, B is short for
billion and the same below), we consider the Flan-T5-XL and Flan-T5-XXL for comparison.
• ChatGLM [129]: ChatGLM is an open-source bilingual dialogue LLM that supports both Chinese and English,
based on the General Language Model (GLM) [129] with the prefix decoder architecture. The team released the
second version ChatGLM2 and the third version ChatGLM3 in June 2023 and October 2023, respectively.
• LLaMA [95]: LLaMA is an open-source language model introduced by MetaAI from the causal decoder architecture
with four sizes (6B, 13B, 35B and 65B). Due to its outstanding performance and low computational cost, LLaMA has
received much attention from researchers so far, and Vicuna [8] is one of the most popular variants by extending
LLaMA. To further improve abilities of LLaMA, MetaAI released LLaMA2 as the next generation of open-source
large language models in July, 2023. In addition to the regular version, MetaAI also provide the chatting version of
LLaMA2 (i.e., LLaMA2-chat) [96], and it is specifically tuned for the dialogue scenario by Reinforcement Learning
with Human Feedback (RLHF).

# 4.3 Observations and Discussion

4.3.1 LLMs Compared to Traditional Recommenders (RQ1). Compared to traditional models based on collaborative
filtering of interacted data in fully-trained settings, we provide the cold-start recommendation performance of LLMs
in zero-shot settings. In what follows, we introduce the empirical findings on the recommendation effect of different
models from three aspects, i.e., recommendation performance, the impact of historical item sequences and inference time.
• Recommendation performance of LLMs. As shown in Table 4, we provide the fully-trained results of four traditional
methods, as well as the zero-shot recommendation performance of various LLMs. For traditional recommenders, BPR [84]
based on collaborative filtering is significantly better than Pop based on popularity, and the sequential recommendation

4.3.1 LLMs Compared to Traditional Recommenders (RQ1). Compared to traditional models based on collaborative
filtering of interacted data in fully-trained settings, we provide the cold-start recommendation performance of LLMs
in zero-shot settings. In what follows, we introduce the empirical findings on the recommendation effect of different
models from three aspects, i.e., recommendation performance, the impact of historical item sequences and inference time.

in zero-shot settings. In what follows, we introduce the empirical findings on the recommendation effect of different
models from three aspects, i.e., recommendation performance, the impact of historical item sequences and inference time.
• Recommendation performance of LLMs. As shown in Table 4, we provide the fully-trained results of four traditional
methods, as well as the zero-shot recommendation performance of various LLMs. For traditional recommenders, BPR [84]
based on collaborative filtering is significantly better than Pop based on popularity, and the sequential recommendation
model SASRec [39] combined with attention mechanism and temporal information is significantly better than BPR,
which is consistent with the results in existing literature [39, 84, 144]. It is worth noting that for the 20 candidate
items, LLMs that rely on natural languages cannot completely recall all items in most cases, and several models with
poor abilities can only output a dozen items, greatly limiting the accuracy of recommendation results. Therefore,
recall@20 indicates the ability of LLMs to memorize, re-rank and output candidate items, and several approaches such
as the re-generation method [56] and probability distribution outputs [128] have been proposed to improve the recall

Table 4. Overall performance of different models on recommendation. We consider both the fully-trained settings for traditiona models and zero-shot settings for LLMs. Note that there are always ground-truth items in the randomly selected 20 candidates, s the ideal recall@20 equals to 1. “IT” is for “Inference Time”, and we record the average inference time for each user measured i seconds (s). “N/A” is the abbreviation for “Not Applicable” since the inference time of closed-source models is unknown.

MovieLens-1M
Amazon-Books
model
context
length
param.
size
recall@20
ndcg@1
ndcg@10
IT (s)
recall@20
ndcg@1
ndcg@10
IT (s)
Fully-trained settings for traditional models
Random
-
0
1.0000
0.0300
0.2081
0.01
1.0000
0.0350
0.2628
0.01
Pop
-
1
1.0000
0.1800
0.4841
0.03
1.0000
0.1000
0.2672
0.03
BPR
-
<1M
1.0000
0.2550
0.5743
0.04
1.0000
0.2950
0.6236
0.04
SASRec
-
<1M
1.0000
0.6400
0.7916
1.07
1.0000
0.6800
0.8305
1.49
Zero-shot settings for LLMs
Closed-source LLMs
4K
-
0.9583
0.1817
0.3985
n/a
0.9850
0.2467
0.4276
n/a
ChatGPT
16K
-
0.9600
0.1500
0.3735
n/a
0.9800
0.2400
0.4032
n/a
GPT-4
8K
-
0.9900
0.3100
0.5828
n/a
1.0000
0.3300
0.5631
n/a
Open-source LLMs with the encoder-decoder architecture
3B
0.0050
0.0000
0.0016
3.51
0.0000
0.0000
0.0000
4.33
Flan-T5
0.5k
11B
0.0050
0.0000
0.0016
5.21
0.0000
0.0000
0.0000
8.28
Open-source LLMs with the prefix decoder architecture
ChatGLM
2K
6B
0.7750
0.0300
0.1945
19.12
0.7000
0.0350
0.2026
19.52
ChatGLM2
32K
6B
0.1900
0.0450
0.0885
11.06
0.1600
0.0250
0.0680
13.62
2K
6B
0.6900
0.0950
0.2762
7.95
0.6550
0.0450
0.2273
10.79
ChatGLM3
32K
6B
0.7750
0.0700
0.2579
14.06
0.7050
0.0550
0.2068
16.89
Open-source LLMs with the causal decoder architecture
7B
0.2700
0.0350
0.1068
9.11
0.2650
0.0200
0.0992
9.35
13B
0.2500
0.0300
0.1028
9.51
0.2250
0.0250
0.0867
9.94
33B
0.3900
0.0400
0.1328
92.88
0.2950
0.0350
0.1015
106.61
LLaMA
2K
65B
0.5300
0.0450
0.1913
171.39
0.3750
0.0500
0.1253
182.51
7B
0.2650
0.0500
0.1078
8.17
0.4400
0.0550
0.1559
11.12
Vicuna
2K
13B
0.4100
0.0550
0.1507
9.26
0.4800
0.0700
0.1813
12.72
7B
0.2350
0.0500
0.0888
9.97
0.5600
0.0650
0.1648
10.01
13B
0.4500
0.0700
0.1215
15.64
0.5100
0.0750
0.2150
14.44
LLaMA2
4K
70B
0.7600
0.1250
0.2918
24.38
0.6600
0.1150
0.2912
24.63
7B
0.7950
0.0900
0.2744
9.80
0.7050
0.1500
0.3217
9.86
13B
0.8050
0.1650
0.3866
14.80
0.7500
0.1650
0.3411
12.25
LLaMA2
(chat)
4K
70B
0.9550
0.2430
0.4344
23.07
0.8850
0.2230
0.3827
25.01
performance. While for the traditional models, all candidate items can be recalled (i.e., recall@20 equals to 1). Because
the candidate items are not recalled completely, the recommendation effect of a few LLMs (e.g., Flan-T5 and ChatGLM)

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9b04/9b0453bc-5891-4df6-b82a-e9d131e1ae4c.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8f70/8f709133-0e0b-4607-a14e-f2e3ccf69816.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4a3c/4a3c7cd5-0c76-432d-913b-93c8ef907894.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) MovieLens-1M
</div>
is not even as good as the random baseline. For most LLMs, the zero-shot recommendation performance is not as good
as the baseline method Pop based on popularity of interactions in the dataset [12, 32, 68]. However, the powerful LLMs
like ChatGPT and LLaMA-70B (chat) [96] can achieve better results than Pop in zero-shot settings. Furthermore, GPT-4
can even perform better than the fully-trained matrix factorization model BPR on two datasets, indicating the potential
of LLMs to serve as the backbone of recommender systems. In addition, the significant differences between the results
of LLMs demonstrate the importance of selecting appropriate LLMs for downstream recommendation tasks [40, 68, 69].
• The impact of historical item sequences. As for ranking tasks in Fig. 2(a), the recently interacted historical items
are used as the user representations. However, there is no standard value for the number of items that represent users.
To analyze the impact of historical item sequences on the recommendation performance, we conduct experiments to
explore the recommendation effect of the sequential recommender SASRec [39] and the closed-source LLM ChatGPT
with different numbers of historical interactions. For the fully-trained SASRec, the maximum length of the historical
item sequence will affect the model framework and prediction results [31, 39]. In order to ensure the model uniformity,
we fix the model checkpoint with the historical item sequence of 50 as the “SASRec (fixed)” for comparison, and evaluate
the recommendation performance (NDCG@10 [36]) with the number of historical items at 1, 5, 10, 20, 30, 40 and 50,
respectively. For ChatGPT and GPT-4, we get zero-shot results with different items to verify whether the powerful
LLMs can deal with the long context for recommendation. As illustrated in Fig. 3, with the increasing number of
historical items, the results of SASRec improve steadily, while the the recommendation performance of LLMs changes
little. Consistent conclusions on two datasets can be drawn that even if LLMs can accept more historical items for user
representations, increasing the number of historical items does not bring significant gains in the recommendation
performance. The performance trends of LLMs show that the increased historical item sequence is not fully utilized
by the language model, indicating the importance of selecting appropriate item sequences to represent users, and the
inadequacy of LLMs for user interest mining. To improve the mining of user interest for LLMs, approaches such as
retrieval augmentation [62] and prompting strategies [108, 125] can be used, which will be analyzed in Section 5.
• Inference time of LLMs. In the actual deployment of recommender algorithms, the inference efficiency is the
decisive factor for industrial applications [91, 98]. In general, the inference time of models is closely related to the size
of parameters. As shown in the last column of Table 4, for the lightweight traditional recommenders, the inference
time of SASRec is about 1 second for each user. However for LLMs, except that closed-source models cannot accurately

<div style="text-align: center;">(b) Amazon-Books
</div>
obtain inference time due to limitations of the API, the inference time of open-source models takes ne
seconds for one prediction, leading to an unacceptable time delay in practical applications.

Observations on the Overall Recommendation Performance of LLMs
• In zero-shot scenarios, LLMs have cold-start capabilities, and GPT-4 even surpasses collaborative filtering
models. However, all LLMs are inferior to fully-trained sequential recommendation models.
• Even if LLMs can accept more historical items for user representations, increasing the number of historical
items does not bring significant gains in the recommendation performance.
• Compared to recommenders, the inference time of LLMs is unacceptable to be used for real applications.

4.3.2 LLMs on Recommendations w.r.t. Four Aspects (RQ2). For different LLMs, differences in public availability and
model architecture will lead to different recommendation scenarios, results and inference time [40, 69]. For the same
LLM, the parameter scale and context length also affect the efficiency and effectiveness of language models [140].
Therefore, we explore the impact of different LLMs on recommendations from four aspects, namely public availability,
model architecture, parameter scale and context length as follows.
• Public availability. As shown in Table 4, closed-source models achieve significantly better results than the open
source models in the cold-start scenario, but they cannot outperform fully-trained sequential models. In terms of LLMs,
ChatGPT with zero-shot settings has comparable recommendation performance with the fully-trained Pop especially on
the sparse Amazon-Books dataset, indicating the fundamental ability of LLMs on recommendation tasks. Furthermore,
the upgraded GPT-4 exceeds ChatGPT by a large margin due to its strong zero-shot generalization ability. The superior
zero-shot performance of GPT sheds lights on leveraging LLMs for recommendation. However, the open-source models
always get poor results compared to GPT-4 in the zero-shot settings, while LLaMA2-chat-70B has the comparable
recommendation performance with ChatGPT. The reason is that open-source models lack comprehensive cold-start
capabilities, and their strength lies in the ability to integrate domain knowledge through strategies such as prompt
tuning. In line with previous studies on LLMs [32, 40, 68, 74], employing a closed-source model in cold-start scenarios
yields better results, while an open-source model is more flexible and easy to use when tuning is needed [3, 54, 72, 141].
• Model architecture. For the model architecture, Flan-T5 [10] based on the encoder-decoder architecture has almost
no ability to recommend items in the cold-start setting, as its training corpus does not involve specialized instructions of
our task. Trained with prompts of recommendations, the encoder-decoder architecture is suitable for prompt tuning and
instruction tuning [9, 21, 133]. Similarly, the first and second versions of ChatGLM [129] based on the prefix decoder
perform poorly on the zero-shot ranking task, and are not as good as Vicuna and LLaMA2 based on the causal decoder.
However, the third version of ChatGLM, i.e., ChatGLM3 has comparable recommendation performance with Vicuna [8]
and LLaMA2 [96], which further indicates the importance of selecting an advanced foundation model. In terms of the
series of LLaMA models [95], LLaMA2 is better than Vicuna, and Vicuna is better than LLaMA, which is related to
their training data and release time. Furthermore, the chat version of LLaMA2, i.e., LLaMA2-chat is a series that uses
conversational dialog instructions to fine-tune LLaMA2 [96], which is more suitable for our tasks in ranking settings.
Therefore, the results of LLaMA2-chat are significantly improved compared with LLaMA2, and LLaMA2-70B-chat even
achieves better performance than the closed-source model ChatGPT. Generally speaking, researchers prefer to study

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b3e9/b3e9c0aa-0db9-4fb4-b8f8-cee19408523d.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) MovieLens-1M
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9d34/9d348003-a437-4ce7-b2ee-8276f3fbd677.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0543/0543e44e-7245-4eef-9de7-d393795f8e1b.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) MovieLens-1M
</div>
<div style="text-align: center;">Fig. 5. The recommendation performance of LLMs w.r.t. the context len
</div>
<div style="text-align: center;">recommendation tasks based on the causal decoder framework such as LLaMA, and the second version of LLaMA has
better generalization ability than the first version in recommendation tasks.
</div>
recommendation tasks based on the causal decoder framework such as LLaMA, and the second version of LLaMA has a
better generalization ability than the first version in recommendation tasks.
• Parameter scale. It is widely recognized that the larger the parameter size, the more powerful the LLMs [32, 40, 140],
the same applies in the field of recommender systems. To compare the effect and efficiency of LLMs w.r.t. the parameter
scale, we compare the recommendation performance and inference time of LLaMA [95], Vicuna [8], LLaMA2 [96],
and LLaMA2-chat at different parameter scales in Fig. 4. As the scale of parameters enlarges, the recommendation
performance and inference time of LLMs steadily increases, and both datasets (Fig. 4(a) and Fig. 4(b)) have consistent
conclusions. Therefore, it is necessary to consider the trade-off between performance and efficiency when choosing the
parameter scale. Moreover, the performance improvement on the increasing scale of LLaMA2 is more significant than
that of LLaMA, indicating that the scale effect of LLMs has to do with the capabilities of the base model.
• Context length. Different LLMs have different maximum input limitations [95, 96, 140], and a longer context input
means LLMs can accommodate more historical items for recommendation. However, it remains to be explored whether
the maximum input length of LLMs will affect the recommendation results when the length limitation is not exceeded.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7230/7230b1de-cf31-434c-bf46-760fd10cce22.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Amazon-Books
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/78e7/78e70252-a678-4a85-91b5-3f63fbe619d9.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Amazon-Books
</div>
<div style="text-align: center;">Table 5. Overall performance of LLMs on CTR predictions. There are three settings, i.e., zero-shot setting without fine-tuning parameter-efficient fine-tuning (PEFT) setting with a few parameters tuned, and fine-tuning (FT) setting with all parameters tuned
</div>
dataset
ml-1m
Amazon-Books
model
zero-shot
PEFT
FT
zero-shot
PEFT
FT
LLaMA-7B
0.4683
0.5479
0.6658
0.6488
0.8262
0.8469
Alpaca-LoRA-7B
0.5264
0.5767
0.6702
0.6558
0.8440
0.8533
LLaMA2-7B
0.5284
0.6133
0.6457
0.6754
0.8550
0.8542
LLaMA2-chat-7B
0.5255
0.6275
0.6731
0.7174
0.8560
0.8660
Therefore, we conduct experiments to investigate the differences in the recommendation performance between the
two length versions of ChatGPT (4K and 16K) and ChatGLM3 (2K and 32K). As shown in Fig. 5, expanding the length
limitation of LLMs does not necessarily mean the better recommendation performance, while there is a slight decrease
in NDCG@10. Furthermore, when the maximum input of LLMs remains unchanged, increasing the historical input of
users results in insignificant gains as shown in Fig. 3. Therefore, the key to the recommendation problem is to enable
LLMs to effectively utilize the information within the limited context [62, 125], and a suitable context length selection
for LLMs as recommender systems is worthy of deep consideration.

• As for the public availability, closed-source models outperform the open-source models in terms of the
recommendation performance, but have poorer flexibility.
• As for the model architecture, different frameworks are adapted to different recommendation tasks and
fine-tuning strategies, while LLMs with the casual decoder architecture are still mainstream.
• As for the parameter scale, the larger the parameter scale, the better the recommendation ability.
• As for the context length, a longer maximum context length leads to worse recommendation results.

4.3.3 Comparisons of Tuning Strategies for LLMs (RQ3). Due to the fact that LLMs are not customized to recommender
systems during the training process, it is insufficient to only consider the zero-shot recommendation performance in cold
start scenarios [24, 40, 69, 72]. In order to explore the impact of different training strategies of LLMs on recommendations,
we compare the click-through rate prediction performance of four LLaMA-based LLMs on two datasets. Specifically,
we consider three training settings of LLMs, i.e.,  the zero-shot setting without fine-tuning, Parameter-Efficient Fine
Tuning (PEFT) setting (we use the LoRA [33] here) with a few parameters tuned, and Fine-Tuning (FT) setting with all
parameters tuned, and summarize empirical conclusions from three aspects: overall performance of different settings, the
impact of instruction tuning, and the impact of training data.
• Overall performance of different settings. As shown in Table 5, we can see that the results of fine-tuning
LLMs (PEFT and FT) on only 256 samples are significantly better than the zero-shot performance in cold-start scenarios,
and empirical findings are consistent across four LLMs (LLaMA, Alpaca-LoRA, LLaMA2, LLaMA2-chat) on both datasets.
Furthermore, considering the two kinds of fine-tuning strategies, the performance of the fine-tuning setting is even
better than that of the PEFT setting since more parameters are tuned [33]. In addition to recommendation effects,
training efficiency is also a performance that deserves attention. Therefore, we compare the training time of the two
fine-tuning strategies on two datasets. As shown in Fig. 6, the time for parameter-efficient fine-tuning is significantly
22

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e78b/e78bf74f-ade7-4eec-9aaa-95affd496a0b.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6341/634153c1-ddcb-4b04-822a-626abd5ac6e8.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) MovieLens-1M
</div>
<div style="text-align: center;">Fig. 6. The comparison of training time w.r.t. the parameter-efficient fine-tuning (PEFT) and fine-tuning (FT) strategies.
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn