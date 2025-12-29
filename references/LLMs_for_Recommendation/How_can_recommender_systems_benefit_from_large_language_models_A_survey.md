# How Can Recommender Systems Benefit from Large Language Models: A Survey

JIANGHAO LIN ∗, Shanghai Jiao Tong University, China XINYI DAI ∗, Noah’s Ark Lab, Huawei, China YUNJIA XI, Shanghai Jiao Tong University, China WEIWEN LIU and BO CHEN, Noah’s Ark Lab, Huawei, China HAO ZHANG and YONG LIU, Noah’s Ark Lab, Huawei, Singapore CHUHAN WU and XIANGYANG LI, Noah’s Ark Lab, Huawei, China CHENXU ZHU and HUIFENG GUO, Noah’s Ark Lab, Huawei, China YONG YU, Shanghai Jiao Tong University, China RUIMING TANG †, Noah’s Ark Lab, Huawei, China WEINAN ZHANG †, Shanghai Jiao Tong University, China

With the rapid development of online services and web applications, recommender systems (RS) have become increasingly indispensable
for mitigating information overload and matching users’ information needs by providing personalized suggestions over items. Although
the RS research community has made remarkable progress over the past decades, conventional recommendation models (CRM) still
have some limitations, e.g., lacking open-domain world knowledge, and difficulties in comprehending users’ underlying preferences
and motivations. Meanwhile, large language models (LLM) have shown impressive general intelligence and human-like capabilities
for various natural language processing (NLP) tasks, which mainly stem from their extensive open-world knowledge, logical and
commonsense reasoning abilities, as well as their comprehension of human culture and society. Consequently, the emergence of LLM
is inspiring the design of recommender systems and pointing out a promising research direction, i.e., whether we can incorporate LLM
and benefit from their common knowledge and capabilities to compensate for the limitations of CRM. In this paper, we conduct a
comprehensive survey on this research direction, and draw a bird’s-eye view from the perspective of the whole pipeline in real-world
recommender systems. Specifically, we summarize existing research works from two orthogonal aspects: where and how to adapt
LLM to RS. For the “WHERE” question, we discuss the roles that LLM could play in different stages of the recommendation pipeline,
i.e., feature engineering, feature encoder, scoring/ranking function, user interaction, and pipeline controller. For the “HOW” question,
we investigate the training and inference strategies, resulting in two fine-grained taxonomy criteria, i.e., whether to tune LLM or not
during training, and whether to involve conventional recommendation models for inference. Detailed analysis and general development
paths are provided for both “WHERE” and “HOW” questions, respectively. Then, we highlight the key challenges in adapting LLM to
RS from three aspects, i.e., efficiency, effectiveness, and ethics. Finally, we summarize the survey and discuss the future prospects. To
further facilitate the research community of LLM-enhanced recommender systems, we actively maintain a GitHub repository for
papers and other related resources in this rising direction 1.

CCS Concepts: • Information systems → Recommender system

# CCS Concepts: • Information systems → Recommender systems

1 https://github.com/CHIANGEL/Awesome-LLM-for-RecSys * Jianghao Lin and Xinyi Dai are the co-first authors. † Ruiming Tang and Weinan Zhang are the co-corresponding authors.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. Manuscript submitted to ACM

Additional Key Words and Phrases: Recommender Systems, Large Language Models
ACM Reference Format:
Jianghao Lin ∗, Xinyi Dai ∗, Yunjia Xi, Weiwen Liu, Bo Chen, Hao Zhang, Yong Liu, Chuhan Wu, Xiangyang Li, Chenxu Zhu, Huifeng
Guo, Yong Yu, Ruiming Tang †, and Weinan Zhang †. 2018. How Can Recommender Systems Benefit from Large Language Models: A
Survey. In Proceedings of Make sure to enter the correct conference title from your rights confirmation emai (Conference acronym ’XX).
ACM, New York, NY, USA, 46 pages. https://doi.org/XXXXXXX.XXXXXXX

Jianghao Lin ∗, Xinyi Dai ∗, Yunjia Xi, Weiwen Liu, Bo Chen, Hao Zhang, Yong Liu, Chuhan Wu, Xiangyang Li, Chenxu Zhu, Huifeng
Guo, Yong Yu, Ruiming Tang †, and Weinan Zhang †. 2018. How Can Recommender Systems Benefit from Large Language Models: A
Survey. In Proceedings of Make sure to enter the correct conference title from your rights confirmation emai (Conference acronym ’XX).
ACM, New York, NY, USA, 46 pages. https://doi.org/XXXXXXX.XXXXXXX

# 1 INTRODUCTION

With the rapid development of online services, recommender systems (RS) have become increasingly important to
match users’ information needs [27, 46, 122] and mitigate information overload [54, 119]. They offer personalized
suggestions across diverse domains such as e-commerce [182], movie [53], music [189], etc. Despite the varied forms of
recommendation tasks (e.g., top𝑁 recommendation, and sequential recommendation), the common learning objective
for recommender systems is to estimate a given user’s preference towards each candidate item, and finally arrange a
ranked list of items presented to the user [117, 240].
Despite the remarkable progress of conventional recommender systems over the past decades, their recommendation
performance is still suboptimal, hampered by two major drawbacks as follows: (1) Conventional recommender systems
are domain-oriented systems generally built based on discrete ID features within specific domains [242]. Therefore,
they lack open-domain world knowledge to obtain better recommendation performance (e.g., enhancing user interest
modeling and item content understanding), and transferring abilities across different domains and platforms [15, 56, 129].
(2) Conventional recommender systems often aim to optimize specific user feedback such as clicks and purchases in a
data-driven manner, where the user preference and underlying motivations are often implicitly modeled based on user
behaviors collected online. As a result, these systems might lack recommendation explainability [12, 48], and cannot
fully understand the complicated and volatile intent of users in various contexts. Moreover, users cannot actively guide
the recommender system to follow their requirements and customize recommendation results by providing detailed
instructions in natural language [43, 216, 219].
With the emergence of large foundation models in recent years, they provide promising and universal insights
when handling many challenging problems in the data mining field [14, 45, 196]. A representative form is the large
language model (LLM), which has shown impressive general intelligence in various language processing tasks due to
their huge memory of open-world knowledge, the ability of logical and commonsense reasoning, and the awareness
of human society and culture [8, 73, 277]. By using natural language as a universal information carrier, knowledge in
different forms, modalities, domains, and platforms can be generally integrated, exploited, and interpreted. Consequently,
the rise of large language models is inspiring the design of recommender systems, i.e., whether we can incorporate
LLM and benefit from their common knowledge to address the aforementioned ingrained drawbacks of conventional
recommender systems.
Recently, RS researchers and practitioners have made many pioneer attempts to employ LLM in current recommenda
tion pipelines, and have achieved notable progress in boosting the performance of different canonical recommendation
processes such as feature modeling [242] and ranking [4]. There exist several related survey works that delve into
the potential of LLM for general recommender systems. Wu et al. [237] conduct a review on both discriminative
and generative LLMs for recommendation with different tuning strategies. Fan et al. [42] focus on the pretraining,
finetuning and prompting approaches when leveraging LLM for recommendation. Huang et al. [72] investigate the
recommendation foundation models from aspects of both different model types and various downstream tasks. Other
2

<div style="text-align: center;">Large Language Models
(LLM)
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c961/c9616276-595f-43c6-9163-5b6b5fbc7eef.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c721/c72145be-c79e-47a5-82e3-4a4f01a18d1f.png" style="width: 50%;"></div>
<div style="text-align: center;">Recommender Systems
(RS)
</div>
Fig. 1. The decomposition of our core research question about adapting large language models to recommender systems. We analyze the question from two orthogonal perspectives: (1) where to adapt LLM, and (2) how to adapt LLM. Note that CRM stands for conventional recommendation model.

works also concentrate on one specific aspect of recommender systems with LLM enhancements, e.g., prompting
strategy [245], generative recommendation [29, 97, 110], and explainable recommendation [12]. However, it still lacks
a bird’s-eye view of where and how recommender systems can embrace large language models and integrate them
into the overall recommendation pipeline, which is essential in building a technique map to systematically guide the
research, practice, and service in LLM-empowered recommendation.
Different from existing surveys on this topic, in this paper, we propose a systematic view of the LLM-enhanced
recommendation, from the angle of the whole pipeline in industrial recommender systems. LLM is currently utilized
in various stages of recommendation systems and are integrated with current systems via different techniques. To
conduct a comprehensive review of latest research progress, as shown in Figure 1, we propose research questions about
LLM-enhanced recommender systems from the following two perspectives:
• “WHERE” question focuses on where to adapt LLM for RS, and discusses the roles that LLM could play at different
stages of current recommender system pipeline, i.e., feature engineering, feature encoder, scoring/ranking function,
user interaction, and pipeline controller.
• “HOW” question centers on how to adapt LLM for RS, where two orthogonal taxonomy criteria are carried out: (1)
whether we will freeze the parameters of the large language model during the training phase, and (2) whether we
will involve conventional recommendation models (CRM) during the inference phase.
From the two perspectives, we propose feasible and instructive suggestions for the evolution of existing online
recommendation platforms in the era of large language models 23.
The rest of this paper is organized as follows. In Section 2, we briefly introduce the background and preliminary for
recommender systems and large language models. Section 3 and Section 4 thoroughly analyze the aforementioned

2 To provide a thorough survey and a clear development path, we broaden the scope of large language models, and bring those relatively smaller language models (e.g., BERT [31], GPT2 [168]) into the discussion as well. 3 We focus on works that leverage LLM together with their pretrained parameters to handle textual features via prompting, and exclude works that simply apply pretraining paradigms from NLP domains to pure ID-based traditional recommendation models (e.g., BERT4Rec [191]). Interested readers can refer to [128, 255].

taxonomies from two perspectives (i.e., “WHERE” and “HOW”), followed by detailed discussion and analysis of the
general development path. In Section 5, we highlight the key challenges and future directions for the adaption of LLM to
RS from three aspects (i.e., efficiency, effectiveness, and ethics), which mainly arise from the real-world applications
of recommender systems. Finally, Section 6 concludes this survey and draws a hopeful vision for future prospects in
research communities of LLM-enhanced recommender systems. Furthermore, we give a comprehensive look-up table
of related works that adapt LLM to RS in Appendix A (i.e., Table 1), attaching the detailed information for each work,
e.g., the stage that LLM is involved in, LLM backbone, and LLM tuning strategy, etc.

# 2 BACKGROUND AND PRELIMINARY

Before elaborating on the detail of our survey, we would like to introduce the following background and basic concepts:
(1) the general pipeline of modern recommender systems based on deep learning techniques, and (2) the general
workflow and concepts for large language models.

# 2.1 Modern Recommender Systems

# The core task of recommender systems is to provide a ranked list of items [𝑖 𝑘] 𝑁 𝑘 = 1,𝑖 𝑘 ∈I for the user 𝑢 ∈U given a
certain context 𝑐, where I and U are the universal sets of items and users, respectively. Note that scenarios like next
tem prediction are special cases for such a formulation with 𝑁 = 1. We denote the goal as follows:

The core task of recommender systems is to provide a ranked list of items [𝑖 𝑘] 𝑁 𝑘 = 1,𝑖 𝑘 ∈I for the user 𝑢 ∈U
certain context 𝑐, where I and U are the universal sets of items and users, respectively. Note that scenarios li
item prediction are special cases for such a formulation with 𝑁 = 1. We denote the goal as follows:

[𝑖 𝑘] 𝑁 𝑘 = 1 ← RS (𝑢,𝑐, I), 𝑢 ∈U,𝑖 𝑘 ∈I.

As shown in Figure 2, the modern deep learning based recommender systems can be characterized as an information
ycle that encompasses six key stages: (1) Data Collection, where the users’ feedback data is gathered; (2) Feature
ngineering, which involves preparing and processing the collected raw data; (3) Feature Encoder, where data features
re transformed into neural embeddings; (4) Scoring/Ranking Function, which selects and orders the recommended items;
5) User Interaction, which determines how users engage with the recommendations; and finally, (6) Recommendation
ipeline Controller, which serves as the central mechanism tying all the stages above together in a cohesive process.
Next, we will briefly go through each of the stages as follows:
• Data Collection. The data collection stage gathers both explicit and implicit feedback from online services by
presenting recommended items to users. The explicit feedback indicates direct user responses such as ratings, while
the implicit feedback is derived from user behaviors like clicks, downloads, and purchases. In addition to gathering
user feedback, the data to be collected also encompasses a range of raw features including item attributes, user
demographics, and contextual information. The raw data is stored in the database in certain formats such as JSON for
further processing. It is worth noting that, in this paper, the data collection mainly refers to the process of collecting
real-world user behavior data from online services without any manual transformation or data synthesis.
• Feature Engineering. Feature engineering is the process of selecting, manipulating, transforming, and augmenting
the raw data collected online into structured data that is suitable as inputs of neural recommendation models. As
shown in Figure 2, the major outputs of feature engineering consist of various forms of features, which will be then
encoded by feature encoders of different modalities, e.g., language models for textual features, vision models for
visual features, and conventional recommendation models (CRM) for ID features. Note that the feature engineering
stage stands for the general data engineering process, which does not only conduct feature-level manipulation, but
also involves sample-level synthesizing and augmentation for enhancements of both training and offline evaluation.

As shown in Figure 2, the modern deep learning based recommender systems can be characterized as an information
cycle that encompasses six key stages: (1) Data Collection, where the users’ feedback data is gathered; (2) Feature
Engineering, which involves preparing and processing the collected raw data; (3) Feature Encoder, where data features
are transformed into neural embeddings; (4) Scoring/Ranking Function, which selects and orders the recommended items;
(5) User Interaction, which determines how users engage with the recommendations; and finally, (6) Recommendation
Pipeline Controller, which serves as the central mechanism tying all the stages above together in a cohesive process.
Next, we will briefly go through each of the stages as follows:

• Data Collection. The data collection stage gathers both explicit and implicit feedback from online services by
presenting recommended items to users. The explicit feedback indicates direct user responses such as ratings, while
the implicit feedback is derived from user behaviors like clicks, downloads, and purchases. In addition to gathering
user feedback, the data to be collected also encompasses a range of raw features including item attributes, user
demographics, and contextual information. The raw data is stored in the database in certain formats such as JSON for
further processing. It is worth noting that, in this paper, the data collection mainly refers to the process of collecting
real-world user behavior data from online services without any manual transformation or data synthesis.
• Feature Engineering. Feature engineering is the process of selecting, manipulating, transforming, and augmenting
the raw data collected online into structured data that is suitable as inputs of neural recommendation models. As
shown in Figure 2, the major outputs of feature engineering consist of various forms of features, which will be then
encoded by feature encoders of different modalities, e.g., language models for textual features, vision models for
visual features, and conventional recommendation models (CRM) for ID features. Note that the feature engineering
stage stands for the general data engineering process, which does not only conduct feature-level manipulation, but
also involves sample-level synthesizing and augmentation for enhancements of both training and offline evaluation.

(1)

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2a5c/2a5c3222-16e1-4f72-a794-046910b9a599.png" style="width: 50%;"></div>
<div style="text-align: center;">Neural Embeddings
</div>
<div style="text-align: center;">Structured Data
</div>
<div style="text-align: center;">Raw Data
</div>
<div style="text-align: center;">Fig. 2. The illustration of deep learning based recommender system pipeline. We characterize the modern recommender system as an information cycle that consists of six stages: data collection, feature engineering, feature encoder, scoring/ranking function, user interaction, and recommendation pipeline controller, which are denoted by different colors.
</div>
• Feature Encoder. Generally speaking, the feature encoder takes as input the processed features from the feature
engineering stage, and generates the corresponding neural embeddings for scoring/ranking functions in the next
stage. Various encoders are employed depending on the data modality. Typically, this process is executed as an
embedding layer for one-hot encoded categorical features in standard recommendation models. Features of other
modalities, such as text, vision, video, or audio, are further used and encoded to enhance content understanding.
• Scoring/Ranking function. Scoring/Ranking function serves as the core part of recommendation to select or
rank the top-relevant items to satisfy users’ information needs based on the neural embeddings generated by the
feature encoders. Researchers develop various neural methods to precisely estimate the user preference and behavior
patterns based on various techniques, e.g., collaborative filtering [59, 190], sequential modeling [19, 132], graph
neural networks [211, 221], etc.
• User Interaction. User interaction refers to the way we represent the recommended items to the target user, and the
way users give their feedback back to the recommender system. While traditional recommendation pages basically
involve a single list of items, various complex and multi-modal scenarios are recently proposed and studied [260]. For
example, conversational recommendation provides natural language interface and enables multi-round interactive
recommendation for the user [194]. Besides, multi-block page-level user interactions are also widely considered for
nested user feedback [46, 178].
• Recommendation Pipeline Control. Pipeline controller monitors and controls the operations of the whole
recommendation pipeline mentioned above. It can even provide fine-grained control over different stages for
recommendation (e.g., matching, ranking, reranking), or decide to combine different downstream models and APIs to
accomplish the final recommendation tasks.
It is worth noting that the multi-stage formulation above serves as a general overview of the modern recommendation
pipeline, and some of the stages might be skipped, linked, or merged with specific modeling techniques. For example, if
we adapt LLM as the scoring/ranking function, the input of the recommender should be textual data from the feature

<div style="text-align: center;">Ranked Item List
</div>
engineering [4, 120], instead of neural embeddings from the feature encoder, i.e., we skip the feature encoder stage.
Moreover, some works [103, 275] would explore advanced techniques to inject domain knowledge of neural embeddings
from CRM into LLM as the scoring/ranking function, i.e., we obtains the inputs by merging the outcomes from both
feature engineering and feature encoder stages.

# 2.2 Large Language Models

Language models aim to conduct the probabilistic modeling of natural languages to predict the word tokens given a
specific textual context. Nowadays, most language models are built based on transformer-like [204] architectures to
proficiently model the context dependency for human languages. They are first pretrained on a massive amount of
unlabeled text data, and then further finetuned with task-oriented data for different downstream applications. These
pretrained language models (PLM) can be mainly classified into three categories: encoder-only models like BERT [31],
decoder-only models like GPT [168], and encoder-decoder models like T5 [169].
Large language models (LLM) are the scaled-up derivatives of traditional pretrained language models mentioned
above, in terms of both model sizes and data volumes, e.g., GPT-3 [8], PaLM [21], LLaMA [202], ChatGLM [40, 263].
A typical LLM usually consists of billion-level or even trillion-level parameters, and is pretrained on much larger
volumes of textual corpora with up-to trillions of tokens crawled from various Internet sources like Wikipedia, GitHub,
ArXiv, etc. As illustrated by the scaling law [65, 87], the scaling up of model size, data volume and training scale can
continuously contribute to the growth of model performance for a wide range of downstream NLP tasks. Furthermore,
researchers find that LLM can exhibit emergent abilities, e.g., few-shot in-context learning, instruction following and
step-by-step reasoning, when the model size continues to scale up and reaches a certain threshold [229]
LLM has revolutionized the field of NLP by demonstrating impressive capabilities in understanding natural languages
and generating human-like texts. Moreover, LLM has gone beyond the field of NLP and shown remarkable potential
in various deep learning based applications, such information system [287], education [99], finance [238] and health
care [152, 197]. Therefore, recent studies start to investigate the application of LLM to recommender systems. Equipped
with the extensive open-world knowledge and powerful emergent abilities like reasoning, LLM is able to analyze the
individual preference based on user behavior sequences, and promote the content understanding and expansion for
items, which can largely enhance the recommendation performance [4, 25, 242, 250]. Besides, LLM can also support
more complex scenarios like conversational recommendation [48], explainable recommendation [12], as well as task
decomposition and tool usage (e.g., search engines) [225] for recommendation enhancements.

# 3 WHERE TO ADAPT LARGE LANGUAGE MODELS

Based on the decomposition of modern recommender systems discussed in Section 2.1, we answer the “WHERE”
question by elaborating on the adaptation of LLM to different parts of the recommendation pipeline: (1) feature
engineering, (2) feature encoder, (3) scoring/ranking function, (4) user interaction, and (5) pipeline controller. It is worth
noting that, the utilization of LLM in the same research work may involve multiple stages of the recommendation
pipeline due the multi-task nature of LLM. For example, LLM is leveraged in both stages of feature engineering and
scoring/ranking function in CUP [201].

# 3.1 LLM for Feature Engineering

In the feature engineering stage, LLM takes as inputs the original features (e.g., item descriptions, user profiles, and
user behaviors), and generates auxiliary textual features for data augmentation with varied goals, e.g., enriching the

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ab5e/ab5e5a72-61be-425f-bc4e-bfa842059830.png" style="width: 50%;"></div>
training data, alleviating the long-tail problem, etc. Different prompting strategies are employed to make full use of
the open-world knowledge and reasoning ability exhibited by LLM. According to the type of data augmentation, the
research works of this line can be mainly classified into two categories: (1) user- and item-level feature augmentation,
(2) instance-level training sample generation.

3.1.1 User- and Item-level Feature Augmentation. Equipped with powerful reasoning ability and open-world knowledge,
LLM is often treated as a flexible knowledge base [140]. Hence, it can provide auxiliary features for better user
preference modeling and item content understanding. As a representative, KAR [242] adopts LLM to generate the
user-side preference knowledge and item-side factual knowledge, which serve as the plug-in features for downstream
conventional recommendation models. SAGCN [124] introduces a chain-based prompting approach to uncover semantic
aspect-aware interactions, which provides clearer insights into user behaviors at a fine-grained semantic level. CUP [201]
adopts ChatGPT to summarize each user’s interests with a few short keywords according to the user review texts.
In this way, the user profiling data is condensed within 128 tokens and thus can be further encoded with small-scale
language models that are constrained by the context windows size (e.g., 512 for BERT [31]). Moreover, instead of
using a frozen LLM for feature augmentation, LLaMA-E[183] and EcomGPT [111] finetune the base large language
models for various downstream generative tasks in e-commerce scenarios, e.g., product categorization and intent
speculation. Other works also utilize LLM to further enrich the training data from different perspectives, e.g., text

refinement [39, 137, 279], knowledge graph completion and reasoning [15, 24, 224, 232], attribute generation [7, 92, 253]
and user interest modeling [22, 36, 142, 176].

3.1.2 Instance-level Sample Generation. Apart from feature-level augmentations, LLM is also leveraged to generate
synthetic samples, which enrich the training dataset [151] and improve the model prediction quality [123, 195]. GReaT [6]
tunes a generative language model to synthesize realistic tabular data as augmentations for the training phase. Carranza
et al. [11] explore to train a differentially private (DP) large language model for synthetic user query generation, in order
to address the privacy problem in recommender systems. ONCE [129] applies manually designed prompts to obtain
additional news summarization, user profiles, and synthetic news pieces for news recommendation. AnyPredict [227]
leverages LLM to consolidate datasets with different feature fields, and align out-domain datasets for a shared target
task. TF-DCon [235] aims to compress and condensate the training data by using LLM to generate fewer amount of
synthetic samples from views of both user history and item content. Zhang et al. [265] further attempt to incorporate
multiple large language models as agents to simulate the fine-grained user communication and interaction for more
realistic recommendation scenarios. Moreover, RecPrompt [123] and PO4ISR [195] propose to perform automatic
prompt template optimization with powerful LLM (e.g., ChatGPT or GPT4), and therefore iteratively improve the
recommendation performance with gradually better textual inputs for LLM-based recommenders. BEQUE [156] finetunes
and deploys LLM for query rewriting in e-commercial scenarios to bridge the semantic gaps inherent in the semantic
matching process, especially for long-tail queries. Li et al. [104] use Chain-of-Thought [230] (CoT) technology to
leverage LLM as agent to emulate various demographic profiles for robust and efficient query rewriting.

# 3.2 LLM as Feature Encoder

In conventional recommender systems, the structured data are usually formulated as one-hot encodings, and a embedding
layer is adopted as the feature encoder to obtain dense embeddings. With the emergence of language models, researchers
propose to adopt LLM as auxiliary textual feature encoder to gain two major benefits: (1) further enriching the user/item
representations with semantic information for the later neural recommendation models; (2) achieving cross-domain 4
recommendation with natural language as the bridge, where ID feature fields might not be shared.

3.2.1 Representation Enhancement. For item representation enhancement, LLM is leveraged as feature encoder for
scenarios with abundant textual features available (e.g., item title, body text, detailed description), including but
not limited to: document ranking [134, 290], news recommendation [130, 177, 233, 234, 256], tweet search [272],
tag selection [57], For While the item content is generally static, the user interest is highly dynamic and keeps
evolving over time, therefore requiring sequential modeling over the fast-evolving user behaviors and underlying
preferences [38, 85, 158, 282]. For example, U-BERT [164] ameliorates the user representation by encoding review texts
into a sequence of dense vectors via BERT [31], followed by specially designed attention networks for user interest
modeling. LLM4ARec [98] uses GPT2 [168] to extract personalized aspect terms and latent vectors from user profiles
and reviews to better assist recommendations. In some special cases, the semantic representation encoded by LLM is not
directly used as the input for the later scoring/ranking function. Instead, it is converted into a sequence of discrete tokens
through quantization to adapt to scoring/ranking functions that require discrete inputs (e.g., generative recommendation).
TIGER [173] proposes to apply vector quantization techniques [203, 254, 262] over the semantic item representations to
further compress each item into a tuple of discrete semantic tokens. Hence, the sequential recommendation can be

expressed as a sequence modeling task over a list of discrete tokens, where classical transformer [204] architectures
can be employed. Based on the idea of item vector quantization, LMIndexer [82] designs a self-supervised semantic
indexing framework to capture the item’s semantic representation and the corresponding semantic tokens at the same
time in an end-to-end manner.

3.2.2 Unified Cross-domain Recommendation. Apart from the user/item representation improvement, adopting LLM as
feature encoder also enables transfer learning and cross-domain recommendation, where natural language serves as the
bridge to align the heterogeneous information from different domains [100, 109, 213]. ZESRec [34] applies BERT [31] to
convert item descriptions into universal semantic representations for zero-shot recommendation. In UniSRec [67], the
item representations are learned for cross-domain sequential recommendation via a fixed BERT model followed by a
lightweight MoE-enhanced network. Built upon UniSRec, VQ-Rec [66] introduces vector quantization techniques to
better align the textual embeddings generated by LLM to the recommendation space. Uni-CTR [47] leverages layer-wise
semantic representations from a shared LLM to sufficiently capture the commonalities among different domains,
which leads to better multi-domain recommendation. Other works [52, 199] leverage unified cross-domain textual
embeddings from a fixed LLM (e.g., ChatGLM [40], Sheared-LLaMA [243]) to tackle scenarios with cold-start users/items
or low-frequency long-tail features. Fu et al. [44] further explore layerwise adapter tuning on large language models to
obtain better embeddings over textual features from different domains.

# 3.3 LLM as Scoring/Ranking Function

The ultimate goal of the scoring/ranking stage is highly tied with the general purpose of recommender systems as
discussed in Section 2.1, i.e., to provide a ranked list of items [𝑖 𝑘] 𝑁 𝑘 = 1,𝑖 𝑘 ∈I for target user 𝑢 ∈U, where I and U
are the universal set of items and users (next item prediction is a special case where 𝑁 = 1). When directly adapting
LLM as the scoring/ranking function, such a goal could be achieved through various kinds of tasks for LLM (e.g., rating
prediction, item ID generation). According to different tasks that LLM solves, we classify related research works into
three categories: (1) item scoring task, (2) item generation task, and (3) hybrid task. Moreover, as discussed in Section 2.1,
when adapting LLM as the scoring/ranking function, the input for this stage can be textual data, neural embeddings
from other encoders, or a combination of both. In this section, we mainly focus on the task formulations and solution
paradigms of LLM for scoring & ranking. We would omit the input format unless necessary.

3.3.1 Item Scoring Task. In item scoring tasks, the large language model serves as a pointwise function 𝐹 (𝑢,𝑖), ∀ 𝑢 ∈
U, ∀ 𝑖 ∈I, which estimates the utility score of each candidate item 𝑖 for the target user 𝑢. Here U and I denote the
universal set of users and items, respectively. The final ranked list of items is obtained by sorting the utility score
calculated between the target user 𝑢 and each item 𝑖 in the candidate set C:
C ← Pre-filter (𝑢, I),

C ← Pre-filter (𝑢, I),

[𝑖 𝑘] 𝑁 𝑘 = 1 ← Sort ({𝐹 (𝑢,𝑖) | ∀ 𝑖 ∈I}), 𝑁 ≤|C|,

where C is the candidate set obtained via a pre-filter function (e.g., the retrieval and pre-ranking models for the ranking
stage). The pre-filtering is conducted to reduce the number of candidate items, thus saving the computational cost. The
pre-filter can be an identity-mapping function (i.e., C = I) for the first retrieval stage for recommender systems.
Without loss of generality, the large language model takes as inputs the discrete tokens of textual prompt 𝑥, and
generates the target token ˆ 𝑡 as the output for either the masked token in masked language modeling or the next token

(2)

# in causal language modeling. The process can be formulated as follows:
ℎ = LLM (𝑥),

in causal language modeling. The process can be formulated as follows
ℎ = LLM (𝑥),

ℎ = LLM (𝑥),
𝑠 = LM_Head (ℎ) ∈ R 𝑉,
𝑝 = Softmax (𝑠) ∈ R 𝑉,
ˆ 𝑡 ∼ 𝑝,

where ℎ is the final representation, 𝑉 is the vocabulary size, and ˆ 𝑡 is the predicted token sampled from the probabilit

where ℎ is the final representation, 𝑉 is the vocabulary size, and ˆ 𝑡 is the predicted token sampled from the probability
distribution 𝑝.
However, the item scoring task requires the model to do pointwise scoring for a given user-item pair (𝑢,𝑖), and the
output should be a real number ˆ 𝑦 = 𝐹 (𝑢,𝑖), instead of generated discrete tokens ˆ 𝑡. The output ˆ 𝑦 should fall within a
certain numerical range to indicate the user preference, e.g., ˆ 𝑦 ∈[0, 1] for click-through rate (CTR) estimation and
ˆ 𝑦 ∈[0, 5] for rating prediction. There are three major approaches to address such an issue that the output requires
continuous numerical values while LLM produces discrete tokens.
The first type of solution [86, 88, 103, 115, 125, 207, 210, 286, 289] adopts the single-tower paradigm [165, 271]. To be
specific, they directly abandon the language modeling decoder head (i.e., LM_Head (·)), and feed the final representation
ℎ of LLM in Eq. 3 into a delicately designed projection layer to calculate the final score ˆ 𝑦 for classification or regression
tasks, i.e.,

distribution 𝑝.
However, the item scoring task requires the model to do pointwise scoring for a given user-item pair (𝑢,𝑖), and the
output should be a real number ˆ 𝑦 = 𝐹 (𝑢,𝑖), instead of generated discrete tokens ˆ 𝑡. The output ˆ 𝑦 should fall within a
certain numerical range to indicate the user preference, e.g., ˆ 𝑦 ∈[0, 1] for click-through rate (CTR) estimation and
ˆ 𝑦 ∈[0, 5] for rating prediction. There are three major approaches to address such an issue that the output requires
continuous numerical values while LLM produces discrete tokens.
The first type of solution [86, 88, 103, 115, 125, 207, 210, 286, 289] adopts the single-tower paradigm [165, 271]. To be
specific, they directly abandon the language modeling decoder head (i.e., LM_Head (·)), and feed the final representation
ℎ of LLM in Eq. 3 into a delicately designed projection layer to calculate the final score ˆ 𝑦 for classification or regression
tasks, i.e.,
ˆ 𝑦 = 𝐹 (𝑢,𝑖) = MLP (ℎ), (4)
where MLP (short for multi-layer perceptron) is the projection layer. The input prompt 𝑥 needs to contain information
from both the user 𝑢 and item 𝑖 to support the preference estimation based on one single latent representation ℎ.
E4SRec [103] constructs personalized prompts with the help of pre-learned user & item ID embeddings for precise
preference estimation. FLIP [210] and ClickPrompt [115] propose to conduct fine-grained knowledge alignment and
fusion over the semantic and collaborative information in parallel and stacking paradigms, respectively. CER [167]
reinforces the coherence between recommendations and their natural language explanations to improve the rating
prediction performance. Kang et al. [86] finetune the large language model for rating prediction in a regression manner,
which exhibits a surprising performance by scaling the model size of finetuned LLM up to 11 billion. Other typical
examples in this line of research include: LSAT [184], BERT4CTR [207], CLLM4Rec [286], and PTab [125].
Similar to the first method, the second type of solution [94, 138, 198, 200, 201, 249] also discards the decoder head
of LLM. However, what sets it apart is that it adopts the popular two-tower structure [58, 59, 221] in conventional
recommender systems. They maintain both two separate towers to obtain the representations for user and item
respectively, and the preference score is calculated via a certain distance metric between the two representations:
ˆ 𝑦 = 𝐹 (𝑢,𝑖) = 𝑑 (𝑇 𝑢 (𝑥 𝑢),𝑇 𝑖 (𝑥 𝑖)), (5)
where 𝑑 (·, ·) is the distance metric function (e.g., cosine similarity, L2 distance). 𝑇 𝑢 (·) and 𝑇 𝑖 (·) are the user and item
towers that consist of LLM backbones to extract the useful knowledge representations from both user and item texts
(i.e., 𝑥 𝑢 and 𝑥 𝑖). In this line of works, different auxiliary structures are designed to augment the dual-side information
with LLM. For example, CoWPiRec [249] applies word graph neural networks to item texts within the user behavior
sequence to amplify the semantic information correlation. By employing the encoder-decoder LLM, TASTE [138] first
encodes each user behavior into a soft prompt vector and then leverages the decoder to extract the user preference
from the sequence of soft prompts. Other typical examples include: RecFormer [94], LLM-Rec [198], and CUP [201].
10

where 𝑑 (·, ·) is the distance metric function (e.g., cosine similarity, L2 distance). 𝑇 𝑢 (·) and 𝑇 𝑖 (·) are the user and item
towers that consist of LLM backbones to extract the useful knowledge representations from both user and item texts
(i.e., 𝑥 𝑢 and 𝑥 𝑖). In this line of works, different auxiliary structures are designed to augment the dual-side information
with LLM. For example, CoWPiRec [249] applies word graph neural networks to item texts within the user behavior
sequence to amplify the semantic information correlation. By employing the encoder-decoder LLM, TASTE [138] first
encodes each user behavior into a soft prompt vector and then leverages the decoder to extract the user preference
from the sequence of soft prompts. Other typical examples include: RecFormer [94], LLM-Rec [198], and CUP [201].

(4)

(5)

Different from the aforementioned two solutions that both replace the original language modeling decoder head
(i.e., LM_Head (·)) with manually designed predictive modules, the last type of solution [13, 61, 63, 116, 120, 140, 145,
159, 166, 186, 192, 209, 236, 239, 241, 274, 276, 280, 288] proposes to preserve the decoder head and perform preference
estimation based on the probability distribution 𝑝 ∈ R 𝑉. TALLRec [4], ReLLa [120], CoLLM [275], PromptRec [239],
BTRec [63] and CR-SoRec [159] append a binary question towards the user preference after the textual description
of user profile, user behaviors, and target item, and therefore convert the item scoring task into a binary question
answering problem. Then, they can intercept the estimated score 𝑠 ∈ R 𝑉 or probability 𝑝 ∈ R 𝑉 in Eq. 3 and conduct a
bidimensional softmax over the corresponding logits of the binary key answer words (i.e., the token used to denote
label, for example, Yes/No) for pointwise scoring:

where 𝑝 𝑌𝑒𝑠 and 𝑝 𝑁𝑜 denote the logits for “Yes” and “No” tokens, respectively. Other typical examples that extract
the softmax probabilities of corresponding label tokens for item scoring include TabLLM [61], Prompt4NR [276], and
GLRec [236]. Moreover, another line of research intends to concatenate the item description (e.g., title) to the user
behavior history with different templates, and estimates the score by calculating the overall perplexity [145, 166],
log-likelihood [181, 186], or joint probability [274] of the prompting text as the final predicted score ˆ 𝑦 for user preference.
Besides, Zhiyuli et al. [280] instruct LLM to predict the user rating in a textual manner, and restrict the output format
as a value with two decimal places through manually designed prompts.
3.3.2 Item Generation Task. In item generation tasks, the large language model serves as a generative function 𝐹 (𝑢) to
directly produce the final ranked list of items, requiring only one forward of function 𝐹 (𝑢). Generally speaking, the
item generation task highly relies on the intrinsic reasoning ability of LLM to infer the user preference and generate
the ranked item list, the process of which can be formulated as:

where 𝑝 𝑌𝑒𝑠 and 𝑝 𝑁𝑜 denote the logits for “Yes” and “No” tokens, respectively. Other typical examples that extract
the softmax probabilities of corresponding label tokens for item scoring include TabLLM [61], Prompt4NR [276], and
GLRec [236]. Moreover, another line of research intends to concatenate the item description (e.g., title) to the user
behavior history with different templates, and estimates the score by calculating the overall perplexity [145, 166],
log-likelihood [181, 186], or joint probability [274] of the prompting text as the final predicted score ˆ 𝑦 for user preference.
Besides, Zhiyuli et al. [280] instruct LLM to predict the user rating in a textual manner, and restrict the output format
as a value with two decimal places through manually designed prompts.
3.3.2 Item Generation Task. In item generation tasks, the large language model serves as a generative function 𝐹 (𝑢) to
directly produce the final ranked list of items, requiring only one forward of function 𝐹 (𝑢). Generally speaking, the
item generation task highly relies on the intrinsic reasoning ability of LLM to infer the user preference and generate
the ranked item list, the process of which can be formulated as:
[𝑖 𝑘] 𝑁 𝑘 = 1 = 𝐹 (𝑢), 𝑠.𝑡. 𝑖 𝑘 ∈I. (7)
According to whether a set of candidate items is provided for LLM to accomplish the item generation task, we can
categorize the related solutions into two classes: (1) open-set item generation, and (2) closed-set item generation.
In open-set item generation tasks [3, 33, 50, 56, 70, 71, 76, 80, 95, 96, 105, 113, 121, 147, 163, 180, 253, 264, 278, 285],
LLM is required to directly generate the ranked item list that the user might prefer according to the user profile and
behavior history without a given candidate item set. Since the candidate items are not provided in the input prompt, the
large language model is actually not aware of the universal item pool I, thus bringing the generative hallucination
problem [147], where the generated items might fail to match the exact items in the item pool I. Therefore, apart
from the design of input prompt templates [68, 107] and finetuning algorithms [96], the post-processing operations for
item grounding and matching after the item generation are also required to overcome the generative hallucination
problem [147]. We formulate the process as follows:

[𝑖 𝑘] 𝑁 𝑘 = 1 = 𝐹 (𝑢), 𝑠.𝑡. 𝑖 𝑘 ∈I.

According to whether a set of candidate items is provided for LLM to accomplish the item generation task, we can
categorize the related solutions into two classes: (1) open-set item generation, and (2) closed-set item generation.
In open-set item generation tasks [3, 33, 50, 56, 70, 71, 76, 80, 95, 96, 105, 113, 121, 147, 163, 180, 253, 264, 278, 285],
LLM is required to directly generate the ranked item list that the user might prefer according to the user profile and
behavior history without a given candidate item set. Since the candidate items are not provided in the input prompt, the
large language model is actually not aware of the universal item pool I, thus bringing the generative hallucination
problem [147], where the generated items might fail to match the exact items in the item pool I. Therefore, apart
from the design of input prompt templates [68, 107] and finetuning algorithms [96], the post-processing operations for
item grounding and matching after the item generation are also required to overcome the generative hallucination
problem [147]. We formulate the process as follows: � �

�� � �
where Match (·, ·) is the matching function, ˆ 𝑖 𝑘 is the LLM-generated items, and 𝑖 𝑘 is the actual item matched from I
according to ˆ 𝑖 𝑘. LANCER [80] employs knowledge-enhanced prefix tuning for generation ground and further applies
cosine similarity to match the encoded representation of generated item text with the universal item pool I. Di Palma

(6)

(7)

(8)

et al. [33] leverage ChatGPT for user interest modeling and next item title generation with Damerau-Levenshtein
distance [148] for item matching.
Apart from generating the items in textual manners, another line of research focuses on aligning the language space
with the ID-based recommendation space, and therefore enables LLM to generate the item IDs directly. For instance, Hua
et al. [71] explore better ways for item indexing (e.g., sequential indexing, collaborative indexing) in order to enhance
the performance of such index generation tasks. LightLM [147] designs a lightweight LLM with carefully designed
user & item indexing, and applies constrained beam search for open-set item ID generation. Besides, LLaRA [113]
represents items in LLM’s input prompts using a novel hybrid approach that integrates ID-based item embeddings
from traditional recommenders with textual item features. Other typical works for open-set item generation include:
GenRec [78], TransRec [121], LC-Rec [278], ControlRec [163], and POD [96].
In closed-set item generation tasks [18, 50, 51, 68, 70, 123, 140, 143, 188, 195, 206, 217, 226, 244, 251, 258, 266], LLM
is required to rank or select from a given candidate item set. That is, we will first employ a lightweight retrieval model
to pre-filter the universal item set I into a limited number of candidate items denoted as C = {𝑖 𝑗} 𝐽 𝑗 = 1, 𝐽 ≪|I|. The
number of candidate items is usually set up to 20 due to the context window limitation of LLM. The content of candidate
items is then presented in the input prompt for LLM to generate the ranked item list, which can be formulated as:
C ← Pre-filter (𝑢, I),
[𝑖 𝑘] 𝑁 𝑘 = 1 ← LLM (𝑢, C), 𝑁 ≤|C|, (9)
For example, LlamaRec [258] adopts LRURec [259] as the retriever, and finetunes LLaMA2 for listwise ranking over the
pre-filtered items. DRDT [226] ranks the given candidates with iterative multi-round reflection to to gradually refine
the ranked list. LiT5 [188] proposes to distill the zero-shot ranking ability from a proficient LLM (e.g., RankGPT4 [193])
into a relatively smaller one (e.g., T5-XL [169]). AgentCF [267] incorporates LLM as the recommender by simulating
user-item interactions in recommender systems through agent-based collaborative filtering. Other typical examples
include: JobRecoGPT [51], InstructMK [206], RecPrompt [123], PO4ISR [195], etc.
In comparison of these two tasks, open-set generation tasks generally suffer from the generative hallucination
problem, where the generated items might fail to match the exact items in the universal item pool. Therefore, the
post-generation matching function is heavily required, which increases the inference overhead and might even hurt
the final recommendation performance, especially for scenarios with item texts that largely differ from the language
distribution of LLM. On the contrary, closed-set generation tasks use a lightweight retrieval model as the pre-filter
to provide a clear set of candidate items, and therefore the large language model is able to mitigate the hallucination
problem. However, the introduction of candidate items in the input prompt of LLM can cause other problems. Firstly,
LLM cannot handle a large number of candidates (usually less than 20) due to the context window limitation, and the

et al. [33] leverage ChatGPT for user interest modeling and next item title generation with Damerau-Levenshtein
distance [148] for item matching.
Apart from generating the items in textual manners, another line of research focuses on aligning the language space
with the ID-based recommendation space, and therefore enables LLM to generate the item IDs directly. For instance, Hua
et al. [71] explore better ways for item indexing (e.g., sequential indexing, collaborative indexing) in order to enhance
the performance of such index generation tasks. LightLM [147] designs a lightweight LLM with carefully designed
user & item indexing, and applies constrained beam search for open-set item ID generation. Besides, LLaRA [113]
represents items in LLM’s input prompts using a novel hybrid approach that integrates ID-based item embeddings
from traditional recommenders with textual item features. Other typical works for open-set item generation include:
GenRec [78], TransRec [121], LC-Rec [278], ControlRec [163], and POD [96].
In closed-set item generation tasks [18, 50, 51, 68, 70, 123, 140, 143, 188, 195, 206, 217, 226, 244, 251, 258, 266], LLM
is required to rank or select from a given candidate item set. That is, we will first employ a lightweight retrieval model
to pre-filter the universal item set I into a limited number of candidate items denoted as C = {𝑖 𝑗} 𝐽 𝑗 = 1, 𝐽 ≪|I|. The
number of candidate items is usually set up to 20 due to the context window limitation of LLM. The content of candidate
items is then presented in the input prompt for LLM to generate the ranked item list, which can be formulated as:
C ← Pre-filter (𝑢, I),
(9)

For example, LlamaRec [258] adopts LRURec [259] as the retriever, and finetunes LLaMA2 for listwise ranking over the
pre-filtered items. DRDT [226] ranks the given candidates with iterative multi-round reflection to to gradually refine
the ranked list. LiT5 [188] proposes to distill the zero-shot ranking ability from a proficient LLM (e.g., RankGPT4 [193])
into a relatively smaller one (e.g., T5-XL [169]). AgentCF [267] incorporates LLM as the recommender by simulating
user-item interactions in recommender systems through agent-based collaborative filtering. Other typical examples
include: JobRecoGPT [51], InstructMK [206], RecPrompt [123], PO4ISR [195], etc.
In comparison of these two tasks, open-set generation tasks generally suffer from the generative hallucination
problem, where the generated items might fail to match the exact items in the universal item pool. Therefore, the
post-generation matching function is heavily required, which increases the inference overhead and might even hurt
the final recommendation performance, especially for scenarios with item texts that largely differ from the language
distribution of LLM. On the contrary, closed-set generation tasks use a lightweight retrieval model as the pre-filter
to provide a clear set of candidate items, and therefore the large language model is able to mitigate the hallucination
problem. However, the introduction of candidate items in the input prompt of LLM can cause other problems. Firstly,
LLM cannot handle a large number of candidates (usually less than 20) due to the context window limitation, and the
final recommendation performance can somehow be limited by the retrieval model (i.e., pre-filter). Moreover, Ma et al.
[143] and Hou et al. [68] reveal that shuffling the order of candidate items in the prompt can affect the ranking output,
leading to unstable recommendation results. The aforementioned issues of closed-set generation tasks intrinsically
stem from the existence of candidate item set in the input prompt, which can be well solved in open-set generation
tasks. In summary, we can observe that the open-set and closed-set generation tasks have complementary strengths
and weaknesses compared with each other. Hence, the choice between them in practical applications actually depends
on specific situations and problems we meet in real-world scenarios.

(9)

3.3.3 Hybrid Task. In hybrid tasks, the large language model serves in a multi-task manner, where both the item scoring
and generation tasks could be handled by a single LLM through a unified language interface. The basis for supporting
this hybrid functionality is that large language models are inherent multi-task learners [8, 153]. P5 [49], M6-Rec [25]
and InstructRec [268] tune the encoder-decoder models for better alignment towards a series of recommendation
tasks including both item scoring and generation tasks via different prompting templates. RecRanker [139] combines
the pointwise scoring, pairwise comparison and listwise ranking tasks to explore the potential of LLM for top-N
recommendation. BDLM [270] bridges the information gap between the domain-specific models and the general large
language models for hybrid recommendation tasks via an information sharing module with memory storage mechanism.
UniMP [231] builds a unified large foundation for multiple tasks in personalized systems, e.g., preference prediction,
personalized search. Other works [26, 126, 193] manually design task-specific prompts to call a unified central LLM
(e.g., ChatGPT API) to perform multiple tasks, including but not restricted to pointwise rating prediction, pairwise item
comparison, and listwise ranking list generation. There also exist benchmarks (e.g., LLMRec [127], OpenP5 [246]) that
test the LLM-based recommenders on various recommendation tasks like rating prediction, sequential recommendation,
and direct recommendation.

# 3.4 LLM for User Interaction

In many of practical applications, recommending is a one-turn interaction process, where the system monitors the user
behaviors (e.g., click and purchase) over time and then presents a tailored set of relevant items in certain pre-defined
situations. Such a one-turn interaction lacks effective and versatile ways to acquire user interests and detect the user’s
current situation or needs in complex scenarios. To this end, the advent of large language models presents a promising
alternative, by offering a more active and adaptive form of user interaction Deng et al. [30], Wang et al. [220, 223].
Instead of relying solely on the past user behaviors passively, LLM could engage in real-time interactions with the users
to gather more nuanced natural language feedback about their preferences.
In general, the user interaction based on LLM in recommendation is commonly formed as a multi-turn dialogue,
which is covered in conversational recommender systems [30, 216, 223, 284]. During such a dialogue, LLM provides an
unprecedented richness in understanding users’ interests and requirements by integrating context in conversation and
applying the extensive open-world knowledge. LLM can support a recommender to make highly relevant and tailored
recommendations through eliciting the current preferences of user, providing explanations for the item suggestions,
or processing feedback by users on the made suggestions [75]. In other words, the introduction of large language
models makes recommender systems more feasible and user-friendly in terms of user interaction. Specifically, from
the perspective of interactive content [101, 283], the modes of LLM-based user interaction can be categorized into (1)
task-oriented user interaction, and (2) open-ended user interaction .
3.4.1 Task-oriented User Interaction. The task-oriented user interaction [30, 175, 212, 247, 273, 284] supposes that the
user has a clear intent and the recommender system needs to support the user’s decision making process or assist the
user in finding relevant items. To be specific, LLM is integrated as a component of the recommender system, specially
aiming at analyzing user intentions. As a typical work, TG-ReDial [284] proposes to incorporate topic threads to enforce
natural semantic transitions towards the recommendation and develops a topic-guided conversational recommendation
method. It deploys three BERT [31] modules to encode user profiles, dialogue history, and track conversation topics,
respectively. Then, the encoded features are fed into a pre-set recommendation module to recommend items, followed
by a GPT2 [168] module to aggregate the encoded knowledge for response generation. After each turn, the results are
13

gathered and will be used to support the next round of dialogue interaction, such as understanding changes in user
interest and analyzing user feedback, etc. The subsequent works roughly follow a similar process for task-oriented user
interaction. While earlier works attempt to manage the dialogue understanding and response generation with relatively
small language models (e.g., BERT and GPT2), recent works start to incorporate billion-level large language models
for better conversational recommendation and improving the satisfaction of user interaction. MuseChat [37] builds a
multi-modal LLM based on Vicuna-7B [20] to provide reasonable explanation for the music recommendation during the
user dialogue. Liu et al. [136]  leverage the complementary collaboration between conversational RS and LLM for e
commercial pre-sales dialogue understanding and generation. He et al. [60] construct a conversational recommendation
dataset with more diverse textual contexts, and find that LLM is able to outperform finetuned traditional conversational
recommenders in zero-shot settings. Other typical works for task-oriented user interaction include: MESE [247],
KECR [175], UniMIND [30], VRICR [273], TCP [212].
3.4.2 Open-ended User Interaction. The task-oriented user interaction draws a strong assumption that the user engages
in the recommender system with specific goals to seek certain items. Differently, the open-ended user interaction [90,
174, 216, 219, 222, 223] assumes that the user’s intent is vague, and the system needs to gradually acquire user interests or
guide the user through interactions (including topic dialogue, chitchat, QA, etc.) to achieve the goal of recommendation
eventually. Consequently, the role of LLM for open-ended user interaction is no longer limited to a simple component
for dialogue encoding and response generation as discussed in Section 3.4.1. Instead, LLM plays a key role in driving the
interaction process by leading and acquiring the user interests for final recommendation. Specifically, BARCOR [219]
proposes a unified framework based on BART [91] to first conduct user preference elicitation, and then perform
response generation with recommended items, which aims to maximize the mutual information between conversation
interaction and item recommendation. T5-CR [174] focuses on user interaction modeling and formulates conversation
recommendation as a language generation problem. It adopts T5 [170] to achieve dialogue context understanding,
user preference elicitation, item recommendation and response generation in an end-to-end manner. Specifically, it
adopts a special token symbol as the trigger to generate recommended item during the response generation. Wang et al.
[222] investigate the ability of ChatGPT to converse with user for item recommendation and explanation generation
through manually designed prompts without any demonstration (i.e., zero-shot prompting). Then, they utilize LLM
as an auxiliary user interaction component for dialogue understanding and user preference elicitation. Other related
research works include: UniCRS [223], RecInDial [216], and TtW [90].

KECR [175], UniMIND [30], VRICR [273], TCP [212].
3.4.2 Open-ended User Interaction. The task-oriented user interaction draws a strong assumption that the user engages
in the recommender system with specific goals to seek certain items. Differently, the open-ended user interaction [90,
174, 216, 219, 222, 223] assumes that the user’s intent is vague, and the system needs to gradually acquire user interests or
guide the user through interactions (including topic dialogue, chitchat, QA, etc.) to achieve the goal of recommendation
eventually. Consequently, the role of LLM for open-ended user interaction is no longer limited to a simple component
for dialogue encoding and response generation as discussed in Section 3.4.1. Instead, LLM plays a key role in driving the
interaction process by leading and acquiring the user interests for final recommendation. Specifically, BARCOR [219]
proposes a unified framework based on BART [91] to first conduct user preference elicitation, and then perform
response generation with recommended items, which aims to maximize the mutual information between conversation
interaction and item recommendation. T5-CR [174] focuses on user interaction modeling and formulates conversation
recommendation as a language generation problem. It adopts T5 [170] to achieve dialogue context understanding,
user preference elicitation, item recommendation and response generation in an end-to-end manner. Specifically, it
adopts a special token symbol as the trigger to generate recommended item during the response generation. Wang et al.
[222] investigate the ability of ChatGPT to converse with user for item recommendation and explanation generation
through manually designed prompts without any demonstration (i.e., zero-shot prompting). Then, they utilize LLM
as an auxiliary user interaction component for dialogue understanding and user preference elicitation. Other related
research works include: UniCRS [223], RecInDial [216], and TtW [90].

# 3.5 LLM for Pipeline Controller

As the model size scales up, LLM tends to exhibit emergent behaviors that may not be observed in previous smaller
language models, e.g., few-shot in-context learning, instruction following, step-by-step reasoning, and tool usage [229,
277]. With such emergent abilities, LLM is no longer just a part of the recommender system mentioned above, but
could actively participate in the pipeline control over the system, possibly leading to a more interactive and explainable
recommendation process [83]. Chat-REC [48] leverages ChatGPT to bridge the conversational interface and traditional
recommender systems, where it is required to infer user preferences, decide whether or not to call the backend
recommendation API, and further modify (e.g., filter and rerank) the returned item candidates before presenting them
to the user. These operations enable LLM to step beyond the role for user interaction in Section 3.4, and cast controls for
the multi-stage recommendation pipeline with certain API calls and tool usage for conversational recommender systems.
RecLLM [43] further extends the permission of LLM, and proposes a roadmap for building an integrated conversational

recommender system, where LLM is able to manage the dialogue, understand user preference, arrange the ranking
stage, and even provide a controllable LLM-based user simulator to generate synthetic conversations. RAH [185]
designs the Learn-Act-Critic loop and a reflection mechanism for LLM-based agents to improve the alignment with
user preferences during the interaction period. InteRecAgent [74] serves as the interactive agent for conversational
recommendation with the users, and is accessible to a range of plug-in tools including but not limited to intention
detection, information query, item retrieval, and item ranking. Besides, instead of allowing LLM to take over control of
the entire recommendation pipeline, RecMind [225] makes finer-grained control over the recommendation process
with task deconstruction. It proposes to address the user queries under self-inspiring prompting strategy and multi-step
reasoning with tool usage (e.g., expert models, SQL tool, search engine).

# 3.6 Discussion

We could observe that the development path about where to adapt LLM to RS is fundamentally aligned with the progress
of large language models. Back in the year 2021 and early days of 2022, the parameter sizes of pretrained language
models are still relatively small (e.g., 110M for BERT-base, 1.5B for GPT2-XL). Therefore, earlier works usually tend to
either incorporate these small-scale language models as simple textual feature encoders, or as scoring/ranking functions
finetuned to fit the data distribution of recommender systems. In this way, the recommendation process is simply
formulated as a one-shot straightforward predictive task, and can be better solved with the help of language models.
As the model size gradually increases, researchers discover that large language models have gained emergent abilities
(e.g., instruction following and reasoning), as well as a vast amount of open-world knowledge with powerful text
generation capacities. Equipped with these amazing features brought by large-scale parameters, LLM starts to not only
deepen its usage in the feature encoder and scoring/ranking function stage, but also further extend their roles into
other stages of the recommendation pipeline. For instance, in the feature engineering stage, we could instruct LLM to
generate reliable auxiliary features and synthetic data samples [129] to assist the model training and evaluation. In this
way, the open-world knowledge from LLM is injected into the closed-domain recommendation models. Furthermore,
large language models also revolutionize the user interaction with a more human-friendly natural language interface
and free-form dialogue for various information systems. Not to mention, participating in the pipeline control further
requires sufficient logical reasoning and tool utilization capabilities, which are possessed by large language models.
In summary, we believe that, as the abilities of large language models are further explored, they will form gradually
deeper couplings and bindings with multiple stages of the recommendation pipeline. Even further, we might need to
customize large language models specifically tailored to satisfy the unique requirements of recommender systems [114].

# 4 HOW TO ADAPT LARGE LANGUAGE MODELS

To answer the “HOW” question about adapting LLM to RS, we carry out two orthogonal taxonomy criteria to distinguis
the adaptation of LLM to RS, resulting in a four-quadrant classification shown in Figure 4:

• Tune/Not Tune LLM denotes whether we will tune LLM based on the in-domain recommendation data during the
training phase. The definition of tuning LLM includes both full finetuning and other parameter-efficient finetuning
methods (e.g., LoRA [69], prompt tuning [89]).
• Infer with/without CRM denotes whether we will involve conventional recommendation models (CRM) during
the inference phase. Note that there are works that only use CRM to serve as independent pre-filter functions to

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b5f2/b5f245d2-9033-4de0-babf-177db5fb47e8.png" style="width: 50%;"></div>
Fig. 4. Four-quadrant classification about how to adapt LLM to RS. Each circle in the quadrants denotes one research work with the corresponding model name attached below the circle. The size of each circle means the largest size of LLM leveraged in the research work. The color of each circle indicates the best compared baseline that the proposed model defeats as reported in the corresponding paper. For example, the green circle of Chat-REC in quadrant 3 denotes that it utilizes a large language model with size larger than 100B (i.e., ChatGPT) and defeats the MF baseline. Besides, we summarize the general development path with light-colored arrows. Abbreviations: MF is short for matrix factorization; MLP is short for multi-layer perceptron.

generate the candidate item set for LLM [51, 206, 258]. We categorize them as “infer without CRM”, since the CRM is
independent of LLM, and could be decoupled from the final recommendation task.

In Figure 4, we use different marker sizes to indicate the size of the large language model the research works adapt,
and use different colors to indicate the best baseline they have defeated in terms of item recommendation. Thus, a few
works are not presented in Figure 4 since they do not provide traditional recommendation evaluation, e.g., RecLLM [43]
only investigates the system architecture design to involve LLM for RS pipeline control without experimental evaluation.
Moreover, it is noteworthy that some research works might propose techniques that are applied across different
quadrants. For instance, ReLLa [120] designs semantic user behavior retrieval to help LLM better comprehend and
model the lifelong user behavior sequence in both zero-shot prediction (i.e., quadrant 3) and few-shot finetuning (i.e.,
quadrant 4) settings.
Given the four-quadrant taxonomy, we demonstrate that the overall development path in terms of “HOW” research
question generally follows the light-colored arrows in Figure 4. Accordingly, we will introduce the latest research works
in the order of quadrant 1, 3, 2, 4, followed by in-detail discussions for each quadrant subsection.

<div style="text-align: center;">RecRanker '23
</div>
# 1 Tune LLM & Infer with CRM (Quadrant 1)

Quadrant 1 refers to research works that not only finetune the large language models with in-domain recommendation
data during the training phase, but also introduce conventional recommendation models to provide better collaborative
knowledge. Based on their development over time, the works in quadrant 1 can be mainly classified into two stages.
Back to years 2021 and 2022, earlier works in quadrant 1 mainly focus on applying relatively smaller pretrained
language models (e.g., BERT [31] and GPT2 [168]) to the downstream domains with abundant textual features, e.g., news
recommendation [130, 233, 256, 269], web search [134, 290] and e-commercial advertisement [150, 208]. As discussed in
Section 3.6, the primary roles of these small-scale language models are only limited to feature encoders for semantic
representation enhancement. Consequently, a conventional recommendation model (CRM) is required to make the final
recommendation, with generated textual representations as auxiliary inputs. Additionally, the small model size makes
it affordable to fully finetune the language model during the training phase. UNBERT [269], PLM-NR [233], PREC [130]
and Tiny-NewsRec [256] conduct full finetuning over small-scale language models (e.g., BERT [31], RoBERTa [135],
UniLMv2 [2]) to enhance the content understanding for better news recommendation with sequential CRMs. Zou et al.
[290] and Liu et al. [134] customize pretrained language models with further pretraining over the domain-specific
corpus for web search scenarios. Moreover, TransRec [44] proposes layerwise adapter tuning over BERT [31] to ensure
both the training efficiency and multi-modality enhanced representations. Although these earlier works can defeat
strong baseline models with attention mechanisms by tuning language models and involving CRM, they only leverage
small-scale language models as feature encoders, and thus the key capacities (e.g., reasoning, instruction following) of
large foundation models remain underexplored.
At around the beginning of the year 2023, the rise of LLM (e.g., ChatGPT) demonstrates impressive emergent
abilities like reasoning and instruction following, pointing out promising directions for LLM-enhanced recommendation.
Therefore, researchers start to investigate the potential of incorporating billion-level large language models (e.g.,
LLaMA [202] and ChatGLM [40, 263]) to the field of recommender systems. Compared to earlier works with small-scale
language models that we have discussed above, there are two major differences to be clarified for these recent works
that incorporate large language models:

• Due to the massive amount of model parameters possessed by LLM, we can hardly perform full finetuning on LLM
as it can lead to an unaffordable cost in computational resources. Instead, parameter-efficient finetuning (PEFT)
methods are commonly adopted for training efficiency with usually less than 1% parameters need to be updated, e.g.,
low-rank adaption (LoRA) [69] and prompt tuning [89, 108].
• The role of LLM is no longer a simple tunable feature encoder for CRM. To make better use of the reasoning ability
and open-world knowledge exhibited by LLM, researchers tend to place LLM and CRM on an equal footing (e.g.,
both as the recommenders), mutually leveraging their respective strengths to collaborate and achieve improved
recommendation performance. Moreover, as discussed in Section 3, LLM can also be finetuned for the stages of
feature engineering [111], user interaction [90] and pipeline control [43] as well.

CoLLM [275] and E4SRec [103] adopt LoRA to finetune Vicuna-7B [20] and LLaMA2-13B [202] respectively, and build
personalized prompts by injecting the user & item embedding from a pretrained CRM via a linear mapping laye
CTRL [102] conducts knowledge distillation between LLM and CRM for better alignment and interaction between the
semantic and collaborative knowledge, where the size of involved LLM scales up to 6 billion (ChatGLM-6B [263]) with

last-layer finetuning strategy. LLaMA-E[183] and EcomGPT [111] finetune the base large language models (i.e., LLaMA
30B [202] and BLOOMZ-7.1B [149]) to assist the conventional recommendation models with augmented generative
features, e.g., item attributes and topics of user reviews.
As shown in Figure 4, since CRM is involved and LLM is tunable, the research works in quadrant 1 could better align
to the data distribution of recommender systems and thus all achieve satisfying performance, even when the size of
involved LLM is relatively small. Moreover, we can observe the clear trend that researchers intend to consider larger
language models from the million level up to the billion level, thus benefiting from their vast amount of open-world
semantic knowledge, as well as the instruction following and reasoning abilities. Nevertheless, when it comes to
low-resource scenarios, the small-scale language model (e.g., BERT) is still an economic choice to balance between
LLM-based enhancement and computational efficiency.

# 4.2 Not Tune LLM & Infer w/o CRM (Quadrant 3)

Quadrant 3 refers to research works that exclude the conventional recommendation model and solely adopt a frozen
large language model as the recommender. This line of research generally emerges since the advent of large foundation
models, especially ChatGPT, where researchers aim to analyze the zero-shot or few-shot performance of LLM in
recommendation domains with LLM fixed and CRM excluded. It should be noted that, in the context of quadrant 3, the
“few-shot” setting specifically refers to the in-context learning (ICL) approach for LLM, rather than tuning LLM based
on a few training samples.
Earlier works [26, 126, 193, 217] investigate the zero-shot and few-shot recommendation settings based on the
ChatGPT API, with delicate prompt engineering to instruct the LLM to model the user interest and perform tasks like
rating prediction, pairwise comparison, and listwise ranking. However, the performance of these approaches is not
satisfactory. Based on these previous works, several attempts [48, 120, 225, 267] are made to improve the zero-shot or few
shot recommendation performance of LLM. Lin et al. [120] identify the lifelong sequential behavior incomprehension
problem of LLM, i.e., LLM fails to extract the useful information from a textual context of long user behavior sequence
for recommendation tasks, even if the length of context is far from reaching the context limitation of LLM. To mitigate
such an issue, ReLLa [120] proposes to perform semantic user behavior retrieval to replace the simply truncated top𝐾
recent behaviors with the top𝐾 semantically relevant behaviors towards the target item. In this way, the quality of
data samples is improved, thus making it easier for LLM to comprehend the user sequence and achieve better zero-shot
recommendation performance. RecMind [225] designs the self-inspiring prompt strategy and enables LLM to explicitly
access the external knowledge with extra tools, such as SQL for recommendation database and search engine for web
information. Chat-REC [48] instructs ChatGPT to not only serve as the score/ranking function, but also take control
over the recommendation pipeline, e.g., deciding when to call an independent pre-ranking model API.
As illustrated in Figure 4, although a larger model size might bring performance improvement, the zero-shot or
few-shot learning of LLM in quadrant 3 is much inferior compared with the light-weight CRM tuned on the training
data. Even when equipped with advanced techniques such as user behavior retrieval and tool usage, the performance of
a frozen LLM without CRM is still suboptimal and far from the SOTA performance. The knowledge contained in LLM
is global and factual, but recommendation is a personalized task that requires preference-oriented knowledge. This
indicates the importance of in-domain collaborative knowledge from the training data of recommender systems, and
that solely relying on a fixed large language model is currently unsuitable to well tackle the recommendation tasks.
Consequently, there are two major approaches to further inject the in-domain collaborative knowledge for LLM to

mprove the recommendation performance: (1) involving CRM for inference, and (2) tuning LLM based on the training
data, which refer to works of quadrant 2 and quadrant 4 in Figure 4, respectively.

# 4.3 Not Tune LLM & Infer with CRM (Quadrant 2)

# 4.3 Not Tune LLM & Infer with CRM (Quadrant 2)

Research works in quadrant 2 utilize different key capabilities (e.g., rich semantic information, reasoning ability) of
LLM without finetuning to help CRM better accomplish the recommendation tasks. Similar to works in quadrant 1,
the utilization of a frozen LLM in quadrant 2 generally demonstrates a development path in terms of the size of LLM
evolving over time, i.e., from small-scale language models to large language models.
Early works [34, 66, 67] propose to extract transferable text embeddings from a fixed BERT [31] model with rich
semantic information. The text embeddings are then fed into several projection layers to better produce the cross-domain
representations as the input of trainable conventional recommendation models. The projection layers are designed
as a single-layer neural network for ZESRec [34], a mixture-of-expert (MoE) network for UniSRec [67], and a vector
quantization based embedding lookup table for VQ-Rec [66]. We can observe from Figure 4 that the direct usage of
a single-layer neural network as an adapter does not yield satisfactory results. However, with a carefully designed
adapter module, the semantic representations from the fixed BERT parameters can be better aligned with the subsequent
recommendation module, leading to impressive recommendation performances.
As discussed in Section 3.6, with the model size scaling up, the emergent abilities and abundant open-world knowledge
enable large foundation models to extend their roles to other stages of the recommendation pipeline, such as feature
engineering stage [92, 129, 227, 242] and user interaction [60, 67, 175, 222]. AnyPredict [227] leverages ChatGPT
APIs to consolidate tabular samples to overcome the barrier across tables with varying schema, resulting in unified
expanded training data for the follow-up conventional predictive models. ONCE [129] utilizes ChatGPT to perform
news piece generation, user profiling, and news summarization, and thus augments the news recommendation model
with LLM-generated features. KAR [242] and RLMRec [176] leverage LLM to enhance the user behavior modeling
with specially designed input templates as well as chained prompting strategies, aiming to provide user-level feature
augmentation for CRM. Wang et al. [222] and He et al. [60] investigate the integration of LLM (e.g., ChatGPT and GPT4)
to handle the open-ended free-form chatting during the user interaction of conversational recommendation.
In these works, although LLM is frozen, the involvement of CRM for the inference phase generally guarantees better
recommendation performance, compared with works from quadrant 3 (i.e., Not Tune LLM; Infer w/o CRM) in terms of
the best baseline they defeat. When compared with quadrant 1 (i.e., Tune LLM; Infer with CRM), since the large language
model is fixed, the role of LLM in quadrant 2 is mostly auxiliary for CRM at different stages of the recommendation
pipeline, including but not limited to feature engineering and feature encoder.

# 4.4 Tune LLM & Infer w/o CRM (Quadrant 4)

Research works in quadrant 4 aim to finetune the large language models to serve as the scoring/ranking function
based on the training data from recommender systems, excluding the involvement of CRM. Since CRM is excluded,
we have to apply prompt templates to obtain textual input-output pairs, and therefore convert the recommendation
tasks (e.g., click-through rate estimation and next item prediction) into either a text classification task [4, 120] or a
sequence-to-sequence task [49, 188, 206, 231].
As an early attempt, LMRecSys [274] tunes language models to estimate the score of each candidate item via joint
inference over multiple masked tokens, resulting in unsatisfying performance. The reason might be that its scoring
manners are somehow problematic, where the authors simply pad or truncate the length of all the titles for items
19

to 10 tokens. Prompt4NR [276] finetunes BERT by predicting the key answer words (e.g., Yes/No, Good/Bad) based
on the prompting templates. PTab [125] transforms tabular data into text and finetunes a BERT model based on the
masked language modeling task followed by classification tasks. UniTRec [145] finetunes a BART [91] model with
a joint contrastive loss to optimize the discriminative score and a perplexity-based score. RecFormer [94] adopts
two-stage finetuning based on masked language modeling loss and item-item contrastive loss with LongFormer [5]
as the backbone model. P5 [49], FLAN-T5 [86], and InstructRec [268] adopt T5 [169] as the backbone, and train the
model in a sequence-to-sequence manner. GPT4Rec [95] tunes GPT [168] models as a generative function for next item
prediction via causal language modeling.
The works mentioned above all adopt full finetuning over relatively small-scale language models (e.g., 110M for
BERT-base, 149M for LongFormer), which could be considerably expensive and unscalable as the size of the language
model continuously increases up to tens of or even hundreds of billions. Although the large model parameter capacity
enables proficient knowledge and capabilities, fully finetuning such a big model can lead to substantial resource
consumption. As a result, parameter-efficient finetuning methods (PEFT) are usually required to efficiently adapt
billion-level LLM to RS. Among those PEFT methods, low-rank adaption (LoRA) serves as the most popular choice. For
instance, ReLLa [120], GenRec [77], BIGRec [3], RecSysLLM [23] and LSAT [184] adopt the LoRA [69] technique to
finetune a base large language model (usually LLaMA-7b [202] or Vicuna-7B [20]) for item scoring or generation tasks.
Apart from LoRA, M6-Rec [25] designs option tuning as an improved version of prompt tuning to empower M6 [118]
for varied downstream tasks like item retrieval and ranking. VIP5 [50] performs layerwise adapter tuning to unify
various modalities (e.g., ID, text, and image) via a universal foundation model for recommendation tasks.
Although the introduction of PEFT alleviates the training inefficiency issue, the computational overhead can still
be excessive for real-world applications where the number of training samples might scale up to billions. In such a
situation, even PEFT methods like LoRA are not efficient enough for LLM to go over the entire training dataset. To
this end, recent works start to investigate the strong inductive learning capability [160] of LLM by downsampling the
whole training set into a small-scale subset [18, 86, 120, 139]. As a representative, ReLLa [120] uniformly samples less
than 10% of the training instances and surprisingly finds that LLM, which is finetuned only based on less than 10%
samples, is able to outperform the conventional recommendation baseline models that are trained on the entire training
dataset. Such a phenomenon about the strong few-shot inductive learning capability of LLM in recommendation is also
validated by other related works [18, 86, 139]. As for different downsampling strategies, PALR [18] randomly selects
20% of the user to construct the training subset for efficient finetuning of LLaMA-7B [202]. RecRanker [139] designs an
adaptive user sampling strategy, which consists of both importance-aware and clustering-based sampling followed the
repetitive penalty.
As shown in Figure 4, the performance of finetuning LLM based on recommendation data is promising with proper
task formulation, even if the model size is still relatively small (i.e., less than 1 billion). Apart from the design of input
prompt and model architecture to achieve superior recommendation performance, scalability and efficiency are also
the major challenges in this line of research. That is, how to efficiently finetune a large-scale language model on a
large-scale training dataset, where various PEFT methods and data downsampling strategies would be considered.

# 4.5 Discussion

We first conclude the necessity of collaborative knowledge injection when adapting LLM to RS, and then summarize
the overall development path in terms of the “HOW” question, as well as possible future directions. Next, we cast a

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e514/e5146753-67a7-4a8d-89ce-3c8cc15747e6.png" style="width: 50%;"></div>
Fig. 5. The illustration of the development trend for adapting LLM to RS in terms of the “HOW” research question based on the four-quadrant classification. Earlier attempts generally perform joint optimization of small-scale language models and conventional recommendation models based on the training data (i.e., Quadrant 1). Then, researchers try to introduce a frozen LLM for recommendation without the help of CRM (i.e., Quadrant 3), which results in inferior performance. To this end, the golden principle, i.e., in-domain collaborative knowledge injection