# Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation
Alireza Salemi University of Massachusetts Amherst Amherst, MA, United States asalemi@cs.umass.edu Surya Kallumadi Lowe’s Companies, Inc. Charlotte, NC, United States surya@ksu.edu
Surya Kallumadi Lowe’s Companies, Inc. Charlotte, NC, United States surya@ksu.edu Hamed Zamani University of Massachusetts Amher Amherst, MA, United States zamani@cs.umass.edu
# ABSTRACT
This paper studies retrieval-augmented approaches for personalizing large language models (LLMs), which potentially have a substantial impact on various applications and domains. We propose the first attempt to optimize the retrieval models that deliver a limited number of personal documents to large language models for the purpose of personalized generation. We develop two optimization algorithms that solicit feedback from the downstream personalized generation tasks for retrieval optimization–one based on reinforcement learning whose reward function is defined using any arbitrary metric for personalized generation and another based on knowledge distillation from the downstream LLM to the retrieval model. This paper also introduces a pre- and post-generation retriever selection model that decides what retriever to choose for each LLM input. Extensive experiments on diverse tasks from the language model personalization (LaMP) benchmark reveal statistically significant improvements in six out of seven datasets.
# • Computing methodologies →Natural language generation; • Information systems →Learning to rank; Personalization.
# KEYWORDS
Ranking optimization; retrieval-augmented generation; personalization; text generation
ACM Reference Format: Alireza Salemi, Surya Kallumadi, and Hamed Zamani. 2024. Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24), July 14–18, 2024, Washington, DC, USA. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3626772.3657783
# 1 INTRODUCTION
Personalization has been extensively explored by information retrieval (IR), recommender systems, and human-computer interaction communities, particularly in the context of information access [17, 36, 51]. Even though the recent advancements in large language models (LLMs) have revolutionized various applications, the existing commercial and open-source LLMs exhibit a significant
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/246a/246a73e9-ab13-4a31-92b0-29894656cca3.png" style="width: 50%;"></div>
Figure 1: An overview of retrieval augmentation approaches for LLM personalization. First, the query function 𝜙𝑞produces a query from the input 𝑥. Relevant personal information is then retrieved and fed to the personalized prompt generation function 𝜙𝑝for LLM consumption.
limitation by failing to tailor their generated outputs according to the backgrounds and historical preferences of their users. As LLMpowered conversational agents become more prevalent, the need for LLM personalization becomes increasingly apparent [43]. LLM personalization has diverse applications, from customizing educational content and curating news feeds to improving e-commerce suggestions and delivering personalized healthcare information. Various approaches can be envisioned to personalize LLMs: (1) fine-tuning LLM parameters, either entirely or partially, for individual users, (2) integrating latent user representations with LLMs, and (3) enriching LLM prompts with user-specific content and/or context. The first two approaches involve adjusting LLM architecture and parameters, which is costly or even impractical in terms of storage, computation cost, and/or time. Besides, they cannot perform well for cold-start users. As an instantiation of retrievalenhanced methods [55], the third approach, on the other hand, is applicable to any off-the-shelf LLM. To efficiently and effectively utilize the potentially extensive personal data for each active user, it is essential to implement a retrieval mechanism. As presented in Figure 1, this mechanism selects personal information that best enhances the LLM for the purpose of personalized text generation. This paper focuses on optimization of personal information retrieval for the purpose of personalizing LLMs. For this purpose, standard learning-to-rank optimization methods are not applicable, since they typically require query-document-relevance triplets for training and it is not clear what documents is “relevant” for the downstream personalized text generation tasks. We study two retrieval optimization solutions for personalizing LLMs. First, we develop a reinforcement learning approach, where the training process involves sampling documents from the user’s profile with respect to the probabilities generated based on retrieval scores. The sampled documents are then fed to the LLM and its downstream
performance for producing personalized output texts (using any arbitrary metric) is used to compute a reward function for optimizing the retrieval model. Second, we optimize the retrieval model by distilling knowledge from the LLM. We minimize the divergence between the retrieval score distribution and a target distribution computed using the downstream performance of LLM on producing personalized output texts using each individual retrieved document. To gauge the efficacy of these two optimization methods, we apply them to train dense retrieval models for LLM personalization. Moreover, we observe that LLM personalization has multiple dimensions and existing retrieval models do not address all of them. For instance, retrieving recent user interactions may be needed for effective personalized text generation for an input, while a keywordmatching, a semantic matching, or a model that is aware of user’s writing style may be optimal for another input. According to this observation, we hypothesize that a retrieval selection model that selects a model from a diverse pool of retrievers can impact LLM personalization. Following this hypothesis, we further develop a pre- and a post-generation model mode for retrieval selection that decides what retrieval model should be chosen for each given input. In our study, these models could choose between (1) no retrieval (i.e., no personalization), (2) recency-based retrieval, (3) term matching (i.e., BM25 [41]), (4) zero-shot semantic matching (Contriever [19]), and (5, 6) the two dense retrieval models developed in this paper that are specifically trained for LLM personalization. To optimize the retrieval selection models, we align the probability distribution obtained from the LLM downstream performance and the scores generated by the selection model for various retrieval models. We evaluate our models using the LaMP benchmark [43], consisting of seven diverse personalization tasks, including three personalized text classification (binary, categorical, and ordinal) and four personalized text generation tasks. The methods proposed in this paper advance the state-of-the-art performance on six out of seven tasks in LaMP with statistically significant improvements. Our best-performing method exhibits an average of 5.5% state-ofthe-art improvements across all LaMP datasets. Comparing to the non-personalized LLM, our best approach demonstrate 1.0%-33.8% improvements across all tasks, with an average improvement of 15.3%. To facilitate the research on this domain, we share our codes and trained model parameters to support future research, promoting transparency and reproducibility. 1
# 2 RELATED WORK
Personalized Text Generation. Personalization has been a focal point of research in various domains, particularly within search and recommendation systems [5, 10, 14, 45, 56]. This exploration spans diverse contexts, encompassing areas such as query autocompletion [21] and collaborative personalized search [51]. Within the NLP community, personalization has been a subject of exploration in various applications, including but not limited to dialogue agents [31, 39, 46, 49, 58, 60], review[27] and recipe generation [30], translation[50], headline generation [1], and classification tasks [13, 16], such as personalized sentiment analysis [33]. With the emergence of LLMs and their application across various NLP tasks, Salemi et al. [43] proposed a retrieval-augmented
approach for personalizing LLMs. They also introduced LaMP, a benchmark designed to assess the performance of personalized NLP models across diverse classification and short text generation tasks. The work by Li et al. [26] addresses a similar issue, focusing on personalized long text generation. Furthermore, Mysore et al. [35] assesses the capabilities of LLMs in the role of writing assistants. Various approaches have been explored for personalizing LLMs, encompassing techniques such as summarizing user profile [40], aligning language models with personalized human feedback [22], automatic prompt generation tailored to individual users [25], and incorporating long and short-term memory-based personalization strategies [57]. In this study, we adhere to the methodology outlined by Salemi et al. [43] and conduct experiments using the LaMP benchmark, focusing on training a component for retrieving personal information from user profile.
approach for personalizing LLMs. They also introduced LaMP, a benchmark designed to assess the performance of personalized NLP models across diverse classification and short text generation tasks. The work by Li et al. [26] addresses a similar issue, focusing on personalized long text generation. Furthermore, Mysore et al. [35] assesses the capabilities of LLMs in the role of writing assistants. Various approaches have been explored for personalizing LLMs, encompassing techniques such as summarizing user profile [40], aligning language models with personalized human feedback [22], automatic prompt generation tailored to individual users [25], and incorporating long and short-term memory-based personalization strategies [57]. In this study, we adhere to the methodology outlined by Salemi et al. [43] and conduct experiments using the LaMP benchmark, focusing on training a component for retrieving personal information from user profile. Retrieval Optimization in Retrieval-Augmented Generation. The optimization of retrieval models within the RAG (RetrievalAugmented Generation) pipelines has emerged as a focal point in recent research, particularly in the context of question answering. Yang and Seo [52] focuses on distilling knowledge from the LM to the retriever by minimizing the KL-divergence between the LM’s performance for each document in the retrieved set and the assigned score by the retriever to that document in the set. Additionally, Izacard and Grave [20] employs the attention weights of the LM to determine the importance of each document. This information is then utilized to distill knowledge from the LM to the retriever, aligning with the objectives set forth by Yang and Seo [52]. The approach presented by Wang et al. [47] involves the use of reinforcement learning, where the reward function is derived from the performance of the LM. In this work, we adopt methods similar to that of Wang et al. [47] and Yang and Seo [52], given the absence of relevance data in the LaMP benchmark. Notably, our work stands out as the pioneering effort in leveraging feedback from LLMs to train personalized retrievers for personalizing LLMs. Furthermore, in all the previously mentioned approaches, the language model is trained after/with the retrieval model. In contrast, our approach assumes the language model is frozen, and our focus is solely on optimizing the retrieval model.
Retrieval Optimization in Retrieval-Augmented Generation. The optimization of retrieval models within the RAG (RetrievalAugmented Generation) pipelines has emerged as a focal point in recent research, particularly in the context of question answering. Yang and Seo [52] focuses on distilling knowledge from the LM to the retriever by minimizing the KL-divergence between the LM’s performance for each document in the retrieved set and the assigned score by the retriever to that document in the set. Additionally, Izacard and Grave [20] employs the attention weights of the LM to determine the importance of each document. This information is then utilized to distill knowledge from the LM to the retriever, aligning with the objectives set forth by Yang and Seo [52]. The approach presented by Wang et al. [47] involves the use of reinforcement learning, where the reward function is derived from the performance of the LM. In this work, we adopt methods similar to that of Wang et al. [47] and Yang and Seo [52], given the absence of relevance data in the LaMP benchmark. Notably, our work stands out as the pioneering effort in leveraging feedback from LLMs to train personalized retrievers for personalizing LLMs. Furthermore, in all the previously mentioned approaches, the language model is trained after/with the retrieval model. In contrast, our approach assumes the language model is frozen, and our focus is solely on optimizing the retrieval model.
# Information Access with Multiple Retrieval Models. Combin-
ing rank lists generated by different retrievers has been extensively explored in the literature [9, 15, 29, 37]. However, the process of rank fusion presents challenges, especially when dealing with discrepancies in scoring scales among retrieval systems or the absence of overlapping documents in the ranked lists [29, 59]. Alternatively, methods for selecting specific retriever from a retriever pool for different datasets has been explored [23]. Furthermore, Arabzadeh et al. [3] investigates the optimal use of dense and sparse retrievers for each query, considering efficiency trade-offs. In our study, we concentrate on the performance-oriented selection of query-specific retrievers from a retriever pool.
# 3 NOTATIONS AND TASK FORMULATION
Generative language models often take an input 𝑥and generate the most probable sequence tokens 𝑦. This paper focuses on the task of personalized generation with the goal of generating outputs that
are tailored for the preferences and characteristics of the language model user. Let 𝑇= {(𝑢1,𝑥1,𝑦1), (𝑢2,𝑥2,𝑦2), · · · , (𝑢𝑁,𝑥𝑁,𝑦𝑁)} be a set of 𝑁training instances, each consisting of a user 𝑢, an input text 𝑥submitted by the user 𝑢, and the ground truth personalized output 𝑦. For each user 𝑢, a user profile 𝑃𝑢exists that can be employed for developing personalized generation models. A user profile 𝑃𝑢is a set of personal documents associated with the user 𝑢. As discussed in Section 1, this paper focuses on retrieval augmented solutions for personalization, depicted in Figure 1. In such solutions, we first retrieve a set of personal documents from the user profile 𝑃𝑢. This is achieved through 𝐿= R(𝜙𝑞(𝑥); 𝑃𝑢) where 𝜙𝑞is a query generation function that produces a search query string given the LLM input 𝑥and R is a retrieval model that retrieves personal documents from 𝑃𝑢given a query produced by 𝜙𝑞. Hence, R returns a list of personal documents 𝐿. A prompt generation function 𝜙𝑝is then applied to the LLM input and the retrieved result list as follows: 𝜙𝑝(𝑥, 𝐿). The constructed personalized prompt is then fed into a LLM 𝑀. The goal is to minimize the error between the generated output and the ground truth personalized output 𝑦. We assume that the LLM 𝑀is given and we do not aim at updating the LLM parameters for personalized generation. The rational behind this decision is that (1) fine-tuning 𝑀is often very expensive, and more importantly (2) the LLM 𝑀can memorize personal information if it is fine-tuned on data retrieved from the user’s personal data 𝑃𝑢. Such memorization can put the user’s privacy at risk. That said, this paper focuses on minimizing the personalized text generation error by solely updating the retrieval results 𝐿. Section 4 studies methods for optimizing the retrieval model R𝜃 parameterized by 𝜃for updating the result list 𝐿. Section 5 extends Section 4 by exploring optimization solutions for retrieval model selection from a set of pre-defined retrieval models for updating the result list 𝐿.
# 4 LEARNING TO RETRIEVE FOR PERSONALIZING LLMS
Learning-to-rank (LTR) methods are often employed to train ranking models for search and recommendation [6]. For personalized LTR, the user’s profile and long-term history are often utilized. Such personalized implicit feedback signals are often document-level and directly provided by the user, such as ratings, clicks, views, dwell time, and/or purchases [5, 56]. In the context of retrieval-augmented personalized text generation, user feedback manifests in the form of text written or edited by the user (i.e., the label 𝑦𝑖for each input 𝑥𝑖), taking into account the user preferences and interests. Therefore, accessing user feedback on a per-document basis within the user profile for training retrieval models is not feasible; neither is collecting document-level feedback through annotation, e.g., crowdsourcing. The main reason is that we do not know what documents serve the LLM best for generating personalized outputs for each input text. Therefore, learning to rank documents for LLM personalization is fundamentally different from developing personalized search or recommendation engines. Considering these, we propose optimization methods that leverage feedback from the LLM itself, obtained by evaluating the impact of retrieved documents on the LLM performance for generating personalized outputs. Our first method uses the LLM performance
(in terms of any arbitrary metric) to form a reward function and employs reinforcement learning [48] for optimization. Our second approach is based on knowledge distillation from the LLM to the retriever based on the LLM’s performance in terms of personalized text generation. They are described in the subsequent subsections.
# 4.1 Retrieval Optimization for Personalized Generation using Reinforcement Learning
This section introduces ROPG-RL–a reinforcement learning approach that encourages the retrieval model to produce rankings that lead to more accurate personalized generation. We utilize the vanilla policy gradient optimization algorithm, drawing upon rewards supplied by the downstream LLM, as depicted in Figure 2 (a). In this approach, we establish a parameterized policy (here, the retrieval model) that assigns a probability to each action (specifically, selecting personal documents to be fed to the LLM). Subsequently, we formulate a reward function, well aligned with the goal of personalized text generation, that the parameterized policy aims to maximize. This approach enables us to effectively refine and improve the performance of the retrieval model. The specifics of our methodology will be discussed in the subsequent paragraphs. Parameterized Policy (𝜋𝜃). In this context, defining the policy function necessitates a clear delineation of the states and actions applicable within the scope of this study. In this formulation, the policy is parameterized through the retrieval model. Essentially, given a query, the retrieval model undergoes training to assign a higher probability to the documents from the user profile that are deemed more valuable for personalizing the LLM. Here, an action is considered as selecting a document given a query.2 The state, on the other hand, corresponds to the given query itself. Meaning that the goal is to update the policy such that it produces more effective results for personalization. In this study, we restrict our focus to trajectories comprising a single state. This implies that the model initiates from the initial state, executes a singular action, and concludes the trajectory. The probability of each action is computed using the following formula:
This section introduces ROPG-RL–a reinforcement learning approach that encourages the retrieval model to produce rankings that lead to more accurate personalized generation. We utilize the vanilla policy gradient optimization algorithm, drawing upon rewards supplied by the downstream LLM, as depicted in Figure 2 (a). In this approach, we establish a parameterized policy (here, the retrieval model) that assigns a probability to each action (specifically, selecting personal documents to be fed to the LLM). Subsequently, we formulate a reward function, well aligned with the goal of personalized text generation, that the parameterized policy aims to maximize. This approach enables us to effectively refine and improve the performance of the retrieval model. The specifics of our methodology will be discussed in the subsequent paragraphs.
� where 𝜙𝑞is the query generation function as explained in Section 3, R𝜃denotes the retrieval model parameterized by 𝜃, and 𝑃𝑢is a set of documents containing user’s personal data (see Section 3). The probability assigned by the policy model to any document that is not in the user profile would be zero. Note that the user profile can grow over time that leads to inefficient calculation of policy function. To address this efficiency issues, we can approximate the policy function, either using hierarchical softmax, similar to [32, 34, 53], or by marginalization through top 𝑙approximation. Without loss of generality we choose the second approach and compute the policy function based on 𝑃𝑙𝑢, a set of 𝑙documents from 𝑃𝑢that achieve highest retrieval scores according to our initial retrieval
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4771/47712922-6ad9-453b-a933-a1121c7691c4.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
<div style="text-align: center;">Figure 2: Overview of training dense retrievers for personalizing LLMs using LLMs feedback with policy gradient optimization (a) and knowledge distillation (b). 𝜙𝑞represents the query generation function, 𝜙𝑝is the prompt generation function, and "Critic" denotes the evaluation metric employed for the personalized task.</div>
weights. In our experiments, we set 𝑙= 16. Of course, leveraging the complete user profile during training could potentially yield superior performance, albeit at the cost of increased training time. Reward Function (𝑅). Defining trajectories involving the selection of multiple documents from the user profile, with rewards for each, is computationally intensive. This is due to the need to compute rewards using the LLM for every sampled set of documents from the user profile. On the contrary, if we limit trajectories to selecting only one document from the user profile, we can precompute the rewards for each document in the profile. Therefore, we only consider trajectories with one action. Moreover, to expedite the learning process and minimize variance, we subtract a previously calculated evaluation score from the effectiveness of the current sampled document. The reward function is then defined as:
Reward(𝑑;𝑥,𝑦) =  �
() Eval �𝑦, 𝑀(𝜙𝑝(𝑥, [𝑑]))�−Eval �𝑦, 𝑀(𝜙𝑝(𝑥, [𝑑𝑏]))�
 �� �� where 𝑑is a sampled document from the user profile 𝑃𝑢using the parameterized policy 𝜋𝜃and 𝑑𝑏is the document selected by the policy with initial weighting (i.e., the retrieval model prior to fine-tuning with RL). 𝑀is the LLM that generates personalized text based on the given personalized prompt 𝜙𝑝. Eval denotes an arbitrary metric for evaluating personalized text generation.
Training Objective (𝐽). The objective in ROPG-RL is to maximize the expected reward obtained by the parameterized policy of the retriever (𝜋𝜃). To achieve this objective, we employ a gradient ascent algorithm with the update rule of 𝜃𝑘+1 = 𝜃𝑘+ 𝛼∇𝜃𝐽(𝜋𝜃). For each mini-batch 𝐵⊂𝑇, the objective function is presented below:
arg max 𝜃 1 |𝐵| ∑︁ (𝑢,𝑥,𝑦)∈𝐵 E 𝑑∼𝜋𝜃 [Reward(𝑑;𝑥,𝑦) log 𝜋𝜃(𝑑|𝑥)] (3)
# 4.2 Retrieval Optimization for Personalized Generation using Knowledge Distillation
An alternative approach for training a retrieval model with feedback from the LLM involves knowledge distillation from the LLM to the retrieval model. Contrary to ROPG-RL, this approach, called ROPG-KD, considers the relative usefulness of different items in
the user profile for the LLM on performing the downstream personalized task. Indeed, this approach endeavors to allocate a higher probability to items that are more useful than others for the LLM. Conversely, ROPG-RL only considers the impact of each (or possibly a subset of documents grouped together in trajectories with more than one action) on the final score that the LLM achieves. Hence, it seeks to reward the model for actions that yield positive outcomes and penalize for actions that result in negative consequences. This implies that where there are no favorable actions, the model would face punishment; however, this is not the case with knowledge distillation. On the other hand, RL optimization processes tend to be less stable and are more susceptible to overfitting. Considering all the aforementioned aspects, we propose an alternative approach based on knowledge distillation. In the context of knowledge distillation, the primary objective is to encourage the retriever model to assign higher similarity scores to the documents from the user profile that are more useful for the language model in fulfilling its task. Figure 2 (b) illustrates the pipeline for this approach. To accomplish this objective, we employ Equation (1) to allocate a probability to individual elements within the user profile. Subsequently, we use the following function to produce the target probability distribution:  ��
∀𝑑∈𝑃𝑢
  � ∈ �� where Eval is an arbitrary metric for evaluating personalized text generation models and 𝑀denotes the LLM being used. Similar to ROPG-RL, for efficiency purposes, we approximate the distribution 𝑝𝑡by only focusing on the top 𝑙retrieved document w.r.t. the initial retrieval parameters. Inspired by previous work on knowledge distillation in IR [52], for each mini-batch 𝐵⊂𝑇, we minimize the following loss function based on KL-divergence: ∑︁ ∑︁
(4)
# 4.3 Retrieval Model Architecture
The proposed optimization solutions can be applied to any neural ranking model. Without loss of generality, we use them to train dense retrieval models. We adopt Contriever [19], a pre-trained biencoder model for dense retrieval. Contriever encodes the query and document text using an encoder with shared parameters and applies
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c574/c574e01e-5b2e-41f8-9e9f-7aabf2f6caaf.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Relative winning rate for each selected retrieval model. When multiple retrieval models get the highest score, we consider all of them with the highest score as the winner.</div>
# Figure 3: Relative winning rate for each selected retrieval model. When multiple retrieval models get the highest score, we consider all of them with the highest score as the winner.
dot product to compute the relevance score. After training, an exact or approximate nearest neighbor (kNN) algorithm is used to index the learned document representations for each user profile. We use exact kNN in our experiments. During inference, each document is scored independently and the scored documents are sorted in descending order with respect to their score.
# 5 RETRIEVAL MODEL SELECTION FOR PERSONALIZING LLMS
We hypothesize that there are multiple aspects to LLM personalization and each existing retrieval model does not address all of them. For instance, retrieving recent user interactions may lead to the highest personalized text generation performance for an input, while a keyword-matching model or a semantic matching model may be optimal for another input. To validate this, we utilize the LaMP benchmark–a recent benchmark consisting of seven diverse tasks for training and evaluating personalized LLMs [43]. Statistics of these datasets are presented in Table 1. More information on the LaMP benchmark is provided in Section 6.1. We evaluate personalization of an 11B parameter FlanT5-XXL [8] using the following retrieval augmentation approaches: (1) no personalization (i.e., no retrieval augmentation), (2) term matching retrieval using BM25 [41], (3) zero-shot semantic matching model using Contriever [19], (4) ranking documents in the user profile based on recency, and (5 & 6) the proposed retrieval models for LLM personalization–ROPGRL and ROPG-KD. Figure 3 illustrates the winning rate achieved by each of these models when retrieving from the user profile. To compute the winning rate, we first evaluate the recommended evaluation metric for each of the datasets by the LaMP benchmark (i.e., accuracy for categorical classification, MAE for ordinal classification, and Rouge-1 for text-generation tasks). For each retrieval model, we then count the number of inputs for which it achieves the highest personalized text generation performance among the mentioned six models. If two or more retrieval models achieve the same and the highest performance, they all get rewarded. The count is then normalized to estimate the winning rate. As evident in Figure 3, there is no consistent winner across all tasks in the LaMP benchmark. For instance, between the best text generation performance for 8.5% to 18.4% of the inputs across datasets can be achieved when no personalization is conducted. Recency-based ranking can lead to the best performance for 15.2% to 18.1% of the inputs depending on the dataset. Even the proposed
ROPG-RL and ROPG-KD models provide the highest performance for 14.9% to 20.2% of the inputs across different datasets in LaMP. Given these observations, we hypothesize that selecting what ranking function to use for each input, or even when to apply personalization, can improve end-to-end personalized text generation performance. According to this hypothesis and inspired by the query performance prediction literature [7, 42, 44, 54], we introduce two retriever selection models. In the first approach, referred to as RSPG-Pre, we retrieve items from the user profile using each retriever in the retriever pool to construct personalized prompts. These constructed prompts are then fed into RSPG-Pre for selection and consumption by the LLM. In the second approach, termed RSPGPost, the personalized prompts produced by all retrieval models are also presented to the LLM, and the resulting outputs, along with the original prompts, are fed into RSPG-Post for retrieval model selection. These models are presented below. Optimizing Retriever Selection Models. The training pipeline for retrieval selection is illustrated in Figure 4. Let R be a set of retrieval models in the pipeline and S𝜔be a retrieval selection model parameterized by 𝜔that produces a selection score for each retrieval model in R. We use a knowledge distillation loss from the downstream LLM performance to train the retrieval selection model. For this purpose, the target selection probability distribution over retrieval models in R for an input (𝑢,𝑥,𝑦) is computed as:  ��
�) (5)
�  �� where Eval is an arbitrary evaluation metric for personalized text generation, and 𝑀is the LLM used for text generation. The retrieval selection model probability for the retriever R𝑖∈R is calculated as:
(6)
� For a mini-batch 𝐵⊂𝑇, we use KL-divergence loss as follows to train the retrieval selection models:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b949/b94944a1-d07f-4a56-bbad-f74ecf452d2a.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 1: Statistics of the datasets within the LaMP benchmark [43] with time-based data separation.</div>
Table 1: Statistics of the datasets within the LaMP benchmark [43] with time-based data separation.
Task
#train
#dev
#test
Input Length
Output Length
#Profile Size
#classes
LaMP-1: Personalized Citation Identification
6542
1500
1500
51.43 ± 5.70
-
84.15 ± 47.54
2
LaMP-2: Personalized Movie Tagging
5073
1410
1557
92.39 ± 21.95
-
86.76 ± 189.52
15
LaMP-3: Personalized Product Rating
20000
2500
2500
128.18 ± 146.25
-
185.40 ± 129.30
5
LaMP-4: Personalized News Headline Generation
12500
1500
1800
29.97 ± 12.09
10.07 ± 3.10
204.59 ± 250.75
-
LaMP-5: Personalized Scholarly Title Generation
14682
1500
1500
162.34 ± 65.63
9.71 ± 3.21
87.88 ± 53.63
-
LaMP-6: Personalized Email Subject Generation
4821
1250
1250
454.87 ± 889.41
7.37 ± 2.78
55.67 ± 36.32
-
LaMP-7: Personalized Tweet Paraphrasing
13437
1498
1500
29.72 ± 7.01
16.96 ± 5.67
15.71 ± 14.86
-
Selection Model Architecture. To select a retrieval model for the given input, we initiate the process by employing all retrieval models (see Figure 3 for the list of retrieval models in our experiments) to retrieve relevant documents. As mentioned earlier, we envision two categories of retrieval selection models: pre-generation and post-generation models. We use an encoder-only model for estimating a selection score for each retrieval model. For the pre-generation scenario, the input to the encoder is the LLM prompt constructed by each retrieval model: 𝜙𝑝(𝑥, R𝑖(𝜙𝑞(𝑥); 𝑃𝑢)). For the post-generation scenario, this prompt is concatenated with the LLM’s output text given this prompt. The encoder’s final layer representation is then fed into a linear projection layer for producing a scalar value as the selection score. Given the long length of prompts in retrieval selection, we use Longformer [4] as the encoder in our retrieval selection model.3 Once all retrieval models are scored using this approach, we choose the one with the highest selection score and feed the corresponding prompt to the LLM.
# 6 EXPERIMENTS
# 6.1 Experimental Setup
Datasets. We adopt the LaMP benchmark [43]–a public benchmark that encompasses a diverse set of personalized text generation tasks.4 Specifically, the benchmark comprises three personalized text classification tasks and four personalized text generation tasks. They include (1) Personalized Citation Identification (binary classification), (2) Personalized Movie Tagging (categorical classification with 15 classes), (3) Personalized Product Rating (ordinal
classification from 1 to 5-star rating for e-commerce products), (4) Personalized News Headline Generation, (5) Personalized Scholarly Title Generation, (6) Personalized Email Subject Generation, and (7) Personalized Tweet Paraphrasing. We use the time-based separation setting offered by LaMP for data splitting. In this setting, the data for each user is split into train, development, and test sets based on their timestamp, modeling a real-world scenario in which personalized outputs for test inputs are generated using the personal documents created earlier by that user. The reason behind opting for a time-based separation in the LaMP benchmark is to investigate the impact of recency in our experiments. Table 1 reports the statistics of the datasets.
Evaluation Metrics. Following Salemi et al. [43], we evaluate LaMP-1 using Accuracy, LaMP-2 using Accuracy and F1-measure, and LaMP-3 using mean absolute error (MAE) and root mean squared error (RMSE). We use ROUGE-1 and ROUGE-L [28] to evaluate text generation performance on text generation datasets (LaMP4 to LaMP-7). Statistically significant differences are identified using two-tailed paired t-test for ROUGE-1/ROUGE-L/MAE/RMSE and McNemar test for Accuracy/F1.
Training Configurations. An integral part of our training pipeline is the evaluation function Eval. We use the standard metrics suggested by the LaMP benchmark for each dataset to implemented the Eval function. In more detail, we measure accuracy for the binary and categorical classification tasks (LaMP-1 and LaMP-2) and ROUGE-1 [28] for the text generation tasks (LaMP-4, LaMP-5, LaMP-6, and LaMP-7). Given that the evaluation metric for LaMP-3 is MAE (Mean Absolute Error), where lower values are preferable, we need to adapt the evaluation function to use it as a reward
Table 2: Templates used to create prompt to augment the input for LLM with the retrieved items from the user profile (i.e 𝜙𝑝). Function concat concatenates the strings in its first argument by placing the second argument between them. Functio add_to_paper_title adds the string in its first argument to the paper’s title in the LaMP-1 task. Function PPEP creates th prompt for each retrieved item from the profile. [INPUT] is the task’s input (𝑥).
Task
Per Profile Entry Prompt (PPEP)
Aggregated Input Prompt (AIP)
LaMP-1: Citation Ident.
"𝑃𝑖[title]"
add_to_paper_title(concat([PPEP(𝑃1), ..., PPEP(𝑃𝑛)], ", and "), [INPUT])
LaMP-2: : Movie Tag.
the tag for the movie: "𝑃𝑖[description]"
is "𝑃𝑖[tag]"
concat([PPEP(𝑃1), ..., PPEP(𝑃𝑛)], ", and "). [INPUT]
LaMP-3: Product Rat.
𝑃𝑖[score] is the score for "𝑃𝑖[text]"
concat([PPEP(𝑃1), ..., PPEP(𝑃𝑛)], ", and "). [INPUT]
LaMP-4: News Headline
"𝑃𝑖[title]" is the title for "𝑃𝑖[text]"
concat([PPEP(𝑃1), ..., PPEP(𝑃𝑛)], ", and "). [INPUT]
LaMP-5: Scholarly Title
"𝑃𝑖[title]" is the title for "𝑃𝑖[abstract]"
concat([PPEP(𝑃1), ..., PPEP(𝑃𝑛)], ", and "). Following the given patterns [INPUT]
LaMP-6: Email Subject
"𝑃𝑖[title]" is the title for "𝑃𝑖[text]"
concat([PPEP(𝑃1), ..., PPEP(𝑃𝑛)], ", and "). [INPUT]
LaMP-7: Tweet Para.
"𝑃𝑖[text]"
concat([PPEP(𝑃1), ..., PPEP(𝑃𝑛)], ", and ") are written by a person. Following the
given patterns [INPUT]
<div style="text-align: center;">Table 3: The performance of our methods and the baselines on the LaMP benchmark. For all metrics, the higher values the better, except for RMSE and MAE which are used in LaMP-3. In this table, the superscript 1, 2, 3, 4, and 5 indicate significan improvement over No Personalization, BM25, Recency, Contriever, and RRF, respectively (𝑝< 0.05).</div>
Dataset
Metric
No
Personalization Baselines
Our Methods
Personalization
BM25
Recency
Contriever
RRF
ROPG-RL
ROPG-KD
RSPG-Pre
RSPG-Post
LaMP-1: Personalized
Citation Identification
Accuracy ↑
0.502
0.626
0.622
0.636
0.570
0.65512345
0.66812345
0.66312345
0.67212345
LaMP-2: Personalized
Movie Tagging
Accuracy ↑
0.359
0.387
0.377
0.396
0.375
0.391135
0.396135
0.4051235
0.43012345
F1 ↑
0.276
0.306
0.295
0.304
0.299
0.300135
0.306135
0.3141235
0.33912345
LaMP-3: Personalized
Product Rating
MAE ↓
0.308
0.298
0.296
0.299
0.314
0.286145
0.29015
0.28212345
0.26412345
RMSE ↓
0.611
0.611
0.605
0.616
0.614
0.591145
0.60415
0.58512345
0.56812345
LaMP-4: Personalized
News Headline Generation
ROUGE-1 ↑
0.176
0.186
0.189
0.183
0.190
0.1911
0.1871
0.1901
0.20312345
ROUGE-L ↑
0.160
0.171
0.173
0.169
0.176
0.1771
0.1721
0.1761
0.18612345
LaMP-5: Personalized
Scholarly Title Generation
ROUGE-1 ↑
0.478
0.477
0.475
0.483
0.478
0.475
0.477
0.4831235
0.480
ROUGE-L ↑
0.428
0.427
0.426
0.433
0.428
0.427
0.428
0.4311235
0.429
LaMP-6: Personalized
Email Subject Generation
ROUGE-1 ↑
0.335
0.412
0.403
0.401
0.394
0.394
0.4151345
0.42612345
0.43312345
ROUGE-L ↑
0.319
0.398
0.389
0.386
0.381
0.381
0.4001345
0.41112345
0.41812345
LaMP-7: Personalized
Tweet Paraphrasing
ROUGE-1 ↑
0.449
0.446
0.444
0.440
0.446
0.4484
0.441
0.4502345
0.46112345
ROUGE-L ↑
0.396
0.394
0.393
0.390
0.395
0.3974
0.391
0.4002345
0.40912345
# function. The modification is as follows:
(8)
where ˆ𝑦is the prediction and 𝑦is the target output. This reward function normalizes the MAE score by measuring its distance from the worst score achievable based on the model’s prediction.5 In this scenario, a correct prediction by the model results in a score of 1, while in the worst-case prediction, it receives a score of 0. In this paper, we use the Adam optimizer [24] with a learning rate of 10−5. We dedicate 5% of the training steps to warmup with a linear scheduler. We also use gradient clipping with the value of 1. To accommodate the task requirements, we set the maximum input and output lengths to 512 tokens for LLMs following Salemi et al. [43]. However, we use the maximum input length of 1024 for retrieval selection in order to incorporate a prompt and the corresponding output from the LLM. We train the retrieval models for 10 and the retriever selection models for 20 epochs. In all experiments, following Salemi et al. [43], we utilize FlanT5-XXL[8]– an instruction-tuned open-source LLM with 11B parameters. We
use a beam size of 4 in beam search for text generation [18]. The effective batch size in all experiments is set to 64 (8 accumulation steps with batch size 8). We performed all the experiments on a single A100 Nvidia GPU with 80GB memory and 128GB of RAM. In all experiments, following Salemi et al. [43], we use the nontemplate parts of the LLM input 𝑥as the query for personal document retrieval. We followed Salemi et al. [43] for prompt templates (i.e., 𝜙𝑝) for each dataset in LaMP. They are listed in Table 2. In all experiments, the process of creating personalized prompts for evaluating models involves retrieving four items from the user profile. In crafting documents from each user profile, we adhere to the approach established in [43], appending the date of the document to it. The date is prefixed with date: [date]. For implementing BM25, we use the rank_bm25 library.6 All the neural models in this paper are implemented using the PyTorch library [38]. Baseline Methods. We compare the proposed approaches against the following retrieval models for personalized text generation. • No Personalization: We employ FlanT5-XXL [8] as a nonpersonalized baseline. In this baseline, the model is presented with the original task’s input without any modification.
 Personalized Baselines: Following Salemi et al. [43], we utilize BM25 [41], Recency, and Contriever [19] to retrieve items from the user profile for LLM personalization. There are, of course, many other neural ranking models that may outperform these baselines on some retrieval benchmarks. However, it is important to note that our optimization approaches can be applied to any neural ranking model, including any missing baseline from this list. That being said, we do not aim at comparing different model architectures, instead we aim at demonstrating the impact of our optimization methods. Note that no other retrieval models have ever been used on LaMP and, to the best of our knowledge, this list consists of all methods in the retrieval-augmented LLM personalization literature. Furthermore, we apply Reciprocal Rank Fusion (RRF) [9] to integrate the retrieval lists generated by all the retrievers. This fusion-based approach is employed for comparison with our retriever selection method. Subsequently, we employ these retrieved items to formulate a personalized input prompt for FlanT5-XXL.
# 6.2 Empirical Results
This section provides empirical evidence to answer research questions that shed light into the proposed approaches in this paper.
How does personalization using the proposed approaches impact text generation performance? To answer this question, we compare our methods with the non-personalized baseline, i.e., FlanT5-XXL without augmentation with personal information. Table 3 presents the results on the LaMP benchmark. LLM Personalization using both ROPG-RL and ROPG-KD improves the performance on LaMP-1, LaMP-2, LaMP-3, LaMP-4, and LaMP-6. After applying the retrieval model selection methods (RSPG-Pre and RSPG-Post), the non-personalized model is beaten on all datasets and in terms of all metrics. The performance gains are statistically significant in almost all cases. This is an important finding in the sense that no personalized baseline could perform better than a non-personalized LLM on LaMP-7, while RSPG-Pre and RSPG-Post demonstrate that personalized LLMs can ultimately demonstrate performance gain on these datasets. This finding also suggests the impact of retrieval augmentation for the purpose of LLM personalization.
# How do ROPG optimization algorithms impact text genera To answer this, we must compare ROPG-RL
tion performance? To answer this, we must compare ROPG-RL and ROPG-KD with the Contriever baseline, since they are initialized with the Contriever model and fine-tuned using our proposed ROPG-RL and ROPG-KD algorithms. The results in Table 3 suggest that applying ROPG-RL to Contriever yields performance gain on LaMP-1, LaMP-3, LaMP-4, and LaMP-7. ROPG-KD additionally outperforms Contriever on LaMP-6. Notably ROPG-KD performs better than ROPG-RL on the tasks with binary feedback from the language model (LaMP-1 and LaMP-2). Comparing ROPG-RL and ROPG-KD suggests there is no clear winner among them. This once again attests that each personalization task have different requirements, thus motivating the need for retrieval selection in retrieval-augmented LLM personalization.
To assess the efficacy of retriever selection, we measure its success rate in selecting the best performing retriever in the retriever
Table 4: The success rate of models in selecting the best performing retriever for each input. The superscript ∗indicates significant improvement over the best baseline (𝑝< 0.05).
<div style="text-align: center;">Table 4: The success rate of models in selecting the best performing retriever for each input. The superscript ∗indicates significant improvement over the best baseline (𝑝< 0.05).</div>
Dataset
Success Rate
WIG
NQC
𝜎max
𝜎50%
RSPG-Pre
RSPG-Post
LaMP-1: Personalized
Citation Identification
0.858
0.824
0.848
0.847
0.865∗
0.874∗
LaMP-2: Personalized
Movie Tagging
0.908
0.918
0.910
0.910
0.936∗
0.962∗
LaMP-3: Personalized
Product Rating
0.896
0.890
0.896
0.894
0.903
0.920∗
LaMP-4: Personalized
News Headline Generation
0.401
0.404
0.394
0.398
0.401
0.447∗
LaMP-5: Personalized
Scholarly Title Generation
0.562
0.572
0.562
0.557
0.600∗
0.577
LaMP-6: Personalized
Email Subject Generation
0.613
0.606
0.614
0.617
0.633∗
0.641∗
LaMP-7: Personalized
Tweet Paraphrasing
0.851
0.840
0.852
0.851
0.860
0.898∗
<div style="text-align: center;">Dataset</div>
pool R for each input. Note that for inputs with multiple bestperforming retrieval models, a selection is considered successful if any of them is selected. In this experiment, we incorporated several unsupervised Query Performance Prediction (QPP) methods, including WIG [61], NQC [12], and 𝜎max and 𝜎x% [11], to perform a comparative analysis with our proposed method. To accomplish this, we assign the task to each QPP approach of providing a score to the retrieved results for each retriever. The retriever with the highest score is then selected for that particular input. It is important to note that since Recency does not provide score for retrieved results, we consider the reciprocal rank of each document as its score. In no personalization retriever, we assign a score of zero to all items in the profile. It is crucial to highlight that the utilization of supervised QPP methods, such as BERT-QPP [2], was not feasible due to their reliance on query-document relevance labels, which are unavailable in our datasets. The results in Table 4 demonstrate the superior performance of both the RSPG-Pre and RSPG-Post across nearly all datasets. Specifically, RSPG-Post achieves a significant improvement over all baselines for the all datasets, except for the LaMP-5 dataset. In this dataset, RSPG-Pre shows a significant improvement compared to the baselines. Likewise, RSPG-Pre outperforms all baselines in all datasets except LaMP-4, with significant improvements observed in LaMP-1, LaMP-2, LaMP-5, and LaMP-6. Overall, the outcomes of this experiment suggest that the proposed approach for retrieval model selection consistently outperforms the baselines. The results also indicate that for all classification datasets (i.e., LaMP-1, LaMP-2, and LaMP-3) and LaMP-7 for generation, both preand post-generation models achieve a success rate of over 80%. This suggests that the model has to some extent successfully learned to choose the most suitable retriever for each input. Conversely, for the remaining text generation datasets (i.e., LaMP-4, LaMP-5, and LaMP-6), the accuracy is lower, ranging from 40% to 65%. We attribute this to the inherent complexity of text generation tasks compared to text classification. Comparing RSPG-Pre and RSPGPost, the latter exhibits higher success rate in all tasks except LaMP5. This suggests that employing the generated output in retrieval selection can have a substantial impact on the performance. Looking back to the results in Table 3, the post-generation retrieval selection model (RSPG-Post) performs better than the pregeneration selection model (RSPG-Pre) on six out of seven datasets;
<div style="text-align: center;">Table 5: Studying the impact of ROPG algorithms on the end-to-end performance of our pipeline with retrieval model selection In this table, the superscript ∗indicates significant improvement over w/o ROPG approach (𝑝< 0.05).</div>
In this table, the superscript ∗indicates significant improvement over w/o ROPG approach ().
Dataset
Metric
RSPG-Pre
RSPG-Post
Oracle
w/o ROPG
w/ ROPG
w/o ROPG
w/ ROPG
Lower-bound
Upper-bound
LaMP-1: Personalized Citation Identification
Accuracy ↑
0.646
0.663∗
0.644
0.672∗
0.381
0.798
LaMP-2: Personalized Movie Tagging
Accuracy ↑
0.403
0.405
0.425
0.430
0.296
0.468
F1 ↑
0.311
0.314
0.330
0.339
0.216
0.380
LaMP-3: Personalized Product Rating
MAE ↓
0.287
0.282
0.269
0.264
0.450
0.181
RMSE ↓
0.595
0.585
0.577
0.568
0.772
0.462
LaMP-4: Personalized News Headline Generation
ROUGE-1 ↑
0.189
0.190
0.191
0.203∗
0.103
0.269
ROUGE-L ↑
0.174
0.176
0.177
0.186∗
0.098
0.243
LaMP-5: Personalized Scholarly Title Generation
ROUGE-1 ↑
0.478
0.483∗
0.475
0.480
0.388
0.548
ROUGE-L ↑
0.428
0.431∗
0.427
0.429
0.349
0.492
LaMP-6: Personalized Email Subject Generation
ROUGE-1 ↑
0.426
0.426
0.394
0.433∗
0.239
0.511
ROUGE-L ↑
0.412
0.411
0.381
0.418∗
0.228
0.492
LaMP-7: Personalized Tweet Paraphrasing
ROUGE-1 ↑
0.449
0.450
0.448
0.461∗
0.410
0.470
ROUGE-L ↑
0.398
0.400
0.397
0.409∗
0.361
0.417
LaMP-5 is the exception, which is explained by the results in Table 4. RSPG-Pre consistently outperforms the baselines significantly in all classification tasks (LaMP-1, LaMP-2, and LaMP-3) as well as in LaMP-6. In the remaining datasets, RSPG-Pre achieves comparable or superior results to the baselines, although the differences are not statistically significant. Finally, RSPG-Post leads to the best personalized text generation performance on six out of seven datasets. What is the method that results in the highest personalized text generation performance? According to Table 3, RSPG-Post performs best on six out of seven datasets (all but LaMP-5). Contriever, on the other hand, demonstrates the highest performance on LaMP-5. The performance gains by RSPG-Post on all the remaining six datasets are statistically significant, according to a two-tailed paired t-test for generation and ordinal classification datasets and McNemar test for binary and categorical text classification datasets. What is the impact of ROPG algorithms on retrieval selection results? To investigate the impact of training personalized retrievers using the proposed ROPG algorithms (i.e., both ROPGRL and ROPG-KD) on the end-to-end performance of the pipeline with retrieval selection, we conduct an analysis by excluding the fine-tuned retrievers from the retriever pool R. The outcomes of this experiment along with the Oracle performance (both lower and upper bound for retrieval selection) are reported in Table 5. The results indicate that the models without ROPG in its retriever pools achieve lower performance on all datasets in both pre- and post-generation settings. The only exception is the RSPG-Pre results on LaMP-6, where including ROPG algorithms does not make any significant impact. This suggests that our ultimate performance gain is not just because of the retrieval selection models; Instead, the retrieval optimization approaches presented in Section 4 are effective in enhancing the end-to-end performance of the pipeline. Finally, a comparison between our results and the Oracle performance in Table 5 provides insight into the potential for further improvements in retrieval selection. For instance, in LaMP-3 and LaMP-4, our best performing method only achieves 68.3% and 75.4% of the Oracle’s upper-bound, respectively. This suggests that there are still substantial room for improvement. However, the corresponding numbers for LaMP-1, LaMP-2, LaMP-5, LaMP-6,
and LaMP-7 are 84.2%, 91.8%, 88.1%, 84.7%, and 98.0%, respectively, suggesting that our best performing retrieval selection method is performing very close to the upper-bound performance.
# 7 CONCLUSIONS AND FUTURE WORK
This paper explored personalization of LLMs through a retrieval augmentation pipeline with a focus on optimizing the retrieval component. We introduced two solutions for optimizing ranking models by soliciting personalized feedback from the language model, one based on reinforcement learning where the reward function is defined based on the personalized text generation quality, and another based on knowledge distillation from the language model to the retrieval model. Subsequently, we observed that personalization tasks can benefit from different retrieval models, depending on specific needs and requirements. Given this observation, we developed a pre-generation and a post-generation retriever selection model. Evaluation on seven diverse personalization tasks from the LaMP benchmark showed that our proposed methods outperform competitive baselines on six out of seven datasets with statistically significant improvements. Through careful ablation studies, we demonstrate the impact of each component used our pipeline. In this work, we solely focused on optimizing and selecting the ranking models for LLM personalization. One limitation of this work lies in the use of static templates for generating prompts for the LLM. In the future, we aim at optimizing the prompt generation component using the feedback obtained from the downstream LLM performance. In addition, all datasets in the LaMP benchmark focus on short text generation tasks. We will explore personalized long text generation methods in the future.
# ACKNOWLEDGMENT
This work was supported in part by the Center for Intelligent Information Retrieval, in part by Lowe’s, in part by an Amazon Research Award, Fall 2022 CFP, in part by an award from Google, and in part by an award from Microsoft. Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect those of the sponsor.
# REFERENCES
