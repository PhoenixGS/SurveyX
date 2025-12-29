# Enhancing Recommendation Diversity by Re-ranking with Large Language Models
# ommendation Diversity by Re-ranking with Large Lan
# Enhancing Recommendation Diversity by Re-ranking with Large Language
DIEGO CARRARO, Insight Centre for Data Analytics, School of Computer Science & IT, University College Cork Ireland DEREK BRIDGE, Insight Centre for Data Analytics, School of Computer Science & IT, University College Cork Ireland
It has long been recognized that it is not enough for a Recommender System (RS) to provide recommendations based only on their relevance to users. Among many other criteria, the set of recommendations may need to be diverse. Diversity is one way of handling recommendation uncertainty and ensuring that recommendations offer users a meaningful choice. The literature reports many ways of measuring diversity and improving the diversity of a set of recommendations, most notably by re-ranking and selecting from a larger set of candidate recommendations. Driven by promising insights from the literature on how to incorporate versatile Large Language Models (LLMs) into the RS pipeline, in this paper we show how LLMs can be used for diversity re-ranking. We begin with an informal study that verifies that LLMs can be used for re-ranking tasks and do have some understanding of the concept of item diversity. Then, we design a more rigorous methodology where LLMs are prompted to generate a diverse ranking from a candidate ranking using various prompt templates with different re-ranking instructions in a zero-shot fashion. We conduct comprehensive experiments testing state-of-the-art LLMs from the GPT and Llama families. We compare their re-ranking capabilities with random re-ranking and various traditional re-ranking methods from the literature. We open-source the code of our experiments for reproducibility. Our findings suggest that the trade-offs (in terms of performance and costs, among others) of LLM-based rerankers are superior to those of random re-rankers but, as yet, inferior to the ones of traditional re-rankers. However, the LLM approach is promising. LLMs exhibit improved performance on many natural language processing and recommendation tasks and lower inference costs. Given these trends, we can expect LLM-based re-ranking to become more competitive soon.
arXiv:2401.11506v2
and Phrases: Recommender Systems, Large Language Models, Diversity, Re-ranking
# 1 INTRODUCTION
Large Language Models (LLMs) have rapidly become a breakthrough technology since the advent of their popular pioneers GPT [49] and BERT [13], introduced in 2018. Since then, many more LLMs with enhanced capabilities have been proposed, such as ChatGPT, Bard and Llama. They can perform various language-related tasks, including, for example, translation, summarization and conversation; and they give the appearance of understanding complex contexts and exhibiting reasoning, planning and problem-solving capabilities [35, 58, 70]. LLMs are typically pre-trained with large datasets of text to serve as general-purpose models and then adapted to different downstream tasks and domains by a
Authors’ addresses: Diego Carraro, Insight Centre for Data Analytics, School of Computer Science & IT, University College Cork, Ireland, diego.carraro@ insight-centre.org; Derek Bridge, Insight Centre for Data Analytics, School of Computer Science & IT, University College Cork, Ireland, derek.bridge@ insight-centre.org.
Authors’ addresses: Diego Carraro, Insight Centre for Data Analytics, School of Computer Science & IT, University College Cork, Ireland, diego.carraro@ insight-centre.org; Derek Bridge, Insight Centre for Data Analytics, School of Computer Science & IT, University College Cork, Ireland, derek.bridge@
Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses, contact the owner/author(s). © 2024 Copyright held by the owner/author(s). Manuscript submitted to ACM
variety of approaches [44] — prompting, Supervised Fine Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) being among the most widely used. Prompting refers to giving the LLM a specific input (e.g. a question, a sentence, a set of instructions), known as a prompt, designed to drive the LLM to generate a desired output. Supervised Fine-Tuning adapts the model by further training the LLM on a task-specific labelled dataset. Reinforcement Learning from Human Feedback (RLHF) [77] fine-tunes the model using human preference data and is typically used in settings where large amounts of labelled training data are not available or where the domain requirements are human-centred. Given their abilities and their versatility, researchers in Recommender Systems (RSs) have proposed different ways of integrating LLMs into a recommendation pipeline, for example, to enhance data augmentation, perform feature engineering, design scoring functions, directly provide recommendations, and furnish explanations of recommendations [19, 40]. Some researchers have taken a unified approach, where the entire recommendation task is framed as a textgeneration task: a single LLM can perform all the functionalities mentioned above (e.g. [11, 21]), showing how well LLMs can generalise across different recommendation settings. All the work on RSs we have cited focuses on providing recommendations that are as relevant as possible to a user. However, long before the use of LLMs in RSs, the RS literature recognized that relevant recommendations might not be enough to satisfy a user fully: providing a set of recommendations that is diverse (different from one another) or recommendations that are novel (new or previously unexplored), serendipitous (surprising, unexpected, and going beyond the user’s typical preferences) and fair (not discriminating between certain groups of users or items) is also crucial, see, e.g., [31, 59, 65]. Strategies that optimize for these so-called beyond-accuracy objectives have been proposed for traditional RSs, while the topic is under-investigated for LLM-based RSs. The goal of this paper is to contribute to filling this gap. Our work investigates enhancing the top-푛recommendation diversity by re-ranking using general-purpose stateof-the-art LLMs. Other works propose to re-rank recommendations leveraging LLMs to improve relevance, but to the best of our knowledge, we are the first ones focusing on beyond-relevance objectives. In particular, our contributions are tailored to the following research questions: • RQ1: Can LLMs interpret the building blocks of RS diversification re-ranking, i.e. item diversity and item re-ranking? • RQ2: Given an initial candidate set of items ranked by their relevance to a user, can general-purpose LLMs re-rank these items to increase their recommendation diversity? • RQ3: How do different zero-shot prompts affect the re-ranking performance in terms of relevance and diversity? • RQ4: What are the trade-offs of LLM-based and traditional greedy diversification re-rankers in terms of performance, costs and other important aspects such as control over data, memory footprint and generalisation across domains, among others? To answer RQ1, we run an informal preliminary study where, first, we ask ChatGPT to perform some re-ranking tasks, supplementing existing positive evidence already provided in the information retrieval [55] and RS [27] domains. Then, we prompt the model to compare two lists of items in terms of their item diversity. Encouraged by positive findings in these informal studies, where the LLM exhibits re-ranking capabilities and knowledge of item diversity, we then propose a more rigorous methodology to answer RQ2-4. First, we frame diversification re-ranking as a text generation task in which we prompt an LLM in a zero-shot fashion to generate a more diverse ranking from a candidate ranking. We design eight prompt templates that contain different instructions for re-ranking to explore different ways to approach the generation. We explore a wide range of templates that differ in their complexity, from simple high-level task descriptions to more advanced ones where we provide Manuscript submitted to ACM
variety of approaches [44] — prompting, Supervised Fine Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) being among the most widely used. Prompting refers to giving the LLM a specific input (e.g. a question, a sentence, a set of instructions), known as a prompt, designed to drive the LLM to generate a desired output. Supervised Fine-Tuning adapts the model by further training the LLM on a task-specific labelled dataset. Reinforcement Learning from Human Feedback (RLHF) [77] fine-tunes the model using human preference data and is typically used in settings where large amounts of labelled training data are not available or where the domain requirements are human-centred. Given their abilities and their versatility, researchers in Recommender Systems (RSs) have proposed different ways of integrating LLMs into a recommendation pipeline, for example, to enhance data augmentation, perform feature engineering, design scoring functions, directly provide recommendations, and furnish explanations of recommendations [19, 40]. Some researchers have taken a unified approach, where the entire recommendation task is framed as a textgeneration task: a single LLM can perform all the functionalities mentioned above (e.g. [11, 21]), showing how well LLMs can generalise across different recommendation settings. All the work on RSs we have cited focuses on providing recommendations that are as relevant as possible to a user. However, long before the use of LLMs in RSs, the RS literature recognized that relevant recommendations might not be enough to satisfy a user fully: providing a set of recommendations that is diverse (different from one another) or recommendations that are novel (new or previously unexplored), serendipitous (surprising, unexpected, and going beyond the user’s typical preferences) and fair (not discriminating between certain groups of users or items) is also crucial, see, e.g., [31, 59, 65]. Strategies that optimize for these so-called beyond-accuracy objectives have been proposed for traditional RSs, while the topic is under-investigated for LLM-based RSs. The goal of this paper is to contribute to filling this gap. Our work investigates enhancing the top-푛recommendation diversity by re-ranking using general-purpose stateof-the-art LLMs. Other works propose to re-rank recommendations leveraging LLMs to improve relevance, but to the best of our knowledge, we are the first ones focusing on beyond-relevance objectives. In particular, our contributions are tailored to the following research questions: • RQ1: Can LLMs interpret the building blocks of RS diversification re-ranking, i.e. item diversity and item re-ranking?
To answer RQ1, we run an informal preliminary study where, first, we ask ChatGPT to perform some re-ranking tasks, supplementing existing positive evidence already provided in the information retrieval [55] and RS [27] domains. Then, we prompt the model to compare two lists of items in terms of their item diversity. Encouraged by positive findings in these informal studies, where the LLM exhibits re-ranking capabilities and knowledge of item diversity, we then propose a more rigorous methodology to answer RQ2-4. First, we frame diversification re-ranking as a text generation task in which we prompt an LLM in a zero-shot fashion to generate a more diverse ranking from a candidate ranking. We design eight prompt templates that contain different instructions for re-ranking to explore different ways to approach the generation. We explore a wide range of templates that differ in their complexity, from simple high-level task descriptions to more advanced ones where we provide Manuscript submitted to ACM
additional information about the items and further re-ranking guidelines. We also introduce ways of addressing LLM hallucinations (i.e. where the model produces invalid recommendations), a common problem that is usually ignored by most of the LLM literature on RSs. Then, we run experiments on an anime movie dataset and on a book dataset where a Matrix Factorization RS optimized for relevance provides candidate recommendations. We apply our proposed LLM-based diversification reranking to these recommendations using state-of-the-art LLMs. Specifically, we use proprietary models from OpenAI1 (namely, ChatGPT and InstructGPT) and also open-source models from Meta’s Llama family2 (namely, Llama2-7B-Chat and Llama2-13B-Chat). We assess to what extent recommendation diversity improves and how relevance is affected (RQ2, RQ3). We compare with a random re-ranker and some traditional state-of-the-art greedy strategies that intelligently re-arrange items to balance relevance and diversity. To address RQ4, we run a comprehensive analysis beyond performance for both approaches, where we discuss their trade-offs in terms of costs, control over data, and memory footprint, among other aspects. Our results demonstrate that, in both datasets, the LLM-based re-rankers we proposed can interpret the re-ranking task, sacrificing some relevance to improving diversity. While LLM-based re-rankers achieve better relevance/diversity trade-offs than random re-ranking, they are still inferior to traditional re-rankers, particularly in relevance-aware metrics, due to issues such as drawing lower-ranked items and the presence of random recommendations. OpenAI’s models, particularly ChatGPT, outperform Meta’s models, such as Llama2-13B-Chat, aligning with existing literature that credits OpenAI models with more advanced capabilities. The effectiveness of different prompt templates varies, indicating the need for tailored prompt design, with feature-aware templates generally performing better. Finally, the traditional greedy approach are faster and less resource-intensive, making the costs and constraints of LLM-based reranking a significant barrier for practical use. Nevertheless, our study shows that LLM-based has considerable promise. It will become more competitive as LLMs improve and as the costs of using them continues to fall. The remainder of this paper is structured as follows. Section 2 reviews related work; Section 3 presents both the preliminary study and our LLM-based re-ranking approach; Section 4 describes experiments and Section 5 analyses their results. Section 6 offers a broader perspective on the trade-offs of LLM-based and greedy re-ranking approaches. Finally, 7 wraps up our findings, discusses the limitations of our study and outlines future directions.
# 2 RELATED WORK
# 2.1 Measuring recommendation diversity
The most common way to measure the diversity of a set of items, such as a set of 푛recommendations, uses item dissimilarity. In particular, the average [53] or aggregate [76] pairwise distance between items in the set are popular diversity measures of this kind. Different ways of quantifying item distance have been used. In content-based systems, for example, the complement of the Jaccard similarity on sets of features such as genres [61], the complement of the cosine similarity on term vectors [18], and distance in a taxonomy [76] are possible methods. When items have vectors of user ratings, item distance can be computed using the complement of Pearson correlation [61], the complement of cosine similarity [51] or Hamming distance [34]. Item distance can also be computed on the latent item feature vectors in Matrix Factorization approaches [52, 54, 62].
1https://openai.com/ 2https://llama.meta.com/
Some researchers have argued that the traditional diversity metrics do not correlate well with users’ perceptions and have proposed alternatives, especially when item features are based on item genres. For example, these new metrics also account for coverage of item feature values, redundancy and the size of the recommendation list [60]. There are also metrics that consider both diversity and relevance at the same time: they quantify the trade-off between diversity and relevance with a single metric. 훼-NDCG, for example, was introduced by [10] to evaluate information retrieval systems, where the score of a retrieved document is penalized if it shares features with documents ranked higher in the list of retrieved documents. This metric has also been used in RS research in [6, 33]. Inspired by 훼-NDCG, Vargas & Castells weight a diversity score with item relevance, under the assumption that diversity on irrelevant items does not play a major role in satisfying a user [61]. They designed a framework that allows one to define several metrics that account for a diversity and relevance trade-off based on a ranking discount. In our experiments, we use genres as item features, and we employ their framework and some metrics from it, such as 훼-NDCG, Expected Intra-List Diversity, and relevancy-aware Subtopic Recall; see Section 4.4. Finally, for completeness, we note that metrics have also been defined for other types of diversity, such as temporal diversity metrics that capture the extent to which recommendations generated for the same user differ over time, or metrics that take a system-centric, rather than user-centric, perspective by analysing how recommendations differ across users. We refer to [31] and [7] for comprehensive surveys of diversity-aware metrics.
# 2.2 Increasing recommendation diversity
As well as measuring diversity, researchers have proposed ways of increasing the diversity of a set of recommendations. Diversification methods can be categorised into two main groups: diversity modelling and diversity post-processing. The first group of works includes recommendation models that are directly optimized to produce diverse recommendations. Most of these RSs are derivations or extensions of the popular and effective Matrix Factorization (MF) model. For example, Shi et al. [52] designed an objective function that balances the expected relevance and variance of the recommendation list. The variance level is obtained from the latent factor vectors of an MF model and correlates with a user’s spectrum of tastes. In other words, the latent factors of a user who rates diverse items have higher variance compared to a user who rates similar items; the former user will consequently receive more diversified recommendations from the model than the latter user does. Another example is [28], which modified the learning function of a pairwise learning-to-rank approach (for implicit feedback datasets) to account for item dissimilarities. A third example is [54], which proposed a revised pairwise learning-to-rank model that works at the item set level (rather than the individual item level): their model is trained by comparing item sets using both relevance and diversity criteria to establish the pairwise ranking. The second group of works includes approaches that apply a post-processing step (typically, using a greedy strategy) that refines an initial list of recommendations from a conventional RS to obtain a final set of recommendations that balances relevance and diversity. More formally, these techniques typically produce a final list of recommended items 푅퐿of size 푛from a larger set of candidate recommendations 퐶퐿of size 푚(with 푚> 푛), where 퐶퐿is generated by a baseline recommendation algorithm that ranks items based on their relevance. The vast majority of the RS literature on this uses greedy re-ranking, which involves an iterative process where items are selected from퐶퐿one by one and added to 푅퐿. At each step, the greedy algorithm selects the candidate item that maximizes a re-ranking objective function 푓표푏푗, like the one defined in Eq. 1. Here, 푓표푏푗is a linear combination of two scores, i.e. a candidate item’s relevance and the diversity this item brings when added to the current version of 푅퐿. In particular, in the equation, 푟푒푙(푖) denotes the relevance score of the item 푖to a user. 푑푖푣(푖,푅퐿) denotes the diversity of 푅퐿when the item 푖is included in 푅퐿and it Manuscript submitted to ACM
typically makes use of a distance function between items. Finally, the hyperparameter 휆∈[0, 1] controls the trade-off between the influence of relevance and diversity during the re-ranking.
푓표푏푗(푖, 푅퐿) = 휆· 푟푒푙(푖) + (1 −휆) · 푑푖푣(푖,푅퐿)
Greedy re-ranking has been widely adopted in the RS literature because it is easy to implement and integrate into existing recommender system pipelines. Indeed, its relevance 푟푒푙(푖) and diversification 푑푖푣(푖,푅퐿) components can be adapted based on the application domain, and their trade-off can be explicitly controlled with 휆. Maximal Marginal Relevance (MMR) [5] is one of the first diversification techniques of this kind. It was originally proposed for the information retrieval domain but later adopted in the RS literature, e.g. [14, 62]. In MMR, 푑푖푣(푖, 푅퐿) is computed as the negative of 푖’s maximum similarity to items already in 푅퐿. In a case-based recommender scenario, Smyth & McClave implemented 푟푒푙(푖) as the similarity between the user’s query and an item or case 푖, while 푑푖푣(푖,푅퐿) was the average pairwise distance between 푖and the items or cases already in 푅퐿, the distance being the complement of their similarity [53]. In a conversational recommender scenario, Kelly & Bridge took 푟푒푙(푖) as the predicted item’s relevance and 푑푖푣(푖,푅퐿) is again the average pairwise distance between 푖and the items in 푅퐿, but this time using the normalized Hamming distance of the two items’ binary rating vectors [34]. A more recent stream of research arose from Vargas’ work [59], which proposed greedy re-ranking strategies inspired by Intent-Aware (IA) diversification from the field of information retrieval [1]. In essence, IA methods in information retrieval seek to optimize the relevance and diversity trade-off by retrieving documents that are relevant to the user’s query but which also cover different aspects of the query (e.g. different meanings of an ambiguous query). In the RS domain, IA methods seek to retrieve items that are predicted to be relevant to the user while covering different aspects such as item feature values or user tastes. For example, Vargas presented explicit Query Aspect Diversification (xQuAD) [62] and its variant Relevance-based xQuAD (RxQuAD) [63]. xQuAD focuses on maximizing the probability that a user will select at least one of the recommended items by building an 푅퐿that covers all the different user aspects. RxQuAD introduces a relevance model into xQuAD to maximise relevance rather than the probability of choosing a single item. There are also works that perform post-processing but do not use greedy re-ranking. Instead, they typically aim at solving optimization problems to find the optimal ranking for balancing relevance and diversity; some examples are [29, 51, 71]. In the experiment described in this paper, we use MMR, xQuAD and RxQuAD as our greedy re-ranking baselines against which we compare the performance of our LLM-based re-rankers (see Section 4).
# 2.3 LLMs and recommender systems
LLMs in the RS domain can be categorized into supporting recommendations or dir
Broadly speaking, work on using LLMs in the RS domain can be categorized into supporting recommendations or directly providing recommendations. Representative examples from the first group of works use an LLM to augment training data, extract features, and explain recommendations. Borisov et al. [4] exploit LLMs to generate synthetic examples from tabular data; Liu et al. [42] prompt an LLM to craft news summaries, user profiles, and synthetic news for a news recommender; Mysore et al. [45] propose an LLM-based pipeline that converts user-item interaction data, commonly used to train collaborative filtering systems, into datasets to train narrative-driven recommenders. Inspired by the Retrieval-Augmented Generation (RAG) paradigm [20], Di Palma [15] advocates a comprehensive approach, which he calls a Retrieval-augmented
(1)
Recommender System, which combines the strengths of retrieved information about items with LLMs to enhance the recommendation task. When used as a feature extractor, an LLM is typically fed with textual user features, item features or user-item interaction data (e.g. user behaviour or reviews) and returns vectorized embeddings that can be used by a traditional recommender model (often a neural network) to perform various recommendation tasks. Features are extracted, for example, from code snippets [50], from news [72], from tweets [73], and from documents [43], and applied to enhance cross-domain recommendation [16, 25, 26]. As they can generate fluent and coherent text, LLMs are also powerful tools used to provide explanations in natural language for different kinds of recommendations, e.g. for products in e-commerce [8, 11], for leisure products [38], and for music, video games and food [24]. Works in the second group use LLMs to provide recommendations directly, either by generating item suggestions or by giving scores to user-item pairs. When LLMs generate item suggestions, they can be in the form of item names [11, 37] or identifiers [21, 47]; and the LLM can make next-item recommendations [21, 64] or it can rank a set of candidate items [11]. When LLMs generate scores, they can be in the form of predicted ratings [21, 36], click-through rates and conversion rates [11], or preferences [3, 75]. Other works, different from the two groups we have reviewed so far, are more related to re-ranking recommendations, akin to what we propose in this paper. We briefly review them in the following and we draw explicit comparison with our work in Section 3.2. Dai et al. [12], Li et al. [39] and Hou et al. [27] all use several models from the ChatGPT family to rank an unordered candidate set of items in a zero-shot scenario for domains such as movies, books, music and news, additionally leveraging historical user interactions in the prompts. Sun et al. [55] investigate the capabilities of ChatGPT and GPT-4 at re-ranking candidate documents retrieved by a traditional information retrieval system (in a zero-shot generation and by distilling a specialized model). Suzgun et al. [56] use LLMs for textual style transfer: the system first generates a set of candidate texts in the target style (by zero-shot or few-shot prompting with GPT2) and then re-ranks them according to three types of scores provided by different LLMs. All these works show promising results for LLM-based re-ranking (especially for ChatGPT) but they all struggle to outperform state-of-the-art competitors. Moreover, there is no consistent methodology for evaluating LLM-based re-rankers (taking into account, e.g., invalid recommendations). In Section 3.2, we will compare in detail our work with this literature, highlighting similarities and differences for each component of our system.
# 3 RECOMMENDATION DIVERSIFICATION USING LLMS FOR RE-RANKING
Our work investigates recommendation diversification by using an LLM to re-rank a set of pre-computed candidate recommendations. We frame this task as a zero-shot text generation task and discuss more advanced approaches in Section 7. For re-ranking, we use and compare two state-of-the-art families of LLMs, i.e. ChatGPT/InstructGPT and Llama-2-Chat, which, using SFT and RLHF, have been fine-tuned from their foundation models (GPT and Llama respectively) to follow instructions.
# 3.1 A preliminary study
Although there is extensive evidence that both ChatGPT and Llama-2-Chat can successfully perform many different tasks, to the best of our knowledge, little is known about whether they are good at re-ranking and whether this can be done in the context of diversity in RSs. The closest works to ours that offer some positive evidence in these regards are [12, 39], which explore ranking an unordered candidate set of items; and [55], where the task is re-ranking a candidate Manuscript submitted to ACM
list of items previously ranked by a retrieval system. These works focus on ChatGPT only. To the best of our knowledge, there is no published research to assess whether LLMs are capable of reasoning about item diversity. Thus, we begin our informal investigations by answering RQ1. More specifically, we find answers to the following: Q1 Can LLMs interpret and perform re-ranking tasks? Q2 Do LLMs have knowledge about item diversity?
Q2 Do LLMs have knowledge about item diversity? We note that this preliminary study is rather informal and is not a comprehensive evaluation of an LLM’s re-ranking capabilities. Instead, it serves as an indication of their potential capabilities towards our larger goal. To answer Q1, we designed three simple prompts asking the LLMs to re-rank a list of the titles of 20 anime movies. (We use anime movies for this preliminary study because they come from one of the datasets we use in the more substantial experiments described in Section 4.) Specifically, we ask the LLMs to re-rank the lists so that they are ordered alphabetically, or by popularity, or by release date. We expect that the complexity of the tasks should affect the re-ranking performance: when re-ranking by popularity and release date, an LLM must leverage additional details about the items to accomplish the task, whereas alphabetical re-ordering can be done by using the items’ titles only. In Appendix A.1, we show three representative examples of the prompts and ChatGPT’s outputs. (Llama-2-Chat’s outputs are similar, so we do not show them.) As these examples demonstrate, ChatGPT can perform the re-ranking tasks with different degrees of accuracy. In particular, alphabetical re-ranking (Q1-P1) is the most accurate, with only one item incorrectly ranked. By contrast, popularity (Q2-P2) and release date (Q3-P3) re-ranking are only partially consistent with the ground-truth rankings. We ran several examples of this kind (with different item lists to re-rank). We compared the ChatGPT outputs with ground-truth rankings. We obtained scores that vary between 0.5 and 0.8 for popularity re-ranking and between 0.35 and 0.7 for release date re-ranking, according to the rank-biased-overlap metric [66]. To answer Q2, we designed one prompt that asks the LLM to indicate which of two lists, each composed of 10 anime titles, is the more diverse. We deliberately do not provide any definition of diversity in the prompt so that the LLM must interpret its meaning for itself. Similarly to Q1, we built examples that exhibit different degrees of complexity: an “easier” prompt that compares list A that covers one genre (e.g. all items are comedies) and list B that covers multiple genres; and a “harder” prompt that compares two lists that both cover multiple genres. Again, we ran several examples of this kind (with different item lists to re-rank). In Appendix A.1, we show two representative examples of the prompts and ChatGPT’s outputs. Q2-P1 is the easier prompt, while Q2-P2 and Q2-P3 are the harder prompts. (Again, Llama-2-Chat’s outputs are similar, so we do not show them.) In each example, ChatGPT’s output first includes its interpretation of the meaning of diversity (i.e. a genre-based definition), followed by the candidate lists augmented with the genres it associates with each item in the list, concluding with the answer to the prompt’s question along with an associated explanation. As expected, for Q2-P1, ChatGPT correctly indicates that list B is the more diverse, and it justifies its answer from a genre-based diversity perspective, i.e. the number of different genres a list covers. For Q2-P2, ChatGPT indicates that list A is the more diverse and again justifies its answer from a genre-based diversity coverage perspective. According to the genres that ChatGPT assigns to the lists, its answer is correct (list A covers 15 genres, and list B 13 genres). However, if we extract ground-truth genres from the dataset that we use in Section 4, then list B is more diverse (list A covers 18 ground-truth genres, and list B 19). Finally, For Q2-P3, ChatGPT indicates that list B is the more diverse (for the same reason above). However, ChatGPT’s conclusion does not correspond with the genres that it assigns because, in fact, it assigns 11 genres to list B
against 12 for list A. The incorrectness of the answer is also confirmed by the ground-truth genres we extracted from the dataset, i.e. 19 for B and 24 for A. This preliminary study suggests that an LLM can re-rank lists of items and can provide insights into their diversity, using its embedded knowledge (without any additional fine-tuning), albeit imperfectly. Moreover, the LLM showed some degree of reasoning when performing the tasks and was able to provide highly structured answers,. Correctness also seems correlated with task difficulty. Again, we emphasis that this preliminary study is not rigorous, and its findings are not measures of LLM abilities. In particular, we loosely defined “correctness” when evaluating answers to Q2. What we take from this informal investigation is an indication that it is worth proceeding to the more rigorous study that we present in the remainder of this paper.
# 3.2 Prompt-based zero-shot diversification re-ranking
Following the previous section’s largely positive findings, we propose a methodology to answer the research question RQ2 presented in Section 1, i.e. whether LLMs can enhance recommendation diversity by re-ranking. We consider the greedy re-ranking framework presented in Section 2.2, where the main components are the candidate list 퐶퐿, the final recommendation list 푅퐿, a relevance function 푟푒푙defined over the items in 퐶퐿, a diversification function 푑푖푣defined over the items in 퐶퐿and the list 푅퐿, and a hyperparameter 휆to control the relevance/diversification trade-off. We also assume the candidate list 퐶퐿is produced by a baseline recommender. In Section 4, we choose a Matrix Factorization model to provide 퐶퐿for our experiments, but in practice this can be done by any recommender that can produce a ranked list of items. We frame the re-ranking task as a text generation task to be accomplished by an LLM in response to specific prompts that request the production of 푅퐿from 퐶퐿. The prompt template comprises three sections, see Prompt 1. The first section of the prompt provides the context and instructions the LLM must follow to generate 푅퐿. The second section provides the output format the LLM must adhere to. The third section provides the candidate list 퐶퐿that the LLM must re-rank. However, except for the use of triple backticks to delimit 퐶퐿, we do not explicitly mark-up the three sections: the tags in Prompt 1 are for the purposes of exposition in this paper, they are not included in the final prompt. The decision to exclude them is in line with what the literature suggests (e.g. [12, 21, 55]). We will now discuss each of the three sections in more detail. Clear instructions are crucial to obtaining the desired output since the LLMs we use are aligned with high-quality instruction-based prompts. We designed eight subtypes of the generic template, each identified by a different re-ranking goal; see Table 1. Each goal maps to a different 휆value, with T1, T4-T7 aiming at equally balancing relevance and diversity (i.e. 휆= 0.5 in traditional greedy re-ranking approaches) and T2, T3, T8 aiming at maximizing diversity only (i.e. 휆= 0). The related research literature proposes alternative ways to model different re-ranking trade-offs. Li et al [39] address the trade-off between recommendation relevance and fairness by explicitly instructing the LLM to select a specific number of candidate items from different groups of items — in their case, popular and unpopular items. Suzgun et al. [56] use a combination of scores from different LLMs to adjust the trade-off between textual similarity, target style strength, and fluency for arbitrary textual style transfer. The prompt template also includes placeholders {m} and {n}. These are replaced with the size of 퐶퐿and 푅퐿. We explain how we choose the values of 푚and 푛in Section 4. The prompt template specifies an output format to guide the LLM’s generation, simplify output parsing (i.e. extracting the final ranking from the LLM’s answer), and reduce its ambiguities. Similarly to [55], we opted for a numbered list Manuscript submitted to ACM
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1b53/1b53f209-88ca-416c-9eb8-6d91643a3b5f.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 1. Mapping of placeholders in the generic prompt template.</div>
Template id
휆휆휆
{goal_string}
{additional_features}
T1
0.5
“balance relevance and diversity”
None
T2
0
“maximize the items’ diversity in the list”
None
T3
0
“maximize the items’ genre-based diversity in the list”
None
T4
0.5
“balance relevance and genre-based diversity”
None
T5
0.5
“balance relevance and diversity”
List of the candidate item’s genres
T6
0.5
“balance relevance and genre-based diversity”
List of the candidate item’s genres
T7
0.5
“balance relevance and diversity.
Guidelines to perform the re-ranking are:
Use the plot summary information of each
item attached in curly bracke”
Candidate item’s plot summary
T8
0
“maximize the books’ diversity in the list.
Guidelines to perform the re-ranking are:
Use the plot summary information of each
item attached in curly bracket”
Candidate item’s plot summary
(with numbers representing an item’s rank, i.e. its relevance in 퐶퐿), followed by the string ->, to help to identify item names when parsing the output. To the best of our knowledge, the literature offers no advice about how to represent ranked lists in a prompt. There have, however, been works in the literature that represent un-ranked candidate lists in prompts, e.g. by identifying items with letters [12] or by simply appending items one after another [39]. Finally, the prompt must include the list 퐶퐿. We represent its items with their names, e.g. anime titles or book titles (see Section 4), similar to what is done in most of the literature (although some works use item identifiers, e.g. Manuscript submitted to ACM
[21]). Items are also numbered in our prompts to report their rank (i.e. their relevance). Also, for templates T5, T6, T7 and T8, which we call feature-aware templates, we replace the {additional_features} placeholders to augment an item with its genres (T5, T6) and its description (T7, T8), attempting to equip the LLM with explicit item features to leverage for re-ranking. (In our experiments in Section 4, we use genres extracted from the datasets and descriptions extracted from prompting ChatGPT.) Feature-aware templates can be seen as a form of RAG-based prompting, where each prompt is augmented with some extra contextual information. The only difference between traditional RAG and our approach is that traditional RAG systems typically retrieve this additional context on-the-fly (i.e. when generation occurs), while we retrieve the whole set of item features in advance of re-ranking (see Section 4). In Appendix A.2, we show an example of a prompt (i.e. after the placeholders in the template have been instantiated). We also show ChatGPT’s output. Templates T1-T8 explore different degrees of complexity and help us in addressing RQ3. However, several other prompting techniques might have been explored — for example, few-shot prompting (i.e. providing task-related demonstrative examples), or Chain-Of-Thoughts (CoT) prompting (i.e. where the LLM is driven to provide the requested output through intermediate steps) [41, 67]. However, the main goal of this study is to shed light on the capabilities of LLMs to understand the diversification re-ranking task and produce meaningful recommendations rather than finding the prompting strategy that performs best. The exploration of further more advanced LLM-based re-ranking strategies is left as future work. Finally, in our prompt-based formulation, we do not explicitly model 푓표푏푗, nor the푟푒푙or푑푖푣functions, in the prompts. We instead ‘implement’ 푓표푏푗by including key terms such as “relevance” and “diversity” (or “genre-based diversity”) for the 푟푒푙and 푑푖푣functions, respectively, and terms such as “balance” and “maximize” to model the 휆trade-off. However, we posit that our prompt-based implementation is a raw form of greedy re-ranking. This is likely because LLMs such as ChatGPT and Llama2 are autoregressive models that generate text (i.e. the final list 푅퐿) in an iterative way, i.e. item by item until 푅퐿is composed of 푛elements.
# 4 EXPERIMENTS
In this section, we outline the experimental methodology and the datasets we used in our experiments. We have opensourced the code for reproducibility.3
# 4.1 Datasets and preprocessing
In our experiments, we use the Anime Recommendation Database4 and the Goodreads Book Graph Datasets5. The former contains about 17 thousand anime movies, and ratings from about 325 thousand users. The latter contains about 2 million books, and ratings from about 876 thousand users. Both datasets include item genres that we use as item features. We apply the following preprocessing procedure to prepare the datasets for the experiments. We remove items (and their ratings) whose genres are unknown. For the Anime dataset, we also remove movies whose titles are not written in the Roman alphabet and whose release date is from 2021 onwards to increase the chance that LLMs have been exposed to data about these movies. This is not necessary for the Goodreads dataset because the books all have titles
We apply the following preprocessing procedure to prepare the datasets for the experiments. We remove items (and their ratings) whose genres are unknown. For the Anime dataset, we also remove movies whose titles are not written in the Roman alphabet and whose release date is from 2021 onwards to increase the chance that LLMs have been exposed to data about these movies. This is not necessary for the Goodreads dataset because the books all have titles
3https://github.com/cdiego89phd/chat-reranking 4https://github.com/Hernan4444/MyAnimeList-Database 5https://mengtingwan.github.io/data/goodreads.html
3https://github.com/cdiego89phd/chat-reranking 4https://github.com/Hernan4444/MyAnimeList-Database 5https://mengtingwan.github.io/data/goodreads.html Manuscript submitted to ACM
https://mengtingwan.github.io/data/goodreads.htm Manuscript submitted to ACM
Manuscript submitted to ACM
<div style="text-align: center;">Table 2. Statistics of the datasets</div>
# ratings
# users
# items
avg. # ratings
per user
avg. # ratings
per item
# genres
avg. length of
item titles
sparsity
Anime
17M
118k
2.6k
143
6.4k
40
20.6
94.50%
Goodreads
8M
166k
8k
50
1k
16
37.1
99.40%
in the Roman alphabet and were published before 2018. We then map ratings into a 1 to 5 scale and filter out users with fewer than 70 and more than 300 ratings. Table 2 reports the statistics of the datasets after the preprocessing.
# 4.2 Methodology
The following methodology applies to both datasets. We randomly split the dataset into training and test sets. We use a user-based split, with 80-20% proportions. We train a Matrix Factorization baseline recommender (which we denote with MF) on the training set, using the RankSys implementation6 of the factorization algorithm that is proposed in [48]. We then randomly sample 500 test users and their data to run the experiments with the following procedure. For each test user, we use the MF baseline to compute a ranked candidate item list 퐶퐿, such that none of the items in 퐶퐿is in the training dataset (i.e. MF recommends only unseen items). We employ InstructGPT (gpt-3.5-turbo-instruct version) and ChatGPT (gpt-3.5-turbo-0613 version) from the OpenAI GPT family7, and Meta Llama2-Chat (7B8 and 13B9 versions) to re-rank 퐶퐿as outlined in Section 3.2. We used the proprietary OpenAI API service when re-ranking with InstructGPT and ChatGPT.10 In the case of the non-proprietary Llama2-7B-Chat and Llama2-13B-Chat, we used two NVIDIA A40 48GB GPUs to host the models and perform re-ranking. As anticipated, LLMs can return partially invalid outputs. When the output is invalid in fairly simple ways, we try to remedy the situation intelligently. In cases when the LLM appends to the items’ titles additional details (e.g. lists of genres, item descriptions, release dates), despite all prompts including clear instructions to outlaw this, we remove such extra content; we also remove any extra content that is prepended or appended to the final recommendation list (e.g. repetition of the re-ranking instructions from the prompt such as “here is your re-ranked list of anime”); and we implement very simple inexact string matching methods (e.g. LLMs tend to add an extra space when encountering semicolons in item titles and our inexact matching takes care of this). But there are cases where invalid output is not so easily remedied, especially when the items themselves are invalid. Examples of invalid items include the following: the output list 푅퐿may contain items whose titles did not appear in the candidate list 퐶퐿; and there may be cases where there is a significant mismatch between the title of the item in 퐶퐿and its title in 푅퐿, e.g. “Naruto” instead of “Naruto:Shippuden”. We discard these items from 푅퐿because there is not a straightforward solution to address the invalid generation. In these cases, the 푅퐿is incomplete, |푅퐿| < 푛. We fill the list by inserting random items from 퐶퐿 until |푅퐿| = 푛. Instead, we could, but do not, repeat the generation step to try to get a valid list of items: at this stage, we prefer to quantify the problem and leave more advanced solutions for future work. Prompts T5-T8 presented in Section 3.2 need item features such as genres or item descriptions such as plot summaries. The former are already included in the datasets (see Table 2). The latter are extracted beforehand for each item
6https://github.com/RankSys/RankSys 7https://platform.openai.com/docs/models/gpt-3-5 8https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 9https://huggingface.co/meta-llama/Llama-2-13b-chat-hf 10https://openai.com/index/openai-api/. At the time we ran the experiments, the inference cost for InstructGPT was 1.5$/1M for input tokens and 2$/1M for output tokens; for ChatGPT, the inference cost was 0.5$/1M for input tokens and 1.5$/1M for output tokens.
by prompting ChatGPT with the following prompt: Please provide a one-sentence description of the following item: {item name}. To compare our LLM-based re-rankers, we employ the MMR, xQuAD and RxQuAD greedy re-rankers presented in Section 2.2. Their re-ranking strategy is explicit, they are fast and effective at diversifying recommendations, and they can use the item genres already available in the datasets as features to re-rank the 퐶퐿of each user. Note that, unlike the LLM-based re-rankers and the MRR re-ranker, xQuAD and RxQuAD additionally make use of the training data (to calculate aspect probabilities) to produce the re-ranking. In Section 5, we show that this might explain why xQuAD and RxQuAD perform better than the other re-rankers on some metrics. We use the implementations of MRR, xQuAD and RxQuAD that are provided within the RankSys library, mentioned earlier. As a baseline, we also use a Random re-ranker from the same library that randomly samples items from 퐶퐿to generate 푅퐿. We point out that, although we could use other more complex (and expensive) re-rankers for comparison (e.g. [68]), our focus is on exploring the applicability and the relative performance of LLM-based re-ranking, for which the MRR, xQuAD and RxQuAD re-rankers suffice.
# 4.3 Hyperparameter tuning
We use a validation set, a random sample of 20% of the ratings from the training set, to tune the following hyperparameters for each dataset. We set the number푘of factors of the baseline recommender MF, choosing푘from {20, 50, 100, 150} and optimizing for NDCG at cutoff 10. We found 푘= 20 for the Anime dataset and 푘= 50 for the Goodreads dataset. As is common, we set 푛, the size of the final set of recommendations to 10. Finally, to set 푚, first we find the mean 휇 and standard deviation 휎of the greatest rank in 퐶퐿of an item inserted in 푅퐿for each re-ranker, i.e. max ∀푖∈푅퐿{푟푎푛푘퐶퐿(푖)}; second, we take the largest value among these and set 푚= 휇+ 휎. We found 푚= 40 for the Anime dataset and 푚= 50 for the Goodreads dataset. This means that the values we are using for 푚are bigger than those we find in the closest works to ours in the literature, e.g. in [27, 55], 푚= 20; in [12], 푚= 5. This will make LLM-based re-ranking in our scenario more challenging because we will have greater prompt lengths and higher inference costs. But, at the same time, our scenario is more realistic because in real-world applications 푚is unlikely to be as small as 5 or 20.
# 4.4 Evaluation metrics
To assess the relevance of recommendations, we measured Precision, Recall and NDCG at cutoff 10. However, in Section 5, we report only NDCG because the three metrics reveal consistent findings. To measure recommendation diversity, we employ 훼-NDCG, Intra-List Diversity (ILD), Expected Intra-List Diversity (EILD), Subtopic-Recall (SRecall) and relevancy-aware SRecall (rSRecall), extensively discussed in [59]. ILD calculates the average pairwise distance of the items in 푅퐿; to measure the distance, we use the complement of the Jaccard similarity between pairs of items calculated on the item genres. SRecall calculates the proportion of genres covered by the items in 푅퐿. EILD and rSRecall calculate ILD and SRecall on the sublist of relevant items in 푅퐿, respectively. 훼-NDCG is based on NDCG, but it is designed to account for relevance but also coverage of aspects (i.e. genres) and their redundancy; we set 훼= 0.5 following the argument in [59].
# 5 RESULTS
For ease of exposition, we divide this section into three subsections. In the first, we focus our analysis on comparing LLM-based re-rankers with a random re-ranker (we want to know whether LLM re-ranking is more than arbitrary) Manuscript submitted to ACM
Table 3. Re-ranking performance for the Anime dataset. We report the scores achieved by the baseline MF on the six metrics. Then, for each re-ranker, we report the performances in terms of percentage difference with respect to the scores of the baseline MF. We report the average performance across the eight templates for each LLM-based re-ranker. All metrics are computed with a cutoff of size 푛= |푅퐿| = 10. For all metrics, the higher the scores, the beter the performance, and we highlight in bold the best performance for each metric.
NDCG
훼-NDCG
EILD
ILD
rSRecall
SRecall
MF
0.316
0.325
0.221
0.776
0.234
0.473
Re-ranker
Random
-58.8
-52.6
-57.3
4.5
-38.0
5.6
Difference
(%)
MMR
-5.4
-3.4
0.4
12.6
1.5
15.0
xQuad
-4.6
7.9
-4.2
0.5
7.5
14.0
RxQuad
-6.5
4.1
-6.4
-0.8
4.3
7.0
ChatGPT
-10.1
-6.0
-9.4
4.7
-2.5
7.8
InstructGPT
-14.9
-10.2
-14.7
4.7
-6.1
7.2
Llama2-7B-Chat
-21.5
-15.5
-23.3
4.6
-10.8
8.0
Llama2-13B-Chat
-26.0
-21.6
-25.1
4.4
-14.7
4.7
and traditional re-rankers (we want to know how LLM re-ranking is positioned with respect to simple state-of-the-art approaches). In the second subsection, we focus our analysis on comparing different prompt templates for LLM-based re-rankers: we want to know how different prompts affect performance. Lastly, in the third subsection we analyse the costs incurred by the different re-rankers. Tables 11 and 12 in Appendix B report the full set of results presented in this section. In these tables we also report confidence intervals calculated with a 95% confidence level and we use them through our analysis to help outline the uncertainty of our results. Non-overlapping intervals imply statistical significance between performances while, despite common belief, overlapping intervals do not necessarily imply lack of statistical significance [22]. In fact, the confidence intervals in our results are quite large for all metrics except for ILD and SRecall — due to high variability in the values of the metrics but also the small number of users in our experiments, i.e. 500. This limits the number of cases where we can be sure of statistical significance (non-overlapping intervals). In future work, we hope to extend our experiments with a larger group of users and that will allow us to perform a deeper statistical analysis and strengthen our findings. This should be possible assuming that the costs (both monetary and computational) of using LLMs continues to fall.
# 5.1 Comparison of LLM-based re-rankers with random and traditional re-rankers
Tables 3 and 4 report the results of the re-rankers on the Anime and Goodreads datasets, respectively. For each reranker, we report the percentage difference with respect to the baseline recommendations from MF. For the LLM-based re-rankers, we report the average performance across the eight templates. We postpone the fine-grained analysis across different prompt templates to the next subsection. As expected, re-ranking decreases relevance and increases the diversity of recommendations for both datasets. In particular, for the Random re-ranker, relevance, measured by NDCG, drops significantly, between 58.8% and 66.3%. Relevance also plays a major role in the훼-NDCG, EILD and rSRecall metrics, so we see large decreases in their values for the Random re-ranker. The diversity of its outputs (ILD and SRecall) increases, as typically happens when introducing randomness into recommendations, but only slightly for Goodreads.
Table 4. Re-ranking performance for the Goodreads dataset. We report the scores achieved by the baseline MF on the six metrics. Then, for each re-ranker, we report the performances in terms of percentage difference with respect to the scores of the baseline MF. We report the average performance across the eight templates for each LLM-based re-ranker. All metrics are computed with a cutoff of size 푛= |푅퐿| = 10. For all metrics, the higher the scores, the beter the performance, and we highlight in bold the best performance for each metric.
NDCG
훼-NDCG
EILD
ILD
rSRecall
SRecall
MF
0.249
0.271
0.092
0.689
0.355
0.871
Re-ranker
Random
-66.3
-59.6
-71.3
2.0
-43.8
2.1
Difference
(%)
MMR
-4.7
-6.6
-2.5
17.7
-5.3
7.2
xQuad
-7.8
13.7
-9.3
-8.2
8.8
10.0
RxQuad
-8.1
11.9
-8.6
-7.7
7.4
8.6
ChatGPT
-33.4
-19.3
-46.0
-4.8
-20.0
1.8
InstructGPT
-53.5
-36.6
-66.5
-4.7
-35.0
1.4
Llama2-7B
-59.5
-50.4
-70.0
-1.1
-43.7
1.1
Llama2-13B
-61.3
-48.4
-71.9
-2.2
-41.8
1.5
In both datasets, all the intelligent re-rankers (both the traditional ones and the LLM-based ones) outperform the Random re-ranker for all the metrics to a great extent (and their performances are statistically significantly different), except for ILD and SRecall (where the superiority is not consistent across all intelligent re-rankers). This confirms what we were looking to prove with RQ2: that LLMs can interpret diversification re-ranking and produce rankings that are more than arbitrary. Regarding traditional re-ranking, MMR and xQuAD seem to have a better trade-off than RxQuAD, but this strongly depends on which metrics one considers to be more important. RxQuAD has the highest drop in relevance as measured by NDCG in both datasets and, while xQuAD is superior in this metric to MMR for the Anime dataset, the opposite stands for the Goodreads dataset. As expected, MMR is the best for ILD on both datasets since it optimizes this metric. While MMR also improves SRecall on both datasets (even having the best score on the Anime dataset), it fails to improve 훼-NDCG, and it slightly improves the other relevance-aware diversity metrics (i.e. EILD and rSRecall) on the Anime dataset but not in the Goodreads dataset. On both datasets, xQuAD is the best re-ranker at improving 훼-NDCG as this metric is very similar to what it optimises. Similarly, also RxQuAD improves on this metric as expected, but to a lesser extent than xQuAD. xQuAD improves the other diversity metrics too, except for EILD and ILD on both datasets (and it is the best for rSRecall and SRecall on both datasets and for SRecall on Goodreads). Finally, RxQuAD improves 훼-NDCG, rSRecall and SRecall (as expected, being similar to xQuAD), but in none of these metrics does its scores outperform all the other traditional rerankers. The lower performance of RxQuAD in terms of relevance-aware metrics is quite surprising since this approach is designed to focus on relevance more than the other two re-rankers. Other works have found similar outcomes [33]: we posit that this is because the datasets exhibit a weak correlation between relevance and item features, in contrast with the assumption made by the re-ranking objective function RxQuAD is built upon. On the Anime dataset, the decrease in NDCG for the LLM-based re-rankers (roughly 10-26% loss) is greater than that for the traditional re-rankers (4.6-6.5%), although at its best (i.e. for the template on which it performs best) ChatGPT outperforms all the other re-rankers (losing just 2.8%); see the full results in the Appendix. There are similar results for the other metrics, particularly those where recommendation relevance plays a major role (i.e. 훼-NDCG, EILD and rSRecall). In particular, traditional re-rankers are statistically significantly better than LLM-based re-rankers (except Manuscript submitted to ACM
Table 5. Average lowest rank of items in the candidate list퐶퐿recommended in the final list 푅퐿. High values correspond to low ranks To calculate the lowest rank, we do not consider the random recommendations in 푅퐿provided to the user when invalid generations occur.
Re-ranker
Anime
Goodreads
MMR
19.5
22.58
xQuAD
21
17.9
RxQuAD
16
15.9
ChatGPT
22.6
31.9
InstructGPT
22.3
37.7
Llama2-7B-Chat
25.5
39.5
Llama2-13B-Chat
28.9
43.1
<div style="text-align: center;">Table 6. Average percentage of random recommendations (due to invalid generation) provided to the 500 users, distributed across the datasets, the LLM-based re-rankers and the prompt templates. In bold we report the lowest value in each group.</div>
Dataset
Re-ranker
Prompt template
Anime
Goodreads
ChatGPT
InstructGPT
Llama2-7B-Chat
T1
T2
T3
T4
T5
T6
T7
T8
2.3%
13.1%
3.0%
1.2%
18.9%
4.0%
3.1%
8.8%
4.8%
9.3%
8.9%
18.0%
22.4%
ChatGPT) for NDCG, 훼-NDCG and EILD. The metric where LLM-based re-ranking does best in comparison with traditional re-ranking is ILD (and, where better, statistical significance is achieved). For this metric, the OpenAI rerankers perform better than xQuAD and RxQuAD. Finally, there is a clear difference between the LLM-based re-rankers themselves: ChatGPT is superior to all others according to all metrics (except for SRecall where Llama2-7B-Chat is the best). Also, OpenAI models are better than Meta ones, with Llama2-13B-Chat being the worst of all the re-rankers. On the Goodreads dataset, there is a similar pattern of results for the LLM-based re-ranking, but this time the performance gap with traditional re-rankers is higher across all metrics (in particular for NDCG and 훼-NDCG). ChatGPT is not as close to some traditional re-rankers as it was on the Anime dataset (and differences are statistically significant). It is still the case that the LLM-based re-rankers perform better than xQuAD and RxQuAD for ILD (with a statistically significant difference), and, among themselves, their performance from best to worst remains ChatGPT, then InstructGPT, then Llama2-7B and lastly Llama2-13B. Three factors might explain why LLM-based re-rankers are inferior to traditional re-rankers in these experiments, particularly for relevance-aware metrics. First, we argue that relevance in 푅퐿should fall when drawing low-rank items from 퐶퐿, assuming that items in 퐶퐿are accurately ranked by their relevance. Indeed, Table 5 shows that LLM-based re-rankers draw items ranked lower in 퐶퐿than the ones drawn by traditional re-rankers. Second, the proportion of random recommendations in the final 푅퐿lists of LLM-based re-rankers (Table 6) might harm relevance, in particular for the Goodreads dataset. (As we explained in Section 4.2, when the output of the LLM contains invalid items, we replace the invalid items with randomly chosen items from 퐶퐿, and this is likely to harm the relevance of the final 푅퐿.) Third, traditional re-rankers such as xQuAD and RxQuAD might have an advantage over LLM-based re-ranking as the former explicitly use user profiles to estimate relevance and genre distributions within profiles, while the latter do not have access to such information. Indeed, the LLMs might struggle to capture relevance (in terms of rank) and diversity (in terms of genres) and the interplay between the two from the prompt templates we have designed, even though the prompts explicitly highlight item relevance (by including each item’s rank) and some of them (T5–T8) highlight Manuscript submitted to ACM
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b753/b7535438-4a5e-4ee2-b124-1512f6b70880.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 1. Comparison of the LLM-based re-rankers across different prompt templates for the Anime dataset. For each metric, we plo the percentage difference with respect to the scores of the baseline MF.</div>
diversity (by providing an item’s genres or a description of the item) — see the detailed analysis of the prompt template results in the next subsection.
# 5.2 Comparison LLM-based re-rankers
We use Figures 1 & 2 and Table 7 to draw a comparison across different prompt templates and LLM-based re-rankers for both the Anime and Goodreads datasets. In Figures 1 and 2 for Anime and Goodreads dataset respectively, each plot corresponds to a different metric. In each plot, on the horizontal axis, there are groups of results for each of the eight templates (T1–T8) and, within each group, there are three re-rankers, i.e. ChatGPT, InstructGPT, Llama2-7B-Chat. On the vertical axis, values represent the percentage difference between the re-ranker and the MF baseline. Table 7 reports the performances of the LLM-based re-rankers grouped by prompt template. For ease of exposition, we do not include results for Llama2-13B-Chat (although they, along with all results are in the Appendix). Its performances is generally worse than all the other re-rankers, hence its results do not add valuable insights to the discussion. It is surprising that Llama2-13B-Chat is worse than Llama2-7B-Chat, since LLM capabilities usually correlate with LLM size when considering models of the same family. However, differences in human preference alignment may play a major confounding role [46], and there are indications in the literature that Llama213B-Chat is worse than Llama2-13B-Chat for some tasks, e.g. see human evaluation in [57]. But, beyond these general points, we have no clear explanation for why Llama2-13B-Chat performs so poorly in our experiments and we leave its investigation for future work. Manuscript submitted to ACM
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b24a/b24a6190-ffca-4966-82c2-32ff93844947.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 2. Comparison of the LLM-based re-rankers across different prompt templates for the Goodreads dataset. For each metric, w plot the percentage difference with respect to the scores of the baseline MF.</div>
Overall, we can draw three main findings. First, LLM-based re-ranking improves the baseline ranking of MF for ILD in the Anime dataset and SRecall in both datasets (and all the LLM-based re-ranking performances except for the ones of ChatGPT are statistically significantly different from the MF ones). For the other metrics, performance degradation is more severe on the Goodreads than on the Anime dataset. The fact that the Anime dataset is less prone to invalid generation than the Goodreads dataset (2.3% vs 13.1%, Table 6) might partially explain the difference in degradation. Also, an LLM’s internal knowledge about the specific recommendation domain is another key factor affecting re-ranking. We speculate that generating content related to books is more challenging than anime, probably due to the nature of the domain and its content (e.g. the presence of similar titles in a prompt or the length of the prompts, which are longer for books than for anime on average, see Table 2). However, our experiments cannot confirm our speculation or that this even applies to other domains. We plan to address this in future work. Second, overall, ChatGPT exhibits the best performance among LLMs (within each specific template and averaged across all templates) for all metrics, with the only exception being SRecall and ILD in both datasets, where the model is not the best in all the different templates. We posit that the extent to which an LLM re-ranker draws low-rank items and includes random recommendations from 퐶퐿in the final recommendation list 푅퐿correlates with performance differences across datasets and templates, which partially explain these results. Indeed, Tables 5 and 6 show that OpenAI’s models (on average) draw higher ranked items from퐶퐿and need fewer random recommendations than Meta’s models. This is consistent across the Anime and Goodreads datasets, but to a greater extent for the former. The only exception is between ChatGPT and InstructGPT, where the former has more random recommendations than the latter. Thus, Manuscript submitted to ACM
Anime
Goodreads
NDCG
훼-NDCG
EILD
ILD
rSRecall
SRecall
NDCG
훼-NDCG
EILD
ILD
rSRecall
SRecall
T1
-15.9
-11.0
-16.9
4.7
-6.6
8.2
-47.6
-33.4
-62.6
-3.8
-33.1
1.1
T2
-14.5
-10.7
-15.6
5.0
-7.3
6.8
-45.7
-32.4
-56.6
-2.7
-31.3
1.4
T3
-16.7
-11.4
-18.5
5.4
-8.0
8.6
-52.0
-39.1
-64.0
-3.6
-37.1
1.2
T4
-18.3
-12.1
-19.1
4.8
-7.2
9.0
-49.6
-35.6
-66.6
-3.8
-34.2
1.7
T5
-16.2
-10.3
-15.9
4.5
-6.1
7.8
-43.5
-30.1
-53.6
-3.8
-27.7
1.6
T6
-15.7
-10.1
-15.5
4.6
-6.0
8.3
-45.3
-31.1
-57.8
-4.3
-29.1
2.0
T7
-13.9
-9.3
-13.5
3.8
-5.0
6.4
-52.8
-40.0
-63.7
-3.6
-34.9
0.7
T8
-12.7
-9.5
-11.5
4.5
-5.6
6.3
-54.1
-41.6
-61.8
-2.6
-35.8
1.7
we argue that the proportion of random recommendations only partly explains the loss of relevance in LLM-based rerankers. We believe that a model’s reasoning capability also plays a major role in performance differences. According to several benchmarks11 and the literature [9, 32], OpenAI models are superior to Meta’s models, and ChatGPT shows superior capabilities over InstructGPT (although these benchmarks and the literature are not specific to recommender systems). Our results on diversification re-ranking seem to confirm what the benchmarks and literature have shown. Third, different prompt templates show different performance across re-rankers, datasets and metrics. This suggests that there is no best template. Prompt template design needs to be tailored to specific LLMs, recommendation domains and metrics. In the next two paragraphs, we will highlight the most significant of our findings about the different templates. Templates T1 and T4–T7 are designed to strike a balance between relevance and diversity (similar to setting 휆= 0.5) and should exhibit the best performance for relevancy-aware metrics, while T2, T3 and T8 should increase diversity the most (휆= 0). This is reflected in their average re-ranking performances, but not consistently (see Table 7). On the Anime dataset, for example, T8 and T7 are the templates where relevance-aware metrics degrade the least, even though T8 is not explicitly designed to account for relevance. Also, T2, T3 and T4 improve ILD and rSRecall the most, even though T4 should focus more on balancing relevance (but instead degrades relevance-aware metrics more than any other templates). The results of the Goodreads dataset are more consistent with what we expect from the template design. T5 and T6 show the least amount of degradation on the relevance-aware metrics as expected;T2, T8 benefit ILD the most; and T4, T6 & T8 benefit SRecall the most. On average, feature-aware templates (i.e. T5–T8) benefit the re-ranking performances more than templates that do not include item features (i.e. T5–T8) on both datasets, even though the feature-aware templates result in more invalid and hence random recommendations (which may affect performance negatively). Notably, T8 has the best performance on relevance-aware metrics for the Anime dataset and on diversity metrics for the Goodreads dataset despite generating the largest number of invalid recommendations (22.8%). Overall, these findings are a positive indication that LLMs can effectively use additional information to improve re-ranking results.
# 5.3 Re-ranking cost analysis
Finally, we analyse the cost of performing re-ranking. Table 8 reports the average inference time to re-rank the candidate recommendations of a user. For re-ranking using OpenAI’s LLMs, the table shows the same value (i.e. 25 seconds).
11https://artificialanalysis.ai/models Manuscript submitted to ACM
11https://artificialanalysis.ai/models Manuscript submitted to ACM
Re-rankers
MMR
xQuAD
RxQuAD
ChatGPT/InstructGPT
Llama2-7B-Chat
Llama2-13B-Chat
Anime
0.05
0.1
0.122
<25
12.6
23.7
Goodreads
0.08
0.13
0.18
<25
16.1
31.6
0.065
0.115
0.151
<25
14.35
27.6
This value is not the actual cost. It is an upper bound which we have to use in this table due to problems we encountered using OpenAI APIs. In a preliminary experiment to test the APIs, we noticed that the model’s response time had a very large variability and sometimes the model did not return any outputs at all (i.e. the service seemed not to be available), probably due to unknown usage policies and high workloads. We found that introducing a sleep time of 25 seconds between API calls was a way of ensuring we obtaining outputs. Unfortunately, this prevents us from providing an accurate estimation of response time. Nevertheless, the table shows that traditional re-rankers have a clear advantage over LLM-based re-rankers here: they are three orders of magnitude faster than the LLM-based re-rankers. In the case of LLM-based re-rankers, it also shows, as expected, that larger models lead to longer re-ranking time: Llama2-7B-Chat is faster than Llama2-13B-Chat. Table 9 breaks down the financial costs in terms of API calls for the OpenAI models12. There are no API costs for re-ranking using Meta’s LLMs since we hosted these models on our own GPUs. In principle, we could provide a cost comparison between the OpenAI and Meta model re-rankers if we were to calculate their energy costs. In practice, estimates of energy costs are hard to give due, e.g., to the volatility of energy market prices over the period during which we ran the experiments. The total cost of re-ranking for 500 users in our experiment is roughly 161$, and around 75% of this expense is incurred by InstructGPT, whose API calls are more expensive. Prices are correlated with the number of tokens processed: the Anime dataset experiment is cheaper than the one with Goodreads as the latter’s item titles and features are more wordy; feature-aware prompt templates (i.e. T5-T8) are the most expensive ones (in particular T7 and T8, because item descriptions in the form of plot summaries generally have more tokens than genre lists). Therefore, recalling the results of the earlier sections, we find that ChatGPT is superior to InstructGPT in terms of accuracy/diversity and cost. In summary, traditional re-ranking is much faster and less resource-demanding than LLM-based re-ranking. Among LLM-based re-ranking, ChatGPT is cheaper than InstructGPT, and use of these models comes with a lack of control and transparency due to the API services they rely on. Finally, Meta’s models ensure more control over data and inference but need private computational resources, with difficult-to-estimate costs that are subject to market volatility. However, some recent reports13 report that training and inference are rapidly becoming cheaper (decreasing annually by 75% and 86% respectively), which will probably break down the barrier for LLM adoption and reduce the cost advantage of traditional re-ranking.
# 6 BEYOND PERFORMANCE: LLM-BASED AND GREEDY RE-RANKING TRADE-OFFS
In the previous section we have analysed and discussed accuracy/diversity performance and costs of LLM-based re rankers and compared them with the ones of traditional greedy re-ranking. In this section we focus on RQ4: we discuss
12The prices relate to the timeframe of our experiments, i.e. from October 2023 to January 2024.Prcies and pricing policies may have changed. 13https://www.ark-invest.com/big-ideas-2024
Manuscript submitted to ACM
Dataset
Template id
# Input tokens
# Output tokens
ChatGPT
InstructGPT
ChatGPT
InstructGPT
Anime
T1-T4
∼1.8M
∼118k
∼121k
T5, T6
∼4.2M
∼159k
∼210k
T7, T8
∼8.5M
∼176k
∼245k
Goodreads
T1-T4
∼3.3M
∼212k
∼220k
T5, T6
∼6.9M
∼221k
∼285k
T7, T8
∼9.4M
∼261k
∼269k
Total # of tokens
∼78.4M
∼1.1M
∼1.3M
Unitary price
0.5$/1M
1.5$/1M
1.5$/1M
2$/1M
Total cost ChatGPT
78.4M · 0.5 + 1.1M · 1.5$ = 40.85$
Total cost InstructGPT
78.4M · 1.5$ + 1.3M · 2$ = 120.2$
Total cost
40.85$ + 120.2$ = 161.05$
the trade-offs RS providers should take into account when selecting one of these two re-ranking approaches. Table 10 summarizes our analysis. Many of the traditional approaches need explicit item features to use in the re-ranking. Acquiring these features might require human labelling or other expensive information retrieval procedures. The LLM-based approaches do not need to be given explicit item features to perform the re-ranking. They can instead take advantage of implicit knowledge about items that is embedded in their parameters during pre-training or human alignment. This implicit knowledge might be more complex than the simple feature sets (e.g. genres) typically used in traditional re-ranking, and this might enhance the diversity of the final recommendations. Moreover, LLMs might even leverage cross-domain relationships; for example, when recommending movies, the LLM may implicitly leverage relationships with related books and music. On the other hand, an LLM’s internal knowledge is often unknown by the users who prompt the LLM (especially for proprietary models), and it might be hard to drive the model to use its implicit knowledge when re-ranking. With prompt templates T5-T8, we attempted to reduce this gap by providing the LLM with explicit item features to use during re-ranking, similarly to what is done in RAG with LLMs [20] and in the same way we do with traditional re-rankers. The results presented in the previous section show that explicit item features can sometimes benefit re-ranking. Greedy approaches can re-rank candidate lists composed of a high number of items (limited only by the amount of memory available), while the candidate list size when re-ranking LLM-based is limited by the size of the input context that can be fed into an LLM. Although some LLMs have long context size windows of millions of tokens (e.g. Google Gemini 1.514) and some tricks are available to increase the size of such windows (e.g. [17]), the OpenAI and Meta LLMs we used all have a limited input size of roughly 4k tokens. Most of the traditional re-ranking approaches that we presented in Section 2.2 do not require expensive computations (they can run on CPUs) and do not have high memory requirements. Moreover, simple techniques might drastically reduce the re-ranking time (for example, item distances might be pre-computed and cached). By contrast, resource requirements are one drawback of LLMs: see the insights from the previous section. Hosting and running efficiently an LLM-based service (in our scenario, an LLM-based re-ranker) needs resources such as GPUs and several gigabytes of memory. These are expensive and might require skilled technical expertise. An LLMs’ high inference time might also
14https://deepmind.google/technologies/gemini/ Manuscript submitted to ACM
Re-rankers
Traditional
LLM-based
Require explicit item features
Yes
No (but can use them if available)
Can re-rank large candidate lists
Yes
Depends on the size of the context window
Require expensive computations
No
Yes (but can be reduced with optimizations)
Have a heavy memory footprint
No
Yes (but can be reduced with optimizations)
Control over data, computations, usage
Full
Limited (depending on the model )
Easily address the cold-start problem
Yes
No
Always produce valid rankings
Yes
No
Easily identify items
Yes
No (depending on the domain)
be a bottleneck in settings where recommendations must be generated on the fly (e.g. session-based recommenders). There is a growing amount of recent work on both reducing LLM inference time (e.g. using batch inference, flash attention, and weight sharing) and reducing their memory footprint (e.g. using quantization and model distillation) (see [69] for a broad overview). However, the gap with traditional re-rankers remains substantial. When using traditional re-rankers, data and computations are typically hosted on a local infrastructure, as the cost and the technical challenges are quite low. By contrast, when running LLM-based re-ranking using a proprietary model or a proprietary platform (as we do with ChatGPT leveraging OpenAI APIs, see Section 4), control over data and processing is generally limited. For example, data needs to be uploaded to and downloaded from the data centre where the LLM is hosted; the inference time (i.e. generation time) might be higher when the endpoint is overloaded; inference availability may even be disrupted; usage limits on the service might be applied; and there is need for the external platform provider to guarantee data protection. However, when using open source alternatives such as Meta’s LLMs, these problems are mitigated because control over data and computations is retained. From a re-ranking point of view, traditional approaches address the well-known item cold-start problem fairly smoothly: they require only that each new item be associated with its features. By contrast, LLM-based approaches might struggle with this problem. Items that are new to the system (e.g. newly released movies) and information related to them (such as item features or general descriptions) will be unknown to an LLM that was trained or tuned prior to release of the new item. Updating its parameters might be impractical even if fresh item information is available. Re-training will likely be periodic, possibly introducing a substantial delay before the LLM becomes aware of the new items. One solution to this problem might be the RAG approach [20], which could be used to gather fresh contextual information related to items to enrich the re-ranking prompts. However, implementing a RAG system could be complex and expensive and would also increase the re-ranking time due to the inference time of the retrieval components. Traditional re-rankers always produce valid rankings. By contrast, LLMs may generate text that is inconsistent, hallucinated, harmful, or irrelevant to the instructions provided in the prompt. The results of our experiments when using LLM-based re-rankers also report invalid (malformed) outputs, e.g. recommendations that have titles of items that do not exist, or titles that do not appear in퐶퐿(see Section 5). Countermeasures have been proposed in the literature to mitigate such problems [30, 74], but it appears there is no one-size-fits-all solution. Advanced parsing of an LLM’s output can also be implemented (and we indeed partly address invalid generations with this technique in Section 4.2), but these solutions do not necessarily scale well to a wide range of invalid generation corner cases.
Manuscript submitted to ACM
Traditional re-rankers typically identify items by identifiers such as product codes. This can work across different recommendation domains. In LLM-based re-ranking, the prompts we use rely on working in domains where items have useful titles to identify them. This allows the LLM to draw on its knowledge of these items. Domains such as books, movies, and music use titles in this way but this might be challenging or impractical in other domains. For example, consider domains where people are recommended to other people, such as dating or social media. An LLM may not know who the individuals are by their names. If they do know, it will be because the training data includes data about these individuals (and whether this is so is often unknown upfront, as training data is often undisclosed for state-of-the-art models). Non-uniqueness may also be a problem (the extent of the problem perhaps being greater in some domains, such as hotels, than others, such as movies). Names and titles may need to be extended (e.g. hotels with their cities) to reduce non-uniqueness. Finally, there are domains where there may not be titles at all, e.g. short posts to social media streams. When titles/names are unavailable, LLMs might still use items features or descriptions to identify items and re-rank them with proper prompts modification. In our future work we plan to explore re-ranking in this challenging scenario.
# 7 CONCLUSION AND FUTURE WORK
This paper investigates the use of general-purpose conversational LLMs to enhance top-푛recommendation diversity through re-ranking. In this paper, we proposed to prompt LLMs in a zero-shot fashion to re-rank a large set of candidate items and provide a final recommendation set. We wanted to discover whether LLMs can interpret diversification re-ranking. We also explored the impact of different prompt templates we designed: some of them aim to strike a balance between relevance and diversity, and some focus on improving diversity only. Some templates include item features as contextual information to help re-ranking, and others do not. We run experiments on two publicly available datasets (from the anime movies and books domains) where we compare LLM-based re-ranking such as ChatGPT and InstructGPT from OpenAI and Llama2-7B-Chat and Llama2-13B-Chat from Meta with traditional greedy diversification re-rankers and a random re-ranker. We analyzed their cost-performance trade-offs and provided a comprehensive discussion of the pros and cons beyond accuracy/diversity performances and computational costs to help inform the choice between LLM-based and traditional greedy approaches to perform diversification re-ranking. First of all, our results show that the LLM-based re-rankers that we have proposed can interpret the re-ranking task. For both datasets, all re-rankers sacrifice relevance to improve diversity, but to a different extent. LLM-based rankings have better relevance/diversity trade-offs than the ones of random re-ranking but worse than the ones of the traditional re-rankers. We argue that LLM-based re-rankers are inferior to traditional re-rankers, particularly for relevance-aware metrics, because: they draw lower-ranked items from the candidate list; they are penalised by the presence of random recommendations in the final list due to invalid generation; and they sometimes struggle to effectively capture relevance and item diversity, whereas traditional re-rankers directly leverage user profiles and item features to drive re-ranking. Re-ranking from OpenAI models are better than the ones from Meta’s models: ChatGPT is, in general, the best performing LLM re-ranker, and Llama2-13B-Chat is the worst (although comparable to the others on ILD and SRecall). Our findings are in line with the majority of the literature that credits OpenAI models with more advanced capabilities than Llama models. Also, different prompt templates have different performances across LLMs and