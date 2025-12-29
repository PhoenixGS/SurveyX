# Generating Query Recommendations via LLMs
Andrea Bacciu∗ Sapienza University Rome, Italy bacciu@diag.uniroma1.it Enrico Palumbo Spotify Turin, Italy Andreas Damianou Spotify Cambridge, United Kingdom andreasd@spotify.com
Andrea Bacciu∗ Sapienza University Rome, Italy bacciu@diag.uniroma1.it Enrico Palumbo Spotify Turin, Italy An Camb an en
Andrea Bacciu∗ Sapienza University Rome, Italy bacciu@diag.uniroma1.it
Nicola Tonellotto Pisa University Pisa, Italy nicola.tonellotto@unipi.it Fabrizio Silvestri Sapienza University , CNR Rome, Italy fsilvestri@diag.uniroma1.it
o pi.it Fabrizio Silvestri Sapienza University , CNR Rome, Italy fsilvestri@diag.uniroma1.it
ABSTRACT
# ABSTRACT
Query recommendation systems are ubiquitous in modern search engines, assisting users in producing effective queries to meet their information needs. However, many of these systems require a large amount of data to produce good recommendations, such as a large collection of documents to index and query logs. In particular, query logs and user data are not available in cold start scenarios. Query logs are expensive to collect and maintain and require complex and time-consuming cascading pipelines for creating, combining, and ranking recommendations. To address these issues, we frame the query recommendation problem as a generative task, proposing a novel approach called Generative Query Recommendation (GQR). GQR uses a Large Language Model (LLM) as its foundation and does not require to be trained or fine-tuned to tackle the query recommendation problem. We design a prompt that enables the LLM to understand the specific recommendation task, even using a single example. We then improved our system by proposing a version that exploits query logs called Retriever-Augmented GQR (RA-GQR). RA-GQR can dynamically compose its prompt by retrieving similar queries from query logs. GQR approaches reuses a pre-existing neural architecture resulting in a simpler and more ready-to-market approach, even in a cold start scenario. Our proposed GQR obtains state-of-the-art performance in terms of NDCG@10 and clarity score against two commercial search engines and the previous state-of-the-art approach on the Robust04 and ClueWeb09B collections, improving on average the NDCG@10 performance up to ∼4% on Robust04 and ClueWeb09B w.r.t the previous state-of-the-art competitor. While the RA-GQR further improve the NDCG@10 obtaining an increase of ∼11%, ∼6% on Robust04 and ClueWeb09B w.r.t the previous state-of-the-art competitor. Furthermore, our system obtained ∼59% of user preferences
arXiv:2405.19749v
∗Work done before joining Amazon
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. IR-RAG @ SIGIR ’24, July 14–18, 2024, Washington D.C., USA © 2024 Association for Computing Machinery. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/XXXXXXX.XXXXXXX
Andreas Damianou Spotify Cambridge, United Kingdom andreasd@spotify.com enricop@spotify.com
Fabrizio Silvestri Sapienza University , CNR Rome, Italy fsilvestri@diag.uniroma1.it
in a blind user study, proving that our method produces the most engaging queries.
Web search, Large Language Model, Query recommendation formation Retrieval
ACM Reference Format: Andrea Bacciu, Enrico Palumbo, Andreas Damianou, Nicola Tonellotto, and Fabrizio Silvestri. 2024. Generating Query Recommendations via LLMs. In IR-RAG @ SIGIR ’24: The 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, July 14–18, 2024, Washington D.C., USA. ACM, New York, NY, USA, 11 pages. https://doi.org/ XXXXXXX.XXXXXXX
# 1 INTRODUCTION
“Related Searches” is a fundamental module of a Search Engines Result Page (SERP). By recommending new queries related to the query submitted by a user, the related searches module guides the users toward their information needs by suggesting more focused and refined queries, called query recommendations. Typically, a “Related Searches” module is implemented by a query recommendation system. To exemplify why related searches are important, consider a user trying to get information about the company Corporate Ltd. the user enters the query “Corporate” and, by assuming this is a navigational query, the search engine’s first result is the Corporate Ltd’s website. This is not what the user was looking for, though, and the Related Searches component offers recommendations for queries such as “Corporate products”, “Corporate job openings”, and so on. Without needing to reformulate the query manually, the user is only “one tap away” from the SERP containing the information it was looking for. However, from a practical point of view, building or improving a query recommendation system is challenging [33] as it involves three distinct requirements: past user-system interactions stored in query logs, a machine-learned model trained on them, and a serving infrastructure to support the real-time recommendation in response to user queries. In contrast to most previous query recommendation solutions, here we show that it is possible to develop an effective “Related Searches” module without the need for a system-specific machinelearned model trained on it. We present a novel way of conceiving a query recommendation system by using the great expressiveness and language generation power of Large Language Models (LLMs),
such as OpenAI’s GPT3 [7]. An earlier attempt to build a query recommendation system without query logs was by Bhatia et al. [3] in 2011. However, when that research was done, the tools available to “generate” text were limited. In a nutshell, their approach was based on finding key phrases in a document collection whose prefixes were related to, or matching, the user-submitted query. However, this purely statistical word-based approach does not leverage complex correlations found in the language (which gives rise to learned semantics), and the language generation capabilities are also limited; these issues result in poor performance compared to modern query recommendation systems. In this paper, we take a completely different approach inspired by the success of prompt-based zero-shot learning [22]. We prompt an LLM, in our case, OpenAI’s GPT3, with examples of queries and respective recommendations, and we let the LLM generate recommendations related to a given new query. An example is shown in Figure 1. The LLM is prompted with two queries, namely Ryanair and New York, together with their recommendations, and a last query, Nutella, for which the LLM generates recommendations. Essentially, in that example, we build a prompt composed of two pairs of query recommendations, followed by the user query and a “recommendations” token. In doing so, we “instruct” the LLM on what is the structure and the semantics of the continuation to the last token, and the system is able to find the relevant recommendations even if the category for the query is different from the two example queries in the prompt. This is the essence of promptbased approaches to zero-shot learning, providing an “in-context” set of examples that will drive the LLM generation phase toward solving a task. Dai et al. [11] recently postulated that prompt-based approaches are similar to meta-learning applied to a specific task. Prompts are comparable to support sets in meta-learners and act as a guide for generating the rest of the sentence, in our case, query recommendations. From a time-to-market perspective, traditional query recommendation systems based on mining query logs [19] suffer from the cold-start problem, as they need to collect and maintain large amounts of user data and query logs to estimate a meaningful correlation between co-observed queries in search sessions. This is far from ideal for newly launched search engines, but also for more established search engines that start operating in new regions or markets where the query logs may not be representative of the user behavior. Our approach can greatly speed up the bootstrapping of a query recommendation system, as we only require a pre-trained LLM and a handful of hand-curated prompts, which manual annotators and domain experts can easily create. These are the major benefits of using a prompt-based LLM approach for implementing a “Related Searches” component:
• There is no need to build a specific data structure and model for the query recommendation service. One can use an available LLM and apply the prompt we devise in this research. • Recommendations are generated without using any user information, past queries, or past interactions, except the current query. This greatly reduces the time-to-market of query recommendation systems and allows us to tackle the cold-start issue effectively.
• There is no need to build a specific data structure and model for the query recommendation service. One can use an available LLM and apply the prompt we devise in this research. • Recommendations are generated without using any user information, past queries, or past interactions, except the current query. This greatly reduces the time-to-market of query recommendation systems and allows us to tackle the cold-start issue effectively.
• Generating recommendations for previously unseen and long-tail queries is a challenging task in traditional query recommender systems [5, 34]. Using our technique, this task becomes trivial, and it does not require any specifically designed mechanisms to manage tail queries.
Query logs remain an important source of knowledge, for that reason, we propose an advanced version of GQR called Retrieval Augmented GQR (RA-GQR) able to leverage query logs to build automatically its prompt. This version first retrieves similar users’ written queries from a query log and then compose dynamically the GQR prompt to generate better recommendations. In that way the examples in the prompt are tailored within the topic of the current user query. We introduce the Generative Query Recommendation (GQR) system. GQR uses an LLM as its foundation and leverages the prompting abilities of the LLM to understand the recommendation task through a few examples provided in the prompt (retrieved or handcrafted). Our preliminary experiments on two publicly available test collections show that GQR and RA-GQR outperform the other tested systems on average and obtain stable performance along query recommendations in the various ranks. For example, our GQR, exploiting the GPT-3 LLM, outperforms other query recommendation systems obtaining an NDCG@10 improvement of at least ∼4% in Robust04 and ClueWeb09B. With the aid of 12 annotators, we conduct a blind user study that shows that GQR produces the most engaging and relevant query recommendations, with ∼59% preferences.
# 2 RELATED WORK
In this section, we discuss the related work on query recommendations (Section 2.1), LLMs (Section 2.2), and prompting in Information Retrieval (IR) (Section 2.3).
# 2.1 Query Recommendations
Query recommendations are ubiquitous in modern search engines, assisting users in formulating and re-formulating effective queries and supporting exploratory searches. The query recommendation problem is generally modelled in two alternative ways. The first approach is based on the idea that the query recommendation provider is able to guide the user toward high-quality queries by looking into which queries have statistically led to successful search sessions in a set of query logs. For instance, the notion of query shortcut [1] corresponds to creating a link between the current user query and a successful query that is highly correlated with the user query, typically appearing at the end of a successful search session [23]. This line of work is generally quite effective as it directly optimizes for successful searches, but it requires a way to determine whether a search session was successful, which is a non-trivial problem for public datasets, where metrics such as the stream or dwell time on a document are typically unavailable. Hence, many works in the literature have relied on the simplifying assumption that, within a search session, users optimize their queries until they find what they are looking for, making query reformulations within a search session a natural candidate for query recommendation. As a consequence, several models have relied on mining query-query
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0152/01527388-9916-4d5d-84d2-614f1e26de47.png" style="width: 50%;"></div>
Figure 1: An example of prompt-based query recommendations for the query “Nutella”. The Large Language Model is prompted with two queries “Ryanair” and “New York”, and their relative recommendations, and a user query, and corresponding query recommendations are generated
<div style="text-align: center;">Figure 1: An example of prompt-based query recommendations for the query “Nutella”. The Large Language Model is prompted with two queries “Ryanair” and “New York”, and their relative recommendations, and a user query, and corresponding query recommendations are generated</div>
correlations from query logs, either through more traditional statistical approaches such as Log-Likelihood Ratio [19], graph-based approaches [2, 4, 9, 16] or using modern language technologies that can generalize to new queries and effectively model sequences such as RNNs [42] or transformers [26]. Nevertheless, these approaches still require access to a set of high-quality query logs, which are expensive to collect, maintain, and might not be available or be representative in a cold-start situation, e.g. launch in a new country, for a new product, etc. To deal with cold-start issues, some approaches for query recommendations in the absence of query logs have been proposed in the past. For example, Bhatia et al. [3] build a (pre-neural) language model, that is based on a database of sentences, such as those contained in the document collection. This language model is obtained by extracting the n-grams (in the order of unigram, bigram and trigram) from the collection. Then, given a user query, their approach relies on the last word of the user’s query to find the most probable subsequent term to obtain a phrase completion with maximum likelihood estimation. However, these collection-extractive approaches are based on finding relevant keyphrases in a collection of documents and do not take advantage of the outstanding generative capabilities of Large Language Models, which have shown to be extremely effective in IR tasks such as query generation (see Section 2.3). To the best of our knowledge, this is the first approach to generate query recommendations that only requires a pre-trained neural network, without any access to user data or documents.
# 2.2 Large Language Models
Pre-trained Language Models (LMs) such as BERT [13] and T5 [30] are nowadays the main building block of the major Natural Language Processing (NLP) architectures. After being fine-tuned, these LMs demonstrate impressive capabilities in a variety of tasks such as Question Answering, Machine Translations, Semantic Parsing, and so on [21]. Recent advances in the field of language modelling
have led to the creation of pre-trained Large Language Models (LLMs) such as GPT [7], which are LMs with a huge number of parameters, trained on large amounts of data. While LMs require task-specific fine-tuning to update their parameters to correctly address a specific downstream task, LLMs are able to perform incontext learning, i.e., the ability to address a specific downstream task by learning from just a few examples in the task demonstration context [40, 41]. These few examples are usually written in natural language templates, or prompts. In-context learning concatenates a few examples of the demonstration context together with a question, which is fed into the LLM to generate an answer. In doing so, the LLM does not require supervised fine-tuning, and directly performs inference using the provided prompt. For example, given a LLMs and an english sentence, we can condition the model’s output by asking to address different tasks such as Translate the utterance in German or Paraphrase this utterance, and we can also combine them like Translate and paraphrase this sentence in German. Many LLM have been developed and trained, such as GPT3 [7], BLOOM [32]), and OPT [45]). Despite their capabilities, the performance of LLMs is sensitive to the prompting template [46]. For our solution we propose a task-specific prompt to generate query recommendations, and provides the first evaluation of a LLM’s incontext learning for the query recommendations generation task.
# 2.3 Prompting in Information Retrieval
Nogueira et al. [27] have been among the first to propose a promptbased approach for relevance ranking using a LM [30]. Given a query and a document, their texts are concatenated with taskspecific tokens, and the task was to predict if the document would be relevant for the given query. However, the proposed template has been used to generate training samples to fine-tune the LM on the relevance classification task. Sachan et al. [31] demonstrate how prompting can be used for re-ranking documents without
fine-tuning, i.e., in a zero-shot setting. Their re-ranker scores documents with a zero-shot LM by computing the probability of the input query conditioned on a retrieved document. InPars [6, 18] and Promptagator [12] demonstrated how prompting and LLM can be used to generate synthetic training datasets for IR tasks. Both models create a prompt composed of a concatenation of few query-relevant document pairs followed by a document, and the goal is to generate a query for which the final document is relevant. This prompt is then used with the GPT3 [6, 12], GTP-J [38], or FLAN [39] LLMs to generate new synthetic query-document pairs. Recently, prompt-based IR pushed as far as replacing the classic index-retrieval pipeline with a single large language model that directly generates relevant document IDs given a user query [35]. Inspired by the successes of these works that use prompting to generate queries, we focus on prompting the query recommendation generation problem.
# 3 PROPOSED APPROACH
We reframe the query recommendation problem as a generative task introducing Generative Query Recommendation (GQR). GQR uses a pre-trained sequence-to-sequence LLM as its foundation, and it leverages the prompting abilities of the LLM to understand the recommendation task through a few examples provided in the prompt. Indeed, our methodology consists in building a proper prompt composed of pairs of queries and a list of corresponding query recommendations to instruct an LLM to generate recommendations for a new, previously unseen user query. More formally, given a query input text 𝑞and a candidate recommendations set L = {𝑟1,𝑟2, . . .}, i.e., a set of free-text recommendations, a pre-trained LLM M takes the top 𝑘candidate recommendations in L with the highest probabilities of the language model M, conditioned on 𝑛examples C = � (𝑞′ 1, ℓ′ 1), (𝑞′ 2, ℓ′ 2), . . . , (𝑞′𝑛, ℓ′𝑛) � , where (𝑞′ 𝑖, ℓ′ 𝑖) is a query-list of recommendations pair whose query is different from the query input 𝑞. Since the recommendation space L is potentially very large, we do not explicitly compute and store it, but we leverage the text generation capabilities of M to synthesize recommendations. Our proposed GQR system generates the highest scoring 𝑘recommendations ˆℓ= {ˆ𝑟1, . . . , ˆ𝑟𝑘} ⊂L by selecting the 𝑘elements of L with the highest generation probability:
(1)
In practice, we use a pre-defined template, or prompt, to format the examples and pre-append them to the input query. Let 𝑇(·, ·) the formatting function of an input example, i.e.,
(2)
The context C and the input query 𝑞are provided as input to the the LLM M according to the following prompt:
(3)
Then, to generate the query recommendation, we provide the input build according to this prompt to the LLM, and we let the LLM generate the recommendations, as shown in Figure 1. Hence, an example of our prompt is
query: 𝑞1
# recommendations: ℓ1
... query: 𝑞𝑛 recommendations: ℓ𝑛 query: 𝑞′ recommendations:
Then, our LLM will condition its output using the prompt. It will continue to complete the input text adding the recommendations for the query 𝑞′. To get the most general prompt possible, we use different examples that cover several topics, as shown in Figure 2. The examples composing our prompt include queries related to entities of different natures, such as company names, famous person names, product names, historical events, etc. For each prompt, we built hand-crafted recommendations, which will be released upon acceptance. Prompting brings several advantages in query recommendation generation, considerably easing the different constraints required in the current query recommendation systems. GQR consists of a prompt and a general-purpose LLM, that, by using an in-context learning strategy, is able to generate query recommendations in a single forward pass, and it does not need additional ad hoc systems to produce, rank, and combine query recommendations. The system’s overall inductive biases that affect the final output are easily controllable through the human-created prompt in plain language. Further, GQR exploits the large pre-training knowledge of the LLM to retrieve the information needed to generate the recommendations, removing the need to index large collections of documents to extract query recommendations. By re-using the pre-trained knowledge of the LLM, we obtain an approach that does need specific training (or fine-tuning) to produce query recommendations. In that way, we discontinue the use of users’ data, such as query logs, because no data is involved in our query suggestion pipeline, resulting in an approach that is robust to cold-start. Indeed, the LLM is typically pre-trained on an extremely large and diverse set of texts, allowing it to distill both language and factual knowledge into a single model. It, therefore, equips our approach with an open, unconstrained source of recommendations in contrast to being bound to a finite set of recommendations extracted from a query log. As for the language aspect, our approach leverages all the common advantages of pre-trained transformer architectures [13], such as Word Pieces embeddings [43], making the model more robust to typing mistakes [44] and able to handle different word forms effectively, and able to understand the context and semantics of the query. To sum up, this approach completely eradicates the cold-start issues, it demonstrates that there is no need for large collections of documents, to generate query recommendations, and the need to construct a query-specific architecture, allowing the reuse of existing models, thus promoting a greener approach [29].
# 4 EXPERIMENTAL SETUP
In this section, we describe our experimental setup, introducing our research questions (RQs) and the datasets, the competing models, the hardware, and the evaluation metrics used to answer our RQs.

# e of one of our prompts with two examples. As can be noticed from this Figure, we concatenated the user ompt and then we let the LLM to continue the generation of the query recommendations.
Figure 2: Instance of one of our prompts with two examples. As can be noticed from this Figure, we concate query with the prompt and then we let the LLM to continue the generation of the query recommendations.
# 4.1 Research Questions
In our experiments, we focus on answering the following research questions
• RQ1: Can our proposed GQR system generate relevant and useful query recommendations compared with existing query recommendation systems? • RQ2: Are the queries generated by our GQR system more engaging for users than those generated by other systems? • RQ3: Does our GQR system generate recommendations for long tail, i.e., rare, queries? • RQ4: Are query logs still bring value in generative query recommendation?
# 4.2 Datasets
We use three datasets, TREC Robust 2004 Disk 4-5 [17, 36, 37]. with 250 queries and 528,155 documents, ClueWeb09B [8] with 200 queries and 50,220,423 documents and a subset of queries from the AOL dataset [28] , specifically 192 queries randomly sampled to perform a user study and the first 200 queries extracted from the query log’s long tail distribution, i.e., queries appearing just once in the log. We will release the datasets upon acceptance.
# 4.3 Tested Models
We compare our proposed GQR approaches with two query recommendation systems implemented by publicly available Web search engines, referred to as System 1 and System 2 to maintain anonymity. To compute their performance, we collect the recommendations they produce by scraping their outputs in response to our test queries. Moreover, we compare GQR against the most similar query recommendation system available in the research literature, which means a system not exploiting query logs for producing query recommendations [3]. Given that their implementation cannot be found, we reproduce their work in our experiments. We conducted the experiments of Bhatia et al. 2011 [3] only on Robust 2004 because computing 𝑛-grams on a large collection like ClueWeb09B would be too demanding on memory. 1 We implement our GQR system using two different LLMs: GPT3 [7] using the implementation of text-davinci-003 model with 175B parameters accessed through the OpenAI APIs2, and Bloom [32]
1Indeed, by a simple back-of-the-envelope calculation, the approach of Bhathia et al. would require more than 1TB of RAM to store just the index. By observing the experimental results we shall present in the next sections, we decided that adding this experiment would not have added any additional insights. 2https://platform.openai.com/playground
with 176B parameters, using the Huggingface’s APIs3. Both GQR systems exploit a prompt composed of 10 examples by default. We implement RA-GQR using our best-performing model, GPT-3. RA-GQR leverages past user queries from Lucchese et al. (2013) [24]. Instead of relying on generic prompts, RA-GQR dynamically builds its prompts by incorporating real user queries similar to the current one. This approach aims to provide GQR with relevant in-context learning examples. To achieve this, RA-GQR first embeds the queries using a pre-trained model, msmarco-roberta-base-v2. Then, it employs FAISS [14] to efficiently retrieve past user queries that share the same topic as the current query. RA-GQR builds upon the concept of Retrieval Augmented Generation (RAG) introduced by Lewis et al. [20]. For all the GQR implementations, we use the default hyperparameters: temperature of 0.7, with a maximum of 256 input tokens for both models. To maintain consistency with the outputs of System 1 and System 2, which typically generate six related searches, we decided to evaluate the top six query recommendations generated by GPT3 and Bloom.
# 4.4 Evaluation Protocols
To assess the performance of query recommendation systems, we exploit two different evaluation protocols. Given a query 𝑞and a set of 𝑘query recommendations 𝑟1,𝑟2, ...,𝑟𝑘generated by a given query recommendation system, our first protocol, named Substitution, evaluates the effectiveness of each recommendation independently, i.e., the original query is replaced with one of the generated recommendations:
Substitution(𝑞,𝑖) ≡𝑟𝑖.
In doing so, we simulate a user selecting a single query recommendation to be used in place of the submitted query. However, this protocol does not take into account the query recommendation capabilities of a system as a whole. Hence we introduce our second protocol, named Concat, consisting of an iterative concatenation of the original query with the first 𝑖query recommendations:
In doing so, we simulate a user selecting a single query recommendation to be used in place of the submitted query. However, this protocol does not take into account the query
(5)
Concat(𝑞,𝑖) ≡𝑞⊕𝑟1 ⊕· · · ⊕𝑟𝑖.
The goal of this protocol is to assess the increase in performance that a user might get when the search results are (possibly) enriched by the additional information coming from the results retrieved thanks to the considered set of query recommendations.
3https://huggingface.co/bigscience/bloom
IR-RAG @ SIGIR ’24, July 14–18, 2024, Washington D.C., USA
# 4.5 Performance Metrics
Simplified Clarity Score. A common metric used in query recommendations is the Clarity Score (CS) [10], used to measure the specificity or ambiguity of a query. We exploit the Simplified Clarity Score (SCS) implementation [15], which is less time-consuming when compared to the complexity of Clarity Score. SCS relies on calculating a relevance model based on the query and compares it to the relevance model generated by the entire corpus using Kullback–Leibler Divergence: ∑︁
(6)
where 𝑝(𝑤|𝑞) is the statistical language model built over the query, while 𝑝(𝑤|𝐶) is the statistical language model built over the document collection The quantity 𝑝(𝑤|𝑞) is estimated with the maximum likelihood of the query model of the word 𝑤in query 𝑞and it is computed as the ratio between the number of occurrences of the word 𝑤in the query and the query length. The quantity 𝑝(𝑤|𝐶) is computed as the ratio between the number of occurrences of the word 𝑤in the collection 𝐶and the total number of words in 𝐶. In other words, a high SCS indicates a well-formed query with little ambiguity that leads to retrieving a highly coherent set of documents. It has been shown to correlate well with relevance judgments and it is widely used as a query performance predictor in absence of explicit relevance judgments [10]. Retrieval effectiveness. As mentioned in Section 1, query recommendations are a useful tool to help users in producing better queries to retrieve the documents that satisfy their information needs. So, to assess the retrieval effectiveness of the generated query recommendations, we test their effectiveness to retrieve relevant documents concerning the original query. Our reference effectiveness metric is the Normalized Discounted Cumulative Gain at cutoff 10 (NDCG@10). To perform our experiments, we used the BM25 ranking function from the PyTerrier library [25]. All statistical significance differences are computed with the paired t-test (𝑝< 0.01) with the Holm-Bonferroni multiple testing correction.
# 4.6 User study
The main goal of a query recommender system should be that of producing helpful recommendations, not only effective from a retrieval point of view recommendations. Therefore, we aim to determine which system, amongst the ones we tested, presents the most engaging query recommendations from a user’s perspective. We engage twelve professional annotators with the recommendations generated by System 1, System 2, and GQR (GPT-3). We divide annotators into three groups, i.e., four annotators per group. We also randomly partition the 192 queries randomly sampled from the AOL query log into the above-mentioned three groups, ending up with 64 queries per group. The user study is conducted as follows: each system is anonymously identified and shuffled for each query to eliminate bias. In the user study, our input queries always get at least six recommendations from each system. The order of the recommendations shown to the annotators is not taken into account. We ask the annotators to choose the most helpful and engaging set of recommended
queries that do not repeat the same information. As an example of our guideline recommendations, if one of the systems under study provides recommendations such as "Ryanair support, Ryanair contact, Ryanair customer service" w.r.t. the input query "Ryanair", that system is not helpful in satisfying the user’s information need because it is just repeating the same recommendation. A better system would provide diverse recommendations, such as "Ryanair careers, Ryanair history, Ryanair cheap destinations."
# 5 RESULTS AND ANALYSIS
In this section, we present the results to answer our posed five research questions (Section 4.1). RQ1: Can our proposed GQR system without query-logs generate relevant and useful query recommendations compared with existing query recommendation systems? In Tables 1 and 2 we report the simplified clarity score and the NDCG@10 metrics, respectively, for the tested query recommendation systems, according to the Substitution evaluation protocol. For each system, collection, and metric, we report the minimum and maximum metric values across the six generated recommendations, as well as their average values and the standard deviations. According to the SCS metric, our GQR systems get the highest values with both the worst and the best-performing recommendations on the Robust04 dataset, outperformed only by System 2’s best recommendation on the ClueWeb09B dataset, but by a small margin. However, on average, our proposed GQR (GPT-3) system outperforms all other methods, including commercial systems. Moreover, the corresponding standard deviation values are the smallest, meaning that all the generated query recommendations obtain a similar, high SCS score. According to the NDCG@10 metric, our proposed GQR (GPT-3) system outperforms all other systems. With respect to the best competitor, namely System 2, our experiments show that GQR (GPT-3), on average, has relative improvements of +23.86% and a +26.63% in NDCG@10 on Robust04 and ClueWeb09B, respectively. As with the SCS metric, the standard deviation of the NDCG@10 scores across the six generated recommendations is small, so the recommendations get an NDCG@10 score concentrated around the average value. In Tables 3 and 4, we report the clarity score and the NDCG@10 metrics, respectively, for the tested query recommendation systems, according to the Concat evaluation protocol. This protocol aims to assess the performance improvements that a user query may get when the generated query recommendations are used as additional information sources together with the query. In doing so, the query recommendations are expected to better specify the information need of the user expressed as a text query. In doing so, it is important to take into account the number of query recommendations concatenated with the user query, so we report the metric scores for increasing the number of recommendations concatenated, referred to as rank In both datasets, the concatenation of one or more query recommendations increases the SCS value, for all tested systems, as expected. In fact, according to Cronen-Townsend et al. [10], the clarity score of a query increases if we add terms that reduce the query ambiguity. Since we are enriching the user queries with recommendations, these increases confirm that the generated recommendations do not increase the ambiguity of the queries.
Model
Min
Max
Avg ± Std
Robust04
System 1 (a)
9.21
10.17
9.80 ± 0.35
System 2 (b)
9.35
10.56
9.82 ± 0.39
Bhatia [3] (c)
7.64
9.20
8.28 ± 0.53
GQR (Bloom) (d)
7.50
11.00
8.84 ± 1.66
GQR (GPT-3) (e)
10.54
10.77
10.65 ± 0.08
RA-GQR (GPT-3)
16.71𝑎𝑏𝑐𝑑𝑒
17.10𝑎𝑏𝑐𝑑𝑒
16.98 ± 0.19𝑎𝑏𝑐𝑑𝑒
ClueWeb09B
System 1 (a)
9.80
10.80
10.37 ± 0.32
System 2 (b)
10.49
11.31
10.87 ± 0.30
Bhatia [3] (c)
-
-
-
GQR (Bloom) (d)
7.68
11.20
9.84 ± 1.19
GQR (GPT-3) (e)
10.94
11.22
11.12 ± 0.10
RA-GQR (GPT-3)
19.42𝑎𝑐𝑑𝑒
19.76𝑎𝑐𝑑𝑒
19.53 ± 0.20𝑎𝑏𝑐𝑑𝑒
Table 1: SCS for the Substitution protocol for each system on
<div style="text-align: center;">ClueWeb09B</div>
Table 1: SCS for the Substitution protocol for each system on Robust04 and ClueWeb09B. The best values across all systems are boldfaced. The letters indicate a statistically significant difference w.r.t. GQR (GPT-3).
Model
Min
Max
Avg ± Std
Robust04
System 1 (a)
0.2038
0.2638
0.2377 ± 0.0211
System 2 (b)
0.2546
0.3739
0.3102 ± 0.0478
Bhatia [3] (c)
0.2388
0.2640
0.2566 ± 0.0108
GQR (Bloom) (d)
0.1743
0.3655
0.2628 ± 0.0714
GQR (GPT-3) (e)
0.3727
0.3947
0.3842 ± 0.0092
RA-GQR (GPT-3)
0.4197𝑎𝑏𝑑𝑒
0.4476𝑎𝑏𝑐𝑑𝑒
0.4339 ± 0.008𝑎𝑏𝑐𝑑
Model
Min
Max
Avg ± Std
Robust04
System 1 (a)
0.2038
0.2638
0.2377 ± 0.0211
System 2 (b)
0.2546
0.3739
0.3102 ± 0.0478
Bhatia [3] (c)
0.2388
0.2640
0.2566 ± 0.0108
GQR (Bloom) (d)
0.1743
0.3655
0.2628 ± 0.0714
GQR (GPT-3) (e)
0.3727
0.3947
0.3842 ± 0.0092
RA-GQR (GPT-3)
0.4197𝑎𝑏𝑑𝑒
0.4476𝑎𝑏𝑐𝑑𝑒
0.4339 ± 0.008𝑎𝑏𝑐𝑑
ClueWeb09B
System 1 (a)
0.0940
0.1129
0.1056 ± 0.0082
System 2 (b)
0.0946
0.1294
0.1108 ± 0.0164
Bhatia [3] (c)
-
-
-
GQR (Bloom) (d)
0.0468
0.1144
0.0802 ± 0.0268
GQR (GPT-3) (e)
0.1280
0.1659
0.1403 ± 0.0146
RA-GQR (GPT-3)
0.1597𝑎𝑏𝑑𝑒
0.1832𝑎𝑏𝑑𝑒
0.1708 ± 0.10𝑎𝑏𝑑𝑒
Table 2: NDCG@10 for the Substitution protocol for each
<div style="text-align: center;">ClueWeb09B</div>
System 1 (a)
0.0940
0.1129
0.1056 ± 0.0082
System 2 (b)
0.0946
0.1294
0.1108 ± 0.0164
Bhatia [3] (c)
-
-
-
GQR (Bloom) (d)
0.0468
0.1144
0.0802 ± 0.0268
GQR (GPT-3) (e)
0.1280
0.1659
0.1403 ± 0.0146
RA-GQR (GPT-3)
0.1597𝑎𝑏𝑑𝑒
0.1832𝑎𝑏𝑑𝑒
0.1708 ± 0.10𝑎𝑏𝑑𝑒
 ± Table 2: NDCG@10 for the Substitution protocol for each system on Robust04 and ClueWeb09B. The best values across all systems are boldfaced. The letters indicate a statistically significant difference w.r.t. GQR (GPT-3).
Among the tested systems, our proposed GQR (GPT-3) outperforms all other systems across all ranks and datasets. Similar results are obtained w.r.t. NDCG@10. In this case, the more recommendations we concatenate to the query, the higher the metric value. With six recommendations, the effectiveness performance increases by ∼16.44% on the Robust04 dataset and by ∼22.58% on the ClueWeb09 dataset by comparing with their single query recommendation with the highest NDCG@10.
Concerning RQ1, we can conclude that our proposed GQR (GPT3) system is able to generate query recommendations without query logs that are, on average, less ambiguous than other systems and better or on par with commercial competitors. RQ2: Are the queries generated by our GQR system more engaging for users than those generated by other systems? To answer this research question, we report the results of our user study in Table 5. We exclude Bathia’s approach because it needs to index a collection of documents which is not provided with the AOL queries. The user study shows a clear preference for our GQR (GPT-3) system with respect to the other two commercial systems. All groups report a preference above 50% of tested queries for the GQR (GPT-3) system, with an overall preference of ∼59%. The second most preferred system is System 1 with ∼26% of the preferences on average, and the last system is System 2, with ∼15% of preferencess on average. Hence, on RQ2, we can conclude that, according to our user study, the recommendations generated by our GQR (GPT-3) system are more engaging than the recommendations generated by the two commercial competitors from the user’s point of view. It is worth noticing that AOL queries cover a wide variety of topics and since it is impossible to make assumptions about users’ prior knowledge and their personal preferences, for that reason, subjectivity plays a crucial role. RQ3: Does our GQR system generate recommendations for long tail, i.e., rare, queries? During our experiments on Robust04 and ClueWeb09B, we observed that some systems fail to produce query recommendations for some queries, while our GQR (GPT-3) system is always successful in generating at least 1 suggestion for each query. To further investigate on this topic, we extracted 200 rare queries from the tail distribution of the AOL query log, thus considering the queries with a frequency equal to 1-2 in the query log. In Table 6 we report, in percentage, the number of times our tested systems have been able to generate at least one and all six recommendations for the queries Robust04, ClueWeb09B and the 200 AOL tail queries datasets. We also report the average number of query recommendations each system successfully generated for each dataset. For the Robust04 and ClueWbe09B datasets, all competitors are able to generate more than 5 recommendations on average. On average, all systems generate at least one recommendation for more than 90% of the Robust04 queries, and for more than 98% on ClueWeb09B. System 1 and System 2 are not able to generate at least one recommendation 9% and 17% of the times on AOL tail queries, respectively. Our GQR (Bloom) system generates at least one query recommendation 99% of the times, but it is the worst performing system when generating all six recommendations. However, our GQR (GPT-3) system always succeeds in generating six recommendations on all tested datasets. So, concerning RQ3, we can conclude that our GQR (GPT-3) system always produces recommendations for all ranks. RQ4: Are query logs still bringing value in generative query recommendation? In our analysis, we show how a system comprised of a single component (the LLM) without using query logs can outperform more complex state-of-the-art query recommender systems that utilize query logs, such as System 1 and System 2. This suggests that query
Concerning RQ1, we can conclude that our proposed GQR (GPT3) system is able to generate query recommendations without query logs that are, on average, less ambiguous than other systems and better or on par with commercial competitors. RQ2: Are the queries generated by our GQR system more engaging for users than those generated by other systems? To answer this research question, we report the results of our user study in Table 5. We exclude Bathia’s approach because it needs to index a collection of documents which is not provided with the AOL queries. The user study shows a clear preference for our GQR (GPT-3) system with respect to the other two commercial systems. All groups report a preference above 50% of tested queries for the GQR (GPT-3) system, with an overall preference of ∼59%. The second most preferred system is System 1 with ∼26% of the preferences on average, and the last system is System 2, with ∼15% of preferencess on average. Hence, on RQ2, we can conclude that, according to our user study, the recommendations generated by our GQR (GPT-3) system are more engaging than the recommendations generated by the two commercial competitors from the user’s point of view. It is worth noticing that AOL queries cover a wide variety of topics and since it is impossible to make assumptions about users’ prior knowledge and their personal preferences, for that reason, subjectivity plays a crucial role.
# RQ4: Are query logs still bringing value in generative query recommendation?
Model
rank 1
rank 2
rank 3
rank 4
rank 5
rank 6
Robust04
System 1 (a)
16.29
20.00
23.32
26.55
29.18
31.43
System 2 (b)
19.35
24.81
31.10
36.63
42.84
47.88
Bhatia [3] (c)
18.21
23.89
29.52
34.49
39.56
44.28
GQR (Bloom) (d)
19.13
25.41
31.08
36.12
40.48
44.20
GQR (GPT-3) (e)
20.00
27.69
34.98
42.23
49.47
56.67
RA-GQR (GPT-3)
27.23𝑎𝑏𝑐𝑑𝑒𝑓
40.69𝑎𝑏𝑐𝑑𝑒𝑓
54.51𝑎𝑏𝑐𝑑𝑒𝑓
68.09𝑎𝑏𝑐𝑑𝑒𝑓
81.98𝑎𝑏𝑐𝑑𝑒𝑓
95.48𝑎𝑏𝑐𝑑𝑒𝑓
ClueWeb09B
System 1 (a)
19.20
24.89
30.54
36.03
41.23
46.24
System 2 (b)
19.28
24.50
30.77
35.55
41.55
46.70
Bhatia [3] (c)
-
-
-
-
-
-
GQR (Bloom) (d)
18.49
25.20
31.15
37.07
42.28
46.72
GQR (GPT-3) (e)
20.41
27.73
35.01
42.27
49.30
56.25
RA-GQR (GPT-3)
29.63𝑎𝑏𝑐𝑑𝑒
46.43𝑎𝑏𝑐𝑑𝑒
63.12𝑎𝑏𝑐𝑑𝑒
79.32𝑎𝑏𝑐𝑑𝑒
95.87𝑎𝑏𝑐𝑑𝑒
112𝑎𝑏𝑐𝑑𝑒
S for the Concat protocol for each system on Robust04 and ClueWeb09B. The best values per rank across
<div style="text-align: center;">le 3: SCS for the Concat protocol for each system on Robust04 and ClueWeb09B. The best values per rank across all systems boldfaced. The letters denote a statistically significant difference w.r.t. GQR (GPT-3).</div>
<div style="text-align: center;">able 3: SCS for the Concat protocol for each system on Robust04 and ClueWeb09B. The best values per rank across all sy re boldfaced. The letters denote a statistically significant difference w.r.t. GQR (GPT-3).</div>
Model
rank 1
rank 2
rank 3
rank 4
rank 5
rank 6
Robust04
System 1 (a)
0.4151
0.4270
0.4320
0.4373
0.4450
0.4417
System 2 (b)
0.4236
0.4292
0.4361
0.4386
0.4433
0.4373
Bhatia [3] (c)
0.4044
0.4012
0.3956
0.3917
0.3900
0.3922
GQR (Bloom) (d)
0.4252
0.4254
0.4149
0.4136
0.4172
0.4203
GQR (GPT-3) (e)
0.4316
0.4429
0.4507
0.4543
0.4597
0.4596
RA-GQR (GPT-3)
0.4553𝑎𝑏𝑐𝑑
0.4690𝑎𝑏𝑐𝑑𝑒
0.4788𝑎𝑏𝑐𝑑𝑒
0.4869𝑎𝑏𝑐𝑑𝑒
0.4876𝑎𝑏𝑐𝑑𝑒
0.4889𝑎𝑏𝑐𝑑𝑒
ClueWeb09B
System 1 (a)
0.1598
0.1627
0.1832
0.1797
0.1912
0.1954
System 2 (b)
0.1591
0.1429
0.1506
0.1528
0.1562
0.1645
Bhatia [3] (c)
-
-
-
-
-
-
GQR (Bloom) (d)
0.1586
0.1626
0.1576
0.1544
0.1443
0.1427
GQR (GPT-3) (e)
0.1960
0.1891
0.1919
0.1939
0.2010
0.2030
RA-GQR (GPT-3)
0.1937𝑎𝑏𝑐𝑑
0.1951𝑎𝑏𝑐𝑑
0.1985𝑎𝑏𝑐𝑑
0.2003𝑎𝑏𝑐𝑑
0.2078𝑎𝑏𝑐𝑑
0.2058𝑎𝑏𝑐𝑑
CG@10 for the Concat protocol for each system on Robust04 and ClueWeb09B. The best values per ran
logs are not essential for building effective query recommender systems. Nevertheless, we recognize query logs as an important source of knowledge, as demonstrated by the performance of RA-GQR. According to the Substitution protocol, RA-GQR demonstrates significant improvements over all other systems tested, including GQR (GPT-3). As shown in Table 1 (1), RA-GQR enhances the SCS of GQR by approximately ∼59% on Robust04 and ∼76% on ClueWeb09B, compared to GQR (GPT-3). Additionally, Table 2 (1) shows that RA-GQR increases the NDCG@10 of GQR by roughly ∼13% on Robust04 and ∼22% on ClueWeb09B. In the Concat protocol, we observe similar improvements. Table 3 (3) confirms that RA-GQR nearly doubles the performance of GQR (GPT-3) on both datasets. Moreover, Table 4 (4) indicates that
RA-GQR improves the NDCG@10 by about ∼6% on Robust04 and ∼2% on ClueWeb09B, compared to GQR (GPT-3). Overall, RA-GQR enhances the NDCG@10 by approximately ∼11% on Robust04 and ∼6% on ClueWeb09B with respect to the second-best system. Therefore, in response to RQ4, while query logs are no longer essential, behavioral data still prove beneficial in enhancing the performance of generative query recommender systems.
# 6 CONCLUSIONS AND FUTURE WORK
In this work, we reframed the query recommendation task as a generative task presenting Generative Query Recommendation (GQR): a system that generates recommendations GQR generates query recommendations in a single forward pass without relying
Annotator ID
System 1
System 2
GQR (GPT-3)
Group 1
Ann 1
20.31%
15.62%
64.06%
Ann 2
32.81%
15.63%
51.56%
Ann 3
31.25%
18.75%
50.00%
Ann 4
26.98%
11.11%
61.90%
Avg ± Std
27.84% ± 4.84%
15.28% ± 2.72%
56.88% ± 6.17%
Group 2
Ann 5
20.31%
23.44%
56.25%
Ann 6
39.06%
14.06%
46.88%
Ann 7
25.00%
12.50%
62.50%
Ann 8
31.25%
17.19%
51.56%
Avg ± Std
28.91% ± 7.03%
16.80% ± 4.19%
54.30% ± 5.78%
Group 3
Ann 9
17.19%
18.76%
64.06%
Ann 10
26.56%
9.38%
64.06%
Ann 11
18.75%
14.06%
67.19%
Ann 12
25.00%
4.69%
70.63%
Avg ± Std
21.88% ± 3.98%
11.72% ± 5.24%
66.41% ± 2.71%
Overall
26.21% ± 6.26%
14.60% ± 4.69%
59.19% ± 7.33%
Table 5: The annotator preferences (in percentage) in our
Table 5: The annotator preferences (in percentage) in our user study, for each group and system under evaluation.
Table 5: The annotator preferences (in percentage) in our user study, for each group and system under evaluation.
Model
One reccom.
Six reccom.
Average
Robust04
System 1
92%
87%
5.39
System 2
98%
86%
5.54
Bhatia
95%
89%
5.51
GQR (Bloom)
99%
55%
4.69
GQR (GPT-3)
100%
100%
6
RA-GQR (GPT-3)
100%
100%
6
ClueWeb09B
System 1
98%
92%
5.75
System 2
99%
95%
5.85
GQR (Bloom)
99%
63%
5.1
GQR (GPT-3)
100%
100%
6
RA-GQR (GPT-3)
100%
100%
6
AOL long tail queries
System 1
83%
71%
4.61
System 2
91%
86%
5.37
GQR (Bloom)
99%
66%
5.15
GQR (GPT-3)
100%
100%
6
RA-GQR (GPT-3)
100%
100%
6
Table 6: Percentage of queries for which at least one and
<div style="text-align: center;">AOL long tail queries</div>
System 1
83%
71%
4.61
System 2
91%
86%
5.37
GQR (Bloom)
99%
66%
5.15
GQR (GPT-3)
100%
100%
6
RA-GQR (GPT-3)
100%
100%
6
Table 6: Percentage of queries for which at least one and all six recommendations (reccom.) are generated, and the average number of generated recommendations per query for different systems and datasets.
on long cascading pipelines and multiple systems to generate and re-rank the recommendations but leveraging a single component of
an LLM and its knowledge. GQR discontinues the use of users’ data, such as query logs, resulting in an approach that is robust to longtail queries, and cold-start recommendations. Additionally, GQR does not require any task-specific architecture to be trained or finetuned because no data are involved in our query recommendation pipeline. From a practical standpoint, this work has huge implications, as it proves that a query recommendation system that is competitive or even better than industry standards can be created with a pre-trained LLM that does not require any specific fine-tuning, but only a handful of curated prompts that can be obtained straightforwardly. This approach strongly democratizes the creation of query recommendation systems, and drastically reduces the timeto-market for new systems or existing systems that want to expand in new domains or markets. We compared GQR against the previous state-of-the-art model presented by Bhatia et al. 2011 and two of the most popular search engines. Analyzing the results, our approach outclasses the other systems on Simplified Clarity Score and Retrieval Effectiveness metrics and in our blind user study. In terms of effectiveness, GQR (GPT3) achieves an improvement of up to ∼4% of NDCG@10 score in Robust04 and ClueWeb09B w.r.t. the best competitor system. Regarding the Simplified Clarity Score, GQR (GPT-3) outperforms the other systems on average, and it turns out to be the system with the lowest standard deviation, demonstrating that it has the most stable performance along the recommendations generated in the various ranks. We then propose an enhanced GQR proposing Retrieval Augmented GQR, which is capable of retrieving similar facts from a query log to compose its prompt dynamically, which further improves its performance. Indeed, it obtains improvements of ∼11% and ∼6% in Robust04 and ClueWeb09B with respect to the second-best competitor. In our blind user study, we recorded 60% of preferences for our system, indicating that our method is not only superior in terms of metrics but also capable of generating engaging queries for users. From our methodological perspective, we have proven that genera purpose LLMs can compete or even outperform other established methods for a fundamental search task. In future work, we plan to delve deeper into the customization of prompts for query recommendations, investigating how nuanced modifications can enhance the effectiveness of our system. Additionally, we aim to extend our exploration to include exploratory queries, assessing how different configurations of prompts can improve engagement and relevance in more complex or open-ended information-seeking tasks.
# ACKNOWLEDGMENTS
This work is supported by the Spoke “FutureHPC & BigData" of the ICSC – Centro Nazionale di Ricerca in High-Performance Computing, Big Data and Quantum Computing, the Spoke “Humancentered AI” of the M4C2 - Investimento 1.3, Partenariato Esteso PE00000013 - "FAIR - Future Artificial Intelligence Research", SERICS (PE00000014), IR0000013 - SoBigData.it, funded by European Union – NextGenerationEU, the FoReLab project (Departments of Excellence), and the NEREO PRIN project funded by the Italian Ministry of Education and Research Grant no. 2022AEFHAZ. This work
was carried out while Andrea Bacciu was enrolled in the Italian
National Doctorate on Artificial Intelligence run by the Sapienza University of Rome.
# REFERENCES
[1] Ranieri Baraglia, Fidel Cacheda, Victor Carneiro, Diego Fernandez, Vreixo Formoso, Raffaele Perego, and Fabrizio Silvestri. 2009. Search shortcuts: a new approach to the recommendation of queries. , 77–84 pages. [2] Doug Beeferman and Adam Berger. 2000. Agglomerative clustering of a search engine query log. , 407–416 pages. [3] Sumit Bhatia, Debapriyo Majumdar, and Prasenjit Mitra. 2011. Query suggestions in the absence of query logs. , 795–804 pages. [4] Paolo Boldi, Francesco Bonchi, Carlos Castillo, Debora Donato, and Sebastiano Vigna. 2009. Query suggestions using query-flow graphs. , 56–63 pages. [5] Francesco Bonchi, Raffaele Perego, Fabrizio Silvestri, Hossein Vahabi, and Rossano Venturini. 2012. Efficient query recommendations in the long tail via center-piece subgraphs. , 345–354 pages. [6] Luiz Bonifacio, Hugo Abonizio, Marzieh Fadaee, and Rodrigo Nogueira. 2022. InPars: Unsupervised Dataset Generation for Information Retrieval. , 6 pages. https://doi.org/10.1145/3477495.3531863 [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. , 1877–1901 pages. [8] Charles L. A. Clarke, Nick Craswell, and Ian Soboroff. 2009. Overview of the TREC 2009 Web Track. [9] Nick Craswell and Martin Szummer. 2007. Random walks on the click graph. , 239–246 pages. [10] Steve Cronen-Townsend, Yun Zhou, and W Bruce Croft. 2002. Predicting query performance. , 299–306 pages. [11] Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Zhifang Sui, and Furu Wei. 2022. Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta Optimizers. [12] Zhuyun Dai, Vincent Y Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B Hall, and Ming-Wei Chang. 2022. Promptagator: Few-shot dense retrieval from 8 examples. [13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. [14] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024. The faiss library. arXiv preprint arXiv:2401.08281 (2024). [15] Ben He and Iadh Ounis. 2004. Inferring query performance using pre-retrieval predictors. , 43–54 pages. [16] Chien-Kang Huang, Lee-Feng Chien, and Yen-Jen Oyang. 2003. Relevant term suggestion in interactive web search based on contextual information in query session logs. , 638–649 pages. [17] Samuel Huston and W. Bruce Croft. 2014. A Comparison of Retrieval Models using Term Dependencies. [18] Vitor Jeronymo, Luiz Bonifacio, Hugo Abonizio, Marzieh Fadaee, Roberto Lotufo, Jakub Zavrel, and Rodrigo Nogueira. 2023. InPars-v2: Large Language Models as Efficient Dataset Generators for Information Retrieval. [19] Rosie Jones, Benjamin Rey, Omid Madani, and Wiley Greiner. 2006. Generating query substitutions. , 387–396 pages. [20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems 33 (2020), 9459–9474. [21] J. Lin, R. Nogueira, and A. Yates. 2021. Pretrained Transformers for Text Ranking: BERT and Beyond. [22] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2022. Pre-Train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing. https://doi.org/10.1145/ 3560815 Just Accepted. [23] Claudio Lucchese, Salvatore Orlando, Raffaele Perego, Fabrizio Silvestri, and Gabriele Tolomei. 2011. Identifying Task-Based Sessions in Search Engine Query
Logs. , 10 pages. https://doi.org/10.1145/1935826.1935875 [24] Claudio Lucchese, Salvatore Orlando, Raffaele Perego, Fabrizio Silvestri, Gabriele Tolomei, et al. 2013. Modeling and predicting the task-by-task behavior of search engine users. OAIR 13 (2013), 77–84. [25] Craig Macdonald, Nicola Tonellotto, Sean MacAvaney, and Iadh Ounis. 2021. PyTerrier: Declarative experimentation in Python from BM25 to dense retrieval. , 4526–4533 pages. [26] Agnès Mustar, Sylvain Lamprier, and Benjamin Piwowarski. 2021. On the study of transformers for query suggestion. , 27 pages. [27] Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. 2020. Document Ranking with a Pretrained Sequence-to-Sequence Model. , 708–718 pages. [28] Greg Pass, Abdur Chowdhury, and Cayley Torgeson. 2006. A picture of search. [29] David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David So, Maud Texier, and Jeff Dean. 2021. Carbon emissions and large neural network training. [30] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. [31] Devendra Singh Sachan, Mike Lewis, Mandar Joshi, Armen Aghajanyan, Wen-tau Yih, Joelle Pineau, and Luke Zettlemoyer. 2022. Improving Passage Retrieval with Zero-Shot Question Generation. [32] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022. Bloom: A 176b-parameter open-access multilingual language model. [33] Fabrizio Silvestri. 2009. Mining query logs: Turning search usage data into knowledge. , 174 pages. [34] Alessandro Sordoni, Yoshua Bengio, Hossein Vahabi, Christina Lioma, Jakob Grue Simonsen, and Jian-Yun Nie. 2015. A hierarchical recurrent encoder-decoder for generative context-aware query suggestion. , 553–562 pages. [35] Yi Tay, Vinh Q Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, et al. 2022. Transformer memory as a differentiable search index. [36] Ellen Voorhees. 2004. Overview of the TREC 2004 Robust Retrieval Track. [37] Ellen M. Voorhees. 1996. NIST TREC Disks 4 and 5: Retrieval Test Collections Document Set. https://doi.org/10.18434/t47g6m [38] Ben Wang and Aran Komatsuzaki. 2021. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. [39] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. [40] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022. Emergent abilities of large language models. [41] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. [42] Bin Wu, Chenyan Xiong, Maosong Sun, and Zhiyuan Liu. 2018. Query suggestion with feedback memory network. , 1563–1571 pages. [43] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. [44] Shaohua Zhang, Haoran Huang, Jicong Liu, and Hang Li. 2020. Spelling error correction with soft-masked BERT. [45] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. [46] Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. , 12697– 12706 pages.
