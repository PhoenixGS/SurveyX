# MAKING TEXT EMBEDDERS FEW-SHOT LEARNERS
Chaofan Li1,2∗, MingHao Qin1,3,∗, Shitao Xiao1, Jianlyu Chen1,4, Kun Luo1,3, Yingxia Shao2, Defu Lian4, Zheng Liu1†
1: Beijing Academy of Artificial Intelligence 2: Beijing University of Posts and Telecommunications 3: Chinese Academy of Sciences 4: University of Science and Technology of China
{cfli, shaoyx}@bupt.edu.cn qinminghao24@ia.ac.cn stxiao@baai.ac.cn chenjianlv@mail.ustc.edu.cn liandefu@ustc.edu.cn {luokun695, zhengliu1026}@gmail.com
# ABSTRACT
Large language models (LLMs) with decoder-only architectures demonstrate remarkable in-context learning (ICL) capabilities. This feature enables them to effectively handle both familiar and novel tasks by utilizing examples provided within their input context. Recognizing the potential of this capability, we propose leveraging the ICL feature in LLMs to enhance the process of text embedding generation. To this end, we introduce a novel model bge-en-icl, which employs few-shot examples to produce high-quality text embeddings. Our approach integrates task-related examples directly into the query side, resulting in significant improvements across various tasks. Additionally, we have investigated how to effectively utilize LLMs as embedding models, including various attention mechanisms, pooling methods, etc. Our findings suggest that retaining the original framework often yields the best results, underscoring that simplicity is best. Experimental results on the MTEB and AIR-Bench benchmarks demonstrate that our approach sets new state-of-the-art (SOTA) performance. Our model, code and dataset are freely available at https://github.com/FlagOpen/FlagEmbedding.
# INTRODUCTION
Text embeddings are vector representations that capture the semantic and contextual meaning of natural language text. They play a pivotal role in natural language processing (NLP) tasks, facilitating a wide range of applications such as information retrieval, text classification, item recommendation, and question answering (Karpukhin et al., 2020; Xiong et al., 2020; Lu et al., 2020). Pre-trained bidirectional encoder and encoder-decoder architectures have been widely adopted as backbone models for embedding model, owing to their effectiveness in producing high-quality vector embeddings for text thanks to their extensive pre-training (Xiao et al., 2022; Gao et al., 2021). Recent advancements in LLMs have significantly shifted the focus towards embedding models that rely primarily on decoder-only architectures (Ma et al., 2023; Li et al., 2024; Wang et al., 2023). These LLM-based embedding models have demonstrated remarkable improvements in in-domain accuracy and generalization, particularly when trained using supervised learning approaches (Wang et al., 2023). However, despite these advances, embedding models still struggle to follow unseen task instructions and execute complex retrieval tasks Su et al. (2024); Weller et al. (2024). This limitation stems from a mismatch between the relatively narrow range of instructions encountered during training and the broader variety of real-world text embedding tasks. In-context learning (ICL) is a core capability of LLMs, enabling them to incorporate task-specific examples directly into input prompts to generate desired outputs (Radford et al., 2019; Brown, 2020;
∗Co-first authors †Corresponding author
Gao et al., 2020). The scope of ICL extends beyond tasks seen during training; it enables LLMs to generalize to new and complex tasks by learning patterns from the provided examples. This allows LLMs to adapt dynamically to novel tasks without additional training, making them highly applicable to large-scale, real-world scenarios (Wei et al., 2022; Yao et al., 2022; Dong et al., 2022). Recognizing the robust ICL abilities of LLMs, in this study, we propose to generate more adaptable text embeddings with ICL strategy. Specifically, we guide the model by including task-specific examples directly within the query prompt. By doing so, we leverage the ICL capabilities of LLMs to produce embeddings that are not only more relevant to the specific domain but also more generalizable across various contexts. Moreover, LLMs are predominantly utilized for text generation tasks, and adapting them for embedding representation tasks requires specific fine-tuning strategies. Recent studies have introduced various approaches, including the generation of high-quality training data through LLMs (Wang et al., 2023), modifications to attention mechanisms, and changes in pooling methods (Ma et al., 2023; Li et al., 2024). Following previous works (Muennighoff et al., 2024; BehnamGhader et al., 2024), we investigate how to effectively utilize LLMs as embedding models by modifying various architectures, e.g., bidirectional attention, meaning pooling, etc. Our experimental findings indicate that in the ICL scenario, making complex modifications to the models does not lead to significant improvements. Surprisingly, the best results are obtained using the original, unmodified architecture. By employing only the ICL strategy, our model bge-en-icl achieves state-of-the-art (SOTA) results on both the MTEB and AIR-Bench benchmarks. We have also released a multilanguage embedding model bge-multilingual-gemma2 and a lightweight reranker bge-rerankerv2.5-gemma2-lightweight. The lightweight reranker also serves as the teacher model for training embedding models through distillation. Further details are provided in Appendices C and D.
• We propose bge-en-icl, which incorporate few-shot examples into the query side to enhance the query embeddings. This integration leverages the in-context learning (ICL) capabilities of large language models (LLMs) in text embedding tasks. • We rethink and explore how to effectively utilize LLMs as embedding models by evaluating various attention mechanisms, pooling methods, and the incorporation of passage prompts. Our findings highlight that simplicity is best; simply combining ICL capabilities with embedding models can achieve excellent performance. • In contrast to other leading models on the MTEB benchmark, we provide open access to our model checkpoint, dataset, and training scripts.
• We propose bge-en-icl, which incorporate few-shot examples into the query side to enhance the query embeddings. This integration leverages the in-context learning (ICL) capabilities of large language models (LLMs) in text embedding tasks. • We rethink and explore how to effectively utilize LLMs as embedding models by evaluating various attention mechanisms, pooling methods, and the incorporation of passage prompts. Our findings highlight that simplicity is best; simply combining ICL capabilities with embedding models can achieve excellent performance. • In contrast to other leading models on the MTEB benchmark, we provide open access to our model checkpoint, dataset, and training scripts.
# 2 RELATED WORK
Text embedding is a critical research direction in the field of information retrieval, with wide-ranging applications including web search, question answering, and dialogue systems. The fundamental principle involves encoding both queries and documents into embedding vectors within the same latent space. By calculating similarity scores between these vectors, effective retrieval is achieved. In recent years, numerous studies have leveraged pre-trained language models such as BERT (Devlin, 2018), T5 (Raffel et al., 2020), and RoBERTa (Liu, 2019) as the backbone for embedding models. These models have consistently demonstrated superior performance compared to traditional sparse retrieval methods. The capability of the backbone is a crucial determinant in the effectiveness of retrieval systems. (Luo et al., 2024) have demonstrated that performance improves with increased scale and extensive pretraining. Currently, numerous studies have explored the effectiveness of utilizing LLMs as backbone encoders for text embedding tasks. Repllama (Ma et al., 2023) fine-tuned Llama-2 to serve as both a dense retriever and a reranker, demonstrating the effectiveness of applying large language models (LLMs) in text embedding tasks. To further align LLMs with text embedding tasks, Llama2Vec (Li et al., 2024) introduced two pretraining tasks specifically designed to enhance the model’s performance in such tasks, which led to significant improvements on the BEIR benchmark. E5-mistral and Gecko (Wang et al., 2023; Lee et al., 2024b) advanced the training of LLM-based embedding models through the use of syn-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/436d/436d986f-341c-469f-8645-d62a2485aa69.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: The architecture of the ICL-based model.</div>
thetic data, markedly boosting their performance across a diverse range of retrieval and non-retrieval tasks. NV-Embed (Lee et al., 2024a) innovatively proposed a latent attention layer to replace conventional pooling methods and implemented a two-stage training strategy to address the challenge of false negatives in non-retrieval tasks. This model has shown strong performance in both retrieval and non-retrieval domains. Additionally, GRIT (Muennighoff et al., 2024) successfully integrated text embedding and generation within a single LLM, achieving performance levels on par with specialized models focused solely on either embedding or generation. In the exploration of LLMs as embedding models from an unsupervised perspective, LLM2Vec (BehnamGhader et al., 2024) presented a novel unsupervised method to transform decoder-only LLMs into embedding models. This approach demonstrated significant potential for modifying LLM backbone encoders to perform retrieval without any supervision. Similarly, PromptReps (Zhuang et al., 2024) leveraged chat-based LLMs aligned with human preferences to generate high-quality dense representations in an unsupervised manner. The LLM-based embedding models mentioned above exhibit commendable performance across both retrieval and non-retrieval tasks. However, much of the existing work has disproportionately focused on altering model architectures, thereby neglecting the intrinsic capabilities of LLMs. Even models like GritLM, which integrate generation and embedding functionalities, fail to fully exploit the potential ICL capabilities of LLMs within the embedding process. By leveraging the innate ICL capabilities of LLMs, embedding models can be more versatile and adapt to diverse scenarios without necessitating additional fine-tuning. Our model not only achieves SOTA results on the MTEB and AIR-Bench benchmarks but also effectively utilizes the inherent strengths of LLMs across tasks.
# 3 METHOLOGY
# 3.1 IN-CONTEXT LEARNING FOR EMBEDDING MODELS
Previous embedding models often involve directly inputting the query into the model to generate target embeddings. However, this method struggles to handle tasks with different intents, limiting the model’s adaptability and generalization capabilities. To address this, researchers have introduced task instructions (Su et al., 2022) appended to queries, enabling a single embedding model to generalize across tasks in various domains by altering the instructions.
Despite these advances, studies such as Su et al. (2024); Weller et al. (2024) reveal that embedding models have a limited ability to follow unseen embedding task instructions and conduct complex retrieval tasks. This limitation arises from a gap between the limited diversity of instructions seen during training and the vast range of real-world scenarios. Inspired by the ability of LLMs to generalize to unseen tasks through in-context learning (ICL), we explore whether embedding models can be enhanced by leveraging ICL, thereby significantly improving their generalization and versatility across diverse embedding tasks with various user intents. In this work, we demonstrate the potential of embedding models to benefit from ICL through fewshot contrastive training. Consider a query-passage pair (qi, pi) in an embedding task. We first construct an example template as follows:
⟨Instruct⟩{task definition} ⟨query⟩{qi} ⟨response⟩{pi}
Here, ”task definition” represents the description of the specific embedding task. This example template is applied to new input queries for each embedding task (Figure 1). For a relevant querypassage pair (q+, p+), the modified query q+ exp is constructed as follows:
example 1} ... {example n} ⟨Instruct⟩{task definition} ⟨query⟩{q+}
All modified queries and passages in the corpus are encoded using the same LLM to obtain their embedding representations. Specifically, we append an [EOS] token to the end of the input modified queries and passages, feeding them into the LLM to obtain embeddings (hq+ exp, hp+) by extracting the final layer’s [EOS] vector. We employ the standard InfoNCE (Izacard et al., 2021) loss function L, utilizing both in-batch negatives and hard negatives for training:
  p− j denotes the set of negative passages, and s(q, p) is the scoring function between the query and passage. In this work, we adopt a temperature-scaled cosine similarity function defined as:
where τ is a temperature hyperparameter, which is fixed at 0.02 during tr
# 3.2 REPRESENTATION METHOD
The attention mechanism in LLM-based embedding models is typically unidirectional, aligned with the next-token prediction task fundamental to their pre-training (Touvron et al., 2023).
However, recent studies indicate that unidirectional attention may limit the model’s capacity for representation learning. Evidence suggests that bidirectional attention is more effective at integrating contextual information, resulting in improved performance on certain tasks. For example, LLM2Vec (BehnamGhader et al., 2024) introduces an additional training phase with a masked token prediction task, preconditioning the model for bidirectional attention. Approaches such as NV-Embed (Lee et al., 2024a) and GritLM (Muennighoff et al., 2024) replace unidirectional attention with bidirectional attention during the embedding training phase, often employing mean pooling or more sophisticated latent attention layers to obtain representations for queries and passages. Despite these advances, we argue that incorporating bidirectional attention during embedding finetuning creates a mismatch with the model’s pre-training design, potentially undermining its incontext learning and generative properties. To address the trade-off between enhancing embedding representations for specific tasks and preserving the model’s inherent generative properties for deep semantic pattern understanding, our approach retains the unidirectional attention mechanism, consistent with the majority of existing embedding methods. We use the [EOS] token’s output embedding as the vector representation for queries and passages, positioning it at the end of inputs to capture both semantic and ICL patterns through causal attention
(1)
(2)
(3)
(4)
# mechanisms, thereby aligning with the foundational pretraining methodology of LLMs. Specifically, given the tokenized input sequence T: [BOS], t1, ..., tN is sent into the LLM (Figure 1):
ht = LLM(T)[EOS]
The text embedding is taken from the output embedding of the special token [EOS].
3.3 ICL-BASED INSTRUCTION-TUNING
While previous works (Wang et al., 2023; Lee et al., 2024a) have proposed the training method of instruction-tuning, which incorporates a large number of task-specific instructions during the training process, enabling the model to adapt to various downstream retrieval tasks based on different instructions, it is not applicable to the ICL strategy. As demonstrated by GRIT (Muennighoff et al., 2024), directly supplying few-shot examples when generating embeddings can actually degrade model performance. To incorporate ICL capabilities into models, we need to modify the conventional instruction tuning strategy. Our approach involves integrating ICL abilities during the training phase. Specifically, we provide task-relevant examples to the query throughout the training process, allowing the model to develop ICL capabilities as it learns. Recognizing the risk of compromising zero-shot capabilities if examples are consistently provided during training, we propose a dynamic training process. In each training step, queries are supplied with a variable number of few-shot examples, ranging from zero to n, determined by a sampling function. This approach maintains a balance between developing ICL abilities and preserving zeroshot performance. To further enhance the model’s ICL capabilities, we introduce an innovative technique for examples selection. By incorporating in-batch pairs as few-shot examples, we train the model to better differentiate between examples and inputs, aims to improve the model’s ability to generate reliable embeddings based on the provided examples.
# 4 EXPERIMENTENS
his section, we examine the effectiveness of the ICL training pipeline and reconsider the training thodologies for LLM-based embedding models. • RQ 1: What is the effectiveness of the ICL training strategy for both zero-shot and few-shot learning scenarios? • RQ 2: How does the ICL training strategy impact performance compared to traditional training methods? • RQ 3: How does the integration of in-batch examples affect the performance of the ICL training strategy. • RQ 4: What are the implications of replacing a causal attention mask with a bidirectional attention mask within the framework of LLMs? • RQ 5: What is the impact of various representation strategies, including last token pooling and mean pooling, on model performance? • RQ 6: Do passage-based prompts enhance performance in the ICL training strategy?
# 4.1 SETUP
LLM. Following E5-Mistral (Wang et al., 2023), SFR, and NV-Embedder (Lee et al., 2024a), we have adopted Mistral-7B (Jiang et al., 2023) as the backbone for our framework. Evaluation. We evaluate the performance of our model on MTEB (Muennighoff et al., 2022) and AIR-Bench. MTEB is a comprehensive benchmark designed to evaluate the performance of text embedding models. AIR-Bench is dedicated to the evaluation of retrieval performance, its testing data is automatically generated by large language models without human intervention.
raining Data. To ensure a fair comparison, we use the same public datasets from E5-Mistral Wang et al., 2023), which includes ELI5 (Fan et al., 2019), HotpotQA (Yang et al., 2018), FEVER Thorne et al., 2018), MIRACL (Zhang et al., 2023), MSMARCO passage and document ranking Nguyen et al., 2016), NQ (Karpukhin et al., 2020), NLI (Gao et al., 2021), SQuAD (Karpukhin t al., 2020), TriviaQA (Karpukhin et al., 2020), Quora Duplicate Questions (DataCanary et al., 017), MrTyDi (Zhang et al., 2021), DuReader (Qiu et al., 2022), and T2Ranking (Xie et al., 2023), ll of which are also used for LLM2Vec (BehnamGhader et al., 2024). However, methods that typically perform exceptionally well, such as NV-Embedder (Lee et al., 024a) and SFR, often require more training data. Additionally, some of these methods, such as GTE-Qwen2 (Li et al., 2023), do not disclose their sources of training data. In response, we have eveloped an enhanced version of our model that leverages a more comprehensive dataset, which ncludes the following training sets: • Retrieval: ELI5, HotpotQA, FEVER, MSMARCO passage and document ranking, NQ, NLI, SQuAD, TriviaQA, Quora Duplicate Questions, Arguana (Wachsmuth et al., 2018), and FiQA (Maia et al., 2018). • Reranking: SciDocsRR (Cohan et al., 2020) and StackOverFlowDupQuestions (Liu et al., 2018). • Classification: AmazonReviews-Classification (McAuley & Leskovec, 2013), AmazonCounterfactual-Classification (O’Neill et al., 2021), Banking77Classification (Casanueva et al., 2020), Emotion-Classification (Saravia et al., 2018), TweetSentimentExtraction-Classification (Maggie, 2020), MTOPIntent-Classification (Li et al., 2020), IMDB-Classification (Maas et al., 2011), ToxicConversations-Classification (Adams et al., 2019). • Clustering: {Arxiv/Biorxiv/Medrxiv/Reddit/StackExchange}-Clustering-{S2S/P2P}, TwentyNewsgroups-Clustering (Lang, 1995). • STS: STS12 (Agirre et al., 2012), STS22 (Chen et al., 2022), STS-Benchmark (Cer et al., 2017).
Training Detail. We fine-tune the Mistral-7B model using a contrastive loss and conduct the process over a single epoch. For efficient fine-tuning, we employ Low-Rank Adaptation (LoRA) (Hu et al., 2021), setting the LoRA rank to 64 and the LoRA alpha to 32, with a learning rate of 1e-4. For retrieval tasks, we use in-batch negatives, a strategy not adopted for other tasks. Each dataset incorporates 7 hard negatives. The batch size is set to 512 for retrieval tasks and 256 for other types of tasks. We maintain consistency by using the same dataset throughout one training step, and the maximum sequence length is set at 512 tokens. To distill the score from reranker in retrieval tasks, we use the bge-reranker model as the teacher. For in-context learning training, we implement a randomized sampling method. For each query, we select between 0 to 5 examples from the in-batch training data. The maximum allowable lengths for example queries and documents are set to 256 tokens each, and the combined length for a query with examples is set at 2048 tokens. Evaluation. We evaluate the performance of our model under both zero-shot and few-shot conditions. In the few-shot scenario, a consistent set of in-context examples is applied to each query. The examples utilized for evaluation are sourced from training datasets. In cases where training datasets are unavailable, examples are generated using ChatGPT.
# 4.2 MAIN RESULTS
MTEB. Table 1 presents the performance of our model, bge-en-icl, evaluated on the MTEB benchmark. This evaluation contrasts the results obtained from using the full dataset with those obtained from using only the public dataset. When leveraging the full dataset, our model demonstrates strong capabilities in both zero-shot and few-shot settings, achieving SOTA results in few-shot scenarios. However, it is important to note that the use of full datasets may introduce inconsistencies, as different models often rely on varying datasets. Notably, many of these models do not disclose the specific datasets they use, leading to potential unfair comparisons. For a fairer comparison and to better understand the impact of in-context learning, we conducts an evaluation using only the public dataset. Under these constraints, our model’s performance in
Task
Retr.
Rerank.
Clust.
PairClass.
Class.
STS
Summ.
Avg.
# of datasets →
15
4
11
3
12
10
1
56
w/ full data
E5-mistral-7b-instruct
56.90
60.21
50.26
88.34
78.47
84.66
31.40
66.63
GritLM-7B
57.41
60.49
50.61
87.16
79.46
83.35
30.37
66.76
SFR-Embedding
59.00
60.64
51.67
88.54
78.33
85.05
31.16
67.56
Linq-Embed-Mistral
60.19
60.29
51.42
88.35
80.20
84.97
30.98
68.17
voyage-large-2-instruct
58.28
60.09
53.35
89.24
81.49
84.31
30.84
68.23
NV-Embed-v1
59.36
60.59
52.80
86.91
87.35
82.84
31.20
69.32
bge-multilingual-gemma2
59.24
59.72
54.65
85.84
88.08
83.88
31.20
69.88
stella en 400M v5
58.97
60.16
56.70
87.74
86.67
84.22
31.66
70.11
gte-Qwen2-7B-instruct
60.25
61.42
56.92
85.79
86.58
83.04
31.35
70.24
SFR-Embedding-2 R
60.18
60.14
56.17
88.07
89.05
81.26
30.71
70.31
stella en 1.5B v5
61.01
61.21
57.69
88.07
87.63
84.51
31.49
71.19
bge-en-icl (zero-shot)
61.67
59.66
57.51
86.93
88.62
83.74
30.75
71.24
bge-en-icl (few-shot)
62.16
59.82
57.89
88.14
88.95
84.24
30.77
71.67
w/ public data only
E5-mistral-7b-instruct
52.78
60.38
47.78
88.47
76.80
83.77
31.90
64.56
GritLM-7B
53.10
61.30
48.90
86.90
77.00
82.80
29.40
64.70
LLM2Vec-Mistral-supervised
55.99
58.42
45.54
87.99
76.63
84.09
29.96
64.80
bge-en-icl (zero-shot)
59.59
56.85
42.61
87.87
75.47
83.30
29.52
64.67
bge-en-icl (few-shot)
60.08
56.67
46.55
88.51
77.31
83.69
30.68
66.08
<div style="text-align: center;">Table 1: Top MTEB leaderboard models as of August 27, 2024.</div>
Domain
wiki
web
news
healthcare
law
finance
arxiv
msmarco
Avg.
# of datasets →
1
1
1
1
1
1
1
1
8
w/ full data
E5-mistral-7b-instruct
61.67
44.41
48.18
56.32
19.32
54.79
44.78
59.03
48.56
SFR-Embedding
63.46
51.27
52.21
58.76
23.27
56.94
47.75
58.99
51.58
NV-Embed-v1
62.84
50.42
51.46
58.53
20.65
49.89
46.10
60.27
50.02
Linq-Embed-Mistral
61.04
48.41
49.44
60.18
20.34
50.04
47.56
60.50
49.69
gte-Qwen2-7B-instruct
63.46
51.20
54.07
54.20
22.31
58.20
40.27
58.39
50.26
stella en 1.5B v5
61.99
50.88
53.87
58.81
23.22
57.26
44.81
61.38
51.53
bge-en-icl (zero-shot)
64.61
54.40
55.11
57.25
25.10
54.81
48.46
63.71
52.93
bge-en-icl (few-shot)
64.94
55.11
56.02
58.85
28.29
57.16
50.04
64.50
54.36
w/ public data only
bge-en-icl (zero-shot)
64.82
54.96
55.82
57.06
28.87
54.46
49.60
63.25
53.60
bge-en-icl (few-shot)
66.98
56.38
57.17
59.54
32.03
58.81
51.36
65.05
55.92
Table 2: QA (en, nDCG@10) performance on AIR-Bench 24.04.
the zero-shot scenario is on par with, or slightly below, that of other models such as LLM2Vec and GritLM. However, in the few-shot settings, our model show significant enhancements (↑1.41), particularly in the classification and clustering tasks that were not part of the training data. These improvements underscore the potential advantages of in-context learning, emphasizing its efficacy in adapting to tasks beyond the direct scope of initial training parameters. Furthermore, in contrast to training exclusively with public datasets, the utilization of full training data effectively familiarizes the model with these datasets. As a result, the model’s ability to generalize effectively is compromised, leading to only a modest improvement in few-shot settings (↑0.43). AIR-Bench. The performance of our model is also evaluated using the AIR-Bench dataset. As illustrated in Tables 2 and 3, the model demonstrates superior performance compared to prior models in both zero-shot and few-shot scenarios, excelling across qa and long-doc tasks. Notably, there is no overlap between the training dataset and the evaluation data for these tasks, highlighting the robustness of the model in scenarios with limited prior exposure. In the few-shot setting, the model exhibits significant improvements over the zero-shot scenario, achieving gains of 1.43 points in the qa task and 1.08 points in the long-doc task. This improvement underscores the efficacy of in-context learning in enhancing the model’s generalization capabilities.
Domain
arxiv
book
healthcare
law
Avg.
# of datasets →
4
2
5
4
15
w/ full data
text-embedding-3-large
74.53
73.16
65.83
64.47
68.77
E5-mistral-7b-instruct
72.14
72.44
68.44
62.92
68.49
SFR-Embedding
72.79
72.41
67.94
64.83
69.00
NV-Embed-v1
77.65
75.49
72.38
69.55
73.45
Linq-Embed-Mistral
75.46
73.81
71.58
68.58
72.11
gte-Qwen2-7B-instruct
63.93
68.51
65.59
65.26
65.45
stella en 1.5B v5
73.17
74.38
70.02
69.32
71.25
bge-multilingual-gemma2
71.77
76.46
73.96
70.86
72.88
bge-en-icl (zero-shot)
78.30
78.21
73.65
67.09
73.75
bge-en-icl (few-shot)
79.63
79.36
74.80
67.79
74.83
w/ public data only
bge-en-icl (zero-shot)
79.73
78.66
72.88
70.59
74.86
bge-en-icl (few-shot)
79.82
80.37
74.60
71.66
75.98
Table 3: Long-Doc (en, Recall@10) performance on AIR-Bench 24.04.
However, when the model is trained exclusively using public data, it achieves better results compared to training with the full dataset. This could be attributed to the presence of an excessive amount of MTEB-related data, such as clustering and classification, within the full dataset. Such data might introduce the risk of overfitting, thereby potentially hampering the model’s generalization performance on the AIR-Bench dataset.
# 4.3 IN-CONTEXT LEARNING
<div style="text-align: center;">4.3 IN-CONTEXT LEARNING</div>
Task
Retr.
Rerank.
Clust.
PairClass.
Class.
STS
Summ.
Avg.
# of datasets →
15
4
11
3
12
10
1
56
w/ full data
w/o in-context learning
59.11
57.02
42.60
87.99
76.27
83.93
30.50
64.83
w/ fix examples (zero-shot)
48.98
56.48
41.84
85.94
74.38
84.31
29.68
61.50
w/ fix examples (few-shot)
59.00
56.90
45.75
88.54
75.56
84.67
30.66
65.46
w/ in-batch examples (zero-shot)
59.59
56.85
42.61
87.87
75.47
83.30
29.52
64.67
w/ in-batch examples (few-shot)
60.08
56.67
46.55
88.51
77.31
83.69
30.68
66.08
To evaluate the impact of the ICL strategy, we conduct a series of ablation studies using the MTEB benchmark. In these studies, we compare the performance of models fine-tuned with the ICL strategy against those fine-tuned without it. Specifically, for ICL training, we employ two distinct training approaches: fixed examples and in-batch examples. In the fixed examples approach, each task was trained using three predetermined examples. In Table 4, we present various results from our experiment. When the model is trained without ICL strategy, its average performance is 64.83. This performance is comparable to GritLM (Muennighoff et al., 2024), LLM2Vec (BehnamGhader et al., 2024), etc. When fixed examples are used during ICL training, there is a significant decline in zero-shot evaluation performance, with a decrease of 3.33 points. This decline is attributed to the model’s consistent exposure to specific training examples, which can impair its zero-shot capabilities. On the other hand, in few-shot scenarios, the model demonstrates improved performance, exceeding zero-shot results by 3.96 points and surpassing models trained without ICL by 0.63 points. This also confirms the effectiveness of the ICL strategy. Meanwhile, the use of in-batch examples, where training may involve zero examples, has preserved the zero-shot capability of the model. There is a modest decrease of 0.16 points compared to the model trained without ICL. Notably, in few-shot scenarios, the performance of the model employing in-batch examples rises to 66.08 (↑1.25), indicating a robust improvement. Furthermore, when compared with the model utilizing fixed examples, the model trained with in-batch examples displays
superior performance in tasks that diverge significantly from the training data, such as classification and clustering tasks.
# 4.4 ATTENTION
<div style="text-align: center;">4.4 ATTENTION</div>
Task
Retr.
Rerank.
Clust.
PairClass.
Class.
STS
Summ.
Avg.
# of datasets →
15
4
11
3
12
10
1
56
causal attention & last token pooling
w/o in-context learning
59.11
57.02
42.60
87.99
76.27
83.93
30.50
64.83
w/ in-context learning (zero-shot)
59.59
56.85
42.61
87.87
75.47
83.30
29.52
64.67
w/ in-context learning (few-shot)
60.08
56.67
46.55
88.51
77.31
83.69
30.68
66.08
causal attention & mean pooling
w/o in-context learning
58.50
53.74
36.82
82.14
72.37
77.62
29.10
61.03
bidirectional attention & last token pooling
w/o in-context learning
59.59
56.96
44.34
87.61
74.77
83.81
30.12
64.96
w/ in-context learning (zero-shot)
59.77
58.09
44.04
87.87
75.35
83.97
29.75
65.19
w/ in-context learning (few-shot)
60.23
57.81
44.45
88.64
77.00
83.77
29.99
65.74
bidirectional attention & mean pooling
w/o in-context learning
59.13
57.03
43.44
87.25
75.03
84.08
29.17
64.73
w/ in-context learning (zero-shot)
59.53
57.48
43.88
88.12
74.86
83.64
29.58
64.90
w/ in-context learning (few-shot)
59.42
57.29
44.93
88.36
75.26
83.75
29.60
65.18
Recent studies have explored modifying causal attention in LLMs to adopt bidirectional attention and employ mean pooling for embedding generation. Notably, models such as GritLM (Muennighoff et al., 2024), NV-Embed (Lee et al., 2024a), and LLM2Vec (BehnamGhader et al., 2024) have utilized these techniques with considerable experimental success. Motivated by these advancements, we explore the potential benefits of implementing bidirectional attention in the ICL scenario. Specifically, we investigate the impacts of various attention and pooling mechanisms, including causal and bidirectional attention, coupled with last token pooling and mean pooling. In a causal attention framework, each token is limited to accessing only preceding tokens’ information and not the subsequent ones. Consequently, employing mean pooling tends to yield suboptimal results because of this restriction. We find that the model could not be trained effectively under the ICL setting. Therefore, only results from experiments without ICL are presented in this specific configuration. Table 5 presents our experimental setup and results in both non-ICL and ICL scenarios. It demonstrates that in non-ICL scenarios, most methods yield consistent performance, aside from the combination of causal attention with mean pooling. In contrast, within ICL scenarios, the integration of causal attention and last token pooling emerges as the superior approach. This configuration appears to resonate with the competencies fostered during the initial training phase of the model, suggesting a strong alignment with the foundational strategies employed during pre-training. Moreover, shifting from causal attention to bidirectional attention does not result in significant improvements, and mean pooling is not necessary for the implementation of bidirectional attention. Additionally, configurations utilizing bidirectional attention paired with last token pooling are notably effective, excelling in both non-ICL and zero-shot scenarios. This configuration’s performance is also pronounced in few-shot reranking tasks, highlighting its adaptability and potential applicability across various demands.
# 4.5 PROMPT
Recently, most LLM-based embedding models have incorporated instruction-based prompts on the query side. However, there has been limited investigation into the efficacy of utilizing prompts on the passage side. To address this gap, our study introduces and explores the use of prompts on the passage side. The specific prompt employed in our study is as follows:
# {passage} Summarize the above passage:
Table 6 presents the results obtained using passage prompts. The results demonstrate that the integration of passage prompts leads to a significant decline in performance across all tasks except
(6)
Task
Retr.
Rerank.
Clust.
PairClass.
Class.
STS
Summ.
Avg.
# of datasets →
15
4
11
3
12
10
1
56
w/o passage prompt (zero-shot)
59.59
56.85
42.61
87.87
75.47
83.30
29.52
64.67
w/o passage prompt (few-shot)
60.08
56.67
46.55
88.51
77.31
83.69
30.68
66.08
w/ passage prompt (zero-shot)
59.50
46.84
39.57
81.25
71.41
80.38
30.26
61.61
w/ passage prompt (few-shot)
59.93
46.39
39.40
82.25
72.00
79.81
30.97
61.74
retrieval. This indicates that further exploration and experimentation are needed when employing prompts at the passage level.
# 5 CONCLUSION
In this paper, we explore the utilization of in-context learning (ICL) derived from large language models (LLMs) for generating text embeddings and investigate various methods of LLMs as embedding models. Specifically, we examine the integration of attention mechanisms, different pooling methods, and passage prompts. We advocate for maintaining the model’s original architecture while embedding in-context learning capabilities into the dense retrieval process. Our approach necessitates no modifications to the model’s architecture; instead, it involves altering the prompt on the query side to include in-context learning features in the embedding generation task. Despite its simplicity, our method proves highly effective on the MTEB and AIR-Bench benchmarks.
# REFERENCES
C.J. Adams, Daniel Borkan, Jeffrey Sorensen, Lucas Dixon, Lucy Vasserman, and Nithum Thain. Jigsaw unintended bias in toxicity classification, 2019. URL https://kaggle.com/ competitions/jigsaw-unintended-bias-in-toxicity-classification. Eneko Agirre, Daniel Cer, Mona Diab, and Aitor Gonzalez-Agirre. Semeval-2012 task 6: A pilot on semantic textual similarity. in* sem 2012: The first joint conference on lexical and computational semantics–volume 1: Proceedings of the main conference and the shared task, and volume 2: Proceedings of the sixth international workshop on semantic evaluation (semeval 2012). Association for Computational Linguistics. URL http://www. aclweb. org/anthology/S12-1051, 2012. Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. Llm2vec: Large language models are secretly powerful text encoders. arXiv preprint arXiv:2404.05961, 2024. Tom B Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. I˜nigo Casanueva, Tadas Temˇcinas, Daniela Gerz, Matthew Henderson, and Ivan Vuli´c. Efficient intent detection with dual sentence encoders. arXiv preprint arXiv:2003.04807, 2020. Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation. arXiv preprint arXiv:1708.00055, 2017. Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 2024. Xi Chen, Ali Zeynali, Chico Q Camargo, Fabian Fl¨ock, Devin Gaffney, Przemyslaw A Grabowicz, Scott A Hale, David Jurgens, and Mattia Samory. Semeval-2022 task 8: Multilingual news article similarity. 2022. Mathieu Ciancone, Imene Kerboua, Marion Schaeffer, and Wissam Siblini. Mteb-french: Resources for french sentence embedding evaluation and analysis. arXiv preprint arXiv:2405.20468, 2024.
Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S Weld. Specter: Document-level representation learning using citation-informed transformers. arXiv preprint arXiv:2004.07180, 2020.
Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S Weld. Specter: Document-level representation learning using citation-informed transformers. arXiv preprint arXiv:2004.07180, 2020. Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzm´an, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 8440–8451, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.747. URL https://aclanthology.org/ 2020.acl-main.747. DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, and tomtung. Quora question pairs, 2017. URL https://kaggle.com/competitions/quora-question-pairs. Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu,
Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu and Zhifang Sui. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022.
bhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne, Onur C¸ elebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay
Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzm´an, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, V´ıtor Albiero, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118, 2021. Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. Vladimir Karpukhin, Barlas O˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906, 2020. Ken Lang. Newsweeder: Learning to filter netnews. In Machine learning proceedings 1995, pp. 331–339. Elsevier, 1995. Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nv-embed: Improved techniques for training llms as generalist embedding models. arXiv preprint arXiv:2405.17428, 2024a. Jinhyuk Lee, Zhuyun Dai, Xiaoqi Ren, Blair Chen, Daniel Cer, Jeremy R Cole, Kai Hui, Michael Boratko, Rajvi Kapadia, Wen Ding, et al. Gecko: Versatile text embeddings distilled from large language models. arXiv preprint arXiv:2403.20327, 2024b. Chaofan Li, Zheng Liu, Shitao Xiao, Yingxia Shao, and Defu Lian. Llama2vec: Unsupervised adaptation of large language models for dense retrieval. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3490– 3500, 2024. Haoran Li, Abhinav Arora, Shuohui Chen, Anchit Gupta, Sonal Gupta, and Yashar Mehdad. Mtop: A comprehensive multilingual task-oriented semantic parsing benchmark. arXiv preprint arXiv:2008.09335, 2020. Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281, 2023. Xueqing Liu, Chi Wang, Yue Leng, and ChengXiang Zhai. Linkso: a dataset for learning to retrieve similar question answer pairs on software development forums. In Proceedings of the 4th ACM SIGSOFT International Workshop on NLP for Software Engineering, pp. 2–5, 2018. Yinhan Liu. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019. Dingkun Long, Qiong Gao, Kuan Zou, Guangwei Xu, Pengjun Xie, Ruijie Guo, Jian Xu, Guanjun Jiang, Luxi Xing, and Ping Yang. Multi-cpr: A multi domain chinese dataset for passage retrieval. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’22, pp. 3046–3056, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450387323. doi: 10.1145/3477495.3531736. URL https: //doi.org/10.1145/3477495.3531736. Wenhao Lu, Jian Jiao, and Ruofei Zhang. Twinbert: Distilling knowledge to twin-structured compressed bert models for large-scale retrieval. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pp. 2645–2652, 2020. Kun Luo, Minghao Qin, Zheng Liu, Shitao Xiao, Jun Zhao, and Kang Liu. Large language models as foundations for next-gen dense retrieval: A comprehensive empirical assessment. arXiv preprint arXiv:2408.12194, 2024. Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. Fine-tuning llama for multi-stage text retrieval. arXiv preprint arXiv:2310.08319, 2023.
Andrew Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies, pp. 142–150, 2011. Wei Chen Maggie, Phil Culliton. Tweet sentiment extraction, 2020. URL https://kaggle. com/competitions/tweet-sentiment-extraction. Macedo Maia, Siegfried Handschuh, Andr´e Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. Www’18 open challenge: financial opinion mining and question answering. In Companion proceedings of the the web conference 2018, pp. 1941–1942, 2018. Julian McAuley and Jure Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. In Proceedings of the 7th ACM conference on Recommender systems, pp. 165–172, 2013. Sepideh Mollanorozy, Marc Tanti, and Malvina Nissim. Cross-lingual transfer learning with {P}ersian. In Lisa Beinborn, Koustava Goswami, Saliha Murado uglu, Alexey Sorokin, Ritesh Kumar, Andreas Shcherbakov, Edoardo M. Ponti, Ryan Cotterell, and Ekaterina Vylomova (eds.), Proceedings of the 5th Workshop on Research in Computational Linguistic Typology and Multilingual NLP, pp. 89–95, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.sigtyp-1.9. URL https://aclanthology.org/2023.sigtyp-1.9. Niklas Muennighoff, Nouamane Tazi, Lo¨ıc Magne, and Nils Reimers. Mteb: Massive text embedding benchmark. arXiv preprint arXiv:2210.07316, 2022. Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela. Generative representational instruction tuning. arXiv preprint arXiv:2402.09906, 2024. Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. Ms marco: A human-generated machine reading comprehension dataset. 2016. James O’Neill, Polina Rozenshtein, Ryuichi Kiryo, Motoko Kubota, and Danushka Bollegala. I wish i would have loved this one, but i didn’t–a multilingual dataset for counterfactual detection in product reviews. arXiv preprint arXiv:2104.06893, 2021. Rafał Po´swiata, Sławomir Dadas, and Michał Perełkiewicz. Pl-mteb: Polish massive text embedding benchmark. arXiv preprint arXiv:2405.10138, 2024. Yifu Qiu, Hongyu Li, Yingqi Qu, Ying Chen, Qiaoqiao She, Jing Liu, Hua Wu, and Haifeng Wang. Dureader retrieval: A large-scale chinese benchmark for passage retrieval from web search engine. arXiv preprint arXiv:2203.10232, 2022. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen. Carer: Contextualized affect representations for emotion recognition. In Proceedings of the 2018 conference on empirical methods in natural language processing, pp. 3687–3697, 2018. Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A Smith, Luke Zettlemoyer, and Tao Yu. One embedder, any task: Instruction-finetuned text embeddings. arXiv preprint arXiv:2212.09741, 2022. Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan Shi, Zachary S Siegel, Michael Tang, et al. Bright: A realistic and challenging benchmark for reasoning-intensive retrieval. arXiv preprint arXiv:2407.12883, 2024.
Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, L´eonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ram´e, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024. Nandan Thakur, Nils Reimers, Andreas R¨uckl´e, Abhishek Srivastava, and Iryna Gurevych. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663, 2021. James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. Fever: a large-scale dataset for fact extraction and verification. arXiv preprint arXiv:1803.05355, 2018. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. Henning Wachsmuth, Shahbaz Syed, and Benno Stein. Retrieval of the best counterargument without prior topic knowledge. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 241–251, 2018. Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improving text embeddings with large language models. arXiv preprint arXiv:2401.00368, 2023. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. Orion Weller, Benjamin Chang, Sean MacAvaney, Kyle Lo, Arman Cohan, Benjamin Van Durme, Dawn Lawrie, and Luca Soldaini. Followir: Evaluating and teaching information retrieval models to follow instructions. arXiv preprint arXiv:2403.15246, 2024. Shitao Xiao, Zheng Liu, Yingxia Shao, and Zhao Cao. Retromae: Pre-training retrieval-oriented language models via masked auto-encoder. arXiv preprint arXiv:2205.12035, 2022. Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-pack: Packed resources for general chinese embeddings. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 641–649, 2024. Xiaohui Xie, Qian Dong, Bingning Wang, Feiyang Lv, Ting Yao, Weinan Gan, Zhijing Wu, Xiangsheng Li, Haitao Li, Yiqun Liu, et al. T2ranking: A large-scale chinese benchmark for passage ranking. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 2681–2690, 2023. Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval. arXiv preprint arXiv:2007.00808, 2020. An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018. Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.
Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, L´eonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ram´e, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024. Nandan Thakur, Nils Reimers, Andreas R¨uckl´e, Abhishek Srivastava, and Iryna Gurevych. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663, 2021. James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. Fever: a large-scale dataset for fact extraction and verification. arXiv preprint arXiv:1803.05355, 2018. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. Henning Wachsmuth, Shahbaz Syed, and Benno Stein. Retrieval of the best counterargument without prior topic knowledge. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 241–251, 2018. Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improving text embeddings with large language models. arXiv preprint arXiv:2401.00368, 2023. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. Orion Weller, Benjamin Chang, Sean MacAvaney, Kyle Lo, Arman Cohan, Benjamin Van Durme, Dawn Lawrie, and Luca Soldaini. Followir: Evaluating and teaching information retrieval models to follow instructions. arXiv preprint arXiv:2403.15246, 2024. Shitao Xiao, Zheng Liu, Yingxia Shao, and Zhao Cao. Retromae: Pre-training retrieval-oriented language models via masked auto-encoder. arXiv preprint arXiv:2205.12035, 2022. Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-pack: Packed resources for general chinese embeddings. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 641–649, 2024. Xiaohui Xie, Qian Dong, Bingning Wang, Feiyang Lv, Ting Yao, Weinan Gan, Zhijing Wu, Xiangsheng Li, Haitao Li, Yiqun Liu, et al. T2ranking: A large-scale chinese benchmark for passage ranking. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 2681–2690, 2023. Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval. arXiv preprint arXiv:2007.00808, 2020. An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018. Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629,
Li Yudong, Zhang Yuqing, Zhao Zhe, Shen Linlin, Liu Weijie, Mao Weiquan, and Zhang Hui. Csl: A large-scale chinese scientific literature dataset. arXiv preprint arXiv:2209.05034, 2022. Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li, and Min Zhang. mgte: Generalized long-context text representation and reranking models for multilingual text retrieval, 2024. Xinyu Zhang, Xueguang Ma, Peng Shi, and Jimmy Lin. Mr. tydi: A multi-lingual benchmark for dense retrieval. arXiv preprint arXiv:2108.08787, 2021. Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. Miracl: A multilingual retrieval dataset covering 18 diverse languages. Transactions of the Association for Computational Linguistics, 11:1114–1131, 2023. Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, and Guido Zuccon. Promptreps: Prompting large language models to generate dense and sparse representations for zero-shot document retrieval. arXiv preprint arXiv:2404.18424, 2024.
# A INSTRUCTION
Task Name
Instruction Template
ArguAna
Given a claim, find documents that refute the claim.
ClimateFEVER
Given a claim about climate change, retrieve documents that support or refute
the claim.
CQADupStack
Given a question, retrieve detailed question descriptions from Stackexchange that are
duplicates to the given question.
DBPedia
Given a query, retrieve relevant entity descriptions from DBPedia.
FEVER
Given a claim, retrieve documents that support or refute the claim.
FiQA2018
Given a financial question, retrieve user replies that best answer the question.
HotpotQA
Given a multi-hop question, retrieve documents that can help answer the question.
MSMARCO
Given a web search query, retrieve relevant passages that answer the query.
NFCorpus
Given a question, retrieve relevant documents that best answer the question.
Natural Question
Given a question, retrieve Wikipedia passages that answer the question.
QuoraRetrieval
Given a question, retrieve questions that are semantically equivalent to the given
question.
SCIDOCS
Given a scientific paper title, retrieve paper abstracts that are cited by the given paper.
SciFact
Given a scientific claim, retrieve documents that support or refute the claim.
Touche2020
Given a question, retrieve detailed and persuasive arguments that answer the question.
TREC-COVID
Given a query, retrieve documents that answer the query.
STS*
Retrieve semantically similar text.
SummEval
Given a news summary, retrieve other semantically similar summaries.
AmazonCounterfactualClassification
Classify a given Amazon customer review text as either counterfactual
or not-counterfactual.
AmazonPolarityClassification
Classify Amazon reviews into positive or negative sentiment.
AmazonReviewsClassification
Classify the given Amazon review into its appropriate rating category.
Banking77Classification
Given a online banking query, find the corresponding intents.
EmotionClassification
Classify the emotion expressed in the given Twitter message into one of the six
emotions: anger, fear, joy, love, sadness, and surprise.
ImdbClassification
Classify the sentiment expressed in the given movie review text from
the IMDB dataset.
MassiveIntentClassification
Given a user utterance as query, find the user intents.
MassiveScenarioClassification
Given a user utterance as query, find the user scenarios.
MTOPDomainClassification
Classify the intent domain of the given utterance in task-oriented conversation.
MTOPIntentClassification
Classify the intent of the given utterance in task-oriented conversation.
ToxicConversationsClassification
Classify the given comments as either toxic or not toxic.
TweetSentimentExtractionClassification
Classify the sentiment of a given tweet as either positive, negative, or neutral.
ArxivClusteringP2P
Identify the main and secondary category of Arxiv papers based on the titles
and abstracts.
ArxivClusteringS2S
Identify the main and secondary category of Arxiv papers based on the titles.
BiorxivClusteringP2P
Identify the main category of Biorxiv papers based on the titles and abstracts.
BiorxivClusteringS2S
Identify the main category of Biorxiv papers based on the titles.
MedrxivClusteringP2P
Identify the main category of Medrxiv papers based on the titles and abstracts.
MedrxivClusteringS2S
Identify the main category of Medrxiv papers based on the titles.
RedditClustering
Identify the topic or theme of Reddit posts based on the titles.
RedditClusteringP2P
Identify the topic or theme of Reddit posts based on the titles and posts.
StackExchangeClustering
Identify the topic or theme of StackExchange posts based on the titles.
StackExchangeClusteringP2P
Identify the topic or theme of StackExchange posts based on the given paragraphs.
TwentyNewsgroupsClustering
Identify the topic or theme of the given news articles.
AskUbuntuDupQuestions
Retrieve duplicate questions from AskUbuntu forum.
MindSmallReranking
Retrieve relevant news articles based on user browsing history.
SciDocsRR
Given a title of a scientific paper, retrieve the titles of other relevant papers.
StackOverflowDupQuestions
Retrieve duplicate questions from StackOverflow forum.
SprintDuplicateQuestions
Retrieve duplicate questions from Sprint forum.
TwitterSemEval2015
Retrieve tweets that are semantically similar to the given tweet.
TwitterURLCorpus
Retrieve tweets that are semantically similar to the given tweet.
AIR-Bench
Given a question, retrieve passages that answer the question.
Table 7: The instruction we used on the MTEB and AIR-Bench benchmarks.
Table 7: The instruction we used on the MTEB and AIR-Bench benchmarks.
Dataset
NV-Em
bed-v1
bge-multilin
gual-gemma2
gte-Qwen2-
7B-instruct
SFR-Embe
dding-2 R
stella en
1.5B v5
bge-en-icl
(zero-shot)
bge-en-icl
(few-shot)
ArguAna
68.21
77.37
64.27
62.34
65.27
82.76
83.08
ClimateFEVER
34.72
39.37
45.88
34.43
46.11
45.35
45.43
CQADupStack
50.51
47.94
46.43
46.11
47.75
47.23
47.31
DBPEDIA
48.29
51.37
52.42
51.21
52.28
50.42
51.63
FEVER
87.77
90.38
95.11
92.16
94.83
91.96
92.83
FiQA2018
63.10
60.04
62.03
61.77
60.48
58.77
59.67
HotpotQA
79.92
83.26
73.08
81.36
76.67
84.98
85.14
MSMARCO
46.49
45.71
45.98
42.18
45.22
46.72
46.79
NFCorpus
38.04
38.11
40.60
41.34
42.00
40.69
41.85
Natural Question
71.22
71.45
67.00
73.96
71.80
73.85
73.88
QuoraRetrieval
89.21
90.04
90.09
89.58
90.03
91.02
90.95
SCIDOCS
20.19
26.93
28.91
24.87
26.64
25.25
25.26
SciFact
78.43
72.05
79.06
85.91
80.09
78.33
79.09
Touche2020
28.38
30.26
30.57
28.18
29.94
29.67
30.48
TREC-COVID
85.88
64.27
82.26
87.28
85.98
78.11
79.08
BIOSSES
85.59
85.74
81.37
87.60
83.11
86.35
86.47
SICK-R
82.80
82.66
79.28
77.01
82.89
83.87
83.87
STS12
76.22
77.71
79.55
75.67
80.09
77.73
78.14
STS13
86.30
87.45
88.83
82.40
89.68
85.98
86.59
STS14
82.09
83.48
83.87
79.93
85.07
82.34
82.83
STS15
87.24
87.63
88.54
85.82
89.39
87.35
87.77
STS16
84.77
86.70
86.49
84.50
87.15
86.54
87.04
STS17
87.42
91.18
88.73
88.93
91.35
91.25
91.25
STS22
69.85
69.02
66.88
67.10
68.10
68.08
70.07
STSBenchmark
86.14
87.25
86.85
83.60
88.23
87.92
88.42
SummEval
31.20
31.20
31.35
30.71
31.49
30.75
30.77
SprintDuplicateQuestions
95.94
90.94
92.82
97.62
96.04
95.06
97.23
TwitterSemEval2015
78.73
79.64
77.96
78.57
80.58
78.54
79.34
TwitterURLCorpus
86.05
86.95
86.59
88.03
87.58
87.19
87.84
AmazonCounterfactual
95.12
89.48
91.31
92.72
92.87
92.88
93.15
AmazonPolarity
97.14
96.90
97.50
97.31
97.16
96.86
96.98
AmazonReviews
55.47
61.60
62.56
61.04
59.36
61.28
61.46
Banking77
90.34
92.53
87.57
90.02
89.79
91.42
91.49
Emotion
91.71
92.97
79.45
93.37
84.29
93.31
93.36
Imdb
97.06
96.66
96.75
96.80
96.66
96.91
96.91
MassiveIntent
80.07
82.05
85.41
85.97
85.83
82.26
82.93
MassiveScenario
81.74
84.40
89.77
90.61
90.20
83.92
85.60
MTOPDomain
96.51
98.61
99.04
98.58
99.01
97.99
98.42
MTOPIntent
89.77
95.51
91.88
91.30
92.78
93.56
94