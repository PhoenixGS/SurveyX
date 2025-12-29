# Comparative Analysis of Demonstration Selection Algorithms for LLM In-Context Learning
# Comparative Analysis of Demonstration Selection Algorithms for LLM In-Context Learning Dong Shu1, Mengnan Du2
1Northwestern University 2New Jersey Institute of Technology dongshu2024@u.northwestern.edu, mengnan.du@njit.edu
Abstract
# Abstract
In-context learning can help Large Language Models (LLMs) to adapt new tasks without additional training. However, this performance heavily depends on the quality of the demonstrations, driving research into effective demonstration selection algorithms to optimize this process. These algorithms assist users in selecting the best k input-label pairs (demonstration examples) based on a given test input, enabling LLMs to in-context learn the relationship between the provided examples and the test inputs. Despite all the proposed demonstration selection algorithms, their efficiency and effectiveness remain unclear. This lack of clarity make it difficult to apply these algorithms in real-world scenarios and poses challenges for future research aimed at developing improved methods. This paper revisits six proposed algorithms, evaluating them on five datasets from both efficiency and effectiveness perspectives. Our experiments reveal significant variations in algorithm performance across different tasks, with some methods struggling to outperform random selection in certain scenarios. We also find that increasing the number of demonstrations does not always lead to better performance, and that there are often trade-offs between accuracy and computational efficiency. Our code is available at https: //github.com/Tizzzzy/Demonstration Selection Overview.
arXiv:2410.23099v
# Introduction
Large Language Models (LLMs) have achieved state-of-theart performance across a wide range of natural language processing tasks (Achiam et al. 2023; Dubey et al. 2024; AnthropicAI 2023). One of the key factors contributing to this success is their capability for in-context learning, which allows these models to adapt to new tasks without additional training (Xie et al. 2021). However, their performance is highly sensitive to the quality of the provided demonstrations. Recently, various demonstration selection algorithms have been developed to enhance this quality by selecting the most informative and relevant examples from the data pool. These algorithms have significantly reduced the time required for LLMs to address unseen tasks and have greatly improved their overall performance. Despite the success of these approaches, the effectiveness of selected examples and the efficiency of the selection and inference processes are not well understood. This Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.
Demonstration
Demonstration 2
Demonstration 1
Positive
Negative
Neutral
Input Question
_____
LLM
Positive
Prediction
Data Pool
...
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9a6a/9a6aaefa-a584-4065-84c8-67b6183637fe.png" style="width: 50%;"></div>
Figure 1: An overview of demonstration selection algorithms: These algorithms select demonstrations from the data pool, which the LLMs then use to generate answers.
lack of understanding makes it challenging for future research to identify areas for improvement and makes it difficult to use these algorithms in real-life situations. In this paper, we present a comparative analysis of six prominent demonstration selection algorithms, evaluating them on five diverse datasets from both efficiency and effectiveness perspectives. Our study aims to provide valuable insights into the strengths and limitations of current approaches, serving as a benchmark for future research and guiding practitioners in selecting appropriate demonstration selection methods for their specific use cases. Our comparative experiments yield the following key findings: • Contrary to expectations, not all demonstration selection algorithms consistently outperform random selection. Some algorithms struggle to surpass the random selection in certain scenarios. • We observe substantial accuracy gaps between different algorithms on the same dataset with the same number of demonstrations. For instance, in the MRPC dataset, this accuracy difference can be as high as 45% when comparing CBDS with RD-direct. • Our analysis reveals that increasing the number of demonstrations does not always lead to better performance. The relationship between the number of demonstrations and accuracy is not monotonic and varies significantly across different tasks and algorithms. • We find that while some algorithms like CBDS and UPRISE achieve high accuracy, they do so at the cost of poor efficiency, requiring over 5 and 3 seconds respectively per sample for demonstration selection. This trade-off poses challenges for real-world applications where quick response times are crucial.
lack of understanding makes it challenging for future research to identify areas for improvement and makes it difficult to use these algorithms in real-life situations. In this paper, we present a comparative analysis of six prominent demonstration selection algorithms, evaluating them on five diverse datasets from both efficiency and effectiveness perspectives. Our study aims to provide valuable insights into the strengths and limitations of current approaches, serving as a benchmark for future research and guiding practitioners in selecting appropriate demonstration selection methods for their specific use cases. Our comparative experiments yield the following key findings:
# Demonstration Selection Algorithms
In this study, we compare six demonstration selection algorithms for in-context learning: Concept Based Demonstration Selection (CBDS), Rethinking Demonstrations direct (RD-direct) and channel (RD-channel), LLM Retriever, UPRISE, and OpenICL TopK. These algorithms employ various strategies, from leveraging latent concepts to using retrieval models, to select the best examples for LLMs. Additionally, we include OpenICL Random as a baseline, which randomly selects examples from the data pool. Here are more details of the comparing algorithms: • Concept Based Demonstration Selection (CBDS) (Wang et al. 2024b): This algorithm proposes a Bayesian approach to select demonstration examples for in-context learning in LLMs. The approach involves two main steps: (1) The optimal value of the latent concept variable θ is learned as a set of new token embeddings using a small LLM. This step aims to align the latent concept variable with the token embedding space of the LLM. (2) After learning the optimal latent concept, the algorithm selects demonstrations that maximize the likelihood of inferring the optimal latent variable for the task at hand. These selected demonstrations can then be used with larger LLMs to improve performance. The selection process is mathematically grounded in the following equation:
In this study, we compare six demonstration selection algorithms for in-context learning: Concept Based Demonstration Selection (CBDS), Rethinking Demonstrations direct (RD-direct) and channel (RD-channel), LLM Retriever, UPRISE, and OpenICL TopK. These algorithms employ various strategies, from leveraging latent concepts to using retrieval models, to select the best examples for LLMs. Additionally, we include OpenICL Random as a baseline, which randomly selects examples from the data pool. Here are more details of the comparing algorithms:
 Concept Based Demonstration Selection (CBDS) (Wang et al. 2024b): This algorithm proposes a Bayesian approach to select demonstration examples for in-context learning in LLMs. The approach involves two main steps: (1) The optimal value of the latent concept variable θ is learned as a set of new token embeddings using a small LLM. This step aims to align the latent concept variable with the token embedding space of the LLM. (2) After learning the optimal latent concept, the algorithm selects demonstrations that maximize the likelihood of inferring the optimal latent variable for the task at hand. These selected demonstrations can then be used with larger LLMs to improve performance. The selection process is mathematically grounded in the following equation:
(1)
 Rethinking Demonstrations (Min et al. 2022): This approach examines the factors that contribute to the success of in-context learning in LLMs. It explores the impact of different aspects of demonstrations, particularly focusing on the distribution of the input text, the label space, and the overall format. The key findings are encapsulated in the following:
P(y | x1, y1, . . . , xk, yk, x) ≈format + label space + input distribution (2
(2)
This formula reflects the idea that in-context learning performance is driven by the structure and content of the demonstrations, rather than the accuracy of the label pairings.  LLM Retriever (Wang, Yang, and Wei 2023): This framework focuses on selecting high-quality in-context examples to enhance the performance of LLMs. The key approach involves an iterative process where an initial set of example candidates is retrieved using an unsupervised method like BM25 (Robertson, Zaragoza et al. 2009). These candidates are then ranked based on the conditional log probabilities of the ground truth outputs pro-
This formula reflects the idea that in-context learning performance is driven by the structure and content of the demonstrations, rather than the accuracy of the label pair-
 LLM Retriever (Wang, Yang, and Wei 2023): This framework focuses on selecting high-quality in-context examples to enhance the performance of LLMs. The key approach involves an iterative process where an initial set of example candidates is retrieved using an unsupervised method like BM25 (Robertson, Zaragoza et al. 2009). These candidates are then ranked based on the conditional log probabilities of the ground truth outputs pro-
vided by the LLM. The ranking is formalized in the following equation:
(3)
 | ∀ ∈{} where p(y | x, xi, yi) represents the conditional probability of the output y given the input x and the i-th candidate example (xi, yi). A reward model, based on a cross-encoder architecture, is then trained to distill these ranking preferences into a dense retriever. This retriever is further refined iteratively by leveraging the feedback from the LLM, ultimately improving the selection of incontext examples.  UPRISE (Cheng et al. 2023): UPRISE enhances the performance of LLMs in zero-shot settings by retrieving and utilizing effective prompts. The approach involves two main steps: first, the retriever retrieves a set of positive prompts P + from a pre-constructed pool P based on a given task input x, as formulated in the equation:
(4)
where R(x, P) is the retrieval function. Then, these retrieved prompts are concatenated with the input and fed into a frozen LLM to generate the output y:
(5)
 | ⊕ This approach allows for cross-task and cross-model generalization, meaning the retriever is trained on diverse tasks with a smaller LLM but can be applied to larger LLMs and unseen tasks during inference.  OpenICL (Wu et al. 2023): It is designed to facilitate ICL research and improve the evaluation of LLMs. The core approach for selecting demonstration examples involves retrieving examples based on various methods like TopK, VoteK, and BM25, which are then used to construct the context for the LLM’s inference. The retrieval of examples can be formalized as:
 ∈ where R(X, Y ) represents the retrieval function applied to the training data X and Y . The selected examples are then concatenated with the test input to form a single sequence, which is fed into the LLM to generate predictions. In this paper, we adopt the TopK and Random retrieval strategies. Notice that the RD algorithm supports both direct and channel approaches for demonstrating in-context examples, denoted as E. As the Figure 2 shows, for each example ei in E, it is paired with a input x and a label y. In the direct approach, ei is structured with x presented first, followed by y. Conversely, in the channel approach, ei is structured with y presented first, followed by x. It is also important to note that we used “demonstrations with gold labels” for both the direct and channel approaches. As for other algorithms, we use the direct approach. By evaluating these diverse approaches, we aim to provide a comprehensive analysis of current demonstration selection methods and their impact on LLM performance.
<div style="text-align: center;">Table 1: Datasets Statistics</div>
DATASETS
MRPC
QNLI
SST2
CMSQA
SWAG
Task
Classification
Classification
Classification
Multi-choice
Multi-choice
# Training Set
3,670
104,743
67,349
9,741
73,546
# Validation Set
408
5,463
872
1,221
20,000
# Test Set
1,730
5,463
1,821
1,284
20,000
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/652c/652cf329-9356-42c8-adc6-16bad0ddd80e.png" style="width: 50%;"></div>
Figure 2: An visual understanding the difference between direct and channel approach
# Experiments
In this section, we present a comprehensive evaluation of the six demonstration selection algorithms and a baseline random selection method.
# Experimental Settings
Datasets: The demonstration selection algorithms will be evaluated on 5 datasets, categorized into two groups: classification and multi-choice. The classification datasets include GLUE-MRPC, GLUE-QNLI, and GLUE-SST2. The multichoice datasets are CMSQA and SWAG. Their statistic is shown in Table 1, and below are more details of the datasets: • GLUE-MRPC (Wang et al. 2018): MRPC is a dataset consisting of sentence pairs, each annotated with a binary label indicating whether the sentences in the pair are semantically equivalent. • GLUE-QNLI: QNLI dataset is derived from the Stanford Question Answering Dataset (SQuAD). It is a binary classification task where the goal is to determine whether the context sentence contains the answer to the question. • GLUE-SST2: SST-2 is a binary sentiment classification dataset that includes movie review excerpts, where the task is to predict whether the sentiment of the review is positive or negative. • CMSQA (Talmor et al. 2018): The CommonsenseQA dataset consists of multiple-choice questions that require commonsense knowledge. Each question is paired with five answer choices, where only one is correct. • SWAG (Zellers et al. 2018): SWAG dataset is a largescale benchmark for grounded commonsense inference.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/493a/493a1a0c-e89c-497c-bfed-84f0d0a5a0d0.png" style="width: 50%;"></div>
Figure 3: The effectiveness of the algorithms on MRPC dataset
<div style="text-align: center;">Figure 3: The effectiveness of the algorithms on MRPC dataset</div>
It contains multiple-choice questions about grounded situations, where the task is to select the most plausible continuation of a given scenario.
Metrics: Given that all datasets are either classification or multi-choice, accuracy is an appropriate metric for evaluating the performance. To show the algorithm’s computational efficiency, we present the results in seconds.
Implementation Details: All of the demonstration selection algorithms were tested on LLaMa3-8B (Touvron et al. 2023) to evaluate their effectiveness. As shown in Figure 1, each algorithm first selects k demonstration examples from the available data pool based on the input question, which serve as in-context learning exemplars. These exemplars, along with the input question, are then fed into LLaMa3 for evaluation. We measured the absolute time each algorithm took to select k demonstration examples per single input, as well as the model inference time for the same input. To ensure fairness, we excluded all offline computational time, such as pre-training or other preliminary computations, for all algorithms. The values of k considered in this study were {4, 8, 10, 20}. All experiments were conducted using an NVIDIA RTX A6000 GPU, with the LLaMa3-8B model directly loaded from Huggingface, and we used the default sampling parameters such as temperature and top p.
# Main Results
Table 2 presents the effectiveness of each demonstration selection algorithm across different datasets and varying numbers of k in-context examples. The highest accuracy in each column is highlighted in bold format. We evaluated the computational efficiency of the six algorithms and a baseline,
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/cbd0/cbd0c1fd-33d3-4e1d-a232-cd1e37895545.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: An visualize trend of the algorithms effectiveness</div>
<div style="text-align: center;">Table 2: Effectiveness of the Demonstration Selection Algorithms</div>
Datasets
MRPC
QNLI
CMSQA
SWAG
SST2
k=4 k=8 k=10 k=20 k=4 k=8 k=10 k=20 k=4 k=8 k=10 k=20 k=4 k=8 k=10 k=20 k=4 k=8 k=10 k=20
CBDS
62.3 80.8 78.8
57.3 51.1 50.7 54.4
54.8 29.2 29.4 34.4
43.2 36.8 38.1 34.6
36.9 91.4 93.9 93.5
73.2
RD-direct
40.6 40.6 34.2
34.0 33.6 35.1 35.9
35.5 42.3 42.8 43.1
41.2 56.2 56.6 57.3
56.4 91.9 75.7 94.6
83.8
RD-channel
44.0 45.1 45.3
42.9 46.7 47.8 44.9
38.8 53.6 53.4 56.6
48.8 53.2 52.0 51.8
57.8 91.2 87.9 79.3
76.4
LLM Retriever
74.7 75.4 75.9
76.2 71.5 72.3 72.9
72.7 50.6 52.2 52.8
53.8 65.1 64.7 64.9
64.7 93.7 93.8 93.6
93.5
UPRISE
75.2 74.7 75.9
76.0 81.8 81.9 81.8
81.1 61.4 63.2 64.4
66.2 69.0 66.8 69.9
61.9 93.4 92.5 92.4
91.3
OpenICL TopK
64.1 64.2 65.0
65.2 78.1 74.7 70.8
69.9 34.2 37.6 36.7
32.9 35.4 36.7 37.1
34.9 91.2 92.1 92.3
94.5
Random Selection 59.0 62.6 60.1
62.3 51.8 48.3 47.9
47.4 26.7 28.1 28.1
28.5 29.6 31.8 31.4
31.1 86.2 89.5 91.3
91.7
and report the results in Table 3. We will show our findings from two perspectives: effectiveness and efficiency.
Effectiveness Perspective: Contrary to expectations, not all demonstration selection algorithms consistently outperform random selection. While some algorithms show significant improvements, others struggle to surpass the baseline set by random selection in certain scenarios. For instance, as shown in Table 2, we observed that for the MRPC dataset, both the RD-direct and RD-channel algorithms demonstrate limited effectiveness compared to random selection. This finding challenges the assumption that sophisticated selection methods always yield better results and highlights the importance of thorough evaluation before implementation. Another notable observation arises when we plot the effectiveness of each algorithm on the MRPC dataset, as shown in Figure 3. There is a substantial accuracy gap between different algorithms on the same dataset with the same k value. In the MRPC dataset, this accuracy difference can be as high as 45%, as seen when comparing CBDS with k = 10 to RDdirect with k = 10. A closer examination of the SST2 dataset in Table 2 reveals that for simpler classification tasks, the effectiveness of demonstration selection algorithms is often limited and does not significantly improve accuracy compared to more challenging multi-choice tasks like CommonsenseQA and SWAG. Furthermore, the RD algorithm provides both direct and channel approaches for presenting demonstration examples to the model. In our comparisons, we used “demonstrations with gold labels” for both the direct and channel approaches. We found that the channel approach generally outperforms the direct approach. This observation aligns with the results presented in Figures 8-10 of the original paper (Min et al. 2022). A possible explanation for this is that the channel ap-
proach better aligns the input features with the class labels, making it easier for the model to capture and understand the underlying relationships. Our analysis also reveals that increasing the number of demonstrations (k) does not always lead to better performance. The relationship between k and accuracy is not monotonic and varies significantly across tasks and algorithms. In Figure 4, we visualize the effectiveness trends of six algorithms on the CMSQA dataset. We observed two primary patterns: (1) as shown on the left, performance peaks at a certain k before declining, and (2) on the right, accuracy continues to improve as k increases. For the pattern on the left, we hypothesize that for some tasks, an excessive number of in-context examples may introduce conflicting knowledge, either with the model’s internal knowledge or among the examples themselves. Additionally, some examples might be irrelevant to the input question and may act as noise. This finding underscores the importance of selecting an appropriate k value tailored to the specific task. Thus, it is crucial to tune the number of demonstrations for each specific task and algorithm combination. Efficiency Perspective: We also evaluated the computational efficiency of the six algorithms and a baseline. As shown in Table 3, we measured efficiency in two ways: (1) the absolute demonstration selection time (selection) required for an algorithm to select k in-context examples for a single input question, and (2) the absolute inference time (inference) required for the model to generate an answer for the same input question. All efficiency experiments were conducted with k = 10. Among the compared algorithms, CBDS and UPRISE stand out for their high accuracy but also for their poor efficiency. On average, these algorithms require over 5 seconds and 3 seconds respectively per sample for demonstra-
proach better aligns the input features with the class labels, making it easier for the model to capture and understand the underlying relationships. Our analysis also reveals that increasing the number of demonstrations (k) does not always lead to better performance. The relationship between k and accuracy is not monotonic and varies significantly across tasks and algorithms. In Figure 4, we visualize the effectiveness trends of six algorithms on the CMSQA dataset. We observed two primary patterns: (1) as shown on the left, performance peaks at a certain k before declining, and (2) on the right, accuracy continues to improve as k increases. For the pattern on the left, we hypothesize that for some tasks, an excessive number of in-context examples may introduce conflicting knowledge, either with the model’s internal knowledge or among the examples themselves. Additionally, some examples might be irrelevant to the input question and may act as noise. This finding underscores the importance of selecting an appropriate k value tailored to the specific task. Thus, it is crucial to tune the number of demonstrations for each specific task and algorithm combination.
Efficiency Perspective: We also evaluated the computational efficiency of the six algorithms and a baseline. As shown in Table 3, we measured efficiency in two ways: (1) the absolute demonstration selection time (selection) required for an algorithm to select k in-context examples for a single input question, and (2) the absolute inference time (inference) required for the model to generate an answer for the same input question. All efficiency experiments were conducted with k = 10. Among the compared algorithms, CBDS and UPRISE stand out for their high accuracy but also for their poor efficiency. On average, these algorithms require over 5 seconds and 3 seconds respectively per sample for demonstra-
<div style="text-align: center;">Table 3: Efficiency of the Demonstration Selection Algorithms</div>
Datasets
MRPC
QNLI
CMSQA
SWAG
SST2
selection
inference
selection
inference
selection
inference
selection
inference
selection
inference
CBDS
5.44
1.82
5.35
1.80
5.43
4.58
5.45
3.63
5.44
1.82
RD-direct
2.34 ∗10−3
1.80
2.37 ∗10−3
1.81
2.44 ∗10−3
4.56
2.50 ∗10−3
3.63
1.44 ∗10−3
1.78
RD-channel
2.24 ∗10−3
1.80
2.50 ∗10−3
1.79
2.49 ∗10−3
4.55
2.46 ∗10−3
3.62
1.44 ∗10−3
1.77
LLM Retriever
8.32 ∗10−1
12.81
6.74 ∗10−1
12.84
8.47 ∗10−1
12.87
8.83 ∗10−1
11.85
8.16 ∗10−1
12.62
UPRISE
3.06
3.67
3.23
3.66
3.25
3.69
3.09
3.68
3.62
3.43
OpenICL TopK
1.46 ∗10−1
1.20
2.22 ∗10−1
1.19
1.48 ∗10−1
2.94
2.60 ∗10−1
2.61
4.79 ∗10−1
0.81
OpenICL Random
5.24 ∗10−6
1.11
5.63 ∗10−6
1.10
5.34 ∗10−6
3.04
6.48 ∗10−6
2.61
8.11 ∗10−6
0.61
tion selection. Referring to Table 2, out of 20 combinations (4 k values × 5 datasets), CBDS or UPRISE achieved the best accuracy in 14 cases. However, this superior accuracy comes at the cost of extremely high computational complexity. The low efficiency of these algorithms makes them challenging to apply in real-world applications where quick response times are crucial. We also found out that five algorithms are relatively efficient, taking less than one second per input question to select 10 examples. Notably, the OpenICL Random algorithm, along with both the RD-direct and RD-channel approaches, demonstrated particularly fast demonstration selection times. However, when cross-referencing these results with Table 2, it becomes clear that this speed often comes at a cost to accuracy. These faster algorithms generally do not perform as effectively as others. On the other hand, the LLM Retriever achieves a more balanced trade-off, offering reasonable demonstration selection speed while maintaining good overall effectiveness. We also observed variations in inference times across different algorithms. As shown in the ‘inference’ column, the CBDS and RD algorithms exhibit similar inference times, as both utilize model inference code from the MetaICL codebase (Min et al. 2021). An interesting finding emerges when comparing inference times for different datasets within the same algorithm: for most algorithms, including CBDS, RD, and OpenICL, inference times increase for more complex tasks, such as CMSQA and SWAG, compared to simpler binary classification tasks. As for the LLM Retriever, although it demonstrates fast demonstration selection times, the overall computational time remains slower when factoring in inference time. This slowdown is likely due to the constrained generation technique employed by the algorithm (Celikyilmaz, Clark, and Gao 2020). Specifically, when employing constrained generation with a prefix trie, the model needs to check each generated token against the trie to ensure it aligns with the allowed prefixes. This additional step can slow down the generation process because it introduces extra computational overhead to enforce these constraints. This finding suggests that future research should not only focus on improving demonstration selection efficiency but also consider optimizing model inference time to achieve better overall computational performance.
# Related Work
In-context learning has emerged as a powerful capability of LLMs, enabling them to adapt to new tasks without fine-tuning, but the effectiveness of this approach heavily depends on the quality and relevance of the provided demonstrations. With numerous demonstration selection algorithms in the field, the optimal method for selecting these examples remains an open question. For instance, Wang et al. (2024b) emphasize aligning demonstrations with the model’s latent structure, improving performance by uncovering latent variable explanations. Zhang, Feng, and Tan (2022); Qin et al. (2023) employ an active and iterative selection approach to identify the most informative examples. Retrieval-based methods are utilized by Wang, Yang, and Wei (2023); Wang et al. (2024a); Li et al. (2023); Xu et al. (2024) to select demonstrations. Liu, Zhu, and Dou (2024) adopt a ranking mechanism that considers relevance, similarity, and diversity for optimal demonstration selection. Ye et al. (2023) suggest constructing demonstrations through the combination of simpler examples to better capture complex task dynamics. Van, Wu et al. (2024) introduce InfICL, which uses influence functions to identify the most impactful examples for selection. Liu et al. (2024) uncover the importance of task-agnostic multi-level similarity and taskspecific label similarities, proposing methods that integrate these factors for improved performance across diverse tasks. Additionally, Kim et al. (2022); Zhou et al. (2024) introduce the concept of self-generated demonstrations, which involves generating examples. Together, these works underscore the critical role of demonstration selection in enhancing the effectiveness of in-context learning across varied applications. Our work differs from previous studies by providing a comparative analysis of multiple demonstration selection algorithms across various tasks and datasets. While existing literature has primarily focused on developing individual algorithms or comparing a limited set of methods, we offer a holistic evaluation of both the effectiveness and efficiency of six demonstration selection algorithms.
# Conclusions and Future Work
In this paper, our comprehensive evaluation of demonstration selection algorithms reveals significant variations in both effectiveness and efficiency across different tasks and datasets. While some algorithms show promising results, the
trade-offs between accuracy and computational cost present challenges for real-world applications. Our findings highlight the need for task-specific optimization and suggest potential avenues for future research. These include developing adaptive algorithms that can dynamically adjust the number of demonstrations based on task complexity, integrating more advanced retrieval techniques, and exploring methods to balance effectiveness with computational efficiency.
# Acknowledgment
The work is in part supported by NSF #2310261. The views and conclusions in this paper are those of the authors and should not be interpreted as representing funding agencies.
# References
Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774. AnthropicAI. 2023. Introducing claude. Celikyilmaz, A.; Clark, E.; and Gao, J. 2020. Evaluation of text generation: A survey. arXiv preprint arXiv:2006.14799. Cheng, D.; Huang, S.; Bi, J.; Zhan, Y.; Liu, J.; Wang, Y.; Sun, H.; Wei, F.; Deng, D.; and Zhang, Q. 2023. Uprise: Universal prompt retrieval for improving zero-shot evaluation. arXiv preprint arXiv:2303.08518. Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Yang, A.; Fan, A.; et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783. Kim, H. J.; Cho, H.; Kim, J.; Kim, T.; Yoo, K. M.; and Lee, S.-g. 2022. Self-generated in-context learning: Leveraging auto-regressive language models as a demonstration generator. arXiv preprint arXiv:2206.08082. Li, X.; Lv, K.; Yan, H.; Lin, T.; Zhu, W.; Ni, Y.; Xie, G.; Wang, X.; and Qiu, X. 2023. Unified demonstration retriever for in-context learning. arXiv preprint arXiv:2305.04320. Liu, H.; Wang, W.; Sun, H.; Tian, C. X.; Kong, C.; Dong, X.; and Li, H. 2024. Unraveling the Mechanics of LearningBased Demonstration Selection for In-Context Learning. arXiv preprint arXiv:2406.11890. Liu, W.; Zhu, Y.; and Dou, Z. 2024. DemoRank: Selecting Effective Demonstrations for Large Language Models in Ranking Task. arXiv preprint arXiv:2406.16332. Min, S.; Lewis, M.; Zettlemoyer, L.; and Hajishirzi, H. 2021. Metaicl: Learning to learn in context. arXiv preprint arXiv:2110.15943. Min, S.; Lyu, X.; Holtzman, A.; Artetxe, M.; Lewis, M.; Hajishirzi, H.; and Zettlemoyer, L. 2022. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837. Qin, C.; Zhang, A.; Dagar, A.; and Ye, W. 2023. In-context learning with iterative demonstration selection. arXiv preprint arXiv:2310.09881.
Anadkat, S.; et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774. AnthropicAI. 2023. Introducing claude. Celikyilmaz, A.; Clark, E.; and Gao, J. 2020. Evaluation of text generation: A survey. arXiv preprint arXiv:2006.14799. Cheng, D.; Huang, S.; Bi, J.; Zhan, Y.; Liu, J.; Wang, Y.; Sun, H.; Wei, F.; Deng, D.; and Zhang, Q. 2023. Uprise: Universal prompt retrieval for improving zero-shot evaluation. arXiv preprint arXiv:2303.08518. Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Yang, A.; Fan, A.; et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783. Kim, H. J.; Cho, H.; Kim, J.; Kim, T.; Yoo, K. M.; and Lee, S.-g. 2022. Self-generated in-context learning: Leveraging auto-regressive language models as a demonstration generator. arXiv preprint arXiv:2206.08082. Li, X.; Lv, K.; Yan, H.; Lin, T.; Zhu, W.; Ni, Y.; Xie, G.; Wang, X.; and Qiu, X. 2023. Unified demonstration retriever for in-context learning. arXiv preprint arXiv:2305.04320. Liu, H.; Wang, W.; Sun, H.; Tian, C. X.; Kong, C.; Dong, X.; and Li, H. 2024. Unraveling the Mechanics of LearningBased Demonstration Selection for In-Context Learning. arXiv preprint arXiv:2406.11890. Liu, W.; Zhu, Y.; and Dou, Z. 2024. DemoRank: Selecting Effective Demonstrations for Large Language Models in Ranking Task. arXiv preprint arXiv:2406.16332. Min, S.; Lewis, M.; Zettlemoyer, L.; and Hajishirzi, H. 2021. Metaicl: Learning to learn in context. arXiv preprint arXiv:2110.15943. Min, S.; Lyu, X.; Holtzman, A.; Artetxe, M.; Lewis, M.; Hajishirzi, H.; and Zettlemoyer, L. 2022. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837. Qin, C.; Zhang, A.; Dagar, A.; and Ye, W. 2023. In-context learning with iterative demonstration selection. arXiv preprint arXiv:2310.09881.
Robertson, S.; Zaragoza, H.; et al. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends® in Information Retrieval, 3(4): 333–389. Talmor, A.; Herzig, J.; Lourie, N.; and Berant, J. 2018. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937. Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.; Babaei, Y.; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale, S.; et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288. Van, M.-H.; Wu, X.; et al. 2024. In-Context Learning Demonstration Selection via Influence Analysis. arXiv preprint arXiv:2402.11750. Wang, A.; Singh, A.; Michael, J.; Hill, F.; Levy, O.; and Bowman, S. R. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461. Wang, H.; Wu, J.; Sun, H.; Xia, Z.; Cheng, D.; Wang, J.; Qi, Q.; and Liao, J. 2024a. MDR: Model-Specific Demonstration Retrieval at Inference Time for In-Context Learning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 4189–4204. Wang, L.; Yang, N.; and Wei, F. 2023. Learning to retrieve in-context examples for large language models. arXiv preprint arXiv:2307.07164. Wang, X.; Zhu, W.; Saxon, M.; Steyvers, M.; and Wang, W. Y. 2024b. Large language models are latent variable models: Explaining and finding good demonstrations for incontext learning. Advances in Neural Information Processing Systems, 36. Wu, Z.; Wang, Y.; Ye, J.; Feng, J.; Xu, J.; Qiao, Y.; and Wu, Z. 2023. Openicl: An open-source framework for in-context learning. arXiv preprint arXiv:2303.02913. Xie, S. M.; Raghunathan, A.; Liang, P.; and Ma, T. 2021. An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080. Xu, X.; Liu, Y.; Pasupat, P.; Kazemi, M.; et al. 2024. Incontext learning with retrieved demonstrations for language models: A survey. arXiv preprint arXiv:2401.11624. Ye, J.; Wu, Z.; Feng, J.; Yu, T.; and Kong, L. 2023. Compositional exemplars for in-context learning. In International Conference on Machine Learning, 39818–39833. PMLR. Zellers, R.; Bisk, Y.; Schwartz, R.; and Choi, Y. 2018. Swag: A large-scale adversarial dataset for grounded commonsense inference. arXiv preprint arXiv:1808.05326. Zhang, Y.; Feng, S.; and Tan, C. 2022. Active example selection for in-context learning. arXiv preprint arXiv:2211.04486. Zhou, X.; Ye, W.; Wang, Y.; Jiang, C.; Lee, Z.; Xie, R.; and Zhang, S. 2024. Enhancing In-Context Learning via Implicit Demonstration Augmentation. arXiv preprint arXiv:2407.00100.
Robertson, S.; Zaragoza, H.; et al. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends® in Information Retrieval, 3(4): 333–389. Talmor, A.; Herzig, J.; Lourie, N.; and Berant, J. 2018. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937. Touvron, H.; Martin, L.; Stone, K.; Albert, P.; Almahairi, A.; Babaei, Y.; Bashlykov, N.; Batra, S.; Bhargava, P.; Bhosale, S.; et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288. Van, M.-H.; Wu, X.; et al. 2024. In-Context Learning Demonstration Selection via Influence Analysis. arXiv preprint arXiv:2402.11750. Wang, A.; Singh, A.; Michael, J.; Hill, F.; Levy, O.; and Bowman, S. R. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461. Wang, H.; Wu, J.; Sun, H.; Xia, Z.; Cheng, D.; Wang, J.; Qi, Q.; and Liao, J. 2024a. MDR: Model-Specific Demonstration Retrieval at Inference Time for In-Context Learning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 4189–4204. Wang, L.; Yang, N.; and Wei, F. 2023. Learning to retrieve in-context examples for large language models. arXiv preprint arXiv:2307.07164. Wang, X.; Zhu, W.; Saxon, M.; Steyvers, M.; and Wang, W. Y. 2024b. Large language models are latent variable models: Explaining and finding good demonstrations for incontext learning. Advances in Neural Information Processing Systems, 36. Wu, Z.; Wang, Y.; Ye, J.; Feng, J.; Xu, J.; Qiao, Y.; and Wu, Z. 2023. Openicl: An open-source framework for in-context learning. arXiv preprint arXiv:2303.02913. Xie, S. M.; Raghunathan, A.; Liang, P.; and Ma, T. 2021. An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080. Xu, X.; Liu, Y.; Pasupat, P.; Kazemi, M.; et al. 2024. Incontext learning with retrieved demonstrations for language models: A survey. arXiv preprint arXiv:2401.11624. Ye, J.; Wu, Z.; Feng, J.; Yu, T.; and Kong, L. 2023. Compositional exemplars for in-context learning. In International Conference on Machine Learning, 39818–39833. PMLR. Zellers, R.; Bisk, Y.; Schwartz, R.; and Choi, Y. 2018. Swag: A large-scale adversarial dataset for grounded commonsense inference. arXiv preprint arXiv:1808.05326. Zhang, Y.; Feng, S.; and Tan, C. 2022. Active example selection for in-context learning. arXiv preprint arXiv:2211.04486. Zhou, X.; Ye, W.; Wang, Y.; Jiang, C.; Lee, Z.; Xie, R.; and Zhang, S. 2024. Enhancing In-Context Learning via Implicit Demonstration Augmentation. arXiv preprint arXiv:2407.00100.
