# Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning
Xinyi Wang1, Wanrong Zhu1, Michael Saxon1, Mark Steyvers2, William Yang Wang1 1Department of Computer Science, University of California, Santa Barbara 2Department of Cognitive Sciences, University of California, Irvine {xinyi_wang, wanrongzhu, saxon}@ucsb.edu, msteyver@uci.edu, william@cs.ucsb.edu
# Abstract
In recent years, pre-trained large language models (LLMs) have demonstrated remarkable efficiency in achieving an inference-time few-shot learning capability known as in-context learning. However, existing literature has highlighted the sensitivity of this capability to the selection of few-shot demonstrations. Current understandings of the underlying mechanisms by which this capability arises from regular language model pretraining objectives remain disconnected from the realworld LLMs. This study aims to examine the in-context learning phenomenon through a Bayesian lens, viewing real-world LLMs as latent variable models. On this premise, we propose an algorithm to select optimal demonstrations from a set of annotated data with a small LM, and then directly generalize the selected demonstrations to larger LMs. We demonstrate significant improvement over baselines, averaged over eight GPT models on eight real-world text classification datasets. We also demonstrate the real-world usefulness of our algorithm on GSM8K, a math word problem dataset. Our empirical findings support our hypothesis that LLMs implicitly infer a latent variable containing task information. 1
# 1 Introduction
Transformer-based [41] pre-trained large language models (LLMs) have demonstrated significant advancements in a variety of natural language processing (NLP) tasks. As the size of these LLMs increases, they gain “in-context learning” capabilities, whereby the models achieve state-of-the-art (SOTA) or near-SOTA performance by conditioning on a small number of demonstration examples at inference time, without any need for updating model parameters [4]. Below is an example input sequence for semantic analysis with in-context learning: Great movie. Positive.\n The worst movie ever. Negative.\n Can’t wait to see the second movie! The first two lines are two demonstrations, and the third line is a test input. We expect an LLM to output the correct label Positive as a continuation. In-context learning has been demonstrated to be an effective technique for a wide range of NLP tasks. However, it is sensitive to the choice, format, and even the order of the demonstrations used [29, 20]. This makes achieving optimal performance with in-context learning a significant challenge, requiring real human effort to adjust the format and selection of demonstration examples. Heuristic solutions, such as selecting demonstrations based on the similarity between the demonstrations and test input
[19, 37] have been proposed, but a comprehensive understanding of why certain demonstrations are effective while others are not remains elusive. Additionally, the mechanisms by which LLMs acquire in-context learning capabilities through training on natural text under the standard language model pre-training objective are not fully understood. Recent works on understanding in-context learning provide valuable insights and theoretical results [5, 1, 42, 14, 12], but are limited in scope, focusing on synthetic experiments to validate their hypotheses, while it remains unclear if these results generalize to LLMs pre-trained on real-world natural language data. Xie et al. [50] introduced a prominent result providing a latent topic (concept) variable interpretation for in-context learning. They showed that the in-context learning predictor approaches the Bayes optimal predictor when the number of demonstrations approaches infinity, under the assumption that both the pre-training data distribution and task-specific data distribution are Hidden Markov Models (HMM). However, the assumption that the data generation process is Hidden Markovian makes extrapolation of the result to natural language questionable, and restricts empirical verification to synthetic data with toy models. We are inspired by this prior work and introduce a more general and natural explanation built on realistic assumptions, which gives rise to a practical demonstration selection algorithm. Our explanation is inspired by the generation process of a topic model, i.e. a simple latent variable model: �
� Where θ ∈Θ represents a potentially high dimensional topic/concept variable, Θ is the space of the topic/concept variable, and w1:T refers to the token sequence of a piece of text. Note that the topic model here refers to the modern neural topic models [23, 22]. On the other hand, generative LLMs model text data according to the general probabilistic decomposition:
� While in practice, LLMs generate new tokens based on all previous tokens, we investigate whether a simplified assumption similar to that of topic models can be made for LLMs: �
� In this scenario, the generated tokens are assumed to be conditionally independent of previous tokens, given the latent topic (concept) variable that acts like an approximate sufficient statistic for the posterior information related to the prompt w1:t. For in-context learning, this concept variable includes format and task information. By conditioning on an appropriate latent concept variable, LLMs would generate the desired continuation with P(wt+1:T |θ). As LLMs do not explicitly learn a latent variable distribution like LDA-style topic models [3], we can instead utilize this formulation under an Empirical Bayesian formulation inspired by Lester et al. [17] to only approximate the optimal latent variable value for a desired task, using a small LLM (with less than 1B parameters), which is computationally efficient. We empirically validate our explanation by selecting examples (w1:t in the equations) that are most likely to infer the optimal latent variable value (those with the highest posterior probability P(θ|wt+1:T )). We then directly use them as demonstrations for in-context learning with other larger LLMs (up to 175B parameters) and observed a significant performance improvement. The generalization of demonstrations between LLMs is likely a result of similar pre-training data distributions. While our work is inspired by that of Xie et al. [50], our approach differs significantly in both theoretical analysis and experimental settings. Our main contributions are as follows: • We assume a general data generation process specified by a three-variable causal graph, without constraints on the distribution function or the number of demonstrations. • We prove under these realistic assumptions that the in-context learning predictor can reach the Bayes optimal predictor with a finite number of demonstrations chosen using the latent concept variable. • We introduce an efficient, practical demonstration selection algorithm based on our theoretical results, which can select demonstrations using a small LLM and then directly generalize the demonstrations to other LLMs. The effectiveness of our algorithm is empirically validated using real-world LLMs on both text classification tasks and math word problems.
Our goal is to close the gap between theoretical understandings and real-world LLMs. To the best of our knowledge, our proposed latent variable explanation of in-context learning is the first Bayesian explanation that yields an effective algorithm in real-world scenarios.
# 2 Theoretical Analysis
In in-context learning, the prompt w1:t is composed of several demonstrations and a test input. The generated tokens wt+1:T represent the model’s prediction for the test input.
# 2.1 Notations and Problem Setting
Suppose the objective of our task is to predict a discrete target variable Y ∈Y, given a token sequence X ∈X, where X is the space of all possible token sequences. θ ∈Θ is a potentially high dimensional latent variable, where Θ is the high dimensional space of the variable. Unlike the traditional topic model, θ is not assumed to be discrete, but continuously distributed over Θ. To define the data generation process, we posit the existence of an underlying causal relation between X, Y , and θ. We examine two potential directions of this causal relation, namely X �Y �θ and Y �X �θ, which can be represented mathematically as the following structural equations:
 � � Here ϵ ∈E is an independent noise variable, f : X × Θ × E →Y and g : Y × Θ × E →X are two deterministic functions. Furthermore, we denote the joint data distribution by X, Y, θ ∼P, and assume that Y is sampled from a uniform distribution over Y. The distinction between these two directions is crucial, as it allows us to utilize the direction in which the child variable (Y or X) is independent of the other variables, given its parents. We hypothesize that the causal direction depends on the nature of the task. For instance, in the task of predicting the sentiment (Y ) of a movie review (X), it is reasonable to assume that the opinion about the movie is formed before writing the review, thus making Y the cause of X, along with the task concept of “writing a passage to express one’s opinion about the movie" (θ). Conversely, for the task of classifying whether a product review (X) is helpful to other customers (Y ), it is the quality of the review (X) that cause other customers to upvote it (Y ), along with the task concept of “rating the helpfulness of this review" (θ). In the rest of the paper, we will focus on the X �Y �θ direction and leave a detailed discussion of the other direction in the Appendix. Suppose we are interested in a task (e.g. semantic analysis) denoted by d ∈T , where T is the space of all possible tasks. We assume there is an injective function between T and Θ. i.e. for each task d, there is a concept variable θd, such that each data (Xd, Y d) sampled from task d is generated by:
To perform in-context learning with an LLM (generically denoted by model label M), we condition on a fixed set of k demonstration examples (Xd 1, Y d 1 ), (Xd 2, Y d 2 ), ..., (Xd k, Y d k ) sampled from task d. Following previous works [24, 26], as we are not using any instruction fine-tuned models, we do not include a task description in the prompt, with the aim of focusing on the examination of the demonstrations. To naturally project Y into the token space X, we define injective mappings τ d : Y →X, which are typically defined by human understanding of the task d. e.g. for sentiment analysis, τ d map positive class to the token “positive" and negative class to the token “negative". Additionally, a delimiter token wd is defined, typically an empty space or a new line token, to separate the demonstrations when concatenated. We denote the LLM output probability of X, Y , and θ, with the aforementioned preprocessing applied, by P d M:
# 2.2 Problem Analysis and Theoretical Results
Suppose a set of observed data sampled from task d, denoted as Dd, is available, allowing for the selection of the k most suitable demonstrations from it. For any incoming test example X, we have: �
Suppose a set of observed data sampled from task d, denoted as Dd, is available, allowing for the selection of the k most suitable demonstrations from it. For any incoming test example X, we have:  d d  d d  d �  d  d d  d d  d (1)
Suppose a set of observed data sampled from task d, denoted as Dd, is available, allowing for the selection of the k most suitable demonstrations from it. For any incoming test example X, we have: P d M(Y |Xd 1, Y d 1 , ..., Xd k, Y d k , X) = � Θ P d M(Y |θ, X)P d M(θ|Xd 1, Y d 1 , ..., Xd k, Y d k , X)dθ (1)
(1)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2918/29188955-eb89-48cb-afd9-13215fd8f85b.png" style="width: 50%;"></div>
Figure 1: An overview of our proposed two-phased algorithm. Demonstration selection and latent concept learning share the same LLM as demonstration selection needs to reuse the learned concept tokens, while at the in-context learning time, any other generative LLMs can be used. Here we only illustrate the X �Y �θ direction. The Y �X �θ direction can be illustrated similarly by exchanging X and Y in the above figure. Here, we assume the sampling of the test example is independent of the sampling of the demonstrations, so Y is independent of the demonstrations given θ and X. We also assume that the pre-trained data distribution P d M is a suitable approximation of the assumed data distribution P: Assumption 2.1. Assume that PM(X) = P(X), and P d M(Y |θ, X) ∝P(Y |θ, X) for X �Y �θ. Note that the assumption that a large language model captures the true distribution of language is fairly common in the literature studying LLMs [50, 34, 47]. With this assumption, we establish: Proposition 2.2. If task d follows the X �Y �θ direction, then arg maxy∈Y P d M(Y = y|θd, X) is the Bayes optimal classifier. In this case, only when P d M(θ|Xd 1, Y d 1 , ..., Xd k, Y d k , X) completely concentrate on θd, can the incontext learning classifier become the Bayes optimal classifier [11]: Theorem 2.3. If task d follows the X �Y �θ direction, then the in-context learning classifier arg max y∈Y P d M(Y = y|Xd 1, Y d 1 , ..., Xd k, Y d k , X) always has a higher or equal probability of misclassification to the Bayes optimal classifier arg maxy∈Y P d M(Y = y|θd, X). Equality only holds when ∀x ∈X, P d M(θd|Xd 1, Y d 1 , ..., Xd k, Y d k , X = x) = 1. A similar argument can be made for the Y �X �θ direction. 2 Here, Equation (1) would become: P d M(X|Y d 1 , Xd 1, ..., Y d k , Xd k, Y ) = � Θ P d M(X|θ, Y )P d M(θ|Y d 1 , Xd 1, ..., Y d k , Xd k, Y )dθ (2) Note that the left-hand side of Equation (1) and Equation (2) are similar to the direct and channel method introduced by Min et al. [24]. However, our analysis differs from theirs in that we do not treat (Y �X �θ) as the universally superior channel direction for modeling in-context learning, rather arguing that depending on the end task, the causal direction (X �Y �θ) is sometimes better. This view is supported by our empirical results in Appendix B.
A similar argument can be made for the Y �X �θ direction. 2 Here, Equation (1) would become P d M(X|Y d 1 , Xd 1, ..., Y d k , Xd k, Y ) = � Θ P d M(X|θ, Y )P d M(θ|Y d 1 , Xd 1, ..., Y d k , Xd k, Y )dθ (2)
� Note that the left-hand side of Equation (1) and Equation (2) are similar to the direct and channel method introduced by Min et al. [24]. However, our analysis differs from theirs in that we do not treat (Y �X �θ) as the universally superior channel direction for modeling in-context learning, rather arguing that depending on the end task, the causal direction (X �Y �θ) is sometimes better. This view is supported by our empirical results in Appendix B.
# 3 Method
Here we demonstrate how the proposed theory can be practically applied to select optimal demonstration examples. Since latent variable θ encodes both the task and format information, the whole distribution over Θ is too complex to model. Unlike traditional topic models, we will only focus on estimating an optimal value θd corresponding to task d. First, we perform latent concept learning, wherein the task latent θd is learned as a set of new token embeddings using prompt tuning over the full demonstration candidate set. With this optimal task latent, we then perform demonstration selection, where a smaller set of demonstrations is chosen to maximize the likelihood of postfixing the latent concept tokens. We only need to use a small LLM to do the above steps to obtain an optimal set of demonstrations that can be directly transferred to other LLMs. Figure 1 is an overall illustration of our proposed method.
Here we demonstrate how the proposed theory can be practically applied to select optimal demonstration examples. Since latent variable θ encodes both the task and format information, the whole distribution over Θ is too complex to model. Unlike traditional topic models, we will only focus on estimating an optimal value θd corresponding to task d.
First, we perform latent concept learning, wherein the task latent θd is learned as a set of new token embeddings using prompt tuning over the full demonstration candidate set. With this optimal task latent, we then perform demonstration selection, where a smaller set of demonstrations is chosen to maximize the likelihood of postfixing the latent concept tokens. We only need to use a small LLM to do the above steps to obtain an optimal set of demonstrations that can be directly transferred to other LLMs. Figure 1 is an overall illustration of our proposed method.
2The detailed argument of the Y �X �θ direction can be found in Appendix 
(2)
Algorithm 1 Latent concept learning
Input: Dataset D = {(xi, yi, di)}i associated with a set of tasks S, LLM M, number of concept
tokens per task c, learning rate α, and number of training steps N.
Output: LLM M ′ with fine-tuned concept tokens.
Add c|S| new tokens to the vocabulary. i.e. The concept tokens ˆθd for each task in S. Randomly
initialize their embeddings Enew. Freeze all parameters in M except Enew;
for step = 1 to N do
Sample a random batch B in D and initialize gradient g ←0;
for each data point (x, y, d) in B do
g = g + ∂ℓ(X,Y ;ˆθd)
∂Enew
;
end for
Enew = Enew −αg;
end for
# 3.1 Latent Concept Learning
We want to first find the optimal value of the latent concept variable θd corresponding to a task d ∈T . As arg maxy∈Y P d M(Y = y|θd, X) is the Bayes optimal classifier according to Proposition 2.2, θd should be able to minimize −EX,Y,d[log P d M(Y |θd, X)] for the X �Y �θ direction. In practice, we try to align θd to the token embedding space by adding new tokens to the vocabulary. After this alignment, we hope to be able to use the learned new tokens of θd as regular tokens. More specifically, building upon the methodology proposed by Lester et al. [17], for each specific task d, c new concept tokens (denoted as ˆθd) are added to the original vocabulary of LLM M to represent the corresponding task concept θd. Subsequently, the embedding of these new tokens Enew(ˆθd) is fine-tuned while freezing the remaining parameters of LLM M. The variable c is treated as a hyperparameter. In practice, in order to condition on θd, the corresponding c concept tokens are appended to the input X (or Y ) as shown in the example provided below, where c = 2: <sentiment_token_1><sentiment_token_2> Can’t wait to see the second movie! By giving the above input tokens, we ask the LLM to predict the correct label Positive for us. Note that <sentiment_token_1> here is just a label assigned to the newly added concept token. It can be anything as long as it does not overlap with the original vocabulary of LLM. The fine-tuning objective would then be minimizing L(ˆθd) = EX,Y [ℓ(X, Y ; ˆθd)], where ℓ(X, Y ; ˆθd) = � −log P d M(Y |ˆθd, X) if X �Y �θ −log P d (X|ˆθd, Y ) if Y �X �θ.
ℓ(X, Y ; ˆθd) = � −log P d M(Y |ˆθd, X) if X �Y �θ −log P d M(X|ˆθd, Y ) if Y �X �θ.
 � � Theoretically, if we can minimize the above loss function, a Bayes optimal classifier can be obtained, and the concept tokens would be a reasonable delegate of the real latent concept variable: Proposition 3.1. When L(ˆθd) is minimized, P d M(Y |ˆθd, X) = P(Y |θd, X) for X �Y �θ. If the LLM M is invertible, then ˆθd = θd.3 We denote the LLM M with fine-tuned concept tokens by M ′. Since we add the concept tokens into the regular token vocabulary, the raw LLM output probability PM ′(ˆθd|w1:t) (w1:t denote a given prompt) would be in the token sequence space X instead of the concept space Θ. Since learning all possible θd ∈Θ is infeasible, we propose to approximate the concept space Θ by sampling a diverse subset of tasks S ⊆T . Then the estimated conditional probability of θd would be:
� To obtain the concept tokens for all tasks in S, we fine-tune all tasks together with the loss � d∈S L(θd). We summarize the proposed algorithm in Algorithm 1.
3More discussion can be found in Appendix A.3.
Algorithm 2 Demonstration selection
Input: dataset Dd for a task d. LLM with fine-tuned concept tokens M ′. The number of
demonstrations k.
Output: A set of selected demonstrations.
for each (Xd, Y d) in Dd do
Compute ˆP d
M(ˆθd|Xd, Y d);
end for
Select top k examples with the largest ˆP d
M(ˆθd|Xd, Y d), denoted as (Xd
1, Y d
1 ), ..., (Xd
k, Y d
k );
Note that the embedding matrix of a generative LLM is shared on both the input and output sides. So while we only see the concept tokens on the input side at the training time, they can be viewed as regular word tokens that can be generated on the output side.
# 3.2 Demonstration Selection
According to Theorem 2.3, for a task d, to make the in-context learning classifier closer to the Bayes optimal classifier, we need to select demonstrations (Xd 1, Y d 1 ), ..., (Xd k, Y d k ) that maximize P d M(θd|Xd 1, Y d 1 , ..., Xd k, Y d k , X) for all X ∈X. Then our goal then becomes selecting demonstrations that can best infer the task concept for all test inputs on average:
As test examples are sampled independent of the demonstrations, and PM(X) = P(X) according to Assumption 2.1, we have
As test examples are sampled independent of the demonstrations, and PM(X) = P(X) according to Assumption 2.1, we have
P d M(θd|Xd 1, Y d 1 , ..., Xd k, Y d k ) = �k i=1 P d M(θd|Xd i , Y d i ) P d M(θd)k−1
Assuming that θ has a uniform prior, then our goal becomes finding the top k demonstrations that maximize ˆP d M ′(ˆθd|Xd i , Y d i ). Note that the independence between demonstrations is a simplified assumption to reduce the combinatory search space of (Xd 1, Y d 1 ), ..., (Xd k, Y d k ). In practice, selected demonstrations are likely correlated as some demonstrations may work well together but not necessarily work well by themselves. However, it would be too expensive to search the O(|Dd|k) combinations over the candidate set Dd. In practice, this simplification works reasonably well. We leave this combinatory search problem to future research. Also, as we are using an LLM to approximate the data distribution, the order of the demonstrations might matter. We will show in the Experiment section that the order does not matter, so no reordering of the selected demonstrations is needed. The full selection algorithm is shown in Algorithm 2.
Also, as we are using an LLM to approximate the data distribution, the order of the demonstrations might matter. We will show in the Experiment section that the order does not matter, so no reordering of the selected demonstrations is needed. The full selection algorithm is shown in Algorithm 2.
# 4 Experiments
Datasets. We conduct experiments on eight datasets from five different types of NLP classification tasks: sentiment analysis, linguistic analysis, topic classification, emotion classification, and hate speech detection. For sentiment analysis, we choose the Stanford Sentiment Treebank (SST2) dataset [35] from the GLUE benchmark [43] and the financial phrase bank (FPB) dataset [21]. SST2 is constructed based on movie reviews labeled “positive" or “negative", and FPB is based on financial news labeled “positive", “negative", or “neutral". For linguistic analysis, we choose the Corpus of Linguistic Acceptability (COLA) dataset [46] from the GLUE benchmark, based on sentences collected from linguistic books, labeled with “acceptable" or “unacceptable". For topic classification, we choose the DBpedia ontology classification dataset [52], based on DBpedia 2014 [16], labeled with 14 different ontology classes. For emotion classification, we choose the dataset from Chatterjee et al. [6] and Saravia et al. [33], both of which are collected from Twitter. Chatterjee et al. [6] (EmoC)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/576b/576b6441-6032-441c-b610-0e47f006c43a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Accuracy of 4-shot in-context learning using demonstrations selected by our method and other baselines, averaged over eight datasets. Our demonstrations are selected using GPT2-large, and the same set of demonstrations is then applied to all other LLMs.</div>
predict emotion given a three-turn contextual dialogue, while Saravia et al. [33] predict emotion given a Twitter message with clear emotion. For hate speech detection, we choose the online hate speech detection dataset (ETHOS) [27], collected from online social media platforms. Here we detect two types of hate speech: sexual orientation (ETHOS-SO) and religion (ETHOS-R). While in Section 2, we assume that all tasks share the same label space Y, here we relax such assumption and allow a different number of labels for different tasks. We use minimal formatting to process each example. A detailed description of the datasets and our data processing procedure can be found in Appendix B. Experiment settings. To determine the causal direction for each task, we select the direction that can give higher accuracy when using random demonstrations4. We adopt the Y →X ←θ direction for sentiment analysis, topic classification, and emotion classification tasks, which is consistent with the intuition that people usually have some sentiment, topic, or emotion in mind before writing a piece of text. We adopt the X →Y ←θ direction for the linguistic analysis and hate speech detection type of tasks. While this is less intuitive, we can understand this as linguistic error and hate speech detection are more of a post hoc task in contrast to the previous tasks. Without specification, we use k = 4 number of demonstrations and c = 10 number of concept tokens per dataset for our experiments, as the context length of GPT2 is 1024, and a larger number of demonstrations may not be able to completely fit into it. We use GPT2-large to learn the concept tokens and then compute the probability of each candidate demonstration example. We select our demonstrations from a randomly selected 100 example subset of the train set as the candidate set Dd. We use the same set of demonstrations selected by GPT2-large for all other LLMs. We test the performance of the selected demonstrations using at most 1000 examples randomly sampled from the test set. Each experiment is repeated for five runs with different random seeds (the randomness comes from the sampling of the candidate set and the sampling of the test set). We adopt a large portion of the code from Min et al. [25], which is based on Huggingface [49]. Baselines. We consider the following baselines:
• Uniform: We uniformly select k demonstrations from D for each test example. • Similar: According to Liu et al. [19], demonstrations that are semantically similar to the test example would hare more performant. Following their method, we use a pre-trained sentence Transformer [31] to calculate the cosine similarity between the demonstrations and test examples. We choose the top k similar demonstrations from D for each test example.
Main results.5 Figure 2 shows our main results averaged over all eight datasets, using the firstgeneration GPT2s and GPT3s, without any instruction fine-tuning [28] or Reinforcement Learning from Human Feedback (RLHF) [36]. Our method significantly outperforms baselines on eight different LLMs, with 12.5% relative improvement to the uniform selection baseline on average, which shows the effectiveness of our method. The demonstrations selected by our method are exclusively based on GPT2-large, while the same set of demonstrations can be generalized to all other GPTs. Results with non-GPT models. In Figure 3a, we test the demonstrations selected by our method using GPT2-large on more LLMs (GPT3 [4], GPT3-instruct [28, 36], GPT-J [44], OPT [51], and LLaMA [38]) with similar sizes (6-7B), and show that the selected demonstrations improve in-context learning performance of all of them. The fact that GPT3-curie obtains the largest performance improvement is likely because similar pre-training data distributions help the generalization of the
4Detailed results see Figure 6 in Appendix B. 5The complete results with standard deviations in this section can be found in A
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5447/54472321-cba3-4018-a072-94ae652033e1.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Proposed method v.s. randomly selected demonstrations randomly selected tokens Figure 3: In-context learning accuracy averaged over all eight datasets.</div>
<div style="text-align: center;">Figure 3: In-context learning accuracy averaged over all eight datasets.</div>
Uniform
Similar
Ours w/ Llama 2 (7B)
Ours w/ GPT2-XL (1.5B)
Prompt tuning
-
-
15.2
7.3
Llama 2 (7B)
11.4
13.1
19.3
15.9
Llama 2 (13B)
17.0
18.3
21.6
20.5
Llama 2 (70B)
50.2
53.5
54.3
52.9
ChatGPT (gpt-3.5-turbo)
76.5
78.1
81.2
80.4
Table 1: Prompt tuning and 4-shot in-context learning accuracy on a subset of GSM8K test set. Our
selected demonstrations. Different-size GPT2 models share the same pre-training corpus [30], while GPT3s are pre-trained on a dataset expanded from the GPT2 pre-training corpus [4]. Thus the pre-training distribution of GPT3-curie and GPT2-large can be assumed to be similar. Results on GSM8K. Since our primary goal is to connect the theory with real-world models and datasets, we did not try to include harder tasks in the main results in Figure 2. In practice, our proposed method is most effective with hard tasks that even parameter-efficient fine-tuning with smaller models cannot outperform in-context learning with the same or larger models. To showcase the usefulness of our proposed algorithm, We added a new dataset, GSM8K [9], which is a math word problem-solving dataset with chain-of-thoughts solutions. Table 1 shows the test accuracy of the final numerical answer with greedy generation. We randomly select a test set of 200 examples instead of using the full test set for computation efficiency. 6 As shown in the first row of Table 1, prompt tuning with ten new tokens can only obtain less than 4% accuracy on the GSM8K test set. The last four rows show the in-context learning results with different size Llama 2 models [39] and ChatGPT. Our proposed demonstration selection method (last two columns) significantly outperformed the Uniform and Similar baseline. We also find that the demonstrations selected with a larger model (7B) are more effective than those selected with a smaller model (1.5B). The results show that our demonstration selection method is a good choice under a low data setting, with a small computing budget and minimal inference latency. Our proposed method can also potentially be combined with other prompting techniques [8] to boost performance further. Learned tokens v.s. Random tokens. To confirm the critical role of the latent concept variable in the proposed demonstration selection algorithm, we compare the performance of using the learned concept tokens versus using randomly selected tokens from the original vocabulary in Figure 3b. The demonstrations selected by random tokens only obtain the same performance as randomly selected demonstrations, showing that the performance gain of our method comes from the learned concept tokens containing the task and format information, not other elements of our algorithm. k ablation study. While we use k = 4 demonstrations for all experiments, we also test the effectiveness of our method using different k. As shown in Figure 4a, our method significantly outperforms the random selection baseline with k = 2, 4, 8, and 16. To fit in large ks, we use GPT3-ada with a longer context length (2048). Note that for real-world tasks, it is in general not true that more demonstrations guarantee higher performance [7]. We can see that the uniform baseline performance increases from k = 2 to k = 8, then drops a little at k = 16. Our method improves the uniform baseline by around 5% absolute for all ks, while k = 4 improves the most (6.6%). Our method appears to have a diminishing effect when k becomes larger, which is likely because the effect of more demonstrations overwhelms the effect of demonstration choices.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6dbe/6dbee534-494d-4b28-8fc8-205260a67229.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Proposed method v.s. using randomly selected tokens</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3f1c/3f1ca00e-0421-4a2a-ac01-e9f78c63e66a.png" style="width: 50%;"></div>
c ablation study. While we use c = 10 number of concept tokens for all experiments, we also investigate the effect of different c on our method. When c is small (c = 5), the concept tokens cannot effectively capture the task and format information, thus cannot im-
<div style="text-align: center;">(a) k ablation study.</div>
<div style="text-align: center;">Figure 4: In-context learning accuracy of our method versus random selection baseline averaged over all eight datasets with GPT3-ada.</div>
prove the performance. When c increases from 10 to 20, we observe a drop in the performance. It is likely because the selectivity of the concept tokens decreases when c increases. The longer the concept token sequence is, the more likely it will contain meaningless tokens that do not contribute to demonstration selection.
Effect of demonstrations’ order. We find that the demonstrations selected by our method are insensitive to their order in most cases.7 An exception is the EmoC dataset, where our method has a high variance. On the contrary, Lu et al. [20] found that the order of the demonstration matters, and a good ordering cannot be transferred between different LLMs. We suspect that the ordering only matters when the demonstration selection method is not robust. Since Lu et al. [20] randomly selects one set of demonstrations for the whole test set, the variance in performance is high with different demonstrations, thus ordering matters. And since such ordering is not transferable while our selected demonstrations are highly transferable, we suspect the core task information is stored in the content of the demonstrations, while the ordering mainly captures model-specific artifacts.
Qualitative analysis. In Figure 5, we provide a t-SNE [40] projection of the learned concept token embeddings. The tokens corresponding to semantically similar tasks are close together. Note that this result only aims to provide a straightforward illustration of concept tokens. The effect of concept tokens should be understood by the previous quantitative results.8 We also list the top 4 selected demonstrations in Table 14 in Appendix B. Compared to the examples with lower scores, the selected examples for GSM8K have more deductive reasoning (i.e. with the connecting words ‘so’, ‘then’, ‘thus’, etc.), instead of listing parallel conditions. For SST2, the selected examples are longer and more complex, sometimes including a ‘but’. This can be understood as these
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/027b/027bb897-3f73-4c9a-ac9f-860a81a4cd64.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: t-SNE plot of the learned concept tokens for each task. Concept tokens that can be explained by similar tokens are summarized in the graph.</div>
harder examples can represent the task more comprehensively. This conclusion also aligns with the findings in [13] that hard examples in the pre-training data contribute to in-context learning the most. The label distribution of the selected demonstrations is usually balanced in class, which reduces the possible biases introduced by the demonstrations.
# 5 Related Work
Heuristic solutions, such as selecting demonstrations based on the similarity between the demonstrations and test input [19, 37, 32] have been proposed. [20] propose to reorder the demonstration based on the entropy of the predicted labels. In this paper, we use the similarity-based selection method
7Detailed results see Figure 9 in Appendix B. 8The list of similar tokens for these concept tokens can be found in Table 13 in A
as a baseline while do not include the label entropy-based reordering method as we show that the ordering of the demonstrations does not matter for our method.
as a baseline while do not include the label entropy-based reordering method as we show that the ordering of the demonstrations does not matter for our method. Previous research on the phenomenon of in-context learning in Transformers has identified a number of pre-training data distributions that can lead to the emergence of this capability, including a Hidden Markov Model distribution [50] and a skewed Zipfian distribution with high burstiness [5]. Other studies have sought to understand the underlying mechanisms of in-context learning by making connections with gradient descent [42, 10, 1], formalizing it as an algorithm learning problem [18], or proposing a latent variable theory similar as ours [14, 12, 50]. While providing valuable insights on how in-context learning works, these works are limited to synthetic datasets and toy Transformers, while it remains unclear if these results generalize to LLMs pre-trained on real-world text data and whether these results can help in-context learning performance. In contrast, we propose a Bayesian explanation of in-context learning that can be verified with real-world LLMs on various NLP datasets. Dai et al. [10] provide a practical algorithm based on the understanding that the Transformer has a dual form of gradient descent. However, their empirical results are smaller in scale, with six datasets and only one model (350M), and has less significant improvements (5.4% relative to baseline). There are also works trying to understand in-context learning from an empirical perspective [2, 24]. Min et al. [26] found demonstrations’ ground truth labels do not matter for in-context learning, which we find is not entirely accurate in Appendix B. On the other hand, chain-of-thoughts prompting [48, 53, 45] find that providing step-by-step explanations improves in-context learning performance.
# 6 Conclusion
In this work, we endeavor to comprehend large language models (LLMs) through a Bayesian lens and posit them as implicit topic models that infer a latent conceptual variable from prompts. Motivated by this understanding, we propose a two-step algorithm that first extracts latent conceptual tokens from a small LLM and then selects demonstrations that have the greatest probability of predicting the corresponding conceptual tokens. The selected demonstrations can then be directly generalized to other LLMs. The efficacy of our algorithm across various text classification datasets and GPT models validates our explanation of in-context learning.
# Acknowledgements
This work was supported by the National Science Foundation award #2048122. The views expressed are those of the author and do not reflect the official policy or position of the US government. We thank Google for its generous gift to the University of California.
# References
[1] E. Akyürek, D. Schuurmans, J. Andreas, T. Ma, and D. Zhou. What learning algorithm is in-context learning? investigations with linear models. arXiv preprint arXiv:2211.15661, 2022. [2] H. Bansal, K. Gopalakrishnan, S. Dingliwal, S. Bodapati, K. Kirchhoff, and D. Roth. Rethinking the role of scale for in-context learning: An interpretability-based case study at 66 billion scale. arXiv preprint arXiv:2212.09095, 2022. [3] D. M. Blei, A. Y. Ng, and M. I. Jordan. Latent dirichlet allocation. J. Mach. Learn. Res., 3 (null):993–1022, mar 2003. ISSN 1532-4435. [4] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. [5] S. C. Chan, A. Santoro, A. K. Lampinen, J. X. Wang, A. Singh, P. H. Richemond, J. McClelland, and F. Hill. Data distributional properties drive emergent few-shot learning in transformers. arXiv preprint arXiv:2205.05055, 2022. [6] A. Chatterjee, K. N. Narahari, M. Joshi, and P. Agrawal. Semeval-2019 task 3: Emocontext contextual emotion detection in text. In Proceedings of the 13th International Workshop on Semantic Evaluation, pages 39–48, Minneapolis, Minnesota, USA, 2019. Association for
Computational Linguistics. doi: 10.18653/v1/S19-2005. URL https://www.aclweb.org/ anthology/S19-2005. [7] J. Chen, L. Chen, and T. Zhou. It takes one to tango but more make trouble? in-context training with different number of demonstrations. arXiv preprint arXiv:2303.08119, 2023. [8] W. Chen, X. Ma, X. Wang, and W. W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588, 2022. [9] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. [10] D. Dai, Y. Sun, L. Dong, Y. Hao, Z. Sui, and F. Wei. Why can gpt learn in-context? language models secretly perform gradient descent as meta optimizers. arXiv preprint arXiv:2212.10559, 2022. [11] L. Devroye, L. Györfi, and G. Lugosi. A probabilistic theory of pattern recognition. In Stochastic Modelling and Applied Probability, 1996. [12] M. Hahn and N. Goyal. A theory of emergent in-context learning as implicit structure induction. arXiv preprint arXiv:2303.07971, 2023. [13] X. Han, D. Simig, T. Mihaylov, Y. Tsvetkov, A. Celikyilmaz, and T. Wang. Understanding in-context learning via supportive pretraining data. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12660–12673, 2023. [14] H. Jiang. A latent space theory for emergent abilities in large language models. arXiv preprint arXiv:2304.09960, 2023. [15] B. LeBrun, A. Sordoni, and T. J. O’Donnell. Evaluating distributional distortion in neural language modeling. In International Conference on Learning Representations, 2022. [16] J. Lehmann, R. Isele, M. Jakob, A. Jentzsch, D. Kontokostas, P. N. Mendes, S. Hellmann, M. Morsey, P. Van Kleef, S. Auer, et al. Dbpedia–a large-scale, multilingual knowledge base extracted from wikipedia. Semantic web, 6(2):167–195, 2015. [17] B. Lester, R. Al-Rfou, and N. Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, 2021. [18] Y. Li, M. E. Ildiz, D. Papailiopoulos, and S. Oymak. Transformers as algorithms: Generalization and implicit model selection in in-context learning. arXiv preprint arXiv:2301.07067, 2023. [19] J. Liu, D. Shen, Y. Zhang, B. Dolan, L. Carin, and W. Chen. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pages 100–114, Dublin, Ireland and Online, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.deelio-1.10. URL https://aclanthology.org/2022.deelio-1.10. [20] Y. Lu, M. Bartolo, A. Moore, S. Riedel, and P. Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, 2022. [21] P. Malo, A. Sinha, P. Korhonen, J. Wallenius, and P. Takala. Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65, 2014. [22] Y. Miao, L. Yu, and P. Blunsom. Neural variational inference for text processing. In M. F. Balcan and K. Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1727–1736, New York, New York, USA, 20–22 Jun 2016. PMLR. URL https://proceedings.mlr. press/v48/miao16.html. [23] Y. Miao, E. Grefenstette, and P. Blunsom. Discovering discrete latent topics with neural variational inference. In International conference on machine learning, pages 2410–2419. PMLR, 2017.
[24] S. Min, M. Lewis, H. Hajishirzi, and L. Zettlemoyer. Noisy channel language model prompting for few-shot text classification. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5316–5330, 2022. [25] S. Min, M. Lewis, L. Zettlemoyer, and H. Hajishirzi. MetaICL: Learning to learn in context. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2791–2809, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. naacl-main.201. URL https://aclanthology.org/2022.naacl-main.201. [26] S. Min, X. Lyu, A. Holtzman, M. Artetxe, M. Lewis, H. Hajishirzi, and L. Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? In EMNLP, 2022. [27] I. Mollas, Z. Chrysopoulou, S. Karlos, and G. Tsoumakas. Ethos: an online hate speech detection dataset, 2020. [28] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022. [29] E. Perez, D. Kiela, and K. Cho. True few-shot learning with language models. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=ShnM-rRh4T. [30] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners. 2019. [31] N. Reimers and I. Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019. URL https://arxiv.org/abs/1908. 10084. [32] O. Rubin, J. Herzig, and J. Berant. Learning to retrieve prompts for in-context learning. arXiv preprint arXiv:2112.08633, 2021. [33] E. Saravia, H.-C. T. Liu, Y.-H. Huang, J. Wu, and Y.-S. Chen. CARER: Contextualized affect representations for emotion recognition. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3687–3697, Brussels, Belgium, Oct.-Nov. 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1404. URL https://www.aclweb.org/anthology/D18-1404. [34] N. Saunshi, S. Malladi, and S. Arora. A mathematical exploration of why language models help solve downstream tasks. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=vVjIW3sEc1s. [35] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Ng, and C. Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631– 1642, Seattle, Washington, USA, Oct. 2013. Association for Computational Linguistics. URL https://aclanthology.org/D13-1170. [36] N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020. [37] H. Su, J. Kasai, C. H. Wu, W. Shi, T. Wang, J. Xin, R. Zhang, M. Ostendorf, L. Zettlemoyer, N. A. Smith, et al. Selective annotation makes language models better few-shot learners. arXiv preprint arXiv:2209.01975, 2022. [38] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. [39] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov,
P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: Open foundation and fine-tuned chat models. 7 2023. URL http://arxiv.org/abs/2307.09288. [40] L. van der Maaten and G. Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(86):2579–2605, 2008. URL http://jmlr.org/papers/v9/vandermaaten08a. html. [41] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. [42] J. von Oswald, E. Niklasson, E. Randazzo, J. Sacramento, A. Mordvintsev, A. Zhmoginov, and M. Vladymyrov. Transformers learn in-context by gradient descent. arXiv preprint arXiv:2212.07677, 2022. [43] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. EMNLP 2018, page 353, 2018. [44] B. Wang and A. Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021. [45] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, and D. Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022. [46] A. Warstadt, A. Singh, and S. R. Bowman. Neural network acceptability judgments. arXiv preprint arXiv:1805.12471, 2018. [47] C. Wei, S. M. Xie, and T. Ma. Why do pretrained language models help in downstream tasks? an analysis of head and prompt tuning. Neural Information Processing Systems (NeurIPS), 2021. [48] J. Wei, X. Wang, D. Schuurmans, M. Bosma, E. Chi, Q. Le, and D. Zhou. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903, 2022. [49] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, et al. Huggingface’s transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771, 2019. [50] S. M. Xie, A. Raghunathan, P. Liang, and T. Ma. An explanation of in-context learning as implicit bayesian inference. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=RdJVFCHjUMI. [51] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022. [52] X. Zhang, J. Zhao, and Y. LeCun. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28, 2015. [53] D. Zhou, N. Schärli, L. Hou, J. Wei, N. Scales, X. Wang, D. Schuurmans, O. Bousquet, Q. Le, and E. Chi. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625, 2022.
# A Proofs
# A.1 Direct direction
Assumption A.1. (Assumption 2.1) Assume that PM(X) = P(X), and P d M(Y |θ, X) ∝P(Y |θ, X) for X �Y �θ. Proposition A.2. (Proposition 2.2) If task d follows the X �Y �θ direction, arg maxy∈Y P d M(Y = y|θd, X) is the Bayes optimal classifier.
arg max y∈Y P d M(Y = y|θd, X) = arg max y∈Y P(Y = y|θd, X).
Thus arg maxy∈Y P d M(Y = y|θd, X) is the Bayes optimal classifier. Theorem A.3. (Theorem 2.3) If task d follows the X �Y �θ direction, then the in-context learning classifier arg max y∈Y P d M(Y = y|Xd 1, Y d 1 , ..., Xd k, Y d k , X)
always has a higher or equal probability of misclassification to the Bayes optimal classifier arg maxy∈Y P d M(Y = y|θd, X). Equality only takes when ∀x ∈X, P d M(θd|Xd 1, Y d 1 , ..., Xd k, Y d k , X = x) = 1.
always has a higher or equal probability of misclassification to the Bayes optimal classifier arg maxy∈Y P d M(Y = y|θd, X). Equality only takes when
Proof. Recall that in Equation (1), we have
By Proposition A.2, arg maxy∈Y P d M(Y = y|θd, X) is the Bayes optimal classifier. Let Cθ(X) = arg maxy∈Y P d M(Y = y|θ, X), then the risk is defined as the probability of misclassification
Such risk is minimized if and only if Ck(X) = Cθd(X), which only holds when P d M(θd|Xd 1, Y d 1 , ..., Xd k, Y d k , X = x) = 1 for all x ∈X.
# A.2 Channel direction

By Proposition A.5, arg maxy∈Y P d M(X|θd, Y = y) is the Bayes optimal classifier. Let Cθ(X) = arg maxy∈Y P d M(X|θ, Y = y), then the risk is defined as the probability of misclassification
Denote the in-context learning classifier arg maxy∈Y P d M(X|Y d 1 , Xd 1, ..., Y d k , Xd k, Y = y) by Ck(X We then have �
Such risk is minimized if and only if Ck(X) = Cθd(X), which only holds when P d M(θd|Y d 1 , Xd 1, ..., Y d k , Xd k, Y = y) = 1 for all y ∈Y.
# A.3 Method
Proposition A.7. (Proposition 3.1) When L(ˆθd) is minimized, P d M(Y |ˆθd, X) = P(Y |θd, X) for X �Y �θ, and P d M(X|ˆθd, Y ) = P(X|θd, Y ) for Y �X �θ. If the LLM M is invertible, then ˆθd = θd.
Proof. The proof of this proposition is straightforward. Since
Proof. The proof of this proposition is straightforward.
L(ˆθd) = H(P(Y |θd, X)) + KL(P(Y |θd, X)||P d M(Y |ˆθd, X))
when L(ˆθd) is minimized, we have P d M(Y |ˆθd, X) = P(Y |θd, X) for X �Y �θ, and P d M(X|ˆθd, Y ) = P(X|θd, Y ) for Y �X �θ. If M is invertible, since the embedding matrix is invertible with or without new concept tokens, P d M(Y |ˆθ, X) = P d M(Y |ˆθ′, X) implies that ˆθ = ˆθ′. Thus θ is identifiable, which means ˆθd = θd.
<div style="text-align: center;">Table 2: Prompt template and label mapping for the datasets we use. Since almost all sentences from ETHOS contain offensive content, we mask out the key offensive words in the examples below.</div>
S contain offensive content, we mask out the key offensive words in the examples belo
Dataset
Prompt
Label Mapping
SST-2
sentence: well worth revisiting as many times
positive
negative/positive
FPB
The company anticipates its turnover for the whole 2010 to
surpass that of the previous year when it was EUR 67.1 million .
positive
negative/neutral/positive
COLA
It is this hat that I know the boy who is wearing.
unacceptable
acceptable/unacceptable
DBPedia
The Nucet River is a tributary of the Chiojdeanca
River in Romania.
NaturalPlace
Album/Animal/Artist/
Athlete/Building/Company/
EducationalInstitution/Film/
MeanOfTransportation/
NaturalPlace/OfficeHolder/
Plant/Village/WrittenWork
EmoC
fast i mean fastingis a way of skipping meals i mena
you move on too fast
others
angry/happy/others/sad
EmoS
i feel this place was tragic
sadness
anger/fear/joy/love/
sadness/surprise
ETHOS-SO
[Masked] should be removed from the face of the earth
true
false/true
ETHOS-R
I hate being a [Masked], wish I was a [Masked]
and no [Masked] on earth existed
false
false/true
# B Experiments
Dateset. In Table 2, we show how we process the text classification datasets into prompts. For each dataset, we take at most 16384 examples from the training set for training, and uniformly sample at most 1000 examples from the test set to test the in-context learning performance. In Table 3, we show the train size and test size we used for each dataset. We also list the set of diverse tasks trained with each dataset, which are denoted by their name in Huggingface datasets.9 The license for SST2, ETHOS-SO and ETHOS-R is GNU General Public License v3. FPB is under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License. Note that these two datasets are hate speech detection datasets for different kinds of hate speech and contain many offensive texts. COLA is excerpted from the published works available on the website, and the copyright (where applicable) remains with the original authors or publishers. DBpedia is under a Creative Commons Attribution-ShareAlike License and the GNU Free Documentation License. EmoC and EmoS should be used for educational and research purposes only. Experiment details. We run our experiments on A100, V100, and A6000 GPUs. We adopt a large portion of the code from the MetaICL repository [25]10. The training takes around 20 to 40 hours on a single GPU. We use a learning rate of 1e-4 and a batch size of 16, and train for 10k steps in total. Main results. In Table 4, we list the detailed results of our method and baselines with different LLMs on different datasets in Figure 2. Causal direction results. The detailed results with anti-causal direction (the opposite direction to what we described in Section 4 are in Table 7) are shown in Table 7, corresponding to Figure 6 in the main text. Other LLMs results. The detailed results with other LLMs are shown in Table 6, corresponding to Figure 3a in the main text. Random token results. The detailed results with random tokens are shown in Table 5, corresponding to Figure 3b in the main text.
9https://huggingface.co/docs/datasets/index 10https://github.com/facebookresearch/MetaICL
datset d
train size
test size
task set S
SST2 (glue-sst2)
16384
1000
glue-cola/glue-mnli/glue-qqp/
glue-mrpc/glue-qnli/glue-rte/glue-sst2/glue-wnli
FPB (financial_phrasebank)
1811
453
glue-sst2/glue-mnli/math_qa/sciq/
social_i_qa/wino_grande/glue-qqp/
ag_news/financial_phrasebank/
poem_sentiment/anli/quarel/quartz/
medical_questions_pairs/paws/dbpedia_14
COLA (cola-sst2)
8551
1000
glue-cola/glue-mnli/glue-qqp/glue-mrpc/
glue-qnli/glue-rte/glue-sst2/glue-wnli
DBpedia (dbpedia_14)
16384
1000
glue-sst2/glue-mnli/math_qa/sciq/
social_i_qa/wino_grande/glue-qqp/
ag_news/financial_phrasebank/
poem_sentiment/anli/quarel/quartz/
medical_questions_pairs/paws/dbpedia_14
EmoC (emo)
16384
1000
glue-sst2/amazon_polarity/
financial_phrasebank/poem_sentiment/
yelp_polarity/glue-cola/blimp/ag_news/
dbpedia_14/ethos/emo/emotion
EmoS (emotion)
16000
1000
glue-sst2/amazon_polarity/
financial_phrasebank/poem_sentiment/
yelp_polarity/glue-cola/blimp/ag_news/
dbpedia_14/ethos/emo/emotion
ETHOS-SO (ethos-sexual_orientation)
346
87
glue-sst2/amazon_polarity/
financial_phrasebank/poem_sentiment/
yelp_polarity/glue-cola/blimp/ag_news/
dbpedia_14/ethos/emo/emotion
ETHOS-R (ethos-religion)
346
87
glue-sst2/amazon_polarity/
financial_phrasebank/poem_sentiment/
yelp_polarity/glue-cola/blimp/ag_news/
dbpedia_14/ethos/emo/emotion
Table 3: Dataset details
Table 3: Dataset details
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f865/f8653116-7b6c-45a9-80d0-2a9ea0155ee5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Accuracy of randomly selected demonstrations averaged over seven different LLMs except or GPT3-davinci, using the adopted causal direction and the anti-causal direction.</div>
k-ablation study results. The detailed results of k ablation study are shown in Table 10, corresponding to Figure 4a in the main text. In this experiment, we do not reorder the selected demonstrations according to Equation (3), as we need to use GPT2-large for the reordering, and it cannot fit in all the demonstrations. Instead, we order the selected demonstrations from the largest ˆP d M(θd|Xd, Y d) to the smallest. c-ablation study results. The detailed results of c ablation study are shown in Table 11, corresponding to Figure 4b in the main text.
Effect of using ground truth labels. According to [26], the ground truth label is not necessary for demonstrations to have a good in-context learning performance, which we found is not entirely true for all the tasks. We compare our method with the randomly selected demonstration baseline under three scenarios: (a) Original: demonstrations with the correct labels; (b) Random words: using a random label projection map τ d instead of a meaningful one. i.e., map each label to a fixed random word. In this case, the mapping from the input tokens X to the labels Y is still preserved; (c) Random labels: assign a random label to each demonstration, with the original label projection map τ d. As shown in Figure 7, by using a random label projection map or randomly assigning the labels, the performance of the randomly selected demonstration baseline drops considerably. And randomize the label assignment gives a larger performance drop than only using a random label projection map,
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1c68/1c68777f-9547-40f1-8d2a-886470efdaa6.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: In-context learning accuracy of our method versus random selection baseline, with (a) ground truth labels (original), (b) random label mapping (random words), or random label assignments (random label), averaged over all eight datasets. Numbers are obtained with GPT2-large.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/346f/346fb672-d877-406e-be03-2ff585b34ba0.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: Accuracy of in-context learning using our method versus the theoretical maximum accuracy obtained using the learned concept tokens as prefixes. Numbers are obtained with GPT2-large.</div>
which shows that the mapping between X and Y in the demonstrations matters. This indicates that in-context learning infers the mapping between X and Y from the demonstrations instead of merely invoking some learned function stored in the LLM parameters based on the appearance of X and Y . We also show that the demonstrations selected by our method represent the X −Y mapping better, as under the Random words condition, our method performs better than the random selection baseline, while our method does not improve the random selection baseline under the Random labels condition. The detailed results with random words and random labels are shown in Table 8 Optimal performance As stated in Theorem 2.3, the optimal performance of an in-context learning classifier is the Bayes optimal classifier arg maxy∈Y P d M(Y = y|θd, X), which is approximated by using the learned concept tokens as prefixes. Note that this approximated Bayes optimal classifier cannot be transferred across different LLMs, as the learned concept tokens embeddings are aligned with a specific LLM. The advantage of in-context learning with our method is that the demonstrations can be transferred to any LLMs without training. Here we only compare the accuracy of in-context learning with our method and the approximated Bayes optimal classifier using GPT2-large, as it is the LLM that concept tokens are fine-tuned with. As shown in Figure 8, our method comes close to the optimal accuracy on many datasets, while there are some datasets that our method is lagging. This indicates that there are two ways to improve our method: the first is to improve the performance of the optimal classifier, by introducing a better latent concept learning algorithm. The other way is to reduce the performance gap between our method and the optimal classifier, by improving the demonstration selection algorithm. The detailed results using the learned concept tokens as prefixes are shown in Table 9. Reordering results. We reorder the selected demonstrations to maximize the posterior of the concept tokens:
Where π((Xd 1, Y d 1 ), ..., (Xd k, Y d k )) is a permutation of (Xd 1, Y d 1 ), ..., (Xd k, Y d k ). Π is the set of all possible permutations of the k demonstrations. The detailed results with and without reordering are shown in Table 12, corresponding to Figure 9. Similar tokens. We show the top ten similar tokens to some learned concept tokens in Table 13, as summarized in Figure 5 in the main text.
(3)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/23f2/23f2aa16-f773-44e1-a758-7179577dfbfa.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: In-context learning accuracy of our method versus random selection baseline, with an without reordering. The red error bars represent the standard deviation across five runs. Numbers a obtained with GPT2-large.</div>
GPT3s.
LLM
Method
SST2
FPB
COLA
DBpedia
EmoC
EmoS
ETHOS-SO
ETHOS-R
Avg
GPT2
Uniform
69.7 ± 1.8
52.9 ± 2.3
61.9 ± 1.4
48.0 ± 0.7
35.3 ± 1.7
26.4 ± 1.0
64.1 ± 4.8
71.0 ± 1.8
53.7
(124M)
Similar
69.5 ± 0.6
55.9 ± 1.7
63.2 ± 1.2
44.7 ± 3.1
36.4 ± 2.0
26.6 ± 1.3
77.7 ± 2.7
80.0 ± 3.7
56.8
Ours
76.8 ± 2.9
64.5 ± 3.2
69.1 ± 0.2
53.5 ± 2.95
37.2 ± 11.1
30.6 ± 4.8
80.9 ± 1.9
76.8 ± 2.6
61.2
GPT2-m
Uniform
70.8 ± 1.3
52.0 ± 1.7
57.8 ± 1.3
49.3 ± 2.0
34.2 ± 1.8
34.2 ± 1.8
76.3 ± 4.9
74.7 ± 2.2
56.2
(355M)
Similar
75.0 ± 1.9
57.7 ± 2.0
57.5 ± 2.2
47.9 ± 6.0
37.2 ± 3.6
35.2 ± 1.8
86.9 ± 2.9
84.6 ± 4.3
60.3
Ours
81.2 ± 1.3
59.3 ± 4.3
69.0 ± 0.2
52.9 ± 2.3
40.4 ± 21.5
37.2 ± 2.4
83.7 ± 1.1
76.8 ± 1.1
62.6
GPT2-l
Uniform
77.1 ± 1.2
51.3 ± 2.4
62.7 ± 0.8
54.4 ± 0.9
38.7 ± 2.1
34.5 ± 1.2
67.6 ± 4.3
72.9 ± 2.8
57.4
(774M)
Similar
80.7 ± 1.6
54.8 ± 3.8
50.9 ± 1.4
51.1 ± 5.2
39.9 ± 2.6
35.1 ± 2.1
80.9 ± 2.8
84.4 ± 2.6
59.7
Ours
86.2 ± 1.4
60.4 ± 2.5
69.1 ± 0.2
56.5 ± 3.2
48.4 ± 17.0
38.6 ± 2.8
82.5 ± 1.5
76.6 ± 1.2
64.8
GPT2-xl
Uniform
74.7 ± 0.9
53.2 ± 1.9
55.8 ± 1.6
53.0 ± 1.9
38.2 ± 1.5
38.2 ± 1.5
67.8 ± 6.4
72.6 ± 4.1
56.7
(1.5B)
Similar
80.6 ± 1.3
53.0 ± 2.5
55.0 ± 2.5
51.6 ± 5.9
39.9 ± 2.0
32.9 ± 2.1
82.8 ± 2.2
83.9 ± 4.5
60
Ours
83.1 ± 3.6
62.0 ± 2.5
68.9 ± 0.2
58.6 ± 3.3
43.6 ± 16.4
43.6 ± 16.4
83.0 ± 1.3
77.9 ± 1.3
65.1
GPT3-a
Uniform
76.9 ± 0.7
56.6 ± 1.1
53.1 ± 1.8
62.1 ± 1.4
38.6 ± 1.4
27.7 ± 1.3
65.5 ± 5.7
74.0 ± 3.0
56.8
(350M)
Similar
78.7 ± 1.0
52.2 ± 2.7
53.1 ± 1.8
54.6 ± 1.7
42.4 ± 3.5
37.2 ± 1.1
84.1 ± 2.2
87.8 ± 3.5
61.3
Ours
85.4 ± 1.7
61.9 ± 10.5
58.2 ± 7.0
64.0 ± 4.4
43.0 ± 7.2
37.9 ± 2.3
84.4 ± 1.4
78.9 ± 0.9
64.2
GPT3-b
Uniform
80.8 ± 0.6
55.2 ± 3.3
46.8 ± 2.0
66.5 ± 1.4
42.0 ± 0.7
27.0 ± 1.2
71.0 ± 4.6
72.6 ± 3.1
57.7
(1.3B)
Similar
83.9 ± 1.3
56.2 ± 2.3
45.1 ± 1.8
59.8 ± 1.8
42.9 ± 3.5
38.1 ± 1.7
86.7 ± 3.0
86.4 ± 3.0
62.4
Ours
87.3 ± 2.0
64.3 ± 5.9
67.2 ± 0.9
70.2 ± 3.2
43.6 ± 13.0
38.9 ± 5.0
84.6 ± 0.9
78.9 ± 1.2
66.9
GPT3-c
Uniform
84.2 ± 1.4
52.6 ± 1.8
59.1 ± 1.5
70.6 ± 0.8
44.3 ± 2.5
32.3 ± 1.9
77.5 ± 4.7
77.5 ± 0.6
62.3
(6.7B)
Similar
85.7 ± 1.4
62.2 ± 0.9
58.0 ± 1.7
62.2 ± 2.0
47.4 ± 4.3
39.8 ± 1.7
89.2 ± 1.4
89.7 ± 1.9
66.8
Ours
88.8 ± 0.7
64.1 ± 5.7
69.0 ± 0.3
73.6 ± 2.9
50.3 ± 11.9
43.1 ± 4.6
86.2 ± 0.0
78.2 ± 0.0
69.2
GPT3-d
Uniform
86.5 ± 0.9
59.2 ± 2.4
45.5 ± 2.8
73.6 ± 1.9
39.4 ± 0.7
40.6 ± 1.7
77.2 ± 2.6
76.8 ± 3.5
62.4
(175B)
Similar
88.5 ± 0.8
55.4 ± 3.3
45.4 ± 1.5
67.2 ± 1.8
37.6 ± 1.6
39.8 ± 1.4
86.9 ± 2.4
89.0 ± 3.8
63.7
Ours
87.8 ± 3.4
62.7 ± 3.3
58.5 ± 8.2
75.5 ± 2.4
41.3 ± 3.6
42.7 ± 3.9
85.1 ± 0.0
79.3 ± 0.0
66.6
Avg
Uniform
77.6
54.1
55.3
59.7
38.8
32.6
70.9
74.0
57.9
Similar
80.3
55.9
53.5
54.9
40.5
35.6
84.4
85.7
61.4
Ours
84.6
62.4
66.1
63.1
43.5
39.1
83.8
77.9
65.0
Likelihood histogram. We also show histograms of the probability of each example predicting corresponding concept tokens in different datasets. We can see that the probability of prediction concept tokens can well differentiate examples in a dataset. Selected demonstrations. Table 14 shows the selected top 4 demonstration by our proposed algorithm.
GSM8K
Question:
It takes Bryan 5 minutes to walk from his house
to the bus station.
Then he rides the bus for 20 minutes.
After that, he walks 5 minutes from the bus station to his
job.
It takes the same amount of time in the morning and
the evening.
How many hours per year does Bryan spend
traveling to and from work, if he works every day?
Bryan
spends 5+20+5 =«5+20+5=30»30 minutes traveling to work.
He
does this twice a day, so he spends 30*2=«30*2=60»60 minutes
traveling per day.
Thus, he spends 60/60=«60/60=1»1 hour
traveling to and from work every day.
Thus, he spends 1
hour*365 days =«1*365=365»365 hours traveling per year.
The
answer is:
365
GSM8K
Question:
Cherry put up a delivery service.
She charges
$2.50 for a 3-5 kilograms cargo and $4 for a 6-8 kilograms
cargo.
If she delivers four 5 kilograms cargo and two
8 kilograms cargo per day, how much money will she earn
in a week?
Cherry earns $2.50 x 4= $«2.5*4=10»10 after
delivering four 5 kilograms of cargo per day.
She earns $4
x 2= $«4*2=8»8 after delivering two 8 kilograms of cargo per
day.
So, her total earning per day is $8+$10= $«8+10=18»18.
Therefore, she will earn $18 x 7= $«18*7=126»126 in a week.
The answer is:
126
GSM8K
Question:
Bill is laying power cable for a new neighborhood.
There are going to be 18 east-west streets that are 2 miles
long and 10 north-south streets that are four miles long.
It takes 5 miles of cable to electrify 1 mile of street.
If
cable costs $2000/mile, what is the total cost of cable
for the neighborhood?
First find the total distance
of the east-west streets:
18 streets * 2 miles/street
= «18*2=36»36 miles.
Then find the total distance of
the north-south streets:
10 streets * 4 miles/street =
«10*4=40»40 miles.
Then add the number of miles from each
type of street to find the total distance:
36 miles + 40
miles = «36+40=76»76 miles.
Then multiply that number by
5 to find the number of miles of cable needed:
76 miles
street * 5 miles cable/mile street = «76*5=380»380 miles of
cable.
Then multiply that number by the cost of one mile
of cable to find the total cost:
380 miles * $2000/mile =
$«380*2000=760000»760,000.
The answer is:
760000
GSM8K
Question:
John buys a gaming PC for $1200.
He decides to
replace the video card in it.
He sells the old card for
$300 and buys a new one for $500.
How much money did he
spend on his computer, counting the savings from selling
the old card?
He spent an extra 500-300=$«500-300=200»200
on the video card.
That means the total cost was
1200+200=$«1200+200=1400»1400.
The answer is:
1400
SST2
sentence:
faced and spindly attempt at playing an ingenue
makes her nomination as best actress even more of a an a
positive
SST2
sentence:
holofcener’s film offers just enough insight to
keep it from being simpleminded, and positive
SST2
sentence:
i’m not a fan of the phrase ‘ life affirming’
because it usually means ‘ schmaltzy,’ but real women have
curves truly is life affirming negative
SST2
sentence:
the script is about as interesting as a recording
of conversations at the wal-mart checkout line negative
DBpedia
OfficeHolder Lucie Papin (born September 7 1936) is a former
Canadian politician who served in both the House of Commons
and Senate.
DBpedia
Village Kunkalamarru is very renowned village under
Karamchedu Mandal which is located about 15 km from the
busy commercial town of Chirala in Prakasam district in the
state of Andhra Pradesh India.Its neighbouring villages are
Karamchedu Veerannapalem.
DBpedia
EducationalInstitution The Pontifical Catholic University
of Puerto Rico at Mayagez is a university located in the
city of Mayagez Puerto Rico.
It is part of the Pontifical
Catholic University of