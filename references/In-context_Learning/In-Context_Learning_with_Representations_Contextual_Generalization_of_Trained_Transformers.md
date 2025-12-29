# In-Context Learning with Representations: Contextual Generalization of Trained Transformers 
Tong Yang ∗ CMU Yu Huang † UPenn Yingbin Liang ‡ OSU Yuejie Chi § CMU 
September 27, 2024 
# September 27, 2024 
# Abstract 
In-context learning (ICL) refers to a remarkable capability of pretrained large language models, which can learn a new task given a few examples during inference. However, theoretical understanding of ICL is largely under-explored, particularly whether transformers can be trained to generalize to unseen examples in a prompt, which will require the model to acquire contextual knowledge of the prompt for generalization. This paper investigates the training dynamics of transformers by gradient descent through the lens of non-linear regression tasks. The contextual generalization here can be attained via learning the template function for each task in-context, where all template functions lie in a linear space with m basis functions. We analyze the training dynamics of one-layer multi-head transformers to in-contextly predict unlabeled inputs given partially labeled prompts, where the labels contain Gaussian noise and the number of examples in each prompt are not sufficient to determine the template. Under mild assumptions, we show that the training loss for a one-layer multi-head transformer converges linearly to a global minimum. Moreover, the transformer effectively learns to perform ridge regression over the basis functions. To our knowledge, this study is the first provable demonstration that transformers can learn contextual (i.e., template) information to generalize to both unseen examples and tasks when prompts contain only a small number of query-answer pairs. words: transformers, in-context learning, contextual generalization 
Keywords: transformers, in-context learning, contextual generalization 
# Contents 
1 Introduction 1.1 Our contributions. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1.2 Related work. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
# 2 Problem Setup 
# 3 Theoretical Analysis 
3 Theoretical Analysis 3.1 Training time convergence. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.2 Inference time performance. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.3 Further interpretation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1 
# 4 Experiments 
# 5 Conclusion 
∗ Department of Electrical and Computer Engineering, Carnegie Mellon University; email: tongyang@andrew.cmu.edu. † Department of Statistics and Data Science, Wharton School, University of Pennsylvania; email: yuh42@wharton.upenn.edu. ‡ Department of Electrical and Computer Engineering, The Ohio State University; email: liang.889@osu.edu. § Department of Electrical and Computer Engineering, Carnegie Mellon University; email: yuejiechi@cmu.edu. 
∗ Department of Electrical and Computer Engineering, Carnegie Mellon University; email: tongyang@andrew.cmu.edu. † Department of Statistics and Data Science, Wharton School, University of Pennsylvania; email: yuh42@wharton.upenn.edu. ‡ Department of Electrical and Computer Engineering, The Ohio State University; email: liang.889@osu.edu. § Department of Electrical and Computer Engineering, Carnegie Mellon University; email: yuejiechi@cmu.edu. 
# 12 
A Proof Preparation A.1 Summary of key notation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . A.2 Auxiliary lemmas. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
# C Proof of Theorem 2 
n 1. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 
# 1 Introduction 
Transformers [Vaswani et al., 2017] have achieved tremendous successes in machine learning, particularly in natural language processing, by introducing self-attention mechanisms that enable models to capture long-range dependencies and contextualized representations. In particular, these self-attention mechanisms endow transformers with remarkable in-context learning (ICL) capabilities, allowing them to adapt to new tasks or domains by simply being prompted with a few examples that demonstrate the desired behavior, without any explicit fine-tuning or updating of the model’s parameters [Brown et al., 2020]. A series of papers have empirically studied the underlying mechanisms behind in-context learning in transformer models [Garg et al., 2022, Von Oswald et al., 2023, Wei et al., 2023, Olsson et al., 2022, Xie et al., 2021, Chen and Zou, 2024, Agarwal et al., 2024], which have shown that transformers can predict unseen examples after being prompted on a few examples. The pioneering work of Garg et al. [2022] showed empirically that transformers can be trained from scratch to perform in-context learning of simple function classes, providing a theoretically tractable in-context learning framework. Following this well-established framework, several works have investigated various properties of in-context learning in transformers. For instance, studies have explored generalization and stability [Li et al., 2023], expressive power [Bai et al., 2023, Akyürek et al., 2022, Giannou et al., 2023], causal structures [Nichani et al., 2024, Edelman et al., 2024], statistical properties [Xie et al., 2021, Jeon et al., 2024], to name a few. In particular, analysis from an optimization perspective can provide valuable insights into how these models acquire and apply knowledge that enable in-context learning. A few works [Huang et al., 2023, Chen et al., 2024, Li et al., 2024, Nichani et al., 2024] thus studied the training dynamics of shallow transformers with softmax attention in order to in-context learn simple tasks such as linear regression [Huang et al., 2023, Chen et al., 2024], binary classification tasks [Li et al., 2024], and causal graphs [Nichani et al., 2024]. Their theoretical analyses illuminated how transformers, given an arbitrary query token, learn to directly apply the answer corresponding to it from the query-answer pairs that appear in each prompt. Therefore, they all require the sequence length of each prompt to be large enough so that all query-answer pairs have been seen in each prompt with sufficiently high probability, whereas practical prompts are often too short to contain many query examples. This suggests that in-context learning can exploit inherent contextual information of the prompt to generalize to unseen examples, which further raise the following intriguing theoretical question: How do transformers learn contextual information from more general function classes to predict unseen examples given prompts that contain only partial examples? Since our paper studies ICL of non-linear function regression, the function mapping (which we also term 
Since our paper studies ICL of non-linear function regression, the function mapping (which we also term as “template”) naturally serves as the “contextual information” that can be learned for generalization to unseen examples. When each prompt contains only a small number of (noisy) examples, the template that generates the labels may be underdetermined, i.e., multiple templates could generate the same labels in the prompt. Such an issue of underdetermination further raises a series of intriguing questions, such as: When the template that generates a prompt is underdetermined, what is the transformer’s preference for choosing the template and how good is such a choice? 
# 1.1 Our contributions 
this paper, we answer the above questions by analyzing the training dynamics of a one-layer transformer th multi-head softmax attention through the lens of non-linear regression tasks. In our setting, the template nction for each task lies in the linear space formed by m nearly-arbitrary basis functions that capture presentation (i.e., features) of data. Our goal is to provide insights on how transformers trained by gradient scent (GD) acquire template information from more general function classes to generalize to unseen amples and tasks when each prompt contains only a small number of query-answer pairs. We summarize r contributions are as follows. • We first establish the convergence guarantee of a one-layer transformer with multi-head softmax attention trained with gradient descent on general non-linear regression in-context learning tasks. We assume each prompt contains only a few (i.e., partial) examples with their Gaussian noisy labels, which are not sufficient to determine the template. Under mild assumptions, we establish that the training loss of the transformer converges at a linear rate. Moreover, by analyzing the limit point of the transformer parameters, we are able to uncover what information about the basic tasks the transformer extracts and memorizes during training in order to perform in-context prediction. • We then analyze the transformer’s behavior at inference time after training, and show that the transformer chooses its generating template by performing ridge regression over the basis functions. We also provide the iteration complexity for pretraining the transformer to reach ε-precision with respect to its choice of the template given an arbitrary prompt at inference time. We further compare the choice of the transformer and the best possible choice over the template class and characterize how the sequence length of each prompt influences the inference time performance of the model. • Under more realistic assumptions, our analysis framework allows us to overcome a handful of assumptions made in previous works such as large prompt length [Huang et al., 2023, Chen et al., 2024, Li et al., 2024, Nichani et al., 2024], orthogonality of data [Huang et al., 2023, Chen et al., 2024, Li et al., 2024, Nichani et al., 2024], restrictive initialization conditions [Chen et al., 2024], special structure of the transformer [Nichani et al., 2024], and mean-field models [Kim and Suzuki, 2024]. Further, the function classes we consider are a generalization of those considered in most theoretical works [Huang et al., 2023, Chen et al., 2024, Li et al., 2024, Wu et al., 2023, Zhang et al., 2023a]. We also highlight the importance of multi-head attention mechanism in this process. To our best knowledge, this is the first work that analyzes how transformers learn contextual (i.e., mplate) information to generalize to unseen examples and tasks when prompts contain only a small number 
In this paper, we answer the above questions by analyzing the training dynamics of a one-layer transformer with multi-head softmax attention through the lens of non-linear regression tasks. In our setting, the template function for each task lies in the linear space formed by m nearly-arbitrary basis functions that capture representation (i.e., features) of data. Our goal is to provide insights on how transformers trained by gradient descent (GD) acquire template information from more general function classes to generalize to unseen examples and tasks when each prompt contains only a small number of query-answer pairs. We summarize our contributions are as follows. 
Reference
nonlinear
attention
multi
head
task
shift
GD
convergence
noisy
data
representation
learning
Wu et al. [2023]
✗
✗
✓
✓
✓
✗
Zhang et al. [2023a]
✗
✗
✓
✓
✓
✗
Huang et al. [2023]
✓
✗
✓
✓
✗
✗
Li et al. [2024]
✓
✗
✓
✓
✓
✗
Chen et al. [2024]
✓
✓
✗
✗
✓
✗
Kim and Suzuki [2024]
✗
✗
✓
✗
✗
✓
Ours
✓
✓
✓
✓
✓
✓
Table 1: Comparisons with existing theoretical works that study the learning dynamics of transformers in ICL. Here, the last column refers to the fact that the response in the regression task is generated by a linearly weighted unknown representation (feature) model. 
# 1.2 Related work 
In-context learning. Recent research has investigated the theoretical underpinnings of transformers’ ICL capabilities from diverse angles. For example, several works focus on explaining the in-context learning of transformers from a Bayesian perspective [Xie et al., 2021, Ahuja et al., 2023, Han et al., 2023, Jiang, 2023, Wang et al., 2023, Wies et al., 2024, Zhang et al., 2023b, Jeon et al., 2024, Hahn and Goyal, 2023]. Li et al. [2023] analyzed the generalization and stability of transformers’ in-context learning. Focusing on the representation theory, Akyürek et al. [2022], Bai et al. [2023] studied the expressive power of transformers on the linear regression task. Akyürek et al. [2022] showed by construction that transformers can represent GD of ridge regression or the closed-form ridge regression solution. Bai et al. [2023] extended Akyürek et al. [2022] and showed that transformers can implement a broad class of standard machine learning algorithms in-context. Dai et al. [2022], Von Oswald et al. [2023] showed transformers could in-context learn to perform GD. More pertinent to our work, Guo et al. [2023] considered an ICL setting very similar to ours, where the label depends on the input through a basis of possibly complex but fixed template functions, composed with a linear function that differs in each prompt. By construction, the optimal ICL algorithm first transforms the inputs by the representation function, and then performs linear ICL on top of the transformed dataset. Guo et al. [2023] showed the existence of transformers that approximately implement such algorithms, whereas our work is from a different perspective, showing that (pre)training the transformer loss by GD will naturally yield a solution with the aforementioned desirable property characterized in Guo et al. [2023]. Training dynamics of transformers performing ICL. A line of work initiated by Garg et al. [2022] aims to understand the ICL ability of transformers from an optimization perspective. [Zhang et al., 2023a, Kim and Suzuki, 2024] analyzed the training dynamics of transformers with linear attention. Huang et al. [2023], Chen et al. [2024], Li et al. [2024] studied the optimization dynamics of one-layer softmax attention transformers performing simple in-context learning tasks, such as linear regression [Huang et al., 2023, Chen et al., 2024] and binary classification [Li et al., 2024]. Among them, Huang et al. [2023] was the first to study the training dynamics of softmax attention, where they gave the convergence results of a one-layer transformer with single-head attention on linear regression tasks, assuming context features come from an orthogonal dictionary and each token in the prompts is drawn from a multinomial distribution. In order to leverage the concentration property inherent to multinomial distributions, they require the sequence length to be much larger than the size of dictionary. Their analysis indicates that the prompt tokens that are the same as the query will have dominating attention weights, which allows the transformer to copy-paste the correct answer from those prompt tokens. Li et al. [2024] studied the training of a one-layer single-head transformer in ICL on binary classification tasks. Same as Huang et al. [2023], they required the data to be pairwise orthogonal, and shared the same copy-paste mechanism as in Huang et al. [2023]. To be precise, a fraction of their context inputs needs to contain the same pattern as the query to guarantee that the total attention weights on contexts matching the query pattern outweigh those on other contexts. Chen et al. [2024] studied the dynamics of gradient flow for training a one-layer multi-head softmax attention model for ICL of multi-task linear regression, where the coefficient matrix has certain spectral properties. They required the sequence length to be sufficiently large [Chen et al., 2024, Assumption 2.1], together with restrictive initialization conditions [Chen et al., 2024, Definition 3.1]. While using the copy-paste analysis framework as in Huang et al. [2023], Li et al. [2024], the attention probability vector in their work is delocalized, so that the attention is spread out to capture the information from similar tokens in regression tasks. Kim and Suzuki [2024] studied the dynamics of Wasserstein gradient flow for training a one-layer transformer with an infinite-dimensional fully-connected layer followed by a linear attention layer for ICL of linear regression, assuming infinite prompt length. Nichani et al. [2024] analyzed the optimization dynamics of a simplified two-layer transformer with gradient descent on in-context learning a latent causal graph. Notation. Boldface small and capital letters denote vectors and matrices, respectively. Sets are denoted with curly capital letters, e.g., W. We let (R d, ∥·∥) denote the d-dimensional real coordinate space equipped with norm ∥·∥. I d is the identity matrix of dimension d. The ℓ p-norm of v is denoted by ∥ v ∥ p, where 1 ≤ p ≤∞, and the spectral norm and the Frobenius norm of a matrix M are denoted by ∥ M ∥ 2 and ∥ M ∥ F, 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4929/49296aa3-a94b-4d45-bfe1-abbb8513ba19.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: The structure of a one-layer transformer with multi-head softmax attention. </div>
respectively. M † stands for the Moore-Penrose pseudoinverse of matrix M, and M:,i stands for its i-th column vector. We let [N] denote {1, . . . , N}, and denote 1 N to represent the all-one vector of length N, and by 0 a vector or a matrix consisting of all 0’s. We allow the application of functions such as exp (·) to vectors or matrices, with the understanding that they are applied in an element-wise manner. We use e i to denote the one-hot vector whose i-th entry is 1 and the other entries are all 0. 
# 2 Problem Setup 
In-context learning with representation. We consider ICL of regression with unknown representation, similar to the setup introduced in Guo et al. [2023]. To begin, let f: R d → R m be a fixed representation map that f (x) = (f 1 (x), · · ·, f m (x)) ⊤ for any x ∈ R d. The map f can be quite general, which can be regarded as a feature extractor that will be learned by the transformer. We assume that each ICL task corresponds to a map λ ⊤ f (·) that lies in the linear span of those m basis functions in f (·), where λ is generated according to the distribution D λ. Thus, for each ICL instance, the (noisy) label of an input v k (∀ k ∈ [K]) is given as 
y k = λ ⊤ (f (v k) + ϵ k), λ ∼D λ, ϵ k i.i.d. ∼N (0, τ I m) 
where τ > 0 is the noise level. The goal of ICL is to form predictions on query x query given in-context labels of the form (1) on a few inputs, known as prompts. In this paper, we use V to denote the dictionary set that contains all K unit-norm distinct tokens, i.e., V: = {v 1, · · ·, v K} ⊂ R d with each token ∥ v k ∥ 2 = 1. We assume that each prompt P = P λ provides the first N tokens (with N ≪ K) and their labels, and is embedded in the following matrix 
where 
is the collection of prompt tokens, and y: = (y 1, · · ·, y N) ⊤ is the prompt label. Given the prompt as the input, the transformer predicts the labels for all the K tokens y 1, · · ·, y K in the dictionary set. 
Transformer architecture. We adopt a one-layer transformer with multi-head softmax attention [Chen et al., 2024] — illustrated in Figure 1— to predict the labels of all the tokens in the dictionary V, where H is the number of heads. Denote the query embedding as 
(1) 
(2) 
(3) 
(4) 
and denote the embedding of both the prompt and the query as E: = (E P, E Q) ∈ R (d +1) × K. We define the output of each transformer head as 
� � j e x j. The attention map of the transformer T (E) is defined as where W Q h ∈ R d e × (d +1), W K h ∈ R d e × (d +1), and W V h ∈ R K × (d +1) are the query, key, and value matrices, respectively, and the softmax is applied column-wisely, i.e., given a vector input x, the i-th entry of softmax (x) is given by e x i / � 
  where W O is the output matrix. Following recent theoretical literature to streamline analysis [Huang et al., 2023, Nichani et al., 2024, Deora et al., 2023, Chen et al., 2024], we assume that the embedding matrices take the following forms: 
  where W O is the output matrix. Following recent theoretical literature to streamline analysis [Huang et al 2023, Nichani et al., 2024, Deora et al., 2023, Chen et al., 2024], we assume that the embedding matrices tak the following forms: 
where w h = (w h, 1, · · ·, w h,K) ⊤ ∈ R K and Q h ∈ R d × d are trainable parameters for all h ∈ [H]. The prediction of the labels is provided by the diagonal entries of T (E), which we denote by � y = (� y 1, · · ·, � y K) ∈ R K. Note that � y k takes the following form under our parameter specification: 
� Training via GD. Let θ = {Q h, w h} H h =1 denote all trainable parameters of T. Let ϵ: = (ϵ 1, · · ·, ϵ K) ∈ R m × K denote the noise matrix. Given training data over ICL instances, the goal of training is to predict labels y k for all v k ∈V. Specifically, we train the transformer using gradient descent (GD) by optimizing the following mean-squared population loss: 
� We apply different learning rates η Q, η w> 0 for updating {Q h} H h =1 and {w h} H h =1, respectively, i.e., at the th () step, we have 
∀ h ∈ [H] : Q (t) h = Q (t − 1) h − η Q ∇ Q h L (θ (t − 1)), w (t) h = w (t − 1) h − η w ∇ w h L (θ (t − 1)), 
where θ (t) = {Q (t) h, w (t) h} H h =1 is the parameter at the t-th step. 
Inference time. At inference time, given a prompt P = P λ with N examples, where λ may not be in the support of the generation distribution D λ, the transformer applies the pretrained parameters and predicts the labels of all K tokens without further parameter updating. 
# 3 Theoretical Analysis 
# 3.1 Training time convergence 
3.1 Training time convergence 
In this section, we show that the training loss L converges to its minimum value at a linear rate during training, i.e., the function gap ∆ (t):= L (θ (t)) − inf θ L → 0, t →∞ (11 
(5) 
(6) 
(7a) (7b) 
(8) 
(9) 
(10) 
(11) 
Key assumptions. We first state our technical assumptions. The first assumption is on the distribution D λ for generating the coefficient vector λ of the representation maps. Assumption 1 (Assumption on distribution D λ). We assume that in (1) each entry λ i is drawn independently and satisfies E [λ i] = 0 and E [λ 2 i] = 1 for all i ∈ [m]. To proceed, we introduce the following notation: Z:= (f (v) · · · f (v)) ∈ R m × N, ¯ Z:= � Z ⊤ Z + mτ I � 1 / 2 ∈ R N × N, ¯ f:= max ∥ ¯ z ∥, (12) 
Before stating our main theorem, let us examine when the initialization condition in Assumption 2 is met. Fortunately, we only require the following mild assumption on V to ensure our parameter initialization has good properties. 
Assumption 3 (Assumption on V). There exists one row vector x = (x 1, · · ·, x N) ⊤ of the prompt toke matrix V (cf. (3)) such that x i ̸ = x j, ∀ i ̸ = j. 
Assumption 3 implies that V has distinct tokens, i.e., v j ̸ = v k when j ̸ = k. It is worth noting that Assumption 3 is the only assumption we have on the dictionary V. In comparison, all other theoretical works in Table 1 impose somewhat unrealistic assumptions on V. For example, Huang et al. [2023], Li et al. [2023], Nichani et al. [2024] assume that the tokens are pairwise orthogonal, which is restrictive since it implies that the dictionary size K should be no larger than the token dimension d, whereas in practice it is often the case that K ≫ d [Reid et al., 2024, Touvron et al., 2023]. In addition, Chen et al. [2024], Zhang et al. [2023a], Wu et al. [2023] assume that each token is independently sampled from some Gaussian distribution, which also does not align with practical scenarios where tokens are from a fixed dictionary and there often exist (strong) correlations between different tokens. The following proposition states that when the number of heads exceeds the number of prompts, i.e. H ≥ N, we can guarantee that Assumption 2 holds with probability 1 by simply initializing {Q h} H h =1 using Gaussian distribution. 
≫ et al. [2023] assume that each token is independently sampled from some Gaussian distribution, which also does not align with practical scenarios where tokens are from a fixed dictionary and there often exist (strong) correlations between different tokens. The following proposition states that when the number of heads exceeds the number of prompts, i.e. H ≥ N, we can guarantee that Assumption 2 holds with probability 1 by simply initializing {Q h} H h =1 using Gaussian distribution. Proposition 1 (Initialization of {Q h} H h =1). Suppose Assumptions 1, 3 hold and H ≥ N. For any fixed β > 0, let Q (0) h (i, j) i.i.d. ∼N (0, β 2), then Assumption 2 holds almost surely. Proof. See Appendix D.1. 
Choice of learning rates. Define 
where ∆ (0) is the initial function gap (c.f. (11)). Assumption 2 indicates that ζ 0> 0. Let γ be any p constant that satisfies � √ � 
We set the learning rates as 
η Q ≤ 1 /L and η w = γ 2 η Q, 
(12) 
(13) 
(14) 
(15) 
(16) 
�� �� �� �� � � Theoretical guarantee. Now we are ready to state our first main result, regarding the training dynamic of the transformer. Theorem 1 (Training time convergence). Suppose Assumptions 1, 2 hold. We let w (0) k = 0 and set the learning rates as in (16). Then we have 
# Proof. See Appendix B. 
Proof. See Appendix B. 
Theorem 1, together with Proposition 1, shows that the training loss converges to its minimum value at a linear rate, under mild assumptions of the task coefficients and token dictionary. This gives the first convergence result for transformers with multi-head softmax attention trained using GD to perform ICL tasks (see Table 1). Our convergence guarantee (18) also indicates that the convergence speed decreases as the size K of the dictionary or the number H of attention heads increases, which is intuitive because training with a larger vocabulary size or number of parameters is more challenging. However, a small H will limit the expressive power of the model (see Section 3.3 for detailed discussion), and we require H ≥ N to guarantee Assumption 2 holds, as stated in Proposition 1. 
# 3.2 Inference time performance 
We now move to examine the inference time performance, where the coefficient vector λ corresponding to the inference task may not drawn from D λ. In fact, we only assume that the coefficient vector λ at inference time is bounded as in the following assumption. Assumption 4 (Boundedness of λ at inference time). We assume that at inference time ∥ λ ∥ 2 ≤ B for some B > 0. For notational simplicity, let Z Q ∈ R m × (K − N) denote 
We now move to examine the inference time performance, where the coefficient vector λ corresponding to the inference task may not drawn from D λ. In fact, we only assume that the coefficient vector λ at inference time is bounded as in the following assumption. 
The following theorem characterizes the performance guarantee of the transformer’s output � y (after sufficient training) at the inference time. Theorem 2 (Inference time performance). Let � λ be the solution to the following ridge regression problem: � � � 
� � � � Under the assumptions in Theorem 1, for any ε > 0 and δ ∈ (0, 1), if the number of training iterates satisfies 
then given any prompt P that satisfies Assumption 4 at the inference time, with probability at least 1 − δ, output of the transformer � y satisfies 
hen given any prompt P that satisfies Assumption 4 at the inference time, with probability at least 1 − δ, the utput of the transformer � y satisfies 
(17) 
(18) 
(19) 
(20) 
(21) 
(22) 
In Theorem 2, (22) shows that after training, the transformer learns to output the given labels of the first N tokens in each prompt, and more importantly, predicts the labels of the rest K − N tokens by implementing the ridge regression given in (20). Note that Akyürek et al. [2022] studied the expressive power of transformers on the linear regression task and showed by construction that transformers can represent the closed-form ridge regression solution. Interestingly, here we show from an optimization perspective that transformers can in fact be trained to do so. 
# Generalization capabilities of the pretrained transformer. Theorem 2 captures two generalization capabilities that the pretrained transformer can have. 
i) Contextual generalization to unseen examples: Theorem 2 suggests that the transformer exploits the inherent contextual information (to be further discussed in Section 3.3) of the function template in the given prompt, and can further use such information to predict the unseen tokens. 
i) Contextual generalization to unseen examples: Theorem 2 suggests that the transformer exploits the inherent contextual information (to be further discussed in Section 3.3) of the function template in the given prompt, and can further use such information to predict the unseen tokens. ii) Generalization to unseen tasks: Theorem 2 also suggests that the pretrained transformer can generalize to a function map corresponding to any λ ∈ R m at the inference time (albeit satisfying Assumption 4) which is not necessarily sampled from the support of its training distribution D λ. 
ii) Generalization to unseen tasks: Theorem 2 also suggests that the pretrained transformer can generaliz to a function map corresponding to any λ ∈ R m at the inference time (albeit satisfying Assumption 4 which is not necessarily sampled from the support of its training distribution D λ. 
We note that the contextual generalization that the transformer has here is different in nature from the prediction ability shown in previous works on ICL [Huang et al., 2023, Chen et al., 2024, Li et al., 2024, Nichani et al., 2024]. Those work focuses on a setting where each prompt contains a good portion of tokens similar to the query token, allowing the transformer to directly use the label of the corresponding answers from the prompt as the prediction. However, in practical scenarios, prompts often contain only partial information, and our analysis sheds lights on explaining how transformers generalize to unseen examples by leveraging ridge regression to infer the underlying template. 
How does the representation dimension affect the performance? Beyond the above discovery, several questions are yet to be explored. For instance, while we demonstrate that transformers can be trained to implement ridge regression, how good is the performance of the ridge regression itself? What is the best choice of ridge regression we could expect? How close is the transformer’s choice to the best possible choice? We address these questions as follows. Given any prompt P at inference time, since there is no label information about the rest K − N tokens, the best prediction we could hope for from the transformer shall be 
� where Z Q is defined in (19), and � λ τ satisfies: 
� 2 N � N i =1 (y i − λ ⊤ (f (v i) + ϵ i)) 2 �. � λ τ:= arg min λ E ˜ ϵ � 1 
� � � � In other words, we hope the transformer outputs the given N labels as they are. For the rest K − N labels, the best we could hope for is that the transformer estimates the coefficient vector λ by solving the above regression problem to obtain � λ τ, and predict the k-th label by � λ ⊤ τ f (v k) for k = N + 1, · · ·, K. Note that (24) is equivalent to the following ridge regression problem (see Lemma 4 in the appendix for its derivation): 
� � � � The only difference between the two ridge regression problems (20) and (25) is the coefficient of the regularization term. This indicates that at the training time, the transformer learns to implement ridge regression to predict the labels of the rest K − N tokens, assuming the noise level is given by m N τ. This observation also reflects how the sequence length N affects the transformer’s preference for choosing templates and its performance at inference time: 
(23) 
(24) 
(25) 
• The closer m is to N, the closer the transformer’s choice of templates is to the best possible choice, and the better the transformer’s prediction will be; • When N < m, the transformer tends to underfit by choosing a λ with small ℓ 2-norm; • When N > m, the transformer tends to overfit since it underestimates the noise level and in turn captures noise in the prediction. 
# 3.3 Further interpretation 
We provide more interpretation on our results, which may lead to useful insights into the ICL ability of the transformer. 
We provide more interpretation on our results, which may lead to useful insights into t transformer. 
How does the transformer gain ICL ability with representations? Intuitively speaking, our pretrained transformer gains in-context ability by extracting and memorizing some “inherent information” of all basic function maps f i (i ∈ [m]) during the training. Such information allows it to infer the coefficient vector λ from the provided labels in each prompt and calculate the inner product ⟨ λ, f (v k) ⟩ to compute y k given any token v k ∈V at inference time. To be more specific, the “inherent information” of all basic tasks could be described by the N-byK matrix A defined as follows (see also (34)): 
� � � � � where � Z: = (f (v 1), · · ·, f (v K)) = (Z, Z Q) ∈ R m × K. During training, the transformer learns to approximate A:,k by � H h =1 w h,k softmax (V ⊤ Q h v k) for each k ∈ [K]. To further elaborate, we take a closer look at the special case when the labels do not contain any noise, i.e., τ = 0, and N ≥ m. In this case, A becomes Z † � Z, and given any prompt P = P λ, the coefficient vector λ could be uniquely determined from the provided token-label pairs in the prompt. It is straightforward to verify that the label of each token v k could be represented by the inner product of the given label vector y and the k-th column of Z † � Z, i.e., � � 
� � � � Comparing the above equation with (8), it can be seen that in order to gain the in-context ability, th transformer needs to learn an approximation of Z † � Z:,k by � H h =1 w h,k softmax (V ⊤ Q h v k) for each k ∈ [K]. More generally, in the proof of Theorem 2, we show that 
� comparing which with (8) suggests that a small training error implies that � H h =1 w h,k softmax (V ⊤ Q h v k) i close to A:,k. In fact, this is the necessary and sufficient condition for the training loss to be small. A rigorou argument is provided in Lemma 5. 
The necessity and trade-offs of multi-head attention mechanism. Multi-head attention mechanism is essential in our setting. In fact, it is generally impossible to train a shallow transformer with only one attention head to succeed in the ICL task considered in our paper. This is because, as we have discussed above, the key for the transformer is to approximate A:,k by � H h =1 w h,k softmax (V ⊤ Q h v k) for each k ∈ [K]. If H = 1, the transformer could not approximate each A:,k by w 1,k softmax (V ⊤ Q 1 v k) in general since the entries of the latter vector are either all positive or all negative. In addition, Proposition 1 indicates that when H ≥ N, the weights of the transformer with a simple initialization method satisfy our desired property that is crucial to guarantee the fast linear convergence. However, (18) implies that we should not set H to be too large, since larger H yields slower convergence rate. 
# 4 Experiments 
This section aims to provide some empirical validation to our theoretical findings and verify that some of our results could be generalized to deeper transformers. 
(26) 
(27) 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3094/309411db-67de-46d2-b04f-51bb9bdedc37.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b9d8/b9d8dbb1-4068-4001-8c82-f68fdb5a5e77.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) 1-layer transformer </div>
<div style="text-align: center;">igure 2: Training and inference losses of (a) 1-layer and (b) 4-layer transformers, which validate Theorem 2 s well as the transformer’s contextual generalization to unseen examples and to unseen tasks. </div>
Setup. We conduct experiments on a synthetic dataset, where we randomly generate each token v k and their representation f (v k) from standard Gaussian distribution. We employ both the 1-layer transformer described in Section 2 and a standard 4-layer transformer in Vaswani et al. [2017] with d model = 256 and d ff = 512. We set the training loss to be the population loss defined in (9), and initialize {Q (0) h} h ∈ [H] using standard Gaussian and set {w (0) h} h ∈ [H] to be 0, identical to what is specified in Section 3. We generate λ from standard Gaussian distribution to create the training set with 10000 samples and in-domain test set with 200 samples; we also create an out-of-domain (ood) test set with 200 samples by sampling λ from N (1 m, 4 I m). Given λ, we generate the label y k of token v k using (1), for k ∈ [K]. We train with a batch size 256. All experiments use the Adam optimizer with a learning rate 1 × 10 − 4. Training and inference performance. We set N = 30, K = 200, d = 100, m = 20, and set H to be 64 and 8 for 1-layer and 4-layer transformers, respectively. Figure 2 shows the training and inference losses of both 1-layer and 4-layer transformers, where we measure the inference loss by 1 K ∥ � y − � y ⋆ ∥ 2 2 to validate (22): after sufficient training, the output of the transformer � y converges to � y ⋆. From Figure 2 we can see that for both 1-layer and 4-layer transformers, the three curves have the same descending trend, despite the inference loss on the ood dataset is higher than that on the in-domain dataset. This experiment also shows the transformer’s contextual generalization to unseen examples and to unseen tasks, validating our claim in Section 3.2. Figure 3 plots the performance gap 1 K ��� y ⋆ − � y best �� 2 2 of the one-layer transformer with respect to different N ranging from 50 to 150, when we fix m = 100 and τ = 0. 01. This verifies that the ridge regression implemented by the pretrained transformer has a better performance when m is close to N, again verifying our claim at the end of Section 2. 
Impact of the number of attention heads. We now turn to examine the impact of the number of attention heads. In this experiment, we use the population loss (9), and set the other configurations same as those in Figure 2. Figure 4 shows the training loss curves for different H with respect the iteration number, which validates our claims. From Figure 4, we can see that we need to set H large enough to guarantee the convergence of the training loss. However, setting H too large (H = 400) leads to instability and divergence of the loss. Recall that in Proposition 1, we require H ≥ N to guarantee our convergence results hold. Although this condition may not be necessary, Figure 4 shows that when H < N = 30, the loss stopped descending even when it is far from the minimal value. On the other side, the loss keeps descending when H = 30 (though slowly). We also explore how H affects the training of the 4-layer transformer, as displayed in Figure 5, where we set K = 200 and the configurations other than H are the same as in Figure 3. We fix the wall-clock time to 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/cf11/cf11eab3-2c3f-46ed-b91b-e068422f05eb.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) 4-layer transformer </div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3c34/3c34e07d-069f-4e5c-b4ee-445a50e21382.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: The performance gap 1 K ��� y ⋆ − � y best �� 2 2 with different N when m = 100, which validates that the closer N is to m, the better the transformer’s prediction is. </div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/675e/675ee759-402e-450c-a32f-6ab9f69b99ba.png" style="width: 50%;"></div>
Figure 4: Training losses of the 1-layer transformer with different number of attention heads H, where H should be large enough to guarantee the convergence of the training loss, but setting H too large leads t instability and slower divergence. 
<div style="text-align: center;">Figure 4: Training losses of the 1-layer transformer with different number of attention heads H, where H should be large enough to guarantee the convergence of the training loss, but setting H too large leads t instability and slower divergence. </div>
be 100 seconds and plot the training loss curves with different H. Figure 5 (a) shows the final training and inference losses with respect to H. It reflects that the losses converge faster with smaller H (here the final training loss is the smallest when H = 4). The training curves in Figure 5 (b) corresponding to different H within 100s may provide some explanation to this phenomenon: (i) transformers with larger H could complete less iterations within a fixed amount of time (the curves corresponding to larger H are shorter); (ii) the training loss curves corresponding to large H (H = 32, 64) descend more slowly. This suggests our claim that larger H may yield slower convergence rate is still valid on deeper transformers. Note that unlike the 1-layer transformer, deeper transformers don’t require a large H to guarantee convergence. This is because deeper transformers have better expressive power even when H is small. 
# 5 Conclusion 
We analyze the training dynamics of a one-layer transformer with multi-head softmax attention trained by gradient descent to solve complex non-linear regression tasks using partially labeled prompts. In this setting, the labels contain Gaussian noise, and each prompt may include only a few examples, which are insufficient to determine the underlying template. Our work overcomes several restrictive assumptions made in previous studies and proves that the training loss converges linearly to its minimum value. Furthermore, we analyze the transformer’s strategy for addressing the issue of underdetermination during inference and evaluate its performance by comparing it with the best possible strategy. Our study provides the first analysis of how transformers can acquire contextual (template) information to generalize to unseen examples when prompts contain a limited number of query-answer pairs. 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a8db/a8dbf470-5ab0-4156-9266-00347f82e467.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) final losses vs H </div>
Figure 5: Training losses of a 4-layer transformer with different H, fixing wall-clock time to be 100 s. This experiment shows that unlike 1-layer transformers, deeper transformers don’t require H to be large to guarantee convergence of the loss. 
# Acknowledgement 
The work of T. Yang and Y. Chi is supported in part by the grants NSF CCF-2007911, DMS-2134080 and ONR N00014-19-1-2404. The work of Y. Liang was supported in part by the U.S. National Science Foundation under the grants ECCS-2113860, DMS-2134145 and CNS-2112471. 
# References 
R. Agarwal, A. Singh, L. M. Zhang, B. Bohnet, S. Chan, A. Anand, Z. Abbas, A. Nova, J. D. Co-Reyes, E. Chu, et al. Many-shot in-context learning. arXiv preprint arXiv:2404.11018, 2024. K. Ahuja, M. Panwar, and N. Goyal. In-context learning through the bayesian prism. arXiv preprint arXiv:2306.04891, 2023. E. Akyürek, D. Schuurmans, J. Andreas, T. Ma, and D. Zhou. What learning algorithm is in-context learning? investigations with linear models. arXiv preprint arXiv:2211.15661, 2022. Y. Bai, F. Chen, H. Wang, C. Xiong, and S. Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. Advances in neural information processing systems, 36, 2023. T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. S. Chen, H. Sheen, T. Wang, and Z. Yang. Training dynamics of multi-head softmax attention for in-context learning: Emergence, convergence, and optimality. arXiv preprint arXiv:2402.19442, 2024. X. Chen and D. Zou. What can transformer learn with varying depth? case studies on sequence learning tasks. arXiv preprint arXiv:2404.01601, 2024. D. Dai, Y. Sun, L. Dong, Y. Hao, S. Ma, Z. Sui, and F. Wei. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers. arXiv preprint arXiv:2212.10559, 2022. P. Deora, R. Ghaderi, H. Taheri, and C. Thrampoulidis. On the optimization and generalization of multi-head attention. arXiv preprint arXiv:2310.12680, 2023. B. L. Edelman, S. Goel, S. Kakade, and C. Zhang. Inductive biases and variable creation in self-attention mechanisms. In International Conference on Machine Learning, pages 5793–5831. PMLR, 2022. 
R. Agarwal, A. Singh, L. M. Zhang, B. Bohnet, S. Chan, A. Anand, Z. Abbas, A. Nova, J. D. Co-Reyes, E. Chu, et al. Many-shot in-context learning. arXiv preprint arXiv:2404.11018, 2024. K. Ahuja, M. Panwar, and N. Goyal. In-context learning through the bayesian prism. arXiv preprint arXiv:2306.04891, 2023. E. Akyürek, D. Schuurmans, J. Andreas, T. Ma, and D. Zhou. What learning algorithm is in-context learning? investigations with linear models. arXiv preprint arXiv:2211.15661, 2022. Y. Bai, F. Chen, H. Wang, C. Xiong, and S. Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. Advances in neural information processing systems, 36, 2023. T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. S. Chen, H. Sheen, T. Wang, and Z. Yang. Training dynamics of multi-head softmax attention for in-context learning: Emergence, convergence, and optimality. arXiv preprint arXiv:2402.19442, 2024. X. Chen and D. Zou. What can transformer learn with varying depth? case studies on sequence learning tasks. arXiv preprint arXiv:2404.01601, 2024. D. Dai, Y. Sun, L. Dong, Y. Hao, S. Ma, Z. Sui, and F. Wei. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers. arXiv preprint arXiv:2212.10559, 2022. P. Deora, R. Ghaderi, H. Taheri, and C. Thrampoulidis. On the optimization and generalization of multi-head attention. arXiv preprint arXiv:2310.12680, 2023. B. L. Edelman, S. Goel, S. Kakade, and C. Zhang. Inductive biases and variable creation in self-attention mechanisms. In International Conference on Machine Learning, pages 5793–5831. PMLR, 2022. 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/24a7/24a76bbf-ffcc-45a5-8a57-6a9c0c821b18.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) training loss curves for different H </div>
B. L. Edelman, E. Edelman, S. Goel, E. Malach, and N. Tsilivis. The evolution of statistical induction heads: In-context learning markov chains. arXiv preprint arXiv:2402.11004, 2024. S. Garg, D. Tsipras, P. S. Liang, and G. Valiant. What can transformers learn in-context? a case study of simple function classes. Advances in Neural Information Processing Systems, 35:30583–30598, 2022. A. Giannou, S. Rajput, J.-y. Sohn, K. Lee, J. D. Lee, and D. Papailiopoulos. Looped transformers as programmable computers. In International Conference on Machine Learning, pages 11398–11442. PMLR, 2023. T. Guo, W. Hu, S. Mei, H. Wang, C. Xiong, S. Savarese, and Y. Bai. How do transformers learn in-context beyond simple functions? a case study on learning with representations. arXiv preprint arXiv:2310.10616, 2023. M. Hahn and N. Goyal. A theory of emergent in-context learning as implicit structure induction. arXiv preprint arXiv:2303.07971, 2023. C. Han, Z. Wang, H. Zhao, and H. Ji. In-context learning of large language models explained as kernel regression. arXiv preprint arXiv:2305.12766, 2023. Y. Huang, Y. Cheng, and Y. Liang. In-context convergence of transformers. arXiv preprint arXiv:2310.05249, 2023. H. J. Jeon, J. D. Lee, Q. Lei, and B. Van Roy. An information-theoretic analysis of in-context learning. arXiv preprint arXiv:2401.15530, 2024. H. Jiang. A latent space theory for emergent abilities in large language models. arXiv preprint arXiv:2304.09960, 2023. H. Karimi, J. Nutini, and M. Schmidt. Linear convergence of gradient and proximal-gradient methods under the Polyak-Łojasiewicz condition. In European Conference on Machine Learning and Knowledge Discovery in Databases, pages 795–811, 2016. J. Kim and T. Suzuki. Transformers learn nonlinear features in context: Nonconvex mean-field dynamics on the attention landscape. In Forty-first International Conference on Machine Learning, 2024. B. Laurent and P. Massart. Adaptive estimation of a quadratic functional by model selection. Annals of statistics, pages 1302–1338, 2000. H. Li, M. Wang, S. Lu, X. Cui, and P.-Y. Chen. Training nonlinear transformers for efficient in-context learning: A theoretical learning and generalization analysis. arXiv preprint arXiv:2402.15607, 2024. Y. Li, M. E. Ildiz, D. Papailiopoulos, and S. Oymak. Transformers as algorithms: Generalization and stability in in-context learning. In International Conference on Machine Learning, pages 19565–19594. PMLR, 2023. Q. N. Nguyen and M. Mondelli. Global convergence of deep networks with one wide layer followed by pyramidal topology. Advances in Neural Information Processing Systems, 33:11961–11972, 2020. E. Nichani, A. Damian, and J. D. Lee. How transformers learn causal structure with gradient descent. arXiv preprint arXiv:2402.14735, 2024. C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, B. Mann, A. Askell, Y. Bai, A. Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022. M. Reid, N. Savinov, D. Teplyashin, D. Lepikhin, T. Lillicrap, J.-b. Alayrac, R. Soricut, A. Lazaridou, O. Firat, J. Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. J. Von Oswald, E. Niklasson, E. Randazzo, J. Sacramento, A. Mordvintsev, A. Zhmoginov, and M. Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pages 35151–35174. PMLR, 2023. X. Wang, W. Zhu, M. Saxon, M. Steyvers, and W. Y. Wang. Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning. In Workshop on Efficient Systems for Foundation Models@ ICML2023, 2023. J. Wei, J. Wei, Y. Tay, D. Tran, A. Webson, Y. Lu, X. Chen, H. Liu, D. Huang, D. Zhou, et al. Larger language models do in-context learning differently. arXiv preprint arXiv:2303.03846, 2023. N. Wies, Y. Levine, and A. Shashua. The learnability of in-context learning. Advances in Neural Information Processing Systems, 36, 2024. J. Wu, D. Zou, Z. Chen, V. Braverman, Q. Gu, and P. L. Bartlett. How many pretraining tasks are needed for in-context learning of linear regression? arXiv preprint arXiv:2310.08391, 2023. S. M. Xie, A. Raghunathan, P. Liang, and T. Ma. An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080, 2021. R. Zhang, S. Frei, and P. L. Bartlett. Trained transformers learn linear models in-context. arXiv preprint arXiv:2306.09927, 2023a. Y. Zhang, F. Zhang, Z. Yang, and Z. Wang. What and how does in-context learning learn? bayesian model averaging, parameterization, and generalization. arXiv preprint arXiv:2305.19420, 2023b. 
# A Proof Preparation 
<div style="text-align: center;">A.1 Summary of key notation We summarize the frequently used notation in Table 2 for ease of reference </div>
notation
meaning
K ∈N+
total number of tokens
d ∈N+
token dimension
m ∈N+
number of basic tasks
H ∈N+
number of attention heads
N ∈N+
number of examples in each prompt
vk ∈Rd, k ∈[K]
the k-th token
fi : Rd →R, i ∈[m]
the i-th basic task
λ ∈Rm
coefficient vector
yk = λ⊤(f(vk) + ϵk), k ∈[K]
the k-th label
where s h jk is defined in (28). 
Proof. See Corollary A.7 in Edelman et al. [2022]. We also need to make use of the following form of Young’s inequality. Lemma 3. For any x 1, · · ·, x l ∈ R p, we have 
(28) 
(29) 
(30) 
(31) 
Lemma 4 (Equivalence of the regression problems). Given any prompt P λ: = (v 1, y 1, · · ·, v N, y N), the following equivalence: 
� Proof. See Appendix D.2. 
# B Proof of Theorem 1 
We first outline the proof. To prove Theorem 1, we first remove the expectation in the expression of the loss function L in (9) by reformulating it to a deterministic form (see Lemma 5). With this new form, we show by induction that the loss function L is smooth (Lemma 10) and satisfies the Polyak-Łojasiewicz (PL) condition (c.f. (49)). Provided with both smoothness and PL conditions, we are able to give the desired linear convergence rate [Karimi et al., 2016]. We define � 
We also define the following matrices: � 
� where � Z:= (z 1, · · ·, z K) ∈ R m × K. We first reformulate the loss function to remove the expectation in the population loss. Lemma 5 (Reformulation of the loss function). Under Assumption 1, the loss function L (θ) could rewritten into the following equivalent form: 
where 
� �� � � � is a constant that does not depend on θ, and ¯ Z is defined in (12). Proof. See Appendix D.3. Lemma 5 indicates that L ⋆ is a lower bound of L. We’ll later show that L ⋆ is actually the infimum of L, i.e., L ⋆ = inf θ L (θ). Lemma 5 also indicates that, the necessary and sufficient condition for L (θ (t)) to converge to L ⋆ during training is 
� To simplify the analysis, we introduce the following reparameterization to unify the learning rates of all parameters, and we’ll consider the losses after reparameterization in the subsequent proofs. 
(33) 
(34) 
(36) 
(37) 
(38) 
and let 
We denote α as α:= (α h,k) h ∈ [H],k ∈ [K] ∈ R H × K. The following lemma bounds the gradient norms by the loss function, which is crucial to the proof o Theorem 1. � Lemma 7 (Upper bound of the gradient norms). Suppose Assumption 1 holds and | α (t) h,k | ≤ α. Then for a h ∈ [H], we have �� �� 
where 
� � where the first equality follows from Lemma 5, C k, B k is defined in (13). Let b h k denote the h-th column vector of B k, h ∈ [H], i.e., B k: = (b 1 k, · · ·, b H k), then for any k ∈ [K] and t ∈ N +, we have ��� (b h k) (t) − (b h k) (0) ��� 2 ≤ �� ¯ Z �� 2 ��� (s h k) (t) − (s h k) (0) ��� 2 
� � where the first equality follows from Lemma 5, C k, B k is defined in (13). Let b h k denote the h-th column vector of B k, h ∈ [H], i.e., B k: = (b 1 k, · · ·, b H k), then for any k ∈ [K] an t ∈ N +, we have 
(40) 
(42) 
(44) 
(45) 
where the third line uses Lemma 2, and that 
� where the second inequality follows from Lemma 7 (cf. (42)) and the third inequality follows from the inductive hypothesis and the fact that ℓ (ξ (s)) = L (θ (s)), ∀ s. Combining (47) with (46), we have 
� �� here the last inequality follows from (15). The above inequality (48) indicates that ζ 0 / 2 ∀ x ∈ R K: ��� x ⊤ B (t) k ��� 2 ≥ ��� x ⊤ B (0) k ��� 2 − ��� x ⊤ (B (t) k − B (0) k) ��� 2 ≥ � 
��� ��� ��� ��� which gives (43b). Therefore, we obtain the following PL condition: 
which gives (43b). 
�� �� where the equality comes from (45), and the last equality follows from (36). 
Step 2: verify the smoothness of the loss function. We first give the following lemma that bounds the Lipschitzness of b h k and δ θ k, which will be used later on. For notation simplicity, we let B, Q, α denote B (θ), Q (θ), α (θ), respectively, and let B ′, Q ′, α ′ denote B (θ ′), Q (θ ′), α (θ ′), respectively. Lemma 8 (Lipschitzness of b h k and δ θ k). For all k ∈ [K] and h ∈ [H], and all transformer parameters θ, θ ′, if max {| α h,k |, | α ′ h,k |} ≤ α, then we have � � � � 
(46) 
(47) 
(48) 
(49) 
(50) 
(50) 
(51) 
� where we use (46) again to bound the first term in the second line, and use the fact that �� s h k (θ ′) �� 2 ≤ 1 and Cauchy-Schwarz inequality to bound the second term in the second line. We also need the following lemma which bounds the norm of B k and δ θ k. Lemma 9 (Upper bounds of b h k and δ θ k). For all k ∈ [K] and h ∈ [H], if max {| α h,k |, | α ′ h,k |} ≤ α, then we have 
�� b h k �� 2 ≤ �� ¯ Z �� 2, �� δ θ k �� 2 ≤ γHα + ∥ A ∥ 2, 
� � � � � � � � H � �� δ θ k �� 2 ≤ γ h =1 | α h,k | �� s h k �� 2 + ∥ Ae k ∥ 2 ≤ γHα + ∥ A ∥ 2. 
(52) (53) 
(54) 
(55) 
where the first inequality uses Young’s inequality (c.f. Lemma 3). To obtain the smoothness of the loss function w.r.t. Q h, we first note that by (82) we ha 
Therefore, if max {| α h,k |, | α ′ h,k |} ≤ α, we have 
�� �� ��� ��� where the third inequality uses Cauchy-Schwarz inequality. Combining the above inequality (57) Lemma 8 and Lemma 9, we have 
�� �� ��� ��� ere the third inequality uses Cauchy-Schwarz inequality. Combining the above inequality (57) with mma 8 and Lemma 9, we have 
where the last line uses (46) to bound �� s h k (θ) − s h k (θ ′) �� 2. The above inequality (58) further gives 
�� � �� �� � where the first inequality makes use of Young’s inequality (c.f. Lemma 3). Combining the above two relations (55) and (59), we obtain the smoothness of ℓ w.r.t. ξ as follows: 
(57) 
(58) 
(59) 
Lemma 10 (Smoothness of the loss function). Let γ: = � η w /η Q. For all transformer parameters ξ, ξ ′, i max {| α h,k |, | α ′ h,k |} ≤ α, then we have 
Lemma 10 (Smoothness of the loss function). Let γ: = � η w /η Q. For all transformer parameters ξ, ξ ′, if max {| α h,k |, | α ′ h,k |} ≤ α, then we have ∥∇ ξ ℓ (ξ) −∇ ξ ℓ (ξ ′) ∥ 2 ≤ L ∥ ξ − ξ ′ ∥ 2, (60) 
where 
� � Step 3: verify (43a). (45) implies 
which, combining with (52), gives 
Combining this with (36) we obtain 
which indicates 
Therefore, we have 
�� �� �� �� where the second inequality follows from (62) and the third inequality follows from the induction hypothesis (43c). (43a) follows from plugging σ defined in (44) into the above inequality and using the initializtion condition that α (0) = 1 γ w (0) = 0. 
(60) 
(61) 
(62) 
(63) 
�� �� which, combined with the fact that L (θ (s)) = ℓ (ξ (s)) for all s (see Lemma 6), verifies (43c). Note that (36) implies that L ⋆ ≤L (θ) holds for all θ. And from (43c) we know that L (θ (t)) →L t →∞. Therefore, it follows that 
�� �� which, combined with the fact that L (θ (s)) = ℓ (ξ (s)) for all s (see Lemma 6), verifies (43c). Note that (36) implies that L ⋆ ≤L (θ) holds for all θ. And from (43c) we know that L (θ (t)) →L ⋆ as. Therefore, it follows that 
Consequently, (43c) is equivalent to (18). 
# C Proof of Theorem 2 
By (43c) we know that L (θ (t)) →L ⋆ as t →∞. Thus from (36) we know that (37) and (38) hold. By Sherman-Morrison-Woodbury formula, we have 
Thus we have 
� � �� where Z Q is defined in (19). On the other hand, it’s straightforward to verify that � λ defined in (20) admits the following clos 
here Z is defined in (19). On the other hand, it’s straightforward to verify that � λ defined in (20) admits the following closed form 
On the other hand, it’s straightforward to verify that � λ defined in (20) admits the following closed form: 
Combining the above two equations, we obtain � � 
� �� � � � �� where the last equality follows from (22). Now we give the iteration complexity for the mean-squared error between the prediction � y and the limit point � y ⋆ to be less than ε. Given any prompt P = P λ, where λ satisfies Assumption 4, we have 
� �� � � � �� where the last equality follows from (22). Now we give the iteration complexity for the mean-squared error between the prediction � y and the limit point � y ⋆ to be less than ε. Given any prompt P = P λ, where λ satisfies Assumption 4, we have 
Letting x i = y i − λ ⊤ z i ∥ λ ∥ 2 √ τ, we have x i ∼N (0, 1). Define 
Z = 2 − Nτ ∥ λ ∥ 2 2. N � i =1 ∥ λ ∥ 2 2 τ (x 2 i − 1) = �� y − Z ⊤ λ �� 2 
(64) 
(65) 
(66) 
(67) 
� where we use (68) in the second inequality, and the third inequality follows from Assumption On the other hand, by (36) we have 
which gives 
� � ��� �� Thus we know that w.p. at least 1 − δ, we have 
2 1 2 K 2 K 2 ≤ 1 2 ∥ y ∥ 2 2 ≤ ε, 2 K ∥ � y − � y ⋆ ∥ 2 2 = 1 ���� A − A ��� 2 ���� � � A − A � ⊤ y ���� 
�� where the last relation follows from (69), (70) and (21). ��� � 
# D Proof of Key Lemmas 
# D.1 Proof of Proposition 1 
For notation simplicity we drop the superscript (0) in the subsequent proof. Let D k:= � V ⊤ Q 1 v k, · · ·, V ⊤ Q H v k � ∈ R N × H. Note that 
P (x ∈C ( ˜ A) | ˜ A) dµ ( ˜ A)> 0, This suggests the column vectors of D k are i.i.d. and the density of each column vector is positive at any point x ∈R (V), where R (V) ⊂ R N is the row space of V. Since ¯ Z has full rank, to prove B k has full rank a.s., we only need to argue that C k (:, 1 : N) has full rank w.p. 1. Below we prove this by contradiction (recall that by definition C k = softmax (D k), and we assume H ≥ N). Suppose w.p. larger than 0, there exists one of C k (:, 1 : N) ’s column vector that could be linearly represented by its other N − 1 column vectors. Without loss of generality, we assume this colomn vector is C k (:, 1) = softmax (D k (:, 1)). Let x = x (q 1): = exp (D k (:, 1)) = exp (V ⊤ q 1). Then x could be linearly represented by exp(D k (:, i)), i = 2, · · ·, N. Let ˜ A: = exp (D k (:, 2 : N)), then w.p. larger than 0, x ∈C (˜ A), where C (˜ A) is the column vector space of ˜ A. i.e., we have � 
(68) 
(69) 
(70) 
(71) 
which further indicates that there exists ˜ A ∈ R N × (N − 1) such that P (x ∈C (˜ A))> 0. Since the dimension o C ( ˜ A) is at most N − 1, there exists y ∈ R N, y ̸ = 0 such that y ⊥C ( ˜ A). Therefore, we have 
By Assumption 3, without loss of generality, we assume that u 1 = (v 11, v 12, · · ·, v 1 N) ⊤ has different entries. For any vector w = (w 1, · · ·, w d) ⊤ ∈ R d, we let ˜ w = (w 2, · · ·, w d) ⊤ ∈ R d − 1 denote the vector formed by deleting the first entry of w. Let q 1 = (q, ˜ q ⊤ 1) ⊤. For any fixed ˜ q 1 ∈ R d − 1, the function g (·| ˜ q 1) : R → R defined by 
� � � � has finite zero points and thus {q ∈ R | g (q | ˜ q 1) = 0} is a zero-measure set. Therefore, we have 
R d − 1 P (g (q | ˜ q 1) = 0 | ˜ q 1) dµ (˜ q 1) = 0, P (⟨ y, x ⟩ = 0) = � 
which contradicts (72). Therefore, C k (:, 1 : N) has full rank with probability 1. 
# D.2 Proof of Lemma 4 
Lemma 4 can be verified by the following direct computation (recall that the noise in each label satisfies ϵ i i.i.d ∼N (0, τ I m), ∀ i ∈ [N]): 
D.3 Proof of Lemma 5 We let ϵ P: = (ϵ 1, · · ·, ϵ N) ∈ R m × N, ϵ: = (ϵ 1, · · ·, ϵ K) ∈ R m × K. Recall that y = (y 1, · · ·, y N) ⊤ we have y = (Z + ϵ P) ⊤ λ, 
We let ϵ P: = (ϵ 1, · · ·, ϵ N) ∈ R m × N, ϵ: = (ϵ 1, · · ·, ϵ K) ∈ R m × K. Recall that y = (y 1, · · ·, y N) ⊤ ∈ R N. The we have y = (Z + ϵ P) ⊤ λ, (73 
and 
(72) 
(73) 
(74) 
� � � � � � where � a k denote the k-th column vector of matrix � A (θ) defined in (35), and the fifth line uses Assumption 1. Note that for all k ∈ [K], we have 
� � � � � � where � a k denote the k-th column vector of matrix � A (θ) defined in (35), and the fifth line uses Assumption 1. Note that for all k ∈ [K], we have E ϵ (Z � a k − z k) ⊤ (ϵ P � a k − ϵ k) = 0, (76) 
and that 
�� � �� � � � � where 1 {k ∈ [N]} is the indicator function that equals 1 if k ∈ [N] and 0 otherwise, and we have made use of the assumption that ϵ k i.i.d. ∼N (0, τ 2 I m). Combining the above two equations with (75), we know that for k ∈ [N], it holds that 
��� � � � � � � � � �� �� where c k = − � Z ⊤ z k + mτ e k � ⊤ � Z ⊤ Z + mτ I � − 1 � Z ⊤ z k + mτ e k � + ∥ z k ∥ 2 2 + mτ. By a similar argument, we can show that for k ∈ [K] \ [N], it holds thet 
��� � � � � � where c ′ k = − � Z ⊤ z k � ⊤ � Z ⊤ Z + mτ I � − 1 � Z ⊤ z k � + ∥ z k ∥ 2 2. (78), (79) together with (33) and the definition of L ⋆ give (36). 
D.4 Proof of Lemma 6 First, it holds that 
Q (t) h = Q (t − 1) h − η Q ∇ Q h ℓ (ξ (t − 1)) = Q (t − 1) h − η Q ∇ Q h ℓ (ξ (t − 1)). 
Second, note that 
� � Dividing both sides of the above equality by γ, we have α (t) h = α (t − 1) h − η Q ∇ α h ℓ (ξ (t − 1)). Hence, (41) follows from combining (80) and (81). 
(76) 
(77) 
(78) 
(79) 
(80) 
(81) 
Throughout this proof, we omit the superscript (t) for simplicity. We first compute the gradient of L w.r.t. Q. By (36) we know that 
Throughout this proof, we omit the superscript (t) for simplicity. We first compute the gradient of L w.r.t Q h. By (36) we know that 
Throughout this proof, we omit the superscript (t) for simplicity. We first compute the gradient of L w.r.t. 
Note that 
���� where we use the fact that �� (v j − v i) v ⊤ k �� 2 ≤ 2 (recall that we assume each v k has unit norm, k ∈ [K Combining (82) and (83), we have the desired result 
� where ¯ f max is defined in (12) and the third line follows from Cauchy-Schwarz inequality. 
(82) 
(83) 
(84) 
