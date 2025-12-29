# ICLGuard: Controlling In-Context Learning Behavior for Applicability Authorization
Wai Man Si, Michael Backes, Yang Zhang
CISPA Helmholtz Center for Information Security
In-context learning (ICL) is a recent advancement in the capabilities of large language models (LLMs). This feature allows users to perform a new task without updating the model. Concretely, users can address tasks during the inference time by conditioning on a few input-label pair demonstrations along with the test input. It is different than the conventional fine-tuning paradigm and offers more flexibility. However, this capability also introduces potential issues. For example, users may use the model on any data without restriction, such as performing tasks with improper or sensitive content, which might violate the model policy or conflict with the model owner’s interests. As a model owner, it is crucial to establish a mechanism to control the model’s behavior under ICL, depending on the model owner’s requirements for various content. To this end, we introduce the concept of “applicability authorization” tailored for LLMs, particularly for ICL behavior, and propose a simple approach, ICLGuard. It is a fine-tuning framework designed to allow the model owner to regulate ICL behavior on different data. ICLGuard preserves the original LLM and fine-tunes only a minimal set of additional trainable parameters to “guard” the LLM. Empirical results show that the guarded LLM can deactivate its ICL ability on target data without affecting its ICL ability on other data and its general functionality across all data. 2407.06955v1  [cs.CR]  9 Jul 2024
# 1 Introduction
Natural language processing tasks, such as sentence classification and inference, are often solved using a pre-trained model and fine-tuning the model with the downstream dataset. Pre-trained language models like GPT [29] and BERT [9] are trained on a huge amount of data from the internet through self-supervised methods. These models contain extensive knowledge across a wide range of topics, and fine-tuning these models produces remarkable performance on the downstream task. Recently, large language models (LLMs) have demonstrated powerful in-context learning (ICL) capabilities [3,23,38]. This approach shifts the use of LLMs from the traditional fine-tuning procedure to direct application. Users can prompt the LLM with a set of input-label pairs and the test input for performing the desired task like sentence classification and question answering [23, 26, 38].
ICL has an advantage over fine-tuning as it enables LLMs to “learn” new tasks during inference without the need for additional training. It saves human effort and resources while showcasing LLMs’ ability to generalize from a few examples, similar to human capabilities. However, ICL also presents both opportunities and challenges. On one hand, it can adapt to user input with just a few examples and offer tremendous convenience. On the other hand, the user might leverage ICL with content that violates the model policy or conflicts with the model owner’s interests. For instance, teenagers can perform tasks with adult data via ICL, or users can execute tasks with copyrightissued or sensitive data. These scenarios not only raise an accountability risk for the model owner, potentially implicating them in unlawful activities, but also emphasize the importance of regulating ICL behavior. The challenge differs from the issue of toxicity or bias in LMs [11,14,31,32], which usually arises from biased training datasets and can be mitigated by improving the data quality or adversarial training. Neither the ICL ability nor the data used in ICL are part of the LLM’s training objectives or dataset, which raises the question of how such capabilities can be regulated if they are not intentionally trained during the model’s development. Drawing inspiration from [37], we address this concern by determining which data can be used for ICL, leading to the concept of “ICL applicability authorization.” Prior research on authorization related to machine learning falls into two main categories: model usage authorization [2, 5], and applicability authorization [37]. The former ensures that only authorized users can access and deploy the model, and the latter focuses on determining which data can be utilized with the model. Our goal aligns with the objective of applicability authorization. Wang et al. [37] fine-tunes a model optimized for the MNIST dataset with a dedicated patch and becomes ineffective for the USPS dataset, even though both datasets are used for digit recognition. Their research emphasizes permitting only specific data to be used on the model. In contrast, we intend to prohibit the use of ICL on target data while preserving the ability to use ICL on all other (non-target) data. In this paper, we integrate applicability authorization within ICL to regulate ICL behavior for classification tasks on various datasets. To implement ICL applicability authorization on the LLM for classification tasks, we can explicitly retrain the model
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5a96/5a96deb5-60d6-4cc1-96f8-10dac2ea8385.png" style="width: 50%;"></div>
# Figure 1: Overview of ICL applicability authorization
<div style="text-align: center;">Figure 1: Overview of ICL applicability authorization.</div>
to produce incorrect output probabilities (labels) when it encounters target data with demonstrations. Nevertheless, this approach can be computationally expensive and impractical. For instance, the model owner might face multiple requests to adjust their model over time, and it is inconvenient to rebuild the model every time. Therefore, we propose a simple alternative, ICLGuard, a flexible fine-tuning framework for ICL applicability authorization. We first leverage the parameter-efficient fine-tuning (PEFT) method [16, 18, 19]. PEFT is an innovative approach that requires only fine-tuning a small number of extra parameters while freezing the pretrained model. It significantly reduces computational resources and allows the owner to make on-the-fly adjustments to their models. In addition to PEFT, we utilize three different loss functions to optimize the PEFT module to guard the LLM. The first loss, i.e., the disable loss, is designed to deactivate the ICL ability on target data. Next is the maintenance loss, which ensures the ICL ability is unaffected on non-target data. Lastly, the utility loss guarantees that the generated content from the guarded LLM is consistent with the original LLM when handling non-ICL prompts on target and non-target data. In summary, ICLGuard fine-tunes the PEFT module with these loss functions to authorize the ICL applicability while preserving the integrity of the guarded LLM with respect to the original LLM, as shown in Figure 1. Our empirical evaluations over 4 datasets (i.e., FP, SST-2, TREC, and AGnews) and 3 LLMs (i.e., LLaMA, OPT, and Cerebras) illustrate the efficacy of ICLGuard in deactivating the ICL ability on target data without compromising the ICL and regular LM function on all data. We also delve into the influence of each loss in controlling ICL behavior and show their effectiveness. To improve performance further, we explore various strategies for generating data outside the data space of the target data. Then, we investigate the potential of adaptive attacks to bypass the ICL applicability authorization. Furthermore, we study the impact of different setups of ICLGuard and compare it to existing fine-tuning methods. Last, we extend ICLGuard to control the ICL behavior for the generative task. In summary, we make the following contributions:
• We propose ICLGuard, the first fine-tuning framework to control ICL behavior on LLMs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1125/11259ad7-61d0-4cb2-a0ec-df80a75d5b7d.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Example of in-context learning.</div>
• A comprehensive study shows that the guarded LLM effectively controls the ICL functionality on target and non-target data. • Exploration of diverse data generation methods to improve the generalization of the guarded LLM. • A thorough investigation into adaptive attacks aiming to bypass the ICL applicability authorization.
• A comprehensive study shows that the guarded LLM effectively controls the ICL functionality on target and non-target data.
• A thorough investigation into adaptive attacks aiming to bypass the ICL applicability authorization.
# 2 Preliminary 2.1 Language Model
# 2 Preliminary
# 2.1 Language Model
The language model (LM) is an autoregressive model that takes a sequence of tokens as input and generates the next token iteratively. Formally, an LM computes the probability over the sequence x = {w1,w2,··· ,wn}, and it can be computed as the conditional probability distribution p(wn+1|w1,w2,··· ,wn), derived by recursively applying the chain rule:
Before the emergence of transformer architecture, Recurrent Neural Network (RNN) structures, such as GRU [7] and LSTM [15], were predominant in the field of natural language processing. The transformer has since become the prevailing and state-of-the-art neural network architecture, consistently exhibiting outstanding performance across various tasks [34]. In implementing the transformer, the input sequence x is used to produce an output vector h first and is projected to a posterior vector with linear transformation as P(x) = Softmax(Wh), where |P(x)| is the size of vocabulary V and W is a linear layer. Then, various decoding techniques can be employed to obtain the next token wn+1. These techniques include the use of greedy search, where we select the token with the highest probability, or the sampling approach, where we sample the token uniformly from the output distribution. Furthermore, LMs can be broadly divided into two primary categories: conditional LMs [29] and masked LMs [9]. In this paper, we focus on conditional LMs, which follow Equation 1 to compute the conditional probability of the next token based on a preceding context, but our work can be adapted to masked LM.
# 2.2 In-Context Learning
ICL involves prompting the LLM with k input-label pair demonstrations before presenting the test input. Then, the model produces the label based on the output probability as depicted in Figure 2, even if the task was neither seen nor explicitly learned during the pre-training phase. This paper focuses on the classification task, where x is the input sequence and y is the ground truth label. To construct the demonstration, k input-label pairs are sampled from the dataset and concatenated as d = (x1,y1,x2,y2,··· ,xk,yk), which is then combined with the test input xtest. Also, it is common to integrate the input and label into a template to improve performance. These templates are commonly designed by humans, as shown in Section A.1. The entire ICL procedure shares the same setup as previous ICL works [23, 26, 38], and can be viewed as follows:
(2)
where y is the prediction, and C is a small discrete set of possible labels.
# 2.3 Parameter-Efficient Fine-Tuning
As the sizes of deep learning models increase, fine-tuning the whole model becomes more expensive. Parameter-efficient fine-tuning (PEFT) offers a viable solution to the rising computational costs. Instead of updating all parameters of the pre-trained model, PEFT demonstrates the feasibility of updating a relatively small number of parameters. It can drastically reduce the memory and storage requirements for training and saving the model. Besides, certain PEFT methods only require attaching the extra parameters before [18,19] or at the end side of the model [16]. Such flexibility allows the user to perform different tasks using the same model. Similarly, the model owner can implement various ICL policies with different sets of parameters. In this paper, we assemble ICLGuard with two different PEFT methods independently. LoRA. Low-Rank Adaptation (LoRA) is one of the most popular PEFT approaches [16]. In the implementation of the transformer, it contains the self-attention module (Wq/Wk/Wv) [34]. For each weight in the module, LoRA represents the weight updates with two smaller matrices through low-rank decomposition as:
# h = W0x+BAx
where W0 ∈Rd×k is the LM’s weight (Wq/Wk/Wv), and B ∈ Rd×r and A ∈Rr×k are the update matrices and rank r << min(d,k). In every training iteration, W0 remains frozen, and only B and A are updated. Prompt Tuning. Prompt tuning is another PEFT approach [18], and it focuses on learning “soft” prompts for specific downstream tasks by appending a trainable continuous embedding to the model’s input or activations while freezing the pre-trained model.
# 3 Problem Statement
In our problem formulation, we have two parties, i.e., model owner and users.
Model Owner. The model owner offers an LLM that is capable of performing ICL. For the regular prompt, the LLM returns a sequence of full posterior vectors. These vectors are then converted into discrete tokens based on the selected decoding strategy. With ICL, users can prompt the LLM by presenting a few demonstrations with the test input. The LLM also returns a sequence of full posterior vectors. For the classification problem, users can determine the label by comparing the probability associated with the token (label). However, there are concerns associated with ICL. First, users can exploit ICL to perform tasks with any data that might violate the model policy or conflict with the model owner’s interests. Second, the model owner might face requests or policies that demand modifications to the model’s ICL behavior. For example, using an LLM to determine the sentiment of a specific political group’s tweets might be banned. To address these concerns, the owner introduces ICL applicability authorization over the LLMs. Concretely, they use ICLGuard to fine-tune the PEFT module to regulate the ICL behavior on various data. To evaluate the successful implementation of ICL applicability authorization on the LLM, we have outlined the following criteria:
• Requirement 1: the guarded LLM should deactivate the ICL ability on target data by producing incorrect output probability for the possible label set.
• Requirement 3: the guarded LLM should function similarly to the original LLM on processing regular prompts.
Model Owner’s Background Knowledge. To deactivate the ICL ability on target data, we first assume that the owner has access to a small subset of the data. In practice, this subset can be sourced from the model usage log or provided by the requester. Second, the template and label set selections for ICL can significantly influence performance. We assume the model owner has only basic knowledge about them, thus utilizing the minimal template and default label set in Section A.1. Users. We categorize users into two types: regular users, and malicious users. All users have full access to the model. They can query the model and retrieve the full posterior vectors. Therefore, the user can use the model with regular or ICL format prompts. However, some malicious users may attempt to prompt the LLM with prohibited data in ICL format for their own interests, potentially breaking the rules or harming the model owner. In addition, these users might use different templates or label sets in the adaptive attack scenario to bypass the ICL applicability authorization.
# 4 Methodology
In this section, we introduce ICLGuard, a fine-tuning framework designed to control the LLM’s ICL behavior. We start
Notation
Description
DsdICL,DsgICL Shadow/Surrogate ICL Dataset
Dsg
Surrogate Dataset
θ
LLM
φ
PEFT module
h
Text representation
d
Input-label pair demonstrations
x
Regular input
z
Input with demonstrations
y, ˆy
Non-distorted/distorted soft label
C
Label set
PPL
Perplexity
<div style="text-align: center;">Table 1: List of notations.</div>
by presenting the building block of ICLGuard. Then, we provide an overview of the fine-tuning procedure. Table 1 shows some important notations used in this paper.
# 4.1 Building Blocks
Shadow Dataset. Ideally, when the guarded LLM encounters target data in ICL format, we expect the model to produce an incorrect prediction. It can be thought of as training a model that takes a few input-label pairs with a test input and outputs an incorrect probability vector. For this purpose, the owner first constructs a shadow ICL dataset DsdICL that contains ICL prompts paired with corresponding “incorrect” labels. Given that the owner has access to the target dataset size of m, they can sample k data to create u unique demonstrations (d1,d2,··· ,du). Each target data is paired with all demonstrations zi j = (di,xj), resulting in a shadow ICL dataset DsdICL = {zi j,··· ,zum}, where |DsdICL| = u×m. The next step is to generate the label for each input from DsdICL. A straightforward approach is to use a random or flipped label as the new one. To recap, the label for ICL is determined by examining the corresponding probability of the label set, such as “negative” and “positive.” However, using a one-hot label as the target label can lead to overshooting, which will likely impact the model significantly. Instead, we modify the posterior P(zsdICL) from the original LLM θ to create the soft label ˆysdICL. We set the P(zsdICL) for all known labels (i.e., “negative” and “positive”) to zero, such as setting index 6374 (positive) and 8178 (negative) to 0 for LLaMA-13B, and produce the soft distorted label ˆysdICL as presented:
where i is the token index from the vocabulary V. At this point, we have the shadow ICL dataset ready DsdICL = {(zsdICL, ˆysdICL)i}u×m i=1 . Surrogate Dataset. In contrast to causing the model to generate incorrect labels, the guarded LLM should produce the correct label when presented with non-target data in ICL format. We define data that does not belong to the target data set as non-target data. This process is also similar to training
a classification model. Thus, we can simply build a surrogate ICL dataset DsgICL to serve this purpose. In practice, non-target data is either unknown or unavailable, given the infinite nature of possible data. Instead, we random sample m data from the original LLM and use these samples as surrogate data in place of non-target data. To create the surrogate ICL input zsgICL, we follow the process of constructing the shadow ICL input by assembling u unique k input-label pairs with the input. Also, we assign random labels for each input-label pair since these sampled data do not come with a label. Regarding the label ysgICL, we can directly use the posterior obtained from the original LLM as the soft label and assemble the surrogate ICL dataset as DsgICL = {(zsgICL,ysgICL)i}u×m i=1 . However, random sampling data from the LLM might generate data that is similar to the target data in terms of semantics or structure. It might hurt the performance of the guarded LLM if the token or sentence overlaps between the shadow and surrogate ICL datasets. Hence, we have further evaluated various generation strategies to mitigate the issue and improve the performance, as discussed in Section 6.3. In addition, the guarded LLM should function normally on regular sentences for all data. We build a surrogate dataset Dsg, which is distinct from DsgICL, and defined as Dsg = {x1,x2,··· ,xm}. Disable Loss. We introduce the first loss of the ICLGuard, i.e., the disable loss LD. This loss is designed to prevent any target data from being utilized for ICL on the model. Given a data point (zsdICL, ˆysdICL) from DsdICL, we can maximize the log-likelihood of ˆysdICL to optimize φ while freezing θ as:
(3)
This objective should force the guarded LLM to produce identical probability given target data in ICL format for every label from the label set that the owner knows. Maintenance Loss. Prior studies found that fine-tuning models for a specific downstream task can adversely affect their performance on other tasks [24]. Our results in Section 6.2 also align with these findings. If we only finetune the model with LD, the ICL and regular LM function of the guarded LLM degenerate compared to the original LLM. Therefore, we present the maintenance loss, LM, which aims to retain the ICL ability on non-target data using DsgICL. Similar to how LD operates, we also maximize the log-likelihood of ysgICL to optimize φ while θ remains fixed. The loss LM is computed as:
(4)
This objective should compel the guarded LLM to produce a posterior similar to the original LLM on non-target data in ICL format, thereby maintaining the ICL performance. Utility Loss. Both LD and LM regulate the model’s ICL behavior on target and non-target data, respectively. Next, we focus on the model utility of processing regular data, i.e., non-ICL format prompt. Similar to the ICL performance drops on non-target data, we also observe that the performance of processing regular sentences on the guarded LLM
is compromised. A possible explanation for this might be the issue of “concept forgetting” [24]. Therefore, we introduce the utility loss LU to re-align the guarded LLM with respect to the original LLM. A straightforward approach is to fine-tune the model with Dsg using the autoregressive loss. However, the model might overfit the dataset since the size of the surrogate dataset is much smaller than the pretraining dataset. As an alternative, we minimize the distance between text representations from the guarded and original models, which was previously demonstrated to be effective in [24, 41]. The token representation at i-th location from layer j is extracted from the guarded and original LLMs, denoted as h′ j i and hj i , respectively. In addition, extending the optimization scope across layers enhances the model utility performance [41]. Therefore, we extract all token representations from every l layer, normalize each vector using the L2-norm as h j i ||h j i ||2 , and concatenate them into a single vector. The objective is to minimize the L2 distance between these sets of representations, and thus, the loss LA is defined as:
(5)
This objective should force the guarded LLM to produce similar outputs as the original LLM on all data.
Algorithm 1: ICLGuard
Input
: DsdICL, DsgICL, Dsg, LLM θ, PEFT
module φ
Output
: PEFT module φ
1 for t ←0 to T do
2
(zsdICL, ˆysdICL) ∼DsdICL;
3
LD = log P(ˆysdICL|zsdICL;θ;φ);
4
(zsgICL,ysgICL) ∼DsgICL;
5
LM = log P(ysgICL|zsgICL;θ;φ);
6
xsg ∼Dsg;
7
{h′1,h′2,··· ,h′n} = φ(θ(xsg));
8
h′ = (h′1;h′2;··· ;h′n);
9
{h1,h2,··· ,hn} = θ(xsg);
10
h = (h1;h2;··· ;hn);
11
LU = ||h′ −h||2;
12
L = LD +LM +LU;
13
updating φ with L;
14 end
After presenting the building blocks, we introduce the overall process of ICLGuard. First, the owner aims to guard the original LLM by fine-tuning the PEFT module and plugging it into the LLM. For each epoch, ICLGuard samples the data from DsdICL and obtains the output from the guarded LLM to compute the cross-entropy for LD. Second, it computes the cross-entropy for LM with the data from DsgICL. Last, ICLGuard samples the data from Dsg to obtain a sequence of normized vector {h′1,h′2,··· ,h′n} from the guarded LLM and {h1,h2,··· ,hn} from the original LLM, where n is the sentence length. Then, ICLGuard concatenates them and
computes LU by minimizing the L2 distance between h′ and h. The final objective function is:
(6)
Importantly, ICLGuard only updates the PEFT module φ, and the original LLM θ is frozen. The entire optimization process is repeated for T epochs. Further details can be found in algorithm 1.
# 5 Experimental Setup
In this section, we describe our evaluation setup, including the datasets, implementation details, and evaluation metrics.
# 5.1 Dataset
# 5.1 Dataset
5.1 Dataset We use the following five datasets to conduct our experiments.
We use the following five datasets to conduct our experiments.
• FP [22] contains sentences from financial news annotated as positive, negative, or neutral.
Each FP, SST-2, TREC, and Agnews are treated interchangeably as the target data, while the remaining is treated as the auxiliary (non-target) data. Also, we sample 100 examples from each dataset for the evaluation.
# 5.2 Implementation Details
All experiments are implemented using PyTorch [1] and Transformers [39]. We describe the experimental setup used for evaluation. Models. We experiment with 3 models: LLaMA [33], OPT [42], and Cerebras [10]. Each model is pre-trained on distinct datasets. Both OPT and Cerebras support English, whereas LLaMA can process multiple languages. LLaMA has a vocabulary size of 32,000, which is smaller than the 50,257 of both OPT and Cerebras. Besides, LLaMA (13B) outperforms OPT (13B) and Cerebras (13B) in several evaluations 1, including text reasoning, understanding, and generation. It also demonstrates impressive zero-shot and few-shot
1https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
Target Dataset
Auxiliary Datasets
Target
Accuracy (%)
FP (%)
SST-2 (%)
TREC (%)
AGnews (%)
FP
0.31 (-0.49)
-
0.80 (-0.13)
0.68 (-0.02)
0.82 (-0.08)
SST-2
0.41 (-0.52)
0.71 (-0.09)
-
0.68 (-0.02)
0.88 (-0.02)
TREC
0.13 (-0.57)
0.76 (-0.04)
0.93 (+0.00)
-
0.89 (-0.01)
AGnews
0.21 (-0.69)
0.62 (-0.18)
0.92 (-0.01)
0.71 (-0.01)
-
<div style="text-align: center;">Table 2: The ICL performance and changes of the guarded LLM.</div>
Target Dataset
Auxiliary Datasets
Target
∆PPL
FP
SST-2
TREC
AGnews
Lambada
FP
+0.020
-
-2.164
+0.024
-0.004
-0.012
SST-2
+2.885
+0.091
-
+0.172
+0.006
0.012
TREC
+0.044
-0.004
-2.190
-
-0.007
+0.002
AGnews
+0.003
+0.025
-2.006
+0.131
-
+0.006
capability. Therefore, we primarily focus on LLaMA-13B. We also assess a range of LLaMA model sizes, including 7B and 30B.
Training. To regulate ICL behavior, we apply the PEFT method on top of the original LLM. We primarily experiment with LoRA on top of the LLaMA-13B through the paper unless specifically stated. For LoRA, the rank is set at 8, alpha at 32, and the dropout rate is 0.1. In addition, we also evaluate ICLGuard with prompt tuning in the PEFT study. For prompt tuning, we set the embedding length to 8 and 16, and the hidden size is the same as the LLM input embedding. We use a batch size of 4 and a learning rate of 1 x 10−4 with the Adam optimizer. We train the model for 20 epochs. 2 Test Set. Despite the success of ICL, the performance is unstable based on the choice of template and combination of demonstrations. While many prior studies have explored the factors and potential mitigation approaches, the study is beyond the scope of this paper. For simplicity, we utilize the minimal template as shown in Section A.1. To construct the test set, we random sample data from the training set to construct the demonstration with distinct random seeds from 40 to 50. Then, we select the demonstration (random seed) that yields the highest performance for each dataset, and these sampled data are excluded from training. Other Details. In all experiments presented in this paper, we default to using k = 16 input-label pairs and u = 40 unique input-label demonstrations unless stated otherwise. For the target data, 100 data points suffice to deactivate the ICL ability on target data. For the utility loss, we extract text representation from every l = 2 transformer layer.
We evaluate the performance of the guarded LLM using 2 metrics: ICL performance and utility.
2We also explored training longer, but it did not improve performance.
ICL Performance. We evaluate the ICL performance of the guarded LLM using both the target and auxiliary datasets. For the target dataset, we compute the test set accuracy. When the accuracy approximates random guessing, it indicates the effectiveness of deactivating the ICL ability on target data. For the auxiliary dataset, we use datasets distinct from the target dataset to serve as auxiliary datasets and compute the accuracy of the auxiliary test set. If the accuracy closely aligns with the ICL performance of the original LLM, it demonstrates the effectiveness of persevering the ICL ability on non-target data. Utility. Besides considering the ICL performance on target and non-target data, it is necessary to evaluate the model’s utility. We use perplexity (PPL) to evaluate the utility of LM:
PPL is a metric to measure the quality of sentence generation, i.e., how well the model predicts wn+1 given the proceeding sequence w1<n. While a lower perplexity typically indicates greater utility, we care about the PPL change between the guarded and original LLM in this evaluation. The utility of the guarded LLM should align with the original LLM, given all data is in non-ICL format. Hence, we compute the PPL change for regular sentences between the two models:
∆PPL = PPLoriginal −PPLguarded
# 6 Evaluation
We examine the experimental results from different perspectives. First, we evaluate the overall performance of the guarded LLM while deactivating the ICL ability on different target data. Second, we investigate the effectiveness of each loss function and the impact. Following this, we explore different generation strategies for constructing the surrogate dataset and examine the potential for bypassing the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/cc9b/cc9b9784-afa1-4636-949d-d125aad67045.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/864d/864d5696-ed3c-424f-ac1c-bd9fbf93bff6.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) SST-2</div>
<div style="text-align: center;">(a) FP</div>
Figure 3: The ICL performance changes during fine-tuning under different loss combinations. D U = utility loss.
ICL applicability authorization. Finally, we study the effects of varying setups on ICLGuard. We provide the original ICL performance and utility of LLaMA-13B for each dataset in Table 15.
# 6.1 General
We first evaluate the ICL performance of the guarded LLM after deactivating the ICL ability on target data (i.e., FP, SST2, TREC, and AGnews). Table 2 illustrates the accuracy and changes compared to the original LLM. In general, the guarded LLM can deactivate the ICL ability on target data and bring the accuracy close to random guessing. Meanwhile, the ICL performance on auxiliary datasets only drops negligibly overall. For instance, when the target dataset is TREC, its accuracy drops from 70% to 13%, while the accuracy on FP, SST-2, and AGnews is maintained at 76%, 93%, and 89%, respectively. However, we observe that the ICL performance on the auxiliary dataset is affected in some cases. When the target object is FP, the accuracy on SST-2 drops by 13%, which is slightly higher than other auxiliary datasets. This unexpected result could be due to the overlapping label set between FP and SST-2, and deactivating the ICL ability on FP might also affect SST-2 accidentally. We observe the same phenomenon when targeting SST-2 as the accuracy on FP drops by 9%. Next, we evaluate the utility of the guarded LLM. As shown in Table 3, the PPL changes are negligible across target and auxiliary datasets. The result demonstrates that the guarded LLM can produce outputs almost identical to the original on regular prompts. Note that the PPL change on SST-2 is generally higher since it has a higher PPL on the original LLM. Overall, these results indicate the efficacy of ICLGuard to fine-tune the guarded LLM on deactivating the ICL ability on target data without compromising the ICL performance on non-target data and the regular LM function on all data simultaneously. In the following section, we primarily use FP as the target dataset to investigate the effectiveness of ICLGuard from other aspects.
# 6.2 Impact of the Loss Function
In this section, we investigate the impact of each loss function on controlling the ICL ability and regular LM ability on
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/681d/681d5a94-a9c4-4629-8cb9-81fe63cb663e.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) TREC</div>
<div style="text-align: center;">(d) AGnews</div>
Target
Auxiliary
Loss
FP (%)
SST-2 (%)
TREC (%)
AGnews (%)
D
-0.52
-0.34
-0.38
-0.67
D+M
-0.49
-0.49
-0.09
-0.29
D+U
-0.50
-0.32
-0.53
-0.70
D+M+U
-0.49
-0.13
-0.02
-0.08
Table 4: The ICL performance changes with different loss combinations. D = disable loss, M = maintenance loss, U = utility loss.
Target
Auxiliary
Loss
FP
SST-2
TREC
AGnews
Lambada
D
+24.08
-198.5
+20.75
+48.89
+32.12
D+M
-0.484
-18.46
-0.192
-0.062
-0.099
D+U
+0.004
-2.220
-0.039
-0.007
-0.006
D+M+U
+0.020
-2.164
+0.024
-0.004
-0.012
Table 5: The utility changes with different loss combinations. D = disable loss, M = maintenance loss, U = utility loss.
target and auxiliary data. In Table 4 and Table 5, we first note that using the disable loss alone can deactivate the ICL ability on FP completely from 80% to 28%. As expected, the ICL ability on auxiliary datasets is affected significantly. For instance, SST-2, TREC, and AGnews have a 34%, 38%, and 67% accuracy drop, respectively. The PPL change on them is also massive, with more than 20. All these results reflect that the guarded LLM has altered completely. Then, when we optimize the model with both disable and maintenance loss, the ICL ability on auxiliary datasets recovers to a certain extent without sacrificing the deactivating performance. For instance, TREC and AGnews have only 9% and 29% accuracy drops now, and the PPL changes are also much lower. Although the overall performance is not the best, the guarded LLM can function adequately close to the original LLM on data in non-ICL format. Besides, we note that both the ICL performance and utility on SST-2 are affected significantly. As mentioned in Section 6.1, the result is likely to be related to the overlapping label set. Therefore, fine-tuning the model with both disable and maintenance loss
Target
Auxiliary
Method
FP (%)
SST-2 (%)
TREC (%)
AGnews (%)
Random
-0.49
-0.13
-0.02
-0.08
Auxiliary (SST-2)
-0.62
+0.00
-0.02
-0.06
JS
-0.50
-0.12
+0.00
+0.00
M-Cos
-0.66
-0.10
-0.01
+0.00
M-L2
-0.65
-0.09
-0.01
+0.00
L-Cos
-0.49
-0.50
+0.00
+0.00
L-L2
-0.63
-0.24
-0.06
-0.04
Table 6: The ICL performance changes with different generation strategies. JS = Jaccard Similarity, FP=, M = MiniLM, L = LLaMA, Cos = cosine similarity, L2 = Euclidean distance.
Target
Auxiliary
Method
FP
SST-2
TREC
AGnews
Lambada
Random
+0.020
-2.164
+0.024
-0.004
-0.012
Auxiliary (SST-2)
+0.000
+0.332
+0.105
+0.000
+0.019
JS
+0.022
+3.085
+0.005
-0.009
-0.001
M-Cos
-0.005
+2.814
+0.178
+0.002
-0.001
M-L2
+0.011
+0.468
+0.244
+0.001
-0.008
L-Cos
+0.022
+3.187
+0.009
+0.002
-0.001
L-L2
+0.004
+3.058
+0.028
+0.002
+0.012
Table 7: The utility changes with different generation strategies. JS = Jaccard Similarity, M = MiniLM, L = LLaMA, Cos = cosine similarity, L2 = Euclidean distance.
together might hurt the optimization. On the other hand, we find that the utility changes across datasets are negligible when using both disable and utility loss. It confirms that the utility loss effectively preserves the guarded LLM’s functionality in processing regular data. Interestingly, we observe that the ICL performance on SST-2 is better than using the maintenance loss as the accuracy goes up to 61%. Taken together, using all three losses produces the best performance in controlling ICL behavior and maintaining the LLM functionality. Fine-Tuning Process. Furthermore, we report the ICL performance changes over the fine-tuning process in Figure 3. First, with no surprise, all loss combinations can deactivate the ICL ability on FP in the early stage, as shown in Figure 3a. Second, we observe that the ICL ability on the auxiliary dataset is affected at the beginning of the fine-tuning, but it is recovered later. For instance, the accuracy on TREC drops initially and is recovered after 5 epochs, as depicted in Figure 3c. Besides, we observe that the performance on SST-2 fluctuates when using only disable and maintenance loss during the fine-tuning in Figure 3b. It confirms our hypothesis that the disable and maintenance loss conflict with each other, leading to unstable optimization. Takeaways. In summary, the disable loss can deactivate the ICL ability on target data, the maintenance loss retains the ICL ability on non-target data, and the utility loss ensures the regular function of the LLM on all data. Implementing all three losses simultaneously leads to successful ICL applicability authorization on the LLM.
together might hurt the optimization. On the other hand, we find that the utility changes across datasets are negligible when using both disable and utility loss. It confirms that the utility loss effectively preserves the guarded LLM’s functionality in processing regular data. Interestingly, we observe that the ICL performance on SST-2 is better than using the maintenance loss as the accuracy goes up to 61%. Taken together, using all three losses produces the best performance in controlling ICL behavior and maintaining the LLM functionality.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/428b/428bbeba-30d2-4389-9dd8-1cd1b1d11baa.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) MiniLM-L2</div>
Figure 4: The t-SNE visualization on the shadow and surrogate data with MiniLM using L2 distance.
<div style="text-align: center;">Figure 4: The t-SNE visualization on the shadow and surrogate data with MiniLM using L2 distance.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/24f0/24f0f3ed-57a3-4ce4-b743-56445a52ebb4.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) LLaMA-L2</div>
<div style="text-align: center;">(a) Random</div>
<div style="text-align: center;">Figure 5: The t-SNE visualization on the shadow and surrogate data with LLaMA-13B using L2 distance.</div>
# 6.3 Surrogate Dataset Augmentation
In Section 4, we build the surrogate dataset by random sampling data from the original LLM. However, the sampled data can be similar to the target data, which might hurt the optimization. If the sampled data is similar to auxiliary data like SST-2, we might mitigate the ICL performance drops. To justify our hypothesis, we first assume that we have access to the SST-2 dataset and use it to replace the surrogate dataset. In Table 6 and Table 7, we observe that the ICL performance of the guarded LLM is now identical to the original LLM as the accuracy change on SST-2 is 0%. The results demonstrate that a good surrogate dataset can reduce the impact of fine-tuning. Since the auxiliary or non-target data is unavailable, our best solution is to generate sample data that differs from the target and other sample data. Concretely, we aim to produce data by increasing the sentence distance between the shadow and surrogate datasets. We consider the distance in terms of discrete token and latent representation space. Discrete Token. First, we attempt to generate data that is different with respect to the token level. We use Jaccard Similarity (JS) to measure the distance between target and sample data in terms of token. JS computes the overlapping ratio between two text sentences:
(7)
� It evaluates how closely the two sentences are contextually related by calculating the proportion of common tokens to total tokens. The idea here is to generate data with tokens
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4519/45196bf9-e181-466e-a6f6-67b8e422cdd5.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c8df/c8df114f-57d0-40bc-932c-34326e2319a8.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Different templates.</div>
<div style="text-align: center;">Figure 6: The ICL performance on FP against the adaptive adversary.</div>
different from the target and other sample data. In detail, we retain only the data with a JS score of less than 0.1, meaning a maximum of 10% token overlap between the target and sample data. In Table 6, we find that it can improve the ICL ability on TREC and AGnews, which has a 0% accuracy drop now. In addition, the utility remains decent as the PPL changes are still negligible across datasets. Latent Representation. Second, we explore to generate data that is different in latent space. We use the original LLaMA (conditional LM) and MiniLM 3 (Masked LM) to extract the sentence representation. For MiniLM, we obtain the sentence representation by applying the mean pooling on top of the token representation. However, LLaMA is a conditional LM, where process input is from left to right. Hence, the first token representation does not contain any information about the later token, which is not ideal for applying mean pooling. Instead, we use the last token representation to approximate the sentence representation, which has attended to all the tokens in the sentence. We then measure the sentence distance with the Euclidean distance (L2) or cosine similarity (Cos) and only keep the generated data when the score is larger or smaller than the threshold respectively. 4 Our objective is to produce data that covers more latent space than random sampling. It can be seen from the data in Table 6 that using the representation from MiniLM outperforms LLaMA in general. For instance, generating data based on L2 or Cos with MiniLM decreases the ICL performance on FP further to 14% and 15% and mitigates the impact on TREC and AGnews. Also, using L2 distance can preserve the ICL ability on SST-2 better up to 84% accuracy. On the other hand, using the representation from LLaMA does not improve the performance of controlling ICL behavior in general. For instance, using Cos with LLaMA can also mitigate the impact on TREC and AGnews, but the ICL performance on SST-2 is worse than the random sampling. While using L2 with LLaMA can decrease the ICL performance on FP to 17%, the ICL perfor-
3https://www.sbert.net/index.html 4We determine the threshold by measuring the score between data points in the target dataset.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/43c7/43c70953-1506-48b0-b8bc-b39a4e84ff99.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) Different numbers of demonstrations.</div>
mance on SST-2 also drops to 69%. In Table 7, we report the PPL changes. What stands out in the table is that using L2 with MiniLM produces better utility on SST-2 with a PPL change of 0.468, which is close to using FP as the surrogate data. It shows the effectiveness of generating data with MiniLM based on L2 distance. We also visualize the shadow and surrogate data in Figure 4 and Figure 5. As the figure shows, generating data with L2 distance indeed has more space coverage than random sampling, and the newly generated data has less overlapping with the shadow data. Takeaways. In summary, compared to random sampling, we find that generating data that is distant in token level or latent space produces better performance in terms of the ICL ability and utility of the guarded LLM.
# 6.4 Adaptive Attacks
In Section 4 and Section 5, we present the details of constructing the shadow and surrogate ICL dataset and set up the demonstration (16 examples) using the minimal template with the default label set. In practical scenarios, the user has the freedom to decide the use of the template, label set, and number of demonstrations. If the malicious user realizes that some demonstration setups are ineffective, they can switch to another setup, and we consider it an adaptive attack. In this section, we evaluate the robustness of the guarded LLM against these adaptive demonstrations by considering three variables: template, label, and number of demonstrations. Template. First, we conduct our experiment by using the other three templates in Section A.1. As depicted in Figure 6a, the guarded LLM remains effective at deactivating the ICL ability on FP, even when encountering templates that are not presented during the fine-tuning phase. We observe that the ICL ability on FP with template 1 and 3 still has over 40% and 60% accuracy drops, respectively. On the other hand, using template 2 can perverse the ICL performance slightly up to 46%, but that is not enough to be usable for the adversary. Label. Second, we investigate the implications of label mismatches using other label sets in Section A.1. Apart from altering the template, modifying the label is another viable avenue for adversaries. In Figure 6b, we find that switching
Target
Auxiliary
Models
FP (%)
SST-2 (%)
TREC (%)
AGnews (%)
LLaMA-13B
0.31 (-0.49)
0.80 (-0.13)
0.68 (-0.02)
0.82 (-0.08)
OPT-13B
0.17 (-0.56)
0.86 (-0.07)
0.59 (+0.03)
0.30 (-0.48)
Cerebras-13B
0.31 (-0.56)
0.81 (-0.08)
0.40 (-0.01)
0.37 (-0.40)
<div style="text-align: center;">Table 8: The ICL performance with different models.</div>
Target
Auxiliary
Models
FP (%)
SST-2 (%)
TREC (%)
AGnews (%)
7B
0.51 (-0.26)
0.92 (-0.03)
0.77 (-0.03)
0.85 (-0.04)
13B
0.31 (-0.49)
0.80 (-0.13)
0.68 (-0.02)
0.82 (-0.08)
30B
0.31 (-0.52)
0.91 (-0.04)
0.92 (+0.00)
0.91 (+0.00)
Table 9: The ICL performance with different model sizes of LLaMA model.
labels does not undermine the ICL applicability authorization. Instead, the accuracy on FP drops further from 31% to 17%. Number of Demonstrations. While our study constructs the shadow and surrogate ICL dataset utilizing 16 demonstrations, an adversary might employ a different quantity (e.g., 4, 8, or 32). We conduct the experiment to investigate the impact of varying the number of demonstrations. As shown in Figure 6c, reducing the number of demonstrations marginally weakens the guarded LLM’s ability to deactivate ICL on FP. For instance, the accuracy increases from 31% to 52% as the number of demonstrations decreases from 16 to 4. But still, the adversary should not be able to use ICL to perform the task with low accuracy. Takeaways. In general, the guarded LLM is able to deal with ICL prompts with different setups. Although we can see some drops in certain cases, we are not concerned about it since the ICL performance on target data is still maintained at a low level.
# 6.5 Further Study
We now investigate the effects of different setups for ICLGuard. First, we examine the impact of applying ICLGuard to other models and the influence of model size. Second, we demonstrate the effectiveness of deactivating multiple datasets. Third, we compare performance based on the number of unique demonstrations and different amounts of sampling data. Finally, we evaluate the performance of ICLGuard compared to existing fine-tuning methods, and the performance between using LoRA and prompt tuning. Models. We start by evaluating the efficacy of ICLGuard across different models: LLaMA-13B, OPT-13B, and Cerebras-13B. The objective is to ascertain that ICLGuard can be applied universally across varied LLMs. We provide the original ICL performance and utility for OPT and Cerebras in Section A.2. In Table 8, all guarded LLMs successfully reduce the ICL performance on FP to nearly random guessing. We observe that both OPT and Cerebras have better ICL performance (less performance drops) on SST-2 with 86% and 81% accuracy, respectively. Besides, they exhibit
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d8e1/d8e165ea-4856-40d8-9493-4061da5a14e0.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: The ICL performance of deactivating different multiple target datasets.</div>
Figure 7: The ICL performance of deactivating different multiple target datasets.
a higher accuracy drop in using ICL on AGnews, achieving only 30% and 37% accuracy. This result may be explained by the fact that AGnews data is similar to the pre-training data of these models. In Table 20 (Section A.3), AGnews generally has lower perplexity, and even mirror changes to the model with AGnews data would likely affect the model. But still, the overall performance is comparable to the original model. Model Size. We then extended our evaluation by varying the model size of LLaMA from 7B to 30B, as shown in Table 9. We provide the original ICL performance and utility for the 7B and 30B models in Section A.2. We observed that the smaller model has a weaker capability to deactivate the ICL ability on FP. In contrast, the 13B and 30B models can completely deactivate the ICL ability on FP. Since the fine-tuning effect of ICLGuard on the 7B model is smaller, it also has a smaller impact on SST-2 and AGnews. On the other hand, the 30B model is able to deactivate the ICL ability on FP without compromising the ICL performance on other auxiliary datasets, including SST-2. Table 21 (Section A.3) reports the utility changes, demonstrating that the impact of ICLGuard on the 7B and 30B models is less significant. Multiple Target. In reality, the model owner might need to deactivate multiple data simultaneously. Therefore, we study the effectiveness of ICLGuard in deactivating multiple datasets in two scenarios: 1) FP and SST-2, and 2) FP, SST-2, and AGnews. As shown in Figure 7 and Table 22, ICLGuard successfully deactivates multiple targets without compromising the model utility. In the first scenario, the accuracy of FP and SST-2 drops to 16% and 41%, respectively, while the accuracy of TREC and AGnews only decreases by 3% and 3%. In the second scenario, the accuracy of FP, SST2, and AGnews drops to 17%, 41%, and 19%, respectively. Although there is a slight change in the accuracy and utility of TREC in both scenarios, the general ICL performance and model utility remain comparable to the original model. The results indicate the effectiveness of using ICLGuard to manage multiple ICL control requirements.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9cfc/9cfcad4e-f8c4-4928-a609-8b08256be57b.png" style="width: 50%;"></div>
Figure 8: The ICL performance with different numbers of unique demonstrations.
<div style="text-align: center;">Figure 8: The ICL performance with different numbers of unique demonstrations.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/55e6/55e66570-8fb6-4766-8cf9-875e63ad636b.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: The ICL performance with different sampling sizes.</div>
Number of Unique Demonstrations. We evaluate the effect of varying the number of unique demonstrations from 10 to 80. Typically, increasing the number of unique demonstrations should enhance the model’s ability to control ICL behavior on both target and non-target data. In Figure 8, the results reveal that even with a smaller number of unique demonstrations, it is still possible to deactivate the ICL ability on FP. However, we also suspect that performance drops at 20 demonstrations are due to the instability of ICL, and using a larger number of unique demonstrations can mitigate the issue. In addition, using more unique demonstrations can preserve the ICL ability on other datasets better. For instance, the accuracy on SST-2 increases from 46% to 90%, and for TREC, it increases from 60% to 70%. These relationships may be explained by using more unique demonstrations, which enable the guarded LLM to better distinguish the data from SST-2 or FP during fine-tuning. In Table 23 (Section A.3), we observe that the PPL change on TREC decreases as the number of unique demonstrations increases, although the difference is negligible. Sampling Size. In addition to selecting the number of unique demonstrations, the owner can also choose to sample more data from the original LLM. We investigate the impact of varying sampling sizes (100, 200, 400) and believe this study
Target
Auxiliary
# Sample
FP
SST-2
TREC
AGnews
ICLGuard
0.31 (-0.49)
0.80 (-0.13)
0.68 (-0.02)
0.82 (-0.08)
SFT
0.21 (-0.59)
0.82 (-0.11)
0.59 (-0.11)
0.26 (-0.64)
DPO
0.03 (-0.77)
0.41 (-0.52)
0.43 (-0.27)
0.22 (-0.68)
Table 10: The ICL performance changes with different finetuning methods.
can further enhance the results in Section 6.3. In Figure 9, we observe that using more sample data can enhance the deactivating effect on FP, with the ICL performance dropping to 17% when using 200 and 400 sample data. Also, using more sample data can slightly improve the ICL performance on TREC and AGnews. In Table 24 (Section A.3), we also observe that the PPL change on SST-2 is lower with more sampling. For instance, the PPL change is 0.339 and 0.399 when the sampling size is 200 and 400, respectively. Generally, utilizing more sampled data can enhance the ICL controllability of the guarded LLM, but it also demands additional computational resources. Fine-tuning Methods. To control the ICL behavior, one straightforward approach is to fine-tune the model to produce an incorrect prediction given the ICL prompt. Another approach is to fine-tune LLM with reinforcement learning from human feedback, which has become a popular method for aligning models with human values [6]. Therefore, we compare ICLGuard with Supervised Fine-Tuning (SFT) [25] and Direct Preference Optimization (DPO) [30] approaches on deactivating the ICL for the FP dataset (training details in Section A.4). As shown in Table 10 and Table 25, we report the ICL performance and the model utility. First, we observe a sharp decline in FP accuracy after applying SFT, showing the success of deactivating the target data. However, there is a varying degree of decrease in accuracy across other non-target datasets, along with notable changes in the model utility. We suspect this is due to significant domain shifts or distortions in the model, which could potentially lead to “concept forgetting.” In fact, we can consider that training with disable loss is a softer version of SFT, where the model is trained to maximize the log-likelihood of the softly distorted label. In contrast, SFT aims to maximize the log-likelihood of the incorrect one-hot label. Therefore, training with SFT will likely distort the model significantly, similar to the case where only disable loss is applied, as discussed in Section 6.2. Besides, the result shows that optimizing the model with the one-hot incorrect label will heavily alter it, which demonstrates the need of the soft distorted label. Second, we find that DPO can deactivate the ICL ability for FP without affecting the model utility. However, the ICL performance for other data is affected. For instance, the accuracy drops to 41% for SST-2, 43% for TREC, and 22% for AGnews. This result follows a similar trend observed in Section 6.2 where only disable and utility losses are applied to train the model. It suggests the need to use the maintenance loss to preserve the ICL ability on non-target data.
can further enhance the results in Section 6.3. In Figure 9, we observe that using more sample data can enhance the deactivating effect on FP, with the ICL performance dropping to 17% when using 200 and 400 sample data. Also, using more sample data can slightly improve the ICL performance on TREC and AGnews. In Table 24 (Section A.3), we also observe that the PPL change on SST-2 is lower with more sampling. For instance, the PPL change is 0.339 and 0.399 when the sampling size is 200 and 400, respectively. Generally, utilizing more sampled data can enhance the ICL controllability of the guarded LLM, but it also demands additional computational resources.
Target
Auxiliary
PEFT
FP (%)
SST-2 (%)
TREC (%)
AGnews (%)
LoRA
-0.49
-0.13
-0.02
-0.08
PT (8)
-0.28
+0.02
-0.01
-0.21
PT (16)
-0.28
-0.06
+0.03
-0.22
Table 11: The ICL performance with different PEFT methods. PT = prompt tuning (length).
<div style="text-align: center;">Table 11: The ICL performance with different PEFT methods. PT = prompt tuning (length).</div>
Target
Auxiliary
WMT19 (BLEU)
FP (%)
SST-2 (%)
TREC (%)
AGnews (%)
Original
18.24
0.80
0.93
0.70
0.90
Guarded
1.262
0.82
0.94
0.70
0.89
Table 12: The ICL performance of the guarded LLM on deactivating English-to-German translation.
PEFT. We study the implications of different PEFT techniques, primarily focusing on LoRA and prompt tuning. Our findings, as shown in Table 11, indicate that prompt tuning can partially deactivate the ICL ability on FP, with a 28% drop in accuracy. However, prompt tuning results in low ICL performance on AGnews, achieving only 69% and 68% accuracy with 8 and 16-length prompts, respectively. These results are likely related to the mechanism of prompt tuning, as it is required to append to the input, which could potentially impact the model’s performance. Also, using prompt tuning would produce higher PPL change on SST-2 and TREC, as shown in Table 26 (Section A.3). Takeaways. In this section, we conduct various experiments to explore ICLGuard from different perspectives. We find that ICLGuard remains functional with other models or different model sizes. Additionally, ICLGuard is capable of deactivating multiple target data simultaneously, and increasing the number of unique demonstrations and sample data improves the guarded LLM’s performance. Moreover, we compare the performance of ICLGuard to existing fine-tuning methods and show the importance of our loss design. Last, the use of LoRA generally also yields better results. We hope that the findings in this section will assist model owners in selecting the optimal setup when implementing ICLGuard on their models.
# 6.6 Extension to Generative Task
In the previous section, we have shown that our ICLGuard is effective in controlling the ICL behavior for classification tasks. In practice, the model owner might want to prevent the model from being used for specific generative tasks via ICL, such as generating propaganda. Although different alignment methods have been proposed to fine-tune the model with human value, we have shown that such methods do not work well for ICL data. In this section, we investigate the possibility of extending ICLGuard for generative tasks. Setup. To conduct the study for the generative task, we use the WMT19 dataset as an example, which is a collection of parallel corpora used for the 2019 Conference on Machine Translation (WMT19). [13]. In this study, we as-
sume the model owner wants to prevent users from using the model for English-to-German translation via ICL. Specifically, we evaluate ICLGuard on the LLaMA-13B model using the same ICL setup as Section 5 with the template “sentence Output:”. To build the shadow ICL dataset in Section 4, we set the output logit of the first tokens of each ground truth German translation to 0. This ensures that the guarded model cannot generate the correct translation or even complete the task. To evaluate the translation performance, we compute the BLEU score using the NLTK package 5. Results. First, we compute the BLEU score for performing WMT19 on LLaMA-13B without ICL. Specifically, we use the template “Translate the sentence to German: sentence Output:” in a zero-shot manner and obtain a BLEU score of 7.17. Then, we perform the translation with 16 demonstrations and achieve a BLEU score of 18.24, as shown in Table 12. This substantial increase indicates the efficiency and potential of ICL to enhance the model’s performance in handling unseen generative tasks. Next, we evaluate the performance of ICLGuard in deactivating ICL for the WMT19 data. We observe that the BLEU score drops significantly to 1.262. This decrease illustrates the effectiveness of ICLGuard in preventing the model from performing the English-to-German translation task. Meanwhile, the model’s utility and ICL performance for other tasks remains comparable to the original performance, as shown in Table 27. The results indicate that ICLGuard successfully restricts specific unwanted generative tasks while maintaining the model’s capability for other tasks.
# 7 Related Works
# 7.1 Large Language Model Safety
Despite the success of LLMs, they also attract the attention of adversaries. One major safety concern is adversarial attacks, which are expected to be directed at LLMs. Zou et al. [44] demonstrated that specific prompts can induce LLMs to produce undesirable behaviors, and these prompts are transferable between different LLMs. In addition, adversaries might deploy backdoored LLMs designed to produce specific outputs when presented with certain triggers. BadPrompt [4] generated invisible triggers for each sample in backdoored continuous prompt models. Besides, a recent concern in the safe and ethical use of LLMs is the phenomenon of “jailbreaking” [21]. In this, the adversary design input prompts to bypass the LLM’s safety mechanisms and generate content that should be restricted. The potential attacks via these meticulously crafted prompts can be further expanded to ICL. Therefore, we aim to implement ICL applicability authorization to oversee LLMs.
# 7.2 In-Context Learning
ICL has been discovered in [3] that can perform tasks without any parameter updates. Since then, various studies have been proposed to understand the mechanism behind it. Oswald et al. [35] suggested that ICL in transformers approx-
5https://www.nltk.org/
imates gradient-based few-shot learning during its forward pass. Conversely, Xie et al. [40] interpreted ICL as implicit Bayesian inference, proposing that ICL emerges when the pre-training distribution blends with the hidden Markov models. Besides, Min et al. [23] explored various factors, such as the use of ground-truth labels, that can affect ICL performance. In addition, Dai et al. [8] demonstrated similarities between in-context learning and explicit fine-tuning through measuring with different metrics on different datasets. Prior studies mainly focus on understanding ICL and how to improve it. Instead, we explore the possibility of controlling ICL behavior.
# 7.3 In-Context Learning Safety
As the popularity of ICL surges, researchers have increasingly focused on its safety implications. Kandpal et al. [17] introduced backdoors to the LLM using ICL prompts, and the backdoored model exhibited malicious behavior when presented with triggered inputs. Duan et al. [12] studied privacy leakage on LLMs using membership inference attacks. They assume that some LLMs contain pre-defined ICL prompts for the downstream task, which might contain sensitive data. They argue that deploying prompted models presents a significant privacy risk. Panda et al. [27] proposed Differentially Private In-context Learning (DP-ICL) for privatizing ICL tasks. DP-ICL produces differentially private responses with a set of LLM responses and a noisy consensus mechanism. In this paper, we pivot to the safety concerns of ICL from a unique angle, emphasizing the control of model behavior in response to ICL prompts with different data.
# 7.4 Model Authorization
Model authorization was initially proposed to protect model intellectual property. Alam et al. [2] encrypted network parameters using a secret key. Chakraborty et al. [5] generated a secret key from the hardware fingerprints of a specific device, and only users with that device are allowed to load and utilize the model. On the other hand, Wang et al. [37] introduced applicability authorization for data-centric protection. Instead of specifying who can access the model, this approach dictates which data the model can process. For instance, a model designed for the MNIST dataset might underperform when presented with the USPS dataset. We build upon the concept of applicability authorization, extending its principles to ICL. Our objective is to regulate ICL behavior by determining what data the model can process using ICL.
# 8 Conclusion
This paper introduces the first fine-tuning framework, ICLGuard, designed to regulate ICL behavior in LLMs. Drawing inspiration from the "applicability authorization" concept tailored for LLMs, ICLGuard employs three distinct loss functions to optimize the PEFT module to safeguard the LLM. Our approach aims to deactivate the ICL capability when presented with target data while preserving its function for non-target data. Additionally, the regular functionalities of the LLM remain intact for all data. Experimental find-
ings highlight that ICLGuard effectively protects the LLM from potential misuse. For example, when applied to the LLaMA-13B model, ICLGuard reduces its ICL ability on FP to 31%, yet it sustains strong ICL performance on benchmark datasets such as SST-2, TREC, and AGnews. Further, our research investigates the individual impact of each loss function on modulating ICL behaviors and explores advanced techniques for creating data sets that extend beyond the target domain. In the later stages of our study, we explore the resilience of ICLGuard against adaptive attacks. This involves simulating scenarios where malicious users might adjust their prompting to bypass the applicability authorization mechanisms implemented by ICLGuard. By assessing these potential vulnerabilities, we can better understand the robustness of our system. We also delve into the effect of various setups that can influence the framework’s effectiveness in regulating ICL behavior within LLMs. We find that ICLGuard is still effective on different models and different model sizes and is capable of deactivating multiple target data simultaneously. Also, ICLGuard has better performance in controlling the ICL performance compared to existing fine-tuning methods. Last, we are able to adapt ICLGuard to control the generative task under ICL. We hope that these findings can offer valuable insights for model owners in selecting the optimal setup when implementing ICLGuard on their models.
# References
# 1] https://pytorch.org/. 5
[2] Manaar Alam, Sayandeep Saha, Debdeep Mukhopadhyay, and Sandip Kundu. Deep-Lock: Secure Authorization for Deep Neural Networks. CoRR abs/2008.05966, 2020. 1, 13 [3] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language Models are FewShot Learners. In Annual Conference on Neural Information Processing Systems (NeurIPS). NeurIPS, 2020. 1, 12 [4] Xiangrui Cai, Haidong Xu, Sihan Xu, Ying Zhang, and Xiaojie Yuan. BadPrompt: Backdoor Attacks on Continuous Prompts. In Annual Conference on Neural Information Processing Systems (NeurIPS). NeurIPS, 2022. 12 [5] Abhishek Chakraborty, Ankit Mondal, and Ankur Srivastava. Hardware-assisted intellectual property protection of deep learning models. IACR Cryptol. ePrint Arch., page 1016, 2020. 1, 13 [6] Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep Reinforcement Learning from Human Preferences. In Annual Conference on Neural Information Processing Systems (NIPS), pages 4299– 4307. NIPS, 2017. 11 [7] Junyoung Chung, Çaglar Gülçehre, KyungHyun Cho, and Yoshua Bengio. Empirical Evaluation of Gated Recur-
abs/1412.3555, 2014. 2 [8] Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why Can GPT Learn InContext? Language Models Secretly Perform Gradient Descent as Meta-Optimizers. In Annual Meeting of the Association for Computational Linguistics (ACL), pages 4005–4019. ACL, 2023. 13 [9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACLHLT), pages 4171–4186. ACL, 2019. 1, 2 [10] Nolan Dey, Gurpreet Gosal, Zhiming Chen, Hemant Khachane, William Marshall, Ribhu Pathria, Marvin Tom, and Joel Hestness. Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster. CoRR abs/2304.03208, 2023. 5 [11] Emily Dinan, Angela Fan, Adina Williams, Jack Urbanek, Douwe Kiela, and Jason Weston. Queens are Powerful too: Mitigating Gender Bias in Dialogue Generation. In Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8173–8188. ACL, 2020. 1 [12] Haonan Duan, Adam Dziedzic, Mohammad Yaghini, Nicolas Papernot, and Franziska Boenisch. On the Privacy Risk of In-context Learning. In Workshop on Trustworthy Natural Language Processing (TrustNLP), 2023. 13 [13] Wikimedia Foundation. Acl 2019 fourth conference on machine translation (wmt19), shared task: Machine translation of news. 12 [14] Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models. CoRR abs/2009.11462, 2020. 1 [15] Sepp Hochreiter and Jürgen Schmidhuber. Long Short-Term Memory. Neural Computation, 1997. 2 [16] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR), 2022. 2, 3 [17] Nikhil Kandpal, Matthew Jagielski, Florian Tramèr, and Nicholas Carlini. Backdoor Attacks for In-Context Learning with Language Models. CoRR abs/2307.14692, 2023. 13 [18] Brian Lester, Rami Al-Rfou, and Noah Constant. The Power of Scale for Parameter-Efficient Prompt Tuning. In Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3045–3059. ACL, 2021. 2, 3 [19] Xiang Lisa Li and Percy Liang. Prefix-Tuning: Optimizing Continuous Prompts for Generation. In Annual Meeting of the Association for Computational Linguistics and International Joint Conference on Natural Language Processing (ACL/IJCNLP), pages 4582–4597. ACL, 2021. 2, 3 [20] Xin Li and Dan Roth. Learning Question Classifiers. In International Conference on Computational Linguistics (COLING). ACL, 2002. 5 [21] Yi Liu, Gelei Deng, Zhengzi Xu, Yuekang Li, Yaowen Zheng, Ying Zhang, Lida Zhao, Tianwei Zhang, and Yang Liu. Jailbreaking ChatGPT via Prompt Engineering: An Empirical Study. CoRR abs/2305.13860, 2023. 12
[22] Pekka Malo, Ankur Sinha, Pekka J. Korhonen, Jyrki Wallenius, and Pyry Takala. Good debt or bad debt: Detecting semantic orientations in economic texts. J. Assoc. Inf. Sci. Technol., pages 782–796, 2014. 5 [23] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the Role of Demonstrations: What Makes In-Context Learning Work? In Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 11048–11064. ACL, 2022. 1, 3, 13 [24] Jishnu Mukhoti, Yarin Gal, Philip H. S. Torr, and Puneet K. Dokania. Fine-tuning can cripple your foundation model; preserving features may be the solution. CoRR abs/2308.13320, 2023. 4, 5 [25] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Annual Conference on Neural Information Processing Systems (NeurIPS). NeurIPS, 2022. 11 [26] Jane Pan, Tianyu Gao, Howard Chen, and Danqi Chen. What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task Learning. CoRR abs/2305.09731, 2023. 1, 3, 15 [27] Ashwinee Panda, Tong Wu, Jiachen T. Wang, and Prateek Mittal. Differentially Private In-Context Learning. CoRR abs/2305.01639, 2023. 13 [28] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Annual Meeting of the Association for Computational Linguistics (ACL), page 1525–1534. ACL, 2016. 5 [29] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language Models are Unsupervised Multitask Learners. OpenAI blog, 2019. 1, 2 [30] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In International Conference on Learning Representations (ICLR). ICLR, 2023. 11 [31] Timo Schick, Sahana Udupa, and Hinrich Schütze. SelfDiagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP. Transactions of the Association for Computational Linguistics, 2021. 1 [32] Emily Sheng, Kai-Wei Chang, Premkumar Natarajan, and Nanyun Peng. The Woman Worked as a Babysitter: On Biases in Language Generation. In Conference on Empirical Methods in Natural Language Processing and International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3405–3410. ACL, 2019. 1 [33] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and Efficient Foundation Language Models. CoRR abs/2302.13971, 2023. 5 [34] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia
Polosukhin. Attention is All you Need. In Annual Conference on Neural Information Processing Systems (NIPS), pages 5998–6008. NIPS, 2017. 2, 3 [35] Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers Learn In-Context by Gradient Descent. In International Conference on Machine Learning (ICML), pages 35151–35174. PMLR, 2023. 12 [36] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. In International Conference on Learning Representations (ICLR), 2019. 5 [37] Lixu Wang, Shichao Xu, Ruiqi Xu, Xiao Wang, and Qi Zhu. Non-Transferable Learning: A New Approach for Model Ownership Verification and Applicability Authorization. In International Conference on Learning Representations (ICLR), 2022. 1, 13 [38] Jerry W. Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, and Tengyu Ma. Larger language models do in-context learning differently. CoRR abs/2303.03846, 2023. 1, 3 [39] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. HuggingFace’s Transformers: State-of-the-art Natural Language Processing. CoRR abs/1910.03771, 2019. 5 [40] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An Explanation of In-context Learning as Implicit Bayesian Inference. In International Conference on Learning Representations (ICLR). ICLR, 2022. 13 [41] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 586–595. IEEE, 2018. 5 [42] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. OPT: Open Pre-trained Transformer Language Models. CoRR abs/2205.01068, 2022. 5 [43] Xiang Zhang, Junbo Zhao, and Yann LeCun. Characterlevel Convolutional Networks for Text Classification. In Annual Conference on Neural Information Processing Systems (NIPS), pages 649–657. NIPS, 2015. 5 [44] Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson. Universal and Transferable Adversarial Attacks on Aligned Language Models. CoRR abs/2307.15043, 2023. 12
Template #
Example
0
<Sentence>
Label: <Label>
1
<Sentence>
Sentiment: <Label>
2
<Sentence>
The sentiment is <Label>
3
<Sentence>
The sentiment of the text is <Label>
Table 13: List of templates taken from [26].
# A Experiment Details
# A Experiment Details A.1 Prompt Templates
# A.1 Prompt Templates
For each task, we use the minimum template and default label set. For FP, we use three extra templates for the study of the adaptive attack. All prompts and labels are presented in Table 13 and Table 14.
# A.2 Baseline Performance
For the model and size study, we provide the original ICL performance and utility for OPT-13B, Cerebras-13B, LLaMA-7B, and LLaMA-30B in Table 16, Table 17, Table 18, and Table 19, respectively.
# A.3 More Results
We report the utility performance for Section 6.5, including different models, different LLaMA model sizes, different numbers of unique demonstrations, different sampling sizes, and different PEFT methods in Table 20, Table 21, Table 23, Table 24, and Table 26.
# A.4 Training Details
For SFT and DPO, we use the same setup as Section 5 for fine-tuning the LoRA. We set the rank to 8, alpha to 32, and the dropout rate to 0.1. In addition, we use a batch size of 4 and a learning rate of 1 x 10−4 with the Adam optimizer. For SFT, we randomly select an incorrect label as the continuous on the target ICL prompt, and truncate the input length to 400. For DPO, we set the ground label as the rejected answer, and randomly select an incorrect label as the chosen answer. In addition, we quantize the model to 4 bits. We implement the training using Huggingface’s TRL package 6.
6https://github.com/huggingface/trl
Dataset
Label #
FP
1
negative/neutral/positive
2
terrible/neutral/good
SST-2
1
terrible/good/neutral
TREC
1
expression/entity/description/
human/location/number
AGnews
1
World/Sports/Business/Science
Dataset
Accuracy (%)
PPL
FP
0.80
31.42
SST-2
0.93
405.0
TREC
0.70
51.92
AGnews
0.90
11.25
Lambada
-
27.64
Table 15: The original ICL performance and utility of LLaMA13B for each dataset.
Table 15: The original ICL performance and utility of LLaMA13B for each dataset.
Dataset
Accuracy (%)
PPL
FP
0.73
98.01
SST-2
0.93
443.3
TREC
0.56
129.1
AGnews
0.78
25.81
Lambada
-
37.41
Table 16: The original ICL performance and utility of OPT-13B for each dataset.
Table 16: The original ICL performance and utility of OPT-13B for each dataset.
Dataset
Accuracy (%)
PPL
FP
0.55
100.3
SST-2
0.89
350.1
TREC
0.41
345.1
AGnews
0.77
33.81
Lambada
-
44.46
<div style="text-align: center;">Table 17: The original ICL performance and utility of Cerebras-13B for each dataset.</div>
Dataset
Accuracy (%)
PPL
FP
0.77
32.28
SST-2
0.95
572.9
TREC
0.80
52.61
AGnews
0.89
12.05
Lambada
-
28.39
Table 18: The original ICL performance and utility of LLaMA7B for each dataset.
Table 18: The original ICL performance and utility of LLaMA7B for each dataset.
Dataset
Accuracy (%)
PPL
FP
0.83
28.75
SST-2
0.95
509.9
TREC
0.92
51.18
AGnews
0.91
10.66
Lambada
-
26.45
Table 19: The original ICL performance and utility of LLaMA30B for each dataset.
Table 19: The original ICL performance and utility of LLaMA30B for each dataset.
Model
FP
SST-2
TREC
AGnews
Lambada
LLaMA-13B
