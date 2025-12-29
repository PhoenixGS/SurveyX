# n-context Example Selection with Influ
Tai Nguyen TAING@SEAS.UPENN.EDU University of Pennsylvania Eric Wong EXWONG@CIS.UPENN.EDU University of Pennsylvania
# Abstract
In-context learning (ICL) is a powerful paradigm emerged from large language models (LLMs). Despite its promises, ICL performance is known to be highly sensitive to input examples. In this work, we use incontext influences to analyze few-shot ICL performance directly from the in-context examples. Our proposed influence-based example selection method can identify both positive and negative examples, outperforming several baselines when evaluated on 9 SuperGLUE tasks. Our analysis uncovers up to a 16.3% performance gap between using the most negative in-context examples compared to the most positive. In a case study, we apply our influence-based framework to quantify the phenomena of recency bias in example ordering for few-shot ICL.1
# 1 Introduction
Large language models (LLMs) such as GPT-3 have recently become capable of in-context learning (ICL) [Bro+20]. In ICL, users provide the model with a few labeled examples as input before asking the model to make a prediction on a new example. This paradigm has enabled the rapid adaptation of LLMs to new tasks without requiring any modifications to the model. ICL has several advantages over traditional learning paradigms. First, the ability to do few-shot learning directly reduces the need for human-labeled data. Second, in contrast to other popular training paradigms such as finetuning a pretrained model [Rad+19; Dev+19], ICL enables inference without any gradient updates. Lastly, ICL also displays amazing versatility through different modes of prompting. Recent work shows that GPT-3 can do step-by-step reasoning when being demonstrated a few examples containing reasoning [Wei+22; Nye+22; Lyu+23]. Despite these promises, ICL performance is known to be highly variable. In particular, ICL volatility has been linked to biases such as the order of the examples [Lu+22], their templates [Lu+22; KT21], and example selection [Liu+22a]. Various mitigation methods were proposed to address this brittleness, such as model calibration [Zha+21] and template engineering [Liu+22b]. Given that not all in-context examples are equal, several others have focused on finding more optimal prompts. Liu et al. [Liu+22a] proposes a distance-based selection method, using semantic similarity to the validation query to rank candidate examples. Gonen et al. [Gon+22] finds a strong negative correlation between example perplexity and task performance. Similarly, Chen et al. [Che+22] suggests a sensitivitybased selection method which perturbs examples and chooses the ones with more robust predictions. While these methods have varying effectiveness, there lacks a consensus on which of these signals are most important in ICL. Motivated by this problem, our paper studies the relationship between influences and ICL, to better understand and quantify the impact of examples on ICL. Influences naturally lend to an offline example selection method that directly measures the effect of examples on ICL performance. In particular, we use in-context influences to measure and rank the impact of in-context examples on task performance. The framework can be customized to study different aspects of ICL, such as optimizing for the best classification accuracy or quantifying the impact of position.
Eric Wong
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5146/51463021-10bf-40ea-bf08-902cf70f1475.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Test accuracy increases when examples are selected in increasing in-context influence percen bins.</div>
<div style="text-align: center;">Figure 1: Test accuracy increases when examples are selected in increasing in-context influence percentile bins.</div>
On 8 language models and 9 natural language tasks, we demonstrate the efficacy of influence-based example selection in ICL at estimating the effect of training examples on downstream performance. We find that in-context influences outperform all other selection baselines at estimating ICL performance in both positive and negative selections. In-depth analysis exposes a significant gap between the most positive and most negative examples. For example, constructing prompts from the top influence bin improves ICL performance by up to 16.3% over the bottom influence bin on LLaMA-13B. Overall, our contributions are as follows:
• We study in-context influence as a metric for selecting and analyzing in-context examples in few-shot ICL. In both positive and negative selections, our method outperforms several baselines at estimating ICL performance. • We demonstrate a substantial performance gap between positively and negatively influential examples. Our framework quantifies this gap, and further confirms the variability of example-selection in ICL. • While we focus on classification accuracy, our framework generalizes to any combination of performance metric, model, and task. For example, we leverage our framework to quantify emergent phenomena in LLMs, such as the impact of recency bias in example ordering.
# 2 In-context Influences
A variety of methods have been developed to understand how training data affects model performance. To estimate this effect, some methods use gradient information [KL17; Koh+19; HWT20; Pru+20] while others retrain models on subsets of the training data [GZ19; Ily+22]. These methods all aim to quantify how a training example affects the prediction of a test example after training. Inspired by these frameworks, our goal is to trace how ICL performance depends on the in-context examples and calculate the corresponding influences. Our setup follows the retraining-based influence frameworks, which have two main steps. Let S be a training set, and let f (S) be the validation performance after training on a dataset S. Retraining-based influences first collect a “dataset” of M training runs D = {(Si, f (Si)}M i=1 where Si ⊆S are random subsets of the original training dataset. The second step is to use this dataset to estimate the influence of each example x ∈S, e.g. by learning a linear mapping [Ily+22].
Influences in k-shot prompting. To compute influences for in-context examples, we leverage the following key observation: in ICL, “training” a model on a subset S′ reduces to prompting the model on a sequence containing S′. Consequently, constructing the dataset D of training runs for ICL requires no gradient updates and is as costly as computing forward passes through the model. This drastically reduces the cost of calculating retraining-based influences, and can be calculated with only query-access to the model. Specifically, for the first step, we construct the dataset of training runs D by performing k-shot prompting with subsets S′ ⊆S where |S′| = k. For a fixed subset S′, the performance of the resulting prompt containing S′ is measured with a validation query appended to the end of the prompt. We repeat this inference over the entire validation set to compute the metric f (S′), which measures the validation performance after prompting with S′. This metric can be any evaluation method suitable for a natural language task—here, we focus on classification accuracy. We repeat this process over multiple random subsets S′ ⊆S until each example in S has been seen in multiple prompts, resulting in a dataset of prompting runs D = {(Si, f (Si)}M i=1. In the second step, we calculate the influence of each in-context example. We define the in-context influence I(xj) as the effect of an example xj on few-shot ICL performance. In other words, the influence is the difference between the average performance of prompts including xj and the average performance of prompts omitting xj. More formally, this can be written as:
where Si is a specific uniformly sampled subset, M is the number of total subsets used to estimate influences, Nj is the total number of subsets containing example xj, and f (Si) is the performance metric when evaluated on the validation set. When f measures validation performance, a higher score for I(xj) corresponds to a higher average improvement in validation performance when including xj in the prompt, analogous to the meaning of influences in the classic, non-prompted setting. As the number of collected subsets grows, estimates of in-context influences become more accurate. A sufficiently large M is one with good coverage for each example—this means that each xj ∈S is seen multiple times. In our experiments, each xj gets seen at least 30 times on average.
Influence-based Example Selection. We use the proposed in-context influences to identify highly impactful in-context examples. Specifically, we can use the top influential examples to create the “best” prompt for ICL (with respect to the influence scores). On the converse, we can also use the bottom influential examples to
create the “worst” performing prompt for ICL. In summary, to do example selection for ICL, we carry out he following steps: 1. Prompt the model on random training subsets and measure validation performance to create the dataset of prompting runs D. 2. Calculate the in-context influence I(xj) for each example xj ∈S following Equation 1 using D. 3. Select k examples with the most positive influences to use for k-shot prompting. The examples can be arranged in any ordering. A summary of the entire pipeline is shown in Algorithm 1.
# 2.1 Cost Analysis & Hyperparameters
Training cost. Retraining-based influence frameworks [Ily+22; GZ19] can require training hundreds of thousands of models. This is necessary to collect a sufficiently large enough dataset D to accurately estimate influences. In contrast, the cost of computing in-context influences is relatively cheap, as we do not need to train an end-to-end model. Instead of training, we simply prompt the LLM using a randomly sampled S′ from original training set S. Thus, the complexity of calculating the validation performance from a sampled subset is proportional to a forward pass through the LLM.
Size of subsets. Our method has one parameter k, which controls the size of the random subsets S′ ⊆S from which D is constructed. For ICL, k = |S′| corresponds to the number of in-context examples given in the prompt. Unlike in the traditional setting, the context window length limit enforces a hard upper limit on the number of examples an LLM can be trained on via prompting. These context windows are typically limited to 2048 characters. Taking the context window into account, we select k to be the maximal number of examples that can be inserted into the context window. The value of k can vary by different choices of model (context window size) and the lengths of the individual examples in a dataset. Since the number of shots can impact ICL performance, we keep a consistent k for each model and task. Table 6 provides the precise k value associated with each task.
# 3 Experiments
We conduct experiments to obtain in-context influences for 72 combinations of natural language tasks and LLMs. The goal is to a select a set of good ICL examples by running influences on the Dev set. At test time, such a set requires no further modification or computation.
Datasets. We choose 9 datasets for our study, 5 of which are binary classification tasks and 4 are multichoice tasks.2 These datasets cover a wide range of common natural language tasks, including textual entailment (RTE), question-answering (PIQA), and text summarization (BoolQ). Each example instance has a definitively correct answer, making them convenient to be evaluated through classification accuracy. Beyond acquiring the original data, we subsample Train/Dev/Test sets with 400/200/500 in-context examples for each task.
K-shot selection. We select examples uniformly by their label class. If a multi-choice task has 3 options, each option would make up roughly one-third of the examples. This balance prevents majority label bias [Zha+21] from skewing the model’s inference.
Inference details. There are multiple ways to do inference on multi-choice tasks with autoregressive models [Hol+21]. We follow one popular approach, which ranks all possible continuations to a prompt and chooses the continuation with the highest log-likelihood. Thus, given a prompt x0:m and a possible prompt continuation xm:n, the score for xm:n can be defined as:
where P(xj|x0:m) is the likelihood of token xj given the preceding context tokens x0:j. The prediction is then defined as the most likely continuation, arg maxxm:n ℓ(xm:n). We do not perform any token length or answer normalization tricks [Bro+20].
# 3.1 Influence-based methods
In our main results, we evaluate the effectiveness of influence-based example selection using the following strategies:
In our main results, we evaluate the effectiveness of influence-based example selection using the following strategies:
1. Influence (+/−). We select examples with the most positive or negative influence scores according to Algorithm 1. If influence estimates are meaningful, we would expect examples with positive influences to perform the best among all baselines, while examples with negative influences would have the poorest performance. 2. In-context datamodels. We consider an alternative influence-based example selection based on the datamodels [Ily+22] framework that we adapt for ICL. Specifically, we fit a linear model3 gθ on the dataset D of input-output pairs from Section 2 to predict validation performance:
2. In-context datamodels. We consider an alternative influence-based example selection based on the datamodels [Ily+22] framework that we adapt for ICL. Specifically, we fit a linear model3 gθ on the dataset D of input-output pairs from Section 2 to predict validation performance:
gθ(S′) = θ · 1T S′ + θ0
where S′ ⊆S is an example subset and 1S′ is an indicator vector with the dimension of the training set S. A value of 1 at position i indicates that the example i is included in the subset S′ and a value of 0 means otherwise. Following the datamodels framework, we can treat the parameters θ as influence estimates, and select in-context examples based on these estimates. Note that θ has a close connection to in-context influences as they both use the same training set D, but in-context datamodels assumes a linear model.
# 3.2 Non influence-based methods
We compare influence-based example selection methods described in the previous section to the following baselines, which optimize various metrics for selection.
3. One-shot. We do one-shot prompting (k = 1) on each Train example and rank them by their accuracy on the Dev set. One-shot selection assumes that we can extrapolate the performance of one-shot prompting to the few-shot setting.
<div style="text-align: center;">Table 1: Positive example selection methods on OPT-30B and their overall Rank Aggregation.</div>
OPT-30B
All Models
Binary Classification (Acc. ↑)
Multi-choice (Acc. ↑)
Rank Agg (↓)
PIQA
BoolQ
RTE
WIC
WSC
ARC-c
ARC-e
HS
OBQA
All Tasks
Perplexity (+)
76.80.0
72.70.2
61.90.3
53.50.2
43.50.6
40.30.1
76.30.1
56.60.0
28.50.1
4.59
Random
77.00.0
71.10.2
63.20.2
54.80.1
49.10.5
41.50.1
76.00.1
55.40.1
29.60.1
4.37
Similarity (+)
77.70.1
70.10.4
63.90.1
53.30.1
57.10.7
42.00.1
76.20.1
56.70.0
29.30.0
4.33
One-shot (+)
77.50.0
76.50.1
52.40.1
51.10.2
61.60.0
41.50.0
76.10.1
56.60.1
31.20.0
4.24
Best set
76.90.0
72.60.0
64.10.3
55.10.2
54.80.4
40.80.0
75.80.1
56.10.0
31.50.0
3.62
IC Datamodels (+)
78.10.0
77.00.0
65.90.1
51.40.2
56.40.1
42.10.0
76.60.0
58.20.0
31.70.1
2.98
Influence (+)
78.00.0
74.10.1
64.60.1
52.50.1
51.40.3
41.60.1
77.00.0
57.40.0
33.30.0
2.96
<div style="text-align: center;">Table 2: Negative example selection methods on LLaMA-13B and their overall Rank Aggregation</div>
LLaMA-13B
All Models
Binary Classification (Acc. ↓)
Multi-choice (Acc. ↓)
Rank Agg (↓)
PIQA
BoolQ
RTE
WIC
WSC
ARC-c
ARC-e
HS
OBQA
All Tasks
Similarity (-)
79.20.0
83.20.0
58.70.1
54.60.2
43.90.5
51.10.0
82.30.0
62.10.0
37.10.0
5.19
Random
78.50.1
82.60.1
61.10.3
51.80.2
42.90.4
50.40.1
82.70.0
62.50.1
35.70.1
4.94
Worst set
78.80.0
79.20.1
54.10.2
53.30.1
45.70.6
50.30.1
83.00.0
62.10.1
33.60.1
4.37
Perplexity (-)
74.90.0
82.40.1
57.90.1
55.40.2
42.80.4
49.40.0
81.40.0
58.70.0
33.10.1
3.69
One-shot (-)
78.70.0
68.20.2
53.90.1
53.10.1
55.40.7
50.00.1
81.40.0
61.00.0
26.10.1
2.96
IC Datamodels (-)
78.50.0
69.30.3
50.00.0
51.60.2
38.90.1
50.00.1
82.80.1
61.80.0
22.00.1
3.03
Influence (-)
78.60.0
68.30.3
50.00.0
50.60.2
39.80.3
49.30.1
82.40.1
61.60.0
22.90.1
2.90
4. Semantic similarity. Examples close to the test queries in the embedding space can substantially improve ICL performance on semantic parsing tasks [Liu+22a]. We search for a set of examples with the closest distance to Dev set, then applying them on the unseen Test set. We use RoBERTa-large [Liu+19] sentence encoder implemented by Reimers and Gurevych [RG19]. 5. Perplexity. Perplexity measures the degree of uncertainty of the LLM when generating new tokens, where a lower perplexity means a higher confidence on the example. On perplexity, Gonen et al. [Gon+22] has linked prompt confidence to good ICL performance. We follow this insight to select examples based on their individual perplexity, which is more computationally friendly than calculating full prompt perplexity across many different example combinations. After selecting a set of k examples using the proposed baselines, we construct the prompt by ordering
After selecting a set of k examples using the proposed baselines, we construct the prompt by ordering examples randomly. We compute Test accuracy and rank all baselines by single-task accuracy. We aggregate the ranks of each method my taking their average for both positive and negative example selection. In the main results, averages and standard errors are reported over 7 seeds.
# 3.3 Results
Positive selection. Table 1 shows results for all positive example selection baselines. Overall, influencebased selection methods outperform all non-influence counterparts. Specifically, across all models and tasks, Influence (+) and IC Datamodels (+) frequently select the best set of examples for an average rank of 2.98 and 2.96 (where both methods are considered in the ranking). Best set selection is our next most competitive baseline, with the rest of the methods trailing far behind. For Best set, strong performance on the task WIC
(word sense disambiguation) contributes mostly to this success. Likewise, One-shot example selection sees exceptional performance on WSC, but does not work well for other tasks. We also note that random selection ranks better than Perplexity (+), although the latter outperforms random selection on many tasks (see Table 7 in Appendix).
Negative selection. Similarly, our influence-based example selection methods can consistently identify low-performing examples. As shown in Table 2, results indicate that Influence (-) achieves the highest rank (2.90) among all other methods. Notably, in this context, One-shot slightly outperforms IC Datamodels (-), suggesting that selecting examples based on their individual validation performance can be effective Compared to positive selection, the ability to pinpoint negative examples is equally meaningful: we can avoid these examples to achieve better ICL performance, or further study them to identify the factors that make them ineffective.
Binary vs. Multi-choice. Our experiments find influence-based example selection methods to work better on Multi-choice tasks compared to Binary classification tasks. In most models (see Table 7 in Appendix), Influence (+) and IC Datamodel (+) often outperform all other methods on multi-choice tasks, but less so on binary classification tasks. Specifically, for PIQA and WIC, the small gaps between the performance of Influence (-) and Influence (+) suggest that our influence-based method might not have captured example helpfulness well for these tasks. Many factors related to both the model and task could explain such disparity. For one, Zhao et al. [Zha+21] demonstrates that LLMs can have a strong bias towards selecting certain labels for ICL, which could hinder both model performance and influence attribution. We suspect that these biases can exacerbate when the LLM is asked to choose between binary labels (T/F) compared to many labels in the multi-choice setup. Furthermore, the inconsistent scaling of 3 OPT models on the SuperGLUE benchmark for few-shot ICL could play a factor [Zha+22a]. The capabilities of the models themselves factor majorly into the accuracy improvements of our selection methods.
# 4 Analysis
In this section, we analyze in-context influences across a number of distinct axes. Specifically, we study (1) the cost of influence-based example selection, (2) distinguishing factors between examples with positive and negative influences, (3) influence agreement between model families, and (4) the scaling behavior of influence-based selection across the number of shots. We conclude our analysis with a case study quantifying the effect of recency bias in example ordering.
# 4.1 Cost Comparison
Figure 2 compares the cost of influence-based example selection against Best set and One-shot example selections on different tokens budgets for multi-choice tasks.4 Recall that computing our in-context influences on more training runs often leads to more accurate influence estimations. Our visualization demonstrates that in-context influences can realize favorable gains over other selection methods at a fraction of the full budget (20M tokens). In fact, in-context influences also scale well beyond this number, while the same effect does not guarantee for Best set. Note that One-shot selection scales linearly by the size of the Train set S and Dev set, while Influence (+) has more flexible scaling depends on compute budget.
# 4.2 Do models agree on high-influence examples?
This section analyzes whether or not the best and worst examples on a task are shared across models. When considering the overlap between all 7 individual models, our work finds that they rarely agree on the most positive and negative influence examples (≤4 for all tasks). However, within the same model families, we identify a decent overlap. Figure 3 plots the inter-family agreement between three families considered in our study. Compared to OPT, both GPT-NeoX and LLaMA models often identify a smaller set of top and
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/25cf/25cf0b61-540f-4724-a225-a8addb7bccc8.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2a60/2a6066f2-b99e-4092-9180-5f9f73ee91b6.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Token budget comparison for different baselines evaluated on LLaMA-7B (|S| = 400).</div>
ID
Prompt
Influence
Reason
PIQA
12444
Goal: flashlight
Answer: shines a light
-0.001854
Unnatural
WIC
3890
Go to the supermarket and buy some tea.
Would you like some tea?
question: Is the word ’tea’ used in the same sense in the
two sentences above?
answer: false
-0.007068
Mislabeled
OBQA
3771
Context: Single cell organisms can put an animal in the
Answer: emergency room
-0.006058
Unnatural
bottom influence examples, while agreeing on these examples more often. This suggests that our in-context influences are picking up signals specific to the nature of these model families. These can include variations in model architecture, training data, training stability, tokenizers, and others [Zha+22a; Bla+22; Rad+19; Tou+23]. For instance, model performance has been linked to term frequencies in the pretraining data [Raz+22].
# 4.3 Negative vs. Positive Examples
Prior work has associated various characteristics with examples that are strongly positive or negative [KL17; HWT20; Ily+22]. Positive examples are found to sometimes be instances of data leakage during the training process, while negative examples are often mislabeled. For LLMs, we do not have access to the pretraining data to identify data leakage. However, we identify many instances in the bottom influence bin that appear as either “unnatural” or mislabeled. Table 3 shows instances of these negative examples and their associated potential cause. For PIQA example #12444, the overall plausibility of the statement could improve if the order of statements Goal and Answer gets switched. Alternatively, a better template could possibly help achieve better input-output coherence. We suspect that the prompt template might play an important role in determining the influence of an example. Additionally, we identify WIC example #3890 as a falselyannotated instance. Related to input-label mapping, Min et al. [Min+22b] has shown that label correctness is not important for good ICL performance.5 Quantitatively, we measure several metrics from the literature to compare examples with positive and negative influences. As Figure 4.3 illustrates, we find little to no association between in-context influences
<div style="text-align: center;">Figure 3: Model family agreement (overlap) when considering all examples (union) in the Top and Bottom 20th influence bins.</div>
Figure 3: Model family agreement (overlap) when considering all examples (union) in the Top and Bottom 20th influence bins.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d7e3/d7e31170-b8b1-480a-802d-7379f7852adb.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: On Superglue-WIC, in-context influences do not correlate with any previously known e characteristics.</div>
<div style="text-align: center;">Figure 4: On Superglue-WIC, in-context influences do not correlate with any previously known example</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/df1a/df1a55ac-be74-44d9-ae64-5b78743ae16d.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: On LLaMA-7B, influence-based example selection scales well with increasing k-shot.</div>
ICL Sensitivity. Naturally, we can leverage in-context influences to quantify the gap between the most positive and negative examples in ICL. For example, on OpenBookQA, we observe an impressive 16.3% accuracy difference between the best and worst in-context examples on LLaMA-13B (See Table 4 in Appendix). Our framework adds to a list of previous works reporting ICL sensitivity [Liu+22a; Lu+22; Zha+21].
Influence bins. Additionally, we demonstrate that our influence framework can analyze example selection in more fine-grain. For this experiment, we group examples by their influence percentile, where each bin contains 20% of the Train set. From these bins, we randomly select a set of k examples in any ordering for 10 seeds for evaluation. If the influences are meaningful, we expect to see increasing performance gains as examples are selected in increasing percentile bins along the x-axis. Figure 1 identifies clear positive trends confirming our hypothesis on most models and tasks. This shows that in-context influences produce well-behaved results when examples are selected in specific influence regions. There are few exceptions, such as the BoolQ task on LLaMA-7B, where selecting examples in increasingly positive influences does not consistently improve validation accuracy.
# 4.4 How do in-context influences generalize across k-shot?
Thus far, our estimation of in-context influences has assumed a many-shot setting where a maximal number of examples is packed into the context window. In this study, we are interested in knowing how in-context influences generalize to different numbers of in-context examples k. In comparing different selection methods, Figure 5 finds that the impact of in-context influences is most prevalent when k is many (generally ≥8). At one-shot and very few-shot, in-context influences can sometimes perform worse than other methods (RTE)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1ef4/1ef4b722-e45e-4aeb-b93b-ec7025718e95.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ec50/ec50bb0a-4ef7-42f6-af8a-39ee7698f3b7.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Aggregated influences of each position in 4-shot prompting. Influence magnitudes are bigger at later positions.</div>
but steadily improve with increasing k. In contrast, the performance for One-shot and random selection not always improve (and sometimes decline) with increasing k (on Hellaswag and RTE).
# but steadily improve with increasing k. In contrast, the performance for One-shot and random selection do not always improve (and sometimes decline) with increasing k (on Hellaswag and RTE).
# 4.5 Case study: Example Ordering
We want to apply our influence-based framework to demonstrate its effectiveness for studying a phenomenon in ICL, which deals with recency bias in example ordering [Lu+22; Zha+21].
Setup. To do this, we randomly choose 100 examples from another SuperGLUE task, CB, and assign them into 4 groups for 4-shot prompting. On OPT-6.7B, we compute an in-context influence estimate for each example-position pair over all possible ordering permutations (4! = 24). Given an arbitrary example, we are interested in quantifying its impact at any position in the ordering and the overall influence of each position. Results. Figure 6 confirms the presence of recency bias in ICL [Zha+21], showing that influence estimates of examples increase as their position ID moves down in the order. Between Position #0 and Position #3, there is a notable 2% difference in the estimated absolute influence. Figure 7 elaborates on this result: on the same set of in-context examples, the influence estimates computed in position #3 has the biggest spread among all positions. Once again, we observe a steadily increasing trend in the widths of the spread as an example is moved down in order.
Setup. To do this, we randomly choose 100 examples from another SuperGLUE task, CB, and assign them into 4 groups for 4-shot prompting. On OPT-6.7B, we compute an in-context influence estimate for each example-position pair over all possible ordering permutations (4! = 24). Given an arbitrary example, we ar interested in quantifying its impact at any position in the ordering and the overall influence of each position
Results. Figure 6 confirms the presence of recency bias in ICL [Zha+21], showing that influence estimates of examples increase as their position ID moves down in the order. Between Position #0 and Position #3, there is a notable 2% difference in the estimated absolute influence. Figure 7 elaborates on this result: on the same set of in-context examples, the influence estimates computed in position #3 has the biggest spread among all positions. Once again, we observe a steadily increasing trend in the widths of the spread as an example is moved down in order.
# 5 Related Work
Example selection. In parallel and independent work, Chang and Jia [CJ22] also study the use of influences for selecting in-context examples for k-shot prompting, and also find that influence-based selection outperforms baseline methods. While we both consider influence estimates based on datamodels and data shapley influences, there are some differences. Chang and Jia [CJ22] integrate the position of an in-context example into the datamodel to directly calculate the influence of position for each example. In contrast, we consider the vanilla datamodel that does not model position, but demonstrate positional bias in a case study in Section 4.5. Although the formulation of the CondAcc score from Chang and Jia [CJ22] may appear slightly different from our influence metric, Chang and Jia [CJ22] prove in their Appendix that the two quantities rank examples identically. The experimental setups cover two distinct use-cases – Chang and Jia [CJ22] focus on a smaller number of in-context examples (i.e. k = 4) and find that influences could greatly reduce the variance of ICL, while we study a large number of examples (i.e. k up to 52) that also leads to less variance and performance gains. Finally, the corresponding analyses complement each other well. Chang and Jia [CJ22] analyze the embedding distance of examples, while we analyze the scaling pattern of the number of in-context examples k and the level of influence agreement across model families. Both work find little
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8bdc/8bdc085c-0201-49ff-8fb6-dbc82dd760b3.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Influence distribution of each position in 4-shot prompting. Bigger spreads are observed at later positions.</div>
Figure 7: Influence distribution of each position in 4-shot prompting. Bigger spreads are observed at later positions.
correlation between in-context influences and known signals such as example perplexity. The combined analyses present a more comprehensive understanding of influence-based example selection for ICL. Outside of in-context influences, examples have been found to be inequal when used in ICL. Liu et al. [Liu+22a] finds that the best in-context examples are the ones most semantically similar to the test sample, which translates well for semantic parsing tasks. Chen et al. [Che+22] links exemplars to a sensitivity measure, while Gonen et al. [Gon+22] recently shows a correlation between prompt perplexity and model performance. Others have improved ICL performance by focusing on prompt retrieval [RHB22], or applying reinforcement learning to improve ICL performance through prompt editing [Zha+22b]. Our in-context influence framework differs in its focus on identifying good examples from a Dev set that generalize well to any unseen evaluation, removing the need to perform any prompt editing or retrieval at test time. In-context learning. ICL comes with high volatility to factors beyond example selection. In the few-shot setting, models have shown a tendency to overly rely on the most frequent labels (majority bias) or labels that appear at late positions in a prompt (recency bias) [Zha+21]. The latter suggests that the ordering of examples can be optimized for performance gain [Lu+22]. The prompt template – the format in which the example is presented – also matters [Min+22a]. Other findings have discovered that correct input-label mapping has little relevance [Min+22b] and example diversity is more important [Su+22]. Recently, Aky¨urek et al. [Aky+22] links the underlying computations of ICL to linear algorithms. Training data influence. Influence functions [KL17] have been used as a way to trace a model’s output back to the training data. Influence of a specific training point measures the change in a model’s performance when the point is removed from the training set. Data Shapley [GZ19] and Ilyas et al. [Ily+22] measure similar quantities via retraining the model on subsets of the dataset. Outside of individual attributions, influence functions have also been used to measure group effects, where prior work found the influence estimates of individual data points to be the lower bound of groups [Koh+19].
Training data influence. Influence functions [KL17] have been used as a way to trace a model’s output back to the training data. Influence of a specific training point measures the change in a model’s performance when the point is removed from the training set. Data Shapley [GZ19] and Ilyas et al. [Ily+22] measure similar quantities via retraining the model on subsets of the dataset. Outside of individual attributions, influence functions have also been used to measure group effects, where prior work found the influence estimates of individual data points to be the lower bound of groups [Koh+19].
# 6 Conclusion
Our work proposes in-context influences as a way to analyze and select examples for ICL. Influence-based example selection methods (in-context influences and in-context datamodels) outperform all baselines for both positive and negative selections, showing stronger results on multi-choice tasks compared to binary classification tasks. In-context influences can identify problematic examples, scale performance with the choice of k-shot, and generalize to many nidek families. In a case study, we further examine known biases found in ICL such as recency bias in example ordering. Our work adds to a growing body of work that aims to understand and debug different emerging phenomena in LLMs. One limitation of influence-based frameworks is that they predict ICL performance from a fixed training set. However, practitioners can generate original prompts and examples, which may not exist in the training set. One potential research direction is to predict the performance of any input example constructed on the fly, in addition to those in the training set. Our influence-based framework can also be leveraged to study ICL beyond classification performance. For example, future work can potentially calculate influences for other natural language tasks such as text generation, summarization, or other multi-task settings.
# 7 Full Results
We provide more results, experimental, and discussion details that did not fit into the main paper
# 7.1 Influence distribution
Table 4: Mean difference of test accuracy (%) between the top 20 and bottom 20 percentile bin for ea model-task pair. The disparity between the two groups is clear, though it may vary by choice of model a task.
GPT-J
OPT-6.7B
LLaMA-7B
LLaMA-13B
PIQA
0.26
-1.47
-0.10
0.90
BoolQ
0.20
1.33
4.30
4.40
RTE
1.33
5.10
11.80
10.07
WIC
4.23
2.00
3.57
3.00
WSC
-5.84
-8.38
-1.69
10.77
Arc (Chal.)
-0.83
1.16
0.66
4.03
Arc (Easy)
0.23
0.00
2.70
0.96
Hellaswag
2.10
1.54
3.04
3.32
OBQA
4.27
5.96
7.60
16.34
Table 4 shows the performance gaps between using the most positive and the most negative influence examples for ICL inference. Figure 9.4 plots the distribution of influence estimates for all models and tasks. Figure 9.4 visualizes the fine-grain behavior of influence-based example selection when examples are selected in increasing order of influences.
# 7.2 Choice of k-shot for in-context influences
# 8 Discussion
# 8.1 Can linear datamodels predict in-context learn
Recall from Section 3.1 that we train linear in-context datamodels to derive θ as another influence measure for example selection. To evaluate these datamodels, we hold out a fraction of the training pairs (arbitrarily selected) from the collection process and use them afterwards as the ground truth. We do this for each model and task combination. If the Pearson correlation (ρ) between the predicted outputs and actual outputs are strong and statistically significant, we say that the linear datamodels models have capably captured the relationship between the in-context examples and ICL performance. Figure 9.4 visualizes the correlation between the predicted and actual model outputs for all models and tasks. We observe strong linear trends across the board, implying that the fitted in-context datamodels can predict few-shot ICL performance on unseen subsets of examples. Among all tasks, SuperGLUE-WSC is the most difficult to predict, which can be explained by the high variance from having the smallest Test set.
# Erratic behavior with OPT models on SuperGLUE
Authors of OPT report the model’s erratic behaviors when evaluated on many SuperGLUE tasks [Zha+22a]. Specifically, on the task WSC, zero- and multi-shot performance do not improve with respect to scale. They suspect that the small size of the validation sets in these datasets can be a factor. There are also reported
accidents during the training process related to hardware failures and loss divergences. These factors could partially explain signals found in our in-context influence estimates.
# 9 Implementation Details
# 9.1 Models
Language models. All autoregressive models are downloaded from their HuggingFace checkpoints using the transformers module6. To conserve memory, we load all models in 16FP half precision. We thank the authors of these models for making their work available to the research community.
Seed. By default, we keep a fixed seed=42. For experiments involving random example ordering, we also use other seeds in {51, 56, 67, 75, 82, 98}.
# 9.2 Datasets
All datasets were downloaded using Huggingface’s datasets module. Table 6 details the sizes of the ubsampled sets and the number of shots that fit in the in-context windows. • SuperGLUE [Wan+19] This benchmark includes 5 binary classification tasks: BoolQ, RTE, WIC, WSC, and CB. • PIQA [Cla+18] This benchmark includes 2 multi-choice tasks: AI2 Arc (Challenge) & AI2 Arc (Easy). • Hellaswag [Zel+19] Single multi-choice task: Hellaswag. • OpenBookQA [Mih+18] Single multi-choice task: OpenBookQA.
<div style="text-align: center;">Table 5: Models used in our work.</div>
Model
Parameter #
Window
Open-source
GPT-J
6B
2048
√
GPT-NeoX
20B
2048
√
OPT-6.7B
6.7B
2048
√
OPT-13B
13B
2048
√
OPT-30B
30B
2048
√
LLaMA-7B
7B
2048
√
LLaMA-13B
13B
2048
√
# 9.3 Prompts
Table 9 shows full prompt formats used the paper.
# 9.4 Hardware
We run all experiments on the NVIDIA A100 and NVIDIA RTX A60
Table 6: Datasets used in the paper. We sample 400 examples for train and 200 examples for  wherever possible. k denotes the number of demonstrations necessary to fill up the 2048 charac context windows.
Type
| Train |
| Dev |
| Test |
k
PIQA
Binary
400
200
500
38
Superglue BoolQ
Binary
400
200
500
10
Superglue RTE
Binary
400
200
500
12
Superglue WIC
Binary
400
200
500
32
Superglue WSC
Binary
400
104
154
32
AI2 Arc (Challenge)
MC
400
200
500
46
AI2 Arc (Easy)
MC
400
200
500
52
Hellaswag
MC
400
200
500
18
OpenBookQA
MC
400
200
500
52
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0523/0523f75c-1714-4247-91b7-4bbb2ca6e94e.png" style="width: 50%;"></div>
Figure 8: Linear in-context datamodels can predict ICL performance on arbitrary subset S′. Pearson correlation is calculated over all tasks for the model.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/853b/853b5acc-db57-4ff0-a034-a0c4917a003d.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: Influence distributions across all models and tasks. A wide spread signifies existence  high-influence points.</div>
<div style="text-align: center;">Figure 9: Influence distributions across all models and tasks. A wide spread signifies existence of many high-influence points.</div>
<div style="text-align: center;">9: Influence distributions across all models and tasks. A wide spread signifies existence of many fluence points.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/02b9/02b9923e-5a70-4bc5-ba74-78fdd5fd5e3b.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10: In most models and tasks, Test accuracy increases when in-context examples are selected increasing influence percentile bins. Many task and model observes linear trends outside of few exceptio (ie. WSC).</div>
<div style="text-align: center;">Figure 10: In most models and tasks, Test accuracy increases when in-context examples are selected in increasing influence percentile bins. Many task and model observes linear trends outside of few exceptions</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7edd/7edd4813-b2b3-43cc-982f-01bfb5c001b5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 11: How different positive example selection methods generalize with the number of k demonstrations.</div>
PIQA
BoolQ
RTE
WIC
WSC
ARC-c
ARC-e
HS
OBQA
Rank (↓)
GPT-J-6B
One-shot (+)
75.00.0
53.30.1
50.00.0
50.00.0
61.70.0
37.60.0
73.70.0
49.10.0
30.60.0
4.57
Random
75.70.1
61.70.3
56.20.3
51.90.3
48.50.4
37.70.1
73.30.0
49.10.1
27.60.1
4.32
Perplexity (+)
76.20.0
64.40.1
54.10.2
50.10.0
38.50.2
38.40.1
73.10.0
49.10.0
26.80.0
4.22
Similarity (+)
75.50.0
61.80.1
59.00.2
51.90.2
55.80.4
37.60.1
72.90.0
49.70.0
27.00.0
4.06
IC Datamodels (+)
75.30.0
57.50.1
50.50.0
55.70.2
46.00.3
38.50.1
73.60.0
50.50.0
30.90.0
3.46
Influence (+)
75.40.0
62.10.2
50.40.1
55.30.2
49.70.6
39.40.1
73.50.1
51.10.0
30.50.1
3.17
Best set
76.00.0
65.00.1
59.20.2
51.70.2
52.70.3
37.70.1
73.90.1
49.40.0
29.60.0
3.14
GPT-NeoX-20B
Perplexity (+)
76.60.0
73.20.1
63.20.2
51.90.2
42.30.4
43.20.0
78.00.0
54.40.0
29.50.0
4.97
Random
77.30.0
67.00.4
62.40.3
51.70.2
43.50.6
43.90.1
77.80.1
55.10.1
30.30.0
4.41
Similarity (+)
77.00.0
64.10.5
65.00.3
51.20.1
56.00.6
43.50.1
77.90.1
54.80.1
29.80.1
4.35
Best set
77.60.0
73.70.1
63.70.2
50.50.2
46.30.5
43.50.0
77.60.1
55.10.1
31.30.1
4.08
One-shot (+)
76.50.0
57.30.1
53.50.2
48.90.1
61.70.0
44.50.1
78.50.1
55.90.0
32.80.0
4.06
Influence (+)
78.00.0
73.60.2
65.30.1
51.90.2
47.20.4
43.80.1
78.70.1
54.90.1
32.70.0
2.95
IC Datamodels (+)
77.70.0
75.20.0
66.30.2
51.60.2
44.30.4
44.10.0
79.50.1
55.50.1
32.90.0
2.52
LLaMA-7B
Similarity (+)
77.80.0
79.60.1
64.30.3
52.50.3
40.20.3
44.70.1
76.90.1
58.20.0
31.90.1
4.60
One-shot (+)
77.20.0
78.20.1
60.90.3
51.60.1
39.10.1
43.60.0
78.70.0
59.90.0
33.50.1
4.52
Perplexity (+)
78.20.0
78.10.1
68.10.1
50.40.1
42.40.6
44.20.1
77.30.1
59.30.0
32.00.1
4.32
Random
78.00.1
80.30.1
63.40.2
51.10.2
41.70.3
44.30.0
78.70.1
58.80.1
32.30.0
4.11
Influence (+)
78.00.0
72.70.3
65.70.2
54.00.3
43.00.4
44.80.0
78.70.1
60.00.0
38.30.0
3.25
IC Datamodels (+)
77.70.0
75.40.3
65.10.1
52.90.3
42.30.4
45.40.0
78.90.0
60.20.1
38.70.1
3.19
Best set
78.20.0
81.10.1
67.60.1
51.70.1
44.50.3
45.00.1
77.70.0
59.80.0
35.10.0
2.90
LLaMA-13B
Similarity (+)
78.50.0
81.90.1
57.30.3
54.00.3
42.20.6
50.10.1
82.80.0
62.10.0
36.30.1
4.78
Perplexity (+)
79.10.0
82.40.1
61.80.3
55.00.2
40.80.1
50.50.0
82.50.0
61.60.0
35.60.1
4.57
Random
78.50.1
82.60.1
61.10.3
51.80.2
42.90.4
50.40.1
82.70.0
62.50.1
35.70.1
4.51
One-shot (+)
78.10.1
84.50.1
58.30.1
50.00.0
38.30.0
52.60.0
82.30.0
63.30.0
38.80.1
4.43
Best set
78.70.0
83.20.1
69.40.3
54.70.2
46.90.5
51.90.1
82.50.0
63.70.0
36.40.1
3.14
IC Datamodels (+)
78.80.0
83.90.1
66.60.1
54.20.2
41.90.5
52.90.0
82.90.0
62.30.0
42.50.0
2.86
Influence (+)
78.70.0
84.30.1
68.40.2
54.50.1
42.50.5
53.10.0
82.70.0
62.60.0
42.70.1
2.70
OPT-6.7B
Perplexity (+)
76.00.0
69.80.1
51.10.0
49.30.1
47.60.6
37.60.1
69.60.0
53.20.0
25.40.1
4.78
Similarity (+)
75.50.0
66.90.2
53.60.3
50.80.1
58.30.3
39.00.1
69.90.1
52.10.1
26.90.1
4.38
Random
75.50.1
68.20.3
55.40.3
51.70.2
49.40.5
38.30.1
70.40.1
52.00.1
27.70.1
4.25
One-shot (+)
76.20.0
58.10.1
55.40.3
50.00.0
61.70.0
38.10.1
68.90.1
53.00.1
30.50.1
3.97
Best set
75.30.0
69.10.1
57.50.3
50.70.1
50.90.3
38.30.1
70.90.1
52.60.0
27.90.0
3.89
IC Datamodels (+)
75.60.0
68.50.1
59.60.1
55.00.2
53.20.3
37.60.0
71.20.0
53.70.1
30.80.1
3.02
Influence (+)
75.90.0
67.70.2
62.70.1
53.20.2
52.90.3
38.10.1
70.60.1
53.70.1
31.30.1
2.86
Similarity (+)
75.90.1
71.20.2
52.90.2
51.00.2
58.60.1
37.40.1
73.10.1
54.20.0
28.90.1
4.30
Random
76.10.0
69.50.2
51.20.1
53.60.3
54.80.3
37.60.1
73.20.0
53.60.1
30.00.1
4.27
One-shot (+)
75.80.0
69.00.1
57.30.2
50.00.0
61.70.0
39.60.0
72.40.0
53.30.1
32.00.0
4.17
Perplexity (+)
76.20.1
71.80.1
55.90.2
53.10.3
42.40.2
38.00.0
73.10.0
54.10.1
28.00.1
4.14
Best set
75.80.0
72.80.1
53.30.3
54.50.3
50.30.4
37.80.1
73.10.0
53.30.0
31.40.1
3.87
IC Datamodels (+)
75.90.0
72.00.1
65.10.2
56.30.1
48.10.4
37.20.0
72.60.0
54.30.0
34.40.1
3.38
Influence (+)
75.80.0
71.90.1
61.70.2
55.70.1
57.10.2
36.80.1
73.80.0
54.40.0
34.00.1
2.89
OPT-30B
Perplexity (+)
76.80.0
72.70.2
61.90.3
53.50.2
43.50.6
40.30.1
76.30.1
56.60.0
28.50.1
5.16
Random
77.00.0
71.10.2
63.20.2
54.80.1
49.10.5
41.50.1
76.00.1
55.40.1
29.60.1
4.75
Best set
76.90.0
72.60.0
64.10.3
55.10.2
54.80.4
40.80.0
75.80.1
56.10.0
31.50.0
4.30
One-shot (+)
77.50.0
76.50.1
52.40.1
51.10.2
61.60.0
41.50.0
76.10.1
56.60.1
31.20.0
3.92
Similarity (+)
77.70.1
70.10.4
63.90.1
53.30.1
57.10.7
42.10.1
76.20.1
56.70.0
29.30.0
3.84
Influence (+)
78.00.0
74.10.1
64.60.1
52.50.1
51.40.3
41.60.1
77.00.0
57.40.0
33.30.0
2.89
IC Datamodels (+)
78.10.0
77.00.0
65.90.1
51.40.2
56.40.1
42.10.0
76.60.0
58.20.0
31.70.1
2.41
Similarity (+)
75.90.1
71.20.2
52.90.2
51.00.2
58.60.1
37.40.1
73.10.1
54.20.0
28.90.1
4.30
Random
76.10.0
69.50.2
51.20.1
53.60.3
54.80.3
37.60.1
73.20.0
53.60.1
30.00.1
4.27
One-shot (+)
75.80.0
69.00.1
57.30.2
50.00.0
61.70.0
39.60.0
72.40.0
53.30.1
32.00.0
4.17
Perplexity (+)
76.20.1
71.80.1
55.90.2
53.10.3
42.40.2
38.00.0
73.10.0
54.10.1
28.00.1
4.14
Best set
75.80.0
72.80.1
53.30.3
54.50.3
50.30.4
37.80.1
73.10.0
53.30.0
31.40.1
3.87
IC Datamodels (+)
75.90.0
72.00.1
65.10.2
56.30.1
48.10.4
37.20.0
72.60.0
54.30.0
34.40.1
3.38
Influence (+)
75.80.0
71.90.1
61.70.2
55.70.1
57.10.2
36.80.1
73.80.0
54.40.0
34.00.1
2.89
OPT-30B
Perplexity (+)
76.80.0
72.70.2
61.90.3
53.50.2
43.50.6
40.30.1
76.30.1
56.60.0
28.50.1
5.16
Random
77.00.0
71.10.2
63.20.2
54.80.1
49.10.5
41.50.1
76.00.1
55.40.1
29.60.1
4.75
Best set
76.90.0
72.60.0
64.10.3
55.10.2
54.80.4
40.80.0
75.80.1
56.10.0
31.50.0
4.30
One-shot (+)
77.50.0
76.50.1
52.40.1
51.10.2
61.60.0
41.50.0
76.10.1
56.60.1
31.20.0
3.92
Similarity (+)
77.70.1
70.10.4
63.90.1
53.30.1
57.10.7
42.10.1
76.20.1
56.70.0
29.30.0
3.84
Influence (+)
78.00.0
74.10.1
64.60.1
52.50.1
51.40.3
41.60.1
77.00.0
57.40.0
33.30.0
2.89
IC Datamodels (+)
78.10.0
77.00.0
65.90.1
51.40.2
56.40.1
42.10.0
76.60.0
58.20.0
31.70.1
2.41
PIQA
BoolQ
RTE
WIC
WSC
ARC-c
ARC-e
HS
OBQA
Rank (↓)
GPT-J-6B
Similarity (-)
76.00.0
63.20.2
53.50.2
55.30.2
48.90.5
38.20.0
73.50.0
49.60.0
27.20.0
5.51
Random
75.70.1
61.70.3
56.20.3
51.90.3
48.50.4
37.70.1
73.30.0
49.10.1
27.60.1
4.98
Worst set
76.20.0
60.20.1
52.60.1
52.10.2
48.70.3
37.70.0
71.80.1
48.50.1
27.10.0
4.22
IC Datamodels (-)
76.30.0
58.80.2
50.40.0
50.20.1
53.00.2
38.30.1
71.60.0
47.10.0
24.90.0
3.43
Perplexity (-)
73.10.1
60.10.1
52.80.1
51.40.1
43.80.3
37.10.1
72.40.1
46.30.0
27.30.0
3.32
Influence (-)
75.60.0
57.40.2
50.60.0
48.40.1
47.90.4
38.50.0
71.10.0
47.40.0
26.20.1
2.98
One-shot (-)
76.40.0
58.70.2
50.00.0
50.00.0
38.30.0
37.40.1
72.80.1
46.50.1
25.30.1
2.68
GPT-NeoX-20B
Similarity (-)
76.80.1
62.50.2
63.50.1
52.40.2
44.30.4
43.70.1
77.50.1
55.30.0
32.50.0
5.08
Random
77.30.0
67.00.4
62.40.3
51.70.2
43.50.6
43.90.1
77.80.1
55.10.1
30.30.0
4.94
Perplexity (-)
76.30.0
72.40.1
61.20.3
53.50.1
46.00.6
44.30.1
77.40.1
52.60.0
30.70.1
4.57
Worst set
76.40.0
66.90.2
61.50.4
51.50.3
45.50.3
43.10.1
77.10.1
54.80.0
30.50.1
4.29
One-shot (-)
77.40.0
50.90.0
62.20.2
50.00.0
60.90.0
42.40.1
76.50.0
52.80.0
28.10.1
3.13
Influence (-)
76.70.1
53.90.1
58.10.3
50.80.1
38.10.2
41.80.1
76.60.0
53.90.0
29.20.0
2.68
IC Datamodels (-)
77.10.1
54.30.1
57.70.3
49.40.1
40.60.2
41.40.1
76.90.0
53.10.0
29.10.1
2.52
LLaMA-7B
Random
78.00.1
80.30.1
63.40.2
51.10.2
41.70.3
44.30.0
78.70.1
58.80.1
32.30.0
5.16
Similarity (-)
77.80.0
79.20.1
59.10.1
51.60.2
42.60.3
44.70.1
78.40.0
58.60.1
31.90.1
4.75
Worst set
78.30.0
77.60.1
58.10.2
52.70.1
41.90.4
44.00.0
79.00.0
58.80.0
30.10.1
4.65
One-shot (-)
78.40.0
78.70.1
69.10.1
51.50.1
61.80.0
42.30.1
74.70.1
57.70.0
32.00.1
4.40
Perplexity (-)
75.50.0
81.30.1
57.90.2
50.70.1
41.10.5
43.00.1
77.70.0
56.70.0
28.70.1
3.13
IC Datamodels (-)
78.20.0
73.50.3
53.50.1
50.60.1
45.60.8
43.60.1
76.90.0
57.30.0
27.90.1
2.94
Influence (-)
78.10.0
71.90.1
52.70.1
49.30.1
40.40.4
43.10.1
76.50.1
57.20.0
27.20.1
2.16
LLaMA-13B
Similarity (-)
79.20.0
83.20.0
58.70.1
54.60.2
43.90.5
51.10.0
82.30.0
62.10.0
37.10.0
5.51
Random
78.50.1
82.60.1
61.10.3
51.80.2
42.90.4
50.40.1
82.70.0
62.50.1
35.70.1
4.84
Worst set
78.80.0
79.20.1
54.10.2
53.30.1
45.70.6
50.30.1
83.00.0
62.10.1
33.60.1
4.60
Perplexity (-)
74.90.0
82.40.1
57.90.1
55.40.2
42.80.4
49.40.0
81.40.0
58.70.0
33.10.1
3.56
One-shot (-)
78.70.0
68.20.2
53.90.1
53.10.1
55.40.7
50.00.1
81.40.0
61.00.0
26.10.1
3.22
IC Datamodels (-)
78.50.0
69.30.3
50.00.0
51.60.2
38.90.1
50.00.1
82.80.1
61.80.0
22.00.1
2.84
Influence (-)
78.60.0
68.30.3
50.00.0
50.60.2
39.80.3
49.30.1
82.40.1
61.60.0
22.90.1
2.43
OPT-6.7B
Similarity (-)
75.60.0
65.40.2
56.70.2
52.70.0
50.80.2
38.00.1
70.90.1
53.00.0
26.90.0
4.94
Random
75.50.1
68.20.3
55.40.3
51.70.2
49.40.5
38.30.1
70.40.1
52.00.1
27.70.1
4.70
Worst set
76.10.0
66.40.1
53.00.3
52.50.1
55.30.3
38.40.1
69.60.1
51.40.0
26.80.0
4.44
Perplexity (-)
75.10.0
70.70.1
51.90.2
50.70.0
59.60.2
37.10.1
69.50.0
47.80.0
27.50.0
3.81
Influence (-)
76.30.0
61.90.3
50.80.1
50.50.1
51.90.7
37.20.1
70.30.1
51.40.1
25.10.0
3.38
One-shot (-)
76.00.0
65.10.2
50.00.0
50.00.0
38.30.0
37.60.1
71.20.0
50.60.1
26.80.1
3.13
IC Datamodels (-)
75.70.1
66.30.1
50.80.1
48.10.2
47.80.2
36.90.1
69.80.0
51.20.0
23.60.1
2.65
OPT-13B
Random
76.10.0
69.50.2
51.20.1
53.60.3
54.80.3
37.60.1
73.20.0
53.60.1
30.00.1
5.10
Similarity (-)
76.00.1
68.50.1
50.60.1
55.80.1
52.70.3
38.60.0
73.60.0
53.00.1
29.50.1
4.75
Perplexity (-)
73.70.1
71.60.1
50.30.0
50.00.0
52.00.5
38.70.1
72.50.1
51.60.0
30.40.0
4.00
IC Datamodels (-)
77.00.0
69.10.2
50.30.0
50.90.1
50.70.3
36.70.1
72.60.1
53.10.0
28.80.1
3.84
Worst set
76.10.0
67.20.2
50.40.0
50.70.1
48.80.4
37.40.1
72.90.0
53.10.1
28.60.0
3.83
Influence (-)
76.60.0
69.40.1
50.40.1
49.20.1
45.40.3
36.50.1
72.50.1
52.70.0
28.20.1
3.05
One-shot (-)
76.20.1
52.10.0
50.00.0
50.00.0
38.30.0
35.70.1
72.50.1
50.20.0
29.00.1
2.14
OPT-30B
Similarity (-)
77.70.1
67.30.2
64.20.2
54.60.1
49.30.4
42.30.1
76.50.1
56.30.0
30.30.0
5.79
Random
77.00.0
71.10.2
63.20.2
54.80.1
49.10.5
41.50.1
76.00.1
55.40.1
29.60.1
4.87
Worst set
77.60.0
66.50.4
64.30.2
54.90.1
46.50.3
40.90.0
75.10.1
55.50.0
29.80.0
4.59
Influence (-)
77.60.0
61.00.4
59.10.2
51.70.1
43.90.3
41.30.1
76.10.0
54.40.0
27.80.1
3.59
Perplexity (-)
75.10.0
73.70.1
52.00.2
51.60.1
46.60.5
40.90.1
74.50.1
53.50.0
30.60.0
3.48
IC Datamodels (-)
77.40.1
59.10.1
60.10.3
51.10.1
42.80.2
40.40.1
75.40.1
55.20.0
26.90.1
2.98
One-shot (-)
77.70.0
51.70.0
50.00.0
50.00.0
38.30.0
40.30.0
75.80.1
51.70.0
28.20.1
2.02
Task
Template
PIQA
Goal: {goal}
Answer: {answer}
BoolQ
{passage}
question: {question}?
answer: {answer}
RTE
{premise}
question: {hypothesis}. true or false?
answer: {answer}
WIC
{sentence1}
{sentence2}
question: Is the word ‘{word}’ used in the same sense in the two sentences
above?
answer: {answer}
WSC
Passage: {text}
Question: In the passage above, does the pronoun ‘{span2}’ refer to
{span1}?
Answer: {answer}
Arc (Chal.)
Question: {question}
Answer: {answer}
Arc (Easy)
Question: {question}
Answer: {answer}
Hellaswag
Context: {context}
Answer: {answer}
OBQA
Context: {context}
Answer: {answer}
Task
ID
Influence
Prompt
PIQA
2305
0.004877
Goal: sand paper
Answer: can be used to smooth wood for furniture
RTE
1439
0.016923
As a real native Detroiter, I want to remind everyone that Madonna is from
Bay City, Mich., a nice place in the thumb of the state’s lower peninsula.
question: Madonna was born in Bay City, Mich.. true or false?
answer: true
WIC
2033
0.01273
Efface the memory of the time in the camps.
Efface oneself.
question: Is the word ‘efface’ used in the same sense in the two sentences
above?
answer: false
WSC
98
0.031323
Passage: The man lifted the boy onto his bunk bed.
Question: In the passage above, does the pronoun ‘his’ refer to The man?
Answer: false
Arc (Chal.)
684
0.004829
Question: Which energy resource is non-renewable?
Answer: oil
Arc (Easy)
859
0.003275
Question: Which processes change magma into igneous rock?
Answer: cooling and crystallization
Hellaswag
30980
0.00546
Context: Education and Communications: [header] How to calculate
consumer surplus [title] Understand the law of demand. [step] Most people
have heard the phrase “supply and demand” used in reference to the
mysterious forces governing market economies, but many don’t
understand these concepts’ full implications. “demand” refers to the desire
for a good or service in the marketplace.
Answer: Generally, if all other factors are equal, demand for a product will
fall as its price increases. [substeps] For example, let’s say that a company is
about to release a new model of television.
OBQA
3640
0.006248
Context: Scavengers eat dead what?
Answer: fauna
Question: Which processes change magma into igneous rock? Answer: cooling and crystallization
Task
ID
Influence
Prompt
PIQA
10777
-0.001315
Goal: baby wipe
Answer: Can be pierced by a fork Using the tines
RTE
2391
-0.022086
Since the fear of death is virtually a universal phenomenon, the death
penalty is an unparalleled deterrent for people considering a crime.
question: Capital punishment is a deterrent to crime.. true or false?
answer: true
WIC
4233
-0.007281
After the fire a still small voice. – 1 Kings 19:12.
Conservatism has many voices.
question: Is the word ‘voice’ used in the same sense in the two sentences
above?
answer: false
WSC
334
-0.007789
Passage: Sara borrowed the book from the library because she needs it for
an article she is working on. She reads it when she gets home from work.
Question: In the passage above, does the pronoun ‘it’ refer to the book?
Answer: true
Arc (Chal.)
596
-0.006895
Question: A research scientist repeatedly observes a bird avoiding a specific
butterfly species even though it eats other types of butterflies. Which
statement most likely explains the behavior of the bird?
Answer: The behavior is learned over the lifetime of the bird.
Arc (Easy)
1940
-0.002372
Question: The organisms that convert solar energy and raw materials into
food are
Answer: producers.
Hellaswag
8891
-0.005678
Context: Surfing: People are surfing on a large wave in the water. A boat is
in the water. a large wave
Answer: crashes in the water.
OBQA
978
-0.004077
Context: Do objects change size with distance for Stevie Wonder?
Answer: No
