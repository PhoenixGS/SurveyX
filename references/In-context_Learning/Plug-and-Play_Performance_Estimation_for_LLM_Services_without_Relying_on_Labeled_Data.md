# Plug-and-Play Performance Estimation for LLM Services without Relying on Labeled Data
Can Wang , Dianbo Sui, Hongliang Sun, Hao Ding, Bolin Zhang(�), and Zhiying Tu(�)
Harbin Institute of Technology, Harbin, Heilongjiang, China {23B903072}@stu.hit.edu.cn {suidianbo}@hit.edu.cn {21B903094, dinghao}@stu.hit.edu.cn {brolin, tzy_hit}@hit.edu.cn
Abstract. Large Language Model (LLM) services exhibit impressive capability on unlearned tasks leveraging only a few examples by in-context learning (ICL). However, the success of ICL varies depending on the task and context, leading to heterogeneous service quality. Directly estimating the performance of LLM services at each invocation can be laborious, especially requiring abundant labeled data or internal information within the LLM. This paper introduces a novel method to estimate the performance of LLM services across different tasks and contexts, which can be "plug-and-play" utilizing only a few unlabeled samples like ICL. Our findings suggest that the negative log-likelihood and perplexity derived from LLM service invocation can function as effective and significant features. Based on these features, we utilize four distinct meta-models to estimate the performance of LLM services. Our proposed method is compared against unlabeled estimation baselines across multiple LLM services and tasks. And it is experimentally applied to two scenarios, demonstrating its effectiveness in the selection and further optimization of LLM services.
Keywords: Generative AI as a Service · Large Language Model · Performance Estimation · Service Selection · Optimization Tuning.
# 1 Introduction
Large language models (LLM) have the capability to understand and generate natural language text, making them valuable tools for a variety of natural language processing tasks such as text generation, translation [23], summarization [29], question answering [9], and more. LLM services, such as OpenAI LLM API 1, allow users to conveniently solve their tasks by interacting with LLM in a flexible conversational manner, without needing to know whether the LLM has been trained on these tasks or not. This remarkable capability is realized through the paradigm of In-Context Learning (ICL) [3], which enables the LLM to generalize rapidly only employing
a few labeled examples without requiring additional training. However, such a paradigm is not flawless. Many studies have revealed such a reality: ICL is highly sensitive to task and context [22,32]. ICL can demonstrate significant advantages in certain tasks when using appropriate LLM services and contexts, such as solving entity linking tasks with the Phi-2 service. But it can be virtually ineffective in other scenarios, like solving web-question tasks with the Llama-7B service2. Therefore, in the face of different tasks and contexts, it is both challenging and necessary to estimate the performance of LLM services in advance. To estimate the LLM services’ performance, typical solutions use labeled data to invoke LLM services [17, 27], which necessitate collecting labels and testing them for each task. However, generally purposed LLM services address a wide range of natural language tasks and most of these tasks are not human-labeled. Especially in domain-specific tasks that require expertise, such as medical or law text understanding, the high cost of annotation poses a significant challenge. Another solutions avoiding labeled data by exploiting the information within the LLM during inference, potentially requiring the LLM’s architecture and parameters to be open [12,16,18,20]. These approaches have limitations for LLM services that only provide usage access without disclosing internal information, and in practice, extracting internal information on a large number of heterogeneous LLM services is also a time-consuming and laborious work. Building on this, we explore a more practical and appealing idea to estimate the performance of LLM services, which can be "plug-and-play" for various LLM services and unlabeled tasks in different contexts. In detail, we explore the common relationship between the semantic features exhibited during the invocation of LLM services and performance. We find two useful features, negative loglikelihood and perplexity, which rely solely on the answers generated during LLM service invocation, but can reflect the performance potential of the LLM service on the current task and context. Then, we propose our meta-model based approach tightly integrated with the ICL paradigm: for a quick and reliable performance estimation, only the answers of the LLM service to a few examples are needed. Our novel training and inference approach using linear interpolation makes the meta-model effective and generalizable, which can be used for a wide range of different LLM services and unlabeled tasks without retraining. The contributions of this paper are:
– We explore the common phenomenon exhibited during the invocation of LLM services, and select the available features based on their relevance. – We propose a method for LLM service performance estimation that is able to reach low-error estimates at little cost on various unlabeled tasks, which can be applied to most LLM services that do not know the internal information. – We verify the effectiveness of our method in two scenarios: the selection of LLM services and the further optimization for few-shot tasks of LLM services, proving that it can be helpful in various future works. 2 We conducted experiments to prove this. For more detailed information about this paper, including the dataset, hyperparameter settings, etc., please see: https:// github.com/WangCan1178/Plug-and-Play-Estimation
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ba04/ba04b139-b231-4012-a54f-ea149723f0a4.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 1: Distribution of the four features and the LLM service performance, as well as the fitting curve (from two randomly selected task invocation results).</div>
Fig. 1: Distribution of the four features and the LLM service performance, a well as the fitting curve (from two randomly selected task invocation results).
# 2 Pilot Experiments
This section describes the phenomenon we observe when LLM services are invoked to perform ICL. Specifically, we mainly explore the following two research questions: RQ1: What features can be extracted on unlabeled task when invoking LLM services through ICL paradigm? RQ2: How to select appropriate features to reflect the performance of LLM services?
# 2.1 Experiment Setup
In the pilot experiments, we select the top 5 generative LLM services according to the downloads of hugging face model library (https://huggingface.co/ models), which have different sizes and structures. Besides, we choose a representative benchmark dataset: CrossFit [30], a benchmark to study the generalization capability of LLMs, containing 160 different few-shot NLP tasks. Given that our method is based on ICL and few-shot tasks, we don’t use datasets for common NLP tasks that LLM services may have seen the data during training. We sample examples from the training dataset of each task, constitute the context of the unlabeled data, and invoke the LLM service on the testing dataset. And F1-score [26] is used to calculate the accuracy of the generation on unlearned tasks, reflecting the performance of LLM service when invoked.
To answer the RQ1, we survey LLM services on the market and find that almost all of them provide (top-few) word-list probabilities of the reasoned answer. We use this probability to come up with the following usable features. A intuitive illustration of their strong correlation can be observed in Figure 1, by fitting the distribution between these features and performance on these tasks.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/cd55/cd559d0f-253e-4923-a870-6829dc1db24d.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 2: Pearson correlation coefficient of features and LLM services performance.</div>
Negative log-likelihood (NLL). We treat the process of the LLM service invocation as a generation task. NLL can be used for measuring how well LLM fit on the dataset, which can be obtained from each generated sequence as following:
where x is the output sequence of the LLM service with parameter θ, and P(xt | x<t) is the maximum probability assigned at t-th token. The smaller the value, the more confident about the generated sentence. Perplexity (PPL). Perplexity reflects the likelihood of a LLM having seen and learned (in other word, be pretrained on) this data before. It is calculated based on the probability that the LLM reconstructs the input sequence.
Its calculation is similar to NLL, except the predicated token xt is replaced by the input token ˜xt in the conditional probability. The smaller the value, the more likely it is that the LLM has seen and learned the generated sentence. GAP. GAP is defined as the difference between the probability of the most likely token (i.e., the first ranked token) and the probability of the second most likely token (i.e., the second ranked token) in the probability distribution generated for the current word.
GAP takes into account the effect of potentially possible answers. And the bigger the value, the more accurate about the generated sentence. Maximum Entropy (MaxEnt). Entropy is an indicator to measure the uncertainty about the generated tokens. Preliminary experiments [10] show that simply taking the maximum token entropy significantly outperforms other aggregation methods such as averaging or taking the minimum.
(1)
(2)
(3)
(4)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0f5b/0f5bc42f-b442-42f3-b852-8ff869d926eb.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 3: Procedure of our meta-model based LLM service performance estimation.</div>
where H(xt | x<t; θ) is the entropy of the current token calculated from its probability. The smaller the value, the more certain about the generated sentence.
# 2.3 How to Select Appropriate Features to Reflect the Performance of LLM Services? (RQ2)
Based on the proposed features in Section 2.2, we can compute the correlation between these features and the LLM services performance, in units of tasks. Pearson correlation coefficient is used as the indicator of measurement. And F1score is used to calculate the accuracy of the generation on unlearned tasks, reflecting the performance of LLM service. Figure 2 shows the overall correlations by simply taking the average over all results. These features can constitute different combinations that reflect the LLM service performance. According to the theory of correlation and collinearity [1,14], we expect to select combinations that are strongly correlated with the performance, but not strongly correlated with other features. Thus, we define a score to this end as followed.
where F = {f1, f2, ...fn} is a combination of different features, and corr(fi, fj) denotes the correlation between the feature fi and fj, without repeating calculating corr(fj, fi). Through this score, we find the best combination of features is F = {NLL, PPL}, denoting these two features can reflect the performance of LLM service best from two different aspects.
# Methodology: LLM Services Performance Estim
In this section, we first illustrate the definition of LLM services performance estimation problem in subsection 3.1. Then, the meta-model based method we proposed is introduced in subsection 3.2, detailing its novel training and inference process in subsection 3.3. The whole procedure of the proposed meta-model based method is shown as shown in Figure 3.
(5)
# 3.1 Problem Definition
In this paper, we focus on estimating performance of LLM services on unlearned and unlabeled tasks. Our goal can be formalized as investigating how to quickly and cheaply estimate the performance � perS T,C of ICL, given a LLM service S, an unlearned and unlabeled task T = {x(1), . . . , x(n)}, with the context C. Absolute error is used to measure the effectiveness of our method, defined as |perS T,C −� perS T,C|, where perS T,C is the true performance of the LLM service S invoked to handle the same task T with C. We explore performance estimation on a broad LLM service market {Si}I i=1, a wide range of natural language tasks {Tj}J j=1 and different contexts {Ck}K k=1. Therefore, the final mean abstract error (MAE) is calculated by the following method, using the average absolute error:
� In different subsequent works, several of the terms in the Equation 6 can be fixed to simplify the calculation. Such as selecting the most appropriate LLM service for a specific fixed task with a given context, it only needs to minimize the average error of all LLM services.
# 3.2 Meta-Model Based Method
Meta-model 3 refers to a high-level model that does not predict the data directly, but makes the final prediction or decision by combining and analyzing the prediction results of other base models. In our proposed method, the meta-model is able to accept the features from generated answers when invoking the LLM service, and to estimate the performance when inference. For efficiency, we choose meta-models with simple structures containing much fewer parameters than LLM, which are easy to train and have fast inference speed. Four meta-models with different architectures are selected: k-Nearest Neighbors (k-NN), which estimates the performance of the LLM service by measuring the similarity of features between samples [8]. Multilayer Perceptron (MLP), which estimates the performance of LLM service by the probability propagation between neurons [24]. RandomForest, which estimates the performance of LLM service by bagging multiple weak Learners [2]. eXtreme Gradient Boosting (XBoost), which estimates the performance of LLM service by boosting multiple weak Learners [6].
# 3.3 Training and Inference
The goal of the training phase is to obtain a good meta-model M, which mus possess sufficient generalization capability to be applied across a wide rang of LLM services and open-domain tasks. It is satisfied through three require inputs: a set of LLM services {Si}I i=1, a set of labeled tasks {T labeled j }J j=1, an 3 Extend from metamodeling( https://wikipedia.org/wiki/Metamodeling)
The goal of the training phase is to obtain a good meta-model M, which must possess sufficient generalization capability to be applied across a wide range of LLM services and open-domain tasks. It is satisfied through three required inputs: a set of LLM services {Si}I i=1, a set of labeled tasks {T labeled j }J j=1, and
3 Extend from metamodeling( https://wikipedia.org/wiki/Metamodeling)
3 Extend from metamodeling( https://wikipedia.org/wiki/Metamodeling)
a set of contexts {Ck}K k=1 sampled from the respective task. These three inputs can be arbitrarily combined to obtain the results of invocations of different LLM services under various tasks and contexts. As mentioned in Section 2, we can extract the useful meta-model features nllSi T labeled j ,Ck and pplSi T labeled j ,Ck of the series of invocations, as well as the true performance perSi T labeled j ,Ck demonstrated. A key problem is that different scale sizes of tasks result in different dimensions of features. We borrow the idea of profile [11], and map the features to the same dimension d by linear interpolation. It is defined as follows, where D is the dataset size of the current task T labeled j , and fn is the n-th index of the features such as nllSi T labeled j ,Ck or pplSi T labeled j ,Ck.
In this way, by continuously reducing the difference between the estimated performance � perSi T labeled j ,Ck and the true performance perSi T labeled j ,Ck, the metamodel gradually converges to the point where it can estimate the LLM service performance on different unlearned tasks. In the inference phase, a partial subset of the unlabeled task T ∗unlabeled ⊆ T unlabeled is selected, and the estimated LLM service S is invoked with the context C for inference. Using the same method as in the training phase, the features nllS T ∗unlabeled,C and pplS T ∗unlabeled,C and are obtained and mapped to a d dimensional space. Then, the trained meta-model M is applied and the estimated performance of LLM service is obtained as defined.
� In summary, our trained meta-model achieves estimation on a wide range of unlabeled tasks and contexts, by exploring the relationship between the features exhibited during LLM service invocation and its performance.
# 4 Experiments
In this Section, we first introduce the baseline methods, including methods using both labeled and unlabeled data. Then in subsection 4.2, the details of our experiments are presented. The main results are given in subsection 4.3, demonstrating the effectiveness and practicality of our approach. Finally, we conduct an ablation study to show the impact of features and the number of unlabeled samples used to make estimations.
# 4.1 Baselines
We design the following baselines that do not involve LLM internal information to compare with our method, which allows LLMs to be used as black-box services. Sample accuracy of labeled examples (Samplen). It is straightforward to estimate the performance of LLM services exhibited in different tasks by
(7)
(8)
labeling the few data [27]. This method samples n examples from the dataset of the task T unlabeled to label, and calculate the accuracy of these n examples as the performance of the whole task, which we call it Samplen. According to the law of large numbers, the more sample examples are labeled, the closer the estimated accuracy is to the true performance, at the cost of more expensive labeling costs. When estimating LLM service performance, we want our method to be able to approximate the accuracy of Samplen without using labeled data.
� where perS x(i),C is the true performance of the LLM service S invoked on the labeled sample x(i) sampled from the unlabeled task T dataset. Average accuracy on the training dataset (AvgTrain). Similarly, the average over all labeled tasks can also be used as a baseline for LLM service.
# � perS T,C(AvgTrain) = 1 J 1 K � j∈J � k∈K perS Tj,Ck
� Average threshold of confidence (ATC). Another practical baseline approach is to obtain a threshold based on the confidence of LLM service on tasklevel [12], whereby the accuracy is predicted by the proportion of unlabeled instances where model confidence surpasses the threshold. And the average of the ATC obtained from the estimates of each seen task is used as a baseline.
� where atcS Tj,Ck is the accuracy estimation of task Tj using the threshold o model confidence.
# 4.2 Experimental Details.
We use the dataset mentioned in Section 2 because it provides enough few-shot NLP tasks to facilitate the study of performance of LLM services across tasks. Thirteen open-ended generation tasks are selected for our experiments. We combine the "train" and "dev" set to train our meta-model as labeled tasks. For each task, three examples are sampled from it at a time as context. And we conducted a total of 50 different sampling times to fully investigate the impact of context on the invocation of different LLM services. In total, we have experimented with executing 5 LLM services on 13 tasks with 50 contexts (3250 ICL settings). We use 1000 unlabeled samples and 5-fold cross-validation to verify our method’s effectiveness. The number of features d is set to 100, and the optimal hyperparameters are selected by grid search for each meta-model.
<div style="text-align: center;">Table 1: Experimental results (MAE) and variations (SD) for different LLM services performance estimation on our method and baselines.</div>
Table 1: Experimental results (MAE) and variations (SD) for different LL services performance estimation on our method and baselines.
LLM service
Llama-7B
Mistral-7B
OPT-6.7B
Phi-2
Zephyr-7B-β
Total
Baselines
Sample8
7.98 ± 4.10
5.27 ± 1.34
10.23 ± 3.09 10.20 ± 3.79
4.74 ± 2.45
8.28 ± 4.25
Sample16
6.78 ± 2.82
4.12 ± 1.59
8.70 ± 2.75
9.22 ± 2.61
3.98 ± 1.49
6.12 ± 3.60
Sample32
4.15 ± 1.54
3.12 ± 0.94
6.24 ± 2.38
6.78 ± 2.41
3.10 ± 1.36
5.14 ± 2.84
AvgTrain
6.20 ± 2.70
5.74 ± 4.64
8.96 ± 3.71
8.60 ± 2.23
5.80 ± 2.48
6.74 ± 4.62
ATC
40.91 ± 10.24 38.82 ± 5.36 30.56 ± 11.42 31.10 ± 9.98
39.20 ± 9.02
39.82 ± 5.37
Meta Models
3-NN
6.50 ± 2.50
7.30 ± 2.69
7.02 ± 2.63
7.42 ± 3.61
6.18 ± 0.23
7.30 ± 2.96
MLP
7.24 ± 5.60
5.30 ± 0.71
7.24 ± 0.82
6.58 ± 1.31
5.50 ± 1.44
5.30 ± 1.70
RandomForest
5.80 ± 0.99
4.02 ± 1.61
8.60 ± 1.60
5.38 ± 1.46 4.04 ± 1.34 4.72 ± 1.61
XGBoost
4.76 ± 0.63
5.42 ± 1.59
6.17 ± 0.81
5.44 ± 1.60
4.56 ± 1.17
5.42 ± 2.59
In this subsection, we first compare the error of our proposed method and baselines. Then we demonstrate the effectiveness of our method in terms of the accuracy improvement of two subsequent works. LLM services performance estimation performs all unlabeled baselines. Table 1 shows the MAE obtained by performance estimation on different LLM services. On all LLM services, our method performs better than the unlabeled baselines on average across the 13 tasks. And in the best case (XGBoost with OPT-6.7B), the meta-model’s MAE is 31.1% lower than the best baseline method without labels. Compared with method Samplen using labeled data, our method can mostly outperform the method with 16 samples, and sometimes even outperform the method with 32 samples. In particular, when not differentiated by LLM service, our model performs at a comparable level to sampling 64 samples(4.78 ± 2.10), which shows the high generalization of our method. The implication is that we can significantly save annotation costs to estimate the performance of different LLM services on a wide range of natural language tasks. Furthermore, note that the standard deviation of the best meta-model on the LLM performance estimation using our method is significantly smaller than that of the other method. It indicates the stability of our method, that is, it is almost unaffected by the sampled different unlabeled samples. We also explore the performance of LLM service performance estimation on different tasks, and the results are presented in Figure 4. It shows that in all 13 few-shot tasks, our best meta-model outperforms all baseline methods, including the previous best performing method SAMPLE32. This may be due to the fact that the probability distribution of the results for the same task is similar, which leads to faster convergence and better performance of our method. It is more practical and attractive than estimating the LLM service performance on multiple tasks, because the performance estimation on a certain given task is more in line with the actual demands.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d3db/d3dbc0ed-0bcd-49dc-b423-fc79737c4af3.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 4: Experimental results (MAE) for different tasks of the LLM services performance estimation (our method) and baselines.</div>
Fig. 4: Experimental results (MAE) for different tasks of the LLM services pe formance estimation (our method) and baselines.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7ea0/7ea00b4d-592a-4dcd-b36b-85c69037deb5.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 5: Execution performance under the settings of our method and randomly selected services or contexts.</div>
Fig. 5: Execution performance under the settings of our method and random selected services or contexts.
LLM services performance estimation helps the subsequent works. To verify the practicality of our method, we conduct experiments of two common scenarios, applying LLM services performance estimation to subsequent works. The first scenario considers the selection of services and contexts when the user invokes the LLM service to perform some unlearned task. We experiment with the best hyperparameters of the best meta-model architecture (RandomForest). Five LLM services and ten different contexts are randomly sampled for a total of 50 ICL settings. The estimation performance on different settings are given through the inference of the meta-model, which the best one are selected as the recommended service and context of our method. It is compared with the random selection of services and contexts that often happens in the actual invocation scenario, and the results shown in Figure 5 are obtained. In 13 different few-shot tasks, the selected services and contexts using our proposed LLM service performance estimation indeed exhibit stronger ICL capability. This is undoubtedly appealing, as users can improve the performance of the current task by even up to 21% at no additional annotation cost. Another scenario considers domain tasks that are difficult for all LLM services, such as lama-conceptnet (concept question with answer is a single word), which performs best with only an F1-score of 0.12. At this time, further optimization of the LLM service is necessary, and a common approach is to fine-tune the LLM’s parameters, which is often laborious and resource-consuming. Our approach can help indicate which LLM service has a wider optimization space, and to make a better choice in advance.
Table 2: Fine-tuning effects on the low-performing tasks of LLM services selected by our method and all LLM services.
Task
kilt_zsre
lama-conceptnet lama-google_re
Llama-7B
0.09 - 0.02
0.06 + 0.09
0.00 + 0.08
Mistral-7B 0.12 + 0.13
0.12 + 0.05
0.01 + 0.10
OPT-6.7B
0.04 + 0.05
0.09 + 0.02
0.01 + 0.02
Phi-2
0.03 + 0.06
0.22 + 0.09
0.00 - 0.01
Zephyr-7B-β 0.12 + 0.07
0.05 + 0.03
0.02 + 0.11
Table 3: Evaluation results (MAE) and variants (SD) of different features selected to use in our method.
Table 3: Evaluation results (MAE) and variants (SD) of different features selecte to use in our method.
LLM service Llama-7B
Mistral-7B OPT-6.7B
Phi-2
Zephyr-7B-β
Total
NLL only
3-NN
8.22 ± 3.50
8.15 ± 3.29
7.02 ± 2.63 10.67 ± 4.64
8.82 ± 2.23
9.10 ± 4.37
MLP
9.23 ± 5.51
7.03 ± 4.25
8.51 ± 2.46
7.70 ± 3.27
7.47 ± 3.40
6.52 ± 3.86
RandomForest 7.85 ± 2.24 6.11 ± 3.57 9.00 ± 3.97
7.42 ± 2.34
6.83 ± 3.57 6.90 ± 3.13
XGBoost
6.61 ± 1.90 7.94 ± 2.55 6.17 ± 3.24 7.32 ± 2.34
6.97 ± 3.76
7.17 ± 4.05
PPL only
3-NN
14.50 ± 4.78 9.23 ± 4.31 12.78 ± 3.79 11.45 ± 5.82
9.14 ± 2.56
11.68 ± 4.56
MLP
12.67 ± 6.34 11.95 ± 5.13 9.72 ± 2.89 14.79 ± 5.14
10.56 ± 4.60
9.19 ± 4.02
RandomForest 10.97 ± 5.67 9.85 ± 4.98 11.14 ± 5.42 10.06 ± 2.68
8.05 ± 3.99
11.21 ± 3.68
XGBoost
13.03 ± 5.22 10.67 ± 3.97 11.44 ± 3.57 11.00 ± 2.81
10.89 ± 3.32
14.36 ± 4.19
The reason we believe that estimated performance can represent the optimization space is that the LLM services are always under-fitting and lowperformaning on these tasks. The higher the estimated performance of the LLM service, the stronger its potential language modeling ability for the task, indicating it can perform best after further optimization. Similarly, we performed the experiments in different 50 settings, and use the difference of performance to indicate how much the LLM service has improved after fine-tuning, which is defined as diff = perSfineturn T,C −perS T,C. Table 2 presents the results (in the form of perS T,C ± diff to display the changes before and after fine-tuning) of further optimization on the three worst performing tasks. It is shown that the estimation of LLM service performance can subconsciously indicate the best suitable service for fine-turning. And it can provide useful guidance for further optimization of LLM services on lowperformance tasks.
# 4.4 Ablation Study
We explore the influence of two important factors in our method. The effect of the different features selected on the results is shown in the Table 3, including NLL only and PPL only. Regardless of the meta-model architecture, using one feature alone resulted in larger estimation errors compared to using both features. This corroborates our idea that these two different features reflect the performance of LLM services from different perspectives.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/48a5/48a5045a-1c5d-4ea3-8b49-ce89c2f8c8ae.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 6: True and estimated performance on four different tasks. The number of unlabeled samples is from 200 to 1600, sampling in the interval of 200. The more opaque the color, the more the unlabeled samples.</div>
Another ablation study for the number of unlabeled samples is performed on the best meta-model architecture RandomForest. And we chose the best parameters setting obtained by grid search for ablation study with the number of unlabeled samples: the depth of the tree is 10, the number of weak learners is 260, and the sampling ratio is 0.8. For simplicity, we define n = |T ∗unlabeled| to represent the number of unlabeled samples that need to be used for the estimation of LLM service performance. And we performed experiments on all the tasks, reducing the average MAE from 6.30 for n = 200 to 2.52 for n = 1600. This effect is visually shown in Figure 6, where increasing n achieves better estimations, despite the need to perform additional LLM service invocations on unlabeled samples. In practice, we recommend 400 unlabeled samples for a task, which can accurately estimate the LLM service performance on the basis of controlling the invocation cost.
# 5 Related Work
# 5.1 Language Model as a Service
Large language models represent the latest development of generative AI and have shown outstanding performance in the service field, due to their outstanding natural language understanding and representation capability [31]. It is shown to have an attractive capability to "learn" [3, 7], that is, to perform unlearned tasks correctly given only a few labeled examples. However, a rapidly growing number of LLM services have different costs and qualities, resulting in the heterogeneity of the execution performance of the same task [5]. On the one hand, many studies have explored the problem of LLM service selection and composition [21, 25, 28], to obtain a more affordable and accurate solution to the invocation of LLM services. On the other hand, the performance of LLM services also strongly depends on the choice of prompt templates and examples [22,32]. The selection [33] and enhancement of the prompts [15] have been widely discussed to enhance the performance and generalization of LLM services. Among all these works, the estimation of the LLM service performance is key because it gives an indication that can be quantified and compared. And
this indication can be used to give guidance in the selection of services, order of invocations, optimization of prompts, and so on.
# 5.2 LLM Performance Estimation
LLM performance estimation aims to estimate LLM performance on a specific task in advance. Unlike evaluation [4], performance estimation occurs before invocation and focuses on unlearned datasets (out-of-distribution predictions) [13]. Early work focused mainly on estimation based on labeled data, and it is a straightforward idea to take a subset of the dataset and design experiments or benchmarks for performance prediction [17,27]. However, these methods are limited by the representativeness of the data and the cost of annotation. In recent years, with the increasing interest in unsupervised learning [19], researchers explore how to utilize the LLM internal information for performance estimation [16,20]. These approaches can bypass the need for labeled data, estimating the performance by analyzing LLMs’ hidden states or attention weights. approaches without labeled data are especially effective in domain tasks [12, 18], which learn model confidence to improve the performance estimates for specific tasks. But these approaches require the model to disclose internal details and have limitations for most LLM services that are published in black-box form. Previous studies provide us with rich experience and enlightenment to explore the LLM service performance estimation methods that do not rely on the labeled dataset. Our approach follows the idea of exploring the available features revealed when scaling on a wide range of unlearned unlabeled tasks, and based on this to estimate the service performance.
# 6 Conclusion
In conclusion, this paper presents a promising approach to addressing the challenge of estimating LLM service performance without labeled data, which can be conveniently applied in a "plug-and-play" manner to a variety of LLM services and tasks. By leveraging the meta-model based approach integrated with the ICL paradigm, our method offers accurate performance estimates that exceed baselines, facilitating informed decision-making in LLM service selection and optimization. Our work still has limitations that need to be explored, such as how to extend the method to larger models and more complex tasks, and how to leverage the ICL capabilities to enhance the generalization of meta-models. We believe that our contributions pave the way for the application of LLM services in practical scenarios, and we look forward to further research based on that.
Acknowledgments. The work is supported by the National Key R&D Program of China (Grant No.2022YFF0902703), the National Natural Science Foundation of China (Grant No.62472121), the National Natural Science Foundation of China (Grant No. 62306087), and the Special Funding Program of Shandong Taishan Scholars Project.
# References
1. Bobrowski, L.: Collinearity models in the eigenvalue problem. In: Nguyen, N.T., Hoang, D.H., Hong, T.P., Pham, H., Trawiński, B. (eds.) Intelligent Information and Database Systems. pp. 402–409. Springer International, Cham (2018) 2. Breiman, L.: Random forests. Mach. Learn. 45(1), 5–32 (oct 2001), https://doi. org/10.1023/A:1010933404324 3. Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., et al.: Language models are few-shot learners. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual (2020), https://proceedings.neurips.cc/ paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html 4. Chang, Y., Wang, X., Wang, J., Wu, Y., Yang, L., Zhu, K., Chen, H., Yi, X., Wang, C., Wang, Y., Ye, W., Zhang, Y., Chang, Y., Yu, P.S., Yang, Q., Xie, X.: A survey on evaluation of large language models. ACM Trans. Intell. Syst. Technol. (jan 2024), https://doi.org/10.1145/3641289 5. Chen, L., Zaharia, M., Zou, J.: Frugalgpt: How to use large language models while reducing cost and improving performance. CoRR abs/2305.05176 (2023), https: //doi.org/10.48550/arXiv.2305.05176 6. Chen, T., Guestrin, C.: Xgboost: A scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM (Aug 2016), http://dx.doi.org/10.1145/2939672.2939785 7. Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., el al.: Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311 (2022) 8. Cover, T., Hart, P.: Nearest neighbor pattern classification. IEEE Transactions on Information Theory 13(1), 21–27 (1967). https://doi.org/10.1109/TIT.1967. 1053964 9. Etezadi, R., Shamsfard, M.: The state of the art in open domain complex question answering: a survey. Appl. Intell. 53(4), 4124–4144 (2023) 10. Fadeeva, E., Rubashevskii, A., Shelmanov, A., Petrakov, S., Li, H., Mubarak, H., el al.: Fact-checking the output of large language models via token-level uncertainty quantification (2024) 11. Fu, H.Y., Ye, Q., Xu, A., Ren, X., Jia, R.: Estimating large language model capabilities without labeled test data. In: Bouamor, H., Pino, J., Bali, K. (eds.) Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023. pp. 9530–9546. Association for Computational Linguistics (2023), https://aclanthology.org/2023.findings-emnlp.639 12. Garg, S., Balakrishnan, S., Lipton, Z.C., Neyshabur, B., Sedghi, H.: Leveraging unlabeled data to predict out-of-distribution performance. In: The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net (2022) 13. Guillory, D., Shankar, V., Ebrahimi, S., Darrell, T., Schmidt, L.: Predicting with confidence on unseen distributions. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1134–1144 (2021) 14. Hall, M.A.: Correlation-based feature selection for machine learning. Ph.D. thesis, The University of Waikato (1999) 15. Haurum, J.B., Escalera, S., Taylor, G.W., Moeslund, T.B.: Which tokens to use? investigating token reduction in vision transformers. In: Proc. of ICCV (2023)
16. Huang, H., Wu, S., Liang, X., Wang, B., Shi, Y., Wu, P., et al.: Towards making the most of LLM for translation quality estimation. In: Liu, F., Duan, N., Xu, Q., Hong, Y. (eds.) Natural Language Processing and Chinese Computing Proceedings, Part I. Lecture Notes in Computer Science, vol. 14302, pp. 375–386. Springer (2023), https://doi.org/10.1007/978-3-031-44693-1_30 17. Kang, W., Ni, J., Mehta, N., Sathiamoorthy, M., Hong, L., Chi, E.H., Cheng, D.Z.: Do llms understand user preferences? evaluating llms on user rating prediction. CoRR abs/2305.06474 (2023), https://doi.org/10.48550/arXiv.2305.06474 18. Kaur, R., Jha, S., Roy, A., Sokolsky, O., Lee, I.: Predicting out-of-distribution performance of deep neural networks using model conformance. In: 2023 IEEE International Conference on Assured Autonomy (ICAA). pp. 19–28 (2023) 19. Kryeziu, L., Shehu, V.: A survey of using unsupervised learning techniques in building masked language models for low resource languages. In: 11th Mediterranean Conference on Embedded Computing, MECO 2022, Budva, Montenegro, June 7-10, 2022. pp. 1–6. IEEE (2022) 20. Li, Z., Kamnitsas, K., Islam, M., Chen, C., Glocker, B.: Estimating model performance under domain shifts with class-specific confidence scores. In: Medical Image Computing and Computer Assisted Intervention–MICCAI 2022: 25th International Conference, Singapore, September 18–22. pp. 693–703. Springer (2022) 21. Lu, K., Yuan, H., Lin, R., Lin, J., Yuan, Z., Zhou, C., Zhou, J.: Routing to the expert: Efficient reward-guided ensemble of large language models. CoRR (2023) 22. Perez, E., Kiela, D., Cho, K.: True few-shot learning with language models. Advances in neural information processing systems 34, 11054–11070 (2021) 23. Ranathunga, S., Lee, E.A., Skenduli, M.P., Shekhar, R., Alam, M., Kaur, R.: Neural machine translation for low-resource languages: A survey. ACM Comput. Surv. 55(11), 229:1–229:37 (2023), https://doi.org/10.1145/3567592 24. Rumelhart, D.E., Hinton, G.E., Williams, R.J.: Learning representations by backpropagating errors. nature 323(6088), 533–536 (1986) 25. Sakota, M., Peyrard, M., West, R.: Fly-swat or cannon? cost-effective language model choice via meta-modeling. CoRR (2023) 26. Sasaki, Y.: The truth of the f-measure. Teach Tutor Mater (01 2007) 27. Singhal, P., Forristal, J., Ye, X., Durrett, G.: Assessing out-of-domain language model performance from few examples. arXiv preprint arXiv:2210.06725 (2022) 28. Wang, C., Zhang, B., Sui, D., Tu, Z., Liu, X., Kang, J.: A survey on effective invocation methods of massive llm services. ArXiv abs/2402.03408 (2024) 29. Watanangura, P., Vanichrudee, S., Minteer, O., Sringamdee, T., Thanngam, N., Siriborvornratanakul, T.: A comparative survey of text summarization techniques. SN Comput. Sci. 5, 47 (2024), https://doi.org/10.1007/s42979-023-02343-6 30. Ye, Q., Lin, B.Y., Ren, X.: Crossfit: A few-shot learning challenge for cross-task generalization in NLP. In: Moens, M., Huang, X., Specia, L., Yih, S.W. (eds.) Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, Punta Cana, Dominican Republic, 7-11 November, 2021. pp. 7163–7189 (2021), https://doi.org/10.18653/v1/2021.emnlp-main.572 31. Yu, L., Chen, Q., Lin, J., He, L.: Black-box prompt tuning for vision-language model as a service. In: Proc. of IJCAI (2023) 32. Zhao, Z., Wallace, E., Feng, S., Klein, D., Singh, S.: Calibrate before use: Improving few-shot performance of language models. In: International Conference on Machine Learning. pp. 12697–12706. PMLR (2021) 33. Zhou, J., Li, F., Dong, J., Zhang, H., Hao, D.: Cost-effective testing of a deep learning model through input reduction. In: 31st IEEE International Symposium on Software Reliability Engineering, Coimbra, Portugal, October 12-15, 2020
