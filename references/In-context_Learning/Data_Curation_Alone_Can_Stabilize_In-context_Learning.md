Ting-Yun Chang and Robin Jia University of Southern California {tingyun,robinjia}@usc.edu
# Abstract
In-context learning (ICL) enables large language models (LLMs) to perform new tasks by prompting them with a sequence of training examples. However, it is known that ICL is very sensitive to the choice of training examples: randomly sampling examples from a training set leads to high variance in performance. In this paper, we show that carefully curating a subset of training data greatly stabilizes ICL performance without any other changes to the ICL algorithm (e.g., prompt retrieval or calibration). We introduce two methods to choose training subsets—both score training examples individually, then select the highest-scoring ones. CONDACC scores a training example by its average dev-set ICL accuracy when combined with random training examples, while DATAMODELS learns linear regressors that estimate how the presence of each training example influences LLM outputs. Across five tasks and two LLMs, sampling from stable subsets selected by CONDACC and DATAMODELS improves average accuracy over sampling from the entire training set by 7.7% and 6.3%, respectively. Surprisingly, the stable subset examples are not especially diverse in content or low in perplexity, in contrast with other work suggesting that diversity and perplexity are important when prompting LLMs.
arXiv:2212.10378v2
# 1 Introduction
In-context learning (ICL) is a new paradigm for few-shot learning with pretrained large language models (LLMs) without any parameter updates. In ICL, an LLM can perform a new task simply by conditioning on a prompt1 consisting of a sequence of labeled training examples. First introduced by GPT-3 (Brown et al., 2020), ICL with LLMs has reached state-of-the-art few-shot performance across many tasks (Rae et al., 2021; Smith et al.,
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/51fb/51fb511f-8783-4d53-8fea-7e33563cdd8f.png" style="width: 50%;"></div>
Figure 1: 4-shot ICL performance of GPTJ on SST2. Each boxplot summarizes the results of 50 sampled prompts. Compared with baselines (blue), our methods (pink) can greatly stablilize performance, having higher average accuracy (red diamonds) and lower variance.
2022; Thoppilan et al., 2022; Chowdhery et al., 2022). Compared with alternatives that use finetuning (Devlin et al., 2018; Schick and Schütze, 2020; Gao et al., 2020), ICL does not require taskspecific training, which enables its use with very large language models, and it uses a unified model for all tasks, enabling easier deployment. Despite its impressive few-shot performance, ICL often exhibits unintuitive behavior (Min et al., 2022). The standard ICL approach is to randomly sample a few examples from a training set to construct a prompt (Brown et al., 2020); however, prior work (Liu et al., 2021; Zhao et al., 2021; Lu et al., 2021) has found that ICL is very sensitive to the choice of training examples and their order in the prompt. ICL is also sensitive to small changes in prompt format (Chen et al., 2022). In this paper, we show that carefully curating a smaller training dataset from a larger pool can make ICL much more stable. We define a training subset E to be stable if randomly sampling a sequence of examples as a prompt from E yields much higher average and worst-case accuracy than randomly sampling from the original training set. We propose two methods to identify such a sta-
ble subset. Our CONDACC method scores a training example by its average dev-set ICL accuracy when combined with random training examples; these scores are closely related to Data Shapley values (Ghorbani and Zou, 2019). Our DATAMODELS method fits a linear regressor that predicts the LLM’s output based on which example is present at each index in the prompt; we score a training example highly if the associated weights from the linear model indicate that its presence improves accuracy. For both methods, we then select training examples with the highest scores to form the stable subset. While some prior work improves ICL accuracy by retrieving a suitable prompt for each test example (Liu et al., 2021; Rubin et al., 2021; Su et al., 2022), we show that it is possible to achieve stably good accuracy with a randomly sampled prompt for all test examples, when given the “right” training (sub)set. Our subset selection methods greatly improve performance across 5 classification datasets and 4 LLMs, with main experiments on GPTJ-6B (Wang and Komatsuzaki, 2021) and OPT-13B (Zhang et al., 2022a). On average, CONDACC and DATAMODELS outperform the baseline that uses the entire training set without selection (named ALL) by 7.7% and 6.3%, respectively, when comparing the average accuracy over multiple sampled prompts. In contrast, baselines that choose examples found in high-performing prompts (TOPPROMPTS) or examples that lead to high one-shot ICL accuracy (ONESHOT) to form the subsets do not perform as well (see Figure 1). Our stable subset examples generalize to out-of-distribution test data, and we can even find stable subsets for binary classification tasks that only contain examples of one label; these findings suggest that the stable subset examples help LLMs understand the overall task definition. Finally, we study what makes stable subset examples special by analyzing sequence length, perplexity, and diversity in both raw text and embedding spaces. We find that these examples do not have abnormally long sequence lengths or high perplexities. In contrast with prior work optimizing diversity for prompt selection (Su et al., 2022; Ye et al., 2022), we find our stable subsets no more diverse than random subsets of the training data. In summary, we show that curating training data appropriately leads to more stable and accurate ICL performance; we hope future work can develop new strategies for writing such helpful ex-
# amples. Code and data are publicly available at https://github.com/terarachang/DataICL.
# 2 Problem Setups
We use the original ICL formulation proposed by GPT-3, also known as the direct method, for all our experiments. Specifically, an LLM performs in-context learning on a new task based on a taskspecific prompt Z formed by concatenating K labeled training examples, i.e., Z = [z1, ..., zK], where each zj is a training example (x, y) consisting of an input x and label y. The LLM then makes predictions on a test input xtest conditioned on the prompt Z followed by xtest, denoted by arg maxy∈CP(y|Z, xtest), where C is the set of possible labels. Given a training set Dtr, a dev set Ddev, a heldout test set Dtest, and a predefined number of shots K, our goal is to select a stable training subset E ⊂Dtr, |E| > K, such that randomly sampling a sequence of K examples from E to form a prompt generally yields good performance on Dtest. We propose two setups, Labeled and Unlabeled. In Labeled, our goal is to study which training examples consistently lead to high ICL accuracy. We assume access to a large labeled DL tr = {(xi, yi)}Ntr i=1 of input-label pair, and a small labeled Ddev. Unlabeled is closer to the true fewshot learning setup (Perez et al., 2021), where we only have access to Ddev and a large unlabeled training set {xi}Ntr i=1. We pair each input xi with every possible label y ∈C to create an unlabeled training set DU tr = {{(xi, ˜y)|˜y ∈C}}Ntr i=1. In both setups, our goal is to select a subset of E training examples from either DL tr or DU tr . Note that we may select examples with incorrect labels under Unlabeled. We only use the large labeled Dtest for evaluating our selection methods.
# 3 Methods
We propose the following steps to identify a stable subset of E training examples:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/dc54/dc541465-648f-4f86-be8c-46c9e5dba365.png" style="width: 50%;"></div>
<div style="text-align: center;">stable subset consists of the highest-scroing examples</div>
Figure 2: An overview of our CONDACC method, which scores training examples individually using its averag accuracy (red diamonds) when combined with other random training examples. Each boxplot summarizes th dev-set accuracies conditioned on a training example appearing in the sampled prompts.
3. Select training examples with the highest importance scores per class to make up E (§3.3).
We hypothesize that a good training example leads to higher accuracy, on average, when occurs in a prompt. Given a prompt Z, we denote its dev set ICL accuracy as Acc(Z). Thus, a simple way to score the i-th training example (xi, yi) ∈Dtr is to calculate the expected accuracy conditioned on this example appearing in a prompt Z from DICL:
(1)
We ensure that each training example occurs in DICL multiple times in different orders and with different examples. In Appendix A.1, we show that Eq. 1 is similar to Data Shapley value (Ghorbani and Zou, 2019), where we define the valuation function of Data Shapley on subsets of K training examples since we focus on K-shot ICL.
# 3.2 Datamodels
The CONDACC method does not consider the order of training examples, which has a great impact on ICL performance. Also, it takes a simple average over the dev set, ignoring the LLM’s confidence in individual dev examples. Thus, we propose another data valuation method that leverages Datamodels (Ilyas et al., 2022) for ICL. Ilyas et al. (2022) study a complex target model’s behavior in terms of the training data by replacing it with a linear, easy-to-analyze, proxy model called a datamodel. Specifically, given a subset of training data S, a datamodel predicts the outcome of a
test input ¯x if training the target model on S. The outcome f(¯x; S) is defined as the margin of the correct class, i.e., the logit for the correct class minus the highest incorrect logit. To train datamodels, Ilyas et al. (2022) first create a dataset consisting of subset-outcome pairs (S, f(¯x; S)), requiring training the target model from scratch multiple times on different subsets to obtain the outcomes. In our work, the target model is a pretrained LLM, and is inference-only during ICL. Thus, our dataset collection only requires inference of the LLM multiple times with different prompts in DICL. In particular, given a prompt Z = [z1, ..., zK], we run ICL on a dev example (¯x, ¯y) and define an LLM’s outcome as f(¯x; Z) = o(¯y|Z, ¯x) − maxy′∈C, y′̸=¯y o(y′|Z, ¯x), where o(y|Z, ¯x) is the output logit of the LLM on label y before softmax. We hypothesize that we can approximate an LLM’s outcome with a linear regressor taking two simple features: the existence of a training example and its index in Z, which we consider to be the most important factors in ICL.Our linear datamodel is parameterized by weights w ∈RNtr×K and bias b ∈R; we use w(i, j) ∈R to denote the weight for the i-th training example appearing at index j. For each dev example (¯x, ¯y) ∈Ddev, we train a datamodel gw,b to predict the LLM’s outcome of ¯x, f(¯x; Z) ∈R, with mean-squared error:
(2)
where zj is the training example at index j in the
prompt Z, and id(·) maps zj back to its example ID i in the training set, i.e., zj = (xi, yi). By definition, f(¯x; Z) > 0 if the LLM is correct on ¯x. Hence, a positive w(i, j) indicates that having the i-th training example at index j in the prompt encourages answering correctly. We hypothesize that a good training example should have a beneficial effect for many dev examples, regardless of its index in the prompts. Thus, we can aggregate the datamodels of all dev examples and marginalize over all possible orders to calculate the score of the i-th training example:
(4)
where w¯x(i, j) is the weight value w(i, j) of the datamodel for ¯x. Calculating the total number of positive weights empirically works better than averaging the weights of all the datamodels. In Appendix A.3, we validate our hypothesis that we can linearize an LLM’s outcomes with two simple features. Table 6 shows that our datamodels can accurately approximate the outcomes of unseen prompts outside of DICL across different tasks.
# 3.3 Select Training Examples
Now that we assign a score for each training example (Eq. 1, 4), let C be the number of classes and E′ = E/C, we select the top-E′ training examples of each class with the highest scores to form the stable subset E ⊂Dtr, |E| = E.
# 4 Experiment
# 4.1 Setups
Tasks. We experiment on 5 classification tasks: SST-2 (Socher et al., 2013), BoolQ (Clark et al., 2019), Subj (Pang and Lee, 2004), Scicite (Cohan et al., 2019), and AGNews (Zhang et al., 2015). We set the stable training subset size E = 20 for all the tasks. For binary classification tasks, we set K = 4 and do not balance the classes in the prompts. Thus, the collected DICL for a binary task covers all 24 label patterns, including prompts with all positive ([1, 1, 1, 1]) and all negative ([0, 0, 0, 0]) labels, allowing us to better understand the impact of label patterns on ICL. For multiclass tasks (Scicite and AGNews), we balance the classes, sampling a training example per class to form the prompts. Table 8 in the appendix summarizes our setups.
Data Splits. For all the tasks, we use classbalanced Dtr, Ddev, and Dtest sampled from the original training set, as we do not have the gold labels of the original test set. We choose |Dtr| = 1000 to ensure a diverse range of training examples for subset selection, and |Dtest| = 1000 for reliable evaluation. Ddev consists of 50 examples per class. All three sets are balanced, randomly sampled from the original training set, and do not overlap.
Models. We run our main experiments with two LLMs: GPTJ-6B (Wang and Komatsuzaki, 2021) and OPT-13B (Zhang et al., 2022a). More experiments on GPT-Neo-2.7B (Black et al., 2021) and OPT-6.7B can be found in Table 9 in the appendix, where our methods also work well.
# 4.2 Evaluation and Baselines
Recall that our goal is to select a training subset E ⊂Dtr that is more stable than Dtr. To evaluate a selection method, we randomly sample 50 prompts from the selected subset E, run ICL on the test set Dtest, and report the average accuracy, standard deviation, and worst accuracy. As shown in Zhao et al. (2021), when a prompt only contains examples of a single label, LLMs are prone to always predict that label on every test example. Thus, in our main experiments (§5.1), we ensure that every sampled prompt contains at least one example from every class. In §5.2, we separately investigate performance when the prompt contains only one label of binary tasks. We split the selected subset E into two subsets, E0 and E1, where E0 only contains negative training examples and E1 only contains positive examples. We then sample 50 prompts from E0 and E1, respectively. Besides the two proposed selection methods, CONDACC and DATAMODELS, we design 5 baseline methods: ALL, CALIB, RANDOM, ONESHOT, and TOPPROMPTS. ALL uses the entire training set as E. CALIB uses the same prompts as ALL, but with calibration to prevent LLMs biased toward certain labels, where we closely follow the implementation of Zhao et al. (2021). RANDOM randomly selects a balanced training subset of E = 20 examples. ONESHOT first runs ICL with K = 1, using each training example alone as the prompt, and then scores the example by the corresponding ICL accuracy on Ddev; these scores are used in the same way as our main methods to select examples (§3.3). ONESHOT tests if we can extrapolate ICL performance from K = 1 to K > 1. TOP-
SST-2
BoolQ
Subj
Scicite
AGNews
Avg std
Min
Avg std
Min
Avg std
Min
Avg std
Min
Avg std
Min
Avg.
Tasks
GPTJ-6B
ALL
77.8 11.2
50.8
61.0 3.8
49.7
59.8 8.3
50.1
43.8 7.2
33.6
83.5 3.8
70.4
65.2
+ CALIB
75.5 9.5
53.6
61.2 3.9
50.4
70.4 7.7
55.7
35.4 2.6
32.8
85.2 2.7
78.0
65.5
RANDOM
74.6 11.4
50.3
60.0 4.3
49.5
59.9 10.4
50.1
46.4 6.9
35.5
82.5 4.7
67.1
64.7
ONESHOT
79.6 10.5
52.1
63.8 2.7
56.4
63.3 10.1
50.1
44.8 5.9
33.8
83.3 3.4
71.9
67.0
TOPPROMPTS-5
82.8 8.6
56.0
62.3 3.0
54.3
65.5 9.7
50.1
50.4 6.0
36.9
84.4 3.3
74.3
69.1
TOPPROMPTS-10
78.5 9.3
52.4
61.2 4.0
51.1
65.1 10.7
50.1
49.4 5.5
36.2
85.4 2.4
76.3
67.9
CONDACC
86.7 5.9
68.2
65.1 1.6
61.1
70.5 10.4
50.2
52.3 4.4
42.0
87.3 2.6
70.5
72.4
DATAMODELS
86.0 7.5
60.8
65.2 0.9
63.4
69.4 10.7
50.4
54.5 3.8
43.9
86.9 1.4
82.8
72.4
UN-ALL
71.0 11.9
50.0
60.8 3.5
49.6
60.1 8.8
50.1
42.0 7.0
33.5
75.1 9.9
46.5
61.8
UN-ONESHOT
81.9 6.3
68.5
62.6 3.3
55.6
61.0 8.7
50.1
43.5 6.7
33.4
78.1 4.2
69.8
65.4
UN-TOPPROMPTS-5
80.1 10.5
56.8
61.2 3.3
51.9
60.7 10.0
50.1
48.7 6.9
33.0
76.4 9.8
53.0
65.4
UN-CONDACC
85.3 6.8
60.5
63.7 2.2
56.0
66.0 10.6
50.1
54.2 3.4
45.9
87.1 1.1
84.6
71.3
OPT-13B
ALL
68.5 14.0
50.0
65.2 5.6
49.7
60.9 10.2
49.8
42.8 3.6
35.0
81.6 5.9
64.2
63.8
+ CALIB
84.7 6.8
51.7
65.5 4.9
51.8
63.7 8.9
47.9
35.5 1.8
31.2
81.8 4.1
70.7
66.2
RANDOM
67.7 14.1
50.0
64.7 6.4
49.3
61.2 9.5
49.9
41.2 4.6
33.3
78.0 7.5
61.4
62.6
ONESHOT
75.6 13.1
50.7
68.3 2.3
62.7
60.5 9.9
49.9
41.9 3.8
33.4
84.2 2.9
73.1
66.1
TOPPROMPTS-5
69.6 14.7
50.0
63.5 6.3
51.0
67.4 12.7
50.0
45.9 4.3
36.0
83.9 3.1
74.0
66.1
TOPPROMPTS-10
72.9 15.6
50.0
65.5 5.2
50.4
68.5 13.4
49.9
44.6 3.9
36.7
84.4 3.5
70.9
67.2
CONDACC
83.6 9.1
56.1
69.4 2.1
62.8
70.6 11.9
50.0
49.4 3.3
41.1
87.0 1.0
83.6
72.0
DATAMODELS
81.3 10.3
60.3
69.3 3.8
57.3
63.0 9.4
50.1
46.3 3.9
37.4
85.7 1.7
81.8
69.1
UN-ALL
61.6 13.6
50.0
64.8 5.3
49.3
55.8 8.9
35.6
41.9 3.6
35.7
67.3 17.2
26.4
58.3
UN-ONESHOT
74.8 15.6
50.0
68.0 2.5
59.8
54.8 6.2
47.1
41.5 4.1
33.7
82.3 4.5
64.9
64.3
UN-TOPPROMPTS-5
70.5 17.0
50.0
66.2 3.4
54.6
63.4 12.3
48.3
45.7 4.7
33.6
81.8 6.9
51.8
65.5
UN-CONDACC
80.3 12.8
50.0
69.0 2.6
61.5
63.7 11.7
49.9
48.1 4.0
39.2
84.6 3.1
72.5
69.2
Table 1: Main results with different selection methods. The last column average accuracies across all tasks. Overall the proposed methods CONDACC and DATAMODELS perform the best. Our method under the unlabeled setup (UN-CONDACC) even outperforms the ALL baseline that uses gold labels.
PROMPTS-5 and TOPPROMPTS-10 select the union of the examples from the top-{5, 10} prompts in DICL with the highest dev set accuracy, where the subsets contain at most K × 5 and K × 10 examples, respectively. Finally, we apply baselines and our CONDACC method to the unlabeled setup (§2), named with the UN- prefix.
# 5.1 Main Results
The proposed methods outperform all baselines. Table 1 shows the test set accuracy with different training subset selection methods. Among methods without calibration, our CONDACC and DATAMODELS methods are the most stable, achieving substantially higher average and worst-case accuracy across all tasks, and lower variances on most tasks. Compared with CALIB, our methods perform better in 8/10 setups. Overall, CONDACC and DATAMODELS outperform the no-selection baseline ALL by 7.7% and 6.3% on average, respectively.
TOPPROMPTS is the strongest baseline. Within the baselines, ALL and RANDOM have similar performance. Applying calibration improves the worst accuracy on most tasks and the average accuracy on some tasks, but is not always beneficial, especially on Scicite. ONESHOT outperforms ALL and RANDOM on SST-2 and BoolQ, but performs similarly on other tasks, indicating that we cannot extrapolate ICL behavior from K = 1 to K = 3 or K = 4. TOPPROMPTS-5 and TOPPROMPTS-10 are the strongest baselines, performing especially well on SST-2, Subj, and Scicite, showing that the training examples that compose the highestaccuracy prompts are more stable than others.
Our method works without training set labels. Randomly sampling prompts from the unlabeled training set (UN-ALL) underperforms sampling from the original labeled training set (ALL), especially on SST-2 and AGNews. This shows that gold labels do matter in ICL in general, in contrast with the findings of Min et al. (2022). However, apply-
ing our selection method to the unlabeled training set (UN-CONDACC) surprisingly outperforms not only UN-ALL but ALL (which uses correctly labeled examples), although some selected training examples actually have the wrong labels, implying that having gold-labeled prompts is not necessary for ICL. Other baselines, UN-ONESHOT and UN-TOPPROMPTS, outperform UN-ALL but substantially underperform our method. Overall, UNCONDACC outperforms baselines UN-ALL and ALL by 10.2% and 5.7% on average, respectively. Does UN-CONDACC benfit from gold labels? We study the number of the stable training examples selected by UN-CONDACC that indeed have gold labels. In most of the tasks, the numbers are much higher than the expected numbers by majority guess, where BoolQ is the exception with half of the selected examples having wrong labels. We thus study if we can achieve even better results by correcting those selected examples that have wrong labels with their gold labels; the other correct examples in the subset remain unchanged. After the label correction on BoolQ, the average and worst accuracy drops by 1.9% and 4.5% respectively on GPTJ, 0.4% and 5.7% respectively on OPT. These results again suggest that on the one hand, ICL benefits from gold-labeled examples in most cases; on the other hand, some training examples with wrong labels can surprisingly achieve better performance. The full results are in Table 10 in the appendix. Finally, we study the alternative that uses more shots in A.8. Table 12 shows that using 4 curated examples (CONDACC) can outperform K = 24, 16 randomly sampled ones in SST-2 and AGNews.
# 5.2 Single-Label Prompts
We now evaluate whether it is possible to achieve good accuracy while only using training examples of a single class in a prompt (See §4.2 for more details). Table 2 compares the results of different methods. First, the baselines ALL and TOPPROMPTS perform near chance in most cases, as the LLMs are biased by the prompts to predict the same label on every test example. In contrast, single-label prompts sampled from the subsets of CONDACC and DATAMODELS substantially outperform majority guessing across all setups. We conclude that the selected training examples are beneficial because they help LLMs understand the overall definition of the task. Thus, even when used in single-label prompts, they can still give LLMs
SST-2
BoolQ
[0,0,0,0]
[1,1,1,1]
[0,0,0,0]
[1,1,1,1]
GPTJ-6B
ALL
51.7 1.7
52.6 3.1
52.8 1.6
50.7 1.0
TOPPROMPTS
52.1 2.1
56.0 3.7
54.2 1.9
51.8 1.6
CONDACC
61.8 4.9
60.3 2.8
58.3 2.1
55.5 1.7
DATAMODELS
72.8 4.9
68.4 4.9
61.7 1.6
56.9 2.2
OPT-13B
ALL
54.0 4.1
73.3 7.4
52.4 2.6
51.2 1.7
TOPPROMPTS
53.0 3.0
76.5 6.9
53.0 3.2
51.4 2.1
CONDACC
66.3 3.7
81.0 4.6
65.3 2.9
53.7 1.9
DATAMODELS
63.9 4.2
84.5 2.6
69.2 2.0
60.1 2.2
Table 2: Results of single-labeled prompts with different selection methods. Each prompt consists of 4 training examples with the same labels.
OOD Tasks
IMDb
BoolQ Cst.
Avg std
Min
Avg std
Min
GPTJ-6B
ALL
86.5 5.7
63.6
56.6 3.0
50.1
TOPPROMPTS
87.2 5.2
63.0
56.7 2.6
49.9
CONDACC
90.5 1.8
84.8
58.9 1.7
54.6
DATAMODELS
91.6 1.5
84.0
57.6 1.9
54.0
OPT-13B
ALL
79.2 12.1
50.1
59.8 2.9
51.6
TOPPROMPTS
80.5 14.0
50.8
60.3 3.5
51.0
CONDACC
83.5 10.8
54.6
60.1 2.1
56.7
DATAMODELS
84.1 9.3
58.9
60.6 3.3
54.3
Table 3: Accuracy of IMDb and BoolQ Contrast Set, where the prompts consist of the selected SST-2 and BoolQ training examples, respectively.
# useful signal to perform the desired task.
# 5.3 Out-of-Distribution Tasks
We further evaluate on out-of-distribution (OOD) tasks, where there is a distribution shift between prompts and test data. Specifically, we apply our selection methods on a source task, sampling 50 prompts from the selected subsets of the source task as done in the main experiments, and then evaluate on test data of a target task. We use SST-2 and BoolQ as the source tasks, and IMDb (Maas et al., 2011) and BoolQ Contrast Set (Gardner et al., 2020) as our target tasks, respectively. Table 3 shows that our CONDACC and DATAMODELS methods still outperform baselines on OOD tasks, especially on IMDb, implying that instead of simply overfitting the source tasks, the selected stable examples are indeed task-level examples that can generalize to OOD test data.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d147/d1471356-3c39-459e-8e0a-b052b6b92fcd.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Accuracy versus sequence length (left) and accuracy versus perplexity (right). Each dot corresponds to a training example. Examples in good subsets are not outliers with abnormally long lengths or high perplexities.</div>
# 6 Analysis
We further analyze what makes the selected training examples special along different dimensions: sequence length, perplexity, and diversity. We compare good with bad training examples, where we use our CONDACC method to identify a bad (resp., good) subset by selecting E′ training examples with the lowest (resp., highest) importance scores in each class (§3.3). Please refer to Table 11 in the appendix for the full results of the bad subset.
# 6.1 Sequence Length and Perplexity
In Figure 3, we plot the accuracy against either sequence length or perplexity, where each dot corresponds to a training example. Here, the accuracy (Y-axis) is the importance score sca assigned to each training example by CONDACC in Eq. 1, i.e., the average dev-set accuracy when that example is combined with random other training examples in a random order.
Sequence Length. The first two subfigures show that while the bad examples (red dots) span across different sequence lengths, the good examples (blue dots) do not cover the tail distribution of long sequences. We observe little correlation between accuracy and sequence length across different tasks and LLMs (see more in Figure 6), except for a slightly negative correlation when the sequence length is very long, suggesting that abnormally long training examples can hurt ICL performance. Perplexity. We calculate the perplexity of the inputs of training examples with respect to the same LLMs we run ICL on. Figure 3 shows that good examples are not outliers that have high perplexity. This could suggest future work filter out examples that have extraordinarily high perplexity in the training set before running ICL, and could be combined with active learning for ICL (Zhang et al., 2022b; Su et al., 2022), as we only need the
unlabeled inputs to calculate perplexity. However, we observe no correlation between accuracy and perplexity across all the tasks and LLMs (Figure 7), implying that using perplexity alone is not enough for identifying good training examples. Our findings are inconsistent with concurrent work (Gonen et al., 2022), which shows that lower prompt perplexity strongly correlates with better performance, probably because Gonen et al. (2022) focus on perplexities under different instructions, while we focus on the differences between training inputs.
# 6.2 Diversity
DIV-I and DIV-F. We measure the diversity of a training subset with DIV-I (Yuan et al., 2020) and DIV-F (Zhdanov, 2019) metrics, following prior work in active learning. DIV-I measures diversity in raw text, while DIV-F measures diversity in a feature space. For DIV-F, we use SentBERT (Reimers and Gurevych, 2019) to encode the inputs of training examples into sentence embeddings, following Su et al. (2022). We compare the selected good subset and bad subset with 5000 randomly sampled subsets ⊂Dtr, each containing E′ training examples per class. Figure 4 shows that good subsets (blue dots) sometimes have low diversity scores in both metrics, especially on BoolQ and AGNews. Overall, they do not seem to be more diverse than randomly sampled subsets. Our findings are different from Su et al. (2022), which finds that diverse training subsets are better for prompt retrieval. We hypothesize that diversity matters more when retrieving similar training examples for each test input, but is less important when using a single fixed prompt. On the other hand, the bad subsets have much higher DIV-I scores than random subsets across different tasks, because they often include examples with long sequence lengths (see Figure 3), covering more distinct unigrams. However, in the SentBERT feature space, the bad subsets are often less diverse than random subsets.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/937e/937e4bc7-9ad8-4c5b-892b-3725cda7a8cd.png" style="width: 50%;"></div>
Figure 4: Different ways to visualize the diversity of examples. (a) and (b) compare the diversity of the good subset bad subset, and randomly sampled subsets (boxplot). For both DIV-I and DIV-F, a higher number means a subset is more diverse. Overall, good subsets are no more diverse than random subsets. (c) visualizes the stable training examples selected by CONDACC and DATAMODELS methods in Datamodels embeddings space, where each dot is a training example in AGNews. Both methods choose tightly cluttered examples instead of diverse ones.
Datamodels Embeddings. In §3.2, we learn a datamodel for each dev example in Ddev. Here, we concatenate the weights assigned on a training example learned by all the datamodels, creating an embedding ∈R|Ddev|×K for each training example. We then project the embeddings of the entire training set to a two-dimensional space with UMAP (McInnes et al., 2018) for visualization. Figure 4 and Figure 8 in the appendix show that both CONDACC and DATAMODELS choose tightly clustered sets of examples in the embedding space, instead of diverse ones scattering over the training set. Moreover, the two methods actually select similar examples in the embedding space, although having very different scoring methods.
# 6.3 Do LLMs find the same stable examples?
We further study if the identified stable subset examples are transferable across different LLMs. Given two LLMs, we calculate the Pearson correlation coefficient between their example scores assigned by CONDACC (§3.1) and the actual number of overlapped examples in the stable subsets of the two LLMs. Table 4 shows the mixed results: some pairs have moderate correlation, especially when both LLMs are from the OPT family; however, we find little correlation between many pairs and there are only a few overlapped examples between the stable subsets identified by different LLMs. Interestingly, when using the four stable SST2 examples shared by GPTJ-6B and OPT-13B as the prompt, evaluating all 4! permutations, we achieve very high test accuracy: 88.6 ± 3.7 on GPTJ, 89.8 ± 3.0 on OPT-6.7B, and 87.3 ± 5.2 on OPT-13B. This may indicate that there exist successful factors of training examples shared among LLMs. We include the four stable examples in
Corr
Overlap
GPTJ-6B vs OPT-13B
SST-2
0.15
4
AGNews
0.46
1
BoolQ
0.41
2
Subj
-0.03
2
Scicite
0.02
2
GPTJ-6B vs OPT-6.7B
SST-2
0.27
3
AGNews
0.08
1
OPT-6.7B vs OPT-13B
SST-2
0.76
3
AGNews
0.42
1
Table 4: Pearson correlation between the example scores (sca) of two LLMs and the number of overlapped examples in their stable subsets.
Examples
Review: k-19 : the widowmaker is a great yarn.
Sentiment: positive
Review: spiffy animated feature
Sentiment: positive
Review: a plot cobbled together from largely flat and
uncreative moments
Sentiment: negative
Review: has the thrown-together feel of a summer-camp
talent show : hastily written, underrehearsed, arbitrar-
ily plotted and
Sentiment: negative
Table 5: The four stable subset examples shared by GPTJ-6B and OPT-13B in SST-2 dataset, where the last example is also selected by OPT-6.7B. All three models achieve high average accuracy and low variance across the 4! permutations of these four examples.
Table 5 and hope future work can discover what distinguishes them from other examples.
# 7 Discussion and Related Work
# 7.1 Prompt Retrieval
The performance of ICL greatly depends on the choice of the prompt. A common way to do prompt selection is to retrieve the top-K similar training examples for each test input (Liu et al., 2021; Rubin et al., 2021; Su et al., 2022), where the similarity is captured by sentence embeddings (Reimers and Gurevych, 2019; Robertson et al., 2009). Such instance-dependent prompt retrieval is critical for semantic parsing tasks (Rubin et al., 2021), as LLMs need to see relevant logical forms in the context to generate the appropriate predicates for the test example. In this paper, we focus on classification tasks and identify task-level training examples that work for all test examples, avoiding the retrieval process.
# 7.2 The influence of in-context labels
The proportions of labels appearing in context can greatly bias LLMs’ predictions (Zhao et al., 2021). However, with careful prompt selection, Zhang et al. (2022b) finds that LLMs can perform well without observing the entire label space of a classification task in the prompt. In §5.2, we identify a subset, instead of just one prompt, of single-label examples that perform well. On the other hand, the correctness of in-context labels may not matter as much, as Min et al. (2022) find that randomly flipping them barely hurts performance. Kim et al. (2022) re-examine the importance of gold labels, showing it varies largely across tasks and experimental settings; our unlabeled experiments also show varying importance of gold labels across different tasks. Our method also identifies some training examples with wrong labels that can yield surprisingly good performance.
# 7.3 Data Valuation
Given a learning algorithm trained on a training set to produce a predictor, data valuation quantifies the value of each training example to the predictor performance. Prior work includes influence functions (Koh and Liang, 2017), Data Shapley (Ghorbani and Zou, 2019), DVRL (Yoon et al., 2020), TracIn (Pruthi et al., 2020), and Datamodels (Ilyas et al., 2022). Our setup also aims to attribute the performance of a predictor (in our case, an LLM) to each training example (in our case, in-context examples). Our CONDACC method closely resembles Data Shapley, and we adapt Ilyas et al. (2022)
as our DATAMODELS method. However, the main difference is that we are doing K-shot ICL, where training examples are used as prompts, and there are no parameter updates to LLMs. In concurrent and independent work, Nguyen and Wong (2023) also propose methods based on Data Shapley and Datamodels to study the influence of training examples. The main differences are: (1) we adapt Datamodels to consider the positions of training examples, while Nguyen and Wong (2023) follow the original implementation and study example ordering with influence scores. (2) They propose a Perplexity baseline that selects examples according to individual perplexity, while we use perplexity in analysis to study the correlation between example perplexity and their average performance. (3) Nguyen and Wong (2023) explore a larger number of shots in ICL (K ∈[10, 52]), while we assume a few-shot setting and fix K ∈{3, 4}. Our findings on good training examples are consistent with each other: both papers find little correlation between performance and potential factors such as example length and perplexity. In general, our work demonstrates the importance of data curation on in-context examples, even in the unlabeled and OOD scenarios, while Nguyen and Wong (2023) focus more on developing influence-based example selection frameworks. Taken together, two papers present a comprehensive view of data valuation for in-context learning.
# 8 Conclusion
We propose two methods to identify stable training subsets for in-context learning, achieving substantially higher average and worst-case accuracy across different setups. Our CONDACC method is intuitive and easy-to-implement, while our DATAMODELS method provides informative weights that enable further analyses. The success of our methods implies that when provided with the “right” training set (in our case, a subset of 20 examples to randomly sample prompts from), ICL could be far less sensitive to the choice of training examples and their orders in a prompt. Our analyses on stable subsets find that they do not contain outliers with especially long sequence lengths or high perplexities, and are also no more diverse than random subsets of the training data. We hope our work is a step towards developing guidelines for finding or writing better training examples.
The main limitation of our work is the high memory and computation cost. As both our methods estimate the importance of training examples based on the prompt-performance statistics, we first need to run in-context learning on the dev set multiple times with different prompts in DICL. Although ICL does not require any parameter updates, LLMs still require a large amount of memory footprint during inference, especially when the model size is large and the average sequence length is long. For each setup, our DICL contains around 50,000 prompts (see Table 8) and 50 dev examples per class, so the most expensive setup (running OPT13B on BoolQ) costs more than 500 GPU hours on an RTXA6000 GPU. Our preliminary study shows that the proposed methods need the statistics of at least 10,000 randomly sampled prompts to perform well. Future work may use search algorithms instead of random sampling during data collection to reduce the number of prompts in DICL. We also release our collected data of every setup in https://github.com/terarachang/DataICL to support future studies on ICL. In this paper, we only study classification tasks, for the sake of easy evaluation. Future work may study the influence of in-context examples in generation tasks under different evaluation metrics. Due to hardware constraints, we do not study LLMs of sizes over 13B, and we fix the number of shots and prompt templates for simplicity. In independent work, Nguyen and Wong (2023) complement these limitations of our paper, showing that similar approaches work well on larger models and a diverse number of shots for in-context example selection. Still, the influence of in-context examples for gigantic LLMs larger than 100B parameters has not been studied. Due to emergent abilities of LLMs (Wei et al., 2022), it is unclear whether our methods and findings would still apply when prompting these gigantic LLMs.
# Acknowledgements
We thank Ameya Godbole, Johnny Wei, Wang Zhu, Albert Xu, Deqing Fu, Qinyuan Ye, the members of USC NLP, and our anonymous reviewers for valuable feedback on the paper. We thank Sameer Singh and Matt Gardner for helpful discussions. We thank Jesse Thomason for his support. This work was funded in part by gifts from Open Philanthropy and Google.
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis.
Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. 2022. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990. Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1631–1642. Hongjin Su, Jungo Kasai, Chen Henry Wu, Weijia Shi, Tianlu Wang, Jiayi Xin, Rui Zhang, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, et al. 2022. Selective annotation makes language models better fewshot learners. arXiv preprint arXiv:2209.01975. Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. 2022. Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239. Ben Wang and Aran Komatsuzaki. 2021. GPT-J6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/ mesh-transformer-jax. Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682. Xi Ye, Srini Iyer, Asli Celikyilmaz, Ves Stoyanov, Greg Durrett, and Ramakanth Pasunuru. 2022. Complementary explanations for effective in-context learning. ArXiv, abs/2211.13892. Jinsung Yoon, Sercan Arik, and Tomas Pfister. 2020. Data valuation using reinforcement learning. In International Conference on Machine Learning, pages 10842–10851. PMLR. Michelle Yuan, Hsuan-Tien Lin, and Jordan BoydGraber. 2020. Cold-start active learning through self-supervised language modeling. arXiv preprint arXiv:2010.09535. Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022a. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068. Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. Advances in neural information processing
# A Appendix
# A.1 Connection with Data Shapley
Data Shapley has a valuation function V (S) for any subset of training examples. We define V (S) as the expected dev set ICL accuracy across all permutations of S if |S| = K, and V (S) = 0 otherwise since we focus on K-shot learning. Then, the Data Shapley value sshap(i) for the i-th training example ei = (xi, yi) is:
� � �� We claim that sshap(i) is a monotonically increasing (in fact, affine) function of sca(i), thus establishing a very tight connection between CONDACC and Data Shapley. First, note that the first term is exactly equal to sca(i), the expected conditional accuracy when ei occurs in a prompt, since V returns the expected accuracy over all orders of {z1, . . . , zK−1, ei}. Similarly, the second term is the expected conditional accuracy when ei does not occur in a prompt. Denote this quantity as t(i), and let A denote the overall expected accuracy when randomly sampling a prompt. Since the probability of choosing a given example to be in a prompt is exactly K Ntr , we have
    Since Ntr, K, and A are all constants that do not depend on i, and Ntr, K > 0, this establishes that sshap(i) is a monotonically increasing affine function of sca(i).
# A.2 Relation to Prompt Tuning
At test time, our setup is similar to Prompt Tuning (Lester et al., 2021), where a fixed prompt for
GPTJ-6B
OPT-13B
L1
Corr
L1
Corr
SST2
0.133
0.962
0.264
0.930
Boolq
0.147
0.941
0.167
0.937
Subj
0.269
0.946
0.260
0.949
Scicite
0.088
0.937
0.151
0.938
AGNews
0.296
0.891
0.340
0.840
Table 6: Test results of datamodels. Our datamodels can accurately predict LLMs’ outcomes on unseen prompts, having high correlation and low L1 distance to the ground-truth outcomes.
a task is prepended to the test inputs. In our case, a prompt is a sequence of training examples randomly drawn from the selected subset (fixed for all test examples). In Prompt Tuning, it is a set of continuous embeddings, called a soft prompt, learned through backpropagation. At training time, however, Prompt Tuning needs to backpropagate through the LLM and thus is more memory-expensive and tends to suffer from training instability (Asai et al., 2022). In comparison, our method finds good prompts without accessing the LLM’s parameters, which is a realistic setup as many LLMs only provide API access.
# A.3 Evaluating Datamodels
Recall that given a prompt Z and a dev example (¯x, ¯y), a datamodel learns to approximate an LLM’s outcome of ¯x (§3.2). To evaluate how accurate our datamodels can approximate an LLM, we create a test set for datamodels DDM, consisting of 5000 pairs of newly sampled Z and the LLM’s groundtruth outcomes of every dev example, where our sampling assures that every Z is made up of an unseen combination of training examples. We evaluate the learned datamodels on DDM, calculating the correlation and L1 distance between the predicted outcomes and the ground-truth outcomes. More specifically, each datamodel yields 5000 outcomes, which we calculate the Pearson correlation coefficient with the ground-truth outcomes of the associated dev example. We report the average correlation and L1 distance over all datamodels in Table 6. We also randomly sample 10,000 outcomes across all datamodels to visualize the ground truths against predictions in Figure 5. Overall, our datamodels can accurately approximate LLMs’ outcomes across different tasks and
Task
Example
Label Mapping
SST-2
Review: contains no wit , only labored gags
Sentiment: negative
negative/positive
BoolQ
Exercise: read the text and answer the question by yes or no.
Good Samaritan laws offer legal protection to people who give reasonable assistance...
Question: do good samaritan laws protect those who help at an accident? yes
no/yes
Subj
Input: the tucks have a secret , they ’re immortal .
Type: objective
objective/subjective
Scicite
Is the following citation from a scientific paper describing a method, a result, or background?
However, how frataxin interacts with the Fe-S cluster biosynthesis components...
Answer: background
method/result/
background
MNLI
"yeah well you’re a student right" Based on the previous passage, is it true that "Well you’re
a mechanics student right"? Yes, no, or maybe? maybe
yes/maybe/no
AGNews
Article: Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers...
Answer: Business
World/Sports/
Business/Technology
Table 7: Our templates and label mappings in different tasks. For simplicity, all the label words we use consist of  single token, so we can easily get the probability of each label.
LLMs. As our datamodels only consider two simple features, the existence of a training example and its index in Z, accurate test predictions may imply that these two features have a dominating effect on ICL.
# A.4 Training Details of Datamodels
As the pattern of in-context labels (e.g., [0,0,0,0], [0,0,0,1], [1,0,0,1]) has a great impact on LLMs’ predictions, for each label pattern, we train a set of |Ddev| datamodels. Specifically, we apply twophase training: in the first phase, we train on all data with shared weights. In the second phase, we first bucket prompts in DICL by their label patterns. Initializing from the weights learned in the first phase, we then separately fine-tune a set of datamodels for each label pattern. For example, for a binary task with 4-shot learning, there are 24 = 16 label patterns; thus, we have 16 sets of datamodels after the second-phase training, namely, 16×|Ddev| datamodels in total. We find that having two-phase training leads to more accurate predictions of LLMs’ outcomes in appendix A.3. Thus, when assigning scores for training examples, we aggregate all sets of datamodels to obtain sdm. When creating datamodel embeddings (§6), we use the unified weights learned by the first phase.
# A.5 Implementation Details
We use PyTorch and Huggingface transformers to implement in-context learning on GPT-Neo2.7B (Black et al., 2021), GPTJ-6B (Wang and Komatsuzaki, 2021), OPT-6.7B (Zhang et al., 2022a),
and OPT-13B. We run all our evaluations on a single RTXA6000 GPU (48GB). Most of our experiments can also be run on an RTX3090 GPU (24GB), except that OPT-13B model requires a GPU with larger memory. Our data collection on DICL costs hundreds of GPU hours on an RTXA6000. Once we finish the collection, our DATAMODELS method only takes about 5 seconds to train a datamodel on an i7-10700 CPU, and our CONDACC method does not involve any training, but simply calculates accuracy. Table 7 shows our task templates and label mappings, where we closely follow Bach et al. (2022); Lu et al. (2021). Table 8 summarizes our experimental setups on different tasks.
# A.6 More Experiments
Table 9 shows experiments on more LLMs and the MNLI task, where we evaluate on test data using 50 sampled prompts, as done in the main experiments (§4.2). Since ICL performs poorly on MNLI (majority: 33.3%) in both prior work (Su et al., 2022) and our results in Table 9, we do not experiment more on this task. Overall, our methods CONDACC and DATAMODELS substantially outperform other 4 baselines on all setups.
# A.7 Why not Instruction-Finetuned LLMs?
The massive multitask learning in instructiontuning leads to leakage in the datasets we evaluate on. For example, T0 (Sanh et al., 2021), FLANT5 (Chung et al., 2022), and OPT-IML (Iyer et al., 2022) are all trained on several tasks in our paper.
Task
Nclass
Bal.
K
|DL
tr |
|DU
tr |
PermutL
PermutU
|DICLL|
|DICLU|
SST-2
2
N
4
1000
2000
�1000
4
�
× 4!
�2000
4
�
× 4!
100,000
50,000
BoolQ
2
N
4
1000
2000
�1000
4
�
× 4!
�2000
4
�
× 4!
100,000
50,000
Subj
2
N
4
1000
2000
�1000
4
�
× 4!
�2000
4
�
× 4!
100,000
50,000
Scicite
3
Y
3
999
2997
(333)3 × 3!
(999)3 × 3!
40,000
50,000
AGNews
4
Y
4
1000
4000
(250)4 × 4!
(1000)4 × 3!
40,000
50,000
Table 8: Setups on different tasks. (1) We balance the classes in the prompts (Bal.) for multiclass tasks. (2) To create the unlabeled training set DU tr , we pair each input with every possible label; therefore, DU tr is Nclass times larger than the gold-labeled training set DL tr . (3) PermutL and PermutU denote the total number of possible permutations of training examples for K-shot ICL in the labeled and unlabeled setups, respectively. (4) |DICL L| and |DICL U| denote the number of the prompts we collect in the labeled and unlabeled setups, respectively.
SST-2
AGNews
Avg std
Min
Avg std
Min
GPT-Noe-2.7B
ALL
64.5 13.0
50.0
74.8 5.8
61.8
RANDOM
65.2 12.8
50.0
74.3 6.8
56.2
ONESHOT
66.1 12.8
50.0
78.3 4.4
64.9
TOPPROMPTS-5
64.1 12.7
50.0
79.9 3.4
71.1
CONDACC
76.5 10.5
52.4
82.3 2.2
77.4
DATAMODELS
72.6 14.2
50.4
83.5 1.4
80.0
OPT-6.7B
ALL
76.8 11.8
52.4
67.9 15.8
26.0
RANDOM
72.6 12.7
50.2
66.1 14.7
27.6
ONESHOT
84.7 6.6
65.7
76.3 7.8
56.7
TOPPROMPTS-5
78.8 10.9
50.4
78.2 9.7
30.8
CONDACC
88.2 5.8
59.4
83.2 4.3
67.2
DATAMODELS
87.4 5.3
71.7
84.2 3.0
74.0
MNLI
Avg std
Min
GPTJ-6B
ALL
43.8 2.9
35.8
RANDOM
44.1 2.7
33.8
ONESHOT
41.7 4.3
33.5
TOPPROMPTS-5
44.7 2.5
38.0
CONDACC
45.6 1.8
40.0
DATAMODELS
44.9 1.7
40.3
Table 9: More results of GPT-Neo-2.7B and OPT-6.7B (left) and GPTJ-6B on MNLI task (right). Overall, th proposed subset selection methods CONDACC and DATAMODELS significantly outperform other baselines.
GPTJ-6B
OPT-13B
Majority
SST-2
20
19
10
BoolQ
11
10
10
Subj
20
20
10
Scicite
16
11
6.6
AGNews
18
13
5
Table 10: The number of gold-labeled training examples identified by UN-CONDACC in the unlabeled setup. The subset size E = 20 for all tasks.
MNLI
Avg std
Min
GPTJ-6B
ALL
43.8 2.9
35.8
RANDOM
44.1 2.7
33.8
ONESHOT
41.7 4.3
33.5
TOPPROMPTS-5
44.7 2.5
38.0
CONDACC
45.6 1.8
40.0
DATAMODELS
44.9 1.7
40.3
GPTJ-6B
OPT-13B
Majority
SST-2
66.0 12.3
55.3 10.6
50.0
BoolQ
50.5 3.4
55.6 5.8
50.0
Subj
51.8 4.2
50.6 1.8
50.0
Scicite
33.9 1.2
36.2 2.6
33.3
AGNews
60.8 9.4
54.6 10.1
25.0
Table 11: Results of the bad training subsets, which consist of examples of the lowest scores assigned by CONDACCmethod.
A.8 Larger Number of Shots We further compare our methods with an alternative that takes as many labeled, balanced, incontext examples as the context window can fit, named MAXSHOT. Similar to the ALL baseline, MAXSHOT samples 50 prompts from the entire training set, each containing K training examples. The differences are that MAXSHOT uses a much larger K and balances the classes in the prompt for binary tasks as well. Here, our LLM is GPTJ-6B, which has a context window of 2048 tokens. Table 12 shows the number of shots K for each task and compares the average test set accuracy of MAXSHOT with the ones of ALL and CONDACC in Table 1. The ∆All column shows that using K ∈[8, 24] examples in the prompt substantially improves over only 3 or 4 examples in most tasks, except for AGNews. However, MAXSHOT only outperforms CONDACC on two out of five tasks (∆Our, blue). This shows the advantages of curated training examples over randomly sampled ones.
K
Avg std
∆All
∆Our
SST-2
24
85.8 5.5
8.0
-0.9
Subj
24
77.1 8.2
17.2
6.6
BoolQ
8
63.6 1.8
2.7
-1.5
Scicite
12
57.1 6.2
13.2
4.8
AGNews
16
82.3 4.8
-1.2
-5.0
Table 12: Test results of MAXSHOT on GPTJ, where ∆All and ∆Our show its improvements of average accuracy over ALL and CONDACC, respectively (both only use K = 3, 4 training examples).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a595/a5957695-8bde-49e0-bb0f-d3778aaa0eed.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) OPT-13B</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d608/d6083502-496c-4afe-8226-1c381c72e600.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) OPT-13B</div>
Figure 6: The accuracy versus sequence length across different tasks. Each dot corresponds to a training example. Note that we select the top E′ examples per class. As a class may have much lower average accuracy than others, the good (resp., bad) examples may not be the examples with the globally highest (resp., lowest) accuracy.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/eeb7/eeb7a299-b5f0-4332-94c1-c20206e7c46d.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) OPT-13B</div>
Figure 7: The accuracy versus perplexity across different tasks. Each dot corresponds to a training example. We do not observe any correlation between perplexity and accuracy. Note that we select the top E′ examples per class. As a class may have much lower average accuracy than others, the good (resp., bad) examples may not be the examples with the globally highest (resp., lowest) accuracy.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8ec2/8ec26b8c-45e7-40a5-9f9e-2e0e293ecf6b.png" style="width: 50%;"></div>
Figure 8: Visualizing the good training examples selected by CONDACC and DATAMODELS in datamodels embedding space across different tasks with GPTJ. Each dot is a training example, where datamodels spontaneously learn to encode class information in the embeddings.
