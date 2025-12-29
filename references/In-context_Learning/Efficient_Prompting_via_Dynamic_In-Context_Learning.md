# Efficient Prompting via Dynamic In-Context Learning
# Wangchunshu Zhou Yuchen Eleanor Jiang Ryan Cotterell Mrinmaya Sachan
wangchunshu.zhou, yuchen.jiang, ryan.cotterell, mrinmaya.sachan}@inf.ethz.ch
# Abstract
The primary way of building AI applications is shifting from training specialist models to prompting generalist models. A common practice for prompting generalist models, often referred to as in-context learning, is to append a few examples (demonstrations) to the prompt to help the model better understand the task. While effective, in-context learning can be inefficient because it makes the input prompt much longer, consuming valuable space in the context window and leading to larger computational costs. In this paper, we propose DYNAICL, a recipe for efficient prompting with black-box generalist models that dynamically allocate in-context examples according to the input complexity and the computational budget. To achieve this, we train a meta controller that predicts the number of in-context examples suitable for the generalist model to make a good prediction based on the performance-efficiency trade-off for a specific input. We then dynamically allocate the number of demonstrations for an input according to predictions from the meta controller and the given computation budget. Experimental results show that dynamic example allocation helps achieve a better performance-efficiency trade-off in two practical settings where computational resources or the required performance is constrained. Specifically, DYNAICL saves up to 46% token budget compared to the common practice that allocates the same number of in-context examples to each input. We also find that a meta controller trained on a certain backbone model and tasks can successfully generalize to unseen models and tasks.
The field of Artificial Intelligence is witnessing a major paradigm shift from training and deploying multiple specialist models for specific tasks to pre-training a generalist model (e.g., a large language model (LLM)) and prompting for different tasks [1–8]. While prompting is an elegant and effective way to utilize generalist models, the issue of computational inefficiency remains a major limitation. We identify two key sources of the computational inefficiency of prompting generalist models: model size and sample size. The former is arguably a prerequisite for generalist models to solve all kinds of tasks via prompting and there already exist a number of model compression techniques [9–12] that aim to reduce the size of generalist models. One obvious limitation of these approaches is that they all require users to have access to the model parameters, which may not be the case in the era of generalist models. For instance, some state-of-the-art generalist models like ChatGPT, Bard, PaLM, and Claude, are pre-trained by corporations and therefore closed-sourced for commercial reasons. In this paper, we instead focus on reducing sample size, a relatively new perspective for improving the efficiency of black-box generalist models of which the parameters are unavailable to users. This particular direction has received relatively limited or negligible exploration within the era of specialist models, as the inputs and outputs associated with it are clearly defined and largely devoid of redundancy. This is no longer true in the context of prompting generalist models such as LLMs
Preprint. Work In Progress.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2adf/2adf6ad0-d280-45de-b945-8427b7ee434c.png" style="width: 50%;"></div>
because we have a lot of different ways to prompt a model that results in prompts of different lengths. We identify the main factor influencing the prompt length to be the use of in-context learning and the number of in-context examples (demonstrations) in the prompt. Specifically, in-context learning [3] refers to the practice of adding a few exemplar input-output pairs that are related to the input, which helps the generalist model better understand and solve the problem. Although it is still unclear how in-context examples help a generalist model [13–15], it is evident that samples of greater complexity necessitate a greater number of in-context examples for a generalist model to acquire contextual understanding. Conversely, simpler samples may be solvable even without relying on in-context learning. This is confirmed by our preliminary study, which also finds that assigning more in-context examples to simple samples occasionally confuses the generalist model and turns its prediction from correct to erroneous. This suggests that the current practice of allocating a fixed number of in-context examples for all inputs is sub-optimal. To this end, we propose Dynamic In-Context Learning (DYNAICL), a dynamic computation framework for prompting generalist models. DYNAICL is conceptually similar to previous work on input adaptive computation for specialist models [16–21]. The main difference is that DYNAICL aims to dynamically adjust the size of the input while previous work focuses on adjusting the complexity of the model. This results in a major advantage of DYNAICL: it only operates on inputs, thus is disentangled with model architectures or parameters, and suits an increasingly common scenario in the era of generalist models where the users do not have access to the model’s parameters. To achieve this, we train a meta controller that predicts the number of in-context examples suitable for the generalist model to make a good performance-efficiency trade-off given a specific input. The meta controller can be instantiated with a smaller pre-trained model (e.g., FLAN-T5 [22]) and multi-task fine-tuned with the combination of supervised learning with a novel data synthesis algorithm and reinforcement learning with rewards based on performance-efficiency trade-off. Then at test time, we can dynamically allocate the number of demonstrations for an input according to both the predictions from the meta controller and the computation budget. We illustrate the procedure of efficient prompting with DYNAICL in Figure 1. We test the effectiveness of DYNAICL in the context of prompting LLMs due to its prominence as the predominant use case for generalist models at present. We experiment with ChatGPT as the generalist model and train a meta controller on a subset of the FLAN datasets collection [23]. We evaluate DYNAICL in two practical settings where either the computational resources or the performance is under constraints. We find that compared with the common practice of uniformly allocating in-context examples, DYNAICL can achieve an averaged absolute performance improvement of 2.6% within a certain computational budget or reach a certain performance requirement with up to 46% less compute (in terms of total token consumption) across 8 tasks. We also find that a meta controller trained on certain tasks with a certain generalist model (i.e., ChatGPT) can generalize well to unseen tasks (even with different output formats) and other generalist models (e.g., LLAMA [8]). To the best
of our knowledge, our work is among the first approaches that can accelerate a black-box generalist model without access to its parameters.
# 2 Methodology
# 2.1 Background: Prompting and In-Context Learning
We first recall some basics of prompting and in-context learning. Prompting refers to the process of providing a prompt, which typically contains a description of the task and the task input, to a generalist model that guides its response generation. Formally, let G be a generalist model and P be a prompt. Then, the output O is given by: O = G(P). Prompting relies on the generalist model’s ability to understand and follow abstract instructions, which sometimes leads to unsatisfactory empirical performance, especially for hard tasks that require complex reasoning. On the other hand, in-context learning leverages the ability of a generalist model to adapt to new information provided within the input context. Formally, given N labeled examples {(xi, yi)}N i=1 and a hand-crafted template T , in-context learning first verbalizes each input-output pair with a template, resulting in demonstrations di = T (xi, yi). Then the generalist model takes the concatenation of the original prompt and the demonstrations to generate the output:
O = G(P ⊕d1 ⊕d2 ⊕· · · ⊕dN) where ⊕denotes the concatenation of token sequences.
# 2.2 Meta Controller
Architecture and Input/Output Formats: The meta controller C can be modeled by any sequence generation model including both encoder-decoder models and decoder-only models. We use an instruction-tuned model such as FLAN-T5 as the backbone for the meta controller to facilitate training. As illustrated in Figure 1, it receives a task instruction and an input, which is identical to most instruction tuning literature [24, 22, 25]. But instead of generating the corresponding outputs like instruction-tuned models, our meta controller is trained to generate the number of in-context examples suitable for the input to achieve the best performance-efficiency trade-off, which we denote as k. This process can be expressed by k = C(P). The output expresses the confidence modeling of the meta controller for the generalist model to some extent. This method pertains to, albeit distinguishes itself from, prior existing work on model calibration [26, 27], which addresses the inherent confidence levels of the model itself. Training We then present our two-stage training framework for the meta controller. In the first stage, we train the meta controller to predict the minimum number of in-context examples for the generalist model to produce a good output. “A good output” can have different definitions for different tasks. For example, it can be defined as predicting the correct label with a high probability for classification tasks and generating outputs similar to the ground truth for generation tasks. In this paper, we consider only classification tasks following [28, 29]. To synthesize training data for supervised training, we propose a simple and intuitive data generation method. Specifically, for a prompt P, we consider the minimum number of in-context examples k∗for it to be the number that makes the generalist model’s expected accuracy exceed a certain (hand-crafted) threshold t:
k∗= min k∈N � k ���E(xi1,yi1)...(xik ,yik ))∼Dk [Acc(G(P, T (x1, y1) ⊕· · · ⊕T (xk, yk)))] > t �
� ��� � where Dk denotes all subsets of the training data of size k. We synthesize (P, k∗) pairs on a mixture of instruction-tuning datasets from the FLAN collection and train the meta controller with maximum likelihood estimation.
(1)
(2)
we define the reward R to be a linear interpolation of the expected performance (defined as accuracy in case of classification task), and the efficiency, defined as the number of in-context examples k:
where α is the weight controlling whether the controller should lean towards better performance o efficiency. The meta controller C is then fine-tuned with policy gradient:
where P is the set of prompts from a mixture of instruction tuning datasets, b is the baseline calculated as the moving average of previous rewards, and C(k|P, θ) denotes the predicted probability mass of k from the meta controller C for a prompt P. The training framework can be easily adapted for generation tasks by changing the accuracy metric to some generation metrics such as BLEU [30] or BERTScore [31], and doing some normalization to make it compatible with classification tasks. We leave this for future work.
where P is the set of prompts from a mixture of instruction tuning datasets, b is the baseline calculated as the moving average of previous rewards, and C(k|P, θ) denotes the predicted probability mass of k from the meta controller C for a prompt P.
The training framework can be easily adapted for generation tasks by changing the accuracy metric to some generation metrics such as BLEU [30] or BERTScore [31], and doing some normalization to make it compatible with classification tasks. We leave this for future work.
# 2.3 Dynamic In-Context Example Allocation
After training, the meta controller predicts the number of in-context examples for a specific input. This is a naive version of DYNAICL. However, in practice one may have a different computation budget. Therefore it is often desirable to normalize the predictions from the meta controller and dynamically adjust the actual number of in-context examples according to the computation budget. In this work, we propose a simple recipe for dynamic in-context example allocation. Assuming we have a budget of N tokens1 for K samples. The uniform baseline is to allocate N/(K · L) in-context examples for each sample assuming L is the average length of an example. DYNAICL instead allocates E in-context examples for an input P following:
where C(P) is the prediction from the meta controller, [] denotes the rounding operator, ˜C is the averaged prediction for all examples, and β is the token saving ratio ranging from 0 to 1.
# 3 Experiments
In this section, we test the empirical effectiveness of DYNAICL by experimenting on some NLP tasks with ChatGPT, a popular large language model, as the generalist model. We first describe the experimental settings. Then we begin with a preliminary study about the impact of the number of in-context examples to motivate our approach. After that, we evaluate DYNAICL by answering two research questions for two realistic settings:
• RQ1: To what extent can DYNAICL improves the performance of a generalist model with fixed computational budgets? • RQ2: To what extent can DYNAICL reduce computational cost or token consumption for a generalist model to achieve a fixed target performance?
# 3.1 Experimental Settings
Models We consider ChatGPT as the generalist model for training the meta controller and the main experiments. We use LLAMA-65B as an unseen generalist model for evaluating the generalization ability of the meta controller. We use FLAN-T5-large, which has less than 1B parameters, to initialize the meta controller. We also test with FLAN-T5-base in the analysis.
Tasks We use a subset in the FLAN collection containing 30+ classification tasks to train the meta controller. For evaluation, we test DYNAICL on both seen and unseen tasks, which are explicitly excluded from the training data for the meta controller. To be specific, we use SST-2 [32] AGNews [33], RTE [34–37], CB [38], ARC-E [39], ARC-C [39], MRPC [40], and COPA [41] as
(4)
(5)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/335b/335bd525-2eb6-40b6-a3c0-3168b27fdff7.png" style="width: 50%;"></div>
Figure 2: Distribution of the number of incontext examples that suffice for making the correct prediction for samples that cannot be answered correctly by zero-shot inference with generalist models but can be solved with in-context learning for up to 10 shots. The generalist model we consider are ChatGPT and LLAMA-65B, and the dataset is CSQA.
the seen tasks, and PIQA [42], OpenBookQA [43], CommonsenseQA [44], TriviaQA [45], Natural Questions [46], and Web Questions [47] as unseen tasks. It is noteworthy that TriviaQA, Natural Questions, and Web Questions are not classification tasks but a trained meta controller can still be used despite being trained only on classification tasks. This is because its input format (i.e., instruction + input) is agnostic to the type of the task. Training Details We follow Wei et al. [22] and fine-tune the meta controller for 30k/5k gradient steps with a batch size of 8,192 tokens using the Adafactor Optimizer [48] with a learning rate of 3e-5/1e-5, for the first/second training stage, respectively. Baselines We mainly compare DYNAICL with the uniform baseline that allocates the same number of in-context examples for each sample, and the random baseline that randomly samples a number of in-context examples from a Gaussian distribution. We only compare these two naive baselines
Baselines We mainly compare DYNAICL with the uniform baseline that allocates the same number of in-context examples for each sample, and the random baseline that randomly samples a number of in-context examples from a Gaussian distribution. We only compare these two naive baselines because there is no prior work in this direction and popular methods for efficient NLP can not be applied in this setting.
# 3.2 Preliminary Study: How Much Do More In-Context Examples Help?
We first conduct a preliminary study investigating the role of adding more in-context examples to the prompt for different samples. We first test if most samples for a task require a similar amount of in-context examples for a generalist model to generate a good output. We plot the distribution of the number of in-context examples that suffice for making the correct prediction for samples from the CommonsenseQA dataset that cannot be answered correctly by zero-shot inference with ChatGPT or LLAMA-65B but can be solved with in-context learning for up to 10 shots. As shown in Figure 2, different samples requires a very different amount of in-context examples. Some hard examples require 10 in-context examples for a generalist model to make the correct prediction while most examples require only one in-context example or can be solved with zero-shot inference. This observation confirms the necessity of dynamically allocating in-context examples according to sample difficulties. Moreover, we can see that ChatGPT and LLAMA-65B share similar trends in the Figure. This suggests that a meta controller trained with one generalist model may be able to generalize to other generalist models, which is later proved in our analysis. Then we further analyze the effect of scaling more in-context examples. As shown in Figure 3, the effectiveness of adding more in-context examples to the prompt is amortized when there are already a few (e.g., 5) in-context examples. This also supports our motivation that only a few samples require many in-context examples and uniformly allocating an equal number of in-context examples for all samples is a waste of tokens and computation. More interestingly, we find that sometimes it can be
∆Accuracy
→
→
zero-shot →1-shot
+ 2.5%
3.9%
1.4%
1-shot →5-shots
+ 1.4%
1.9%
0.5%
5-shots →64-shots
+ 0.3%
0.7%
0.4%
Figure 3: The impact of adding more incontext examples. ∆Accuracy denotes the change of accuracy after adding more incontext examples. → and → denotes the percentage of examples of which the predictions are changed from incorrect to correct, and vice versa, after adding more in-context examples. We use ChatGPT as the generalist model and TriviaQA as the dataset.
Models
SST-2
AGNews
RTE
CB
ARC-E
ARC-C
MRPC
COPA
Avg. Acc
zero-shot
ChatGPT
88.5
84.5
84.5
89.5
85.1
61.0
88.4
67.2
81.1
Budget: 5-shots on average
Uniform
93.2
87.9
86.1
91.1
88.3
64.8
90.4
88.2
86.2
Random
93.0
87.7
86.1
91.0
88.1
65.0
90.4
89.4
86.3
DYNAICL
95.3
90.2
88.1
92.9
90.5
68.4
91.8
93.0
88.8
Budget: 10-shots on average
Uniform
95.8
90.9
88.5
93.1
90.8
68.3
92.0
93.4
89.1
Random
95.9
90.7
88.4
93.3
90.8
68.2
92.1
92.8
88.9
DYNAICL
96.7
92.5
90.0
94.1
91.9
70.0
93.1
95.8
90.5
Table 1: Main results on seen tasks during meta controller training. The total computation/token budget is the same inside each group. DYNAICL consistently outperforms all baselines across all tasks under different computation/token budgets.
harmful to include more in-context examples for a sample that can already be correctly solved by the generalist model, which is shown by a non-negligible amount of samples’ predictions are changed from correct to incorrect after adding more in-context examples. This further confirms the potential of DYNAICL to achieve better performance while consuming fewer tokens.
# 3.3 Main Results
We first compare the performance of DYNAICL with the baselines in Table 1. We can see that DYNAICL leads to an averaged performance improvement of 2.6% and 1.4% over the uniform baseline with budgets of 5 and 10 in-context examples for each sample, respectively. This confirms that DYNAICL leads to improved performance with fixed budgets. We also plot the trend of averaged performance on seen tasks with different token-saving ratios in Figure 4 (a). We can see that DYNAICL leads to consistent improvements across all budgets and the improvements are larger when the computation/token budget is more limited. We then show the extent to which DYNAICL can save tokens for achieving a fixed target performance in Figure 4 (b). We can see that DYNAICL consistently require fewer tokens to match the performance achieved by the uniform baseline with certain budgets. Specifically, DYNAICL only consumes 108 tokens on average to match the performance of the common practice with 200 tokens on average. This confirms that DYNAICL can effectively reduce token/computation consumption for achieving a fixed target performance.
# 3.4 Analysis
We then conduct an analysis investigating the impact of different components in DYNAICL and the generalization ability of DYNAICL on unseen tasks or generalist models when training the meta controller.
We then conduct an analysis investigating the impact of different components in DYNAICL and the generalization ability of DYNAICL on unseen tasks or generalist models when training the meta
Ablation Study We first analyze the impact of the two training stages, the size of the meta controller, and the number of tasks the meta controller is trained with. The results are shown in Table 2. We find that both training stages contributes to the performance of DYNAICL and the first stage is more important. We think this is because the first training stage provides an important starting point for the second stage using reinforcement learning. We also find that DYNAICL with a smaller meta controller or a meta controller train on fewer tasks also achieves competitive performances. Generalization on Unseen Tasks We then test how well DYNAICL can generalize on unseen tasks. The results are shown in Table 3. We find that DYNAICL consistently leads to performance improvements across all 6 unseen tasks. Notably, DYNAICL also leads to substantial improvements on Natural Questions and Web Questions, which are generative question answering datasets that are very different from text classification tasks during training. This confirms that DYNAICL can generalize well on tasks that are not used to train the meta controller.
Ablation Study We first analyze the impact of the two training stages, the size of the meta controller, and the number of tasks the meta controller is trained with. The results are shown in Table 2. We find that both training stages contributes to the performance of DYNAICL and the first stage is more important. We think this is because the first training stage provides an important starting point for the second stage using reinforcement learning. We also find that DYNAICL with a smaller meta controller or a meta controller train on fewer tasks also achieves competitive performances.
Generalization on Unseen Tasks We then test how well DYNAICL can generalize on unseen tasks. The results are shown in Table 3. We find that DYNAICL consistently leads to performance improvements across all 6 unseen tasks. Notably, DYNAICL also leads to substantial improvements on Natural Questions and Web Questions, which are generative question answering datasets that are very different from text classification tasks during training. This confirms that DYNAICL can generalize well on tasks that are not used to train the meta controller.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/35c8/35c87a68-6778-46d1-b241-d527f9b99bc8.png" style="width: 50%;"></div>
Figure 4: Performance of DYNAICL on settings where either the computation (token) budget is fixed (a) or the target performance is fixed (b).
Models
SST-2
AGNews
RTE
CB
ARC-E
ARC-C
MRPC
COPA
Avg. Acc
Budget: 5-shots on average
Uniform
93.2
87.9
86.1
91.1
88.3
64.8
90.4
88.0
86.2
DYNAICL
95.3
90.2
88.1
92.9
90.5
68.4
91.8
93.0
88.8
- first stage
93.8
88.4
86.6
91.8
89.1
65.5
90.8
89.6
86.9
- second stage
94.4
89.5
87.5
92.1
89.5
67.1
91.2
91.4
87.8
w/ smaller model
94.8
89.2
87.5
92.3
90.2
67.7
91.3
92.2
88.2
w/ fewer tasks
95.0
89.3
87.3
92.5
90.0
68.0
91.5
92.4
88.3
Table 2: Ablation study results. “- first stage” and “- second stage” denotes the ablated variants where the meta controller is not trained with the first or second stage training, respectively. “w/ smaller model” and “w/ fewer tasks” denotes the ablated variants where the meta controller is parameterized with FLAN-T5-Base and the meta controller is trained with 50% less training tasks.
Generalization on Unseen Generalist Models We also test if DYNAICL can generalize to other generalist models that are not used for training the meta controller by applying the meta controller trained with ChatGPT with LLAMA-65B as the generalist model. Results in Figure 5 (a) show that DYNAICL still saves a great number of tokens for achieving the same performance with the uniform baseline even tested with a different generalist model. This confirms that DYNAICL can generalize well on generalist models that are not used to train the meta controller. Distribution of In-context Examples Count We then plot the distribution of samples according to the number of in-context examples allocated for them to better understand the meta controller. As shown in Figure 5 (b), with a target budget of 5 in-context examples, a large portion of samples are allocated with 5 in-context examples in DYNAICL. This indicates that most samples are predicted to need a similar number of in-context examples as the averaged prediction. We also find that more samples are assigned with fewer than 5 in-context examples while a few hard samples are assigned with more in-context examples. We present a qualitative study of different samples and the corresponding number of in-context examples allocated to them in the Appendix. Computation Cost of the Meta Controller Finally, it is noteworthy that the meta controller does add some computational cost and latency overhead to the overall prompting procedure. However, since the meta controller can use a very small backbone such as T5-large or T5-base, its computation cost is negligible compared to that of a generalist model. To be specific, the computational cost (in
Generalization on Unseen Generalist Models We also test if DYNAICL can generalize to other generalist models that are not used for training the meta controller by applying the meta controller trained with ChatGPT with LLAMA-65B as the generalist model. Results in Figure 5 (a) show that DYNAICL still saves a great number of tokens for achieving the same performance with the uniform baseline even tested with a different generalist model. This confirms that DYNAICL can generalize well on generalist models that are not used to train the meta controller.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e265/e26596a3-6414-4242-96bc-a62dce68bce2.png" style="width: 50%;"></div>
(b) Token saving ratio of DYNAICL compared to the uniform baseline under performance constraints, which are defined by the performance of the uniform baseline with different token budgets. Each point (x,y) in the line indicates that on average, DYNAICL needs to use y tokens to match the performance of the uniform baseline with x tokens.
Models
PIQA
OBQA
CSQA
TriviaQA (EM)
NaturalQ (EM)
WebQS (EM)
Avg.
zero-shot
ChatGPT
83.3
60.9
74.5
80.2
27.5
22.9
58.2
Budget: 5-shots on average
Uniform
84.3
61.5
76.6
84.1
37.1
26.3
61.6
DYNAICL
85.4
62.8
77.2
84.4
40.2
28.8
63.1
Budget: 10-shots on average
Uniform
85.9
63.1
77.4
84.3
40.8
29.2
63.4
DYNAICL
86.3
63.7
77.9
84.5
42.4
29.9
64.1
Table 3: Analysis of the generalization ability of DYNAICL on datasets that are unseen when training the meta controller. Tasks with (EM) suffix denotes the task is generative question answering and we use exact match as the metric. DYNAICL still consistently outperforms the baseline across all tasks.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1f2a/1f2a975c-9860-4579-995b-003b86bd82df.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Token saving ratio of DYNAICL compared to the uniform baseline under different performance constraints on seen tasks. DYNAICL is trained with ChatGPT but tested with LLAMA-65B.</div>
<div style="text-align: center;">Figure 5: Analysis on the generalization ability of DYNAICL on unseen generalist models and the distribution of samples according to the number of in-context examples allocated for them.</div>
Figure 5: Analysis on the generalization ability of DYNAICL on unseen generalist models and t distribution of samples according to the number of in-context examples allocated for them.
terms of FLOPs) of a T5-large based meta controller for a sample of 50 tokens is less than 0.1% of the change of the computation cost when changing the input from 200 tokens to 199 tokens, or less than 0.0005% of the computational cost saved by reducing one in-context example from the prompt. Similarly, since the meta controller only needs to predict 1 or 2 tokens, the latency overhead accounts for only 0.1% to 0.2% of the latency of calling the GPT-3.5-turbo API, and reducing one in-context example will lead to a speedup of around 10%. In sum, we believe the computational and latency overhead from the meta controller is almost negligible.
# 4 Related Works
# 4.1 Generalist Models, Prompting, and In-context Learning
Training a generalist model that can solve a wide range of tasks without task-specific training has been a long-standing goal in the field of artificial intelligence. One pioneering work dates back to Collobert and Weston [49] that attempted to solve all NLP tasks with a shared architecture using multi-task learning. This idea is further improved by decaNLP [50] that proposes to convert all NLP tasks to question answering format. T5 [51] then improves this paradigm by using text-to-text format for unifying all NLP tasks, which is more general and friendly to scaling. Finally, GPT-3 [3] show that by scaling model size, training data, and training FLOPs, a large language model can serve as a generalist model that solves many tasks by simply writing a prompt that describes the task and the input. They also showed that the zero-shot ability of a large language model can be further improved
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4911/49118d48-8a76-4ef0-85c8-8f1d84a5eb60.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Distribution of samples (on seen tasks) according to the number of in-context examples allocated for them. The computational budget is fixed to 5 in-context examples per sample.</div>
by adding a few input-output demonstrations in the prompt to help the model better understand the task. Since then, a large number of work has been done for improving and understanding prompting and in-context learning with large language models. For instance, Schick and Schütze [52] show that small encoder models can also be prompted. Min et al. [13] show that in-context examples mainly help a generalist model learn output label space and distribution of input text. Kadavath et al. [27] prove that generalist models are well calibrated and can be trained to model their confidence level. Hao et al. [28] and Li et al. [29] show that in-context learning with many examples improves the overall performance of a generalist model. Recently, the paradigm of prompting generalist models is successfully transferred to other modalities other than language. For example, Zhou et al. [53] and Li et al. [54] explored prompting vision models. It is foreseeable that prompting generalist models will become the de-facto paradigm for most domains in artificial intelligence.
# 4.2 Efficient Deep Learning
Improving the performance-efficiency trade-off of deep learning models is a very interesting research problem and is attracting more and more attention since the rise of large generalist models [55, 56]. A large number of work has been done on improving the speed and efficiency of large language models, including both static methods such as knowledge distillation [57, 58, 9], pruning [59, 10], quantization [60, 61, 11] and module replacing [12]; and dynamic methods such as adaptive computation [17], early-exiting [18–20], and model cascade [62, 63]. However, most of the aforementioned methods require access to the model parameters, which may not be possible in the era of generalist models since most state-of-the-art generalist models such as ChatGPT and PaLM are closed-sourced. One potential exemption is model cascade, which first sends the input to a cheaper model and optionally sends it to a more powerful model if the previous model is not confident enough. This method, however, also faces server latency issues because harder samples will be computed by multiple generalist models sequentially. Modarressi et al. [64] proposed dynamic input reduction methods to reduce the length of the input. However, their approach requires access to the model parameters and needs to modify the model architecture and additional training. Concurrently to our work, Mu et al. [65] proposes to train gist tokens to replace long-form prompts and show promising results on prompt compression. However, this approach is still limited to white-box settings where the model parameter is available and also compromises interpretability. Our approach instead focuses on the black-box setting and can be applied to any closed-sourced generalist models.
# 5 Limitations
As for technical limitations, the main limitation of this work is that we only test DYNAICL on NLP tasks with LLMs as the backbone, while it may also be interesting to test on other modalities such as vision tasks with multi-modal generalist models. This is because the main experiments are conducted before multi-modal instruction following models such as LLAVA came out. We leave this for future work. Another limitation is that we only train the meta controller with text classification datasets. We explain how the meta controller can be trained on generation tasks at the end of Section 2.2. We also experiment with some generative question answering datasets and show DYNAICL trained only on classification tasks can successfully transfer to these tasks. Finally, the dynamic in-context example allocation algorithm is quite naive. Potential improvements may be made using some more sophisticated planning or optimization algorithms. We also leave this for future work. As for social impact, this work aims to reduce the token/computation consumption of prompting generalist models. It probably leads to a positive environmental impact and will unlikely lead to any negative social impact.
# 6 Conclusions
This paper introduces DYNAICL, a framework for efficiently prompting generalist models. We propose to train a meta controller that predicts the suitable number of in-context examples for a specific sample with a two-stage training framework. During inference, DYNAICL dynamically allocate different number of in-context examples to samples according to the predictions from the meta controller and the given computational budget. Our experiments show that DYNAICL consistently
leads to better performance-efficiency trade-offs across tasks and generalist models that are either seen or unseen when training the meta controller. This work is among the first that investigates how to prompt generalist models more efficiently even they are only available through API calls and will hopefully shed light on this direction.
# References
1] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.
[2] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
[3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/ 1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf. [4] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022. [5] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Gray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/ forum?id=TG8KACxEON.
[5] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Gray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/ forum?id=TG8KACxEON.
# 6] OpenAI. Gpt-4 technical report, 2023.
[7] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022. [8] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. [9] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter, 2020. [10] Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper_files/paper/2019/file/ 2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf. [11] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. GPT3.int8(): 8-bit matrix multiplication for transformers at scale. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=dXiGWqBoxaD.
[7] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.
[40] William B. Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005), 2005. URL https://aclanthology.org/I05-5002. [41] Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In AAAI spring symposium: logical formalizations of commonsense reasoning, pages 90–95, 2011. [42] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning about physical commonsense in natural language. In AAAI, pages 7432–7439. AAAI Press, 2020. [43] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1260. URL https://aclanthology.org/D18-1260. [44] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149– 4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1421. URL https://aclanthology.org/N19-1421. [45] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1147. URL https://aclanthology.org/P17-1147. [46] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019. doi: 10.1162/tacl_a_00276. URL https://aclanthology.org/Q19-1026. [47] Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on Freebase from question-answer pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1533–1544, Seattle, Washington, USA, October 2013. Association for Computational Linguistics. URL https://aclanthology.org/D13-1160. [48] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost, 2018. [49] Ronan Collobert and Jason Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of the 25th International Conference on Machine Learning, ICML ’08, page 160–167, New York, NY, USA, 2008. Association for Computing Machinery. ISBN 9781605582054. doi: 10.1145/1390156.1390177. URL https://doi.org/10.1145/1390156.1390177. [50] Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. The natural language decathlon: Multitask learning as question answering, 2018. [51] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(140):1–67, 2020. [52] Timo Schick and Hinrich Schütze. It’s not just size that matters: Small language models are also few-shot learners. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2339– 2352, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021. naacl-main.185. URL https://aclanthology.org/2021.naacl-main.185.
