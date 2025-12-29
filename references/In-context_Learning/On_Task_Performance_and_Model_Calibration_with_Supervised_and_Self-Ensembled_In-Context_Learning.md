# On Task Performance and Model Calibration with Supervised and Self-Ensembled In-Context Learning
Chengzu Li1, Han Zhou1, Goran Glavaš2, Anna Korhonen1, Ivan Vuli´c1 1Language Technology Lab, University of Cambridge 2Center for Artificial Intelligence and Data Science, University of Würzburg {cl917, hz416, iv250, alk23}@cam.ac.uk goran.glavas@uni-wuerzburg.de
# Abstract
Following the standard supervised fine-tuning (SFT) paradigm, in-context learning (ICL) has become an efficient approach propelled by the recent advancements in large language models (LLMs), yielding promising performance across various tasks in few-shot data setups. However, both paradigms are prone to suffer from the critical problem of overconfidence (i.e., miscalibration), especially in such limited data setups. In this work, we deliver an in-depth analysis of the behavior across different choices of learning methods from the perspective of both performance and calibration, as well as their interplay. Through extensive controlled experiments, we find that simultaneous gains for both task performance and calibration are difficult to achieve, and the problem of miscalibration exists across all learning methods in low-resource scenarios. To address this challenging trade-off between performance and calibration, we then investigate the potential of self-ensembling techniques applied at different modeling stages (e.g., variations of in-context examples or variations in prompts or different ensembling strategies). We justify the feasibility of self-ensembling on SFT in addition to ICL, to make the predictions more calibrated and have comparable or even better performance. Our work sheds light on which learning paradigm to choose and how to enhance both task performance and calibration of LLMs.
  22 Dec 2023
# 1 Introduction
Machine learning and NLP have undergone a significant transformation recently, largely propelled by large language models (LLMs) (Radford et al., 2019; Brown et al., 2020; Chowdhery et al., 2022; OpenAI, 2023). Among different learning paradigms, Supervised Fine-Tuning (SFT) and InContext Learning (ICL) have emerged as predominant methodologies (Raffel et al., 2020; Dong et al., 2022), demonstrating commendable efficacy across
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f3f7/f3f7a7f8-d449-4d66-9ed8-7ea8995136b3.png" style="width: 50%;"></div>
Figure 1: Illustration of the self-ensembled learning methods. We introduce different types of variations to the input and feed them to a single language model. After having the predictions, we run self-ensembling to obtain final predictions and their confidence. many tasks. SFT tunes the model’s parameter and effectively specializes a (general-purpose) model to specific tasks by learning the knowledge in the training data and optimizing the objective. ICL, for each input, leverages the few-shot examples (i.e., the so-called demonstrations) to generate predictions without tuning model parameters and treating the model as a ‘black box’. Considering the different input format between training with SFT and inference with ICL, Min et al. (2022) and Chen et al. (2022) introduce the in-context examples into training phrase, which we call supervised in-context learning (SICL). However, when the demonstrations as a strong inductive bias get combined with SFT, it has been shown to become more likely to fall into the problem of overconfidence (Desai and Durrett, 2020; Jiang et al., 2021), and the predicted confidence distribution of ICL may be miscalibrated due to the bias in in-context examples (Fei et al., 2023). Through our extensive experiments, we observe that both paradigms, SFT and ICL, suffer from the problem of miscalibration in low-resource scenarios.
The important challenges of overconfidence and miscalibration, particularly in scenarios marked by limited data availability, underscore the need for a nuanced understanding of these paradigms. However, most of the previous work (Mosbach et al., 2023; Sun et al., 2023) only focus on comparing solely the performance of SFT and ICL on out-of-distribution (OOD) data, targeting generalpurpose LLMs. Here, we instead focus on studying task-specialized language models, where the behavior of different paradigms’ in-task performance along with their calibration remains an open research question. In addition, considering the possible issue of overconfidence and miscalibration, we pose and study another crucial research question: is it possible to ensure both in-task performance and well-calibrated LM behavior at the same time? Satisfying both requirements is critical to the application of the model in real-life setups: an applied system should provide both accurate and calibrated (trustworthy) predictions to be responsible. To address the challenges above, in this paper we first investigate the performance and calibration of different model-tuning and ICL methods, along with their interplay, on 7 classification datasets in limited data setups. Our empirical investigations unveil a task-dependant phenomenon of in-task performance and calibration with regard to whether the corpus has been seen by the model before, which also gains increasing attention over the research community in data contamination (Zhu et al., 2023; Deng et al., 2023). We further observe that most of the results show relatively high calibration errors, which are in contrast with responsible LLM applications. In response to the possible miscalibration in lowresource settings, we introduce and explore the application of self-ensembling, inspired by the effect of ensembling of multiple independent models in improving the reliability of the final predictions (Ovadia et al., 2019). We incorporate diverse variations in in-context examples and prompts, tailor them to different learning methods, and find that self-ensembling fortifies the model’s calibration without compromising performance. The overall flow of the self-ensembling method is illustrated in Figure 1. This strategic approach not only contributes to reducing the calibration error of model predictions by 43% on average but also streamlines resource utilization and time expenditures by focusing on modifications to a single model.
Contributions. 1) We deliver a comprehensive empirical analysis with different choices of learning methods across a variety of tasks in limited data scenarios. 2) We show the task-dependent relationship between in-task performance and calibration of LLMs and provide practical guidelines for the choice of learning paradigms (§6.1). 3) We investigate and justify the feasibility of the self-ensembling method in enhancing both the performance and calibration of LLMs (§6.2). We release the code at https://github.com/ cambridgeltl/ensembled-sicl.
# 2 Related Work
Learning Paradigms. Fine-tuning (FT) pretrained LMs has been used as an effective method to adapt them to specific tasks and datasets (Devlin et al., 2019; Raffel et al., 2020). With the generation of much larger and more powerful LMs (Brown et al., 2020; Chung et al., 2022), parameter-efficient finetuning (PEFT) (He et al., 2022; Zhou et al., 2023b) has been proposed, where the central idea is to tune only a fraction of the model parameters to reduce computation and memory costs. Without tuning the model, in-context learning (ICL) has shown great potential, achieving promising performance on various tasks with demonstration examples (Brown et al., 2020). Motivated by the positive results of ICL by concatenating multiple in-context examples to the LLM input at inference, Min et al. (2022) and Chen et al. (2022) introduce labeled in-context examples to the supervised training process (SICL). This has been further improved and utilized with pretraining (Gu et al., 2023; Shi et al., 2023) and other training strategies (Ye et al., 2023b,a; Wei et al., 2023). Sun et al. (2023) study task performance and stability with different PEFT methods in addition to ICL, including prompt tuning, and instruction tuning (IT) (Singhal et al., 2022). Duan et al. (2023) explore the relationship between ICL and IT, and interpret ICL as implicit IT. Mosbach et al. (2023) compare the generalization ability of ICL and FT on out-of-distribution (OOD) data. Zhou et al. (2023c) show the superior performance of hard prompt tuning over standard FT in low-data scenarios. However, previous work has not explored all the learning methods systematically within low-data scenarios and has not investigated them through the joint optics of in-task performance, confidence, and their trade-offs.
Calibrating LLMs. A well-calibrated LM should be accurate in terms of performance while also producing reliable confidence estimates for their predictions. When concatenating different in-context examples with various labels, tokens, and example ordering, ICL would be influenced by the bias in the concatenated examples (Zhao et al., 2021), and various calibration methods have been proposed to mitigate this issue (Wang et al., 2023a; Zhou et al., 2023a). Moreover, the model itself also contains certain ‘implicit’ label biases due to its (pre)training corpus, which would have an effect on the confidence estimation as well (Fei et al., 2023). Furthermore, FT may suffer from miscalibration for both in-distribution and OOD data due to overparameterization when adapting to the specific data (Kong et al., 2020), and the miscalibration effect, as we investigate in this paper, might be even more pronounced in limited data scenarios. Ensembling Model Predictions has been used to mitigate the problem of overconfidence and improve the reliability of the final model predictions (Ovadia et al., 2019). A standard ensembling practice is to train the model with different hyper-parameters or different initialization (Wenzel et al., 2020; Lakshminarayanan et al., 2017). Concerning LLMs, Sun et al. (2022) ensemble fine-tuned LLMs to quantify the uncertainty with disagreement among different ensembling components. Gleave and Irving (2022) and Wang et al. (2023b) ensemble partially tuned LLMs, considering the computation resources for training and the storage for saving different LLM checkpoints. Although the proposed method achieves better and more reliable results, it takes a considerable amount of time, computation, and storage resources to train and save multiple models, which often makes them inapplicable to LLMs. In contrast to having several tuned models with supervised learning, Yao et al. (2023) demonstrate the feasibility of using self-ensembling in ICL without tuning the model. In this paper, we explore self-ensembling in the novel context of diverse learning paradigms and low-data setups and demonstrate that it is possible to improve model calibration without compromising performance.
# 3 Background: Learning Paradigms
In this paper, we analyze and compare four different learning paradigms in low-resource scenar-
ios: zero-shot learning (ZSL),1 in-context learning (ICL), supervised fine-tuning (SFT) and supervised in-context learning (SICL). With classification tasks in focus, we briefly describe each paradigm in what follows. Zero-Shot Learning (ZSL). Given the input x and the prompting template fp, the prediction ˆy from the LM can be represented as ˆy = arg maxj P(yj|fp(x)), where the parameters of the underlying LM are fixed. The prompting template fp(x) includes the task instructions and special symbols which can be replaced by the input x. We attach the prompting templates for different classification tasks in Appendix D. In-Context Learning (ICL). Similar to ZSL, instead of only feeding the input x to the model, we first prepend M in-context examples (IC) (also called demonstrations) [fp(x1), y1; ...; fp(xM), yM] to the input x. The examples are retrieved from the pool of examples R following (random or non-random) selection strategy. The prediction is then defined as ˆy = arg maxj P(yj|[fp(xIC), yIC], fp(x)). Supervised Fine-Tuning (SFT). As mentioned, ZSL and ICL are inference-only paradigms treating the LM as a black box. On the other hand, FT first trains the model on the training set following the input format fp(x) from ZSL. Note that here we use SFT to refer to instruction-style finetuning with a prompting template. Inference with the tuned model P′ is then conducted in the same way as with ZSL. Both during training and inference, we can use different prompting templates to create variations in the model input, which we further elaborate on in §4.1. Supervised In-Context Learning (SICL). Based on the propositions from Min et al. (2022) and
# Supervised In-Context Learning (SICL). Based
on the propositions from Min et al. (2022) and Chen et al. (2022), we can also fine-tune the model to directly optimize the in-context learning objective. For each training step, M in-context examples (x1, y1), ..., (xM, yM) are selected from the pool R. We then prepend the selected in-context examples to the input x as before with ICL and use the concatenation as the final model input, and train the model to generate y. Inference proceeds in the same way as with ICL, except that we now use the task-tuned model P′.
1In the context of ZSL and later ICL there is no actual ’learning’ taking the place, and the model simply reacts to the provided prompt, but we have chosen the term ZSL for consistency with previous literature.
# 4 Methodology
There are two points that create possible variations that can be used for ensembling: 1) variation in the selection of in-context examples (for ICL and SICL), and 2) variation in the chosen prompt (for all the paradigms). Previous work focuses on 1) selecting a better combination of in-context examples (Su et al., 2023) for the model or 2) generating an optimal prompting template (Zhou et al., 2023c). On the other side, how the variation of multiple demonstration combinations and prompting templates influences the model behavior is still unexplored. Furthermore, we can 3) ‘self-ensemble’ the model based on different ensembling strategies. We now introduce all these variants.
# 4.1 Variation of Ensembling Components
Variation of In-Context Examples (Var-IC). For ICL and SICL, IC examples and their ordering [fp(xIC), yIC] create variations in the model inputs with a fixed template fp, while not impacting the test pair fp(x) ∼y. This allows us to create various in-context example combinations as different inputs to a single model and obtain different ensemble components. Variation of Prompting Templates (Var-Prompt). Different prompting templates have shown high variance in task performance (Mishra et al., 2022). By changing the wording in the templates fp, we can also create variations even with the same input to the model. For each input x, we randomly select a prompting template f′ p from a set of available prompting template candidates. In ICL and SICL, the same template is also applied to the in-context examples, formatting the final input as [f′ p(x1), y1; ...; f′ p(xM), yM, f′ p(x)]. This makes it applicable not only to ICL and SICL, but also to ZSL and SFT as well. Variation of Both (Var-Both). When we create a set of ensembling components, we can also combine these two variations.
# 4.2 Self-Ensembling Strategy
For each variant, we obtain the predicted results ˆy and the confidence ˆp for each component. The next step involves ensembling the predictions over K different components. We experiment with three (self-) ensembling strategies to compare their impact on both performance and calibration. Majority Vote. We select the predicted results that have the highest accumulated probability across K
variants as the ensembling predictions. The accumulated probability Pacc for the predicted label li is defined as follows:
(1)
We pick the variants that have the same prediction as the ensembling prediction and average the probability distribution of the selected components,
(2)
where K′ is the number of selected variants. Mean Probability. We average the predicted probability distribution of K variants and use the prediction that has the largest probability in the averaged distribution as the ensemble result ˆ yens:
(3) (4)
(3)
(4)
Max Probability. For each possible value in the output space, we find the maximum probability of the predicted values across K variants and use this as the prediction’s probability.
Because the probability is obtained from different components, the summation of these probabilities is not guaranteed to be 1. Therefore, we apply the normalization on the new probability distribution: Pens(y|x) = Norm(P′(y|x)). The ensemble prediction is determined as the ˆy that has the highest probability after the ensembling step.
# 4.3 Estimating Calibration
Beyond task performance of all the possible variants, we estimate the calibration of the model’s predictions (as a proxy towards model confidence) by using Expected Calibration Error (ECE) (Guo et al., 2017). It divides the n predicted results based on their confidence into M bins B1 to BM and then computes a weighted average over the absolute difference between the accuracy acc(Bm) and mean confidence conf(Bm) of the predictions within each bin. We set M to 10 in this work.
(5)
ECE measures the difference between the model’s empirical accuracy and its confidence (predicted probability). The smaller the ECE, the more confident the model prediction would be. We also report the negative log-likelihood (NLL) (Hastie et al., 2001) −�n i=1 log(pi) and information entropy (IE) −�n i=1 pilog(pi) as supplementary metrics of model’s (lack of) confidence.
# 5 Experimental Setup
Datasets and Evaluation Metrics. We consider 7 classification datasets that cover a range of label numbers and scenarios: SST-2, SST-5 (Socher et al., 2013), RTE (Wang et al., 2019), ANLI (Nie et al., 2020), Measuring Hate Speech corpus (Sachdeva et al., 2022), Intent Detection from NLU++ (Casanueva et al., 2022) and Manifestos (Lehmann et al., 2023). In order to simulate low-data setups, we sub-sample smaller training data from the full data for each dataset. The details of the datasets along with their corresponding evaluation metrics are provided in Appendix A.1. Implementation Details. Unless noted otherwise, we use Flan-T5large (Chung et al., 2022) as the main model in the experiments. Detailed training environments and hyper-parameters are provided in Appendix A.2 and A.3. Further, we provide all the prompting templates in Appendix D.1 and D.2. Regarding the self-ensembling experiments, for variations of in-context examples, we generate 20 different combinations of IC examples per each original training example. For variations of prompting templates, we manually create 4 different templates for each task. For Var-Both, we experiment with the same 4 templates and generate 4 × 5 variants with different combinations of IC examples.
# 6.1 Comparison between Learning Methods
We first present a comprehensive analysis of various learning methods in low-resource settings, detailing model performance and calibration errors in Table 1. For full experimental results, we refer the readers to Appendix B. Performance and calibration of learning methods are task-dependent. We find that learning methods perform differently depending on the datasets and we divide the tasks into different families depending on their observed behavior. ICL demonstrates comparable performance to SFT/SICL on SST-2
and RTE. However, tuning on these datasets with SFT/SICL yields increased ECE along with higher NLL and lower IE scores as shown in Table 2, but no substantial in-task performance improvement. This indicates that the model does not recover the ground truth distribution in the test set while becoming more confident and certain about its predictions, which serves as a sign of miscalibration. Conversely, tasks such as intent detection (NLU++), Manifestos, and Hate speech, show noticeable performance enhancement and better calibration with lower ECE by using SFT/SICL. Nevertheless, despite these task-dependent variations, ECE remains relatively high across all methods except for intent detection, indicating the problem of miscalibration across all learning methods. ICL can achieve comparable performance with SFT on ‘seen’ data. We suspect the divergent behaviors are possibly due to data contamination of FLAN training corpus (Longpre et al., 2023) wherein ZSL and ICL exhibit similar performances with SFT and SICL on training datasets labeled as seen (e.g., SST-2, RTE)2. To further investigate the performance on seen datasets, we apply the Batch Calibration method (Zhou et al., 2023a), as shown in Table 3. Surprisingly, we find that ICL performs on par or even better than SICL with calibration across all the (possibly) seen data, which reveals the ability of LLMs that have been recovered by calibration techniques on these seen tasks. However, for unseen datasets (NLU++, Manifestos, etc.), the performance of ICL is not comparable to either SICL or SFT even with calibration. Different learning methods excel with different task families. Given the comparison of the performances and calibration on different datasets, we suggest that the choice of learning methods should be task-dependent. The experiments and analysis indicate that unseen datasets obtain better performance and more trustworthy results with supervised tuning methods. For the seen datasets, ICL combined with other ‘tweaks’ such as model calibration can be a better choice, since the supervised tuning methods are more likely to make the model over-confident and less trustworthy. Within supervised tuning methods, for SFT and SICL, we empirically observe that SICL shows marginally higher performance (↑1.23) and lower 2We corroborate findings from other concurrent work on data contamination (Zhu et al., 2023; Deng et al., 2023) that also reports the unfair practice of evaluations on seen datasets.
2We corroborate findings from other concurrent work on data contamination (Zhu et al., 2023; Deng et al., 2023) that also reports the unfair practice of evaluations on seen datasets.
Metrics
Methods
SST-2
RTE
ANLI
SST-5
NLU++
Manifestos
Hate Speech
Performance
ZSL
94.67
86.64
52.30
42.00
29.20
14.50
37.08
ICL
95.220.12
88.450
52.170.47
37.590.23
40.110.09
13.010.19
40.090.08
SFT
95.610.20
88.810.29
61.631.68
46.272.09
79.980.59
35.761.23
58.011.01
SICL
95.630.29
88.570.45
63.900.14
47.121.93
80.760.31
37.551.61
59.481.79
ECE
ZSL
0.907
0.809
0.356
0.142
0.231
0.432
0.318
ICL
0.9150.001
0.8150.003
0.3510.005
0.1830.002
0.1290.000
0.4760.002
0.2710.002
SFT
0.9410.011
0.8420.002
0.3160.023
0.4030.032
0.0110.001
0.2010.072
0.3540.036
SICL
0.9450.013
0.8760.003
0.2800.011
0.3600.025
0.0020.001
0.2140.038
0.1930.113
Table 1: Results for different learning methods across all 7 datasets. We report the average of 3 independent runs with different random seeds; variance is reported in the subscript. Numbers in bold represent the best performance and calibration score per dataset. The datasets ‘seen’ by Flan-T5 at pretraining are labeled in italic.
Evaluation Metrics
SST-2
SST-5
ICL
SICL
ICL
SICL
Calibration
ECE
0.915
0.945
0.183
0.360
NLL
0.135
0.271
1.226
2.339
IE
0.056
0.015
0.152
0.049
Evaluation Metrics
NLU++
Manifestos
ICL
SICL
ICL
SICL
Calibration
ECE
0.129
0.002
0.476
0.214
NLL
0.214
0.084
3.942
2.026
IE
0.142
0.002
0.101
0.145
Table 2: Calibration errors and other uncertainty metrics of different learning methods across tasks (part of). We refer the readers to Appendix B.1 for full results.
Evaluation Metrics
SST-2
SST5
ICL
SICL
ICL
SICL
Performance
acc
95.22
95.63
50.48
54.25
macro f1
-
-
37.59
47.12
acc
95.95
95.63
50.89
49.91
+ calibrated
macro f1
-
-
49.80
49.89
Evaluation Metrics
Manifestos
Hate speech
ICL
SICL
ICL
SICL
Performance
micro f1
19.29
38.12
40.18
61.33
macro f1
13.01
37.55
40.09
59.48
micro f1
31.00
38.33
45.11
59.58
+ calibrated
macro f1
29.15
37.83
42.36
57.81
Table 3: A selection of results after applying Batch Calibration. We refer the readers to Appendix B.1 for the full set of results with the calibration method.
Table 3: A selection of results after applying Batch Calibration. We refer the readers to Appendix B.1 for the full set of results with the calibration method.
ECE (↓0.05) than SFT on average across unseen datasets in Table 1. We believe this is possibly due to 1) the knowledge in the IC examples in addition to the training input-label pairs and 2) different combinations of in-context examples as a way of data augmentation in low-resource scenarios.
# 6.2 Self-Ensembling Results
So far, we have observed the common miscalibration issues for all learning methods. We then investigate the feasibility of self-ensembling to improve calibration.
Self-ensembling works across learning methods and enhances calibration performance. In Table 4, with different learning methods combined with self-ensembling variations in designs, we find that by changing the in-context example combinations or prompting templates, the best performance of self-ensembling outperforms the baseline without any ensembling by 0.79. Even though the performance gains seem marginal, self-ensembling substantially enhances the calibration performance, reducing the mean ECE value by 43%. The empirical results also show that in addition to ICL, SFT, and SICL also benefit from self-ensembling in both performance and calibration scores. SFT and SICL exhibit a larger drop in ECE after selfensembling than ICL. We also notice that when self-ensembling over SFT and SICL, the model has lower ECE scores than ICL, but with much better performances. This indicates the efficiency of selfensembling in making the predictions more trustworthy while maintaining or even improving the performances. It also indicates that self-ensembling has the potential to mitigate the prominent problem of overconfidence in supervised tuning methods, as shown by Figure 2.
Different Variations and Ensembling Strategies. Our results suggest that with ICL, Var-IC yields more improvements than Var-Prompt, while the latter shows its efficiency with SFT and SICL. This difference stems from inappropriate prompting templates, requiring model tuning to adhere to the prompt structure. We also find that combining both variations may not necessarily improve the performance but is helpful in enhancing the trustworthiness empirically. Regarding ensemble strategies, we notice that the majority vote improves the performances but struggles to reduce the calibration error. Ensembling with max probability consistently
Systems
Manifestos
Macro F1
ECE
Ori.
Max
Mean
Majority
∆
Ori.
Max
Mean
Majority
∆
ZSL
14.50
↑0.79
0.432
↓0.170
+ Var-Prompt
14.50
13.69
12.93
15.29
↑0.79
0.432
0.262
0.335
0.437
↓0.170
ICL
13.01
↑0.68
0.476
↓0.283
+ Var-IC
13.01
13.50
13.46
13.69
↑0.68
0.476
0.415
0.465
0.469
↓0.061
+ Var-Prompt
13.01
13.25
11.67
11.53
↑0.24
0.476
0.268
0.366
0.472
↓0.208
+ Var-Both
13.01
11.19
11.50
11.21
↓1.51
0.476
0.193
0.354
0.480
↓0.283
FT
35.76
↑0.73
0.201
↓0.134
+ Var-Prompt
35.39
36.49
35.66
34.91
↑1.10
0.144
0.066
0.105
0.135
↓0.077
SupICL
37.55
↑0.06
0.214
↓0.090
+ Var-IC
37.55
36.57
37.35
37.61
↑0.06
0.214
0.179
0.210
0.215
↓0.035
+ Var-Prompt
37.04
37.14
37.25
36.77
↑0.21
0.229
0.139
0.191
0.219
↓0.090
+ Var-Both
37.04
36.67
37.15
37.50
↑0.46
0.229
0.124
0.192
0.230
↓0.105
Systems
Hate Speech
Macro F1
ECE
Ori.
Max
Mean
Majority
∆
Ori.
Max
Mean
Majority
∆
ZSL
37.08
↓0.13
0.318
↓0.049
+ Var-Prompt
37.08
36.54
36.95
36.95
↓0.13
0.318
0.269
0.302
0.320
↓0.049
ICL
40.09
↑1.10
0.271
↓0.111
+ Var-IC
40.09
40.01
39.98
40.49
↑0.40
0.271
0.233
0.267
0.269
↓0.038
+ Var-Prompt
40.09
41.03
41.19
41.05
↑1.10
0.271
0.194
0.236
0.275
↓0.077
+ Var-Both
40.09
39.68
40.30
40.49
↑0.40
0.271
0.160
0.237
0.278
↓0.111
FT
58.01
↓0.82
0.354
↓0.115
+ Var-Prompt
55.92
57.16
57.17
57.19
↑1.27
0.350
0.239
0.290
0.345
↓0.111
SupICL
59.48
↑0.74
0.193
↓0.078
+ Var-IC
59.48
59.98
59.82
59.83
↑0.50
0.193
0.141
0.179
0.191
↓0.052
+ Var-Prompt
58.66
59.96
60.10
59.86
↑1.44
0.251
0.165
0.211
0.246
↓0.086
+ Var-Both
58.66
60.12
60.22
59.97
↑1.56
0.251
0.115
0.206
0.246
↓0.136
Systems
Macro F1
ECE
Ori.
Max
Mean
Majority
∆
Ori.
Max
Mean
Majority
∆
ZSL
37.08
↓0.13
0.318
↓0.049
+ Var-Prompt
37.08
36.54
36.95
36.95
↓0.13
0.318
0.269
0.302
0.320
↓0.049
ICL
40.09
↑1.10
0.271
↓0.111
+ Var-IC
40.09
40.01
39.98
40.49
↑0.40
0.271
0.233
0.267
0.269
↓0.038
+ Var-Prompt
40.09
41.03
41.19
41.05
↑1.10
0.271
0.194
0.236
0.275
↓0.077
+ Var-Both
40.09
39.68
40.30
40.49
↑0.40
0.271
0.160
0.237
0.278
↓0.111
FT
58.01
↓0.82
0.354
↓0.115
+ Var-Prompt
55.92
57.16
57.17
57.19
↑1.27
0.350
0.239
0.290
0.345
↓0.111
SupICL
59.48
↑0.74
0.193
↓0.078
+ Var-IC
59.48
59.98
59.82
59.83
↑0.50
0.193
0.141
0.179
0.191
↓0.052
+ Var-Prompt
58.66
59.96
60.10
59.86
↑1.44
0.251
0.165
0.211
0.246
↓0.086
+ Var-Both
58.66
60.12
60.22
59.97
↑1.56
0.251
0.115
0.206
0.246
↓0.136
Table 4: Results of self-ensembling with different variations (selection). We mark the cells of baseline system without self-ensembling and their results in grey. Numbers in bold represents the best values for each learnin method. ∆calculates the difference of performance and calibration error between the original results (Ori.) and th best self-ensembled results, where green denotes better results and red denotes worse results. We refer the reader to Appendix B.2 for full self-ensembling results.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8084/8084cc93-8060-4539-9720-0a33d62aa9a4.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) E-SICL(Var-Both)</div>
Figure 2: The confidence histograms and reliability diagrams of SICL and self-ensembling with max probability on SST5. E- means the self-ensembled systems. produces the most faithful predictions with promising performances. This can be explained by the
Figure 2: The confidence histograms and reliability diagrams of SICL and self-ensembling with max probability on SST5. E- means the self-ensembled systems.
produces the most faithful predictions with promising performances. This can be explained by the
idea that normalizing over the max probabilities in a way smooths the probability distribution, making the max probabilities less extreme and mitigating over-confidence issues.
# 6.3 Ablation Studies
More ensembling components with Var-IC lead to better calibration. We explore how varying the number of ensembling components with different in-context examples affects performance and calibration. Figure 3 shows the performance and ECE scores with different components. We observe that although the performances remain comparable, the calibration is improved with more components in both ICL and SICL. This highlights the effectiveness of the self-ensembling method in making the predictions more trustworthy through increased input variations. Diversity in Var-Prompt influences self-ensembling.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a7dd/a7dd27f5-384a-4296-aedd-51b14006c8d3.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Performance and calibration errors of different number of components with variations in in-context examples on Manifestos.</div>
Figure 3: Performance and calibration errors of different number of components with variations in in-context examples on Manifestos.
<div style="text-align: center;">Figure 3: Performance and calibration errors of different number of components with variations in in-context examples on Manifestos.</div>
Hate speech
Metrics
Num
SFT
E-SFT
SICL
E-SICL
Performance
1
58.01
58.01
59.48
59.48
4
55.92
57.16
58.66
59.96
8
56.00
56.60
58.37
59.61
ECE
1
0.354
0.354
0.193
0.193
4
0.350
0.239
0.251
0.165
8
0.335
0.208
0.403
0.200
SST-5
Metrics
Num
SFT
E-SFT
SICL
E-SICL
Performance
1
46.27
46.27
47.12
47.12
4
48.11
47.43
47.99
47.70
8
47.55
47.88
45.75
45.12
ECE
1
0.403
0.403
0.360
0.360
4
0.396
0.301
0.271
0.170
8
0.362
0.244
0.403
0.272
Table 5: Results with different numbers of prompting templates on Hate speech and SST-5.
Table 5: Results with different numbers of prompting templates on Hate speech and SST-5.
In Table 5, we show that when tuning with more templates, SFT has lower calibration errors whereas those of SICL increase. Regarding self-ensembling results, we empirically find that by introducing more prompting templates, selfensembling yields lower calibration error with SFT, but shows worse calibration with SICL, meanwhile yielding similar performances. Nonetheless, selfensembling consistently improves the performance and calibration within each setting. The key findings are robust across different training data sizes. We experiment with larger training data and report the results of SFT and SICL on SST-5 in Figure 4. We observe that both SFT and SICL yield better performance and lower ECE with more training data, whereas ICL maintains similar performance. This indicates that supervised methods can better calibrate the model, and produce more calibrated predictions if provided with sufficient data. We also find that self-ensembling remains effective in improving the performance and mitigating the calibration error of the model on average. The findings hold across different model sizes. In order to assess the impact of model sizes, we fur-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1920/1920411d-0275-4c10-94b2-3362585983e6.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Performance and calibration errors on SST-5 with different numbers of training data.</div>
ther conduct experiments with Flan-T5xl. For full detailed experiment results, we refer the readers to Table 12 and 13 in the Appendix. We find that a larger model achieves a better calibration score with SFT and SICL than ZSL and ICL. We also witness similar behavior of ICL, surpassing SFT and SICL with or without Batch Calibration on seen data, which aligns with previous findings. The results on SST-5 and Hate Speech show that by applying the self-ensembling method, Flan-T5xl achieves better performance and lower calibration scores, indicating that the model becomes more ‘task-specialized’. It is also worth noticing that our method is able to improve the performance and decrease the calibration scores on some tasks (Hate Speech) where traditional calibration currently fails. In general, this indicates the potential of self-ensembling in improving both task performance and calibration.
# 7 Conclusion
We have provided a comprehensive analysis of the intricate relationship between in-task performance and calibration across various learning methods in low-resource scenarios. Our findings illuminate the nuances of in-task performance and calibration across different task families, meanwhile addressing the inherent miscalibration over all learning methods. We have also investigated effective strategies to enhance both aspects simultaneously, offering a viable solution through self-ensembling: it results in more calibrated predictions with comparable or superior task performance. We hope that this study will contribute valuable insights into the dynamic landscape of LLMs. These discoveries also offer practical guidance to practitioners, aiding them in choosing suitable learning paradigms and paving the way for the development of more reliable and high-performing LLMs across diverse applications.
# Limitations
Our experimental results are conducted with the Flan-T5 model family, which is an encoder-decoder architecture, where we have not investigated how the behaviours of other popular choices of decoderonly models, such as Llama (Touvron et al., 2023), would behave in low-resource scenarios with different learning methods and self-ensembling strategies. Secondly, limited by the training resources, our experiments only consider LLMs within 3B parameters. We will endeavor to scale our experiments to cover larger language models as part of future work. Moreover, there is a variety of sophisticated ensembling methods (Mohammed and Kora, 2023), where we have only studied the max, mean, and majority vote variants for self-ensembling. In future work, we aim to extend the analysis and the self-ensembling methods to more families of tasks, diverse types of models, and other ensembling techniques.
# References
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901. Inigo Casanueva, Ivan Vuli´c, Georgios Spithourakis, and Paweł Budzianowski. 2022. NLU++: A multilabel, slot-rich, generalisable dataset for natural language understanding in task-oriented dialogue. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 1998–2013, Seattle, United States. Association for Computational Linguistics. Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, and He He. 2022. Meta-learning via language model in-context tuning. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 719–730, Dublin, Ireland. Association for Computational Linguistics. Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam M. Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Benton C. Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier García,
akanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam M. Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Benton C. Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier García,
Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Díaz, Orhan Firat, Michele Catasta, Jason Wei, Kathleen S. Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022. Palm: Scaling language modeling with pathways. J. Mach. Learn. Res., 24:240:1–240:113. Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416. Chunyuan Deng, Yilun Zhao, Xiangru Tang, Mark Gerstein, and Arman Cohan. 2023. Investigating data contamination in modern benchmarks for large language models. arXiv preprint arXiv:2311.09783. Shrey Desai and Greg Durrett. 2020. Calibration of pre-trained transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 295–302, Online. Association for Computational Linguistics. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2022. A survey for in-context learning. arXiv preprint arXiv:2301.00234. Hanyu Duan, Yixuan Tang, Yi Yang, Ahmed Abbasi, and Kar Yan Tam. 2023. Exploring the relationship between in-context learning and instruction tuning. arXiv preprint arXiv:2311.10367. Yu Fei, Yifan Hou, Zeming Chen, and Antoine Bosselut. 2023. Mitigating label biases for in-context learning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14014–14031, Toronto, Canada. Association for Computational Linguistics. Adam Gleave and Geoffrey Irving. 2022. Uncertainty estimation for language reward models. arXiv preprint arXiv:2203.07472. Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. 2023. Pre-training to learn in context. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages
Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Díaz, Orhan Firat, Michele Catasta, Jason Wei, Kathleen S. Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022. Palm: Scaling language modeling with pathways. J. Mach. Learn. Res., 24:240:1–240:113.
Adam Gleave and Geoffrey Irving. 2022. Uncertainty estimation for language reward models. arXiv preprint arXiv:2203.07472.
Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. 2023. Pre-training to learn in context. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. 2017. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR. Trevor Hastie, Robert Tibshirani, and Jerome Friedman. 2001. The Elements of Statistical Learning. Springer Series in Statistics. Springer New York Inc., New York, NY, USA. Junxian He, Chunting Zhou, Xuezhe Ma, Taylor BergKirkpatrick, and Graham Neubig. 2022. Towards a unified view of parameter-efficient transfer learning. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 2529, 2022. OpenReview.net. Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net. Zhengbao Jiang, Jun Araki, Haibo Ding, and Graham Neubig. 2021. How can we know when language models know? on the calibration of language models for question answering. Transactions of the Association for Computational Linguistics, 9:962–977. Lingkai Kong, Haoming Jiang, Yuchen Zhuang, Jie Lyu, Tuo Zhao, and Chao Zhang. 2020. Calibrated language model fine-tuning for in- and outof-distribution data. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1326–1340, Online. Association for Computational Linguistics. Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. 2017. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30. Pola Lehmann, Simon Franzmann, Tobias Burst, Sven Regel, Felicia Riethmüller, Andrea Volkens, Bernhard Weßels, and Lisa Zehnter. 2023. The manifesto data collection. manifesto project (mrg/cmp/marpor). version 2023a. Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V. Le, Barret Zoph, Jason Wei, and Adam Roberts. 2023. The flan collection: Designing data and methods for effective instruction tuning. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 22631–22648. PMLR. Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2022. MetaICL: Learning to learn in context. In Proceedings of the 2022 Conference of
Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2022. MetaICL: Learning to learn in context. In Proceedings of the 2022 Conference of
Computational Linguistics: Human Language Technologies, pages 2791–2809, Seattle, United States. Association for Computational Linguistics. Swaroop Mishra, Daniel Khashabi, Chitta Baral, Yejin Choi, and Hannaneh Hajishirzi. 2022. Reframing instructional prompts to GPTk’s language. In Findings of the Association for Computational Linguistics: ACL 2022, pages 589–612, Dublin, Ireland. Association for Computational Linguistics. Ammar Mohammed and Rania Kora. 2023. A comprehensive review on ensemble deep learning: Opportunities and challenges. Journal of King Saud University - Computer and Information Sciences, 35(2):757– 774. Marius Mosbach, Tiago Pimentel, Shauli Ravfogel, Dietrich Klakow, and Yanai Elazar. 2023. Few-shot fine-tuning vs. in-context learning: A fair comparison and evaluation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 12284– 12314, Toronto, Canada. Association for Computational Linguistics. Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. 2020. Adversarial NLI: A new benchmark for natural language understanding. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4885–4901, Online. Association for Computational Linguistics. OpenAI. 2023. Gpt-4 technical report. ArXiv, abs/2303.08774. Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. 2019. Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift. Advances in neural information processing systems, 32. Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551. Evgeniia Razumovskaia, Goran Glavaš, Anna Korhonen, and Ivan Vuli´c. 2023. Sqatin: Supervised instruction tuning meets question answering for improved dialogue nlu. arXiv preprint arXiv:2311.09502. Pratik Sachdeva, Renata Barreto, Geoff Bacon, Alexander Sahn, Claudia von Vacano, and Chris Kennedy. 2022. The measuring hate speech corpus: Leveraging rasch measurement theory for data perspectivism. In Proceedings of the 1st Workshop on Perspectivist
Approaches to NLP @LREC2022, pages 83–94, Marseille, France. European Language Resources Association.
Approaches to NLP @LREC2022, pages 83–94, Marseille, France. European Language Resources Association.
ciation. Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Victoria Lin, Noah A Smith, Luke Zettlemoyer, Scott Yih, and Mike Lewis. 2023. Incontext pretraining: Language modeling beyond document boundaries. arXiv preprint arXiv:2310.10638. Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. 2022. Large language models encode clinical knowledge. arXiv preprint arXiv:2212.13138. Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA. Association for Computational Linguistics. Hongjin Su, Jungo Kasai, Chen Henry Wu, Weijia Shi, Tianlu Wang, Jiayi Xin, Rui Zhang, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Tao Yu. 2023. Selective annotation makes language models better few-shot learners. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. Meiqi Sun, Wilson Yan, Pieter Abbeel, and Igor Mordatch. 2022. Quantifying uncertainty in foundation models via ensembles. In NeurIPS 2022 Workshop on Robustness in Sequence Modeling. Simeng Sun, Yang Liu, Dan Iter, Chenguang Zhu, and Mohit Iyyer. 2023. How does in-context learning help prompt tuning? arXiv preprint arXiv:2302.11521. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971. Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In the Proceedings of ICLR. Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. 2023a. Large language models are not fair evaluators. arXiv preprint arXiv:2305.17926. Xi Wang, Laurence Aitchison, and Maja Rudolph. 2023b. Lora ensembles for large language model fine-tuning. arXiv preprint arXiv:2310.00035.
Jerry Wei, Le Hou, Andrew Lampinen, Xiangning Chen, Da Huang, Yi Tay, Xinyun Chen, Yifeng Lu, Denny Zhou, Tengyu Ma, and Quoc Le. 2023. Symbol tuning improves in-context learning in language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 968–979, Singapore. Association for Computational Linguistics. Florian Wenzel, Jasper Snoek, Dustin Tran, and Rodolphe Jenatton. 2020. Hyperparameter ensembles for robustness and uncertainty quantification. Advances in Neural Information Processing Systems, 33:6514–6527. Bingsheng Yao, Guiming Chen, Ruishi Zou, Yuxuan Lu, Jiachen Li, Shao Zhang, Sijia Liu, James Hendler, and Dakuo Wang. 2023. More samples or more prompt inputs? exploring effective in-context sampling for llm few-shot prompt engineering. arXiv preprint arXiv:2311.09782. Qinyuan Ye, Iz Beltagy, Matthew Peters, Xiang Ren, and Hannaneh Hajishirzi. 2023a. FiD-ICL: A fusionin-decoder approach for efficient in-context learning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8158–8185, Toronto, Canada. Association for Computational Linguistics. Seonghyeon Ye, Hyeonbin Hwang, Sohee Yang, Hyeongu Yun, Yireun Kim, and Minjoon Seo. 2023b. In-context instruction learning. arXiv preprint arXiv:2302.14691. Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In International Conference on Machine Learning, pages 12697–12706. PMLR. Han Zhou, Xingchen Wan, Lev Proleev, Diana Mincu, Jilin Chen, Katherine Heller, and Subhrajit Roy. 2023a. Batch calibration: Rethinking calibration for in-context learning and prompt engineering. arXiv preprint arXiv:2309.17249. Han Zhou, Xingchen Wan, Ivan Vuli´c, and Anna Korhonen. 2023b. Autopeft: Automatic configuration search for parameter-efficient fine-tuning. arXiv preprint arXiv:2301.12132. Han Zhou, Xingchen Wan, Ivan Vuli´c, and Anna Korhonen. 2023c. Survival of the most influential prompts: Efficient black-box prompt search via clustering and pruning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 13064– 13077, Singapore. Association for Computational Linguistics. Wenhong Zhu, Hongkun Hao, Zhiwei He, Yunze Song, Yumeng Zhang, Hanxu Hu, Yiran Wei, Rui Wang, and Hongyuan Lu. 2023. Clean-eval: Clean evaluation on contaminated large language models. arXiv preprint arXiv:2311.09154.
# A Experiment Setup
# A.1 Dataset Details
SST-2. The SST-2 dataset, a widely-used benchmark in sentiment analysis, comprises sentences from movie reviews annotated with binary sentiment labels (positive or negative). We train the model with the data randomly sampled from the original training set and report the performance on the test set. We evaluate the model’s performance based on accuracy. SST-5. SST-5, an extension of SST-2, enhances sentiment analysis with five classes: very negative, negative, neutral, positive, and very positive. Derived from movie reviews, this dataset provides a nuanced perspective on sentiment, allowing models to distinguish fine-grained emotional tones. With all other practices aligned with SST-2, the results are evaluated with micro f1 and macro f1 scores because it has more than 2 labels. RTE. Recognizing Textual Entailment is a benchmark dataset assessing the task of determining logical entailment between pairs of text snippets. Annotated with binary labels indicating entailment or not, RTE is crucial for evaluating models’ logical reasoning abilities. We report the accuracy in accordance with other binary classification tasks. ANLI. Adversarial NLI is a benchmark dataset introducing adversarial examples to challenge models with nuanced reasoning and complex inferences. With labeled sentence pairs denoting entailment, contradiction, or neutrality, ANLI is crucial for assessing models’ robustness and generalization in the face of diverse linguistic challenges. ANLI has three different rounds of contexts, with later rounds having a better base model, thus being more difficult for the model to distinguish. In this paper, we conduct the experiments mainly on the first round, which is easier than other rounds, in order to compare the performance with ICL. Since it is a multiclass classification task, we report the performance with micro and macro F1 scores. In this paper, we mainly use r1 level data for experiments. NLU++. NLU++ is a more challenging benchmark in task-oriented dialogue system with more finegrained domain ontologies and sentences with multiple intents. It has two tasks: intent detection and slot labeling, covering the banking and hotels two domains. In this work, we focus on the intent detection task, which is a multi-label classification task
Dataset
Label number
Main Metric
Train size
SST-2
2
acc
50
RTE
2
acc
50
ANLI
3
acc
50
SST5
5
macro f1
50
NLU++
2
micro f1
50
Manifestos
8
macro f1
800
Hate speech
3
macro f1
50
Table 6: Summary of the datasets, main evaluation metric for performance and training data size used for experiment.
and we follow the setting from recent work with state-of-the-art results (Razumovskaia et al., 2023), which formats it as a binary yes/no classification task. See the cited work for further details. Regarding the data split, 1,000 sentences from NLU++ were held out for testing and 50 sentences from the leftover 2k+ sentences were sub-sampled for training.
and we follow the setting from recent work with state-of-the-art results (Razumovskaia et al., 2023), which formats it as a binary yes/no classification task. See the cited work for further details. Regarding the data split, 1,000 sentences from NLU++ were held out for testing and 50 sentences from the leftover 2k+ sentences were sub-sampled for training. Manifestos. Manifestos was originally created to collect the manifestos of parties from different countries. It also includes the analytical variables that indicates the respective categories of the quasisentences. The corpus have 8 domains overall, which are listed as follows: None (of the below) / Other, External Relations, Freedom and Democracy, Political System, Economy, Welfare and Quality of Life, Fabric of Society, Social Groups. In this paper, we use the sentences that only have one golden domain and exclude the ones with multiple labels. Measuring Hate Speech Corpus. Measuring Hate Speech Corpus, in short Hate speech, contains1 10 constituent ordinal labels and the continuous hate speech score to measure the extent of hate. We use the hate speech score as indicator of hate speech in this paper. We follow the original division of approximate hate speech provided by the authors, where > 0.5 is approximately hate speech, < -1 is counter or supportive speech, and -1 to 0.5 is neutral or ambiguous. We only experiment on the intent detection task in the NLU++ bank domain and for ANLI we mainly discuss r1 level data. We summarize the training data size, main performance evaluation metrics, and the number of labels for each dataset in Table 6. We also list the label verbalizers for all datasets in Table 7.
Task
Label Verbalizer
SST2
postive, negative
RTE
yes, no
ANLI
yes, maybe, no
SST5
terrible, bad, neutral, good, great
NLU++
yes,no
Manifestos
other, external, democracy, political,
economy, welfare, fabric, group
Hate Speech
support, neutral, hate
Table 7: Label verbalizer for different tasks.
<div style="text-align: center;">Table 7: Label verbalizer for different tasks.</div>
# A.2 Environment Setup
We mainly use Flan-T5large (783M parameters) as the task models for all the datasets. We also use Flan-T5xl (2.85B parameters) on some of the task to see whether the findings still hold on the larger model. For SFT and SICL, we use LoRA (Hu et al., 2022) to tune Flan-T5xl. Due to the computational limitations, we can’t obtain the results on all the datasets with Flan-T5xl. All the experiments are conducted on Cambridge High-Performance Clusters with a single A100 (80G) and a 32-core vCPU. We release the code and the environment dependencies for reproducible purposes at https://github.com/ cambridgeltl/ensembled-sicl.
# A.3 Hyperparameters
In order to evaluate the model’s performance and trustworthiness in low-resource scenarios, we sample a subset of the training set and evaluate it on a fixed set of data as an evaluation and test set. For Manifestos, because it has 8 classes and is more expertise in specialized domains (politics, economics and etc.), we use a relatively larger training set to adapt the model to the task itself. For Hate Speech, we manually sample the training set and test set ourselves since the corpus didn’t provide the split. We randomly sample 1500 data as the fixed test set and 500 examples as the fixed evaluation set. All the main experiments are conducted three times with 0, 21, 42 as the random seeds. We report the mean values of three runs in the main content. Across different learning paradigms (ICL and SICL), we concatenate 3 in-context examples in front of the input for the main experiments. For supervised fine-tuning methods, we attach the detailed hyperparameters in Table 8 for reproducibility. Because tuning the model in the lowresource setting is prone to over-confidence, in or-
der to mitigate the problem, we apply the early stopping with the patience of 5. Regarding the configuration hyper-parameters of PEFT, they are listed in Table 9. Unlisted properties use the default values in PEFT implementation from huggingface3.
# B Full Experiment Results
To solidify the empirical findings, in this section, we present full experiment results with more metrics in addition to the table in the main content for readers’ reference.
# B.1 Results of different learning methods
Table 10 shows the full results on all 7 datasets. We report the accuracy, micro f1, and macro f1 as the performance metrics. We report ECE as the measurement of calibration. We also include NLL and IE as supplementary uncertainty metrics. We find that on SST-2 and RTE, the model achieves comparable or even better performance with ZSL and ICL than SFT and SICL. In the meantime, the predictions have a relatively high ECE, indicating that on these two datasets, the model has the issue of miscalibration. On ANLI, Manifestos, Hate speech, and NLU++, with SFT and SICL the model has lower calibration error than ICL and achieves better performance in both performance metrics. In addition to the original results, we include the Batch Calibration results across all the datasets. On SST-5 and ANLI, although ZSL and ICL achieve similar micro f1 scores, there is still a gap in the original macro f1 scores between ZSL/ICL and SFT/SICL. However, after applying Batch Calibration, we find that ZSL/ICL has a comparable macro f1 score to SFT/SICL. On Manifestos, Hate speech, and NLU++, we don’t observe comparable performance between ZSL/ICL and SFT/SICL either with or without Batch Calibration.
# B.2 Results of self-ensembling
Table 11 shows the self-ensembling results across 4 datasets. We exclude the seen datasets (SST-2 and RTE) for fair evaluation, as well as NLU++ since it’s almost well-calibrated with supervised tuning. We still include ANLI for comparison even though it is included during pre-training. We report the mean values of the results with 3 different random seeds (0, 21, 42).
3https://huggingface.co/docs/peft/index
Hyperparameters
ICL
FT
SupICL
SST2
train batch size
-
8
8
eval batch size
64
32
32
grad accumulation
-
1
1
learning rate
-
5e-5
5e-5
evaluation per steps
-
10
10
max training epochs
-
200
200
early stopping patience
-
5
5
early stopping metric
-
accuracy
accuracy
RTE
train batch size
-
8
4
eval batch size
64
32
32
grad accumulation
-
1
2
learning rate
-
5e-5
5e-5
evaluation per steps
-
10
10
max training epochs
-
200
200
early stopping patience
-
5
5
early stopping metric
-
accuracy
accuracy
ANLI
train batch size
-
8
4
eval batch size
64
32
32
grad accumulation
-
1
2
learning rate
-
5e-5
5e-5
evaluation per steps
-
10
10
max training epochs
-
200
200
early stopping patience
-
5
5
early stopping metric
-
accuracy
accuracy
SST5
train batch size
-
8
8
eval batch size
64
32
32
grad accumulation
-
1
1
learning rate
-
5e-5
5e-5
evaluation per steps
-
10
10
max training epochs
-
200
200
early stopping patience
-
5
5
early stopping metric
-
macro f1
macro f1
NLU++
train batch size
-
16
16
eval batch size
64
32
32
grad accumulation
-
2
2
learning rate
-
5e-5
5e-5
evaluation per steps
-
500
500
max training epochs
-
200
200
early stopping patience
-
5
5
early stopping metric
-
micro f1
micro f1
Manifestos
train batch size
-
8
4
eval batch size
32
32
32
grad accumulation
-
1
2
learning rate
-
5e-5
5e-5
evaluation per steps
-
10
10
max training epochs
-
200
200
early stopping patience
-
5
5
early stopping metric
-
macro f1
macro f1
Hate speech
train batch size
-
8
4
eval batch size
32
32
32
grad accumulation
-
1
2
learning rate
-
5e-5
5e-5
evaluation per steps
-
10
10
max training epochs
-
200
200
early stopping patience
-
5
5
early stopping metric
-
macro f1
macro f1
Table 8: Hyper-parameters for each dataset when com-
paring different learning methods.
Table 8: Hyper-parameters for each dataset when comparing different learning methods.
Hyperparameters
Values
r
8
lora alpha
32
lora dropout
0.05
target modules
q, v
Table 9: Hyper-parameters for PEFT with FlanT5-xl.
From the perspective of performance, we find that on SST-5, Manifestos, and Hate speech, selfensembled results achieve slightly better performances on average and show positive improvements with each learning method. On ANLI, we observe no significant improvement in the accuracy of self-ensembled results and the decreases in performance are trivial as well. However, from the perspective of calibration, we find that selfensembling with max probability consistently decreases the calibration over all settings, as shown in Figure 5. Introducing variations in both in-context examples and prompting templates yields the lowest calibration error in all experiments. Among different ensembling methods, we find that majority vote can achieve better performances sometimes but it doesn’t help to reduce the calibration error or even make it worse. Mean probability and max probability are able to improve the performance meanwhile reducing the calibration error. The empirical experiment results suggest that although majority vote as a widely used ensemble method achieves better performance, it is worth noting that it may deliver unfaithful predictions, which is not preferred in real application.
# C Supplementary Results for Ablations
# C Supplementary Results for Ablations
# C.1 How about larger models?
Table 12 shows the results on SST-5 and Hate speech with different learning methods using FlanT5xl. With ZSL and ICL, we observe that xl version model has larger calibration errors than Flan-T5large model on possibly seen datasets (SST-2 and SST5), whereas on unseen datasets (Hate speech and Manifestos) it shows lower ECE. Regarding the performances, the xl model shows better performances on unseen datasets than the large version model but doesn’t guarantee better performances on seen datasets. After tuning the model with SFT or SICL, we find that the calibration errors are reduced across all tasks, which is different from Flan-T5large. Due to the computation constraint, we leave the discrepancy in the behaviors of different-
Evaluation Metrics
SST2
RTE
ZSL
ICL
SFT
SICL
ZSL
ICL
SFT
SICL
Performance
acc
94.67
95.220.12
95.610.20
95.630.29
86.64
88.450
88.810.29
88.570.45
macro f1
-
-
-
-
-
-
-
-
acc
95.50
95.950.19
95.610.20
95.630.29
89.53
90.250.51
89.170.29
87.970.45
+ calibrated
macro f1
-
-
-
-
-
-
-
-
Trustworthiness
ECE
0.9069
0.91490.0014
0.94080.0113
0.94490.0129
0.8092
0.81500.0030
0.84180.0224
0.87590.0025
NLL
0.1465
0.13470.0006
0.21460.0962
0.27100.1396
0.3438
0.29100.0037
0.44210.2133
1.12340.1133
IE
0.0615
0.05990.0003
0.02150.0149
0.01540.0124
0.0977
0.09750.0002
0.06300.0304
0.01530.0030
+ calibrated
ECE
0.7877
0.79610.0016
0.80400.0056
0.80640.0096
0.6831
0.69910.0054
0.70220.0061
0.71110.0045
Evaluation Metrics
SST5
ANLI
ZSL
ICL
SFT
SICL
ZSL
ICL
SFT
SICL
Performance
micro f1
52.58
50.480.15
50.591.38
54.250.46
52.30
52.170.47
61.631.68
63.900.14
macro f1
42.00
37.590.23
46.272.09
47.120.0193
42.07
42.060.39
61.171.76
63.530.43
micro f1
50.05
50.890.86
50.350.50
49.910.21
62.30
61.270.82
62.471.60
64.500.41
+ calibrated
macro f1
48.98
49.800.91
50.430.68
49.890.26
61.98
61.150.83
62.361.67
64.460.40
Trustworthiness
ECE
0.1416
0.18330.0020
0.40300.0321
0.36020.0250
0.3555
0.35110.0050
0.31610.0230
0.28030.0108
NLL
1.1762
1.22610.0021
3.19861.4010
2.33900.2827
4.7484
3.89610.0233
2.25990.3255
1.92260.2912
IE
0.1599
0.15220.0001
0.04540.0209
0.04890.0107
0.0945
0.09870.0001
0.05870.0070
0.06630.0113
+ calibrated
ECE
0.1010
0.10270.0084
0.07200.0129
0.04560.0035
0.0709
0.04300.0079
0.06260.0145
0.04950.0085
Evaluation Metrics
Manifestos
Hate speech
ZSL
ICL
SFT
SICL
ZSL
ICL
SFT
SICL
Performance
micro f1
20.87
19.290.16
37.541.10
38.122.01
39.67
40.180.14
59.670.47
61.332.05
macro f1
14.50
13.010.19
35.761.23
37.551.61
37.08
40.090.08
58.011.01
59.481.79
micro f1
33.63
31.000.37
38.580.46
38.330.33
43.87
45.110.46
59.890.17
59.583.26
+ calibrated
macro f1
30.86
29.150.53
37.571.05
37.830.41
40.86
42.360.42
58.060.27
57.813.03
Trustworthiness
ECE
0.4319
0.47600.0018
0.20050.0723
0.21380.0376
0.3175
0.27080.0024
0.35410.0364
0.19280.1131
NLL
3.7344
3.94230.0091
2.07250.1637
2.02640.0751
1.3836
1.21360.0057
4.47202.4332
2.08271.6186
IE
0.1141
0.10060.0001
0.14690.0182
0.14460.0104
1.0112
0.24260.0007
0.03910.0286
0.14470.0963
+ calibrated
ECE
0.0356
0.07210.0020
0.04870.0120
0.06540.0043
0.1107
0.11730.0045
0.06460.0274
0.04530.0114
Evaluation Metrics
NLU++
ZSL
ICL
SFT
SICL
Performance
micro f1
29.2
40.110.09
79.980.59
80.760.31
macro f1
40.26
51.960.04
80.581.00
80.490.12
micro f1
11.18
12.250.02
16.450.72
21.004.78
+ calibrated
macro f1
11.35
12.560.05
21.272.05
27.073.58
Trustworthiness
ECE
0.2311
0.12910.0001
0.01120.0007
0.00200.0007
NLL
0.3435
0.21400.0001
0.13050.0348
0.08390.0127
IE
0.2268
0.14190.0001
0.00140.0007
0.00200.0007
+ calibrated
ECE
0.4679
0.45590.0001
0.41830.0038
0.40980.0049
Table 10: Full experiment results across 7 datasets with different learning methods. We report the mean value for 3 runs with different random seeds and list the variance in the subscripts. We color the Batch Calibration results in grey.
Table 10: Full experiment results across 7 datasets with different learning methods. We report the mean value for 3 runs with different random seeds and list the variance in the subscripts. We color the Batch Calibration results in
Systems
SST5
Macro F1
ECE
Ori.
Max
Mean
Majority
∆
Ori.
Max
Mean
Majority
∆
ZSL
42.00
↑4.79
0.1416
↓0.0634
+ Var-Prompt
42.00
46.79
45.70
44.82
↑4.79
0.1416
0.0782
0.1132
0.1374
↓0.0634
ICL
37.59
↑0.16
0.1833
↓0.0911
+ Var-IC
37.59
37.75
37.33
37.26
↑0.16
0.1833
0.1198
0.1749
0.1857
↓0.0635
+ Var-Prompt
37.59
37.13
36.52
36.83
↓0.46
0.1833
0.1363
0.1838
0.2025
↓0.0470
+ Var-Both
37.59
33.82
35.33
35.78
↓1.81
0.1833
0.0955
0.1832
0.2107
↓0.0911
FT
46.27
↑2.08
0.4030
↓0.1022
+ Var-Prompt
48.11
47.43
48.35
48.33
↑0.24
0.3960
0.3008
0.3466
0.3973
↓0.0952
SupICL
47.12
↑0.79
0.3602
↓0.2402
+ Var-IC
47.12
47.30
47.37
47.31
↑0.25
0.3602
0.2755
0.3428
0.3615
↓0.0847
+ Var-Prompt
47.99
47.70
47.88
47.91
↓0.08
0.2714
0.1698
0.2342
0.2801
↓0.1016
+ Var-Both
47.99
47.18
47.56
47.54
↓0.43
0.2714
0.1200
0.2296
0.2791
↓0.1514
Systems
Manifestos
Macro F1
ECE
Ori.
Max
Mean
Majority
∆
Ori.
Max
Mean
Majority
∆
ZSL
14.50
↑0.79
0.4319
↓0.1699
+ Var-Prompt
14.50
13.69
12.93
15.29
↑0.79
0.4319
0.2620
0.3349
0.4374
↓0.1699
ICL
13.01
↑0.68
0.4760
↓0.2828
+ Var-IC
13.01
13.50
13.46
13.69
↑0.68
0.4760
0.4146
0.4645
0.4689
↓0.0614
+ Var-Prompt
13.01
13.25
11.67
11.53
↑0.24
0.4760
0.2682
0.3661
0.4718
↓0.2078
+ Var-Both
13.01
11.19
11.50
11.21
↓1.51
0.4760
0.1932
0.3537
0.4796
↓0.2828
FT
35.76
↑0.73
0.2005
↓0.1343
+ Var-Prompt
35.39
36.49
35.66
34.91
↑1.10
0.1435
0.0662
0.1049
0.1352
↓0.0773
SupICL
37.55
↑0.06
0.2138
↓0.0902
+ Var-IC
37.55
36.57
37.35
37.61
↑0.06
0.2138
0.1789
0.2096
0.2152
↓0.0349
+ Var-Prompt
37.04
37.14
37.25
36.77
↑0.21
0.2285
0.1388
0.1912
0.2190
↓0.0897
+ Var-Both
37.04
36.67
37.15
37.50
↑0.46
0.2285
0.1236
0.1918
0.2303
↓0.1049
Systems
Hate Speech
Macro F1
ECE
Ori.
Max
Mean
Majority
∆
Ori.
Max
Mean
Majority
∆
ZSL
37.08
↓0.13
0.3175
↓0.0489
+ Var-Prompt
37.08
36.54
36.95
36.95
↓0.13
0.3175
0.2686
0.3024
0.3200
↓0.0489
ICL
40.09
↑1.10
0.2708
↓0.1112
+ Var-IC
40.09
40.01
39.98
40.49
↑0.40
0.2708
0.2332
0.2668
0.2694
↓0.0376
+ Var-Prompt
40.09
41.03
41.19
41.05
↑1.10
0.2708
0.1944
0.2359
0.2745
↓0.0764
+ Var-Both
40.09
39.68
40.30
40.49
↑0.40
0.2708
0.1596
0.2366
0.2776
↓