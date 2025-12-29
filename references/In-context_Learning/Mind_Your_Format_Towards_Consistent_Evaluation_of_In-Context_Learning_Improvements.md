# Mind Your Format: Towards Consistent Evaluation of In-Context Learning Improvements
Anton Voronov Yandex, HSE University, MIPT voronov.ad@phystech.edu Lena Wolf Yandex, HSE University e.a.volf@yandex.ru Max Ryabinin Together AI mryabinin0@gmail.com
Anton Voronov Yandex, HSE University, MIPT voronov.ad@phystech.edu Lena Wolf Yandex, HSE University e.a.volf@yandex.ru
# Abstract
Large language models demonstrate a remarkable capability for learning to solve new tasks from a few examples. The prompt template, or the way the input examples are formatted to obtain the prompt, is an important yet often overlooked aspect of in-context learning. In this work, we conduct a comprehensive study of the template format’s influence on the in-context learning performance. We evaluate the impact of the prompt template across 21 models (from 770M to 70B parameters) and 4 standard classification datasets. We show that a poor choice of the template can reduce the performance of the strongest models and inference methods to a random guess level. More importantly, the best templates do not transfer between different setups and even between models of the same family. Our findings show that the currently prevalent approach to evaluation, which ignores template selection, may give misleading results due to different templates in different works. As a first step towards mitigating this issue, we propose Template Ensembles that aggregate model predictions across several templates. This simple test-time augmentation boosts average performance while being robust to the choice of random set of templates.
6 Jun 2024
arXiv:2401.06766v3
# 1 Introduction
Pretrained language models have emerged as a dominant paradigm for solving many NLP problems in a unified framework (Brown et al., 2020; Chowdhery et al., 2022; Scao et al., 2023; Touvron et al., 2023a). In particular, these models can achieve impressive downstream results with just a few demonstrations given as a part of their input (Liu et al., 2021; Min et al., 2022c), which is often called a prompt in this case.
These few-shot or in-context learning (ICL) abilities (Brown et al., 2020) of large models are a subject of frequent study, as the primary factors behind them are not yet fully understood. For example, one
Together AI mryabinin0@gmail.com
line of work investigates in-context learning within different theoretical frameworks (Xie et al., 2022; Garg et al., 2022; Akyürek et al., 2023). In addition, multiple publications study the importance of different prompt attributes, such as the order of input demonstrations (Lu et al., 2022a) and their labels (Min et al., 2022d).
As shown in Zhao et al. (2021); Min et al. (2022a), the prompt format (i.e., a transformation from a set of examples to a natural language input) is also highly important. However, this aspect is often overlooked in most existing studies. Namely, works proposing modifications of ICL frequently present their results for a specific template without specifying the criteria guiding its selection. Furthermore, even when the results are averaged over a set of templates, they are compared to methods that were evaluated on a different set of templates. We illustrate this common discrepancy in Appendix A. Such inconsistency can lead to a misinterpretation of the reported results: the difference between the performance of two methods may be explained by the variation across prompt formats rather than the methods themselves.
In this work, we evaluate the template sensitivity of 21 models from 8 families, including state-ofthe-art open-access models, such as Llama 2 (Touvron et al., 2023b) and Falcon (Almazrouei et al., 2023), as well as latest instruction-tuned models, such as Mistral (Jiang et al., 2023) and Llama 3 Instruct (AI@Meta, 2024). We show that this issue persists regardless of the model size and the number of demonstrations. Moreover, comparing various in-context learning enhancements while taking the template influence into account renders the superiority of one method over others less apparent. Therefore, it is likely that the gains reported for advanced prompting methods can often be attributed to a luckily chosen template.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/327f/327fd555-4288-4393-9db9-925efc6fdec8.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: An example template transformation for two demonstrations. Different prompt formats lead to different rankings both for models and ICL methods, and the best template for one method can be suboptimal for others.</div>
Crucially, there are no universally best templates for a given task. The best performing demonstration format for a fixed evaluation setting (i.e., the dataset, the model, the demonstration set, and the prediction method) does not transfer consistently across models (even within the same family), demonstration sets, or different prediction methods. We find this concerning, as even the best template for a given setting can produce poor results after slight changes, which makes “tuning” the template a very difficult task. As a first step towards addressing template sensitivity in a practical way, we propose Template Ensembles — a test-time augmentation approach that averages model predictions over several prompt formats. This method is easy to implement and increases the average performance across templates for multiple prompting methods while reducing the sensitivity of these methods to the template choice. In summary, our contributions are as follows:
1. We conduct a broad evaluation1 of prompt template sensitivity across 21 models and 4 datasets, showing that the performance gains similar to using in-context learning improvements can be achieved solely by selecting a proper template.
2. We show that the choice of the best template depends on a combination of factors and that it is not possible to transfer the best template between models or prompting methods without a negative impact on quality.
2. We show that the choice of the best template depends on a combination of factors and that it is not possible to transfer the best template between models or prompting methods without a negative impact on quality.
3. We propose Template Ensembles as a baseline solution for improving the template robustness for in-context learning.
1Our code and results of all evaluations can be found at github.com/yandex-research/mind-your-format
# 2 Background and Related Work
# 2.1 In-Context Learning
An important property of LLMs is their ability to learn new tasks from only a few demonstrations (Radford et al., 2019; Brown et al., 2020). This capability, known as in-context learning, forms the focus of our work. We focus on sequence classification, as it is the most widely studied task for understanding and improving ICL performance. Formally, classifying an input xtest with incontext learning can be described as finding the class c in the space of label tokens C that yields a sequence with the highest probability according to a language model. The input sequence consists of demonstration inputs and labels (xi, yi) and a test input (xtest, c); to obtain a natural language input, demonstrations are formatted with a template. Each template consists of four components: input and output verbalizers vI(x) and vO(y, C) that transform (xi, yi) into a natural language text, an intra-separator to divide input from output, and an inter-separator to join several demonstrations. Figure 1 shows an example of transforming a set of demonstrations into a context for ICL.
# 2.2 In-Context Learning Analysis
Recent work has shown that ICL can perform at levels comparable to finetuning (Chowdhery et al., 2022; Hoffmann et al., 2022). Still, in-context learning is known to be highly dependent on the way the model input is formed: a prompt is defined by several components, and altering any of them can lead to unpredictable changes in performance. Template Selection There are multiple ways to construct a template for a task. The most straightforward approach is to use minimal templates (vI = {x}, vO = {C[y]}) or universal verbalizers like “input/output”, as done in Wang et al. (2023) and Wei et al. (2023).
Another strategy is to create task-specific templates. Jiang et al. (2020) generate paraphrases of templates for the relation extraction task. Authors show the sensitivity of masked language models to the prompt format and propose to ensemble predictions over the best templates. Compared to this method, our approach is task-agnostic and does not require evaluating all templates in advance. Several studies aim to find templates that directly optimize in-context learning performance (Shin et al., 2020; Gao et al., 2021). Our work unifies the results of previous research, using the verbalizers proposed by Gao et al. (2021), as well as minimal and universal templates.
# Choice and Order of Demonstrations
# s The
choice of examples for ICL is highly important, as they enable the model to condition on correct input and label distributions for the task (Wu et al., 2023; Nguyen and Wong, 2023; Min et al., 2022d; Chang and Jia, 2023). Furthermore, the order of examples also significantly affects the results and does not transfer between models even within the same family (Lu et al., 2022b; Zhao et al., 2021). In this work, we analyze two recent methods for selecting demonstrations. Wang et al. (2023) propose learning latent concept variables for a task and using them to find examples that can best predict the task concept. We refer to this method as Implicit Topic Models or ITM. In turn, Z-ICL (Lyu et al., 2023) generates pseudo-demonstrations by retrieving most similar examples to the test sentence from an unlabeled dataset and assigning random labels to retrieved examples. Crucially, both methods are evaluated on single templates that differ across two works. Therefore, it is unclear whether the reported performance gains arise from the methods themselves or from a particular combination of the example selection strategy, the model, and the chosen template.
Prediction Methods The standard approach for classification with LLMs is to compute the sequence probability with each of the possible labels and select the label with the highest probability. We refer to this method as DIRECT further on. Alternatively, one can use more advanced prediction methods that aim to reduce the variance across prompt formats. The CALIBRATION method (Zhao et al., 2021) computes a correction factor based on the deviation of the model’s predictions for a placeholder input from a uniform distribution over labels and applies this factor to test set predictions.
Recent work has proposed multiple improvements of this method (Fei et al., 2023; Han et al., 2023; Zhou et al., 2024); to limit the scope of our study, we focus only on the base CALIBRATION approach in this work. Lastly, the CHANNEL prompting technique, proposed in Min et al. (2022b), maximizes P(x|y) instead of P(y|x). Both of these methods aim to mitigate the issue of ICL sensitivity to the prompt template choice. However, as we show in Appendix A, these methods are evaluated on their own sets of templates. In this paper, we strive for a more unified view on the robustness of advanced prompting methods and compare their performance across a broader range of templates and models.
# Prompt and Template Robustness Although
the problem of prompt robustness is relatively wellknown, until recently, the discussion of template robustness has been limited. Notably, Sclar et al. (2023) present a highly relevant study of prompt format sensitivity, reporting a significant performance variation across formats even for large models or minor template changes. While their experiments are conducted in the standard setup (randomly selected examples and default prompting), our work instead focuses on alternative prompting and example selection methods, several of which (Zhao et al., 2021; Min et al., 2022b) were proposed to improve the prompt robustness of ICL. Similarly to papers in other subfields of machine learning arguing for a more consistent methodology (Dacrema et al., 2019; Musgrave et al., 2020; Platonov et al., 2023), the goal of our work is to demonstrate that disparate experiment setups lead to an invalid comparison of competing methods. Moreover, several works study prompt robustness in a broader sense by considering models that use natural language instructions instead of labeled demonstrations (Webson and Pavlick, 2022; Leidinger et al., 2023; Weber et al., 2023). Recently, Mizrahi et al. (2023) have shown that very similar instructions can lead to drastic differences in task performance for a variety of instruction-tuned models. Although we study a similar issue, the focus of our work is on in-context learning and the transfer of best prompts between evaluation setups. Still, we find that instruction-tuned models lack in-context robustness as well, which confirms previous observations and emphasizes the need for language model evaluation that takes prompt design into account.
<div style="text-align: center;">Model family</div>
Model family
Parameters (B)
GPT-J (Wang and Komatsuzaki, 2021)
6
GPT-NeoX (Black et al., 2022)
20
BLOOM (Scao et al., 2023)
1.7, 3, 7.1
OPT (Zhang et al., 2022)
6.7, 30, 66
Pythia (Biderman et al., 2023)
6.9, 12
LLaMA (Touvron et al., 2023a)
7, 13, 30, 65
Llama 2 (Touvron et al., 2023b)
13, 70
Falcon (Almazrouei et al., 2023)
1, 7, 40
Llama 3 Instruct (AI@Meta, 2024)
8
Mistral v0.3 Instruct (Jiang et al., 2023)
7
# 3 Setup & Methodology
# 3.1 Models and Data
We evaluate the robustness of in-context learning to template selection across a wide range of models on classification tasks. All models used in our work are listed in Table 1: we run experiments on model families frequently used in literature (such as OPT and BLOOM), as well as the latest models with the highest quality (such as Llama 2 and Falcon). In preliminary experiments, we observed that the performance of some models in the few-shot regime lags behind their zero-shot results. Hence, we excluded these models from further investigation. Further details regarding this selection procedure can be found in Appendix B. We experiment with 4 sequence classification datasets: SST-2 (Socher et al., 2013), DBPedia ontology classification task (Lehmann et al., 2015), AGNews (Zhang et al., 2015), and TREC Question Classification (Li and Roth, 2002). Although these datasets are frequently used in ICL studies, there is no consensus regarding the templates that should be used for each task. One can construct an input for in-context learning from a set of demonstrations by using a template consisting of four parts, as illustrated in Figure 1. We present all options for verbalizers and separators for each dataset we study in Table 2.
Dataset
Input verbalizer
Output verbalizer
Intra-separator
Inter-separator
SST-2
“output: {}”, “target: {}”, “label: {}”,
“{}”
“sentence: {}”,
“text: {}”,
“input: {}”,
“emotion: {}”, “sentiment: {}”, “A {} one.”
“\n”
“ ”,
“\n\n”
“\n”,
“ ”,
“It was {}.”, “All in all {}.”, “A {} piece.”
DBPedia
“output: {}”, “target: {}”, “label: {}”,
AGNews
“Topic: {}.”, “Subject: {}.”,
TREC
‘This is about {}.”, “It is about {}.”
Table 2: Possible choices for all components of templates used in our work.
Any combination of these components results in a valid template. This set of options results in 216 possible prompt formats for SST-2 and 168 for DBPedia, AGNews and TREC. A single evaluation run of all models on 10 random templates in one setup takes 17–48 hours on a single NVIDIA A100-80GB GPU, depending on the dataset.
# 3.2 Methods
Along with studying the robustness of standard in-context learning, we consider its improvements proposed in prior work. We focus on two main directions of ICL enhancements mentioned in Section 2: example selection and prediction methods. For each setting, we aggregate the results over 3 random seeds for example selection, with 10 random templates used for each seed and report the mean and standard deviation of classification accuracy, unless specified otherwise. As a baseline for demonstration selection, we choose the most straightforward approach of selecting N random examples from the training dataset. Intuitively, selecting more relevant examples for ICL should yield better performance. Therefore, we investigate the template sensitivity of two demonstration selection methods described in Section 2.1: ITM and Z-ICL. Specifically, we select N = 4 examples using official implementations of each method. Importantly, ITM requires training a concept model before choosing the examples. For GPT2 Large, this procedure takes approximately 30 hours on a single NVIDIA A100-80GB. Repeating it for each model would be infeasible, especially given that the largest model has 86 times more parameters. Therefore, we use the checkpoints of the GPT2 Large concept model provided by the authors to select demonstrations. Also, we reuse the same examples for all models, leveraging authors’ observations that demonstrations chosen with ITM can be transferred between models.
Model
SST-2
DBPedia
AGNews
TREC
2-shot
4-shot
2-shot
4-shot
2-shot
4-shot
2-shot
4-shot
Falcon 1B
0.650.17
0.770.15
0.360.25
0.440.23
0.520.17
0.560.19
0.260.09
0.310.09
Falcon 7B
0.770.16
0.830.16
0.400.21
0.490.18
0.510.20
0.600.19
0.320.09
0.390.11
Falcon 40B
0.790.17
0.920.07
0.420.15
0.540.06
0.640.23
0.750.09
0.360.07
0.460.10
Llama 2 13B
0.790.17
0.920.07
0.400.15
0.510.09
0.700.15
0.760.09
0.320.09
0.410.14
Llama 2 70B
0.830.14
0.920.09
0.460.15
0.600.05
0.760.14
0.820.05
0.410.07
0.510.06
As discussed in Section 2, more advanced prediction techniques can improve in-context learning accuracy. Therefore, we compare DIRECT prompting with CHANNEL (Min et al., 2022b) and CALIBRATION (Zhao et al., 2021) prediction methods.
# 4 Evaluation
# 4.1 Baseline Results
We begin with analyzing the robustness of language models to the template choice in the baseline setup. Specifically, we evaluate models in zero-shot and few-shot settings, selecting 2/4 random demonstrations and using the DIRECT prediction method. Our results in Table 3 show that even the most capable models such as Falcon and Llama 2 are highly sensitive to the prompt format; Appendix C contains the results for the full set of 19 base models, and Appendix L reports the results for instructiontuned models. Although the variance caused by this sensitivity makes it harder to observe the increase in ICL performance with the model size or the number of demonstrations, both trends still persist. However, even the largest models have standard deviations of scores up to 35% of their mean values. To mitigate this lack of robustness, we could remove consistently underperforming prompt formats from the template pool. We analyze the impact of separate template components in Appendix D and find that there are no specific parts (for example, verbalizers or separators) which could be excluded from evaluation. Furthermore, we observe that a combination of “suboptimal” parts may result in an optimal template.
# 4.2 Prediction Methods
Next, we aim to evaluate the performance of different prediction methods in a unified setting. Ideally, we would like these modifications to reduce the
variance across templates, making the model behavior less dependent on the input format. We evaluate CHANNEL and CALIBRATION methods in the 2-shot setting along with the DIRECT baseline. As depicted in Figure 2, both CHANNEL and CALIBRATION generally exhibit improved performance in comparison with DIRECT. Still, for a number of models and datasets, the range of scores for DIRECT substantially overlaps with those of advanced methods. This suggests that there are templates reaching the best performance with the DIRECT prediction method. Additionally, Table 10 of Appendix E reveals that despite CALIBRATION yielding the highest mean accuracy more often than other methods, it is more sensitive to the template choice than CHANNEL. Similar findings for instruction-tuned models are contained in Appendix L. Therefore, the choice of the prediction method should likely rely on the downstream usage scenario and the target evaluation setting.
# 4.3 Example Selection Methods
Another area of ICL improvements that we evaluate on the matter of template sensitivity is the example selection strategy. We compare ITM and Z-ICL methods to the RANDOM baseline in 4-shot setting, since using 4 demonstration was the main evaluation setting in the works proposing these methods. We use DIRECT prediction method to evaluate the gains of advanced example selection strategies independently from other ICL modifications. Results in Figure 3 and Table 14 illustrate that when taking template sensitivity into account, advanced example selection methods often perform worse than random choice baseline. ITM increases the average performance in most cases but still has a remarkably high standard deviation across templates. Examples selected using the Z-ICL method lead to more consistent but worse performance.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/578e/578ee627-622e-4ca5-b822-1e44fc558765.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Comparison of in-context learning prediction methods in the 2-shot setting</div>
Note that our evaluation setup differs from those described in the original works, which might explain the discrepancy between our findings and the results reported by authors. Namely, we use the DIRECT prompting method and sample 10 random templates that may not include the templates used by authors of ITM and Z-ICL. To confirm template instability for prediction methods in their original implementations, we reproduce both methods and report our findings in Appendix F. We observe high sensitivity to the prompt format, which raises a question of how much the reported gains of these methods can be attributed to the methods themselves and not to the template choice. We conclude that the prompt format should be viewed as important as the example selection or the prediction method for ICL evaluation. However, the search space of possible templates is infinite, which makes exhaustive search for each combination of the dataset, the model and the number of examples impractical. Ideally, the best template for one setting would be optimal for all others or at least for similar settings. However, as we demonstrate in the following section, this is not the case.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ecfd/ecfdb015-34ad-43e8-a394-0144c8bac361.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Comparison of the selection methods in the DIRECT 4-shot setting. For the evaluation results of other models and datasets, please refer to Appendix G.</div>
# 5 Template Transfer Evaluation
# 5.1 Setup
We begin by defining a successful transfer between ICL settings. In order to do so, we evaluate how the quality of model predictions varies across 30 random templates from Table 2. The results described in Appendix H demonstrate that the top-10 template on average yields 90% of the best template score. Therefore, if a prompt format is present in top-10 for both of the two compared setups, we can consider this an instance of successful transfer. To compare sets of the best templates for a pair of settings, we compute Intersection-over-Union (IoU), also known as the Jaccard similarity coefficient (Jaccard, 1912), for top-10 best templates in each setting. We also consider using the ρ rank correlation coefficient (Spearman, 1904) as another measure of template transfer. However, its value can increase when low-performing templates have similar rankings in different ICL setups, while the transfer of efficient templates remains low. Still, we provide the results for this metric in Appendix I.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2ead/2ead2500-50ba-487b-8d80-f93e9342ef7b.png" style="width: 50%;"></div>
<div style="text-align: center;">GPT-J GPT-NeoX OPT 6.7B OPT 30B OPT 66B BLOOM 1.7B BLOOM 3B BLOOM 7.1B Pythia 6.9B Pythia 12B LLaMA 7B LLaMA 13B LLaMA 30B LLaMA 65B Falcon 1B Falcon 7B Falcon 40B Llama 2 13B Llama 2 70B</div>
<div style="text-align: center;">Figure 4: IoU of top-10 templates for all base models with 2 random demonstrations and the DIRECT prediction method on the DBPedia dataset.</div>
Figure 4: IoU of top-10 templates for all base models with 2 random demonstrations and the DIRECT prediction method on the DBPedia dataset.
# 5.2 Transfer Between Models
Next, we analyze the transfer of the bestperforming templates between models in the baseline setup. Specifically, we collect the results of each model in the 2-shot learning setting with DIRECT prediction method and RANDOM demonstrations (fixed throughout the experiments) for 30 templates. A heatmap of IoU for the transfer of top-10 best templates between 19 base models on the DBPedia dataset is presented in Figure 4; for other datasets, please see Appendix J. We observe that the IoU values exceed 0.5 only for a few model pairs on all datasets, meaning that the capacity for template transfer between models in the same setup is generally low. This is especially concerning for models within a single family: as these models are trained on the same data and have the same architecture, one would expect them to perform similarly on the same prompt formats. These observations signify that comparing ICL methods across models with a single template can lead to incorrect conclusions: a template that is effective for one model can easily be one of the worst choices for another model.
# 5.3 Transfer Between Prediction Methods
As discussed in Section 4.2, no prediction method that we evaluate can consistently outperform others across all models and datasets. Therefore, to find an optimal setup for a new ICL improvement, one needs to evaluate every prediction technique in multiple templates. We investigate the possibility of finding a universally optimal prompt for different methods to reduce the total computational cost.
Direct ↔
Direct ↔Channel ↔
Calibration Channel Calibration
SST-2
0.490.17
0.300.11
0.310.08
DBPedia
0.540.17
0.470.15
0.450.14
AG News
0.360.11
0.250.13
0.350.14
TREC
0.310.12
0.230.09
0.280.13
Table 4: Intersection-over-Union for pairs of prompting methods averaged over the results of 19 base models obtained in the RANDOM 2-shot setup. Standard deviations are in subscript.
To answer this question, we calculate the IoU between top 10 performing templates for each method for a fixed set of demonstrations. Results in Table 4 display that similarly to the models, the transfer between prediction methods is also low. Consequently, the prompt format sensitivity issue creates a burden on authors of new ICL modifications; they must tune templates for every prediction method they want to combine with their own approach.
# 5.4 Transfer Between Demonstration Selection Methods
Having found that the best-performing templates are specific both to the model and the prediction method, we now aim to find whether the best formats would be the same for different demonstration sets in the same setup. Similarly to previous experiments, we calculate IoU for 10 templates that yield the highest scores for each method. Results in Figure 5 illustrate that simply adding demonstrations, even if they were obtained with the same method, can significantly alter the ranking of the best templates. This justifies the necessity to evaluate example selection methods on a range of templates to avoid misinterpretation of the results.
Based on the above findings, we conclude that the results of evaluation of various ICL improvements without consideration of template sensitivity issue are hardly reliable for several reasons. First, as the best templates do not transfer between models even within the same family, scoring a method across several models using the same format will inevitably lead to underestimation of the method for all models except the one for which the format was tuned. Next, as there is little evidence of transfer between setups, the format selection procedure needs to be precisely described and applied in all evaluated settings for a fair comparison.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e359/e3596cc8-5f46-4879-b706-e7c34dfe5716.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: IoU of 10 best templates for example selection methods on the AG News dataset. METHOD-N indicates that METHOD was used to select N examples.</div>
In summary, we find that there are no universally well-performing prompt formats. Therefore, the results of in-context learning evaluation can be reliable only if they are aggregated over several templates or if each setting is evaluated in its best-performing template. The former approach requires accounting for the variance of the scores and makes comparison less apparent, while the latter can be computationally expensive.
# 6 Template Ensembles
To reduce the variance in performance caused by the template choice, we propose to ensemble model predictions across multiple templates. This approach is widely used in machine learning (Ho, 1995; Lakshminarayanan et al., 2017) for improving the predictive performance of the model, as well as its robustness, and can be viewed as a form of test-time augmentation (Krizhevsky et al., 2012; Simonyan and Zisserman, 2015). Prior work on prompt ensembling has shown significant gains
Model
Direct
Channel
Calibration
Single
Ensemble
Single
Ensemble
Single
Ensemble
LLaMA 2 13B
0.790.17
0.850.09
0.820.10
0.900.02
0.880.09
0.930.03
LLaMA 2 70B
0.830.14
0.950.01
0.830.08
0.920.01
0.880.13
0.940.03
Falcon 1B
0.650.17
0.740.07
0.770.10
0.890.01
0.710.17
0.890.01
Falcon 7B
0.770.16
0.810.00
0.780.09
0.900.00
0.790.15
0.930.02
Falcon 40B
0.790.17
0.930.01
0.810.09
0.910.01
0.870.13
0.950.00
Table 5: Comparison of 2-shot learning performance on the SST-2 dataset using ensembles of 5 templates and a single template. Standard deviations over 5 random seeds are in subscript, best accuracy for each model is in bold.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7050/7050b519-023c-4f9d-8efb-8cc0d034fdfa.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Template ensemble accuracy as a function of its size for Falcon 40B on the SST-2 dataset in the 2-shot learning setup. Dashed lines depict the results of baseline methods averaged over 10 templates.</div>
Figure 6: Template ensemble accuracy as a function of its size for Falcon 40B on the SST-2 dataset in the 2-shot learning setup. Dashed lines depict the results of baseline methods averaged over 10 templates.
by training a boosting algorithm on model outputs (Hou et al., 2023); by contrast, our method needs only the pretrained model predictions without additional training. Formally, our method computes label probabilities across predictions for each of N templates, where N is the ensemble size, and outputs the label with the highest average probability. In early experiments, we tried selecting the most common label among the predictions; however, we found this voting strategy to perform poorly on tasks with a large number of classes. It is also important to note that ensembling N predictions involves running the model N times more compared to singleformat evaluation, which makes this approach more computationally expensive. We view template ensembles as a baseline solution for the problem of prompt format sensitivity and leave the exploration of more efficient methods to future work. We begin with determining the minimal ensemble size that consistently reduces variance while increasing the average performance. We observe that for the majority of models and prediction methods, an ensemble achieves the best accuracy when its size reaches 4 or 5 (see an example in Figure 6), with further expansion being less effective. We also found that smaller ensembles may demonstrate unstable behavior, with the possibility of a drop in
performance if a suboptimal template is sampled. Therefore, we report results for ensembles of size 5 and average the results over 5 random seeds. Next, we evaluate the performance gains of Template Ensembles for different prediction methods. Our findings in Tables 5 and 20 and Appendices K and L indicate that ensembles increase the accuracy for all evaluated models and prediction methods. Most importantly, they also significantly reduce the variance caused by the template choice for most setups. Therefore, we conclude that template ensembling allows to preserve the increase in accuracy provided by ICL modifications while mitigating the template sensitivity issue.
# 7 Conclusion
In this work, we study the inconsistencies in the evaluation of in-context learning advancements introduced by the template sensitivity of large language models. Specifically, we find that ICL improvements exhibit high variation across template formats and that it is not possible to reuse the same template across different modifications. This aspect is often overlooked in prior work, despite the fact that the impact of template selection on prediction accuracy may be comparable with the choice of demonstrations or prompting methods. While we propose Template Ensembles as an initial solution to this problem, the general sensitivity of language models to minor prompt variations is yet to be addressed. Consequently, we believe that the research community should take this problem into account when developing new models, evaluation benchmarks, or in-context learning methods.
# Limitations
Due to limited computational resources and the high cost for evaluation on a large range of models, we only focus on four classification datasets. Moreover, we only compare two example selection methods to a random baseline, potentially overlooking other effective approaches. Additionally, the space of templates could be expanded for more comprehensive experimentation. For example, we did not explore label mapping, including random labels, which is an important aspect of the template. We would like to notice that our study focuses on a template selection impact on a performance and a degree of template transfer between different setups but not on templates themselves. Future
work should further analyze not only which templates lead to a change in performance but also on why they affect it.
# References
AI@Meta. 2024. Llama 3 model card.
# AI@Meta. 2024. Llama 3 model card.
Ting-Yun Chang and Robin Jia. 2023. Data curation alone can stabilize in-context learning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8123–8144, Toronto, Canada. Association for Computational Linguistics.
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek
Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022. Palm: Scaling language modeling with pathways. Preprint, arXiv:2204.02311.
Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022. Palm: Scaling language modeling with pathways. Preprint, arXiv:2204.02311. Maurizio Ferrari Dacrema, Paolo Cremonesi, and Dietmar Jannach. 2019. Are we really making much progress? a worrying analysis of recent neural recommendation approaches. In Proceedings of the 13th ACM Conference on Recommender Systems, RecSys ’19, page 101–109, New York, NY, USA. Association for Computing Machinery. Yu Fei, Yifan Hou, Zeming Chen, and Antoine Bosselut. 2023. Mitigating label biases for in-context learning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14014–14031, Toronto, Canada. Association for Computational Linguistics. Tianyu Gao, Adam Fisch, and Danqi Chen. 2021. Making pre-trained language models better few-shot learners. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3816–3830, Online. Association for Computational Linguistics. Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. 2022. What can transformers learn incontext? a case study of simple function classes. In Advances in Neural Information Processing Systems. Zhixiong Han, Yaru Hao, Li Dong, Yutao Sun, and Furu Wei. 2023. Prototypical calibration for fewshot learning of language models. In The Eleventh International Conference on Learning Representations. Tin Kam Ho. 1995. Random decision forests. In Proceedings of 3rd international conference on document analysis and recognition, volume 1, pages 278– 282. IEEE. Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals,
Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022. Palm: Scaling language modeling with pathways. Preprint, arXiv:2204.02311. Maurizio Ferrari Dacrema, Paolo Cremonesi, and Dietmar Jannach. 2019. Are we really making much progress? a worrying analysis of recent neural recommendation approaches. In Proceedings of the 13th ACM Conference on Recommender Systems, RecSys ’19, page 101–109, New York, NY, USA. Association for Computing Machinery. Yu Fei, Yifan Hou, Zeming Chen, and Antoine Bosselut. 2023. Mitigating label biases for in-context learning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14014–14031, Toronto, Canada. Association for Computational Linguistics. Tianyu Gao, Adam Fisch, and Danqi Chen. 2021. Making pre-trained language models better few-shot learners. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3816–3830, Online. Association for Computational Linguistics. Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. 2022. What can transformers learn incontext? a case study of simple function classes. In Advances in Neural Information Processing Systems. Zhixiong Han, Yaru Hao, Li Dong, Yutao Sun, and Furu Wei. 2023. Prototypical calibration for fewshot learning of language models. In The Eleventh International Conference on Learning Representations. Tin Kam Ho. 1995. Random decision forests. In Proceedings of 3rd international conference on document analysis and recognition, volume 1, pages 278– 282. IEEE. Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals,
ordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals,
Bairu Hou, Joe O’Connor, Jacob Andreas, Shiyu Chang, and Yang Zhang. 2023. PromptBoosting: Blackbox text classification with ten forward passes. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 13309–13324. PMLR. Jaccard. 1912. The distribution of the flora of the alpine zone. In New Phytologist, volume 11, pages 37–50. Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825. Zhengbao Jiang, Frank F. Xu, Jun Araki, and Graham Neubig. 2020. How can we know what language models know? Transactions of the Association for Computational Linguistics, 8:423–438. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2012. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, volume 25. Curran Associates, Inc. Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. 2017. Simple and scalable predictive uncertainty estimation using deep ensembles. Preprint, arXiv:1612.01474. Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick Van Kleef, Sören Auer, et al. 2015. Dbpedia–a large-scale, multilingual knowledge base extracted from wikipedia. Semantic web, 6(2):167–195. Alina Leidinger, Robert van Rooij, and Ekaterina Shutova. 2023. The language of prompting: What linguistic properties make a prompt successful? In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 9210–9232, Singapore. Association for Computational Linguistics. Xin Li and Dan Roth. 2002. Learning question classifiers. In COLING 2002: The 19th International Conference on Computational Linguistics. Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2021. What makes good in-context examples for gpt-3? In Workshop on Knowledge Extraction and Integration for Deep Learning Architectures; Deep Learning Inside Out. Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022a. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In Proceedings of the
Xin Li and Dan Roth. 2002. Learning question classifiers. In COLING 2002: The 19th International Conference on Computational Linguistics.
Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2021. What makes good in-context examples for gpt-3? In Workshop on Knowledge Extraction and Integration for Deep Learning Architectures; Deep Learning Inside Out.
Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022a. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In Proceedings of the
Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022a. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In Proceedings of the
Moran Mizrahi, Guy Kaplan, Dan Malkin, Rotem Dror, Dafna Shahaf, and Gabriel Stanovsky. 2023. State of what art? a call for multi-prompt llm evaluation. Preprint, arXiv:2401.00595.
Kevin Musgrave, Serge Belongie, and Ser-Nam Lim. 2020. A metric learning reality check. In Computer Vision – ECCV 2020, pages 681–699, Cham. Springer International Publishing.
Tai Nguyen and Eric Wong. 2023. In-context example selection with influences. Preprint, arXiv:2302.11042.
Tai Nguyen and Eric Wong. 2023. In-context example selection with influences. Preprint, arXiv:2302.11042.
Oleg Platonov, Denis Kuznedelev, Michael Diskin, Artem Babenko, and Liudmila Prokhorenkova. 2023. A critical look at the evaluation of GNNs under heterophily: Are we really making progress? In The Eleventh International Conference on Learning Representations.
Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners.
even Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, Dragomir Radev, Eduardo González Ponferrada, Efrat Levkovizh, Ethan Kim, Eyal Bar Natan, Francesco De Toni, Gérard Dupont, Germán Kruszewski, Giada Pistilli, Hady Elsahar, Hamza Benyamina, Hieu Tran, Ian Yu, Idris Abdulmumin, Isaac Johnson, Itziar Gonzalez-Dios, Javier de la Rosa, Jenny Chim, Jesse Dodge, Jian Zhu, Jonathan Chang, Jörg Frohberg, Joseph Tobing, Joydeep Bhattacharjee, Khalid Almubarak, Kimbo Chen, Kyle Lo, Leandro Von Werra, Leon Weber, Long Phan, Loubna Ben allal, Ludovic Tanguy, Manan Dey, Manuel Romero Muñoz, Maraim Masoud, María Grandury, Mario Šaško, Max Huang, Maximin Coavoux, Mayank Singh, Mike Tian-Jian Jiang, Minh Chien Vu, Mohammad A. Jauhar, Mustafa Ghaleb, Nishant Subramani, Nora Kassner, Nurulaqilla Khamis, Olivier Nguyen, Omar Espejel, Ona de Gibert, Paulo Villegas, Peter Henderson, Pierre Colombo, Priscilla Amuok, Quentin Lhoest, Rheza Harliman, Rishi Bommasani, Roberto Luis López, Rui Ribeiro, Salomey Osei, Sampo Pyysalo, Sebastian Nagel, Shamik Bose, Shamsuddeen Hassan Muhammad, Shanya Sharma, Shayne Longpre, Somaieh Nikpoor, Stanislav Silberberg, Suhas Pai, Sydney Zink, Tiago Timponi Torrent, Timo Schick, Tristan Thrush, Valentin Danchev, Vassilina Nikoulina, Veronika Laippala, Violette Lepercq, Vrinda Prabhu, Zaid Alyafeai, Zeerak Talat, Arun Raja, Benjamin Heinzerling, Chenglei Si, Elizabeth Salesky, Sabrina J. Mielke, Wilson Y. Lee, Abheesht Sharma, Andrea Santilli, Antoine Chaffin, Arnaud Stiegler, Debajyoti Datta, Eliza Szczechla, Gunjan Chhablani, Han Wang, Harshit Pandey, Hendrik Strobelt, Jason Alan Fries, Jos Rozen, Leo Gao, Lintang Sutawika, M Saiful Bari, Maged S. Al-shaibani, Matteo Manica, Nihal Nayak, Ryan Teehan, Samuel Albanie, Sheng Shen, Srulik Ben-David, Stephen H. Bach, Taewoon Kim, Tali Bers, Thibault Fevry, Trishala Neeraj, Urmish Thakker, Vikas Raunak, Xiangru Tang, ZhengXin Yong, Zhiqing Sun, Shaked Brody, Yallow Uri, Hadar Tojarieh, Adam Roberts, Hyung Won Chung, Jaesung Tae, Jason Phang, Ofir Press, Conglong Li, Deepak Narayanan, Hatim Bourfoune, Jared Casper, Jeff Rasley, Max Ryabinin, Mayank Mishra, Minjia
Zhang, Mohammad Shoeybi, Myriam Peyrounette, Nicolas Patry, Nouamane Tazi, Omar Sanseviero, Patrick von Platen, Pierre Cornette, Pierre François Lavallée, Rémi Lacroix, Samyam Rajbhandari, Sanchit Gandhi, Shaden Smith, Stéphane Requena, Suraj Patil, Tim Dettmers, Ahmed Baruwa, Amanpreet Singh, Anastasia Cheveleva, Anne-Laure Ligozat, Arjun Subramonian, Aurélie Névéol, Charles Lovering, Dan Garrette, Deepak Tunuguntla, Ehud Reiter, Ekaterina Taktasheva, Ekaterina Voloshina, Eli Bogdanov, Genta Indra Winata, Hailey Schoelkopf, JanChristoph Kalo, Jekaterina Novikova, Jessica Zosa Forde, Jordan Clive, Jungo Kasai, Ken Kawamura, Liam Hazan, Marine Carpuat, Miruna Clinciu, Najoung Kim, Newton Cheng, Oleg Serikov, Omer Antverg, Oskar van der Wal, Rui Zhang, Ruochen Zhang, Sebastian Gehrmann, Shani Pais, Tatiana Shavrina, Thomas Scialom, Tian Yun, Tomasz Limisiewicz, Verena Rieser, Vitaly Protasov, Vladislav Mikhailov, Yada Pruksachatkun, Yonatan Belinkov, Zachary Bamberger, Zdenˇek Kasner, Alice Rueda, Amanda Pestana, Amir Feizpour, Ammar Khan, Amy Faranak, Ana Santos, Anthony Hevia, Antigona Unldreaj, Arash Aghagol, Arezoo Abdollahi, Aycha Tammour, Azadeh HajiHosseini, Bahareh Behroozi, Benjamin Ajibade, Bharat Saxena, Carlos Muñoz Ferrandis, Danish Contractor, David Lansky, Davis David, Douwe Kiela, Duong A. Nguyen, Edward Tan, Emi Baylor, Ezinwanne Ozoani, Fatima Mirza, Frankline Ononiwu, Habib Rezanejad, Hessie Jones, Indrani Bhattacharya, Irene Solaiman, Irina Sedenko, Isar Nejadgholi, Jesse Passmore, Josh Seltzer, Julio Bonis Sanz, Karen Fort, Livia Dutra, Mairon Samagaio, Maraim Elbadri, Margot Mieskes, Marissa Gerchick, Martha Akinlolu, Michael McKenna, Mike Qiu, Muhammed Ghauri, Mykola Burynok, Nafis Abrar, Nazneen Rajani, Nour Elkott, Nour Fahmy, Olanrewaju Samuel, Ran An, Rasmus Kromann, Ryan Hao, Samira Alizadeh, Sarmad Shubber, Silas Wang, Sourav Roy, Sylvain Viguier, Thanh Le, Tobi Oyebade, Trieu Le, Yoyo Yang, Zach Nguyen, Abhinav Ramesh Kashyap, Alfredo Palasciano, Alison Callahan, Anima Shukla, Antonio MirandaEscalada, Ayush Singh, Benjamin Beilharz, Bo Wang, Caio Brito, Chenxi Zhou, Chirag Jain, Chuxin Xu, Clémentine Fourrier, Daniel León Periñán, Daniel Molano, Dian Yu, Enrique Manjavacas, Fabio Barth, Florian Fuhrimann, Gabriel Altay, Giyaseddin Bayrak, Gully Burns, Helena U. Vrabec, Imane Bello, Ishani Dash, Jihyun Kang, John Giorgi, Jonas Golde, Jose David Posada, Karthik Rangasai Sivaraman, Lokesh Bulchandani, Lu Liu, Luisa Shinzato, Madeleine Hahn de Bykhovetz, Maiko Takeuchi, Marc Pàmies, Maria A Castillo, Marianna Nezhurina, Mario Sänger, Matthias Samwald, Michael Cullan, Michael Weinberg, Michiel De Wolf, Mina Mihaljcic, Minna Liu, Moritz Freidank, Myungsun Kang, Natasha Seelam, Nathan Dahlberg, Nicholas Michio Broad, Nikolaus Muellner, Pascale Fung, Patrick Haller, Ramya Chandrasekhar, Renata Eisenberg, Robert Martin, Rodrigo Canalli, Rosaline Su, Ruisi Su, Samuel Cahyawijaya, Samuele Garda, Shlok S Deshmukh, Shubhanshu Mishra, Sid Kiblawi, Simon Ott, Sinee Sang-aroonsiri, Srishti Kumar, Ste-
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971.
ugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu,
Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models. Preprint, arXiv:2307.09288.
Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, and William Yang Wang. 2023. Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning. Preprint, arXiv:2301.11916.
Lucas Weber, Elia Bruni, and Dieuwke Hupkes. 2023. The icl consistency test. Preprint, arXiv:2312.04945.
Han Zhou, Xingchen Wan, Lev Proleev, Diana Mincu, Jilin Chen, Katherine A Heller, and Subhrajit Roy. 2024. Batch calibration: Rethinking calibration for in-context learning and prompt engineering. In The Twelfth International Conference on Learning Representations.
Han Zhou, Xingchen Wan, Lev Proleev, Diana Mincu, Jilin Chen, Katherine A Heller, and Subhrajit Roy. 2024. Batch calibration: Rethinking calibration for in-context learning and prompt engineering. In The Twelfth International Conference on Learning Representations.
# A Templates from Prior Work
Tables 6 to 9 provide a comparison of all the templates used in the works presenting all methods we evaluate. Noticeably, prompt formats (and the choice of label words for some formats and datasets) used in works proposing investigated methods have no intersection. This is also concerning, since the original papers proposing these methods refer to each other. For instance, CHANNEL prompting outperforming CALIBRATION in Min et al. (2022b) might be explained by selecting a more favorable set of templates for the method proposed in the paper rather than by the advantages of the method itself.
# B Model Selection
Our initial evaluation pool consisted of 23 models. We evaluated each of them in 0-shot and 2shot settings with three prediction methods on four datasets, resulting in 12 runs. For each run in both 0-shot and 2-shot setups, we compare the model performance averaged over 10 random templates. Based on the results presented in Table 10, we restricted the final pool of models for evaluation to those that have a consistent increase in performance in the 2-shot setting, in other words, to those demonstrating a performance boost from ICL. More specifically, we kept the models that had 8 or more wins in 2-shot evaluation against 0-shot.
# C Full Baseline Results
Table 11 shows the results of evaluation of all 19 models in the default setting with a varying number of few-shot examples. These results illustrate that the template sensitivity issue is present in all models regardless of their size, and is not efficiently mitigated with the increase in the number of demonstrations.
# D Template Parts Analysis
In addition to studying prompt format sensitivity in general, we analyze how each part of a template impacts model performance. For instance, it could be possible that the inclusion of a certain verbalizer in a template consistently leads to a decline in accuracy, irrespective of the other components. To find that out, we decompose all templates into their parts and measure the distribution of scores for different variations of each component separately. The results presented in Figure 7 illustrate
that even for state-of-the-art models, such as Llama 2 70B and Falcon 40B, many components exhibit high variance; also, the variance differs between two models. In other words, even if a certain template yields good performance and low variance for a given setup, it is not guaranteed to work consistently well in other setups, and changing a single component could have detrimental effects. Along with the non-transferability of whole templates, we notice that individual components also do not transfer both between models and prediction methods. For instance, while “It was {}” ranks highest among output verbalizers for Llama 2 70B with the DIRECT prediction method, it is one of the worst for Falcon 40B. Moreover, while a combination of best verbalizers is often a well-performing template, it is not necessarily the best one; the same is applicable for “bad” verbalizers too. For example, “input: {}\n sentiment: {}\n\n” is the best template for Falcon 40B with the DIRECT method, even though “sentiment: {}” is one of the “worst” output verbalizers for that model. In summary, there is a complex interaction between the components of a template and their influence on model performance. We hypothesize that the transfer of both whole prompt templates and their parts is limited and requires further analysis.
# E Prediction Methods
We provide the results of advanced prediction methods evaluation for all models in 0-shot and 2-shot setting with random demonstrations in Table 10. We conclude from this comparison that neither of the advanced prediction strategies do not decrease prompt format sensitivity consistently across models and datasets. Moreover, when accounting for the spread in accuracy scores caused by this issue, the advantages of these methods over DIRECT become less apparent.
# F Reproduction of Results for Advanced Selection Methods
To evaluate the sensitivity of example selection methods to the template choice, we compare how the results reported in original works on these methods change when evaluated on a set of random templates instead of a predefined single one. For an accurate reproduction of original setups, we evaluate both Z-ICL and ITM using CHANNEL prompting with corresponding templates from Tables 6 and 7.
Method
Input verbalizer
Output verbalizer
Intra-sep
Inter-sep
Label words
ITM
“sentence: {}”
“{}”
“ "
“ ”
negative, positive
z-ICL
“Review: {}”
“Sentiment: {}”
“\n”
“\n\n\n”
terrible, great
Channel
“{}”
“A {} one”
“ "
“ ”
terrible, great
“{}”
“It was {}.”
“ ”
“ ”
terrible, great
“{}”
“All in all {}.”
“ ”
“ ”
terrible, great
“{}”
“A {} piece.”
“ ”
“ ”
terrible, great
Calibration
“Review: {}”
“Answer: {}”
“\n”
“\n\n”
Negative, Positive
“Review: {}”
“Answer: {}”
“\n”
“\n\n”
bad, good
“Review: {}”
“Positive review? {}”
“\n”
“\n\n”
No, Yes
“Input: {}”
“Sentiment: {}”
“\n”
“\n\n”
Negative, Positive
“Review: {}”
“Positive: {}”
“\n”
“\n\n”
False, True
“My review for last
night’s film: {}”
“The critics agreed that
this movie was {}”
“ ”
“\n\n”
bad, good
“One of our critics
wrote {}”
“Her sentiment towards
the film was {}”
“ ”
“\n\n”
Negative, Positive
“In a contemporary
review, Roger Ebert
wrote {}.”
“Entertainment Weekly
agreed, and the overall
critical reception of
the film was {}”
“ ”
“\n\n”
bad, good
“Review: {}”
“Question: Is the
sentiment of the
above review
Positive or Negative?
\nAnswer: {}”
“\n”
“\n\n”
Negative, Positive
“Review: {}”
“Question: Did the
author think that the
movie was good or
bad?\nAnswer: {}”
“\n”
“\n\n”
bad, good
“Question: Did the
author of the
following tweet
think that the
movie was good
or bad?\nTweet: {}”
“Answer: {}”
“\n”
“\n\n”
bad, good
“{}”
“My overall feeling
was that the
movie was {}”
“ ”
“\n\n”
bad, good
“{}”
“I {} the movie.”
“ ”
“\n\n”
hated, liked
“{}”
“My friend asked
me if I would
give the movie 0 or
5 stars, I said {}”
“ ”
“\n\n”
0, 5
Table 6: All templates used in methods we evaluate for SST-2 dataset
Method
Input verbalizer
Output verbalizer
Intra-sep
Inter-sep
Label words
Channel
“{}”
“{}”
“{}”
“{}”
“Topic: {}.”
“Subject: {}.”
“This is about {}.”
“It is about {} one.”
“ ”
“ ”
“ ”
“ ”
“ ”
“ ”
“ ”
“ ”
Company,
Educational Institution,
Artist, Athlete,
Office Holder,
Building,
Natural Place, Village,
Animal, Plant,
Album, Film,
Written Work,
Mean of Transportation
ITM
“{}”
“{}”
“ ”
“ ”
Same as above
Calibration
“Classify the
documents based
on whether they
are about
a [Label words]
\n\n Article: {}”
“Answer: {}”
“\n”
“\n\n”
Company, School,
Artist, Athlete,
Politician,
Building, Nature,
Village, Animal,
Plant, Album,
Film, Book,
Transportation
Table 7: All templates used in methods we evaluate for DBPedia dataset.
<div style="text-align: center;">Table 7: All templates used in methods we evaluate for DBPedia dataset.</div>
Method
Input verbalizer
Output verbalizer
Intra-sep
Inter-sep
Label words
Channel
“{}”
“{}”
“ ”
“ ”
Description, Entity,
Expression, Human,
Location, Number
“{}”
“Q: {}."
“ ”
“ ”
“{}”
“Why {}?"
“ ”
“ ”
“{}”
“Answer: {}”
“ ”
“ ”
Calibration
“Classify the
questions based
on whether their
answer type is
a [Label words]\n\n
Question: {}”
“Answer Type: {}”
“\n”
“\n\n”
Number, Location,
Person, Description,
Entity, Abbreviation
Table 8: All templates used in methods we evaluate for TREC dataset.
Method
Input verbalizer
Output verbalizer
Intra-sep
Inter-sep
Label words
Channel
“{}”
“Topic: {}."
“ ”
“ ”
World, Sports,
Business, Technology
“{}”
“Subject: {}."
“ ”
“ ”
“{}”
“This is about {}."
“ ”
“ ”
“{}”
“It is about {} one."
“ ”
“ ”
Calibration
“Article: {}”
“Answer: {}”
“\n”
“\n\n”
Same as above.
Model
N
SST-2
DBPedia
AGNews
TREC
2-shot
wins
Direct Channel Calib.
Direct Channel Calib.
Direct Channel Calib.
Direct Channel Calib.
0 0.650.09 0.720.04 0.700.06 0.340.13 0.400.07 0.490.09 0.480.16 0.560.04 0.640.14 0.250.06 0.280.11 0.300.08
GPT-2 Large
2 0.590.10 0.700.12 0.620.08 0.140.08 0.530.10 0.580.10 0.320.12 0.580.07 0.570.09 0.260.09 0.290.08 0.300.06 6/12
0 0.760.04 0.730.05 0.700.09 0.400.05 0.430.08 0.540.09 0.520.09 0.560.05 0.650.08 0.230.03 0.240.07 0.260.05
GPT-2 XL
2 0.580.11 0.710.09 0.630.11 0.150.09 0.540.09 0.500.15 0.400.20 0.610.08 0.560.15 0.260.07 0.340.09 0.330.08 5/12
GPT-J
0 0.710.09 0.680.08 0.680.08 0.410.07 0.440.06 0.570.08 0.610.08 0.640.03 0.640.07 0.320.05 0.200.07 0.330.04 9/12
2 0.650.14 0.770.11 0.680.11 0.250.16 0.680.06 0.710.16 0.470.19 0.670.09 0.730.11 0.260.07 0.320.09 0.330.06
GPT-NeoX
0 0.710.08 0.670.06 0.700.09 0.480.04 0.420.05 0.600.07 0.670.06 0.560.04 0.580.07 0.300.08 0.220.06 0.320.05 9/12
2 0.690.15 0.820.06 0.790.12 0.320.19 0.670.05 0.720.16 0.520.22 0.670.10 0.700.13 0.310.08 0.320.07 0.360.08
0 0.780.07 0.680.07 0.790.07 0.410.05 0.330.08 0.570.12 0.490.07 0.600.01 0.670.07 0.270.04 0.170.06 0.240.03
OPT 1.3B
2 0.690.15 0.800.06 0.710.16 0.210.11 0.580.08 0.610.12 0.480.24 0.660.06 0.610.11 0.270.08 0.380.09 0.350.08 6/12
OPT 6.7B
0 0.790.07 0.670.07 0.800.07 0.460.04 0.490.05 0.610.06 0.590.08 0.610.06 0.640.07 0.240.04 0.270.09 0.330.02 9/12
2 0.670.16 0.810.06 0.720.19 0.270.14 0.690.05 0.710.17 0.450.17 0.690.09 0.700.14 0.270.08 0.340.08 0.340.07
OPT 30B
0 0.790.06 0.720.05 0.770.08 0.480.04 0.480.07 0.610.08 0.640.05 0.600.06 0.650.11 0.240.03 0.260.05 0.310.01 8/12
2 0.640.17 0.790.09 0.730.17 0.340.21 0.730.05 0.780.14 0.550.19 0.690.09 0.760.11 0.310.06 0.350.09 0.330.06
OPT 66B
0 0.730.12 0.730.07 0.740.10 0.410.03 0.480.07 0.610.09 0.640.07 0.580.06 0.620.07 0.260.03 0.230.06 0.310.05 8/12
2 0.650.15 0.810.08 0.760.16 0.340.16 0.770.06 0.810.15 0.450.17 0.740.05 0.700.14 0.280.07 0.380.08 0.340.07
BLOOM 1.7B 0 0.680.11 0.670.06 0.680.11 0.470.03 0.470.06 0.470.07 0.610.08 0.530.04 0.580.06 0.270.04 0.240.08 0.330.03 9/12
2 0.660.12 0.750.06 0.710.10 0.270.19 0.620.08 0.570.13 0.430.19 0.590.08 0.610.11 0.310.09 0.390.07 0.370.08
BLOOM 3B
0 0.710.10 0.710.06 0.700.08 0.390.06 0.400.07 0.480.05 0.660.02 0.480.06 0.600.08 0.220.06 0.200.07 0.200.06 9/12
2 0.720.14 0.770.09 0.770.10 0.270.21 0.670.06 0.570.14 0.450.19 0.620.07 0.670.13 0.340.09 0.350.08 0.360.09
BLOOM 7.1B 0 0.720.09 0.710.06 0.680.06 0.440.05 0.450.08 0.510.08 0.640.06 0.560.04 0.640.10 0.350.07 0.220.08 0.320.04 9/12
2 0.690.15 0.760.09 0.760.11 0.260.18 0.700.06 0.670.14 0.430.17 0.690.06 0.680.12 0.330.08 0.340.07 0.360.06
Pythia 6.9B
0 0.750.08 0.720.05 0.690.11 0.450.05 0.430.04 0.630.09 0.580.14 0.590.04 0.640.08 0.310.07 0.210.07 0.320.03 8/12
2 0.630.12 0.780.09 0.770.11 0.280.16 0.670.08 0.680.14 0.430.17 0.680.09 0.690.14 0.340.09 0.370.07 0.380.06
Pythia 12B
0 0.730.07 0.710.08 0.690.10 0.430.05 0.430.04 0.510.18 0.610.09 0.570.05 0.650.09 0.330.06 0.230.05 0.320.03 8/12
2 0.630.13 0.790.10 0.740.12 0.290.15 0.680.07 0.710.14 0.530.18 0.680.08 0.700.12 0.290.09 0.350.06 0.330.08
LLaMA 7B
0 0.770.08 0.700.07 0.740.12 0.460.04 0.530.05 0.550.10 0.720.05 0.650.06 0.660.06 0.340.04 0.250.07 0.300.03 8/12
2 0.720.17 0.830.07 0.830.11 0.380.21 0.760.06 0.730.13 0.610.24 0.750.08 0.720.13 0.290.10 0.380.07 0.380.11
LLaMA 13B
0 0.810.03 0.690.07 0.770.08 0.420.04 0.520.07 0.650.11 0.740.03 0.620.07 0.730.04 0.340.04 0.180.05 0.340.03 10/12
2 0.750.17 0.830.08 0.820.14 0.380.17 0.750.06 0.800.12 0.680.15 0.750.07 0.800.08 0.350.09 0.360.08 0.420.09
LLaMA 30B
0 0.760.08 0.710.07 0.760.08 0.510.03 0.470.09 0.670.08 0.750.04 0.660.05 0.740.06 0.330.08 0.210.05 0.300.04 10/12
2 0.780.17 0.810.10 0.830.14 0.430.19 0.760.09 0.800.09 0.650.22 0.740.13 0.780.07 0.340.11 0.410.08 0.430.13
LLaMA 65B
0 0.780.10 0.710.05 0.750.10 0.450.05 0.490.07 0.620.08 0.740.06 0.610.07 0.740.03 0.310.06 0.190.07 0.310.03 11/12
2 0.820.17 0.840.09 0.870.13 0.450.17 0.780.06 0.800.13 0.680.20 0.780.08 0.820.05 0.380.08 0.380.09 0.450.11
Falcon 1B
0 0.720.08 0.720.03 0.730.07 0.540.03 0.550.04 0.620.10 0.680.04 0.640.06 0.630.08 0.240.04 0.250.04 0.310.02 9/12
2 0.650.17 0.770.10 0.710.17 0.360.25 0.720.05 0.740.14 0.520.17 0.720.08 0.770.08 0.260.09 0.330.06 0.330.06
Falcon 7B
0 0.720.09 0.680.05 0.730.08 0.500.06 0.510.13 0.660.06 0.750.06 0.640.03 0.720.06 0.310.04 0.210.07 0.290.03 10/12
2 0.770.16 0.780.09 0.790.15 0.400.21 0.760.06 0.800.17 0.510.20 0.760.07 0.730.12 0.320.09 0.330.08 0.370.11
Falcon 40B
0 0.760.05 0.680.07 0.740.11 0.450.03 0.570.07 0.690.08 0.750.07 0.620.07 0.720.08 0.310.07 0.270.10 0.270.02 11/12
2 0.790.17 0.810.09 0.870.13 0.420.15 0.830.06 0.850.12 0.640.23 0.790.09 0.800.06 0.360.07 0.410.06 0.450.06
0 0.700.12 0.590.08 0.620.16 0.350.04 0.290.09 0.210.22 0.680.04 0.450.10 0.410.23 0.300.06 0.130.05 0.150.16
Llama 2 7B
2 0.660.13 0.690.10 0.660.16 0.140.11 0.170.14 0.160.18 0.370.14 0.400.11 0.420.20 0.260.09 0.290.09 0.210.19 5/12
Llama 2 13B
0 0.770.09 0.710.04 0.740.10 0.450.03 0.590.05 0.630.10 0.750.07 0.660.05 0.760.05 0.330.03 0.270.07 0.340.03 9/12
2 0.790.17 0.820.10 0.880.09 0.400.15 0.790.06 0.830.11 0.700.15 0.760.09 0.810.05 0.320.09 0.350.11 0.460.11
Llama 2 70B
0 0.850.05 0.680.08 0.770.10 0.480.04 0.530.06 0.720.13 0.780.06 0.640.05 0.770.06 0.340.03 0.180.07 0.340.04 11/12
2 0.830.14 0.830.08 0.880.13 0.460.15 0.790.10 0.840.12 0.760.14 0.780.09 0.810.09 0.410.07 0.410.10 0.510.06
Highest mean, %
37.5
32.5
30.0
5.0
12.5
82.5
32.5
10.0
57.5
25.0
15.0
60.0
Lowest std, %
12.5
80.0
7.5
42.5
55.0
2.5
20.0
62.5
17.5
17.5
25.0
57.5
Table 10: Evaluation of advanced prediction methods for all models on 4 datasets in 0-shot and 2-shot with random
able 10: Evaluation of advanced prediction methods for all models on 4 datasets in 0-shot and 2-shot with random emonstrations. Models that were removed from further evaluation are highlighted in gray. “Calib.” stands for the alibration prompting method.
Model
SST-2
DBPedia
AGNews
TREC
0
2
4
0
2
4
0
2
4
0
2
4
GPT-J
0.710.09 0.650.14 0.660.14 0.410.07 0.250.16 0.340.18 0.610.08 0.470.19 0.480.23 0.320.05 0.260.07 0.360.11
GPT-NeoX
0.710.08 0.690.15 0.820.12 0.480.04 0.320.19 0.370.21 0.670.06 0.520.22 0.480.22 0.300.08 0.310.08 0.400.10
OPT 6.7B
0.790.07 0.670.16 0.800.14 0.460.04 0.270.14 0.330.18 0.590.08 0.450.17 0.470.22 0.240.04 0.270.08 0.340.10
OPT 30B
0.790.06 0.640.17 0.790.14 0.480.04 0.340.21 0.390.19 0.640.05 0.550.19 0.610.18 0.240.03 0.310.06 0.340.10
OPT 66B
0.730.12 0.650.15 0.840.13 0.410.03 0.340.16 0.400.16 0.640.07 0.450.17 0.530.19 0.260.03 0.280.07 0.330.09
BLOOM 1.7B 0.680.11 0.660.12 0.670.13 0.470.03 0.270.19 0.310.20 0.610.08 0.430.19 0.420.19 0.270.04 0.310.09 0.360.11
BLOOM 3B
0.710.10 0.720.14 0.760.12 0.390.06 0.270.21 0.330.21 0.660.02 0.450.19 0.460.22 0.220.06 0.340.09 0.390.12