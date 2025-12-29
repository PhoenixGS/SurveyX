# Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
Sewon Min1,2 Xinxi Lyu1 Ari Holtzman1 Mikel Artetxe2 Mike Lewis2 Hannaneh Hajishirzi1,3 Luke Zettlemoyer1,2 1University of Washington 2Meta AI 3Allen Institute for AI {sewon,alrope,ahai,hannaneh,lsz}@cs.washington.edu {artetxe,mikelewis}@meta.com
# Abstract
Large language models (LMs) are able to incontext learn—perform a new task via inference alone by conditioning on a few inputlabel pairs (demonstrations) and making predictions for new inputs. However, there has been little understanding of how the model learns and which aspects of the demonstrations contribute to end task performance. In this paper, we show that ground truth demonstrations are in fact not required—randomly replacing labels in the demonstrations barely hurts performance on a range of classification and multi-choce tasks, consistently over 12 different models including GPT-3. Instead, we find that other aspects of the demonstrations are the key drivers of end task performance, including the fact that they provide a few examples of (1) the label space, (2) the distribution of the input text, and (3) the overall format of the sequence. Together, our analysis provides a new way of understanding how and why in-context learning works, while opening up new questions about how much can be learned from large language models through inference alone.
arXiv:2202.12837v2
# 1 Introduction
Large language models (LMs) have shown impressive performance on downstream tasks by simply conditioning on a few input-label pairs (demonstrations); this type of inference has been referred to as in-context learning (Brown et al., 2020). Despite incontext learning consistently outperforming zeroshot inference on a wide range of tasks (Zhao et al., 2021; Liu et al., 2021), there is little understanding of how it works and which aspects of the demonstrations contribute to end task performance. In this paper, we show that ground truth demonstrations are in fact not required for effective incontext learning (Section 4). Specifically, replacing the labels in demonstrations with random labels barely hurts performance in a range of classification and multi-choice tasks (Figure 1). The result
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c659/c6598dc5-b7dc-44fe-992e-4b1fe366d062.png" style="width: 50%;"></div>
Figure 1: Results in classification (top) and multichoice tasks (bottom), using three LMs with varying size. Reported on six datasets on which GPT-3 is evaluated; the channel method is used. See Section 4 for the full results. In-context learning performance drops only marginally when labels in the demonstrations are replaced by random labels.
is consistent over 12 different models including the GPT-3 family (Radford et al., 2019; Min et al., 2021b; Wang and Komatsuzaki, 2021; Artetxe et al., 2021; Brown et al., 2020). This strongly suggests, counter-intuitively, that the model does not rely on the input-label mapping in the demonstrations to perform the task. Further analysis investigates which parts of demonstrations actually do contribute to the performance. We identify possible aspects of demonstrations (e.g., the label space and the distribution of the input text) and evaluate a series of variants of the demonstrations to quantify the impact of each (Section 5). We find that: (1) the label space and the distribution of the input text specified by the demonstrations are both key to in-context learning (regardless of whether the labels are correct for individual inputs); (2) specifying the overall format is also crucial, e.g., when the label space is unknown, using random English words as labels is significantly better than using no labels; and
(3) meta-training with an in-context learning objective (Min et al., 2021b) magnifies these effects—the models almost exclusively exploit simpler aspects of the demonstrations like the format rather than the input-label mapping. In summary, our analysis provides a new way of understanding the role of the demonstrations in in-context learning. We empirically show that the model (1) counter-intuitively does not rely on the ground truth input-label mapping provided in the demonstrations as much as we thought (Section 4), and (2) nonetheless still benefits from knowing the label space and the distribution of inputs specified by the demonstrations (Section 5). We also include a discussion of broader implications, e.g., what we can say about the model learning at test time, and avenues for future work (Section 6).  =
# 2 Related Work
Large language models have been key to strong performance in a wide range of downstream tasks (Devlin et al., 2019; Radford et al., 2019; Liu et al., 2019; Raffel et al., 2020; Lewis et al., 2020). While finetuning has been a popular approach to transfer to new tasks (Devlin et al., 2019), it is often impractical to finetune a very large model (e.g. ≥10B parameters). Brown et al. (2020) propose in-context learning as an alternative way to learn a new task. As depicted in Figure 2, the LM learns a new task via inference alone by conditioning on a concatenation of the training data as demonstrations, without any gradient updates. In-context learning has been the focus of significant study since its introduction. Prior work proposes better ways of formulating the problem (Zhao et al., 2021; Holtzman et al., 2021; Min et al., 2021a), better ways of choosing labeled examples for the demonstrations (Liu et al., 2021; Lu et al., 2021; Rubin et al., 2021), meta-training with an explicit in-context learning objective (Chen et al., 2021; Min et al., 2021b), and learning to follow instructions as a variant of in-context learning (Mishra et al., 2021b; Efrat and Levy, 2020; Wei et al., 2022a; Sanh et al., 2022). At the same time, some work reports brittleness and oversensitivity for in-context learning (Lu et al., 2021; Zhao et al., 2021; Mishra et al., 2021a). Relatively less work has been done to understand why in-context learning works. Xie et al. (2022) provide theoretical analysis that in-context learning can be formalized as Bayesian inference that
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6813/681362c6-686d-4ef8-ae4b-e0cbdf56728d.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7156/7156b6a7-db7d-489a-afe8-71b8a08ff406.png" style="width: 50%;"></div>
<div style="text-align: center;">Positive Prediction</div>
Figure 2: An overview of in-context learning. The demonstrations consist of k input-label pairs from the training data (k = 3 in the figure).
Model
# Params
Public
Meta-trained
GPT-2 Large
774M


MetaICL
774M


GPT-J
6B


fairseq 6.7B†
6.7B


fairseq 13B†
13B


GPT-3
175B‡


Table 1: A list of LMs used in the experiments: GPT-2 (Radford et al., 2019), MetaICL (Min et al., 2021b), GPT-J (Wang and Komatsuzaki, 2021), fairseq LMs (Artetxe et al., 2021) and GPT-3 (Brown et al., 2020). ‘Public’ indicates whether the model weights are public; ‘Meta-trained’ indicates whether the model is meta-trained with an in-context learning objective. †We use dense models in Artetxe et al. (2021) and refer them as fairseq LMs for convenience. ‡We use the Davinci API (the base version, not the instruct version) and assume it to be 175B, following Gao et al. (2021) and Artetxe et al. (2021).
uses the demonstrations to recover latent concepts. Razeghi et al. (2022) show that in-context learning performance is highly correlated with term frequencies in the pretraining data. To the best of our knowledge, this paper is the first that provides an empirical analysis that investigates why in-context learning achieves performance gains over zero-shot inference. We find that the ground truth input-label mapping in the demonstrations has only a marginal effect, and measure the impact of finer-grained aspects of the demonstrations.
# 3 Experimental Setup
We describe the experimental setup used in our analysis (Section 4 and 5).
Models. We experiment with 12 models in total. We include 6 language models (Table 1), all of which are decoder-only, dense LMs. We use each LM with two inference methods, direct and channel, following Min et al. (2021a). The sizes of LMs vary from 774M to 175B. We include the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9e29/9e29f639-ce25-4545-96e2-dec9b206c3ad.png" style="width: 50%;"></div>
Figure 3: Results when using no-demonstrations, demonstrations with gold labels, and demonstrations with ran dom labels in classification (top) and multi-choice tasks (bottom). The first eight models are evaluated on 16 classification and 10 multi-choice datasets, and the last four models are evaluated on 3 classification and 3 mult choice datasets. See Figure 11 for numbers comparable across all models. Model performance with random labels is very close to performance with gold labels (more discussion in Section 4.1).
largest dense LM (GPT-3) and the largest publicly released dense LM (fairseq 13B) at the time of conducting experiments. We also include MetaICL, which is initialized from GPT-2 Large and then meta-trained on a collection of supervised datasets with an in-context learning objective, and ensure that our evaluation datasets do not overlap with those used at meta-training time.
Evaluation Data. We evaluate on 26 datasets, including sentiment analysis, paraphrase detection, natural language inference, hate speech detection, question answering, and sentence completion (full list and references provided in Appendix A).1 All datasets are classification and multi-choice tasks. We use these datasets because they (1) are true low-resource datasets with less than 10K training examples, (2) include well-studied benchmarks from GLUE (Wang et al., 2018) and SuperGLUE (Wang et al., 2019a), and (3) cover diverse domains including science, social media, finance, and more.
Other Details. We use k = 16 examples as demonstrations by default for all experiments in the paper, unless otherwise specified. Examples are sampled at uniform from the training data. We choose a set of k training examples using 5 different random seeds and run experiments 5 times. For fairseq 13B and GPT-3, due to limited resources, we experiment with a subset of 6
datasets2 and 3 random seeds. We report MacroF13 for classification tasks and Accuracy for multichoice tasks. We compute per-dataset average over seeds, and then report macro-average over datasets. We use the minimal templates in forming an input sequence from an example. We refer to Appendix B for more details. All experiments are reproducible from github.com/Alrope123/ rethinking-demonstrations.
# 4 Ground Truth Matters Little
# 4.1 Gold labels vs. random labels
To see the impact of correctly-paired inputs and labels in the demonstrations—which we call the ground truth input-label mapping—we compare the following three methods.4
No demonstrations is a typical zero-shot method that does not use any labeled data. A prediction is made via argmaxy∈CP(y|x), where x is the test input and C is a small discrete set of possible labels. Demonstrations w/ gold labels are used in a typical in-context learning method with k labeled examples (x1, y1)...(xk, yk). A concatenation of k input-label pairs is used to make a prediction via argmaxy∈CP(y|x1, y1...xk, yk, x).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/83b0/83b0c53d-a61c-44b5-8f38-5df7c21a8a15.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Results with varying number of correct labels in the demonstrations. Channel and Direct used for classification and multi-choice, respectively. Performance with no demonstrations (blue) is reported as a reference.</div>
Demonstrations w/ random labels are formed with random labels, instead of gold labels from the labeled data. Each xi (1 ≤ i ≤ k) is paired with ˜yi that is randomly sampled at uniform from C. A concatenation of (x1, ˜y1)...(xk, ˜yk) is then used to make a prediction via argmaxy∈CP(y|x1, ˜y1...xk, ˜yk, x). Results are reported in Figure 3. First, using the demonstrations with gold labels significantly improves the performance over no demonstrations,5 as it has been consistently found in much of prior work (Brown et al., 2020; Zhao et al., 2021; Liu et al., 2021). We then find that replacing gold labels with random labels only marginally hurts performance. The trend is consistent over nearly all models: models see performance drop in the range of 0–5% absolute. There is less impact in replacing labels in multi-choice tasks (1.7% on average) than in classification tasks (2.6% absolute). This result indicates that the ground truth inputlabel pairs are not necessary to achieve performance gains. This is counter-intuitive, given that correctly paired training data is critical in typical supervised training—it informs the model of the expected input-label correspondence required to perform the downstream task. Nonetheless, the models do achieve non-trivial performance on the downstream tasks. This strongly suggests that the models are capable of recovering the expected inputlabel correspondence for the task; however, it is not directly from the pairings in the demonstrations. It is also worth noting that there is particularly little performance drop in MetaICL: 0.1–0.9% absolute. This suggests that meta-training with an explicit in-context learning objective actually encourages the model to essentially ignore the input5There are some exceptions, e.g., in the classification tasks, Direct GPT-2, Direct GPT-J and Direct fairseq 6.7B models are not significantly better than random guessing on many datasets; Channel fairseq 13B has significantly better nodemonstrations performance compared to demonstrations with gold labels. We thus discuss the results from these models less significantly for the rest of analysis.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5f68/5f68707f-7037-4328-b92d-f25240965291.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Ablations on varying numbers of examples in the demonstrations (k). Models that are the best under 13B in each task category (Channel MetaICL and Direct GPT-J, respectively) are used.</div>
label mapping and exploit other components of the demonstrations (more discussion in Section 5.4). In Appendix C.2, we provide additional results showing that (1) selecting random labels from a true distribution of labels (instead of a uniform distribution) reduces the gap even further, and (2) the trends may depend on the dataset, although the overall trend is consistent over most datasets.
# 4.2 Ablations
For additional ablations, we experiment with 5 classification and 4 multi-choice datasets.6
Does the number of correct labels matter? To further examine the impact of correctness of labels in the demonstrations, we conduct an ablation study by varying the number of correct labels in the demonstrations. We evaluate “Demonstrations w/ a% correct labels” (0 ≤a ≤100) which consist of k × a/100 correct pairs and k × (1 −a/100) incorrect pairs (see Algorithm 1 in Appendix B). Here, a = 100 is the same as typical in-context learning, i.e., demonstrations w/ gold labels. Results are reported in Figure 4. Model performance is fairly insensitive to the number of correct labels in the demonstrations. In fact, always using incorrect labels significantly outperforms no-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/822e/822e4634-aa4b-43b8-b199-943237649343.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Results with minimal templates and manual templates. ‘+T’ indicates that manual templates are used. Channel and Direct used for classification and multi-choice, respectively.</div>
demonstrations, e.g., preserving 92%, 100% and 97% of improvements from using the demonstrations with MetaICL in classification, MetaICL in multi-choice, and GPT-J in multi-choice, respectively. In contrast, GPT-J in classification sees relatively significant performance drop with more incorrect labels, e.g., nearly 10% drop in performance when always using incorrect labels. Still, always using incorrect labels is significantly better than no demonstrations.
Is the result consistent with varying k? We study the impact of the number of input-label pairs (k) in the demonstrations. Results are reported in Figure 5. First, using the demonstrations significantly outperforms the no demonstrations method even with small k (k = 4), and performance drop from using gold labels to using random labels is consistently small across varying k, in the range of 0.8–1.6%.7 Interestingly, model performance does not increase much as k increases when k ≥8, both with gold labels and with random labels. This is in contrast with typical supervised training where model performance rapidly increases as k increases, especially when k is small. We hypothesize that larger labeled data is beneficial mainly for supervising the input-label correspondence, and other components of the data like the example inputs, example labels and the data format are easier to recover from the small data, which is potentially a reason for minimal performance gains from larger k (more discussion in Section 5).  =
Is the result consistent with better templates? While we use minimal templates by default, we also explore manual templates, i.e., templates that are manually written in a dataset-specific manner, taken from prior work (details in Appendix B). Figure 6 shows that the trend—replacing gold labels with random labels barely hurting performance— holds with manual templates. It is worth noting
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2ae4/2ae43a6f-a86f-4f77-8e94-c2f71d868a8d.png" style="width: 50%;"></div>
Figure 7: Four different aspects in the demonstrations: the input-label mapping, the distribution of the input text, the label space, and the use of input-label pairing as the format of the demonstrations.
that using manual templates does not always outperform using minimal templates.
# 5 Why does In-Context Learning work?
Section 4 shows that the ground truth input-label mapping in the demonstrations has little impact to performance gains from in-context learning. This section further examines what other aspects of the demonstrations lead to good performance of incontext learning. We identify four aspects of the demonstrations (x1, y1)...(xk, yk) that potentially provide learning signal (depicted in Figure 7).
As Section 4 does for the input-label mapping, we design a series of variants of the demonstrations that quantify the impact of each aspect in isolation (Section 5.1–5.3). We then additionally discuss the trend of the models meta-trained with an in-context learning objective (Section 5.4). For all experiments, models are evaluated on five classification
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6e07/6e070d9d-d667-4231-bc69-240862f6e647.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: Impact of the distribution of the inputs. Evaluated in classification (top) and multi-choice (bottom). The impact of the distribution of the input text can be measured by comparing ■and ■. The gap is substantial, with an exception in Direct MetaICL (discussion in Section 5.1).</div>
and four multi-choice datasets as in Section 4.2. See Appendix B and Table 4 for implementation details and example demonstrations, respectively.
# 5.1 Impact of the distribution of the input text
# 5.1 Impact of the distribution of the input text
We experiment with OOD demonstrations which include out-of-distribution (OOD) text instead of the inputs from unlabeled training data. Specifically, a set of k sentences {xi,rand}k i=1 are randomly sampled from an external corpus, and replace x1...xk in the demonstrations. This variant assesses the impact of the distribution of the input text, while keeping the label space and the format of the demonstrations.
Results. Figure 8 shows that using out-ofdistribution inputs instead of the inputs from the training data significantly drops the performance when Channel MetaICL, Direct GPT-J or Channel GPT-J are used, both in classification and multichoice, by 3–16% in absolute. In the case of Direct GPT-J in multi-choice, it is even significantly worse than no demonstrations. Direct MetaICL is an exception, which we think is the effect of meta-training (discussion in Section 5.4). This suggests that in-distribution inputs in the demonstrations substantially contribute to performance gains. This is likely because conditioning on the in-distribution text makes the task closer to language modeling, since the LM always conditioned on the in-distribution text during training.
# 5.2 Impact of the label space
We also experiment with demonstrations w/ random English words that use random English words as labels for all k pairs. Specifically, we
sample a random subset of English words Crand where |Crand| = |C|, and randomly pair ˜yi ∈Crand with xi. This variant assesses the impact of the label space, while keeping the distribution of the input text and the format of the demonstrations.
Results. Based on Figure 9, direct models and channel models exhibit different patterns. With direct models, the performance gap between using random labels within the label space and using random English words is significant, ranging between 5–16% absolute. This indicates that conditioning on the label space significantly contributes to performance gains. This is true even for multi-choice tasks where there is no fixed set of labels—we hypothesize that multi-choice tasks still do have a particular distribution of the choices (e.g., objects like “Bolts” or “Screws” in the OpenBookQA dataset) that the model uses. On the other hand, removing the output space does not lead to significant drop in the channel models: there is 0–2% drop in absolute, or sometimes even an increase. We hypothesize that this is because the channel models only condition on the labels, and thus are not benefiting from knowing the label space. This is in contrast to direct models which must generate the correct labels.
# 5.3 Impact of input-label pairing
Section 5.1 and 5.2 focus on variants which keep the format of the demonstrations as much as possible. This section explores variants that change the format. While there are many aspects of the format, we make minimal modifications to remove the pairings of inputs to labels. Specifically, we evaluate demonstrations with no labels where the LM is
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ca22/ca222287-c491-4535-8ea7-668090e678fe.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: Impact of the label space. Evaluated in classification (top) and multi-choice (bottom). The impact of the label space can be measured by comparing ■and ■. The gap is significant in the direct models but not in the channel models (discussion in Section 5.2).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e9d5/e9d5e8da-58a4-48df-8983-f2ed3b5016cd.png" style="width: 50%;"></div>
Figure 10: Impact of the format, i.e., the use of the input-label pairs. Evaluated in classification (top) and multichoice (bottom). Variants of demonstrations without keeping the format (■and ■) are overall not better than no demonstrations (■). Keeping the format is especially significant when it is possible to achieve substantial gains with the label space but without the inputs (■vs. ■in Direct MetaICL), or with the input distribution but without the labels (■vs. ■in Channel MetaICL and Channel GPT-J). More discussion in Section 5.3.
<div style="text-align: center;">Figure 10: Impact of the format, i.e., the use of the input-label pairs. Evaluated in classification (top) and multichoice (bottom). Variants of demonstrations without keeping the format (■and ■) are overall not better than no demonstrations (■). Keeping the format is especially significant when it is possible to achieve substantial gains with the label space but without the inputs (■vs. ■in Direct MetaICL), or with the input distribution but without he labels (■vs. ■in Channel MetaICL and Channel GPT-J). More discussion in Section 5.3.</div>
conditioned on the concatenation of x1...xk, and demonstrations with labels only where the LM is conditioned on the concatenation of y1...yk. These ablations provide the no-format counterparts of the ‘demonstrations with random English words’ and ‘demonstrations with OOD inputs’, respectively. Results. Based on Figure 10, removing the format is close to or worse than no demonstrations, indicating the importance of the format. This is likely because conditioning on a sequence of inputlabel pairs triggers the model to mimic the overall format and complete the new example as expected when the test input is given. More interestingly, keeping the format plays a significant role in retaining a large portion of performance gains by only using the inputs or only using the labels. For instance, with Direct MetaICL, it is possible to retain 95% and 82% of improvements from in-context learning (demonstrations
conditioned on the concatenation of x1...xk, and demonstrations with labels only where the LM is conditioned on the concatenation of y1...yk. These ablations provide the no-format counterparts of the ‘demonstrations with random English words’ and ‘demonstrations with OOD inputs’, respectively.
with gold labels) by simply sampling random sentences from a corpus and randomly pairing them with the label set (■in Figure 10) in classification and multi-choice, respectively. Similarly, with the channel models, it is possible to retain 82%, 87%, 86% and 75% of improvements from in-context learning by simply pairing each input from the unlabeled training data with a random English word (■in Figure 10) in MetaICL classification, GPTJ classification, MetaICL multi-choice and GPT-J multi-choice, respectively. For all of these cases, removing inputs instead of using OOD inputs, or removing labels instead of using random English words is significantly worse, indicating that keeping the format of the input-label pairs is key.
# 5.4 Impact of meta-training
Different from other models, MetaICL is trained with an in-context learning objective, in line with
recent work that uses multi-task training on a large collection of supervised datasets (called metatraining) for generalization to new tasks (Aghajanyan et al., 2021; Khashabi et al., 2020; Wei et al., 2022a; Sanh et al., 2022). We aim to better understand the role of this meta-training in relation with our findings by closely examining the result of MetaICL. In particular, we observe that the patterns we see so far are significantly more evident with MetaICL than with other models. For instance, the ground truth input-label mapping matters even less, and keeping the format of the demonstrations matters even more. There is nearly zero influence of the input-label mapping and the input distribution in Direct MetaICL, and the input-label mapping and the output space in Channel MetaICL. Based on this observation, we hypothesize that meta-training encourages the model to exclusively exploit simpler aspects of the demonstrations and to ignore others. This is based on our intuition that (1) the input-label mapping is likely harder to exploit, (2) the format is likely easier to exploit, and (3) the space of the text that the model is trained to generate is likely easier to exploit than the space of the text that the model conditions on.8
# 6 Discussion & Conclusion
In this paper, we study the role of the demonstrations with respect to the success of in-context learning. We find that the ground truth input-label mapping in the demonstrations matters significantly less than one might think—replacing gold labels with random labels in the demonstrations only marginally lowers the performance. We then identify a series of aspects in the demonstrations and examine which aspect actually contributes to performance gains. Results reveal that (1) gains are mainly coming from independent specification of the input space and the label space, (2) the models can still retain up to 95% of performance gains by using either the inputs only or the label set only if the right format is used, and (3) meta-training with an in-context learning objective magnifies these trends. Together, our findings lead to a set of broader indications about in-context learning, as well as avenues for future work.
Does the model learn at test time? If we take a strict definition of learning: capturing the input-
label correspondence given in the training data, then our findings suggest that LMs do not learn new tasks at test time. Our analysis shows that the model may ignore the task defined by the demonstrations and instead use prior from pretraining. However, learning a new task can be interpreted more broadly: it may include adapting to specific input and label distributions and the format suggested by the demonstrations, and ultimately getting to make a prediction more accurately. With this definition of learning, the model does learn the task from the demonstrations. Our experiments indicate that the model does make use of aspects of the demonstrations and achieve performance gains. Capacity of LMs. The model performs a downstream task without relying on the input-label correspondence from the demonstrations. This suggests that the model has learned the (implicit notion of) input-label correspondence from the language modeling objective alone, e.g., associating a positive review with the word ‘positive’. This is in line with Reynolds and McDonell (2021) who claim that the demonstrations are for task location and the intrinsic ability to perform the task is obtained at pretraining time.9 On one hand, this suggests that the language modeling objective has led to great zero-shot capacity, even if it is not always evident from the naive zero-shot accuracy. On the other hand, this suggests that in-context learning may not work on a task whose input-label correspondence is not already captured in the LM. This leads to the research question of how to make progress in NLP problems that in-context learning does not solve: whether we need a better way of extracting the input-label mappings that are already stored in the LM, a better variant of the LM objective that learns a wider range of task semantics, or explicit supervision through fine-tuning on the labeled data. Connection to instruction-following models.
Connection to instruction-following models. Prior work has found it promising to train the model that reads the natural language description of the task (called instructions) and performs a new task at inference (Mishra et al., 2021b; Efrat and Levy, 2020; Wei et al., 2022a; Sanh et al., 2022). We think the demonstrations and instructions largely have the same role to LMs, and hypothesize that our
findings hold for instruction-following models: the instructions prompt the model to recover the capacity it already has, but do not supervise the model to learn novel task semantics. This has been partially verified by Webson and Pavlick (2022) who showed that the model performance does not degrade much with irrelevant or misleading instructions. We leave more analysis on instruction-following models for future work.
Significantly improved zero-shot performance. One of our key findings is that it is possible to achieve nearly k-shot performance without using any labeled data, by simply pairing each unlabeled input with a random label and using it as the demonstrations. This means our zero-shot baseline level is significantly higher than previously thought.10 Future work can further improve the zero-shot performance with relaxed assumptions in access to the unlabeled training data.
# Limitation
Effect of types of tasks and datasets. This paper focuses on the tasks from established NLP benchmarks that have real natural language inputs. Synthetic tasks with more limited inputs may actually use the ground truth labels more, as observed by Rong (2021). We report macro-level analysis by examining the average performance over multiple NLP datasets, but different datasets may behave differently. Appendix C.2 discusses this aspect, including findings that there are larger gaps between using the ground truth labels and using the random labels in some dataset-model pairs (e.g., in the most extreme case, nearly 14% absolute on the financial_phrasebank dataset with GPT-J). Since the first version of our paper, Kim et al. (2022) showed that using negated labels substantially lowers the performance in classification.11 We believe it is important to understand to what extend the model needs the ground truth labels to successfully perform in-context learning.
Extensions to generation. Our experiments are limited to classification and multi-choice tasks. We hypothesize that ground truth output may not be
necessary for in-context learning in the open-set tasks such as generation, but leave this to future work. Extending of our experiments to such tasks is not trivial, because it requires a variation of the output which has incorrect input-output correspondence while keeping the correct output distribution (which is important based on our analysis in Section 5). Since the first version of our paper, Madaan and Yazdanbakhsh (2022) conducted a similar analysis with the chain of thought prompting (Wei et al., 2022b) which generates a rationale to perform complex tasks such as math problems. Madaan and Yazdanbakhsh (2022) show that, while simply using a random rationale in the demonstrations (e.g., pairing with a rationale from a different example) significantly degrades the performance, other types of counterfactual rationales (e.g., wrong equations) do not degrade the performance as much as we thought. We refer to Madaan and Yazdanbakhsh (2022) for more discussions on what aspects of the rationale matter or do not matter.
# Acknowledgements
We thank Gabriel Ilharco, Julian Michael, Ofir Press, UW NLP members and anonymous reviewers for their comments in the paper. This research was supported by NSF IIS-2044660, ONR N0001418-1-2826, a Sloan fellowship and gifts from AI2.
# References
Luisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. 2009. The fifth pascal recognizing textual entailment challenge. In TAC.
Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick von Platen, Suraj Patil, Julien Chaumond, Mariama Drame, Julien Plu, Lewis Tunstall, Joe Davison, Mario Šaško, Gunjan Chhablani, Bhavitvya Malik, Simon Brandeis, Teven Le Scao, Victor Sanh, Canwen Xu, Nicolas Patry, Angelina McMillan-Major, Philipp Schmid,
Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2021. Learning to retrieve prompts for in-context learning. arXiv preprint arXiv:2112.08633.
ictor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Stella Biderman, Leo Gao, Tali Bers, Thomas Wolf, and Alexander M. Rush. 2022. Multitask prompted training enables zero-shot task generalization. In ICLR.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022b. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.
# A Full Datasets
We include 26 datasets as follows: financial_phrasebank (Malo et al., 2014), poem_sentiment (Sheng and Uthus, 2020), medical_questions_pairs (McCreery et al., 2020), glue-mrpc (Dolan and Brockett, 2005), gluewnli (Levesque et al., 2012), climate_fever (Diggelmann et al., 2020), glue-rte (Dagan et al., 2005; Bar-Haim et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., 2009), supergluecb (de Marneffe et al., 2019), sick (Marelli et al., 2014) , hate_speech18 (de Gibert et al., 2018), ethos-national_origin (Mollas et al., 2020), ethosrace (Mollas et al., 2020), ethos-religion (Mollas et al., 2020), tweet_eval-hate (Barbieri et al., 2020), tweet_eval-stance_atheism (Barbieri et al., 2020), tweet_eval-stance_feminist (Barbieri et al., 2020), quarel (Tafjord et al., 2019a), openbookqa (Mihaylov et al., 2018), qasc (Khot et al., 2020), commonsense_qa (Talmor et al., 2019), ai2_arc (Clark et al., 2018), codah (Chen et al., 2019), supergluecopa (Gordon et al., 2012), dream (Sun et al., 2019), quartz-with_knowledge (Tafjord et al., 2019b), quartz-no_knowledge (Tafjord et al., 2019b). The choice of datasets is made following low-resource datasets in Min et al. (2021b), with the exact same set of k-shot train data using 5 random seeds. We use the HuggingFace version of the data (Lhoest et al., 2021) and use the development data for evaluation, following Ye et al. (2021). See Table 2 for statistics.
# B Experimental Details
Example template We follow Ye et al. (2021); Min et al. (2021b); Logan IV et al. (2021) in using the minimal format to transform the input to a sequence (e.g. a concatenation of multiple inputs) and using the label words from each dataset as it is. We also explore manual templates taken from prior work (Holtzman et al., 2021; Zhao et al., 2021) as reported in Section 4.2, although we find that using these templates is not consistently better than using minimal templates. We thus run main experiments with minimal templates. Example templates are provided in Table 3.
Format of the demonstrations We follow the standard of each model for formatting the demonstrations, either from exploration in prior work or the example code provided in the official tutorial. For GPT-2, we separate the input and the label,
Dataset
# Train
# Eval
Task category: Sentiment analysis
financial_phrasebank
1,811
453
poem_sentiment
892
105
Task category: Paraphrase detection
medical_questions_pairs
2,438
610
glue-mrpc
3,668
408
Task category: Natural language inference
glue-wnli
635
71
climate_fever
1,228
307
glue-rte
2,490
277
superglue-cb
250
56
sick
4,439
495
Task category: Hate speech detection
hate_speech18
8,562
2,141
ethos-national_origin
346
87
ethos-race
346
87
ethos-religion
346
87
tweet_eval-hate
8,993
999
tweet_eval-stance_atheism
461
52
tweet_eval-stance_feminist
597
67
Task category: Question answering
quarel
1,941
278
openbookqa
4,957
500
qasc
8,134
926
commonsense_qa
9,741
1,221
ai2_arc
1,119
299
Task category: Sentence completion
codah
1665
556
superglue-copa
400
100
dream
6116
2040
quartz-with_knowledge
2696
384
quartz-no_knowledge
2696
384
Table 2: 26 datasets used for experiments, classified into 6 task categories. # Train and # Test indicate the number of training and test examples of the dataset. Note that # train is based on the original training dataset but we use k random samples for k-shot evaluation.
and each demonstration example with a space. For MetaICL, GPT-J and GPT-3, we separate the input and the label with a newline (\n), and each demonstration example with three newlines. For fairseq models, we use a newline to separate the input and the label as well as each demonstration example.
“demonstrations w/ a% accurate labels” (0 ≤ a ≤100), we use k × a/100 correct pairs and k × (1 −a/100) incorrect pairs in a random order, as described in Algorithm 1. For “OOD demonstrations”, we use CC-News (Nagel, 2016) as an external corpus. We consider the length of the text during sampling, so that sampled sentences have similar length to the test input. For “demonstrations with random English words”, we use pypi.org/ project/english-words for the set of En-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7a58/7a583064-38c1-4a27-be22-255a3cb497a5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 11: Results of No-demonstration, Gold demonstration and Random demonstration on 3 classification datasets (top) and 3 multi-choice datasets (bottom). Details in Section 4.1. This figure is for providing numbers that are comparable across models—full results with more datasets are reported in Figure 3.</div>
Algorithm 1 Forming the demonstrations with an accuracy of a%.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/17fc/17fc6951-1a83-49f3-9699-4fd3e73b3261.png" style="width: 50%;"></div>
glish words, which consists of 61,569 words. Table 4 provides a list of example demonstrations for each method used in Section 5.
# C More Experimental Results
# C.1 Gold labels vs. random labels
Figure 11 shares the same interface as Figure 3, but all models are evaluated on 3 classification and 3 multi-choice datasets and are thus comparable to each other.
# C.2 Random labels from true distribution of labels & Task breakdown
In Section 4, random labels are sampled from the label space from a uniform distribution. We experiment with another variant of demonstrations in the classification tasks, where labels are randomly sampled from the true distribution of labels on the training data. This may have large impact if labels are far from uniform on the training data. Results indicate that performance drop from using gold
labels is further reduced compared to using uniformly random labels: with Channel MetaICL, the gap is reduced from 1.9% to 1.3% absolute, and with Channel GPT-J, the gap is reduced from 5.0% to 3.5% absolute. Figure 12 shows performance gap between using gold labels and using random labels per dataset. We find that the trend that the gap is smaller than previously thought is consistant across most datasets. Nonetheless, there are a few outlier datasets where performance gap is non-negligible, such as financial_phrasebank and a few hate speech detection datasets. Future work may investigate on which tasks the model makes more use of the correctly paired training data.
# C.3 More variants of the demonstrations
We explored demonstrations with a constant label where all labels in the demonstrations are replaced with a constant text, “answer”. Specifically, a prediction is made via argmaxy∈CP(y|x1, answer...xk, answer, x). This can be viewed as another way to remove the impact of the label space while keeping the impact of the distribution of the input text. However, results are consistently worse than the results of demonstrations with random English labels. We think this is because constant labels actually change the format of the demonstrations, since they can be viewed as part of a separator between different demonstration examples. We also explored demonstrations with the test input where all inputs in the demonstrations are replaced with the test input, each paired with a ran-
dom label. Specifically, a prediction is made via argmaxy∈CP(y|x, ˜y1...x, ˜yk, x), where ˜yi (1 ≤ i ≤k) is randomly sampled at uniform from C. This variant is seemingly a reasonable choice given that it satisfies the condition that the inputs in the demonstrations come from the same distribution as the test input (since they are identical), and using random labels is as good as using gold labels. Nonetheless, we find that this variant is significantly worse than most other methods with demonstrations. We think this is because using the constant input for all demonstration example significantly changes the format of the sequence, since the input can be viewed as part of a separator between different demonstration examples.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9530/9530646f-06e9-4d3e-bc08-e31372ec822c.png" style="width: 50%;"></div>
Figure 12: Performance gap from using the demonstrations with gold labels to using the demonstrations with random labels. Datasets are sorted in descending order. The top two figures use random labels that are sampled at uniform, with Channel MetaICL and Channel GPT-J, respectively. The bottom two figures use random labels that are sampled from a true distribution of labels on the training data, with Channel MetaICL and Channel GPT-J, respectively.
Dataset
Type
Example
MRPC
Minimal
sentence 1: Cisco pared spending to compensate for sluggish sales . [SEP] sentence 2: In
response to sluggish sales , Cisco pared spending . \n {equivalent|not_equivalent}
Manual
Cisco pared spending to compensate for sluggish sales . \n The question is: In response to
sluggish sales , Cisco pared spending . True or False? \n The answer is:{True|False}
RTE
Minimal
sentence 1: The girl was found in Drummondville. [SEP] sentence 2: Drummondville
contains the girl. \n {entailment|not_entailment}
Manual
The girl was found in Drummondville. \n The question is: Drummondville contains the
girl. True or False? \n The answer is:{True|False}
Tweet_eval-hate
Minimal
The Truth about #Immigration \n {hate|non-hate}
Manual
Tweet: The Truth about #Immigration \n Sentiment: {against|favor}
SICK
Minimal
sentence 1: A man is screaming. [SEP] sentence 2: A man is scared. \n
{contradiction|entailment|neutral}
Manual
A man is screaming. \n The question is: A man is scared. True or False? \n The answer is:
{False|True|Not sure}
poem-sentiment
Minimal
willis sneered: \n {negative|no_impact|positive}
Manual
willis sneered: \n The sentiment is: {negative|no_impact|positive}
OpenbookQA
Minimal
What creates a valley? \n {feet|rock|water|sand}
Manual
The question is: What creates a valley? \n The answer is: {feet|rock|water|sand}
CommonsenseQA
Minimal
What blocks sunshine? \n {summer|park|desktop|sea|moon}
Manual
The question is: What blocks sunshine? \n The answer is: {summer|park|desktop|sea|moon}
COPA
Minimal
Effect: I coughed. \n {Cause: I inhaled smoke.|Cause: I lowered my voice.}
Manual
I coughed because {I inhaled smoke.|I lowered my voice.}
ARC
Minimal
Which biome has the most vegetation? \n {desert|forest|grassland|tundra}
Manual
The question is: Which biome has the most vegetation? \n The answer is: {desert|forest|
grassland|tundra}
Table 3: A list of minimal templates taken from Ye et al. (2021); Min et al. (2021b) and manual templates taken from Holtzman et al. (2021); Zhao et al. (2021). Details provided in Appendix B. See Figure 6 for discussion in empirical results. The input and the label are in the red text and in the blue text, respectively. Note that | is used to separate different options for the labels.
Demos
w/ gold labels
(Format  Input distribution  Label space  Input-label mapping )
Circulation revenue has increased by 5% in Finland and 4% in Sweden in 2008. \n positive
Panostaja did not disclose the purchase price. \n neutral
Demos
w/ random labels
(Format  Input distribution  Label space  Input-label mapping )
Circulation revenue has increased by 5% in Finland and 4% in Sweden in 2008. \n neutral
Panostaja did not disclose the purchase price. \n negative
OOD Demos
w/ random labels
(Format  Input distribution  Label space  Input-label mapping )
Colour-printed lithograph. Very good condition. Image size: 15 x 23 1/2 inches. \n neutral
Many accompanying marketing claims of cannabis products are often well-meaning. \n negative
Demos
w/ random English words
(Format  Input distribution  Label space  Input-label mapping )
Circulation revenue has increased by 5% in Finland and 4% in Sweden in 2008. \n unanimity
Panostaja did not disclose the purchase price. \n wave
Demos
w/o labels
(Format  Input distribution  Label space  Input-label mapping )
Circulation revenue has increased by 5% in Finland and 4% in Sweden in 2008.
Panostaja did not disclose the purchase price.
Demos
labels only
(Format  Input distribution  Label space  Input-label mapping )
positive
neutral
Table 4: Example demonstrations when using methods in Section 5. The financial_phrasebank dataset with C = {“positive”, “neutral”, “negative”} is used. Red text indicates the text is sampled from an external corpus; blue text indicates the labels are randomly sampled from the label set; purple text indicates a random English word.
# Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
Anonymous ACL submission
We are glad that all reviewers find that the paper is novel (8jk5, LQ6N, 92YB, 7E5P), of interest to the broader NLP community (LQ6N, 92YB, 7E5P), supported by solid experiments (8jk5, LQ6N, 92YB, 7E5P), and well-written (8jk5, LQ6N, 92YB). Reviewers gave comments on more discussion, limitations and avenues for future work. We will incorporate them in the next version of the paper.1 Adding discussion on robustness of LMs (8jk5, 7E5P): We think the fact that LMs do not use the input-label correspondence does not necessarily mean that LMs are robust to other aspects of the demonstration. Nonetheless, it will be an interesting avenue for future work, given that LMs are highly sensitive to small changes in the demonstrations (??). Adding discussion with respect to the model size (8jk5): Absolute differences are similar across different model sizes (Figure 3), but since the large models have higher absolute performance, the relative differences are larger with larger models. When our findings hold or do not hold (8jk5): We believe that the findings will hold only when some notion of input-label correspondences are already captured during pre-training—for tasks whose input-label correspondences during pretraining is sparse, the demonstrations with random labels are highly unlikely to work. This has been shown by ? in a synthetic setup, and future work can revisit this with more natural data. Risk in applying in-context learning, Recommendation to practitioners (8jk5, LQ6N): Since the models do not capture the correspondence from the demonstrations, it is possible that models are simply relying on some notation of input-label correspondence during pre-training. Thus, applying in-context learning for problems that were not 1We answered reviewers’ questions on the OpenReview page, and address higher-level comments here.
in the pretraining data would be potentially risky, and practitioners may want to collect the labeled data and fine-tune the model.
in the pretraining data would be potentially risky, and practitioners may want to collect the labeled data and fine-tune the model. Where do the gains from demonstrations come from? (92YB): We think gains from demonstrations are mainly due to the specification of the input distribution and the label space rather than the input-label correspondence, as Section 5 indicates. We will clarify this in the next version of the paper. Concrete answer to “why” using random labels keeps reasonable performance (LQ6N): We think it is highly likely to be because the model has been exposed to some notion of input-label correspondence during pre-training, so that the demonstrations play a role of triggering them at inference. Consideration when training large LMs (LQ6N): Due to compute limitations, we were not be able to provide recommendations that are empirically supported. Future work may investigate aspects of language model pretraining that affect the findings, including the pretraining data and the objective. Word ordering matters? (7E5P): We think it is an important avenue for future work. For this paper, we did not include it in our scope due to requiring multiple times more experiments. More task types, e.g., generation (7E5P): Extending this work to generation tasks is not very trivial because designing the demonstrations that do not keep the input-output correspondence but keep the output distribution is difficult. For instance, if we simply replace the output with a random output as we did in the classification tasks, it will destroy both the input-output correspondence and the output distribution. We hope future work can investigate more in this direction. Stronger link with instruction-following models (7E5P): We will add discussion on ? who find that instructions that are irrelevant or even misleading lead to performance gains as much as “good”
# Limitation
This paper focuses on the tasks from established NLP benchmarks that have real natural language inputs. Synthetic tasks with more limited inputs may actually use the ground truth labels more, as observed by ?. Our paper mainly includes macrolevel analysis by examining the average performance over multiple NLP datasets, but different datasets may behave differently. Appendix discusses this aspect, including findings that there are larger gaps between using the ground truth labels and using the random labels in some dataset-model pairs (e.g., in the most extreme case, nearly 14% absolute on the financial_phrasebank dataset with GPT-J). We believe it is important to understand in which cases the ground truth labels matter or not, which we leave for future work. Furthermore, our work is limited to classification and multi-choice tasks. Extension to the open-set tasks such as generation is not trivial, since it is unclear how to remove the input-output correspondence while keeping the correct output distribution. We leave the extension for future work.
