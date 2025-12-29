# Efficient LLM Context Distillation
# Rajesh Upadhayaya, Zachary Smith, Christopher Kottmyer, Manish Raj Osti Georgia Institute of Technology
Rajesh Upadhayaya, Zachary Smith, Christopher Kottmyer, Manish Raj Osti Georgia Institute of Technology
upadraj@gatech.edu
# Abstract
Large Language Models (LLMs) demonstrate proficiency across diverse tasks but often require targeted adaptations for specific applications. Various methods have been proposed to facilitate this adaptation, including fewshot fine-tuning, in-context learning, and context distillation. This paper specifically investigates context distillation — a method that extends the utility of task-specific examples by internalizing them, thus augmenting the example set accessible for model inference. We conduct a comparative analysis of context distillation with in-context learning (ICL) and few-shot fine-tuning (FT), aiming to ascertain the efficacy of context distillation in adapting models using minimal in-context examples. Employing matched datasets from Mobach, our experiments leverage OPT models of various sizes. The results indicate that context distillation effectively adapts models, with student models attaining comparable in-domain and out-of-domain accuracies to in-context learning. Although context distillation surpasses ICL in out-of-domain generalization, it does not achieve the performance levels of FT. However, the reduced dataset size and computational demands position context distillation as a viable alternative, especially for smaller datasets. Overall, this study presents context distillation as an efficient and potent method for customizing LLMs to specific tasks.
arXiv:2409.01930v1
# 1. Introduction
Large language models (LLM) excel at knowledge extraction and reasoning, but often require adaptation to individual tasks. There are several proposed methods to perform this adaptation including, but not limited to, fewshot fine-tuning, in-context learning, and context distillation. Each method has its own advantages and disadvantages. For example, few-shot fine-tuning (FT) requires substantial amounts of task specific data, which poses a challenge when labeled examples are scarce. In-context learning (ICL) attempts to alleviate FT data needs by providing fewer examples through the query prompt used during inference. However, LLM have a constrained context win-
dow that is both consumed by task examples and limit the number of examples that models can learn from. Given this constrained context window, context distillation (CD) extends accessible task-specific examples by internalizing them, greatly increasing the number of available examples outside of the query prompt [1]. This not only limits the number of task examples these models can learn from simultaneously but also affects their ability to integrate and recall relevant information across different parts of the text. Context distillation (CD) addresses these limitations by allowing a model to internalize and condense key information from task-specific examples. This process effectively extends the usable information beyond what is immediately present in the query prompt, increasing the number of examples the model can learn from and utilize, without being directly constrained by the size of the context window. In this paper, we explore context distillation by directly comparing our results to the ICL results in Mosbach et al. [6]. Mosbach et al. show that ICL is a viable task adaptation method despite it under-performance relative to FT. A plausible explanation for this under-performance is that FT learns from many training examples whereas ICL learns from only a few in-context examples provided during inference. This results in ICL having less data for task generalization. By contrast, context distillation strikes a balance between the large number of examples required for FT and the small number of examples that fit into the context window of ICL. Additionally, Snell et al. shows that CD outperforms direct gradient descent learning [9]. Thus, it is hypothesized that context distillation should perform better than ICL without requiring as many training examples as FT. Our goal is to demonstrate that context distillation can be performant by using only a few in-context examples. This not only signifies efficient training data utilization but also showcases task-specific improvements relative to conventional fine-tuning methods trained on small datasets. When combined with the efficiency gains of low-rank adaptation layers (LoRa), context distillation becomes an efficient training method for learning task specific adaptations and provides increased flexibility. To accomplish this a fixed
reference model with task-specific LoRa layers is used during inference time. In terms of data, we employ matched datasets similar to those utilized in previous studies, ensuring consistency and comparability. Our datasets encompass a range of tasks, including natural language inference (NLI) and paraphrase identification. Datasets are sourced from widely available repositories such as Hugging Face. By leveraging these datasets, we provide a comprehensive evaluation of context distillation on a diverse set of tasks and domains. We examine how context distillation mitigates limitations in traditional fine-tuning by comparing our results to previous studies. We provide insights into the comparative efficacy of context distillation relative to FT on task adaption.
# 2. Approach
Overall, our approach was to implement context distillation by adapting methods used in Mosbach et al. [6]. This includes referencing and modifying their code to adapt it to our teacher-student training procedure and to retrieve their datasets for a comparative analysis. Adaption in this context means selectively using their code, changing it to be more efficient and flexible, and even fixing bugs encountered in their code. We note that the original code base was bloated and badly abstracted. By the end of the project, we completely rewrote sections of their code and used the original code to selectively validate our code. A significant part of validation was confirming we reconstructed their datasets to guarantee our results were comparable to theirs. For example, we needed to confirm that they binarized one of the datasets and make sure we replicated it in our dataset. Python was used to access PyTorch, a deep learning framework, as well as Hugging Face Transformers [11] and Datasets [3].
# 2.1. Datasets
The datasets were consistent with those used by Mosbach et al. [6]. Two common natural language processing tasks were used: natural language inference (NLI) and paraphrase identification. For the NLI task, MNLI [10] and RTE [2] where used for our in-domain datasets and the lexical overlap subset of HANS [4] was used as our out-of-domain (OOD) dataset. MNLI is binarized by removing neutral examples matching the labels in RTE and HANS datasets. For the paraphrase identification task, QQP [8] was used as our in-domain dataset and PAW-QQP [13] as our OOD. All datasets were configured to use yes labels for entailment or paraphrase and no labels otherwise. These datasets are used for natural language processing (NLP) and serve as common benchmarks. They are widely available through platforms such as Hugging Face. Each indomain dataset comes with training and validation subsets.
The training sets where first sampled to create in-context examples. They are then randomly sampled for our query examples which are used for inference. Care was taken to ensure that the in-context and query examples did not overlap. Regarding data preparation, both the context distillation teacher and student models are structurally the same, decoder only transformers, and so the expected inputs are tokenized strings. The output, after applying softmax, is a series of tokens that after using a model-specific tokenizer for decoding results in a string representation. Additionally, no pre- or post-processing was required since the datasets are used as NLP benchmarks.
# 2.2. Models
All experiments were run using 4 OPT models [12] - 125 million, 350 million, 1.3 billion, and 2.7 billion parameters. Using the OPT family of models guarantees that the models where trained on the same dataset. This isolates the impact of model size from a possible confounding factor of the training datasets used.
# 2.3. Context Distillation Setup
The context distillation was set-up using both a teacher and a student model tasked to infer a label for an inference request. The teacher model receives a string of in-context examples followed by an inference request without a label. The student model receives only the inference request. Both models generate an answer and the difference between the two answers, via KL-divergence loss, is used to update the student model weights. The teacher model weights stay frozen. To generate training contexts for the teacher model, we randomly sampled the in-domain training set, varying the number of context examples n ∈{2, 16, 32} . This process is repeated four times for each n, resulting in four unique context example sets per n. However, for the RTE task, n = 32 consistently exceeded the context window, so we could not run it. All in-context examples and inference requests were formatted according to the pattern in Table 1. During training, each set of context examples served as the training context for a single run per model. During that run, we fine-tuned the student model on 32 inference requests, which were randomly sampled from the in-domain training set. To prevent data leakage, none of the 32 inference requests overlapped with the teacher’s context examples.
# 2.4. Fine-tuning
The same fine-tuning process was used for all models and tasks. Each context distillation run used an in-domain dataset for training and was validated using both the indomain and OOD datasets for that task. The last token of
Type
Dataset(s)
Pattern text
Answer prefix
Target tokens
General pattern
All
Premise: {premise} \n Hypothesis: {hypothesis}
\n
Label: {label}
Yes, No
Teacher pattern
MNLI,
RTE
Think logically.
Are the following sentences
examples of entailment, yes or no?\n {context
example n1}\n\n {context example n2}\n\n
...{context example nn}\n\n {inference exam-
ple}
Label:
Yes, No
Teacher pattern
QQP
Think logically.
Are the following sentences
duplicates or paraphrases of each other, yes
or no?\n {context example n1}\n\n {context
example n2}\n\n ...{context example nn}\n\n
{inference example}
Label:
Yes, No
Student pattern
MNLI,
RTE
Are the following sentences examples of entail-
ment, yes or no?\n {inference example}
Label:
Yes, No
Student pattern
QQP
Are the following sentences duplicates or para-
phrases of each other, yes or no?\n {inference ex-
ample}
Label:
Yes, No
Table 1: Patterns used for context distillation. All context examples and inference examples are formatted using the general pattern. Context examples include the answer after the answer prefix. The inference examples do not.
the teacher and student model outputs were assumed to represent the label. KL divergence between the teacher and student output was used to update the student LoRa layers’ weights. Since both the teacher and student model are the same size model, LoRa adaptors where utilized. The implementation of the low-rank adaptors require that the pre-trained model parameters are frozen, creating a reference model. A low-rank adaptation matrix layer is then applied on top of the reference model. During fine-tuning, the adaptation matrix, as represented by a pair of rank decomposition matrices, is updated. This drastically reduces the number of updated parameters for each training step. The frozen reference model was used as our teacher model. To make our results comparable to Mosbach et al. we used their hyperparameters [6]. These parameters were specifically recommended by Mosbach et al. [5] and are referenced in this Table 2
# 2.5. Evaluation
For validation, we randomly sampled 100 examples from both the in-domain validation and OOD validation datasets. A model was then evaluated on the both the in-domain validation set followed by the OOD validation dataset that matched the in-domain dataset’s task. After completing all of the Context Distillation experiments - each model size, trained on each in-domain dataset, is validated using the student model on both corresponding in-domain and OOD datasets. The in-domain validation and OOD results are compared to Mosbach et al. [6] to as-
sess the efficacy of our approach. Snell et al. [9] shows that Context Distillation out performs direct gradient descent learning on the T5 model [7]. Therefore, we felt that our approach should reasonably improve on the ICL results. A direct comparison of in-context learning vs context distillation across our 4 model sizes has not been performed previously. We focused on filling this knowledge gap by conducting experiments that compare CD to ICL results.
# 3. Experiments
As outlined in the approach section, we conducted a series of teacher-student context distillation experiments. Paraphrasing what was presented, we evaluate CD by training on three benchmark datasets - MNLI and RTE for the NLI task and QQP for the paraphrase task. For each dataset we construct teacher contexts of three sizes, n ∈ {2, 16, 32}. For each context size we sample 4 unique incontext example sets. Each of these example sets equates to a CD training run. We then trained OPT models of 4 different sizes on each training run by sampling 32 training examples and stripping to the label to use as an inference point. We measure success by comparing the performance of student models (trained with CD) to teacher models (untrained) and previously established methods (ICL and FT) on held-out validation datasets both in-domain and out-ofdomain.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bbdb/bbdb2ed1-c074-4936-b844-270dccbf4fe1.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 2: Hyperparameters used in training</div>
# 4.1. Context Distillation
Our context distillation experiments reveal several notable findings. The average accuracy of all four runs for each dataset and model are presented in Table 3. Figures presenting the results of all runs for all teacher context lengths can be found in the appendix. In terms of success, context distillation proved useful. From Table 3, one can see that for all scenarios the student model achieved comparable in-domain and OOD accuracy to the teacher’s in-domain accuracy. For the MNLI and RTE datasets the student model only had a slight reduction in OOD accuracy compared to its in-domain accuracy. However, QQP presents a peculiar behavior by performing better on OOD data than in-domain. The small parameter models, 125m and 350m, performed better after CD on indomain validation. This shows the benefit of the student model seeing more examples through context distillation than the teacher. The teacher having seen only the 16 in context examples, while the student benefits from those and the 32 inference examples used to update the LoRa weights during tuning. For the larger models, 1.3b and 2.7b, the teacher performed better than the student, likely a result of the knowledge ingrained with more parameters. Interestingly, context distillation seems to alleviate the impact of model size on the capability of the model. Where the teacher accuracy increased with model size the student accuracy was more stable across the sizes. The indicates CD may be a valuable in enabling smaller models to perform as well as larger models on specific task. Finally, CD did not overfit on the training data. As evidenced by the good performance of the context distilled student model on OOD validation set. Using LoRa layers with a frozen reference model helped to prevent overfitting and catastrophic forgetting as well.
# 4.2. Comparison of Task Tuning Methods
In comparing the different task tuning methods, our findings suggest that CD offers notable quality advantages over ICL. The results in Figure 1 show our experimental outcomes in comparison to Mosbach et al. [6] for the n = 16 context examples training runs. CD shows comparable indomain performance, but clearly improves on the OOD performance across datasets and model sizes. On the other hand, CD does not perform as well as FT. However, with CD the training dataset size and computation required to compute are drastically reduced. Of note, the improvement achieved through FT is less drastic on the million parameter models that on the billion plus parameter models.
# 5. Experience
During our research, we anticipated writing code and uncovering bugs we generated. We did not anticipate how much code we would write and refactor. The infrastructure was not trivial to set-up and involved running 8+ instances on Google Colab in parallel with T4, L4 and A100 GPUs (2+ students simultaneously). We spent a significant amount of time reverse-engineering the code from Mosbach et al [6]. We wanted to be through and have comparable results relative to our benchmark. This meant replicating any transformation to their dataset. For example, the benchmark binarized the MNLI dataset. Understanding their experiment set-up felt like trial and error, we spent time in several group calls discussing the finer details of how they set-up context examples and inference requests. This resulted in us re-writing our entire code base at least twice. In each rewrite, we had to fix bugs including ones that we found in the benchmark code base itself. When writing training code our team spent considerable time understanding the tokenizer and trying to retrieve labels. We ended up confirming that the benchmark didn’t use labels provided within the dataset and had to recast them to: yes and no labels.
Dataset
Model size
Accuracy teacher
Accuracy in-domain
Accuracy OOD
MNLI
125m
0.547
0.583
0.523
350m
0.531
0.595
0.548
1.3b
0.609
0.525
0.525
2.7b
0.664
0.593
0.515
RTE
125m
0.438
0.530
0.518
350m
0.508
0.538
0.508
1.3b
0.516
0.480
0.533
2.7b
0.547
0.498
0.473
QQP
125m
0.625
0.615
0.710
350m
0.406
0.420
0.490
1.3b
0.406
0.430
0.415
2.7b
0.438
0.428
0.433
Table 3: Comparison of accuracy for the teacher model on in-domain data and post-tuning student model on in-domain and OOD data. Results represent the scenario using n = 16 context examples. Best result per model size in bold.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f61e/f61e03ba-b59a-4df3-8f7c-afd1a70bee56.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Sub-figure (a) comprises the results of ICL as published by Mosbach et al. [6]. Sub-figure (b) are the results from our experiments. Both figures represent the scenario using n = 16 context examples.</div>
During training, we reduced the benchmark hyperparameters, because we had significantly less GPU compute and memory available to us. Our reduced hyper-parameters diminished GPU compute needed to 5% of the benchmark. This reduced training time from multiple days to 4-5 daily sessions over the weekend. While running models, one teammate modified our DL model with LoRa adapter sig-
nificantly cutting both compute and memory requirements. Even after these changes, the OPT 2.7b model struggled to complete a run on the A100 often running out of memory. For the 32 context RTE model, the context was too large and so it didn’t run. We had to cut it from our experiments. After generating the results for all 128 runs, we aggregated the results from JSON into CSV file. One of our teammates
then thoroughly analyzed the results to generate the figures and charts presented in this paper. Overall, the team put in a valiant effort and overcame many hurdles to create this paper and we are happy with the result.
# 6. Conclusion
Models trained with context distillation internalize context examples allowing them to be used during inference. This greatly increases examples that a model learns from avoiding the context window limit. In our paper, context distillation performed well on the out-of-domain dataset and had comparable results to in-context learning on the in-domain dataset. Our study did not tease out the influence of model size on performance. We believe this is due to our limited sample size: 4 runs per model per dataset per context length. We plan on rectifying this with future runs, but note that our initial 120 runs required significant compute. This compute includes upwards of 2 A100s and a minimum of 4 GPUs run in parallel over multiple days. The computational requirement was noticeable for the OPT 2.7b model, which caused the 40 GB memory A100 GPU on multiple occasions to run out of memory for both the MNLI and RTE datasets. Overall, the results were promising with CD showing advanced performance over ICL especially for out-of-domain generalization. Context distillation is a state-of-the-art training regime and is used in multiple domains. We thought about using context distillation for LLM code generation and internalizing code repositories. We are considering a future ablation study where we investigate performance of an amalgamated model: LoRa, knowledge distillation, and context distillation and then selectively remove components to measure their impact on performance. As for our current research presented in this paper, we would like to run the models to generate more samples so that we can properly tease out the effect of model size on performance. An alternative approach is using 128 inference requests instead of 32 during training. During hyper-parameter tuning, this change looked promising as several large models had over 70% accuracy for a few tasks. We chose 32 inference requests due to computational constraints. Snell et al. [9] performs context distillation with a scratchpad and we would love to replicate that work. We also wanted to conduct an experiment by providing the student model with a small in-context dataset to measure its performance and we also thought about using context distillation on instruction-finetuned models.
[1] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Benjamin Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a laboratory for alignment. ArXiv, abs/2112.00861, 2021. https://arxiv.org/abs/2112.00861. 1 [2] Ido Dagan, Oren Glickman, and Bernardo Magnini. The pascal recognising textual entailment challenge. In Joaquin Qui˜nonero-Candela, Ido Dagan, Bernardo Magnini, and Florence d’Alch´e Buc, editors, Machine Learning Challenges. Evaluating Predictive Uncertainty, Visual Object Classification, and Recognising Tectual Entailment, pages 177–190, Berlin, Heidelberg, 2006. Springer Berlin Heidelberg. 2 [3] Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick von Platen, Suraj Patil, Julien Chaumond, Mariama Drame, Julien Plu, Lewis Tunstall, Joe Davison, Mario ˇSaˇsko, Gunjan Chhablani, Bhavitvya Malik, Simon Brandeis, Teven Le Scao, Victor Sanh, Canwen Xu, Nicolas Patry, Angelina McMillanMajor, Philipp Schmid, Sylvain Gugger, Cl´ement Delangue, Th´eo Matussi`ere, Lysandre Debut, Stas Bekman, Pierric Cistac, Thibault Goehringer, Victor Mustar, Franc¸ois Lagunas, Alexander Rush, and Thomas Wolf. Datasets: A community library for natural language processing. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 175–184, Online and Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. 2 [4] Tom McCoy, Ellie Pavlick, and Tal Linzen. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. In Anna Korhonen, David Traum, and Llu´ıs M`arquez, editors, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3428–3448, Florence, Italy, July 2019. Association for Computational Linguistics. 2 [5] Marius Mosbach, Maksym Andriushchenko, and Dietrich Klakow. On the stability of fine-tuning bert: Misconceptions, explanations, and strong baselines. In International Conference on Learning Representations, 2021. 3 [6] Marius Mosbach, Tiago Pimentel, Shauli Ravfogel, Dietrich Klakow, and Yanai Elazar. Few-shot fine-tuning vs. incontext learning: A fair comparison and evaluation. ArXiv, abs/2305.16938, 2023. Github repo: https://github.com/udslsv/llmft. https://arxiv.org/abs/2305.16938. 1, 2, 3, 4, 5 [7] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. ArXiv, abs/1910.10683, 2023. https://arxiv.org/abs/1910.10683. 3 [8] Lakshay Sharma, Laura Graesser, Nikita Nangia, and Utku Evci. Natural language understanding with the quora question pairs dataset. ArXiv, abs/1907.01041, 2019. https://arxiv.org/abs/1907.01041. 2
[9] Charlie Snell, Dan Klein, and Ruiqi Zhong. Learning by distilling context. ArXiv, abs/2209.15189, 2022. https://arxiv.org/abs/2209.15189. 1, 3, 6 [10] Adina Williams, Nikita Nangia, and Samuel Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In Marilyn Walker, Heng Ji, and Amanda Stent, editors, Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112–1122, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. 2 [11] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, Oct. 2020. Association for Computational Linguistics. 2 [12] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer language models. ArXiv, abs/2205.01068, 2022. https://arxiv.org/abs/2205.01068. 2 [13] Yuan Zhang, Jason Baldridge, and Luheng He. PAWS: Paraphrase adversaries from word scrambling. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1298–1308, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. 2
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1e35/1e35769e-ac40-4ff4-a3a6-f789d5a70009.png" style="width: 50%;"></div>
