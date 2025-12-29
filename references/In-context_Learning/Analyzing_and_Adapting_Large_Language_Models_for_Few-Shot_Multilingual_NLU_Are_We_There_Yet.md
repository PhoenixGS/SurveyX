# Analyzing and Adapting Large Language Models for Few-Shot Multilingual NLU: Are We There Yet?
Evgeniia Razumovskaia Ivan Vuli´c Anna Korhonen Language Technology Lab, University of Cambridge, UK {er563, iv250, alk23}@cam.ac.uk
Abstract
Supervised fine-tuning (SFT), supervised instruction tuning (SIT) and in-context learning (ICL) are three alternative, de facto standard approaches to few-shot learning. ICL has gained popularity recently with the advent of LLMs due to its simplicity and sample efficiency. Prior research has conducted only limited investigation into how these approaches work for multilingual few-shot learning, and the focus so far has been mostly on their performance. In this work, we present an extensive and systematic comparison of the three approaches, testing them on 6 high- and low-resource languages, three different NLU tasks, and a myriad of language and domain setups. Importantly, performance is only one aspect of the comparison, where we also analyse the approaches through the optics of their computational, inference and financial costs. Our observations show that supervised instruction tuning has the best trade-off between performance and resource requirements. As another contribution, we analyse the impact of target language adaptation of pretrained LLMs and find that the standard adaptation approaches can (superficially) improve target language generation capabilities, but language understanding elicited through ICL does not improve and remains limited, with low scores especially for low-resource languages.
 4 Mar 2024
[cs.CL]
arXiv:2403.01929v1
# 1 Introduction and Motivation
Recent advances in data-efficient, few-shot learning have been crucial for increasing and promoting language inclusiveness of NLP technology (Devlin et al., 2019; Conneau et al., 2020; ImaniGooghari et al., 2023), substantially lowering the dataset sizerelated ‘entry point’ for a new language. This was made possible by pretrained language models which can generalise to a new task or language from the knowledge stored in their parameters complemented with only a handful of in-task data.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e71a/e71a4d7e-0884-41d2-8acd-ca16bc992f65.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Spanish</div>
Figure 1: Comparison of practical aspects of different learning paradigms (§3.1) in the intent detection task from Multi3NLU++ (Moghe et al., 2023), with exactly the same data setup, for Amharic and Spanish. In-context learning (ICL) has low performance and high inference and computational costs while being comparatively inexpensive. Supervised fine-tuning (SFT) and supervised instructiontuning (SIT), on the other hand, have a larger financial cost but they are much more efficient in terms of inference aspects and computational resources while also performing much better both for Amharic as a representative low-resource language (1a) and Spanish as a high-resource language (1b). The standard approaches for such few-shot adaptations are Supervised Fine-Tuning (SFT), which also subsumes more recent Supervised InstructionTuning (SIT), and In-Context Learning (ICL). SFT and SIT use knowledge in pretrained model parameters for initialisation and then adapt the parameters to a language-task combination via supervised
Figure 1: Comparison of practical aspects of different learning paradigms (§3.1) in the intent detection task from Multi3NLU++ (Moghe et al., 2023), with exactly the same data setup, for Amharic and Spanish. In-context learning (ICL) has low performance and high inference and computational costs while being comparatively inexpensive. Supervised fine-tuning (SFT) and supervised instructiontuning (SIT), on the other hand, have a larger financial cost but they are much more efficient in terms of inference aspects and computational resources while also performing much better both for Amharic as a representative low-resource language (1a) and Spanish as a high-resource language (1b).
The standard approaches for such few-shot adaptations are Supervised Fine-Tuning (SFT), which also subsumes more recent Supervised InstructionTuning (SIT), and In-Context Learning (ICL). SFT and SIT use knowledge in pretrained model parameters for initialisation and then adapt the parameters to a language-task combination via supervised
training on available, even if scarce, resources. Importantly, they yield a model specialised for a single language-task combination and can get increasingly better at the task if a larger training dataset becomes available. ICL, in contrast, uses one model ‘as is’ to complete any task, without any parameter adaptation or fine-tuning. Instead, the model is adapted via prompting: given an explanation of a task (i.e., instruction) and a set of ‘training’ examples (i.e., annotated demonstrations), the model is tasked to generate the label for every input (Radford et al., 2019). Due to the model’s context size, the number of demonstrations used in the input is limited, meaning that the ICL performance is capped by the model’s pretraining and the demonstrations that fit into the input context. Existing generative models used for ICL (termed Large Language Models, or LLMs henceforth) are usually pretrained in an English-centric manner with the vast majority of the pretraining corpus in English and only limited coverage of other languages (Sitaram et al., 2023), even with ‘accidentally encountered’ bilingual and translation data (Briakou et al., 2023). As a result, current LLMs are very far from serving the world’s languages equally: while demonstrating impressive ICL results in English (Mishra et al., 2022a), they still face difficulties when transferring to other languages (Winata et al., 2021; Tanwar et al., 2023), especially low-resource ones (Ojo et al., 2023). In contrast, a number of encoder and encoder-decoder models, such as XLM-R (Conneau et al., 2020) or mT5 (Xue et al., 2021), used for initialisation in SFT are pretrained with much wider language coverage1 (termed multilingually Pretrained Language Models, or mPLMs) (Conneau et al., 2020; ImaniGooghari et al., 2023). This property enables sample-efficient transfer and adaptation of natural language understanding (NLU) models to a much larger array of languages (Ansell et al., 2021) than what is supported by ICL-based LLMs. While SFT, SIT and ICL are comparable approaches for few-shot multilingual NLU, there has been little attention drawn to which of the techniques works better in practice. Therefore, this paper aims to delve deeper into analysing a variety of factors which critically impact effective use of 1For instance, XLM-R (Conneau et al., 2020), mBERT (Devlin et al., 2019), LaBSE (Feng et al., 2022), and mT5 (Xue et al., 2021) cover ∼100 languages at pretraining (albeit with different pretraining data amounts), while Glot500 (ImaniGooghari et al., 2023) covers up to 500 languages.
1For instance, XLM-R (Conneau et al., 2020), mBERT (Devlin et al., 2019), LaBSE (Feng et al., 2022), and mT5 (Xue et al., 2021) cover ∼100 languages at pretraining (albeit with different pretraining data amounts), while Glot500 (ImaniGooghari et al., 2023) covers up to 500 languages.
either from a more practical point of view. Our first aim is to provide answers to the following question: (Q1) Given the same annotated examples, which of the approaches is better in practice? In particular, the sometimes vague term ‘practice’ in our work comprises the following crucial aspects: 1) sample efficiency (i.e., ‘data cost’); 2) computational requirements (‘computational cost’); 3) latency (‘inference cost’); and 4) overall financial or ‘economic’ cost. Prior work has been mainly focused on benchmarking ICL on subgroups of languages (Ojo et al., 2023) and only considered and optimised task performance of the models as the ultimate comparison criterion. In contrast, our work presents an extensive analysis evaluating crosslingual capabilities of SFT and ICL both on high and low-resource languages, considering not only the task performance but also the above listed practical aspects, as illustrated in Figure 1. Furthermore, prior work has demonstrated the effectiveness of parameter-efficient fine-tuning (PEFT) to improve the model’s cross-task capabilities and to promote aspects of its generation abilities (e.g., open-domain chat; Dettmers et al., 2023). In this work, we also analyse how language adaptation of LLMs ‘beyond English’ impacts their NLU and NLG performance in a target language. This gives rise to another core research question: (Q2) Given the benefits of ICL as a learning paradigm (but its inferior performance in comparison to SFT), could we use the standard adaptation strategies to improve LLMs’ generation and understanding capabilities in other languages? To our knowledge, this is the first work analysing how multilingual NLU capabilities of ICL with LLMs are effected by their target language adaptations, as well as studying the trade-offs for NLG. Contributions. 1) Related to Q1, we conduct a comprehensive analysis of ICL versus SFT and SIT paradigms in the context of multilingual few-shot adaptation, with the focus on multiple practical angles and cost. Our analyses show that not only the SFT and SIT approaches with smaller models lead to improved task performance but also they remain more data-, computation-, inference-effective than ICL with general-purpose LLMs. 2) Related to Q2, we investigate the effectiveness of target language adaptation, adopted from the work on mPLMs, for ICL with LLMs. The main finding is that language adaptation leads to superficially improved generation capabilities in the target language with only
limited improvements on the actual tasks, calling for further research that will mitigate the large language gap in LLM development and deployment between English and other languages.
# 2 Related Work
Instruction-Tuning LLMs aims to increase their cross-task generalisation capabilities. Instruction tuning is in essence an SFT technique where the input includes textual description of the task, demonstrations and user input queries while the output is the desirable model output for a given task in text form. Through inclusion of task descriptions into the input, at inference time the model becomes capable of completing tasks unseen during training when provided with task description (Sanh et al., 2022; Chung et al., 2022, interalia). Instruction tuning has become a standard approach to turn an LLM into a model with general capabilities to perform any task, given the instructions, off-the-shelf (Wei et al., 2022a; Mishra et al., 2022b). Extending LLMs to Other Languages. Although there is a growing trend to make NLP systems more linguistically inclusive (Bender, 2011; Doddapaneni et al., 2021), widely used generative LLMs remain predominantly English. For instance, pretraining data of LLaMA-2 and PaLM consists of 90% and 82% English text, respectively (Touvron et al., 2023; Sitaram et al., 2023), which substantially hinders their capabilities in languages other than English (Ojo et al., 2023). In an attempt to equate the models’ performance across languages, there is an increasing interest in extending their multilingual capabilities. A wide range of techniques including continued pretraining (Cui et al., 2023), using self-instruction (Wei et al., 2023) or vocabulary extension (Zhao et al., 2024) have been applied. Due to wide adoption of ICL, another line of work focuses on improving cross-lingual instruction following capabilities via parameter-efficient multilingual instruction tuning (Li et al., 2023b), multilingual pretraining (Shliazhko et al., 2022) and injection of several multilingual examples in fine-tuning (Shaham et al., 2024). The methods show gains in various aspects of model’s target language generation capabilities, while providing no systematic empirical comparisons to prove that improved NLG necessarily correlates with stronger NLU performance via ICL. These works provide initial insights into LLMs processing for languages other than English. In-
terestingly, the success of mPLMs in cross-lingual transfer has always been attributed to their massively multilingual pretraining while, at first sight, LLMs seem to operate differently: they perform surprisingly well while having only a small percentage of multilingual text in their pretraining corpora (Blevins and Zettlemoyer, 2022). At the same time, little to no work has studied multilingual performance of these models in direct comparison with standard mPLMs, and even more so the practical aspects such as memory requirements or latency.
# 3 Preliminaries: On Learning Paradigms and Practical Aspects
We analyse three established learning paradigms for few-shot learning in monolingual and multilingual setups, which are compared across four practical aspects: data cost, computational cost, inference cost and financial cost. We now outline each learning paradigm and practical aspect.
# 3.1 Learning Paradigms
Let D = (x1, y1), ..., (xN, yN) denote a training dataset where xi is the model input, yi is the label annotation and N is the number of training examples, and let M refer to a pretrained language model (LLM or mPLM). Supervised Fine-Tuning (SFT). M is adapted to a task or a language (or both) by fine-tuning its parameters on D and minimising a loss function. Note that here we use SFT in its narrower sense, to refer to ‘standard‘ fine-tuning where an encoderbased model (such as mBERT) or encoder-decoder model (e.g., mT5) is tuned directly for the target task (Devlin et al., 2019; Wei et al., 2022b). At inference, the fine-tuned model M′ is then used. In-Context Learning (ICL). Unlike with SFT, the parameters of M stay fixed and the model is treated as a ‘black box’. ICL relies on generative capabilities of general-purpose LLMs (Brown et al., 2020; Han et al., 2023). The model is adapted to a task by conditioning it on task instructions and in-context examples (demonstrations). Each demonstration included into a prompt consists of an input x and ground-truth annotated label y. In other words, the demonstrations are an alternative way to use the data available in D. Then, M is expected to generate the label for the test input usually included at the end of the prompt. While in SFT the model parameters are adapted to a target task, with ICL the model is expected to learn the
task by analogy, via the provided task description combined with demonstrations.
task by analogy, via the provided task description combined with demonstrations. Supervised Instruction-style Tuning (SIT). To unlock full potential of ICL, sufficiently large language models need to be used (Wei et al., 2022c), drastically raising the computational overhead at inference in comparison with SFT. SIT thus presents the middle ground between the two. Here, one fine-tunes small(er) instruction-based models to specific tasks. While SFT fine-tunes the model directly on annotated data D, SIT extends each input in D with task-specific instructions leveraging model’s instruction-following capabilities (Wei et al., 2022b) obtained during pretraining. SIT typically does not include demonstrations into input, although including them there is also possible (Min et al., 2022; Chen et al., 2022), typically with small to negligible performance gains in few-shot setups but increased computational cost (Li et al., 2023a). For simplicity, we experiment only with the SIT variant without any demonstrations.
# 3.2 Practical Aspects
We consider practical costs of the ‘full cycle‘ of model development – from data collection costs to inference cost, and aim to associate those costs with the learning paradigms described in §3.1. Data Cost. One key limiting factor for the model adaptation to new task-language (or even finergrained task-language-domain) combinations is the costly and complex data collection process, especially for low-resource languages and specialised domains. Therefore, it is crucial to develop methods which can efficiently generalise from a small number of annotated examples. In §5, we analyse this data cost, that is, sample efficiency as the relationship between the number of training examples and task performance. Computational Cost. The memory requirements of LLMs keep growing proportionally to the number of their parameters. Deploying such a model to the users means that one needs to have access to and support costly infrastructure with large vRAM (Aminabadi et al., 2022; Alizadeh et al., 2023). Here, we analyse the memory requirements of each learning paradigm both for model storage and training, where applicable, and how they correlate with the target task performance. Inference Cost. Latency, or time needed by the model to complete the prediction (Huyen, 2022, Chapter 1), has the largest impact on user-facing
applications such as task-oriented dialogue. To make the system usable, it is critical to strike a balance between strong performance and low latency. We thus analyse the inference cost in two ways as: 1) wall-clock inference time, aiming to directly approximate (relative) latency of different models; and 2) inference FLOPs, a hardware-independent metric to compare inference complexity. Financial Cost. Each of the aspects above contributes to the overall cost of each model’s life cycle. As financial resources are usually limited, we also aim to (roughly) estimate the overall financial expenditure needed for each learning paradigm, including data collection, GPU and inference costs.
# 4 Experimental Setup
We focus on the comparison between SFT, SIT and ICL in few-shot multilingual and cross-lingual setups, aiming to make the comparison as fair as possible across languages, learning paradigms and models, and targeting the following setups: In-Language Generalisation. We evaluate the model’s ability to generalise on new examples in the same language in which fine-tuning examples or demonstrations were provided to the model. Cross-Language Generalisation. We use a model trained in one language to perform the task in another one, where the transfer typically proceeds from a high-resource language to a low-resource one. In our experiments, we assume the typical transfer direction with English as the (highresource) source language. In-Domain and Cross-Domain Generalisation. For many NLU tasks (e.g., for task-oriented dialogue) it is common to consider transferring the systems between different domains, e.g., from flight booking to the restaurant booking domain. If a model can be transferred across domains, it means that it has in-depth understanding of the classes used in the respective domain definitions/ontologies.
# 4.1 Evaluation Tasks and Datasets
The main focus of the analyses, revolving around Q1 and Q2 from §1, is on NLU tasks for taskoriented dialogue as one widely used and established practical application of NLP technology, due to multiple reasons. 1) Dialogue is a user-facing application where computational and memory requirements, data collection cost, inference latency and other practical concerns of the model development cycle are of ultimate importance. 2) Dia-
Dataset
LANGS
# TEST EX.
# CLASSES
Multi3NLU++
ID
AM, EN, ES, MR, TR
300
62
VE
17
XNLI
NLI
EN, RU, TR, ES
5,010
3
logue NLU tasks provide well-defined ontologies and evaluation setups, with evaluation benchmarks that comprise comparable and semantically aligned training and test data across multiple languages, including high- and low-resource ones (Moghe et al., 2023; Hu et al., 2023a), and multiple domains. 3) In contrast to standard ‘non-dialogue’ NLU tasks, dialogue NLU datasets are unlikely to have been seen and ‘absorbed‘ by LLMs during their pretraining, which avoids test data leakage (Balloccu et al., 2024; Sainz et al., 2023). Dialogue-oriented evaluation is conducted on the tasks of intent detection (ID) and value extraction (VE). ID aims to classify user’s utterance into a set of intent classes predefined in the domain ontology. The aim of VE is to identify the presence of ontology-related domain-specific slotvalue pairs in a given sentence. Here, we use the Multi3NLU++ dataset (Moghe et al., 2023), covering English (Casanueva et al., 2022) and the following 4 languages: Amharic (AM), Marathi (MR), Spanish (ES) and Turkish (TR). The dataset also spans two different domains: BANKING and HOTELS with a partial overlap in intent classes and slots. For both tasks we report micro-F1 scores.2 To verify that our findings extend beyond only dialogue-related NLU tasks, we also evaluate on the standard NLI task with XNLI (Conneau et al., 2018) which provides training and evaluation data in 14 languages, while we focus on a subset of 3: ES, TR, and Russian (RU), and report accuracy as the evaluation metric. Additional information on the evaluation datasets is provided in Table 1. Cross-Language Parallel Few-Shot Setup. To ensure fair comparisons of all learning paradigms across languages, we make use of the multi-parallel nature of the datasets we use. For each languagedomain combination in Multi3NLU++ (or just lan2We also note that 1) each utterance in Multi3NLU++ may have multiple intents; ID is thus a multi-label classification task. 2) Further, for VE, we consider the slot value as correctly
# Cross-Language Parallel Few-Shot Setup. 
ensure fair comparisons of all learning paradigms across languages, we make use of the multi-parallel nature of the datasets we use. For each languagedomain combination in Multi3NLU++ (or just lan-
2We also note that 1) each utterance in Multi3NLU++ may have multiple intents; ID is thus a multi-label classification task. 2) Further, for VE, we consider the slot value as correctly labelled only if it exactly matches the gold value. Finally, 3) as in prior work (Casanueva et al., 2022; Moghe et al., 2023), the cross-domain performance for the two tasks is evaluated only on the intents and slots shared across domains. We refer to the original Multi3NLU++ work for further details.
guage in XNLI) we sample 300 test examples.3 We also randomly sample training sets consisting of {30, 50, 100, 500, 1,000} examples which are kept exactly the same across all languages to ensure the content in training data does not coincidentally favour one of the languages.4 SFT Evaluation. We use two standard models, XLM-R-Base (Conneau et al., 2020) and LaBSE (Feng et al., 2022). LaBSE is a sentence encoder model where, following prior work (Moghe et al., 2023), we train only task-specific classifiers on top of the fixed encoder. We refer to this approach, applied only to sentence-level tasks (ID and NLI), as LaBSE+CL.5 ICL Evaluation. We evaluate the following models: Flan-T5-XL, mT0-XL, LLaMA-2-7B and GPT3.5.6 Flan-T5-XL (3B parameters; Chung et al., 2022), mT0-XL (3.7B; Muennighoff et al., 2022) and GPT-3.5 (Achiam et al., 2023) are massively instruction-tuned models. While Flan-T5 was pretrained mostly in English and several high-resource languages, mT0-XL offers a more comprehensive and balanced multilingual pretraining set. The inputs for ICL were designed in a crosslingual manner, where the task descriptions and context were in English while the few-shot examples and the sentence to be analysed were provided in the target language. This follows the recommendations from prior work where it was empirically verified that English instructions led to stronger results than in-language instructions (Shi et al., 2022; Lin et al., 2022). For each task we design the instructions (i) to match the instructions in pretraining as closely as possible, while (ii) yielding reasonable output when tested on several validation examples.7 Note that, given a fixed input context of each model, the number of demonstrations to be 3We conduct the sampling step due to a large number of experiments run; preliminary experiments with full test sets indicated exactly the same relative trends, but with much increased computational cost and time overheads. While sampling, we ensured that each intent and slot in the domain ontology occurred at least twice in the test set 4To ensure reproducibility, unique ids of the examples in the training and test splits will be made publicly available. 5For NLI as a single-label classification task, the softmax output activation function is used. In contrast, for ID we use sigmoid and consider all intents where the sigmoid activation is larger than a predefined threshold value θ. Following prior work, we set θ = 0.3. 6We use GPT-3.5-turbo-instruct due to its instructionfollowing capabilities proven in prior work (Ye et al., 2023) 7For reproducibility, we will share the full instructions templates for all languages and tasks.
used for ICL is limited: for all the models in our comparison it is less than 30, and 30 is the lowest amount of training samples we use in SFT. SIT Evaluation. Here, we include individual perclass questions into instructions: this design (i) was previously shown to result in much improved SIT performance (Fuisz et al., 2022; Razumovskaia et al., 2023), while (ii) it also fits into the model input context for tasks with a large number of classes. We rely on the same instructions as with ICL. We experiment with two models: (i) (mostly English pretrained) Flan-T5-Base (250M parameters) and multilingually oriented mT0-Base (580M).8
# 5 (Q1) Results and Discussion: Learning Paradigms and Practical Aspects
We first delve into comparisons revolving around Q1 (§1). The main results across different setups, models, training data sizes and learning paradigms are summarised in Figure 2, while full (numerical) results are provided in Appendix C. We now zoom into discussions originating from the results.
We first delve into comparisons revolving around Q1 (§1). The main results across different setups, models, training data sizes and learning paradigms are summarised in Figure 2, while full (numerical) results are provided in Appendix C. We now zoom into discussions originating from the results. Data Efficiency. One of the core reasons to use ICL is its inherent data/sample efficiency. Comparing the supervised methods against ICL, we observe that the former reach or overcome the performance of ICL with all tested open-source models (Flan-T5-XL, mT0-XL and LLaMA-2-7B), even when fine-tuned with mere 30 in-task examples, while they outperform GPT-3.5 with 50 or 100 in-task examples. These findings hold across all evaluation tasks and setups. At the same time, the results also reveal several key differences between tasks and languages. Comparing the trends for ID and VE (cf., Figures 2a and 2d): for sentence classification tasks where the outputs are language-independent, the improvements of SFT over ICL are less pronounced. For instance, the gains over GPT-3.5 with 100 training examples for MR and ES are 3.15 and 4.17 F1 points,9 respectively. In contrast, the gains over ICL for value extraction as a more language-specific task are very large, 19.8 and 25.28 for MR and ES, respectively, when comparing the best-performing SFT method with ICL. Moreover, for VE, the gaps with ICL are considerable even with 30 training examples are used (3.4 and 7.5 F1 points for MR and ES).
8SFT and SIT training hyperparameters are in Appendix B. 9The cited numbers are for in-language in-domain setups; the trends are the same in the other setups.
For AM a supervised model surpasses GPT-3.5 performance already with 30 training examples, while for ES 50 or 100 training examples are required, depending on the setup. For high-resource languages (EN, ES) SIT-based Flan-Base with 30 examples performs consistently better than ICL with GPT-3.5 ICL for ID and VE across the setups. We hypothesise that high performance of SIT-based Flan is caused by i) its instruction following capabilities, and ii) large-scale English pretraining which is helpful for both few-shot in-language generalisation and cross-lingual transfer from English to Spanish, a linguistically close language. For lowresource languages (AM, MR) we notice a different tendency across the domain setups: LaBSE+CL and SIT-based mT0 show the highest performance for ID and VE, respectively. This shows the importance of multilingual pretraining for the model to generalise to unseen or ‘less seen’ languages. In-Domain vs Cross-Domain Evaluation. The comparison in cross-domain setups consistently shows that SIT outperforms SFT and ICL (see Figure 2b), corroborating findings from prior work on English (Razumovskaia et al., 2023). We speculate that the success of SIT in cross-domain setups stems from the model’s ability to follow instructions obtained during pretraining and the ability to extract the class semantics from instructions obtained during fine-tuning. The best-performing instruction-tuned LLM, however, depends on the target language: on low-resource languages multilingually pretrained models such as mT0 perform consistently better than English-pretrained models such as Flan, while we observe reversed trends for high-resource languages (ES). Cross-Lingual Zero-Shot Transfer. Figure 2c presents the results for zero-shot transfer in indomain setups: performance across languages for all approaches is substantially lower than the performance in English. Further, as expected, performance on low-resource languages is considerably lower than on high resource languages. The results in Table 2 show that for ICL, unlike for SFT (Lauscher et al., 2020), providing the model with data examples in the target language does not always improve the final performance. Target language demonstrations seem to be helpful to the models which have strong instruction-following capabilities and are familiar with the target language (e.g., ES performance of Flan and GPT-3.5). Seen vs Unseen Tasks. Figure 2e demonstrates
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c2d1/c2d1ae84-7d80-4c58-af0e-9aa711d6db52.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) ID: In-Language In-Domain</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6b09/6b09f957-f4d6-4231-83ce-f818f8e11e64.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) ID: In-Language Cross-Domain</div>
<div style="text-align: center;">(c) ID: Cross-Lingual In-Domain</div>
<div style="text-align: center;">(e) XNLI</div>
<div style="text-align: center;">Figure 2: Intent detection, value extraction and NLI results for the six languages in our evaluation. This performance is in line with other prior work (Hu et al., 2023b). We exclude LLaMa-2 results as its performance was 0.0 across all tasks. Results for VE in other setups are provided in Appendix D.</div>
the effectiveness of ICL with Flan and GPT-3.5 for XNLI as the ‘seen task’,10 with different patterns observed for the tasks with unseen data (i.e., ID and VE). This discrepancy is especially pronounced for high-resource languages. SIT vs ICL. In general, the results indicate that SIT consistently leads to better results than ICL in few-shot setups. Smaller SIT-based models can
10XNLI is based on the English MultiNLI data (Williams et al., 2018), which has been used for instruction-training of many LLMs (Muennighoff et al., 2022; Chung et al., 2022).
10XNLI is based on the English MultiNLI data (Williams et al., 2018), which has been used for instruction-training of many LLMs (Muennighoff et al., 2022; Chung et al., 2022).
Model
AM
MR
ES
TR
GPT-3.5
ICLt
19.19
48.28
63.25
59.27
ICLen
14.87
38.67
57.64
47.50
Flan
ICLt
3.28
3.02
45.26
31.36
ICLen
6.33
6.68
44.70
28.04
mT0
ICLt
3.61
4.71
4.60
3.36
ICLen
7.19
7.99
7.23
6.45
Table 2: ICL results on the ID task with English (ICLen) or target language (ICLt) demonstrations. even outperform ICL with GPT3.5 when 100+ task examples are available. Due to its sample efficiency and strong performance in cross-domain and cross-
Table 2: ICL results on the ID task with English (ICLen) or target language (ICLt) demonstrations.
even outperform ICL with GPT3.5 when 100+ task examples are available. Due to its sample efficiency and strong performance in cross-domain and cross-
Paradigm: Model
MEMORY cost
INFERENCE cost
Max (GB)
Storage (GB)
Time (s)
FLOPs (109)
SFT: LaBSE+CL
1.80
1.80
0.004
2.05
SFT: XLM-R
4.14
1.04
0.004
11.18
SIT: Flan-Base
3.32
0.85
0.059
23.23
SIT: mT0-Base
5.82
1.45
0.081
20.44
ICL: Flan-XL
10.37
10.37
1.58
39.03
ICL: mT0-XL
12.03
12.03
1.20
55.00
ICL: GPT-3.5
-
-
2.78
-
Table 3: Memory and inference costs of SFT, SIT and ICL, measured on the ID test dataset. MEMORY Max is the peak fine-tuning or storage memory cost of each approach. MEMORY Storage refers to storage requirements per models. For all models but GPT-3.5 the measurements were conducted on a single RTX-3090 GPU. For GPT-3.5, we report average response time per example.
lingual setups, SIT also mitigates the issue of using a separate model for each task-language or tasklanguage-domain combination.
# 5.1 Analyses of Practical Costs
Given that the results above indicate that the two supervised paradigms (SFT and SIT) substantially outscore ICL in terms of task performance in general, we now focus on comparing them in terms of practical aspects. The summary is presented in Figure 1 (see §1) and Table 3. Computational (Memory) Cost. Besides improved task performance, another advantage of SFT and SIT is that the underlying high-performing models are much smaller and thus have lower memory requirements. The largest memory cost for ICL is storing the model’s parameters at inference time while for SFT and SIT it is the memory requirements during fine-tuning. We rely on HuggingFace Memory Calculator to establish vRAM needed for training and inference of every paradigm. We measure the memory requirements in full precision and using the AdamW optimiser (Loshchilov and Hutter, 2019), when applicable.11 The results indicate that models used for ICL have more than 2× higher memory needs than mT0 and Flan-T5-base used for SIT in our experiments. Another angle to memory requirements is the storage cost, i.e., how much memory is needed to store a given model (‘as is’ for ICL and after fine-tuning for SFT and SIT). Table 3 suggests that storage cost for models used for ICL is at least 4× higher than the models used in SIT and SFT. Inference Cost. Beyond average wall-clock in-
11Closed-source GPT-3.5 is excluded from the comparison.
11Closed-source GPT-3.5 is excluded from the comparison.
ference time per test example. we also report the number of FLOPs measured using fvcore, also averaged per test example. As expected, the inference cost scales with the size of the underlying model, with inference time of GPT-3.5 being more than 3x higher than that of SIT-ed models, and inference FLOPs of open-source ICL models being 2.5× higher than for their smaller SIT-ed counterparts. While SFT methods demonstrate even higher inference efficiency, we observe that SIT has the best trade-off between inference cost and performance. Financial Cost. Having demonstrated considerably higher inference costs of ICL, we also consider overall economic costs required for SFT, SIT, and ICL. Target language data annotation accounts for the largest expenditure in the process. We calculate the annotation cost based on Moghe et al. (2023). ICL consumes up to 30 annotated examples, with total costs of £15.9 and £18.6 for high-resource and low-resource languages, respectively, where the annotations get obtained both for ID and VE. In the VE task, SIT-based methods reach or surpass the ICL performance of the strongest model (GPT3.5) already with 20 extra examples (i.e., with 50+ training examples for supervised learning), which only adds £11 or £10 to the overall cost for lowand high-resource languages, respectively. Given the larger inference time and computational costs of the ICL, the total ongoing costs are likely to be larger than the one-time additional annotation budget. To put the numbers in context, the inference cost of 300 test examples with GPT-3.5 is between £3 and £4 for high- and low-resource languages, respectively. Put simply, the actual cost balance should take into account also the tentative number of inference calls. Further, while increasing the input context length of LLMs is an active research area (Press et al., 2022; Rubin and Berant, 2023, among others), many models relying on the ICL paradigm are still constrained by context length, and there is evidence that ICL performance even gets quickly saturated with the addition of extra in-context examples (Chen et al., 2023; Li et al., 2023a) and that the long context is not leveraged adequately (Liu et al., 2023). On the contrary, unlike with ICL, our experiments demonstrate that performance of SFT and SIT improves with more annotated examples (both in-language and cross-lingually, see Figure 2). Data annotation of 100 training examples raises the annotation cost by an average of £37.5 while in-
creasing the ID and VE performance by an average of 15 F-1 points over ICL with GPT-3.5.
# 6 (Q2) Results and Discussion: Target Language Adaptation of LLMs
§5 indicates that ICL is consistently inferior to the two supervised learning paradigms, SFT and SIT, not only in terms of task performance but also concerning computational and inference costs. At the same time, ICL relies on a single model and is thus appealing when extending a system to a large number of language-task combinations. Prior work on ‘decoder-only’ LLMs demonstrated the effectiveness of parameter-efficient finetuning (PEFT) to improve their cross-task generalisation capabilities (Page-Caccia et al., 2024), whereas PEFT is a standard approach for crosslingual adaptation of ‘encoder-only’ and ‘encoderdecoder’ models such as XLM-R or mT5 (Conneau et al., 2020; Xue et al., 2021). In this work, we explore whether such language-specific PEFT-style adaptation can improve ICL and generation capabilities of LLMs in languages other than English. We focus on LLaMA-2-7B as our base model, as: (i) it is a ‘decoder-only’ model that (ii) has been trained as the ‘English-first’ model, with almost 90% of its pretraining data in English; and (iii) it displayed the lowest performance in our experiments in §4 while being the largest model in our evaluation. Language Adaptation Setup. We use QLoRA (Dettmers et al., 2023) as a standard PEFT-based language adaptation technique. QLoRA performs two modifications to the base LLM. The model is first quantised to reduce the memory requirements and then a low-rank adapter (Hu et al., 2022) is trained on top of the quantised model. In our experiments the adapter is tuned on the target language data, aiming to boost the target language capabilities of the underlying LLM. For the adaptation experiments, we focus on three languages: Spanish, Turkish and Marathi. The adapter for each language is trained on the respective portion of mC4 (Xue et al., 2021). Hyperparameters are set following Dettmers et al. (2023), with exact details available in Appendix E.12
# 6.1 Generation after Language Adaptation?
First, we assess whether target language adaptation boosts generation capabilities of the LLM in the target language. To this end, we use the 12Training each QLoRA adapter requires over 24 GPU-h.
Bactrian-X dataset (Li et al., 2023b), a multilingual instruction dataset containing parallel instructionresponse pairs in 52 languages. For our evaluation, we use a subset of 100 randomly sampled examples ensuring the same parallel examples across the three languages in our evaluation (ES, TR, MR). Generation Evaluation: True or Superficial Improvements? We focus on the three aspects of generation capabilities: (i) whether the model outputs text in the same language as expected by the input (i.e., I/O language agreement, similarly to Kew et al., 2023); (ii) naturalness of the generated text; (iii) lexical overlap between golden responses and generation outputs. I/O language agreement involves doing automated language identification of the generated text and establishing whether it corresponds to the input text. For this purpose, we use the current state-of-the-art language identification model, GlotLID-500 (Kargaran et al., 2023). We evaluate naturalness via MAUVE (Pillutla et al., 2021) which measures the distributional gap between human written and generated texts. For lexical overlap, we report ROUGE (Lin, 2004) and BLEU (Papineni et al., 2002). Figure 3 shows consistent gains of generation capabilities over the three evaluation aspects after target language adaptation, with especially large improvements for Marathi as the lowest-resource language. The I/O agreement scores suggest that through language adaptation LLM’s abilities to generate text in the target languages are reinforced. However, those standard metrics still do not fully capture the potential usefulness of generated output and (improved) generation capabilities. We thus also conduct human-based evaluation for Spanish across the following two axes: naturalness and usefulness. Each output is evaluated on a simple 3point Likert-like scale.13 Interestingly, the average naturalness score raises from 1.4 to 2.2 after language adaptation while usefulness only increases from 1.4 to 1.6. In practice, this means that even after language adaptation the model is still far from being useful for the target language speakers. This finding corroborates preliminary observations of Kew et al. (2023) that the English-centric models can learn to generate text in a target language comparatively easily, but useful instruction-following capabilities still remain largely out of reach. Put simply, while generated text in the target language becomes more fluent, its coherence and relevance 13Annotation instructions are provided in Appendix G.
13Annotation instructions are provided in Appendix G.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/65a9/65a92c4d-e0b5-427f-a286-4bb72c5c21a0.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Generation evaluation after target language adaptation (LLaMA-2).</div>
remain limited.
# 6.2 NLU after Language Adaptation?
Given only superficial improvements in generation capabilities, we now assess whether the ICL capabilities improve for NLU tasks. For brevity, we focus on XNLI as the least complex NLU task. Even for XNLI, we observe only a negligible nonsignificant improvement from the average accuracy score of 30.5 to 30.7.14 Performance is in fact below random (33%), supporting the observations from §5 that resource-efficient ICL requires both multilingual pretraining and instruction tuning. From qualitative assessment of the outputs, we notice that the models struggle to follow the task description and instructions, and often do not adhere to output formatting requirements. Massively Multilingually Adapted LLMs in NLU tasks. The results above suggest that direct target language adaptation of ‘English-first’ LLMs such as LLaMA-2 does not yield any benefits to ICL performance in NLU tasks. Next, we study whether massively multilingual adaptation of ‘English-first’ models, as done in very recent work, can improve their ICL capabilities in different languages. We evaluate the MaLa-500 model (Lin et al., 2024) which was adapted for 534 languages using the Glot-500-c dataset (ImaniGooghari et al., 2023), and is also based on LLaMA-2.15 We focus on the ID task to evaluate MaLa’s ICL performance in the (easiest) in-language in-domain setup, with results summarised in Table 4. They reveal that, while adaptation gives marginal improvements for ICL, the performance is still extremely low and 14Per-language scores are provided in Appendix F. Similar relative trends have been observed in preliminary experiments on another, more complex NLU task: Belebele (Bandarkar et al., 2023), where the results are on-par or lower than the random choice baseline, as well as in more complex NLU tasks from our evaluation in §5. 15MaLA-500 was adapted using: (i) LoRA-based parameter adaptation; (ii) vocabulary extension to accommodate for languages that do not use the Latin script.
Model
AM
EN
MR
ES
TR
ICL: LLaMA-2
0.0
0.0
0.0
0.0
0.0
ICL: MaLA-500
1.0
3.01
0.0
1.0
3.01
ICL: GPT-3.5
19.19
64.22
48.28
58.25
46.12
SIT: mT0
26.13
68.00
51.44
61.69
51.60
Tr-Test + ICL: GPT-3.5
32.25
–
48.49
47.95
48.60
Table 4: ICL results on the ID task in the inlanguage in-domain setup.
lags substantially behind GPT-3.5 performance and SIT with mT0-Base with 100 training examples. A comparison with translate-test baseline shows that while translate-test benefits low-resource AM, it is still outperformed by SIT on all other languages. Overall, the scores suggest that more work is needed on multilingual adaptation of LLMs to unlock their ICL capabilities in other languages, and current adaptation strategies do not yield models with competitive (nor even useful at all) NLU.
# 7 Conclusions and Future Work
This work has provided a series of in-depth analyses of multilingual capabilities of three learning paradigms, two supervised ones versus in-context learning (ICL), with the focus on few-shot learning and NLU tasks. Besides task performance, the focus of the analyses has also been on multiple practical aspects (e.g., data efficiency, memory requirements, inference latency). As some of the key findings, we highlight that supervised approaches outperform ICL, even when substantially larger LLMs with higher inference costs are used for ICL. In addition, the analysis of target language adaptation on top of standard LLMs also does not paint a bright picture for multilingual NLP at the moment: while fluency of generated output improves post-adaptation, the output coherence and usefulness remains limited, plus the adapted LLMs lag substantially behind other (weakly supervised) approaches in NLU tasks for target languages. In general, our work has affirmed the importance of multilingual pretraining and the potential of supervised training on top of LLMs. Future work should invest more effort into the creation of massively multilingual- and multitask-pretrained LLMs with higher language coverage. Further, our analysis in §6 calls for new and improved language adaptation methods atop the LLMs.
# Acknowledgments
The work has been in part supported by a Huawei research donation to the Language Technology Lab
at the University of Cambridge. It has also been supported by the UK Research and Innovation (UKRI) Frontier Research Grant EP/Y031350/1 EQUATE (the UK government’s funding guarantee for ERC Advanced Grants) awarded to Anna Korhonen at the University of Cambridge. The work of Ivan Vuli´c has been supported in part by a Royal Society University Research Fellowship ‘Inclusive and Sustainable Language Technology for a Truly Multilingual World’ (no 221137; 2022-).
# References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. ArXiv preprint, abs/2303.08774.
Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, Karen Khatamifard, Minsik Cho, Carlo C Del Mundo, Mohammad Rastegari, and Mehrdad Farajtabar. 2023. Llm in a flash: Efficient large language model inference with limited memory. ArXiv preprint, abs/2312.11514.
Reza Yazdani Aminabadi, Samyam Rajbhandari, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Olatunji Ruwase, Shaden Smith, Minjia Zhang, Jeff Rasley, et al. 2022. Deepspeedinference: enabling efficient inference of transformer models at unprecedented scale. In SC22: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–15. IEEE.
Alan Ansell, Edoardo Maria Ponti, Jonas Pfeiffer, Sebastian Ruder, Goran Glavaš, Ivan Vuli´c, and Anna Korhonen. 2021. MAD-G: Multilingual adapter generation for efficient crosslingual transfer. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 4762–4781, Punta Cana, Dominican Republic. Association for Computational Linguistics.
Simone Balloccu, Patrícia Schmidtová, Mateusz Lango, and Ondˇrej Dušek. 2024. Leak, cheat, repeat: Data contamination and evaluation malpractices in closed-source llms. ArXiv preprint, abs/2402.03927.
Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. 2023. The Belebele Benchmark: a parallel reading comprehension dataset in 122 language variants. ArXiv preprint, abs/2308.16884.
Emily M Bender. 2011. On achieving and evaluating language-independence in nlp. Linguistic Issues in Language Technology, 6.
Terra Blevins and Luke Zettlemoyer. 2022. Language contamination helps explain the crosslingual capabilities of english pretrained models. ArXiv preprint, abs/2204.08110.
Eleftheria Briakou, Colin Cherry, and George Foster. 2023. Searching for needles in a haystack: On the role of incidental bilingualism in PaLM’s translation capability. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9432–9452, Toronto, Canada. Association for Computational Linguistics.
om B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.
nigo Casanueva, Ivan Vuli´c, Georgios Spithourakis, and Paweł Budzianowski. 2022. NLU++: A multi-label, slot-rich, generalisable dataset for natural language understanding in task-oriented dialogue. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 1998–2013, Seattle, United States. Association for Computational Linguistics.
Jiuhai Chen, Lichang Chen, Chen Zhu, and Tianyi Zhou. 2023. How many demonstrations do you
need for in-context learning? In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 11149–11159, Singapore. Association for Computational Linguistics.
Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, and He He. 2022. Meta-learning via language model in-context tuning. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 719–730, Dublin, Ireland. Association for Computational Linguistics.
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instructionfinetuned language models. ArXiv preprint, abs/2210.11416.
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440–8451, Online. Association for Computational Linguistics.
Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: Evaluating cross-lingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2475–2485, Brussels, Belgium. Association for Computational Linguistics.
Yiming Cui, Ziqing Yang, and Xin Yao. 2023. Efficient and effective text encoding for chinese llama and alpaca. ArXiv preprint, abs/2304.08177.
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient finetuning of quantized llms. ArXiv preprint, abs/2305.14314.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. Sumanth Doddapaneni, Gowtham Ramesh, Mitesh M Khapra, Anoop Kunchukuttan, and Pratyush Kumar. 2021. A primer on pretrained multilingual language models. ArXiv preprint, abs/2107.00676. Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. 2022. Languageagnostic BERT sentence embedding. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 878–891, Dublin, Ireland. Association for Computational Linguistics. Gabor Fuisz, Ivan Vulic, Samuel Gibbons, Iñigo Casanueva, and Paweł Budzianowski. 2022. Improved and efficient conversational slot labeling through question answering. ArXiv preprint, abs/2204.02123. Xiaochuang Han, Daniel Simig, Todor Mihaylov, Yulia Tsvetkov, Asli Celikyilmaz, and Tianlu Wang. 2023. Understanding in-context learning via supportive pretraining data. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12660–12673, Toronto, Canada. Association for Computational Linguistics. Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net. Songbo Hu, Han Zhou, Mete Hergul, Milan Gritta, Guchun Zhang, Ignacio Iacobacci, Ivan Vuli´c, and Anna Korhonen. 2023a. Multi 3 woz: A multilingual, multi-domain, multi-parallel dataset for training and evaluating culturally adapted task-oriented dialog systems. Transactions of the Association for Computational Linguistics, 11:1396–1415. Songbo Hu, Han Zhou, Moy Yuan, Milan Gritta, Guchun Zhang, Ignacio Iacobacci, Anna Korhonen, and Ivan Vuli´c. 2023b. A systematic study
Gabor Fuisz, Ivan Vulic, Samuel Gibbons, Iñigo Casanueva, and Paweł Budzianowski. 2022. Improved and efficient conversational slot labeling through question answering. ArXiv preprint, abs/2204.02123.
Xiaochuang Han, Daniel Simig, Todor Mihaylov, Yulia Tsvetkov, Asli Celikyilmaz, and Tianlu Wang. 2023. Understanding in-context learning via supportive pretraining data. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12660–12673, Toronto, Canada. Association for Computational Linguistics.
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.
ongbo Hu, Han Zhou, Mete Hergul, Milan Gritta, Guchun Zhang, Ignacio Iacobacci, Ivan Vuli´c, and Anna Korhonen. 2023a. Multi 3 woz: A multilingual, multi-domain, multi-parallel dataset for training and evaluating culturally adapted task-oriented dialog systems. Transactions of the Association for Computational Linguistics, 11:1396–1415.
Songbo Hu, Han Zhou, Moy Yuan, Milan Gritta, Guchun Zhang, Ignacio Iacobacci, Anna Korhonen, and Ivan Vuli´c. 2023b. A systematic study
of performance disparities in multilingual taskoriented dialogue systems. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6825–6851, Singapore. Association for Computational Linguistics.
Chip Huyen. 2022. Designing machine learning systems. " O’Reilly Media, Inc.".
Ayyoob ImaniGooghari, Peiqin Lin, Amir Hossein Kargaran, Silvia Severini, Masoud Jalili Sabet, Nora Kassner, Chunlan Ma, Helmut Schmid, André Martins, François Yvon, and Hinrich Schütze. 2023. Glot500: Scaling multilingual corpora and language models to 500 languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1082–1117, Toronto, Canada. Association for Computational Linguistics.
Amir Kargaran, Ayyoob Imani, François Yvon, and Hinrich Schuetze. 2023. GlotLID: Language identification for low-resource languages. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 6155–6218, Singapore. Association for Computational Linguistics.
Tannon Kew, Florian Schottmann, and Rico Sennrich. 2023. Turning english-centric llms into polyglots: How much multilinguality is needed? ArXiv preprint, abs/2312.12683.
Anne Lauscher, Vinit Ravishankar, Ivan Vuli´c, and Goran Glavaš. 2020. From zero to hero: On the limitations of zero-shot language transfer with multilingual Transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4483–4499, Online. Association for Computational Linguistics.
Chengzu Li, Han Zhou, Goran Glavaš, Anna Korhonen, and Ivan Vuli´c. 2023a. On task performance and model calibration with supervised and self-ensembled in-context learning.
Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin. 2023b. Bactrian-x: A multilingual replicable instruction-following model with low-rank adaptation. ArXiv preprint, abs/2305.15011.
Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.
Peiqin Lin, Shaoxiong Ji, Jörg Tiedemann, André FT Martins, and Hinrich Schütze. 2024. Mala-500: Massive language adaptation of large language models. ArXiv preprint, abs/2401.13303.
i Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O’Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, and Xian Li. 2022. Few-shot learning with multilingual generative language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9019–9052, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023. Lost in the middle: How language models use long contexts. ArXiv preprint, abs/2307.03172.
Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.
Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2022. MetaICL: Learning to learn in context. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2791–2809, Seattle, United States. Association for Computational Linguistics.
# Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2022. MetaICL: Learn-
Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. 2022a. Cross-task generalization via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 3470–3487, Dublin, Ireland. Association for Computational Linguistics.
for Computational Linguistics. Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. 2022b. Cross-task generalization via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3470–3487, Dublin, Ireland. Association for Computational Linguistics. Nikita Moghe, Evgeniia Razumovskaia, Liane Guillou, Ivan Vuli´c, Anna Korhonen, and Alexandra Birch. 2023. Multi3NLU++: A multilingual, multi-intent, multi-domain dataset for natural language understanding in task-oriented dialogue. In Findings of the Association for Computational Linguistics: ACL 2023, pages 3732–3755, Toronto, Canada. Association for Computational Linguistics. Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. 2022. Crosslingual generalization through multitask finetuning. ArXiv preprint, abs/2211.01786. Jessica Ojo, Kelechi Ogueji, Pontus Stenetorp, and David I Adelani. 2023. How good are large language models on african languages? ArXiv preprint, abs/2311.07978. Lucas Page-Caccia, Edoardo Maria Ponti, Zhan Su, Matheus Pereira, Nicolas Le Roux, and Alessandro Sordoni. 2024. Multi-head adapter routing for cross-task generalization. Advances in Neural Information Processing Systems, 36. Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics. Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaïd Harchaoui. 2021. MAUVE: measuring the gap between neural text and human text using divergence frontiers. In Advances in Neural Information Processing Systems 34:
Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. 2022b. Cross-task generalization via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3470–3487, Dublin, Ireland. Association for Computational Linguistics.
Lucas Page-Caccia, Edoardo Maria Ponti, Zhan Su, Matheus Pereira, Nicolas Le Roux, and Alessandro Sordoni. 2024. Multi-head adapter routing for cross-task generalization. Advances in Neural Information Processing Systems, 36.
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.
Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaïd Harchaoui. 2021. MAUVE: measuring the gap between neural text and human text using divergence frontiers. In Advances in Neural Information Processing Systems 34:
Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 4816–4828.
Ofir Press, Noah A. Smith, and Mike Lewis. 2022. Train short, test long: Attention with linear biases enables input length extrapolation. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.
Evgeniia Razumovskaia, Goran Glavaš, Anna Korhonen, and Ivan Vuli´c. 2023. Sqatin: Supervised instruction tuning meets question answering for improved dialogue nlu. ArXiv preprint, abs/2311.09502.
Ohad Rubin and Jonathan Berant. 2023. Longrange language modeling with self-retrieval. ArXiv preprint, abs/2306.13421.
ebastian Ruder, Jonathan Clark, Alexander Gutkin, Mihir Kale, Min Ma, Massimo Nicosia, Shruti Rijhwani, Parker Riley, Jean-Michel Sarr, Xinyi Wang, John Wieting, Nitish Gupta, Anna Katanova, Christo Kirov, Dana Dickinson, Brian Roark, Bidisha Samanta, Connie Tao, David Adelani, Vera Axelrod, Isaac Caswell, Colin Cherry, Dan Garrette, Reeve Ingle, Melvin Johnson, Dmitry Panteleev, and Partha Talukdar. 2023. XTREME-UP: A user-centric scarce-data benchmark for under-represented languages. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1856–1884, Singapore. Association for Computational Linguistics.
Oscar Sainz, Jon Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, and Eneko Agirre. 2023. Nlp evaluation in trouble: On the need to measure llm data contamination for each benchmark. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10776–10787.
Oscar Sainz, Jon Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, and Eneko Agirre. 2023. Nlp evaluation in trouble: On the need to measure llm data contamination for each benchmark. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10776–10787.
Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler,
Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler,
Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Févry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M. Rush. 2022. Multitask prompted training enables zero-shot task generalization. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.
Uri Shaham, Jonathan Herzig, Roee Aharoni, Idan Szpektor, Reut Tsarfaty, and Matan Eyal. 2024. Multilingual instruction tuning with just a pinch of multilinguality. ArXiv preprint, abs/2401.01854.
Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, et al. 2022. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations.
Oleh Shliazhko, Alena Fenogenova, Maria Tikhonova, Vladislav Mikhailov, Anastasia Kozlova, and Tatiana Shavrina. 2022. mgpt: Fewshot learners go multilingual. ArXiv preprint, abs/2204.07580.
Sunayana Sitaram, Monojit Choudhury, Barun Patra, Vishrav Chaudhary, Kabir Ahuja, and Kalika Bali. 2023. Everything you need to know about multilingual llms: Towards fair, performant and reliable models for languages of the world. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts), pages 21–26.
shaan Tanwar, Subhabrata Dutta, Manish Borthakur, and Tanmoy Chakraborty. 2023. Multilingual LLMs are better cross-lingual incontext learners with alignment. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6292–6307, Toronto, Canada. Association for Computational Linguistics.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. ArXiv preprint, abs/2307.09288.
Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022a. Finetuned language models are zero-shot learners. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.
Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022b. Finetuned language models are zero-shot learners. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022c. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022.
Xiangpeng Wei, Haoran Wei, Huan Lin, Tianhao Li, Pei Zhang, Xingzhang Ren, Mei Li, Yu Wan, Zhiwei Cao, Binbin Xie, et al. 2023. Polylm: An open source polyglot large language model. ArXiv preprint, abs/2307.06018.
dina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112–1122, New Orleans, Louisiana. Association for Computational Linguistics.
Genta Indra Winata, Andrea Madotto, Zhaojiang Lin, Rosanne Liu, Jason Yosinski, and Pascale Fung. 2021. Language models are few-shot multilingual learners. In Proceedings of the 1st Workshop on Multilingual Representation Learn-
ing, pages 1–15, Punta Cana, Dominican Republic. Association for Computational Linguistics.
Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483–498, Online. Association for Computational Linguistics.
Junjie Ye, Xuanting Chen, Nuo Xu, Can Zu, Zekai Shao, Shichun Liu, Yuhan Cui, Zeyang Zhou, Chao Gong, Yang Shen, et al. 2023. A comprehensive capability analysis of gpt-3 and gpt-3.5 series models. ArXiv preprint, abs/2303.10420. Jun Zhao, Zhihao Zhang, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024. Llama beyond english: An empirical study on language capability transfer. ArXiv preprint, abs/2401.01055.
Junjie Ye, Xuanting Chen, Nuo Xu, Can Zu, Zekai Shao, Shichun Liu, Yuhan Cui, Zeyang Zhou, Chao Gong, Yang Shen, et al. 2023. A comprehensive capability analysis of gpt-3 and gpt-3.5 series models. ArXiv preprint, abs/2303.10420.
Jun Zhao, Zhihao Zhang, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024. Llama beyond english: An empirical study on language capability transfer. ArXiv preprint, abs/2401.01055.
Task
Instruction Text
ID
The aim is to understand user’s intent from the
utterance.
Include all applicable options exactly as they
are provided. Separate the classes
by hyphen.
If no options are applicable, return an empty
string.
Options:
- to deny something
- to ask about savings account
<list of all options applicable in the domain>
Utterance: {demonstration1}
Intents: {intent1}-{intent2}-{intent3}
<all in-context demonstrations>
Utterance: {test example}
Intents:
VE
The aim is to extract slot values from the user
utterance.
Use $$ as delimiter between slot-value pairs.
The slot values should be tagged as:
- amount_of_money: specific amount of money
- adults: number of adults
<list of all slot classes applicable in a given
domain>
Utterance: {demonstration1}
Values: {slot_class1}:{value1}$${slot_class2}:{value2}
<all in-context demonstrations>
Utterance: {test example}
Values:
NLI
The aim is to determine whether the premise
entails, contradicts or is neutral
with respect to the hypothesis. Only output the
label.
Premise: {premise-demonstration1}
Hypothesis: {hypothesis-demonstration2}
Does the premise entail, contradict, is neutral to
the hypothesis?
Answer: {label1}
<all in-context demonstrations>
Premise: {premise-test}
Hypothesis: {hypothesis-test}
Does the premise entail, contradict or is neutral
to the hypothesis?
Answer:
Table 5: Text of instructions used in ICL. ID and NLI: instructions were adapted from Flan (Chung et al., 2022) with intent descriptions from the ontology provided with NLU++ (Casanueva et al., 2022). VE: instructions were adapted from XTREME-UP (Ruder et al., 2023) and Ojo et al. (2023).
Hyperparameter
Value
LaBSE+CL: dim
512
LaBSE+CL: non-linearity
tanh
Batch size
32
Learning rate
2e-5
Weight Decay
0.1
Evaluation Frequency
500 steps
Max Epochs
500
Optimiser
AdamW
Table 6: Fine-tuning hyperparameters used across supervised training experiments. The rest of the parameters were set to the default values in Huggingface Transformers.
# C Full Experimental Results
# D Further Value Extraction Results
D Further Value Extraction Results
In-domain results
Cross-domain results
Samples
AM
EN
MR
ES
TR
AM
EN
MR
ES
TR
SFT: LaBSE+CL
30
0.2998
0.3253
0.308
0.3295
0.3224
0.2041
0.1978
0.1507
0.178
0.1773
50
0.3502
0.3863
0.3679
0.4014
0.3826
0.2362
0.2305
0.1700
0.2123
0.1962
100
0.4409
0.5007
0.4773
0.4836
0.4815
0.2688
0.2774
0.2304
0.2485
0.2432
500
0.6606
0.7509
0.7235
0.7412
0.7328
0.4728
0.5204
0.4780
0.4936
0.5169
1000
0.7116
0.7978
0.7736
0.7900
0.7825
0.5119
0.5759
0.5225
0.5516
0.5539
SFT: XLM-R
30
0.1434
0.1435
0.1457
0.1857
0.1317
0.0284
0.0456
0.0463
0.0799
0.0544
50
0.1676
0.1946
0.1696
0.2060
0.1750
0.0200
0.0042
0.0226
0.0318
0.0021
100
0.2879
0.3363
0.3115
0.3421
0.288
0.1196
0.1176
0.0848
0.1009
0.1123
500
0.5882
0.742
0.6592
0.7075
0.6898
0.4107
0.5076
0.4441
0.4694
0.4806
1000
0.6715
0.8066
0.7391
0.7862
0.7721
0.4943
0.5822
0.5282
0.5223
0.5584
SIT: Flan-T5-Base
30
0.156
0.6625
0.1542
0.4969
0.2727
0.0750
0.5369
0.0677
0.4081
0.1419
50
0.1520
0.7110
0.1434
0.5468
0.3282
0.0999
0.5794
0.0904
0.4535
0.1888
100
0.1432
0.7483
0.1501
0.6242
0.3865
0.1168
0.6103
0.0638
0.4882
0.2094
500
0.1769
0.8601
0.1780
0.7873
0.6341
0.1716
0.7355
0.1620
0.6421
0.4434
1000
0.2040
0.8877
0.1680
0.8333
0.6957
0.1699
0.7602
0.1679
0.7017
0.5453
SIT: mT0-Base
30
0.0560
0.3735
0.1558
0.2646
0.1505
0.0125
0.097
0.0269
0.0684
0.0228
50
0.0962
0.5375
0.3309
0.4614
0.3068
0.0169
0.2868
0.1059
0.1957
0.0825
100
0.2613
0.68
0.5142
0.6169
0.516
0.0936
0.4795
0.3009
0.4209
0.3151
500
0.6488
0.8222
0.7466
0.7978
0.7579
0.5394
0.6711
0.5985
0.6441
0.6163
1000
0.6980
0.8559
0.7889
0.8393
0.8157
0.5892
0.7113
0.6264
0.6798
0.6681
ICL: Flan-T5-XL
0.0328
0.4927
0.0302
0.4526
0.3136
0.0554
0.5375
0.0581
0.4176
0.3012
ICL: mT0-XL
0.0361
0.064
0.0471
0.0460
0.0336
0.0969
0.0989
0.0947
0.1049
0.1006
ICL: GPT-3.5
0.1919
0.6422
0.4828
0.5825
0.4612
0.1501
0.5552
0.3283
0.4728
0.4320
Table 7: Per-language intent detection results for in-domain and cross-domain setups.
In-domain results
Cross-domain results
Samples
AM
EN
MR
ES
TR
AM
EN
MR
ES
TR
SFT: XLM-R
30
0.1566
0.2748
0.1953
0.2444
0.2681
0.0275
0.0336
0.0369
0.03
0.0232
50
0.2199
0.3234
0.2603
0.3221
0.3098
0.049
0.0543
0.0329
0.0462
0.031
100
0.4003
0.4991
0.3598
0.4615
0.4665
0.0362
0.0279
0.0229
0.0469
0.0072
500
0.6130
0.7392
0.6118
0.6508
0.6937
0.036
0.06
0.05
0.103
0.098
1000
0.6468
0.7801
0.6614
0.6855
0.7539
0.05
0.087
0.083
0.137
0.117
SIT: Flan-T5-Base
30
0.0191
0.327
0.0156
0.2091
0.1174
0.0019
0.2514
0.0018
0.07
0.0511
50
0.0362
0.4486
0.0083
0.2627
0.1537
0.009
0.3006
0.0031
0.1089
0.0887
100
0.0555
0.5728
0.0198
0.3678
0.2705
0.0103
0.4043
0.005
0.1987
0.1395
500
0.0896
0.7314
0.042
0.5073
0.4615
0.0313
0.5956
0.0141
0.3689
0.3272
1000
0.1055
0.8041
0.0484
0.5707
0.5552
0.0445
0.6244
0.014
0.3975
0.3577
SIT: mT0-Base
30
0.1193
0.3182
0.1246
0.2893
0.1886
0.0433
0.1615
0.0688
0.1162
0.1011
50
0.1774
0.3954
0.1899
0.347
0.2488
0.0511
0.2153
0.1118
0.1679
0.1371
100
0.3313
0.58
0.3167
0.4723
0.4055
0.0972
0.3345
0.14
0.212
0.1927
500
0.6093
0.779
0.5596
0.6458
0.6596
0.3864
0.5838
0.3562
0.