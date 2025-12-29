firstname.lastname@linguacustodia.com 2Univ. Grenoble Alpes, CNRS, Grenoble INP, LIG, 38000 Grenoble, France
# Abstract
Natural language processing (NLP) has recently gained relevance within financial institutions by providing highly valuable insights into companies and markets’ financial documents. However, the landscape of the financial domain presents extra challenges for NLP, due to the complexity of the texts and the use of specific terminology. Generalist language models tend to fall short in tasks specifically tailored for finance, even when using large language models (LLMs) with great natural language understanding and generative capabilities. This paper presents a study on LLM adaptation methods targeted at the financial domain and with high emphasis on financial sentiment analysis. To this purpose, two foundation models with less than 1.5B parameters have been adapted using a wide range of strategies. We show that through careful fine-tuning on both financial documents and instructions, these foundation models can be adapted to the target domain. Moreover, we observe that small LLMs have comparable performance to larger scale models, while being more efficient in terms of parameters and data. In addition to the models, we show how to generate artificial instructions through LLMs to augment the number of samples of the instruction dataset.
 26 Jan 2024
arXiv:2401.14777v1
Natural Language Processing (NLP) has become an increasingly important field in the financial industry, with applications ranging from sentiment analysis and named entity recognition to question answering. Information retrieved using machine learning from financial reports, news or posts in social media can be used as indicators of companies’ performance or as insights of a market. Many industry actors are interested in extracting this information to use it as a resource that can provide them
∗Corresponding author.
∗Corresponding author.
with a competitive advantage, such as firms forecasting internal future benefits and losses, investors extracting differential information for trading purposes or any practitioner interested in tracking financial assets. Nevertheless, some characteristics of financial text make these tasks especially challenging for models that have been trained on general domain data. The use of specific terminology along with the high complexity of the documents, leads these generalist language models to underperform on financial tasks, which suggests that domain adaptation might be required to improve accuracy of interpretation and analysis. Furthermore, the rapid evolution of large language models (LLMs) and their proven capabilities for NLP tasks has made them stand out and become an interesting option to study. Due to the fact that even the best general language models fall short for some financial tasks, some proposals have been recently presented for a financial domain adaptation of LLMs. These models tailored for finance, such as BloombergGPT (Wu et al., 2023), have been introduced as multitasking generative models specifically designed for financial text understanding and generation. However, these fine-tuned models still show room for improvement, both in performance and in the efficiency of the proposed training strategies. This paper tackles various aspects of adapting LLMs to the financial domain. In particular, we explore diverse strategies of domain adaptation and fine-tuning of LLMs for financial sentiment analysis, and conduct a series of experiments over two different foundation models. The study focuses particularly on smaller manageable models, up to 1.5B parameters, in order to explore the possibilities of models that can be accessible with relatively low hardware requirements. Although the adapted models are smaller than the current state-of-the-art ones, results show that they achieve similar or higher performance. In addition, a curated data collection
with two main datasets is also presented. One constructed with financial documents and reports, and the other a set of instructions for financial tasks. We show, step by step, the process of creating these datasets and particularly focus on the use of more powerful LLMs to generate synthetic instructions to fine-tune smaller LLMs. Finally, apart from the main focus of the study which is on financial sentiment analysis, other tasks have also been evaluated to analyze the multitasking capabilities of our models.
# 2 Related work
Sentiment analysis is one of the most common use cases of NLP. In this task, a model classifies a text according to the sentiment detected, usually between positive, negative and neutral. However, while any sentiment analysis model would be capable of undertaking financial sentiment analysis, an adaptation is required. In this section, the evolution of sentiment analysis in the financial domain is studied using models based on Transformers (Vaswani et al., 2023). FinBERT1 (Araci, 2019) is based on the idea of training a BERT (Devlin et al., 2018) model in two steps to adapt it to the financial domain and the sentiment analysis task. The first step consists of further pre-training the model on financial documents, as this strategy has already be proven to be effective (Howard and Ruder, 2018) for domain adaptation. This step aims at helping the model to understand financial terminologies better than the base model. Authors used a subset of Reuters’ TRC24 dataset2, a collection of news articles published by Reuters that was filtered with keywords related to finance, to fine-tune the model. In the second step, the model is prepared for the sentiment analysis task by adding a dense layer to the last hidden state of the classification token CLS of the encoder-based architecture, a recommended practice for classification with BERT. This task is finetuned using the Financial PhraseBank (FPB) (Malo et al., 2014), a financial sentiment analysis dataset. FinBERT presents remarkable results on financial sentiment analysis, outperforming the state-of-theart. Nevertheless, the model is strongly limited to sentiment analysis and underperforms greatly on other tasks.
1Several models under the name of FinBERT exist, how ever in this work we only discuss the first model. 2https://trec.nist.gov/data/reuters/reuters. html
# 2.1 Base large language models
Recent advances in the field of large language models (LLMs) have shown that these models can achieve remarkable capabilities in understanding complex natural language. They are also capable of performing zero-shot and few-shot learning, in which they can generate accurate responses for tasks that they have not seen during training (Radford et al., 2019). This makes LLMs a great choice in multitask settings where one model is expected to perform several tasks. Most of today’s LLMs are based on Transformer models (Vaswani et al., 2023), typically set in decoder-only architectures. Training of LLMs is typically split in two stages. The first part of the training is the most computationally expensive since the model is trained using large amounts of text. For this reason, conducting the training of a LLM from scratch requires high computational resources. Nevertheless, many research groups and companies are releasing these models to the public to be used as base or foundation for other models, to enable research to move forward. Using these pre-trained models is highly beneficial for researchers with fewer data or hardware resources, as they can be used as a starting point for fine-tuning on specific tasks, such as chatting, following instructions or giving outputs in a specific style or format. Although most LLMs are trained on general domain data, there have been a few works recently to adapt LLMs to the financial domain. In the next subsections two such work are reviewed.
# 2.2 Financial large language models
BloombergGPT. One of the first decoderonly LLMs trained specifically for finance is BloombergGPT (Wu et al., 2023), a model of 50B parameters based on BLOOM’s architecture (Scao et al., 2023). The corpus collected for the training of this LLM consisted in the combination of 363 billion tokens from financial documents with 345 billion tokens from general purpose datasets. The model was trained from scratch, without using any foundation model as a base, with the objective of predicting the next token of the documents, and without fine-tuning on instructions. However, the results presented by BloombergGPT are far from the ones achieved by other models, some of them of a much smaller scale. In addition, the results reported did not outperform other generalist LLMs, as we will show later in this paper.
FinMA. The open model FinMA from PIXIU’s framework (Xie et al., 2023) introduced by ChanceFocus reported better scores on several financial tasks than larger generalist LLMs, such as GPT4 (OpenAI, 2023), and BloombergGPT. They used LLaMA (Touvron et al., 2023a) as the pre-trained model and fine-tuned it with instructions tailored for financial multitasking. The instruction dataset consists of texts formed by an instruction, an input and an answer. The dataset includes a data augmentation strategy in which the inputs of those tasks with few samples were used with 10 different instructions. This augmentation strategy, while increasing the number of samples in the dataset, did not increase its diversity as the same set of 10 instructions were always repeated. In the same paper in which FinMA was presented, the PIXIU framework also included FLARE, a financial evaluation benchmark. This benchmark has been used to evaluate the experiments carried out in this project.
# 2.3 Financial benchmark
For the evaluation of large language models, the FLARE benchmark3 from PIXIU framework has been used. The tasks of this benchmark which are relevant to our work are presented below. Financial Sentiment Analysis. Financial sentiment analysis task over two different benchmarks, the Financial Phrase Bank (FPB) (Malo et al., 2014) and FIQA-SA (Maia et al., 2018). News Headline Classification. Headlines task contains 9 different subtasks, each one associated with 9 different gold questions, in which the expected answers are “yes” or “no”. The inputs analyzed are gold news from the Gold dataset (Sinha and Khandait, 2020). Named Entity Recognition. NER task is based on detecting financial named-entities in U.S. public agreements in the (Salinas Alvarado et al., 2015) dataset. The tagged entities correspond to people, organizations and locations.
# 3 Methodology
In this section, we describe the methods designed to conduct the experiments. First, we list the foundation models that are used as a part of this project. Then, two new dataset collections are introduced, one with data based on documents and the second with instructions. We also give details of designing
3https://github.com/chancefocus/PIXIU
a data augmentation strategy for the instructions as well as the description of the training process carried out to fine-tune the foundation models.
# 3.1 Foundation models
As stated earlier, the focus in this work is on smaller sized models that can be adapted to achieve performance of larger models. The two models that we use are listed below: OPT. Meta AI’s large language models suite OPT (Open Pre-trained Transformers) (Zhang et al., 2022) were presented as a collection of 9 models ranging from 125M to 175B parameters, being one of the first publicly available LLMs. Pythia. EleutherAI presented Pythia (Biderman et al., 2023), a suite of decoder-only language models with sizes ranging from 70M to 12B parameters. These models are trained on the Pile dataset (Gao et al., 2020), a curated collection of English texts from a wide variety of sources.
# 3.2 Datasets
In order to train LLMs, two main different approaches can be taken with respect to data. When a model is trained from scratch, the data used are collections of documents, for which the model has the objective of predicting the next token. This is usually the training carried out to obtain foundation LLMs. However, these models can be further pre-trained for domain adaptation, in the same way that FinBERT was trained. This approach is based on the idea of continuing the training of the model with financial documents to shift from a general to a financial language model. Moreover, it has been proven that large language models can improve their performances, especially on unseen (or zero-shot) tasks by fine-tuning them to follow instructions. For this fine-tuning method, the training objective is the same, predicting the next token of the text, with the only difference being that the format of this data relies on an “instruction”, “input”, “answer” format. For this project, one dataset was collected for each of these two training strategies. In addition, the instruction-based dataset was augmented artificially with samples generated from another LLM (LLaMA 2 13B (Touvron et al., 2023b)). Document dataset. The collection of documents used to further pre-train the base LLMs is a combination of general and financial documents from different sources. The purpose of this mixture is to add diversity to the training data, with finance
being the most represented domain. Having general data in the training set prevents the model to completely drift the domain and result in a model that is unable to understand general language. The data sources of these documents are described below:
 EDGAR Files (Financial). EDGAR is the Electronic Data Gathering, Analysis, and Retrieval system online platform operated by the SEC4 (United States Securities and Exchange Commission). It is used by companies to electronically file registration statements, periodic reports, and other forms required by the SEC. The database of these documents is open to everyone, allowing the retrieval of high-quality financial text.
 Reuters News (Financial). Reuters is a news agency specializing in business and finance that released Reuters Corpora, a collection of financial news made available for use in NLP research. The collection used in this dataset is TRC2 (Thomson Reuters Text Research Collection), that contains more than 1.8 million news.
# • In-house Dataset (Financial). As a part o
 In-house Dataset (Financial). As a part of this project, a diverse collection of in-house financial text has been obtained. The text in this dataset is mostly at sentence level, as they were originally used for machine translation. This is the only private set used for the project.
 The Pile (General). The Pile (Gao et al., 2020), from EleutherAI, is a dataset that comprises 22 diverse high-quality subsets, several of which originate from academic or professional sources. The idea behind this dataset’s construction is that diversity enhances general cross-domain knowledge and downstream generalization capability of large language models. It includes data from general news to scientific articles, code, etc...The proportions of these subsets are kept as-is in the subsample used for this project.
The lengths of the documents of this dataset had to be adapted to the models’ context length, which corresponds to the longest sequence of tokens that the model can support. In this project the context was limited to 2048 tokens. The pre-processing of
4https://www.sec.gov/edgar/searchedgar/ companysearch
the dataset consisted in concatenation of all documents from the same source, using a special token (<endoftext>) to separate them. The long concatenated text is then sliced in blocks of 2048 tokens, which are mixed and shuffled with all the other blocks of the dataset. Since some datasets used in this project are extremely large, we decided to take a smaller proportion from each one. The summary of the ratio used for each partition is shown in Table 1.
Subset
Domain
# Tokens
%
EDGAR
Finance
100k
25.7
Reuters
Finance
36k
9.3
In-house
Finance
38k
9.7
The Pile
General
215k
55.3
Total
389k
100
Table 1: Proportion and absolute number of tokens taken from each dataset.
Instruction-based dataset. Instruction finetuning is a strategy used to improve LLMs’ performance for specific tasks by teaching them to follow specific format of questions and answers. LLMs learn by being trained on this specific format of text, while keeping the same training objective, predicting the next sequence of tokens. Fine-tuning on instructions is the most common technique to adapt foundation models to specific use-cases, mainly because this method not only improves performance on the trained tasks, but also augments zero-shot and few-shot capabilities. Models trained on instructions are usually consistent in the format in which data is presented. In Table 2 the format used for our dataset is displayed.
Template
### Instruction: Description of the task
### Input: Input to analyze
### Answer: Answer to predict
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ca93/ca936054-466f-4f69-8b0b-8f73450ba69d.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 2: Template for instructions.</div>
Table 2: Template for instructions.
To create the instruction dataset, we used the instructions dataset published by PIXIU that targeted the FLARE benchmark. However, this dataset has poorly curated prompts and includes a suboptimal data augmentation strategy. For instance, certain parts of the dataset have been up-sampled by using ten different instructions over the same input. Despite having more samples, the up-sampled version
Subset
Instr
PIXIU
Augm
FPB
4,838
48,380
6,633
FiQA-SA
1,173
11,730
2,825
NER
609
6,090
2,609
Headline
102,708
102,708
102,708
FinQA
8,242
8,242
8,242
ConvFinQA
3,892
3,892
3,892
BigData22
0
7,164
0
ACL18
0
27,053
0
CIKM18
0
4,967
0
TOTAL
121,462
220,226
126,909
Table 3: Comparison of the instructions used for each task in the original instructions dataset (Instr) with one input-answer by instruction, the up-sampled version (PIXIU), and the dataset augmented by LLM inference (Augm).
of dataset lacks diversity which may lead to poor performance as discussed by Zhou et al. (2023). For this reason, the dataset proposed in this project has been designed from scratch, only reusing the unique input - answer pairs from a down-sampled version of PIXIU’s dataset that includes a single instruction for each input. Instruction data augmentation. The main idea behind instruction data augmentation is to bring new inputs to the dataset, so the model has more diverse examples to learn from. Two different methods have been defined for generating these instructions dependent on the target task. For sentiment analysis, the model has to generate an input for a given sentiment given an example with that label. This strategy has been used to augment both FPB and FIQA-SA subsets. In Table 7 of Appendix A the template used for this task is presented. For NER, since it is not a sequence classification task, the inference method is different. The first solution proposed was based on letting the model generates both the new sentence and its NER tags at the same time, only guiding the model by including a few examples in the prompt. However, the variety of the sentences generated by the model was too short and the tags were incorrect, indicating that the task was too hard for the model. Our solution was to use existing unlabeled sentences, which reduced the generative task to a tagging process. The sentences used for this augmentation were in-house financial sentences. Moreover, in this case the example given to the model is fixed in order to make sure all types of entities are present in the prompt. The format of the tagging was chosen using prompt
engineering. In Table 8 of Appendix A, the template used for NER data augmentation is shown. The Headline task was not augmented since it had enough samples, even when considering that there are 9 subtasks in the benchmark. Using the above-mentioned two templates, new inputs are inferred to be added to the instructions dataset. The model used for the generation of these new samples is LLaMA-2-13B, quantized in 4-bits to reduce the GPU memory required. Table 3 shows a comparison of the number of samples targeting each task before and after the augmentation. The decision on the number of synthetic samples generated was taken considering the number of original samples. The reason behind not generating even a larger number of instructions is that despite their high-quality, artificial samples could introduce some noise to the dataset by generating sentences too different from the original distribution, introducing erratic input-answer pairs or NER tags or to duplicate some inputs after several iterations. In Table 3 there is a comparison between PIXIU’s dataset before and after down-sampling as well as after the data augmentation.
# 3.3 Training method
As stated earlier, in this work, we use two pretrained foundation models, namely Pythia-1.4B and OPT-1.3B, and fine-tune them in two stages as detailed below:
 Further pre-training. The models are finetuned to predict the next token of the text in the document-based dataset, following the same idea as in FinBERT and without being fine-tuned on a specific task. The idea here is to tilt the models to become more familiar with the financial domain. Both models are trained on a total of 389, 000 tokens introduced in context blocks of 2, 048 tokens. The models are trained for two epochs, saving 4 checkpoints at every epoch. The best checkpoint is selected for each model.
structed to perform financial tasks using the instructions dataset. Since the length of these instructions is generally shorter than on the document-based dataset, the context length is reduced to 1, 000 tokens to speed up the training. Sequences shorter than this length are padded with a padding token. Instructions
longer than 1, 000 tokens are cut off. For instruction fine-tuning, models are trained for 1 epoch.
For both set-ups, training is performed with AdamW optimizer (Loshchilov and Hutter, 2019), a batch size of 32, and applying gradient accumulation of 4 for training efficiency. The initial learning rate is set to 1e-4, while the weight decay is adjusted to 0.1. These values remained the same for all the conducted experiments. The training of these models was carried out on a H100 GPU.
# 4 Results
# 4.1 Classical algorithms versus LLMs for financial sentiment analysis
Prior to conducting an evaluation of our fine-tuned LLMs for financial sentiment analysis, we study the performance of the current state-of-the-art models and classical machine learning algorithms. Compared to LLMs, classical algorithms do not require a lot of computation, they could be easily trained and tested. For the sake of simplicity, the evaluation has been carried out only on the FPB. Based on the results in Table 4, the lowest score, unsurprisingly, is obtained by the lexicon approach. Classical machine learning algorithms on the other hand are able to obtain results considerably higher than lexicon, and even match or pass LLM scores in some cases. Overall conclusions that can be depicted from the results can be summarized as follows:
• The domain adaptation and training of FinBERT on this specific task, gives the model an advantage over general models. Comparing FinMA-30B with GPT-4, it can be seen that a smaller model fine-tuned for finance has better performance than a generalist one.
• BloombergGPT was a good starting point for financial LLMs. However, its performance on tasks like sentiment analysis is poor. One likely reason is that this model has not been fine-tuned on instructions.
 FinMA-30B proves the relevance of finetuning on instructions to improve performance on financial tasks. Nevertheless, as mentioned before, the train dataset might be not sufficiently diverse, which may impact the model’s capability in real-world scenario.
Algorithm and features
Accuracy
Lexicon approach
Loughran-McDonald dictionary
0.59
Classical ML algorithms
SVM
0.77
Naive Bayes
0.73
XGB
0.80
Transformers approach
FinBERT
0.85
GPT-4
0.71
BloombergGPT
-
FinMA-30B
0.87
Table 4: Performance of financial sentiment analysis. A comparison between traditional approaches and modern transformer based models.
# 4.2 Financial domain adaptation
In this section we show the impacts of the two stage fine-tuning as well as improvements brought by the artificially augmented instruction dataset. Models are evaluated using a subset of the tasks proposed in the FLARE benchmark: FPB, FIQA-SA, Headlines and NER. For the classification tasks (FPB, FIQA-SA and Headlines), the predictions are obtained by forcing the model to generate one of the expected class label. For example, in FPB, this means to choose the next token only amongst the ones needed to generate the labels (positive, negative or neutral), and sticking with the most probable ones (the highest logits). When evaluating on NER, the generation is not constrained. The first experiment that we carry out is to see the effects of fine-tuning on documents versus instructions. Based on the results of Table 5, it is clear that performance of both Pythia and OPT models show similar behaviors and that fine-tuning brings significant improvements over the base models. Particularly, instruction fine-tuning improvement is much higher than just further pre-training on documents. This conclusion seems to be aligned with what we observe in the literature of instruction tuning of other domains. Next, in order to evaluate the effect of augmenting the number of instructions using the strategy designed for this dataset, the models are compared after fine-tuning with the base instructions dataset and with the augmented instructions dataset. The results of using data augmentation are not straightforward. In Table 5, it can be seen that the performance of the models augmented instructions is improved for the sentiment analysis tasks, but the scores goes down for the other two. In the
F1 scores
Fine-tuning data
FPB
FIQA-SA
Headlines
NER
Pythia-1.4B
(base)
0.20
0.29
0.16
0
Docs
0.41
0.16
0.57
0.30
Instr
0.82
0.73
0.93
0.59
Augmented instr
0.82
0.79
0.90
0.56
Docs + Augmented instr
0.84
0.83
0.97
0.69
OPT-1.3B
(base)
0.19
0.48
0.29
0
Docs
0.13
0.58
0.39
0
Instr
0.84
0.77
0.93
0.53
Augmented instr
0.86
0.79
0.97
0.29
Docs + Augmented instr
0.86
0.81
0.96
0.34
Table 5: Comparison of Pythia-1.4B and OPT-1.3B fine-tuned with different strategies. The results reported correspond to the base models without fine-tuning (base), models with document further pre-training (Docs), models fine-tuned on instructions (Instr), models fine-tuned on augmented instructions dataset (Augmented instr), fine-tuning first with documents and then with augmented instructions (Docs + Augmented instr).
case of Headlines, this effect can be caused by the fact that this task is the most represented in the dataset and, by introducing new samples, the model is less focused on this task. For NER, the issue can be explained by the difference between the text of the synthetic samples and the original test set. As explained in previous sections, NER is augmented using in-house data, and even though the chosen sentences were also in the financial domain, the sources are different and that might have introduced errors in the predictions. Finally, we can test the implications of instruction fine-tuning after further pre-training the model with the financial documents. This simply means that the model is fine-tuned two times. Since the augmented instructions proved to be better than the original instructions, this experiment is conducted on the earlier instruction dataset. As shown in the last row of Table 5, this approach seems to lead to a higher score in every task. Therefore, the domain adaptation method inspired by FinBERT’s training strategy, proves to be effective for decoder-only LLMs and not only for financial sentiment analysis, but for multiple financial NLP tasks.
# 4.3 Comparison with other Financial LLMs
In this section, the results of the best models obtained through the previous experiments (Docs + Augmented instr) are compared against the stateof-the-art LLMs for finance. As can be seen in Table 6, both fine-tuned Pythia-1.4B and OPT-1.3B over perform GPT-4 in classification tasks, which includes financial sentiment analysis. This is made possible because of the domain adaptation conducted for these two base models. For NER, which
<div style="text-align: center;">F1 scores</div>
is a generative task, GPT-4 is still the LLM with the highest score. When the models of these projects are compared to BloombergGPT, the biggest current LLM tailored for finance, it can be observed that the scores obtained are much higher for classification tasks, specially for sentiment analysis, and that Pythia also obtains better score for NER. In terms of efficiency, these results are achieved with models that have approximately 97% fewer training parameters than BloombergGPT5. In the comparative with the collection of FinMA models, the PIXIU LLMs still outperform the models fine-tuned with our domain adaptation strategy in some tasks, specially when compared to FinMA30B. However, when FinMA-7B, the model with the closest size to the models presented in this project, is evaluated in financial sentiment analysis and Headlines, it can be observed that the scores are almost equivalent to the fine-tuned Pythia-1.4B and OPT-1.3B. In this case, however, the biggest improvement with respect to FinMA-7B is in terms of efficiency. Pythia-1.4B and OPT-1.3B have approximately 78% fewer training parameters than FinMA-7B, and the number of instructions used goes from 220, 226 down to 126, 909, which is only a 57% of the number of samples used for PIXIU models. Therefore, from the general comparison it can be seen that the models fine-tuned in this project over perform most LLMs in financial tasks, with the only exception of FinMA models. In addition, the size of the models and the training strategy
5BloombergGPT has 50B trainable parameters. Pythia1.4B and OPT-1.3B have approximately 1.5B parameters. The amount of data used is not comparable since BloombergGPT was trained from scratch.
F1 scores
Models
FPB
FIQA-SA
Headlines
NER
BloombergGPT
0.51
0.75
0.82
0.61
GPT-4
0.78
-
0.86
0.83
FinMA-7B
0.86
0.84
0.98
0.75
FinMA-30B
0.88
0.87
0.97
0.62
Pythia-1.4B
0.84
0.83
0.97
0.69
OPT-1.3B
0.86
0.81
0.96
0.34
Table 6: Comparison of state-of-the-art with Pythia-1.4B and OPT-1.3B fine-tuned on documents and the augmented dataset. Performance of models retrieved from BloombergGPT and PIXIU papers. BloombergGPT is not a publicly available model, so it is not possible to evaluate it under the same conditions as the other models. Thus, ChatGPT, GPT-4 and FinMA-30B are evaluated on zero-shot, BloombergGPT is only reported in a five-shot setting and its accuracy was not published.
have been proven to be more efficient than the ones proposed for other models.
# 5 Conclusion
This project has covered a wide range of aspects of financial LLMs. Through a series of experiments, using Pythia-1.4B and OPT-1.3B as base models, we studied the adaptation of relatively small LLMs for finance. The experiments we conducted first show that LLMs adapted to the financial domain through further pre-training followed by instruction fine-tuning perform better than some of the best current generalist LLMs (such as GPT-4) on financial tasks. Second, it validates our training strategy since our LLMs obtain higher or similar scores than other financial LLMs that were trained with much more parameters and larger datasets. Lowering the requirements to fine-tune LLMs for this specific industry can be key for the future of several companies, since it can enable smaller organizations to host their own LLMs or, at least, to make them more accessible. Furthermore, it is worth mentioning that the models used for this project as well as most datasets, except for the inhouse subset (only 9.7% of the documents dataset), are open and publicly available. In addition to the findings related to domain adaptation of LLMs for financial tasks and the models presented, a strategy for the generation of samples for the instructions dataset is introduced. Moreover, the two datasets used for the project are described with enough details to be reproduced by other researchers. Finally, the paper also presented a comprehensive study that delves into the state-of-the-art and the evolution of approaches for financial sentiment analysis, ranging from traditional dictionary-based methods to the more recent advancements in LLMs.
<div style="text-align: center;">F1 scores</div>
Despite the fact that the results showed great performance of the small-sized models, in further research these fine-tuning strategies could be applied to larger models and study their impact on different scales and domains. An interesting option to study in the future are Low-Rank Adapters or LoRA (Hu et al., 2021), a method that reduces the number of trainable parameters by freezing the foundation model weights and injecting trainable rank decomposition matrices into each layer of the LLM.
# Limitations
The limitations of this work can be summarized as following:
• Generative capabilities: The final fine-tuned model seems to perform very well on classification tasks such as sentiment analysis, while still lagging behind in generative ones.
• Unseen tasks: our work concentrates on certain tasks that have been studied in previous similar work, but for a full understanding of its limitations, one needs to test it on unseen tasks.
• Large models: we believe that testing the same strategy of multiple fine-tuning stages would yield even better results with larger models such as LLaMA-2-7B or even larger models.
• Large models: we believe that testing the same strategy of multiple fine-tuning stages would yield even better results with larger models such as LLaMA-2-7B or even larger models.
# References
Dogu Araci. 2019. Finbert: Financial sentiment analysis with pre-trained language models.
Dogu Araci. 2019. Finbert: Financial sentiment analysis with pre-trained language models.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d054/d054ce30-5442-4cee-b0d6-311b38aa652f.png" style="width: 50%;"></div>
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models.
# OpenAI. 2023. Gpt-4 technical report.
Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, and et al. 2023. Bloom: A 176bparameter open-access multilingual language model. Ankur Sinha and Tanmay Khandait. 2020. Impact of
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2023. Attention is all you need. hijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023. Bloomberggpt: A large language model for finance. Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. 2023. Pixiu: A large language model, instruction data and evaluation benchmark for finance. usan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. Opt: Open pretrained transformer language models. Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023. Lima: Less is more for alignment. A Instruction data augmentation
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models.
Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023. Lima: Less is more for alignment.
# A Instruction data augmentation examples
Template
Write a sentence with a {yi} financial sentiment.
Use the format <stc> sentence </stc>.
Reuse
terms from the example.
Example: ’<stc> {xi}
</stc>’
Example
Write
a
sentence
with
a
positive
financial
sentiment. Use the format <stc> sentence </stc>.
Reuse terms from the example. Example: ’<stc>
Shares of Standard Chartered ( STAN ) rose 1.2
% in the FTSE 100 , while Royal Bank of Scotland
( RBS ) shares rose 2 % and Barclays shares (
BARC ) ( BCS ) were up 1.7 % . </stc>’
Table 7: Template for sentiment analysis input generation with financial sentiment fixed and dynamic shot. Example given for positive input inference. {xi, yi} is an input-answer pair sampled from one of the two subsets.
Template
Identify the named entities that represent a
person (’PER’), an organization (’ORG’), or
a
location
(’LOC’)
in
a
financial
context.
Use the format ’Entities: entity name, entity
type’.
Sentence: ’The Bank gave money to the Borrower
to open a business in New York.’;
Entities:
’Bank, ORG | Borrower, PER | New York, LOC’
Do the same with this sentence, identifying
’PER’, ’ORG’, ’LOC’ entities.
Sentence: {xi}; Entities:
Example
Identify the named entities that represent a
person (’PER’), an organization (’ORG’), or
a
location
(’LOC’)
in
a
financial
context.
Use the format ’Entities: entity name, entity
type’.
Sentence: ’The Bank gave money to the Borrower
to open a business in New York.’;
Entities:
’Bank, ORG | Borrower, PER | New York, LOC’
Do the same with this sentence, identifying
’PER’, ’ORG’, ’LOC’ entities.
Sentence:
‘350 ,
Wellesley ,
Massachusetts
02481 doing business as " Silicon Valley East
" and AKAMAI TECHNOLOGIES , INC . (" Borrower
"), whose address is 201 Broadway , 4th Floor
, Cambridge , Massachusetts 02139 provides the
terms on which Bank will lend to Borrower and
Borrower will repay Bank’; Entities:
Table 8: Template for NER tags generation given a sentence of the financial domain.
