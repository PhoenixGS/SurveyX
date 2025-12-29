# GigaCheck: Detecting LLM-generated Content
Irina Tolstykh irinakr4snova@gmail.com Aleksandra Tsybina sastsybina@gmail.com Sergey Yakubson serg.yakubson@gmail.com Aleksandr Gordeev asegordeev@gmail.com Vladimir Dokholyan doholyan.vs@phystech.edu Maksim Kuprashevich * mvkuprashevich@gmail.com
9 Nov 2024
mvkuprashevich@gmail.com
Layer Team, R&D Department, SaluteDevices
# Abstract
With the increasing quality and spread of LLM-based assistants, the amount of LLM-generated content is growing rapidly. In many cases and tasks, such texts are already indistinguishable from those written by humans, and the quality of generation tends to only increase. At the same time, detection methods are developing more slowly, making it challenging to prevent misuse of generative AI technologies. In this work, we investigate the task of generated text detection by proposing the GigaCheck. Our research explores two approaches: (i) distinguishing human-written texts from LLM-generated ones, and (ii) detecting LLMgenerated intervals in Human-Machine collaborative texts. For the first task, our approach utilizes a general-purpose LLM, leveraging its extensive language abilities to finetune efficiently for the downstream task of LLM-generated text detection, achieving high performance even with limited data. For the second task, we propose a novel approach that combines computer vision and natural language processing techniques. Specifically, we use a finetuned general-purpose LLM in conjunction with a DETRlike detection model, adapted from computer vision, to localize AI-generated intervals within text. We evaluate the GigaCheck on five classification datasets with English texts and three datasets designed for Human-Machine collaborative text analysis. Our results demonstrate that GigaCheck outperforms previous methods, even in out-of-distribution settings, establishing a strong baseline across all datasets.
*Corresponding author
*Corresponding author
# 1. Introduction
The rapid development of Large Language Models (LLMs) in recent years has made LLM-generated text difficult to distinguish from human-written content. This raises concerns, primarily because LLMs are prone to hallucinations [30, 69] and reliance on outdated information, which can lead to the spread of incorrect knowledge. Additionally, there is a risk of such assistants being used maliciously for fraud [20, 62], the spread of spam and misinformation [53, 24], cheating in academic sphere [66, 33, 58, 75], and even intentional harm to society [36]. Thus, there is an urgent need for robust detectors to effectively identify LLMgenerated content. However, detecting generated text can be challenging because the content may be also produced in collaboration with a human, and existing online and opensource detectors are not reliable enough [49, 81], especially in identifying Human-Machine collaborative texts [49]. Existing detection approaches have focused primarily on improving binary classification, i.e., distinguishing AIgenerated from human-written texts. Most of them utilize language models such as BERT [14] and RoBERTa [46] to address this problem. Some recent works analyzing Human-Machine collaborative texts either tackle multiclass classification [90] or divide the text into sections and identify the boundary between sections of different authorship [87, 86, 78]. We propose GigaCheck framework, which includes methods for both classification and identification machine-written parts of a Human-Machine collaborative text. Our method for detecting LLM-generated text parts analyzes the entire text and outputs a set of intervals in characters, making it more universal compared to other approaches. General-purpose LLMs, like GPT-4 [57] or Mistral [31], have demonstrated impressive language understanding.
Their versatility is achieved through extensive training on diverse domain data, making them adaptable to various applications. We propose leveraging the flexibility of generalpurpose LLMs to solve the detection of LLM-generated content through fine-tuning. In realistic scenarios, humans increasingly collaborate with LLMs to co-write text, where binary classification approaches become less effective in detecting generated content. To address the need for a more nuanced analysis of mixed-authorship texts, we propose a novel method for detecting LLM-generated intervals. Our approach replaces previous classifier-based methods with an architecture based on DETR [8]. Specifically, we use a fine-tuned LLM model to generate text representations, which are then processed by an encoder-decoder transformer to detect 1D intervals of LLM-generated content. This formulation treats interval detection as a direct set prediction problem, eliminating the need for the manual post-processing steps often required in prior methods [38, 87, 78]. We conduct extensive experiments, demonstrating that our classification approach outperforms existing methods across five classification datasets. We also assess the generalization capabilities of our approach through out-of-model and out-of-domain experiments, evaluating the model on unseen data or generations from an LLM not used in the training set. Additionally, we evaluate our detection approach for identifying generated intervals in text, showcasing high performance on three datasets designed for Human-Machine collaborative text analysis. Finally, we also assess the detection model’s performance in an out-ofdomain scenario. Overall, we propose GigaCheck, a unified framework comprising two methods: (i) a classification approach that achieves state-of-the-art results across several classification datasets for distinguishing LLM-written texts from humanwritten ones, while also exploring the generalization capabilities of our approach in out-of-distribution settings; (ii) an LLM-generated intervals detection method based on a DETR-like architecture, which establishes a strong baseline on three datasets. To the best of our knowledge, we are the first to apply a DETR architecture for analyzing LLMgenerated content. We hope our work will inspire and encourage further research in this direction.
# 2. Related Works
# 2.1. Machine Generated Text Detection
The task of detecting generated content has been approached by researchers from various perspectives. Numerous works [90, 49, 22, 4, 48, 47, 84, 73] have focused on solving the binary classification task, aiming to distinguish texts generated by machines from those written by humans. Some studies [74, 54, 80, 73, 72, 82]
have also tackled the multiclass classification task, attempting to determine the specific neural language model that was used for text generation.
To solve the binary classification task, a line of works [54, 18, 17, 68, 79] applied statistics-based methods, which typically require access to the investigated LLMs. Such approaches may use metrics like entropy, perplexity, n-gram frequency, and other statistical features for text classification. Another approach involves training neural-based detectors [43, 22, 80, 48, 84, 65, 73, 1], mostly based on the RoBERTa [46] language model, which provides more accurate results than statistical methods [43, 48], but lacks robustness [43, 34, 35, 9, 71]. Some studies also train classification models based on statistical features [82, 23]. Recent works [72, 71, 37] employ topological data analysis to separate human-written texts from machine-generated ones. In [37], a binary logistic regression classifier is built using different types of topological features derived from the transformer model’s attention maps. To solve the multi-class problem, the authors of [72] trains a classification model with features extracted by the transformer model, which are concatenated with topological features derived from these features. [71] uses a one-feature thresholding classifier, where the feature is calculated as the intrinsic dimensionality of a text using the persistent homology dimension estimator. There are also methods that utilize LLMs as detectors. The authors of [4] apply GPT-3.5-turbo [56] and GPT-4 [57] models for the zero-shot binary classification task and demonstrate that both models have a very high misclassification rate. [60] trains an LLM to rewrite human-written texts while leaving AI-generated texts unchanged, then uses the edit distance between the input text and the rewritten output for text classification. [84] fine-tunes their Grover model to detect Grover’s generations. Our method, proposed for distinguishing between real and machine-generated text, utilizes a fine-tuned LLM as a detector and falls under the neural-based type of detectors. Texts may be written in collaboration between a human and a language model. Existing studies [90, 49] utilize neural-based classification models to detect HumanMachine collaborative texts. The authors of [38] address the boundary detection task to determine where humanwritten text ends and machine-generated text begins.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/cdbe/cdbe1810-cbf6-4514-9423-a6a30c0ec44a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Overall architecture of GigaCheck framework. For detecting AI-generated texts, we propose fine-tuning an LLM model. For detecting generated parts of the text, we use a two-stage approach: first, obtaining vector representations that best describe the text using fine-tuned LLM, then using them to identify intervals with a detection transformer model.</div>
tences. Another study [86] solves the sentence-level LLM-generated text detection task through a two-step pipeline, involving segment detection followed by segment classification into three categories: AI-generated, humanwritten, or Human-Machine collaborative. The authors of [78] propose a straightforward detection method by identifying the exact authorship of each sentence. Almost all described methods analyze text at the sentence level, and some have limitations on the number of LLM-generated text parts they can detect. We propose a universal method that outputs LLM-generated intervals at the character level and can predict as many intervals in a text as defined during training.
# 2.2. Transformer-based detection models
DETR [8] is an end-to-end object detector based on transformers. DETR-like architectures have proven effectiveness not only in object detection tasks [93, 26, 29], but also in other tasks such as video action detection [88] and moment retrieval [40, 55, 19], where it is used to find temporal intervals in videos corresponding to a given text query. Inspired by these works, we propose to use a detection transformer model to detect 1D intervals in texts. A number of recent works have proposed modifications of the DETR [8] architecture. For example, the authors of DeformableDETR [92] propose a deformable attention module that only attends to a small set of key sampling points around a reference, helping to speed up model convergence and reduce computational cost. DNDETR [41] introduces a denoising training method, which is also needed to accelerate the training process and improve detection accuracy. For more accurate box prediction, DAB-DETR [45] introduces learnable anchor boxes
as DETR positional queries, where each query is associated with a reference point (anchor) represented in the form of the object’s center coordinates and its dimensions. During the decoding process, the anchors are updated, refining the model’s predictions about the location and size of objects. Building on these advancements, DINO DETR [89] refined key features of both DN-DETR and DAB-DETR and integrated RPN into DETR architecture. CO-DETR [93] utilizes auxiliary training heads to create more efficient and effective DETR-based detectors. In this paper, we rely on DN-DAB-DETR as it provides a strong baseline with high localization accuracy [41].
# 3. Methodology
The overview of the GigaCheck a unified framework comprising two complementary methods, is depicted in Figure 1. It comprises two options: the fine-tuning of an LLM using the LoRA method [27] for the binary classification task, and the fine-tuning of an LLM for classification, followed by a DETR-like model training for the interval detection task.
# 3.1. LLM-generated Text Detection
A detection model is needed to identify whether a text is written by a human or generated by a LLM, framing it as a binary classification task. The text could be produced by proprietary models, such as ChatGPT, or by open-source models like Llama[70], Mamba[21], and others. Formally, for a given text X, we aim to learn a function fθ : X → {0, 1}, where fθ(X) = 0 indicates the text is written by a human, and fθ(X) = 1 indicates it was generated by an LLM. We use a general-purpose LLM, namely, Mistral-
7B [31], which achieves high performance while maintaining efficiency and surpasses other models of similar size in benchmarks. A special [CLS] token is placed at the end of each text, and the corresponding final hidden state is extracted. This hidden state is passed through a classification head, consisting of two linear layers, to predict the label. To adapt the LLM for the task of generated text detection, we employ a Parameter-Efficient Fine-Tuning (PEFT) method called LoRA [27]. LoRA decomposes the weight matrix into two trainable low-rank matrices, while keeping the pre-trained weights frozen. This approach significantly reduces the number of trainable parameters while still enabling effective adaptation to downstream tasks.
# 3.2. LLM-generated Intervals Detection in HumanMachine Collaborative Texts
For the AI-generated intervals detection task, we propose to train a DETR model. The goal is to identify intervals within the text, represented as (c, w), where c is the center and w is the width, both in normalized units. After converting to absolute values, we obtain an interval in the form of (x1, x2), which define the start and end of the interval in characters. As illustrated in Figure 1, the model is composed of two key components: obtaining a text representation and performing detection.
Representation To obtain a vector representation that best describes a text for the given task, we apply the finetuning method described in 3.1 for the Mistral-7B model to classify a text generally into three categories: humanwritten, LLM-generated, and Human-Machine collaborative. After training the model, the text representation is derived as follows: first, the text is tokenized using the pretrained tokenizer, then we directly extract the latent vector representation from the final decoder layer of the Mistral model for each token. Overall, the representation stage can be formalized as:
(1)
where T denotes a sequence of length n consisting of individual tokens ti, and Tokenizer refers to the Byte-Pair Encoding (BPE) utilized by Mistral. E denotes a sequence consisting of individual embedding ei , and LLMft refers to the fine-tuned Mistral-7B model. Notably, if there is not enough data to fine-tune the LLM model for the downstream task, or in case of limited computational budget, it is possible to use embeddings from the pre-trained LLM without any fine-tuning. This may even yield better results in cases of limited data or imbalanced multiclass datasets, although generally, fine-tuned models
produce better text embeddings, leading to performance improvements. We examine how embeddings obtained from both pre-trained and fine-tuned Mistral models affect the resulting detection metrics in Appendix A.
Detection Extracted embeddings are used as inputs for the detection transformer model, which consists of a linear layer to project embeddings into a smaller subspace, a Transformer encoder, and a Transformer decoder. The mapped text features are fed into the Transformer encoder with positional encodings to obtain refined representation of the content. Following DAB-DETR [45], we use anchor-based learnable queries instead of the abstract vectors used in vanilla DETR. Each anchor is represented by a 2D reference point (c, w), serving as an initial hypothesis for where LLM-generated intervals might appear in the text. To pass these anchors in self-attention or cross-attention, sinusoidal encoding is applied. In the cross-attention block, the positional (learnable queries) and content (decoder selfattention output) embeddings are concatenated to separate their contributions to the query-to-feature similarity. The decoder refines anchors layer-by-layer, based on the encoded text features, adjusting each anchor’s center and width to more accurately localize LLM-generated sections. The output of each decoder layer contains a tuple (∆c, ∆w) for each anchor, which is updated to (c+∆c, w+∆w). With N learnable queries, the final result consists of up to N 1D intervals, each described by a center and width, indicating the likely positions of LLM-generated text segments. To stabilize early training, following DN-DETR [41], we pass both learnable anchor queries and noised GT boxes as inputs into the Transformer decoder. For the noised queries, we perform a denoising task to reconstruct their corresponding GT boxes, using an attention mask to prevent leakage to the learnable queries. We can formulate our method like:
(2)
where E is the set of mapped text features obtained in Equation 1. R is the set of refined text features obtained through the Transformer encoder. o = {o0, o1, ..., oN−1} is the Transformer decoder output, and q = {q0, q1, ..., qN−1} represents the decoder learnable anchor queries. Training objectives. We employ standard L1, gIoU [61] losses, and Focal Loss [44] for DETR training. Before calculating the loss to determine which GT interval corresponds to each predicted interval, we use Hungarian matching to compute an assignment between the targets and the
predictions. Denoised GT queries are not involved in the matching process. Our overall objective can be formulated as:
Lobj = λspan ∗Lspan + λgiou ∗Lgiou +λfocal ∗Lce + λdn span ∗Ldn span +λdn giou ∗Ldn giou,
(3)
where Lspan, Lgiou, Lfocal denotes L1, gIoU and Focal Loss for learnable queries and Ldn span, Ldn giou denotes L1 and gIoU learning objectives for noised queries. The λ terms are the weighting coefficients assigned to each respective loss component.
# 3.3. Implementation Details
Our LLM fine-tuning experiments for English texts are based on the Mistral-7B-v0.3 1 model. The model finetuning was done using Hugging Face Transformers2 with bfloat16 numeric precision. The LoRA configuration was set via the PEFT library3 as follows: the dimension of the low-rank matrices r was set to 8, the scaling factor for the weight matrices lora alpha was 16, the dropout probability of the LoRA layers lora dropout was 0.1, and the bias parameters were not trained for better performance (bias = ”none”). We only adapted the query and value projection matrices in the attention modules. To optimize the models, we used the AdamW optimizer [51] with a cosine learning rate scheduler [52]. The values of λspan, λgiou, λfocal, λdn span, and λdn giou in Equation 3 were set to 10.0, 1.0, 4.0, 9.0, and 3.0, respectively. The number of layers in the transformer encoder and transformer decoder of the DETR model was set to 3. The used computational resources, along with datasetspecific hyperparameters such as sequence length, batch size, training epochs, or number of queries, are detailed in Appendix B. When training a detection model to find LLM-generated intervals in text, we follow three steps: 1) we fine-tune the Mistral-7B model on two or three categories if the dataset contains enough data for this, 2) using the pretrained model, we extract features for the dataset, 3) we train the DETR model for interval detection using the extracted features as input data. The training is divided into three stages, firstly because this significantly speeds up the training process, and secondly because LLM and DETR models converge at different rates. While the LLM model may converge within one or two epochs, the DETR model requires significantly more.
1https://huggingface.co/mistralai/Mistral-7B-v0.3 2https://github.com/huggingface/transformers 3https://github.com/huggingface/peft
# 4. Datasets and metrics
We evaluate two aspects separately: the classification method for distinguishing machine-generated texts from human-written ones, and the approach for detecting LLMgenerated intervals in Human-Machine collaborative texts. In this section, we describe datasets and metrics used for both tasks. These datasets will be used in Section 5 for both training and testing to compare with other approaches trained on the same data.
# 4.1. Classification datasets
Metrics We evaluate GigaCheck as an LLM-generated content detector using classification accuracy (Acc). Following [43], we compute average recall (AvgRec) by averaging the recall scores for human-written (HumanRec) and machine-generated (MachineRec) texts. Also, we report the F1-score and the AUROC metric.
Datasets We evaluate the proposed approach for machine-written text classification using five datasets: TuringBench [74], TweepFake [16], Ghostbusters [76], MixSet [90], and MAGE [43]. The authors of the TuringBench dataset provide 19 subsets for the binary classification task. In our experiments, we selected only two of these, generated by the FAIR wmt20[10] and GPT-3[7] models. According to the authors, these models produce texts that are the most indistinguishable from human-written ones across five different detection models. The GPT-3 collection contains 17,018 texts, while the FAIR wmt20 collection includes 17,163 texts. The TweepFake [16] dataset consists of 25,572 social media messages posted either by bots or humans on Twitter. Each bot imitated a human account and was based on various generative techniques, including Markov Chains, RNN, RNN+Markov, LSTM, and GPT-2. The authors of the Ghostbusters [76] dataset used the GPT-3.5-turbo model to generate texts in the domains of creative writing, news, and student essays, providing 2,000 texts in the first two domains and 1,994 in the latter. We also use these subsets to evaluate the robustness of our method for out-of-domain detection. The authors of the MixSet [90] dataset explore the ability of various detection models to detect a mixture of machine-generated and human-written texts. They provide 3,600 texts, which are either human-written texts revised by LLMs or machine-generated texts edited by a human. Llama2-70b[70] and GPT-4[57] were used to generate LLM-revised texts by completing, rewriting, or polishing human texts. LLMs were also used to assist in humanrevised operations to enhance text humanization. Additionally, eight human experts were invited to adapt machine-
generated texts. For training, the authors also provide a set of 10,000 pre-processed samples of both purely humanwritten and machine-generated texts. This dataset is used to assess how effectively our model differentiates between texts co-written by humans and LLMs and those that are purely human-written. The authors of the MAGE [43] dataset provide a large set of generated texts using 27 LLMs from seven different groups: OpenAI GPT[7], LLaMA[70], GLM130B[85], FLAN-T5[12], OPT[91], BigScience[64, 5], and EleutherAI[77, 6]. In total, the dataset contains 432,682 texts, along with two additional sets. The first is an additional test set with texts from unseen domains generated by an unseen model, namely GPT-4[57]. The second set is designed to evaluate the robustness of detectors against paraphrasing attacks. To achieve this, the GPT-3.5-turbo model was employed to paraphrase the sentences from the first set, with all paraphrased texts treated as machine-generated. We utilize this dataset to assess the effectiveness of our method in different scenarios.
# 4.2. Detection datasets
Metrics To thoroughly analyze the performance of our system, we employed a set of detection metrics. We use metrics such as the Kappa score [86], sentence-wise MSE, Accuracy, and Soft Accuracy from [38], as well as a specialized form of the F1 score from [87], to assess the quality of the model’s predictions of the boundaries between sentences written by a human or an LLM. The authors of [87] consider LtopK, which represents the top-K boundaries identified by the algorithm, and LGt, which refers to the number of ground-truth boundaries. The F1 score is then determined using the following formula:
(4)
For their analysis, the authors set K = 3, as the maximum number of boundaries in the dataset is 3. Therefore, for the breakdown of results for the groups where Boundaries=1, 2, and 3, the ideal (upper-bound) F1@3 values are 0.5, 0.8, and 1.0, respectively. Further details on the calculation of each metric are provided in Appendix C.
Datasets We considered three different datasets for Human-Machine collaborative text analysis, which have been created to address the task of identifying a boundary between human-written and machine-generated text. The annotation for all datasets was converted into a specific format required for our model training, with each text annotated by a set of intervals represented as (x1, x2), where x1 indicates the character position where the generated in-
terval starts, and x2 indicates the character position where the generated interval ends. RoFT [15] dataset and its variation, RoFT-chatgpt [38], are both constructed in the same manner: each dataset sample consists of ten sentences, with the first part written by a human and the remainder completed by an LLM. Consequently, every sample has a boundary indicating the index of the sentence where authorship changes. These datasets differ in terms of generators used and the number of texts, while maintaining the same label distribution. Specifically, after removing duplicates, there are 8,943 samples generated by models from the GPT family (GPT-2, GPT-2 XL, CTRL) in RoFT dataset, and 6,940 samples samples generated by GPT-3.5 Turbo in RoFT-chatgpt. CoAuthor [39] is another dataset we used to evaluate our model. The dataset consists of 1,445 essays produced through the collaboration of human writers and GPT-3based writing assistant. Human writers were first given a prompt to start the essay, and throughout the writing process, they could accept or reject assistant’s suggestions at their own discretion. Notably, this writing process resulted in an indeterminate number of LLM-generated intervals in each text. Unlike other detection datasets, CoAuthor dataset does not make an assumption that these intervals correspond to complete sentences. TriBERT [87] dataset consists of 12,049 training, 2,527 validation and 2,560 test Human-Machine collaborative texts. Each text contains both human-written and LLMgenerated parts, which can appear in different orders (human →AI, AI →human). Therefore, each sample has between 1 and 3 boundaries, indicating the sentences where authorship changes. The texts were created using humanwritten essays with LLM-generated sections added using ChatGPT.
# 5. Experimental Results
In this section, we present GigaCheck’s results for five classifications and three detection datasets. We also demonstrate classification performance for out-of-domain and outof-model setups, and interval detection performance in outof-domain scenario. We evaluate the proposed approach for the text classification task using five datasets. For all datasets and their subsets, we fine-tune the Mistral-7B v0.3 model with LoRA. We compare our results with the baseline methods provided by the authors of these datasets. All our models were trained on the same training sets used by the authors. Tables 1 and 2 show results for the TweepFake dataset and TuringBench subsets, demonstrating high effectiveness on texts from various domains with machine-generated texts created by different generators. In comparison with the baseline methods, which include both the statistical method GLTR and fine-tuned LM models, our classification mod-
els yield better results. Based on the F1 metric, our model trained on the FAIR wmt20 subset surpasses the GLTR model by 50.59%, while the model trained on the GPT-3 subset outperforms the fine-tuned BERT model by 17.65%. Our approach also exceeds the fine-tuned RoBERTa model by 4.5% on the TweepFake dataset.
Method
HUMAN/MACHINE
ALL TEXTS
Recall
F1
Acc
BERT[14]
0.882/0.901
0.890/0.892
0.891
DistilBERT[63]
0.880/0.895
0.886/0.888
0.887
RoBERTa[46]
0.890/0.902
0.895/0.897
0.896
XLNet[83]
0.832/0.922
0.871/0.882
0.877
GigaCheck
(Mistral-7B)
0.951/0.934
0.944/0.942
0.943
GigaCheck with Mistral-7B also demonstrates an 11.4% improvement in the classification F1 score on the MixSet dataset compared to the fine-tuned Radar[28] model, achieving an F1 score of 99% across all types of mixed data, as shown in Table 3. Our method outperforms both metricbased approaches and model-based detectors trained by the authors of the MixSet dataset. The results show that our model effectively distinguishes LLM-generated texts from human-written ones and successfully differentiates complex texts co-written by humans and LLMs using various methods from purely human-written texts.
Table 5 presents the results on the Ghostbuster dataset across different settings. First, the model was trained and tested on all three domains together; second, on each domain individually. In the final out-of-domain setup, the model was trained on two domains and tested on one heldout domain. Results showed a minor drop in F1 score, from 100% to 97.4%, in only one out-of-domain case, while in all other cases, the model achieved 100% accuracy, outperforming the authors’ model and other baselines.
The results for the large-scale MAGE dataset, detailed in Table 4, include the performance of the GigaCheck classification model in different setups. First, we trained and validated the model on the full dataset, covering the entire range of data. Our model outperforms the baseline models (with an AUROC of 0.99 and an AvgRec of 96.11%). The resulting model was then evaluated in a more practical scenario, on a dataset with texts from unseen domains generated by an unseen model, and also tested for robustness against paraphrasing attacks, showing better than other methods results on both test sets (with an AvgRec of 88.54% on the first set and 68.95% on the second). Finally, in out-of-model setting, the classification model was trained on each of the seven subsets, where texts generated by a specific model set were excluded from training, and the weighted average performance was reported. Among all methods, the GigaCheck with Mistral-7B detector performed the best (with an AUROC of 0.98 and an AvgRec of 92.32%).
In summary, our approach effectively distinguishes LLM-generated texts from human-written ones when trained on both small and large datasets. The experiments demonstrate the robustness of our method for out-ofdomain and out-of-model detection, as well as its resistance to paraphrasing attacks.
Table 2: Experimental results (F1) on two TuringBench subsets. F1 is calculated for the machine-generated category.
Method
FAIR wmt20
GPT-3
GLTR[18]
0.4907
0.3476
BERT[14]
0.4701
0.7944
RoBERTa[46]
0.4531
0.5209
GigaCheck (Mistral-7B)
0.9966
0.9709
# 5.1. Classification Results 5.2. Detection Results
# 5.1. Classification Results
# 5.2. Detection Results
RoFT and RoFT-chatgpt results. In all experiments conducted on the RoFT and RoFTchatgpt datasets, we first fine-tuned Mistral-7B for the task of distinguishing between human-written texts and texts cowritten by humans and LLMs. The vector features obtained from the last layer of the trained models were then used to train the DN-DAB-DETR model with one learnable query. As a prediction, DN-DAB-DETR returns intervals in characters. To evaluate the method on this dataset, we converted the obtained intervals to a specific sentence number. Let tI be the start of the interval I, and starti, endi be the indexes of the first and last characters of the i-th sentence. If the i-th sentence contains tI, the sentence number i′, to which we map DN-DAB-DETR’s prediction, is calculated as follows:
(5)
Table 6 presents the main results for experiments on datasets. We also include the best result reported by [38], obtained from different types of detection models. The GigaCheck model achieves a significant accuracy boost over the RoBERTa-based classifier, outperforming it by 14.99% on the RoFT dataset and 13.04% on the RoFT-chatgpt dataset, while also demonstrating an MSE three times lower on the RoFT-chatgpt dataset. Table 7 demonstrates model performance for crossdomain transfer between four text domains in the RoFTchatgpt dataset. Each model was trained on three domains
Method
Average
LLM-Revised
Human-Revised
Complete
Rewrite
Polish-Tok.
Polish-Sen.
Humanize
Adapt-Sen.
Adapt-Tok.
Llama2
GPT-4
Llama2
GPT-4
Llama2
GPT-4
Llama2
GPT-4
Llama2
GPT-4
log-rank[54]
0.615
0.695
0.686
0.637
0.479
0.617
0.606
0.647
0.595
0.617
0.454
0.676
0.667
log likelihood[65]
0.624
0.695
0.695
0.637
0.492
0.657
0.627
0.657
0.657
0.637
0.386
0.676
0.667
GLTR[18]
0.588
0.686
0.647
0.606
0.441
0.574
0.585
0.637
0.540
0.617
0.400
0.657
0.667
DetectGPT[54]
0.635
0.715
0.651
0.656
0.560
0.632
0.587
0.657
0.632
0.692
0.587
0.641
0.609
Entropy[18]
0.648
0.690
0.671
0.681
0.613
0.681
0.671
0.681
0.671
0.623
0.430
0.681
0.681
Radar[28]
0.876
0.867
0.877
0.877
0.877
0.877
0.877
0.877
0.877
0.877
0.877
0.877
0.877
GPT-sentinel[11]
0.713
0.714
0.714
0.714
0.714
0.714
0.714
0.714
0.714
0.696
0.714
0.714
0.714
DistilBERT[63]
0.664
0.667
0.667
0.667
0.667
0.667
0.667
0.667
0.667
0.639
0.667
0.667
0.667
GigaCheck (Mistral-7B)
0.99
0.99
0.99
0.99
0.99
0.99
0.99
0.99
0.99
0.99
0.99
0.99
0.99
Table 4: Classification performance on MAGE dataset in different scenarios including performance on the two test sets. To test on challenging test sets the model trained on Arbitrary-domains & Arbitrary-models dataset w model trained on texts from all domains with generations from all model sets.
Methods
HumanRec
MachineRec
AvgRec
AUROC
Arbitrary-domains & Arbitrary-models
FastText[32]
86.34%
71.26%
78.80%
0.83
GLTR[18]
12.42%
98.42%
55.42%
0.74
DetectGPT[54]
86.92%
34.05%
60.48%
0.57
Longformer[3]
82.80%
98.27%
90.53%
0.99
GigaCheck (Mistral-7B)
95.72%
96.49%
96.11%
0.99
Unseen Domains & Unseen Model
FastText[32]
71.78%
68.88%
70.33%
0.74
GLTR[18]
16.79%
98.63%
57.71%
0.73
Longformer[3]
52.50%
99.14%
75.82%
0.94
GigaCheck (Mistral-7B)
79.71%
97.38%
88.54%
0.96
Paraphrasing Attack
FastText[32]
71.78%
50.00%
60.89%
0.66
GLTR[18]
16.79%
82.44%
49.61%
0.47
Longformer[3]
52.16%
81.73%
66.94%
0.75
GigaCheck (Mistral-7B)
79.66%
58.24%
68.95%
0.74
Out-of-distribution Detection: Unseen models
FastText[32]
83.12%
54.09%
68.61%
0.74
GLTR[18]
25.77%
89.21%
57.49%
0.65
DetectGPT[54]
48.67%
75.95%
62.31%
0.60
Longformer[3]
83.31%
89.90%
86.61%
0.95
GigaCheck (Mistral-7B)
95.65%
89.00%
92.32%
0.98
FastText[32]
83.12%
54.09%
68.61%
0.74
GLTR[18]
25.77%
89.21%
57.49%
0.65
DetectGPT[54]
48.67%
75.95%
62.31%
0.60
Longformer[3]
83.31%
89.90%
86.61%
0.95
GigaCheck (Mistral-7B)
95.65%
89.00%
92.32%
0.98
and tested on the fourth domain, which was excluded from the training set. Compared to the approaches described by the authors in [38], our model demonstrates the best cross-
domain generalization across all domains. However, the accuracy for the Recipes domain remains relatively low. Several examples of the raw model’s output, trained on
domain generalization across all domains. However, the accuracy for the Recipes domain remains relatively low. Several examples of the raw model’s output, trained on
Table 5: Results across a variety of text domains (F1) on Ghostbusters test sets. F1 is calculated for the machine-generated category. We first trained and evaluated model on each of three domains individually (news, creative writing, or studen essays); in the “All Domains” condition, models were trained and evaluated on all three domains at once. For each out-of domain setting, model was trained on two domains and evaluated on one held-out domain.
In-Domain
Out-of-Domain
Method
All
Domains
News
Creative
Writing
Student
Essays
News
Creative
Writing
Student
Essays
Perplexity only[76]
81.5
82.2
84.1
92.1
71.9
49.0
93.4
DetectGPT[54]
57.4
56.6
48.2
67.3
56.6
48.2
67.3
RoBERTa[46]
98.1
99.4
97.6
97.4
88.3
95.7
71.4
Ghostbuster[76]
99.0
99.5
98.4
99.5
97.9
95.3
97.7
GigaCheck (Mistral-7B)
100
100
100
100
100
100
97.4
<div style="text-align: center;">Table 6: Boundary detection results on RoFT and RoFT-chatgpt datasets. The results for all methods, except ours, were tak from [38].</div>
Method
RoFT
RoFT-chatgpt
Acc
SoftAcc1
MSE
Acc
SoftAcc1
MSE
RoBERTa + SEP[13]
49.64 %
79.71 %
2.63
54.61 %
79.03 %
3.06
RoBERTa[46]
46.47 %
74.86 %
3.00
39.01 %
75.18 %
3.15
GigaCheck (DN-DAB-DETR)
64.63 %
86.68 %
1.51
67.65 %
88.98 %
1.03
Based on Perplexity
DetectGPT[54] + GB classifier
19.79 %
37.40 %
8.35
21.69 %
43.52 %
6.87
DetectGPT[54] + LR classifier
19.45 %
33.82 %
9.03
15.35 %
41.43 %
7.22
Phi-1.5[42] Perpl. + GB regressor
17.1 %
44.6 %
6.11
32.0 %
71.0 %
3.07
Phi-1.5[42] Perpl. + LR classifier
27.0 %
49.5 %
11.9
47.3 %
72.7 %
4.77
Based on TDA
PHD + TS ML[38]
23.50 %
46.32 %
14.14
17.29 %
35.81 %
14.45
TLE + TS Binary[38]
12.58 %
30.41 %
22.23
20.02 %
34.58 %
18.52
Human baseline[13]
22.62 %
40.31 %
13.88
—
the RoFT-chatgpt dataset, are presented in Appendix D. CoAuthor results. We conducted experiments in two setups: (i) fine-tuning the Mistral-7B to classify text segments into three categories: LLM-generated, humanwritten, or Human-Machine collaborative, and (ii) training the DN-DAB-DETR model to detect LLM-generated intervals within the text, then converting them into segment labels. For classifier-based experiments, we used sentence splitting as the segmentation method (referred to as the ”Naive” method in the original work), as it is one of the ways to obtain segments without relying on an external model. In total, we compare three types of methods: the first takes individual sentences as input and analyzes them separately, the second takes a sequence of pre-computed embeddings for all sentences in the text as input and makes predictions for each of these sentences, and the third, DN-
DAB-DETR method, processes the text as a whole, making predictions for the entire text. Since the calculation of the target metric involves determining the label for each sentence, the resulting intervals from our DN-DAB-DETR model underwent post-processing. Let si represent the ith sentence, and let p denote the predicted LLM-generated intervals. The overlap oi is calculated as:
(6)
The label L(si) for each sentence is assigned based on oi as follows:
(7)
Pres.
Recipes New York Short
Pred. Model
Context Speeches
Times
Stories
Text
GigaCheck (DN-DAB-DETR) global
50.0
32.8
54.5
63.6
Text
RoBERTa SEP[13]
global
31.4
13.1
38.1
28.6
Text
RoBERTa[46]
global
36.3
14.8
38.0
36.1
Perpl. Phi1.5[42], GB
sent.
52.0
24.1
45.7
56.1
Perpl. Phi1.5[42], LR
sent.
41.2
21.1
45.2
51.5
PHD TS multi[38]
100 tkn 13.
19.5
17.2
17.6
TLE
TS Binary[38]
20 tkn
14.7
16.3
17.1
11.1
<div style="text-align: center;">Pred. Model</div>
Table 8: Comparison of methods with respect to the Kappa Score on the dataset CoAuthor[39], using different types of classification methods. The results for all methods, except ours, were taken from [86].
Method
Context
Pred.
Kappa Score
GigaCheck
(Mistral-7B-v0.3)
sent
sent
0.4158
DeBERTa-v3[25]
sent
sent
0.4002
BERT[14]
sent
sent
0.3748
SeqXGPT[78]
sent
sent
0.3522
RoBERTa[46]
sent
sent
0.3607
DistilBERT[63]
sent
sent
0.3612
GPT-3.5 (Fine-tuned)[56]
sent
sent
0.2707
GPT2[59]
sent
sent
0.2071
SegFormer[2]
global
sent
0.3180
Transformer2[50]
global
sent
0.2519
GigaCheck
(DN-DAB-DETR)
global
text
0.1885
Here, |si ∩p| is the length of the intersection (i.e., the overlap) between the sentence si and the predicted intervals p, and |si| is the total length of sentence si. Table 8 presents the results for all described approaches. As shown, our method surpasses baseline approaches with a Kappa score of 0.4158. Results on full texts, however, are lower than on sentences, likely due to the very short LLM-generated intervals in the dataset, which complicate accurate detection. Identifying exact intervals is inherently more challenging than classifying pre-defined segments. TriBERT results. We use the pre-trained Mistral-7Bv0.3 to obtain text representations, as the dataset does not contain enough data for fine-tuning the model on several categories. A trained DN-DAB-DETR model is used to detect LLM-generated intervals based on text features from the pre-trained model. The F1@3 (Equation 4) metric, as proposed by the authors of [87], serves as the performance
measure. Since this metric assumes sentence-level measurements, the interval predictions made by the DN-DABDETR model have been post-processed. Let bi and bi+1 denote the beginnings of the n and n + 1 sentences in characters and let pj denote the beginning or the end of the predicted interval in characters. Then the boundary B for pj is calculated as:
 ≥ Therefore, if the predicted start or end of the interval falls in the first half of sentence n, we map it to the beginning of sentence n. If it falls in the second half, we map it to the beginning of the next sentence, n + 1. As a result, each boundary determines the sentence number where the text’s authorship changes. Note that if a boundary is equal to the beginning or the end of the whole text, we remove it, since a boundary can only be between two sentences. Measurements were made for groups of texts with different numbers of boundaries, as well as for the overall dataset (All). The results are presented in two forms: original and rescaled. As previously described, the ideal F1@3 values are 0.5, 0.8, and 1.0 for groups with 1, 2, and 3 boundaries, respectively. Since the original metrics are difficult to interpret due to the different value ranges, we rescaled them to a common scale, where the ideal F1@3 is 1.0 for all boundary groups. As shown in Table 9, our model outperforms the best TriBERT model [87] by 7.1% according to the F1@3 metric on the All data. The best results were obtained across all measurements, except for the subset with only 1 boundary. Nonetheless, our method is more stable across different boundary counts, whereas TriBERT’s metrics decline significantly as the number of boundaries increases.
# 6. Conclusions
In this paper, we propose approaches for analyzing LLM-generated texts through two tasks: (i) binary classi-
Table 9: Boundary detection results (F1@3) on the TriBERT dataset. #Bry denotes the number of ground-truth boundaries in the texts. Measurements are presented in two formats: original and rescaled.
Methods
#Bry=1
#Bry=2
#Bry=3
All
Original values
TriBERT (p=2)[87]
0.455
0.692
0.622
0.575
GigaCheck
(DN-DAB-DETR)
0.444
0.693
0.801
0.646
Rescaled values
TriBERT (p=2)[87]
0.910
0.865
0.622
-
GigaCheck
(DN-DAB-DETR)
0.888
0.867
0.801
-
fication to distinguish human from machine-generated content, and (ii) detection of generated sections within collaboratively written Human-Machine texts. We demonstrate the classification model ability to crossdomain and cross-model generalization, and also show the good detection performance of the method on the two challenging test sets, one with the applied paraphrased attack and the second with the texts from unseen domains and generated by unseen model. Our method for detecting LLM-generated sections in texts is flexible, as it is not tied to specific sentences or sections of text. We demonstrated its effectiveness in different setups (using either a pre-trained or fine-tuned LLM model for text feature extraction) and conducted out-of-domain experiments. We evaluated our method on five classification datasets and three detection datasets, achieving state-of-the-art results on all of them.
# 7. Ethics and Limitations
Limitations Both the classification and detection models have a limited context window. As a result, for the analysis of long texts, it may be necessary to split the text into smaller parts to ensure effective processing. This fragmentation could impact the model’s performance and the overall coherence of the analysis. We did not assess how our models perform on multilingual datasets. Therefore, it is possible that our approach may not work as optimally for other languages. However, since the utilized Mistral-7B model is multilingual, the methods we describe can still be applied for analyzing texts in different languages. In our experiments, we used the Mistral-7B model as a representative of high-performance models. We did not evaluate how other LLMs might perform in analyzing gen-
Experiment
CO2 Emission (kg)
MAGE
249.07
MixSet
15.57
Ghostbuster
5.19
TurningBench
10.38
TweepFake
10.38
RoFT
41.52
RoFT-chatgpt
25.95
CoAuthor
36.33
TriBERT
31.14
erated content. Future work could benefit from exploring a wider range of models to determine the generalizability of our approach across various LLMs.
Possible Misuse Our approach does not guarantee perfect results in every case, as its performance can vary significantly depending on factors such as the specific use case, generator model, text length, writing style, and domain. Therefore, no critical decisions or conclusions should be based solely on the output of this system. We disclaim responsibility for any reputational damage that may result from inaccurate conclusions, as we have clearly stated that the model is currently imperfect and cannot be applied as a final arbiter.
# Energy Efficiency and Usage We compute the CO2 emissions for our experiments via Equation 9 [67]:
Energy Efficiency and Usage We compute the CO2 emissions for our experiments via Equation 9 [67]:
CO = PUE × kWh × ICO2
CO2 = PUE × kWh × ICO2 1000
(9)
The power usage effectiveness of our server is 1.3. We use single H100 GPU for all our experiments, which has a peak power usage of ≈700 watts. The number of hours used for each experiment can be found in the Appendix B. We briefly present the CO2 emissions for each dataset experiment in Table 10.
# References
[1] Wissam Antoun, Virginie Mouilleron, Benoˆıt Sagot, and Djam´e Seddah. Towards a robust detection of language model generated text: Is chatgpt that easy to detect? arXiv preprint arXiv:2306.05871, 2023. 2 [2] Haitao Bai, Pinghui Wang, Ruofei Zhang, and Zhou Su. Segformer: A topic segmentation model with controllable range of attention. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 12545–12552, 2023. 10 [3] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020. 8
[4] Amrita Bhattacharjee and Huan Liu. Fighting fire with fire: can chatgpt detect ai-generated text? ACM SIGKDD Explorations Newsletter, 25(2):14–21, 2024. 2 [5] BigScience. Bloom: A 176b-parameter open-access multilingual language model, 2023. 6 [6] Sidney Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, Usvsn Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. GPTNeoX-20B: An open-source autoregressive language model. In Proceedings of BigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models, pages 95–136, virtual+Dublin, 2022. Association for Computational Linguistics. 6 [7] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and HsuanTien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. 5, 6 [8] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 2, 3 [9] Souradip Chakraborty, Amrit Singh Bedi, Sicheng Zhu, Bang An, Dinesh Manocha, and Furong Huang. On the possibilities of ai-generated text detection. arXiv preprint arXiv:2304.04736, 2023. 2 10] Peng-Jen Chen, Ann Lee, Changhan Wang, Naman Goyal, Angela Fan, Mary Williamson, and Jiatao Gu. Facebook ai’s wmt20 news translation task submission. arXiv preprint arXiv:2011.08298, 2020. 5 11] Yutian Chen, Hao Kang, Vivian Zhai, Liangze Li, Rita Singh, and Bhiksha Raj. Gpt-sentinel: Distinguishing human and chatgpt generated content. arXiv preprint arXiv:2305.07969, 2023. 8 12] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models, 2022. 6
[13] Joseph Cutler, Liam Dugan, Shreya Havaldar, and Adam Stein. Automatic detection of hybrid human-machine text boundaries. 2021. 9, 10 [14] Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 1, 7, 10 [15] Liam Dugan, Daphne Ippolito, Arun Kirubarajan, Sherry Shi, and Chris Callison-Burch. Real or fake text?: Investigating human ability to detect boundaries between humanwritten and machine-generated text. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 12763–12771, 2023. 6 [16] Tiziano Fagni, Fabrizio Falchi, Margherita Gambini, Antonio Martella, and Maurizio Tesconi. Tweepfake: About detecting deepfake tweets. Plos one, 16(5):e0251415, 2021. 5 [17] Leon Fr¨ohling and Arkaitz Zubiaga. Feature-based detection of automated language models: tackling gpt-2, gpt-3 and grover. PeerJ Computer Science, 7:e443, 2021. 2 [18] Sebastian Gehrmann, Hendrik Strobelt, and Alexander M Rush. Gltr: Statistical detection and visualization of generated text. arXiv preprint arXiv:1906.04043, 2019. 2, 7, 8 [19] Aleksandr Gordeev, Vladimir Dokholyan, Irina Tolstykh, and Maksim Kuprashevich. Saliency-guided detr for moment retrieval and highlight detection, 2024. 3 [20] Dijana Vukovic Grbic and Igor Dujlovic. Social engineering with chatgpt. In 2023 22nd International Symposium INFOTEH-JAHORINA (INFOTEH), pages 1–5. IEEE, 2023. 1 [21] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. 3 [22] Biyang Guo, Xin Zhang, Ziyuan Wang, Minqi Jiang, Jinran Nie, Yuxuan Ding, Jianwei Yue, and Yupeng Wu. How close is chatgpt to human experts? comparison corpus, evaluation, and detection. arXiv preprint arXiv:2301.07597, 2023. 2 [23] Zhen Guo and Shangdi Yu. Authentigpt: Detecting machinegenerated text via black-box language models denoising. arXiv preprint arXiv:2311.07700, 2023. 2 [24] Hans WA Hanley and Zakir Durumeric. Machine-made media: Monitoring the mobilization of machine-generated articles on misinformation and mainstream news websites. In Proceedings of the International AAAI Conference on Web and Social Media, volume 18, pages 542–556, 2024. 1 [25] Pengcheng He, Jianfeng Gao, and Weizhu Chen. Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing. arXiv preprint arXiv:2111.09543, 2021. 10 [26] Xiuquan Hou, Meiqin Liu, Senlin Zhang, Ping Wei, Badong Chen, and Xuguang Lan. Relation detr: Exploring explicit position relation prior for object detection. arXiv preprint arXiv:2407.11699, 2024. 3 [27] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3, 4
[28] Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. Radar: Robust ai-text detection via adversarial learning. Advances in Neural Information Processing Systems, 36:15077–15095, 2023. 7, 8 [29] Kuan-Chih Huang, Tsung-Han Wu, Hung-Ting Su, and Winston H Hsu. Monodtr: Monocular 3d object detection with depth-aware transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4012–4021, 2022. 3 [30] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023. 1 [31] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. 1, 4 [32] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification. arXiv preprint arXiv:1607.01759, 2016. 8 [33] Enkelejda Kasneci, Kathrin Seßler, Stefan K¨uchemann, Maria Bannert, Daryna Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan G¨unnemann, Eyke H¨ullermeier, et al. Chatgpt for good? on opportunities and challenges of large language models for education. Learning and individual differences, 103:102274, 2023. 1 [34] Ryuto Koike, Masahiro Kaneko, and Naoaki Okazaki. Outfox: Llm-generated essay detection through in-context learning with adversarially generated examples. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 21258–21266, 2024. 2 [35] Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, and Mohit Iyyer. Paraphrasing evades detectors of ai-generated text, but retrieval is an effective defense. Advances in Neural Information Processing Systems, 36, 2024. 2 [36] Sachin Kumar, Vidhisha Balachandran, Lucille Njoo, Antonios Anastasopoulos, and Yulia Tsvetkov. Language generation models can cause harm: So what can we do about it? an actionable survey. arXiv preprint arXiv:2210.07700, 2022. 1 [37] Laida Kushnareva, Daniil Cherniavskii, Vladislav Mikhailov, Ekaterina Artemova, Serguei Barannikov, Alexander Bernstein, Irina Piontkovskaya, Dmitri Piontkovski, and Evgeny Burnaev. Artificial text detection via examining the topology of attention maps. arXiv preprint arXiv:2109.04825, 2021. 2 [38] Laida Kushnareva, Tatiana Gaintseva, Dmitry Abulkhanov, Kristian Kuznetsov, German Magai, Eduard Tulchinskii, Serguei Barannikov, Sergey Nikolenko, and Irina Piontkovskaya. Ai-generated text boundary detection with roft. In First Conference on Language Modeling. 2, 6, 7, 8, 9, 10, 16 [39] Mina Lee, Percy Liang, and Qian Yang. Coauthor: Designing a human-ai collaborative writing dataset for exploring language model capabilities. In Proceedings of the 2022 CHI
conference on human factors in computing systems, pages 1– 19, 2022. 6, 10 [40] Jie Lei, Tamara L Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. Advances in Neural Information Processing Systems, 34:11846–11858, 2021. 3 [41] Feng Li, Hao Zhang, Shilong Liu, Jian Guo, Lionel M Ni, and Lei Zhang. Dn-detr: Accelerate detr training by introducing query denoising. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13619–13627, 2022. 3, 4 [42] Yuanzhi Li, S´ebastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023. 9, 10 [43] Yafu Li, Qintong Li, Leyang Cui, Wei Bi, Zhilin Wang, Longyue Wang, Linyi Yang, Shuming Shi, and Yue Zhang. Mage: Machine-generated text detection in the wild. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 36–53, 2024. 2, 5, 6 [44] T Lin. Focal loss for dense object detection. arXiv preprint arXiv:1708.02002, 2017. 4 [45] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. Dab-detr: Dynamic anchor boxes are better queries for detr. arXiv preprint arXiv:2201.12329, 2022. 3, 4 [46] Yinhan Liu. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019. 1, 2, 7, 9, 10 [47] Yikang Liu, Ziyin Zhang, Wanyang Zhang, Shisen Yue, Xiaojing Zhao, Xinyuan Cheng, Yiwen Zhang, and Hai Hu. Argugpt: evaluating, understanding and identifying argumentative essays generated by gpt models. arXiv preprint arXiv:2304.07666, 2023. 2 [48] Zeyan Liu, Zijun Yao, Fengjun Li, and Bo Luo. Check me if you can: Detecting chatgpt-generated academic writing using checkgpt. arXiv preprint arXiv:2306.05524, 2023. 2 [49] Zeyan Liu, Zijun Yao, Fengjun Li, and Bo Luo. On the detectability of chatgpt content: benchmarking, methodology, and evaluation through the lens of academic writing. arXiv e-prints, pages arXiv–2306, 2023. 1, 2 [50] Kelvin Lo, Yuan Jin, Weicong Tan, Ming Liu, Lan Du, and Wray Buntine. Transformer over pre-trained transformer for neural text segmentation with enhanced topic coherence. arXiv preprint arXiv:2110.07160, 2021. 10 [51] I Loshchilov. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5 [52] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016. 5 [53] Yisroel Mirsky, Ambra Demontis, Jaidip Kotak, Ram Shankar, Deng Gelei, Liu Yang, Xiangyu Zhang, Maura Pintor, Wenke Lee, Yuval Elovici, et al. The threat of offensive ai to organizations. Computers & Security, 124:103006, 2023. 1
[54] Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D Manning, and Chelsea Finn. Detectgpt: Zero-shot machine-generated text detection using probability curvature. In International Conference on Machine Learning, pages 24950–24962. PMLR, 2023. 2, 8, 9 [55] WonJun Moon, Sangeek Hyun, SuBeen Lee, and Jae-Pil Heo. Correlation-guided query-dependency calibration in video representation learning for temporal grounding. arXiv preprint arXiv:2311.08835, 2023. 3 [56] OpenAI. ChatGPT: A Large Language Model. Online; accessed February 13, 2024, 2023. Available at https://www.openai.com/. 2, 10 [57] OpenAI. Gpt-4 technical report, 2023. 1, 2, 5, 6 [58] Mike Perkins, Jasper Roe, Darius Postma, James McGaughran, and Don Hickerson. Game of tones: faculty detection of gpt-4 generated content in university assessments. arXiv preprint arXiv:2305.18081, 2023. 1 [59] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. 10 [60] Llama Rewrite. Learning to rewrite: Generalized llmgenerated text detection. 2 [61] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 658–666, 2019. 4 [62] Sayak Saha Roy, Krishna Vamsi Naragam, and Shirin Nilizadeh. Generating phishing attacks using chatgpt. arXiv preprint arXiv:2305.05133, 2023. 1 [63] V Sanh. Distilbert, a distilled version of bert: Smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019. 7, 8, 10 [64] Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault F´evry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M. Rush. Multitask prompted training enables zero-shot task generalization. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. 6 [65] Irene Solaiman, Miles Brundage, Jack Clark, Amanda Askell, Ariel Herbert-Voss, Jeff Wu, Alec Radford, Gretchen Krueger, Jong Wook Kim, Sarah Kreps, et al. Release strategies and the social impacts of language models. arXiv preprint arXiv:1908.09203, 2019. 2, 8 [66] Chris Stokel-Walker. Ai bot chatgpt writes smart essaysshould academics worry? Nature, 2022. 1 [67] Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in NLP.
In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3645–3650, Florence, Italy, July 2019. Association for Computational Linguistics. 11 [68] Jinyan Su, Terry Yue Zhuo, Di Wang, and Preslav Nakov. Detectllm: Leveraging log rank information for zeroshot detection of machine-generated text. arXiv preprint arXiv:2306.05540, 2023. 2 [69] H Holden Thorp. Chatgpt is fun, but not an author, 2023. 1 [70] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. 3, 5, 6 [71] Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva, Daniil Cherniavskii, Sergey Nikolenko, Evgeny Burnaev, Serguei Barannikov, and Irina Piontkovskaya. Intrinsic dimension estimation for robust detection of ai-generated texts. Advances in Neural Information Processing Systems, 36, 2024. 2 [72] Adaku Uchendu, Thai Le, and Dongwon Lee. Toproberta: Topology-aware authorship attribution of deepfake texts. arXiv preprint arXiv:2309.12934, 2023. 2 [73] Adaku Uchendu, Thai Le, Kai Shu, and Dongwon Lee. Authorship attribution for neural text generation. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 8384–8395, 2020. 2 [74] Adaku Uchendu, Zeyu Ma, Thai Le, Rui Zhang, and Dongwon Lee. Turingbench: A benchmark environment for turing test in the age of neural text generation. arXiv preprint arXiv:2109.13296, 2021. 2, 5 [75] Christoforos Vasilatos, Manaar Alam, Talal Rahwan, Yasir Zaki, and Michail Maniatakos. Howkgpt: Investigating the detection of chatgpt-generated university student homework through context-aware perplexity analysis. arXiv preprint arXiv:2305.18226, 2023. 1 [76] Vivek Verma, Eve Fleisig, Nicholas Tomlin, and Dan Klein. Ghostbuster: Detecting text ghostwritten by large language models. arXiv preprint arXiv:2305.15047, 2023. 5, 9 [77] Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/ mesh-transformer-jax, 2021. 6 [78] Pengyu Wang, Linyang Li, Ke Ren, Botian Jiang, Dong Zhang, and Xipeng Qiu. Seqxgpt: Sentence-level aigenerated text detection. arXiv preprint arXiv:2310.08903, 2023. 1, 2, 3, 10 [79] Rongsheng Wang, Qi Li, and Sihong Xie. Detectgpt-sc: Improving detection of text generated by large language models through self-consistency with masked predictions. arXiv preprint arXiv:2310.14479, 2023. 2 [80] Yuxia Wang, Jonibek Mansurov, Petar Ivanov, Jinyan Su, Artem Shelmanov, Akim Tsvigun, Osama Mohanned Afzal, Tarek Mahmoud, Giovanni Puccetti, Thomas Arnold, et al. M4gt-bench: Evaluation benchmark for black-box machinegenerated text detection. arXiv preprint arXiv:2402.11175, 2024. 2, 16
[81] Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F Wong, and Lidia S Chao. A survey on llm-gernerated text detection: Necessity, methods, and future directions. arXiv preprint arXiv:2310.14724, 2023. 1 [82] Kangxi Wu, Liang Pang, Huawei Shen, Xueqi Cheng, and Tat-Seng Chua. Llmdet: A third party large language models generated text detection tool. arXiv preprint arXiv:2305.15004, 2023. 2 [83] Zhilin Yang. Xlnet: Generalized autoregressive pretraining for language understanding. arXiv preprint arXiv:1906.08237, 2019. 7 [84] Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, and Yejin Choi. Defending against neural fake news. Advances in neural information processing systems, 32, 2019. 2 [85] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Peng Zhang, Yuxiao Dong, and Jie Tang. Glm-130b: An open bilingual pre-trained model, 2022. 6 [86] Zijie Zeng, Shiqi Liu, Lele Sha, Zhuang Li, Kaixun Yang, Sannyuya Liu, Dragan Gaˇsevic, and Guanliang Chen. Detecting ai-generated sentences in human-ai collaborative hybrid texts: Challenges, strategies, and insights. 1, 3, 6, 10, 16 [87] Zijie Zeng, Lele Sha, Yuheng Li, Kaixun Yang, Dragan Gaˇsevi´c, and Guangliang Chen. Towards automatic boundary detection for human-ai collaborative hybrid essay in education. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 22502–22510, 2024. 1, 2, 6, 10, 11, 16 [88] Chuhan Zhang, Ankush Gupta, and Andrew Zisserman. Temporal query networks for fine-grained video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4486–4496, 2021. 3 [89] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022. 3 [90] Qihui Zhang, Chujie Gao, Dongping Chen, Yue Huang, Yixin Huang, Zhenyang Sun, Shilin Zhang, Weiye Li, Zhengyan Fu, Yao Wan, et al. Llm-as-a-coauthor: Can mixed human-written and machine-generated text be detected? In Findings of the Association for Computational Linguistics: NAACL 2024, pages 409–436, 2024. 1, 2, 5 [91] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer language models, 2022. 6 [92] X Zhu, W Su, L Lu, B Li, X Wang, and J Dai. Deformable detr: Deformable transformers for end-to-end object detection. arxiv 2020. arXiv preprint arXiv:2010.04159, 2010. 3 [93] Zhuofan Zong, Guanglu Song, and Yu Liu. Detrs with collaborative hybrid assignments training. In Proceedings of
the IEEE/CVF international conference on computer vision, pages 6748–6758, 2023. 3
# A. Pre-trained VS fine-tuned models’ embeddings
Table 11 presents a comparison of detection model performance on the RoFT and RoFT-ChatGPT datasets using two different setups. In the first experiment, we fine-tuned the Mistral-7B model to perform a text classification task with two labels: ’Human’ and ’AI-Human Collaborative’, and used this model to extract text features for DETR model training. In the second experiment, we utilized the pretrained Mistral-7B v0.3 model for feature extraction. Two DN-DAB-DETR models were then trained using these two types of features. The results indicate that the detection model performs better with features from the fine-tuned model; however, the model trained with text representations from the pre-trained model also achieves strong results on both datasets. We also provide results from [38] for comparison.
# B. Hyperparameters
We fine-tune Mistral-7B for a binary classification task to distinguish between human-written and machinegenerated content using LoRA. During training, we augmented the data by randomly selecting between ’minimum sequence length’ to ’maximum sequence length’ tokens from each text. To optimize the models, we used the AdamW optimizer with a cosine learning rate schedule and also applied a weight for the ’Human’ category in the cross-entropy function. The dataset-specific hyperparameters used for the experiments in Section 5.1 are listed in the table 12. To train DN-DAB-DETR models, we also used the AdamW optimizer with a cosine learning rate schedule. During training we did not apply any text augmentations. The dataset-specific hyperparameters used for the experiments in Section 5.2 are listed in the table 13.
# C. Evaluation metrics for detection datasets
For each detection dataset, we compute specific metrics. Followed the approach of the authors in [38], we compute mean squared error MSE= 1 N �N i=1 (yi −ˆyi)2 between the predicted boundaries ˆy and the true boundaries y, where a boundary is the sentence number at which authorship in the text changes from human to LLM, and N represents the number of samples. It is worth noting that in both datasets from [38], each text contains no more than one boundary. The authors also propose reporting accuracy (Acc) of boundary detection and soft accuracy (SoftAcc1), the proportion of predictions that are off from the correct label by no more than one. In [86], the authors suggest assessing interval prediction quality by computing Kappa score, a correlation statistic.
In this context, it is used to measure the extent to which the predictive model and ground-truth labels assign the same authorship to text segments, pre-obtained by a segmentation model or other means (inter-rater agreement). The Kappa score is defined as:
(10)
 − where p0 is the relative observed agreement and pe is the probability of random agreement. The maximum value of the score is 1 (indicating complete agreement), and the minimum value is -1. Finally, the authors of [80] evaluate model prediction quality using the mean absolute error MAE= 1 N �N i=1 |yi −ˆyi|, where ˆy denotes the predicted word number that separates human and AI-generated parts of the text, y represents ground-truth word number, and N is the number of samples. The problem statement in [80] implies that there is only one such word boundary per text. F1@K metric proposed by [87] to asses the performance of model in boundaries detection task is described in Eq. . K was set to 3 for all measurements on TriBERT dataset.
# D. Examples of the DETR model output
Tables 14 and 15 present examples of work of the model trained on the RoFT-chatgpt dataset. Table 14 shows the ground truth and output result for test samples from the ’Short Stories’ and ’New York Times’ domains. Table 15 shows the ground truth and output result for test samples from the ’Recipes’ and ’Presidential Speeches’ domains.
<div style="text-align: center;">able 11: Boundary detection results on ROFT and ROFT-chatgpt datasets. ‘†’ denotes the DETR model was trained on text eatures from pre-trained Mistral-7B v0.3 model. Bold shows the best method, underlined - second best.</div>
Method
RoFT
RoFT-chatgpt
Acc
SoftAcc1
MSE
Acc
SoftAcc1
MSE
RoBERTa + SEP
49.64 %
79.71 %
2.63
54.61 %
79.03 %
3.06
RoBERTa
46.47 %
74.86 %
3.00
39.01 %
75.18 %
3.15
GigaCheck (DN-DAB-DETR)†
60.10 %
81.48 %
2.77
51.37 %
80.12 %
1.93
GigaCheck (DN-DAB-DETR)
64.63 %
86.68 %
1.51
67.65 %
88.98 %
1.03
<div style="text-align: center;">Table 12: The dataset-specific hyperparameters used for the experiments from Section 5.1.</div>
Parameter
MAGE
MixSet
Ghostbuster
TurningBench
TweepFake
max sequence length
1024
512
1024
1024
1024
minimum sequence
length for augmentatoins
900
250
256
15
900
train batch size
64
32
32
32
32
gradient accumulation
steps
1
2
2
2
2
learning rate
3e-4
3e-4
3e-5
3e-4
3e-4
cross entropy weight for
human category
2
3
3
1
1
num train epochs
3
15
15
5
4
GPUs
1xNvidia
H100
1xNvidia
H100
1xNvidia
H100
1xNvidia
H100
1xNvidia
H100
the fine-tuning time
48h
3h
1h
2h
2h
<div style="text-align: center;">Table 13: The dataset-specific hyperparameters used for the experiments from Section 5.2.</div>
Parameter
RoFT
RoFT-chatgpt
CoAuthor
TriBERT
number of queries
1
1
30
18
max sequence length
512
512
1024
1024
train batch size
32
32
64
64
gradient accumulation
steps
2
2
2
1
learning rate
1e-4
1e-4
2e-4
2e-4
num train epochs
75
75
100
75
GPUs
1xNvidia
H100
1xNvidia
H100
1xNvidia
H100
1xNvidia
H100
the DETR training time
5h
3h
4h
6h
the Mistral fine-tuning time
3h
2h
3h
-
Table 14: Examples from the test set of the raw model’s output, trained on the RoFT-chatgpt dataset. Bold text indicates either the ground truth interval or the predicted one.
Domain: Short Stories GT: Aryton blinked and rubbed his head. It had been a very high speed crash. He expected the impact to hurt more, but the whole thing just felt quite... fuzzy. There didn’t seem to be any track marshals around, which was odd, Aryton looked back towards the corner where he’d lost control. Nothing there, he pulled himself out of the car and scurried over the crash barrier to safety. That’s funny, he thought as he looked back at the crash, the car doesn’t seem damaged. Aryton walked back towards his car and inspected it closely. It was as if the crash had never happened, there wasn’t a scratch on it. He checked the fuel gauge, it was full, and the tires were still warm to the touch. It was a brand new car and one of the fastest ones that he had ever driven. Output: Aryton blinked and rubbed his head. It had been a very high speed crash. He expected the impact to hurt more, but the whole thing just felt quite... fuzzy. There didn’t seem to be any track marshals around, which was odd, Aryton looked back towards the corner where he’d lost control. Nothing there, he pulled himself out of the car and scurried over the crash barrier to safety. That’s funny, he thought as he looked back at the crash, the car doesn’t seem damaged. Aryton walked back towards his car and inspected it closely. It was as if the crash had never happened, there wasn’t a scratch on it. He checked the fuel gauge, it was full, and the tires were still warm to the touch. It was a brand new car and one of the fastest ones that he had ever driven.
Domain: New York Times
Domain: New York Times
GT: Last week, I.B.M. announced it would be opening a laboratory devoted to designing computers that link up hundreds or thousands of chips to gain tremendous speed, an approach to supercomputing known as massively parallel processing. For many in the