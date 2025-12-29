# Super-Prompting: Utilizing Model-Independent Contextual Data to Reduce Data Annotation Required in Visual Commonsense Tasks
Navid Rezaei and Marek Z. Reformat University of Alberta Edmonton, T6G 1H9, Canada
{nrezaeis,marek.reformat}@ualberta.ca
# Abstract
Pre-trained language models have shown excellent results in few-shot learning scenarios using in-context learning. Although it is impressive, the size of language models can be prohibitive to make them usable in on-device applications, such as sensors or smartphones. With smaller language models, task-specific data annotation is needed to fine-tune the language model for a specific purpose. However, data annotation can have a substantial financial and time burden for small research groups, startups, and even companies. In this paper, we analyze different prompt-based finetuning techniques to improve results on both language and multimodal causal transformer models. To evaluate our results, we use a dataset focusing on visual commonsense reasoning in time. Our results show that by simple model-agnostic promptbased fine-tuning, comparable results can be reached by only using 35%-40% of the fine-tuning training dataset. The proposed approaches result in significant time and financial savings. As the proposed methods make minimal architectural assumptions, other researchers can use the results in their transformer models with minimal adaptations. We plan to release the source code freely to make it easier for the community to use and contribute to our work.
# 1. Introduction
Human annotation is time-consuming and is also a financial burden for research groups, startups, and companies. To put it in context, almost $240,000 has been spent on the annotation of the Visual Commonsense Reasoning in Time (VisualCOMET) dataset and this figure only includes the payment to crowd-workers from Amazon Mechanical Turk [19]. The real financial burden can be much higher when including the time value of the staff involved in the annotation
process. Although large pre-trained language models, such as GPT-3 transformer [2], are impressive at multi-task fewshot learning, their huge size can be prohibitive for different scenarios, including on-device applications. Fine-tuning still plays an important role in achieving the state of the art, even with a relatively smaller model. As an example, the two current leading models1 (better than human baseline) on SuperGLUE task [28] are fine-tuned variants of T5 [21] and DeBERTa [9] language models, while GPT-3 is at 14th place. Our goal is to devise a model-independent process that could improve results based on fine-tuning with much less annotated training data.
Several recent works have focused on improving finetuning methods in language models, such as [12], [5], [14], and [30]. The focus has been put mostly on optimization and regularization, but not on using less data for fine-tuning. The results from those studies are complementary to our work. Some previous efforts have been put on prompt-based fine-tuning to improve classification or regression tasks in natural language processing (NLP). [24] and [25] convert textual inputs into cloze-style questions with a task description. [6] studies smaller language models for fewshot learning capability by using automatically-generated prompts for fine-tuning and by incorporating demonstrations into context. On another topic, a group of recent research studies, including [11], [20] and [16], aim at task-dependent added parameters to adapt models to different tasks. This way, one does not need to re-train a complete model to fine-tune it to a specific task but only needs to re-train a fraction of parameters.
There is a recent body of work that utilizes inherent knowledge of language models combined with fine-tuning on specialized large-scale training datasets to infer different commonsense and causal scenarios. [1] uses generative language models to expand on ATOMIC [23] and ConceptNet [26] commonsense knowledge graphs. [13] introduces an updated knowledge graph similar to ATOMIC and uses BART [15] encoder-decoder model to generate new knowledge. [17] uses generative language models to expand on an introduced knowledge base of causal mini-story explanations. Given the success of prompt-based fine-tuning and incontext learning in classification and regression tasks, we are motivated to assess similar principles in the context of commonsense generation using generative language models, which are fine-tuned on a commonsense knowledge graph.
# 3. Dataset
For this paper, we have selected a multi-modal commonsense knowledge graph for fine-tuning. The Visual Commonsense Reasoning in Time (VisualCOMET) dataset [19] consists of 1.4 million commonsense inferences over 59,356 images and 139,377 specific events at present. The dataset has human-annotated inferences regarding three different aspects: the intention of the person mentioned, the possible events that could happen next, and the possible preceding events. The inferences are made based on a single image. The annotators have access to short clips before and after the event, which are not part of the dataset. Each image is also annotated with event and place descriptions. There is a total amount of 1,465,704 commonsense inferences. The images are sourced from the VCR dataset [29]. The images usually have a complex visual scene with multiple people and activities present. This dataset includes automatically-detected object bounding boxes and people are annotated with numerical tags.
# 4. Method
In this work, we focus on using generative language models and analyze how prompt-based fine-tuning and incontext learning could help to reduce the size of the data required for fine-tuning training. As seen in Fig. 1, there are several scenarios where extra context could help lead the generative language model to a correct answer, but lack of correct understanding about the scene and the event text can result in incorrect results. Extra human annotations, focused on these shortcomings, could improve the results, but that comes with extra time and money expenditure. We propose using the underutilized context already present in text and image, then transforming them to a form
that is usable by most transformer models, which is a sequence. We analyze if this kind of addition helps the language model achieve better results in the case of limited annotated data available. Assuming the added context text is represented with c and its tokenized version with {c}, we can represent the context with {c} = {wc 1, wc 2, ...wc q}, where wc i represents each token created from tokenization of the context c. This context is merged with tokenized versions of event and place, which are represented as: {e} = {we 1, we 2, ...we n} and {p} = {wp 1, wp 2, ...wp m}, respectively. Using the merged versions of event and place texts with context, the updated sequence-to-sequence loss can be written as:
(1)
where v represents visual features, including overall images and person-specific boxes, r represents inference prompts, which could be intent, before and after, and w∗ <i represents past tokens for each case.
# 5. Experiments
The goal of the experiments is to see how much we could reduce the annotated data and still achieve results comparable to a case where the full human-annotated data is used. We tried different contextual data, which did not require extra annotations, such as captions, facial expressions, and related concepts. As shown in Fig. 1, we can intuitively see that some extra context could potentially help the language model to reach a more logical deduction of intention, past, and future events. For each scenario, the VisualCOMET dataset provides several human annotations for comparison, each showing intent of a person, what could happen next, and what happened before. The experiments are evaluated using BLEU [18], METEOR [4] and CIDEr [27] automatic metrics to compare the generated texts for different scenarios of before, intent and after with the human-annotated texts. We tried two different methods of adding relevant concepts. One method is based on converting relevant concept graphs into a readable sentence and the other method is based on only prepending concept words to the target sentence. In either method, the text is scanned for concepts, and the related concepts are extracted based on a commonsense knowledge graph such as in [26, 22]. Although sentencebased inputs perform well, they require a longer input width
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/32e9/32e9ff9b-d9cd-4a1e-af9c-45054a79b04f.png" style="width: 50%;"></div>
Method
BLEU-2
METEOR
CIDEr
GPT-2 [19]
13.81
10.85
15.37
Concept Word (NVP)
17.25
12.17
19.79
Concept Word (VP)
17.17
12.28
19.34
Concept Sent. (NVP)
14.92
11.25
16.9
Figure 1: Predictions based on the fine-tuned language model introduced in [19]. Each example shows a piece of missing contextual information that could be utilized.
that may not be available given the language model. To sort relevant concepts, crowd-based scores or frequency scores
Table 1: Effects of adding relevant concepts. Results are shown at the fourth epoch using almost 25,000 (22%) of the available annotated data. NVP: No Validation Prompt. VP: Validation Prompt.
Method
BLEU-2
METEOR
CIDEr
GPT-2 [19]
13.81
10.85
15.37
FE (NVP)
14.45
11.27
15.7
FE (VP)
15.11
11.23
16.03
Table 2: Effects of adding information about facial expressions. Results are shown at the fourth epoch using almost 25,000 (22%) of the available annotated data. NVP: No Validation Prompt. VP: Validation Prompt. FE: Facial Expressions.
are used based on the specific knowledge graph used. These triples are then converted to text with some hand-designed rules. An example of this process is shown in Fig. 2a. Table 1 shows three of the top-performing models with added conceptual contexts. They are compared with the original data, which does not have any added context. Evaluation is done on a validation dataset with a size of a hundred. Concept words added in this specific scenario are connected via HasProperty and PartOf predicates. Concept sentences use the HasProperty predicate. Adding similar information during inference time does not result in much improvement in this specific case. More comparisons can be found in the Appendix. As seen in Fig. 1a, lack of the model’s attention to some visual cues, such as facial expressions, could also result in errors of judgment. To fix this issue, we trained a ResNet [8] model on FER2013 [7] dataset with almost 70% accuracy. The dataset consists of human face images and emotion labels of angry, disgust, fear, happy, neutral, sad, and surprise. Only the emotion of people mentioned in the event text is processed. The results are then prepended to the event text. An example of this process is shown in Fig. 2b. Table 2 shows effects of adding facial expressions as a context in the final performance of the model. Contrary to relevant concepts, adding facial expressions during inference time improves the results. Evaluation is done on a validation data size of a hundred. Another type of automatically-generated context that
Method
BLEU-2
METEOR
CIDEr
GPT-2 [19]
13.81
10.85
15.37
Caption (NVP)
14.08
10.78
15.63
Caption (VP)
16.49
11.85
18.8
Table 3: Effects of adding image captions. Results are shown at the fourth epoch using almost 25,000 (22%) of the available annotated data. NVP: No Validation Prompt. VP: Validation Prompt.
we experimented with is image captioning. The idea is that some of the image dynamics may have been missed, even though image features are fed into the GPT-2 model. Adding generated captions proves to be effective as shown in Table 3. Meshed-Memory transformer model [3] with beam search decoding is used for image captioning. The process of adding these captions is illustrated in Fig. 2c. A mixture of different contextual information is shown to be more effective than individual ones. A combination of concept words, image captions, and facial expressions of relevant individuals in the image achieve the best result compared to other experiments. As seen in Table 4, this combination can achieve comparable results to fulldata finetuning by only using 35%-40% of the annotated data. This results in less human time spent doing annotations and can potentially reduce costs and completion times of projects. Results of other experiments are included in the Appendix. To reduce the effects of other variables in these experiments, we have limited ourselves to only train the final models for five epochs. The decoding method and hyperparameters are also kept constant throughout the experiments. We use nucleus sampling [10] with p = 0.9 to generate five sentences for each scenario of intent, before and after. The finetuning was run on two NVIDIA RTX GPUs with 24 GB memory each. For the case with all concept words, captions, and facial expression contexts, the fine-tuning time is around 1.5 hours per epoch while using mixed precision.
# 6. Conclusion
In this work, we analyzed the effects of automaticallygenerated contexts in multimodal transformer models used in a commonsensical task. These prompts can help us reduce the human annotation needed in the task by as much as 60%-65% and still, achieve comparable results to when the whole human-annotated dataset is used. These findings result in time and cost savings for future multimodal data annotation projects. As future work, it is interesting to find a lower bound for data annotation reduction without affecting the final result of a model. It is also useful to find a method to automat-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ce57/ce570fec-4e36-4abf-b835-2b7bf00b4b15.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9ac7/9ac7f80b-ebb4-4068-b2ec-e706b3dac7d8.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) Process of adding image captions</div>
Figure 2: The process of extracting and adding prompts shown through examples.
ically find and apply the best contextual data for different tasks and models.
# References
[1] Antoine Bosselut, Hannah Rashkin, Maarten Sap, Chaitanya Malaviya, Asli Celikyilmaz, and Yejin Choi. COMET: Commonsense transformers for automatic knowledge graph construction. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4762– 4779, Florence, Italy, July 2019. Association for Computational Linguistics. 2 [2] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. 1 [3] Marcella Cornia, Matteo Stefanini, Lorenzo Baraldi, and Rita Cucchiara. Meshed-Memory Transformer for Image Captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. 4
Method
Inference Data
Data Size
BLEU-2
METEOR
CIDEr
GPT-2 [19]
N/A
111,796 (100%)
18.05
13.21
22.72
CW + C + FE
C + CW + FE
39,000 ( 35%)
18.38
12.97
22.65
CW + C + FE
C + CW + FE
45,000 ( 40%)
18.58
13.01
22.97
le 4: Analyzing the effect of combining multiple contextual data. All models are finetuned for five epochs. Contexts ar ed based on the order shown. CW: Concept Words. C: Captions. FE: Facial Expressions.
# A. Appendix
Experimentation results from using different types of training and inference prompts are included in this appendix. The model used in the experiments is GPT-2 as described in [19]. The best types of prompts are chosen to be combined. The experiments show that the order in which prompts are added can affect the final results. The vision-based inference prompts seem to better affect the final metric results when compared to the text-based inference prompts. This could be due to the lack of enough visual attention paid during the decoding process. Future work could involve developing a multimodal model that makes better use of visual contexts not only during the training phase, but also the inference time. The quality of the annotated data can have an impact on the training model. We do not hand-select the annotated data based on quality and this may result in variability in final results when training with different data sizes. It can be a good practice to assess the quality of the annotated data and prompts based on the final goal of the model.
Training Prompt
Inference Prompt
Training Data Size
BLEU-2
METEOR
CIDEr
None
None
111,796 (100%)
17.94
13.14
22.71
None
None
25,000 ( 22%)
13.81
10.85
15.37
CS (AtLocation)
None
25,000 ( 22%)
12.26
10.42
15.09
CS (AtLocation) + Place
None
25,000 ( 22%)
14.3
11.08
14.66
CS (CapableOf)
None
25,000 ( 22%)
12.65
10.6
15.9
CS (CapableOf) + Place
None
25,000 ( 22%)
14.78
11.16
15.1
CS (HasA)
None
25,000 ( 22%)
12.73
10.65
15.83
CS (HasA) + Place
None
25,000 ( 22%)
14.58
11.15
15
CS (HasProperty)
None
25,000 ( 22%)
12.25
10.5
15.52
CS (HasProperty) + Place
None
25,000 ( 22%)
15.25
11.38
16.3
CS (IsA)
None
25,000 ( 22%)
12.73
10.41
15.73
CS (IsA) + Place
None
25,000 ( 22%)
14.04
11.03
14.32
CS (PartOf)
None
25,000 ( 22%)
12.65
10.51
15.71
CS (PartOf) + Place
None
25,000 ( 22%)
14.26
11.19
14.64
FE
None
25,000 ( 22%)
14.45
11.27
15.7
FE
FE
25,000 ( 22%)
15.11
11.23
16.03
CW (PartOf + HasProperty)
None
25,000 ( 22%)
17.25
12.17
19.79
CW (PartOf + HasProperty)
CW (PartOf + HasProperty)
25,000 ( 22%)
17.17
12.28
19.34
C
None
25,000 ( 22%)
14.08
10.78
15.63
C
C
25,000 ( 22%)
16.49
11.85
18.8
C + FE
C + FE
25,000 ( 22%)
16.75
12.19
19.16
CW + C + FE
None
25,000 ( 22%)
12.6
10.18
14.57
CW + C + FE
CW + C + FE
25,000 ( 22%)
17.4
11.97
20.03
CW + C + FE
CW + FE
39,000 ( 35%)
16.57
12.32
19.68
CW + C + FE
C + FE
39,000 ( 35%)
16.71
12.27
19.01
CW + C + FE
CW + C + FE
39,000 ( 35%)
16.75
12.4
19.86
CW + C + FE
CW + C + FE + Syns
39,000 ( 35%)
16.7
12.43
19.94
CW + C + FE + PCW
CW + C + FE + PCW
39,000 ( 35%)
17.74
12.73
20.52
CW + C + FE
C + CW + FE
39,000 ( 35%)
17.34
12.45
20.11
CW + C + FE
CW + C + FE
45,000 ( 40%)
17.46
12.84
21.56
CW + C + FE
C + CW + FE
45,000 ( 40%)
17.66
12.95
21.41
CW (PartOf + HasProperty) CW (PartOf + HasProperty)
Table 5: Experimentation results of using different training and inference prompts. The model used is GPT-2 [19]. Results are shown at epoch four and evaluated on validation data of size 100. Prompts are added based on the order shown. CW: Concept Words. C: Captions. FE: Facial Expressions. CS: Concept Sentences. Syns: Synonyms. PCW: Place Concept
