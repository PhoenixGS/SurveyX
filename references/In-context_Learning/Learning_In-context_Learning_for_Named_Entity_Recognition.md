4University of Chinese Academy of Sciences, Beijing, China {jiawei2020,yaojie,hongyu,boxi2020,xianpei,sunle}@iscas.ac.cn {loujie,jiawei07,daidai,wu_hua}@baidu.com
# Abstract
Named entity recognition in real-world applications suffers from the diversity of entity types, the emergence of new entity types, and the lack of high-quality annotations. To address the above problems, this paper proposes an in-context learning-based NER approach, which can effectively inject in-context NER ability into PLMs and recognize entities of novel types on-the-fly using only a few demonstrative instances. Specifically, we model PLMs as a meta-function λinstruction, demonstrations, text.M1, and a new entity extractor can be implicitly constructed by applying new instruction and demonstrations to PLMs, i.e., (λ.M)(instruction, demonstrations) →F where F will be a new entity extractor, i.e., F: text →entities. To inject the above in-context NER ability into PLMs, we propose a meta-function pre-training algorithm, which pre-trains PLMs by comparing the (instruction, demonstration)-initialized extractor with a surrogate golden extractor. Experimental results on 4 few-shot NER datasets show that our method can effectively inject in-context NER ability into PLMs and significantly outperforms the PLMs+fine-tuning counterparts.
# 1 Introduction
Named entity recognition (NER) aims to detect and classify named entities in text, such as People, Disease, and Movie. Traditional NER methods (Lample et al., 2016; Li et al., 2020c; Yan et al., 2021) have achieved remarkable success ∗This work was partially done when Jiawei Chen interned at Baidu. †Corresponding authors. 1This paper represents functions using lambdacalculus (Barendregt, 1992), and each function is represented as λx,y,z.M, where x, y, z are variables and M is function definition/abstraction. The function can apply to arguments such as (λx,y,z.M)(x = A, y = B, z = C) (fully applied) or (λx,y,z.M)(x = A, y = B) (partially applied).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6d9c/6d9c82cd-c4e4-4f0a-b97f-899edb6c697b.png" style="width: 50%;"></div>
<div style="text-align: center;">In-context NER</div>
Figure 1: Illustration of in-context NER, which uses instruction, demonstrations, and text as input to identify entities. The in-context learning model can be regarded as a meta-function that takes instruction and demonstrations as input and produces an entity extractor capable of identifying the desired entities (Akyürek et al., 2022). when entity types are pre-defined and massive highquality annotations are provided. Unfortunately, real-world NER still suffers from the diversity of entity types (e.g., the extraction of Movie is very different to Disease), the emergence of new entity types (e.g., Virus of Cov-19 ), and the lack of high-quality annotations. To address these problems, recent studies often employ few-shot learning techniques, including fine-tuning-based and metric-based methods. Finetuning-based methods extract entities of new types by adjusting model weights using new instances (Ma et al., 2022a; Chen et al., 2022a; Das et al., 2022). The main drawbacks of these methods are that re-training is often expensive (especially for large-scale models) and new entity types cannot be addressed on-the-fly. Metric-based methods are free from updating parameters and identifying entities by learning to compare query instances with support instances (or prototypes) (Yang and Katiyar, 2020; Tong et al., 2021). These methods are limited to the matching architectures and are sensitive to domain shift since they do not fully explore the information of target domain (Ma et al., 2022c). In this paper, we propose an in-context learning-
based NER approach, which can effectively address the above problems by injecting in-context NER ability into PLMs and then recognizing entities of new types on-the-fly using only a few demonstrative instances. Specifically, we model PLMs as a meta-function (Akyürek et al., 2022) for NER λinstruction, demonstrations, text.M, and a new entity extractor can be implicitly constructed by applying new instruction and demonstrations to PLMs, i.e., (λ.M)(instructions, demonstrations) →F where F will be a new entity extractor F: text →entities. For example, in Figure 1, our method can construct entity extractors of new Disease and Virus types on-the-fly by applying PLMs using demonstrations such as “Text: Cancer is a leading cause of death worldwide. Entities: Cancer is disease”. Furthermore, we propose a meta-function pre-training algorithm to inject the above in-context NER ability into PLMs. The algorithm pre-trains PLMs by comparing the implicitly (instruction, demonstration)constructed extractor with an explicitly fine-tuned surrogate golden extractor. The comparison ensures that the meta-function (λ.M) will generate an entity extractor F from instructions and demonstrations as accurately as possible. The proposed method can seamlessly leverage the powerful language understanding and generation capabilities of large-scale PLMs (Brown et al., 2020), effectively address diverse and new entity types through in-context learning, and only requires a couple of demonstrations for each entity type. Compared to fine-tuning methods, our method does not require expensive retraining, and new entity types can be extracted on-the-fly, with no need for model weight adjusting. Compared with metricbased methods, our method can dynamically utilize the information entailed in instruction and demonstrations rather than be limited to the fixed metric space. To verify the effectiveness of our method, we further pre-train PLMs using a large-scale distantly annotated NER dataset from Wikipedia and Wikidata. Experimental results on 4 few-shot NER benchmarks show that our method can effectively inject in-context NER ability into PLMs and significantly outperforms the PLMs+fine-tuning counterparts2. In general, this paper’s main contributions are: • We propose an in-context NER method that
• We propose an in-context NER method that can effectively extract entities of novel types
on-the-fly using only a few demonstrative in stances.
 We design a meta-function pre-training algorithm, which models PLMs as a meta-function and injects in-context NER ability into PLMs by comparing the (instruction, demonstration)constructed extractor with a surrogate golden extractor.
• How to inject in-context ability into small models is an important research direction of NLP in the big model era. Our work can benefit new directions for future works.
# 2 Related work
Few-shot NER Few-shot learning is a promising technique for low-resource NER. Currently, there are two main categories of FS-NER methods: fine-tuning-based methods and metric-based methods. Fine-tuning-based FS-NER methods re-train NER models using new instances. Metric-based methods identify entities by pre-training to compare query instances with support instances (Snell et al., 2017; Fritzler et al., 2019; Yang and Katiyar, 2020; Tong et al., 2021; Wang et al., 2022; Ji et al., 2022) using given NER datasets. FS-NER is a challenging task, and several improvements have been proposed to enhance its performance. These include leveraging label information (Hou et al., 2020; Wang et al., 2021a; Lu et al., 2022b; Ma et al., 2022a; Chen et al., 2022a; Yang et al., 2022), designing new paradigms such as decomposition methods (Ji et al., 2022; Ma et al., 2022c; Yang et al., 2022), prompt-based methods (Cui et al., 2021; Liu et al., 2022; Ma et al., 2022b), and demonstration-based methods (Lee et al., 2022; Zhang et al., 2022a); , and proposing new learning strategies like meta-learning (Li et al., 2020a,b; de Lichy et al., 2021; Ma et al., 2022c), contrastive learning (Das et al., 2022), and self-training (Huang et al., 2021; Wang et al., 2021b). In this paper, we address FS-NER via in-context learning (Gutiérrez et al., 2022), which empowers PLMs with incontext learning ability and entities of new entity types can be extracted on-the-fly.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a954/a9547702-2c4a-4ae4-b061-6e0ccb49ef5a.png" style="width: 50%;"></div>
Figure 2: The formats of input and output of in-context few-shot NER. The input is formed by instruction, demonstrations, and text.
aim to enhance in-context learning by selecting valuable demonstrations (Liu et al., 2021; Rubin et al., 2022), optimizing the order of demonstrations (Lu et al., 2022a), and calibrating output distributions (Zhao et al., 2021). Some studies try to replicate in-context learning in smaller models (Min et al., 2022a; Chen et al., 2022b). Additionally, some researchers attempt to replicate incontext learning using smaller models (Min et al., 2022b; Chan et al., 2022). Furthermore, there are efforts to understand the underlying mechanisms (Akyürek et al., 2022) of in-context learning which suggest that it can be compared to a metafunction and facilitate implicit fine-tuning (Dai et al., 2022; von Oswald et al., 2022). This paper is inspired by previous studies and considers incontext named entity recognition (NER) as a metafunction. To enhance the ability of pre-trained language models (PLMs) to perform in-context NER, we propose an effective pre-training algorithm. Unlike MetaICL (Min et al., 2022a), which only transforms multi-task learning into the form of incontext learning for pre-training, our approach also includes meta-function pre-training (Section 4.3) based on the underlying mechanisms of in-context learning.
# 3 In-context Named Entity Recognition
This section describes how to recognize entities through in-context NER. In in-context learning, the model will read the information of target entity types from both instruction and demonstrations, and then extract entities of target types within the text. In this way, new entity types can be extracted on-the-fly, without the need for model retraining.
Concretely, this paper formulates in-context NER as a sequence-to-sequence generation process. The input X = [I; D; T] includes instruction I, demonstrations D, and text T while the output is a list of extracted entities Y = [e1, ..., en]. Figure 2 shows an example, where an in-context NER model will identify that the target entity types are Disease and Virus, distill the knowledge about these types from demonstrations(e.g., the context patterns of a disease), and finally recognize "SARS-CoV-2" as virus and “COVID-19” as disease using the above knowledge. The details are described as follows. Instruction The instruction is a sequence of target entity types, guiding the model to extract what entity types (Min et al., 2022a). The instruction for target entity types {l1, . . . , ln} is I =“Target types: l1; . . . ; ln”. For example, in Figure 2 the instruction is “Target types: disease; virus”. Demonstrations Demonstrations provide the intra-class knowledge of target entity types (e.g., entity semantics and context patterns) and illustrate the form of outputs. As shown in Figure 2, the demonstrations contain the illustrative instances for different target types, and each instance is “Text: {text} Entities: {extractions}”, where {extractions} are entities presented in the {text}. Extractions The output of the extraction process is a list of entities, denoted as Y = [e1, . . . , en] where ei is i-th extracted entities. Each extraction e is represented as “ENTITY is type”. For instance, in Figure 2, the extraction “COVID-19 is disease.” indicates that “COVID-19” is an entity mention with the type “Disease”. This natural languagelike representation allows us to better utilize the text generation capabilities of pre-trained language models. During inference, we locate all mentions in the text and further output their locations. Architecture Given the above task formulation, we employ an encoder-decoder network like T5 (Raffel et al., 2020), where the encoder encodes <instruction, demonstrations, text> and the decoder generates all extractions as a tokenized text sequence Y = [y1, . . . , yn]. The success of in-context NER depends on two critical abilities: the in-context learning ability and the extraction ability. For in-context learning, the models should be able to implicitly construct accurate extractors of new entity types by following the instruction and capturing the knowledge in demon-
Extractions The output of the extraction process is a list of entities, denoted as Y = [e1, . . . , en] where ei is i-th extracted entities. Each extraction e is represented as “ENTITY is type”. For instance, in Figure 2, the extraction “COVID-19 is disease.” indicates that “COVID-19” is an entity mention with the type “Disease”. This natural languagelike representation allows us to better utilize the text generation capabilities of pre-trained language models. During inference, we locate all mentions in the text and further output their locations.
Architecture Given the above task formulation, we employ an encoder-decoder network like T5 (Raffel et al., 2020), where the encoder encodes <instruction, demonstrations, text> and the decoder generates all extractions as a tokenized text sequence Y = [y1, . . . , yn]. The success of in-context NER depends on two critical abilities: the in-context learning ability and the extraction ability. For in-context learning, the models should be able to implicitly construct accurate extractors of new entity types by following the instruction and capturing the knowledge in demon-
strations. In this way, we can see a PLM as a meta-function, i.e., a function of extractors whose input is (instruction, demonstrations) and whose output is an entity extractor. For extraction, the models should be able to locate specific spans and categorize them into target entity types. The following section demonstrates how to inject such an in-context learning ability into PLMs and construct an effective in-context NER model.
# 4 Meta-Function Pre-training for In-Context NER
In this section, we will explain how to incorporate in-context named entity recognition (NER) capabilities into pre-trained language models (PLMs). Although large-scale PLMs like GPT-3 have demonstrated the ability to learn in-context, this capability is not always controllable or predictable. Additionally, unlike classification and question-answering tasks that align with the pre-training objective of language models (i.e., producing natural text output), NER requires more complex span extraction and type specification. As a result, Gutiérrez et al. (2022) show that LMs aren’t well-suited for incontext NER tasks. In this paper, we propose metafunction pre-training, an algorithm that can inject in-context NER ability into PLMs in a controllable and predictable way. Specifically, we model PLMs as a metafunction (Akyürek et al., 2022) for NER λinstruction, demonstrations, text.M, and a new entity extractor can be implicitly constructed by applying new instructions and demonstrations to PLMs, i.e., (λ.M)(instructions, demonstractions) →F where F will be a new entity extractor F:text →entities. Based on the meta-function formulation, we further pre-train PLMs for in-context NER abilities by: • optimizing PLMs via a meta-function loss, so that the implicitly (instruction, demonstration)-constructed extractor F will be as close as an explicitly fine-tuned surrogate golden extractor; • optimizing PLMs via an extraction loss, so that the in-context NER can effectively locate and categorize entities in a text. The details are described in the following.
# 4.1 Pre-training Settings
Pre-training Corpus Construction To continually pre-train PLMs for in-context NER, we first collect an in-context pre-training NER corpus
Din-context = {x1, x2, ..., xn}, where each x is an n-context NER task represented as a tuple = (intruction, demonstrations, text, entities). Specifically, to sample in-context NER task x, we use traditional NER corpus DNER where each NER instance is a (text, entities) pair as follows: 1. In-context Task Sampling: To construct an in-context NER task x = (instruction, demonstrations, text, entities): (1) we first sample N target entity types from DNER to form instruction and sample K instances for each type to form demonstrations; (2) then we sample the text and the entities of x by either randomly sample an instance from N target entity types, or randomly sample from instances of other entity types, i.e., their extractions are NIL. We sample NIL instances because in real-world applications many instances will not contain target entities, and NIL instances are sampled with a predefined proportion γ. 2. Type Anonymization: To ensure the models rely on in-context demonstrations for entity knowledge and avoid overfitting to entity type names, we anonymize entity types by randomly substituting them with a set of type indicators {<type1>, . . ., <type99>}, rather than directly using the original type names such as Disease and Virus. We found this anonymization strategy can significantly improve the in-context learning ability of PLMs. Specifically, we randomly substitute each entity type name with pre-defined 99 type indicators {<type1>, . . ., <type99>}, and the substitute probability for each name is 80%.
# Pre-training Loss Based on the in-context pretraining corpus Din-context, we pre-train our incontext NER model by optimizing the loss:
L = αLmeta-function + Lextraction
(1)
where Lmeta-function is the meta-function loss which ensures PLMs can implicitly generate accurate entity extractors (Section 4.2), Lextraction is the extraction loss which ensures PLMs have good extraction ability (Section 4.3), α is the coefficient of metafunction loss.
# 4.2 Meta-function Pre-training
As mentioned above, a good in-context NER model should be able to implicitly construct an accurate entity extractor by partially applying PLMs with
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/02d1/02d1bc79-85a5-49e4-a0f9-6cc8c3c46210.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Overview of our meta-function pre-training. Our goal is to ensure that the extractor F(instruction,demonstrations) closely resembles the golden extraction function. To obtain the golden extraction function, we use a surrogate strategy and the surrogate extraction function is the fine-tuned encoder using demonstrations.</div>
# instruction I and demonstrations D:
(2)
For example, given the instruction and demonstrations in Figure 2, we want PLMs to implicitly build an accurate extractor for Disease and Virus. Therefore if we know the golden extraction function F∗ for target entity types, we can optimize PLMs for in-context NER ability by minimizing the distance ||F∗−F||. Unfortunately, the golden extraction function F∗is unknown. In this paper, we approximate F∗using a surrogate extractor which is the finetuned counterpart using demonstrations D. That is, for each in-context pre-training task x, we first recover all NER (text, entities) instances from x as x′, then we fine-tune the model and use the fine-tuned encoder F′ as the surrogate of F∗. The overall meta-function pre-training is shown in Figure 3. Formally, given instruction I, demonstration D, and text T, we first feed them into the encoder and obtain the feature of I and T,
l1, ..., ln, d1, ..., dm, t1, ..., tk = Encoder(I; D; T) (3
(3)
Then we obtain the feature of the implicitly generated function F using the features of instruction I and text T, and ignore the features of D: F = [l1, ..., ln, t1, ..., tk]. In Figure 3, the feature F can be seen as the output of Disease and Virus extractor F. To obtain the feature of the fine-tuned counterpart, we perform a one-step gradient descent3 on
the encoder using the instances in the demonstration D and get the surrogate encoder, which can be seen as an approximation of golden F∗. Note that this fine-tuning operation is performed after the model has been copied, so there is no impact on the parameters of the original model. In the example in Figure 3, Encoder′ is a Disease and Virus extractor. After performing one-step updating, we feed instruction and text [I; T] into the surrogate encoder to get their features:
(4)
where F′ = {l′ 1, . . . , l′ n, t′ 1, . . . , t′ k} is features of instruction I and text T. In the example in Figure 3, the feature F′ can be seen as the estimated output of golden extractor F∗for Virus and Disease entity types. Then, we pre-train our in-context NER model to be a good meta-function by making the output of F and F ∗consistent, i.e., minimizing the distance between F and F′. The meta-function loss is:
(5)
where d(·) is euclidean distance. Note that when calculating the gradient of Lmeta-function, F′ is seen as constant. To this end, the meta-function gradient can be estimated as:
(6)
where θencoder is the parameters of the encoder and X = [I; D; T] is the input. The estimated gradient will be used to update the parameters of the encoder.
In this way, the in-context NER models will be trained to be a good meta-function (Akyürek et al., 2022), which can also be seen as an ability for implicit fine-tuning (Dai et al., 2022; von Oswald et al., 2022).
# 4.3 Extraction Function Pre-training
Besides the in-context learning ability, we also pretrain PLMs to be good extractors via extraction loss. Given instruction I, demonstrations D, and text T, the sequence-to-sequence entity extractor directly models the generation probability token by token in an auto-regressive way. Formally, we optimize the model parameters θ by minimizing the negative likelihood of in-context instances:
(7)
And the extraction gradient is computed as:
(8)
To learn the above extraction ability, we design two extraction pre-training tasks, including an entity extraction task and a pseudo extraction language modeling task: Entity Extraction Task. This task is used to train the ability to extract entities from text, we use both in-context NER settings whose input is (instruction, demonstrations, text) and traditional NER settings whose input is (instruction, text), and output is entities. Note that type anonymization is only conducted in in-context NER setting.
# Pseudo Extraction Language Modeling Task
Because there is a mismatch between the entity extraction task and the original language modeling task, and the size of the NER corpus is usually far smaller than the text corpus for language modeling pre-training, we design a pseudo extraction LM task to bridge the above gap. Specifically, we randomly sample unlabeled sentences from the text corpus and automatically build pseudo extraction (instruction, demonstrations, text, pseudo entities) tasks. For instance, given a demonstration sentence such as “I think this movie is cool and I really like it very much” and a text “I do not like it.”: (1) To begin with, we choose some spans from demonstrations (such as "this movie" and "like") and designate them as pseudo entities4. We assign 4We introduce how to select spans in Appendix.
random types to these entities from type indicators. For instance, we consider "this movie" as a pseudo entity of type <type2> and "like" as a pseudo entity of type <type14>. (2) The input of the pseudo extraction task is instruction="Target types:<type2>; <type14>"; the demonstrations="Text: [MASK1] is cool and I really [MASK2] it [MASK3]. Entities: [MASK1] is <type2>. [MASK2] is <type14>" where the entities (“this movie” and “like”) and other random spans (“very much”) in demonstrations are masked. The text=“Text: I do not like it.” which is not masked. (3) The output of the pseudo extraction task is “like is <type14>” since the model will learn from demonstrations that <type14> corresponds to "like". (4) We also conduct traditional NER settings whose input is (instruction, text). The entities in the text will be masked as in demonstrations, e.g. “Target types: this movie; like Text: I [MASK1] not [MASK2] it.”. The output will be “Entities: [MASK2] is like.”. We can see that the pseudo extraction LM task can benefit in-context NER in two ways. Firstly, it can significantly increase the size and diversity of in-context NER pre-training tasks from a largescale unlabeled corpus. Secondly, this task pretrains PLMs with a mixture of extraction target and span prediction task, therefore avoiding PLMs overfit to only extraction task. When pre-training, We transformed the NER and language model tasks into a uniform format and sampled input instances alternately.
# 5 Experiments
This section evaluates our method by conducting experiments on few-shot NER settings.
# 5.1 Experimental Settings
Pre-training settings. Following Chen et al. (2022a), we build a large-scale distant NER dataset by aligning Wikipedia and Wikidata. Specifically, our dataset was made from Wikipedia text with hyperlinks to Wikidata, where we labeled entity types using the linked Wikidata item’s attributes. Entity types were gathered from Wikidata’s SubclassOf and InstanceOf attributes for each span. We filtered ambiguous and low-frequency types (occurrences <100k) to obtain higher-quality demonstrations. Finally, we retained 2046 types and 55 million (text, entities) pairs and use a 40/15 million split for training/validation. We sample 5 million in-context tasks for training and 10k for valida-
Models
#Param
CoNLL03
WNUT17
NCBI-disease
SEC-filings
AVE
1-shot
5-shot
1-shot
5-shot
1-shot
5-shot
1-shot
5-shot
Pre-trained Language Models
T5v1.1-large
770M
38.61
44.90
25.52
26.32
26.02
37.63
41.89
53.44
36.79
GPT2-xl
1.5B
33.69
39.55
22.63
24.86
25.54
33.25
42.83
57.05
34.93
T5-xl
3B
38.99
45.74
26.39
26.31
23.10
36.78
30.58
42.22
33.76
GPT-J-6B
6B
46.14
50.10
31.41
30.93
35.82
40.98
40.12
39.61
39.39
T5-xxl
11B
40.97
46.14
24.76
25.27
12.19
26.34
32.65
42.44
31.35
OPT-13B
13B
46.65
51.71
27.74
28.36
23.73
34.00
41.60
43.10
37.11
GPT-Neox-20B
20B
52.68
58.12
36.29
35.68
35.42
42.85
45.07
45.17
43.91
OPT-30B
30B
42.86
44.77
25.85
27.44
22.31
32.76
40.83
46.52
35.42
OPT-66B
66B
43.83
53.89
30.77
32.00
25.87
34.58
39.15
47.01
38.39
Pre-trained NER Models
ProtoNet
345M
30.04
60.26
9.74
23.03
24.73
42.32
16.79
23.67
28.82
NNShot
345M
41.92
58.39
15.76
21.78
31.59
33.14
30.19
37.86
33.83
StructShot
345M
42.34
58.44
15.78
22.05
19.87
31.48
30.40
38.44
32.35
CONTAINER
345M
45.43
61.69
15.64
20.37
23.24
27.02
34.07
40.44
33.49
MetaNER-base
220M
53.94
62.59
25.55
30.41
35.00
37.24
46.88
51.39
42.88
MetaNER
770M
57.40
63.45
31.59
36.52
40.01
44.92
52.07
54.87
47.60
<div style="text-align: center;">Pre-trained NER Models</div>
Pre-trained NER Models
ProtoNet
345M
30.04
60.26
9.74
23.03
24.73
42.32
16.79
23.67
28.82
NNShot
345M
41.92
58.39
15.76
21.78
31.59
33.14
30.19
37.86
33.83
StructShot
345M
42.34
58.44
15.78
22.05
19.87
31.48
30.40
38.44
32.35
CONTAINER
345M
45.43
61.69
15.64
20.37
23.24
27.02
34.07
40.44
33.49
MetaNER-base
220M
53.94
62.59
25.55
30.41
35.00
37.24
46.88
51.39
42.88
MetaNER
770M
57.40
63.45
31.59
36.52
40.01
44.92
52.07
54.87
47.60
Table 1: Micro-F1 scores of 1-shot and 5-shot in-context NER on test set. For a fair comparison, the results of each model are based on a single frozen model without fine-tuning and the pre-trained NER models are pre-trained using the same dataset as MetaNER.
tion, where each instance with type number N is 10 and instance number K is 10. We employ the T5-v1.1-large (Raffel et al., 2020) model as the initial model for MetaNER and further pre-train 500k steps with learning rate=5e-5 and warm-up steps=10k. In this paper, we refer to the pre-trained model as MetaNER. Few-shot settings. Our experiments follow the standard k-shot NER setting Huang et al. (2021): For each entity type, we sample k training instances as in-context demonstrations. We evaluate models by micro-F1 and report the average performance by repeating each experiment 10 times. We conducts experiments on 4 datasets across differnt domains: (1) CoNLL03 (Sang and Meulder, 2003) from news domain. (2) WNUT17 (Derczynski et al., 2017) from social media domain. (3) NCBI-disease (Do˘gan et al., 2014) from biology domain. (4) SEC-filings (Alvarado et al., 2015) from finance domain. Baselines. For fair comparison, we use frozen models for all baselines in the in-context learning experiments, i.e., a pre-trained language/NER model is used for entity extraction without finetuning. In addition, we will discuss fine-tuning based methods in section 5.3.3. Two kinds of baselines are compared:
Baselines. For fair comparison, we use frozen models for all baselines in the in-context learning experiments, i.e., a pre-trained language/NER model is used for entity extraction without finetuning. In addition, we will discuss fine-tuning based methods in section 5.3.3. Two kinds of baselines are compared:
1) Pre-trained language models include models with different scales and architectures: (1) Encoderdecoder models – T5 models (Raffel et al., 2020), includes T5-v1.1-large (770M), T5-xl (3B) and T5xxl (11B). (2) Causal LM models – GPT and OPT models (Radford et al., 2019; Zhang et al., 2022b), includes GPT2-xl (1.5B), GPT-j-6B (Wang and Komatsuzaki, 2021), GPT-Neox-20B (Black et al., 2022), OPT-13B, OPT-30B and OPT-66B. Notice that, for PLMs, we use original type names rather than type indicators to capture the label semantics. For encoder-decoder models like T5, we formulate in-context NER as a span corruption task and the model will generate the extraction task. For example, for input “Target entity types: disease. Text: COVID-19 is spreading. Entities: COVID-19 is disease. Text: HIV is spread by three main routes. Entities: <extra_id_0>”, the span corruption task requires the decoder to generate the extraction result “<extra_id_0> HIV is disease.”. 2) Pre-trained NER models are metric-based few-shot methods, includes prototype network (ProtoNet) (Snell et al., 2017), NNshot (Yang and Katiyar, 2020), StructShot (Yang and Katiyar, 2020) and CONTAINER (Das et al., 2022). We employed BERT-Large (Devlin et al., 2019) as the backbone and pre-trained them using the same dataset as MetaNER. For a fair comparison, we
also pre-train a 220M T5-v1.1-base (Raffel et al., 2020) model with our meta-function pre-training algorithm (MetaNER-base).
# 5.2 Main Results
The experimental results are shown in Table 1. We can see that: 1) Few-shot NER is challenging even for large language models, while MetaNER can achieve good in-context NER performance. Compare with best-performed PLMs, MetaNER achieves 8.4% F1 improvements. Moreover, due to the gap between language model task and NER task, large language models achieve poor in-context learning performance on some datasets. 2) Our in-context NER method can achieve robust performance, even under a large sourcetarget domain gap. Compared with bestperformed metric-based NER models, MetaNERbase and MetaNER achieves 26.8% and 40.7% F1 improvement. Specifically, the performance improvement is more significant when source-target domain gap is larger, i.e., the NCBI-disease (biology domain) and SEC-filings (finance domain). 3) Meta-function pre-training can effectively inject in-context learning ability into both small and large PLMs. Both MetaNER-base and MetaNER achieve impressive performance in 1-shot and 5-shot settings, which verified that MetaNER can effectively inject in-context NER ability into small PLMs, although currently incontext learning has been seen an ability only emerged only on large language models such as GPT-3.
# 5.3 Detailed Analysis 5.3.1 Ablation Studies
CoNLL03
NCBI-disease
P
R
F1
P
R
F1
MetaNER
73.59
57.19
64.34
54.96
36.85
43.79
w/o MF
68.97
57.62
62.77
38.27
35.26
36.28
w/o LM
70.86
57.99
63.77
37.54
34.82
35.67
w/o anonymization
74.75
52.86
61.93
47.47
35.30
40.48
Table 2: Ablation studies on dev set. The results are based on 5-shot setting.
To analyze and understand the effect of type anonymization, meta-function pre-training, entity extraction pre-training, and pseudo extraction LM pre-training, we conduct the following ablation experiments: (1) MetaNER w/o MF: remove the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5f70/5f708953-8ca3-4bef-9367-5d74d4433403.png" style="width: 50%;"></div>
Figure 4: The visualization of feature comparison between meta-function F and the surrogate extractor F′. The x-axis represents the different datasets, and the yaxis represents the distances between the features from the original encoder and the features from the surrogate encoder.
meta-function pre-training; (2) MetaNER w/o LM: remove pseudo extraction LM pre-training; (3) MetaNER w/o anonymization: we use the original entity type names in both pre-training and incontext NER, without using type anonymization. The results are shown in Table 2, we can see that: 1) meta-function pre-training is critical for in-context learning ability. By removing the meta-function pre-training, the results drop significantly when the domain gaps are larger, i.e., NCBI-disease. At the same time, meta-function pre-training is helpful for the model to make more precise predictions. 2) The pseudo extraction LM task significantly benefits in-context NER. We found MetaNER w/o LM results in a performance drop than MetaNER. We believe this is because, although using an automatically constructed pseudo dataset, this task can significantly improve the size and the diversity of in-context NER tasks, meanwhile can retain a good language modeling ability. 3) Type name anonymization prevents incontext NER model from type name overfitting, and therefore enhances the in-context learning ability. The ablation of type name anonymization results a 5.7% performance drop in Table 2. We believe this is because type names will let models tend to memorize entity knowledge using type names, thus the model will not learn to capture entity knowledge from demonstrations on-the-fly.
# 5.3.2 Effects of Meta-function Pre-training
One main idea of this paper is that in-context NER model can be viewed as a meta-function which can
implicitly build new entity extractors. To demonstrate whether meta-function pre-training can train a good meta-function, we sample 1000 instances from each dataset, and show the difference between the (instruction, demonstrations)-initialized entity extractor F and the surrogate entity extractor F′, i.e., ||F′ −F|| in Section 4.2 in Figure 4. We can see that meta-function pre-training can equip PLMs with a good meta-function ability, i.e., the (instruction, demonstrations)-initialized entity extractor after pre-training is significantly close to its fine-tuned counterpart.
CoNLL03
WNUT17
1shot
5shot
1shot
5shot
BERT-large (Devlin et al., 2019)
14.66
52.43
8.95
32.77
T5-v11-large (Raffel et al., 2020)
11.65
42.13
12.51
39.54
GPT-NEO-20B (Black et al., 2022)*
52.68
58.12
36.29
35.68
UIE-large (Lu et al., 2022b)
46.28
67.62
32.86
42.67
SDNet (Chen et al., 2022a)
/
71.40
/
44.10
CONTAINER-FT (Das et al., 2022)
48.56
66.45
19.46
24.95
MetaNER-ICL*
57.40
63.45
31.59
36.52
MetaNER-FT
61.51
72.70
39.68
47.26
Table 3: The experiments of fine-tuning based methods. * indicates in-context learning settings. CONTAINER is pre-trained using the same NER dataset as MetaNER. All the models are implemented by us except SDNet.
# 5.3.3 In-context Learning vs Fine-tuning
MetaNER can also be directly fine-tuned using traditional NER instances. We employed the identical fine-tuning approach as previous works (Huang et al., 2021; Lu et al., 2022b; Chen et al., 2022a). Following Lu et al. (2022b), we also implemented the Rejection Mechanism when fine-tuning the T5v11-large and MetaNER to achieve better few-shot performance. To compare in-context NER with fined-tuned NER, Table 3 reports the performance of the finetuned counterpart of MetaNER – MetaNER-FT(its training is similar to surrogate entity extractor but with multi-step gradient descent until coverage), together with several fine-tuned few-shot NER baselines. We can see that: 1) MetaNER is an effective architecture, which achieves good performance on both in-context learning and fine-tuning settings; 2) Currently, fine-tuning can achieve better performance than their in-context learning counterpart. We believe this is because fine-tuned models’ parameters need to be specialized to specific entity types, meanwhile in-context learning needs to generalize to different types on-the-fly, i.e., generalization-specialization trade-off. We believe this also verified the reasonableness of using
a fine-tuned surrogate extractor to approximate the golden extractor.
# 6 Conclusion
In this paper, we propose an in-context learningbased NER approach and model PLMs as a metafunction, which can inject in-context NER ability into PLMs and recognize entities of new types onthe-fly using only a few demonstrative instances. Experimental results show that our method is effective for in-context NER. For future work, we will extend our method to different NLP tasks like event extraction and relation extraction.
# Limitations
In-context learning is an useful ability, this paper only focuses on in-context named entity recognition, leaves the learning of other NLP tasks’ incontext learning abilities for future work. Currently, we learn in-context learning via metafunction pre-training, by comparing an in-context extraction function and a fined-tuned surrogate extraction function at the representation level of their encoders. There are two approximation here: one is fined-tuned surrogate extraction function for approximating golden extraction function, and the difference between representations for approximating the divergence between functions. We think the above two approximations can be further improved for better and faster in-context learning.
# Acknowledgements
We sincerely thank the reviewers for their insightful comments and valuable suggestions. This research work is supported by the CAS Project for Young Scientists in Basic Research under Grant No.YSBR-040 and the National Natural Science Foundation of China under Grants no. 62122077, 62106251.
# References
Ning Bian, Xianpei Han, Bo Chen, Hongyu Lin, Ben He, and Le Sun. 2021. Bridging the gap between language model and reading comprehension: Unsupervised mrc via self-supervision. arXiv preprint arXiv:2107.08582.
Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. 2022. GPT-NeoX-20B: An opensource autoregressive language model. In Proceedings of the ACL Workshop on Challenges & Perspectives in Creating Large Language Models.
om Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.
Stephanie CY Chan, Adam Santoro, Andrew K Lampinen, Jane X Wang, Aaditya Singh, Pierre H Richemond, Jay McClelland, and Felix Hill. 2022. Data distributional properties drive emergent fewshot learning in transformers. arXiv preprint arXiv:2205.05055.
Jiawei Chen, Qing Liu, Hongyu Lin, Xianpei Han, and Le Sun. 2022a. Few-shot named entity recognition with self-describing networks. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5711–5722, Dublin, Ireland. Association for Computational Linguistics.
Mingda Chen, Jingfei Du, Ramakanth Pasunuru, Todor Mihaylov, Srini Iyer, Veselin Stoyanov, and Zornitsa Kozareva. 2022b. Improving in-context few-shot learning via self-supervised training. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3558–3573, Seattle, United States. Association for Computational Linguistics.
Leon Derczynski, Eric Nichols, Marieke van Erp, and Nut Limsopatham. 2017. Results of the WNUT2017 shared task on novel and emerging entity recognition. In Proceedings of the 3rd Workshop on Noisy User-generated Text, NUT@EMNLP 2017, Copenhagen, Denmark, September 7, 2017, pages 140–147. Association for Computational Linguistics.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.
Rezarta Islamaj Do˘gan, Robert Leaman, and Zhiyong Lu. 2014. Ncbi disease corpus: a resource for disease name recognition and concept normalization. Journal of biomedical informatics, 47:1–10.
iaxin Huang, Chunyuan Li, Krishan Subudhi, Damien Jose, Shobana Balakrishnan, Weizhu Chen, Baolin Peng, Jianfeng Gao, and Jiawei Han. 2021. Fewshot named entity recognition: An empirical baseline study. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10408–10423, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
Bin Ji, Shasha Li, Shaoduo Gan, Jie Yu, Jun Ma, Huijun Liu, and Jing Yang. 2022. Few-shot named entity recognition with entity-level prototypical network enhanced by dispersedly distributed prototypes. In Proceedings of the 29th International Conference on Computational Linguistics, pages 1842–1854, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.
Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022a. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland. Association for Computational Linguistics.
Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In International Conference on Machine Learning, pages 12697–12706. PMLR.
# A Experiment Details
# A.1 Datasets for the extraction language model task
Rather than randomly generating spans to form target labels in instruction, we use informative spans (Bian et al., 2021) as target labels. Unlike informative span selection at passage level for MRC (Bian et al., 2021), we select informative spans at a cross-document level. Specifically, we take 10 Wikipedia documents as a set and select informative spans according to the following rules: (1) spans that have appeared simultaneously in at least two and at most five documents. (2) spans that have appeared in only one document but have appeared in more than two. Rule (1) avoids some
low-information general spans, such as stop words, and rule (2) retains some important spans in each document. Note that we consider at most 4-gram as a span and select the target labels from the informative spans during pre-training.
# A.2 Cost of pre-training
We used one A-100 80g GPU for pre-training the base/large model, which took approximately one to three days. The total FLOPs for the base model are 2.30e+18 and for the large model are 7.64e+18.
