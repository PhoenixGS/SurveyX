# Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation
Bowen Zheng∗, Yupeng Hou†, Hongyu Lu‡, Yu Chen‡, Wayne Xin Zhao∗�, Ming Chen‡, and Ji-Rong W ∗Gaoling School of Artificial Intelligence, Renmin University of China, China †University of California San Diego, United States ‡WeChat, Tencent, China
owen Zheng∗, Yupeng Hou†, Hongyu Lu‡, Yu Chen‡, Wayne Xin Zhao∗�, Ming Chen‡, and Ji-Rong Wen∗ ∗Gaoling School of Artificial Intelligence, Renmin University of China, China †University of California San Diego, United States ‡WeChat, Tencent, China bwzheng0324@ruc.edu.cn, yphou@ucsd.edu, luhy94@gmail.com, nealcui@tencent.com, batmanfly@gmail.com, mingchen@tencent.com, jrwen@ruc.edu.cn
bwzheng0324@ruc.edu.cn, yphou@ucsd.edu, luhy94@gmail.com, nealcui@tencen batmanfly@gmail.com, mingchen@tencent.com, jrwen@ruc.edu.cn
Abstract—Recently, large language models (LLMs) have shown great potential in recommender systems, either improving existing recommendation models or serving as the backbone. However, there exists a large semantic gap between LLMs and recommender systems, since items to be recommended are often indexed by discrete identifiers (item ID) out of the LLM’s vocabulary. In essence, LLMs capture language semantics while recommender systems imply collaborative semantics, making it difficult to sufficiently leverage the model capacity of LLMs for recommendation. To address this challenge, in this paper, we propose a new LLMbased recommendation model called LC-Rec, which can better integrate language and collaborative semantics for recommender systems. Our approach can directly generate items from the entire item set for recommendation, without relying on candidate items. Specifically, we make two major contributions in our approach. For item indexing, we design a learning-based vector quantization method with uniform semantic mapping, which can assign meaningful and non-conflicting IDs (called item indices) for items. For alignment tuning, we propose a series of specially designed tuning tasks to enhance the integration of collaborative semantics in LLMs. Our fine-tuning tasks enforce LLMs to deeply integrate language and collaborative semantics (characterized by the learned item indices), so as to achieve an effective adaptation to recommender systems. Extensive experiments demonstrate the effectiveness of our method, showing that our approach can outperform a number of competitive baselines including traditional recommenders and existing LLM-based recommenders. Our code is available at https://github.com/RUCAIBox/LC-Rec/. Index Terms—Large Language Model, Semantic Integration, Sequential Recommendation
# I. INTRODUCTION
Nowadays, recommender systems have become an essential part of various application platforms, aiming to recommend potential information resources to users based on their specific preferences. Since user preferences dynamically evolve over time, sequential recommendation has attracted great research attention due to its advantages in capturing the sequential characteristics of user behaviors. To develop sequential recommenders, existing recommendation models [1], [2] are mostly built on sequential formatting of user interaction logs, taking item ID as the basic unit. In existing literature, sequential recommenders adopt various deep neural networks to model user historical behaviors represented by item ID sequences, including RNN [1], [3],
CNN [4], [5], GNN [6], [7], and Transformer [2], [8]. In addition to collaborative semantics within users’ historical behaviors, some studies [9]–[11] also try to enhance item sequence modeling by leveraging content information (e.g., title, description, category). Moreover, pre-trained language models (PLMs) have also been employed for capturing the textual semantics reflected in item texts [12]–[15], to improve the recommendation performance. Recently, the emergence of large language models (LLMs) has triggered a significant revolution in the research community. LLMs have shown great potential in various language based tasks, due to their excellent capabilities in semantic understanding and generation [16]. Specifically, there are also several attempts [17] that adapt LLMs for recommender systems (RS), to improve the item ranking performance [18] or boost the comprehensive recommendation capacities [19]. To develop capable LLM-based recommendation models, a fundamental challenge is that there exists a large gap between the language semantics modeled by LLMs and collaborative semantics implied by recommender systems. The key point is that, in existing recommendation models, user behaviors are often formatted into item ID sequences (possibly with feature IDs), but not textual descriptions. In other words, language models and recommendation models indeed employ two different vocabularies (token IDs v.s. item IDs) to learn their own semantic spaces. Such a semantic gap makes it difficult to sufficiently leverage the model capacity of LLMs for tackling the recommendation tasks. To address this issue, existing efforts can be divided into two main approaches. The first approach [18]–[21] verbalizes the user behaviors into text sequences (e.g., concatenating the titles and category labels of the interacted items), and designs special prompts to instruct LLMs for fulfilling the recommendation tasks. Such an approach only captures limited item information (only considering language semantics), and can’t guarantee the generation of in-domain items (relying on a candidate set). As another alternative approach, several studies [22], [23] design special item indexing mechanisms for building item vocabulary, and then learn to generate the target item for recommendation. However, given the large semantic gap, simple (e.g., vocabulary building with vanilla item IDs) or shadow integration (e.g., fine-tuning only with the target task) would be less effective
to adapt LLMs for recommender systems. Considering these issues, we aim to design a more effective semantic integration approach for developing LLM-based recommendation models. We tackle this semantic integration problem in two main aspects, namely item indexing and alignment tuning1. For item indexing, an ideal allocation mechanism should produce meaningful (capturing item similarities), unique (without allocation conflicts), and extensible (generalizable to new items) IDs for effectively representing the items. For alignment tuning, it should be able to sufficiently integrate language semantics with collaborative semantics in LLMs, but not superficially fit the target recommendation task. Overall, our goal is to effectively establish the connections between the two kinds of different semantics and fully leverage the model capacity of LLMs for sequential recommendation. To this end, in this paper, we propose LC-Rec, a new approach to integrate Language and Collaborative semantics for improving LLMs in Recommender systems. Our approach is built in a generative manner, where the recommendation task is cast into a token generation task as well. To achieve this, the key point lies in the semantic integration between language and collaborative semantics, so that LLMs can make the item recommendations just like they generate normal text contents. Our approach has made two major contributions in the aforementioned two aspects. For item indexing, we propose a tree-structured vector quantization (VQ) method to index the items with discrete IDs (called item indices). These item indices are learned based on the text embeddings of items encoded by LLMs, enabling the learned IDs to capture the intrinsic similarity among items. However, original VQ methods are likely to assign the same IDs to multiple items, which should be avoided in recommender systems. To tackle this problem, we further design a uniform semantic mapping method to mitigate the potential conflicts in ID allocation. For alignment tuning, we design a series of specific tasks to finetune LLMs for achieving semantic integration. In addition to the sequential item prediction, we consider both explicit indexlanguage alignment and implicit recommendation-oriented alignment. Our fine-tuning tasks enforce LLMs to deeply integrate language and collaborative semantics, so as to achieve an effective adaptation to recommender systems. To evaluate our approach, we conduct extensive experiments on three real-world datasets. Our method achieves the best performance compared to a number of competitive baselines. Experimental results demonstrate that our approach can effectively align language and collaborative semantics via specially learned item indices, thereby significantly improving the recommendation performance. The contributions of this work can be summarized as follows: • We present LC-Rec, a LLM-based sequential recommendation model, by effectively integrating language and collaborative semantics. LC-Rec can fulfill the sequential 1Note that the words “align” and “alignment” mainly refer to the integration between language semantics and collaborative semantics, but not what it
1Note that the words “align” and “alignment” mainly refer to the integration between language semantics and collaborative semantics, but not what it means in human alignment [24] that instructs LLMs to follow human values or preferences.
1Note that the words “align” and “alignment” mainly refer to the integration between language semantics and collaborative semantics, but not what it means in human alignment [24] that instructs LLMs to follow human values or preferences.
recommendation task in an autoregressive generation way, without relying on candidate sets. • Our approach is built on a specially designed VQ method, which can capture item similarity and avoid ID conflicts in index allocation. Further, we propose a series of carefully designed tuning tasks for achieving effective semantic integration via item indices. • We implement our method based on LLaMA [25] with 7B parameters. Extensive experiments on three public datasets demonstrate the effectiveness of our approach in integrating collaborative semantics into LLMs. The proposed method LC-Rec achieves an average performance improvement of 25.5% in full ranking evaluations, compared to all baseline methods.
recommendation task in an autoregressive generation way, without relying on candidate sets. • Our approach is built on a specially designed VQ method, which can capture item similarity and avoid ID conflicts in index allocation. Further, we propose a series of carefully designed tuning tasks for achieving effective semantic integration via item indices. • We implement our method based on LLaMA [25] with 7B parameters. Extensive experiments on three public datasets demonstrate the effectiveness of our approach in integrating collaborative semantics into LLMs. The proposed method LC-Rec achieves an average performance improvement of 25.5% in full ranking evaluations, compared to all baseline methods.
# II. RELATED WORK
# A. Sequential Recommendation
Sequential recommendation aims to infer user preferences by analyzing historical interactions and predict the next item that would be suitable for that user [1], [2], [8]. Many early methods are frequently based on Markov Chains techniques [26], [27]. Recently, typical methods become to adopt various deep neural networks to model user historical behaviors represented by item ID sequences, including RNN [1], [3], CNN [4], [5], GNN [6], [7], and Transformer [2], [8]. However, these methods only capture the collaborative relationship between items from useritem interactions, while ignoring the additional information rich in the item content information (e.g., title, description, category). Therefore, several studies are devoted to utilizing additional information associated with items to enhance ID sequence modeling [9]–[11]. Furthermore, the inherent natural language characteristics of the item title and description have motivated numerous researchers to explore the utilization of pre-trained language models (PLMs) in recommender systems [12]–[15]. In this paper, we aim to combine LLMs and recommendation tasks in a more effective way, which is reached through the proposed new item indexing and alignment tuning methods.
# B. Large Language Models for Recommendation
Recently, large language models (LLMs) have gained significant popularity, with a wide range of applications spanning various domains of artificial intelligence [16], [28]–[30]. This is largely attributed to their superior capabilities in language semantic understanding and generation. In the context of RS, researchers have been working on adapting LLMs for RS to improve recommendation performance. A common approach is to represent user behaviors as text sequences (e.g., by concatenating the titles of historical items), and then design prompts to guide LLMs to perform the recommendation task [31]–[33]. However, a major challenge remains: there is a large gap between the language semantics modeled by LLMs and the collaborative semantics implied by recommender systems, which cannot be bridged by simple prompt design alone. To address this problem, existing efforts can be categorized into two main approaches. The first approach is to fine-tune the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5ccf/5ccfb392-1c6b-4940-80a0-4e25f2c58c17.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 1: The overall framework of our LC-Rec. We enhance language learning models (LLMs) by integrating language an collaborative semantics based on item indexing and alignment tuning, thereby adapting LLMs to recommender systems.</div>
LLMs with text-based user behavior sequences [19]–[21], [34]. However, these methods cannot guarantee the generation of indomain items. Due to the constraints of limited context window size, these methods can only rank on a given candidate set, and can hardly be applied in a full ranking scenario. The second approach maintains the use of item IDs or introduces unique item indexing mechanisms. Given pure item ID sequences, models are trained to directly generate target item IDs for recommendation [22], [23]. Although collaborative semantics between item indices are established, the language semantics modeled by LLMs and these item indices are not well aligned. Additionally, we are aware of some concurrent works [35]– [37], which also recognize the issue of the large semantic gap between recommendation tasks and natural language tasks. For instance, TransRec [35] employs multi-facet identifiers, combining ID, title, and attributes to balance item distinctiveness and semantics. CoLLM [36] incorporates collaborative semantics into LLMs by using representations of an external collaborative model as part of the input. CLLM4Rec [37] learns dual user/item embeddings based on recommendation task and content (e.g., reviews) generation task, respectively. A mutuallyregularization loss is introduced for interaction between these two kinds of embeddings. Only the recommendation-task embeddings are for the final recommendation. Contrasting these methodologies, our focus is to develop a deep and unified integration of language and collaborative semantics within LLMs through carefully crafted item indices. In particular, our approach uses a tree-structured vector quantization method to construct item indices. This method allows for better semantic integration by adding a small number (usually ∼1,000) of additional tokens to LLMs. Furthermore, we introduce a series of semantic alignment tasks to fine-tune LLMs, aiming to achieve unified semantic integration in a
practical recommendation setting (e.g., full ranking).
practical recommendation setting (e.g., full ranking). III. METHODOLOGY
In this section, we present the proposed LLM-based recommendation model LC-Rec, which integrates Language and Collaborative semantics for improving LLMs in Recommender systems.
# A. Overview of the Approach
As we discussed in Section I, there exists a large gap between the language semantics modeled by LLMs and collaborative semantics implied by recommender systems, which limits the capacities of LLMs in recommender systems. To effectively bridge this gap, we consider enhancing the semantic integration in two major aspects. • For item indexing (Section III-B), we represent an item with several learned discrete IDs via vector quantization based on text embeddings by LLMs, and further propose a uniform semantic mapping method to mitigate the potential conflicts in index assignment. In this way, the learned item indices can capture similarities between the textual semantics of item information, and provide a unique indexing representation for a specific item. • For alignment tuning (Section III-C), we design a series of specific tuning tasks that enhance the integration between language semantics and collaborative semantics, not limited to the target recommendation task. Our approach can effectively integrate the collaborative semantics into LLMs, and sufficiently leverage the powerful model capacity of LLMs for recommendation tasks. The overall framework of the proposed approach LC-Rec is shown in Figure 1. Next, we will present the details of our method.
The overall framework of the proposed approach LC-Rec is shown in Figure 1. Next, we will present the details of our method.
B. Learning Item Indices for Semantic Integration
To extend the capacities of LLMs for recommendation, a fundamental problem is how to represent an item with index IDs (called item indices) and integrate these item indices into LLMs. We don’t adopt the original item ID (resulting in a very large vocabulary), but instead employ vector quantization techniques to represent an item with a small number of discrete indices. These indices are constructed by leveraging relevant item information (e.g., item text representations), and the token embeddings associated with these discrete indices can be further optimized to fit the recommendation task (Section III-C). In this part, we present the approach for learning item indices for subsequent semantic integration. The approach consists of two major steps: it first conducts vector quantization based on text embeddings of items, so that the original representations of item indices can capture latent textual semantic correlations between items; then, it proposes a uniform semantic mapping to mitigate the potential conflicts in item index assignment. Next, we introduce the two parts in detail. 1) Vector Quantization Indexing: In recommender systems, it is common to associate each item with a single unique ID (called vanilla ID). However, it would directly introduce a large vocabulary of item IDs when dealing with a great number of items (i.e., a large item set). Further, such an approach is easy to suffer from the OOV issue when adapting to new items (e.g., cold-start items). To address this issue, we borrow the idea of existing studies [15], [23], [38] to learn indices associated with latent semantics for items. Specifically, each item is represented by a composition of discrete indices corresponding to its own latent semantic, and each discrete index can be shared by multiple items. The basic idea is that similar items tend to be assigned with a portion of common semantic indices, such that each unique semantic index can be aligned to some kind of latent semantics. To derive these semantic indices, we first employ LLMs (e.g., LLaMA) to encode the attached text information for an item, and obtain the text embeddings as the initial item representation. Further, we propose to use a Vector Quantization (VQ) approach to create discrete indices based on item embeddings. Specifically, we take the item embeddings encoded by LLMs as input, and then train a Residual-Quantized Variational AutoEncoder (RQ-VAE) for generating item indices. RQVAE [39] is a multi-level vector quantizer, which recursively quantized the residual vectors from coarse to fine to generate a set of codewords (i.e., item indices). For an item embedding e, RQ-VAE first encodes it into a latent representation z. At each level h, we have a codebook Ch = {vh k}K k=1, where each codebook vector vh k is a learnable cluster center. Then the residual quantization process can be expressed as:
(1) (2)
(1)
(2)
(2)
where ci is the i-th codeword of the item indices and ri is the residual vector in the i-th RQ level, and we set r1 = z.
Algorithm 1 RQ with Uniform Semantic Mapping
Input: Batch item representations B = {zn}|B|
n=1; H-level
codebooks {Ch}H
h=1.
Output: Item indices {[cn
1, cn
2, ..., cn
H]}|B|
n=1; Quantified repre-
sentations {ˆzn}|B|
n=1.
1: Let initial residual vectors rn
1 = zn, ∀zn ∈B
2: for i = 1 to H do
3:
if i < H then
4:
Solve {cn
i }|B|
n=1 according to Eqn. (1)
5:
else
6:
Solve {cn
H}|B|
n=1 according to Eqn. (6) via Sinkhorn-
Knopp algorithm
7:
end if
8:
Obtain {rn
i+1}|B|
n=1 according to Eqn. (2)
9: end for
10: for all zn ∈B do
11:
Calculate quantified representations by ˆzn = �H
i=1 vi
cn
i
12: end for
13: return {[cn
1, cn
2, ..., cn
H]}|B|
n=1 and {ˆzn}|B|
n=1
When we have H-level codebooks, the quantization representation of z can be obtained according to ˆz = �H i=1 vi ci. Then ˆz will be used as decoder input to reconstruct the item embedding e. The overall loss function is as follows:
(3) (4) (5)
(3)
(5)
where ˆe is the output of the decoder, sg[] represents the stopgradient operator, and β is a loss coefficient, usually set to 0.25. The overall loss is divided into two parts, LRECON is the reconstruction loss, and LRQ is the RQ loss used to minimize the distance between codebook vectors and residual vectors. Compared with traditional VQ approaches, RQ offers the advantage of achieving a larger expression space with a smaller codebook size [39], [40]. Besides, its coarse-to-fine quantification method results in a tree-structured item index, which is beneficial for autoregressive generation. In fact, the RQ approach has demonstrated its effectiveness across various autoregressive generation tasks, such as autoregressive image generation [40] and generative recommendation [38]. Instead of simply employing VQ for item indexing [15], [38], we consider two key improvements for deriving meaningful item indices. First, there should be no conflicts in item indices, which is a common issue with VQ but should not occur in recommender systems. Second, the established semantic spaces of item indices should be aligned with the semantics of LLMs, in order to better leverage the powerful model capacity of LLMs for recommendation. We next introduce the two major improvements in our approach. 2) Conflict Mitigation via Uniform Semantic Mapping: Since we adopt the tree structure for learning item indices, it
might lead to index conflicts among items within the same leaf node. To address this issue, existing solutions [23], [38] typically add an additional layer to the index tree and assign a distinct supplementary index ID to each item in a node with conflicts. However, this approach introduces semantically irrelevant distributions in the tree’s final layer. Additionally, these newly integrated IDs might also affect the original item representations. Considering these issues, we propose a new conflict mitigation method to avoid the clustering of multiple items within the same leaf node. Our objective is to ensure that item semantics are uniformly distributed across different codebook embeddings at the last index level. To achieve this, we introduce a uniform distribution constraint to the original formulation:
(6)
where B is a batch of residual vectors in the last index level. Following [41]–[44], by considering ||rH −vH k ||2 2 as the cost of semantic mapping, this problem can be viewed as an optimal transmission problem. In this setting, q(cH = k|rH) represents the transmission or mapping scheme that needs to be solved. In our implementation, we solve this equation by SinkhornKnopp algorithm [45]. The overall process of RQ with uniform semantic mapping is shown in Algorithm 1. By optimizing the loss in Eqn. (5), we can obtain a trained encoder and multi-level codebooks. During the construction of item indices, we first generate indices based on Eqn. (1). After that, for each group of conflicting items, the codewords of these items at the last level will be redistributed uniformly based on Eqn. (6). Such a two-stage process can also improve efficiency and reduce unnecessary random noise introduced by batching items [41].
C. Aligning Language and Collaborative Semantics in LLM
After learning item indices, a straightforward approach is to integrate these index IDs into the LLM vocabulary, so that LLM can fulfill the recommendation task in a generative way that gradually omits the indices of items. However, these item indices are essentially OOV tokens for LLMs, and it is necessary to conduct the alignment between language and collaborative semantics. For this purpose, we design a series of semantic alignment tasks to assign language and collaborative semantics for tuning LLMs, including the primary objective of sequential item prediction, explicit index-language alignment (identifying the corresponding item via their indices), and implicit recommendation-oriented alignment (enhancing comprehension of the language and collaborative semantics). As discussed below, these tuning tasks are very effective
# in enhancing the alignment between language models and collaborative semantics.
1) Sequential Item Prediction: Since our approach is built in a LLM-based generative manner, we consider employing sequential item prediction as the major tuning objective. Specifically, we construct personalized recommendation instructions based on the user’s current historical interactions. Then, LLMs are prompted by the instructions and the interaction history, to predict the next item that the target user is likely to interact with. Here, the user’s historical interactions are described and identified as an index sequence of interacted items arranged in chronological order. A sample instance is given as follows:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f148/f1481f7d-8961-4d25-97a0-9a6a972463e4.png" style="width: 50%;"></div>
However, due to the large semantic gap, simply fine-tuning LLMs with the above target task, it is difficult to sufficiently integrate language and collaborative semantics in LLMs. 2) Explicit Index-Language Alignment: Although our item indices are constructed based on titles and descriptions of items, they rely solely on shared prefix codewords to establish a weak correlation among items with similar language semantics. To further endow item indices with language semantics, we propose two explicit index-language alignment tasks for tuning LLMs. On the one hand, the LLM should be capable of accurately identifying the item indices based on the associated title or description. On the other hand, it is expected that LLM can naturally capture relevant item information from its indices. Considering the two aspects, we first instruct the LLM to generate the corresponding item indices according to the item’s title/description or a combination of both. Then, we instruct the LLM to recover the item information based on its indices. We present two instruction samples to illustrate the two alignment tuning tasks in the following.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7f3e/7f3e3fc0-0ddd-4c06-9fdf-0980102eff52.png" style="width: 50%;"></div>
Instruction: Please tell me what item <a_66><b_197><c_236><d_223> is called, along with a brief description of it. Response: Item Title: Pok´emon Moon - Nintendo 3DS Item Description: Pok´emon Moon will launch in the ... become a Pok´emon Champion!
Actually, such a mutual prediction method is essentially similar to what has been in cross-modal semantic alignment. It can be an analogy with the mutual association or mapping between images and text [29], or the conversion between speech and text [46]. By instruction tuning the LLM with these alignment instructions, item indices can be seamlessly integrated into the semantic space spanned by the LLM. 3) Implicit Recommendation-oriented Alignment: After being tuned with the above alignment tasks, LLMs can acquire basic knowledge of collaborative semantics. In this part, we further consider enhancing the model capacity via recommendation-oriented alignment tasks, so that LLM can better leverage both language and collaborative semantics to fulfill various recommendation tasks in a more accurate way. Specifically, we design the following three alignment tasks: a) Asymmetric item prediction: As discussed in Section III-C1, for sequential item prediction, both the interaction history (condition) and the target item (target) are formatted in the representation of item indices. We call this tuning task symmetric since both the condition and target for prediction are based on item indices. To further enhance the semantic alignment, we increase the prediction difficulty by changing the representations of condition and target, so as to derive different combinations of semantic representations for items. Specially, we consider the following three representation methods: (1) replacing the indices of target item with the item title, instructing the LLM to generate the item title directly based on the item index sequence; (2) replacing the indices of target item with the item description, instructing the LLM to generate the item features and attributes expected by the user; (3) representing the user interaction history as a text sequence of item titles instead of an index sequence, instructing the LLM to infer user preferences based on the title sequence. The sample instructions for the three scenarios are as follows:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/32a6/32a6e9b4-4d31-4741-a2e3-c13f5e17a5fa.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/cc4a/cc4a4f48-90e1-4276-a0e8-fb6eca3e592e.png" style="width: 50%;"></div>
To make a comparison, the item prediction task in Section III-C1 involves mapping an index sequence to the target indices, while the tuning tasks in Section III-C2 explicitly align item indices with their corresponding language information. These asymmetric tasks are more difficult, which enforces LLMs to unify item indices, language semantics, and collaborative semantics for fulfilling the recommendation tasks. As will be shown in the experiment part (Section IV-C), these tuning tasks are useful in adapting LLMs to recommender systems. b) Item prediction based on user intention: Drawing inspiration from [19], a recommender system in real life should possess the ability to understand the actual intentions of users and provide high-quality recommendations accordingly. This leads to a task similar to item retrieval. Referring to the approach in [19], as reviews offer valuable evidence regarding users’ personal tastes and motivations for making a specific interaction, we consider extracting intentions from the related reviews of the target item. To accomplish this, we utilize GPT3.5 to process these reviews and extract user intentions. As for instructions, we mainly design two types of tasks: the former queries an item recommendation directly based on instant user intention, and the latter provides the user’s interaction history for a personalized recommendation.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/62d0/62d033ea-b2d6-4735-9fcf-5cb59d4bfc32.png" style="width: 50%;"></div>
Instruction: As a recommender system, you are assisting a user who has recently interacted with the following items: <a_64><b_159><c_1><d_89>,...,<a_119><b_98><c_162><d_155>. The user expresses a desire to obtain another item with the following characteristics: “The console offers 500GB of storage, ... 4K HDR gaming”. Please recommend an item that meets these criteria. Response: <a_227><b_206><c_156><d_156> (PlayStation 4 500GB Console)
c) Personalized preference inference: Intuitively, a user’s interaction history can implicitly reflect his or her personal preferences, but explicit preferences are generally absent from the dataset. Thus, we employ GPT-3.5 to infer the user’s explicit preferences from items the user has interacted with in the past. Unlike prior work [19], we infer user preferences based on the index sequence of historical items rather than the title sequence. This task requires the index sequence to act as an effective substitute for the title sequence, enabling the LLM to understand the joint language and collaborative semantics within the index sequence and accurately extract user preferences. The instruction could be:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a46d/a46d4f85-6055-4fbf-871c-4b5827f8d17f.png" style="width: 50%;"></div>
In this work, we mainly focus on the setting of sequential item prediction, i.e., sequential recommendation, while our approach can be easily extended to other tuning tasks in recommender systems, e.g., bundle prediction and explanation generation. Actually, our index mechanism can support various instruction tuning tasks as in standard language models [47], since these indices are endowed with both language and collaborative semantics, acting as common tokens for LLMs.
# D. Training and Inference
In this section, we discuss how to optimize our base LLM using the aforementioned tuning tasks and how to utilize it to fulfill the recommendation task over the entire item set. 1) Training: In this paper, we strive to leverage the semantic understanding and generation capabilities of LLMs to facilitate sequential recommendation. To this end, we employ LLaMA [25] as our backbone model and then optimize it via
instruction tuning. The tuning tasks mentioned above can be conveniently formatted as conditional language generation tasks in a sequence-to-sequence manner. We optimize the negative log-likelihood of the generation target as follows:
(7)
where ⟨I, Y ⟩represents a pair of instruction and target response in the batch data, Yj is the j-th token of Y and Y<j denotes the tokens before Yj. For each task, we designed multiple instruction templates to enhance the instruction diversity. However, during a training epoch, each data is only combined with one sampled instruction template, which is different from those in prior approaches [19], [22], [23]. This strategy is based on our observation that LLaMA, as an LLM with over 7B parameters, achieves better results by examining specific data only a few times [48]. In contrast, repeating data may lead to overfitting. 2) Inference: Our objective is to generate the top n items from the entire item set which most match the preference of a given user during inference. To accomplish this, the decoder module performs a beam search across the index tokens. Here, we use the index structure built in Section III-B for item decoding. Additionally, when calculating logits, the probabilities of tokens that may result in illegal item indices will be assigned as 0 to ensure generation quality. Given an input sequence, the inference time is mainly consumed in the multi-layer self-attention calculation. The time complexity of a forward process in the vanilla Transformer is O(N 2dL), where L is the number of model layers, N is the sequence length, and d is the dimension of hidden states. Overall, in order to autoregressively generate complete target item indices, the time consumption is O(HN 2dL), where H is the number of index levels (usually a small value like H = 4). But in fact, the attention key and value tensors of each layer can be cached for subsequent decoding, called KV Cache [49]. After applying KV Cache, the time complexity can be optimized to O(N 2dL + HNdL). In addition to the inference speed, the memory efficiency can also be improved through various technologies such as model quantization [50] and PagedAttention [51].
# E. Discussion
In this part, we compare the proposed LC-Rec with existing language model based methods for recommendation to highlight the contributions of our approach. Text-based methods such as TALLRec [21] and InstructRec [19] typically represent user historical behavior as a sequence of item titles, thereby formatting the sequential recommendation task into a natural language question or instruction, which can be easily adapted to the LLM. However, these methods are not suitable for the full ranking setting, since they have difficulty in understanding and generating the item information over the entire item set. They either consider a discrimination question that can be answered with
TABLE I: Comparison of our method with several related studies. “FR” denotes full ranking across the entire item set. “LS” denotes language semantics. “CS” denotes collaborative semantics. “ILC” denotes the integration of language and collaborative semantics in LLMs.
Methods
Scale
Backbone
FR
LS
CS
ILC
TIGER [38]
N/A
N/A
�
�
�
�
P5 [22], [23]
220M
T5
�
�
�
�
InstructRec [19]
3B
Flan-T5
�
�
�
�
TALLRec [21]
7B
LLaMA
�
�
�
�
LC-Rec
7B
LLaMA
�
�
�
�
“Yes/No” [21] or perform reranking based on a small number of candidate items [19]. Furthermore, this approach mainly relies on language semantics to tackle the recommendation tasks, which neglects the collaborative semantic information in recommender systems. Index-based methods such as TIGER [38] and P5 [22], [23] (specifically, we focus on the sequential recommendation task of P5) directly convert the traditional item ID-based sequential recommendation into a generative paradigm. TIGER is not based on language models. Instead, it trains an encoder-decoder Transformer model from scratch to predict the next item given an input item sequence, where each item is identified by multiple discrete IDs. P5 adapts recommendation tasks into the text-to-text format to enable unified modeling. However, within this framework, sequential recommendation is still organized as a mapping process from item ID sequence to target item ID, which only establishes collaborative semantics between item IDs and is independent of language semantics in LLMs. We are also aware of several concurrent studies [35]–[37] that aim to adapt LLMs for recommender systems. They mainly consider enhancing the semantics of items from different aspects, including setting multi-type identifiers (e.g., ID, title, and attributes) [35], incorporating external collaborative representations [36], and learning dual user/item embeddings [37]. As a comparison, our work focuses on the integration of language and collaborative semantics for enhancing the recommendation capacity of LLMs. Specifically, we adopt a new item indexing mechanism that ensures index uniqueness and effectively reduces the vocabulary size. Moreover, we further design various alignment tasks for enhancing the semantic integration. Based on these improvements, our approach can effectively integrate collaborative semantics into LLMs, and further leverage the enhanced capacity of LLMs for fulfilling the recommendation tasks. The comparison of our method with several related studies is shown in Table I.
# IV. EXPERIMENT
In this section, we first set up the experiments and then present the results as well as analysises of our proposed approach.
TABLE II: Statistics of the preprocessed datasets. “Avg. len” represents the average length of item sequences.
<div style="text-align: center;">TABLE II: Statistics of the preprocessed datasets. “Avg. len” represents the average length of item sequences.</div>
Datasets
#Users
#Items
#Interactions
Sparsity
Avg. len
Instruments
24,773
9,923
206,153
99.92%
8.32
Arts
45,142
20,957
390,832
99.96%
8.66
Games
50,547
16,860
452,989
99.95%
8.96
# A. Experiment Setup
1) Dataset: We evaluated the proposed approach on three subsets of Amazon review data [52], including “Musical Instruments”, “Arts, Crafts and Sewing”, and “Video Games”. All three datasets contain user review data from May 1996 to October 2018. Each item in the dataset is associated with a title and a description. Following previous work [12], we first filter out unpopular users and items with less than five interactions. Then, we create user behavior sequences based on the chronological order. The maximum item sequence length is uniformly set to 20 to meet all baseline requirements. The statistics of our preprocessed datasets are shown in Table II. 2) Baseline Models: We adopt the following representative sequential recommendation models as baselines for comparison with our LC-Rec: • Caser [4] is a CNN-based method that models user behaviors by applying horizontal and vertical convolutional filters. • HGN [53] utilizes hierarchical gating networks to capture both long-term and short-term user interests from historical behaviors. • GRU4Rec [1] is an RNN-based sequential recommendation model that utilizes GRU to encode the item sequence. • BERT4Rec [8] adopts a bidirectional Transformer model and combines it with a mask prediction task for the modeling of item sequences. • SASRec [2] exploits a unidirectional Transformer-based neural network to model the item sequences and predict the next item. • FMLP-Rec [54] proposes an all-MLP model with learnable filters, which ensures efficiency and reduces noise signals. • FDSA [9] focuses on the transformation patterns between item features, modeling both item-level and feature-level sequences separately through self-attention networks. • S3-Rec [10] utilizes mutual information maximization to pre-train a self-supervised sequential recommendation model, learning the correlation between items and attributes. • P5-CID [22], [23] organizes multiple recommendation tasks in a text-to-text format and models different tasks uniformly using the T5 model. Subsequently, the author team explores the construction of item indices for sequential recommendation, including sequential indexing and collaborative indexing. Here, we employ P5 with collaborative indexing as the baseline and implement it
1) Dataset: We evaluated the proposed approach on three ubsets of Amazon review data [52], including “Musical nstruments”, “Arts, Crafts and Sewing”, and “Video Games”. All three datasets contain user review data from May 1996 o October 2018. Each item in the dataset is associated with  title and a description. Following previous work [12], we rst filter out unpopular users and items with less than five nteractions. Then, we create user behavior sequences based on he chronological order. The maximum item sequence length s uniformly set to 20 to meet all baseline requirements. The tatistics of our preprocessed datasets are shown in Table II. 2) Baseline Models: We adopt the following representative equential recommendation models as baselines for comparison with our LC-Rec: • Caser [4] is a CNN-based method that models user behaviors by applying horizontal and vertical convolutional filters. • HGN [53] utilizes hierarchical gating networks to capture both long-term and short-term user interests from historical behaviors. • GRU4Rec [1] is an RNN-based sequential recommendation model that utilizes GRU to encode the item sequence. • BERT4Rec [8] adopts a bidirectional Transformer model and combines it with a mask prediction task for the modeling of item sequences. • SASRec [2] exploits a unidirectional Transformer-based neural network to model the item sequences and predict the next item. • FMLP-Rec [54] proposes an all-MLP model with learnable filters, which ensures efficiency and reduces noise signals. • FDSA [9] focuses on the transformation patterns between item features, modeling both item-level and feature-level sequences separately through self-attention networks. • S3-Rec [10] utilizes mutual information maximization to pre-train a self-supervised sequential recommendation model, learning the correlation between items and attributes. • P5-CID [22], [23] organizes multiple recommendation tasks in a text-to-text format and models different tasks uniformly using the T5 model. Subsequently, the author team explores the construction of item indices for sequential recommendation, including sequential indexing and collaborative indexing. Here, we employ P5 with collaborative indexing as the baseline and implement it
• Caser [4] is a CNN-based method that models user behaviors by applying horizontal and vertical convolutional filters. • HGN [53] utilizes hierarchical gating networks to capture both long-term and short-term user interests from historical behaviors. • GRU4Rec [1] is an RNN-based sequential recommendation model that utilizes GRU to encode the item sequence. • BERT4Rec [8] adopts a bidirectional Transformer model and combines it with a mask prediction task for the modeling of item sequences. • SASRec [2] exploits a unidirectional Transformer-based neural network to model the item sequences and predict the next item. • FMLP-Rec [54] proposes an all-MLP model with learnable filters, which ensures efficiency and reduces noise signals. • FDSA [9] focuses on the transformation patterns between item features, modeling both item-level and feature-level sequences separately through self-attention networks. • S3-Rec [10] utilizes mutual information maximization to pre-train a self-supervised sequential recommendation model, learning the correlation between items and attributes. • P5-CID [22], [23] organizes multiple recommendation tasks in a text-to-text format and models different tasks uniformly using the T5 model. Subsequently, the author team explores the construction of item indices for sequential recommendation, including sequential indexing and collaborative indexing. Here, we employ P5 with collaborative indexing as the baseline and implement it
Dataset
Metrics
Caser
HGN
GRU4Rec BERT4Rec SASRec FMLP-Rec
FDSA
S3-Rec
P5-CID
TIGER
LC-Rec
Improv.
Instruments
HR@1
0.0149
0.0523
0.0571
0.0435
0.0503
0.0480
0.0520
0.0367
0.0587
0.0608
0.0706
+16.12%
HR@5
0.0543
0.0813
0.0821
0.0671
0.0751
0.0786
0.0834
0.0863
0.0827
0.0863
0.1002
+16.11%
HR@10
0.0710
0.1048
0.1031
0.0822
0.0947
0.0988
0.1046
0.1136
0.1016
0.1064
0.1220
+7.39%
NDCG@5
0.0355
0.0668
0.0698
0.0560
0.0627
0.0638
0.0681
0.0626
0.0708
0.0738
0.0856
+15.99%
NDCG@10
0.0409
0.0744
0.0765
0.0608
0.0690
0.0704
0.0750
0.0714
0.0768
0.0803
0.0926
+15.32%
Arts
HR@1
0.0138
0.0300
0.0421
0.0337
0.0225
0.0310
0.0451
0.0245
0.0485
0.0465
0.0634
+30.72%
HR@5
0.0379
0.0622
0.0749
0.0559
0.0757
0.0757
0.0734
0.0767
0.0724
0.0788
0.1011
+28.30%
HR@10
0.0541
0.0875
0.0964
0.0713
0.1016
0.1046
0.0933
0.1051
0.0902
0.1012
0.1266
+20.46%
NDCG@5
0.0262
0.0462
0.0590
0.0451
0.0508
0.0541
0.0595
0.0521
0.0607
0.0631
0.0828
+31.22%
NDCG@10
0.0313
0.0544
0.0659
0.0500
0.0592
0.0634
0.0660
0.0612
0.0664
0.0703
0.0906
+28.88%
Games
HR@1
0.0085
0.0154
0.0176
0.0136
0.0145
0.0152
0.0161
0.0119
0.0177
0.0188
0.0317
+68.62%
HR@5
0.0367
0.0517
0.0586
0.0482
0.0581
0.0571
0.0644
0.0606
0.0506
0.0599
0.0800
+24.22%
HR@10
0.0617
0.0856
0.0964
0.0763
0.0940
0.0930
0.1041
0.1002
0.0803
0.0939
0.1174
+12.78%
NDCG@5
0.0227
0.0333
0.0381
0.0311
0.0365
0.0361
0.0404
0.0364
0.0342
0.0392
0.0560
+38.61%
NDCG@10
0.0307
0.0442
0.0502
0.0401
0.0481
0.0476
0.0531
0.0491
0.0437
0.0501
0.0681
+28.25%
according to the code2 provided by the authors. • TIGER [38] adopts the generative retrieval paradigm for sequential recommendation and introduces a semantic ID to uniquely identify items. Due to the official code not being released by the authors, here we implement it ourselves by Transformers3 following the implementation details provided in the paper. 3) Evaluation Settings: To evaluate the performance of sequential recommendation, we adopt two widely used metrics, top-K Hit Ratio (HR) and top-K Normalized Discounted Cumulative Gain (NDCG). In this paper, we set K as 1, 5, and 10. Following previous works [2], [10], [54], we employ the leave-one-out strategy for evaluation. Concretely, for each user behavior sequence, the most recent item is used as the test data, the second most recent item is used as the validation data, and the remaining interaction records are used for training. We perform full ranking evaluation over the entire item set instead of sample-based evaluation. For the generative methods based on beam search, the beam size is uniformly set to 20. 4) Implementation Details: To construct item indices, we utilize LLaMA to encode the title and description of the item as its embedding and use mean pooling to aggregate multiple token representations. The level of item indices is set to 4, with each level consisting of 256 codebook vectors, and each vector has a dimension of 32. Both the encoder and decoder of RQ-VAE are implemented as Multi-Layer Perceptrons (MLPs) with ReLU activation functions. The model is optimized using the AdamW optimizer, employing a learning rate of 0.001 and a batch size of 1024. For LLM fine-tuning, we implemented LC-Rec based on LLaMA through Transformers3 and accelerated training by DeepSpeed4. All tokens related to item indices are appended to
2https://github.com/Wenyueh/LLM-RecSys-ID/ 3https://github.com/huggingface/transformers/ 4https://github.com/microsoft/DeepSpeed/
the tokenizer as out-of-vocabulary (OOV) tokens. We employ the AdamW optimizer for model optimization, setting the learning rate to 5e-5 and weight decay to 0.01. During the finetuning, a cosine scheduler with warmup is utilized to adjust the learning rate. With the application of data parallelism and gradient accumulation, the overall batch size amounts to 128. We conduct training for 4 epochs on each dataset. To prevent overfitting, we ensure that during each epoch, a single data is combined with one sampled instruction template and appears only once.
# B. Overall Performance
We compare the proposed approach with the different baseline models on three datasets, and the overall results are shown in Table III. Based on these results, we can find: For the baseline methods, the sequential recommendation methods that incorporate item content information (i.e., FDSA and S3-Rec) perform better than traditional sequential recommendation methods that solely rely on ID and collaborative relationships (i.e., Caser, HGN, GRU4Rec, BERT4Re, SASRec, FMLP-Rec) on several datasets. This indicates that item content information introduced as additional information can effectively improve recommendation performance. As for P5-CID and TIGER, they demonstrate competitive performance across the first two datasets, particularly excelling in HR@1 and the metrics related to item ranking (i.e., NDCG). In terms of the Games dataset, they have an improvement compared to the ID-only model, but no significant improvement compared to the methods that already include auxiliary content information. One possible reason for this is the difference in the effects of content information and the difficulty of modeling it in different data and scenarios. Our proposed LC-Rec consistently maintains the best performance on three datasets and shows significant improvements compared to the baseline methods. This superior performance
Methods
Arts
Games
HR@1
HR@5
HR@10
NDCG@5
NDCG@10
HR@1
HR@5
HR@10
NDCG@5
NDCG@10
SEQ
0.0561
0.0909
0.1133
0.0740
0.0812
0.0243
0.0626
0.0930
0.0437
0.0535
+ MUT
0.0593
0.0926
0.1141
0.0765
0.0832
0.0275
0.0703
0.1038
0.0491
0.0598
+ ASY
0.0602
0.0945
0.1172
0.0776
0.0848
0.0281
0.0725
0.1073
0.0506
0.0615
+ ITE
0.0638
0.0996
0.1232
0.0813
0.0889
0.0294
0.0770
0.1125
0.0534
0.0648
+ PER
0.0634
0.1011
0.1266
0.0828
0.0906
0.0317
0.0800
0.1174
0.0560
0.0681
can be attributed to two factors: (1) The item indexing mechanism via vector quantization combined with uniform semantic mapping, which captures similarities between items and ensures a semantically lossless generation process at the last index level. (2) The effective integration of collaborative semantics into LLMs, which results in a seamless fusion of language semantics and collaborative semantics. By employing these strategies, our approach is able to leverage the powerful modeling capabilities of LLMs, thereby achieving significant improvements in the recommendation task.
# C. Ablation Study
a) Various semantic alignment tasks: Our proposed LCRec consists of various semantic alignment tasks, including (1) SEQ: the sequential item prediction task introduced in Section III-C1 as our primary objective, (2) MUT: the mutual prediction task for explicit index-language alignment in Section III-C2, (3) ASY: the asymmetric item prediction task in Section III-C3a, (4) ITE: the item prediction based on user intention in Section III-C3b, (5) PER: the personalized preference inference task in Section III-C3c. The latter three tasks all belong to the implicit recommendation-oriented alignment introduced in Section III-C3. To validate the effectiveness of each component, we conduct the ablation study on Arts and Games dataset to analyze the contribution of each part. The results, as shown in Table IV, indicate that the gradual incorporation of multiple semantic alignment tasks into the sequential recommendation, which involves only collaborative semantics, can significantly improve performance. All these instruction tuning tasks in LC-Rec are shown beneficial for enhancing sequential recommendation, and there is potential for further improvements by adding more semantic alignment tasks. b) Other item indexing methods: In addition to the semantic alignment tasks, we also examine the proposed item indexing method, by comparing it to another three indexing methods. (1) Vanilla ID is the same as the traditional recommendation model, using a single and unique ID for each item. (2) Random Indices uses multi-level indexing, but the indices at each level are derived from random sampling and are not semantically related. (3) LC-Rec w/o USM removes the uniform semantic mapping in our indexing method and assigns a distinct supplementary index ID to each conflicting item. As shown in Figure 2, our approach (red dotted line, LCRec) outperforms all three base indexing methods, indicating
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0fd8/0fd844e1-b9d0-487b-a312-e835878e32aa.png" style="width: 50%;"></div>
Fig. 2: The performance of our framework on three indexing methods, we report HR@5 and NDCG@5 on Games dataset. “SEQ” denotes fine-tuning only with the sequential item prediction task. “w/ ALIGN” denotes combining with our semantic alignment tasks.
the effectiveness of the proposed item indexing method (Section III-B). In addition, if we apply the proposed semantic alignment tasks (“w/ ALIGN” in Figure 2) to these three base indexing methods, their performance can be boosted by a large margin, especially for methods also based on multi-level indexing (e.g., Random Indices and LC-Rec w/o USM), outperforming all baseline methods. The results also demonstrate that the proposed alignment tasks can improve recommendation performance in an indexing-agnostic way.
# D. Further Analysis
a) Item prediction based on user intention: We further evaluate the ability of LC-Rec to understand the semantics contained in the item index. The evaluation is performed through a user intention-based item prediction task on Games dataset, as described in Section III-C3b. Following the widely used setups in sequential recommendation task, the most recent record in each user behavior sequence is used for testing. User intentions are used as the query and are generated by GPT3.5 based on review data. We employ DSSM [55], a widely validated retrieval model, as our baseline. It adopts a twotower architecture to search for relevant items based on textual similarity between a given user query and item titles. In our implementation, BERT [56] is used to encode queries and item titles. As shown in Figure 3, our approach exhibits a significant performance improvement compared to the baseline model. This improvement can be attributed to the integration of language and collaborative semantics in the LLM through item indices. Additionally, “LC-Rec (Zero-Shot)” represents
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b60c/b60cbd9f-ccae-4e1c-bf4c-442fec4a78ca.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 3: Performance of item prediction based on user intention.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/31f0/31f09337-75ae-4371-939f-2e34ffdcd5bf.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Fine-tuning only with the target sequential item prediction task. (b) LC-Rec that consists of a series of alignment tasks.</div>
Fig. 4: 2D visualization of LLM token embeddings via PCA.
the LC-Rec variant that is not trained in the item prediction task regarding user intention. Interestingly, we can observe that basic language and collaborative semantic alignment can still link item indices to user intentions to some extent, even without prior training on the target task. b) Embedding visualization analysis: To further investigate the effects of our proposed framework in terms of semantic integration, we delve deeper to study the connection between item indices and the original semantic space of the LLM. Following previous work [57], [58], we employ Principal Component Analysis (PCA) to visualize the embeddings corresponding to different tokens. As shown in Figure 4, “Item Indices” represents index tokens added to the vocabulary, while “Item Texts” represents tokens related to item texts (e.g., title and description). According to 2D visualization results, it is evident that a lack of semantic integration leads to incompatibility between item index tokens and the LLM semantic space. In contrast, our framework is capable of incorporating item indices into the LLM and aligning language and collaborative semantics. c) Performance on semantically similar negative items: In order to understand why integrating language and collaborative semantics can improve LLMs in recommendation tasks, we further evaluate our LC-Rec with a ranking task with different negative samples that are similar to ground truth in either language or collaborative semantics. Specifically, we first select two types of semantically similar negative items: (1) Items with similar language semantics, which are selected based on the cosine similarity between item text embeddings. (2) Items with similar collaborative semantics, which are selected based on the cosine similarity between item embeddings from the
<div style="text-align: center;">TABLE V: Performance on semantically similar negative items</div>
Methods
Language Neg.
Collaborative Neg.
Random Neg.
SASRec
73.52
52.25
89.78
LLaMA
56.67
51.23
61.14
ChatGPT
60.94
51.30
66.66
LC-Rec (Title)
67.74
56.72
84.64
LC-Rec
75.73
60.01
90.19
trained SASRec [2] model. Subsequently, we use the same test data as sequential recommendation task and utilize the model to choose between the ground-truth target item and the negative item with similar language/collaborative semantics. In addition, we use random negative items as a comparison benchmark and measure the performance by accuracy. We adopt SASRec, LLaMA without fine-tuning, and ChatGPT as the comparison methods. “LC-Rec (Title)” refers to our approach but makes recommendations based on item titles rather than indices. The results are shown in Table V. In the task of distinguishing items with similar language semantics, our method achieved the best performance, benefiting from the integration of collaborative semantics implied by recommender systems. Additionally, substituting item indices with titles for recommendations also yielded competitive results, which can be attributed to the implicit alignment between item indices and titles within our model. Another task, distinguishing items with similar collaborative semantics, is often considered more challenging. This is due to the fact that the item with similar collaborative semantics may also have language semantic relevance to the ground-truth target item. However, even for such a difficult task, our LC-Rec still shows better performance than the strong baselines, thanks to the unification of language and collaborative semantics. Furthermore, the nonfine-tuned LLaMA and ChatGPT perform sub-optimally in these challenging scenarios, demonstrating that utilizing LLMs directly for recommendation purposes is often inadequate due to the large gap between recommendation tasks and natural language tasks.
# E. Case Study
To intuitively explore the semantic information implicitly learned in the item indices, we present two types of illustrative cases in Figure 5. On the one hand, we analyze the hierarchical semantics in the multi-level item index. Specifically, we initially attempt to generate the item title using only the first index and gradually include more until all four indices are used. As shown in Figure 5(a), when relying solely on the first-level index, the generated content often fails to match the ground-truth item, but it already possesses some relevant semantic information. For example, in the first case, a single index can generate the keyword “Spider-Man”, whereas in the second case, a game belonging to the same categories (i.e., adventure) and similar platform (i.e., PlayStation) as the ground-truth item can be generated. As more indices are included, the generated content progressively converges towards the target title. Notably, at the second level, our LC-Rec is already capable of inferring
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a649/a649b4b5-92f4-431e-aed5-d1ed63ad5198.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Generate the item title based on different number of indices</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/eb8c/eb8c9fa4-4a8a-41ae-8d8b-310efb21e872.png" style="width: 50%;"></div>
Fig. 5: Case study about the semantics within item indices. For the cases in Figure 5(a), it can be observed that as the number of index increases, the generated content progressively converges towards the target title, and the semantic changes show a trend from coarse to fine. For the cases in Figure 5(b), compared to those based solely on language semantics, related items generated using item indices that integrate both language and collaborative semantics are more suitable for recommendation scenarios.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2ea4/2ea43828-b169-47a2-8fb5-816d988482a6.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 6: Content changes caused by each level index.</div>
the item name to a significant extent. The subsequent third level further refines the semantic information, while the fourth level contains relatively less semantic information, which is consistent with the coarse-to-fine quantization process employed during index construction. Moreover, we also count the proportion of generated results changes caused by each level of indices. As shown in Figure 6, also consistent with our conjecture, the proportion of content changes gradually decreases as the index level increases. On the other hand, we try to generate the item that are most relevant or similar to a given item through its indices. We then compare the generated results with the similar item obtained based on cosine similarity between item text embeddings. As presented in Figure 5(b), the similar item generated by our LC-Rec is a game of the same category and platform as the source item, while a duplicate game for another platform is
obtained simply based on language semantic similarity. In recommendation scenarios, the former that integrates both language and collaborative semantics is usually more suitable to meet user needs.
# V. CONCLUSION
In this paper, we proposed a LLM-based recommendation approach, named LC-Rec. In order to adapt LLMs to sequential recommendation tasks, we focused on two main aspects: item indexing and alignment tuning. Concretely, we introduced a vector quantization method combined with uniform semantic mapping for item index learning. To facilitate the integration of item indices into the LLM, we proposed a series of semantic alignment tasks to align language and collaborative semantics for recommendation. These tasks include sequential item prediction, explicit index-language alignment, and implicit recommendation-oriented alignment. Based on the learned item indices, our approach employed these alignment tuning tasks to effectively adapt LLMs for sequential recommendation. Extensive experiments on three large datasets demonstrated the effectiveness of our approach, outperforming a number of competitive baseline models. As future work, we will explore how to extend the current approach in a multi-turn chat setting, so that it can support more flexible interaction with users. In addition, we will also investigate how to better reserve the general abilities of LLMs when making domain adaptations.
# REFERENCES
[1] B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk, “Sessionbased recommendations with recurrent neural networks,” arXiv preprint arXiv:1511.06939, 2015. [2] W.-C. Kang and J. McAuley, “Self-attentive sequential recommendation,” in ICDM, 2018. [3] J. Li, P. Ren, Z. Chen, Z. Ren, T. Lian, and J. Ma, “Neural attentive session-based recommendation,” in CIKM, 2017. [4] J. Tang and K. Wang, “Personalized top-n sequential recommendation via convolutional sequence embedding,” in WSDM, 2018. [5] F. Yuan, A. Karatzoglou, I. Arapakis, J. M. Jose, and X. He, “A simple convolutional generative network for next item recommendation,” in WSDM, 2019. [6] S. Wu, Y. Tang, Y. Zhu, L. Wang, X. Xie, and T. Tan, “Session-based recommendation with graph neural networks,” in AAAI, 2019. [7] C. Xu, P. Zhao, Y. Liu, V. S. Sheng, J. Xu, F. Zhuang, J. Fang, and X. Zhou, “Graph contextualized self-attention network for session-based recommendation.” in IJCAI, 2019. [8] F. Sun, J. Liu, J. Wu, C. Pei, X. Lin, W. Ou, and P. Jiang, “Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer,” in CIKM, 2019. [9] T. Zhang, P. Zhao, Y. Liu, V. S. Sheng, J. Xu, D. Wang, G. Liu, X. Zhou et al., “Feature-level deeper self-attention network for sequential recommendation.” in IJCAI, 2019. [10] K. Zhou, H. Wang, W. X. Zhao, Y. Zhu, S. Wang, F. Zhang, Z. Wang, and J.-R. Wen, “S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization,” in CIKM, 2020. [11] Y. Xie, P. Zhou, and S. Kim, “Decoupled side information fusion for sequential recommendation,” in SIGIR, 2022. [12] Y. Hou, S. Mu, W. X. Zhao, Y. Li, B. Ding, and J.-R. Wen, “Towards universal sequence representation learning for recommender systems,” in SIGKDD, 2022. [13] J. Li, M. Wang, J. Li, J. Fu, X. Shen, J. Shang, and J. McAuley, “Text is all you need: Learning language representations for sequential recommendation,” in SIGKDD, 2023. [14] H. Ding, Y. Ma, A. Deoras, Y. Wang, and H. Wang, “Zero-shot recommender systems,” arXiv preprint arXiv:2105.08318, 2021. [15] Y. Hou, Z. He, J. McAuley, and W. X. Zhao, “Learning vector-quantized item representation for transferable sequential recommenders,” in WWW, 2023. [16] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong et al., “A survey of large language models,” arXiv preprint arXiv:2303.18223, 2023. [17] L. Wu, Z. Zheng, Z. Qiu, H. Wang, H. Gu, T. Shen, C. Qin, C. Zhu, H. Zhu, Q. Liu et al., “A survey on large language models for recommendation,” arXiv preprint arXiv:2305.19860, 2023. [18] Y. Hou, J. Zhang, Z. Lin, H. Lu, R. Xie, J. McAuley, and W. X. Zhao, “Large language models are zero-shot rankers for recommender systems,” arXiv preprint arXiv:2305.08845, 2023. [19] J. Zhang, R. Xie, Y. Hou, W. X. Zhao, L. Lin, and J.-R. Wen, “Recommendation as instruction following: A large language model empowered recommendation approach,” arXiv preprint arXiv:2305.07001, 2023. [20] Z. Cui, J. Ma, C. Zhou, J. Zhou, and H. Yang, “M6-rec: Generative pretrained language models are open-ended recommender systems,” arXiv preprint arXiv:2205.08084, 2022. [21] K. Bao, J. Zhang, Y. Zhang, W. Wang, F. Feng, and X. He, “Tallrec: An effective and efficient tuning framework to align large language model with recommendation,” arXiv preprint arXiv:2305.00447, 2023. [22] S. Geng, S. Liu, Z. Fu, Y. Ge, and Y. Zhang, “Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5),” in RecSys, 2022. [23] W. Hua, S. Xu, Y. Ge, and Y. Zhang, “How to index item ids for recommendation foundation models,” SIGIR-AP, 2023. [24] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” NeurIPS, 2022. [25] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023. [26] S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme, “Factorizing personalized markov chains for next-basket recommendation,” in WWW, 2010.
[27] R. He and J. McAuley, “Fusing similarity models with markov chains for sparse sequential recommendation,” in ICDM, 2016. [28] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” in ICLR, 2023. [29] D. Zhang, S. Li, X. Zhang, J. Zhan, P. Wang, Y. Zhou, and X. Qiu, “Speechgpt: Empowering large language models with intrinsic crossmodal conversational abilities,” arXiv preprint arXiv:2305.11000, 2023. [30] J. Jiang, K. Zhou, Z. Dong, K. Ye, W. X. Zhao, and J.-R. Wen, “Structgpt: A general framework for large language model to reason over structured data,” arXiv preprint arXiv:2305.09645, 2023. [31] Y. Gao, T. Sheng, Y. Xiang, Y. Xiong, H. Wang, and J. Zhang, “Chatrec: Towards interactive and explainable llms-augmented recommender system,” arXiv preprint arXiv:2303.14524, 2023. [32] S. Dai, N. Shao, H. Zhao, W. Yu, Z. Si, C. Xu, Z. Sun, X. Zhang, and J. Xu, “Uncovering chatgpt’s capabilities in recommender systems,” in RecSys, 2023. [33] L. Wang and E.-P. Lim, “Zero-shot next-item recommendation using large pretrained language models,” arXiv preprint arXiv:2304.03153, 2023. [34] Z. Yue, S. Rabhi, G. de Souza Pereira Moreira, D. Wang, and E. Oldridge, “Llamarec: Two-stage recommendation using large language models for ranking,” arXiv preprint arXiv:2311.02089, 2023. [35] X. Lin, W. Wang, Y. Li, F. Feng, S.-K. Ng, and T.-S. Chua, “A multi-facet paradigm to bridge large language model and recommendation,” arXiv preprint arXiv:2310.06491, 2023. [36] Y. Zhang, F. Feng, J. Zhang, K. Bao, Q. Wang, and X. He, “Collm: Integrating collaborative embeddings into large language models for recommendation,” arXiv preprint arXiv:2310.19488, 2023. [37] Y. Zhu, L. Wu, Q. Guo, L. Hong, and J. Li, “Collaborative large language model for recommender systems,” arXiv preprint arXiv:2311.01343, 2023. [38] S. Rajput, N. Mehta, A. Singh, R. H. Keshavan, T. Vu, L. Heldt, L. Hong, Y. Tay, V. Q. Tran, J. Samost, M. Kula, E. H. Chi, and M. Sathiamoorthy, “Recommender systems with generative retrieval,” in NeurIPS, 2023. [39] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “Soundstream: An end-to-end neural audio codec,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2021. [40] D. Lee, C. Kim, S. Kim, M. Cho, and W.-S. Han, “Autoregressive image generation using residual quantization,” in CVPR, 2022. [41] J. Zhan, J. Mao, Y. Liu, J. Guo, M. Zhang, and S. Ma, “Learning discrete representations via constrained clustering for effective and efficient dense retrieval,” in WSDM, 2022. [42] Y. Asano, C. Rupprecht, and A. Vedaldi, “Self-labelling via simultaneous clustering and representation learning,” in ICLR, 2019. [43] M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, and A. Joulin, “Unsupervised learning of visual features by contrasting cluster assignments,” NeurlPS, 2020. [44] S. Lin, C. Liu, P. Zhou, Z.-Y. Hu, S. Wang, R. Zhao, Y. Zheng, L. Lin, E. Xing, and X. Liang, “Prototypical graph contrastive learning,” TNNLS, 2022. [45] M. Cuturi, “Sinkhorn distances: Lightspeed computation of optimal transport,” NIPS, vol. 26, 2013. [46] Y. Jin, K. Xu, L. Chen, C. Liao, J. Tan, B. Chen, C. Lei, A. Liu, C. Song, X. Lei et al., “Unified language-vision pretraining with dynamic discrete visual tokenization,” arXiv preprint arXiv:2309.04669, 2023. [47] X. Wang, X. Tang, X. Zhao, J. Wang, and J. Wen, “Rethinking the evaluation for conversational recommendation in the era of large language models,” in EMNLP, 2023. [48] K. Lee, D. Ippolito, A. Nystrom, C. Zhang, D. Eck, C. Callison-Burch, and N. Carlini, “Deduplicating training data makes language models better,” in ACL, 2022. [49] R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, J. Heek, K. Xiao, S. Agrawal, and J. Dean, “Efficiently scaling transformer inference,” MLSys, vol. 5, 2023. [50] A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, and K. Keutzer, “A survey of quantization methods for efficient neural network inference,” arXiv preprint arXiv:2103.13630, 2021. [51] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” in SOSP, 2023. [52] J. Ni, J. Li, and J. McAuley, “Justifying recommendations using distantlylabeled reviews and fine-grained aspects,” in EMNLP-IJCNLP, 2019.
[53] C. Ma, P. Kang, and X. Liu, “Hierarchical gating networks for sequential recommendation,” in SIGKDD, 2019. [54] K. Zhou, H. Yu, W. X. Zhao, and J.-R. Wen, “Filter-enhanced mlp is all you need for sequential recommendation,” in WWW, 2022. [55] P.-S. Huang, X. He, J. Gao, L. Deng, A. Acero, and L. Heck, “Learning deep structured semantic models for web search using clickthrough data,” in CIKM, 2013. [56] J. D. M.-W. C. Kenton and L. K. Toutanova, “Bert: Pre-training of deep
bidirectional transformers for language understanding,” in NAACL-HLT, 2019. [57] J. Gao, D. He, X. Tan, T. Qin, L. Wang, and T. Liu, “Representation degeneration problem in training natural language generation models,” in ICLR, 2018. [58] L. Wang, J. Huang, K. Huang, Z. Hu, G. Wang, and Q. Gu, “Improving neural language generation with spectrum control,” in ICLR, 2019.
