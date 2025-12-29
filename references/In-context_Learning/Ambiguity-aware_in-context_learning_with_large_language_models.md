Lingyu Gao ∗
Aditi Chaudhary
Toyota Technological Institute at Chicago
Google Research
lygao@ttic.edu
aditichaud@google.com

Lingyu Gao ∗
Aditi Chaudhary
Krishna Srini
Toyota Technological Institute at Chicago
Google Research
Google Resear
lygao@ttic.edu
aditichaud@google.com
krishnaps@googl

Lingyu Gao ∗
Toyota Technological Institute at Chicago
lygao@ttic.edu

# Kazuma Hashimoto

# Karthik Raman

Google Research
kazumah@google.com

Google Research
karthikraman@google.com

# Abstract

In-context learning (ICL), i.e., showing large language models (LLMs) only a few taskspecific demonstrations, has led to downstream gains without task-specific fine-tuning. However, LLMs are sensitive to the choice of prompts, and therefore a crucial research question is how to select good demonstrations for ICL. One effective strategy is leveraging semantic similarity between the ICL demonstrations and test inputs by using a text retriever, which however is sub-optimal as that does not consider the LLM’s existing knowledge about that task. From prior work (Lyu et al., 2023), we already know that labels paired with the demonstrations bias the model predictions. This leads us to our hypothesis whether  considering LLM’s existing knowledge about the task, especially with respect to the output label space can help in a better demonstration selection strategy. Through extensive experimentation on three text classification tasks, we find that it is beneficial to not only choose semantically similar ICL demonstrations but also to choose those demonstrations that help resolve the inherent label ambiguity surrounding the test example. Interestingly, we find that including demonstrations that the LLM previously mis-classified and also fall on the test example’s decision boundary, brings the most performance gain.

# Introduction

Leveraging LLMs (Brown et al., 2020; Chowdhery et al., 2022; Thoppilan et al., 2022) via in-context learning  (ICL) is now a popular strategy for improving downstream task performance, wherein the model is able to perform a task by simply being conditioned on the task definition and/or few task demonstrations (input-output examples) (Brown et al., 2020; Xie et al., 2021). As ICL gets increasingly adopted, it has brought to light (Lester et al., 2021; Liu et al., 2022; Zhang
∗ Work done as an intern at Google Research.

Google Research
bemike@google.com

et al., 2022; Lu et al., 2022) that LLMs are sensitive to the choice of prompts, making “prompt engineering” for different tasks challenging and time-consuming. However, prompt engineering does not have to be a complete guessing game; rather it can be governed by some data-derived signals. For example, selecting demonstrations that are semantically similar to a new input has shown to be more effective over randomly sampled demonstrations (Das et al., 2021; Liu et al., 2022; Margatina et al., 2023), wherein a text retriever is used to select the topk training examples for each test example based on the input text. The motivation is that using information from existing similar situations will help solve a new problem (Aamodt and Plaza, 1994). However, the solely input-based selection does not explicitly capture the LLM’s existing knowledge about the task-specific label space of both the ICL demonstration as well as the test input. For example, on a five-way sentiment classification task (SST (Socher et al., 2013)), we have observed that the Flan-PaLM 2 model (size L) (Anil et al., 2023) is confused between two specific labels, ‘Very Negative’ and ‘Negative,’ a lot more than say between ‘Neutral’ and ‘Very Negative’, as shown in  Figure 2. This motivates us to investigate whether the model’s existing knowledge can also be leveraged to select even more effective demonstrations. Specifically, we derive signals from the underlying LLM about the output label space of both the new test example and the training data from which we select the demonstrations. As motivated above, the model’s ambiguity around the new test example’s output label will help us know what the model is most confused about, which in turn can be used to select those demonstrations that help reduce this confusion. For selecting such demonstrations from the training data, we propose to consider not only the ground truth labels paired with these demonstrations, but also the usefulness by looking at their

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8ada/8ada831b-ede5-4989-99ab-293a33155583.png" style="width: 50%;"></div>
Figure 1: Overview of our proposed method for selecting ICL demonstrations: For each test example, we first use a retriever to rank training data by semantic similarity. At the same time, we identify the ambiguous label set for each test example and also obtain the output predictions on the retrieved training data. Next, we apply three constraints on the top-ranked demonstrations which are: 1) select those demonstrations whose gold label is in the ambiguous label set, 2) select those which are also mis-classified by the model, and 3) select those mis-classified examples whose predicted label is in the ambiguous label set. Finally, we construct prompts with selected ICL demonstrations to get the final model predictions.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d6d5/d6d516a2-2056-4cce-b721-9f66b2f6f95b.png" style="width: 50%;"></div>
Figure 2: Confusion Matrix of zero-shot experiments on SST with Flan-PaLM 2 (L). Labels: VPos (Very Positive), Pos (Positive), Neu (Neutral), Neg (Negative), VNeg (Very Negative).

model prediction. First, given a test example and pool of training data, for each test example we use an off-the-shelf retriever to retrieve topk examples that have similar input text. For each test example, we identify an ambiguous label set of two output labels that the model is most confused about. Next, we select top-ranked demonstrations such that their ground truth labels lie in the above label set. To further find useful demonstrations, we identify those which are mis-classified by the model; the intu

<div style="text-align: center;">LLM Model prediction
</div>
ition is that showing the model a previously misclassified demonstration could force it to correct it (Tan, 2006; Wang et al., 2020). Finally, on top of the mis-classified demonstrations we add a constraint to select only those demonstrations whose model prediction falls within the ambiguous label set, i.e., on the test example’s decision boundary. To test our hypothesis, we focus on multi-class text classification tasks that have fine-grained nuance in the label space. We conduct extensive experimentation across three tasks, namely SST (Socher et al., 2013), GoEmotions (Demszky et al., 2020), and EDOS (Task-B) (Kirk et al., 2023), all of which have fine-grained label space, making the model more likely to be confused across labels. Our key observations are:

1.  Incrementally adding constraints, i.e., 1) considering label ambiguity of test example, 2) limiting ICL demonstrations to mis-classified demonstrations, and 3) considering  label ambiguity of training examples leads to +1.5%, +2.2%, +2.6% improvement in F1 macro scores over the retriever-based ICL, averaged across all datasets (Table 3).

2.  We find that adding such label-based constraints helps more on a smaller model, i.e., on Flan-PaLM 2 (M) (+3.9% gain) compared

. We also attribute this success of our proposed methods to the observation that the  ambiguous label set acts as a good proxy to the gold test label, and as noted by Min et al. (2022), labels in the ICL demonstrations bias the model predictions the most. Therefore, showing the models the ‘likely’ gold label guides the model to make the correct prediction (Table 5).

# 2 Proposed Method

Typically, in an ICL regime, we assume access to training data D train = {(x 0, y 0), · · ·, (x T, y T)} from which the goal is to select d demonstrations to be used as prompt. As motivated in the introduction, we follow a three-step approach for selecting demonstrations, for each test example, we need to 1) extract semantically similar examples from D train, 2) identify the ambiguous label-set and 3) extract model predictions for D train  to identify misclassified examples. Below, we describe each step in more detail and how they are used together to select the “best” demonstrations.

Typically, in an ICL regime, we assume access to training data D train = {(x 0, y 0), · · ·, (x T, y T)} from which the goal is to select d demonstrations to be used as prompt. As motivated in the introduction, we follow a three-step approach for selecting demonstrations, for each test example, we need to 1) extract semantically similar examples from D train, 2) identify the ambiguous label-set and 3) extract model predictions for D train  to identify misclassified examples. Below, we describe each step in more detail and how they are used together to select the “best” demonstrations.
Extract Semantically Similar Demonstrations Typically, in this approach, demonstrations are selected for each test example x t by finding those examples from the D train that are semantically similar to the test input. The motivation being that observing demonstrations that are similar to the new input text will act as a hint for the model (Margatina et al., 2023). This requires the use of a retriever R, either an off-the-shelf one such as (Liu et al., 2022; Agrawal et al., 2023; Margatina et al., 2023; Luo et al., 2023) or a retriever trained specifically for that task (Das et al., 2021; Rubin et al., 2022). For each test example x t, the retriever R is used to rank examples from D train  based on semantic similarity of the text inputs. Topk input-output pairs are then selected from the ranked D train to be used as ICL demonstrations.
Identify Ambiguous Label-Set As we can observe from the confusion matrix in Figure 2, the model is often confused between two labels. We hypothesize that in addition to semantic similarity, providing demonstrations that help the model resolve this ambiguity will help the model correct itself. Thus, as a next step, we construct a prompt θ for the test example x t, and use the model loglikelihood to score each output label l ∈ L given

Identify Ambiguous Label-Set As we can observe from the confusion matrix in Figure 2, the model is often confused between two labels. We hypothesize that in addition to semantic similarity, providing demonstrations that help the model resolve this ambiguity will help the model correct itself. Thus, as a next step, we construct a prompt θ for the test example x t, and use the model loglikelihood to score each output label l ∈ L given

the prompt. Using this we identify top-2 labels that have the highest scores, which we refer to as the “ambiguous label set” of x t, denoted as L ambig,t = {ˆ y (1) t, ˆ y (2) t}, where ˆ y (1) t and ˆ y (2) t are the first and second most likely labels, respectively.

L ambig,t = {ˆ y (1) t, ˆ y (2) t}, where ˆ y (1) t and ˆ y (2) t are the first and second most likely labels, respectively.
Extract Mis-classified Demonstrations The final component in our recipe is to consider the model prediction of the training data. While prior work Min et al. (2022); Yoo et al. (2022); Margatina et al. (2023) has looked at training data label-space from the lens of ground-truth labels, i.e., whether to retain them in the ICL or not, we aim to look at label-space from the perspective of model predictions. Specifically, we are interested in identifying “hard” demonstrations, i.e., examples on which the model makes mistakes. We hope that by showing the model such examples with their ground truth labels will force the model to correct itself. Prior work has underscored the potential value of leveraging mis-classified examples from the training set to enhance model performance (Tan, 2006; Wang et al., 2020), but they haven’t tested it for ICL demonstration selection on text classification. In addition to the mis-classified examples, we further constrain the model prediction of these mis-classified examples to be one of the ambiguous labels, identified in the above step. Given that we already know which output labels the model is confused between for the test examples, showing the model those demonstrations (with their ground truth labels) which fall on the decision boundary will likely guide the model to choose the correct label for the test input.

# 3 Experimental Setup

# 3.1 Model

We experiment with the Flan-PaLM 2 model, an instruction-tuned model which is finetuned on the Flan dataset (Chung et al., 2022; Longpre et al., 2023) based on PaLM-2 (Anil et al., 2023), a multilingual large language model pretrained on web documents, books, code, mathematics and conversational data. We chose these models as Luo et al., 2023 find that retrieved demonstration for ICL works better with instruction-tuned models over general LLMs (e.g., GPT). In particular, we experiment with two variants of the model, namely Flan-PaLM-2 (M) and Flan-PaLM-2 (L), where the

latter is a larger parameter model. 1 The ICL demonstrations are selected using an off-the-shelf retriever which is finetuned on mT5-base (Xue et al., 2021) using the unsupervised objective proposed by  Izacard et al. (2021). Since the order of demonstrations may impact the model performance (Kumar and Talukdar, 2021; Lu et al., 2022), we randomly shuffle the order of demonstrations for three random seeds and report the average results.

As mentioned above, the Flan-PaLM 2 models are finetuned on the Flan dataset which is a mixture of many supervised datasets. Specifically, we choose three text classification datasets that satisfy the following desiderata, 1) the output label space shows fine-grained nuance that spans multiple labels, and 2) these datasets are not part of the Flan mixture to avoid any inherent bias from the underlying model. We describe them below, with dataset statistics shown in Table 1. All datasets are in English.

EDOS (Task-B): The Task B of Explainable Detection of Online Sexism (Kirk et al., 2023), is a topic classification task where the sexist content is classified into four categories, i.e., 1) Threats, plans to harm & incitement, 2) Derogation, 3) Animosity, and 4) Prejudiced Discussion.

SST: The Stanford Sentiment Treebank (SST,
Socher et al., 2013) is a 5-way  sentiment classification dataset for movie reviews with labels: Very Negative, Negative, Neutral, Positive, and Very Positive.

GoEmotions: The GoEmotions (Demszky et al.,
2020) is a multi-class sentiment classification dataset with “neutral” and 27 emotional classes, e.g., “admiration” and “fear”, collected from Reddit comments. As the label space is very large and given that we have limited sequence length, it becomes even more crucial to select a concise but effective prompt. 2

# 3.3 Baselines

We compare our proposed method against the following baselines:

1 Please refer to Anil et al. (2023) for more details on the models. 2 We exclude 24,848 examples (19,925 from training set, 2,474 and 2,449 from dev and test set, respectively) that have multiple labels annotated for a single input, for a simpler experimental setting. We refer the reader to Demszky et al. (2020) for more information on the single-label setting.

1 Please refer to Anil et al. (2023) for more details on the models. 2 We exclude 24,848 examples (19,925 from training set, 2,474 and 2,449 from dev and test set, respectively) that have multiple labels annotated for a single input, for a simpler experimental setting. We refer the reader to Demszky et al. (2020) for more information on the single-label setting.

train
dev
test
EDOS
3,398
486
970
SST
8,544
1,101
2,210
GoEmotions
23,485
2,952
2,978
Table 1: Number of examples in each dataset split.

Frequent Label (FREQ). Select the most frequent label as the model prediction for all test examples.

quent label as the model prediction for all test examples.
Zero-shot ICL (ZERO). For each test example x t, we prepend the task definition to each test input and prompt the models. 3 To obtain the model prediction, we use the model log-likelihood to score each output label l ∈ L, given the prompt. Then, we select the label with the highest score. y t = arg max L score (l, θ) where θ refers to the prompt specifically used for this setting, and score refers to the model’s log-likelihood.
Static N-shot ICL (STATIC N). We manually select N demonstrations from D train, one for each of the N output labels (N = |L|). Note that these demonstrations are static for all test examples. Thus, we concatenate the task definition, N demonstrations and test example x t as the prompt for ICL and use the log-likelihood scores, as described above, to get the model prediction.
Retriever-based ICL (RETR). Unlike above, where we used the same prompt for all test inputs, in this baseline, we retrieve demonstrations for each test input x t. We use an off-the-shelf retriever R (subsection 3.1) to retrieve k nearest neighbors {x 1,t, · · ·, x k,t} from D train, similar to Das et al. (2021). We encode the input text of training set and the test example, rank the training data by the inner product of the vectors. Of these k examples, we select n = 4, 8 as ICL demonstrations. 4

Retriever-based ICL (RETR). Unlike above, where we used the same prompt for all test inputs, in this baseline, we retrieve demonstrations for each test input x t. We use an off-the-shelf retriever R (subsection 3.1) to retrieve k nearest neighbors {x 1,t, · · ·, x k,t} from D train, similar to Das et al. (2021). We encode the input text of training set and the test example, rank the training data by the inner product of the vectors. Of these k examples, we select n = 4, 8 as ICL demonstrations. 4

# 3.4 Proposed Method: A MBIG-ICL

As described in section 2, our proposed method considers both semantic similarity and the label ambiguity for selecting demonstrations. Below,

3 Please refer to Appendix A.1 for the exact prompt and prompt template used in this setting, as well as for few shot settings such as the subsequent STATIC N and RETR. 4 We chose k = 4, 8  for two reasons: a) to limit the sequence length to 1024 tokens for faster inference, and b) in some settings we found k = 4 often outperforming k = 8 (Table 2), which led us to believe that adding more examples will not benefit much.

EDOS
SST
GoEmotions
Avg.
M
L
M
L
M
L
M
L
Baselines
FREQ
15.9
15.9
7.5
7.5
0.8
0.8
8.1
8.1
ZERO
50.7
60.5
49.2
54.1
40.5
43.4
46.8
52.7
STATIC-N
51.1±0.3
58.5±0.4
50.3±0.4
56.5±0.3
34.3±0.5
44.4±0.3
45.2
53.1
RETR-4
48.5±0.3
62.3±0.4
49.9±0.3
55.4±0.3
38.3±0.3
46.2±0.4
45.6
54.6
RETR-8
47.1±0.2
61.8±0.1
51.5±0.1
55.2±0.4
37.5±0.2
46.7±0.1
45.4
54.6
Ours
AMBIG-4
+GOLD
49.3±0.6
62.6±0.2
51.5±0.4
56.1±0.0
40.7±0.3
48.2±0.2
47.2
55.6
+GOLD+MIS
52.2±0.5
61.7±0.9
52.3±0.1
57.4±0.1
40.1±0.2
47.6±0.1
48.2
55.6
+GOLD+MIS+PRED
53.9±0.5
62.9±0.4
53.3±0.4
58.0±0.0
42.3±0.5
47.7±0.2
49.8
56.2
AMBIG-8
+GOLD
47.5±0.1
63.2±0.2
52.9±0.1
56.5±0.6
42.0±1.2
47.7±0.1
47.5
55.8
+GOLD+MIS
50.4±0.4
62.0±0.4
53.4±0.1
57.7±0.1
43.9±0.2
47.6±0.4
49.2
55.8
+GOLD+MIS+PRED
50.9±0.6
62.7±0.2
54.3±0.2
57.2±0.3
41.3±0.3
47.4±0.3
48.8
55.8
Table 2: F1 macro (%) comparison between our baselines (top) and our proposed methods (bottom) with Flan-PaLM 2 (M/L). 4 or 8 refers to the number of ICL demonstrations. The best performance across all method is highlighted and the best performing baseline is underlined. The “Avg.” column shows the average scores across all datasets. The standard deviations are computed over three random seeds, with the order of demonstrations shuffled.

ZERO
STATIC-N
AMBIG-ICL
+GOLD
+MIS
+PRED
M
1.3
-0.2
1.9
3.3
3.9
L
-1.9
-1.5
1.1
1.1
1.4
all
-0.3
-0.9
1.5
2.2
2.6
<div style="text-align: center;">We omitted RETR in the table, which are inherently zero as we compare against RETR. For both RETR and A MBIG-ICL, we average results on both 4 and 8 shots before computing differences.
</div>
Table 3: F1 macro (%) differences compared to RETR, averaged across all datasets as detailed in Table 2. M and L refers to Flan-PaLM 2 sizes, and “all” is averaged on results of size M and L. “+ MIS” and “+ PRED” refer to “+ GOLD + MIS” and “+ GOLD + MIS + PRED”, respectively.
we summarize our proposed model variants. For each setting, we first retrieve the topk  most similar examples from the training data D train for each test example x t. We denote these candidates by R (x t) = {(x 0,t, y 0,t), · · ·, (x k,t, y k,t)}. At the same time, for each x t, we also identify the ambiguous label-set L ambig,t = {l i, l j | l ∈ L}. This set contains the top-2 labels, l i and l j, that the model is most confused about, where both labels belong to the set L of all output labels.
+ GOLD Select those examples from R (x t) as demonstrations where the ground truth label of each demonstration belongs to the ambiguous label set of x t denoted by:

+ GOLD + MIS Select those examples from R (x t) as demonstrations where the ground truth labels

fall in L ambig,t and they are mis-classified, denote by:

Same as above, the model predictions on the training data are obtained from ZERO. For all our proposed model variants, we select n demonstrations where n = 4 and n = 8.

# 4 Results and Discussion

We report all our results in Table 2. Specifically, we use the F1 macro scores to compare the model performance, as all our tasks have unbalanced datasets. 5 First, we note across all three tasks, our proposed methods outperform the baselines. We also note that the zero-shot model (ZERO) which only uses a task definition but no task demonstrations, already is a strong baseline for both the Flan-PaLM 2 models (M/L). In particular, comparing the average scores of the few-shot baselines

5 We report the accuracy, precision and recall in A.2.

Test Example: Ok! I like making friends
Lambig,t: Love, Joy
Gold label: Love
RETR
1. Disappointment: I want to make friends too :( but I feel like I have nothing good
to offer
2. Joy: I, too, am a lot of fun at parties. We can stand together in the corner!
3. Gratitude: Thanks. I am. I make some new friends.
4. Disapproval: Not really. My group of friends are awesome in every way possible
except they are homophobic
Predicted:
Joy
AMBIG-ICL
+GOLD
1. Joy: I, too, am a lot of fun at parties. We can stand together in the corner!
2. Love: I ... I like you
3. Love: Married to the love of my life. LOL
4. Love: I do. but some people love it
Predicted:
Love
+GOLD+MIS
1. Joy: I, too, am a lot of fun at parties. We can stand together in the corner!
2. Love: Too cute for me. Why cant i have a boyfriend *[NAME]*
3. Joy: FaceTime with wifey!! Happy anniversary!
4. Love: Stick around! Would love your input POV!
Predicted:
Love
+GOLD+MIS+PRED
1. Joy: FaceTime with wifey!! Happy anniversary!
2. Joy: She want to take it slow, I can see that... I deal with those girls all the time,
they my favorite
3. Love: Ha! I like that one.
4. Love: Ooh I like that one :)
Predicted:
Love
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e6a0/e6a0b890-6514-4c3d-a69f-157bff108395.png" style="width: 50%;"></div>
<div style="text-align: center;">Test Example: Ok! I like making friends
</div>
Table 4: Example demonstrations selected by the RETR and our proposed method A MBIG-ICL for the GoEmotions task, for n = 4. Each demonstration comprises of the input text and the ground truth label, as selected from the training data. On Flan-PaLM 2 (L), where RETR mis-classified it as “Joy”, A MBIG-ICL predicted correctly under all three settings.

and ZERO, we find that ZERO outperforms few-shot baselines by 1.4% on Flan-PaLM 2 (M), but the larger model Flan-PaLM 2 (L) benefits from the addition of ICL demonstrations (+1.4% gain). This is because larger-parameter models make better use of in-context learning (Chan et al., 2022; Akyürek et al., 2023; Wei et al., 2023). Interestingly, we also observe that for SST and GoEmotions, the FlanPaLM 2 (L) model achieves higher performance with n = 4 over n = 8, which highlights that quantity does not necessarily lead to better performance.
Considering output label space is more important than semantic similarity. Within the fewshot methods, where we use ICL demonstrations along with the task definition, we compute from  Table 3 that our proposed methods AMBIG-* outperforms retriever-based models (RETR-*) by +3.0% (avg.) for Flan-PaLM 2 (M), and by +1.2% (avg.) for Flan-PaLM 2 (L), suggesting that considering output label space for selecting demonstrations is as important as considering the input similarity. In particular, we find that considering mis-classified demonstrations that fall on the test example’s decision boundary leads to the overall best performance. In Table 4, we show the demonstrations selected for the n = 4  setting for one example of the GoEmotions task. We see that for the test input “Ok! I like making friends”, the RETR method retrieved

and ZERO, we find that ZERO outperforms few-shot baselines by 1.4% on Flan-PaLM 2 (M), but the larger model Flan-PaLM 2 (L) benefits from the addition of ICL demonstrations (+1.4% gain). This is because larger-parameter models make better use of in-context learning (Chan et al., 2022; Akyürek et al., 2023; Wei et al., 2023). Interestingly, we also observe that for SST and GoEmotions, the FlanPaLM 2 (L) model achieves higher performance with n = 4 over n = 8, which highlights that quantity does not necessarily lead to better performance.

similar examples from D train (all examples refer to friends). Now from the ZERO model, we calculated the model prediction scores and found that Love and Joy  are the two labels the model is most confused about. However, because we do not consider any test example ambiguity in RETR, only one of the retrieved examples represent the labels Love or Joy, which are the two labels the model is most confused about for this test example. Whereas, in the A MBIG-ICL setting, because of our constraints, all the examples chosen for ICL belong to the ambiguous label set. This allows all our proposed methods to better understand this fine-grained nuance across label space and make the correct model prediction of Love. Below, we conduct some analysis to further explain the way our proposed methods work.

# Considering output label space compensates for

the sacrifice in semantic similarity. As we introduce more constraints (i.e., + GOLD, + MIS, and + PRED), we find that we need to sacrifice the semantic similarity to the test input. For example, consider the 4-shot A MBIG-ICL experiment on EDOS (Task-B), to satisfy the constraints for the + GOLD setting we need to select up to top-16 retrieved examples in order to obtain the 4 ICL demonstrations; for + GOLD + MIS we need top-55 retrieved examples and more than top-250 retrieved exam

ples for + GOLD + MIS + PRED. 6 Clearly, by selecting lower ranked examples from the retrieved set R (x t) we are sacrificing the semantic similarity to the test input. While previous studies, such as (Das et al., 2021; Liu et al., 2022; Margatina et al., 2023), have indicated that greater semantic similarity can enhance model performance, we can see that our methods can still outperform the retriever-based baselines which prioritize it.

# The ambiguous label set is a good proxy for th

test gold label. While Min et al. (2022) find that using pseudo-demonstrations i.e. demonstrations with random labels instead of the ground truth labels, does not affect the downstream performance much, Lyu et al. (2023) find that for demonstrations that are similar to the test input, such as those from a retriever, pseudo-demonstrations hurt the performance. They refer to this as the copying-effect hypothesis which says that the “model prediction is biased towards the labels paired with the inputs in the demonstrations, especially when the inputs are similar to the test inputs”. This, in turn, suggests that the best performance could be achieved if the labels paired with the inputs are same as the gold label of the test example. Given that we do not know the gold label of the test example apriori, the question then becomes how do we approximate the gold label?. We find that our ambiguous label set acts as a close proxy. In Table 5, we compute how many times is the label paired with ICL demonstrations the same as the test example gold label. We find that 44.2% of our proposed methods’ (AMBIG) demonstrations have the same gold label as the test example on average, compared to 30.9% from the
RETR  method. This is why including the ambiguous label set in the demonstration selection process leads to a higher performance. This analysis also sheds light on the effectiveness of retriever-based ICL. From Table 5  we can see that the demonstrations selected solely based on input text similarity is only 13.3% points (avg.) behind our proposed methods. This confirms that finding demonstrations similar to the input text also leads to selecting demonstrations that have the ‘likely’ gold label.

A MBIG-ICL helps reduce the model confusion. To understand whether including test label ambi

6 We set a strict constraint on our selection (top-250 retrieved example for + GOLD, and top-250 misclassified retrieved examples for the other two). If there aren’t sufficient examples for + GOLD + MIS + PRED  within the top-250 misclassified retrieved example, we fall-back on the previous setting (+ GOLD + MIS).

EDOS
SST
GoEmotions
M
L
M
L
M
L
4-shot
42.6
29.6
21.6
8-shot
42.5
28.6
20.5
AMBIG-4
+GOLD
49.5 50.3
46.5 47.1
41.3
41.9
+GOLD+MIS
46.4 44.3
46.1 44.3
38.7
38.8
+GOLD+MIS+PRED
48.3 42.3
46.1 44.6
37.8
40.7
AMBIG-8
+GOLD
50.3 50.3
46.0 46.8
41.2
41.7
+GOLD+MIS
46.9 43.8
46.4 44.7
38.7
38.6
+GOLD+MIS+PRED
48.8 42.9
46.5 44.9
37.5
40.3
Table 5: Average percentage (%) of examples in the top 4, 8 retrieved demonstrations that share the same gold labels with test example.

EDOS
SST
GoEmotions
M
L
M
L
M
L
uniform
2.00
2.32
4.75
ZERO
0.98 1.08
1.58 1.19
2.44 1.92
STATIC-N
0.87 1.07
1.41 1.11
1.76
1.77
RETR-4
0.78 0.97
1.40 1.06
1.89
1.70
RETR-8
0.82 0.96
1.38 1.04
1.79
1.69
AMBIG-4
+GOLD
0.77 0.93
1.39 1.02
1.86
1.43
+GOLD+MIS
0.85 0.98
1.41 1.06
1.92
1.48
+GOLD+MIS+PRED
0.86 1.00
1.42 1.07
1.92
1.46
AMBIG-8
+GOLD
0.81 0.91
1.36 0.98
1.68
1.33
+GOLD+MIS
0.89 0.97
1.39 1.03
1.74
1.39
+GOLD+MIS+PRED
0.90 1.00
1.40 1.04
1.76
1.37
Table 6: Average entropy of predicted probability distribution. “uniform” refers to the entropy computed for an uniform probability distribution over the labels. Lower entropy is better.

guity indeed helps decrease the model confusion, we calculate the model entropy over the predicted probability distribution of the output labels in  Table 6. 7 Overall, we observe that our A MBIG-* methods achieve the lowest entropy across all three datasets and models. This suggests that by explicitly identifying the point of model confusion (in this case the confusion across fine-grained labels) and selecting demonstrations that help resolve this confusion is indeed effective in reducing the confusion across labels, and thereby resulting in higher downstream performance (Table 2). In particular, we find that for the Flan-PaLM 2 (L), the gap between the few-shot baselines and the A MBIG-* methods is larger, perhaps because larger models are better able to use the ICL demonstrations (Chan et al., 2022; Akyürek et al., 2023; Wei et al., 2023).

7 We compute entropy with a base of 2.

We also compute the Pearson correlation coefficient between F1 macro scores and average entropy of predicted probability distribution (shown in  Table 2 and Table 6, respectively), for all the three datasets. We find that for the Flan-PaLM 2 (L) model, there is a negative correlation for all three datasets, i.e., r = − 0. 78 for EDOS, − 0. 48 for SST and − 0. 92 for GoEmotions, which suggests that lower entropy translates to higher task performance. However, for the Flan-PaLM 2 (M), we have mixed results, as r is positive for EDOS (0. 47), negative for SST (− 0. 55), and close to zero for GoEmotions (0. 03).

# 5 Related Work

The performance of large language models (LLMs) is significantly influenced by the quality of ICL demonstrations, as demonstrated in multiple studies (Zhao et al., 2021; Liu et al., 2022; Zhang et al., 2022). Consequently, the focus on retrieving superior demonstrations has increased. One prominent strategy is to finetune a retriever for specific tasks by similarity metrics (Das et al., 2021; Hu et al., 2022; Poesia et al., 2022) or by scores derived from language models (Rubin et al., 2022; Shi et al., 2022). While some works introduce an unified retriever trained across various tasks (Li et al., 2023; Cheng et al., 2023) for generalizabilty, another direction is to leverage off-the-shelf retrievers. Liu et al., 2022 propose a KNN-based method to select ICL demonstrations based on semantic similarities; Margatina et al., 2023 select ICL demonstrations with active learning algorithms based on uncertainty, diversity, and similarity, and show that selecting based on input text similarity consistently outperforms other methods; and Agrawal et al., 2023 focus on selecting diverse demonstrations as well as promoting n-gram overlap between demonstrations and test examples. In our work, we adopt the off-the-shelf retriever approach as our focus is to show the generalizability of our approach across different classification tasks. However, we expect that our method will also benefit from a task-specific retriever. Additionally, to the best of our knowledge, we are the first ones to leverage the LLM’s existing knowledge surrounding the test example for selecting demonstrations. Prior works have typically explored the LLM’s existing knowledge, considering the model prediction for the training data.

Luo et al., 2023 use the LLM prediction score on the training data to train a task-specific retriever, and also use Chain-of-Thought prompting (Wei et al., 2022) to improve model performance. Some works (Kumar and Talukdar, 2021; Lu et al., 2022) have found that ordering of the ICL demonstrations also affects the downstream performance, that is why in Table 2 we report the results across three shuffle orders. These works are orthogonal to our work but can be used in combination with our proposed methods.

# 6 Conclusion and Next Steps

In this work, we find that using LLM’s existing knowledge (e.g., the model prediction) regarding the output label space of both the test example and the ICL demonstration pool is as important as considering the semantic similarity of the input text alone. We find that our proposed method consistently outperform the baselines for all three tasks. Although, we only consider the top-2 most ambiguous labels in selecting the ICL demonstrations, it would be interesting to expand the ambiguous label set to more than two labels. This would especially be more important for datasets like GoEmotions where the label space is large and much more fine-grained. We leave this effort for future work. Furthermore, in this work, we focus on sentence classification tasks, thus paving the way for others to use our proven techniques to also explore label ambiguity for other token/span-level tasks such as Named Entity Recognition (NER), and Part-Of-Speech (POS) tagging.

# 7 Limitations

We focus on reducing LLM’s label ambiguity by incorporating demonstrations that are misclassified by the LLM and reside on the test example’s decision boundary. While we show this methodology’s effectiveness across datasets, even those with a granular label structure, potential pitfalls remain. If the actual gold label of test example often deviates from the LLM’s top two label choices in a particular dataset or model, this can be indicative of subpar zero-shot performance or flawed ambiguous label set selection. In these scenarios, our method may lead to unsatisfying performance, necessitating further enhancements.

# 8 Ethics Statement

We use pretrained large language models (LLMs) for text classification. Notably, LLMs are shown to exhibit biases, which is a well-recognized challenge and the broader community is currently working to address. Since our main goal is to improve the downstream task performance, an improved performance on an offensive content classification task could be misused. In particular, the EDOS dataset used in our work, contains offensive content. We selected this dataset for its fine-grained label nuances and to ensure our research isn’t biased by models inherently familiar with the data.

# References

Agnar Aamodt and Enric Plaza. 1994.  Case-based reasoning: Foundational issues, methodological variations, and system approaches. AI Commun., 7(1):39– 59.
Sweta Agrawal, Chunting Zhou, Mike Lewis, Luke Zettlemoyer, and Marjan Ghazvininejad. 2023.  Incontext examples selection for machine translation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8857–8873, Toronto, Canada. Association for Computational Linguistics.
Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. 2023.  What learning algorithm is in-context learning? investigations with linear models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernández Ábrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan A. Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vladimir Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, and et al. 2023. Palm 2 technical report. CoRR, abs/2305.10403.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child,

Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In  Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022.  Palm: Scaling language modeling with pathways. CoRR, abs/2204.02311.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam

Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instruction-finetuned language models. CoRR, abs/2210.11416.

2022. Scaling instruction-finetuned language models. CoRR, abs/2210.11416.
Rajarshi Das, Manzil Zaheer, Dung Thai, Ameya Godbole, Ethan Perez, Jay Yoon Lee, Lizhen Tan, Lazaros Polymenakos, and Andrew McCallum. 2021.  Casebased reasoning for natural language queries over knowledge bases. In  Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 9594–9611, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. 2020.  GoEmotions: A dataset of fine-grained emotions. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4040–4054, Online. Association for Computational Linguistics.
Yushi Hu, Chia-Hsuan Lee, Tianbao Xie, Tao Yu, Noah A. Smith, and Mari Ostendorf. 2022. Incontext learning for few-shot dialogue state tracking. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 2627–2643, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Towards unsupervised dense information retrieval with contrastive learning. CoRR, abs/2112.09118.
Hannah Kirk, Wenjie Yin, Bertie Vidgen, and Paul Röttger. 2023. SemEval-2023 task 10: Explainable detection of online sexism. In Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 2193–2210, Toronto, Canada. Association for Computational Linguistics.
Sawan Kumar and Partha Talukdar. 2021. Reordering examples helps during priming-based few-shot learning. In  Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 4507–4518, Online. Association for Computational Linguistics.
Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.
The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
Xiaonan Li, Kai Lv, Hang Yan, Tianyang Lin, Wei Zhu, Yuan Ni, Guotong Xie, Xiaoling Wang, and Xipeng Qiu. 2023.  Unified demonstration retriever for incontext learning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4644–4668, Toronto, Canada. Association for Computational Linguistics.

learning. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2655–2671, Seattle, United States. Association for Computational Linguistics.
Peng Shi, Rui Zhang, He Bai, and Jimmy Lin. 2022.  XRICL: Cross-lingual retrieval-augmented incontext learning for cross-lingual text-to-SQL semantic parsing. In  Findings of the Association for Computational Linguistics: EMNLP 2022, pages 5248– 5259, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In  Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA. Association for Computational Linguistics.
Songbo Tan. 2006. An effective refinement strategy for KNN text classifier. Expert Syst. Appl., 30(2):290– 298.
Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Kathleen S. Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Agüera y Arcas, Claire Cui, Marian Croak, Ed H. Chi, and Quoc Le. 2022. Lamda: Language models for dialog applications. CoRR, abs/2201.08239.
Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, ˙ Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. 2020. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261–272.
Xiaobo Wang, Shifeng Zhang, Shuo Wang, Tianyu Fu, Hailin Shi, and Tao Mei. 2020.  Mis-classified vector guided softmax loss for face recognition. In The

Xiaobo Wang, Shifeng Zhang, Shuo Wang, Tianyu Fu, Hailin Shi, and Tao Mei. 2020.  Mis-classified vector guided softmax loss for face recognition. In The

Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 12241– 12248. AAAI Press.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022.  Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS.
Jerry W. Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, and Tengyu Ma. 2023. Larger language models do in-context learning differently. CoRR, abs/2303.03846.
Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2021. An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080.
Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483–498, Online. Association for Computational Linguistics.
Kang Min Yoo, Junyeob Kim, Hyuhng Joon Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-Woo Lee, Sang-goo Lee, and Taeuk Kim. 2022. Ground-truth labels matter: A deeper look into input-label demonstrations. In  Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2422– 2437, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
Yiming Zhang, Shi Feng, and Chenhao Tan. 2022.  Active example selection for in-context learning. In  Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9134– 9148, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021.  Calibrate before use: Improving few-shot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 12697–12706. PMLR.

# A Appendix

# A.1 Prompt Construction

We show our templates in Table 7 (we use 4-shot as an example for few-shot). Task definitions are listed below, denoted by x defn:

• EDOS: Given a text input, the task is to classify the input as being a Threat, Prejudiced, Animosity, or Derogation category of sexism. Threat refers to language where an individual expresses intent andor encourages others to take action against women which inflicts or incites serious harm and violence against them. It includes threats of physical, sexual or privacy harm. Prejudiced refers to language which denies the existence of discrimination, and justifies sexist treatment. It includes denial and justification of gender inequality, excusing women’s mistreatment, and the ideology of male victimhood. Animosity refers to language which expresses implicit or subtle sexism, stereotypes or descriptive statements. It includes benevolent sexism, i.e., framed as a compliment. Derogation refers to language which explicitly derogates, dehumanises, demeans or insults women. It includes negative descriptions and stereotypes about women, objectification of their bodies, strong negative emotive statements, and dehumanising comparisons. It covers negative statements directed at a specific woman and women in general.
• SST: Given sentences from movie reviews, the task is to classify the sentences as being a Great, Good, Okay, Bad, or Terrible category of sentiment. Great refers to language that expresses extremely positive sentiment. Good refers to language that expresses positive sentiment, but not to the extreme. Okay refers to language that is neutral, i.e., neither expresses clear positive nor negative sentiments. Bad refers to language that expresses negative sentiment, but not to the extreme. Terrible refers to language that expresses extremely negative sentiment.
• GoEmotions: Given sentences from Reddit comments, the task is to classify the sentences as being an Admiration, Approval, Annoyance, Gratitude, Disapproval, Amusement, Curiosity, Love, Optimism, Disappointment, Joy, Realization, Anger, Sadness, Confusion, Caring, Excitement, Surprise, Disgust, Desire, Fear, Remorse, Embarrassment, Nervousness, Pride, Relief, or Grief category of emotions.

# A.2 Accuracy, Precision, Recall
Please refer to Table 8, 9 and 10.

A.3 Label-wise Percentage Analysis of Gold Label Inclusion in L ambig, t

# A.3 Label-wise Percentage Analysis of Gold Label Inclusion in L ambig, t

L We compute the percentage of times that the test example’s gold label is in L ambig,t (as obtained with ZERO) in Table 11, and we present label-wise results in Table 12.

# A.4 Sorting Order with Predicted Probability Distribution Entropy

Since we have the predicted probability distribution of ICL demonstrations, we tried to sort the ICL demonstrations by increasing entropy order. However, it doesn’t consistently improve model performance, which is shown in Table 13 and 14.

xdefn
Thus given the following input:
input: xt
answer:
xdefn
Some examples are:
input: x1,t
answer: y1,t
input: x2,t
answer: y2,t
input: x3,t
answer: y3,t
input: x4,t
answer: y4,t
Thus given the following input:
input: xt
answer:
Table 7: Prompt templates for zero-shot and few-shot ICL. x t refers to the test example, and x i,t, y i,t refers to the text inputs and gold labels of ICL demonstrations selected for x t, respectively.

# A.5 Example on SST for Comparison between
RETR and A MBIG-ICL

We list example demonstrations on SST where the model correctly predict the test example in our proposed method, but wrongly classify it in RETR in Table 15.

# A.6 Responsible AI Checklist

Packages for Evaluations. We use SciPy (Virtanen et al., 2020) and scikit-learn (Buitinck et al., 2013) for evaluations.

EDOS
SST
GoEmotions
M
L
M
L
M
L
Baselines
FREQ
11.7
11.7
4.6
4.6
0.4
0.4
ZERO
65
60.7
54
56.2
42.6
46.3
STATIC-N
65.2±0.6
58.1±0.4
54.5±0.6
58.2±0.3
42.6±1.2
46.2±0.3
RETR-4
67.1±1.1
63.6±0.5
53.4±0.3
57.4±0.4
43.7±0.4
47.6±0.4
RETR-8
65.0±0.2
63.9±0.3
54.4±0.1
57.6±0.5
43.7±0.4
48.3±0.1
Ours
AMBIG-4
+GOLD
65.9±0.8
63.6±0.4
54.1±0.3
57.7±0.1
45.7±0.3
50.5±0.2
+GOLD+MIS
66.6±1.1
63.6±1.0
54.1±0.2
58.8±0.1
44.8±0.4
49.2±0.1
+GOLD+MIS+PRED
67.4±0.4
65.0±0.5
54.8±0.5
59.4±0.0
46.9±1.3
47.9±0.2
AMBIG-8
+GOLD
66.4±1.1
64.8±0.1
54.7±0.2
58.5±0.7
48.0±1.8
49.9±0.1
+GOLD+MIS
68.4±0.8
64.4±0.6
54.5±0.1
59.6±0.1
48.7±0.5
48.8±0.5
+GOLD+MIS+PRED
66.6±1.2
66.4±0.3
54.9±0.2
59.1±0.4
43.7±0.5
47.4±0.3
EDOS
SST
GoEmotions
M
L
M
L
M
L
Baselines
FREQ
25
25
20
20
3.7
3.7
ZERO
46
62.8
53.8
55.2
42.4
47.2
STATIC-N
46.2±0.3
63.0±0.3
54.0±0.4
56.5±0.2
34.8±0.5
49.5±0.4
RETR-4
44.8±0.3
63.4±0.2
53.4±0.3
55.7±0.3
38.5±0.2
49.7±0.3
RETR-8
44.0±0.1
62.1±0.2
54.2±0.1
55.3±0.4
37.8±0.3
50.1±0.3
Ours
AMBIG-4
+GOLD
45.1±0.6
64.1±0.2
54.6±0.4
56.4±0.1
41.4±0.3
51.3±0.2
+GOLD+MIS
48.0±0.4
62.1±0.9
54.9±0.1
57.3±0.1
40.9±0.1
51.0±0.4
+GOLD+MIS+PRED
49.5±0.4
63.1±0.3
55.6±0.4
57.7±0.0
42.7±0.2
51.7±0.4
AMBIG-8
+GOLD
43.6±0.1
64.0±0.2
55.0±0.1
56.5±0.6
41.9±0.9
50.8±0.5
+GOLD+MIS
47.3±0.4
61.8±0.3
54.9±0.1
57.3±0.1
44.4±0.2
51.4±0.3
+GOLD+MIS+PRED
48.0±0.4
61.7±0.2
55.6±0.2
56.7±0.2
43.2±0.3
51.3±0.1
EDOS
SST
GoEmotions
M
L
M
L
M
L
Baselines
FREQ
46.8
46.8
23.1
23.1
11.7
11.7
ZERO
55.4
59.2
49.9
57.1
47.1
46.2
STATIC-N
54.3±0.3
57.6±0.1
50.5±0.4
59.0±0.2
39.8±0.2
46.7±0.2
RETR-4
53.6±0.3
61.0±0.5
50.0±0.3
58.5±0.3
45.9±0.0
50.1±0.2
RETR-8
53.8±0.3
61.1±0.2
51.8±0.1
58.6±0.4
45.3±0.0
51.0±0.2
Ours
AMBIG-4
+GOLD
54.3±0.4
61.3±0.4
51.5±0.4
58.9±0.1
46.6±0.2
50.3±0.1
+GOLD+MIS
56.1±0.2
60.9±0.6
52.1±0.1
59.7±0.1
45.6±0.1
49.5±0.1
+GOLD+MIS+PRED
56.5±0.1
61.4±0.4
53.0±0.4
60.1±0.1
45.6±0.1
50.0±0.2
AMBIG-8
+GOLD
53.6±0.2
61.8±0.0
52.9±0.1
59.5±0.6
46.8±0.1
50.4±0.2
+GOLD+MIS
55.4±0.6
61.1±0.3
53.2±0.1
60.2±0.1
45.8±0.2
50.0±0.3
+GOLD+MIS+PRED
55.1±0.6
61.5±0.3
54.2±0.2
59.6±0.3
44.9±0.2
50.2±0.2
EDOS
SST
GoEmotions
Flan-PaLM 2 (M)
91.2
85.8
61.2
Flan-PaLM 2 (L)
88.2
87.6
61.6
Table 11: Percentage of times the test example’s gold label is in L ambig,t (as obtained from ZERO model).

Table 11: Percentage of times the test example’s gold label is in L ambig,t (as obtained from ZERO model).

EDOS
M
Animosity 99.1, Derogation 97.4, Prejudiced 52.1, Threat 71.9
L
Animosity 90.1, Derogation 90.1, Prejudiced 68.1, Threat 93.3
SST
M
Bad 78.4, Good 98.0, Great 88.0, Okay 73.8, Terrible 93.9
L
Bad 89.3, Good 99.2, Great 99.2, Okay 59.1, Terrible 85.3
GoEmotions
M
Admiration 72.7, Amusement 90.3, Anger 58.8, Annoyance 44.3, Approval 24.6, Caring 52.9, Confu-
sion 73.2, Curiosity 64.2, Desire 35.7, Disappointment 58.0, Disapproval 52.3, Disgust 55.3, Embarrass-
ment 30.4, Excitement 56.4, Fear 75.4, Gratitude 75.7, Grief 100.0, Joy 80.4, Love 86.8, Nervousness
83.3, Optimism 51.4, Pride 42.9, Realization 27.0, Relief 28.6, Remorse 38.6, Sadness 71.6, Surprise
62.1
L
Admiration 40.2, Amusement 84.9, Anger 52.7, Annoyance 40.2, Approval 36.0, Caring 36.5, Confu-
sion 74.2, Curiosity 64.8, Desire 58.9, Disappointment 59.1, Disapproval 81.0, Disgust 38.2, Embarrass-
ment 30.4, Excitement 61.8, Fear 73.8, Gratitude 88.4, Grief 100.0, Joy 84.8, Love 86.2, Nervousness
83.3, Optimism 65.4, Pride 71.4, Realization 41.6, Relief 42.9, Remorse 70.5, Sadness 67.6, Surprise
64.4
Table 12: Label-wise percentage of times the test example’s gold label is in L ambig,t (as obtained f where M and L refers to Flan-PaLM 2 sizes.

EDOS
SST
M
L
M
L
AMBIG-4
+GOLD
50.2
62.9
51.6
55.8
+GOLD+MIS
51.2
62.7
53.0
57.0
+GOLD+MIS+PRED
53.4
63.8
52.7
57.7
AMBIG-8
+GOLD
48.1
63.3
53.2
56.5
+GOLD+MIS
50.4
62.9
53.6
57.4
+GOLD+MIS+PRED
50.3
62.9
54.3
57.1
Table 13: F1 macro scores (%) of our method. M and L refers to size of Flan-PaLM 2. The ICL demonstrations are sorted by increased entropy order.

EDOS
SST
M
L
M
L
AMBIG-4
+GOLD
0.9
0.3
0.1
-0.3
+GOLD+MIS
-1
1
0.7
-0.4
+GOLD+MIS+PRED
-0.5
0.9
-0.6
-0.3
AMBIG-8
+GOLD
0.6
0.1
0.3
0
+GOLD+MIS
0
0.9
0.2
-0.3
+GOLD+MIS+PRED
-0.6
0.2
0
-0.1
Table 14: The difference of F1 macro scores (%) between the “increased entropy order” and the “averaged over 3 random seeds”.

Test Example: A hip ride into hyper-time, Clockstoppers is a lively and enjoyable adventure for all ages at any time. L ambig, t: Great, Good Gold label: Great
Predicted: Good
RETR 1. Bad: See Clockstoppers if you have nothing better to do with 94 minutes. 2. Bad: Time stands still in more ways that one in Clockstoppers, a sci-fi thriller as lazy as it is interminable. 3. Bad: Clockstoppers is one of those crazy, mixed-up films that doesn’t know what it wants to be when it grows up. 4. Good: Even with all its botches, Enigma offers all the pleasure of a handsome and well-made entertainment.
Predicted: Great
A MBIG-ICL + GOLD 1. Good: Even with all its botches, Enigma offers all the pleasure of a handsome and well-made entertainment. 2. Great: A breathtaking adventure for all ages, Spirit tells its poignant and uplifting story in a stunning fusion of music and images. 3. Great: A rollicking ride, with jaw-dropping action sequences, striking villains, a gorgeous color palette, astounding technology, stirring music and a boffo last hour that leads up to a strangely sinister happy ending. 4. Great: This gorgeous epic is guaranteed to lift the spirits of the whole family.
Predicted: Great
+ GOLD + MIS 1. Good: As action-adventure, this space-based homage to Robert Louis Stevenson’s Treasure Island fires on all plasma conduits. 2. Good: Horns and Halos benefits from serendipity but also reminds us of our own responsibility to question what is told as the truth. 3. Great: Return to Never Land is reliable, standard Disney animated fare, with enough creative energy and wit to entertain all ages. 4. Great: It’s a smart, solid, kinetically-charged spy flick worthy of a couple hours of summertime and a bucket of popcorn.
Predicted: Great
+ GOLD + MIS + PRED 1. Good: As action-adventure, this space-based homage to Robert Louis Stevenson’s Treasure Island fires on all plasma conduits. 2. Good: Horns and Halos benefits from serendipity but also reminds us of our own responsibility to question what is told as the truth. 3. Great: Return to Never Land is reliable, standard Disney animated fare, with enough creative energy and wit to entertain all ages. 4. Great: It’s a smart, solid, kinetically-charged spy flick worthy of a couple hours of summertime and a bucket of popcorn.

Table 15: Example demonstrations selected by the RETR and our proposed method A MBIG-ICL for the SST task, for n = 4. The model used here for prediction is Flan-PaLM 2 (L).

