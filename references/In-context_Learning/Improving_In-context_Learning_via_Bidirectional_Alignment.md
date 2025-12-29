# Improving In-Context Learning via Bidirectional Alignment
Chengwei Qin∗†, Wenhan Xia∗♣, Fangkai Jiao†, Chen Chen†, Yuchen Hu†, Bosheng Ding†, Shafiq Joty†♠ †Nanyang Technological University,♣Princeton University,♠Salesforce Research
# Abstract
Large language models (LLMs) have shown impressive few-shot generalization on many tasks via in-context learning (ICL). Despite their success in showing such emergent abilities, the scale and complexity of larger models also lead to unprecedentedly high computational demands and deployment challenges. In reaction, researchers explore transferring the powerful capabilities of larger models to more efficient and compact models by typically aligning the output of smaller (student) models with that of larger (teacher) models. Existing methods either train student models on the generated outputs of teacher models or imitate their token-level probability distributions. However, these distillation methods pay little to no attention to the input, which also plays a crucial role in ICL. Based on the finding that the performance of ICL is highly sensitive to the selection of demonstration examples, we propose Bidirectional Alignment (BiAlign) to fully leverage the models’ preferences for ICL examples to improve the ICL abilities of student models. Specifically, we introduce the alignment of input preferences between student and teacher models by incorporating a novel ranking loss, in addition to aligning the token-level output distribution. With extensive experiments and analysis, we demonstrate that BiAlign can consistently outperform existing baselines on a variety of tasks involving language understanding, reasoning, and coding.
# 1 Introduction
With the recent advancements in model scale and pretraining data, large language models (LLMs) have demonstrated impressive few-shot learning capabilities via in-context learning (ICL). With ICL, the LLM generates an output for a given query by conditioning on a few demonstration examples and optionally a task description, and it does so without any parameter updates [3]. Despite the success of ICL in few-shot generalization, the high computational demands and deployment challenges posed by the size of the LLMs hinder their widespread application. Serving an LLM with 175B parameters requires at least 350GB GPU memory [22], which is far beyond what is affordable in most real-world settings. Also, the serving cost increases with model size – it costs 1-2 FLOPs per parameter to infer on one token [28]. To alleviate this issue, researchers have proposed a number of methods to transfer the emergent capabilities of larger (teacher) models to more efficient and compact smaller (student) models, an approach commonly known as knowledge distillation [21]. In this approach, the student models are trained to align their output space with that of the teachers. This is typically achieved by either
∗Equal contribution
Preprint. Under review.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2def/2deff25e-858d-4a01-a7f3-77d5b17a866a.png" style="width: 50%;"></div>
Figure 1: Comparison between different types of approaches to aligning student models. Existing methods typically fine-tune student models on generated outputs of teacher models or to match their token-level output probability distributions (left part). In contrast, our method (BiAlign) considers the models’ preferences for different inputs (the more helpful an input is for generating the target, the more the model prefers that input) to achieve input preference alignment (right part).
training on the generated outputs of the teacher models [22, 60, 69] or by imitating their token-level probability distributions [1, 18, 25].2
While existing distillation methods demonstrate improved ICL results, they pay little attention to the input, specifically the demonstrations, which have been shown to have a significant impact on the performance of ICL [75, 68, 45]. Indeed, selecting different sets of demonstration examples can yield performance ranging from almost random to better than state-of-the-art fine-tuned models [16, 35], indicating that the model has different preferences for different inputs. Inspired by this finding, we propose Bidirectional Alignment (BiAlign), a simple yet effective framework for improving the ICL abilities of student models (Figure 1). Specifically, BiAlign introduces the alignment of input preferences between student and teacher models through the incorporation of a novel ranking loss, in addition to aligning the token-level output distributions. Our main hypothesis is that for an effective knowledge distillation, the student model should align with not only the teacher model’s output distribution but also its input preference (i.e., the more helpful an input is for generating the target, the more the model prefers that input). BiAlign allows student models to obtain more fine-grained supervision from teacher models by fully leveraging their preferences for different demonstrations in ICL. Empirical results on tasks spanning language understanding, symbolic reasoning, mathematical reasoning, logical reasoning, and coding show that BiAlign can consistently outperform previous baselines. In summary, our main contributions are:
• To the best of our knowledge, we for the first time consider aligning student models with teacher models from an input preference perspective. We propose Bidirectional Alignment (BiAlign) to fully leverage the models’ preferences for different demonstration examples to improve the ICL capabilities of student models. • With extensive experiments and analysis, we demonstrate the effectiveness of BiAlign on a variety of tasks. For example, it brings about 18% relative improvement on GSM8K [9] and 13% on LogiQA [34].
# 2 Related Work
This work concerns how to improve the ICL ability of student models by aligning the student and teacher models’ preferences for different few-shot demonstrations. In light of this, we review three lines of work that form the basis of this work: few-shot learning, in-context learning, and alignment.
2Different from the conventional strong-to-weak generalization, [4] recently introduces weak-to-strong generalization, which explores leveraging weaker (smaller) models to elicit “superalignment” from the stronger (larger) models. This paper however considers the conventional strong-to-weak approach.
Few-shot learning (FSL) aims to learn tasks with only a few labeled examples, which faces the challenge of over-fitting due to the scarcity of labeled training data. Existing methods to address this challenge can be mainly divided into three categories: (i) reducing the hypothesis space with prior knowledge [57, 23], (ii) optimizing the strategy for searching the best hypothesis in whole space [49, 15], and (iii) augmenting the few-shot data [17, 44, 13]. More recently, LLMs have achieved promising performance on various few-shot tasks via in-context learning (ICL).
# 2.2 In-context Learning (ICL)
By conditioning on a prompt that includes several demonstration examples and optionally a task description, a frozen LLM, by virtue of ICL, showcases impressive few-shot generalization [3]. ICL has drawn a great deal of attention from the research community in recent days. [7, 38, 65] have explored ways to enhance the ICL capabilities of language models by either self-supervised or supervised training. In parallel, extensive analytical studies have been conducted to understand factors influencing the performance of ICL [75, 62, 53, 72, 39, 66], as well as to elucidate the underlying mechanisms that contribute to the success of ICL [40, 68, 42, 29, 10]. Furthermore, there is a series of ongoing research dedicated to various aspects of ICL: (i) demonstration designing strategies, including demonstration organization [33, 50, 59, 45] and demonstration formatting[64, 60, 74, 77], (ii) multi-modal ICL [24, 61, 58, 78], and (iii) applications of ICL [12, 36, 76].
# 2.3 Alignment
Existing work on alignment can be mainly divided into two parts based on the objectives: aligning with humans and aligning with teacher models. To align with humans, reinforcement learning from human feedback (RLHF) [8, 41] explores how human feedback can be used to train language models to better align with human preferences and values using reinforcement learning algorithms such as PPO [52]. Several recent studies have introduced lightweight alternatives of PPO, including RRHF [73], DPO [46], ReMax [31], IPO [2] and KTO [14]. Alignment with teacher models, also known as distillation [21], aims to transfer the powerful capabilities of large teacher models to more efficient and compact student counterparts. It has emerged as a powerful solution to reduce the high computational demands and serving challenges posed by large models. Current distillation methods typically train student models on generated outputs of teacher models [22, 60, 69] or to imitate teacher models’ token-level probability distributions [51, 27, 1, 18, 25], i.e., these approaches focus on aligning the output of student models with that of teachers. However, they pay little attention to the input demonstrations which also significantly influence the performance of ICL [45]. In contrast to these methods, our proposed method (BiAlign) leverages the models’ preferences for different in-context examples to achieve input preference alignment.
# 3 Methodology
# 3.1 Problem Setting
Given a training set Dtrain consisting of a set of source tasks T src, the goal of ICL alignment is to align the ICL ability of a student model S with that of a teacher model T. Upon successful alignment, the model S is expected to show improved ICL ability on a held-out set of target tasks T tgt. We divide the whole process into two stages, as illustrated in Figure 2.
• Upstream ICL alignment on T src: In this alignment stage, the model has access to T src. We formalize samples in Dtrain in the k-shot ICL format { ˆXi = (x1, y1), ..., (xk, yk), (ˆxi, ˆyi)}, where (xj, yj), 1 ≤j ≤k denotes the k demonstration
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/109a/109aea9b-9220-46f1-b881-abad8f5664d2.png" style="width: 50%;"></div>
Figure 2: In the upstream ICL alignment stage, we align a student model with a teacher on the source tasks. Then in the downstream evaluation stage, we evaluate the ICL performance of the aligned student model on a held-out set of target tasks, which are different from the source tasks.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/46be/46be588a-b4b0-4c95-9e7a-d757fbb8b953.png" style="width: 50%;"></div>
Figure 3: Illustration of our Bidirectional Alignment (BiAlign) framework. It attains token-level output distribution alignment by minimizing the KL divergence loss between the student and teacher models on the constructed ICL samples. Furthermore, after sampling several subsets from the set of all demonstrations, it optimizes a ranking loss for input preference alignment to align the student and teacher models’ preferences for different demonstration examples.
examples and (ˆxi, ˆyi) is the test sample. We concatenate these examples to form an ICL training sample ˆXi. We then align the student model S with the teacher model T on this formatted ICL data. • Downstream ICL evaluation on T tgt: Following the upstream ICL alignment stage, we evaluate the ICL ability of the aligned model S∗on T tgt, where T tgt has no overlap with T src. For every target task Tk, we use the default ICL demonstrations by convention to evaluate the model performance.
# 3.2 Bidirectional Alignment (BiAlign)
Based on the finding that the performance of ICL is highly sensitive to the selection of demonstration examples [75], we propose Bidirectional Alignment (BiAlign) to fully leverage the models’ preferences for different demonstration examples with the goal of improving the ICL ability of the student model. Figure 3 illustrates our approach.
Aligning Token-level Distributions Given the ICL training examples in the concatenated form { ˆXi = (x1, y1), ..., (xk, yk), (ˆxi, ˆyi)} as discussed above, to achieve token-level output distribution alignment on ˆXi, we minimize a KL divergence loss between the student model and teacher model for the whole sequence instead of only ˆyi following [19].3 More formally,
where m is the number of ICL training samples in Dtrain, t is the number of tokens in ˆXi, V is the models’ common vocabulary of tokens; θT and θS are the parameters of the teacher model and the student model, respectively.
Aligning Preferences for Demonstrations Intuitively, for the student and teacher models to be well-aligned, the demonstrations preferred by the teacher model should also be preferred by the
(1)
student, i.e., to truly emulate the teacher model, the student needs to learn “what to output” as well as “which input demonstrations should be preferred” in order to generate high-quality outputs. This is similar in spirit to the scenario where a well-trained reward model can rank model responses just like human experts (responses that humans consider good or bad should also be recognized or criticized by the reward model) [41]. To this end, we introduce input preference alignment to align the student and teacher models’ preferences for different demonstrations. For simplicity, let Ri = {(x1, y1), ..., (xk, yk)} denote the k-shot demonstrations in each ICL training sample ˆXi = (x1, y1), ..., (xk, yk), (ˆxi, ˆyi). To rank the model’s preferences for different demonstration examples, we first need to obtain a set Drank = {Rij, (ˆxi, ˆyi)}N j=1, where Rij is a subset of Ri and N is the number of subsets considered for ranking. Modeling on the full subset space of Ri can be computationally prohibitive as it grows exponentially with |Ri|. Therefore, we set N ≪|P(Ri)|, where P(Ri) is the power set of Ri. We randomly sample N distinct subsets from P(Ri). Note that the number of demonstrations in different subsets can be different. We use both the student and teacher models to measure their preferences for each subset Rij, which we estimate using the prediction probability of ˆyi given Rij and ˆxi as input:4
where QT and QS are the preference scores of the teacher and student models, respectively. Intuitively, the more helpful the subset Rij is for generating the target ˆyi, the more the model prefers this subset. To align the preferences of the student and teacher models for different subsets, we introduce a novel ranking loss:
� �� � � �� � (3) where Rall i = {Rij}N j=1 contains all subsets sampled for the test example (ˆxi, ˆyi), (R+, R−) refers to the pair of positive and negative subsets determined by the preference score of the teacher model (the subset with the higher preference score is considered as the positive one), and rank() stands for the function that measures the relative ranking of subset scores which ranges from 1 to N. The left part of Lrank measures the difference in preference scores of the student model for the pair (R+, R−) and the right part reflects the relative ranking difference between R+ and R−. Therefore, Lrank allows the student model to obtain more fine-grained supervision from the teacher model by matching the relative ranking of their preference scores for different demonstration examples in ICL. The overall loss that BiAlign optimizes for alignment is: L = LKL + λLrank, where λ is the weight of the ranking loss. Besides, we illustrate the whole learning process in Algorithm 1.
# 4 Experimental Setup
In this section, we first describe the tasks and datasets, and then introduce methods compared in our work. Finally, we present the implementation details.
# 4.1 Tasks and Datasets
In this work, we use CrossFit [71], a large and diverse collection of few-shot tasks covering various types including classification, question answering and generation, as the source tasks T src. For each task in CrossFit, we combine the original training and validation data as the new training data which is then randomly partitioned into a set of ICL samples with 4 ≤k ≤10 demonstration examples.
4Under the assumption that the prior P(Rij|ˆxi, θ) is uniform, it is easy to show using the Bayes rule Q(Rij) ∝P(Rij|ˆyi, ˆxi, θ) = P (ˆyi|Rij,ˆxi,θ)P (Rij|ˆxi,θ) � j P (ˆyi|Rij,ˆxi,θ)P (Rij|ˆxi,θ)
� (3)
Algorithm 1 Learning process of BiAlign
Input: ICL training set DICL = { ˆXi = (x1, y1), ..., (xk, yk), (ˆxi, ˆyi)}, teacher model θT , student
model θS, number of subsets N, weight of ranking loss λ
1: for mini-batch B in DICL do
2:
CALCULATE the KL divergence loss LKL on B using Equation 1
3:
for ˆXi ∈B do
4:
SAMPLE N subsets {Rij}N
j=1 for the test sample (ˆxi, ˆyi)
5:
MEASURE preferences QT and QS for {Rij}N
j=1 using Equation 2
6:
end for
7:
CALCULATE the ranking loss Lrank on B using Equation 3
8:
UPDATE θS by back-propagating with L = LKL + λLrank
9: end for
For each ICL example, we randomly sample N = 4 subsets from the set of all demonstrations for calculating the ranking loss. After the preprocessing, we obtain 12K ICL examples in total. We evaluate the ICL performance of the aligned model on 5 target tasks spanning language understanding, symbolic reasoning, mathematical reasoning, logical reasoning, and coding: MMLU [20], BBH [54], GSM8K [9], LogiQA [34] and HumanEval [6]. Note that there is no overlap between CrossFit and target tasks, and we obtain all outputs from the models using greedy decoding following [70]. Details of different target tasks are provided in Appendix A.3.
# 4.2 Methods Compared
We mainly experiment with Llama 2-7B [56] as the student model and Llama 2-13B or 70B as the teacher model. For Llama 2-70B, we use the quantized version TheBloke/Llama-2-70B-GPTQ [55] due to resource constraints. We compare BiAlign with the following methods: • Vanilla simply evaluates the ICL performance of the student model on target tasks without any alignment, serving as the baseline for all other approaches. • Fine-tuning (FT) tunes the student model on the 12K ICL examples constructed from CrossFit using a multi-task learning scheme, which is indeed the meta-training in [38]. • Continual Pretraining (C-Pretrain) simply performs continual pretraining, i.e., next token prediction for the whole sequence, of the student model on the 12K samples. • Output Alignment (Output-Align) trains the student model to align token-level output distributions with the teacher model [25].
# 4.3 Implementation Details
Our methods are implemented with the PyTorch and Transformers library [67]. Model training is conducted utilizing DeepSpeed [48, 47] on 4 NVIDIA A100 GPUs. During the training phase, we set the learning rate to 1e−6 and the batch size to 64. The weight λ for the ranking loss is set to 1.0 For evaluation, we train the student model on the constructed ICL data for 10 epochs and assess the final checkpoint.
# 5 Results and Analysis
# 5.1 Main Results
Table 1 shows the performance scores of different methods on all investigated target tasks. From the results, we can observe that
• Our proposed BiAlign consistently outperforms baseline approaches on all datasets with different sizes of teacher models, demonstrating its superiority. Simply pretraining the model on source tasks does not improve the average performance since there is no overlap between source and target tasks. While fine-tuning brings marginal improvement, token-level output distribution alignment
Table 1: Performance (%) of different methods on 5 target tasks. We use Llama 2-7B as a student and Llama 2-13B or 70B as a teacher model. The rows with “Teacher" (grey) indicate the corresponding teacher model’s performance on the target tasks. Bold indicates the best result for Llama 2-7B (student). BiAlign is consistently better than all previous baselines.
Method
MMLU
BBH
GSM8K
LogiQA
HumanEval
Average
No Alignment Baselines
Vanilla
45.4
39.5
15.2
30.3
14.6
29.0
FT
46.2
39.7
15.6
31.5
14.2
29.4
C-Pretrain
46.2
38.5
16.0
31.2
13.2
29.0
Llama 2-13B Teacher
Teacher
55.3
47.8
27.8
37.8
18.3
37.4
Output-Align
46.3
38.9
14.8
32.0
14.0
29.2
BiAlign
46.5
40.3
16.3
32.7
15.2
30.2
Llama 2-70B Teacher
Teacher
67.2
64.2
53.3
48.0
26.8
51.9
Output-Align
46.9
39.3
16.1
32.9
14.6
30.0
BiAlign
48.5
41.8
17.9
34.2
15.9
31.7
Table 4: Average results (%) across 5 tasks of all methods with two different backbones. We use Llama 2-13B as the teacher for Llama 2-7B and Phi-2 (2.7B) as the teacher for Phi-1.5 (1.3B).
Table 3: Average results (%) of Output-Align and BiAlign with different sizes of student models (Llama 2-70B as the teacher).
Method
7B
13B
Output-Align
30.0
38.3
BiAlign
31.7
39.6
Method
Vanilla
FT
C-Pretrain
Output-Align
BiAlign
Llama 2-7b
29.0
29.4
29.0
29.2
30.2
Phi-1.5 (1.3B)
33.6
33.9
33.4
33.8
34.9
with a stronger (70B) teacher model can achieve better performance. Thanks to incorporating input preference alignment, BiAlign yields about 1.2% performance boost on average when using a 13B teacher model, and this gain is 2.7% for a 70B teacher. Besides, when examining the effects of scaling up the teacher model, the performance of BiAlign sees an improvement on all tasks. • In particular, BiAlign using a 13B teacher model achieves relative performance improvements of 7.9% on LogiQA and 7.2% on GSM8K compared with Vanilla, while using the 70B teacher, it achieves 12.9% on LogiQA and 17.8% on GSM8K. These results indicate that BiAlign can better improve the performance of tasks requiring more fine-grained reasoning; see appendix A.10 for an example in LogiQA. This is because BiAlign allows the student model to obtain more fine-grained supervision from the teacher model by fully leveraging their preferences for different inputs.
with a stronger (70B) teacher model can achieve better performance. Thanks to incorporating input preference alignment, BiAlign yields about 1.2% performance boost on average when using a 13B teacher model, and this gain is 2.7% for a 70B teacher. Besides, when examining the effects of scaling up the teacher model, the performance of BiAlign sees an improvement on all tasks.
• In particular, BiAlign using a 13B teacher model achieves relative performance improvements of 7.9% on LogiQA and 7.2% on GSM8K compared with Vanilla, while using the 70B teacher, it achieves 12.9% on LogiQA and 17.8% on GSM8K. These results indicate that BiAlign can better improve the performance of tasks requiring more fine-grained reasoning; see appendix A.10 for an example in LogiQA. This is because BiAlign allows the student model to obtain more fine-grained supervision from the teacher model by fully leveraging their preferences for different inputs.
To better support our claim, we further conduct experiments on four mathematical reasoning tasks ranging from low to high difficulty: ASDiv [37], SVAMP [43], GSM8K [9], and AQUA-RAT [32]. The comparison between BiAlign and Vanilla, as illustrated in Table 2, demonstrates that BiAlign is indeed more beneficial for more complex reasoning tasks.
<div style="text-align: center;">Table 2: Relative gain (%) of BiAlign on math reasoning tasks of varying difficulty levels.</div>
ASDiv
SVAMP
GSM8K
AQUA-RAT
Vanilla
46.6
41.2
15.2
24.4
BiAlign
48.5
42.7
16.3
26.4
Relative Gain
4.1
3.6
7.2
8.2
• Both fine-tuning and output alignment sometimes hurt the zero-shot learning capability of the model as shown by the performance on HumanEval. In contrast, BiAlign brings an average relative improvement of about 4% on HumanEval. We speculate that this is due to the subset sampling in input preference alignment, which helps the model generalize better to the unseen zero-shot setting.
# 5.2 Analysis
Larger Student Model We further experiment with a larger student model to verify the effectiveness of BiAlign. Specifically, we use Llama 2-13B as the student model and Llama 2-70B as the teacher model. We employ QLoRA [11] to fine-tune the student model with consideration of computational resource limitations. The results averaged over the 5 tasks are reported in Table 3, which demonstrate the consistent superiority of BiAlign across model scales.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a1b4/a1b45a5f-c8fc-484c-a5ad-700da6d85912.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Preference score consistency (%) of different methods. Figure 5: Average performance (%) of BiAlign with different numbers of source tasks.</div>
Different Backbone Models Our experiments and analysis so far use Llama 2 as the backbone model. To verify whether the performance gain of BiAlign is consistent across different backbone models, we extend the experiments to the Phi series. Specifically, we use Phi-1.5 (1.3B) [30] as the student model and Phi-2 (2.7B) as the teacher model. From the average results shown in Table 4, we can see that BiAlign still outperforms all baseline approaches when using another language model as the backbone, showing its robustness to model types. Comment on Training-time Computational Overhead Smaller models are a preferred choice for resource-constrained deployments, where the inference cost matters the most. BiAlign does not introduce any additional cost during inference. The additional computational overhead only occurs once during model training. To quantify the increase in computational overhead caused by the ranking loss, we use DeepSpeed Flops Profiler [48] to calculate the training FLOPs of Output-Align and BiAlign, which are 8.1×1017 and 1.9×1018 respectively (about 2.3 times). Therefore, we further design two experiments to compare BiAlign and Output-Align under the same training FLOPs: (i) we combine the original ICL training examples with the sampled subset data and conduct Output-Align on the combined data (roughly the same FLOPs as BiAlign), which performs (29.3) similarly to the original Output-Align method (29.2), verifying the superiority of BiAlign; (ii) we reduce the training epochs of BiAlign from 10 to 5 (roughly the same FLOPs as Output-Align) and assess the final checkpoint. There is no significant performance degradation (from 30.2 to 30.0), which also demonstrates that BiAlign can outperform baselines under the same training FLOPs. Different Numbers of Subsets While we use N = 4 subsets for calculating the ranking loss, we also evaluate the effectiveness of BiAlign with different N. Specifically, we conduct controlled experiments with {3, 5, 6} subsets and report the average results of the 5 target tasks in Table 5. We can observe that increasing the number of subsets does not always improve performance. BiAlign achieves the best performance (30.3) with 6 subsets and the performance with 4 subsets (30.2) is comparable. Besides, all variants consistently outperform baseline methods in Table 1, demonstrating the effectiveness of our designed input preference alignment. Effect of Demonstration Numbers As mentioned in Section 4.1, each constructed ICL training sample contains 4 ≤k ≤10 demonstration examples, which could enhance the model’s ability to generalize to different numbers of demonstrations. To investigate the effect of demonstration numbers in source tasks, we further conduct training on examples containing a fixed number k ∈{5, 8, 10} of demonstrations. The average results of the 5 target tasks are reported in Table 6. We can see that training with a fixed number of demonstrations results in performance degradation to a certain degree, justifying our training set construction strategy. Preference Score Consistency As illustrated in Section 3.2, Lrank enables the student model to match the relative ranking of the preference scores for different ICL demonstrations with that of the teacher model. To verify this, we report the preference score consistency comparison between
Preference Score Consistency As illustrated in Section 3.2, Lrank enables the student model to match the relative ranking of the preference scores for different ICL demonstrations with that of the teacher model. To verify this, we report the preference score consistency comparison between BiAlign and Output-Align in Figure 4. Specifically, we randomly select 500 examples from MMLU. For each example, we randomly sample 5 subsets from the set of all demonstrations and obtain their preference scores using different models. The preference score consistency of different methods is then calculated as the proportion of the highest/lowest scoring subsets that are consistent between the corresponding student model and the teacher model. From the results, we can see that BiAlign
Table 6: Average results (%) of BiAlign with different k (demonstration number) for constructed ICL training samples.
Table 5: Average performance (%) of BiAlign with different numbers of subsets N.
Method
Subset number
3
4
5
6
BiAlign
29.9
30.2
30.0
30.3
Method
Demonstration number
Default (4 ≤k ≤10)
5
8
10
BiAlign
30.2
30.0
29.7
29.8
Table 7: Output examples of Vanilla and BiAlign on GSM8K. The upper part is the input to LLMs, including few-shot demonstrations, and the lower part is the generated response. We color correct outputs in green and wrong outputs in red.
Answer the following questions.
Few-shot demonstrations
Question: Lindsay is doing the laundry, and thinks she has missed some socks. There are 50 socks that
need washing. If she washes 10 pairs of socks and 15 loose socks, how many socks has Lindsay missed?
Answer:
Vanilla: Lindsay started with 50 socks. She washed 10 pairs of socks and 15 loose socks. So she had 50 -
10 - 15 = 25 socks left. So the answer is 25. ✗
BiAlign: Lindsay started with 50 socks. She washed 10 pairs of socks and 15 loose socks. 10 pairs of
socks is 10 * 2 = 20 socks. 15 loose socks is 15 * 1 = 15 socks. So she missed 50 - 20 - 15 = 15 socks. So
the answer is 15. ✓
can indeed achieve much higher preference score consistency than Output-Align, indicating the effectiveness of Lrank.
Different Source Task Selections We hypothesize that the diversity of source tasks has a considerable influence on target task performance. To verify this, we study the effect of the number of source tasks by conducting controlled experiments on {50, 100} randomly selected source tasks. From the results in Figure 5, we can observe that the performance of BiAlign keeps improving as the number of source tasks increases, indicating the importance of source task diversity.
Different Ranking Loss Formulations We experiment with a different ranking loss formulation (Equation 4) to better verify the effectiveness of BiAlign. All other implementation details are the same. The average results reported in Table 8 demonstrate that BiAlign is robust to different ranking loss choices.
Case Study We select GSM8K as a representative task and show several examples of output in Table 7. Compared with Vallina, BiAlign is able to generate more precise and fine-grained reasoning paths, e.g., BiAlign can successfully understand the meaning of ‘pair’ and generate the rationale ‘10 pairs of socks is 10 * 2 = 20 socks’ while Vallina fails to do so. In addition, for interested readers, we also show the results with different subset sampling methods, the analysis of KL divergence calculation, training steps and additional training data, the influence of ranking loss weight, and the effect of contrastive pair selection in Appendix A.4 ∼A.9, respectively.
# 6 Conclusion
In this work, we have introduced Bidirectional Alignment (BiAlign) that can improve the ICL capabilities of student models by aligning the input preferences between student and teacher models in addition to aligning the token-level output distributions. Extensive experimental results and analysis show that BiAlign consistently outperforms previous baseline approaches.
Table 8: Average results (%) of BiAlign with different ranking loss formulations.
Default
Variant
BiAlign
30.2
30.4
(4)
[1] Rishabh Agarwal, Nino Vieillard, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. Gkd: Generalized knowledge distillation for auto-regressive sequence models. arXiv preprint arXiv:2306.13649, 2023. [2] Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pages 4447–4455. PMLR, 2024. [3] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. [4] Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, et al. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. arXiv preprint arXiv:2312.09390, 2023. [5] Stephanie C.Y. Chan, Adam Santoro, Andrew Kyle Lampinen, Jane X Wang, Aaditya K Singh, Pierre Harvey Richemond, James McClelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers. In Advances in Neural Information Processing Systems, 2022. [6] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. [7] Mingda Chen, Jingfei Du, Ramakanth Pasunuru, Todor Mihaylov, Srini Iyer, Veselin Stoyanov, and Zornitsa Kozareva. Improving in-context few-shot learning via self-supervised training. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3558–3573, Seattle, United States, July 2022. Association for Computational Linguistics. [8] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017. [9] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 10] Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers. In ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models, 2023. 11] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023. 12] Bosheng Ding, Chengwei Qin, Linlin Liu, Lidong Bing, Shafiq Joty, and Boyang Li. Is gpt-3 a good data annotator? arXiv preprint arXiv:2212.10450, 2022. 13] Bosheng Ding, Chengwei Qin, Linlin Liu, Yew Ken Chia, Boyang Li, Shafiq Joty, and Lidong Bing. Is GPT-3 a good data annotator? In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11173–11195, Toronto, Canada, July 2023. Association for Computational Linguistics. 14] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024. 15] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pages 1126–1135. PMLR, 2017. 16] Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot learners. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3816–3830, Online, August 2021. Association for Computational Linguistics. 17] Tianyu Gao, Xu Han, Ruobing Xie, Zhiyuan Liu, Fen Lin, Leyu Lin, and Maosong Sun. Neural snowball for few-shot relation learning. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 7772–7779. AAAI Press, 2020.
[18] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Knowledge distillation of large language models. arXiv preprint arXiv:2306.08543, 2023. [19] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Pre-training to learn in context. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4849–4870, Toronto, Canada, July 2023. Association for Computational Linguistics. [20] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021. [21] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015. [22] Cheng-Yu Hsieh, Chun-Liang Li, Chih-kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8003–8017, Toronto, Canada, July 2023. Association for Computational Linguistics. [23] Zikun Hu, Xiang Li, Cunchao Tu, Zhiyuan Liu, and Maosong Sun. Few-shot charge prediction with discriminative legal attributes. In Proceedings of the 27th International Conference on Computational Linguistics, pages 487–498, Santa Fe, New Mexico, USA, August 2018. Association for Computational Linguistics. [24] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al. Language is not all you need: Aligning perception with language models. arXiv preprint arXiv:2302.14045, 2023. [25] Yukun Huang, Yanda Chen, Zhou Yu, and Kathleen McKeown. In-context learning distillation: Transferring few-shot learning ability of pre-trained language models. arXiv preprint arXiv:2212.10670, 2022. [26] Fangkai Jiao, Zhiyang Teng, Shafiq Joty, Bosheng Ding, Aixin Sun, Zhengyuan Liu, and Nancy F Chen. Logicllm: Exploring self-supervised logic-enhanced training for large language models. arXiv preprint arXiv:2305.13718, 2023. [27] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. TinyBERT: Distilling BERT for natural language understanding. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4163–4174, Online, November 2020. Association for Computational Linguistics. [28] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. [29] Yingcong Li, Muhammed Emrullah Ildiz, Dimitris Papailiopoulos, and Samet Oymak. Transformers as algorithms: Generalization and stability in in-context learning. 2023. [30] Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023. [31] Ziniu Li, Tian Xu, Yushun Zhang, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient method for aligning large language models. arXiv preprint arXiv:2310.10505, 2023. [32] Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 158–167, Vancouver, Canada, 2017. Association for Computational Linguistics. [33] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pages 100–114, Dublin, Ireland and Online, May 2022. Association for Computational Linguistics. [34] Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. arXiv preprint arXiv:2007.08124, 2020. [35] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland, May 2022. Association for Computational Linguistics. [36] Nicholas Meade, Spandana Gella, Devamanyu Hazarika, Prakhar Gupta, Di Jin, Siva Reddy, Yang Liu, and Dilek Hakkani-Tür. Using in-context learning to improve dialogue safety. arXiv preprint arXiv:2302.00871, 2023.
[37] Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing English math word problem solvers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 975–984, Online, July 2020. Association for Computational Linguistics. [38] Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. MetaICL: Learning to learn in context. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2791–2809, Seattle, United States, July 2022. Association for Computational Linguistics. [39] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. [40] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022. [41] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022. [42] Jane Pan. What In-Context Learning “Learns” In-Context: Disentangling Task Recognition and Task Learning. PhD thesis, Princeton University, 2023. [43] Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2080–2094, Online, 2021. Association for Computational Linguistics. [44] Chengwei Qin and Shafiq Joty. Continual few-shot relation learning via embedding space regularization and data augmentation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2776–2789, Dublin, Ireland, May 2022. Association for Computational Linguistics. [45] Chengwei Qin, Aston Zhang, Anirudh Dagar, and Wenming Ye. In-context learning with iterative demonstration selection. arXiv preprint arXiv:2310.09881, 2023. [46] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023. [47] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020. [48] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3505–3506, 2020. [49] Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. [50] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2655–2671, Seattle, United States, July 2022. Association for Computational Linguistics. [51] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019. [52] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. [53] Seongjin Shin, Sang-Woo Lee, Hwijeen Ahn, Sungdong Kim, HyoungSeok Kim, Boseop Kim, Kyunghyun Cho, Gichang Lee, Woomyoung Park, Jung-Woo Ha, and Nako Sung. On the effect of pretraining corpora on in-context learning by a large-scale language model. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5168–5186, Seattle, United States, July 2022. Association for Computational Linguistics. [54] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022. [55] TheBloke. Thebloke/llama-2-70b-gptq: Gptq model for meta’s llama 2 70b, 2023.
[56] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. [57] Eleni Triantafillou, Richard Zemel, and Raquel Urtasun. Few-shot learning through an information retrieval lens. arXiv preprint arXiv:1707.02610, 2017. [58] Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, et al. Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111, 2023. [59] Xinyi Wang, Wanrong Zhu, and William Yang Wang. Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning. arXiv preprint arXiv:2301.11916, 2023. [60] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022. [61] Zhendong Wang, Yifan Jiang, Yadong Lu, Yelong Shen, Pengcheng He, Weizhu Chen, Zhangyang Wang, and Mingyuan Zhou. In-context learning unlocked for diffusion models. arXiv preprint arXiv:2305.01115, 2023. [62] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022. Survey Certification. [63] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS 2022), 2022. [64] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022. [65] Jerry Wei, Le Hou, Andrew Lampinen, Xiangning Chen, Da Huang, Yi Tay, Xinyun Chen, Yifeng Lu, Denny Zhou, Tengyu Ma, et al. Symbol tuning improves in-context learning in language models. arXiv preprint arXiv:2305.08298, 2023. [66] Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, et al. Larger language models do in-context learning differently. arXiv preprint arXiv:2303.03846, 2023. [67] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, October 2020. Association for Computational Linguistics. [68] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. In International Conference on Learning Representations, 2022. [69] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023. [70] Yiheng Xu, Hongjin Su, Chen Xing, Boyu Mi, Qian Liu, Weijia Shi, Binyuan Hui, Fan Zhou, Yitao Liu, Tianbao Xie, et al. Lemur: Harmonizing natural language and code for language agents. arXiv preprint arXiv:2310.06830, 2023. [71] Qinyuan Ye, Bill Yuchen Lin, and Xiang Ren. CrossFit: A few-shot learning challenge for cross-task generalization in NLP. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7163–7189, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. [72] Kang Min Yoo, Junyeob Kim, Hyuhng Joon Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-Woo Lee, Sang-goo Lee, and Taeuk Kim. Ground-truth labels matter: A deeper look into input-label demonstrations. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2422–2437, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. [73] Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. Rrhf: Rank responses to align language models with human feedback without tears. arXiv preprint arXiv:2304.05302, 2023.
[56] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. [57] Eleni Triantafillou, Richard Zemel, and Raquel Urtasun. Few-shot learning through an information retrieval lens. arXiv preprint arXiv:1707.02610, 2017. [58] Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, et al. Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111, 2023. [59] Xinyi Wang, Wanrong Zhu, and William Yang Wang. Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning. arXiv preprint arXiv:2301.11916, 2023. [60] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022. [61] Zhendong Wang, Yifan Jiang, Yadong Lu, Yelong Shen, Pengcheng He, Weizhu Chen, Zhangyang Wang, and Mingyuan Zhou. In-context learning unlocked for diffusion models. arXiv preprint arXiv:2305.01115, 2023. [62] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022. Survey Certification. [63] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS 2022), 2022. [64] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022. [65] Jerry Wei, Le Hou, Andrew Lampinen, Xiangning Chen, Da Huang, Yi Tay, Xinyun Chen, Yifeng Lu, Denny Zhou, Tengyu Ma, et al. Symbol tuning improves in-context learning in language models. arXiv preprint arXiv:2305.08298, 2023. [66] Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, et al. Larger language models do in-context learning differently. arXiv preprint arXiv:2303.03846, 2023. [67] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, October 2020. Association for Computational Linguistics. [68] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. In International Conference on Learning Representations, 2022. [69] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023. [70] Yiheng Xu, Hongjin Su, Chen Xing, Boyu Mi, Qian Liu, Weijia Shi, Binyuan Hui, Fan Zhou, Yitao Liu, Tianbao Xie, et al. Lemur: Harmonizing natural language and code for language agents. arXiv preprint arXiv:2310.06830, 2023. [71] Qinyuan Ye, Bill Yuchen Lin, and Xiang Ren. CrossFit: A few-shot learning challenge for cross-task generalization in NLP. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7163–7189, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. [72] Kang Min Yoo, Junyeob Kim, Hyuhng Joon Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-Woo Lee, Sang-goo Lee, and Taeuk Kim. Ground-truth labels matter: A deeper look into input-label demonstrations. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 2422–2437, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. [73] Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. Rrhf: Rank responses to align language models with human feedback without tears. arXiv preprint arXiv:2304.05302, 2023.
[74] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. In The Eleventh International Conference on Learning Representations, 2023. [75] Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving fewshot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 12697–12706. PMLR, 18–24 Jul 2021. [76] Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, and Baobao Chang. Can we edit factual knowledge by in-context learning? arXiv preprint arXiv:2305.12740, 2023. [77] Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers. In The Eleventh International Conference on Learning Representations, 2023. [78] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
Table 11: Average performance (%) of BiAlign using different types of KL divergence calculation methods.
Method
Type
Whole Sequence
Label Only
BiAlign
30.2
30.0
# A Appendix
# A.1 Limitations
As the first work on input preference alignment, one limitation of our paper is the additional computationa overhead introduced by the ranking loss. A further improvement could be to explore more efficient inpu alignment methods to improve the ICL capabilities of student models.
# A.2 Impact Statements
This paper aims to explore improving the ICL abilities of smaller models. There are no direct potential societal consequences as this paper does not release any new model or dataset. There are many potential indirect societal consequences of our work, none of which we feel must be specifically highlighted here.
<div style="text-align: center;">Table 9: Details of different datasets. # refers to ‘the number of’. CrossFit [71] is used to construct training data and other tasks are used for evaluation.</div>
CrossFit
MMLU
BBH
GSM8K
LogiQA
HumanEval
# Samples
12K
15K
6.5K
8.5K
651
164
# Shot
4∼10
5
3
8
5
0
# A.3 Details of Target Tasks
• MMLU: The MMLU (Massive Multitask Language Understanding) benchmark [20] consists of 57 diverse tasks covering various fields like computer science, history and law, aiming to evaluate the knowledge obtained during pretraining. Following its original setup, we use 5-shot ICL demonstrations for evaluation. • BBH: The BBH (BIG-Bench Hard) [54] includes several types of reasoning tasks that are believed to be difficult for current language models. Following the guidelines in [54], we conduct the evaluation using 3-shot ICL demonstration examples with chain-of-thought prompting [63]. • GSM8K: The GSM8K [9] dataset encompasses 8.5K grade school math word problems, aiming to evaluate the multi-step mathematical reasoning capabilities. We evaluate the ICL performance on it using 8-shot in-context examples with chain-of-thought prompting. • LogiQA: LogiQA [34] is a logical reasoning benchmark sourced from logical examination papers intended for reading comprehension. Following [26], we conduct 5-shot evaluation. • HumanEval: HumanEval [6] consists of 164 programming challenges for evaluating coding capabilities. We follow the official zero-shot setting in [6] to verify whether bidirectional alignment hurts the zero-shot learning ability of models.
We summarize the details of all used datasets in Table 9.
# A.4 Different Subset Sampling Methods
To explore the influence of subset sampling methods, we replace the original method ‘Randomly sample N subsets’ with ‘Sample N subsets with different numbers of demonstrations’ which can improve diversity. The comparison between the two methods is shown in Table 10. We can observe that there is no significant performance gap between them.
Table 12: Comparison between BiAlign and Output-Align at different proportions of training steps.
Table 12: Comparison between BiAlign and Output-Align at different proportions of training steps.
Method
25%
50%
100%
Output-Align
28.9
29.1
29.2
BiAlign
29.4
30.0
30.2
Default Variant
BiAlign
30.2
30.1
Table 13: Average performance (%) of BiAlign with different λ for the ranking loss Lrank.
 L
Method
λ
0.2
0.5
1.0
2.0
5.0
BiAlign
30.1
30.4
30.2
30.1
29.4
# A.5 Whole Sequence vs. Label Only
To maintain the basic in-weights capability of the student model, we minimize the KL divergence loss for the whole sequence instead of only the label following [19]. In Table 11, we show the performance comparison between using the whole sequence and using only the label. We can see that using the whole sequence also results in slightly better average performance.
# A.6 Different Proportions of Training Steps
Table 12 reports the performance comparison between BiAlign and Output-Align at different proportions (roughly 25%, 50%, and 100%) of training steps. We can observe that BiAlign consistently outperforms Output-Align at different steps.
# A.7 Additional Training Data
The analysis in Section 5.2 shows that conducting Output-Align on the combination of the original ICL training examples and the sampled subset data achieves similar performance to the original Output-Align method. We further experiment with the fine-tuning approach. However, the performance becomes even worse (from 29.4 to 29.2), once again demonstrating that simply increasing training data does not necessarily lead to better performance.
# A.8 Ranking Loss Weights
To further investigate the influence of the ranking loss Lrank (Equation 3), we conduct experiments with different weights λ and report the results in Table 13. All variants except the variant with λ = 5.0 (too large) outperform baseline approaches by a large margin, which demonstrates the superiority of Lrank.
# A.9 Contrastive Pair Selection
While we use all C(N, 2) (N = 4 is the number of subsets) pairs of positive and negative subsets for input preference alignment, we also investigate the effect of contrastive pair selection. Specifically, we conduct controlled experiments on {3, 4, 5} randomly selected contrastive pairs and report the average results in Table 14. The best performance is achieved using all pairs, justifying our selection strategy.
# A.10 Example in LogiQA
We show the reasoning path required to solve a sample in LogiQA in Table 15, which is quite fine-grained.
Table 14: Average results (%) of BiAlign with different numbers of contrastive pairs.
Method
Pair number
3
4
5
All (6)
BiAlign
29.5
30.0
29.9
30.2
Reasoning path: To evaluate the argument effectively, the focus should be on understanding the impact and justification of the proposed ban on cigarette vending machines, especially in the context of preventing minors from smoking. The argument draws a parallel between the proposed ban and the hypothetical scenario of setting up roadblocks to prevent driving without a license, suggesting that while the measure may target a minority (in this case, underage smokers or unlicensed drivers), it disproportionately inconveniences the majority (licensed drivers or adult smokers).
Therefore, understanding the extent of inconvenience to adult smokers is crucial in determining whethe the proposed solution is proportionate and justified, making Option B the most relevant and importan question for evaluating the argument.
