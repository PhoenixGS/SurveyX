# Life-long Learning for Multilingual Neural Machine Translation with Knowledge Distillation
Yang Zhao1,2, Junnan Zhu1,2, Lu Xiang1,2, Jiajun Zhang
Yu Zhou1,3, Feifei Zhai1,3, and Chengqing Zong1,2 National Laboratory of Pattern Recognition, Institute of Automation, CAS, Beijing, China 2University of Chinese Academy of Sciences, Beijing, China 3Fanyu AI Research, Beijing Fanyu Technology Ltd., Beijing, China {yang.zhao, junnan.zhu, lu.xiang, jjzhang, yzhou, cqzong} @nlpr.ia.ac.cn feifeizhai@zkyf.com
A common scenario of Multilingual Neural Machine Translation (MNMT) is that each translation task arrives in a sequential manner, and the training data of previous tasks is unavailable. In this scenario, the current methods suffer heavily from catastrophic forgetting (CF). To alleviate the CF, we investigate knowledge distillation based life-long learning methods. Specifically, in one-tomany scenario, we propose a multilingual distillation method to make the new model (student) jointly learn multilingual output from old model (teacher) and new task. In many-toone scenario, we find that direct distillation faces the extreme partial distillation problem, and we propose two different methods to address it: pseudo input distillation and reverse teacher distillation. The experimental results on twelve translation tasks show that the proposed methods can better consolidate the previous knowledge and sharply alleviate the CF.
arXiv:2212.02800v1
# 1 Introduction
Recently, multilingual neural machine translation (MNMT) (Dong et al., 2015; Firat et al., 2016; Johnson et al., 2017; Gu et al., 2018; Aharoni et al., 2019) draws much attention due to its remarkable improvement on low-resource language pairs and simple implementation, especially the approach with one universal encoder and decoder (Johnson et al., 2017; Aharoni et al., 2019). The current MNMT methods are studied in the conventional setting that bilingual pairs for all the translation tasks are available at training time. In practice, however, we always face an incremental scenario where each task arrives in a sequential manner. Assuming that we have already built an MNMT model (old model) from English to Italy and Dutch (EN⇒IT, NL), and we hope to extend the model with English-to-Romanian (EN⇒RO)
Model
EN⇒IT
EN⇒NL
EN⇒RO
Initial
28.11
29.79
∼
Fine-tuning
1.08
0.99
25.96
Joint training
30.46
31.48
27.55
Table 1: The BLEU scores of fine-tuning and joint training method. Fine-tuning method suffers from CF.
translation. Generally, two basic methods can be adopted: i) Fine-tuning. We can fine-tune the old system with the new translation data.This method suffers from severe degradation on previous translation tasks, this phenomenon is known as Catastrophic Forgetting (CF) (McCloskey and Cohen, 1989). Table 1 shows the results, where after fine-tuning with EN⇒RO task, BLEU scores of the previous tasks sharply drop from 28.51 to 1.08 (EN⇒IT) and from 29.79 to 0.99 (EN⇒NL), respectively. ii) Joint training. We can train a new task jointly with the previous and new training data. This method can achieve good performance. While as the number of translation tasks grows, storing and retraining on all training data becomes infeasible and cumbersome (Li and Hoiem, 2017). More seriously, in many cases, the training data for previously learned tasks is unavailable due to the data privacy and protection (Shokri and Shmatikov, 2015; Yang et al., 2019), and it is impossible to jointly train an MNMT model under this situation. Life-long learning aims at adapting a learned model to a new task while retaining the previous knowledge without accessing the previous training data. Among them, knowledge distillation based methods (Li and Hoiem, 2017; Hou et al., 2018; Belouadah and Popescu, 2019) are very common ways. In these methods when a new task arrives, the new model (student) is jointly learned by the old model’s output (teacher) and new task. However, these methods are specifically designed for image classification (Li and Hoiem, 2017; Hou
et al., 2018; Aharoni et al., 2019), or object detection (Shmelkov et al., 2017), the incremental MNMT scenarios are not studied. Therefore, in this paper, we focus on the lifelong learning for incremental MNMT, and the scenarios we study here are setting as follows: i) Each task arrives in a sequential manner. ii) Training data of previous tasks is unavailable. We can only access the training data of new translation task and a learned old MNMT model. iii) A single MNMT model needs to perform well on all tasks after learning a new one. Specifically, two common incremental MNMT scenarios are considered: Incremental one-to-many scenario: An MNMT model incrementally learns to translate one same source language into different target languages. Incremental many-to-one scenario: An MNMT model incrementally learns to translate different source languages into one same target language1. In incremental one-to-many scenario, we propose a multilingual distillation method, in which the old model is treated as a teacher and the new model is treated as a student. To distillate the multilingual knowledge in the teacher, we first add the corresponding indicator of learned languages in the beginning of source sentence, which then be fed into the teacher model to get the multilingual outputs. Finally, the student model is jointly learned by the new task and multilingual distillation results. In incremental many-to-one scenario, we find that direct distillation faces the Extreme Partial Distillation problem: Extreme Partial Distillation: Given an old many-to-one model (such as IT, NL⇒EN), and a new task (such as RO⇒EN), if we treat the old model as a teacher and directly input the new source sentences (RO) into it, the teacher model actually is fed by a sentence filled with UNKs due to the UNK replacing strategy2. Ideally, we hope that the student model could learn the knowledge from teacher on various tokens (whole knowledge) of previous source languages (IT,NL). While in this situation, the student model can only learn from teacher how to handle UNKs (partial knowledge).
Therefore, we define this problem as extreme partial distillation. To address this problem in many-to-one scenario, we propose two methods: 1) pseudo input distillation, and 2) reverse teacher distillation. In the former one, we still utilize the old many-to-one MNMT model as a teacher. While instead of directly inputting the new source sentences (RO) into it, we first construct pseudo inputs by replacing the new tokens (RO) with learned tokens (IT,NL) via a frequency mapping. Then pseudo inputs are utilized to distillate the knowledge from teacher. In the later one, we utilize the reversed one-tomany model (EN ⇒IT,NL) as a teacher. When a new task arrives (RO⇒EN), we input the target language (EN) into the teacher and get the multilingual source outputs (IT,NL). We test the proposed methods on twelve different translation tasks. The experimental results show that the proposed methods can sharply alleviate the CF. The contributions of this paper are listed as follows: i) We focus on the incremental MNMT scenario and investigate the knowledge distillation based life-long learning method. ii) In one-to-many scenario, we propose a multilingual distillation method to make the new model jointly learn multilingual output from old model and new task. iii) In many-to-one scenario, we find that direct distillation faces the extreme partial distillation problem, and propose two different methods (pseudo input distillation and reverse teacher distillation) to address it.
# 2 Multilingual NMT
To make full use of multilingual data within a single system, various MNMT methods are proposed (Firat et al., 2016; Johnson et al., 2017; Gu et al., 2018), where (Johnson et al., 2017) propose a simple while effective MNMT method. In this method, it is no need to change the network architecture. The only modification is that they introduce a special indicator at the beginning of the source sentence to indicate source and target language. For example, consider the following English-toItaly sentence pair: you probably saw it on the news . →forse lo avete visto sui notiziari . It will be modified to: <en2it> you probably saw it on the news . →
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4b9d/4b9d33c8-12ab-4a17-8065-5234752d3a8c.png" style="width: 50%;"></div>
<div style="text-align: center;">(c)  Reverse Teacher Distillation</div>
Figure 1: The framework of proposed methods, where multilingual distillation (a) is proposed for incremental one to-many scenario. Pseudo input distillation (b) and reverse teacher distillation (c) are proposed for incremental many-to-one scenario.
forse lo avete visto sui notiziari . where <en2it> is an indicator to show that the source is English and the target is Italy. Notation: We denote a one-to-many MNMT model by θ(X⇒Y1,...,Yn), where X is a source language, and Y1, ..., Yn denotes n different target languages. Similarly, a many-to-one MNMT model is denoted by θ(X1,...,Xn⇒Y). We denote a translation task from X to Y by X ⇒Y, whose training sentence pairs are denoted by DX⇒Y = {(X, Y )}, where X is the source sentence and Y is the target sentence. When we adding a indicator <X2Yj> for target language Yj into a source sentence X, we denote the source sentence by X+<X2Yj>.
# 3 Method Description
In incremental one-to-many scenario, given an old one-to-many model θ(X⇒Y1,...,Yn) and the training data DX⇒Yn+1 = {(X, Yn+1)} of a new task X ⇒Yn+1, our goal is to get a new model θ(X⇒Y1,...,Yn+1). To achieve this, we propose a multilingual distillation method to let the new model (student) jointly learn the new task and multilingual knowledge in the old model (teacher). Fig. 1 (a) illustrates the framework, which contains three
steps: Step 1: For each learned target language Yi ∈ (Y1, ..., Yn), we first add the indicator <X2Yi> into source sentence X. We denote the source sentence with indicator by X+<X2Yi>. Step 2: For each X+<X2Yi>, i ∈[1, n], we input it into the old model θ(X⇒Y1,...,Yn) (teacher) and get the corresponding result Yi with beam search:
Step 3: Train a new model θ(X⇒Y1,...,Yn+1 (student) by maximizing the following objective function:
� �� � � �� � where the first n items denote the loss of previous tasks produced by the teacher. The last one denotes the loss of new task. Discussion. We call this method as multilingual distillation, since the new model could learn multilingual knowledge in the old model. During distillation, we can also utilize the k-best sentence distillation (k > 1) (Kim and Rush, 2016; Tan et al., 2019) or greedy search distillation. The experimental results can be found in Sec. 5.1.
# 3.2 Incremental Many-to-one Scenario
In this scenario, our goal is to get a new model θ(X1,...,Xn+1⇒Y) with an old many-to-one model θ(X1,...,Xn⇒Y) and training sentence pairs DXn+1⇒Y = {(Xn+1, Y )} of a new task Xn+1 ⇒ Y. As we mentioned before, direct distillation faces the extreme partial distillation problem. To address this, we propose two different methods: 1) pseudo input distillation (Fig. 1 (b)), and 2) reverse teacher distillation (Fig. 1 (c)).
# 3.2.1 Pseudo Input Distillation
In this method, instead of inputting Xn+1 into the teacher, we first construct a pseudo input by transforming the new token of Xn+1 into the previous tokens. Then the pseudo input is utilized to distillate the knowledge. Specifically, this method contains four steps: Step 1: For each source language Xi ∈ (X1, ..., Xn+1), we sort its vocabulary VXi in descending order by the frequency as follows:
(3)
N(sj Xi) represents the frequency of token sj Xi in the corresponding training data, and j is the frequency ranking of this token in all vocabulary. Then we can construct a mapping between a new token sj Xn+1 and a learned token sj Xi, if these two tokens have the same ranking j. Formally,
(4)
where sj Xn+1 is a new token in language Xn+1 which ranked jth in VXn+1 and sj Xi is a learned token in language Xi which also ranked jth in VXi Step 2: Given a new source sentence Xn+1, we replace the tokens in Xn+1 with learned tokens in languages Xi with the mapping M(Xn+1→Xi) (Eq. (4)) by
(5)
where Xp i is the pseudo input, which contains the tokens of language Xi. Step 3: For each pseudo input Xp i ∈ (Xp 1, ..., Xp n), we input it into the old model θ(X1,...,Xn⇒Y) (teacher) and get the distillation result Yi with beam search:
(6)
Step 4: Train a new model θ(X1,...,Xn+1⇒Y) (student) by maximizing the following objective function:
(7)
� �� � � �� � where the first n items denote the loss of previous tasks produced by the teacher. The last one denotes the loss of new task. Discussion. This pseudo input distillation could alleviate extreme partial distillation problem, since the pseudo inputs Xp i contains the various tokens of previous learned language. Thus, when we utilize Xp i as the distillation input, the student model could learn from teacher that how to translate these tokens while not just unks. Meanwhile, the additional cost of this method is small, we only need to maintain a sorted vocabulary in descending order for each learned source languages.
# 3.2.2 Reverse Teacher Distillation
In reverse teacher distillation, beside the old manyto-one model θ(X1,...,Xn⇒Y), we also need a reverse one-to-many model θ(Y⇒X1,...,Xn) at the same time. To alleviate the extreme partial distillation problem, when the training data of a new task DXn+1⇒Y = {(Xn+1, Y )} arrives, we treat the one-to-many model θ(Y⇒X1,...,Xn) as a teacher, and input the target sentence Y in it. Specifically, this method contains four steps: Step 1: For each learned source languages Xi ∈ (X1, ..., Xn), we first add the indicator <Y2Xi> into target sentence Y , and denote the target sentence with indicator by Y +<Y2Xi>. Step 2: For target sentence with indicator Y +<Y2Xi> ∈(Y +<Y2X1>, ..., Y +<Y2Xn>), we input it into the reverse one-to-many model θ(Y⇒X1,...,Xn) (reverse teacher) and get the distillation result Xi with beam search:
Step 3: Train a new many-to-one model (student) θ(X1,...,Xn+1⇒Y) by maximizing the following objective function:
(9)
where the first n items denote the loss of previous tasks produced by the reverse teacher. The last one denotes the loss of new task. Step 4: Train a new reverse one-to-many model (reverse student) θ(Y⇒X1,...,Xn+1) by maximizing the following objective function:
(10)
� �� � � �� � We also need to update the reverse student, since it will be utilized as a reverse teacher when the next task Xn+2 ⇒Y arrives. Discussion. This reverse teacher distillation could alleviate extreme partial distillation problem, since we utilize the reverse one-to-many model as a teacher and input the target into it. This idea is partially inspired by the back-translation (Sennrich et al., 2016a) and generative replay methods in lifelong learning (Shin et al., 2017; Zhai et al., 2019). Meanwhile, the additional cost of this method is acceptable. We need to maintain a revered one-tomany model when a new task arrives.
# 4 Experimental Setting
Dataset. We test the proposed methods on 12 tasks (Table 2), where English-Italian (EN⇔IT), English-Dutch (EN⇔NL) and English-Romanian (EN⇔RO) come from TED dataset3. UygurChinese (UY⇔CH), Tibetan-Chinese (TI⇔CH) and Mongolian-Chinese (MO⇔CH) come from CCMT-19 dataset. Chinese-English (CH⇔EN) is LDC dataset. Japanese-English (JA⇔EN) is KFTT dataset4. German-English (DE⇔EN), FinnishEnglish (FI⇔EN), Latvian-English (LV⇔EN) and Turkish-English (TR⇔EN) come from WMT-17 dataset5. Training and Evaluation Details. We implement our approach based on the THUMT toolkit (Zhang et al., 2017)6. We use the “base” parameters in Transformer (Vaswani et al., 2017). We use the BPE (Sennrich et al., 2016b) method to merge 30K steps. For evaluation, we use beam search with a beam size of k = 4 and length penalty. We evaluate the translation quality with BLEU (Papineni et al., 2002) for all tasks. For 3https://wit3.fbk.eu/ 4http://www.phontron.com/kftt/ 5http://data.statmt.org/wmt17/ translation-task/preprocessed/ 6https://github.com/THUNLP-MT/THUMT
Dataset
Task
Train
Dev
Test
TED
EN⇔IT
232k
929
1566
EN⇔NL
237k
1003
1777
EN⇔RO
221k
914
1678
CCMT-19
MO⇔CH
254k
2000
1000
TI⇔CH
155k
2000
1000
UY⇔CH
168k
2000
1000
LDC
CH⇔EN
2.1M
919
6146
KFTT
JA⇔EN
440k
1166
1160
WMT-17
DE⇔EN
5.9M
3003
5168
FI⇔EN
2.6M
2870
3000
LV⇔EN
4.5M
1000
1003
TR⇔EN
207K
2870
3000
each task, we set both source and target vocabularies by 30K. When learning a new task, the new vocabularies are the union of previous vocabularies and vocabularies of new arrival task, i.e., V(X1,...,Xn⇒Y) = V(X1,...,Xn−1⇒Y) ∪V(Xn⇒Y). Comparing methods: We compare the proposed models against the following systems: i) Single: We train each translation task with each single model by using transformer. ii) Joint Training: We implement the joint training method (Johnson et al., 2017) as an upper bound of proposed life-long learning methods. iii) Fine-tuning: We fine-tune the old system with the new training data of a new task. iv) EWC: This is the Elastic Weight Consolidation (EWC) model (Kirkpatrick et al., 2017). The approach injects a penalty on the difference between the parameters for the old and the new tasks into the loss function to alleviate the CF. v) Multi-Distill: This is the proposed Multilingual Distillation in one-to-many scenario. MultiDistill (greedy) and Multi-Distill (beam) denote that during distillation, we utilize the greedy search and beam search, respectively. vi) Direct-Distill: In this method, we directly utilize the new source as distillation input. vii) PseudoInput-Distill: This is the proposed Pseudo Input Distillation in many-to-one scenario. viii) ReverseTeacher-Distill: This is the proposed Reverse Teacher Distillation in many-to-one scenario.
# 5 Experimental Results
# 5.1 Incremental One-to-many Scenario
Main results. Table 3 shows the translation results in incremental one-to-many scenario, where line 18 report the results that the first task is EN⇒IT, the second one is EN⇒NL, and the last one is EN⇒RO.
<div style="text-align: center;">BLEU-task1 BLEU-task2 BLEU-task3 BLEU-avg △</div>
#
Model
BLEU-task1
BLEU-task2
BLEU-task3
BLEU-avg
△
EN⇒IT →EN⇒NL →EN⇒RO
1
Single
27.76
28.64
25.27
27.22
∼
2
Joint Training
30.46
31.48
27.55
29.83
+2.61
3
Fine-tuning
1.05
0.82
25.82
9.23
−17.99
4
EWC
10.32
11.17
22.31
14.60
−12.62
5
Multi-Distill (greedy)
29.00∗
28.97
26.49∗
28.15
+0.93
6
Multi-Distill (beam)
30.31∗
30.11†
26.86∗
29.09
+1.87
7
Multi-Distill (2-best)
30.10∗
30.37†
27.18∗
29.22
+2.00
8
Multi-Distill (4-best)
30.51∗
30.52∗
26.81∗
29.28
+2.06
CH⇒MO →CH⇒TI →CH⇒UY
9
Single
29.19
29.67
16.20
25.02
∼
10
Joint Training
28.97
29.48
17.51
25.32
+0.30
11
Fine-tuning
1.04
0.93
16.67
6.21
−18.81
12
EWC
11.32
12.43
15.53
13.09
−11.93
13
Multi-Distill (greedy)
28.88
29.30
16.52
24.90
−0.12
14
Multi-Distill (beam)
29.11
29.44
16.79†
25.11
+0.09
15
Multi-Distill (2-best)
29.34
29.39
16.90∗
25.21
+0.19
16
Multi-Distill (4-best)
29.50†
29.51
16.98∗
25.33
+0.31
Table 3: The translation results in incremental one-to-many scenario. BLEU-task1, BLEU-task2 and BLEUtask3 show the BLEU scores of the corresponding three tasks. BLEU-avg is the average BLEU scores. △is the improvement comparing with single method. “†” indicates that the proposed system is statistically significant better (p < 0.05) than the single system and “*” indicates p < 0.01.
Line 9-16 report the results that the first task is CH⇒MO, the second one is CH⇒TI, and the last one is CH⇒UY. From the results, we can reach the following conclusions: i) Fine-tuning method suffers heavily from CF problem. Compared with the single method, the average BLEU scores sharply dropped to 9.23 (line 3) and 6.21 (line 11), respectively. EWC method can partially alleviate CF, while the average BLEU scores remain below the single method by 12.62 (line 4) and 11.93 (line 12) , respectively. ii) The proposed multilingual distillation method can sharply alleviate the CF. After learning three continuous tasks, the average BLEU points are 29.09 (line 6) and 25.11 (line 14), respectively. Compared with the single method, its improvement can reach to 1.87 and 0.09 BLEU points. iii) We also investigate the k-best distillation in our method. We can find that k-best distillation (k = 2 and 4) can only slightly improve the current 1-best distillation. Meanwhile, we can also find that during distillation, the beam search distillation (line 6 and 14) can achieve better results than greedy search distillation (line 5 and 13). Results on an already existing MNMT model. We also conduct an experiment on the basis of an existing MNMT model. The existing model is trained by EN⇒IT, NL, RO, JA, DE, FI, LV, TR tasks, and the new task is EN⇒CH. Table 4 lists the results, where the initial average BLEU score of learned tasks is 23.68. After learn-
Model
BLEU-prev
BLEU-new
Existing
23.68
∼
Single
23.49
21.14
Joint Training
23.94
20.88
Fine-tuning
1.07
21.21
Multi-Distill
23.80
21.19
Table 4: The translation results of an already existing on-to-many model. The existing model is trained by EN⇒IT, NL, RO, JA, DE, FI, LV, TR tasks, and the new task is EN⇒CH.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d7de/d7def9b4-e8f0-456d-a177-5987b121ec07.png" style="width: 50%;"></div>
<div style="text-align: center;">0 2 4 6 8 16 12 14 2 4 6 8 10 14 Learning with task 1  12 10 3 10  Learning with task 2 (EN  IT) (EN  NL) 16</div>
Figure 2: The BLEU scores during training in one-tomany scenario. x axis denotes the training epoch and y axis denotes the BLEU scores of development set.
ing EN⇒CH, the multilingual distillation improves the BLEU score of learned tasks to 23.80 and new task to 21.19. Compared with the joint training, it achieves comparable results (23.94 vs. 23.80 and 20.88 vs. 21.19). The results show that the proposed method can consolidate the previous knowledge when learning a new task. Results during training. We are also curious
#
Model
BLEU-task1
BLEU-task2
BLEU-task3
BLEU-avg
△
IT⇒EN →NL⇒EN →RO⇒EN
1
Single
30.15
32.50
30.86
31.17
∼
2
Joint Training
32.04
34.62
33.49
33.38
+2.21
3
Fine-tuning
9.77
14.30
29.51
17.86
−13.31
4
EWC
21.29
24.32
28.07
24.56
−6.61
5
Direct-Distill
1.12
1.43
29.43
10.66
−20.51
6
PseudoInput-Distill (greedy)
27.01
29.33
31.86∗
29.40
−1.77
7
ReverseTeacher-Distill (greedy)
31.04∗
33.87∗
32.75∗
32.55
+1.38
8
PseudoInput-Distill (beam)
27.93
30.01
32.27∗
30.07
−1.10
9
ReverseTeacher-Distill (beam)
31.80∗
34.32∗
32.95∗
33.02
+1.85
MO⇒CH →TI⇒CH →UY⇒CH
10
Single
40.60
32.79
22.62
32.00
∼
11
Joint Training
42.30
31.89
24.10
32.76
+0.76
12
Fine-tuning
10.59
11.05
22.49
14.71
−17.29
13
EWC
24.98
21.38
21.33
22.56
−9.44
14
Direct-Distill
1.09
2.11
20.98
8.06
−23.94
15
PseudoInput-Distill (greedy)
36.42
28.77
23.17
29.45
−2.55
16
ReverseTeacher-Distill (greedy)
39.52
31.89
23.29†
31.57
−0.43
17
PseudoInput-Distill (beam)
38.01
30.19
23.30†
30.50
−1.50
18
ReverseTeacher-Distill (beam)
42.04∗
32.14
24.01∗
32.73
+0.73
Table 5: The translation results in incremental many-to-one scenario. “†” indicates that the proposed system i tatistically significant better (p < 0.05) than the single system and “*” indicates p < 0.01.
about the results at each training epoch. Thus we record the BLEU scores of development set during training, where the first task is EN⇒IT and the second one is EN⇒NL. Fig. 2 reports the results, where x axis denotes the training epoch and y axis denotes the BLEU scores of development set. The results show that when we fine-tune the learned model (θEN⇒IT) with EN⇒NL task, the BLEU score of the first one sharply drops to 0.99. The proposed multilingual distillation can alleviate this CF problem.
# 5.2 Incremental Many-to-one Scenario
Main results. Table 5 shows the translation results in incremental many-to-one scenario, where line 19 report the results that the first task is IT⇒EN, the second one is NL⇒EN, and the last one is RO⇒EN. Line 10-18 report the results that the first task is MO⇒CH, the second one is TI⇒CH, and the last one is UY⇒CH. From the results, we can reach the following conclusions: i) Fine-tuning method also suffers heavily from CF problem, whose average-BLEU scores seriously drop from 31.17 (line 1) to 17.86 (line 3) and from 32.00 (line 10) to 14.71 (line 12), respectively. We can also see that CF here is not serious as that in incremental one-to-many scenario (see Table 3). EWC method can partially alleviate CF, while it is still lower than the single model by 6.61 (line 4) and 9.44 (line 13) BLEU points, respectively. ii) Surprisingly, direct distillation further wors-
Model
BLEU-prev
BLEU-new
Initial
27.34
∼
Single
27.01
44.40
Joint Training
27.41
44.28
Fine-tuning
1.00
44.53
PseudoInput-Distill
23.04
43.87
ReverseTeacher-Distill
27.27
44.25
Table 6: The translation results of an already existing many-to-one model. The existing model is trained by IT, NL, RO, JA, DE, FI, LV, TR⇒EN tasks, and the new task is CH⇒EN.
ens the CF due to the extreme partial distillation, whose average-BLEU scores reduce to 10.66 (line 5) and 8.06 (line 14), respectively. iii) The pseudo input distillation method can sharply alleviate the CF. The average BLEU scores can reach to 30.07 (line 8) and 30.50 (line 17), respectively. The reverse teacher distillation can exceed the single model by 1.85 (line 9) and 0.73 (line 18), respectively. Compared with the joint training, this method can achieve comparable results (33.02 vs. 33.38 and 32.73 vs. 32.76). iv) In both pseudo input distillation and reverse teacher distillation, we find that during distillation, the beam search distillation can also achieve better results than the greedy search distillation. Results on an already existing MNMT model. We also conduct an experiment on the basis of an existing many-to-one model. Here, the existing model is trained by IT, NL, RO, JA, DE, FI, LV, TR⇒EN tasks, and the new task
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b451/b45113c7-85e4-42a1-9a84-6a29149d4da2.png" style="width: 50%;"></div>
<div style="text-align: center;">Learning with task 1 Learning with task 2 (IT  EN) (NL  EN) 0 2 4 6 8 12 14 2 4 6 8 10 14 12 10 16 16 3 10 </div>
Figure 3: The BLEU scores during training in many-toone scenario. x axis denotes the training epoch and y axis denotes the BLEU scores of development set.
is CH⇒EN. Table 6 lists the results, where the initial BLEU score of learned tasks is 27.34. After learning CH⇒EN, the reverse teacher distillation can retain the BLEU score of learned tasks to 27.27. Meanwhile its BLEU score of new task is 44.25. These results are comparable with that of joint training (27.27 vs. 27.41 and 44.25 vs. 44.28). The results show that the proposed methods could also alleviate the CF in many-to-one scenario. Results during training. Fig. 3 reports the BLEU scores of development set at each training epoch, where the first task is IT⇒EN and the second one is NL⇒EN. From the results, we can see that fine-tuning also faces the CF when learning the second task. Both pseudo input distillation and reverse teacher distillation could alleviate this problem. In particular, reverse teacher distillation could further improve performance of the first task when learning the second one. Meanwhile, compared with fine-tuning method, these two methods can also improve the BLEU scores of the second task.
# 6 Related Work
Multilingual Neural Machine Translation. To facilitate the deployment and improve the performance, various MNMT models are proposed (Dong et al., 2015; Johnson et al., 2017; Gu et al., 2018; Wang et al., 2018; Tan et al., 2019; Aharoni et al., 2019; Wang et al., 2019; Kudugunta et al., 2019; Bapna and Firat, 2019), where Tan et al. (2019) propose a knowledge distillation method for MNMT. However, these studies are conducted in the conventional setting that bilingual pairs for all the translation tasks are available at training time. Different from these studies, we focus on incremental scenario that the training data of previous tasks is
unavailable. Life-long Learning and Its Application in NLP. Lifelong learning aims at adapting a learned model to new tasks while retaining the previous knowledge. De Lange et al. (2019) classify these methods into three categories: i) replay-based methods (Lopez-Paz and Ranzato, 2017; Wu et al., 2018), ii) regularization-based methods (Li and Hoiem, 2017), and iii) parameter isolation-based methods (Rusu et al., 2016). Meanwhile, several studies apply these methods into NLP tasks, such as sentiment analysis (Chen et al., 2018; Xia et al., 2017), word and sentence representation learning (Xu et al., 2018; Liu et al., 2019), language modeling (Sun et al., 2019; d’Autume et al., 2019), domain adaptation for NMT (Barone et al., 2017; Thompson et al., 2019) and post-editors for NMT (Turchi et al., 2017; Thompson et al., 2019). Different from these studies, we focus on the incremental MNMT scenario that each tasks arrive in a sequential manner and training data of previous tasks is unavailable. Recently, Escolano et al. (2019) propose an incremental training method for MNMT, in which they train the independent encoders and decoders for each languages. While with the increasing of learning language pair, the model parameters become larger. Different from this study, we apply the life-long learning method on a more challenging MNMT framework with one universal encoder and decoder (Johnson et al., 2017).
# 7 Conclusion and Future Work
In this paper, we aim at enabling the MNMT to learn incremental translation tasks over a lifetime. To achieve this, we investigate knowledge distillation based life-long learning for MNMT. In one-tomany scenario, we propose a multilingual distillation method. In incremental many-to-one scenario, we find that direct distillation faces the extreme partial distillation problem, and propose pseudo input distillation and reverse teacher distillation to address this problem. The extensive experiments demonstrate that our method can retain the previous knowledge when learning a new task. As a novel attempt of life-long learning for MNMT, the proposed methods still have a drawback that they cost more computational overhead due to the knowledge distillation. Therefore, in the future we will study how to reduce the computational overhead of our methods and extend them into the incremental many-to-many scenario.
