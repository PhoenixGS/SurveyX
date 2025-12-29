# Yao Qiang∗and Xiangyu Zhou∗and Saleh Zare Zade Prashant Khanduri and Dongxiao Zhu
Department of Computer Science, Wayne State University {yao, xiangyu, salehz, khanduri.prashant, dzhu}@wayne.edu
# Abstract
In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific downstream tasks by utilizing labeled examples as demonstrations (demos) in the precondition prompts. Despite its promising performance, ICL suffers from instability with the choice and arrangement of examples. Additionally, crafted adversarial attacks pose a notable threat to the robustness of ICL. However, existing attacks are either easy to detect, rely on external models, or lack specificity towards ICL. This work introduces a novel transferable attack against ICL to address these issues, aiming to hijack LLMs to generate the target response or jailbreak. Our hijacking attack leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demos without directly contaminating the user queries. Comprehensive experimental results across different generation and jailbreaking tasks highlight the effectiveness of our hijacking attack, resulting in distracted attention towards adversarial tokens and consequently leading to unwanted target outputs. We also propose a defense strategy against hijacking attacks through the use of extra clean demos, which enhances the robustness of LLMs during ICL. Broadly, this work reveals the significant security vulnerabilities of LLMs and emphasizes the necessity for indepth studies on their robustness.
# Introduction
In-context learning (ICL) is an emerging technique for rapidly adapting large language models (LLMs), i.e., GPT-4 (Achiam et al., 2023) and LLaMA2 (Touvron et al., 2023), to new tasks without finetuning the pre-trained models (Brown et al., 2020). The key idea behind ICL is to provide LLMs with labeled examples as in-context demonstrations (demos) within the prompt context before a test query. LLMs are able to generate responses to queries via learning from the in-context demos (Dong et al., 2022; Min et al., 2022).
Several existing works, however, have demonstrated the highly unstable nature of ICL (Zhao et al., 2021; Chen et al., 2022). Specifically, performance on target tasks using ICL can vary wildly based on the selection and order of demos, giving rise to highly volatile outcomes ranging from random to near state-of-the-art (Qiang et al., 2020; Lu et al., 2021; Min et al., 2022; Pezeshkpour and Hruschka, 2023; Qiang et al., 2024). Correspondingly, several approaches (Liu et al., 2021; Wu et al., 2022; Nguyen and Wong, 2023) have been proposed to address the unstable issue of ICL. Further research has examined how adversarial examples can undermine the performance of ICL (Zhu et al., 2023a; Wang et al., 2023c,b; Shayegani et al., 2023). These studies show that maliciously designed examples injected into the prompt instructions (Zhu et al., 2023a; Zou et al., 2023; Xu et al., 2023), demos (Wang et al., 2023c; Mo et al., 2023a), or queries (Wang et al., 2023b; Kandpal et al., 2023) can successfully attack LLMs to degrade their performance, revealing the significant vulnerabilities of ICL against adversarial inputs. While existing adversarial attacks have been applied to evaluate LLM robustness, they have some limitations in practice. Most character-level attacks, e.g., TextAttack (Morris et al., 2020) and TextBugger (Li et al., 2018), can be easily detected and evaded through grammar checks, limiting realworld effectiveness (Qiang et al., 2022; Jain et al., 2023). Some other attacks like BERTAttack (Li et al., 2020) even require an extra model to generate adversarial examples. Crucially, existing attacks are not specifically crafted for ICL. As such, the inherent security risks of ICL remain largely unexplored. There is an urgent need for red teaming specifically designed for ICL to expose significant risks and improve the robustness of LLMs against potential real-world threats. This work proposes a novel adversarial attack specifically targeting ICL. We develop a gradient-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a72c/a72c4c65-0d0b-48ed-b733-23b52f703f04.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Illustrations of hijacking attack during ICL. First, our proposed GGI algorithm learns and appends adversarial suffixes like ‘For’ and ‘Location’ to the system or the user-provided in-context demos for hijacking LLMs to generate the target response, e.g., the ‘negative’ sentiment, regardless of the user queries. Second, GGI can accomplish jailbreaking by adding adversarial suffixes to in-context demos, eliciting harmful responses while bypassing the safeguards in LLMs. More detailed examples are provided in the Appendix.</div>
based prompt search algorithm to learn adversarial suffixes in order to efficiently and effectively hijack LLMs via adversarial ICL, as illustrated in Figure 1. (Wang et al., 2023b) is the closest work to ours where they ‘search’ adversarial examples to simply manipulate model outputs. Yet, our attack method ‘learns’ adversarial tokens that directly hijack LLMs to generate the unwanted target that disrupts alignment with the desired output. This enables our attack to be used in more complex generation tasks, such as jailbreaking, as illustrated in Figure 1. Furthermore, instead of manipulating the prompt instructions (Zhu et al., 2023a), demos (Wang et al., 2023c), or queries (Wang et al., 2023b) leveraging standard adversarial examples, e.g., character-level attacks (Morris et al., 2020; Li et al., 2018), which are detectable easily, our hijacking attack is imperceptible in that it adds only 1-2 suffixes to the demos. Specifically, these suffixes are semantically incongruous but not easily identified as typos or gibberish compared to the existing ICL attack (Wang et al., 2023c). Finally, direct attacks on user queries, such as backdoors (Kandpal et al., 2023), which require a trigger, are easily detectable and may not be practical for realworld applications. In contrast, our attack hijacks the LLM to generate the unwanted target without triggering or compromising the user’s queries directly. Our adversary attacker only needs to append the adversarial tokens to system-provided demos. Our extensive experiments validate the efficacy and scalability of the proposed hijacking attacks. First, the attacks reliably induce LLMs to generate the targeted and misaligned output from the de-
sired ones. Second, the learned adversarial tokens are transferable, remaining effective on different demo sets. Third, the adversarial transferability holds even across different datasets for the same task. Finally, our analysis shows that the adversarial suffixes distract LLMs’ attention away from the task-relevant concepts. Our hijacking attacks pose a considerable threat to practical LLM applications during ICL due to their robust transferability, imperceptibility, and scalability. As this work represents one of the first efficient adversarial demo attacks during ICL, strategies for defending against such attacks have yet to be thoroughly investigated. Recently, (Mo et al., 2023b) introduced a method for defending against backdoor attacks at test time, leveraging few-shot demos to correct the inference behavior of poisoned LLMs. Similarly, (Wei et al., 2023b) explored the power of in-context demos in manipulating the alignment ability of LLMs and proposed in-context attack and in-context defense methods for jailbreaking and guarding the aligned LLMs. Consequently, we explore the potential of using in-context demos exclusively to rectify the behavior of LLMs subjected to our hijacking attacks. Our defense strategy employs additional clean in-context demos at test time to safeguard LLMs from being hijacked by adversarial in-context demos. The experimental results demonstrate the efficacy of our proposed defense method against adversarial demo attacks. This work makes the following contributions: (1) We propose a novel stealthy adversarial attack targeting in-context demos to hijack LLMs to generate unwanted target output during ICL. (2) We
design a novel and efficient gradient-based prompt search algorithm to learn adversarial suffixes to demos. (3) Comprehensive experimental results across various generation tasks demonstrate the effectiveness of our hijacking attack. (4) Our extensive experiments reveal the transferability of the proposed attack across demo sets and datasets. (5) The proposed defense strategy effectively protects LLMs from being compromised by our attacks.
# 2 Preliminaries 2.1 ICL Formulation
Formally, ICL is characterized as a problem involving the conditional generation of text (Liu et al., 2021), where an LLM M is employed to generate response yQ given an optimal task instruction I, a demo set C, and an input query xQ. I specifies the downstream task that M should perform, e.g., “Choose sentiment from positive or negative” used in the sentiment generation task. C consists of N (e.g., 8) concatenated data-label pairs following a specific template S, formally: C = [S(x1, y1); · · · ; S(xN, yN)], ‘;’ here denotes the concatenation operator. Thus, given the input prompt as p = [I; C; S(xQ, _)], M generates the response as ˆyQ = M(p). S(xQ, _) here means using the same template as the demos but with the label empty.
# 2.2 Adversarial Attack on LLMs
In text-based adversarial attacks, the attackers manipulate the input x with the goal of misleading the model to generate inaccurate or malicious outputs (Zou et al., 2023; Maus et al., 2023). Specifically, given the input-output pair (x, y), the attackers aim to learn the adversarial perturbation δ adding to x by maximizing the model’s objective function but without misleading humans by bounding the perturbation within the “perceptual” region ∆. The objective function of the attacking process thus can be formulated as:
(1)
∈ L here denotes the task-specific loss function, for instance, cross-entropy loss for classification tasks. 3 The Threat Model
L here denotes the task-specific loss function, for instance, cross-entropy loss for classification tasks.
# 3 The Threat Model 3.1 ICL Hijacking Attack
ICL consists of an instruction I, a demo set C, and an input query xQ, providing more potential attack vectors than conventional text-based adversarial attacks. This work focuses on manipulating C without changing I and xQ.
Specifically, our hijacking attack learns the adversarial suffix tokens to the in-context demos to manipulate LLMs’ output via a new greedy gradient-based prompt injection algorithm. Given a clean demo set C = [S(x1, y1); · · · ; S(xN, yN)], our hijacking attack automatically produces an adversarial suffix for each demo in c, formally:
C′ = [S(x1+δ1, y1); · · · ; S(xN+δN, yN)], (2)
(2)
where C′ denotes the perturbed demo set. To make it clear, the adversarial suffixes appended to each demo as perturbations are different. In this case, the attack or perturbation budget refers to the number of tokens in each adversarial suffix. As a result, our hijacking attack induces M to generate an unwanted target output yT via appending adversarial suffix tokens on the in-context demos as yT = M(p′). In other words, M generates the same or different responses for the clean and perturbed prompts depending on the True or False of M(p) = yT :
where p = [I; C; S(xQ, _)] and p′ = [I; C′; S(xQ, _)], respectively.
# 3.2 Hijacking Attack Objective
We express the goal of the hijacking attack as a formal objective function. Let us consider the LLM M as a function that maps a sequence of tokens x1:n, with x ∈{1, · · · , V } where V denote the vocabulary size, namely, the number of tokens, to a probability distribution over the next token xn+1. Specifically, P(xn+1|x1:n) denotes the probability that xn+1 is the next token given the previous tokens x1:n. Using the notations defined earlier, the hijacking attack objective we want to optimize is simply the negative log probability of the target token xn+1. The generated target output yT differs from the ground truth label yQ for the training query (xQ, yQ). Formally:
(3)
where yT neqyQ, demonstrating the attack hijacks mathcalM to generate the target output. For instance, the target output for the sentiment analysis task can be set as ‘positive’ or ‘negative’. For the jailbreaking task, we set the target token as ‘Sure’
aiming to elicit the following harmful responses. In summary, the problem of optimizing the adversarial suffix tokens can be formulated as the following optimization objective:
(4)
where i and N denote the indices and the number of the demos, respectively.
# 3.3 Greedy Gradient-guided Injection
A primary challenge in optimizing Eq. 4 is optimizing over a discrete set of possible token values. Motivated by prior works (Shin et al., 2020; Zou et al., 2023; Wen et al., 2024), we propose a simple yet effective algorithm for LLMs hijacking attacks, called greedy gradient-guided injection (GGI) algorithm (Algorithm 1 in the Appendix). The key idea comes from greedy coordinate descent: if we could evaluate all possible suffix token injections, we could substitute the tokens that maximize the adversarial loss reduction. Since exhaustively evaluating all tokens is infeasible due to the large candidate vocabulary size, we instead leverage gradients with respect to the suffix indicators to find promising candidate tokens for each position. We then evaluate all of these candidate injections with explicit forward passes to find the one that decreases the loss the most. This allows an efficient approximation of the true greedy selection. We can optimize the discrete adversarial suffixes by iteratively injecting the best tokens. We compute the linearized approximation of replacing the demo xi in C by evaluating the gradient ∇exj i L(xQ) ∈R|V |, where exj i denotes the vector representing the current value of the j-th adversarial suffix token. Note that because LLMs typically form embeddings for each token, they can be written as functions of exj i , and thus we can immediately take the gradient with respect to this quantity (Ebrahimi et al., 2017; Shin et al., 2020). The key aspects of our GGI algorithm are: firstly, it uses gradients of the selected token candidates to calculate the top candidates; secondly, it evaluates the top candidates explicitly to identify the most suitable one; and lastly, it iteratively injects the best token at each position to optimize the suffixes. This approximates an extensive greedy search in a computationally efficient manner.
# 4 The Defense Method
Having developed the hijacking attack by incorporating adversarial tokens into the in-context demos,
we now present a straightforward yet potent defense strategy to counter this attack. Initially, we assume that defenders treat LLMs as black-box, lacking any insight into their training processes or underlying parameters. The defenders apply defense on the input prompt p directly during testtime evaluation. Their goal is to rectify the behavior of LLMs and induce LLMs to generate desired responses to user queries. Given an input prompt p′ that includes adversarial tokens within the demos C′, we assume that LLMs, when presented with demos containing clean data for the same tasks, can understand the genuine intent of the user’s query through ICL, rather than being misled by the adversarial demos. In this context, ‘clean data’ refers to data without any adversarial tokens and is randomly selected from the training set. More precisely, the defenders modify the input prompt p′ into ˜p by appending or inserting more clean demos into the demo set C′, as follows: ˜p = [I; C′; ˜C; S(xQ, _)]. ˜C = [S(˜x1, ˜y1); · · · ; S(˜xN, ˜yN)] here denotes the clean demos. Through this approach, the defender guarantees that the in-context demos align with the user’s query and possess resilience against adversarial attacks. In our experiments, we maintained an equal number of demos in C′ and ˜C and observed that this method resulted in effective defense across various datasets and tasks.
# 5 Experiment Setup
Datasets: We evaluate the performance of our LLM hijacking algorithm and other baseline algorithms on several text generation benchmarks. SST-2 (Socher et al., 2013) and Rotten Tomatoes (RT) (Pang and Lee, 2005) are binary sentiment analysis datasets of movie reviews. AG’s News (Zhang et al., 2015) is a multi-class news topic generation dataset. AdvBench (Zou et al., 2023) is a new adversarial benchmark to evaluate jailbreak attacks for circumventing the specified guardrails of LLMs to generate harmful or objectionable content. These datasets enable us to evaluate the proposed hijacking attacks across a variety of text generation tasks, including both single token and long sequential text generation. More details of the dataset statistics are provided in Table 5 of the Appendix. Large Language Models: The experiments are conducted using various LLMs covering a diverse set of architectures and model sizes, i.e., GPT2-XL (Radford et al., 2019), LLaMA-7b/13b (Touvron et al., 2023), OPT-2.7b/6.7b (Zhang et al., 2022),
Table 1: The performance on sentiment analysis task with and without attacks on ICL. The ‘Clean’ row in gray color represents the accuracy with clean in-context demos. Other rows illustrate the accuracies with adversarial in-context demos. The details of the baselines in green color are present in Section B of the Appendix. Specifically, we employ TextAttack (TA) (Morris et al., 2020) following the attack in (Wang et al., 2023c) as the most closely related baseline for our attack (GGI). The accuracies of positive (P) and negative (N) sentiments are reported separately to highlight the effectiveness of our hijacking attack.
ectiveness of our hijacking attack.
Model
Method
SST-2
RT
2-shots
4-shots
8-shots
2-shots
4-shots
8-shots
P
N
P
N
P
N
P
N
P
N
P
N
GPT2-XL
Clean
94.7
52.2
88.6
49.4
91.6
69.0
93.3
54.7
88.6
76.9
90.2
80.5
Square
99.4
2.0
99.8
4.2
99.4
11.0
99.8
1.5
100
4.1
99.3
7.5
Greedy
100
10.8
100
6.2
100
0.2
100
5.3
100
2.8
100
0.0
TA
95.0
2.2
99.8
17.8
99.6
21.6
95.9
8.1
96.3
41.3
96.4
47.3
GGI
100
1.2
100
0.0
100
0.0
100
2.8
100
0.0
100
0.0
OPT-6.7b
Clean
69.4
87.8
70.2
93.8
77.8
93.0
84.4
91.4
84.4
93.1
88.6
92.8
Square
99.2
31.4
93.8
72.2
99.6
29.0
98.1
42.2
97.0
68.7
99.4
33.2
Greedy
100
25.0
97.8
39.0
100
2.0
99.4
31.7
99.8
4.7
100
0.8
TA
94.8
80.8
54.8
98.6
91.6
89.4
92.5
86.1
77.6
96.4
94.0
86.3
GGI
100
0.0
98.4
2.0
100
0.2
100
2.6
99.8
0.0
100
0.2
Vicuna-7b
Clean
91.4
81.2
88.2
81.4
94.6
82.6
84.8
78.4
85.9
80.5
90.4
85.4
Square
89.2
84.4
86.6
85.8
94.0
83.8
85.9
85.4
84.6
88.6
91.6
88.4
Greedy
93.0
83.4
88.4
87.0
94.6
80.0
91.2
82.8
86.9
88.7
91.9
85.9
TA
87.0
85.2
76.2
88.2
94.2
80.6
83.3
84.2
79.6
88.6
92.1
84.4
GGI
90.6
42.2
96.4
23.2
100
0.8
87.6
36.4
95.1
35.7
100
0.2
LLaMA-7b
Clean
81.4
86.3
74.4
91.9
82.7
92.4
86.0
83.6
81.9
91.6
89.3
97.8
Square
86.8
80.0
96.8
58.6
98.0
56.4
86.9
57.4
97.4
50.1
97.8
57.4
Greedy
95.0
47.6
100
0.0
100
0.0
88.9
2.8
99.8
0.0
100
0.0
TA
87.2
77.8
93.8
69.0
99.8
8.8
83.1
57.4
94.2
68.9
99.6
3.80
GGI
100
0.4
100
0.0
100
0.0
96.8
0.0
100
0.0
100
0.0
LLaMA-13b
Clean
97.8
76.4
95.6
88.0
95.8
90.0
94.2
84.8
92.7
92.1
91.4
91.9
Square
98.4
72.8
98.2
78.4
97.8
85.4
93.6
87.4
94.4
84.1
94.2
87.6
Greedy
98.0
41.4
100
3.0
100
0.0
55.9
11.3
92.9
0.0
100
0.4
TA
98.2
72.2
92.8
92.8
97.5
87.6
94.8
81.8
88.0
94.0
92.5
89.3
GGI
99.2
37.8
100
7.2
100
0.0
99.1
3.8
86.1
3.6
100
0.0
<div style="text-align: center;">Table 2: The performance of AG’s News topic generation task with and without attacks on ICL. The clean and atta accuracies are reported separately for the four topics. These results highlight the effectiveness of our hijacki attacks to induce LLMs to generate the target token, i.e., “tech”, regardless of the query content.</div>
induce LLMs to generate the target token, i.e., “tech”, regardless of the query content.
Model
Method
4-shots
8-shots
word
sports
business
tech
word
sports
business
tech
GPT2-XL
Clean
48.5
87.0
64.9
71.9
48.2
50.6
71.0
83.6
Square
2.0
66.0
26.8
96.0
19.6
65.6
28.0
97.2
Greedy
12.8
60.4
29.2
96.4
8.0
21.2
10.0
98.8
TA
54.8
84.0
73.2
82.4
82.0
82.4
91.2
57.6
GGI
0.0
2.0
0.4
100
0.0
0.0
0.0
100
LLaMA-7b
Clean
68.2
96.8
66.6
49.0
88.6
97.4
78.2
61.0
Square
78.4
98.0
76.0
36.8
94.4
98.0
60.0
57.6
Greedy
69.6
98.8
75.2
51.6
89.6
100
68.4
73.6
TA
42.4
94.8
67.6
32.4
95.2
96.0
39.2
24.8
GGI
0.0
20.0
0.00
98.0
29.6
56.0
0.0
100
and Vicuna-7b (Chiang et al., 2023). This enables us to comprehensively evaluate attack effectiveness on both established and SOTA LLMs.
# 6 Result and Discussion 6.1 ICL Performance
The rows identified as ‘Clean’ in Table 1 and Table 2 show the ICL performance on the respective tasks when using clean in-context demos. In particular, Table 1 presents the accuracies for the generation of positive (P) and negative (N) sentiments in the SST-2 and RT datasets. All the tested LLMs perform well, achieving an average accuracy of 83.6% on SST-2 and 86.7% on RT across various in-context few-shot settings. Table 2 indicates that LLMs with ICL also perform well in the context of multi-class generation on AG’s News dataset. The average accuracies stand at 69.1% for 4-shot settings and 72.3% for 8-shot settings across vari-
ous LLMs. Additionally, LLMs with ICL exhibit improved performance with an increased number of in-context demos, particularly achieving best results with 8-shot settings.
While LLMs utilizing ICL show strong performance with clean in-context demos, Tables 1 and 2 reveal that hijacking attacks significantly undermine their effectiveness. While the baseline methods, i.e., Square, Greedy, and TA, deteriorate model performance on the smaller LLM, e.g., GPT2-XL, they fail to effectively manipulate the larger LLMs, e.g., LLaMA-7/13 b. Additionally, these methods become inefficient as the number of in-context demonstrations increases. Compared to the baselines, our hijacking attacks successfully induce LLMs to generate the targeted positive sentiment through a few shots of adversarially perturbed
Table 3: The performance of the defenses using ASRs across various LLMs and datasets. Adv denotes our hijacking attack using the adversarial demos. Adv+Clean, i.e., Pre and Pro, represents the proposed defense method, leveraging extra clean demos with adversarial demos. Onion (Qi et al., 2020) is the defense method based on outlier word
n and filtering.
Model
SST-2
RT
AG’s News
Adv
Adv+Clean
Onion
Adv
Adv+Clean
Onion
Adv
Adv+Clean
Onion
Pre
Pro
Pre
Pro
Pre
Pro
GPT2-XL
100
100
99.6
100
100
100
97.4
100
99.1
75.5
80.5
83.7
OPT-6.7b
98.2
44.9
52.5
59.3
99.9
50.2
57.8
74.2
65.6
23.5
22.5
14.1
LLaMA-7b
100
49.1
98.3
99.6
100
53.1
99.8
99.9
82.8
42.2
88.2
9.8
<div style="text-align: center;">Table 4: Jailbreaking performance on 200 randomly selected harmful queries from AdvBench.</div>
selected harmful queries from AdvBench.
Model
Method
ASR
2-shots
4-shots
LLaMA2-7b-chat
Clean Query Only
1.5
ICA (Wei et al., 2023b)
3.5
4.0
GGI (ours)
39.5
54.5
Vicuna-7b
Clean Query Only
65.0
ICA (Wei et al., 2023b)
4.0
67.5
GGI (ours)
80.0
91.5
LLaMA3-8b-chat
Clean Query Only
21.0
ICA (Wei et al., 2023b)
20.0
61.0
GGI (ours)
63.5
83.5
demos, resulting in predominantly higher positive accuracies than the negative ones, as shown in Tables 1. The positive test samples achieve almost 100% accuracy. On the contrary, the negative ones get nearly 0% accuracy in most settings. For the more complex multi-class AG’s News topic generation task, the effectiveness of those baseline attacks decreases significantly. Only our GGI attack successfully hijacks the LLMs to generate the target topic ‘tech’, as shown in Table 2.
demos, resulting in predominantly higher positive accuracies than the negative ones, as shown in Tables 1. The positive test samples achieve almost 100% accuracy. On the contrary, the negative ones get nearly 0% accuracy in most settings. For the more complex multi-class AG’s News topic generation task, the effectiveness of those baseline attacks decreases significantly. Only our GGI attack successfully hijacks the LLMs to generate the target topic ‘tech’, as shown in Table 2. 6.3 Jailbreaking Performance We randomly select 200 samples from AdvBench (Zou et al., 2023) as harmful queries to evaluate whether our GGI can learn adversarial tokens that generate harmful or objectionable responses. As long as LLMs generate harmful responses instead of refusal answers, as illustrated in Figure 12 of the Appendix, we consider it as a successful attack. When we input clean queries directly into the tested LLMs, i.e., LLaMA2-7b-chat, Vicuna-7b, and LLaMA3-8b-chat, their safeguards generally prevent the generation of harmful content, resulting in only a few harmful responses, as evidenced by the low ASRs in Table 4. Recently, (Wei et al., 2023b) proposed In-Context Attack (ICA), which employs harmful demos to subvert LLMs for jailbreaking, which achieves slightly higher ASRs as illustrated in Table 4. Furthermore, we utilize GGI to efficiently learn adversarial tokens from harmful demos and then append them to the demos during ICL. Our attack achieves the highest ASRs compared to the baselines, demonstrating the effectiveness of our hijacking attack in inducing harmful responses for jailbreaking, as shown in Figure 12 of the Appendix. The jailbreaking results further illus-
trate the applicability of our GGI method to more complex generative tasks, effectively hijacking the model to generate malicious responses.
# 6.4 Defense Method Performance
Table 3 presents ASRs of our hijacking attack when countered with the proposed defense mechanism that uses additional clean demos and the baseline defense Onion (Qi et al., 2020). Our proposed defense method is tested in two different settings. The preceding (Pre) setting places the clean demos before the adversarial demos in the sequence ˜p = [I; ˜C; C′; S(xQ, _)]. Conversely, the proceeding (Pro) setting adds the clean demos after the adversarial demos as ˜p = [I; C′; ˜C; S(xQ, _)]. The decreases in ASRs of our hijacking attack affirm the effectiveness of these defense methods. Notably, the results of Pre in considerably lower ASRs compared to Pro, which relates to the mechanism through which our hijacking attack induces LLMs to generate target outputs, as discussed in Appendix Sec G. Although the Onion method is ineffective at defending against hijacking attacks in sentiment analysis tasks, it successfully protects LLMs from hijacking attacks in more complex topic generation tasks. Furthermore, the results indicate that all the defense methods are ineffective on small-sized LLMs, such as the GPT2-XL used in our experiments, due to their limited emergent abilities.
# 6.5 Transferability of GGI
Our GGI exhibits two advanced transferabilities: across different demo sets and across different datasets of the same task. Firstly, the adversarial tokens derived from any demo can be used in any ICL demo set. Once selected, these adversarial tokens consistently hijack LLMs regardless of the demos employed by developers or users, demonstrating their robustness and effectiveness. As illustrated in Figure 2, we evaluated the same adversarial tokens on three distinct demo sets from SST-2 and RT, respectively. Both sets resulted in high ASRs on both SST-2 and RT datasets, highlighting their transferability across different demo sets. Furthermore, the adversarial tokens, such as ‘NULL’ and ‘Remove,’ as illustrated in Figure 10 of the Appendix, used in
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6844/6844c657-7c08-4f9d-ba85-26444d1e00c3.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Transferability of GGI across different demo sets and different datasets of the same task. The normal and striped bars indicate the demos are from SST-2 and RT, respectively. Different colors represent test queries from different datasets.</div>
Figure 2: Transferability of GGI across different demo sets and different datasets of the same task. The normal and striped bars indicate the demos are from SST-2 and RT, respectively. Different colors represent test queries from different datasets. sentiment analysis tasks were learned from the RT dataset and effectively applied to the SST-2 dataset. Our attack GGI achieves promising adversarial attack success rates on both SST-2 and RT datasets, as demonstrated by Figure 2.
sentiment analysis tasks were learned from the RT dataset and effectively applied to the SST-2 dataset. Our attack GGI achieves promising adversarial attack success rates on both SST-2 and RT datasets, as demonstrated by Figure 2.
# 6.6 Stealthiness of GGI
Figure 3 presents the perplexity scores for the input prompts from different attack methods. The perplexity scores for the word-level adversarial attacks, i.e., Greedy, Square, and Ours, exhibit nonsignificant increases compared to the clean samples, highlighting their stealthiness. This demonstrates that using a perplexity-based filter, e.g., Onion (Qi et al., 2020), would be challenging to defend against our attacks. However, the character-level attack TA, used in (Wang et al., 2023c), results in significantly higher perplexity scores than others. This makes it more easily detected or corrected by basic grammar checks, as illustrated in Figure 10 and Figure 11 in the Appendix.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/841b/841b9863-904f-4bee-87e2-f140f436891b.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Average perplexity scores from LLaMA-7b on 100 random samples under 4-shots setting of RT derived from three separate runs under various attacks.</div>
Figure 3: Average perplexity scores from LLaMA-7b on 100 random samples under 4-shots setting of RT derived from three separate runs under various attacks.
# 7 Related Work 7.1 In-Context Learning
LLMs have shown impressive performance on numerous NLP tasks (Devlin et al., 2018; Lewis et al., 2019; Radford et al., 2019). Although fine-tuning has been a common method for adapting models
to new tasks, it is often less feasible to fine-tune extremely large models with over 10 billion parameters. As an alternative, recent work has proposed ICL, where the model adapts to new tasks solely via inference conditioned on the provided in-context demos, without any gradient updates (Brown et al., 2020). By learning from the prompt context, ICL allows leveraging massive LLMs’ knowledge without the costly fine-tuning process, showcasing an exemplar of the LLMs’ emergent abilities (Schaeffer et al., 2023; Wei et al., 2022). Intensive research has been dedicated to ICL. Initial works attempt to find better ways to select labeled examples for the demos (Liu et al., 2021; Rubin et al., 2021). For instance, (Liu et al., 2021) presents a simple yet effective retrievalbased method that selects the most semantically similar examples as demos, leading to improved accuracy and higher stability. Follow-up works have been done to understand why ICL works (Xie et al., 2021; Razeghi et al., 2022; Min et al., 2022; Wei et al., 2023a; Kossen et al., 2023). (Xie et al., 2021) provides theoretical analysis that ICL can be formalized as Bayesian inference that uses the demos to recover latent concepts. Another line of research reveals the brittleness and instability of ICL approaches: small changes to the demo examples, labels, or order can significantly impact performance (Lu et al., 2021; Zhao et al., 2021; Min et al., 2022; Nguyen and Wong, 2023).
# 7.2 Adversarial Attacks on LLMs
Early adversarial attacks on LLMs apply simple character or token operations to trigger the LLMs to generate incorrect predictions, such as TextAttack (Morris et al., 2020) and BERT-Attack (Li et al., 2020). Since these attacks usually generate misspelled and/or gibberish prompts that can be detected using spell checker and perplexitybased filters, they are easy to block in real-world applications. Some other attacks struggled with optimizing over discrete text, leading to the manual or semi-automated discovery of vulnerabilities through trial-and-error (Li et al., 2021; Perez and Ribeiro, 2022; Li et al., 2023c; Qiang et al., 2023; Casper et al., 2023; Kang et al., 2023; Li et al., 2023a; Shen et al., 2023). For example, jailbreaking prompts are intentionally designed to bypass an LLM’s built-in safeguard, eliciting it to generate harmful content that violates the usage policy set by the LLM vendor (Shen et al., 2023; Zhu et al., 2023b; Chao et al., 2023; Mehrotra et al., 2023;
Jeong, 2023; Guo et al., 2024; Yu et al., 2024). These red teaming efforts craft malicious prompts in order to understand LLM’s attack surface (Ganguli et al., 2022). However, the discrete nature of text has significantly impeded learning more effective adversarial attacks against LLMs. Recent work has developed gradient-based optimizers for efficient text modality attacks. For example, (Wen et al., 2023) presented a gradientbased discrete optimizer that is suitable for attacking the text pipeline of CLIP, efficiently bypassing the safeguards in the commercial platform. (Zou et al., 2023), building on (Shin et al., 2020), described an optimizer that combines gradient guidance with random search to craft adversarial strings that induce LLMs to respond to the questions that would otherwise be banned. More recently, (Zhao et al., 2024) proposed poisoning demo examples and prompts to make LLMs behave in alignment with pre-defined intentions. Our hijacking attack algorithm falls into this stream of work, yet we target few-shot ICL instead of zero-shot queries. We use gradient-based prompt search to automatically learn effective adversarial suffixes rather than manually engineered prompts. Importantly, we show that LLMs can be hijacked to output the targeted unwanted output by appending optimized adversarial tokens to the ICL demos, which reveals a new lens of LLM vulnerabilities that prior approaches may have missed.
# 7.3 Defense Against Attacks on LLMs
The existing literature on the robustness of LLMs includes various strategies for defense (Liu et al., 2023; Xu et al., 2024; Wu et al., 2024). However, most of these defenses, such as those involving adversarial training (Liu et al., 2020; Li et al., 2023b; Formento et al., 2024; Wang et al., 2024) or data augmentation (Qiang et al., 2024; Yuan et al., 2024), need to re-train or fine-tune the models, which is computationally infeasible for LLM users. Moreover, restricting many closed-source LLMs to only permit query access for candidate defenses introduces new challenges. Recent studies focus on developing defenses against attacks on LLMs that utilize adversarial prompting. (Jain et al., 2023) and (Alon and Kamfonas, 2023) have suggested using perplexity filters to detect adversarial prompts. While the filters are effective at catching the attack strings that contain gibberish words or character-level adversarial tokens with high perplexity scores, they fall short
in detecting more subtle adversarial prompts, like the ones used in our adversarial demo attacks with as low perplexity as clean samples shown in Figure 3. Recently, (Mo et al., 2023b) introduced a method to mitigate backdoor attacks at test time by identifying the task and retrieving relevant defensive demos. These demos are combined with user queries to counteract the adverse effects of triggers present in backdoor attacks. This defense strategy eliminates the need for modifications or tuning of LLMs. Its objective is to re-calibrate and correct the behavior of LLMs during test-time evaluations. Similarly, (Wei et al., 2023b) investigated the role of in-context demos in enhancing the robustness of LLMs and highlighted their effectiveness in defending against jailbreaking attacks. The authors developed an in-context defense strategy that constructs a safe context to caution the model against generating any harmful content. So far, defense mechanisms against adversarial demo attacks have not been extensively explored. Our approach introduces a test-time defense strategy that uses additional clean in-context demos to safeguard LLMs from adversarial in-context manipulations. In line with prior works (Mo et al., 2023b; Wei et al., 2023b; Wang et al., 2024), this defense strategy avoids the necessity for retraining or fine-tuning LLMs. Instead, it focuses on re-calibrating and correcting the behavior of LLMs during evaluations at test time.
# 8 Conclusion
This work reveals the vulnerability of ICL via crafted hijacking attacks. By appending imperceptible adversarial suffixes to the in-context demos using a greedy gradient-based algorithm, our attack GGI effectively hijacks the LLMs to generate the unwanted target outputs by diverting their attention from the relevant context to the adversarial suffixes. Furthermore, GGI can accomplish jailbreaking by adding adversarial suffixes to in-context demos, eliciting harmful responses while bypassing the safeguards in LLMs. The advanced transferability of GGI makes it significantly more efficient and scalable for real-world applications. GGI’s imperceptibility and stealthiness highlight the difficulty of defending against it with simple grammar checks and perplexity-based filters. We propose a test-time defense strategy that effectively protects LLMs from being compromised by our attack. We will continue studying novel attack and defense techniques for more robust ICL approaches.
# 9 Limitations and Risks
This work uncovers a potential vulnerability of LLMs during in-context learning. By inserting adversarial tokens, which our algorithm has learned, into in-context demos, we can make the LLM generate undesired target outputs without the need for a trigger in the query nor contaminating the user’s queries. This work represents a purple teaming effort to discover LLM’s vulnerabilities during in-context learning and defend against attacks. It offers a unified platform that enables both the red team and blue team to collaborate more effectively. Moreover, it facilitates a seamless knowledge transfer between the teams. As such, it will not pose risks for natural users or LLM vendors. Rather, our findings can be utilized by these stakeholders to guard against malicious uses and enhance the robustness of LLMs to such threats.
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774. Gabriel Alon and Michael Kamfonas. 2023. Detecting language model attacks with perplexity. arXiv preprint arXiv:2308.14132. Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion, and Matthias Hein. 2020. Square attack: a query-efficient black-box adversarial attack via random search. In European conference on computer vision, pages 484–501. Springer. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901. Stephen Casper, Jason Lin, Joe Kwon, Gatlen Culp, and Dylan Hadfield-Menell. 2023. Explore, establish, exploit: Red teaming language models from scratch. arXiv preprint arXiv:2306.09442. Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong. 2023. Jailbreaking black box large language models in twenty queries. arXiv preprint arXiv:2310.08419. Yanda Chen, Chen Zhao, Zhou Yu, Kathleen McKeown, and He He. 2022. On the relation between sensitivity and accuracy in in-context learning. arXiv preprint arXiv:2209.07661. Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. 2023. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2022. A survey for in-context learning. arXiv preprint arXiv:2301.00234. Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. 2017. Hotflip: White-box adversarial examples for text classification. arXiv preprint arXiv:1712.06751. Brian Formento, Wenjie Feng, Chuan Sheng Foo, Luu Anh Tuan, and See-Kiong Ng. 2024. Semrode: Macro adversarial training to learn representations that are robust to word-level attacks. arXiv preprint arXiv:2403.18423.
Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. 2022. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858.
Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, and Bin Hu. 2024. Cold-attack: Jailbreaking llms with stealthiness and controllability. arXiv preprint arXiv:2402.08679.
Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. 2023. Baseline defenses for adversarial attacks against aligned language models. arXiv preprint arXiv:2309.00614.
Nikhil Kandpal, Matthew Jagielski, Florian Tramèr, and Nicholas Carlini. 2023. Backdoor attacks for in-context learning with language models. arXiv preprint arXiv:2307.14692.
Xin Li, Xiangrui Li, Deng Pan, Yao Qiang, and Dongxiao Zhu. 2023c. Learning compact features via intraining representation alignment. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 8675–8683. Xin Li, Xiangrui Li, Deng Pan, and Dongxiao Zhu. 2021. Improving adversarial robustness via probabilistically compact loss with logit constraints. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 8482–8490. Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. 2021. What makes good in-context examples for gpt-3? arXiv preprint arXiv:2101.06804. Qin Liu, Fei Wang, Chaowei Xiao, and Muhao Chen. 2023. From shortcuts to triggers: Backdoor defense with denoised poe. arXiv preprint arXiv:2305.14910. Xiaodong Liu, Hao Cheng, Pengcheng He, Weizhu Chen, Yu Wang, Hoifung Poon, and Jianfeng Gao. 2020. Adversarial training for large neural language models. arXiv preprint arXiv:2004.08994. Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2021. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. arXiv preprint arXiv:2104.08786. Natalie Maus, Patrick Chao, Eric Wong, and Jacob R Gardner. 2023. Black box adversarial prompting for foundation models. In The Second Workshop on New Frontiers in Adversarial Machine Learning. Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin Karbasi. 2023. Tree of attacks: Jailbreaking black-box llms automatically. arXiv preprint arXiv:2312.02119. Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837. Lingbo Mo, Boshi Wang, Muhao Chen, and Huan Sun. 2023a. How trustworthy are open-source llms? an assessment under malicious demonstrations shows their vulnerabilities. arXiv preprint arXiv:2311.09447. Wenjie Mo, Jiashu Xu, Qin Liu, Jiongxiao Wang, Jun Yan, Chaowei Xiao, and Muhao Chen. 2023b. Testtime backdoor mitigation for black-box large language models with defensive demonstrations. arXiv preprint arXiv:2311.09763. John X Morris, Eli Lifland, Jin Yong Yoo, Jake Grigsby, Di Jin, and Yanjun Qi. 2020. Textattack: A framework for adversarial attacks, data augmentation, and adversarial training in nlp. arXiv preprint arXiv:2005.05909.
Tai Nguyen and Eric Wong. 2023. In-context example selection with influences. arXiv preprint arXiv:2302.11042. Bo Pang and Lillian Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the ACL. Fábio Perez and Ian Ribeiro. 2022. Ignore previous prompt: Attack techniques for language models. arXiv preprint arXiv:2211.09527. Pouya Pezeshkpour and Estevam Hruschka. 2023. Large language models sensitivity to the order of options in multiple-choice questions. arXiv preprint arXiv:2308.11483. Fanchao Qi, Yangyi Chen, Mukai Li, Yuan Yao, Zhiyuan Liu, and Maosong Sun. 2020. Onion: A simple and effective defense against textual backdoor attacks. arXiv preprint arXiv:2011.10369. Yao Qiang, Supriya Tumkur Suresh Kumar, Marco Brocanelli, and Dongxiao Zhu. 2022. Tiny rnn model with certified robustness for text classification. In 2022 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE. Yao Qiang, Chengyin Li, Prashant Khanduri, and Dongxiao Zhu. 2023. Interpretability-aware vision transformer. arXiv preprint arXiv:2309.08035. Yao Qiang, Xin Li, and Dongxiao Zhu. 2020. Toward tag-free aspect based sentiment analysis: A multiple attention network approach. In 2020 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE. Yao Qiang, Subhrangshu Nandi, Ninareh Mehrabi, Greg Ver Steeg, Anoop Kumar, Anna Rumshisky, and Aram Galstyan. 2024. Prompt perturbation consistency learning for robust language models. arXiv preprint arXiv:2402.15833. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9. Yasaman Razeghi, Robert L Logan IV, Matt Gardner, and Sameer Singh. 2022. Impact of pretraining term frequencies on few-shot reasoning. arXiv preprint arXiv:2202.07206. Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2021. Learning to retrieve prompts for in-context learning. arXiv preprint arXiv:2112.08633. Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. 2023. Are emergent abilities of large language models a mirage? arXiv preprint arXiv:2304.15004. Erfan Shayegani, Md Abdullah Al Mamun, Yu Fu, Pedram Zaree, Yue Dong, and Nael Abu-Ghazaleh. 2023. Survey of vulnerabilities in large language models revealed by adversarial attacks. arXiv preprint arXiv:2310.10844.
Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang. 2023. " do anything now": Characterizing and evaluating in-the-wild jailbreak prompts on large language models. arXiv preprint arXiv:2308.03825. Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. 2020. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. arXiv preprint arXiv:2010.15980. Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1631–1642. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971. Haoyu Wang, Guozheng Ma, Cong Yu, Ning Gui, Linrui Zhang, Zhiqi Huang, Suwei Ma, Yongzhe Chang, Sen Zhang, Li Shen, et al. 2023a. Are large language models really robust to word-level perturbations? arXiv preprint arXiv:2309.11166. Jindong Wang, Xixu Hu, Wenxin Hou, Hao Chen, Runkai Zheng, Yidong Wang, Linyi Yang, Haojun Huang, Wei Ye, Xiubo Geng, et al. 2023b. On the robustness of chatgpt: An adversarial and out-of-distribution perspective. arXiv preprint arXiv:2302.12095. Jiongxiao Wang, Jiazhao Li, Yiquan Li, Xiangyu Qi, Muhao Chen, Junjie Hu, Yixuan Li, Bo Li, and Chaowei Xiao. 2024. Mitigating fine-tuning jailbreak attack with backdoor enhanced alignment. arXiv preprint arXiv:2402.14968. Jiongxiao Wang, Zichen Liu, Keun Hee Park, Muhao Chen, and Chaowei Xiao. 2023c. Adversarial demonstration attacks on large language models. arXiv preprint arXiv:2305.14950. Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682. Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, et al. 2023a. Larger language models do in-context learning differently. arXiv preprint arXiv:2303.03846. Zeming Wei, Yifei Wang, and Yisen Wang. 2023b. Jailbreak and guard aligned language models with only few in-context demonstrations. arXiv preprint arXiv:2310.06387.
Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2023. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery. arXiv preprint arXiv:2302.03668. Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery. Advances in Neural Information Processing Systems, 36.
blum, Jonas Geiping, and Tom Goldstein. 2023. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery. arXiv preprint arXiv:2302.03668. Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery. Advances in Neural Information Processing Systems, 36. Fangzhou Wu, Ning Zhang, Somesh Jha, Patrick McDaniel, and Chaowei Xiao. 2024. A new era in llm security: Exploring security concerns in real-world llmbased systems. arXiv preprint arXiv:2402.18649. Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, and Lingpeng Kong. 2022. Self-adaptive in-context learning. arXiv preprint arXiv:2212.10375. Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2021. An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080. Jiashu Xu, Mingyu Derek Ma, Fei Wang, Chaowei Xiao, and Muhao Chen. 2023. Instructions as backdoors: Backdoor vulnerabilities of instruction tuning for large language models. arXiv preprint arXiv:2305.14710. Zihao Xu, Yi Liu, Gelei Deng, Yuekang Li, and Stjepan Picek. 2024. Llm jailbreak attack versus defense techniques–a comprehensive study. arXiv preprint arXiv:2402.13457. Zhiyuan Yu, Xiaogeng Liu, Shunning Liang, Zach Cameron, Chaowei Xiao, and Ning Zhang. 2024. Don’t listen to me: Understanding and exploring jailbreak prompts of large language models. arXiv preprint arXiv:2403.17336. Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, and Bo Li. 2024. Rigorllm: Resilient guardrails for large language models against undesired content. arXiv preprint arXiv:2403.13031. Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. Opt: Open pretrained transformer language models. Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In NIPS. Shuai Zhao, Meihuizi Jia, Luu Anh Tuan, and Jinming Wen. 2024. Universal vulnerabilities in large language models: In-context learning backdoor attacks. arXiv preprint arXiv:2401.05949.
Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In International Conference on Machine Learning, pages 12697–12706. PMLR. Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, et al. 2023a. Promptbench: Towards evaluating the robustness of large language models on adversarial prompts. arXiv preprint arXiv:2306.04528. Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, and Tong Sun. 2023b. Autodan: Automatic and interpretable adversarial attacks on large language models. arXiv preprint arXiv:2310.15140. Andy Zou, Zifan Wang, J Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043.
# A Experiments Details
Dataset Statistics: We show the dataset statistics in Table 5. Specifically for the SST-2 and RT sentiment analysis tasks, we employ only 2 training queries to train adversarial suffixes using our GGI method. We use 4 training queries for the more complex multi-class topic generation tasks, i.e., AG’s News. We randomly select 1,000 samples as user queries for testing. Similarly, we utilize 4 training queries from Advbench (Zou et al., 2023) for the jailbreaking task and evaluate the attack success rate on 200 randomly selected harmful queries.
<div style="text-align: center;">Table 5: Statistics of the training queries used in Algorithm 1 and test queries for the three datasets.</div>
Datasets
Training Queries
Test Queries
SST-2
2
1,000
RT
2
1,000
AG’s News
4
1,000
AdvBench
4
200
ICL Settings: For ICL, we follow the setting in (Wang et al., 2023c) and use their template to incorporate the demos for prediction. The detailed template is provided in Figure 9. We evaluate the 2-shot, 4-shot, and 8-shot settings for the number of demos. Specifically, for each test example, we randomly select the demos from the training set and repeat this process 5 times, reporting the average accuracy over the repetitions. Evaluation Metrics: Several different metrics evaluate the performance of ICL and hijacking attacks. Clean accuracy evaluates the accuracy of ICL on downstream tasks using clean demos. Attack accuracy evaluates the accuracy of ICL given the perturbed demos. Defense accuracy demonstrates the accuracy of ICL with the defense method against the hijacking attack. We further evaluate the effectiveness of hijacking attacks using attack success rate (ASR). Given a test sample (x, y) from a test set D, the clean and perturbed prompts are denoted as p = [I; C; x] and p′ = [I; C′; x], respectively. For the general generation tasks, such as sentiment analysis and news topic generation, ASR is calculated as
(5)
where 1 denotes the indicator function and yT ̸= y. For the jailbreaking task, ASR is calculated as:
(6)
# where y represents a refusal response by safeguards and yH here denotes the harmful response.
where y represents a refusal response by safeguards and yH here denotes the harmful response.
# B Baseline Attacks
Greedy Search: We consider a heuristics-based perturbation strategy, which conducts a greedy search over the vocabulary to select tokens, maximizing the reduction in the adversarial loss from Eq. 3. Specifically, it iteratively picks the token that decreases the loss the most at each step. Square Attack: The square attack (Andriushchenko et al., 2020) is an iterative algorithm for optimizing high-dimensional black-box functions using only function evaluations. To find an input x + δ in the demo set C that minimizes the loss in Eq. 3, the square attack has three steps: Step 1: Select a subset of inputs to update; Step 2: Sample candidate values to substitute for those inputs; Step 3: Update x + δ with the candidate values that achieve the lowest loss. The square attack can optimize the hijacking attack objective function without requiring gradient information by iteratively selecting and updating a subset of inputs. Text Attack: We also utilize TextAttack (TA) (Morris et al., 2020), adopting a similar approach to the attack described by (Wang et al., 2023c), which serves as the most closely related baseline for our hijacking attack. Unlike our word-level attack, the use of TA at the character level includes minor modifications to some words in the in-context demos and simply flips the labels of user queries, as depicted in Figure 8. In our experiments, we employ a transformation where characters are swapped with those on adjacent QWERTY keyboard keys, mimicking errors typical of fast typing, as done in TextAttack (Morris et al., 2020). Specifically, we use the adversarial examples for the same demos in our hijacking attack during the application of TA.
# C Attack Performance
In addition to the attack accuracy performance provided in Table 1 and 2, we present ASRs for various attacks across the three datasets. As outlined in Table 6, our GGI attack achieves the highest ASRs, substantiating its highest effectiveness in hijacking the LLM to generate the targeted output. In sentiment analysis tasks like SST-2 and RT, some attacks exhibit high ASRs. Meanwhile, for the more complex multi-class topic generation task, such as AG’s News, only our GGI attack achieves
<div style="text-align: center;">Table 6: ASR among different datasets, models, and attack methods. Best scores are in bold.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4e57/4e57168d-4a58-4fe6-a191-3e07aafcb6ed.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/27dd/27ddc9e0-83aa-42a8-b767-7de63d9aaea5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Impact of LLM size on adversarial robustness. ASRs on the AG’s News topic generation task using different sizes of OPT models, i.e., OPT-2.7b and OPT6.7b, with two different few-shot settings.</div>
Figure 4: Impact of LLM size on adversarial robustness. ASRs on the AG’s News topic generation task using different sizes of OPT models, i.e., OPT-2.7b and OPT6.7b, with two different few-shot settings.
high ASRs. This further emphasizes the potential effectiveness of our hijacking attack on more complex generative tasks, such as question answering.
# D Impact of Number of In-context Demos
We extend our investigation to explore the impact of in-context demos on adversarial ICL attacks. We observe a substantial impact on the attack performance in ICL based on the number of demos employed. As indicated in Tables 1 and 2, an increase in the number of in-context demos correlates with a higher susceptibility of the attack to hijack LLMs, resulting in the generation of target outputs with greater ease. Specifically, in the 8-shot setting, LLMs consistently exhibit significantly lower accuracies in negative sentiment generation, demonstrating a higher rate of successful attacks compared to the 2-shot and 4-shot settings. Moreover, the attacks demonstrate higher ASRs as the number of in-context demos used in ICL increases, as shown in Table 6.
# E Impact of Sizes of LLMs
Results in Table 6 reveal that the ASRs on GPT2XL are significantly higher than those on LLaMA7b, suggesting that hijacking the larger LLM is more challenging. Here, we continue examining how the size of LLMs influences the performance of hijacking attacks. Table 7 illustrates the performance of sentiment analysis tasks with and without attacks on ICL using different sizes of OPT, i.e., OPT-2.7b and OPT-6.7b. These results further highlight that the smaller LLM, i.e., OPT-2.7b, is much easier to attack and induce to generate unwanted target outputs, such as ‘positive’, in the sentiment analysis tasks. Figure 4 illustrates our proposed hijacking attack performance using ASR on two OPT models of varying sizes in AG’s News topic generation task. It clearly shows that attacking the smaller OPT2-2.7b model achieves a much higher ASR in both settings, confirming our finding and others (Wang et al., 2023a) that larger models are more resistant to adversarial attacks.
# F Comparison of Hijacking Attacks
In contrast to baseline hijacking attacks, i.e., Square and Greedy, our GGI exhibits superior performance in generating targeted outputs, as evidenced by the results in Table 1 and 2, along with the highest ASRs highlighted in Table 6. This underscores the effectiveness of GGI as a more potent method of attack. To further illustrate the efficiency of our GGI, we present the objective function values of Eq. 3 in Figure 5 for various attack methods. Since our GGI attack enjoys the advantages of both greedy and gradient-based search strategies as depicted in Algorithm 1, the values of the object function decrease steadily and rapidly, ultimately reaching
<div style="text-align: center;">Table 7: The performance of sentiment analysis task with and without attacks on ICL using different sizes of OP</div>
Model
Method
SST-2
RT
2-shots
4-shots
8-shots
2-shots
4-shots
8-shots
P
N
P
N
P
N
P
N
P
N
P
N
OPT-2.7b
Clean
98.5
38.6
85.6
62.8
58.4
76.4
98.1
36.6
81.2
68.4
57.8
89.6
Square
100
0.0
100
0.0
100
1.8
100
1.3
100
0.0
99.6
7.5
Greedy
100
0.0
100
0.0
100
0.0
100
0.4
100
0.2
100
0.0
TA
99.6
13.8
99.8
26.8
99.0
7.2
97.6
52.9
97.2
59.7
99.4
6.8
GGI
100
0.0
100
0.0
100
0.0
100
0.0
100
0.0
100
0.0
OPT-6.7b
Clean
69.4
87.8
70.2
93.8
77.8
93.0
84.4
91.4
84.4
93.1
88.6
92.8
Square
99.2
31.4
93.8
72.2
99.6
29.0
98.1
42.2
97.0
68.7
99.4
33.2
Greedy
100
25.0
97.8
39.0
100
2.0
99.4
31.7
99.8
4.7
100
0.8
TA
94.8
80.8
54.8
98.6
91.6
89.4
92.5
86.1
77.6
96.4
94.0
86.3
GGI
100
0.0
98.4
2.0
100
0.2
100
2.6
99.8
0.0
100
0.2
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c2e9/c2e901c5-4296-4fe4-b050-02a9cf6f10de.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: An illustration of the learning objective values during iterations among different attacks on SST2 using GPT2-XL with 8-shots.</div>
the minimum loss value. On the other hand, both the Square and Greedy attacks use a greedy search strategy, with fluctuating results that increase and decrease the loss value, unable to converge to the minimum loss value corresponding to the optimal adversarial suffixes.
# G Diverting LLM Attention
Attempting to interpret the possible mechanism of our hijacking attacks, we show an illustrative example using attention weights from LLaMA-7b on the SST2 task with both clean and perturbed prompts. As depicted in Figure 6b, the model’s attention for generating the sentiment token of the test query has been diverted towards the adversarial suffix tokens ‘NULL’ and ‘Remove’. Compared to the attention maps using the clean prompt (Figure 6a), these two suffixes attain the largest attention weights represented by the darkest green color. This example illuminates a possible mechanism for why our hijacking attack can induce the LLM to generate the targeted outputs - the adversarial suffixes divert the LLMs’ attention away from the original query. Additionally, Figure 7 illustrates the attention
distribution for the perturbed prompts after applying the preceding and proceeding defense methods. Notably, in the demos, the model primarily focuses on the front segments of demos, which are indicated by a darker green color. Therefore, the model converts its attention to the front segments, which are the extra clean samples, in the preceding method. These clean samples effectively re-calibrate and rectify the model’s behavior, leading to a significant reduction in ASRs, as shown in Table 3. In contrast, the first few demos remain adversarial in the proceeding method, rendering it ineffective in defending against the adversarial demo attack. Overall, these attention maps visualize how the adversarial suffixes distract LLMs from focusing on the relevant context to generate the unwanted target output and how our proposed defense methods rectify the behavior of LLMs given the extra clean demos.
# H More Results
Figure 9 illustrates the prompt template employed in ICL for various tasks. For the SST2/RT dataset, the template is structured to include an instruction, a demo set composed of reviews and sentiment labels, and the user query. Similarly, the AG’s News dataset template comprises the instruction, the demo set with articles and topic labels, and the user query. The AdvBench template includes instructions, a demo set of harmful queries and responses, and a user’s harmful query. Additionally, examples are provided in Figure 10, Figure 11, and Figure 12 to enhance understanding.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6f42/6f42c8ad-d1b7-447e-8e21-808de9672696.png" style="width: 50%;"></div>
Figure 6: Attentions maps generated using (a) clean and (b) adversarial perturbed prompts. In (b), the adversarial suffix tokens, i.e., ‘NULL’ and ‘Remove’, are underlined in red. Darker green colors represent larger attention weights. The prompts are tokenized to mimic the actual inputs to the LLMs. Best viewed in color.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f406/f406b696-25de-4267-9847-9fc0236af134.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/11aa/11aa661f-542f-45c4-85b1-1225e7125a31.png" style="width: 50%;"></div>
Figure 8: Illustrations of ICL using clean prompt and adversarial prompt. Given the clean in-context demos, LLMs can correctly generate the sentiment of the test queries. The previous attacks (Wang et al., 2023c) at the character level involve minor edits in some words, such as altering ‘so’ to ‘s0’ and ‘film’ to ‘fi1m’, of these in-context demos, leading to incorrect sentiment generated for the test queries. However, ours learns to append adversarial suffixes like ‘For’ and ‘Location’ to the in-context demos to efficiently and effectively hijack LLMs to generate the unwanted target, e.g., the ‘negative’ sentiment, regardless of the test query content. It is important to highlight that the adversary attacker only needs to append the adversarial tokens to either the system or the user-provided demos without compromising the user’s queries directly.
Algorithm 1: Greedy Gradient-guided Injection (GGI)
Input
: Model: M, Iterations: T, Batch Size: b, Instruction: I, Demos: C, Query: (xQ, yQ)
Target: yT
Initialization: p′
0 = [I; [S(x1 + δ1, y1); · · · ; S(xN + δN, yN)]; S(xQ, yT )]
repeat
for i ∈N do
[δi1; ...; δik] = Top−k(−∇p′L(M(ˆy|p′
t−1), yT ))
/* Compute top-k substitutions */
K = {[δi1; ...; δik] | i = 1, ..., N}
B = {(δi1, . . . , δib) | (δi1, . . . , δik) ∈K}
/* Introducing variability by selecting different
subsets of substitutions in each iteration
helps avoid local minima */
for i ∈N do
δ⋆
i = δij, where j = argminδibL(M(ˆy|p′
t−1), yT )
/* Compute best replacement */
∆= [δ⋆
1; ...; δ⋆
N]
p′
t = [I; [S(x1 + δ⋆
1, y1); · · · ; S(xN + δ⋆
N, yN)]; S(xQ, yT )]
/* Update prompt */
until T times;
Output :Optimized prompt suffixes [δ⋆
1, · · · , δ⋆
N]

Figure 9: Template designs for all the datasets used in our experiments. We also provide examples for these datasets to ensure a better understanding.


<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1f09/1f09a921-a634-4329-974d-44e5550325c3.png" style="width: 50%;"></div>

