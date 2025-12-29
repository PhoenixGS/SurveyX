#  Demonstration Shortcut in In-Context Learni
# onwon Jang1 Sanghwan Jang2 Wonbin Kweon3 Minjin Jeon1 Hwanjo Yu1,2,∗
Graduate School of AI, POSTECH1 Department of Computer Science and Engineering, POSTECH2 Institute of Artificial Intelligence, POSTECH3 kaoara, s.jang, kwb4453, minjinj, hwanjoyu}@postech.ac.kr
# Abstract
Large language models (LLMs) are able to solve various tasks with only a few demonstrations utilizing their in-context learning (ICL) abilities. However, LLMs often rely on their pre-trained semantic priors of demonstrations rather than on the input-label relationships to proceed with ICL prediction. In this work, we term this phenomenon as the ‘Demonstration Shortcut’. While previous works have primarily focused on improving ICL prediction results for predefined tasks, we aim to rectify the Demonstration Shortcut, thereby enabling the LLM to effectively learn new inputlabel relationships from demonstrations. To achieve this, we introduce In-Context Calibration, a demonstration-aware calibration method. We evaluate the effectiveness of the proposed method in two settings: (1) the Original ICL Task using the standard label space and (2) the Task Learning setting, where the label space is replaced with semantically unrelated tokens. In both settings, In-Context Calibration demonstrates substantial improvements, with results generalized across three LLM families (OPT, GPT, and Llama2) under various configurations. 1
15 Apr 2024
arXiv:2403.09488v3
# 1 Introduction
Large language models (LLMs) have demonstrated their effectiveness on a wide range of tasks through in-context learning (ICL), where models learn to perform a task from demonstrations (Brown et al., 2020). Leveraging their pre-trained knowledge, LLMs can associate various words in the demonstration with specific semantics (e.g., associating ‘extremely painful’ with ‘negative’), thereby performing new tasks using only a small set of inputlabel examples, without requiring parameter updates (Dong et al., 2022; Wei et al., 2023).
* Corresponding author 1https://github.com/Lainshower/ In-Context-Calibration.git
However, LLMs often rely on the semantics from their pre-trained knowledge of given demonstrations, resulting in insufficient task learning for the patterns of the provided input-label pairs. (Reynolds and McDonell, 2021; Min et al., 2022; Wei et al., 2023; Pan et al., 2023). This issue intensifies as the model size decreases (Wei et al., 2023). Kossen et al. (2023) suggest that smaller LLMs show promise in learning new mappings from demonstrations in some tasks, yet they still struggle to override semantic priors acquired during pre-training. Therefore, it is necessary to develop a method that enables LLMs of various sizes to effectively mitigate semantic priors preferences and learn to perform unseen tasks from demonstrations. Prior works have primarily focused on the instabilities of LLMs in ICL prediction (Holtzman et al., 2021; Fei et al., 2023). To mitigate these instabilities, these studies introduced content-free tokens or utilized the entire test set to calibrate prediction probabilities (Holtzman et al., 2021; Fei et al., 2023; Zhou et al., 2023). However, they lack consideration of the semantic priors of LLMs on the demonstration and do not verify whether their approach enhances LLMs to learn new tasks from the demonstrations. In this work, we investigate how the LLMs’ pretrained knowledge on the demonstrations affects ICL. We define the following phenomenon as a Demonstration Shortcut: the reliance of LLMs on their pre-trained semantic priors of demonstrations in ICL prediction, rather than learning from the input-label relationships presented in these demonstrations. Due to the Demonstration Shortcut, LLMs’ ICL predictions may be overly dependent on the semantics of the given demonstrations even when the label distribution is uniform and the order is identical (Figure 1). To tackle this problem, we propose In-Context Calibration, a method designed to rectify the Demonstration Shortcut in ICL. In-Context Cal-
ibration estimates the semantic prior of LLMs on each demonstration sample with the in-context examples. Formally, for each example in the demonstration, we estimate its semantic prior relative to the remaining examples and calculate the expected semantic priors of the demonstrations. At test time, we use this term to rectify LLMs’ dependency on semantic priors and enable the model to learn the intended input-label relationships from the demonstrations. We evaluate the effectiveness of In-Context Calibration on 27 classification datasets from two perspectives: (1) Original ICL Task and (2) Task Learning settings. In the Original ICL Task, we use the standard label space. In the Task Learning setting, the label space is replaced with semantically unrelated tokens. This requires LLMs to learn the novel input-label relationships to achieve high performance, as these relationships are never seen in pre-training. Our proposed method not only demonstrated enhanced performance across various tasks but also showed improvement in task learning abilities. Specifically, In-Context Calibration outperforms other ICL methods in Natural Language Inference (NLI) tasks, which demand high task learning ability. We also demonstrate that In-Context Calibration enhances ICL performance across various model types and sizes, effectively rectifying the ‘Demonstration Shortcut’ problem.
# 2 Backgrounds
How LLMs utilize demonstrations in ICL Following ICL’s accomplishments, extensive prior works have sought to understand how LLMs use demonstrations, yet there is still no consensus on the following two contradictory perspectives. One line of research claims that LLMs do not learn new input-label relationships from the demonstrations, with the evidence that ICL performance only marginally drops when labels in the demonstrations are replaced with random labels (Min et al., 2022). Instead, LLMs independently recognize the semantics of input and label of in-context demonstrations using their pre-trained knowledge and perform ICL prediction with its language modeling objective (Reynolds and McDonell, 2021; Min et al., 2022). On the other hand, while some studies suggest that LLMs can learn novel tasks through demonstrations, there is a notable lack of concrete experimental proof in real-world LLM applications (Xie et al., 2021; Zhang et al., 2023). Addressing this
gap, Wei et al. (2023) provides evidence that larger LLMs can learn the new input-label mappings from demonstrations. Summarizing these perspectives, Pan et al. (2023) show that applying pre-trained knowledge to demonstrations for task recognizing is a broad capability across scales while learning new input-label mappings becomes more feasible as the scale increases. This indicates that as LLMs decrease in size, they rely more on pre-trained knowledge of demonstrations in ICL prediction. Furthermore, when labels in the demonstrations are flipped with different labels (e.g., labeling ‘positive’ as ‘negative’ and vice versa), these models struggle to override semantic priors obtained during pre-training. (Wei et al., 2023; Kossen et al., 2023). Improving ICL through Calibration Various studies have focused on the instability of ICL prediction in LLMs. (Zhao et al., 2021; Jiang et al., 2023) reveal the instability of ICL prediction arises from the majority label bias and recency label bias, and Fei et al. (2023) identifies domain as a factor contributing to label bias behind this instability. These studies have attempted to estimate the instability of ICL prediction by introducing content-free tokens (Zhao et al., 2021) or using the entire test set to calibrate ICL prediction probabilities (Fei et al., 2023; Zhou et al., 2023). Although their approaches show an improvement in ICL prediction, they do not address the reliance of LLMs on the semantic priors of the demonstration. In other words, the primary objective of these methods is to enhance ICL prediction performance for pre-defined tasks, rather than enabling the model to learn inputlabel mappings from the demonstrations. Furthermore, they fail to demonstrate whether their calibration method allows LLMs to learn new input-label mappings through demonstrations. Moreover, it is unreasonable to assume that the entire test set is available when learning new input-label relationships. Based on the discussions above, our analysis focuses on the reliance of LLMs on their semantic priors of demonstrations, offering a novel perspective on the calibration method.
# 3 Demonstration Shortcut
In this section, we first introduce a new typology termed Demonstration Shortcut. This concept refers to the reliance of LLMs on their pretrained semantic priors of demonstrations for making ICL predictions, rather than learning the input-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3462/346204c1-74ba-40c4-befa-da2235f0fc14.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7a60/7a60a8fb-6407-4c6d-800d-d41f3995ab33.png" style="width: 50%;"></div>
# In-Context Learning
Figure 1: The overall illustration of the Demonstration Shortcut. In a zero-shot setting, an LLM predicts the test article to the world label. With the first demonstration set, the LLM predicts the business label through ICL. However, with the second demonstration set — which has the same label order but different semantics in the examples — the LLM predicts the sports label. GPT-J is used for these experiments. See Appendix A for a full description of the demonstrations.
label relationships presented in the demonstrations. Figure 1 illustrates the underlying concept of the Demonstration Shortcut. In a zero-shot setting, the LLM predicts the test article to the world label (the ground-truth label is technology). Next, we constructed two demonstration sets from the same training dataset, each with a uniform label distribution. While these sets have identical label orders, they differ in the semantics of the examples. The first demonstration set mostly features businessrelated semantics across all examples, while the second set leans more toward sports-related semantics. After ICL with the first demonstration set, the LLM predicted the article to the business label. However, LLM predicted the article to the sports label with the second demonstration set. Despite both sets having uniform label distributions and identical orders, LLM relies on the semantic prior from each demonstration set, exhibiting a Demonstration Shortcut and failing to predict the correct answer. This indicates that an over-dependence on semantic prior interrupts the ability of LLMs
<div style="text-align: center;">0 0.1 0.2 0.3 0.4 0.5 0.6</div>
Demo #1
Demo #2
Demo #3
Demo #4
World
0.20
0.22
0.26
0.17
Sports
0.25
0.30
0.25
0.25
Business
0.34
0.28
0.25
0.27
Technology
0.19
0.18
0.21
0.28
Table 1: Prediction distributions of GPT-J with different demonstration semantics sets. All four demonstration sets have uniform and identical label distribution.
to learn the new input-label mapping relationships from the demonstrations. To deeply describe how semantic priors acquired from pre-training affect ICL prediction, consider the first example in Demo #1 of Figure 1, titled ‘Vietnam Hosts Investment Conference - Hoping to Boost Business Ties with Singapore ∼’ (see Appendix A for the full example). The overall context of the article allows the LLM’s pre-training knowledge to associate its semantics with the business label. Meanwhile, word-by-word examination (e.g., Vietnam or Singapore) also reveals potential associations of its pre-trained semantics with
the world label (Tang et al., 2023). By iterating this process for all examples, regardless of the assigned ground-truth label, the LLM may proceed with ICL prediction based on pre-trained semantic distributions of the demonstrations, leading to the Demonstration Shortcut. Substantiating our intuition, we conducted additional experiments based on Figure 1. We constructed Demo #3 to be characterized by world semantics across all examples, while Demo #4 features across technology-related semantics. All demonstration sets were designed to have a uniform label distribution and an identical sequence, as depicted in Figure 1. We ensured the test set also followed a uniform label distribution (25 samples for each label) and reported the average label prediction probabilities for these examples in Table 1. Despite having uniform and identical label distributions in the demonstrations, the LLM predictions for all demonstration sets exhibit different behaviors; this variance aligns with the overall semantics of each demonstration.
# 4 In-Context Calibration
This section introduces a novel calibration method to rectify the Demonstration Shortcut. We first revisit existing calibration methods designed to improve ICL predictions and analyze their limitations, particularly regarding Demonstration Shortcut. We then propose In-Context Calibration, our approach to rectifying the Demonstration Shortcut in ICL.
Revisiting Previous Methods Prior works on calibrating LLMs attempt to adjust ICL predictions by estimating the prompt prior with a content-free token ‘N/A’ (Zhao et al., 2021) or by estimating the task prior by utilizing the entire test distribution (Fei et al., 2023; Zhou et al., 2023). Specifically, Contextual Calibration (CC) estimates the content-free prediction prior as PLM(y|‘N/A’, [(xi, yi)]i∈[K]), where xi represents the text input, yi corresponds to a verbalized label name, and K denotes the total number of examples. Meanwhile, Domain Calibration (DC) estimates task prior with 1 M �M r=1 PLM(y|[r/w]r, [(xi, yi)]i∈[K]), where [r/w] represents random words drawn from the entire test set. However, the introduced terms are not entirely content-free; their neutrality depends on the task type and the demonstrations (Fei et al., 2023; Zhou et al., 2023). Therefore, if these terms’ pre-trained semantics mismatch with the semantic
priors of demonstrations, they are limited in rectifying the Demonstration Shortcut. Moreover, relying on the entire test distribution is impractical in real-world settings.
In-Context Calibration To overcome these limitations, we propose In-Context Calibration, which rectifies the Demonstration Shortcut of the model in the ICL setting. For each xi in the demonstration, we estimate the semantic prior of LLMs on each demonstration sample with remaining in-context demonstrations:
where Pi represents the semantic distribution of xi given the remaining K −1 demonstrations. This allows us to estimate the contextual semantic prior of each demonstration sample while preserving the remaining in-context demonstrations order and conditions present in the original ICL setting. Additionally, we estimate each demonstration sample’s word-by-word semantic distribution by applying the random shuffling function R() to xi, which shuffles the order of the words as follows:
PR(i) = PLM(y | R(xi), [(xj, yj)]j∈[K]\{i}), ∀i. (
(2)
The resulting random order of R(xi) is not grammatically meaningful, yet it enables contextagnostic estimation of the LLMs’ pre-trained semantics for each demonstration sample. In other words, R(xi) is stripped of context and retains only the words with their meanings, thereby preventing the LLMs from making predictions based solely on the semantics of individual words (Fei et al., 2023; Tang et al., 2023) 2 We iterate this process across all K demonstrations and compute the average to estimate the semantic priors of the demonstrations:
where λ is the hyperparameter that controls the balance between the two terms. Considering every dependency from the K demonstrations and taking the average, we can estimate the expected semantic priors of demonstrations, enabling more effective demonstration-aware calibration. The model then makes ICL predictions based on the following estimates:
2Please refer to Appendix E for additional details for the random shuffling function.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b5d1/b5d19a35-57dc-46b5-a13e-d92a1fad82ad.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Averaged Macro F1 scores for OPT (Top), GPT (Medium), and Llama2 (Bottom) across Sentiment, NLI, and Detection Tasks. The left the left three columns depict the performance on the Original ICL Task. The right hree columns plot the Task Learning scores. In both graphs, the x-axis represents the model size.</div>
(4)
where PLM(y | xtest, [(xi, yi)]i∈[K]) is the original ICL prediction.
# 5 Experimental Setups
We investigate the effectiveness of In-Context Calibration in two aspects: (1) the model’s performance on the Original ICL Task (using standard label space from the dataset) and (2) the Task Learning setting (label space is randomly mapped to semantically unrelated tokens), following the experimental settings of Pan et al. (2023). Since the newly introduced input-label mappings have never been parameterized during pre-training, the model must utilize its task learning abilities to handle the problem. In our main experiment, unless stated otherwise, we conduct task learning experiments using string numbers3, and to avoid any pre-trained bias, each label is randomly assigned to a unique string number for every seed.
3String numbers demonstrate better task learning ability than other tokens in Pan et al. (2023).
# 5.1 Datasets
We conducted experiments on 27 classification datasets across three types of tasks: Sentiment, NLI, and Detection classification task. Our dataset selection and prompts largely follow the methodologies of prior ICL works (Zhao et al., 2021; Min et al., 2022; Fei et al., 2023), and more details are described in Appendix B.
# 5.2 Base Models
To validate our method across a diverse set of models, we use three state-of-the-art LLM families: GPT (2.7B, J (6B), 20B) (Brown et al., 2020), OPT (2.7B, 6.7B, 13B) (Zhang et al., 2022), and Llama2 (7B, 13B) (Touvron et al., 2023). For the GPT models, we use the open-sourced versions provided by EleutherAI (Gao et al., 2020; Wang, 2021; Black et al., 2022) as Fei et al. (2023). Consequently, we utilize checkpoints from the Transformers library (Wolf et al., 2020) for all the aforementioned models.
# 5.3 Implementation Details
Adopting a sampling-based evaluation approach, we sample different sets of demonstrations from the training set for each seed and report the mean and standard deviation of the results. We use K = 8 ex-
amples and conduct five evaluations using different random seeds, per the methodology described by Fei et al. (2023). Unless stated otherwise, we set λ to 0.5. For the baselines, we selected Contextual Calibration (CC) (Zhao et al., 2021) and Domain Calibration (DC) (Fei et al., 2023) to assess the performance on the Original ICL Task and Task Learning setting. For Domain Calibration, the original method involves constructing a bag of words from the entire test set, which is impractical for real-world inference. To facilitate a fair comparison, we adapted this method. The adapted version samples random words from the demonstration and is labeled as “w/ Demo” (with Demo), while we refer to the original method as “w/ Test” (with Test).
# 6 Experimental Results
# 6.1 Main Results
Figure 2 shows the Macro F1-scores of OPT, GPT, and Llama2 on three categories of datasets in the Original ICL Task and Task Learning setting. The results demonstrate that In-Context Calibration effectively rectifies the Demonstration Shortcut, enhancing performance in the Original ICL Task and improving learning capabilities for new input-label tasks. Specifically, for Llama2, In-Context Calibration resulted in an average F1 score improvement of 23% compared to the original inference in the Original ICL Task. Regarding CC using ‘N/A’ token and DC w/ Test sampling random words from the test set, if the newly introduced tokens fail to accurately estimate the neutrality of the demonstration’s semantic distribution, the model remains constrained by the Demonstration Shortcut, limiting performance improvements. This is particularly pronounced in NLI tasks, where previous works (Pan et al., 2023; Kossen et al., 2023) have shown to be challenging for LLMs to learn input-label pairs due to strong reliance on semantic priors in the ICL setting, while In-Context Calibration effectively rectifies the Demonstration Shortcut for all LLMs, resulting in notable performance increases. This improvement is also evident in the Task Learning setting, where the label space is replaced with string numbers. For GPT, In-Context Calibration achieved an average F1 Score increase of 27% over the original inference. Particularly in the NLI task, other methodologies struggled to mitigate the dependency of the semantic priors of the models on demonstration, leading to decreased performance compared to original inference in some
cases. Across various dataset categories and model types, In-Context Calibration consistently outperformed baseline methods on tasks with novel inputlabel pairs by effectively reducing the model’s reliance on the demonstration’s semantic prior (see Appendix G for comprehensive results).
# 6.2 Ablation Study
Method
Original
TL
Original Inference
0.48
0.40
R(xi) −→‘N/A’
0.49
0.38
R(xi) −→random words
0.57
0.46
In Context Calibration (λ = 0.5)
0.60
0.49
In Context Calibration (λ = 0)
0.56
0.46
In Context Calibration (λ = 1)
0.58
0.50
Table 2: Analysis of In-Context Calibration: Performance comparing R(xi) replacement with ‘N/A’ (CC) or randomly sampled test set words (DC), and effects of varying λ values, shown through average Macro F1 scores across 27 datasets in Original ICL Task (Original) and Task Learning (TL) settings using GPT-J.
We conducted an ablation study of the proposed method. First, we replaced the R(xi) term with either a ‘N/A’ token or words randomly sampled from the bag of words constructed from the entire test set. The replacement with the ‘N/A’ token resulted in lower performance compared to the original inference in the Task Learning setting, likely due to the instability of the ‘N/A’ token in estimating prompt neutrality (Fei et al., 2023). Furthermore, words randomly sampled from the test set underperformed in both tasks compared to InContext Calibration with λ = 0.5. This underperformance could stem from the semantic distribution mismatch between the sampled words and the demonstrations, limiting the model’s ability to rectify the Demonstration Shortcut. However, the demonstration-aware calibrating of In-Context Calibration leads to performance improvements. Ablation studies on λ demonstrate that using only R(xi) for calibration (λ = 0), which shuffles the token order for each demonstration, leads to limited performance gains in both the Original ICL Task and Task Learning setting, attributing to the loss of contextual information. On the other hand, not utilizing R(xi) for calibration (λ = 1) results in better learning of new input-label relationships. However, this approach is less effective in mitigating the model’s dependency on word-wise pre-trained semantics of demonstration when considering labels in the Original ICL Task. We con-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1d61/1d61ad4a-058d-49c7-83cb-9d35996b8119.png" style="width: 50%;"></div>
Figure 3: Averaged Macro F1 scores for Llama2-Chat across Sentiment, NLI, and Detection Tasks. The left thre columns depict the performance of the Original ICL Task. The right three columns plot the Task Learning scores. I both graphs, the x-axis represents the model size.
duct a detailed task-wise analysis of the λ value’s effect in Section 6.5 and Appendix F. Therefore, after comprehensive analysis for λ values, we set λ to 0.5 in the main experiment.4
# 6.3 Analysis with Enhanced Models
Instruction-Tuned Model In previous studies (Min et al., 2022; Wei et al., 2023), instruction tuning has been demonstrated to strengthen the usage of semantic priors in the demonstration. Therefore, we conducted experiments to determine whether In-Context Calibration consistently rectifies the Demonstration Shortcut and improves task learning ability in instruction-tuned LLMs. As depicted in Figure 3, In-Context Calibration increases the model’s F1 score across all Original ICL Tasks and particularly shows substantial improvement over other calibration methods in the Task Learning setting. Especially in the Task Learning setting, other calibration methods often underperformed the original inference. These experiments demonstrate that In-Context Calibration remains effective and enhances the model’s task learning abilities, even after instruction tuning has strengthened the LLM’s reliance on semantic priors. Over 50B Scale Models We conducted experiments to verify that the proposed method consistently improves performance in larger models. As demonstrated by Pan et al. (2023), in the Task Learning setting, the performance of some models (e.g., OPT) starts to match their performance in the Original ICL Task for models larger than 50B. These models can utilize the mapping information provided in the demonstrations. Therefore, we conducted experiments on OPT 66B, Llama2 70B, and Llama2-chat 70B, reporting average F1
Instruction-Tuned Model In previous studies (Min et al., 2022; Wei et al., 2023), instruction tuning has been demonstrated to strengthen the usage of semantic priors in the demonstration. Therefore, we conducted experiments to determine whether In-Context Calibration consistently rectifies the Demonstration Shortcut and improves task learning ability in instruction-tuned LLMs. As depicted in Figure 3, In-Context Calibration increases the model’s F1 score across all Original ICL Tasks and particularly shows substantial improvement over other calibration methods in the Task Learning setting. Especially in the Task Learning setting, other calibration methods often underperformed the original inference. These experiments demonstrate that In-Context Calibration remains effective and enhances the model’s task learning abilities, even after instruction tuning has strengthened the LLM’s reliance on semantic priors.
Over 50B Scale Models We conducted experiments to verify that the proposed method consistently improves performance in larger models. As demonstrated by Pan et al. (2023), in the Task Learning setting, the performance of some models (e.g., OPT) starts to match their performance in the Original ICL Task for models larger than 50B. These models can utilize the mapping information provided in the demonstrations. Therefore, we conducted experiments on OPT 66B, Llama2 70B, and Llama2-chat 70B, reporting average F1
4Please refer to the Appendix F for the comprehensive analysis for λ value.
scores for 27 datasets in both the Original ICL Task and Task Learning settings, as shown in Figure 4. Particularly for Llama2 70B, In-Context Calibration improved F1-score performance by 15% over the original inference in the Original ICL Task and by 9% in the Task Learning setting, while some methods hurt the model’s performance. These findings suggest that In-Context Calibration not only boosts performance in smaller-scale models but also consistently improves the learning ability of larger-scale models, by effectively rectifying the Demonstration Shortcut through its demonstrationaware calibration.
# 6.4 Analysis with Other Input-Label Mappings
Overridding Semantic Priors (Permutating Label Space) Wei et al. (2023) and Kossen et al. (2023) reveal that LLMs struggle to override semantic priors preference with input-label mappings in ICL setting. To test whether our proposed method helps the models override semantic priors preference by using its task learning abilities, we evaluate performance by randomly permuting the label space. For instance, in the AGNews dataset, the original label ‘sports’ is permuted to ‘world,’ ‘business’ data to ‘technology,’ ‘technology’ data to ‘sports,’ and ‘world’ data to ‘business’ to construct a demonstration set. The test labels are similarly permuted for evaluation. Due to semantic priors preferences, the models tend to rely more on their pre-trained knowledge than on learning new relationships from input-label pairs, resulting in lower performance on permuted datasets. We report the average F1 score across 27 datasets. Figure 5 illustrates that the GPT model outperforms other calibration methods using In-Context Calibration. For results related to other models, please refer to Figure 7 in Appendix G. This indicates
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3640/364090f4-1970-47fe-9882-4a66dffc80b7.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6cee/6cee9a2a-4ddb-48e6-9a84-2d6814068ad9.png" style="width: 50%;"></div>
<div style="text-align: center;">Original ICL</div>
Figure 4: Averaged Macro F1 scores across 27 classification tasks for over 50B scale LLMs. The left graphs depic performance in the Original ICL Task, while the right graphs plot task learning scores. In both sets of graphs, the x-axis denotes the model type. Full details of the data-type scores are provided in Appendix G.
<div style="text-align: center;">Figure 4: Averaged Macro F1 scores across 27 classification tasks for over 50B scale LLMs. The left graphs depict performance in the Original ICL Task, while the right graphs plot task learning scores. In both sets of graphs, the x-axis denotes the model type. Full details of the data-type scores are provided in Appendix G.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/28f4/28f41229-1610-44d8-a193-fab99579d3e2.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Averaged Macro F1 scores for the GPT model are presented across 27 classification tasks, each featuring a permuted label space. The x-axis represents the model size. Results for the OPT and Llama2 models are provided in Appendix G.</div>
Figure 5: Averaged Macro F1 scores for the GPT model are presented across 27 classification tasks, each featuring a permuted label space. The x-axis represents the model size. Results for the OPT and Llama2 models are provided in Appendix G.
that In-Context Calibration’s demonstration-aware calibrating is needed to diminish the model’s semantic priors preferences and let it learn new tasks from demonstrations, especially those that contradict pre-trained knowledge.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ef97/ef9756b4-491d-4eb9-a5f2-cb36e47d4679.png" style="width: 50%;"></div>
Figure 6: Averaged Macro F1 scores for the 6-7B scale model families are presented across 27 datasets with each label space replaced by symbol tokens. The x-axis represents the model type. Results for the 13-20B scale models are available in Figure 8.
<div style="text-align: center;">Task Learning</div>
Task Learning with different label mapping (Symbol Token) Pan et al. (2023) demonstrate that replacing the label space with symbols in the Task Learning setting leads to underperformance, attributing to their unnaturalness in pre-training stages. To verify whether our method enhances learning ability in a more general task learning environment, we conducted experiments by mapping the label space to symbols. In other words, we randomly replace the label space with one of [@, #, $, ...] at every seed5. We report the average F1-score across 27 datasets as experimental results. Figure 6 shows that our method outperforms other calibration methods, particularly demonstrating a significant improvement over the original inference. This indicates that our method enhances LLM’s task learning ability in a broader context.
<div style="text-align: center;">6.5 Analysis of λ across Different Task Categories</div>
# 6.5 Analysis of λ across Different Task Categories
Task
Orig.
ICC (λ = 1)
ICC (λ = 0.5)
Sentiment
0.69
0.75
0.78
NLI
0.33
0.51
0.49
Detection
0.41
0.52
0.54
Table 3: GPT-J’s averaged F1 scores across different task categories with and without applying R(). Orig. denotes original inference, while ICC stands for InContext Calibration.
We investigated the utility of R() across the different task categories, by calculating the task-wise average F1 score as shown in Table 3. For the Sentiment and Detection task, where the semantics of each word are crucial in associating a spe5Please refer to the Appendix D for the detailed implementation.
cific label in pre-trained knowledge, applying R() for calibration proves more effective. In contrast, for NLI tasks, where the LLMs must discern the logical relationship between two input sentences, In-Context Calibration demonstrates its effectiveness over the baseline in mitigating the reliance of LLMs on semantic priors of the demonstration (as show in Figure 2). However, applying R() in NLI tasks disrupts the order of the sentences, necessitating contextual awareness calibrating for better performance. We reserved for additional results in Appendix F.
# 7 Conclusion
In this paper, we introduced the term ‘Demonstration Shortcut’, which refers to the reliance of LLMs on their pre-trained semantic priors of demonstrations for making ICL predictions. To rectify this Demonstration Shortcut and enable the model to learn the input-label relationships from the demonstrations, we proposed a novel method, In-Context Calibration, based on the provided demonstrations. This demonstration-aware calibration consistently yields the improved performance, regardless of model sizes or types, across various settings. With the introduction of In-Context Calibration, we anticipate more reliable applications of Large Language Models.
# Limitations
In this work, we investigate the reliance of Large Language Models on their pre-trained semantic priors on demonstrations in in-context learning prediction. While In-Context Calibration demonstrates its effectiveness across various tasks and enhances the LLMs’ task learning abilities, our experiments primarily focus on classification tasks. However, the effect of the Demonstration Shortcut might manifest differently in generation tasks. Further analysis and adaptation of our In-Context Calibration method for these tasks are left for future research. Due to budget constraints, experiments with larger models (e.g., GPT4 API) and in multilingual settings were not feasible. Future studies with diverse settings and sufficient resources could provide a more comprehensive understanding. Due to computational constraints, it was impractical to explore every possible λ value for each model. While there may be variations across different models that remain unexplored, we believe that the comprehensive analysis provided in the appendix will offer practical guidelines for selecting the λ value.
# Ethical Considerations
Our work focuses on how Large Language Models utilize demonstrations in in-context learning. To enhance the ability of LLMs to learn input-label relationships from demonstrations, we conducted several additional inferences, requiring only minimal computational resources compared to updating model parameters. Additionally, we used only open-source LLMs and publicly available text classification datasets. Therefore, we do not concern about significant ethical issues arising from our work. On the contrary, we anticipate that future works could utilize our analysis to rectify harmful biases inherent in pre-trained model knowledge through demonstration-based methods.
# Acknowledgements
We appreciate Keonwoo Kim, Jaehee Kim, Yukyung Lee, and Hyowon Cho for their invaluable comments. We also thank POSTECH DI LAB members and anonymous reviewers for their comments on the paper.
Valerio Basile, Cristina Bosco, Elisabetta Fersini, Debora Nozza, Viviana Patti, Francisco Manuel Rangel Pardo, Paolo Rosso, and Manuela Sanguinetti. 2019. SemEval-2019 task 5: Multilingual detection of hate speech against immigrants and women in Twitter. In Proceedings of the 13th International Workshop on Semantic Evaluation, pages 54–63, Minneapolis, Minnesota, USA. Association for Computational Linguistics. Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. 2022. Gpt-neox-20b: An open-source autoregressive language model. arXiv preprint arXiv:2204.06745. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901. Ido Dagan, Oren Glickman, and Bernardo Magnini. 2005. The pascal recognising textual entailment challenge. In Machine learning challenges workshop, pages 177–190. Springer. Ona de Gibert, Naiara Perez, Aitor García-Pablos, and Montse Cuadros. 2018. Hate Speech Dataset from a White Supremacy Forum. In Proceedings of the 2nd Workshop on Abusive Language Online (ALW2), pages 11–20, Brussels, Belgium. Association for Computational Linguistics. Marie-Catherine De Marneffe, Mandy Simons, and Judith Tonhauser. 2019. The commitmentbank: Investigating projection in naturally occurring discourse. In proceedings of Sinn und Bedeutung, volume 23, pages 107–124. Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2022. A survey for in-context learning. arXiv preprint arXiv:2301.00234. Yu Fei, Yifan Hou, Zeming Chen, and Antoine Bosselut. 2023. Mitigating label biases for in-context learning. arXiv preprint arXiv:2305.19148. Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. 2020. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027. Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, and Luke Zettlemoyer. 2021. Surface form competition: Why the highest probability answer isn’t always right. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7038–7051, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
Eduard Hovy, Laurie Gerber, Ulf Hermjakob, ChinYew Lin, and Deepak Ravichandran. 2001. Toward semantics-based answer pinpointing. In Proceedings of the First International Conference on Human Language Technology Research. Minqing Hu and Bing Liu. 2004. Mining and summarizing customer reviews. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 168–177. Zhongtao Jiang, Yuanzhe Zhang, Cao Liu, Jun Zhao, and Kang Liu. 2023. Generative calibration for incontext learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 2312–2333. Jannik Kossen, Tom Rainforth, and Yarin Gal. 2023. In-context learning in large language models learns label relationships but is not conventional learning. arXiv preprint arXiv:2307.12375. Hector Levesque, Ernest Davis, and Leora Morgenstern. 2012. The winograd schema challenge. In Thirteenth international conference on the principles of knowledge representation and reasoning. P. Malo, A. Sinha, P. Korhonen, J. Wallenius, and P. Takala. 2014. Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65. Marco Marelli, Stefano Menini, Marco Baroni, Luisa Bentivogli, Raffaella Bernardi, and Roberto Zamparelli. 2014. A SICK cure for the evaluation of compositional distributional semantic models. In Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), pages 216–223, Reykjavik, Iceland. European Language Resources Association (ELRA). Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. Saif Mohammad, Svetlana Kiritchenko, Parinaz Sobhani, Xiaodan Zhu, and Colin Cherry. 2016. Semeval2016 task 6: Detecting stance in tweets. In Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016), pages 31–41. Ioannis Mollas, Zoe Chrysopoulou, Stamatis Karlos, and Grigorios Tsoumakas. 2020. Ethos: an online hate speech detection dataset. arXiv preprint arXiv:2006.08328. Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. 2020. Adversarial nli: A new benchmark for natural language understanding. In Proceedings of the 58th Annual Meeting
of the Association for Computational Linguistics. Association for Computational Linguistics.
Jane Pan, Tianyu Gao, Howard Chen, and Danqi Chen. 2023. What in-context learning “learns” in-context: Disentangling task recognition and task learning. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8298–8319, Toronto, Canada. Association for Computational Linguistics.
Bo Pang and Lillian Lee. 2004. A sentimental education: sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the 42nd Annual Meeting on Association for Computational Linguistics, pages 271–es.
L PaNgB. 2005. Exploitingclassrelationshipsforsentimentcate gorizationwithrespectratingsales. IN: ProceedingsofACL r05.
Laria Reynolds and Kyle McDonell. 2021. Prompt programming for large language models: Beyond the few-shot paradigm. In Extended Abstracts of the 2021 CHI Conference on Human Factors in Computing Systems, pages 1–7.
Maarten Sap, Saadia Gabriel, Lianhui Qin, Dan Jurafsky, Noah A. Smith, and Yejin Choi. 2020. Social bias frames: Reasoning about social and power implications of language. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5477–5490, Online. Association for Computational Linguistics.
Emily Sheng and David Uthus. 2020. Investigating societal biases in a poetry composition system. In Proceedings of the Second Workshop on Gender Bias in Natural Language Processing, pages 93–106, Barcelona, Spain (Online). Association for Computational Linguistics.
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA. Association for Computational Linguistics.
Ruixiang Tang, Dehan Kong, Lo li Huang, and Hui Xue. 2023. Large language models can be lazy learners: Analyze shortcuts in in-context learning. In Annual Meeting of the Association for Computational Linguistics.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.
Cynthia Van Hee, Els Lefever, and Véronique Hoste. 2018. Semeval-2018 task 3: Irony detection in english tweets. In Proceedings of The 12th International Workshop on Semantic Evaluation, pages 39– 50. Ben Wang. 2021. Mesh-Transformer-JAX: ModelParallel Implementation of Transformer Language Model with JAX. https://github.com/ kingoflolz/mesh-transformer-jax. Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, et al. 2023. Larger language models do in-context learning differently, 2023. URL https://arxiv. org/abs/2303.03846. Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Huggingface’s transformers: State-of-the-art natural language processing. Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2021. An explanation of in-context learning as implicit bayesian inference. In International Conference on Learning Representations. Marcos Zampieri, Shervin Malmasi, Preslav Nakov, Sara Rosenthal, Noura Farra, and Ritesh Kumar. 2019. Semeval-2019 task 6: Identifying and categorizing offensive language in social media (offenseval). In Proceedings of the 13th International Workshop on Semantic Evaluation, pages 75–86. Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068. Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In NIPS. Yufeng Zhang, Fengzhuo Zhang, Zhuoran Yang, and Zhaoran Wang. 2023. What and how does incontext learning learn? bayesian model averaging, parameterization, and generalization. arXiv preprint arXiv:2305.19420. Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 12697–12706. PMLR. Han Zhou, Xingchen Wan, Lev Proleev, Diana Mincu, Jilin Chen, Katherine Heller, and Subhrajit Roy. 2023. Batch calibration: Rethinking calibration for
Han Zhou, Xingchen Wan, Lev Proleev, Diana Mincu, Jilin Chen, Katherine Heller, and Subhrajit Roy. 2023. Batch calibration: Rethinking calibration for
Han Zhou, Xingchen Wan, Lev Proleev, Diana Mincu, Jilin Chen, Katherine Heller, and Subhrajit Roy. 2023. Batch calibration: Rethinking calibration for
in-context learning and prompt engineering. arXiv preprint arXiv:2309.17249.
# A Full Description of the Articles
A full description of the articles in Figure 1 is provided in Figure 10. We used the AGNews dataset (Zhang et al., 2015) for this experiment. In a zeroshot setting, the LLM (GPT-J) predicts the test article the world label, although the ground-truth label is technology. We then sampled two demonstration sets from the same training set, each having a uniform label distribution and the same label order. The first demonstration set predominantly features business-related semantics across all examples, while the second set leans more toward sports-related semantics. Despite both demonstration sets having the same uniform label distribution and identical order, the LLM fails to learn the inputlabel relationships, instead relying on semantic priors of demonstration to make ICL predictions.
# B Datasets
We use 27 text classification datasets for our experiments, most of which are widely used in existing ICL works (Min et al., 2022; Fei et al., 2023; Pan et al., 2023). Sentiment task datasets include AGNews (Zhang et al., 2015), CR (Hu and Liu, 2004), financial_phrasebank (Malo et al., 2014), poem_sentiment (Sheng and Uthus, 2020), MR (PaNgB, 2005), sst2 (Socher et al., 2013), Subj (Pang and Lee, 2004), and TREC (Hovy et al., 2001); Natural Language Inference task datasets include ANLI (Nie et al., 2020), WNLI (Levesque et al., 2012), RTE (Dagan et al., 2005), CB (De Marneffe et al., 2019), and SICK (Marelli et al., 2014) For the Detection task we use social_bias_frames (Sap et al., 2020), tweet_eval_stance_athesim, tweet_eval_stance_feminist (Mohammad et al., 2016), tweet_eval_hate (Basile et al., 2019), tweet_eval_irony (Van Hee et al., 2018), tweet_eval_offensive (Zampieri et al., 2019), hate_speech18 (de Gibert et al., 2018), ethos_binary, ethos_disability, ethos_gender, ethos_national_origin, ethos_race, ethos_religion, and ethos_violence (Mollas et al., 2020). We constructed demonstrations by sampling from the training set and using the validation set for evaluation. In cases where a validation set does not exist, we utilized the test set. For evaluation, we sampled either a maximum of 500 examples or the entire dataset size, whichever is larger.
# C Prompt Templates
Our natural language prompts are largely based on Zhao et al. (2021) and Fei et al. (2023). We have adjusted the templates as needed to better align with each dataset’s specific intent. The complete list of prompts is provided in Table 7.
# D Implementation Details
We set M = 20 for the Domain Calibration implementation, following the original settings used by Fei et al. (2023). We mainly followed the Pan et al. (2023)’s setting for symbol token selection and incorporated additional symbol tokens to enhance generalization. Consequently, in each seed, every label was randomly mapped to one of the following symbols: [“@”, “#”, “$”, “%”, “ ∗”, “ ∧ ”, “##”, “$$”, “%%”, “ ∗∗”].
# E Random Shuffling Function
In our main experiment, we applied the random shuffling function only once to each xi to calculate PR(i). To validate whether a single shuffling process introduces randomness, we conducted additional experiments, varying the number of shuffles to 1, 5, and 10. We present supplementary ablation studies using GPT-J with default settings in our main experiments.
Method
Original ICL
TL
Original Inference
0.48
0.40
Context Calibration
0.47
0.34
Domain Calibration (w/ Test)
0.55
0.43
ICC (1)
0.60
0.49
ICC (5)
0.60
0.48
ICC (10)
0.60
0.49
Table 4: ICC stands for In-Context Calibration, and the number in (N) means shuffling number. We calculated the expectation of the randomly shuffled term when the shuffling number exceeded 1.
The results illustrate that a single shuffle does not incur significant randomness in the estimations. These observations are similar to those in Fei et al. (2023), which used different random seeds to demonstrate the rapid stabilization and convergence of the sampling process.
# F Comprehensive analysis for λ Values
We present the complete results for various λ values of GPT-J in Table 5. GPT-J performs best with an
Method
Original
TL
Original Inference
0.48
0.40
Context Calibration
0.47
0.34
Domain Calibration (w/ Test)
0.55
0.43
ICC (λ = 0)
0.56
0.46
ICC (λ = 0.25)
0.56
0.45
ICC (λ = 0.5)
0.60
0.49
ICC (λ = 0.75)
0.57
0.47
ICC (λ = 1)
0.58
0.50
Table 5: Average Macro F1 scores across 27 datasets for both Original ICL Task (Original) and Task Learning (TL) with different λ values using GPT-J. ICC stands for In-Context Calibration.
Original ICL Task at 0.5 and the higher λ values lead to higher task learning ability.
Original
TL
OPT-6.7B
Llama2-7B
OPT-6.7B
Llama2-7B
Original Inference
0.48
0.48
0.38
0.45
Context Calibration
0.48
0.48
0.35
0.43
Domain Calibration (w/ Test)
0.52
0.55
0.43
0.49
ICC (λ = 0)
0.55
0.56
0.46
0.51
ICC (λ = 0.25)
0.55
0.58
0.47
0.51
ICC (λ = 0.5)
0.57
0.61
0.49
0.52
ICC (λ = 0.75)
0.56
0.60
0.47
0.51
ICC (λ = 1.0)
0.57
0.59
0.49
0.53
Table 6: Average Macro F1 scores across 27 datasets for both Original ICL Task (Original) and Task Learning (TL) with different λ values for OPT-6.7B and Llama27B. ICC stands for In Context Calibration.
To further analyze the comprehensive impact of λ, we conducted supplementary studies using OPT6.7B and Llama2-7B, which have similar sizes to GPT-J, exploring broader λ values (Table 6). As with GPT-J’s λ value, OPT-6.7B and Llama2-7B also perform best with an Original ICL Task at 0.5 (OPT-6.7B also performs best at the value of 1.0). In line with our primary findings (Table 3), a higher λ value enhances task learning capabilities, leading to less grammatical information corruption in the original input sentences. We also analyze the task-wise value as in Table 8. Consistent across all models, a higher λ improves performance in the NLI task, corroborating our initial findings. Similar to the results seen in Fei et al. (2023), the semantics of words play a critical role in Sentiment and Detection tasks (where a lower λ still shows comparative performance, despite the lost of grammatical information), as the language model’s dependence on the label’s semantics is significant in these tasks. Therefore, (1) we recommend starting with a λ value of 0.5 in In-Context Calibration and adjust-
ing it based on the task-wise experimental results obtained from the validation set in the Original ICL Task and (2) search for a value with 0.5 or higher in the Task Learning setting. However, due to computational constraints, performing a grid search across all models for every λ value was impractical, so we opted for λ of 0.5 in the main experiment for all models for efficiency.
# G Detailed Results
We demonstrate the results of the OPT, GPT, Llama2, and Llama2-Chat models in both the Original ICL Task and the Task Learning setting across Table 9 to 16. For models above 50B scale, please refer to Table 17 and 18. We average the performance across a total of 5 seeds, recording mean values and standard deviations. ‘Orig.’ denotes original inference, ‘CC’ refers to Context Calibration, ‘DC’ stands for Domain Calibration, and ‘ICC’ represents In-Context Calibration. Furthermore, our results indicate that In-Context Calibration outperforms the baselines in most tasks.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/be58/be58f891-98dd-455c-a349-a26663401b98.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Averaged Macro F1 scores for the OPT (Top Graph) and Llama2 (Bottom graph) model are presented across 27 classification tasks, each featuring a permuted label space. The x-axis represents the model size.</div>
Figure 7 presents experimental results for OPT and Llama2 under the same setting as figure 5. We demonstrate that our method is effective across various model types and sizes. Specifically, OPT shows a 19% performance increase compared to the original inference, while Llama2 exhibits a 15%
improvement. In Figure 8, we present the performance of LLMs when labels are mapped to symbols. InContext Calibration significantly improves performance across different model types and sizes, highlighting the consistent enhancement of task learning ability.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9730/97309bd2-cf79-47aa-859d-7c986bb9c211.png" style="width: 50%;"></div>
Figure 8: Averaged Macro F1 scores for the 13-20B scale model families are presented across 27 datasets with each label space replaced by symbol tokens. The x-axis represents the model type.
<div style="text-align: center;">Figure 8: Averaged Macro F1 scores for the 13-20B scale model families are presented across 27 datasets with each label space replaced by symbol tokens. The x-axis represents the model type.</div>
# H Adding More In-Context Examples (K=8/12/16)
H Adding More In-Context Examples (K=8/12/16)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f3d5/f3d56a17-5a91-48ff-a740-4a479c1b5770.png" style="width: 50%;"></div>
Figure 9: The top graph depicts the average Macro F1 score for the Original ICL Task across 27 datasets for GPT-J. The bottom graph plots the average Macro F1 score for the Task Learning setting. In both graphs, the x-axis represents the number of demonstrations.
<div style="text-align: center;">Figure 9: The top graph depicts the average Macro F1 score for the Original ICL Task across 27 datasets for GPT-J. The bottom graph plots the average Macro F1 score for the Task Learning setting. In both graphs, the x-axis represents the number of demonstrations.</div>
We study the effect of adding more in-context
examples by evaluating the GPT-J. As shown in Figure 9, In-Context Calibration achieves approximately a 6% higher performance than other calibration methods in the Original ICL Task. Similar to Pan et al. (2023), Task Learning performance further improves as the number of demonstrations increases. In-Context Calibration consistently outperforms the baselines, whereas Context Calibration lags behind the original inference. In conclusion, In-Context Calibration consistently demonstrates superior performance in both the Original ICL Task and Task Learning settings, irrespective of the number of demonstrations.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/054f/054ffa67-9192-4475-83a5-09be5b63317d.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/aee1/aee1e933-349d-4331-af5b-6b15492210d9.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10: Full description of the Demonstration Shortcut illustrated in Figure 1. All articles and labels for the demonstrations and the test article were selected from the AGNews (Zhang et al., 2015) dataset. GPT-J was used in</div>
Figure 10: Full description of the Demonstration Shortcut illustrated in Figure 1. All articles and labels for t demonstrations and the test article were selected from the AGNews (Zhang et al., 2015) dataset. GPT-J was used this experiment.
<div style="text-align: center;">0 0.1 0.2 0.3 0.4 0.5 0.6</div>
<div style="text-align: center;">Natural Language Template</div>
Dataset
Natural Language Template
Label Space
Sentiment Task
sst2, MR, CR
Review: [INPUT]
positive, negative
Sentiment: [LABEL]
financial_phrasebank
Sentence: [INPUT]
positive, negative,
Sentiment: [LABEL]
neutral
poem_sentiment
Verse text: [INPUT]
positive, negative,
Sentiment: [LABEL]
neutral
Subj
Input: [INPUT]
objective
Label: [LABEL]
subjective/personal
AG News
Classify the news articles into the categories of Label Space.
world, technology
Article: [INPUT]
sports, business
Answer: [LABEL]
TREC
Classify the questions based on whether their answer type is a Label Space.
number, location,
Question: [INPUT]
description, entity,
Answer: [LABEL]
abbre, person
NLI Task
ANLI, SICK
[PREMISE] question: [HYPOTHESIS]
true, neutral, false
true, neutral, or false? answer: [LABEL]
RTE, WNLI
[PREMISE] question: [HYPOTHESIS] True or False?
True, False
answer: [LABEL]
CB
[PREMISE] question: [HYPOTHESIS] true, false, or neither?
true, false, neither
answer: [LABEL]
Detection Task
social_bias_frames
Post: [INPUT]
neutral
Label: [LABEL]
offensive/hate
tweet_eval_hate
Tweet: [INPUT]
neutral, hate
Label: [LABEL]
tweet_eval_irony
Tweet: [INPUT]
neutral
Label: [LABEL]
ironic/contradict
tweet_eval_offensive
Tweet: [INPUT]
neutral
Label: [LABEL]
offensive/hate
tweet_eval_stance_athesim,
Tweet: [INPUT] Label: [LABEL]
none, against, favor
tweet_eval_stance_feminist
ethos_binary,
Text: [INPUT]
neutral, hate
ethos_disability,
Label: [LABEL]
ethos-religion,
ethos_national_origin,
ethos_race,
ethos_religion,
ethos_violence,
hate_speech18
Table 7: Prompt templates used in our experiments are primarily adapted from the work of Zhao et al. (2021) and Fei et al. (2023). In the Task Learning setting, each label is randomly mapped to a string number corresponding to the size of the Label Space (i.e., from 0 to len(Label Space) - 1) for each seed.
neutral, hate
Sentiment
NLI
Detection
OPT-6.7B
Llama2-7B
OPT-6.7B
Llama2-7B
OPT-6.7B
Llama2-7B
Original Inference
0.674
0.720
0.342
0.478
0.457
0.371
Contextual Calibration
0.685
0.735
0.326
0.515
0.465
0.423
Domain Calibration w/ Test
0.734
0.752
0.316
0.496
0.524
0.523
ICC (λ = 0)
0.764
0.774
0.338
0.454
0.543
0.523
ICC (λ = 0.25)
0.764
0.796
0.337
0.511
0.522
0.555
ICC (λ = 0.5)
0.749
0.792
0.398
0.551
0.552
0.558
ICC (λ = 0.75)
0.751
0.782
0.428
0.557
0.532
0.564
ICC (λ = 1.0)
0.749
0.775
0.445
0.547
0.530
0.532
<div style="text-align: center;">Sentiment</div>
Table 8: OPT-6.7B and Llama2-7B’s averaged F1 scores across different task categories using various λ value. ICC stands for In Context Calibration.
aged F1 scores across different task categories using various λ value. IC
Dataset
Size
Method
Sentiment Task
Orig.
CC
DC w/Demo
DC w/Test
ICC
AGNews
2.7B
0.298 (0.072)
0.669 (0.117)
0.676 (0.676)
0.803 (0.018)
0.811 (0.024)
6.7B
0.582 (0.104)
0.716 (0.108)
0.677 (0.677)
0.765 (0.061)
0.811 (0.023)
13B
0.722 (0.070)
0.717 (0.104)
0.735 (0.735)
0.796 (0.045)
0.828 (0.051)
CR
2.7B
0.888 (0.031)
0.892 (0.021)
0.911 (0.911)
0.915 (0.008)
0.908 (0.018)
6.7B
0.860 (0.042)
0.883 (0.024)
0.911 (0.911)
0.921 (0.007)
0.921 (0.005)
13B
0.860 (0.082)
0.912 (0.013)
0.921 (0.921)
0.920 (0.013)
0.926 (0.006)
financial_phrasebank
2.7B
0.623 (0.191)
0.459 (0.062)
0.607 (0.607)
0.719 (0.115)
0.701 (0.088)
6.7B
0.693 (0.075)
0.526 (0.106)
0.522 (0.522)
0.441 (0.041)
0.723 (0.069)
13B
0.471 (0.116)
0.481 (0.085)
0.637 (0.637)
0.676 (0.169)
0.690 (0.098)
poem_sentiment
2.7B
0.410 (0.077)
0.367 (0.112)
0.414 (0.414)
0.542 (0.090)
0.545 (0.083)
6.7B
0.516 (0.136)
0.423 (0.028)
0.500 (0.500)
0.511 (0.069)
0.618 (0.004)
13B
0.231 (0.067)
0.485 (0.097)
0.414 (0.414)
0.472 (0.044)
0.492 (0.061)
MR
2.7B
0.875 (0.029)
0.878 (0.034)
0.804 (0.804)
0.900 (0.018)
0.900 (0.022)
6.7B
0.895 (0.017)
0.908 (0.013)
0.913 (0.913)
0.917 (0.002)
0.918 (0.004)
13B
0.893 (0.039)
0.921 (0.011)
0.926 (0.926)
0.924 (0.008)
0.925 (0.008)
sst2
2.7B
0.846 (0.130)
0.919 (0.048)
0.910 (0.910)
0.917 (0.034)
0.923 (0.035)
6.7B
0.934 (0.030)
0.952 (0.007)
0.947 (0.947)
0.953 (0.010)
0.954 (0.009)
13B
0.900 (0.084)
0.941 (0.011)
0.943 (0.943)
0.945 (0.017)
0.945 (0.022)
Subj
2.7B
0.360 (0.034)
0.546 (0.150)
0.489 (0.489)
0.566 (0.050)
0.614 (0.067)
6.7B
0.581 (0.182)
0.628 (0.161)
0.642 (0.642)
0.714 (0.061)
0.702 (0.130)
13B
0.794 (0.079)
0.772 (0.116)
0.760 (0.760)
0.726 (0.191)
0.852 (0.046)
TREC
2.7B
0.232 (0.027)
0.246 (0.035)
0.339 (0.339)
0.382 (0.033)
0.385 (0.039)
6.7B
0.356 (0.049)
0.287 (0.044)
0.308 (0.308)
0.360 (0.026)
0.361 (0.020)
13B
0.382 (0.112)
0.342 (0.041)
0.342 (0.342)
0.342 (0.041)
0.439 (0.051)
NLI Task
Orig.
CC
DC w/Demo
DC w/Test
ICC
ANLI
2.7B
0.224 (0.059)
0.269 (0.043)
0.229 (0.229)
0.222 (0.047)
0.294 (0.024)
6.7B
0.215 (0.055)
0.263 (0.053)
0.212 (0.212)
0.258 (0.035)
0.316 (0.020)
13B
0.216 (0.044)
0.268 (0.029)
0.243 (0.243)
0.281 (0.037)
0.307 (0.016)
WNLI
2.7B
0.317 (0.027)
0.302 (0.003)
0.330 (0.330)
0.350 (0.039)
0.410 (0.012)
6.7B
0.333 (0.056)
0.337 (0.033)
0.380 (0.380)
0.330 (0.046)
0.461 (0.053)
13B
0.360 (0.012)
0.374 (0.021)
0.398 (0.398)
0.378 (0.033)
0.414 (0.049)
RTE
2.7B
0.489 (0.044)
0.368 (0.034)
0.372 (0.372)
0.515 (0.020)
0.529 (0.035)
6.7B
0.530 (0.035)
0.465 (0.052)
0.355 (0.355)
0.514 (0.057)
0.540 (0.024)
13B
0.376 (0.064)
0.499 (0.076)
0.455 (0.455)
0.434 (0.095)
0.547 (0.037)
CB
2.7B
0.358 (0.076)
0.333 (0.133)
0.242 (0.242)
0.269 (0.014)
0.392 (0.065)
6.7B
0.291 (0.047)
0.240 (0.029)
0.113 (0.113)
0.163 (0.057)
0.267 (0.064)
13B
0.280 (0.076)
0.217 (0.076)
0.218 (0.218)
0.226 (0.008)
0.317 (0.067)
SICK
2.7B
0.285 (0.063)
0.270 (0.024)
0.317 (0.317)
0.299 (0.067)
0.328 (0.072)
6.7B
0.267 (0.088)
0.361 (0.070)
0.393 (0.393)
0.359 (0.065)
0.407 (0.073)
13B
0.225 (0.066)
0.225 (0.066)
0.225 (0.225)
0.352 (0.171)
0.443 (0.107)
Detection Task
Orig.
CC
DC w/Demo
DC w/Test
ICC
social_bias_frames
2.7B
0.466 (0.075)
0.644 (0.078)
0.615 (0.615)
0.665 (0.025)
0.667 (0.036)
6.7B
0.610 (0.169)
0.651 (0.077)
0.665 (0.665)
0.674 (0.034)
0.685 (0.062)
13B
0.556 (0.099)
0.687 (0.066)
0.600 (0.600)
0.639 (0.089)
0.711 (0.016)
tweet_eval_stance_athesim
2.7B
0.089 (0.007)
0.086 (0.002)
0.107 (0.107)
0.109 (0.014)
0.138 (0.015)
6.7B
0.279 (0.094)
0.150 (0.063)
0.151 (0.151)
0.162 (0.044)
0.291 (0.048)
13B
0.178 (0.044)
0.147 (0.063)
0.181 (0.181)
0.191 (0.055)
0.227 (0.044)
tweet_eval_stance_feminist
2.7B
0.175 (0.115)
0.185 (0.096)
0.273 (0.273)
0.298 (0.063)
0.347 (0.043)
6.7B
0.398 (0.051)
0.306 (0.065)
0.267 (0.267)
0.312 (0.075)
0.395 (0.080)
13B
0.135 (0.023)
0.247 (0.145)
0.325 (0.325)
0.337 (0.102)
0.390 (0.048)
tweet_eval_hate
2.7B
0.366 (0.062)
0.513 (0.084)
0.548 (0.548)
0.545 (0.029)
0.530 (0.052)
6.7B
0.344 (0.039)
0.379 (0.089)
0.550 (0.550)
0.599 (0.034)
0.559 (0.072)
13B
0.548 (0.114)
0.599 (0.047)
0.630 (0.630)
0.633 (0.036)
0.623 (0.044)
tweet_eval_irony
2.7B
0.387 (0.059)
0.517 (0.155)
0.601 (0.601)
0.653 (0.070)
0.647 (0.079)
6.7B
0.451 (0.113)
0.581 (0.101)
0.597 (0.597)
0.625 (0.083)
0.650 (0.085)
13B
0.434 (0.128)
0.394 (0.084)
0.622 (0.622)
0.602 (0.049)
0.602 (0.058)
tweet_eval_offensive
2.7B
0.538 (0.059)
0.603 (0.045)
0.588 (0.588)
0.613 (0.031)
0.621 (0.046)
6.7B
0.472 (0.072)
0.602 (0.021)
0.626 (0.626)
0.627 (0.019)
0.632 (0.020)
13B
0.585 (0.063)
0.605 (0.054)
0.604 (0.604)
0.655 (0.010)
0.643 (0.022)
ethos_binary
2.7B
0.490 (0.151)
0.581 (0.051)
0.625 (0.625)
0.681 (0.029)
0.674 (0.025)
6.7B
0.619 (0.135)
0.740 (0.049)
0.656 (0.656)
0.754 (0.036)
0.762 (0.026)
13B
0.674 (0.052)
0.658 (0.035)
0.705 (0.705)
0.722 (0.032)
0.719 (0.031)
ethos_disability
2.7B
0.402 (0.134)
0.442 (0.044)
0.232 (0.232)
0.334 (0.071)
0.421 (0.048)
6.7B
0.418 (0.116)
0.367 (0.125)
0.261 (0.261)
0.421 (0.064)
0.460 (0.053)
13B
0.446 (0.097)
0.383 (0.115)
0.327 (0.327)
0.408 (0.050)
0.457 (0.103)
ethos_gender
2.7B
0.368 (0.148)
0.412 (0.108)
0.350 (0.350)
0.380 (0.042)
0.406 (0.057)
6.7B
0.507 (0.071)
0.329 (0.092)
0.400 (0.400)
0.440 (0.070)
0.483 (0.046)
13B
0.397 (0.058)
0.376 (0.088)
0.381 (0.381)
0.449 (0.022)
0.453 (0.029)
ethos_national_origin
2.7B
0.293 (0.154)
0.326 (0.098)
0.373 (0.373)
0.375 (0.046)
0.410 (0.074)
6.7B
0.376 (0.088)
0.350 (0.123)
0.411 (0.411)
0.412 (0.056)
0.429 (0.096)
13B
0.427 (0.067)
0.433 (0.041)
0.306 (0.306)
0.397 (0.068)
0.434 (0.061)
ethos_race
2.7B
0.394 (0.114)
0.475 (0.066)
0.428 (0.428)
0.406 (0.058)
0.579 (0.066)
6.7B
0.441 (0.042)
0.361 (0.079)
0.503 (0.503)
0.516 (0.077)
0.569 (0.043)
13B
0.331 (0.141)
0.374 (0.117)
0.479 (0.479)
0.482 (0.039)
0.490 (0.081)
ethos_religion
2.7B
0.372 (0.173)
0.411 (0.150)
0.390 (0.390)
0.419 (0.060)
0.420 (0.037)
6.7B
0.326 (0.131)
0.243 (0.104)
0.350 (0.350)
0.385 (0.053)
0.394 (0.090)
13B
0.361 (0.108)
0.361 (0.108)
0.355 (0.355)
0.395 (0.108)
0.399 (0.134)
ethos_violence
2.7B
0.310 (0.084)
0.381 (0.115)
0.373 (0.373)
0.426 (0.062)
0.423 (0.108)
6.7B
0.426 (0.100)
0.380 (0.104)
0.505 (0.505)
0.490 (0.060)
0.513 (0.026)
13B
0.379 (0.124)
0.437 (0.052)
0.431 (0.431)
0.475 (0.066)
0.492 (0.081)
hate_speech18
2.7B
0.454 (0.193)
0.396 (0.086)
0.413 (0.413)
0.460 (0.046)
0.455 (0.027)
6.7B
0.481 (0.022)
0.547 (0.028)
0.638 (0.638)
0.628 (0.040)
0.612 (0.050)
13B
0.488 (0.047)
0.585 (0.066)
0.571 (0.571)
0.590 (0.048)
0.582 (0.045)
Table 9: Macro F1-score across 27 datasets for OPT model family under the Original ICL Task.
Dataset
Size
Method
Sentiment Task
Orig.
CC
DC w/Demo
DC w/Test
ICC
AGNews
2.7B
0.635 (0.092)
0.724 (0.041)
0.596 (0.596)
0.771 (0.043)
0.727 (0.039)
J (6B)
0.755 (0.129)
0.787 (0.073)
0.726 (0.726)
0.755 (0.046)
0.867 (0.012)
20B
0.640 (0.066)
0.826 (0.031)
0.798 (0.798)
0.813 (0.014)
0.818 (0.030)
CR
2.7B
0.744 (0.106)
0.797 (0.095)
0.857 (0.857)
0.878 (0.032)
0.887 (0.019)
J (6B)
0.869 (0.052)
0.885 (0.026)
0.904 (0.904)
0.911 (0.002)
0.912 (0.004)
20B
0.895 (0.031)
0.919 (0.010)
0.914 (0.914)
0.918 (0.008)
0.919 (0.005)
financial_phrasebank
2.7B
0.561 (0.204)
0.481 (0.067)
0.673 (0.673)
0.688 (0.052)
0.696 (0.036)
J (6B)
0.593 (0.110)
0.418 (0.115)
0.540 (0.540)
0.617 (0.073)
0.681 (0.056)
20B
0.649 (0.194)
0.477 (0.062)
0.522 (0.522)
0.590 (0.077)
0.688 (0.075)
poem_sentiment
2.7B
0.410 (0.077)
0.367 (0.112)
0.414 (0.414)
0.542 (0.090)
0.545 (0.083)
J (6B)
0.512 (0.157)
0.323 (0.148)
0.359 (0.359)
0.524 (0.098)
0.639 (0.082)
20B
0.410 (0.146)
0.435 (0.107)
0.468 (0.468)
0.494 (0.057)
0.517 (0.064)
MR
2.7B
0.847 (0.067)
0.856 (0.027)
0.847 (0.847)
0.897 (0.005)
0.897 (0.003)
J (6B)
0.893 (0.028)
0.900 (0.007)
0.903 (0.903)
0.902 (0.013)
0.915 (0.006)
20B
0.875 (0.037)
0.894 (0.019)
0.917 (0.917)
0.915 (0.009)
0.917 (0.004)
sst2
2.7B
0.602 (0.269)
0.791 (0.101)
0.867 (0.867)
0.920 (0.017)
0.884 (0.029)
J (6B)
0.927 (0.044)
0.940 (0.014)
0.942 (0.942)
0.954 (0.012)
0.950 (0.007)
20B
0.944 (0.015)
0.956 (0.011)
0.958 (0.958)
0.956 (0.012)
0.960 (0.009)
Subj
2.7B
0.327 (0.005)
0.710 (0.177)
0.712 (0.712)
0.778 (0.051)
0.801 (0.042)
J (6B)
0.606 (0.091)
0.556 (0.143)
0.598 (0.598)
0.675 (0.116)
0.696 (0.100)
20B
0.578 (0.119)
0.578 (0.147)
0.721 (0.721)
0.709 (0.050)
0.740 (0.055)
TREC
2.7B
0.431 (0.057)
0.387 (0.037)
0.566 (0.566)
0.535 (0.077)
0.547 (0.075)
J (6B)
0.416 (0.078)
0.503 (0.066)
0.533 (0.533)
0.564 (0.030)
0.564 (0.013)
20B
0.481 (0.045)
0.615 (0.054)
0.572 (0.572)
0.620 (0.047)
0.608 (0.057)
NLI Task
Orig.
CC
DC w/Demo
DC w/Test
ICC
ANLI
2.7B
0.221 (0.035)
0.208 (0.052)
0.260 (0.260)
0.255 (0.061)
0.294 (0.050)
J (6B)
0.256 (0.112)
0.235 (0.028)
0.267 (0.267)
0.252 (0.044)
0.394 (0.109)
20B
0.237 (0.039)
0.229 (0.050)
0.198 (0.198)
0.184 (0.023)
0.296 (0.017)
WNLI
2.7B
0.334 (0.042)
0.325 (0.025)
0.373 (0.373)
0.400 (0.066)
0.417 (0.070)
J (6B)
0.304 (0.000)
0.394 (0.065)
0.414 (0.414)
0.444 (0.055)
0.497 (0.044)
20B
0.344 (0.032)
0.353 (0.085)
0.358 (0.358)
0.391 (0.040)
0.465 (0.079)
RTE
2.7B
0.385 (0.110)
0.392 (0.074)
0.407 (0.407)
0.384 (0.070)
0.492 (0.060)
J (6B)
0.495 (0.084)
0.492 (0.085)
0.553 (0.553)
0.489 (0.090)
0.580 (0.049)
20B
0.505 (0.050)
0.483 (0.034)
0.477 (0.477)
0.408 (0.055)
0.566 (0.021)
CB
2.7B
0.260 (0.059)
0.189 (0.116)
0.205 (0.205)
0.206 (0.032)
0.334 (0.107)
J (6B)
0.278 (0.071)
0.345 (0.118)
0.237 (0.237)
0.256 (0.079)
0.484 (0.095)
20B
0.312 (0.030)
0.188 (0.121)
0.291 (0.291)
0.229 (0.159)
0.353 (0.076)
SICK
2.7B
0.262 (0.120)
0.313 (0.062)
