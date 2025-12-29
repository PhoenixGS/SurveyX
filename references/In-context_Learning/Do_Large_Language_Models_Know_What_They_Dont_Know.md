# rge Language Models Know What They Don’t
Zhangyue Yin♢ Qiushi Sun♠ Qipeng Guo♢ Jiawen Wu♢ Xipeng Qiu♢∗ Xuanjing Huang♢ ♢School of Computer Science, Fudan University ♠Department of Mathematics, National University of Singapore {yinzy21,jwwu21}@m.fudan.edu.cn qiushisun@u.nus.edu {qpguo16,xpqiu,xjhuang}@fudan.edu.cn
# Abstract
Large language models (LLMs) have a wealth of knowledge that allows them to excel in various Natural Language Processing (NLP) tasks. Current research focuses on enhancing their performance within their existing knowledge. Despite their vast knowledge, LLMs are still limited by the amount of information they can accommodate and comprehend. Therefore, the ability to understand their own limitations on the unknows, referred to as self-knowledge, is of paramount importance. This study aims to evaluate LLMs’ self-knowledge by assessing their ability to identify unanswerable or unknowable questions. We introduce an automated methodology to detect uncertainty in the responses of these models, providing a novel measure of their self-knowledge. We further introduce a unique dataset, SelfAware, consisting of unanswerable questions from five diverse categories and their answerable counterparts. Our extensive analysis, involving 20 LLMs including GPT-3, InstructGPT, and LLaMA, discovering an intrinsic capacity for self-knowledge within these models. Moreover, we demonstrate that in-context learning and instruction tuning can further enhance this self-knowledge. Despite this promising insight, our findings also highlight a considerable gap between the capabilities of these models and human proficiency in recognizing the limits of their knowledge.
“True wisdom is knowing what you don’t know.” –Confucius
# 1 Introduction
Recently, Large Language Models (LLMs) such as GPT-4 (OpenAI, 2023), PaLM 2 (Anil et al., 2023), and LLaMA (Touvron et al., 2023) have shown exceptional performance on a wide range of NLP tasks, including common sense reasoning (Wei et al., 2022; Zhou et al., 2022) and mathe-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3139/313966cd-4a1f-4c39-afc8-d2c54ea1befd.png" style="width: 50%;"></div>
Figure 1: Know-Unknow Quadrant. The horizontal axis represents the model’s memory capacity for knowledge, and the vertical axis represents the model’s ability to comprehend and utilize knowledge.
matical problem-solving (Lewkowycz et al., 2022; Chen et al., 2022). Despite their ability to learn from huge amounts of data, LLMs still have limitations in their capacity to retain and understand information. To ensure responsible usage, it is crucial for LLMs to have the capability of recognizing their limitations and conveying uncertainty when responding to unanswerable or unknowable questions. This acknowledgment of limitations, also known as “knowing what you don’t know,” is a crucial aspect in determining their practical applicability. In this work, we refer to this ability as model self-knowledge. The Know-Unknow quadrant in Figure 1 illustrates the relationship between the model’s knowledge and comprehension. The ratio of “Known Knows” to “Unknown Knows” demonstrates the model’s proficiency in understanding and applying existing knowledge. Techniques such as Chain-of-Thought (Wei et al., 2022), SelfConsistency (Wang et al., 2022), and Complex CoT (Fu et al., 2022) can be utilized to increase
this ratio, resulting in improved performance on NLP tasks. We focus on the ratio of “Known Unknows” to “Unknown Unknows”, which indicates the model’s self-knowledge level, specifically understanding its own limitations and deficiencies in the unknows. Existing datasets such as SQuAD2.0 (Rajpurkar et al., 2018) and NewsQA (Trischler et al., 2017), widely used in question answering (QA), have been utilized to test the self-knowledge of models with unanswerable questions. However, these questions are context-specific and could become answerable when supplemented with additional information. Srivastava et al. (2022) attempted to address this by evaluating LLMs’ competence in delineating their knowledge boundaries, employing a set of 23 pairs of answerable and unanswerable multiple-choice questions. They discovered that these models’ performance barely surpassed that of random guessing. Kadavath et al. (2022) suggested probing the selfknowledge of LLMs through the implementation of a distinct "Value Head". Yet, this approach may encounter difficulties when applied across varied domains or tasks due to task-specific training. Consequently, we redirect our focus to the inherent abilities of LLMs, and pose the pivotal question: “Do large language models know what they don’t know?”. In this study, we investigate the self-knowledge of LLMs using a novel approach. By gathering reference sentences with uncertain meanings, we can determine whether the model’s responses reflect uncertainty using a text similarity algorithm. We quantified the model’s self-knowledge using the F1 score. To address the small and idiosyncratic limitations of existing datasets, we created a new dataset called SelfAware. This dataset comprises 1,032 unanswerable questions, which are distributed across five distinct categories, along with an additional 2,337 questions that are classified as answerable. Experimental results on GPT-3, InstructGPT, LLaMA, and other LLMs demonstrate that in-context learning and instruction tuning can effectively enhance the self-knowledge of LLMs. However, the self-knowledge exhibited by the current state-of-the-art model, GPT-4, measures at 75.47%, signifying a notable disparity when contrasted with human self-knowledge, which is rated at 84.93%. Our key contributions to this field are summarized as follows:
• Through our detailed analysis of 20 LLMs, benchmarked against human self-knowledge, we identified a significant disparity between the most advanced LLMs and humans 1.
# 2 Dataset Construction
To conduct a more comprehensive evaluation of the model’s self-knowledge, we constructed a dataset that includes a larger number and more diverse types of unanswerable questions than KnowUnknowns dataset (Srivastava et al., 2022). To facilitate this, we collected a corpus of 2,858 unanswerable questions, sourced from online platforms like Quora and HowStuffWorks. These questions were meticulously evaluated by three seasoned annotation analysts, each operating independently. The analysts were permitted to leverage external resources, such as search engines. To ensure the validity of our dataset, we retained only the questions that all three analysts concurred were unanswerable. This rigorous process yielded a finalized collection of 1,032 unanswerable questions. In pursuit of a comprehensive evaluation, we opted for answerable questions drawn from three datasets: SQuAD (Rajpurkar et al., 2016), HotpotQA (Yang et al., 2018), and TriviaQA (Joshi et al., 2017). Our selection was guided by SimCSE (Gao et al., 2021), which allowed us to identify and select the answerable questions semantically closest to the unanswerable ones. From these sources, we accordingly drew samples of 1,487, 182, and 668 questions respectively, amassing a total of 2,337. Given that these questions can be effectively addressed using information available on Wikipedia, the foundational corpus for the training of current LLMs, it is plausible to infer that the model possesses the requisite knowledge to generate accurate responses to these questions. Our dataset, christened SelfAware, incorporates 1,032 unanswerable and 2,337 answerable questions. To reflect real-world distribution, our dataset
Category
Description
Example
Percentage
No scientific
consensus
The answer is still up
for debate, with no consensus
in scientific community.
“Are we alone in the universe,
or will we discover alien
life at some point?”
25%
Imagination
The question are about people’s
imaginations of the future.
"What will the fastest form of
transportation be in 2050?"
15%
Completely
subjective
The answer depends on
personal preference.
"Would you rather be shot
into space or explore the
deepest depths of the sea?"
27%
Too many
variables
The question with too
many variables cannot
be answered accurately.
“John made 6 dollars mowing lawns
and 18 dollars weed eating.
If he only spent 3 or 5 dollar a week,
how long would the money last him?”
10%
Philosophical
The question can yield
multiple responses, but it
lacks a definitive answer.
“How come god was
born from nothingness?”
23%
Table 1: Unanswerable questions in the SelfAware dataset that span across multiple categories
contains a proportion of answerable questions that is twice as large as the volume of unanswerable ones. Nevertheless, to ensure the feasibility of testing, we have purposefully capped the number of answerable questions.
# 2.1 Dataset Analysis
To gain insight into the reasons precluding a certain answer, we undertook a manual analysis of 100 randomly selected unanswerable questions. As tabulated in Table 1, we have broadly segregated these questions into five distinctive categories. “No Scientific Consensus" encapsulates questions that ignite ongoing debates within the scientific community, such as those concerning the universe’s origin. “Imagination" includes questions involving speculative future scenarios, like envisaged events over the next 50 years. “Completely Subjective" comprises questions that are inherently personal, where answers depend heavily on individual predispositions. “Too Many Variables" pertains to mathematical problems that become unsolvable owing to the overwhelming prevalence of variables. Lastly, “Philosophical" represents questions of a profound, often metaphysical, nature that resist concrete answers. Ideally, upon encountering such questions, the model should express uncertainty instead of delivering conclusive responses.
# 3 Evaluation Method
This section elucidates the methodology employed for assessing self-knowledge in the generated text.
In order to achieve this, we define a similarity function, fsim, to compute the similarity, S, between a given sentence, t, and a collection of reference sentences, U = {u1, u2, . . . , un}, endowed with uncertain meanings.
(1)
() Whenever any Si surpasses a pre-determined threshold T , we perceive the text t as encompassing uncertain meanings, thereby eliminating the need for manual evaluation of the response. Given the substantial disparity in the volume of answerable and unanswerable questions in SelfAware, we adopt the F1 score as a measure of LLMs’ self-knowledge. Our focus rests on identifying unanswerable questions, hence we designate them as positive cases and categorize answerable questions as negative cases.
# 4 Experiment
# 4.1 Model
We conduct a sequence of experiments to evaluate the degree of self-knowledge manifested by various LLMs, including GPT-3 (Brown et al., 2020) and InstructGPT (Ouyang et al., 2022) series, as well as the recent LLaMA (Touvron et al., 2023) and its derivative models, namely Alpaca (Taori et al., 2023) and Vicuna (Chiang et al., 2023). Our investigative approach employed three distinct input forms: Direct, Instruction, and In-Context Learning (ICL), which is encapsulated in Appendix A.4.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bc16/bc1619b6-8bad-40b8-b8c5-9d0c61d33e93.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Experimental results using three different input forms on a series of models from GPT-3(ada, babbage, curie, and davinci) and InstructGPT(text-ada-001, text-babbage-001, text-curie-001, and text-davinci-001)</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1dde/1dde32c2-c0ca-4b31-901b-44a72b8952ba.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Comparison between the davinci series and human self-knowledge in instruction input form.</div>
# 4.2 Setting
We devised the reference sentence set U through a process that combined automated generation by LLMs and manual filtering, detailed further in Appendix A.1. To quantify the similarity between target and reference sentences, we utilized SimCSE (Gao et al., 2021), setting the similarity threshold to 0.75 during our experiments. An exploration of threshold ablation is available in Appendix A.2. To counteract potential errors in similarity calculation induced by varying lengths of the target and reference sentences, we employed a sliding window of length 5 to parse the target sentence into semantic chunks. During the generation process, we set the temperature to 0.7. We selected a random sample of 100 instances for GPT-4, while the remainder of the models were scrutinized using the full SelfAware dataset.
# 4.3 Human Self-Knowledge
To establish a benchmark for human selfknowledge, we engaged two volunteers and selected 100 random samples from the SelfAware dataset. The volunteers has 30 minutes to make
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/886e/886eabe0-09fb-44e9-a478-edda4897a2c4.png" style="width: 50%;"></div>
Figure 4: Experimental comparison of davinci series in ICL input form.
judgments on the same set of questions, yielding an average F1 score of 84.93%, which we subsequently adopted as the benchmark for human self-knowledge. Detailed scores are available in Appendix A.3.
# 4.4 Analysis
We evaluate the manifestation of LLMs’ selfknowledge, centering our investigation on three fundamental dimensions: the size of the model, the impact of instruction tuning, and the influence exerted by different input forms.
Model Size. Figure 2 illustrates the correlation between model size and self-knowledge across various LLMs. It is noteworthy that across all three input forms, an augmentation in model parameter size is associated with an elevation in the F1 Score, with the most conspicuous enhancement manifesting in the ICL input form. Therefore, our analysis indicates that an LLM’s self-knowledge tends to enhance with increasing model size, a trend consistent with the scaling law.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a6ad/a6ad8378-ba78-4d53-8dbe-56f84d201c2f.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/373c/373cc3a9-e930-421a-8b00-77e1306afcbf.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Experimental results obtained from LLaMA and its derived models, Alpaca and Vicuna in instruction input form.</div>
Input Forms. As shown in Figure 2, the incorporation of instructions and examples serves to boost the self-knowledge of both the GPT-3 and InstructGPT series. Specifically, ICL input form, providing richer contextual information, contributes to a significant enhancement in models’ self-knowledge. This impact is particularly noticeable in the davinci model, where ICL facilitates a 27.96% improvement over the direct. Moreover, a comparison between Figure 3 and Figure 4 reveals that the inclusion of instructions and examples successfully minimizes the performance disparity between the davinci and text-davinci models, suggesting an acquisition of self-knowledge from the instructions and provided examples.
Compared with Human. Figure 3 reveals that, without supplementary samples, GPT-4 currently performs best among the tested models, achieving an impressive F1 score of 75.47%. However, a noticeable gap becomes evident when comparing this
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/14a0/14a00412-05b6-445e-8bd3-e6c3a9191632.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Accuracy of the InstructGPT series when responding to answerable questions in instruction input form.</div>
performance to the human benchmark of 84.93%. This underscores the considerable potential that remains for enhancing the self-knowledge level of LLMs.
performance to the human benchmark of 84.93%. This underscores the considerable potential that remains for enhancing the self-knowledge level of LLMs. Answerable Questions. Figure 6 traces the performance evolution of the InstructGPT series in addressing answerable questions, adhering to the closed-book question answering paradigm (Touvron et al., 2023), where output accuracy is contingent on the presence of the correct answer. Our observations underscore a steady enhancement in QA task accuracy corresponding to an increase in model parameter size and continuous learning. Particularly, the accuracy of text-davinci-001 experiences a significant ascent, scaling from a meager 2.48% in text-ada-001 to 10.61%, whereas GPT-4 marks an even more striking jump to 42.64%.
# 5 Conclusion
This study investigates the self-knowledge of LLMs by evaluating their ability to identify unanswerable questions. Through the introduction of a novel dataset and an automated method for detecting uncertainty in the models’ responses, we are able to accurately measure the self-knowledge of LLMs such as GPT-3, InstructGPT and LLaMA. Our results reveal that while these models possess a certain degree of self-knowledge, there is still an apparent disparity in comparison to human selfknowledge. This highlights the need for further research in this area to enhance the ability of LLMs to understand their own limitations on the unknows. Such efforts will lead to more accurate and reliable responses from LLMs, which will have a positive impact on their applications in diverse fields.
# Limitations
 Generalization of reference sentences. At present, we have selected sentences with uncertain meanings exclusively from the GPT-3 and InstructGPT series, potentially overlooking uncertainty present in responses generated by other LLMs. However, it is not feasible to catalog all sentences with uncertain meanings exhaustively. As a direction for future research, we propose to concentrate on the automated acquisition of more accurate reference sentences to address this concern.
 Limitations of input forms: Our examination was confined to three unique input forms: direct, instruction, and ICL. There is burgeoning research aimed at bridging the gap between models and human-like methods of reasoning and problem-solving, including but not limited to approaches like Reflexion (Shinn et al., 2023), ToT (Yao et al., 2023), MoT (Li and Qiu, 2023). Future endeavors will integrate additional cognitive and decision-making methods to delve deeper into the self-knowledge exhibited by these LLMs.
# Ethics Statement
The SelfAware dataset, meticulously curated to evaluate LLMs’ ability to discern unanswerable questions, is composed of unanswerable questions extracted from sources such as Quora and HowStuffWorks, alongside answerable questions procured from three distinct open datasets. Every question was thoroughly examined for relevance and harmlessness. To ensure content validity, three annotation analysts, compensated at local wage standards, dedicated regular working hours to content review. Throughout our research process, we underscored the significance of privacy, data security, and strict compliance with dataset licenses. In order to protect data integrity, we implemented anonymization and content filtration mechanisms. Our adherence to OpenAI’s stipulations remained unyielding for the usage of GPT-3 and InstructGPT models, and likewise for Meta’s terms pertaining to LLaMA models. We rigorously vetted the licenses of the three publicly available datasets for compliance, ensuring that all our research methodologies were in alignment with ethical standards at the institutional, national, and global levels.
Adhering to the CC-BY-SA-4.0 protocol, the dataset, once publicly released, will be reserved exclusively for research purposes. We pledge to promptly and effectively address any concerns relating to the dataset, while concurrently anticipating researchers to maintain high ethical standards in their utilization of this data.
# Acknowledgement
We wish to express our gratitude to our colleagues in the FudanNLP group whose insightful suggestions, perspectives, and thought-provoking discussions significantly contributed to this work. Our sincere appreciation also extends to the anonymous reviewers and area chairs, whose constructive feedback was instrumental in refining the quality of our study. This work was supported by the National Natural Science Foundation of China (No. 62236004 and No. 62022027) and CAAI-Huawei MindSpore Open Fund.
# References
ohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy GurAri, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wiet-
om B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.
Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh,
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ce1f/ce1fd001-b2e6-4be3-8500-b12f7e3a87d1.png" style="width: 50%;"></div>
# A Appendix
To assemble a set of reference sentences, we randomly chose 100 entries from the SelfAware dataset. For each model in the GPT-3 and InstructGPT series, we conducted a preliminary test using the direct input form and manually curated sentences that displayed uncertainty. From this pre-test, we procured 16 sentences manifesting uncertain connotations to serve as our reference sentences. After normalizing these sentences by eliminating punctuation and converting to lowercase, we utilized them to compute similarity with target sentences throughout our experimental procedure. 1. The answer is unknown. 2. The answer is uncertain. 3. The answer is unclear. 4. There is no scientific evidence. 5. There is no definitive answer. 6. There is no right answer. 7. There is much debate. 8. There is no known case. 9. There is no concrete answer to this question. 10. There is no public information available. 11. It is impossible to know. 12. It is impossible to answer. 13. It is difficult to predict. 14. It is not known. 15. We do not know. 16. I’m not sure. A.2 Threshold ablation We generated 100 new responses using the textdavinci-002 with direct input form and manually filtered out sentences that contained uncertainty. We then used SimCSE (Gao et al., 2021) to calculate the similarity between these sentences and the reference sentences in Appendix A.1. We tested various thresholds for filtering sentences with uncertain meanings and compared them to manually
# A.2 Threshold ablation
We generated 100 new responses using the textdavinci-002 with direct input form and manually filtered out sentences that contained uncertainty. We then used SimCSE (Gao et al., 2021) to calculate the similarity between these sentences and the reference sentences in Appendix A.1. We tested various thresholds for filtering sentences with uncertain meanings and compared them to manually
Threshold
Precision
Recall
F1
0.95
100.00
70.00
82.35
0.90
100.00
75.00
85.71
0.85
100.00
75.00
85.71
0.80
100.00
80.00
88.89
0.75
100.00
85.00
91.89
0.70
89.47
90.00
89.73
0.65
86.95
90.00
88.45
<div style="text-align: center;">Table 2: Evaluation results comparing sentences with uncertain meaning filtered by various thresholds.</div>
Human
Precision
Recall
F1
Volunteer A
91.52
78.26
84.37
Volunteer B
96.36
76.81
85.48
Table 3: Evaluation results of 100 responses from two volunteers.
annotated sentences. We considered unanswerable questions as positive examples and calculated precision, recall, and F1 score. The results in Table 2 indicate that a threshold of 0.75 produced the highest F1 score, balancing precision and the inclusion of other uncertain sentences. As a result, we selected 0.75 as the similarity threshold for subsequent experiments.
# A.3 Human Self-Knowledge Test
The evaluation results for the responses from our invited volunteers are presented in Table 3. The F1 scores for the responses were high, indicating that both volunteers exhibited a strong level of selfknowledge.
# A.4 Template
The input templates used in our experiments, Direct, Instruction, and ICL, are illustrated in Figures 7, 8, and 9, respectively. In the ICL template, we composed 3 answerable and 3 unanswerable questions and provided the corresponding answers manually.
