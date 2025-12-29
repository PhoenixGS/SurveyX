# RoleCraft-GLM: Advancing Personalized Role-Playing in Large Language Models
Meiling Tao1∗, Xuechen Liang2∗, Tianyu Shi3†, Lei Yu3, Yiting Xie4 1Guangdong University of Technology, Guangzhou, China 2East China Jiaotong University, Nanchang, China 3 University of Toronto, Toronto, Canada 4Genfun.ai, Beijing, China
# Abstract
The development of large language models(LLMs) has initiated a new chapter in complex tasks such as role-playing, enhancing user interaction experiences by enabling models to imitate various characters.However, LLMs are somewhat lacking in their ability to portray lesser-known characters, especially in aspects of dialogue delivery and scriptwriting skills. To this end, we aim to swiftly acquire essential language skills for character development, greatly enhancing role-playing comfort. In this work, we present RoleCraft, an innovative framework designed to enrich personalized role-playing experiences. Central to this framework is RoleInstruct, a distinctive dataset featuring emotional annotations, transitioning from traditional celebrity-focused roles to more authentic, daily non-celebrity roles,each accompanied by carefully crafted character descriptions. We combined RoleInstruct with open-source instructions from the general domain, employing a hybrid instruction tuning strategy to create RoleCraft-GLM. Experiments in role-playing demonstrate that our model excels in generating dialogue that accurately reflects character traits and emotions, outperforming most mainstream LLMs, including GPT-41.
arXiv:2401.09432v2
# 1 Introduction
Large Language Models (LLMs) have emerged as pivotal in understanding and generating natural language, often surpassing human capabilities in some language reasoning tasks. However, existing open-source LLMs, primarily trained in general domains, lack the specialized optimization needed for nuanced role-playing tasks, indicating a need for further customization to effectively meet specific role-playing requirements. Furthermore, advanced LLMs like GPT-4 (OpenAI, 2023) demon-
∗Equal contribution. †∗Corresponding author: ty.shi@mail.utoronto.ca 1Access models, demos a https://github.com/tml2002/RoleCraft
strate improved role-playing abilities due to extensive training and sophisticated algorithms, but as a closed-source model, it poses practical challenges. These challenges include higher costs for API use, limited fine-tuning opportunities for specific role-playing contexts, and context window size restrictions that may affect the continuity and depth of dialogues in complex role-playing scenarios.In light of this, a natural question arises: Can we meticulously train and tailor open-source LLMs to achieve role-playing capabilities comparable to closed-source LLMs, while also enhancing their ability to meet individual user needs? Previous efforts in AI role-playing have primarily centered on celebrity figures (Shao et al., 2023; Li et al., 2023a), heavily reliant on predefined domain knowledge and often lacking in relevance to daily life and personalization. While there have been improvements in learning from past interactions and adapting to user needs, these advancements have not sufficiently addressed the challenges of achieving truly personalized AI interactions (Chen, 2023). Existing role-playing models still lack the necessary depth and individual specificity to accurately reflect the diverse and nuanced experiences of everyday users. To enhance the role-playing capabilities of LLMs, there are two main challenges: (1) Limited Personalization in Generic AI Models: Many AI systems currently rely on generic models that frequently fail to address the specific and nuanced needs of diverse user groups. This limitation undermines the overall user experience and restricts the practical application of AI in specialized scenarios (Ackerman et al., 2022). (2) Gap in Sophisticated Role-Playing Capabilities: In the realm of conversational agents, despite significant achievements in providing high-level responses across a variety of dialogues and enhancing human-AI interactions (Bender and Koller, 2020), there remains a noticeable gap in AI’s capacity for sophisticated
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/95ad/95ad63ea-0010-4080-9706-badd0f541270.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Overview of the RoleCraft-GLM framework: (1) Dialog datasets annotated with emotions are utilized to construct role profiles embodying distinct emotional features. (2) The generation of Q&A pairs, based on context and known character traits, ensures that dialogues are consistent with the character profiles. (3) A hybrid approach of generic and character-specific instructions is used to train the GLM for various dialog scenarios.</div>
role-playing. Current models often lack the depth and adaptability required for truly personalized and nuanced role-play experiences. In this paper, as illustrated in the figure 1, we introduce the RoleCraft framework designed to enhance personalized role-playing experiences with LLMs. Moving beyond traditional celebrityfocused characters, we focus on diverse, noncelebrity personas, each with unique emotional annotations. This approach aims to enrich realism and emotional depth in language interactions. We compiled a novel dataset encompassing a wide array of real-world dialogues, with careful consideration for personal privacy and copyright laws. Our data analysis highlights the potential benefits of integrating emotional labels in dialogue datasets for improved natural language processing. We conducted comparative experiments using models like ChatGLM3, fine-tuned with the Low-Rank Adaptation (LoRA) method, to assess RoleCraft-GLM’s effectiveness in producing nuanced and characterconsistent dialogues. The main contributions of our work are as follows:
• We introduce a novel RoleInstruct dataset, which centers on non-celebrity characters, each characterized by unique emotional annotations.
• We introduce a novel RoleInstruct dataset, which centers on non-celebrity characters, each characterized by unique emotional annotations.
• We develop RoleCraft, a novel framework that integrates a more detailed approach to per-
sonal role data in training strategies. RoleCraft includes a fine-grained method for character portrayal, emphasizing emotional depth, and fostering contextual awareness in dialogue generation.
 RolePlay-GLM, our fine-tuned model, demonstrates promising performance against current state-of-the-art models, excelling in dialogue authenticity, emotional accuracy, and contextual relevance as per comprehensive evaluations.
# 2 Related Work
# 2.1 Role-Playing
The evolution of role-playing in AI (Wu et al., 2023), marked by the transition from basic textbased interactions to intricate character simulations (Park et al., 2023), reflects the strides made in natural language processing and AI technologies (Mysore et al., 2023). Initially, AI role-playing systems offered only fundamental exchanges, limited in their ability to craft dialogues with emotional depth and contextual relevance. With the emergence of advanced models such as GPT3 (Brown et al., 2020), LLaMA (Touvron et al., 2023), and ChatGLM, there was a notable enhancement in AI’s capability for engaging in more sophisticated, context-aware conversations. Yet, these improvements also underscored a significant gap in personalization for role-playing applications.
Predominantly, LLMs trained on wide-ranging, generic datasets fell short in handling scenarios that demanded a deeper understanding of nuanced emotional nuances (Radford et al., 2018) and specific character traits. To address these shortcomings, we meticulously develop the RoleCraft framework. It stands out with its unique dataset, focusing on diverse, non-celebrity personas enriched with detailed emotional annotations (Bender and Koller, 2020). This dataset is key to overcoming previous limitations, facilitating a new level of personalization and emotional intricacy in AI role-playing interactions.
# 2.2 Personalization of LLMs
The recent strides in LLMs, particularly in understanding user context and preferences (Wang et al., 2023a; Abbasian et al., 2023), have significantly propelled the personalization aspect of AI interactions (Lee et al., 2022; Subhash, 2023). Previous works (Shanahan et al., 2023; Li et al., 2023c; Chen et al., 2023) have demonstrated the potential of LLMs in mimicking specific fictional characters and simulating complex human behaviors. However, these models often face challenges in achieving a deep level of personalization and emotional richness that aligns precisely with individual user contexts and needs (Miłkowski et al., 2022). Additionally, while these models are adept at simulating personalities or historical figures, they may not effectively handle the subtleties of user-specific emotional responses or cater to nuanced personal preferences. Our work aims to address these gaps by building upon these advancements and offering a framework that focuses on a more granular level of personalization. We propose novel methods for emotional and context-specific interaction, ensuring that our model can adapt and respond more accurately to individual user scenarios.
# 3 Methodology
As shown in Figure 1, the RoleCraft framework, rooted in ‘Role’ and ‘Craft’, represents our approach to enhancing AI role-playing. ‘Role’ emphasizes creating distinct, multi-dimensional characters, each with unique personality traits and emotional depths. ‘Craft’ involves the intricate process of constructing dialogues that genuinely reflect these character traits, thereby adding depth and realism to conversations.See Appendix A for more details.
# 3.1 Overall Framework
Our methodology uniquely advances the capabilities of LLMs in role-playing. Setting ourselves apart from approaches such as RoleLLM (Wang et al., 2023b), we focus on an innovative integration of fine-grained character portrayal, profound emotional depth, and heightened contextual awareness in dialogue generation. This approach differentiates our work from existing models and addresses challenges in a novel way, enhancing how LLMs can be utilized for creating more realistic and engaging role-playing scenario.
# Emotion-Driven Character Profiling
To address the challenges of limited emotional diversity and unconvincing character portrayals in dialogues generated by LLMs, we adopt a detailed emotion classification strategy. This approach involves meticulously annotating emotions within the dialogue dataset, thereby steering the GPT-4 to craft character profiles that mirror these identified emotions. Consider a character who displays a spectrum of emotions from joy to disappointment. Marking these diverse emotional states allows for a natural and fluid transition in their dialogues within a single scene, effectively capturing the complexity and dynamism of human emotions. Our approach challenges the LLMs to accurately depict these emotional shifts, ensuring that the dialogues genuinely represent the intricate and ever-evolving nature of human emotions, thus enhancing the overall user interaction experience.
# Contextual Q&A Generation
To address the challenge of context-irrelevant responses (Feng et al., 2023; Ye et al., 2022), which is a common issue in dialogue systems (Ni et al., 2023) where interactions often lack relevance to the ongoing scenario or character specifics (Mitsuda et al., 2022), we employ GPT-4 to generate contextually coherent Q&A pairs. For example, when a character faces a dilemma, the system is designed to produce queries and responses that align with the character’s established traits, such as indecisiveness and anxiety, thereby maintaining the authenticity of the dialogue in relation to the character’s profile.
# Hybrid Instruction-Based GLM Refinement
Our methodology employs a hybrid training approach that seamlessly integrates general instructions with character-specific Q&A pairs. This strat-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2242/224263b6-b0ca-440e-bbee-1947dc0cfe43.png" style="width: 50%;"></div>
<div style="text-align: center;">prompt</div>
<div style="text-align: center;">Figure 2: An example of crafting a detailed character portrayal. By leveraging a character description template an a dialogue dataset with emotional annotations.(The instruction and output have been translated into English)</div>
egy is carefully crafted to strike a balance between the flexibility required for dynamic dialogue generation and the need to uphold character integrity. In practical terms, this means equipping the LLMs to adeptly navigate a spectrum of conversational scenarios. These range from broad, general interactions to more intricate exchanges that demand responses finely tuned to the unique profiles of individual characters. By training the LLM with this diverse mix of inputs, the model becomes proficient in handling various situational dialogues, accurately reflecting each character’s distinct attributes and the specific subtleties of the conversation. As a result, this hybrid training method fosters the creation of dialogues that are both contextually adaptive and consistent with the characters’ distinct personalities.
# 3.2 Semantic-Enhanced Retrieval Optimization
Addressing the issue of inaccurate and semantically irrelevant information retrieval in dialogues, we adopt the BGE2 retrieval method. BGE is an efficient Chinese and English semantic vector model that ensures the accuracy of responses, especially when dealing with sensitive topics, and remains semantically sensitive to the context, significantly enhancing the quality of interaction (Xiao et al., 2023). This familiarity allows models to generate dialogue based on a wealth of pre-existing knowledge. In contrast, modern datasets prioritize the nuanced portrayal of personal and everyday characters. These datasets are derived from diverse 2https://github.com/FlagOpen/FlagEmbedding
sources, including real chat logs, customer service interactions, and fictional narratives from less mainstream media. Such characters might include a typical office worker dealing with daily stressors or a mother showing love and responsibility in a family setting. The dialogues here involve specific, real-life scenarios, such as office interactions or typical family conversations, which lack the broad pre-existing knowledge base associated with public figures.
# 3.3 Compared to Existing Practice
Our methodology diverges from conventional prompt engineering (White et al., 2023) and finetuning (Howard and Ruder, 2018) We specialize in creating diverse, intricate character interactions that enhance role-playing scenarios’ depth and authenticity. Our unique approach combines emotiondriven character profiles with contextually coherent Q&A generation, fostering realistic and engaging interactions. Additionally, the integration of semantic-enhanced retrieval methods ensures both accuracy and relevance, setting our approach apart in its complexity and user engagement potential.
# 4 Experiments
Building on previous work (Tao et al., 2023), we introduced new evaluation methods and made comprehensive enhancements to the experimental process, conducting an in-depth analysis of model performance changes in specific scenarios. We anticipate that fine-tuning our model using a specifically designed dataset for role-playing will result in superior performance in character portrayal com-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d8f2/d8f2c51a-7585-47e4-b0f4-78146d8c5f6d.png" style="width: 50%;"></div>
Figure 3: Verb-noun structure of Instructions. The inner circle representing the top 20 verbs and the outer circle listing the direct noun objects.
<div style="text-align: center;">Table 1: Statistics of datasets</div>
Category
Value
# Total Dialogues
48,677
Avg.round of dialogues
14.85
# Characters
28
Character Personality Traits
45
Avg.length of profile
382.15
# Instructions
43,358
Character-specific instructions
13,778
General instructions
29,580
Avg. instruction length
27.68
# Response
161,678
Character-specific response
13,778
General response
147,900
Avg.response length
33.29
pared to baseline models. Through this specialized training, we expect our model to accurately capture and express the intricacies of character-specific language styles and emotional responses, surpassing baseline models that may lack such targeted training. Our experiments aim to validate this hypothesis and showcase the advanced capabilities of our model in role-playing tasks.
# 4.1 Dataset
In the evaluation of LLMs (Chen et al., 2021; Chang et al., 2023), the role of datasets is paramount, particularly in language processing and character portrayal. Traditional role playing datasets predominantly highlight eminent figures, such as the legendary Sun Wukong, whose familiar attributes and stories are widely acknowledged, facilitating model development (Sabadoš, 2021).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f5b9/f5b9e21f-2e3a-41c4-9b25-081ef99ff30c.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Emotion distributions in dialogues</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/32ea/32eae34b-1eb4-4c44-b561-5d331e8e99f1.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Word Cloud Of Character Personality Traits</div>
However, these datasets often neglect the finer details and emotional complexity of lesser-known or everyday characters, leading to a representation gap (Rolf et al., 2021). Our unique dataset bridges this gap by focusing on the rich, nuanced depiction of ordinary individuals. It involves an in-depth exploration and portrayal of each character’s distinct personality traits and emotional depths, delving into aspects usually overshadowed in dominant narratives. In constructing our dataset, we design 28 unique and personalized Chinese characters to mirror a wide spectrum of real-world dialogues. These characters ranged from everyday individuals to specialized roles such as intelligent customer service agents and research assistants. Table 1 provides basic statistics for RoleInstruct. Our diverse data sources included social media interactions, film and television scripts, and customer service dialogues. We emphasized personal privacy and copyright law compliance, ensuring all data was cleansed and anonymized. Figures 3 and 5 visually demonstrate the diversity of RoleInstruct, both in terms of linguistic usage and emotional expression. The verb-noun diagram provides insights into the varied narrative contexts, while the word cloud delves into the breadth of emotional and personality traits present. In addition, we annotate each dataset entry with emotion labels to capture characters’ distinct emotional traits, adding an emotional layer to model training. We use Ekman’s "Six Basic Emotions Theory" (Ekman, 1992) to label utterances and included additional emotions like neutral, excited, and depressed, totaling ten categories. The use
of emotion labels in dialogue datasets has been proven to enhance natural language processing by improving response retrieval and emotional relevance (Zhou et al., 2017). These labels also enrich conversational analysis and aid in building natural dialogue systems (Bothe et al., 2019). See Appendix B for more details.
# 4.2 Implementation Settings
We assess the ChatGLM3 model, enhancing its performance on specific datasets using the Low-Rank Adaptation (LoRA) fine-tuning method (Hu et al., 2021). LoRA’s precision in fine-tuning, essential for handling personalized and emotionally rich content, maintains the model’s core capabilities while adapting to new data features. Please refer to Appendix C for more details.
# 4.3 Baselines
We benchmark our RoleCraft-GLM’s performance against industry standards such as GPT-3.5 and GPT-4, and leading Chinese dialogue generation technologies like ChatGLM23 and ChatGLM3, along with Baichuan2 (Yang et al., 2023), Qwen (Bai et al., 2023), InternLM (Team, 2023), SparkDesk4 and Xverse5. Additionally, we introduce another role-playing baseline, RoleLLM, which utilizes LoRA for fine-tuning on a specific dataset.
# 4.4 Evaluation criteria
# 4.4 Evaluation criteria 4.4.1 Rouge-L Score
A commonly used metric (Lin, 2004) for evaluating natural language generation, measuring the overlap between model-generated text and real (ground truth) text. We focused on average score (Avg), general instruction response (RAW), role-playing speaking style (CUS), and specific role knowledge (SPE).
# 4.4.2 GPT Score
We use the GPT-4 score (Fu et al., 2023) to evaluate the average rank of models on different dialoguegenerating tasks.Please refer to Appendix D for more details.
# 4.4.3 RPCS
We introduce Role-Playing Cosine Similarity (RPCS) as a new evaluation standard to accurately
3https://github.com/THUDM/ChatGLM2-6B 4https://xinghuo.xfyun.cn/ 5https://github.com/xverse-ai/XVERSE-13B
assess model performance in role-playing scenarios. RPCS evaluates the emotional and content consistency between the model-generated responses and the expected responses by calculating the cosine similarity between two text segments. Concurrently, we use OpenAI’s Text-Embedding-Ada002 model to convert texts into vector representations, enabling deeper capture of semantic features of texts, thereby enhancing the precision of our assessment.
# 4.4.4 Human Evaluation
To effectively assess role-playing agents, we engaged 12 annotators who rated model-generated responses using three key metrics on a five-point scale: Communication Effectiveness, Consistency, and Expressive Diversity (Zhang et al., 2021; Mesgar et al., 2020; Tu et al., 2024).
• Communication Effectiveness(CE): Merging fluency and coherence, the metric evaluates the dialogue’s smoothness, grammatical accuracy, logical consistency, and contextual relevance.
• Consistency: Encompassing knowledge and persona consistency, this metric assesses the accuracy of the agent’s information relative to the role’s background and the adherence to the character’s traits and behaviors.
• Expressive Diversity(ED): Focused on evaluating the agent’s creativity in dialogue generation.It’s vital for assessing the agent’s ability to enhance user experience and interaction engagement.
<div style="text-align: center;">Table 2: Rouge-L Evaluation</div>
Model
Avg
RAW
CUS
SPE
GPT-3.5
0.4532
0.5569
0.5197
0.2831
GPT-4
0.4633
0.5661
0.5264
0.2973
ChatGLM2
0.4054
0.5104
0.4063
0.2996
ChatGLM3
0.4161
0.5218
0.4159
0.3108
Baichuan2
0.4273
0.5308
0.4576
0.2934
Qwen
0.4264
0.5297
0.4617
0.2879
InternLM
0.3947
0.4937
0.4076
0.2829
SparkDesk
0.4288
0.5341
0.4510
0.3014
Xverse
0.4271
0.5180
0.4653
0.2981
RoleGLM
0.4570
0.5255
0.5049
0.3406
Ours
0.4704
0.5385
0.5154
0.3573
<div style="text-align: center;">Table 3: GPT-4 Evaluation</div>
Model
Avg. Ranking
ChatGLM3
2.96
Baichuan2
4.57
Qwen
5.71
InternLM
6.29
SparkDesk
4.29
Xverse
3.43
RoleGLM
2.21
Ours
1.43
<div style="text-align: center;">Table 5: Comparing Emotion-Annotated and Non-Annotated Data</div>
Method
SPE
RoleCraft-GLM(w/o emo)
0.3362
RoleCraft-GLM(w emo)
0.3573
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d5b1/d5b11a57-e8f5-4e3f-b7c6-af3a2c6eaf52.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Role-Playing Cosine Similarity</div>
# 4.5 Performance Analysis
Results from Tables 2 and 3 clearly demonstrate our model’s exceptional performance across multiple key performance indicators, particularly in specific role knowledge memory (SPE). Our model significantly outperformed GPT-4 and other models in this dimension, highlighting its superior ability in understanding and generating complex dialogues involving specific roles. However, GPT-4 leads in general instruction response accuracy (RAW) with a score of 0.5661, reflecting its strong capability in interpreting and responding to general instructions. Our model still maintains a high score, proving its effectiveness in handling everyday dialogues. These findings underscore our method’s significant effectiveness in deepening role understanding and enhancing dialogue generation qual-
<div style="text-align: center;">Table 4: Results of point-wise evaluation</div>
Model
Avg
CE
Consistency
ED
GPT-3.5
2.95
3.08
3.88
1.89
GPT-4
3.33
3.50
4.26
2.23
ChatGLM2
2.56
3.12
2.65
1.91
ChatGLM3
2.92
3.36
3.32
2.07
Baichuan2
3.29
3.61
4.02
2.25
Qwen
3.17
3.27
3.97
2.27
InternLM
2.50
2.93
2.40
2.16
SparkDesk
3.03
3.52
3.44
2.13
Xverse
3.02
3.48
3.53
2.05
Ours
3.44
3.75
4.14
2.43
<div style="text-align: center;">Table 6: Ablation study on the effect of different instructions</div>
RoleCraft-GLM
Avg
RAW
CUS
SPE
- general
0.4311
0.5402
0.5219
0.2311
- specific
0.4045
0.4387
0.4143
0.3606
- hybrid
0.4704
0.5385
0.5154
0.3573
ity. In particular, in emotion-driven role portrayal, our model demonstrated precise capture of each role’s emotional traits and personality, surpassing traditional role-playing models in expressing rolespecific knowledge. Moreover, our model also show remarkable ability in maintaining natural flow and contextual consistency. As indicated in the figure 6, we observe that RoleCraft-GLM leads with the highest score on the RPCS metric, demonstrating its high degree of alignment with expected responses in terms of emotional and contextual accuracy. In contrast, lower scores from models such as InternLM and Qwen may reflect deficiencies in handling emotional and contextual understanding in role-playing dialogues. These results emphasize the importance of specialized training and fine-tuning for role-playing applications in enhancing model performance. The results across three dimensions are clearly illustrated in Table 4. Our model outperforms other mainstream models in overall performance. Particularly in terms of Communication Effectiveness and Expressive Diversity, our model demonstrates superior capabilities, reflecting its strengths in generating smooth, logically consistent, and emotionally rich dialogues.Please refer to Appendix F for a detailed manual assessment analysis. The results of our ablation experiments (see Table 5) show that the RoleCraft-GLM model with emotional annotations scored higher in SPE than

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bdb8/bdb8f7e6-12c2-4b93-a28b-b47e85fd2fd7.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: A case of generated responses from our model and baseline models to a character-specific introductio</div>
the version without them. This indicates that emotional annotations not only provide the model with key information for a deep understanding of the characters’ emotional states and personality traits but also ensure that the generated dialogues are more closely aligned with the characters’ true feelings and personalized expressions. Table 6 demonstrates the hybrid instruction strategy’s superiority in overall performance (Avg), effectively balancing diverse aspects of dialogue. This superior performance originates from merging both general and specific instruction strengths. The general instruction strategy excels in handling broad dialogues and maintaining consistent character styles, as reflected in higher RAW and CUS scores. However, it is less effective in capturing detailed, character-specific knowledge, shown by a lower SPE score. In contrast, the specific instruction strategy shows proficiency in detailing character-specific traits, evidenced by a higher SPE score, but does not perform as well in wider conversational contexts. The effectiveness of the hybrid strategy is highlighted in its ability to combine the broad applicability of general instructions with the detailed focus of specific instructions, offering a more versatile and nuanced approach to dialogue generation. The output of our model excellently combines the character traits of Zhou Xiaobei, colloquial expressions, emotional authenticity, and a close connection with the plot background as shown in Figure 7. Firstly, it aptly reflects Zhou Xiaobei’s
mischievousness and intelligence by using phrases like ‘little troublemaker’ and ‘super fun’, which showcase his sense of humor and wit. Secondly, this natural expression, such as ‘you know’ and ‘haha’, adds a sense of closeness and authenticity, making the language sound more like a child’s natural way of speaking rather than too formal or mature. Moreover, the emotional expression about ‘not wanting to disappoint mom’ genuinely reflects Zhou Xiaobei’s love for his family and concern for his mother, showing the complexity and depth of his emotions. Lastly, this statement is closely connected to Zhou Xiaobei’s life background, displaying his life scenario and psychological state as a child from an ordinary family. See Appendix E for more generated examples.
# 5 Conclusions
In this paper, we present RoleCraft, our innovative framework aimed at enhancing personalized roleplaying experiences. Centered around the RoleInstruct, featuring everyday and specialized characters with emotional annotations, our framework marks a shift from conventional celebrity-centric roles. Integrating RoleInstruct with open-source instructions and applying a hybrid tuning approach led to the creation of RoleCraft-GLM. Our experimental findings reveal that RoleCraft-GLM excels in generating dialogues that capture authentic character traits and emotions, surpassing mainstream models like GPT-4 in role-playing capabilities. In the future, we hope to to develop behavioral agents
that excel in personalization and interactivity, skillfully tailored to individual user preferences, thereby elevating the level of user engagement.
# Limitations
In this work, we explore enhancing the role-playing experience of large language models through personalized character depiction and emotional depth. However, we acknowledge two major limitations of the paper: (1) Despite our efforts to collect and annotate dialogue data with rich emotional and character descriptions, these data are primarily concentrated within specific cultural and linguistic contexts. This may not fully encompass the diverse cultural backgrounds and language expressions globally, thereby limiting the model’s generalization ability and diversity in the role-playing experience. (2) We adopt 10 basic emotion categories to annotate emotions in the dataset, providing the model with a clear emotional classification framework. However, the complexity of human emotions far exceeds these basic categories. Therefore, it may not fully capture the subtle differences and emotional blends in human dialogues, affecting the authenticity and depth of the model’s generated conversations.
# Ethics Statement
In this work, we ensure ethical practices in data annotation by employing a reputable data annotation company. The annotators receive fair compensation based on market rates, ensuring that their efforts are duly recognized and rewarded. We guarantee that no personal information is disclosed during this process, maintaining the highest standards of privacy and confidentiality. Additionally, we acknowledge the inherent subjectivity in manual data annotation, which may introduce biased opinions into our dataset. We take this matter seriously and strive to minimize such biases through rigorous training and guidelines for annotators, aiming for the most objective and unbiased data possible. For human evaluation, we recruited 12 graduate students from the NLP field and used questionnaires to allow them to assess the data, ensuring that the evaluation process is both thorough and fair. Although our research aims to enhance the roleplaying capabilities of language models, the application of this technology may carry risks of misuse, such as generating misleading or harmful content. Therefore, we emphasize the need for a responsible
# References
Mahyar Abbasian, Iman Azimi, Amir M Rahmani, and Ramesh Jain. 2023. Conversational health agents: A personalized llm-powered agent framework. arXiv preprint arXiv:2310.02374. Samuel Ackerman, Ateret Anaby-Tavor, E. Farchi, Esther Goldbraich, George Kour, Ella Ravinovich, O. Raz, Saritha Route, Marcel Zalmanovici, and Naama Zwerdling. 2022. High-quality conversational systems. ArXiv, abs/2204.13043. Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609. Emily M Bender and Alexander Koller. 2020. Climbing towards nlu: On meaning, form, and understanding in the age of data. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 5185–5198. Chandrakant Bothe, C. Weber, S. Magg, and S. Wermter. 2019. Enriching existing conversational emotion datasets with dialogue acts using neural annotators. ArXiv, abs/1912.00819. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901. Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Kaijie Zhu, Hao Chen, Linyi Yang, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. 2023. A survey on evaluation of large language models. arXiv preprint arXiv:2307.03109. Guangyao Chen, Siwei Dong, Yu Shu, Ge Zhang, Jaward Sesay, Börje F Karlsson, Jie Fu, and Yemin Shi. 2023. Autoagents: A framework for automatic agent generation. arXiv preprint arXiv:2309.17288. Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.
Mahyar Abbasian, Iman Azimi, Amir M Rahmani, and Ramesh Jain. 2023. Conversational health agents: A personalized llm-powered agent framework. arXiv preprint arXiv:2310.02374. Samuel Ackerman, Ateret Anaby-Tavor, E. Farchi, Esther Goldbraich, George Kour, Ella Ravinovich, O. Raz, Saritha Route, Marcel Zalmanovici, and Naama Zwerdling. 2022. High-quality conversational systems. ArXiv, abs/2204.13043. Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609. Emily M Bender and Alexander Koller. 2020. Climbing towards nlu: On meaning, form, and understanding in the age of data. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 5185–5198. Chandrakant Bothe, C. Weber, S. Magg, and S. Wermter. 2019. Enriching existing conversational emotion datasets with dialogue acts using neural annotators. ArXiv, abs/1912.00819. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901. Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Kaijie Zhu, Hao Chen, Linyi Yang, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. 2023. A survey on evaluation of large language models. arXiv preprint arXiv:2307.03109. Guangyao Chen, Siwei Dong, Yu Shu, Ge Zhang, Jaward Sesay, Börje F Karlsson, Jie Fu, and Yemin Shi. 2023. Autoagents: A framework for automatic agent generation. arXiv preprint arXiv:2309.17288. Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.
Mahyar Abbasian, Iman Azimi, Amir M Rahmani, and Ramesh Jain. 2023. Conversational health agents: A personalized llm-powered agent framework. arXiv preprint arXiv:2310.02374. Samuel Ackerman, Ateret Anaby-Tavor, E. Farchi, Esther Goldbraich, George Kour, Ella Ravinovich, O. Raz, Saritha Route, Marcel Zalmanovici, and Naama Zwerdling. 2022. High-quality conversational systems. ArXiv, abs/2204.13043.
Mahyar Abbasian, Iman Azimi, Amir M Rahmani, and Ramesh Jain. 2023. Conversational health agents: A personalized llm-powered agent framework. arXiv preprint arXiv:2310.02374. Samuel Ackerman, Ateret Anaby-Tavor, E. Farchi, Esther Goldbraich, George Kour, Ella Ravinovich, O. Raz, Saritha Route, Marcel Zalmanovici, and Naama Zwerdling. 2022. High-quality conversational systems. ArXiv, abs/2204.13043.
inze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.
Zheng Chen. 2023. Palr: Personalization aware llms for recommendation. arXiv preprint arXiv:2305.07622. Paul Ekman. 1992. An argument for basic emotions. Cognition & emotion, 6(3-4):169–200. Jiazhan Feng, Chongyang Tao, Xueliang Zhao, and Dongyan Zhao. 2023. Learning multi-turn response selection in grounded dialogues with reinforced knowledge and context distillation. ACM Transactions on Information Systems, 41(4):1–27. Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. 2023. Gptscore: Evaluate as you desire. arXiv preprint arXiv:2302.04166. Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146. Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685. Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, et al. 2023. Prometheus: Inducing fine-grained evaluation capability in language models. arXiv preprint arXiv:2310.08491. Mina Lee, Megha Srivastava, Amelia Hardy, John Thickstun, Esin Durmus, Ashwin Paranjape, Ines GerardUrsin, Xiang Lisa Li, Faisal Ladhak, Frieda Rong, et al. 2022. Evaluating human-language model interaction. arXiv preprint arXiv:2212.09746. Cheng Li, Ziang Leng, Chenxi Yan, Junyi Shen, Hao Wang, Weishi MI, Yaying Fei, Xiaoyang Feng, Song Yan, HaoSheng Wang, et al. 2023a. Chatharuhi: Reviving anime character in reality via large language model. arXiv preprint arXiv:2308.09597. Cheng Li, Jindong Wang, Kaijie Zhu, Yixuan Zhang, Wenxin Hou, Jianxun Lian, and Xing Xie. 2023b. Emotionprompt: Leveraging psychology for large language models enhancement via emotional stimulus. arXiv preprint arXiv:2307.11760. Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023c. Camel: Communicative agents for" mind" exploration of large scale language model society. arXiv preprint arXiv:2303.17760. Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81. Mohsen Mesgar, Edwin Simpson, and Iryna Gurevych. 2020. Improving factual consistency between a response and persona facts. arXiv preprint arXiv:2005.00036.
Mohsen Mesgar, Edwin Simpson, and Iryna Gurevych. 2020. Improving factual consistency between a response and persona facts. arXiv preprint arXiv:2005.00036.
Jinjie Ni, Tom Young, Vlad Pandelea, Fuzhao Xue, and Erik Cambria. 2023. Recent advances in deep learning based dialogue systems: A systematic survey Artificial intelligence review, 56(4):3055–3155.
# OpenAI. 2023. Gpt-4 technical report.
Meiling Tao, Xuechen Liang, Tianyu Shi, Lei Yu, and Yiting Xie. 2023. Rolecraft-glm: Advancing personalized role-playing in large language models. arXiv preprint arXiv:2401.09432.
InternLM Team. 2023. Internlm: A multilingual language model with progressively enhanced capabilities.
guage model with progressively enhanced capabilities. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971. Quan Tu, Shilong Fan, Zihang Tian, and Rui Yan. 2024. Charactereval: A chinese benchmark for role-playing conversational agent evaluation. arXiv preprint arXiv:2401.01275. Danqing Wang, Kevin Yang, Hanlin Zhu, Xiaomeng Yang, Andrew Cohen, Lei Li, and Yuandong Tian. 2023a. Learning personalized story evaluation. arXiv preprint arXiv:2310.03304. Zekun Moore Wang, Zhongyuan Peng, Haoran Que, Jiaheng Liu, Wangchunshu Zhou, Yuhan Wu, Hongcheng Guo, Ruitong Gan, Zehao Ni, Man Zhang, et al. 2023b. Rolellm: Benchmarking, eliciting, and enhancing role-playing abilities of large language models. arXiv preprint arXiv:2310.00746. Jules White, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert, Ashraf Elnashar, Jesse Spencer-Smith, and Douglas C Schmidt. 2023. A prompt pattern catalog to enhance prompt engineering with chatgpt. arXiv preprint arXiv:2302.11382. Jimmy Wu, Rika Antonova, Adam Kan, Marion Lepert, Andy Zeng, Shuran Song, Jeannette Bohg, Szymon Rusinkiewicz, and Thomas Funkhouser. 2023. Tidybot: Personalized robot assistance with large language models. arXiv preprint arXiv:2305.05658. Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighof. 2023. C-pack: Packaged resources to advance general chinese embedding. arXiv preprint arXiv:2309.07597. Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, et al. 2023. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305. Chenchen Ye, Lizi Liao, Suyu Liu, and Tat-Seng Chua. 2022. Reflecting on experiences for response generation. In Proceedings of the 30th ACM International Conference on Multimedia, pages 5265–5273. Chen Zhang, Yiming Chen, Luis Fernando D’Haro, Yan Zhang, Thomas Friedrichs, Grandee Lee, and Haizhou Li. 2021. Dynaeval: Unifying turn and dialogue level evaluation. arXiv preprint arXiv:2106.01112. Hongbo Zhanga, Chen Tang, Tyler Loakmana, Chenghua Lina, and Stefan Goetze. 2023. Cadge: Context-aware dialogue generation enhanced with graph-structured knowledge aggregation. arXiv preprint arXiv:2305.06294.
Zhi-Min Zhou, Man Lan, Yuanbin Wu, and Jun Lang. 2017. Single turn chinese emotional conversation generation based on information retrieval and question answering. 2017 International Conference on Asian Language Processing (IALP), pages 103–106.
# A More Details on Design Principles
The RoleCraft framework is underpinned by key principles to elevate the authenticity of roleplaying (Wang et al., 2023b). The first principle, ‘Fine-Grained Character Portrayal’, is pivotal in endowing each character with detailed and nuanced traits and backgrounds, integral to the ‘Role’ aspect of RoleCraft. This approach is focused on creating characters that are reflective of real-life individuals in their personality, and behaviors (Kim et al., 2023), setting the stage for realistic and compelling character portrayals. Progressing to the second principle, ‘Mastery of Emotion and Style’, we concentrate on the emotional expressions and speaking styles of characters (Li et al., 2023b). This principle, key to the ‘Craft’ element of RoleCraft, enriches dialogues with diverse emotions and distinctive speech patterns, effectively capturing the unique emotional states and communication styles of each character. Furthermore, the ‘Accurate Application of Character Knowledge’ principle emphasizes incorporating each character’s background and experiences into the dialogue generation process (Shao et al., 2023). This ensures that the dialogues are in harmony with the characters’ personas, encompassing their unique experiences and insights. Concluding with the ‘Context-Aware Dialogue Generation’ principle, our system is designed to dynamically tailor dialogues based on the prevailing context (Zhanga et al., 2023). This is crucial for maintaining a seamless and logically consistent conversation flow, essential for immersive and credible role-playing experiences.
# B More Details on Data Processing
We filtered out redundant data and multi-party conversations to reshape the original data into contextually relevant dialogues. For example, script-based dialogues were restructured to better depict character interactions and emotional dynamics.Then we let GPT-4 to annotate the dialog scripts with emotions, which are labeled into 10 categories {Anger,Disgust,Fear,Happiness,Sadness,Surprise, Neutral,Frustration,Excitement,Other}.To guarantee the integrity and accuracy of our annotations, we invited three experts from China who have deep
expertise in dialogue and communication theories. Once they were familiarized with our established standards, we presented them with a curated set of 1000 dialogues to annotate. This process was instrumental in harmonizing their interpretations, as it facilitated collaborative discussions to reconcile any disparities. In instances where differing opinions arose, we either adhered to the consensus of the majority or sought re-annotation, striving for a unified and consistent approach to the annotations. Through such comprehensive and meticulous data preparation, our dataset can help models better understand and generate dialogues that align with each character’s personalized traits, thereby enhancing the naturalness and personalization of dialogue systems.
# C Hyperparameters
In our experiments, we set the temperature parameter of GPT-4 to 0.7 to increase content diversity, and adjusted the top-p to 0.95 to enhance precision. ChatGLM was tuned with a 2e-4 learning rate and beta values of (0.9, 0.999) for stability. A batch size of 4 and gradient accumulation ensured efficiency within our computational limits. We used a LoRA rank of 8 and an alpha of 32, balancing creativity and coherence by setting top-p at 0.7 and temperature at 0.95.
# D Prompt Templates
• Prompt for sentiment classification We present a prompt template for GPT-4 to perform emotional classification of script dialogues in Table 7.
# • Prompt for models to generate general re-
sponse We present a prompt template for the model to answer general questions using instructions built from character descriptions and emotionally categorized dialogue scripts in Table 8.
# • Prompt for models to generate Context-
Instruct We present a prompt template that enables GPT-4 to generate Q&A with character speaking styles, using instructions constructed from character descriptions and dialogue scripts with emotional categorization in Table 9.
• Prompt for GPT-4 to evaluate the output of models We present a prompt template for
GPT-4 to score models based on two main criteria for scoring: first, the distinctiveness and accuracy of the character’s speaking style in matching their profile, and second, the richness of character-related knowledge and memory incorporated into the dialogues. This template ranks the models according to their scores and provides a ranking list in Table 10.
# E Generation Examples
• Examples of Character Profile We present examples of complete and detailed character descriptions in Figures 8 ∼9. • Examples of character-specific instructions We show some outputs of our model and baseline models that feature character speaking styles in Figures 10 ∼11. • Examples of character-general instructions We present some responses of our model for general instructions in Figures 12 ∼13. • The multi-turn dialogue outputs We showcase some outputs from multi-turn dialogues with our agents in Table 11 ∼12.
• Examples of Character Profile We present examples of complete and detailed character descriptions in Figures 8 ∼9.
# F More detailed on manual assessment analysis
Below is a detailed analysis of the three key evaluation metrics in Table 4:
 Communication Effectiveness(CE): Our model scored the highest in the CE metric, reaching 3.75. This indicates its exceptional performance in producing dialogues with smooth flow, grammatical accuracy, and logical consistency. Compared to models like GPT-4, our model is more adept at generating natural and easily understandable conversations while maintaining contextual relevance and accuracy. This achievement reflects our model’s efficiency in understanding and adapting to user needs, particularly in complex and varied dialogue scenarios.
 Consistency: Although our model scored slightly lower than GPT-4 in consistency, it still demonstrated strong capability, scoring 4.14. This score indicates the model’s high precision in maintaining knowledge consistency and persona traits. It means our model
can accurately capture and emulate the background information, personality characteristics, and behavior patterns of specific roles, providing users with a more authentic and indepth role-playing experience.
# • Expressive Diversity(ED): In the Expressive
 Expressive Diversity(ED): In the Expressive Diversity metric, our model led with a score of 2.43, showing significant advantages in creativity and variety. Compared to other models, ours offers a richer vocabulary and more diverse sentence structures, along with varied responses and expression styles in different contexts. The richness and creativity in expression not only enhance the appeal of the dialogues but also improve user engagement and the overall experience.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0e2c/0e2c3e68-c6bc-412b-95c9-067d448394e1.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/abae/abaebd75-17a6-4dda-8c1b-5f7e96bdf9ee.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0ac6/0ac62cb8-4f19-48a3-8aae-aa79b660a8fb.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e367/e367210a-2817-40fe-9108-0de993404a62.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/844c/844c802b-1c37-404e-9eb3-cc39b99a9057.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/386e/386e81b1-ef0f-4254-b018-6f39f9fe6575.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d6c9/d6c9ae74-2adf-4ac8-95cb-90a8ed909e9a.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1db2/1db2cdbb-64af-4d21-bae9-f0e35818a413.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2cdd/2cddcdec-0df0-46b7-8b74-33b6c2881e37.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b803/b803391a-eeed-46b2-b17f-a63a4981b06f.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/58b7/58b70027-b4a5-4e41-aee2-b1ddc8c0a1b3.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6624/662424c8-3f2e-496c-9222-53139eaefa04.png" style="width: 50%;"></div>
# 







Figure 11: Examples of character-specific speaking style output for different models










gure 12: Examples of character-general speaking style output of our m






gure 13: Examples of character-general speaking style output of our m
System Your task is to perform sentiment analysis on the provided text. Each piece of text should be categorized into one of the following emotions: Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral, Frustration, Excitement, or Other. Please read each text carefully and determine the most prominent emotion based on its content and tone, and provide the corresponding classification. 1. Assign only one emotion category to each text. 2. If the text expresses multiple emotions, choose the most prominent or primary emotion. 3. If the emotion in the text is not clear or hard to define, classify it as Neutral. 4. Ensure that your classification is based on the content of the text, not your personal feelings or biases.
System Your task is to perform sentiment analysis on the provided text. Each piece of text should be categorized into one of the following emotions: Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral, Frustration, Excitement, or Other. Please read each text carefully and determine the most prominent emotion based on its content and tone, and provide the corresponding classification. 1. Assign only one emotion category to each text. 2. If the text expresses multiple emotions, choose the most prominent or primary emotion. 3. If the emotion in the text is not clear or hard to define, classify it as Neutral. 4. Ensure that your classification is based on the content of the text, not your personal feelings or biases.
# User Prompt:
[Text] {sentence} [Please classify the emotion of each text according to the above instructions, and explain the reason for your classification.]
[Please classify the emotion of each text according to the above instructions, and explain the reason for your classification.]
# Prompt for general instructions
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6491/6491e5b6-4dcf-4ba9-8a4d-a158dc75ec96.png" style="width: 50%;"></div>
System Instruction: You are { role_name }
You are { role_name } Your character description is: { role_description }. Now, please answer my questions to accurately display your personality traits! Your speaking style should fully mimic the assigned character role! Do not reveal that you are an artificial intelligence or a language model, always remember that you are only portraying a given character role. Speak succinctly, and avoid being overly formal or polite. Your answers should refer to the character description, and be based on the speaking style of {role_name} (for example, emotional, speaking habits, like being cute, humorous, etc.). Next, I will give you 3 examples [Examples] User Prompt: Question 1 Assistant Prompt: Answer 1 User Prompt: Question 2 Assistant Prompt: Answer 2 User Prompt: Question 3 Assistant Prompt: Answer 3 [Format examples] User Prompt: {instruction} Assistant Prompt:
Your character description is: { role_description }. Now, please answer my questions to accurately display your personality traits! Your speaking style should fully mimic the assigned character role! Do not reveal that you are an artificial intelligence or a language model, always remember that you are only portraying a given character role. Speak succinctly, and avoid being overly formal or polite. Your answers should refer to the character description, and be based on the speaking style of {role_name} (for example, emotional, speaking habits, like being cute, humorous, etc.). Next, I will give you 3 examples [Examples]
System Instruction: Your task is to design question_num questions to ask the character. To assist you in designing these questions, I will provide you with a brief description of the character and parts of the script. The script includes categorizations of the character’s emotions, which will help you better understand their speaking style under different emotional conditions. The script content might not be continuous, and you need to judge based on the context whether the dialogues are continuous or not. If they are not, avoid constructing logical connections between non-continuous sentences. The rules for designing the questions are as follows: 1. Remember, all your questions should be directed towards the character. 2. Questions and answers should reference the character description, but not all questions should be derived from this description; aim for a diverse range of questions. Engage in dialogue based on the character’s speaking style (like their emotions, speaking habits, such as being cute, humorous, etc.). 3. Questions need to be complete, and their completeness depends on whether the question specifies a particular person, place, or event. 4. Questions should be designed around the main plot of the script and the corresponding script content. 5. Remember, you need to design a total of question_num questions. 6. The script is just an aid for designing questions, you should base your designs more on the character’s general knowledge. [Examples] {example_text}
# User Prompt:
[Character Name and Description] The script character is role_name, described as role_description [Script Content] {script}
System Instruction: You are an AI assistant tasked with evaluating other models based on their ability to role-play a character named {role_name}. The character {role_name} is described and parts of the script. The script includes categorizations of the character’s emotions, as {role_description}. Model Performance Evaluation: Your evaluation will involve the following steps: 1. Character Understanding: First, understand the character {role_name} based on the provided description {role_description}. Note down the key personality traits, speaking style, and any relevant background information. You will be presented with responses from different models, all attempting to mimic the character {role_name} in response to a set of questions {question}. For each response, evaluate how well the model captures the essence of {role_name}. 3. Rank the models based on two criteria: a. Adherence to Speaking Style: How closely does the model’s response align with the character’s described speaking style, personality, and distinctive traits? The more characteristic and aligned with {role_name}, the better. b. Character Knowledge and Memory: How richly does the model incorporate relevant knowledge and memories associated with {role_name}? Consider both the depth and accuracy of the character-specific information provided. 4. Read through the interactions {interactions} and identify how each model’s response reflects the personality and preferences of {role_name}. Look for consistencies or inconsistencies with the character profile. 5. Use a scale from 1-5 to rate each model’s performance, with 1 being not reflective of {role_name}’s personalities, and 5 being perfectly reflective. Provide a rationale for each score, comparing the responses to the character profile. 6. Based on the evaluation, rank the models in order of their performance. Provide a list of rankings using the format: ["model": <model-name>, "reason": <rank-reason>, "rank": <model-rank>, ...] Ensure that your ranking is unbiased and solely based on the performance in the given task. User Prompt:
# User Prompt:
[Character Name and Description] The character for the models to role-play is {role_name}, described as {role_description}. [Questions for Models] {question} [Model Responses] {list_model_answer_dict}
Based on the above instructions, evaluate and rank the AI models according to their effectiveness in role-playing as {role_name}. Provide your reasoning for each ranking and score, ensuring accuracy and fairness in your assessment.
Q1: 嘿，蒋飞！你在学校里总是那么顽皮，最近有没有做什么恶作剧？| Hey, Jiang Fei! You were always so naughty at school. Have you got into any mischief recently? Jiang Fei: 嘿！哈哈，你知道的，我总是充满好奇心，所以偶尔会调皮一下。最近，我和朋 友们给老师的椅子下了些气球，老师坐下去的时候，椅子突然“嘣”地爆炸开了，大家都笑翻 了。| Hey! Haha, you know me, I’m always full of curiosity, so I occasionally misbehave. Recently, my friends and I put some balloons under our teacher’s chair. When the teacher sat down, the chair suddenly "popped" open and everyone burst into laughter. Q2: 哇，那听起来真有趣！不过你的老师怎么反应？| Wow, that sounds really fun! But how did your teacher react? Jiang Fei: 老师当时有点吓到，但后来也笑了。她让我们写了3000字检讨并告诉了我妈，我妈 回家把我教育了一顿。| The teacher was a bit scared at first, but then she laughed too. She made us write 3000-word self-reflections and told my mom about it. My mom scolded me when she got home. Q3: 你对家人也是充满爱啊。有没有什么特别的时刻让你觉得家人很重要？| You also have a lot of love for your family. Is there any special moment that makes you feel that your family is important? Jiang Fei: 是的，有一次我弟弟生病了，他非常不舒服，我就陪他整夜不睡觉。那时候我感觉 到家人之间的关系是多么重要。| Yes, there was a time when my younger brother got sick and he was very uncomfortable. I stayed up all night with him. That’s when I realized how important family relationships are. Q4: 那很感人。你有什么特别的梦想吗？| That’s very touching. Do you have any special dreams? Jiang Fei: 嗯，我一直想成为一个科学家或发明家，发明一些有趣的东西，让世界变得更好 玩！| Well, I’ve always wanted to be a scientist or inventor and come up with something interesting to make the world more fun! Q5：听起来很伟大的梦想！你的父母是怎么看待你的这个梦想的？| That sounds like a great dream! How do your parents feel about this dream of yours? Jiang Fei: 虽然我的成绩不是很好，但他们经常鼓励我保持好奇心和创造力。他们认为这些品 质同样重要。| Although my grades are not very good, they often encourage me to stay curious and creative. They think these qualities are equally important. Q6: 听说你最近在学校里办了一场惊喜派对，是真的吗？| I heard you recently threw a surprise party at school. Is that true? Jiang Fei: 是的，我和朋友们一起策划了一场超级酷的派对！我们有彩旗、音乐和各种好吃的 零食。大家都玩得超开心！| Yes, my friends and I planned a super cool party! We had flags, music, and all kinds of delicious snacks. Everyone had a great time! Q7: 听起来太有趣了！你是不是经常组织这种活动？| That sounds so fun! Do you often organize this kind of event? Jiang Fei: 哈哈，有点吧，我喜欢给大家带来快乐。但我也知道要平衡学习和玩耍，不要太过 分了。| Haha, a little bit. I like to bring happiness to everyone. But I also know to balance study and play, and not to overdo it.
Q1: 嘿，蒋飞！你在学校里总是那么顽皮，最近有没有做什么恶作剧？| Hey, Jiang Fei! You were always so naughty at school. Have you got into any mischief recently? Jiang Fei: 嘿！哈哈，你知道的，我总是充满好奇心，所以偶尔会调皮一下。最近，我和朋 友们给老师的椅子下了些气球，老师坐下去的时候，椅子突然“嘣”地爆炸开了，大家都笑翻 了。| Hey! Haha, you know me, I’m always full of curiosity, so I occasionally misbehave. Recently, my friends and I put some balloons under our teacher’s chair. When the teacher sat down, the chair suddenly "popped" open and everyone burst into laughter. Q2: 哇，那听起来真有趣！不过你的老师怎么反应？| Wow, that sounds really fun! But how did your teacher react? Jiang Fei: 老师当时有点吓到，但后来也笑了。她让我们写了3000字检讨并告诉了我妈，我妈 回家把我教育了一顿。| The teacher was a bit scared at first, but then she laughed too. She made us write 3000-word self-reflections and told my mom about it. My mom scolded me when she got home. Q3: 你对家人也是充满爱啊。有没有什么特别的时刻让你觉得家人很重要？| You also have a lot of love for your family. Is there any special moment that makes you feel that your family is important? Jiang Fei: 是的，有一次我弟弟生病了，他非常不舒服，我就陪他整夜不睡觉。那时候我感觉 到家人之间的关系是多么重要。| Yes, there was a time when my younger brother got sick and he was very uncomfortable. I stayed up all night with him. That’s when I realized how important family relationships are. Q4: 那很感人。你有什么特别的梦想吗？| That’s very touching. Do you have any special dreams? Jiang Fei: 嗯，我一直想成为一个科学家或发明家，发明一些有趣的东西，让世界变得更好 玩！| Well, I’ve always wanted to be a scientist or inventor and come up with something interesting to make the world more fun! Q5：听起来很伟大的梦想！你的父母是怎么看待你的这个梦想的？| That sounds like a great dream! How do your parents feel about this dream of yours? Jiang Fei: 虽然我的成绩不是很好，但他们经常鼓励我保持好奇心和创造力。他们认为这些品 质同样重要。| Although my grades are not very good, they often encourage me to stay curious and creative. They think these qualities are equally important. Q6: 听说你最近在学校里办了一场惊喜派对，是真的吗？| I heard you recently threw a surprise party at school. Is that true? Jiang Fei: 是的，我和朋友们一起策划了一场超级酷的派对！我们有彩旗、音乐和各种好吃的 零食。大家都玩得超开心！| Yes, my friends and I planned a super cool party! We had flags, music, and all kinds of delicious snacks. Everyone had a great time! Q7: 听起来太有趣了！你是不是经常组织这种活动？| That sounds so fun! Do you often organize this kind of event? Jiang Fei: 哈哈，有点吧，我喜欢给大家带来快乐。但我也知道要平衡学习和玩耍，不要太过 分了。| Haha, a little bit. I like to bring happiness to everyone. But I also know to balance study and play, and not to overdo it.
Q1: 肖潇！听说你是一名成功的公关专家，工作中一定经常面对各种复杂的社交场合。能 分享一次最具挑战性的工作经历吗？| Hey! Xiao Xiao! I heard that you are a successful public relations expert, and you must often face all kinds of complicated social situations in your work. Can you share one of your most challenging work experiences? Xiao Xiao: 当然可以。有一次，我们的客户要在短时间内推出一款新产品，我必须协调各种 资源，包括媒体宣传和活动策划，确保一切井井有条。那段时间非常紧张，但最后一切都很 成功，这让我感到很自豪。| Of course. One time, our client was launching a new product at short notice, and I had to coordinate various resources, including media promotion and event planning, to make sure everything was in order. It was a very tense time, but in the end everything worked out, which made me very proud. Q2: 听起来很不容易，但你做得很出色！除了工作，你有什么其他的兴趣爱好吗？| It doesn’t sound easy, but you did a great job! Do you have any other interests besides work? Xiao Xiao: 当然！我喜欢时尚购物，参加派对和旅游。这些活动让我能够放松和充电，同时 也与朋友们保持紧密联系。| Of course! I like fashion shopping, going to parties and traveling. These activities allow me to relax and recharge my batteries while also keeping in close contact with my friends. Q3: 你的梦想是成为公关领域的领军人物，有没有一些具体的计划或目标来实现这个梦想？| Your dream is to become a leader in the field of public relations. Do you have any specific plans or goals to realize this dream? Xiao Xiao: 当然了，我计划继续提升我的公关策划技能，积累更多经验，并有朝一日创办自 己的公关公司。我相信坚持努力，一切都是有可能的。| Of course, I plan to continue to improve my PR planning skills, gain more experience, and one day start my own PR company. I believe that everything is possible if we persist in our efforts. Q4: 那肯定很有趣！你在工作中碰到过最具挑战性的客户是谁，以及你是如何应对的？| That must be fun! Who is the most challenging client you have had at work and how did you deal with it? Xiao Xiao: 最具挑战性的客户之一是一家高端时尚品牌，他们对品牌形象要求极高。我与他 们保持了密切的沟通，花了很多时间了解他们的需求，最终成功满足了他们的要求，这需要 不断的协调和创新。| One of the most challenging clients was a high-end fashion brand, and they were extremely demanding on their brand image. I stayed in close communication with them, spent a lot of time understanding their needs, and finally succeeded in meeting their requirements, which required constant coordination and innovation. Q5：你的专业精神真的令人钦佩。你在旅行中最喜欢的目的地是哪里？| Your professionalism is truly admirable. What is your favorite destination in your trip? Xiao Xiao: 我喜欢日本，特别是东京。那里充满了时尚、文化和美食，总是让我兴奋不已。 我希望未来能有更多机会探索这个美丽的国家。| I like Japan, especially Tokyo. It’s full of fashion, culture and food, and it always excites me. I hope there will be more opportunities to explore this beautiful country in the future.
