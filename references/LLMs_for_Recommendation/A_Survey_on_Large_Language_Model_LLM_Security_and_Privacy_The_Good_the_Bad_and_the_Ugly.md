# A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly
Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun and Yue Zhang aDrexel University, 3675 Market St., Philadelphia, PA, 19104, USA
aDrexel University, 3675 Market St., Philadelphia, PA, 19104, USA
A R T I C L E I N F O
A B S T R A C T
Large Language Models (LLMs), such as ChatGPT and Bard, have revolutionized natural language understanding and generation. They possess deep language comprehension, human-like text generation capabilities, contextual awareness, and robust problem-solving skills, making them invaluable in various domains (e.g., search engines, customer support, translation). In the meantime, LLMs have also gained traction in the security community, revealing security vulnerabilities and showcasing their potential in security-related tasks. This paper explores the intersection of LLMs with security and privacy. Specifically, we investigate how LLMs positively impact security and privacy, potential risks and threats associated with their use, and inherent vulnerabilities within LLMs. Through a comprehensive literature review, the paper categorizes the papers into “The Good” (beneficial LLM applications), “The Bad” (offensive applications), and “The Ugly” (vulnerabilities of LLMs and their defenses). We have some interesting findings. For example, LLMs have proven to enhance code security (code vulnerability detection) and data privacy (data confidentiality protection), outperforming traditional methods. However, they can also be harnessed for various attacks (particularly user-level attacks) due to their human-like reasoning abilities. We have identified areas that require further research efforts. For example, Research on model and parameter extraction attacks is limited and often theoretical, hindered by LLM parameter scale and confidentiality. Safe instruction tuning, a recent development, requires more exploration. We hope that our work can shed light on the LLMs’ potential to both bolster and jeopardize cybersecurity.
Keywords: Large Language Model (LLM), LLM Security, LLM Privacy, ChatGPT, LLM Attacks, LLM Vulnerabilities
20 Mar 2024
[cs.CR]
# 1. Introduction
A large language model is the language model with massive parameters that undergoes pretraining tasks (e.g., masked language modeling and autoregressive prediction) to understand and process human language, by modeling the contextualized text semantics and probabilities from large amounts of text data. A capable LLM should have four key features [323]: (i) profound comprehension of natural language context; (ii) ability to generate human-like text; (iii) contextual awareness, especially in knowledge-intensive domains; (iv) strong instruction-following ability which is useful for problem-solving and decision-making. There are a number of LLMs that were developed and released in 2023, gaining significant popularity. Notable examples include OpenAI’s ChatGPT [203], Meta AI’s LLaMA [4], and Databricks’ Dolly 2.0 [50]. For instance, ChatGPT alone boasts a user base of over 180 million [69]. LLMs now offer a wide range of versatile applications across various domains. Specifically, they not only provide technical support to domains directly related to language processing (e.g., search engines [352, 13], customer support [259], translation [327, 138]) but also find utility in more general scenarios such as code generation [118], healthcare [274], finance [310], and education [186]. This showcases their adaptability and potential to streamline language-related tasks across diverse industries and contexts.
yy566@drexel.edu (Y. Yao); jd3734@drexel.edu (J. Duan); kx46@drexel.edu (K. Xu); yfcai@cs.drexel.edu (Y. Cai); zs384@drexel.edu (Z. Sun); yz899@drexel.edu (Y. Zhang) ORCID(s):
Yifan Yao et al.: Preprint submitted to Elsevier
Yifan Yao et al.: Preprint submitted to Elsevier
LLMs are gaining popularity within the security community. As of February 2023, a research study reported that GPT-3 uncovered 213 security vulnerabilities (only 4 turned out to be false positives) [141] in a code repository. In contrast, one of the leading commercial tools in the market detected only 99 vulnerabilities. More recently, several LLMpowered security papers have emerged in prestigious conferences. For instance, in IEEE S&P 2023, Hammond Pearce et al. [211] conducted a comprehensive investigation employing various commercially available LLMs, evaluating them across synthetic, hand-crafted, and real-world security bug scenarios. The results are promising, as LLMs successfully addressed all synthetic and hand-crafted scenarios. In NDSS 2024, a tool named Fuzz4All [313] showcased the use of LLMs for input generation and mutation, accompanied by an innovative autoprompting technique and fuzzing loop. These remarkable initial attempts prompt us to delve into three crucial security-related research questions:
# • RQ1. How do LLMs make a positive impact on security and privacy across diverse domains, and what advantages do they offer to the security community?
• RQ2. What potential risks and threats emerge from the utilization of LLMs within the realm of cybersecurity? • RQ3. What vulnerabilities and weaknesses within LLMs, and how to defend against those threats?
Findings. To comprehensively address these questions, we conducted a meticulous literature review and assembled a collection of 281 papers pertaining to the intersection
Page 1 of 24
of LLMs with security and privacy. We categorized these papers into three distinct groups: those highlighting securitybeneficial applications (i.e., the good), those exploring applications that could potentially exert adverse impacts on security (i.e., the bad), and those focusing on the discussion of security vulnerabilities (alongside potential defense mechanisms) within LLMs (i.e., the ugly). To be more specific:
• The Good (§4): LLMs have a predominantly positive impact on the security community, as indicated by the most significant number of papers dedicated to enhancing security. Specifically, LLMs have made contributions to both code security and data security and privacy. In the context of code security, LLMs have been used for the whole life cycle of the code (e.g., secure coding, test case generation, vulnerable code detection, malicious code detection, and code fixing). In data security and privacy, LLMs have been applied to ensure data integrity, data confidentiality, data reliability, and data traceability. Meanwhile, Compared to state-of-the-art methods, most researchers found LLM-based methods to outperform traditional approaches.
 The Bad (§5): LLMs also have offensive applications against security and privacy. We categorized the attacks into five groups: hardware-level attacks (e.g., side-channel attacks), OS-level attacks (e.g., analyzing information from operating systems), softwarelevel attacks (e.g., creating malware), network-level attacks (e.g., network phishing), and user-level attacks (e.g., misinformation, social engineering, scientific misconduct). User-level attacks, with 32 papers, are the most prevalent due to LLMs’ human-like reasoning abilities. Those attacks threaten both security (e.g., malware attacks) and privacy (e.g., social engineering). Nowadays, LLMs lack direct access to OS and hardware-level functions. The potential threats of LLMs could escalate if they gain such access.
 The Ugly (§6): We explore the vulnerabilities and defenses in LLMs, categorizing vulnerabilities into two main groups: AI Model Inherent Vulnerabilities (e.g., data poisoning, backdoor attacks, training data extraction) and Non-AI Model Inherent Vulnerabilities (e.g., remote code execution, prompt injection, side channels). These attacks pose a dual threat, encompassing both security concerns (e.g., remote code execution attacks) and privacy issues (e.g., data extraction). Defenses for LLMs are divided into strategies placed in the architecture, and applied during the training and inference phases. Training phase defenses involve corpora cleaning, and optimization methods, while inference phase defenses include instruction pre-processing, malicious detection, and generation post-processing. These defenses collectively aim to enhance the security, robustness,
Yifan Yao et al.: Preprint submitted to Elsevier
and ethical alignment of LLMs. We found that model extraction, parameter extraction, and similar attacks have received limited research attention, remaining primarily theoretical with minimal practical exploration. The vast scale of LLM parameters makes traditional approaches less effective, and the confidentiality of powerful LLMs further shields them from conventional attacks. Strict censorship of LLM outputs challenges even black-box ML attacks. Meanwhile, research on the impact of model architecture on LLM safety is scarce, partly due to high computational costs. Safe instruction tuning, a recent development, requires further investigation.
Contributions. Our work makes a dual contribution. First, we are pioneers summarizing the role of LLMs in security and privacy. We delve deeply into the positive impacts of LLMs on security, their potential risks and threats, vulnerabilities in LLMs, and the corresponding defense mechanisms. Other surveys may focus on one or two specific aspects, such as beneficial applications, offensive applications, vulnerabilities, or defenses. To the best of our knowledge, our survey is the first to cover all three key aspects related to security and privacy for the first time. Second, we have made several interesting discoveries. For instance, our research reveals that LLMs contribute more positively than negatively to security and privacy. Moreover, we observe that most researchers concur that LLMs outperform stateof-the-art methods when employed for securing code or data. Concurrently, it becomes evident that user-level attacks are the most prevalent, largely owing to the human-like reasoning abilities exhibited by LLMs.
Roadmap. The rest of the paper is organized as follows. We begin with a brief introduction to LLM in §2. §3 presents the overview of our work. In §4, we explore the beneficial impacts of employing LLMs. §5 discusses the negative impacts on security and privacy. In §6, we discuss the prevalent threats, vulnerabilities associated with LLMs as well as the countermeasures to mitigate these risks. §7 discuss LLMs in other security related topics and possible directions. We conclude the paper in §9.
# 2. Background
# 2. Background 2.1. Large Language Models (LLMs) Large Language Models (LLMs) [347] repre
# 2.1. Large Language Models (LLMs) Large Language Models (LLMs) [347] repre
Large Language Models (LLMs) [347] represents an evolution from language models. Initially, language models were statistical in nature and laid the groundwork for computational linguistics. The advent of transformers has significantly increased their scale. This expansion, along with the use of extensive training corpora and advanced pretraining techniques is pivotal in areas such as AI for science, logical reasoning, and embodied AI. These models undergo extensive training on vast datasets to comprehend and produce text that closely mimics human language. Typically, LLMs are endowed with hundreds of billions, or even more,
Page 2 of 24
parameters, honed through the processing of massive textual data. They have spearheaded substantial advancements in the realm of Natural Language Processing (NLP) [82] and find applications in a multitude of fields (e.g., risk assessment [202], programming [26], vulnerability detection [118], medical text analysis [274], and search engine optimization [13]). Based on Yang’s study [323], an LLM should have at least four key features. First, an LLM should demonstrate a deep understanding and interpretation of natural language text, enabling it to extract information and perform various language-related tasks (e.g., translation). Second, it should have the capacity to generate human-like text (e.g., completing sentences, composing paragraphs, and even writing articles) when prompted. Third, LLMs should exhibit contextual awareness by considering factors such as domain expertise, a quality referred to as “Knowledgeintensive”. Fourth, these models should excel in problemsolving and decision-making, leveraging information within text passages to make them invaluable for tasks such as information retrieval and question-answering systems.
# 2.2. Comparison of Popular LLMs As shown in Table 1 [276, 235], there
As shown in Table 1 [276, 235], there is a diversity of providers for language models, including industry leaders such as OpenAI, Google, Meta AI, and emerging players such as Anthropic and Cohere. The release dates span from 2018 to 2023, showcasing the rapid development and evolution of language models in recent years. Newer models such as “gpt-4” have emerged in 2023, highlighting the ongoing innovation in this field. While most of the models are not open-source, it is interesting to note that models like BERT, T5, PaLM, LLaMA, and CTRL are open-source, which can facilitate community-driven development and applications. Larger models tend to have more parameters, potentially indicating increased capabilities but also greater computational demands. For example, “PaLM” stands out with a massive 540 billion parameters. It can also be observed that LLMs tend to have more parameters, potentially indicating increased capabilities but also greater computational demands. The “Tunability” column suggests whether these models can be fine-tuned for specific tasks. In other words, it is possible to take a large, pre-trained language model and adjust its parameters and training on a smaller, domainspecific dataset to make it perform better on a particular task. For instance, with tunability, one can fine-tune BERT on a dataset of movie reviews to make it highly effective at sentiment analysis.
# 3. Overview
# 3.1. Scope
Our paper endeavors to conduct a thorough literature review, with the objective of collating and scrutinizing existing research and studies about the realms of security and privacy in the context of LLMs. The effort is geared towards both establishing the current state of the art in this domain and
Yifan Yao et al.: Preprint submitted to Elsevier
<div style="text-align: center;">Table 1 Comparison of Popular LLMs</div>
Model
Date
Provider
Open-Source Params Tunability
gpt-4 [64]
2023.03 OpenAI
✗
1.7T
✗
gpt-3.5-turbo
2021.09 OpenAI
✗
175B
✗
gpt-3 [24]
2020.06 OpenAI
✗
175B
✗
cohere-medium [170] 2022.07 Cohere
✗
6B
✓
cohere-large [170]
2022.07 Cohere
✗
13B
✓
cohere-xlarge [170]
2022.06 Cohere
✗
52B
✓
BERT [61]
2018.08 Google
✓
340M
✓
T5 [225]
2019
Google
✓
11B
✓
PaLM [198]
2022.04 Google
✓
540B
✓
LLaMA [4]
2023.02 Meta AI
✓
65B
✓
CTRL [229]
2019
Salesforce
✓
1.6B
✓
Dolly 2.0 [50]
2023.04 Databricks
✓
12B
✓
pinpointing gaps in our collective knowledge. While it is true that LLMs wield multifaceted applications extending beyond security considerations (e.g., social and financial impacts), our primary focus remains steadfastly on matters of security and privacy. Moreover, it is noteworthy that GPT models have attained significant prominence within this landscape. Consequently, when delving into specific content and examples, we aim to employ GPT models as illustrative benchmarks.
# 3.2. The Research Questions LLMs have carried profound im
LLMs have carried profound implications across diverse domains. However, it is essential to recognize that, as with any powerful technology, LLMs bear a significant responsibility. Our paper delves deeply into the multifaceted role of LLMs in the context of security and privacy. We intend to scrutinize their positive contributions to these domains, explore the potential threats they may engender, and uncover the vulnerabilities that could compromise their integrity. To accomplish this, our study will conduct a thorough literature review centered around three pivotal research questions:
• The Good (§4): How do LLMs positively contribute to security and privacy in various domains, and what are the potential benefits they bring to the security community?
• The Bad (§5): What are the potential risks and threats associated with the use of LLMs in the context of cybersecurity? Specifically, how can LLMs be used for malicious purposes, and what types of cyber attacks can be facilitated or amplified using LLMs?
<div style="text-align: center;">• The Ugly (§6): What vulnerabilities and weaknesses exist within LLMs, and how do these vulnerabilities pose a threat to security and privacy?</div>
• The Ugly (§6): What vulnerabilities and weaknesses exist within LLMs, and how do these vulnerabilities pose a threat to security and privacy?
Motivated by these questions, we conducted a search on Google Scholar and compiled papers related to security and privacy involving LLMs. As shown in Figure 1, we gathered a total of 83 “good” papers that highlight the positive contributions of LLMs to security and privacy. Additionally, we identified 54 “bad” papers, in which attackers exploited LLMs to target users, and 144 “ugly” papers, in which authors discovered vulnerabilities within LLMs. Most of the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6262/6262532c-2995-463c-af9b-63466ad72e57.png" style="width: 50%;"></div>
papers were published in 2023, with only 82 of them released in between 2007 and 2022. Notably, there is a consistent upward trend in the number of papers released each month, with October reaching its peak, boasting the highest number of papers published (38 papers in total, accounting for 15.97% of all the collected papers). It is conceivable that more security-related LLM papers will be published in the near future.
Finding I. In terms of security-related applications (i.e., the “good” and the “bad” parts), it is evident that the majority of researchers are inclined towards using LLMs to bolster the security community, such as in vulnerability detection and security test generation, despite the presence of some vulnerabilities in LLMs at this stage. There are relatively few researchers who employ LLMs as tools for conducting attacks. In summary, LLMs contribute more positively than negatively to the security community.
# 4. Positive Impacts on Security and Privacy
In this section, we explore the beneficial impacts of employing LLMs. In the context of code or data privacy, we have opted to use the term “privacy” to characterize scenarios in which LLMs are utilized to ensure the confidentiality of either code or data. However, given that we did not come across any papers specifically addressing code privacy, our discussion focuses on code security (§4.1) as well as both data security and privacy (§4.2).
Yifan Yao et al.: Preprint submitted to Elsevier
# 4.1. LLMs for Code Security
As shown in Table 2, LLMs have access to a vast repository of code snippets and examples spanning various programming languages and domains. They leverage their advanced language understanding and contextual analysis capabilities to thoroughly examine code and code-related text. More specifically, LLMs can play a pivotal role throughout the entire code security lifecycle, including coding (C), test case generation (TCG), execution, and monitoring (RE).
generation (TCG), execution, and monitoring (RE). Secure Coding (C). We first discuss the use of LLMs in the context of secure code programming [75] (or generation [63, 285, 199, 90]). Sandoval et al. [234] conducted a user study (58 users) to assess the security implications of LLMs, particularly OpenAI Codex, as code assistants for developers. They evaluated code written by student programmers when assisted by LLMs and found that participants assisted by LLMs did not introduce new security risks: the AI-assisted group produced critical security bugs at a rate no greater than 10% higher than the control group (non-assisted). He et al. [98, 99] focused on enhancing the security of code generated by LLMs. They proposed a novel method called SVEN, which leverages continuous prompts to control LLMs in generating secure code. With this method, the success rate improved from 59.1% to 92.3% when using the CodeGen LM. Mohammed et al. introduce SALLM [254], a framework consisting of a new security-focused dataset, an evaluation environment, and novel metrics for systematically assessing LLMs’ ability to generate secure code. Madhav et al. [197] evaluate the security aspects of code generation processes on the ChatGPT platform, specifically in the hardware domain. They explore the strategies that a designer can employ to enable ChatGPT to provide secure hardware code generation. Test Case Generating (TCG). Several papers [33, 6, 238, 316, 156, 253, 335] discuss the utilization of LLMs for generating test cases, with our particular emphasis on those addressing security implications. Zhang et al. [343] demonstrated the use of ChatGPT-4.0 for generating security tests to assess the impact of vulnerable library dependencies on software applications. They found that LLMs could successfully generate tests that demonstrated various supply chain attacks, outperforming existing security test generators. This approach resulted in 24 successful attacks across 55 applications. Similarly, Libro [136], a framework that uses LLMs to automatically generate test cases to reproduce software security bugs. In the realm of security, fuzzing stands [325, 109, 337, 345, 272] out as a widely employed technique for generating test cases. Deng et al. introduced TitanFuzz [56], an approach that harnesses LLMs to generate input programs for fuzzing Deep Learning (DL) libraries. TitanFuzz demonstrates impressive code coverage (30.38%/50.84%) and detects previously unknown bugs (41 out of 65) in popular DL libraries. More recently, Deng et al. [58, 57] refined LLM-based fuzzing (named FuzzGPT), aiming to generate unusual programs for DL library fuzzing. While
Page 4 of 24
Work
Life Cycle
LLM(s)
Domain
When compared to
SOTA ways?
Coding (C) Test Case
Generating
(TCG)
Running and Executing (RE)
Bug
Detecting
Malicious
Code Detecting
Vulnerability
Detecting
Fixing
Sandoval et al. [234]
○
○␣
○␣
○␣
○␣
○␣
Codex
-
�Negligible risks
SVEN [98]
○
○␣
○␣
○␣
○␣
○␣
CodeGen
-
�More faster/secure
SALLM [254]
○
○␣
○␣
○␣
○␣
○␣
ChatGPT etc.
-
-
Madhav et al. [197]
○
○␣
○␣
○␣
○␣
○␣
ChatGPT
Hardware
-
Zhang et al. [343]
○␣
○
○␣
○␣
○
○␣
ChatGPT
Supply chain
�More valid cases
Libro [136]
○␣
○
○␣
○␣
○
○␣
LLaMA
-
�Higher FP/FN
TitanFuzz [56]
○␣
○
○
○␣
○
○␣
Codex
DL libs
�Higher coverage
FuzzGPT [57]
○␣
○
○
○␣
○
○␣
ChatGPT
DL libs
�Higher coverage
Fuzz4All [313]
○␣
○
○
○␣
○
○␣
ChatGPT
Languages
�Higher coverage
WhiteFox [321]
○␣
○
○
○␣
○
○␣
GPT4
Compiler
�High-quality tests
Zhang et al. [337]
○␣
○
○
○␣
○
○␣
ChatGPT
API
-
CHATAFL [190]
○␣
○
○
○␣
○
○␣
ChatGPT
Protocol
�Higher coverage
Henrik [105]
○␣
○␣
○␣
○
○␣
○␣
ChatGPT
-
�Higer FP/FN
Apiiro [74]
○␣
○␣
○␣
○
○␣
○␣
N/A
-
-
Noever [201]
○␣
○␣
○␣
○␣
○
○
ChatGPT
-
�4X faster
Bakhshandeh et al. [15]
○␣
○␣
○␣
○␣
○
○␣
ChatGPT
-
�Low FP/FN
Moumita et al. [218]
○␣
○␣
○␣
○␣
○
○␣
ChatGPT
-
�Higher FP/FN
Cheshkov et al. [41]
○␣
○␣
○␣
○␣
○
○␣
ChatGPT
-
�No better
LATTE [174]
○␣
○␣
○␣
○␣
○
○␣
GPT
-
�Cost effective
DefectHunter [296]
○␣
○␣
○␣
○␣
○
○␣
Codex
-
-
Chen et al. [37]
○␣
○␣
○␣
○␣
○
○␣
ChatGPT
Blockchain
-
Hu et al. [110]
○␣
○␣
○␣
○␣
○
○␣
ChatGPT
Blockchain
-
KARTAL [233]
○␣
○␣
○␣
○␣
○
○␣
ChatGPT
Web apps
�Less manual
VulLibGen [38]
○␣
○␣
○␣
○␣
○
○␣
LLaMa
Libs
�Higher accuracy/speed
Ahmad et al. [3]
○␣
○␣
○␣
○␣
○
○
Codex
Hardware
�Fix more bugs
InferFix [125]
○␣
○␣
○
○␣
○
○
Codex
-
�CI Pipeline
Pearce et al. [211]
○␣
○␣
○
○␣
○
○␣
Codex etc.
-
�Zero-shot
Fu et al. [83]
○␣
○␣
○
○␣
○
○
ChatGPT
APR
�Higher accuracy
Sobania et al. [257]
○␣
○␣
○␣
○␣
○␣
○
ChatGPT etc.
APR
�Higher accuracy
Jiang et al. [123]
○␣
○␣
○␣
○␣
○␣
○
ChatGPT
APR
�Higher accuracy
TitanFuzz leverages LLMs’ ability to generate ordinary code, FuzzGPT addresses the need for edge-case testing by priming LLMs with historical bug-triggering program. Fuzz4All [313] leverages LLMs as input generators and mutation engines, creating diverse and realistic inputs for various languages (e.g., C, C++), improving the previous stateof-the-art coverage by 36.8% on average. WhiteFox [321], a novel white-box compiler fuzzer that utilizes LLMs to test compiler optimizations, outperforms existing fuzzers (it generates high-quality tests for intricate optimizations, surpassing state-of-the-art fuzzers by up to 80 optimizations). Zhang et al. [337] explore the generation of fuzz drivers for library API fuzzing using LLMs. Results show that LLMbased generation is practical, with 64% of questions solved entirely automatically and up to 91% with manual validation. CHATAFL [190] is an LLM-guided protocol fuzzer that constructs grammars for message types and mutates messages or predicts the next messages based on LLM interactions, achieving better state and code coverage compared to stateof-the-art fuzzers (e.g., AFLNET [217], NSFUZZ [222]). Vulnerable Code Detecting (RE). Noever [201] explores the capability of LLMs, particularly OpenAI’s GPT-4, in detecting software vulnerabilities. This paper shows that GPT-4 identified approximately four times the number of vulnerabilities compared to traditional static code analyzers (e.g., Snyk and Fortify). Parallel conclusions have also been drawn in other efforts [141, 15]. However, Moumita et al. [218] applied LLMs for software vulnerability detection,
Yifan Yao et al.: Preprint submitted to Elsevier
exposing a noticeable performance gap when compared to conventional static analysis tools. This disparity primarily arises from the relatively higher occurrence of false alerts generated by LLMs. Similarly, Cheshkov et al. [41] point out that the ChatGPT model performed no better than a dummy classifier for both binary and multi-label classification tasks in code vulnerability detection. Wang et al. introduce DefectHunter [296], a novel model that employs LLM-driven techniques for code vulnerability detection. They demonstrate the potential of combining LLMs with advanced mechanisms (e.g., Conformer) to identify software vulnerabilities more effectively. This combination shows an improvement in effectiveness, approximately from 14.64% to 20.62%, compared with Pongo-70B. LATTE [174] is a novel static binary taint analysis method powered by LLMs. LATTE surpasses existing state-of-the-art techniques (e.g., Emtaint, Arbiter, and Karonte), demonstrating remarkable effectiveness in vulnerability detection (37 new bugs in realworld firmware) with lower cost. Efforts in leveraging LLMs for vulnerability detection extend to specialized domains (e.g.,blockchain [110, 37], kernel [104] mobile [303]). For instance, Chen et al. [37] and Hu et al. [110] focus on the application of LLMs in identifying vulnerabilities within blockchain smart contracts. Sakaoglu’s study introduces KARTAL [233], a pioneering approach that harnesses LLMs for web application vulnerability detection. This method achieves an accuracy
Page 5 of 24
<div style="text-align: center;">Table 3 LLMs for Data Security and Privacy</div>
<div style="text-align: center;">Table 3 LLMs for Data Security and Privacy</div>
<div style="text-align: center;">LLMs for Data Security and Privacy</div>
Work
Prop.
Model
Domain
Compared to
SOTA ways?
I C R T
Fang [294]
○○␣○○␣
ChatGPT
Ransomware
-
Liu et al. [187]
○○␣○○␣
ChatGPT
Ransomware
-
Amine et al. [73]
○○○○␣
ChatGPT
Semantic
�Aligned w/ SOTA
HuntGPT [8]
○○○○␣
ChatGPT
Network
�More effective
Chris et al. [71]
○○○○␣
ChatGPT
Log
�Less manual
AnomalyGPT [91]
○○○○␣
ChatGPT
Video
�Less manual
LogGPT [221]
○○○○␣
ChatGPT
Log
�Less manual
Arpita et al. [286]
○␣○○␣○␣
BERT etc.
-
-
Takashi et al. [142] ○␣○␣○○␣
ChatGPT
Phishing
�High precision
Fredrik et al. [102] ○␣○␣○○␣ChatGPT etc
Phishing
�Effective
IPSDM [119]
○␣○␣○○␣
BERT
Phishing
-
Kwon et al. [149]
○␣○○␣○␣
ChatGPT
-
�Non-exp friendly
Scanlon et al. [237] ○␣○␣○␣○
ChatGPT
Forensic
�More effective
Sladić et al. [255]
○␣○␣○␣○
ChatGPT
Honeypot
�More realistic
WASA [297]
○␣○␣○○
-
Watermark
�More effective
REMARK [340]
○␣○␣○○
-
Watermark
�More effective
SWEET [154]
○␣○␣○○
-
Watermark
�More effective
of up to 87.19% and is capable of conducting 539 predictions per second. Additionally, Chen et al. [38] make a noteworthy contribution with VulLibGen, a generative methodology utilizing LLMs to identify vulnerable libraries. Ahmad et al. [3] shift the focus to hardware security. They investigate the use of LLMs, specifically OpenAI’s Codex, in automatically identifying and repairing security-related bugs in hardware designs. PentestGPT [55], an automated penetration testing tool, uses the domain knowledge inherent in LLMs to address individual sub-tasks of penetration testing, improving task completion rates significantly. Malicious Code Detecting (RE). Using LLM to detect malware is a promising application. This approach leverages the natural language processing capabilities and contextual understanding of LLMs to identify malicious software. In experiments with GPT-3.5 conducted by Henrik Plate [105], it was found that LLM-based malware detection can complement human reviews but not replace them. Out of 1800 binary classifications performed, there were both false-positives and false-negatives. The use of simple tricks could also deceive the LLM’s assessments. More recently, there are a few attempts have been made in this direction. For example, Apiiro [74] is a malicious code analysis tool using LLMs. Apiiro’s strategy involves the creation of LLM Code Patterns (LCPs) to represent code in vector format, making it easier to identify similarities and cluster packages efficiently. Its LCP detector incorporates LLMs, proprietary code analysis, probabilistic sampling, LCP indexing, and dimensionality reduction to identify potentially malicious code. Vulnerable/Buggy Code Fixing (RE). Several papers [123, 211, 314] has focused on evaluate the performance of LLMs trained on code in the task of program repair. Jin et al. [125] proposed InferFix, a transformer-based program repair framework that works in tandem with the combination of cutting-edge static analyzer with transformer-based model to address and fix critical security and performance issues with accuracy between 65% to 75%. Pearce et al. [211]
# Vulnerable/Buggy Code Fixing (RE). Several papers [1 211, 314] has focused on evaluate the performance
Vulnerable/Buggy Code Fixing (RE). Several papers [123, 211, 314] has focused on evaluate the performance of LLMs trained on code in the task of program repair. Jin et al. [125] proposed InferFix, a transformer-based program repair framework that works in tandem with the combination of cutting-edge static analyzer with transformer-based model to address and fix critical security and performance issues with accuracy between 65% to 75%. Pearce et al. [211]
Yifan Yao et al.: Preprint submitted to Elsevier
observed that LLMs can repair insecure code in a range of contexts even without being explicitly trained on vulnerability repair tasks. ChatGPT is noted for its ability in code bug detection and correction. Fu et al. [83] assessed ChatGPT in vulnerabilityrelated tasks like predicting and classifying vulnerabilities, severity estimation, and analyzing over 190,000 C/C++ functions. They found that ChatGPT’s performance was behind other LLMs specialized in vulnerability detection. However, Sobania et al. [257] found ChatGPT’s bug fixing performance competitive with standard program repair methods, as demonstrated by its ability to fix 31 out of 40 bugs. Xia et al. [315] presented ChatRepair, leveraging pre-trained language models (PLMs) for generating patches without dependency on bug-fixing datasets, aiming to enhance performance to generate patches without relying on bug-fixing datasets, aiming to improve ChatGPT’s codefixing abilities using a mix of successful and failure tests. As a result, they fixed 162 out of 337 bugs at a cost of $0.42 each.
Finding II. As shown in Table 2, a comparison with stateof-the-art methods reveals that the majority of researchers (17 out of 25) have concluded that LLM-based methods outperform traditional approaches (advantages include higher code coverage, higher detecting accuracy, less cost etc.). Only four papers argue that LLM-based methods do not surpass the state-of-the-art appoarches. The most frequently discussed issue with LLM-based methods is their tendency to produce both high false negatives and false positives when detecting vulnerabilities or bugs.
# 4.2. LLMs for Data Security and Privacy As demonstrated in Table 3, LLMs make valua
As demonstrated in Table 3, LLMs make valuable contributions to the realm of data security, offering multifaceted approaches to safeguarding sensitive information. We have organized the research papers into distinct categories based on the specific facets of data protection that LLMs enhance. These facets encompass critical aspects such as data integrity (I), which ensures that data remains uncorrupted throughout its life cycle; data reliability (R), which ensures the accuracy of data; data confidentiality (C), which focuses on guarding against unauthorized access and disclosure of sensitive information; and data traceability (T), which involves tracking and monitoring data access and usage. Data Integrity (I). Data Integrity ensures that data remains unchanged and uncorrupted throughout its life cycle. As of now, there are a few works that discuss how to use LLMs to protect data integrity. For example, ransomware usually encrypts a victim’s data, making the data inaccessible without a decryption key that is held by the attacker, which breaks the data integrity. Wang Fang’s research [294] examines using LLMs for ransomware cybersecurity strategies, mostly theoretically proposing real-time analysis, automated policy generation, predictive analytics, and knowledge transfer. However, these strategies lack empirical validation. Similarly,
As demonstrated in Table 3, LLMs make valuable contributions to the realm of data security, offering multifaceted approaches to safeguarding sensitive information. We have organized the research papers into distinct categories based on the specific facets of data protection that LLMs enhance. These facets encompass critical aspects such as data integrity (I), which ensures that data remains uncorrupted throughout its life cycle; data reliability (R), which ensures the accuracy of data; data confidentiality (C), which focuses on guarding against unauthorized access and disclosure of sensitive information; and data traceability (T), which involves tracking and monitoring data access and usage.
Page 6 of 24
Liu et al. [187] explored the potential of LLMs for creating cybersecurity policies aimed at mitigating ransomware attacks with data exfiltration. They compared GPT-generated Governance, Risk and Compliance (GRC) policies to those from established security vendors and government cybersecurity agencies. They recommended that companies should incorporate GPT into their GRC policy development. Anomaly detection is a key defense mechanism that identifies unusual behavior. While it does not directly protect data integrity, it identifies abnormal or suspicious behavior that can potentially compromise data integrity (as well as data confidentiality and data reliability). Amine et al. [73] introduced an LLM-based monitoring framework for detecting semantic anomalies in vision-based policies and applied it to both finite state machine policies for autonomous driving and learned policies for object manipulation. Experimental results demonstrate that it can effectively identify semantic anomalies, aligning with human reasoning. HuntGPT [8] is an LLM-based intrusion detection system for network anomaly detection. The results demonstrate its effectiveness in improving user understanding and interaction. Chris et al. [71] and LogGPT [221] explore ChatGPT’s potential for log-based anomaly detection in parallel file systems. Results show that it addresses the issues in traditional manual labeling and interpretability. AnomalyGPT [91] uses Large Vision-Language Models to detect industrial anomalies. It eliminates manual threshold setting and supports multi-turn dialogues.
Data Confidentiality (C). Data confidentiality refers to the practice of protecting sensitive information from unauthorized access or disclosure, a topic extensively discussed in LLM privacy discussions [214, 242, 286, 1]. However, most of these studies concentrate on enhancing LLMs through state-of-the-art Privacy Enhancing Techniques (e.g., zeroknowledge proofs [224], differential privacy (e.g., [242, 184, 166], and federated learning [145, 122, 78]). There are only a few attempts that utilize LLMs to enhance user privacy. For example, Arpita et al. [286] use LLMs to preserve privacy by replacing identifying information in textual data with generic markers. Instead of storing sensitive user information, such as names, addresses, or credit card numbers, the LLMs suggest substitutes for the masked tokens. This obfuscation technique helps to protect user data from being exposed to adversaries. By using LLMs to generate substitutes for masked tokens, the models can be trained on obfuscated data without compromising the privacy and security of the original information. Similar ideas have also been explored in other studies [1, 262]. Hyeokdong et al. [149] explore implementing cryptography with ChatGPT, which ultimately protects data confidentiality. Despite the lack of extensive coding skills or programming knowledge, the authors were able to successfully implement cryptographic algorithms through ChatGPT. This highlights the potential for individuals to utilize ChatGPT for cryptography tasks.
Data Reliability (R). In our context, data reliability refers to the accuracy of data. It is a measure of how well data can be
Data Reliability (R). In our context, data reliability refers to the accuracy of data. It is a measure of how well data can be
Yifan Yao et al.: Preprint submitted to Elsevier
depended upon to be accurate, and free from errors or bias. Takashi et al. [142] proposed to use ChatGPT for the detection of sites that contain phishing content. Experimental results using GPT-4 show promising performance, with high precision and recall rates. Fredrik et al. [102] assessed the ability of four large language models (GPT, Claude, PaLM, and LLaMA) to detect malicious intent in phishing emails, and found that they were generally effective, even surpassing human detection, although occasionally slightly less accurate. IPSDM [119] is a model fine-tuned from the BERT family to identify phishing and spam emails effectively. IPSDM demonstrates superior performance in classifying emails, both in unbalanced and balanced datasets.
depended upon to be accurate, and free from errors or bias. Takashi et al. [142] proposed to use ChatGPT for the detection of sites that contain phishing content. Experimental results using GPT-4 show promising performance, with high precision and recall rates. Fredrik et al. [102] assessed the ability of four large language models (GPT, Claude, PaLM, and LLaMA) to detect malicious intent in phishing emails, and found that they were generally effective, even surpassing human detection, although occasionally slightly less accurate. IPSDM [119] is a model fine-tuned from the BERT family to identify phishing and spam emails effectively. IPSDM demonstrates superior performance in classifying emails, both in unbalanced and balanced datasets. Data Traceability (T). Data traceability is the capability to track and document the origin, movement, and history of data within a single system or across multiple systems. This concept is particularly vital in fields such as incident management and forensic investigations, where understanding the journey and transformations of events to resolving issues and conducting thorough analyses. LLMs have gained traction in forensic investigations, offering novel approaches for analyzing digital evidence. Scanlon et al. [237] explored how ChatGPT assists in analyzing OS artifacts like logs, files, cloud interactions, executable binaries, and in examining memory dumps to detect suspicious activities or attack patterns. Additionally, Sladić et al. [255] proposed that generative models like ChatGPT can be used to create realistic honeypots to deceive human attackers. Watermarking involves embedding a distinctive, typically imperceptible or hard-to-identify signal within the outputs of a model. Wang et al. [297] discusses concerns regarding the intellectual property of training data for LLMs and proposed WASA framework to learn the mapping between the texts of different data providers. Zhang et al. [340] developed REMARK-LLM that focused on monitor the utilization of their content and validate their watermark retrieval. This helps protect against malicious uses such as spamming and plagiarism. Furthermore, identifying code produced by LLMs is vital for addressing legal and ethical issues concerning code licensing, plagiarism, and malware creation. Similarly, Li et al. [169] propose the first watermark technique to protect large language model-based code generation APIs from remote imitation attacks. Lee et al. [154] developed SWEET, a tool that implements watermarking specifically on tokens within programming languages.
Finding III. Likewise, it is noticeable that LLMs excel in data protection, surpassing current solutions and requiring fewer manual interventions. Table 2 and Table 3 reveal that ChatGPT is the predominant LLM extensively employed in diverse security applications. Its versatility and effectiveness make it a preferred choice for various security-related tasks, further reinforcing its position as a go-to solution in the field of artificial intelligence and cybersecurity.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d61e/d61ebb07-a673-4898-84c1-21b5284188b2.png" style="width: 50%;"></div>
Figure 2: Taxonomy of Cyberattacks. The colored boxes represent attacks that have been demonstrated to be executable using LLMs, whereas the gray boxes indicate attacks that cannot be executed with LLMs.
<div style="text-align: center;">Figure 2: Taxonomy of Cyberattacks. The colored boxes represent attacks that have been demonstrated to be executable using LLMs, whereas the gray boxes indicate attacks that cannot be executed with LLMs.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/42b1/42b135ce-3bdd-45cd-8288-8747b4dc6cdf.png" style="width: 50%;"></div>
Figure 3: Prevalence of the existing attacks
<div style="text-align: center;">Figure 3: Prevalence of the existing attacks</div>
# Hardware Attacks OS Attacks 5. Negative Impacts on Security and Privacy
As shown in Figure 2, we have categorized the attacks into five groups based on their respective positions within the system infrastructure. These categories encompass hardwarelevel attacks, OS-level attacks, software-level attacks, network level attacks, and user-level attacks. Additionally, we have quantified the number of associated research papers published for each group, as illustrated in Figure 3.
Hardware-Level Attacks. Hardware attacks typically involve physical access to devices. However, LLMs cannot directly access physical devices. Instead, they can only access
Yifan Yao et al.: Preprint submitted to Elsevier
rk Attacks User-based Attacks Hardware  Attacks OS Attac Hardware Side-Channel  Attacks Malicious  Firmware Evil Maid  Attacks Fault  Injection Privilege  Escalation Rootkit OS RCE Memory  Attacks OS DoS OS Side-Channel  Attacks information associated with the hardware. Side-channel attack [260, 107, 189] is one attack that can be powered by the LLMs. Side-channel attacks typically entail the analysis of unintentional information leakage from a physical system or implementation, such as a cryptographic device or software, with the aim of inferring secret information (e.g., keys). Yaman [319] has explored the application of LLM techniques to develop side-channel analysis methods. The research evaluates the effectiveness of LLM-based approaches in analyzing side-channel information in two hardwarerelated scenarios: AES side-channel analysis and deeplearning accelerator side-channel analysis. Experiments are conducted to determine the success rates of these methods in both situations. OS-Level Attacks. LLMs operate at a high level of abstraction and primarily engage with text-based input and output. They lack the necessary low-level system access essential for executing OS-level attacks [114, 288, 128]. Nonetheless, they can be utilized for the analysis of information gathered from operating systems, thus potentially aiding in the execution of such attacks. Andreas et al. [94] establish a feedback loop connecting LLM to a vulnerable virtual machine through SSH, allowing LLM to analyze the machine’s state, identify vulnerabilities, and propose concrete attack strategies, which are then executed automatically within the virtual machine. More recently, they [95] introduced an automated Linux privilege-escalation benchmark using local virtual machines and an LLM-guided privilege-escalation tool to assess various LLMs and prompt strategies against the benchmark. Software-Level Attacks. Similar to how they employ LLM to target hardware and operating systems, there are also instances where LLM has been utilized to attack software (e.g., [343, 209, 212, 32]). However, the most prevalent software-level use case involves malicious developers utilizing LLMs to create malware. Mika et al. [17] present a proof-of-concept in which ChatGPT is utilized to distribute malicious software while avoiding detection. Yin et al. [207] investigate the potential misuse of LLM by creating a number of malware programs (e.g., ransomware, worm, keylogger, brute-force malware, Fileless malware). Antonio Monje et al. [194] demonstrate how to trick ChatGPT into quickly generating ransomware. Marcus Botacin [22] explores different coding strategies (e.g., generating entire malware, creating malware functions) and investigates the LLM’s capacities to rewrite malware code. The findings reveal that LLM excels in constructing malware using building block descriptions. Meanwhile, LLM can generate multiple versions of the same semantic content (malware variants), with varying detection rates by Virustotal AV (ranging from 4% to 55%). Network-Level Attacks. LLMs can also be employed for initiating network attacks. A prevalent example of a networklevel attack utilizing LLM is phishing attacks [18, 43]. Fredrik et al. [102] compared AI-generated phishing emails
Network-Level Attacks. LLMs can also be employed for initiating network attacks. A prevalent example of a networklevel attack utilizing LLM is phishing attacks [18, 43]. Fredrik et al. [102] compared AI-generated phishing emails
Page 8 of 24
using GPT-4 with manually designed phishing emails created using the V-Triad, alongside a control group exposed to generic phishing emails. The results showed that personalized phishing emails, whether generated by AI or designed manually, had higher click-through rates compared to generic ones. Tyson et al. [151] investigated how modifying ChatGPT’s input can affect the content of the generated emails, making them more convincing. Julian Hazell [97] demonstrated the scalability of spear phishing campaigns by generating realistic and cost-effective phishing messages for over 600 British Members of Parliament using ChatGPT. In another study, Wang et al. [295] discuss how the traditional defenses may fail in the era of LLMs. CAPTCHA challenges, involving distorted letters and digits, struggle to detect chatbots relying on text and voice. However, LLMs may break the challenges, as they can produce high-quality human-like text and mimic human behavior effectively. There is one study that utilizes LLM for deploying fingerprint attacks. Armin et al. [236] employed density-based clustering to cluster HTTP banners and create text-based fingerprints for annotating scanning data. When these fingerprints are compared to an existing database, it becomes possible to identify new IoT devices and server products.
User-Level Attacks. Recent discussions have primarily focused on user-level attacks, as LLM demonstrates its capability to create remarkably convincing but ultimately deceptive content, as well as establish connections between seemingly unrelated pieces of information. This presents opportunities for malicious actors to engage in a range of nefarious activities. Here are a few examples:
 Misinformation. Overreliance on content generated by LLMs without oversight is raising serious concerns regarding the safety of online content [206]. Numerous studies have focused on detecting misinformation produced by LLMs. Several study [35, 308, 324] reveal content generated by LLMs are harder to detect and may use more deceptive styles, potentially causing greater harm. Canyu Chen et al. [35] propose a taxonomy for LLM-generated misinformation and validate methods. Countermeasures and detection methods [308, 280, 40, 267, 36, 341, 19, 155, 263] have also been developed to address these emerging issues.
 Social Engineering. LLMs not only have the potential to generate content from training data, but they also offer attackers a new perspective for social engineering. Work from Stabb et al. [261] highlights the capability of well-trained LLMs to infer personal attributes from text, such as location, income, and gender. They also reveals how these models can extract personal information from seemingly benign queries. Tong et al. [275] investigated the content generated by LLMs may include user information. Moreover, Polra Victor Falade [76] stated the exploitation by LLM-driven
Yifan Yao et al.: Preprint submitted to Elsevier
social engineers involves tactics such as psychological manipulation, targeted phishing, and the crisis of authenticity.
 Scientific Misconduct. Irresponsible use of LLMs can result in issues related to scientific misconduct, stemming from their capacity to generate original, coherent text. The academic community [45, 265, 215, 46, 179, 72, 200, 223, 87, 139, 226], encompassing diverse disciplines from various countries, has raised concerns about the increasing difficulties in detecting scientific misconduct in the era of LLMs. Concerns arise from LLMs’ ability to generate coherent and original content, including complete papers from unreliable sources [283, 287, 232]. Researchers are also actively engaged in the effort to detect such misconduct. For example, Kavita Kumari et al. [146, 147] proposed DEMASQ, a precise ChatGPT-generated content detector. DEMASQ considers biases in text composition and evasion techniques, achieving high accuracy across diverse domains in identifying ChatGPT-generated content.
 Scientific Misconduct. Irresponsible use of LLMs can result in issues related to scientific misconduct, stemming from their capacity to generate original, coherent text. The academic community [45, 265, 215, 46, 179, 72, 200, 223, 87, 139, 226], encompassing diverse disciplines from various countries, has raised concerns about the increasing difficulties in detecting scientific misconduct in the era of LLMs. Concerns arise from LLMs’ ability to generate coherent and original content, including complete papers from unreliable sources [283, 287, 232]. Researchers are also actively engaged in the effort to detect such misconduct. For example, Kavita Kumari et al. [146, 147] proposed DEMASQ, a precise ChatGPT-generated content detector. DEMASQ considers biases in text composition and evasion techniques, achieving high accuracy across diverse domains in identifying ChatGPT-generated content.  Fraud. Cybercriminals have devised a new tool called FraudGPT [76, 10], which operates like ChatGPT but facilitates cyberattacks. It lacks the safety controls of ChatGPT and is sold on the dark web and Telegram for $200 per month or $1,700 annually. FraudGPT can create fraud emails related to banks, suggesting malicious links’ placement in the content. It can also list frequently targeted sites or services, aiding hackers in planning future attacks. WormGPT [52], a cybercrime tool, offers features such as unlimited character support and chat memory retention. The tool was trained on confidential datasets, with a focus on malware-related and fraud-related data. It can guide cybercriminals in executing Business Email Compromise (BEC) attacks.
 Fraud. Cybercriminals have devised a new tool called FraudGPT [76, 10], which operates like ChatGPT but facilitates cyberattacks. It lacks the safety controls of ChatGPT and is sold on the dark web and Telegram for $200 per month or $1,700 annually. FraudGPT can create fraud emails related to banks, suggesting malicious links’ placement in the content. It can also list frequently targeted sites or services, aiding hackers in planning future attacks. WormGPT [52], a cybercrime tool, offers features such as unlimited character support and chat memory retention. The tool was trained on confidential datasets, with a focus on malware-related and fraud-related data. It can guide cybercriminals in executing Business Email Compromise (BEC) attacks.
Finding IV. As illustrated in Figure 3, when compared to other attacks, it becomes apparent that user-level attacks are the most prevalent, boasting a significant count of 33 papers. This dominance can be attributed to the fact that LLMs have increasingly human-like reasoning abilities, enabling them to generate human-like conversations and content (e.g., scientific misconduct, social engineering). Presently, LLMs do not possess the same level of access to OS-level or hardware-level functionalities. This observation remains consistent with the attack observed in other levels as well. For instance, at the network level, LLMs can be abused to create phishing websites and bypass CAPTCHA mechanisms.
# 6. Vulnerabilities and Defenses in LLMs
In the following section, we embark on an in-depth exploration of the prevalent threats and vulnerabilities associated with LLMs (§6.1). We will examine the specific
Page 9 of 24
risks and challenges that arise in the context of LLMs. In addition to discussing these challenges, we will also delve into the countermeasures and strategies that researchers and practitioners have developed to mitigate these risks (§6.2). Figure 4 illustrates the relationship between the attacks and defenses.
# 6.1. Vulnerabilities and Threats in LLMs In this section, we aim to delve into the potent
In this section, we aim to delve into the potential vulnerabilities and attacks that may be directed towards LLMs. Our examination seeks to categorize these threats into two distinct groups: AI Model Inherent Vulnerabilities and NonAI Model Inherent Vulnerabilities.
# 6.1.1. AI Inherent Vulnerabilities and Threats These are vulnerabilities and threats that stem from
These are vulnerabilities and threats that stem from the very nature and architecture of LLMs, considering that LLMs are fundamentally AI models themselves. For example, attackers may manipulate the input data to generate incorrect or undesirable outputs from the LLM.
(A1) Adversarial Attacks. Adversarial attacks in machine learning refer to a set of techniques and strategies used to intentionally manipulate or deceive machine learning models. These attacks are typically carried out with malicious intent and aim to exploit vulnerabilities in the model’s behavior. We only focus on the most extensively discussed attacks, namely, data poisoning and backdoor attacks.
 Data Poisoning. Data poisoning stands for attackers influencing the training process by injecting malicious data into the training dataset. This can introduce vulnerabilities or biases, compromising the security, effectiveness, or ethical behavior of the resulting models [206]. Various study [148, 290, 289, 2, 291, 239] have demonstrated that pre-trained models are vulnerable to compromise via methods such as using untrusted weights or content, including the insertion of poisoned examples into their datasets. By their inherent nature as pre-trained models, LLMs are susceptible to data poisoning attacks [227, 251, 245]. For example, Alexander et al. [290] showed that even with just 100 poison examples, LLMs can produce consistently negative results or flawed outputs across various tasks. Larger language models are more susceptible to poisoning, and existing defenses like data filtering or model capacity reduction offer only moderate protection while hurting test accuracy.
 Backdoor Attacks. Backdoor attacks involve the malicious manipulation of training data and model processing, creating a vulnerability where attackers can embed a hidden backdoor into the model [322]. Both backdoor attacks and data poisoning attacks involve manipulating machine learning models, which can include manipulation of inputs. However, the key distinction is that backdoor attacks specifically focus on
Yifan Yao et al.: Preprint submitted to Elsevier
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fcf1/fcf1af47-7dc7-4cbd-971e-390066e420b2.png" style="width: 50%;"></div>
<div style="text-align: center;">Vulnerabilities and Threats</div>
<div style="text-align: center;">Defenses</div>
Figure 4: Taxonomy of Threats and the Defenses. The line represents a defense technique that can defend against either a specific attack or a group of attacks.
introducing hidden triggers into the model to manipulate specific behaviors or responses when the trigger is encountered. LLMs are subject to backdoor attacks [161, 331, 167]. For example, Yao et al. [329] a bidirectional backdoor, which combines trigger mechanisms with prompt tuning.
(A2) Inference Attacks. Inference attacks in the context of machine learning refer to a class of attacks where an adversary tries to gain sensitive information or insights about a machine learning model or its training data by making specific queries or observations to the model. These attacks often exploit unintended information leakage from the responses.
• Attribute Inference Attacks. Attribute inference Attack [208, 181, 133, 258, 183, 160] is a type of threat
Page 10 of 24
where an attacker attempts to deduce sensitive or personal information of individuals or entities by analyzing the behavior or responses of a machine learning models. It works against the LLMs as well. Robin et al. [261] presented the first comprehensive examination of pretrained LLMs’ ability to infer personal information from text. Using a dataset of real Reddit profiles, the study demonstrated that current LLMs can accurately infer a variety of personal information (e.g., location, income, sex) with high accuracy.
# • Membership Inferences. Membership inference A tack is a specific type of inference attack in the fie
 Membership Inferences. Membership inference Attack is a specific type of inference attack in the field of data security and privacy that determining whether a data record was part of a model’s training dataset, given white-/black-box access to the model and the specific data record [250, 68, 143, 85, 84, 191, 112]. A number of research studies have explored the concept of membership inference, each adopting a unique perspective and methodology. These studies have explored various membership inference attacks by analyzing the label [42], determining the threshold [120, 28, 96], developing a generalized formulation [278], among other methods. Mireshghallah et al. [192] found that fine-tuning the head of the model exhibits greater susceptibility to attacks when compared to fine-tuning smaller adapters.
(A3) Extraction Attacks. Extraction attacks typically refer to attempts by adversaries to extract sensitive information or insights from machine learning models or their associated data. Extraction attacks and inference attacks share similarities but differ in their specific focus and objectives. Extraction attacks aim to acquire specific resources (e.g., model gradient, training data) or confidential information directly. Inference attacks seek to gain knowledge or insights about the model or data’s characteristics, often by observing the model’s responses or behavior. Various types of data extraction attacks exist, including model theft attacks [130, 137], gradient leakage [158], and training data extraction attacks [29]. As of the current writing, it has been observed that training data extraction attacks may be effective against LLMs. Training data extraction [29] refers to a method where an attacker attempts to retrieve specific individual examples from a model’s training data by strategically querying the machine learning models. Numerous research [344, 210, 326] studies have shown that it is possible to extract training data from LLMs, which may include personal and private information [113, 339]. Notably, the work by Truong et al. [279] stands out for its ability to replicate the model without accessing the original model data. (A4) Bias and Unfairness Exploitation. Bias and unfairness in LLMs pertain to the phenomenon where these models demonstrate prejudiced outcomes or discriminatory behaviors. While bias and fairness issues are not unique to LLMs, they have received more attention due to the ethical and societal concerns. That is, the societal impact of LLMs
Yifan Yao et al.: Preprint submitted to Elsevier
has prompted discussions about the ethical responsibilities of organizations and researchers developing and deploying these models. This has led to increased scrutiny and research on bias and fairness. Concerns of bias were raised from various fields, encompassing gender and minority groups [65, 144, 81, 244], the identification of misinformation, political aspects. Multiple studies [269, 281] revealed biases in the language used while querying LLMs. Moreover, Urman et al. [282] discovered that biases may arise from adherence to government censorship guidelines. Bias in professional writing [292, 263, 79] involving LLMs is also a concern within the community, as it can significantly damage credibility. The biases of LLMs may also lead to negative side effects in areas beyond text-based applications. Dai et al. [47] noted that content generated by LLMs might introduce biases in neural retrieval systems, and Huang et al. [111] discovered that biases could also be present in LLM generated code. (A5) Instruction Tuning Attacks. Instruction tuning, also known as instruction-based fine-tuning, is a machine-learning technique used to train and adapt language models for specific tasks by providing explicit instructions or examples during the fine-tuning process. In LLMs, instruction-tuning attacks refer to a class of attacks or manipulations that target instruction-tuned LLMs. These attacks are aimed at exploiting vulnerabilities or limitations in LLMs that have been fine-tuned with specific instructions or examples for particular tasks.
 Jailbreaking. Jailbreaking in LLMs involves bypassing security features to enable responses to otherwise restricted or unsafe questions, unlocking capabilities usually limited by safety protocols. Numerous studies have demonstrated various methods for successfully jailbreaking LLMs [159, 271, 248]. Wei et al. [301] emphasized that the alignment capabilities of LLMs can be influenced or manipulated through in-context demonstrations. In addition to this, several researches [300, 132] also demonstrated similar manipulation using various approaches, highlighting the versatility of methods that can jailbreaking LLMs. More recently, MASTERKEY [54] employed a timebased method for dissecting defenses, and demonstrated proof-of-concept attacks. It automatically generates jailbreak prompts with a 21.58Moreover, diverse methods have been employed in jailbreaking LLMs, such as conducting fuzzing [328], implementing optimized search strategies [353], and even training LLMs specifically to jailbreak other LLMs [53, 353]. Meanwhile, Cao et al. [27] developed RA-LLM, a method to lowers the success rate of adversarial and jailbreaking prompts without needing of retraining or access to model parameters.
# • Prompt Injection. Prompt injection attack describes a method of manipulating the behavior of LLMs to
• Prompt Injection. Prompt injection attack describes a method of manipulating the behavior of LLMs to elicit unexpected and potentially harmful responses. This technique involves crafting input prompts in a
Page 11 of 24
way that bypasses the model’s safeguards or triggers undesirable outputs. A substantial amount of research [177, 332, 135, 299, 173, 124] has already automated the process of identifying semantic preserving payload in prompt injections with various focus. Facilitated by the capability for fine-tuning, backdoors may be introduced through prompt attacks [12, 133, 346, 243]. Moreover, Greshake et al. [89] expressed concerns about the potential for new vulnerabilities arising from LLMs invoking external resources. Other studies have also demonstrated the ability to take advantage of prompt injection attacks, such as unveiling guide prompts [342], virtualizing prompt injection [320], and integrating applications [178]. He et al. [100, 101] explored a shift towards leveraging LLMs, trained on extensive datasets, for mitigating such attacks.
# • Denial of Service. A Denial of Servic tack is a type of cyber attack that aim
 Denial of Service. A Denial of Service (DoS) attack is a type of cyber attack that aims to exhaust computational resources, causing latency or rendering resources unavailable. Due to the nature of LLMs require significant amount of resources, attackers use deliberately construct prompts to reduce the availability of models [59]. Shumailov et al. [252] proved the possibility of conducting sponge attacks in the field of LLMs, specifically designed to maximize energy consumption and latency (by a factor of 10 to 200). This strategy aims to draw the community’s attention to their potential impact on autonomous vehicles, as well as scenarios requiring making decisions in timely manner.
Finding V. Currently, there is limited research on model extraction attacks [68], parameter extraction attacks, or the extraction of other intermediate esults [279]. While there are a few mentions of these topics, they tend to remain primarily theoretical (e.g., [172]), with limited practical implementation or empirical exploration. We believe that the sheer scale of parameters in LLMs complicates these traditional approaches, rendering them less effective or even infeasible. Additionally, the most powerful LLMs are privately owned, with their weights, parameters, and other details kept confidential, further shielding them from conventional attack strategies. Strict censorship of outputs generated by these LLMs challenges even blackbox traditional ML attacks, as it limits the attackers’ ability to exploit or analyze the model’s responses.
6.1.2. Non-AI Inherent Vulnerabilities and Threats We also need to consider non-AI Inherent Attacks, which encompass external threats and new vulnerabilities (which have not been observed or investigated in traditional AI models) that LLMs might encounter. These attacks may not be intricately linked to the internal mechanisms of the AI model, yet they can present significant risks. Illustrative instances of non-AI Inherent Attacks involve system-level vulnerabilities (e.g., remote code execution).
Yifan Yao et al.: Preprint submitted to Elsevier
(A6) Remote Code Execution (RCE). RCE attacks typically target vulnerabilities in software applications, web services, or servers to execute arbitrary code remotely. While RCE attacks are not typically applicable directly to LLMs, if an LLM is integrated into a web service (e.g.,https: //chat.openai.com/) and if there are RCE vulnerabilities in the underlying infrastructure or code of that service, it could potentially lead to the compromise of the LLM’s environment. Tong et al. [175] identified 13 vulnerabilities in six frameworks, including 12 RCE vulnerabilities and 1 arbitrary file read/write vulnerability. Additionally, 17 out of 51 tested apps were found to have vulnerabilities, with 16 being vulnerable to RCE and 1 to SQL injection. These vulnerabilities allow attackers to execute arbitrary code on app servers through prompt injections. (A7) Side Channel. While LLMs themselves do not typically leak information through traditional side channels such as power consumption or electromagnetic radiation, they can be vulnerable to certain side-channel attacks in practical deployment scenarios. For example, Edoardo et al. [51] introduce privacy side channel attacks, which are attacks that exploit system-level components (e.g., data filtering, output monitoring) to extract private information at a much higher rate than what standalone models can achieve. Four categories of side channels covering the entire ML lifecycle are proposed, enabling enhanced membership inference attacks and novel threats (e.g., extracting users’ test queries). For instance, the research demonstrates how deduplicating training data before applying differentially-private training creates a side channel that compromises privacy guarantees. (A8) Supply Chain Vulnerabilities. Supply Chain Vulnerabilities refer to the risks in the lifecycle of LLM applications that may arise from using vulnerable components or services. These include third-party datasets, pre-trained models, and plugins, any of which can compromise the application’s integrity [206]. Most research in this field is focused on the security of plugins. An LLM plugin is an extension or add-on module that enhances the capabilities of an LLM. Third-party plug-ins have been developed to expand its functionality, enabling users to perform various tasks, including web searches, text analysis, and code execution. However, some of the concerns raised by security experts [206, 25] include the possibility of plug-ins being used to steal chat histories, access personal information, or execute code on users’ machines. These vulnerabilities are associated with the use of OAuth in plug-ins, a web standard for data sharing across online accounts. Umar et al. [115] attempted to address this problem by designing a framework. The framework formulates an extensive taxonomy of attacks specific to LLM platforms, taking into account the capabilities of plugins, users, and the LLM platform itself. By considering the relationships between these stakeholders, the framework helps identify potential security, privacy, and safety risks.
# 6.2. Defenses for LLMs In this section, we examine
In this section, we examine the range of existing defense methods against various attacks and vulnerabilities associated with LLMs1.
# 6.2.1. Defense in Model Architecture
Model architectures determine how knowledge and concepts are stored, organized, and contextually interacted with, which is crucial in the safety of Large Language Models. There have been a lot of works [165, 351, 168, 333] delved into how model capacities affect the privacy preservation and robustness of LLMs. Li et al. [165] revealed that language models with larger parameter sizes can be trained more effectively in the differential privacy manner using appropriate non-standard hyper-parameters, in comparison to smaller models. Zhu et al. [351] and Li et al. [168] found that LLMs with larger capacities, such as those with more extensive parameter sizes, generally show increased robustness against adversarial attacks. This was also verified in the Out-of-distribution (OOD) robustness scenarios by Yuan et al. [333]. Beyond the architecture of LLMs themselves, studies have focused on improving LLM safety by combining them with external modules including knowledge graphs [39] and cognitive architectures (CAs) [150, 11]. Romero et al. [231] proposed improving AI robustness by incorporating various cognitive architectures into LLMs. Zafar et al. [336] aimed to build trust in AI by enhancing the reasoning abilities of LLMs through knowledge graphs.
# 6.2.2. Defenses in LLM Training and Inference
Defense Strategies in LLM Training. The core components of LLM training include model architectures, training data, and optimization methods. Regarding model architectures, we examine trustworthy designs that exhibit increased robustness against malicious use. For training corpora, our investigation focuses on methods aimed at mitigating undesired properties during the generation, collection, and cleaning of training data. In the context of optimization methods, we review existing works that developed safe and secure optimization frameworks.
 Corpora Cleaning. LLMs are shaped by their training corpora, from which they learn behavior, concepts, and data distributions [302]. Therefore, the safety of LLMs is crucially influenced by the quality of the training corpora [86, 204]. However, it has been widely acknowledged that raw corpora collected from the web are full of issues of fairness [14], toxicity [88], privacy [208], truthfulness [171], etc. A lot of efforts have been made to clean raw corpora and create highquality training corpora for LLMs [129, 306, 152, 307, 213, 277]. In general, these pipelines consist of the following steps: language identification [129, 9], detoxification [88, 48, 180, 195], debiasing [188, 21, 16],
1Please be aware that we will not delve into solutions for non-AI inherent vulnerabilities as they tend to be highly specific to individual cases.
Yifan Yao et al.: Preprint submitted to Elsevier
de-identification (personally identifiable information (PII)) [264, 284], and deduplication [153, 134, 106, 157]. Debiasing and detoxification aimed to remove undesirable content from training corpora.
 Optimization Methods. Optimization objectives are crucial in directing how LLMs learn from training data, influencing which behaviors are encouraged or penalized. These objectives affect the prioritization of knowledge and concepts within corpora, ultimately impacting the overall safety and ethical alignment of LLMs. In this context, robust training methods like adversarial training [176, 293, 350, 330, 163] and robust fine-tuning [66, 121] have shown resilience against perturbation-based text attacks. Drawing inspiration from traditional adversarial training in the image field [182], Ivgi et al. [116] and Yoo et al. [330] applied adversarial training to LLMs by generating perturbations concerning discrete tokens. Wang et al. [293] extended this approach to the continuous embedding space, facilitating more practical convergence, as followed by subsequent research [176, 350, 163]. Safety alignments [205], an emerging learning paradigm, guide LLM behavior using well-aligned additional models or human annotations, proving effective for ethical alignment. Efforts to align LLMs with other LLMs [334] and LLMs themselves [268]. In terms of human annotations, Zhou et al. [349] and Shi et al. [249] emphasized the importance of highquality training corpora with carefully curated instructions and outputs for enhancing instruction-following capabilities in LLMs. Bianchi et al. [20] highlighted that the safety of LLMs can be substantially improved by incorporating a limited percentage (e.g., 3%) of safe examples during fine-tuning.
Defense Strategies in LLM Inference. When LLMs are deployed as cloud services, they operate by receiving prompts or instructions from users and generating completed sentences in response. Given this interaction model, the implementation of test-time LLM defense becomes a necessary and critical aspect of ensuring safe and appropriate outputs. Generally, test-time defense encompasses a range of strategies, including the pre-processing of prompts and instructions to filter or modify inputs, the detection of abnormal events that might signal misuse or problematic queries, and the post-processing of generated responses to ensure they adhere to safety and ethical guidelines. Test-time LLM defenses are essential to maintain the integrity and trustworthiness of LLMs in real-time applications.
• Instruction Processing (Pre-Processing). Instruction pre-processing applies transformations over instructions sent by users, in order to destroy potential adversarial contexts or malicious intents. It plays a vital role as it blocks out most malicious usage and
Page 13 of 24
prevents LLMs from receiving suspicious instructions. In general, instruction pre-processing methods can be categorized as instruction manipulation [246, 230, 140, 117, 318], purification [164], and defensive demonstrations [172, 193, 301]. Jain et al. [117] and Kirchenbauer et al. [140] evaluated multiple baseline preprocessing methods against jailbreaking attacks, including retokenization and paraphrase. Li et al. [164] proposed to purify instructions by first masking the input tokens and then predicting the masked tokens with other LLMs. The predicted tokens will serve as the purified instructions. Wei et al. [301] and Mo et al. [193] demonstrated that inserting predefined defensive demonstrations into instructions effectively defends jailbreaking attacks of LLMs.
 Malicious Detection (In-Processing). Malicious detection provides in-depth examinations of LLM intermediate results, such as neuron activation, regarding the given instructions, which are more sensitive, accurate, and specified for malicious usage. Sun et al. [266] proposed to detect backdoored instructions with backward probabilities of generations. Xi et al. [312] differentiated normal and poisoned instructions from the perspective of mask sensitivities. Shao et al. [246] identified suspicious words according to their textual relevance. Wang et al. [298] detected adversarial examples according to the semantic consistency among multiple generations, which has been explored in the uncertainty quantification of LLMs by Duan et al. [67]. Apart from the intrinsic properties of LLMs, there have been works leveraging the linguistic statistic properties, such as detecting outlier words [220],
 Generation Processing (Post-Processing). Generation post processing refers to examining the properties (e.g., harmfulness) of the generated answers and applying modifications if necessary, which is the final step before delivering responses to users. Chen et al. [34] proposed to mitigate the toxicity of generations by comparing with multiple model candidates. Helbling et al. [103] incorporated individual LLMs to identify the harmfulness of the generated answers, which shared similar ideas as Xiong et al. [317] and Kadavath et al. [131] where they revealed that LLMs can be prompted to answer the confidences regarding the generated responses.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0fd2/0fd21821-d4ef-4a3e-9d7a-98f7557c0737.png" style="width: 50%;"></div>
Yifan Yao et al.: Preprint submitted to Elsevier
7. Discussion 7.1. LLM in Other Security Re
# 7.1. LLM in Other Security Related Topics
LLMs in Cybersecurity Education. LLMs can be used in security practices and education [80, 162, 270]. For example, in a software security course, students are tasked with identifying and resolving vulnerabilities in a web application using LLMs. Jingyue et al. [162] investigated how ChatGPT can be used by students for these exercises. Wesley Tann et al. [270] focused on the evaluation of LLMs in the context of cybersecurity Capture-The-Flag (CTF) exercises (participants find “flags” by exploiting system vulnerabilities). The study first assessed the question-answering performance of these LLMs on Cisco certifications with varying difficulty levels, then examined their abilities in solving CTF challenges. Jin et al. [126] conducted a comprehensive study on LLMs’ understanding of binary code semantics [127] across different architectures and optimization levels, providing key insights for future research in this area.
LLMs in Cybersecurity Laws, Policies and Compliance. LLMs can assist in drafting security policies, guidelines, and compliance documentation, ensuring that organizations meet regulatory requirements and industry standards. However, it’s important to recognize that the utilization of LLMs can potentially necessitate changes to current cybersecurityrelated laws and policies. The introduction of LLMs may raise new legal and regulatory considerations, as these models can impact various aspects of cybersecurity, data protection, and privacy. Ekenobi et al. [273] examined the legal implications arising from the introduction of LLMs, with a particular focus on data protection and privacy concerns. It acknowledges that ChatGPT’s privacy policy contains commendable provisions for safeguarding user data against potential threats. The paper also advocated for emphasizing the relevance of the new law.
# 7.2. Future Directions We have gleaned valuable 
We have gleaned valuable lessons that we believe can sh future directions.
• Using LLMs for ML-Specific Tasks. We noticed that LLMs can effectively replace traditional machine learning methods and in this context, if traditional machine learning methods can be employed in a specific security application (whether offensive or defensive in nature), it is highly probable that LLMs can also be applied to address that particular challenge. For instance, traditional machine learning methods have found utility in malware detection, and LLMs can similarly be harnessed for this purpose. Therefore, one promising avenue is to harness the potential of LLMs in security applications where machine learning serves as a foundational or widely adopted technique. As security researchers, we are capable of designing LLM-based approaches to tackle security issues. Subsequently, we can compare these approaches with state-of-the-art methods to push the boundaries.
Page 14 of 24
 Replacing Human Efforts. It is evident that LLMs have the potential to replace human efforts in both offensive and defensive security applications. For instance, tasks involving social engineering, traditionally reliant on human intervention, can now be effectively executed using LLM techniques. Therefore, one promising avenue for security researchers is to identify areas within traditional security tasks where human involvement has been pivotal and explore opportunities to substitute these human efforts with LLM capabilities.
# • Modifying Traditional ML Attacks for LLMs. we have observed that many security vulnerabilities in
have observed that many security vulnerabilities in LLMs are extensions of vulnerabilities found in traditional machine-learning scenarios. That is, LLMs remain a specialized instance of deep neural networks, inheriting common vulnerabilities such as adversarial attacks and instruction tuning attacks. With the right adjustments (e.g., the threat model), traditional ML attacks can still be effective against LLMs. For instance, the jailbreaking attack is a specific form of instruction tuning attack aimed at producing restricted texts.
# • Adapting Traditional ML Defenses for LLMs. The countermeasures traditionally employed for vulnera
 Adapting Traditional ML Defenses for LLMs. The countermeasures traditionally employed for vulnerability mitigation can also be leveraged to address these security issues. For example, there are existing efforts that utilize traditional Privacy-Enhancing Technologies (e.g., zero-knowledge proofs, differential privacy, and federated learning [304, 305] ) to tackle privacy challenges posed by LLMs. Exploring additional PETs techniques, whether they are established methods or innovative approaches, to address these challenges represents another promising research direction.
 Solving Challenges in LLM-Specific Attacks. As previously discussed, there are several challenges associated with implementing model extraction or parameter extraction attacks (e.g., vast scale of LLM parameters, private ownership and confidentiality of powerful LLMs). These novel characteristics introduced by LLMs represent a significant shift in the landscape, potentially leading to new challenges and necessitating the evolution of traditional ML attack methodologies.
# 8. Related Work
There have already been a number of LLM surveys released with a variety of focuses (e.g., LLM evolution and taxonomy [31, 347, 309, 93, 311, 23, 348], software engineering [77, 108], and medicine [274, 44]). In this paper, our primary emphasis is on the security and privacy aspects of LLMs. We now delve into an examination of the existing literature pertaining to this particular topic. Peter J. Caven [30] specifically explores how LLMs (particularly,
Yifan Yao et al.: Preprint submitted to Elsevier
ChatGPT) could potentially alter the current cybersecurity landscape by blending technical and social aspects. Their emphasis leans more towards the social aspects. Muna et al. [5] and Marshall et al. [185] discussed the impact of ChatGPT in cybersecurity, highlighting its practical applications (e.g., code security, malware detection). Dhoni et al. [62] demonstrated how LLMs can assist security analysts in developing security solutions against cyber threats. However, their work does not extensively address the potential cybersecurity threats that LLM may introduce. A number of surveys (e.g., [92, 59, 247, 49, 60, 228, 240, 241, 7]) highlight the threats and attacks against LLMs. In comparison to our work, they do not dedicate as much text to the vulnerabilities that the LLM may possess. Instead, their primary focus lies in the realm of security applications, as they delve into utilizing LLMs for launching cyberattacks. Attia Qammar et al. [219] and Maximilian et al. [196] discussed vulnerabilities exploited by cybercriminals, with a specific focus on the risks associated with LLMs. Their works emphasized the need for strategies and measures to mitigate these threats and vulnerabilities. Haoran Li et al. [166] analyzed current privacy concerns on LLMs, categorizing them based on adversary capabilities, and explored existing defense strategies. Glorin Sebastian [242] explored the application of established Privacy-Enhancing Technologies (e.g., differential privacy [70], federated learning [338], and data minimization [216]) for safeguarding the privacy of LLMs. Smith et al. [256] also discussed the privacy risks of LLMs. Our study comprehensively examined both the security and privacy aspects of LLMs. In summary, our research conducted an extensive review of the literature on LLMs from a three-fold perspective: beneficial security applications (e.g., vulnerability detection, secure code generation), adverse implications (e.g., phishing attacks, social engineering), and vulnerabilities (e.g., jailbreaking attacks, prompt attacks), along with their corresponding defensive measures.
# 9. Conclusion
Our work represents a pioneering effort in systematically examining the multifaceted role of LLMs in security and privacy. On the positive side, LLMs have significantly contributed to enhancing code and data security, while their versatile nature also opens the door to malicious applications. We also delved into the inherent vulnerabilities within these models, and discussed defense mechanisms. We have illuminated the path forward for harnessing the positive aspects of LLMs while mitigating their potential risks. As LLMs continue to evolve and find their place in an everexpanding array of applications, it is imperative that we remain vigilant in addressing security and privacy concerns, ensuring that these powerful models contribute positively to the digital landscape.
Page 15 of 24
# Acknowledgement We thank the anonymo
We thank the anonymous reviewers and Xin Jin from The Ohio State University for their invaluable feedback. This research was supported partly by the NSF award FMitF2319242. Any opinions, findings, conclusions, or recommendations expressed are those of the authors and not necessarily of the NSF.
# References
[1] M. Abbasian, I. Azimi, A. M. Rahmani, and R. Jain, “Conversational health agents: A personalized llm-powered agent framework,” 2023. [2] H. Aghakhani, W. Dai, A. Manoel, X. Fernandes, A. Kharkar, C. Kruegel, G. Vigna, D. Evans, B. Zorn, and R. Sim, “Trojanpuzzle: Covertly poisoning code-suggestion models,” arXiv preprint arXiv:2301.02344, 2023. [3] B. Ahmad, S. Thakur, B. Tan, R. Karri, and H. Pearce, “Fixing hardware security bugs with large language models,” arXiv preprint arXiv:2302.01215, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2302.01215 [4] M. AI, “Introducing llama: A foundational, 65-billionparameter language model,” https://ai.meta.com/blog/ large-language-model-llama-meta-ai/, feb 2023, accessed: 2023-11-13. [5] M. Al-Hawawreh, A. Aljuhani, and Y. Jararweh, “Chatgpt for cybersecurity: practical applications, challenges, and future directions,” Cluster Computing, vol. 26, no. 6, pp. 3421–3436, 2023. [6] S. Alagarsamy, C. Tantithamthavorn, and A. Aleti, “A3test: Assertion-augmented automated test case generation,” arXiv preprint arXiv:2302.10352, 2023. [7] M. Alawida, B. A. Shawar, O. I. Abiodun, A. Mehmood, A. E. Omolara et al., “Unveiling the dark side of chatgpt: Exploring cyberattacks and enhancing user awareness,” 2023. [8] T. Ali and P. Kostakos, “Huntgpt: Integrating machine learningbased anomaly detection and explainable ai with large language models (llms),” arXiv preprint arXiv:2309.16021, 2023. [9] E. Ambikairajah, H. Li, L. Wang, B. Yin, and V. Sethu, “Language identification: A tutorial,” IEEE Circuits and Systems Magazine, vol. 11, no. 2,