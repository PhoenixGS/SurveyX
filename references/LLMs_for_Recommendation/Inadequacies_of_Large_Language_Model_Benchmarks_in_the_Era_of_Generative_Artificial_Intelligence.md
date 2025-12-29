# Inadequacies of Large Language Model Benchmarks in the Era of Generative Artificial Intelligence
Timothy R. McIntosh *, Teo Susnjak , Nalin Arachchilage , Tong Liu , Dan Xu , Paul Watters , Senior Member, IEEE, and Malka N. Halgamuge , Senior Member, IEEE
Abstract—The rapid rise in popularity of Large Language Models (LLMs) with emerging capabilities has spurred public curiosity to evaluate and compare different LLMs, leading many researchers to propose their own LLM benchmarks. Noticing preliminary inadequacies in those benchmarks, we embarked on a study to critically assess 23 state-of-the-art LLM benchmarks, using our novel unified evaluation framework through the lenses of people, process, and technology, under the pillars of benchmark functionality and integrity. Our research uncovered significant limitations, including biases, difficulties in measuring genuine reasoning, adaptability, implementation inconsistencies, prompt engineering complexity, evaluator diversity, and the overlooking of cultural and ideological norms in one comprehensive assessment. Our discussions emphasized the urgent need for standardized methodologies, regulatory certainties, and ethical guidelines in light of Artificial Intelligence (AI) advancements, including advocating for an evolution from static benchmarks to dynamic behavioral profiling to accurately capture LLMs’ complex behaviors and potential risks. Our study highlighted the necessity for a paradigm shift in LLM evaluation methodologies, underlining the importance of collaborative efforts for the development of universally accepted benchmarks and the enhancement of AI systems’ integration into society. Index Terms—Artificial Intelligence (AI), AI Evaluation, Benchmark, Evaluation Frameworks, Large Language Model (LLM).
arXiv:2402.09880v2
# I. INTRODUCTION
I. INTRODUCTION
L ARGE Language Models (LLMs), a sophisticated branch of generative Artificial Intelligence (AI), have evolved from narrow, task-specific systems to versatile models capable of few-shot learning and handling diverse tasks [1], [2]. Evaluating LLMs is crucial for understanding their capabilities and limitations. Automatic evaluation methods, which employ standard metrics, offer computational efficiency, while human evaluations provide nuanced insights into the quality and accuracy of LLM responses [3], [4]. Recent advancements like GPT-4 and Gemini, with their multimodal capabilities, and Mistral 8x7B’s integration of Mixture of Experts (MoE), have enhanced LLMs’ ability to process specialized domain knowledge and reason across diverse tasks [1]–[3]. The proliferation of over 700,000 LLMs on platforms like HuggingFace, alongside many publicly available commercial LLMs, has intensified competition among developers, driving the need for benchmarks to uniformly evaluate and compare LLM performance across dimensions such as accuracy, robustness, and reasoning, as these directly influence an LLM’s reliability in real-world applications [3]. For example, an LLM that excels
Manuscript received October 15, 2024. Corresponding Author: Timothy R. McIntosh (e-mail: timothy.mcintosh@rmit.edu.au).
in accuracy but lacks robustness might fail when confronted with unexpected inputs or novel scenarios. Benchmarks serve as standardized sets of tasks or datasets that assess key aspects such as accuracy, efficiency, and ethical considerations like bias or fairness, guiding both development and deployment decisions [3]. Widely used benchmarks, such as GLUE, SuperGLUE, and MMLU, are expected to enable consistent LLM evaluations, allowing researchers to fine-tune LLMs for specific tasks or domains, and facilitating performance comparisons across different models in real-world scenarios. Unlike the automobile and aviation industries, where clear regulations and well-defined public consensus guide benchmarking practices [5], the advanced AI field lacks such universally accepted standards, leading many researchers to devise their own benchmarks. Benchmarks of LLMs, such as BIG-bench [6] by Google DeepMind and PromptBench [7] by Microsoft Research, have encompassed diverse methods, each with unique approaches and criteria, focusing mainly on exam-style or task-based assessments. Such methods, evaluating models’ abilities to perform specific functions or solve problems, typically emphasized tasks with predefined answers or scenarios with limited variability [7], [8]. Models were evaluated using metrics like accuracy, perplexity, and F1-score on fixed datasets (e.g., [9], [10]), or using human evaluators to measure “human-level” performance (e.g., [11], [12]). The common approach of LLM benchmarking has often assumed a standardized “correct” answer or a marking rubric for each question, which LLMs are expected to reproduce, and the same benchmark sets are repeatedly used to compare and rank LLMs, ignoring their broader implications and real-world applicability. While providing insights, we believe this approach often fails to capture the subtleties and complexities of realworld use, as previous studies mainly focused on functionality within the technological context, neglecting processual and human aspects, especially the diversity of human values and cultures [13]. For instance, an LLM that performs well on standardized tasks may still struggle in contexts requiring cultural sensitivity, such as understanding sophisticated social interactions or ethical concerns in decision-making, which can vary significantly across different societies and user groups. Additionally, LLM benchmarks have often lacked a thorough examination, especially in evaluating LLMs’ full capabilities, limitations, and safety issues [14], highlighting the need for a more comprehensive and sophisticated approach. Evaluating LLMs and generative AI requires a method that assesses not only standardized task performance but also real-world applicability and safety, as a lack of such holistic evaluations
can lead to LLMs that perform well in controlled environments but fail in critical real-world applications, posing risks such as perpetuating bias, making unsafe decisions, or being vulnerable to manipulation [2], [3]. This study was motivated by our concern over the rapid proliferation of LLM benchmark studies and the competition among them to establish superiority, often at the expense of well-rounded functionality and integrity assurance. When evaluating LLM benchmarks, two core aspects are critical for a reliable and unbiased evaluation of LLMs: functionality, which refers to how well a benchmark measures the specific capabilities of an LLM in alignment with real-world applications, and integrity, which ensures that the benchmark resists manipulation or gaming by models that exploit its criteria to produce misleading results. An inadequacy, in this context, refers to any deficiency or shortcoming in a benchmark’s ability to fully capture an LLM’s functionality or maintain its integrity during evaluation. Our preliminary observations indicated systemic inadequacies in most current LLM benchmarks, where superficial metrics often replaced comprehensive evaluations of LLM functionality and the assurance of benchmark integrity. As a result, many benchmarks failed to measure the complex, evolving capabilities of LLMs in real-world settings or address the risks of LLMs gaming and overfitting them. We propose a re-evaluation of current LLM benchmarking criteria, hypothesizing that a comprehensive evaluation framework must integrate both functionality and integrity, to provide a balanced, holistic, in-depth assessment of LLMs. This study aims to systematically analyze the inadequacies of current exam-style benchmarking in multimodal LLMs through the following research questions: (1) How can we identify, categorize, and explain the common inadequacies of state-of-the-art LLM benchmarks? (2) Are these inadequacies evident in the popular benchmarks we have identified? (3) What should a comprehensive evaluation of LLM benchmarks include, considering both functionality and integrity for a complete understanding of LLM risks and societal impacts? Through this survey study, we propose a unified evaluation framework for LLM benchmarks, aligned with the domains of people, process, and technology, designed to facilitate a thorough examination of benchmarks. Our framework aims to enhance both functionality and integrity in benchmark design, guiding the development of more effective and secure LLM evaluation methods that reflect real-world applicability and ensure robust model development and deployment. The major contributions of this study are as follows: 1) We proposed a unified evaluation framework for LLM benchmarks, based on the domains of people, process, and technology, as a foundation to comprehensively and holistically assess both the functionality and integrity of LLMs in real-world applications. 2) Using this framework, we conducted a systematic critique of 23 state-of-the-art LLM benchmarks, revealing key inadequacies and offering targeted methodological improvements to enhance the accuracy and fairness of LLM assessments. 3) Building on the insights from our framework, we introduced an advanced evaluation method that extended
traditional benchmarking with behavioral profiling and regular audits, creating a dynamic and ongoing assessment process that addressed evolving issues of realworld applicability, inclusivity, and security. The rest of this paper is organized as follows: Section II offers a comprehensive analysis of prevalent benchmarks. Section III introduces the proposed unified evaluation framework for assessing LLMs. Section IV provides a detailed overview of 23 selected LLM benchmarks, summarizing their characteristics, focus areas, and evaluation methodologies. Section V examines the technological challenges in current LLM benchmarking practices. Section VI discusses the processual elements affecting LLM evaluations. Section VII explores the human dynamics influencing LLM benchmarking. Section VIII provides a discussion on the overarching themes and implications of our findings. Finally, Section IX concludes the paper, summarizing the key findings and contributions.
II. BACKGROUND AND RELATED WORK
# II. BACKGROUND AND RELATED WORK
Benchmarking is a critical process in computer science, serving as a standardized method to evaluate and compare the performance of hardware and software systems [15]. This section reviews traditional benchmarking practices in computer science and contrasts them with the emerging challenges of benchmarking in generative AI and LLMs.
# A. Traditional Benchmarking in Computer Science
A. Traditional Benchmarking in Computer Science
In computer science, benchmarking involves running a set of standardized tests on hardware or software systems to measure their performance under controlled conditions [15]. Such benchmarks are designed to be repeatable and objective, allowing for fair comparisons between different systems or configurations. For instance, the SPEC CPU benchmark suite evaluates a processor’s ability to handle compute-intensive tasks by measuring execution times of standardized workloads [16]. Standardization is essential in benchmarking to ensure consistency across evaluations, making the results reliable and widely accepted within the community [17]. Benchmarking methodologies typically focus on specific performance metrics such as processing speed, memory bandwidth, or energy efficiency [15], [18]. Tools like Cinebench and 3DMark assess computational performance by performing intensive CPU or GPU tasks, which is critical for high-performance computing applications, and they help identify system bottlenecks and guide optimization efforts, contributing to advancements in hardware and software design [15], [17], [18]. The development of benchmarks requires careful consideration of workload representativeness to ensure that the tests reflect realworld usage scenarios [19]. However, traditional benchmarking faces challenges like benchmark manipulation, where systems are engineered to perform exceptionally well on specific benchmarks without delivering proportional real-world performance [20]. For example, hardware manufacturers might optimize compilers or system settings to inflate benchmark scores, leading to misleading conclusions [21]. This has prompted calls for more robust and comprehensive benchmarking practices that
can resist gaming and provide a true measure of system capabilities [22]. Additionally, the reliability of benchmarking results in computer science and AI depends critically on their reproducibility, generalizability across different contexts, consistency over time, and objectivity in measurement, but at the same time, it raises the question of who holds the authority to set benchmark standards [17], [22]. Consider that the same anti-malware products can score differently in tests like VB100, AV-Comparatives, and AV-Test, even when tests are administered around the same time, indicating variations in benchmarking criteria and evaluation methodologies [23]. Moreover, within the context of cybersecurity, our previous survey on ransomware revealed concerns about the self-benchmarking practices in many anti-ransomware studies, which often used diverse ransomware samples and methodologies and based their claims of superiority merely on higher detection rates [24]. This situation underscores the need for rigorous, comprehensive, and scientifically sound benchmarking practices that can provide reliable, meaningful and universally accepted comparisons.
# B. Benchmarking in Generative AI and LLMs
Benchmarking in generative AI and LLMs introduces unique challenges that differ fundamentally from traditional computer science benchmarking. LLMs like Llama and GPT4 are capable of understanding and generating human-like text, performing tasks ranging from translation to creative writing [1], [4]. Evaluating such models requires benchmarks that can assess not just quantitative performance, but also qualitative aspects like coherence, relevance, and ethical considerations [3]. Unlike traditional benchmarks with clear numerical metrics, LLM benchmarking often involves subjective judgments and complex evaluation criteria [13], [25]. LLM benchmarks commonly involve tasks such as language understanding, reasoning, and dialogue generation, using datasets like GLUE to provide standardized evaluation platforms [26]. However, the open-ended nature of language means that there can be multiple valid responses to a given prompt, complicating the assessment of correctness [1]. Furthermore, LLMs operate as black boxes with opaque internal mechanisms, making it difficult to interpret their decision-making processes and identify potential biases or ethical issues [4], [27]. This opacity poses risks when models generate harmful or biased content without transparent accountability mechanisms [28]. For instance, an LLM used in a customer service chatbot might produce inappropriate or biased responses based on hidden internal biases, without any easy way to diagnose the underlying issue or improve the LLM’s behavior [29]. In high-stakes domains like healthcare or legal services, black-box LLMs could make critical errors without providing interpretable justifications, undermining trust and making it difficult to ensure the LLM’s safety or fairness [30]. The current LLM evaluation processes could vary significantly, involving text string comparisons for straightforward answers or relying on human judgment for more complex and subtle responses [3]. Unlike traditional benchmarks that are often static and hardware-focused, LLM benchmarking must account for the
dynamic and evolving nature of language and societal norms [31]. LLMs can inadvertently learn and propagate biases present in training data, requiring benchmarks that can detect and mitigate such issues [32]. Additionally, LLMs may overfit to benchmark datasets, memorizing answers rather than demonstrating true understanding, which undermines the validity of the evaluation [33]. The potential for data contamination, where test data leaks into training datasets, further complicates the benchmarking process [34]. Moreover, the global deployment of LLMs requires benchmarks that consider linguistic diversity and cultural nuances, moving beyond English-centric evaluations to include multiple languages and dialects, which contrasts with traditional benchmarks that are largely language-agnostic or focused on English [35]. Ethical considerations are also more pronounced in LLM benchmarking, as LLMs must navigate complex social norms and avoid generating inappropriate content [36]. The aforementioned factors require a multidisciplinary approach to LLM benchmarking, integrating insights from linguistics, ethics, and social sciences to create robust evaluation frameworks [3], [13], [30].
# III. UNIFIED EVALUATION FRAMEWORK FOR LLM B
# III. UNIFIED EVALUATION FRAMEWORK FOR LLM
BENCHMARKS
In this section, we developed a comprehensive framework to benchmark LLMs, grounded in principles of cybersecurity risk assessment. We based the framework on a thorough review of existing literature and best practices in the field, tailoring it to address the unique capabilities and challenges presented by generative AI especially LLMs. This framework was designed not only to assess technical performance but also to evaluate the broader applicability, robustness, and integrity of LLM benchmarks.
A. Applying the People, Process, Technology (PPT) Frame-
# A. Applying the People, Process, Technology (PPT) Framework
The People, Process, Technology (PPT) framework, widely used in cybersecurity, is also well-suited for evaluating LLM benchmarks due to its holistic approach, addressing not just technical factors, but also human and procedural elements [37]. While other frameworks (e.g., the Information Systems Success Model [38]) focus on narrower aspects, PPT is uniquely comprehensive, allowing for an in-depth assessment across multiple dimensions. This ensures a thorough evaluation of benchmarks, which require more than just technical soundness; they must also integrate human expertise and follow robust, adaptable processes. The PPT framework aligns with the core challenges of LLM benchmarking. Benchmarks involve complex interactions between developers, evaluators, and endusers (People), they must follow consistent and adaptable methodologies to remain relevant and resilient (Process), and they depend on scalable and reliable technical infrastructures (Technology) to accurately assess evolving LLM capabilities. The success of an LLM benchmark is not only in its ability to test specific tasks but also in how well it adapts to new challenges, handles real-world use cases, and resists manipulation or overfitting.
• People: The expertise and diversity of developers, evaluators, and end-users influence the validity and applicability of benchmark results. A lack of diversity in evaluators could bias results, making benchmarks less reflective of LLMs’ real-world performance across varied scenarios. • Process: Standardization and adaptability in benchmark construction and implementation are essential. Without well-defined processes, benchmarks may become outdated, inconsistent, or vulnerable to LLMs being optimized solely for specific test sets rather than demonstrating true capability. • Technology: The technical infrastructure supporting benchmarks, including algorithms, datasets, and evaluation metrics, must be scalable and robust. Technical limitations can impede the benchmark’s ability to provide accurate assessments, while ensuring the infrastructure’s scalability is critical as LLMs evolve and grow more complex. The PPT framework thus provides a holistic lens to comprehensively evaluate LLM benchmarks, addressing both functionality and integrity. By integrating people, process, and technology, this framework allows for a more nuanced assessment compared to models that may overlook key aspects of LLM benchmarking. The PPT approach aligns with the need for multidisciplinary evaluation methods in complex AI systems, as highlighted in recent literature [3], [31]. Leveraging the PPT framework ensures our evaluation encompasses all necessary facets to produce reliable and meaningful insights into LLM performance and benchmarking practices.
# B. Functionality and Integrity in LLM Benchmarks
When evaluating LLM benchmarks, two core aspects must be considered: functionality and integrity. Functionality assesses how well a benchmark measures the specific capabilities of LLMs, while integrity examines the benchmark’s resistance to manipulation or gaming by models that may exploit its criteria to achieve misleading results. Evaluating both functionality and integrity is essential to ensure that benchmarks provide a comprehensive and authentic assessment of LLMs, free from bias or manipulation. • Functionality Assessment: This determines if the benchmark accurately evaluates the intended capabilities of an LLM, such as reasoning, language comprehension, or multimodal integration. For instance, a benchmark should be designed to test the model’s true understanding rather than its ability to mimic expected answers. A counterexample is using benchmarks that rely on superficial elements like keyword matching, which can lead to overestimating an LLM’s language understanding when it simply exploits patterns without grasping the content. Functionality also considers the alignment of the benchmark with real-world applications, ensuring it tests skills relevant to practical use rather than artificial metrics. • Integrity Considerations: This involves assessing the robustness of the benchmark against potential exploitation
by models tuned to optimize for specific evaluation metrics. For example, a benchmark lacking integrity might be susceptible to LLMs that achieve high scores through memorization of training data or by exploiting specific test set characteristics. An integrity breach occurs when LLMs appear to excel in benchmarks through these surface-level optimizations without demonstrating genuine capability. Ensuring integrity means the benchmark cannot be easily manipulated and that it provides a true reflection of an LLM’s performance, independent of prior exposure to similar tests or over-fitting strategies.
# C. Data Collection
We conducted a Structured Literature Review (SLR) to systematically identify relevant LLM benchmarks. Our focus was on benchmarks explicitly designed for evaluating large language models, particularly those with multimodal capabilities. The search encompassed peer-reviewed publications, prominent technical preprints, and widely cited benchmarking platforms up to October 2023. • Inclusion Criteria: – Benchmarks explicitly designed for LLM evaluation. – Benchmarks covering a wide range of generative AI tasks, including language understanding, reasoning, coding, legal analysis, medical question answering, and tool usage. – Benchmarks with substantial academic or technical adoption, indicated by citation counts or recognition in the AI community. • Exclusion Criteria: – Benchmarks that are too niche or focus on specialized tasks not representative of general LLM capabilities. – Benchmarks lacking sufficient methodological detail for thorough evaluation.
# • Exclusion Criteria:
– Benchmarks that are too niche or focus on specialized tasks not representative of general LLM capabilities. – Benchmarks lacking sufficient methodological detail for thorough evaluation.
# D. Data Analysis
We applied thematic analysis to systematically evaluate the selected benchmarks, focusing on identifying patterns of inadequacies in functionality and integrity. Our analysis aimed to determine whether the benchmarks accurately measured LLM capabilities without relying on superficial metrics or presentation nuances, and whether they could be manipulated by LLMs exploiting specific evaluation criteria. Our methodology involved the following steps: 1) Identification of Inadequacies: We critically reviewed each benchmark to identify potential inadequacies related to functionality and integrity, such as response variability, susceptibility to overfitting, and lack of consideration for linguistic diversity. 2) Coding and Categorization: We coded the identified inadequacies and categorized them based on common themes, aligning with the domains of people, process, and technology as outlined in our unified evaluation framework.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d16c/d16cbd15-7d98-48f5-ad64-b9bc82dd352e.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 1: Evaluation Flowchart for LLM Benchmarks</div>
3) Assessment of Prevalence: We defined prevalence as the number of benchmarks (out of 23) exhibiting each specific inadequacy (except those resolved or not present). This metric quantifies the extent to which each inadequacy is present across the benchmarks. 4) Evaluation of Benchmark Responses: For each identified inadequacy, we examined whether it was acknowledged or addressed by the benchmark creators. We categorized the benchmarks accordingly: • Present and Unacknowledged (marked as ✓): The inadequacy exists without recognition. • Acknowledged but Unresolved (marked as △): The inadequacy is recognized but not yet addressed. • Considered Addressed (marked as ×): The inadequacy has been recognized and mitigated. The evaluation process is illustrated in Fig. 1, which guided our systematic assessment of each benchmark’s robustness and reliability.
# IV. OVERVIEW OF LLM BENCHMARKS
In this section, we list the overview 23 LLM benchmarks we surveyed, including what they have achieved, and our preliminary findings.
Applying such criteria of our Unified Evaluation Framework, we selected 23 LLM benchmarks that represented a diverse set of tasks and domains, ensuring comprehensive coverage of the LLM evaluation landscape. Table I provides an overview of these benchmarks and their characteristics, categorized into general-purpose benchmarks and specializeddomain benchmarks. General-purpose benchmarks primarily assess broad knowledge and reasoning across multiple domains. MMLU [39] from UC Berkeley focused on evaluating language understanding across diverse subjects using multiple-choice questions (MCQs) with zero-shot and few-shot evaluation methodologies. The benchmark used human exam data to test the general knowledge and reasoning abilities of LLMs in English. Chain-of-Thought Hub [40] from the University of Edinburgh used curated reasoning benchmarks to assess reasoning performance in LLMs. The benchmark featured diverse reasoning tasks in both English and Simplified Chinese, using few-shot chain-of-thought prompting and final answer accuracy as evaluation criteria. KoLA [11] from Tsinghua University focused on world knowledge evaluation using tasks related to memorization, understanding, applying, and creating knowledge. It employed standardized overall scoring and a self-contrast metric for assessing knowledge creation, using known and evolving data sources. ARB [41] from DuckAI and Georgia Tech evaluated advanced reasoning in LLMs using standardized tests and problem books. The benchmark involved MCQs, short-answer, and open-response questions, with task-specific instructions and both automatic and manual evaluation methods. Xiezhi [42] from Fudan University assessed multidisciplinary domain knowledge across 516 disciplines in English and Simplified Chinese. It used MCQs with zero-shot and few-shot evaluation, accuracy measurement, and automated test accuracy reporting. BIG-bench [6] by Google focused on general knowledge, with tasks evaluated through human annotation. It used both algorithms and human raters to assess performance on diverse tasks in English. AGIEval [12] from Microsoft assessed human-centric reasoning tasks using official public and high-standard exams in English and Simplified Chinese. It evaluated models with MCQs and fill-inthe-blank tasks, using zero-shot and few-shot methods, chainof-thought reasoning, and both quantitative and qualitative analysis. HELM [31] from Stanford University compiled a benchmark collection to evaluate LLMs based on a metricsdriven approach in English. It covered 16 scenarios and 7 metrics, focusing on multi-metric measurement and dense evaluation across different tasks. PromptBench [7] from Microsoft focused on general knowledge, benchmarking LLMs using a collection of APIs, datasets, and models. It employed quick performance assessments, dynamic evaluations, and semantic evaluations. C-Eval [43] from Shanghai Jiao Tong University focused on multilevel, multidiscipline knowledge evaluation in Simplified Chinese using mock and high-standard exams. It employed MCQs with zero-shot and few-shot evaluation, accuracy measurement, and automated test reporting. In contrast, specialized benchmarks target domain-specific
LLM capabilities. HumanEval [8] from OpenAI was designed to assess the functional correctness of code synthesis from docstrings. It utilised programming problems with unit tests sourced from GitHub, focusing on evaluating LLMs’ ability to generate accurate Python functions in English. LegalBench [14] by Stanford University created a benchmark for legal reasoning using manually constructed tasks. It employed IRAC (Issue, Rule, Application, Conclusion) and classification tasks, using performance evaluations and comparative analysis across models to assess their legal knowledge. FLUE [44] from Georgia Institute of Technology developed a benchmark for financial sentiment analysis and news headline classification using datasets like Financial PhraseBank and FiQA 2018. It employed sentiment analysis through regression and classification tasks to evaluate the performance of LLMs on financial data in English. MultiMedQA [10] from Google evaluated medical question answering through a collection of medical datasets such as MedQA, MedMCQA, and PubMedQA. It used MCQs, few-shot prompting, self-consistency, and human evaluation to test LLMs’ performance in medical contexts. M3KE [9] from Tianjin University evaluated multilevel, multisubject knowledge using official exams and educational materials in Simplified Chinese. It utilised MCQs with zero-shot and fewshot evaluation methodologies, accompanied by comparative analysis across models. T-Bench [45] by SambaNova Systems Inc. focused on the tool manipulation capabilities of LLMs using empirical analysis of open-source models. It evaluated performance across diverse software tools and conducted comparative analysis of models’ capabilities in real-world scenarios. SciBench [46] from UCLA evaluated LLMs’ ability to solve scientific problems using open-ended questions based on college-level textbooks and exams. It compared model outputs with correct answers and applied human-verified solutions graded by instructors’ rubrics. ToolAlpaca [47] from the Chinese Academy of Sciences examined generalized tool learning in LLMs through multi-agent simulations. It evaluated performance on simulated tool-use instances across multiple categories, focusing on unseen tools and the diversity of the toolset. ToolBench [48] from Tsinghua University tested LLM performance in API and tool-augmented tasks. It utilized curated real-world APIs across various scenarios and assessed API retrieval accuracy, task performance, and generalization using both ground-truth and generated APIs. AgentBench [49] from Tsinghua University examined LLMs in agent performance across interactive tasks in English and Simplified Chinese. The benchmark involved diverse tasks in code, game, and web environments, evaluating LLMs across multiple environments through comparative analysis. APIBank [50] from Alibaba Group evaluated tool-augmented LLM performance in English using synthetic conversational dialogues. It measured the accuracy of API calls and response quality through the ROUGE-L metric. BOLAA [51] from Salesforce Research examined autonomous agent orchestration in simulated environments. It focused on decision-making and compared the performance of orchestrated agent architectures through quantitative analysis of agent interactions. HaluEval [52] from Renmin University of China investigated LLM hallucinations in English using ChatGPT-generated and human-
annotated samples. The evaluation involved QA, dialogue, and summarization, relying on API-based automatic assessments of provided answers.
# B. Preliminary Findings Overview
Our analysis, as summarised in Table II, revealed widespread inadequacies across LLM benchmark studies, indicating the need for more refined and comprehensive evaluation practices to better assess the true capabilities and limitations of large language models (LLMs). A key observation is that many benchmarks predominantly focus on English, with limited representation of other languages, such as Simplified Chinese, and frequently neglect cultural and contextual variations in their questions and answers. This raises concerns about the generalizability of these benchmarks across diverse linguistic and cultural settings. For instance, benchmarks often assume singular correct answers in culturally nuanced questions, overlooking alternative valid responses from different cultural or religious perspectives, which limits the inclusivity of these evaluations. Another critical issue is that current benchmarks generally adopt task-based formats, such as Multiple Choice Questions (MCQs) and dialogue-based evaluations, which tend to be static and do not capture the evolving nature of human-AI interactions. Real-world LLM usage often involves continuous dialogues, yet many benchmarks assess only the first attempt of an LLM response, without considering the consistency and coherence of answers across multiple interactions. This focus on isolated responses reduces the relevance of these benchmarks in evaluating models designed for dynamic, ongoing interactions. Our review also found that only 6 of the 23 surveyed benchmark studies ( [7], [10], [31], [39], [44], [52]) were peerreviewed at the time of this article, reflecting the early-stage nature of research in this domain. While preprints provide valuable insights, the lack of rigorous peer review raises questions about the scientific validity and reproducibility of many LLM benchmark results. Moreover, the speed of LLM output generation—a crucial factor for user experience in real-time applications—is frequently overlooked in current benchmarks, which tend to focus solely on the qualitative correctness of generated answers. Perhaps most concerning, many benchmarks failed to account for the possibility that LLMs can optimize their responses specifically to perform well on standardized tests, rather than genuinely demonstrating deep understanding or reasoning. This risk of ”benchmark gaming” undermines the integrity of evaluations, as models may be engineered to exploit the structure of the test rather than showcasing their full capabilities across diverse tasks. Given the rapid advancement of LLMs, it is imperative to develop evaluation practices that measure genuine reasoning skills and adaptability, rather than technical optimization for specific test formats. Our critique is grounded in published opinions and supported by our evaluative rationale, detailed in Sections V, VI, and VII. The aim of this analysis is not to elevate any specific benchmark but to highlight common limitations that many benchmarks share. By addressing these issues, we seek to encourage the development of more robust,
<div style="text-align: center;">ABLE I: Comparison of Various Generative AI and LLM Benchmarks (ranked by chronological publication dates)</div>
Name
Main Affiliation
Data Source
Focus Area
Language
Benchmark Format
Evaluation Methodology
MMLU [39]
UC Berkeley
Human exams
Language understanding
English
MCQs
Zero-shot and few-shot evaluation
HumanEval [8]
OpenAI
GitHub code
Code Synthesis from
Docstrings
English
Programming problems
with unit tests
Functional correctness via unit tests
LegalBench [14]
Stanford
University
Manual task
construction
Legal reasoning
English
IRAC & classification
tasks
Task performance evaluation,
comparative analysis across models
FLUE [44]
Georgia Institute
of Technology
Financial PhraseBank,
FiQA 2018, Gold news
headline dataset
Financial sentiment
analysis, news
headline classification
English
Sentiment analysis
(regression, classification)
Evaluation on datasets
MultiMedQA [10]
Google
MedQA, MedMCQA,
PubMedQA, etc.
Medical question
answering
English
MCQs
Few-shot prompting, self
-consistency, human evaluation
M3KE [9]
Tianjin
University
Official exams and
educational materials
Multilevel, multisubject
knowledge evaluation
Simplified Chinese
MCQs
Zero-shot and few-shot evaluation,
comparative analysis across models
T-Bench [45]
SambaNova
Systems Inc.
Empirical analysis of
open-source LLMs
Software tool
manipulation capability
English
Diverse software tools
Comparative analysis across models,
performance evaluation
Chain-of-Thought
Hub [40]
University of
Edinburgh
Curated reasoning
benchmarks
Reasoning performance
English,
Simplified Chinese
Diverse reasoning tasks
Few-shot chain-of-thought prompting,
Final answer accuracy evaluation
KoLA [11]
Tsinghua
University
Known and evolving
data sources
World knowledge
English
Knowledge memorization,
understanding, applying,
and creating tasks
Standardized overall scoring,
self-contrast metric for
knowledge creation
SciBench [46]
UC LA
College-level
textbooks and
course exams
Scientific problems
English
Open-ended questions
Comparison with correct answers;
graded by instructors’ rubrics;
human-verified solutions
ARB [41]
DuckAI &
Georgia Tech
Standardized tests,
problem books
Advanced reasoning
English
MCQs, short answer,
open response questions
Task-specific instructions,
automatic and manual evaluation
Xiezhi [42]
Fudan University
Comprehensive domain
knowledge evaluation
Multidisciplinary,
516 disciplines
English,
Simplified Chinese
MCQs
Zero-shot and few-shot evaluation,
accuracy measurement,
automated test accuracy reporting
BIG-bench [6]
Google
Human annotation
General knowledge
English
Tasks
Algorithm and human raters
AGIEval [12]
Microsoft
Official public and
high-standard exams
Human-centric reasoning
tasks
English,
Simplified Chinese
MCQs,
fill-in-the-blank
Zero-shot and few-shot evaluation,
chain-of-thought reasoning,
quantitative and qualitative analysis
ToolAlpaca [47]
Chinese
Academy
of Sciences
Multi-agent simulation
Generalized tool learning
for language models
English
Simulated tool-use
instances in
multiple categories
Performance evaluation on
unseen tools, diversity in toolset
HELM [31]
Stanford
University
Benchmark collection
Metrics-based evaluation
English
16 Scenarios, 7 metrics
Multi-metric measurement, Dense
evaluation of scenarios and metrics
ToolBench [48]
Tsinghua
University
Curated from 16,000+
real-world APIs using
ChatGPT
API and tool-augmented
LLM performance
English
API-based tasks in
various scenarios
Assessment of API retrieval accuracy,
task performance, and generalization
capability using both ground-truth
and generated APIs
PromptBench [7]
Microsoft
Benchmark collection
General knowledge
English
APIs, datasets, models
Quick performance assessment,
dynamic evaluation, semantic
evaluation
AgentBench [49]
Tsinghua
University
Code, game, and web
environments
Agent performance
in interactive tasks
English,
Simplified Chinese
Diverse interactive tasks
across environments
Evaluation across multiple environments,
comparative analysis of LLMs as agents
APIBank [50]
Alibaba Group
Synthetic dialogues
Tool-augmented LLM
performance
English
Conversational dialogues
Accuracy of API calls and response
quality measured by ROUGE-L metric
C-Eval [43]
Shanghai
Jiao Tong
University
Mock exams, high
-standard exams
Multilevel, multidiscipline
knowledge evaluation
Simplified Chinese
MCQs
Zero-shot and few-shot evaluation,
Accuracy measurement,
Automated test accuracy reporting
BOLAA [51]
Salesforce
Research (USA)
Simulated
environments
Autonomous agent
orchestration
English
Decision-making
Performance comparison of orchestrated
agent architectures, quantitative
analysis of agent interactions
HaluEval [52]
Renmin
University
of China
ChatGPT-generated
and human-annotated
samples
LLM hallucinations
English
QA, dialogue, and
summarization
API-based automatic evaluation
according to provided answers
transparent, and representative benchmarks that better reflect the full range of LLM capabilities. The following sections provide an in-depth exploration of these findings, categorized into technological aspects, processual elements, and human dynamics.
# V. TECHNOLOGICAL ASPECTS
In this section, we examine the technological intricacies and limitations inherent in current LLM benchmarks (Fig. 2).
# A. Response Variability in Standardized Evaluations
Prevalence: 22/23 A significant inadequacy in LLM benchmarks is the response variability under standardized evaluations, especially when these benchmarks fail to account for LLMs tailored to specific formats or use cases. Despite the breadth of frameworks like [6]–[12], [14], [31], [39]–[41], [43]–[52], they often overlook the subtle behaviors of LLMs designed for particular scenarios (Appendix A-A). Benchmarks that
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0361/0361b0b5-7a84-452b-bad3-3c169f24a363.png" style="width: 50%;"></div>
<div style="text-align: center;">Prevalence (max 23)</div>
Fig. 2: Technological Inadequacies in LLM Benchmarking
<div style="text-align: center;">TABLE II: Prevalence of LLM Benchmark Inadequacies (✓: present and unacknowledged; △: acknowledged but unresolved; ×: resolved or not present)</div>
Benchmark
Research
Technological Aspects (Sec V)
Processual Elements (Sec VI)
Human Dynamics (Sec VII)
Response Variability in
Standardized Evaluations (Sec V-A)
Genuine Reasoning vs Technical
Optimization (Sec V-B)
Tension Between Helpfulness
and Harmlessness (sec:V-C)
Linguistic Variability and
Embedded Logic Diversity (Sec V-D)
Benchmark Installation
and Scalability (Sec V-E)
Biases in LLM-Generated
LLM Evaluations (Sec V-F)
Inconsistent Benchmark
Implementation (Sec VI-A)
Slow Test Iteration time (Sec VI-B)
Challenge of Proper Prompt
Engineering (Sec VI-C)
Diversity in Human Curators
and Evaluators (Sec VII-A)
Diverse Cultural, Social,
Political, Religious and
Ideological Norms (Sec VII-B)
MMLU [39]
✓
✓
△
△
✓
×
✓
✓
△
△
△
HumanEval [8]
✓
✓
✓
✓
×
×
△
×
×
✓
×
LegalBench [14]
✓
✓
✓
✓
✓
×
△
×
△
△
✓
FLUE [44]
✓
✓
✓
✓
×
×
✓
✓
✓
×
✓
MultiMedQA [10]
✓
△
✓
△
×
×
△
✓
△
△
△
M3KE [9]
✓
✓
✓
✓
✓
✓
✓
×
✓
✓
△
T-Bench [45]
✓
✓
✓
✓
✓
✓
×
△
△
✓
×
Chain-of-Thought Hub [40]
△
✓
✓
✓
×
×
✓
✓
△
✓
✓
KoLA [11]
✓
×
✓
△
✓
×
△
✓
×
✓
△
SciBench [46]
✓
✓
✓
×
✓
×
△
△
×
△
✓
ARB [41]
△
✓
✓
×
✓
✓
△
△
△
×
✓
Xiezhi [42]
×
✓
×
✓
✓
✓
✓
×
×
△
△
BIG-Bench [6]
✓
✓
×
×
✓
✓
×
✓
△
△
✓
AGIEval [12]
✓
✓
✓
△
✓
×
✓
✓
×
△
✓
ToolAlpaca [47]
✓
✓
×
✓
✓
✓
✓
✓
×
△
×
HELM [31]
△
✓
✓
△
×
×
△
△
△
×
△
ToolBench [48]
✓
✓
✓
✓
×
×
△
×
×
✓
×
PromptBench [7]
✓
△
✓
×
✓
×
✓
✓
×
✓
✓
AgentBench [49]
✓
✓
✓
×
✓
✓
✓
✓
△
✓
✓
APIBank [50]
✓
✓
×
△
✓
✓
×
✓
×
✓
✓
C-Eval [43]
△
✓
✓
✓
×
×
×
✓
✓
✓
△
BOLAA [51]
✓
×
×
×
×
×
×
×
△
×
×
HaluEval [52]
×
×
×
×
×
×
△
△
△
△
×
standardize formats without considering context-specific requirements can inadvertently skew the perceived functionality of these LLMs. For example, minor formatting changes in prompts, such as altering choice indicators from (A) to [A] or adding extra spaces, have been shown to shift response accuracy by approximately 5%, highlighting the LLMs’ sensitivity to superficial input variations [53]. From a functionality perspective, this variability raises concerns about the accuracy of these benchmarks in measuring true model capabilities. If benchmarks fail to reflect the intended application context, they may mislead users and developers regarding the model’s practical performance, influencing deployment strategies inappropriately [3]. Integrity concerns also arise when standardized benchmarks become predictable and exploitable, allowing LLMs to achieve artificially high scores through superficial optimization rather than genuine comprehension. This vulnerability to input sensitivity indicates a lack of robustness in the benchmarks, which could be exploited in practical applications, posing risks to system reliability and security [54]. To address such issues, LLM benchmark designs must be refined to accommodate the specificities of model architecture and intended use cases, ensuring they accurately capture the capabilities of LLMs in varied contexts. This approach will enhance the functionality and integrity of the
evaluations, reducing the potential for misrepresentation and gaming of results, and ultimately leading to a more authentic assessment of LLM capabilities.
# B. Genuine Reasoning vs Technical Optimization
Prevalence: 22/23 The challenge in distinguishing between responses driven by genuine reasoning and those resulting from technical optimization, such as overfitting to benchmark answers, was seen in [6]–[10], [12], [14], [31], [39]–[52] (Appendix A-B), due to the opaque nature of LLMs, where the mechanisms behind their outputs are often not transparent. Consequently, LLMs can appear to demonstrate advanced reasoning when, in reality, they may simply be leveraging specific benchmark characteristics or exploiting superficial patterns. For example, an LLM trained on data that includes benchmarklike questions may produce correct answers by recognizing patterns rather than understanding the underlying concepts [53], [55]. For example, HumanEval [8] evaluated Codex’s performance through functional correctness on programming problems, which might not fully capture the model’s reasoning capabilities across diverse real-world scenarios. Similarly, LegalBench [14] applied generic benchmarks to specialized
legal reasoning tasks, potentially allowing LLMs to optimize for benchmark-specific patterns rather than demonstrating true legal comprehension that could include court negotiations. From a functionality perspective, this ambiguity undermines the benchmarks’ ability to accurately assess LLMs’ reasoning capabilities. If a benchmark fails to differentiate between true reasoning and optimization, it risks overestimating the model’s practical utility and cognitive abilities. A critical insight here is that benchmarks should be designed to probe deeper understanding rather than surface-level pattern recognition, ensuring that LLMs cannot succeed merely by exploiting familiar test elements. Regarding integrity, the inability to distinguish genuine reasoning from optimization raises concerns about the potential for LLMs to manipulate benchmark outcomes. LLMs that can achieve high scores through memorization or strategic over-fitting challenge the validity of the evaluation process. This issue is particularly problematic when benchmarks are inadvertently included in training data, allowing LLMs to produce artificially inflated results without demonstrating true competence [55], [56]. Addressing this requires designing benchmarks that are resistant to such exploitation, ensuring they accurately reflect the LLM’s ability to engage in authentic reasoning processes rather than optimized mimicry. To mitigate such challenges, benchmark designs must evolve to include tasks that require dynamic reasoning and adaptability, reducing the risk of LLMs gaming the system through technical optimizations. Continuous refinement and innovation in benchmark methodologies are essential to differentiate genuine reasoning from mere pattern exploitation, thereby providing a more reliable assessment of LLM capabilities.
# C. Tension Between Helpfulness and Harmlessness
Prevalence: 19/23 The tension between helpfulness and harmlessness in LLM responses presents a significant challenge in benchmark evaluations, seen in [7]–[12], [14], [31], [39]–[41], [43]–[46], [48], [49], [51], [52] (Appendix A-C). Human evaluations, such as A/B tests, often struggle to balance those two aspects, particularly in open-ended dialogues. For instance, MultiMedQA [10] assessed Flan-PaLM’s performance in medical question answering, revealing that while the model could provide accurate information, it occasionally failed to align with clinical consensus, indicating a struggle to balance helpfulness with the necessity of harmlessness in sensitive medical contexts. Similarly, LegalBench [14] evaluated LLMs on legal reasoning tasks, where the models needed to deliver precise legal information without overstepping into providing authoritative legal advice, thereby maintaining a balance between being helpful and avoiding potential legal ramifications. Such difficulty arises from the need to provide information without causing harm or controversy, a balance that is particularly delicate when addressing sensitive topics. From a functionality perspective, LLM benchmarks must accurately measure an LLM’s ability to provide informative yet harmless responses. This requires sophisticated evaluation criteria that distinguish between helpfulness in providing useful information and harmlessness in avoiding misinformation
or offensive content. Benchmarks that fail to account for this balance risk either over-penalizing LLMs for providing valuable but sensitive information or underestimating the potential for harm in their responses [3]. An insightful example is the evaluation of LLMs in the context of cybersecurity advice, where helpfulness could involve explaining malware analysis techniques, but doing so without risking the dissemination of potentially harmful information. Regarding integrity, benchmarks must be robust against LLMs that could exploit this tension to appear more capable [28]. For instance, LLMs might evade challenging inquiries by refusing to respond, prioritizing harmlessness over helpfulness to avoid any potential controversy. This behavior could lead to a misleading assessment of the model’s abilities, giving the impression of ethical consideration when it may simply be avoiding complexity. To ensure integrity, benchmarks should test for situations where LLMs can demonstrate both helpfulness and harmlessness without compromising one for the other. Addressing this inadequacy requires developing benchmarks that incorporate high-level norms and values, ensuring a comprehensive evaluation of how LLMs navigate this tension.
# D. Linguistic Variability and Embedded Logic Diversity
Prevalence: 17/23 Current LLM benchmarks often fail to account for the linguistic and logical diversity inherent in different languages, leading to an inadequate evaluation of LLMs’ true capabilities, as seen in [8]–[12], [14], [31], [39], [40], [42]–[45], [47], [48], [50], [52] (Appendix A-D). Predominantly English-centric benchmarks overlook the sophisticated reasoning patterns embedded within languages, when language structure can shape distinct cognitive processes [28], [57]. This lack of consideration results in benchmarks that wrongfully assume a uniform cognitive framework, failing to capture how LLMs handle linguistic variations across different cultures and languages [13]. From a functionality perspective, this inadequacy undermines the benchmarks’ ability to evaluate LLMs in a multilingual context accurately. By not reflecting the cognitive diversity inherent in language structures, these benchmarks may misrepresent a model’s comprehension and reasoning capabilities. For instance, an LLM evaluated solely on Englishcentric benchmarks might appear proficient but could struggle to maintain accuracy and context in languages with different syntactic and semantic frameworks. This bias risks overestimating a model’s universality and applicability across languages. In terms of integrity, the oversight in linguistic variability can be exploited, as uneven moderation across languages may allow harmful or prohibited content to bypass detection in less moderated languages [35]. A model might score well on benchmarks in its primary language but fail to uphold the same standards in others, potentially leading to security vulnerabilities. Besides, a blend of different languages could sometimes be used to jailbreak LLMs without triggering LLM safety mechanisms like RLHF [28]. Addressing such challenges requires benchmarks that recognize and adapt to linguistic diversity, testing LLMs on native language capabilities rather than relying on translation-based evaluations.
# E. Benchmark Installation and Scalability
Prevalence: 16/23 A key inadequacy in current LLM benchmarks is the challenge of installation and scalability, affecting their functional utility and integrity [6], [7], [9], [11], [12], [14], [39], [41], [42], [45]–[47], [49]–[52] (Appendix A-E). Many benchmarks demand considerable engineering effort to install and adapt to different computational environments, hindering broad accessibility and consistent evaluation across diverse LLMs. For example, SciBench [46] involved manual data extraction and complex problem formulations, further complicating the installation process. Moreover, scaling such benchmarks to handle larger LLMs or more extensive datasets often requires substantial infrastructure and resources, as reported by Anthropic [53], posing a barrier to comprehensive and efficient assessments. For instance, APIBank [50] involved the complex implementation of 73 APIs, demanding extensive engineering efforts and resources to scale effectively. From a functionality standpoint, these technical hurdles delay and complicate the evaluation process, potentially limiting the ability to accurately measure LLM performance. For instance, if a benchmark cannot be efficiently scaled, it may fail to test an LLM’s capabilities under more demanding or realistic scenarios, resulting in an incomplete functional assessment. This limitation can lead to biased evaluations that do not fully represent an LLM’s operational capacities, especially for larger LLMs requiring more intensive testing environments. In terms of integrity, the difficulty in installing and scaling benchmarks can create inconsistencies in how LLMs are evaluated, opening the door to manipulation. If a benchmark is too complex to implement consistently, results may vary between environments, allowing LLMs to be selectively tested in conditions favorable to them. This lack of standardization can compromise the benchmark’s resistance to gaming by LLMs, undermining the reliability of its evaluation metrics. Improving the installation and scalability of benchmarks is crucial for ensuring that evaluations are both comprehensive and resistant to exploitation. Streamlined, welldocumented benchmarks that can be easily deployed across different systems will enhance both the functional assessment of LLMs and the integrity of the evaluation process, enabling fairer and more accurate comparisons of model performance.
# F. Biases in LLM-Generated LLM Evaluations
Prevalence: 9/23 An emerging concern in LLM benchmarks is the use of LLMs themselves to generate evaluation tasks or assess other LLMs, effectively acting as both creator and judge [6], [9], [41], [42], [45], [47], [49], [50], [52] (Appendix A-F). This practice introduces the risk of amplifying inherent biases and inaccuracies present in the evaluating LLMs, which can significantly undermine the benchmark’s functionality and integrity. For instance, M3KE [9] utilized ChatGPT-generated multiple-choice questions to assess Chinese LLMs, inherently incorporating the biases of ChatGPT. From a functionality standpoint, such biases can distort the evaluation process, leading to inaccurate assessments of LLM
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/935b/935b8901-a0c7-43cc-b115-6c55850ae5ec.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/af02/af022a3b-d926-4ecd-876b-ea03652509c7.png" style="width: 50%;"></div>
Challenge of Proper Prompt Engineering Slow Test Iteration time Inconsistent Benchmark Implementation
<div style="text-align: center;">Prevalence (max 23)</div>
Fig. 3: Processual Inadequacies in LLM Benchmarking
<div style="text-align: center;">Fig. 3: Processual Inadequacies in LLM Benchmarking</div>
capabilities. For instance, an LLM-generated benchmark might inadvertently favor certain types of responses or reasoning patterns, resulting in an overestimation or underestimation of the evaluated model’s performance [1], [3]. This issue raises concerns about the benchmark’s ability to provide a true measure of an LLM’s diverse capabilities, potentially skewing its alignment with real-world applications. Regarding integrity, relying on LLMs for evaluation introduces the risk of circular reasoning, where the biases inherent in the LLMs generating the benchmarks could lead to evaluations that are not truly independent. This undermines the benchmark’s resistance to manipulation, as LLMs could exploit these biases to achieve misleadingly high scores. For example, if an LLM-generated benchmark unintentionally reinforces specific patterns or knowledge it was trained on, it may allow LLMs to perform well through pattern recognition rather than demonstrating genuine understanding or reasoning [56], [58]. To ensure both functionality and integrity, a more balanced approach that integrates human expertise is necessary. Human involvement can mitigate the biases introduced by LLMgenerated content, ensuring a more rigorous and objective evaluation process. This hybrid approach would enhance the benchmark’s ability to provide a comprehensive and unbiased assessment of LLM capabilities, ensuring that evaluations remain reliable and reflective of true model performance.
# VI. PROCESSUAL ELEMENTS
This section looks into the various process-related challenges and intricacies inherent in the implementation and evaluation of LLM benchmarks (Fig. 3). Please note that we use the word “Processual” to refer to the overall nature and development of processes, instead of “procedural” which focuses on specific steps or methods.
# A. Inconsistent Benchmark Implementation
Prevalence: 18/23 A critical issue identified in LLM benchmarks is the inconsistency in implementation across different research teams, as observed in [7]–[12], [14], [31], [39]–[42], [44], [46]– [49], [52] (Appendix B-A). Variations in the interpretation and execution of benchmarks, including differences in fewshot learning setups or chain-of-thought prompting, have led to diverse results that challenge the uniformity and reliability of these evaluations. For example, benchmarks like MMLU showed significant score variations due to differing implementation strategies [53]. Similarly, M3KE [9] required substantial
customization to accommodate language-specific subtleties, leading to inconsistencies in assessing Chinese LLMs across different research environments. Functionally, this inconsistency hampers the benchmarks’ ability to provide a uniform, objective assessment of LLMs. The lack of standardization in implementation can result in skewed outcomes, making it difficult to compare LLMs fairly or derive meaningful insights into their performance [3], [53]. This undermines the benchmarks’ functionality, as the results become less reliable for comparative analysis or performance tracking across different LLMs. From an integrity perspective, such inconsistencies can introduce vulnerabilities in the evaluation process. The absence of standardized implementation protocols opens up opportunities for manipulation or biased interpretations, where variations in methodology could be exploited to produce favorable outcomes without reflecting the model’s genuine capabilities [55], [56]. This compromises the benchmark’s integrity, as differing approaches might obscure true performance or create loopholes that can be exploited. To address such concerns, establishing standardized protocols and guidelines for benchmark implementation is essential. A uniform approach would enhance the consistency and reliability of LLM evaluations, ensuring that benchmarks provide a fair and accurate reflection of model capabilities.
# B. Slow Test Iteration Time
Prevalence: 18/23 Slow iteration time is a key inadequacy in LLM benchmarks, particularly in comprehensive third-party frameworks like BIG-bench and HELM [6], [7], [10]–[12], [31], [39]–[41], [43]–[47], [49]–[52] (Appendix B-B). These frameworks often involve extensive evaluation processes, which can span weeks or months, hindering timely feedback on new LLMs. External involvement in evaluations adds further delays, requiring coordination and engineering support, as seen in benchmarks like BIG-bench [6] and HELM [31]. The need for comprehensive testing across diverse scenarios inherently extends evaluation time. Functionally, prolonged iteration times hinder the benchmarks’ ability to provide current insights into LLM performance. For instance, SciBench [46] involved manual data extraction and complex problem formulations, which significantly extended the evaluation process, making it difficult to keep pace with the rapid advancements in LLM development. With AI advancing rapidly, an LLM may undergo significant updates during its evaluation period, potentially rendering the results outdated or irrelevant by the time they are published [2], [3]. From an integrity perspective, slow iteration times pose security risks. Delayed evaluations mean emerging threats or vulnerabilities in LLMs may go undetected for extended periods, exposing systems to potential exploitation [54], [59]. For example, ToolBench [48] integrated LLMs with a vast number of software APIs, requiring extensive manual configuration and adaptation, which could delay the identification and mitigation of security vulnerabilities. Additionally, the lag in benchmark feedback to LLMs would limit the ability to promptly address identified issues, allowing weaknesses to
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/181e/181e8f88-3a05-4f28-b948-cad52e0a7c15.png" style="width: 50%;"></div>
Diverse Cultural, Social, Political, Religious and Ideological Norms Diversity in Human Curators and Evaluators
# Fig. 4: Human Dynamics Inadequacies in LLM Benchmarking
<div style="text-align: center;">Fig. 4: Human Dynamics Inadequacies in LLM Benchmarking</div>
persist longer than necessary. To mitigate such challenges, streamlining evaluation processes and improving coordination with external parties is crucial. However, aligning the need for thorough evaluation with the rapid development pace of AI remains a complex issue, requiring innovative solutions to balance depth and timeliness in LLM benchmarking.
C. Challenge of Proper Prompt Engineering in Benchmarking Prevalence: 14/23 Prompt engineering plays a crucial role in evaluating LLMs, as it shapes the interaction between the model and the test. While exploring different prompts is valuable for understanding LLM behavior, in the context of benchmarking, it is important to ensure that prompts are designed to accurately and consistently assess the LLM’s capabilities. Challenges can arise when variations in prompt formulation lead to significant differences in model performance, which can complicate the interpretation of benchmark results [6], [9], [10], [14], [31], [39]–[41], [43]–[45], [49], [51], [52] (Appendix B-C). For example, MMLU [39] employed a variety of prompts across different subjects to assess knowledge, but inconsistencies in prompt phrasing could lead to overestimation of the LLM’s capabilities in certain areas. From a functionality perspective, inconsistency in prompt design can lead to assessments that do not accurately reflect the model’s true capabilities. For example, prompts that are overly leading or ambiguous may result in responses that either overestimate or underestimate the LLM’s performance, and this variability can make it difficult to compare results across different models or studies [60]. Regarding integrity, if benchmarks do not standardize prompt construction, there is a risk that LLMs could be fine-tuned or engineered to perform well on specific prompt styles, potentially compromising the fairness of the evaluation [35], [54]. To address such challenges, it is essential for benchmarks to adopt careful prompt engineering practices that aim for clarity, neutrality, and consistency, thereby ensuring that the evaluation accurately captures the model’s capabilities without introducing unintended biases.
This section explores the subtle complexities of human factors influencing LLM benchmarks (Fig. 4).
This section explores the subtle complexities of human factors influencing LLM benchmarks (Fig. 4).
A. Diversity in Human Curators and Evaluators Prevalence: 19/23
The heterogeneity among human curators and evaluators significantly impacts the construction and assessment of LLM benchmarks. Even with standardized guidelines, differences in cultural, religious, political, and academic or commercial backgrounds can introduce inconsistencies, particularly in subjective evaluations where linguistic and interpretative diversity play a role. This challenge was evident in benchmarks involving human judgment, such as those relying on sophisticated response evaluation or red-teaming (for cybersecurity purposes) [6]–[12], [14], [39], [40], [42], [43], [45]–[50], [52] (Appendix C-A). From a functionality perspective, this diversity can lead to subjective biases in benchmark construction and evaluation. For instance, cultural nuances can result in varying interpretations of responses, affecting the consistency and objectivity of the assessment [1], [3], [53]. For instance, LegalBench [14] utilized legal professionals from diverse jurisdictions to evaluate LLMs’ legal reasoning, yet the varying interpretations of legal principles across different regions could skew the results and lead to inconsistent assessments, making it difficult to ensure a uniform evaluation standard across different LLMs. Regarding integrity, inconsistent application and interpretation by diverse evaluators can open avenues for manipulation or biased outcomes. Disparities in evaluation criteria might allow LLMs to exploit certain nuances to achieve favorable results, undermining the benchmark’s robustness and trustworthiness [59]. This inconsistency can make it challenging to safeguard against potential biases or exploitation within the evaluation process. Addressing such issue requires a careful balance between leveraging diverse perspectives and ensuring consistency in evaluation. While diversity among evaluators enriches the assessment process, standardized protocols and training are necessary to minimize subjective biases and enhance the reliability of LLM benchmarks.
# B. Diverse Cultural, Social, Political, Religious, and Ideological Norms
B. Diverse Cultural, Social, Political, Religious, and Ideological Norms
Prevalence: 18/23 LLM benchmarks face the inherent challenge of encompassing a wide range of cultural, religious, political, and ideological norms [6], [7], [9]–[12], [14], [31], [39]–[44], [46], [49], [50], [52] (Appendix C-B). Given that humans often disagree on fundamental issues, including interpretations of universal principles like human rights, pursuing a universal benchmark that addresses all cultural and ideological perspectives may be unrealistic. Benchmarks that rely on standardized answers or rubrics can inadvertently clash with diverse values, highlighting the impossibility of creating a onesize-fits-all evaluation. For example, LegalBench [14], with its focus on American legal reasoning, may overestimate its applicability across diverse global legal contexts. Conversely, the integrity of Simplified Chinese LLM benchmarks like CEval [43] can be easily gamed by specific prompt structures or leverage training data tailored to the Simplified Chinese context, such as offering responses that align with or praise Chinese ideologies, to achieve artificially high scores without demonstrating genuine understanding or reasoning [61].
Functionally, this diversity means that benchmarks may struggle to fairly and accurately evaluate LLMs across varying viewpoints. An LLM’s response to sensitive topics might be deemed appropriate in one cultural context but controversial in another [13], [28]. This inconsistency challenges the creation of benchmarks that can impartially assess LLMs’ ability to navigate complex societal norms. An example of compromising benchmark integrity could be when benchmarks are tailored to avoid controversy (saying what people want to hear), leading to superficial assessments that do not truly evaluate the LLM’s ability to handle subtle or sensitive topics, thereby allowing LLMs to appear more competent than they actually are in real-world scenarios [28]. However, LLM providers and benchmark assessors are faced with the difficult decision of potentially offending some individuals, regardless of whether they aim for inclusivity or specificity. This is not a failure on their part, but rather an acknowledgment of the inherently subjective nature of humanity norms. Given such complexities, it may be more pragmatic for researchers to focus on context-specific benchmarks rather than striving for a universal standard benchmark for all human norms.
# VIII. DISCUSSIONS
This section provides a comprehensive discussion of the underlying causes that have resulted in the previously aforementioned inadequacies in current LLM benchmarking methods.
# A. Misuse of the Term “Benchmark”
The term “benchmark”, which should mean to the process of evaluating and comparing the performance of LLMs using standardized tests and metrics, has been applied more liberally in academic publishing than warranted, reflecting the evolving stage of AI regulatory compliance and public consensus on generative AI’s emerging role and impact. Consequently, a wide array of non-standardized LLM test sets have been self-declared as LLM benchmarks, often lacking the rigor and uniformity required for true benchmarking. Genuine benchmarks should provide standardized, comprehensive evaluations, distinguishing them from more informal assessments rooted in researchers’ subjective perspectives. Many so-called benchmarks serve merely as initial test sets tailored to specific LLM tasks, often falling short in integrity and functionality. Unlike fields such as automotive or aviation, where benchmarks are guided by established regulations and public consensus, the AI domain remains largely unregulated and in flux. This has led researchers to create varied task sets and questions, adding complexity to LLM evaluation and potentially misguiding interpretations of these sets as complete assessments of LLM capabilities. While such efforts contribute to the evolving landscape of LLM evaluation, highlighting the need for evolving standards, they also demonstrate the urgency for structured, universally accepted benchmarking frameworks. We believe that effective LLM benchmarks should rigorously assess functionality while being resilient to manipulation, ensuring they accurately reflect true model capabilities rather than superficial optimizations. The current research landscape requires benchmarks that align with
emerging regulatory mandates and societal needs, particularly as LLMs become increasingly integrated into various aspects of life. Through rigorously designed benchmarks, we can ensure the accurate assessment of LLMs’ growing capabilities, guiding their responsible development and deployment.
B. Assessment Limitations for Reasoning and Multimodality Current LLM benchmarks exhibit significant limitations in both functionality and integrity when assessing comprehensive reasoning and multimodal capabilities. Functionally, many benchmarks fail to evaluate the depth of LLMs’ reasoning and multimodal integration because they primarily focus on text-based tasks, neglecting the intricate processing required for integrating visual, auditory, and other sensory data [2], [3]. This narrow focus often results in benchmarks that assess only surface-level performance, without engaging the models in scenarios that require true multimodal understanding or complex reasoning. For example, benchmarks that rely on simple question-answer formats may inadvertently assess a model’s ability to recall learned patterns rather than its capacity for creative problem-solving, thereby conflating crystal (recalled knowledge) and fluid (creative problem-solving) intelligence. In terms of integrity, such benchmarks are vulnerable to being manipulated by LLMs that have been overfitted to specific datasets or tuned to exploit particular evaluation metrics, when benchmarks allow models to achieve high scores through memorization or pattern exploitation rather than demonstrating genuine capability. For instance, a model trained on benchmark-specific data can deliver seemingly impressive results without possessing the underlying understanding or adaptability required in real-world applications. Consequently, current benchmarks may provide misleading assessments of LLM capabilities, due to superficial optimization strategies [28].
# C. Unpredictability and Non-Repeatability
The rapid evolution of commercial and open source LLMs introduces significant unpredictability and non-repeatability in benchmark assessments. From a functionality standpoint, the frequent updates and model variations released by vendors make it difficult to ascertain whether a benchmark measures an LLM’s core capabilities or merely reflects transient optimizations tailored to the latest version. This instability undermines the benchmark’s ability to consistently evaluate reasoning, comprehension, or multimodal integration, as the results may vary with each model iteration. Regarding integrity, the evolving nature of LLMs opens avenues for potential manipulation, as vendors can modify models to perform well on known benchmarks without genuinely improving underlying capabilities. This dynamic can lead to misleadingly high benchmark scores that do not represent the LLM’s true functional performance in diverse, real-world scenarios. The lack of stable benchmarks complicates efforts to ensure that evaluations remain impartial and unaffected by superficial enhancements, thereby raising concerns about the authenticity of the assessments. Such issues require universally accepted, adaptable benchmarking frameworks that can accommodate
the rapid advancements in LLM and generative AI, while providing reliable, repeatable measures of LLM performance. Without such standards, the evaluation landscape risks becoming fragmented and inconsistent, impeding the ability to accurately gauge the evolving capabilities of LLMs.
# D. Knowledge Boundaries in AI Benchmarking
LLM benchmarks are constrained by the current limits of the benchmark creators’ human knowledge, hindering their ability to fully assess and cultivate emerging AI capabilities that may surpass conventional human understanding, potentially stalling innovation in fields that rely on advanced AI insights [4]. Additionally, the lack of specialized domain knowledge among benchmark evaluators can compromise the benchmark integrity, as generalist approaches often fail to address the subtle requirements of critical sectors such as national security or healthcare. Such deficiency not only introduces benchmarking biases, but also creates vulnerabilities, allowing LLMs to take advantage of “benchmarks” created by people who do not fully understand the specific field. To address such challenges, benchmarks must evolve to include a broader knowledge base and involve evaluators with deep expertise in specialized fields, which is crucial for creating assessments that genuinely reflect the advanced capabilities of LLMs and possibly soon Artificial General Intelligence (AGI), and are robust against manipulation. The rapid progression of LLMs requires LLM benchmarks that are as innovative and adaptable as the systems they evaluate, ensuring accurate measurement of LLM functionalities while maintaining integrity against evolving AI threats. Developing such benchmarks requires a concerted effort to integrate domain-specific knowledge into the evaluation process, ensuring that the assessments remain relevant and resilient in the face of AI’s advancing frontier.
# E. Challenges in Inclusive Benchmarking
Standardizing benchmarks for LLMs faces significant challenges due to the diversity of human values and perspectives, which extend beyond linguistic, cultural, religious, and political realms to encompass a broad spectrum of societal norms and ethical considerations, and are sometimes conflicting and irreconcilable. The predominance of English and Simplified Chinese (overlooking Traditional Chinese differences) in those benchmarks frequently reflected the default values and beliefs of their creators, overlooking the pluralistic nature of global beliefs and practices. For example, the English and Simplified Chinese centrism in benchmarks like BIG-Bench [6], and the focus on crystallized knowledge in MultiMedQA [10] and ARB [41], highlighted such challenges, potentially disadvantaging LLMs not trained in those more prevalent languages and limiting assessments of their broader reasoning capabilities. Such a narrower focus would compromise the benchmarks’ ability to comprehensively and holistically evaluate LLMs, potentially favoring LLMs that aligned with specific cultural or ideological norms at the expense of broader applicability and acceptance, and raising concerns about the trustworthiness of LLMs in the absence of cultural and ideological inclusivity. The reliance on standardized answers or rubrics
exacerbated this issue, especially in assessments designed to gauge fairness, societal appropriateness, or ethical decisionmaking, where the rich tapestry of human diversity demanded a more sophisticated approach. Addressing the aforementioned challenges would require moving beyond technical fixes to incorporate ethical decision-making and cultural sensitivity, highlighting the need for benchmarks that facilitate the development of globally aware LLMs, acknowledging the vast array of human values [13]. As the development of future LLM benchmarks progresses, it is anticipated that many will encounter similar challenges, particularly when the diversity of benchmark creators, evaluators, or the content itself does not fully embrace the broad spectrum of human diversity. However, different jurisdictions, religions, or cultures may have varying interpretations of inclusivity, ethics, or societal fairness that could lead to disagreements on what constitutes a superior LLM, which extends beyond technical benchmarks into broader considerations of values and politics.
# F. Limitations of This Study
F. Limitations of This Study
This study, while aiming to provide a thorough critique of LLM benchmarks, encounters its own set of limitations. Firstly, our analysis inherently carries our subjective bias, a challenge we have attempted to mitigate by transparently listing our evaluative rationale in the Appendices. Due to the varying level of detail in benchmark descriptions, when we could not conclusively determine whether a study had acknowledged or addressed a specific benchmark inadequacy, or evaluate the correctness of questions and answers (which themselves could become outdated as technology and modern civilization evolves, and the study in [53] found mislabeled or unanswerable examples within MMLU [39]), we opted to give these studies the benefit of the doubt, acknowledging the potential for possible oversight. Secondly, while only 5 out of the 23 surveyed benchmark studies were peer-reviewed, our critical inclusion of preprints should not be seen as diminishing their scientific contribution, although it highlighted the evolving and dynamic nature of research in this field. Thirdly, our decision not to attempt reproducing results from the surveyed benchmark studies, was informed by the substantial time and effort required, coupled with the challenge of LLM updates potentially altering outcomes during the lengthy evaluation process. Furthermore, while we addressed linguistic differences between Simplified Chinese and Traditional Chinese, we did not explore the subtle distinctions among various English dialects (e.g., American, British) and their inherent logical differences related to cultural attitudes, thereby acknowledging another layer of complexity in language-based evaluations. Lastly, the absence of specific regulations and guidelines for generative AI and LLMs, and a lack of public consensus on acceptable AI behaviors, constrained our ability to definitively enumerate all benchmark inadequacies. The evolving capabilities of advanced AI systems would require ongoing amendments to our critique, reflecting the dynamic nature of the field and the continuous emergence of new AI features.
G. Future Research Direction: Extending Benchmarks with Behavioral Profiling and Regular Audits
Behavioral Profiling and Regular Audits To effectively evaluate and utilize LLMs amid the swift advancements in generative AI, their evaluation methods can be extended beyond traditional benchmarks to include both initial screenings and in-depth, ongoing assessments that align with changing technological and societal needs. Similar to the initial candidate selection and aptitude tests in the human recruitment process, traditional benchmarks can serve as the first filter to screen out LLMs that fail to meet basic competence levels, akin to identifying candidates who might be fraudulent, incompetent, non-compliant or otherwise unsuitable. This step ensures that only LLMs with a foundational level of proficiency and regulatory compliance proceed to more rigorous evaluations, optimizing resource allocation for subsequent stages of the assessment process. Mirroring the interview stage of candidate selection, behavioral profiling explores deeper into the subtleties of LLM performance by assessing models beyond mere task completion, examining adaptability, ethical reasoning, and creativity in scenarios reflecting real-world complexities. Such AI behavioral analysis and profiling, a topic for future research, will prioritize LLMs most likely to fulfill specific roles or tasks, providing a sophisticated understanding of model capabilities and informing better selection decisions for deployment in sensitive or critical applications. Echoing the probation period in employment, regular audits post-deployment serve to continuously assess LLM performance against evolving standards and expectations. This step is crucial for catching any discrepancies between initial evaluations and actual performance, ensuring LLMs remain aligned with the requirements and ethical standards of evaluators and regulators over time. However, there are certain methodological limitations inherent in our approach. The subjective nature of benchmark evaluations, particularly when assessing more abstract qualities like adaptability or ethical reasoning, introduces potential bias. Additionally, the exclusion of certain benchmarks or evaluation frameworks may limit the generalizability of our findings. Future studies should address these gaps by exploring more objective, quantifiable methods for behavioral profiling and expanding the scope to include underrepresented benchmarks. Given the rapid advancements in AI, our analysis risks becoming outdated as LLMs and benchmarking practices evolve. This highlights the need for ongoing research to continuously refine both the criteria and methods for LLM evaluation, ensuring they remain relevant and reflective of real-world complexities. Future research should focus on developing dynamic benchmarking systems that adapt to these changes, incorporating real-time audits and frequent updates to evaluation standards to capture the full scope of LLM capabilities and risks. Implementing this comprehensive evaluation framework will require collaboration across academia, industry, and regulatory bodies to develop standardized methodologies and ethical guidelines for each stage, ensuring assessments are rigorous and reflective of societal values and technological advancements. This will facilitate the creation of a dynamic evaluation ecosystem capable of adapting to the rapid pace of
AI innovation and ensuring deployed LLMs are both effective and secure.
# IX. CONCLUSION
This study has critically analyzed 23 state-of-the-art LLM benchmarks, exposing significant inadequacies across technological, processual, and human dynamics that compromise their accuracy and reliability. The lack of standardized frameworks in AI benchmarking, unlike the established practices in regulated industries, has led to a proliferation of researcherdefined benchmarks that inadequately capture the complexity and evolving nature of LLMs. Our key contributions include: (i) formulating a unified evaluation framework rooted in cybersecurity principles to systematically identify and address deficiencies in benchmark design, focusing on functionality and integrity; (ii) conducting a detailed critique of 23 prominent LLM benchmarks, highlighting widespread issues that could impair comprehensive LLM assessment; and (iii) proposing the integration of LLM behavioral profiling and regular audits to enhance evaluation methodologies. As we arguably edge closer to the realization of AGI-like capabilities, it becomes crucial to rectify such benchmarking deficiencies, to ensure the responsible and secure deployment of LLMs. Furthermore, the lack of research consensus on how to properly benchmark LLMs, coupled with the academic liberty of releasing preprints, is likely to lead to a continued influx of question-and-answer sets self-claimed as benchmarks. It is therefore critical to point out their limitations and encourage deeper reflection within the research community. Our proposed extension of benchmarks with behavioral profiling and audits can provide a more subtle and rigorous evaluation of LLM capabilities and risks. Future work should prioritize the development of such benchmarks and establish standardized evaluation guidelines. We advocate for an international collaborative effort involving academia, industry, and regulatory bodies to continuously refine LLM benchmarks, aligning them with technological advancements and societal needs to ensure the development of robust and trustworthy AI systems.
# REFERENCES
[1] B. Min, H. Ross, E. Sulem, A. P. B. Veyseh, T. H. Nguyen, O. Sainz, E. Agirre, I. Heintz, and D. Roth, “Recent advances in natural language processing via large pre-trained language models: A survey,” ACM Computing Surveys, vol. 56, no. 2, pp. 1–40, 2023. [2] T. R. McIntosh, T. Susnjak, T. Liu, P. Watters, and M. N. Halgamuge, “From google gemini to openai q*(q-star): A survey of reshaping the generative artificial intelligence (ai) research landscape,” arXiv preprint arXiv:2312.10868, 2023. [3] Y. Chang, X. Wang, J. Wang, Y. Wu, L. Yang, K. Zhu, H. Chen, X. Yi, C. Wang, Y. Wang et al., “A survey on evaluation of large language models,” ACM Transactions on Intelligent Systems and Technology, 2023. [4] T. R. McIntosh, T. Liu, T. Susnjak, P. Watters, and M. N. Halgamuge, “A reasoning and value alignment test to assess advanced gpt reasoning,” ACM Transactions on Interactive Intelligent Systems, 2024. [5] R. Liu and N. Moini, “Benchmarking transportation safety performance via shift-share approaches,” Journal of Transportation Safety & Security, vol. 7, no. 2, pp. 124–137, 2015. [6] A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso et al., “Beyond the imitation game: Quantifying and extrapolating the capabilities of language models,” Transactions on machine learning research, 2022.
[7] K. Zhu, Q. Zhao, H. Chen, J. Wang, and X. Xie, “Promptbench: A unified library for evaluation of large language models,” Journal of Machine Learning Research, vol. 25, no. 254, pp. 1–22, 2024. [8] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. d. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman et al., “Evaluating large language models trained on code,” arXiv preprint arXiv:2107.03374, 2021. [9] C. Liu, R. Jin, Y. Ren, L. Yu, T. Dong, X. Peng, S. Zhang, J. Peng, P. Zhang, Q. Lyu et al., “M3ke: A massive multi-level multi-subject knowledge evaluation benchmark for chinese large language models,” arXiv preprint arXiv:2305.10263, 2023. [10] K. Singhal, S. Azizi, T. Tu, S. S. Mahdavi, J. Wei, H. W. Chung, N. Scales, A. Tanwani, H. Cole-Lewis, S. Pfohl et al., “Large language models encode clinical knowledge,” Nature, vol. 620, no. 7972, pp. 172– 180, 2023. [11] J. Yu, X. Wang, S. Tu, S. Cao, D. Zhang-Li, X. Lv, H. Peng, Z. Yao, X. Zhang, H. Li et al., “Kola: Carefully benchmarking world knowledge of large language models,” arXiv preprint arXiv:2306.09296, 2023. [12] W. Zhong, R. Cui, Y. Guo, Y. Liang, S. Lu, Y. Wang, A. Saied, W. Chen, and N. Duan, “Agieval: A human-centric benchmark for evaluating foundation models,” in Findings of the Association for Computational Linguistics: NAACL 2024, 2024, pp. 2299–2314. [13] T. R. McIntosh, T. Liu, T. Susnjak, P. Watters, A. Ng, and M. N. Halgamuge, “A culturally sensitive test to evaluate nuanced gpt hallucination,” IEEE Transactions on Artificial Intelligence, 2023. [14] N. Guha, J. Nyarko, D. Ho, C. R´e, A. Chilton, A. Chohlas-Wood, A. Peters, B. Waldon, D. Rockmore, D. Zambrano et al., “Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models,” Advances in Neural Information Processing Systems, vol. 36, 2024. [15] A. V. Papadopoulos, L. Versluis, A. Bauer, N. Herbst, J. Von Kistowski, A. Ali-Eldin, C. L. Abad, J. N. Amaral, P. Tuma, and A. Iosup, “Methodological principles for reproducible performance evaluation in cloud computing,” IEEE Transactions on Software Engineering, vol. 47, no. 8, pp. 1528–1543, 2019. [16] H. Brunst, S. Chandrasekaran, F. M. Ciorba, N. Hagerty, R. Henschel, G. Juckeland, J. Li, V. G. M. Vergara, S. Wienke, and M. Zavala, “First experiences in performance benchmarking with the new spechpc 2021 suites,” in 2022 22nd IEEE International Symposium on Cluster, Cloud and Internet Computing (CCGrid). IEEE, 2022, pp. 675–684. [17] B. Li, W. Ren, D. Fu, D. Tao, D. Feng, W. Zeng, and Z. Wang, “Benchmarking single-image dehazing and beyond,” IEEE Transactions on Image Processing, vol. 28, no. 1, pp. 492–505, 2018. [18] M. Hort, M. Kechagia, F. Sarro, and M. Harman, “A survey of performance optimization for mobile applications,” IEEE Transactions on Software Engineering, vol. 48, no. 8, pp. 2879–2904, 2021. [19] M. S. Aslanpour, S. S. Gill, and A. N. Toosi, “Performance evaluation metrics for cloud, fog and edge computing: A review, taxonomy, benchmarks and standards for future research,” Internet of Things, vol. 12, p. 100273, 2020. [20] F. Xia, W. B. Shen, C. Li, P. Kasimbeg, M. E. Tchapmi, A. Toshev, R. Mart´ın-Mart´ın, and S. Savarese, “Interactive gibson benchmark: A benchmark for interactive navigation in cluttered environments,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 713–720, 2020. [21] A. Romano, X. Liu, Y. Kwon, and W. Wang, “An empirical study of bugs in webassembly compilers,” in 2021 36th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2021, pp. 42–54. [22] E. Davis, “Benchmarks for automated commonsense reasoning: A survey,” ACM Computing Surveys, vol. 56, no. 4, p. 41, 2023. [23] S. Zhu, J. Shi, L. Yang, B. Qin, Z. Zhang, L. Song, and G. Wang, “Measuring and modeling the label dynamics of online {Anti-Malware} engines,” in 29th USENIX Security Symposium (USENIX Security 20), 2020, pp. 2361–2378. [24] T. McIntosh, A. Kayes, Y.-P. P. Chen, A. Ng, and P. Watters, “Ransomware mitigation in the modern era: A comprehensive review, research challenges, and future directions,” ACM Computing Surveys (CSUR), vol. 54, no. 9, pp. 1–36, 2021. [25] T. A. van Schaik and B. Pugh, “A field guide to automatic evaluation of llm-generated summaries,” in Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024, pp. 2832–2836. [26] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman, “Glue: A multi-task benchmark and analysis platform for natural language understanding,” in 7th International Conference on Learning Representations, ICLR 2019, 2019.
[27] Z. Tan, T. Chen, Z. Zhang, and H. Liu, “Sparsity-guided holistic explanation for llms with interpretable inference-time intervention,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 19, 2024, pp. 21 619–21 627. [28] T. R. McIntosh, T. Susnjak, T. Liu, P. Watters, and M. N. Halgamuge, “The inadequacy of reinforcement learning from human feedbackradicalizing large language models via semantic vulnerabilities,” IEEE Transactions on Cognitive and Developmental Systems, 2024. [29] G. F. Almeida, J. L. Nunes, N. Engelmann, A. Wiegmann, and M. de Ara´ujo, “Exploring the psychology of llms’ moral and legal reasoning,” Artificial Intelligence, vol. 333, p. 104145, 2024. [30] T. R. McIntosh, T. Susnjak, T. Liu, P. Watters, D. Xu, D. Liu, R. Nowrozy, and M. N. Halgamuge, “From cobit to iso 42001: Evaluating cybersecurity frameworks for opportunities, risks, and regulatory compliance in commercializing large language models,” Computers & Security, vol. 144, p. 103964, 2024. [31] P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar et al., “Holistic evaluation of language models,” arXiv preprint arXiv:2211.09110, 2022. [32] K. Kenthapadi, M. Sameki, and A. Taly, “Grounding and evaluation for large language models: Practical challenges and lessons learned (survey),” in Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024, pp. 6523–6533. [33] Z. Chen, H. Mao, H. Li, W. Jin, H. Wen, X. Wei, S. Wang, D. Yin, W. Fan, H. Liu et al., “Exploring the potential of large language models (llms) in learning on graphs,” ACM SIGKDD Explorations Newsletter, vol. 25, no. 2, pp. 42–61, 2024. [34] M. Jegorova, C. Kaul, C. Mayor, A. Q. O’Neil, A. Weir, R. MurraySmith, and S. A. Tsaftaris, “Survey: Leakage and privacy at inference time,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 7, pp. 9090–9108, 2022. [35] Y. Deng, W. Zhang, S. J. Pan, and L. Bing, “Multilingual jailbreak challenges in large language models,” arXiv preprint arXiv:2310.06474, 2023. [36] I. Cheong, K. Xia, K. K. Feng, Q. Z. Chen, and A. X. Zhang, “(a) i am not a lawyer, but...: Engaging legal experts towards responsible llm policies for legal advice,” in The 2024 ACM Conference on Fairness, Accountability, and Transparency, 2024, pp. 2454–2469. [37] M. I. Javaid and M. M. W. Iqbal, “A comprehensive people, process and technology (ppt) application model for information systems (is) risk management in small/medium enterprises (sme),” in 2017 International Conference on Communication Technologies (ComTech). IEEE, 2017, pp. 78–90. [38] D. Al-Fraihat, M. Joy, J. Sinclair et al., “Evaluating e-learning systems success: An empirical study,” Computers in human behavior, vol. 102, pp. 67–86, 2020. [39] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” arXiv preprint arXiv:2009.03300, 2020. [40] Y. Fu, L. Ou, M. Chen, Y. Wan, H. Peng, and T. Khot, “Chain-of-thought hub: A continuous effort to measure large language models’ reasoning performance,” arXiv preprint arXiv:2305.17306, 2023. [41] T. Sawada, D. Paleka, A. Havrilla, P. Tadepalli, P. Vidas, A. Kranias, J. J. Nay, K. Gupta, and A. Komatsuzaki, “Arb: Advanced reasoning benchmark for large language models,” arXiv preprint arXiv:2307.13692, 2023. [42] Z. Gu, X. Zhu, H. Ye, L. Zhang, J. Wang, S. Jiang, Z. Xiong, Z. Li, Q. He, R. Xu et al., “Xiezhi: An ever-updating benchmark for holistic domain knowledge evaluation,” arXiv preprint arXiv:2306.05783, 2023. [43] Y. Huang, Y. Bai, Z. Zhu, J. Zhang, J. Zhang, T. Su, J. Liu, C. Lv, Y. Zhang, J. Lei et al., “C-eval: A multi-level multidiscipline chinese evaluation suite for foundation models,” arXiv preprint arXiv:2305.08322, 2023. [44] R. Shah, K. Chawla, D. Eidnani, A. Shah, W. Du, S. Chava, N. Raman, C. Smiley, J. Chen, and D. Yang, “When flue meets flang: Benchmarks and large pretrained language model for financial domain,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022, pp. 2322–2335. [45] Q. Xu, F. Hong, B. Li, C. Hu, Z. Chen, and J. Zhang, “On the tool manipulation capability of open-s