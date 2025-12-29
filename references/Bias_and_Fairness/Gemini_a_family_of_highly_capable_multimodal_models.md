# Gemini: A Family of Highly Capable Multimodal Models
Gemini Team, Google1
This report introduces a new family of multimodal models, Gemini, that exhibit remarkable capabilities across image, audio, video, and text understanding. The Gemini family consists of Ultra, Pro, and Nano sizes, suitable for applications ranging from complex reasoning tasks to on-device memory-constrained use-cases. Evaluation on a broad range of benchmarks shows that our most-capable Gemini Ultra model advances the state of the art in 30 of 32 of these benchmarks — notably being the first model to achieve human-expert performance on the well-studied exam benchmark MMLU, and improving the state of the art in every one of the 20 multimodal benchmarks we examined. We believe that the new capabilities of the Gemini family in cross-modal reasoning and language understanding will enable a wide variety of use cases. We discuss our approach toward post-training and deploying Gemini models responsibly to users through services including Gemini, Gemini Advanced, Google AI Studio, and Cloud Vertex AI.
# 1. Introduction
We present Gemini, a family of highly capable multimodal models developed at Google. We trained Gemini models jointly across image, audio, video, and text data for the purpose of building a model with both strong generalist capabilities across modalities alongside cutting-edge understanding and reasoning performance in each respective domain. Gemini 1.0, our first version, comes in three sizes: Ultra for highly-complex tasks, Pro for enhanced performance and deployability at scale, and Nano for on-device applications. Each size is specifically tailored to address different computational limitations and application requirements. After large-scale pre-training, we post-train our models to improve overall quality, enhance target capabilities, and ensure alignment and safety criteria are met. Due to the varied requirements of our downstream applications, we have produced two post-trained Gemini model family variants. Chat-focused variants, referred to as Gemini Apps models, are optimized for Gemini and Gemini Advanced, our conversational AI service formerly known as Bard. Developer-focused variants, referred to as Gemini API models, are optimized for a range of products and are accessible through Google AI Studio and Cloud Vertex AI. We evaluate the performance of pre- and post-trained Gemini models on a comprehensive suite of internal and external benchmarks covering a wide range of language, coding, reasoning, and multimodal tasks. The Gemini family advances state-of-the-art in large-scale language modeling (Anil et al., 2023; Brown et al., 2020; Chowdhery et al., 2023; Hoffmann et al., 2022; OpenAI, 2023a; Radford et al., 2019; Rae et al., 2021), image understanding (Alayrac et al., 2022; Chen et al., 2022; Dosovitskiy et al., 2020; OpenAI, 2023b; Reed et al., 2022; Yu et al., 2022a), audio processing (Radford et al., 2023; Zhang et al., 2023), and video understanding (Alayrac et al., 2022; Chen et al., 2023). It also builds on the work on sequence models (Sutskever et al., 2014), a long history of work in deep learning based on neural networks (LeCun et al., 2015), and machine learning distributed systems
We present Gemini, a family of highly capable multimodal models developed at Google. We trained Gemini models jointly across image, audio, video, and text data for the purpose of building a model with both strong generalist capabilities across modalities alongside cutting-edge understanding and reasoning performance in each respective domain.
1See Contributions and Acknowledgments section for full author list. Please send correspo report@google.com
© 2024 Google. All rights reserved
# (Barham et al., 2022; Bradbury et al., 2018; Dean et al., 2012) that enable large-scale traini
Our most capable model, Gemini Ultra, achieves new state-of-the-art results in 30 of 32 benchmarks we report on, including 10 of 12 popular text and reasoning benchmarks, 9 of 9 image understanding benchmarks, 6 of 6 video understanding benchmarks, and 5 of 5 speech recognition and speech translation benchmarks. Gemini Ultra is the first model to achieve human-expert performance on MMLU (Hendrycks et al., 2021a) — a prominent benchmark testing knowledge and reasoning via a suite of exams — with a score above 90%. Beyond text, Gemini Ultra makes notable advances on challenging multimodal reasoning tasks. For example, on the recent MMMU benchmark (Yue et al., 2023), that comprises questions about images on multi-discipline tasks requiring college-level subject knowledge and deliberate reasoning, Gemini Ultra achieves a new state-of-the-art score of 62.4%, outperforming the previous best model by more than 5 percentage points. It provides a uniform performance lift for video question answering and audio understanding benchmarks. Qualitative evaluation showcases impressive crossmodal reasoning capabilities, enabling the model to understand and reason across an input sequence of audio, images, and text natively (see Figure 5 and Table 13). Consider the educational setting depicted in Figure 1 as an example. A teacher has drawn a physics problem of a skier going down a slope, and a student has worked through a solution to it. Using Gemini models’ multimodal reasoning capabilities, the model is able to understand the messy handwriting, correctly understand the problem formulation, convert both the problem and solution to mathematical typesetting, identify the specific step of reasoning where the student went wrong in solving the problem, and then give a worked through correct solution to the problem. This opens up exciting educational possibilities, and we believe the new multimodal and reasoning capabilities of Gemini models have dramatic applications across many fields. The reasoning capabilities of large language models show promise toward building generalist agents that can tackle more complex multi-step problems. The AlphaCode team built AlphaCode 2 (Leblond et al, 2023), a new Gemini-model-powered agent, that combines Gemini models’ reasoning capabilities with search and tool-use to excel at solving competitive programming problems. AlphaCode 2 ranks within the top 15% of entrants on the Codeforces competitive programming platform, a large improvement over its state-of-the-art predecessor in the top 50% (Li et al., 2022). In tandem, we advance the frontier of efficiency with Gemini Nano, a series of small models targeting on-device deployment. These models excel in on-device tasks, such as summarization, reading comprehension, text completion tasks, and exhibit impressive capabilities in reasoning, STEM, coding, multimodal, and multilingual tasks relative to their sizes. In the following sections, we first provide an overview of the model architecture, training infrastructure, and pre-training dataset. We then present detailed evaluations of the pre- and post-trained Gemini model family, covering well-studied benchmarks across text, code, image, audio and video — which include both English performance and multilingual capabilities. Next we discuss our approach to post-training, highlight common and distinct aspects of the Gemini Apps and Gemini API model variants, and benchmark their performance on key capabilities. Responsible deployment is critical: we explain our process for impact assessments, developing model policies, evaluations, and mitigations of harm before deployment decisions. Finally, we discuss the broader implications of Gemini models, their limitations alongside their potential applications — paving the way for a new era of research and innovation in AI.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2e57/2e575a4c-e3d1-414d-98ae-930e7a2f1280.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/21c4/21c44664-e152-4fb8-b060-0e41e6e2f327.png" style="width: 50%;"></div>
Figure 1 | Verifying a student’s solution to a physics problem. The model is able to correctly recognize all of the handwritten content and verify the reasoning. On top of understanding the text in the image, it needs to understand the problem setup and correctly follow instructions to generate LATEX.
# 2. Model Architecture
Gemini models build on top of Transformer decoders (Vaswani et al., 2017b) that are enhanced with improvements in architecture and model optimization to enable stable training at scale and optimized inference on Google’s Tensor Processing Units. They are trained to support 32k context length, employing efficient attention mechanisms (for e.g. multi-query attention (Shazeer, 2019a)). Our first version, Gemini 1.0, comprises three main sizes to support a wide range of applications as discussed in Table 1. Gemini models are trained to accommodate textual input interleaved with a wide variety of audio and visual inputs, such as natural images, charts, screenshots, PDFs, and videos, and they can produce text and image outputs (see Figure 2). The visual encoding of Gemini models is inspired by our own foundational work on Flamingo (Alayrac et al., 2022), CoCa (Yu et al., 2022a), and PaLI (Chen et al., 2022), with the important distinction that the models are multimodal from the beginning and can natively output images using discrete image tokens (Ramesh et al., 2021; Yu et al., 2022b). Video understanding is accomplished by encoding the video as a sequence of frames in the large context window. Video frames or images can be interleaved naturally with text or audio as part of the model input. The models can handle variable input resolution in order to spend more compute on tasks that require fine-grained understanding. In addition, Gemini models can directly ingest audio
Model size
Model description
Ultra
Our most capable model that delivers state-of-the-art performance across a wide
range of highly complex tasks, including reasoning and multimodal tasks. It is
efficiently serveable at scale on TPU accelerators due to the Gemini architecture.
Pro
A performance-optimized model in terms of cost as well as latency that delivers
significant performance across a wide range of tasks. This model exhibits strong
reasoning performance and broad multimodal capabilities.
Nano
Our most efficient model, designed to run on-device. We trained two versions of
Nano, with 1.8B (Nano-1) and 3.25B (Nano-2) parameters, targeting low and high
memory devices respectively. It is trained by distilling from larger Gemini models. It
is 4-bit quantized for deployment and provides best-in-class performance.
Table 1 | An overview of the Gemini 1.0 model family.
Table 1 | An overview of the Gemini 1.0 model family.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/feea/feea76ec-6f43-46f7-805c-09f6452eab37.png" style="width: 50%;"></div>
Figure 2 | Gemini models support interleaved sequences of text, image, audio, and video as input (illustrated by tokens of different colors in the input sequence). They can output responses wit interleaved image and text.
signals at 16kHz from Universal Speech Model (USM) (Zhang et al., 2023) features. This enables the model to capture nuances that are typically lost when the audio is naively mapped to a text input (for example, see audio understanding demo on the website).
Training the Gemini family of models required innovations in training algorithms, dataset, and infrastructure. For the Pro model, the inherent scalability of our infrastructure and learning algorithms enable us to complete pre-training in a matter of weeks, leveraging a fraction of the Ultra’s resources. The Nano series of models leverage additional advancements in distillation and training algorithms to produce the best-in-class small language models for a wide variety of tasks, such as summarization and reading comprehension, which power our next generation on-device experiences.
# 3. Training Infrastructure
We trained Gemini models using TPUv5e and TPUv4 (Jouppi et al., 2023), depending on their sizes and configuration. Training Gemini Ultra used a large fleet of TPUv4 accelerators owned by Google
across multiple datacenters. This represents a significant increase in scale over our prior flagship model PaLM-2 which presented new infrastructure challenges. Scaling up the number of accelerators results in a proportionate decrease in the mean time between failure of hardware in the overall system. We minimized the rate of planned reschedules and preemptions, but genuine machine failures are commonplace across all hardware accelerators at such large scales. TPUv4 accelerators are deployed in “SuperPods” of 4096 chips, each connected to a dedicated optical switch, which can dynamically reconfigure 4x4x4 chip cubes into arbitrary 3D torus topologies in around 10 seconds (Jouppi et al., 2023). For Gemini Ultra, we decided to retain a small number of cubes per superpod to allow for hot standbys and rolling maintenance. TPU accelerators primarily communicate over the high speed inter-chip-interconnect, but at Gemini Ultra scale, we combine SuperPods in multiple datacenters using Google’s intra-cluster and inter-cluster network (Poutievski et al., 2022; Wetherall et al., 2023; yao Hong et al., 2018). Google’s network latencies and bandwidths are sufficient to support the commonly used synchronous training paradigm, exploiting model parallelism within superpods and data-parallelism across superpods. The ‘single controller’ programming model of Jax (Bradbury et al., 2018) and Pathways (Barham et al., 2022) allows a single Python process to orchestrate the entire training run, dramatically simplifying the development workflow. The GSPMD partitioner (Xu et al., 2021) in the XLA compiler partitions the training step computation, and the MegaScale XLA compiler (XLA, 2019) pass statically schedules appropriate collectives so that they maximally overlap with the computation with very little variation in step time. Maintaining a high goodput2 at this scale would have been impossible using the conventional approach of periodic checkpointing of weights to persistent cluster storage. For Gemini models, we instead made use of redundant in-memory copies of the model state, and on any unplanned hardware failures, we rapidly recover directly from an intact model replica. Compared to both PaLM and PaLM-2 (Anil et al., 2023), this provided a substantial speedup in recovery time, despite the significantly larger training resources being used. As a result, the overall goodput for the largest-scale training job increased from 85% to 97%. Training at unprecedented scale invariably surfaces new and interesting systems failure modes and in this instance one of the problems that we needed to address was that of “Silent Data Corruption (SDC)” (Dixit et al., 2021; Hochschild et al., 2021; Vishwanathan et al., 2015). Although these are extremely rare, the scale of Gemini models means that we can expect SDC events to impact training every week or two. Rapidly detecting and removing faulty hardware required several new techniques that exploit deterministic replay to isolate incorrect computations, combined with proactive SDC scanners on idle machines and hot standbys. Our fully deterministic infrastructure allowed us to quickly identify root causes (including hardware failures) during the development leading up to the Ultra model, and this was a crucial ingredient towards stable training.
# 4. Pre-Training Dataset
Gemini models are trained on a dataset that is both multimodal and multilingual. Our pre-training dataset uses data from web documents, books, and code, and includes image, audio, and video data. We use the SentencePiece tokenizer (Kudo and Richardson, 2018) and find that training the tokenizer on a large sample of the entire training corpus improves the inferred vocabulary and subsequently improves model performance. For example, we find Gemini models can efficiently 2We define goodput as the time spent computing useful new steps over the elapsed time of the training job.
Gemini models are trained on a dataset that is both multimodal and multilingual. Our pre-trainin dataset uses data from web documents, books, and code, and includes image, audio, and video dat
We use the SentencePiece tokenizer (Kudo and Richardson, 2018) and find that training th tokenizer on a large sample of the entire training corpus improves the inferred vocabulary an subsequently improves model performance. For example, we find Gemini models can efficientl
2We define goodput as the time spent computing useful new steps over the elapsed time of the training job.
# nize non-Latin scripts which can, in turn, benefit model quality as well as training and inference d.
tokenize non-Latin scripts which can, in turn, benefit model quality as well as training and inference speed.
tokenize non-Latin scripts which can, in turn, benefit model quality as well as training and inference speed. The number of tokens used to train the largest models were determined following the approach in Hoffmann et al. (2022). The smaller models are trained for significantly more tokens to improve performance for a given inference budget, similar to the approach advocated in Touvron et al. (2023a). We apply quality filters to all datasets, using both heuristic rules and model-based classifiers. We also perform safety filtering to remove harmful content based on our policies. To maintain the integrity of evaluations, we search for and remove any evaluation data that may have been in our training corpus before using data for training. The final data mixtures and weights were determined through ablations on smaller models. We stage training to alter the mixture composition during training – increasing the weight of domain-relevant data towards the end of training. We find that data quality is an important factor for highly-performing models, and believe that many interesting questions remain around finding the optimal dataset distribution for pre-training.
# 5. Evaluation
The Gemini models are natively multimodal, as they are trained jointly across text, image, audio, and video. One open question is whether this joint training can result in a model which has strong capabilities in each domain – even when compared to models and approaches that are narrowly tailored to single domains. We find this to be the case: Gemini models set a new state of the art across a wide range of text, image, audio, and video benchmarks. ww
# 5.1. Text
5.1. Text
5.1.1. Academic Benchmarks
# 5.1.1. Academic Benchmarks
We compare pre- and post-trained Gemini Pro and Ultra models to a suite of external LLMs and our previous best model PaLM 2 across a series of text-based academic benchmarks covering reasoning, reading comprehension, STEM, and coding. We report these results in Table 2. Broadly, we find that the performance of Gemini Pro outperforms inference-optimized models such as GPT-3.5 and performs comparably with several of the most capable models available, and Gemini Ultra outperforms all current models. In this section, we examine some of these findings. On MMLU (Hendrycks et al., 2021a), Gemini Ultra can outperform all existing models, achieving an accuracy of 90.04%. MMLU is a holistic exam benchmark, which measures knowledge across a set of 57 subjects. Human expert performance is gauged at 89.8% by the benchmark authors, and Gemini Ultra is the first model to exceed this threshold, with the prior state-of-the-art result at 86.4%. Achieving high performance requires specialist knowledge across many domains (e.g. law, biology, history, etc.), alongside reading comprehension and reasoning. We find Gemini Ultra achieves highest accuracy when used in combination with a chain-of-thought prompting approach (Wei et al., 2022b) that accounts for model uncertainty. The model produces a chain of thought with k samples, for example 8 or 32. If there is a consensus above a preset threshold (selected based on the validation split), it selects this answer, otherwise it reverts to a greedy sample based on maximum likelihood choice without chain of thought. We refer the reader to appendix for a detailed breakdown of how this approach compares with only chain-of-thought prompting or only greedy sampling. In mathematics, a field commonly used to benchmark the analytical capabilities of models, Gemini Ultra shows strong performance on both elementary exams and competition-grade problem sets. For the grade-school math benchmark, GSM8K (Cobbe et al., 2021), we find Gemini Ultra reaches 94.4%
We compare pre- and post-trained Gemini Pro and Ultra models to a suite of external LLMs and ou previous best model PaLM 2 across a series of text-based academic benchmarks covering reasoning reading comprehension, STEM, and coding. We report these results in Table 2. Broadly, we find that the performance of Gemini Pro outperforms inference-optimized models such as GPT-3.5 and performs comparably with several of the most capable models available, and Gemini Ultra outperform all current models. In this section, we examine some of these findings.
accuracy with chain-of-thought prompting and self-consistency (Wang et al., 2022) compared to the previous best accuracy of 92% with the same prompting technique. Similar positive trends are observed in increased difficulty math problems drawn from middle- and high-school math competitions (MATH benchmark), with the Gemini Ultra model outperforming all competitor models, reaching 53.2% using 4-shot prompting. The model also outperforms the state of the art on even harder tasks derived from American Mathematical Competitions (150 questions from 2022 and 2023). Smaller models perform poorly on this challenging task scoring close to random, but Gemini Ultra can solve 32% of the questions, compared to the 30% solve rate for GPT-4. Gemini Ultra also excels in coding, a popular use case of current LLMs. We evaluate the model on many conventional and internal benchmarks and also measure its performance as part of more complex reasoning systems such as AlphaCode 2 (see Section 5.1.7 on complex reasoning systems). For example, on HumanEval, a standard code-completion benchmark (Chen et al., 2021) mapping function descriptions to Python implementations, instruction-tuned Gemini Ultra correctly implements 74.4% of problems. On a new held-out evaluation benchmark for python code generation tasks, Natural2Code, where we ensure no web leakage, Gemini Ultra achieves the highest score of 74.9%. Evaluation on these benchmarks is challenging and may be affected by data contamination. We performed an extensive leaked data analysis after training to ensure the results we report here are as scientifically sound as possible, but still found some minor issues and decided not to report results on e.g. LAMBADA (Paperno et al., 2016). As part of the evaluation process, on a popular benchmark, HellaSwag (Zellers et al., 2019), we find that an additional hundred fine-tuning steps on specific website extracts corresponding to the HellaSwag training set (which were not included in the Gemini model pretraining set) improve the validation accuracy of Gemini Pro to 89.6% and Gemini Ultra to 96.0%, when measured with 1-shot prompting (we measured GPT-4 obtained 92.3% when evaluated 1-shot via the API). This suggests that the benchmark results are susceptible to the pretraining dataset composition. We choose to report HellaSwag decontaminated results only in a 10-shot evaluation setting. We believe there is a need for more robust and nuanced standardized evaluation benchmarks with no leaked data. So, we evaluate Gemini models on several new held-out evaluation datasets that were recently released, such as WMT23 and Math-AMC 2022-2023 problems, or internally generated from non-web sources, such as Natural2Code. We refer the reader to Appendix 10.3 for a comprehensive list of our evaluation benchmarks. Even so, model performance on these benchmarks gives us an indication of the model capabilities and where they may provide impact on real-world tasks. For example, Gemini Ultra’s impressive reasoning and STEM competencies pave the way for advancements in LLMs within the educational domain3. The ability to tackle complex mathematical and scientific concepts opens up exciting possibilities for personalized learning and intelligent tutoring systems.
Even so, model performance on these benchmarks gives us an indication of the model capabilitie and where they may provide impact on real-world tasks. For example, Gemini Ultra’s impressiv reasoning and STEM competencies pave the way for advancements in LLMs within the educationa domain3. The ability to tackle complex mathematical and scientific concepts opens up excitin possibilities for personalized learning and intelligent tutoring systems.
# 5.1.2. Trends in Capabilities
We investigate the trends in capabilities across the Gemini model family by evaluating them on a holistic harness of more than 50 benchmarks in six different capabilities, noting that some of the most notable benchmarks were discussed in the last section. These capabilities are: “Factuality” covering open/closed-book retrieval and question answering tasks; “Long-Context” covering longform summarization, retrieval and question answering tasks; “Math/Science” including tasks for mathematical problem solving, theorem proving, and scientific exams; “Reasoning” tasks that require arithmetic, scientific, and commonsense reasoning; “Multilingual” tasks for translation, summarization, and reasoning in multiple languages. Several of these capabilities are targeted by post-training (Section 6). Please see Appendix 10.3 for a detailed list of tasks included for each capability.
3See demos on website https://deepmind.google/gemini.
Gemini
Ultra
Gemini
Pro
GPT-4
GPT-3.5
PaLM 2-L
Claude 2
Inflect-
ion-2
Grok 1
LLAMA-2
MMLU
Multiple-choice questions
in 57 subjects
(professional &
academic)
(Hendrycks et al., 2021a)
90.04%
CoT@32∗
83.7%
5-shot
79.13%
CoT@8∗
71.8%
5-shot
87.29%
CoT@32
(via API∗∗)
86.4%
5-shot
(reported)
70%
5-shot
78.4%
5-shot
78.5%
5-shot CoT
79.6%
5-shot
73.0%
5-shot
68.0%∗∗∗
GSM8K
Grade-school math
(Cobbe et al., 2021)
94.4%
Maj1@32
86.5%
Maj1@32
92.0%
SFT &
5-shot CoT
57.1%
5-shot
80.0%
5-shot
88.0%
0-shot
81.4%
8-shot
62.9%
8-shot
56.8%
5-shot
MATH
Math problems across
5 difficulty levels &
7 subdisciplines
(Hendrycks et al., 2021b)
53.2%
4-shot
32.6%
4-shot
52.9%
4-shot
(via API∗∗)
50.3%
(Zheng et al.,
2023)
34.1%
4-shot
(via API∗∗)
34.4%
4-shot
—
34.8%
23.9%
4-shot
13.5%
4-shot
BIG-Bench-Hard
Subset of hard BIG-bench
tasks written as CoT prob-
lems
(Srivastava et al., 2022)
83.6%
3-shot
75.0%
3-shot
83.1%
3-shot
(via API∗∗)
66.6%
3-shot
(via API∗∗)
77.7%
3-shot
—
—
—
51.2%
3-shot
HumanEval
Python coding tasks
(Chen et al., 2021)
74.4%
0-shot
(PT∗∗∗∗)
67.7%
0-shot
(PT∗∗∗∗)
67.0%
0-shot
(reported)
48.1%
0-shot
—
70.0%
0-shot
44.5%
0-shot
63.2%
0-shot
29.9%
0-shot
Natural2Code
Python code generation.
(New held-out set with no
leakage on web)
74.9%
0-shot
69.6%
0-shot
73.9%
0-shot
(via API∗∗)
62.3%
0-shot
(via API∗∗)
—
—
—
—
—
DROP
Reading comprehension
& arithmetic.
(metric: F1-score)
(Dua et al., 2019)
82.4
Variable
shots
74.1
Variable
shots
80.9
3-shot
(reported)
64.1
3-shot
82.0
Variable
shots
—
—
—
—
HellaSwag
(validation set)
Common-sense multiple
choice questions
(Zellers et al., 2019)
87.8%
10-shot
84.7%
10-shot
95.3%
10-shot
(reported)
85.5%
10-shot
86.8%
10-shot
—
89.0%
10-shot
—
80.0%∗∗∗
WMT23
Machine translation (met-
ric: BLEURT)
(Tom et al., 2023)
74.4
1-shot
(PT∗∗∗∗)
71.7
1-shot
73.8
1-shot
(via API∗∗)
—
72.7
1-shot
—
—
—
—
Table 2 | Gemini performance on text benchmarks with external comparisons and PaLM 2-L. ∗The model produces a chain of thought with k = 8 or 32 samples, if there is a consensus above a threshold (chosen based on the validation split), it selects this answer, otherwise it reverts to a greedy sample. Further analysis in Appendix 10.2. ∗∗Results self-collected via the API in Nov, 2023. ∗∗∗Results shown use the decontaminated numbers from Touvron et al. (2023b) report as the most relevant comparison to Gemini models which have been decontaminated as well.) ∗∗∗∗PT denotes a post-trained Gemini API model.
We observe consistent quality gains with increased model size in Figure 3, especially in reasoning, math/science, summarization and long-context. Gemini Ultra is the best model across the board for all six capabilities. Gemini Pro, the second-largest model in the Gemini family of models, is also quite competitive while being a lot more efficient to serve.
5.1.3. Nano
# 5.1.3. Nano
Bringing AI closer to the user, we discuss the Gemini Nano 1 and Nano 2 models engineered for on-device deployments. These models excel in summarization and reading comprehension tasks with per-task fine-tuning. Figure 3 shows the performance of these pre-trained models in comparison to the much larger Gemini Pro model, while Table 3 dives deeper into specific factuality, coding, Math/Science, and reasoning tasks. Nano-1 and Nano-2 model sizes are only 1.8B and 3.25B parameters respectively. Despite their size, they show exceptionally strong performance on factuality, i.e. retrieval-related tasks, and significant performance on reasoning, STEM, coding, multimodal and
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d7ab/d7ab79bf-87da-4e8f-8152-b75b6f2db07e.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3 | Language understanding and generation performance of Gemini model family acros different capabilities (normalized by the Gemini Pro model).</div>
<div style="text-align: center;">multilingual tasks. With new capabilities accessible to a broader set of platforms and devices, the Gemini models expand accessibility to everyone.</div>
multilingual tasks. With new capabilities accessible to a broader set of platforms and devices, the Gemini models expand accessibility to everyone.
Gemini Nano 1
Gemini Nano 2
accuracy
normalized
by Pro
accuracy
normalized
by Pro
BoolQ
71.6
0.81
79.3
0.90
TydiQA (GoldP)
68.9
0.85
74.2
0.91
NaturalQuestions (Retrieved)
38.6
0.69
46.5
0.83
NaturalQuestions (Closed-book) 18.8
0.43
24.8
0.56
BIG-Bench-Hard (3-shot)
34.8
0.47
42.4
0.58
MBPP
20.0
0.33
27.2
0.45
MATH (4-shot)
13.5
0.41
22.8
0.70
MMLU (5-shot)
45.9
0.64
55.8
0.78
Table 3 | Performance of Gemini Nano series on factuality, summarization, reasoning, coding and STEM tasks compared to significantly larger Gemini Pro model.
# 5.1.4. Multilinguality
The multilingual capabilities of the Gemini models are evaluated using a diverse set of tasks requiring multilingual understanding, cross-lingual generalization, and the generation of text in multiple languages. These tasks include machine translation benchmarks (WMT 23 for high-medium-low resource translation; Flores, NTREX for low and very low resource languages), summarization benchmarks (XLSum, Wikilingua), and translated versions of common benchmarks (MGSM: professionally translated into 11 languages).
# 5.1.4.1 Machine Translation
Translation is a canonical benchmark in machine learning with a rich history. We evaluated a posttrained Gemini API Ultra model (see Section 6.5.3) on the entire set of language pairs in the WMT 23 translation benchmark in a few-shot setting. Overall, we found that Gemini Ultra (and other Gemini models) performed remarkably well at translating from English to any other language, and surpassed
the LLM-based translation methods when translating out-of-English, on high-resource, mid-resource and low-resource languages. In the WMT 23 out-of-English translation tasks, Gemini Ultra achieved the highest LLM-based translation quality, with an average BLEURT (Sellam et al., 2020) score of 74.8 compared to GPT-4’s score of 73.6, and PaLM 2’s score of 72.2. When averaged across all language pairs and directions for WMT 23, we see a similar trend with Gemini Ultra 74.4, GPT-4 73.8 and PaLM 2-L 72.7 average BLEURT scores on this benchmark.
WMT 23
(Avg BLEURT)
Gemini Ultra
Gemini Pro
Gemini Nano 2
Gemini Nano 1
GPT-4
PaLM 2-L
High Resource
74.2
71.7
67.7
64.1
74.0
72.6
Mid Resource
74.7
71.8
67.0
64.8
73.6
72.7
Out-of-English
74.8
71.5
66.2
65.2
73.6
72.2
Into-English
73.9
72.0
69.0
63.5
74.1
73.4
All languages
74.4
71.7
67.4
64.8
73.8
72.7
In addition to the languages and translation tasks above, we also evaluate Gemini Ultra on very low-resource languages. These languages were sampled from the tail of the following language sets: Flores-200 (Tamazight and Kanure), NTREX (North Ndebele), and an internal benchmark (Quechua). For these languages, both from and into English, Gemini Ultra achieved an average chrF score of 27.0 in 1-shot setup, while the next-best model, PaLM 2-L, achieved a score of 25.3.
# 5.1.4.2 Multilingual Math and Summarization
Beyond translation, we evaluated how well Gemini models perform in challenging tasks across a range of languages. We specifically investigated the math benchmark MGSM (Shi et al., 2023), which is a translated variant of the math benchmark GSM8K (Cobbe et al., 2021). We find Gemini Ultra achieves an accuracy of 79.0%, an advance over PaLM 2-L which scores 74.7%, when averaged across all languages in an 8-shot setup. We also benchmark Gemini models on the multilingual summarization benchmarks – XLSum (Hasan et al., 2021) and WikiLingua (Ladhak et al., 2020). In XLSum, Gemini Ultra reached an average of 17.6 rougeL score compared to 15.4 for PaLM 2. For Wikilingua, Gemini Ultra (5-shot) trails behind PaLM 2 (3-shot) measured in BLEURT score. See Table 5 for the full results. Overall the diverse set of multilingual benchmarks show that Gemini family models have a broad language coverage, enabling them to also reach locales and regions with low-resource languages.
Gemini Ultra
Gemini Pro
GPT-4
PaLM 2-L
MGSM
(8-shot)
79.0
63.5
74.5
74.7
XLsum
(3-shot)
17.6
16.2
—
15.4
Wikilingua
48.9
47.8
—
50.4
nce of Gemini models on multilingual math and summarizat
Table 5 | Performance of Gemini models on multilingual math and summarization.
# 5.1.5. Long Context
Gemini models are trained with a sequence length of 32,768 tokens and we find that they make use of their context length effectively. We first verify this by running a synthetic retrieval test: we place key-value pairs at the beginning of the context, then add long filler text, and ask for value associated with a particular key. We find that the Ultra model retrieves the correct value with 98% accuracy when queried across the full context length. We further investigate this by plotting the negative log
likelihood (NLL) versus the token index across a held-out set of long documents in Figure 4. We find that the NLL decreases with sequence position up to the full 32K context length. The longer context length of Gemini models enable new use cases such as retrieval over documents and video understanding discussed in Section 5.2.2.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d9e9/d9e9ea4e-25c7-4f6c-b103-6322a1ce6e27.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4 | Negative log likelihood as a function of token index across 32K context length on a held-ou set of long documents.</div>
Figure 4 | Negative log likelihood as a function of token index across 32K context l set of long documents.
# 5.1.6. Factuality
Factuality (Maynez et al., 2020) is a key focus of our model’s training and deployment. We evaluate three aspects of factuality for our Gemini API models:
1. Closed-Book Factuality: If provided with a fact-seeking prompt without any given source, Gemini API models should not hallucinate incorrect information (see Section 2 of Roberts et al. (2020) for a definition). These prompts can range from information-seeking prompts (e.g. “Who is the prime minister of India?”) to semi-creative prompts that may request factual information (e.g. “Write a 500-word speech in favor of the adoption of renewable energy”). 2. Attribution: If instructed to generate a response grounded to a given context, we aim to ensure that Gemini API models produce a response with the highest degree of faithfulness to the context (Maynez et al., 2020; Rashkin et al., 2023). This may include the summarization of a user-provided source, generating fine-grained citations given a question and provided snippets akin to Menick et al. (2022); Peng et al. (2023), answering questions from a long-form source such as a book (Mihaylov et al., 2018), and transforming a given source to a desired output (e.g. an email from a portion of a meeting transcript). 3. Hedging: If prompted with an input that is “unanswerable”, Gemini API models must acknowledge that it cannot provide a response by hedging to avoid hallucination. These include scenarios where the input prompt contains false-premise questions [see examples in Hu et al. (2023)], the input prompt instructs the model to perform open book QA, but the answer is not derivable from the given context, and so forth.
Factuality is evaluated via human annotators who fact-check each response manually; we report the percentage of factually inaccurate responses as judged by annotators. Attribution is evaluated via human annotators who check for attribution to sources in the prompt for each response manually; the reported metric is AIS (Rashkin et al., 2023). For hedging, we use an automatic evaluation setup where we measure whether models hedge accurately. We compare Gemini API Pro with a version without any factuality-focused adaptation in Table 6. We see that the rate of inaccuracy is halved in the factuality set, the accuracy of attribution is increased
by 50% from the attribution set, and the model successfully hedges 70% (up from 0%) in the pro hedging set task.
by 50% from the attribution set, and the model successfully hedges 70% (up from 0%) in the provided hedging set task.
Factuality
(Inaccurate Rate)
Attribution
(AIS)
Hedging
(Accuracy)
Gemini API Pro
No factuality-focused adaptation
6.7%
[5.8%, 7.8%]
40.2%
[37.9%, 42.5%]
0%
Gemini API Pro
Final stage of post-training
3.8%
[3.1%, 4.8%]
60.0%
[57.6%, 62.1%]
69.3%
Table 6 | Factuality mitigations: Impact of post-training on the rate of inaccuracy, presence of attributio and the rate of accurate hedging on Gemini API Pro (with corresponding 95% confidence intervals)
# 5.1.7. Complex Reasoning Systems
Gemini models can also be combined with additional techniques such as search and tool-use to create powerful reasoning systems that can tackle more complex multi-step problems. One example of such a system is AlphaCode 2, a new state-of-the-art agent that excels at solving competitive programming problems (Leblond et al, 2023). AlphaCode 2 uses a specialized version of Gemini Pro – tuned on competitive programming data similar to the data used in Li et al. (2022) – to conduct a massive search over the space of possible programs. This is followed by a tailored filtering, clustering and reranking mechanism. Gemini Pro is fine-tuned both to be a coding model to generate proposal solution candidates, and to be a reward model that is leveraged to recognize and extract the most promising code candidates. AlphaCode 2 is evaluated on Codeforces,4 the same platform as AlphaCode, on 12 contests from division 1 and 2, for a total of 77 problems. AlphaCode 2 solved 43% of these competition problems, a 1.7x improvement over the prior record-setting AlphaCode system which solved 25%. Mapping this to competition rankings, AlphaCode 2 built on top of Gemini Pro sits at an estimated 85th percentile on average – i.e. it performs better than 85% of entrants. This is a significant advance over AlphaCode, which only outperformed 50% of competitors. The composition of powerful pre-trained models with search and reasoning mechanisms is an exciting direction towards more general agents; another key ingredient is deep understanding across a range of modalities which we discuss in the next section.
# 5.2. Multimodal
Gemini models are natively multimodal. These models exhibit the unique ability to seamlessly combine their capabilities across modalities (e.g. extracting information and spatial layout out of a table, a chart, or a figure) with the strong reasoning capabilities of a language model (e.g. its state-of-art-performance in math and coding) as seen in examples in Figures 5 and 14. The models also show strong performance in discerning fine-grained details in inputs, aggregating context across space and time, and applying these capabilities over a temporally-related sequence of video frames and/or audio inputs. The sections below provide more detailed evaluation of the model across different modalities (image, video, and audio), together with qualitative examples of the model’s capabilities for image generation and the ability to combine information across different modalities.
# 5.2.1. Image Understanding
We evaluate post-trained Gemini API models on four different capabilities: high-level object recognition using captioning or question-answering tasks such as VQAv2; fine-grained transcription using tasks such as TextVQA and DocVQA requiring the model to recognize low-level details; chart understanding requiring spatial understanding of input layout using ChartQA and InfographicVQA tasks; and multimodal reasoning using tasks such as Ai2D, MathVista and MMMU. For zero-shot QA evaluation, the model is instructed to provide short answers aligned with the specific benchmark. All numbers are obtained using greedy sampling and without any use of external OCR tools.
Gemini
Ultra
(pixel only)
Gemini
Pro
(pixel only)
Gemini
Nano 2
(pixel only)
Gemini
Nano 1
(pixel only)
GPT-4V
Prior SOTA
MMMU (val)
Multi-discipline college-level problems
(Yue et al., 2023)
59.4%
pass@1
62.4%
Maj1@32
47.9%
32.6%
26.3%
56.8%
56.8%
GPT-4V, 0-shot
TextVQA (val)
Text reading on natural images
(Singh et al., 2019)
82.3%
74.6%
65.9%
62.5%
78.0%
79.5%
Google PaLI-3, fine-tuned
DocVQA (test)
Document understanding
(Mathew et al., 2021)
90.9%
88.1%
74.3%
72.2%
88.4%
(pixel only)
88.4%
GPT-4V, 0-shot
ChartQA (test)
Chart understanding
(Masry et al., 2022)
80.8%
74.1%
51.9%
53.6%
78.5%
(4-shot CoT)
79.3%
Google DePlot, 1-shot PoT
(Liu et al., 2023)
InfographicVQA (test)
Infographic understanding
(Mathew et al., 2022)
80.3%
75.2%
54.5%
51.1%
75.1%
(pixel only)
75.1%
GPT-4V, 0-shot
MathVista (testmini)
Mathematical reasoning
(Lu et al., 2023)
53.0%
45.2%
30.6%
27.3%
49.9%
49.9%
GPT-4V, 0-shot
AI2D (test)
Science diagrams
(Kembhavi et al., 2016)
79.5%
73.9%
51.0%
37.9%
78.2%
81.4%
Google PaLI-X, fine-tuned
VQAv2 (test-dev)
Natural image understanding
(Goyal et al., 2017)
77.8%
71.2%
67.5%
62.7%
77.2%
86.1%
Google PaLI-X, fine-tuned
Table 7 | Image understanding Gemini Ultra consistently outperforms existing approaches even in zero-shot, especially for OCR-related image understanding tasks for natural images, text, documents, and figures without using any external OCR engine (‘pixel only’). Many existing approaches fine-tune on the respective tasks, highlighted in gray, which makes the comparison with 0-shot not apples-toapples.
We find that Gemini Ultra is state of the art across a wide range of image-understanding benchmarks in Table 7. It achieves strong performance across a diverse set of tasks such as answering questions on natural images and scanned documents as well as understanding infographics, charts and science diagrams. When compared against publicly reported results from other models (most notably GPT-4V), the Gemini model is better in zero-shot evaluation by a significant margin. It also exceeds several existing models that are specifically fine-tuned on the benchmark’s training sets for the majority of tasks. The capabilities of the Gemini models lead to significant improvements in the state of the art on academic benchmarks like MathVista (+3.1%)5 or InfographicVQA (+5.2%). MMMU (Yue et al., 2023) is a recently released evaluation benchmark, which consists of questions about images across 6 disciplines with multiple subjects within each discipline that require collegelevel knowledge to solve these questions. Gemini Ultra achieves the best score on this benchmark advancing the state-of-the-art result by more than 5 percentage points and outperforms the previous best result in 5 of 6 disciplines (see Table 8), thus showcasing its multimodal reasoning capabilities.
MMMU (val)
Gemini Ultra (0-shot)
GPT-4V (0-shot)
Maj@32
pass@1
pass@1
Art & Design
74.2
70.0
65.8
Business
62.7
56.7
59.3
Science
49.3
48.0
54.7
Health & Medicine
71.3
67.3
64.7
Humanities & Social Science
78.3
78.3
72.5
Technology & Engineering
53.0
47.1
36.7
Overall
62.4
59.4
56.8
<div style="text-align: center;">Table 8 | Gemini Ultra performance on the MMMU benchmark (Yue et al., 2023) per discipline. Each discipline covers multiple subjects, requiring college-level knowledge and complex reasoning.</div>
Gemini models are also capable of operating across modalities and a diverse set of global languages simultaneously, both for image understanding tasks (e.g., images containing text in Icelandic) and for generation tasks (e.g., generating image descriptions for a wide range of languages). We evaluate the performance of generating image descriptions on a selected subset of languages in the Crossmodal3600 (XM-3600) benchmark in a 4-shot setting, using the Flamingo evaluation protocol (Alayrac et al., 2022), without any fine-tuning for all models. As shown in Table 9, Gemini models achieve a significant improvement over the existing best model, Google PaLI-X.
XM-3600 (CIDER)
Gemini Ultra
4-shot
Gemini Pro
4-shot
Google PaLI-X
4-shot
English
86.4
87.1
77.8
French
77.9
76.7
62.5
Hindi
31.1
29.8
22.2
Modern Hebrew
54.5
52.6
38.7
Romanian
39.0
37.7
30.2
Thai
86.7
77.0
56.0
Chinese
33.3
30.2
27.7
Average (of 7)
58.4
55.9
45.0
Table 9 | Multilingual image understanding Gemini models outperform existing models in captionin images in many languages when benchmarked on a subset of languages in XM-3600 dataset (Thapliya et al., 2022).
5MathVista is a comprehensive mathematical reasoning benchmark consisting  datasets and three newly created datasets. Our MathVista results were obtaine evaluation script.
5MathVista is a comprehensive mathematical reasoning benchmark consisting of 28 previously published multim datasets and three newly created datasets. Our MathVista results were obtained by running the MathVista au evaluation script.
5MathVista is a comprehensive mathematical reasoning benchmark consisting of 28 previously published multimodal datasets and three newly created datasets. Our MathVista results were obtained by running the MathVista authors’ evaluation script.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0a18/0a186068-67f9-4f76-8c24-6382ed4c687a.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c777/c777e574-431e-443d-9e5c-af4bb453a7bc.png" style="width: 50%;"></div>
Figure 5 | Using Gemini models’ multimodal reasoning capabilities to generate matplotlib code for rearranging the subplots. The multimodal prompt is shown at the top-left in gray. Gemini Ultra’s response, including its generated code, is shown in the right column in blue. The bottom left figure shows rendered version of the generated code. Successfully solving this task shows the model’s capability to combine several capabilities: (1) recognition of the functions depicted in the plots; (2) inverse graphics to infer the code that would have generated the subplots; (3) instruction-following to put subplots in their desired positions; and (4) abstract reasoning to infer that the exponential plot must stay in its original place, because the sine plot must move out of the way for the 3-dimensional plot.
Qualitative evaluation in Figure 5 illustrates an example of Gemini Ultra’s multimodal reasoning capabilities. The model is required to solve the task of generating matplotlib code that would rearrange
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7265/72656312-fc22-49a2-bc48-bc60177004b5.png" style="width: 50%;"></div>
a set of subplots provided by the user. The model output shows that it successfully solves this task combining multiple capabilities of understanding the user plot, inferring the code required to generate it, following user instructions to put subplots in their desired positions, and abstract reasoning about the output plot. This highlights Gemini Ultra’s native multimodality and alludes to its more complex reasoning abilities across interleaved sequences of image and text. We refer the reader to the appendix for more qualitative examples.
# 5.2.2. Video Understanding
Understanding video input is an important step towards a useful generalist agent. We measure the video understanding capability across several established benchmarks that are held-out from training. These tasks measure whether the model is able to understand and reason over a temporally-related sequence of frames. For each video task, we sample 16 equally-spaced frames from each video clip and feed them to the Gemini models. For the YouTube video datasets (all datasets except NextQA and the Perception test), we evaluate the Gemini models on videos that were still publicly available in the month of November, 2023. Gemini Ultra achieves state-of-the-art performance on various few-shot video captioning tasks as well as zero-shot video question answering tasks as shown in Table 10. This demonstrates its capability of strong temporal reasoning across several frames. Figure 23 in the appendix provides a qualitative example of understanding the video of the ball-striking mechanics of a soccer player and reasoning about the player can improve their game.
Task
Gemini Ultra
Gemini Pro
Few-shot SoTA
VATEX (test)
62.7
57.4
56.0
English video captioning
(Wang et al., 2019)
4-shots
4-shots
DeepMind Flamingo, 4-shots
VATEX ZH (test)
51.3
50.0
–
Chinese video captioning
(Wang et al., 2019)
4-shots
4-shots
YouCook2 (val)
135.4
123.2
74.5
English cooking video captioning
(Zhou et al., 2018)
4-shots
4-shots
DeepMind Flamingo, 4-shots
NextQA (test)
29.9
28.0
26.7
Video question answering
(Xiao et al., 2021)
0-shot
0-shot
DeepMind Flamingo, 0-shot
ActivityNet-QA (test)
52.2
49.8
45.3
Video question answering
(Yu et al., 2019)
0-shot
0-shot
Video-LLAVA, 0-shot
Perception Test MCQA (test)
54.7
51.1
46.3
Video question answering
(Pătrăucean et al., 2023)
0-shot
0-shot
SeViLA (Yu et al., 2023), 0-shot
Table 10 | Few-shot video understanding across tasks and languages on selected academic benchmarks. The reported metric is CIDER for video captioning, WUPS for NextQA, and top-1 accuracy for the Perception Test and ActivityNet-QA. For ActivityNet-QA, we use the Video-LLAVA (Lin et al., 2023) evaluation protocol.
# 5.2.3. Image Generation
Gemini models are able to output images natively, without having to rely on an intermediate natural language description that can bottleneck the model’s ability to express images. This uniquely enables the model to generate images with prompts using interleaved sequences of image and text in a
few-shot setting. For example, the user might prompt the model to design suggestions of images and text for a blog post or a website (see Figure 12 in the appendix). Figure 6 shows an example of image generation in 1-shot setting. Gemini Ultra model is prompted with one example of interleaved image and text where the user provides two colors (blue and yellow) and image suggestions of creating a cute blue cat or a blue dog with yellow ear from yarn. The model is then given two new colors (pink and green) and asked for two ideas about what to create using these colors. The model successfully generates an interleaved sequence of images and text with suggestions to create a cute green avocado with pink seed or a green bunny with pink ears from yarn.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/767a/767a3a5f-91fc-4aa7-8ec4-2c1496304bbc.png" style="width: 50%;"></div>
Figure 6 | Image Generation. Gemini models can output multiple images interleaved with text given a prompt composed of image and text. In the left figure, Gemini Ultra is prompted in a 1-shot setting with a user example of generating suggestions of creating cat and dog from yarn when given two colors, blue and yellow. Then, the model is prompted to generate creative suggestions with two new colors, pink and green, and it generates images of creative suggestions to make a cute green avocado with pink seed or a green bunny with pink ears from yarn as shown in the right figure.
# 5.2.4. Audio Understanding
We evaluate the Gemini Nano-1 and Gemini Pro models on a variety of public benchmarks and compare it with Universal Speech Model (USM) (Zhang et al., 2023) and Whisper (large-v2 (Radford et al., 2023) or large-v3 (OpenAI, 2023) as indicated). These benchmarks include automatic speech recognition (ASR) tasks such as FLEURS (Conneau et al., 2023), VoxPopuli, (Wang et al., 2021), Multi-lingual Librispeech (Pratap et al., 2020), as well as the speech translation task CoVoST 2, translating different languages into English (Wang et al., 2020). We also report on an internal benchmark YouTube test set. ASR tasks report a word error rate (WER) metric, where a lower number is better. Translation tasks report a BiLingual Evaluation Understudy (BLEU) score, where a higher number is better. FLEURS is reported on 62 languages that have language overlap with the training data. Four segmented languages (Mandarin, Japanese, Korean and Thai) report character error rate (CER), instead of WER, similar to Whisper (Radford et al., 2023). Table 11 indicates that our Gemini Pro model significantly outperforms the USM and Whisper models across all ASR and AST tasks, both for English and multilingual test sets. Note that there is a large gain in FLEURS, compared to USM and Whisper, as our model is also trained with the FLEURS training dataset. However, training the same model without FLEURS dataset results in a WER of 15.8, which still outperforms Whisper. Gemini Nano-1 model also outperforms both USM and Whisper on all datasets except FLEURS. Note that we did not evaluate Gemini Ultra on audio yet, though we expect better performance from increased model scale.
Task
Metric
Gemini
Pro
Gemini
Nano-1
Whisper
(OpenAI, 2023;
Radford et al.,
2023)
USM
(Zhang
et
al.,
2023)
Automatic Speech
Recognition
YouTube
(en-us)
WER (↓)
4.9%
5.5%
6.5%
(v3)
6.2%
Multilingual
Librispeech
(en-us)
(Pratap et al., 2020)
WER (↓)
4.8%
5.9%
6.2%
(v2)
7.0 %
FLEURS
(62 lang)
(Conneau et al., 2023)
WER (↓)
7.6%
14.2%
17.6%
(v3)
11.8%
VoxPopuli
(14 lang)
(Wang et al., 2021)
WER (↓)
9.1%
9.5%
15.9%
(v2)
13.4%
Automatic Speech
Translation
CoVoST 2
(21 lang)
(Wang et al., 2020)
BLEU (↑)
40.1
35.4
29.1
(v2)
30.7
Table 11 | Speech evaluation results on selected benchmarks for ASR and AST. For ASR, the reported
metric is WER where lower is better. For AST, the reported metric is BLEU where higher is better.
Table 12 shows further error analysis with USM and Gemini Pro. We find that Gemini Pro produces
more understandable responses, particularly on rare words and proper nouns.
<div style="text-align: center;">Table 11 | Speech evaluation results on selected benchmarks for ASR and AST. For ASR, the reported metric is WER where lower is better. For AST, the reported metric is BLEU where higher is better. Table 12 shows further error analysis with USM and Gemini Pro. We find that Gemini Pro produces more understandable responses, particularly on rare words and proper nouns.</div>
<div style="text-align: center;">Table 12 shows further error analysis with USM and Gemini Pro. We find that Gemini Pro produces more understandable responses, particularly on rare words and proper nouns.</div>
Domain
Truth
USM
Gemini Pro
Wav
Fleurs
Scotturb bus 403 travels regularly to
Sintra, stopping at Cabo da Roca.
Scotboard bus four3 traversed regu-
larly to Centra stopping at Cabo de
Roga.
Scotturb bus 403 travels regularly to
Sintra, stopping at Cabo da Roca.
▲
)))
Fleurs
The archipelago lies 120 km north
of the Peninsula. The largest is King
George Island, with the settlement of
Villa Las Estrellas.
The archipelago lines 120 km north
of peninsula. The largest is Kingurch
island with the settlement of Cua
Losas.
The archipelago lies 120 km north
of the Peninsula. The largest is King
George Island, with the settlement of
Villa Las Estrellas.
▲
)))
Table 12 | Qualitative examples for the ASR task in the benchmark. Incorrect transcript highlighted in red.
# 5.2.5. Modality Combination
Multimodal demonstrations often include a combination of text interleaved with a single modality, usually images. We demonstrate the ability to process a sequence of audio and images natively. Consider a cooking scenario about making an omelet where we prompt the model with a sequence of audio and images. Table 13 indicates a turn-by-turn interaction with the model, providing pictures and verbally asking questions about the next steps for cooking an omelet. We note that the model response text is reasonably accurate, and shows that model processes fine-grained image details to evaluate when the omelet is fully cooked. See demo on the website.
Input Image
Input Audio (transcribed)
Model Response: Text
▲
))) What’s the first step to make a veggie omelet
with these ingredients?
Crack the eggs into a bowl and whisk them.
▲
))) Thank you for the instructions. I started mak-
ing my omelet, does it look ready now?
It looks like it’s almost ready. You can flip it over
to cook the other side.
(No image - it’s a follow up
on the previous question)
▲
))) Why is it not ready?
It’s not ready because the eggs are still runny.
▲
))) What about now?
It looks ready now. You can take it off the heat and
serve it.
Table 13 | Audio-visual qualitative example showcasing the ability of Gemini models to process
Table 13 | Audio-visual qualitative example showcasing the ability of Gemini models to process interleaved sequences of text, vision, and audio, as well as reason across modalities. This example inputs interleaved images and audio from the user in a cooking scenario. The user prompts the model for instructions to make an omelet and to inspect whether it is fully cooked.
It looks ready now. You can take it off the heat and serve it.
# 6. Post-Training Models
After large-scale pre-training, we apply post-training, where one trains on top of a pre-trained model in order to extend the model’s proficiency and to enable a wide variety of capabilities. Namely, we seek to improve overall quality, enhance target capabilities such as coding and multilingual, and ensure alignment and safety criteria are met. We discuss our approach to post-training in this section, highlighting common and distinct aspects of the Gemini Apps and Gemini API model variants.
# 6.1. Gemini Apps: Gemini and Gemini Advanced
Gemini and Gemini Advanced offer direct access to Google’s family of AI models, consisting of the core post-trained Gemini Apps models and the system around it. These models are created by applying specialized post-training on top of Gemini pre-trained models: currently, Gemini gives access to Pro 1.0 and Gemini Advanced gives access to Ultra 1.0. Beyond the core models, the system determines how the models interact with external tools (such as Google Flights, Maps, and Google Workspace), and how to generate responses (filtering, ranking, and streaming). As an area, conversational AI presents several challenges, including: How to understand users’ requests across multi-turn interactions? How to make sure responses are safe, factually grounded, and helpful? How to help users accomplish tasks by using tools external to the models? We discuss how we approach these challenges in the following sections.
# 6.2. Gemini APIs: Google AI Studio and Cloud Vertex A
Our developer-focused Gemini API models are designed to support both conversational and nonconversational use cases. These models are available through Google AI Studio and Cloud Vertex AI through an easy to use API. Google AI Studio is a free, web-based developer tool to prototype and launch apps quickly with an API key. Vertex AI is a comprehensive AI platform that enables developers to leverage Gemini API models with varied tooling, fully-managed infrastructure, and built-in enterprise security and privacy settings. Gemini APIs make it easy to integrate Gemini API models into any production product or workflow, empowering developers to build applications that can reason across different modalities.
# 6.3. Post-Training Methods & Data
Post-training Gemini models to produce Gemini API and Apps variants involves several stages; see Figure 7. Careful data curation is critical for all stages. First, we collect a diverse set of prompts that are representative of real-world use cases. Second, we apply supervised fine-tuning (SFT) on demonstration data of what the model’s output should be for a given prompt (Mishra et al., 2021; Ouyang et al., 2022; Wei et al., 2022a). Third, we further collect different possible responses to a given prompt, and collect feedback data over these to train a Reward Model (RM). Finally, using the trained RM, a Reinforcement Learning from Human Feedback (RLHF) stage (Bai et al., 2022a) is applied to further align the model’s outputs with human preferences. We discuss our methods in more detail below: (1) Prompt Data Collection: A prompt is a user’s input to the model. As well as the most recent user input, this can also include previous user-model interactions. We curate datasets of target prompts. The datasets serve as the basis for our demonstration and feedback data collections, and they are used directly during reinforcement learning. It is important to cover a diverse set of crucial use cases and in both single-turn and multi-turn formats. Data sources include vendor-created data, third-party licensed sources, and synthetic approaches.
(2) SFT on Demonstration Data: SFT trains the model to output a desired target response given a prompt. Our Demonstration Data target responses can be directly written by a human expert, or generated by a model and in some cases revised or reviewed by a human. Additionally, we use data analysis tools and heuristics to ensure high data diversity across capabilities, use cases, and semantic clusters. (3) RM Training on Feedback Data: We further collect Feedback Data, for which human raters provide feedback such as relative preferences over candidate responses and feedback regarding individual responses to a given prompt. For many capabilities, rating relative preferences is an easier task than demonstrating an ideal response. Feedback data are collected across creativity, safety, factuality, other capabilities, and other target criteria. We found that the utility of the resulting human feedback data greatly depends on the prompt selection and the sampling strategy used to produce candidate responses. We use this data to train RMs to output rewards that align with human preferences as closely as possible. (4) RLHF: Applying reinforcement learning from human feedback (RLHF) to our models provides further gains over SFT alone. Our approach creates an iterative process in which RL continually pushes the boundaries of the RM, while the RM is continuously improved through evaluation and data collection, leading to progressive improvements in both.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6659/66596146-c059-4cb8-8c4e-9bc97d3c439a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7 | Modeling overview. Post-training utilizes an optimized data flywheel in order to acquire human-AI feedback and continually improve on key areas. The data mixtures for supervised finetuning, reward modeling, and reinforcement learning serve as the foundation for our models.</div>
# 6.4. Evaluation
Evaluation of human preferences over model outputs provides critical signals for measuring performance. As part of our development process, we conduct human evaluation extensively across targeted capabilities. Human evaluation is instantiated as side-by-side blind evaluations where human raters judge responses of two models to the same prompt, as single-response ratings for certain capabilities, and as online testing. In addition, we build models for automated evaluation that faithfully imitate human preferences in order to guide development and continuously monitor online performance.
# 6.5. Model Capabilities
Beyond the general post-training outlined above, we apply techniques to improve a set of key capabilities. These capabilities cover a range of use cases inspired by current user needs and research-inspired
future applications. We outline capability examples not detailed in previous sections below. The post training recipes are carefully designed to balance multiple objectives, including creativity, factuality safety and more (Bai et al., 2022b; Thoppilan et al., 2022). We have a particular focus on safety and alignment, and hence address this in a further dedicated section.
# 6.5.1. Instruction Following
Following a user’s prompt accurately is a fundamental capability for LLMs, especially as these models become more sophisticated and are presented with increasingly complex user prompts. User prompts vary in granularity, specificity, and requirements (e.g., content, format, length). Individual instructions can also be ambiguous, optional, or even impossible or undesirable to satisfy (He et al., 2023; Xu et al., 2023). We improve Gemini Apps and Gemini API models’ instruction following (IF) abilities by collecting data for a diverse set of instruction following categories. For instructions that are verifiable programmatically such as word count, we generate synthetic data via prompting and response editing to ensure that such instructions are satisfied. Complex prompts evaluation: We investigate performance on complex prompts containing multiple instructions using a fine-grained evaluation method that assesses how well models adhere to each instruction. Human raters are presented with a prompt-response pair and a list of the individual (sub)-instructions contained in the prompt. Each prompt may have anywhere from one to dozens of individual instructions, and the annotators are tasked with determining whether each instruction is followed (or not) by the response. Table 14 reports results on an internal dataset of prompts with instructions of varying complexity that encompass a wide range of instructions and are designed to be challenging for LLMs. We report two metrics: per-instruction accuracy (the percentage of sub instructions in the eval set that are followed), and full-response accuracy (the percentage of eval set prompts where all sub-instructions are followed).
Following a user’s prompt accurately is a fundamental capability for LLMs, especially as these models become more sophisticated and are presented with increasingly complex user prompts. User prompts vary in granularity, specificity, and requirements (e.g., content, format, length). Individual instructions can also be ambiguous, optional, or even impossible or undesirable to satisfy (He et al., 2023; Xu et al., 2023). We improve Gemini Apps and Gemini API models’ instruction following (IF) abilities by collecting data for a diverse set of instruction following categories. For instructions that are verifiable programmatically such as word count, we generate synthetic data via prompting and response editing to ensure that such instructions are satisfied.
Complex prompts evaluation: We investigate performance on complex prompts containing multiple instructions using a fine-grained evaluation method that assesses how well models adhere to each instruction. Human raters are presented with a prompt-response pair and a list of the individual (sub)-instructions contained in the prompt. Each prompt may have anywhere from one to dozens of individual instructions, and the annotators are tasked with determining whether each instruction is followed (or not) by the response. Table 14 reports results on an internal dataset of prompts with instructions of varying complexity that encompass a wide range of instructions and are designed to be challenging for LLMs. We report two metrics: per-instruction accuracy (the percentage of sub instructions in the eval set that are followed), and full-response accuracy (the percentage of eval set prompts where all sub-instructions are followed).
Post-trained PaLM 2
Gemini (with Pro)
Gemini Advanced (with Ultra)
Per-instruction accuracy
59.5±3.0%
77.8±2.0%
87.4±1.4%
Full-response accuracy
25.5±3.3%
38.5±3.6%
54.1±3.7%
Table 14 | Performance of Gemini on our complex prompts instruction-following internal benchmark. Gemini Advanced (with Ultra) achieves an average per-instruction accuracy close to 90%, representing a significant improvement over Gemini (with Pro) and a post-trained PaLM 2 model. We find that the sub-instructions that aren’t followed are well-distributed across responses. As a result Gemini Advanced’s full-response accuracy is lower, at around 54%. This indicates that there is further headroom for models to fully satisfy all instructions.
# 6.5.2. Tool Use
By training LLMs to use tools, we greatly expand LLM capabilities beyond their internal knowledge. We treat tool use for both Gemini Apps and Gemini API models as a code generation problem, leveraging the base model’s preexisting strong coding capabilities. Every tool invocation is represented as a code block in which tool calls are invoked. This process allows the model to both compose multiple tools in each code block, as well as observe and react to the results of tool execution. At inference time, to generate a response to a user prompt, our system executes the loop shown in Figure 8, where sampling from the LLM and execution of tool code work together to create a final response.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a6d2/a6d2be0b-1b4f-44ab-8de0-6a55fecbdb99.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8 | A Gemini tool-use control loop.</div>
Gemini Apps models: Gemini draws on a range of tools via Gemini Extensions, including Google Workspace, Google Maps, YouTube, Google Flights, and Google Hotels. These tool-use capabilities also enable Gemini to be integrated as part of Gmail, Docs, Slides, Sheets and more. We are aiming to bring further tool-use capabilities in order to both enhance Gemini models and integrate Gemini models into further products. We created an internal benchmark to assess Gemini performance on tasks that may benefit from access to these extensions. This benchmark measures human preference in domains such as travel planning and video discovery. We find models equipped with tools are preferred on this set 78% of the time over models without tools (excluding ties). Gemini API models: We have found that fine-tuning Gemini API models is very effective at teaching the model tool-use behaviors. Furthermore, training models to use programming and search as tools leads to improved performance on a range of academic benchmarks. In Table 15, we compare tool-use models fine-tuned from an early version of Gemini API Pro against equivalent models that do not use tools.
Gemini API models: We have found that fine-tuning Gemini API models is very effective a teaching the model tool-use behaviors. Furthermore, training models to use programming and search as tools leads to improved performance on a range of academic benchmarks. In Table 15, we compare tool-use models fine-tuned from an early version of Gemini API Pro against equivalent models that do not use tools.
Mathematical Reasoning
Factuality & Knowledge
Retrieval
GSM8K
Cobbe et al. (2021)
MATH
Hendrycks
et
al.
(2021b)
NQ
Kwiatkowski et al.
(2019b)
Realtime QA
Kasai et al. (2022a)
Gemini API Pro
with tools
80.1%
41.8%
68.0%
70.8%
Gemini API Pro
without tools
69.7%
30.7%
59.0%
39.2%
Table 15 | Comparison between Gemini API tool-use models and comparable models that do not us ools. Gemini API Pro without tools is an early version of our Pro model trained without tool-use dat Gemini API Pro with tools is the same model fine-tuned with tool-use data.
# 6.5.3. Multilinguality
Multilinguality is critical to make sure Gemini models effectively support a wide range of languages. We discuss our key approaches for Gemini Apps and Gemini API models respectively below. Gemini Apps models: Scaling Gemini from English to 40+ languages imposed research challenges in data quality. We leverage abundant high-quality English data by localization to native cultures (e.g., “president of the United States” -> “ 日本の首相”). Table 16 shows the performance of Gemini (with Pro) on 5 languages compared to Bard with
an older post-training recipe and based on PaLM 2. For side-by-side comparisons between a model A and a model B, we calculate a metric called SxS score. Each rating is converted to an ordinal value centered at 0: ratings preferring A are positive and ratings preferring B are negative over a scale between -1.5 and 1.5. The converted values are averaged to return the SxS score. Intuitively, a positive SxS score indicates the extent to which model A is preferred over model B. Here, we find quality improved by more than 0.1 SxS score for all five languages. Coding and reasoning gains from Gemini Pro are preserved across languages.
Language
Quality
SxS
Coding
MBPP Pass@1
Austin et al. (2021)
Reasoning
MMLU
Hendrycks
et
al.
(2021a)
ja-JP
+0.14
+22.2%
+3.6%
pt-BR
+0.17
+23.2%
+5.2%
de-DE
+0.1
+21.4%
+7.5%
es-419
+0.12
+22.8%
+9.3%
it-IT
+0.13
+13.8%
+7.5%
Table 16 | Multilingual performance of Gemini (with Pro) compared to Gemini with an older po training recipe and PaLM 2.
Gemini API models: Similar to Gemini Apps models, we train Gemini API models on additional multilingual post-training data, effectively adapting the original English model for use in various languages. We experiment with both human-generated non-English prompt-response pairs as well as automatically translated pairs. For the latter, we leverage abundant high-quality English demonstration data by translation. We ensure the quality of such translated data by translationability filtering and response rating by humans. Translatability Filtering: Not all prompt-response pairs make sense when automatically translated, and may require expensive localization instead. Example prompts of this type (responses omitted for space) include:
• (strict word requirements) Write a 1000 word essay about world peace. • (too English centric) Write a poem in iambic pentameter about apples. • (too Latin-script centric) What is a word with 1 E, 2 As, and 1 U?
Translation Quality Validation: Each translated prompt-response pair was rated for translation quality by at least 3 human raters, and was kept in the final mixture if the majority of raters rated it as accurate. Section 5.1.4 reports evaluations of the multilingual capabilities of post-trained Gemini API models.
# 6.5.4. Multimodal Vision
Multimodal post-training enhances the capabilities of our natively multimodal Gemini models for a wide range of useful applications. In the following, we discuss how image understanding ability is incorporated into Gemini Apps and Gemini API models. For this evaluation, we further train both of these Gemini model variants on a mixture of text data and expert curated image-text data over several vertically-defined multimodal use cases Gemini Apps models: We empower Gemini and Gemini Advanced with image understanding capabilities by fine-tuning pre-trained Gemini models on a mixture of text-only and image-text data. Careful balancing of text and multimodal data ensures the model develops robust image understanding without adversely affecting the quality of the text-only interactions. To assess our
Multimodal post-training enhances the capabilities of our natively multimodal Gemini models for  wide range of useful applications. In the following, we discuss how image understanding ability i incorporated into Gemini Apps and Gemini API models. For this evaluation, we further train both of these Gemini model variants on a mixture of text data and expert curated image-text data ove several vertically-defined multimodal use cases
Gemini Apps models: We empower Gemini and Gemini Advanced with image understanding capabilities by fine-tuning pre-trained Gemini models on a mixture of text-only and image-tex data. Careful balancing of text and multimodal data ensures the model develops robust image understanding without adversely affecting the quality of the text-only interactions. To assess ou
models, we compile a dataset of human-curated and synthetic image-text prompts and responses, spanning various categories and difficulty levels. This dataset facilitates human evaluation for model comparison and selection. We find that introducing this image-text data preserves Gemini Apps model quality on text-only tasks, with a SxS score on text-only tasks of +0.01±0.01 for a Gemini Apps Pro model trained on this data versus an equivalent model trained only on text data. In addition, post-training via RLHF improves performance on multimodal tasks, with a SxS score on image-understanding tasks of +0.223±0.06 for a Gemini Apps Pro model post-trained with SFT & RLHF vs SFT alone. Gemini API models: We evaluate the impact of post-training via SFT on Gemini API models’ multimodal vision performance by tracking the performance of both pre-trained models and posttrained Gemini API Vision models on a series of standard benchmarks. These post-trained results have already been given in Table 7, in Table 17 we further report the difference in performance between pre-trained and post-trained Gemini API models.
Gemini Ultra
Pre-trained only
0-shot
(pixel only)
Gemini API Ultra
0-shot
(pixel only)
Gemini Ultra
pre- to post-trained
improvement
MMMU (val)
Multi-discipline college-level problems
(Yue et al., 2023)
n/a
59.4%
pass@1
62.4%
Maj1@32
n/a
TextVQA (val)
Text reading on natural images
(Singh et al., 2019)
81.4%
82.3%
+0.9%
DocVQA (test)
Document understanding
(Mathew et al., 2021)
90.1%
90.9%
+0.8%
ChartQA (test)
Chart understanding
(Masry et al., 2022)
80.8%
80.8%
0.0%
InfographicVQA (test)
Infographic understanding
(Mathew et al., 2022)
77.9%
80.3%
+2.4%
MathVista (testmini)
Mathematical reasoning
(Lu et al., 2023)
n/a
53.0%
n/a
AI2D (test)
Science diagrams
(Kembhavi et al., 2016)
76.6%
79.5%
+2.9%
VQAv2 (test-dev)
Natural image understanding
(Goyal et al., 2017)
74.5%
77.8%
+3.3%
Table 17 | Post-trained model image understanding Post-training improves image understanding capabilities of Gemini API Ultra over the base pre-trained model. Comparisons of Gemini API Ultra to other models on these benchmarks are given in Table 7.
The results indicate that the pre-trained model already has high performance across the capabilities represented by these benchmarks, in line with previous observations. However, the post-training SFT stage used for the Gemini API Vision models succeeds in improving the performance over several of these benchmarks (InfographicVQA, AI2D, VQAv2), most likely due to the model’s increased instruction-following capabilities that succeed in aligning the model output style with that of the golden references.
# 6.5.5. Coding
Despite the strong coding benchmark performance of the base model, post-training data still provides a significant boost to both code quality and code correctness. This highlights the benefit of high-quality demonstration data and feedback data for coding use cases. Gemini Apps and Gemini API models use a combination of human and synthetic approaches to collect such data. We evaluate our Gemini Apps models’ coding performance on a set of internally curated prompts, distributed across code use cases and languages. Table 18 reports SxS scores, where Gemini (with Pro) significantly improves upon Bard with an older post-training recipe and based on PaLM 2. Gemini Advanced (with Ultra) further improves upon Gemini (with Pro).
We evaluate our Gemini Apps models’ coding performance on a set of internally curated prompts distributed across code use cases and languages. Table 18 reports SxS scores, where Gemini (with Pro) significantly improves upon Bard with an older post-training recipe and based on PaLM 2. Gemini Advanced (with Ultra) further improves upon Gemini (with Pro).
Side A
Side B
SxS score
Gemini (with Pro)
Bard (PaLM 2, Sept. 2023)
0.19±0.03
Gemini Advanced (with Ultra)
Gemini (with Pro)
0.13± 0.02
For the coding capabilities of post-trained Gemini API Models, see Table 2 which reports thei academic benchmark performance.
# 7. Responsible Deployment
During the development of Gemini models, we follow a structured approach to responsible deployment to identify, measure, and manage foreseeable downstream societal impacts of our models, in line with previous releases of Google’s AI technology (Kavukcuoglu et al., 2022). Throughout the lifecycle of a project, we follow the structure below. This section provides more detail about our approach and includes key findings where available. We are committed to ongoing transparency and will continue to provide updated information on our approach and testing in upcoming reports.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7d4d/7d4d8bc5-cb2f-42a4-ba23-2eb1c9848982.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
At Google we apply an impact assessment framework throughout the product development lifecycle related to Google’s AI Principles (Google, 2023). This means we assess the risk and impact of AI models we’re building at both a model-level (e.g. for Gemini API Ultra 1.0, as deployed on Cloud
Studio or Vertex AI), and once embedded within a broader product or service (e.g. for Gemini Advanced).
7.1.1. Model Assessment
# 7.1.1. Model Assessment
We conduct model impact assessments to identify, assess, and document societal benefits and harms associated with the capabilities of Gemini models. Our impact assessments for Gemini API models describe downstream benefits and risks that we identify, spanning across the models’ modalities (text-to-text; image-to-text; and video-to-text). Model impact assessments are conducted by the Google DeepMind Responsible Development and Innovation team, and are reviewed by the Google DeepMind Responsibility and Safety Council. We draw from various sources in producing impact assessments, including a wide range of literature, external expertise, and our in-house ethics and safety research. Gemini models introduce various benefits to people and society. Gemini models’ various modalities, including language, image and video understanding, can help users process information more efficiently, for example through content summarisation. These efficiency benefits can apply to commercial entities, and can assist use cases dependent on text, image or video processing such as video captioning, analytics or product descriptions. Video and image understanding modalities can also be deployed for social good applications downstream, such as enabling descriptions of visual outputs for accessibility purposes. Generative multimodal models may also raise downstream societal risks, with the Gemini models assessments considering a range of risks previously identified within research such as Weidinger et al. (2021) and Shelby et al. (2023). We assessed a range of content risks such as exposure of users to potentially unsafe content, such as sexually explicit, violent or hateful outputs (Weidinger et al., 2021), child safety harms, and representation harms, subsequently designing evaluations across these domains to enable measurement. Beyond content related risks, we analyzed the potential misuse of capabilities for surveillance applications, particularly for mediato-text capabilities, and considered the broader environmental and economic impact of multimodal models. We are continuously conducting research into emerging risks of advanced models, including for dangerous capabilities (e.g. cyber security threats) which form a part of our evaluation approach (Section 7.4).
# 7.1.2. Product Assessments
Beyond the assessment conducted at the model-level, additional risk assessments are conducted on the products by the Google AI Principles team prior to launch (e.g. on the Gemini Advanced product) These risk and impact assessments, alongside both model- and product-level assurance evaluations are used to guide mitigation and product delivery efforts, and inform deployment decisions.
Beyond the assessment conducted at the model-level, additional risk assessments are conducted on the products by the Google AI Principles team prior to launch (e.g. on the Gemini Advanced product). These risk and impact assessments, alongside both model- and product-level assurance evaluations, are used to guide mitigation and product delivery efforts, and inform deployment decisions. For Gemini Advanced, we conducted extensive deep-dive red teaming via dogfooding and adversarial testing in the areas of safety, accountability, and inclusion to prepare for the initial experimental rollout of Gemini and subsequent updates. Further cross-functional work helps to ensure appropri-
Beyond the assessment conducted at the model-level, additional risk assessments are conducted on the products by the Google AI Principles team prior to launch (e.g. on the Gemini Advanced product). These risk and impact assessments, alongside both model- and product-level assurance evaluations, are used to guide mitigation and product delivery efforts, and inform deployment decisions. For Gemini Advanced, we conducted extensive deep-dive red teaming via dogfooding and adversarial testing in the areas of safety, accountability, and inclusion to prepare for the initial experimental rollout of Gemini and subsequent updates.