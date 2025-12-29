# Exploring the Integration of Large Language Models into Automatic Speech Recognition Systems: An Empirical Study
Zeping Min1 and Jinbo Wang2
1 Peking University No.5 Yiheyuan Road, Haidian District, Beijing 100871, P.R.China zpm@pku.edu.cn 2 Peking University No.5 Yiheyuan Road, Haidian District, Beijing 100871, P.R.China wangjinbo@stu.pku.edu.cn
Abstract. This paper explores the integration of Large Language Models (LLMs) into Automatic Speech Recognition (ASR) systems to improve transcription accuracy. The increasing sophistication of LLMs, with their in-context learning capabilities and instruction-following behavior, has drawn significant attention in the field of Natural Language Processing (NLP). Our primary focus is to investigate the potential of using an LLM’s in-context learning capabilities to enhance the performance of ASR systems, which currently face challenges such as ambient noise, speaker accents, and complex linguistic contexts. We designed a study using the Aishell-1 and LibriSpeech datasets, with ChatGPT and GPT-4 serving as benchmarks for LLM capabilities. Unfortunately, our initial experiments did not yield promising results, indicating the complexity of leveraging LLM’s in-context learning for ASR applications. Despite further exploration with varied settings and models, the corrected sentences from the LLMs frequently resulted in higher Word Error Rates (WER), demonstrating the limitations of LLMs in speech applications. This paper provides a detailed overview of these experiments, their results, and implications, establishing that using LLMs’ in-context learning capabilities to correct potential errors in speech recognition transcriptions is still a challenging task at the current stage.
Keywords: Automatic Speech Recognition · Large Language Models · In-Context Learning
# 1 Introduction
In today’s era of cutting-edge technology, automatic speech recognition (ASR) systems have become an integral part. The advent of end-to-end ASR models, which are based on neural networks [8,13,4,10,6,3,11,12], coupled with the rise of prominent toolkits such as ESPnet [29] and WeNet [32], have spurred the progression of ASR technology. Nevertheless, ASR systems [26,25,16,22,34,14,21]
occasionally yield inaccurate transcriptions, which can be attributed to ambient noise, speaker accents, and complex linguistic contexts, thus limiting their effectiveness. Over the years, considerable emphasis has been placed on integrating a language model [15,30] into the ASR decoding process. Language models have gradually evolved from statistical to neural. Recently, large language models (LLMs) [35,23,33,1,19,18,27] have gained prominence due to their exceptional proficiency in a wide array of NLP tasks. Interestingly, when the parameter scale surpasses certain thresholds, these LLMs not only improve their performance but also exhibit unique features such as in-context learning and instruction following, thereby offering a novel interaction method. Nevertheless, the attempts to leverage recent LLMs such as [19,18,27] to boost ASR model performance are still in the nascent stages. This paper seeks to address this gap. Our primary focus is to explore the potential of employing an LLM’s in-context learning capability to enhance ASR performance. Our methodology revolves around providing an appropriately designed instruction to the LLM, supplying it with the ASR transcriptions, and analyzing if it can rectify the mistakes. We employed the Aishell-1 and LibriSpeech datasets for our experiments and selected well-known LLM benchmarks, such as ChatGPT and GPT-4, which are generally considered superior to other LLMs for their comprehensive capabilities. We concentrated on the potential of GPT-3.5 and GPT-4 to correct possible errors in speech recognition transcriptions.3 Our initial experiments with the GPT-3.5-16k (GPT-3.5-turbo-16k-0613) model, in a one-shot learning scenario, did not yield lower WER. Consequently, we undertook further investigation using diverse settings, including variations in the LLM model (GPT-3.5-turbo-4k-0301, GPT-3.5-turbo4k-0613, GPT-4-0613), modification of instructions, increasing the number of attempts (1, 3, and 5), and varying the number of examples supplied to the model (1, 3, and 5-shot settings). This paper presents a thorough discussion of these experiments, their outcomes, and our insights. Regrettably, the findings of these experiments suggest that, at the present stage, directly employing the in-context learning ability of LLMs to correct potential errors in speech recognition transcriptions is extremely challenging and often leads to a higher WER. This may be due to the lack of ability of LLMs in speech transcription. This study contributes to the field in three ways:
1. Exploration of LLMs for ASR Improvement: We explore the potential of large language models (LLMs), particularly focusing on GPT-3.5 and GPT-4, to improve automatic speech recognition (ASR) performance by their in-context learning ability. This is an emerging area of research, and our work contributes to its early development.
3 Although we conducted preliminary trials with models like llama, opt, bloom, etc., these models often produced puzzling outputs and rarely yielded anticipated transcription corrections.
2. Comprehensive Experiments Across Various Settings: We conduct comprehensive experiments using the Aishell-1 and LibriSpeech datasets and analyze the effect of multiple variables, including different LLM models, alterations in instructions, varying numbers of attempts, and the number of examples provided to the model. Our work contributes valuable insights into the capabilities and limitations of LLMs in the context of ASR. 3. Evaluation of the Performance: Regrettably, our findings indicate that leveraging the in-context learning ability of LLMs to correct potential errors in speech recognition transcriptions often leads to a higher word error rate (WER). This critical evaluation underscores the current limitations of directly applying LLMs in the field of ASR, thereby identifying an important area for future research and improvement.
# 2 Related Work
The use of large language models (LLMs) to enhance the performance of automatic speech recognition (ASR) models has been the subject of numerous past studies [28,24,5,31,9,17]. These works have explored various strategies, including distillation methods [9,17] and rescoring methods [28,24,5,31]. In the distillation approach, for instance, [9] employed BERT in the distillation approach to produce soft labels for training ASR models. [17] strived to convey the semantic knowledge that resides within the embedding vectors. For rescoring methods, [24] adapted BERT to the task of n-best list rescoring. [5] redefined N-best hypothesis reranking as a prediction problem. [31] attempted to train a BERT-based rescoring model with MWER loss. [28] amalgamated LLM rescoring with the Conformer-Transducer model. However, the majority of these studies have employed earlier LLMs, such as BERT [7]. Given the recent explosive progress in the LLM field, leading to models with significantly more potent NLP abilities, such as ChatGPT, it becomes crucial to investigate their potential to boost ASR performance. Although these newer LLMs have considerably more model parameters, which can pose challenges to traditional distillation and rescoring methods, they also possess a crucial capability, in-context learning, which opens up new avenues for their application.
# 3 Methodology
Our approach leverages the in-context learning abilities of LLMs. We supply the LLMs with the ASR transcription results and a suitable instruction to potentially correct errors. The process can be formalized as:
where x represents the ASR transcription result, and y is the correct transcription. The pairs (xi, yi)k i=1 are the k examples given to the LLM, and I is the instruction provided to the LLM. The prompt is represented by (I, (x1, y1), (x2, y2) , ..., (xk, yk), x). The entire process is visually illustrated in Figure 1.
We conducted thorough experimentation, varying GPT versions, the design of the instruction, and the number of examples k provided to GPT, in order to assess the potential of using Large Language Models (LLMs) to improve Automatic Speech Recognition (ASR) performance. We tested three versions of GPT-3.5, as well as the high-performing GPT-4. We used four carefully crafted instructions and varied the number of examples, where k = 1, 2, 3, supplied to the LLM. Unfortunately, we found that directly applying the in-context learning capabilities of the LLM models for improving ASR transcriptions presents a significant challenge, and often leads to a higher Word Error Rate (WER). We further experimented with multiple attempts at sentence-level corrections. That is, for each transcription sentence x, the LLM generates multiple corrected outputs, and the final corrected result of the transcription sentence x is chosen as the output with the least WER.4 Regrettably, even with multiple attempts, the corrected output from the LLM still results in a higher WER, further substantiating the challenges associated with directly leveraging the LLM’s in-context learning capabilities for enhancing ASR transcriptions.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d88e/d88e0daa-7d6e-42ee-b990-deaa6dbf4b58.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/58ad/58ad65e4-f9f2-4313-9e0f-70f79d495fe8.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 1. Overview of the methodology leveraging the in-context learning capability of large language models (LLMs) for potential correction of errors in automatic speech recognition (ASR) transcriptions.</div>
Fig. 1. Overview of the methodology leveraging the in-context learning capability of large language models (LLMs) for potential correction of errors in automatic speech recognition (ASR) transcriptions.
 Selecting the output with the lowest WER is not practical in real-world scenarios, as we cannot know the actual transcription y. Nonetheless, this technique aids in comprehending the limitations of using LLM’s in-context learning capabilities for enhancing ASR transcriptions.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1249/124999fa-6239-4745-89ff-3619111c458b.png" style="width: 50%;"></div>
# 4 Experiments
# 4.1 Setup
Dataset For our investigation, we selected two distinct datasets to evaluate the efficacy of utilizing advanced LLMs to improve ASR performance: the Aishell-1 dataset for the Chinese language and the LibriSpeech dataset for the English language. These datasets are greatly appreciated in the ASR research field, serving as standard benchmarks for numerous studies and methodologies. The Aishell-1 [2] dataset has a total duration of 178 hours, with the precision of manual transcriptions exceeding 95%. The dataset is meticulously organized into training, development, and testing subsets. In contrast, the LibriSpeech [20] dataset comprises approximately 1000 hours of English speech sampled at 16kHz. The content is extracted from audiobooks as a part of the LibriVox project. Similar to Aishell-1, LibriSpeech is also partitioned into subsets for training, development, and testing. Furthermore, each subset is classified into two groups based on data quality: clean and other.
ASR Model To ensure the applicability of our experimental results, we utilized a state-of-the-art hybrid CTC/attention architecture, highly regarded in the field of speech recognition. We employed pretrained weights provided by the Wenet [32] speech community. The ASR model, trained on the Aishell-1 dataset, includes an encoder set up with a swish activation function, four attention heads, and 2048 linear units. The model employs an 8-kernel CNN module with layer normalization, and normalizes the input layer before activation. The encoder consists of 12 blocks, has an output size of 256, and uses gradient clipping (value=5) to prevent gradient explosions. The model leverages the Adam optimizer with a learning rate of 0.001 and a warm-up learning rate scheduler that escalates the learning rate for the initial 25,000 steps. The ASR model, trained on the Librispeech dataset, implements a bitransformer decoder and a conformer encoder. The encoder follows the same configuration as that of the Aishell-1 model. The decoder incorporates four attention heads, with a dropout rate of 0.1. The model adheres to the same optimization and learning rate strategies as the Aishell-1 model.
LLM For the LLM models, considering that ChatGPT and GPT-4 are recognized benchmarks, we inspected three versions from ChatGPT (GPT-3.5-turbo4k-0301, GPT-3.5-turbo-4k-0613, GPT-3.5-turbo-16k-0613) and GPT-4 (GPT4-0613). While other LLMs such as Llama, Opt, and Bloom claim to equal or outperform ChatGPT in certain aspects, they generally fall behind ChatGPT, and even more so GPT-4, in terms of overall competency for generic tasks. For the instruction I, we tested four variations, as detailed in Table 1. Concerning the examples input to the LLM, we assessed 1-shot, 2-shot, and 3-shot scenarios. For the number of attempts, we explored situations with 1attempt, 3-attempts, and 5-attempts.
Instruction ID
Description
Instruction 1
Correct the following transcription from speech recognition.
Instruction 2
Now, you are an ASR transcription checker. You should correct
all possible errors from transcriptions from speech recognition
models. These errors tend to appear where the semantics do not
make sense.
Instruction 3
I have recently started using a speech recognition model to rec-
ognize some speeches. Of course, these recognition results may
contain some errors. Now, you are an ASR transcription checker,
and I need your help to correct these potential mistakes. You
should correct all possible errors from transcriptions from speech
recognition models. These errors often occur where the seman-
tics do not make sense and can be categorized into three types:
substitution, insertion, and deletion.
Instruction 4
I have recently been using a speech recognition model to rec-
ognize some speeches. Naturally, these recognition results may
contain errors. You are now an ASR transcription checker, and
I require your assistance to correct these potential mistakes.
Correct all possible errors from transcriptions provided by the
speech recognition models. These errors typically appear where
the semantics don’t make sense and can be divided into three
types: substitution, insertion, and deletion. Please use ’[]’ to en-
close your final corrected sentences.
Since the LLM output may contain some irrelevant content with ASR transcription, for testing convenience, we devised a method suite to extract transcriptions from LLM output. Specifically, from the prompt perspective, we tell the model to enclosed the corrected transcription in ’[ ]’, either by presenting the example to the model or directing the model in the instruction. After the LLM generates the text, we initially extract the text within ’[ ]’ from the LLM output text. In the following step, we eliminate all the punctuation within it. This is because the ground truth transcription provided by the Aishell-1 dataset and Librispeech dataset does not include punctuation.
# 4.2 Results
In our preliminary experiments, we established a baseline using the GPT-3.5 (GPT-3.5-turbo-16k-0613 version) model. We employed the Instruction 1: Correct the following transcription from speech recognition. We employed a single attempt with one-shot learning. The outcomes of these initial tests, as shown in Table 2, were unsatisfactory.
Table 2. WER (%) results using GPT-3.5-16k-0613 for ASR transcription correction with one-shot learning.
Aishell-1
LibriSpeech
Clean
Other
with LLM
12.36
47.93
51.25
without LLM
4.73
3.35
8.77
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d0b9/d0b9fee5-0aee-479e-b934-8fadc778c049.png" style="width: 50%;"></div>
Furthermore, we provide several examples from LibriSpeech Clean in Figure 2. These instances underscore the challenges the LLM encounters when interpreting and correcting ASR transcriptions, which result in unsatisfactory
performance for our task. In the first example, the original transcription read, "STUFFED INTO YOU HIS BELLY COUNSELLED HIM". Yet, the LLM amended "YOU" to "HIS" in the corrected transcription, deviating from the ground truth. In the second instance, the original transcript stated, "NUMBER TEN FRESH NELLIE IS WAITING ON YOU GOOD NIGHT HUSBAND". However, the LLM altered "FRESH NELLIE" to "EXPRESS DELI", thereby significantly modifying the intended meaning. In the third case, the initial transcription was "HELLO BERTIE ANY GOOD IN YOUR MIND". However, the LLM misinterpreted "BERTIE" as "BIRDIE" and superfluously appended "IDEA" to the corrected transcription.
Results with Different LLM Models Initially, we varied the LLM models utilized in our experiments, considering three different versions: GPT-3.5-turbo4k-0301, GPT-3.5-turbo-4k-0613, and GPT-3.5-turbo-16k-0613 models. The results are consolidated in Table 3. The observations from Table 3 demonstrate that all LLM models have a higher Word Error Rate (WER) than the scenario without the utilization of an LLM. This finding suggests that while LLM models exhibit potential for a broad range of NLP applications, their application for error correction in ASR transcriptions still requires refinement. Notably, the WER for all LLM models in the Aishell-1 dataset is significantly higher than the WER in the scenario without an LLM. A similar pattern is evident in the LibriSpeech dataset, for both clean and other data. Furthermore, the performance of GPT-3.5-turbo-4k0613 and GPT-3.5-turbo-16k-0613 models is markedly better than that of the GPT-3.5-turbo-4k-0301 model. This disparity could be due to the enhancements in the GPT-3.5 model.
Table 3. WER (%) performance comparison of different LLM models on ASR transcription error correction.
Aishell-1
LibriSpeech
Clean
Other
GPT-3.5-turbo-4k-0301
16.05
57.83
51.20
GPT-3.5-turbo-4k-0613
12.32
47.57
51.10
GPT-3.5-turbo-16k-0613
12.36
47.93
51.25
without LLM
4.73
3.35
8.77
Results with Varying Instructions Next, we carried out a series of experiments using a variety of instructions. We precisely constructed four different types of instructions, which are displayed in Table 1. These instructions gradually provided more specific guidance for the task. We utilized the GPT-3.5-turbo-
16k-0613 model for this purpose. The outcomes for the different instructions are tabulated in Table 4. Furthermore, we tested varying instructions with two different models, specifically GPT-3.5-turbo-4k-0301 and GPT-3.5-turbo-4k-0613, the results of which are included in Appendix 5. Our findings suggested that supplying detailed instructions to the Language Model (LLM) improves its performance. However, even with extremely detailed instructions, the LLM model does not demonstrate adequate performance in the task of rectifying errors in speech recognition transcriptions. That is to say, the Word Error Rate (WER) escalates after correction.5
Table 4. WER (%) comparison for varying instructions with the GPT-3.5-turbo-16 0613 model.
<div style="text-align: center;">Table 4. WER (%) comparison for varying instructions with the GPT-3.5-turbo-16k0613 model.</div>
Aishell-1
LibriSpeech
Clean
Other
Instruction 1
12.36
47.93
51.25
Instruction 2
34.08
48.58
64.60
Instruction 3
22.32
37.21
48.14
Instruction 4
12.22
23.93
17.17
without LLM
4.73
3.35
8.77
Results with Varying Shots Subsequently, we explored the impact of varying the number of examples given to the model, using 1-shot, 2-shot, and 3-shot configurations. We utilized the GPT-3.5-turbo-16k-0613 model. For instructions, we employed the most detailed Instruction 4, which was proven to yield superior results in Subsection 4.2. The results are encapsulated in Table 5. Moreover, we also tested Instructions 1, 2, and 3, with the outcomes detailed in Appendix 5. Our observations revealed that providing the model with more examples led to enhanced performance, aligning with findings observed in many NLP tasks involving LLM. However, for the 1-shot, 2-shot, and 3-shot scenarios we experimented with, none of them resulted in a satisfactory WER, indicating an increase in errors post-correction. This is consistent with our previous observation that more progress is needed to harness LLMs effectively for ASR transcription error correction.
Results with Varying Attempts In the previous subsections, we established that the performance of Language Model Large (LLMs) in correcting errors in
<div style="text-align: center;">Table 5. WER (%) comparison for varying shots with Instruction 4 and the GPT-3.5turbo-16k-0613 model.</div>
Table 5. WER (%) comparison for varying shots with Instruction 4 and the GPTturbo-16k-0613 model.
Aishell-1
LibriSpeech
Clean
Other
1-shot
12.22
23.93
17.17
2-shot
14.19
23.38
17.68
3-shot
12.71
22.68
17.43
without LLM
4.73
3.35
8.77
Automatic Speech Recognition (ASR) transcriptions is currently unsatisfactory, as corrections generally increase the number of errors. To deepen our understanding of the LLM’s limitations in error correction for ASR transcriptions, we conducted further tests allowing the model multiple attempts. Specifically, for each transcription sentence x, the LLM generates multiple corrected outputs, and the final corrected result of the transcription sentence x is chosen as the output with the least Word Error Rate (WER). In practical applications, choosing the output with the lowest WER is not feasible, as the correct transcription y is unknown. Nevertheless, this approach aids in elucidating the constraints of leveraging LLM’s in-context learning capabilities for ASR transcription enhancement. We present the results for 1, 3, and 5 attempts in Table 6. We utilized the GPT-3.5-turbo-16k-0613 model. For instructions, we employed the most effective Instruction 4 from Subsection 4.2. Additionally, in the prompt, we provided the model with three examples, that is, the 3-shot setup. Refer to Table 6 for the experimental results. We discovered that even with up to five trials allowed, and the optimal result taken on a per-sentence basis, the outputs of the LLM still introduce more errors.
<div style="text-align: center;">Table 6. WER (%) comparison for varying attempts with Instruction </div>
Aishell-1
LibriSpeech
Clean
Other
1 Attempt
12.71
22.68
17.43
3 Attempts
6.81
17.50
12.54
5 Attempts
5.77
15.90
11.49
without LLM
4.73
3.35
8.77
GPT4 Experimentations We further extended our study to include the latest GPT4 model, currently deemed the most advanced. Due to the high computational demand and RPM restrictions of GPT4, we limited our testing to the LibriSpeech clean test set. We conducted tests using a one-shot setting for the
four detailed instructions provided in Table 1. The outcomes are encapsulated in Table 7. Our findings indicated that, despite employing the state-of-the-art GPT4 model, the ASR transcriptions corrected with LLM still yielded a higher number of errors.
<div style="text-align: center;">Table 7. WER (%) results with the GPT4 model for the LibriSpeech clean test set.</div>
Instruction 1
Instruction 2
Instruction 3
Instruction 4
Without LLM
28.97
23.91
16.76
14.90
3.35
# 5 Conclusion
This paper has provided an exploratory study on the potential of Large Language Models (LLMs) to rectify errors in Automatic Speech Recognition (ASR) transcriptions. Our research focused on employing renowned LLM benchmarks such as GPT-3.5 and GPT-4, which are known for their extensive capabilities. Our experimental studies included a diverse range of settings, variations in the LLM models, changes in instructions, and a varied number of attempts and examples provided to the model. Despite these extensive explorations, the results were less than satisfactory. In many cases, sentences corrected by LLMs resulted in higher Word Error Rates (WERs), thus revealing the limitations of LLMs in speech applications. This outcome points to the significant challenges in directly leveraging the in-context learning abilities of LLMs to improve ASR transcriptions. These findings do not imply that the application of LLMs in ASR technology should be dismissed. On the contrary, they suggest that further research and development are required to optimize the use of LLMs in this area. As LLMs continue to evolve, their capabilities might be harnessed more effectively in the future to overcome the challenges identified in this study. In conclusion, while the use of LLMs for enhancing ASR performance is in its early stages, the potential for improvement exists. This study hopes to inspire further research in this field, with the aim of refining and improving the application of LLMs in ASR technology.
# References
1. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Advances in neural information processing systems 33, 1877–1901 (2020) 2. Bu, H., Du, J., Na, X., Wu, B., Zheng, H.: Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline. In: 2017 20th conference of the oriental chapter of the international coordinating committee on speech databases and speech I/O systems and assessment (O-COCOSDA). pp. 1–5. IEEE (2017)
34. Zhang, B., Wu, D., Yao, Z., Wang, X., Yu, F., Yang, C., Guo, L., Hu, Y., Xie, L., Lei, X.: Unified streaming and non-streaming two-pass end-to-end model for speech recognition. arXiv preprint arXiv:2012.05481 (2020) 35. Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X.V., et al.: Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068 (2022)
# Appendix
# Results with Varying Instructions
We conducted experiments with various instructions. Four distinct types of instructions were meticulously designed, as depicted in Table 1. These instructions progressively provided more detailed task directives. The experimental results for the GPT-3.5-turbo-4k-0301 and GPT-3.5-turbo-4k-0613 models, under the conditions of these four instructions, are presented in Table 8 and Table 9, respectively. Our findings suggest that supplying the LLM model with detailed instructions aids in achieving enhanced performance. However, even with highly detailed instructions, the LLM model’s performance in the task of correcting speech recognition transcription errors is not satisfactory. That is to say, the Word Error Rate (WER) increases post-correction.
Table 8. WER comparison for varying instructions with the GPT-3.5-turbo-4k-030 model.
Aishell-1
LibriSpeech
Clean
Other
Instruction 1
16.05
57.83
51.20
Instruction 2
16.81
30.85
36.99
Instruction 3
14.12
24.42
26.19
Instruction 4
14.16
25.56
20.26
without LLM
4.73
3.35
8.77
Table 9. WER comparison for varying instructions with the GPT-3.5-turbo-4k-06 model.
Table 9. WER comparison for varying instructions with the GPT-3.5-turbo-4k-0613 model.
Aishell-1
LibriSpeech
Clean
Other
Instruction 1
12.32
47.57
51.10
Instruction 2
34.61
48.33
65.06
Instruction 3
23.19
37.05
48.10
Instruction 4
12.13
23.07
17.18
without LLM
4.73
3.35
8.77
We evaluated the effect of varying the number of examples provided to the model, using 1-shot, 2-shot, and 3-shot configurations. We employed the GPT-3.5-turbo16k-0613 model for this purpose. Tables 10, 11, and 12 depict the experimental results using Instructions 1, 2, and 3, respectively. Our findings suggest that providing more examples to the model leads to improved performance. This is consistent with results observed in numerous NLP tasks involving LLM. However, in the 1-shot, 2-shot, and 3-shot scenarios we tested, none yielded a satisfactory WER, indicating an increase in errors after correction. This aligns with our previous observation that additional efforts are required to effectively employ LLMs for ASR transcription error correction.
We evaluated the effect of varying the number of examples provided to the model, using 1-shot, 2-shot, and 3-shot configurations. We employed the GPT-3.5-turbo16k-0613 model for this purpose. Tables 10, 11, and 12 depict the experimental results using Instructions 1, 2, and 3, respectively.
Our findings suggest that providing more examples to the model leads to improved performance. This is consistent with results observed in numerous NLP tasks involving LLM. However, in the 1-shot, 2-shot, and 3-shot scenarios we tested, none yielded a satisfactory WER, indicating an increase in errors after correction. This aligns with our previous observation that additional efforts are required to effectively employ LLMs for ASR transcription error correction.
Table 10. WER comparison for varying shots with Instruction 1 and the GPT-3.5turbo-16k-0613 model.
Aishell-1
LibriSpeech
Clean
Other
1-shot
12.36
47.93
51.25
2-shot
20.67
70.93
73.32
3-shot
38.49
81.43
76.58
without LLM
4.73
3.35
8.77
<div style="text-align: center;">Table 11. WER comparison for varying shots with Instruction 2 and the GPT-3.5turbo-16k-0613 model.</div>
Aishell-1
LibriSpeech
Clean
Other
1-shot
34.08
48.58
64.60
2-shot
45.70
80.04
94.20
3-shot
76.48
80.39
90.79
without LLM
4.73
3.35
8.77
Table 12. WER comparison for varying shots with Instruction 3 and the G turbo-16k-0613 model.
<div style="text-align: center;">Table 12. WER comparison for varying shots with Instruction 3 and the GPT-3.5turbo-16k-0613 model.</div>
Aishell-1
LibriSpeech
Clean
Other
1-shot
22.32
37.21
48.14
2-shot
52.52
67.10
72.88
3-shot
86.49
66.78
69.46
without LLM
4.73
3.35
8.77
