# Shortcomings of LLMs for Low-Resource Translation: Retrieval and Understanding are Both the Problem
Sara Court The Ohio State University court.22@osu.edu
Sara Court The Ohio State University court.22@osu.edu Micha Elsner The Ohio State University elsner.14@osu.edu
Abstract
This work investigates the in-context learning abilities of pretrained large language models (LLMs) when instructed to translate text from a low-resource language into a high-resource language as part of an automated machine translation pipeline. We conduct a set of experiments translating Southern Quechua to Spanish and examine the informativity of various types of context retrieved from a constrained database of digitized pedagogical materials (dictionaries and grammar lessons) and parallel corpora. Using both automatic and human evaluation of model output, we conduct ablation studies that manipulate (1) context type (morpheme translations, grammar descriptions, and corpus examples), (2) retrieval methods (automated vs. manual), and (3) model type. Our results suggest that even relatively small LLMs are capable of utilizing prompt context for zero-shot low-resource translation when provided a minimally sufficient amount of relevant linguistic information. However, the variable effects of context type, retrieval method, model type, and language-specific factors highlight the limitations of using even the best LLMs as translation systems for the majority of the world’s 7,000+ languages and their speakers.
 24 Oct 2024
arXiv:2406.15625v3 
# 1 Introduction
Despite great progress in the quality of today’s state of the art machine translation (MT) systems, constraints on the amount and kinds of data available in the majority of the world’s 7,000+ languages have led to yet another disparity in access and support for speakers of these languages: low-resource MT continues to be a major challenge (Hendy et al., 2023; Nicholas and Bhatia, 2023; Robinson et al., 2023; Stap and Araabi, 2023). Although many languages lack the kinds of large, standardized corpora necessary for traditional MT methods, recent work suggests it may be possible to leverage a smaller amount of existing resources, for example
Micha Elsner The Ohio State University elsner.14@osu.edu
pedagogical materials used for language instruction, to develop MT systems with Large Language Models (LLMs), albeit with varying results (Elsner and Needle, 2023; Tanzer et al., 2024; Zhang et al., 2024). These materials are often the result of community-driven or government-led initiatives to support language revitalization, reclamation, and mother-tongue education (Schreiner et al., 2020; Liu et al., 2022; Riestenberg et al., 2024). Such discrepancies in the needs and priorities of academic, commercial, and community-led efforts to develop digital resources and language technologies is what Gessler (2022) terms the “NLP Gap”. In this study, we investigate one way to lessen the NLP Gap, comparing LLMs’ in-context learning abilities when translating from a low-resource language (a Peruvian variety of Southern Quechua) to a high-resource language (Spanish) using information retrieved from a database of pedagogical materials. We replicate results of earlier studies on a new language pair by comparing the effects of morpheme translations, sentences from a parallel corpus, and passages from a grammar instruction document on translation quality. We then conduct a more focused analysis by annotating translation outputs by hand using a modified MQM error typology (Burchardt, 2013). Finally, we conduct an ablation study on the effects of automated retrieval by manually constructing prompts using the same set of materials. Our results suggest that while, unsurprisingly, translation quality improves with model size, such improvements seem to primarily be the result of previous exposure to the low-resource language during model pretraining, rather than an improved ability for the model to utilize prompt context, as evidenced by high scores in response to baseline (zero-shot) translation prompts. However, we also find evidence that in-context learning abilities may be inconsistent across different models of similar size. As found in previous studies, prompts contain-
ing morpheme and word-level translations reliably improve model outputs, but information from the grammar and corpus have a null or even negative effect on results. Human evaluation on a selection of outputs from two models – GPT-3.5 Turbo and GPT-4o – align with the quantitative measures we obtain using BLEURT (Sellam et al., 2020) as an automatic metric. Quantitative results also show an effect of automated retrieval on translation quality that is most evident in prompts containing morpheme translations and for models with lower baseline scores. Finally, we highlight a number of ethical concerns and limitations that arise from the proposed methods that are supported by our findings, and discuss the potential risks and challenges LLM-based methods for low-resource MT face moving forward.
# 2 LLMs for Machine Translation
Modern LLMs are now capable of translating many high-resource languages, but lack sufficient coverage of even modestly resourced languages to achieve comparable results without additional support (Kocmi et al., 2023). Retrieval-augmented generation (Rubin et al., 2022) may provide such support in the form of parallel sentences (Agrawal et al., 2022), dictionary definitions (Ghazvininejad et al., 2023; Lu et al., 2023) or other linguistic meta-knowledge such as a grammatical description. Retrieval-augmented methods offer exciting possibilities for low-resource translation, since the LLM might (in principle) be able to “teach itself” the language from learner-oriented resources produced by community members or language specialists. Studies to date (Elsner and Needle, 2023; Reid et al., 2024; Zhang et al., 2024) experiment with four dimensions of variability: source language, LLM, type(s) of information retrieved, and retrieval method. Since the source languages in these studies have relatively little presence in public corpora or on the web, differing results across LLMs can tentatively be attributed to differences in their incontext learning and instruction-following abilities. All studies find that word-level translations are helpful additions to prompts. Zhang et al. (2024) and Tanzer et al. (2024) also add sentence pairs from a parallel corpus, while Elsner and Needle (2023) add usage examples from a dictionary. Each improves results, although to a lesser degree. Elsner and Needle (2023) and Zhang et al. (2024) experiment with small fixed “grammar lesson” pas-
sages to provide explicit syntactic instruction, but find these ineffective. Tanzer et al. (2024) uses passages retrieved from a grammar book, also with relatively disappointing results. Reid et al. (2024) use the entire grammar book and a very long-context model to obtain better translations, but without exploring the role explicit grammar instruction actually plays in doing so. Zhang et al. (2024) find that sentences from the corpus retrieved using BM25 embeddings (Robertson et al., 2009) work better than random ones. Tanzer et al. (2024), however, report that retrieval with longest common substring (LCS) matching outperforms embedding-based retrieval. Overall, the question of how to best retrieve relevant passages containing grammar material or sentences in a low-resource language is still open. This also complicates the interpretation of the mostlynegative results found for grammar passages. It is not clear whether these stem from poor retrieval, from the LLM’s inability to process the retrieved content, or both. Moreover, although Reid et al. (2024) conducts human evaluation of the results for quality, to the best of our knowledge no study to date systematically investigates specific grammatical errors in the output. Finally, each of these studies finds a significant decrease in LLMs’ abilities to translate from a highresource language into a low-resource language relative to experiments in the opposite direction. This is in line with McCoy et al. (2023), who find that while the accuracy of an LLM’s output highly depends on the probability of both the input and the output text, output probability has a greater influence on model performance. We therefore focus this study on a single translation direction, instructing LLMs to output translations from a lowresource language into the language with which they are likely to have had more exposure during training, i.e, from Southern Quechua into Spanish, and leave the reverse direction for future work.
# 3 Quechuan Languages
Quechua is a family of languages Indigenous to the Andes in South America. This study focuses on varieties of Southern Quechua (S. Quechua, also known as urin quechua or quechua sureño) spoken in parts of Peru.1 While previous studies investigated language-LLM pairs for which the baseline
1Unless noted otherwise, we use Quechua in this study to refer Southern Quechua and related varieties.
LLM lacked any pretrained knowledge, we find that newer LLMs can translate some S. Quechua sentences in a zero-shot setting. We expect this to be typical of many low-resource languages which, while often endangered, still may have some presence on the web. Quechuan languages have by far the largest representation of all Indigenous Latin American languages in NLP research (Tonja et al., 2024) and are often included in ACL-affiliated workshops, datasets, and shared tasks (Cotterell et al., 2020; Ebrahimi et al., 2022, 2023). S. Quechua has a robust language toolkit (Rios, 2015), including the morphological parser we use in our pipeline. It has also been the subject of numerous studies on MT for both text and speech, developed in conjunction with monolingual and parallel corpora (Rios, 2015; Cardenas et al., 2018; Ortega et al., 2020; Zevallos et al., 2022). Nonetheless, such tools continue to face challenges, and Quechuan languages continue to lack the resources necessary to develop most of today’s state of the art models. Since Quechua is primarily spoken in South America, the majority of available digital resources, including all materials used in this study, use Spanish as the language of translation, explanation, and/or instruction. We therefore also use Spanish, rather than English or any other high-resource language, as the language of translation and prompting when testing our system.
# 3.1 Language-Specific Factors
While the proposed methods are general enough to be applied to any language pair, model outputs may reflect certain language-specific characteristics of the source and target languages, respectively. In this section, we provide a brief description of selected language-specific factors in S. Quechua as they relate to their translated Spanish counterparts. For a discussion of their potential effects on our results, please see Section 6.1.
# 3.1.1 Morphological Segmentation
S. Quechua is primarily agglutinating, i.e., much of the morphology may be described in terms of isomorphic form-meaning relationships, morphemes generally maintain a consistent form regardless of their phonological environment, and morpheme boundaries tend to be transparent. In contrast, morpheme segmentation in Spanish may be rendered opaque due to its fusional morphology and widespread use of conditioned allomorphy.
While LLMs are trained to process text via tokenbased rather than morpheme-based segmentation, it is possible that a lack of direct correspondence between the expression of morphosyntactic categories in S. Quechua and Spanish may affect a model’s ability to leverage the information we provide as prompt context in our experiments. Correspondences in form and meaning across parallel usage examples may be particularly obscured, limiting the use of corpora designed for traditional MT methods. It may be possible to mitigate such issues with more advanced retrieval or prompting techniques, for example by explicitly instructing an LLM to conduct morphological analysis as part of the translation process, but we leave this for future work.
# 3.1.2 Syncretism and Polysemy
Although the language is primarily agglutinating, a number of morphemes in S. Quechua are syncretic, such that a given form may be used to express more than one grammatical category. For example, the 1SG.POSS marker, -y, shares the same form as both the 2SG.IMP marker and the infinitival marker, as illustrated in the following examples:
(1)
(2)
(3)
mikhu-y-ta eat-INF-ACC muna-ni want-1SG ‘I want to eat’
Similarly, words in S. Quechua may be polysemous, with the potential to express more than one meaning depending on their use in context. For example, the S. Quechua word miski (misk’i) may be translated as either dulce ‘sweet’ or rico/a ‘delicious’, and both dulce and rico/a are themselves polysemous in Spanish. Dulce may be used as an adjective, i.e., ‘sweet’, or a noun, i.e.,‘candy’, and rico/a may describe either richness in flavor, i.e., ‘delicious’, or in monetary wealth, i.e., ‘rich’. The exact forms displaying syncretism or polysemy must be identified on a language-specific basis, but the ambiguity they present poses a clear problem for our proposed methods in general, with
# potential effects on both retrieval and generation. We discuss this issue further in Section 6.1.
potential effects on both retrieval and generation. We discuss this issue further in Section 6.1.
# 3.1.3 Variation
Both S. Quechua and Spanish are characterized by extensive regional and dialectal variation. In S. Quechua, this includes differences in orthographic and/or phonological conventions as well as the specific lexical items and expression of morphosyntactic content. For example, the S. Quechua word for ‘dog’ may be rendered orthographically as alqo, allqo, allku, allqu, or ashko, and the additive suffix may be expressed as either -pas or -pis, depending on the community. Variation in the attested usage of specific lexical items and morphemes across communities is also common in S. Quechua. For example, the evidential marker -mi /-m is frequently attested in the Peruvian variety of S. Quechua used in this study, but essentially absent in many Bolivian varieties. Variation across Spanish-speaking communities may also affect models’ abilities to produce translations that are both accurate and appropriate. The Andean Spanish reference translations used in this study do not appear to affect the results of our automatic evaluation. However, were the proposed methods to be applied in a realistic setting, it would be especially important to assess the degree of alignment between any prescriptive linguistic standards that have been implicitly acquired by the LLM and the usage conventions of the language community or communities of interest.
# 4 Methods
# 4.1 Data
We conduct experiments on a collection of 50 pairs of S. Quechua - Spanish sentences sourced from one of the author’s personal notes. These were selected to highlight a range of specific grammatical phenomena at multiple levels of difficulty— they include simple clauses and tenses (Example (4)), as well as more advanced constructions such as those involving past participles (Example (5)) and simultaneous events (Example (6).
(4) qam you allin-ta good-ACC tusu-nki dance-2SG tu bailas bien ‘you dance well’
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a044/a0446acc-1d29-4779-a0fb-4064243bc68a.png" style="width: 50%;"></div>
Figure 1: Example BASELINE prompt. English: [TASK] Translate the following sentence from Quechua to Spanish. Respond only with the translation: Quechua: kay wasiqa turiypam; Spanish:
(5)
) awa-sqa-y knit-PP-1SG wali-qa skirt-TOP sumaq-mi great-ASSERTIVE la falda que tejí es linda ‘the skirt that I knit is pretty’ ) qam-qa you-TOP taki-ta song-ACC uyari-spa listen-SUBR wasi-yki-ta house-2.POSS-ACC picha-chka-nki clean-PROG-2SG tú estás limpiando tu casa escuchand música ‘you’re cleaning your house listening to m sic’
(6)
The first author, a foreign-language student of S. Quechua, received permission from her instructor to use notes from their lessons for the study. All sentence pairs were inspected by the instructor, a native bilingual speaker of both S. Quechua and Peruvian Spanish, to eliminate any errors and confirm the accuracy of all reference translations.
# 4.2 Prompt Construction
As a baseline, each sentence is inserted into a prompt template that instructs the model in Spanish to translate the S. Quechua sentence into Spanish and respond only with the translation (Figure 1). We automate a process for building on this template and compare the effects of adding information from three different sources to the prompt context.
# 4.2.1 Morpheme Translations (MORPH)
We use a morphological parser (Rios, 2015) to segment each word of the source segment into morphemes, each with gloss symbols and a Spanish translation.2 Some morphemes have multiple candidate meanings, all of which are retrieved. As an example, the word rantikuq is segmented as ranti-ku-q and glossed as “comprar.DB.VRootDB.VDeriv.+RflxInt+Ag.NS.”
2We set aside valid concerns regarding the theoretical status of the morpheme for this study and define a morph(eme) loosely as a recognizable form-meaning pair that recurs in a language.
While numerous orthographic standards have been developed and promoted across Quechuanspeaking communities in South America, considerable variation in orthographic conventions may be found even within a particular community or variety (Rios and Castro Mamani, 2014). We discuss the implications of this for our results in Section 4.2.5. We supplement the output from the parser using a Quechua-Spanish bilingual dictionary (Qheswa Simi Hamut’ana Kurak Suntur, 2005). We retrieve any dictionary entry whose headword exactly matches a morpheme in our segmentation. By default, we include all senses and any usage examples or contextual information in the dictionary entry as part of the prompt. We then concatenate the output of the parser with the retrieved dictionary entries and include this MORPH information as prompt context preceding the source sentence and baseline translation prompt.
# 4.2.2 Grammar Descriptions (GRAMMAR)
We also experiment with the inclusion of grammar lessons found in student-facing pedagogical materials, retrieving grammatical explanations relevant to each source sentence from a PDF document developed for students and teachers of S. Quechua (Pinto Tapia et al., 2005). The document is organized into short sections (1-3 sentences, plus paradigm tables or usage examples) that describe the particular grammatical concept associated with an affix in Quechua. For each source sentence, we retrieve sections associated with any affix listed in the document that is an exact match of a morpheme and include this in prompts using contextual information from the grammar. This improves on the methods described in Tanzer et al. (2024), who use LCS-based retrieval over an entire textbook, and Elsner and Needle 2023 and Zhang et al. 2024, whose grammatical description remains consistent across prompts regardless of the source text being translated.
# 4.2.3 Parallel Usage Examples (CORPUS)
Finally, we experiment with sentence-level examples from a S. Quechua-Spanish parallel corpus designed for traditional NLP tasks. We combine data made available via the AmericasNLP 2021 Shared Task on Open Machine Translation and the 2023 IWSLT shared task on low-resource SLT (Tiedemann, 2012; Agi´c and Vuli´c, 2019; Ortega et al., 2020; Mager et al., 2021; Agarwal et al.,
2023). For each source sentence, we retrieve the three best matches from the corpus using a LCS search against the full source sentence.
# 4.2.4 Combined Prompt Types
Combinations of information from all three sources yields 8 total conditions, including the baseline. An example prompt from each information source is given in Appendix E.
# 4.2.5 Manually Revised Prompts
To compute a soft upper bound on the improvements possible with better retrieval, we conduct an additional set of experiments using manually revised prompts. We first examine the content retrieved from the morphological parser, dictionary, and grammar document and remove all instances of ambiguity and irrelevant or misleading information from the prompt context.3 For example, many S. Quechua speakers use the term runasimi (lit: ‘people mouth’, ‘the people’s language’), as an endonym for the language. The parser, however, returns only the literal decomposition (runa ‘ser humanos’/‘people’ and simi ‘boca’/‘mouth’), and the dictionary does not list runasimi as a headword but rather as one of eight different senses of simi. We thus remove all such irrelevant examples and translations from the prompt and retain only the content indicating a translation of runasimi in the linguistic sense. We also manually retrieve content from the dictionary and grammar documents that were overlooked by the automated retriever. For example, the verb yanuy ‘to cook’ does not appear as a headword in the dictionary, but rather as a regional variant of wayk’uy ‘to cook’. We also eliminate content from the grammar that was retrieved because of syncretism, or mistakes that cascaded from the morphological parser to result in irrelevant retrievals. We manually parse each source sentence to only retrieve and include relevant information in the prompt context. All content in the revised prompts is sourced from the same material available to the automated retriever systems, and we do not add any additional information or use supplemental materials of any sort to create the revised prompts.
3We do not experiment with retrieval methods for corpus examples, which were retrieved using LCS match in both conditions. Improving on LCS-based retrieval remains an open question in low-resource LLM-MT, and we leave this for future work.
# 4.3 Models
We experiment with three proprietary models, GPT3.5 Turbo (gpt-3.5-turbo-0125, Brown et al., 2020), GPT-4o (gpt-4o, Achiam et al., 2023), and Gemini 1.5 Pro (gemini-1.5-pro, Reid et al., 2024), and one open-source model, Llama 3 (llama-3-8b-instruct, AI@Meta, 2024). We use the pretrained models with their default settings, and do not adjust hyperparameters or conduct any finetuning as part of our experiments.
# 4.4 Evaluation
We conduct both automatic and human evaluations to identify trends in model errors and outputs in the various experimental conditions. We calculate BLEURT and BLEU scores as automatic metrics, and report mean BLEURT scores across items as the primary quantitative measure of translation quality for each of the conditions and models. We also use an adapted MQM schema to conduct qualitative human evaluation of the outputs of GPT-3.5 and GPT-4o for all prompts using automatic retrieval. Each item selected for human evaluation is annotated by at least one of the authors by comparing the model’s output to the source text and reference translation. We refer to the complete MQM typology to design our own four-dimensional framework of commonly attested errors in LLM-MT, each with a defined set of specific subtypes. Precise definitions and examples for all error categories and subtypes may be found in Appendix D. Many of the categories in our schema are defined as in the core MQM framework. However, to capture some of the key behaviors reported in previous studies on LLM-MT and to evaluate the effects of prompt type on model outputs, we make the following adjustments. First, we utilize the Addition and Omission errors defined as Accuracy subtypes in the original MQM typology, but distinguish these from three additional subtypes: Substitution - Incorrect Subject, Substitution - Incorrect Tense/Aspect/Modality (TAM), and Substitution Other. This is intended to capture LLM translations that differ from the source in terms of discrete lexical material or case, person, number, and/or TAM markings while otherwise maintaining the lexical and structural content needed to appropriately translate the source text. Although they are not the only grammatical phenomena that may be similarly misrendered, we select subject and TAM
markers for analysis as they are straightforward to identify and give a good indication of how well the LLMs cope with more abstract information about the meanings of functional morphemes. Rather than including Mistranslation and MT Hallucination as Accuracy Error subtypes as in the original MQM typology, we define a separate NonTranslation category with three possible subtypes: Complete Mistranslation, Mistranslation with Lexical Correspondences, and Refusal. The third dimension of our typology, Model Error, was ultimately not used to classify any output in this study, but characterizes more generic model “misbehavior” such as failing to follow instructions, producing garbled text, or inappropriately generating content in the source language. Finally, Target Errors identify outputs that are ungrammatical, stylistically inappropriate, or semantically incoherent in the target language, regardless of their accuracy. Detailed annotation guidelines were drafted and agreed upon to encourage consistency across annotators and experimental items. Annotators are instructed to identify and tag up to three specific errors for each translation output, with the exception of Target Errors, which do not count towards the three-error maximum. Each model output is also tagged for quality along a four-point scale as defined in Table 5. Before proceeding with annotation over the larger dataset, both annotators also completed a test evaluation of the same 12 experimental items (96 sentences total) to assess inter-annotator agreement. Statistical measures (κ = 0.72 for quality judgments, α = 0.55 for error categories) indicated some discrepancies in annotator judgments, especially for categories, since determining the three most important errors is especially subjective. These were identified and discussed, and agreement was ultimately deemed sufficient to proceed.
# 5 Results
# 5.1 Quality Metrics
We present BLEURT scores for prompts generated using automated retrieval in Table 1 and summarize human quality judgments for GPT-3.5 and GPT4o with automated retrieval in Table 2. The complete distribution of BLEURT, BLEU, and humanannotated quality ratings for all of our experiments is provided in Appendix F. We find clear effects of LLM, prompt type, and retrieval method, as well as interactions among all three factors.
<div style="text-align: center;">GPT3.5 GPT4o</div>
<div style="text-align: center;">GPT3.5 GPT4o Gem. L</div>
GPT3.5
GPT4o
Gem.
Lla3
BASE
0.19
0.66
0.56
0.15
CORPUS
0.27
0.59
0.49
0.19
GRAM
0.23
0.56
0.55
0.17
MORPH
0.44
0.54
0.61
0.39
C+G
0.26
0.59
0.54
0.21
C+M
0.44
0.59
0.59
0.36
G+M
0.41
0.53
0.61
0.39
C+G+M
0.43
0.57
0.61
0.15
Table 1: Mean BLEURT scores by LLM and prompt type. Shaded rows include morpheme contexts.
<div style="text-align: center;">GPT-3.5 GPT-4o</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b1a0/b1a03a8c-bd15-4871-b646-b13948b7d127.png" style="width: 50%;"></div>
LLM
GPT-3.5
GPT-4o
BASELINE
21
108
CORPUS
43
101
GRAMMAR
33
99
MORPH
79
102
C+G
41
101
C+M
75
110
G+M
68
100
C+G+M
77
109
Table 2: Human-annotated quality ratings summarized as 3 × high + 2 × med + low. Shaded rows include morpheme contexts.
<div style="text-align: center;">GPT3.5 GPT4 Gem. Lla3</div>
G-AUTO
0.23
0.56
0.55
0.17
G-MAN
0.24
0.58
0.54
0.15
M-AUTO
0.44
0.54
0.61
0.39
M-MAN
0.56
0.63
0.66
0.49
CGM-AUTO
0.43
0.57
0.61
0.15
CGM-MAN
0.54
0.63
0.63
0.26
Table 3: Comparison of mean BLEURT scores for automatic versus manual retrieval of material in GRAMMAR, MORPH, and CORPUS-GRAMMAR-MORPH prompts.
Comparing across models, we find that Gemini and GPT-4o outperform Llama 3 and GPT-3.5 for every prompt type. This gap is highest for the least informative prompts, indicating that the Llama 3 and GPT-3.5 base models have relatively poor coverage of S. Quechua, while GPT-4o and Gemini have much better coverage. The effect is evident in both automatic and human quality evaluations. Effects of prompt type are mediated by the quality of the pretrained model. Llama 3 and GPT-3.5 show a clear improvement in quality when MORPH information is included in the prompt. Gemini also improves when this information is added, but to a lesser extent. GPT-4o, on the other hand, performs best in response to the BASELINE (zero-shot) prompts, which attain the highest BLEURT scores across all models, prompt types, and retrieval methods evaluated in this study. In other words, providing additional information in the prompt’s context actually degrades GPT-4o’s ability to translate from S. Quechua to Spanish in all experimental conditions.
# 5.2 Effects of Automated Retrieval
To highlight the effects of automated retrieval on model output, we present BLEURT scores for a selection of prompt types and all four models in Table 3 (full scores may be found in Appendix F). The effect of manual retrieval for MORPH information is positive for all models, although this gap is smallest for Gemini (probably because its performance for these prompts is already highest). The effect for GRAMMAR prompts is either minor or negative.
# 5.3 Human Analysis of Translation Errors
The most common error type identified by the annotators is Substitution - Other, which includes a diverse assortment of lexical and phrasal incongruencies of varying degrees of severity. These are largely item-specific and therefore hard to characterize as a group. Using the error categories described in Section 4.4, we instead identify three more clearly interpretable phenomena and provide a detailed discussion of each in the following sections. We present counts for selected prompt types in Table 4, with examples in Appendix A and counts for all errors in Appendix G.
# 5.4 Mistranslations
Outright mistranslations are most common for GPT3.5, making up 30 of the 50 responses in the BASE-
Mistranslation: complete +
lexical correspondence
GPT-3.5
45
11
12
GPT-4o
4
6
4
Target Fluency: grammar +
coherence + style
GPT-3.5
0
14
10
GPT-4o
3
13
9
Grammatical Divergence:
subject + TAM
GPT-3.5
0
24
31
GPT-4o
17
13
11
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1aeb/1aeb9f77-cd1d-4b4d-8824-425b5a018b80.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 4: Counts of human-annotated error types (per 50 sentences) by LLM and prompt type.</div>
LINE condition. We also consider outputs that retain only minimal traces of the source content, which we label as Mistranslations with Lexical Correspondence. Approximately 1/3 of the 637 total errors tagged across all prompt types for GPT3.5 are mistranslations of either type, roughly split between complete mistranslations and those with lexical correspondence (15.07% and 18.37%, respectively, of all errors tagged for GPT-3.5). As reported in previous work, adding morphemeand word-level translations to the prompt greatly reduces the rate of this kind of response. GPT4o also produces drastically fewer mistranslations compared to its predecessor. However, it is notable that both models produce at least one mistranslation for each prompt type. In general, complete mistranslations are in fluent Spanish and contain no overt indications that something has been misrepresented. We return to the ethical implications of these errors in the Discussion. We also note that many of the items tagged as Mistranslation with Lexical Correspondence show correspondence only for words that were already in Spanish in the source text. For example, some sentences contain Spanish loan names for the days of the week. While some of these errors are produced in deceptively fluent Spanish, we find many to be accompanied by semantic incoherence or ungrammaticality in the output. We discuss such target language fluency errors in the following section.
# 5.5 Target Fluency
Target Fluency errors occur when the output is not grammatical, coherent, or stylistically appropriate – for instance, if an output contains a nonsensical repetition or a verb with missing arguments. Outputs of this type bear a strong similarity to human “translationese” in that structural features of the source language may surface in the translation at the expense of naturalness (Koppel and Ordan, 2011; Freitag et al., 2019). Both GPT-3.5 and GPT-4o tend
<div style="text-align: center;">BASE MORPH C+G+M</div>
to produce more such outputs when the prompt is more informative – 10 to 20% of the time (510 instances per 50) in prompts with morpheme translations.
# 5.6 Grammatical Divergence
We group misrendered verbal subjects and tense/aspect/morphology (TAM) markers together as Grammatical Divergence errors. Such errors are distinct from the Target Fluency errors described in the previous section— the Spanish output is grammatical, but fails to accurately reflect the content from the source. TAM divergences are much more prevalent than divergences in subject; for instance, only one of GPT-4o’s 13 Grammatical Divergence errors in the MORPH condition misrender the subject marker. Grammatical Divergence errors are annotated only for sentences that are not mistranslated outright, so GPT-3.5 produces none of these in the BASELINE condition. For more informative prompts, it is clear that GPT-4o is better than GPT3.5 at translating both functional and lexical meanings. However, a relatively large number of sentences (over 20%) still contain such an error even with the highest performing model and prompt type. The relatively small drop in error between different prompt types for GPT-4o suggests that neither the corpus-based usage examples nor example paradigms and descriptions from the grammar document can fully prevent this type of error.
# 6 Discussion
We observe large differences between LLMs, both in terms of the overall quality of their generated translations as well as the effects of prompt type on their outputs. GPT-4o and Gemini, which have the highest baseline scores, benefit least from additional information— BLEURT scores actually decrease when CORPUS and GRAMMAR information is included. This occurs even with manually
curated prompts, suggesting it is not an effect of including irrelevant material. Nonetheless, the baseline results do not represent a ceiling on quality, since both models still produce errors in the BASELINE condition (GPT-4o produces 10 LOW-quality translations in our set of 50). These results suggest that even relevant grammar explanations, when written in prose with examples, do little to help the newest generation of LLMs to translate a lowresource language such as Southern Quechua. Although GPT-4o and Gemini results are similar in many ways, we do find evidence for differences in their in-context learning abilities. Baseline prompts and the GPT-4o model produce the highest BLEURT scores across the dataset, but these outputs still show a number of errors characteristic of LLMs, particularly lexical substitution errors that are not necessarily corrected with the inclusion of more context. In contrast, Gemini, which has near-comparable performance across prompt types, shows an increase in scores when prompts include MORPH information, regardless of retrieval type, suggesting a greater ability to identify and utilize relevant word- and morph-level translations in the prompt’s context. Previous work suggests that newer builds of GPT-4 are less capable of following instructions (Chen et al., 2023); such differences may be masked by the effects of pretraining when automatically evaluating translations. This suggests that researchers should continue to carefully select and compare among different LLMs when experimenting with retrieval-based translation.
# 6.1 Language-Specific Effects
We identify a number of translation errors of varying types that appear to be due to language-specific factors such as those discussed in Section 3.1. For example, we find an effect of polysemous lexical items for all prompt types in the outputs of both models on which we conduct human evaluation. In the most straightforward cases, the model incorrectly generates an alternate sense of the word that is inappropriate given the content of the source sentence. We also find a number of instances in which the presence of such ambiguity has a cascading effect on lexical selection in the rest of the model’s output. For example, when co-occuring with miski in the source text, the word lawa ‘soup’ is translated at times as mazamorra, a sweet porridge or pudding, crema ‘cream’, miel ‘honey’, golosina ‘candy’, or dulces, the nominal form of dulce meaning ‘candy’ or ‘sweets.’
It may be possible to moderate such effects with additional refinement of the database structure and retrieval methods, which we leave for future work.
# 6.2 Ethical Concerns
Both our work and much of the previous work in this paradigm is motivated by the desire to close the “NLP Gap” among researchers, community members, and software developers interested in lowresource language technologies. Machine translation is listed as a welcome topic of research by some (though not all) members of American Indigenous communities (Mager et al., 2023), and is potentially an important tool for language learners (Jolley and Maimone, 2022). Even an imperfect translation system might be a useful tool for users with a clear understanding of its limitations. However, the systems evaluated in this work have two problematic tendencies that limit their potential for deployment in real community settings. First, unfaithful translations often tend to be highly fluent (Section 5.4). While fluency ratings for older MT systems correlate well with accuracy scores, and have even been used as a proxy for overall translation quality (Gamon et al., 2005; Estrella et al., 2007), this correlation is reversed for our systems. LLMs are well-known for making false statements that seem plausible and authoritative (Bickmore et al., 2018; Dinan et al., 2021); this could be particularly problematic when they project illusions of expertise at the expense of an already marginalized group. Second, some mistranslations identified in our study appear to draw on stereotypes of Indigenous groups (Appendix B). These are most apparent for the BASELINE system and GPT-3.5, but also (less frequently) occur with more informative prompts and better LLMs. Stereotypical sentences can involve flowery language with an emphasis on tradition or connectedness to nature (Erhart and Hall, 2019), as well as the unprompted addition of Indigenous Andean cultural customs and products (e.g., traditional medicine and chicha) to translations that are otherwise faithful to the source text. The overall effect is to exoticize Southern Quechua speakers and writers in ways that the original sentences do not. Similar stereotypes have also been noted in LLM-generated responses to open-ended prompts (Cheng et al., 2023; Delgado Solorzano and Toxtli, 2023; Shieh et al., 2024). While we prompt models to output only the translation for evaluation purposes, models may have
some capacity to explain or qualify their translations and give reminders for responsible use of the technology. Should a retrieval-based translation system ever be deployed in a real-world setting for language learning, its developers should maximize transparency by presenting the content of any retrieved information and its source to the user along with the translation, reminding users directly of potential inaccuracies, and offering vetted resources for additional fact-checking when available.
# 7 Conclusion
Our results suggest a number of key limitations and concerns regarding the use of LLMs in a lowresource MT context, and have greater implications for our understanding of the seemingly “humanlike” conceptual, analytical, and in-context learning abilities of LLMs. For the majority of the world’s languages and their speakers, powering and supplying LLMs with enough pretraining data to overcome their limitations is not feasible. We therefore offer the following suggestions to those looking to develop lowresource LLM-MT: (1) improve data structures and methods for interacting with a language-specific database for retrieval-aided generation, (2) continue analysis of the mechanisms driving in-context learning in LLMs, for example by comparing ICL to the effects of finetuning (Dai et al., 2023), and (3) experiment with prompt structures and techniques, for example by altering the order of information (Liu et al., 2024) or by iteratively prompting the model to guide its reasoning towards a suitable translation (Wang et al., 2022). Finally, we wish to emphasize the continued risks of prematurely deploying this or similar methods in any low-resource language community, particularly given the vulnerability and disproportionate lack of resources many such communities face in domains where these technologies would likely be used. As AI research continues to rapidly develop, we urge those conducting it to increase community engagement, amplify the voices of those traditionally at a disadvantage, and collaboratively develop research infrastructures that may lessen the NLP Gap (Brinklow, 2021). While there’s still much to be done before low-resource LLM-MT may be safely implemented, we believe such a tool has the potential to empower speakers of any variety, including nonstandard varieties of highresource languages such as English, to develop
technologies that reflect their preferences and serve their unique needs.
# 8 Limitations
Limitations on the scope and replicability of this work may be attributed to one or more characteristics of the data and models used in this study, in addition to limitations inherent to the respective identities of its authors. First, the automatic metrics (i.e., BLEURT and BLEU scores) that we report are limited in their statistical validity. We have conducted some constrained tests to explore potential variance in scores, but expenses associated with text generation using proprietary models such as those developed by OpenAI and Google on a larger dataset may be prohibitive. This is compounded by the widely-acknowledged “black box” nature of the models powering both LLMs and BLEURT, as well as an increasing opacity with respect to the exact content and methods used to pretrain modern state of the art LLMs. For this reason, we focus our discussion on those results that show clear trends in both the quantitative and human evaluations we conduct. There are also some constraints on our study and its methodology that are largely tied to linguistic factors, such as variation in orthography (and the need for digitized text-based resources as a prerequisite) as well as the lexical and grammatical variation that may be found in all languages, particularly the low-resource varieties we wish to support. We discuss some of these factors in Sections 3.1 and 6.1. Our results suggest it may be possible to guide the outputs of LLMs towards the specific usage conventions of a given community, but this is itself limited by the content of the materials used to develop the database from which prompt contexts are retrieved. Neither of the authors is a native speaker of any Quechua or Spanish varieties, and only one is a student of these languages and has relationships to Quechua speakers and communities. While we have strived to be consistent in the Quechua and Spanish varieties used in our study (both the dictionary and grammar materials were provided by the same instructor who shared and proofread the 50 sentence pairs we use, and we select a morphological parser and corpora intended for use with Southern Quechua), variation is widespread among and within Quechua-speaking communities, and we do not have access to a dictionary, grammar,
morphological parser, and corpus developed by a unified and consistent set of authors. Future work should continue to explore ways to faithfully represent the diversity of linguistic conventions employed by communities interested in developing such technologies. We acknowledge, as well, limitations that arise from the size of our dataset and database and the methods used to curate them. The 50 sentence pairs we use were selected to highlight a range of specific grammatical phenomena, not all of which were well represented in our database, and differ in their structural complexity. We are grateful for the guidance provided by the Quechua instructor whose lessons were a source for such examples and proofread the sentences before their inclusion in our experiments, but are limited by our status as non-native speakers. Human evaluation of model outputs was partially conducted using machine-translated English texts as references, but all annotations were inspected by the Spanish- and Quechua-speaking author who removed a small number of evaluations that reflected linguistic discrepancies between Quechua, Spanish, and English or inaccuracies in the machine-translated English.
# 9 Ethics Statement
We consulted the first author’s Quechua instructor, Prof. Carmen Cazorla Zen, who gave us permission to use the sentences from the notes in this project and verified their accuracy. We cite the Quechua dictionary and grammar materials used to provide prompt information, and believe that our use of these materials is consonant with their original purpose. However, we do not distribute machine-readable versions of them as a contribution of this project, since this would violate the rights of the publisher. These materials were developed for use as pedagogical resources by institutions affiliated with the governments of Cuzco, Peru and Apurímac, Peru, respectively. Their authors were not contacted or consulted as part of the project. We wish to acknowledge the delicate issue of academic extractiveness and its harmful impact on Indigenous and minority language communities and speakers. We are also aware of some of the controversial ideologies and policies associated with Qheswa Simi Hamut’ana Kuraq Suntur, the government-afilliated institution who published the dictionary we use in this study, and the potentially
negative effects of government-sponsored linguistic standardization more broadly (see, e.g., Coronel Molina (2008) for an analysis of the effects of the institution’s ideologies on revitalization efforts in Peru). We do not endorse such policies, and have sought to avoid representing the diversity of Southern Quechua-speaking communities as a monolith. Instead, we hope our continued efforts to improve methods for low-resource translation will empower speakers of Southern Quechua and other Indigenous and minority languages to develop language technologies capable of representing their own community’s unique language variety to serve the unique needs of its speakers. There are numerous ethical issues related to the training and use of LLMs, such as labor issues and energy costs. While these issues are inextricable from the methods used in this project, we believe the potential impact of making low-resource translation viable and accessible to minority language communities who want them (our primary goal in this line of research) outweighs the problems inherent in using LLMs at all. We discuss the potential risks of deploying systems like the ones described here further in Section 6.2 of the main text.
# Acknowledgments
We thank Prof. Carmen Cazorla Zen, Professor of Quechua, for her help curating the data used in this study and for deepening our understanding of Southern Quechua and its speakers. We also thank Prof. Elvia Andía Grágeda, Professor of Quechua, for her instruction and advice, and the OSU Linguistics department for their feedback on a preliminary presentation of the work.
# References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774. See also: https://openai.com/index/hello-gpt-4o/. Milind Agarwal, Sweta Agrawal, Antonios Anastasopoulos, Luisa Bentivogli, Ondˇrej Bojar, Claudia Borg, Marine Carpuat, Roldano Cattoni, Mauro Cettolo, Mingda Chen, William Chen, et al. 2023. FINDINGS OF THE IWSLT 2023 EVALUATION CAMPAIGN. In Proceedings of the 20th International Conference on Spoken Language Translation (IWSLT 2023), pages 1–61, Toronto, Canada (in-person and online). Association for Computational Linguistics.
Željko Agi´c and Ivan Vuli´c. 2019. JW300: A widecoverage parallel corpus for low-resource languages. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3204–3210, Florence, Italy. Association for Computational Linguistics. Sweta Agrawal, Chunting Zhou, Mike Lewis, Luke Zettlemoyer, and Marjan Ghazvininejad. 2022. Incontext examples selection for machine translation. Preprint, arXiv:2212.02437.
In Proceedings of the 57th Annual Meeting of the
Association for Computational Linguistics, pages
3204–3210, Florence, Italy. Association for Com-
putational Linguistics.
Sweta Agrawal, Chunting Zhou, Mike Lewis, Luke
Zettlemoyer, and Marjan Ghazvininejad. 2022. In-
context examples selection for machine translation.
Preprint, arXiv:2212.02437.
AI@Meta. 2024. Llama 3 model card.
Timothy W Bickmore, Ha Trinh, Stefan Olafsson,
Teresa K O’Leary, Reza Asadi, Nathaniel M Rick-
les, and Ricardo Cruz. 2018. Patient and consumer
safety risks when using conversational assistants for
medical information: An observational study of Siri,
Alexa, and Google Assistant. J Med Internet Res,
20(9):e11510.
Nathan Thanyehténhas Brinklow. 2021. Indigenous lan-
guage technologies: Anti-colonial oases in a coloniz-
ing (digital) world. WINHEC: International Journal
of Indigenous Education Scholarship, (1):239–266.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell,
et al. 2020.
Language models are
few-shot learners.
Preprint, arXiv:2005.14165.
See also: https://openai.com/index/new-embedding-
models-and-api-updates/.
Aljoscha Burchardt. 2013.
Multidimensional qual-
ity metrics: a flexible system for assessing trans-
lation quality. In Proceedings of Translating and the
Computer 35, London, UK. Aslib.
Ronald Cardenas, Rodolfo Zevallos, Reynaldo Ba-
querizo, and Luis Camacho. 2018. Siminchik: A
speech corpus for preservation of Southern Quechua.
ISI-NLP, 2:21.
Lingjiao Chen, Matei Zaharia, and James Zou. 2023.
Analyzing chatgpt’s behavior shifts over time. In
R0-FoMo: Robustness of Few-shot and Zero-shot
Learning in Large Foundation Models.
Myra Cheng, Esin Durmus, and Dan Jurafsky. 2023.
Marked personas: Using natural language prompts
to measure stereotypes in language models.
In
Proceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 1504–1532, Toronto, Canada.
Association for Computational Linguistics.
Serafín M. Coronel Molina. 2008. Language Ideologies
of the High Academy of the Quechua Language in
Cuzco, Peru. Latin American and Caribbean Ethnic
Studies, 3(3):319–340.
Ryan Cotterell, Christo Kirov, John Sylak-Glassman,
Géraldine Walther, Ekaterina Vylomova, Arya D. Mc-
Carthy, Katharina Kann, Sabrina J. Mielke, Garrett
AI@Meta. 2024. Llama 3 model card.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/aa86/aa8684d0-19c4-46a7-bef7-3115920c6104.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b95b/b95b3d41-8fbf-407e-8f72-4c190c83e257.png" style="width: 50%;"></div>
Abteen Ebrahimi, Manuel Mager, Shruti Rijhwani,
Enora Rice, Arturo Oncevay, Claudia Baltazar, María
Cortés, Cynthia Montaño, John E. Ortega, Rolando
Coto-solano, Hilaria Cruz, Alexis Palmer, and Katha-
rina Kann. 2023.
Findings of the AmericasNLP
2023 shared task on machine translation into Indige-
nous languages. In Proceedings of the Workshop
on Natural Language Processing for Indigenous
Languages of the Americas (AmericasNLP), pages
206–219, Toronto, Canada. Association for Compu-
tational Linguistics.
Micha Elsner and Jordan Needle. 2023.
Translat-
ing a low-resource language using GPT-3 and a
human-readable dictionary. In Proceedings of the
20th SIGMORPHON workshop on Computational
Research in Phonetics, Phonology, and Morphology,
pages 1–13, Toronto, Canada. Association for Com-
putational Linguistics.
Ryan S Erhart and Deborah L Hall. 2019. A descriptive
and comparative analysis of the content of stereo-
types about Native Americans.
Race and Social
Problems, 11:225–242.
Paula Estrella, Andrei Popescu-Belis, and Maghi King.
2007. A new method for the study of correlations be-
tween MT evaluation metrics. In Proceedings of the
11th Conference on Theoretical and Methodological
Issues in Machine Translation of Natural Languages:
Papers, Skövde, Sweden.
Markus Freitag, Isaac Caswell, and Scott Roy. 2019.
APE at scale and its implications on MT evaluation
biases. In Proceedings of the Fourth Conference on
Machine Translation (Volume 1: Research Papers),
pages 34–44, Florence, Italy. Association for Com-
putational Linguistics.
Michael Gamon, Anthony Aue, and Martine Smets.
2005.
Sentence-level MT evaluation without ref-
erence translations:
Beyond language modeling.
In Proceedings of the 10th EAMT Conference:
Practical applications of machine translation.
Luke Gessler. 2022. Closing the NLP gap: Documen-
tary linguistics and NLP need a shared software in-
frastructure. In Proceedings of the Fifth Workshop
on the Use of Computational Methods in the Study
of Endangered Languages, pages 119–126, Dublin,
Ireland. Association for Computational Linguistics.
Marjan Ghazvininejad, Hila Gonen, and Luke Zettle-
moyer. 2023. Dictionary-based phrase-level prompt-
ing of large language models for machine translation.
Preprint, arXiv:2302.07856.
Amr Hendy, Mohamed Abdelrehim, Amr Sharaf,
Vikas Raunak, Mohamed Gabr, Hitokazu Matsushita,
Young Jin Kim, Mohamed Afify, and Hany Has-
san Awadalla. 2023. How good are GPT models
at machine translation? A comprehensive evaluation.
Preprint, arXiv:2302.09210.
Jason R Jolley and Luciane Maimone. 2022. Thirty
years of machine translation in language teaching
and learning: A review of the literature. L2 Journal,
14(1):26–44.
Tom Kocmi, Eleftherios Avramidis, Rachel Bawden,
Ondˇrej Bojar, Anton Dvorkovich, Christian Feder-
mann, Mark Fishel, Markus Freitag, Thamme Gowda,
Roman Grundkiewicz, Barry Haddow, et al. 2023.
Findings of the 2023 Conference on Machine Trans-
lation (WMT23): LLMs are here but not quite there
yet.
In Proceedings of the Eighth Conference on
Machine Translation, pages 1–42, Singapore. Asso-
ciation for Computational Linguistics.
Moshe Koppel and Noam Ordan. 2011. Translationese
and its dialects. In Proceedings of the 49th Annual
Meeting of the Association for Computational
Linguistics: Human Language Technologies, pages
1318–1326, Portland, Oregon, USA. Association for
Computational Linguistics.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the Middle: How Language
Models Use Long Contexts.
Transactions of the
Association for Computational Linguistics, 12:157–
173.
Zoey Liu, Crystal Richardson, Richard Hatcher, and
Emily Prud’hommeaux. 2022. Not always about
you: Prioritizing community needs when developing
endangered language technology. In Proceedings
of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1:
Long
Papers), pages 3933–3944, Dublin, Ireland. Asso-
ciation for Computational Linguistics.
Hongyuan Lu, Haoyang Huang, Dongdong Zhang, Hao-
ran Yang, Wai Lam, and Furu Wei. 2023. Chain-
of-dictionary prompting elicits translation in large
language models. Preprint, arXiv:2305.06575.
Manuel Mager, Elisabeth Mager, Katharina Kann, and
Ngoc Thang Vu. 2023.
Ethical considerations
for machine translation of Indigenous languages:
Giving a voice to the speakers.
In Proceedings
of the 61st Annual Meeting of the Association
for Computational Linguistics (Volume 1:
Long
Papers), pages 4871–4897, Toronto, Canada. Associ-
ation for Computational Linguistics.
Manuel Mager, Arturo Oncevay, Abteen Ebrahimi, John
Ortega, Annette Rios, Angela Fan, Ximena Gutierrez-
Vasques, Luis Chiruzzo, Gustavo Giménez-Lugo, Ri-
cardo Ramos, et al. 2021. Findings of the Amer-
icasNLP 2021 shared task on open machine trans-
lation for Indigenous languages of the Americas.
In Proceedings of the First Workshop on Natural
Language Processing for Indigenous Languages of
the Americas, pages 202–217, Online. Association
for Computational Linguistics.
R Thomas McCoy, Shunyu Yao, Dan Friedman,
Matthew Hardy, and Thomas L Griffiths. 2023. Em-
bers of autoregression: Understanding large language
models through the problem they are trained to solve.
arXiv preprint arXiv:2309.13638.
Gabriel Nicholas and Aliya Bhatia. 2023. Lost in trans-
lation: Large language models in non-english content
analysis. Preprint, arXiv:2306.07377.
John E Ortega, Richard Castro Mamani, and Kyunghyun
Cho. 2020.
Neural machine translation with a
polysynthetic low resource language.
Machine
Translation, 34(4):325–346.
Miguel Ángel Pinto Tapia, Luis Quispe Zúñiga, et al.
2005. Didáctica quechua i. Documento de trabajo,
Dirección Regional de Educación Apurímac | Direc-
ción Gestión Pedagógica.
Qheswa
Simi
Hamut’ana
Kurak
Suntur.
2005.
Diccionario Quechua - Español - Quechua Qheswa -
Español - Qheswa Simi Taqe, 2 edition. Multiservi-
cios e Imprenta Edmundo Pantigozo EIRL, Cusco,
Peru.
Machel Reid, Nikolay Savinov, Denis Teplyashin,
Dmitry Lepikhin, Timothy Lillicrap, Jean-Baptiste
Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Fi-
rat, Julian Schrittwieser, et al. 2024. Gemini 1.5: Un-
locking multimodal understanding across millions of
tokens of context. arXiv preprint arXiv:2403.05530.
Katherine
J.
Riestenberg,
Ally
Freemond,
Brook Danielle Lillehaugen,
and Jonathan N.
Washington. 2024. Prioritizing Community Partners’
Goals in Projects to Support Indigenous Language
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8d1f/8d1f6493-e2d7-4cc6-bedb-b12fe42f8f64.png" style="width: 50%;"></div>
<div style="text-align: center;">Gabriel Nicholas and Aliya Bhatia. 2023. Lost in translation: Large language models in non-english content analysis. Preprint, arXiv:2306.07377.</div>
University Press.
Annette Rios. 2015.
A basic language technology
toolkit for Quechua.
Ph.D. thesis, University of
Zurich.
Annette Rios and Richard Castro Mamani. 2014. Mor-
phological disambiguation and text normalization for
southern Quechua varieties.
Stephen Robertson, Hugo Zaragoza, et al. 2009. The
probabilistic relevance framework: Bm25 and be-
yond.
Foundations and Trends in Information
Retrieval, 3(4):333–389.
Nathaniel Robinson, Perez Ogayo, David R. Mortensen,
and Graham Neubig. 2023. ChatGPT MT: Competi-
tive for high- (but not low-) resource languages. In
Proceedings of the Eighth Conference on Machine
Translation, pages 392–418, Singapore. Association
for Computational Linguistics.
Ohad Rubin, Jonathan Herzig, and Jonathan Berant.
2022. Learning to retrieve prompts for in-context
learning.
In Proceedings of the 2022 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies, pages 2655–2671, Seattle, United
States. Association for Computational Linguistics.
Sylvia L.R. Schreiner, Lane Schwartz, Benjamin
Hunt, and Emily Chen. 2020.
Multidirectional
leveraging for computational morphology and lan-
guage documentation and revitalization. Language
documentation and conservation, 14.
Thibault Sellam, Dipanjan Das, and Ankur P. Parikh.
2020. BLEURT: Learning robust metrics for text
generation. Preprint, arXiv:2004.04696.
Evan Shieh, Faye-Marie Vassel, Cassidy Sugimoto, and
Thema Monroe-White. 2024. Laissez-faire harms:
Algorithmic biases in generative language models.
Preprint, arXiv:2404.07475.
David Stap and Ali Araabi. 2023.
ChatGPT
is
not
a
good
Indigenous
translator.
In
Proceedings of the Workshop on Natural Language
Processing
for
Indigenous
Languages
of
the
Americas (AmericasNLP), pages 163–167, Toronto,
Canada. Association for Computational Linguistics.
Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Juraf-
sky, and Luke Melas-Kyriazi. 2024. A benchmark
for learning to translate a new language from one
grammar book. Preprint, arXiv:2309.16575.
Jörg Tiedemann. 2012.
Parallel data, tools and in-
terfaces in OPUS.
In Proceedings of the Eighth
International Conference on Language Resources
and Evaluation (LREC’12), pages 2214–2218, Istan-
bul, Turkey. European Language Resources Associa-
tion (ELRA).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0921/0921e310-6842-40b3-9e52-0027c1636cb5.png" style="width: 50%;"></div>
The following section provides examples of errors analyzed in Section 5.3, one error per type.
The following section provides examples of errors
analyzed in Section 5.3, one error per type.
Mistranslation: Complete Mistranslation
Model: GPT-3.5 - BASELINE - AUTO
Source:
qamqa taytaykipa munasqan lawata
yanurqanki
Gloss:
qam-qa
you-TOP
tayta-yki-pa
father-2.POSS-GEN
muna-sqa-n
like-SUBR-3SG
lawa-ta
soup-ACC
yanu-rqa-nki
cook-PST-2SG
Reference: tú cocinaste la sopita que le gusta a tu
papá
you cooked the soup that your dad likes
Output: yo te quiero mucho y mi amor por ti
nunca se acabará
I love you very much and my love for you will
never end
Mistranslation: Lexical Correspondence
Model: GPT-4o - C+G+M - AUTO
Source:
ñuqaqa illani runakunawan rimayta
munaptiy
Gloss:
ñuqa-qa
I-TOP
illa-ni
travel-1SG
runa-kuna-wan
person-PL-with
rima-y-ta
speak-ACC
muna-pti-y
like-SUBR-1SG
Reference: yo viajo porque me gustar hablar con
las personas
I travel because I like to talk to people
Output: yo quiero hablar con las personas cuando
amanece
I want to talk to people when the sun rises
Target Fluency: Coherence
Unresolvable pronoun
Source: qamqa yachayta tukuptiyki, puñunki
Gloss:
qam-qa
you-TOP
yacha-y-ta
study-INF-ACC
tuku-pti-yki,
finish-SUBR-2SG,
puñu-nki
sleep-2SG
Reference:
cuando termines de estudiar, tu
duermes
when you finish studying, you sleep
Output: acabaré de aprender lo que tú duermes
I will finish learning what you sleep
Target Fluency: Grammar
Missing causative and possessive
Model: GPT-4o - CORPUS - AUTO
Source: ñuqaqa ñañayman wasita pichachini
Gloss:
ñuqa-qa
I-TOP
ñaña-y-man
sister-1SG.POSS-ILL
wasi-ta
house-ACC
picha-chi-ni
clean-CAUS-1SG
Reference:
yo le hago limpiar la casa a mi
hermana
I make my sister clean the house
Output: yo hermana casa limpio
I sister house clean
Grammatical Divergence: TAM
Tense
Model: GPT-3.5 - C+G+M - AUTO
Source: qam allinta tusunki
Gloss:
qam
you
allin-ta
good-ACC
tusu-nki
dance-2SG
Reference: tu bailas bien
you dance well
Output: tú bailarás bien
you will dance well
Grammatical Divergence: Subject
‘you’ to ‘I’
Model: GPT-3.5 - MORPH - AUTO
Source: qamqa mamaykipa wasinta yanuq rinki
Gloss:
qam-qa
you-TOP
mama-yki-pa
mother-2.POSS-GEN
wasi-n-ta
house-3SG.POSS-ACC
yanu-q
cook-AGT
ri-nki
go-2SG
Reference: tú irás a cocinar a la casa de tu mamá
you will go to cook at your mother’s house
Output: voy a cocinar en la casa de tu madre
I’m going to cook at your mother’s house
Grammatical Divergence: Subject
‘you’ to ‘the man’
Model: GPT-4o - C+G+M - AUTO
Source:
qamqa misita qawaspa qari warata
awachkarqanki
Gloss:
qam-qa
you-TOP
misi-ta
cat-ACC
qawa-spa
watch-SUBR
qari
male
wara-ta
pants-ACC
awa-chka-rqa-nki
knit-PROG-PST-2SG
Reference: tú estabas tejiendo un pantalón de
varón mirando al gato
you were knitting a pair of men’s pants looking at
the cat
Output: tú mirabas al gato mientras el hombre
tejía el pantalón
you looked at the cat while the man knitted the
pants
Grammatical Divergence: TAM
Missing conditional
Model: GPT-4o - MORPH - AUTO
Model: GPT-4o - MORPH - AUTO
qam you runa-simi-ta people-mouth-ACC yacha-paya-pti-yki, learn-REP-COND-2SG, allin-ta-m good-ACC-ASRT rima-nki speak-2SG
Reference: si practicas el quechua lo hablarás bien if you practice Quechua you will speak it well
Output: tú ya sabías el runasimi, hablas muy bien you already knew runasimi, you speak very well
# B Unprompted Stereotypes
The following section provides examples of errors involving cultural stereotypes as discussed in Section 6.2.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6af6/6af6e5cb-ee8d-4eb9-857c-21d24d12efdc.png" style="width: 50%;"></div>
The following section provides examples of errors
involving cultural stereotypes as discussed in
Section 6.2.
Model: GPT-3.5 - BASELINE - AUTO
Source: payqa qatupi ñañanpa rantisqan apita
rurarqan
Gloss:
pay-qa
3-TOP
qatu-pi
market-LOC
ñaña-n-pa
sister-3SG.POSS-GEN
ranti-sqa-n
buy-SUBR-3SG
api-ta
mazamorra-ACC
rura-rqa-n
make-PST-3SG
Reference:
ella preparó una mazamorra que
compró su hermana en el mercado
she prepared a mazamorra [pudding] that her
sister bought at the market
Output: el sol brilla en nuestra tierra, calienta y
nos da vida
the sun shines on our land, warms us and gives us
life
Model: GPT-3.5 - GRAMMAR
Source:
paykunaqa yachachiqninkupa rakta
medicina liwru apasqanta likanku
Gloss:
pay-kuna-qa
3-PL-TOP
yacha-chi-q-ni-nku-pa
learn-CAUS-AGT-NI-3PL.POSS-GEN
rakta
thick
medicina
medicine
liwru
book
apa-sqa-n-ta
carry-SUBR-3SG-ACC
lika-nku
read-3PL
Reference: ellos leerán el libro grueso de medicina
que trajo su profesor
they will read the thick medical book that their
teacher brought
Output:
los médicos de la comunidad nos
enseñaron a utilizar plantas medicinales
the doctors from the community taught us how to
use medicinal plants
Output: el sol brilla en nuestra tierra, calienta y nos da vida the sun shines on our land, warms us and gives us life
Source: paykunaqa yachachiqninkupa rakta medicina liwru apasqanta likanku
Source: qamkunapa taytamamaykichikta watukunaykichik, paykunata purikuq apanaykichik
<div style="text-align: center;">C Quality Descriptions</div>
Quality
Description
High
Output is an accurate and/or acceptable translation of the source content.
Med
Output contains errors that prevent it from being an acceptable translation, but is generally
high in quality otherwise.
Low
Output contains errors that prevent it from being an acceptable translation, with minor
correspondences that vaguely identify it as relevant to the source.
None
Output does not appear to be relevant to the source.
<div style="text-align: center;">D Annotation Error Typology</div>
Dimension
Error
Description
Accuracy
Addition
Translation includes information not present in the source, but does
not result in the displacement of source content.
Accuracy
Omission
Translation is missing content from the source.
Accuracy
Substitution - Subject
The translated segment contains content identified as relevant to the
source in other spans, but substitutes novel subject markers for those
present in the source in the highlighted span; Classify an error as a
“substitution” when the error appears to result in both Addition and
Omission errors that cannot be distinguished into two distinct spans.
Accuracy
Substitution - TAM
The translated segment contains content identified as relevant to the
source in other spans, but substitutes novel TAM for those present in
the source in the highlighted span; Classify an error as a “substitution”
when the error appears to result in both Addition and Omission errors
that cannot be distinguished into two distinct spans.
Accuracy
Substitution - Other
Substitution errors that do not involve mistranslated subject markers or
TAM. See above.
Accuracy
Overtranslation
Error occurring in the target content that is inappropriately more
specific than the source content.
Accuracy
Undertranslation
Error occurring in the target content that is inappropriately less spe-
cific than the source content.
Target Error
Grammar
Other spans in the translated segment may be identified as relevant to
the source, but the highlighted span is not grammatical in the target
language.
Target Error
Coherence
Other spans in the translated segment may be identified as relevant to
the source, but the highlighted span is unnatural or incoherent in the
target language.
Target Error
Style/Register
Other spans in the translated segment may be identified as relevant to
the source, but the highlighted span is produced in a style or register
that is inappropriate given the content.
Non-Translation
Complete Mistransla-
tion
The entire segment is coherent in the target language but the core
predicate shows no immediate connection to the reference translation.
Non-Translation
Mistranslation - Lexical
Correspondence
The entire segment is coherent in the target language but only minor
correspondences to the reference translation may be identified.
Non-Translation
Refusal
Model does not attempt to translate into the target language, e.g.,
because it "does not understand".
Model error
Garbled
Output does not contain coherent text in the target language.
Model error
ChattyGPT
Output contains translated content, but is wordy, over-explanatory,
and/or abruptly truncated.
<div style="text-align: center;">Table 6: Adapted MQM typology for human error annotation</div>
# E Example Prompts
The following are examples of prompts generated used automated retrieval from the database. English is included in italics for the reader, but was not provided to the models as part of the prompt.
BASELINE
[TAREA] Traduce la siguiente frase del quechua al
español. Responde sólo con la traducción:
quechua: qam allinta tusunki
español:
[TASK] Translate the following sentence from
Quechua to Spanish.
Respond only with the
translation:
Quechua: You dance well
Spanish:
[CONTEXTO] qam: [PrnPers+2sg] allin: bueno [ ˆDB][NRoot] ta: [+Acc][Cas] tusu: bailar [VRoot][ ˆDB] nki: [+2sg.Subj][VPers]
allin. adj. Bueno (término de aprobación). SINÓN: kusa. EJEM: allin p’unchay, buenos días: allin tuta, buenas noches; allin tutamanta, buena mañana, buenos días; allin inti chinkay, buenas tardes; allin iñiyniyoq, de buena fe, fiel, justo, íntegro: allin nunayoq, de espíritu bueno; allin puriq, de comportamiento bueno; allin puriy, comportamiento bueno; allin rikuy, tratamiento bueno; allin rikuq, el que trata bien; allin ruway, obrar bien, beneficiar; lo que se hace bien, beneficioso; allin ruwaq, el que hace bien; allin yuyay, pensar bien; pensamiento bueno; allin qolqeyoq, poseedor de plata fina; adinerado. ta. s. Gram. Sufijo que desempeña los papeles de artículo y preposición. EJEM: llamata qatiy, arrea la llama; Urkusmanta hamuni, vengo de Urcos.
# [TAREA] Traduce la siguiente frase . . .
[TAREA] Traduce la siguiente frase . . .
[CONTEXT] qam: [PrnPers+2sg] allin: bueno [ ˆDB][NRoot] ta: [+Acc][Cas]
tusu: bailar [VRoot][ ˆDB] nki: [+2sg.Subj][VPers] allin. adj. Good (term of approval). SYN: kusa. EX: allin p’unchay, good day: allin tuta, good evening; allin tutamanta, good morning, good day; allin inti chinkay, good afternoon; allin iñiyniyoq, good faith, faithful, just, upright: allin nunayoq, in good spirits; allin puriq, well behaved; allin puriy, good behavior; allin rikuy, good treatment; allin rikuq, one who treats others well; allin ruway, to do good, to benefit; one who does good, beneficial; allin ruwaq, one who does good; allin yuyay, think well; good thought; allin qolqeyoq, possessor of fine silver; wealthy. ta. s. Gram. Suffix that plays the roles of article and preposition. EX: llamata qatiy, herd the llama; Urkusmanta hamuni, I come from Urcos.
[CONTEXTO] ta: CASO ACUSATIVO. Su marca es –ta, esta es una marca de objeto directo con los verbos que no son de movimiento (quietud). Ejemplo: Quyllur–ta qhawani Veo una estrella T’anta–ta apay Lleva pan Ñuqa quylluyta qhawani Pedrucha t’antata rantin En cambio con los verbos de movimiento –ta indica (hacia) que es igual a meta. Ejemplos: Punu–ta rini Voy a Puno Llaqta-ta risaq Iré al pueblo Hamawt’anchis Punuta rinqa Llanta umalliq llaqtata richkan nki: FLEXIÓN DE TIEMPO. TIEMPO FUTURO. TIEMPO FUTURO. Los sufijos para cada una de las personas gramaticales son: saq, nki, nqa, sun, saqku, nkichis, nqaku; en singular y plural respectivamente. Ejemplos: Puklla-saq jugaré Puklla-nki jugarás Puklla-nqa jugará Puklla-sun jugaremos Puklla-saqku jugaremos Puklla-nkichis Uds. jugarán Puklla-nqaku ellos jugarán
[TAREA] Traduce la siguiente frase . . .
[CONTEXT] ta: ACCUSATIVE CASE. Marked by –ta, this is a direct object marker with verbs that don’t indicate movement. Example: Quyllur–ta qhawani I see a star T’anta–ta apay Bring bread Ñuqa quylluyta qhawani Pedrucha t’antata rantin On the other hand, with verbs of motion -ta indicates (towards) the same goal Examples: Punu–ta rini I go to Puno Llaqta-ta risaq I will go to town Hamawt’anchis Punuta rinqa Llanta umalliq llaqtata richkan nki: TENSE INFLECTION. FUTURE TENSE. FUTURE TENSE. The suffixes for each of the grammatical persons are: saq, nki, nqa, sun, saqku, nkichis, nqaku; in singular and plural respectively. Ejemplos: Puklla-saq jugaré Puklla-nki jugarás Puklla-nqa jugará Puklla-sun jugaremos Puklla-saqku jugaremos Puklla-nkichis Uds. jugarán Puklla-nqaku ellos jugarán
CORPUS-ONLY
[CONTEXTO]
quechua: rimanakunapaq wawakunapa rimasqan
simi
aswan
allinta
takyachinaraq
piwanpas
maywanpas
mana
manchakuspa
rimananpaq
chaymi qillqanapaqpas ñawichanapaqpas aswan
allin kanqa
español: para este diálogo saber la lengua que
dominan los niños sería importante para que ellos
se expresen sin miedo de ahí será que la escritura y
la lectura salga de manera óptima
quechua:
kay tiqsipi sumaq rimanakunapaqa
kawsayninchikmi allinta kallpachawanchik runaku-
nahina allinta tiyanapaq chaymi ñuqanchikkqa
allinta ñawichayta qillqayta yachananchik ñawpa
ayllunchikkuna rurasqankuta maytukunapi tukuy
puyñukunapi tiqsi muyu qhawarisqankuta
español:
para vivir en armonía tenemos que
conocer bien nuestra forma de vivir y luego
escribir leer tambien a valorar lo que nos dejaron
[CONTEXTO] quechua: rimanakunapaq wawakunapa rimasqan simi aswan allinta takyachinaraq piwanpas maywanpas mana manchakuspa rimananpaq chaymi qillqanapaqpas ñawichanapaqpas aswan allin kanqa español: para este diálogo saber la lengua que dominan los niños sería importante para que ellos se expresen sin miedo de ahí será que la escritura y la lectura salga de manera óptima quechua: kay tiqsipi sumaq rimanakunapaqa kawsayninchikmi allinta kallpachawanchik runakunahina allinta tiyanapaq chaymi ñuqanchikkqa allinta ñawichayta qillqayta yachananchik ñawpa ayllunchikkuna rurasqankuta maytukunapi tukuy puyñukunapi tiqsi muyu qhawarisqankuta español: para vivir en armonía tenemos que conocer bien nuestra forma de vivir y luego escribir leer tambien a valorar lo que nos dejaron
[CONTEXTO] quechua: rimanakunapaq wawakunapa rimasqan simi aswan allinta takyachinaraq piwanpas maywanpas mana manchakuspa rimananpaq chaymi qillqanapaqpas ñawichanapaqpas aswan allin kanqa español: para este diálogo saber la lengua que dominan los niños sería importante para que ellos se expresen sin miedo de ahí será que la escritura y la lectura salga de manera óptima quechua: kay tiqsipi sumaq rimanakunapaqa kawsayninchikmi allinta kallpachawanchik runakunahina allinta tiyanapaq chaymi ñuqanchikkqa allinta ñawichayta qillqayta yachananchik ñawpa ayllunchikkuna rurasqankuta maytukunapi tukuy puyñukunapi tiqsi muyu qhawarisqankuta español: para vivir en armonía tenemos que conocer bien nuestra forma de vivir y luego escribir leer tambien a valorar lo que nos dejaron
nuestros antecesores en cada visión sobre el mundo quechua: winsislawcha chayarqamuptinsi tuparquspanku allinta qatunakusqanku suwakuypi purinankupaq español: cuando había llegado wenseslau y a su encuentro se habían reforzarón para andar a robar
# [TAREA] Traduce la siguiente frase . . .
[TAREA] Traduce la siguiente frase . . .
[CONTEXT] quechua: rimanakunapaq wawakunapa rimasqan simi aswan allinta takyachinaraq piwanpas maywanpas mana manchakuspa rimananpaq chaymi qillqanapaqpas ñawichanapaqpas aswan allin kanqa Spanish: For this dialogue, knowing the language that the children speak would be important for them to express themselves without fear, and that is why writing and reading will be optimal. quechua: kay tiqsipi sumaq rimanakunapaqa kawsayninchikmi allinta kallpachawanchik runakunahina allinta tiyanapaq chaymi ñuqanchikkqa allinta ñawichayta qillqayta yachananchik ñawpa ayllunchikkuna rurasqankuta maytukunapi tukuy puyñukunapi tiqsi muyu qhawarisqankuta Spanish:To live in harmony we have to know our way of life well and then write and read to also value what our ancestors left us in each vision of the world. quechua: winsislawcha chayarqamuptinsi tuparquspanku allinta qatunakusqanku suwakuypi purinankupaq español: cuando había llegado wenseslau y a su encuentro se habían reforzarón para andar a robar
[TASK] Translate the following sentence . . .
# F Full Quality Scores
This section contains tables showing all automatic and human-annotated quality scores for each of our experiments. Table 7 contains the full set of BLEURT scores summarized in Tables 1 and 3 of the main text. Table 8 shows the corresponding BLEU scores for the same experiments. Table 9 and Table 10 contain the full set of the human-annotated scores summarized in Table 3.
<div style="text-align: center;">GPT-3.5</div>
GPT-3.5
GPT-4o
Gemini-1.5
Llama 3
auto
manual
auto
manual
auto
manual
auto
manual
BASELINE
0.19
0.22
0.66
0.66
0.56
0.57
0.15
0.16
CORPUS-ONLY
0.27
0.29
0.59
0.61
0.49
0.47
0.19
0.18
GRAMMAR-ONLY
0.23
0.24
0.56
0.58
0.55
0.54
0.17
0.15
MORPH-ONLY
0.44
0.56
0.54
0.63
0.61
0.66
0.39
0.49
CORPUS-GRAMMAR
0.26
0.28
0.59
0.59
0.54
0.53
0.21
0.21
CORPUS-MORPH
0.44
0.52
0.59
0.64
0.59
0.64
0.36
0.38
GRAMMAR-MORPH
0.41
0.54
0.53
0.61
0.61
0.64
0.39
0.37
CORPUS-GRAMMAR-MORPH
0.43
0.54
0.57
0.63
0.61
0.63
0.15
0.26
Table 7: BLEURT scores for all LLMs and prompt types.
GPT-3.5
GPT-4o
Gemini-1.5 Pro
Llama 3 8B
auto
manual
auto
manual
auto
manual
auto
manual
BASELINE
0.01
0.02
0.19
0.18
0.12
0.14
0.00
0.00
CORPUS-ONLY
0.02
0.02
0.16
0.22
0.14
0.13
0.02
0.01
GRAMMAR-ONLY
0.01
0.03
0.14
0.12
0.18
0.17
0.01
0.01
MORPHS-ONLY
0.06
0.08
0.12
0.13
0.15
0.18
0.03
0.05
CORPUS-GRAMMAR
0.01
0.01
0.14
0.17
0.12
0.08
0.01
0.01
CORPUS-MORPHS
0.05
0.08
0.19
0.18
0.17
0.17
0.02
0.04
GRAMMAR-MORPHS
0.03
0.04
0.11
0.10
0.15
0.16
0.02
0.01
CORPUS-GRAMMAR-MORPHS
0.04
0.04
0.16
0.16
0.17
0.20
0.00
0.01
<div style="text-align: center;">Table 8: BLEU scores for all LLMs and prompt types.</div>
<div style="text-align: center;">GPT-3.5 Turbo</div>
None
Low
Med
High
BASELINE
31
17
2
0
CORPUS-ONLY
18
23
8
1
GRAMMAR-ONLY
20
27
2
1
MORPHS-ONLY
3
22
16
9
CORPUS-GRAMMAR
18
23
9
0
CORPUS-MORPH
2
28
12
8
GRAMMAR-MORPH
3
29
13
5
CORPUS-GRAMMAR-MORPH
2
27
12
9
Table 9: Human quality annotation of GPT-3.5 outputs with automated retrieval (raw counts out of 50) by promp
<div style="text-align: center;">GPT-4o</div>
None
Low
Med
High
BASELINE
0
10
20
20
CORPUS-ONLY
1
16
13
20
GRAMMAR-ONLY
0
17
16
