# Improving Performance of Automatic Keyword Extraction (AKE) Methods Using PoS-Tagging and Enhanced Semantic-Awareness
Enes Altuncua,∗, Jason R.C. Nursea, Yang Xub, Jie Guob, Shujun Lia
aInstitute of Cyber Security for Society (iCSS) & School of Computing, University of Kent, Canterbury, CT2 7NP, UK bShanghai Jiao Tong University, Shanghai, China
# Abstract
Automatic keyword extraction (AKE) has gained more importance with the increasing amount of digital textual data that modern computing systems process. It has various applications in information retrieval (IR) and natural language processing (NLP), including text summarisation, topic analysis and document indexing. This paper proposes a simple but effective post-processing-based universal approach to improve the performance of any AKE methods, via an enhanced level of semantic-awareness supported by PoS-tagging. To demonstrate the performance of the proposed approach, we considered word types retrieved from a PoS-tagging step and two representative sources of semantic information – specialised terms defined in one or more context-dependent thesauri, and named entities in Wikipedia. The above three steps can be simply added to the end of any AKE methods as part of a post-processor, which simply re-evaluate all candidate keywords following some context-specific and semantic-aware criteria. For five stateof-the-art (SOTA) AKE methods, our experimental results with 17 selected datasets showed that the proposed approach improved their performances both consistently (up to 100% in terms of improved cases) and significantly (between 10.2% and 53.8%, with an average of 25.8%, in terms of F1-score and across all five methods), especially when all the three enhancement steps
are used. Our results have profound implications considering the ease to apply our proposed approach to any AKE methods and to further extend it. Keywords: keyword extraction, pos-tagging, semantic-awareness, context-awareness
# 1. Introduction
Keyword extraction (KE), also known as keyphrase or key term extraction, is an information extraction task that aims to identify a number of words/phrases that best summarise the nature or the context of a piece of text. It has several applications in information retrieval (IR) and natural language processing (NLP), including text summarisation, topic analysis and document indexing. Considering the vast amount of text-based documents online in today’s digital society, it is very useful to be able to extract keywords from online documents automatically to support large-scale textual analysis. Therefore, for many years the research community has been investigating automatic keyword extraction (AKE) methods, especially with the recent advancements in artificial intelligence (AI) and NLP. Despite these efforts, however, AKE has been shown to be a challenging task and AKE methods with very high performance are still to be found (Papagiannopoulou and Tsoumakas, 2020). Two main challenges are the lack of a precise definition of the AKE task and the lack of consistent performance evaluation metrics and benchmarks (Merrouni et al., 2020). Since there is no consensus on the definition and characteristics of a keyword, KE datasets created by researchers have different characteristics, e.g., the minimum/average/maximum numbers of keywords, if absent keywords (human-labeled keywords that do not appear in the text) are allowed, and what part-of-speech (PoS) tags such as verbs are accepted as valid keywords. This makes performance evaluation and comparison of AKE methods more difficult. Based on whether a labelled training set is used, AKE methods reported in the literature can be grouped into unsupervised and supervised methods. Unsupervised methods include statistical, graph-based, embeddingbased and/or language model-based methods, while supervised ones use either traditional or deep machine learning models (Papagiannopoulou and Tsoumakas, 2020). Surprisingly, most AKE methods have not considered semantic information to align the returned keywords with the semantic context of the input document.
In this work, to fill the above-mentioned gap on the lack of use of semantic information in the state-of-the-art (SOTA) AKE methods, we propose a universal performance improvement approach for any AKE methods, as a post-processor that can consider semantic information more explicitly, with the support of PoS-tagging. To start with, we conducted an analysis of human-created ‘gold standard’ keywords in 17 KE datasets to better understand some relevant characteristics of such keywords, focusing on PoS-tag patterns, n-gram sizes, and the possible consideration of semantic information by human labellers when extracting keywords. Our proposed approach is demonstrated using the following three postprocessing steps that can be freely combined: (1) keeping candidate keywords with a desired PoS-tag only; (2) matching candidate keywords with one or more context-specific thesauri containing more semantically relevant terms; and (3) prioritising candidate keywords that appear as a valid Wikipedia named entity. We applied different combinations of the above three postprocessing steps to five SOTA AKE methods, YAKE! (Campos et al., 2020), KP-Miner (El-Beltagy and Rafea, 2009), RaKUn (ˇSkrlj et al., 2019), LexRank (Ushio et al., 2021), and SIFRank+ (Sun et al., 2020), and compared the performances of the original methods with those of the enhanced versions. The experimental results with the 17 KE datasets showed that our proposed postprocessing steps helped improve the performances of all the five SOTA AKE methods both consistently (up to 100% in terms of improved cases) and significantly (between 10.2% and 53.8%, with an average of 25.8%, in terms of F1-score and across all five methods), particularly when all the three steps are combined. The source code and the complete experimental results of our work (including some details that are not reported in this paper due to the space limit) will be released after the paper is accepted for publication. The rest of the paper is organised as follows. Section 2 briefly surveys AKE methods in the literature. The analysis of the human-created keywords in 17 KE datasets is given in Section 3. In Section 4, we present the methodology of our study. Section 5 explains the experimental setup for evaluation as well as the results. Finally, the paper is concluded with some further discussions in Section 6, and an overall summary in Section 7.
# 2. Related Work
# 2.1. Unsupervised AKE Methods
Some unsupervised AKE methods have been proposed, including statistical, graph-based, embedding-based and language model-based methods (Papagiannopoulou and Tsoumakas, 2020). Statistical AKE methods rely on some selected statistical metrics, e.g., term frequency, relevance to context, and co-occurrences, for ranking candidate keywords. One of the mostly used metrics is TF-IDF (Jones, 1972), which combines two aspects of a term: term frequency within the input article, and the inverse document frequency across several domains. One AKE method using TF-IDF is KP-Miner (El-Beltagy and Rafea, 2009), which also considers other metrics such as word length and word position. A more recent method in this category is YAKE! (Campos et al., 2020), which leverages a range of statistical metrics, such as casing, word position, word frequency, word relatedness to context and how often a term appears in different sentences. Finally, LexSpec (Ushio et al., 2021) makes use of lexical specificity – a statistical metric to select the most representative keywords from a given text based on the hypergeometric distribution. Graph-based AKE methods consider candidate keywords as nodes in a directed graph, often with weighted edges reflecting syntactic/semantic relatedness of different keywords. They leverage graph-based methods, such as PageRank (Brin and Page, 1998), for ranking the nodes of the graph in terms of their overall importance. The earliest AKE method in this category is TextRank (Mihalcea and Tarau, 2004), which uses an unweighted graph of candidate keywords after filtering the ones that are not a noun or an adjective, and uses PageRank for ranking the nodes. As an extension to TextRank, SingleRank (Wan and Xiao, 2008) adds edge weights to the graph, which reflect the number of co-occurrences of the candidate keywords represented by any pair of two connected nodes. Another graph-based AKE method is RAKE (Rose et al., 2010), which builds a word-word co-occurrence graph, and assigns a score for each candidate by using word frequency and word degree. A more recent graph-based AKE method is RaKUn (ˇSkrlj et al., 2019), which introduces meta-vertices by aggregating similar vertices, and employs load centrality metrics for candidate ranking. Finally, LexRank (Ushio et al., 2021) and TFIDFRank (Ushio et al., 2021) are two different enhanced versions of SingleRank, which use lexical specificity and TF-IDF, respectively.
Embedding-based AKE methods utilise word representation techniques, such as Doc2Vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014). An example method in this category is EmbedRank (Bennani-Smires et al., 2018), which uses sentence embeddings, and ranks candidate keywords in terms of cosine similarity. A more recent method is SIFRank (Sun et al., 2020), which combines sentence embedding model SIF and autoregressive pre-trained language model ELMo, and it was upgraded to SIFRank+ by position-biased weight to improve its performance for long documents. Lastly, MDERank (Zhang et al., 2022) considers the similarity between the embeddings of the source document and its masked version for candidate ranking. Apart from the AKE methods mentioned above, there also exist a number of AKE methods based on other techniques. Rabby et al. (2020) proposed TeKET, a domain- and language-independent AKE technique utilising a binary tree for extracting final keywords from candidate ones. As another example, Liu et al. (2009) introduced an AKE algorithm based on term clustering considering semantic relatedness to identify the exemplar terms. The identified exemplar terms are, then, used to extract keywords.
# 2.2. Supervised AKE Methods
Although unsupervised methods are preferred for AKE, supervised methods have also been proposed. One of the earliest methods is KEA (Witten et al., 2005), which calculates TF-IDF scores and the position of the first occurrence of each candidate, and employs the Naive Bayes learning algorithm to decide if a candidate should be selected. More recently, there is a growing interest to use deep learning for AKE. For example, Basaldella et al. (2018) proposed an AKE method based on Bi-LSTM, which is capable of exploiting the context of each candidate word. Another AKE method, TNTKID (Martinc et al., 2021), leverages transformers, and allows users to train their own language model on a domain-specific corpus. A third example is TANN (Wang et al., 2018), an AKE method based on a topic-based artificial neural network model, which aims to improve the performance of AKE by transferring knowledge from a resource-rich source domain to an unlabelled or an insufficiently labelled target domain.
# 2.3. PoS-Tagging and Semantics in AKE
Many AKE methods have considered how to extract more semantically meaningful keywords. For this purpose, PoS-tagging has been used so that extracted keywords are restricted to a pre-defined set of PoS-tag patterns,
e.g., noun phrases only (Hulth, 2003; Pay, 2016; Zervanou, 2010). Some methods utilise external knowledge to provide useful contextual information for extracting more semantically sensible keywords. For instance, Li and Wang (2014) proposed a TextRank-based AKE method that benefits from domain knowledge by using author-assigned keywords of scientific publications, and Gazendam et al. (2010) proposed to use semantic relations between thesaurus terms for ranking candidate keywords without a reference corpus (Gazendam et al., 2010). Thesaurus relations have also been combined with machine learning techniques to improve performance of AKE methods (Hulth et al., 2001; Medelyan and Witten, 2006). Some AKE methods also make use of Wikipedia, a useful source of semantic information. Shi et al. (2008) utilised Wikipedia to extract semantic features of candidate keywords. Their method constructs a semantic graph connecting candidate keywords to document topics based on the hierarchical relations extracted from Wikipedia, and semantic feature weights are assigned to candidate keywords with a link analysis algorithm. WikiRank is another AKE method leveraging Wikipedia (Yu and Ng, 2018). It employs the TAGME annotator (Ferragina and Scaiella, 2010) to link meaningful word sequences in the input document to concepts in Wikipedia, and constructs a semantic graph. Then, it transforms the KE task to an optimisation problem on the graph, and tries to obtain the optimal keyword set that has the best coverage of the identified concepts. Finally, several embedding-based AKE methods utilise Wikipedia for pre-training and/or fine-tuning their underlying embedding methods (Bennani-Smires et al., 2018; Papagiannopoulou and Tsoumakas, 2018). Compared with existing AKE methods that have considered PoS-tagging or semantic information more explicitly, our proposed approach is more universal and can be applied to any AKE methods as a post-processor, which simply re-evaluate candidate keywords generated by an AKE method before top n keywords are returned. Our approach is easily generalisable and can be used flexibly to eliminate candidate keywords that are unlikely to be a keyword, and prioritise ones that are more likely to be a keyword.
# 3. Analysis of Human-Created Keywords
Our proposed approach was motivated by some of our observations regarding how human labellers extracted “golden” (i.e., ground truth) keywords in 17 KE datasets. Such observations also helped us to determine
some fine details of our proposed approach. In the following, we describe the 17 datasets we used and the key observations.
# 3.1. Datasets Inspected
Considering the subjectivity of the keyword extraction task, there is no standard approach to follow in constructing keyword extraction datasets (Zesch and Gurevych, 2009). This brings an extreme diversity to the datasets constructed so far, which makes testing a keyword extraction algorithm thoroughly harder. Therefore, to achieve a better understanding of humancreated keywords, we aimed at collecting a large number of datasets used in the literature. In total, we selected 17 datasets covering multiple contexts, including agriculture, computer science and health, and several types of documents, such as scientific papers, news, theses and abstracts. Further details regarding the datasets can be seen in Table 1.
# 3.2. Observations: PoS-Tag Patterns
There have been a lot of research on linguistic properties of different multi-word expression types, such as collocations (Smadja, 1993) and technical terms (Justeson and Katz, 1995). However, these are unable to properly explain the linguistic properties of keywords used in AKE research because of the lack of linguistic standards for human-created keywords. Therefore, firstly, we reviewed the structure of human-created keywords in the 17 datasets, in terms of the used PoS-tag patterns. For this purpose, we used the NLTK (Bird, 2006) library’s PoS tagger and computed the distribution of different PoS-tag patterns. As shown in Table 2, nine of the top ten PoS-tag patterns correspond to either noun or gerund phrases. The only non-noun/gerund pattern in the top ten PoS-tag patterns is a single adjective (JJ), with an average percentage of 6.85%. The top ten PoS-tag patterns count 80% of all patterns. These observations imply that leveraging knowledge about how human labellers define keywords based on PoS-tag patterns for a specific domain can potentially help improve the performance of any AKE methods for the corresponding domain.
# 3.3. Observations: n-Gram Size
AKE methods generally include a parameter for the maximum n-gram size, corresponding to the maximum number of words a keyword is allowed to contain. Although it is well-known that multi-word expressions (MWEs) are more likely to be of length two to three in English (Choueka, 1988),
Dataset
Content Context
Size
Avg. #(Keys) Abs. Keys
Annotators1
KPCrowd (Marujo
et al., 2013)
News
Misc.
500
48.92
13.5%
Readers
citeulike180 (Medelyan
et al., 2009)
Paper
Misc.
183
18.42
32.2%
Readers
DUC-2001 (Wan and
Xiao, 2008)
News
Misc.
308
8.1
3.7%
Readers
fao30 (Medelyan and
Witten, 2008)
Paper
Agr.
30
33.23
41.7%
Experts
fao780 (Medelyan and
Witten, 2008)
Paper
Agr.
779
8.97
36.1%
Experts
Inspec (Hulth, 2003)
Abstract
CS
2,000
14.62
37.7%
Experts
KDD (Das Gollapalli
and Caragea, 2014)
Abstract
CS
755
5.07
53.2%
Authors
KPTimes
(test) (Gallina et al.,
2019)
News
Misc.
20,000
5.0
54.7%
Editors
Krapivin2009 (Krapivin
et al., 2009)
Paper
CS
2,304
6.34
15.3%
Authors
Nguyen2007 (Nguyen
and Kan, 2007)
Paper
CS
209
11.33
17.8%
Authors & Readers
PubMed (Gay et al.,
2005)
Paper
Health
500
15.24
60.2%
Authors
Schutz2008 (Schutz,
2008)
Paper
Health
1,231
44.69
13.6%
Authors
SemEval2010 (Kim
et al., 2010)
Paper
CS
243
16.47
11.3%
Authors & Readers
SemEval2017 (Augen-
stein et al.,
2017)
Paragr
Misc.
493
18.19
0.0%
Experts & Readers
theses1002
Thesis
Misc.
100
7.67
47.6%
Unknown
wiki20 (Medelyan
et al., 2008)
Report
CS
20
36.50
51.2%
Readers
WWW (Das Gollapalli
and Caragea, 2014)
Abstracts
CS
1,330
5.80
55.0%
Authors
 
it is less clear how human labellers of the 17 datasets were instructed to consider the n-gram size. Therefore, we analysed the golden keywords across the 17 datasets to see how human labellers decided the n-gram sizes. On average, bigrams (n = 2) constitute 45.55% of the golden keywords in the 17 datasets, while this rate is 36.45% for unigrams (n = 1) and 12.73% for trigrams (n = 3). In addition, the percentages for keywords with n ≥4 are considerably low – 5.12% on average. More detailed statistics can be seen in Table 3. These results show that human labellers largely used two or three as the maximum n-gram size, covering 82.01% and 94.74% of the golden
<div style="text-align: center;">Table 2: Percentages of top 10 PoS-tag patterns across 17 datasets. PoS tags: NN – noun (singular), NNS – noun (plural), JJ – adjective, VBG – verb gerund.</div>
Dataset
NN
NN NN JJ NN NNS
JJ
JJ NNS NN NNS JJ NN NN VBG NN NN NN
KPCrowd
31.38
2.18
3.29
11.65 10.13
0.95
0.95
0.26
5.27
0.17
citeulike180
48.71
7.03
4.78
12.93 12.74
1.61
1.56
0.15
1.95
0.05
DUC-2001
19.13
15.90
15.28 10.49 1.80
8.73
10.16
3.65
0.28
1.52
fao30
32.60
14.68
7.92
15.84 5.06
6.62
9.35
0.00
0.78
0.26
fao780
29.56
14.11
9.11
15.18 3.78
6.02
10.88
0.06
1.21
0.04
Inspec
19.05
12.57
12.49
6.64
3.85
8.11
5.95
4.35
1.11
2.50
KDD
27.93
13.49
9.06
5.89
9.25
5.13
3.55
2.22
4.81
0.76
KPTimes
15.32
16.65
15.67
4.27
2.83
8.62
6.26
2.92
1.76
1.51
Krapivin2009 35.15
4.70
4.06
14.14 5.67
2.17
1.47
0.27
0.95
0.17
Nguyen2007
20.85
19.83
11.31
4.84
2.53
4.79
3.37
3.06
1.51
2.66
PubMed
30.88
9.23
3.87
15.43 12.01
3.51
5.50
0.77
0.56
2.03
Schutz2008
30.15
6.20
10.61 18.63 10.91
5.04
3.19
1.61
0.31
0.66
SemEval2010 19.45
21.74
21.54
0.08
3.20
0.17
0.06
6.40
0.42
3.15
SemEval2017 14.57
8.73
9.00
7.23
2.12
5.95
4.46
3.31
0.66
1.62
theses100
27.88
8.55
5.39
9.48 15.24
6.13
4.28
0.00
1.30
0.19
wiki20
41.91
18.65
11.06
1.49
6.60
0.50
1.82
2.81
2.81
0.99
WWW
32.33
13.44
8.98
5.41
8.74
3.88
3.88
1.63
2.86
1.05
Average (%) 28.05
12.22
9.61
9.39
6.85
4.59
4.51
1.97
1.68
1.13
keywords across the different datasets, respectively. The results are aligned with those in the research literature on MWEs. Based on such observations, we can see that AKE methods could benefit from focusing more on keywords with a shorter word length.
# 3.4. Observations: Semantic Information
Finally, we analysed the human-created keywords to see if human labellers explicitly or implicitly relied on semantic information to select keywords. We first calculated the percentage of golden keywords that are covered by Wikipedia across all the datasets. This quantitative analysis indicated that, on average, 64.39% of the golden keywords are Wikipedia named entities, i.e., titles of Wikipedia articles. This interesting (previously unreported) finding justifies that Wikipedia can be a very useful knowledge base for AKE algorithms as it covers so many golden keywords chosen by human labellers for all the 17 datasets we chose. Although unexpected, this finding can be explained by the diversity and richness of the content of Wikipedia. More detailed results of the analysis can be seen in Table 4.
<div style="text-align: center;">Table 3: n-gram distributions of the 17 datasets</div>
Dataset
n = 1
n = 2
n = 3
n ≥4
n = 1, 2
1 ≤n ≤3
KPCrowd
73.78
18.47
4.90
2.83
92.25
97.15
citeulike180
77.10
19.98
2.79
0.09
97.08
99.87
DUC-2001
17.32
61.29
17.73
3.66
78.61
96.34
fao30
43.02
52.74
3.41
0.83
95.76
99.17
fao780
42.32
53.72
3.62
0.34
96.04
99.66
Inspec
16.44
53.68
23.05
6.84
70.12
93.17
KDD
25.48
56.32
13.97
4.24
81.80
95.77
KPTimes
46.68
34.39
12.55
6.38
81.07
93.62
Krapivin2009
18.95
61.61
15.74
3.70
80.56
96.30
Nguyen2007
27.53
49.96
15.42
6.97
77.49
92.91
PubMed
35.79
43.74
15.90
4.58
79.53
95.43
Schutz2008
57.83
30.22
8.15
1.67
88.05
96.20
SemEval2010
20.05
52.97
20.66
6.31
73.02
93.68
SemEval2017
25.23
33.74
17.19
23.84
58.97
76.16
theses100
31.63
50.37
11.09
6.90
82.00
93.09
wiki20
26.20
53.52
18.17
2.11
79.72
97.89
WWW
34.36
47.71
12.15
5.78
82.07
94.22
Average (%)
36.45
45.55
12.73
5.12
82.01
94.74
<div style="text-align: center;">Table 4: The percentages of golden keywords covered by Wikipedi</div>
Dataset
%
Dataset
%
KPCrowd
71.77
Nguyen2007
52.19
citeulike180
83.78
PubMed
81.28
DUC-2001
51.05
Schutz2008
67.43
fao30
80.97
SemEval2010
41.27
fao780
79.00
SemEval2017
31.02
Inspec
39.08
theses100
68.82
KDD
62.92
wiki20
89.01
KPTimes
79.09
WWW
63.83
Krapivin2009
52.12
In addition to Wikipedia named entities, we also manually inspected many golden keywords and observed that many collected datasets contain domain-specific golden keywords. This observation indicates that consider ing domain-specific terms can potentially help improve performance of AKE methods, too.
# 4. Methodology
Our proposed approach is based on three post-processing steps that can be applied to any baseline AKE methods: 1) filtering candidate keywords with unlikely PoS-tag patterns, 2) using one or more context-aware (i.e., domain specific) thesauri to prioritise important candidate keywords for the target domain, and 3) prioritising candidate keywords that are Wikipedia named entities. In the following, we explain the three steps with more details.
# 4.1. Filtering Specific PoS-Tag Patterns
As mentioned in Section 2.3, PoS-tagging has been extensively used in AKE methods to consider morpho-syntactic features. Motivated by the observations in Section 3.2, we attempted to leverage a PoS-tagger to filter out candidate keywords labelled with unlikely PoS-tag patterns. More precisely, candidate keywords that do not conform with any of the following PoS-tag patterns were discarded: (i) simple nouns and noun phrases – one or more nouns (optionally with one or more adjectives appearing before the first noun); (ii) two or more simple nouns and/or noun phrases connected by one or more prepositions or conjunctions1; and (iii) a single adjective. In the PoS-tag patterns mentioned above, nouns and adjectives mean any PoS tags that can provide the corresponding functionality in a sentence. Therefore, nouns also include gerunds, and adjectives also include past participle verbs. Considering the most common PoS-tag patterns mentioned in Section 3.2, our proposed PoS-tag patterns correspond to over 90% of the patterns observed across all the 17 datasets. We used NLTK to extract PoS tags for each term in the input documents. For pattern matching, we took the advantage of regular expressions. Since regular expressions return the longest possible matches, we extracted the shorter matches from the longest ones separately. Note that the proposed PoS-tag patterns can be further changed to reflect any domain-specific needs, e.g., we observed that gerunds are quite uncommon in health domain so they can be removed if preferred.
# 4.2. Context-Aware Thesauri
Context means any kind of domain, topic or field that has its own set of terms semantically specific to itself. While the set of terms specific to a context can be covered in a more structured vocabulary, such as a thesaurus or an ontology, a simple word list can often be sufficient for the purpose of AKE. As reported in Section 3.4, many keywords are related to the context of the input text, contextual consideration can be quite useful for AKE. Therefore, we propose to make use of external resources to inform AKE methods more about semantically useful keywords for the relevant domain. More specifically, we proposed to integrate one or more domain-specific thesauri, which contain terms specific to a target context, and to prioritise candidate keywords included in such thesauri. At the implementation level, we introduce a weight for each candidate keyword and increase the weight of any candidate keyword appearing in one of the thesauri. In our experiments, we doubled the weights of candidate keywords in a thesaurus, but the actual weight increase can be a parameter, which can be empirically determined based on some training data or qualitative evidence observed. To determine if a candidate keyword exists in a given thesaurus, we applied exact matching with lemmatisation. Although using stemming with exact matching is a more common practice in AKE (Papagiannopoulou and Tsoumakas, 2020), we preferred to use lemmatisation due to its context-awareness. In our experiments, we focused on thesauri with a single context, but using multiple contexts in a single thesaurus is of course also possible. Regarding integrating relevant thesauri, we considered two different approaches explained below. Manual Context Consideration:. This approach is more useful when documents processed by an AKE method are known to belong to a specific context. It utilises one or more thesauri containing a list of terms relevant to the context, which are given a higher weight for prioritisation by the AKE method. In our experiments, we assigned a single domain-specific thesaurus to each of the datasets to represent the relevant context. Note that it is possible that multiple contexts and multiple thesauri are used in some applications of AKE.
Automatic Context Identification:. Considering the wide range of applications in which AKE methods can be utilised, manually providing a thesaurus for each input document may not be very usable. Therefore, we also studied how to identify the context of an input document automatically, which can
allow assigning a different context and a corresponding thesaurus automatically. This can be achieved by building a machine learning based classifier, which produce a class label representing the context or a context-specific thesaurus of a given document or its abstract. Once the classifier predicts the context of an input abstract, we identify a thesaurus corresponding to the context, as defined in a context-to-thesaurus look-up table, to inform the AKE method. Unlike the manual approach, the automatic identification allows us to use a different thesaurus for each document in the dataset, therefore can be applied to many real-world scenarios where the documents processed can belong to multiple contexts.
# 4.3. Wikipedia Named Entities
Based on the Wikipedia-related observations reported in Section 3.4, we propose to use Wikipedia as a context-independent thesaurus to improve performance of any AKE methods working in any context(s). Similar to how a thesaurus can be used, we prioritise the candidate keywords covered by Wikipedia as an entry by increasing their weight, and we apply exact matching with lemmatisation to identify if a candidate keyword is a Wikipedia named entity. Since Wikipedia also contains a vast amount of entries with too general semantic meanings, e.g., unigrams such as ‘father’, ‘school’ and ‘table’ that are less likely to be a useful keyword for any AKE task, we utilised the NLTK’s words corpus (i.e., a wordlist including common English dictionary words) to identify such unigrams and remove them from the desired Wikipedia entities. For the Wikipedia named entities, we used the 2021-10-01 version of the English Wikipedia dump2, containing only page titles. We firstly cleaned the dump data by removing the disambiguation tags3 added next to the title by Wikipedia. Then, we normalised the data with lemmatisation and lower-casing by following the common practice.
# 5. Experiments and Results
# 5.1. Evaluation Metrics
As the evaluation metrics, we used precision, recall and F1-score at top ten keywords, which have been commonly used in AKE evaluation (Pa-
pagiannopoulou and Tsoumakas, 2020). Furthermore, we adopted microaveraging, and exact matching with stemming when calculating the scores.
# 5.2. Selecting Baseline Methods
5.2. Selecting Baseline Methods To show the effectiveness and generalisability of the proposed methods, we first attempted to identify some representative AKE algorithms with different key characteristics for our experiments. We reviewed existing AKE algorithms in terms of multiple aspects, e.g., recency, ease of reconfiguration, and if they already use one or more of our proposed methods by any means, as shown in Table 5. These methods are considered more representative because they have open-source implementations, are applicable to any document type, were validated on a number of datasets, and do not require training (i.e., unsupervised so that it is easier to use and less likely to have generalisation problems)4. Among these methods, we selected two statistical methods, i.e., KP-Miner and YAKE!, two graph-based methods, i.e., RaKUn and LexRank, and an embedding-based method, i.e., SIFRank+, as baseline methods for our experiments. Since SIFRank+ is very computationally costly, we used only seven of the datasets, containing shorter documents (i.e., KPCrowd, DUC-2001, Inspec, KDD, KPTimes, SemEval2017 and WWW) for its evaluation when our methods were applied. For the implementations of the selected AKE methods, we utilised the PKE (Boudin, 2016) library for KP-Miner, and the original implementations of the other four. We used the default parameters for all the methods, except the maximum n-gram size parameter. Considering the n-gram size across the datasets being mostly limited up to 3, as mentioned in Section 3.3, we set the maximum n-gram size to be 3.
# 5.3. PoS-Tag Patterns
As the first step, we applied our PoS-tagging-based post-processing approach to the selected AKE methods, and evaluated on all the datasets. Results show that the proposed approach improved all the methods except SIFRank+ on the average, in terms of precision, recall, and F1-score. While KP-Miner achieved a better performance for 14 of the 17 datasets with an average of 6.08% in F1-score, RaKUn was improved by a cross-dataset average of 4.46% for 14 of the 17 datasets. We observed the most change in
Table 5: An overview of some existing open-source unsupervised AKE methods, showing a number of key characteristics.
Method
Easy to
PoS-tagging
Thesaurus
Wikipedia
Reconfigure
Statistical Methods
KP-Miner (El-Beltagy and Rafea, 2009)
✓
–
–
–
YAKE! (Campos et al., 2020)
✓
–
–
–
LexSpec (Ushio et al., 2021)
✓
✓
–
–
Graph-based Methods
TextRank (Mihalcea and Tarau, 2004)
✓
✓
–
–
SingleRank (Wan and Xiao, 2008)
✓
✓
–
–
RAKE (Rose et al., 2010)
✓
–
–
–
RaKUn (ˇSkrlj et al., 2019)
✓
–
–
–
LexRank (Ushio et al., 2021)
✓
✓
–
–
TFIDFRank (Ushio et al., 2021)
✓
✓
–
–
Embeddings-based Methods
EmbedRank (Bennani-Smires et al., 2018)
✓
✓
–
✓
SIFRank (Sun et al., 2020)
✓
✓
–
✓
SIFRank+ (Sun et al., 2020)
✓
✓
–
✓
MDERank (Zhang et al., 2022)
–
✓
–
✓
the performance of YAKE! – it was improved in 16 of the 17 datasets by a cross-dataset average of 18.05%. We believe this is because YAKE! does not benefit from linguistic features as a more language-independent (multilingual) approach. Finally, we observed a limited improvement in the scores of LexRank (0.84% on average) for 12 of the 17 datasets, and a slight decrease in the performance of SIFRank+, which is likely due to the fact that these two methods already use a PoS-tagging-based filtering. The obtained scores for YAKE! and SIFRank+ are shown in Tables 6 and 7, respectively, as examples. These results provide new evidence for the effectiveness of PoStagging in AKE algorithms, and imply that there is still room to improve the use of PoS-tagging in many AKE methods. Finally, we studied how tailoring the selected PoS-tag patterns according to domain-specific needs may affect the performance of AKE methods. To this end, we considered the example given in Section 4.1, i.e., the observation that gerunds are rarely seen as a keyword in the health domain. We selected the health datasets (i.e., PubMed and Schutz2008) from our collection and applied the tailored PoS-tag-based filtering that disregards gerunds. For this experiment, we used YAKE! since it is more sensitive to linguistic-based improvements as a language-independent algorithm. As shown in Table 8, the tailored filtering approach provided some small improvement to our original
<div style="text-align: center;">Table 6: Comparison of the precision, recall, and F1-score of the original YAKE! and the one utilising PoS-tagging, at 10 extracted keywords</div>
Dataset
YAKE!
YAKE!+PoS
P%
R%
F1%
P%
R%
F1%
KPCrowd
24.20
4.92
8.17
33.98
6.90
11.47
citeulike180
23.11
13.27
16.86
25.68
14.74
18.73
DUC-2001
12.01
14.87
13.29
17.44
21.58
19.29
fao30
22.00
6.83
10.42
25.33
7.86
12.00
fao780
11.93
14.95
13.27
13.18
16.52
14.67
Inspec
19.82
14.05
16.44
24.57
17.41
20.38
KDD
6.01
14.68
8.53
5.83
14.23
8.27
KPTimes
7.97
15.83
10.61
11.37
22.58
15.12
Krapivin2009
9.54
17.88
12.44
9.93
18.61
12.95
Nguyen2007
19.00
15.82
17.26
19.19
15.98
17.43
PubMed
7.28
5.11
6.01
8.66
6.08
7.15
Schutz2008
37.29
8.06
13.26
47.63
10.30
16.93
SemEval2010
20.37
13.08
15.93
20.82
13.37
16.28
SemEval2017
20.61
11.91
15.10
29.41
17.00
21.55
theses100
9.40
14.09
11.28
10.50
15.74
12.60
wiki20
19.50
5.49
8.57
22.00
6.20
9.67
WWW
6.49
13.47
8.76
6.58
13.66
8.88
Avg. Score (%)
16.27
12.02
12.13
19.54
14.04
14.32
Improvement (%)
20.10
16.81
18.05
<div style="text-align: center;">Table 7: Comparison of the precision, recall, and F1-score of the original SIFRank+ and the one utilising PoS-tagging, at 10 extracted keywords</div>
Table 7: Comparison of the precision, recall, and F1-score of the original SIFRan the one utilising PoS-tagging, at 10 extracted keywords
Dataset
SIFRank+
SIFRank+ + PoS
P%
R%
F1%
P%
R%
F1%
KPCrowd
26.08
5.30
8.81
26.20
5.32
8.85
DUC-2001
28.34
35.09
31.36
27.86
34.49
30.82
Inspec
35.68
25.29
29.60
35.10
24.88
29.12
KDD
5.68
13.87
8.06
4.42
10.80
6.28
KPTimes
7.92
15.74
10.54
7.74
15.37
10.30
SemEval2017
41.66
24.08
30.52
40.16
23.21
29.42
WWW
6.59
13.69
8.90
5.26
10.93
7.10
Avg. Score (%)
21.71
19.01
18.26
20.96
17.86
17.41
Improvement (%)
-3.45
-6.05
-4.65
filtering proposal in terms of precision, recall, and F1-score. The limited improvement is likely due to the small percentage of gerunds as candidate keywords.
# 5.4. Context-Aware Thesauri
For this step, we selected 10 datasets mentioned in Section 3.1 that have a particular context. The included contexts (and datasets) are agri-
<div style="text-align: center;">Table 8: Comparison of the precision, recall, and F1-score of YAKE! when the original (PoS) and the tailored (PoS*) filtering approaches are used, at 10 extracted keywords</div>
Dataset
YAKE!+PoS
YAKE!+PoS*
P%
R%
F1%
P%
R%
F1%
PubMed
8.66
6.08
7.15
8.70
6.11
7.18
Schutz2008
47.63
10.30
16.93
47.80
10.34
17.00
Avg. Score (%)
28.15
8.19
12.04
28.25
8.23
12.09
Improvement (%)
0.36
0.49
0.42
culture (fao30 and fao780), health (PubMed) and computer science (Inspec, Krapivin2009, Nguyen2007, SemEval2010, KDD, Wiki20 and WWW). In addition, we constructed another context-specific dataset, KPTimes-Econ, by extracting economy-related news from the KPTimes dataset, which includes 3,258 news articles. For extracting economy-related news articles, we have looked for the records involving the term “economy” in the keyword and/or categories field(s). Based on the 11 datasets, we collected a thesaurus (or something similar, e.g., dictionary, ontology, or wordlist) for each context. More specifically, we used the following thesauri: (i) AGROVOC 2021-07 (Caracciolo et al., 2013) – a multilingual controlled vocabulary constructed by the Food and Agriculture Organization of the United Nations (FAO), with 844,000 agriculture-related terms including 50,163 English ones; (ii) Medical Subject Headings (MeSH) 2021 (Lipscomb, 2000) – a thesaurus covering biomedical and health-related terms produced by the National Library of Medicine (NLM), with over 1.4 million terms in English; (iii) Computer Science Ontology (CSO) v3.3 (Salatino et al., 2018) – a large-scale computer science ontology automatically produced by Klink-2 (Osborne and Motta, 2015) algorithm from 16 million computer science publications, with 14,000 terms; and (iv) STW v9.10 (Kempf and Neubert, 2016) – a bilingual thesaurus (in English and German) for economics produced by the Leibniz Information Center for Economics (ZBW), with over 20,000 terms including 6,217 English ones. For the initial step aiming to experiment with manual integration, we fed each of the baseline methods with each of the datasets and their corresponding thesaurus depending on the context. As in the previous experiment, SIFRank+ was evaluated on only the datasets with shorter documents, i.e., Inspec, KDD, WWW, and KP-Times-Econ in this case. The experiments showed that the manual integration of context-aware thesaurus improved all
five AKE methods in terms of precision, recall, and F1-score significantly for all the datasets. The improvement in F1-score was observed to be 29.03%, 23.88%, 12.85%, 13.19%, and 7.09% for RaKUn, LexRank, YAKE!, KPMiner, and SIFRank+, respectively. Table 9 and Table 10 show more detailed results of the experiment for LexRank and SIFRank+, respectively. The obtained results in this experiment indicates solid evidence of the effectiveness of using context-aware thesaurus to improve the performance of AKE methods.
Table 9: Comparison of precision, recall, and F1-score of the original LexRank and its enhanced versions with manual (M) and automatic (A) thesaurus integration, at 10 extracted keywords
Dataset
Context
LexRank
LexRank+T (M)
LexRank+T (A)
P%
R%
F1%
P%
R%
F1%
P%
R%
F1%
fao30
Agr.
20.33
6.31
9.63
30.33
9.41
14.36
—
—
—
fao780
Agr.
8.55
10.72
9.51
13.04 16.35 14.51
—
—
—
Inspec
CS
30.49 21.61 25.29 31.10 22.04 25.79
30.97
21.95
25.69
KDD
CS
6.07
14.81
8.61
6.23
15.20
8.83
6.25
15.26
8.87
Krapivin2009
CS
7.01
13.14
9.15
8.79
16.48 11.47
8.74
16.37
11.39
Nguyen2007
CS
13.25 11.04 12.04 15.69 13.07 14.26
15.45
12.87
14.04
SemEval2010
CS
13.13
8.43
10.27 15.10
9.70
11.81 15.10
9.70
11.81
wiki20
CS
14.00
3.94
6.15
23.00
6.48
10.11 23.00
6.48
10.11
WWW
CS
6.66
13.83
8.99
6.95
14.43
9.38
6.93
14.40
9.36
PubMed
Health
4.22
2.96
3.48
8.98
6.31
7.41
8.92
6.26
7.36
Schutz2008
Health
28.32
6.12
10.07 34.35
7.43
12.21
34.00
7.35
12.09
KPTimes-Econ
Econ.
3.27
7.03
4.46
4.09
8.80
5.59
4.09
8.79
5.58
Avg. Score (%)
12.94
9.99
9.80
16.47 12.14 12.14
15.35
11.94
11.63
Improvement (%)
27.28
21.52
23.88
21.44
16.03
18.07
Table 10: Comparison of precision, recall, and F1-score of the original SIFRank+ and its enhanced versions with manual (M) and automatic (A) thesaurus integration, at 10
Table 10: Comparison of precision, recall, and F1-score of the original SIFRank+ an its enhanced versions with manual (M) and automatic (A) thesaurus integration, at 1 extracted keywords
Dataset
Context
SIFRank+
SIFRank+ + T (M) SIFRank+ + T (A)
P%
R% F1%
P%
R%
F1%
P%
R%
F1%
Inspec
CS
35.68 25.29 29.60 36.62 25.95
30.37
36.03 25.53
29.88
KDD
CS
5.68 13.87 8.06
5.97
14.58
8.48
5.95 14.52
8.44
WWW
CS
6.59 13.69 8.90
7.32
15.19
9.88
7.27 15.10
9.81
KPTimes-Econ
Econ.
3.49
7.50
4.76
4.56
9.81
6.23
4.56 9.81
6.23
Avg. Score (%)
12.86 15.09 12.83 13.62 16.38
13.74
13.45 16.24
13.59
Improvement (%)
5.91
8.55
7.09
4.59
7.62
5.92
process. In our experiments, especially for datasets covering mainly scientific papers, we built a classifier for classifying a given article’s title and abstract into the main discipline the article belongs to. The classifier was trained on samples extracted from the arXiv.org dataset5 containing metadata of over 1.7M preprints in multiple disciplines. Before the training process, we filtered the arXiv.org dataset by the main discipline reflected by its categories field so as to include the following three disciplines: 1) cs (Computer Science, e.g., cs.AI), 2) bio (Biology, e.g., q-bio), and 3) fin (Finance, e.g., q-fin.CP) and econ (Economics, e.g., econ.EM). After this filtering process, we obtained a dataset of 583,796 samples (551,443 computer science, 20,110 biology, and 12,243 finance/economics samples). Since the resulting dataset is highly imbalanced, we applied random downsampling to equate the number of samples from each discipline to the size of the smallest class, 12,243, which made the final size of our training set 36,729. In our classifier, we utilised the TF-IDF vectoriser for feature extraction. We chose to use the calibrated linear support vector classifier (SVC) with the default parameters and the one-vs-rest setting, rather than a multi-class classification method or more advanced feature extraction methods such as BERT, to show that even a lightweight classifier is sufficient for the task of automatic context detection. The classifier was evaluated with a stratified 5-fold cross-validation. The testing accuracies6 of computer science, biology and finance/economics models were 93.2%, 94.9%, and 97.0%, respectively. The classifier can also be extended to support multiple contexts for a single article, although in our experiments we considered the case of a single context per article for the sake of simplicity and clarity. We used the Scikit-learn library (Pedregosa et al., 2011) to implement all of the mentioned components. Since the training set of the classifier does not cover agriculture preprints, and we were unable to find a proper agriculture dataset for training, we excluded the agriculture context and the corresponding datasets, fao30 and fao780, for this part of the experiments. The results of the experiments performed with our classifier indicated that the automatic thesaurus integration approach achieved as good as the manual integration approach with a negligible performance decrease. More precisely, the F1-score was improved by an
average of 23.23%, 18.07%, 9.60%, 11.27%, and 5.92% for RaKUn, LexRank, YAKE!, KP-Miner, and SIFRank+, respectively, compared to the baseline scores. Table 9 and Table 10 show more detailed results of the experiment for LexRank and SIFRank+, respectively. The obtained results imply that automatic integration can be generalised to cover more contexts and thesauri, which can be quite useful in real-world AKE applications.
# 5.5. Wikipedia Named Entities
For this part of the experiments, we used the entire set of datasets as we did in Section 3.2. The results of the experiment indicated that leveraging Wikipedia named entities improved the performance of KP-Miner and RaKUn for 16 of the datasets, and the performance of YAKE! and LexRank for all the datasets, in terms of all the evaluation metrics. Furthermore, the average improvement rates of the F1-score were observed as 18.83%, 11.11%, 10.96%, and 10.11% for RaKUn, LexRank, YAKE!, and KP-Miner, respectively. However, we observed a slight decrease in the average F1-score of SIFRank+, although it improved for most (5 out of 7) of the datasets, which may be explained by its underlying sentence embedding approach, SIF (Arora et al., 2017), which already leverages Wikipedia for pre-training and fine-tuning. Table 11 and Table 12 show more detailed results for RaKUn and SIFRank+ as examples.
# 5.6. Combining Post-Processing Steps
In the final part of our experiments, we tried combining multiple postprocessing steps to improve the performance further. With this respect, we tried to apply all the combinations of the three proposed enhancements. The generated heatmaps from the F1@10 scores and the percentages of improved cases with different combinations for each baseline method can be seen in Figure 1. The results show that the best F1 scores for YAKE!, RaKUn, and KP-Miner were obtained when all the proposed post-processing steps were applied. For LexRank and SIFRank+, however, the best combination was integrating context-aware thesaurus and Wikipedia since they already benefited from PoS-tagging-based filtering. In addition, the applied postprocessing steps improved the baselines significantly – the improvement rate reached up to 23.7% for YAKE!, 21.3% for KP-Miner, 53.8% for RaKUn, 20.1% for LexRank, and 10.2% for SIFRank+. Finally, the improvements were consistent – at least one combination of the post-processing steps was observed for each method, resulting in a higher performance across all the
<div style="text-align: center;">Table 11: Comparison of precision, recall, and F1-score of the original RaKUn and its enhanced versions with Wikipedia, at 10 extracted keywords</div>
Dataset
RaKUn
RaKUn+Wiki
P%
R%
F1%
P%
R%
F1%
KPCrowd
42.52
8.64
14.36
42.64
8.66
14.40
citeulike180
16.56
9.50
12.08
17.92
10.29
13.07
DUC-2001
5.68
7.03
6.29
6.17
7.64
6.82
fao30
15.00
4.65
7.10
18.67
5.79
8.84
fao780
6.50
8.14
7.23
7.64
9.57
8.50
Inspec
6.54
4.64
5.43
6.74
4.77
5.59
KDD
3.66
8.92
5.19
3.63
8.86
5.15
KPTimes
8.07
16.03
10.74
8.15
16.18
10.84
Krapivin2009
2.77
5.20
3.62
4.94
9.26
6.44
Nguyen2007
6.79
5.66
6.17
9.67
8.05
8.78
PubMed
4.30
3.02
3.55
6.58
4.62
5.43
Schutz2008
33.14
7.16
11.78
40.09
8.67
14.25
SemEval2010
6.75
4.33
5.28
10.04
6.45
7.85
SemEval2017
11.42
6.60
8.37
11.74
6.79
8.60
theses100
3.90
5.85
4.68
4.80
7.20
5.76
wiki20
9.50
2.68
4.18
19.50
5.49
8.57
WWW
4.32
8.98
5.84
4.39
9.12
5.93
Avg. Score (%)
11.02
6.88
7.17
13.14
8.08
8.52
Improvement (%)
19.24
17.44
18.83
<div style="text-align: center;">Table 12: Comparison of the precision, recall, and F1-score of the original SIFRank+ and the one utilising Wikipedia named entities, at 10 extracted keywords</div>
Table 12: Comparison of the precision, recall, and F1-score of the original SIFRa the one utilising Wikipedia named entities, at 10 extracted keywords
Dataset
SIFRank+
SIFRank+ + Wiki
P%
R%
F1%
P%
R%
F1%
KPCrowd
26.08
5.30
8.81
27.46
5.58
9.27
DUC-2001
28.34
35.09
31.36
22.82
28.26
25.25
Inspec
35.68
25.29
29.60
36.60
25.94
30.36
KDD
5.68
13.87
8.06
6.11
14.90
8.66
KPTimes
7.92
15.74
10.54
9.22
18.31
12.26
SemEval2017
41.66
24.08
30.52
41.34
23.89
30.28
WWW
6.59
13.69
8.90
7.50
15.57
10.12
Avg. Score (%)
21.71
19.01
18.26
21.58
18.92
18.03
Improvement (%)
-0.60
-0.47
-1.26
datasets. The results showed that even for more modern AKE methods there is still room for improvement using simple post-processing steps like those proposed in this paper.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b59c/b59c9691-e4ee-4ab5-a14a-303d165f38f6.png" style="width: 50%;"></div>
<div style="text-align: center;">(e) SIFRank+</div>
Figure 1: Average improvements in F1 scores across all the datasets (upper side), an percentages of the improved cases across all the datasets (bottom side), for different AK methods. (B: Baseline, P: PoS-tagging, T: Thesaurus integration, W: Wikipedia integra tion)
# 6. Further Discussions
The proposed post-processing steps in this study were applied to five SOTA AKE methods, showing their universality to work with any AKE methods. PoS-tagging can be easily integrated to AKE methods to implement a filtering mechanism. However, it should be separately considered for each dataset since AKE datasets lack linguistic standards for golden keywords. This can significantly increase the accuracy of the AKE methods
benefiting from PoS-tagging. Thesaurus and Wikipedia integration can also be applied to any AKE method without much effort. Considering that a text document can cover multiple contexts, the results we reported can be further improved by integrating multiple contexts. This can be achieved by utilising a multi-label classifier. Since one-vs-rest classifiers can be used for multi-label classification, our classifier can be refined to cover multiple contexts. In addition, more advanced models, such as BERT, can be utilised to develop a more accurate classifier. It is also worth noting that two of the proposed post-processing steps in this study were selected as representative examples of semantic elements. Other semantic elements can also be used to further improve the performance of AKE methods. Although our experiments on the proposed post-processing steps are based on English NLP tools, they can also be applied to multilingual AKE methods, e.g., YAKE!, for any language. The language of input documents can be identified automatically with a language identifier, which can achieve a high accuracy for many languages (Jauhiainen et al., 2019). Then, the corresponding PoS-tagger and Wikipedia data can be utilised, although the set of acceptable PoS-tag patterns will need updating according to the identified language. Nevertheless, utilising a context-aware thesaurus could be tricky for some languages especially small ones as there might be no thesaurus relevant to the context of the document in the identified language. This study has a number of limitations that can be addressed in future works. Firstly, the selected baseline AKE methods are just examples of SOTA methods so may not be sufficiently representative. As our focus was improving AKE methods in general, we did not aim to achieve the best scores among the studies on AKE. As a result, this study is limited to opensource, unsupervised, and general-purpose AKE methods. In addition, this study leveraged multiple elements for English language, and used English datasets for evaluation. Therefore, it disregarded non-English settings, which are needed especially for multilingual AKE methods, such as YAKE!. Besides, the proposed mechanisms have been applied separately throughout the experiments. Therefore, the results could be improved further if different mechanisms benefit from each other (e.g., applying PoS-tag-based filtering to Wikipedia integration mechanism to disregard the Wikipedia named entities that cannot be a keyword). Finally, a better matching strategy considering word ambiguities can be developed for checking if a candidate keyword appears in a thesaurus or Wikipedia, with the help of techniques such as word sense disambiguation.
# 7. Conclusion
AKE has a more important role in IR and NLP with the increasingly vast amount of digital textual data that modern systems process. In this paper, we aimed to show that an enhanced level of semantic-awareness supported by PoS-tagging can improve AKE algorithms. We selected five algorithms as the baseline methods upon experiments comparing several state-of-the-art AKE methods. Then, we used PoS-tagging, integrated thesauri and Wikipedia named entities for improving the baselines. Our experiments on 17 English datasets indicated that the three proposed mechanisms improved the baseline algorithms significantly and consistently.
# Acknowledgements
Acknowledgements We would like to thank Ricardo Campos for clarification and additional information about the YAKE! algorithm. The first author E. Altuncu was supported by funding from the Ministry of National Education, Republic of Turkey, under the grant number MoNE-YLSY-2018.
# References
Arora, S., Liang, Y., Ma, T., 2017. A simple but tough-to-beat baseline for sentence embeddings, in: Proceedings of the 2017 International Conference on Learning Representations (ICLR ’17), pp. 1–16. URL: https://open review.net/forum?id=SyK00v5xx.
Martinc, M., ˇSkrlj, B., Pollak, S., 2021. TNT-KID: Transformer-based neural tagger for keyword identification. Natural Language Engineering , 1– 40doi:10.1017/S1351324921000127.
Marujo, L., Gershman, A., Carbonell, J., Frederking, R., Neto, J.P., 2013. Supervised topical key phrase extraction of news stories using crowdsourcing, light filtering and co-reference normalization. arXiv:1306.4886 [cs.CL]. doi:10.48550/arXiv.1306.4886.
Salatino, A.A., Thanapalasingam, T., Mannocci, A., Osborne, F., Motta, E., 2018. The Computer Science Ontology: A large-scale taxonomy of research areas, in: The Semantic Web: Proceedings of the 17th International Semantic Web Conference (ISWC ’18), Part II, Springer. pp. 187–205. doi:10.1007/978-3-030-00668-6 12.
Schutz, A.T., 2008. Keyphrase Extraction from Single Documents in the Open Domain Exploiting Linguistic and Statistical Methods. Master’s thesis. National University of Ireland, Galway. URL: https://citeseer x.ist.psu.edu/viewdoc/download?doi=10.1.1.394.5372&rep=rep1&t ype=pdf. Shi, T., Jiao, S., Hou, J., Li, M., 2008. Improving keyphrase extraction using wikipedia semantics, in: Proceedings of the 2nd International Symposium on Intelligent Information Technology Application, IEEE. pp. 42–46. doi:10.1109/IITA.2008.211. Smadja, F., 1993. Retrieving collocations from text: Xtract. Computational Linguistics 19, 143–178. URL: https://aclanthology.org/J93-1007. pdf. Sun, Y., Qiu, H., Zheng, Y., Wang, Z., Zhang, C., 2020. SIFRank: A new baseline for unsupervised keyphrase extraction based on pre-trained language model. IEEE Access 8, 10896–10906. doi:10.1109/ACCESS.202 0.2965087. Ushio, A., Liberatore, F., Camacho-Collados, J., 2021. Back to the basics: A quantitative analysis of statistical and graph-based term weighting schemes for keyword extraction, in: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP ’21), ACL. pp. 8089–8103. doi:10.18653/v1/2021.emnlp-main.638. Wan, X., Xiao, J., 2008. Single document keyphrase extraction using neighborhood knowledge, in: Proceedings of the 23rd National Conference on Artificial Intelligence - Volume 2, AAAI. pp. 855–860. URL: https://www.aaai.org/Papers/AAAI/2008/AAAI08-136.pdf. Wang, Y., Liu, Q., Qin, C., Xu, T., Wang, Y., Chen, E., Xiong, H., 2018. Exploiting topic-based adversarial neural network for cross-domain keyphrase extraction, in: Proceedings of the 2018 IEEE International Conference on Data Mining (ICDM ’18), IEEE. pp. 597–606. doi:10.1109/ICDM.2018. 00075. Witten, I.H., Paynter, G.W., Frank, E., Gutwin, C., Nevill-Manning, C.G.,
