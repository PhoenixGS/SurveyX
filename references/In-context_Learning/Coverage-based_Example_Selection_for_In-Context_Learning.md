Shivanshu Gupta1 Matt Gardner2 Sameer Singh1 1University of California Irvine 2Scaled Cognition {shivag5,sameer}@uci.edu, mgardner@scaledcognition.com
# Abstract
In-context learning (ICL), the ability of large language models to perform novel tasks by conditioning on a prompt with a few task examples, requires these examples to be informative about the test instance. The standard approach of independently ranking and selecting the most similar examples selects redundant examples while omitting important information. In this work, we show that BERTScore-Recall (BSR) selects better examples that demonstrate more of the salient aspects, e.g. reasoning patterns, of the test input. We further extend BSR and many standard metrics to easily optimizable set-level metrics, giving still better coverage of those salient aspects. On 15 datasets spanning 6 tasks and with 7 diverse LLMs, we show that (1) BSR is the superior metric for in-context example selection across the board, and (2) for compositional tasks, set selection using SetBSR outperforms independent ranking by up to 17 points on average and, despite being trainingfree, surpasses methods that leverage task or LLM-specific training.1
arXiv:2305.14907v3
arXiv:2305.14
# 1 Introduction
Large language models (LLMs) (Devlin et al., 2019; Brown et al., 2020) are capable of generalizing to novel tasks (Brown et al., 2020) by conditioning on textual prompts consisting of a few task examples. This training-free paradigm of fewshot inference, known as in-context learning (ICL), reduces the cost of modeling new tasks while also providing an interpretable and customizable interface (Liu et al., 2022; Wei et al., 2023) and improving generalization (Anil et al., 2022; Qiu et al., 2022b; Drozdov et al., 2023) and reasoning skills (Wei et al., 2023). However, ICL performance is critically sensitive to the choice of demonstrations (Zhao et al., 2021; Liu et al., 2022; Lu et al., 2022; Rubin et al., 2022; Schick and Schütze,
1https://github.com/Shivanshu-Gupta/ icl-coverage
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9946/9946d32e-1220-4302-a44e-4d74d7beeb97.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) Coverage-based Set Selection</div>
Figure 1: (a) Test input with salient aspects highlighted. (a) Independently selecting similar examples leads to redundancy and failure to demonstrate all salient aspects, in this case, the need to identify the manager. (b) Coverage-based selection using SET-BSR mitigates this by selecting a less similar example that contains the missing information. Blue indicates LLM generation.
# 2021), as the LLM relies on them for understanding and solving the test instance.
The standard approach to selecting ICL examples or demonstrations from a pool of candidates is to independently score them using a relevance metric and choose the top-ranked ones. However, cosine similarity and BM25, the two commonly used metrics, are sub-optimal for selecting demonstrations due to their reliance on a single dense embedding and unigram overlap, respectively. Moreover, since it selects examples independently, this approach ignores their utility as a set. It is particularly inadequate for complex compositional tasks like semantic parsing (Levy et al., 2022) where no single candidate might contain all reasoning patterns, and an independent selection would select multiple redundant examples with the same reasoning patterns but fail to demonstrate the others. Figure 1 shows a
failure case where similarity-based selection picks paraphrased examples that fail to demonstrate how to find a manager. Prior work on selecting demonstrations as a set (Ye et al., 2023; Levy et al., 2022) required task and/or LLM-specific training, limiting their utility. For this reason, simple yet widely applicable training-free methods like BM25 and cosine similarity remain the most popular approaches for ICL example selection. In this work, we propose a novel framework for selecting sets of maximally informative demonstrations for the salient aspects of the test input, e.g., reasoning patterns, entities, etc. Examples selected using this framework are informative about the test input and help the LLM understand and perform the task. We use this framework to explore different ways to characterize salient aspects, including syntactic structures like dependency parse subtrees and contextual token embeddings, while using BM25 and BERTScore (Zhang et al., 2020) to measure their coverage, respectively. To select the demonstrations as a set, we extend the coverage metrics to measure the overall informativeness of a set of demonstrations. We show that these set-level metrics are submodular and can be efficiently optimized to find demonstration sets that maximally cover the salient aspects. We evaluate our ICL example selection methods on 15 diverse datasets, including 6 semantic parsing, 2 numerical reasoning, and 7 classification datasets, and with 7 LLMs of varying sizes and pretraining. Among instance-level metrics, BSR, the recall version of BERTScore, consistently outperforms standard retrieval metrics on all datasets and LLMs, beating cosine similarity by up to 8 points on average in semantic parsing datasets and 15 points in the rest. Selecting demonstrations as a set using SET-BSR, the set-extension of BSR, leads to further gains in semantic parsing and is particularly effective in compositional settings where the gains grow with LLM size. With Codex, a 175B parameter LLM, SET-BSR outperforms cosine similarity by 17% on average with up to 49% improvement in some splits, and, despite being training-free, outperforms even trained methods like those from Rubin et al. (2022), Levy et al. (2022), and Ye et al. (2023) that require task and/or LLM-specific training.
# 2 Related Work
In-context learning for few-shot inference facilitates the use of LLMs for novel tasks without the
need for expensive supervised fine-tuning. In addition to reduced cost, it has several other advantages over supervised fine-tuning: it provides a more interpretable and customizable interface to using LLMs (Liu et al., 2022; Wei et al., 2023); and retention of linguistic understanding and knowledge from pretraining leading to improved generalization (Anil et al., 2022; Qiu et al., 2022b; Drozdov et al., 2023) and reasoning skills (Wei et al., 2023). However, the performance of ICL is critically sensitive to the choice of demonstrations (Zhao et al., 2021; Liu et al., 2022). This has led to a growing interest in techniques for selecting good demonstrations. Prior work can be roughly classified into (1) independently scoring and retrieving examples (Liu et al., 2022; Rubin et al., 2022), (2) selecting diverse examples to reduce redundancy among them (Su et al., 2022; Levy et al., 2022; Agrawal et al., 2022; Ye et al., 2022), and (3) selecting examples that minimize the entropy of the LLM’s output distribution for the test input (Lu et al., 2022; Wu et al., 2023). Recent work has also trained RL agents (Lu et al., 2023) and used Bayesian inference (Wang et al., 2023). The most similar studies to ours are Levy et al. (2022) and Ye et al. (2023). Levy et al. (2022) select diverse demonstrations that cover substructures of the target output predicted by task-specific classifiers but are limited in applicability to a few semantic parsing tasks. Ye et al. (2023) use Determinantal Point Processes (Kulesza, 2012) to select a diverse set of demonstrations similar to the test instance but do not optimize for coverage directly and require training with the LLM. Moreover, both methods require task or LLM-specific training that limits their use and effectiveness for larger LMs.
# 3 Preliminaries
In-context learning is the ability of LLMs to solve novel tasks by merely conditioning on a few task demonstrations. Formally, given demonstrations {(xi, yi)}k i=1 and the test input xtest, it involves using textual templates to linearize instance inputs and outputs into sequences of tokens from the LLM vocabulary, x = I(x) = ⟨x1 . . . x|x|⟩and y = O(y) = ⟨y1 . . . y|y|⟩. The linearizations are then concatenated to form a prompt and fed to the LLM for conditional generation of the test output:
ytest ∼PLM (· | x1, y1, . . . , xK, yK, xtest ) 
The interpretable and training-free nature of ICL makes it an attractive alternative to supervised finetuning. However, its performance is highly sensitive to the choice and order of demonstrations.
Demonstration Selection identifies which examples to include in the prompt for any test instance. Formally, given a test input xtest and a pool of candidates T = {zi}N i=1 = {(xi, yi)}N i=1, the goal is to select a subset of k ≪N demonstrations that when included in the context make ytest the most likely generation. A naive approach is to randomly sample k instances from T , but this is sub-optimal since the demonstrations are often completely unrelated to the test input. Instead, the standard approach to selecting demonstrations that are informative about the test input is to independently assign each candidate z a score score(xtest, z) using a relevance metric and then select the top k candidates. Relevance Metrics The two most commonly used relevance metrics for scoring demonstration are cosine similarity and BM25. Cosine similarity uses a representation function R to independently map the textual linearizations of inputs to unit-norm embeddings rx = R(x) in a common vector space and then scores the candidate z using the dot product, cosine(xtest, z) = rT xtestrz. BM25, on the other hand, is a sparse information retrieval algorithm belonging to a class of TF-IDF measures that view the test input and the candidates as bags of terms and measures relevance as a weighted recall or coverage of these terms:
tfidf(xtest, z) = � s∈Txtest idf(s)tf(s, Tz) (2)
(2)
Here Tx and Tz are the set of terms in x and z respectively, and tf(s, Tz) and idf(s) are the term frequency and inverse document frequency statistics that measure the coverage of a particular term and the relative importance of terms respectively. We use tf and idf as per the Okapi variant of BM25 (Robertson et al., 1993; Jones et al., 2000).
# 4 Informative Demonstrations
The limitation of the standard demonstration selection approach is that by independently scoring the demonstrations, it ignores their utility as a set. For ICL to work, the demonstrations included in the context need to be informative about how to understand and solve the test input. In this section
and the next, we describe our approach to selecting informative sets of demonstrations for ICL. We begin by defining our notion of informativeness of demonstrations in ICL and describing how to measure it. Thereafter, in §5, we will discuss how to extend this notion to an algorithm for selecting optimally informative sets of demonstrations. Informativeness Demonstrations should demonstrate the salient aspects, e.g., reasoning patterns, entities, etc., of the test input. Formally, denoting Sxtest as the set of salient aspects of the test input, we measure the informativeness of a demonstration z in terms of the coverage of such salient aspects,
where c(s, z) measures the coverage (or recall) of a single salient aspect s by z. Salient Aspects Both cosine similarity and BM25 are special cases of Eq. 3 for different notions of salient aspects. For BM25, Sxtest = Txtest, the set of unigrams in x, and c(s, z) = idf(s)tf(s, Tz). And cosine similarity, although not explicitly a recall metric, can also be interpreted as evaluating coverage of the dimensions of the test input embedding by defining Sxtest = [1, d], the dimensions of the dense embedding as the salient aspects, i.e.,
(4)
The above interpretations reveal why neither cosine similarity nor BM25 are good measures of informativeness. While cosine similarity captures some aspects of semantic similarity (depending on the embedding), it is limited to a single embedding. And, unigrams, the commonly used terms with BM25, are too small to capture most salient aspects. A good measure of informativeness necessitates an accurate characterization of salient aspects. One way might be to use larger syntactic substructures of the input as terms with BM25. We experiment with using larger n-grams and subtrees of the dependency parse tree. However, such syntactic structures are constrained to the surface form of the instance and hence may not capture meaning and aspects like reasoning patterns. A better way to capture salient aspects is to use contextualized token embeddings, the idea behind the BERTScore (Zhang et al., 2020) metric.
BERTScore was originally proposed as a metric for evaluating the quality of machine-generated text (e.g., machine translation) by comparing it to a reference text. It leverages pre-trained contextual embeddings to match words in the candidate and reference sentences by cosine similarity and compute precision, recall, and F1 measures. Formally, given the sequences of contextual embeddings ⟨x1, x2, . . . , x|x|⟩and ⟨z1, z2, . . . , z|z|⟩ of tokens in x = ⟨x1, x2, . . . , x|x|⟩and z = ⟨z1, z2, . . . , z|z|⟩respectively, the recall measure, BERTScore-Recall (BSR), is defined as:
(5)
Here, w(xi) is a weight assigned to token xi and can be defined as 1 |x| if treating each token as equally important or idf(xi) � xi∈x idf(xi) if downweighting rare words. The precision measure is defined analogously, while the F1 measure is the harmonic mean of the two. BSR is also a special case of Eq. 3 with contextualized tokens as salient aspects, i.e., Sx = ⟨x1, x2, . . . , x|x|⟩and can be used to select examples by treating them as candidates and the test input as the reference. The following table summarizes the informativeness measures and salient aspects in this work.
Metric
Salient Aspects
Cosine
embedding dimensions
BM25
unigrams, n-grams, dependency parse subtrees
BERTScore contextual token embeddings
# 5 Set-level Information Coverage
So far, we have focused on measuring the informativeness of a single demonstration to rank and independently select the most informative ones. However, as depicted in Fig. 1, when no single single candidate demonstrates all salient aspects, this approach can fail to cover all of them while also selecting redundant demonstrations that provide no new information. A scenario where this can happen is when the candidate pool contains close paraphrases (or duplicates). This suggests that demonstrations should be selected as a set. Set Metric To evaluate the informativeness of a set of examples Z, we propose to extend the coverage measure in Eq. 3 to a measure for sets as follows:
(6)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/74b3/74b36a16-1d2d-4583-86e5-f3ff6e5e9098.png" style="width: 50%;"></div>
Algorithm 1 Greedy Optimization of Set Coverage
Require: Instance pool T ; test input xtest; desired number of
demonstrations k; coverage scoring function setcov
1: Z ←∅
▷Selected Demonstrations
2: Zcurr ←∅
▷Current Set Cover
3: curr_cov ←−inf
4: while |Z|< k do
5:
z∗, next_cov = argmax
z∈T −Z
setcov (xtest , Zcurr ∪z)
6:
if next_cov > curr_cov then
▷Pick z∗
7:
curr_cov ←next_cov
8:
Z ←Z ∪z∗
9:
Zcurr ←Zcurr ∪z∗
10:
else
▷Or start new cover
11:
Zcurr ←∅, curr_cov ←−inf
12:
end if
13: end while
14: return Z
Intuitively, this measures the coverage of each salient aspect as the best coverage it receives from any example in the set. In other words, maximizing it requires that every salient aspect appears at least once in some demonstration without considering which or how many. Since cosine similarity, BM25, and BSR are all special cases of Eq. 3, they can be extended to set measures using Eq. 6. Submodularity Given the combinatorial space of sets of demonstrations, for a measure on sets to be practical, it needs to be efficiently optimizable. Fortunately, the set-level metric, as defined above, is also submodular for any definition of c(s, z). We prove this in Appendix A. Intuitively, this follows from the facts that (1) for any given test instance, c(s, z) assigns a scalar weight to each demonstration z ∈Z, (2) the maximum of weights across set elements is submodular, and (3) the sum of submodular functions is also submodular. This means that the set-level metric can be optimized using a greedy algorithm with a constant factor approximation guarantee (Nemhauser et al., 1978). Algorithm The greedy algorithm we use to select the optimal set is shown in Algorithm 1. In every iteration, it selects the example that maximally increases the coverage of the current set of demonstrations (lines 5-9). If no such example exists, it resets (lines 11). Using the following identity when computing the score for candidate sets (line 5),
(7)
and assuming constant time for computing each c(s, z), the time complexity of algorithm is
O(kNL), where L = |Sxtest|. For BSR, the complexity of computing c(x, z) for all z ∈Z is O(Td), where T is the total number of tokens in Z and d is the token embedding size. Thus, the time complexity of both instance and set-level BSR is dominated by the computation of c(x, z), and is O(LTd). While slower than cosine and BM25, we found it to be a small overhead to in-context learning for most datasets considered in this work. We discuss this further in App. C.
# 6 Experimental Setup
# 6.1 Datasets
We experiment with a total of 15 datasets including six diverse semantic parsing datasets viz. GeoQuery (Zelle and Mooney, 1996), ATIS (Hemphill et al., 1990; Dahl et al., 1994), Overnight (Wang et al., 2015), SMCalFlow (Andreas et al., 2020), BREAK (Wolfson et al., 2020), and MTOP (Li et al., 2021); a math-word problems (GSM8K (Cobbe et al., 2021)) and a machine reading comprehension (DROP (Dua et al., 2019)) dataset requiring multi-step numeric reasoning; and seven classification datasets spanning natural language inference, paraphrase detection and sentiment classification viz. QNLI (Wang et al., 2018), MNLI (Williams et al., 2018), RTE (Bentivogli et al., 2009), MRPC (Dolan and Brockett, 2005), PAWS (Zhang et al., 2019), QQP (Wang et al., 2018), and SST2 (Socher et al., 2013). We refer the reader to App. B for detailed descriptions of each dataset along with sample instances and prompt templates. In addition to the standard IID splits, we also evaluate compositional generalization using compositional splits wherever available. For GeoQuery we use three types of compositional splits: Template (Finegan-Dollak et al., 2018), TMCD (Keysers et al., 2020), and Length. Following Levy et al. (2022), we use the compositional splits— three Template, three TMCD, and one Length— generated by Qiu et al. (2022a) and average results across the TMCD and Template splits. For ATIS and Overnight, we experiment with Template splits (Finegan-Dollak et al., 2018) generated by Gupta et al. (2022). For SMCalFlow, we experiment with splits in SMCalFlow-CS (Yin et al., 2021): an IID split (8-S) and a compositional split (32-C). For all the splits, following prior work (Ye et al., 2023; Rubin et al., 2022) we randomly subsample 44,000 instances from the train set to use as pool to select demonstrations from. For evaluation, we
use a random subsample of 1000 instance of the validation set if available, and the test set otherwise. We use Exact Match (EM) accuracy for all datasets except BREAK where we use LF-EM (Hasson and Berant, 2021), which is preferred over EM for semantic equivalence.
We experiment with the following LLMs: GPTNeo-2.7B (Black et al., 2021): A 2.7B-parameter LM trained on The Pile (Gao et al., 2020), an 825 GB text corpus. LLaMA (Touvron et al., 2023): A collection of LMs ranging from 7B to 65B parameters pretrained on CommonCrawl, GitHub, Arxiv, etc. We experiment with LLaMA7B and LLaMA-13B. StarCoder (Li et al., 2023): A 15.5B parameter model trained on 80+ programming languages (Kocetkov et al., 2022). GPT3.5-Turbo2: 175B LM trained with RL to follow instructions and optimized for chat. Cushman, Codex3 (Chen et al., 2021): 12B and 175B parameter code-pretrained LMs. GPT-Neo-2.7B, LLaMA-7B, LLaMA-13B, and Cushman have context window lengths of 2048, GPT-3.5-Turbo of 4096, Codex of 8001, and StarCoder of 8192.
# 6.3 Methods
We compare the following training-free metrics: Cosine similarity (COSINE) We use the SentenceBert library (Reimers and Gurevych, 2019) with the all-mpnet-base-v2 model. For independent selection, we use FAISS 4 (Johnson et al., 2019) retrieve the most similar examples. BM25 (BM25) We use the Okapi variant (Robertson et al., 1993; Jones et al., 2000) of BM25 from the rank_bm255 library with three syntactic structures as terms: unigrams, size-4 or smaller n-grams, and size-4 or smaller subtrees of the input dependency parse (obtained using the spaCy6). BERTScore We use the bert_score7 library (Zhang et al., 2020) with deberta-large-mnli and deberta-base-mnli models which are DeBERTa models (He et al., 2021) finetuned on the MNLI dataset (Williams et al., 2018). We will refer
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/99a8/99a8c360-a280-4dee-8ad9-81a92a344846.png" style="width: 50%;"></div>
Selector
GPT-Neo
LLaMA-7B LLaMA-13B
Cushman
StarCoder
GPT-3.5-Turbo
Codex
Training
Free
RANDOM
5.5 (-23.3)
5.7 (-28.7)
9.8 (-28.9)
12.0 (-32.9)
13.6 (-33.5)
13.0 (-31.9)
20.7 (-32.6)
COSINE
28.8
34.4
38.7
44.9
47.1
44.9
53.4
BM25
31.2 (+2.4)
36.7 (+2.3)
42.8 (+4.0)
49.7 (+4.8)
52.9 (+5.7)
50.3 (+5.4)
60.9 (+7.5)
Trained
EPR
38.3 (+9.5)
43.7 (+9.3)
48.1 (+9.4)
51.8 (+7.0)
53.5 (+6.4)
47.4 (+2.5)
58.5 (+5.1)
CEIL
38.1 (+9.3)
44.5 (+10.1)
49.9 (+11.2)
54.8 (+9.9)
57.3 (+10.2)
51.2 (+6.3)
64.0 (+10.7)
Ours
BSR
34.1 (+5.3)
40.1 (+5.8)
46.5 (+7.8)
52.6 (+7.7)
54.8 (+7.7)
52.7 (+7.8)
61.2 (+7.9)
SET-BSR
35.8 (+7.0)
43.8 (+9.4)
51.4 (+12.7)
59.5 (+14.6) 61.6 (+14.5)
60.1 (+15.2)
70.3 (+16.9)
Table 1: Average 8-shot ICL performance across all splits of semantic parsing datasets using different LLMs and demonstration-selection methods with absolute improvement over COSINE in brackets. Both BSR and SET-BSR outperform prior training-free methods, with the latter outperforming even trained methods with larger LLMs.
to the recall, precision, and F1 variants as BSR, BSP, and BSF1, respectively. Unless specified otherwise, we do not apply importance weighting (IDF) and use deberta-large-mnli. Additionally, we experiment with (1) a random baseline (RANDOM) that randomly selects demonstrations from the pool, and (2) with the set-extensions of COSINE, BM25 and BSR as described in §5 which will be referred to as SETCOSINE, SET-BM25, and SET-BSR respectively.
# 6.3.2 Trained Methods
We also compare with methods that require task or LLM-specific training. EPR (Rubin et al., 2022) uses LLM perplexity to train a dense retriever for each dataset. CEIL (Ye et al., 2023) uses EPR and an LLM to train a Determinantal Point Process (Kulesza, 2012) for each dataset and then uses it to select examples. We use Ye et al. (2023)’s implementation of EPR and CEIL and use GPTNeo-2.7B LLM. We also compare with LFCOV (Levy et al., 2022), a method for semantic parsing, specifically SMCalFlow-CS and GeoQuery. It trains a classifier to predict logical form substructures and then selects diverse examples containing them. We use the shots provided by the authors.
# 6.4 Prompt Construction
For k-shot (we use k = 8 unless specified otherwise) ICL with any given dataset (§ 6.1), demonstration selection method (§ 6.3) and LLM (§ 6.2), we construct the prompt as follows: (1) select up to k demonstrations depending on the context window of the LLM; (2) order the demonstrations in increasing order of relevance so that the most relevant demonstrations appear closest to the test input; and (3) linearize the ordered demonstrations and the test input using the dataset’s prompt template in Table 5 and concatenate to form the prompt. For set-selection methods, the demonstrations are or-
Selector
8_S
32_C
Training
Free
RANDOM
31.9 (-22.8)
7.4 (-4.5)
COSINE
54.7
11.9
BM25
65.4 (+10.7) 29.4 (+17.5)
Trained
EPR
76.3 (+21.6)
21.7 (+9.8)
CEIL
77.5 (+22.8) 40.1 (+28.2)
LFCOV
66.3 (+11.6) 45.9 (+33.9)
Ours
BSR
72.5 (+17.8) 31.5 (+19.6)
SET-BSR
75.7 (+21.0) 61.2 (+49.3)
Table 2: 8-shot ICL accuracy on SMCalFlow-CS using Codex with absolute improvement over COSINE in brackets. SET-BSR is competitive with trained methods on the IID split while dramatically outperforming them on the compositional split.
dered by their corresponding instance-level score. For the trained baselines, we use orderings recommended by the corresponding authors.
# 7 Results
We begin by comparing the performance of our proposed methods, BSR and SET-BSR, with prior training-free and state-of-the-art trained methods in § 7.1. We then analyze the different metrics for measuring informativeness of individual demonstrations (§ 7.2) and the impact of coverage-based set selection using our set extension (§ 7.3).
# 7.1 Main Results
Table 1 compares average performance across all semantic parsing splits for seven LLMs of varying sizes. See Table 2 for comparison with LFCOV, which only works with GeoQuery and SMCalFlowCS and Table 11 for results on individual splits. While BSR consistently outperforms COSINE and BM25 for all LLMs, set-selection using SET-BSR leads to further dramatic gains with upto 17% improvement over COSINE with Codex, beating even state-of-the-art trained methods like EPR and CEIL by 12 and 6 points, respectively. Further, from
<div style="text-align: center;">Selector GSM8K DROP MNLI PAWS SST2</div>
Training
Free
Random
60.6
62.7
41.9
48
86.9
Cosine
64
65.4
44.0
52.5
81.9
BM25
64.8
66.9
42.2
55.2
82.6
Trained
EPR
61.7
-
66.1†
-
-
CEIL
63.1
-
71.7†
-
-
Ours
BSR
68.1
68.1
76.7
75
90.9
Set-BSR
67.4
66.4
78.6
74.9
61.5
Table 3: 8-shot ICL performance for tasks other than semantic parsing (using GPT-Neo-2.7B for the classification tasks and Codex for the harder GSM8K and DROP). BSR is competitive with prior methods, however, as these are IID splits, SET-BSR doesn’t lead to further gains. † 50-shot results from Ye et al. (2023).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c1cc/c1cce5bf-713e-441d-9475-0a358ade1d80.png" style="width: 50%;"></div>
Figure 2: Gain in average ICL accuracy compared to COSINE on IID and COMPositional splits in semantic parsing. Trained methods (EPR and CEIL) become less effective with larger LLMs on IID splits. This is unlike SET-BSR, which, on compositional splits, even becomes more effective with larger LLMs. Table 3, we see that, unlike SET-BSR, BSR is effective even for non-semantic parsing datasets outperforming COSINE by 15 points on average with GPT-Neo-2.7B (see Table 12), and often even EPR and CEIL (see Table 13). All the above improvements were statistically significant (p < 0.05) under paired permutation-tests. SET-BSR is more effective with larger LLMs The effectiveness of SET-BSR monotonically improves as LLMs become more powerful. The trend is particularly pronounced in compositional splits, where it gets 25% absolute improvement v/s COSINE on average (see Fig. 2) and 49% improvement on the 32-C split of SMCalFlow-CS (see Table 2). Trained methods do not leverage larger LLMs As EPR and CEIL are trained using GPT-Neo-2.7B, they have difficulty generalizing to and taking ad-
Selector
ALL
IID
COMP
BSF1
60.6 71.0
50.1
BSP
54.3 65.5
43.2
BSR
61.2 71.5
50.9
BM25
60.9 68.9
52.8
+ Coverage
56.4 63.4
49.5
BM25[4-gram]
59.1 67.1
51.0
+ Coverage
64.5 68.9
60.2
BM25[4-depst]
57.8 65.5
50.0
+ Coverage
64.9 68.6
61.2
Table 4: Average 8-shot ICL performance with Codex on IID, COMPositional, and ALL semantic parsing splits. Top compares different variants of BERTScore, white Bottom compares the different variants of BM25. vantage of larger, more powerful LLMs, becoming less effective on IID splits (Fig. 2), and failing on GSM8K (Table 3). The latter is likely because GPTNeo-2.7B itself fails on GSM8K (Table 13), which requires Chain-of-Thought reasoning, an emergent ability of larger LLMs (Wei et al., 2022). As training with increasingly large LLMs is prohibitively expensive and impractical, these results demonstrate serious limitations of trained methods.
# 7.2 Measure of Informativeness
Contextual embeddings capture salient aspects From Tables 1 and 3, it is clear that BSR consistently outperforms COSINE and BM25. This is true even when using the same encoder (see App. D), is seen in both IID and compositional splits (see Fig. 2), and with varying number of demonstrations (see Fig. 4). Larger syntactic substructures did not improve BM25 as seen in Table 4 (Bottom). These results show that contextual embeddings are indeed better at capturing salient aspects.
# Recall outperforms other measures
Recall outperforms other measures Comparing the variants of BERTScore, for Codex in Table 4 (Top), and other LLMs in Fig. 7 in App. D, it is evident that recall is on par with, or better than, the F1 metric. This supports our hypothesis that recall or coverage (of salient aspects) is a useful metric for informativeness. We include additional ablations in App. D, analyzing the effect of using importance weighting (IDF) and using a larger LM to compute token embeddings for BSR.
# 7.3 Coverage-based Set Selection
Impact on performance From Fig. 3, we see that coverage-based set selection is most effective in compositional splits where it improves the average performance of all metrics, including COSINE.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/56b1/56b14602-5d42-4ce4-ac4a-fa439534199a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Change in average performance on different types of splits of semantic parsing datasets from set-selection using our set metrics v/s the corresponding instance-level metric. Coverage-based set selection is most useful in compositional splits and when covering larger syntactic structures (BM25) or contextual embeddings (BSR).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/02be/02bef1cb-6aef-4bd6-9dac-9f14a13c6751.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4019/40198dc6-818a-4826-86fd-3e23c151840d.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Average performance on IID and COMP semantic parsing splits with Codex. SET-BSR consistently outperforms independent selection.</div>
This shows the importance of selecting demonstrations as a set in compositional settings where examples demonstrating all the salient aspects of the test input are even less likely to exist. The set extension is less effective in IID splits and even hurts performance for COSINE and vanilla unigram BM25. Overall, BSR and BM25 with larger substructures benefit the most from the set extension. We provided further analyses of improvements from set selection and the impact of reordering in App. D. Illustrative Example We present a GeoQuery test input in Fig 5 along with demonstrations (only the inputs) selected by COSINE and SET-BSR (more examples in Appendix E). COSINE selections tend to be redundant, with repeated operations, and are somewhat restrictive, mostly limited to the min operation. Contrastingly, SET-BSR exhibits a more balanced selection, opting for demonstrations of comparable complexity to the test instance and collectively encapsulating all necessary operations. Failure Cases There are a few limitations of coverage-based set-selection using SET-BSR. First, by only considering uncovered aspects, it sacrifices
Figure 5: Demonstrations selected for a GeoQuery input (outputs omitted for clarity). COSINE demonstrations are redundant (repeated operations) and limited (only cover “population” aspect). SET-BSR, instead, selects demonstrations that are similarly complex as the test instance and, together, cover all required operations. the relevance of individual demonstrations to prioritize coverage of all aspects with the set (see Table 9 for an example from GSM8K). Additionally, even contextual token embeddings can only capture salient aspects that are explicitly expressed in the input text and thus may not be suitable for tasks where the salient aspects are more abstract and require reasoning themselves (see Table 10 for an example from QNLI). We leave it to future work to explore better measures of informativeness, including better characterizations of salient aspects.
# 8 Conclusion
This paper presents a novel framework for selecting informative sets of demonstrations that cover salient aspects of the test input to aid the language model (LLM) in solving it. We explore different ways to characterize these aspects and quantify their coverage. Evaluation on a wide
range of tasks and LLMs validates the effectiveness of BERTScore-Recall as a measure of informativeness of individual demonstrations. Further, our results demonstrate the superiority of SET-BSR in selecting informative sets of demonstrations compositional tasks like semantic parsing and highlight the ability of coverage-based demonstration selection, unlike trained methods, to leverage increasingly powerful larger LLMs. Our code base is available at https://github.com/ Shivanshu-Gupta/icl-coverage.
# Acknowledgements
We would like to thank the anonymous reviewers for their feedback. This work was sponsored in part by the DARPA MCS program under Contract No. N660011924033 with the United States Office Of Naval Research and in part by the NSF award #IIS2046873. The views expressed are those of the authors and do not reflect the policy of the funding agencies.
Contextual token embeddings require the salient aspects to be expressed in text and hence may not be able to capture them for all tasks. Moreover, since it requires computing a dot product for every pair of test and candidate instance tokens, this causes it to scale quadratically with the average number of tokens making it computationally infeasible for tasks with very long textual linearizations. Future work can thus explore more general characterizations of salient aspects and more efficient methods for selecting demonstrations covering them.
# References
Sweta Agrawal, Chunting Zhou, Mike Lewis, Luke Zettlemoyer, and Marjan Ghazvininejad. 2022. Incontext examples selection for machine translation.
cob Andreas, John Bufe, David Burkett, Charles Chen, Josh Clausman, Jean Crawford, Kate Crim, Jordan DeLoach, Leah Dorner, Jason Eisner, Hao Fang, Alan Guo, David Hall, Kristin Hayes, Kellie Hill, Diana Ho, Wendy Iwaszuk, Smriti Jha, Dan Klein, Jayant Krishnamurthy, Theo Lanman, Percy Liang, Christopher H. Lin, Ilya Lintsbakh, Andy McGovern, Aleksandr Nisnevich, Adam Pauls, Dmitrij Petters, Brent Read, Dan Roth, Subhro Roy, Jesse Rusak, Beth Short, Div Slomin, Ben Snyder, Stephon Striplin, Yu Su, Zachary Tellman, Sam Thomson, Andrei Vorobev, Izabela Witoszko, Jason Wolfe, Abby Wray, Yuchen Zhang, and Alexander Zotov. 2020.
om B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems.
Deborah A. Dahl, Madeleine Bates, Michael Brown, William Fisher, Kate Hunicke-Smith, David Pallett, Christine Pao, Alexander Rudnicky, and Elizabeth Shriberg. 1994. Expanding the scope of the ATIS task: The ATIS-3 corpus. In Human Language Technology: Proceedings of a Workshop held at Plainsboro, New Jersey, March 8-11, 1994. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. William B. Dolan and Chris Brockett. 2005. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005). Andrew Drozdov, Nathanael Schärli, Ekin Akyürek, Nathan Scales, Xinying Song, Xinyun Chen, Olivier Bousquet, and Denny Zhou. 2023. Compositional semantic parsing with large language models. In The Eleventh International Conference on Learning Representations. Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. 2019. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2368–2378, Minneapolis, Minnesota. Association for Computational Linguistics. Catherine Finegan-Dollak, Jonathan K. Kummerfeld, Li Zhang, Karthik Ramanathan, Sesh Sadasivam, Rui Zhang, and Dragomir Radev. 2018. Improving textto-SQL evaluation methodology. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 351–360, Melbourne, Australia. Association for Computational Linguistics. Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2020. The pile: An 800gb dataset of diverse text for language modeling. Shivanshu Gupta, Sameer Singh, and Matt Gardner. 2022. Structurally diverse sampling for sampleefficient training and comprehensive evaluation. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 4966–4979, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. Matan Hasson and Jonathan Berant. 2021. Question decomposition with dependency graphs.
Matan Hasson and Jonathan Berant. 2021. Question decomposition with dependency graphs.
Karen Spärck Jones, Steve Walker, and Stephen E. Robertson. 2000. A probabilistic model of information retrieval: development and comparative experiments - part 2. Inf. Process. Manag., 36:809–840.
Haoran Li, Abhinav Arora, Shuohui Chen, Anchit Gupta, Sonal Gupta, and Yashar Mehdad. 2021. MTOP: A comprehensive multilingual task-oriented semantic parsing benchmark. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 2950–2962, Online. Association for Computational Linguistics.
aymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo
Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo
Wang, Rudra Murthy, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. 2023. Starcoder: may the source be with you!
Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. Fantastically ordered prompts and where to find them: Overcoming fewshot prompt order sensitivity. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland. Association for Computational Linguistics.
Linlu Qiu, Peter Shaw, Panupong Pasupat, Tianze Shi, Jonathan Herzig, Emily Pitler, Fei Sha, and Kristina Toutanova. 2022b. Evaluating the impact of model scale for compositional generalization in semantic parsing. In Proceedings of the 2022 Conference on
Empirical Methods in Natural Language Processing, pages 9157–9179, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.
Nils Reimers and Iryna Gurevych. 2019. SentenceBERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.
Stephen Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford. 1993. Okapi at trec. 500207, pages 109–123. National Institute of Standards and Technology.
Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2022. Learning to retrieve prompts for in-context learning. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2655–2671, Seattle, United States. Association for Computational Linguistics.
Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, and William Yang Wang. 2023. Large language models are implicitly topic models: Explaining and finding good demonstrations for in-context learning. Yushi Wang, Jonathan Berant, and Percy Liang. 2015. Building a semantic parser overnight. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1332–1342, Beijing, China. Association for Computational Linguistics. Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022. Emergent abilities of large language models. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models. Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1112–1122, New Orleans, Louisiana. Association for Computational Linguistics. Tomer Wolfson, Mor Geva, Ankit Gupta, Matt Gardner, Yoav Goldberg, Daniel Deutch, and Jonathan Berant. 2020. Break it down: A question understanding benchmark. Transactions of the Association for Computational Linguistics, 8:183–198. Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, and Lingpeng Kong. 2023. Self-adaptive in-context learning: An information compression perspective for incontext example selection and ordering. Jiacheng Ye, Zhiyong Wu, Jiangtao Feng, Tao Yu, and Lingpeng Kong. 2023. Compositional exemplars for in-context learning. Xi Ye, Srinivasan Iyer, Asli Celikyilmaz, Ves Stoyanov, Greg Durrett, and Ramakanth Pasunuru. 2022. Complementary explanations for effective in-context learning. Pengcheng Yin, Hao Fang, Graham Neubig, Adam Pauls, Emmanouil Antonios Platanios, Yu Su, Sam Thomson, and Jacob Andreas. 2021. Compositional generalization for neural semantic parsing via spanlevel supervised attention. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2810–2823, Online. Association for Computational Linguistics.
John M. Zelle and Raymond J. Mooney. 1996. Learning to parse database queries using inductive logic programming. In Proceedings of the Thirteenth National Conference on Artificial Intelligence - Volume 2, AAAI’96, pages 1050–1055. AAAI Press. Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with BERT. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net. Yuan Zhang, Jason Baldridge, and Luheng He. 2019. PAWS: Paraphrase adversaries from word scrambling. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1298–1308, Minneapolis, Minnesota. Association for Computational Linguistics. Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 12697–12706. PMLR.
# A Submodularity
Definition A.1 (Submodular Function). If Ωis a finite set, a submodular function is a set function f : 2Ω→R, where 2Ωdenotes the power set of Ω, which satisfies one of the following equivalent conditions.
Theorem A.1. The function fmaxw (X) = max x∈X wx is submodular for any assignment of weights wx to the elements x ∈Ω.
Adding these two inequalities together, we get the third definition of submodularity and thus fmaxw is submodular.
Theorem A.2. If {fi}n i=1 are all submodular functions, then n� i=1 fi is also submodular.
� Proof. We show this for n = 2:
Therefore, f1 + f2 is submodular using the second definition of submodularity. By induction, this is true for any number n of functions.
Theorem A.3. The set-level coverage metric setcov (xtest , Z) as defined in Eq. 6 is submodular for any definition of c(s, z).
Proof. From Theorem A.1, the function fs(Z) defined as fs(Z) = max z∈Z c(s, z) is submodular for any definition of c(s, z). Further, since from Theorem A.2, the sum of submodular functions is also submodular, setcov (xtest , Z) = � s∈S fs(Z) is
# B Datasets
We use 15 diverse datasets, including 6 semantic parsing, 2 numerical reasoning, and 7 classification datasets.
# B.1 Semantic Parsing
We use 6 semantic parsing datasets with IID and compositional splits for our experiments. Table 5 shows sample instances from each dataset we experiment with along with the textual template we use to linearize the instances. The ICL prompt is constructed by concatenating the templatized demonstrations and the test instance using \n\n as the separator. GeoQuery (Zelle and Mooney, 1996): A dataset containing 880 natural language questions about US geography paired with Prolog programs. In addition to the standard (IID) split, we experiment with three types of compositional splits: (1) Template split where the training and test sets have disjoint program templates (Finegan-Dollak et al., 2018); (2) TMCD split which creates train and test sets with maximal compound divergence and minimal atom divergence (Keysers et al., 2020); and (3) Length split which evaluates for length generalization by testing on sequences longer than ones in training. Following Levy et al. (2022), we use the compositional splits — three Template, three TMCD, and one Length — generated by Qiu et al. (2022a) and average results across the TMCD and Template splits. ATIS (Hemphill et al., 1990; Dahl et al., 1994): A dataset of natural language queries about aviation paired with λ-calculus programs. We experiment with an IID split and a Template split (Finegan-Dollak et al., 2018) for evaluating compositional generalization, both taken from (Gupta et al., 2022). Overnight (Wang et al., 2015): A dataset containing both synthetic and natural language utterances from 11 domains (e.g. socialnetwork, restaurants, etc.) paired with Lambda-DCS logical forms. We experiment with an IID and a Template split of
the socialnetwork domain taken from (Gupta et al 2022).
2022). SMCalFlow (Andreas et al., 2020): A dataset of task-oriented natural language dialogs about calendars, weather, places, and people paired with executable dataflow programs. SMCalFlow-CS (Yin et al., 2021) is a subset of SMCalFlow containing single-turn dialogs involving two domains (organization structure and calendar event creation), each having its own set of program symbols with two types of test sets: a cross-domain (C) test set containing only instances where both domains appear and meant to test for compositional generalization, and a single-domain (S) test set contains instances with only single-domain for in-distribution evaluation. For compositional evaluation, we use the 32-C split which is a few-shot cross-domain split where the training set includes 32 cross-domain examples. For our IID evaluation, following Levy et al. (2022), we use the 8-S split. Additionally, we use the programs with the simplified syntax provided by (Meron, 2022). BREAK (Wolfson et al., 2020) is a dataset that maps complex natural language questions into a language-based meaning representation (QDMR) comprising an ordered list of atomic steps necessary to answer the question. Following (Rubin et al., 2022), we use the low-level Break subset where the targets are logical forms comprising lists of operators with their arguments based on the corresponding QDMR. MTOP (Li et al., 2021): A multilingual taskoriented semantic parsing dataset spanning six languages and 11 domains. The target commands are complex queries featuring nested intent-slot prediction. We use the English subset of MTOP from (Rubin et al., 2022).
# B.2 Non-Semantic Parsing
We additionally experiment with the standard IID splits of 9 non-semantic parsing datasets from the following categories: Numerical Reasoning: For this category, we experiment with GSM8K (Cobbe et al., 2021), a chain-of-thought reasoning (Wei et al., 2023) dataset of grade school-level arithmetic reasoning problems expressed in natural language and DROP (Dua et al., 2019), a dataset of question-answer pairs where the questions are about paragraphs containing numerical information and the answers are spans in the paragraph.
Classification: For this category, we experiment with three Natural Language Inference (NLI) datasests (QNLI (Wang et al., 2018), MNLI (Williams et al., 2018), and RTE (Bentivogli et al., 2009)), three Paraphrase Detection datasets (MRPC (Dolan and Brockett, 2005), PAWS (Zhang et al., 2019), and QQP (Wang et al., 2018)) and one Sentiment Classification dataset (SST2 (Socher et al., 2013)).
# C Selection Time
Despite their O(LTd) time complexity, we found example selection using both BSR and SET-BSR to be fast enough to not be a bottleneck to incontext learning for most datasets considered in this work. By using a GPU to compute c(x, z)s, we could get both to work in the order tens of milliseconds per test input on average which was significantly faster than the LLM inference time itself. The exceptions were DROP, PAWS, QQP, MNLI and QNLI for which the selection took >1 second due to much longer instances and/or larger instance pool. We leave it to future work to explore more efficient ways to measure informativeness.
# D Additional Analyses
BM25 From Fig. 6 we can see that coverage-based selection using BM25 with larger substructures outperforms vanilla unigram BM25 in compositional splits. BERTScore-Recall Examining the impact of importance weighting in Fig. 8 which compares the performance change with using importance weighting (IDF) in BSR, we can see that its effect is not consistent across different LLMs. We also did not see any consistent improvement from using larger deberta-large-mnli for computing token embeddings for instance-level BSR (see Fig. 9). However, it did help with set-level selection using SET-BSR. Reordering We found the reordering of demonstrations according to the corresponding instance-level metric to only be necessary for smaller LLMs (see Fig. 10), with it even hurting the performance of larger LLMs. We believe this is because larger and code-pretrained LLMs are more capable at composing the salient aspects in the different demonstrations and taking advantage of the full context. BSR outperforms Cosine even with the same encoder In § 7.2, we showed that BSR with
Dataset
Example Template
Sample Instance
Overnight
{source}\t{target} source: employees who finish after alices birthday
target:
(call listValue (call getProperty ((lambda s (call filter (var
s) (call ensureNumericProperty (string employment_end_date)) (string
>) (call ensureNumericEntity (call getProperty en.person.alice (string
birthdate))))) (call domain (string employee))) (string employee)))
ATIS
{source}\t{target} source: give me the flights from pittsburgh to los angeles thursday evening
target:
( lambda $0 e ( and ( flight $0 ) ( during_day $0 evening :
pd ) ( from $0 pittsburgh : ci ) ( to $0 los_angeles : ci ) ( day $0
thursday : da ) ) )
GeoQuery
{source}\t{target} source: which river traverses most states
target:
answer ( most ( river, traverse_2, state ) )
SMCalFlow {source}\t{target} source: Please put a 2 o’clock on my schedule where I’m meeting with boss Daniel.
target:
CreateEvent(AND(with_attendee("
Daniel
"),starts_at(NextTime(time=NumberPM(2)))))
BREAK
{source}\t{target} source: Is there another cube that is the same size as the cyan cube; what color is it?
target:
return the cyan cube ;return size of #1 ;return cubes besides
#1 ;return sizes of #3 ;return #3 where #4 is the same as #2 ;return
color of #5
MTOP
{source}\t{target} source: latest news from washington times please
target:
[IN:GET_STORIES_NEWS [SL:DATE_TIME latest ] [SL:NEWS_TYPE news
] [SL:NEWS_SOURCE washington times ] ]
Table 5: Semantic Parsing Datasets with corresponding sample instances and example templates used in for ICL.
Table 5: Semantic Parsing Datasets with corresponding deberta-large-mnli outperforms Cosine with all-mpnet-base-v2. Tables 15, 16, 17, and 18 show that the same trend holds even when using the same encoder, bert-base-uncased, for both metrics confirming that contextual embeddings are indeed better at capturing salient aspects. Recall of Syntactic Structures The improvements from set-based selection may be explained by Fig. 11 where we see that set-extensions COSINE and unigram BM25 reduce the recall of substructures of the test input whereas the recalls increase with set-extensions of both BM25[4-GRAM] and BM25[4-DEPST], and even BSR, which does not explicity consider these substructures.
# E Qualitative Analysis of Prompts
Tables 7, 8 show demonstrations selected using COSINE and SET-BSR for instances from MTOP and SMCalFlow-CS respectively. In each case, COSINE find demonstrations that are all very similar to the test input but fails to demonstrate some salient aspect, whereas BSR selects less similar instances but ensures complete coverage of all salient aspects. Tables 9 and 10 additionally illustrate limitations of set-selection and of token-embeddings in capturing salient aspects.
# F All Results
Tables 11 contains 8-shot ICL results for our proposed methods and prior learning-free and learningbased demonstration selection on all the LLMs
for all the semantic parsing datasets. For numerical reasoning and classification datasets, Tables 12 and 13 compare 8-shot ICL performance with prior training-free and trained methods, respectively. Table 14 provides average performances across all datasets. Additionally, Tables 15, 16, 17, 18, 20, and 21 contain results on semantic parsing datasets of all ablations of learning-free selection methods we ran, with GPT-Neo-2.7B, LLaMA-7B, LLaMA13B, StarCoder, Cushman, and Codex, respectively. We did not run ablations on GPT-3.5-Turbo due to its cost.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/88ea/88eaccd4-4818-43ad-b8f9-17b4911e256f.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Absolute improvement in average 8-shot ICL performance on different types of semantic parsing splits rom using the set extensions SET-BM25 with larger substructures over vanilla BM25.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/98a2/98a25f07-5e7f-4bca-9800-fece41978e90.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Comparison of 8-shot ICL performance of different variants of BERTScore with token embeddings computed using deberta-base-mnli. For easier visualization, since we found BERTScore-Precision to consistently perform worst, we show absolute improvement in average performance on different types of splits from the recall and F1 metrics over the precision metric.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7f56/7f568a6c-d219-4a93-8fb0-51fab193b009.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: Impact on average 8-shot ICL performance on semantic parsing splits from using importance weighting (IDF) in BSR.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f409/f4091f64-41a1-4a05-b177-a39cf1d6e21c.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: Impact on average 8-shot ICL performance on semantic parsing splits from using a larger deberta-large-mnli LLM for computing contextual token embeddings v/s using deberta-base-mnli in BSR and SET-BSR.</div>
<div style="text-align: center;">Figure 9: Impact on average 8-shot ICL performance on semantic parsing splits from using a larger deberta-large-mnli LLM for computing contextual token embeddings v/s using deberta-base-mnli in BSR</div>
Dataset
Example Template
Sample Instance
GSM8K Question: {question}
Solution: {solution}
question: Natalia sold clips to 48 of her friends in April, and then she sold half as
many clips in May. How many clips did Natalia sell altogether in April and May?
solution: Natalia sold 48/2 = «48/2=24»24 clips in May. Natalia sold 48+24 =
«48+24=72»72 clips altogether in April and May. #### 72
DROP
Passage: {passage}
Question: {question}
Answer: {answer}
passage: To start the season, the Lions traveled south to Tampa, Florida to take
on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a
23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field
goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from
Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field
goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by
Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard
TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The
second half was relatively quiet, with each team only scoring one touchdown. First,
Detroit’s Calvin Johnson caught a 1-yard pass in the third quarter. The game’s final
points came when Mike Williams of Tampa Bay caught a 5-yard pass. The Lions
won their regular season opener for the first time since 2007
question: How many points did the buccaneers need to tie in the first?
answer: 3
QNLI
Question: {question}
Sentence: {sentence}
Answer: {label}
sentence: Unlike the two seasons before it and most of the seasons that followed,
Digimon Tamers takes a darker and more realistic approach to its story featuring
Digimon who do not reincarnate after their deaths and more complex character
development in the original Japanese.
question: When did the third Digimon series begin?
label: No
MNLI
Premise: {premise}
Hypothesis: {hypothesis}
Answer: {label}
premise: The new rights are nice enough
hypothesis: Everyone really likes the newest benefits
label: Maybe
RTE
Premise: {premise}
Hypothesis: {hypothesis}
Answer: {label}
premise: Dana Reeve, the widow of the actor Christopher Reeve, has died of lung
cancer at age 44, according to the Christopher Reeve Foundation.
hypothesis: Christopher Reeve had an accident.
label: Yes
MRPC
Sentence 1: {sentence1}
Sentence 2: {sentence2}
Answer: {label}
sentence1: He said the foodservice pie business doesn ’t fit the company ’s long-term
growth strategy.
sentence2: " The foodservice pie business does not fit our long-term growth strategy
.
label: Yes
PAWS
Sentence 1: {sentence1}
Sentence 2: {sentence2}
Answer: {label}
sentence1: Bradd Crellin represented BARLA Cumbria on a tour of Australia with 6
other players representing Britain , also on a tour of Australia .
sentence2: "Bradd Crellin also represented BARLA Great Britain on a tour through
Australia on a tour through Australia with 6 other players representing Cumbria .
label: No
QQP
Question 1: {question1}
Question 2: {question2}
Answer: {label}
question1: Why are African-Americans so beautiful?
question2: "Why are hispanics so beautiful?
label: No
SST2
Review: {sentence}
Answer: {label}
sentence: it ’s a charming and often affecting journey .
label: Positive
Table 6: Non-Semantic Parsing Datasets with corresponding sample instances and example templates used for ICL.
GSM8K Question: {question} Solution: {solution} question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? solution: Natalia sold 48/2 = «48/2=24»24 clips in May. Natalia sold 48+24 = «48+24=72»72 clips altogether in April and May. #### 72 DROP Passage: {passage} Question: {question} Answer: {answer} passage: To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit’s Calvin Johnson caught a 1-yard pass in the third quarter. The game’s final points came when Mike Williams of Tampa Bay caught a 5-yard pass. The Lions won their regular season opener for the first time since 2007 question: How many points did the buccaneers need to tie in the first? answer: 3 QNLI Question: {question} Sentence: {sentence} Answer: {label} sentence: Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese. question: When did the third Digimon series begin? label: No MNLI Premise: {premise} Hypothesis: {hypothesis} Answer: {label} premise: The new rights are nice enough hypothesis: Everyone really likes the newest benefits label: Maybe RTE Premise: {premise} Hypothesis: {hypothesis} Answer: {label} premise: Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, according to the Christopher Reeve Foundation. hypothesis: Christopher Reeve had an accident. label: Yes MRPC Sentence 1: {sentence1} Sentence 2: {sentence2} Answer: {label} sentence1: He said the foodservice pie business doesn ’t fit the company ’s long-term growth strategy. sentence2: " The foodservice pie business does not fit our long-term growth strategy . label: Yes PAWS Sentence 1: {sentence1} Sentence 2: {sentence2} Answer: {label} sentence1: Bradd Crellin represented BARLA Cumbria on a tour of Australia with 6 other players representing Britain , also on a tour of Australia . sentence2: "Bradd Crellin also represented BARLA Great Britain on a tour through Australia on a tour through Australia with 6 other players representing Cumbria . label: No QQP Question 1: {question1} Question 2: {question2} Answer: {label} question1: Why are African-Americans so beautiful? question2: "Why are hispanics so beautiful? label: No SST2 Review: {sentence} Answer: {label} sentence: it ’s a charming and often affecting journey . label: Positive
Table 6: Non-Semantic Parsing Datasets with corresponding sample instances and example templates used for ICL.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/114e/114e1f33-2c8f-4578-8f0c-1f2c05f7b807.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10: Impact on average 8-shot ICL performance on semantic parsing splits from reordering the demonstration selected by the different set-level metric using the corresponding instance-level metric as absolute gain v/s th unreordered version.</div>
Selector
Prompt
COSINE
Sentence: Easy vegan recipes
Logical Form: [IN:GET_RECIPES [SL:RECIPES_ATTRIBUTE Easy ] [SL:RECIPES_TYPE vegan ] ]
Sentence: Vegetarian recipes
Logical Form: [IN:GET_RECIPES [SL:RECIPES_TYPE Vegetarian ] ]
Sentence: Please find me vegan recipes
Logical Form: [IN:GET_RECIPES [SL:RECIPES_TYPE vegan ] ]
Sentence: Give me vegan recipes
Logical Form: [IN:GET_RECIPES [SL:RECIPES_TYPE vegan ] ]
SET-BSR
Sentence: I have a nut allergy. Find me a dessert recipe
Logical Form: [IN:GET_RECIPES [SL:RECIPES_EXCLUDED_INGREDIENT nut ] [SL:RECIPES_MEAL dessert ]
]
Sentence: Create a video message for Victoria with plan options for dinner with family next
week
Logical Form: [IN:SEND_MESSAGE [SL:TYPE_CONTENT video ] [SL:RECIPIENT Victoria ] ]
Sentence: What are some no-bake dessert ideas
Logical Form: [IN:GET_RECIPES [SL:RECIPES_COOKING_METHOD no - bake ] [SL:RECIPES_MEAL dessert
] ]
Sentence: Vegan birthday cakes
Logical Form: [IN:GET_RECIPES [SL:RECIPES_TYPE Vegan ] [SL:RECIPES_DISH birthday cakes ] ]
Table 7: Demonstrations selected for the MTOP input: Vegan desert options with target output [IN:GET_RECIPES [SL:RECIPES_TYPE Vegan ] [SL:RECIPES_DISH birthday cakes ] ]. COSINE’s reliance on a single dense embedding means it is unable to account for the fact that "options" could mean dishes and not just recipes.
Selector
Prompt
COSINE
Sentence: I need a meeting with Elli tomorrow at 11 pm
Logical Form: CreateEvent(AND(with_attendee (" Elli "),starts_at(Tomorrow ()),starts_at(NumberPM
(11))))
Sentence: Set a meeting with Elli for tomorrow at 2 pm through the end of the day and call it
Recap
Logical Form: CreateEvent(AND(ends_at(AND(GE(DateTime ?(date=Tomorrow (),time=NumberPM (2))),
EndOfWorkDay ())),with_attendee (" Elli "),has_subject (" Recap "),starts_at(Tomorrow ()),
starts_at(NumberPM (2))))
Sentence: Schedule a meeting with Elli for tomorrow at 4 pm through the end of the workday
Logical Form: CreateEvent(AND(ends_at(AND(GE(DateTime ?(date=Tomorrow (),time=NumberPM (4))),
EndOfWorkDay ())),with_attendee (" Elli "),starts_at(Tomorrow ()),starts_at(NumberPM (4))))
Sentence: Schedule a meeting with Elli from 4 PM until the end of the day tomorrow .
Logical Form: CreateEvent(AND(ends_at(AND(GE(DateTime ?(date=Tomorrow (),time=NumberPM (4))),
EndOfWorkDay ())),with_attendee (" Elli "),starts_at(Tomorrow ()),starts_at(NumberPM (4))))
SET-BSR
Sentence: I need a doctor 's appointment on Wednesday morning .
Logical Form: CreateEvent(AND(has_subject (" doctor 's appointment "),starts_at(Morning ()),
starts_at(NextDOW (" WEDNESDAY "))))
Sentence: I need to see Alice and her boss next Monday at 3 pm .
Logical Form: CreateEvent(AND(with_attendee (" Alice "),with_attendee(FindManager (" Alice ")),
starts_at(NextDOW (" MONDAY ")),starts_at(NumberPM (3))))
Sentence: Schedule a meeting with Jake , Elli , and Jesse for Friday at 2 pm .
Logical Form: CreateEvent(AND(with_attendee (" Jesse "),with_attendee (" Jake "),with_attendee ("
Elli "),starts_at(NextDOW (" FRIDAY ")),starts_at(NumberPM (2))))
Sentence: I need to schedule a meeting with Jeff 's supervisor Lynne for tomorrow at 10 AM .
Logical Form: CreateEvent(AND(with_attendee (" Lynne "),starts_at(Tomorrow ()),starts_at(
NumberAM (10))))
Table 8: Demonstrations selected for the SMCalFlow-CS input: Schedule a meeting with Elli and he manager ’s boss tomorrow morning. SET-BSR is able to find demonstrations covering all fragments of the te input while COSINE fails to include anything which involves finding someones manager.
Selector
Prompt
COSINE
Question: Justin has a box that is 12 inches in height. The length of the box is 3 times its
height and 4 times its width. What is the volume of the box?
Question: John builds a box.
The box is 26 inches by 26 inches by 14 inches.
The walls are 1
inch thick on each side.
How much is the internal volume in cubic feet?
BSR
Question: A window is made up of 8 glass panes. Each pane has a length of 12 inches and a
width of 8 inches. What is the area of the window?
Question: John builds a box.
The box is 26 inches by 26 inches by 14 inches.
The walls are 1
inch thick on each side.
How much is the internal volume in cubic feet?
SET-BSR
Question: Jazel has 3 sticks. One stick is 3 centimeters long. The second stick is twice as
long while the third stick is 1 centimeter shorter than the second stick. What is the
total length of Jazel 's sticks when they are put together?
Question: John builds a box.
The box is 26 inches by 26 inches by 14 inches.
The walls are 1
inch thick on each side.
How much is the internal volume in cubic feet?
Table 9: Demonstrations selected by different methods for the GSM8K input: John has 3 boxes. Each box is 5 inches by 6 inches by 4 inches. The walls are 1 inch thick. What is the total inner volume of all 3 boxes? We only show the inputs for clarity. Only BSR solves this input (2-shot ICL with Codex). All three methods select one example that demonstrates most of the aspects of the test input, i.e., computing the volume of a box after subtracting wall thickness. The remaining aspect is computing the total of a quantity computed for 3 identical items. COSINE fails to do so, selecting yet another example that requires computing a single box’s volume. Since SET-BSR prioritizes coverage of the remaining aspect, it selects an example that has exactly three items whose total length has to be computed but overall is not very similar in reasoning. BSR on the other hand tries to find an example that demonstrates all aspects by itself and happens to find one that partially demonstrates the remaining aspect as well.
Selector
Prompt
BSR
Begun in 1960 and opened to traffic in 1968, the bridge is a two -tiered road and rail design
spanning 4,600 metres on the upper deck , with approximately 1,580 metres spanning the
river itself. Can we
know "What type of design is the bridge ?"? Yes
The BBC also introduced Ceefax , the first teletext service , starting in 1974. Can we know "What
kind of service was Ceefax ?"? Yes
The Water , Sanitation and Hygiene (WSH) program of the Gates Foundation was launched in mid
-2005 as a "Learning Initiative ," and became a full -fledged program under the Global
Development Division in early 2010. Can we know "What was the WSH program launched in
2005"? Yes
Television broadcasting in Hyderabad began in 1974 with the launch of Doordarshan , the
Government of India 's public service broadcaster , which transmits two free -to -air
terrestrial television channelsand one satellite channel. Can we know "What is Doordarshan
?"? Yes
Table 10: Top four demonstrations selected by different methods for the QNLI input: Telenet was incorporated in 1973 and started operations in 1975. Can we know "What was telenet"? Since BSR doesn’t hav access to the labels and also cannot reason about the inputs themselves, it cannot account for the fact that the contex in the test input does not contain the answer for the question and selects demonstrations that are all answered "Yes even though the answer to the test input is "No".
Dataset
ATIS
Overnight
Break MTOP
GeoQuery
SMCalFlow-CS
AVERAGE
Split
IID/Templ. IID/Templ.
IID
IID
IID/Templ./TMCD/Len.
8_S/32_C
All/IID/Comp.
LM Selector
GPT-Neo-2.7B
EPR
66.1 / 12.2
52.3 / 0.9
29.9
62.2
71.4 / 33.6 / 43.6 / 28.8
54.5 / 3.6
38.3 / 56.1 / 20.5
CEIL
67.8 / 18.7
50.7 / 2.1
29.9
60.5
65.4 / 30.2 / 43.6 / 25.2
59.1 / 3.8
38.1 / 55.6 / 20.6
Random
12.4 / 0.0
3.6 / 0.0
1.9
1.3
17.5 / 11.0 / 14.0 / 0.9
3.0 / 0.0
5.5 / 6.6 / 4.3
Cosine
46.1 / 6.5
38.3 / 0.4
22.3
43.9
67.9 / 24.1 / 41.4 / 28.5
25.2 / 1.2
28.8 / 40.6 / 17.0
BM25
49.5 / 7.4
33.7 / 3.0
26.5
47.7
63.6 / 40.6 / 42.1 / 25.5
32.0 / 3.2
31.2 / 42.2 / 20.3
BSR
48.3 / 7.8
40.1 / 2.6
29.1
54.5
67.1 / 40.7 / 47.7 / 28.2
39.7 / 3.5
34.1 / 46.5 / 21.7
Set-BSR
54.6 / 13.2
43.2 / 4.9
28.6
55.1
67.1 / 45.3 / 45.4 / 26.4
41.5 / 4.8
35.8 / 48.4 / 23.3
LLaMA-7B
EPR
73.0 / 21.0
57.7 / 1.8
33.2
65.2
75.4 / 49.3 / 45.8 / 30.3
64.0 / 8.0
43.7 / 61.4 / 26.0
CEIL
74.0 / 30.5
55.8 / 4.4
36.1
66.8
66.8 / 50.5 / 45.3 / 24.2
67.4 / 11.9
44.5 / 61.1 / 27.8
Random
9.5 / 0.0
4.2 / 0.5
8.8
2.8
9.3 / 13.3 / 9.2 / 4.5
6.2 / 0.0
5.7 / 6.8 / 4.6
Cosine
56.7 / 11.5
48.7 / 0.0
26.1
49.8
73.9 / 33.5 / 42.6 / 29.4
37.3 / 3.2
34.4 / 48.8 / 20.0
BM25
61.0 / 12.5
45.1 / 2.5
30.1
53.6
67.9 / 39.5 / 44.9 / 30.6
43.4 / 9.5
36.7 / 50.2 / 23.3
BSR
60.9 / 14.3
51.2 / 3.0
32.5
59.1
72.5 / 47.2 / 46.9 / 30.3
54.1 / 9.8
40.1 / 55.0 / 25.2
Set-BSR
64.3 / 21.2
51.5 / 6.3
33.7
61.9
76.1 / 52.3 / 48.5 / 35.8
54.5 / 19.3
43.8 / 57.0 / 