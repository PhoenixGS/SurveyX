# In-Context Learning with Many Demonstration Examples
Mukai Li 1 Shansan Gong 1 Jiangtao Feng 1 Yiheng Xu 1 2 Jun Zhang 1 Zhiyong Wu 1 Lingpeng Kong 1 2
# Abstract
Large pre-training language models (PLMs) have shown promising in-context learning abilities. However, due to the backbone transformer architecture, existing PLMs are bottlenecked by the memory and computational cost when scaling up to a large context size, leaving instruction tuning and in-context learning of many demonstration examples, as well as long-range language modeling under-explored. In this study, we propose a long-range language model EVALM based on an efficient transformer mechanism. EVALM is trained with 8k tokens per batch line and can test up to 256k-lengthed contexts with extrapolation, 128× to the limit of existing PLMs (e.g. GPT3). Based on EVALM, we scale up the size of examples efficiently in both instruction tuning and in-context learning to explore the boundary of the benefits from more annotated data. Experimental results on a diverse set of tasks show that EVALM achieves 4.1% higher accuracy on average, and the average length of achieving the best accuracy score over tasks is around 12k. We find that in-context learning can achieve higher performance with more demonstrations under many-shot instruction tuning (8k), and further extending the length of instructions (16k) can further improve the upper bound of scaling incontext learning. Code is available on https: //github.com/Shark-NLP/EVALM.
arXiv:2302.04931v1
# 1. Introduction
With the increasing scale of pre-trained language models (PLMs), in-context learning (ICL) has emerged as a novel paradigm for utilizing PLMs (Brown et al., 2020b; Zhang et al., 2022c; Chowdhery et al., 2022). Unlike learning
1Shanghai Artificial Intelligence Laboratory 2Department of Computer Science,The University of HongKong. Correspondence to: Jiangtao Feng <fengjiangtao@pjlab.org.cn>, Lingpeng Kong <lpk@cs.hku.hk>.
methods that require updating parameters, in-context learning allows for good model performance with a prompt that only includes natural language instructions and/or a few demonstrations (Dong et al., 2023). In addition to that, a recent line of research on instruction tuning shed new light on closing the gap between pre-training and in-context learning (Chung et al., 2022; Min et al., 2022), facilitating the usage of natural language instructions to interact with the PLMs.
methods that require updating parameters, in-context learning allows for good model performance with a prompt that only includes natural language instructions and/or a few demonstrations (Dong et al., 2023). In addition to that, a recent line of research on instruction tuning shed new light on closing the gap between pre-training and in-context learning (Chung et al., 2022; Min et al., 2022), facilitating the usage of natural language instructions to interact with the PLMs. However, the computational overhead of the backbone vanilla transformer architecture prevents existing PLMs from a longer context. A maximum context size (i.e., 2048) is set in the most popular pre-training models (e.g., GPT3, Brown et al. 2020b; OPT, Zhang et al. 2022c; PaLM Chowdhery et al. 2022). The direct consequence is scaling up to large numbers of samples in instruction tuning or in-context learning becomes under-explored. How effectively can we improve the in-context learning performance of the PLMs by serving more demonstration examples? To answer this question, we start from responding to the challenge of long-range language models (LRLMs). We train an LRLM named EVALM (§ 3.2), which backbones on a state-of-the-art efficient transformer architecture EVA (Zheng et al., 2023), with modifications to handle the extrapolation of position embeddings (§ 3.1). EVALM with many-shot instruction tuning achieves better performance in long-range language modeling with cheap memory and computational costs (§ 4.4). The learned circular position embedding and incremental encoding we propose help EVALM to extrapolate to an input length of 256k tokens effectively. We then conduct a series of experiments testing the performance of EVALM when scaling up the number of demonstration examples in ICL in various tasks. We find that with more demonstration examples, EVALM is able to achieve better ICL performance than comparable PLMs with rare extra overheads. We summarize our contribution as follows:
# 2. Related Work
Pre-trained Language Model PLMs are trained on large and general corpora and then finetuned or few-shot transferred to perform various NLU and NLG tasks. Among them, besides encoder-decoder Transformer (Vaswani et al., 2017) architecture such as T5 (Raffel et al., 2020), there are auto-regressively pre-trained models, like XLNet (Yang et al., 2019), GPT (Radford et al., 2019; Brown et al., 2020a; Black et al., 2022), OPT (Zhang et al., 2022c), PaLM (Chowdhery et al., 2022), BLOOM (Scao et al., 2022), and etc. These decoder-based causal language models soon occupy kinds of NLP leaderboards, showing excellent language modeling and in-context learning ability of them. However, the huge computing overhead (including memory and time consumption) makes nonprofits and smaller labs difficult to create or even use PLMs. Furthermore, this also prevents PLMs from encoding longer inputs. Efficient Attention A surge of efficient attention models are devised to enhance the efficiency of the original Transformer model (Vaswani et al., 2017). These models explore diverse philosophies to improve the efficiency, including sparse attention matrix (Luong et al., 2015; Tay et al., 2020; Beltagy et al., 2020; Zaheer et al., 2020; Ainslie et al., 2020), memory compression (Liu et al., 2018; Lee et al., 2019; Rae et al., 2020; Wang et al., 2020) low-rank decomposition (Xiong et al., 2021; Lu et al., 2021; Chen et al., 2021), kernel-based linear attention (Choromanski et al., 2021; Peng et al., 2021; 2022; Zheng et al., 2022; 2023), state-space model (Gu et al., 2022; Gupta et al., 2022; Dao et al., 2022b), and CUDA re-implementation (Dao et al., 2022a). Thus models with efficient attention architecture are promising to handle longer input sequences when memory consumption is saved. H3 (Dao et al., 2022b) is pre-trained as an efficient language model but fails to scale up the training sequence length which remains 2048. In-Context Learning With the increasing scale and capacity of PLMs, ICL has become a new paradigm for
In-Context Learning With the increasing scale and capacity of PLMs, ICL has become a new paradigm for NLP (Brown et al., 2020a). The success of ICL has been
demonstrated on a wide range of NLP tasks, including question answering (Joshi et al., 2017), information retrieval (Tay et al., 2022), math word problem (Cobbe et al., 2021), commonsense reasoning (Geva et al., 2021), and fact checking (Rae et al., 2021) etc. Several recent studies (Liu et al., 2022; Wu et al., 2022) have observed a positive correlation between the number of in-context examples and ICL’s performance: increasing the number of in-context examples can bring steady improvements. Further investigation is carried out to pack and/or distill more examples into the context through continued pre-training (Choi et al., 2022), and instruction tuning (Snell et al., 2022). However, the input length limitation of current PLMs still restricts us from directly feeding more in-context examples into the model.
# 3. EVALM
We propose a long-range language model named EVALM to scale up the sequence length reached by existing pre-trained language models. The rest of this section is organized as follows: § 3.1 introduces the overall architecture of EVALM; § 3.2 focuses on learning EVALM on both of pre-training and instruction tuning; § 3.3 shows how EVALM scales up the maximum size of shots in in-context learning, with an incremental encoding technique. The overall architecture is shown in Figure 1.
# 3.1. Architecture
We adopt EVA (Zheng et al., 2023), a recently introduced attention competitor, as an efficient alternative to vanilla softmax attention (Vaswani et al., 2017), for its high efficiency in long sequence modeling and strong performance. The original EVA performs both causal and noncausal attention in sequence modeling, and here we focus on its causal version for its adaption to language modeling. A general computation process of causal EVA is described as follows. Given a query qt ∈Rd, and key-value sequences K1:t, V1:t ∈Rt×d, where d is the dimensionality and t is the timestamp, EVA learns attentive features as: a) chunking key-value features K1:t, V1:t as Kr, Kl = C(K1:t), Vr, Vl = C(V1:t), where C(·) is a chunking function with chunk size c, and superscripts r and l denote the remote features beyond present chunk of qt and the local features within the chunk; b) compressing remote features within each chunk by another efficient attention and pooling operation M(·) as ˆKr = M(Kr), ˆVr = M(Vr), where the efficient attention here is LARA (Zheng et al., 2022); c) performing vanilla attention on concatenated remote and local features by EVA(qt) = softmax(qt[ ˆKr; Kl]⊤)[ ˆVr; Vl]⊤. It is worth noting that EVA is capable of handling long-term dependencies by performing attention on remote compressed features ˆKr, ˆVr. We refer interested readers to (Anonymous, 2023)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0ce8/0ce82960-dab9-4a1c-82d3-5d7e76ef788c.png" style="width: 50%;"></div>
chunk size … Figure 1. The illustration of EVALM scaling up in-context learning. The pre-training stage empowers the language modeling capacity of EVALM, and the instruction tuning explicitly aligns EVALM with instructions from different tasks. For different downstream tasks EVALM can in-context learn from the demonstrations. With the help of CPE and incremental encoding technique, k could be scaled up
for further details.
Apart from the advanced attention mechanism EVA, we present circular positional embedding (CPE) to enforce position information. For i-th token, its positional embedding is set to pi%M ∈Rd, where M is the maximum size of learned positional embeddings. An intriguing characteristic of CPE is its ability on extrapolation and long-term dependency. CPE implicitly learns a position-aligned matrix P = {p⊤ i%Mpj%M} between each pair of tokens, which is added to attention matrices. The matrix P is close to the pattern of strided attention (Ho et al., 2019; Tay et al., 2020) with stride size M, and encourages feature interaction to distant features.
Extrapolation Extrapolation is a vital challenge in longrange language modeling. Remind that the LRLMs are expected to scale the sequence length to tens, hundreds, or even more times to the current limitation with thousands of tokens from existing mainstream pre-trained language models such as GPT (Brown et al., 2020a) and OPT (Zhang et al., 2022b). The challenges of LRLMs lie in the two aspects. On the one hand, such length is still unaffordable for current models, even for efficient attention models, during the training stage, despite incremental decoding (Ott et al., 2019) helping reduce memory consumption in the inference stage. On the other hand, pre-trained data from long-range texts are limited. Thus a practical solution is “train short, test long”, a.k.a. extrapolation. Thus finding an architecture with extrapolation capability is important for
LRLMs. In EVALM, we enhance its extrapolation in two aspects: a) based on the observation that locality contributes to extrapolation (Zhang et al., 2022a), we choose EVA that also models the locality; b) we use circular positional embedding that fledges vanilla learned positional embedding to extrapolate to longer contexts.
# 3.2. Pre-training & Instruction Tuning
We pre-train a causal language model EVALM based on EVA transformer decoder with our preprocessed Pile (Gao et al., 2020) corpus and further tune it using Many-Shot Instruction Tuning (MSIT).
Pre-training Data processing details are in Appendix A.1, Pre-training details are in Appendix A.2. Our EVALM was trained on a widely-used corpus the Pile (Gao et al., 2020), which is a massive dataset designed for training large language models. We built a preprocessing pipeline including filtering, deduplicating, and blending to prepare the pretraining corpus to support the large-scale distributed training process. We conducted catalog and content filtering following BLOOM (Scao et al., 2022) and deduplicated the filtered data using fuzzy deduplication similar to previous work (Zhang et al., 2022b; Smith et al., 2022). The final corpus roughly contains 121B tokens. Please refer to Appendix A.1 for detailed data processing and comparison. The training process EVALM mainly follows GPT3 (Brown et al., 2020a) and OPT (Zhang et al., 2022c), optimizing the
negative log-likelihood of next tokens in an auto-regressive way. We scaled the training sequence length to 8192 to accommodate more in-context samples. Fully sharded data parallel (FSDP) was applied in our pre-training stage, which can reduce the memory footprint of a single GPU to accommodate longer sequences. Please refer to Appendix A.2 for a detailed pre-training setting.
Many-Shot Instruction Tuning Instruction tuning simulates the in-context learning settings and shows the promising ability to activate the model’s respective capacity during inference (Min et al., 2022), with maximum training shots to 32. Based on our long-range EVALM, we can further investigate the impact of instruction tuning after scaling up the shots. We instruction-tuned EVALM on m instruction tuning tasks DIT j = {(xj i, yj i)}Nj i=1, which m is the number of tasks and Nj is sample number for each dataset. Each input-output pair (xj i, yj i) is turned into an instruction sequence sIT i = I(xi, yi) wrapped by the instruction I(·) written in natural language. I(·) is derived from an instruction templates pool that is manually designed for different tasks j. Before feeding into EVALM, we concatenate instructions until their total length reaches the limitation of 8192, in a batch-by-token way, named many-shot instruction tuning (MSIT). Further, we introduce the plus version of MSIT, which extrapolates the total length of instructions per batch line to 2 × 8192. The pressed EVALM learns the concatenated sIT i , under the supervision of the negative log-likelihood objective as language modeling. The data used in our instruction tuning refers to Appendix B.
# 3.3. In-Context Learning with EVALM
Consider a downstream task with dataset D, from which we construct a demonstration exemplar set De = {(xi, yi)}k i=1. As instruction tuning, the demonstration exemplars are turned into instruction sequences and then concatenated to se = [I(xi, yi)]k i=1. For a test input text x and its corresponding candidate categories Y, we first concatenate se and I(x, y) together to form a prompt for y ∈Y. The prompt is then fed into the pre-trained EVALM to compute the likelihood of the current answer y along with x, and we choose the most possible one as the predicted label:
(1)
There are several approaches to constructing De specifically, listed in § 4.1. For instance-level ICL, the same test sample x shares the same se, and for dataset-level ICL, all test samples share the same se (Wu et al., 2022). In this situation, Eq. (1) turns into:
arg max y∈Y P(I(x, y)|se).
(2)
Limited by the maximum encoding length of current PLMs (e.g., 2048), the maximum k of ICL is generally about 32 (Min et al., 2022). The upper bound of ICL when scaling up k remains a question. Intuitively, scaling up the shot number k of ICL can further help ICL reach the capacity of finetuning. Beyond the maximum encoding length of 8192, further scaling up k in an efficient way needs the incremental encoding technique.
Incremental Encoding Incremental decoding (Ott et al., 2019) enhances the sequence generation efficiency by caching useful historical states, namely incremental states, for future usage, which saves memory from the redundant computation. Inspired by this, we devise incremental encoding, which updates EVA cache states incrementally, for long context encoding. According to EVA architecture (§ 3.1), we maintain all local and remote features as incremental states S: {Kl, Vl, ˆKr, ˆVr}. When encoding the incoming tokens, we first concatenate them with local features and then compress full-chunk-sized local features into remote features and update S, details in Algorithm 1.
Algorithm 1 Incremental Encoding
Input: chunk size c, compression attention and pooling
operation M(·), incoming token xt, previous incremental
states St−1 : {Kl
t−1, Vl
t−1, ˆKr
t−1, ˆVr
t−1}
Output: updated incremental states St
qt, kt, vt = projection(xt)
Kl
t = [Kl
t−1; kt], Vl
t = [Vl
t−1; vt]
if length(Kl
t) is c then
ˆKr
t = [ ˆKr
t−1; M(Kl
t)], ˆVr
t = [ ˆVr
t−1; M(Vl
t)]
Kl
t := ∅, Vl
t := ∅
else
ˆKr
t = ˆKr
t−1, ˆVr
t = ˆVr
t−1
end if
return St : {Kl
t, Vl
t, ˆKr
t, ˆVr
t }
Previously, encoding long-range context requires quadratic memory complexity, and incremental encoding consequently scales it down to linear by caching previous S, where the memory consumption grows linearly along with the increase of S. Powered by this, EVALM further reduces the memory bottleneck and ensures the input length is scalable. In practice, the upper bound of encoding length is 32× than training, and it is possible to encode an extremely long sequence into incremental states losslessly, with the compression rate c. Incremental encoding thus brings numerous benefits for many scenarios like ICL. For ICL, considering many test samples share the same demonstration sequence se, we can encode it once, cache the long-term incremental states, and reuse them for further possible encoding. The test samples are then fed forward, conditioned on S, to predict the result using Eq. (2). Reusing the incremental states of demonstration saves the extra overheads of scaling k.
# 4. Experiments
In this section, we conduct in-context learning experiments to validate our EVALM and its instruction-tuned version on various tasks.
# 4.1. Experimental setting
Pre-training We pre-trained EVALM (350M and 1.3B) on 32 NVIDIA A100 80G GPUs. The hyper-parameters of EVALMs are identical to GPT3 (Brown et al., 2020a) and OPT (Zhang et al., 2022c) in the same scale, where the hidden size, number of attention heads and number of layers are 1024, 16, 24 respectively for the 350M model and 2048, 32, 24 respectively for the 1.3B model.
Instruction Tuning Following FLAN (Wei et al., 2021), we experiment ICL on the downstream tasks using EVALM instruction-tuned on FLAN datasets. FLAN dataset that belongs to the same cluster with the current test task is excluded during the instruction tuning stage, preventing the evaluation from data leakage and remaining our setting regarded as zero-shot or many-shot. There are three settings for instruction tuning in our experiments: a) IT: one-shot IT; b) MSIT: many-shot instruction tuning with a maximum of 8192 per batch line; c) MSIT+1: MSIT with a maximum of 2 × 8192 per batch line. More instruction tuning details can be seen in Appendix B.
In-Context Learning We mainly follow Wu et al. (2022) and Wei et al. (2021) to select several datasets from different NLP tasks. We choose SST-2 and SST-5 for sentiment classification (Socher et al., 2013), MNLI (Williams et al., 2018) for natural language inference, MultiRC (Khashabi et al., 2018) and BoolQ (Clark et al., 2019) for reading comprehension, AgNews (Zhang et al., 2015) for topic classification, WSC (Levesque et al., 2012) for coreference resolution, COPA (Roemmele et al., 2011) for commonsense reasoning and Trec (Hovy et al., 2001) along with WiC (Pilehvar & Camacho-Collados, 2019) for miscellaneous tasks.
We mainly adopt zero-shot and many-shot settings. The zero-shot setting directly wraps up the testing input with a task-specific template for inference. The many-shot approach randomly selects k demonstrations from the training set and uses the same demonstrations for the whole test set. This approach is universally used as dataset-level ICL. We also adopt Top-k approach following Wu et al. in § 4.4. Prompt designs are detailed in Appendix C.2.
We find the best shot number on the validation set and test on the test set when the label of the test set is available
(AgNews, Trec, SST-5). For other datasets, we split 500 samples from each training set as a validation set and report our results on the test set. The demonstration number k is set from 1 to 2000, please refer to Appendix C.1 for more in-context learning details.
Baselines We use OPT (Zhang et al., 2022c) as the main baseline due to its similar model architecture, number of parameters, training flops, training data, and training framework to our EVALM, allowing for a fair comparison. We conduct experiments using models of 350M and 1.3B parameters.
# 4.2. Main Results
The overall in-context learning results are shown in Table 1. Based on this, we make the following observations.
Scaling up demonstration examples helps ICL Since EVALM pre-trained with longer sequence length and adapted for extrapolation, we can use more demonstrations when conducting in-context learning experiments. At both 350M and 1.3B scale, EVALM outperforms OPT on both zero-shot and many-shot settings, and tends to achieve the best score at higher average shot number k (about 10 times to OPT). This shows that long-range EVALM can effectively utilize the information in demonstrations to get better results. The specific best shot numbers for each dataset and model are in Appendix C.1.
MSIT arouses the potential of many-shot ICL Table 1 shows that the model with MSIT, especially MSIT+, obtains the most growth, from zero-shot to many-shot setting, which is indicated by the relative improvement scores. This is partly because MSIT learns to align the language modeling with many-shot in-context learning scenarios, making it more suitable for testing in many-shot settings. Another reason is the relatively poor zero-shot performance with MSIT. A potential explanation is that learning too many tasks fills the capacity of small PLMs, which can be harmful to their zero-shot performance, as mentioned in FLAN (Wei et al., 2021). Thus, we speculate that combining MSIT and scaling in-context shot number k together is essential for getting the best in-context learning results.
Larger PLMs suit many-shot ICL Both EVALM-1.3B and OPT-1.3B show more significant progress compared with the 350M model. This is also consistent with the rule of scaling law (Chung et al., 2022). Large PLM contains more knowledge and can better conduct ICL through more demonstrations. This suggests that scaling up ICL may yield greater benefits on larger models.
Table 1. Main results of in-context learning on diverse tasks. The light grey shade refers to the ablation modules of IT. We average the
shot number of demonstrations when the best score is achieved. The best overall results are bolded. The abbreviation avg. is for average,
imprv. is for improvement, acc is for accuracy. The relative improvements of models in the many-shot setting are compared with
the same model but in the zero-shot setting respectively.
Models
Sentiment
NLI
Miscellaneous
Reading
Topic
Coreference Commonsense Avg.
acc Imprv. Avg.
shot
SST-2 SST-5 MNLI
Trec
WiC
MultiRC BoolQ AgNews
WSC
COPA
zero-shot
OPT-350M
64.6
29.9
21.6
23.0
52.7
46.3
53.8
50.9
63.4
65.0
47.1
-
-
EVALM-350M
61.4
25.8
27.5
21.8
51.7
56.9
56.9
46.6
63.5
64.0
47.6
-
-
w/ MSIT
50.8
28.2
28.6
20.4
50.6
43.3
49.9
47.5
63.5
65.0
44.8
-
-
w/ MSIT+
64.0
29.3
28.0
22.2
50.4
42.0
53.3
48.6
63.5
62.0
46.3
-
-
OPT-1.3B
73.0
31.3
20.0
22.0
50.3
41.7
51.4
56.6
62.5
72.0
48.1
-
-
EVALM-1.3B
82.3
31.3
21.6
22.8
52.1
41.7
58.5
55.3
58.6
72.0
49.6
-
-
w/ MSIT
58.4
33.5
21.6
23.0
52.3
52.2
52.4
55.2
58.2
71.0
47.8
-
-
many-shot
OPT-350M
62.3
31.0
33.8
27.6
51.6
57.2
62.8
63.8
63.4
64.0
51.7
4.6
10
EVALM-350M
61.0
32.3
32.1
49.6
52.0
55.7
60.6
69.7
63.4
63.0
53.9
6.3
97
w/ MSIT
65.2
31.2
34.1
39.4
52.4
53.4
57.5
70.1
63.5
72.0
53.9
9.1
236
w/ MSIT+
70.6
33.7
34.5
40.4
50.4
53.1
59.2
73.3
63.5
73.0
55.2
8.8
208
OPT-1.3B
73.0
40.1
31.3
45.6
50.3
52.5
65.2
60.0
63.4
74.0
55.3
7.3
14
EVALM-1.3B
76.6
40.4
30.2
46.8
54.3
58.9
62.5
62.5
63.5
74.0
57.0
7.3
152
w/ MSIT
84.2
45.4
33.9
49.4
54.2
60.2
64.2
63.2
65.4
74.0
59.4
11.6
269
<div style="text-align: center;">Table 2. Average accuracy and input length when achieving highest scores over all datasets with different IT strategies.</div>
Models
Vanilla
w/ IT
w/ MSIT
Acc. Length Acc. Length Acc. Length
OPT-350M
51.7
584.5
50.9
560.2
51.5 1592.3
EVALM-350M 53.9 3904.9 53.7
3682
53.9 8087.7
OPT-1.3B
55.3
665.0
54.5
670.3
54.8 1809.6
EVALM-1.3B 57.0 7337.0 56.9 8140.6 59.4 12558.0
# 4.3. Analysis on MSIT
Efficacy of MSIT To further investigate the effectiveness of MSIT, we average the best ICL results on instructiontuned EVALM as the shot number increases. Considering the average example length of different datasets varying from each other, we also count the length of demonstrations at the peak of accuracy instead of using the shot number. All results are averaged over 10 datasets in Table 2. We also conduct the same experiment on the same size OPT model, but with 2048 tokens per batch line for MSIT. We observe that as the number of IT examples grows, the average lengths of many-shot examples increase accordingly, for both OPT and EVALM. It reflects that MSIT indeed learns the alignment with many-shot ICL. Such alignment helps EVALM’s enhance its capability on many-shot ICL, but becomes helpless or even harmful to OPT. A possible reason is that EVALM is specialized in long-range language modeling with extrapolation whilst OPT is not. Thus, many-
Scaling k-shot ICL We dig into the specific accuracy curve as the demonstration length rises, taking the AgNews dataset using randomly selected many-shot ICL and the Trec dataset using Top-k many-shot ICL as examples. Please refer to § 4.4 for more analysis about the Top-k setting. We choose this setting to analyze considering the robustness of the approach and stability of the curve. As shown in Figure 2, we observe that EVALM without instruction tuning or just with one-shot instruction tuning achieves the highest accuracy within 128 shots, which corresponds to around 2k length, and further adding demonstrations makes the accuracy curve drop quickly. With many-shot instruction tuning, the best accuracy is improved and the heavy drop gets alleviated. With instruction tuning on longer range (MSIT+), the accuracy grows steadily along with increasing demonstration length and peaks at 768 shots, which corresponds to around 15k length. Similar trends can be found in Figure 3 but OPT can not. This trend indicates that MSIT encourages our language model to achieve higher accuracy, and scaling up the shot number of IT further improves the upper bound of scaling in-context learning on downstream tasks. However, the increasing trend is not endless, even for MSIT+. When the length of demonstration examples reaches 20k, the rapid drop of the accuracy curve can be seen. The possible reasons are listed as follows. On the one hand, modeling the longer input length relies on the extrapolation ability of models, and the size of models could also be the limiting factor. There are more discussions in § 4.4. On
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8ce8/8ce84c05-5cd3-4f4a-aeea-ff18061fdbb3.png" style="width: 50%;"></div>
Figure 2. The ICL accuracy curve along with demonstration length on Trec dataset using the Top-k approach, for EVALM-350M models with different instruction tuning strategies.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6813/68130b23-923c-4f1c-9ced-aa2fcbf16356.png" style="width: 50%;"></div>
<div style="text-align: center;">Length of tokens (×8192)</div>
Figure 3. The ICL accuracy curve along with demonstration length on AgNews dataset using the random in-context examples, for EVALM-350M with different instruction tuning strategies and OPT-350M.
the other hand, the setting of enlarging the demonstration example sizes, in both of instruction tuning and in-context learning, is under-explored, due to the lack of pre-trained LRLMs. Therefore, advanced ICL algorithms are demanded for further investigation on many-shot in-context learning.
# 4.4. Discussion
Extrapolation Ability To ensure the extrapolation ability of EVALM, we simply adopt CPE (§ 3.1), and incremental encoding (§ 3.3) is deployed to save memory consumption. With these techniques, EVALM is able to encode super-long inputs, i.e. 256k on 80G NVIDIA A100, during inference. For comparison, we also adapt the incremental encoding technique to the OPT model of the same size, whose maximum context size still lags behind EVALM’s. Detailed comparison of memory consumption between OPT and EVALM can be found in Appendix C.1.
Based on this, we further evaluate the extrapolation ability of different models using perplexity. The experiment is conducted on PG-19 dataset (Rae et al., 2019), a dataset focusing on long-range language modeling, following the setting by Zhang et al. (2022a). The perplexity curve along with the input length is addressed in Figure 4. The perplexity of OPT grows steeply once the input length is over 2048, indicating its poor extrapolation ability. The vanilla EVALM with MSIT increases the perplexity, which is expected considering that the instruction tuning will adapt PLMs from
<div style="text-align: center;">Table 3. Results of using Top-k ICL approach. The light grey shade refers to the ablation modules of IT. The best results are bolded. The abbreviation avg. is for average.</div>
bolded. The abbreviation avg. is for average.
Models
SST-2 SST-5 MNLI Trec AgNews Avg.
OPT-350M
86.1
44.5
33.8
74.8
91.0
66.0
EVALM-350M
88.2
46.5
29.5
76.8
91.8
66.6
w/ MSIT
86.0
43.6
27.2
78.0
90.9
65.1
w/ MSIT+
88.3
44.9
27.7
83.8
91.9
67.3
OPT-1.3B
86.7
43.2
25.1
77.0
91.3
64.7
EVALM-1.3B
87.7
47.4
30.2
79.0
91.8
67.2
w/ MSIT
88.2
47.0
32.2
76.0
91.0
66.9
the general corpus towards several specific tasks. Compared with MSIT, EVALM with MSIT+ achieves lower perplexity even lower than the vanilla model. MSIT+ also reaches the lowest perplexity at a larger input length around 16k. These observations explain why the OPT benefits less while EVALM benefits more from MSIT especially MSIT+ in Table 2, Figure 2 and Figure 3.
Top-k ICL Following Wu et al., we also deploy Top-k approach (instance-level) which selects the k most similar samples from the training dataset based on embedding similarities (Liu et al., 2022; Gao et al., 2021) and puts the samples with higher similarity closer to the testing input. We conduct Top-k ICL in our commonly used datasets to further verify the effects of MSIT.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b32e/b32e1b55-ede2-40f2-b25f-8df7deab6e75.png" style="width: 50%;"></div>
Figure 4. The perplexity of OPT-350M and EVALM-350M on PG19 dataset when the length of input sequence scaled up. The extrapolation of OPT starts from 2048 and others from 8192
this situation, MSIT+ still shows a positive effect on most of the tasks, showing the considerable effects of more shots instruction tuning.
Besides, Top-k approach, an instance-level ICL algorithm, selects different demonstration examples for each test sample, demanding heavy computation resources in ICL inference. In contrast, the random approach, a dataset-level ICL algorithm, is much cheaper by sharing and caching incremental encoded examples for all the test samples. Thus, we believe that the random approach or advanced datasetlevel ICL algorithms are more compatible and promising to LRLMs.
Efficiency We test the efficiency of EVALM with training FLOPs and inference times, which are considered crucial for PLMs in upstream training and downstream usage. As shown in Table 4, EVALM can achieve better in-context learning performance with OPT in the same size with even lower training costs. This is due to the efficiency of the causal EVA and our deduplicated training data. Compared with pre-training, the cost of instruction tuning is significantly lower, making it a more easily adopted way to improve the in-context learning performance of PLMs.
<div style="text-align: center;">Table 4. Training FLOPs of different models</div>
Table 4. Training FLOPs of different models
Models
Vanilla
w/ MSIT
OPT-350M
3.84E+20 1.60E+18
EVALM-350M 2.80E+20 1.15E+18
OPT-1.3B
1.42E+21 5.94E+18
EVALM-1.3B
1.03E+21 4.27E+18
As for inference efficiency, according to § 3.3, with incremental encoding and the reuse of incremental states, the additional cost of k-shot ICL when scaling up k in EVALM is relatively low in many-shot settings. Figure 5 illustrates the time consumption of EVALM-350M for each test sample along with the number of shots k, and the results are conducted on SST-5 averaged over 1000 samples. It indicates that without the reuse of incremental states, the overheads grow rapidly while reusing saves redundant computation. The consumption of first encoding long-range demonstrations is diluted by the number of test samples.
# 5. Conclusions & Future Work
The under-investigated pre-trained long-range language model limits the exploration of more shots instruction tuning and in-context learning. In this work, we first pre-train a casual language model EVALM based on an efficient attention mechanism EVA, successfully enabling training with 8k tokens and extrapolating with 256k-length contexts. With techniques such as incremental encoding for efficiency and
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/64d0/64d04661-a9f5-4f50-82a4-e340cf1d7217.png" style="width: 50%;"></div>
Figure 5. Inference time for each sample on SST-5 with or without the reuse of incremental states. 2000 shots corresponds to 58k input length for this dataset
circular position embedding for extrapolation, we consequently inspect the effectiveness of increasing the shot number of both instruction tuning and in-context learning using EVALM. Experimental results across a variety of tasks show EVALM with many-shot instruction tuning and plus outperforms the same size OPT by 4.1% accuracy on average. Interestingly, we find that many-shot instruction tuning can help ICL achieve higher performance with larger demonstrations, and with longer instructions, this phenomenon is more obvious. Notably, such many-shot ICL, with incremental encoding and caching, demands rare extra computational overheads.
EVALM takes the first step towards many-shot in-context learning with pre-trained long-range language models, but it still has several limitations. First, due to our limited computational resources, the experimented EVALM is relatively small in model size compared to existing large-scale language models, e.g. GPT, OPT and PaLM. We will actively work on scaling up its capacity, and it would be interesting to expect its performance on larger LRLMs. Second, although the backbone attention model EVA is efficient and competitive with vanilla attention, it still struggles to scale to longer sequence modeling, due to its quadratic complexity to sequence length in causal language modeling. We will improve LRLMs with linear attention mechanisms to further scale up the reachable length of contexts. Third, when scaling up in-context examples, EVALM is incapable of gaining performance from marginal ones, consistently. We will explore new many-shot in-context learning algorithms that consistently gain performance from the increasing number of in-context examples.
# 6. Acknowledgements
We thank Lin Zheng for proposing the state-of-art efficient attention EVA and providing a well-designed codebase. This work is partially supported by the Shanghai Committee of Science and Technology (Grant No. 21DZ1100100) and the joint research scheme of the National Natural Science Foundation of China (NSFC) and the Research Grants Council (RGC) under grant number N HKU714/21.
# References
Ainslie, J., Ontanon, S., Alberti, C., Cvicek, V., Fisher, Z., Pham, P., Ravula, A., Sanghai, S., Wang, Q., and Yang, L. ETC: Encoding long and structured inputs in transformers. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 268–284, Online, 2020. Association for Computational Linguistics.
Anonymous. Efficient attention via control variates. In Submitted to The Eleventh International Conference on Learning Representations, 2023. under review.
Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. ArXiv preprint, abs/2004.05150, 2020.
Black, S., Biderman, S., Hallahan, E., Anthony, Q., Gao, L., Golding, L., He, H., Leahy, C., McDonell, K., Phang, J., Pieler, M., Prashanth, U. S., Purohit, S., Reynolds, L., Tow, J., Wang, B., and Weinbach, S. GPT-NeoX20B: An open-source autoregressive language model. In Proceedings of BigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models, pp. 95–136, virtual+Dublin, 2022. Association for Computational Linguistics.
rown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020a.
rown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020b.
Chen, Y., Zeng, Q., Ji, H., and Yang, Y. Skyformer: Remodel self-attention with gaussian kernel and nystr\”om method. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems, 2021.
Choi, E., Jo, Y., Jang, J., and Seo, M. Prompt injection: Parameterization of fixed inputs. ArXiv preprint, abs/2206.11349, 2022.
Choromanski, K. M., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarl´os, T., Hawkins, P., Davis, J. Q., Mohiuddin, A., Kaiser, L., Belanger, D. B., Colwell, L. J., and Weller, A. Rethinking attention with performers. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.
Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. Palm: Scaling language modeling with pathways. ArXiv preprint, abs/2204.02311, 2022.
Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, E., Wang, X., Dehghani, M., Brahma, S., Webson, A., Gu, S. S., Dai, Z., Suzgun, M., Chen, X., Chowdhery, A., Narang, S., Mishra, G., Yu, A., Zhao, V., Huang, Y., Dai, A., Yu, H., Petrov, S., Chi, E. H., Dean, J., Devlin, J., Roberts, A., Zhou, D., Le, Q. V., and Wei, J. Scaling instruction-finetuned language models. ArXiv preprint, abs/2210.11416, 2022.
Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2924–2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
Cobbe, K., Kosaraju, V., Bavarian, M., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. ArXiv preprint, abs/2110.14168, 2021.
Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and R´e, C. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022a.
Dong, Q., Li, L., Dai, D., Zheng, C., Wu, Z., Chang, B., Sun, X., Xu, J., Li, L., and Sui, Z. A survey for in-context learning. ArXiv preprint, abs/2301.00234, 2023.
Gupta, A., Gu, A., and Berant, J. Diagonal state spaces are as effective as structured state spaces. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), Advances in Neural Information Processing Systems, 2022.
Ho, J., Kalchbrenner, N., Weissenborn, D., and Salimans, T. Axial attention in multidimensional transformers. ArXiv preprint, abs/1912.12180, 2019.
Human Language Technologies, Volume 1 (Long Papers), pp. 252–262, New Orleans, Louisiana, June 2018. Association for Computational Linguistics.
Levesque, H. J., Davis, E., and Morgenstern, L. The winograd schema challenge. In Proceedings of the Thirteenth International Conference on Principles of Knowledge Representation and Reasoning, KR’12, pp. 552–561. AAAI Press, 2012. ISBN 9781577355601.
Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., and Chen, W. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pp. 100– 114, Dublin, Ireland and Online, 2022. Association for Computational Linguistics.
Liu, P. J., Saleh, M., Pot, E., Goodrich, B., Sepassi, R., Kaiser, L., and Shazeer, N. Generating wikipedia by summarizing long sequences. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018.
toolkit for sequence modeling. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), pp. 48–53, Minneapolis, Minnesota, 2019. Association for Computational Linguistics.
Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., and Potts, C. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pp. 1631–1642, Seattle, Washington, USA, October 2013. Association for Computational Linguistics.
Tay, Y., Bahri, D., Yang, L., Metzler, D., and Juan, D. Sparse sinkhorn attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 1318 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pp. 9438–9447. PMLR, 2020.
Tay, Y., Tran, V. Q., Dehghani, M., Ni, J., Bahri, D., Mehta, H., Qin, Z., Hui, K., Zhao, Z., Gupta, J., et al. Transformer memory as a differentiable search index. ArXiv preprint, abs/2202.06991, 2022.
Zhang, J., Jiang, S., Feng, J., Zheng, L., and Kong, L. Cab: Comprehensive attention benchmarking on long sequence modeling. ArXiv preprint, abs/2210.07661, 2022a.
Zhang, X., Zhao, J., and LeCun, Y. Character-level convolutional networks for text classification. In Cortes, C., Lawrence, N., Lee, D., Sugiyama, M., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015.
Zheng, L., Wang, C., and Kong, L. Linear complexity randomized self-attention mechanism. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings
# A. Pre-training Details
We build the pre-training corpus based on the Pile (Gao et al., 2020), and the pipeline includes filtering, deduplicating,  blending.
Filtering Many of our content filtering strategies were inspired by the data preparation pipeline of BLOOM (Scao et al., 2022) model.2 We filtered raw data from the Pile, including catalog and content filtering. For the catalog filtering, the pre-training corpus contains a subset of the Pile, including BookCorpus2, Books3, DM Mathematics, Project Gutenberg, HackerNews, OpenSubtitles, OpenWebText2, Pile-CC, USPTO, and Wikipedia. We exclude the other subsets of the Pile. On the one hand, based on this project’s scope, we aim to demonstrate our model on the general natural language tasks, and the other domain-specific subsets of the Pile are unsuitable for this purpose. On the other hand, these subsets are relatively noisy, which increases the difficulty and instabilities of the pre-training process, according to the tendency to cause spikes in gradient norms (Zhang et al., 2022a). For the content filtering, we first modified the raw data by standardizing the whitespace and removing the non-ASCII characters. Then we filtered the text documents on (1) the flagged harmful words, (2) the stop word ratio, (3) the word/character repetition ratio, and (4) the specific character ratio.
For the content filtering, we first modified the raw data by standardizing the whitespace and removing the non-ASC characters. Then we filtered the text documents on (1) the flagged harmful words, (2) the stop word ratio, (3) th word/character repetition ratio, and (4) the specific character ratio.
Deduplicating We opted to take the fuzzy deduplication inspired by previous works (Zhang et al., 2022b; Smith et al., 2022). In our implementation, we calculated the mini-hashes and performed LSH using datasketch3, computed the connected components using scipy4, cached the hash fingerprint using Redis5. We first whitespace-tokenized the documents into words and vectorized the documents with the 1-gram language model. Then we calculated the mini-hashes of the document vectors to obtain the document fingerprints with 100-bit hash length. We perform Locality Sensitive Hashing (LSH) through all the document fingerprints to find the neighborhoods of each document with a Jaccard similarity larger than 0.95. After that, we constructed a sparse graph with each document as a node and connected the nodes with their neighborhoods. In this way, we can find the sets of near-duplicated documents by computing the connected components of the graph. Finally, we selected the high-quality documents from each set and removed the other documents in the order of predefined priority. After the filtering and deduplication, we blended the filtered data into heterogeneous batches to obtain the final pre-training corpus. The details are shown in Table 5.
<div style="text-align: center;">Table 5. Number of tokens per dataset in the final pre-training corpus</div>
ble 5. Number of tokens per dataset in the final pre-trainin
Datasets
Tokens (billion)
BookCorpus2
1.6
Gutenberg (PG-19)
3.0
Wikipedia (en)
12.1
OpenWebText2
15.7
Books3
26.0
Pile-CC
52.2
DM Mathematics
3.8
HackerNews
1.1
OpenSubtitles
1.6
USPTO Backgrounds
4.0
Total
121
A.2. Training Details
We pre-trained EvaLM based on metaseq6, the pre-training hyperparameters are listed in Table 6.
<div style="text-align: center;">Table 6. Hyperparameters used for pre-training</div>
Table 6. Hyperparameters used for pre-training
Hypermeters
EVALM-350M
EVALM-1.3B
Dropout
0.1
Weight Decay
0.1
Clip Norm
1.0
Clip Norm Type
L2
LR Schedular
Polynomial decay
Learning Rate
8e-5
Global Batch Size
64
128
DDP Backend
DDP
FSDP
# B. Instruction Tuning Details
We mainly follow settings in FLAN (Wei et al., 2021) to conduct ICL experiment. FLAN dataset consists of 12 dataset clusters including 9 NLU clusters and 3 NLG clusters. As we treat Agnews as a classification task, we only block out this dataset itself rather than the whole summarization cluster. The training hyperparameters are the same in the pre-training stage. We train all models for 5 epochs on selected FLAN datasets to get a fair comparison between IT, MSIT, and MSIT+ during the instruction tuning stage.
# C. In-Context Learning Results
# C.1. In-context learning details
We conduct in-context experiments with 0, 1, 3, 4, 8, 16, 32, 64, 80, 128, 192, 256, 372, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2000 shots considering the limited computing resources. We compare the memory consumption for EVALM-350M and OPT-350M on single NVIDIA 80G A100 in Figure 6.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/808b/808bdedf-b578-4c0f-b6f7-bdac138f6319.png" style="width: 50%;"></div>
We provide supplementary results for the many-shot setting in Table 7, which is the shot number of demonstrations when the best score is achieved for each dataset respectively.
# C.2. Prompt Template
For the sake of reproduction, we list the prompt template and label mapping used in our experiments for different tasks. We refer to templates protocol used in GPT3 and other works (Wu et al., 2022).
<div style="text-align: center;">ble 7. Supplementary results for many-shot setting: the shot number of demonstrations when the best score is achieved for each datas</div>
Models
Sentiment
NLI
Miscellaneous
Reading
Topic
Coreference CMS
SST-2 SST-5 MNLI Trec
WiC
MultiRC BoolQ AgNews
WSC
COPA
OPT350M
1
1
80
1
1
1
3
8
4
1
EVALM350M
1
4
372
372
128
8
4
64
16
1
w/ MSIT
16
8
512
1280
128
16
3
80
256
64
w/ MSIT+
16
4
1280
372
128
64
3
80
8
128
OPT1.3B
8
16
80
16
1
1
4
8
4
1
EVALM1.3B
192
16
256
256
192
8
128
64
372
16
w/ MSIT
192
16
1280
256
192
16
128
64
512
16
<div style="text-align: center;">Table 8. Prompt template and label mapping in our experiment</div>
Dataset
Template
Labal Space
SST-2
{Label} Movie Review: {Sentence}
Negative / Positive
SST-5
{Sentence} It is {Label}
terrible / bad / okay / good / great
MNLI
{Premise}?{Label}, {Hypothesis}
No / Maybe / Yes
Trec
{Sentence} It is about {Label}
abbreviation / entity / description and abstract
concept / human being / location / numeric value
WIC
{Sentence1}\n {Sentence2}\n
question: Is the word {Word} used in the
same way in the two sentences above?\n
answer: {Label}
no / yes
MultiRC
Context: {Paragraph}\n\n {Questions}\n
{Label} answer: {Answer}
incorrect / correct
BoolQ
Context:{Passage}\n Question: {Question}?\n
answer: {Label}
no / yes
AgNews
{Sentence} It is about {Label}
world / sports / business / technology
WSC
{Paragraph}\n Question: In the passage above,
what does the pronoun {Span2} refer to?\n
Answer:{Span1} This is a {Label} answer.
false / true
COPA
Context: {Premise}\n
Correct Answer: {Choices}
false / true
