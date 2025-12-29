# Theoretical Understanding of In-Context Learning in Shallow Transformers with Unstructured Data
Yue Xing
Department of Statistics and Probability
Michigan State University
xingyue1@msu.edu
Xiaofeng Lin
Department of Statistics
University of California Los Angeles
bernardo1998@g.ucla.edu
Chenheng Xu
Department of Statistics
University of California Los Angeles
chenhengx0101@g.ucla.edu
Namjoon Suh
Department of Statistics
University of California Los Angeles
namjsuh@ucla.edu
Qifan Song
Department of Statistics
Purdue University
qfsong@purdue.edu
Guang Cheng
Department of Statistics
University of California Los Angeles
guangcheng@ucla.edu
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/20f5/20f5692a-46bc-4388-b753-47f233b66abd.png" style="width: 50%;"></div>
# Abstract
Large language models (LLMs) are powerful models that can learn concepts at the inference stage via in-context learning (ICL). While theoretical studies, e.g., Zhang et al. [2023], attempt to explain the mechanism of ICL, they assume the input xi and the output yi of each demonstration example are in the same token (i.e., structured data). However, in real practice, the examples are usually text input, and all words, regardless of their logic relationship, are stored in different tokens (i.e., unstructured data Wibisono and Wang [2023]). To understand how LLMs learn from the unstructured data in ICL, this paper studies the role of each component in the transformer architecture and provides a theoretical understanding to explain the success of the architecture. In particular, we consider a simple transformer with one/two attention layers and linear regression tasks for the ICL prediction. We observe that (1) a transformer with two layers of (self-)attentions with a look-ahead attention mask can learn from the prompt in the unstructured data, and (2) positional encoding can match the xi and yi tokens to achieve a better ICL performance.
# 1 Introduction
Large Language Models (LLMs), built upon transformer architectures, have experienced a surge in popularity in recent years, demonstrating their superior power in answering human questions and finishing various tasks. In addition to the extensive empirical studies, theoretical works started to investigate the behavior of LLMs recently. While LLMs share similar properties as other deep neural networks, LLMs still enjoy some unique characteristics, e.g., in-context learning (ICL). Numerous studies in the literature investigate transformers to explain the success of ICL. For example, Zhang et al. [2023] explores the convergence of transformers and their ICL ability using transformers with one linear attention layer and structured data. Structured data means that the input
where (xi, yi)s are in-context examples, and xq is the target input. After feeding E1 into a welltrained transformer, the transformer can compare the similarity between xq and other xis and output a prediction of yq via assembling yis, i.e., conduct an in-context learning directly in the inference stage without learning the relationship between xq and yq in the training stage. In later works, this theoretical framework is extended to softmax attention and multi-head attention, e.g., Zhang et al. [2023], Huang et al. [2023], Cui et al. [2024], Chen et al. [2024]. However, there are some missing pieces in the study of Zhang et al. [2023], Huang et al. [2023], Cui et al. [2024], Chen et al. [2024]: Separating columns of xi and yi In real practice, when inputting the examples, e.g., sentences, each word is a token, and the whole sentence will take up several columns in the input embedding matrix. In contrast, the prompt format E1 merges every information of one example into one column, which is not realistic. Observing this gap between E1 and the common practice, we step one way from E1. Similar to the implementation of Garg et al. [2022] and the discussion of Wibisono and Wang [2023], we aim to train a transformer to conduct ICL using the following prompt format with “unstructured” data:
Separating columns of xi and yi In real practice, when inputting the examples, e.g., sentences, each word is a token, and the whole sentence will take up several columns in the input embedding matrix. In contrast, the prompt format E1 merges every information of one example into one column, which is not realistic. Observing this gap between E1 and the common practice, we step one way from E1. Similar to the implementation of Garg et al. [2022] and the discussion of Wibisono and Wang [2023], we aim to train a transformer to conduct ICL using the following prompt format with “unstructured” data: � �
In Garg et al. [2022], Wibisono and Wang [2023], it is empirically observed that deep transformers are able to learn from unstructured data. In contrast, we explicitly show the critical difference between one and two attention layers, and provide mathematical intuitions that the attention mask together with two attention layers plays an important role.
Other components of transformers While Zhang et al. [2023], Huang et al. [2023], Cui et al. [2024], Chen et al. [2024] studies the performance of a single-layer transformer, they do not consider other important components: attention mask and positional encoding (PE). While existing literature mainly focuses on largescale experiments to demonstrate the effectiveness of LLMs in learning from structured/unstructured data, in our paper, observing the above missing analysis, we consider small-scale shallow transformers and control the experiment settings to explicitly show the benefit and interaction of each component in the transformer. In particular, our key observations are summarized in Figure 1. The contributions are as follows:
Other components of transformers While Zhang et al. [2023], Huang et al. [2023], Cui et al. [2024], Chen et al. [2024] studies the performance of a single-layer transformer, they do not consider other important components: attention mask and positional encoding (PE).
While existing literature mainly focuses on largescale experiments to demonstrate the effectiveness of LLMs in learning from structured/unstructured data, in our paper, observing the above missing analysis, we consider small-scale shallow transformers and control the experiment settings to explicitly show the benefit and interaction of each component in the transformer. In particular, our key observations are summarized in Figure 1. The contributions are as follows:
• As in Figure 1, for the unstructured data E2, a one-layer transformer fails. With a two-layer transformer and a look-ahead attention mask, the transformer can conduct ICL. Both the twolayer structure and attention mask are crucial. (Section 4) • With a two-layer transformer and attention mask, PE can further improve performance. Based on our theoretical results, PE can better match xi and yis in the first attention layer to transform the unstructured data to structured data. A larger input embedding dimension and multiple layers of attention can also help improve the matching. Then, in the second attention layer, one can use structured data to perform ICL. (Section 5)
• As in Figure 1, for the unstructured data E2, a one-layer transformer fails. With a two-layer transformer and a look-ahead attention mask, the transformer can conduct ICL. Both the twolayer structure and attention mask are crucial. (Section 4) • With a two-layer transformer and attention mask, PE can further improve performance. Based on our theoretical results, PE can better match xi and yis in the first attention layer to transform the unstructured data to structured data. A larger input embedding dimension and multiple layers of attention can also help improve the matching. Then, in the second attention layer, one can use structured data to perform ICL. (Section 5)
Note that our study doesn’t claim the superiority of unstructured data over structured data. Based on our understanding, the transformer needs to match (xi, yi)s, i.e., process the unstructured data into structured data in order to perform ICL. From this perspective, structured data is actually preferred
(1)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0c74/0c74cc1d-15e8-42b7-b2a0-50269b9425cc.png" style="width: 50%;"></div>
Figure 1: ICL performance with different number of layers, mask, position. The twolayer structure and the attention mask are crucial, while positional encoding significantly improves performance.
to unstructured data. However, the study of unstructured data is still necessary because it is not avoidable in real practice. For instance, text prompts are usually not curated and contain no direct structural/causal information between words. The LLMs need to capture the relationship among the words first, and then perform predictions.
# 2 Related Works
Theoretical studies It is well known that LLMs can perform in-context learning: the LLMs can infer the output based on the input and given examples in the prompt in the testing stage, without being trained using related data in the training stage. Various empirical studies are conducted to understand ICL, e.g., Garg et al. [2022]. Recently, there are some researches studying the theoretical perspectives of ICL. The first direction considers the simple one-attention-layer architecture to study how linear regression and other simple statistics tasks can be performed in such a neural network using structured data. Various theories can be investigated in terms of the expressive power of the transformer, training convergence, and generalization, e.g., Von Oswald et al. [2023], Ahn et al. [2023a], Aky¨urek et al. [2022], Zhang et al. [2023], Ahn et al. [2023b], Zhang et al. [2023], Huang et al. [2023], Han et al. [2023], Cui et al. [2024], Chen et al. [2024], Kim and Suzuki [2024], Nichani et al. [2024]. In addition to softmax attention, the linear attention is one of the proposed methods that has been widely studied in the theoretical literature as well, e.g., Li et al. [2020], Katharopoulos et al. [2020], Shen et al. [2021], Zheng et al. [2022], Liu et al. [2023], Qin et al. [2022]. Transformer Architecture As will be introduced later, there are several components in the transformer architecture considered in our paper: attention mask, multiple layers of attention, PE. Masking is needed to prevent the attention mechanism of a transformer from “cheating” in the decoder when training. While tasks such as translation need the information from the whole sentence, in other tasks, attention mask has shown empirical successes in many studies, specifically in the field of computer vision and multi-modality models Cheng et al. [2022], Fan et al. [2021], Pang et al. [2019], Harley et al. [2017], Song et al. [2020]. Another important component in transformer architecture is its compositional structure of multiple attention layers. We are aware of two works Van Aken et al. [2019], Simoulin and Crabb´e [2021] which provided interesting analysis on the internal layer mechanisms of BERT. Positional encoding (PE), suggested by Vaswani et al. [2017], successfully facilitates the transformers to know the positional information of each word in the sentence. PE originally introduced for language-based models has many variants based on different data modalities; the instances include tree-based data Shiv and Quirk [2019], graph data Br¨uel-Gabrielsson et al. [2022], time series data Nyborg et al. [2022], video data Ning et al. [2020], etc. Nonetheless, there is not much literature on the theoretical understanding of PE. In the context of ICL, Bai et al. [2023] proves transformers with PE can implement more complex ICL procedures involving in-context algorithm selection.
# 3 Notations and Architecture
In the following, we define the data distribution and the transformer to be considered.
# In the following, we define the data distribution and the transformer to be considered.
Data In this study, we assume all examples and the final query have an x following i.i.d. N(0, Id). The response y is a noiseless linear model, y = x⊤θ, for some θ ∼N(0, Id/d). All examples and the query share the same θ in the same prompt. For different prompts, θs are different. This data assumption is widely used in existing theoretical literature Zhang et al. [2023], Huang et al. [2023], Cui et al. [2024]. Since our focus on the role of PE rather than the relationship between data distribution of ICL ability, we do not consider the other complex relationship in Chen et al. [2024]. To train the transformer, we assume there are infinitely many samples of prompts with their examples and minimize the following loss
 � � Transformer architecture When considering a transformer with two layers of attention, we consider the following structure. In the first attention layer, we use a softmax (self-)attention with a look-ahead attention mask. In particular, define O as the output of the first attention layer, we have f(E) := O := WinE + P + WV,1(WinE + P)ϕ � (WinE + P)⊤WKQ,1(WinE + P) + M � , where P ∈Rp×(2D+1) is the PE matrix, E ∈E(d+1)×(2D+1) is the input matrix, Win ∈Rp×(d+1) is an embedding matrix mapping (xi, 0) and (0, yi) columns of E into a p-dimensional space. The function ϕ is a column-wise softmax operator, i.e., for a vector v, ϕ(v) = exp(v)/∥exp(v)∥1. In addition, the attention mask matrix M is an upper-triangular matrix, with all non-zero elements as −∞. In the second layer, we further pass O into another attention layer to obtain the corresponding output O2, and finally output Wout(O2):,2D+1, a linear function of the last column of O2, as the predicted yq. The corresponding parameters in the second layer are WKQ,2 and WV,2. In the second layer, we use linear attention to simplify the derivations, and its detailed architecture design can be found in the corresponding theorem. The setup of one-layer transformers is similar, with details postponed to the corresponding theorem. In the attention mechanism, the quantities calculated from ϕ are called the attention scores. When ϕ is the column-wise softmax operator, for each column of
 � � Transformer architecture When considering a transformer with two layers of attention, we con sider the following structure. In the first attention layer, we use a softmax (self-)attention with  look-ahead attention mask. In particular, define O as the output of the first attention layer, we have
ϕ � (WinE + P)⊤WKQ,1(WinE + P) + M � ,
� � all the elements are non-negative, and sum up to 1. As a result, when multiplying the WV,1(WinE + P), it can be viewed as a weighted average of the different columns, i.e., WV,1(WinE + P)ϕ:,i = �ϕj,i(WV,1(WinE+P))j, which can be viewed as a kernel method. Similarly, for linear attention, e.g., Zhang et al. [2023], we obtain a linear kernel.
Experiment Settings Throughout this paper, we modify the implementation of Garg et al. [2022] to examine the effect of the different components in the transformer. In the tasks, we take x dimension as d = 5 and the number of examples in each prompt as D = 30. We train different models for 50k to ensure the convergence of optimization and the ICL prediction performance no longer changes. A bad performance under such a large number of steps implies the failure of convergence, e.g., Figure 2. Table 2 in the appendix summarizes how to control each component in the transformer. In addition, for all experiments, we show the result for one repetition in the main content, and postpone the ten-repetition results in Appendix F. In order to correctly measure the effect of PE and mask, we use the following procedure to calculate the testing loss. In the first step, to predict y1 using x1, we only include (x1, 0) in the prompt. To predict y2 using (x1, 0), (0, y1) and (x2, 0), we take E2 = ((x1, 0), (0, y1), (x2, 0)). With an attention mask, the training and testing loss is reduced to the same formula. Without an attention mask, during training, PE will “steal” information from the later yi tokens and achieve an almostzero loss. However, such a strategy does not help prediction in the inference phase when the input E2 does not contain the corresponding yis.
# 4 Two-Layer Transformer + Attention Mask
In this section, we demonstrate that the two-layer structure and the attention mask together play a crucial role in ICL when the examples are in a format of E2. We do not consider PE in this section.
# 4.1 Empirical Observation
Under E2, it is crucial that the transformer architecture is capable of connecting each yi with its corresponding xi. This goal cannot be achieved without any positional information built into the model. We find that the transformer can connect xis to yis when there is more than one layer of attention, along with the attention mask. Figure 2 demonstrates our preliminary experiment results. We take xi dimension d = 5, embedding dimension p = 64, and number of heads h = 1. We use E2 as the input to the transformer and examine the ICL performance. Figure 2 considers transformers of one/two layers of attention,
Under E2, it is crucial that the transformer architecture is capable of connecting each yi with its corresponding xi. This goal cannot be achieved without any positional information built into the model. We find that the transformer can connect xis to yis when there is more than one layer of attention, along with the attention mask.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2dfd/2dfd34a2-ee57-41d8-b817-2b561a0519fa.png" style="width: 50%;"></div>
with/without attention mask, and without PE. One can see that when using a two-layer transformer together with the attention mask, the prediction loss quickly drops when increasing the number of examples D, while missing either the second layer or the attention mask results in a failure of ICL.
# 4.2 Why One-Layer Transformer Fails
# In the following, we explain the poor performance of single-layer transformers
In the following, we explain the poor performance of single-layer transformers. Intuitively, �yq is a weighted average of the last entry across all tokens in the prompt. In a simplified scenario where W V p,: = (0, . . . , 0, 1), this becomes �yq = �D i=1 0 · ϕ2i−1,D + �D i=1 yi · ϕ2i,D, where ϕk,D is the attention score of xq with the k-th token in the prompt. One attention layer, regardless of h, aims to capture the column dependency of its input prompt. Therefore, given a single layer of attention, (i) ϕ2i,D’s can only capture the dependency information between (0, yi) tokens and (xq, 0) and fails to deliver any useful in-context knowledge to �yq; (ii) ϕ2i−1,D’s do capture similarity information about xi’s and xq but cannot pass this information to �yq as they are multiplied by 0’s. This leads to the failure of �yq prediction.
#  �  � The following theorem converts the above intuition into a theorem:
for some A ∈Rd×d and b ∈Rd such that ∥A∥and ∥b∥are both bounded. Assume D →∞, then regardless of whether the attention mask is in the transformer or not, the optimal solution of b satisfies ∥b∥2 →0, and the minimal value of E(�yq −yq)2 is in Θ(1).
� The proof of Theorem 1 can be found in Appendix A. The key idea to prove Theorem 1 is that, when taking θ and −θ, the corresponding �yqs are similar. Thus, the optimal solution is that �yq ≈0.
#  � 4.3 Why Two-Layer Transformer Succeeds
To explain why two-layer transformer + attention mask facilitates ICL in E2, the following theorem demonstrates that even without PE, the transformer is able to perform ICL using unstructured data: Theorem 2 (Two layers with attention mask facilitate ICL). Assume there is no Win. Assume WKQ,1 is a zero matrix, and WV,1 = Id+1. With the attention mask, the output f(E2) of the first layer is
the second layer, considering a linear attention without mask and taking
ϕ � E⊤(W K)⊤W QE � :,2D+1 = 1 log(D)E⊤ � Id 0 0 0 �� xq 0 � ,
� with ϵ/ log(D) = Op( � d/ log(D)), i.e., the second layer is able to predict yq. The notation ϵ represents some cross terms of x⊤ q xix⊤ j θ, which is negligible when D is large enough.
The proof of Theorem 2 can be found in Section B in Appendix.
To further justify Theorem 2, we also conduct the corresponding simulations. We fix the first layer WKQ,1 to be zero, and train the other parameters in the transformer. From Figure 3, there is only a slight performance difference regarding whether WKQ,1 is fixed or not. It implies that the training of the first layer focuses on mixing all columns of E2 together, and as long as the columns of E2 are mixed in an asymmetric way among different is, one can perform ICL in the second layer.
<div style="text-align: center;">Figure 3: ICL prediction performance of training two layers vs training the second layer only.</div>
# 5 Positional Encoding
This section studies the effect of the completely trainable PE. In short, as in Figure 1, given infinite training prompts, the ICL performance is improved by PE.
# 5.1 Intuition: Why Positional Encoding Helps
To explain why PE improves the ICL performance, we draw the heat map of the attention scores of the two attention layers without/with PE (Figure 4 and Figure 5).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/14b5/14b520cd-8556-47a5-b2de-a20ff8e652a8.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8709/8709023b-5050-4931-8cd0-b190dd9b41c9.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Attention scores of the first 5 input pairs on a single head, two layers, with mask, E2 format. One prompt. Each row is the attention of one token. No PE.</div>
For the sake of the readability of the figures, we only plot attention scores for the first 5 pairs (i.e., 10 columns) in Figure 4 and Figure 5. The full plots can be found in Figures 7 and 8 in the appendix. There are two observations from Figure 4 and Figure 5. First, Figure 4 verifies the analysis in Section 4. In the attention score matrix of the first layer (left panel of Figure 4), the attention scores are almost the same along each row. On the other hand, in the second layer, in each row, the high scores often appear on the odd positions (position index starts from 0 in the figures), which contains information of yis. These observations suggest that the first layer aggregates different columns, and the second layer learns from the examples and makes an ICL prediction.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5963/5963bedf-8014-4c43-947d-2b289461d314.png" style="width: 50%;"></div>
Figure 3: ICL prediction performance of training two layers vs training the second layer only.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5d79/5d79085a-99d0-40a7-b69b-a064a42770c0.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Attention scores of the first 5 input pairs on a single head, two layers, with mask, E2 format. One prompt. Each row is the attention of one token. With PE.</div>
Second, in Figure 5, when adding PE, the attention scores in the first layer are almost zero for  odd positions in each row and apply a high weight for xis. Denote the P matrix as � �
where px ij ∈Rd and py ij ∈R when P ∈R(d+1)×D (no Win). Recall that in the formulation, assuming WV,1 = I and no Win, the (2ith column of) the output of the first attention layer becomes � � �
� � � ≤ � Introducing PE helps the attention score matrix focus more on matching xi and yi, instead of naively taking an average to obtain a mixture of different (xj, yj)s. As illustrated by the previous toy example and will be theoretically investigated later, with PE, for each (0, yi) column, the corresponding attention score of (xi, 0) (ϕ2i−i,2i) and (0, yi) (ϕ2i,2i), will be large (around 1/2), while the other attention scores (ϕ2j−1,2i and ϕ2j,2i for j < i) will be almost zero. Therefore, 2ith column of f(E2) reduces to a weighted average of (xi, 0) and (0, yi) (plus some position encoding constants).
# 5.2 Positional Encoding without Input Embedding
When studying the expressive power of PE, since all features in x are symmetric, we employ th following assumption in the transformer parameters to simplify the analysis: Assumption 1. In the first layer, we assume WV,1 = vId+1, and � �
When studying the expressive power of PE, since all features in x are symmetric, we employ the following assumption in the transformer parameters to simplify the analysis:
While Theorem 2 considers a scenario where WKQ,1 is a zero matrix, Assumption 1 allows us to adjust the attention score based on xis and PE. This assumption also aligns with the format of the optimal solution in Zhang et al. [2023], Cui et al. [2024], Huang et al. [2023]. To quantify the benefit of positional encoding, while Garg et al. [2022] considers a completely trainable PE, to simplify the theoretical analysis, we construct the following PE with a trainable scalar c: � �
In the above construction, the PE is the same for the two columns of the same example, and is perpendicular to the other examples.
To validate the effectiveness of the above PE, we conduct a small simulation. We use the implementation of Garg et al. [2022] in the simulation, the transformer has 2 layers, 1 attention head, and is trained for 500k iterations. In each prompt, d = 5 and D = 30. To demonstrate the clear difference among the no PE case, new PE, and the completely trainable PE, we take
p = 64. One can see that the overall MSE of using (3) is close to the completely trainable PE, and can effectively improve the performance compared to the no PE case. Using the above PE construction as in (3), the following theorem demonstrates that the first attention layer with PE successfully matches xis with their corresponding yis. We consider the scenario without the input embedding matrix Win in this section and consider Win in the next section. Theorem 3 (First layer without embedding matrix). When D < d, using the PE construction in (3), taking proper c and vx such that vx = Θ( � log(d)/d) and c = c0 � d log(d) for some large constant c0, with d →∞, with probability tending to 1 over the randomness of the examples (xi, yi)s, the output of the first attention layer combines xi and yi well uniformly, i.e., � � � �
(3)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c41e/c41e69d7-da4c-4475-88f5-f749832e6f9a.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 1: MSE using different PE methods.</div>
� � The notation op represents a negligible term with probability tending to 1 uniformly for all is. The index number starts from 1. Theorem 3 shows that when designing P properly, the attention mechanism helps match xis and their corresponding yis in the first layer. To prove Theorem 3, the key is to use probability bounds to quantify the behavior of the attention scores. The details are in Section C in Appendix. While the above theorem shows how xis and yis are matched in the 2ith column in the first layer, there are two issues. First, the asymptotic of d →∞, i.e., infinite number of features, is impractical. Second, the constraint on D < d also limits the ICL performance. If we do not apply input embedding Win and insist on increasing D, when D ≫d, the any PE does not have enough flexibility to match xis and yis. Since D is much larger than p, some columns of P should be in a similar direction. As a result, PE cannot perfectly match xi and yi in such a direction. The detailed mathematical description can be found as follows: Theorem 4 (PE is not flexible when D ≫p = d + 1). The output of the first layer satisfies � �
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4738/47382e27-9d62-4410-b0b1-919cfe0b0f9a.png" style="width: 50%;"></div>
� �� � � �� � Based on (4) in Theorem 4, when matching xi and yi in the first layer, O:,2i not only contains information about xi, but also the other columns for other xjs with j < i. This implies that for large D, xi and yi cannot be perfectly matched in the first layer. The proof and the condition in PE can be found in Appendix C.
<div style="text-align: center;">� �� � � �� � Based on (4) in Theorem 4, when matching xi and yi in the first layer, O:,2i not only contains information about xi, but also the other columns for other xjs with j < i. This implies that for large D, xi and yi cannot be perfectly matched in the first layer. The proof and the condition in PE can be found in Appendix C.</div>
# 5.3 Benefit of Positional Encoding with Large Input Embedding Dimension
When introducing a large embedding dimension, a large p improves the flexibility of the PE design, allowing us to have more in-context examples. Theorem 5 (First layer with embedding matrix). Under some conditions of Win and WKQ,1, when D < p, and p ≫d, using the PE in (3), taking vx = Θ( � log p/p), c = c0 √p log p for some large enough constant c0, the output of the first attention layer combines xi and yi well, i.e., for i →∞, � � � �
In terms of O:,2i, similarly, �
(4)
From Theorem 5, the output of the first layer can be viewed as structured data with PE as bias. The conditions on Win and WKQ,1 and the proof of Theorem 5 are postponed to the appendix. In the second layer, following Zhang et al. [2023] for linear attention, one can obtain a good ICL performance: Theorem 6 (ICL performance with embedding matrix). Consider the following input for a transformer with one single-head attention layer:
and
where pi = pi1 = pi2 from the designed PE (3).
where pi = pi1 = pi2 from the designed PE (3). Then, there exists some transformer with linear attention such that E(�yq −yq)2 = O(d/D). Remark 1. To simplify our analysis, we use linear attention in the second attention layer to avoi the bias from PE. In the first layer, while successfully matching xi and yi, PE can greatly affec the attention scores because of the nonlinear exp function in the softmax operator. However, in th second layer, PE has no significant effect on the attention score due to the use of linear attention If the second layer is also softmax attention, PE can lead to a bias in the final prediction. In suc a case, the corresponding c in the PE formulation (3) should be smaller to balance the trade-o between the matching effect in the first layer and the ICL prediction performance.
Then, there exists some transformer with linear attention such that E(�yq −yq)2 = O(d/D). Remark 1. To simplify our analysis, we use linear attention in the second attention layer to a
� − Remark 1. To simplify our analysis, we use linear attention in the second attention layer to avoid the bias from PE. In the first layer, while successfully matching xi and yi, PE can greatly affect the attention scores because of the nonlinear exp function in the softmax operator. However, in the second layer, PE has no significant effect on the attention score due to the use of linear attention. If the second layer is also softmax attention, PE can lead to a bias in the final prediction. In such a case, the corresponding c in the PE formulation (3) should be smaller to balance the trade-off between the matching effect in the first layer and the ICL prediction performance.
Due to the page limit, we postpone some additional simulation studies to Section F in Appendix. Briefly speaking, we keep increasing the number of examples D and observe that the ICL performance using small p cannot be as good as the case with large p but small D.
# 5.4 With PE but without Attention Mask
While in the previous sections we explain the importance of attention mask and two attention layers, in this section, we consider the case where no attention mask is applied. Briefly speaking, when PE is applied, even for a one-layer transformer, the minimal training loss is close to zero through stealing the label information from later tokens. On the other hand, in terms of the testing performance, the loss does not converge. Proposition 1. For transformers with one softmax attention layer, following the same condition as Theorem 5, when taking c = c0 √p log p,
From Proposition 1, the information of yi at the 2ith token is leaked to the 2i −1th token, and therefore the training loss can be very small.
We conduct some simulations to examine the ICL prediction performance with PE when changing the number of layers and the attention mask. In Figure 6, with PE, one can see that the ICL prediction loss is small only when there are two layers and the attention mask is applied. Besides Figure 6, we also observe that the training loss for the one-/two-layer transformers with PE without mask are almost zero, i.e., the information of yi at the 2ith token is completely leaked to the 2i −1th token.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c810/c8100e3e-5695-43b1-850a-eaee83cdf012.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
Figure 6: ICL performance of one/two attention layer, with/without mask, with PE.
In this paper, we conduct experiments and explain why transformers can learn the unstructured data E2. Different from the structured prompt format E1 where x and y are in the same column, E2 separates them. When training with infinite prompts, (1) the transformer should be of at least two attention layers with the look-ahead attention mask to learn E2; (2) PE can further connect x and y and balance the mean and variance in the prediction loss, and in addition (3) a high embedding dimension p improves the ICL performance. There are several possible future directions. First, as mentioned in Theorem 6, linear attention in the second layer facilitates ICL using the unstructured data even if PE is amplified in the first layer with softmax attention. Further analysis can be conducted to quantify how softmax attention in the second layer affects the ICL performance. Second, one may also consider the effect of multiple attention layers. In particular, when there are more than two layers, it is not necessary to match the two layers perfectly. One can apply a weaker PE and accumulate its effect through the layers. In addition, as ICL compares the similarity of xq and other xis to determine the weight for yis, this process may also happen in all the attention layers. Detailed quantification of these two effects can deepen the understanding of how multiple attention layers work. Third, another possible direction is to study the effect of PE in the training process. This paper mainly considers PE when training the transformer, with almost infinite examples, and does not consider the generalization gap. One may study the generalization performance given PE. On one hand, for the completely trainable PE, the number of trainable parameters can be significantly increased, and the generalization gap can be larger. On the other hand, from how the PE is applied in the transformer, its effect on the generalization gap may differ from the other parameters. Finally, one may extend the analysis to chain-of-thought, e.g., Li et al. [2024, 2023]. In terms of broader impact, since this paper primarily focuses on theoretical understanding of the mechanism of transformer in learning unstructured data rather than developing new methodologies, there is no direct negative social impact. For limitations, since this paper only considers a simple transformer architecture, it may not capture the full characteristics as the transformers used in real practice. Future works may be considered to generalize the analysis to large-scale transformers.
# References
Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement preconditioned gradient descent for in-context learning. arXiv preprint arXiv:2306.00297, 2023a. Kwangjun Ahn, Xiang Cheng, Minhak Song, Chulhee Yun, Ali Jadbabaie, and Suvrit Sra. Linear attention is (maybe) all you need (to understand transformer optimization). arXiv preprint arXiv:2310.01082, 2023b. Ekin Aky¨urek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. arXiv preprint arXiv:2211.15661, 2022. Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. arXiv preprint arXiv:2306.04637, 2023. Rickard Br¨uel-Gabrielsson, Mikhail Yurochkin, and Justin Solomon. Rewiring with positional encodings for graph neural networks. arXiv preprint arXiv:2201.12674, 2022. Siyu Chen, Heejune Sheen, Tianhao Wang, and Zhuoran Yang. Training dynamics of multi-head softmax attention for in-context learning: Emergence, convergence, and optimality. arXiv preprint arXiv:2402.19442, 2024. Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Maskedattention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022. Yingqian Cui, Jie Ren, Pengfei He, Jiliang Tang, and Yue Xing. Superiority of multi-head attention in in-context linear regression. arXiv preprint arXiv:2401.17426, 2024. Zhihao Fan, Yeyun Gong, Dayiheng Liu, Zhongyu Wei, Siyuan Wang, Jian Jiao, Nan Duan, Ruofei Zhang, and Xuanjing Huang. Mask attention networks: Rethinking and strengthen transformer. arXiv preprint arXiv:2103.13597, 2021. Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. Advances in Neural Information Processing Systems, 35:30583–30598, 2022. Chi Han, Ziqi Wang, Han Zhao, and Heng Ji. In-context learning of large language models explained as kernel regression. arXiv preprint arXiv:2305.12766, 2023. Adam W Harley, Konstantinos G Derpanis, and Iasonas Kokkinos. Segmentation-aware convolutional networks using local attention masks. In Proceedings of the IEEE International Conference on Computer Vision, pages 5038–5047, 2017. Yu Huang, Yuan Cheng, and Yingbin Liang. In-context convergence of transformers. arXiv preprint arXiv:2310.05249, 2023. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR, 2020. Juno Kim and Taiji Suzuki. Transformers learn nonlinear features in context: Nonconvex mean-field dynamics on the attention landscape. arXiv preprint arXiv:2402.01258, 2024. Rui Li, Jianlin Su, Chenxi Duan, and Shunyi Zheng. Linear attention mechanism: An efficient attention for semantic segmentation. arXiv preprint arXiv:2007.14902, 2020. Yingcong Li, Kartik Sreenivasan, Angeliki Giannou, Dimitris Papailiopoulos, and Samet Oymak. Dissecting chain-of-thought: A study on compositional in-context learning of mlps. arXiv preprint arXiv:2305.18869, 2023. Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems. arXiv preprint arXiv:2402.12875, 2024.
Langming Liu, Liu Cai, Chi Zhang, Xiangyu Zhao, Jingtong Gao, Wanyu Wang, Yifu Lv, Wenqi Fan, Yiqi Wang, Ming He, et al. Linrec: Linear attention mechanism for long-term sequential recommender systems. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 289–299, 2023. Eshaan Nichani, Alex Damian, and Jason D Lee. How transformers learn causal structure with gradient descent. arXiv preprint arXiv:2402.14735, 2024. Ke Ning, Lingxi Xie, Fei Wu, and Qi Tian. Polar relative positional encoding for video-language segmentation. In IJCAI, volume 9, page 10, 2020. Joachim Nyborg, Charlotte Pelletier, and Ira Assent. Generalized classification of satellite image time series with thermal positional encoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1392–1402, 2022. Yanwei Pang, Jin Xie, Muhammad Haris Khan, Rao Muhammad Anwer, Fahad Shahbaz Khan, and Ling Shao. Mask-guided attention network for occluded pedestrian detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4967–4975, 2019. Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Lingpeng Kong, and Yiran Zhong. cosformer: Rethinking softmax in attention. arXiv preprint arXiv:2202.08791, 2022. Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, and Hongsheng Li. Efficient attention: Attention with linear complexities. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 3531–3539, 2021. Vighnesh Shiv and Chris Quirk. Novel positional encodings to enable tree-based transformers. Advances in neural information processing systems, 32, 2019. Antoine Simoulin and Benoit Crabb´e. How many layers and why? an analysis of the model depth in transformers. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: Student Research Workshop, pages 221–228, 2021. Kaitao Song, Xiu-Shen Wei, Xiangbo Shu, Ren-Jie Song, and Jianfeng Lu. Bi-modal progressive mask attention for fine-grained recognition. IEEE Transactions on Image Processing, 29:7006– 7018, 2020. Betty Van Aken, Benjamin Winter, Alexander L¨oser, and Felix A Gers. How does bert answer questions? a layer-wise analysis of transformer representations. In Proceedings of the 28th ACM international conference on information and knowledge management, pages 1823–1832, 2019. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, Jo˜ao Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pages 35151–35174. PMLR, 2023. Kevin Christian Wibisono and Yixin Wang. On the role of unstructured training data in transformers’ in-context learning capabilities. In NeurIPS 2023 Workshop on Mathematics of Modern Machine Learning, 2023. Ruiqi Zhang, Spencer Frei, and Peter L Bartlett. Trained transformers learn linear models in-context. arXiv preprint arXiv:2306.09927, 2023. Lin Zheng, Chong Wang, and Lingpeng Kong. Linear complexity randomized self-attention mechanism. In International Conference on Machine Learning, pages 27011–27041. PMLR, 2022.
• Section A: proofs for one-layer transformers. • Section B: proofs for two-layer transformers without PE. • Section C: proofs for for transformers with PE. • Section D: details of configurations of experiments. • Section E: additional figures. • Section F: additional simulations.
Notation-wise, in the proofs, we use “o” to represent some negligible terms which are statistical estimation error such as (�n i=1 xi)/n = E(�n i=1 xi)/n+o, and use “≈” to represent approximation error caused by operators such as �n i=1 1/i ≈log(n).
#  � A One-Layer Transformer without PE
Proof of Theorem 1. We first consider the case with an attention mask. The key idea of the proof is that, since θ ∼N(0, Id/d), we also have −θ ∼N(0, Id/d). When minimizing the loss associated with y = θ⊤x and y = −θ⊤x, the optimal solution is that �y ≈0. When y = θ⊤x, we have
� When D →∞, fixing θ, we have
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/557e/557e3737-36fa-473f-8203-091d5f481e8f.png" style="width: 50%;"></div>
<div style="text-align: center;">The term E(θ) is the generic representation for the term on the denominator. The term ”o” in the above derivation represents some negligible term. In particular, since � �</div>
<div style="text-align: center;">The term E(θ) is the generic representation for the term on the denominator. The term ”o” in th above derivation represents some negligible term. In particular, since</div>
<div style="text-align: center;">The term E(θ) is the generic representation for the term on the denominator. The term ”o” in th above derivation represents some negligible term. In particular, since � exp(x⊤ i Axq) + exp(x⊤ q Axq) + � exp(yib⊤xq)</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f55a/f55aa168-5efe-4e77-bcfc-7da22f34c4dd.png" style="width: 50%;"></div>
� using Taylor expansion, we obtain that
� and it is negligible compared to
� � Similarly, when taking y = −θ⊤x, we have E(θ) = E(−θ). In addition, we again obtain that E �� yi exp(yib⊤xq) � = D∥θ∥2b⊤xq exp �1 2∥θ∥2b⊤xq � .
� � Similarly, when taking y = −θ⊤x, we have E(θ) = E(−θ). In addition, we again obtain tha � � � �
�� � When calculating the prediction loss, we have
to minimize which we need to take ∥b∥→0. Besides xq, the above argument also applies for xis for large is. To analyze the case without attention mask, the different thing compared to the above steps is that we know the later columns in the input, i.e., when y = θ⊤x, �  �
When y = −θ⊤x,
� E With large probability, when ∥b∥is finite, D∥θ∥2b⊤xj exp(∥θ∥2b⊤xj/2) ≫exp(b⊤xjx⊤ j θ), thus the optimal b still satisfies that ∥b∥→0.

# B Two-Layer Transformer without PE
Proof of Theorem 2. In the first layer, since W K = W Q = 0, we have
which is the formula (2) in Section 4.
To obtain (3), in the second layer, taking a linear ϕ and (W K)⊤W Q = � Id/ log(D) 0 0 0 � , the last column is
where
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e293/e2930106-02f8-4fb9-a87c-7165018b331c.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/33a8/33a8c2c5-69f8-4572-8264-0f2995278764.png" style="width: 50%;"></div>
<div style="text-align: center;">� �� While A1 is closely related to xq, A2 is more related to the average of xis. To show that A2 is negligible, we have</div>
 � � � � Since xis are i.i.d. N(0, Id), with probability tending to 1, uniformly for all is,
� � On the other hand, in terms of A1, we have all the elements in A1 are in Op( √ d). As a result, A2 is negligible.
where the term ϵ is the cross term of yix⊤ j xq. In particular,
with
where o represents the difference between �1/i2 and � 1/i2.
When taking the expectation of ϵ, Eϵ = 0. In terms of Eϵ2, we have E(x⊤ i x⊤ j xq)2 = d,
and
When D is large enough, ϵ/ log(D) = Op( � d/ log(D)) P−→0.
# � C Two-Layer Transformer with Attention Mask and PE
# C.1 Theorem 3
Proof of Theorem 3. To prove Theorem 3, since we assume c = c0 � d log(d), the positional encod ing will be the dominate component when determining the attention score matrix. We calculate O1 step by step. For (E + P)⊤WKQ,1(E + P), we have
and

While the above shows that exp(vxc2) dominates the sum of all the other attention scores in the same column, we also have exp(vx∥xi∥2 + vxx⊤ i px i2 + vxc2) = exp(vxc2)(1 + o(1)). As a result, for the columns O:,2is, the attention score for (px i1 + xi, 0) and (px i2, yi) will be both 1/2 + o, and all the other attention scores will be negligible.
# C.2 Theorem 4
Proof of Theorem 4. From the definition of E and P, we have Eϕ � (E + P)⊤WKQ,1(E + P) + M � :,2i = Eϕ � (E + P)⊤WKQ,1(E + P):,2i + M:,2i � = 1 Bi � j≤i �xj + px j1 0 � exp � vx(xj + px j1)⊤(px i2) � + 1 Bi � j<i �px j2 yj � + 1 Bi � px i2 yi � exp(vx∥pi1∥2) where Bi = � j<i � exp(vxx⊤ j px i2) + exp(vx(px j2)⊤px i1) � + exp(vxx⊤ i px i2 + vxc2) + exp(vxc2). When i ≫d, we have � j≤i �xj + px j1 0 � exp � vx(xj + px j1)⊤(px i2) � = � j<i E �xj + px j1 0 � exp � vx(xj + px j1)⊤(px i2) � + � xi + px i1 0 � exp � vx(xi + px i1)⊤px i2 � + � j<i �xj + px j1 0 � exp � vx(xj + px j1)⊤(px i2) � − � j<i E �xj + px j1 0 � exp � vx(xj + px j1)⊤(px i2) � = � j<i �vxpx i2 + px j1 0 � exp �v2 x 2 ∥px i2∥2 + vx(px j1)⊤px i2 � � �� � :=ζ1 + � xi + px i1 0 � exp � vx(xi + px i1)⊤px i2 � � �� � :=ζ2 For simplicity, we assume ∥px jk∥s are all the same for j = 1, . . . , D and k = 1, 2, and ∥px jk∥≤ √ d For ζ2, we have x⊤ i px i2 = Op(∥px i2∥). When i = Θ(D), 1. If ∥px jk∥= o(1), then ζ3 will dominate ζ2.
where
Bi = � � exp(vxx⊤ j px i2) + exp(vx(px j2)⊤px i1) � + exp(vxx⊤ i px i2 + vxc2) + exp(vxc2).
When i ≫d, we have
<div style="text-align: center;">� �� � For simplicity, we assume ∥px jk∥s are all the same for j = 1, . . . , D and k = 1, 2, and ∥px jk∥≤ √ d For ζ2, we have x⊤ i px i2 = Op(∥px i2∥). When i = Θ(D), 1. If ∥px jk∥= o(1), then ζ3 will dominate ζ2.</div>

2. If ∥px jk∥≫1 and vx ≫1, then ζ1 will dominate ζ2 when D ≫d2. 3. If ∥px jk∥≫1 and vx = o(1), then taking px i1 = px i2,
When D ≫exp(d), ζ3 will dominate ζ2.
# C.3 Theorem 5
Proof. From the definition of E and P, we obtain
� Based on the above, we assume WKQ,1 and Win satisfy that
and all the singular values of Win are the same as κ � p/d for some positive number κ. In addition, denote the original PE design in Theorem 3 as P0 with columns p0 jk, then take
Then we obtain
WinEϕ � (WinE + P)⊤WKQ,1(WinE + P) + M � :,2i−1 � � � �
� � − = 1 Ai � j≤i � Win � xj 0 � + pj1 � exp � vxx⊤ i xj + vx(p0 j1)⊤p0 i1 + [xi 0] W ⊤ inW 1/2 KQ,1p0 j1 + [xj 0] W ⊤ inW 1/2 KQ
� � = 1 Ai � � Win � xj 0 � + pj1 � exp � vxx⊤ i xj + vx(p0 j1)⊤p0 i1 + [xi 0] W ⊤ inW 1/2 KQ,1p0 j1 + [xj 0] W ⊤ inW 1/2 KQ,1p0 i1
Eξ1 = � j<i E � Win � xj 0 � + pj1 � exp � vxx⊤ i xj + vx(p0 j1)⊤p0 i1 + √vx [xi 0] W ⊤ inW 1/2 KQ,1p0 j1 + √vx [xj 0] W ⊤ inW 1/2 KQ,1p0 i + � Win � xi 0 � + pj1 � exp � vx∥xi∥2 + vxc2 + 2√vx [xi 0] W ⊤ inW 1/2 KQ,1p0 i1 � = � j<i EWin � xj 0 � exp � vxx⊤ i xj + vx(p0 j1)⊤p0 i1 + √vx [xi 0] W ⊤ inW 1/2 KQ,1p0 j1 + √vx [xj 0] W ⊤ inW 1/2 KQ,1p0 i1 � + � j<i EWinpj1 exp � vxx⊤ i xj + vx(p0 j1)⊤p0 i1 + √vx [xi 0] W ⊤ inW 1/2 KQ,1p0 j1 + √vx [xj 0] W ⊤ inW 1/2 KQ,1p0 i1 �
For the variance, we have
× exp � vxx⊤ i xj + vx(p0 j1)⊤p0 i1 + √vx [xi 0] W ⊤ inW 1/2 KQ,1p0 j1 + √vx [xj 0] W ⊤ inW 1/2 KQ,1p0 i1 � × exp � vxx⊤ i xk + vx(p0 k1)⊤p0 i1 + √vx [xi 0] W ⊤ inW 1/2 KQ,1p0 j1 + √vx [xk 0] W ⊤ inW 1/2 KQ,1p0 i1
× exp2 � vxx⊤ i xj + vx(p0 j1)⊤p0 i1 + √vx [xi 0] W ⊤ inW 1/2 KQ,1p0 j1 + √vx [xj 0] W ⊤ inW 1/2 KQ,1p0 i
× exp � vxx⊤ i xj + vx(p0 j1)⊤p0 i1 + √vx [xi 0] W ⊤ inW 1/2 KQ,1p0 j1 + √vx [xj 0] W ⊤ inW 1/2 KQ,1p0 i1 � × exp � vx∥xi∥2 + vxc2 + 2√vx [xi 0] W ⊤ inW 1/2 KQ,1p0 i1 �
× exp2 � vx∥xi∥2 + vxc2 + 2√vx [xi 0] W ⊤ inW 1/2 KQ,1p0 i1 � .
Then
and
On the other hand, since Ai →EAi, we also have
Therefore,
WinEϕ � (WinE + P)⊤WKQ,1(WinE + P) + M � :,2i−1 = 1 EAi �� Win � xi 0 � + pj1 � exp � vx∥xi∥2 + vxc2 + 2√vx [xi 0] W ⊤ inW 1/2 KQ,1p0 i1 �� = � Win � xi 0 � + pj1 � + o,
In terms of O:,2i, similarly,
# C.4 Theorem 6
Proof of Theorem 6. When
and
� := v1C1 + v2C2 + v3C3 + v4C4 + v5C5.
and
Given the input format of E, the prediction of yq satisfies
� � � � �� = w⊤ v � v1DWinW ⊤ in + v3 � pip⊤ i + v5DWin �0 θ θ⊤ 0 � W ⊤ in �� Win � xq 0 � + pD+1 �
for some wv.
and denoting m = p/d, we have
thus
When taking
we have
On the other hand,
That is, the PE does not results in any bias in the ICL prediction in the second layer, and E(�y yq)2 = O(d/D).
# D Configuration Details
Table 2 below shows how we implement the changes in the components of the transformer.
# Table 2 below shows how we implement the changes in the components of the transformer.
<div style="text-align: center;">Table 2 below shows how we implement the changes in the components of the transformer</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d68c/d68c88a0-8958-4078-9575-2da373f33ce2.png" style="width: 50%;"></div>
# E Other Additional Experiment Results
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8861/88612697-fdfe-4d87-92d7-ccc045d8547c.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Full attention scores on single head, two layer, with mask, no PE, E2 format. There are 30+1 pairs of x and y, thus 31 * 2 = 62 tokens in the prompt. Only attentions score from one example is shown. Each rows is the attention of one token.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/96f2/96f25d44-cbb7-449c-a0de-beff8efcafc2.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
# F Additional Simulations
# F.1 Ten Repetitions
Figure 9 shows the ICL performance of 10 different repetitions under each setting.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fe57/fe57d9d4-a318-49eb-9c18-ee8859a4bc5b.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: Ten repetitions of the simulations for one/two-layer transformers with/without PE and with/without mask.</div>
<div style="text-align: center;">Figure 9: Ten repetitions of the simulations for one/two-layer transformers with/without PE and with/without mask.</div>
In this simulation, we aim to verify the effectiveness of PE given large p, and examine the performance when D keeps growing. We follow the implementation of Garg et al. [2022]1 in this experiment. We take d = 5 and p = 12. We configure different number of examples (D) in the experiment. From Table 3, there are two observations. First, comparing with Table 1, when D = 30, p = 64 gives a better ICL performance. Second, when taking p = 12 and further increasing D to 42, the ICL performance is still worse than Table 1.
Average
Std
D
Train loss
Test loss
Train loss
Test loss
6
0.5987
0.6803
0.0611
0.0153
12
0.4655
0.4936
0.0497
0.0070
18
0.4248
0.4099
0.0584
0.0497
24
0.3940
0.3870
0.1241
0.0984
30
0.3855
0.3913
0.1751
0.1855
36
0.3192
0.3106
0.1314
0.1152
42
0.2228
0.2310
0.0212
0.0068
mance when increasing.,. Benchmark: When
Table 3: ICL performance when increasing D.  D = 30, the test loss is 0.1877 in Table 1.
Table 3: ICL performance when increasing D. d = 5, p = 12. Benchmark: When d = 5, p = 64, D = 30, the test loss is 0.1877 in Table 1.
1https://github.com/dtsip/in-context-learning
