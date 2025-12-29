# Can Large Language Models Explain Themselves? A Study of LLM-Generated Self-Explanations
yuan Huang∗ Siddarth Mamidanna∗ Shreedhar Jangam Yilun Zhou Leilani H. Gilpin  Santa Cruz UC Santa Cruz UC Santa Cruz MIT CSAIL UC Santa Cruz {shuan101, spmamida, sjangam}@ucsc.edu, yilun@csail.mit.edu, lgilpin@ucsc.edu
Abstract—Large language models (LLMs) such as ChatGPT have demonstrated superior performance on a variety of natural language processing (NLP) tasks including sentiment analysis, mathematical reasoning and summarization. Furthermore, since these models are instruction-tuned on human conversations to produce “helpful” responses, they can and often will produce explanations along with the response, which we call selfexplanations. For example, when analyzing the sentiment of a movie review, the model may output not only the positivity of the sentiment, but also an explanation (e.g., by listing the sentimentladen words such as “fantastic” and “memorable” in the review). How good are these automatically generated self-explanations? In this paper, we investigate this question on the task of sentiment analysis and for feature attribution explanation, one of the most commonly studied settings in the interpretability literature (for pre-ChatGPT models). Specifically, we study different ways to elicit the self-explanations, evaluate their faithfulness on a set of evaluation metrics, and compare them to traditional explanation methods such as occlusion or LIME saliency maps. Through an extensive set of experiments, we find that ChatGPT’s selfexplanations perform on par with traditional ones, but are quite different from them according to various agreement metrics, meanwhile being much cheaper to produce (as they are generated along with the prediction). In addition, we identified several interesting characteristics of them, which prompt us to rethink many current model interpretability practices in the era of ChatGPT(-like) LLMs. Index Terms—Machine Learning, Natural Language Processing, Interpretability, Large Language Model (LLM)
I. INTRODUCTION
Over the past couple years, instruction-tuned [30] large language models (LLMs) have revolutionized the field of natural language processing (NLP) with its superior performance on a wide variety of tasks and benchmarks such as sentiment analysis [44], mathematical reasoning [14] and summarization [48]. Commercial models such as ChatGPT and GPT-4 by OpenAI, Bard by Google and Claude by Anthropic have been used to (partially) automate many human labor-intensive processes such as education [22], trip planning [46] and customer service [15]. A notable feature of these models is the ability to provide (seemingly) reasonable explanations. For example, when asked to solve a math problem, they often include detailed derivation steps even when not explicitly instructed to do so. Similarly, when asked to analyze the sentiment of a movie or book review, they often spontaneously explain their decision with supporting evidence, such as by highlighting sentiment-laden words or explaining the sarcasm in the text.
Is this review positive? Why or why not?  “A fantastic movie directed by the famous  Lucas Johnson, who has a track-record of  producing terrific novel adaptions.”
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7d20/7d205132-eb3e-4f3f-a4e3-f1a94f600c48.png" style="width: 50%;"></div>
Fig. 1. An overview of our investigation. Current conversational LLMs can explain their answers (e.g., by highlighting important words in the input), often automatically or at least when asked to. How should we think of these self-explanations? In this paper, we study them in relationship to traditional model interpretability techniques such as occlusion saliency and LIME, and on various metrics such as comprehensiveness, sufficiency and rank agreement. Our findings suggest that we may need to rethink the model interpretability pipeline for analyzing these models.
How should we think about these explanations? In particular, do these explanations faithfully support the model’s prediction (e.g., the final answer of the math problem or the positive/negative judgment in sentiment analysis), or do they simply give a sense of transparency while being decoupled from the prediction? To what extent are they similar to or different from the explanations generated from traditional techniques [e.g. 24, 25, 28, 34, 37, 49], which are commonly used to study predecessor models such as BERT [12] and GPT2 [33]? For any “negative results” about the effectiveness of these explanations, should we attribute them to the model’s inability to generate the explanations, or the limitations in our evaluations [13, 23] about them? In this paper, we seek answers to the above questions, using OpenAI’s ChatGPT as the model, feature attributions as the explanation type and sentiment analysis as the task. All three design choices are made based on popularity in the respective aspect: ChatGPT is arguably the most widely used instructiontuned LLM, feature attribution (also known as saliency map or feature importance) explanation is by far the most popular
type of explanations studied in the interpretability literature, and sentiment analysis is often the first task to consider when benchmarking these explanations due to the intuitive alignment between strong sentiment words (e.g, “fantastic” or “terrible”) with feature attribution scores. In the experiment, we study two paradigms of generating the explanations: generating the explanation first before the prediction (explain-then-predict or E-P), versus generating the prediction first and then explaining it (predict-and-explain or P-E) [7]. In each paradigm, we construct two methods to eliciting the explanations: asking the model to generate a full list of feature attribution explanations containing the importance score of every word, and asking the model to just return the top-k most important words. We then compare them with two classic interpretability methods, occlusion [25] and LIME [34]. The comparison is done on two fronts, a suite of faithfulness evaluation metrics [51] and a set of disagreement measurements among explanations [23]. Quantitatively, the model’s self-explanations perform on par with traditional methods via faithfulness evaluations, but at the same time differ greatly from them (on various agreement metrics). More importantly, we find that the classic interpretability pipeline of defining and evaluating model explanations may be fundamentally ill-suited for these LLMs with quite human-like reasoning ability. Specifically, the model does not produce sufficiently fine-grained model predictions for us to meaningfully compute and evaluate explanations, in ways that we have been done for predecessor models such as BERT [12]. Nonetheless, we believe that this should be considered as a limitation on the current interpretability practice, rather than the model, as many such computations (e.g., identifying the effect of removing a single word in a long sentence) are ill-posed for humans as well. Overall, we hope that this work could bring formal investigations of LLM-generated self-explanations to the attention of the community, and inspire more work towards better understanding and using them.
# II. RELATED WORK
Our research lies at the intersection of three areas within interpretability research: feature attribution explanations, evaluations of feature attribution explanations and LLM-generated self-explanations. The former two areas are well-studied, so we give emphasis to works that are directly used in our paper. For broader contexts and more comprehensive coverage, we refer the reader to the textbook by Molnar [29]. The last area is relatively new, and while, to the best of our knowledge, we are the first to rigorously study LLM-generated feature attribution explanations, we briefly discuss some related ideas.
# A. Feature Attribution Explanations
At a high level, feature attribution explanations assign attribution scores to input features [52]. There are in general two ways of defining input features. The first is based on the model’s response to infinitesimal perturbation of the input feature value, with representative methods such as the gradient saliency [37], smooth gradient [39] and integrated gradient
[41]. For NLP models, the perturbation is often done in the embedding space [24]. However, since ChatGPT is a closedsource model that does not provide input gradients, none of these explanations are possible to calculate. Instead, we focus on the second approach: defining feature importance based on a model’s response to inputs with various features removed. The simplest definition is the occlusion saliency [25, 49], with the importance of a feature (i.e., word) defined as the change in the model prediction on the sentence with that word removed. To capture non-linear interactions among different words, LIME [34] and SHAP [28] remove multiple features at the same time and define feature importance as linear regression coefficients or the game-theoretic Shapley value [35] respectively. The word removal strategy can be influential, with Pham et al. [31] finding that some strategies favor some evaluations. In this paper, we adopt the mainstream approach of word deletion (i.e., removing the word “This” from the sentence “This is a great movie” results in “is a great movie”), and leave the exploration of other strategies (e.g., mask infilling) to future work.
# B. Evaluations of Feature Attribution Explanations
Our main contribution of this paper is an investigation into the relative strengths and weaknesses of LLM-generated self-explanations compared to traditional explanations such as occlusion and LIME. In this paper, we evaluate both their faithfulness and agreement with each other. For faithfulness evaluations, we employ several popular metrics, including comprehensiveness [13], sufficiency [13], decision flip rate under the removal of the most important token/word [10], fraction of tokens need to be removed for decision flip [36] and deletion rank correlation [3]. They all subscribe to a fundamental principle – modifying a significant feature, as identified by the explanation, should markedly influence the prediction – but implement it differently. Furthermore, recognizing that there is no “ground truth” model reasoning available (due to the black-box nature of these models), one type of evaluation modifies the dataset and retrains the model [1, 2, 6, 47, 52], such that the model has to use a specific type of reasoning for high accuracy, and then evaluates whether the saliency maps of such high-accuracy models align with the expectation. Given the high cost and limited flexibility (API access only) of finetuning, we leave these evaluations to future work. We also study the correlation between the LLM selfexplanation and traditional techniques using the set of disagreement metrics introduced by Krishna et al. [23]. Additional manifestation of the disagreement problem includes the finding that the explanations themselves are often unstable [16] and that they are very sensitive to hyperparameter choices [5].
# C. LLM-Generated Self-Explanations
Generative models, those that can generate texts such as the GPT family of models [32, 33], offer a distinct way of explaining them – by simply generating the explanations directly. Camburu et al. [7] trained models on supervised
pairs of labels and explanations and considered two orderings of prediction and explanation: making a prediction and then explaining it, and generating an explanation and using it to make a prediction. We study both approaches in our paper. In addition, Camburu et al. [7] found that training the model to generate the explanation, both before and after the prediction, results in a slight accuracy drop, casting doubt on the inherent usefulness of the explanation to the prediction. For ChatGPT and later models, chain-of-thought generation [45], which can be considered as the model’s explanation, is often helpful for the accuracy of the final answer, especially on complex reasoning tasks such as solving math problems [18]. This evidence implies that the model could meaningfully reason from such explanations. By contrast, [8] studied properties of self-generated counterfactual explanations, proposed several metrics and found that these explanations are often misleading and lead to the wrong mental model of human inspectors.
Our main contribution in this paper is a suite of systematic analyses of LLM-generated self-explanations in the sentiment analysis domain. We first covers the basics of auto-regressive LLMs (such as ChatGPT and similar conversational models), and then describe our prompting strategy and considerations. After that, we introduce the two traditional interpretability methods that we compare to, and also evaluation metrics.
# A. Auto-regressive LLMs
Auto-regressive LLMs, as pioneered by GPT-1 [32] and GPT-2 [33], operate by successively generating the next word (or token) given the history (called context window), until a special stop token is generated. The generation can be
TABLE I THE PROMPT AND AN EXAMPLE QUERY OF THE E-P SETUP. BLACK TEXT IN NORMAL FONT IS PROVIDED LITERALLY. Gray text in italics IS CHANGED FOR EVERY QUERY. BLUE TEXT IN BOLDFACE IS THE MODEL’S OUTPUT.
Role
Text
System
You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. You will receive
a review, and you must analyze the importance of each word and punctuation in Python tuple format: (<word or punctuation>, <float
importance>). Each word or punctuation is separated by a space. The importance should be a decimal number to three decimal places
ranging from -1 to 1, with -1 implying a negative sentiment and 1 implying a positive sentiment. Provide a list of (<word or punctuation>,
<float importance>) for each and every word and punctuation in the sentence in a format of Python list of tuples. Then classify the review
as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose and output the classification and confidence in the
format (<int classification>, <float confidence>). The confidence should be a decimal number between 0 and 1, with 0 being the lowest
confidence and 1 being the highest confidence.
It does not matter whether or not the sentence makes sense. Do your best given the sentence.
The movie review will be encapsulated within <review> tags. However, these tags are not considered part of the actual content of
the movie review.
Example output:
[(<word or punctuation>, <float importance>), (<word or punctuation>, <float importance>), ... ]
(<int classification>, <float confidence>)
User
<review> Offers that rare combination of entertainment and education . <review>
Assistant
[(‘Offers’, 0.500), (‘that’, 0.000), (‘rare’, 0.500), (‘combination’, 0.000), (‘of’, 0.000), (‘entertainment’, 0.750), (‘and’, 0.000),
(‘education’, 0.750), (‘.’, 0.000)]
(1, 1.000)
stochastic or deterministic, and various decoding strategies could be used to improve the quality of the generation [19]. In all of our experiments, we use deterministic greedy decoding (i.e., with temperature = 0). Thus, the generation process can be modeled as the function
(1)
where xc is the sequence of context words, and xg is the sequence of generated words. A conversational LLM (i.e., an LLM that can engage in human conversations) typically has three message types: an initial “system” message to set the general tone and instruction of the conversation, followed by rounds of interleaving “user” and “assistant” messages. While the exact details of ChatGPT is unknown due to its proprietary nature, similar open-source models such as Alpaca [42] and Vicuna [9] all adopt the same practice to represent the message history: by concatenating individual message prepended with the role indicator. For example, if the model (i.e., assistant) needs to answer the user’s second question below:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ab97/ab97226a-8980-4600-be8b-0369705712eb.png" style="width: 50%;"></div>
<div style="text-align: center;">the context xc is constructed to be the following text:</div>
which is a concatenation of all of the past messages.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/60ce/60ce9ecb-f83e-41c0-ad30-f6b99ba3756d.png" style="width: 50%;"></div>
# ‘A’, 0.2), (‘film’, 0.5), (‘that’, 0.1), (‘takes’, 0.3), (‘you’, 0.4), (‘inside’, 0.6), (‘the’, 0.1), (‘rhythms’, 0.7), (‘of’, 0.1), (‘its’, 0.1), subject’, 0.5), (‘:’, 0.1), (‘You’, 0.4), (‘experience’, 0.6), (‘it’, 0.3), (‘as’, 0.2), (‘watch’, 0.4), (‘.’, 0.1)]
# B. Prompting
Our central goal is to have the model generate a prediction on the sentiment of the input sentence and a feature attribution explanation for the prediction. We include both explanation pipeline setups studied by Camburu et al. [7], explain-then-predict (E-P), in which the model first generates the explanation and then arrive at a prediction based on it, and predict-and-explain (P-E), in which the model first generates the prediction and then explains it. Our prompts for E-P and P-E setups are shown in Table I and II respectively. In the design of both prompts, we have several considerations. First, in order to minimize the number of turns in the conversation, we instruct the model to output both the explanation and prediction at the same time, but in clearly defined ordering. Second, in some of our experiments, we need to feed the model with partial inputs (i.e., sentences with some words removed) such as “This is a really great movie with beautiful music and intricate plot” ⇒“This a really movie with music and intricate”. We empirically find that models without any guidance sometimes will refuse to
TABLE III A CHATGPT RESPONSE WITH HUMAN “IN THE WILD,” WHERE CHATGPT AUTOMATICALLY GIVES THE EXPLANATION IN THE FORM OF A FEW HIGHLIGHTED PHRASES.
Role
Text
User
Can you tell me whether the movie review is positive or negative?
The gorgeously elaborate continuation of "The Lord of the Rings" trilogy is so huge that a column of words can not adequately describe
co-writer/director Peter Jackson ’s expanded vision of J.R.R. Tolkien ’s Middle-earth .
System
Based on the given text, it appears to be a positive movie review. The phrases "gorgeously elaborate," "expanded vision," and "Middle-earth"
suggest admiration and appreciation for the film.
provide an answer, so we intentionally ask them to give their best guess despite input ambiguity. Finally, we initially had significant difficulty making the model to output the word attribution values in a consistent format, but in the end find that the “Python” format works the best for the model to follow. Despite the success in generating syntactically correct outputs, a concern with this Python format guiding is that the unnaturalness of output (compared to, e.g., Table III) may result in the explanation values also being less faithful to the model’s true reasoning. Thus, to remove any effect of the unnatural prompt, and recognizing that humans, to which these models are aligned [30], are not be able to produce detailed attribution scores, we additionally consider two top-k versions of the E-P and P-E setups. We construct these prompts to be as natural as possible while ensuring that the result remains easy to parse, and the final prompts are shown in Table IV and V respectively. The number k is chosen dynamically according to each input sentence as min(1, ⌊L/5⌋), where L is the number of words in the sentence. For each prompt, the model generates a decimal point num-
Role
Text
System
As a movie review analyst, your role is to analyze the sentiment of movie reviews and provide insights on the importance of each word
and punctuation in determining the overall positivity level. Your task is to identify the top 3 most significant words, ranked from the
most positive sentiment to the least positive sentiment. Additionally, you need to determine whether the movie review is positive or
negative along with your confidence in your prediction. A positive review is represented by the number 1, while a negative review will
be represented by the number 0. The confidence should be a decimal score between 0 and 1, with 0 being the lowest confidence and 1
being the highest confidence. Please note that the coherence of the sentence is not relevant; your focus should be on analyzing the sentiment.
The movie review will be enclosed within <review> tags, but these tags should not be included in the evaluation of the review’s
content.
Only output the list of 3 words in the form of a comma separated list, with the prediction(as a number) and confidence following
after, nothing more.
User
<review> A film that takes you inside the rhythms of its subject : You experience it as you watch . <review>
Assistant
rhythms, experience, watch, 1, 0.9
Role Text System As a movie review analyst, your role is to analyze the sentiment of movie reviews and provide insights on the importance of each word and punctuation in determining the overall positivity level. Your task is to identify the top 3 most significant words, ranked from the most positive sentiment to the least positive sentiment. Additionally, you need to determine whether the movie review is positive or negative along with your confidence in your prediction. A positive review is represented by the number 1, while a negative review will be represented by the number 0. The confidence should be a decimal score between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Please note that the coherence of the sentence is not relevant; your focus should be on analyzing the sentiment. The movie review will be enclosed within <review> tags, but these tags should not be included in the evaluation of the review’s content. Only output the list of 3 words in the form of a comma separated list, with the prediction(as a number) and confidence following after, nothing more. User <review> A film that takes you inside the rhythms of its subject : You experience it as you watch . <review> Assistant rhythms, experience, watch, 1, 0.9
ber for the prediction confidence. Since we do not have model weight access, we cannot compute the logit or probability of any generated token. Thus, we take this decimal point (in the range of 0.0 to 1.0) as the model’s prediction, which is used to evaluate the model’s accuracy.
# C. Traditional Interpretability Methods
We benchmark the LLM-generated self-explanations with traditional interpretability methods that are often used to explain classifiers. As mentioned in Section II-A, gradientbased attribution values cannot be used due to the lack of input gradient access. Instead, we consider two perturbationbased methods, occlusion [25] and LIME [34]. The occlusion method defines the attribution value on word w as
(2)
where f(·) is the model prediction (as the generated decimal point number in the range of 0.0 to 1.0) for the input sentence, x is the original sentence, and x−w is the input sentence with
MPT AND AN EXAMPLE QUERY OF THE P-E TOP-k SETUP. BLACK TEXT IN NORMAL FONT IS PROVIDED LITERALLY. Gray text in italics IS CHANGED FOR EVERY QUERY. BLUE TEXT IN BOLDFACE IS THE MODEL’S OUTPUT.
Role
Text
System
As a movie review analyst, your role is to analyze the sentiment of movie reviews and provide insights on the importance of each
word and punctuation in determining the overall positivity level. Your task is to determine whether the movie review is positive or
negative along with your confidence in your prediction. A positive review is represented by the number 1, while a negative review will
be represented by the number 0. The confidence should be a decimal score between 0 and 1, with 0 being the lowest confidence and 1
being the highest confidence. In addition, you need to identify the top 3 most significant words, ranked from the most positive sentiment
to the least positive sentiment. Please note that the coherence of the sentence is not relevant; your focus should be on analyzing the sentiment.
The movie review will be enclosed within <review> tags, but these tags should not be included in the evaluation of the review’s
content.
Only output the prediction(as a number) and confidence, with the list of 3 words in the form of a comma separated list following
after, nothing more.
User
<review> A film that takes you inside the rhythms of its subject : You experience it as you watch . <review>
Assistant
1, 0.8, rhythms, experience, watch
the word w removed. Intuitively, the occlusion value captures the effect of a single word removal on model prediction. By comparison, LIME recognizes that the effect of word removal may not be linearly additive (i.e., the total prediction change from removing two words is not the sum of prediction changes from removing each word), and thus proposes to remove random subsets of words, computes the model prediction change, and attributes the change to individual words by solving a linear regression problem.
# D. Evaluation Metrics
For faithfulness evaluations of the self-explanations versus occlusion and LIME, we use five evaluation metrics: comprehensiveness [13], sufficiency [13], decision flip rate under the removal of most important token (DFMIT) [10], minimum fraction of tokens needing to be removed in order to cause decision flip (DFFrac) [36] and rank correlation with model prediction after word deletion (RankDel) [3]. Given a candidate attribution explanation to be evaluated, these metrics produce
a score based on the model predictions (and their changes) on (1) the original input and (2) the input with some words removed according to the explanation. We follow the definition and implementation by Zhou and Shah [51]. As before, the model prediction is taken from the decimal point number in the model’s textual output. During the evaluation, when multiple words have the same saliency value (a feature unique to these LLM-based explanations, elaborated in Section IV-E), we perform random tie-breaking among them to determine the word removal order. To evaluate the top-k explanations (which does not have any ranking among the top-k words or among the remaining words), we adapt comprehensiveness, sufficiency and DFMIT to their top-k versions. Specifically, let x be the original input, x−k be the input with the top-k words removed, and xk be the input with only top-k words kept. Then we define
(3) (4) (5)
where the indicator function in DFMIT@k evaluates to 1 if removing all k words together flips the label, and 0 otherwise. For non-top-k explanations, we can also evaluate them on these metrics by considering the k features with highest explanation values (random tie-breaking if necessary) as the top-k features. Besides these faithfulness evaluations (i.e., computing a metric score for an individual explanation), we also evaluate their (dis)agreement, motivated by Krishna et al. [23] and directly using their definitions.
# E. A Caveat on Model Sameness
Since the model is auto-regressive in nature, its output is affected by two factors: the model weight and the context (i.e., the prompt). Thus, given the same model (e.g., ChatGPT), prompting it in different ways (e.g., via different system messages) effectively leads to different models. Thus, in our
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/03ff/03ff87ab-fd43-4a27-ae3c-08678cc995f1.png" style="width: 50%;"></div>
experiment, we need to treat the E-P and P-E prompt as two different models, and the generated self-explanations are not comparable to each other. In addition, in order to make meaningful comparisons between self-explanation and occlusion/LIME, we need the latter to be generated with the same system prompt, too. Concretely, as an example, for the sentence “I really like this movie”, to obtain the occlusion value for the word “like” that is comparable to E-P self-explanation, we need to compute the model prediction difference on the input “I really like this movie” vs. “I really this movie”, both using the prompt as shown in Table I, even though we do not need the explanation value for the latter sentence. The same reasoning applies on the evaluation side, where in order to retrieve the model prediction values, we need to use the original prompt and ignore the explanation output byproduct. This also means that the top-k E-P and P-E explanations are not those for the E-P and P-E models. However, there is no way around this issue: in order for the model to generate the top-k explanation, it has to be instructed in that way, which effectively results in a different model. Thus, in our study, we ignore the difference, and consider the topk explanation effectively as an explanation generated by an external explainer, for the target E-P or P-E model.
We conduct experiments on the Stanford Sentiment Treebank (SST) dataset [40], which is a collection of short- to medium-length movie reviews, with their associated labels. While the labels are provided as a continuous value between 0.0 (totally negative) and 1.0 (totally positive), we binarize the label at 0.5. Due to the cost and rate limit of the OpenAI’s API access which makes the full set of experiments on a single instance taking around half an hour, we select a random subset of 100 sentences from the test set for our investigations. With this setup, Figure 2 presents the explanations for two sentences. We see that the LLM-generated self-explanations
Prediction Only
92%
E-P
85%
P-E
88%
E-P top-k
80%
P-E top-k
83%
are intuitively reasonable in highlighting words of strong intrinsic sentiment values such as “superbly” and “void”. In addition, there are other features, such as that the explanation values are quite sparse and distributed on a few “levels,” which are discussed further in Section IV-E.
# B. Model Accuracy
Table VI summarizes the classification accuracy of each model. First, we note that the classification accuracy of E-P is lower than that of P-E, at 85% vs 88%. This difference suggests that generating an explanation before making the prediction is not always helpful, unlike the benefits observed in chain-of-thought prompting [45]. We believe that this difference is due to the nature of the tasks: the “reasoning” in sentiment analysis cannot be adequately represented by attribution scores assigned to each word, and hence forcing the model to make a prediction based on preceding explanation (i.e., the E-P setup) hurts the performance. Furthermore, both models have worse performance compared to the model that does not need to generate any explanation, which has an accuracy of 92%, with the system prompt shown below.
You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>).
You are a creative and intelligent movie review analyst, whose purpose is to aid in sentiment analysis of movie reviews. A review will be provided to you, and you must classify the review as either 1 (positive) or 0 (negative), as well as your confidence in the score you chose. The confidence should be a decimal number between 0 and 1, with 0 being the lowest confidence and 1 being the highest confidence. Output this in the Python tuple format (<int classification>, <float confidence>). The movie review will be surrounded by <review> tags. Example output: (<int classification>, <float confidence>)
This result is reminiscent of that by Camburu et al. [7], where asking the model to provide explanations hurts the performance, despite with different mechanisms: prompting in our work and supervised training in the other work. Last, the prediction is even less accurate when it needs to be made based on a few important words (in the E-P top-k setting), or when it needs to be justified later with a few words (in the P-E top-k setting). All thse model accuracy results imply that feature attribution explanations may not be the best form of explanation for sentiment analysis, which forces the
<div style="text-align: center;">TABLE VII COMPARISON BETWEEN THE SELF-EXPLANATION GENERATED BY CHATGPT AND TRADITIONAL EXPLANATIONS COMPUTED ON IT.</div>
Comp ↑
Suff ↓
DFMIT ↑
DFFrac ↓
RankDel ↑
E-P (Accuracy: 85%)
Occlusion
0.15
0.26
0.18
0.57
0.00
LIME
0.17
0.22
0.13
0.50
-0.02
SELFEXP
0.19
0.25
0.16
0.55
-0.03
P-E (Accuracy: 88%)
Occlusion
0.20
0.23
0.14
0.64
-0.02
LIME
0.27
0.20
0.10
0.56
0.02
SELFEXP
0.27
0.22
0.07
0.56
-0.01
model into an uncomfortable accuracy-interpretability tradeoff [17].
# C. Faithfulness Evaluation Results
Table VII presents the performance of SELFEXP for E-P and P-E models, compared to occlusion and LIME generated for each model respectively. We make several remarks below.
First, we note the lack of a clear winner across the different metrics. Unexpectedly, we see that the occlusion saliency performs the best on DFMIT, because its definition (i.e., the importance of a word defined as the model prediction change under word removal) inherently makes its most important word being the most likely to flip the model prediction when removed – an observation of solvability made by Zhou and Shah [51]. We should also expect it to achieve the first place on RankDel, but all the numbers are very close to 0 and, as we will show in Section IV-E, the insensitivity of model prediction under single word removal makes this metric highly prone to the effect of stochasticity in tie-breaking. Among the remaining metrics, occlusion is the best on none, which is not surprising as it could not capture the non-linear effects of feature removals. Such non-linear effects are considered by LIME, which performs quite evenly with SELFEXP. Nonetheless, running LIME is highly expensive: since it needs to solve a linear regression problem, the default number of perturbations by the Python lime package1 is 5,000, meaning that we need to query the OpenAI API 5,000 times in order to generate the explanation for a single sentence. In practice, we adaptively scale this number to be 10 times the number of tokens in the sentence with an average of 183 perturbations per explanation, but this still is quite expensive, taking approximately 20 minutes per explanation. Thus, given that SELFEXP performs on par with LIME, we see that it could be low-cost substitute of LIME, as this selfexplanation is generated along with the prediction. As mentioned in the Section III-B, while the Pythonlike format guiding helps the model to generate syntactically valid explanation and prediction, we are concerned that the explanation quality may suffer. Thus, we also prompt the
<div style="text-align: center;">TABLE VIII COMPARISON OF ALL METHODS WITH THE TOP-k EXPLANATIONS ON TOP-k-ADAPTED METRICS.</div>
Comp@k ↑
Suff@k ↓
DFMIT@k ↑
E-P (Accuracy: 80%)
Occlusion
0.16
0.34
0.29
LIME
0.17
0.28
0.40
SELFEXP
0.11
0.32
0.32
TOPK
0.10
0.27
0.34
P-E (Accuracy: 83%)
Occlusion
0.11
0.29
0.17
LIME
0.18
0.26
0.31
SELFEXP
0.11
0.30
0.29
TOPK
0.11
0.25
0.31
model to generate the top-k most important words, and without attribution values attached to them. Table VIII shows three metrics that are compatible with the top-k explanation, and to compare it with other (fully formed) explanations, we extract the top-k words from the latter. As we can see, the top-k explanation is not distinctively better – while it achieves the best result on some metrics, it (or any other explanation) is by no means the clear winner.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/879a/879a6c1d-46ea-493e-b87f-956f38cef110.png" style="width: 50%;"></div>
Thus, given that explanations in the wild are often in the top-k format (c.f., Table III), the results suggest caution to us when interpreting these explanations.
# D. Agreement Evaluation Results
Figure 3 shows the agreement among various explanations, for both the E-P and P-E setups. Similar to the findings by [23], high disagreement among every pair of explanations is exhibited. However, the faithfulness evaluation does not identify any method as a top performer across the board, but instead all methods perform very closely to each other. This is in stark contrast with traditional interpretability methods generated for classifiers, where the performance difference among different methods is usually much larger (c.f., results by Zhou and Shah [51]). In other words, while Krishna et al. [23] found that explanations that perform very differently on faithfulness metrics (for classifier models) have high disagreement, we find that explanations that perform similarly on faithfulness metrics also have high disagreement (for the ChatGPT model). There could be two possible reasons for our finding. The first one is that our evaluations do not have the discriminative power to differentiate the good explanations from the bad ones. Some evidence supporting this claim is presented in Section
IV-E2 below. The second reason is that there could be some explanations that achieve distinctively higher faithfulness metric values, which understandably also has high disagreement with all of the current ones. In other words, we are only obtaining mediocre explanations with occlusion, LIME and self-explanation, while missing out on the “optimal one.” We leave further investigation of this topic to future work.
# E. Qualitative Analyses
During our experiments, we notice several differences between the explanation generation and evaluation for ChatGPT versus those for other models. These differences often stem from fundamental differences between ChatGPT and other models, and serve as important caveats when generating and understanding these explanations. 1) Level of Saliency Values: In the SELFEXP attribution values, it is very common for them to be on a few levels, whose values are often “well-rounded” decimal numbers such as 0.5 and 0.75, rather than arbitrary ones such as 0.6423 and 0.1982 (often produced by traditional techniques for traditional models). One example is shown in Table I and Table II, and additional instances are presented in Table IX. Quantitatively, on average each sentence has 18.2 words, but there are only 2.72 levels of unique explanations per sentence for the E-P model and 5.69 for the P-E model.
E-P Prompting
<review> Ford deserves to be remembered at Oscar time for crafting
this wonderful portrait of a conflicted soldier . <review>
[(‘Ford’, 0.500), (‘deserves’, 0.500), (‘to’, 0.000), (‘be’, 0.000), (‘re-
membered’, 0.500), (‘at’, 0.000), (‘Oscar’, 0.000), (‘time’, 0.000),
(‘for’, 0.000), (‘crafting’, 0.500), (‘this’, 0.000), (‘wonderful’,
1.000), (‘portrait’, 0.500), (‘of’, 0.000), (‘a’, 0.000), (‘conflicted’,
-0.500), (‘soldier’, 0.000), (‘.’, 0.000)]
(1, 0.800)
<review> It ’s never a good sign when a film ’s star spends the entirety
of the film in a coma . <review>
[(‘It’, 0.5), (“’s”, 0.5), (‘never’, -0.5), (‘a’, 0.0), (‘good’, 0.8), (‘sign’,
0.6), (‘when’, 0.0), (‘a’, 0.0), (‘film’, 0.0), (“’s”, 0.5), (‘star’, 0.4),
(‘spends’, 0.2), (‘the’, 0.0), (‘entirety’, 0.0), (‘of’, 0.0), (‘the’, 0.0),
(‘film’, 0.0), (‘in’, 0.0), (‘a’, 0.0), (‘coma’, -0.7), (‘.’, 0.0)]
(0, 0.700)
P-E Prompting
<review> Ford deserves to be remembered at Oscar time for crafting
this wonderful portrait of a conflicted soldier . <review>
(1, 0.9)
[(‘Ford’, 0.2), (‘deserves’, 0.8), (‘to’, 0.1), (‘be’, 0.1), (‘remem-
bered’, 0.7), (‘at’, 0.1), (‘Oscar’, 0.5), (‘time’, 0.2), (‘for’, 0.1),
(‘crafting’, 0.6), (‘this’, 0.1), (‘wonderful’, 0.9), (‘portrait’, 0.7),
(‘of’, 0.1), (‘a’, 0.1), (‘conflicted’, 0.8), (‘soldier’, 0.7), (‘.’, 0.1)]
<review> Even die-hard fans of Japanese animation ... will find this
one a challenge . <review>
(0, 0.8)
[(‘Even’, 0.2), (‘die-hard’, 0.1), (‘fans’, 0.3), (‘of’, 0.0), (‘Japanese’,
0.5), (‘animation’, 0.6), (‘...’, 0.0), (‘will’, 0.0), (‘find’, 0.0), (‘this’,
0.0), (‘one’, 0.0), (‘a’, 0.0), (‘challenge’, 0.4), (‘.’, 0.0)]
We believe that this phenomenon occurs due to the humanlike behavior and reasoning capability of ChatGPT. In its attempt to mimic a human (which is the central goal of the reinforcement learning from human feedback (RLHF) alignment [30]), it outputs explanation values that are plausibly produced by humans, who will not give very fine-grained saliency values (e.g., 0.6423 and 0.1982) as these values are often beyond the human’s reasoning ability. As recent papers have noted [50, 53], model explanations are often less understandable to humans than desirable in that humans may not be able to draw the correct high-level conclusions about the model’s reasoning from individual (i.e., local) explanations. One potential culprit could be the traditionally very fine-grained and varied saliency values, even if they faithfully reflect the model’s reasoning. Thus, the ChatGPT self-explanation may offer a different kind of faithfulnessunderstandability trade-off, where the explanations are intentionally vague but easier for human consumption. Testing this hypothesis requires a formal evaluation of understandability with human subjects, which we leave to future work. Orthogonally, comparing the explanations generated by aligned and unaligned models could identify the impact of alignment, though getting access to the latter model may prove difficult as companies are often reluctant to release these models due to safety concerns [20]. We attempted to combat the issue with few-shot prompting, where we provide some explanation values according to our intuitions on some examples before asking the model to generate its own explanation and prediction. However, the results are less than satisfactory. Few-shot prompting does solve the issue of saliency values being too “rounded:” when provided with more fine-grained values (e.g. 0.8694 and 0.1285) as attribution scores, the generated explanations also tend to be less rounded. However, still not too many saliency levels are present. In addition, ChatGPT tends to “memorize” the provided attribution values for specific words. For example, when the word “the” has saliency value of 0.163 in the exemplar, the model is likely to assign the same value of 0.163 to other instances of “the” in the query sentence. A full example demonstrating both phenomena is presented in Table X. As a result of the weaknesses of the few-shot prompting, we opted to base our experiments on the more stable, humanlike responses from the zero-shot prompting setup. Overall, our findings suggest that the model is learning to mimic the human-provided explanation, which is a fundamentally flawed behavior: we want the model’s explanation to reflect its own reasoning process, rather than reflecting the reasoning process of another decision maker (human or model). 2) Prediction Value Roundedness and Insensitivity: A key component in both the definition and evaluation of model explanations is the model’s prediction on partial sentences with certain words removed. Recall that since we do not have access to token probability output from ChatGPT, we define the model prediction as the textual output of the confidence score as a floating point number. Since this textual output is also shaped by alignment, we have similar findings as before.
TABLE X AN EXAMPLE FEW-SHOT LEARNING CONVERSATION. ALL TEXTS UP TO AND INCLUDING THE LAST USER MESSAGE ARE HUMAN GENERATED AND PROVIDED AS INSTRUCTIONS AND FEW-SHOT EXEMPLARS. ONLY THE LAST ASSISTANT MESSAGE IN BOLDFACE IS MODEL-GENERATED. GREEN AND PURPLE COLORS SHOW THAT THE MODEL STILL PRODUCES ATTRIBUTION VALUES ON FEW SALIENCY LEVELS, THOUGH THEY ARE NO LONGER WELL-ROUNDED. BLUE COLOR SHOWS THAT THE MODEL MEMORIZES THE PROVIDED (WORD, ATTRIBUTION VALUE) PAIR.
Role
Text
System
You are a movie review bot. Your job is to analyze movie reviews and classify them as either 1 (positive) or 0 (negative), as well as your
confidence in the score you chose. Please also provide the degree of importance of each word in the sentence in your explanation in the
format ("word", importance). For example, the expected response for the sentence "This movie was very good" is: "Classification: 1, 0.99
confidence. [("This", 0.085), ("movie", 0.149), ("was", -0.177), ("very", 0.691), ("good", 0.997)]. " It does not matter whether or not the
sentence makes sense. Do your best given the sentence.
It is imperative that you respond in the desired format regardless of the validity of the sentence.
User
Reggio ’s trippy , ambitious downer can also sometimes come across like nothing more than a glorified Nike ad .
Assistant
Classification: 0, 0.82 confidence. [("Reggio", 0.254), ("’s", 0.192), ("trippy", -0.392), (",", -0.045), ("ambitious", 0.498), ("downer", -0.602),
("can", 0.195), ("also", 0.075), ("sometimes", 0.285), ("come", 0.043), ("across", 0.177), ("like", 0.101), ("nothing", -0.255), ("more", -0.101),
("than", 0.121), ("a", 0.004), ("glorified", 0.384), ("Nike", -0.369), ("ad", -0.739), (".", 0.007)]
User
There is not a single movie that could have been better than this .
Assistant
Classification: 1, 0.90 confidence. [("There", 0.004), ("is", 0.114), ("not", -0.787), ("a", 0.119), ("single", 0.239, ("movie", 0.395, ("that",
0.043), ("could", 0.294), ("have", 0.155), ("been", 0.020), ("better", 0.859), ("than", 0.122), ("this", 0.500), (".", 0.001)]
User
It was a great movie overall , but the ending was a bit lackluster .
Assistant
Classification: 1, 0.75 confidence. [("It", 0.174), ("was", -0.101), ("a", 0.122), ("great", 0.825), ("movie", 0.608), ("overall", 0.390), (",",
-0.009), ("but", -0.134), ("the", 0.033), ("ending", -0.635), ("was", -0.145), ("a", 0.103), ("bit", -0.396), ("lackluster", -0.859), (".", -0.003)]
User
The film provides some great insight into the neurotic mindset of all comics even those who have reached the absolute top of the game .
Assistant
Classification: 1, 0.98 confidence. [("The", 0.033), ("film", 0.607), ("provides", 0.346), ("some", 0.091), ("great", 0.825), ("insight",
0.537), ("into", 0.091), ("the", 0.033), ("neurotic", -0.498), ("mindset", -0.498), ("of", 0.033), ("all", 0.033), ("comics", 0.537),
("even", 0.033), ("those", 0.033), ("who", 0.033), ("have", 0.033), ("reached", 0.033), ("the", 0.033), ("absolute", 0.033), ("top",
0.033), ("of", 0.033), ("the", 0.033), ("game", 0.033), (".", 0.000)]
Specifically, we notice that the model’s prediction values are often “well-rounded” and being one of several distinct values, as shown in Table XI. Furthermore, the removal of a few words often does not change the model’s prediction, as Table XII shows. This phenomenon has two causes: first, the model could likely infer the missing words and treat them as minor typographical errors; second, the model is aligned to human thinking, which habitually ignores these errors when making predictions. This model behavior has two implications. First, the occlu-
# TABLE XI
Sentence
Pred
Guided more by intellect than heart , his story flattens instead
of sharpens .
0.70
These are names to remember , in order to avoid them in the
future .
0.50
A hideous , confusing spectacle , one that may well put the
nail in the coffin of any future Rice adaptations .
0.00
Eastwood is an icon of moviemaking , one of the best
actors , directors and producers around , responsible for some
excellent work .
0.90
While Super Troopers is above Academy standards , its
quintet of writers could still use some more schooling .
0.80
sion saliency for a word is defined as the model prediction difference on inputs with and without the word, which results in the saliency value being exactly 0 for 82.6% of words for the E-P model and 54.5% for the P-E model. More importantly, all evaluations are affected to different extents by this behavior. The most severely affected ones are DFMIT and RankDel, which are computed from the model
Sentence
Pred
One of the greatest family-oriented , fantasy-adventure
movies ever .
0.90
One of the greatest family-oriented , fantasy-adventure
movies .
0.90
One of the family-oriented , fantasy-adventure movies .
0.80
One of the, fantasy-adventure movies .
0.80
One of the , movies .
0.80
of the , movies .
0.50
of the , movies
0.50
of the ,
0.50
the ,
0.50
,
0.50
prediction on the original sentence and that of one-wordremoved sentence only, similar to occlusion saliency. As a result, all explanations have very low scores on these two metrics, compared to the same evaluation settings on a finetuned RoBERTa model [27] which has much more gradual model prediction changes under word removal (∼25% for DFMIT and ∼0.5% for RankDel on average as reported by [51]). Notably, we did not observe a perfect 1.0 score for occlusion on RankDel, because many saliency values and word removal prediction changes are the same (at which point random tie-breaking is applied). Even with more words being removed, the model’s prediction still stays the same for a long time, which decreases the comprehensiveness score, as the initial few word removals could not significantly decrease the model’s confidence, and increases the sufficiency score, as the initial few word insertions could not significantly increase the model’s confidence.
# V. CONCLUSION AND FUTURE WORK
In this paper, we conducted the first rigorous assessment in LLM’s capability to self-generate feature attribution explanations – i.e., explaining its own prediction by stating the importance of every word in the input. Using ChatGPT and the SST dataset as our testbed, we construct prompts that reliably generate two types of LLM-generated self-explanations: full feature attribution explanation that assigns every word an importance score and a top-k explanation that only highlights the few most important words. The explanations are generated in two settings: one where they are generated first and then the model predictions (i.e., sentiment assessment) is made based on them, and the other where the prediction is generated first and the explanations are generated to justify the predictions. We compare these explanations to traditional explanation techniques – occlusion saliency and LIME – on both faithfulness and agreement metrics. We have two major findings from our investigation. First, none of the explanations, self-generated or not, hold a distinct advantage over the rest according to the faithfulness evaluations, yet they are highly different according to the agreement evaluations. As a result, there could potentially be some explanation which is much better than the current ones, and novel techniques may be needed to reveal them. Second, and possibly of independent interest in the study of LLMs, both the model prediction values and word attribution values are highly well-rounded, taking values such as 0.25, 0.67, 0.75, etc. This finding may also explain the similarity in faithfulness metric values in across different explanations: due to the lack of fine-grained variations in the explanation and prediction values, the evaluation metrics may not have the sufficient power to distinguish the good explanations from the bad ones. These findings also inform directions for future work. First, given that there is no prior work on studying LLM-generated feature attribution explanations, it is likely that our solution is not optimal, and better ways to elicit self-explanations could be developed. Second, we may need to rethink the evaluation
practice, which has shown to not perform very well. Indeed, if we consider ChatGPT’s reasoning ability as being more similar to humans than to earlier machine learning models, the result is not surprising: a human will not be able to give a precise number for the effect of removing one word in a sentence of twenty words (as used by DFMIT and RankDel metrics), or give a list of the cumulative influence of a word deletion or insertion sequence (as used by comprehensiveness, sufficiency and DFFrac metrics). Thus, alternative evaluation strategies, possibly involving carefully designed human subject studies [e.g. 4] are needed to definitively assess the strengths and weaknesses of these explanations. An orthogonal direction would be to evaluate other LLMs, such as GPT-4, Bard and Claude. A comparative study could reveal important insights in how these models understand themselves (i.e., introspection), as explanations are essentially properties about predictions (e.g., occlusion saliency is the prediction change under single word removal). In addition, we can conduct the analogous studies on other types of explanations, such as counterfactual explanations [43] and concept-based explanations [21, 26]. Finally, now that we have demonstrated that these explanations are feasible options to consider, we need to make sure that they are indeed beneficial and not used for harmful purposes. For example, how easily can they be manipulated [38], and could they hide fairness issues of the model [11]? Identifying and solving any observed issues is a necessary step before the deployment of these explanations.
# REFERENCES
[1] Julius Adebayo, Michael Muelly, Ilaria Liccardi, and Been Kim. Debugging tests for model explanations. In Advances in Neural Information Processing Systems (NeurIPS), 2020. [2] Julius Adebayo, Michael Muelly, Harold Abelson, and Been Kim. Post hoc explanations may be ineffective for detecting unknown spurious correlation. In International Conference on Learning Representations (ICLR), 2022. [3] David Alvarez-Melis and Tommi S Jaakkola. On the robustness of interpretability methods. In ICML Workshop on Human Interpretability in Machine Learning, 2018. [4] Gagan Bansal, Tongshuang Wu, Joyce Zhou, Raymond Fok, Besmira Nushi, Ece Kamar, Marco Tulio Ribeiro, and Daniel Weld. Does the whole exceed its parts? the effect of AI explanations on complementary team performance. In ACM CHI Conference on Human Factors in Computing Systems (CHI), pages 1–16, 2021. [5] Naman Bansal, Chirag Agarwal, and Anh Nguyen. SAM: The sensitivity of attribution methods to hyperparameters. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8673–8683, 2020. [6] Jasmijn Bastings, Sebastian Ebert, Polina Zablotskaia, Anders Sandholm, and Katja Filippova. A protocol for evaluating the faithfulness of input salience methods for text classification. In Conference on Empirical Methods
in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2022. [7] Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom. e-snli: Natural language inference with natural language explanations. CoRR, abs/1812.01193, 2018. URL http://arxiv.org/abs/1812.01193. [8] Yanda Chen, Ruiqi Zhong, Narutatsu Ri, Chen Zhao, He He, Jacob Steinhardt, Zhou Yu, and Kathleen McKeown. Do models explain themselves? counterfactual simulatability of natural language explanations. arXiv preprint arXiv:2307.08678, 2023. [9] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https://lmsys.org/blog/2023-03-30-vicuna/. [10] George Chrysostomou and Nikolaos Aletras. Improving the faithfulness of attention-based explanations with taskspecific information for text classification. In Annual Meeting of the Association for Computational Linguistics and International Joint Conference on Natural Language Processing (ACL-IJCNLP), pages 477–488. Association for Computational Linguistics, 2021. [11] Jessica Dai, Sohini Upadhyay, Ulrich Aivodji, Stephen H. Bach, and Himabindu Lakkaraju. Fairness via explanation quality: Evaluating disparities in the quality of post hoc explanations. In AAAI/ACM Conference on AI, Ethics, and Society (AIES), page 203–214. Association for Computing Machinery, 2022. [12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pages 4171–4186. Association for Computational Linguistics, 2019. [13] Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, and Byron C. Wallace. ERASER: A benchmark to evaluate rationalized NLP models. In Annual Meeting of the Association for Computational Linguistics (ACL), pages 4443–4458. Association for Computational Linguistics, 2020. [14] Simon Frieder, Luca Pinchetti, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Christian Petersen, Alexis Chevalier, and Julius Berner. Mathematical capabilities of chatgpt. arXiv preprint arXiv:2301.13867, 2023. [15] A Shaji George and AS Hovan George. A review of chatgpt ai’s impact on several business sectors. Partners Universal International Innovation Journal, 1(1):9–23, 2023. [16] Amirata Ghorbani, Abubakar Abid, and James Zou. Interpretation of neural networks is fragile. In AAAI Conference on Artificial Intelligence (AAAI), volume 33,
pages 3681–3688, 2019. [17] David Gunning and David Aha. Darpa’s explainable artificial intelligence (xai) program. AI magazine, 40(2): 44–58, 2019. [18] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. [19] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In International Conference on Learning Representations (ICLR), 2019. [20] Geoffrey Irving, Paul Christiano, and Dario Amodei. Ai safety via debate. arXiv preprint arXiv:1805.00899, 2018. [21] Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In International conference on machine learning, pages 2668–2677. PMLR, 2018. [22] Lucas Kohnke, Benjamin Luke Moorhouse, and Di Zou. Chatgpt for language teaching and learning. RELC Journal, page 00336882231162868, 2023. [23] Satyapriya Krishna, Tessa Han, Alex Gu, Javin Pombra, Shahin Jabbari, Steven Wu, and Himabindu Lakkaraju. The disagreement problem in explainable machine learning: A practitioner’s perspective. CoRR, abs/2202.01602, 2022. URL https://arxiv.org/abs/2202.01602. [24] Jiwei Li, Xinlei Chen, Eduard Hovy, and Dan Jurafsky. Visualizing and understanding neural models in NLP. In Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pages 681–691. Association for Computational Linguistics, 2016. [25] Jiwei Li, Will Monroe, and Dan Jurafsky. Understanding neural networks through representation erasure. arXiv:1612.08220, 2016. [26] Oscar Li, Hao Liu, Chaofan Chen, and Cynthia Rudin. Deep learning for case-based reasoning through prototypes: A neural network that explains its predictions. In AAAI Conference on Artificial Intelligence (AAAI), 2018. [27] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized BERT pretraining approach. arXiv:1907.11692, 2019. [28] Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems (NIPS), pages 4765– 4774, 2017. [29] Christoph Molnar. Interpretable Machine Learning. 2 edition, 2022. [30] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022. [31] Thang Pham, Trung Bui, Long Mai, and Anh Nguyen. Double trouble: How to not explain a text classifier’s decisions using counterfactuals synthesized by masked language models? In Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and International Joint Conference on Natural Language Processing (AACL-IJCNLP), pages 12–31. Association for Computational Linguistics, 2022. [32] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018. [33] Alec Radford, Jeffrey Wu, Rewon Child, David ’Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8): 9, 2019. [34] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. "Why should I trust you?" explaining the predictions of any classifier. In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016. [35] Alvin E Roth. The Shapley Value: Essays in Honor of Lloyd S. Shapley. Cambridge University Press, 1988. [36] Sofia Serrano and Noah A. Smith. Is attention interpretable? In Annual Meeting of the Association for Computational Linguistics (ACL), pages 2931–2951. Association for Computational Linguistics, 2019. [37] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv:1312.6034, 2013. [38] Dylan Slack, Sophie Hilgard, Emily Jia, Sameer Singh, and Himabindu Lakkaraju. Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods. In AAAI/ACM Conference on AI, Ethics, and Society (AIES), pages 180–186. Association for Computing Machinery, 2020. [39] Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg. SmoothGrad: Removing noise by adding noise. arXiv:1706.03825, 2017. [40] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA, October 2013. Association for Computational Linguistics. URL https://www.aclweb.org/ anthology/D13-1170. [41] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In International Conference on Machine Learning (ICML), pages 3319–
3328, 2017. [42] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpaca: A strong, replicable instruction-following model. Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html, 3(6):7, 2023. [43] Sandra Wachter, Brent Mittelstadt, and Chris Russell. Counterfactual explanations without opening the black box: Automated decisions and the GDPR. Harvard Journal of Law & Technology, 31:841, 2017. [44] Zengzhi Wang, Qiming Xie, Zixiang Ding, Yi Feng, and Rui Xia. Is chatgpt a good sentiment analyzer? a preliminary study. arXiv preprint arXiv:2304.04339, 2023. [45] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. CoRR, abs/2201.11903, 2022. URL https: //arxiv.org/abs/2201.11903. [46] IpKin Anthony Wong, Qi Lilith Lian, and Danni Sun. Autonomous travel decision-making: An early glimpse into chatgpt and generative ai. Journal of Hospitality and Tourism Management, 56:253–263, 2023. [47] Mengjiao Yang and Been Kim. Benchmarking attribution methods with relative feature importance. arXiv:1907.09701, 2019. [48] Xianjun Yang, Yan Li, Xinlu Zhang, Haifeng Chen, and Wei Cheng. Exploring the limits of chatgpt for query or aspect-based text summarization. arXiv preprint arXiv:2302.08081, 2023. [49] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In European Conference on Computer Vision (ECCV), pages 818–833. Springer, 2014. [50] Yiming Zheng, Serena Booth, Julie Shah, and Yilun Zhou. The irrationality of neural rationale models. In 2nd Workshop on Trustworthy Natural Language Processing (TrustNLP). Association for Computational Linguistics, 2022. [51] Yilun Zhou and Julie Shah. The solvability of interpretability evaluation metrics. arXiv:2205.08696, 2022. [52] Yilun Zhou, Serena Booth, Marco Tulio Ribeiro, and Julie Shah. Do feature attribution methods correctly attribute features? In AAAI Conference on Artificial Intelligence (AAAI), 2022. [53] Yilun Zhou, Marco Tulio Ribeiro, and Julie Shah. ExSum: From local explanations to model understanding. In Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT). Association for Computational Linguistics, 2022.
