# Prompt, Condition, and Generate: Classification of Unsupported Claims with In-Context Learning
eter Ebert Christensen pec@di.ku.dk Srishti Yadav University of Copenhagen & Pioneer Centre for AI srya@di.ku.dk Serge Belongie s.belongie@di.ku.
Abstract
Unsupported and unfalsifiable claims we encounter in our daily lives can influence our view of the world. Characterizing, summarizing, and – more generally – making sense of such claims, however, can be challenging. In this work, we focus on fine-grained debate topics and formulate a new task of distilling, from such claims, a countable set of narratives. We present a crowdsourced dataset of 12 controversial topics, comprising more than 120k arguments, claims, and comments from heterogeneous sources, each annotated with a narrative label. We further investigate how large language models (LLMs) can be used to synthesise claims using In-Context Learning. We find that generated claims with supported evidence can be used to improve the performance of narrative classification models and, additionally, that the same model can infer the stance and aspect using a few training examples. Such a model can be useful in applications which rely on narratives , e.g. fact-checking.
# 1 Introduction
Online platforms have revolutionized the landscape of public discourse, facilitating extensive debates across a wide range of topics. However, these online discussions often suffer from a lack of coherent and concise arguments. Despite this inherent challenge, it is possible to discern particular motions (Levy et al., 2014), opinions (Li et al., 2020), human values (Kiesel et al., 2022), and narratives (Christensen et al., 2022) within the seemingly disorganized discourse. The ability to identify narratives in online debates is paramount for factchecking and argument mining, as it enables the evaluation of unsupported claims and their validity. In our methodology, narratives are differentiated from arguments and claims by incorporating additional attributes: topics, stances, aspects, and evidence. The topic refers to the subject under discussion, such as the ethical aspects of cloning humans for reproductive purposes. The stance rep-
Serge Belongie
resents the viewpoint taken on the topic, for example, a negative stance indicating that cloning for reproductive purposes is considered unethical and unacceptable. Within the narrative, the aspect focuses on a specific perspective, providing a more nuanced understanding of the topic. For instance, the aspect within the context of cloning could be the creation of cloned embryos solely for research purposes, which delves into a particular subtopic. These attributes aid in identifying arguments in non-argumentative sources (Stab et al., 2018). Evidence plays a crucial role in assessing the credibility of a statement. When supported by evidence, a statement gains strength and credibility, classifying it as an argument (Hansen and Hershcovich, 2022). For instance, in the text “Cloning humans for reproductive purposes is unethical and unacceptable, but creating cloned embryos solely for research – which involves destroying them anyway – is downright criminal,” the presence of evidence highlighting the destruction of embryos strengthens the argument. Conversely, without evidence, a statement such as “Cloning humans for reproductive purposes is unethical” would be categorized as a claim, lacking the necessary substantiation to be considered an argument. Additional support is required to validate a claim as an argument. With these definitions, we can briefly differentiate between narrative, arguments and claims as: 1. Narrative: Concise expression of an individual’s perspective on a specific topic. 2. Claim: Statement or proposition without supporting evidence. 3. Argument: Claim supported by evidence and reasoning. Aim to justify a specific stance on a topic
As seen above, claims can lack or have insufficient evidence and be unverifiable or unfalsifiable for purposes of fact-checking in real world scenarios (Glockner et al., 2022) and are hence often not suitable for fact-checking pipelines and
thus discarded (Augenstein, 2021). Instead of discarding the claims and arguments, we propose that one should instead identify the individual unsupported claim or narrative, e.g., “human cloning is wrong”. We call this task Narrative Prediction which forms the basis of this paper. We use the word “Prediction” as an umbrella term, as it can be viewed as either classification (due to the small set of unsupported claims that reflect different viewpoints in the debate) or alternatively a generation task. In addition to fact-checking the existing literature on claim generation using large language models (LLMs) lacks attention to the relationship between narratives extracted from online debate portals (Christensen et al., 2022) and argumentative texts (Habernal and Gurevych, 2016), as well as the effective modeling of narratives by LLMs. This work addresses these gaps by formalizing narratives in online debates and understanding their elements: topics, aspects, stance, and evidence. Additionally, a curated dataset of 120k tweets, with around 40 narratives per topic, is introduced to train and evaluate narrative prediction techniques for fact-checking systems. Furthermore, we propose a method to enhance narrative prediction by generating synthetic tweets through argumentative attributes such as stance and aspects using fewshot In Context Learning (ICL) as illustrated this in Fig. 1. The task of narrative prediction simply corresponds to only the right hand side of the figure using no generated candidates where we fine-tune the LLM on tweets. In summary, the contributions of this paper are: 1. A specific definition for narratives, along with an analysis of how this differs from arguments, claims, and motions. 2. A new dataset and task, consisting of online comments and tweets labelled for narrative prediction. 3. A narrative prediction approach that maps all the tweets from a fine-grained debate into a list of narratives using a LLM. 4. A computational approach that generates synthetic arguments/claims with a specified aspect and stance.
# 2 Related Works
Corpora of textual claims considering various controversial topics have often been used in the study of rhetoric and argumentation, including summarization (Stammbach and Ash, 2020), optimization (Skitalinskaya et al., 2022), identifying human
values (Kiesel et al., 2022), robustness of arguments (Sofi et al., 2022), controllable text generation (Schiller et al., 2021), stance detection (Stab et al., 2018), and studying what constitutes an argument (Trautmann et al., 2020). Prior work on claim and argument summarization has been beneficial in different tasks and domains. In early works, summarization was used for explainable fact-checking (Stammbach and Ash, 2020; Mishra et al., 2020) and has recently been used to denoise tweets (Bhatnagar et al., 2022). However, abstractive summarization techniques for real-world tweets are still underdeveloped compared to traditional text summarization methods. Given the effectiveness of prompt-based methods in tasks like abstractive summarization and binary classification (Chung et al., 2022; Sanh et al., 2022), we propose exploring these methods to enhance the text generation of arguments, particularly within the fine-grained topic debates. While fine-grained approaches have been explored in argument mining (Hansen and Hershcovich, 2022; Trautmann et al., 2020; Schiller et al., 2021), they often address broader controversial topics (“minimum wage.”) rather than narrow debates (“crypto currencies as a fiat currency,”). Similarly other works that classify if a claim is mentioned in a text (Mirkin et al., 2018) studies motions which are an action that should be taken (as can be seen with the example “we should introduce goal line technology”). Other lines of works focus on scaling up by detecting “generic” claims frequent across topics (Orbach et al., 2019) or mining candidate claims from corpora (Lavee et al., 2019). In comparison we are envisioning our work to be applicable for unfalsifiable or unverifiable claims coming from short noisy tweets rather than a high quality curated database (iDebate) which contain minutes long speeches. In our work we create a new dataset, focusing on narrow debate topics, by relying on an argument mining annotation scheme based on Hansen and Hershcovich (2022), consisting of various categories of claims and arguments found in online debates. Where Hansen and Hershcovich (2022) compare arguments in terms of categories (normative or factual arguments), we propose and study the new task of predicting controversial narratives from tweets. Perhaps most similar to our work is Christensen et al. (2022), which proposed a human-inthe-loop-based model to cluster unfalsifiable claims using crowdsourced triplets similarities.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/12e6/12e6e1bd-c412-46f5-bec1-6318acdfe3eb.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Prompt, Condition, and Generate: A framework to enhance narrative prediction by synthetizing tweets. We first prompt a LLM for the stance and aspects of a new tweet using ICL with some examples, we then condition the LLM on these attributes to synthesize tweets. Lastly, we fine-tune a LLM on all tweets to generate narratives.</div>
# 3 Task and Data
This section introduces our definition of a narrative, and a proposed task, and presents the data used for development and evaluation.
As mentioned in the introduction, we define the term narrative as a concise statement lacking supporting evidence, which can originate from an unfalsifiable or unverifiable claim. Additionally, narratives can include arguments supported by evidence types such as anecdotal, expert, or normative sources as defined in (Hansen and Hershcovich, 2022), instead of empirical studies. The objective of our methodology is to identify a small set of unsupported claims that reflect diverse viewpoints in a debate and require attention from fact-checkers. After defining the term narrative we can now focus on a theoretical underpinning of this paper which is a proposed relationship between number of narratives and the scope of the fine-grained debate, that we call the parrot hypothesis.
The Parrot Hypothesis In a given social media debate, the thoughts and opinions contributed by commentators resolve to a finite set of distinct narratives. While users could, in principle, state their views in a concise, distilled manner, they often prefer to write embellished variants or personal takes that require reading between the lines. At its core, the parrot1 hypothesis seeks to propose a concept to manage the variations of state-
1We use “parrot” in the sense of “parroting talking points,” except that we don’t assume the commentators are necessarily being fed talking points without their knowledge.
ments in a debate. By grouping statements into a finite set of narratives related to common topics, the hypothesis narrows the scope of the debate and transforms it into a classification problem. Narratives, representing individual unsupported claims or viewpoints, play a crucial role in capturing diverse perspectives and supporting fact-checking efforts. Despite the potential for infinite arguments, a limited number of distinct claims tend to emerge in online debates, backed by the majority of users (Boltuži´c and Šnajder, 2015). Our hypothesis is that a narrow enough topic will emit such behaviour from users. Incorporating the parrot hypothesis and identifying narratives could enable a more systematic analysis to improve understanding of narratives and facilitating fact-checking.
# 3.2 Narrative Prediction
Task Given a single tweet t, a statement by a participant in a debate, and a set of possible narratives N, rewrite t into a narrative n ∈N such that:
• the narrative is written as an unsupported claim, • only one narrative n can be selected for each tweet from N, and • n preserves the meaning of t as much as possible.
• the narrative is written as an unsupported claim, • only one narrative n can be selected for each tweet from N, and • n preserves the meaning of t as much as possible.
The set of possible narratives, denoted as N, is sourced from domain experts. Although a tweet may implicitly or explicitly contain multiple nar-
ratives, our aim is to identify and assign only one explicitly stated narrative for each tweet. By addressing narrative prediction in this manner, we strive to transform tweets into coherent and explicit unsupported claims, contributing to a deeper understanding and analysis of the content within the context of social media discourse.
# 3.3 Annotation scheme
To collect relevant data, we use an annotation scheme comprising a fine-grained topic, a sentence, and a narrative (unfalsifiable and unverfiable claim). Additionally, we explore the augmentation of an existing dataset, following an alternative annotation scheme (Schiller et al., 2021), to incorporate attributes such as stance (polarity of the argument) and aspect (subtopics or viewpoints) and use it for the generation of synthetic tweets. Though there exist narratives that are claims with supporting evidence (can be anecdotal, factual or normative which are found in (Hansen and Hershcovich, 2022)), the type of evidence is not considered for annotation. The details of this will be explained in the following section.
We present two datasets: Twitter-Narratives-9 (TN9) and an augmented version of the UKPCorpus-Aug dataset. UKP-Corpus-Aug, which is the augmented UKP-Corpus dataset includes stance, aspect, and narrative annotations for three randomly selected topics from the original UKPCorpus (Schiller et al., 2021; Stab et al., 2018). On the other hand, TN9 consists of narrative annotations for 9 carefully selected controversial topics. Table 1 provides an overview of the datasets, including key statistics and a comparison between them. Additionally, Table 2 presents examples from TN9.
UKP-Corpus-Aug
TN9
Annotations
Aspect/Stance/Narrative
Narrative
Tweets (train/test)
30k/1.9k
90k/5.4k
Topics
Abortion, Cloning,
AGI, Attractiveness,
Nuclear Energy
Alternative Meat,
Corporate culture,
Crypto, Baby Formula,
Influencer, Transport,
Mental health
Source (Sentence)
Reddit
Twitter
Source (Labels)
mTurk
mTurk
Table 1: Summary of the datasets. (The original UKPCorpus consists of only Stance and Aspects, we provide narrative annotations)
Topic
Sentence
Narrative
Crypto
you are promoting crypto which is
Influencers are scamming
a scam helping thieves and criminals
their fans using crypto
you are also full of plastic parts
and fillers profitable for the
pharmaceutical and cosmetic industry
Formula My congressman here voted NO on ,
People are reselling baby
lowering gas prices, NO on the baby
formula to other countries
formula bill, NO on contraception (?!!),
for higher prices
and NO on other helpful bills. It is
unbecoming to complain about economic
hardship and then contribute to it.
AGI
And on the other side, AGI will be
AI will not replace
the single greatest technology to
humans but augment them
alleviate human suffering in all of history.
Table 2: Example sentences and annotated narratives.
# 3.4.1 Scraping
We start with scraping relevant data from Twitter. First a series of searches is executed combining different keywords and sentences/phrases, highlighting different statements in a topic. We search for 40 different keywords per topic from year 2016-2022 and search for as many fields (e.g., images, links, and other metadata) as possible using the Twitter API. The specific keywords used for each topic can be found in Appendix B.
# 3.4.2 Filtering and Data Cleaning
To ensure that we are working with claims, we perform filtering steps. First, we remove duplicates but maintain identical sentences with different hashtags after removing retweets, quote tweets, links and videos, as well as mentions of users, token and media mentions. Second, we replace unreadable hexadecimal representations of unicode characters with their respective character, and encode the text with ascii characters. This results in 98,187 English tweets in total, around 11k tweets for each topic. The geographic distribution of the tweets as shown in Figure 2.
# 3.4.3 Dataset Annotation
Annotation of narratives is conducted using Amazon Mechanical Turk in 2 rounds. In round 1, we design a pretest to ensure that the workers know the difference between an argument, claim and evidence using the dataset from (Hansen and Hershcovich, 2022). Given a tweet the annotators classify whether the tweet is a claim, argument or neither. Furthermore they also classify the evidence type given an argument (Hansen and Hershcovich, 2022). Learning to distinguish between these claims will help them in determining the narrative of the tweet. After passing at least 4 out of 5 questions the workers could begin annotating our 120k tweets. Using lists of around 40 narratives
per topic compiled by domain experts we proceed to round 2. Each task consists of 1 tweet from 1 topic and annotators are asked to pick 1 out of ∼ 40 different narratives this tweet follows (if any). Furthermore, by using the definition of argument as in Hansen and Hershcovich (2022) we can decompose it into 1) a claim and 2) its evidence, and use their schema to categories the type of evidence an argument may have. We thus additionally ask if the tweet is a claim or an argument with an evidence type that is not a study. A study here refers to “Results of a quantitative analysis of data, given as numbers, or as conclusions.” That is statements that are cited, or easy to verify numbers should they appear in another argument. (Hansen and Hershcovich, 2022) The pay is 18$/hr. Detailed instructions can be found in Appendix D.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c212/c212ad2e-8d57-407b-b390-c163ffc670b4.png" style="width: 50%;"></div>
Figure 2: Visualization of the percentages of the number of tweets per country. Like (Huang and Carley, 2019) only 2% of all tweets had available geotags and the tweets are found to be predominantly from the US, where the userbase is numerically the largest.
# 4 Method
Problem statement: Our goal is to create a model that output an estimate of the true narrative n given tweet t from debate d. We do so by
• Investigating if identifying the narrative of a tweet is best suited as a text2text approach or a classification approach.
• Creating a data augmentation step using different kinds of ICL to help the best fine tuning procedure.
Narrative prediction approach Given observed data {ti, ni}N i=1, we could parameterize a model as ˜n = fϕ(t), where f is a pretrained LLM with parameters ϕ is the model that we finetune. This is illustrated as step 3 in fig. 1. We note that our prediction ˜n in this case would be free-form text, a text2text approach. As the last step in fig. 1. illustrates a finetuning procedure we could alternatively parameterise a model as ˜n = g(hθ(fω(t))), where
g is a lookup function that maps class ci to narrative ni for debate d, h is a multi-class classifier with parameters θ, and f the model from before, but only using the encoder with parameters ω and the classification head h.The prediction ˜n is now a class, a multi-class classification approach. The lookup function g enables us to take a predicted class and look up the actual narrative in the list written in appendix F. But doing this substitution we can calculate a Rouge score between the target narrative and the predicted one. However, during training we opt for optimizing the crossentropy loss on the narrative classes.
only finetuning f, which is a LLM, we argue for using new methodologies focusing on ICL to exploit the capabilities of the LLM further and validate their performance on our new dataset. As such we let us inspire by prior work Schiller et al. (2021) on generating synthetic arguments (candidates) using aspects and a stance, that we call Prompt, Condition and Generate (PCG) and add candidates to our finetuning procedure as illustrated in step 3 in fig. 1. In contrast to their work our setup requires no training and can be done using a few examples using ICL as illustrated in fig. 1. That is as a first step we annotate a few handmade examples with topics names, a binary stance and a snippet of the example tweet which forms the aspect and then we prompt the frozen model to output the stance and aspect of a novel tweet t. Then we condition the same model anew on its predicted stance and aspect to generate a candidate, by asking it to write a tweet knowing only about the debate topic, a stance and the aspect. To complete the creation of synthetic data we copy the original narrative n from t to form the candidate. This data can be used for additional fine-tuning of the text generation model that generate narratives, as shown in fig. 1.
ICL Methods In the context of using LLM for ICL, a simple but effective approach called fewshot learning is to provide several examples of a task in the same prompt with the given input (Brown et al., 2020). Additionally one can also first generate an explanation as to why certain outputs are favourable before generating the final answer, this is called Chain-of-Thought (CoT) (Wei et al., 2023). Furthermore one should be careful with the selected examples for ICL. As noted in Zhao et al. (2021), standard ICL can be biased towards
the training examples and the order of their occurrence. To mitigate this effect one can estimate the bias towards each answer by feeding in a test input that is content-free, e.g., “N/A” and “”. In practice on can fit an affine transformation to “calibrate” (Cal) the model’s output probabilities to cause a uniform prediction for “N/A”. We will investigate these methods in our PCG setup using different numbers of shots for aspect and stance prediction.
# 5 Experiments
In this section we investigate the performance of our finetuning approaches, including synthetic tweet generation for performance enhancement and the subtasks of stance and aspect prediction using different ICL techniques. We predict narratives on 7548 test cases (629 per topic).
# 5.1 Setup
Classification: As described in Section 4.1, our classification model (SFT_head) consists of an encoder fω, being a T0 encoder (Sanh et al., 2022) model and a multi-class classifier h, which is a single MLP that project the hidden layer down to the number of narratives present in one topic. We only finetune h using using the crossentropy loss. Finally using g we can report the Rouge-L F1 score as we convert the predicted class into the written narrative and comparing it with ground truth. Text generation: In contrast to the classification model fϕ is the full T0 model. We add new parameters and make a parameter efficient fine-tuning setup known as LoRA (Liu et al., 2022) on the T0 model. LoRA incorporates two low-rank matrices that are added to each parameter matrix in T0. We measure the Rougle-L F1 score between the generated narrative ˆn and the ground truth narrative. Prompt, Condition and Generate: To enhance the above mentioned setups we generate synthetic tweets. We do this we first infer the stance and aspect of a new tweet by insert up to 4 such examples into the prompt, and then second simply prompt our frozen model to write a tweet with the predicted stance containing the sentence from the aspect and about the same topic as the new tweet, this is shown in Figure 1. To test how well the model can predict aspects and stances we focus on stance and aspect data from the UKP-Corpus on 3 randomly select topics, as these sentences are annotated with a stance and
Classification: As described in Section 4.1, our classification model (SFT_head) consists of an encoder fω, being a T0 encoder (Sanh et al., 2022) model and a multi-class classifier h, which is a single MLP that project the hidden layer down to the number of narratives present in one topic. We only finetune h using using the crossentropy loss. Finally using g we can report the Rouge-L F1 score as we convert the predicted class into the written narrative and comparing it with ground truth.
Prompt, Condition and Generate: To enhance the above mentioned setups we generate synthetic tweets. We do this we first infer the stance and aspect of a new tweet by insert up to 4 such examples into the prompt, and then second simply prompt our frozen model to write a tweet with the predicted stance containing the sentence from the aspect and about the same topic as the new tweet, this is shown in Figure 1. To test how well the model can predict aspects and stances we focus on stance and aspect data from the UKP-Corpus on 3 randomly select topics, as these sentences are annotated with a stance and
aspects. We compare 3 methods: standard ICL, CoT (Wei et al., 2023) and Cal (Zhao et al., 2021), with a fully supervised BERT span predictors as a baseline. We train 3 BERTBASE baselines with only indicating it is only trained on this topic using 10k examples. The second remain uses 60k total examples training on the 5 remaining out of distribution topics from (Stab et al., 2018) ( i.e. excluding abortion, cloning & nuclear energy) before fine-tuning to a new topic and finally all trained on all 8 topics (80k examples) from (Schiller et al., 2021).The ICL setups use 4 examples to predict the attributes, though we experiment with including fewer examples for aspect prediction (Figure 3 ). We do not use verbalizers for ICL but restrict the possible decoding output only to the words considered in the sentence for aspect prediction or “for” or “against” in case of stance prediction.
In addition to generating candidates with our original model f, we test the generality of generating candidates of these attributes by conditioning other LLMs on them. These include T5-Flan-3B (Chung et al., 2022), BLOOM-175B (Workshop et al., 2023) and CTRLUKP (Schiller et al., 2021) for the UKP-Corpus. When finetuning using candidates and tweets, we compare them using several metrics like precision oriented BLEU (Papineni et al., 2002), recall oriented Rouge-L (Lin, 2004), METEOR (Banerjee and Lavie, 2005), and finally chrF (Popovi´c, 2015). To automatically quantify to what extent a candidate contains the meaning of the original claim, we compute their semantic similarity in each case using the BERT-score(Zhang* et al., 2020). Additionally we conduct a human evaluation of the generated candidates to ensure readability for humans. For each generative model and topic we select 10 candidates and acquire 2 independent crowdworkers via MTurk at 18$/hour. The annotators scored all candidates on four quality metrics: (1) argument quality (2) persuasiveness, (3) meaning preservation and (4) fluency. We follow Schiller et al. (2021) for assessing the argument quality, Habernal and Gurevych (2016) for persuasiveness and Skitalinskaya et al. (2022) for quality using these Likert scales: Argument Quality. 1 (much worse than original) - 5 (notably improved), Persuasiveness. 1 (generated text less persuasive than original) - 3 (generated text is more persuasive), Fluency. 1 (major errors) - 3 (fluent) and Meaning Preservation. 1 (entirely different) - 5 (identical). Lastly we report the inter-annotator
# agreement (Cohen, 1960) and krippendorff’s alpha (Krippendorff, 2004) between 2 annotators.
Stance prediction To do stance prediction, we classify a tweet as either “for” or “against” a particular topic. For ICL methods we output the most likely word and convert it to 0 or 1 to compare it with the binary class output for the baseline model. We use binary cross entropy loss to compare the predicted stance with the true label.
Aspect prediction Here we aim to identify the correct span of text within a tweet. The span is represented using the beginning-inside-outside (BIO) tags format (Ramshaw and Marcus, 1995). Here the initial word within the span is given the label “B” for beginning, the following words within the span is given the label “I” for inside, and finally any other word is given the label “O” for outside, making it a ternary classification task for the baselines. We sample multiple completions using beam search and report the average micro F1 and accuracy for both stance and aspect prediction.
# 5.2 Results
Classification 10 epochs of fine-tuning LM head results in a 38.54 Rouge-L F1 score for the UKP corpus and 38.72 Rouge-L F1 score for TN9.
Text generation Fine-tuning the LoRA weights results in a 39.32 Rouge-L F1 score for the UKP corpus and 39.49 Rouge-L F1 score for TN9, similar to other summarization tasks (Zhang et al., 2022). Inspecting the results of this models for a couple of outputs is shown in Table 3. Analysing the second example we see a more concisely written narrative than the target, this lowers the resulting Rouge F1 score due to its shorter common sub-sequence.
Tweet t
Model Prediction tn Target Narrative n
Animals are not ingredients! eating meat is murder Eating meat is murder
Yall find hypermasculinity
resulting in insecurities
about the lack of a better
Hypermasculinity is
Hypermasculinity in and
body attractive? Lmaaoo
problematic
of itself is the problem
Table 3: Sentences with predicted and target narrative. Prompt condition generate Given prior results we proceed with the best setup, the text generation setup from before and Table 4 shows the average Rouge-L F1 micro accuracy using additional candidate examples generated by T0-3B, T5-flan-3B model (Chung et al., 2022), an API call to BLOOM176B (Workshop et al., 2023) and the CTRL gener-
ative model from Schiller et al. (2021) respectively. Using 629 candidates we get a 4 percentage point increase from the 39.49 Rouge-L F1 for TN9 from before, highlighting the strength of our approach.
Setups
UKP
TN9
SFT_T0
43.56
43.89
SFT_T5F
44.12
44.34
SFT_BLOOM
42.64
42.97
SFT_CTRL
43.34
–
Table 4: Summary of Rouge-L F1 scores using text2text supervised fine-tuning on the original dataset as well as candidates generated with different models.
Table 5 shows the quantitative metrics of our candidates. The relatively low BLEU (6.5) and ROUGE-L (9.2) indicate that revisions take place, however due to the high BERT-score (90.5) the meaning is largely preserved. Also the METEOR and Rouge-L scores are similar to Schiller et al. (2021) indicating similar generative behaviour. TN9 lower scores indicate that the model has difficulty generating similar sentences to the original tweet using a predicted stance and aspect.
Approach
BLEU RouL Meteor BERT-score chrf
UKP-Corpus
CTRLUKP
8.3
12.1
16.4
83.7
23.1
BLOOM
6.5
13.6
16.2
84.8
31.1
T5-flan
10.8
20.6
16.4
90.5
25.1
T0
13.6
20.3
16.7
90.2
25.2
TN9
BLOOM
7.94
9.2
9.7
82.1
23.8
T5-flan
11.2
13.7
9.5
87.4
18.7
T0
12.3
13.1
9.2
87.8
18.9
Table 5: Automatic evaluation: Average performance of each model on 629 test cases per topic
Table 6 show generally low Krippendorff’s alpha agreement of 0.24 on average, which are common in subjective tasks (Wachsmuth et al., 2017). The inter-annotator agreement (Cohen, 1960) varies from model and attribute but is on average .25, which can be interpreted as “fair” agreement (Landis and Koch, 1977). Table 6 shows that human annotators find text generated by T0, having a higher persuasiveness (2.6) and having similar meaning to the source text (4.5) than the other methods. However, candidates from BLOOM and CTRL-UKP have a higher argument quality (3.5 vs. 3.6 and 4.2) and are more fluently written. Table 7 shows T0 being preferred for generating meaningful and persuasive texts. This is important as we will use the data in a fine-tuning setup.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b0d7/b0d7ef03-7366-4141-9f89-cfcc1bf7ed4b.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Average Aspect accuracy of few-shot ICL (T0-3B) on the Abortion, Cloning and the Nuclear Energy topic in the UKP dataset using random subsets of k′ = 1 . . . 4 examples. We display the best performances of the best fine-tuned BERTBASE baselines, the tags only, remain and all indicate the same setup from table 8.</div>
Model
Persuasiveness Fluency Argument Meaning
UKP-Corpus
CTRLUKP
2.1
2.3
3.6
3.4
BLOOM
1.9
2.8
4.2
4.1
T5-flan
2.2
1.8
3.2
3
T0
2.6
2.8
3.5
4.5
TN9
BLOOM
2
2.7
3.4
3.5
T5-flan
2.4
2.3
3.6
3.3
T0
2.4
2.5
3.4
3.5
<div style="text-align: center;">Table 6: Human evaluation: Average scores on 10 candidates per topic using different models.</div>
Model
Persuasiveness Fluency Argument Meaning
CTRLUKP
0.2/0.3
0.2/0.3
0.2/0.2
0.4/0.4
BLOOM
-0.1/-0.1
0.4/0.4
0.1/0.2
0.3/0.1
T5-flan
0.1/0.1
0.3/0.3
0.1/0.4
0.5/0.4
T0
0.3/0.3
0.4/0.3
0.2/0.3
0.5/0.3
Stance prediction Table 8 shows our methods outperform the baselines on at least two topics in UKP-Corpus with Cloning topic being an exception. We believe this is because the distribution of stances in this topic makes it highly polarized. While imperfect it suffices to generate candidates.
Method
Abor. (F1 / Acc)
Clon. (F1 / Acc)
Nucl. (F1 / Acc)
only
50.1 / 53.1
75.5 / 75.8
37.1 / 58.9
ICL
54.4 / 53.8
59.3 / 54.9
54.9 / 52.9
CoT
55.7 / 54.7
62.4 / 56.7
57.4 / 53.6
Cal
57.3 / 55.6
60.6 / 55.9
58.7 / 54.1
remain
36.1 / 56.4
35.6 / 55.3
37.1 / 58.9
all
52.6 / 55.6
77.1 / 77.6
37.1 / 58.9
Table 8: Average micro F1 and accuracy for stance prediction using binary classification (for=1,against=0). Aspect prediction Table 9 shows that our method perform worse than our best baseline trained on 80k examples, but perform at a similar level to the official results reported in Table 3 in (Schiller et al., 2021). Additionally our baseline in-
creases performance on the number of topics it has been trained on.Figure 3 visualises the performance of T0 few-shot prediction given k ≤4 examples and baseline models. T0-3B using 4 tweets is competitive to baselines trained on 10k+ tweets, despite the variance of the predictions being rather large, which reflect that using the samples is not always beneficial to the model. We proceed with the ICL setup for generating candidates despite this.
Method
Abor. (F1 / Acc)
Clon. (F1 / Acc)
Nucl. (F1 / Acc)
only
68.5 / 87.7
71.8 / 88.9
73.1 / 89.9
ICL
66.9 / 87.1
66.5 / 86.6
66.1 / 86.3
CoT
67.2 / 87.3
67.7 / 87.7
68.8 / 88.3
Cal
68.2 / 87.8
68.5 / 88.2
68.4 / 88.1
remain
71.6 / 88.7
74.9 / 90.5
75.5 / 91
all
72.9 / 89.4
75.2 / 90.9
76.6 / 91.5
Table 9: Average micro F1 and accuracy for aspect prediction using BIO tags.
# 6 Conclusion
In this paper we introduced a new definition of narratives and how to model these in fine-grained debates with large language models. Our approach is based on parameter efficient fine-tuning using controlled text generation using attributes predicted using a handful of examples. We show that claims generated using our approach are genuine and sensible in general. We fine-tune of model on our own dataset and the augmented UKP-corpus and outperform baseline approaches. In future work, we seek to examine multiple completions and ensembles similar to (Liévin et al., 2023) which enables to include examples of up to 100 examples for ICL, to reduce variance and outperform single-sample CoT methods using larger models (GPT-4, ChatGPT, LLama). Moreover, our approach considers each topic independently using a LLM but could be made to consider all simultaneously.
# Acknowledgements
PEC, SY and SB was supported by the Pioneer Centre for AI, DNRF grant number P1.
# 7 Limitations
Scaling to multiple topics For our approach, the prediction of narratives is topic specific and the number of models scales linearly with the with the topics. This is primarily because both the baseline method using a LM head cannot predict new classes and for the text2text approach it is theoretically possible to simply use one model, though initial experiments suggested a model per topic worked better. Instead of directly predicting the narratives, one could instead have ranked the list of narratives given a tweet. This gives us contextual information about the narratives, since they are written in text and not just as a class and provides a number of benefits including having one model for all topics but also new topics. Additionally it could also provide temporal evaluations by adding new emerging narratives to the list. Scaling to more narratives The current approach requires a domain expert to writing down the particular narratives from the fine grained debate and does not model that there is a countable number of narratives within a specific domain. Finding the particular narratives is bottlenecked by knowing enough about the particular topic. Moreover, since it takes time to gather enough information about the different topics it makes it difficult to scale up to larger numbers of taxons. Future work can explore automatic generation of the narratives given a list of tweets, and condense this list iteratively, and patch templates e.g., using pre-trained language models. Directly modelling the initial argumentative text Finally, the approach we develop can operate on text that is claims or argument discourse units, but has no way of distinguishing between these or nonarguments. This precludes the model from being able to only predict a narrative if the text is indeed from the fine grained debate and can be tricked into providing narratives which the text
# References
Isabelle Augenstein. 2021. Towards explainable fact checking.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners.
Peter Ebert Christensen, Frederik Warburg, Menglin Jia, and Serge Belongie. 2022. Searching for structure in unfalsifiable claims.
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instruction-finetuned language models. Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46. Max Glockner, Yufang Hou, and Iryna Gurevych. 2022. Missing counter-evidence renders nlp fact-checking unrealistic for misinformation. Ivan Habernal and Iryna Gurevych. 2016. Which argument is more convincing? analyzing and predicting convincingness of web arguments using bidirectional LSTM. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics
Shachar Mirkin, Guy Moshkowich, Matan Orbach, Lili Kotlerman, Yoav Kantor, Tamar Lavee, Michal Jacovi, Yonatan Bilu, Ranit Aharonov, and Noam Slonim. 2018. Listening comprehension over argumentative content. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 719–724, Brussels, Belgium. Association for Computational Linguistics. Rahul Mishra, Dhruv Gupta, and Markus Leippold. 2020. Generating fact checking summaries for web claims. In Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020), pages 81– 90, Online. Association for Computational Linguistics. Matan Orbach, Yonatan Bilu, Ariel Gera, Yoav Kantor, Lena Dankin, Tamar Lavee, Lili Kotlerman, Shachar Mirkin, Michal Jacovi, Ranit Aharonov, and Noam Slonim. 2019. A dataset of general-purpose rebuttal. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5591– 5601, Hong Kong, China. Association for Computational Linguistics. Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics. Maja Popovi´c. 2015. chrF: character n-gram F-score for automatic MT evaluation. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 392–395, Lisbon, Portugal. Association for Computational Linguistics.
<div style="text-align: center;">Lance A. Ramshaw and Mitchell P. Marcus. 1995. Tex chunking using transformation-based learning.</div>
ictor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M Rush. 2022. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations.
Benjamin Schiller, Johannes Daxenberger, and Iryna Gurevych. 2021. Aspect-controlled neural argument generation. In Proceedings of the 2021 Conference of the North American Chapter of the Association
Benjamin Schiller, Johannes Daxenberger, and Iryna Gurevych. 2021. Aspect-controlled neural argument generation. In Proceedings of the 2021 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, pages 380–396, Online. Association for Computational Linguistics.
Gabriella Skitalinskaya, Maximilian Spliethöver, and Henning Wachsmuth. 2022. Claim optimization in computational argumentation.
Mehmet Sofi, Matteo Fortier, and Oana Cocarascu. 2022. A robustness evaluation framework for argument mining. In Proceedings of the 9th Workshop on Argument Mining, pages 171–180, Online and in Gyeongju, Republic of Korea. International Conference on Computational Linguistics.
Christian Stab, Tristan Miller, Benjamin Schiller, Pranav Rai, and Iryna Gurevych. 2018. Cross-topic argument mining from heterogeneous sources. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3664– 3674, Brussels, Belgium. Association for Computational Linguistics.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models.
igScience Workshop, :, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon,
Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, Dragomir Radev, Eduardo González Ponferrada, Efrat Levkovizh, Ethan Kim, Eyal Bar Natan, Francesco De Toni, Gérard Dupont, Germán Kruszewski, Giada Pistilli, Hady Elsahar, Hamza Benyamina, Hieu Tran, Ian Yu, Idris Abdulmumin, Isaac Johnson, Itziar Gonzalez-Dios, Javier de la Rosa, Jenny Chim, Jesse Dodge, Jian Zhu, Jonathan Chang, Jörg Frohberg, Joseph Tobing, Joydeep Bhattacharjee, Khalid Almubarak, Kimbo Chen, Kyle Lo, Leandro Von Werra, Leon Weber, Long Phan, Loubna Ben allal, Ludovic Tanguy, Manan Dey, Manuel Romero Muñoz, Maraim Masoud, María Grandury, Mario Šaško, Max Huang, Maximin Coavoux, Mayank Singh, Mike Tian-Jian Jiang, Minh Chien Vu, Mohammad A. Jauhar, Mustafa Ghaleb, Nishant Subramani, Nora Kassner, Nurulaqilla Khamis, Olivier Nguyen, Omar Espejel, Ona de Gibert, Paulo Villegas, Peter Henderson, Pierre Colombo, Priscilla Amuok, Quentin Lhoest, Rheza Harliman, Rishi Bommasani, Roberto Luis López, Rui Ribeiro, Salomey Osei, Sampo Pyysalo, Sebastian Nagel, Shamik Bose, Shamsuddeen Hassan Muhammad, Shanya Sharma, Shayne Longpre, Somaieh Nikpoor, Stanislav Silberberg, Suhas Pai, Sydney Zink, Tiago Timponi Torrent, Timo Schick, Tristan Thrush, Valentin Danchev, Vassilina Nikoulina, Veronika Laippala, Violette Lepercq, Vrinda Prabhu, Zaid Alyafeai, Zeerak Talat, Arun Raja, Benjamin Heinzerling, Chenglei Si, Davut Emre Ta¸sar, Elizabeth Salesky, Sabrina J. Mielke, Wilson Y. Lee, Abheesht Sharma, Andrea Santilli, Antoine Chaffin, Arnaud Stiegler, Debajyoti Datta, Eliza Szczechla, Gunjan Chhablani, Han Wang, Harshit Pandey, Hendrik Strobelt, Jason Alan Fries, Jos Rozen, Leo Gao, Lintang Sutawika, M Saiful Bari, Maged S. Al-shaibani, Matteo Manica, Nihal Nayak, Ryan Teehan, Samuel Albanie, Sheng Shen, Srulik Ben-David, Stephen H. Bach, Taewoon Kim, Tali Bers, Thibault Fevry, Trishala Neeraj, Urmish Thakker, Vikas Raunak, Xiangru Tang, ZhengXin Yong, Zhiqing Sun, Shaked Brody, Yallow Uri, Hadar Tojarieh, Adam Roberts, Hyung Won Chung, Jaesung Tae, Jason Phang, Ofir Press, Conglong Li, Deepak Narayanan, Hatim Bourfoune, Jared Casper, Jeff Rasley, Max Ryabinin, Mayank Mishra, Minjia Zhang, Mohammad Shoeybi, Myriam Peyrounette, Nicolas Patry, Nouamane Tazi, Omar Sanseviero, Patrick von Platen, Pierre Cornette, Pierre François Lavallée, Rémi Lacroix, Samyam Rajbhandari, Sanchit Gandhi, Shaden Smith, Stéphane Requena, Suraj Patil, Tim Dettmers, Ahmed Baruwa, Amanpreet Singh, Anastasia Cheveleva, Anne-Laure Ligozat, Arjun Subramonian, Aurélie Névéol, Charles Lovering, Dan Garrette, Deepak Tunuguntla, Ehud Reiter, Ekaterina Taktasheva, Ekaterina Voloshina, Eli Bogdanov, Genta Indra Winata, Hailey Schoelkopf, JanChristoph Kalo, Jekaterina Novikova, Jessica Zosa Forde, Jordan Clive, Jungo Kasai, Ken Kawamura,
Liam Hazan, Marine Carpuat, Miruna Clinciu, Najoung Kim, Newton Cheng, Oleg Serikov, Omer Antverg, Oskar van der Wal, Rui Zhang, Ruochen Zhang, Sebastian Gehrmann, Shachar Mirkin, Shani Pais, Tatiana Shavrina, Thomas Scialom, Tian Yun, Tomasz Limisiewicz, Verena Rieser, Vitaly Protasov, Vladislav Mikhailov, Yada Pruksachatkun, Yonatan Belinkov, Zachary Bamberger, Zdenˇek Kasner, Alice Rueda, Amanda Pestana, Amir Feizpour, Ammar Khan, Amy Faranak, Ana Santos, Anthony Hevia, Antigona Unldreaj, Arash Aghagol, Arezoo Abdollahi, Aycha Tammour, Azadeh HajiHosseini, Bahareh Behroozi, Benjamin Ajibade, Bharat Saxena, Carlos Muñoz Ferrandis, Danish Contractor, David Lansky, Davis David, Douwe Kiela, Duong A. Nguyen, Edward Tan, Emi Baylor, Ezinwanne Ozoani, Fatima Mirza, Frankline Ononiwu, Habib Rezanejad, Hessie Jones, Indrani Bhattacharya, Irene Solaiman, Irina Sedenko, Isar Nejadgholi, Jesse Passmore, Josh Seltzer, Julio Bonis Sanz, Livia Dutra, Mairon Samagaio, Maraim Elbadri, Margot Mieskes, Marissa Gerchick, Martha Akinlolu, Michael McKenna, Mike Qiu, Muhammed Ghauri, Mykola Burynok, Nafis Abrar, Nazneen Rajani, Nour Elkott, Nour Fahmy, Olanrewaju Samuel, Ran An, Rasmus Kromann, Ryan Hao, Samira Alizadeh, Sarmad Shubber, Silas Wang, Sourav Roy, Sylvain Viguier, Thanh Le, Tobi Oyebade, Trieu Le, Yoyo Yang, Zach Nguyen, Abhinav Ramesh Kashyap, Alfredo Palasciano, Alison Callahan, Anima Shukla, Antonio Miranda-Escalada, Ayush Singh, Benjamin Beilharz, Bo Wang, Caio Brito, Chenxi Zhou, Chirag Jain, Chuxin Xu, Clémentine Fourrier, Daniel León Periñán, Daniel Molano, Dian Yu, Enrique Manjavacas, Fabio Barth, Florian Fuhrimann, Gabriel Altay, Giyaseddin Bayrak, Gully Burns, Helena U. Vrabec, Imane Bello, Ishani Dash, Jihyun Kang, John Giorgi, Jonas Golde, Jose David Posada, Karthik Rangasai Sivaraman, Lokesh Bulchandani, Lu Liu, Luisa Shinzato, Madeleine Hahn de Bykhovetz, Maiko Takeuchi, Marc Pàmies, Maria A Castillo, Marianna Nezhurina, Mario Sänger, Matthias Samwald, Michael Cullan, Michael Weinberg, Michiel De Wolf, Mina Mihaljcic, Minna Liu, Moritz Freidank, Myungsun Kang, Natasha Seelam, Nathan Dahlberg, Nicholas Michio Broad, Nikolaus Muellner, Pascale Fung, Patrick Haller, Ramya Chandrasekhar, Renata Eisenberg, Robert Martin, Rodrigo Canalli, Rosaline Su, Ruisi Su, Samuel Cahyawijaya, Samuele Garda, Shlok S Deshmukh, Shubhanshu Mishra, Sid Kiblawi, Simon Ott, Sinee Sang-aroonsiri, Srishti Kumar, Stefan Schweter, Sushil Bharati, Tanmay Laud, Théo Gigant, Tomoya Kainuma, Wojciech Kusa, Yanis Labrak, Yash Shailesh Bajaj, Yash Venkatraman, Yifan Xu, Yingxin Xu, Yu Xu, Zhe Tan, Zhongli Xie, Zifan Ye, Mathilde Bras, Younes Belkada, and Thomas Wolf. 2023. Bloom: A 176b-parameter open-access multilingual language model.
Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations.
Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations.
Yusen Zhang, Ansong Ni, Ziming Mao, Chen Henry Wu, Chenguang Zhu, Budhaditya Deb, Ahmed Awadallah, Dragomir Radev, and Rui Zhang. 2022. Summn: A multi-stage summarization framework for long input dialogues and documents. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1592– 1604, Dublin, Ireland. Association for Computational Linguistics.
Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. 2021. Calibrate before use: Improving few-shot performance of language models.
# A Implementation Details
Here we describe the implementation details for fine-tuning the 3B T0 model for narrative prediction, in addition to using different ICL strategies for stance and aspect prediction. For all downstream tasks, we use the same AdamW optimiser with linear learning rate decay and weight decay. Finetuning details such as number of epochs and learning rate is reported in Table 10
Aspect prediction For our aspect prediction models we use the standard BERT model to predict a sequence of BIO tokens. We tokenise a given sentence using the TreebankWordTokenizer from the nltk package available for the Python programming language. For the ICL setup we force the T0 model to consider only the words in the given sentence by tokenizing the sentence and feeding it into force_words_ids, additionally we also force the decoding step to not include stop words in addition to special characters like ” that appear in the sentence. During decoding, we set the temperature τ = 0.7, top_p=0.9, number of beams equal to 5 to provide a variety of sentences following the same narrative.
Stance prediction For the Stance prediction we restrict decoding to one word only, and giving the model two choices for or against for the T0 model. For the baseline model we simply attach a LM head and do binary classification 0 =for and 1 =against.
Narrative prediction During finetuning we switch the standard T0 model out with T0-few with the LoRA setup and mainly keep the defaulthyperparameters but reduce the batch size to 4 and train a model for each topic for for 10 epochs. Each model takes around 3hrs to train on the 10k training sentences. In addition to this setup we also include sentences that we generated sentences using the topic, predicted stance and aspect using the CTRLUKP model, T0-3B, T5-flan-3B and 175B Bloom model. The tweets we predict the stance and aspect is from the test set. Using these attributes we can generate similar sentences to the teat set to help enhance performance. We simply copy the target narratives as labels for the generated sentences and include them in the training dataset. To give an example of the runtime for our code it takes 12 hours to complete 10 epochs for the T0-3B model using 1 TitanRTX-24GB and 1 Xeon
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/93e1/93e14344-9de5-4196-9b3a-e7b9296ca32e.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 10: Fine-tuning setting.</div>
E5-2620 v4 8c/16t - 2.1 GHz CPU, and 8 hrs using 1 A100-40GB and the same CPU for the T0-11B parameter model. We always have access to a minimum of 48GB of RAM but run our experiments using 64GB RAM.
# B Search Query and Narrative Synonyms
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8d8c/8d8c118b-bc40-453d-9411-0132351640f6.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 11: Twitter search keywords</div>
Table 11 - 13 lists the Twitter queries we used to retrieve the initial training data. Note that many of the words are either in their stem or shorted format in order to ensure a wider range of search results being returned. Per default twitter filter out sentences that does not contain tokens from the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/80c5/80c5b0a6-5281-43bf-b1f1-3c2e4a66f265.png" style="width: 50%;"></div>
corporate culture
corporate culture HR,company culture HR,work culture HR
corporate culture toxic,company culture toxic,work culture toxic
corporate culture unlawful,company culture unlawful,work culture unlawful
company culture risk,corporate culture risk,work culture risk
corporate culture speak up,company culture speak up,work culture speak up
corporate culture abuse,company culture abuse,work culture abuse
anti union company,don’t trust non profit,corporate culture no trust
company culture no trust,work culture no trust
corporate culture greed,company culture greed
work culture greed,corporate culture millenialcompany culture millenial
work culture millenial,their company do what they want
corporate culture manager,company culture manager,work culture manager
corporate culture stress,company culture stress,work culture stress
corporate culture hard work,work pregnant,side hussle culture
work culture loyalty,corporate culture loyalty,company culture loyalty
work culture remote,corporate culture remote,company culture remote
work culture ethic,corporate culture ethic,company culture ethic
work culture family,corporate culture family,company culture family
corporate culture cult,company culture cult,work culture cult
corporate culture fun,work culture fun,company culture fun
work culture perks,company culture perks,corporate culture perks
job hop look bad,company dress code,corporate mass firing, quite quitting
let it rot job,corporate culture disgusthing
work culture disgusthing, company culture disgusthing
crypto
china crypto ban,china crypto mining,el salvador crypto legal
crypto steal constitution,crypto banking the unbanked
crypto financially free,crypto diversify asset,crypto people of color
crypto trust technology not people,crypto access financial
crypto bank failure,crypto better digital payment,crypto wealth builder
crypto upwards mobility,crypto is an investment,crypto digital gold
crypto short the bankers,crypto not democracy,crypto ruthless investors
Bitcoin is a Platypus,crypto should be regulated,crypto needs rules
crypto is a scam,crypto is for terrorists,crypto is for criminals
crypto rich bailouts,crypto stock bubble,crypto unsustainable environment
crypto ponzi scheme,crypto pump dump,crypto influencer ponzi
crypto carbon tax,crypto great reset ,crypto own nothing happy
crypto money laundering,crypto funding party
crypto same as database, crypto is toxic
baby formula
baby formula scam,baby formula poison,baby formula breat milk
baby formula inflammatory,baby formula infection,baby formula virus
baby formula sustainable,baby formula weight
baby formula replacement feeding,baby formula economic, baby formula hospital
baby formula shame,baby formula guilt,baby formula husband feed
formula feeding mental health,breastfeeding mastitis,breast milk propaganda
breast milk infection,breast milk risk ,breast milk health
breast milk is best,breast milk germ,baby formula propaganda
breastfeeding guilt,breastfeeding negativity, breastfeeding anxiety
breastfeeding public,breastfeed good citizen,breastfeeding shame
breastfeeding sleeping,breastfeeding formula all nothing
breastfeeding gender role,politically correct breastfeeding
<div style="text-align: center;">Table 12: Twitter search keywords</div>
query.
# C Annotation Details
For our crowdsourcing of narrative annotations and human evaluation we use Amazon Mechanical Turk .Workers had to take a qualification test, have an acceptance rate of at least 80% based on 5 question, be located within the US, have successfully completed more than 1000 HITs before and have an approval rate of 98%. We paid 1 dollar per HIT for the dataset task which is to classify one tweet into one in roughly 40 narrative categories. Initially time spend on a HIT is much higher than when they complete their 25th hit as workers learn to memories the categories. For the human evaluation we get annotations from two crowdworkers and pay 2 dollars per HIT. Consent was obtained from the crowdworkers by including the warning for the pretest annotation: "By completing this test you will agree that subsequent HITs using this pretest as a prerequisite can be used for data collection in relation to research projects", similarly for
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0876/08761ce6-dcdd-4f78-a4bd-dd55690496b2.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 13: Twitter search keywords</div>
# Table 13: Twitter search keywords
we get consent from people whose data we are using though the Twitter Term of Services. The data collection procedure was approved by our internal ethics review board.
During our annotation of the narrative labels we discovered that the returned answers tend to be biased towards the top 10 first possible answers that could be selected in the HIT. To mitigate potential bias we manually went though the top 3 most frequent answer for each topic in the validation set and relabel the corresponding tweets.
# D Crowdsourced Annotations
For this paper, gathering annotations has happened over three annotations rounds, each focusing on different sections of the paper.
# D.1 Pretest
The first crowdsourcing task is that of a pretest, which is used to determine if workers are suitable for our main annotation mask. It is based on data from (Hansen and Hershcovich, 2022) and focuses on correctly classifying two different types of labels: Pro/con and evidence.
Pro/con is a binary label. The tweet is annotated as (+1) for pro when a clear claim has a positive or supportive stance towards the topic. It is annotated as (-1) when it has a clearly antagonistic or attacking stance towards it the topic. We exclude data for which has no clear stance.
Instructions for annotators: Given a tweet your task is to annotate it with its stance in relation to the topic topic. The stance is either pro or con (for or against a topic). In this case select pro if you find that the tweet is supportive towards the topic, and con if it is hostile instead. Remember that a tweet with hostile remarks can still be supportive of the topic, as we want to find the stance towards the topic and not the tweet itself.
# D.1.2 Evidence
Evidence as a label has 6 classes. The tweet is annotated using any of the labels: Normative, Study, Expert, Fact, Anecdotal or unrelated/no evidence. The description for each of these labels are taken from (Hansen and Hershcovich, 2022): Anecdotal refers to "a description of an episode(s), centred on individual(s) or clearly located in place and/or in time." Expert refers to a "testimony by a person, group, committee, organisation with some known expertise / authority on the topic. Study refers to "results of a quantitative analysis of data, given as numbers, or as conclusions" Fact refers to "A known piece of information about the world without a clear source for the information" Normative refers to "an added description for a belief about the world" No evidence refers to "the tweet does contain evidence, but it is not related to the topic, or it does not have any evidence."
Instructions for annotators: The task is to annotate a tweet with the type of evidence it contains.
Evidence is a statement used to support or attack a topic or claim. Evidence can be present in combination with a claim, or it can also be self-contained if it is just stating facts or referencing studies related to the topic. If the evidence is unrelated to the discussed topic, it is marked as unrelated. If you feel that multiple types of evidence is present in the tweet, choose the one that you think best describes the main piece of evidence in the tweet. Remember that your task is to annotate the type of evidence that is in the tweet regardless of your views and if the evidence is true or not.
# D.2 Narrative annotation
The main crowdsourcing task of this paper is essentially claim classification. Given a tweet the workers determine if the tweet is a claim or argument with an evidence type that is not a study (as taken from the definition of study in (Hansen and Hershcovich, 2022)). Then if the tweet is a claim then they should select the most similar claim from a list of options. If no option is suitable, they should select "No claim in list is similar to the tweet".
to annotate a tweet given a list of claims that the tweet might be similar to. Of course, each tweet can be relevant for more than one claim, but it can also be irrelevant and should be annotated as such. Therefore, given that the topic is {} select the claim which you find the tweet most similar to (regardless of your views on the list of claims, the topic and the tweet itself). Remember that the surrounding context of a tweet can be missing, and that people may be sarcastic.
# D.3 Human evaluation of generated claims
The last crowd sourcing campaign is the human evaluation in which we evaluate how well a generated claim compared against the original claim (It is generated from the predicted stance and aspect from the original claim ). We follow primarily (Skitalinskaya et al., 2022) for definition of argument quality, meaning and fluency, but also (Schiller et al., 2021) for fluency and persuasiveness. These generated claims are then used for finetuning a LLM for improved narrative prediction. Instructions for annotators: In this task, you will identify if a generated claim is similar to or has improved, without changing the overall meaning of the text. Each field contains a par of tweets, one being the original and the other a synthetic tweet
that is trying to mimic it. Please rate each candidate along the following four perspectives: argument quality, fluency, meaning and persuasiveness. Argument Quality has a scale from 1 to 5: 1 (notably worse than original), 2 (slightly worse), 3 (same as original), 4 (slightly improved), 5 (notably improved) Does the generated claim improve over the original claim? Things to look for include: specifying a fact, simplifying the sentence, adding clarity, adding additional information such as facts, adding, editing or removing links for external resources. Meaning has a scale from 1 to 5: 1 (entirely different), 2 (substantial differences), 3 (moderate differences), 4 (minor differences), 5 (identical) Here we wish to measure if the generated claim have the same overall meaning as the original. Adding extra information that does change the objects or events described in the claim should not penalise the score. Persuasiveness runs from 1 to 3. 1 (generated text less persuasive than original), 2 (equally persuasive), 3 (generated text is more persuasive) (choose one argument as being more persuasive or both as being equally persuasive.) Here we wish to measure if the generated claim is more useful in a debate about a certain topic than the original claim. Adding additional text that explains an event or fact more in depth should be rewarded. Fluency runs from a scale form 1 to 3: 1 (major errors, disfluent), 2 (minor errors), 3 (fluent) Here we want you to to compare the generated sentence with the original one and ask if the sentence is written in fluent English and makes sense? You should consider rewarding the generated claim in case of improved grammar, spelling and punctuation of generated claim over the original claim.
# E Narratives per topic
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0091/0091dd30-de05-4e45-8d4a-90ce011993f4.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 14: First list of narratives</div>
Topic
Narrative
Alternative meat
we could stop subsidising highly processed foods
Investing in Meat alternatives is good and profitable
meat production is not sustainable
alternative meat is not viable for a healthy diet
big pharma is behind the alternative meat
animals eat meat so humans should too
Eating meat is immoral
alternative meat does have enough proteins
plant based food is made to remove meat
meat cause cancer and can be deadly
plant based food are sustainable food
red meat is bad
We should subsidise Meat alternatives and Plant based food
Being Vegetarian or vegan allows you to be healthy
plant food and meat alternatives is great
animals are not ingredients
You do not need meat to hit the gym often
transport of goods is more harmful to the planet than meat or plants
alternative meat is a pyramid scheme
alternative meat tastes bad
alternative meat is unhealthy as a diet
alternative meat is forced upon the consumer
Plant based meals are highly processed and is not good
We should eat plants to save planet
alternative meat is fake
Eating fewer plants and more meat will save the plant
We should reduce meat consumption to protect the planet
We should import a carbon tax to food production
We should increase production of meat
We should stop subsidising meat to allow for alternative meat
Eating meat is murder
exempt meat production from carbon taxes
Being flexitarian allows you to get enough nutritions
fresh organic food is good
A vegan diet is unsustainable for the planet
soy meat will not and cannot replace meat
Eating bugs instead of meat will never be reality
No claim in the list is describing the tweet
Attractiveness
Women and especially lesbians are exploiting the male gaze
Forcing your standard of beauty on every women is trans phobic
The male gaze encourages physical and sexual violence against women
the lack of beauty standards warrants cheating
beauty standards are pathetic and fake
Academia is like any other industry where beauty standards play into how women are treated
the female gaze and male gaze are distorted terms used on social media
Just because people do not fit the beauty standard, does not mean that you can disrespect them.
misogynists hate women that take back ownership of their bodies and reject beauty standards
beauty standards serve to perpetuate a misogynistic society
Hygienic actions like shaving is beyond beauty standards or gender roles
Beauty standards are sexist
We should be free from sexual objectification and beauty standards
Corporations try to make you buy stuff though beauty standards
beauty standards are unrealistic
feminism and gender bending is enforcing stereotypical beauty standards
beauty standards are toxic
women who do not meet conventional beauty standards are not women
beauty standards is nothing but a money making scheme
beauty standards that cater to minorities are trained to be inclusive
Women who don’t fit societal beauty standards get catcalled and harassed
social media attempts to hierarchize beauty to maintain dominance over others
beauty standards are hard to escape
Masculinity is not toxic but attractive
Beauty standards are bad and stressful for young people
No natural humans look like that
Beauty is everywhere
Beauty standards are racist
Hypermasculinity in of itself is the problem
No claim in the list is describing the tweet
Cloning
clones can perfect can give humans preferable qualities
Cloning can save lives or cure humans
Scientist that clone are acting unethically
Couples without kids would rather use cloning than employ surrogates or IVF
transplanting organs can be made easier and more successful by cloning
Cloning can potentially create premature ageing
Cloning in morally wrong
cloning could provide childless couples with an enhanced or enlarged family
cloning can be use to reduce risks
Cloning will cause parents to customise their children
Cloning is medicine; advances in cloning are advances in medicine
People could keep on living due to cloning
Cloning affects negatively to the reproductive processes
Better cloning techniques offer higher chances of success with less moral hazards
People that get cloned maintain their personality
Cloning is playing God
Cloning is for evil purposes
Humans will become a product with cloning
Cloning animals is cruel
Cloning someone is not a safe thing to do
Cloning is a natural
Cloned people do not have souls
Cloning will be accepted some day
Cloning is akin to murder or manslaughter
You lose a sense of individuality when cloned
No claim in the list is describing the tweet
<div style="text-align: center;">Table 15: Second list of narratives</div>
Topic
Narrative
Corporate culture
non profit companies and for profit companies are equally untrustworthy
Remote work makes it difficult to maintain company culture
Jobs needs to be treated with respect
Companies simply want drones that do not ask questions
there is a lot of hustle culture for millennials
Working remove does not mean that company culture are not important
If you want a great company culture you should hire great people with a good work ethic
Corporate culture is bad
corporate culture is racist
Companies can do what they want to do
millennials do not want to work
side effect of hustle culture is often a counterproductive narrowing of focus
corporations are getting tax cuts while you are getting fired
So many popular companies actually have a terrible culture and a bad work ethic
Victims of abuse are ignored and silenced
corporations that mass fire employees say that Nobody wants to work in the media or get record high profits
mass firings are commonplace in corporate takeover
Our company culture is good because we have fun
corporate culture do not reward hard work
job hopping looks bad on your resume
Companies are anti union
corporate culture so rotten to the core by greed
Getting a stable job is getting harder over time
A fun company culture does not care about your work life balance
Some office cultures are like a cult and are not healthy for you
loyalty to the company trumps everything
Managers of a company cause nothing but trouble
perks from a company are useless
Young people refuse to enter or stay in the workforce
Companies that make you feel like a family is good
Companies that say they that you are a family is brainwashing you to comply
Hard work gets you everywhere
corporate culture promising generous pay and perks in order to mask workers disposability and exploitation
loyalty to the company means absolutely nothing in this day and age
You can make a remote workers feel part of the team without being in the same physical space
An employee is a representative of a company and should respect the dress code and look professional
work dress code is discriminatory
Corporations that do mass firings are greedy
Do not trust companies
No claim in the list is describing the tweet
Crypto
People will sell you out but you can trust crypto
crypto is used solely for pump and dump schemes
crypto allows a complete transformation of the global economy though the great reset
Influencer are scamming their fans using crypto
crypto is ponzi scheme
crypto is a scam
crypto is for money laundering
crypto is undiscriminatory to people of color
crypto should be regulated
crypto is legal way to pay
crypto is hijacked by ruthless investors
crypto is for terrorists
crypto is toxic
crypto is just a way to diversify your assets
crypto can bank the unbanked
crypto is for criminals
crypto is a mean for the rich to get bailouts
crypto is unsustainable for the environment
crypto better digital payment than credit cards
crypto is used to fund political parties
crypto is yet another tech bubble ready to burst and fail
Bitcoin is like a Platypus it is not real
crypto is simply digital gold
crypto allows one to become financially free
crypto allows easy access to financial services
crypto can help fighting greedy bankers
crypto is an investment
crypto is used to steal the constitution
Crypto is not democratic
crypto should be carbon taxed
crypto is useful when banks are failing
crypto mining should be banned
blockchain is no different from a database
crypto is a wealth builder
WEF want you to own nothing and be happy, crypto avoids this problem
crypto provides upwards mobility
No claim in the list is describing the tweet
<div style="text-align: center;">Table 16: Third list of narratives</div>
Topic
Narrative
(Baby) Formula
breastfeeding is natural and is not politically correct
women are pressured into breastfeeding and gets stressed
baby formula is costly
baby formula is killing babies
It does not have to be all or nothing
the political right vote against bills to make things more expensive
breastfeeding and baby formula is risky
baby formula ends up in foreign countries instead of the US where it should be
Some people are allergic to breast milk and need formula
People hate babies when they make abortion illegal and remove baby formula
baby food industry is promoting propaganda
breastfeeding can cause HIV
baby formula is poisonous
baby formula is good as fathers can feed their baby
baby formula is best if you cannot breastfeed
It is important to secure enough baby formula in an economic crisis
breast milk is best
The political left is out to remove babies
People are reselling baby formula to other countries for higher prices
Do what is best for you and the baby
The baby formula shortage is one big scam
Some people cannot tolerate formula and need milk
breast milk has a lot of antibodies that can help the baby fight off infection
people publicly shame women who breastfeed in public
mental health is more important than breast feeding
breastfeeding is healthier than baby formula
baby formula can cause infection
breastfeeding will lowers risk of breast cancer
The political right have caused a baby formula shortage
breast is best campaign causes anxiety in moms who cannot breastfeed
No claim in the list is describing the tweet
Influencers
Influencers can be damaged by everything they say
social media in the west is like opium for kids
influencers just want to get rich quick
influencer marketing are not authentic
social media career is not sustainable
influencers are creative
influencers earn too much money
social media people are toxic and rude
an influencer is a social media celebrity
Normal jobs are boring
influencing is indeed hard work
influencers understand the use cases of products and want to help
you should not quit your job and become an influencer
social media people are just plagiarising other people
Becoming an influencer allows you to live the good life
influencers do not know hard work
influencers wants to be their own boss
dealing with hate is part of a social media job
the numbers of followers do not make you successful
influencers are not respected
influencer is not an adult job
jobless youth are spending too much time on social media platforms
If you have too few followers you should get a job
influencers are wasting their time
being an influencer is easy
influencer is not a real job title
No claim in the list is describing the tweet
Mental Health in sports
male dominated sports are toxic for women
In sports people get into drugs when mental health declines
sport help you alleviate stress
The money made in sports should go to mental health organisations not admin staff
team sport is a brutal business
female athlete are not projected
When athletes get in trouble the blame the media
sports is like religion it is bad for your mental health
In sports racism and mental health issues goes hand in hand
athletes do not have any problems
athletes are or must become hard workers
athletes does not have real mental health problems
Work late nights, don’t take sick days
Be tough, vulnerability is weakness
Entering sports in an early age led to burnout
getting help is stigmatising
elite athletes have unfair genetic advantages
sports athletes are manipulated
mental health is not masculine
athletes are only thriving professionally if they thrive personally
mental health should not be treated like the flu
trans people should not participate in male or female sports
be ashamed if talking about mental health
sports athletes are depressed
you do as your told as an athlete
alienation cause mental health issues in sports
athletes should not worry because they have a lot of money
talking about mental health is showing weakness
athletes that speaks up about health issues are silenced
No claim in the list is describing the tweet
Table 17: Fourth List of narratives
Topic
Narrative
Nuclear Energy
Nuclear reactor is easy to control
Deciding what to do with regards to long term disposal of nuclear energy waste is difficult
Nuclear energy will be available for use longer than oil for example
Nuclear energy will contaminate the environment
Nuclear energy is dangerous
Nuclear energy can give us unlimited energy
nuclear energy waste can be recycled
Nuclear energy is good
Using nuclear energy to solve problems that arise is logical
Nuclear energy leads to more violence
nuclear power produce carbon free energy
nuclear energy is not efficient
nuclear power is financially burdensome
There is no significant risk with nuclear energy that cannot be said about other agents as well
nuclear energy is dirty
nuclear energy is not safe
nuclear energy makes poor nations dependant on rich nations
Every country can use nuclear energy unlike everything else
Nuclear energy relies too heavily on subsidies
There is not a good plan for storing or disposing of nuclear energy waste so we should use it
Nuclear plants only produce electricity and cannot replace oil and gas
Using nuclear power will lead to nuclear war.
renewable energy is a more viable option than nuclear energy
Nuclear energy are favoured by certain social structures like capitalism
decentralised nuclear energy production is efficient
Nuclear power is needed to stabilise climate change.
Nuclear energy should not even be considered as an energy source
Nuclear energy is much more harmful than beneficial
Green energy will make nuclear energy obsolete
Nuclear energy will increase the cancer in humans
nuclear energy is not renewable energy
Nuclear reactors are vulnerable to terrorist attack
nuclear energy is more reliable than renewable energy sources like solar
No claim in the list is describing the tweet
Transport
trains are better than flights
public transportation is unsustainable for rural areas
public transportation is useless car is better
cars give you the freedom of Independence
buses are safer than cars
fewer drivers equals safer streets
public transportation is comfortable
public transport results in less pollution
trains are better for the climate
public transportation has no personal space
It is important that public transit works
cars are good when there is no good alternative
flights are better than trains
public transportation is for poor people
public transportation is ridden with disease
highways are not profitable
public transit only works if affordable
using bikes are dangerous
public transportation is filled with germs
cars are worse than buses as it carries less people
buses are better than cars
public transportation is good business
taxing car and roads hurt the poor people
cars are for rich people
public transportation is unsafe at night
car centric infrastructure is bad
trains are too expensive
No good transportation is a reflection of the government
cycling will decrease car traffic
public transit is not profitable
public transportation is good
No claim in the list is describing the tweet
<div style="text-align: center;">Table 18: Fifth list of narratives</div>
