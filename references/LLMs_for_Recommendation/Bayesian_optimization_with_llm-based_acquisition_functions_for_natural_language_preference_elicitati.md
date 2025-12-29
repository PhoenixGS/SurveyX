# ayesian Optimization with LLM-Based Acquisition Functions Natural Language Preference Elicitation

David Eric Austin ∗
deaustin@uwaterloo.ca University of Waterloo Waterloo, Ontario, Canada

ABSTRACT

Designing preference elicitation (PE) methodologies that can quickly ascertain a user’s top item preferences in a cold-start setting is a key challenge for building effective and personalized conversational recommendation (ConvRec) systems. While large language models (LLMs) enable fully natural language (NL) PE dialogues, we hypothesize that monolithic LLM NL-PE approaches lack the multiturn, decision-theoretic reasoning required to effectively balance the exploration and exploitation of user preferences towards an arbitrary item set. In contrast, traditional Bayesian optimization PE methods define theoretically optimal PE strategies, but cannot generate arbitrary NL queries or reason over content in NL item descriptions – requiring users to express preferences via ratings or comparisons of unfamiliar items. To overcome the limitations of both approaches, we formulate NL-PE in a Bayesian Optimization (BO) framework that seeks to actively elicit NL feedback to identify the best recommendation. Key challenges in generalizing BO to deal with natural language feedback include determining: (a) how to leverage LLMs to model the likelihood of NL preference feedback as a function of item utilities, and (b) how to design an acquisition function for NL BO that can elicit preferences in the infinite space of language. We demonstrate our framework in a novel NL-PE algorithm, PEBOL, which uses: 1) Natural Language Inference (NLI) between user preference utterances and NL item descriptions to maintain Bayesian preference beliefs, and 2) BO strategies such as Thompson Sampling (TS) and Upper Confidence Bound (UCB) to guide LLM query generation. We numerically evaluate our methods in controlled simulations, finding that after 10 turns of dialogue, PEBOL can achieve an MRR@10 of up to 0.27 compared to the best monolithic LLM baseline’s MRR@10 of 0.17, despite relying on earlier and smaller LLMs. 1

∗ Both authors contributed equally to this research. 1 Our code is publically available at https://github.com/D3Mlab/llm-pe.

∗ Both authors contributed equally to this research. 1 Our code is publically available at https://github.com/D3Mlab/llm-pe.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. RecSys ’24, October 14–18, 2024, Bari, Italy © 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0505-2/24/1 https://doi.org/10.1145/3640457.3688142

Anton Korikov ∗
Armin Toroghi Scott Sanner anton.korikov@mail.utoronto.ca University of Toronto Toronto, Ontario, Canada

CCS CONCEPTS
• Information systems → Recommender systems;  Personalization; Language models.

# KEYWORDS

Conversational Recommendation, Preference Elicitation, Bayesian Optimization, Online Recommendation, Query Generation

ACM Reference Format: David Eric Austin, Anton Korikov, Armin Toroghi, and Scott Sanner. 2024. Bayesian Optimization with LLM-Based Acquisition Functions for Natural Language Preference Elicitation. In 18th ACM Conference on Recommender Systems (RecSys ’24), October 14–18, 2024, Bari, Italy. ACM, New York, NY, USA, 19 pages. https://doi.org/10.1145/3640457.3688142

# 1 INTRODUCTION

Personalized conversational recommendation (ConvRec) systems require effective natural language (NL) preference elicitation (PE) strategies that can efficiently learn a user’s top item preferences in cold start settings, ideally requiring only an arbitrary set of NL item descriptions. While the advent of large language models (LLMs) has introduced the technology to facilitate NL-PE conversations [14, 23] we conjecture that monolithic LLMs have limited abilities to strategically conduct active, multi-turn NL-PE dialogues about a set of arbitrary items. Specifically, we hypothesize that LLMs lack the multi-turn decision-theoretic reasoning to interactively generate queries that avoid over-exploitation or over-exploration of user-item preferences, thus risking over-focusing on already revealed item preferences or wastefully exploring preferences over low-value items. Further challenges faced by monolithic LLM NL-PE approaches include the need to jointly reason over large, potentially unseen sets of item descriptions, and the lack of control and interpretability in system behaviour even after prompt engineering or fine-tuning [28]. In contrast, conventional PE algorithms [21, 22, 27, 39, 40], including Bayesian optimization methods [2, 5, 12, 32, 36], establish formal decision-theoretic policies such as Thompson Sampling (TS) and Upper Confidence Bound (UCB) [16] to balance exploration and exploitation with the goal of quickly identifying the user’s most preferred items. However, these techniques typically assume a user can express preferences via direct item ratings or comparisons – an unrealistic expectation when users are unfamiliar with most items [1]. While recent work has extended Bayesian PE to a fixed set of template-based queries over pre-defined keyphrases [36], no

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e5b2/e5b26fa9-3303-47bc-9617-261e4b27139e.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: PEBOL’s belief updates over a cold-start user’s item utilities during three turns of NL dialo beliefs not only facilitate recommendation, but also enable Bayesian optimization policies to guid avoiding over-exploration (asking about clearly low-value items) and over-exploitation (over-focusin
</div>
• We introduce the first Bayesian optimization formalization of NL-PE for arbitrary NL dialogue over a generic set of NL item descriptions – establishing a new framework for research on steering LLMs with decision-theoretic reasoning.
• We present PEBOL (P reference E licitation with B ayesian O ptimization augmented L LMs), a novel NL-PE algorithm which 1) infers item preferences via Natural Language Inference (NLI) [37] between dialogue utterances and item descriptions to maintain Bayesian preference beliefs and 2) introduces LLM-based acquisition functions, where NL query generation is guided by decision-theoretic strategies such as TS and UCB over the preference beliefs.
•  We numerically evaluate PEBOL against monolithic GPT3.5 and Gemini-Pro NL-PE methods via controlled NL-PE dialogue experiments over multiple NL item datasets and levels of user noise.
• We observe that after 10 turns of dialogue, PEBOL can achieve a mean MRR@10 of up to 0.27 compared to the best monolithic LLM baseline’s MRR@10 of 0.17, despite relying on earlier and smaller LLMs.

# 2 BACKGROUND AND RELATED WORK

# 2.1 Bayesian Optimization

Given an objective function 𝑓: X → R, (standard) optimization systematically searches for a point 𝑥 ∗ ∈X that maximizes 2 𝑓. Bayesian optimization focuses on settings where 𝑓 is a black-box function which does not provide gradient information and cannot be evaluated exactly – rather, 𝑓 must be evaluated using indirect or noisy observations which are expensive to obtain [10, 30]. To address these challenges, Bayesian optimization maintains probabilistic beliefs over 𝑓 (𝑥) and its observations to guide an uncertainty-aware optimization policy which decides where to next observe 𝑓 (𝑥). Bayesian optimization begins with a prior 𝑝 (𝑓) which represents the beliefs about 𝑓 before any observations are made. Letting 𝑦 𝑖

2 We take the maximization direction since this paper searches for items with maximum utility for a person.

represent a noisy or indirect observation of 𝑓 (𝑥 𝑖), and collecting a sequence of observations into a dataset D = (x, y), an observation model defines the likelihood 𝑝 (D| 𝑓). We then use the observed data and Bayes theorem to update our beliefs and obtain the posterior

(1)

(D)
This posterior informs an acquisition function 𝛾 (𝑥 |D)  which determines where to next observe 𝑓 (𝑥)  in a way that balances exploitation (focusing observations where 𝑓 is likely near its maximum) with exploration (probing areas where 𝑓 has high uncertainty).

# 2.2 Preference Elicitation

PE has witnessed decades of research, and includes approaches based on Bayesian optimization (e.g., [3, 8, 11, 13, 18]), Bandits (e.g., [5, 24, 25, 40]), constrained optimization [29], and POMDPs [2]. In the standard PE setting, a user is assumed to have some hidden utilities u = [𝑢 1, ...,𝑢 𝑁] over a set I of 𝑁 items, where item 𝑖 is preferred to item 𝑗 if 𝑢 𝑖> 𝑢 𝑗. The goal of PE is typically to search for an item 𝑖 ∗ ∈ arg max 𝑖 𝑢 𝑖 that maximizes user utility in a minimal number of PE queries, which most often ask a user to express item preferences as item ratings (e.g., [3, 5, 24, 25, 40]) or relative preferences between item pairs or sets (e.g., [2, 8, 11, 12, 14, 32]). An alternative form of PE asks users to express preferences over predefined item features, also through rating- or comparison-based queries [22, 27, 39]. Central to the above PE methods are query selection strategies that balance the exploration and exploitation of user preferences, with TS and UCB algorithms (cf. Sec. 4.2) often exhibiting strong performance [5, 27, 36, 39, 40]. However, none of these methods are able to interact with users through NL dialogue or reason about NL item descriptions.

# 2.3 Language-Based Preference Elicitation

Yang et al. [36] introduce Bayesian PE strategies using TS and UCB for keyphrase rating queries, where keyphrases are first mined from NL item reviews and then co-embedded with user-item preferences in a recommendation system. Handa et al. [14] propose using LLMs to interface with a conventional Bayesian PE system, suggesting a preprocessing step to extract features from NL descriptions and a verbalization step to fluidly express pairwise item comparison

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/62b6/62b6bbf0-4974-4c99-9aa3-53574b032df1.png" style="width: 50%;"></div>
Figure 2: The PEBOL NL-PE algorithm, which maintains a Bayesian belief state over a user’s item preferences given an arbitrary set of NL item descriptions x. This belief is used by a decision-theoretic policy to balance the exploration and exploitation of preferences by strategically selecting an item description 𝑥 𝑖 𝑡 as the basis for LLM query generation. Belief updates are computed through Bayesian inference with NLI entailment scores between item descriptions and query-response pairs.

queries. Li et al. [23] prompt an LLM to generate PE queries for some specific domain (e.g., news content, morals), observe user responses, and evaluate LLM relevance predictions for a single item. While these works make progress towards NL-PE, they do not study how LLM query generation can strategically explore user preferences towards an arbitrary item set outside the realm of item-based or category-based feedback.

# 2.4 Conversational Recommendation

Recent work on ConvRec uses language models 3 to facilitate NL dialogue while integrating calls to a recommender module which generates item recommendations based on user-item interaction history [4, 26, 33, 35]. He et al. [15] report that on common datasets, zero-shot GPT-3.5/4 outperforms these ConvRec methods, which generally use older language models and require user-item interaction history for their recommendation modules.

# 2.5 Natural Language Inference

Binary Natural Language Inference (NLI) [37] models predict the likelihood that one span of text called a premise is entailed by (i.e., can be inferred from) a second span called the hypothesis. For example, an effective NLI model should predict a high likelihood that the premise “I want to watch Iron Man” entails the hypothesis “I want to watch a superhero movie”. As illustrated by this example, the hypothesis typically must be more general than the premise. NLI models are trained by fine-tuning encoder-only LLMs on NLI datasets [6, 31, 34], which typically consist of short text spans for the premise and hypothesis – thus enabling relatively efficient performance on similar tasks with a fairly small number LLM parameters.

# 3 PROBLEM DEFINITION

We now present a Bayesian optimization formulation of NL-PE. The goal of NL-PE is to facilitate a NL dialogue which efficiently discovers a user’s most preferred items out of a set of 𝑁 items. Each item 𝑖 ∈I has a NL description 𝑥 𝑖, which might be a title, long-form description, or even a sequence of reviews, with the item
3 Earlier systems (e.g. [4, 26]) use relatively small RNN-based language models.

set I collectively represented by x ∈X with x = [𝑥 1, ...,𝑥 𝑁]. We assume the user has some (unknown) utility function 𝑓: X → R establishing hidden utilities u = 𝑓 (x) so that item 𝑖 is preferred to item 𝑗 if 𝑢 𝑖> 𝑢 𝑗. Our goal is to find the most preferred item(s):

(2)

∈I
In contrast to standard Bayesian PE formalisms (c.f. Sec 2.2), we do not assume that the user can effectively convey direct item-level preferences by either: 1) providing item ratings (i.e., utilities) or 2) pairwise or listwise item comparisons. Instead, we must infer user preferences by observing utterances during a NL system-user dialogue. At turn 𝑡 of a dialogue, we let 𝑞 𝑡 and 𝑟 𝑡 be the system and user utterance, respectively, with q 𝑡 = [𝑞 1, ...,𝑞 𝑡] and r 𝑡 = [𝑟 1, ...,𝑟 𝑡] representing all system and user utterances up to 𝑡. In this paper, we call 𝑞 𝑡 the query and 𝑟 𝑡 the response, though extensions to more generic dialogues (e.g., when users can also ask queries) are discussed in Section 7. We let H 𝑡 = (q 𝑡, r 𝑡) be the conversation history at turn 𝑡. To formulate NL-PE as a Bayesian optimization problem, we place a prior belief on the user’s utilities 𝑝 (u | x), potentially conditioned on item descriptions since they are available before the dialogue begins. We then assume an observation model that gives the likelihood 𝑝 (r 𝑡 | x, u, q 𝑡), letting us define the posterior utility belief as t (3)

(3)

(| H) ∝(|)(|) This posterior informs an acquisition function 𝛾 (x, H 𝑡)  which generates 4 a new NL query

(4)

( H)
to systematically search for 𝑖 ∗. The preference beliefs also let us define an Expected Utility (EU) 𝜇 𝑡 𝑖 for every item as

(5)

(|H)
which allows the top𝑘 items to be recommended at any turn based on their expected utilities. Our Bayesian optimization NL-PE paradigm lets us formalize several key questions, including:

4 To represent the generative acquisition of NL outputs, we deviate from the conventional definition of acquisition functions as mapping to R.

4 To represent the generative acquisition of NL outputs, we deviate from the conventional definition of acquisition functions as mapping to R.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/308e/308ed71f-5ba9-4118-857e-68c32301aefc.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
Figure 3: Cherry-picked system-generated dialogues from our NL-PE experiments. The Monolithic GPT-3.5 dialogue (left) demonstrates over-exploitation, with 𝑞 3 directly extending 𝑞 2 after a positive user preference is observed and leading to the xtreme case of query repetition (𝑞 4 = 𝑞 3). In contrast, PEBOL (right) continues exploring even after a positive response, while ocusing on promising aspects (three out of four queries elicit a positive response) by using UCB-guided query generation.

These questions reveal a number of novel research directions discussed further in Section 7. In this paper, we present PEBOL, a NL-PE algorithm based on the above Bayesian optimization NL-PE formalism, and numerically evaluate it against monolithic LLM alternatives through controlled, simulated NL dialogues (cf. Sec. 6).

# 4 METHODOLOGY

Limitations of Monolithic LLM Prompting. An obvious NL-PE approach, described further as baseline in Section 5.1, is to prompt a monolithic LLM with all item descriptions x, dialogue history H 𝑡, and instructions to generate a new query at each turn. However, providing all item descriptions [𝑥 1, ...,𝑥 𝑁] in the LLM context window is very computationally expensive for all but the smallest item sets. While item knowledge could be internalized through finetuning, each item update would imply system retraining. Critically, an LLM’s preference elicitation behaviour cannot be controlled other than by prompt-engineering or further fine-tuning, with neither option offering any guarantees of predictable or interpretable behaviour that balances the exploitation and exploration of user preferences.

PEBOL Overview. We propose to addresses these limitations by augmenting LLM reasoning with a Bayesian Optimization procedure in a novel algorithm, PEBOL, illustrated in Figure 2. At each turn 𝑡, our algorithm maintains a probabilistic belief state over user preferences as a Beta belief state (cf. Sec. 4.1). This belief state guides an LLM-based acquisition function to generate NL queries explicitly balancing exploration and exploitation to uncover the top user preferences (cf. Sec. 4.2). In addition, our acquisition function reduces the context needed to prompt the LLM in each turn from all 𝑁 item descriptions x  to a single strategically selected item description 𝑥 𝑖 𝑡. PEBOL then uses NLI over elicited NL preferences and item

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d498/d49882a1-b8ea-4c56-95ec-faffea241498.png" style="width: 50%;"></div>
descriptions to map dialogue utterances to numerical observatio (c.f. Sec 4.3).

# 4.1 Utility Beliefs

4.1.1 Prior Beliefs. Before any dialogue, PEBOL establishes an uninformed prior belief 𝑝 (u) on user-item utilities. We assume item utilities are independent so that

(6)

and that the prior for each utility 𝑢 𝑖 is a Beta distribution

Since this paper focuses on fully cold start settings, we assume a uniform Beta prior with (𝛼 0 𝑖, 𝛽 0 𝑖) = (1, 1). Beta distributions, illustrated in Figure 1, lie in the domain [0, 1]– a normalized interval for bounded ratings in classical recommendation systems. We can thus interpret utility values of 𝑢 𝑖 = 1 or 𝑢 𝑖 =  0 to represent a complete like or dislike of item 𝑖, respectively, while values 𝑢 𝑖 ∈(0, 1) provide a strength of preference between these two extremes.
4.1.2 Observation Model. To perform a posterior update on our utility beliefs given observed responses r 𝑡, we need an observation model that represents the likelihood 𝑝 (r 𝑡 | x, u, q 𝑡). Modelling the likelihood of r 𝑡 is a challenging task, so we will require some simplifying assumptions. Firstly, we assume that the likelihood of a single response 𝑟 𝑡 is independent from any previous dialogue history H 𝑡 − 1, so that:

Since this paper focuses on fully cold start settings, we assume a uniform Beta prior with (𝛼 0 𝑖, 𝛽 0 𝑖) = (1, 1). Beta distributions, illustrated in Figure 1, lie in the domain [0, 1]– a normalized interval for bounded ratings in classical recommendation systems. We can thus interpret utility values of 𝑢 𝑖 = 1 or 𝑢 𝑖 =  0 to represent a complete like or dislike of item 𝑖, respectively, while values 𝑢 𝑖 ∈(0, 1) provide a strength of preference between these two extremes.

4.1.2 Observation Model. To perform a posterior update on our utility beliefs given observed responses r 𝑡, we need an observation model that represents the likelihood 𝑝 (r 𝑡 | x, u, q 𝑡). Modelling the likelihood of r 𝑡 is a challenging task, so we will require some simplifying assumptions. Firstly, we assume that the likelihood of a single response 𝑟 𝑡 is independent from any previous dialogue history H 𝑡 − 1, so that:

(8)

Note that this independence assumption will allow incremental posterior belief updates, so that

(9)

4.1.3 Binary Item Response Likelihoods and Posterior Update. With the factorized distributions over item utilities and observational

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/657e/657e46d0-f868-4dc4-888a-8571b0426dbd.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: MRR@10 for MonoLLM and PEBOL-P with uncertainty-informed policies (UCB, TS, ER) learning over time and MonoLLM is generally outperformed by PEBOL.
</div>
<div style="text-align: center;">for MonoLLM and PEBOL-P with uncertainty-informed policies (UCB, TS, ER). All methods show preference and MonoLLM is generally outperformed by PEBOL.
</div>
likelihood history now defined, we simply have to provide a concrete observational model of the response likelihood conditioned on the query, item descriptions, and latent utility: 𝑝 (𝑟 𝑡 | x, u,𝑞 𝑡). Because the prior is factorized over conditionally independent 𝑢 𝑖 (cf. (6)), we can likewise introduce individual per-item factorized binary responses 𝑟 𝑡 𝑖 ∈{0 (dislike), 1 (like)} to represent the individual relevance of each item 𝑖 to the preference elicited at turn 𝑡. Critically, we won’t actually require an individual response per item — this will be computed by a natural language inference (NLI) model [6] to be discussed shortly — but we’ll begin with an individual binary response model for 𝑟 𝑡 𝑖 for simplicity:

(10)

With our response likelihood defined, this now leads us to our first pass at a full posterior utility update that we term PEBOL-B for observed Binary rating feedback. Specifically, given observed binary ratings 𝑟 𝑡 𝑖, the update at 𝑡 = 1 uses the Beta prior (7) with the Bernoulli likelihood (10) to form a standard Beta-Bernoulli conjugate pair and compute the posterior utility belief

(11)
(12)

where 𝛼 1 𝑖 = 𝛼 0 𝑖 + 𝑟 1 𝑖, 𝛽 1 𝑖 = 𝛽 0 𝑖 + (1 − 𝑟 1 𝑖). Subsequent incremental updates updates follow Eq. (9) and use the same conjugacy to give

4.1.4 Natural Language Inference and Probabilistic Posterior Update. As hinted above, effective inference becomes slightly more nuanced since we don’t need to observe an explicit binary response per item in our PEBOL framework. Rather, we receive general preference feedback 𝑟 𝑡 on whether a user generically prefers a text description 𝑞 𝑡 and then leverage an NLI model [6] to infer whether the description 𝑥 𝑖 of item 𝑖 would be preferred according to this feedback. For instance, for a (𝑞 𝑡,𝑟 𝑡) pair (“Want to watch a children’s movie?”,“Yes”), NLI should infer a rating of 𝑟 𝑡 1 = 1 for 𝑥 1 = “The Lion King” and 𝑟 𝑡 2 = 0 for 𝑥 2 = “Titanic”. To deal with the fact that NLI models actually return an  entailment probability, our probabilistic observation variant, PEBOL-P leverages the probability that item description 𝑥 𝑖 entails 𝑞 𝑡, which we denote as 𝑤 𝑡 𝑖. We provide a full graphical model and derivation of the Bayesian posterior update given this entailment probability in the Supplementary Material, but note that we can summarize the final result as a relaxed version of the binary posterior update of (13)

that replaces the binary observation 𝑟 𝑖 ∈{0, 1} with the entailment probability 𝑤 𝑡 𝑖 ∈[0, 1], i.e., 𝛼 𝑡 𝑖 = 𝛼 𝑡 − 1 𝑖 + 𝑤 𝑡 𝑖, 𝛽 𝑡 𝑖 = 𝛽 𝑡 − 1 𝑖 + (1 − 𝑤 𝑡 𝑖). To visually illustrate how this posterior inference process works in practice, Figure 1 shows the effect of PEBOL’s posterior utility belief updates based on NLI for three query-response pairs – we can see the system gaining statistical knowledge about useful items for the user from the dialogue.

# 4.2 LLM-Based Acquisition Functions

Recall from Sec. 2.1 that in Bayesian optimization, the posterior informs an acquisition function which determines where to make the next observation. PEBOL generates a new query 𝑞 𝑡 with a two-step acquisition function 𝛾, first using Bayesian Optimization policies (step 1) based on the posterior utility beliefs 𝑝 (u | x, H 𝑡) to select NL context, and then using this selected context to guide LLM prompting (step 2). We express the overall acquisition function 𝛾 = 𝛾 𝐺 ◦ 𝛾 𝐶 as a composition of a context acquisition function 𝛾 𝐶
(cf. Sec. 4.2.1) and a NL generation function 𝛾 𝐺 (cf. Sec. 4.2.2).
4.2.1 Context Acquisition via Bayesian Optimization Policies. First, PEBOL harnesses Bayesian optimization policies to select an item description 𝑥 𝑖 𝑡 which will be used to prompt an LLM to generate a query about an aspect described by 𝑥 𝑖 𝑡 (cf. Sec. 4.2.2). Selecting an item 𝑖 𝑡 whose utility 𝑢 𝑖 𝑡 is expected to be near the maximum, 𝑢 𝑖 ∗, will generate exploitation queries asking about properties of items that are likely to be preferred by the user. In contrast, selecting an item 𝑖 𝑡 associated with high uncertainty in its utility 𝑢 𝑡 𝑖 will generate exploration queries that probe into properties of items for which user preferences are less known. Thus, strategically selecting 𝑥 𝑖 𝑡 allows PEBOL to balance the exploration and exploitation behaviour of NL queries, decreasing the risks of becoming stuck in local optima (over-exploitation) or wasting resources exploring low utility item preferences (over-exploration). We define the item selected by the context acquisition function as

Recall from Sec. 2.1 that in Bayesian optimization, the posterior informs an acquisition function which determines where to make the next observation. PEBOL generates a new query 𝑞 𝑡 with a two-step acquisition function 𝛾, first using Bayesian Optimization policies (step 1) based on the posterior utility beliefs 𝑝 (u | x, H 𝑡) to select NL context, and then using this selected context to guide LLM prompting (step 2). We express the overall acquisition function 𝛾 = 𝛾 𝐺 ◦ 𝛾 𝐶 as a composition of a context acquisition function 𝛾 𝐶
(cf. Sec. 4.2.1) and a NL generation function 𝛾 𝐺 (cf. Sec. 4.2.2).

4.2.1 Context Acquisition via Bayesian Optimization Policies. First, PEBOL harnesses Bayesian optimization policies to select an item description 𝑥 𝑖 𝑡 which will be used to prompt an LLM to generate a query about an aspect described by 𝑥 𝑖 𝑡 (cf. Sec. 4.2.2). Selecting an item 𝑖 𝑡 whose utility 𝑢 𝑖 𝑡 is expected to be near the maximum, 𝑢 𝑖 ∗, will generate exploitation queries asking about properties of items that are likely to be preferred by the user. In contrast, selecting an item 𝑖 𝑡 associated with high uncertainty in its utility 𝑢 𝑡 𝑖 will generate exploration queries that probe into properties of items for which user preferences are less known. Thus, strategically selecting 𝑥 𝑖 𝑡 allows PEBOL to balance the exploration and exploitation behaviour of NL queries, decreasing the risks of becoming stuck in local optima (over-exploitation) or wasting resources exploring low utility item preferences (over-exploration). We define the item selected by the context acquisition function as

(14)

( H)
and list several alternatives for 𝛾 𝐶, including the well-known strategies of TS and UCB [30]:
(1) Thompson Sampling (TS): First, a sample of each item’s utility ˆ 𝑢 𝑡 𝑖 is taken from the posterior, ˆ 𝑢 𝑡 𝑖 ∼ 𝑝 (𝑢 𝑖 | 𝑥 𝑖, H 𝑡). Then, the item with the highest sampled utility is selected:

(15)

TS explores more when beliefs have higher uncertainty and exploits more as the system becomes more confident.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f232/f2322130-5be0-4797-855d-f1b888cd8a47.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: MRR@10 for PEBOL-P with various context acquisition policies.
</div>
(2) Upper Confidence Bound (UCB): Let 𝑃 𝑘 (𝛼, 𝛽) represent the 𝑘 ’th percentile of Beta (𝛼, 𝛽), which provides a confidence bound on the posterior. UCB selects the item with the highest confidence bound

(16)

following a balanced strategy because confidence bounds are increased by both high utility and high uncertainty.
(3) Entropy Reduction (ER): An explore-only strategy that selects the item with the most uncertain utility:

(17)

(4) Greedy: An exploit-only strategy that selects the item with the highest expected utility 𝜇 𝑡 𝑖 (Eq. 5):

(18)

(5) Random: An explore-only heuristic that selects the next item randomly.

4.2.2 Generating Short, Aspect-Based NL Queries. Next, PEBOL prompts an LLM to generate a NL query 𝑞 𝑡 based on the selected item description 𝑥 𝑖 𝑡 while also using the dialogue history H 𝑡 to avoid repetitive queries. We choose to generate “yes-or-no” queries asking if a user prefers items with some aspect 𝑎 𝑡, which is a short text span extracted dynamically from 𝑥 𝑖 𝑡 to be different from any previously queried aspects 𝑎 1, ...,𝑎 𝑡 − 1. We adopt this query generation strategy to: 1) reduce cognitive load on the user, who may be frustrated by long and specific queries about unfamiliar items and 2) better facilitate NLI through brief, general phrases [37]. Letting 𝜙 represent the query generation prompt, we let

(19)

be the LLM generated query and aspect at turn 𝑡, with prompting details discussed in Section 5.2.3. An example of such a query and aspect (bold) is “Are you interested in movies with patriotic themes?”, generated by PEBOL in our movie recommendation experiments and shown in Figure 2.

# .3 NL Item-Preference Entailment

4.3.1 Preference Descriptions from Query Response Pairs. Next, PEBOL receives a NL user response 𝑟 𝑡, which it must convert to individual item preference observations. Since the LLM is instructed to generate "yes-or-no" queries 𝑞 𝑡 asking a user if they like aspect 𝑎 𝑡, we assume the user response will be a"yes" or a"no", and create a NL description of the users preference 𝜌 𝑡, letting 𝜌 𝑡 = 𝑎 𝑡 if 𝑟 𝑡 = “yes”, and 𝜌 𝑡 = concat(“not ”,𝑎 𝑡) if 𝑟 𝑡 = “no”. For example,

given a query that asks if the user prefers the aspect “patriotism” in an item, if the user response is “yes”, then the user preference 𝜌 𝑡 is “patriotism”, and “not patriotism” otherwise. This approach produces short, general preference descriptions that are well suited for NLI models [37].
4.3.2 Inferring Item Ratings from NL Preferences.  Given a NL preference 𝜌 𝑡, PEBOL must infer whether the user would like an item described by 𝑥 𝑖. Specifically, PEBOL acquires ratings w 𝑡 = [𝑤 𝑡 1, ...,𝑤 𝑡 𝑁] (cf. Sec. 4.1.4) by using NLI to predict whether an item description 𝑥 𝑖 entails (i.e., implies) the preference 𝜌 𝑡. For example, we expect that an NLI model would predict that 𝑥 𝑖 = “The Lion King” entails 𝜌 𝑡 = “animated” while 𝑥 𝑗 = “Titanic” does not, inferring that a user who expressed preference 𝜌 𝑡 would like item 𝑖 but not 𝑗. We use an NLI model 𝑃 𝜔 (𝑥 𝑖, 𝜌 𝑡) to predict the probability 𝑤 𝑡 𝑖 that 𝑥 𝑖 entails 𝜌 𝑡, and return 𝑟 𝑡 𝑖 = ⌊ 𝑤 𝑡 𝑖 ⌉ in the case of binary observations (PEBOL-B) and 𝑤 𝑡 𝑖 in the case of probabilistic observations (PEBOL-P).

given a query that asks if the user prefers the aspect “patriotism” in an item, if the user response is “yes”, then the user preference 𝜌 𝑡 is “patriotism”, and “not patriotism” otherwise. This approach produces short, general preference descriptions that are well suited for NLI models [37].

# 4.4 The Complete PEBOL System

This concludes the PEBOL specification – the entire process from prior utility belief to the LLM-based acquisition function generation of a query to the posterior utility update is illustrated in Figure 2.

# 5 EXPERIMENTAL METHODS

We numerically evaluate our PEBOL variations through controlled NL-PE dialogue experiments across multiple datasets and response noise levels – comparing against two monolithic LLM (MonoLLM) baselines. Specifically, these baselines directly use GPT-3.5-turbo0613 (GPT MonoLLM) or Gemini-Pro (Gemini MonoLLM) as the NL-PE system, as described in Section 5.1. We do not compare against ConvRec methods [4, 26, 33, 35] because they are not coldstart systems, requiring observed user-item interactions data to drive their recommendation modules. We also do not base our experiments on ConvRec datasets such as ReDIAL [26], since they are made up of pre-recorded conversation histories and cannot be used to evaluate active, cold-start NL-PE systems.

# 5.1 MonoLLM Baseline

A major challenge of using MonoLLM for NL-PE is that item descriptions x either need to be internalized through training or be provided in the context window (cf. Sec. 4) – since we focus on fully cold-start settings, we test the latter approach as a baseline. In each turn, given the full conversation history H 𝑡 and x, we prompt the MonoLLM to generate a new query to elicit user preferences – all prompts are shown in the Supplementary Materials. We evaluate

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/34d2/34d2203b-2177-4e7f-8585-70805b9fd85c.png" style="width: 50%;"></div>
<div style="text-align: center;">PEBOL using binary vs. probabilistic entailment scores. PEBOL-P with the best policy (TS on Yelp and ecipe-MPR) generally outperforms PEBOL-B.
</div>
<div style="text-align: center;">Figure 6: MRR@10 for PEBOL using binary vs. probabilistic entailment scores. PEBOL-P with the b MovieLens, UCB on Recipe-MPR) generally outperforms PEBOL-B.
</div>
recommendation performance after each turn by using another prompt to recommend a list of ten item names from x given H 𝑡. Due to context window limits, this MonoLLM approach is only feasible for small item sets with short item descriptions; thus, we have to limit |I| to 100 for fair comparison to the MonoLLM baseline.

# 5.2 Simulation Details

We test PEBOL and MonoLLM through NL-PE dialogues with LLMsimulated users, where the simulated users’ item preferences remain hidden from the system. We evaluate recommendation performance over 10 turns of dialogue.

5.2.1 User Simulation. For each experiment, we simulate 100 users, each of which likes a single item 𝑖 ∈I. Each user is simulated by GPT-3.5-turbo-0613, which is given item description 𝑥 𝑖 and instructed to provide only “yes” or “no” responses to a query 𝑞 𝑡 as if it was a user who likes item 𝑖.

5.2.2 Evaluating Recommendations.  We evaluate the top-10 recommendations in each turn using the Mean Reciprocal Rank (MRR@10) of the preferred item, which is equivalent to MAP@10 for the case of a single preferred item.

5.2.3 PEBOL Query Generation. In turn 𝑡, given an item description 𝑥 𝑖 and previously generated aspects (𝑎 1, ...,𝑎 𝑡 − 1), an LLM (GPT-3.5turbo-0613) 5 is prompted to generate an aspect 𝑎 𝑡 describing the item 𝑖 that is no more than 3 words long. The LLM is then prompted again to generate a “yes-or-no” query asking if a user prefers 𝑎 𝑡.

5.2.4 NLI. We use the 400M FAIR mNLI 6 model to predicts logits for entailment, contradiction, and neutral, and divide these logits by an MNLI temperature 𝑇 ∈{1, 10, 100} As per the FAIR guidelines, we pass the temperature-scaled entailment and contradiction scores through a softmax layer and take the entailment probabilities. We report PEBOL results using the best MNLI temperature for the most datasets.

5.2.5 User Response Noise. We test three user response noise levels ∈ {0,0.25,0.5} corresponding to the proportion or user responses that are randomly selected between "yes" and "no".

5 Experiments with newer LLMs such as Gemini or GPT4 for PEBOL query generation are left for future work due to the API time and cost requirements needed to simulate the many variants of PEBOL reported in Section 6. We do, however, compare PEBOL against a Gemini-MonoLLM baseline. 6 https://huggingface.co/facebook/bart-large-mnli

5.2.6 Omitting Query History Ablation. We test how tracking query history in PEBOL effects performance with an ablation study that removes previously generated aspects (𝑎 1, ...,𝑎 𝑡 − 1) from the aspect extraction prompt.

# 5.3 Datasets

We obtain item descriptions from three real-world datasets: MovieLens25M 7, Yelp 8, and Recipe-MPR [38] (example item descriptions from each shown in Table 1 in the Supplementary Materials). After the filtering steps below for Yelp and MovieLens, we randomly sample 100 items to create x. For Yelp, we filter restaurant descriptions to be from a single major North American city (Philadelphia) and to have at least 50 reviews and five or more category labels. For MovieLens, 9 we filter movies to be in the 10% by rating count with at least 20 tags, and let movie descriptions use the title, genre labels, and 20 most common user-assigned tags.

# 5.4 Research Questions

# 6 EXPERIMENTAL RESULTS

# 6.1 RQ1 - PEBOL vs. MonoLLM

Figure 4 shows MRR@10 over 10 dialogue turns for MonoLLM and PEBOL (UCB,TS,ER), 10 with 95% confidence intervals (CIs) at turn 10 shown in Figure 8 (see Supplementary Materials for CIs for all turns and experiments). All methods start near random guessing, reflecting a cold start, and show clear preference learning over time.

7 https://grouplens.org/datasets/movielens/25m/ 8 https://www.yelp.com/dataset 9 For all experiments with MovieLens, we use the 16k version of GPT-3.5-turbo-0613, due to MonoLLM requiring extra context length for x. 10 For each PEBOL policy, we use the MNLI temperature that performed best on the most datasets with continuous responses (see Supplementary Materials).

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1c00/1c00ba01-e8e1-4782-a4d6-c1d9a39487d7.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: The effect of including the generated aspect history in the aspect generation prompt. Incl performance, which we hypothesize is due to reducing repeated or uninformative queries.
</div>
<div style="text-align: center;">ffect of including the generated aspect history in the aspect generation prompt. Including the history improves which we hypothesize is due to reducing repeated or uninformative queries.
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bcf4/bcf46484-7ce2-48e6-863f-83f83a5e38d4.png" style="width: 50%;"></div>
Compared to GPT-MonoLLM, which uses the same LLM (GPT3.5-turbo-0613) as PEBOL does for query generation, after 10 turns of dialogue PEBOL achieves: a mean MRR@10 of 0.27 vs. GPTMonoLLM’s MRR@10 of 0.12 on Yelp; 0.18 vs 0.09 on MovieLens; and 0.17 vs 0.11 on Recipe-MPR, respectively. Compared to GeminiMonoLLM, which uses a newer generation LLM (Gemini-Pro) than PEBOL for query generation, after 10 turns PEBOL still achieves a higher mean MRR@10 of 0.27 vs. Gemini-MonoLLM’s mean MRR@10 of 0.17 on Yelp; 0.18 vs. 0.15 on MovieLens, and 0.17 vs 0.16 on Recipe-MPR, respectively. While we did not have the resources to test PEBOL with Gemini query generation, we hypothesize that using a newer LLM for query generation can further improve PEBOL performance, since using the newer LLM (Gemini) shows performance improvements for the MonoLLM baseline.

# 6.2 RQ2 - Binary vs. Probabilistic Responses

Figure 6 compares PEBOL performance using binary (PEBOL-B) vs. continuous (PEBOL-P) feedback, and shows that performance is typically better when continuous responses are used – indicating that binary feedback models discard valuable information from the entailment probabilities.

# 3 RQ3 - Effect of User Response Noise

Figure 8 shows the impact of user response noise on MRR@10 at turn 10 – PEBOL generally continues to outperform MonoLLM under user response noise. Specifically, at all noise levels, both MonoLLM baselines are outperformed by all PEBOL-P variants on Yelp, and by at least one PEBOL-P variant on MovieLens and Recipe-MPR.

# 6.4 RQ4 - Comparison of Context Acquisition Policies

Figure 5 compares the performance of various PEBOL context acquisition policies – all policies show active preference learning, other than random item selection on RecipeMPR. There is considerable overlap between methods, however for most turns TS does well on Yelp and MovieLens while being beaten by Greedy, ER, and UCB on Recipe-MPR. As expected due to the randomness in sampling, TS performance is correlated with random item selection, while UCB performs quite similarly to greedy.

# 6.5 RQ5 - Effect of Aspect History in Query Generation

# 6.5 RQ5 - Effect of Aspect History in Query Generation

# 6.5 RQ5 - Effect of Aspect History in Query Generation

As shown in Figure 7, we see improvements in PEBOL performance from including a list of previously generated aspects in the aspect generation prompt. For instance, the differences in mean MRR@10 from including vs. excluding the query history for TS after 10 turns were: 0.27 vs 0.16 for Yelp; 0.18 vs 0.14 for MovieLens, and 0.13 vs 0.09 for Recipe-MPR, respectively. Practically, including the aspect generation history also helps to avoid repeat queries, which gain no information and could frustrate a user.

# 7 CONCLUSION AND FUTURE WORK

This paper presents a novel Bayesian optimization formalization of natural language (NL) preference elicitation (PE) over arbitrary NL item descriptions, as well as introducing and evaluating PEBOL, an algorithm for NL P reference E licitation with B ayesian O ptimization augmented L LMs. As discussed below, our study also presents many opportunities for future work, including for addressing some of the limitations of PEBOL and our experiment setup.

User Studies. Firstly, our experiments limited by their reliance on LLM-simulated users. While the dialogue simulations indicate reasonable behaviour in the observed results, such as initial recommendation performance near random guessing, preference learning over time, and coherent user responses in logs such as those shown in Figure 7, future work would benefit from human user studies.
Multi-Item Belief Updates.  While the assumption that item utilities can be updated independently allows the use of a simple and interpertable Beta-Bernouilli update model for each item, it also requires a separate NLI calculation to be performed for each item, which is computationally expensive. A key future direction is thus to explore alternative belief state forms which enable the joint updating of beliefs over all items from a single NLI computation.
Collaborative Belief Updates. Since PEBOL does not leverage any historical interactions with other users, an important future direction is to study NL-PE which leverages collaborative, multiuser data. One possibility is to initialize a cold start user’s prior beliefs based on interaction histories with other users. Another direction is adapting collaborative filtering based belief updating, such as the methods used in item-based feedback PE techniques (e.g., [5]), to NL-PE.
Diverse Query Forms.  While PEBOL uses a pointwise query generation strategy that selects one item description at a time for LLM context, future work can explore LLM-based acquisition functions with pairwise and setwise context selection. Such multi-item context selection would enable contrastive query generation that could better discriminate between item preferences.
NL-PE in ConvRec Architectures. Another direction for future research is the integration of NL-PE methodologies such as PEBOL into conversational recommendation (ConvRec) system architectures (e.g., [7, 9, 17, 20]), which must balance many tasks including recommendation, explanation, and personalized question answering. Thus, in contrast to PEBOL’s pointwise queries and “yes-or-no” user responses, the use of PE in ConvRec systems implies that future algorithms will need to elicit preferences based on arbitrary pairs of NL system-user utterances. In these potential extensions, aspectbased NLI could be enabled by extracting aspects from utterances with LLMs [19].

NL-PE in ConvRec Architectures. Another direction for future research is the integration of NL-PE methodologies such as PEBOL into conversational recommendation (ConvRec) system architectures (e.g., [7, 9, 17, 20]), which must balance many tasks including recommendation, explanation, and personalized question answering. Thus, in contrast to PEBOL’s pointwise queries and “yes-or-no” user responses, the use of PE in ConvRec systems implies that future algorithms will need to elicit preferences based on arbitrary pairs of NL system-user utterances. In these potential extensions, aspectbased NLI could be enabled by extracting aspects from utterances with LLMs [19].

# REFERENCES

[1] Erdem Biyik, Fan Yao, Yinlam Chow, Alex Haig, Chih-wei Hsu, Mohammad Ghavamzadeh, and Craig Boutilier. 2023. Preference Elicitation with Soft Attributes in Interactive Recommendation. ArXiv abs/2311.02085 (2023). https: //api.semanticscholar.org/CorpusID:265034238
[2] Craig Boutilier. 2002. A POMDP formulation of preference elicitation problems. In AAAI/IAAI. Edmonton, AB, 239–246.
[3] Eric Brochu, Tyson Brochu, and Nando De Freitas. 2010. A Bayesian interactive optimization approach to procedural animation design. In Proceedings of the 2010 ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 103–112.
[4] Qibin Chen, Junyang Lin, Yichang Zhang, Ming Ding, Yukuo Cen, Hongxia Yang, and Jie Tang. 2019. Towards Knowledge-Based Recommender Dialog System. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong, China, 1803–1813. https://doi.org/10.18653/v1/D19-1189
[5] Konstantina Christakopoulou, Filip Radlinski, and Katja Hofmann. 2016. Towards Conversational Recommender Systems. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (San Francisco,

[1] Erdem Biyik, Fan Yao, Yinlam Chow, Alex Haig, Chih-wei Hsu, Mohammad Ghavamzadeh, and Craig Boutilier. 2023. Preference Elicitation with Soft Attributes in Interactive Recommendation. ArXiv abs/2311.02085 (2023). https: //api.semanticscholar.org/CorpusID:265034238
[2] Craig Boutilier. 2002. A POMDP formulation of preference elicitation problems. In AAAI/IAAI. Edmonton, AB, 239–246.
[3] Eric Brochu, Tyson Brochu, and Nando De Freitas. 2010. A Bayesian interactive optimization approach to procedural animation design. In Proceedings of the 2010 ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 103–112.
[4] Qibin Chen, Junyang Lin, Yichang Zhang, Ming Ding, Yukuo Cen, Hongxia Yang, and Jie Tang. 2019. Towards Knowledge-Based Recommender Dialog System. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong, China, 1803–1813. https://doi.org/10.18653/v1/D19-1189
[5] Konstantina Christakopoulou, Filip Radlinski, and Katja Hofmann. 2016. Towards Conversational Recommender Systems. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (San Francisco,

California, USA) (KDD ’16). Association for Computing Machinery, New York, NY, USA, 815–824.
[6]  Ido Dagan, Oren Glickman, and Bernardo Magnini. 2005. The PASCAL recognising textual entailment challenge. In Machine Learning Challenges Workshop. Springer, 177–190.
[7] Yashar Deldjoo, Zhankui He, Julian McAuley, Anton Korikov, Scott Sanner, Arnau Ramisa, René Vidal, Maheswaran Sathiamoorthy, Atoosa Kasirzadeh, and Silvia Milano. 2024. A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys). In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’24), August 25–29, 2024, Barcelona, Spain.
[8] Brochu Eric, Nando Freitas, and Abhijeet Ghosh. 2007. Active preference learning with discrete choice data. Advances in Neural Information Processing Systems 20 (2007).
[9] Luke Friedman, Sameer Ahuja, David Allen, Terry Tan, Hakim Sidahmed, Changbo Long, Jun Xie, Gabriel Schubiner, Ajay Patel, Harsh Lara, et al. 2023. Leveraging Large Language Models in Conversational Recommender Systems. arXiv preprint arXiv:2305.07961 (2023).
[10] Roman Garnett. 2023. Bayesian optimization. Cambridge University Press. [11] Javier González, Zhenwen Dai, Andreas Damianou, and Neil D Lawrence. 2017. Preferential bayesian optimization. In  International Conference on Machine Learning. PMLR, 1282–1291.
[12]  Shengbo Guo and Scott Sanner. 2010. Real-time multiattribute Bayesian preference elicitation with pairwise comparison queries. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. JMLR Workshop and Conference Proceedings, 289–296.
[13]  Shengbo Guo, Scott Sanner, and Edwin V Bonilla. 2010. Gaussian process preference elicitation. Advances in Neural Information Processing Systems 23 (2010).
[14] Kunal Handa, Yarin Gal, Ellie Pavlick, Noah Goodman, Jacob Andreas, Alex Tamkin, and Belinda Z. Li. 2024. Bayesian Preference Elicitation with Language Models. arXiv:2403.05534 [cs.CL]
[15] Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bodhisattwa Prasad Majumder, Nathan Kallus, and Julian McAuley. 2023. Large Language Models as Zero-Shot Conversational Recommenders. Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (2023).
[16] Giannis Karamanolakis, Kevin Raji Cherian, Ananth Ravi Narayan, Jie Yuan, Da Tang, and Tony Jebara. 2018. Item recommendation with variational autoencoders and heterogeneous priors. In Proceedings of the 3rd Workshop on Deep Learning for Recommender Systems. 10–14.
[17]  Sara Kemper, Justin Cui, Kai Dicarlantonio, Kathy Lin, Danjie Tang, Anton Korikov, and Scott Sanner. 2024. Retrieval-Augmented Conversational Recommendation with Prompt-based Semi-Structured Natural Language State Tracking. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (Washington, DC, USA) (SIGIR ’24). ACM, New York, NY, USA.
[18] Mohammad M. Khajah, Brett D. Roads, Robert V. Lindsey, Yun-En Liu, and Michael C. Mozer. 2016. Designing Engaging Games Using Bayesian Optimization. In Proceedings of the 2016 CHI Conference on Human Factors in Computing Systems (San Jose, California, USA) (CHI ’16). Association for Computing Machinery, New York, NY, USA, 5571–5582. https://doi.org/10.1145/2858036.2858253
[19] Anton Korikov, George Saad, Ethan Baron, Mustafa Khan, Manav Shah, and Scott Sanner. 2024. Multi-Aspect Reviewed-Item Retrieval via LLM Query Decomposition and Aspect Fusion. In Proceedings of the 1st SIGIR’24 Workshop on Information Retrieval’s Role in RAG Systems, July 18, 2024, Washington D.C., USA.
[20]  Anton Korikov, Scott Sanner, Yashar Deldjoo, Francesco Ricci, Zhankui He, Julian McAuley, Arnau Ramisa, Rene Vidal, Maheswaran Sathiamoorthy, Atoosa Kasirzadeh, and Silvia Milano. 2024. Large Language Model Driven Recommendation. arXiv preprint arXiv:2404.XXXXX (2024).
[21] Hoyeop Lee, Jinbae Im, Seongwon Jang, Hyunsouk Cho, and Sehee Chung. 2019. MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (Anchorage, AK, USA) (KDD ’19). Association for Computing Machinery, New York, NY, USA, 1073–1082. https://doi.org/10.1145/ 3292500.3330859
[22] Wenqiang Lei, Gangyi Zhang, Xiangnan He, Yisong Miao, Xiang Wang, Liang Chen, and Tat-Seng Chua. 2020. Interactive Path Reasoning on Graph for Conversational Recommendation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (Virtual Event, CA, USA) (KDD ’20). Association for Computing Machinery, New York, NY, USA, 2073–2083. https://doi.org/10.1145/3394486.3403258
[23] Belinda Z. Li, Alex Tamkin, Noah Goodman, and Jacob Andreas. 2023. Eliciting Human Preferences with Language Models. arXiv:2310.11589 [cs.CL]
[24]  Lihong Li, Wei Chu, John Langford, and Robert E Schapire. 2010. A contextualbandit approach to personalized news article recommendation. In Proceedings of the 19th International Conference on World Wide Web. 661–670.
[25] Lihong Li, Wei Chu, John Langford, and Xuanhui Wang. 2011. Unbiased offline evaluation of contextual-bandit-based news article recommendation algorithms.

In Proceedings of the Fourth ACM International Conference on Web Search and Data Mining. 297–306.
[26] Raymond Li, Samira Ebrahimi Kahou, Hannes Schulz, Vincent Michalski, Laurent Charlin, and Chris Pal. 2018. Towards deep conversational recommendations. Advances in Neural Information Processing Systems 31 (2018).
[27] Shijun Li, Wenqiang Lei, Qingyun Wu, Xiangnan He, Peng Jiang, and Tat-Seng Chua. 2021. Seamlessly Unifying Attributes and Items: Conversational Recommendation for Cold-start Users. ACM Trans. Inf. Syst. 39, 4, Article 40 (aug 2021), 29 pages. https://doi.org/10.1145/3446427
[28] Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. On Faithfulness and Factuality in Abstractive Summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (Eds.). Association for Computational Linguistics, Online, 1906–1919. https://doi.org/10.18653/v1/2020.aclmain.173
[29] Francesca Rossi and Allesandro Sperduti. 2004. Acquiring both constraint and solution preferences in interactive constraint systems. Constraints 9, 4 (2004), 311–332.
[30]  Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P. Adams, and Nando de Freitas. 2016. Taking the Human Out of the Loop: A Review of Bayesian Optimization. Proc. IEEE 104, 1 (2016), 148–175. https://doi.org/10.1109/JPROC.2015.2494218
[31] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a Large-scale Dataset for Fact Extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), Marilyn Walker, Heng Ji, and Amanda Stent (Eds.). Association for Computational Linguistics, New Orleans, Louisiana, 809–819. https://doi.org/10. 18653/v1/N18-1074
[32]  Ivan Vendrov, Tyler Lu, Qingqing Huang, and Craig Boutilier. 2020. GradientBased Optimization for Bayesian Preference Elicitation. Proceedings of the AAAI Conference on Artificial Intelligence 34, 06 (Apr. 2020), 10292–10301. https: //doi.org/10.1609/aaai.v34i06.6592
[33] Xiaolei Wang, Kun Zhou, Ji-Rong Wen, and Wayne Xin Zhao. 2022. Towards unified conversational recommender systems via knowledge-enhanced prompt learning. In  Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1929–1937.
[34] Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. In  Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers) (New Orleans, Louisiana). Association for Computational Linguistics, 1112–1122. http://aclweb.org/anthology/N18-1101
[35]  Bowen Yang, Cong Han, Yu Li, Lei Zuo, and Zhou Yu. 2022. Improving Conversational Recommendation Systems’ Quality with Context-Aware Item MetaInformation. In Findings of the Association for Computational Linguistics: NAACL 2022. 38–48.
[36] Hojin Yang, Scott Sanner, Ga Wu, and Jin Peng Zhou. 2021. Bayesian Preference Elicitation with Keyphrase-Item Coembeddings for Interactive Recommendation. In Proceedings of the 29th ACM Conference on User Modeling, Adaptation and Personalization. 55–64.
[37] Wenpeng Yin, Jamaal Hay, and Dan Roth. 2019. Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach. In  Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong, China, 3914–3923. https://doi.org/10.18653/v1/D19-1404
[38]  Haochen Zhang, Anton Korikov, Parsa Farinneya, Mohammad Mahdi Abdollah Pour, Manasa Bharadwaj, Ali Pesaranghader, Xi Yu Huang, Yi Xin Lok, Zhaoqi Wang, Nathan Jones, and Scott Sanner. 2023. Recipe-MPR: A Test Collection for Evaluating Multi-aspect Preference-based Natural Language Retrieval. In  Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (Taipei, Taiwan) (SIGIR ’23). Association for Computing Machinery, New York, NY, USA, 2744–2753. https: //doi.org/10.1145/3539618.3591880
[39] Xiaoying Zhang, Hong Xie, Hang Li, and John C.S. Lui. 2020. Conversational Contextual Bandit: Algorithm and Application. In  Proceedings of The Web Conference 2020 (Taipei, Taiwan) (WWW ’20). Association for Computing Machinery, New York, NY, USA, 662–672. https://doi.org/10.1145/3366423.3380148
[40] Xiaoxue Zhao, Weinan Zhang, and Jun Wang. 2013. Interactive collaborative filtering. In Proceedings of the 22nd ACM International Conference on Information & Knowledge Management. 1411–1420.

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f554/f554bce9-6b77-4ad1-8700-d0ba95a698e3.png" style="width: 50%;"></div>
Figure 9: Graphical model used for posterior utility updates. 𝑢 𝑖 is the random variable representing the utility of item 𝑖 ∈I and 𝑥 𝑖 is the item description. 𝑞 𝑡 is the query presented at iteration 𝑡, 𝑟 𝑡 𝑖 is the variable representing the latent (not directly observed) relevance of item 𝑖 at step 𝑡, and 𝑒 𝑡 𝑖 is the binary observation representing whether the item description entails the user preference. Unshaded and shaded nodes indicate unobserved and observed variables respectively.

# PROBABILISTIC GRAPHICAL MODEL FOR

# A PROBABILISTIC GRAPHICAL MODEL FOR POSTERIOR UTILITY UPDATE

In this section, we present the probabilistic graphical model for the posterior utility updates introduced in our paper with a more detailed derivation of the posterior utility belief. As discussed in the paper, the objective of posterior inference is to update the prior belief maintained over the utility of each item 𝑖 ∈I denoted by 𝑝 (𝑢 𝑖) given the query 𝑞 𝑡, item description 𝑥 𝑖, and 𝑒 𝑡 𝑖, the binary observation variable representing whether the item description entails the user’s response to the queried preference 𝑞 𝑡. Figure 9 shows the graphical model representation of these variables. The presented query 𝑞 𝑡 and the item description 𝑥 𝑖 are observed (shaded), while the relevance of item 𝑖 to query 𝑞 𝑡 is latent (unshaded) and denoted by 𝑟 𝑡 𝑖 ∈{0, 1}  and conditioned on the latent item utility 𝑢 𝑖. We observe whether the item 𝑥 𝑖 “truly” entails (i.e., 𝑒 𝑡 𝑖 = True) the user’s response to query 𝑞 𝑡 (as determined by the NLI entailment probability 𝑤 𝑡 𝑖 if the item is relevant, i.e., 𝑟 𝑡 𝑖 = 1). The conditional probability distributions in this graphical model are formally defined as follows:

(22)

To further explain the rationale for Eq (22), we note that 𝑤 𝑡 𝑖 is the natural language entailment probability that item description 𝑥 𝑖 entails the aspect queried in the user’s response to 𝑞 𝑡 given that item 𝑖 is relevant (𝑟 𝑡 𝑖 = 1). This entailment probability is obtained from the NLI model, which produces the probability that the entailment is true, hence the reason why 𝑒 𝑡 𝑖 = True. If item 𝑖 is instead irrelevant

(23)

Considering the conditional independencies determined from the graphical model, the joint distribution factorizes as

(24)

(|)()(|)(|)
Next, we replace the probability distribution of each factor according to Equations (20), (21), (22) in (23), to obtain

(25)

# Expanding the summation yields

∝ 𝑤 𝑡 𝑖 Beta (𝑢 𝑖; 𝛼 + 1, 𝛽) + (1 − 𝑤 𝑡 𝑖) Beta (𝑢 𝑖; 𝛼, 𝛽 + 1)

The latter term represents a mixture of Beta distributions that is challenging to handle since multiple posterior updates would cause the number of components in the mixture to grow exponentially with the number of query observations 𝑚, leading to substantial computational and memory complexity. To address this issue, several methods have been proposed for approximating the posterior distribution to allow for tractable computations. In this work, we use the Assumed Density Filtering (ADF) approach, a technique widely used in Bayesian filtering and tracking problems to project a complex posterior to an assumed simpler form (often the same form as the prior to maintain a closed-form). In our case, we project the Beta mixture posterior to a single Beta in order to maintain a closed-form Beta approximation of the posterior update matching the form of the Beta prior in Eq (20). To apply ADF, we assume a Beta distribution with parameters 𝛼 ′ and 𝛽 ′ for the posterior, and approximate the original mixture of Beta’s with this distribution by equating their first moments (i.e., their means):

(28)

Equating the numerators yields
� +

(29)

(30)

Thus, the “mean matched” posterior is Beta (𝑢 𝑖; 𝛼 + 𝑤 𝑡 𝑖, 1 + 𝛽 − 𝑤 𝑡 𝑖).

11 We note that an alternative approach (not used here) could attempt to use NLI to determine the probability of an incorrect true entailment (or confusion) given that item 𝑖 is irrelevant (𝑟 𝑡 𝑖 = 0). That is, there is no inherent requirement for the two cases of Eq (22) to sum to 1 since 𝑟 𝑡 𝑖 is on the conditional side.

Matching two distributions by equating their first moments is a special case of a more general technique called  “moment matching”, which is widely used to approximate a complex probability distribution with a simpler one by equating their moments. In our work, we adopted a special case of this approach by matching the first moments, which we refer to as “mean matching”  of the distributions that we used for its simplicity and intuitive interpretation. However, this is only one of the possible solutions, and a complete moment matching derivation results in a slightly different solution. With this “mean matching” derivation and current item 𝑖 posterior 𝐵𝑒𝑡𝑎 (𝑢 𝑖; 𝛼, 𝛽) at step 𝑡 − 1, we can now perform an incremental posterior update after at step 𝑡 given the probability 𝑤 𝑡 𝑖 that the item description 𝑥 𝑖 entails preference query 𝑞 𝑡 yielding the closed-form Beta posterior 𝐵𝑒𝑡𝑎 (𝑢 𝑖; 𝛼 + 𝑤 𝑡 𝑖, 1 + 𝛽 − 𝑤 𝑡 𝑖) as used in PEBOL-P.

<div style="text-align: center;">Table 1: Examples of LLM Aspect-Based Query Generation from an Item Description
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/21c7/21c7a973-49e9-45c3-bab2-ea0566d4ceb1.png" style="width: 50%;"></div>
Dataset
Item Description 𝑑𝑖
Generated Aspect𝑎𝑡
Generated Query 𝑞𝑡
MovieLens
Movie Title: Meet John Doe (1941)
Genres: Comedy, Drama
Tags: Christianity, Frank Capra, acting, anti-fascism, class
issues, journalism, patriotic, pro american, thought pro-
voking, AFI 100 (Cheers), BD-R, Barbara Stanwyck, Gary
Cooper, baseball player, compare: This Is Our Land (2017),
domain, funny, radio broadcast, reviewed in the NYer by
Anthony Lanne (2018-04-30), suicide note
patriotism
Are you interested in movies with
patriotic themes?
classic
Do you enjoy classic movies?
Recipe-MPR
Spaghetti with mushrooms, onion, green pepper, chicken
breasts, and alfredo sauce
alfredo sauce
Do you like alfredo sauce?
chicken breast
Do you like chicken breasts?
Yelp
name: Le Pain Quotidien
categories: Restaurants, Bakeries, Break-
fast & Brunch, Coffee & Tea, Food, Bel-
gian, French
bakery
Do you like bakeries?
French pastries
Do you like French pastries?
<div style="text-align: center;">Figure 10: MAP@10 for all turns on all datasets and noise levels
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7aba/7aba5e09-33e1-4b4d-af64-64e114e3fa35.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 11: The effect of MNLI temperature on MAP@10 without noise – error bars are 95% C.I.s. Other using the temperatures that perform best across the most datasets for each policy.
</div>
<div style="text-align: center;">ct of MNLI temperature on MAP@10 without noise – error bars are 95% C.I.s. Other results are reported ures that perform best across the most datasets for each policy.
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3a16/3a1646a1-4875-45f0-8d44-b67e6f9df67e.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 12: Effect of MNLI Temperature with noise 0.25
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d855/d85515ae-f5ec-4bf0-8886-8dc7de2b8318.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 13: Effect of MNLI Temperature with noise 0.5
</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/01eb/01ebb22f-aac3-4221-a2dd-1b672568b2f1.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 14: LLM Prompting Templates
</div>
<div style="text-align: center;">Table 2: MRR@10 Mean and 95% C.I. for PEBOL-P vs MonoLLM on MovieLens without Respo
</div>
Turn
1
2
3
4
5
6
7
8
9
10
Method
MonoLLM GPT Mean
0.04
0.06
0.06
0.07
0.08
0.08
0.10
0.11
0.09
0.09
MonoLLM GPT CI LB
0.01
0.02
0.02
0.02
0.03
0.03
0.05
0.06
0.04
0.04
MonoLLM GPT CI UB
0.08
0.10
0.11
0.11
0.12
0.13
0.15
0.16
0.14
0.15
MonoLLM Gemini Mean
0.04
0.02
0.06
0.06
0.07
0.08
0.11
0.10
0.10
0.15
MonoLLM Gemini CI LB
0.01
0.01
0.02
0.03
0.04
0.05
0.07
0.05
0.06
0.09
MonoLLM Gemini CI UB
0.07
0.04
0.10
0.10
0.11
0.12
0.16
0.15
0.14
0.20
ER Mean
0.05
0.05
0.06
0.08
0.09
0.10
0.12
0.12
0.13
0.14
ER CI LB
0.01
0.02
0.03
0.04
0.05
0.05
0.06
0.07
0.07
0.08
ER CI UB
0.08
0.09
0.09
0.12
0.13
0.15
0.17
0.18
0.19
0.20
Greedy Mean
0.05
0.05
0.07
0.07
0.09
0.09
0.10
0.10
0.11
0.13
Greedy CI LB
0.02
0.02
0.03
0.03
0.05
0.04
0.05
0.05
0.06
0.07
Greedy CI UB
0.09
0.09
0.10
0.11
0.14
0.14
0.15
0.15
0.17
0.19
Random Mean
0.05
0.05
0.08
0.11
0.12
0.14
0.14
0.16
0.16
0.14
Random CI LB
0.01
0.02
0.04
0.05
0.06
0.08
0.08
0.09
0.09
0.09
Random CI UB
0.08
0.09
0.13
0.16
0.18
0.21
0.20
0.22
0.22
0.20
TS Mean
0.05
0.08
0.11
0.10
0.12
0.15
0.15
0.16
0.17
0.18
TS CI LB
0.01
0.03
0.06
0.05
0.06
0.09
0.09
0.10
0.10
0.11
TS CI UB
0.08
0.13
0.17
0.15
0.17
0.22
0.21
0.23
0.23
0.24
UCB Mean
0.05
0.05
0.06
0.07
0.09
0.09
0.10
0.10
0.12
0.13
UCB CI LB
0.02
0.02
0.03
0.03
0.05
0.04
0.05
0.05
0.06
0.07
UCB CI UB
0.09
0.09
0.10
0.10
0.13
0.14
0.15
0.15
0.17
0.19
Turn
1
2
3
4
5
6
7
8
9
10
Method
MonoLLM GPT Mean
0.05
0.09
0.07
0.09
0.05
0.06
0.07
0.08
0.09
0.11
MonoLLM GPT CI LB
0.01
0.05
0.03
0.05
0.02
0.02
0.03
0.04
0.04
0.06
MonoLLM GPT CI UB
0.08
0.13
0.10
0.14
0.08
0.10
0.11
0.13
0.13
0.17
MonoLLM Gemini Mean
0.02
0.03
0.04
0.07
0.07
0.11
0.11
0.13
0.10
0.16
MonoLLM Gemini CI LB
0.01
0.02
0.02
0.04
0.03
0.06
0.07
0.08
0.06
0.10
MonoLLM Gemini CI UB
0.03
0.05
0.07
0.10
0.10
0.15
0.15
0.18
0.13
0.21
ER Mean
0.06
0.05
0.08
0.11
0.11
0.13
0.14
0.15
0.16
0.16
ER CI LB
0.02
0.02
0.04
0.06
0.06
0.08
0.08
0.09
0.09
0.10
ER CI UB
0.10
0.08
0.13
0.16
0.16
0.18
0.19
0.22
0.22
0.22
Greedy Mean
0.06
0.06
0.07
0.10
0.11
0.13
0.15
0.15
0.16
0.17
Greedy CI LB
0.02
0.02
0.03
0.05
0.06
0.07
0.09
0.09
0.10
0.11
Greedy CI UB
0.10
0.09
0.11
0.15
0.17
0.18
0.21
0.21
0.23
0.24
Random Mean
0.05
0.03
0.03
0.03
0.04
0.04
0.05
0.06
0.07
0.07
Random CI LB
0.02
0.01
0.01
0.00
0.01
0.01
0.02
0.02
0.03
0.02
Random CI UB
0.08
0.06
0.06
0.05
0.06
0.07
0.08
0.10
0.11
0.11
TS Mean
0.06
0.06
0.06
0.08
0.09
0.10
0.12
0.13
0.13
0.13
TS CI LB
0.02
0.02
0.02
0.03
0.05
0.05
0.06
0.07
0.07
0.07
TS CI UB
0.10
0.09
0.10
0.12
0.14
0.14
0.17
0.19
0.18
0.19
UCB Mean
0.06
0.06
0.08
0.11
0.13
0.14
0.16
0.16
0.17
0.17
UCB CI LB
0.02
0.02
0.03
0.06
0.07
0.08
0.10
0.09
0.11
0.11
UCB CI UB
0.10
0.09
0.12
0.16
0.18
0.19
0.23
0.22
0.24
0.24
<div style="text-align: center;">Table 4: MRR@10 Mean and 95% C.I. for PEBOL-P vs MonoLLM on Yelp without Response Nois
</div>
Turn
1
2
3
4
5
6
7
8
9
10
Method
MonoLLM GPT Mean
0.03
0.05
0.04
0.05
0.07
0.07
0.07
0.09
0.09
0.12
MonoLLM GPT CI LB
0.01
0.01
0.01
0.02
0.03
0.03
0.03
0.04
0.04
0.06
MonoLLM GPT CI UB
0.05
0.08
0.07
0.08
0.10
0.12
0.12
0.14
0.14
0.17
MonoLLM Gemini Mean
0.06
0.09
0.11
0.09
0.10
0.14
0.13
0.16
0.20
0.17
MonoLLM Gemini CI LB
0.02
0.05
0.06
0.05
0.06
0.08
0.09
0.10
0.14
0.12
MonoLLM Gemini CI UB
0.10
0.14
0.15
0.13
0.14
0.20
0.18
0.21
0.26
0.23
ER Mean
0.06
0.