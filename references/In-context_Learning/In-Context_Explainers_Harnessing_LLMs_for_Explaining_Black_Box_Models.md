# In-Context Explainers: Harnessing LLMs for Explaining Black Box Models
Nicholas Kroeger1*, Dan Ley2*, Satyapriya Krishna2, Chirag Agarwal2, and Himabin Lakkaraju2
# Nicholas Kroeger1*, Dan Ley2*, Satyapriya Krishna2, Chirag Agarwal2, and Himabindu Lakkaraju2
1 University of Florida 2 Harvard University
*Equal contribution. Corresponding author: Nick Kroeger (nkroeger@ufl.edu)
Abstract. Recent advancements in Large Language Models (LLMs) have demonstrated exceptional capabilities in complex tasks like machine translation, commonsense reasoning, and language understanding. One of the primary reasons for the adaptability of LLMs in such diverse tasks is their in-context learning (ICL) capability, which allows them to perform well on new tasks by simply using a few task samples in the prompt. Despite their effectiveness in enhancing the performance of LLMs on diverse language and tabular tasks, these methods have not been thoroughly explored for their potential to generate post hoc explanations. In this work, we carry out one of the first explorations to analyze the effectiveness of LLMs in explaining other complex predictive models using ICL. To this end, we propose a novel framework, In-Context Explainers, comprising of three novel approaches that exploit the ICL capabilities of LLMs to explain the predictions made by other predictive models. We conduct extensive analysis with these approaches on real-world tabular and text datasets and demonstrate that LLMs are capable of explaining other predictive models similar to state-of-the-art post hoc explainers, opening up promising avenues for future research into LLM-based post hoc explanations of complex predictive models.
arXiv:2310.05797v4
# 1 Introduction
Large Language Models (LLMs) have become ubiquitous across various industries and are being increasingly employed for a large range of applications, including language understanding [1], genomics [2], tabular medical data records [3, 4], and drug discovery [5]. While LLMs attain high performance and generalization capabilities for numerous tasks [6], they have also increased their parameter sizes and the computational costs for additional fine-tuning on new downstream tasks. To alleviate this, recent works have shown that LLMs can learn new tasks using in-context learning (ICL), which allows them to perform well on new tasks by simply using a few task samples in the prompt [7]. In-context learning allows language models to dynamically understand, adapt, and generate responses based on the immediate context provided in the prompt, eliminating the need for expensive retraining or fine-tuning. This capability of LLMs enables them to adapt to diverse applications. By generalizing from patterns within the input, ICL supports problem solving and complex reasoning, offering a powerful tool for LLMs [8]. Despite its effectiveness in enhancing the performance of LLMs, the ability of in-context learning for data interpretation and analysis remains underexplored. In particular, there is very little work on systematically analyzing the potential of using in-context learning to explain the behavior of other complex predictive models. Thus, the potential of LLMs to serve as reliable explainers and enhance the understanding of predictive models remains an open question. Present work. In this work, we investigate whether in-context learning can help LLMs explain the behavior of other complex predictive models? (see Fig. 1). To answer this question, we propose a novel framework called In-Context Explainers. This framework comprises two broad in-context learning
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/07ea/07ea9cf0-3f00-4586-acf9-e030b23a2496.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 1: Overview of the in-context explanation generation and evaluation process. Given a dataset and a model to explain, we introduce novel ICL strategies to generate explanations of model predictions using LLMs. The resulting LLM-based explanations are then parsed, and their faithfulness is evaluated using diverse metrics.</div>
strategies: perturbation-based and explanation-based ICL, which provide the foundation for exploring the explanation-generating capabilities of state-of-the-art LLMs. Perturbation-based ICL builds on the ideas from explainable AI literature which involve constructing local linear approximations [9] to explain the predictions made by a given ML model. To this end, we provide instance perturbations and their corresponding model outputs in the prompt to the LLM and ask the LLM to explain (i.e., output the top features driving) these predictions. We consider a couple of variants under this strategy, namely, Perturb ICL (P-ICL) and Perturb+Guide ICL (PG-ICL). For P-ICL, we prompt the LLM to use its chain-of-thought reasoning. For PG-ICL, we add detailed guidelines in the prompt for generating explanations. In our Explain ICL strategy, we prompt the LLM with a small, random selection of model input-output pairs and their respective explanations (generated using a state-of-the-art post hoc explanation method) to generate explanations for new samples. We analyze the effectiveness of LLMs in explaining other predictive models on eight real-world datasets, four black box models, and three Gpt models. Our extensive investigation reveals the following key findings: 1 ICL enables LLMs to generate faithful explanations that are on par with several state-of-the-art post hoc explanation methods (despite some of these methods having access to the underlying black box model); 2 our proposed Explain ICL prompting strategy allows LLMs to mimic the behavior of six state-of-the-art explanation methods; 3 On average across five tabular datasets, we find that providing the LLM with detailed guidelines shows more consistent and higher faithfulness scores than simply providing generic instructions; 4 Our results with three text datasets show that ICL capabilities aid LLMs to identify most important words in sentiment classification tasks. Finally, our exploration highlights LLMs’ effectiveness as post hoc explainers, paving the way for future research on LLM-generated explanations.
# 2 Related Work
Our work lies at the intersection of post hoc explanations and LLMs, which we discuss below. Post Hoc Explanations. The task of understanding model predictions has become increasingly intricate with the growing popularity of complex ML models [10] due to their inherent black box nature, which makes it difficult to interpret their internal reasoning. To this end, a plethora of feature attribution methods (commonly referred to as post hoc explanation methods) have been proposed to provide explanations for these models’ predictions. These explanations are predominantly presented in the form of feature attributions, which highlight the importance of each input feature on the model’s prediction. Broadly, post hoc explainers can be divided into perturbation-based and gradient-based methods. While perturbation-based methods [9, 11, 12] leverage perturbations of the given instance to construct an interpretable approximation of the black box model behavior, gradient-based methods [13, 14] leverage gradients w.r.t. the given instance to explain model predictions. In this work, we primarily focus on state-of-the-art local post hoc explainers, i.e., methods explaining individual feature importance for model predictions of individual instances. Large Language Models. LLMs have seen exponential growth in recent years, both in terms of their size and the complexity of tasks they can perform [15]. Recent advances in LLMs are changing
# Using the Perturbed Sample x′ Input: A: 0.430, B: -0.001, C: -0.196, D: -0.139, E: 0.050, F: 0.987 Output: 0 # Using the Raw Perturbation 𝛿 Change in Input: A: 0.316, B: -0.112, C: -0.200, D: -0.139, E: 0.050, F: -0.013 Change in Output: 1
Fig. 2: Sample serialization template for the Recidivism dataset with six features.
the paradigm of NLP research and have led to their widespread use across applications spanning machine translation [16], question-answering [1], text generation [15], and medical data records [3, 4]. In this work, we, for the first time, explore their use in explaining other predictive models.
# 3 Our Framework: In-Context Explainers
Here, we describe our proposed prompting strategies to generate natural language explanations from LLMs that can explain the behavior of predictive models. We will first discuss the notation used to describe the prompting templates and proceed to detail the prompts in Secs. 3.1-3.2. Notation. Let 𝑓: R𝑑→[0, 1] denote a black box ML model trained on a tabular data that takes an input x ∈R𝑑and returns the probability of x belonging to a class 𝑐∈C and the predicted label 𝑦. Following previous works in XAI [9, 13], we randomly sample points from the local neighborhood N𝑥of the given input x to generate explanations, where N𝑥= N (0, 𝜎2) denotes the neighborhood of perturbations around x using a Normal distribution with mean 0 and variance 𝜎2.
# 3.1 Perturb ICL
Existing post hoc explainers, such as those relying on a large number of neighborhood samples [9, 13], often encounter computational bottlenecks. In contrast, we explore the utility of our proposed perturbation-based ICL strategy that leverages the power of LLMs in explaining the behavior of predictive models efficiently. In addition, unlike post hoc explainers like LIME [9], SmoothGrad [13], and SHAP [11], LLMs have the potential to produce natural language explanations that are more plausible and coherent to human practitioners. We aim to explore this by utilizing LLMs to articulate the top-𝑘most important features in determining the output of a given model 𝑓in a rank-ordered manner. In particular, we sample input-output pairs from the neighborhood N𝑥of x and generate their respective strings following a serialization template. For instance, let a neighborhood sample for x = [0.114, 0.111, 0.004, 0] be x′ = [0.430, −0.001, −0.196, −0.139], where x′ = x + 𝛿and 𝛿= [0.316, −0.112, −0.200, −0.139], and assume the model predictions for x and x′ are 1 and 0, respectively. Considering a binary classification problem, the corresponding change in the model predictions then belongs to {−1, 0, 1}. Next, we provide examples of transforming the perturbed sample (x′) and the perturbation (𝛿) into a natural-language string for our prompt templates. For brevity, in the following sections, we will use the perturbed sample in our template for describing our prompting strategies but explore both in our experiments. Motivated by the local neighborhood approximation works in XAI, the Perturb ICL prompting strategy presumes that the local behavior of model 𝑓is a simple linear decision boundary, contrasting with the often globally exhibited complex non-linear decision boundary. Hence, assuming a sufficient number of perturbations in N𝑥, the LLM is expected to accurately approximate the black box model’s behavior and utilize this information to identify the top-𝑘most important features. In addition, to alleviate the computational problems of post hoc explainers, we explore using a small number of 𝑛ICL samples (16 in our experiments) from N𝑥in our prompting templates for LLMs to generate
Fig. 3: A sample prompt generated using our proposed Perturb ICL (P-ICL) prompting strategy.
explanations. For samples in N𝑥, we select those with the highest predictive confidence by the underlying ML model, helping the LLM produce explanations centered on model certainty. In our study, we explore two paradigms of generating the explanations: i) Perturb ICL ( PICL), and ii) Perturb+Guide ICL ( PG-ICL), which we will describe next using the local neighborhood perturbations discussed above. Both of our strategies use four distinct steps in their template: Context, Dataset, Question, and Instructions. In Context, we provide a general background of the underlying ML model, the number of features in the dataset, the number of classes, and the model predictions. In Dataset, we leverage the success of ICL and provide a list of inputs and their respective model outputs using randomly sampled points from the local neighborhood of a given test input. In Question, we specify the task we want the underlying LLM to perform. Finally, in the Instructions, we enumerate the guidelines we want the LLM to follow while generating the output explanations. For P-ICL, we only prompt the LLM to leverage its chain-of-thought (CoT) reasoning abilities using “Think about the question” before generating the response, but for PG-ICL, we prompt the model with a detailed guideline of how to think about the given problem. P-ICL. Given the test sample x to be explained, we combine the Context of the predictive model, the Dataset of 𝑛ICL input-output pairs from N𝑥, the Question, and the general Instructions in our prompt to explore the effectiveness of the LLM in generating explanations. In Figure 3, we provide a sample prompting template for the P-ICL strategy. PG-ICL. The PG-ICL prompting strategy transitions from specifying general instructions in the prompt to providing detailed guidance on the strategy for task execution. Rather than solely instructing the LLM to “Think about the question” of what the task entails, this strategy delineates how to conduct the given task. The objective remains to explore the effectiveness of LLMs in explaining the behavior of other predictive models by identifying the top-𝑘most important features. However, with step-by-step guidelines, we aim to induce a more structured and consistent analytical process within the LLM to generate more faithful explanations. We follow a similar prompting template as in Method 1, including the four components, viz., Context, Dataset, Question, and Instructions, but we modify the Instructions component. The PG-ICL prompt template is provided in Figure 4.
# PG-ICL prompt template Context: “We have . . . outputs.” Dataset: Input: . . .
Question: “Based on . . . output?” Instructions: “For each feature, starting with ‘A’ and continuing to ‘F’: 1. Analyze the feature in question. Rate the importance of the feature in determining the output on a scale of 0-100, considering both positive and negative correlations. Ensure to give equal emphasis to both positive and negative correlations and avoid focusing only on absolute values. 2. After analyzing the feature, position it in a running rank compared to the features already analyzed. For instance, after analyzing feature ‘B’, determine its relative importance compared to ‘A’ and position it accordingly in the rank (e.g., BA or AB). Continue this process until all features from ‘A’ to ‘F’ are ranked. After explaining your reasoning, provide your answer as the final rank of features from ‘A’ to ‘F’ from most important to least important, in descending order, separated by commas. Only provide the feature names on the last line. Do not provide any further details on the last line.” # LLM Response: To determine the most important features, we need to . . . . . .
# LLM Response: To determine the most important features, we need to . . . . . . B, A, C, F, D, E
Fig. 4: A sample prompt generated using the Perturb+Guide ICL (PG-ICL) prompting strategy. Note that the Context, Dataset, and Question are the same as in Fig. 3.
Here, we provide some detailed guidelines to the LLM for understanding the notion of important features and how to analyze them through the lens of correlation analysis. To achieve this, we instruct LLMs to study each feature sequentially and ensure that positive and negative correlations are equally emphasized. The LLM assigns an importance score to each feature in the given dataset and then positions it in a running rank. This rank encourages the LLM to differentiate features and avoid ties in its evaluation of feature importance. The final line in the template ensures that the LLM’s responses are strictly analytical, minimizing non-responsiveness or digressions.
# 3.2 Explain ICL
Recent studies show that LLMs can learn new tasks through ICL, enabling them to excel in new downstream tasks by merely observing a few instances of the task in the prompt. Unlike Perturb ICL (Sec. 3.1), which samples input-output pairs from the neighborhood N𝑥of a test sample x for ICL prompts, the Explain ICL (E-ICL) strategy uses random input-output-explanation triplets generated from a held-out dataset for prompting. Here, the explanations used in the ICL are generated by any post hoc explanation method as a ground truth. Intuitively, we explore whether an LLM can learn the behavior of a given post hoc explanation method by looking at some of its generated explanations. For constructing the ICL set, we randomly select 𝑛ICL input instances XICL from the ICL split of the dataset and generate their predicted labels yICL using the given predictive model 𝑓. Next, we generate explanations EICL for samples (XICL, yICL) using any post hoc explainer. Using the above input-output-explanation triplets, we construct a prompt by concatenating them to form the prompt in Figure 5. The E-ICL prompting strategy explores how LLMs can produce faithful explanations by analyzing the 𝑛ICL input-output-explanations generated by state-of-the-art post hoc explainer.
# Explain ICL Prompt Template Input: A: 0.172, B: 0.000, C: 0.000, D: 1.000, E: 0.000, F: 0.000 Output: 1 Explanation: A,C,B,F,D,E . . . Input: A: 0.052, B: 0.053, C: 0.073, D: 0.000, E: 0.000, F: 1.000 Output: 0 Explanation: A,B,C,E,F,D Input: A: 0.180, B: 0.222, C: 0.002, D: 0.000, E: 0.000, F: 1.000 Output: 0 Explanation:
# 4 Experimental Evaluation
Next, we evaluate the effectiveness of LLMs as post hoc explainers, focusing on three key questions: Q1) Can LLMs generate post hoc explanations for predictive models trained on tabular datasets? Q2) How well can LLMs generate explanations for sentiment classifiers? Q3) What impact do variations in the LLM’s prompting strategy have on the faithfulness of explanations?
# 4.1 Datasets and Experimental Setup
We first describe the datasets and models used to study the reliability of LLMs as post hoc explainers and then outline the experimental setup. Datasets. For tabular datasets, we follow previous LLM works [17] and perform analysis on five realworld tabular datasets: Blood [18], Recidivism [19], Adult [20], Credit [21], and HELOC [22]. For text datasets, we use three sentiment datasets: Amazon reviews, IMDb, and Yelp [23]. The datasets come with a random train-test split, and we further divide the train set, allocating 80% for training and the remaining 20% for ICL sample selection, as detailed in Sec. 3.2. See Appendix A.1 for more details. Predictive Models. For the tabular dataset classifiers, we consider three ML models with varying non-linearity in our experiments: Logistic Regression (LR), a three-layer Artificial Neural Network (ANN), and a six-layer ANN. For the sentiment classifier, we train a single layer transformer model for the text datasets to show the generalization capability of our analysis in generating explanations for diverse predictive models. We use PyTorch [24] to implement the LR, ANN, and transformer model. Please refer to Appendix A.1 for architecture details and Tables 2 and 3 for predictive performances of these models. Large Language Models. We consider Gpt-3.5, Gpt-4, and Gpt-4-0125-preview as LLMs for all experiments. All experiments default to using Gpt-4, unless stated otherwise. Baseline Methods. For the tabular dataset classifiers, we use six post hoc explainers as baselines to investigate the effectiveness of explanations generated using LLMs: LIME [9], SHAP [11], Vanilla Gradients [12], SmoothGrad [13], Integrated Gradients [14], and Gradient x Input (ITG) [25]. For text classifiers, we cannot directly retrieve gradient-based attributions at the input layer since the model uses an embedding layer with discrete vocabulary inputs. Hence, we compute attributions at the embedding layer of each token, where the core idea here is to use a layer-wise attribution method to compute how much each dimension of the embedding vector contributes to the final prediction. Finally, we sum and normalize attributions across all dimensions of each token’s embedding vector, resulting in a single scalar value per token. The six gradient-based post hoc explainers for the text classifiers are Layer Integrated Gradients (LIG), Layer Gradient SHAP (LGS), Layer Deep
Lift (LDL), Layer Gradient x Activation (LGxA), Layer Activation (LA), and Layer Conductance (LC) [26] from the Captum python library [27]. Performance Metrics. We employ four distinct metrics to measure the faithfulness of an explanation. We use the Feature Agreement (FA) and Rank Agreement (RA) metrics introduced in [28] that compares the LLM’s top-𝑘directly with the LR model’s coefficient (see Appendix A.1 for details). The FA and RA metrics range from [0, 1], where 0 means no agreement and 1 means full agreement. In the absence of a top-𝑘model coefficient (as is the case with ANNs), we use the Prediction Gap on Important feature perturbation (PGI) and the Prediction Gap on Unimportant feature perturbation (PGU) metrics from OpenXAI [29]. While PGI measures the change in prediction probability that results from perturbing the features deemed as influential, PGU examines the impact of perturbing unimportant features. Here, the perturbations are generated using Gaussian noise N (0, 𝜎2). For text datasets, we adapt the PGI and PGU metrics to evaluate the faithfulness of explanations by measuring the impact of perturbing the top-𝑘important and unimportant words respectively, as identified by the explanation model. The perturbations are implemented by masking the specified words rather than using Gaussian noise, which is more suited for continuous data-types. These modified metrics for text provide insights into how specific words influence the model’s predictions. All reported values use the area under the curve (AUC) of the faithfulness score evaluated from 𝑘= 1 to 𝑘= 3, as a default setting, unless otherwise stated. Implementation Details. To generate perturbations for each tabular dataset’s ICL prompt, we use a neighborhood size of 𝜎=0.1 and generate local perturbation neighborhoods N𝑥for each test sample x. We sample n𝑥=10, 000 points for each neighborhood, where the values for 𝜎and n𝑥were chosen to give an equal number of samples for each class, whenever possible. For the text dataset’s ICL prompt, we generate neighborhood sentences by randomly selecting a subset of words to omit, aiming to simulate a range of linguistic variations that enable sensitivity analysis of word-level changes in input sentences. As shown in Figure 2, we present the ICL set in two main formats: 1) as the perturbed sample (x′) and its corresponding predicted output, or 2) as the perturbation (𝛿) around a sample (x) and the corresponding change in output. Note that the perturbation (𝛿) format for the ICL set is used as a default setting in all experiments unless otherwise stated. Additionally, both of these formats are absent from Sec. 3.2, which uses test samples directly and does not compute perturbations. For the LLMs, we use OpenAI’s text generation API with a temperature of 𝜏= 0 for our main experiments. To evaluate the LLM explanations, we extract and process its answers to identify the top-𝑘most important features. We first save each LLM query’s reply to a text file and use a script to extract the listed features. We added explicit instructions like “. . . provide your answer as a feature name on the last line. Do not provide any further details on the last line.” to ensure reliable parsing of LLM outputs. In rare cases, the LLM won’t follow our requested response format or it replies with “I don’t have enough information to determine the most important features.” See Appendix A.1 for further details.
# 4.2 Results
Next, we discuss our results that answer key questions highlighted at the beginning of this section about LLMs as post hoc explainers (Q1-Q3). 1) ICL strategies can identify important tabular features. We compare the proposed prompting based LLM explanation strategies, namely P-ICL, PG-ICL, and E-ICL to existing post hoc explainers on the task of identifying important features for understanding different predictive models for tabular and text classification datasets. For the ANN model, the LLM-based explanations perform on par with post hoc explainers (despite having white-box access to the underlying predictive model or training surrogate linear model). We observe that LLM explanations, on average, achieve 50.13% lower PGU and 144.82% higher PGI than ITG, SHAP, and random baselines for larger datasets (more number of features) like Adult and Credit compared to 27.49% lower PGU and 20.67% higher
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6997/69970437-f500-41fb-a780-bc9ac44d9d6c.png" style="width: 50%;"></div>
Fig. 6: Top: FA (left) and RA (right) scores of explanations generated using post hoc explainers and GPT-4 (P-ICL, PG-ICL, and E-ICL strategies) for an LR model. Center: PGU (left) and PGI (right) scores for a three-layer ANN model. Bottom: PGU (left) and PGI (right) scores for a six-layer ANN. On average, across four datasets and three predictive models, ICL strategies demonstrate non-trivial post hoc explanation capabilities: E-ICL explanations (with in-context examples selected from LIME) match the faithfulness of gradient-based/LIME methods; P-ICL and PG-ICL explanations achieve more faithful scores than ITG and SHAP methods, while demonstrating similar faithfulness to LIME16 (i.e., given the same number of input perturbations).
PGI for Blood and Recidivism datasets. The improved performance of LLM explanations in datasets with more features suggests that LLMs may be better suited for handling the complexities of larger datasets. While the LLM prompting strategies achieve competitive PGU and PGI scores across different datasets for ANN models, the PG-ICL strategy, on average across four datasets, achieves higher FA and RA scores than P-ICL for the LR model (Fig. 6). Moreover, on average, PG-ICL achieves 6.89% higher FA and 16.43% higher RA across datasets compared to P-ICL. We find that gradientbased methods and LIME achieve almost perfect FA and RA scores as they can get accurate model gradients and approximate the model behavior with high precision. Interestingly, the LLM-based explanations perform better than ITG, SHAP, and Random baseline methods, even for a linear model. Additionally, for the E-ICL prompting strategy, LLM-augmented explainers achieve similar faithfulness to their vanilla counterparts. The results show that LLMs generate explanations that achieve faithfulness performance on par with those generated using post hoc explanation methods for LR and ANN predictive models across all five datasets (Fig. 12; see Table 5 for complete results) and four evaluation metrics. We demonstrate that very few in-context examples (𝑛ICL = 4) are sufficient to make the LLM mimic the behavior of any post hoc explainer and generate faithful explanations, suggesting the effectiveness of LLMs as an explanation method, indicating that LLMs can effectively utilize their inherent capabilities to maintain the faithfulness of explanations in line with traditional methods. 2) ICL strategies identify important words for sentiment classifiers. Natural language is one domain of particular interest when assessing the capabilities of LLMs as explainers of other predictive models. When providing examples of removed words and corresponding changes in sentiment classification, we find that GPT-4 is able to consistently identify a top-3 most important wordset that achieves higher faithfulness than gradient-based layer-wise attributions (LGxA, LIG,
Fig. 7: PGU-text (left) and PGI-text (right) faithfulness scores of the sentiment classifier’s explanations generated using post hoc explainers and LLMs (P-ICL and PG-ICL strategies) for Yelp, IMDb, and Amazon review datasets. Across three datasets, ICL strategies consistently demonstrate non-trivial post hoc explanation capabilities, achieving higher faithfulness than gradient-based layer-wise attributions, and approaching the performance of current state-of-the-art (LIME).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6f7f/6f7f4510-e1e9-4908-bd9d-e2bfbcdc0897.png" style="width: 50%;"></div>
Fig. 8: ICL set representation: Faithfulness explanation performance of P-ICL and PG-ICL using the perturbed samples (x′) and the raw perturbations (𝛿) as shown in Fig. 2 in the prompt for LR (left) and ANN (right) models. On average, across both prompting strategies and Blood and Adult datasets, we find that generating ICL samples using the raw perturbation format results in significantly better faithfulness performance across all four metrics.
<div style="text-align: center;">Fig. 8: ICL set representation: Faithfulness explanation performance of P-ICL and PG-ICL using the perturbed samples (x′) and the raw perturbations (𝛿) as shown in Fig. 2 in the prompt for LR (left) and ANN (right) models. On average, across both prompting strategies and Blood and Adult datasets, we find that generating ICL samples using the raw perturbation format results in significantly better faithfulness performance across all four metrics.</div>
LA, LGS, LDL, and LC), according to the modified PGU/PGI metrics described in Sec. 4.1. While LIME yields more faithful top-3 wordsets on average, we do observe cases where GPT-4 outperforms LIME-16, e.g., a 35% vs 30% increase in PGI-text over the random baseline on the Amazon dataset, using the P-ICL prompting strategy. This approaches the gold standard of LIME at around 41%, indicating promising capabilities given that a) only 16 examples are provided to the LLM, b) our strategies use solely the prompt in order to derive explanations (i.e., ICL rather than finetuning). PG-ICL achieves higher faithfulness than P-ICL for the Yelp dataset (roughly a 30% vs 10% increase over random), whereas P-ICL, on average, yields better increases on the Amazon and IMDb datasets (around 14% vs 9%, and 35% vs 27%, respectively). 3) Ablation study. Here, we show how prompt modifications and choice of LLMs affect explanation faithfulness. a) ICL set representation. Does the choice between the raw perturbation (𝛿) and the perturbed sample (x′) affect faithfulness? On average, across the Blood and Adult datasets for both LR and ANN (Fig. 8), our results show that using raw perturbation in the prompts significantly aids LLMs in discerning the most important features. We find that providing only the raw perturbation bypasses the LLM’s need to internally compute the difference w.r.t. the test sample and this relational perspective allows the LLM to focus directly on variations in input and output. b) Choice of LLMs. How do different LLMs impact faithfulness? Here, we perform an ablation using different models from the Gpt family, viz. Gpt-3.5, Gpt-4, and Gpt-4-0125-preview. Our results in Fig. 9 show that faithfulness performance, on average, improves with LLM’s capabilities. In particular, we observe that Gpt-4-0125-preview achieves better FA, RA, and PGI scores across both datasets and prompting strategies. We attribute this improvement to the LLM’s updated knowledge and ability to reduce the “laziness” cases, where the model doesn’t complete a task. Please refer to Appendix A.2 for additional ablation results on the impact of enforcing CoT, using context in the prompting template, different ICL set sizes (𝑛ICL = {4, 8, 12, 16, 32}), and other
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b269/b269159b-26ec-4a2d-b9d2-6ffda54d16e2.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 9: Choice of LLMs: Faithfulness explanation performance of P-ICL and PG-ICL prompting strategies on Blood and Adult datasets for different Gpt models. On average, across the LR (left) and ANN (right) models and both datasets, the Gpt-4-0125-preview explanations outperform the Gpt-3.5 and Gpt-4 models.</div>
open-sourced LLMs like Llama and Mixtral. We detail this ablation in the Appendix due to space constraints.
open-sourced LLMs like Llama and Mixtral. We detail this ablation in the A constraints.
# 5 Conclusion
We introduce a novel framework, In-Context Explainers, and explore the potential of using LLMs as post hoc explainers. To this end, we propose three prompting strategies — Perturb ICL, Perturb+Guide ICL, and Explain ICL— leveraging the context, dataset, and varying levels of instructions to generate explanations using LLMs for other predictive models. We conducted many experiments to evaluate LLM-generated explanations using eight datasets. Our results across different prompting strategies highlight that LLMs can generate faithful explanations, similar to post hoc explainers. Our work paves the way for several exciting future directions in explainable artificial intelligence (XAI) to explore LLM-based explanations.
# References
1. Brown T, Mann B, Ryder N, et al. Language models are few-shot learners. NeurIPS 2020. 2. Zvyagin M, Brace A, Hippe K, et al. GenSLMs: Genome-scale language models reveal SARSCoV-2 evolutionary dynamics. The International Journal of High Performance Computing Applications 2023. 3. Lee J, Yoon W, Kim S, et al. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics 2020. 4. Alsentzer E, Murphy JR, Boag W, et al. Publicly available clinical BERT embeddings. arXiv 2019. 5. Li T, Shetty S, Kamath A, et al. CancerGPT for few shot drug pair synergy prediction using large pretrained language models. NPJ Digital Medicine 2024. 6. Wei J, Tay Y, Bommasani R, et al. Emergent abilities of large language models. arXiv 2022. 7. Liu P, Yuan W, Fu J, Jiang Z, Hayashi H, and Neubig G. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys 2023. 8. Aky¨urek E, Schuurmans D, Andreas J, Ma T, and Zhou D. What learning algorithm is incontext learning? investigations with linear models. arXiv preprint arXiv:2211.15661 2022. 9. Ribeiro MT, Singh S, and Guestrin C. “Why should I trust you?” Explaining the predictions of any classifier. In: KDD. 2016. 10. Doshi-Velez F and Kim B. Towards a rigorous science of interpretable machine learning. arXiv 2017.
11. Lundberg SM and Lee SI. A Unified Approach to Interpreting Model Predictions. In: NeurIPS. 2017. 12. Zeiler MD and Fergus R. Visualizing and understanding convolutional networks. In: ECCV. 2014. 13. Smilkov D, Thorat N, Kim B, Vi´egas F, and Wattenberg M. Smoothgrad: Removing noise by adding noise. arXiv 2017. 14. Sundararajan M, Taly A, and Yan Q. Axiomatic Attribution for Deep Networks. In: ICML. 2017. 15. Radford A, Jozefowicz R, and Sutskever I. Learning to Generate Reviews and Discovering Sentiment. 2017. arXiv: 1704.01444 [cs.LG]. 16. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. NeurIPS 2017. 17. Hegselmann S, Buendia A, Lang H, Agrawal M, Jiang X, and Sontag D. Tabllm: Few-shot classification of tabular data with large language models. In: AISTATS. PMLR. 2023. 18. Yeh IC, Yang KJ, and Ting TM. Knowledge discovery on RFM model using Bernoulli sequence. Expert Systems with applications 2009. 19. ProPublica. How We Analyzed the COMPAS Recidivism Algorithm. https://www.propublica. org/article/how-we-analyzed-the-compas-recidivism-algorithm. Accessed: 2024-0614. 20. Kaggle. Adult Income Dataset. https : / / www . kaggle . com / wenruliu / adult - income dataset. Accessed: 2020-01-01. 21. UCI. Default of Credit Card Clients Data Set. https://archive.ics.uci.edu/ml/datasets/ default+of+credit+card+clients. Accessed: 2020-01-01. 22. FICO. Explainable Machine Learning Challenge. 2019. url: https://community.fico.com/ s/explainable-machine-learning-challenge?tabset-158d9=3 (visited on 06/13/2024). 23. Kotzias D. Sentiment Labelled Sentences. UCI Machine Learning Repository. 2015. 24. Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In: NeurIPS. 2019. arXiv: 1912.01703 [cs.LG]. 25. Shrikumar A, Greenside P, and Kundaje A. Learning important features through propagating activation differences. In: ICML. 2017. 26. Dhamdhere K, Sundararajan M, and Yan Q. How Important Is a Neuron? arXiv:1805.12233 [cs, stat]. 2018. doi: 10.48550/arXiv.1805.12233. url: http://arxiv.org/abs/1805.12233 (visited on 06/14/2024). 27. Kokhlikyan N, Miglani V, Martin M, et al. Captum: A unified and generic model interpretability library for PyTorch. 2020. arXiv: 2009.07896 [cs.LG]. 28. Krishna S, Han T, Gu A, et al. The Disagreement Problem in Explainable Machine Learning: A Practitioner’s Perspective. arXiv 2022. 29. Agarwal C, Krishna S, Saxena E, et al. Openxai: Towards a transparent evaluation of model explanations. NeurIPS 2022.
# A Appendix: Additional results and Experimental details
# A.1 Additional Experimental Details
Bad replies. The total number of occurrences in which either the LLM didn’t follow our requested response format or it replied with “I don’t have enough information to determine the most importan features” is detailed in Table 1.
<div style="text-align: center;">Table 1: Percentage of ‘bad replies’ produced by Gpt-4 across all datasets, models, and prompting strategies (P-ICL, PG-ICL, and E-ICL under default prompt settings). Bad replies include refusal to answer or failure to return answers in a parseable format. Here, Rec. stands for Recidivism.</div>
Method
LR
ANN
Blood Rec. Credit Adult
Blood Reci. Credit Adult
Mean percentage
GPT-3.5
P-ICL
0%
1%
2%
1%
0%
0%
0%
1%
0.62%
PG-ICL
0%
0%
0%
0%
0%
0%
0%
0%
0.00%
E-ICL
0%
0%
0%
0%
0%
0%
0%
0%
0.00%
GPT-4
P-ICL
0%
0%
3%
18%
0%
0%
3%
14%
4.75%
PG-ICL
0%
0%
0%
0%
0%
0%
0%
0%
0.00%
E-ICL
0%
0%
0%
0%
0%
0%
0%
0%
0.00%
GPT-4-Preview
P-ICL
0%
0%
3%
2%
0%
0%
3%
2%
1.25%
PG-ICL
0%
0%
0%
4%
0%
0%
0%
1%
0.62%
E-ICL
0%
0%
4%
18%
0%
0%
0%
1%
2.88%
Datasets. The Blood dataset [18] comprises of 4 attributes of 748 donors to a blood transfusion service from Taiwan. The task is to determine whether patients return for another donation. The Recidivism dataset [19] has criminal records and demographics features for 6,172 defendants released on bail at U.S state courts during 1990-2009. The task is to classify defendants into bail (unlikely to commit a violent crime if released) vs. no bail (likely to commit one). The Credit dataset [21] includes financial and demographic data from credit card users at a bank, such as age, gender, education, marital status, credit limit, payment history, and bill amounts for several months. The primary task is to predict whether an individual will default on their payment. The Adult Income dataset [20] contains demographic (e.g., age, race, and gender), education (degree), employment (occupation, hours-per week), personal (marital status, relationship), and financial (capital gain/loss) features for 45,222 individuals. The task is to predict whether an individual’s income exceeds $50K per year vs. not. The HELOC dataset [22] comprises of financial (e.g., total number of trades, average credit months in file) attributes from anonymized applications submitted by 9,871 real homeowners. A HELOC (Home Equity Line of Credit) is a line of credit typically offered by a bank as a percentage of home equity. The task is to predict whether applicants will repay their HELOC within 2 years. The Amazon reviews dataset contains textual reviews from users describing their experiences with products. Each review includes a rating provided by the user. The task involves predicting the sentiment (positive or negative) of each review based on its content. The IMDb dataset consists of movie reviews taken from the Internet Movie Database website. Each entry includes a textual review along with an associated sentiment label. The task involves predicting the sentiment (positive or negative) of each review based on its content. The Yelp dataset comprises user-generated reviews from Yelp, an online platform where individuals review various businesses and services. It contains reviews along with their corresponding
star ratings, ranging from 1 to 5. The task involves predicting the sentiment (positive or negative) of each review based on its content. Architecture details of ANNs. For the tabular dataset classsifiers, we used two different neural network models in our experiments: ANN-L, which has three hidden layers of size 64, 32, and 16, using ReLU for the hidden layers and Softmax for the output, and ANN-XL, which has six hidden layers of size 512, 256, 128, 64, 32, and 16 using ReLU for the hidden layers and Softmax for the output. Architecture details of the Transformer. For the sentiment classification task, we use a transformer model with a token embedding layer to transform a tokenized sentence into real-dimensional vectors. The embeddings are added with sinusoidal positional encodings and then passed through a transformer encoder, configured with a single layer and four attention heads. The transformer also was trained with a dropout layer with 𝑝= 0.5, followed by a linear binary-classification output layer and a softmax activation. LLM perturbation hyperparameters. We use the LLM’s top-𝑘features to calculate explanation faithfulness using four evaluation metrics. For calculating PGU and PGI metrics, we use perturbation mean 𝜇𝑃𝐺=0, standard deviation 𝜎𝑃𝐺=0.1, and the number of perturbed samples 𝑚𝑃𝐺=10, 000. We follow the default hyperparameters from OpenXAI for generating explanations from standard post hoc explainers. Metrics. We follow [29] and used their evaluation metrics in our work. Below, we provide their respective definitions. a) Feature Agreement (FA) metric computes the fraction of top-𝐾features that are common between a given post hoc explanation and the corresponding ground truth explanation. b) Rank Agreement (RA) metric measures the fraction of top-𝐾features that are not only common between a given post hoc explanation and the corresponding ground truth explanation, but also have the same position in the respective rank orders. c) Prediction Gap on Important feature perturbation (PGI) metric measures the difference in prediction probability that results from perturbing the features deemed as influential by a given post hoc explanation. d) Prediction Gap on Unimportant feature perturbation (PGU) which measures the difference in prediction probability that results from perturbing the features deemed as unimportant by a given post hoc explanation. For a given instance x, we first obtain the prediction probability ˆ𝑦output by the underlying model 𝑓, i.e., ˆ𝑦= 𝑓(x). Let 𝑒x be an explanation for the model prediction of x. In the case of PGU, we then generate a perturbed instance x′ in the local neighborhood of x by holding the top-𝑘features constant, and slightly perturbing the values of all the other features by adding a small amount of Gaussian noise. In the case of PGI, we generate a perturbed instance x′ in the local neighborhood of x by slightly perturbing the values of the top-𝑘features by adding a small amount of Gaussian noise and holding all the other features constant. Finally, we compute the expected value of the prediction difference between the original and perturbed instances as:
PGI(x, 𝑓, 𝑒x, 𝑘) = Ex′∼perturb(x, 𝑒x, top-𝐾)[|ˆ𝑦−𝑓(x′)|],
PGU(x, 𝑓, 𝑒x, 𝑘) = Ex′∼perturb(x, 𝑒x, non top-𝐾)[|ˆ𝑦−𝑓(x′)|],
where perturb(·) returns the noisy versions of x as described above. For text datasets, we adapt the prediction gap metrics to assess explanation faithfulness more appropriately for natural language inputs: e) Prediction Gap on Important word perturbation (PGI-text) and f) Prediction Gap on Unimportant word perturbation (PGU-text) measure the impact of the presence or absence of specific words identified as mportant or unimportant in the text classifier’s decision. Unlike tabular datasets
(1)
(2)
where perturbations involve adding noise, in text datasets, perturbations involves selectively removing words based on the post hoc explainer’s importance ranking. For a given text instance x represented by a sentence, we first obtain the classifier’s prediction probability ˆ𝑦= 𝑓(x). Let 𝑒x denote an explanation that identifies the importance of words within x. In the case of PGU-text, we generate a perturbed version x′ by removing words not highlighted as top-𝐾important, simulating the omission of supposedly unimportant words. Conversely, for PGItext, x′ is formed by removing the top-𝐾words deemed important, to observe how their absence affects the prediction:
The Area Under the Curve (AUC) of these metrics across varying 𝑘captures the overall faithfulness of the model’s explanations. Hyperparameters for XAI methods. Below, we provide the values for all hyperparameters of the explanation methods used in our experiments. a) LIME. kernel width = 0.75; std LIME = 0.1; mode = ‘tabular’; sample around instance = True; n samples LIME = 1000 or 16; discretize continuous = False b) Grad. absolute value = True c) Smooth grad. n samples SG = 100; std SG = 0.005 d) Integrated gradients. method = ‘gausslegendre’; multiply by inputs = False; n steps = 50 e) SHAP. n samples = 500 f) Layer Integrated Gradients. method = ‘gausslegendre’; n steps = 500 g) Layer Gradient SHAP. n samples = 5; stdevs = 0.0 h) Layer Conductance. method = ‘gausslegendre’; n steps = 50
# A.2 Additional Results
Here, we include additional and detailed results of the experiments discussed in Sec. 4. Identifying top-k=1 feature. To demonstrate the LLM’s capability in identifying the most important feature, we show the faithfulness performance of generated explanations across four datasets. In particular, for the LR model (in Table 7 and Fig. 13), we find significant variations in the feature agreement scores for the most important feature (top-𝑘= 1). Gradient-based methods like Grad, SG, and IG consistently achieved perfect FA scores across all datasets, demonstrating their reliability in identifying key features. In contrast, ITG and SHAP showed considerable variability in FA scores, particularly in the Recidivism and Adult datasets, with ITG recording as low as 0.190±0.039 and 0.020±0.014, respectively. Among the LLM methods, E-ICL performed notably well, achieving 0.490±0.050 in Recidivism and 0.926±0.027 in Credit, outperforming P-ICL and PG-ICL. However, P-ICL showed a significant disparity, with a low FA score of 0.011±0.011 in Recidivism but high scores of 1.000±0.000 in Credit and 0.988±0.012 in Blood dataset. This lower performance in Recidivism is attributed to the LLM’s approach to similarly important features, defaulting to alphabetical order in cases of near-equal importance. PG-ICL generally showed more consistent and higher FA scores than P-ICL, particularly in Recidivism and Adult, indicating its effectiveness in certain contexts by adding detailed guidance to assist in task execution. Our experiments for the ANN model (in Table 8 and Fig. 13) show that the top-𝑘=1 features identified by our proposed prompting strategies achieve similar PGU and PGI scores to gradientbased and LIME post hoc explainers, showing the utility of LLMs when explaining non-linear ML models. In particular, we observe that for Recidivism, Credit, and Blood datasets, the PGI scores obtained by P-ICL, PG-ICL, and E-ICL are on par with that of all gradient-based and LIME explainers. Overall, our results across four datasets and two ML models highlight
(3)
(4)
the reliability of gradient-based methods and highlight the varying effectiveness of our prompting strategies. Ablation: Open-Sourced LLMs. We conduct additional experiments using two open-source LLMs, namely Llama-2 70B and Mixtral 8x7B. Our results show that both models achieve comparable performance on the evaluation metrics. As expected, however, they do not exhibit capabilities as strong as GPT-4 (Fig. 10). This mirrors the trends shown in Fig. 9, demonstrating increasing post hoc explanation capabilities with increasing model size. Ablation: Impact of Chain-of-Thought (CoT). Does incorporating a CoT approach before answering affect faithfulness? We leverage the CoT capability of LLMs by including the phrase “Think about the question” in the instructions. Our results show that for the Blood dataset, CoT’s influence varies and does not consistently improve performance across models and prompting strategies (P-ICL and PG-ICL). Conversely, for the Adult dataset, omitting CoT generally enhances performance across both models and prompting strategies (Fig. 11), highlighting the significance of providing additional guidelines to the prompt in certain contexts. Ablation: Impact of setting the Context. How does the inclusion or exclusion of the prompt Context impact faithfulness? Here, we take the original prompts in Figs. 3-4 and remove the Context from the template. On average, across two datasets and models, adding context doesn’t have a significant impact on the faithfulness scores (Fig. 14), suggesting that the LLM can generate explanations using only the ICL set, question, and instructions. This could imply that the LLMs are either inherently capable of generating faithful explanations without needing extra contextual clues, or that the specific type of context provided in these experiments does not contribute meaningfully to the explanation process.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b8d2/b8d21c9f-73f7-47d7-821a-8da7d5c3d213.png" style="width: 50%;"></div>
Fig. 10: Faithfulness explanation performance of P-ICL and PG-ICL prompting strategies on Blood and Adult datasets for Llama-2 70B and Mixtral 8x7B models. Both achieve comparable performance on the evaluation metrics. However, GPT-4, on average across both datasets and metrics, outperforms the two open-sourced LLMs.
<div style="text-align: center;">Fig. 10: Faithfulness explanation performance of P-ICL and PG-ICL prompting strategies on Blood and Adult datasets for Llama-2 70B and Mixtral 8x7B models. Both achieve comparable performance on the evaluation metrics. However, GPT-4, on average across both datasets and metrics, outperforms the two open-sourced LLMs.</div>
Ablation: ICL set size. What impact do different ICL set sizes (𝑛ICL = 4, 8, 12, 16, and 64) have on faithfulness? Our ablation on the number of ICL samples (Fig. 15) shows that fewer and larger numbers of ICL samples are not beneficial for LLMs to generate post hoc explanations. While fewer ICL samples provide insufficient information to the LLM to approximate the predictive behavior of the underlying ML model, a large number of ICL samples increases the input context, where the LLM struggles to retrieve relevant information from longer prompts, resulting in a decrease in the faithfulness of the explanations generated by LLMs. In contrast to LIME, the faithfulness of LLM explanations deteriorates upon increasing the number of ICL samples (analogous to the neighborhood of a given test sample).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0b13/0b1306da-8b9e-4fbe-83c9-ca92b3fce296.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 11: Faithfulness explanation performance of P-ICL and PG-ICL prompting strategies on Blood and Adult datasets w/ and w/o CoT (i.e., “Think about the question”) in the prompt template. On average, across the LR (left) and ANN (right) models, both datasets, and prompting strategies, the effectiveness of CoT varies but consistently enhances explanation faithfulness for the Adult dataset.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c120/c12081c8-5c4b-4fcd-b2f0-aa2f94b465e8.png" style="width: 50%;"></div>
<div style="text-align: center;">LLM-Augmented Explainer</div>
Fig. 12: Faithfulness metrics for E-ICL on the Recidivism dataset for six post hoc explainers and their LLM-augmented counterparts for a given LR (left) and ANN (right) model. LLM-augmented explanations achieve on-par performance w.r.t. post hoc methods across all four metrics (see Table 5 for complete results on all other datasets). Faithfulness metrics were computed for the top-𝑘, 𝑘 being the number of features in each respective dataset.
# A.3 LLM Replies
We provide five example prompts below on the Blood, Adult, and IMDb datasets for P-ICL. Se Fig. 16 for a correct reply and Fig. 17 for an incorrect reply on the Blood dataset. See Fig. 18 fo a partially correct reply and Fig. 19 for an incorrect reply on the Adult dataset. See Fig. 20 for a example reply on the IMDb sentiment classification task.
<div style="text-align: center;">Base Post Hoc Explainer</div>
Logistic Regression
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/aaa8/aaa88b41-a766-4a04-91da-aa35eef03b4e.png" style="width: 50%;"></div>
Fig. 14: Faithfulness explanation performance of P-ICL and PG-ICL prompting strategies on Blood and Adult datasets w/ and w/o Context in the prompt template. On average, across LR (left) and ANN (right) models and both datasets, we find that Context doesn’t have a significant impact on the faithfulness scores.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d50c/d50cc61c-6c75-4b9c-a905-c35c841be12e.png" style="width: 50%;"></div>
Fig. 15: FA and RA performance of P-ICL and PG-ICL prompting strategies and LIME as we increase the number of ICL samples (analogous to neighborhood samples in LIME) for the LR model. In contrast to LIME, the faithfulness of LLM explanations across different metrics decreases for a higher number of ICL samples. This is likely due to limited capabilities of the LLM in tackling longer prompts, and/or its reluctance to analyze them (the number of successfully parsed replies decreases as we increase the number of samples).
Dataset
LR
ANN
Blood
Recidivism
Credit
Adult
70.59%
76.90%
87.37%
77.37%
64.71%
76.90%
88.34%
80.11%
Neural Network
Dataset
Transformer
Amazon reviews
IMDb
Yelp
72.50%
76.25%
71.25%
Table 4: Average faithfulness metric values and their standard errors for explanations across 100 test instances. It compares explanations from Perturb ICL and Perturb+Guide ICL using Gpt-4, six post hoc explanation methods, and a random baseline for LR and ANN predictions on four datasets. For the LLM methods, we queried the LLM for the top-𝑘= 5 (𝑘= 4 for Blood) most important features and calculated each metric’s area under the curve (AUC) for 𝑘= 3 (where the AUC is calculated from 𝑘= 1 to 𝑘= 3). Arrows (↑, ↓) indicate the direction of better performance.
↑ ↓
LR
ANN
Dataset
Method
FA (↑)
RA (↑)
PGU (↓)
PGI (↑)
PGU (↓)
PGI (↑)
Grad
1.000±0.000 1.000±0.000 a 0.010±0.000 0.042±0.000 0.060±0.009 0.115±0.013
SG
1.000±0.000 1.000±0.000 0.010±0.000 0.042±0.000 0.060±0.009 0.115±0.013
IG
1.000±0.000 1.000±0.000 0.010±0.000 0.042±0.000 0.061±0.009 0.116±0.013
ITG
0.722±0.019 0.563±0.037 0.019±0.001 0.037±0.001 0.081±0.010 0.100±0.012
SHAP
0.723±0.020 0.556±0.037 0.019±0.001 0.036±0.001 0.085±0.011 0.098±0.012
LIME
1.000±0.000 1.000±0.000 0.010±0.000 0.042±0.000 0.061±0.009 0.116±0.013
Random
0.502±0.022 0.232±0.032 0.029±0.001 0.026±0.001 0.091±0.011 0.090±0.012
Prediction-based ICL 0.834±0.013 0.724±0.023 0.013±0.000 0.041±0.000 0.061±0.010 0.105±0.013
Blood
Perturb+Guide ICL 0.875±0.016 0.742±0.039 0.013±0.000 0.041±0.001 0.064±0.010 0.104±0.013
Grad
1.000±0.000 1.000±0.000 0.059±0.003 0.106±0.005 0.095±0.008 0.149±0.011
SG
1.000±0.000 1.000±0.000 0.059±0.003 0.106±0.005 0.095±0.008 0.149±0.011
IG
1.000±0.000 1.000±0.000 0.059±0.003 0.106±0.005 0.096±0.008 0.149±0.011
ITG
0.493±0.021 0.214±0.030 0.090±0.005 0.078±0.004 0.129±0.011 0.122±0.010
SHAP
0.473±0.023 0.217±0.032 0.092±0.005 0.076±0.004 0.130±0.011 0.122±0.010
LIME
1.000±0.000 1.000±0.000 0.059±0.003 0.106±0.005 0.096±0.008 0.149±0.011
Random
0.308±0.023 0.127±0.024 0.101±0.005 0.063±0.005 0.146±0.011 0.092±0.009
Prediction-based ICL 0.746±0.003 0.088±0.005 0.061±0.003 0.105±0.005 0.096±0.008 0.147±0.012
Recidivism
Perturb+Guide ICL 0.756±0.013 0.275±0.037 0.065±0.004 0.103±0.005 0.099±0.009 0.146±0.012
Grad
1.000±0.000 1.000±0.000 0.065±0.005 0.195±0.009 0.072±0.008 0.173±0.011
SG
1.000±0.000 1.000±0.000 0.065±0.005 0.195±0.009 0.072±0.008 0.172±0.011
IG
1.000±0.000 1.000±0.000 0.065±0.005 0.195±0.009 0.074±0.008 0.172±0.010
ITG
0.211±0.026 0.157±0.026 0.150±0.006 0.106±0.012 0.155±0.009 0.089±0.011
SHAP
0.212±0.026 0.161±0.026 0.150±0.006 0.107±0.012 0.150±0.008 0.098±0.012
LIME
0.988±0.005 0.985±0.007 0.065±0.005 0.195±0.009 0.071±0.008 0.173±0.010
Random
0.173±0.020 0.095±0.020 0.185±0.010 0.054±0.006 0.176±0.011 0.053±0.007
Prediction-based ICL 0.622±0.008 0.604±0.008 0.077±0.006 0.192±0.009 0.080±0.009 0.171±0.011
Credit
Perturb+Guide ICL 0.646±0.014 0.594±0.020 0.081±0.007 0.186±0.009 0.084±0.009 0.165±0.011
Grad
0.999±0.001 0.999±0.001 0.056±0.006 0.221±0.011 0.081±0.011 0.228±0.014
SG
0.999±0.001 0.999±0.001 0.056±0.006 0.221±0.011 0.080±0.011 0.227±0.014
IG
1.000±0.000 1.000±0.000 0.056±0.006 0.221±0.011 0.082±0.011 0.228±0.014
ITG
0.385±0.012 0.099±0.019 0.215±0.011 0.061±0.007 0.227±0.014 0.075±0.010
SHAP
0.387±0.012 0.150±0.020 0.215±0.011 0.061±0.007 0.225±0.014 0.075±0.010
LIME
0.963±0.012 0.953±0.015 0.056±0.006 0.221±0.011 0.078±0.011 0.229±0.014
Random
0.130±0.017 0.053±0.015 0.198±0.012 0.054±0.008 0.213±0.014 0.064±0.010
Prediction-based ICL 0.541±0.022 0.450±0.033 0.086±0.009 0.197±0.012 0.110±0.013 0.197±0.015
Adult
Perturb+Guide ICL 0.669±0.021 0.622±0.028 0.075±0.007 0.210±0.011 0.107±0.013 0.208±0.014
<div style="text-align: center;">Table 5: Average faithfulness metric values and their standard errors for explanations across 100 test instances. It compares explanations from Explain ICL using Gpt-4 and six post hoc methods for LR and ANN predictions on four datasets. Metrics were calculated for the top-𝑘features, with 𝑘matching the dataset feature count. Arrows (↑, ↓) denote the direction of improved performance.</div>
atching the dataset feature count. Arrows (↑, ↓) denote the direction of improved performa
LR
ANN
Dataset
Method
FA (↑)
RA (↑)
PGU (↓)
PGI (↑)
PGU (↓)
PGI (↑)
LLM-Lime 1.000±0.000 0.978±0.011 0.000±0.000 0.041±0.001 0.074±0.009 0.099±0.012
Lime
1.000±0.000 1.000±0.000 0.008±0.000 0.043±0.000 0.044±0.006 0.121±0.013
LLM-Grad 0.997±0.003 0.996±0.004 0.008±0.000 0.043±0.000 0.058±0.009 0.116±0.012
Grad
1.000±0.000 1.000±0.000 0.008±0.000 0.043±0.000 0.044±0.006 0.120±0.013
LLM-SG
0.990±0.006 0.983±0.011 0.008±0.000 0.043±0.000 0.055±0.008 0.116±0.012
SG
1.000±0.000 1.000±0.000 0.008±0.000 0.043±0.000 0.044±0.006 0.120±0.013
LLM-IG
0.989±0.005 0.982±0.009 0.008±0.000 0.043±0.000 0.046±0.007 0.120±0.013
IG
1.000±0.000 1.000±0.000 0.008±0.000 0.043±0.000 0.044±0.006 0.120±0.013
LLM-Shap 0.684±0.013 0.401±0.025 0.020±0.001 0.034±0.001 0.069±0.009 0.102±0.012
Shap
0.773±0.014 0.516±0.033 0.015±0.001 0.038±0.001 0.066±0.009 0.107±0.012
LLM-ITG
0.702±0.013 0.387±0.029 0.017±0.001 0.036±0.001 0.069±0.010 0.105±0.012
Blood
ITG
0.774±0.014 0.532±0.034 0.014±0.001 0.038±0.001 0.063±0.008 0.108±0.012
LLM-Lime 0.990±0.001 0.958±0.005 0.029±0.001 0.115±0.002 0.048±0.001 0.165±0.004
Lime
1.000±0.000 1.000±0.000 0.029±0.002 0.116±0.006 0.044±0.004 0.164±0.012
LLM-Grad 0.997±0.001 0.990±0.003 0.029±0.001 0.115±0.002 0.048±0.001 0.165±0.004
Grad
1.000±0.000 1.000±0.000 0.029±0.002 0.116±0.006 0.043±0.004 0.165±0.012
LLM-SG
0.997±0.001 0.990±0.003 0.029±0.001 0.115±0.002 0.047±0.001 0.165±0.004
SG
1.000±0.000 1.000±0.000 0.029±0.002 0.116±0.006 0.043±0.004 0.165±0.012
LLM-IG
0.996±0.001 0.988±0.003 0.029±0.001 0.115±0.002 0.048±0.001 0.166±0.004
IG
1.000±0.000 1.000±0.000 0.029±0.002 0.116±0.006 0.044±0.004 0.165±0.012
LLM-Shap 0.666±0.004 0.216±0.008 0.057±0.001 0.098±0.002 0.082±0.002 0.151±0.004
Shap
0.670±0.012 0.200±0.024 0.058±0.003 0.099±0.005 0.087±0.008 0.146±0.011
LLM-ITG
0.690±0.004 0.247±0.008 0.056±0.001 0.099±0.002 0.085±0.002 0.148±0.004
Recidivism
ITG
0.689±0.011 0.195±0.022 0.056±0.003 0.100±0.005 0.078±0.007 0.149±0.011
LLM-Lime 0.909±0.001 0.632±0.005 0.023±0.001 0.222±0.003 0.035±0.002 0.230±0.004
Lime
0.907±0.005 0.743±0.017 0.018±0.002 0.224±0.011 0.029±0.005 0.235±0.014
LLM-Grad 0.938±0.000 0.801±0.001 0.022±0.001 0.223±0.003 0.035±0.002 0.230±0.004
Grad
0.999±0.001 0.997±0.003 0.018±0.002 0.224±0.011 0.029±0.004 0.234±0.014
LLM-SG
0.938±0.000 0.802±0.001 0.022±0.001 0.223±0.003 0.035±0.002 0.230±0.004
SG
0.999±0.001 0.997±0.003 0.018±0.002 0.224±0.011 0.029±0.004 0.234±0.014
LLM-IG
0.938±0.000 0.804±0.000 0.022±0.001 0.223±0.003 0.033±0.002 0.231±0.004
IG
1.000±0.000 1.000±0.000 0.018±0.002 0.224±0.011 0.031±0.005 0.235±0.014
LLM-Shap 0.676±0.002 0.069±0.003 0.109±0.002 0.148±0.003 0.123±0.003 0.153±0.004
Shap
0.662±0.007 0.107±0.012 0.139±0.009 0.127±0.009 0.144±0.011 0.149±0.013
LLM-ITG
0.665±0.002 0.039±0.002 0.107±0.002 0.150±0.003 0.132±0.003 0.146±0.004
Adult
ITG
0.627±0.006 0.068±0.010 0.175±0.010 0.099±0.009 0.170±0.011 0.130±0.013
LLM-Lime 0.954±0.001 0.787±0.003 0.030±0.001 0.189±0.003 0.042±0.002 0.178±0.003
Lime
0.977±0.004 0.878±0.015 0.030±0.003 0.201±0.009 0.037±0.004 0.186±0.010
LLM-Grad 0.984±0.000 0.896±0.001 0.029±0.001 0.189±0.003 0.042±0.002 0.178±0.003
Grad
1.000±0.000 1.000±0.000 0.030±0.003 0.201±0.009 0.038±0.005 0.185±0.011
LLM-SG
0.984±0.000 0.897±0.000 0.029±0.001 0.189±0.003 0.072±0.003 0.165±0.003
SG
1.000±0.000 1.000±0.000 0.030±0.003 0.201±0.009 0.037±0.004 0.185±0.011
LLM-IG
0.984±0.000 0.896±0.001 0.029±0.001 0.189±0.003 0.041±0.002 0.179±0.003
IG
1.000±0.000 1.000±0.000 0.030±0.003 0.201±0.009 0.041±0.005 0.185±0.010
LLM-Shap 0.543±0.003 0.067±0.004 0.088±0.002 0.140±0.003 0.094±0.003 0.126±0.003
Shap
0.525±0.009 0.086±0.012 0.088±0.005 0.163±0.010 0.091±0.006 0.146±0.011
LLM-ITG
0.526±0.003 0.052±0.003 0.088±0.002 0.139±0.003 0.091±0.002 0.129±0.003
Credit
ITG
0.516±0.010 0.076±0.012 0.086±0.005 0.165±0.010 0.084±0.006 0.152±0.010
Table 6: Average faithfulness metric values (PGI-text and PGU-text) and their standard errors for explanations across 100 test instances. It compares explanations from Perturb ICL and Perturb+Guide ICL using Gpt-4 and Gpt-4-0125-preview for the top-𝑘= 3 most important words on the Yelp, IMDb, and Amazon datasets. We calculated each metric’s area under the curve (AUC) for 𝑘= 3 (where the AUC is calculated from 𝑘= 1 to 𝑘= 3). Arrows (↑, ↓) denote the direction of improved performance.
improved performance.
Yelp
IMDb
Amazon
Method
PGI-text (↑)PGU-text (↓)PGI-text (↑)PGU-text (↓)PGI-text (↑)PGU-text (↓)
PG-ICL (Gpt-4-0125-preview) 0.244±0.035
0.260±0.034
0.329±0.045
0.214±0.039
0.371±0.042
0.205±0.035
P-ICL (Gpt-4-0125-preview)
0.300±0.038
0.231±0.033
0.295±0.040
0.263±0.039
0.390±0.041
0.252±0.037
PG-ICL (Gpt-4)
0.349±0.039
0.243±0.033
0.318±0.043
0.251±0.038
0.386±0.038
0.238±0.036
P-ICL (Gpt-4)
0.300±0.036
0.293±0.037
0.332±0.038
0.205±0.031
0.409±0.038
0.222±0.036
LA
0.212±0.035
0.371±0.041
0.140±0.028
0.418±0.048
0.218±0.034
0.290±0.038
LC
0.170±0.030
0.355±0.042
0.178±0.033
0.339±0.042
0.216±0.035
0.303±0.037
LDL
0.301±0.038
0.305±0.038
0.208±0.034
0.367±0.046
0.297±0.040
0.265±0.038
LGS
0.291±0.039
0.355±0.038
0.186±0.033
0.375±0.046
0.240±0.035
0.313±0.040
LGxA
0.280±0.038
0.331±0.042
0.229±0.037
0.373±0.045
0.327±0.042
0.272±0.039
LIG
0.291±0.040
0.346±0.040
0.217±0.035
0.394±0.047
0.263±0.040
0.358±0.042
LIME
0.382±0.038
0.239±0.033
0.486±0.039
0.163±0.031
0.429±0.040
0.181±0.031
LIME-16
0.380±0.041
0.258±0.033
0.359±0.044
0.201±0.034
0.396±0.041
0.171±0.032
Random
0.271±0.029
0.251±0.027
0.291±0.032
0.284±0.031
0.304±0.032
0.281±0.034
Table 7: Faithfulness scores for the most important feature value, top-𝑘= 1, identified by existing post hoc explanation methods as well as the three LLM methods which generated explanations from Gpt-4 across four datasets and the LR model. (Since FA = RA for top-𝑘= 1, we omit RA to avoid
redundancy).
Recidivism
Adult
Credit
Blood
Method
FA (↑)
PGU (↓)
FA (↑)
PGU (↓)
FA (↑)
PGU (↓)
FA (↑)
PGU (↓)
Grad
1.000±0.000 0.096±0.005 1.000±0.000 0.073±0.007 1.000±0.000 0.081±0.006 1.000±0.000 0.020±0.000
SG
1.000±0.000 0.095±0.005 1.000±0.000 0.073±0.007 1.000±0.000 0.081±0.006 1.000±0.000 0.020±0.000
IG
1.000±0.000 0.096±0.005 1.000±0.000 0.073±0.007 1.000±0.000 0.081±0.006 1.000±0.000 0.020±0.000
ITG
0.190±0.039 0.108±0.006 0.020±0.014 0.221±0.011 0.270±0.044 0.163±0.007 0.700±0.046 0.026±0.001
SHAP
0.210±0.041 0.108±0.006 0.020±0.014 0.221±0.011 0.270±0.044 0.163±0.007 0.700±0.046 0.026±0.001
LIME
1.000±0.000 0.096±0.005 0.990±0.010 0.221±0.011 1.000±0.000 0.081±0.006 1.000±0.000 0.020±0.000
Random 0.130±0.034 0.113±0.006 0.060±0.024 0.214±0.011 0.070±0.026 0.195±0.010 0.190±0.039 0.038±0.001
P-ICL
0.011±0.011 0.102±0.005 0.716±0.050 0.116±0.012 1.000±0.000 0.081±0.007 0.988±0.012 0.020±0.000
PG-ICL 0.269±0.046 0.101±0.005 0.869±0.034 0.094±0.009 0.918±0.028 0.092±0.008 0.845±0.039 0.023±0.001
E-ICL
0.490±0.050 0.098±0.005 0.919±0.027 0.086±0.009 0.926±0.027 0.090±0.007 0.758±0.045 0.025±0.001
<div style="text-align: center;">Table 8: Faithfulness scores for the most important feature value, top-𝑘= 1, identified by existing post hoc explanation methods as well as the three LLM methods which generated explanations from Gpt-4 across four datasets and the ANN model.</div>
Gpt-4 across four datasets and the ANN model.
Recidivism
Adult
Credit
Blood
Method
PGU (↓)
PGI (↑)
PGU (↓)
PGI (↑)
PGU (↓)
PGI (↑)
PGU (↓)
PGI (↑)
Grad
0.147±0.011 0.117±0.010 0.103±0.013 0.224±0.014 0.085±0.009 0.166±0.010 0.087±0.012 0.103±0.012
SG
0.146±0.011 0.117±0.010 0.103±0.013 0.224±0.014 0.084±0.009 0.167±0.010 0.087±0.012 0.102±0.012
IG
0.147±0.011 0.116±0.010 0.103±0.013 0.225±0.014 0.085±0.009 0.167±0.010 0.087±0.012 0.103±0.012
ITG
0.154±0.012 0.084±0.009 0.232±0.014 0.056±0.009 0.181±0.010 0.057±0.009 0.103±0.012 0.083±0.012
SHAP
0.152±0.012 0.092±0.009 0.231±0.014 0.047±0.008 0.169±0.009 0.076±0.011 0.104±0.012 0