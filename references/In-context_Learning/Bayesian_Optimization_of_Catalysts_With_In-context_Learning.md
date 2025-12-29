BAYESIAN OPTIMIZATION OF CATALYSTS WITH IN-CONTEXT LEARNING
A PREPRINT
Mayk Caldas Ramos
Department of Chemical Engineering
University of Rochester
mcaldasr@ur.rochester.edu
Shane S. Michtavy
Department of Chemical Engineering
University of Rochester
smichtav@che.rochester.edu
Marc D. Porosoff
Department of Chemical Engineering
University of Rochester
marc.porosoff@rochester.edu
Andrew D. White∗
Department of Chemical Engineering
University of Rochester
andrew.white@rochester.edu
Large language models (LLMs) are able to do accurate classification with zero or only a few examples (in-context learning). We show a prompting system that enables regression with uncertainty for in-context learning with frozen LLMs (GPT-3, GPT-3.5, and GPT-4), allowing predictions without features or architecture tuning. By incorporating uncertainty, our approach enables Bayesian optimization for catalyst or molecule optimization using natural language, eliminating the need for training or simulation. Here, we performed the optimization using the synthesis procedure of catalysts to predict properties. Working with natural language mitigates the challenges of synthesizability since the literal synthesis procedure is the model’s input. We show that in-context learning improves past a model context window (maximum number of tokens the model can process at once) as data is gathered via example selection, allowing the model to scale better. Although our method does not outperform all baselines, it requires zero training, feature selection, and minimal computing while maintaining satisfactory performance. We also find Gaussian Process Regression on text embeddings is strong at Bayesian optimization. The code is available in our GitHub repository: https://github.com/ur-whitelab/BO-LIFT.
arXiv:2304.05341v1
Keywords Bayesian Optimization, large language model, in-context learning, catalysis, materials design, AI
# 1 Introduction
Large language models (LLMs) are revolutionizing a wide range of domains across various fields of knowledge. Whil their most prominent applications lie in computational areas (such as natural language processing[1], text transla tion, classification[2–4] and summarization[5], question answering[6], sentiment analysis[7], code generation[8–10] LLMs have also made significant contributions in diverse areas including legal analysis[11], biomedical research[12 finance[13], education[14, 15], social sciences[16], medicine[17], and entertainment[18–20].
Large language models (LLMs) are revolutionizing a wide range of domains across various fields of knowledge. While their most prominent applications lie in computational areas (such as natural language processing[1], text translation, classification[2–4] and summarization[5], question answering[6], sentiment analysis[7], code generation[8–10]), LLMs have also made significant contributions in diverse areas including legal analysis[11], biomedical research[12], finance[13], education[14, 15], social sciences[16], medicine[17], and entertainment[18–20]. In chemistry, LLMs have been employed to support research efforts in the areas of material design[21–24], reaction design[25, 26] and prediction[27–30], property prediction[31–34], and drug design[35–37]. Most of these applications are based on training a model using the transformer[38] architecture on a specific dataset. Recent advances have also led to the development of pre-trained models like the generative pre-trained transformer (GPT)[39] that are task-agnostic models capable of learning from a few examples shown to the model during inference time, termed in-context learning
In chemistry, LLMs have been employed to support research efforts in the areas of material design[21–24], reaction design[25, 26] and prediction[27–30], property prediction[31–34], and drug design[35–37]. Most of these applications are based on training a model using the transformer[38] architecture on a specific dataset. Recent advances have also led to the development of pre-trained models like the generative pre-trained transformer (GPT)[39] that are task-agnostic models capable of learning from a few examples shown to the model during inference time, termed in-context learning
(ICL)[40]. These models can even perform classification, and to a lesser extent regression, from tabular data, without needing any changes to their architecture or training methods[41]. Jablonka et al. [34] recently showed that it is possible to predict material and chemical properties with decoder-only models like GPT-3[40] using Language-Interfaced Fine-Tuning (LIFT)[41]. LIFT is the process of converting features and labels into a complete sentence, followed by fine-tuning. For example, using mapped experimental conditions “Oxidative coupling of methane using Mn-Na2WO4/ZSM-5 was ran at 750 C, with a total reactant flow rate of 20 mL/min (Ar: 8.0 mL/min, CH4: 8.0 mL/min, O2: 4.0 mL/min). The resultant C2 yield is 4.81 percent” as a training example and at inference, "Oxidative coupling of methane using Co-Na2WO4/SiO2 was ran at 775 C, with a total reactant flow rate of 20 mL/min (Ar: 3.0 mL/min, CH4: 12.8 mL/min, O2: 4.3 mL/min). The resultant C2 yield is " will be prompted for completion." An overview of this process and our prompts is shown in Figure 1.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/dd6a/dd6adbd7-9950-42de-82a8-391ee1632a42.png" style="width: 50%;"></div>
Figure 1: Our approach uses a Language-Interfaced Fine-Tuning (LIFT) framework with a Generative Pre-trained Transformer (GPT) to generate tokens that represent the reaction conditions that include a synthesis procedure. The catalyst synthesis and testing data is converted to an embedding vector and mapped to an objective function, e.g. C2 yield. We create prompts involving multiple-choice options ("Multi", left panel) and single context completions ("Topk", right panel). In this scheme, multi selects only one example (k = 1) from the list, while topk selects two. The prompt starts with an instruction to the LLM of what the generated completion should resemble. Following this, the formatted examples for in-context learning are created. While multi generates five alternatives around the label, topk only uses the label. Finally, the sequence for which the model should respond is added to both prompts. The two lower panels show actual prompts generated by using Multi (left panel) and Topk (right panel) templates. Both examples illustrate the string to the LLM when selecting k = 1 to create the prompt. For multi, the model generates all five options, whereas for topk the model only generates the numerical value of the prediction.
ICL is prefixing the prompt with examples to “train” without actually modifying the LLM parameters[40]. ICL has been found to be less accurate than fine-tuning for LIFT-style regression or classification of materials and molecules[34]. Our
goal in this paper is to obtain prediction uncertainty from the LLM to optimize materials and molecules using Bayesian optimization (BO), which is iterative cycles of training and proposing new points to label[42]. ICL is preferred because it simply requires changing the prompt at each iteration, but limited context windows of the underlying LLM prevent ICL from being directly applied past a certain training amount. The main contributions of our work are to show: (1) ICL can scale and improve accuracy up to 1,000s of examples by context selection; (2) regression with uncertainty is possible via token scores and prompts; (3) uncertainties enable BO to design catalysts and reaction conditions. To evaluate our methodology, we employed it for predicting two properties: aqueous solubility of drug-like molecules and reaction yield. Specifically, we used the ESOL dataset[43] and catalyst dataset for oxidative coupling of methane derived from the work of Nguyen et al.[44]. Designing new catalysts is a critical challenge to address climate change. About one-third of the global gross domestic product (GDP) relies on catalytic processes to produce value-added chemicals and fuels from fossil-based resources[45]. One of the main challenges is navigating the complex experimental space to identify materials capable of converting carbon-neutral or negative feed stocks into value-added products.[46–48] Here we represent catalysts with the natural language used in their synthesis recipes, which makes zero assumptions about catalyst structure and mechanism, with implicit synthesizability embedded into the experiments proposed by the model. The way we represent materials is obviously critical for the ability to predict properties and is challenging in complex materials, such as inorganic zeolites.[49, 50]. The idea of using natural language has been explored in predicting materials properties, where the materials description is embedded into a vector via a model trained with unsupervised learning [1, 21, 51, 52]. In contrast, we are using decoder-only models – the LLMs are directly predicting the catalyst property from reaction conditions and synthesis procedures. Such decoder-only models are highly saught after in catalysis, because they can also be used to do inverse design and suggest an actionable experiment[24]. Aqueous solubility is a fundamental physicochemical property of chemical compounds, defined as the maximum concentration of a solute that can dissolve in water at a given temperature and pressure.[53] It plays a crucial role in various fields, including drug design, environmental studies, and chemical engineering [54–58]. Accurate prediction of solubility is essential for the development of efficient pharmaceuticals, understanding the fate of chemicals in the environment, and optimizing industrial processes. However, its accurate prediction remains a challenge.[59, 60] Traditional methods for predicting solubility have been based on experimental measurements, quantitative structureactivity relationships (QSAR), and molecular simulations[61, 62]. However, these approaches can be time-consuming, expensive, and sometimes inaccurate, especially for large and diverse sets of compounds[63]. In this context, LLMs have shown promising results in predicting chemical properties, reaction outcomes, and synthesizability[25–27, 34]. There has been much more work in exploring language in the related fields of molecules and bio-macromolecules[21– 23, 28–30, 34, 64, 65]. Specifically, the use of pre-trained LLMs with Bayesian optimization has been done in Yang et al. [66]. In chemistry, Frey et al. [67] have explored using decoder-only GPT models for molecular design. There has also been multiple encoder-decoder models as well, which can produce novel compounds and be used to do fine-tuning for property predictions[68–70]. However, there is to our knowledge no work on regression with ICL and Bayesian optimization. There has been recent work exploring what is possible with ICL, and improving its accuracy. ICL[40] combined with chain of thought techniques[71–73] or symbolic tools (e.g., programming languages) has been explored for improving accuracy[74, 75]. The order and identity of the examples are important[76] and can be arranged. Dai et al. [77] also showed that ICL may be viewed as a kind of gradient-descent like process.
# 2 Methods
IUPAC names of molecules were used to predict aqueous solubility (from ESOL[43] dataset), and natural language descriptions of synthesis procedures were used to predict C2 yield for oxidative coupling of methane (2CH4 + O2 → C2H4 + 2H2O). Nguyen et al.[44] evaluated 12,708 different experimental configurations for a range of parameters, including catalyst composition, temperature, and flow rate. The conditions were converted to natural language with the following prompt: “Given {x}, what is {property_name}?” An example prompt for querying the C2 yield of the reaction is “To synthesize Mn-Na2WO4/BN , BN (1.0 g) was impregnated with 4.5 mL of an aqueous solution consisting of Mn ( 0.37 mol) , Na ( 0.37 mol) , W ( 0.185 mol) , at 50 ºC for 6 h. Once activated the reaction is ran at 900 ºC. The total flow rate was 20 mL/min (Ar: 14.0 mL/min, CH4: 4.8 mL/min, O2: 1.2 mL/min), leading to a contact time of 0.38 s.” BO of materials can be reviewed in Shahriari et al. [78], Liang et al. [79], Baird et al. [80], and Valleti et al. [81]. BO is a black-box optimization method where we seek to maximize a function that is expensive, noisy and/or has unknown analytical form.[82] By incorporating uncertainty information, BO guides the exploration-exploitation trade-
off, selecting the most informative points for evaluating the true objective function.[83] Consequently, it enables a more efficient search for optimal materials and chemical systems, minimizing the number of expensive experimental evaluations required [84]. In the case of our two problems, we are trying to maximize C2 yield and solubility. BO reduces the amount of experimental conditions and catalysts synthesis needed to optimize yield[85]. BO requires uncertainty in model predictions, usually achieved via Gaussian process regression (GPR)[42]. We used LangChain[86] to call the GPT-3[40] and the GPT-4 [87] models, trained and maintained by the OpenAI company. We apply the LIFT[41] approach to create the prompts to be fed into the LLMs (see Figure 1). In the LIFT procedure, features are incorporated into natural language sequences to allow LLM to address non-language downstream tasks. To do ICL, we prefix previous examples. To enable ICL beyond a few examples, instead of prefixing all examples, we down-sample our in-context examples to solve the problem of ICL past the LLM context window using max marginal relevance (MMR) selection[88, 89]. MMR combines the search for the most similar examples to a specific sequence while maximizing diversity. We compute similarity via Euclidean distance between Ada embeddings[90]. Given the features (prompt) that we want to predict, we find the most relevant examples and prefix them at inference time. This prompt creation procedure is implemented in LangChain[86] using the FAISS library[91] and Ada-002 embeddings[90]. We do regression with uncertainty by using token probabilities, similar to the action selection process used in Ahn et al. [92] (Figure 1). To get uncertainty, we devised two prompting strategies: (i) multiple choice option template and (ii) Top k completions template. Those templates are referred to as “multi” and “topk” throughout this study. The rationale for multi was that only a single token (the option letter) indicated the choice so that the probability of that one token could succinctly give a range of choices. Topk requires generating k completions and then the logprobs of the entire completion are marginalized to compare. This produces discrete probability distributions, which can then be used directly with acquisition functions for BO[42]. Finally, these two methods are combined into a BO loop for optimizing catalysts and reaction conditions. This is advantageous because the BO approach requires no training with minimal computing required for LLM inference. Figure 1 illustrates how each template is constructed by selecting k = 1 examples as context. Both templates follow the same general prompt structure: “{prefix}{few-shot template}{suffix}”. The {prefix} gives context on how the structure of the LLM’s response to avoid hallucinations. {few-shot template} creates the context by concatenating k examples using the following structure: “Given {representation}. What is {property_name}? {completion}”. Since the {prefix} is fixed and the {suffix} uses the same structure: “Given {representation}. What is {property_name}?”, multi and topk differ on how they build {completion}. When using multi, five options (A, B, C, D, and E) are considered. We used five options because that is the number of token probabilities returned in the OpenAI API. The correct label is randomly placed as one option, and the four options are filled via sampling from y × s, s ∼N(1, 0.1), where N is a normal distribution. Examples of prompts created using each template are shown in the bottom panels of Figure 1. Hence, the LLM completes all five alternatives, generating five possible completions. On the other hand, topk simply uses the label as {completion}, and then it requests generating n = 5 completions. For “chat” type models (e.g., GPT-4), logprobs are not provided and we weight each completion as equally probable – a strong limitation. The above prompting creates a discrete probability distribution. The completions are converted into real numbers and their associated probabilities describe the distribution. The usual BO acquisition functions expected improvement (EI), and upper confidence bound (UCB) can be reformulated for discrete distributions. EI, uei(x), balances exploration and exploitation and is defined for discrete probability distributions as follows:
# uei(x) = E [max(0, y∗−yi)] = � pi max(0, y∗−yi)
� Where pi, yi are the pairs of predicted values/probabilities at x using the model and y∗is the maximum known measured y value. The other acquisition functions can be similarly defined.
y value. The other acquisition functions can be similarly defined. Sometimes we observe only one value from topk because the additional completions differ in a whitespace token, punctuation, or number formatting. When this occurs, we replace the discrete distribution with a normal distribution centered at the value with a standard deviation equal to the sample standard deviation of all yi in training data. For our baselines, we employed LIFT (fine-tuning), Kernel Ridge Regression[93] (KRR), Gaussian Process Regressor[94] (GPR), and k-Nearest Neighbors[95] (KNN). LIFT, based on the text-ada-001 model, utilized the topk template and was trained for eight epochs with a learning rate multiplier of 0.02, following the parameters established by Jablonka et al. [34]. For KRR, we embedded the prompts generated by the topk approach using the text-embedding-ada-002 model. Both the embedded input and the labels were normalized during the training, and we employed a regularization strength of α = 0.5. In contrast, we trained GPR using three types of inputs: (1) LIFT
(1)
prompts embedded with the "text-embedding-ada-002" model, (2) the same prompt using the Material BERT [21] model, and (3) the numerical features vector employed in crafting the natural language prompts. Due to the high dimensionality of these embeddings (e.g., 1536 dimensions for the ada embeddings), training GPR models with a large number of data points proved to be computationally challenging. To avoid these limitations, we applied isometric mapping with 32 components for dimensionality reduction[96], facilitating more efficient model training for GPR. Lastly, we implemented KNN using the SemanticSimilarityExampleSelector from the LangChain [86] library. In this work, we focused on only k = 1 nearest neighbor for our analysis. The ICL and the GPR baseline model predict uncertainty and are used as surrogate models in BO[79, 97]. Unless otherwise stated, all examples are with a finite pool of data and the goal of the Bayesian optimization is to find the point with the maximum label. We implemented an ask-tell interface for our methods. Firstly, a random example is taken from the pool of possible elements. This example is used to initialize the ICL or GPR model. Since fine-tuning takes significant GPU resources and time, it is infeasible to fine-tune repeatedly throughout the BO. For that reason, we did not use the fine-tuned models as surrogate models. Secondly, the model goes through repeated ask (finding the pool member that maximizes u(x)) and tells (labeling the point chosen). To increase efficiency, rather than evaluate all pool members on the acquisition function, we used a Hyde[98] like strategy of inversely designing predicting a hypothetical ˆx from 1.5 × y∗(a point 50% larger than current maximum) by inverting the prompts. This ˆx is then used with above prompt selection strategy to choose 16 pool examples. All Bayesian optimization plots show y∗ N – the current best at sample count N. The random baseline was estimated via a quantiling of the datapoints. Namely, for random sampling y∗ N is estimated with:
� � � where qi is the ith quantile of y (out of Q) and K is the number of datapoints in the pool. The datasets are divided into training/test groups using a split of 80/20. The ICL model can select from all training examples and the training data is used for the LIFT/GPR training. Unless otherwise stated, we use the text-curie-001 model for hyperparameter selection, which has only 6.7B parameters[40] due to the cost and time of larger models. We did experiments up to GPT-4[87] on select tasks and observed consistent improvement. Thus, we expect our conclusions about comparing prompting strategies and datasets to hold for text-curie-001 and above.
# 3 Results and discussion
First we consider the effect of available examples on ICL in Figure 2. At zero shot (N = 0) the model is relying on potential pre-training on the tasks. The difference between topk and multi prompts in zero-shot is the prefix prompt that explicitly specifies the desired format for predictions. Zero-shot did poorly. For the solubility data, only the multi method produced valid responses, though the accuracy of these predictions was poor. The ESOL[43] dataset, published in 2004, has since served as a benchmark for solubility models; however, despite potential contamination there was poor zero-shot accuracy. The C2 dataset generated constant values for each prompt (refer to the initial panel in Figures A.13 and A.16). The C2 dataset was published after text-curie-001 training, and thus cannot be part of the model’s pre-training data. The model can choose data from N training points to create context, but only with k examples. As k increases from 1 to 10, improvements in mean absolute error (MAE) and Pearson correlation (r) can be observed, with subtle differences between the multi and topk methods. Parity plots are in Figures A.3 and A.6. However, we kept k at 5 due to the large context window required for C2 results. When N > k, examples were selected using the max marginal relevance[89] as implemented in LangChain[86]. This gives the ability to improve accuracy beyond N = k, as shown in Figure 2. Parity plots are shown in Figures A.2 and A.5. The multi method exhibits a stronger dependence on the number of points, indicating its larger sensitivity to the examples chosen for context creation and a reduced ability to leverage pre-training information. Using text-curie-001 with topk prompt shows good results in a low-data regime, with an MAE of ∼1.1 and a correlation of ∼0.6 with few data points. In comparison, using multi requires ∼500 data points to reach similar performance.
(2)
The nearest neighbor prediction (KNN) baseline shows how much accuracy comes from just choosing the closest example (via text embeddings). The kernel ridge regression (KRR) baseline shows how much accuracy comes from averaging around nearby examples. The conclusions from Figure 2 are topk is better than multi, ICL is less accurate than trained baselines relying on text embeddings, and ICL has good likelihoods – meaning they are suited to Bayesian Optimization.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2647/264702f3-fe5c-49a2-be77-3579f418ca66.png" style="width: 50%;"></div>
Figure 2: Dependence of the six models considered in this work as a function of the number of training points N from where the model could select examples to create the context (for ICL models) or to train (for baseline models). In these experiments, our ICL models have a fixed example selector size of k = 5, temperature of T = 1.0, and used the text-curie-001 model. The negative log-likelihood (neg-ll) does not follow a trend on changes in N despite the improvement in the other metrics. The arrows in y-axes indicate the direction of improvement. The results show that using in-context learning (ICL) is enough to be comparable to the fine-tuned model for the solubility dataset. The C2 yield dataset was assembled from text and tables fromNguyen et al. [44] and could not have been in any pre-training data from the models considered here. The C2 dataset exhibits greater complexity than solubility data, as it contains natural language descriptions of catalyst synthesis processes instead of explicit molecular representations. The experimental measurements in the C2 dataset also likely have larger error because catalysts are both synthesized and tested in a high throughput apparatus, potentially leading to the lower accuracy. Among both examples, topk gave good likelihoods but relatively poor correlation or standard deviations. Parity plots for all experiments can be found in Figures A.13 and A.16. Analogous to solubility dataset, both multi and topk methods benefited from increasing the number of training points N. However, as illustrated in Figure 2, performance appears to plateau and does not improve beyond N = 200 in this instance. Unlike the solubility dataset, we did not observe significant improvements as k or T values were increased in the C2 dataset (See Figure A.11). The ranking of model performance in the C2 dataset is similar to the solubility dataset. However, the baselines performed significantly better. We hypothesize that the higher complexity of the C2 dataset, along with the pronounced similarity between data points, made the text-curie-001 model incapable of effectively learning from context alone. As a result, incorporating additional context may inadvertently distract the model and diminish the quality of its predictions. Figure A.11 provides a visual summary of our findings. Figure 3 shows the effect of larger and more recent models for the C2 catalyst dataset. Table 1 contains the hyperparameters used in each experiment depicted in Figure 3 (see Supporting Information for justification of hyperparameters). Specifically, we provided N = 1000 training points to the model and constructed a context with a size of k = 5. The
text-curie-001 model, which was trained using data collected up to October 2019, yielded an MAE of 2.333 and a correlation of 0.530. In contrast, the more recent gpt-4 model, resulted in an MAE of 1.854 and a correlation of 0.613. The observed enhancement in performance from text-curie-001 to gpt-4 cannot be attributed to contamination in the training set, as C2 is first assembled in this work. Instead, it is likely associated with the multi-benchmark continued improvement in accuracy with model size[99]. Figure 3 also display parity plots for KNN, KRR, and finetune. GPT-4 outperformed the KNN baseline, which had an MAE of 2.366 and a correlation of 0.547, but not all baselines. Although GPT-4 has better performance, it has no logprobs and thus we assign equal probability to all outputs. Thus, we continued to use text-davinci-003 in our analysis.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9b2c/9b2cbde5-c07c-407e-ace0-c763747d8143.png" style="width: 50%;"></div>
Figure 3: Parity plots with uncalibrated uncertainties comparing the performance of different large language models (LLM) evaluated on the C2 yield dataset. In the first row, from left to right, we show our results for the topk approach employing the following LLMs: text-curie-001, text-davinci-003, and gpt-4. We can observe that the predictions improved over time as newer models were released. In the second row, we show the results for our baselines: nearest neighbor (KNN), kernel ridge regression (KRR), and fine-tuning using text-ada-001 as base model. Notably, curie struggled to learn from the context for the C2 dataset. Nevertheless, GPT-4’s results are comparable to those of KRR. Hyperparameters are available in Table 1.
The models were found to be over-confident in their calibration [100, 101]. For some data points, the language models predicted only one value with logprobs equal to 1.0. The standard deviation of the training data population was substituted for these examples. We recalibrated the models by introducing an uncertainty scaling factor expected to minimize the miscalibration area associated with each model via the optimize_recalibration_ratio from the Uncertainty-Toolbox[101, 102] as shown in Figure A.22. Figure 4 displays the parity plots after recalibration for the text-davinci-003 and gpt-4 models, as well as the baselines for which uncertainties could be computed. Remarkably, ICL outperforms all considered baselines (see the first row in Figure 4 and Table A.1) — with the additional benefit of not requiring a supervised training process, which can be time-consuming and expensive. Table A.1 presents the metrics obtained in this study and compares them with state-of-the-art (SOTA) models. SOTA models consist of transformers that have been specifically trained for the solubility dataset. We showed that a considerably simpler process — prompt engineering to create the context for ICL
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/202c/202cb566-37ee-47ce-a11d-ea72ad640d16.png" style="width: 50%;"></div>
— could yield comparable performance. On the other hand, ICL could not surpass fine-tuning in the C2 dataset (refer to the second row in Figure 4). ICL cannot provide sufficient context for the model to effectively learn the complexities of the C2 yield dataset. Furthermore, the prompts are larger in this case, and we could not increase the context size to enhance performance. Nevertheless, ICL models still demonstrated performance comparable to that of GPR.
model
prompt
RMSE ↓
MAE↓
r↑
neg-ll↓
text-curie-001
multi
13.487
3.878
0.051
8.139
text-curie-001
topk
3.016
2.271
0.499
16.985
text-davinci-003
multi
3.615
2.576
0.411
15.031
text-davinci-003
topk
2.652
1.996
0.603
4.842
gpt-4
topk
2.683
1.854
0.613
7.629
Fine-tuned text-ada-001
topk
1.936
1.325
0.824
9.775
Kernel Ridge Regression (KRR)
topk
2.114
1.648
0.781
-
Nearest Neighbor (KNN)
topk
3.247
2.366
0.547
-
GPR with ada embeddings
topk
4.173
1.573
0.774
3.160
GPR with MatBERT embeddings
-
5.303
1.786
0.741
2.776
GPR with numerical feature vector
-
4.913
1.722
0.738
3.351
1: Performance metrics associated with predictions on the C dataset with optimal values highlighted in bo
Table 1: Performance metrics associated with predictions on the C2 dataset with optimal values highlighted in bold. Displayed models represent instances for each model type using the hyperparameters selected by the study varying each parameter. For all LLMs, we defined temperature T = 0.7, context size k = 5, and N = 1000 training points. All baselines were trained also considering N = 1000 training points. Displayed metrics were calculated after recalibration.
# 3.1 Bayesian Optimization
After evaluating the models’ performance in regression tasks, we did Bayesian optimization to maximize LogS solubility and the C2 yield. The expected maximum of randomly sampling sequence of values was analytically calculated (refer to Section 2) and used as a baseline for the BO experiments. Fine-tuning for each iteration is impractical, since it requires hours of compute at each iteration. Therefore, LIFT was not included in the BO experiments. An ask-tell interface was used to examine a pool of samples, select the subsequent sample based on an acquisition function, and provide that point to the model with the respective label. These points contributed to constructing the context for future predictions
in ICL or training the GPR model. Five independent trials were carried out for each model, and the average results were computed (Figure 5). Faded lines represent individual trials. Each BO run was started with 100 training data points. The initial training data is used to compute the scaling factor to recalibrate the predicted uncertainty that was used in the BO run.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/509a/509ad4f5-67b8-40c9-a54c-5cd2ca18a3cb.png" style="width: 50%;"></div>
Figure 5: Bayesian optimization (BO) history. We implemented BO using an ask-tell interface. At each new sample, we plotted the maximum label found up to that point. Initially, 100 points were fed to the model to compute the scaling factor to calibrate uncertainty predictions. We employed expected improvement (EI) and upper confidence bound (UCB) as acquisition functions. Experiments with greedy and probability of improvement were also conducted, but we observed that the results do not strongly depend on the acquisition function. The curve labeled “Random” illustrates the expected value of the distribution S = max(y0, y1, y2, ..., yN) for a sequence of size N, representing the scenario when N samples are selected randomly. Optimizing on the solubility dataset outperforms random selection and the GPR baseline, reaching the 99% percentile and identifying the maximum in one of the five replicates. On the other hand, the C2 dataset exhibits increased complexity. Therefore, although in-context learning (ICL) reached the 99% percentile after 15 samples, it failed to locate the maximum value within the pool. It is noteworthy that GPR, while also unable to find the maximum, approached it more closely.
Figure 5 is the progression of BO. Each curve starts from a randomly chosen sample (the same sample was selected for each model in every run to minimize its effect on comparisons). The curves are the maximum value identified up to that point. All models reached the 95% percentile solubility around 5 examples in BO. Notably, both text-davinci-003 and gpt-4 converged to the same maximum value after labeling 15 new samples (beyond initial 100). On average, a maximum of ∼1.2 logS was identified. This value is lower than only three other values in a pool of 882 molecules. Individually, text-davinci-003 found maximum values of LogS 1.11, 1.58, 1.11, 1.11, and 1.11, gpt-4 obtained 1.12, 1.58, 1.09, 1.09, and 1.11. Both models could identify the maximum (1.58) in at least one of the five replicates. These experiments did not strongly depend on the acquisition functions. Training the GPR model with additional samples (beyond the initial 100) did not improve our experiments. The left panel in Figure 6 illustrates the histogram of the solubility dataset, with the average values procured for each model, represented as vertical lines.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d5f1/d5f13d56-0037-4c43-bab1-989d63e8d341.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Histograms illustrating the data distribution for the solubility dataset (left panel) and the C2 dataset (right panel). Dashed lines indicate the mean of 5 BO runs identified by each model during the BO process, utilizing the expected improvement acquisition function. Although gpt-4 and text-davinci-003 outperform GPR in the solubility dataset, their performance in the C2 dataset is slightly inferior in comparison to GPR.</div>
Figure 6: Histograms illustrating the data distribution for the solubility dataset (left panel) and the C2 dataset (right panel). Dashed lines indicate the mean of 5 BO runs identified by each model during the BO process, utilizing the expected improvement acquisition function. Although gpt-4 and text-davinci-003 outperform GPR in the solubility dataset, their performance in the C2 dataset is slightly inferior in comparison to GPR.
C2 showed better performance for GPR with text embeddings than ICL. The ICL models reached the 99th percentile after 15 samples, but their performance did not improve substantially after that. Surprisingly, while text-davinci-003 managed to capture some of the top 66 values in the dataset, gpt-4 was only successful in reaching the top 180. This may be because of the lack of token probabilities in gpt-4, since OpenAI does not expose logprobs for chat models. GPR managed to identify the 18th-best example from a pool of 12,708 procedures. We demonstrated that BO using ICL is feasible in low-data regimes. For the solubility dataset, both text-davinci-003 and gpt-4 using topk outperformed GPR. However, although it did not outperform GPR in our experiments in the C2 yield dataset, we should note that we restricted the number of examples used in the context to mitigate expenses — which could decrease performance — and that LLMs are continuously being enhanced, as discussed earlier. The GPR baseline presented here is also novel to the best of our knowledge, and provides a compelling approach. ICL still may be preferred because it does not require training, can be used for inverse design, and will see continued improvement.
# 3.2 Inverse design
We employed our approach to achieve the inverse design of experimental procedures, building upon the study conducted by Jablonska et al.[34]. For inverse design, we utilized the same ICL concept discussed in previous sections of this work. The prompt for direct prediction was structured as: “Given {representation}. What is {property_name}? {completion}”, which was used to generate the context. A similar structure was used as a suffix to the prompt, removing the {completion} field. Conversely, the inverse design involved reversing the prompt order and requesting the LLM to complete the following prompt structure: “If {property_name} is {property_value}, then {representation} is {completion}”. To explore inverse design, we considered an in-house dataset[103] of CO yield over tungsten carbide catalysts for the reverse water-gas shift reaction (CO2 + H2 ↔CO + H2O). CO yield was experimentally measured for 37 configurations that varied a dopant metal and concentration, tungsten loading, carburization temperature and reaction temperature. We adopted two approaches to predict which experimental procedure should be tested next in the laboratory: (1) feeding all 37 data points to the model and subsequently using the model to select the optimal experiment from a pool of 2158 synthetic data points, and (2) employing the inverse design prompt to inquire which tungsten carbide preparation procedure would result in a CO yield of 20%. We applied gpt-4 to both tasks. Through BO on the existing data, the following procedure was proposed to maximize the EI acquisition function with a CO yield of 19.93 ± 0.69%: “A 15 wt% tungsten carbide catalyst was prepared with a Cu dopant metal at 5 wt% and carburized at 950ºC. The reaction was run at 350ºC." We abbreviate this sample proposed by the EI function as 5Cu-WC(950). Alternatively, when asking gpt-4 to generate an experimental procedure with a CO yield of 20% using ICL and the inverse prompt, it produced the following procedure: “Synthesis procedure: A 25 wt% tungsten carbide catalyst was prepared with an Ni dopant metal at 1.0 wt% and carburized at 900ºC. The reaction was run at 375ºC." We abbreviate the sample proposed by ICL as 1Ni-WC(900).
For the 1Ni-WC(900) experiment at a reaction temperature of 375 ºC predicted by ICL, we obtained a CO yield of 3.2%, which was also lower than expected. Upon inspecting the dataset in an attempt to interpret the prediction by ICL, we observe that the tungsten carbide pool includes three experiments over Ni-WC, two with Ni loading at 0.5 wt% and one with Ni loading of 0.25 wt%, all carburized at 835 ºC. The highest CO yield of the Ni-based samples in the dataset is the 0.25Ni-WC(835) run at 350 ºC, which reached 9.54%. The Ni-based experiments are another important example that demonstrates the limitations of the BO technique for inverse design. Although CO is the desired product from reverse water-gas shift, there is a methanation side reaction that becomes more active with increasing temperature over the Ni-based catalysts as shown in Figure B.23. The CH4 selectivity increases from 59% at 325 ºC up to 84% at 375 ºC, decreasing the CO yield from 4.7% to 3.2%, even though the CO2 conversion increases from 11.9% to 22.7% over the same temperature interval. Because we classified the catalyst with CO yield, which is the product of CO2 conversion and CO selectivity, the model likely could not distinguish between high conversion and CO selectivity and was therefore unable to predict the existence of the methanation side-reaction at elevated temperatures. Upon updating the model with the result of the 1Ni-WC(900) experiment, we anticipate it will become privy to the side reaction over Ni-based catalysts at elevated temperatures, and therefore, avoid the Ni dopant at high temperature in future iterations. Looking back to the C2 dataset from Nguyen et al.[44], it is possible that BO could not predict C2 yields above 20% because the experiments were outliers or anomalies. Out of the 12,708 experiments, only 12 reach C2 yields above 19%. Therefore, there is a distinct possibility that in a pool of 12,708 high throughput experiments, that at least 0.1% are outliers or contain significant measurement error. While inverse design using LLMs is possible, it remains crucial to validate the proposed experimental procedures in the lab to confirm their feasibility, accuracy and reproducibility.
# 4 Conclusion
We showed that using example selection for in-context learning (ICL) enables frozen Large language models (LLM) to learn past their context window size without retraining. The prompting approach developed in the current study enables LLMs to make predictions with uncertainties. Hence, techniques such Bayesian optimization (BO) which use uncertainties are used on the two datasets. In this study, we explored the use of LLMs for downstream regression tasks in chemistry and materials. We applied our models to the prediction of LogS solubility data and C2 yield in catalyst reactions In addition, we carried out BO and inverse design for proposing new experimental procedures to synthesize innovative catalysts. Using well-designed prompts to enable uncertainty prediction from LLMs is a powerful tool for techniques such as Bayesian optimization (BO). Despite the catalyst experimental space’s complexity, ICL-enabled BO performs comparably to BO using Gaussian Process Regression (GPR) with minimal samples. While for more simple prompts, such as the IUPAC names used in the solubility dataset, ICL outperformed GPR. Additionally, LLMs are suitable for inverse design by inverting the prompt order, allowing them to propose new experimental procedures using ICL. LLMs can suggest new plausible experiments.
# Acknowledgments
This research is supported by the National Science Foundation under Grant No. 1764415 and the National Institute of General Medical Sciences of the National Institutes of Health (NIH) under award number R35GM137966. The authors also thank the computational resource and structure provided by the Center for Integrated Research Computing (CIRC at the University of Rochester.
# References
# A Supp. Material
# A.1 Solubility
model
prompt
RMSE ↓
MAE↓
r↑
neg-ll↓
text-curie-001
multi
1.791
1.245
0.613
2.218
text-curie-001
topk
1.811
1.245
0.613
2.218
text-davinci-003
topk
1.185
0.814
0.805
1.901
gpt-4
topk
0.773
0.578
0.921
2.564
Fine-tuned (text-ada-001)
topk
1.558
0.852
0.779
4.166
GPR with ada embeddings
topk
2.652
1.129
0.620
1.795
KNN
-
2.443
1.739
0.456
-
SolTranNet[29]
-
2.99
SMILES-BERT[28]
-
0.47
MolBERT[104]
-
0.531
Regression Transformer[105]
-
0.73
MolFormer[33]
-
0.278
ESOL[43]
-
0.83
0.74
mparison of our ICL models against state-of-the-art(SOTA) models for the solubility data
Table A.1: Comparison of our ICL models against state-of-the-art(SOTA) models for the solubility dataset. All SOTA models are a kind of transformers model trained from scratch to reproduce solubility on ESOL dataset. Missing data in the table means that the authors did not report the metric in their study. The best value for each metric is displayed in bold. Displayed ICL models represent instances for each model type using the hyperparameters selected by the investigation in which we systematically varied each hyperparameter. For all LLMs, we defined temperature T = 0.7, context size k = 5, and N = 700 training points. All baselines were trained also considering N = 700 training points. Displayed metrics were calculated after recalibration.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6cf3/6cf3ce3a-7c6e-44b5-b816-2c3999878648.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.1: Metrics with respect to N, k, and T using the text-curie-001 model on the solubility dataset. The lef panel displays The number of training points N, varying from 1 to 700, while keeping the number of examples k =  and temperature T = 0.05 constant. The middle panel illustrates the relationship between the number of examples  used in the context and model performance, with k ranging from 0 to 10, and fixed values of N = 700 and T = 0.0 The right panel demonstrates the effect of varying the temperature T between 0 and 1, while maintaining N = 700 an k = 5 constant.</div>
# A.1.1 Multi-options prompt
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0188/0188b6f7-98d6-46b3-b1ad-df434a3834ee.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.2: Calibrated parity plots using text-curie-001 with Multi-options prompt on the solubility dataset. Those experiments were made by defining the number of examples used as k = 5 and temperature T = 0.05. The number of training points N from which the model can select the examples was varied in a range from 1 to 700</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/284c/284c8d8d-f180-4c83-ab5a-e02b77ab4161.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4d22/4d222d7c-b8ce-4272-bdd7-0ff7ece45575.png" style="width: 50%;"></div>
# A.1.2 Top k completions prompt
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/056d/056d795f-60d1-4d6e-b3f0-2815762f7878.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.5: Calibrated parity plots using text-curie-001 with topk completions prompt on the solubility dataset. Those experiments were made by defining the number of examples used as k = 5 and temperature T = 0.05. The number of training points N from which the model can select the examples was varied in a range from 1 to 700</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2df4/2df4d337-a22a-49c4-8ea7-5f874ad2b843.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.6: Calibrated parity plots using text-curie-001 with topk completions prompt on the solubility dataset. Those experiments were made by defining the number of examples used as N = 700 and temperature T = 0.05. The number of selected examples k to create the context varied in a range from 0 to 10. For k = 0 (zero-shot), the model mostly hallucinated and generated non-numeric tokens. Therefore, data for k=0 is not shown.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/770c/770cb0fc-7719-4972-b656-d98f42e7c82e.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.7: Calibrated parity plots using text-curie-001 with topk completions prompt on the solubility dataset. Those experiments were made by defining the size of the training set as N = 700 and context size k = 5. The temperature T was varied in a range from 0 to 1.0.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1bd8/1bd87a5d-aa21-4f61-a49d-a59f9293f941.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.8: Calibrated parity plots using fine-tuned text-ada-001 with topk completions prompt on the solubility dataset. Those experiments were made by defining the number of examples used as k = 0 and temperature T = 0.05. The number of training points N from which the model can select the examples was varied in a range from 50 to 700</div>
A.1.4 GPR
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9c2d/9c2d22e5-fdb6-4af4-be1a-c57b8bcbf1fd.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.9: Metrics with respect to N using Gaussian process regressor(GPR) with topk completions prompt on the solubility dataset. To generate the inputs for the GPR, we used text-embeding-ada-002 to embed the prompt. Those experiments were made by defining the number of examples used as k = 0 and temperature T = 0.05. The number of training points N from which the model can select the examples was varied in a range from 1 to 700</div>
A.1.5 KNN
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/03ed/03ed4bd6-24b4-4913-86ab-52b555997539.png" style="width: 50%;"></div>
<div style="text-align: center;">gure A.10: Metrics with respect to N using Nearest Neighbors(KNN) with topk completions prompt on the solubility ataset. To generate the vectors to select the neares neighbor, we used text-embeding-ada-002 to embed the prompt. hose experiments were made by defining the number of examples used as k = 0 and temperature T = 0.05. The umber of training points N from which the model can select the examples was varied in a range from 1 to 700</div>
Figure A.10: Metrics with respect to N using Nearest Neighbors(KNN) with topk completions prompt on the solubility dataset. To generate the vectors to select the neares neighbor, we used text-embeding-ada-002 to embed the prompt. Those experiments were made by defining the number of examples used as k = 0 and temperature T = 0.05. The number of training points N from which the model can select the examples was varied in a range from 1 to 700
# A.2 C2 yield
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1840/1840dc5e-ade9-4998-87f7-6a8d9284a9f1.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6595/6595a2a1-3f6c-494e-9833-2dafd523c3d5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.11: Metrics with respect to N, k, and T using the text-curie-001 model on the C2 yield dataset. The lef panel displays the number of training points N, varying from 1 to 1000, while keeping the number of examples k =  and temperature T = 0.05 constant. The middle panel illustrates the relationship between the number of examples  used in the context and model performance, with k ranging from 0 to 5, and fixed values of N = 1000 and T = 0.05 The right panel demonstrates the effect of varying the temperature T between 0 and 1, while maintaining N = 100 and k = 5 constant.</div>
# A.2.1 Multi-option prompt
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a087/a0870509-4315-473f-8970-fc41e1d405ee.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a5d0/a5d08ac9-1689-432f-b772-24429d041684.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.13: Calibrated parity plots using text-curie-001 with Multi-options prompt on the C2 catalyst dataset. Those experiments were made by defining the number of examples used as N = 700 and temperature T = 0.05. The number of selected examples k to create the context was varied in a range from 1 to 10.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f988/f988f6c1-8cd7-4b4a-b3b6-c86ba0af3383.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.14: Calibrated parity plots using text-curie-001 with Multi-options prompt on the C2 catalyst dataset. Those experiments were made by defining the number of examples used as N = 700 and the number of examples k = 5. The temperature T to generate the completions was varied in a range from 0.05 to 1.0.</div>
# A.2.2 Top k completions prompt
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d2af/d2afeae7-8ef7-4596-8078-634852549452.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2dc2/2dc29f04-937e-49e6-b42f-42f78bab819d.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.16: Calibrated parity plots using text-curie-001 with Topk completions prompt on the C2 catalyst dataset. Those experiments were made by defining the number of examples used as N = 700 and temperature T = 0.05. The number of selected examples k to create the context was varied in a range from 1 to 10.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2323/23235f1d-6ef1-44fe-853c-c4cae95a9a47.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.17: Calibrated parity plots using text-curie-001 with Topk completions prompt on the C2 catalyst dataset. Those experiments were made by defining the number of examples used as N = 700 and the number of examples k = 5. The temperature T to generate the completions was varied in a range from 0.05 to 1.0.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9dfc/9dfc4a37-451a-4bd5-ad2a-7b57a68d0866.png" style="width: 50%;"></div>
Figure A.18: Calibrated parity plots using different models with Multi-options prompt on the C2 catalyst dataset. We considered text-curie-001, text-davinci-003, and gpt-4 defining the number of examples used as N = 1000, the number of examples k = 5 and temperature T = 0.7.
# A.2.3 Finetuning
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7866/7866f3d7-9c67-4f52-a6ee-f7c43083b8dc.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.19: Calibrated parity plots using fine-tuned text-ada-001 with topk completions prompt on the C2 catalyst dataset. Those experiments were made by defining the number of examples used as k = 0 and temperature T = 0.05. The number of training points N from which the model can select the examples was varied in a range from 50 to 1000</div>
A.2.4 GPR
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0071/00711b89-4558-4e19-98bc-190b34aebdf6.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.20: Metrics with respect to N using Gaussian process regressor(GPR) with topk completions prompt on the C2 catalyst dataset. To generate the inputs for the GPR, we used text-embeding-ada-002 to embed the prompt. Those experiments were made by defining the number of examples used as k = 0 and temperature T = 0.05. The number of training points N from which the model can select the examples was varied in a range from 1 to 500</div>
A.2.5 GPR
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/36b5/36b54d25-b30a-42e9-bf63-a6780e2b835e.png" style="width: 50%;"></div>
Figure A.21: Metrics with respect to N using NearestNeighbor (KNN) with topk completions prompt on the C2 catalyst dataset. To generate embedding to select the nearest neighbors, we used text-embeding-ada-002 to embed the prompt. Those experiments were made by defining the number of examples used as k = 0 and temperature T = 0.05. The number of training points N from which the model can select the examples was varied in a range from 1 to 500
# A.3 Calibration
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fd17/fd178eae-0aee-49c0-8184-8329e157d6bb.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure A.22: Miscalibration analysis and correction using Uncertainty-toolbox to introduce a corrective scaling-factor used for regression tasks: davinci | topk (3.783), gpt-4 | topk (4.330), ada | gpr (2.110)</div>
# B Inverse design
# B.1 Reaction Results
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ce72/ce72d46b-1284-4468-93b6-4c0eb9477430.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
<div style="text-align: center;">Figure B.23: Reverse water-gas shift performance of 1Ni-WC(900). The reaction temperature is ramped at a rate of 0.25 ºC/min at a gas hourly space velocity of 27,000 mL/h/gcat with a 3:1 H2:CO2 reactant ratio at 2.1 MPa.</div>
# B.2 In-House Reaction Dataset
<div style="text-align: center;">Table B.2: CO Yield Dataset for Reverse-Water-Gas-Shift Reaction Inverse Catalyst Design</div>
Table B.2: CO Yield Dataset for Reverse-Water-Gas-Shift Reaction Inverse Catalyst Design
Tungsten Loading (wt%)
Dopant
Dopant (wt.%)
Carburization Temp (°C)
Reaction Temperature (°C)
CO yield (%)
15
Fe
0.5
835
280
1.66
15
Fe
0.5
835
350
3.03
15
Fe
5
835
280
1.61
15
Fe
5
835
350
4.12
15
Cu
0.5
835
280
0.52
15
Cu
0.5
835
350
3.36
15
Cu
5
835
280
9.80
15
Cu
5
835
350
18.98
15
Co
0.5
835
280
6.21
15
Co
0.5
835
350
16.35
15
Co
5
835
280
1.73
15
Co
5
835
350
2.85
4.25
–
–
600
350
2.23
4.25
–
–
835
350
5.14
4.25
–
–
1000
350
4.63
15
–
–
600
350
5.72
15
–
–
835
350
8.73
15
–
–
1000
350
5.09
15
Pt
0.5
835
280
2.32
15
Pt
0.5
835
350
7.59
15
Ni
0.5
835
280
2.67
15
Ni
0.5
835
350
7.85
15
Ni
0.25
835
350
9.54
15
Cu
0.25
835
350
4.55
15
Co
0.25
835
350
5.66
15
Fe
0.25
835
350
0.78
15
–
–
680
280
1.47
15
–
–
680
350
10.43
15
–
–
600
350
5.72
15
–
–
835
350
8.73
15
–
–
1000
350
5.09
15
–
–
600
350
5.72
15
–
–
700
350
6.79
15
–
–
800
350
6.87
30
–
–
600
350
7.24
30
–
–
700
350
10.38
30
–
–
800
350
10.89
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/17fc/17fcbce4-7e55-42a8-8c7f-9aff53a718e4.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure B.24: These are the results of applying topk and multi using the SamanticsSimilarity (SS) function from Langchain as opposed to our selected approach of using MMR for context selection. SS uses cosine similarity to select the next context point, whereas MMR offers a more diverse selection of context for robustness. This figure elucidates the importance of distinct context for ICL. These results may be compared to the other regression figures that used the MMR in the rest of the experiments.</div>
