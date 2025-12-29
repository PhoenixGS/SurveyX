# Discovering Attention-Based Genetic Algorithms via Meta-Black-Box Optimization
<div style="text-align: center;">Sebastian Flennerhag DeepMind</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4fd9/4fd9ad9a-ae91-4e58-b2ac-1a15ad160a34.png" style="width: 50%;"></div>
arXiv:2304.03995v1
<div style="text-align: center;">Figure 1: Discovering attention-based Learned Genetic Algorithms (LGA) via MetaBBO. At each meta-iteration one samples a set of inner loop tasks and a set of candidate LGA parameters from a meta-evolutionary optimizers (EO). Afterwards, one runs an inner loop search and compute a normalized meta-fitness score across the tasks. We update the meta-EO and iterate</div>
# ABSTRACT
Genetic algorithms constitute a family of black-box optimization algorithms, which take inspiration from the principles of biological evolution. While they provide a general-purpose tool for optimization, their particular instantiations can be heuristic and motivated by loose biological intuition. In this work we explore a fundamentally different approach: Given a sufficiently flexible parametrization of the genetic operators, we discover entirely new genetic algorithms in a data-driven fashion. More specifically, we parametrize selection and mutation rate adaptation as cross- and selfattention modules and use Meta-Black-Box-Optimization to evolve their parameters on a set of diverse optimization tasks. The resulting Learned Genetic Algorithm outperforms state-of-the-art adaptive
baseline genetic algorithms and generalizes far beyond its metatraining settings. The learned algorithm can be applied to previously unseen optimization problems, search dimensions & evaluation budgets. We conduct extensive analysis of the discovered operators and provide ablation experiments, which highlight the benefits of flexible module parametrization and the ability to transfer (‘plug-in’) the learned operators to conventional genetic algorithms.
# CCS CONCEPTS • Computing methodologies →Genetic algorithms;
genetic algorithm, machine learning, meta-learning
# 1 INTRODUCTION
Motivation. Genetic algorithms (GAs) provide a set of evolutioninspired optimization algorithms, which are flexibly applicable to black-box optimization (BBO) problems. They commonly rely on human designed operators, which impose a restrictive and most importantly subjective set of manual priors. This bears the risk of domain overfitting and limited generalization capabilities. Based on recent results in the discovery of attention-based Evolution Strategies [25], we propose that these limitations can be overcome by meta-learning effective GA operators from data. Thereby, the inductive biases of the GA itself are indirectly encoded by its parametrization & can be discovered in an optimization-driven fashion, i.e. by improving its meta-performance on a distribution of relevant tasks. Approach. Inspired by the recent success of the Set Transformer [27] architecture, we introduce neural network-based architectures to substitute core genetic operators: Selection and mutation rate adaptation are cast as dot-product attention modules, which can flexibly be applied to problems with varying dimensions & population sizes. The resulting family of genetic algorithms can implement different operations based on the specific module weights. We meta-evolve these weights on a set of representative optimization problems using meta-black-box optimization (MetaBBO, [25]). Results. We evaluate the performance and generalization capabilities of the meta-trained Learned Genetic Algorithm (LGA). LGA is capable of generalizing far beyond its meta-training distribution and outperforms established baseline GAs on several BBO benchmarks (BBOB) [2, 13] and neuroevolution problems. This includes different optimization functions, number of search dimensions and population sizes. Furthermore, we analyze the discovered neural network GA operators: The selection operator has learned an adaptive form of truncation, which maintains diversity and redundancy among the parent population. The learned mutation rate adaption operator, on the other hand, automatically scales the amount of exploration in a task-dependent fashion. We investigate the importance of the meta-evolution task distribution: While it is possible to meta-evolve effective LGAs on as little as five BBOB functions, we show that LGAs can overfit their meta-training distribution and one has to ensure sufficient meta-regularization for broad generalization. Trained LGAs are robust to their choice of hyperparameters and the details of their meta-training procedure. Finally, we show that the individual learned operators can replace white-box GA operators inducing a positive transfer effect. Contributions. Our contributions are summarized as follows: (1) We propose a dot-product attention-based parametrization of the selection & mutation rate adaptation GA operators. (2) We discover new GAs by meta-evolving their parameters based on the performance on meta-training BBO tasks. (3) The resulting LGA is capable of generalizing far beyond its meta-training settings and outperforms several GA baselines on various benchmark tasks and evaluation budgets. (4) We perform several ablations to LGA to assess the contributions of learning the individual operator modules. Both learned operators contribute to the overall performance. (5) We highlight the robustness of the trained LGA to its hyperparameters, the transferability of the learned components
# 2 RELATED WORK
Discovery via Meta-Learned Algorithms. Recent efforts have proposed to replace manually designed algorithms by end-to-end optimized inductive biases, by meta-learning parametrized components on a representative task distribution. E.g. this includes the discovery of Reinforcement Learning objectives [28, 33, 44], schedules of algorithm hyperparameters via meta-gradients [9, 34, 45, 47], and the meta-learning of entire learning algorithms [18, 19, 42]. The discovery process can be supported by suitable neural network architectures. Our proposed LGA architecture leverages attention layers to derive a neural network-based family of GAs. Meta-Learned Gradient-Based Optimization. Our work is closel related to the ambition of meta-learning gradient descent-based learning rules [1, 3, 31]. These approaches rely on access to efficient gradient calculations via the backpropagation algorithm and thereby do not apply to BBO problems. A small neural network processes the gradient and standard optimizer statistics (momentum, etc.) to output a weight change. The optimizer network weights in turn have been meta-learned on a task distribution [30]. Metz et al. [29] showed that this results in a highly competitive optimizer for deep learning tasks. Our MetaBBO-discovered LGA, on the other hand, provides a general-purpose BBO, which does not require differentiable objective functions. Meta-Learned Population-Based Optimization. Shala et al. [38] meta-learn a controller for the scalar mutation rate in CMA-ES [15]. Chen et al. [5], Gomes et al. [12], TV et al. [41] previously optimized entire neural network-parametrized algorithms for lowdimensional BBO. All of them use a recurrent network, which processes raw solution candidates and their respective fitness scores. These methods often struggle to generalize to new optimization domains and are often constrained to fixed population sizes and/or search dimensions. Lange et al. [25] recently leveraged the equivariance property of dot-product self-attention to the input ordering [20, 27, 39] to learn adaptive recombination weights for evolution strategies. The proposed LGA extends this attention-based BBO perspective in order to characterize GA operators. After successful meta-training, the learned GA is capable of generalizing to unseen populations and large search spaces. To the best of our knowledge we are the first to demonstrate that a meta-learned GA generalizes to challenging neuroevolution tasks. Finally, the MetaBBO approach does not require access to knowledge of meta-task optima [41] or a teacher algorithm [38]. Baseline Genetic Algorithms. Throughout the paper we compare against four competitive baseline GAs including the following: • Gaussian GA [35]: A simple GA with Gaussian perturbations and fixed mutation strength using truncation selection. • MR-1/5 GA [35]: Doubles the mutation rate if 1/5 of all perturbations were beneficial. Otherwise, the rate is halved. • SAMR-GA [7]: Self-adapts per-parent mutation rates based on a simple co-evolution heuristic and meta-mutation rate. • GESMR-GA [21]: Avoids vanishing mutation rates by using a group elite selection criterion and mutation rate sharing. We additionally consider Sep-CMA-ES [36] as a scalable evolution strategy baseline for neuroevolution tasks. Each baseline GA was tuned using small grid search sweeps (see Appendix B). Otherwise, we adopted the settings provided by the authors.
# 3 BACKGROUND
Black-Box Optimization. Throughout this manuscript, we are interested in efficient continuous black-box optimization: Given a function 𝑓(𝑥) : R𝐷→R with unknown functional form, i.e. we cannot compute its derivative, we seek to find its global optimum:
Genetic Algorithms. GAs provide a class of BBO algorithms, which iteratively evaluate a population consisting of 𝑁solution candidates 𝑋𝐶= [x𝐶 1 , . . . , x𝐶 𝑁]𝑇∈R𝑁×𝐷(‘children’) with fitness f𝐶∈R𝑁. Given a set of 𝐸‘parent’ solutions 𝑋𝑃= [x𝑃 1 , . . . , x𝑃 𝐸]𝑇∈ R𝐸×𝐷with associated fitness f𝑃∈R𝐸, the parents are replaced by the children using a heuristic fitness-based selection criterion. Most GAs make use of truncation selection in which all children and parents [x𝐶 1 , . . . , x𝐶 𝑁, x𝑃 1 , . . . , x𝑃 𝐸]𝑇are jointly sorted by their fitness. The top-𝐸performing solutions replace the parent archive:  
() Commonly the number of parents is set to be small 𝐸≪𝑁and often even 𝐸= 1, enforcing a type of hill climbing. 𝑁children candidates are uniformly sampled with replacement from the parents:
() ∈ Afterwards, they are perturbed using a mutation rate (MR) 𝜎∈R+, which controls the strength of the Gaussian noise, 𝜖𝑗∼N (0𝐷, I𝐷):
Many competitive GAs keep a vector of parent-specific [7] mutation rates 𝝈𝑃∈R𝐸and additionally perform an intermediate mutation rate adaptation (MRA) step to improve the mutation rate(s) given information gathered throughout the fitness evaluations:
() ∈ In this case, the children are perturbed by their sampled individualspecific mutation rate (𝝈𝐶←˜𝝈𝑃∈R𝑁) and selection also applies to the children’s MR. Alternatively, GESMR-GA [21] forms parent sub-groups and online co-evolves group-level mutation rates based on their observed fitness improvements. Set Operations via Dot-Product Self-Attention. Scaled dot-product attention (SDPA) is especially well suited to characterize algorithms performing set operations, since it naturally enforces a permutation invariant function. Consider the standard formulation of SDPA, which embeds a set of 𝑁input tokens, 𝑋∈R𝑁×𝐷, into 𝐷𝐾-dimensional latent query 𝑄, key 𝐾and value𝑉representations:
∈ The output 𝑌is computed as a linear combination of the values: 𝑌= softmax � 𝑄𝐾𝑇/ √︁ 𝐷𝐾 � 𝑉∈R𝑁×𝐷𝐾.
� √︁ � It can be shown that this transformation is equivariant to the ordering of the tokens in 𝑋, i.e. permuting the rows of 𝑋will apply the same permutation to the rows of 𝑌[20, 39]. We will leverage this suitable inductive bias to characterize GAs, which inherently operate on sets of solution candidates and their fitness scores.
Figure 2: Learned Genetic Operators. Top: Cross-attention selection between parent & children fitness features. Bottom: MRA via self-attention on parent mutation & fitness features.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6603/66032c7c-032e-4078-bd9b-68e94feb94bb.png" style="width: 50%;"></div>
# 4 ATTENTION-BASED GENETIC OPERATORS
We now introduce an attention-based parametrization of the genetic Selection and Adaptation operators (Figure 2). These in turn will be meta-optimized on a set of representative optimization tasks in order to capture useful BBO mechanisms. We start by answering a natural question: What inputs should be processed by the operators in order to enable generalization across fitness and solution scales? Attention Features via Fitness Scores & Mutation Strength. To compute attention scores across parents and children, we need to construct sufficient features, which modulate effective selection and mutation rate adaptation. Furthermore, we want the metalearned operations to generalize across different test optimization scales. Hence, we consider scale invariant normalizations: E.g. zscoring and centered ranks (in [−0.5, 0.5]). We transform both the raw fitness scores (𝐷𝐹dim.) and the parent mutation rates (𝐷𝜎dim.) to construct a set of features processed by the attention layers: • 𝐹∈R(𝑁+𝐸)×𝐷𝐹: Joint fitness transformations of parents and children (z-scores & centered ranks). • 𝐹𝐶= 𝐹1:𝑁∈R𝑁×𝐷𝐹: Fitness transformations of children extracted from joint transforms (z-scores & centered ranks). • 𝐹𝑃= 𝐹𝑁:(𝑁+𝐸) ∈R𝐸×𝐷𝐹: Fitness transformations of parents extracted from joint transforms (z-scores & centered ranks). • 𝐹𝑃′ ∈R𝑁×𝐷𝐹: (Separate) fitness transforms of sampled parents after selection operation (z-scores & centered ranks). • ˆ𝝈𝑃∈R𝑁×𝐷𝜎: Mutation strength transformations of sampled parents (z-scores & [-1, 1] normalization). • 𝐹𝑀= [ ˜𝐹𝑃, ˆ𝝈𝑃] ∈R𝑁×(𝐷𝐹+𝐷𝜎): Concatenated fitness and mutation rate features of sampled parents.
The fitness features additionally include a Boolean indicating whethe an individual performs better than the best fitness observed so far. Selection via Cross-Attention. We replace the common sortingbased selection mechanism with a cross-SDPA layer, which compares children and parents. It first embeds the fitness transformations of the parents and children into queries, keys and values:
𝑄𝑃= 𝐹𝑃𝑊𝑄𝑃∈R𝐸×𝐷𝐾, 𝐾𝐶= 𝐹𝐶𝑊𝐾𝐶, 𝑉𝐶= 𝐹𝐶𝑊𝑉𝐶∈R𝑁×𝐷𝐾.
Afterwards, we compute the normalized dot-product cross-attention features 𝐴𝑆and construct a selection matrix 𝑀𝑆∈R𝐸×(𝑁+1):
In the final line we concatenate an 𝐸-dimensional vector of ones to the outer product of 𝑄𝑆and 𝐾𝑆. Intuitively, this column represents a fixed offset used to indicate whether the parent copies any child at all or if it is not replaced. The rows of 𝑀𝑆then specify the probability of each offspring to replace a parent:  
�� �   �� � 𝑀𝑆is row stochastic, i.e. the rows sum to 1. E.g. 𝑀𝑆 11 denotes the probability of replacing parent 1 with child 1, while 𝑀𝑆 1𝑁+1 corresponds to not replacing the first parent. We sample row-wise from a categorical distribution in order to determine whether a child replaces a particular parent. Afterwards, we use the selection matrix to update the parent archive (𝑋𝑃′) and fitness archive (f𝑃′): 𝑆∼Categorical(𝑀𝑆) with 𝑆∈R𝐸×(𝑁+1), 𝑋𝑃′ = 𝑋𝐶· 𝑆:,1:𝑁+ 𝑋𝑃· diag(𝑆:,𝑁:𝑁+1),   
·+·(+) and similarly we obtain f𝑃′ and 𝝈𝑃′ via masked addition. This selection operator can flexibly regulate the amount of truncation selection by replacing multiple parent slots with the same child. Mutation Rate Adaptation (MRA) via Self-Attention. Next to selection we meta-learn MRA. The concatenated fitness and MR features of the sampled parents are processed by a SDPA layer, which outputs a child-specific feature matrix 𝐴𝑀∈R𝑁×𝐷𝐾: 𝐾𝑀= 𝐹𝑀𝑊, 𝑄𝑀= 𝐹𝑀𝑊, 𝑉𝑀= 𝐹𝑀𝑊∈R𝑁×𝐷𝐾,
·+·(+) and similarly we obtain f𝑃′ and 𝝈𝑃′ via masked addition. This selection operator can flexibly regulate the amount of truncation selection by replacing multiple parent slots with the same child. Mutation Rate Adaptation (MRA) via Self-Attention. Next to selection we meta-learn MRA. The concatenated fitness and MR features of the sampled parents are processed by a SDPA layer, which outputs a child-specific feature matrix 𝐴𝑀∈R𝑁×𝐷𝐾: 𝐾𝑀= 𝐹𝑀𝑊𝐾𝑀, 𝑄𝑀= 𝐹𝑀𝑊𝑄𝑀, 𝑉𝑀= 𝐹𝑀𝑊𝑉𝑀∈R𝑁×𝐷𝐾, 𝐴𝑀= softmax �𝑄𝑀(𝐾𝑀)𝑇 √𝐷𝐾 � 𝑉𝑀∈R𝑁×𝐷𝐾.
Δ𝜎= exp(0.5 × 𝐴𝑀𝑊𝜎) ∈R𝑁
The children MR 𝝈𝐶is obtained via element-wise multiplication: 𝝈𝐶= Δ𝜎⊙𝝈𝑃′ ∈R𝑁
⊙ ∈ 𝜃= {𝑊𝑄𝑃,𝑊𝐾𝐶,𝑊𝑉𝐶,𝑊𝑄𝑆,𝑊𝐾𝑆,𝑊𝑄𝑀,𝑊𝐾𝑀,𝑊𝑉𝑀,𝑊𝜎} denotes the joint attention weights, which characterize a specific instance of an LGA. We use a small feature dimension 𝐷𝐾= 16, which results in <1500 trainable meta-parameters. In summary, we introduced two dot-product attention-based operators, which replace the standard selection and MRA operations. Throughout the paper we focus on learned selection and MRA, but in Appendix A we outline how to additionally construct sampling and cross-over operators using self-attention.
# 5 META-TRAINING, OBJECTIVE & TASKS
We meta-optimize the weights of the GA attention modules to perform BBO on a family of representative tasks. More specifically, we make use of the previously introduced MetaBBO procedure [25] and evolve the LGA parameters to maximize performance on a task distribution of 10 BBOB [13] functions. These include functions with different properties, i.e. separability, conditioning, multi-modality (see Table 1). At each meta-generation (see Figure 1) we start by uniformly sampling a set of BBO tasks and LGA parameters from a meta-evolutionary optimization algorithm. We denote the set of parameters characterizing the LGA by 𝜃𝑖for 𝑖= 1, ..., 𝑀metapopulation members. Afterwards, each LGA is evaluated on all tasks by running an inner loop search. We compute an aggregated meta-performance score to update the meta-EO. The MetaBBO-
objective is computed based on the collected inner loop fitness scores of each LGA instance, where 𝜉𝑙denotes task-specific parameters for 𝑙= 1, ..., 𝐽tasks. For each candidate 𝜃𝑖we minimize the final performance of the best population member. Afterwards, we 𝑧-score the task-specific results over meta-population members and compute the median across tasks: � � �� � �
� � �� The outer loop optimizes 𝜃using OpenAI-ES [37] for 750 metagenerations with a meta-population size of 𝑀= 512. We sample
Figure 3: LGA MetaBBO. Meta-evaluation across metagenerations. Evaluation of LGA on two 10 dim. BBOB (Top) and neuroevolution tasks (Bottom). Mean & 1.96 standard error intervals across 3 independent MetaBBO runs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ecee/eceea841-f41b-46e1-b1b9-0b0e0b438057.png" style="width: 50%;"></div>
𝐽= 256 tasks and evaluate each sampled LGA on each task independently. Each LGA is unrolled for 𝑇= 50 generations, with 𝑁= 16 population members in each iterations. Throughout, we assume that 𝐸= 𝑁, i.e. the number of parents is equal to the number of children. This is not a limitation since the selection operator is capable of replacing multiple parents with the same child (see Section 7.1). The large-scale parallel meta-evaluation of all 𝜃𝑖on all 𝜉𝑗tasks is facilitated by the auto-vectorization and device parallelism capabilities provided by the JAX library [4, 23] and runs on multiple accelerators. We provide more details and experiments on the meta-training settings and robustness in Appendix B.1 and C.
# 6 EXPERIMENTS
We now turn to an exhaustive experimental evaluation of the MetaBBO optimization procedure and the discovered LGA. We thereby set out to answer the following questions: (1) Is it possible to meta-evolve competitive LGAs via MetaBBO using a limited set of meta-training BBO tasks (Section 6.1)? (2) Does the resulting LGA outperform GA baselines on unseen BBO problems and different search budgets (Section 6.2)? (3) How much can an LGA discovered on a limited set of tasks generalize beyond its meta-training setting (e.g. hyperparameter optimization & neuroevolution; Sections 6.3 & 6.4)?
We now turn to an exhaustive experimental evaluation of the MetaBBO optimization procedure and the discovered LGA. We thereby set out to answer the following questions:
(1) Is it possible to meta-evolve competitive LGAs via MetaBBO using a limited set of meta-training BBO tasks (Section 6.1)? (2) Does the resulting LGA outperform GA baselines on unseen BBO problems and different search budgets (Section 6.2)? (3) How much can an LGA discovered on a limited set of tasks generalize beyond its meta-training setting (e.g. hyperparameter optimization & neuroevolution; Sections 6.3 & 6.4)?
# 6.1 Meta-Training on BBOB Functions
We start by meta-evolving the LGA parameters on a task distribution consisting of 10 BBOB functions with different random optima offsets, evaluation noise and considered problem dimensionality (𝐷≤10). Throughout meta-training we evaluate the performance of the optimized LGA on several different downstream tasks. These include BBOB functions seen during meta-training, hold-out metatest BBOB functions and small neuroevolution tasks. In Figure 3 we plot the detailed evaluation curves across meta-training. The MetaBBO-trained LGA quickly learns how to perform optimization
Figure 4: Meta-Evaluation on BBOB Tasks. Left: Training Functions. Right: Hold-Out Functions. Scores are normalized by the Gaussian GA baseline performance. Lower is better. Averaged over 50 independent evaluation runs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d86b/d86b7ffb-77fb-4bf2-8b34-3ac757202ef2.png" style="width: 50%;"></div>
on the low-dimensional BBOB meta-training functions. Interestingly, we find that the MetaBBO procedure can lead to an LGA that overfits to the BBOB tasks on which it was meta-trained. The downstream performance of LGA decreases and becomes unstable for both an unseen Pendulum control task with MLP policy and a downsized 14-by-14 MNIST classification task using a CNN. We therefore investigated whether meta-regularization can improve the generalization on such unseen neuroevolution tasks. More specifically, we compared three different meta-mean regulatization coefficients 𝜆∈{0, 0.005, 0.02}, which exponentially decay the meta-mean to zero, 𝜇′ 𝜆= (1 −𝜆)𝜇′. We observe that the generalization to the neuroevolution tasks can be improved and stabilized using a properly chosen decay of 0.005. In Appendix C we further explore the impact of the meta-task distribution, meta-objective and LGA attention size. The MetaBBO procedure is largely robust to the choice of these settings. Small attention layers are sufficient for consistently discovering performant LGAs. This comes with the additional advantage of reducing the FLOPs and memory requirements of executing the LGA. The evaluation of a meta-trained LGA is easily feasible on a single core CPU device.
# 6.2 Meta-Testing on BBOB Functions
Next, we exhaustively evaluate the performance of LGA on the full set of BBOB benchmark functions including test functions unseen during meta-training. We compare against 4 competitive GA baselines: Gaussian GA, MR-1/5 GA, SAMR-GA, GESMR-GA. We compare the performance on all BBOB functions for a population size of 𝑁= 32, 𝐷= 20 search dimensions and for 𝑇= 50 generations. The best-across generations function value is normalized by the performance of the Gaussian GA. In Figure 4 we find that LGA outperforms all baselines on the majority of both BBOB functions seen during meta-training (left) and unseen BBOB functions (right). This holds true for functions with very different characteristics (single/multi-modal, high/low conditioning, separable/not separable), search dimensions and population sizes (Appendix Figure 17 & 18). This provides further evidence that LGA does in fact not
Figure 5: LGA Generalization. Left: BBOB evaluation for different budgets & spaces. Right: Performance across generations. Mean & 1.96 standard error intervals across 50 runs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/998c/998c826a-5cc3-411e-b63d-b3ca53a57494.png" style="width: 50%;"></div>
overfit to the BBO functions seen during meta-training, but instead has discovered a general-purpose GA algorithm. In Figure 5 we further demonstrate that LGA generalizes to different population sizes and problem dimensions. The meta-learned GA achieves lower function values in fewer generations (right) and performs well for different problem settings (left). As the problems become harder with increased dimensionality, LGA can compensate with a larger population size.
# 6.3 Meta-Testing on Continuous HPO-B
Next, we test LGA’s performance on the HPO-B benchmark [2]. The benchmark considers a vast array of hyperparameter optimization tasks including 16 different model types (SVM, XGBoost, etc.) and their respective search spaces (𝐷∈{2, 3, . . . , 16}). Each model is evaluated on 2 to 7 different datasets, which leads to a total of 86 hyperparameter search tasks. We consider the continuous HPO-B version, which uses a previously fitted surrogate model. Note that the LGA has not been trained on such hyperparameter optimization tasks. Figure 6 compares the performance of LGA against the GA and a random search baselines. Additionally, we report the reference performance of the recently proposed OptFormer model [6] after 105 total evaluations. We find that LGA outperforms the majority of considered GA baselines. Again, this observation holds for two considered population sizes. LGA can also achieve similar performance as OptFormer, which has been trained on a much more diverse task distribution. This highlights the transfer capabilities and applicability of LGA to new during meta-training unseen optimization domains.
Figure 6: LGA Evaluation on HPO-B [2]. Left: Small population size (𝑁= 4). Right: Large population size (𝑁= 8). Mean & 1.96 standard error intervals across 5 runs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ea30/ea30db76-cba1-4c47-ad32-0f2889ee5976.png" style="width: 50%;"></div>
<div style="text-align: center;">6.4 Meta-Testing on Neuroevolution Tasks</div>
# 6.4 Meta-Testing on Neuroevolution Tasks
Until now we have evaluated the performance of the discovered LGA on moderately small search spaces (i.e. 𝐷≤20). But is it also possible to deploy LGA to neuroevolution settings with thousands of search dimensions and arguably very different fitness landscape characteristics? Again, note that LGA has never explicitly been trained to evolve such high-dimensional genomes and that this requires strong transfer of the learned GA operators. The considered Reinforcement Learning (RL) tasks consist of 4 robotic control tasks (Pendulum-v1 [24] and 5 Brax tasks [10]) using MLP policies and three MinAtar visual control tasks (SpaceInvaders, Breakout & Asterix [46]) with CNN-based policies. The MLP genomes consist of less than 1000 weights, while the MinAtar CNNs have ca. 50,000. Hence, the search space is many orders higher than what the LGA has been meta-trained on (𝐷≤10). The top three rows of Figure 7 show that LGA can compete with all tuned baseline GAs on the nine RL tasks with different search spaces, fitness landscapes and evaluation budgets. Interestingly, the performance gap between LGA and the considered baselines is the biggest for the CNN policies and tends to increase with the number of search dimensions. Finally, in the final row of Figure 7 we show that LGA can also successfully be applied to three image classification tasks including MNIST, Fashion-MNIST and K-MNIST classification (28-by-28 grayscale images) with a small CNN (2 convolutional layers, ReLU activation and a linear readout) with 11274 evolvable weights. The LGA generalizes far beyond the meta-training search horizon(𝑇= 50 versus 4000) and does not meta-overfit [26].
After having established that the meta-trained LGA is capable of outperforming a set of GA baselines on unseen optimization problems, we now investigate the underlying discovered mechanisms, transfer ability and robustness of the meta-learned GA operators.
# 7.1 Visualization of Learned Genetic Operators
What types of mechanisms underlying the black-box genetic operators has LGA discovered? Has it simply re-discovered fitness-based truncation selection or a more complex parent replacement procedure? In Figure 8 we consider a 2-dim Sphere problem and visualize the selection mask 𝑆:,1:𝑁used to update the parent archive 𝑋𝑃. We observe that the selection operator uses children solution to replace parents based on their improvement over the best seen solution.
Furthermore, one well-performing child often times replaces more than a single parent. This indirectly implies that the selection operator has meta-learned to dynamically adapt its elite archive size and thereby also the effective sampling distribution. A child that has replaced multiple parents will be sampled (with replacement) more frequently in the next generation. Furthermore, this implies a robustness mechanism: Since children can be stored multiple times in the parent archive, they are less likely to be ‘forgotten’ by the stochastic selection. The mutation rates, on the other hand, are decreased over the course of generations in order to explore closer to the global optimum. Furthermore, we observe a grouping of the MR based on the performance of the parents. Children with bad performing parents tend to exhibit a higher mutation rate.
Figure 7: LGA Evaluation on neuroevolution tasks including continuous control tasks (top), visual control (middle) & computer vision (bottom) tasks. Mean & 1.96 standard error intervals across 5 independent runs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e409/e4090cfa-0e44-4a21-96c3-961b95bbf952.png" style="width: 50%;"></div>
<div style="text-align: center;">7.2 Ablation & Transfer of Genetic Operators</div>
How much do the different learned components contribute to the overall performance of LGA? Can the learned modules act as dropin replacements for other genetic algorithms? To answer this question we consider two types of comparative studies: (1) Operator ablation before MetaBBO discovery: We metatrain the LGA with a variable amount of learned genetic operators. E.g. we fix the selection operator to white-box
Figure 8: LGA’s selection operator on a 2-dim Sphere task. Left: Sampled selection matrix 𝑆:,1:𝑁∈R𝐸×𝑁Middle: Matrix indicating whether a child improves the fitness score over the parents. Right: Mutation rates for different children. Rows indicate 4 different generations 𝑡∈{1, 16, 31, 46}.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bfe2/bfe2e479-234b-4517-9df3-f45ebbcf55a8.png" style="width: 50%;"></div>
truncation selection and only meta-learn mutation rate adaptation. This allows us to quantify the joint contributions and synergies between the different learned ingredients. (2) Operator transfer after MetaBBO discovery: After metatraining is completed, we ask whether or not it is possible to substitute the learned operators into other genetic algorithms? This in turn allows us to assess whether the specific learned operator is overfit to the downstream GA computations or whether it can act as a transferable inductive bias for genetic computation.
For the first study we compare meta-training combinations of attention-parametrized selection (SE), mutation rate adaptation (MRA), cross-over (CO) and sampling (SA). For each combination we plot the evaluation performance across meta-generations in Figure 9. We observe that MRA is crucial for good performance on the neuroevolution tasks. Intuitively, this can be explained by the smaller scale of solution parameters associated with neural network weights. The GA benefits from the ability to flexibly downregulate its perturbation strengths. Cross-over, on the other hand, is detrimental for the generalization of LGA to the MNIST CNN
neuroevolution task. This behavior can arguably be attributed to the challenge of finding beneficial crossing over pairs for different neural network genomes. We further observed that learned sampling does not significantly improve the performance of the LGA. We hypothesize that this is due to the indirect effect of the selection mechanism on the sampling of children. Finally, the overall best performing configuration only meta-learns selection and MRA.
Figure 9: Visualization of LGA’s operator ablations during MetaBBO. Evaluation of LGA on two 10 dim. BBOB (Top) and neuroevolution tasks (Bottom). We report mean & 1.96 standard error intervals across 3 independent MetaBBO runs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f6d9/f6d9becf-ed24-4018-b6a5-4cfe4484b4bf.png" style="width: 50%;"></div>
<div style="text-align: center;">Next, we considered replacing the truncation selection and fixed mutation rate of the Gaussian GA baseline with the learned selection and MRA operators. In Figure 10 we show that this can successfully be accomplished for four neuroevolution tasks. Replacing either the selection or adding learned MRA operator improves the performance of the Gaussian GA. The learned operators can act as drop-in replacements and are transferable inductive biases.</div>
Next, we considered replacing the truncation selection and fixed mutation rate of the Gaussian GA baseline with the learned selection and MRA operators. In Figure 10 we show that this can successfully be accomplished for four neuroevolution tasks. Replacing either the selection or adding learned MRA operator improves the performance of the Gaussian GA. The learned operators can act as drop-in replacements and are transferable inductive biases.
Figure 10: Transfer of learned operators to a Gaussian GA. Mean & 1.96 standard error intervals across 5 runs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0313/031316a0-fd65-47cd-a887-c0a66635b653.png" style="width: 50%;"></div>
# .3 Hyperparameter Robustness of LGA
Finally, we assess the sensitivity of LGA to its remaining hyperparameter choices. More specifically, we compare the performance of LGA and the baseline GAs for various initial mutation rate scales 𝜎0 and parent archive sizes 𝐸= ⌈𝜌× 𝑁⌉, where 𝜌denotes the fraction of population members making up the number of parents. Note that while LGA was meta-trained for 𝜌= 1, i.e. 𝐸= 𝑁, we find that it is capable of generalizing to many different archive sizes and is robust to the initial scale parameters (Pendulum control task; see Figure 11). In Section D.2 we provide the same analysis for all neuroevolution tasks. LGA is far less hyperparameter sensitive than the considered baseline GAs. This highlights the robustness of the LGA induced by the MetaBBO process.
Figure 11: Hyperparameter Robustness of LGA. Pendulumv1 performance across elite ratios and initial mutation rates. 𝜌= 0 uses a single parent 𝐸= 1. Results are averaged over 5 independent evaluation runs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6bd7/6bd78ed7-b578-4d6b-be76-b1e01569781e.png" style="width: 50%;"></div>
Summary. In this study, we used evolutionary optimization to discover novel genetic algorithms via meta-learning. We leveraged the insight that GA operators perform set operations and parametrized them with novel attention modules that induce the inductive bias of permutation equivariance. Our benchmark results on BBOB, HPOB and neuroevolution tasks highlight the potential of combining flexible GA parametrization with data-driven meta-evolution. Limitations. We use powerful neural network layers to characterize GA operators. While flexible and interpretable (Section 7.1), the underlying mechanisms do remain partially opaque. Future work needs to be done in order to fully reverse-engineer the discovered operators. In Appendix E we provide a first set of insights unraveling simple linear relationships between the attention inputs and their outputs. We believe these can guide the design of new ‘whitebox’ GAs informed by ‘grey-box‘ discovered LGAs. Furthermore, our analysis highlights the importance of meta-regularization and the potential for the automated design of meta-training curricula. Future Work. We are interested in explicitly regularizing LGAs to maintain diversity in their parent archive. This may provide a bridge to meta-learned quality-diversity methods [8]. Furthermore, it may be possible to parametrize a flexible EO algorithm that can interpolate between the exploration of multiple solution candidates as in GA and a single search distribution mode as in traditional evolution strategies. Finally, we believe that better meta-learned GAs can be discovered by simultaneously co-evolving the meta-task distribution and the learned GA.
# REFERENCES
# ACKNOWLEDGMENTS
This work was funded by DeepMind. We thank Nemanja Rakićević for valuable feedback on this manuscript.
# A ATTENTION-BASED SAMPLING & CROSS-OVER OPERATORS
Next to learned selection and MRA, we additionally experiment with a learned attention-based Sample and CrossOver operator in Section 7.2. Here we provide their formal definitions. Children Sampling Distribution via Self-Attention. We use the parent fitness f𝑃and its transformations 𝐹𝑃together with an age counter 𝑎𝑃and its tanh transformation ˆ𝑎𝑃to compute queries, keys and values. Afterwards, the output is projected and normalized into a probability distribution:
˜𝑄= [𝐹𝑃, ˆ𝑎𝑃]𝑊˜𝑄, 𝐾= [𝐹𝑃, ˆ𝑎𝑃]𝑊˜𝐾∈R𝐸×𝐷𝐾, ˜𝑉= [𝐹𝑃, ˆ𝑎𝑃]𝑊˜𝑉∈R𝐸×1 � �
Figure 12: Extra Learned Genetic Operators. Top: Selfattention parent sampling probabilities. Bottom: Selfattention additive cross-over operator.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6b09/6b09dfad-f6fc-47b0-a517-600244c7d5ab.png" style="width: 50%;"></div>
Cross-Over via Self-Attention. For each dimension 𝑑of the parent matrix 𝑋𝑃 :,𝑑∈R𝐸, we first construct a set of normalized features measuring the diversity across all parents (e.g. z-scored feature, normalized distance, etc.) to obtain ˆ𝑋𝑃 :,𝑑∈R𝐸×𝐷𝐸. We then concatenate [𝐹𝑃, ˆ𝑋𝑃 :,𝑑] ∈R𝐸×(𝐷𝐹+𝐷𝐸), obtain queries, keys and values via embeddings and compute the output of a scaled dot-product self-attention layer:  ˆ  ˆ
 �� � �� � The parameter-specific output 𝑍𝑑is linearly projected to obtain an additive change to the parents. The dimension cross-over between the parents is then computed as the addition of the cross-over output and the original parent archive dimension vectors 𝑋𝑃 :,𝑑: Δ𝑋𝑃 :,𝑑= 𝑍𝑑𝑊Δ𝑋and ¯𝑋𝑃 :,𝑑= 𝑋𝑃 :,𝑑+ Δ𝑋𝑃 :,𝑑∈R𝐸×1, ∀𝑑= 1, ..., 𝐷. This operation can efficiently be parallelized across dimensions 𝐷using for example the auto-vectorization tools provided by JAX.
<div style="text-align: center;">B HYPERPARAMETER SETTINGS B.1 MetaBBO Settings for LGA Discovery</div>
Function
Reference
Property
Tasks
Sphere
Hansen et al. [p. 5, 13]
Separable (Indep.)
Small
Rosenbrock
Hansen et al. [p. 40, 13]
Moderate Condition
Medium
Discus
Hansen et al. [p. 55, 13]
High Condition
Medium
Rastrigin
Hansen et al. [p. 75, 13]
Multi-Modal (Local)
Medium
Schwefel
Hansen et al. [p. 100, 13]
Multi-Modal (Global)
Medium
BuecheRastrigin
Hansen et al. [p. 20, 13]
Separable (Indep.)
Large
AttractiveSector
Hansen et al. [p. 30, 13]
Moderate Condition
Large
Weierstrass
Hansen et al. [p. 80, 13]
Multi-Modal (Global)
Large
SchaffersF7
Hansen et al. [p. 85, 13]
Multi-Modal (Global)
Large
GriewankRosen
Hansen et al. [p. 95, 13]
Multi-Modal (Global)
Large
Table 1: Meta-BBO BBOB-Based Task Families.
We share the task parameters, initialization and fixed randomness for all meta-members 𝜃𝑖. Fixed stochasticity & task-based normalization enhances stable meta-optimization [‘Pegasus-trick’, 32].
Parameter
Value
Parameter
Value
MetaEO
OpenAI-ES [37]
𝑀: Meta-Pop.
512
𝛼0, Decay, Final
{0.01, 0.999, 0.001} 𝐽: Meta-Tasks
256
𝜎0, Decay, Final
{0.1, 0.999, 0.001}
Centered ranks
✓
Meta-Objective
minN-finalT
𝐷𝐾, Att. Heads
{16, 2}
Inner loop 𝐷
∼[2, 10]
Inner loop 𝑇
50
Inner loop 𝜎0
∼[0.01, 0.5]
Inner loop 𝑁
16
Offset 𝑥★−𝑐
∼[−5, 5]
Inner loop noise
Hansen et al. [14]
Table 2: Meta-BBO Hyperparameters (Figure 3).
<div style="text-align: center;">− ∼[−] Table 2: Meta-BBO Hyperparameters (Figure 3). B.2 BBOB Evaluation Settings</div>
<div style="text-align: center;">− ∼[−] Table 2: Meta-BBO Hyperparameters (Figure 3). B.2 BBOB Evaluation Settings</div>
Parameter
Value
Parameter
Value
Population 𝑁
32
Dimensions 𝐷
20
Generations 𝑇
50
𝑋𝑃Initialization
∼[−5, 5]
Table 3: BBOB Hyperparameters (Figures 4, 5).
We tuned the elite ratio 𝜌∈{0.0, 0.15, 0.25, 0.35, 0.5, 1.0} and initial 𝜎0 ∈{0.1, 0.25, 0.5, 0.75, 1.0} of all baselines & LGA via a grid sweep.
# B.3 HPO-B (Continuous) Evaluation Settings
Figure 6 is generated for two different population sizes 𝑁∈{4, 8} and 𝑇= 100. The GA archives are initialized at 0.5 and we follow the evaluation protocol outlined in the benchmark repository. We tuned the elite ratio 𝜌∈{0.0, 0.15, 0.25, 0.35, 0.5, 1.0} and initial 𝜎0 ∈{0.1, 0.25, 0.5, 0.75, 1.0} of all baselines & LGA via a grid sweep.
<div style="text-align: center;">B.4 Neuroevolution Evaluation Settings</div>
Parameter
Value
Parameter
Value
MNIST 𝑁
128
MNIST 𝑇
4000
MNIST CNN L
8 [5, 5] & 16 [3,3]
MNIST Batchsize
1024
Brax 𝑁
256
Brax 𝑇
2000
Norm obs
✓
𝑋𝑃Initialization
0
Episode steps
500
MC Evaluations
8
Brax MLP L
0 - Only readout
Brax Activation
Tanh
MinAtar 𝑁
256
MinAtar 𝑇
4000
MinAtar CNN L
16 [3, 3] Filters
MinAtar MLP L
1 ReLU 32 units
Episode steps
500
MC Evaluations
16
Table 4: Neurevolution Hyperparameters (Figure 7, 10, 11).
MNIST sweep: 𝜌∈{0.0, 0.25, 0.5, 1.0} and 𝜎0 ∈{0.01, 0.025, 0.05}. Brax sweep: 𝜌∈{0.0, 0.25, 0.5, 1.0} and 𝜎0 ∈{0.025, 0.05, 0.1}. MinAtar sweep: 𝜌∈{0.0, 0.25, 0.5, 1.0} and 𝜎0 ∈{0.05, 0.075, 0.1}.
iscovering Attention-Based Genetic Algorithms via Meta-Black-Box Optimizati
C ADDITIONAL METABBO RESULTS
# C.1 Attention Feature Dimension & Heads
Figure 13: LGA MetaBBO - Different model sizes. We report mean/1.96 ste intervals across 3 independent runs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0050/0050f321-90fb-467c-9fbb-f4bc051cfc00.png" style="width: 50%;"></div>
<div style="text-align: center;">C.2 Comparison of Meta-Training Tasks</div>
Figure 14: LGA MetaBBO - Different meta-task distributions. We report mean/1.96 ste intervals across 3 independent runs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b636/b636b8e8-79d9-4145-b92c-ff04a45dbfee.png" style="width: 50%;"></div>
# C.3 Comparison of Meta-Optimizer
Figure 15: LGA MetaBBO - Different meta-EO. We report mean/1.96 ste intervals across 3 independent runs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/45b9/45b9a484-cf7f-4b26-89f5-a7aa4eb55a5e.png" style="width: 50%;"></div>
<div style="text-align: center;"></div>
# C.4 Comparison of Meta-Objective Functions
We also compared differernt meta-objectives, which construct the meta-fitness in different ways: minN-minT: Minimizes over both the population evaluations with a generation and across all generations. minN-finalT: Minimizes over all population evaluations evaluated during the final generation. meanN-minT: Computes the mean performance across members within a generation & minimizes this score across generations. meanN-finalT: Computes the mean performance across all members within the final generation. � �
 𝑙=1 (minN-minT)
    Interpretation of MetaBBO Comparitive Studies. The LGA discovery process is largely robust to the considered MetaBBO specifications. Small attention module parametrizations (single head and 𝐷𝑘= 8) is sufficient to learn powerful GA operators. Furthermore, a small task distributions (single Sphere function with random offsets/noise) already lead to strong performance on BBOB tasks. For MNIST classification more function diversity is required. The choice of the meta-optimizer and objective are more important. OpenAI-ES (Figure 3) and minN-finalT provide the best performance.
Figure 16: LGA MetaBBO - Different meta-objectives. We report mean/std intervals across 3 independent runs.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e746/e746e8ba-ef64-480b-a6c7-092aa175eb0d.png" style="width: 50%;"></div>
D ADDITIONAL EVALUATION RESULTS
# D ADDITIONAL EVALUATION RESULTS
# D.1 Detailed BBOB Results
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ef2e/ef2e2a73-7883-4cc3-97cf-881838a05761.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7d23/7d23bf79-2649-43f3-80a3-7384613f799d.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/841c/841c32c4-dc1e-4cfc-80aa-a7d8316885c3.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/38c9/38c9825c-57bb-4000-b669-eac344efdaf5.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3e7f/3e7f5a85-3561-43c9-b584-6a17ab841880.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ace3/ace39289-9605-4752-a21a-e2d3ec888d18.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 23: Hyperparameter Robustness - Asterix Task.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5e18/5e188a9a-23c8-4465-a597-c4c7dedcbd4c.png" style="width: 50%;"></div>
Figure 24: Hyperparameter Robustness - SpaceInvaders Task.
Figure 24: Hyperparameter Robustness - SpaceInvaders Task.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7041/7041e92f-9adc-49b7-b1ca-163548034f86.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/47b6/47b652b5-fb7c-48ef-8340-e30d671bd18d.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bac6/bac6b287-b093-4f98-9faa-61a81b9b75bb.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9616/9616d6fc-d41d-40f9-80ea-ebf713544309.png" style="width: 50%;"></div>
<div style="text-align: center;">E REVERSE-ENGINEERING THE LEARNED</div>
We further investigated whether there exist simple relationships between the attention input features and the learned operator’s
Figure 28: Hyperparameter Robustness - Walker2d Task.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9b74/9b74f115-7504-4d89-b671-352d1515bbc4.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/803e/803e537e-625e-4eda-911c-11c625c15098.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2a46/2a46ea7f-99f3-4669-b4c5-ad1b160f966a.png" style="width: 50%;"></div>
output. More specifically, we explored if the mutation rate adaption multiplier Δ𝜎and selection logits 𝑚𝑖𝑗can be explained by the normalized fitness features (z-score and centered ranks) of the children. In Figure 31 we plot all features, logits and mutation multipliers
across an LGA evaluation run on a 2-dim Sphere task. We can observe a clear positive correlation between the performance of the children and their selection probability. Furthermore, the mutation rate is decreased for well-performing solutions. This may open up the possibility to reverse-engineer a new discovered GA without the need for arguably opaque neural network modules.
Figure 31: Top: Linear relationship between fitness features and MRA. Bottom: Linear relationship between fitness features and selection logits.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4f7d/4f7d1bf4-6ff8-4c5a-ba59-7404c25e03ec.png" style="width: 50%;"></div>
<div style="text-align: center;">F SOFTWARE, COMPUTE REQUIREMENTS This project has been enabled by the usage of freely available Open Source software. This includes the following:</div>
• A network checkpoint and accompanying architecture code is publicly available in evosax. All experiments (both meta-training and evaluation) were implemented using the JAX library for parallelization of fitness rollout evaluations. Each MetaBBO metatraining was run on 4 RTX2080Ti Nvidia GPUs and take roughly 2.5 hours. The LGA downstream BBOB, HPO-B and gym task evaluations were run on a CPU cluster using 2 CPU cores. They last between 2 and 5 minutes. Finally, the neuroevolution tasks were run on individual NVIDIA V100S and A100 GPUs. The Brax evaluations require between 30 minutes and 1.5 hours depending on the control task. The computer vision evaluation experiments take ca. 10 minutes and the MinAtar experiments last for ca. 1 hour on a V100S GPU.
