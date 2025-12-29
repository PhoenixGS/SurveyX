# Language-Conditioned Imitation Learning with Base Skill Priors under Unstructured Data
Hongkuan Zhou1, Zhenshan Bing 1 †, Xiangtong Yao1, Xiaojie Su2, Chenguang Yang3, Kai Huang4, Alois Knoll1
Abstract—The growing interest in language-conditioned robot manipulation aims to develop robots capable of understanding and executing complex tasks, with the objective of enabling robots to interpret language commands and manipulate objects accordingly. While language-conditioned approaches demonstrate impressive capabilities for addressing tasks in familiar environments, they encounter limitations in adapting to unfamiliar environment settings. In this study, we propose a general-purpose, language-conditioned approach that combines base skill priors and imitation learning under unstructured data to enhance the algorithm’s generalization in adapting to unfamiliar environments. We assess our model’s performance in both simulated and real-world environments using a zero-shot setting. The average completed task length, indicating the average number of tasks the agent can continuously complete, improves more than 2.5 times compared to the baseline method HULC. In terms of the zero-shot evaluation of our policy in a real-world setting, we set up ten tasks and achieved an average 30% improvement in our approach compared to the current state-of-the-art approach, demonstrating a high generalization capability in both simulated environments and the real world. For further details, including access to our appendix, code base, and videos, please refer to this link https://hk-zh.github.io/spil/. Index Terms—Language-conditioned Imitation Learning, Robot Manipulation
# I. INTRODUCTION
Language-conditioned robot manipulation [1] is an emerging field of research at the intersection of robotics, natural language processing, and computer vision. This domain seeks to develop robots capable of understanding their surrounding environments and executing complex manipulation tasks based on natural language commands provided by humans. Substantial progress has been made in recent years, with some studies focusing on deep reinforcement learning techniques to shape reward functions for language instructions, enabling agents to solve tasks through trial-and-error processes by following language instructions [2]–[6]. They are welldesigned to address the low sample efficiency and enable effective learning. Other researchers also leverage languageconditioned imitation learning approaches, which train agents using demonstration datasets. For instance, some studies utilize imitation learning with expert demonstrations that are accompanied by labeled language instructions to solve such language-conditioned tasks [7], [8]. While these methods have
1 Techinical University of Munich, Munich, Germany 2 Chongqing University, ChongQing, China 3 Department of Computer Science, University of Liverpool, U.K. 4 Sun Yat-sen University, Guangzhou, China †Corresponding author: Zhenshan Bing zhenshan.bing@tum.de
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9642/96426b85-2d7a-423e-9802-4b6ee599d1ab.png" style="width: 50%;"></div>
Fig. 1. Comparison of common approaches (dashed red) and our approach (green). Common approaches usually directly learn the actions, depending on current observation and instruction. Our approach aims to learn the extra intermediate-level policy of which base skill to choose, based on current observation and instruction.
demonstrated a high success rate in completing tasks, two main shortcomings still exist. Firstly, the process is limited by the substantial effort required to sample expert demonstrations. As a result, the dataset available for exploring various scenarios in the environment is restricted, ultimately hindering the agent’s potential for better performance. Secondly, the trained agent is deficient in its capacity for generalization, which impedes its ability to carry out tasks in unseen environments. To address the first problem, some researchers employ unstructured data (play data) [9]–[13], which consists of human demonstrations driven by curiosity or other intrinsic motivations, rather than being driven by specific tasks, to reduce the effort required to collect expert data for training. All the play data is obtained through interactions with simulation environments by participants using virtual reality (VR) equipment, with only 1 % of the data is labeled with language instruction. By employing play data, the labor-intensive task of data labeling is significantly reduced, facilitating the creation of larger training datasets for imitation learning. The trained agent demonstrates remarkable performance, exhibiting a high success rate across various tasks. Building upon the ideas presented in [10], HULC [14] was developed to enhance the performance of language-conditioned imitation learning by integrating transformer structures and contrastive representation learning. HULC++ [15] further improves the performance by incorporating a self-supervised visuo-lingual affordance model. Regarding the second problem, current approaches still face a challenge in generalizing to perform tasks in unfamiliar and complex environments. The policy learned through the imitation learning algorithm exhibits outstanding evaluation
performance, primarily in training domains, suggesting that the policy’s effectiveness is restricted to scenarios where training and evaluation environments are identical. Upon conducting sim2real experiments and zero-shot evaluations in novel environments, the discrepancy between the evaluation and training environments results in a substantial decline in success rates. Within the imitation learning framework, agents typically rely on predicting the short-term next action at each time step based on the current observation and goal without learning a high-level long-term procedure. This approach diverges from the more natural approach employed by humans, which typically involves breaking down complex tasks into simpler, basic steps. Skill-based learning [16], [17] is a promising approach that utilizes pre-defined skills to expedite the learning process, leveraging the prior knowledge encoded within these skills, which is typically derived from human expertise. A primary factor contributing to the suboptimal performance of current language-conditioned imitation learning methodologies is the absence of prior knowledge during the training process. The excessive dependence on training data can lead to overfitting and impede generalization to unfamiliar scenarios. By incorporating prior skills into the learning process, the agent can avoid the necessity to start from scratch and reduce the dependency of training data. In this paper, we introduce a base Skill Prior based Imitation Learning (SPIL) framework designed to enhance the generalization ability of an agent in adapting to unfamiliar environments by integrating base skill priors: translation, rotation, and grasping. Specifically, SPIL learns both a low-level policy for skill instance execution based on observations, as well as an intermediate-level policy that determines which base skill (translation, rotation, and grasping) should be performed under the current observation. Figure 1 compares our approach with normal approaches. The intermediate-level policy functions as a manager, interpreting language instructions and appropriately combining these base skills to solve manipulation tasks. For instance, when the intermediate-level policy receives the language instruction “lift the block”, it will decompose the task into several steps involving base skills, such as approaching the block (translation), grasping the block (grasping), and lifting the block (translation). Note that the reason we call it intermediate-level policy is to distinguish it from the more complex high-level policy for tasks like “tidying up the room” which can be decomposed into several subtasks (usually done by LLMs [18]). We evaluate our algorithm using the CALVIN benchmark [19] and achieve outstanding performance in the challenging zero-shot multi-environment settings. Furthermore, we conduct sim-to-real experiments to assess the performance of our approach in real-world environments, yielding outstanding results. We summarize the key contributions as follows: • In this paper, we incorporate the skill priors into imitation learning and design a skill-prior-based imitation learning mechanism to enable learning of an intermediate-level procedure and enhance the generalization ability of the learned policy. • Our proposed method exhibits superior performance compared to previous baselines, particularly in terms of its
ability to generalize and perform well in previously unseen environments. Our evaluation shows that our approach outperforms the current method HULC by a significant margin, achieving 2.5 times the performance. We conducted a series of sim-to-real experiments to investigate further our model’s generalization ability in unseen environments and the potential of our model for real-world applications.
# II. RELATED WORKS
II. RELATED WORKS
In the field of language-conditioned robot manipulation, some studies establish connections between visual perception and linguistic comprehension in the vision-and-language field, facilitating the agent’s ability to tackle multimodal problems [20]–[22]. Other research focuses on grounding language instructions and the agent’s behaviors, empowering the agent to comprehend instructions and effectively interact with the environment [23]–[26]. However, these approaches employ two-stream architectural models to process multimodal data. Such a model require distinct feature representations for each data modality, such as semantic and spatial representations [26], thus potentially compromising learning efficiency. As an alternative, end-to-end models focus on learning feature representations and decision-making directly from raw input data, where the language instructions as a conditioning factor to train the agent. This approach eliminates the need for manual feature engineering [14], [27], thereby offering a more efficient and robust solution for complex tasks and emerging as a trend in language-conditioned robot manipulations. For instance, imitation learning with end-to-end models has been applied to solve language-conditioned manipulation tasks using expert demonstrations accompanied by a large number of labeled language instructions [7] [8]. These approaches necessitate a substantial amount of labeled and structured demonstration data. By extending the idea of [9], Lynch et al. proposed MCIL [28], which grounds the agent’s behavior with language instructions using unlabeled and unstructured demonstration data, reducing data acquisition efforts and achieving more robust performance. HULC [14], as an enhanced version of MCIL, is designed to improve the performance of MCIL even further. It has achieved impressive results in the CALVIN benchmark [19] using the single environment setting. However, when tested in the more challenging Zeroshot Multi Environment setting, where the evaluation environment is not exactly the same as the training environments, HULC’s performance drops significantly. These suboptimal results suggest that current language-conditioned imitation learning approaches lack the ability to adapt to unfamiliar environments. More recently, some approaches [29]–[32] leverage rich knowledge in the pre-trained foundation models to enhance the generalization ability in unseen environments. The concept of skill-based mechanisms in deep reinforcement learning provides valuable insights for enhancing the generalizability of algorithms. Specifically, skill-based reinforcement learning leverages task-agnostic experiences in the form of large datasets to accelerate the learning process [33]– [36]. To extract skills from a large task-agnostic dataset,
several approaches [37], [38] first learn an embedding space of skills and skill priors from the dataset. Inspired by this, we have developed an imitation learning approach that utilizes certain base skill priors. By employing this method, the agent learns intermediate-level processes (composing these base skills) that aid in task completion, thereby enhancing its ability to generalize across different scenarios.
III. METHODOLOGY
# A. Overview
The key idea of our approach is integrating skills into imitation learning by changing the original action space Cartesian End Effector space A ∈ R7 into skill space Askill ∈RNh×7, where Nh indicates the horizon of skills. Note that each skill represents a fixed-length (Nh) action sequence in our setting. Also, we intend to integrate the concept of base skills (translation, rotation, grasping) into the learning procedure so that the agent can learn an extra intermediatelevel policy to decompose tasks into several base skills. Unlike reinforcement learning, the optimization strategy employed in imitation learning involves minimizing the discrepancy between the predicted actions and the corresponding actions observed in the demonstration data. For this reason, a primary challenge in integrating skill priors into imitation learning is the continuous nature of actions in the demonstration data, which requires modeling the skills as a continuous action space to align with the demonstration actions rather than representing the skills by a finite, discrete set of pre-defined action sequences. To address the challenges mentioned above, the rest of this section is organized as follows: 1) We define three base skills (translation, rotation, grasping) for a robotic arm agent and introduce the method to stochastically label action sequences with base skills. 2) We introduce our approach to learning continuous skill embedding space, integrating base skill priors into such skill space. 3) By utilizing a continuous skill space and base skills, we implement an imitation learning algorithm to train the agent to acquire the ability to 1) learn an intermediatelevel base skill composition to accomplish the desired task and 2) develop a policy that can determine which specific skill instance to perform based on each observation, as opposed to a single action. The architecture of our proposed method is illustrated in Figure 4.
The architecture of our proposed method is illustrated in Figure 4.
# B. Base Skill Labeling
This section formally defines three base skills - translation, rotation, and grasping. Since each action sequence can contain multiple base skills, deterministically classifying an action sequence to one of three base skills is not reasonable. Here, we stochastically label each given action sequence x = (a0, a1, ..., aNh−1) of length Nh with probability (p(trans.|x), p(rot.|x), p(grasp.|x)) which indicate the probability of x belongs to these three base skills. For example,
the probability of (0.7, 0.2, 0.1) suggests a dominance of translation skill within the given action sequence, a minor presence of rotation skill, and a minimal grasping skill. We design a non-learning-based approach to label each action sequence. Since the action is defined in the Cartesian EE space, it can be accomplished by assessing the accumulated magnitude of seven degrees of freedom within the temporal dimension of a given horizon Nh. The probability of this sequence belonging to translation, rotation, and grasping skills can be defined as follows:
(1)
  �  �   where y ∈{trans., rot., grasp.} refers to base skills and a = [tx, ty, tz, rα, rβ, rγ, g] with atrans. = [tx, ty, tz], arot. = [rα, rβ, rγ], and agrasp. = [g], indicating the end effector’s displacement, rotation, and gripper control. The “magic weight” wy is introduced to address inconsistencies in scale across different units like meters and degrees. These values act as balancing factors and are determined based on our understanding of the inherent relationships between translation, rotation, and grasping. They reflect the subjective nature of defining translation, rotation, and grasping. Since these classifications may be nuanced and depend on human experience, we’ve chosen ’magic weight’ wk that reflects a common understanding of how these motions are typically defined.
# C. Continuous Skill Embeddings with Base Skill Priors
In this section, we introduce a skill space Askill ∈RNh×7 as the action space for the agent. To better represent such skill space, we compress the action sequences into skill embeddings by following the idea of Variantial AutoEncoders (VAEs), leveraging the action sequences sampled from play data. After training, we acquire a latent space full of skill embeddings and three clusters, indicating the base skills priors for translation, rotation, and grasping. To achieve this, we define y as the variable for base skills and the base skill distribution in the latent space can be written as z ∼p(z|y). For the given action sequence x, we employ the approximate variational posterior q(z|x) and q(y, z|x) to estimate the intractable true posterior. Following the VAEs procedure, we measure the KullbackLeibler (KL) divergence between the true posterior and the posterior approximation to determine the ELBO (the details can be seen in Appendix Theoretical Motivation):
(2)
where pθ(x|y, z) and qϕ(z|x) are the decoder and encoder networks with parameters θ and ϕ, respectively. We also define a network pκ(z|y) with parameters κ for locating the base skills in the latent skill space. q(y = k|x) is calculated by Equation (1). The hyperparameters β1 and β2 are introduced to weigh the regularizer terms. LELBO can be interpreted
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0deb/0deb4907-7764-410c-a38e-380b2ab0b159.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 2. This architecture comprises two encoders - the action sequence encoder and the base skill locator (encoder), and a decoder fo reconstructing the skill embeddings into action sequences. The base skill locator takes one-hot-key embeddings of translation, rotation, an grasping as input and outputs the distribution of the base skill prior in the skill latent space. The action sequence encoder encodes the actio sequences with a fixed horizon of Nh to the skill distribution in the latent space. The decoder then reconstructs the skill embedding int action sequences.</div>
as follows. On the one hand, we intend to achieve higher reconstruction accuracy. As the reconstruction improves, our approximated posterior will also become more accurate. On the other hand, the two introduced regularizers contribute to a more structured latent skill space. The first regularizer, DKL(qϕ(z|x)||p(z)), constrains the encoded distribution to be close to the prior distribution p(z). Likewise, the second regularizer, DKL(qϕ(z|x)||pκ(z|y)), draws the encoded distribution nearer to the prior distribution of its corresponding base skill class. The learning procedure is illustrated in Figure 2 and the overall algorithm can be found in Algorithm 1. After training, we obtain a skill generator fθ = pθ(x|z), which maps the skill embedding to the corresponding action sequence. Since there exists such a one-to-one mapping relationship, the action space Askill is equivalent to Az ∈RNz, where Nz is the skill embedding dimension. The agent should select one skill embedding in the latent space at each timestep rather than one action sequence that we typically consider. Additionally, we have the base skill locator fκ = pκ(z|y) to identify the position of base skill distributions within the skill latent space. Their parameters are frozen during the later imitation learning process. A visualization of the skill latent space helps with understanding. An illustration of the skill latent space by performing the t-SNE algorithm can be found in Figure 3. As the figure demonstrates, three clusters are labeled with different colors, indicating three base skills we define. Each point indicates a skill embedding z ∈Az that corresponds to an action sequence with the length of Nh. A single skill embedding could encompass various base skill features, given that the skill’s latent space is continuous. Consequently, a skill embedding between two base skill clusters would encompass features from both of these base skills.
# D. Imitation Learning with Base Skill Priors
After acquiring the skill embedding space Az and the distributions of base skill priors in such latent space, we can train a policy using imitation learning based on that. This approach results in a policy with enhanced generalization
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3478/34781d21-a93d-4b31-a68a-361aa9f42377.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 3. t-SNE visualization of skill latent space.</div>
capabilities, as incorporating prior knowledge prevents the model from overfitting. The base skill priors we have defined encapsulate human proficiency in task completion. We aim to leverage the prior knowledge contained in base skills to reduce the agent’s reliance solely on training data. In our approach, the agent learns to choose a skill that embodies motion-related human knowledge instead of determining the action at every step. Meanwhile, it also selects the appropriate base skill for the current state, mirroring the habitual approach of humans in accomplishing tasks. We extend the idea of MCIL [10] and HULC [14] by employing an action space Az comprising skill embeddings instead of Cartesian action space A. In this framework, the action performed by the agent is no longer a single 7 DoF movement in one time step, but instead, a skill (action sequence) over a horizon Nh. Consequently, the agent learns to select a skill based on the current observation. After the skill is performed, the agent selects the next skill based on the subsequent observation, and the process continues iteratively until the agent completes the task or the time runs out. Figure 4 depicts the overall structure of our approach. Given the superior performance of the HULC model, we employ its encoder, denoted as fΦ, to transform the static observation, gripper
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/90cb/90cba064-0cb6-4d32-96ca-1cef0b0a4ffd.png" style="width: 50%;"></div>
<div style="text-align: center;">plan emb. lang. goal emb. static obs. emb. gripper obs. emb. language emb.</div>
Fig. 4. The Overall Architecture. Following the encoding process, the static observation, gripper observation, and language instruction are generated to embeddings for the plan, language goal, language, static observation, and gripper observation. The skill selector module subsequently decodes a sequence of skill embeddings using the plan, observation, and language goal embeddings. The skill labeler labels the skill embeddings with the base skills: translation, rotation, and grasping. The base skill regularization loss is calculated based on the base skill prior distributions (from base skill locator fκ), selected skill instance, and labeled probability indicating its belonging to specific base skills. This labeled probability is also leveraged to determine the categorical regularization loss. Finally, the pre-trained and frozen skill generator fθ decodes all the skill embeddings into action sequences, which are then utilized to calculate the reconstruction loss (Huber loss).
Algorithm 1 Learning Continuous Skill Embeddings with
Base Skill Priors
 Given:
1: Given:
• D : {(a0, a1, ..., aH−1)}: A Play dataset full of action
sequences with horizon H.
• F = {fϕ, fθ, fκ}. They are the encoder network with
parameters ϕ, the decoder network, also denoted as
skill generator network with parameters θ, and the
base skill locator network with parameters κ.
2: Randomly initialize model parameters {θ, ϕ, κ}
3: while not done do
4:
Sample an action sequence x ∼D
5:
Encode this sequence with fϕ = qϕ(z|x)
6:
Compute the base skill distributions fκ = pκ(z|y).
7:
Sample one latent embedding z ∼qϕ(z|x)
8:
Feed the sampled z into the decoder fθ = pθ(x|z) to
get the reconstructed action sequence ˆx
9:
Compute the loss based on Equation (2)
10:
Update parameters θ, ϕ, κ to minimize L
11: end while
observation, and language instruction into their corresponding embeddings. All these embeddings align with the definition provided in the HULC model. Additionally, to extract the overall process information from the language instruction, we introduce extra language embedding. This process information is crucial for inferring the intermediate-level compositions of base skills required for successful task completion. We further analyze four key parts in our structure: • Skill Embedding Selector: The skill embedding selector, denoted as fλ, selects skill embeddings in the pre-trained
• Skill Embedding Selector: The skill embedding selector, denoted as fλ, selects skill embeddings in the pre-trained
latent space. A bidirectional LSTM network is employed for this skill embedding selector. • Base Skill Selector: The base skill selector fω, also a bidirectional LSTM network, determines the base skill to which a given skill belongs. • Base Skill Locator: The base skill locator shares the same parameters with base skill locator fκ in Figure 2. It has the task of locating the base skill locations in the latent space. The input to this network is a 3 × 3 identity matrix, signifying the one-hot representing of three base skills. These locations are used to calculate the regularization loss. • Skill Generator: The skill generator, denoted as fθ = pθ(x|z) : Az →Askill, shares the same parameters with the decoder component in Figure 2. Its parameters are frozen during the imitation learning process. Its function is to transform space from skill embedding space Az to skill space Askill. These skills (action sequences) are combined chronologically for a longer action sequence. The objective of our model is to learn a policy π(x|sc, sg) conditioned on the current state sc and the goal state sg and outputting x, a sequence of actions, namely a skill. Since we ntroduced the base skill concept into our model, the policy π(·) should also find the best base skill y for the current observation. We have π(x, y|sc, sg), where y is the base skill he agent chooses based on the current state and goal state. nspired by the conditional variational autoencoder (CVAE): log p(x|c) ≥Eq(z|x,c)[log p(x|z, c)] −DKL(q(z|x, c)||p(z|c)) (3) where c is a symbol to describe a general condition, we would ike to extend the above equation by integrating y which ndicates the base skill. The evidence we want to maximize
(3)
1: Given:
1: Given: • D : {(Dplay, Dlang)}: Play Dataset and Language Dataset • F = {fΦ, fλ, fκ, fω, fθ}. They are the encoder fΦ, the skill embedding selecter fλ, the base skill locator fκ, the base skill selector fω, the skill generator networks fθ with parameters Φ, λ, κ, ω, and θ, respectively. 2: Randomly initialize model parameters {Φ, λ, ω} 3: Initialize parameters θ and κ with pre-trained skill generator and base skill locator 4: Freeze the parameters θ and κ. 5: while not done do 6: L ←0 7: for l in {play, lang} do 8: Sample a (demonstration, context) (xl, cl) ∼Dl 9: Encode the observation, goal, and plan embeddings, using the encoder network fΦ 10: Skill Embedding Selecter fλ selects the skill embedding sequence 11: Determinate a sequence of base skill probabilities with Base Skill Selector fω. 12: Determinate base skill locations in the latent space with Base Skill Locator fκ 13: Skill Generator fθ maps the skill embeddings to action sequences. 14: Calculate the loss function Ll according to (4) 15: Accumulate imitation loss L += Ll 16: end for 17: update parameters {Φ, λ, ω} w.r.t L 18: end while
then turns to p(x, y|c). We employ the approximate variational posterior q(y, z|x, c) to approximate the intractable true posterior p(y, z|x, c) where z indicates the skill embeddings in the skill latent space. We intend to find the ELBO by measuring the KL divergence between the true posterior and the posterior approximation (detailed theoretical motivation in the Appendix). We have
(4)
− � �� � ||| where c represents a combination of the current state and the goal state (sc, sg). z is skill embedding in the latent skill space. pθ(x|z) is the skill generator network fθ with parameters θ and it is trained by VAEs discussed in the previous session and frozen during the imitation learning. fω = qω(y|c) corresponds to the skill labeller with parameter ω. qΦ,λ(z|x, c) refers to the encoder network fΦ plus the skill embedding selector network fλ. Furthermore, pκ(z|y) constitutes the base
skill prior locater fκ with parameter κ. It is also trained by VAEs, as discussed in the previous section and frozen during the training process. Here, we use Huber loss as the metric for reconstructive loss. Intuitively, the base skill regularizer is used to regularize a skill embedding, depending on its base skill category. The categorial regularizer aims to regularize the base skill classification based on the prior categorical distribution of y. The overall algorithm can be seen in Algorithm 2.
# IV. EXPERIMENTS
In this section, we present the experiments conducted to investigate the generalization ability of our model in comparison to other baselines. We choose the CALVIN [19] benchmark to evaluate our model. The CALVIN benchmark is introduced to facilitate learning language-conditioned tasks across four manipulation environments: A, B, C, and D. Each environment features a Franka Emika Panda robot arm equipped with a gripper and a desk that includes a sliding door and a drawer. Additionally, the desk has a button that can toggle a green light and a switch to control a light bulb. Note that each environment has a different desk with various of textures and the position of static elements such as the sliding door, drawer, light, switch, and button are different across each environment. Experiments are conducted in two settings: (1) a single environment where the training and testing environments are the same, and (2) zero-shot multi-environments where training occurs in the first three environments and testing takes place in the fourth, previously unseen environment. We choose long-horizon multi-task language control (LHMTLC) to evaluate the effectiveness of the learned multitask language-conditioned policy in accomplishing several language instructions in a row under the zero-shot multi environment. We also compare other skill-based reinforcement learning approaches to show the advantages of our approaches against theirs. We analyze the result of our model by comparing it to other baselines (shown in Table I). We evaluate the models with 1000 five-task chains. The columns labeled from one to five demonstrate the success rate of continuously completing that number of tasks in a row. The average length indicates the average number of tasks the agent can continuously complete when given five tasks in a row (The remaining tasks are not performed if one task fails in the middle). Subsequently, ablation studies on hyperparameters γ1,γ2 in (4) and the length of skill Nh (the default value is 5) are performed in the zeroshot multi-environment. Each model is evaluated three times across 3 random seeds.
# A. Environment Result
As evidenced in Table I, our model substantially improves compared to our baselines HULC and MCIL in a zero-shot multi-environment setting. Compared to the current SOTA model HULC, the success rate of completing one to five tasks in a row has increased by 32.4% , 29.8%, 21.9 %, 12.8%, and 6.9 %, respectively. The overall average length increased from 0.67 to 1.71. Note that zero-shot multi-environment presents a challenging environment, as the agent must solve tasks in an unfamiliar environment. The performance in this setting
Environment
Method
Train →Test
LH-MTLC
No. Instructions in a Row (1000 chains)
1
2
3
4
5
Avg. Len.
Zero-shot Multi
Environment
MCIL
A,B,C →D
30.4%
1.3%
0.17 %
0%
0%
0.31
HULC
A,B,C →D
41.8% (2.3)
16.5% (2.5)
5.7% (1.3)
1.9% (0.9)
1.1% (0.5)
0.67 (0.1)
SPIL (Ours)
A,B,C →D
74.2% (1.4)
46.3% (3.4)
27.6% (3.4)
14.7% (2.3)
8.0% (1.7)
1.71 (0.11)
3D Diffuser Actor*†
A,B,C →D
92.2 %
78.7 %
63.9 %
51.2 %
41.2 %
3.27
GR-1*†
A,B,C →D
85.4 %
71.2 %
59.6 %
49.7 %
40.1 %
3.06
SuSIE*
A,B,C →D
87.0 %
69.0 %
49.0 %
38.0 %
26.0 %
2.69
RoboFlamingo*
A,B,C →D
82.4 %
61.9 %
46.6 %
33.1 %
23.5 %
2.47
γ1 = 1.0 × 10−2
A,B,C →D
71.3% (1.4)
45.8% (3.8)
25.4% (2.3)
13.1% (0.9)
6.5% (0.5)
1.62 (0.05)
γ2 = 1.0 × 10−4
A,B,C →D
70.6% (4.2)
46.3% (3.2)
25.1% (3.0)
14.1% (1.0)
7.3% (1.3)
1.63 (0.08)
Nh = 4
A,B,C →D
71.4% (2.1)
41,0% (3,1)
24.1% (1,2)
12.1% (1.1)
7.4% (0.6)
1.58 (0.07)
Nh = 6
A,B,C →D
74.0% (1.6)
44.2% (2.6)
25.2% (2.0)
13.0% (2.4)
7.9% (1.7)
1.65 (0.08)
w/o base skills
A,B,C →D
57.5% (1.8)
27.9% (2.1)
12.2% (1.2)
5.0% (1.1)
2.2% (0.7)
1.05 (0.06)
Single
Environment
MCIL
D →D
76.4% (1.5)
48.8% (4.1)
30.1% (4.5)
18.1% (3.0)
9.3% (3.5)
1.82 (0.2)
HULC
D →D
82.7% (0.3)
64.9% (1.7)
50.4% (1.5)
38.5% (1.9)
28.3% (1.8)
2.64 (0.05)
SPIL (Ours)
D →D
84.6% (0.6)
65.1% (1.3)
50.8% (0.4)
38.0% (0.6)
28.6% (0.3)
2.67 (0.01)
* indicates that the model leverages pre-trained foundation models. † means that the model leverages extra proprioceptive state as input
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/74c9/74c9a537-e040-4134-a178-7ec0612609c9.png" style="width: 50%;"></div>
<div style="text-align: center;">e employ the multi-task language control (MTLC) setting in the CALVIN benchmark, en  the simulated CALVIN environment D and directly applied to the real-world setting.</div>
<div style="text-align: center;">Fig. 5. Real-world experiments. We employ the multi-task language control (MTLC) setting in the CALVIN benchmark, encompassing a total of 10 tasks as listed above. The agent is trained in the simulated CALVIN environment D and directly applied to the real-world setting.</div>
represents the agent’s ability to truly understand and connect the concepts in language instructions with real objects and actions. The performance of our model demonstrates a significant improvement, thus confirming our hypothesis that using skill priors to learn intermediate-level task composition can improve generalization capabilities. The other SOTA models - SuSIE [29], RoboFlamingo [30], GR-1 [31], 3D Diffuser Actor [32], which leverage pre-trained foundation models, as listed in Table I for reference. It is worth mentioning that our SPIL model also outperforms the baselines in the single environment setting.
# B. Real-world Experiments
To investigate the viability of the policy trained in a simulated environment to real-world scenarios, we conduct a sim2real experiment without any additional specific adaptation (zero-shot), as shown in Figure 5. We designed the real-world environment to closely resemble the simulated CALVIN environment D. The rightmost part of Figure 4 illustrates that the real-world environment comprises one switch, one cabinet with a slider, one button, one drawer, and three blocks in red, pink, and blue colors. Additionally, two RGB cameras are employed to capture the static observation and gripper observations.
<div style="text-align: center;">TABLE II REAL-WORLD EXPERIMENT RESULTS</div>
Tasks
HULC
SPIL
Task
HULC
SPIL
open drawer
0%
30%
move slider right
0%
30%
close drawer
0%
40%
push button
10%
50%
toggle switch on
10%
40%
lift red block
0%
20%
toggle switch off
10%
30%
lift blue block
0%
20%
move slider left
0%
40%
lift pink block
0%
30%
Average:
HULC (3%)
SPIL (33%)
Table II lists the tasks performed and the corresponding success rate. The agent is trained in four CALVIN environments (A, B, C, D), and the trained policy is directly applied to a real-world environment. To mitigate the influence of the robot’s initial position on the policies, we execute 10 rollouts for each task, maintaining identical starting positions. The table results demonstrate our model’s effectiveness in handling the challenging zero-shot sim2real experiments. Despite the substantial differences between the simulation and real-world contexts, our model still achieves an average success rate of 33% in accomplishing the tasks. Conversely, the HULC model-trained agent struggles with these tasks, with a 3% average success rate, underscoring the difficulty of solving
real-world challenges. The results from real-world experiments further substantiate our claim that our proposed method exhibits superior generalization capabilities, enabling successful task completion even in unfamiliar environments.
# V. CONCLUSION
In this paper, we introduced a novel imitation learning paradigm that integrates base skills into imitation learning. Our proposed SPIL model effectively improves the generalization ability compared to current baselines and substantially surpasses the SOTA models on the language-conditioned robotic manipulation CALVIN benchmark, especially under the challenging zero-shot multi environment setting. This work also aims to contribute towards the development of general-purpose robots that can effectively integrate human language with their perception and actions.
# REFERENCES
[1] H. Zhou et al., “Language-conditioned learning for robotic manipulation: A survey,” 2024. [Online]. Available: https://arxiv.org/abs/2312.10807 [2] D. Bahdanau, F. Hill, J. Leike, E. Hughes, P. Kohli, and E. Grefenstette, “Learning to follow language instructions with adversarial reward induction,” CoRR, vol. abs/1806.01946, 2018. [3] S. Nair, E. Mitchell, K. Chen, I. Brian, S. Savarese, and C. Finn, “Learning language-conditioned robot behavior from offline data and crowd-sourced annotation,” in Proceedings of the 5th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, A. Faust, D. Hsu, and G. Neumann, Eds., vol. 164. PMLR, 08–11 Nov 2022, pp. 1303–1315. [Online]. Available: https://proceedings.mlr.press/v164/nair22a.html [4] P. Goyal, S. Niekum, and R. Mooney, “Pixl2r: Guiding reinforcement learning using natural language by mapping pixels to rewards,” in Proceedings of the 2020 Conference on Robot Learning, ser. Proceedings of Machine Learning Research, J. Kober, F. Ramos, and C. Tomlin, Eds., vol. 155. PMLR, 16–18 Nov 2021, pp. 485–497. [Online]. Available: https://proceedings.mlr.press/v155/goyal21a.html [5] Z. Bing, A. Koch, X. Yao, K. Huang, and A. Knoll, “Meta-reinforcement learning via language instructions,” in Proceedings of the IEEE International Conference on Robotics and Automation, London, UK, 2023. [6] X. Yao et al., “Learning from symmetry: Meta-reinforcement learning with symmetrical behaviors and language instructions,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023, pp. 5574–5581. [7] S. Stepputtis, J. Campbell, M. Phielipp, S. Lee, C. Baral, and H. Ben Amor, “Language-conditioned imitation learning for robot manipulation tasks,” Advances in Neural Information Processing Systems, vol. 33, pp. 13 139–13 150, 2020. [8] E. Jang et al., “BC-z: Zero-shot task generalization with robotic imitation learning,” in 5th Annual Conference on Robot Learning, 2021. [Online]. Available: https://openreview.net/forum?id=8kbp23tSGYv [9] C. Lynch et al., “Learning latent plans from play,” Conference on Robot Learning (CoRL), 2019. [10] C. Lynch and P. Sermanet, “Language conditioned imitation learning over unstructured data,” Robotics: Science and Systems, 2021. [Online]. Available: https://arxiv.org/abs/2005.07648 [11] E. Rosete-Beas, O. Mees, G. Kalweit, J. Boedecker, and W. Burgard, “Latent plans for task-agnostic offline reinforcement learning,” 2022. [Online]. Available: https://arxiv.org/abs/2209.08959 [12] Z. J. Cui, Y. Wang, N. M. M. Shafiullah, and L. Pinto, “From play to policy: Conditional behavior generation from uncurated robot data,” arXiv preprint arXiv:2210.10047, 2022. [13] J. Borja-Diaz, O. Mees, G. Kalweit, L. Hermann, J. Boedecker, and W. Burgard, “Affordance learning from play for sample-efficient policy learning,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 6372–6378. [14] O. Mees, L. Hermann, and W. Burgard, “What matters in language conditioned robotic imitation learning over unstructured data,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 11 205–11 212, 2022. [15] O. Mees, J. Borja-Diaz, and W. Burgard, “Grounding language with visual affordances over unstructured data,” 2023.
[16] L. X. Shi, J. J. Lim, and Y. Lee, “Skill-based model-based reinforcement learning,” in 6th Annual Conference on Robot Learning, 2022. [17] A. Nagabandi, K. Konolige, S. Levine, and V. Kumar, “Deep dynamics models for learning dexterous manipulation,” in Conference on Robot Learning. PMLR, 2020, pp. 1101–1112. [18] Wu et al., “Tidybot: Personalized robot assistance with large language models,” Autonomous Robots, 2023. [19] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard, “Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks,” IEEE Robotics and Automation Letters (RAL), vol. 7, no. 3, pp. 7327–7334, 2022. [20] J. Pont-Tuset, J. Uijlings, S. Changpinyo, R. Soricut, and V. Ferrari, “Connecting vision and language with localized narratives,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16. Springer, 2020, pp. 647–664. [21] J. Lu, D. Batra, D. Parikh, and S. Lee, “Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks,” Advances in neural information processing systems, vol. 32, 2019. [22] L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, and K.-W. Chang, “What does BERT with vision look at?” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online: Association for Computational Linguistics, Jul. 2020, pp. 5265–5275. [Online]. Available: https://aclanthology.org/2020.acl-main.469 [23] M. Shridhar, D. Mittal, and D. Hsu, “Ingress: Interactive visual grounding of referring expressions,” The International Journal of Robotics Research, vol. 39, no. 2-3, pp. 217–232, 2020. [24] A. Magassouba, K. Sugiura, A. T. Quoc, and H. Kawai, “Understanding natural language instructions for fetching daily objects using gan-based multimodal target–source classification,” IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3884–3891, 2019. [25] W. Liu, C. Paxton, T. Hermans, and D. Fox, “Structformer: Learning spatial structure for language-guided semantic rearrangement of novel objects,” in 2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022, pp. 6322–6329. [26] M. Shridhar, L. Manuelli, and D. Fox, “Cliport: What and where pathways for robotic manipulation,” in Conference on Robot Learning. PMLR, 2022, pp. 894–906. [27] Co-Reyes et al., “Guiding policies with language via meta-learning,” in International Conference on Learning Representations, 2018. [28] C. Lynch and P. Sermanet, “Language conditioned imitation learning over unstructured data,” arXiv preprint arXiv:2005.07648, 2020. [29] K. Black, M. Nakamoto, P. Atreya, H. Walke, C. Finn, A. Kumar, and S. Levine, “Zero-shot robotic manipulation with pretrained image-editing diffusion models,” 2023. [Online]. Available: https: //arxiv.org/abs/2310.10639 [30] X. Li, M. Liu, H. Zhang, C. Yu, J. Xu, H. Wu, C. Cheang, Y. Jing, W. Zhang, H. Liu et al., “Vision-language foundation models as effective robot imitators,” arXiv preprint arXiv:2311.01378, 2023. [31] H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H. Li, and T. Kong, “Unleashing large-scale video generative pretraining for visual robot manipulation,” 2023. [Online]. Available: https://arxiv.org/abs/2312.13139 [32] T.-W. Ke, N. Gkanatsios, and K. Fragkiadaki, “3d diffuser actor: Policy diffusion with 3d scene representations,” 2024. [Online]. Available: https://arxiv.org/abs/2402.10885 [33] K. Hausman, J. T. Springenberg, Z. Wang, N. Heess, and M. Riedmiller, “Learning an embedding space for transferable robot skills,” in International Conference on Learning Representations, 2018. [Online]. Available: https://openreview.net/forum?id=rk07ZXZRb [34] J. Merel et al., “Neural probabilistic motor primitives for humanoid control,” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=BJl6TjRcY7 [35] T. Kipf, Y. Li, H. Dai, V. F. Zambaldi, A. Sanchez-Gonzalez, E. Grefenstette, P. Kohli, and P. W. Battaglia, “CompILE: Compositional Imitation Learning and Execution,” in Proceedings of the 36th International Conference on Machine Learning. PMLR, 2019, pp. 3418–3428. [36] Y. Lee, J. Yang, and J. J. Lim, “Learning to coordinate manipulation skills via skill behavior diversification,” in International Conference on Learning Representations, 2020. [Online]. Available: https: //openreview.net/forum?id=ryxB2lBtvH [37] K. Pertsch, Y. Lee, Y. Wu, and J. J. Lim, “Demonstration-guided reinforcement learning with learned skills,” 5th Conference on Robot Learning, 2021. [38] K. Pertsch, Y. Lee, and J. J. Lim, “Accelerating reinforcement learning with learned skill priors,” in Conference on Robot Learning (CoRL), 2020.
