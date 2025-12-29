# Dynamics Models in Object-Oriented Environm
# Learning Causal Dynamics Models in Object-Oriented Environments
# Zhongwei Yu 1 Jingqing Ruan 1 Dengpeng Xing 1 2
Zhongwei Yu 1 Jingqing Ruan 1 Dengpeng Xing 1 2
# Abstract
Causal dynamics models (CDMs) have demonstrated significant potential in addressing various challenges in reinforcement learning. To learn CDMs, recent studies have performed causal discovery to capture the causal dependencies among environmental variables. However, the learning of CDMs is still confined to small-scale environments due to computational complexity and sample efficiency constraints. This paper aims to extend CDMs to large-scale object-oriented environments, which consist of a multitude of objects classified into different categories. We introduce the Object-Oriented CDM (OOCDM) that shares causalities and parameters among objects belonging to the same class. Furthermore, we propose a learning method for OOCDM that enables it to adapt to a varying number of objects. Experiments on large-scale tasks indicate that OOCDM outperforms existing CDMs in terms of causal discovery, prediction accuracy, generalization, and computational efficiency.
arXiv:2405.12615v1
# 1. Introduction
Reinforcement learning (RL) (Sutton & Barto, 2018) and causal inference (Pearl, 2000) have separately made much progress over the past decades. Recently, the combination of both fields has led to a series of successes (Zeng et al., 2023), where the use of causal dynamics models (CDMs) proves a promising direction. CDMs capture the causal structures of environmental dynamics and have been applied to address a wide range of challenges in RL, including learning efficiency, explainability, generalization, state representation, subtask decomposition, and transfer learning (see Section 2.1). For example, a major function of CDMs is to reduce spurious correlations (Ding et al., 2022; Wang
1Institute of Automation, Chinese Academy of Sciences, Beijing, P. R. China 2School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing, P. R. China. Correspondence to: Dengpeng Xing <dengpeng.xing@ia.ac.cn>.
Proceedings of the 41 st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).
Proceedings of the 41 st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).
et al., 2022), which are particularly prevalent in the non-i.i.d. data produced in sequential decision-making.
Early research of CDMs exploits given causal structures of environments (Boutilier et al., 2000; Guestrin et al., 2003b; Madumal et al., 2020b), which may not be available in many applications. Therefore, some recent studies have proposed to develop CDMs using causal discovery techniques to learn such causal structures, i.e. causal graphs (CGs), from the data of history interactions (Volodin, 2021; Wang et al., 2021; 2022; Zhu et al., 2022). These approaches have been successful in relatively small environments consisting of a few variables. Unfortunately, some RL tasks involve many objects (e.g., multiple agents and environment entities in multi-agent domains (Malysheva et al., 2019)), which together contribute to a large set of environment variables. The applicability of CDMs in such large-scale environments remains questionable — the excessive number of potential causal dependencies (i.e., edges in CGs) makes causal discovery extremely expensive, and more samples and effort are required to correctly discriminate causal dependencies. Interestingly, humans can efficiently reason causality from enormous real-world information. One possible explanation for this is that we intuitively perceive tasks through an object-oriented (OO) perspective (Hadar & Leron, 2008) — we decompose the world into objects and categorize them into classes, allowing us to summarize and share causal rules for each class. For example, “exercise causes good health of each person” is a shared rule of the class “Human”, and “each person” represents any instance of that class. This OO intuition has been widely adopted in modern programming languages, referred to as object-oriented programming (OOP), to organize and manipulate data in a more methodical and readable fashion (Stroustrup, 1988). This work aims to extend CDMs to large-scale OO environments. Inspired by OOP, we investigate how an OO description of the environment can be exploited to facilitate causal discovery and dynamics learning. We propose the Object-Oriented Causal Dynamics Model (OOCDM), a novel type of CDM that allows the sharing of causalities and model parameters among objects. To learn the OOCDM, we present a modified version of Causal Dynamics Learning (CDL) (Wang et al., 2022) that can accommodate varying numbers of objects. We theoretically prove
that the proposed approach discovers the ground-truth CG under a few natural assumptions. Additionally, we apply OOCDM to several OO domains and demonstrate that it outperforms state-of-the-art CDMs in terms of causal graph accuracy, prediction accuracy, generalization ability, and computational efficiency, especially for large-scale tasks. To the best of our knowledge, OOCDM is the first dynamics model to combine causality with the object-oriented settings in RL. The source code of this work is made available at https://github.com/EaseOnway/oocdm.
# 2. Related works
# 2.1. Causality and Reinforcement Learning
2.1. Causality and Reinforcement Learning
Causality (see basics in Appendix B) formulates dependencies among random variables and is used across various disciplines (Pearl, 2000; Pearl et al., 2016; Pearl & Mackenzie, 2019). One direction to combine causality with RL is to formulate a known causal structure among macro elements (e.g., the observation, action, reward, and hidden states) of the Markov Decision Process (MDP). Algorithms with improved robustness and efficiency are then derived by applying the causal inference techniques (Buesing et al., 2018; Lu et al., 2018; Zhang et al., 2020; Liao et al., 2021; Guo et al., 2022).
This paper follows another direction on the micro causality that exists among specific components of the environment. Modular models prove capable of capturing such causality using independent sub-modules, leading to better generalization and learning performance (Ke et al., 2021; Mittal et al., 2020; 2022). A popular setting for the micro causality is Factored MDP (FMDP) (Boutilier et al., 2000), where the transition dynamics is modeled by a CDM. Knowledge to this CDM benefits RL in many ways, including 1) efficiently solving optimal policies (Guestrin et al., 2003b; Osband & Van Roy, 2014; Xu & Tewari, 2020), 2) sub-task decomposition (Jonsson & Barto, 2006; Peng et al., 2022), 3) improving explainability (Madumal et al., 2020a;b; Volodin, 2021; Yu et al., 2023), 4) improving generalization of policies (Nair et al., 2019) and dynamic models (Ding et al., 2022; Wang et al., 2022; Zhu et al., 2022), 5) learning taskirrelevant state representations (Wang et al., 2021; 2022), 6) policy transfer to unseen domains (Huang et al., 2022). Additionally, Feng et al. (2022) presents an extension of FMDP that has non-stationary CDMs.
# 2.2. Object-Oriented Reinforcement Learning
It is common in RL to describe environments using multiple objects. Researchers have largely explored object-centric representation (OCR), especially in visual domains, to facilitate policy learning (Zambaldi et al., 2018; Zadaianchuk et al., 2020; Zhou et al., 2022; Yoon et al., 2023) or dy-
namic modeling (Zhu et al., 2018; 2019; Kipf et al., 2020; Locatello et al., 2020). However, OCR typically uses homogeneous representations of objects and struggles to capture the diverse nature of objects. Goyal et al. (2020; 2022) overcome this problem by extracting a set of dynamics templates (called schemata or rules) that are matched with objects to predict next states. Prior to our work, Guestrin et al. (2003a) and Diuk et al. (2008) investigated OOP-style MDP representations using predefined classes of objects.
# 2.3. Relational Causal Discovery
It is common to use the structural priors to improve the efficiency of causal discovery. Similar to this work, relational causal discovery (Maier et al., 2010) attempts to modularize causal dependencies using additional knowledge about objects in relation domains. Marazopoulou et al. (2015) further provides a temporal extension of the technique for sequential data. However, relational causal discovery requires stronger priors than our work – not only the description of classes and objects but also the formulation of inter-object relations. Our work focuses on the FMDP settings where relations are implicit and unknown, which may contribute to more general use. In addition, relational causal discovery does not include a good dynamics model that fully exploits the object-oriented priors.
# 3. Preliminaries
Notations A random variable is denoted by a capital letter (e.g., X1 and X2). Brackets may combine variables or subgroups into a group (an ordered set) denoted by a bold letter, e.g. X = (X1,X2) and Z = (X,Y1,Y2). We use p to denote a distribution. In addition, Appendix A provides a thorough list of notations in this paper.
# 3.1. Causal Dynamics Models
We consider the FMDP setting where the state and action consist of multiple random variables, denoted as S = (S1,⋯,Sns) and A = (A1,⋯,Ana), respectively. S′ i (or S′) denotes the state variable(s) in the next step. The transition probability p(S′∣S,A) is modeled by a CDM (see Definition 3.1), which is also referred to as a Dynamics Bayesian Network (DBN) (Dean & Kanazawa, 1989) adapted to the context of RL. For clarity, we illustrate a simple deterministic CDM in Appendix C.4.
Definition 3.1. A causal dynamics model is a tuple ⟨G,p⟩. G is the causal graph, i.e. a directed acyclic graph (DAG) on (S,A,S′), defining the parent set Pa(S′ j) for each S′ j in S′; p is a transition distribution on (S,A,S′) such that
(1)
In this work, G is unknown and must be learned from the data. Some studies learn CGs using sparsity constraints, which encourage models to predict the transition using fewer inputs (Volodin, 2021; Wang et al., 2021). However, there exists no theoretical guarantee that sparsity can lead to sound causality. In several recent studies (Wang et al., 2022; Ding et al., 2022; Zhu et al., 2022; Yu et al., 2023), conditional independence tests (CITs) are used to discover CGs (Eberhardt, 2017). Theorem 3.2 presents a prevalent approach for causal discovery, with proof in Appendix C.3. Additionally, we also prove the robustness of this approach against mild observational noise in Theorem C.13. Theorem 3.2 (Causal discovery for CDMs). Assuming that state variables transit independently, i.e. p(S′∣S,A) = ∏ns j=1 p(S′ j∣S,A), then the ground-truth causal graph G is bipartite. That is, all edges start in (S,A) and end in S′; if p is a faithful probability function consistent with the dynamics, then G is uniquely identified by
#  G Xi ∈Pa(S′ j) ⇔¬(Xi �p S′ j ∣(S,A) ∖{Xi}), (
(2)
 ∈( ) ⇔¬( �  ∣() ∖{}) for each Xi ∈(S,A) and S′ j ∈S′. Here, “�p” denotes the conditional independence under p, and “∖” means setsubtraction.
The independence “�p” here can be determined by CITs, which utilize samples drawn from p to evaluate whether the conditional independence holds. There are many tools for CITs, such as Fast CIT (Chalupka et al., 2018), Kernelbased CIT (Zhang et al., 2012), and Conditional Mutual Information (CMI) used in this work. Read Appendix B.4 for more information about CITs and CMI.
Testing Eq. 2 leads to sound CGs, yet is hardly scalable. Let n ∶= na + ns denote the total number of environment variables. Then, the time complexity of mainstream approaches reaches up to O(n3), since O(n2) edges must be tested, each costing O(n). Additionally, a larger n impairs sampling efficiency, as CITs require more samples to recover the joint distribution of condition variables.
# 3.2. Object-Oriented Markov Decision Process
Following Guestrin et al. (2003a), we formulate the task as an Object-Oriented MDP (OOMDP) containing a set O = {O1,⋯,ON} of objects. Each object Oi corresponds to a subset of variables (called its attributes), written as Oi = (Oi.S,Oi.A), where Oi.S ⊆S and Oi.A ⊆A respectively are its state attributes and action attributes. The objects are divided into a set of classes C = {C1,⋯,CK}. We call Oi an instance of Ck if Oi belongs to some class Ck, denoted as Oi ∈Ck. Ck specifies a set F[Ck] of fields, which determine the attributes of Oi as well as other instances of Ck. Each field in F[Ck], typically written as Ck.U (where U can be replaced by any identifier), signifies an attribute
Oi.U ∈Oi for each Oi ∈Ck. Note that italic letters are used for identifiers (e.g., Ck.U), while Roman letters are used for attributes (e.g., Oi.U) to highlight that attributes are random variables. A more rigorous definition of OOMDP is given in Appendix D.1.
The dynamics of the OOMDP satisfy that the state variables of objects from the same class transit according to the same (unknown) class-level transition function:
p(Oi.S′∣S, A) = pCk(Oi.S′∣Oi; O1, ⋯, Oi−1, Oi+1, ⋯, ON
(∣) =(∣ ⋯−+ ⋯) for all Oi ∈Ck, which we refer to as the result symmetry. Diuk et al. (2008) further formulates the dynamics using logical rules, which is not necessarily required here.
This OOMDP representation is inherently available in many simulation platforms or can be intuitively specified from human experience. Therefore, we consider the OOMDP representation as prior knowledge and leave its learning to future work. To illustrate our setting, we present Example 3.3 as the OOMDP for a StarCraft environment. Example 3.3. In a StarCraft scenario shown in Figure 1, the set of objects is O = {M1,M2,Z1,Z2,Z3} and the set of classes is C = {CM,CZ}. CM is the class for marines M1 and M2. Similarly, CZ is the class for zerglings Z1, Z2, and Z3. The fields for both C = CM,CZ are given by F[C] = {C.H, C.P, C.A} — the Health, Position, and Action (e.g., move or attack). Therefore, for example, M1.H is the health of marine M1, and M1 = (M1.H,M1.P,M1.A).
# 4. Method
The core of an OOCDM is the Object-Oriented Causal Graph (OOCG), which allows for class-level causality sharing based on the dynamic similarity between objects of the same class (see Section 4.1). Equation 3 has illustrated this similarity with respect to the result terms of the transition probabilities. Furthermore, we introduce an assumption 4.1 concerning the condition terms, called causation symmetry. It provides a natural notion that objects of the same class produce symmetrical effects on other objects. Figure 1 illustrates this assumption using the StarCraft scenario described above — swapping all attributes between two zerglings Z2 and Z3 makes no difference to the transition of other objects such as the marine M2. We also assume that all state variables (attributes) transit independently in accordance with FMDPs (Guestrin et al., 2003b). Assumption 4.1 (Causation Symmetry). Suppose Oi ∈Ck. Assuming that Oa,Ob ∈Cl (a,b ≠i) are two other objects from the same class, then Oa and Ob are interchangeable to the transition of Oi: p(Oi.S′∣Oa = a, Ob = b, ⋯) = p(Oi.S′∣Oa = b, Ob = a, ⋯).
     The workflow for using an OOCDM is illustrated in Figure 2.
𝑝𝑴2. 𝐒′ = 𝒙𝐙2 = 𝒛3, 𝐙3 = 𝒛2, ⋯
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/766e/766e4ed8-8dc7-4b51-a319-7d6bc10d1ffa.png" style="width: 50%;"></div>
<div style="text-align: center;">𝑝𝑴2. 𝐒′ = 𝒙𝐙2 = 𝒛2, 𝐙3 = 𝒛3, ⋯</div>
Figure 1. An OOMDP for Starcraft with the causation symmetry.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f88a/f88a2470-97ee-44e2-8b1c-9b449352a244.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2. The workflow overview.</div>
First, we use domain knowledge about the task to construct its OOMDP representation (Section 3.2). Subsequently, we initialize the OOCDM inclusive of field predictors (Section 4.2) and an OOCG estimation ˆG. This estimation is updated by performing causal discovery on the transition data and the predictors (Section 4.3), and these predictors are optimized using the current OOCG estimation and the stored data (Section 4.4). The learned OOCDM can then be applied to problems that require a CDM or causal graph (some basic applications are tested in Section 5).
# 4.1. Object-Oriented Causal Graph
According to Theorem 3.2, the ground-truth CG of an OOMDP follows a bipartite causal graph (BCG) structure, where no lateral edge is present in S′. In order to simplify the process of causal discovery, we impose a restriction on the structure of G and introduce a special form of CGs that allows class-level causal sharing.
 G Definition 4.2. Let Fs[Ck] ⊆F[Ck] be the set of state fields of class Ck. An Object-Oriented Causal Graph is a BCG where all causal edges are given by a series of classevel causalities: 1. A class-level local causality for class Ck from field Ck.U ∈F[Ck] to state field Ck.V ∈Fs[Ck], denoted as Ck.U →V ′, means that O.U ∈Pa(O.V′) for every instance O ∈Ck. 2. A class-level global causality from field Cl.U ∈F[Cl] to state field Ck.V ∈Fs[Ck], denoted as Cl.U → Ck.V ′, means that Oj.U ∈Pa(Oi.V′) for every Oi ∈ Ck and every Oj ∈Cl (j ≠i).
 G Definition 4.2. Let Fs[Ck] ⊆F[Ck] be the set of state fields of class Ck. An Object-Oriented Causal Graph is a BCG where all causal edges are given by a series of classlevel causalities: 1. A class-level local causality for class Ck from field Ck.U ∈F[Ck] to state field Ck.V ∈Fs[Ck], denoted as Ck.U →V ′, means that O.U ∈Pa(O.V′) for every instance O ∈Ck. 2. A class-level global causality from field Cl.U ∈F[Cl] to state field Ck.V ∈Fs[Ck], denoted as Cl.U → Ck.V ′, means that Oj.U ∈Pa(Oi.V′) for every Oi ∈ Ck and every Oj ∈Cl (j ≠i).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/42c2/42c2a096-6e6f-43a7-b66b-dd48fe477dce.png" style="width: 50%;"></div>
<div style="text-align: center;"> → Figure 3. The class-level causalities in Example 3.3.</div>
Definition 4.2 enables causality sharing by two types of class-level causalities, which are invariant with the number of instances of each class. Similar to relational causal discovery (Marazopoulou et al., 2015), this causality sharing greatly simplifies causal discovery and improves the readability of CGs. The local causality describes shared structures within individual objects of the same class, as illustrated in Figure 3(a). The global causality accounts for shared structures of object pairs, as illustrated in Figure 3(b). Note that the global causality Ck.U →Ck.V ′ (i.e., when k = l) is different from the local causality Ck.U →V ′ by definition. For clarity, the global and local causalities here are different from those considered by Pitis et al. (2020), where “local” means that (S,A) is confined in a small region in the entire space.
Definition 4.2 enables causality sharing by two types of class-level causalities, which are invariant with the number of instances of each class. Similar to relational causal discovery (Marazopoulou et al., 2015), this causality sharing greatly simplifies causal discovery and improves the readability of CGs. The local causality describes shared structures within individual objects of the same class, as illustrated in Figure 3(a). The global causality accounts for shared structures of object pairs, as illustrated in Figure 3(b). Note that the global causality Ck.U →Ck.V ′ (i.e., when k = l) is different from the local causality Ck.U →V ′ by definition. For clarity, the global and local causalities here are different from those considered by Pitis et al. (2020), where “local” means that (S,A) is confined in a small region in the entire space. An OOCG representing the dynamics (i.e., Eq. 1 holds) always exists, as a fully-connected BCG is apparently an OOCG. As shown in Theorem 4.3, our major result is that OOCGs reveal the ground-truth causality given the symmetry assumption, with proof in Appendix D.2. Theorem 4.3. The ground-truth CG of any OOMDP where Assumption 4.1 holds is exactly an OOCG.
Theorem 4.3. The ground-truth CG of any OOMDP where Assumption 4.1 holds is exactly an OOCG.
# 4.2. Object-Oriented Causal Dynamics Model
Definition 4.4. An object-oriented causal dynamics model is a CDM ⟨G, ˆp⟩(see Definition 3.1) such that 1) G is an OOCG, and 2) ˆp satisfies Eqs. 3 and 4.
Based on OOCGs, we are able to define CDMs in an object-oriented manner (see Definition 4.4). In conventional CDMs, there exists an independent predictor for each
Based on OOCGs, we are able to define CDMs in an object-oriented manner (see Definition 4.4). In conventional CDMs, there exists an independent predictor for each
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ca0f/ca0f81bd-e6de-4837-8865-04937f6ac26b.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4. The illustration of fC1.V (O1.V′∣S, A; G).</div>
next-state attribute (variable) in S′. However, Equation 3 offers an opportunity to reduce the number of predictors by class-level sharing. That is, a shared field predictor fC.V is used for each state field C.V ∈F[C] to predict the corresponding attribute O.V′ for every instance O ∈C. We now briefly describe how an OOCDM is implemented in our work. Inspired by Wang et al. (2022), we let an OOCG G be an argument of the predictor fC.V , making it adaptable to various graph structures. Therefore, in our implementation, it follows that
next-state attribute (variable) in S′. However, Equation 3 offers an opportunity to reduce the number of predictors by class-level sharing. That is, a shared field predictor fC.V is used for each state field C.V ∈F[C] to predict the corresponding attribute O.V′ for every instance O ∈C.
 ∈F[] ′ ∈ We now briefly describe how an OOCDM is implemented in our work. Inspired by Wang et al. (2022), we let an OOCG G be an argument of the predictor fC.V , making it adaptable to various graph structures. Therefore, in our implementation, it follows that
ˆp(O.V′∣PaG(O.V′)) = fC.V (O.V′∣S,A;G)
(5)
(∣G()) = (∣G) for every O ∈C, where PaG(O.V′) is the parent set of O.V′ in G. We ensure that fC.V adheres to G by masking off the non-parental variables. In addition, we adopt keyvalue attention (Vaswani et al., 2017) to ensure causation symmetry (Eq. 4) and enable adaptation to varying numbers of objects. A simple illustration of our implementation of fC.V is given as Figure 4, and detail is in Appendix E.
# 4.3. Object-Oriented Causal Discovery
Theorem 4.3 indicates that causal discovery in an OOMDP with Assumption 4.1 becomes looking for an OOCG. If the numbers of instances are fixed, checking each class-level causality in the OOCG only requires one CIT (see Appendix D.3), where most CIT tools are applicable. Further, to perform CITs in environments with changeable instance numbers, we introduce an adaptation of CDL using the class-level conditional mutual information.
Assume that we have a dataset D = {(st,at,st+1)}T t=1, where st, at and st+1 are the observed values of S, A and S′ at step t, respectively. We use O.vt+1 to denote the observed O.V in st+1 for each state field Ck.V and instance O ∈Ck. Some OOCGs are helpful to the estimation of CMI: I) G1 is the full bipartite CG containing all causalities,
which is also an OOCG by definition; II) GC.U /→V ′ contains all causalities except for C.U →V ′; and III) GCk.U /→C.V ′ contains all causalities except for Ck.U →C.V ′. Letting Ct k denotes the set of instances of class Ck at step t, with the predictors introduced in Section 4.2, we respectively write the CMIs for class-level local and global causalities as
(6)
(7)
 (∣ G) Then, each class-level causality (denoted as ς) is confirmed if Iς D > ε, where ε is a positive threshold parameter. In other words, Iς D compare the predictions made with and without the concerned parents within ς, and we confirm the causality if the difference is significant. In this way, no extra models are needed for causal discovery. Finally, the whole OOCG is obtained by checking CMIs for all possible causalities (see Appendix E.3 for the pseudo-code).
 (∣ G) Then, each class-level causality (denoted as ς) is confirmed if Iς D > ε, where ε is a positive threshold parameter. In other words, Iς D compare the predictions made with and without the concerned parents within ς, and we confirm the causality if the difference is significant. In this way, no extra models are needed for causal discovery. Finally, the whole OOCG is obtained by checking CMIs for all possible causalities (see Appendix E.3 for the pseudo-code). Our approach greatly reduces the computational complexities of causal discovery, from a magnitude (asymptotic boundary) of n3 to a magnitude of Nmn, where m denotes the overall number of fields and n denotes the overall number of variables in (S,A). See proofs and more conclusions about computational complexities in Appendix F.
Our approach greatly reduces the computational complexities of causal discovery, from a magnitude (asymptotic boundary) of n3 to a magnitude of Nmn, where m denotes the overall number of fields and n denotes the overall number of variables in (S,A). See proofs and more conclusions about computational complexities in Appendix F.
# 4.4. Model Learning
Dynamics models are usually optimized through Maximum Likelihood Estimation. To better adapt to the varying numbers of instances, we define the average instance loglikelihood (AILL) function on a transition dataset D of T steps for any CDM ⟨G, ˆp⟩as
(8)
(∣G()) where ˆp(⋅)t is the estimated probability when variables take the values observed at step t in D.
(⋅)  D The learning target of an OOCDM mimics that of CDL. First, we optimize the AILL function under a random OOCG denoted as Gλ (re-sampled when each time used) where the probability of each class-level causality item is λ. This will make our model capable of handling incomplete information and adaptable to different OOCGs including those like GC.U /→V ′ or GCk.U /→C.V ′. Furthermore, we also hope
to strengthen our model in two particular OOCGs: 1) the estimation of ground-truth ˆG obtained by causal discovery, where CMIs are estimated by the current model, and 2) the full OOCG G1 to better estimate CMIs in Eqs. 6 and 7. Therefore, two additional items, LG1(D) and L ˆG(D), respectively weighted by α and β, are considered in the overall target function:
(9)
(D) = LG(D) +LG(D) +LG(D) which is optimized by gradient ascent. Pseudo-code of the learning algorithm is in Appendix E.4. During the test phase, all predictions of our OOCDM are made using the discovered OOCG ˆG.
# G 4.5. Releasing Dynamic Symmetries
According to the above definition, OOCDMs comply with the dynamic symmetries (Eqs. 3 and 4), and thus the environmental dynamics are required to have the same properties. This paper mainly discusses symmetric dynamics, as dynamic symmetries are natural in large-scale environments with varying numbers of objects, and they can always be ensured by using a proper representation of OOMDP that provides sufficient detail. For instance, if we formulate the StarCraft case in Example 3.3 using only one class CUnit for both marines and zerglings, it may violate the causation symmetries, whereas using the two-class representation or introducing more attributes (e.g., the attacking range and faction) can ensure the symmetries.
According to the above definition, OOCDMs comply with the dynamic symmetries (Eqs. 3 and 4), and thus the environmental dynamics are required to have the same properties. This paper mainly discusses symmetric dynamics, as dynamic symmetries are natural in large-scale environments with varying numbers of objects, and they can always be ensured by using a proper representation of OOMDP that provides sufficient detail. For instance, if we formulate the StarCraft case in Example 3.3 using only one class CUnit for both marines and zerglings, it may violate the causation symmetries, whereas using the two-class representation or introducing more attributes (e.g., the attacking range and faction) can ensure the symmetries. However, the refinement of representations is sometimes not possible, due to the limited access or insufficient domain knowledge of users. Therefore, we look forward to releasing the requirement of dynamic symmetries on the alreadygiven OOMDPs. Theoretically, it is always plausible to ensure dynamic symmetries via auxiliary attributes (see Appendix D.4). Therefore, we can augment the OOCDM using built-in auxiliary attributes, which carry the information personalizing each object. In fact, the simplest way to do so is to include the indices of the objects as attributes. However, mapping indices into the personal dynamics seems indirect for the neural predictors. Instead, the augmented OOCDM utilizes the hidden encoding of auxiliary attributes:
However, the refinement of representations is sometimes not possible, due to the limited access or insufficient domain knowledge of users. Therefore, we look forward to releasing the requirement of dynamic symmetries on the alreadygiven OOMDPs. Theoretically, it is always plausible to ensure dynamic symmetries via auxiliary attributes (see Appendix D.4). Therefore, we can augment the OOCDM using built-in auxiliary attributes, which carry the information personalizing each object. In fact, the simplest way to do so is to include the indices of the objects as attributes. However, mapping indices into the personal dynamics seems indirect for the neural predictors. Instead, the augmented OOCDM utilizes the hidden encoding of auxiliary attributes:
(∣G()) = (∣ ⋯ G) where hi is the learned hidden encoding of auxiliary attributes for Oi. The hidden encodings for each class are generated by a bi-directional Gate Recurrent Unit, which can handle varying numbers of instances. The implementation detail is provided in Appendix E.2.
The augmentation allows OOCDM to learn the personal dynamics of each object. Since the ground-truth CG in an asymmetric OOMDP may not be an OOCG, object-oriented causal discovery leads to the minimal OOCG that repre-
sents the dynamics. To obtain a precise CG, we can apply OOCDM to non-OO causal discovery based on Theorem 3.2 (e.g., using the CDL approach), which takes advantage of class-level parameter sharing and inter-object attention. However, we suggest using the minimal OOCG to capture the approximate causality, which strikes a good balance between computational efficiency and structural precision, especially in large environments where non-OO CDMs fail. Additional experiments of the augmented OOCDM using OOCGs are included in Appendix H.7.
# 5. Experiments
OOCDM was compared with several state-of-the-art CDMs. CDL uses pooling-based predictors and also adopts CMIs for causal discovery. CDL-A is the attention-based variant of CDL, used to make a fair comparison with our model. GRADER (Ding et al., 2022) employs Fast CIT for causal discovery and Gated Recurrent Units as predictors. TICSA (Wang et al., 2021) utilizes score-based causal discovery. Meanwhile, OOCDM was compared to non-causal baselines, including a widely used multi-layer perceptron (MLP) in model-based RL (MBRL) and an object-aware Graph Neural Network (GNN) that uses the architecture of (Kipf et al., 2020) to learn inter-object relationships. Additionally, we assessed the performance of the dense version of our OOCDM, namely OOFULL, which employs the full OOCG G1 and is trained by optimizing LG1. As mentioned in Section 2.1, CDMs are used for various purposes, and this work does not aim to specify the use of OOCDMs. Therefore, we evaluate the performance of causal discovery and the predicting accuracy, as most applications can benefit from such criteria. As a common application in MBRL, we also evaluate the performance of planning using dynamics models. Our experiments aim to 1) demonstrate that the OO framework greatly improves the effectiveness of CDMs in large-scale environments, and 2) investigate in what occasions causality brings significant advantages. Moreover, additional results on noisy data are included in Appendix I, which evaluate the robustness against observational noise. Results are presented by the means and standard variances of 5 random seeds. Experimental details are presented in Appendix H.
# 5.1. Environments
We conducted experiments in 4 environments. The Block environment consists of several instances of class Block and one instance of class Total. The attributes of each Block object transit via a linear transform; and the attributes of the Total object transit based on the maximums of attributes of the Block objects. The Mouse environment is an 8 × 8 grid world containing an instance of class Mouse, and several instances of class Food, Monster, and Trap. The mouse
<div style="text-align: center;">Table 1. The accuracy (in percentage) of discovered causal graphs. n indicates the number of environmental variables.</div>
Env
n GRADER
CDL
CDL-A
TICSA
OOCDM
Block2 12 94.8±1.3 99.4±0.3 99.2±1.3 97.0±0.4
99.7±0.6
Block5 24 94.0±1.5 97.5±1.5 99.3±0.6 96.3±0.6 100.0±0.0
Block10 44 92.3±0.9 97.6±0.3 99.5±0.3 97.7±0.5 100.0±0.0
Mouse
28 90.5±0.8 90.4±3.2 94.7±0.2 94.1±0.2 100.0±0.0
Table 2. The time (seconds) used in causal discovery. m denotes the number of all fields. The sample sizes are 10k, 50k, 100k, and 200k, respectively in Block, Mouse, CMD, and DZB. The only exception is GRADER in DZB, which only uses 20k samples to make sure that causal discovery finishes within a tolerable time. TICSA is excluded from comparison as it does not involve an explicit causal discovery phase.
Env
n m
GRADER
CDL
CDL-A
OOCDM
Block2 12 8
114.0±1.5
1.4±0.1
2.1±0.4
2.1±0.2
Block5 24 8 927.0±69.3
5.2±0.6
8.5±2.8
2.1±0.2
Block10 44 8 7.0e3±217.7
15.6±0.7
22.8±5.3
2.2±0.2
Mouse
29 10 1.7e4±138.4
57.8±2.4
45.3±2.6
17.7±4.0
CMS
44 4 5.5e4±397.1 209.8±18.9 252.5±27.3
7.4±0.5
DZB
66 10 2.7e4±387.1 715.6±10.9 1.1e3±274.7 66.1±0.6
can be killed by hunger or monsters, and its goal is to survive as long as possible. The Collect-Mineral-Shards (CMS) and Defeat-Zerglings-Baineling (DZB) environments are StarCraftII mini-games (Vinyals et al., 2017). In CMS, the player controls two marines to collect 20 mineral shards scattered on the map, and in DZB the player controls a group of marines to kill hostile zerglings and banelings. Read Appendix G for detailed descriptions of these environments. The Block and Mouse environments are ideal OOMDPs as they guarantee Eqs. 3 and 4. In addition, we intentionally insert spurious correlations in them to verify the effectiveness of causal discovery. In CMS and DZB environments, we formulate the objects and classes based on the units and their types in StarCraftII. Such formulation accounts for more piratical cases of imperfect OOMDPs, as the StarCraftII engine may not guarantee Eqs. 3 and 4, yet dynamic symmetries should roughly hold by intuition. For dynamics that are clearly asymmetric, please read Appendix H.7.
# 5.2. Performance of Causal Discovery
We measured the performance of causal discovery using offline data in Block and Mouse environments. Since non-OO baselines only accept a fixed number of variables, the number of instances of each class is fixed in these environments. Especially, we use “Blockk” to denote the Block environment where the number of Block instances is fixed to k. We exclude CMS and DZB here as their ground-truth CGs are unknown (see learned OOCGs in Appendix H.6). We measure the accuracy of discovered CGs by the Structural
Hamming Distance within the edges from (S,A) to S′. As shown in Table 1, OOCDM outperforms other CDMs in all environments and recovers ground-truth CGs in 3 out of 4 environments. These results demonstrate the improved sample efficiency of OOCDM in large-scale environments. Table 2 shows the computation time used by causal discovery. We note that such results may be influenced by implementation detail and hardware conditions, yet the OOCDM excels baselines with a significant gap beyond these extraneous influences. In addition, Appendix H.5 shows that OOCDM achieves better performance with a relatively smaller size (i.e. fewer model parameters).
# 5.3. Predicting Accuracy
We use the AILL functions (Eq. 8) to measure the predicting accuracy of dynamics models. The models are learned using offline training data. Then, the AILL functions of these models are evaluated on the i.d. (in-distribution) test data sampled from the same distribution as the training data. Especially, in Block and Mouse environments, we can modify the distribution of the starting state of each episode (see Appendix H.3) and obtain the o.o.d. (out-of-distribution) test data, which contains samples that are unlikely to appear during training. The i.d. and o.o.d. test data measure two levels of generalization, respectively considering situations that are alike and unalike to those in training. We do not collect the o.o.d. data for CMS and DZB, as the PySC2 platform provides limited access to modify the initialization process in the StarCraft engine (Vinyals et al., 2017). The results are shown in Table 3. In small-scale environments like Block2, causal models show better generalization ability than dense models on both i.d. and o.o.d. test data. However, in larger-scale environments, the performance of non-OO models declines sharply, and OO models (OOFULL and OOCDM) obtain the highest performance on the i.d. data. In addition, our OOCDM exhibits the best generalization ability on the o.o.d. data; in contrast, the performance of OOFULL is extremely low on such data. These results demonstrate that OO models are more effective in large-scale environments, and that causality greatly improves the generalization of OO models.
# 5.4. Combining Models with Planning
In this experiment, we trained dynamics models using offline data (collected through random actions). Given a reward function, we used these models to guide decisionmaking using Model Predictive Control (Camacho & Bordons, 1999) combined with Cross-Entropy Method (Botev et al., 2013) (see Appendix E.5), which is widely used in MBRL. The Block environment is not included here as it does not involve rewards. In the Mouse environment, the o.o.d. initialization mentioned in Section 5.3 is also consid-
<div style="text-align: center;">rage instance log-likelihoods of the dynamics models on various datasets. We do not show the standard variances for fitting results (less than −100.0, highlighted in brown), as their variances are all extremely large.</div>
 −
Env
data
GRADER
CDL
CDL-A
TICSA
GNN
MLP
OOFULL
OOCDM
Block2
train
21.1±0.3
20.9±1.5
19.3±1.9
17.4±2.2
18.8±0.6
16.5±1.2
21.5±0.9
22.4±0.7
i.d.
17.1±2.5
20.2±1.8
10.4±16.8
16.4±1.9
17.9±0.7
10.1±4.4
−568.2
22.2±0.7
o.o.d.
−65.4
11.5±6.7
−6.0e5
−60.1±2.8
−5.0±23.4
−7.2e4
−4.6e4
21.3±1.9
Block5
train
19.1±3.4
16.5±2.1
18.9±0.7
12.0±0.7
14.9±14.4
12.6±0.5
20.4±1.7
19.6±1.7
i.d.
6.7±4.3
−45.3±113.2
−1.4e7
10.8±0.7
14.4±0.4
−2.2±6.3
19.8±1.7
19.5±1.7
o.o.d. −95.6±41.7
−5.3e6
−1.1e9
−5.5e3
−13.4±3.4
−1.5e7
−4.0e7
13.5±4.3
Block10
train
19.3±0.6
12.9±0.8
16.0±0.6
11.1±1.3
13.3±0.15
8.9±0.6
20.3±0.6
21.2±0.3
i.d.
−26.7±8.4
6.9±6.4
−9.2±42.5
−10.4±39.8
12.9±0.2
−75.3±20.0
20.2±0.6
21.1±0.3
o.o.d.
−119.1
−4.2e6
−1.9e8
−139.4
−17.3±17.3
−780.9
−5.4e3
15.6±5.4
Mouse
train
24.2±0.6
13.9±1.8
22.3±1.4
13.6±3.5
25.6±1.8
5.7±0.4
30.0±1.4
32.2±1.1
i.d.
−3.2e3
−2.0e5
−3.6e4
−1.5e4
−2.7e4
−1.6e7
−65.0±153.3
26.8±6.7
o.o.d.
−7.1e4
−1.1e10
−2.0e10
−2.5e7
−6.3e10
−8.0e10
−1.5e9
11.2±17.2
CMS
train
−1.2±0.1
3.6±0.8
4.1±1.5
2.8±1.6
6.4±6.2
−2.0±1.5
8.5±1.1
9.0±0.5
i.d.
−1.3±0.1
−1.0e6
4.1±1.5
−16.3±7.4
6.3±0.1
−6.4e9
8.5±1.1
8.9±0.5
DZB
train
11.0±1.0
4.2±2.5
12.1±0.1
13.2±1.2
18.0±10.0
−0.9±0.8
29.0±0.6
27.2±2.5
i.d.
−14.9±21.8
−3.3±6.6
5.3±5.3
−2.4e5
13.0±12.8
−1.6e12
22.6±5.6
24.4±5.9
<div style="text-align: center;">Table 4. The average return of episodes when models are used for planning. In the Mouse environment, “o.o.d.” indicates the initial state are sampled from a new distribution.</div>
Env
GRADER
CDL
CDL-A
TICSA
GNN
MLP
OOFULL
OOCDM
Mouse −1.2±1.9
3.9±3.0
−5.0±1.3
−0.8±0.7
6.6±3.2
0.6±2.0
77.9±18.1
80.1±16.9
o.o.d.
−0.4±1.7
1.8±2.5
−0.9±1.1
−1.2±0.6
0.6±0.2
−1.3±0.7
62.2±8.7
75.1±17.5
CMS
−9.5±1.1
−9.8±1.1
−8.8±0.4
−9.3±0.9
−9.8±0.7
−8.8±0.5
−4.1±3.3
3.4±6.3
DZB 202.9±12.3 217.3±12.4 171.7±18.2 188.9±8.5 233.8±19.8 205.4±6.7 269.8±21.5 266.2±11.4
Table 5. Results on various tasks in the Mouse environment, measuring the average instance log-likelihood and the episodic return. “seen” and “unseen” respectively indicate the performances measured in seen and unseen tasks.
Model
log-likelihood
episodic return
train
seen
unseen
seen
unseen
OOCDM 26.9±3.5 25.4±2.8 24.8±2.8 94.8±29.7 88.8±34.8
OOFULL 30.7±1.9 22.5±3.2 7.9±29.8 77.0±24.6 70.8±22.4
ered. The average returns of episodes are shown in Table 4, showing that OOFULL and OOCDM are significantly better than non-OO approaches. Between the OO models, OOCDM obtains higher returns than OOFULL in 3 of 4 environments, which demonstrates that OOCDM better generalizes to the unseen state-action pairs produced by planning. Taking CMS for example, the agent collects only a few mineral shards in the training data. When the agent plans, it encounters unseen states where most mineral shards have been collected. However, we note that OOFULL performs slightly better than OOCDM
in DZB. One reason for this is that DZB possesses a joint action space of 9 marines, which is too large to conduct effective planning. Therefore, planning does not lead to states that are significantly different from those in training, prohibiting the advantage of generalization from converting to the advantage of returns. Additionally, the true CG of DZB is possibly less sparse than those in other environments, making OOFULL contain less spurious edges. Therefore, CDMs would be more helpful, if the true CG is sparse, and there exists a large divergence between the data distributions in training and testing.
# 5.5. Handling Varying Numbers of Instances
In the Mouse environment, we tested whether OOCDM and OOFULL are adaptable to various tasks with different numbers of Food, Moster, and Trap instances. We randomly divide tasks into the seen and unseen tasks (see Appendix H.4). Dynamics models are first trained in seen tasks and then transferred to the unseen without further training. We measured the log-likelihoods on the training data, the i.d. test data on seen tasks, and the test data on unseen tasks. The average episodic returns of planning were also evaluated,
separately on seen and unseen tasks. As shown in Table 5, our results demonstrate that 1) OO models can be learned using data from different tasks, 2) OO models perform a zero-shot transfer to unseen tasks with a mild reduction of performance, and 3) the overall performance is improved when combing the model with causality.
OOCDM greatly outperforms baselines in the generalization performance, where both prior knowledge and causality play a role. We note two possible sources of generalization error for dynamics models: 1) the spurious correlations in the data and 2) the insufficient data representing the joint space of numerous variables. Conventional CDMs can only reduce the first source of errors and still suffer from the second source. For example, CDL-A generalizes worse to o.o.d. datasets even though it implements better causal discovery than CDL, as the attention in CDL-A has a greater regression capacity than the pooling mechanism in CDL, leading to a higher risk of overfitting on the limited data. OOCDM can reduce error from the second source by using shared predictors for each class. Therefore, the improved performance of OOCDM partly derives from good exploitation of OOMDP priors, which suggests a further investigation to learn OOMDP representations in future studies. However, the prior knowledge alone is insufficient for a good generalization. Lacking a causal structure, OOFULL and GNN fail on the o.o.d. datasets even though they are aware of the object-oriented representation. Therefore, the causal structure is crucial in the generalization performance of OOCDM.
However, the prior knowledge alone is insufficient for a good generalization. Lacking a causal structure, OOFULL and GNN fail on the o.o.d. datasets even though they are aware of the object-oriented representation. Therefore, the causal structure is crucial in the generalization performance of OOCDM.
# 6. Conclusion
This paper proposes OOCDMs that capture the causal relationships within OOMDPs. Our main innovations are the OOCGs that share class-level causalities and the use of attention-based field predictors. Furthermore, we present a CMI-based method that discovers OOCGs in environments with changing numbers of objects. Theoretical and empirical data indicate that OOCDM greatly enhances the computational efficiency and accuracy of causal discovery in large-scale environments, surpassing state-of-the-art CDMs. Moreover, OOCDM well generalizes to unseen states and tasks, yielding commendable planning outcomes. In conclusion, this study provides OOCDM as a promising solution to learn and apply CDMs in large-scale object-oriented environments.
Future work would include extracting OOMDP representations from raw pixel-based features, considering potential unobserved confounders, and modeling the relational interaction of objects.
# Acknowledgements
This work was supported by the National Nature Science Foundation of China under Grant 62073324. The paper has gone through multiple revisions, in which suggestions from reviewers are extremely useful in improving the quality of the work, especially in improving the readability and presentation.
# Impact Statement
This paper explores the causality of world models in reinforcement learning, which increases the transparency and robustness of the models. Therefore, as a positive impact, we expect this work to contribute to more reliable modelbased agents in the future. To the best of our knowledge, this work does not raise any ethical issues.
# References
Botev, Z. I., Kroese, D. P., Rubinstein, R. Y., and L’Ecuyer, P. The Cross-Entropy Method for Optimization. In Handbook of Statistics, volume 31, pp. 35–59. Elsevier, 2013. ISBN 9780-444-53859-8. doi: 10.1016/B978-0-444-53859-8. 00003-5. URL https://linkinghub.elsevier. com/retrieve/pii/B9780444538598000035.
Boutilier, C., Dearden, R., and Goldszmidt, M. Stochastic dynamic programming with factored representations. Artificial Intelligence, 121(1):49–107, August 2000. ISSN 0004-3702. doi: 10.1016/S0004-3702(00)00033-3. URL https://www.sciencedirect.com/ science/article/pii/S0004370200000333.
Camacho, E. F. and Bordons, C. Model predictive control. Advanced textbooks in control and signal processing. Springer, Berlin ; New York, 1999. ISBN 978-3-54076241-6.
Chalupka, K., Perona, P., and Eberhardt, F. Fast Conditional Independence Test for Vector Variables with Large Sample Sizes, April 2018. URL http://arxiv.org/ abs/1804.02747. arXiv:1804.02747 [cs, stat].
Dean, T. and Kanazawa, K. A model for reasoning about persistence and causation. Computational Intelligence, 5(2):142–150, February 1989. ISSN 0824-7935, 14678640. doi: 10.1111/j.1467-8640.1989.tb00324.x. URL https://onlinelibrary.wiley.com/doi/ 10.1111/j.1467-8640.1989.tb00324.x. Ding, W., Lin, H., Li, B., and Zhao, D. Generalizing GoalConditioned Reinforcement Learning with Variational Causal Reasoning, July 2022. URL http://arxiv. org/abs/2207.09081. arXiv:2207.09081 [cs, stat]. Diuk, C., Cohen, A., and Littman, M. L. An object-oriented representation for efficient reinforcement learning. In Proceedings of the 25th international conference on Machine learning - ICML ’08, pp. 240–247, Helsinki, Finland, 2008. ACM Press. ISBN 978-1-60558-205-4. doi: 10. 1145/1390156.1390187. URL http://portal.acm. org/citation.cfm?doid=1390156.1390187. Eberhardt, F. Introduction to the foundations of causal discovery. International Journal of Data Science and Analytics, 3(2):81–91, March 2017. ISSN 2364415X, 2364-4168. doi: 10.1007/s41060-016-0038-6. URL http://link.springer.com/10.1007/ s41060-016-0038-6. Feng, F., Huang, B., Zhang, K., and Magliacane, S. Factored Adaptation for Non-Stationary Reinforcement Learning. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 31957–31971. Curran Associates, Inc., 2022. Goyal, A., Lamb, A., Gampa, P., Beaudoin, P., Levine, S., Blundell, C., Bengio, Y., and Mozer, M. Object Files and Schemata: Factorizing Declarative and Procedural Knowledge in Dynamical Systems, November 2020. URL http://arxiv.org/abs/2006. 16225. arXiv:2006.16225 [cs, stat]. Goyal, A., Didolkar, A., Ke, N. R., Blundell, C., Beaudoin, P., Heess, N., Mozer, M., and Bengio, Y. Neural Production Systems: Learning Rule-Governed Visual Dynamics, March 2022. URL http://arxiv.org/abs/ 2103.01937. arXiv:2103.01937 [cs, stat]. Guestrin, C., Koller, D., Gearhart, C., and Kanodia, N. Generalizing plans to new environments in relational MDPs. In Proceedings of the 18th International Joint Conference on Artificial Intelligence, August 2003a. URL https://ai.stanford.edu/˜koller/ Papers/Guestrin+al:IJCAI03.pdf. Guestrin, C., Koller, D., Parr, R., and Venkataraman, S. Efficient Solution Algorithms for Factored MDPs. Journal of Artificial Intelligence Research, 19:399–468,
Guestrin, C., Koller, D., Parr, R., and Venkataraman, S. Efficient Solution Algorithms for Factored MDPs. Journal of Artificial Intelligence Research, 19:399–468,
October 2003b. ISSN 1076-9757. doi: 10.1613/ jair.1000. URL http://arxiv.org/abs/1106. 1822. arXiv:1106.1822 [cs]. Guo, J., Gong, M., and Tao, D. A Relational Intervention Approach for Unsupervised Dynamics Generalization in Model-Based Reinforcement Learning, June 2022. Hadar, I. and Leron, U. How intuitive is object-oriented design? Communications of the ACM, 51(5):41–46, May 2008. ISSN 0001-0782, 1557-7317. doi: 10.1145/ 1342327.1342336. URL https://dl.acm.org/ doi/10.1145/1342327.1342336. Huang, B., Feng, F., Lu, C., Magliacane, S., and Zhang, K. AdaRL: What, Where, and How to Adapt in Transfer Reinforcement Learning, March 2022. URL http:// arxiv.org/abs/2107.02729. arXiv:2107.02729 [cs, stat]. Jang, E., Gu, S., and Poole, B. Categorical Reparameterization with Gumbel-Softmax, August 2017. URL http://arxiv.org/abs/1611. 01144. arXiv:1611.01144 [cs, stat]. Jonsson, A. and Barto, A. Causal Graph Based Decomposition of Factored MDPs. The Journal of Machine Learning Research, 7:2259–2301, December 2006. Ke, N. R., Didolkar, A., Mittal, S., Goyal, A., Lajoie, G., Bauer, S., Rezende, D., Bengio, Y., Mozer, M., and Pal, C. Systematic Evaluation of Causal Discovery in Visual Model Based Reinforcement Learning, July 2021. URL http://arxiv.org/abs/2107. 00848. arXiv:2107.00848 [cs, stat]. Kipf, T., van der Pol, E., and Welling, M. Contrastive Learning of Structured World Models, January 2020. URL http://arxiv.org/abs/1911. 12247. arXiv:1911.12247 [cs, stat]. Liao, L., Fu, Z., Yang, Z., Wang, Y., Kolar, M., and Wang, Z. Instrumental Variable Value Iteration for Causal Offline Reinforcement Learning, July 2021. URL http:// arxiv.org/abs/2102.09907. arXiv:2102.09907 [cs, stat]. Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G., Uszkoreit, J., Dosovitskiy, A., and Kipf, T. Object-Centric Learning with Slot Attention, October 2020. URL http://arxiv.org/abs/2006. 15055. arXiv:2006.15055 [cs, stat]. Lu, C., Sch¨olkopf, B., and Hern´andez-Lobato, J. M. Deconfounding Reinforcement Learning in Observational Settings, December 2018. URL http://arxiv.org/ abs/1812.10576. arXiv:1812.10576 [cs, stat].
Lu, C., Sch¨olkopf, B., and Hern´andez-Lobato, J. M. Deconfounding Reinforcement Learning in Observational Settings, December 2018. URL http://arxiv.org/ abs/1812.10576. arXiv:1812.10576 [cs, stat].
Pearl, J. Causality: models, reasoning, and inference. Cambridge University Press, Cambridge, U.K. ; New York, 2000. ISBN 978-0-521-89560-6 978-0-521-77362-1. Pearl, J. and Mackenzie, D. The book of why: the new science of cause and effect. Penguin science. Penguin Books, London, 2019. ISBN 978-0-14-198241-0. Pearl, J., Glymour, M., and Jewell, N. P. Causal inference in statistics: a primer. Wiley, Chichester, West Sussex, 2016. ISBN 978-1-119-18684-7. Peng, S., Hu, X., Zhang, R., Tang, K., Guo, J., Yi, Q., Chen, R., Zhang, X., Du, Z., Li, L., Guo, Q., and Chen, Y. Causality-driven Hierarchical Structure Discovery for Reinforcement Learning, October 2022. URL http:// arxiv.org/abs/2210.06964. arXiv:2210.06964 [cs]. Peters, J., Janzing, D., and Sch¨olkopf, B. Elements of causal inference: foundations and learning algorithms. Adaptive computation and machine learning series. The MIT Press, Cambridge, Massachuestts, 2017. ISBN 978-0262-03731-0. Pitis, S., Creager, E., and Garg, A. Counterfactual Data Augmentation using Locally Factored Dynamics, December 2020. URL http://arxiv.org/abs/2007. 02863. arXiv:2007.02863 [cs, stat]. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal Policy Optimization Algorithms, August 2017. URL http://arxiv.org/ abs/1707.06347. arXiv:1707.06347 [cs]. Stroustrup, B. What is object-oriented programming? IEEE Software, 5(3):10–20, 1988. doi: 10.1109/52.2020. Sutton, R. S. and Barto, A. G. Reinforcement Learning: An Introduction. The MIT Press, Cambridge, Massachusetts, 2 edition, 2018. ISBN 978-0-262-36401-0. URL https://mitpress.ublish.com/book/ reinforcement-learning-an-introduction Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is All you Need. In Proceedings of the 31st International Conference on Neural Information Processing Systems, Long Beach, CA, USA, 2017. Vinyals, O., Ewalds, T., Bartunov, S., Georgiev, P., Vezhnevets, A. S., Yeo, M., Makhzani, A., K¨uttler, H., Agapiou, J., Schrittwieser, J., Quan, J., Gaffney, S., Petersen, S., Simonyan, K., Schaul, T., van Hasselt, H., Silver, D., Lillicrap, T., Calderone, K., Keet, P., Brunasso, A., Lawrence, D., Ekermo, A., Repp, J., and Tsing, R. StarCraft II: A New Challenge for Reinforcement Learning, August 2017. URL http://arxiv.org/abs/ 1708.04782. arXiv:1708.04782 [cs].
Madumal, P., Miller, T., Sonenberg, L., and Vetere, F. Distal Explanations for Model-free Explainable Reinforcement Learning, September 2020a. URL http://arxiv. org/abs/2001.10284. arXiv:2001.10284 [cs]. Madumal, P., Miller, T., Sonenberg, L., and Vetere, F. Explainable Reinforcement Learning through a Causal Lens. Proceedings of the AAAI Conference on Artificial Intelligence, 34(03):2493–2500, April 2020b. ISSN 2374-3468, 2159-5399. doi: 10.1609/ aaai.v34i03.5631. URL https://aaai.org/ojs/ index.php/AAAI/article/view/5631.
Maier, M., Taylor, B., Oktay, H., and Jensen, D. Learning causal models of relational domains. In Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence, pp. 531–538. AAAI Press, 2010.
Malysheva, A., Kudenko, D., and Shpilman, A. MAGNet: Multi-agent Graph Network for Deep Multi-agent Reinforcement Learning. In 2019 XVI International Symposium “Problems of Redundancy in Information and Control Systems” (REDUNDANCY), pp. 171–176, Moscow, Russia, October 2019. IEEE. ISBN 9781-72811-944-1. doi: 10.1109/REDUNDANCY48165. 2019.9003345. URL https://ieeexplore.ieee. org/document/9003345/.
Marazopoulou, K., Maier, M., and Jensen, D. Learning the structure of causal models with relational and temporal dependence. In Proceedings of the Thirty-First Conference on Uncertainty in Artificial Intelligence, pp. 572–581, Arlington, Virginia, USA, 2015. AUAI Press. ISBN 9780996643108.
Osband, I. and Van Roy, B. Near-optimal Reinforcement Learning in Factored MDPs, October 2014. URL http: //arxiv.org/abs/1403.3741. arXiv:1403.3741 [cs, stat].
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is All you Need. In Proceedings of the 31st International Conference on Neural Information Processing Systems, Long Beach, CA, USA, 2017.
inyals, O., Ewalds, T., Bartunov, S., Georgiev, P., Vezhnevets, A. S., Yeo, M., Makhzani, A., K¨uttler, H., Agapiou, J., Schrittwieser, J., Quan, J., Gaffney, S., Petersen, S., Simonyan, K., Schaul, T., van Hasselt, H., Silver, D., Lillicrap, T., Calderone, K., Keet, P., Brunasso, A., Lawrence, D., Ekermo, A., Repp, J., and Tsing, R. StarCraft II: A New Challenge for Reinforcement Learning, August 2017. URL http://arxiv.org/abs/ 1708.04782. arXiv:1708.04782 [cs].
t. Zhang, A., Lyle, C., Sodhani, S., Filos, A., Kwiatkowska, M., Pineau, J., Gal, Y., and Precup, D. Invariant Causal Prediction for Block MDPs. In Proceedings of the 37th International Conference on Machine Learning, pp. 11214–11224. PMLR, November 2020. URL https://proceedings.mlr.press/ v119/zhang20t.html. ISSN: 2640-3498. Zhang, K., Peters, J., Janzing, D., and Schoelkopf, B. Kernel-based Conditional Independence Test and Application in Causal Discovery, February 2012. URL http: //arxiv.org/abs/1202.3775. arXiv:1202.3775 [cs, stat]. Zhou, A., Kumar, V., Finn, C., and Rajeswaran, A. Policy Architectures for Compositional Generalization in Control, March 2022. URL http://arxiv.org/abs/ 2203.05960. arXiv:2203.05960 [cs]. Zhu, G., Huang, Z., and Zhang, C. Object-Oriented Dynamics Predictor, October 2018. URL http://arxiv. org/abs/1806.07371. arXiv:1806.07371 [cs]. Zhu, G., Wang, J., Ren, Z., Lin, Z., and Zhang, C. ObjectOriented Dynamics Learning through Multi-Level Abstraction, December 2019. URL http://arxiv. org/abs/1904.07482. arXiv:1904.07482 [cs, stat]. Zhu, Z.-M., Chen, X.-H., Tian, H.-L., Zhang, K., and Yu, Y. Offline Reinforcement Learning with Causal Structured World Models, June 2022. URL http://arxiv. org/abs/2206.01474. arXiv:2206.01474 [cs, stat].
Volodin, S. CauseOccam : Learning Interpretable Abstract Representations in Reinforcement Learning Environments via Model Sparsity. PhD thesis, ´Ecole Polytechnique F´ed´erale de Lausanne, 2021.
Representations in Reinforcement Learning Environments via Model Sparsity. PhD thesis, ´Ecole Polytechnique F´ed´erale de Lausanne, 2021. Wang, Z., Xiao, X., Zhu, Y., and Stone, P. Task-Independent Causal State Abstraction. In NeurIPS 2021 Workshop on Robot Learning: Self-Supervised and Lifelong Learning, pp. 10, Virtual, 2021. Wang, Z., Xiao, X., Xu, Z., Zhu, Y., and Stone, P. Causal Dynamics Learning for Task-Independent State Abstraction, June 2022. URL http://arxiv.org/abs/2206. 13452. arXiv:2206.13452 [cs]. Xu, Z. and Tewari, A. Reinforcement Learning in Factored MDPs: Oracle-Efficient Algorithms and Tighter Regret Bounds for the Non-Episodic Setting. In Advances in Neural Information Processing Systems, volume 33, pp. 18226–18236. Curran Associates, Inc., 2020. URL https://proceedings. neurips.cc/paper/2020/hash/ d3b1fb02964aa64e257f9f26a31f72cf-Abstr html. Yoon, J., Wu, Y.-F., Bae, H., and Ahn, S. An Investigation into Pre-Training Object-Centric Representations for Reinforcement Learning, June 2023. URL http:// arxiv.org/abs/2302.04419. arXiv:2302.04419 [cs]. Yu, Z., Ruan, J., and Xing, D. Explainable Reinforcement Learning via a Causal World Model, May 2023. URL http://arxiv.org/abs/2305. 02749. arXiv:2305.02749 [cs]. Zadaianchuk, A., Seitzer, M., and Martius, G. Selfsupervised Visual Reinforcement Learning with Object-centric Representations, November 2020. URL http://arxiv.org/abs/2011.14381. arXiv:2011.14381 [cs]. Zambaldi, V., Raposo, D., Santoro, A., Bapst, V., Li, Y., Babuschkin, I., Tuyls, K., Reichert, D., Lillicrap, T., Lockhart, E., Shanahan, M., Langston, V., Pascanu, R., Botvinick, M., Vinyals, O., and Battaglia, P. Relational Deep Reinforcement Learning, June 2018. URL http://arxiv.org/abs/1806. 01830. arXiv:1806.01830 [cs, stat]. Zeng, Y., Cai, R., Sun, F., Huang, L., and Hao, Z. A Survey on Causal Reinforcement Learning, February 2023. URL http://arxiv.org/abs/2302. 05209. arXiv:2302.05209 [cs].
Zambaldi, V., Raposo, D., Santoro, A., Bapst, V., Li, Y., Babuschkin, I., Tuyls, K., Reichert, D., Lillicrap, T., Lockhart, E., Shanahan, M., Langston, V., Pascanu, R., Botvinick, M., Vinyals, O., and Battaglia, P. Relational Deep Reinforcement Learning, June 2018. URL http://arxiv.org/abs/1806. 01830. arXiv:1806.01830 [cs, stat].
Zeng, Y., Cai, R., Sun, F., Huang, L., and Hao, Z. A Survey on Causal Reinforcement Learning, February 2023. URL http://arxiv.org/abs/2302. 05209. arXiv:2302.05209 [cs].
Zeng, Y., Cai, R., Sun, F., Huang, L., and Hao, Z. A Survey on Causal Reinforcement Learning, February 2023. URL http://arxiv.org/abs/2302. 05209. arXiv:2302.05209 [cs].
Table 6. Symbols used in the paper and appendices
Symbol(s)
Explanation
Si
The i-th state variable in a FMDP.
S′
i
The i-th next-state variable in a FMDP.
S
The group of state variables in a FMDP.
S′
The group of next-state variables in a FMDP.
Ai
The i-th action variable in a FMDP.
A
The group of action variables in a FMDP.
∆
The group of all variables in a transition, i.e. (S,A,S′).
ns
The number of state variables in a FMDP.
na
The number of action variables in a FMDP.
p
The probability distribution of random variables.
ˆp
The estimated distribution for p in a dynamics model.
G
The DAG of a causal model (e.g., a Bayesian network, CDM, or OOCDM).
X →Y
Variable X is a parent of variable Y in some given DAG.
PaG(X)
The parent set of variable X in DAG G.
Pa(X)
The parent set of variable X in the ground-truth causal graph.
C (or Ci)
A (or the i-th) class in an OOMDP.
C
The set of classes in an OOMDP.
F[C]
The set of fields of class C, i.e. Fs[C] ∪Fa[C].
Fs[C]
The set of state fields of class C.
Fa[C]
The set of action fields of class C.
F
The set of all fields in an OOMDP, i.e. ⋃C∈C F[C].
Fs
The set of all state fields in an OOMDP, i.e. ⋃C∈C Fs[C].
C.U
Some filed of C in F[C].
C.V
Some state field of C in Fs[C].
DomC.U
The domain of some field C.U.
O,Oi
An object in an OOMDP.
N
The number of objects in an OOMDP.
K
The number of classes in an OOMDP.
O ∈C
Object O is an instance of class C.
O.U
An attribute of O (derived from the field C.U ∈F[C] where O ∈C).
O.V
A state attribute of O (derived from the field C.S ∈Fs[C] where O ∈C).
O.S
The group of all state attributes of O.
O.A
The group of all action attributes of O.
O
All attributes of O, i.e. (O.S,O.A).
O.V′
The variable of state attribute O.V in the next-step.
O.S′
The group of state variables O.S in the next-step.
Oa ∼Ob
Oa and Ob are instances of the same class.
C.U →V ′
A local causality expression from C.U to C.V .
Cl.U →Ck.V ′
A global causality expression from Cl.U to Ck.V .
D
A dataset of transition samples.
Ct
k
The set of instances of class Ck at step t.
ˆp(⋅)t
The estimation of p when variables take the observed values at step t.
Iς
D
The CMI for class-level causality ς on data D.
fC.V
The predictor for the state field C.V in the OOCDM.
LG(D)
The AILL function of data D under the CG G.
J(D)
The overall target function for model learning.
# A. Acronyms and Symbols
The meanings of symbols used in the paper or will be used in the appendices are described in Table 6 unless otherw specified. The acronyms that appear in our paper are explained in Table 7.
<div style="text-align: center;">Table 7. The meanings of acronyms that appear in the paper.</div>
Acronym
Explanation
AILL
Average instance log-likelihood.
BCG
Bipartite causal graph.
BN
Bayesian Network.
CDM
Causal dynamics model.
CDL
A baseline proposed by Wang et al. (2022).
CG
Causal graph.
CIT
Conditional independence test.
CLCE
Class-level causality expression.
CMS
Collect-Mineral-Shards, a StarCraftII minigame.
CMI
Conditional mutual information.
DAG
Directed acyclic graph.
DBN
Dynamics Bayesian Network.
DZB
Defeat-Zerglings-Banelings, a StarCraftII minigame.
GNN
A baseline based on a graph neural network (Kipf et al., 2020).
i.d.
In-distribution.
MBRL
Model-based reinforcement learning.
MDP
Markov decision process.
MLP
Multi-layer perceptron.
OO
Object-oriented.
OOCDM
Object-oriented causal dynamics model
OOCG
Object-oriented causal graph.
o.o.d.
Out-of-distribution.
OOFULL
Object-oriented full model (a variant of OOCDM that uses full OOCGs).
OOMDP
Object-oriented Markov decision process.
RL
Reinforcement learning.
TICSA
A baseline proposed by Wang et al. (2021).
# B. Basics of Causality
B.1. Causal Models
In this section, we present some of the basic concepts and theorems of causality, which form the foundation of our theory. We first introduce Markov Compatibility (Pearl, 2000), which defines whether a graph can correctly reflect the relationships among variables given a probability function. Definition B.1 (Markov Compatibility). Assume G is an directional acyclic graph (DAG) on a group of random variables X = (X1,...,Xn). Given any probability function p of these variables, if the rule of production decomposition holds:
() = ∏ = then we say that p is compatible with G, or that G represents p.
 G G Causality (the DAG) is a universal concept. The following theorem shows, that no matter what the probability function is, the dependencies between variables can always be represented by some DAG. This leads to a general form of a causal model called the Bayesian Network (BN). Theorem B.2 (Existence of causal graphs). For any probability function p of variables X = (X1,⋯,Xn), there always exists a DAG G that p is compatible with. Proof. Using the chain rule of probability functions, we have p(X1,...,Xn) = p(X1)p(X2∣X1)p(X3∣X1,X2)⋯p(Xn∣X1,...,Xn−1). (12)
(11)
(12)
Letting Pa(Xj) ⊆{X1,...,Xj−1} denote the minimal subset such that p(Xj∣X1,...,Xj−1) = p(Xj∣Pa(Xj)) (the Markovian parents (Pearl, 2000)) for j = 1,...,n, we obtain Eq. 11. Definition B.3 (Bayesian Network). A Bayesian Netowrk is a tuple ⟨G,p⟩, where G is a DAG on a set of random variables X = (X1,...,Xn), and p is a probability function of X such that p is compatible with G. Especially, according to Laplacian’s conception, most stochastic phenomenons in nature are due to deterministic functions combined with unobserved disturbances. This conception leads to a special type of BN called the Structural Causal Model (SCM), which is the most popular model in causal inference. Definition B.4 (Structural Causal Model). A Structural Causal Model is a tuple ⟨G,p,U,F⟩, where ⟨G,p⟩forms a Bayesian Network on variables X = (X1,...,Xn). U = (U1,...,Un) is a set of disturbance variables that are independent of each other. F = {f1,f2,⋯,fn} is a set of structural equations, such that  =(())
# ) Xi = fi(Pa(Xi);Ui).
# B.2. D-Separation
The concept of d-seperation plays an important role in causal inference. Given a DAG G, the criterion of d-separation provides an effective way to determine on what condition two groups of variables are independent. Definition B.5 (d-separation). Assume G is a DAG on a set of variables V. Assume X, Y, and Z are three disjoint groups of variables in V. We say an un-directional path between X and Y is blocked by Z if one of the following requirements is met: 1) The path contains a chain A →B →C or a fork A ←B →C such that B ∈Z; or 2) the path contains a collider A →B ←C such that Z contains no descendent of B. We say X and Y are d-separated by Z, if Z blocks all un-directional paths between X and Y in G, denoted as  � ∣
 G  �G ∣ heorem B.6 (d-separation criterion). Assume G is a DAG on a set of variables V. Assume X, Y, and Z are three disjoint roups of variables in V. We have:  if p is any probability function compatible with G, then
 G (X �G Y ∣Z) ⇒(X �p Y ∣Z),
( �G ∣) ⇒( � ∣) where �p means conditional independence under p, namely p(Y∣Z) = p(Y∣X,Z); 2) if (X �p Y ∣Z) holds for all p that is compatible with G, then (X �G Y ∣Z) also holds
 ( � ∣) G ( �G ∣) Using the d-separation criterion, the following rule is proven by Pearl (2000). Theorem B.7 (Causal Markov Condition). Assume G is a DAG on a set of variables V. Let p denote a probability function for these variables. Then p is compatible with G if and only if (X �p Y∣PaG(X)) holds for any X,Y ∈V such that Y is not a descendant of X.
# B.3. Causal Discovery
Consider that V is a set of variables, and that p is a probability function of these variables. The goal of causal discovery is to recover a DAG G that is compatible with p from a set of observation data (sampled from p) of these variables. However, a probability function p may be compatible with more than one DAG. For example, consider two SCMs on variables {X,Y,Z} where X is the only exogenous variable:   
M ∶ = = + = + + If the distributions of disturbances are the same in both SCMs and UY ≡0, then the two SCMs lead to identical probability functions. Therefore, this probability function is compatible with two different DAGs: In M1, we have Pa(Z) = {X}; in