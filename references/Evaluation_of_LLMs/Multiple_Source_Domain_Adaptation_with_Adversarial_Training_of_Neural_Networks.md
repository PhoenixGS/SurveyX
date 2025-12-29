# Multiple Source Domain Adaptation with Adversarial Learning
Han Zhao†∗ Shanghang Zhang‡∗ Guanhang Wu♮ João P. Costeira♭ José M. F. Moura‡ Geoffrey J. Gordon†
HAN.ZHAO@CS.CMU.EDU SHANGHAZ@ANDREW.CMU.EDU GUANHANW@ANDREW.CMU.EDU JPC@ISR.IST.UTL.PT MOURA@ANDREW.CMU.EDU GGORDON@CS.CMU.EDU
o†∗ HAN.ZHAO@CS.CMU.EDU g Zhang‡∗ SHANGHAZ@ANDREW.CMU.EDU g Wu♮ GUANHANW@ANDREW.CMU.EDU osteira♭ JPC@ISR.IST.UTL.PT  Moura‡ MOURA@ANDREW.CMU.EDU J. Gordon† GGORDON@CS.CMU.EDU
†Machine Learning Department, Carnegie Mellon University, Pittsburgh, PA, USA ‡Department of Electrical and Computer Engineering, Carnegie Mellon University, Pittsburgh, PA, US ♮Robotics Institute, Carnegie Mellon University, Pittsburgh, PA, USA ♭Department of Electrical and Computer Engineering, Instituto Superior Técnico, Lisbon, Portugal
# Abstract
While domain adaptation has been actively researched in recent years, most theoretical results and algorithms focus on the single-source-single-target adaptation setting. Naive application of such algorithms on multiple source domain adaptation problem may lead to suboptimal solutions. As a step toward bridging the gap, we propose a new generalization bound for domain adaptation when there are multiple source domains with labeled instances and one target domain with unlabeled instances. Compared with existing bounds, the new bound does not require expert knowledge about the target distribution, nor the optimal combination rule for multisource domains. Interestingly, our theory also leads to an efficient learning strategy using adversarial neural networks: we show how to interpret it as learning feature representations that are invariant to the multiple domain shifts while still being discriminative for the learning task. To this end, we propose two models, both of which we call multisource domain adversarial networks (MDANs): the first model optimizes directly our bound, while the second model is a smoothed approximation of the first one, leading to a more data-efficient and task-adaptive model. The optimization tasks of both models are minimax saddle point problems that can be optimized by adversarial training. To demonstrate the effectiveness of MDANs, we conduct extensive experiments showing superior adaptation performance on three real-world datasets: sentiment analysis, digit classification, and vehicle counting.
arXiv:1705.09684v2
# 1. Introduction
The success of machine learning algorithms has been partially attributed to rich datasets with abundant annotations (Krizhevsky et al., 2012; Hinton et al., 2012; Russakovsky et al., 2015). Unfortunately, collecting and annotating such large-scale training data is prohibitively expensive and time-consuming. To solve these limitations, different labeled datasets can be combined to build a larger one, or synthetic training data can be generated with explicit yet inexpensive annotations (Shrivastava et al., 2016). However, due to the possible shift between training and test samples, learning algorithms based on these cheaper datasets still suffer from high generalization error. Domain adaptation (DA) focuses on such problems by establishing knowledge transfer from a labeled source domain to an unlabeled target domain, and by exploring domain-invariant structures and representations to bridge the gap (Pan and Yang, 2010). Both theoretical results (Ben-David et al., 2010;
∗. The first two authors contributed equally to this work.
HAN.ZHAO@CS.CMU.EDU SHANGHAZ@ANDREW.CMU.EDU GUANHANW@ANDREW.CMU.EDU JPC@ISR.IST.UTL.PT MOURA@ANDREW.CMU.EDU GGORDON@CS.CMU.EDU
HAN ZHAO, SHANGHANG ZHANG, GUANHANG WU, JOAO COSTEIRA, JOSE MOURA, GEOFFREY GORDON
Mansour et al., 2009a; Mansour and Schain, 2012; Xu and Mannor, 2012) and algorithms (Becker et al., 2013; Hoffman et al., 2012; Ajakan et al., 2014) for DA have been proposed. Recently, DA algorithms based on deep neural networks produce breakthrough performance by learning more transferable features (Glorot et al., 2011; Donahue et al., 2014; Yosinski et al., 2014; Bousmalis et al., 2016; Long et al., 2015). Most theoretical results and algorithms with respect to DA focus on the single-source-single-target adaptation setting (Ganin et al., 2016). However, in many application scenarios, the labeled data available may come from multiple domains with different distributions. As a result, naive application of the single-source-single-target DA algorithms may lead to suboptimal solutions. Such problem calls for an efficient technique for multiple source domain adaptation. In this paper, we theoretically analyze the multiple source domain adaptation problem and propose an adversarial learning strategy based on our theoretical results. Specifically, we prove a new generalization bound for domain adaptation when there are multiple source domains with labeled instances and one target domain with unlabeled instances. Our theoretical results build on the seminal theoretical model for domain adaptation introduced by Ben-David et al. (2010), where a divergence measure, known as the H-divergence, was proposed to measure the distance between two distributions based on a given hypothesis space H. Our new result generalizes the bound (Ben-David et al., 2010, Thm. 2) to the case when there are multiple source domains. The new bound has an interesting interpretation and reduces to (Ben-David et al., 2010, Thm. 2) when there is only one source domain. Technically, we derive our bound by first proposing a generalized H-divergence measure between two sets of distributions from multi-domains. We then prove a PAC bound (Valiant, 1984) for the target risk by bounding it from empirical source risks, using tools from concentration inequalities and the VC theory (Vapnik, 1998). Compared with existing bounds, the new bound does not require expert knowledge about the target domain distribution (Mansour et al., 2009b), nor the optimal combination rule for multiple source domains (Ben-David et al., 2010). Our results also imply that it is not always beneficial to naively incorporate more source domains into training, which we verify to be true in our experiments. Interestingly, our bound also leads to an efficient implementation using adversarial neural networks. This implementation learns both domain invariant and task discriminative feature representations under multiple domains. Specifically, we propose two models (both named MDANs) by using neural networks as rich function approximators to instantiate the generalization bound we derive (Fig. 1). After proper transformations, both models can be viewed as computationally efficient approximations of our generalization bound, so that the goal is to optimize the parameters of the networks in order to minimize the bound. The first model optimizes directly our generalization bound, while the second is a smoothed approximation of the first, leading to a more data-efficient and task-adaptive model. The optimization problem for each model is a minimax saddle point problem, which can be interpreted as a zero-sum game with two participants competing against each other to learn invariant features. Both models combine feature extraction, domain classification, and task learning in one training process. MDANs is generalization of the popular domain adversarial neural network (DANN) (Ganin et al., 2016) and reduce to it when there is only one source domain. We propose to use stochastic optimization with simultaneous updates to optimize the parameters in each iteration. To demonstrate the effectiveness of MDANs as well as the relevance of our theoretical results, we conduct extensive experiments on real-world datasets, including both natural language and vision tasks. We achieve superior adaptation performances on all the tasks, validating the effectiveness of our models.
Mansour et al., 2009a; Mansour and Schain, 2012; Xu and Mannor, 2012) and algorithms (Becker et al., 2013; Hoffman et al., 2012; Ajakan et al., 2014) for DA have been proposed. Recently, DA algorithms based on deep neural networks produce breakthrough performance by learning more transferable features (Glorot et al., 2011; Donahue et al., 2014; Yosinski et al., 2014; Bousmalis et al., 2016; Long et al., 2015). Most theoretical results and algorithms with respect to DA focus on the single-source-single-target adaptation setting (Ganin et al., 2016). However, in many application scenarios, the labeled data available may come from multiple domains with different distributions. As a result, naive application of the single-source-single-target DA algorithms may lead to suboptimal solutions. Such problem calls for an efficient technique for multiple source domain adaptation. In this paper, we theoretically analyze the multiple source domain adaptation problem and propose an adversarial learning strategy based on our theoretical results. Specifically, we prove a new generalization bound for domain adaptation when there are multiple source domains with labeled instances and one target domain with unlabeled instances. Our theoretical results build on the seminal theoretical model for domain adaptation introduced by Ben-David et al. (2010), where a divergence measure, known as the H-divergence, was proposed to measure the distance between two distributions based on a given hypothesis space H. Our new result generalizes the bound (Ben-David et al., 2010, Thm. 2) to the case when there are multiple source domains. The new bound has an interesting interpretation and reduces to (Ben-David et al., 2010, Thm. 2) when there is only one source domain. Technically, we derive our bound by first proposing a generalized H-divergence measure between two sets of distributions from multi-domains. We then prove a PAC bound (Valiant, 1984) for the target risk by bounding it from empirical source risks, using tools from concentration inequalities and the VC theory (Vapnik, 1998). Compared with existing bounds, the new bound does not require expert knowledge about the target domain distribution (Mansour et al., 2009b), nor the optimal combination rule for multiple source domains (Ben-David et al., 2010). Our results also imply that it is not always beneficial to naively incorporate more source domains into training, which we verify to be true in our experiments. Interestingly, our bound also leads to an efficient implementation using adversarial neural networks.
Interestingly, our bound also leads to an efficient implementation using adversarial neural networks. This implementation learns both domain invariant and task discriminative feature representations under multiple domains. Specifically, we propose two models (both named MDANs) by using neural networks as rich function approximators to instantiate the generalization bound we derive (Fig. 1). After proper transformations, both models can be viewed as computationally efficient approximations of our generalization bound, so that the goal is to optimize the parameters of the networks in order to minimize the bound. The first model optimizes directly our generalization bound, while the second is a smoothed approximation of the first, leading to a more data-efficient and task-adaptive model. The optimization problem for each model is a minimax saddle point problem, which can be interpreted as a zero-sum game with two participants competing against each other to learn invariant features. Both models combine feature extraction, domain classification, and task learning in one training process. MDANs is generalization of the popular domain adversarial neural network (DANN) (Ganin et al., 2016) and reduce to it when there is only one source domain. We propose to use stochastic optimization with simultaneous updates to optimize the parameters in each iteration. To demonstrate the effectiveness of MDANs as well as the relevance of our theoretical results, we conduct extensive experiments on real-world datasets, including both natural language and vision tasks. We achieve superior adaptation performances on all the tasks, validating the effectiveness of our models.
# 2. Preliminary
We first introduce the notation used in this paper and review a theoretical model for domain adaptation when there is only one source and one target domain (Kifer et al., 2004; Ben-David et al., 2007; Blitzer et al., 2008; Ben-David et al., 2010). The key idea is the H-divergence to measure the discrepancy between two distributions. Other theoretical models for DA exist (Cortes et al., 2008; Mansour et al., 2009a,c; Cortes and Mohri, 2014); we choose to work with the above model because this distance measure has a particularly natural interpretation and can be well approximated using samples from both domains.
We first introduce the notation used in this paper and review a theoretical model for domain adaptation when there is only one source and one target domain (Kifer et al., 2004; Ben-David et al., 2007; Blitzer et al., 2008; Ben-David et al., 2010). The key idea is the H-divergence to measure the discrepancy between two distributions. Other theoretical models for DA exist (Cortes et al., 2008; Mansour et al., 2009a,c; Cortes and Mohri, 2014); we choose to work with the above model because this distance measure has a particularly natural interpretation and can be well approximated using samples from both domains. Notations We use domain to represent a distribution D on input space X and a labeling function f : X →[0, 1]. In the setting of one source one target domain adaptation, we use ⟨DS, fS⟩and ⟨DT , fT ⟩to denote the source and target domain, respectively. A hypothesis is a binary classification function h : X →{0, 1}. The error of a hypothesis h w.r.t. a labeling function f under distribution DS is defined as: εS(h, f) := Ex∼DS[|h(x) −f(x)|]. When f is also a hypothesis, then this definition reduces to the probability that h disagrees with h under DS: Ex∼DS[|h(x) −f(x)|] = Ex∼DS[I(f(x) ̸= h(x))] = Prx∼DS(f(x) ̸= h(x)). We define the risk of hypothesis h as the error of h w.r.t. a true labeling function under domain DS, i.e., εS(h) := εS(h, fS). As common notation in computational learning theory, we use �εS(h) to denote the empirical risk of h on the source domain. Similarly, we use εT (h) and �εT (h) to mean the true risk and the empirical risk on the target domain. H-divergence is defined as follows: Definition 2.1. Let H be a hypothesis class for instance space X, and AH be the collection of subsets of X that are the support of some hypothesis in H, i.e., AH := {h−1({1}) | h ∈H}. The distance between two distributions D and D′ based on H is:
Notations We use domain to represent a distribution D on input space X and a labeling function f : X →[0, 1]. In the setting of one source one target domain adaptation, we use ⟨DS, fS⟩and ⟨DT , fT ⟩to denote the source and target domain, respectively. A hypothesis is a binary classification function h : X →{0, 1}. The error of a hypothesis h w.r.t. a labeling function f under distribution DS is defined as: εS(h, f) := Ex∼DS[|h(x) −f(x)|]. When f is also a hypothesis, then this definition reduces to the probability that h disagrees with h under DS: Ex∼DS[|h(x) −f(x)|] = Ex∼DS[I(f(x) ̸= h(x))] = Prx∼DS(f(x) ̸= h(x)). We define the risk of hypothesis h as the error of h w.r.t. a true labeling function under domain DS, i.e., εS(h) := εS(h, fS). As common notation in computational learning theory, we use �εS(h) to denote the empirical risk of h on the source domain. Similarly, we use εT (h) and �εT (h) to mean the true risk and the empirical risk on the target domain. H-divergence is defined as follows: Definition 2.1. Let H be a hypothesis class for instance space X, and AH be the collection of subsets of X that are the support of some hypothesis in H, i.e., AH := {h−1({1}) | h ∈H}. The distance between two distributions D and D′ based on H is:
dH(D, D′) := 2 sup A∈AH | Pr D (A) −Pr D′(A)|
When the hypothesis class H contains all the possible measurable functions over X, dH(D, D′) reduces to the familiar total variation. Given a hypothesis class H, we define its symmetric difference w.r.t. itself as: H∆H = {h(x) ⊕h′(x) | h, h′ ∈H}, where ⊕is the xor operation. Let h∗be the optimal hypothesis that achieves the minimum combined risk on both the source and the target domains:
∈H and use λ to denote the combined risk of the optimal hypothesis h∗: λ := εS(h∗) + εT (h∗)
∈H and use λ to denote the combined risk of the optimal hypothesis h∗:
Ben-David et al. (2007) and Blitzer et al. (2008) proved the following generalization bound on th target risk in terms of the source risk and the discrepancy between the source domain and the targe domain:
Ben-David et al. (2007) and Blitzer et al. (2008) proved the following generalization bound on the target risk in terms of the source risk and the discrepancy between the source domain and the target domain: Theorem 2.1 ((Blitzer et al., 2008)). Let H be a hypothesis space of V C-dimension d and US, UT be unlabeled samples of size m each, drawn from DS and DT , respectively. Let �dH∆H be the empirical distance on US and UT ; then with probability at least 1 −δ over the choice of samples, for each h ∈H,  1 � 2d log(2m) + log(4/δ)
(1)
The generalization bound depends on λ, the optimal combined risk that can be achieved by hypothesis in H. The intuition is that if λ is large, then we cannot hope for a successful domain adaptation. One notable feature of this bound is that the empirical discrepancy distance between two samples US and UT can usually be approximated by a discriminator to distinguish instances from these two domains.
# 3. A New Generalization Bound for Multiple Source Domain Adaptation
In this section we first generalize the definition of the discrepancy function dH(·, ·) that is only appropriate when we have two domains. We will then use the generalized discrepancy function to derive a generalization bound for multisource domain adaptation. We conclude this section with a discussion and comparison of our bound and existing generalization bounds for multisource domain adaptation (Mansour et al., 2009c; Ben-David et al., 2010). We refer readers to appendix for proof details and we mainly focus on discussing the interpretations and implications of the theorems. Let {DSi}k i=1 and DT be k source domains and the target domain, respectively. We define the discrepancy function dH(DT ; {DSi}k i=1) induced by H to measure the distance between DT and a set of domains {DSi}k i=1 as follows:
# Definition 3.1.
dH(DT ; {DSi}k i=1) := max i∈[k] dH(DT ; DSi) = 2 max i∈[k] sup A∈AH | Pr DT(A) −Pr DSi (A)|
Again, let h∗be the optimal hypothesis that achieves the minimum combined risk:
# Again, let h∗be the optimal hypothesis that achieves the minimum combined risk:
# h∗:= arg min h∈H � εT (h) + max i∈[k] εSi(h) �
and define
λ := εT (h∗) + max i∈[k] εSi(h∗)
λ := εT (h∗) + max i∈[k] εSi(h∗)
i.e., the minimum risk that is achieved by h∗. The following lemma holds for ∀h ∈H:
em 3.1. εT (h) ≤maxi∈[k] εSi(h) + λ + 1 2dH∆H(DT ; {DSi}k i=1).
Remark. Let us take a closer look at the generalization bound: to make it small, the discrepancy measure between the target domain and the multiple source domains need to be small. Otherwise we cannot hope for successful adaptation by only using labeled instances from the source domains. In this case there will be no hypothesis that performs well on both the source domains and the target domain. It is worth pointing out here that the second term and the third term together introduce a tradeoff (regularization) on the complexity of our hypothesis class H. Namely, if H is too restricted, then the second term λ can be large while the discrepancy term can be small. On the other hand, if H is very rich, then we expect the optimal error, λ, to be small, while the discrepancy measure dH∆H(DT ; {DSi}k i=1) to be large. The first term is a standard source risk term that usually appears in generalization bounds under the PAC-learning framework (Valiant, 1984; Vapnik, 1998). Later we shall upper bound this term by its corresponding empirical risk. The discrepancy distance dH∆H(DT ; {DSi}k i=1) is usually unknown. However, we can bound dH∆H(DT ; {DSi}k i=1) from its empirical estimation using i.i.d. samples from DT and {DSi}k i=1:
Theorem 3.2. Let DT and {DSi}k i=1 be the target distribution and k source distributions over X. Let H be a hypothesis class where V Cdim(H) = d. If �DT and { �DSi}k i=1 are the empirical distributions of DT and {DSi}k i=1 generated with m i.i.d. samples from each domain, then for ϵ > 0, we have:
����  � � ��� � � � � � The main idea of the proof is to use VC theory (Vapnik, 1998) to reduce the infinite hypothesis space to a finite space when acting on finite samples. The theorem then follows from standard union bound and concentration inequalities. Equivalently, the following corollary holds: Corollary 3.1. Let DT and {DSi}k i=1 be the target distribution and k source distributions over X. Let H be a hypothesis class where V Cdim(H) = d. If �DT and { �DSi}k i=1 are the empirical distributions of DT and {DSi}k i=1 generated with m i.i.d. samples from each domain, then, for 0 < δ < 1, with probability at least 1 −δ (over the choice of samples), we have:
����  � � ��� � � � � � The main idea of the proof is to use VC theory (Vapnik, 1998) to reduce the infinite hypothesis spa to a finite space when acting on finite samples. The theorem then follows from standard union boun and concentration inequalities. Equivalently, the following corollary holds:
Corollary 3.1. Let DT and {DSi}k i=1 be the target distribution and k source distributions over X. Let H be a hypothesis class where V Cdim(H) = d. If �DT and { �DSi}k i=1 are the empirical distributions of DT and {DSi}k i=1 generated with m i.i.d. samples from each domain, then, for 0 < δ < 1, with probability at least 1 −δ (over the choice of samples), we have:
���dH(DT ; {DSi}k i=1) −dH( �DT ; { �DSi}k i=1) ���≤2 � 2 m � log 4k δ + d log em d �
���dH(DT ; {DSi}k i=1) −dH( �DT ; { �DSi}k i=1) ���≤2 � 2 m � log 4k δ + d log em d �
��  � � �� Note that multiple source domains do not increase the sample complexity too drastically: it is only the square root of a log term in Corollary. 3.1 where k appears. Similarly, we do not usually have access to the true error maxi∈[k] εSi(h) on the source domains, but we can often have an estimate (maxi∈[k] �εSi(h)) from training samples. We now provide a probabilistic guarantee to bound the difference between maxi∈[k] εSi(h) and maxi∈[k] �εSi(h) uniformly for all h ∈H:
 � Theorem 3.3. Let {DSi}k i=1 be k source distributions over X. Let H be a hypothesis class whe V Cdim(H) = d. If { �DSi}k i=1 are the empirical distributions of {DSi}k i=1 generated with m i.i. samples from each domain, then, for ϵ > 0, we have:
 � Pr � sup h∈H ����max i∈[k] εSi(h) −max i∈[k] �εSi(h) ����≥ϵ � ≤2k �me d �d exp(−2mϵ2
��  � �� Again, Thm. 3.3 can be proved by a combination of concentration inequalities and a reduction from infinite space to finite space, along with the subadditivity of the max function. Equivalently, we hav the following corollary hold:
Corollary 3.2. Let {DSi}k i=1 be k source distributions over X. Let H be a hypothesis class where V Cdim(H) = d. If { �DSi}k i=1 are the empirical distributions of {DSi}k i=1 generated with m i.i.d. samples from each domain, then, for 0 < δ < 1, with probability at least 1 −δ (over the choice of samples), we have:
��  � �� Combining Thm. 3.1 and Corollaries. 3.1, 3.2 and realizing that V Cdim(H∆H) ≤2V Cdim(H) (Anthony and Bartlett, 2009), we have the following theorem:
Theorem 3.4. Let DT and {DSi}k i=1 be the target distribution and k source distributions over X. Let H be a hypothesis class where V Cdim(H) = d. If �DT and { �DSi}k i=1 are the empirical distributions of DT and {DSi}k i=1 generated with m i.i.d. samples from each domain, then, for 0 < δ < 1, with probability at least 1 −δ (over the choice of samples), we have:
 �  � � Remark. Thm. 3.4 has a nice interpretation for each term: the first term measures the worst case accuracy of hypothesis h on the k source domains, and the second term measures the discrepancy between the target domain and the k source domains. For domain adaptation to succeed in the multiple sources setting, we have to expect these two terms to be small: we pick our hypothesis h based on its source training errors, and it will generalize only if the discrepancy between sources and target is small. The third term λ is the optimal error we can hope to achieve. Hence, if λ is large, one should not hope the generalization error to be small by training on the source domains. 1 The last term bounds the additional error we may incur because of the possible bias from finite samples. It is also worth pointing out that these four terms appearing in the generalization bound also capture the tradeoff between using a rich hypothesis class H and a limited one as we discussed above: when using a richer hypothesis class, the first and the third terms in the bound will decrease, while the value of the second term will increase; on the other hand, choosing a limited hypothesis class can decrease the value of the second term, but we may incur additional source training errors and a large λ due to the simplicity of H. One interesting prediction implied by Thm. 3.4 is that the performance on the target domain depends on the worst empirical error among multiple source domains, i.e., it is not always beneficial to naively incorporate more source domains into training. As we will see in the experiment, this is indeed the case in many real-world problems. Comparison with Existing Bounds First, it is easy to see that, upto a multiplicative constant, our bound in (2) reduces to the one in Thm. 2.1 when there is only one source domain (k = 1). Hence Thm. 3.4 can be treated as a generalization of Thm. 2.1. Blitzer et al. (2008) give a generalization bound for semi-supervised multisource domain adaptation where, besides labeled instances from multiple source domains, the algorithm also has access to a fraction of labeled instances from the target domain. Although in general our bound and the one in (Blitzer et al., 2008, Thm. 3) are incomparable, it is instructive to see the connections and differences between them: on one hand, the multiplicative constants of the discrepancy measure and the optimal error in our bound are half of those in Blitzer et al. (2008)’s bound, leading to a tighter bound; on the other hand, because of the access to labeled instances from the target domain, their bound is expressed relative to the optimal error rate on the target domain, while ours is in terms of the empirical error on the source domain. Finally, thanks to our generalized definition of dH(DT ; {DSi}k i=1), we do not need to manually specify the optimal combination vector α in (Blitzer et al., 2008, Thm. 3), which is unknown in practice. Mansour et al. (2009b) also give a generalization bound for multisource domain adaptation under the assumption that the target distribution is a mixture of the k sources and the target hypothesis
1. Of course it is still possible that εT (h) is small while λ is large, but in domain adaptation we do not have ac labeled samples from DT .
can be represented as a convex combination of the source hypotheses. While the distance measure we use assumes 0-1 loss function, their generalized discrepancy measure can also be applied for other losses functions (Mansour et al., 2009a,c,b).
# ultisource Domain Adaptation with Adversarial Neural N
In this section we shall describe a neural network based implementation to minimize the generalization bound we derive in Thm. 3.4. The key idea is to reformulate the generalization bound by a minimax saddle point problem and optimize it via adversarial training.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ba65/ba65061e-754b-4587-8d24-9e0dfe122254.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: MDANs Network architecture. Feature extractor, domain classifier, and task learning are combined in one training process. Hard version: the source that achieves the minimum domain classification error is backpropagated with gradient reversal; Smooth version: all the domain classification risks over k source domains are combined and backpropagated adaptively with gradient reversal.</div>
Suppose we are given samples drawn from k source domains {DSi}, each of which contains m instance-label pairs. Additionally, we also have access to unlabeled instances sampled from the target domain DT . Once we fix our hypothesis class H, the last two terms in the generalization bound (2) will be fixed; hence we can only hope to minimize the bound by minimizing the first two terms, i.e., the maximum source training error and the discrepancy between source domains and target domain. The idea is to train a neural network to learn a representation with the following two properties: 1). indistinguishable between the k source domains and the target domain; 2). informative enough for our desired task to succeed. Note that both requirements are necessary: without the second property, a neural network can learn trivial random noise representations for all the domains, and such representations cannot be distinguished by any discriminator; without the first property, the learned representation does not necessarily generalize to the unseen target domain. Taking these two properties into consideration, we propose the following optimization problem:
# minimize max i∈[k] � �εSi(h) + 1 2dH∆H( �DT ; { �DSi}k i=1) �
One key observation that leads to a practical approximation of dH∆H( �DT ; { �DSi}k i=1) from BenDavid et al. (2007) is that computing the discrepancy measure is closely related to learning a classifier
HAN ZHAO, SHANGHANG ZHANG, GUANHANG WU, JOAO COSTEIRA, JOSE MOURA, GEOFFREY GORDON
     � �  Let �εT,Si(h) be the empirical risk of hypothesis h in the domain discriminating task. Ignoring the constant terms that do not affect the optimization formulation, moving the max operator out, we can reformulate (3) as: � �
     � �  Let �εT,Si(h) be the empirical risk of hypothesis h in the domain discriminating task. Ignoring the constant terms that do not affect the optimization formulation, moving the max operator out, we can
�  � The two terms in (4) exactly correspond to the two criteria we just proposed: the first term asks for an informative feature representation for our desired task to succeed, while the second term captures the notion of invariant feature representations between different domains.
Algorithm 1 Multiple Source Domain Adaptation via Adversarial Training
1: for t = 1 to ∞do
2:
Sample {S(t)
i }k
i=1 and T (t) from { �DSi}k
i=1 and �DT , each of size m
3:
for i = 1 to k do
4:
Compute �ε(t)
i
:= �εS(t)
i (h) −minh′∈H∆H �εT (t),S(t)
i (h′)
5:
Compute w(t)
i
:= exp(�ε(t)
i )
6:
end for
7:
# Hard version
8:
Select i(t) := arg maxi∈[k] �ε(t)
i
9:
Update parameters via backpropagating gradient of �ε(t)
i(t)
10:
# Smoothed version
11:
for i = 1 to k do
12:
Normalize w(t)
i
←w(t)
i / �
i′∈[k] w(t)
i′
13:
end for
14:
Update parameters via backpropagating gradient of �
i∈[k] w(t)
i �ε(t)
i
15: end for
   � Inspired by Ganin et al. (2016), we use the gradient reversal layer to effectively implement (4) by backpropagation. The network architecture is shown in Figure. 1. The pseudo-code is listed in Alg. 1 (the hard version). One notable drawback of the hard version in Alg. 1 is that in each iteration the algorithm only updates its parameter based on the gradient from one of the k domains. This is data inefficient and can waste our computational resources in the forward process. To improve this, we approximate the max function in (4) by the log-sum-exp function, which is a frequently used smooth approximation of the max function. Define �εi(h) := �εSi(h) −minh′∈H∆H �εT,Si(h′):
 � � where γ > 0 is a parameter that controls the accuracy of this approximation. As γ →∞, 1 γ log � i∈[k] exp(γ�εi(h)) →maxi∈[k] �εi(h). Correspondingly, we can formulate a smoothed version
     
(4)
�  � During the optimization, (5) naturally provides an adaptive weighting scheme for the k sour domains depending on their relative error. Use θ to denote all the model parameters, then:
∂ ∂θ 1 γ log � i∈[k] exp � γ(�εSi(h) − min h′∈H∆H �εT,Si(h′)) � = � i∈[k] exp γ�εi(h) � i′∈[k] exp γ�εi′(h) ∂�εi(h) ∂θ
�  � � � The approximation trick not only smooths the objective, but also provides a principled and adaptive way to combine all the gradients from the k source domains. In words, (6) says that the gradient of MDAN is a convex combination of the gradients from all the domains. The larger the error from one domain, the larger the combination weight in the ensemble. We summarize this algorithm in the smoothed version of Alg. 1. Note that both algorithms, including the hard version and the smoothed version, reduce to the DANN algorithm (Ganin et al., 2016) when there is only one source domain.
# 5. Experiments
We evaluate both hard and soft MDANs and compare them with state-of-the-art methods on three real-world datasets: the Amazon benchmark dataset (Chen et al., 2012) for sentiment analysis, a digit classification task that includes 4 datasets: MNIST (LeCun et al., 1998), MNIST-M (Ganin et al., 2016), SVHN (Netzer et al., 2011), and SynthDigits (Ganin et al., 2016), and a public, largescale image dataset on vehicle counting from city cameras (Zhang et al., 2017). Details about network architecture and training parameters of proposed and baseline methods, and detailed dataset description will be introduced in the appendix.
# 5.1 Amazon Reviews
Domains within the dataset consist of reviews on a specific kind of product (Books, DVDs, Electronics, and Kitchen appliances). Reviews are encoded as 5000 dimensional feature vectors of unigrams and bigrams, with binary labels indicating sentiment. We conduct 4 experiments: for each of them, we pick one product as target domain and the rest as source domains. Each source domain has 2000 labeled examples, and the target test set has 3000 to 6000 examples. During training, we randomly sample the same number of unlabeled target examples as the source examples in each mini-batch. We implement the Hard-Max and Soft-Max methods according to Alg. 1, and compare them with three baselines: MLPNet, marginalized stacked denoising autoencoders (mSDA) (Chen et al., 2012), and DANN (Ganin et al., 2016). DANN cannot be directly applied in multiple source domains setting. In order to make a comparison, we use two protocols. The first one is to combine all the source domains into a single one and train it using DANN, which we denote as (cDANN). The second protocol is to train multiple DANNs separately, where each one corresponds to a source-target pair. Among all the DANNs, we report the one achieving the best performance on the target domain. We denote this experiment as (sDANN). For fair comparison, all these models are built on the same basic network structure with one input layer (5000 units) and three hidden layers (1000, 500, 100 units). Results and Analysis We show the accuracy of different methods in Table 1. Clearly, Soft-Max significantly outperforms all other methods in most settings. When Kitchen is the target domain,
(6)
HAN ZHAO, SHANGHANG ZHANG, GUANHANG WU, JOAO COSTEIRA, JOSE MOURA, GEOFFREY GORDO
cDANN performs slightly better than Soft-Max, and all the methods perform close to each other. Hard-Max is typically slightly worse than Soft-Max. This is mainly due to the low data-efficiency of the Hard-Max model (Section 4, Eq. 4, Eq. 5). We argue that with more training iterations, the performance of Hard-Max can be further improved. These results verify the effectiveness of MDANs for multisource domain adaptation. To validate the statistical significance of the results, we run a non-parametric Wilcoxon signed-ranked test for each task to compare Soft-Max with the other competitors, as shown in Table 2. Each cell corresponds to the p-value of a Wilcoxon test between Soft-Max and one of the other methods, under the null hypothesis that the two paired samples have the same mean. From these p-values, we see Soft-Max is convincingly better than other methods.
<div style="text-align: center;">Table 1: Sentiment classification accuracy.</div>
Train/Test
MLPNet
mSDA
sDANN
cDANN
MDANs
H-Max
S-Max
D+E+K/B
0.7655
0.7698
0.7650
0.7789
0.7845
0.7863
B+E+K/D
0.7588
0.7861
0.7732
0.7886
0.7797
0.8065
B+D+K/E
0.8460
0.8198
0.8381
0.8491
0.8483
0.8534
B+D+E/K
0.8545
0.8426
0.8433
0.8639
0.8580
0.8626
<div style="text-align: center;">Table 2: p-values under Wilcoxon test.</div>
MLPNet
mSDA
sDANN
cDANN
H-Max
S-Max
S-Max
S-Max
S-Max
S-Max
B
0.550
0.101
0.521
0.013
0.946
D
0.000
0.072
0.000
0.051
0.000
E
0.066
0.000
0.097
0.150
0.022
K
0.306
0.001
0.001
0.239
0.008
# 5.2 Digits Datasets
Following the setting in (Ganin et al., 2016), we combine four popular digits datasets (MNIST, MNIST-M, SVHN, and SynthDigits) to build the multisource domain dataset. We take each of MNIST-M, SVHN, and MNIST as target domain in turn, and the rest as sources. Each source domain has 20, 000 labeled images and the target test set has 9, 000 examples. We compare Hard-Max and Soft-Max of MDANs with five baselines: i). best-Single-Source. A basic network trained on each source domain (20, 000 images) without domain adaptation and tested on the target domain. Among the three models, we report the one achieves the best performance on the test set. ii). Combine-Source. A basic network trained on a combination of three source domains (20, 000 images for each) without domain adaptation and tested on the target domain. iii). best-Single-DANN. We train DANNs (Ganin et al., 2016) on each source-target domain pair (20, 000 images) and test it on target. Again, we report the best score among the three. iv). Combine-DANN. We train a single DANN on a combination of three source domains (20, 000 images for each). v). Target-only. It is the basic network trained and tested on the target data. It serves as an upper bound of DA algorithms. All the MDANs and baseline methods are built on the same basic network structure to put them on a equal footing.
Results and Analysis The classification accuracy is shown in Table 3. The results show that a naive combination of different training datasets can sometimes even decrease the performance. Furthermore, we observe that adaptation to the SVHN dataset (the third experiment) is hard. In this case, increasing the number of source domains does not help. We conjecture this is due to the large dissimilarity between the SVHN data to the others. For the combined sources, MDANs always perform better than the source-only baseline (MDANs vs. Combine-Source). However, directly training DANN on a combination of multiple sources leads to worse performance when compared with our approach (Combine-DANN vs. MDANs). In fact, this strategy may even lead to worse results than the source-only baseline (Combine-DANN vs. Combine-Source). Surprisingly, using a single domain (best-Single DANN) can sometimes achieve the best result. This means that in domain adaptation the quality of data (how close to the target data) is much more important than the quantity (how many source domains). As a conclusion, this experiment further demonstrates the effectiveness of MDANs when there are multiple source domains available, where a naive combination of multiple sources using DANN may hurt generalization.
ble 3: Accuracy on digit classification. Mt: MNIST; Mm: MNIST-M, Sv: SVHN, Sy: SynthDigits.
<div style="text-align: center;">Table 3: Accuracy on digit classification. Mt: MNIST; Mm: MNIST-M, Sv: SVHN, Sy: SynthDigits.</div>
Train/Test
best-Single
Source
best-Single
DANN
Combine
Source
Combine
DANN
MDAN
Target
Only
Hard-Max
Soft-Max
Sv+Mm+Sy/Mt
0.964
0.967
0.938
0.925
0.976
0.979
0.987
Mt+Sv+Sy/Mm
0.519
0.591
0.561
0.651
0.663
0.687
0.901
Mm+Mt+Sy/Sv
0.814
0.818
0.771
0.776
0.802
0.816
0.898
S
T
MDANs
DANN
FCN
T
MDANs
DANN
FCN
Hard-Max
Soft-Max
Hard-Max
Soft-Max
2
A
1.8101
1.7140
1.9490
1.9094
B
2.5059
2.3438
2.5218
2.6528
3
A
1.3276
1.2363
1.3683
1.5545
B
1.9092
1.8680
2.0122
2.4319
4
A
1.3868
1.1965
1.5520
1.5499
B
1.7375
1.8487
2.1856
2.2351
5
A
1.4021
1.1942
1.4156
1.7925
B
1.7758
1.6016
1.7228
2.0504
6
A
1.4359
1.2877
2.0298
1.7505
B
1.5912
1.4644
1.5484
2.2832
7
A
1.4381
1.2984
1.5426
1.7646
B
1.5989
1.5126
1.5397
1.7324
# 5.3 WebCamT Vehicle Counting Dataset
WebCamT is a public dataset for vehicle counting from large-scale city camera videos, which has low resolution (352 × 240), low frame rate (1 frame/second), and high occlusion. It has 60, 000 frames annotated with vehicle bounding box and count, divided into training and testing sets, with 42, 200 and 17, 800 frames, respectively. Here we demonstrate the effectiveness of MDANs to count vehicles from an unlabeled target camera by adapting from multiple labeled source cameras: we select 8 cameras that each has more than 2, 000 labeled images for our evaluations. As shown in Fig. 3, they are located in different intersections of the city with different scenes. Among these 8 cameras, we randomly pick two cameras and take each camera as the target camera, with the other 7 cameras as sources. We compute the proxy A-distance (PAD) (Ben-David et al., 2007) between each source camera and the target camera to approximate the divergence between them. We then rank the source cameras by the PAD from low to high and choose the first k cameras to form the k source
HAN ZHAO, SHANGHANG ZHANG, GUANHANG WU, JOAO COSTEIRA, JOSE MOURA, GEOFFREY GORDON
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f7ae/f7aec5ce-13f8-49a3-9263-7d29001b9277.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Counting results for target camera A (first row) and B (second row). X-frames; Y-Counts.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e1a3/e1a3bc76-6a9d-41a5-a538-7047113adc5e.png" style="width: 50%;"></div>
<div style="text-align: center;">mera map. Figure 4: Counting error over different source numb</div>
domains. Thus the proposed methods and baselines can be evaluated on different numbers of sources (from 2 to 7). We implement the Hard-Max and Soft-Max MDANs according to Alg. 1, based on the basic vehicle counting network FCN (Zhang et al., 2017). We compare our method with two baselines: FCN (Zhang et al., 2017), a basic network without domain adaptation, and DANN (Ganin et al., 2016), implemented on top of the same basic network. We record mean absolute error (MAE) between true count and estimated count. Results and Analysis The counting error of different methods is compared in Table 4. The HardMax version achieves lower error than DANN and FCN in most settings for both target cameras. The Soft-Max approximation outperforms all the baselines and the Hard-Max in most settings, demonstrating the effectiveness of the smooth and adaptative approximation. The lowest MAE achieved by Soft-Max is 1.1942. Such MAE means that there is only around one vehicle miscount for each frame (the average number of vehicles in one frame is around 20). Fig. 2 shows the counting results of Soft-Max for the two target cameras under the 5 source cameras setting. We can see that the proposed method accurately counts the vehicles of each target camera for long time sequences. Does adding more source cameras always help improve the performance on the target camera? To answer this question, we analyze the counting error when we vary the number of source cameras as shown in Fig. 4. From the curves, we see the counting error goes down with more source cameras at the beginning, while it goes up when more sources are added at the end. This phenomenon corresponds to the prediction implied by Thm. 3.4 (the last remark in Section 3): the performance on the target domain depends on the worst empirical error among multiple source domains, i.e., it is not always
domains. Thus the proposed methods and baselines can be evaluated on different numbers of sources (from 2 to 7). We implement the Hard-Max and Soft-Max MDANs according to Alg. 1, based on the basic vehicle counting network FCN (Zhang et al., 2017). We compare our method with two baselines: FCN (Zhang et al., 2017), a basic network without domain adaptation, and DANN (Ganin et al., 2016), implemented on top of the same basic network. We record mean absolute error (MAE) between true count and estimated count.
Results and Analysis The counting error of different methods is compared in Table 4. The HardMax version achieves lower error than DANN and FCN in most settings for both target cameras. The Soft-Max approximation outperforms all the baselines and the Hard-Max in most settings, demonstrating the effectiveness of the smooth and adaptative approximation. The lowest MAE achieved by Soft-Max is 1.1942. Such MAE means that there is only around one vehicle miscount for each frame (the average number of vehicles in one frame is around 20). Fig. 2 shows the counting results of Soft-Max for the two target cameras under the 5 source cameras setting. We can see that the proposed method accurately counts the vehicles of each target camera for long time sequences. Does adding more source cameras always help improve the performance on the target camera? To answer this question, we analyze the counting error when we vary the number of source cameras as shown in Fig. 4. From the curves, we see the counting error goes down with more source cameras at the beginning, while it goes up when more sources are added at the end. This phenomenon corresponds to the prediction implied by Thm. 3.4 (the last remark in Section 3): the performance on the target domain depends on the worst empirical error among multiple source domains, i.e., it is not always
beneficial to naively incorporate more source domains into training. To illustrate this prediction better, we show the PAD of the newly added camera (when the source number increases by one) in Fig. 4. By observing the PAD and the counting error, we see the performance on the target can degrade when the newly added source camera has large divergence from the target camera.
# 6. Related Work
A number of adaptation approaches have been studied in recent years. From the theoretical aspect, several theoretical results have been derived in the form of upper bounds on the generalization target error by learning from the source data. A keypoint of the theoretical frameworks is estimating the distribution shift between source and target. Kifer et al. (2004) proposed the H-divergence to measure the similarity between two domains and derived a generalization bound on the target domain using empirical error on the source domain and the H-divergence between the source and the target. This idea has later been extended to multisource domain adaptation (Blitzer et al., 2008) and the corresponding generalization bound has been developed as well. Ben-David et al. (2010) provide a generalization bound for domain adaptation on the target risk which generalizes the standard bound on the source risk. This work formalizes a natural intuition of DA: reducing the two distributions while ensuring a low error on the source domain and justifies many DA algorithms. Based on this work, Mansour et al. (2009a) introduce a new divergence measure: discrepancy distance, whose empirical estimate is based on the Rademacher complexity (Koltchinskii, 2001) (rather than the VC-dim). Other theoretical works have also been studied such as (Mansour and Schain, 2012) that derives the generalization bounds on the target error by taking use of the robustness properties introduced in (Xu and Mannor, 2012). See (Cortes et al., 2008; Mansour et al., 2009a,c) for more details. Following the theoretical developments, many DA algorithms have been proposed, such as instancebased methods (Tsuboi et al., 2009); feature-based methods (Becker et al., 2013); and parameterbased methods (Evgeniou and Pontil, 2004). The general approach for domain adaptation starts from algorithms that focus on linear hypothesis class (Blitzer et al., 2006; Germain et al., 2013; Cortes and Mohri, 2014). The linear assumption can be relaxed and extended to the non-linear setting using the kernel trick, leading to a reweighting scheme that can be efficiently solved via quadratic programming (Huang et al., 2006; Gong et al., 2013). Recently, due to the availability of rich data and powerful computational resources, non-linear representations and hypothesis classes have been increasingly explored (Glorot et al., 2011; Baktashmotlagh et al., 2013; Chen et al., 2012; Ajakan et al., 2014; Ganin et al., 2016). This line of work focuses on building common and robust feature representations among multiple domains using either supervised neural networks (Glorot et al., 2011), or unsupervised pretraining using denoising auto-encoders (Vincent et al., 2008, 2010). Recent studies have shown that deep neural networks can learn more transferable features for DA (Glorot et al., 2011; Donahue et al., 2014; Yosinski et al., 2014). Bousmalis et al. (2016) develop domain separation networks to extract image representations that are partitioned into two subspaces: domain private component and cross-domain shared component. The partitioned representation is utilized to reconstruct the images from both domains, improving the DA performance. Reference (Long et al., 2015) enables classifier adaptation by learning the residual function with reference to the target classifier. The main-task of this work is limited to the classification problem. Ganin et al. (2016) propose a domain-adversarial neural network to learn the domain indiscriminate but main-task discriminative features. Although these works generally outperform non-deep learning
HAN ZHAO, SHANGHANG ZHANG, GUANHANG WU, JOAO COSTEIRA, JOSE MOURA, GEOFFREY GORDON
based methods, they only focus on the single-source-single-target DA problem, and much work is rather empirical design without statistical guarantees. Hoffman et al. (2012) present a domain transform mixture model for multisource DA, which is based on non-deep architectures and is difficult to scale up. Adversarial training techniques that aim to build feature representations that are indistinguishable between source and target domains have been proposed in the last few years (Ajakan et al., 2014; Ganin et al., 2016). Specifically, one of the central ideas is to use neural networks, which are powerful function approximators, to approximate a distance measure known as the H-divergence between two domains (Kifer et al., 2004; Ben-David et al., 2007, 2010). The overall algorithm can be viewed as a zero-sum two-player game: one network tries to learn feature representations that can fool the other network, whose goal is to distinguish representations generated from the source domain between those generated from the target domain. The goal of the algorithm is to find a Nash-equilibrium of the game, or the stationary point of the min-max saddle point problem. Ideally, at such equilibrium state, feature representations from the source domain will share the same distributions as those from the target domain, and, as a result, better generalization on the target domain can be expected by training models using only labeled instances from the source domain.
# 7. Conclusion
We derive a new generalization bound for DA under the setting of multiple source domains with labeled instances and one target domain with unlabeled instances. The new bound has interesting interpretation and reduces to an existing bound when there is only one source domain. Following our theoretical results, we propose MDANs to learn feature representations that are invariant under multiple domain shifts while at the same time being discriminative for the learning task. Both hard and soft versions of MDANs are generalizations of the popular DANN to the case when multiple source domains are available. Empirically, MDANs outperform the state-of-the-art DA methods on three real-world datasets, including a sentiment analysis task, a digit classification task, and a visual vehicle counting task, demonstrating its effectiveness for multisource domain adaptation.
# References
H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, and M. Marchand. Domain-adversarial neural networks. arXiv preprint arXiv:1412.4446, 2014. M. Anthony and P. L. Bartlett. Neural network learning: Theoretical foundations. cambridge university press, 2009. P. Arbelaez, M. Maire, C. Fowlkes, and J. Malik. Contour detection and hierarchical image segmentation. IEEE transactions on pattern analysis and machine intelligence, 33(5):898–916, 2011. M. Baktashmotlagh, M. T. Harandi, B. C. Lovell, and M. Salzmann. Unsupervised domain adaptation by domain invariant projection. In Proceedings of the IEEE International Conference on Computer Vision, pages 769–776, 2013. C. J. Becker, C. M. Christoudias, and P. Fua. Non-linear domain adaptation with boosting. In Advances in Neural Information Processing Systems, pages 485–493, 2013.
S. Ben-David, J. Blitzer, K. Crammer, F. Pereira, et al. Analysis of representations for domain adaptation. Advances in neural information processing systems, 19:137, 2007. S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. W. Vaughan. A theory of learning from different domains. Machine learning, 79(1-2):151–175, 2010. J. Blitzer, R. McDonald, and F. Pereira. Domain adaptation with structural correspondence learning. In Proceedings of the 2006 conference on empirical methods in natural language processing, pages 120–128. Association for Computational Linguistics, 2006. J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. Wortman. Learning bounds for domain adaptation. In Advances in neural information processing systems, pages 129–136, 2008. K. Bousmalis, G. Trigeorgis, N. Silberman, D. Krishnan, and D. Erhan. Domain separation networks. In Advances in Neural Information Processing Systems, pages 343–351, 2016. M. Chen, Z. Xu, K. Weinberger, and F. Sha. Marginalized denoising autoencoders for domain adaptation. arXiv preprint arXiv:1206.4683, 2012. C. Cortes and M. Mohri. Domain adaptation and sample bias correction theory and algorithm for regression. Theoretical Computer Science, 519:103–126, 2014. C. Cortes, M. Mohri, M. Riley, and A. Rostamizadeh. Sample selection bias correction theory. In International Conference on Algorithmic Learning Theory, pages 38–53. Springer, 2008. J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, and T. Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. In Icml, volume 32, pages 647–655, 2014. T. Evgeniou and M. Pontil. Regularized multi–task learning. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 109–117. ACM, 2004. Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky. Domain-adversarial training of neural networks. Journal of Machine Learning Research, 17(59):1–35, 2016. P. Germain, A. Habrard, F. Laviolette, and E. Morvant. A pac-bayesian approach for domain adaptation with specialization to linear classifiers. In ICML (3), pages 738–746, 2013. X. Glorot, A. Bordes, and Y. Bengio. Domain adaptation for large-scale sentiment classification: A deep learning approach. In Proceedings of the 28th international conference on machine learning (ICML-11), pages 513–520, 2011. B. Gong, K. Grauman, and F. Sha. Connecting the dots with landmarks: Discriminatively learning domain-invariant features for unsupervised domain adaptation. In ICML (1), pages 222–230, 2013. G. Hinton, L. Deng, D. Yu, G. E. Dahl, A.-r. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. N. Sainath, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. IEEE Signal Processing Magazine, 29(6):82–97, 2012.
J. Hoffman, B. Kulis, T. Darrell, and K. Saenko. Discovering latent domains for multisource domain adaptation. In Computer Vision–ECCV 2012, pages 702–715. Springer, 2012. J. Huang, A. Gretton, K. M. Borgwardt, B. Schölkopf, and A. J. Smola. Correcting sample selection bias by unlabeled data. In Advances in neural information processing systems, pages 601–608, 2006. D. Kifer, S. Ben-David, and J. Gehrke. Detecting change in data streams. In Proceedings of the Thirtieth international conference on Very large data bases-Volume 30, pages 180–191. VLDB Endowment, 2004. V. Koltchinskii. Rademacher penalties and structural risk minimization. IEEE Transactions on Information Theory, 47(5):1902–1914, 2001. A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. M. Long, Y. Cao, J. Wang, and M. Jordan. Learning transferable features with deep adaptation networks. In International Conference on Machine Learning, pages 97–105, 2015. Y. Mansour and M. Schain. Robust domain adaptation. In ISAIM, 2012. Y. Mansour, M. Mohri, and A. Rostamizadeh. Domain adaptation: Learning bounds and algorithms. arXiv preprint arXiv:0902.3430, 2009a. Y. Mansour, M. Mohri, and A. Rostamizadeh. Domain adaptation with multiple sources. In Advances in neural information processing systems, pages 1041–1048, 2009b. Y. Mansour, M. Mohri, and A. Rostamizadeh. Multiple source adaptation and the rényi divergence. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, pages 367–374. AUAI Press, 2009c. Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng. Reading digits in natural images with unsupervised feature learning. In NIPS workshop on deep learning and unsupervised feature learning, volume 2011, page 5, 2011. S. J. Pan and Q. Yang. A survey on transfer learning. IEEE Transactions on knowledge and data engineering, 22(10):1345–1359, 2010. O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211–252, 2015. A. Shrivastava, T. Pfister, O. Tuzel, J. Susskind, W. Wang, and R. Webb. Learning from simulated and unsupervised images through adversarial training. arXiv preprint arXiv:1612.07828, 2016. Y. Tsuboi, H. Kashima, S. Hido, S. Bickel, and M. Sugiyama. Direct density ratio estimation for large-scale covariate shift adaptation. Journal of Information Processing, 17:138–155, 2009.
L. G. Valiant. A theory of the learnable. Communications of the ACM, 27(11):1134–1142, 1984. V. Vapnik. Statistical learning theory, volume 1. Wiley New York, 1998. P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th international conference on Machine learning, pages 1096–1103. ACM, 2008. P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, and P.-A. Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of Machine Learning Research, 11(Dec):3371–3408, 2010. H. Xu and S. Mannor. Robustness and generalization. Machine learning, 86(3):391–423, 2012. J. Yosinski, J. Clune, Y. Bengio, and H. Lipson. How transferable are features in deep neural networks? In Advances in neural information processing systems, pages 3320–3328, 2014. S. Zhang, G. Wu, J. P. Costeira, and J. M. Moura. Understanding traffic density from large-scale web camera data. arXiv preprint arXiv:1703.05868, 2017.
HAN ZHAO, SHANGHANG ZHANG, GUANHANG WU, JOAO COSTEIRA, JOSE MOURA, GEOFFREY GORDON
Appendix A. Outline
# Appendix A. Outline
Organization of the appendix: 1). For the convenience of exposition in showing our technical proofs, we first introduce the technical tools that will be used during our proofs in Sec. B. 2). We provide detailed proofs for all the claims, lemmas and theorems presented in the main paper in Sec. C. 3). We describe more experiment details in Sec. D, including dataset description, network architecture and training parameters of the proposed and baseline methods, and more analysis of the experimental results.
# Appendix B. Technical Tools
∀m ∈N, ΠH(m) = max Xm⊆X |{(h(x1), . . . , h(xm)) | h ∈H}|
  where Xm = {x1, . . . , xm} is a subset of X with size m.
Roughly, the growth function ΠH(m) computes the maximum number of distinct ways in which m points can be classified using hypothesis in H. A closely related concept is the Vapnik–Chervonenkis dimension (VC dimension) (Vapnik, 1998): Definition B.2 (VC dimension). The VC-dimension of a hypothesis class H is defined as: V Cdim(H) = max{m : ΠH(m) = 2m} A well-known result relating V Cdim(H) and the growth function ΠH(m) is the Sauer’s lemma: Lemma B.1 (Sauer’s lemma). Let H be a hypothesis class with V Cdim(H) = d. Then, for m ≥d, the following inequality holds:
Roughly, the growth function ΠH(m) computes the maximum number of distinct ways in which m points can be classified using hypothesis in H. A closely related concept is the Vapnik–Chervonenkis dimension (VC dimension) (Vapnik, 1998): Definition B.2 (VC dimension). The VC-dimension of a hypothesis class is defined as:
Roughly, the growth function ΠH(m) computes the maximum number of distinct ways in which m points can be classified using hypothesis in H. A closely related concept is the Vapnik–Chervonenkis dimension (VC dimension) (Vapnik, 1998): Definition B.2 (VC dimension). The VC-dimension of a hypothesis class H is defined as: V Cdim(H) = max{m : Π(m) = 2m}
A well-known result relating V Cdim(H) and the growth function ΠH(m) is the Sauer’s lemma: Lemma B.1 (Sauer’s lemma). Let H be a hypothesis class with V Cdim(H) = d. Then, for m ≥d, the following inequality holds:
following concentration inequality will be used:
Theorem B.1 (Hoeffding’s inequality). Let X1, . . . , Xn be independent random variables where each Xi is bounded by the interval [ai, bi]. Define the empirical mean of these random variables by ¯X := 1 n �n i=1 Xi, then ∀ε > 0:
��� �� � �   The VC inequality allows us to give a uniform bound on the binary classification error of a hypothesis class H using growth function:
��� �� � �   The VC inequality allows us to give a uniform bound on the binary classification error of a hypothesis class H using growth function: Theorem B.2 (VC inequality). Let ΠH be the growth function of hypothesis class H. For h ∈H, let ε(h) be the true risk of h w.r.t. the generation distribution D and the true labeling function h∗. Similarly, let ˆεn(h) be the empirical risk on a random i.i.d. sample containing n instances from D, then, for ∀ε > 0, the following inequality hold: � �
Theorem B.2 (VC inequality). Let ΠH be the growth function of hypothesis class H. For h ∈H, let ε(h) be the true risk of h w.r.t. the generation distribution D and the true labeling function h∗ Similarly, let ˆεn(h) be the empirical risk on a random i.i.d. sample containing n instances from D, then, for ∀ε > 0, the following inequality hold:
Pr � sup h∈H |ε(h) −ˆεn(h)| ≥ε � ≤8ΠH(n) exp � −nε2/32 �
Although the above theorem is stated for binary classification error, we can extend it to any bounded error. This will only change the multiplicative constant of the bound.
# Appendix C. Proofs
For all the proofs presented here, the following lemma shown by Blitzer et al. (2008) will b repeatedly used: Lemma C.1 ((Blitzer et al., 2008)). ∀h, h′ ∈H, |εS(h, h′) −εT (h, h′)| ≤1 2dH∆H(DS, DT ).
# C.1 Proof of Thm. 3.1
One technical lemma we will frequently use to prove Thm. 3.1 is the triangular inequality w.r.t. εD(h), ∀h ∈H: Lemma C.2. For any hypothesis class H and any distribution D on X, the following triangular inequality holds: ∀h, h′, f ∈H, εD(h, h′) ≤εD(h, f) + εD(f, h′)
Proof.
Now we are ready to prove Thm. 3.1: Theorem 3.1. εT (h) ≤maxi∈[k] εSi(h) + λ + 1 2dH∆H(DT ; {DSi}k i=1). Proof. ∀h ∈H, define ih := arg maxi∈[k] εSi(h, h∗):
Now we are ready to prove Thm. 3.1:
The first and the fifth inequalities are due to the triangle inequality, and the third inequality is based on Lemma C.1. The second holds due to the property of | · | and the others follow by the definition of H-divergence. ■
# C.2 Proof of Thm. 3.2
Theorem 3.2. Let DT and {DSi}k i=1 be the target distribution and k source distributions over X. Le H be a hypothesis class where V Cdim(H) = d. If �DT and { �DSi}k i=1 are the empirical distribution of DT and {DSi}k i=1 generated with m i.i.d. samples from each domain, then for ϵ > 0, we have:
Pr ����dH(DT ; {DSi}k i=1) −dH( �DT ; { �DSi}k i=1) ���≥ϵ � ≤4k �em d �d exp � −mϵ2/8 �
Proof.
The first inequality holds due to the sub-additivity of the max function, and the second inequality is due to the union bound. The third inequality holds because of the triangle inequality, and we use the averaging argument to establish the fourth inequality. The fifth inequality is an application of the VC-inequality, and the sixth is by the Hoeffding’s inequality. Finally, we use the Sauer’s lemma to prove the last inequality. ■
# C.3 Proof of Thm. 3.3
We now show the detailed proof of Thm. 3.3.
Proof.
� � Again, the first inequality is due to the subadditivity of the max function, and the second inequality holds due to the union bound. We apply the VC-inequality to bound the third inequality, and Hoeffding’s inequality to bound the fourth. Again, the last one is due to Sauer’s lemma. ■
# C.4 Derivation of the Discrepancy Distance as Classification Error
We show that the H-divergence is equivalent to a binary classification accuracy in discriminating instances from different domains. Suppose AH is symmetric, i.e., A ∈AH ⇔X\A ∈AH, and we have samples {Si}k i=1 and T from {DSi}k i=1 and DT respectively, each of which is of size m, then:
= max i∈[k] sup h∈H∆H | Pr x∼ˆDT (h(x) = 1) − Pr x∼ˆDSi (h(x = 1))| �
= max i∈[k]   1 −2 min h∈H∆H   1 2m � x∼ˆDT I(h(x) = 1) + 1 2m � x∼ˆDSi I(h(x = 0))      
# Appendix D. Details about Experiments
In this section, we describe more details about the datasets and the experimental settings. We extensively evaluate the proposed methods on three datasets: 1). We first evaluate our methods on Amazon Reviews dataset (Chen et al., 2012) for sentiment analysis. 2). We evaluate the proposed methods on the digits classification datasets including MNIST (LeCun et al., 1998), MNIST-M (Ganin et al., 2016), SVHN (Netzer et al., 2011), and SynthDigits (Ganin et al., 2016). 3). We further evaluate the proposed methods on the public dataset WebCamT (Zhang et al., 2017) for vehicle counting. It contains 60,000 labeled images from 12 city cameras with different distributions. Due to the substantial difference between these datasets and their corresponding learning tasks, we will introduce more detailed dataset description, network architecture, and training parameters for each dataset respectively in the following subsections.
HAN ZHAO, SHANGHANG ZHANG, GUANHANG WU, JOAO COSTEIRA, JOSE MOURA, GEOFFREY GORDON
<div style="text-align: center;">Table 5: Network parameters for proposed and baseline methods</div>
Method
Input layer
Hidden layers
Epochs
Dropout
Domains
Domain adaptation
weight
Gamma
MLPNet
5000
(1000, 500, 100)
50
0.01
N/A
N/A
N/A
DANN
5000
(1000, 500, 100)
50
0.01
1
0.01
N/A
MDAN
5000
(1000, 500, 100)
50
0.7
3
0.1
10
# D.1 Details on Amazon Reviews evaluation
Amazon reviews dataset includes four domains, each one composed of reviews on a specific kind of product (Books, DVDs, Electronics, and Kitchen appliances). Reviews are encoded as 5000 dimensional feature vectors of unigrams and bigrams. The labels are binary: 0 if the product is ranked up to 3 stars, and 1 if the product is ranked 4 or 5 stars.
dimensional feature vectors of unigrams and bigrams. The labels are binary: 0 if the product is ranked up to 3 stars, and 1 if the product is ranked 4 or 5 stars. We take one product domain as target and the other three as source domains. Each source domain has 2000 labeled examples and the target test set has 3000 to 6000 examples. We implement the Hard-Max and Soft-Max methods according to Alg. 1, based on a basic network with one input layer (5000 units) and three hidden layers (1000, 500, 100 units). The network is trained for 50 epochs with dropout rate 0.7. We compare Hard-Max and Soft-Max with three baselines: Baseline 1: MLPNet. It is the basic network of our methods (one input layer and three hidden layers), trained for 50 epochs with dropout rate 0.01. Baseline 2: Marginalized Stacked Denoising Autoencoders (mSDA) (Chen et al., 2012). It takes the unlabeled parts of both source and target samples to learn a feature map from input space to a new representation space. As a denoising autoencoder algorithm, it finds a feature representation from which one can (approximately) reconstruct the original features of an example from its noisy counterpart. Baseline 3: DANN. We implement DANN based on the algorithm described in (Ganin et al., 2016) with the same basic network as our methods. Hyper parameters of the proposed and baseline methods are selected by cross validation. Table 5 summarizes the network architecture and some hyper parameters.
# D.2 Details on Digit Datasets evaluation
We evaluate the proposed methods on the digits classification problem. Following the experiments in (Ganin et al., 2016), we combine four popular digits datasets-MNIST, MNIST-M, SVHN, and SynthDigits to build the multi-source domain dataset. MNIST is a handwritten digits database with 60, 000 training examples, and 10, 000 testing examples. The digits have been size-normalized and centered in a 28 × 28 image. MNIST-M is generated by blending digits from the original MNIST set over patches randomly extracted from color photos from BSDS500 (Arbelaez et al., 2011; Ganin et al., 2016). It has 59, 001 training images and 9, 001 testing images with 32 × 32 resolution. An output sample is produced by taking a patch from a photo and inverting its pixels at positions corresponding to the pixels of a digit. For DA problems, this domain is quite distinct from MNIST, for the background and the strokes are no longer constant. SVHN is a real-world house number dataset with 73, 257 training images and 26, 032 testing images. It can be seen as similar to MNIST, but comes from a significantly harder, unsolved, real world problem. SynthDigits consists of 500; 000 digit images generated by Ganin et al. (2016) from WindowsTM fonts by varying the text, positioning, orientation, background and stroke colors, and the amount of blur. The degrees
of variation were chosen to simulate SVHN, but the two datasets are still rather distinct, with the biggest difference being the structured clutter in the background of SVHN images. We take MNIST-M, SVHN, and MNIST as target domain in turn, and the remaining three as sources. We implement the Hard-Max and Soft-Max versions according to Alg. 1 based on a basic network, as shown in Fig. 5. The baseline methods are also built on the same basic network structure to put them on a equal footing. The network structure and parameters of MDANs are illustrated in Fig. 5. The learning rate is initialized by 0.01 and adjusted by the first and second order momentum in the training process. The domain adaptation parameter of MDANs is selected by cross validation. In each mini-batch of MDANs training process, we randomly sample the same number of unlabeled target images as the number of the source images.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ad67/ad67b2d7-98de-4ec8-aa22-9ce004899418.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: MDANs network architecture for digit classification</div>
# D.3 Details on WebCamT Vehicle Counting
WebCamT is a public dataset for large-scale city camera videos, which have low resolution (352 × 240), low frame rate (1 frame/second), and high occlusion. WebCamT has 60, 000 frames annotated with rich information: bounding box, vehicle type, vehicle orientation, vehicle count, vehicle reidentification, and weather condition. The dataset is divided into training and testing sets, with 42,200 and 17,800 frames, respectively, covering multiple cameras and different weather conditions. WebCamT is an appropriate dataset to evaluate domain adaptation methods, for it covers multiple city cameras and each camera is located in different intersection of the city with different perspectives and scenes. Thus, each camera data has different distribution from others. The dataset is quite challenging and in high demand of domain adaptation solutions, as it has 6, 000, 000 unlabeled images from 200 cameras with only 60, 000 labeled images from 12 cameras. The experiments on WebCamT provide an interesting application of our proposed MDANs: when dealing with spatially and temporally large-scale dataset with much variations, it is prohibitively expensive and time-consuming to label large amount of instances covering all the variations. As a result, only a limited portion of the dataset can be annotated, which can not cover all the data domains in the dataset. MDAN provide an effective
<div style="text-align: center;"></div>
HAN ZHAO, SHANGHANG ZHANG, GUANHANG WU, JOAO COSTEIRA, JOSE MOURA, GEOFFREY GORDON
solution for this kind of application by adapting the deep model from multiple source domains to the
solution for this kind of application by adapting the deep model from multiple source domains to the unlabeled target domain.
We evaluate the proposed methods on different numbers of source cameras. Each source camera provides 2000 labeled images for training and the test set has 2000 images from the target camera. In each mini-batch, we randomly sample the same number of unlabeled target images as the source images. We implement the Hard-Max and Soft-Max version of MDANs according to Alg. 1, based on the basic vehicle counting network FCN described in (Zhang et al., 2017). Please refer to (Zhang et al., 2017) for detailed network architecture and parameters. The learning rate is initialized by 0.01 and adjusted by the first and second order momentum in the training process. The domain adaptation parameter is selected by cross validation. We compare our method with two baselines: Baseline 1: FCN. It is our basic network without domain adaptation as introduced in work (Zhang et al., 2017). Baseline 2: DANN. We implement DANN on top of the same basic network following the algorithm introduced in work (Ganin et al., 2016).
