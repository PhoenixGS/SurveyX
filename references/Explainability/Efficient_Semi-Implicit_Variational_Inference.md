# Efficient Semi-Implicit Variational Inference
Vincent Moens
Huawei R&D UK
vincent.moens@huawei.com
Hang Ren
Huawei R&D UK
hang.ren1@huawei.com
Alexandre Maraval
Huawei R&D UK
alexandre.maravel@huawei.com
Rasul Tutunov
Huawei R&D UK
rasul.tutunov@huawei.com
Jun Wang
Huawei R&D UK
University College London
w.j@huawei.com
Haitham Ammar ∗
Huawei R&D UK
University College London
haitham.ammar@huawei.com
# Abstract
In this paper, we propose CI-VI an efficient and scalable solver for semi-implicit variational inference (SIVI). Our method, first, maps SIVI’s evidence lower bound (ELBO) to a form involving a nonlinear functional nesting of expected values and then develops a rigorous optimiser capable of correctly handling bias inherent to nonlinear nested expectations using an extrapolation-smoothing mechanism coupled with gradient sketching. Our theoretical results demonstrate convergence to a stationary point of the ELBO in general non-convex settings typically arising when using deep network models and an order of O(t−4/5) gradient-bias-vanishing rate. We believe these results generalise beyond the specific nesting arising from SIVI to other forms. Finally, in a set of experiments, we demonstrate the effectiveness of our algorithm in approximating complex posteriors on various data-sets including those from natural language processing.
arXiv:2101.06070v1
# 1 Introduction
Variational Inference (VI) is an approximate Bayesian inference framework that recasts reasoning about latent variable models as an instance of numerical optimisation [15, 34]. This is achieved by positing a class of variational distributions and optimising an evidence lower-bound (ELBO) that involves log-joint densities rather than intractable posteriors. Classical VI introduces simplifying assumptions (e.g., mean-field and/or conditional conjugacy) to allow for tractable optimisation leading to algorithms that ascend to a stationary point of the ELBO [3, 15]. Though successful in many applications [21, 40, 49], such assumptions, unfortunately, restrict “representation-power” and thus, lead to models that underestimate the variance of the posterior. Realising this problem, numerous frameworks aiming at expanding expressiveness of variational families have been proposed. Works in [10, 11, 12, 13, 15, 24, 31, 33, 34, 37, 38], for instance, relax mean-field assumptions and attempt to restore some dependencies in variational distributions but still require analytical probability density functions. Others in [14, 22, 25, 35, 38], moreover, introduce implicit models by sampling noise vectors and propagating these through deep networks [36]. These techniques do lead to distributions that are implicit but render computing log-variational densities and
∗Honorary position at UCL.
their gradients intractable. For this reason, authors resort to density ratio estimation; a methodology hard-to-scale and stabilise in high-dimensional settings.
To avoid density ratio estimation, recent work in [46] proposed semi-implicit variational inference (SIVI) as a hierarchical framework that obtains variational distributions through a mixing parameter accompanied by known-noise priors [36]. Exploiting this definition, original SIVI optimises a sequence of lower bounds that asymptotically converge to the original ELBO. Rather than focusing on approximate lower (upper) bounds, an unbiased estimator of the exact gradient of SIVI’s ELBO has been newly achieved [36] through a Hamiltonian Monte-Carlo simulator that draws samples from a reverse conditional acquiring state-of-the-art status. Though achieving better results than previous works, unbiased estimators based on Markov Chain Monte Carlo (MCMC) easily become computationally expensive in high-dimensional regimes. Hence, an efficient solver for models supporting semi-implicit variational family distributions, largely, remains an open problem to which we contribute in this paper. Tackling the above problem, we present CI-VI, an efficient semi-implicit solver with rigorous theoretical guarantees capable of scaling to high-dimensional scenarios. Our method first maps the ELBO in SIVI to a nonlinear nested expectation (or compositional) form2, i.e., Eν[fν (Eω[gω(θ)])] with ν and ω being random variables3, fν : Rn →R, gw : Rp →Rn are smooth, not necessarily convex functions, and θ the optimisation parameter – and then devises an algorithm capable of handling the bias resulting from the non-linear composition of Eω[·] and Eν[·] through fν(·). To do so, we introduce an extrapolation-smoothing step to an ADAM-like [18, 41] solver and, in turn, show vanishing gradient bias in the order of O(t−4/5) ultimately enabling convergence to a stationary point. Our resulting algorithm shares similarities to recent work from [41] but unlocks novel theoretical results analysing gradient-bias terms. Though similar in spirit to [41], we realise that the original version presented in Algorithm 1 fails to scale to high-dimensions due to the need of computing large matrix-vector products (see Gradients’ instructions in Algorithm 1). Rectifying this problem, we lastly anchor a gradient-sketching mechanism allowing for batched matrix-vector products, and, further, study the theoretical outcomes of such a combination. Finally, we conduct an in-depth empirical study demonstrating that CI-VI outperforms other algorithms from SIVI, nested Monte-Carlo [30], and compositional optimisation [41, 44] literature.
# 2 Compositional implicit variational inference (CI-VI):
In this section, we demonstrate the connection between semi-implicit variational inference and compositional stochastic optimisation4. We, first, present the semi-implicit inference framework as detailed in [36, 46], and then link to compositional optimisation in Section 2.2. We, finally, feature an efficient adaptive solver and provide its relevant theoretical guarantees in Section 2.3.
# 2.1 Semi-implicit variational inference
To approximate the posterior p(z|x) of a probabilistic model p(x, z), we define a semi-implicit variational distribution qθ(z) in a hierarchical fashion using a mixing parameter as introduced in [36]:
� �� � Equation 1 reveals the reason behind qθ(z) being implicit as we can obtain latent variable samples through ϵ but can not (tractably) compute the integral especially when using deep networks in representing qθ(z|ϵ). In this work, we impose two standard assumptions on the nature of the variational distribution as previously explained in [36, 46]. The first assumes that qθ(z|ϵ) is reparameterisable. That is, to obtain samples from z ∼qθ(z|ϵ), one can draw an auxiliary variable u and then set z as a deterministic function hθ(·) of the sampled u i.e., u ∼q(u), z = hθ(u; ϵ) ≡z ∼qθ(z|ϵ). The second, moreover, assumes that we can evaluate the log-density of the conditional i.e., log qθ(z|ϵ) as well as its gradient. As noted in [36], such an assumption is not strong in that it holds for many
2Please notice the decoupling between the inner and outer functions fν(·) and gω(·). This requires us to further analyse SIVI’s ELBO; see Section 2.3. 3Please note we do not assume any independence between ν and ω 4We use compositional and nested stochastic optimisation interchangeably.
reparameterisable distributions, e.g., Gaussian, Laplace, exponential, and many others. Analogous to standard variational inference, model parameters are fit by minimising the negate of an evidence-lowe (ELBO) bound that can be derived as follows:
Contrary to classical VI that assumes tractable expectations, SIVI introduces additional intractability (e.g., in the entropy term) due to the implicit nature of the variational distribution defined in Equation 1. To tackle such intractability, recently the authors in [36] proposed writing the gradient of the entropy term as an expectation and following an MCMC sampler [42], e.g., Hamiltonian Monte Carlo [2] to estimate the gradient of the ELBO. MCMC methods, however, are known to be computationally expensive and can exhibit high variance as they assume no model – see Section 3 for a detailed comparison. Rather than following MCMC, in this paper we contribute by showing that the semiimplicit ELBO can be written as an instance of compositional optimisation and devise an adaptive and efficient solver with rigorous theoretical guarantees.
# 2.2 SIVI in a compositional nested form
In this section, we present a novel connection mapping implicit variational inference to compositional stochastic optimisation, paving-the-way for an efficient and scalable solver that we later develop in Section 2.3. To do so, we start by plugging-in the variational distribution from Equation 1 in the inner-part of the ELBO (i.e., Equation 2) to get: log p(x) ≥Ez∼qθ(z) � log p(x,z) qθ(z) � = Ez∼qθ(z) � log p(x,z) Eˆϵ∼q(ϵ)[qθ(z|ˆϵ)] � , where we used ˆϵ to denote an inner-random variable also sampled according to q(ϵ). As noted earlier, we assume that the conditional qθ(z|ϵ) is reparameterisable through an auxiliary variable u and a deterministic function hθ(u; ϵ) with ϵ ∼q(ϵ) but independent from ˆϵ. Rather than reparametrising both inner and outer expectations, we only reparameterise the outer expectation leading us to:
Remembering that in variational inference one minimises the negate of the ELBO, we can further write:
Remembering that in variational inference one minimises the negate of the ELBO, we can furthe write:
where we used µ = {u, ϵ} to concatenate outer random variables with q(µ) = q(u, ϵ) = q(u)q(ϵ). Hence, the optimisation problem involved in SIVI can be written as: minθ Eµ[log Eˆϵ[Jµ,ˆϵ(θ)]]. Superficially, the aforementioned problem looks compositional in nature due to the non-linear (through the logarithm) nesting of both expectations. It is worth emphasizing, however, that the standard nested form introduced in Section 1 assumes an inherent decoupling between the inner and outer expectations, i.e., Eν[fν (Eω[gω(θ)])]. In light of this realisation, we now introduce a formalisation capable of achieving this decoupling. To do so, we consider a pool of n-µ samples distributed according to q(µ): Pool = {µi = ⟨ui, ϵi⟩}n i=1. Now, we define a vector-valued function, gˆϵ(θ), of size n corresponding to the evaluations of Jµ,ˆϵ(θ) on each of the samples from the pool, i.e., ∀j ∈[1, n] we have: gˆϵ(θ) = [Jµ1,ˆϵ(θ), . . . , Jµn,ˆϵ(θ)]T, with Jµj,ˆϵ(θ) = qθ(hθ(uj;ϵj)|ˆϵ) p(x|hθ(uj;ϵj))p(hθ(uj;ϵj)). To achieve the decoupling between inner an outer expectations, we allow fν(y) = [log y]Teν with eν being the ν’th basis vector in Rn, i.e., a vector of all zeros except a value of one in the ν’th position. Sampling uniformly from the pool, we can finally write SIVI’s optimisation problem in a compositional form as:
� � ��   with fν(y) = [log y]Teν and gˆϵ(θ) = [Jµ1,ˆϵ(θ), . . . , Jµn,ˆϵ(θ)]T. Clearly, our problem becomes exact only when assuming an infinite number of samples, i.e., n →∞. As such, one would naturally choose n to be large-enough for a small variance estimator of the loss. This, in turn, adds complexity
(2)
(3)
in designing a solver that now has to handle nested expectations and high-dimensional regimes. In the next section, we introduce such an algorithm through a novel combination of extrapolation-smoothing and gradient sketching mechanisms.
# 2.3 An adaptive solver
When designing a solver for the optimisation problem in Equation 3, we consider three essential criteria. First, we would like a simple-to-implement (i.e., single loop) yet effective and scalable algorithm. Second, we aim to have a bias-controlling procedure5 and third, we need a rigorous and theoretically-grounded solver. When surveying optimisation literature, we realise that a promising direction is a first-order method as opposed to zero [9] or second-order [39] ones. Such a realisation is grounded in the fact that first-order methods only require gradient information, are typically simple to implement through a single-loop, and perform competitively in large-scale machine learning applications [6, 18, 26]. Among first-order methods, one can further categorise adaptive [6, 26, 48] and momentum-based [20, 28] algorithms. In spite of numerous theoretical developments [1, 7, 23], ADAM – an adaptive optimiser originally proposed in [18], and then theoretically grounded in [32, 47] – (arguably) retains state-of-the-art status. Therefore, following an adaptive-like update scheme to solving the problem in Equation 3 promises ease of implementation and scalability to real-world scenarios. Though meeting two out of the three criteria above, a simple adaptation of standard optimisation techniques to a compositional problem of the form in Equation 3 is challenging due to the bias incurred from a naive Monte-Carlo sampling of non-linear nested expectations; see [30] for a detailed discussion. Of course, such a problem is not unique to this paper and has been previously studied in [30, 43]. Current methods, however, are either not-scalable to high-dimensional large-data problems [4, 27] (e.g., require full gradients – over all data – for variance reduction), or solve a relaxed version that presumes a finite sum empirical-risk-minimisation6 problem [30, 43]. To meet our requirements, we next present a novel solver that combines extrapolation-smoothing for biasreduction and gradient-sketching for efficiency and scalability. It is worth noting that our optimiser shares similarities to the work by [41] but refines analysis to derive bias-handling results (Equation 5) and introduces additional constructs (e.g., gradient-sketching mechanisms). Such additions require new proof foundations and re-derivations that we present in the appendix for completeness. Algorithmic development We aim to offer an adaptive algorithm that exhibits similar (theoretical and practical) performance guarantees to ADAM but that is also capable of correctly handling the bias inherent to the problem in Equation 2. To do so, we introduce an update scheme that resembles ADAM but incorporates auxiliary variables that are updated in a subsequent step. Being at an iteration t, our algorithm first executes the following updates:
When designing a solver for the optimisation problem in Equation 3, we consider three essential criteria. First, we would like a simple-to-implement (i.e., single loop) yet effective and scalable algorithm. Second, we aim to have a bias-controlling procedure5 and third, we need a rigorous and theoretically-grounded solver. When surveying optimisation literature, we realise that a promising direction is a first-order method as opposed to zero [9] or second-order [39] ones. Such a realisation is grounded in the fact that first-order methods only require gradient information, are typically simple to implement through a single-loop, and perform competitively in large-scale machine learning applications [6, 18, 26]. Among first-order methods, one can further categorise adaptive [6, 26, 48] and momentum-based [20, 28] algorithms. In spite of numerous theoretical developments [1, 7, 23], ADAM – an adaptive optimiser originally proposed in [18], and then theoretically grounded in [32, 47] – (arguably) retains state-of-the-art status. Therefore, following an adaptive-like update scheme to solving the problem in Equation 3 promises ease of implementation and scalability to real-world scenarios.
Algorithmic development We aim to offer an adaptive algorithm that exhibits similar (theoretical and practical) performance guarantees to ADAM but that is also capable of correctly handling the bias inherent to the problem in Equation 2. To do so, we introduce an update scheme that resembles ADAM but incorporates auxiliary variables that are updated in a subsequent step. Being at an iteration t, our algorithm first executes the following updates:
Primary ⇒ mt = γ(1) t mt−1 + � 1 −γ(1) t � ∇L(θt) , vt = γ(2) t vt−1 + � 1 −γ(2) t � [∇L(θt)] ⇝θt+1 = θt −αt mt √vt + ξ , with αt being a learning rate, ξ ∈R>0, γ(1) t and γ(2) t denoting hyper-parameters.
Assuming the availability of sub-sampled gradients of the loss (i.e., ∇L(θt)), the above set of instructions simply performs an ADAM-like update on the model’s free parameters7 θ starting from an initialisation for mt and vt. The problem, however, arises when aiming to acquire unbiased gradients of the objective in Equation 3 [30, 36, 46]. To illustrate this, consider computing the actual gradient of L(θ) at some iteration t. This can be written as: ∇L(θt) = Eˆϵ[∇gˆϵ(θt)]TEν[∇fν (Eˆϵ[gˆϵ(θt)])]. It is clear that one can easily implement a Monte-Carlo estimator of the first part of the gradient, i.e., Eˆϵ[∇gˆϵ(θt)]. The second term, on the other hand, is much harder to estimate due to its nested nature:
5Here, it is to be understood that bias is due to estimating nonlinear nested expectations. 6Please note that relaxed empirical-risk versions target a different problem all-together [43]. In other words, solving a finite-sum approximation does not guarantee convergence for the nested expectation problem presented in Equation 3. 7It is worth noting that later in our theoretical analysis we provide a rigorous scheme for tuning all hyperparameters. We also follow such a schedule in our experiments.
5Here, it is to be understood that bias is due to estimating nonlinear nested expectations. 6Please note that relaxed empirical-risk versions target a different problem all-together [43]. In other words, solving a finite-sum approximation does not guarantee convergence for the nested expectation problem presented in Equation 3. 7It is worth noting that later in our theoretical analysis we provide a rigorous scheme for tuning all hyperparameters. We also follow such a schedule in our experiments.
Of course, a simple Monte-Carlo estimator8 of the gradient’s second part is biased. Our methodology in tackling this challenge is to find an “unbiased” approximation with properties allowing us to control such a bias at appropriate rates. To do so, we follow an extrapolation-smoothing scheme [41, 43, 44] originally established in time series [45] and differential equations literature [17]. These methods introduce a two-step procedure to approximate an unknown quantity, e.g., Eν[∇fν (Eˆϵ[gˆϵ(θt)])] in our case. In the first step, a linear extrapolation query vector, z, is computed while in the second, a smoothed average is evaluated around the extrapolated z. Precisely, given two model parameter updates θt and θt+1 we execute the following:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c7ac/c7ac50d5-44b8-4cca-a9fd-83202dd8dd0b.png" style="width: 50%;"></div>
Auxiliary ⇒
zt+1 =
�
1 −1/βt
�
θt + 1/βtθt+1
�
��
�
Extrapolation
and yt+1 = (1 −βt)yt + βtgt(zt+1)
�
��
�
Smoothing
,
with βt being a free parameter, and gt(zt+1) is a sampled estimator of Eˆϵ[gˆϵ (zt+1)].
Simply, the smoothing step is attempting to track Eˆϵ[gˆϵ (zt+1)] which can then be substituted in Equation 4 to approximate the second term of the gradient. Interestingly, we evaluate this smoothing step around a linearly extrapolated vector zt+1 and not only on the updated model parameters θt+1. Though an evaluation around θt+1 can guarantee convergence [43], we demonstrate that following the above extrapolation scheme leads to faster convergence rates by enabling a better control of the bias. Informally, we can, under further technical consideration, demonstrate that the difference between true and sub-sampled gradients abides by:
where ∇gt(θt) and ∇ft(yt) denote estimate gradients of the functions gˆϵ(·) and fν(·), and Etotal[·] the expectation under all incurred randomness in the algorithm. Importantly, the result in Equation 5 shows that the bias in our gradient estimator vanishes with an increased number of iterations. This, in turn, allows us to prove convergence of the resulting algorithm as studied in Section 2.3. We now introduce the complete algorithm combining both of the above primary and auxiliary steps. To that end, we assume a schedule of learning rates η-schedule = {⟨αt, βt, γ(1) t , γ(2) t ⟩}T t=1, and a timevarying set of batch-sizes K-size = {⟨K(1) t , K(2) t , K(3) t ⟩} needed to mini-batch ∇fν(·), ∇gˆϵ(·), and Eˆϵ[gˆϵ(·)] respectively. Our computations are achieved through two oracles that can return gradients and function values when needed9:
Oraclef � yt, K(1) t � = {⟨νti, ∇fνti (yt)⟩}K(1) t i=1 , with {νti}K(1) t i=1 being i.i.d. Oracleg � zt, K(2) t � = {⟨ˆϵti, gˆϵti (zt), ∇gˆϵti (zt)⟩}K(2) t i=1 , with {ˆϵti}K(2) t i=1 als
In addition, we define δ ∈(0, 1) used to measure solution accuracy and a small positive constant ξ for numerical stability. The overall procedure is practical requiring only one implementation loop and is summarised in Algorithm 1. It operates in three main steps. In the first, sub-sampled gradients are computed by calling Oraclef(·) and Oracleg(·). When estimated, the second step computes primary updates leading to improved model parameters θt+1. Given θt and θt+1, the third step executes extrapolation-smoothing to update an estimate of Eˆϵ[gˆϵ(·)] guaranteeing vanishing bias with increased iterations. It is to be noted that the smoothing step requires an additional call to Oracleg(·) to sample gt(zt+1). The updated smoothed variable yt+1 is then used in subsequent iterations where the overall process repeats.
On practicability Algorithm 1, though successful, assumes an idealised setting in which computing products of gradient estimates is feasible. In most reasonably-sized problems, however, such products are prohibitively expensive due to the problem’s dimensionality and number of samples available (e.g.,
8We mean by a simple estimator the following: Eν � 1 (Eˆϵ[gˆϵ(θt)])ν � ≈ 1 N �N i=1 � 1 ( 1 M �M j=1 gj(θt))i � . 9Please note that in Section 2.3 we provide explicit schedules for each of the hyper-parameters introduced.
(5)
gorithm 1 CI-VI: Compositional Implicit Variational Inference
Inputs: Initial variable θ1, δ ∈(0, 1), ξ, T = O(δ−5/4), η-schedule and K-size
Initialisation: Initialise z1 = θ1, y1 = 0, and m0 = v0 = 0
or t = 1 to T do:
Compute sub-sampled gradients by calling oracles:
▷Gradients
∇ft(yt) = 1/K(1)
t
K(1)
t
�
i=1
∇fνti (yt)
=⇒∇L(θt) = ∇gt(θt)
T∇ft(yt)
∇gt(θt) = 1/K(2)
t
K(2)
t
�
j=1
∇gˆϵtj (θt)
Perform the following primary updates:
▷Primary Update
mt = γ(1)
t
mt−1 +
�
1 −γ(1)
t
�
∇L(θt)
=⇒θt+1 = θt −αt
mt
√vt+ξ
vt = γ(2)
t
vt−1 +
�
1 −γ(2)
t
�
[∇L(θt)]2
Perform the following auxiliary updates:
▷Auxiliary Update
zt+1 = (1 −1/βt) θt + 1/βtθt+1
�
��
�
Extrapolation
and yt+1 = (1 −βt)yt + βtgt(zt+1)
�
��
�
Smoothing
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/64fd/64fd0cea-91f1-498d-8ab8-26c347171343.png" style="width: 50%;"></div>
� �� � � : Return solution as a uniform sample from {θt}T t=1
� �� � Output: Return solution as a uniform sample from {θt}T t=
order of thousands). To remedy this problem, we next present a matrix-sketching mechanism (not introduced previously in both variational inference and compositional optimisation literature) from randomised linear algebra [5] enabling scalability. Here, we simply replace gradient computations (in orange) in Algorithm 1 with the set of instructions10 in Algorithm 2 that eases implementation in that it allows for randomised matrix-vector products which come-in handy in big-data problems. Rather than needing to sum overall entries in the corresponding product, we can now simply enable a batch-like version through sketching. Chiefly, as we show in Section 3, we can further specialise Algorithm 2 by accounting for the sparsity pattern of our gradients for improved adjustability to the SIVI setting.
Algorithm 2 Gradient-Sketching for Large-Scale SIVI
Inputs: Oraclef(yt, K(1)
t
), Oracleg(θt, K(2)
t
), #-samples dt ∈[1, n]
Sample subset St ⊆[1, . . . , n] of size dt, where Pr(k ∈St) = 1/n for all k = 1, . . . , n.
Return: ∇L(θt) = 1/K(1)
t
�K(1)
t
a=1 [q/dt
�
k∈St ∇gT
ˆϵta (θt)(:, k)∇fνta (yt)(k)]
Of course, sketching mechanisms can affect speeds of convergence due to additionally induced randomness. To gain insight into such phenomena, we next provide rigorous theoretical guarantees demonstrating convergence of the resulting algorithm (i.e., Algorithm 1 with Sketching for gradient computation) and quantifying oracle complexities.
Theoretical guarantees We demonstrate that Algorithm 1 converges to a stationary point of the non-convex objective11 in Equation 3. As is standard in optimisation literature, we provide our result
10In Algorithm 2, we use A(:, k) to denote the kth column of some matrix A, and b(k) the kth component of a vector b. 11Non-convexity is abundant in our problem due to the usage of neural networks. In these scenarios, one aims at a stationary point as even assessing a local-minimum is NP-Hard [16].
# =⇒∇L(θt) = ∇gt(θt) T∇ft(yt)
in terms of the number of oracle calls needed to convergence12. Due to space constraints, we defer the proof to the appendix. Here, we provide the statement of the main theorem. Our main results are based on the following common assumptions: Assumption 1. We make the following assumptions13: 1) |fν(y)| ≤Bf, and ||∇fν(y)|| ≤Mf, for all y and ν; 2) fν(·) is Lf-smooth, and gˆϵ(x) is Mg- Lipschitz continuous, and Lg-smooth; 3) oracle sample-pairs are independent; and 4) oracles return unbiased gradient estimates with bounded variances.
Now, we present the main theorem analysing convergence and oracle complexities of Algorithm 1: Theorem 1 (Convergence & Oracle Complexities). Consider a parameter setup given by: αt = Cα/t 1 5 , βt = Cβ, K(1) t = C1t 4 5 , K(2) t = C2t 4 5 , K(3) t = C3t 4 5 , γ(1) t = Cγµt, γ(t) 2 = 1 −Cα/t 2 5 (1 − Cγµt)2, for some positive constants Cα, Cβ, C1, C2, C3, Cγ, µ such that Cβ < 1 and µ ∈(0, 1). For any δ ∈(0, 1), Algorithm 1 running gradient-sketching (i.e., using Algorithm 2 to compute gradient products) with a sample-size dt = O(1) outputs, in expectation, a δ-approximate first-order stationary point ˜θ of L(θ). That is: Etotal[||∇L(˜θ)||2 2] ≤δ, with “total” representing all incurred randomness. Moreover, Algorithm 1 acquires ˜θ with an overall oracle complexity of the order O � δ−9/4� .
# � � 3 Experiments and results
In this section, we present an empirical study demonstrating the effectiveness of CI-VI that we implement in PyTorch [29]. We benchmark on three broad tasks covering toy examples, Bayesian logistic regression and variational autoencoders. Of course, our derivations need to be specialised to each of these tasks. We provide such constructs in the appendix due to space constraints. We compare against standard semi-implicit variational inference algorithms (e.g., SIVI and UIVI) in addition to methods from nested Monte-Carlo [30] and compositional optimisation [43, 44]. To improve stability and computational efficiency, our implementation of CI-VI tracks log-gradients instead of ∇L(θt). Furthermore, due to the special structures of fν(·) (e.g., logarithmic function) and gˆϵ, we can further improve gradient sketching by only (uniformly) sampling non-zero elements from ∇gˆϵ(·) rather than the whole dt-set St in Algorithm 2. Due to space constraints, such adaptations in addition to exact experimental settings can be found in the appendix. We ran all experiments on a single NVIDIA GeForce RTX 2080 GPU. Crucially, CI-VI is highly efficient consuming 30 seconds for toy experiments and, at most, 1.5 hours for text modelling when using variational autoencoders. Toy Experiments: In this set of experiments, we apply our method to minimise a KL-divergence between semi-implicit and ground truth distributions on two-modal, star, and banana as taken from [36]. For all semi-implicit distributions,
For all semi-implicit distributions, we chose q(ϵ) to be a multi-variate zero-mean identity-covariance-matrix Gaussian, i.e., q(ϵ) = N(0, I3×3). The conditional distribution qθ(z|ϵ), is also assumed Gaussian with a neural network (two layers 50 by 50 hidden units) parameterised mean and a parameterised diagonal covariance matrix, where qθ(z|ϵ) = N(µθ1(ϵ), diag(θ2)) with θ1 denoting neural network parameters, and θ2 another set of free parameters. Figures 1 compares the contour plots of the optimised variational distribution
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2dfe/2dfee827-e6bd-44b4-8ce9-41c0deb2b353.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Results demonstrating that CI-VI can approximate sophisticated distributions from [36]. Left: Two-Modal distribution, Middle: Star distribution, and Right: Banana distribution.</div>
with ground-truth. We can clearly see that CI-VI can accurately capture sophisticated patterns like skewness, kurtosis and multi-modality. Bayesian Logistic Regression: With our method performing well on toy examples, we ran CI-VI in Bayesian logistic regression that aims to acquire posteriors on classification tasks. Given a 12Please note that analysing other metrics, e.g., generalisation bounds is an interesting avenue for future research. 13Please note that for additional clarity, our assumptions are further elaborated in the appendix.
with ground-truth. We can clearly see that CI-VI can accurately capture sophisticated patterns like skewness, kurtosis and multi-modality.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c376/c376c883-496b-4267-bddf-75faa513d1aa.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5d01/5d0153fe-957f-4d29-bd65-f49401eb9636.png" style="width: 50%;"></div>
<div style="text-align: center;">Nodal data-set</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/83a3/83a368dc-88d7-4eeb-905e-7f13b69dea45.png" style="width: 50%;"></div>
<div style="text-align: center;">Fashion-MNIST data-set <latexit sha1_base64="nH6HesRjHxu5dWkdrKiRHU2Kt9Q=">ACBnicbVDLSgNBEJz1bXxFPYowGAQvCbsi6MGD IgelIhGhSE3knGTL7YKZXDEtOXvwVLx4U8eo3ePNvnDwOmljQUFPVzXSXHytpyHW/nYnJqemZ2bn5zMLi0vJKdnXtxkSJFlgSkYr0nQ8GlQyxRJIU3sUaIfAV3vrt45/e4/ayCi8pk6M1QCaoWxIAWSlWnazQvhA6QmYln3nzy/Orq5HQj yBqlby+bcgtsHyfekOTYEMVa9qtSj0QSYEhCgTFlz42pmoImKR2M5XEYAyiDU0sWxpCgKa9s/o8m2r1Hkj0rZC4n3190QKgTGdwLedAVDLjHo98T+vnFDjoJrKME4IQzH4qJEoThHvZcLrUqMg1bEhJZ2Vy5aoEGQTS5jQ/BGTx4nN7sFzy1 4l3u5o8NhHNsg2xHeaxfXbETlmRlZhgj+yZvbI358l5cd6dj0HrhDOcWd/4Hz+AKylmJM=</latexit> <latexit sha1_base64="nH6HesRjHxu5dWkdrKiRHU2Kt9Q=">ACBnicbVDLSgNBEJz1bXxFPYowGAQvCbsi6MGD IgelIhGhSE3knGTL7YKZXDEtOXvwVLx4U8eo3ePNvnDwOmljQUFPVzXSXHytpyHW/nYnJqemZ2bn5zMLi0vJKdnXtxkSJFlgSkYr0nQ8GlQyxRJIU3sUaIfAV3vrt45/e4/ayCi8pk6M1QCaoWxIAWSlWnazQvhA6QmYln3nzy/Orq5HQj yBqlby+bcgtsHyfekOTYEMVa9qtSj0QSYEhCgTFlz42pmoImKR2M5XEYAyiDU0sWxpCgKa9s/o8m2r1Hkj0rZC4n3190QKgTGdwLedAVDLjHo98T+vnFDjoJrKME4IQzH4qJEoThHvZcLrUqMg1bEhJZ2Vy5aoEGQTS5jQ/BGTx4nN7sFzy1 4l3u5o8NhHNsg2xHeaxfXbETlmRlZhgj+yZvbI358l5cd6dj0HrhDOcWd/4Hz+AKylmJM=</latexit> <latexit sha1_base64="nH6HesRjHxu5dWkdrKiRHU2Kt9Q=">ACBnicbVDLSgNBEJz1bXxFPYowGAQvCbsi6MGD IgelIhGhSE3knGTL7YKZXDEtOXvwVLx4U8eo3ePNvnDwOmljQUFPVzXSXHytpyHW/nYnJqemZ2bn5zMLi0vJKdnXtxkSJFlgSkYr0nQ8GlQyxRJIU3sUaIfAV3vrt45/e4/ayCi8pk6M1QCaoWxIAWSlWnazQvhA6QmYln3nzy/Orq5HQj yBqlby+bcgtsHyfekOTYEMVa9qtSj0QSYEhCgTFlz42pmoImKR2M5XEYAyiDU0sWxpCgKa9s/o8m2r1Hkj0rZC4n3190QKgTGdwLedAVDLjHo98T+vnFDjoJrKME4IQzH4qJEoThHvZcLrUqMg1bEhJZ2Vy5aoEGQTS5jQ/BGTx4nN7sFzy1 4l3u5o8NhHNsg2xHeaxfXbETlmRlZhgj+yZvbI358l5cd6dj0HrhDOcWd/4Hz+AKylmJM=</latexit> <latexit sha1_base64="nH6HesRjHxu5dWkdrKiRHU2Kt9Q=">ACBnicbVDLSgNBEJz1bXxFPYowGAQvCbsi6MGD IgelIhGhSE3knGTL7YKZXDEtOXvwVLx4U8eo3ePNvnDwOmljQUFPVzXSXHytpyHW/nYnJqemZ2bn5zMLi0vJKdnXtxkSJFlgSkYr0nQ8GlQyxRJIU3sUaIfAV3vrt45/e4/ayCi8pk6M1QCaoWxIAWSlWnazQvhA6QmYln3nzy/Orq5HQj yBqlby+bcgtsHyfekOTYEMVa9qtSj0QSYEhCgTFlz42pmoImKR2M5XEYAyiDU0sWxpCgKa9s/o8m2r1Hkj0rZC4n3190QKgTGdwLedAVDLjHo98T+vnFDjoJrKME4IQzH4qJEoThHvZcLrUqMg1bEhJZ2Vy5aoEGQTS5jQ/BGTx4nN7sFzy1 4l3u5o8NhHNsg2xHeaxfXbETlmRlZhgj+yZvbI358l5cd6dj0HrhDOcWd/4Hz+AKylmJM=</latexit></div>
<div style="text-align: center;">Fashion-MNIST data-set</div>
<div style="text-align: center;">MNIST data-set</div>
Figure 2: Results depicting performance of CI-VI on both Bayesian logistic regression (top 2 rows) and variational autoenconders (bottom row). We realise that CI-VI is closer in its estimation to MCMC than is mean-field variational Bayes (MFVB). On the variational autoencoders side, we realise the CI-VI outperforms others on MNIST, Fashion-MNIST, and PBT. We use NMC-1, NMC-2, NMC-3 to denote nested-Monte-Carlo algorithms that use ADAM [18], RMS-Prop [32], and SGD respectively.
data-set D = {xi, yi}N i=1 where xi = (xi1, . . . , xiD)T is a D-dimensional feature vector and yi ∈{0, 1} a binary label, Bayesian logistic regression considers a probabilistic model p(D, z) = p(z) �N i=1 p(yi|xi, z) = p(z) �N i=1 Bernoulli � (1 + exp(−xT i z))−1� and infers the posterior of z given the observed D. We experimented with the Waveform, Spam, and Nodal data-sets from [46]. We fixed the prior p(z) to be a zero-mean Gaussian given by: p(z) = N(0, 100 × ID×D). For the semi-implicit setting, ϵ followed a standard Gaussian whose dimension varied across data-sets. qθ(z|ϵ) was again a Gaussian with parameterised mean (two layer neural network with 200 units each) but with a full covariance matrix: qθ(z|ϵ) = N(µθ1(ϵ), Lθ2LT θ2). Due to space constraints, the full set of results can be found in the appendix. In the first two rows in Figures 1, we demonstrate violin plots on all three data-sets. Clearly CI-VI captures the variance better than Mean-Field Variational Bias (MFVB in the figure) when compared to MCMC distributions 14 across all latent variables. Semi-implicit Variational Autoencoders: In our final evaluation, we extensively experimented with variational autoencoders [19] but ones that exhibited semi-implicit variational distributions. In fact, it has been shown that upon the usage of semi-implicit variational distributions, the gap between the ELBO and marginal data likelihood can further be reduced [36, 46]. We experimented with three data-sets, two of which are standard (MNIST and Fashion-MNIST), while the third considered a text modelling task with the Penn-Tree-Bank (PTB) as presented in [8]. All structural details in each of these scenarios can be found in the appendix. Our results depicted in Figures 2 (bottom-row) compare CI-VI with semi-implicit solvers [36, 46], nested Monte-Carlo algorithms, and compositional
14Please note we also show a marginalised pair-wise posterior plot in the appendi
<div style="text-align: center;">Waveform data-set</div>
<div style="text-align: center;">PBT data-set <latexit sha1_base64="G+DAXyJcFWhg9lzdFsEbVf6Rmow=">AB/HicbVDJSgNBEO1xjXEbzdFLYxC8GZE0IOH oBePEbJBMoSenpqkSc9Cd40YhvgrXjwo4tUP8ebf2FkOmvig4PFeFVX1/FQKjY7zba2srq1vbBa2its7u3v79sFhUyeZ4tDgiUxU2capIihgQIltFMFLPIltPzh7cRvPYDSIonrOErBi1g/FqHgDI3Us0tdhEfMazd1GjBkZxpw3LPLTsWZgi4 Td07KZI5az/7qBgnPIoiRS6Z1x3VS9HKmUHAJ42I305AyPmR96Bgaswi0l0+PH9MTowQ0TJSpGOlU/T2Rs0jrUeSbzojhQC96E/E/r5NheOXlIk4zhJjPFoWZpJjQSRI0EAo4ypEhjCthbqV8wBTjaPIqmhDcxZeXSfO84joV9/6iXL2ex1EgR+S YnBKXJIquSM10iCcjMgzeSVv1pP1Yr1bH7PWFWs+UyJ/YH3+AGlYlJU=</latexit> <latexit sha1_base64="G+DAXyJcFWhg9lzdFsEbVf6Rmow=">AB/HicbVDJSgNBEO1xjXEbzdFLYxC8GZE0IOH oBePEbJBMoSenpqkSc9Cd40YhvgrXjwo4tUP8ebf2FkOmvig4PFeFVX1/FQKjY7zba2srq1vbBa2its7u3v79sFhUyeZ4tDgiUxU2capIihgQIltFMFLPIltPzh7cRvPYDSIonrOErBi1g/FqHgDI3Us0tdhEfMazd1GjBkZxpw3LPLTsWZgi4 Td07KZI5az/7qBgnPIoiRS6Z1x3VS9HKmUHAJ42I305AyPmR96Bgaswi0l0+PH9MTowQ0TJSpGOlU/T2Rs0jrUeSbzojhQC96E/E/r5NheOXlIk4zhJjPFoWZpJjQSRI0EAo4ypEhjCthbqV8wBTjaPIqmhDcxZeXSfO84joV9/6iXL2ex1EgR+S YnBKXJIquSM10iCcjMgzeSVv1pP1Yr1bH7PWFWs+UyJ/YH3+AGlYlJU=</latexit> <latexit sha1_base64="G+DAXyJcFWhg9lzdFsEbVf6Rmow=">AB/HicbVDJSgNBEO1xjXEbzdFLYxC8GZE0IOH oBePEbJBMoSenpqkSc9Cd40YhvgrXjwo4tUP8ebf2FkOmvig4PFeFVX1/FQKjY7zba2srq1vbBa2its7u3v79sFhUyeZ4tDgiUxU2capIihgQIltFMFLPIltPzh7cRvPYDSIonrOErBi1g/FqHgDI3Us0tdhEfMazd1GjBkZxpw3LPLTsWZgi4 Td07KZI5az/7qBgnPIoiRS6Z1x3VS9HKmUHAJ42I305AyPmR96Bgaswi0l0+PH9MTowQ0TJSpGOlU/T2Rs0jrUeSbzojhQC96E/E/r5NheOXlIk4zhJjPFoWZpJjQSRI0EAo4ypEhjCthbqV8wBTjaPIqmhDcxZeXSfO84joV9/6iXL2ex1EgR+S YnBKXJIquSM10iCcjMgzeSVv1pP1Yr1bH7PWFWs+UyJ/YH3+AGlYlJU=</latexit> <latexit sha1_base64="G+DAXyJcFWhg9lzdFsEbVf6Rmow=">AB/HicbVDJSgNBEO1xjXEbzdFLYxC8GZE0IOH oBePEbJBMoSenpqkSc9Cd40YhvgrXjwo4tUP8ebf2FkOmvig4PFeFVX1/FQKjY7zba2srq1vbBa2its7u3v79sFhUyeZ4tDgiUxU2capIihgQIltFMFLPIltPzh7cRvPYDSIonrOErBi1g/FqHgDI3Us0tdhEfMazd1GjBkZxpw3LPLTsWZgi4 Td07KZI5az/7qBgnPIoiRS6Z1x3VS9HKmUHAJ42I305AyPmR96Bgaswi0l0+PH9MTowQ0TJSpGOlU/T2Rs0jrUeSbzojhQC96E/E/r5NheOXlIk4zhJjPFoWZpJjQSRI0EAo4ypEhjCthbqV8wBTjaPIqmhDcxZeXSfO84joV9/6iXL2ex1EgR+S YnBKXJIquSM10iCcjMgzeSVv1pP1Yr1bH7PWFWs+UyJ/YH3+AGlYlJU=</latexit></div>
optimisers. Again, it is clear that CI-VI outperforms others in terms of the number of epochs needed for convergence. Interestingly, such a gap is further signified on the PBT data-set15.
# 4 Conclusions and Future Work
We proposed CI-VI, a compositional solver for scalable and efficient semi-implicit variational inference. Our method rewrites SIVI as an instance of a compositional optimisation and devises a solver that correctly handles nested bias through an extrapolation-smoothing and a gradient sketching mechanism. We tested our method on a variety of tasks, including text modelling from natural language processing. In all these instances, we showed CI-VI’s effectiveness. In papers to follow, we plan to further scale our method to dialogue problems from NLP and to extend our analysis to time-series models. We also think our nested-expectation theoretical results can be broadly applied beyond this paper to cover topics from experimental design. We will also tackle this direction in the future.
# References
[1] Zeyuan Allen Zhu and Elad Hazan. Variance Reduction for Faster Non-Convex Optimization. In Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, volume 48, pages 699–707, 2016. [2] Michael Betancourt. A Conceptual Introduction to Hamiltonian Monte Carlo. arXiv preprint arXiv:1701.02434, 2017. [3] David M. Blei, Alp Kucukelbir, and Jon D. McAuliffe. Variational Inference: A Review for Statisticians. Journal of the American Statistical Association, 112(518):859–877, 2017. [4] P.J. Davis and P. Rabinowitz. Methods of Numerical Integration. Dover Books on Mathematics Series. Dover Publications, 2007. [5] Petros Drineas, Ravi Kannan, and Michael W. Mahoney. Fast Monte Carlo Algorithms for Matrices I: Approximating Matrix Multiplication. SIAM J. Comput., 36(1):132–157, July 2006. [6] John Duchi, Elad Hazan, and Yoram Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. J. Mach. Learn. Res., page 2121–2159, 2011. [7] Cong Fang, Chris Junchi Li, Zhouchen Lin, and Tong Zhang. SPIDER: Near-Optimal NonConvex Optimization via Stochastic Path-Integrated Differential Estimator. In Advances in Neural Information Processing Systems 31, NeurIPS 2018, pages 687–697, 2018. [8] Hao Fu, Chunyuan Li, Ke Bai, Jianfeng Gao, and Lawrence Carin. Flexible Text Modeling with Semi-Implicit Latent Representations. [9] Victor Gabillon, Rasul Tutunov, Michal Valko, and Haitham Bou Ammar. Derivative-Free & Order-Robust Optimisation. arXiv preprint arXiv:1910.04034, 2019. 10] Ryan Giordano, Tamara Broderick, and Michael I. Jordan. Linear Response Methods for Accurate Covariance Estimates from Mean Field Variational Bayes. In Advances in Neural Information Processing Systems 28, NeurIPS 2015, pages 1441–1449, 2015. 11] Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Rezende, and Daan Wierstra. DRAW: A Recurrent Neural Network For Image Generation. In Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, volume 37, pages 1462–1471, 2015. 12] Shaobo Han, Xuejun Liao, David B. Dunson, and Lawrence Carin. Variational Gaussian Copula Inference. In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, AISTATS 2016, volume 51, pages 829–838, 2016. 13] Matthew D. Hoffman, David M. Blei, Chong Wang, and John Paisley. Stochastic Variational Inference. J. Mach. Learn. Res., 14(1):1303–1347, 2013. 14] Ferenc Huszár. Variational Inference using Implicit Distributions. CoRR, abs/1702.08235, 2017.
15Please note that we do not show the results of SIVI and UIVI in Figure 1 (h) as it was hard to get these to correctly operate on NLP tasks due to their source implementations. Staying fair to these methods, we opted-ou of demonstrating their performance.
[15] Tommi S. Jaakkola and Michael I. Jordan. Variational Probabilistic Inference and the QMR-DT Network. J. Artif. Intell. Res., 10:291–322, 1999. [16] Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M. Kakade, and Michael I. Jordan. How to Escape Saddle Points Efficiently. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, volume 70, pages 1724–1732, 2017. [17] D. C. Joyce. Survey of Extrapolation Process in Numerical Analysis. SIAM Review, 13(4):435– 490, 1971. [18] Diederik P. Kingma and Jimmy Ba. ADAM: A Method for Stochastic Optimization. In 3rd International Conference on Learning Representations, ICLR 2015, 2015. [19] Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations, ICLR 2014, 2014. [20] Huan Li and Zhouchen Lin. Accelerated Proximal Gradient Methods for Nonconvex Programming. In Advances in Neural Information Processing Systems 28, NeurIPS 2015, pages 379–387, 2015. [21] Minne Li, Lisheng Wu, Jun Wang, and Haitham Bou-Ammar. Multi-View Reinforcement Learning. In Advances in Neural Information Processing Systems 32, NeurIPS 2019, pages 1418–1429, 2019. [22] Yingzhen Li and Richard E. Turner. Gradient Estimators for Implicit Models. In 6th International Conference on Learning Representations, ICLR 2018, 2018. [23] Liu Liu, Ji Liu, Cho-Jui Hsieh, and Dacheng Tao. Stochastically controlled stochastic gradient for the convex and non-convex composition problem, 2018. [24] Lars Maaløe, Casper Kaae Sønderby, Søren Kaae Sønderby, and Ole Winther. Auxiliary Deep Generative Models. In Proceedings of The 33rd International Conference on Machine Learning, ICML 2016, volume 48, pages 1445–1453, 2016. [25] Shakir Mohamed and Balaji Lakshminarayanan. Learning in Implicit Generative Models. CoRR, abs/1610.03483, 2016. [26] Mahesh Chandra Mukkamala and Matthias Hein. Variants of RMSProp and Adagrad with Logarithmic Regret Bounds. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, page 2545–2553, 2017. [27] Yuji Nakatsukasa. Approximate and Integrate: Variance Reduction in Monte Carlo Integration via Function Approximation. arXiv preprint arXiv:1806.05492, 2018. [28] Yurii Nesterov. Introductory Lectures on Convex Optimization: A Basic Course. Springer Publishing Company, Incorporated, first edition, 2014. [29] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32, NeurIPS 2019, pages 8024–8035. 2019. [30] Tom Rainforth, Robert Cornish, Hongseok Yang, and Andrew Warrington. On Nesting Monte Carlo Estimators. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, volume 80, pages 4264–4273, 2018. [31] Rajesh Ranganath, Dustin Tran, and David M. Blei. Hierarchical Variational Models. 33rd International Conference on Machine Learning, ICML 2016, 1:515–528, 2016. [32] Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. On the Convergence of ADAM and Beyond. In 6th International Conference on Learning Representations, ICLR 2018, 2018. [33] Danilo Jimenez Rezende and Shakir Mohamed. Variational Inference with Normalizing Flows. In 32nd International Conference on Machine Learning, ICML 2015, volume 2, pages 1530– 1538, 2015. [34] Lawrence K. Saul and Michael Jordan. Exploiting Tractable Substructures in Intractable Networks. In Advances in Neural Information Processing Systems 8, NeurIPS 1996, pages 486–492, 1996.
[35] Jiaxin Shi, Shengyang Sun, and Jun Zhu. Kernel Implicit Variational Inference. 6th International Conference on Learning Representations, ICLR 2018, 2018. [36] Michalis K. Titsias and Francisco J. R. Ruiz. Unbiased Implicit Variational Inference. In The 22nd International Conference on Artificial Intelligence and Statistics, AISTATS 2019, volume 89, pages 167–176, 2019. [37] Dustin Tran, Rajesh Ranganath, and David M. Blei. The Variational Gaussian Process. In 4th International Conference on Learning Representations, ICLR 2016, pages 1–14, 2016. [38] Dustin Tran, Rajesh Ranganath, and David M. Blei. Hierarchical Implicit Models and Likelihood-Free Variational Inference. In Advances in Neural Information Processing Systems 30, NeurIPS 2017, pages 5523–5533, 2017. [39] Rasul Tutunov, Haitham Bou-Ammar, and Ali Jadbabaie. Distributed Newton Method for Large-Scale Consensus Optimization. IEEE Trans. Autom. Control., 64(10):3983–3994, 2019. [40] Rasul Tutunov, Dongho Kim, and Haitham Bou Ammar. Distributed Multitask Reinforcement Learning with Quadratic Convergence. In Advances in Neural Information Processing Systems 31, NeurIPS 2018, pages 8907–8916. 2018. [41] Rasul Tutunov, Minne Li, Jun Wang, and Haitham Bou-Ammar. Compositional ADAM: An Adaptive Compositional Solver. CoRR, abs/2002.03755, 2020. [42] Don van Ravenzwaaij, Peter Cassey, and Scott Brown. A Simple Introduction to Markov Chain Monte–Carlo Sampling. Psychonomic Bulletin & Review, 25, 03 2016. [43] Mengdi Wang, Ethan X. Fang, and Han Liu. Stochastic Compositional Gradient Descent: Algorithms for Minimizing Compositions of Expected-value Functions. Math. Program., 161(1-2):419–449, 2017. [44] Mengdi Wang, Ji Liu, and Ethan Fang. Accelerating Stochastic Composition Optimization. In Advances in Neural Information Processing Systems 29, NeurIPS 2016, pages 1714–1722. 2016. [45] Norbert Wiener. Extrapolation, Interpolation, and Smoothing of Stationary Time Series. The MIT Press, 1964. [46] Mingzhang Yin and Mingyuan Zhou. Semi-Implicit Variational Inference. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, volume 80, pages 5646–5655, 2018. [47] Manzil Zaheer, Sashank Reddi, Devendra Sachan, Satyen Kale, and Sanjiv Kumar. Adaptive Methods for Nonconvex Optimization. In Advances in Neural Information Processing Systems 31, NeurIPS 2018, pages 9793–9803. 2018. [48] Matthew D. Zeiler. Adadelta: An Adaptive Learning Rate Method. CoRR, abs/1212.5701, 2012. [49] Cheng Zhang, Judith Bütepage, Hedvig Kjellström, and Stephan Mandt. Advances in Variational Inference. IEEE Trans. Pattern Anal. Mach. Intell., 41(8):2008–2026, 2019.
# A Practical implementation
To improve numerical stability and computational efficiency, our implementation of CI-VI has been adapted to properties of functions fν(·) and gˆϵ(·) defined in Section 2.2. These adaptions are clarified below:
To improve numerical stability and computational efficiency, our implementation of CI-VI has been adapted to properties of functions fν(·) and gˆϵ(·) defined in Section 2.2. These adaptions are clarified
Log trick: The output from gˆϵ(·) is a vector of density ratios whose value can be extreme, especially when the z and x are high-dimensional rvs. In order to do inference in such probabilistic model, we propose a CI-VI implementation which conducts most of the computations in log-scale. The
∇L(θt) = ∇gt(θt) T∇ft(yt) = [∇log gt(θt) T]p×n · exp � log gt(θt) + log ∇ft(yt) � n×1 � �� � kt
The j-th element of kt is then given by Equation 6. Since yt is a smoothed average of gt(θt), their logarithms will cancel each other before the exponentiation is taken.
  � Finalising this log-scale implementation, we track the value of log yt instead of yt for the auxiliar update as follow:
 � For large-scale CI-VI, updating all dimensions of log yt is challenging due to computational and memory constraints. Rather than performing a full update for log yt, we follow a batch-like mechanism that proved effective in our experiments. Namely, we split the index set [1, n] into smaller chunks and sample νt from one of these chunks to execute the updates. This way, only those dimensions indexed by the current chunk need to be updated rather than the whole high-dimensional vector log yt. Of course, such chunks have to vary across iterations so as to guarantee the update of log yt. To do so, we switch to a new chunk occasionally and re-initialize the dimensions of log yt indexed by the new chunk using the value of log gt(zt).
Sparse gradient-sketching: Another insight from Equation 6 is that the kt is a sparse vector with non-zero elements indexed by sampled νt. The gradient-sketching in Algorithm 2 can be adapted by removing the zero elements in kt and corresponding columns in [∇log gt(θt) T]p×n first. The set St is then sampled uniformly from the left indices.
# B Experimental settings and results
Toy Experiments: In toy experiments, we minimize the KL-divergence between semi-implict variational distributions qθ(z) and ground-truth distributions p(z). The compositional objective can be written as: � � ��
esian Logistic Regression: Given a data-set D = {xi, yi}N i=1, we can write the compositional m of negative ELBO as:
Bayesian Logistic Regression: Given a data-set D = {xi, yi}N i=1, we can write the compositional form of negative ELBO as:
where p(D|z) = �N i=1 p(yi|xi, z) = �N i=1 Bernoulli � (1 + exp(−xT i z))−1� .
(6)
# Semi-implicit Variational Autoencoder: Given a dataset D = {xi}N i=1, the negative EL single datapoint xi is given by:
Semi-implicit Variational Autoencoder: Given a dataset D = {xi}N i=1, the negative ELBO of a single datapoint xi is given by:
in which the encoder parameter θ and decoder parameter ˆθ are optimized jointly. We can further construct an estimator of the negative ELBO of the full data-set: � � ��
in which the encoder parameter θ and decoder parameter ˆθ are optimized jointly. We can further construct an estimator of the negative ELBO of the full data-set: −ELBO(D) = N · E � log E � qθ(hθ(ui; ϵi)|ˆϵi, xi) �� ,
in which the encoder parameter θ and decoder parameter ˆθ are optimized jointly. We can further construct an estimator of the negative ELBO of the full data-set: −ELBO(D) = N · Exi∼Uniform(D),ui∼q(u),ϵi∼q(ϵ) � log Eˆϵi∼p(ˆϵi) � qθ(hθ(ui; ϵi)|ˆϵi, xi) pˆ θ(xi|hθ(ui; ϵi)) · p(hθ(ui; ϵi)) �� ,
Two-Modal
Star
Banana
0.5N
��
−2
0
�
, I2×2
�
+
0.5N
��
2
0
�
, I2×2
�
0.5N
�
0,
�
2
1.8
1.8
2
��
+
0.5N
�
0,
�
2
−1.8
−1.8
2
��
z =
�
z1
z2 −z2
1 −1
�
�
z1
z2
�
∼N
�
0,
�
1
0.9
0.9
1
��
Table 1: Ground-truth distributions used in toy experiments
� � � − �� � � � Table 1: Ground-truth distributions used in toy experiments
Two-Modal
Star
Banana
q(ϵ)
N (0, I3×3)
N (0, I3×3)
N (0, I3×3)
qθ(z|ϵ)
N(µθ1(ϵ), diag(θ2))
N(µθ1(ϵ), diag(θ2))
N(µθ1(ϵ), diag(θ2))
µθ1
Hidden units: (50, 50)
Hidden activation: ReLU
Initializer: Xavier_normal
Hidden units: (50, 50)
Hidden activation: ReLU
Initializer: Xavier_normal
Hidden units: (50, 50)
Hidden activation: ReLU
Initializer: Xavier_normal
Hyper
params
K(1)
t
= 1 · 102
K(2)
t
= 1 · 103
Cα = 3 · 10−4
Cβ = 0.99
Cγ = 0.9
µ = 0.999
K(1)
t
= 2 · 102
K(2)
t
= 2 · 103
Cα = 2 · 10−4
Cβ = 0.999
Cγ = 0.9
µ = 0.999
K(1)
t
= 2 · 102
K(2)
t
= 2 · 103
Cα = 3 · 10−4
Cβ = 0.999
Cγ = 1.0
µ = 0.999
Time
elapsed
0.037 sec/iter × 200 iters
7.432 sec
0.040 sec/iter × 300 iters
11.972 sec
0.042 sec/iter × 300 iters
12.604 sec
Table 2: Experimental settings for toy examples
Spam
Nodal
Waveform
q(ϵ)
N (0, 100 · I3×3)
N (0, 100 · I3×3)
N (0, 100 · I10×10)
qθ(z|ϵ)
N(µθ1(ϵ), Lθ2LT
θ2), z ∈R3
N(µθ1(ϵ), Lθ2LT
θ2), z ∈R6
N(µθ1(ϵ), Lθ2LT
θ2), z ∈R22
µθ1(ϵ)
Hidden units: (200, 200)
Hidden activation: ReLU
Initializer: Xavier_normal
Hidden units: (200, 200)
Hidden activation: ReLU
Initializer: Xavier_normal
Hidden units: (200, 200)
Hidden activation: ReLU
Initializer: Xavier_normal
Hyper
params
K(1)
t
= 5 · 101
K(2)
t
= 5 · 102
C(θ1)
α
= 1.5 · 10−4
C(θ2)
α
= 2 · 10−1
Cβ = 0.999
C(θ1)
γ
= 0.7
C(θ2)
γ
= 0.6
µ = 0.999
K(1)
t
= 2 · 102
K(2)
t
= 2 · 103
C(θ1)
α
= 1.7 · 10−4
C(θ2)
α
= 1.7 · 10−4
Cβ = 0.99
C(θ1)
γ
= 0.75
C(θ2)
γ
= 0.85
µ = 0.999
K(1)
t
= 1 · 102
K(2)
t
= 1 · 103
C(θ1)
α
= 3 · 10−4
C(θ2)
α
= 2.5 · 10−4
Cβ = 0.999
C(θ1)
γ
= 0.85
C(θ2)
γ
= 0.85
µ = 0.999
Time
elapsed
0.024 sec/iter × 600 iters
14.180 sec
0.015 sec/iter × 600 iters
8.986 sec
0.015 sec/iter × 3000 iters
45.761 sec
Table 3: Experimental settings for Bayesian logistic regression
Table 3: Experimental settings for Bayesian logistic regression
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b33f/b33fbbe6-6c1c-4a8e-a519-d4ad65e70372.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Boxplot of marginal posteriors (left) and marginal & pairwise joint-posteriors (right) using MCMC, CIVI and MFVB on Spam data-set (top row) and Nodal data-set (bottom row). On the right (grid plots), MCMC is blue, CIVI is green and MFVB is orange.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2ca0/2ca08eab-c96b-449b-b9d3-6597cd81c702.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Boxplot of marginal posteriors using MCMC, CIVI and MFVB on Waveform data-set.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b0ea/b0ea820c-62f5-4da7-b421-90c972f8c7a5.png" style="width: 50%;"></div>
Figure 5: Marginal posteriors and pairwise joint-posteriors using MCMC, CIVI and MFVB on Waveform dataset. MCMC is blue, CIVI is green and MFVB is orange.
Figure 5: Marginal posteriors and pairwise joint-posteriors using MCMC, CIVI and MFVB on Waveform dataset. MCMC is blue, CIVI is green and MFVB is orange.
MNIST
FasionMNIST
PBT
q(ϵ)
N (0, I10×10)
N (0, I10×10)
N (0, I10×10)
qθ(z|ϵ, xi)
N(µθ1(ϵ, xi), diag(θ2))
z ∈R10
N(µθ1(ϵ, xi), diag(θ2))
z ∈R10
N(µθ1(ϵ, xi), diag(θ2))
z ∈R20
µθ1(ϵ, xi)
Hidden units: (200, 200)
Hidden activation: ReLU
Initializer: Xavier_normal
Hidden units: (200, 200)
Hidden activation: ReLU
Initializer: Xavier_normal
Embedding dim: 256
LSTM hidden dim: 256
MLP hidden dim: 256
MLP activation: ReLU
pˆθ(xi|z)
�D
i=1 Bernoulli[πˆθ(z)]
�D
i=1 Bernoulli[πˆθ(z)]
�D
i=1 Categorical[πˆθ(z)]
πˆθ(z)
Hidden units: (200, 200)
Hidden activation: ReLU
Initializer: Xavier_normal
Hidden units: (200, 200)
Hidden activation: ReLU
Initializer: Xavier_normal
Embedding dim: 256
LSTM hidden dim: 256
Hyper
params
K(1)
t
= 300
K(2)
t
= 30
C(θ1)
α
= 1 · 10−5
C(θ2)
α
= 1 · 10−4
C(ˆθ)
α
= 5 · 10−4
Cβ = 0.999
C(θ1)
γ
= 0.2
C(θ2)
γ
= 0.2
C(ˆθ)
γ
= 0.2
µ = 0.999
K(1)
t
= 300
K(2)
t
= 30
C(θ1)
α
= 1 · 10−5
C(θ2)
α
= 1 · 10−4
C(ˆθ)
α
= 5 · 10−4
Cβ = 0.999
C(θ1)
γ
= 0.2
C(θ2)
γ
= 0.2
C(ˆθ)
γ
= 0.2
µ = 0.999
K(1)
t
= 320
K(2)
t
= 100
C(θ1)
α
= 2 · 10−4
C(θ2)
α
= 2 · 10−4
C(ˆθ)
α
= 5 · 10−4
Cβ = 0.99
C(θ1)
γ
= 0.1
C(θ2)
γ
= 0.1
C(ˆθ)
γ
= 0.1
µ = 0.999
Time
elapsed
0.0218 sec/iter × 18000 iters
392 sec
0.0236 sec/iter × 18000 iters
425 sec
0.0989 sec/iter × 40000 iters
3956 sec
Table 4: Experimental settings for variational autoencoder.
Table 4: Experimental settings for variational autoencoder.
Method
Time (ms) per iteration
SIVI
155
UIVI
69
ASCPG
157
SCGD
152
CI-VI [this paper]
56
: Average time per iteration for training VAE on Intel i9-9900X
verage time per iteration for training VAE on Intel i9-9900X 3.50GHz C
# C Detailed Descriptions of Assumptions
Due to the lack of space, we provide more detailed description of all assumptions required to establish theoretical convergence results for the proposed CI-VI Algorithm in this section. Recall, we target the following nested optimisation problem:
ue to the lack of space, we provide more detailed description of all assumptions required to establish heoretical convergence results for the proposed CI-VI Algorithm in this section.
min θ∈Rp L(θ) = Eν [fν (Eˆϵ [gˆϵ(θ)])] re for any ν, ˆϵ we have fν(·) : Rn →R and gˆϵ(·) : Rp →Rn and random e unknown distributions ν ∼pν(·) and ˆϵ ∼pˆϵ(·) correspondingly. Ple ume that these distributions are independent. For brevity, let us denote f ) = Eˆϵ[gˆϵ(θ)], then it is easy to see: L(θ) = f(g(θ)), ∇L(θ) = Eˆϵ[∇gˆϵ(θ)T]Eν[∇fν(Eˆϵ[gˆϵ(θ)])] = ∇g(θ)T∇f( the following assumption holds: umption 1: 1. Function fν(·) is bounded, i.e.∀y ∈Rn: |fν(y)| ≤Bf. for any v. 2. Function fν(·) is Lf−smooth. i.e.∀y1, y2 ∈Rn: ||∇fν(y1) −∇fν(y2)||2 ≤Lf||y1 −y2||2. for any v. 3. Function fν(·) has bounded gradient, i.e ∀y ∈Rn: ||∇fν(y)||2 ≤Mf. for any v. 4. Mapping gˆϵ(θ) is Mg Lipschitz continuous, i.e. ∀θ, z ∈Rp: ||gˆϵ(θ) −gˆϵ(z)||2 ≤Mg||θ −z||2. for any ˆϵ. 5. Mapping gˆϵ(θ) is Lg−smooth. i.e.∀θ, z ∈Rp: ||∇gˆϵ(θ) −∇gˆϵ(z)||2 ≤Lg||θ −z||2. for any ˆϵ. ause distributions ν ∼pν(·) and ˆϵ ∼pˆϵ(·) are unknown, we assume the er oracles FOOf and FOOg, such that given fixed vectors zt ∈Rp,  mbers K(1) t and K(2) t at time step t they return the following collections:
where for any ν, ˆϵ we have fν(·) : Rn →R and gˆϵ(·) : Rp →Rn and random variables ν, ˆϵ follow some unknown distributions ν ∼pν(·) and ˆϵ ∼pˆϵ(·) correspondingly. Please notice, we do not assume that these distributions are independent. For brevity, let us denote f(y) = Eν[fν(y)] and g(θ) = Eˆϵ[gˆϵ(θ)], then it is easy to see: L(θ) = f(g(θ)), ∇L(θ) = Eˆϵ[∇gˆϵ(θ)T]Eν[∇fν(Eˆϵ[gˆϵ(θ)])] = ∇g(θ)T∇f(g(θ)).
Let the following assumption holds:
# Assumption 1:
1. Function fν(·) is bounded, i.e.∀y ∈Rn:
|fν(y)| ≤Bf.
  ||∇fν(y1) −∇fν(y2)||2 ≤Lf||y1 −y2||2.
4. Mapping gˆϵ(θ) is Mg Lipschitz continuous, i.e. ∀θ, z ∈Rp: ||gˆϵ(θ) −gˆϵ(z)||2 ≤Mg||θ −z||2.
for any ˆϵ.
5. Mapping gˆϵ(θ) is Lg−smooth. i.e.∀θ, z ∈Rp:
||∇gˆϵ(θ) −∇gˆϵ(z)||2 ≤Lg||θ −z||2.
for any ˆϵ.
Because distributions ν ∼pν(·) and ˆϵ ∼pˆϵ(·) are unknown, we assume the presence of two first order oracles FOOf and FOOg, such that given fixed vectors zt ∈Rp, yt ∈Rn and integer numbers K(1) t and K(2) t at time step t they return the following collections:
FOOf[yt, K(1) t ] = {νta, ∇fνta (yt)}K(1) t a=1 , where {νta}K(1) t a=1 are i.i.d FOOg[zt, K(2) t ] = {ˆϵta, gˆϵta (zt), ∇gˆϵta (zt)}K(2) t a=1 , where {ˆϵta}K(2) t a=1 are i.i.d
The complexity of the proposed algorithm will be evaluated in terms of total number of calls to fir order oracles FOOf[·, ·], FOOg[·, ·].
Assumption 2: At any given time step t, the oracles FOOf[y]t and FOOg[z]t satisfy the following two conditions for any z ∈Rp and y ∈Rn:
(7)
(8)
� {ν1a} K(1) 1 a=1 , {ˆϵ1a} K(2) 1 a=1 � , � {ν2a} K(1) 2 a=1 , {ˆϵ2a} K(2) 2 a=1 � , . . . , � {νta}K(1) t a=1 , {ˆϵta}K(2) t a=1 �
are independent. 2. Unbiased estimates: for any θ ∈Rp, y ∈Rn:
2. Unbiased estimates: for any θ ∈Rp, y ∈Rn:
� � � � for any t and a ∈[1, . . . , K(2) t ], b ∈[1, . . . , K(1) t ]. 3. Bounded variance of stochastic gradients: for any θ ∈Rp, y ∈Rn:
� for any t and a ∈[1, . . . , K(2) t ], b ∈[1, . . . , K(1) t ].
# D Theoretical Guarantees
In this section, we establish all theoretical results needed for proving the main theorem and then present its proof.
# D.1 L-smoothness of function L(·)
Our first result provides the important property of the overall compositional function L(·) which will be used later in the convergence analysis of the proposed CI-VI Algorithm. Lemma: Let Assumptions 1 and 2 hold, then function L(·) is L−Lipschitz smooth, i,e: ||∇L(θ1) −∇L(θ2)||2 ≤L||θ1 −θ2||2 ∀θ1, θ2 ∈Rp (9) with L = M 2 g Lf + LgMf.
with L = M 2 g Lf + LgMf.
Proof. Assumption 1 implies that ||∇gˆϵ(θ)||2 ≤Mg for any θ ∈Rp. Hence, using Jensen inequality as well as property of the norm we have:
||∇L(θ1) −∇L(θ2)||2 = ||Eˆϵ � ∇gT ˆϵ (θ1) � Eν [∇fν (Eˆϵ [gˆϵ(θ1)])] −Eˆϵ � ∇gT ˆϵ (x2) � Eν [∇fν (Eˆϵ [gˆϵ(θ2)])] ||2 ≤ ||Eˆϵ � ∇gT ˆϵ (θ1) � Eν [∇fν (Eˆϵ [gˆϵ(θ1)])] −Eˆϵ � ∇gT ˆϵ (θ1) � Eν [∇fν (Eˆϵ [gˆϵ(θ2)])] ||2+ ||Eˆϵ � ∇gT ˆϵ (θ1) � Eν [∇fν (Eˆϵ [gˆϵ(θ2)])] −Eˆϵ � ∇gT ˆϵ (x2) � Eν [∇fν (Eˆϵ [gˆϵ(θ2)])] ||2 ≤ Eˆϵ � ||∇gT ˆϵ (θ1)||2 � ||Eν [∇fν (Eˆϵ [gˆϵ(θ1)])] −Eν [∇fν (Eˆϵ [gˆϵ(θ2)])] ||2+ ||Eˆϵ � ∇gT ˆϵ (θ1) � −Eˆϵ � ∇gT ˆϵ (x2) � ||2Eν [||∇fν (Eˆϵ [gˆϵ(θ2)]) ||2] ≤ MgEν [||∇fν (Eˆϵ [gˆϵ(θ1)]) −∇fν (Eˆϵ [gˆϵ(θ2)]) ||2] + MfEˆϵ � ||∇gT ˆϵ (θ1) −∇gT ˆϵ (θ2)||2 � ≤MgLf||Eˆϵ [gˆϵ(θ1)] −Eˆϵ [gˆϵ(θ2)] ||2+ MfEˆϵ � ||∇gT ˆϵ (θ1) −∇gT ˆϵ (θ2)||2 � ≤MgLfEˆϵ [||gˆϵ(θ1) −gˆϵ(θ2)||2] + MfEˆϵ � ||∇gT ˆϵ (θ1) −∇gT ˆϵ (θ2)||2 � ≤MgLfEˆϵ [||gˆϵ(θ1) −gˆϵ(θ2)||2] + MfLg||θ1 −θ2||2 ≤ M 2 g Lf||θ1 −θ2||2 + MfLg||θ1 −θ2||2 = � M 2 g Lf + MfLg � ||θ1 −θ2||2 = L||θ1 −θ