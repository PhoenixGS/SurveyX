# Beyond the Mean-Field: Structured Deep Gaussian Processes Improve the Predictive Uncertainties
# Abstract
Deep Gaussian Processes learn probabilistic data representations for supervised learning by cascading multiple Gaussian Processes. While this model family promises flexible predictive distributions, exact inference is not tractable. Approximate inference techniques trade off the ability to closely resemble the posterior distribution against speed of convergence and computational efficiency. We propose a novel Gaussian variational family that allows for retaining covariances between latent processes while achieving fast convergence by marginalising out all global latent variables. After providing a proof of how this marginalisation can be done for general covariances, we restrict them to the ones we empirically found to be most important in order to also achieve computational efficiency. We provide an efficient implementation of our new approach and apply it to several benchmark datasets. It yields excellent results and strikes a better balance between accuracy and calibrated uncertainty estimates than its state-of-the-art alternatives.
# 1 Introduction
Gaussian Processes (GPs) provide a non-parametric framework for learning distributions over unknown functions from data [21]: As the posterior distribution can be computed in closed-form, they return well-calibrated uncertainty estimates, making them particularly useful in safety critical applications [3, 22], Bayesian optimisation [10, 30], active learning [37] or under covariate shift [31]. However, the analytical tractability of GPs comes at the price of reduced flexibility: Standard kernel functions make strong assumptions such as stationarity or smoothness. To make GPs more flexible, a practitioner would have to come up with hand-crafted features or kernel functions. Both alternatives require expert knowledge and are prone to overfitting. Deep Gaussian Processes (DGPs) offer a compelling alternative since they learn non-linear feature representations in a fully probabilistic manner via GP cascades [6]. The gained flexibility has the drawback that inference can no longer be carried out in closed-form, but must be performed via Monte Carlo sampling [9], or approximate inference techniques [5, 6, 24]. The most popular approximation, variational inference, searches for the best approximate posterior within a pre-defined class of distributions: the variational family [4]. For GPs, variational approximations often build on the inducing point framework where a small set of global latent variables acts as pseudo datapoints summarising the training data [29, 32]. For DGPs, each latent GP is governed by its own set of inducing variables, which, in general, need not be independent from those of other latent GPs. Here, we offer a new class of variational families for DGPs taking the following two requirements into account: (i) all global latent variables, i.e., inducing outputs, can be marginalised out, (ii) correlations between latent GP models can be captured. Satisfying (i) reduces the variance in the estimators and is needed for fast convergence [16] while (ii) leads to better calibrated uncertainty estimates [33].
nference on Neural Information Processing Systems (NeurIPS 2020), Vancouver
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/09a4/09a48c9a-8a35-493a-869e-7b34d3ce1a76.png" style="width: 50%;"></div>
By using a fully-parameterised Gaussian variational posterior over the global latent variables, we automatically fulfil (ii), and we show in Sec. 3.1, via a proof by induction, that (i) can still be achieved. The proof is constructive, resulting in a novel inference scheme for variational families that allow for correlations within and across layers. The proposed scheme is general and can be used for arbitrarily structured covariances allowing the user to easily adapt it to application-specific covariances, depending on the desired DGP model architecture and on the system requirements with respect to speed, memory and accuracy. One particular case, in which the variational family is chain-structured, has also been considered in a recent work [34], in which the compositional uncertainty in deep GP models is studied. In Fig. 1 (right) we depict exemplary inferred covariances between the latent GPs for a standard deep GP architecture. In addition to the diagonal blocks, the covariance matrix has visible diagonal stripes in the offdiagonal blocks and an arrow structure. These diagonal stripes point towards strong dependencies between successive latent GPs, while the arrow structure reflects
Figure 1: Covariance matrices for variational posteriors. We used a DGP with 2 hidden layers (L1, L2) of 5 latent GPs each and a single GP in the output layer (L3). The complexity of the variational approximation is increased by allowing for additional dependencies within and across layers in a Gaussian variational family (left: meanfield [24], middle: stripes-and-arrow, right: fully-coupled). Plotted are natural logarithms of the absolute values of the variational covariance matrices over the inducing outputs.
dependencies between all hidden layers and the output layer. In Sec. 3.2, we further propose a scalable approximation to this variational family, which only takes these stronger correlations into account (Fig. 1, middle). We provide efficient implementations for both variational families, where we particularly exploit the sparsity and structure of the covariance matrix of the variational posterior. In Sec. 4, we show experimentally that the new algorithm works well in practice. Our approach obtains a better balance between accurate predictions and calibrated uncertainty estimates than its competitors, as we showcase by varying the distance of the test from the training points.
# 2 Background
In the following, we introduce the notation and provide the necessary background on DGP models. GPs are their building blocks and the starting point of our review.
# 2.1 Primer on Gaussian Processes
In regression problems, the task is to learn a function f : RD →R that maps a set of N input points xN = {xn}N n=1 to a corresponding set of noisy outputs yN = {yn}N n=1. Throughout this work, we assume iid noise, p(yN|fN) = �N n=1 p(yn|fn), where fn = f(xn) and fN = {fn}N n=1 are the function values at the input points. We place a zero mean GP prior on the function f, f ∼GP(0, k), where k : RD × RD →R is the kernel function. This assumption leads to a multivariate Gaussian prior over the function values, p(fN) = N (fN|0, KNN) with covariance matrix KNN = {k(xn, xn′)}N n,n′=1. In preparation for the next section, we introduce a set of M ≪N so-called inducing points xM = {xm}M m=1 from the input space1 [29, 32]. From the definition of a GP, the corresponding inducing outputs fM = {fm}M m=1, where fm = f(xm), share a joint multivariate Gaussian distribution with fN. We can therefore write the joint density as p(fN, fM) = p(fN|fM)p(fM), where we factorised the joint prior into p(fM) = N (fM|0, KMM), the prior over the inducing outputs, and the conditional p(fN|fM) = N � fN ����KNMfM, �KNN � with
� ���� � � �KNM = KNM (KMM)−1 , �KNN = KNN −KNM (KMM)−1 KMN.
� ���� � � �KNM = KNM (KMM)−1 , �KNN = KNN −KNM (KMM)−1 KMN. ere the matrices K are defined similarly as KNN above, e.g. KNM = {k(xn, xm)}N,M n,m=1.
� � Here the matrices K are defined similarly as KNN above, e.g. KNM = {k(xn, xm)}N,M n,m=1.
(1)
A deep Gaussian Process (DGP) is a hierarchical composition of GP models. We consider a model with L layers and Tl (stochastic) functions in layer l = 1, . . . , L, i.e., a total number of T = �L l=1 Tl functions [6]. The input of layer l is the output of the previous layer, f l N = [f l,1(f l−1 N ), . . . , f l,Tl(f l−1 N )], with starting values f 0 N = xN. We place independent GP priors augmented with inducing points on all the functions, using the same kernel kl and the same set of inducing points xl M within layer l. This leads to the following joint model density:
Here p(f l M) = �Tl t=1 N � f l,t M ���0, Kl MM � and p(f l N|f l M; f l−1 N ) = �Tl t=1 N � f l,t N ����Kl NMf l,t M , K
 �   � ��� �  �   � e �Kl NM and �Kl NN are given by the equivalents of Eq. (1), respectively.2
  �   �    �  � Inference in this model (2) is intractable since we cannot marginalise over the latents f 1 N, . . . , f L−1 N as they act as inputs to the non-linear kernel function. We therefore choose to approximate the posterior by employing variational inference: We search for an approximation q(fN, fM) to the true posterior p(fN, fM|yN) by first choosing a variational family for the distribution q and then finding an optimal q within that family that minimises the Kullback-Leibler (KL) divergence KL[q||p]. Equivalently, the so-called evidence lower bound (ELBO),
� can be maximised. In the following, we choose the variational family [24
� Note that fM = {f l,t M }L,Tl l,t=1 contains the inducing outputs of all layers, which might be covarying. This observation will be the starting point for our structured approximation in Sec. 3.1. In the remaining part of this section, we follow Ref. [24] and restrict the distribution over the inducing outputs to be a-posteriori Gaussian and independent between different GPs (known as mean-field assumption, see also Fig. 1, left), q(fM) = �L l=1 �Tl t=1 q(f l,t M ). Here q(f l,t M ) = N � f l,t M ���µl,t M, Sl,t M � and µl,t M, Sl,t M are free variational parameters. The inducing outputs fM act thereby as global latent variables that capture the information of the training data. Plugging q(fM) into Eqs. (2), (3), (4), we can simplify the ELBO to
Note that fM = {f l,t M }L,Tl l,t=1 contains the inducing outputs of all layers, which might be covarying. This observation will be the starting point for our structured approximation in Sec. 3.1.
� � We first note that the ELBO decomposes over the data points, allowing for minibatch subsampling [12]. However, the marginals of the output of the final layer, q(f L n ), cannot be obtained analytically. While the mean-field assumption renders it easy to analytically marginalise out the inducing outputs (see Appx. D.1), the outputs of the intermediate layers cannot be fully integrated out, since they are kernel inputs of the respective next layer, leaving us with
q(f L n ) = � L � l=1 q(f l n; f l−1 n )df 1 n · · · df L−1 n , where q(f l n; f l−1 n ) = Tl � t=1 N � f l,t n ����µl,t n , �Σl,t n � . (6
q(f L n ) = � � l=1 q(f l n; f l−1 n )df 1 n · · · df L−1 n , where q(f l n; f l−1 n ) = l � t=1 N � f l,t n ����µl,t n , �Σl,t n � . (6)
� The means and covariances are given by
�  covariances are given by
�  � � � � � � 2In order to avoid pathologies created by highly non-injective mappings in the DGP [7], we follow Ref. [24] and add non-trainable linear mean terms given by the PCA mapping of the input data to the latent layers. Those terms are omitted from the notation for better readability.
(2)
f L−1 N
(3)
(4)
(5)
(6)
(7)
We can straightforwardly obtain samples from q(f L n ) by recursively sampling through the layers using Eq. (6). Those samples can be used to evaluate the ELBO [Eq. (5)] and to obtain unbiased gradients for parameter optimisation by using the reparameterisation trick [16, 23]. This stochastic estimator of the ELBO has low variance as we only need to sample over the local latent parameters f 1 n, . . . , f L−1 n , while we can marginalise out the global latent parameters, i.e. inducing outputs, fM.
# 3 Structured Deep Gaussian Processes
Next, we introduce a new class of variational families that allows to couple the inducing outputs fM within and across layers. Surprisingly, analytical marginalisation over the inducing outputs fM is still possible after reformulating the problem into a recursive one that can be solved by induction. This enables an efficient inference scheme that refrains from sampling any global latent variables. Our method generalises to arbitrary interactions which we exploit in the second part where we focus on the most prominent ones to attain speed-ups.
# 3.1 Fully-Coupled DGPs
We present now a new variational family that offers both, efficient computations and expressivity: Our approach is efficient, since all global latent variables can be marginalised out , and expressive, since we allow for structure in the variational posterior. We do this by leaving the Gaussianity assumption unchanged, while permitting dependencies between all inducing outputs (within layers and also across layers). This corresponds to the (variational) ansatz q(fM) = N (fM|µM, SM) with dimensionality TM. By taking the dependencies between the latent processes into account, the resulting variational posterior q(fN, fM) [Eq. (4)] is better suited to closely approximate the true posterior. We give a comparison of exemplary covariance matrices SM in Fig. 1. Next, we investigate how the ELBO computations have to be adjusted when using the fully-coupled variational family. Plugging q(fM) into Eqs. (2), (3) and (4), yields
which we derive in detail in Appx. C. The major difference to the mean-field DGP lies in the marginals q(f L n ) of the outputs of the last layer: Assuming (as in the mean-field DGP) that the distribution over the inducing outputs fM factorises between the different GPs causes the marginalisation integral to factorise into L standard Gaussian integrals. This is not the case for the fullycoupled DGP (see Appx. D.1 for more details), which makes the computations more challenging. The implications of using a fully coupled q(fM) are summarised in the following theorem.
for each data point xn. The means and covariances are given by
 � � � � � � where �µl n = �Kl nMµl M and �Sll′ n = δll′Kl nn −�Kl nM � δll′Kl MM −Sll′ M � �Kl′ Mn.
 �  �  � � � � � In Eqs. (10) and (11) the notation Al,1:l′ is used to index a submatrix of the variable A, e.g. Al,1:l′ = � Al,1 · · · Al,l′� . Additionally, µl M ∈RTlM denotes the subvector of µM that contains the means of the inducing outputs in layer l, and Sll′ M ∈RTlM×Tl′M contains the covariances between the inducing outputs of layers l and l′. For �µl n and �Sll′ n , we introduced the notation Kl = � ITl ⊗Kl� as
(10)
(11)
shorthand for the Kronecker product between the identity matrix ITl and the covariance matrix Kl, and used δ for the Kronecker delta. We verify in Appx. B.2 that the formulas contain the mean-field solution as a special case by plugging in the respective covariance matrix. By Thm. 1, the inducing outputs fM can still be marginalised out, which enables low-variance estimators of the ELBO. While the resulting formula for q(f l n|f 1 n, . . . , f l−1 n ) has a similar form as Gaussian conditionals, this is only true at first glance (cf. also Appx. B.1): The latents of the preceding layers f 1:l−1 n enter the mean ˆµl n and the covariance matrix ˆΣl n also in an indirect way via �Sn as they appear as inputs to the kernel matrices.
� Sketch of the proof of Theorem 1. We start the proof with the general formula for q(f L n ),
which is already (implicitly) used in Ref. [24] and which we derive in Appx. D. In order to show the equivalence between the inner integral in Eq. (12) and the integrand in Eq. (9) we proceed to find a recursive formula for integrating out the inducing outputs layer after layer:
The equation above holds for l = 1, . . . , L after the inducing outputs of layers 1, . . . , l have already been marginalised out. This is stated more formally in Lem. 2 in Appx. A, in which we also provide exact formulas for all terms. Importantly, all of them are multivariate Gaussians with known mean and covariance. The lemma itself can be proved by induction and we will show the general idea of the induction step here: For this, we assume the right hand side of Eq. (13) to hold for some layer l and then prove that it also holds for l →l + 1. We start by taking the (known) distribution within the integral and split it in two by conditioning on f l n:
Then we show that the distribution q(f l n|f 1:l−1 n ) can be written as part of the product in front of the integral in Eq. (13) (thereby increasing the upper limit of the product to l). Next, we consider the integration over f l+1 M , where we collect all relevant terms (thereby increasing the lower limit of the product within the integral in Eq. (13) to l + 2): � �
The terms in the first line are given by Eqs. (14) and (2). All subsequent terms are also multivariate Gaussians that are obtained by standard operations like conditioning, joining two distributions, and marginalisation. We can therefore give an analytical expression of the final term in Eq. (15), which is exactly the term that is needed on the right hand side of Eq. (13) for l →l + 1. Confirming that this term has the correct mean and covariance completes the induction step. After proving Lem. 2, Eq. (13) can be used. For the case l = L the right hand side can be shown to yield �L l=1 q(f l n|f 1 n, . . . , f l−1 n ). Hence, Eq. (9) follows by substituting the inner integral in Eq. (12) by this term. The full proof can be found in Appx. A. Furthermore, we give a heuristic argument for Thm. 1 in Appx. B.1 in which we show that by ignoring the recursive structure of the prior, the marginalisation of the inducing outputs fM becomes straightforward. While mathematically not rigorous, the derivation provides additional intuition. Next, we use our novel variational approach to fit a fully coupled DGP model with L = 3 layers to the concrete UCI dataset. We can clearly observe that this algorithmic work pays off: Fig. 1 shows that there is more structure in the covariance matrix SM than the mean-field approximation allows. This additional structure results in a better approximation of the true posterior as we validate on a
(12)
(13)
(14)
(15)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6b90/6b90dc8b-f27f-403a-ae87-2df19f87ad7f.png" style="width: 50%;"></div>
Figure 2: Convergence behaviour: Analytical vs. MC marginalisation. We plot the ELBO as a function of time in seconds when the marginalisation of the inducing outputs fM is performed analytically via our Thm. 1 (purple) and via MC sampling (green). We used a fully-coupled DGP with our standard three layer architecture (see Sec. 3.2), on the concrete UCI dataset trained with Adam [15].
range of benchmark datasets (see Tab. S5 in Appx. G) for which we observe larger ELBO values for the fully-coupled DGP than for the mean-field DGP. Additionally, we show in Fig. 2 that our analytical marginalisation over the inducing outputs fM leads to faster convergence compared to Monte Carlo (MC) sampling, since the corresponding ELBO estimates have lower variance. Independently from our work, the sampling-based approach has also been proposed in Ref. [34]. However, in comparison with the mean-field DGP, the increase in the number of variational parameters also leads to an increase in runtime and made convergence with standard optimisers fragile due to many local optima. We were able to circumvent the latter by the use of natural gradients [2], which have been found to work well for (D)GP models before [10, 26, 25], but this increases the runtime even further (see Sec. 4.2). It is therefore necessary to find a smaller variational family if we want to use the method in large-scale applications. An optimal variational family combines the best of both worlds, i.e., being as efficient as the meanfield DGP while retaining the most important interactions introduced in the fully-coupled DGP. We want to emphasise that there are many possible ways of restricting the covariance matrix SM that potentially lead to benefits in different applications. For example, the recent work [34] studies the compositional uncertainty in deep GPs using a particular restriction of the inverse covariance matrix. The authors also provide specialised algorithms to marginalise out the inducing outputs in their model. Here, we provide an analytic marginalisation scheme for arbitrarily structured covariance matrices that will vastly simplify future development of application-specific covariances. Through the general framework that we have developed, testing them is straightforward and can be done via simply implementing a naive version of the covariance matrix in our code.3 In the following, we propose one possible class of covariance matrices based on our empirical findings.
# 3.2 Stripes-and-Arrow Approximation
In this section, we describe a new variational family that trades off efficiency and expressivity by sparsifying the covariance matrix SM. Inspecting Fig. 1 (right) again, we observe that besides the M × M blocks on the diagonal, the diagonal stripes [28] (covariances between the GPs in latent layers at the same relative position), and an arrow structure (covariances from every intermediate layer GP to the output GP) receive large values. We make similar observations also for different datasets and different DGP architectures as shown in Fig. S8 in Appx. G. Note that the stripes pattern can also be motivated theoretically as we expect the residual connections realised by the mean functions (footnote 2) to lead to a coupling between successive latent GPs. We therefore propose as one special form to keep only these terms and neglect all other dependencies by setting them to zero in the covariance matrix, resulting in a structure consisting of an arrowhead and diagonal stripes (see Fig. 1 middle). Denoting the number of GPs per latent layer as τ, it is straightforward to show that the number of non-zero elements in the covariance matrices of mean-field DGP, stripes-and-arrow DGP, and fullycoupled DGP scale as O(τLM 2), O(τL2M 2), and O(τ 2L2M 2), respectively. In the example of Fig. 1, we have used τ = 5, L = 3, and M = 128, yielding 1.8 × 105, 5.1 × 105, and 2.0 × 106 non-zero elements in the covariance matrices. Reducing the number of parameters already leads to shorter training times since less gradients need to be computed. Furthermore, the property that makes this form so compelling is that the covariance matrix �S1:l−1,1:l−1 n [needed in Eqs. (10) and
 � 3Python code (building on code for the mean-field DGP [25], GPflow [19] and TensorFlow [1]) imp menting our method is provided at https://github.com/boschresearch/Structured_DGP. pseudocode description of our algorithm is given in Appx. F.
(11)] as well as the Cholesky decomposition4 of SM have the same sparsity pattern. Therefore only the non-zero elements at pre-defined positions have to be calculated which is explained in Appx. E. The complexity for the ELBO is O(NM 2τL2 + Nτ 3L3 + M 3τL3). This is a moderate increase compared to the mean-field DGP whose ELBO has complexity O(NM 2τL), while it is a clear improvement over the fully-coupled approach with complexity O(NM 2τ 2L2+Nτ 3L3+M 3τ 3L3) (see Appx. E for derivations). An empirical runtime comparison is provided in Sec. 4.2. After having discussed the advantages of the proposed approximation a remark on a disadvantage is in order: The efficient implementation of Ref. [26] for natural gradients cannot be used in this setting, since the transformation from our parameterisation to a fully-parameterised multivariate Gaussian is not invertible. However, this is only a slight disadvantage since the stripes-and-arrow approximation has a drastically reduced number of parameters, compared to the fully-coupled approach, and we experimentally do not observe the same convergence problems when using standard optimisers (see Appx. G, Fig. S5).
# 3.3 Joint sampling of global and local latent variables
In contrast to our work, Refs. [9, 36] drop the Gaussian assumption over the inducing outputs fM and allow instead for potentially multi-modal approximate posteriors. While their approaches are arguably more expressive than ours, their flexibility comes at a price: the distribution over the inducing outputs fM is only given implicitly in form of Monte Carlo samples. Since the inducing outputs fM act as global latent parameters, the noise attached to their sampling-based estimates affects all samples from one mini-batch. This can often lead to higher variances which may translate to slower convergence [16]. We compare to Ref. [9] in our experiments.
# 4 Experiments
In Sec. 4.1, we study the predictive performance of our stripes-and-arrow approximation. Since it is difficult to assess accuracy and calibration on the same task, we ran a joint study of interpolation and extrapolation tasks, where in the latter the test points are distant from the training points. We found that the proposed approach balances accuracy and calibration, thereby outperforming its competitors on the combined task. Examining the results for the extrapolation task more closely, we find that our proposed method significantly outperforms the competing DGP approaches. In Sec. 4.2, we assess the runtime of our methods and confirm that our approximation has only a negligible overhead compared to mean-field and is more efficient than a fully-coupled DGP. Due to space constraints, we moved many of the experimental details to Appx. G.
# 4.1 Benchmark Results
We compared the predictive performance of our efficient stripes-and-arrow approximation (STAR DGP) with a mean-field approximation (MF DGP) [24], stochastic gradient Hamiltonian Monte Carlo (SGHMC DGP) [9] and a sparse GP (SGP) [12]. As done in prior work, we report results on eight UCI datasets and employ as evaluation criterion the average marginal test log-likelihood (tll). We assessed the interpolation behaviour of the different approaches by randomly partitioning the data into a training and a test set with a 90 : 10 split. To investigate the extrapolation behaviour, we created test instances that are distant from the training samples: We first randomly projected the inputs X onto a one-dimensional subspace z = Xw, where the weights w ∈RD were drawn from a standard Gaussian distribution. We subsequently ordered the samples w.r.t. z and divided them accordingly into training and test set using a 50 : 50 split. We first confirmed the reports from the literature [9, 24], that DGPs have on interpolation tasks an improved performance compared to sparse GPs (Tab. 1). We also observed that in this setting SGHMC outperforms the MF DGP and our method, which are on par. Subsequently, we performed the same analysis on the extrapolation task. While our approach, STAR DGP, seems to perform slightly better than MF DGP and also SGHMC DGP, the large standard errors of all methods hamper a direct comparison (see Tab. S3 in Appx. G). This is mainly due to the
Table 1: Interpolation behaviour on UCI benchmark datasets. We report marginal tlls (the larger, the better) for various methods, where L denotes the number of layers. Standard errors are obtained by repeating the experiment 10 times. We marked all methods in bold that performed better or as good as the standard sparse GP.
Dataset
SGP
SGHMC DGP
MF DGP
STAR DGP
(N,D)
L1
L1
L2
L3
L2
L3
L2
L3
boston (506,13)
-2.58(0.10)
-2.75(0.18)
-2.51(0.07)
-2.53(0.09)
-2.43(0.05)
-2.48(0.06)
-2.47(0.08)
-2.43(0.05)
energy (768, 8)
-0.71(0.03)
-1.16(0.44)
-0.37(0.12)
-0.34(0.11)
-0.73(0.02)
-0.75(0.02)
-0.75(0.02)
-0.75(0.02)
concrete (1030, 8)
-3.09(0.02)
-3.50(0.34)
-2.89(0.06)
-2.88(0.06)
-3.06(0.03)
-3.09(0.02)
-3.04(0.02)
-3.05(0.02)
wine red (1599,11)
-0.88(0.01)
-0.90(0.03)
-0.81(0.03)
-0.80(0.07)
-0.89(0.01)
-0.89(0.01)
-0.88(0.01)
-0.88(0.01)
kin8nm (8192, 8)
1.05(0.01)
1.14(0.01)
1.38(0.01)
1.25(0.14)
1.30(0.01)
1.31(0.01)
1.28(0.01)
1.29(0.01)
power (9568, 4)
-2.78(0.01)
-2.75(0.02)
-2.68(0.02)
-2.65(0.02)
-2.77(0.01)
-2.76(0.01)
-2.77(0.01)
-2.77(0.01)
naval (11934,16)
7.56(0.09)
7.77(0.04)
7.32(0.02)
6.89(0.43)
7.11(0.11)
7.05(0.09)
7.06(0.08)
6.25(0.31)
protein (45730, 9)
-2.91(0.00)
-2.76(0.00)
-2.64(0.01)
-2.58(0.01)
-2.83(0.00)
-2.79(0.00)
-2.83(0.00)
-2.80(0.00)
Dataset
MF vs. STAR
SGHMC vs. STAR
boston
0.55(0.04)
0.50(0.05)
energy
0.73(0.05)
0.60(0.04)
concrete
0.57(0.04)
0.60(0.03)
wine red
0.57(0.04)
0.63(0.02)
kin8nm
0.36(0.03)
0.44(0.05)
power
0.44(0.06)
0.64(0.03)
naval
0.67(0.06)
0.58(0.03)
protein
0.49(0.03)
0.50(0.03)
Table 2: Extrapolation behaviour: direct comparison of DGP methods. Average frequency µ and its standard error σ (computed over 10 repetitions) of the STAR DGP outperforming the MF DGP (left) and the SGHMC DGP (right) on the marginal tll of individual repetitions of the extrapolation task (see main text for details). Results are for DGPs with three layers. We mark numbers in bold (italics) if STAR outperforms its competitor (vice versa).
random 1D-projection of the extrapolation experiment: The direction of the projection has a large impact on the difficulty of the prediction task. Since this direction changes over the repetitions, the corresponding test log-likelihoods vary considerably, leading to large standard errors. We resolved this issue by performing a direct comparison between STAR DGP and the other two DGP variants: To do so, we computed the frequency of test samples for which STAR DGP obtained a larger log-likelihood than MF/SGHMC DGP on each train-test split independently. Average frequency µ and its standard error σ were subsequently computed over 10 repetitions and are reported in Tab. 2. On 5/8 datasets STAR DGP significantly outperforms MF DGP and SGHMC DGP (µ > 0.50 + σ), respectively, while the opposite only occurred on kin8nm. In Tab. S4 in Appx. G, we show more comparisons, that also take the absolute differences in test log likelihoods into account and additionally consider the comparison of fully-coupled and MF DGP. Taken together, we conclude that our structured approximations are in particular beneficial in the extrapolation scenario, while their performance is similar to MF DGP in the interpolation scenario. Next, we performed an in-depth comparison between the approaches that analytically marginalise the inducing outputs: In Fig. 3 we show that the predicted variance σ2 ∗increased as we moved away from the training data (left) while the mean squared errors also grew with larger σ2 ∗(right). The mean squared error is an empirical unbiased estimator of the variance Var∗= E[(y∗−µ∗)2] where y∗is the test output and µ∗the mean predictor. The predicted variance σ2 ∗is also an estimator of Var∗. It is only unbiased if the method is calibrated. However, we observed for the mean-field approach that, when moving away from the training data, the mean squared error was larger than the predicted variances pointing towards underestimated uncertainties. While the mean squared error for SGP matched well with the predictive variances, the predictions are rather inaccurate as demonstrated by the large predicted variances. Our method reaches a good balance, having generally more accurate mean predictions than SGP and at the same time more accurate variance predictions than MF DGP. Finally, we investigated the behaviour of the SGHMC approaches in more detail. We first ran a one-layer model that is equivalent to a sparse GP but with a different inference scheme: Instead of marginalising out the inducing outputs, they are sampled. We observed that the distribution over the inducing outputs is non-Gaussian (see Appx. G, Fig. S6), even though the optimal approximate posterior distribution is provably Gaussian in this case [32]. A possible explanation for this are convergence problems since the global latent variables are not marginalised out, which, in turn, offers a potential explanation for the poor extrapolation behaviour of SGHMC that we observed in our experiments across different architectures and datasets. Similar convergence problems have also been observed by Ref. [25].
Figure 3: Calibration Study. Left: While the predicted variances increase for all methods as a function of the distance to the training data, we find that at any given distance, the uncertainty decreases from SGP to STAR DGP to MF DGP. Right: We plot the mean squared error as a function of the predicted variance. If the mean squared error is larger than the predicted variance, the latter underestimates the uncertainty. Results are recorded on the kin8nm UCI dataset and smoothed for plotting by using a median filter.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ffc8/ffc89bff-183c-4c76-b78b-2cf443ab8847.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0338/0338a3ff-c0a3-42a5-9423-b9f058354b87.png" style="width: 50%;"></div>
Figure 4: Runtime comparison. We compare the runtime of our efficient STAR DGP versus the FC DGP and the MF DGP on the protein UCI dataset. Shown is the runtime of one gradient step in seconds on a logarithmic scale as a function of the number of inducing points M. The dotted grey lines show the theoretical runtime O(M 2).
# 4.2 Runtime
We compared the impact of the variational family on the runtime as a function of the number of inducing points M. For the fully-coupled (FC) variational model, we also recorded the runtime when employing natural gradients [26]. The results can be seen in Fig. 4, where the order from fastest to slowest method was proportional to the complexity of the variational family: mean-field, stripes-and-arrow, fully-coupled DGP. For our standard setting, M = 128, our STAR approximation was only two times slower than the mean-field but three times faster than FC DGP (trained with Adam [15]). This ratio stayed almost constant when the number of inducing outputs M was changed, since the most important term in the computational costs scales as O(M 2) for all methods. Subsequently, we performed additional experiments in which we varied the architecture parameters L and τ. Both confirm that the empirical runtime performance scales with the complexity of the variational family (see Appx. G, Fig. S7) and matches our theoretical estimates in Sec. 3.2.
# 5 Summary
In this paper, we investigated a new class of variational families for deep Gaussian processes (GPs). Our approach is (i) efficient as it allows to marginalise analytically over the global latent variables and (ii) expressive as it couples the inducing outputs across layers in the variational posterior. Naively coupling all inducing outputs does not scale to large datasets, hence we suggest a sparse and structured approximation that only takes the most important dependencies into account. In a joint study of interpolation and extrapolation tasks as well as in a careful evaluation of the extrapolation task on its own, our approach outperforms its competitors, since it balances accurate predictions and calibrated uncertainty estimates. Further research is required to understand why our structured approximations are especially helpful for the extrapolation task. One promising direction could be to look at differences of inner layer outputs (as done in Ref. [34]) and link them to the final deep GP outputs. There has been a lot of follow-up work on deep GPs in which the probabilistic model is altered to allow for multiple outputs [14], multiple input sources [8], latent features [25] or for interpreting the latent states as differential flows [11]. Our approach can be easily adapted to any of these models and is therefore a promising line of work to advance inference in deep GP models. Our proposed structural approximation is only one way of coupling the latent GPs. Discovering new variational families that allow for more speed-ups either by applying Kronecker factorisations as done in the context of neural networks [18], placing a grid structure over the inducing inputs [13], or by taking a conjugate gradient perspective on the objective [35] are interesting directions for future research. Furthermore, we think that the dependence of the optimal structural approximation on various factors (model architecture, data properties, etc.) is worthwhile to be studied in more detail.
In many applications, machine learning algorithms have been shown to achieve superior predictive performance compared to hand-crafted or expert solutions [27]. However, these methods can be applied in safety-critical applications only if they return predictive distributions allowing to quantify the uncertainty of the prediction [17]. For instance, a medical diagnosis tool can be applied only if each diagnosis is endowed with a confidence interval such that in case of ambiguity a physician can be contacted. Our work yields accurate predictive distributions for deep non-parametric models by allowing correlations between and across layers in the variational posterior. As we validate in our experiments, this also holds true when the input distribution at test time differs from the input distribution at training time. In our medical example, this might be the case if the hospital where the data is recorded is different from the one where the diagnosis tool is deployed.
# Acknowledgements
We thank Buote Xu for valuable comments and suggestions on an early draft of the paper. We furthermore acknowledge the detailed and constructive feedback from the four anonymous reviewers, particularly for suggesting a new experiment which lead to Fig. 2.
# References
[1] Mart´ın Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: a system for large-scale machine learning. In USENIX Symposium on Operating Systems Design and Implementation, 2016. [2] Shun-Ichi Amari. Natural gradient works efficiently in learning. Neural Computation, 1998. [3] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Man´e. Concrete problems in AI safety. arXiv preprint arXiv:1606.06565, 2016. [4] David Blei, Alp Kucukelbir, and Jon McAuliffe. Variational inference: A review for statisticians. Journal of the American Statistical Association, 2017. [5] Thang Bui, Daniel Hern´andez-Lobato, Jose Hernandez-Lobato, Yingzhen Li, and Richard Turner. Deep gaussian processes for regression using approximate expectation propagation. In International Conference on Machine Learning, 2016. [6] Andreas Damianou and Neil Lawrence. Deep gaussian processes. In Artificial Intelligence and Statistics, 2013. [7] David Duvenaud, Oren Rippel, Ryan P. Adams, and Zoubin Ghahramani. Avoiding pathologies in very deep networks. In Artificial Intelligence and Statistics, 2014. [8] Oliver Hamelijnck, Theodoros Damoulas, Kangrui Wang, and Mark Girolami. Multiresolution multi-task gaussian processes. In Advances in Neural Information Processing Systems, 2019. [9] Marton Havasi, Jos´e Lobato, and Juan Fuentes. Inference in deep gaussian processes using stochastic gradient hamiltonian monte carlo. In Advances in Neural Information Processing Systems, 2018. 10] Ali Hebbal, Loic Brevault, Mathieu Balesdent, El-Ghazali Talbi, and Nouredine Melab. Bayesian optimization using deep gaussian processes. arXiv preprint arXiv:1905.03350, 2019. 11] Pashupati Hegde, Markus Heinonen, Harri L¨ahdesm¨aki, and Samuel Kaski. Deep learning with differential gaussian process flows. In Artificial Intelligence and Statistics, 2019. 12] James Hensman, Nicolo Fusi, and Neil Lawrence. Gaussian processes for big data. Conference on Uncertainty in Artifical Intelligence, 2013. 13] Pavel Izmailov, Alexander Novikov, and Dmitry Kropotov. Scalable gaussian processes with billions of inducing inputs via tensor train decomposition. Artificial Intelligence and Statistics, 2018.
[14] Markus Kaiser, Clemens Otte, Thomas Runkler, and Carl Henrik Ek. Bayesian alignments of warped multi-output gaussian processes. In Advances in Neural Information Processing Systems, 2018. [15] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference of Learning Representations, 2015. [16] Diederik Kingma, Tim Salimans, and Max Welling. Variational dropout and the local reparameterization trick. In Advances in Neural Information Processing Systems, 2015. [17] Christian Leibig, Vaneeda Allken, Murat Sec¸kin Ayhan, Philipp Berens, and Siegfried Wahl. Leveraging uncertainty information from deep neural networks for disease detection. Scientific reports, 2017. [18] James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In International Conference on Machine Learning, 2015. [19] Alexander Matthews, Mark Van Der Wilk, Tom Nickson, Keisuke Fujii, Alexis Boukouvalas, Pablo Le´on-Villagr´a, Zoubin Ghahramani, and James Hensman. Gpflow: A gaussian process library using tensorflow. Journal of Machine Learning Research, 2017. [20] Lutz Prechelt. Early stopping-but when? In Neural Networks: Tricks of the trade. Springer, 1998. [21] Carl Rasmussen and Christopher Williams. Gaussian processes for machine learning. The MIT Press, 2005. [22] David Reeb, Andreas Doerr, Sebastian Gerwinn, and Barbara Rakitsch. Learning gaussian processes by minimizing pac-bayesian generalization bounds. In Advances in Neural Information Processing Systems, 2018. [23] Danilo Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In International Conference on Machine Learning, 2014. [24] Hugh Salimbeni and Marc Deisenroth. Doubly stochastic variational inference for deep gaussian processes. In Advances in Neural Information Processing Systems, 2017. [25] Hugh Salimbeni, Vincent Dutordoir, James Hensman, and Marc Peter Deisenroth. Deep gaussian processes with importance-weighted variational inference. International Conference on Machine Learning, 2019. [26] Hugh Salimbeni, Stefanos Eleftheriadis, and James Hensman. Natural gradients in practice: non-conjugate variational inference in gaussian process models. In Artificial Intelligence and Statistics, 2018. [27] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. Nature, 2016. [28] Dennis Smolarski. Diagonally-striped matrices and approximate inverse preconditioners. Journal of Computational and Applied Mathematics, 2006. [29] Edward Snelson and Zoubin Ghahramani. Sparse gaussian processes using pseudo-inputs. In Advances in Neural Information Processing Systems, 2006. [30] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. Practical bayesian optimization of machine learning algorithms. In Advances in neural information processing systems, 2012. [31] Jasper Snoek, Yaniv Ovadia, Emily Fertig, Balaji Lakshminarayanan, Sebastian Nowozin, D Sculley, Joshua Dillon, Jie Ren, and Zachary Nado. Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift. In Advances in Neural Information Processing Systems, 2019. [32] Michalis Titsias. Variational learning of inducing variables in sparse gaussian processes. In Artificial Intelligence and Statistics, 2009. [33] Richard Turner and Maneesh Sahani. Two problems with variational expectation maximisation for time-series models. Bayesian Time Series Models, 2011. [34] Ivan Ustyuzhaninov, Ieva Kazlauskaite, Markus Kaiser, Erik Bodin, Neill D. F. Campbell, and Carl Henrik Ek. Compositional uncertainty in deep gaussian processes. In Conference on Uncertainty in Artifical Intelligence, 2020.
[35] Ke Wang, Geoff Pleiss, Jacob Gardner, Stephen Tyree, Kilian Q. Weinberger, and Andrew Gordon Wilson. Exact gaussian processes on a million data points. In Advances in Neural Information Processing Systems, 2019. [36] Haibin Yu, Yizhou Chen, Zhongxiang Dai, Kian Hsiang Low, and Patrick Jaillet. Implicit posterior variational inference for deep gaussian processes. In Advances in Neural Information Processing Systems, 2019. [37] Christoph Zimmer, Mona Meister, and Duy Nguyen-Tuong. Safe active learning for time-series modeling with gaussian processes. In Advances in Neural Information Processing Systems, 2018.
# Beyond the Mean-Field: Structured Deep Gaussian Processes Improve the Predictive Uncertainties
# Beyond the Mean-Field:
# A Marginalisation of the inducing outputs (proof of Theorem 1)
A Marginalisation of the inducing outputs (proof of Theorem 1)
The aim of this section is to provide a complete proof for Thm. 1. We will do this by starting from the formula q(f l n) that we work out in Appx. D, � �
� Comparing to Eq. (9), we see that it remains to be shown that indeed
� � where the distributions q on the right hand side have the properties described in Eqs. (9) - (11). The terms appearing on the left hand side are given by q(fM) = N (fM|µM, SM), which is interchangeably also denoted as      �     
and
and
where
� � � In order to show that Eq. (17) holds, we will introduce a rather technical lemma in the following and prove it later  induction. Lemma 2. Given the definitions in Eqs. (18) and (19), ∀l = 1, . . . , L we have � �
� � � In order to show that Eq. (17) holds, we will introduce a rather technical lemma in the following and prove it later b induction.
 ���  � �   re ˆµl n and ˆΣl n are as in Eqs. (10) and (11), respectively, and we defined lˆµl+1:L M = µl+1:L M + Sl+1:L,1:l−1 M diag( �K1:l−1 Mn ) � �S1:l−1,1:l−1 n �−1 (f 1:l−1 n −�µ1:l−1 n ) l ˆΣl+1:L,l+1:L M = Sl+1:L,l+1:L M −Sl+1:L,1:l−1 M diag( �K1:l−1 Mn ) � �S1:l−1,1:l−1 n �−1 diag( �K1:l−1 nM )S1:l−1,l+1: M l ˆΣl,l+1:L nM = �Kl nMSl,l+1:L M −�Sl,1:l−1 n � �S1:l−1,1:l−1 n �−1 diag( �K1:l−1 nM )S1:l−1,l+1:L M .
(17)
(18)
(20) (21)
(22)
(23)
(24)
In the equations above we used diag(A1:l) to denote the formation of a block diagonal matrix, where the diagonal blocks are given by A1, . . . , Al. Note that while we only need one index to label ˆµl n and ˆΣl n, we need several for the objects defined in Eq. (24). Take e.g. l ˆΣl+1:L,l+1:L M : The upper left index denotes for which l the formula is valid (which will become important when we do the induction step l →l+1). The upper right indices (try to) capture which terms of SM are most important for the definition, they have nothing to do with the dimensionality of the objects. (In fact, the matrix l ˆΣM contains L −l −1 × L −l −1 blocks of various sizes TlM × Tl′M.) This makes it easier later on when we do calculations with these objects. Before we prove Lem. 2, we will first show how its results can be used to prove Thm. 1:
Proof of Theorem 1. As shown in Appx. D, we can write
Obtaining a formula for the inner integral can be done using Lem. 2 with l = L, in which case Eq. (22) read
Plugging this into Eq. (25) yields
where the distributions q on the right hand side have the properties described in Eqs. (9) - (11). In order to prove Lem. 2, we will regularly need two standard formulas from Gaussian calculus, namely conditioni multivariate Gaussians, �� �� � � � ��
where the distributions q on the right hand side have the properties described in Eqs. (9) - (11). In order to prove Lem. 2, we will regularly need two standard formulas from Gaussian calculus, namely conditionin multivariate Gaussians, N �� x y ����� � a b � , �A C C⊤ B �� = N (x|a, A) N � y ��b + C⊤A−1(x −a), B −C⊤A−1C � , (29
��� and solving Gaussian integrals (”propagation”):
� N (x|a + Fy, A) N (y|b, B) dy = N � x ��a + Fb, A + FBF ⊤�
� Proof of Lemma 2. As already said, we prove the lemma by induction: Base case We need to show that Eq. (22) holds for l = 1, i.e., that
� q(f 1 M, . . . , f L M) L � l′=1 p(f l′ n |f l′ M)df l′ M = � q(f 1 n, f 2 M, . . . , f L M) L � l′=2 p(f l′ n |f l′ M)df l′ M,
where q(f 1 n, f 2 M, . . . , f L M) is given according to Eqs. (23) and (24) In order to do so, we will perform the following steps:
(25)
(26)
(27)
(28)
(29)
(30)
(31)
v) Then we evaluate the integral:
� q(f 1 M)q(f 1 n, f 2 M, . . . , f L M|f 1 M)df 1 M = q(f 1 n, f 2 M, . . . , f L M).
� �� � �� � � � � For step iii) we use the formula we just obtained for q(f 2 M, . . . , f L M|f 1 M) and additionally p(f 1 n|f 1 M), which, accord to Eq. (19) is given by N � f 1 n ����K1 nMf 1 M, �K1 nn � , and then proceed to build their joint Gaussian distribution:
�� � � � � In step iv) we perform the integration using the term above for the joint and q(f 1 M) = N � f 1 M ��µ1 M, S11 M � from Eq. (36) for the marginal. Applying Eq. (30) yields � q(f 1 n, f 2 M, . . . , f L M|f 1 M)q(f 1 M)df 1 M
�� � � � � In step iv) we perform the integration using the term above for the joint and q(f 1 M) = N � f 1 M ��µ1 M, S11 M � from Eq. (36) for the marginal. Applying Eq. (30) yields �
��� � � �   � �K1 nn 0 0 S2:L,2:L M −S2:L,1 M � S11 M �−1 S1,2:L M � + � �K1 nM S2:L,1 M � S11 M �−1 � S11 M � �K1 nM S2:L,1 M � S11 M �−1 �⊤  �
�� � In order to arrive at the last line we simplified the terms and used the definitions of �µ1 n and �S11 n in Thm. 1.
(32)
(33)
(34)
(35)
(37)
��� � � which is the term q(f 1 n, f 2 M, . . . , f L M) on the RHS of Eq. (31). Plugging in the definitions from Eq. (24) we can easily verify that this last term indeed agrees with Eq. (38). Therefore our statement in Lem. 2 holds for l = 1. Inductive step We assume that Lemma 2 holds for some l = 1, . . . , L −1 (induction hypothesis) and then need to show that it also holds for l + 1. That is, assuming that � �
��� � � which is the term q(f 1 n, f 2 M, . . . , f L M) on the RHS of Eq. (31). Plugging in the definitions from Eq. (24) we can easily verify that this last term indeed agrees with Eq. (38). Therefore our statement in Lem. 2 holds for l = 1. Inductive step We assume that Lemma 2 holds for some (induction hypothesis) and then need to
Inductive step We assume that Lemma 2 holds for some l = 1, . . . , L −1 (induction hypothesis) and then need to show that it also holds for l + 1. That is, assuming that � �
� � � holds for some l with the terms on the RHS given by Eqs. (23), and (24) we need to show that we can also write th previous equation as
� � � holds for some l with the terms on the RHS given by Eqs. (23), and (24) we need to show that we can also write  previous equation as
� � where this time the terms are given by Eqs. (23), and (24) but with l →l + 1. The way to show this is very similar to the way we showed the base case, the resulting formulas will only look more complicated and we will need one additional step in the beginning:
 → The way to show this is very similar to the way we showed the base case, the resulting formulas will only look more complicated and we will need one additional step in the beginning:
o) Assuming that Eq. (40) holds for some l, we can start immediately with the RHS. The first step will be to marginalise f l n from the distribution q within the integral and show that the resulting marginal q(f l n|f 1:l−1 n ) has the right form to be written as part of the product in front of the integral: � �
� � Having done this, we will have to do the exact same steps as in the base case, which we will repeat below wi updated indices.
i) Continuing from Eq. (44), we isolate all terms that depend on f l+1 M : � �
ii) Comparing this to Eq. (41), we see that it remains to be shown that the inner integral equals q(f l+1 n , f l+2:L M |f 1: n  [given by Eqs. (23) and (24)]. Therefore we only consider the inner integral and therein condition q on f l+1 M : � �
) Next, we obtain the joint distribution of the two terms that are conditioned on f l+1 M : � q(f l+1 M |f 1:l n )q(f l+2:L M |f 1:l n , f l+1 M )p(f l+1 n |f l+1 M )df l+1 M = � q(f l+1 M |f 1:l n )q(f l+1 n , f l+2:L M |f 1:l n , f l+1 M )df l+1 M .
(40)
(41)
(42)
(43)
(44)
(45)
(46)
(47)
v) Finally, we check that the resulting distribution is given by Eqs. (23) and (24). This then proves the eq Eqs. (40) and (41).
v) Finally, we check that the resulting distribution is given by Eqs. (23) and (24). This then proves the equality of Eqs. (40) and (41). Let us begin with step o): According to Eq. (23), we have
Let us begin with step o): According to Eq. (23), we have  �
 ���  � � which we condition on f l n using Eq. (29) (i.e., going from Eq. (42) to Eq. (43)): q(f l n, f l+1:L M |f 1:l−1 n ) = q(f l n|f 1:l−1 n )q(f l+1:L M |f 1:l n ) = N � f l n ���ˆµl n, ˆΣl n � × � �
� ��� � N � f l+1:L M ���� lˆµl+1:L M + � l ˆΣl+1:L,l nM �⊤� ˆΣl n �−1 (f l n −ˆµl n), l ˆΣl+1:L,l+1:L M − � l ˆΣl+1:L,l nM �⊤� ˆΣl n �−1 l ˆΣl,l+1:L nM � . (50)
��� � �� � � �� � We therefore see that q(f l n|f 1:l−1 n ) = N � f l n ���ˆµl n, ˆΣl n � , which is the right form for it to be included in the product in front of the integral in Eq. (43). This lets us arrive at Eq. (44), hence finishing step o). In step i) nothing really happens, we just note that, according to Eq. (19), p(f l+1 n |f l+1 M ) = N � f l+1 n ����Kl+1 nMf l M, �Kl+1 nn � . (51)
���  � Using q(f l+1:L M |f 1:l n ) from Eq. (50), we perform step ii) according to Eq. (29), resulting in q(f l+1:L M |f 1:l n ) = q(f l+1 M |f 1:l n )q(f l+2:L M |f 1:l n , f l+1 M ),
where
q(f l+1 M |f 1:l n ) = N � f l+1 M ���� lˆµl+1 M + � l ˆΣl+1,l nM �⊤� ˆΣl n �−1 (f l n −ˆµl n), l ˆΣl+1,l+1 M − � l ˆΣl+1,l nM �⊤� ˆΣl n �−1 l ˆΣl,l+1 nM �
and
For step iii) we have to build the joint Gaussian distribution
q(f l+2:L M |f 1:l n , f l+1 M )p(f l+1 n |f l+1 M ) = q(f l+1 n , f l+2:L M |f 1:l n , f l+1 M )
| | | using Eqs. (51) and (54). Since this formula would be even longer than the one in Eq. (54), we refrain from explicitly writing it here. While the corresponding formula for the base case [Eq. (37)] is much simpler the resulting form of Eq. (55) would be similar.
(49)
(50)
(51)
(52)
� (53)
(55)
Next, the integration in step iv) can be performed using Eqs. (30), (53), and (55). The calculations are again ve similar to the ones in the corresponding step for the base case [Eq. (38)] so we only state the final result here: �
q(f l+1 n , f l+2:L M |f 1:l n ) = � q(f l+1 M |f 1:l n )q(f l+1 n , f l+2:L M |f 1:l n , f l+1 M )df l+1 M �
where
� �� � ns to be shown in step v) is that this result does in fact agree with the exp  � 
� �� � s that this result does in fact agree with the expected result from Lem. 2,  �   
 ���  � �   where the terms are defined in Eqs. (10), (11), and (24). That means we have to prove that ˆml+1 n = ˆµl+1 n and similarly for the other terms in Eqs. (58) - (61). Note that this is the point where we need the left indices in order to distinguish e.g. the term lˆµl+2:L M appearing in Eq. (58) from l+1ˆµl+2:L M appearing in the mean of Eq. (62).
  � � �  � � � where we used the definitions in Eqs. (10) and (24) for the termsˆ·. Note that these definitions are part of the induction hypothesis. It will soon become clear why we did not substitute ˆΣl n. We furthermore used the definitions of the �µn and �Sn terms in Thm. 1 to absorb the �K terms. In the following we are going to write Eq. (63) in a vectorized form and additionally substitute
� � � � ˆml+1 n = �µl+1 n + ��Sl+1,1:l−1 n �Sl+1,l n �⊤� A−1 + A−1B �D−1CA−1 −A−1B �D−1 −�D−1CA−1 �D−1 �� f 1:l−1 n −�µ1:l−1 n f l n −�µl n � , (66
(56)
(57)
(59)
(60)
(61)
(62)
(63)
(64)
(65)
(66)
where we additionally exploited that A and �D are symmetric and that B⊤= C. In order to get any further from he we need the block matrix inversion lemma, which states that
 �  � � � where �D = D−CA−1B. Comparing Eqs. (66) and (67) explains why we insisted on vectorising the last few formulas and also our definitions in Eq. (64). Finally, since ˆΣl n = �Sll n −�Sl,1:l−1 n � �S1:l−1,1:l−1 n �−1 �S1:l−1,l n [Eq. (11)], we also identify �Sll n = D. We can therefore rewrite Eq. (66) by reversing the block matrix inversion and resubstituting the terms in Eq. (64):
 � � � � � � = �µl+1 n + �Sl+1,1:l n � �S1:l,1:l n �−1 � f 1:l n −�µ1:l n � .
 �  � � � � � � � In the last step we simply rewrote the vectors and the matrix according to the way we defined the submatrix notation. Comparing the final result to Eq. (10), we realize that this is indeed ˆµl+1 n , i.e., the mean term where we substituted l →l + 1. In exactly the same way, i.e., by reversing the matrix inversion, we can show that the other parameters of the distribution in Eq. (56) indeed coincide with the respective parameters of the distribution in Eq. (62). Since this was the last part that remained to be shown, we finished the proof of Lem. 2.
# B Intuition for the proof of Theorem 1
In this section, we provide some intuition that might be helpful in understanding parts of the proof of Thm. 1. In the first part, we present a different, heuristic way of obtaining the same results, making use of a mathematically wrong (or at least not mathematically rigorous) step. This helped us come up with the exact form of the theorem that we proved above. The second part gives some intuition on how the formulas appearing in Thm. 1, especially Eqs. (10) and (11), can be interpreted. More precisely, we show how these reduce to the mean-field equations when we plug in the mean-field covariance matrix.
# B.1 Heuristic argument for Theorem 1
The aim of this section is to provide a heuristic argument for Thm. 1 as opposed to the complete proof given i Appx. A. For convenience we recap the starting point of that section: We need to show that
� � where the distributions q on the right hand side are defined in Eqs. (9) - (11). The terms appearing on the left han side are given by q(fM) = N (fM|µM, SM), which is interchangeably also denoted as �
and
where
���  � �Kl nM = Kl nM � Kl MM �−1 �Kl nn = Kl nn −Kl nM � Kl MM �−1 Kl Mn.
(67)
(68)
(69)
(70)
(71)
(72) (73)
   ���  �K     �K   Note that this is the point where this “proof” becomes heuristic: The object on the right hand side of Eq. (74) is not really a probability distribution, as the variables over which the distribution is defined (the f l n) appear as parameters of the distribution itself (as inputs of the covariance matrices, e.g. �Kl+1 nn ). In the following we will pretend that rules for (multivariate Gaussian) distributions still apply to this object, making the rest of this proof mathematically wrong. We hope that it can still provide some intuition. Using the standard formula for solving Gaussian integrals (”propagation”), � N (x|a + Fy, A) N (y|b, B) dy = N � x ��a + Fb, A + FBF ⊤� , (75)
� � �� we can then easily plug in Eq. (74) in the left hand side of Eq. (69), yielding    �   
where
�   �  � �   � � Note that the latter two definitions also appear in Thm. 1. The expression in Eq. (76) has still the same problem as the expression in Eq. (74) in that it is not a valid distribution (since �µl n and �Sll′ n depend on f l−1 n ). Pretending further, that rules for distributions still apply, we can use the standard formula for conditioning multivariate Gaussians, �� �� � � � ��
�� ����� � � � �� o repeatedly condition Eq. (76), resulting in
 � � � � � � In Eqs. (81) and (82) the notation Al,1:l′ is used to index a submatrix of the variable A, e.g. Al,1:l′ = � Al,1 · · · Al,l′� . The final result, Eqs. (80) - (82), is once again a valid distribution and exactly matches the outcome of the mathematically rigorous proof of Thm. 1 in Appx. A. The latter, while being much more complicated, is necessary since the intermediate expressions [Eqs. (74) and (76)] rely on mathematically wrong (or at least dubious) steps.
# B.2 Mean-field as a structured approximation
re, we verify that when we plug in the mean-field covariance matrix into the formulas appearing in Thm. 1, we cover the mean-field formulas appearing in Sec. 2.2, i.e., that in this case Eq. (9) reduces to Eq. (6). For convenience  repeat the relevant formulas, starting with the mean-field formula for the marginals of the last layer [Eq. (6)]: q(f L n ) = � L � l=1 q(f l n; f l−1 n )df 1 n · · · df L−1 n , where q(f l n; f l−1 n ) = Tl � t=1 N � f l,t n ����µl,t n , �Σl,t n � , (83)
Here, we verify that when we plug in the mean-field covariance matrix into the formulas appearing in Thm. 1, we recover the mean-field formulas appearing in Sec. 2.2, i.e., that in this case Eq. (9) reduces to Eq. (6). For convenience we repeat the relevant formulas, starting with the mean-field formula for the marginals of the last layer [Eq. (6)]:
q(f L n ) = � L � l=1 q(f l n; f l−1 n )df 1 n · · · df L−1 n , where q(f l n; f l−1 n ) = Tl � t=1 N � f l,t n ����µl,t n , �Σl,t n � ,
(74)
(75)
(76)
(77) (78)
(79)
(80)
(81)
(82)
q(f L n ) = � � q(f l n|f 1 n, . . . , f l−1 n )df 1 n · · · df L−1 n where q(f l n|f 1 n, . . . , f l−1 n ) = N � f l n ���ˆµl n, ˆΣl n � ,
where the means and covariances of the Gaussians in this equation are given by:
where
� � � � � Here we introduced Kl = � ITl ⊗Kl� as shorthand for the Kronecker product between the identity matrix ITl and the covariance matrix Kl, and used δ for the Kronecker delta. Having all relevant formulas in one place, we can proceed to show that if we plug in the mean field covariance matrix, SM = diag({Sll M}L l=1), where Sll M = diag({Sl,t M}Tl t=1) (see also Fig. 1, left), in Eq. (85), we recover Eq. (83): Removing the correlations between the layers by setting SM = diag({Sll M}L l=1), also implies that ˜Sll′ n = 0 if l ̸= l′ [Eq. (89)]. Therefore Eqs. (86) and (87) reduce to ˆµl n = �µl n and ˆΣl n = �Sll n, respectively. The resulting variational posterior factorises between the layers with q(f l n; f l−1 n ) = N � f l n ����µl n, �Sll n � . Comparing with Eq. (83), we can already see that the means are equal [since �µl n = (�µl,1 n , . . . , �µl,Tl n ), cf. Eqs. (84) and (88)]. Removing the correlations within one layer by setting Sll M = diag({Sl,t M}Tl t=1) renders the covariance matrix �Sll n block-diagonal. The diagonal blocks are obtained by evaluating Eq. (89) with Sll M = diag({Sl,t M}Tl t=1). It is easy to see that these diagonal blocks are equal to �Σl,t n in Eq. (84) and we fully recover the mean-field solution [Eq. (83)].
� � � � � Here we introduced Kl = � ITl ⊗Kl� as shorthand for the Kronecker product between the identity matrix ITl and the covariance matrix Kl, and used δ for the Kronecker delta.
# � C ELBO
Here we show how to derive the ELBO of the FC DGP, i.e., Eq. (8), which is given by
L � n=1 � | � −|| � l=1 While this is already done in the supplemental material of Ref. [24], we will do the derivation once again since our notation is different. For convenience we repeat the relevant formulas from the main text, i.e., the general formula for the ELBO [Eq. (3)], the joint DGP prior [Eq. (2)], and the variational family for the DGP [Eq. (4)], which are given by
� � While this is already done in the supplemental material of Ref. [24], we will do the derivation once again since our notation is different. For convenience we repeat the relevant formulas from the main text, i.e., the general formula for he ELBO [Eq. (3)], the joint DGP prior [Eq. (2)], and the variational family for the DGP [Eq. (4)], which are given by
(84)
(85)
(86)
(88) (89)
(88)
(90)
(91)
(93)
respectively. By only using the general form of the distributions, and exploiting that we assumed iid noise, i.e., p(yN|f L N) = �N n=1 p(yn|f L n ), we can get from Eq. (91) to Eq. (90):
In the last step we introduced q(f L n ) as simply summarising all remaining terms in the first integral, hence,
# D Marginalisation of (most) latent layer outputs
Here we show how to get from the general form of q(f L n ) given in Eq. (98) to the starting point of our induction proof [Eq. (12)], where all the latent outputs f l n′ are integrated out for all layers l and for all samples n′ ̸= n:
While this is already shown in Remark 2 in Ref. [24] (note that the indices there are not correct), we will provide a bit more detail here and we can also nicely point out where the difference in the formulas for q(f L n ) arises from. For convenience the relevant formulas are repeated below: q(fN, fM) = q(fM) L � l=1 p(f l N|f l M; f l−1 N ), p(f l N|f l M; f l−1 N ) = N � f l N ����Kl NMf l M, �Kl NN � , (100)
While this is already shown in Remark 2 in Ref. [24] (note that the indices there are not correct), we will provide a bit more detail here and we can also nicely point out where the difference in the formulas for q(f L n ) arises from. For convenience the relevant formulas are repeated below:
where �Kl NM = Kl NM � Kl MM �−1 and �Kl NN = Kl NN −Kl NM � Kl MM �−1 Kl MN. We will start by explicitly writing out Eq. (98) and changing the order of integration: 
  In the following we will only be concerned with the inner integral of the previous equation, which can also be wr as
 �  � ere, the inner integral can be solved by exploiting the nice marginalisation property of multivariate Gaussi � p(f L N|f L M; f L−1 N ) � n′̸=n df L n′ = � N � f L N ����KL NM(f L−1 N )f L M, �KL NN(f L−1 N ) �� n′̸=n df L n′ = N � f L ����KL (f L−1 )f L , �KL (f L−1 ) � ,
 �  � e, the inner integral can be solved by exploiting the nice marginalisation property of multivariate Gaussians, � � �
(95)
(96)
(97)
(98)
(99)
(101)
(102)
(103)
(104)
where we explicitly marked the dependence of the �KL terms on the outputs f L−1 N . While the �Kl nM and �Kl nn could  principle still depend on all the outputs of the previous layer, we see from their definitions after Eq. (100) that they  fact only depend on the marginals f L−1 n and that therefore
� ����  � � Putting the last two equations together results in
We can continue with integrating out the f l N in Eq. (102) in the same fashion (noting at every layer that we can not marginalise out f l n as those are inputs to kernels), arriving at
ng this back into Eq. (101) and changing the order of integration once ag
which is exactly Eq. (99), the result that we set out to show.
# D.1 Difference between mean-field and fully-coupled
From the previous equation it is also possible to see why a proof as in Appx. A was not necessary for the MF DGP This is due to the form of the variational posterior over the inducing outputs, given by
� � � ��� � Using q(fM) from the MF DGP, which can also be written as q(fM) = �L l=1 q(f l M), the inner integral in Eq. (108 can be rewritten as the product of l integrals,
each being a standard integral in Gaussian calculus and the resulting formulas are given in Eqs. (6) and (9). In contrast, a fully coupled multivariate Gaussian can not be written as such a product, which is why the rather straightforward solution presented above is not possible in our case and the proof in Appx. A is needed.
# E Linear algebra to speed up the code
In this section we provide some guidance through the linear algebra that is exploited in our code to speed up or vectorise calculations. We will focus only on the most expensive terms, i.e., the off-diagonal covariance term �Sl,1:l−1 n and how to deal with � �S1:l−1,1:l−1 n �−1 , which are both needed to calculate ˆµl n and ˆΣl n in Eqs. (111) and (112), respectively. First, we show how to deal with the FC DGP and afterwards how the sparsity of SM for the STAR DGP can be used. The linear algebra that can be exploited for all the other terms, e.g. the KL-divergence, will be provided along with the code. Our implementation is in GPflow [19] which provides all the functionalities that are necessary to deal with GPs in Tensorflow [1].
(105)
(106)
(107)
(108)
(109)
(110)
� � � �  � Additionally, here is a more explicit definition of our notation of the covariance matrix SM: � � � �
 � � � �  where SM, Sll′ M, and � Sll′ M � tt′ are matrices of size MT × MT (where T = �L l=1 Tl), MTl × MTl′, and M × M, which store the covariances of the inducing outputs between all inducing points, only those between layer l and l′, and only those between the t-th task in layer l and the t′-th task in layer l′, respectively. In order to ensure that SM is a valid covariance matrix (positive definite) we will numerically only work with its Cholesky decomposition LS (s.t. SM = LSL⊤ S ), which is a lower triangular matrix. Wherever possible we will want to avoid actually computing SM and instead calculate all quantities from LS directly.
# E.1 Fully coupled DGP
� �� � � � which are of size Tl × �l−1 l′=1 Tl′, have to be calculated for l = 1, . . . , L and for n = 1, . . . , N. As the number of layers L is in practice rather small, we will calculate all the individual matrices �Sll′ n in a loop and concatenate them at the end, while we want to avoid a loop over N. Using Eq. (113) and that �Kl Mn = ITl ⊗�Kl Mn, we see that (for an example with Tl, Tl′ = 2)  
�  � � �  � � � �  �  Writing �Sll′ n in this way has two advantages: Firstly, actually performing the multiplication �Kl nMSll′ M �Kl′ Mn is extremely inefficient as the �Kl Mn are block diagonal, which we resolved in this formulation. Secondly, we note that exactly the same operation, i.e., multiplying from left and right by � �Kl Mn �⊤ and �Kl′ Mn, respectively, has to be performed on all TlTl′ blocks of size M × M. This can be exploited since tensorflow has an inbuilt batch mode for most of its matrix operations. In the following we will first show how the relevant block Sll′ M can be efficiently obtained and afterwards show how to deal with the batch matrix multiplication for all n = 1, . . . , N. Let us consider an example with three layers (L = 3), the resulting covariance matrix and its Cholesky decomposition:  
 � �  From this we can read off formulas for the blocks of SM, e.g., S32 M = � L31 S L32 S �� L21 S L22 S �⊤, which in general can be written as Sll′ M = Ll,1:l′ S � Ll′,1:l′ S �⊤ , (118)
(111)
(112)
(113)
(114)
(115)
(116)
(117)
(118)
where we exploited that we only need Sll′ M for l′ < l (the formula above is not valid for l′ ≥l). In this way we avoided calculating unnecessary matrix multiplications involving zero blocks. Avoiding the loop over N requires a bit more linear algebra: For this we note that e.g. the element � �Sll n � 11 can be seen as the n-th diagonal element of the N × N matrix � �Kl MN �⊤ (Sll′ M)11 �Kl′ MN. Fully calculating this matrix is obviously very inefficient as we only need its diagonal elements. For this we use that, generally, for q × p matrices A, C⊤and q × q matrices B diag � C⊤BA � = column sum(C⊤⊙BA), (119) where ⊙denotes the elementwise matrix product. The formula can easily be proved by explicitly writing the matrix products as sums and comparing terms on both sides. Using this on all the blocks of �Sll′ n in Eq. (116) in a batched form and reordering the obtained terms afterwards requires some reshaping, which is explained in the code. The most expensive calculations for this term are obtaining Sll′ M [Eq. (118)], which is O(M 3TlTl′ �l′ l′′=1 Tl′′) and the multiplication of e.g. (Sll′ M)11 �Kl′ MN which has to be done for all TlTl′ blocks of Sll′ M and is therefore O(NM 2TlTl′). Both of these operations have to be performed for all l′ = 1, . . . , l −1 in Eq. (115) and also for all layers l = 1, . . . , L. The total computational cost of this term is therefore O(M 3 �L l=1 Tl �l−1 l′=1 Tl′ �l′ l′′=1 Tl′′ + NM 2 �L l=1 Tl �l−1 l′=1 Tl′).
diag � C⊤BA � = column sum(C⊤⊙BA), (119) where ⊙denotes the elementwise matrix product. The formula can easily be proved by explicitly writing the matrix products as sums and comparing terms on both sides. Using this on all the blocks of �Sll′ n in Eq. (116) in a batched form and reordering the obtained terms afterwards requires some reshaping, which is explained in the code. The most expensive calculations for this term are obtaining Sll′ M [Eq. (118)], which is O(M 3TlTl′ �l′ l′′=1 Tl′′) and the multiplication of e.g. (Sll′ M)11 �Kl′ MN which has to be done for all TlTl′ blocks of Sll′ M and is therefore O(NM 2TlTl′). Both of these operations have to be performed for all l′ = 1, . . . , l −1 in Eq. (115) and also for all layers l = 1, . . . , L. The total computational cost of this term is therefore O(M 3 �L l=1 Tl �l−1 l′=1 Tl′ �l′ l′′=1 Tl′′ + NM 2 �L l=1 Tl �l−1 l′=1 Tl′).
 � � Dealing with the inverse covariance terms First of all, we will never actually calculate � �S1:l−1,1:l−1 n �−1 . We only use (and update) the lower triangular Cholesky decomposition L1:l−1,1:l−1 Sn defined by
� only use (and update) the lower triangular Cholesky decomposition L1:l−1,1:l−1 Sn defined by
 � This can be done since the inverse term only ever appears in the product �Sl,1:l−1 n � �S1:l−1,1:l−1 n �−1 , whose transpose can be efficiently obtained via the solution of two