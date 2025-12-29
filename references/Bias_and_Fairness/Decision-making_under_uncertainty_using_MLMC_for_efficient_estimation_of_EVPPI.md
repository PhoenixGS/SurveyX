# Decision-making under uncertainty: using MLMC for efficient estimation of EVPPI
19 Mar 2018
# Michael B. Giles∗, Takashi Goda†
March 20, 2018
In this paper we develop a very efficient approach to the Monte Carlo estimation of the expected value of partial perfect information (EVPPI) that measures the average benefit of knowing the value of a subset of uncertain parameters involved in a decision model. The calculation of EVPPI is inherently a nested expectation problem, with an outer expectation with respect to one random variable X and an inner conditional expectation with respect to the other random variable Y . We tackle this problem by using a Multilevel Monte Carlo (MLMC) method (Giles 2008) in which the number of inner samples for Y increases geometrically with level, so that the accuracy of estimating the inner conditional expectation improves and the cost also increases with level. We construct an antithetic MLMC estimator and provide sufficient assumptions on a decision model under which the antithetic property of the estimator is well exploited, and consequently a root-mean-square accuracy of ε can be achieved at a cost of O(ε−2). Numerical results confirm the considerable computational savings compared to the standard, nested Monte Carlo method for some simple testcases and a more realistic medical application.
# 1 Introduction
The motivating applications for this research come from two apparently quite different fields, the funding of medical research and the exploration and exploitation of oil and gas reservoirs. The common element in both cases is decision making under a large degree of uncertainty. In the medical case (Ades et al. 2004, Brennan et al. 2007) let X and Y represent independent random variables representing the uncertainty in the effectiveness of different medical treatments. In the absence of any knowledge of X or Y , then given a finite set of possible treatments D, the optimal choice dopt is the one which maximises E [fd(X, Y )] where fd(X, Y ) represents some measure of the patient outcome, such as QALY’s (quality-adjusted life-year), measured on a monetary scale with a larger value being better. Thus, with no
∗Mathematical Institute, University of Oxford, Oxford, United Kingdom, OX2 6GG mike.giles@maths.ox.ac.uk †School of Engineering, University of Tokyo, Tokyo 113-8656, Japan, goda@frcer.t.u tokyo.ac.jp
max d∈D E [fd(X, Y )] .
On the other hand, given perfect information on X and Y , through carrying out some new medical research, the best treatment choice maximises fd(X, Y ), giving the overall average outcome
In the intermediate situation, if X is known but not Y , then the best treatment has average outcome value
EVPI, the expected value of perfect information, is the difference
EVPPI = E � max d E [fd(X, Y ) | X] � −max d E[fd(X, Y )].
EVPPI represents the benefit, on average, of knowing the value of X. If the value of X represents the information arising from a proposed piece of medical research, then one can compare the cost of the research to the benefits which arise from the information obtained. In the oil and gas reservoir scenario (Bratvold et al. 2009, Nakayasu et al. 2016), there are also decisions to be made, such as whether or not to drill additional exploratory wells. There is huge uncertainty in various aspects of an oil reservoir, its dimensions, the oil and gas reserves it contains, the rock porosity, etc. An additional well will yield information which will reduce the uncertainty and increase, on average, the amount of oil and gas which will eventually be extracted. However, there is an additional cost in drilling one more well, and the EVPPI will help determine whether or not it is worth it. The calculation of EVPPI is a nested expectation problem, with an outer expectation over X and an inner conditional expectation over Y . In this paper, we choose to focus on the estimation of the difference
EVPI can be estimated directly using standard Monte Carlo methods with  dependent samples of (X, Y )
1 N N � n=1 max d fd(X(n), Y (n)) −max d 1 N N � n=1 fd(X(n), Y (n)).
Assuming each computation fd(X, Y ) can be performed with unit cost, EVPI can be estimated with root-mean-square accuracy ε by using N = O(ε−2) samples (X(n), Y (n)) at a total cost which is O(ε−2). On the other hand, estimating the difference EVPI −EVPPI using standard, nested Monte Carlo methods requires N outer samples of X and M inner samples of Y , giving
As shown in the next section, in order to estimate EVPI −EVPPI with rootmean-square accuracy ε by this estimator, we need N = O(ε−2) and M = O(ε−1/α) samples for outer and inner expectations, respectively. Here α > 0 denotes the order of convergence of the bias and is typically between 1/2 and 1. Therefore, the computational complexity will be at least O(ε−3), and increase up to O(ε−4) in the worst case. The aim of this paper is to develop an efficient approach to this nested expectation problem, i.e., the estimation of EVPI−EVPPI, by using a Multilevel Monte Carlo (MLMC) method (Giles 2015). MLMC estimators have been used previously for nested expectations of the slightly different form E[f(E[Y |X])] by Haji-Ali (2012) and Giles (2015) for cases in which f is twice-differentiable, and by Bujok et al. (2015) for a case in which f is continuous and piecewise linear. Current research (Giles and Haji-Ali 2018) is also looking at the case in which f is a discontinuous indicator (Heaviside) function. Building on this prior MLMC research, we introduce an antithetic MLMC estimator for EVPI −EVPPI in the next section, and then in Section 3, we provide sufficient assumptions on fd’s such that the antithetic property of the estimator is well exploited, and by building upon the basic MLMC theorem (Theorem 1), the estimator is proven to achieve the optimal computational complexity O(ε−2) (Theorem 3). Numerical experiments in Section 4 confirm the importance of the assumptions made in our theoretical analysis, and also the considerable computational savings compared to the standard, nested Monte Carlo method not only for some simple testcases but also for a more realistic medical application.
# 2 MLMC method
# 2.1 Basic MLMC theory
The MLMC method was introduced by Heinrich (2001) for parametric integration, and by Giles (2008) for the estimation of the expectations arising from SDEs. It was subsequently extended to SPDEs (e.g. Cliffe et al. 2011), stochastic reaction networks (Anderson and Higham 2012), and nested simulation (Bujok et al. 2015, Haji-Ali 2012). For an extensive review of MLMC methods, see the review by Giles (2015). Here we give a brief overview of the MLMC method. The problem we are interested in is to estimate E[P] efficiently for a random output variable P which cannot be sampled exactly. Given a sequence of random variables P0, P1, . . . which approximate P with increasing accuracy but also with increasing cost,
The key idea behind the MLMC method is to independently estimate each of the quantities on the r.h.s. of (1) instead of directly estimating the l.h.s., which is the standard Monte Carlo approach. For the same underlying stochastic sample, Pℓand Pℓ−1 could be well correlated each other, and the variance of the correction Pℓ−Pℓ−1 is expected to get smaller as the level ℓincreases. Thus, in order to estimate each of the quantities on the r.h.s. of (1) with the same accuracy, the necessary number of samples for the finest levels becomes much smaller than that for the coarsest levels, resulting in a significant reduction of the total computational cost as compared to the standard Monte Carlo method. This observation leads to the following theorem (Giles 2015):
The key idea behind the MLMC method is to independently estimate each of the quantities on the r.h.s. of (1) instead of directly estimating the l.h.s., which is the standard Monte Carlo approach. For the same underlying stochastic sample, Pℓand Pℓ−1 could be well correlated each other, and the variance of the correction Pℓ−Pℓ−1 is expected to get smaller as the level ℓincreases. Thus, in order to estimate each of the quantities on the r.h.s. of (1) with the same accuracy, the necessary number of samples for the finest levels becomes much smaller than that for the coarsest levels, resulting in a significant reduction of the total computational cost as compared to the standard Monte Carlo method. This observation leads to the following theorem (Giles 2015): Theorem 1. Let P denote a random variable, and let Pℓdenote the corresponding level ℓnumerical approximation. If there exist independent random variables Zℓwith expected cost Cℓand variance Vℓ, and positive constants α, β, γ, c1, c2, c3 such that α ≥1 2 min(β, γ) and
then there exists a positive constant c4 such that for any ε<e−1 there are values L and Nℓfor which the multilevel estimator
with a computational complexity C with bound
 Remark 1. In the case where the condition Vℓ≤c22−βℓcan be replaced by E[Z2 ℓ] ≤c22−βℓ, H¨older’s inequality gives E[Zℓ] ≤ � E[Z2 ℓ] �1/2 ≤√c22−βℓ/2.
 Remark 1. In the case where the condition Vℓ≤c22−βℓcan be replaced b E[Z2 ℓ] ≤c22−βℓ, H¨older’s inequality gives
E[Zℓ] ≤ � E[Z2 ℓ] �1/2 ≤√c22−βℓ/2.
(1)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/061e/061e545d-9833-475e-bd74-c0ad53496e32.png" style="width: 50%;"></div>
<div style="text-align: center;">Compared this bound to the condition |E[Pℓ−P]| ≤c12−αℓ, we have α ≥β/2 and so the assumption α ≥1 2 min(β, γ) is simplified into α ≥γ/2.</div>
Compared this bound to the condition |E[Pℓ−P]| ≤c12−αℓ, we have α ≥β/2 and so the assumption α ≥1 2 min(β, γ) is simplified into α ≥γ/2.
 ≥  ≥ As far as possible, we try to develop MLMC estimators which are in the first regime, with β > γ, so that the total cost is O(ε−2). This corresponds to O(ε−2) samples each with an average O(1) cost, and it means that most of the computational cost is incurred on the coarsest levels. When the application is in this regime, Rhee and Glynn (2015) have a technique in which they randomise the selection of the level ℓto obtain a method which is unbiased but has a finite variance and average cost per sample. Nevertheless, in any regime, Theorem 1 compares favourably with the complexity bound for the standard Monte Carlo method which directly estimates the l.h.s. of (1) based on N Monte Carlo samples of PL for a fixed L:
In addition to the conditions given in Theorem 1, assume V := supℓV[Pℓ] < ∞. For a given accuracy ε, let us choose N = ⌈2V ε−2⌉and L = ⌈log2( √ 2c1ε−1)/α⌉, so that the variance and the bias of the estimator are bounded simultaneously:
and
which ensures the mean-square-error bound of ˆZ′
E � ( ˆZ′ −E[P])2� = V[ ˆZ′] + (E[P −PL])2 ≤ε2.
Then there exists a positive constant c5 such that the expected cost of ˆZ′ is bounded by
Then there exists a positive constant c5 such that the expected cost of ˆZ′ i bounded by
NCL ≤ � 2V ε−2 + 1 � c32γL ≤ � 2V ε−2 + 1 � c3 �√ 2c12αε−1�γ/α ≤c5ε−2−γ/α.
� � � � � � In general, it seems hard to improve the exponent 2 + γ/α of ε−1. Therefore, the multilevel estimator always has an asymptotically better complexity bound than the standard Monte Carlo estimator.
# 2.2 MLMC estimator for EVPPI
In view of the previous subsection, for the estimation of the difference EVPI − EVPPI let us define a random output variable P by
with the underlying stochastic variable X. Obviously P is nothing but the inner conditional expectation of EVPI −EVPPI, and the problem we tackle in this paper is rephrased into an efficient estimation of E[P]. A sequence of random variables P0, P1, . . . is defined by
Pℓ= 1 2ℓ 2ℓ � i=1 max d fd(X, Y (i)) −max d 1 2ℓ 2ℓ � i=1 fd(X, Y (i)) =: max d fd ℓ−max d fd ℓ
where maxd fd ℓand fd ℓrepresent averages over 2ℓindependent values of Y (i) for a randomly chosen X, respectively. That is to say, Pℓsimply denotes the standard Monte Carlo estimator based on 2ℓsamples for the inner conditional expectation of EVPI −EVPPI, so that the sequence P0, P1, . . . approximate P with increasing accuracy but also with increasing cost. Namely we have
# EVPI −EVPPI = E[P] = lim ℓ→∞E[Pℓ].
As discussed above, in order to achieve a given accuracy ε, the standard, nested Monte Carlo method chooses N = O(ε−2) and M = O(2L) = O(ε−1/α), and so the computational complexity is O(ε−2−1/α). Using the MLMC method, this can be reduced significantly. Following the ideas of Bujok et al. (2015), Giles (2015), Haji-Ali (2012), we use an “antithetic” MLMC estimator
in which
here, for a randomly chosen X, • fd (a) is an average of fd(X, Y ) over 2ℓ−1 independent samples for Y ; • fd (b) is an average over a second independent set of 2ℓ−1 samples; • fd is an average over the combined set of 2ℓinner samples.
It is straightforward to see that γ = 1 and E[Zℓ] = E[Pℓ−Pℓ−1] for ℓ> 0. Here we consider Z0 = P0 ≡0, so that the sum of the multilevel estimator over ℓ starts from ℓ= 1. Note that we have the antithetic property 1 2(fd (a) + fd (b)) −fd = 0, and therefore Zℓ= 0 if the same decision d maximises each of the terms in its definition. This is the key advantage of the antithetic estimator, compared to the alternative fd (a) −fd.
Remark 2. It is straightforward to extend the antithetic MLMC approach to estimate EVPI. The difference is that with EVPI all of the underlying random variables X and Y are inner variables; non are outer variables leading to a conditional expectation. Such an MLMC estimator for the maximum of an unconditional expectation has been introduced by Blanchet and Glynn (2015). As discussed in the introduction, however, EVPI can be estimated with O(ε2) complexity by using standard Monte Carlo methods already, so that the benefit is that one could use a randomisation technique by Rhee and Glynn (2015) to obtain an unbiased estimator, which might be marginal in the current setting.
# 3 MLMC variance analysis
We first show that the MLMC estimator achieves the nearly optimal complexity of O(ε−2(log ε)2) under a quite mild assumption. Theorem 2. If E [V[fd(X, Y ) | X]] is finite for all d,
V [Zℓ] ≤E � |Zℓ|2� ≤6|D| 2ℓ � d E [V[fd(X, Y ) | X]] .
�� �� Hence, by defining Fd(X) = E [fd(X, Y ) | X], we obtain
��� (2)
E � |fd −Fd|2� = E � E � |fd −Fd|2 | X �� = 1 2ℓE [V [fd(X, Y ) | X]]  Similarly
� � � Hence, V [Zℓ] is bounded by
# which completes the proof.
The theorem shows that the parameters for the MLMC theorem are β = 1, and in view of Remark 1, α ≥1/2. Since γ = 1 by the definition of Zℓ, the MLMC estimator is in the second regime, with β = γ, so that the total cost is O(ε−2(log ε)2). This compares favourably with the cost of O(ε−2−1/α) for the standard Monte Carlo estimator, where the exponent increases up to 4 in the worst case. In the proof of the theorem, the antithetic property of the estimator, i.e., 1 2(fd (a) + fd (b)) −fd = 0, is not exploited. In fact, the same upper bound on the variance can be obtained even for the alternative fd (a) −fd. In what follows, we prove a stronger result on the variance under somewhat demanding assumptions to exploit the antithetic structure of Zℓ. In fact, the MLMC variance can be analysed by following the approach used by Giles and Szpruch (2014, Theorem 5.2). Define
Fd(X) = E [fd(X, Y )|X] , dopt(X) = arg max d Fd(X)
so the domain for X is divided into a number of regions in which the optimal decision dopt(X) is unique, with a dividing decision manifold K on which dopt(X) is not uniquely-defined. Again note that 1 2(fd (a) + fd (b)) −fd = 0, and therefore Zℓ= 0 if the same decision d maximises each of the terms in its definition. When ℓis large and so there are many samples, fd (a), fd (b), fd will all be close to Fd(X), and therefore it is highly likely that Zℓ= 0 unless X is very close to K at which there is more than one optimal decision. This idea leads to an improved theorem on the MLMC variance, but we first need to make three assumptions.
Assumption 1. E [|fd(X, Y )|p] is finite for all p ≥2. Comment: this enables us to bound the difference between fd (a), fd (b), fd an Fd(X).
Assumption 2. There exists a constant c0 > 0 such that for all 0 < ǫ < 1
� min x∈K ∥X −x∥≤ǫ � ≤c0ǫ. Comment: this bounds the probability of X being close to the decision manifold K. Assumption 3. There exist constants c1, c2 > 0 such that if X /∈K, then max d Fd(X) − max d̸=dopt(X) Fd(X) > min � c1, c2 min x∈K ∥X −x∥ � . Comment: on K itself there are at least 2 decisions d1, d2 which yield the same optimal value Fd(X); this assumption ensures at least a linear divergence between the values as X moves away from K. Theorem 3. If Assumptions 1-3 are satisfied, and Zℓis as defined previously for level ℓ, then for any δ > 0 V [Zℓ] = o(2−(3/2−δ)ℓ), E [Zℓ] = o(2−(1−δ)ℓ). Comment: a similar O(N −3/2) convergence rate for the variance is proved in Theorem 2.3 in (Bujok et al. 2015) for a different nested simulation application. Before going into the detailed proof of the theorem, we give a heuristic explanation on the variance analysis below: • Due to Assumption 1 and Lemma 1 shown below, fd −Fd = O(2−ℓ/2); • Due to Assumption 2, there is an O(2−ℓ/2) probability of X being within distance O(2−ℓ/2) from the decision manifold K, in which case Zℓ= O(2−ℓ/2); • If X is further away from K, Assumption 3 ensures that there is a clear separation between different decision values, and hence the antithetic property of the estimator can be exploited well to give Zℓ= 0 with high probability; • This results in
  Comment: this bounds the probability of X being close to the decision manifold K.
  Comment: this bounds the probability of X being close to the decision manifold K.
Comment: on K itself there are at least 2 decisions d1, d2 which yield the same optimal value Fd(X); this assumption ensures at least a linear divergence between the values as X moves away from K.
Comment: on K itself there are at least 2 decisions d1, d2 which yield the same optimal value Fd(X); this assumption ensures at least a linear divergence between the values as X moves away from K. Theorem 3. If Assumptions 1-3 are satisfied, and Zℓis as defined previously for level ℓ, then for any δ > 0
Comment: a similar O(N −3/2) convergence rate for the variance is proved in Theorem 2.3 in (Bujok et al. 2015) for a different nested simulation application. Before going into the detailed proof of the theorem, we give a heuristic explanation on the variance analysis below:
E[Zℓ] = O(2−ℓ/2) × O(2−ℓ/2) = O(2−ℓ), E[Z2 ℓ] = O(2−ℓ/2) × (O(2−ℓ/2))2 = O(2−3ℓ/2),
so that we have α ≈1 and β ≈3/2.
To prepare for the proof of the main theorem, we first need a result concerning the deviation of an average of N values from the expected mean. Suppose X is a real random variable with zero mean, and let XN be an average of N i.i.d. samples Xn, n = 1, 2, . . ., N. For p = 2, we have E[XN 2] = N −1E[X2], and hence P[|XN| > c] ≤E[X2]/(c2N). For larger values of p for which E[|X|p] is finite, we have the following lemma: Lemma 1. For p ≥2, if E[|X|p] is finite then there exists a constant Cp, depending only on p, such that E[|XN|p] ≤CpN −p/2E [|X|p] , P[|XN| > c] ≤CpE[|X|p]/(c2N)p/2.
Lemma 1. For p ≥2, if E[|X|p] is finite then there exists a constant C depending only on p, such that
Proof. The discrete Burkholder-Davis-Gundy inequality (Burkholder et al. 1972) gives us
where Cp is a constant depending only on p. The second result follows immediately from the Markov inequality. Proof of Theorem 3. The analysis follows the approach used by Giles et al. (2009) and Giles and Szpruch (2014, Theorem 5.2). For a particular value of δ, we define ǫ = 2−(1/2−δ/2)ℓ, and consider the events
where c2 is as defined in Assumption 3. Using 1A to indicate the indicator function for event A, and Ac to denote the complement of A, we have
Looking at the first of the two terms on the r.h.s. of (3), then H¨older’s inequality gives
for any p, q ≥1, with p−1 + q−1 = 1. Now, P(A) ≤c0 ǫ due to Assumption 1, and
P(B) ≤ � d � P(|fd (a) −Fd| ≥1 2ǫ) + P(|fd (b) −Fd| ≥1 2ǫ) + P(|fd −Fd| ≥1 2ǫ) � .
(3)
Due to Lemma 1,
Similar bounds exists for P(|fd (a) −Fd| ≥ 1 2ǫ) and P(|fd (b) −Fd| ≥ 1 2ǫ). We can take m to be sufficiently large so that 1 2m −1−δ 2 m > 1−δ 2 and hence P(B) = o(2−(1−δ)ℓ/2). Then, q can be chosen sufficiently close to 1 so that (P(A) + P(B))1/q = o(2−(1/2−δ)ℓ). Applying Jensen’s inequality to (2) twice, we obtain
 � � � ≤(2|D|)2p−1 � � 1 2|fd (a) −Fd|2p + 1 2|fd (b) −Fd|2p + |fd −Fd|2p� .
  It follows from Lemma 1 that
E[ |fd −Fd|2p] = E � E[|fd −Fd|2p | X] �
� | −| | � ≤C2p2−pℓE � E[|fd(X, Y ) −E[fd(X, Y ) | X]|2p | X] � = C2p2−pℓE[|fd(X, Y ) −E[fd(X, Y )]|2p],
so Assumption 1 implies that E[|fd −Fd|2p] = O(2−pℓ), with similar bounds for fd (a) and fd (b). Hence,
and therefore the first term on the r.h.s. of (3) has bound o(2−(3/2−δ)ℓ). We now consider the second term on the r.h.s. of (3). For any sample in Ac ∩Bc, we have min x∈K ∥X −x∥≥ǫ, and |fd (a) −Fd| < 1 2c2ǫ, |fd (b) −Fd| < 1 2c2ǫ, |fd −Fd| < 1 2c2ǫ, for all d. For a particular outer sample X, if d ̸= dopt(X) then using Assumption 3 we have
If ℓis sufficiently large so that c2ǫ < c1, then fdopt −fd > 0 and hence dopt = arg maxd fd. The same argument applies to fd (a) opt −fd (a) and fd (b) opt −fd (b), so
the conclusion is that in all three cases, dopt is the decision which maximis fd (a), fd (b) and fd, and therefore
1 2(max d fd (a) + max d fd (b)) −max d fd = 1 2(fd (a) opt + fd (b) opt) −fdopt = 0.
Hence, for sufficiently large ℓ, the second term is zero, which concludes the proof for the bound on V[Zℓ] and the bound on E[Zℓ] is obtained similarly.
The conclusion from the theorem is that the parameters for the MLMC theorem are β ≈3/2, α≈1, and γ =1, giving the optimal complexity of O(ε−2). Again, this compares favourably with the cost of O(ε−3) for the standard Monte Carlo estimator.
# 4 Numerical results
# 4.1 Simple test cases
To validate the importance of the assumptions made in the variance analysis, several simple examples are tested here. Let X and Y be independent univariate standard normal random variables, and let us consider two-treatment decision problems with f1(X, Y ) = 0 and either 1. f2(X, Y ) = X + Y , 2. f2(X, Y ) = X3 + Y , or
3. f2(X, Y ) =      X + Y + 1, X < −1, Y, −1 ≤X ≤1, X + Y −1, X > 1.
  It is easy to check that this simple test case with the first choice of f2 satisfies a of Assumptions 1-3, while the other cases with the second and third choices  f2 do not. With the second choice of f2, we have F1(X) = 0 and F2(X) = X so that K = {0} ⊂R and
max d Fd(X) − max d̸=dopt(X) Fd(X) = |X|3,
which implies that there exist no constants c1, c2 > 0 such that Assumption 3 is satisfied. For the third choice of f2, we have K = [−1, 1] ⊂R whose probability measure is not zero. Hence, by considering the limiting situation ǫ →0 in Assumption 2, we see that there exists no constant c0 > 0 such that Assumption 2 is satisfied. The results for the first choice of f2 are shown in Figure 1. The left top plot shows the behaviours of the variances of both Pℓand Zℓ, where the variances are estimated by using N = 2 × 105 random samples at each level. Note that the logarithm of the empirical variance in base 2 versus the level is plotted here. The slope of the line for Zℓis −1.43, indicating that V[Zℓ] = O(2−1.43ℓ). This result is in good agreement with Theorem 3 which holds for decision models satisfying Assumptions 1-3.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d0f0/d0f08c4b-d581-416f-818f-1bbfa1d49f7e.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: MLMC results for simple test case with the first choice of f2.</div>
The middle top plot shows the behaviours of the estimated mean values of both Pℓand Zℓ. The slope of the line for Zℓis approximately −1, which implies that E[Zℓ] = O(2−ℓ). This is again in good agreement with Theorem 3. The right top plot shows the behaviour of the estimated kurtosis of Zℓ. The way in which the kurtosis increases with the level also confirms that the MLMC corrections are increasingly dominated by a few rare samples yielding Zℓ̸= 0, corresponding to outer samples X which are close to the decision manifold K across which the optimal decision dopt changes. Using the implementation due to Giles (2015, Algorithm 1), the maximum level L and the computational costs Nℓfor levels ℓ= 1, . . . , L, required for the combined multilevel estimator to achieve an MSE less than ε2, are estimated. Each line in the left bottom plot shows the values of Nℓ, ℓ= 1, . . . , L, for a particular value of ε. As expected, the number of samples varies with the level such that many more samples are allocated on the coarsest levels, which is in good agreement with the optimal allocation of computational effort given by Nℓ∝ε−2� Vℓ/Cℓ∝ε−22−(β+γ)ℓ/2 (Giles 2015). It is also shown here that, as the value of ε decreases, the maximum level L increases to ensure the weak convergence |E [P −PL] | ≤ε/ √ 2. The middle bottom plot shows the behaviour of the total computational cost
to achieve an MSE less than ε2. Since it is expected from the MLMC theorem that ε2C is independent of ε, we plot ε2C versus ε here. Indeed, it can be seen
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/80ee/80eec9b7-7212-4531-8e6b-e84542b3b125.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: MLMC results for simple test case with the second choice of f2.</div>
that ε2C is only slightly dependent on ε, indicating that the MLMC estimator gives the optimal complexity of O(ε−2). This result compares favourably with the result for the standard (in this case, nested) Monte Carlo method. The superiority of the MLMC method becomes more evident as the desired accuracy ε decreases. For instance, for ε = 10−4, the MLMC method is more than 50 times more efficient. Let us move on to the second and third choices of f2. Since these test cases do not satisfy one of Assumptions 1-3, Theorem 3 does not apply and it is expected from Theorem 2 that the MLMC estimator achieves the nearly optimal complexity of O(ε−2(log ε)2). The results for the second and third choices of f2 are shown in Figures 2 and 3, respectively. For the second choice of f2, it is seen from the first two top plots that the slopes of the lines for the variance and the mean value of Zℓare −1.12 and −0.64, respectively, which are slightly better than the values −1 and −0.5 which are to be expected from the theory. In the right top plot, the kurtosis increases with the level but not so significantly as compared to the first test case. Because of a smaller value of α, we can observe in the left bottom plot that the maximum level to ensure the weak convergence becomes large. Still, the superiority of the MLMC method over the standard Monte Carlo method is prominent. For ε = 10−4, the MLMC method is approximately 3000 times more efficient. Similar results are also obtained for the third choice of f2.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1f0e/1f0e6eef-34d2-4b89-b000-88b56531334a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: MLMC results for simple test case with the third choice of f2.</div>
# 4.2 Medical decision model
To demonstrate the practical usefulness of the MLMC estimator, the medical decision model introduced in Brennan et al. (2007) is tested. Let X ∪Y = (X1, . . . , X19) with each univariate random variable Xj following the normal distribution with mean µj and standard deviation σj independently except that X5, X7, X14, X16 are pairwise correlated with a correlation coefficient ρ = 0.6. The values for µj and σj and the medical meaning of Xj are listed in Table 1. The problem to be tested is a two-treatment decision problem with
f1(X, Y ) = λ (X5X6X7 + X8X9X10) −(X1 + X2X3X4), and f2(X, Y ) = λ (X14X15X16 + X17X18X19) −(X11 + X12X13X4),
where λ denotes the monetary valuation of health and is set to 104 (£). In what follows, we call this decision model the BKOC test case, named after the authors of Brennan et al. (2007). The results for the BKOC test case with X = (X5, X14) are shown in Figure 4. From the first two top plots we see that the slopes of the lines for the variance and the mean value of Zℓare −1.352 and −0.89, respectively, indicating that the MLMC estimator is in the first regime, with β > γ. The behaviour of the kurtosis of Zℓ, shown in the right top plot, is quite similar to that observed for the simple test case with the first choice of f2. As expected, most of the computational cost is actually incurred on the coarsest levels, and the MLMC method gives savings of factor more than 100 as compared to the standard Monte Carlo method for the desired accuracy ε = 0.1. As shown in Figure 5 and 6, respectively, both of the results for the BKOC
<div style="text-align: center;">Table 1: Variables in the BKOC test case (Table 2 in Brennan et al. (2007)).</div>
: Variables in the BKOC test case (Table 2 in Brennan et al.
variable
µj
σj
meaning
X1
1000
1
Cost of drug (£)
X2
0.1
0.02
Probability of admissions
X3
5.2
1.0
Days in hospital
X4
400
200
Cost per day (£)
X5
0.7
0.1
Probability of responding
X6
0.3
0.1
Utility change if response
X7
3.0
0.5
Duration of response (years)
X8
0.25
0.1
Probability of side effects
X9
-0.1
0.02
Change in utility if side effect
X10
0.5
0.2
Duration of side effect (years)
X11
1500
1
Cost of drug (£)
X12
0.08
0.02
Probability of admissions
X13
6.1
1.0
Days in hospital
X14
0.8
0.1
Probability of responding
X15
0.3
0.05
Utility change if response
X16
3.0
1.0
Duration of response (years)
X17
0.2
0.05
Probability of side effects
X18
-0.1
0.02
Change in utility if side effect
X19
0.5
0.2
Duration of side effect (years)
test case with X = (X5, X6, X14, X15) and X = (X7, X16) are quite similar to the case with X = (X5, X14), and the MLMC method gives savings of factor up to 100. In order to achieve an MSE less than 1, the MLMC method needs the total computational costs of C = 4.1×107, 3.0×107, 2.2×107 for the three respective cases, giving the estimates of the difference EVPI −EVPPI as 799, 206, and 509. The total computational costs for the standard Monte Carlo method are found to be approximately 10 times larger for all cases. The standard Monte Carlo method using 107 random samples of (X, Y ) yields the estimate of EVPI as 1047. Thus, the EVPPI values for the three cases are estimated as 248, 841 and 538.
# 5 Conclusions
In this paper we have developed a Multilevel Monte Carlo method for the estimation of the expected value of partial perfect information, EVPPI, which is one of the most demanding nested expectation applications. The essential difficulty in the theoretical analysis lies in how to deal with the maximum of an unconditional expectation. We provide a set of assumptions on a decision model to exploit the antithetic property of the estimator, and then numerical analysis proves that a root-mean-square accuracy of ε can be achieved at a computational cost which is O(ε−2), and this is also supported by numerical experiments. As we already announced in (Giles et al. 2017), the MLMC estimator introduced in this paper works quite well for real medical application which measures the cost-effectiveness of novel oral anticoagulants in atrial fibrillation. The details
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9655/965538d6-92f0-4d65-a478-32f032d21c25.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: MLMC results for the BKOC test case with X = (X5, X14).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e9d6/e9d6d32d-25ea-4800-aabd-d2afe8b10eed.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: MLMC results for the BKOC test case with X = (X5, X6, X14, X15).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1136/11364abc-f4e4-49ad-a450-d2bc307d43fa.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: MLMC results for the BKOC test case with X = (X7, X16).</div>
on this application shall be summarised in the near future. Future research will address the following topics: • an extension to handle input distributions which are defined empirically, such as through the use of MCMC methods to sample from a Bayesian posterior distribution; • the use of quasi-random numbers in place of pseudo-random numbers, which leads to the Multilevel Quasi-Monte Carlo method which is capable of additional substantial savings (Giles and Waterhouse 2009); • the use of an adaptive number of inner samples, following the ideas of
• an extension to handle input distributions which are defined empirically, such as through the use of MCMC methods to sample from a Bayesian posterior distribution; • the use of quasi-random numbers in place of pseudo-random numbers, which leads to the Multilevel Quasi-Monte Carlo method which is capable of additional substantial savings (Giles and Waterhouse 2009);
• the use of an adaptive number of inner samples, following the ideas of Broadie et al. (2011), since it is only the outer samples which are near the decision manifold K which require great accuracy for the inner conditional expectation.
# Acknowledgements
The authors would like to thank Dr. Howard Thom of the University of Bristol for useful discussions and comments. The research of T. Goda was supported by JSPS Grant-in-Aid for Young Scientists (No. 15K20964) and Arai Science and Technology Foundation.
Ades AE, Lu G, Claxton K (2004) Expected value of sample information calculations in medical decision modeling. Medical Decision Making 24:207–227. Anderson D, Higham D (2012) Multi-level Monte Carlo for continuous time Markov chains, with applications in biochemical kinetics. SIAM Multiscale Modelling and Simulation 10(1):146–179. Blanchet J, Glynn P (2015) Unbiased Monte Carlo for optimization and functions of expectations via multi-level randomization. Proceedings of the 2015 Winter Simulation Conference, 3656–3667, IEEE. Bratvold R, Bickel J, Lohne H (2009) Value of information in the oil and gas industry: past, present, and future. SPE Reservoir Evaluation and Engineering 12:630– 638. Brennan A, Kharroubi S, O’Hagan A, Chilcott J (2007) Calculating partial expected value of perfect information via Monte Carlo sampling algorithms. Medical Decision Making 27:448–470. Broadie M, Du Y, Moallemi C (2011) Efficient risk estimation via nested sequential simulation. Management Science 57(6):1172–1194. Bujok K, Hambly B, Reisinger C (2015) Multilevel simulation of functionals of Bernoulli random variables with application to basket credit derivatives. Methodology and Computing in Applied Probability 17(3):579–604. Burkholder D, Davis B, Gundy R (1972) Integral inequalities for convex functions of operators on martingales. Proc. Sixth Berkeley Symposium Math. Statist. Prob., Vol II, 223–240 (University of California Press, Berkeley). Cliffe K, Giles M, Scheichl R, Teckentrup A (2011) Multilevel Monte Carlo methods and applications to elliptic PDEs with random coefficients. Computing and Visualization in Science 14(1):3–15. Giles M (2008) Multilevel Monte Carlo path simulation. Operations Research 56(3):607–617. Giles M (2015) Multilevel Monte Carlo methods. Acta Numerica 24:259–328. Giles M, Goda T, Thom H, Fang W, Wang Z (2017) MLMC for estimation of expected value of partial perfect information. Presentation at International Conference on Monte Carlo Methods and Applications. Giles M, Higham D, Mao X (2009) Analysing multilevel Monte Carlo for options with non-globally Lipschitz payoff. Finance and Stochastics 13(3):403–413. Giles M, Haji-Ali AL (2018) Multilevel nested simulation for efficient risk estimation. arXiv pre-print 1802.05016. Giles M, Szpruch L (2014) Antithetic multilevel Monte Carlo estimation for multidimensional SDEs without L´evy area simulation. Annals of Applied Probability 24(4):1585–1620. Giles M, Waterhouse B (2009) Multilevel quasi-Monte Carlo path simulation. Advanced Financial Modelling, 165–181, Radon Series on Computational and Applied Mathematics (De Gruyter). Haji-Ali AL (2012) Pedestrian flow in the mean-field limit. MSc thesis, KAUST, URL http://stochastic_numerics.kaust.edu.sa/Documents/publications/ AbdulLateef%20Haji%20Ali%20_Thesis.pdf. Heinrich S (2001) Multilevel Monte Carlo methods. Multigrid Methods, volume 2179 of Lecture Notes in Computer Science, 58–67 (Springer). Nakayasu M, Goda T, Tanaka K, Sato K (2016) Evaluating the value of single-point data in heterogeneous reservoirs with the expectation maximization. SPE Economics and Management 8:1–10.
Rhee CH, Glynn P (2015) Unbiased estimation with square root convergence for SDE models. Operations Research 63(5):1026–1043.
