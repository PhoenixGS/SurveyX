# Multidimensional Sensitivity Analysis of Large-scale Mathematical Models
Ivan Dimov and Rayna Georgieva
19 Jan 2017
Abstract Sensitivity analysis (SA) is a procedure for studying how sensitive are the output results of large-scale mathematical models to some uncertainties of the input data. The models are described as a system of partial differential equations. Often such systems contain a large number of input parameters. Obviously, it is important to know how sensitive is the solution to some uncontrolled variations or uncertainties in the input parameters of the model. Algorithms based on analysis of variances technique (ANOVA) for calculating numerical indicators of sensitivity and computationally efficient Monte Carlo integration techniques have recently been developed by the authors. They have been successfully applied to sensitivity studies of air pollution levels calculated by the Unified Danish Eulerian Model (UNI-DEM) with respect to several important input parameters. In this paper a comprehensive theoretical and experimental study of the Monte Carlo algorithm based on symmetrised shaking of Sobol sequences has been done. It has been proven that this algorithm has an optimal rate of convergence for functions with continuous and bounded second derivatives in terms of probability and mean square error. Extensive numerical experiments with Monte Carlo, quasi-Monte Carlo (QMC) and scrambled quasiMonte Carlo algorithms based on Sobol sequences are performed to support the theoretical studies and to analyze applicability of the algorithms to various classes of problems. The numerical tests show that the Monte Carlo algorithm based on symmetrised shaking of Sobol sequences gives reliable results for multidimensional integration problems under consideration.
Ivan Dimov and Rayna Georgieva Department of Parallel Algorithms, IICT, Bulgarian Academy of Sciences, Acad. G. Bonchev 25 A, 1113 Sofia, Bulgaria, e-mail: ivdimov@bas.bg,rayna@parallel.bas.bg
Ivan Dimov and Rayna Georgieva Department of Parallel Algorithms, IICT, Bulgarian Academy of Sciences, Acad. G. Bonchev 25 A, 1113 Sofia, Bulgaria, e-mail: ivdimov@bas.bg,rayna@parallel.bas.bg
# 1 Introduction
Most existing methods for providing SA rely on special assumptions connected to the behavior of the model (such as linearity, monotonicity and additivity of the relationship between model input and model output) [22]. Such assumptions are often applicable to a large range of mathematical models. At the same time there are models that include significant nonlinearities and/or stiffness. For such models assumptions about linearity and additivity are not applicable. This is especially true when one deals with non-linear systems of partial differential equations. The numerical study and results reported in this paper have been done by using a large-scale mathematical model called Unified Danish Eulerian Model (UNI-DEM) [33, 34]. The model enables us to study the transport of air pollutants and other species over a large geographical region. The system of partial differential equations describes the main physical processes, such as advection, diffusion, deposition, as well as chemical and photochemical processes between the studied species. The emissions, and the quickly changing meteorological conditions are also described. The nonlinearity of the equations are mainly introduced when modeling chemical reactions [33]. If the model results are sensitive to a given process, one can describe it mathematically in a more adequate way, or more precisely. Thus, the goal of our study is to increase the reliability of the results produced by the model, and to identify processes that must be studied more carefully, as well as to find input parameters that need to be measured with a higher precision. A careful sensitivity analysis is needed in order to decide where and how simplifications of the model can be made. That’s why it is important to develop and study more adequate and reliable methods for sensitivity analysis. A good candidate for reliable sensitivity analysis of models containing nonlinearity is the variance based method [22]. The idea of this approach is to estimate how the variation of an input parameter or a group of inputs contributes into the variance of the model output. As a measure of this analysis we use the total sensitivity indices (TSI) (see, Section 2) described as multidimensional integrals:
where g(x) is a square integrable function in Ωand p(x) ≥0 is a probability density function, such that � Ωp(x)dx = 1. That’s why it is important to deal with efficient numerical methods for highdimensional integration. The progress in the area of sensitivity analysis is closely connected to the progress in reliable algorithms for multidimensional integration.
(1)
# 2 Problem Setting
# 2.1 Modeling and Sensitivity
Assume that the mathematical model can be presented as a function
u = f(x), where x = (x1,x2,...,xd) ∈Ud ≡[0;1]d
is the vector of input parameters with a joint probability density function (p.d.f.) p(x) = p(x1,...,xd). Assume also that the input variables are independent (noncorrelated) and the density function p(x) is known, even if xi are not actually random variables (r.v.). The total sensitivity index [10] provides a measure of the total effect of a given parameter, including all the possible coupling terms between that parameter and all the others. The total sensitivity index (TSI) of an input parameter xi,i∈{1,...,d} is defined in the following way [10, 26]:
where Si is called the main effect (first-order sensitivity index) of xi and Sil1...lj−1 is the j-th order sensitivity index. The higher-order terms describe the interaction effects between the unknown input parameters xi1,...,xiν ,ν ∈{2,...,d} on the output variance. The method of global SA used in this work is based on a decomposition of an integrable model function f in the d-dimensional factor space into terms of increasing dimensionality [26]:
where f0 is a constant. The representation (4) is referred to as the ANOVArepresentation of the model function f(x) if each term is chosen to satisfy the following condition [26]:
�1 0 fl1...lν(xl1,xl2,...,xlν)dxlk = 0, 1 ≤k ≤ν, ν = 1,...,d.
Let us mention the fact that if the whole presentation (4) of the right-hand site is used, then it doesn’t simplify the problem. The hope is that a truncated sequence f0 +∑dtr ν=1 ∑l1<...<lν fl1...lν (xl1,xl2,...,xlν ), where dtr < d (or even dtr << d), can be considered as a good approximation to the model function f. The quantities
D = � Ud f 2(x)dx −f 2 0, Dl1 ... lν = � f 2 l1 ... lνdxl1 ...dxlν
D = � Ud f 2(x)dx −f 2 0, Dl1 ... lν = � f 2 l1 ... lνdxl1 ...dxlν
(2)
(3)
(4)
(5)
are the so-called total and partial variances respectively and are obtained after squaring and integrating over Ud the equality (4) on the assumption that f(x) is a square integrable function (thus all terms in (4) are also square integrable functions). Therefore, the total variance of the model output is split into partial variances in the analogous way as the model function, that is the unique ANOVA-decomposition: D = ∑d ν=1 ∑l1<...<lν Dl1...lν . The use of probability theory concepts is based on the assumption that the input parameters are random variables distributed in Ud that defines fl1 ... lν(xl1,xl2,...,xlν ) also as random variables with variances (5). For example fl1 is presented by a conditional expectation: fl1(xl1) = E(u|xl1) −f0 and respectively Dl1 = D[fl1(xl1)] = D[E(u|xl1)]. Based on these assumptions about the model function and the output variance, the following quantities
are referred to as the global sensitivity indices [26]. Based on the formulas (5)-(6) it is clear that the mathematical treatment of the problem of providing global sensitivity analysis consists in evaluating total sensitivity indices (3) of corresponding order that, in turn, leads to computing multidimensional integrals of the form (1). It means that to obtain Stot i in general, one needs to compute 2d integrals of type (5). As we discussed earlier the basic assumption underlying representation (4) is that the basic features of the model functions (2) describing typical real-life problems can be presented by low-order subsets of input variables, containing terms of the order up to dtr, where dtr < d (or even dtr << d). Therefore, based on this assumption, one can assume that the dimension of the initial problem can be reduced. The procedure for computing global sensitivity indices (see [26]) is based on the following representation of the variance
where y = (xk1,...,xkm), 1 ≤k1 < ... < km ≤d, is an arbitrary set of m variables (1 ≤m ≤d −1) and z is the set of d −m complementary variables, i.e. x = (y,z). The equality (7) enables the construction of a Monte Carlo algorithm for evaluating f0,D and Dy:
where ξ = (η,ζ) is a random sample and η corresponds to the input subset denoted by y. Instead of randomized (Monte Carlo) algorithms for computing the above sensitivity parameters one can use deterministic quasi-Monte Carlo algorithms, or randomized quasi-Monte Carlo [13, 14]. Randomized (Monte Carlo) algorithms have
where ξ = (η,ζ) is a random sample and η corresponds to the input subset denoted by y.
(6)
(7)
proven to be very efficient in solving multidimensional integrals in composite domains [3, 23]. At the same time the QMC based on well-distributed Sobol sequences can be considered as a good alternative to Monte Carlo algorithms, especially for smooth integrands and not very high effective dimensions (up to d = 15) [12]. Sobol ΛΠτ are good candidates for efficient QMC algorithms. Algorithms based on ΛΠτ sequences while being deterministic, mimic the pseudo-random sequences used in Monte Carlo integration. One of the problems with ΛΠτ sequences is that they may have bad two-dimensional projection. In this context bad means that the distribution of the points is far away from the uniformity. If such projections are used in a certain computational problem, then the lack of uniformity may provoke a substantial lost of accuracy. To overcome this problem randomized QMC can be used. There are several ways of randomization and scrambling is one of them. The original motivation of scrambling [11, 19] aims toward obtaining more uniformity for quasirandom sequences in high dimensions, which can be checked via two-dimensional projections. Another way of randomisation is to shake the quasi-random points according to some procedure. Actually, the scrambled algorithms obtained by shaking the quasi-random points can be considered as Monte Carlo algorithms with a special choice of the density function. It’s a matter of definition. Thus, there is a reason to be able to compare two classes of algorithms: deterministic and randomized.
# 3 Complexity in Classes of Algorithms
One may pose the task to consider and compare two classes of algorithms: deterministic algorithms and randomized (Monte Carlo) algorithms. Let I be the desired value of the integral. Assume for a given r.v. θ one can prove that the mathematical expectation satisfies Eθ = I. Suppose that the mean value of n values of θ: θ (i), i = 1,...,n is considered as a Monte Carlo approximation to the solution: ¯θn = 1/n∑n i=1 θ (i) ≈I, where θ (i)(i = 1,2,...,n) correspond to values (realizations) of a r.v. θ. In general, a certain randomized algorithm can produce the result with a given probability error. So, dealing with randomized algorithms one has to accept that the result of the computation can be true only with a certain (although high) probability. In most practical computations it is reasonable to accept an error estimate with a probability smaller than 1. Consider the following integration problem:
where x ≡(x1,...,xd) ∈Ud ⊂Rd and f ∈C(Ud) is an integrable function on Ud. The computational problem can be considered as a mapping of function f : {[0,1]d →R} to R: S(f) : f →R, where S(f) = � Ud f(x)dx and f ∈F0 ⊂C(Ud). We refer to S as the solution operator. The elements of F0 are the data, for which the problem has to be solved; and for f ∈F0, S(f) is the exact solution. For a given f, we want to compute exactly or approximately S(f). One may be interested in cases
(8)
when the integrand f has a higher regularity. It is because in many cases of practical computations f is smooth and has high order bounded derivatives. If this is the case, then is it reasonable to try to exploit such a smoothness. To be able to do that we need to define the functional class F0 ≡Wk(∥f∥;Ud) in the following way: Definition 3.1 Let d and k be integers, d,k ≥1. We consider the class Wk(∥f∥;Ud) (sometimes abbreviated to Wk) of real functions f defined over the unit cube Ud = [0,1)d, possessing all the partial derivatives ∂r f(x) ∂xα1 1 ...∂xαd d , α1+...+αd = r ≤k, which are continuous when r < k and bounded in sup norm when r = k. The seminorm ∥·∥on Wk is defined as
when the integrand f has a higher regularity. It is because in many cases of practical computations f is smooth and has high order bounded derivatives. If this is the case, then is it reasonable to try to exploit such a smoothness. To be able to do that we need to define the functional class F0 ≡Wk(∥f∥;Ud) in the following way:
Definition 3.1 Let d and k be integers, d,k ≥1. We consider the class Wk(∥f∥;Ud) (sometimes abbreviated to Wk) of real functions f defined over the unit cube Ud = [0,1)d, possessing all the partial derivatives ∂r f(x) ∂xα1 1 ...∂xαd d , α1+...+αd = r ≤k, which are continuous when r < k and bounded in sup norm when r = k. The seminorm ∥·∥on Wk is defined as
∥f∥= sup ������ ∂k f(x) ∂xα1 1 ...∂xαd d �����, α1 + ...+ αd = k, x ≡(x1,...,xd) ∈Ud � .
��� ��� We call a quadrature formula any expression of the form
which approximates the value of the integral S(f). The real numbers ci ∈R are called weights and the d dimensional points x(i) ∈Ud are called nodes. It is clear that for fixed weights ci and nodes x(i) ≡(xi,1,...,xi,d) the quadrature formula AD(f,n) may be used to define an algorithm with an integration error err(f,AD) ≡ � Ud f(x)dx −AD(f,n). We call a randomized quadrature formula any formula of the following kind: AR(f,n) = ∑n i=1 σi f(ξ (i)), where σi and ξ (i) are random weights and nodes respectively. The algorithm AR(f,n) belongs to the class of randomized (Monte Carlo) denoted by A R.
Definition 3.2 Given a randomized (Monte Carlo) integration formula for the fun tions from the space Wk we define the integration error
# Definition 3.2 Given a randomized (Monte Carlo) integration formula for the functions from the space Wk we define the integration error
err(f,AR) ≡ � Ud f(x)dx −AR(f,n)
by the probability error εP(f) in the sense that εP(f) is the least possible real number, such that Pr ���err(f,AR) ��< εP(f) � ≥P,
# by the probability error εP(f) in the sense that εP(f) is the least possible real number, such that Pr ���err(f,AR) ��< εP(f) � ≥P,
and the mean square error
r(f) = � E � err2(f,AR) ��1/2 .
� � �� We assume that it suffices to obtain an εP(f)-approximation to the solution with a probability 0 < P < 1. If we allow equality, i.e., 0 < P ≤1 in Definition 3.2, then εP(f) can be used as an accuracy measure for both randomized and deterministic algorithms. In such a way it is consistent to consider a wider class A of algorithms that contains both classes: randomized and deterministic algorithms.
Definition 3.3 Consider the set A of algorithms A: A = {A : Pr(|err(f,A)| ≤ε) ≥c}, A ∈{AD,AR}, 0 < c < 1 that solve a given problem with an integration error err(f,A). In such a setting it is correct to compare randomized algorithms with algorithms based on low discrepancy sequences like Sobol ΛΠτ sequences.
# 4 The Algorithms
The algorithms we study are based on Sobol ΛΠτ sequences.
4.1 ΛΠτ Sobol Sequences
# 4.1 ΛΠτ Sobol Sequences
ΛΠτ sequences are uniformly distributed sequences (u.d.s.) The term u.d.s. was introduced by Hermann Weyl in 1916 [30]. For practical purposes an u.d.s. must be found that satisfied three requirements [23, 25]: (i) the best asymptote as n → ∞, (ii) well distributed points for small n, and (iii) a computationally inexpensive algorithm. All ΛΠτ-sequences given in [25] satisfy the first requirement. Suitable distributions such as ΛΠτ sequences are also called (t,m,s)-nets and (t,s)-sequences in base b ≥2. To introduce them, define first an elementary s-interval in base b as a subset of Us of the form E = ∏s j=1 �aj bdj , aj+1 bdj � , where a j,d j ≥0 are integers and a j < bdj for all j ∈{1,...,s}. Given two integers 0 ≤t ≤m, a (t,m,s)-net in base b is a sequence x(i) of bm points of Us such that Card E ∩{x(1),...,x(bm)} = bt for any elementary interval E in base b of hypervolume λ(E) = bt−m. Given a non-negative integer t, a (t,s)-sequence in base b is an infinite sequence of points x(i) such that for all integers k ≥0,m ≥t, the sequence {x(kbm),...,x((k+1)bm−1)} is a (t,m,s)-net in base b. I. M. Sobol [23] defines his Πτ-meshes and ΛΠτ sequences, which are (t,m,s)nets and (t,s)-sequences in base 2 respectively. The terms (t,m,s)-nets and (t,s)sequences in base b (also called Niederreiter sequences) were introduced in 1988 by H. Niederreiter [18]. To generate the j-th component of the points in a Sobol sequence, we need to choose a primitive polynomial of some degree sj over the Galois field of two elements GF(2) Pj = xsj + a1,jxsj−1 + a2,jxsj−2 + ... + asj−1,jx + 1, where the coefficients a1,j,...,asj−1,j are either 0 or 1. A sequence of positive integers {m1,j,m2,j,...} are defined by the recurrence relation
where ⊕is the bit-by-bit exclusive-or operator. The values m1,j,...,msj,j can be chosen freely provided that each mk,j,1 ≤k ≤sj, is odd and less than 2k. Therefore, it is possible to construct different Sobol sequences for the fixed dimension s. In practice, these numbers must be chosen very carefully to obtain really efficient Sobol sequence generators [27]. The so-called direction numbers {v1,j,v2,j,...} are defined by vk,j = mk,j 2k . Then the j-th component of the i-th point in a Sobol sequence, is given by xi,j = i1v1,j ⊕i2v2,j ⊕..., where ik is the k-th binary digit of i = (...i3i2i1)2. Subroutines to compute these points can be found in [2, 24]. The work [15] contains more details.
# 4.2 The Monte Carlo Algorithms based on Modified Sobol Sequences - MCA-MSS
One of the algorithms based on a procedure of shaking was proposed recently in [5]. The idea is that we take a Sobol ΛΠτ point (vector) x of dimension d. Then x is considered as a centrum of a sphere with a radius ρ. A random point ξ ∈Ud uniformly distributed on the sphere is taken. Consider a random variable θ defined as a value of the integrand at that random point, i.e., θ = f(ξ). Consider random points ξ (i)(ρ) ∈Ud,i = 1,...,n. Assume ξ (i)(ρ) = x(i) + ρω(i), where ω(i) is a unique uniformly distributed vector in Ud. The radius ρ is relatively small ρ << 1 2dj , such that ξ (i)(ρ) is still in the same elementary ith interval Ed i = ∏d j=1 � a(i) j 2dj , a(i) j +1 2dj � , where the pattern ΛΠτ point x(i) is. We use a subscript i in Ed i to indicate that the i-th ΛΠτ point x(i) is in it. So, we assume that if x(i) ∈Ed i , then ξ (i)(ρ) ∈Ed i too. It was proven in [5] that the mathematical expectation of the random variable θ = f(ξ) is equal to the value of the integral (8), that is Eθ = S(f) = � Ud f(x)dx. This result allows for defining a randomized algorithm. One can take the Sobol ΛΠτ point x(i) and shake it somewhat. Shaking means to define random points ξ (i)(ρ) = x(i)+ρω(i) according to the procedure described above. For simplicity the algorithm described above is called MCA-MSS-1. The probability error of the algorithm MCA-MSS-1 was analysed in [6]. It was proved that for integrands with continuous and bounded first derivatives, i.e. f ∈ F0 ≡W1(L;Ud), where L = ∥f∥, it holds
where the constants c ′ d and c ′′ d do not depend on n. In this work a modification of algorithm MCA-MSS-1 is proposed and analysed. The new algorithm will be called MCA-MSS-2. It is assumed that n = md, m ≥1. The unit cubeUd is divided into md disjoint subdomains, such that they coincide with the elementary d-dimensional subintervals
defined in Subsection 4.1 Ud = �md j=1Kj, where Kj = ∏d i=1[a(j) i ,b(j) i ), with b(j) i − a(j) i = 1 m for all i = 1,...,d. In such a way in each d-dimensional sub-domain Kj there is exactly one ΛΠτ point x(j). Assuming that after shaking, the random point stays inside Kj, i.e., ξ (j)(ρ) = x(j) + ρω(j) ∈Kj one may try to exploit the smoothness of the integrand in case if the integrand has second continuators and bounded derivatives, i.e., f ∈F0 ≡W2(L;Ud). Then, if p(x) is a probability density function, such that � Ud p(x)dx = 1, then
where c(j) 1 are constants. If d j is the diameter of Kj, then
where c(j) 2 are another constants. In the particular case when the subintervals are with edge 1/m for all constants we have: c(j) 1 = 1 and c(j) 2 = √ d. In each sub-domain Kj the central point is denoted by s(j), where s(j) = (s(j) 1 ,s(j) 2 ,...,s(j) d ). Suppose two random points ξ (j) and ξ (j)′ are chosen, such that ξ (j) is selected during our procedure used in MCA-MSS-1. The second point ξ (j)′ is chosen to be symmetric to ξ (j) according to the central point s(j) in each cube Kj. In such away the number of random points is 2md. One may calculate all function values f(ξ (j)) and f(ξ (j)′), for j = 1,...,md and approximate the value of the integral in the following way:
This estimate corresponds to MCA-MSS-2. Later on it will be proven that this algorithm has an optimal rate of convergence for functions with second bounded derivatives, i.e., for functions f ∈F0 ≡W2(L;Ud), while the algorithm MCA-MSS1 has an optimal rate of convergence for functions with first bounded derivatives: f ∈F0 ≡W1(L;Ud). One can prove the following
Theorem 1. The quadrature formula (9) constructed above for integrands f fro W2(L;Ud) satisfies
and
(9)
�   where the constants �c ′ d and �c ′′ d do not depend on n.
 � Proof. One can see that
For the fixed ΛΠτ point x(j) ∈Kj one can use the d-dimensional Taylor formula to present the function f(x(j)) in Kj around the central point s(j). For simplicity the superscript of the argument (j) will be omitted assuming that the formulas are written for the jth cube Kj:
Now, one can write this formula at previously defined random points ξ and ξ ′ both belonging to Kj. In such a way we have:
where Df(s) is the gradient of f evaluated at x = s and D2 f(s) is the Hessian matrix e.,
where Df(s) is the gradient of f evaluated at x = s and D2 f(s) is the Hessian matr i.e.,  2  2  2  
 Summarising (10) and (11) one can get
� � Because of the symmetry there is no member depending on the gradient Df(s) in the previous formula. If we consider the variance D[f(ξ) + f(ξ ′)] taking into account that the variance of the constant 2 f(s) is zero, then we will get
Ivan Dimov and Rayna Georgieva
(10)
(11)
� � � � Now we return back to the notation with superscript taking into account that the above consideration is just for an arbitrary sub-domain Kj. Since f ∈W2(L;Ud), ∥f∥≤Lj and L = ∥f∥is the majorant for all Lj, i.e., Lj ≤L for j = 1,...,n. Obviously, there is a point (d-dimensional vector) η ∈Kj, such that
� and the variance can be estimated from above in the following way:
D[f(ξ)+ f(ξ ′)] ≤L2 j sup x(j) 1 ,x(j) 2 ���x(j) 1 −x(j) 2 ��� 4 ≤L2 j(c(j) 2 )4n−4/d.
� � Now the variance of θn = ∑n j=1θ (j) can be estimated:
� � Therefore r(f,d) ≤�c ′′ d ∥f∥n −1 2 −2 d . The application of the Tchebychev’s inequality to the variance (12) yields
�   for the probable error ε, where �c ′ d = √ 2d, which concludes the proof.
 � One can see that the Monte Carlo algorithm MCA-MSS-2 has an optimal rate of convergence for functions with continuous and bounded second derivative [3]. This means that the rate of convergence (n−1 2 −2 d ) can not be improved for the functional class W2 in the class of the randomized algorithms A R. Note that both MCA-MSS-1 and MCA-MSS-2 have one control parameter, that is the radius ρ of the sphere of shaking. At the same time, to be able to efficiently use this control parameter one should increase the computational complexity. The problem is that after shaking the random point may leave the multidimensional subdomain. That’s why after each such a procedure one should be checking if the random point is still in the same sub-domain. It is clear that the procedure of checking if a random point is inside the given domain is a computationally expensive procedure when one has a large number of points. A small modification of MCA-MSS-2 algorithm allows to overcome this difficulty. If we just generate a random point ξ (j) ∈Kj uniformly distributed inside Kj and after that take the symmetric point ξ (j)′ according to the central point s(j), then this procedure will simulate the algo-
(12)
rithm MCA-MSS-2. Such a completely randomized approach simulates algorithm MCA-MSS-2, but the shaking is with different radiuses ρ in each sub-domain. We will call this algorithm MCA-MSS-2-S, because this approach looks like the stratified symmetrised Monte Carlo. Obviously, MCA-MSS-2-S is less expensive than MCA-MSS-2, but there is not such a control parameter like the radius ρ, which can be considered as a parameter randomly chosen in each sub-domain Kj. It is important to notice that all three algorithms MCA-MSS-1, MCA-MSS-2 and MCA-MSS-2-S have optimal (unimprovable) rate of convergence for the corresponding functional classes, that is MCA-MSS-1 is optimal in F0 ≡W1(L;Ud) and both MCA-MSS-2 and MCA-MSS-2-S are optimal in F0 ≡W2(L;Ud). We also will be considering the known Owen Nested Scrambling Algorithm [19] for which it is proved that the rate of convergence is n−3/2(log n)(d−1)/2, which is very good but still not optimal even for integrands in F0 ≡W1(L;Ud). One can see that if the logarithmic function from the estimate can be omitted, then the rate will became optimal. Let us mention that it is still not proven that the above estimate is exact, that is, we do not know if the logarithm can be omitted. It should be mentioned that the proved convergence rate for the Owen Nested Scrambling Algorithm improves significantly the rate for the unscrambled nets, which is n−1(log n)d−1. That’s why it is important to compare numerically our algorithms MCA-MSS with the Owen Nested Scrambling. The idea of Owen nested scrambling is based on randomization of a single digit at each iteration. Let x(i) = (xi,1,xi,2,...,xi,s), i = 1,...,n be quasi-random numbers in [0,1)s, and let z(i) = (zi,1,zi,2,...,zi,s) be the scrambled version of the point x(i). Suppose that each xi,j can be represented in base b as xi,j = (0.xi1,j xi2,j ...xiK,j ...)b with K being the number of digits to be scrambled. Then nested scrambling proposed by Owen [19, 20] can be defined as follows: zi1,j = π•(xi1,j), and zil,j = π•xi1,jxi2,j...xil−1,j(xil,j), with independent permutations π•xi1,jxi2,j...xil−1,j for l ≥2. Of course, (t,m,s)-net remains (t,m,s)-net under nested scrambling. However, nested scrambling requires bl−1 permutations to scramble the l-th digit. Owen scrambling (nested scrambling), which can be applied to all (t,s)-sequences, is powerful; however, from the implementation point-of-view, nested scrambling or so-called path dependent permutations requires a considerable amount of bookkeeping, and leads to more problematic implementation. There are various versions of scrambling methods based on digital permutation, and the differences among those methods are based on the definitions of the πl’s. These include Owen nested scrambling [19, 20], Tezuka’s generalized Faure sequences [29], and Matousek’s linear scrambling [17].
# 5 Case-study: Variance-based Sensitivity Analysis of the Unified Danish Eulerian Model
The input data for the sensitivity analysis performed in this paper has been obtained during runs of a large-scale mathematical model for remote transport of air pollu-
tants (Unified Danish Eulerian Model, UNI-DEM, [33]). The model enables us to study concentration variations in time of a high number of air pollutants and other species over a large geographical region (4800 × 4800 km), covering the whole of Europe, the Mediterranean and some parts of Asia and Africa. Such studies are important for environmental protection, agriculture, health care. The model presented as a system of partial differential equations describes the main processes in the atmosphere including photochemical processes between the studied species, the emissions, the quickly changing meteorological conditions. Both non-linearity and stiffness of the equations are mainly introduced when modeling chemical reactions [33]. The chemical scheme used in the model is the well-known condensed CBMIV (Carbon Bond Mechanism). Thus, the motivation to choose UNI-DEM is that it is one of the models of atmospheric chemistry, where the chemical processes are taken into account in a very accurate way. This large and complex task is not suitable for direct numerical treatment. For the purpose of numerical solution it is split into submodels, which represent the main physical and chemical processes. The sequential splitting [16] is used in the production version of the model, although other splitting methods have also been considered and implemented in some experimental versions [4, 7]. Spatial and time discretization makes each of the above submodels a huge computational task, challenging for the most powerful supercomputers available nowadays. That is why parallelization has always been a key point in the computer implementation of DEM since its very early stages. Our main aim here is to study the sensitivity of the ozone concentration according to the rate variation of some chemical reactions. We consider the chemical rates to be the input parameters and the concentrations of pollutants to be the output parameters.
# 6 Numerical Results and Discussion
Some numerical experiments are performed to study experimentally various properties of the algorithms. The expectations based on theoretical results are that for non-smooth functions MCA-MSS algorithms based on the shaking procedures outperform the QMC even for relatively low dimensions. It is also interesting to observe how behave the randomized QMC based on scrambled Sobol sequences. For our numerical tests we use the following non-smooth integrand:
for which even the first derivative does not exist. Such kind of applications appear also in some important problems in financial mathematics. The referent value of the integral S(f1) is approximately equal to 7.22261. To make a comparison we also consider an integral with a smooth integrand:
(13)
<div style="text-align: center;">Table 1 Relative error and computational time for numerical integration of a smooth function (S( f2) ≈0.10897).</div>
2 ≈
n
SFMT
Sobol QMCA
Owen scrambling
MCA-MSS-1
Rel.
Time
Rel.
Time
Rel.
Time
ρ
Rel.
Time
error
(s)
error
(s)
error
(s)
×103
error
(s)
102
0.0562
0.002
0.0365
< 0.001
0.0280
0.001
3.9
0.0363
0.001
13
0.0036
0.001
103
0.0244
0.004
0.0023
0.001
0.0016
0.001
1.9
0.0038
0.010
6.4
0.0019
0.010
104
0.0097
0.019
0.0009
0.002
0.0003
0.003
0.8
0.0007
0.070
2.8
0.0006
0.065
The second integrand (14) is an infinitely smooth function with a referent value of the integral S(f2) approximately equal to 0.10897. The integration domain in both cases is U4 = [0,1]4. Some results from the numerical integration tests with a smooth (14) and a nonsmooth (13) integrand are presented in Tables 1 and 2 respectively. As a measure of the efficiency of the algorithms both the relative error (defined as the absolute error divided by the referent value) and computational time are shown. For generating Sobol quasi-random sequences the algorithm with Gray code implementation [1] and sets of direction numbers proposed by Joe and Kuo [9] are used. The MCAMSS-1 algorithm [5] involves generating random points uniformly distributed on a sphere with radius ρ. One of the best available random number generators, SIMDoriented Fast Mersenne Twister (SFMT) [21, 32] 128-bit pseudo-random number generator of period 219937−1 has been used to generate the required random points. SFMT algorithm is a very efficient implementation of the Plain Monte Carlo method [23]. The radius ρ depends on the integration domain, number of samples and minimal distance between Sobol deterministic points δ. We observed experimentally that the behaviour of the relative error of numerical integration is significantly influenced by the fixed radius of spheres. That is why the values of the radius ρ are presented according to the number of samples n used in our experiments, as well as to a fixed coefficient, radius coefficient κ = ρ/δ. The latter parameter gives the ratio of the radius to the minimal distance between Sobol points. The code of scrambled quasi-random sequences used in our studies is taken from the collection of NAG C Library [31]. This implementation of scrambled quasi-random sequences is based on TOMS Algorithm 823 [11]. In the implementation of the scrambling there is a possibility to make a choice of three methods of scrambling: the first is a restricted form of Owen scrambling [19], the second based on the method of Faure and Tezuka [8], and the last method combines the first two (it is referred to as a combined approach). Random points for the MCA-MSS-1 algorithm have been generated using the original Sobol sequences and modeling a random direction in d-dimensional space. The computational time of the calculations with pseudo-random numbers generated
(14)
<div style="text-align: center;">Table 2 Relative error and computational time for numerical integration of a non-smooth function (S( f1) ≈7.22261).</div>
(1 ≈722261).
n
SFMT
Sobol QMCA
Owen scrambling
MCA-MSS-1
Rel.
Time
Rel.
Time
Rel.
Time
ρ
Rel.
Time
error
(s)
error
(s)
error
(s)
×103
error
(s)
103
0.0010
0.011
0.0027
0.001
0.0021
0.002
1.9
0.0024
0.020
6.4
0.0004
0.025
7.103 0.0009
0.072
0.0013
0.009
0.0003
0.011
1.0
0.0004
0.110
3.4
0.0005
0.114
3.104 0.0005
0.304
0.0003
0.032
0.0003
0.041
0.6
0.0001
0.440
1.9
0.0002
0.480
5.104 0.0007
0.513
0.0002
0.053
2e-05
0.066
0.4
7e-05
0.775
1.4
0.0001
0.788
by SFMT (see columns labeled as SFMT and MCA-MSS in Tables 1 and 2 has been estimated for all 10 algorithm runs. Comparing the results in Tables 1 and 2 one observes that • all algorithms under consideration are efficient and converge with the expected rate of convergence; • in the case of smooth functions, the Sobol algorithm is better than SFMT (the relative error is up to 10 times smaller than for SFMT); • the scrambled QMC and MCA-MSS-1 are much better than the classical Sobol algorithm; in many cases even the simplest shaking algorithm MCA-MSS-1 gives a higher accuracy than the scrambled algorithm. • in case of non-smooth functions SFMT algorithm implementing the plain Monte Carlo method is better than the Sobol algorithm for relatively small samples (n); • in the case of non-smooth functions our Monte Carlo shaking algorithm MCAMSS-1 gives similar results as the scrambled QMC; for several values of n we observe advantages for MCA-MSS-1 in terms of accuracy; • both MCA-MSS-1 and scrambled QMC are better than SFMT and Sobol quasi MC algorithm in the case of non-smooth functions. Another observation is that for the chosen integrands the scrambling algorithm does not outperform the algorithm with the original Sobol points, but the scrambled algorithm and Monte Carlo algorithm MCA-MSS-1 are more stable with respect to relative errors for relatively small values of n. The facts we observed that some further improvements of implementation of more refined shaking algorithms MCA-MSS-2 and MCA-MSS-2-S may be expected for relatively smooth integrands. That’s why we compare Sobol QMCA with MCA-MSS-2 and MCA-MSS-2-S, as well as with simplest shaking algorithm MCA-MSS-1 (see Table 3). The results show that the simplest shaking algorithm MCA-MSS-1 gives relative errors similar to errors of the Sobol QMCA, which is expected since the ΛΠτ Sobol sequences are already quite well distributed. That’s why one should not expect improvement for a very smooth integrand. But the symmetrised shaking algorithm MCA-MSS-2 improves the relative error. The effect
Another observation is that for the chosen integrands the scrambling algorithm does not outperform the algorithm with the original Sobol points, but the scrambled algorithm and Monte Carlo algorithm MCA-MSS-1 are more stable with respect to relative errors for relatively small values of n. The facts we observed that some further improvements of implementation of more refined shaking algorithms MCA-MSS-2 and MCA-MSS-2-S may be expected for relatively smooth integrands. That’s why we compare Sobol QMCA with MCA-MSS-2 and MCA-MSS-2-S, as well as with simplest shaking algorithm MCA-MSS-1 (see Table 3). The results show that the simplest shaking algorithm MCA-MSS-1 gives relative errors similar to errors of the Sobol QMCA, which is expected since the ΛΠτ Sobol sequences are already quite well distributed. That’s why one should not expect improvement for a very smooth integrand. But the symmetrised shaking algorithm MCA-MSS-2 improves the relative error. The effect
<div style="text-align: center;">Table 3 Relative error and computational time for numerical integration of a smooth functio (S( f ) ≈0.10897).</div>
 ≈
# of points n
Sobol QMCA
MCA-MSS-1
MCA-SMS-2
MCA-SMS-2-S
(# of double Rel.
Time
ρ
Rel.
Time
Rel.
Time
Rel.
Time
points 2n)
error
(s)
×103
error
(s)
error
(s)
error
(s)
29
0.0059
< 0.001 2.1
0.0064
0.009
0.0033
0.010
0.0016
0.005
(2×29)
6.4
0.0061
0.010
0.0032
0.010
210
0.0035
0.002
1.9
0.0037
0.010
9e-05
0.020
0.0002
0.007
(2×210)
6.4
0.0048
0.010
0.0002
0.020
216
2e-05
0.027
0.4
3e-05
1.580
7e-06
1.340
9e-06
0.494
(2×216)
1.2
0.0001
1.630
5e-06
1.380
<div style="text-align: center;">Table 4 Relative error and computational time for numerical integration of a smooth function (S( f ) ≈0.10897).</div>
 ≈
n
Sobol QMCA
MCA-MSS-1
MCA-MSS-2-S
Rel.
Time
ρ
Rel.
Time
Rel.
Time
×103
error
(s)
error
(s)
error
(s)
2×44
0.0076
< 0.001
2.1
0.0079
< 0.001
0.0016
0.005
(512)
6.4
0.0048
< 0.001
2×64
0.0028
0.001
1.2
0.0046
0.030
0.0004
0.009
(2592)
4.1
0.0046
0.030
2×84
0.0004
0.004
0.9
0.0008
0.090
0.0002
0.025
(8192)
2.9
0.0024
0.090
2×104
0.0002
0.008
0.6
0.0001
0.220
5e-05
0.070
(20000)
2.0
0.0013
0.230
2×134
0.0001
0.022
0.4
0.0001
0.630
4e-06
0.178
(57122)
1.2
0.0007
0.640
2×144
5e-06
0.029
0.4
1e-05
0.860
1e-05
0.237
(76832)
1.2
0.0005
0.880
2×154
8e-06
0.036
0.4
0.0001
1.220
9e-07
0.313
(101250)
1.2
0.0005
1.250
of this improvement is based on the fact that the second derivatives of the integrand exists, they are bounded and the construction of the MCA-MSS-2 algorithm gives a better convergence rate of order O(n−1/2−2/d). The same convergence rate has the algorithm MCA-MSS-2-S, but the latter one does not allow to control the value of the radius of shaking. As expected MCA-MSS-2-S gives better results than MCA-MSS-1. The relative error obtained by MCA-MSS-2 and MCA-MSS-2-S are of the same magnitude (see Table 3). The advantage of MCA-MSS-2-S is that its computational complexity is much smaller. A comparison of the relative error and computational complexity for different values of n is presented in Table 4. To have a fair comparison we have to consider again a smooth function (14). The observation is that MCA-MSS-2-S algorithm outperforms the simplest shaking algorithm MCA-MSS-1 in terms of relative error and complexity.
Table 5 Relative error (in absolute value) and computational time for estimation of sensitivity indices of input parameters using various Monte Carlo and quasi-Monte Carlo approaches (n = 6600,c ≈0.51365,δ ≈0.08).
<div style="text-align: center;">Table 5 Relative error (in absolute value) and computational time for estimation of sensitivity indices of input parameters using various Monte Carlo and quasi-Monte Carlo approaches (n = 6600,c ≈0.51365,δ ≈0.08).</div>
Table 5 Relative error (in absolute value) and computational time for estimation of sensitivity indices of input parameters using various Monte Carlo and quasi-Monte Carlo approaches (n = 6600051365δ008).
 ≈ ≈
Estimated
Sobol QMCA
Owen scrambling
MCA-MSS-1
quantity
ρ
Rel. error
g0
1e-05
0.0001
0.0007
0.0001
0.007
6e-05
D
0.0007
0.0013
0.0007
0.0003
0.007
0.0140
Stot
1
0.0036
0.0006
0.0007
0.0009
0.007
0.0013
Stot
2
0.0049
6e-05
0.0007
2e-05
0.007
0.0034
Stot
3
0.0259
0.0102
0.0007
0.0099
0.007
0.0211
After testing the algorithms under consideration on the smooth and non-smooth functions we studied the efficiency of the algorithms on real-life functions obtained after running UNI-DEM. Polynomials of 4-th degree with 35 unknown coefficients are used to approximate the mesh functions containing the model outputs. We use various values of the number of points that corresponds to situations when one needs to compute the sensitivity measures with different accuracy. We have computed results for g0 (g0 is the integral over the integrand g(x) = f(x)−c, f(x) is the approximate model function of UNI-DEM, and c is a constant obtained as a Monte Carlo estimate of f0, [28]), the total variance D, as well as total sensitivity indices Stot i ,i = 1,2,3. The above mentioned parameters are presented in Table 5. Table 5 presents the results obtained for a relatively low sample size n = 6600. One can notice that for most of the sensitivity parameters the simplest shaking algorithm MCA-MSS-1 outperforms the scrambled Sobol sequences, as well as the algorithm based on the ΛΠτ Sobol sequences in terms of accuracy. For higher values of sample sizes this effect is even stronger. One can clearly observe that the simplest shaking algorithm MCA-MSS-1 based on modified Sobol sequences improves the error estimates for non-smooth integrands. For smooth functions modified algorithms MCA-MSS-2 and MCA-MSS2-S give better results than MCA-MSS-1. Even for relatively large radiuses ρ the results are good in terms of accuracy. The reason is that centers of spheres are very well uniformly distributed by definition. So that, even for large values of radiuses of shaking the generated random points continue to be well distributed. We should stress on the fact that for relatively low number of points (< 1000) the algorithm based on modified Sobol sequences gives results with a high accuracy.
# 7 Conclusion
A comprehensive theoretical and experimental study of the Monte Carlo algorithm MCA-MSS-2 based on symmetrised shaking of Sobol sequences has been done. The algorithm combines properties of two of the best available approaches - Sobol quasi-Monte Carlo integration and a high quality SFMT pseudo-random number generator. It has been proven that this algorithm has an optimal rate of convergence for functions with continuous and bounded second derivatives in terms of probability and mean square error. A comparison with the scrambling approach, as well as with the Sobol quasiMonte Carlo algorithm and the algorithm using SFMT generator has been provided for numerical integration of smooth and non-smooth integrands. The algorithms mentioned above are tested numerically also for computing sensitivity measures for UNI-DEM model to study sensitivity of ozone concentration according to variation of chemical rates. All algorithms under consideration are efficient and converge with the expected rate of convergence. It is important to notice that the Monte Carlo algorithm MCA-MSS-2 based on modified Sobol sequences when symmetrised shaking is used has a unimprovable rate of convergence and gives reliable numerical results.
# Acknowledgment
The research reported in this paper is partly supported by the Bulgarian NSF Grants DTK 02/44/2009 and DMU 03/61/2011.
# References
1. I. Antonov, V. Saleev, An economic method of computing LPτ-sequences, USSR Comput. Math. Phy. 19 (1979) 252-256. 2. P. Bradley, B. Fox, Algorithm 659: Implementing Sobol’s quasi random sequence generator, ACM Trans. Math. Software 14(1) (1988) 88-100. 3. I.T. Dimov, Monte Carlo methods for applied scientists, World Scientific, London, Singapore, 2008. 4. I.T. Dimov, I. Farago, A. Havasi, Z. Zlatev, Operator splitting and commutativity analysis in the Danish Eulerian Model, Math. Comp. Sim. 67 (2004) 217-233. 5. I.T. Dimov, R. Georgieva, Monte Carlo method for numerical integration based on Sobol’ sequences, in: LNCS 6046, Springer, 2011, 50-59. 6. I. T. Dimov, R. Georgieva, Tz. Ostromsky, Z. Zlatev, Advanced algorithms for multidimensional sensitivity studies of large-scale air pollution models based on Sobol sequences, Comput Math Appl. Elsevier (in press). ISSN: 0898-1221. Doi: 10.1016/j.camwa.2012.07.005. 7. I.T. Dimov, Tz. Ostromsky, Z. Zlatev, Challenges in using splitting techniques for large-scale environmental modeling, in: Advances in Air Pollution Modeling for Environmental Security (Farago, I., Georgiev, K., Havasi, A. - eds.) NATO Science Series 54, 2005, Springer, 115132.
8. H. Faure, S. Tezuka, Another random scrambling of digital (t,s)-sequences Monte Carlo and Quasi-Monte Carlo methods, Springer-Verlag, Berlin, Germany (K. Fang, F. Hickernell, H. Niederreiter, eds.), 2000. 9. S. Joe, F. Kuo, Constructing Sobol’ sequences with better two-dimensional projections, SIAM J. Sci. Comput. 30 (2008), 2635-2654. 10. T. Homma, A. Saltelli, Importance measures in global sensitivity analysis of nonlinear models, Reliability Engineering and System Safety 52 (1996) 1-17. 11. H. Hong, F. Hickernell, Algorithm 823: Implementing scrambled digital sequences, ACM Trans. Math. Software 29(2) (2003) 95-109. 12. S. Kucherenko, B. Feil, N. Shah, W. Mauntz, The identification of model effective dimensions using global sensitivity analysis, Reliability Engineering and System Safety 96 (2011) 440449. 13. P. L’Ecuyer, C. Lecot, B. Tuffin, A randomized quasi-Monte Carlo simulation method for Markov chains. Operations Research 56(4) (2008) 958-975. 14. P. L’Ecuyer, C. Lemieux, Recent advances in randomized quasi-Monte Carlo methods, in: Dror, M., L’Ecuyer, P., Szidarovszki, F. (eds.), Modeling Uncertainty: An Examination of Stochastic Theory, Methods, and Applications, Kluwer Academic Publishers, Boston, 2002, 419-474. 15. Y. Levitan, N. Markovich, S. Rozin, I. Sobol, On quasi-random sequences for numerical computations, USSR Comput. Math. and Math. Phys. 28(5) (1988) 755-759. 16. G.I. Marchuk, Mathematical modeling for the problem of the environment, Studies in Mathematics and Applications, No. 16, North-Holland, Amsterdam, 1985. 17. J. Matousek, On the L2-discrepancy for anchored boxes, Journal of Complexity 14 (1998) 527-556. 18. H. Niederreiter, Low-discrepancy and low-dispersion sequences, Journal of Number Theory 30 (1988) 51-70. 19. A. Owen, Randomly permuted (t,m,s)-nets and (t,s)-sequences, Monte Carlo and QuasiMonte Carlo Methods in Scientific Computing, 106, Lecture Notes in Statistics, 299-317, 1995. 20. A. Owen, Variance and Discrepancy with Alternative Scramblings, ACM Trans. on Computational Logic., V (2002) 1-16. 21. M. Saito, M. Matsumoto, SIMD-oriented fast Mersenne Twister: a 128-bit pseudorandom number generator, in: Keller, A., Heinrich, S., Niederreiter, H. (eds.) Monte Carlo and QuasiMonte Carlo Methods 2006, Springer (2008) 607-622. 22. A. Saltelli, S. Tarantola, F. Campolongo, M. Ratto, Sensitivity analysis in practice: A guide to assessing scientific models, Halsted Press, New York, 2004. 23. I.M. Sobol, Monte Carlo numerical methods, Nauka, Moscow, 1973 (in Russian). 24. I.M. Sobol, On the systematic search in a hypercube, SIAM J. Numerical Analysis 16 (1979) 790-793. 25. I.M. Sobol, On quadratic formulas for functions of several variables satisfying a general Lipschitz condition, USSR Comput. Math. and Math. Phys. 29(6) (1989) 936-941. 26. I.M. Sobol, Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates, Mathematics and Computers in Simulation, 55(1-3) (2001) 271-280. 27. I. Sobol, D. Asotsky, A. Kreinin, S. Kucherenko, Construction and comparison of highdimensional Sobol’ generators, Wilmott Journal (2011) 67-79. 28. I. Sobol, E. Myshetskaya, Monte Carlo estimators for small sensitivity indices, Monte Carlo Methods and Applications 13(5-6) (2007) 455-465. 29. S. Tezuka, Uniform Random Numbers, Theory and Practice. Kluwer Academic Publishers, IBM Japan, 1995. 30. H. Weyl, Ueber die Gleichverteilung von Zahlen mod Eins. Math. Ann. 77(3) (1916) 313-352. 31. www.nag.co.uk/numeric/CL/CLdescription.asp. 32. www.math.sci.hiroshima-u.ac.jp/∼m-mat/MT/SFMT/index.html. 33. Z. Zlatev, I. T. Dimov, Computational and numerical challenges in environmental modelling, Elsevier, Amsterdam, 2006. 34. Z. Zlatev, I.T. Dimov, K. Georgiev, Three-dimensional version of the Danish Eulerian model, Zeitschrift f¨ur Angewandte Mathematik und Mechanik, 76(S4) (1996) 473-476.
