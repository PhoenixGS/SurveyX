# 99% of Distributed Optimization is a Waste of Time: The Issue and How to Fix it
# Konstantin Mishchenko
ntin Mis KAUST
Thuwal, Saudi Arabia
 4 Jun 2019
# Peter Richt´arik∗
er Richt KAUST
Thuwal, Saudi Arabia peter.richtarik@kaust.edu.sa
Many popular distributed optimization methods for training machine learning models fit the following template: a local gradient estimate is computed independently by each worker, then communicated to a master, which subsequently performs averaging. The average is broadcast back to the workers, which use it to perform a gradient-type step to update the local version of the model. It is also well known that many such methods, including SGD, SAGA, and accelerated SGD for over-parameterized models, do not scale well with the number of parallel workers. In this paper we observe that the above template is fundamentally inefficient in that too much data is unnecessarily communicated by the workers, which slows down the overall system. We propose a fix based on a new update-sparsification method we develop in this work, which we suggest be used on top of existing methods. Namely, we develop a new variant of parallel block coordinate descent based on independent sparsification of the local gradient estimates before communication. We demonstrate that with only m/n blocks sent by each of n workers, where m is the total number of parameter blocks, the theoretical iteration complexity of the underlying distributed methods is essentially unaffected. As an illustration, this means that when n = 100 parallel workers are used, the communication of 99% blocks is redundant, and hence a waste of time. Our theoretical claims are supported through extensive numerical experiments which demonstrate an almost perfect match with our theory on a number of synthetic and real datasets.
# 1 Introduction
In this work we are concerned with parallel/distributed algorithms for solving finite sum minimization problems of the form
� where each fi is convex and smooth. In particular, we are interested in methods which employ n parallel units/workers/nodes/processors, each of which has access to a single function fi and its gradients (or unbiased estimators thereof). Let x∗be an optimal solution of (1). In practical parallel ∗Also affiliated with the Moscow Institute of Physics and Technology, Dolgoprudny, Russia.
Preprint. Under review.
(1)
where the expectation is with respect to a distribution of training examples stored locally at machine i. More typically, however, each machine contains a very large but finite number of examples (for simplicity, say there are l examples on each machine), and fi is of the form
In the rest of this section we provide some basic motivation and intuitions in support of our approach. To this purpose, assume, for simplicity of exposition, that fi is of the finite-sum form (3). In typical modern machine learning workloads, the number of machines n is much smaller than the number of data points on each machine l. In a large scale regime (i.e., when the model size d, the number of data points nl, or both are large), problem (1) needs to be solved by a combination of efficient methods and modern hardware. In recent years there has been a lot of progress in designing new algorithms for solving this problem using techniques such as stochastic approximation [34], variance reduction [35, 17, 9], coordinate descent [25, 32, 41] and acceleration [26], resulting in excellent theoretical and practical performance. The computational power of the hardware is increasing as well. In recent years, a very significant amount of such increase is due to parallelism. Since many methods, such as minibatch Stochastic Gradient Descent (SGD), are embarrassingly parallel, it is very simple to use them in big data applications. However, it has been observed in practice that adding more resources beyond a certain limit does not improve iteration complexity significantly. Moreover, having more parallel units makes their synchronization harder due to so-called communication bottleneck. Minibatch versions of most variance reduced methods2 such as SAGA [9] or SVRG [17] scale even worse in parallel setting – they do not guarantee, in the worst case, any speedup from using more than one function at a time. Unfortunately, numerical experiments show that this is not a proof flaw, but rather a real property of these methods [12]. A similar observation was made for SVRG in [42], where it was shown that only a small number of partial derivatives are needed at each iteration. Since there are too many possible situations, we choose to focus on black-box optimization, although we admit that much can be achieved by assuming the sparsity structure. In fact, for any method there exists a toy situation where the method would scale perfectly – one simply needs to assume that each function fi depends on its own subset of coordinates and minimize each fi independently. This can be generalized assuming sparsity patterns [18, 19] to get almost linear scaling if any coordinate appears in a small number of functions. Our interest, however, is in explaining situations as in [12] where the models almost do not scale. In this paper, we demonstrate that a simple trick of independent block sampling can remedy the problem of scaling, to a substantial but limited extent. To illustrate one of the key insights of our paper on a simple example, in what follows consider a thought experiment in which GD is a baseline method we would want to improve on.
# 1.1 From gradient descent to block coordinate descent and back
A simple benchmark in the distributed setting is a parallel implementation of gradient descent (GD). GD arises as a special case of the more general class of block coordinate descent methods (BCD) [33]. The conventional way to run BCD for problem (1) is to update a single or several blocks3 of x, chosen at random, on all n machines [33, 10], followed by an update aggregation step. Such updates on each worker typically involve a gradient step on a subspace corresponding to the selected blocks. Importantly, and this is a key structural property of BCD methods, the same set of blocks is updated on each machine. If communication is expensive, it often makes sense to do more work on each machine, which in the context of BCD means updating more blocks. A particular special case is to update all blocks, which leads to parallel implementation of GD for problem (1), as mentioned above.
2We shall mention that there are already a few variance reduced methods that scale, up to some level, linearly in a parallel setup: Quartz for sparse data [30], Katyusha [2], or SAGA/SVRG/SARAH with importance sampling for non-convex problems [16]. 3Assume the entries of x are partitioned into several non-overlapping blocks.
(2)
(3)
Moreover, it is known that the theoretical iteration complexity of BCD improves as the number of blocks updated increases [33, 28, 29]. For these and similar reasons, GD (or one of its variants, such as GD with momentum), is often the preferable method to BCD. Having said that, we did not choose to describe BCD only to discard it at this point; we shall soon return to it, albeit with a twist.
# 1.2 From gradient descent to independent block coordinate desce
Because of what we just said, iteration complexity of GD will not improve by any variant running BCD; it can only get worse. Despite this, we propose to run BCD, but a new variant which allows each worker to sample an independent subset of blocks instead. This variant of BCD for (1) was not considered before. As we shall show, our independent sampling approach leads to a better-behaved aggregated gradient estimator when compared to that of BCD, which in turn leads to better overall iteration complexity. We call our method independent block coordinate descent (IBCD). We provide a unified analysis of our method, allowing for a random subset of τm out of a total of m blocks to be sampled on each machine, independently from other machines. GD arises as a special case of this method by setting τ = 1. However, as we show (see Corollary 1), the same iteration complexity guarantee can be obtained by choosing τ as low as τ = 1/n. The immediate consequence of this result is that it is suboptimal to run GD in terms of communication complexity. Indeed, GD needs to communicate all m blocks per machine, while IBCD achieves the same rate with m/n blocks per machine only. Coming back to the abstract, consider an example with n = 100 machines. In this case, when compared to GD, IBCD only communicates 1% of the data. Because the iteration complexities of the two methods are the same, and if communication cost is dominant, this means that the problem can be solved in just 1% of the time. In contrast, and when compared to the potential of IBCD, parallel implementation of GD inevitably wastes 99% of the time. The intuition behind why our approach works lies in the law of large numbers. By averaging independent noise we reduce the total variance of the resulting estimator by the factor of n. If, however, the noise is already tiny, as, in non-accelerated variance reduced methods, there is no improvement. On the other hand, (uniform) block coordinate descent (CD) has variance proportional to 1/τ [39], where τ < 1 is the ratio of used blocks. Therefore, after the averaging step the variance is 1/τn, which illustrates why setting any τ > 1/n should not yield a significant speedup when compared to the choice τ = 1/n. It also indicates that it should be possible to throw away a (1 −1/n) fraction of blocks while keeping the same convergence rate.
# 1.3 Beyond gradient descent and further contributions
The goal of the above discussion was to introduce one of the ideas of this paper in a gentle way. However, our independent sampling idea has immense consequences beyond the realm of GD, as we show in the rest of the paper. Let us summarize the contributions here:
However, our independent sampling idea has immense consequences beyond the realm of GD, as we show in the rest of the paper. Let us summarize the contributions here: • We show that the independent sampling idea can be coupled with variance reduction/SAGA (see Sec 4), SGD for problem (1)+(2) (see Sec F), acceleration (under mild assumption on stochastic gradients; see Sec G) and regularization/SEGA (see Sec 5). We call the new methods ISAGA, ISGD, IASGD and ISEGA, respectively. We also develop ISGD variant for asynchronous distributed optimization – IASGD (Sec H). • We present two versions of the SAGA algorithm coupled with IBCD. The first one is for a distributed setting, where each machine owns a subset of data and runs a SAGA iteration with block sampling locally, followed by aggregation. The second version is in a shared data setting, where each machine has access to all functions. This allows for linear convergence even if ∇fi(x∗) ̸= 0. • We show that when combined with IBCD, the SEGA trick [14] leads to a method that enjoys a linear rate for problems where ∇fi(x∗) ̸= 0 and allows for more general objectives which may include a non-separable non-smooth regularizer.
# 2 Practical Implications and Limitations
section, we outline some further limitations and practical implications of 
e outline some further limitations and practical implications of our frame
#
Name
Origin
∇fi(x∗) ̸= 0
Lin. rate
In-machine
randomization
Note
1
IBCD
I+ CD [25]



Simplest
2
ISEGA
I + SEGA [14]



Allows prox
3
IBGD
I+ GD



Bernoulli, no CD
4
ISAGA
I+ SAGA [9]



Shared memory
5
ISAGA
I+ SAGA [9]



6
ISGD
I + SGD [34]



+ Non-convex
7
IASGD
I + ASGD [37]



8
IASGD
I + ASGD [31]



Asynchronous
Table 1: Summary of all algorithms proposed in the paper.
# 2.1 Main limitation
The main limitation of this work is that independent sampling does not generally result in a sparse aggregated update. Indeed, since each machine might sample a different subset of blocks, all these updates add up to a dense one, and this problem gets worse as n increases, other things equal. For instance, if every parallel unit updates a single unique block4, the total number of updated blocks is equal n. In contrast, standard BCD, one that samples the same block on each worker, would update a single block only. For simple linear problems, such as logistic regression, sparse updates allow for a fast implementation of BCD via memorization of the residuals. However, this limitation is not crucial in common settings where broadcast is much faster than reduce.
# 2.2 Practical implications
The main body of this work focuses on theoretical analysis and on verifying our claims via experiments. However, there are several straightforward and important applications of our technique. Distributed synchronous learning. A common way to run a distributed optimization method is to perform a local update, communicate the result to a parameter server using a ’reduce’ operation, and inform all workers using ’broadcast’. Typically, if the number of workers is significantly large, the bottleneck of such a system is communication. In particular, the ’reduce’ operation takes much more time than ’broadcast’ as it requires to add up different vectors computed locally, while ’broadcast’ informs the workers about the same data (see [22] for a numerical validation that ’broadcast’ is 10-20 times faster across a wide range of dimensions). Nevertheless, if every worker can instead send to the parameter server only τ = 1/n fraction of the d-dimensional update, essentially the server node will receive just one full d-dimensional vector, and thus our approach can compete against methods like QSGD [1], signSGD [3], TernGrad [40], DGC [20] or ATOMO [38]. In fact, our approach may completely remove the communication bottleneck. Distributed asynchronous learning. The main difference with the synchronous case is that only one-to-one communications will be used instead of highly efficient ’reduce’ and ’broadcast’. Clearly, the communication to the server will be much faster with τ = 1/n, so the main question is how to make the communication back fast as well. Hopefully, the parameter server can copy the current vector and send it using non-blocking communication, such as isend() in MPI4PY [7]. Then, the communication back will not prevent the server from receiving the new updates. We combine the IBCD approach with asynchronous updates, which leads to a new method: IASGD (Algorithm 8). Distributed sparse learning. Large datasets, such as binary classification data from LibSVM, often have sparse gradients. In this case, the ’reduce’ operation is not efficient and one needs to communicate data by sending positions of nonzeros and their values. Moreover, as we prove later, one can use independent sampling with ℓ1-penalty, which makes the problem solution sparse. In that case, only communication from a worker to the parameter server is slow, so both synchronous and asynchronous methods gain in performance. Methods with local subproblems. One can also try to extend our analysis to methods with exact block-coordinate minimization or primal-dual and proximal methods such as Point-SAGA [8],
4Assume x is partitioned into several “blocks” of variables.
PDHG [4], DANE [36], etc. There, by restricting ourselves to a subset of coordinates, we may obtain a subproblem that is easier to solve by orders of magnitude. Block-separable problems within machines. Given that the local problem on each machine is block coordinate-wise separable, partial derivative blocks can be evaluated 1/τ times cheaper than the gradients. Thus, independent sampling improves scalability at no cost. Such problems can be obtained considering the dual problem, as is done in [21], for example. For a comprehensive list of frequently used notation, see Table 2 in the supplementary material.
# 3 Independent Block Coordinate Descent
# 3.1 Technical assumptions
We present the most common technical assumptions required in order to derive convergence rates. Definition 1. Function F is L smooth if for all x, y ∈Rd we have:
  Similarly, F is µ strongly convex if for all x, y ∈Rd: F(x) ≥F(y) + ⟨∇F(y), x −y⟩+ µ 2 ∥x −y∥2 2.
In most results we present, functions fi are required to be smooth and convex, and f strongly convex. Assumption 1. For every i, function fi is convex, L smooth and function f is µ strongly convex. As mentioned, since independent sampling does not preserve the variance reduction property, in some of our results we shall consider ∇fi(x∗) = 0 for all i. Assumption 2. For all 1 ≤i ≤n we have ∇fi(x∗) = 0.
In Sec 4 we show that Assumption 2 can be dropped once the memory is shared among the machines. Further, in Sec 5 we show that Assumption 2 can be dropped even in the fully distributed setup using the SEGA trick. Lastly, Assumption 2 is naturally satisfied in many applications. For example, in least squares setting min ∥Ax −b∥2 2, it is equivalent to existence of x∗such that Ax∗= b. On the other hand, current state-of-the-art deep learning models are often overparameterized so that they allow zero training loss, which is again equivalent to ∇fi(x∗) = 0 for all i (however, such problems are typically non-convex).
# 3.2 Block structure of Rd
Let Rd be partitioned into m blocks u1, . . . , um of arbitrary sizes, so that the parameter space is R|u1| × · · · R|um|. For any vector x ∈Rd and a set of blocks U we denote by xU the vector that has the same coordinate as x in the set of blocks U and zeros elsewhere.
In order to provide a quick taste of our results, we first present the IBCD method described in th introduction and formalized as Algorithm 1.
A key parameter of the method is 1/m ≤τ ≤1 (chosen so that τm is an integer), representing a fraction of blocks to be sampled by each worker. At iteration t, each machine independently samples a subset of τm blocks U t i ⊆{u1, . . . , um}, uniformly at random. The ith worker then performs a subspace gradient step of the form xt+1 i = xt −γ(∇fi(xt))U t i , where γ > 0 is a stepsize. Note that only coordinates of xt belonging to U t i get updated. This is then followed by aggregating all n gradient updates: xt+1 = 1 n � i xt+1 i .
# � 3.4 Convergence of IBCD
Theorem 1 provides a convergence rate for Algorithm 1. Admittedly, the assumptions of Theorem 1 are somewhat restrictive; in particular, we require ∇fi(x∗) = 0 for all i. However, this is necessary.
(4)
(5)
Algorithm 1
1: Input: x0 ∈Rd, partition of Rd into m blocks u1, . . . , um, ratio of blocks to be sampled τ,
stepsize γ, # of parallel units n
2: for t = 0, 1, . . . do
3:
for i = 1, . . . , n in parallel do
4:
Sample independently and uniformly a subset of τm blocks U t
i ⊆{u1, . . . , um}
5:
xt+1
i
= xt −γ(∇fi(xt))U t
i
6:
end for
7:
xt+1 = 1
n
�n
i=1 xt+1
i
8: end for
Indeed, in general one can not expect to have �n i=1(∇fi(x∗))Ui = 0 (which would be required for the method to converge to x∗) for independently sampled sets of blocks Ui unless ∇fi(x∗) = 0 for all i. As mentioned, the issue is resolved in Sec 5 using the SEGA trick [14]. Theorem 1. Suppose that Assumptions 1, 2 hold. For Algorithm 1 with γ = n τn+2(1−τ) 1 2L we have
E � ∥xt −x∗∥2 2 � ≤ � 1 − µ 2L τn τn+2(1−τ) �t ∥x0 −x∗∥2 2.
� � � � As a consequence of Theorem 1, we can choose τ as small as 1/n and get, up to a constant factor, the same convergence rate as gradient descent, as described next. Corollary 1. If τ = 1/n, the iteration complexity5 of Algorithm 1 is O(L/µ log 1/ϵ).
# 3.5 Optimal block sizes
If we naively use coordinates as blocks, i.e. all blocks have size equal 1, the update will be very sparse and the efficient way to send it is by providing positions of nonzeros and the corresponding values. If, however, we partition Rd into blocks of size approximately equal d/n, then on average only one block will be updated by each worker. This means that it will be just enough for each worker to communicate the block number and its entries, which is twice less data sent than when using coordinates as blocks.
# 4 Variance Reduction
As the first extension of IBCD, we inject independent coordinate sampling into SAGA6 [9], resulting in a new method we call ISAGA. We consider two different settings for ISAGA. The first one is standard distributed setup (1), where each fi is of the fine-sum form (3). The idea is to run SAGA with independent coordinate sampling locally on each worker, followed by aggregating the updates. However, as for IBCD, we require ∇fi(x∗) = 0 for all i. The second setting is a shared data/memory setup; i.e., we assume that all workers have access to all functions from the finite sum. This allows us to drop Assumption 2. Due to space limitations, we present distributed ISAGA in Sec E of the supplementary.
# 4.1 Shared data ISAGA
We now present a different setup for ISAGA in which the requirement ∇fi(x∗) = 0 is not needed. Instead of (1), we rather solve the problem
� with n workers all of which have access to all data describing f. Therefore, all workers can evaluate ∇ψj(x) for any 1 ≤j ≤N. Similarly to plain SAGA, we remember the freshest gradient information in vectors αj, and update them as
5Number of iterations to reach ϵ accurate solution. 6Independent coordinate sampling is not limited to SAGA and can be similarly applied to other variance reduction techniques.
(6)
(7)
where jt i is the index sampled at iteration t by machine i, and j′ refers to all indices that were not sampled at iteration t by any machine. The iterate updates within each machine are taken only on a sampled set of coordinates, i.e., xt+1 i = xt −γ(∇ψjt i (xt) −αt jt i + αt)U t i . where αt stands for the average of all α, and thus it is a delayed estimate of ∇f(xt). Lastly, we set the next iterate as the average of proposed iterates by each machine xt+1 = 1 n �n i=1 xt+1 i . The formal statement of the algorithm is given in the supplementary as Algorithm 4. Theorem 2. Suppose that function f is µ strongly convex and each ψi is L smooth and convex. If γ ≤ 1 L( 3  +τ), then for iterates of Algorithm 4 we have
� � � As in Sec E, the choice τ = 1/n yields a convergence rate which is, up to a constant factor, the same as the convergence rate of SAGA. Therefore, Algorithm 4 enjoys the desired parallel linear scaling, without the additional requirement of Assumption 2. Corollary 2 formalizes the claim. Corollary 2. Consider the setting from Theorem 2. Set τ = 1/n and γ = n/5L. Then c = 3/n2, ρ = min {µ/5L, 1/3N} and the complexity of Algorithm 4 is O (max{L/µ, N} log 1/ε).
# 5 Beyond Assumption 2 and Regularization
For this section only, let us consider a regularized objective of the form minx∈Rd f(x) ≜1 n �n i=1 fi(x) + R(x),
For this section only, let us consider a regularized objective of the form �
� where R is a closed convex regularizer such that its proximal operator is computable: proxγR(x) ≜ argminy � R(y) + 1 2γ ∥y −x∥2 2 � . In this section we propose ISEGA: an independent sampling variant of SEGA [14]. We do this in order to both i) avoid Assumption 2 (while keeping linear convergence) and ii) allow for R. Original SEGA learns gradients ∇f(xt) from sketched gradient information via the so called sketch-and-project process [11], constructing a vector sequence ht. In ISEGA on each machine i we iteratively construct a sequence of vectors ht i which play the role of estimates of ∇fi(xt). This is done via the following rule:
∇ − The key idea is again that these vectors are created from random blocks independently sampled on each machine. Next, using ht, SEGA builds an unbiased gradient estimator gt i of ∇fi(xt) as follows: gt i = ht i + 1 τ (∇fi(xt) −ht i)U t i . (10)
∇ − Then, we average the vectors gt i and take a proximal step.
Unlike coordinate descent, SEGA (or ISEGA) is not limited to separable proximal operators since, as follows from our analysis , ht i →∇fi(x∗). Therefore, ISEGA can be seen as a variance reduced version of IBCD for problems with non-separable regularizers. The price to be paid for dropping Assumption 2 and having more general objective (8) is that updates from each worker are dense, in contrast to those in Algorithm 1. In order to be consistent with the rest of the paper, we only develop a simple variant of ISEGA (Algorithm 2) in which we consider block coordinate sketches with uniform probabilities and non-weighted Euclidean metric (i.e. B = I in notation of [14]). It is possible to develop the theory in full generality as in [14]. However, we do not do this for the sake of simplicity. We next present the convergence rate of ISEGA (Algorithm 2). Theorem 3. Suppose Assumption 1 holds. Algorithm 2 with γ = min{ 1 4L(1+ 1 nτ ), 1 µ τ + 4L nτ } satisfies E[∥xt −x∗∥2 2] ≤(1 −γµ)tΦ0, where Φ0 = ∥x0 −x∗∥2 2 + γ 2Lτn �n i=1 ∥h0 −∇f(x∗)∥2 2. Note that if the condition number of the problem is not too small so that n = O (L/µ) (which is usually the case in practice), ISEGA scales linearly in the parallel setting. In particular, when doubling the number of workers, each worker can afford to evaluate only half of the block partial derivatives while keeping the same convergence speed. Moreover, setting τ = 1/n, the rate corresponds, up to a constant factor, to the rate of gradient descent. Corollary 3 states the result.
�   Note that if the condition number of the problem is not too small so that n = O (L/µ) (which is usually the case in practice), ISEGA scales linearly in the parallel setting. In particular, when doubling the number of workers, each worker can afford to evaluate only half of the block partial derivatives while keeping the same convergence speed. Moreover, setting τ = 1/n, the rate corresponds, up to a constant factor, to the rate of gradient descent. Corollary 3 states the result.
(8)
(9)
(10)
Algorithm 2 ISEGA
1: Input: x0 ∈Rd, initial gradient estimates h0
1, . . . , h0
n ∈Rd, partition of Rd into m blocks
u1, . . . , im, ratio of blocks to be sampled τ, stepsize γ, # parallel units n
2: for t = 0, 1, . . . do
3:
for i = 1, . . . , n in parallel do
4:
Sample independently and uniformly a subset of τm blocks U t
i
5:
gt
i = ht
i + 1
τ (∇fi(xt) −ht
i)U t
i
6:
ht+1
i
= ht
i + τ(gt
i −ht)
7:
end for
8:
xt+1 = proxγR
�
xt −γ 1
n
�n
i=1 gt
i
�
9: end for
<div style="text-align: center;">Figure 1: Comparison of SAGA and Algorithm 4 for various values of n (number of workers) and τ = n−1 on LibSVM datasets. Stepsize γ = 1 L(3n−1+τ) is chosen in each case.</div>
Corollary 3. Consider the setting from Theorem 3. Suppose that L/µ ≥n and choose τ = 1/n. Then, complexity of Algorithm 2 is O(L/µ log 1/ϵ). Remark 1. Parallel implementation Algorithm 2 would be to always send (∇fi(xk))U t i to the server; which keeps updating vector ht and takes the prox step.
# 6 Experiments
In this section, we numerically verify our theoretical claims. Recall that there are various settings where it is possible to make practical experiments (see Sec 2), however, we do not restrain ourselves to any of them in order to deliver as clear a message as possible. Due to space limitations, we only present a small fraction of the experiments here. A full and exhaustive comparison, together with the complete experiment setup description, is presented in Sec I of the supplementary material. In the first experiment presented here, we compare SAGA against ISAGA in a shared data setup (Algorithm 4) for various values of n with τ = 1/n in order to demonstrate linear scaling. We consider logistic regression problem on LibSVM data [5]. The results (Figure 1) corroborate our theory: indeed, setting nτ = 1 does not lead to a decrease in the convergence rate when compared to the original SAGA. The next experiment (Figure 2) supports an analogous claim for ISEGA (Algorithm 2). We run the method for several (n, τ) pairs for which nτ = 1; on logistic regression problems and LibSVM data. We also plot convergence of gradient descent with the analogous stepsize. As our theory predicts, all the methods exhibit almost the same convergence rate.7 Note that for n = 100, Algorithm 2 throws away 99% of partial derivatives while keeping the same convergence speed as GD, which justifies the title of the paper.
have chosen the stepsize γ = 1/2L for GD, as this is the baseline to Algorithm  in fact set γ = 1/L for GD and get 2× faster convergence. However, this is only
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9d0a/9d0ad891-6681-43fd-8329-4997cc473715.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Comparison of Algorithm 2 for various (n, τ) such that nτ = 1 and GD on LibSVM datasets. Stepsize 1/(L � 1 + 1 nτ � ) was chosen for Algorithm 2 and 1 2L for GD.</div>
# References
[1] Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. QSGD: Communication-efficient SGD via gradient quantization and encoding. In Advances in Neural Information Processing Systems, pages 1709–1720, 2017. [2] Zeyuan Allen-Zhu. Katyusha: The first direct acceleration of stochastic gradient methods. In Proceedings of the 49th Annual ACM SIGACT Symposium on Theory of Computing, pages 1200–1205. ACM, 2017. [3] Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Anima Anandkumar. SignSGD: Compressed optimisation for non-convex problems. arXiv preprint arXiv:1802.04434, 2018. [4] Antonin Chambolle and Thomas Pock. A first-order primal-dual algorithm for convex problems with applications to imaging. Journal of Mathematical Imaging and Vision, 40(1):120–145, 2011. [5] Chih-Chung Chang and Chih-Jen Lin. LibSVM: A library for support vector machines. ACM Transactions on Intelligent Systems and Technology (TIST), 2(3):27, 2011. [6] Dominik Csiba and Peter Richt´arik. Importance sampling for minibatches. The Journal of Machine Learning Research, 19(1):962–982, 2018. [7] Lisandro D Dalcin, Rodrigo R Paz, Pablo A Kler, and Alejandro Cosimo. Parallel distributed computing using Python. Advances in Water Resources, 34(9):1124–1139, 2011. [8] Aaron Defazio. A simple practical accelerated method for finite sums. In Advances in Neural Information Processing Systems, pages 676–684, 2016. [9] Aaron Defazio, Francis Bach, and Simon Lacoste-Julien. SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives. In Advances in Neural Information Processing Systems, pages 1646–1654, 2014. 10] Olivier Fercoq and Peter Richt´arik. Accelerated, parallel, and proximal coordinate descent. SIAM Journal on Optimization, 25(4):1997–2023, 2015. 11] Robert M Gower and Peter Richt´arik. Randomized iterative methods for linear systems. SIAM Journal on Matrix Analysis and Applications, 36(4):1660–1690, 2015. 12] Robert M Gower, Peter Richt´arik, and Francis Bach. Stochastic quasi-gradient methods: Variance reduction via Jacobian sketching. arXiv preprint arXiv:1805.02632, 2018. 13] Dmitry Grishchenko, Franck Iutzeler, J´erˆome Malick, and Massih-Reza Amini. Asynchronous distributed learning with sparse communications and identification. arXiv preprint arXiv:1812.03871, 2018. 14] Filip Hanzely, Konstantin Mishchenko, and Peter Richt´arik. SEGA: Variance reduction via gradient sketching. In Advances in Neural Information Processing Systems, pages 2083–2094, 2018. 15] Filip Hanzely and Peter Richt´arik. Accelerated coordinate descent with arbitrary sampling and best rates for minibatches. arXiv preprint arXiv:1809.09354, 2018. 16] Samuel Horv´ath and Peter Richt´arik. Nonconvex variance reduced optimization with arbitrary sampling. arXiv preprint arXiv:1809.04146, 2018. 17] Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In Advances in Neural Information Processing Systems, pages 315–323, 2013. 18] R´emi Leblond, Fabian Pedregosa, and Simon Lacoste-Julien. Asaga: Asynchronous parallel saga. In Artificial Intelligence and Statistics, pages 46–54, 2017. 19] R´emi Leblond, Fabian Pedregosa, and Simon Lacoste-Julien. Improved asynchronous parallel optimization analysis for stochastic incremental methods. The Journal of Machine Learning Research, 19(1):3140–3207, 2018. 20] Yujun Lin, Song Han, Huizi Mao, Yu Wang, and William J Dally. Deep gradient compression: Reducing the communication bandwidth for distributed training. arXiv preprint arXiv:1712.01887, 2017.
[21] Chenxin Ma, Virginia Smith, Martin Jaggi, Michael I. Jordan, Peter Richt´arik, and Martin Tak´aˇc. Adding vs. averaging in distributed primal-dual optimization. In The 32nd International Conference on Machine Learning, pages 1973–1982, 2015. [22] Konstantin Mishchenko, Eduard Gorbunov, Martin Tak´aˇc, and Peter Richt´arik. Distributed learning with compressed gradient differences. arXiv preprint arXiv:1901.09269, 2019. [23] Konstantin Mishchenko, Franck Iutzeler, and J´erˆome Malick. A distributed flexible delay-tolerant proximal gradient algorithm. arXiv preprint arXiv:1806.09429, 2018. [24] Konstantin Mishchenko, Franck Iutzeler, J´erˆome Malick, and Massih-Reza Amini. A delay-tolerant proximal-gradient algorithm for distributed learning. In International Conference on Machine Learning, pages 3584–3592, 2018. [25] Yu Nesterov. Efficiency of coordinate descent methods on huge-scale optimization problems. SIAM Journal on Optimization, 22(2):341–362, 2012. [26] Yurii Nesterov. A method for solving the convex programming problem with convergence rate O(1/kˆ2). In Dokl. Akad. Nauk SSSR, volume 269, pages 543–547, 1983. [27] Yurii Nesterov. Introductory lectures on convex optimization: A basic course. Kluwer Academic Publishers, 2004. [28] Zheng Qu and Peter Richt´arik. Coordinate descent with arbitrary sampling I: Algorithms and complexity. Optimization Methods and Software, 31(5):829–857, 2016. [29] Zheng Qu and Peter Richt´arik. Coordinate descent with arbitrary sampling II: Expected separable overapproximation. Optimization Methods and Software, 31(5):858–884, 2016. [30] Zheng Qu, Peter Richt´arik, and Tong Zhang. Quartz: Randomized dual coordinate ascent with arbitrary sampling. In Advances in Neural Information Processing Systems 28, pages 865–873, 2015. [31] Benjamin Recht, Christopher Re, Stephen Wright, and Feng Niu. Hogwild: A lock-free approach to parallelizing stochastic gradient descent. In J. Shawe-Taylor, R. S. Zemel, P. L. Bartlett, F. Pereira, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 24, pages 693–701. Curran Associates, Inc., 2011. [32] Peter Richt´arik and Martin Tak´aˇc. Iteration complexity of randomized block-coordinate descent methods for minimizing a composite function. Mathematical Programming, 144(1-2):1–38, 2014. [33] Peter Richt´arik and Martin Tak´aˇc. Parallel coordinate descent methods for big data optimization. Mathematical Programming, 156(1-2):433–484, 2016. [34] Herbert Robbins and Sutton Monro. A stochastic approximation method. In Herbert Robbins Selected Papers, pages 102–109. Springer, 1985. [35] Mark Schmidt, Nicolas Le Roux, and Francis Bach. Minimizing finite sums with the stochastic average gradient. Mathematical Programming, 162(1-2):83–112, 2017. [36] Ohad Shamir, Nati Srebro, and Tong Zhang. Communication-efficient distributed optimization using an approximate Newton-type method. In International Conference on Machine Learning, pages 1000–1008, 2014. [37] Sharan Vaswani, Francis Bach, and Mark Schmidt. Fast and faster convergence of SGD for over-parameterized models and an accelerated perceptron. arXiv preprint arXiv:1810.07288, 2018. [38] Hongyi Wang, Scott Sievert, Shengchao Liu, Zachary Charles, Dimitris Papailiopoulos, and Stephen Wright. ATOMO: Communication-efficient learning via atomic sparsification. In Advances in Neural Information Processing Systems, pages 9872–9883, 2018. [39] Jianqiao Wangni, Jialei Wang, Ji Liu, and Tong Zhang. Gradient sparsification for communication-efficient distributed optimization. In Advances in Neural Information Processing Systems, pages 1306–1316, 2018. [40] Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Terngrad: Ternary gradients to reduce communication in distributed deep learning. In Advances in Neural Information Processing Systems, pages 1509–1519, 2017. [41] Stephen J Wright. Coordinate descent algorithms. Mathematical Programming, 151(1):3–34, 2015.
[42] Tuo Zhao, Mo Yu, Yiming Wang, Raman Arora, and Han Liu. Accelerated mini-batch randomized block coordinate descent method. In Advances in neural information processing systems, pages 3329–3337, 2014.
42] Tuo Zhao, Mo Yu, Yiming Wang, Raman Arora, and Han Liu. Accelerated mini-batch randomized block coordinate descent method. In Advances in neural information processing systems, pages 3329–3337, 2014.
# A Table of Frequently Used Notation
General
x∗
Optimal solution of the optimization problem
n
Number of parallel workers/machines
Sec. 3.2
τ
Ratio of coordinate blocks to be sampled by each machine
Sec. 3.2
d
Dimensionality of space x ∈Rd
Sec. 3.2
m
Number of coordinate blocks
Sec. 3.2
fi
Part of the objective owned by machine i
(1)
L
Each fi is L smooth
As. 1 and (4)
µ
f is µ strongly convex
As. 1 and (5)
U t
i
Subset of blocks sampled at iteration t and worker i
γ
Stepsize
g
Unbiased gradient estimator
SAGA
αj
Delayed estimate of j-th objective
(11), (7)
N
Finite sum size for shared data problem
(6)
l
Number of datapoints per machine in distributed setup
(3)
Lt
Lyapunov function
(24)
SGD
gt
i
Unbiased stochastic gradient; Egt
i = ∇fi(xt)
σ2
An upper bound on the variance of stochastic gradients
As. 3
SEGA
R
Regularizer
(8)
ht
i
Sequence of biased estimators for ∇fi(xt)
(9)
gt
i
Sequence of unbiased estimators for ∇fi(xt)
(10)
Φt
Lyapunov function
Thm. 3
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2ef6/2ef61c95-cb44-480f-a8f6-64b513649a71.png" style="width: 50%;"></div>
Table 2: Summary of frequently used notation.
<div style="text-align: center;">Table 2: Summary of frequently used notation.</div>
# B Future Work
We sketch several possible extensions of this work.
• Combining the tricks from the paper. Distributed ISAGA requires ∇fi(x) = 0. We believe it would be possible to develop SEGA approach on top of it (such as it is developed on top of coordinate descent) and drop the mentioned requirement. We also believe it should be possible to accelerate the combination of SEGA and ISAGA. • Convergence in the asynchronous setup. We have provided theoretical results for parallel algorithms in the synchronous setting and the asynchronous theory is a very natural future step. Moreover, as we mentioned before, it has very direct practical implications. We believe that it possible to extend the works [24, 23] to design a method with proximable regularizer (for ℓ1 penalty) that would communicate little both sides. • Importance sampling. Standard coordinate descent exploits a smoothness structure of objective (either via coordinate-wise smoothness constants or more generally using a smoothness matrix) in order to sample coordinates non-uniformly [29, 6, 15]. It would be interesting to derive an importance sampling in our setting in order to converge even faster.
# C IBGD: Bernoulli alternative to IBCD
As an alternative to computing a random block of partial derivatives of size τm, it is possible to compute the whole gradient with probability τ, and attain the same complexity result. While this can be inserted in all algorithms we propose, we only present an alternative to IBCD, which we call IBGD.
Algorithm 3 Independent Bernoulli Gradient Descent (IBGD)
1: Input: x0 ∈Rd, probability of computing the whole gradient τ, stepsize γ, # of parallel units n
2: for t = 0, 1, . . . do
3:
for i = 1, . . . , n in parallel do
4:
Set gt
i =
�∇fi(xt)
with probability
τ
0
with probability
1 −τ
independently
5:
xt+1
i
= xt −γgt
i
6:
end for
7:
xt+1 = 1
n
�n
i=1 xt+1
i
8: end for
# Theorem 4. Suppose that Assumptions 1, 2 hold. For Algorithm 3 with γ = n τn+2(1−τ) 1 2L we have
E � ∥xt −x∗∥2 2 � ≤ � 1 −µ 2L τn τn + 2(1 −τ) �t ∥x0 −x∗∥2 2.
� � Note that IBGD does not perform sparse updates to the server; it is either full (dense), or none. This resembles the most naive asynchronous setup – where each iteration, a random subset of machines communicates with the server8. Our findings thus show that we can expect perfect linear scaling for such unreal asynchronous setup. In the honest asynchronous setup, we shall still expect good parallel scaling once the sequence of machines that communicate with server somewhat resembles a fixed uniform distribution.
# D Shared data ISAGA – algorithm
Algorithm 4 ISAGA with shared data
1: Input: x0 ∈Rd, α0
1, . . . , α0
N partition of Rd into m blocks u1, . . . , um, ratio of blocks to be
sampled τ, stepsize γ, # parallel units n
2: Set α0 ≜1
N
�n
i=1 α0
i
3: for t = 0, 1, . . . do
4:
Sample uniformly set of indices {jt
1, . . . , jt
n} ⊆{1, . . . , N} without replacement
5:
for i = 1, . . . , n in parallel do
6:
Sample independently and uniformly a subset of τm blocks U i
t
7:
xt+1
i
= xt −γ(∇ψjt
i (xt) −αt
jt
i + αt)U t
i
8:
(αt+1
jt
i )U t
i = αt
jt
i + (∇ψjt
i (xt) −αt
jt
i )U t
i
9:
end for
10:
For j ̸∈{jt
1, . . . , jt
n} set (αt+1
j
) = αt
i
11:
xt+1 = 1
n
�n
i=1 xt+1
i
12:
αt+1 = 1
n
�N
j=1 αt+1
j
13: end for
# E Distributed ISAGA
In this section we consider problem (1) with fi of the finite-sum structure (3). Just like SAGA, every machine remembers the freshest gradient information of all local functions (stored in arrays αij), and
In this section we consider problem (1) with fi of the finite-sum structure (3). Just like SAGA, every machine remembers the freshest gradient information of all local functions (stored in arrays αij), and
8In reality, there subset is not drawn from a fixed distribution
updates them once a new gradient information is observed. Given that index jt i is sampled on i-th machine at iteration t, the iterate update step within each machine is taken only on a sampled set of coordinates:
Above, αt i stands for the average of α variables on i-th machine, i.e. it is a delayed estimate  ∇fi(xt). Since the new gradient information is a set of partial derivatives of ∇fijt i (xt), we shall update
Lastly, the local results are aggregated. See Algorithm 5 for details.
Algorithm 5 Distributed ISAGA
1: Input: x0 ∈Rd, # parallel units n, i-th unit owns l functions fI1, . . . , fil, partition of Rd into
m blocks u1, . . . , um, ratio of blocks to be sampled τ, stepsize γ, initial vectors α0
ij ∈Rd for
1 ≤i ≤n, 1 ≤j ≤l
2: Set α0 ≜1
N
�n
i=1 α0
i
3: for t = 0, 1, . . . do
4:
for i = 1, . . . , n in parallel do
5:
Sample independently & uniformly jt
i ∈[l]
6:
Sample independently & uniformly a subset of τm blocks U i
t
7:
xt+1
i
= xt −γ(∇fijt
i (xt) −αt
ijt
i + αt
i)U t
i
8:
αt+1
ijt = αt
ijt
i + (∇fijt
i (xt) −αt
ijt
i )U t
i
9:
For any j ̸= jt
i set αt+1
ij
= αt
ij
10:
αt+1 = 1
l
�l
j=1 αt+1
ij
11:
end for
12:
xt+1 = 1
n
�n
i=1 xt+1
i
13: end for
The next result provides a convergence rate of distributed ISAGA. Theorem 5. Suppose that Assumptions 1, 2 hold. If γ ≤ 1 L( 3 n +τ), for iterates of distributed ISAGA we have
  � � where Ψ0 ≜�n i=1 �l j=1 ∥αt ij −∇fij(x∗)∥2 2, ϑ ≜τ min � γµ, 1 l − 2 n2lc � ≥0 and c ≜1 n( 1 γL − 1 n −τ) > 0.
The choice τ = n−1 yields a convergence rate which is, up to a constant factor, the same as convergence rate of original SAGA. Thus, distributed ISAGA enjoys the desired parallel linear scaling. Corollary 4 formalizes this claim. Corollary 4. Consider the setting from Theorem 5. Set τ = 1 n and γ = n 5L. Then c = 3 n2 , ρ = min �µ 5L, 1 3nl � and the complexity of distributed ISAGA is
# F SGD
In this section, we apply independent sampling in a setup with a stochastic objective. In particular, we consider problem (1) where fi is given as an expectation; see (2). We assume we have access to a stochastic gradient oracle which, when queried at xt, outputs a random vector gt i whose mean is ∇fi(xt): Egt i = ∇fi(xt).
(11)
Our proposed algorithm—ISGD—evaluates a subset of stochastic partial derivatives for the local objective and takes a step in the given direction for each machine. Next, the results are averaged and followed by the next iteration. We stress that the coordinate blocks have to be sampled independently within each machine.
Algorithm 6 ISGD
1: Input: x0 ∈Rd, partition of Rd into m blocks u1, . . . , um, ratio of blocks to be sampled τ,
stepsize sequence {γt}∞
t=1, # parallel units n
2: for t = 0, 1, . . . do
3:
for i = 1, . . . , n in parallel do
4:
Sample independently and uniformly a subset of τm blocks U t
i ⊆{u1, . . . , um}
5:
Sample blocks of stochastic gradient (gt
i)U t
i such that E[gt
i | xt] = ∇fi(xt)
6:
xt+1
i
= xt −γt(gt
i)U t
i
7:
end for
8:
xt+1 = 1
n
�n
i=1 xt+1
i
9: end for
<div style="text-align: center;">Algorithm 6 ISGD</div>
In order to establish a convergence rate of ISGD, we shall assume boundedness of stochastic gradients for each worker. Assumption 3. Consider a sequence of iterates {xt}∞ t=0 of Algorithm 6. Assume that gt i is an unbiased estimator of ∇fi(xt) satisfying E∥gt i −∇fi(xt)∥2 2 ≤σ2. Assumption 4. Stochastic gradients of function fi have bounded variance at the optimum of f: E∥gi −∇fi(x∗)∥2 2 ≤σ2, where gi is a random vector such that Egi = ∇fi(x∗). Next, we present the convergence rate of Algorithm 6. Since SGD is not a variance reduced algorithm, it does not enjoy a linear convergence rate and one shall use decreasing step sizes. As a consequence, Assumption 2 is not required anymore since there is no variance reduction property to be broken. Theorem 6. Let Assumptions 1 and 3 hold. If γt = 1 a+ct, where a = 2 � τ + 2(1−τ) n � L, c = 1 4µτ, then for Algorithm 6 we can upper bound E[f(ˆxt) −f(x∗)] by a2 � 1 −τµ a � ∥x0 −x∗∥2 2 τ(t + 1)a + cτ 2 t(t + 1) + σ2 + (1 −τ) 2 n �n i=1 ∥∇fi(x∗)∥2 2 n � 1 + 1 t � a + nc 2 (t + 1) , �
� Note that the residuals decrease as O(t−1), which is a behavior one expects from standard SGD Moreover, the leading complexity term scales linearly: if the number of workers n is doubled, one can afford to halve τ to keep the same complexity. Corollary 5. Consider the setting from Theorem 6. Then, iteration complexity of Algorithm 6 is � � �
Although problem (1) explicitly assumes convex fi, we also consider a non-convex extension, where smoothness of each individual fi is not required either. Theorem 7 provides the result. Theorem 7 (Non-convex rate). Assume f is L smooth, Assumption 3 holds and for all x ∈Rd the difference between gradients of f and fi’s is bounded: 1 n �n i=1 ∥∇f(x) −∇fi(x)∥2 2 ≤ν2 for some constant ν ≥0. If ˆxt is sampled uniformly from {x0, . . . , xt}, then for Algorithm 6 we have
Again, the convergence rate from Theorem 7 scales almost linearly with τ: with doubling the number of workers one can afford to halve τ to keep essentially the same guarantees. Note that if n is sufficiently large, increasing τ beyond a certain threshold does not improve convergence. This is a slightly weaker conclusion to the rest of our results where increasing τ beyond n−1 might still offer speedup. The main reason behind this is the fact that SGD may be noisy enough on its own to still benefit from the averaging step.
# Corollary 6. Consider the setting from Theorem 7. i) Choose τ ≥ 1 n and γ = √n L √ τt ≤ 1 2L(τ/2+(1−τ)/n). Then
E∥∇f(ˆxt)∥2 2 ≤ 2 √ tτn �f(x0) −f ∗ L + (1 −τ)ν2 � = O �1 √ t � .
ii) For any τ there is sufficiently large n such that choosing γ = O � ϵ τL2 � yields complexity O � L2 ϵ2 � The complexity does not improve significantly when τ is increased.
# G Acceleration
Here we describe an accelerated variant of IBCD in the sense of [26]. In fact, we will do something more general and accelerate ISGD, obtaining the IASGD algorithm. We again assume that machine i owns fi, which is itself a stochastic objective as in (2) with an access to an unbiased stochastic gradient gt every iteration: Egt i = ∇fi(xt). A key assumption for the accelerated SGD used to derive the best known rates [37] is so the called strong growth of the unbiased gradient estimator. Definition 2. Function φ(x) = Eζφ(x, ζ) satisfies the strong growth condition with parameters ρ, σ2, if for all x we have E∥∇φ(x, ζ)∥2  ≤ρ∥∇φ(x)∥2  + σ2.
In order to derive a strong growth property of the gradient estimator coming from the independent block coordinate sampling, we require a strong growth condition on f with respect to f1, . . . , fn and also a variance bound on stochastic gradients of each individual fi.
Assumption 5. Function f satisfies the strong growth condition with respect to f1, . . . , fn :
Similarly, given that gi = gi(x) provides an unbiased estimator of ∇fi(x), i.e. Egi = ∇fi(x), variance of gi is bounded as follows for all i:
Var [gi] ≤¯ρ∥∇fi(x)∥2 2 + ¯σ2.
Note that the variance bound (13) is weaker than the strong growth property as we always have Var [gi] ≤E � ∥gi∥2 2 � .
Note that the variance bound (13) is weaker than the strong growth property as we always have Var [gi] ≤E � ∥gi∥2 2 � . Given that Assumption 5 is satisfied, we derive a strong growth property for the unbiased gradient
Note that the variance bound (13) is weaker than the strong growth property as we always have Var [gi] ≤E � ∥gi∥2 2 � . Given that Assumption 5 is satisfied, we derive a strong growth property for the unbiased gradient estimator q ≜ 1 nτ �n i=1(∇gi)Ui in Lemma 1. Next, IASGD is nothing but the scheme from [37] applied to stochastic gradients q. For completeness, we state IASGD as Algorithm 7.
1: Input:
Starting point y0 = v0 ∈Rd, partition of Rd into m blocks u1, . . . , um, ratio of
blocks to be sampled τ, stepsize γ, number of parallel units n, acceleration parameter sequences
{a, b, η}∞
t=0
2: for t = 0, 1, . . . do
3:
xt = atvt + (1 −at)yt
4:
for i = 1, . . . , n in parallel do
5:
Sample independently and uniformly a subset of τm blocks U t
i ⊂{u1, . . . , um}
6:
Sample blocks of stochastic gradient (gt
i)U t
i such that E[gt
i | xt] = ∇fi(xt)
7:
end for
8:
qt =
1
nτ
�n
i=1(gt
i)U t
i
9:
yt+1 = xt −γqt
10:
vt+1 = btvt + (1 −bt)xt −ηtγqt.
11: end for
(12)
(13)
�   � It remains to use the stochastic gradient q (with the strong growth bound from Lemma 1) as a gradient estimate in [37][Theorem 6], which we restate as Theorem 8 for completeness. Theorem 8. Suppose that f is L smooth, µ strongly convex and Assumption 5 holds. Then, for a specific choice of parameter sequences {a, b, η}∞ t=0 (See [37][Theorem 6] for details), iterates of IASGD admit an upper bound on E � f(xt+1) � −f(x∗) of the form � �
The next corollary provides a complexity of Algorithm 7 in a simplified setting where ¯σ2 = ˜σ2 = 0. Note that ˜σ2 = 0 implies ∇fi(x∗) = 0 for all i. It again shows a desired linear scaling: given that we double the number of workers, we can halve the number of blocks to be evaluated on each machine and still keep the same convergence guarantees. It also shows that increasing τ beyond ˜ρ¯ρ n does not improve the convergence significantly. Corollary 7. Suppose that ¯σ2 = ˜σ2 = 0. Then, complexity of IASGD is � �
Theorem 8 shows an accelerated rate for strongly convex functions applying [37][Thm 6] to the bound. A non-strongly convex rate can be obtained analogously from [37][Thm 7].
# H Asynchronous ISGD
In this section we extend ISGD algorithm to the asynchronous setup. In particular, we revisit the method that was considered in [13], extend its convergence to stochastic oracle and show better dependency on quantization noise.
Algorithm 8 Asynchronous ISGD
1: Input: x0 ∈Rd, partition of Rd into m blocks u1, . . . , um, ratio of blocks to be sampled τ,
stepsize γ, # parallel units n
2: for t = 0, 1, . . . do
3:
Worker i = it is making update
4:
wt−dt
i = 1
n
�n
j=1 xt−dt
i
j
5:
xt−dt
i = proxγR(wt−dt
i)
6:
Sample independently and uniformly a subset of τm blocks U t
i ⊆{u1, . . . , um}
7:
Sample blocks of stochastic gradient (gt
i)U t
i such that E[gt
i | xt] = ∇fi(xt−dt
i)
8:
xt
i = xt−dt
i −γ(gt
i)U t
i
9:
Send (gt
i)U t
i and receive wt+1 = 1
n
�n
j=1 xt+1
j
10: end for
11: Output: xt = proxγR(wt)
Let us denote the delay of worker i at moment t by dt i. Theorem 9. Assume f1, . . . , fn are L-smooth and µ-strongly convex and let Assumption 4 be satisfied. Let us run Algorithm 8 for t iterations and assume that delays are bounded: dt i ≤M for any i and t. If γ ≤ 1 2L(τ+ 2 n ), then
  where C ≜maxi=1,...,n ∥x0 −x∗ i ∥2 2, x∗ i ≜x∗−τγ∇fi(x∗) and ⌊·⌋is the floor operator.
(14) (15)
Plugging γ = 1 2L(τ+ 2 n ) gives complexity that will be significantly improving from increasing τ until τ = 1 n, and then only if τ jumps from 1 n to 1. In contrast, doubling τ from 2 n to 4 n would make little difference. We note that if ℓ1 penalty is used, in practice zt i should be rather computed on the parameter server side because it will sparsify the vector for communication back.
# I Extra Experiments
We present exhaustive numerical experiments to verify the theoretical claims of the paper. The experiments are performed in a simulated environment instead of the honestly distributed setup, as we only aim to verify the iteration complexity of proposed methods. First, in Sec I.1 provides the simplest setting in order to gain the best possible insight – Algorithm 1 is tested on the artificial quadratic minimization problem. We compare Algorithm 1 against both gradient descent (GD) and standard CD (in our setting: when each machine samples the same subset of coordinates). We also study the effect of changing τ on the convergence speed. In the remaining parts, we consider a logistic regression problem on LibSVM data [5]. Recall that logistic regression problem is given as
where A is data matrix and b is vector of data labels: bj ∈{−1, 1}9. In the distributed scenario (everything except of Algorithm 4), we imitate that the data is evenly distributed to n workers (i.e. each worker owns a subset of rows of A and corresponding labels, all subsets have almost the same size). As our experiments are not aimed to be practical at this point (we aim to properly prove the conceptual idea), we consider multiple of rather smaller datasets: a1a (d = 123, n = 1605), mushrooms (d = 112, n = 8124), phishing (d = 68, n = 11055), w1a (d = 300, n = 2477). The experiments are essentially of 2 types: one shows that setting nτ = 1 does not significantly violate the convergence of the original method. In the second type of experiments we study the behavior for varying τ, and show that beyond certain threshold, increasing τ does not significantly improve the convergence. The threshold is smaller as n increases, as predicted by theory.
# I.1 Simple, well understood experiment
In this section we study the simplest possible setting – we test the behavior of Algorithm 1 on the artificial quadratic minimization problem. The considered quadratic objective is set as
# In this section we study the simplest possible setting – we test the behavior of Algorithm 1 on the artificial quadratic minimization problem. The considered quadratic objective is set as
fi(x) ≜1 2x⊤Mix, Mi ≜vv⊤+ � I −vv⊤� AiA⊤ i λmax � AiA⊤ i �� I −vv⊤� ,
fi(x) ≜1 2x⊤Mix, Mi ≜vv⊤+ � I −vv⊤� AiA⊤ i λmax � AiA⊤ i �� I −vv⊤� , v = v′ ∥v′∥, (17
where entries of v′ ∈Rd and Ai ∈Rd×o are sampled independently from standard normal distribution.
where entries of v′ ∈Rd and Ai ∈Rd×o are sampled independently from standard normal distribution.
In the first experiment (Figure 3), we compare Algorithm 1 with nτ = 1 against gradient descent (GD) and two versions of coordinate descent - a default version with stepsize 1 L, and a coordinate descent with importance sampling (sample proportionally to coordinate-wise smoothness constants) and optimal step sizes (inverse of coordinate-wise smoothness constants). In all experiments, gradient descent enjoys twice better iteration complexity than Algorithm 1 which is caused by twice larger stepsize. However, in each case, Algorithm 1 requires fewer iterations to CD with importance sampling, which is itself significantly faster to plain CD.
(16)
(17)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6f63/6f63d141-a117-4663-8065-22c03d546d95.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Comparison of gradient descent, (standard) coordinate descent, (standard) coordinate descent with importance sampling and Algorithm 1 on artificial quadratic problem (17).</div>
Next, we study the effect of changing τ on the iteration complexity of Algorithm 1. Figure 4 provides the result. The behavior predicted from theory is observed – increasing τ over n−1 does not significantly improve the convergence speed, while decreasing it below n−1 slows the algorithm notably.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ebdf/ebdf0ab0-3fde-462c-99b7-b5c4965a9985.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Behavior of Algorithm 1 for different τ on a simple artificial quadratic problem (17).</div>
# I.2 ISGD
In this section we numerically test Algorithm 6 for logistic regression problem. As mentioned, fi consists of set of (uniformly distributed) rows of A from (16). We consider the most natural unbiased stochastic oracle for the ∇fi – gradient computed on a subset data points from fi. In all experiments of this section, we consider constant step sizes in order to keep the setting as simple as possible and gain as much insight from the experiments as possible. Therefore, one can not expect convergence to the exact optimum. In the first experiment, we compare standard SGD (stochastic gradient is computed on single, randomly chosen datapoint every iteration) against Algorithm 6 varying n and choosing τ = 1 n
for each n. The results are presented by Figure 5. We see that, as our theory suggests, SGD and Algorithm 6 have always very similar performance.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5c8c/5c8c214f-d778-433f-81e0-81409b3ddb05.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Comparison of SGD (gradient evaluated on a single datapoint) and Algorithm 6 with nτ = 1. Constant γ = 1 5L was used for each algorithm. Label “batch size” indicates how big minibatch was chosen for stochastic gradient of each worker’s objective.</div>
Next, we study the dependence of the convergence speed on τ for various values of n. Figure 6 presents the results. In each case, τ influences the convergence rate (or the region where the iterates oscillate) significantly, however, the effect is much weaker for larger n. This is in correspondence with Corollary 5.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/422c/422c2601-7bd7-4a64-a194-a0c0536eb32c.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Behavior of Algorithm 6 while varying τ. Label “SGD” corresponds to the choice n = 1, τ = 1. Stepsize γ = 1 3L was used in every case.</div>
# I.3 IASGD
In this section we numerically test Algorithm 7 for logistic regression problem. As in the last section, fi consists of set of (uniformly distributed) rows of A from (16). The stochastic gradient is taken as a gradient on a subset data points from each fi. Note that Algorithm 7 depends on a priori unknown strong growth parameter ˆρ of unbiased stochastic gradient q10. Therefore, we first find empirically optimal ˆρ for each algorithm run by grid search and report only the best performance for each algorithm. The first experiment (Figure 7) verifies the linearity claim – we vary (n, τ) such that nτ = 1. As predicted by theory, the behavior of presented algorithms is almost indistinguishable.
10Formulas to obtain parameters of Algorithm 7 are given in [37].
10Formulas to obtain parameters of Algorithm 7 are given in [37].
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d74d/d74d4ee6-a1d0-46cf-ba75-da274f215d53.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Comparison of Algorithm 7 for various (n, τ) such that nτ = 1. Label “ASGD” corresponds to the choice n = 1, τ = 1. Label “batch size” indicates how big minibatch was chosen for stochastic gradient of each worker’s objective. Parameter ρ was chosen by grid search.</div>
Now, we once again check how different values of τ affect the convergence speed for several values of n. Figure 8 presents the results. In every case, τ slightly influences the convergence rate (or the region where the iterates oscillate), although the effect is weaker for larger n. Note that theory predicts diminishing effect of τ only above ¯ρ˜ρ n , in contrast to other sections, where the limit is 1 n.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8720/87209eeb-5342-4787-a41b-8f4897095be0.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: Behavior of Algorithm 7 while varying τ. Label “ASGD” corresponds to the choice n = 1, τ = 1. Parameter ρ was chosen by grid search.</div>
# I.4 ISAGA
We also study the convergence of shared data ISAGA – Algorithm 4. As previously, first experiment compares default SAGA against Algorithm 4 for various values of n with τ = n−1. Again, the results (Figure 911) shows what theory claims – setting nτ = 1 does not violate a convergence rate of the original SAGA.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f878/f8785564-28bd-4110-919b-0a91cf5fae9f.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: Comparison of SAGA and Algorithm 4 for various values n and τ = n−1. Stepsiz γ = 1 L(3n−1+τ) is chosen in each case.</div>
The second experiment of this section shows the convergence behavior for varying τ of Algorithm 4. The results (Figure 10) show that, for small n, the ratio of coordinates τ affects the speed heavily. However, as n increases, the effect of τ is diminishing.
11Figure 9 is identical to Figure 1. We present it again for completeness.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b149/b149a0f0-0206-4b1a-a71c-10d7d1104d14.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10: Comparison of Algorithm 4 for different values of τ. Stepsize γ = 1 L(3n−1+τ) is chosen in each case. For this experiment, we choose smaller regularization; ℓ2 = 0.000025.</div>
# I.5 ISEGA
Lastly, we numerically test Algorithm 2, and its linear convergence without Assumption 2. For simplicity, we consider R(x) = 0 in (8). In the first experiment (Figure 11), we compare Algorithm 2 for various (n, τ) such that nτ = 1. For illustration, we also plot convergence of gradient descent with the analogous stepsize. As theory predicts, the method has almost same convergence speed.12
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/096f/096f4d9f-4097-431c-988a-623985509714.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 11: Comparison of Algorithm 2 for various (n, τ) such that nτ = 1 and GD. Stepsize 1 L(1+ 1  ) was chosen for Algorithm 2 and 1 2L for GD.</div>
12We have chosen stepsize γ = 1 2L for GD, as this is the baseline to Algorithm 2 with zero variance. One can in fact set γ = 1 L for GD and get 2 times faster convergence. However, this is still only a constant factor.
The second experiment of this section shows the convergence behavior for varying τ of Algorithm 2. Again, the results (Figure 12) indicate that τ has a heavy impact on the convergence speed for small n. However, as n increases, the effect of τ is diminishing. In particular, for increasing τ beyond n−1 does not yield a significant speedup.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8d22/8d22dda2-c299-4374-8918-56a5987a4129.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 12: Comparison of Algorithm 2 for different values of τ. Stepsize γ = 1 L(1+ 1 nτ ) is chosen  each case.</div>
# J Proofs for Section 3
# J.1 Key techniques
The most important equality used many times to prove the results of this paper is a simp decomposition of expected distances into the distance of expectation and variance:
where X is any random vector with finite variance and a is an arbitrary vector from Rd. As almost every algorithm we propose average all updates coming from workers, it will be useful to bound the expected distance of mean of n random variables from the optimum. Lemma 2 provides the result. Lemma 2. Suppose that xt+1 = 1 n �n i=1 xt i. Then, we have � �
where X is any random vector with finite variance and a is an arbitrary vector from Rd. As almost every algorithm we propose average all updates coming from workers, it will be useful to bound the expected distance of mean of n random variables from the optimum. Lemma 2 provides the result. Lemma 2. Suppose that xt+1 = 1 n �n i=1 xt i. Then, we have E∥xt+1 −x∗∥2 2 ≤ ����� 1 n n � i=1 Ext+1 i −x∗ ����� 2 2 + 1 n2 n � i=1 E∥xt+1 i −Ext+1 i ∥2 2.
(18)
��� ��� Now let us proceed to expectations. Note that for any random vector X we have E∥X∥2 2 = ∥EX∥2 2 + E∥X −EX∥2 2.
Applying this to random vector X ≜1 n �n i=1 xt i −x∗, we get
��� ��� ��� ��� ��� ��� In addition, in all minibatching schemes xt+1 i are conditionally independent given xt. Therefore, for the variance term we get
��� ��� ��� ��� ��� ��� In addition, in all minibatching schemes xt+1 i are conditionally independent given xt. Therefore, for the variance term we get
��� � � ��� � Plugging it into our previous bounds concludes the proof.
# J.2 Proof of Theorem 1
f. From Lemma 8, using σ = 0 and ∇fi(x∗) = 0, we immediately obtai �
Proof. From Lemma 8, using σ = 0 and ∇fi(x∗) = 0, we immediately obtain E � ∥xt+1 −x∗∥2 2 ∥xt� ≤(1 −µγτ)∥xt −x∗∥2 2 = � 1 −µ 2L τn τn + 2(1 −τ) � ∥xt −x∗∥2 2. It remains to apply the above inequality recursively.
E � ∥xt+1 −x∗∥2 2 ∥xt� ≤(1 −µγτ)∥xt −x∗∥2 2 = � 1 −µ 2L τn τn + 2(1 −τ) � ∥xt −x∗∥2 2.
E � ∥xt+1 −x∗∥2 2 ∥xt� ≤(1 −µγτ)∥xt −x∗∥2 2 = � 1 −µ 2L τn τn + 2(1 −τ) � ∥xt −x∗∥2 2. It remains to apply the above inequality recursively.
# J.3 Proof of Theorem 4
Proof. Clearly,
Ext+1 i = xt −γτ∇fi(xt).
Let us now elaborate on the second moments. Thus, �
hus, E∥xt+1 i −Ext+1 i ∥2 2 = γ2E � ∥gt i −τ∇fi(xt)∥2 2 � = τ(1 −τ)∥∇fi(x)∥2.
� � Note that the above equality is exactly (27) with σ = 0. Thus, one can use Lemma 8 (with usin σ = 0 and ∇fi(x∗) = 0) obtaining
E � ∥xt+1 −x∗∥2 2 ∥xt� ≤(1 −µγτ)∥xt −x∗∥2 2 = � 1 −µ 2L τn τn + 2(1 −τ) � ∥xt −x∗∥2 2. t remains to apply the above inequality recursively.
� � It remains to apply the above inequality recursively.
# K Missing Parts from Sections 4 and E
# K.1 Useful Lemmata
Let us start with a variance bound, which will be useful for both Algorithm 5 and Algorithm 4. Define Φ(x) = 1 k �k i=1 φi(x), and define x+ = x−γ(∇fj(x)−αj + ¯α)U for (uniformly) randomly chosen index 1 ≤j ≤k and subset of blocks U of size τm. Define also ¯α = 1 k �k i=1 αi.
19)


Lemma 3 (Variance bound). Assume φ is µ-strongly convex and φj is L-smooth and convex for all j. Suppose that x⋆= argmin Φ(x). Then, for any x we have E∥x+ −Ex+∥2 2 ≤2γ2τ  2L(φ(x) −φ(x⋆) + 1 k k � j=1 ∥αj −∇φj(x⋆)∥2 2  . (20) Proof. Since x+ = x −γ(∇φj(x) −αj + α)U and Ex+ = x −γτ∇φ(x), we get E∥x+ −Ex+∥2 2 = γ2E ∥τ∇φ(x) −(∇φj(x) −αj + α)U∥2 2 = γ2E∥(τ∇φ(x) −(∇φj(x) −αj + α))U∥2 2 + γ2E∥τ∇φ(x) −(τ∇φ(x))U∥2 2 = γ2τE∥τ∇φ(x) −(∇φj(x) −αj + α)∥2 2 + γ2(1 −τ)τ 2∥∇φ(x)∥2 2. We will leave the second term as is for now and obtain a bound for the first one. Note that the expression inside the norm is now biased: E[τ∇φ(x) −(∇φj(x) −αj + α)] = (τ −1)∇φ(x). Therefore, E∥τ∇φ(x) −(∇φj(x) −αj + α)∥2 2 = (1 −τ)2∥∇φ(x)∥2 2 + E∥∇φ(x) −(∇φj(x) −αj + α)∥2 2. Now, since ∇φj(x) and αj are not independent, we shall decouple them using inequality ∥a + b∥2 2 ≤ 2∥a∥2 2 + 2∥b∥2 2. In particular, E∥∇φ(x) −(∇φj(x) −αj + α)∥2 2 = E∥∇φ(x) −∇φj(x) + ∇φj(x⋆) −∇φj(x⋆) + αj −α∥2 2 ≤2E∥∇φ(x) −∇φj(x) + ∇φj(x⋆)∥2 2 + 2E∥αj −∇φj(x⋆) −α∥2 2. Both terms can be simplified by expanding the squares. For the first one we have: E∥∇φ(x) −∇φj(x) + ∇φj(x⋆)∥2 2 = ∥∇φ(x)∥2 2 −2 ⟨∇φ(x), E [∇φj(x) −∇φj(x⋆)]⟩ + E∥∇φj(x) −∇φj(x⋆)∥2 2 = −∥∇φ(x)∥2 2 + 1 k k � j=1 ∥∇φj(x) −∇φj(x⋆)∥2 2.
Lemma 3 (Variance bound). Assume φ is µ-strongly convex and φj is L-smooth and convex for all j. Suppose that x⋆= argmin Φ(x). Then, for any x we have  
Similarly,
� Coming back to the first bound that we obtained for this lemma, we deduce E∥x+ −Ex+∥2 2 ≤γ2τ  (1 −τ)2∥∇φ(x)∥2 2 −2∥∇φ(x)∥2 2 + 2 k k � j=1 ∥∇φj(x) −∇
� The coefficient before ∥∇φ(x)∥2 2 is equal to γ2τ((1 −τ)2 −2 + (1 −τ)τ) = γ2τ(1 −τ −2) so we can drop this term. By smoothness of each φj,
 − where in the last step we used 1 k �k j=1 ∇φj(x⋆) = 0.
(20)
21)
(21)
(21)
) � n N � τ∥∇fj(xt) −∇fj(x∗)∥2 2 + (1 −τ) ∥αt j −∇fj(x∗)∥2 2
Similarly, for distributed setup we get  
Similarly, for distributed setup we get E   l � j=1 ∥αt+1 ij −∇fij(x∗)∥2 2  ≤τ 1 l l � j=1 ∥∇fij(xt) −∇fij(x∗)∥2 2 + � 1 −τ l � l � j=1 ∥αt ij −∇fij(x∗)∥ Using (21), the first sum of right hand side can be bounded by 2LN(f(xt) −f(x∗)) or 2Ll(f(xt) − f(x∗)).
  Using (21), the first sum of right hand side can be bounded by 2LN(f(xt) −f(x∗)) or 2Ll(f(xt) − f(x∗)).
# K.2 Proof of Theorem 5
c = 1 n � 1 γL −1 n −τ � ≥1 n �3 n + τ −1 n −τ � > 0. Furthermore, γµ ≥0, so to show ρ ≥0 it is enough to mention 1 l − 2 n2lc = 1 l − 2 n2l( 1 γL −1 n −τ) ≥1 l − 2 n2l( 3 n +τ−1 n −τ) = 0.
  Now we proceed to the proof of convergence. We are going to decompose the expected distance from xt+1 to x∗into its variance and the distance of expected iterates, so let us analyze them separately. The variance can be bounded as follows: 
(22)
As is usually done for SAGA, we are going to prove convergence using a Lyapunov function. Name let us define  
  where c = 1 n � 1 γL −1 n −τ � . Using Lemma 4 together with the bounds above, we get