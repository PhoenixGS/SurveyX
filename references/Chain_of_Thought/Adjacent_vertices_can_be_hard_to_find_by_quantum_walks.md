# Adjacent vertices can be hard to find by quantum walks∗
Nikolajs Nahimovs and Raqueline A. M. Santos
# Nikolajs Nahimovs and Raqueline A. M. Santos
Faculty of Computing, University of Latvia Raina bulv. 19, Riga, LV-1586, Latvia nikolajs.nahimovs@lu.lv, rsantos@lu.lv
# Abstract
Quantum walks have been useful for designing quantum algorithms that outperform their classical versions for a variety of search problems. Most of the papers, however, consider a search space containing a single marked element only. We show that if the search space contains more than one marked element, their placement may drastically affect the performance of the search. More specifically, we study search by quantum walks on general graphs and show a wide class of configurations of marked vertices, for which search by quantum walk needs Ω(N) steps, that is, it has no speed-up over the classical exhaustive search. The demonstrated configurations occur for certain placements of two or more adjacent marked vertices. The analysis is done for the two-dimensional grid and hypercube, and then is generalized for any graph.
# 1 Introduction
Quantum walks are quantum counterparts of classical random walks [9]. Similarly to classical random walks, there are two types of quantum walks: discrete-time quantum walks, first introduced by Aharonov et al. [1], and continuous-time quantum walks, introduced by Farhi et al. [5]. For the discrete-time version, the step of the quantum walk is usually given by coin and shift operators, which are applied repeatedly. The coin operator acts on the internal state of the walker and rearranges the amplitudes of going to adjacent vertices. The shift operator moves the walker between the adjacent vertices.
∗This work was supported by the European Union Seventh Framework Programme (FP7/20072013) under the QALGO (Grant Agreement No. 600700) project and the RAQUEL (Grant Agreement No. 323970) project, the Latvian State Research Programme NeXIT project No. 1, and the ERC Advanced Grant MQC.
Quantum walks have been useful for designing algorithms for a variety of search problems [3, 6, 2]. To solve a search problem using quantum walks, we introduce the notion of marked elements (vertices), corresponding to elements of the search space that we want to find. We perform a quantum walk on the search space with one transition rule at the unmarked vertices, and another transition rule at the marked vertices. If this process is set up properly, it leads to a quantum state in which marked vertices have higher probability than the unmarked ones. This method of search using quantum walks was first introduced in [11], which describes a quantum search in the hypercube, and has been used many times since then. Not many papers in the literature consider search by quantum walks with multiple marked vertices. Wong [13] analyzed the spatial search problem solved by continuous-time quantum walk on the simplex of complete graphs and showed that the location of marked vertices can dramatically influence the required jumping rate of the quantum walk. Wong and Ambainis [14] analysed the discrete-time quantum walk on the simplex of complete graphs and showed that if one of the complete graphs is fully marked then there is no speed-up over classical exhaustive search. Nahimovs and Rivosh [8] studied the dependence of the running time of the AKR algorithm [3] on the number and the placement of marked locations. They found some “exceptional configurations” of marked vertices, for which the probability of finding any of the marked vertices does not grow over time. Another previously known exceptional configuration for the two-dimensional grid is the “diagonal construction” by Ambainis and Rivosh [4]. In this paper, we extend the work of Nahimovs and Rivosh [7]. We study search by quantum walks on general graphs with multiple marked vertices and show a wide class of configurations of marked vertices, for which the probability of finding any of the marked vertices does not grow over time. These configurations occur for certain placements of two and more adjacent marked vertices. We prove that for such configurations the state of the algorithm is close to a stationary state. We start by reviewing the simple example of the two-dimensional grid from [7] by showing that any pair of adjacent marked vertices forms an exceptional configuration. The same construction is valid for the hypercube. We extend the proof to general graphs by showing that any pair of adjacent marked vertices having the same degree d forms an exceptional configuration, for which the probability of finding a marked vertex is limited by const · d2/N. Then, we prove that any k-clique of marked vertices forms an exceptional configuration. Additionally, we formulate general conditions for a state to be stationary given a configuration of marked vertices. Our results greatly extend the class of known exceptional configurations.
# 2 Two-dimensional grid
# 2.1 Quantum walk on the two-dimensional 
2.1 Quantum walk on the two-dimensional grid
Consider a two-dimensional grid of size √ N × √ N with periodic (torus-like) boundary conditions. Let us denote n = √ N. The locations of the grid are labeled by the coordinates (x, y) for x, y ∈{0, . . . , n −1}. The coordinates define a set of state vectors, |x, y⟩, which span the Hilbert space, HP, associated to the position. Additionally, we define a 4-dimensional Hilbert space with the set of states {|c⟩: c ∈{←, →, ↑, ↓}}, HC, associated with the direction. We refer to it as the direction or the coin subspace. The quantum walk has associated Hilbert space HP ⊗HC with basis states |x, y, c⟩for x, y ∈{0, . . . , n −1} and c ∈{↑, ↓, ←, →}. The evolution of a state of the walk is driven by the unitary operator U = S · (I ⊗C), where S is the flip-flop shift operator
and C is the coin operator, given by the Grover’s diffusion transformation
  The spatial search algorithm uses the unitary operator U ′ = S·(I ⊗C)·(Q⊗I), where Q is the query transformation which flips the sign of marked vertices, that is, Q|x, y⟩= −|x, y⟩, if (x, y) is marked and Q|x, y⟩= |x, y⟩, otherwise. The initial state of the algorithm is
Note that |ψ(0)⟩is a 1-eigenvector of U but not of U ′. If there are marked vertices the state of the algorithm starts to deviate from |ψ(0)⟩. In case of one marked vertex, after O(√N log N) steps the inner product ⟨ψ(t)|ψ(0)⟩becomes close to 0. If we measure the state at this moment, we will find the marked vertex with O(1/ log N) probability [3]. This gives the total running time of O( √ N log N) steps with amplitude amplification.
(1) (2) (3) (4)
(5)
(6)
By analyzing the quantum search algorithm for a group of marked vertices of size √ k × √ k, Ref. [7] identified that the algorithm does not work as expected when k is even, meaning that the overlap of the initial state with the state at time t, ⟨ψ(0)|ψ(t)⟩, stays close to 1. Moreover, the same effect holds for any block of size 2k × l or k × 2l, with l and k being positive integers. The reason for such behavior is that blocks of marked vertices form stationary states, as we are going to see below.
# 2.2 Stationary states for the two-dimensional grid
Consider a two-dimensional grid with two marked vertices (i, j) and (i, j + 1) Let |φa stat⟩be a state having amplitudes of all basis states equal to a except for |i, j, →⟩and |i + 1, j, ←⟩, which have amplitudes equal to −3a (see Fig. 1), that is,
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b889/b889a05b-3228-4432-9719-5c6716d3c49c.png" style="width: 50%;"></div>
Figure 1: The amplitudes of the state |φa stat⟩. The vertices (i, j) and (i + 1, j) are marked.
Then, this state is not changed by a step of the algorithm.
Lemma 1. Consider a grid of size √ N × √ N with two adjacent marked vertices (i, j) and (i + 1, j). Then the state |φa stat⟩, given by Eq. (7), is not changed by the step of the algorithm, that is, U ′|φa stat⟩= |φa stat⟩.
Proof. Consider the effect of a step of the algorithm on |φa stat⟩. The query transformation changes the signs of all the amplitudes of the marked vertices. The coin transformation performs an inversion about the average: for unmarked vertices, it does nothing, as all amplitudes are equal to a; for marked vertices, the average is 0, so applying the coin results in sign flip. Thus, (I ⊗C)(Q ⊗I) does nothing for the amplitudes of the non-marked vertices and twice flips the sign of the amplitudes of the marked vertices. Therefore, we have (I ⊗C)(Q ⊗
(7)
I)|φa stat⟩= |φa stat⟩. The shift transformation swaps the amplitudes of near-by vertices. For |φa stat⟩, it swaps a with a and −3a with −3a. Thus, we have S(I ⊗C)(Q ⊗I)|φa stat⟩= |φa stat⟩.
for a = 1/ √ 4N. Therefore, the only part of the initial state which is changed by the step of the algorithm is
Let us establish an upper bound on the probability of finding a marked vertex, � �
where M is the set of marked vertices. √
Lemma 2. Consider a grid of size √ N × √ N with two adjacent marked vertices (i, j) and (i, j + 1). Then for any number of steps, the probability of finding a marked vertex pM is O �1 N � . Proof. Follows from the proof of Theorem 2 by substituting d = 4 and m = 2N.
Lemma 2. Consider a grid of size √ N × √ N with two adjacent marked vertices (i, j) and (i, j + 1). Then for any number of steps, the probability of finding a marked vertex pM is O �1 N � .
� � Proof. Follows from the proof of Theorem 2 by substituting d = 4 and m = 2N.
� � Proof. Follows from the proof of Theorem 2 by substituting d = 4 and m = 2N.
Fig. 2 shows the absolute value of the overlap, |⟨ψ(0)|ψ(t)⟩|, and the probability of finding a marked vertex, pM, during the first 100 steps of the walk for a grid of size 50 × 50 and two different sets of marked vertices. In the first case (solid line), we have two adjacent marked vertices, M = {(0, 0), (0, 1)} and in the second case (dashed line), we have M = {(0, 0), (0, 2)}. Clearly, one can see the effect of the stationary state on the evolution. If the two marked vertices are adjacent, the overlap stays closes to 1 and the probability of finding a marked vertex stays close to the probability in the initial state. If the two marked vertices are not adjacent, the quantum walk behaves as expected (as in the single marked vertex case). Note that if we have a block of marked vertices of size k × m we can construct a stationary state as long as we can tile the block by the blocks of size 1 × 2 and 2 × 1. For example, consider M = {(0, 0), (0, 1), (2, 0), (3, 0)} for n ≥3. Then the stationary state is given by
More details on alternative constructions of stationary states for blocks of marked vertices on the two-dimensional grid can be found in [7].
(8)
(9)
(10)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/40b3/40b3f4d9-8166-4e2e-9a8a-33bb26de23a1.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Probability of finding a marked vertex, pM, and absolute value of the overlap, |⟨ψ(0)|ψ(t)⟩|, for the first 100 steps of the quantum walk on a grid o size 50 × 50. (Solid line) We have two adjacent marked vertices, (0, 0) and (0, 1) (Dashed line) We have two non-adjacent marked vertices, (0, 0) and (0, 2).</div>
Figure 2: Probability of finding a marked vertex, pM, and absolute value of the overlap, |⟨ψ(0)|ψ(t)⟩|, for the first 100 steps of the quantum walk on a grid o size 50 × 50. (Solid line) We have two adjacent marked vertices, (0, 0) and (0, 1) (Dashed line) We have two non-adjacent marked vertices, (0, 0) and (0, 2).
# 3 Hypercube
# 3.1 Quantum walk on the hypercube
The n-dimensional hypercube is a graph with N = 2n vertices where each vertex has degree n. The discrete-time quantum walk has associated Hilbert space H2n ⊗Hn. The evolution operator is given by U = S · (I ⊗C), where the shift operator, S, acts in the following way
with ⃗v being the binary representation of v. This operator moves the walker from state |⃗v⟩|c⟩to |⃗v ⊕⃗ec⟩|c⟩, where ⃗ec is the binary vector with 1 in the c-th position. Note, that vertices are connected to each other if they differ in only one bit position. The coin transformation is the Grover coin C = 2|s⟩⟨s| −I, where |s⟩= 1 √n �n−1 i=0 |i⟩. Searching for marked vertices in the hypercube is done by using the unitary operator U ′ = S · (I ⊗C) · (Q ⊗I), where Q is the query transformation, which flips the sign of marked vertices. The initial state of the algorithm is given by
In case of SKW algorithm with one marked vertex [11], if we measure the state of the quantum walk after O( √ N) time steps, we will find the marked vertex with probability 1/2 −O(1/n). Hence, we expect to repeat the algorithm a constant number of times, which gives the total running time of O( √ N) steps.
(12)
(13)
# 3.2 Stationary states for the hypercube
Consider a hypercube with two adjacent marked vertices i and j. Without loss of generality, suppose ⃗i and ⃗j differ in the first bit. Let |φa stat⟩be a state having amplitudes of all basis states equal to a except for |⃗i, 0⟩and |⃗j, 0⟩which have amplitudes equal to −(n −1)a (see Fig. 3), that is,
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b35d/b35deb19-b7e5-4422-b5db-ddb9c352bd3b.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Amplitudes of the stationary state in an n-dimensional hypercube with two adjacent marked vertices i and j.</div>
Figure 3: Amplitudes of the stationary state in an n-dimensional hypercube with two adjacent marked vertices i and j.
Lemma 3. Consider an n-dimensional hypercube with two adjacent marked vertices i and j. Then |φa stat⟩, given by Eq. (14), is not changed by a step of the algorithm, that is, U ′|φa stat⟩= |φa stat⟩. Proof. Similar to proof of Lemma 1. The initial state of the algorithm, given by Eq. (13), can be written as
Lemma 3. Consider an n-dimensional hypercube with two adjacent marked vertices i and j. Then |φa stat⟩, given by Eq. (14), is not changed by a step of the algorithm, that is, U ′|φa stat⟩= |φa stat⟩.
The initial state of the algorithm, given by Eq. (13), can be written as
� � for a = 1/ √ nN. Therefore, the only part of the initial state, which is changed by a step of the algorithm is
� � From this fact, we can establish an upper bound for the probability of finding a marked vertex.
From this fact, we can establish an upper bound for the probability of finding a marked vertex. Lemma 4. Consider an n-dimensional hypercube with two adjacent marked vertices i and j. Then for any number of steps, the probability of finding a marked vertex pM is O � n2 N � . Proof. Follows from the proof of Theorem 2 by substituting d = n and m = (nN)/2.
(14)

(15)
(16)
Fig. 4 shows the probability of finding a marked vertex and the absolute value of the overlap, |⟨ψ(0)|ψ(t)⟩|, for a hypercube of dimension n = 10 for the first 100 steps of the algorithm. We consider two different sets of marked vertices. In the first case (solid line), we have two adjacent marked vertices M = {0, 1}. In this case, the overlap stays close to 1 and the probability stays close to the probability in the initial state, because the quantum walk has a stationary state. In the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fe79/fe79f8dc-d3fd-4209-8d24-d7cf8645de64.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Probability of finding a marked vertex, pM, and absolute value of the overlap, |⟨ψ(0)|ψ(t)⟩|, for 100 steps of the quantum walk on hupercube with N = 210 vertices. (Solid line) We have two adjacent marked vertices, 0 and 1. (Dashed line) We have two non-adjacent marked vertices, 0 and 3.</div>
second case (dashed line), we have two non-adjacent marked vertices M = {0, 3}. As one can see, the behavior in the second case is very different from the behavior in the first case. Note that any set of marked vertices which can be divided into unique blocks of two adjacent marked vertices will result in a stationary state.
# 4 General graphs
# 4.1 Quantum walks on general graphs
Consider a graph G = (V, E) with a set of vertices V and a set of edges E. Let n = |V | and m = |E|. The discrete-time quantum walk on G has associated Hilbert space H2m with the set of basis states {|v, c⟩: v ∈V, 0 ≤c < dv}, where dv is the degree of vertex v. Note, that the state |v, c⟩cannot be written as |v⟩⊗|c⟩unless G is regular. The evolution operator is given by
U = SC.
(17)
The coin transformation C is the direct sum of coin transformations for individual vertices, i.e. C = Cd1 �· · · �Cdn with Cdi being the Grover diffusion transformation of dimension di. The shift operator S acts in the following way,
where v and v′ are adjacent, c and c′ represent the directions that points v to v′ and v′ to v, respectively. Searching for marked vertices is done by using the unitary operator
where Q is the query transformation, which flips the signs of the amplitudes at the marked vertices, that is,
with M being the set of marked vertices. The initial state of the algorithm is the equal superposition over all vertex-direction pairs:
It can be easily verified that the initial state stays unchanged by the evolution operator U, regardless of the number of steps. The running time of the algorithm depends on both the structure of the graph as well as the placement of marked vertices. Next, we describe a class of exceptional configurations of marked vertices, for which the probability of finding a marked vertex is limited.
# 4.2 Stationary states for general graphs
It is not difficult to see that the construction of stationary states for the twodimensional grid and the hypercube can be generalized to any graph. First, we consider only two adjacent marked vertices. Then, we show we can also construct a stationary state for three adjacent marked vertices. Later, we describe general conditions for a state to be stationary.
# 4.2.1 Two adjacent marked vertices
Consider a graph G = (V, E) with two adjacent marked vertices i and j with the same degree, that is, di = dj = d. Let |φa stat⟩be a state having all amplitudes
(18)
(19)
(20)
(21)
equal to a except of the amplitude of vertex i pointing to vertex j and amplitude of vertex j pointing to vertex i, which are equal to −(d −1)a. Fig. 5 shows the configuration of the amplitudes in the marked vertices. Then, this state is not changed by a step of the algorithm.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/cc30/cc301948-cf9b-4c70-9239-917859fdab42.png" style="width: 50%;"></div>
Figure 5: The amplitudes for the stationary state with two adjacent marke vertices i and j.
Theorem 1. Let G = (V, E) be a graph with two adjacent marked vertices i and j with di = dj = d, and let
� � where c(i,j) represents the direction which points vertex i to vertex j. Then,
� � where c(i,j) represents the direction which points vertex i to vertex j. Then,
is not affected by a step of the algorithm, that is, U ′|φa stat⟩= |φa stat⟩.
Proof. Consider the effect of a step of the algorithm. The query transformation changes the sign of all amplitudes of the marked vertices. The coin flip performs an inversion about the average of these amplitudes: for unmarked vertices, it does nothing as all amplitudes are equal to a; for marked vertices, the average is 0, so it results in sign flip. Thus, CQ does nothing for the amplitudes of the unmarked vertices and twice flips the sign of amplitudes of the marked vertices. Therefore, we have
The shift transformation swaps amplitudes of adjacent vertices. For |φa stat⟩, it swaps a with a and −(d −1)a with −(d −1)a. Thus, we have
The initial state of the algorithm |ψ(0)⟩, given by Eq. (21), can be written as |ψ(0)⟩= |φa stat⟩−|φa i,j⟩, (26)
(22)
(23)
(24)
25)
(25)
(25)
(26)
for a = 1/ √ 2m. Therefore, the only part of the initial state which is changed by a step of the algorithm is |φa i,j⟩. From this fact, we can establish an upper bound for the probability of finding a marked vertex. Theorem 2. Let G = (V, E) be a graph with two adjacent marked vertices i and j with di = dj = d, and let the probability of finding a marked vertex be given by
for a = 1/ √ 2m. Therefore, the only part of the initial state which is changed by a step of the algorithm is |φa i,j⟩. From this fact, we can establish an upper bound for the probability of finding a marked vertex.
Theorem 2. Let G = (V, E) be a graph with two adjacent marked vertices i and j with di = dj = d, and let the probability of finding a marked vertex be given by
where |ψ(t)⟩= U t|ψ(0)⟩. Then, pM = O � d2 m � , where m is the number of edges of he graph.
where |ψ(t)⟩= U t|ψ(0)⟩. Then, pM = O � d2 m � , where m is the number of edges of the graph. Proof. The only part of the initial state |ψ(0)⟩which is changed by the step of the algorithm is |φa i,j⟩= −ad � |i, c(i,j)⟩+ |j, c(j,i)⟩ � , for a = 1/ √ 2m. Since the evolution is unitary, this part will keep its norm unchanged. In this way, we want to find how big amplitudes can get in order to maximize the value of pM. This means we want to maximize the function
Proof. The only part of the initial state |ψ(0)⟩which is changed by the step of the algorithm is |φa i,j⟩= −ad � |i, c(i,j)⟩+ |j, c(j,i)⟩ � , for a = 1/ √ 2m. Since the evolution is unitary, this part will keep its norm unchanged. In this way, we want to find how big amplitudes can get in order to maximize the value of pM. This means we want to maximize the function
# 2(d −1)(a + x1)2 + 2(−(d −1)a −x2)2,
subject to 2(d −1)x2 1 + 2x2 2 = |||φa i,j⟩||2 = 2a2d2. Note that x1 represents the amplitudes going from the marked vertices to unmarked vertices and x2 represents the amplitudes going from one marked vertex to the other. Then, we obtain
pM ≤2a2(2 � (d −1)d3 + d(2d −1)) = O �d2 m � .
One of corollaries of Theorem 2 is that if the degree of the marked vertices is constant or if it does not grow as a function of n, then for large n, the probability of finding a marked vertex will stay close to the initial probability. Consider the example of the graphs in Fig. 6. Intuitively, one might expect that the probability of finding a marked vertex in graph 6a would be bigger than in 6b. However, Fig. 7 shows something different. We calculate the probability of finding a marked vertex and the absolute value of the overlap, |⟨ψ(0)|ψ(t)⟩|, for 100 steps of the evolution and when the size of the two complete graphs attached to the marked vertices is 10. The graph 6a produces a stationary state, and that is why its overlap stays very close to 1 and the probability of finding a marked vertex is smaller than in the other graph.
(27)
(28)
(29)

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8200/820049cd-e1b0-4da6-bf57-02ce812ee8c3.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Two adjacent marked vertices with the same degree.</div>
Figure 6: Graphs with two adjacent marked vertices i and j connected to two complete graphs of size 5. (a) The marked vertices have the same degree. (b) The marked vertices have different degrees.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3820/38203e89-8e78-4f28-8c07-47c1b581e531.png" style="width: 50%;"></div>
Figure 7: Probability of finding a marked vertex, pM, and absolute value of the overlap, |⟨ψ(0)|ψ(t)⟩|, for 100 steps of the quantum walk. (Solid line) Graph 6a with two adjacent marked vertices with the same degree and complete graphs of size 10. (Dashed line) Graph 6b with two adjacent marked vertices with different degree and complete graphs of size 10.
Corollary 1. Suppose the set of marked vertices, M, can be divided in groups of two adjacent marked vertices without intersecting each other. Let M′ be the set of such pairs. Then
is not changed by a step of the algorithm, that is, U ′|φa stat⟩= |φa stat⟩. Proof. Follows from the proof of Theorem 1.
# 4.2.2 Three adjacent marked vertices
Now, consider a graph G = (V, E) with three adjacent marked vertices i, j and k, that is, a marked triangle. The stationary state for this case will have the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2aa5/2aa58723-c7bf-4956-aa09-8dc5bd5e90ae.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Two adjacent marked vertices with different degrees.</div>
(30)

amplitudes in the marked vertices as depicted in Fig. 8.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5d87/5d87f8df-ab57-40dc-b746-61c5c128f52a.png" style="width: 50%;"></div>
Figure 8: Sketch of amplitudes for the stationary state with three adjacent marked vertices i, j and k.
<div style="text-align: center;">Figure 8: Sketch of amplitudes for the stationary state with three adjacent marked vertices i, j and k.</div>
Note that in order to have a stationary state the sum of amplitudes of each marked vertex should be 0, so the action of the coin operator will be a sign flip. By solving the following system of equations:
we obtain,
Theorem 3. Let G = (V, E) be a graph with three adjacent marked vertices i, j and k; and let |φa i,j,k⟩= −a(lij + 1) � |i, c(i,j)⟩+ |j, c(j,i)⟩ � −a(lik + 1) � |i, c(i,k)⟩+ |k, c(k,i)⟩ � − −a(ljk + 1) � |i, c(j,k)⟩+ |k, c(k,j)⟩ � ,
Theorem 3. Let G = (V, E) be a graph with three adjacent marked vertices i, j and k; and let |φa ⟩= −a(l + 1) � |i, c⟩+ |j, c⟩ � −a(l + 1) � |i, c⟩+ |k, c⟩ � −
Theorem 3. Let G = (V, E) be a graph with three adjacent marked vertices i, j and k; and let
|φa i,j,k⟩= −a(lij + 1) � |i, c(i,j)⟩+ |j, c(j,i)⟩ � −a(lik + 1) � |i, c(i,k)⟩+ |k, c(k,i)⟩ � − −a(ljk + 1) � |i, c(j,k)⟩+ |k, c(k,j)⟩ � ,
where lij, lik, and ljk are defined in (32). Then,
is not affected by a step of the quantum walk on G. Proof. Similar to Theorem 1.
(31)
− (32)
(33)
(34)

Fig. 9 shows two graphs with three adjacent marked vertices i, j and k connected to three complete graphs. Remember, that for three adjacent marked vertices it doesn’t matter if their degrees are different or equal. Both cases will result in a stationary state.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/da3f/da3fb3b2-6539-450d-9096-2b3bccbbdee2.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Three adjacent marked vertices with di = 4, dj = 3 and dk = 5.</div>
Figure 9: Graphs with three marked vertices connected to three complete grap of size 5.
<div style="text-align: center;">Figure 9: Graphs with three marked vertices connected to three complete graphs of size 5.</div>
In Fig. 10, we can see how the probability of finding a marked vertex depends on the degree of the marked vertices. Although, both graphs have a stationary state, the maximum probability achieved by 9b is bigger. Moreover, if we consider that the size of the complete graphs can grow, then for graph 9a the probability of finding a marked vertex will stay close to the initial probability (the degree of the marked vertices does not scales as a function of the number of vertices), while for graph 9b it will increase (dk increases with the size of the complete graph).
# 4.2.3 k-clique of marked vertices
Next, we generalize the previous result for any complete subgraph of marked vertices.
Theorem 4. Let G = (V, E) be a graph with k marked vertices v1, . . . , vk forming a k-clique. Then it forms an exceptional configuration.
Proof. Let dvj = (k −1) + d′ j, where d′ j is the number of edges of vj outside the clique. To construct a stationary state, we need to assign amplitudes to internal edges of the clique, so that the amplitudes in vertex vj sum up to d′ j. Without a loss of generality let d′ 1 < d′ 2 < · · · < d′ k. We set the amplitude of the edge (v1, v2) to −ad′ 1 and amplitudes of other edges within the clique outgoing from v1 to 0. By this, we have satisfied the condition for the vertex v1 and reduced
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/06aa/06aafb1b-a20b-4068-9b71-182a054cc5bc.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) Three adjacent marked vertices with di = 4, dj = 3 and dk = 6.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/71dd/71ddc261-c9eb-45b5-86b5-046ec34b2b4e.png" style="width: 50%;"></div>
Figure 10: Probability of finding a marked vertex, pM, and absolute value of the overlap, |⟨ψ(0)|ψ(t)⟩|, for 100 steps of the quantum walk. (Solid line) Graph 9a with complete graphs of size 20 and three adjacent marked vertices with di = 4, dj = 3 and dk = 5. (Dashed line) Graph 9b with complete graphs of size 20 and three adjacent marked vertices with di = 4, dj = 3 and dk = 21. the problem from size k to k −1. I.e. now we have a (k −1)-clique with degrees (d′ 2 −d′ 1), d′ 3, . . . , d′ k. Next, we recursively repeat the previous step until we get a 3-clique, which always have an assignment. In this way, we have constructed a stationary state for a k-clique of marked vertices.
Figure 10: Probability of finding a marked vertex, pM, and absolute value of the overlap, |⟨ψ(0)|ψ(t)⟩|, for 100 steps of the quantum walk. (Solid line) Graph 9a with complete graphs of size 20 and three adjacent marked vertices with di = 4, dj = 3 and dk = 5. (Dashed line) Graph 9b with complete graphs of size 20 and three adjacent marked vertices with di = 4, dj = 3 and dk = 21.
the problem from size k to k −1. I.e. now we have a (k −1)-clique with degrees (d′ 2 −d′ 1), d′ 3, . . . , d′ k. Next, we recursively repeat the previous step until we get a 3-clique, which always have an assignment. In this way, we have constructed a stationary state for a k-clique of marked vertices.
Note that any set of marked vertices which can be divided into unique blocks of two adjacent marked vertices with the same degree and/or k-clique marked vertices will result in a stationary state.
In this section, we describe general conditions in which a state is stationary. Theorem 5 (General conditions). Let |ψ⟩be a state with the following properties: 1 All amplitudes of the unmarked vertices are equal; 2 The sum of the amplitudes of any marked vertex is 0; 3 The amplitudes of two adjacent vertices pointing to each other are equal. Then, |ψ⟩is not changed by a step of the quantum walk, that is, U|ψ⟩= |ψ⟩. Proof. Item 1 is required in order for the coin transformation to have no effect on the unmarked vertices. It is easy to see that Cn|u⟩= |u⟩, where |u⟩= a �n−1 i=0 |i⟩ for some constant a.

Item 2 is necessary so the coin transformation can flip the signs of the amplitudes in the marked vertices. Note that previously the sign of these amplitudes were inverted by the query transformation. Item 3 is necessary for the shift transformation to have no effect on the state.
Item 2 is necessary so the coin transformation can flip the signs of the amplitudes in the marked vertices. Note that previously the sign of these amplitudes were inverted by the query transformation. Item 3 is necessary for the shift transformation to have no effect on the state. Note that the aforementioned conditions are established for the case CQ|ψ⟩= |ψ⟩and S|ψ⟩= |ψ⟩. There might be even more general conditions for the case U ′|ψ⟩= |ψ⟩.
Note that the aforementioned conditions are established for the case CQ|ψ⟩= |ψ⟩and S|ψ⟩= |ψ⟩. There might be even more general conditions for the case U ′|ψ⟩= |ψ⟩.
# 5 Conclusions
In this paper, we have demonstrated a wide class of exceptional configurations of marked vertices for the quantum walk based search on various graphs. The above phenomenon is purely quantum. Classically, additional marked vertices result in the decrease of the number of steps of the algorithm and the increase of the probability of finding a marked location. Quantumly, the addition of a marked vertex can drastically drop the probability of finding a marked location. An open question is whether the found phenomenon has analogs for other models of quantum walks (continuous time quantum walks [5], Szegedy’s quantum walk [12], staggered quantum walks [10], etc.) as well as for alternative coin operators. Another open question is algorithmic applications of the found effect. For example, in case of two-dimensional grid the search algorithm can “distinguish” between odd-times-odd and even-times-even groups of marked locations. Moreover, if there are multiple odd-times-odd and even-times-even groups of marked locations, the algorithm will find only odd-times-odd groups and will “ignore” even-times-even groups. Nothing like this is possible for classical random walks without adding additional memory resources and complicating the algorithm. Another example is the general graphs where the algorithm “ignores” marked triangles. We think that the found phenomenon should have algorithmic applications which would be very interesting to find.
Acknowledgements. The authors thank Andris Ambainis for helpful ideas and suggestions, which was a great help during this research, and Tom Wong for useful comments and careful revision of the manuscript.
# References
[1] Y. Aharonov, L. Davidovich, and N. Zagury. Quantum random walks. Physical Review A, 48(2):1687–1690, 1993.
] Y. Aharonov, L. Davidovich, and N. Zagury. Quantum random walks. Physical Review A, 48(2):1687–1690, 1993.
[2] A. Ambainis. Quantum walk algorithm for element distinctness. In Proceedings of the 45th Annual IEEE Symposium on Foundations of Computer Science, 2004. [3] A. Ambainis, J. Kempe, and A. Rivosh. Coins make quantum walks faster. In Proceedings of the 16th ACM-SIAM Symposium on Discrete Algorithms, pages 1099–1108, 2005. [4] A. Ambainis and A. Rivosh. Quantum walks with multiple or moving marked locations. In Proceedings of SOFSEM, pages 485–496, 2008. [5] E. Farhi and S. Gutmann. Quantum computation and decision trees. Physical Review A, 58:915–928, 1998. [6] F. Magniez, M. Santha, and M. Szegedy. An o(n1.3) quantum algorithm for the triangle problem. In Proceedings of SODA, pages 413–424, 2005. [7] N. Nahimovs and A. Rivosh. Exceptional configurations of quantum walks with Grover’s coin. In Proceedings of MEMICS, pages 79–92, 2015. [8] N. Nahimovs and A. Rivosh. Quantum walks on two-dimensional grids with multiple marked locations, 2015. arXiv:quant-ph/150703788. [9] R. Portugal. Quantum walks and search algorithms. Springer, New York, 2013. 10] R. Portugal, R. A. M. Santos, T. D. Fernandes, and D. N. Gon¸calves. The staggered quantum walk model. Quantum Information Processing, 15(1):85– 101, 2015. 11] N. Shenvi, J. Kempe, and K. B. Whaley. A quantum random walk search algorithm. Physical Review A, 67(052307), 2003. 12] M. Szegedy. Quantum speed-up of Markov chain based algorithms. In Proceedings of the 45th Symposium on Foundations of Computer Science, pages 32–41, 2004. 13] T. G. Wong. Spatial search by continuous-time quantum walk with multiple marked vertices. Quantum Information Processing, 15(4):1411–1443, 2016. 14] T. G. Wong and A. Ambainis. Quantum search with multiple walk steps per oracle query. Phys. Rev. A, 92:022338, Aug 2015.
