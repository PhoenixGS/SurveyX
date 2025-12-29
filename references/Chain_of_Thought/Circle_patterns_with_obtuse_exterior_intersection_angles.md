# Circle patterns with obtuse exterior intersection angle
 21 Nov 2019
# Ze Zhou
 21 Nov 2
Abstract
# Abstract
Thurston’s Circle Pattern Theorem studies existence and rigidity of circle patterns of a given combinatorial type and the given non-obtuse exterior intersection angles. Using topological degree theory, variational principle, Teichm¨uller theory, and Sard’s Theorem, this paper generalizes Circle Pattern Theorem to the case of obtuse exterior intersection angles. Mathematics Subject Classifications (2000): 52C26, 52C25.
# 1 Introduction
The patterns of circles were introduced as useful tools to study hyperbolic 3-manifolds by Thurston [27]. He also conjectured that, under a procedure of refinement, the hexagonal circle packings converge to the classical Riemann mapping [28]. In 1987 this conjecture was resolved by Rodin-Sullivan [23]. Over the past decades, circle patterns (packings) have been bridging combinatorics [25, 18], discrete and computational geometry [8, 26], minimal surfaces [4] and others. Let S be an oriented closed surface and T be a triangulation of S with the sets of vertices, edges and triangles V, E, F. Suppose that S is equipped with a constant curvature metric µ. A circle pattern P on (S, µ) is a collection of oriented circles. Say P is T -type if there exists a geodesic triangulation T (µ) of (S, µ) with the following properties: (i) T (µ) is isotopic to T ; (ii) the vertices of T (µ) coincide with the centers of circles in P. In this paper we focus on these circle patterns P = {Cv : v ∈V} such that Cu,Cw intersect with each other whenever there exists an edge between u and w. Then we have the exterior intersection angle Θ(e) ∈[0, π) for any e ∈E. One refers to Stephenson’s monograph [26] for more background. Given a function Θ : E →[0, π) defined on the edge set of T , let us consider the following question: Does there exist a T -type circle pattern whose exterior intersection angle function is given by Θ? If it does, to what extent is the circle pattern unique? A celebrated answer to this question is the following Circle Pattern Theorem due to Thurston [27, Chap. 13].
Theorem 1.1 (Thurston). Let T be a triangulation of an oriented closed surface S of genus g > 0. Suppose that Θ : E →[0, π/2] is a function satisfying the following conditions: (i) If the edges e1, e2, e3 form a null-homotopic closed curve in S , and if �3 i=1 Θ(ei) ≥π, then these edges form the boundary of a triangle of T . (ii) If the edges e1, e2, e3, e4 form a null-homotopic closed curve in S and if �4 i=1 Θ(ei) = 2π, then these edges form the boundary of the union of two adjacent triangles.
 → (i) If the edges e1, e2, e3 form a null-homotopic closed curve in S , and if �3 i=1 Θ(ei) ≥π, then these edges form the boundary of a triangle of T . (ii) If the edges e1, e2, e3, e4 form a null-homotopic closed curve in S and if �4 i=1 Θ(ei) = 2π, then these edges form the boundary of the union of two adjacent triangles. Then there exists a constant curvature (equal to 0 for g = 1 and equal to −1 for g > 1) metric µ on S such that (S, µ) supports a T -type circle pattern P with the exterior intersection angles given by Θ. Moreover, the pair (µ, P) is unique up to isometry if g > 1, and up to similarity if g = 1. There are many results relating to Circle Pattern Theorem. See, for example, the works of Andreev [2], Marden-Rodin [19], Colin de Verdi`ere [9], Chow-Luo [7] and others. Meanwhile, a natural problem arises: can we relax the requirement of non-obtuse angles in the above theorem? So far few progress has been made, except for some cases considered by Rivin [21, 22], Bao-Bonahon [3], Bobenko-Springborn [5] and Schlenker [24], respectively. The purpose of this paper is to establish some general results. First let us explain some of our terminologies. A closed (not necessarily simple) curve γ in S is called a pseudo-Jordan curve, if the complement S \γ contains a simply connected component whose boundary is equal to γ. For a pseudo-Jordan curve γ in S , an enclosing vertex set of γ consists of all vertices covered by Ω, where Ωis any simply connected component of S \ γ such that ∂Ω= γ. A pseudo-Jordan curve is said to be non-vacant if one of its enclosing vertex sets is non-empty. Theorem 1.2. Let T be a triangulation of an oriented closed surface S of genus g > 1. Suppose that Θ : E →[0, π) is a function satisfying the following conditions: (C1) If the edges e1, e2, e3 form the boundary of a triangle of T , and if �3 i=1 Θ(ei) > π, then Θ(e1) + Θ(e2) < π + Θ(e3), Θ(e2) + Θ(e3) < π + Θ(e1), Θ(e3) + Θ(e1) < π + Θ(e2). (C2) If the edges e1, e2, · · · , es form a non-vacant pseudo-Jordan curve in S , then �s i=1 Θ(ei) < (s −2)π. Then there exists a hyperbolic metric µ on S such that (S, µ) supports a T -type circle pattern P with the exterior intersection angles given by Θ. Remark 1.3. Under the assumption that every exterior intersection angle is non-obtuse, when s > 4, it is easy to see �s i=1 Θ(ei) ≤sπ/2 < (s −2)π. Therefore, Thurston’s conditions imply (C2).
(i) If the edges e1, e2, e3 form a null-homotopic closed curve in S , and if �3 i=1 Θ(ei) ≥π, then these edges form the boundary of a triangle of T .
  ii) If the edges e1, e2, e3, e4 form a null-homotopic closed curve in S and if �4 i=1 Θ(ei) = 2π then these edges form the boundary of the union of two adjacent triangles.
  Then there exists a constant curvature (equal to 0 for g = 1 and equal to −1 for g > 1) metric µ on S such that (S, µ) supports a T -type circle pattern P with the exterior intersection angles given by Θ. Moreover, the pair (µ, P) is unique up to isometry if g > 1, and up to similarity if g = 1.
There are many results relating to Circle Pattern Theorem. See, for example, the works of Andreev [2], Marden-Rodin [19], Colin de Verdi`ere [9], Chow-Luo [7] and others. Meanwhile, a natural problem arises: can we relax the requirement of non-obtuse angles in the above theorem? So far few progress has been made, except for some cases considered by Rivin [21, 22], Bao-Bonahon [3], Bobenko-Springborn [5] and Schlenker [24], respectively. The purpose of this paper is to establish some general results. First let us explain some of our terminologies. A closed (not necessarily simple) curve γ in S is called a pseudo-Jordan curve, if the complement S \γ contains a simply connected component whose boundary is equal to γ. For a pseudo-Jordan curve γ in S , an enclosing vertex set of γ consists of all vertices covered by Ω, where Ωis any simply connected component of S \ γ such that ∂Ω= γ. A pseudo-Jordan curve is said to be non-vacant if one of its enclosing vertex sets is non-empty. Theorem 1.2. Let T be a triangulation of an oriented closed surface S of genus g > 1. Suppose that Θ : E →[0, π) is a function satisfying the following conditions: (C1) If the edges e1, e2, e3 form the boundary of a triangle of T , and if �3 i=1 Θ(ei) > π, then Θ(e1) + Θ(e2) < π + Θ(e3), Θ(e2) + Θ(e3) < π + Θ(e1), Θ(e3) + Θ(e1) < π + Θ(e2). (C2) If the edges e1, e2, · · · , es form a non-vacant pseudo-Jordan curve in S , then �s i=1 Θ(ei) < (s −2)π. Then there exists a hyperbolic metric µ on S such that (S, µ) supports a T -type circle pattern P with the exterior intersection angles given by Θ. Remark 1.3. Under the assumption that every exterior intersection angle is non-obtuse, when s > 4, it is easy to see �s i=1 Θ(ei) ≤sπ/2 < (s −2)π. Therefore, Thurston’s conditions imply (C2).
There are many results relating to Circle Pattern Theorem. See, for example, the works of Andreev [2], Marden-Rodin [19], Colin de Verdi`ere [9], Chow-Luo [7] and others. Meanwhile, a natural problem arises: can we relax the requirement of non-obtuse angles in the above theorem? So far few progress has been made, except for some cases considered by Rivin [21, 22], Bao-Bonahon [3], Bobenko-Springborn [5] and Schlenker [24], respectively. The purpose of this paper is to establish some general results. First let us explain some of our terminologies. A closed (not necessarily simple) curve γ in S is called a pseudo-Jordan curve, if the complement S \γ contains a simply connected component whose boundary is equal to γ. For a pseudo-Jordan curve γ in S , an enclosing vertex set of γ consists of all vertices covered by Ω, where Ωis any simply connected component of S \ γ such that ∂Ω= γ. A pseudo-Jordan curve is said to be non-vacant if one of its enclosing vertex sets is non-empty.
Remark 1.4. The contact graph of a circle pattern is a graph which has a vertex for each circle and an edge between two vertices for each intersection component of the corresponding closed disks. For a T -type circle pattern P with acute exterior intersection angles, the contact graph G(P) is isomorphic to the 1-skeleton of T . However in obtuse angle cases the statement may not hold. There is a discussion on relations of G(P) and T in an early arXiv version of this paper. To relieve the burden of involved details, we do not include it here.
# Theorem 1.5. The pair (µ, P) in Theorem 1.2 is unique up to isometry if ( following condition:
(R1) If the edges e1, e2, e3 form the boundary of a triangle of T , then I(e1) + I(e2)I(e3) ≥ I(e2) + I(e3)I(e1) ≥0, I(e3) + I(e1)I(e2) ≥0, where I(ei) = cos Θ(ei) for i = 1, 2, 3.
Remark 1.6. The conditions (C1), (R1) are motivated by spherical trigonometry. To be specific, (C1) is satisfied if and only if �3 i=1 Θ(ei) ≤π or Θ(e1), Θ(e2), Θ(e3) are the three angles of a spherical triangle, and (R1) is satisfied if and only if �3 i=1 Θ(ei) ≤π or each side of the corresponding spherical triangle has length less than or equal to π/2. In this way one finds that (C1) is strictly wider than (R1). See Proposition 2.7 for details.
Remark 1.6. The conditions (C1), (R1) are motivated by spherical trigonometry. To be specific, (C1) is satisfied if and only if �3 i=1 Θ(ei) ≤π or Θ(e1), Θ(e2), Θ(e3) are the three angles of a spherical triangle, and (R1) is satisfied if and only if �3 i=1 Θ(ei) ≤π or each side of the corresponding spherical triangle has length less than or equal to π/2. In this way one finds that (C1) is strictly wider than (R1). See Proposition 2.7 for details. As a consequence of Theorem 1.2 and Theorem 1.5, we obtain the following result which is related to the Hyperideal Circle Pattern Theorem due to Schlenker [24]. Theorem 1.7. Let T be a triangulation of an oriented closed surface S of genus g > 1. Suppose that Θ : E →[0, π) is a function satisfying �
Theorem 1.7. Let T be a triangulation of an oriented closed surface S of genus g > 1. Suppos that Θ : E →[0, π) is a function satisfying
� whenever e1, e2, · · · , es form a pseudo-Jordan curve in S . Then there exists a hyperbolic metric µ on S such that (S, µ) supports a T -type circle pattern P with the exterior intersection angles given by Θ. Moreover, the pair (µ, P) is unique up to isometry.
� whenever e1, e2, · · · , es form a pseudo-Jordan curve in S . Then there exists a hyperbolic metric µ on S such that (S, µ) supports a T -type circle pattern P with the exterior intersection angles given by Θ. Moreover, the pair (µ, P) is unique up to isometry. One may ask the following question: what can be said regarding rigidity when (R1) is not satisfied? Let W denote the set of functions Θ : E →[0, π) satisfying (C1), (C2). Then W is a convex subset of [0, π)|E|. Below is a part answer. Theorem 1.8. For almost every Θ ∈W, there are at most finitely many T -type circle pattern pairs (µ, P), up to isometry, with the exterior intersection angles given by Θ. The paper is organized as follows: In next section we introduce some properties of three-circle configurations. In Section 3 we prove Theorem 1.2 by using topological degree theory. In Section 4, applying variational principle, we derive Theorem 1.5. As a
One may ask the following question: what can be said regarding rigidity when (R1) is not satisfied? Let W denote the set of functions Θ : E →[0, π) satisfying (C1), (C2). Then W is a convex subset of [0, π)|E|. Below is a part answer.
The paper is organized as follows: In next section we introduce some properties of three-circle configurations. In Section 3 we prove Theorem 1.2 by using topological degree theory. In Section 4, applying variational principle, we derive Theorem 1.5. As a
corollary, Theorem 1.7 is established. In Section 5 we deduce Theorem 1.8 through a combination of Teichm¨uller theory and Sard’s Theorem. The last section contains an appendix concerning some results from manifold theory. Throughout this paper, we denote by |·| the cardinality of a set, and denote by χ(·) the Euler characteristic of a manifold.
# 2 Preliminaries
# 2.1 Three-circle configurations
The following three lemmas played crucial roles in the proof of Circle Patten Theorem Please refer to [27, 7] for more information.
Lemma 2.1. For any three positive numbers ri, rj, rk and three non-obtuse angles Θi, Θ j, Θk, ther exists a configuration of three mutually intersecting circles in hyperbolic geometry, unique up t isometry, having radii ri, rj, rk and meeting in exterior intersection angles Θi, Θ j, Θk.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d7d8/d7d89593-f617-4a50-b7da-9fd4421e6ee5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: A three-circle configuration</div>
As in Figure 1, let ϑi, ϑj, ϑk denote the corresponding inner angles of the triangle o centers.
where a, b, c are positive constants.
# 2.2 Some new observations
In search of generalizing Circle Pattern Theorem, one needs to go a step further to study three-circle configurations with obtuse exterior intersection angles. Below are some new observations.
Lemma 2.4. Suppose Θi, Θj, Θk ∈[0, π) satisfy
Θi + Θ j + Θk ≤π
or
Θi + Θ j < π + Θk, Θ j + Θk < π + Θi, Θk + Θi < π + Θ j.
For any three positive numbers ri, r j, rk, there exists a configuration of three mutually intersecting circles in hyperbolic geometry, unique up to isometry, having radii ri, rj, rk and meeting in exterior intersection angles Θi, Θ j, Θk.
Proof. Set
li = cosh−1(cosh rj cosh rk + cos Θi sinh rj sinh rk)
and l j, lk similarly. It suffices to check that li, lj, lk satisfy the triangle inequalities. Namely, cosh(li + l j) > cosh lk
cosh(li −lj) < cosh lk.
Equivalently, one needs to show
Equivalently, one needs to show �cosh li cosh lj −cosh lk �2 < sinh2 li sinh2 l j = (cosh2 li −1)(cosh2 lj −1).
�cosh li cosh lj −cosh lk �2 < sinh2 li sinh2 l j = (cosh2 li −1)(cosh2 lj −1).
�cosh li cosh lj −cosh lk �2 < sinh2 li sinh2 l j = (cosh2 li −1)(cosh2 lj −1).
(1)
(3)
(4)
am = cosh rm, xm = sinh rm.
Then
Substituting (6) into (5), we need to prove
sin2 Θix2 j x2 k + sin2 Θ jx2 kx2 i + sin2 Θkx2 i x2 j + (2 + 2 cos Θi cos Θ j cos Θk)x2 i x2 j x2 k + 2λijkajakx jxkx2 i + 2λjkiakaixkxix2 j + 2λki jaiajxix jx2 k > 0,
where
λijk = cos Θi + cos Θ j cos Θk. Now we divide the proof into the following two cases: (I) Θi + Θ j + Θk ≤π. Then
Thus we deduce (7).
I) Θi+Θj+Θk > π and Θi+Θj < π+Θk, Θ j+Θk < π+Θi, Θk+Θi < π+Θj. Then there exists a spherical triangle with angles Θi, Θ j, Θk. Denote by φi, φj, φk the lengths of sides opposite to Θi, Θ j, Θk, respectively. By the second cosine law of spherical triangles,
Therefore,
λkij = cos Θk + cos Θi cos Θ j = cos φk sin Θi sin Θ j.
Note that (7) is equivalent to
(6)
(7)
(9)
(10)
where ζijk = sin2 Θi + sin2 Θ j + sin2 Θk −(2 + 2 cos Θi cos Θ j cos Θk) = sin2 Θi sin2 Θ j −(cos Θk + cos Θi cos Θ j)2 = sin2 Θi sin2 Θ j −cos2 φk sin2 Θi sin2 Θ j = sin2 φk sin2 Θi sin2 Θ j. Set yi = sin Θiaixjxk, yj = sin Θ jajxkxi and yk = sin Θkakxixj. Substituting (9) into (10 it remains to check y2 i + y2 j + y2 k + 2 cos φiyjyk + 2 cos φ jykyi + 2 cos φkyiyj > ζi jkx2 i x2 j x2 k. Completing the square gives y2 i + y2 j + y2 k + 2 cos φiyjyk + 2 cos φjykyi + 2 cos φkyiyj = (yi + cos φjyk + cos φkyj)2 + sin2 φjy2 k + sin2 φky2 j + 2(cos φi −cos φj cos φk)yjyk ≥sin2 φ jy2 k + sin2 φky2 j + 2(cos φi −cos φ j cos φk)yjyk. By the cosine law of spherical triangles, one obtains cos φi −cos φj cos φk = cos Θi sin φj sin φk. It follows that y2 i + y2 j + y2 k + 2 cos φiyjyk + 2 cos φjykyi + 2 cos φkyiyj ≥sin2 φ jy2 k + sin2 φky2 j + 2 cos Θi sin φj sin φkyjyk = (sin φjyk + cos Θi sin φkyj)2 + sin2 Θi sin2 φky2 j ≥sin2 Θi sin2 φk sin2 Θ ja2 jx2 kx2 i > sin2 φk sin2 Θi sin2 Θ jx2 i x2 j x2 k = ζijkx2 i x2 j x2 k.
ζijk = sin2 Θi + sin2 Θ j + sin2 Θk −(2 + 2 cos Θi cos Θ j cos Θk) = sin2 Θi sin2 Θ j −(cos Θk + cos Θi cos Θ j)2 = sin2 Θi sin2 Θ j −cos2 φk sin2 Θi sin2 Θ j = sin2 φk sin2 Θi sin2 Θ j.
Set yi = sin Θiaixjxk, yj = sin Θ jajxkxi and yk = sin Θkakxixj. Substituting (9) into (10 it remains to check
y2 i + y2 j + y2 k + 2 cos φiyjyk + 2 cos φjykyi + 2 cos φkyiyj ≥sin2 φ jy2 k + sin2 φky2 j + 2 cos Θi sin φj sin φkyjyk = (sin φjyk + cos Θi sin φkyj)2 + sin2 Θi sin2 φky2 j ≥sin2 Θi sin2 φk sin2 Θ ja2 jx2 kx2 i > sin2 φk sin2 Θi sin2 Θ jx2 i x2 j x2 k = ζijkx2 i x2 j x2 k.
# Thus the lemma is proved.
Thus the lemma is proved.
Remark 2.5. On the other hand, if the triangle inequalities hold for all triples of positive numbers ri, rj, rk, then the condition of Lemma 2.4 must be satisfied. Namely, the above lemma is in the optimal form.
Remark 2.6. If Θi, Θ j, Θk ∈[0, π) satisfy λi jk ≥0, λjki ≥0, λki j ≥0, the above proof can b simplified because (7) automatically holds. In fact the following Proposition 2.7 manifest that this condition is stronger than the condition of Lemma 2.4.
Proposition 2.7. Given Θi, Θ j, Θk ∈[0, π), we have λi jk ≥0, λjki ≥0, λki j ≥0 if and only if one of the following properties holds: (i) Θi + Θ j + Θk ≤π; (ii) Θi, Θ j, Θk are the angles of a spherical triangle with each side less than or equal to π/2. Proof. The ”if” part is an immediate consequence of (8) and (9). To show the ”only if” part, assume Θi + Θ j + Θk > π. It suffices to verify property (ii). First a routine calculation gives
roposition 2.7. Given Θi, Θ j, Θk ∈[0, π), we have λi jk ≥0, λjki ≥0, λki j ≥0 if and only if one f the following properties holds:
Proof. The ”if” part is an immediate consequence of (8) and (9). To show the ”only if” part, assume Θi + Θ j + Θk > π. It suffices to verify property (ii) First a routine calculation gives
If Θ j = 0, one shows Θi + Θk ≤π, which yields Θi + Θj + Θk ≤0. This contradicts to th assumption that Θi + Θ j + Θk > π. If Θk = 0, similar arguments lead to a contradiction. Thus Θ j, Θk ∈(0, π), which implies
As a result,
Θi + Θ j < Θk + π, Θi + Θk < Θ j + π.
Similarly,
Hence there exists a spherical triangle with angles Θi, Θ j, Θk. Using (9), it is easy to see each side of this spherical triangle has length less than or equal to π/2. □
Proof. The proof of (1) is similar to the related results in Ge-Jiang [11] and Ge-Xu [12]. Due to the cosine law of hyperbolic triangles, we have
It suffices to show A, B →0 as ri →+∞. For m = i, j, k, setting cm = min{cos Θm, 0}, on obtains 0 < 1 + cm ≤1. Moreover,
t suffices to show A, B →0 as ri →+∞. For m = i, j, k, setting cm = min{cos Θm, 0}, one obtains 0 < 1 + c ≤1. Moreover,
uffices to show A, B →0 as ri →+∞. For m = i, j, k, setting cm = min{cos Θm, 0}, one
cosh lk = cosh ri cosh r j + cos Θk sinh ri sinh rj ≥cosh ri cosh rj + ck sinh ri sinh rj ≥cosh ri cosh rj + ck cosh ri cosh rj = (1 + ck) cosh ri cosh rj ≥(1 + ck) cosh ri.
# cosh lj ≥(1 + cj) cosh rk cosh ri ≥(1 + cj) cosh ri.
As a result,
It follows that
Then B →0 as ri →+∞, which concludes (1). The formula (2) is derived from a direct computation. Let us consider (3). By the cosine law of hyperbolic triangles, we h
As (ri, rj, rk) →(0, 0, c), one obtains
Consequently,
It remains to prove (4). As (ri, r j, rk) →(0, 0, 0), the area of the triangle tends to zero. Due to the Gauss-Bonnet formula, one obtains the desired result. □
Remark 2.9. From the above proof, one finds that (1) holds uniformly, no matter how the other two radii behave, even if either or both of them tend to infinity or zero. In addition, regarding ϑi as the function of ri, rj, rk, Θi, Θj, Θk, the formula is true as (ri, rj, rk, Θi, Θ j, Θk) varies in R3 + × Λ, where Λ ⊂[0, π)3 is any compact set such that the condition of Lemma 2.4 is satisfied.
# 3 Existence
To prove Theorem 1.2, a natural strategy is to follow Thurston [27]. Unfortunately, for three-circle configurations with obtuse exterior intersection angles, the result similar to Lemma 2.2 may not hold (see Remark 4.2). Part of his method does not work. In order to overcome the difficulty, we will employ the topological degree theory. As a comparison, this approach has the advantage of being independent on rigidity.
# 3.1 Thurston’s construction
Recall that S is an oriented closed surface of genus g > 1 and T is a triangulation of S with the sets of vertices, edges and triangles V, E, F. Let r ∈R|V| + be a radius vector, which assigns each vertex v ∈V a positive number r(v). Then r together with the function Θ : E →[0, π) in Theorem 1.2 determines a hyperbolic cone metric structure on S as follows. For each triangle △(vivjvk) of T , one associates it with the hyperbolic triangle determined by the centers of three mutually intersecting circles with radii
r(vi), r(vj), r(vk)
and exterior intersection angles
Θ�[vi, vj]�, Θ�[vj, vk]�, Θ�[vk, vi]�.
������ Because Θ satisfies (C1), Lemma 2.4 implies the above procedure works well. Gluing these hyperbolic triangles produces a metric surface �S, µ(r)�which is locally hyperbolic with possible cone type singularities at the vertices. For each v ∈V, the vertex curvature k(v) is defined by k(v) = 2π −σ(v),
□
where σ(v) denotes the cone angle at v. More precisely, σ(v) is equal to the sum of all inner angles having vertex v. Write k(v) as k(v)(Θ, r). If there exists a radius vector r∗such that k(v)(Θ, r∗) = 0 for all v ∈V, then it produces a smooth hyperbolic metric on S . For every v ∈V, on �S, µ(r∗)� drawing the circle centered at v with radius r∗(v), one then obtains the demanded circle pattern. Consider the following curvature map
� � � � The major purpose of this section is to show that the origin O = (0, 0, · · · , 0) ∈R|V| belongs to the image of Th(Θ, ·).
# 3.2 Topological degree
# Let us make use of the topological degree theory. Specifically, one finds a relatively compact open set Λ ⊂R|V| + and determines the degree deg(Th(Θ, ·), Λ, O). Once showing
Let us make use of the topological degree theory. Specifically, one finds a relatively compact open set Λ ⊂R|V| + and determines the degree deg(Th(Θ, ·), Λ, O). Once showing
deg(Th(Θ, ·), Λ, O) �0,
it follows from Theorem 6.8 that O is in the image of Th(Θ, ·). We shall compute deg(Th(Θ, ·), Λ, O) via homotopy method. Namely, deform Th(Θ, ·) to another map which is relatively easier to manipulate. For each t ∈[0, 1], note that the function Θt = tΘ satisfies (C1). Applying Thurston’s construction, Tht(·) = Th(Θt, ·) is well-defined. Moreover, Tht forms a homotopy from Th(Θ, ·) to Th(0, ·).
Lemma 3.1. There exists a relatively compact open set Λ ⊂R|V| + such tha Tht �R|V| + \ Λ �⊂R|V| \ {O}, ∀t ∈[0, 1].
Tht �R|V| + \ Λ �⊂R|V| \ {O}, ∀t ∈[0, 1].
�  � Proof. Let us exhaust R|V| + by an increasing sequence of relatively compact open sets {Λn}. Assume on the contrary that the lemma is not true. For each n, one obtains tn ∈[0, 1] and rn ∈R|V| + \ Λn satisfying
Because {Λn} exhausts R|V| + , there exist v0 ∈V and {rnk} such that rnk(v0) →+∞ or rnk(v0) →0.
In the first case, by Lemma 2.8 and Remark 2.9, it follows from formula (1) that k(v0)(tnkΘ, rnk) →2π,
(11)
which contradicts to (11). In the second case, let V0 ⊂V be the set of vertices v ∈V for which rnk(v) →0. Then V0 is a non-empty subset of V. We denote by S (V0) the union of these q-cells (q = 0, 1, 2) of T that have at least one vertex in V0, and denote by Lk(V0) the set of pairs (e, v) an edge e and a vertex v with the following properties:
(i) v ∈V0; (ii) ∂e ∩V0 = ∅; (iii) e and v form a triangle of T .
Evidently, S (V0) is an open set of S and thus is a surface. Suppose that tnk converges to  number t∗∈[0, 1]. Otherwise, one picks up a convergent subsequence. By the followin Proposition 3.2, we have
� v∈V0 k(v)(tnkΘ, rnk) →− � (e,v)∈Lk(V0) �π −t∗Θ(e)�+ 2πχ�S (V0)�.
� Combining with formula (11) gives
0 = − � (e,v)∈Lk(V0) �π −t∗Θ(e)�+ 2πχ�S (V0)�.
0 = − � (e,v)∈Lk(V0) �π −t∗Θ(e)�+ 2πχ�S (V0)�.
� ���� Without loss of generality, assume that S (V0) is connected. Otherwise, one considers the connected component of S (V0). Note that
Substituting these into (12), one derives
0 = 2π(2 −2g),
which contradicts to the condition g > 1. Let h0 > 0. If g0 > 0 or h0 > 1, it is easy to see (12) leads to a contradiction. Let g0 = 0 and h0 = 1. Then S (V0) is a simply connected domain in S . Suppose tha Lk(V0) = {(ei, vi)}s i=1. Formula (12) gives
0 = − �s i=1 �π −t∗Θ(ei)�+ 2π,
(12)
� � Meanwhile, one finds that e1, · · · , es form a non-vacant pseudo-Jordan curve. According to (C2), we have �s
� � Meanwhile, one finds that e1, · · · , es form a non-vacant pseudo-Jordan curve. Accordin to (C2), we have �s Θ(ei) < (s −2)π,
� i=1 Θ(ei) < (s −2)π, which leads to a contradiction. □ Proposition 3.2. Let S (V0) and Lk(V0) be as above. Then � v∈V0 k(v)(tnkΘ, rnk) →− � (e,v)∈Lk(V0) �π −t∗Θ(e)�+ 2πχ�S (V0)�. Proof. For m = 1, 2, 3, let Fm(V0) be the set of triangles having exactly m vertices in V0. Using Lemma 2.8, it follows from formulas (2), (3) and (4) that � v∈V0 k(v)(tnkΘ, rnk) →2π|V0| − � (e,v)∈Lk(V0) �π −t∗Θ(e)�−π|F2(V0)| −π|F3(V0)|. Let E(V0), F(V0) denote the sets of edges and triangles having at least one vertex in V0. It is easy to see |F(V0)| = |F1(V0)| + |F2(V0)| + |F3(V0)|. Meanwhile, note that |F1(V0)| = |Lk(V0)| and 3|F(V0)| = 2|E(V0)| + |Lk(V0)|. Combining the above relations gives 2|V0| −|F2(V0)| −|F3(V0)| = 2|V0| −|F(V0)| + |F1(V0)| −�|F1(V0)| −|Lk(V0)|� = 2|V0| −|F(V0)| + |Lk(V0)| + �3|F(V0)| −2|E(V0)| −|Lk(V0)|� = 2�|V0| −|E(V0)| + |F(V0)|� = 2χ�(S (V0)�. As a result, we have � v∈V0 k(v)(tnkΘ, rnk) →− � (e,v)∈Lk(V0) �π −t∗Θ(e)�+ 2πχ�S (V0)�. □ Theorem 3.3. Let Th(Θ, ·) and Λ be as above. Then deg(Th(Θ, ·), Λ, O) = 1. 13
which leads to a contradiction.
� � ���� Proof. For m = 1, 2, 3, let Fm(V0) be the set of triangles having exactly m vertices in V0 Using Lemma 2.8, it follows from formulas (2), (3) and (4) that
� � ���� Proof. For m = 1, 2, 3, let Fm(V0) be the set of triangles having exactly m vertices in V0. Using Lemma 2.8, it follows from formulas (2), (3) and (4) that � v∈V0 k(v)(tnkΘ, rnk) →2π|V0| − � (e,v)∈Lk(V0) �π −t∗Θ(e)�−π|F2(V0)| −π|F3(V0)|. Let E(V0), F(V0) denote the sets of edges and triangles having at least one vertex in V0. It is easy to see |F(V0)| = |F1(V0)| + |F2(V0)| + |F3(V0)|. Meanwhile, note that |F1(V0)| = |Lk(V0)| and 3|F(V0)| = 2|E(V0)| + |Lk(V0)|. Combining the above relations gives 2|V0| −|F2(V0)| −|F3(V0)| = 2|V0| −|F(V0)| + |F1(V0)| −�|F1(V0)| −|Lk(V0)|� = 2|V0| −|F(V0)| + |Lk(V0)| + �3|F(V0)| −2|E(V0)| −|Lk(V0)|� = 2�|V0| −|E(V0)| + |F(V0)|�
and
Combining the above relations gives
As a result, we have
� v∈V0 k(v)(tnkΘ, rnk) →− � (e,v)∈Lk(V0) �π −t∗Θ(e)�+ 2πχ�S (V0)�.
Theorem 3.3. Let Th(Θ, ·) and Λ be as above. Then deg(Th(Θ, ·), Λ, O) = 1.
Theorem 3.3. Let Th(Θ, ·) and Λ be as above. Then
deg(Th(Θ, ·), Λ, O) = 1.
Proof. It suffices to compute deg(Th0, Λ, O). By Theorem 1.1, Λ ∩Th−1 0 (O) consists of a unique point. Using Lemma 2.2, one shows that the Jacobian matrix of Th0(·) has positive diagonal entries and strictly diagonally dominant columns. As a result, the determinant of this matrix is positive and the tangent map preserves the orientation. Therefore,
deg(Th0, Λ, O) = 1.
Owing to Lemma 3.1 and Theorem 6.7, one deduces the conclusion.
Proof of Theorem 1.2. Because of Theorem 3.3 and Theorem 6.8, O is in the image of the map Th(Θ, ·). Consequently, there exists a hyperbolic metric µ on S such that (S, µ) supports a T -type circle pattern P with the exterior intersection angles given by Θ. □
# 4 Rigidity
# 4.1 Three-circle configurations revisited
For a triple of indices i, j, k, recall that λi jk = cos Θi + cos Θ j cos Θk. Under the condition that λi jk ≥0, λjki ≥0, λkij ≥0, it follows from Remark 2.6 that the inner angles ϑi, ϑj, ϑk are well-defined functions of (ri, rj, rk). Our key observation in this section is the following generalization of Lemma 2.2.
Lemma 4.1. Suppose Θi, Θ j, Θk ∈[0, π) satisfy
Then
Proof. We mention that part of computations here is parallel to these in Colin de Verdi`ere [9], Chow-Luo [7], Guo-Luo [14], Guo [15], Xu [29] and others. For simplicity, we use the same notations in the proof of Lemma 2.4. By the cosine law of hyperbolic triangles,
Taking the partial derivative with respect to li gives
where
Kijk = sin ϑi sinh lj sinh lk.
□
□
Hence
Meanwhile,
Combining (13), (14) and (15), one obtains
It follows that
where the equality holds if and only if Θk = 0 and Θi + Θ j = π. Fix rj, rk and let ri vary. Then li stays constant. Because ∂ϑj/∂ri ≥0 and ∂ϑk/∂ri ≥0, that means the other two sides move outwards or remain unchanged as ri increases. Note that these two sides can not stay unchanged simultaneously. Thus the area Area(△) of the triangle is a strictly increasing function of ri. By the Gauss-Bonnet formula, one derives
Furthermore,
xi = −λijk �1 + |λjki| �t, x j = λ2 i jk t, xk = −3λi jk �1 + |λjki| �t.
xi = −λijk �1 + |λjki| �t, x j = λ2 i jk t, xk = −3λi jk �1 + |λjki| �t.
xi = −λijk �1 + |λjki| �t, x j = λ2 i jk t, xk = −3λi jk �1 + |λjki| �t. Using (16), one finds that ∂ϑi/∂rj < 0 when t is sufficiently small. Thus the condition of Lemma 4.1 can not be relaxed further.
� � � � Using (16), one finds that ∂ϑi/∂rj < 0 when t is sufficiently small. Thus the condition of Lemma 4.1 can not be relaxed further.
(13)
(14)
(15)
(16)
# 4.2 Variational principle
Let us establish rigidity via the variational approach poineered by Colin de Verdi`ere  Remind that similar methods have been used by Rivin [21, 22], Bobenko-Springborn  Guo-Luo [14], Guo [15] and others in many situations. Suppose that V = {v1, · · · , v|V|}. To simplify notations, for i = 1, 2, · · · , |V|, set
For a vertex vi ∈V and a triangle △∈F incident to vi, denote by ϑ△ i the inner angle of the triangle △at vi. Then �� �
 △ ki = 2π − �� △∈F({vi}) ϑ△ i � . Consider the change of variables ui = ln tanh(ri/2). It is easy to see
ki = 2π − �� △∈F({vi}) ϑ△ i � . Consider the change of variables ui = ln tanh(ri/2). It is easy to see
�� � Consider the change of variables ui = ln tanh(ri/2). It is easy to see
� � � � Lemma 4.3. Under the condition (R1), the Jacobian matrix of Th(Θ, ·) in terms of u is symmetri and positive definite.
� � � � Lemma 4.3. Under the condition (R1), the Jacobian matrix of Th(Θ, ·) in terms of u is symmetric and positive definite.
� � � �  Under the condition (R1), the Jacobian matrix of Th(Θ, ·) in terms of u is symmetric
Proof. In case that vi, vj is a pair of non-adjacent vertices, then
In case that [vi, vj] = e ∈E, there exist triangles △1, △2 adjacent to e. Lemma 4.1 gives
Hence the Jacobian matrix of Th(Θ, ·) in terms of u is symmetric. Meanwhile, a simple computation shows
It follows that
����� ∂ki ∂ui �����− � j�i ����� ∂ki ∂uj �����= �|V| j=1 ∂ki ∂uj = �|V| j=1 ∂kj ∂ui = � △∈F({vi}) ∂ ∂ui Area(△) > 0. The Jacobian matrix is strictly diagonally dominant and hence is positive definite.
����� ∂ki ∂ui �����− � j�i ����� ∂ki ∂uj �����= �|V| j=1 ∂ki ∂uj = �|V| j=1 ∂kj ∂ui = � △∈F({vi}) ∂ ∂ui Area(△) > 0.
���� ���� � ���� ���� � � � The Jacobian matrix is strictly diagonally dominant and hence is positive definite
Now it is ready to prove Theorem 1.5.
Proof of Theorem 1.5. Consider the following 1-form
� ki/∂uj = ∂kj/∂ui, it is easy to see ω is closed. Thus the functio
is well-defined and does not depend on the particular choice of a piecewise smooth arc in R|V| −from an initial point u(0) to u. Note that the Hessian of Φ(·) is equal to the Jacobian of Th(Θ, ·), which is positive definite by Lemma 4.3. That means Φ is a strictly convex function of u. From Thurston’s construction, the demanded circle pattern is related to the critical point of Φ. Because Φ is strictly convex, the critical point must be unique. We thus finish the proof. □ Remark 4.4. Under the condition (R1), the above analysis shows that the Jacobian matrix of Th(Θ, ·) is non-singular. It follows from the Implicit Function Theorem that the demanded radius vector r∗depends on Θ smoothly. Remark 4.5. Based on observations in above parts, Xu [29] recently studied rigidity of circle patterns with inversive distances. His work generalizes the results of Guo-Luo [14] and Theorem 1.5. Remark 4.6. Given an initial point, let it evolve with the negative gradient flow of Φ. Following Chow-Luo [7], one can show that the flow converges exponentially fast to the demanded circle pattern vector. See the recent work of Ge-Hua-Zhou [10] for details. Proof of Theorem 1.7. The existence is an immediate consequence of Theorem 1.2. For rigidity, suppose that e1, e2, e3 form the boundary of a triangle of T . According to the condition, we have Θ(e1) + Θ(e2) + Θ(e3) < π. Recall that I(ei) = cos Θ(ei) for i = 1, 2, 3. By Proposition 2.7, one obtains I(e1) + I(e2)I(e3) ≥0, I(e2) + I(e3)I(e1) ≥0, I(e3) + I(e1)I(e2) ≥0. It follows from Theorem 1.5 that the rigidity holds. □
is well-defined and does not depend on the particular choice of a piecewise smooth arc in R|V| −from an initial point u(0) to u. Note that the Hessian of Φ(·) is equal to the Jacobian of Th(Θ, ·), which is positive definite by Lemma 4.3. That means Φ is a strictly convex function of u. From Thurston’s construction, the demanded circle pattern is related to the critical point of Φ. Because Φ is strictly convex, the critical point must be unique. We thus finish the proof. □ Remark 4.4. Under the condition (R1), the above analysis shows that the Jacobian matrix of Th(Θ, ·) is non-singular. It follows from the Implicit Function Theorem that the demanded radius vector r∗depends on Θ smoothly. Remark 4.5. Based on observations in above parts, Xu [29] recently studied rigidity of circle patterns with inversive distances. His work generalizes the results of Guo-Luo [14] and Theorem 1.5. Remark 4.6. Given an initial point, let it evolve with the negative gradient flow of Φ. Following Chow-Luo [7], one can show that the flow converges exponentially fast to the demanded circle pattern vector. See the recent work of Ge-Hua-Zhou [10] for details. Proof of Theorem 1.7. The existence is an immediate consequence of Theorem 1.2. For rigidity, suppose that e1, e2, e3 form the boundary of a triangle of T . According to the condition, we have Θ(e1) + Θ(e2) + Θ(e3) < π. Recall that I(ei) = cos Θ(ei) for i = 1, 2, 3. By Proposition 2.7, one obtains I(e1) + I(e2)I(e3) ≥0, I(e2) + I(e3)I(e1) ≥0, I(e3) + I(e1)I(e2) ≥0. It follows from Theorem 1.5 that the rigidity holds. □
Remark 4.5. Based on observations in above parts, Xu [29] recently studied rigidity o circle patterns with inversive distances. His work generalizes the results of Guo-Luo [14 and Theorem 1.5.
□
# 5 Finiteness
To obtain the proof Theorem 1.8, some knowledge on Teichm¨uller theory is needed. One refers to [1, 17] for basic background. See also [6] from the geometric and topological viewpoint.
# 5.1 Configuration spaces
Recall that S is an oriented closed surface of genus g > 1. Denote by T (S ) the Teichm¨uller space of S , which parameterizes the equivalence classes of marked hyperbolic metrics on S . Here a marking is an isotopy class of orientation-preserving homeomorphism from S to itself. And two marked hyperbolic metrics µ, µ′ ∈T (S ) are equivalent if there exists an isometry φ : (S, µ) →(S, µ′) isotopic to the identity map. T (S ) admits the structure of a smooth manifold of dimension 6g −6. Let pi, pj ∈S be two points and let γ be a simple curve with endpoints pi, pj. For any µ ∈T (S ), on (S, µ) there exists a unique geodesic in the isotopy class [γ] with the endpoints pi, pj. Let d[γ](µ, pi, pj) denote the length of this geodesic. Under the above manifold structure, d[γ](µ, pi, pj) depends on µ smoothly. See Buser’s monograph [6, Chap. 6] for details. We endow S with a smooth structure and define a space Z = T (S ) × (S × R+)|V|. Then Z is a smooth manifold of dimension
Note that
# 2|E| = 3|F|.
Therefore,
# dim (Z) = 6g −6 + 3|V| −�2|E| −3|F|� = 6g −6 + 3�|V| −|E| + |F|�+ |E|
dim (Z) = 6g −6 + 3|V| −�2|E| −3|F|� = 6g −6 + 3�|V| −|E| + |F|�+ |E| = |E|.
 || A point z = (µ, p1, r1, · · · , p|V|, r|V|) ∈Z is called a configuration, since it gives a choice of a marked hyperbolic metric µ on S and assigns each vertex vi an oriented circle Ci on (S, µ), where Ci denotes the circle centered at pi with radius ri. To summarise, a configuration z gives a marked hyperbolic metric µ on S together with a circle pattern P on (S, µ). Briefly, we say z gives a circle pattern pair (µ, P). Let ZT ⊂Z denote the subspace of configurations that give T -type circle pattern pairs. More precisely, z ∈ZT if and only if there exists a geodesic triangulation T (z) of (S, µ) such that T (z) is isotopic to T and the vertices of T (z) are p1, p2, · · · , p|V|. Clearly, ZT is open in Z and thus is a smooth manifolds of dimension |E|.
# For each e = [vi, vj] ∈E, the inversive distance I(e, z) is defined by
where [γe] represents the isotopy class of the edge e. Let ZP ⊂ZT be the subspace of configurations z for which
Because I(e, z) is continuous, ZP is open in ZT , which implies ZP is a smooth manifold o dimension |E|. Obviously, a configuration z ∈ZP gives a T -type circle pattern with the exterior intersection angle Θ(e, z) ∈(0, π) for each e ∈E, where
One establishes the following smooth map
# e establishes the following smooth map
Ev : ZP −→ Y := (0, π)|E| z �−→ �Θ(e1, z), Θ(e2, z), · · · �.
# � 5.2 Rigidity from the viewpoint of Sard’s Theorem
Note that a regular point of the above map gives a circle pattern pair which is locally determined up to isometry by its exterior intersection angles. Sard’s Theorem indicates that local rigidity is a generic property.
Proof of Theorem 1.8. Set W0 = Ev(Z0), where Z0 ⊂ZP denotes the set of critical points of Ev. Owing to Sard’s Theorem (Theorem 6.2), W0 has zero measure. Meanwhile, the boundary set ∂W of W also has zero measure. Thus W \ (W0 ∪∂W) is a subset of W with full measure. Note that dim (Z) dim (Y)E
For each Θ ∈W \ (W0 ∪∂W) ⊂Y, the Regular Value Theorem (Theorem 6.1) implies that Ev−1(Θ) is a discrete set. Combining with the following Proposition 5.1, this set must be finite. Thus the theorem is proved. □
Proposition 5.1. Let Θ be as above. Then Ev−1(Θ) is compact.
Proof. It suffices to show any sequence {zn} ⊂Ev−1(Θ) contains a convergent subsequence in ZP. Let (µn, Pn) be the circle pattern pair give by zn. First we claim that there exists a subsequence {znk} converging to a point z∗∈Z. In view of the topology of Z = T (S ) × (S × R+)|V|, one needs to check the following two properties:
Y := (0, π)|E|
1. No circle in Pn degenerates to a point or becomes infinitely large. 2. The sequence {µn} is included in a compact subset of T (S ). Because Θ satisfies (C1), (C2), similar arguments to the proof of Lemma 3.1 show that the first property holds. For the second property, suppose it is not true. By Fenchel-Nielsen’s coordinates [17, 6], that means there exists at least one simple closed geodesic γn in (S, µn) whose length ℓ(γn) tends to zero or infinity. One claims that the diameter of (S, µn) tends to infinity. Indeed, if ℓ(γn) →+∞, the statement trivially holds. If ℓ(γn) →0, it follows from the Collar Theorem [6, Chap. 4] that there is an embedding cylinder domain Cy(γn) in (S, µn) such that Cy() �p ∈(S) | dist(p) ≤d �
where dn > 0 satisfies
  sinh �ℓ(γn)/2�sinh dn = 1.
 �� Note that dn →∞as ℓ(γn) →0, which also implies the diameter of (S, µn) tends to infinity. On the other hand, because the radius of every circle in Pn is bounded from above, the diameter of (S, µn) must be bounded from above. This leads to a contradiction. Next we show z∗∈ZT . Denote by Tnk the geodesic triangulation for Pnk. One needs to check that no triangle of Tnk becomes infinitely large or degenerates to a point or a line segment. Using Lemma 2.4, this follows from the first property. To finish the proof, it remains to prove z∗∈ZP. Because {znk} ⊂Ev−1(Θ) ⊂ZP, for any e ∈E, we have Θ(e, znk) = Θ(e). As nk →∞, one obtains Θ(e, z∗) = Θ(e) ∈(0, π), which implies z∗∈ZP. □
# 6 Appendix
In this section we give a simple introduction to some results on manifolds, especially the topological degree theory. One refers to [13, 16, 20] for more background. Let M, N be two smooth manifolds. A point x ∈M is called a critical point for a C1 map f : M →N if the tangent map d fx : TxM →T f(x)N is not surjective. Let C f denote the set of critical points of f. And N \ f(C f ) is defined to be the set of regular values of f. Theorem 6.1 (Regular Value Theorem). Let f : M →N be a Cr (r ≥1) map, and let y ∈N be a regular value of f. Then f −1(y) is a closed Cr submanifold of M. If y ∈im(f), then the codimension of f −1(y) is equal to the dimension of N. Theorem 6.2 (Sard’s Theorem). Let M, N be manifolds of dimensions m, n and f : M →N be a Cr map. If r ≥max{1, m −n + 1}, then f(C f ) has zero measure in N.
Now assume that M, N are oriented manifolds of the same dimensions. Let Λ ⊂M be a relatively compact open subset. Namely, Λ has compact closure in M. For a point y ∈N, we use f⋔Λy to denote that y is a regular value of the restriction map f : Λ →N. Suppose f ∈C0( ¯Λ, N) ∩C∞(Λ, N), f⋔Λy, and f(∂Λ) ∈N \ {y}. If Λ ∩f −1(y) is empty, the topological degree deg(f, Λ, y) is defined to be zero. If Λ∩f −1(y) is non-empty, the Regular Value Theorem implies that it consists of finite points.
deg(f, Λ, y) = �s i=1 sgn(f, xi).
� Here sgn( f, xi) = +1, if the tangent map d fxi : TxiM →TyN preserves the orientation; Otherwise, sgn(f, xi) = −1.
Proposition 6.4. Suppose fi ∈C0( ¯Λ, N) ∩C∞(Λ, N), fi⋔Λy and fi(∂Λ) ⊂N \ {y}, i = 0, 1. If there exists a homotopy H ∈C0(I × ¯Λ, N) such that (i) H(0, ·) = f0(·), H(1, ·) = f1(·), (ii) H(I × ∂Λ) ⊂N \ {y},
deg(f0, Λ, y) = deg(f1, Λ, y). The following lemma is a consequence of Sard’s Theorem. Lemma 6.5. Given f ∈C0( ¯Λ, N) and y ∈N, if f(∂Λ) ⊂N \ {y}, then there exists g ∈C0( ¯Λ, N)  C∞(Λ, N) and H ∈C0(I × ¯Λ, N) such that (i) g⋔Λy, (ii) H(0, ·) = f(·), H(1, ·) = g(·), (iii) H(I × ∂Λ) ⊂N \ {y}. It is ready to define the topological degrees for general continuous maps. Definition 6.6. For any f ∈C0( ¯Λ, N) and y ∈N that satisfy f(∂Λ) ⊂N \ {y}, one defines deg(f, Λ, y) = deg(g, Λ, y), where g is given by Lemma 6.5.
deg(f0, Λ, y) = deg(f1, Λ, y).
The following lemma is a consequence of Sard’s Theorem.
where g is given by Lemma 6.5.
By Proposition 6.4, deg(f, Λ, y) is well-defined and does not depend on the particular choice of g. Below are some properties of topological degrees. Theorem 6.7. Suppose fi ∈C0( ¯Λ, N) satisfies fi(∂Λ) ⊂N\{y}, i = 0, 1. If there exists a homotopy H ∈C0(I × ¯Λ, N) such that (i) H(0, ·) = f0(·), H(1, ·) = f1(·), (ii) H(I × ∂Λ) ⊂N \ {y},
By Proposition 6.4, deg(f, Λ, y) is well-defined and does not depend on the particular choice of g. Below are some properties of topological degrees. Theorem 6.7. Suppose fi ∈C0( ¯Λ, N) satisfies fi(∂Λ) ⊂N\{y}, i = 0, 1. If there exists a homotopy H ∈C0(I × ¯Λ, N) such that (i) H(0, ·) = f0(·), H(1, ·) = f1(·), (ii) H(I × ∂Λ) ⊂N \ {y},
then
deg(f0, Λ, y) = deg(f1, Λ, y).
deg(f0, Λ, y) = deg(f1, Λ, y). Theorem 6.8. If deg �f, Λ, y��0, then Λ ∩f −1(y) �∅.
Theorem 6.8. If deg �f, Λ, y��0, then Λ ∩f −1(y) �∅.
Theorem 6.8. If deg �f, Λ, y��0, then Λ ∩f −1(y) �∅.
#   7 Acknowledgement
The first version of this paper considered Lemma 2.4 under the condition that Θi + Θ j ≤ π, Θ j +Θk ≤π, Θk +Θi ≤π. During a workshop on discrete and computational geometry at the Capital Normal University in Beijing, Feng Luo, Xu Xu and Qianghua Luo suggested the current condition. The author is very grateful for their generous sharing. Part of this work was done when the author was visiting Rutgers University, he would like to thank for its hospitality. He also would like to thank NSF of China (No.11601141 and No.11631010) and China Scholarship Council (No. 201706135016) for financial support.
# References
[1] L.V. Ahlfors, Lectures on quasiconformal mappings, AMS 10, New York, 1966. [2] E.M. Andreev, Convex polyhedra in Lobaˇchevskiˇi spaces, Mat.Sb.Nov. 123 (1970), 445478. [3] X. Bao, F. Bonahon, Hyperideal polyhedra in hyperbolic 3-space, Bull. Soc. Math. France. 130 (2002), 457-491. [4] A.I. Bobenko, T. Hoffmann, B.A. Springborn, Minimal surfaces from circle patterns: geometry from combinatorics, Ann. of Math. 164 (2006), 231-264. [5] A.I. Bobenko, B.A. Springborn, Variational principles for circle patterns and Koebe’s theorem, Trans. Amer. Math. Soc. 356 (2004), 659-689.
[6] P. Buser, Geometry and spectra of compact Riemann surfaces, Progress in Mathematics 106, Springer Science and Business media, Boston, 2010. [7] B. Chow, F. Luo, Combinatorial Ricci flows on surfaces, J. Differential Geom. 63 (2003), 97-129. [8] J. Dai, X.D. Gu, F. Luo, Variational principles for discrete surfaces, Advanced Lectures in Mathematics 4, Higher Education Press, Beijing, 2008. [9] Y. Colin de Verdi`ere, Un principe variationnel pour les empilements de cercles, Invent. Math. 104 (1991), 655-669. [10] H. Ge, B. Hua, Z. Zhou, Circle patterns on surfaces of finite topological type, arXiv: 1909.03419 [11] H. Ge, W. Jiang, On the deformation of inversive distance circle packings,II, J. Funct. Anal. 272(9) (2017) 3573-3595. [12] H. Ge, X. Xu, A discrete Ricci flow on surfaces with hyperbolic background geometry, Int. Math. Res. Not. IMRN. 11 (2017), 3510-3527. [13] V. Guillemin, A. Pollack, Differential topology, AMS 370, Englewood Cliffs, 2010. [14] R. Guo, F. Luo, Rigidity of polyhedral surfaces,II, Geom. Topol. 13 (2009), 1265-1312. [15] R. Guo, Local rigidity of inversive distance circle packing, Trans. Amer. Math. Soc. 363 (2011), 4757-4776. [16] M. W Hirsch, Differential topology, GTM 33, Springer-Verlag, New York, 1976. [17] Y. Imayoshi, M. Taniguchi, An introduction to Teichm¨uller spaces, Springer-Verlag, Tokyo, 1992. [18] J. Liu, Z. Zhou, How many cages midscribe an egg, Invent.Math. 203 (2016), 655-673. [19] A. Marden, B. Rodin, On Thurston’s formulation and proof of Andreev’s theorem, LNM 1435, Springer-Verlag, Berlin, 1990. [20] J.W. Milnor, Topology from the differentiable viewpoint, Princeton University Press, Princeton, 1997. [21] I. Rivin, Euclidean structures on simplicial surfaces and hyperbolic volume, Ann. of Math. 139 (1994), 553-580.
[22] I. Rivin, A characterization of ideal polyhedra in hyperbolic 3-space, Ann. of Math. 143 (1996), 51-70. [23] B. Rodin, D. Sullivan, The convergence of circle packings to the Riemann mapping, J. Differential Geom. 26 (1987), 349-360. [24] J.M. Schlenker, Hyperideal circle patterns, Math. Res. Lett. 12 (2005), 85-112. [25] O. Schramm, How to cage an egg, Invent.Math. 107 (1992), 543-560. [26] K. Stephenson, Introduction to circle packing: The theory of discrete analytic functions, Cambridge University Press, Cambridge, 2005. [27] W.P. Thurston, Three-dimensional geometry and topology, Princeton University Notes, Princeton, 1980. [28] W.P. Thurston, The finite Riemann mapping theorem, invited talk, an International Symposium at Purdue University on the occasional of the proof of the Bieberbach Conjecture, 1985. [29] X. Xu, Rigidity of inversive distance circle packings revisited, Adv. Math. 332 (2018), 476509. Ze Zhou, zhouze@hnu.edu.cn
Ze Zhou, zhouze@hnu.edu.cn Institute of Mathematics, Hunan University, Changsha, 410082, P.R. China
