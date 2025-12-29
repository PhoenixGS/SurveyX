# Meta-learning for model-reference data-driven control 
Riccardo Busetto a, Valentina Breschi b, Simone Formentin a
aDipartimento di Elettronica, Bioingegneria e Informazione, Politecnico di Milano, Via Ponzio 34/5, Milano, Italy bDepartment of Electrical Engineering, Eindhoven University of Technology, 5600 MB Eindhoven, The Netherlands
One-shot direct model-reference control design techniques, like the Virtual Reference Feedback Tuning (VRFT) approach, offer time-saving solutions for the calibration of fixed-structure controllers for dynamic systems. Nonetheless, such methods are known to be highly sensitive to the quality of the available data, often requiring long and costly experiments to attain acceptable closed-loop performance. These features might prevent the widespread adoption of such techniques, especially in low-data regimes. In this paper, we argue that the inherent similarity of many industrially relevant systems may come at hand, offering additional information from plants that are similar (yet not equal) to the system one aims to control. Assuming that this supplementary information is available, we propose a novel, direct design approach that leverages the data from similar plants, the knowledge of controllers calibrated on them, and the corresponding closed-loop performance to enhance model-reference control design. More specifically, by constructing the new controller as a combination of the available ones, our approach exploits all the available priors following a meta-learning philosophy, while ensuring non-decreasing performance. An extensive numerical analysis supports our claims, highlighting the effectiveness of the proposed method in achieving performance comparable to iterative approaches, while at the same time retaining the efficiency of one-shot direct data-driven methods like VRFT. eess.SY]  29 Au
Key words: meta-learning, data-driven control, Virtual Reference Feedback Tuning, model-reference control
# 1 Introduction
Real-world control applications often require calibrating parametric controllers with the same structure for systems that are similar, yet not identical, in nature and scope. This is the case, e.g., when considering a platoon of mass-produced cars that navigate in diverse environments or a motor undergoing different industrial cycles. In this paper, we will argue that in such contexts the knowledge of the controllers already tuned for some systems within a certain set and the insights into the achievable closed-loop performance might be valuable assets to expedite and enhance the control design process for a new plant belonging to the same family. The enabling technology to attain this goal will be the use of some similarity measures for the considered plants.
⋆This paper is partially supported by the FAIR (Future Artificial Intelligence Research) project, funded by the NextGenerationEU program within the PNRR-PE-AI scheme (M4C2, Investment 1.3, Line on Artificial Intelligence). Email address: riccardo.busetto@polimi.it (Riccardo Busetto).
Related literature. A similar argument is at the heart of the so-called meta-learning approaches (see [31,23] and references therein), which rely on a set of meta-data comprising all the knowledge acquired from past experience to enhance the learning of a new model or task, by looking at its similarity with the ones already identified or performed. However, existing meta-learning approaches mainly focus on improving the performance of classification [20,18], model fitting (see e.g., [28]), reinforcement learning algorithms ([10,32,34,16]), or, more recently, on reducing the design effort of global optimization tools [5,8,24].
In the systems and control area, strategies exist to exploit similarities for improving model fitting following a federated learning perspective (see e.g., [4,3,9,19] for some examples), whereas only few control design approaches have embraced the meta-learning vision. For instance, the work in [33] deals with the classical problem of identifying a linear model for a linear, time-invariant dynamical system. This work shows that using data collected from a single auxiliary system (similar to the target one) improves the finite sample accuracy of the identified model at the price of introducing a bias dictated by the difference between the auxiliary and the target
systems. Shifting from linear to non-linear model fitting, the work in [21] focuses on modeling a set of systems sharing the same (unknown) dynamics while determining their (possibly different) operational contexts. By casting a bilevel optimization problem to learn these unknowns from data coming from multiple systems, the authors show in simulation that the obtained model can be successfully employed within a model predictive (MPC) scheme to control the motion of a planar fully actuated rotorcraft (PFAR), even when its operational context is time-varying. Instead, the Bayesian approach presented in [2] exploits meta-learning principles to enhance the description of modeling errors for a system that has to undertake multiple tasks. In particular, the authors propose to improve the system’s model by exploiting data collected when it performs different assignments, to use such a model in designing model-based controllers, and prove the effectiveness of this choice on both simulated and real hardware robotic applications by coupling their strategy with a learning-based MPC scheme.
Although leveraging the meta-learning rationale, all the existing approaches still focus on learning a model of the system rather than a controller. This transition is performed in [15] and [22]. In particular, the work in [15] focuses on reconstructing a Linear Quadratic Gaussian (LQG) regulator from closed-loop data. By exploiting the separation principle, the authors show how to effectively leverage input/output data sequences gathered by deploying both the target regulator and other LQG controllers designed with different weighting matrices. Nonetheless, their objective is to reconstruct (and thus imitate) an existing controller rather than calibrating one from scratch. Meanwhile, the strategy presented in [22] leverages ensemble models (constructed from different input/output datasets) to retrieve a parametric adaptive controller that effectively copes with unmodelled disturbances. Through simulation examples involving the motion control of planar fully actuated and underactuated quadrotors subject to wind, the authors show the effectiveness of this control-oriented meta-learning rationale and its superiority over a modeloriented one. Nevertheless, this control design approach requires a preliminary identification phase that can be lengthy and expensive, especially if the number of ensemble models is high. Therefore, none of these methods is model-free and tailored to design a controller directly from data.
Contributions. In this work, we target the above research gap by proposing a novel meta-design approach for calibrating fixed-structure controllers directly from data, without requiring any preliminary identification phase. By focusing on controlling a linear time-invariant system with unknown dynamics, the stepping stone of our method is the well-known Virtual Reference Feedback Tuning (VRFT) approach [7,11]. As for this technique, our first contribution (Contribution C1) is thus to carry out a calibration procedure to tightly match the
behavior of a user-defined reference model, while at the same time incorporating additional meta-data into the design process, so as to ultimately obtain what we will call the meta-controller.
Moreover, we will prove (Contribution C3) that the closed-loop matching error attained with the new controller is bounded, and the bound depends on (i) the performance experienced in the meta-dataset and (ii) the similarity of the other plants and the new controlled system. We exploit these theoretical insights to augment the VRFT loss for meta-design with two additional regularization terms (Contribution C4), thus shaping the calibrated convex combination based on data-driven indicators of similarity and measured closed-loop performance.
Note that our structural choice to consider the convex combination of controllers has never been considered in similar works (see, e.g., [1,17] for approaches exploiting the same structural assumption), as it may lead to stability problems even when all the controllers in the meta-dataset are individually stabilizing the new system. However, by leveraging the seminal work [30], we establish (Contribution C5) theoretical and practical sufficient conditions on the controllers within the metadataset, for their combination to result in a controller that stabilizes the closed loop. We will then translate the above stability constraints into their data-driven counterparts and incorporate them into the direct design procedure.
As a final contribution (Contribution C6), we test the proposed approach in the tuning of a PI (Proportional Integral) controller within a field-oriented scheme for brushless DC motors. The natural differences arising between multiple instances of these motors due to the manufacturing process and their operational conditions once deployed, along with the industrial relevance of containing control calibration time for each new motor
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2b97/2b97c3b2-1972-4159-85e5-04f4f8c97d14.png" style="width: 50%;"></div>
Fig. 1. Scheme of the closed-loop system with the meta– controller C(α). The noise-free tracking error is denoted as eo(t) = r(t) −yo(t), while the dependence on the back-shift operator is omitted for the sake of readability. instance, make this example an ideal benchmark to analyze the impact of the meta-learning approach to datadriven design. By using a relatively small dataset gathered from the motor to be controlled, our results show that the proposed meta-design strategy leads to dramatically improved closed-loop performance with respect to those achieved with a controller tuned with the classical VRFT approach. A performance enhancement is experienced with the proposed method even when compared to more robust (and demanding) iterative approaches. The paper is organized as follows. The objectives of the work are introduced in Section 2, along with the first mathematical formulation of the meta-design problem. Its main properties are then discussed in Section 3. Based on these features, we introduce the data-driven counterpart of the considered problem in Section 4. The effectiveness of the proposed direct, meta-design strategy is finally shown in a simulation case study in Section 4. The paper is ended by some concluding remarks.
Notation. Let N be the set of natural numbers (including zero), R denote the set of real numbers and Rn indicate the set of real column vectors of dimension n. Given a ∈Rn, we denote its transpose as a⊤and k-th component as [a]k, for k = 1, . . . , n. Still considering this vector and an n×n real matrix A ∈Rn×n, we compactly denote the quadratic form a⊤Aa as ∥a∥2 A. Given a set of matrices {Ak ∈Rn×n}N k=1, diag(A1, . . . , AN) ∈RnN×Nn is the block-diagonal matrix having them as its diagonal entries. Given a stochastic process v ∈R, its expected value is E[v] while its variance is denoted as var[v]. The uniform distribution within the interval [a, b] is denoted as U[a,b].
# 2 Setting and goal
Consider a plant G within the class of systems described by the following linear, time-invariant input/output relationship: −
(1)
where u(t) ∈R is the input at time t ∈N, yo(t) is the corresponding noiseless output, and G(q−1) is an unknown, proper rational function in the back-shift operator 1 q−1. Our aim is to design a parametric controller for this system within the class
(2)
� �� � for the closed-loop system to match a desired response yd(t) to a reference r(t) for all t, with θ ∈Rnθ being the set of parameters to be tuned and β(q−1) ∈Rnθ being a vector of prefixed, rational basis function in q−1. The desired behavior is here dictated by a stable, user-defined reference model M, characterized by the relationship:
(3)
where M(q−1) is a proper, rational function in q−1 dif ferent from the unitary gain (i.e., M(q−1) ̸= 1).
Although the true system dynamics is unknown, suppose that we have access to a finite set of input/output data 2 DT = {u(t), y(t)}T t=1 gathered from G, where y(t) is defined as
(4)
and v(t) is the realization of a zero-mean noise at time t. In addition, assume that we have information on N ≥1 plants {Gk}N k=1, whose dynamics is also unknown but nonetheless similar to that of G according to the following definition. Definition 1 (ε-similarity) Two systems G1 and G2 within the class G in (1) are said to be ε-similar if the following property holds:
Specifically, let us suppose to have access to the following data.
(1) A set of controllers Ck = C(q−1; θk) ⊂C(θ), k = 1, . . . , N, which stabilize the corresponding plants Gk in closed-loop and have been tuned with any of the existing data-driven, model-reference strategies to match the reference model M in (3). (2) The input/output pairs Dk T = {u(t), yk(t)}T t=1 used for such a tuning procedure, that share the same inputs of DT while featuring outputs corrupted by a zero-mean noise vk(t), namely
1 q−iu(t) = u(t −i), ∀i ∈Z. 2 Data can be collected in open-loop when G is stable, otherwise they can be gathered from closed-loop experiments, provided a stabilizing (even if poorly performing) controller is available.
These elements constitute what we indicate as the metadataset:
D  { D  D } that, together with DT , represent the set of all the information at our disposal to design the controller for G, namely
D  { D  D } that, together with DT , represent the set of all the information at our disposal to design the controller for G,
(8)
Instead of designing the new controller C(θ) for G from DT only, in this work we propose to leverage all the available information in D to design (in a model-reference fashion) the meta-controller C(α) ⊂C(θ) characterized by:
(9a)
Let Jk define the loss associated with the k-th controller Ck ∈Dmeta N , namely
with
(9c)
namely the controller given by the convex combination of those available within the meta-dataset Dmeta N .
By referring to the closed-loop in Fig. 1, this meta-design problem can be formalized as:
(10a)
where
����� 2 (10e)
��� and F(q−1) is any user-defined weighting filter.
Before shifting our attention to the data-driven counterpart of (10), let us shed light on the performance one
would attain by solving this ideal meta-control problem and, consequently, introduce some changes in the original formulation to enhance the design process. To this end, we rely on the following approximation [30, Section 2] of the loss in (10):
(11)
where Ξ(q−1) = 1−M(q−1). For the sake of simplifying the notation, in the following we do not explicitly show the dependence on the backward-shift operator q−1.
3.1 Non-deteriorating performance
When the unknown system G is not just similar but equal to one of the plants Gk on which Dmeta N is build upon, with k ∈{1, . . . , N}, solving (10) should either enhance the closed-loop matching performance with respect to the ones attained with Ck ∈Dmeta N or, in the worst case, retain the ones achieved with it.
(12)
(13)
Then, we can formalize the first property of the metacontroller as follows.
Then, we can formalize the first property of the metacontroller as follows. Proposition 1 (Non-deteriorating performance) Let us assume that there exists one and only one k ∈{1, . . . , N} such that the unknown plant G satisfies
(14)
and that the controllers corresponding to the other plants are non-canceling, namely
(15)
if and only if [α]i = 0, for all i ̸= k with i = 1, . . . , N. Then, the optimal tuning α⋆in (13) is such that
(16)
Proof According to (12), let us rewrite J(α) in (11) as J(α)≈∥[FΞM −FΞ2CkG]+FΞ2 [Ck−C(α)] G∥2.
(17)
� �� � with ∆Jk(α) being non-negative by definition. Its minimum value ∆Jk(˜α) = 0 is attained when
J(˜α) ≤Jk + ∆Jk(˜α) = Jk.
Since, by definition J(α⋆) ≤J(¯α), this ends the proof. □
3.2 Performance bounds
Toward establishing if and when the choice of designing C(α) (meta-learning) instead of C(θ) (standard datadriven design) can be effective, it is fundamental to analyze the relationship between the closed-loop performance attained with the meta-controller and the actual similarity between G and {Gk}N k=1, along with its link to the closed-loop performance achieved by the controllers in the meta-dataset (7).
With this in mind, let us introduce
(18)
�� �� indicating the performance attained by deploying the controller Ck in feedback with the system Gk, for k = 1, . . . , N. Moreover, let us express the new plant G as a function of the k-th system within the dataset, i.e.,
(19a)
where
(19b)
∥∥ ≤ according to Definition 1 for some (unknown) ε, and ∆Gk = 0 only when G and Gk are equal, for k = 1, . . . , N.
We can now formalize the relationship between J(α) in (10e) and { ˜Jk, ∆Gk}N k=1 as follows.
Proposition 2 (Bound on performance) Let G and {Gk}N k=1 verify (19) for some (unknown) ε. Then, for all α satisfying (10c)-(10d) it holds that:
(20a)
with { ˜Jk}N k=1 defined as in (18) and
(20b)
Proof Consider the approximation in (11) and replace C(α) with its definition in (9), namely
(21)
�� which we can be equivalently recast as
 (22)
�� �� based on (19). Thanks to (10d), it further holds that
where the last equality holds thanks to (10c). Then, the bound in (20a) straightforwardly follows from the definitions in (18) and (20b), with the upper-bound on Sk being an immediate consequence of the Cauchy–Schwartz inequality and the bound in (19b). □
Proposition 2 is a quantitative argument in favor of the (intuitive) fact that α “prioritizing” controllers in
(23)
for i, k = 1, . . . , N and i ̸= k, lead to lower upper bounds on the cost (and potentially lower values of J(α)).
3.3 Closed-loop stability
By solving (10), we have no guarantees that the metacontroller C(α) would lead to a stable closed-loop system. Inspired by [30], we now point out the features that the controllers in the meta-dataset should enjoy to guarantee the stability of the meta-closed-loop, ultimately allowing us to integrate a sufficient condition for closedloop stability in the meta-design problem.
To this end, let us now introduce the following quantity
∆k = M −CkGΞ,
(24)
that depends on the k-th controller within the metadataset, the reference model (3) and the unknown plant G, for k = 1, . . . , N. By relying on this definition, consider the following property for a certain controller Ck in Dmeta N . Assumption 1 Ck ∈Dmeta N is such that:
A.1 ∆k in (24) is stable; A.2 ∃δk ∈(0, 1) so that
(25)
According to [30, Theorem 1], this implies that Ck stabilizes the plant G. We can now formalize the following result. Proposition 3 (Meta-stability condition) The controller C(α) stabilizes the system G if Assumption 1 is satisfied by all Ck ∈Dmeta N , with k = 1, . . . , N. Proof Along the line of [30, Theorem 1], C(α) stabilizes the system G if
P.1 ∆(α) = M −C(α)GΞ is stable; P.2 ∃δ ∈(0, 1) such that δ(α) = ∥∆(α)∥∞≤δ
Therefore, we have to show that these sufficient conditions are verified under our assumptions. To this end, let us exploit the definition of the meta-controller (9) to recast ∆(α) as:
(26)
where the last equality follows from (24). Since {∆k}N k=1 are stable according to Assumption 1 and computing their convex combination does not change their poles, then ∆(α) is also stable, ultimately proving P.1. To show that P.2 holds, let us still rely on the equality in (26). By using the triangle and the Cauchy–Schwartz inequalities, it is straightforward to prove that
where the last equality holds thanks to (10c) and Assumption 1. To complete the proof we have now to show that the upper-bound δ in (27) lays in the interval (0, 1), which can be easily proven by relying once more on Assumption 1. To this end, let us define:
(28)
with ¯δ < 1 and δ > 0 since δk ∈(0, 1) for all k = 1, . . . , N thanks to Assumption 1. Accordingly, the following holds:
(29b)
□
���� also thanks to (10d), and this concludes the proof. □ Remark 1 Assumption 1 may sound like a strong requirement as G is unknown. However, when ε is not too large, e.g., when the systems Gk represent several instances of the same batch production, and Ck are tuned with a suitably stability margin, it is likely that - although the performance may vary for different k’s - the controllers will not destabilize G. Note also that A.1 in Assumption 1 can be satisfied by following the guidelines provided in [30, Section 2] when tuning {Ck}N k=1. Finally, if for any reason we doubt this assumption is not satisfied by one of the controllers in the meta-dataset, the latter can be discarded and C(α) can be constructed based on a reduced meta-dataset Dmeta N−1. This will be shown to be doable in a realistic setting in Remark 4.
To guarantee the verification of P.2 (see again the above proof) we can directly enforce this condition as a con-
(30a)
where δ ∈(0, 1) becomes a tunable parameter, whose choice might lead to a (more or less) conservative tuning of C(α).
3.4 Enhancing (10) with regularization
As highlighted by Proposition 2, the loss J(α) in (10e) is upper-bounded by the convex combination (through α) of the losses { ˜Jk}N k=1 characterizing the matching performance of the controllers within Dmeta N , and the similarity between the new plant G and {Gk}n k=1. Therefore, encouraging the choice of higher coefficients [α]k, with k ∈{1, . . . , N}, for those controllers that are characterized by small losses (and, thus, better matching performance), while being designed to control plants that are more similar to G, would eventually result in improved performance of the meta-controller.
To steer α towards a choice aligned this rationale, we propose to augment the matching cost J(α) in (30) with two regularization terms, shaping α based on experienced performance and plants similarities. Accordingly, the meta-design problem in (30) can be transformed as follows:
Accordingly, the input generated by the meta-controller has to be equal to that comprised in DT . This allows us to cast the data-driven matching loss as
(31a)
(31b) (31c) (31d) (31e)
(31b)
where uL(t) and eL v (t) are obtained by filtering the input and virtual error with L(q−1) defined as
where RJ : RN →R and RS : RN →R are two (possibly different) regularization functions, ˜J = { ˜Jk}N k=1, ∆G = {∆Gk}N k=1, while λJ, λS ≥0 are tunable penalties that modulate the relative importance of the regularization terms with respect to the matching loss.
The upgraded final meta-design problem in (31) (with respect to (10), it incorporate a stability constraint and suitable regularization terms) is not yet applicable to a real-world problem. Indeed, it still depends on the input/output relationship of the controlled system G, the ideal performance ˜J of the controllers in the metadataset and the mismatch ∆G between G and the other plants {Gk}N k=1. All the above details are not known nor directly accessible. Instead, only D in (8) is available to solve the meta-design problem. Hence, in this section, our goal is to translate (31) into its purely direct, datadriven counterpart, thus applicable in a realistic setting.
# 4.1 A data-driven reformulation of J(α)
Before shifting from the loss J(α) in (10e) to its databased counterpart, let us initially replace it with its square:
�� �� which results in a design problem that is easier to handle numerically. We can now follow the footsteps of the VRFT approach [7] to transition toward the data-driven loss. Hence, we define the virtual reference as the set point that would return the measured outputs if fed to the reference model, namely
in turn resulting in the virtual tracking error
(34)
(35)
(36)
with Φu being the spectral density of the input signal in DT . Remark 2 (On the choice of L(q−1)) Since the squared loss in (32) corresponds to the standard objective
 (37)
� � the reasoning carried out in [7, Section 3] for the selection of L(q−1) applies to the considered problem. Remark 3 (Retrieving rv(t)) For the virtual reference to be computed, one needs to invert the reference model. However, this inverse is non-causal every time M(q−1) is a strictly proper rational function. This issue can be overcome by manipulating the reference model as discussed in [12, Proposition 1 and Section 7], in turn resulting in a reduction of the samples available for design. While allowing us to remove the dependence on the plant model, (35) is not yet designed to counteract the impact that noisy data have on meta-design. Once again, we overcome this issue by echoing the VRFT approach and resorting to an instrumental variable scheme [26]. Specifically, we assume to perform an additional experiment on the plant G by feeding it with the same input sequence {u(t)}T t=1 in DT . This allows us to gather a new set of outputs {yIV (t)}T t=1, corrupted by measurement noise uncorrelated with the one affecting the outputs already available in DT . We can now construct the instrument as
(38)
with βmeta(q−1) defined as in (37), and recast the datadriven cost as
(39)
To obtain the data-driven counterpart of the stability constraint in (31e) we rely on the strategy already proposed in [30], which we recall only for the case of stable and minimum-phase G due to its relevance for our numerical example 3 .
Suppose that the data in DT satisfy the assumptions in [30, Section 4]. In this scenario, let us introduce
� (40)
� �� � which allows us to link the quantity whose infinity norm we aim at bounding (see (27)) and the available data.
3 Nonetheless, the extension to unstable and non-minimumphase plants is straightforward following [30] and the approach of this section.
Note that the last approximation is due to the fact that we are neglecting the impact of measurement noise, which is nonetheless legitimate when the measurement noise and the input sequence in DT are uncorrelated (see [30, A4, Section 4]). By relying on (40), δ(α) in (30e) can be approximated as
(41)
��� ��� where ωi = 2πi/(2ℓ+ 1), for i = 0, 1, . . . , ℓ+ 1, and
with j denoting the imaginary unit, and
being the sampled auto-correlation of the input and cross-correlation between the input and es in (40). Note that the approximation in (41) depends on the window length ℓ, which is an additional hyper-parameter of the design problem. The reader is referred to [30] for additional insights on the derivation of this approximation. Remark 4 (Checking Ck ∈Dmeta N ) Apart from being used to approximate δ(α) in (31e), the previous strategy can be used to approximate ∥∆k∥∞in (25), for k = 1, . . . , N. In turn, this allows us to empirically evaluate if the controllers added to the meta-dataset stabilize G, eventually discarding them if they do not satisfy the data-driven stability condition � ∥∆k∥∞≤δk.
Let us then focus on the two regularization terms in (31a). To translate them into their data-driven counterparts, we have to define two data-driven indicators for the experienced performance and the plants’ similarities to replace ˜J and ∆G, respectively.
Under the assumption that the controllers in the metadataset have been deployed and tested in closed-loop, we propose to substitute ˜J with the squared (observed) closed-loop matching error
(42)
where yd(t) is the desired output for a prefixed reference ˜r(t) over T cl time steps, while ycl k (t) is the observed closed-loop output comprised in Dmeta N , for k = 1, . . . , N and t = 1, . . . , T cl. Meanwhile, since we assume that both DT and Dk T comprise the same input sequence, we exploit the squared difference between the measured open-loop outputs as a proxy of ∆G, i.e.,
This choice is in line with our definition of similarity, according to which two plants are similar if the average gain of their difference is limited at all frequencies.
Note that, both ˜Jd k in (42) and Sk in (43) are built from noisy data, whose impact on their statistical means is formalized in the following lemmas 4 . Lemma 1 (Mean features of ˜ Jd k) Given ˜Jd k in (42), assume that the measurement noise vcl k acting on the k-th closed-loop output ycl k is zero-mean. Then:
(44)
with
(45)
Proof By the superimposition principle, the error characterizing (42) can be equivalently rewritten as
Since Ck stabilizes Gk in closed-loop by construction, the auto-regressive process ˜vcl k resulting from this decomposition is weak-sense stationary. Therefore, its mean value is constant over time, and it is equal to zero because of our assumption on vcl k . The result in (44) follows straightforwardly by bringing the term dependent on yd(t)−yo,cl k (t) outside the expectation, since it is deterministic. □ This result allows us to highlight the connection between the selected data-driven performance index and ˜Jk. In particular, the first term on the right-hand-side of (44) can be equivalently rewritten as
This result allows us to highlight the connection between the selected data-driven performance index and ˜Jk. In particular, the first term on the right-hand-side of (44) can be equivalently rewritten as
4 With a slight abuse of notation, stochastic processes will be denoted by dropping their dependence on t.
(46)
�� �� where ˜r compactly denotes {˜r(t)}T cl t=1, and the first element on the right-hand-side of the inequality corresponds to ˜J2 k for F(q−1) = 1. Lemma 2 (Mean features of Sk) Assume that the zero-mean noise sequences acting on the outputs comprised in DT and Dk T are uncorrelated, namely
with v and vk introduced in (4) and (6), respectively. Then:
Accordingly, the normalized expected value of Sk becomes
where the first term is outside the expectation as it is deterministic, while mixed products disappear since v and vk are both assumed to be zero mean. The result in (47) follows by further decomposing the square of the second term on the right-hand-side of the previous equality and by exploiting the lack of correlation between v and vk. □
This lemma allows us to bridge between ∆Gk and the chosen data-based index. Indeed, the first term in (48) can be rewritten as
(49)
thus depending on the similarity between G and Gk (see (19)). At the same time, these results indicate that (as expected) noise impacts on the average values of the proposed (normalized) indexes through its variance. In turn, this might make these indicators poor evaluators of similarities and observed performance when the noise is particularly high. Future work will thus be devoted to refine these indexes toward reducing the influence of noise on the regularization penalties.
Combining all the previous “ingredients”, we can finally write the numerically tractable, data-driven counterpart of the ideal problem in (31) as
(50a)
(50c)
(50e)
where ˜Jd = { ˜Jd k}N k=1 and S = {Sk}N k=1.
In what follows (including the numerical example), we will consider the following as a reasonable choice of the regularizers:
(51a)
with
(51b)
In other words, we use our insights on similarity to promote shrinkage within α in (9) via the 1-norm regularization. This choice is coherent with the idea that controllers designed for systems that are more similar to G are also more likely to be effective on the latter. At the same time, thanks to the performance-oriented Tikhonov regularizer, we steer the elements of α towards similar values if they are associated to controllers that have been proven to perform comparably, while shrinking them whenever the experienced performance are poor. We wish to stress that these are only two possible regularization options, which we will compare to other alternatives in future works.
# 5 Meta-FOC of a brushless DC motor
We now assess the impact of the proposed strategy on a problem of practical relevance, namely the calibration of the PI controller within a Field-Oriented Control (FOC) scheme for the regulation of a brushless DC motor, according to the scheme in [6]. The controller in charge of generating the quadrature axis current u(t) [A] belongs to the following family:
(52)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e3a4/e3a40918-64ee-425c-8931-e630feda5756.png" style="width: 50%;"></div>
Fig. 2. Average magnitude of the frequency response of the family of DC motors (black line) and interval in which the other possible responses may lay (shaded area).
<div style="text-align: center;">Fig. 2. Average magnitude of the frequency response of the family of DC motors (black line) and interval in which the other possible responses may lay (shaded area).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7a48/7a48bf92-5880-4e5f-8cf2-f2d0ca00a5ab.png" style="width: 50%;"></div>
Fig. 3. Current/speeds pairs collected in open-loop from the N meta-motors. The black line indicates their average output, while the shaded area spotlights the outputs’ possible deviations. where T = 0.02 [s] denotes the sampling time. Our goal
Fig. 3. Current/speeds pairs collected in open-loop from the N meta-motors. The black line indicates their average output, while the shaded area spotlights the outputs’ possible deviations. where Ts = 0.02 [s] denotes the sampling time. Our goal is to calibrate θ in (52) for the closed-loop, noiseless motor speed yo(t) [rpm] to match the output of the ideal target behavior dictated by
(53)
In our simulations, we generate data assuming the dynamical structure of the motor to be given by
(54a)
with p1 = 0.9975, κ ∈[1.00, 5.75], and p2 ∈[0, 0.9]. The resulting family of motors has a frequency response with magnitude reported in Fig. 2, and a level of similarity ε = 784.55 (see again (5)). Both G, indicated from now on as the new motor, and the N = 10 ones used to construct the meta-dataset (that we will refer to as metamotors) are uniformly sampled at random from this family, i.e., the parameters characterizing their dynamics are extracted based on the following sampling rules:
(55)
In our tests, we always consider 10 different realizations of the new motor, for the outcome of our analysis not to
be linked to a specific realization of the system’s parameters. We wish to remark that (54), along with the true parameters characterizing the dynamics of each motor, are assumed to be unknown and, thus, not exploited for design purposes.
The data collected from the N = 10 meta-motors are further used to design 10 different controllers with the structure in (52). To this end, here we employ the direct (model-reference) control strategy proposed in [6] 5 . The latter entails a closed-loop calibration experiment, that we have carried out for 3 [s] by considering a step reference with amplitude 1000 [rpm]. Note that, apart from stabilizing the plant each controller has been designed for all these N = 10 data-driven controllers stabilize all the new motors we have extracted and tested. Hence, they are never discarded from the meta-dataset. In learning the meta-controller, we consider a unitary weighting filter F(q−1) (see the ideal cost in (10e)), while we use the regularization terms introduced in Section 4.4.
All results reported hereafter have been obtained by solving the optimization problem in (50) with the CVX package [13,14] on an Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz processor with 16 GB of RAM running MATLAB R2021b.
# 5.1 The benefits of meta-design
5.1 The benefits of meta-design
We initially compare the performance attained by deploying the meta-controller calibrated by solving problem (50) (with penalties fixed at λJ = 30 and λS = 300) and two others, respectively tuned with the iterative method proposed in [6] and the VRFT approach [7]. In solving (50) and using the approach proposed by [6]
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fe61/fe61c38b-c4a1-4342-a80d-c0b282942fc7.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) Controller tuned with the c-VRFT approach</div>
Fig. 4. Comparison of data-driven techniques: set point (black line) and desired response (dashed red line) vs mean (colored line) and standard deviation (shaded area) of the closed-loop responses attained with different controllers.
we neglect the stability constraint (see (50e) and [30, Section 4], respectively), as both methods always result in stable closed-loops. This is not the case with the VRFT approach that, without the stability constraint, results in unstable closed-loop behaviors in 50% of the tested cases. We have thus decided to consider the VRFT scheme equipped with the stability constraint with δ = 0.5 for our comparison to be fairer (a.k.a., not biased by the unstable behaviors).
Remark 5 (Practical choice ℓ) A window of length ℓ= 200 is generally used when computing ˆδ(α), as discussed in Section 4.2. Nonetheless, sometimes this choice leads to a failure of the employed solver [27,29]. In this case, ℓis decreased to 10 and then successively increased up to the maximum value for which the solver is able to retrieve a solution for (50). Note that this operation increases the computational time needed to design the controller.
By looking at the responses reported in Fig. 4, it is clear that the meta-controller and the one tuned with SMGO-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ba9e/ba9e7a13-5cac-434a-b2bb-d8afc7dafb03.png" style="width: 50%;"></div>
Fig. 5. Comparison of data-driven techniques: mismatching error (left panel) and tuning time (right panel). The time required for calibration includes that to carry out all the needed experiments, which are 2 for both the meta-learning and the VRFT approach with stability constraint (c-VRFT), and equal to the number of iterations for SMGO-∆.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fdac/fdacf996-ae35-4546-99c4-6db2a53d7814.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 6. Comparison with the “trivial” meta-controller: set point (black line) and desired response (dashed red line) vs mean (colored line) and standard deviation (shaded area) of the attained closed-loop outputs.</div>
∆lead to similar closed-loop performance, allowing the closed-loop system to mimic the desired behavior. Even if the second controller enables the closed-loop system to attain an average response that is closer to the desired one, this slight improvement in the average matching comes at the price of a considerable increase in tuning time (see Fig. 5). Instead, the VRFT approach returns controllers that result in considerably poorer performance, both in tracking the desired response and the reference signal, most probably due to the limited dimension of the considered dataset DT . These results clearly show the benefits of the proposed meta-learning rationale, which allows for a trade-off between tuning time/effort and performance.
As a final remark for this section, we wish to stress the importance of optimizing the weights of the metacontroller, more than simply using the information coming from the meta-dataset. Let us consider the “trivial”
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a0cc/a0ccd502-bb7f-4107-9907-c04362569a0d.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3c2e/3c2e7657-7033-41fd-9e53-1d8d8c50475f.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/688c/688ca820-557e-4667-9bce-d3dff372088f.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) Elements of α</div>
Fig. 7. Overview of the meta-dataset and outcomes of meta– control design over different experiments: darker circles for the similarity indexes and the values of [α]k indicate the values that occur more frequently.
meta-controller as the one with
The comparison in Fig. 6 shows that such a controller results in worst matching throughout the transient with respect to the optimized one. This result is somehow expected, since the proposed tuning strategy actually leverages all information available in D (see (8)) to calibrate the controller, including insights on the similarities between plants and the performance experienced with the different controllers. In turn, this leads the optimizer to “prefer” some controllers within the metadataset over others, with the least performing ones that are either never or rarely considered in the construction of the controller for the new motor (see Fig. 7).
We now empirically evaluate the capability of C(α) to result in non-deteriorating performance, to test the robustness of the property in Proposition 1 in a non-idealized
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8168/81689397-289e-4425-b79d-ee5560320294.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) λS = 300</div>
<div style="text-align: center;">(b) λS = 3000</div>
<div style="text-align: center;">Fig. 8. Non-deteriorating performance with λJ = 30: 2-norm of the mismatching error in closed-loop.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/caf2/caf24e85-18d5-413f-8652-357db31cb276.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) λS = 300</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/26db/26dbc6a7-60a1-4246-b66c-5e64406cbd5a.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) λS = 3000</div>
<div style="text-align: center;">Fig. 9. Non-deteriorating performance with λJ = 30: 2-norm of the mismatching error in closed-loop for each of the 10 tests picking one of the motors in Dmeta.</div>
setting. To this end, we still consider 10 different new motors, that are in turn equal to one of the meta-motors used to construct Dmeta N .
As shown in Fig. 8 and Fig. 9, the proposed meta-control rationale generally allows us to attain improved or, at least, similar model-reference matching with respect to the one achieved by using the controllers in the metadataset depending on the chosen trade-off between λS and λJ in (50a). Indeed, lower values of λS tend to enhance the overall performance attained by using the meta-controller with respect to that achieved leveraging the controllers within Dmeta (as also confirmed by the closed-loop responses shown in Fig. 10). This improvement comes at the price of having two instances (namely, G1 and G8) for which the meta-controller results in a slight deterioration of performance. Nonetheless, by increasing λS to 3000, the performance of the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d4e8/d4e86781-e502-4085-bd9b-1044f3c0324a.png" style="width: 50%;"></div>
Fig. 10. Non-deteriorating performance with λS = 300 and λJ = 30: and mean (line) and standard deviation (shaded area) of closed-loop responses contained in the meta-dataset [top panel] and attained with the meta-controller [low panel] vs the desired output (dashed red line).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/76eb/76ebc0f9-a6e2-4e5a-ba3b-565c5dffe2ff.png" style="width: 50%;"></div>
meta-controller are indeed non-deteriorating, even if the difference in the closed-loop matching attained with the meta-controller and the ones within Dmeta becomes overall less relevant.
These results thus show that the proposed approach can recognize the meta-motor equal to the new one (thus prioritizing the associated controller), while still generally benefiting from performing controllers tailored to metamotors that are also similar to the new one as long as λS does not excessively dominate over λJ.
# 5.3 Sensitivity analysis to the regularization penalties
Going back to the scenario where the N = 10 new motors are sampled at random, we now evaluate the sensitivity of the attained closed-loop response to the choice
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3808/3808b469-cc35-4bb4-aa73-3175cbc05e9f.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 12. Average 2-norm of the closed-loop mismatching error ϵd [rpm] vs dimension of the meta-dataset and number of SMGO-∆iterations. Dmeta (70) and Dmeta (140) denote the results obtained by running SMGO-∆for 70 and 140 iterations, respectively.</div>
Fig. 12. Average 2-norm of the closed-loop mismatching error ϵd [rpm] vs dimension of the meta-dataset and number of SMGO-∆iterations. Dmeta (70) and Dmeta (140) denote the results obtained by running SMGO-∆for 70 and 140 iterations, respectively.
of λS and λJ in (50), while still neglecting the stability constraint. The results of our analysis are reported in Fig. 11, clearly showing that the meta-control design procedure tends to result in poorer performance when both regularization penalties are low, especially for what concerns λS. This result is somehow expected since low values of λS do not leverage insights on the plant similarities. At the same time, also high values of λS result in a drop in performance. In this case, (50) tends to prioritize controllers based on their similarity only, even when the performance already experienced with them is poor. In turn, this potentially hampers the achievement of the desired closed-loop behavior. Fig. 11 further highlights the importance of retaining lower values of λJ, due to the fact that higher λJ would result in prioritizing performance over similarity. This might be undesired, especially when the most performing controllers in Dmeta N are the ones associated with meta-motors that are more different from the new one. Note that even when λS = λL = 0 the meta-controller attains an average 2-norm of the matching error in closed-loop of about 244.54 [rpm], which is comparable to the one achieved by designing a brand new PI with the VRFT approach, that is equal to 241.35 [rpm].
# 5.4 On the impact of the meta-dataset’s features
We now focus on how the performance attained with the meta-controller is affected by the size N of the metadataset and the “quality” of the included controllers. To this end, we consider 14 meta-datasets of increasing dimension 6 , and two possible values for the maximum number of iterations of SMGO-∆, namely 70 and 140. As expected, Fig. 12 highlights that closed-loop matching generally improves when more iterations of SMGO∆are performed, concurrently with a decrease of the av-
6 The meta-dataset of size N = 10 corresponds to the one used in all other analysis, while Dmeta N−1 ⊂Dmeta N .
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2bac/2bac6260-d330-4924-8cd9-16fcae3c39db.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Matching performance and tuning time</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5293/5293ec8a-481e-4b35-b86f-ff6fcadfa3a6.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) attained closed-loop responses</div>
Fig. 13. Meta-control design with stability guarantees: set point (black line) and desired behavior (red dashed line) vs mean (colored lines) and standard deviation (shaded areas) of the responses obtained with the meta-controllers designed with and without the stability constraint. The former is denoted as c-META. erage value of the index ˜Jd k 7 . Only the results obtained for N = 2 are not consistent with this trend, because the limited dimension of the meta-dataset hampers the overall closed-loop behavior and, thus, the matching performance. Meanwhile, performance tends to improve up to N = 4, then reaching a plateau (on average). This outcome indicates that, at least for the example at hand, a meta-dataset of dimension N = 4 would be sufficiently informative to design a performing meta-controller.
# 5.5 The effect of the stability constraint
We finally incorporate the stability constraint into the design problem, solving (50) for δ = 0.5, as employed in c-VRFT. The window length ℓneeded as for Section 4.2 is once again chosen as detailed in Remark 5.
Since the introduction of the stability constraint has not resulted in changes in performance (but only in an increased design time) with the level of noise considered in the previous analysis, we now increase the standard deviation of the noise corrupting the outputs in DT to 40 [rpm] (SNR = 10.61 [dB]), while keeping the metadataset of size N = 10 unchanged with respect to the
one exploited in the other analysis. The results we obtained are reported in Fig. 13. Clearly, they show that the introduction of the stability constraint in this more challenging setting can be of help in effectively coping with the noise acting on the data. Indeed, we observe a significant reduction in instances where matching performance get further away from the average. This benefit comes at the price of an increased design time (see Fig. 13(a)), whose considerable variance is linked to the procedure exploited to adjust the window length ℓevery time the solver fails.
# 6 Conclusions
In this work, we have employed for the first time a metalearning rationale to enhance both the effectiveness and efficiency of direct, data-driven model reference control design. Like humans gain knowledge from past experiences, we propose to leverage controllers already calibrated for similar systems and data-based insights on their experienced performance and similarities to formulate a novel, meta-design problem. The numerical study carried out to calibrate a PI controller for a brushless DC motor has highlighted the potential of the method to enhance closed-loop performance when a reduced amount of data is collected, while at the same time resulting in a reduced tuning time.
Future work will be devoted to setting the ground for a theoretical framework to analyze the informativity of the meta-dataset. This development would be crucial to extend the proposed approach to other (possibly iterative) data-based control techniques, since it would allow one to limit the dimension of the meta-dataset, by incorporating only the most informative incoming data in it.
# References
