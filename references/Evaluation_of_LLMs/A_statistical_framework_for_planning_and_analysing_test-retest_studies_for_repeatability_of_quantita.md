# A statistical framework for planning and analysing test-retest studies for repeatability of quantitative biomarker
# A statistical framework for planning and analysing test-retest studies for repeatability of quantitative biomarker measurements
measurements
Moritz Fabian Danzer1,*, Maria Eveslage1, Dennis G¨orlich1, and Benjamin Noto1, 2, 3, 4
1Institute of Biostatistics and Clinical Research, University of M¨unster, M¨unster, 48149, Germany 2Clinic for Radiology, University Hospital M¨unster, M¨unster, 48149, Germany 3Department of Nuclear Medicine, University Hospital M¨unster, M¨unster, 48149, Germany 4West German Cancer Centre (WTZ) Essen-M¨unster – M¨unster site, University Hospital M¨unster, M¨unster, 48149, Germany *moritzfabian.danzer@ukmuenster.de
# Abstract
There is an increasing number of potential biomarkers that could allow for early assessment of treatment response or disease progression. However, measurements of quantitative biomarkers are subject to random variability. Hence, differences of a biomarker in longitudinal measurements do not necessarily represent real change but might be caused by this random measurement variability. Before utilizing a quantitative biomarker in longitudinal studies, it is therefore essential to assess the measurement repeatability. Measurement repeatability obtained from test-retest studies can be quantified by the repeatability coefficient (RC), which is then used in the subsequent longitudinal study to determine if a measured difference represents real change or is within the range of expected random measurement variability. The quality of the point estimate of RC therefore directly governs the assessment quality of the longitudinal study. RC estimation accuracy depends on the case number in the test-retest study, but despite its pivotal role, no comprehensive framework for sample size calculation of test-retest studies exists. To address this issue, we have established such a framework, which allows for flexible sample size calculation of test-retest studies, based upon newly introduced criteria concerning assessment quality in the longitudinal study. This also permits retrospective assessment of prior test-retest studies.
# 1 Introduction
A biomarker is a characteristic objectively measured and evaluated as an indicator of normal biological processes, pathogenic processes, or response to a therapeutic intervention [5]. Biomarkers used as indicators of response to a therapeutic intervention, or disease progression, are called treatment response biomarkers. One prime, established treatment response biomarker is lesion size change in cross-sectional imaging. For clinical trials concerning solid tumors, the measurement of lesion size is formalized in the so-called Response Evaluation Criteria in Solid Tumors (RECIST) [11], that categorize treatment response. With the rapid advancement in medical sciences, there is an increasing number of new potential treatment response biomarkers that could possibly allow for early and objective assessment of treatment response or disease progression in clinical trials and clinical practice [17].
However, using a biomarker in practice requires some basic research into the reliability of its measurement. In addition to a fixed systematic measurement error (bias), which can be investigated by comparing measurements with a known target value (e.g. phantom studies), it is important to take into account that measurements of quantitative biomarkers are subject to random variability. Hence, changes in a biomarker in longitudinal measurements made under the same conditions do not necessarily represent real change but might be caused by exactly this random measurement variability. Before testing or even utilizing a quantitative biomarker in longitudinal studies, it is therefore of principal importance to assess the measurement repeatability [27]. The repeatability of measurement is determined by test-retest studies, which then are also referred to as repeatability studies. In such studies, replicate measurements are made on a sample of subjects under conditions that are as constant as possible [2]. Measurement repeatability can be quantified by the within-subject standard deviation (wSD). Using wSD, the repeatability coefficient (RC) can be calculated [6,24,27]. RC is then used in the longitudinal study to determine if a difference in the biomarker represents presumed real change or is within the range of random measurement variability. It is defined in such a way that a desired specificity to detect changes – usually 95% – is targeted. The wSD and the RC, as determined by the test-retest study, are point estimates, and hence suffer from random error. As we will show, the targeted specificity is therefore generally not achieved in practice. Following standard statistical results, the more subjects and the more repeated measurements are included in the test-retest study, the more reliable the estimates of wSD and RC will be. Accordingly, the probability of a relevant deviation of the actually achieved value from the targeted specificity will decrease. The quality of assessments in the longitudinal study and consequently the validity of its results is directly governed by the precision of the estimates of wSD and RC. Of course, exact knowledge of measurement repeatability is not only crucial for biomarkers. For example, excellent measurement repeatability of scales and other laboratory instruments is mandatory. The reliability of a scale can be checked using weights with a known mass and it is possible to perform many repeated measurements. In contrast, many biomarkers are measured in-vivo, rendering attainment of large sample sizes difficult. Also, it might be necessary from an ethical point of view to keep sample sizes as low as possible, since the measurement in question might be inconvenient, invasive, or even harmful for the patient or the healthy test person. For example, a biomarker might be derived from computed tomography, which involves ionizing radiation. Yet, if the sample size in the test-retest study is small, there is a high chance of obtaining suboptimal estimates of RC with associated detrimental effects on sensitivity and specificity in the longitudinal study. In what follows, we will focus on such and related issues concerning repeatability. Before doing so, note that, related to but different from repeatability is reproducibility. While repeatability represents the measurement precision under constant conditions, i.e, same measurement procedure, same operators, same measuring system, etc., reproducibility is, in contrast, measurement precision under differing conditions as various operators, measuring systems, etc. [16]. Statistical literature concerning requirements for test-retest studies is scarce. One notable study investigating sample size requirements is by Obuchowski and Bullen [24]. In their work, Obuchowski and Bullen conducted a simulation study to investigate the relation between the sample size in the test-retest study and the specificity achieved in a following longitudinal study. The authors give a blanket recommendation for sample size of test-retest studies based on their results from a fixed set of simulation parameters. Our goal is to expand upon the results of Obuchowski and Bullen [24] in several areas. First, we want to introduce new quality criteria for the planning of test-retest studies. Furthermore, we will expand the considerations to include sensitivity, which has not been investigated in the literature so far. Finally, we aim to provide analytical solutions. In contrast to simulation studies, this allows for flexible calculation of sample size requirements and also the retrospective assessment of test-retest studies, as we will show. In doing so, we establish a comprehensive framework in which the notions introduced above are precisely defined. In what follows, we will introduce the model used for our framework and study the aspects of specificity and sensitivity in separate sections. Afterwards, we demonstrate the application of our
# 2 Definitions
One possible approach to distinguish true change from random variation in the longitudinal study is to estimate measurement variability in a test-retest study. For this purpose, n patients are measured m times within a short period of time, in which their true value presumably does not change. For our considerations we assume independent subjects, e.g. measurement of one target per patient. In addition, independent replicate measurements are necessary, i.e. measurements on a subject need to be made independent of the knowledge of its previous value(s) [8]. Consequently, we establish the following model for the j-th measurement of the i-th patient Yij of the test-retest study:
where µi is the true value for the i-th patient and εij is the random error. We assume the random errors to be independent and normally distributed with mean 0 and variance w2 SD [24]. In particular, it follows that Yij ∼N(µi, w2 SD) for any i ∈{1, . . . , n} and j ∈{1, . . . , m} . This model is appropriate when true replicates are studied and a learning effect can be ruled out. As we are only addressing measurement repeatability, a fixed bias does not need to be considered since it cancels out. We also assume that measurement error is independent from the magnitude of µi. From this data, we can estimate the within-patient standard deviation wSD [6] by
� � where ¯Yi· := 1/m �m j=1 Yij denotes the mean value of the measurements of patient i. Following Cochran’s theorem [10], the distribution of this entity is given by
 � According to standard asymptotic theory, the following central limit theorem holds for �wSD:
where Z is a standard normally distributed random variable. If the number of repeated measurements differs between subjects, i.e. the i-th subject is measured mi times, the value n(m−1) needs to be replaced by �n i=1(mi −1) in all formulas. For the sake of simplicity, we restrict ourselves to the simple case of an equal number of repetitions m per subject. In order to assess changes in the measurements of a single patient in the subsequent longitudinal study, the repeatability coefficient (RC) is computed [6]. It indicates the range in which two repeated measurements are expected to fall with a certain probability. In what follows, we restrict ourselves to the assessment of changes in both directions. We want to keep our decision rules flexible, i.e. we establish a target specificity psp ∈(0, 1) which shall be reached for patients with no change in their true biomarker value. Hence RC is a function of psp and is given by
In most literature the RC is only considered for a fixed targeted specificity of 95%, i.e. RC(0.95) [26,27]. In practice, wSD is unknown and hence replaced by its consistent estimator �wSD to obtain the estimated repeatability coefficient ˆ RC(psp) := Φ−1(1 −(1 −psp)/2) · √ 2 · �wSD. (6)
(2)
(3)
(5)
(6)
This quantity can then be applied as cutpoint in the longitudinal study to determine whether there has been change between two consecutive measurements Ypre and Ypost. Here, we also assume, that the measured values have independent errors, but the true levels µpre and µpost might actually be different, i.e. we have Ypre = µpre + εpre and Ypost = µpost + εpost with εpre and εpost being independent and normally distributed with mean 0 and variance w2 SD. In case the true values have not changed, i.e. µpre = µpost, the difference Ypost −Ypre is normally distributed with mean 0 and variance 2w2 SD. Hence, with a probability of psp, we have Ypost−Ypre ∈ [−RC(psp), RC(psp)]. The rule to decide whether there is a change for a patient with the two measured values Ypre and Ypost should thus be whether their difference lies outside or inside the interval [−RC(psp), RC(psp)]. As the bounds are unknown in practice, this decision rule is replaced by the decision rule based on the estimated interval [−ˆ RC(psp), ˆ RC(psp)]. Consequently, the targeted specificity psp will never be exactly met. This applies analogously to considerations for the sensitivity of this procedure.
# 3 Effective specificity as a criterion for sample size estim tion
# 3 Effective specificity as a criterion for sample size estima-
Our goal is to quantify the uncertainty introduced by the replacement of wSD by its estimator �wSD. As mentioned, the targeted specificity (psp) is not met in practice. To assess this problem, we introduce the effective specificity Pesp which is the specificity actually achieved if a realisation of the estimate �wSD is plugged in. Hence, Pesp is a random quantity as it depends on the value of �wSD. We use a capital letter to emphasise that it is indeed a random variable. It can be implicitly defined via
Although this quantity is unknown in practice, we can nevertheless analyse its distribution. Firstly, we can compute the expected value E[Pesp] and the bias, i.e. the difference E[Pesp] −psp. This is also the quantity targeted by Obuchowski and Bullen [24]. Their quality criterion requires |E[Pesp] −psp| to be smaller than 0.01, i.e. they want the mean effective specificity to deviate less than 1 percentage point from the target specificity, which they set to 95%. But what is even more important, from our point of view, is that we can compute quantiles of the distribution of Pesp which will enable us to establish quality guarantees on the effective specificity of the longitudinal studies based on the design parameters n and m of the test-retest study.
According to (7), Pesp is given by
 � The function RC can be inverted as it is a continuous, monotonically increasing function on (0, 1). The expectation of this random quantity can be computed exactly using (3) or approximately using the central limit theorem (4), according to which the distribution of �wSD/wSD can be approximated with a normal distribution with expectation 1 and variance 1/(2n(m −1)). Hence, we get
=1 −2 · � 1 − �∞ 0 Φ � Φ−1 � 1 −1 −psp 2 � · w � fχ2 n(m−1)(n(m −1)w2) 2wn(m −1) dw � ≈1 −2 · � 1 − �∞ −∞ Φ � Φ−1 � 1 −1 −psp 2 � · w �� n(m −1) π exp � −n(m −1)(w −1)2� dw
(7)
(8)
(9)
(10)
(11)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/107b/107b1699-bc62-4fc5-8dd0-a1513ff11079.png" style="width: 50%;"></div>
<div style="text-align: center;">n in test−retest study</div>
Figure 1: A) Expected value of the effective specificity (E[Pesp]) as a function of n in the testretest study for a target specificity psp of 95% and m = 2. Already for a small case number of 10 (blue dot) the expected value is comparably high. For a case number of 30 (green dot) the bias (E[Pesp] −psp) is below 1 percentage point and does not change substantially with an increase of the case number to 60 (red dot). B) However, while E[Pesp] is already relatively high for a case number of 10, the tails of the corresponding PDF (blue area) are prominent, resulting in a high chance of obtaining a low Pesp in practice. The green and red area represent the PDF of the effective specificity for n = 30 and n = 60, respectively.
where fχ2 n(m−1) denotes the probability density function (PDF) of a χ2-distributed random variable with n(m −1) degrees of freedom. By numerical evaluation of the terms in (10) and (11), the bias can be computed.
# Quantiles of the distribution of Pesp
We need to be aware that even if E[Pesp] is close to psp, i.e. the bias is low, the probability for a substantial deviation of the actually realized specificity from the targeted specificity might be large (Figure 1). Therefore, we want to know with which confidence pconf we can say that the effective specificity is larger than some lower bound pesp,lb. This is expressed by the formula
We want to introduce a new quality criterion based on this concept. The quantity pconf is a function of pesp,lb and of course also depends on psp, n and m. For notational
(12)
convenience, however, we omit those arguments. After some calculations, one obtains
  � �   where Fχ2 n(m−1) denotes the cumulative distribution function of a χ2-distributed random variable with n(m −1) degrees of freedom. This formulas can now be used in different ways. In the above form, one can determine the confidence with which the effective specificity exceeds a fixed bound pesp,lb with given design parameters n and m of the test-retest study. Analogous considerations can be made for upper bounds by computing the probability of the complementary event. In the planning stage of the test-retest study it could be beneficial to choose the sample size n in such a way that a desired lower bound pesp,lb is achieved with a prespecified confidence pconf. To this end, the asymptotic formula (14) can be solved explicitly for n:
   � �    In our application example we will apply these formulas in the planning stage of a hypothetical test-retest study. If one wants to identify the worst possible cases for given n and m, one could compute the lower bound of the effective specificity which is reached with confidence pconf:
  �   Accordingly, in (1 −pconf) · 100% of all cases, the effective specificity will be even lower than the obtained pesp,lb. From our point of view, the probability of exceeding a lower bound pesp,lb is a valid criterion for evaluating the quality of assessment in a longitudinal study. Different from the expected value of Pesp which has been previously proposed as a quality criterion [24], our criterion considers the tails of the distribution of Pesp. This allows to bound the probability of strongly deviating from the desired specificity.
# 4 Consideration of effective sensitivity
Concerning the sensitivity, i.e. the ability to detect real change between two measurements of one patient in the longitudinal study, we can make similar considerations. Before coming back to
(13)
(14)
(15)
(16)
(17)
(18)
the problem of the uncertainty caused from the estimation of wSD, we first assume, that wSD and hence also RC(psp) is known. Of course, the sensitivity strongly depends on the difference between µpre and µpost. Also, such differences are more difficult to detect if wSD is large and a large target specificity is chosen. To be more precise, the sensitivity pse to detect a difference can be written as a function of µ∆:= µpost −µpre, wSD and the chosen specificity psp. It is given by
In this form, the function can also be seen as a function of the effect size δ := µ∆/wSD, i.e
this form, the function can also be seen as a function of the effect size δ := µ∆/wSD, i.e.
In this form, the function can also be seen as a function of the effect size δ := µ∆/wSD, i.e. p(δ)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/24af/24afebcd-b557-4b2a-b1fd-1cca0ad0d187.png" style="width: 50%;"></div>
As wSD is unknown and needs to be estimated by �wSD which will then be plugged in to compute ˆ RC(psp), the sensitivity computed in (20) will not be reached. Analogously to our considerations for the specificity, we introduce the effective sensitivity Pese which is the sensitivity which is actually achieved if a realisation of the estimate �wSD is plugged in. Of course, it is also a random variable and does depend again on µ∆, wSD and psp. It can be defined by the equation
� With this expression and the exact distribution of �wSD given as in (3) resp. the approximation of the distribution of � wSD wSD by a normal distribution from (4) we can now quantify the bias caused by the replacement of wSD by �wSD and compute quantiles of the distribution of Pese which will enable us to also give quality guarantees on the effective sensitivity. Unlike our considerations for the specificity, these values will also depend from the actual wSD and the difference µ∆of the longitudinal study and hence will be regarded as functions of those.
(19)
(20)
(21)
To compute the bias in dependence from psp, µ∆and wSD, we can take the expectation of the righ hand side of (21) and use the exact distribution (3) and the central limit theorem (4) to obtai the result
� � Please note that this can essentially be seen as a function of δ. Following (21) the bias of the effective sensitivity can be considered as a function of δ for any given psp, i.e. E[Pese(psp, δ)] −pse(psp, δ).
Quantiles of the distribution of Pese
For the most accurate examination of the distribution of Pese we would need to consider both events
However, this leads to expressions that are difficult to handle analytically. Actually, the two probabilities
However, this leads to expressions that are difficult to handle ana probabilities
P[Ypost −Ypre > ˆ RC(psp)| �wSD] and P[Ypost −Ypre < −ˆ RC(psp)| �wSD]
 � sum up to the effective sensitivity. However, in the presence of an effect, one of them will be much larger than the other. In the case δ > 0, the probability in (25) is larger than that from (26) which is bounded from above by 0.025 and quickly converges to 0 as δ increases. To enable the derivation of analytical formulas, we will therefore restrict ourselves to the consideration of δ > 0 and the event (23). It is nevertheless possible to circumvent this simplification by numerical inversion of the relationship given in (21). But here, we will approximate
� In analogy to the previous section we can provide confidence levels pconf which indicate the prob ability that the effective sensitivity for some effect δ exceeds the lower bound pese,lb:
(22)
(23) (24)
(25) (26)
(27) (28)
(29)
Of course, such considerations only make sense if pse > pese,lb for the chosen effect size δ. As above, analogous considerations can be made for upper bounds by computing the probability of the complementary event. While (29) allows to compute the confidence of reaching a certain lower bound of the sensitivity for an effect δ, this formula may also be transformed to be used in the planning stage of the test-retest study. If one wants to achieve a fixed confidence with which the effective sensitivity for an effect size δ exceeds some lower bound, one can use the exact results from above or the approximations made thereafter to determine the sample size n of the test-retest study in which each patient is measured m times. It shall be chosen such that
Analogous to the preceding section, we can use these results in the planning stage of a test-retest study, as we will demonstrate in the following application example. Even if a study is planned based on considerations of the specificity, the formulas allow to assess the distribution of the effective sensitivity for any given effect size of interest. Calculations for δ < 0 follow analogously to the considerations for δ > 0.
# 5 Application example
To illustrate our considerations, we will discuss a hypothetical application for early treatment response assessment in recurrent or metastatic nasopharyngeal carcinoma. While some patients with recurrent nasopharyngeal carcinoma show response or stable disease to systemic treatment, many patients will have progressive disease, which is invariably lethal [13,19]. Nevertheless, futile treatments should be avoided due to associated toxicity [9, 13]. To suspend futile treatment as soon as possible an imaging biomarker is desirable which accurately classifies treatment response earlier than change in morphologic lesion size, the current standard. A promising biomarker in this context is diffusion weighted magnetic resonance imaging (DWI) [18]. DWI depends on the differences in the movement of water molecules based on Brownian motion, which can be quantified by the apparent diffusion coefficient (ADC). An exemplary measurement of ADC is shown in Figure 3. Change in ADC has shown promise as an early treatment response marker in various tumors, including nasopharyngeal carcinoma [18,28–30].
# Prospective planning of test-retest studies
As laid out above, before conducting a longitudinal study in which a biomarker is applied to assess treatment response, a test-retest study should be conducted to assess repeatability. In our example, we will set psp to 95% and m = 2, as these are the usual values in the literature. We imagine the researcher would want to obtain a specificity of at least 90% (pesp,lb) with 95% certainty (pconf) in the longitudinal study. What sample size (n) is necessary in the test-retest study? This question
(30)
(31)
(32)
(33)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/50d0/50d0a7cd-6265-49af-a8bf-c5951649048f.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Example case of a tumor in the right nose showing restricted diffusion (A) with a mean ADC of 610 · 10−6 mm2/s. (B) Region of interest outlined in yellow.</div>
can be answered using the asymptotic formula (15):
This can also be concluded from Figure 4 A. Numerical solution of the exact formula (16) yields a sample size of 54. Resulting sample sizes for other values of pesp,lb can be taken from Figure 4 B. The resulting scenario in terms of the distribution of the relative error in the estimation of �wSD and its effect on Pesp is displayed in Figure 5 A.
Analogous considerations can be made for the effective sensitivity. We consider the sensitivity for an underlying true effect size of δ = 4 in a study with psp = 0.95 and m = 2. According to formula (20), a sensitivity of 80.74% was achieved if wSD was a known quantity. However, this will not be met in practice. What is the minimum sample size (n) of the test-retest study such that we can be 95% (pconf) sure to achieve at least a sensitivity of 75% (pese,lb) for that effect size? This question can be answered using the approximate formula (33).
Accordingly, a sample size of 139 patients in the test-retest study would be recommended to achiev the set targets. Using the exact formula (17) or the asymptotic formula (18), an effective specificity of at least 92.25% resp. 92.27% is reached with a certainty of 95%, in this scenario. This is als depicted by Figure 5 B.
(34)
(35)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c15b/c15bbf0c-5a7b-4f1c-b551-f27cb3882bcd.png" style="width: 50%;"></div>
# Retrospective assessment of test-retest studies
It is not always necessary to conduct a preceding test-retest study when planning a longitudinal study. The RC used in the longitudinal study might be adopted from already published testretest studies. If one intends to use the point estimate of the RC obtained in a previous study, it is advisable to retrospectively assess the resulting distribution of the effective specificity and sensitivity. This allows to evaluate the impact of the sample size of the used test-retest study on quality criteria of the longitudinal study, especially the probability of exceeding a given pesp,lb. Common sample sizes in test-retest studies are around 10 and 20 [3,4,12,15,21,22]. If the point estimator of RC(0.95) resulting from a test-retest study with a sample size of 10 and two repeated measurements is used, the distribution of the effective specificity will have prominent tails as illustrated in Figure 1. According to (17), the lower bound of the effective specificity obtained with 95% confidence is 0.7814 and 0.8512 for a sample size of 10 and 20, respectively, which might be insufficient (Figure 4). Note that for the recommendation by Obuchowski and Bullen [24] of a sample size of 35 for test-retest studies with m = 2 the probability of achieving an effective specificity below 94% is 39.74%. Such considerations are also possible for the effective sensitivity.
# 6 Discussion
We have established a comprehensive framework for planning of test-retest studies concerning repeatability. It enables flexible calculation of sample size requirements and retrospective assessment of such studies with regard to different quality criteria. To better discuss planning of test-retest studies we have introduced the notions of effective specificity (Pesp) and effective sensitivity (Pese), allowing for clearer differentiation of the targeted specificity psp and sensitivity pse from the values actually achieved in the longitudinal study. Both Pesp and Pese are random quantities and their actual values are unknown in practical application.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/28e1/28e1d547-340d-4318-aa42-08daa5b4778c.png" style="width: 50%;"></div>
Figure 5: Visualization of the application example. The x-axis denotes the relative error between �wSD and wSD. The solid black line represents the asymptotic PDF of the relative error. Note the near identity to the dashed red curve, representing the PDF of the exact χ2 distribution. The violet line shows the effective specificity. Analogously, the blue line shows the effective sensitivity for an underlying effect size of δ = 4. A) For n = 53 and m = 2: The area shaded in light gray represents 95% of the area under the normal curve. I.e. there is a 95% chance of obtaining a �wSD from the test-retest study that will result in an effective specificity of greater than 90%. B) For n = 139 and m = 2: The area shaded in light gray represents 95% of the area under the normal curve. I.e. there is a 95% chance of obtaining a �wSD from the test-retest study that will result in a specificity of greater than 75%. In this case, an effective specificity of 92.27% will be reached with a certainty of 95%.
However, we can determine their distribution and thus can compute different characteristics which properly reflect the uncertainty caused by the estimation process. Expanding on the work of Obuchowski and Bullen [24], we have introduced a new quality criterion for sample size calculation of test-retest studies. In their work, Obuchowski and Bullen [24] demand that the mean effective specificity (E[Pesp]) deviates at most by 0.01 from the fixed targeted specificity (psp) of 0.95. However, using the mean effective specificity as sole quality criterion has limitations, since the whole distribution of the effective specificity is not properly taken into account. As illustrated in Figure 1, there is a high probability that the actually achieved effective specificity deviates strongly from its target even if the mean effective specificity may be close to the targeted specificity. Therefore, we propose a quality criterion for sample size calculations based on the probability that the effective specificity exceeds a chosen lower bound, taking into account the tails of the distribution of Pesp. In contrast to previous works we expand our consideration also to issues of sensitivity. Here, of course, it must also be taken into account that the sensitivity depends on the underlying effect size. Nevertheless, we can determine the distribution of the effective sensitivity for any effect size and provide analogous sample size formulas as for the specificity. Finally, our study is the first to provide analytical rather than simulation results. This provides greater flexibility as the targeted specificity psp and number of repeated measurements m may be chosen freely. Hence, it allows the readers to avoid conducting time-consuming simulation studies themselves. While our formulas enable flexible calculations for all scenarios, for convenience of the reader we also provide a table with sample sizes for some exemplary scenarios in Figure 4.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8606/860648fc-9fe6-4db9-8116-1d3249046795.png" style="width: 50%;"></div>
Sample sizes resulting from other choices of the parameters pconf, pesp,lb, psp and m can be found in Supplementary Tables S1-S4. Our study has some limitations. The field of application is restricted to test-retest studies in which true replicates of measurements are possible, for example in quantitative imaging markers. Our considerations are not valid if the measurement process itself results in a change of the measurand (learning / practice effect) as has been described for some psychological assessments [14,20]. Our standard model (1) assumes independent and identically normally distributed errors. It is therefore advisable to examine whether there is a relationship between the within-subject variation and the level of the measured value before applying our approach [23]. If the variability of the measurement error increases with the magnitude of the measured value, a log transformation might resolve the issue [1, 7, 23]. Beyond that, non-normally distributed error terms are not covered so far. We also have not specifically considered the scenario of clustered data, e.g. measuring multiple lesions per subject. However, if a hierarchical model structure with independent errors can be assumed, this does not pose a restriction to application of our approach. It should be noted that exact solutions based on the χ2 distribution for all our considerations are available. In some cases, when an analytic solution is not possible, these exact solutions require the application of numerical methods. In order to give completely analytic solutions, some of our formulas rely on asymptotic results and approximations. The differences between exact and approximate results are most severe for small sample sizes and small effect sizes. Applying both exact and approximate formulas in our application example, it can be seen that these differences are negligible in practically relevant scenarios. Implementations of exact and approximate solutions can be found in our supplementary R code [25]. So far, our considerations are limited to repeatability, i.e. assuming same measurement conditions for the repeated measurements. However, for real world application of biomarkers, consideration of reproducibility is also important since longitudinal measurements are often performed under different measuring conditions, e.g. varying readers or scanners. Therefore, our model should be perspectively enhanced to include aspects of reproducibility such as a fixed bias as e.g. in some models considered by Obuchowski and Bullen [24]. Nevertheless, since repeatability limits reproducibility, a good knowledge of the former is useful in order to interpret reproducibility studies properly [8]. Test-retest studies of repeatability should be well planned to guarantee for a sufficient quality of dependent longitudinal studies. Our framework allows the derivation of analytical solutions for quality criteria that can be used to assess implications of the test-retest study design on subsequent longitudinal studies.
# Acknowledgements
B.N. was funded as a clinician scientist by the Medical Faculty, University of M¨unster, Germany. There was no dedicated funding for this study.
# Additional information
Data availability No datasets were generated or analysed during the current study. Implementations of exact and approximate formulas can be found in the Supplementary R Code. Additionally, we provide sample sizes based on formula (16) in Supplementary Tables S1-S4 generated using our R code. Competing interests The authors declare that they have no conflict of interest.
# References
# Supplementary Material
In the following tables, we want to give sample sizes based on formula (16) in our main manuscript for different choices of parameters pconf, pesp,lb, psp and m. For further constellations that cannot be found in the following tables, we would like to refer to the sample size function provided in our Supplementary R Code. Supplementary Table S1: Sample sizes for different constellations of pconf, pesp,lb and psp for m = 2
In the following tables, we want to give sample sizes based on formula (16) in our main manuscrip for different choices of parameters pconf, pesp,lb, psp and m. For further constellations that canno be found in the following tables, we would like to refer to the sample size function provided in ou Supplementary R Code.
m=2
pconf
pesp,lb
psp
0.800
0.900
0.925
0.950
0.975
0.990
0.800
0.700
13
4
4
3
2
2
0.800
10
7
5
3
3
0.900
68
17
7
4
0.925
48
11
6
0.950
27
9
0.975
25
0.900
0.700
25
7
6
5
4
3
0.800
19
12
8
6
4
0.900
147
34
13
8
0.925
102
22
10
0.950
55
16
0.975
52
0.925
0.700
30
9
7
6
4
4
0.800
23
15
10
7
5
0.900
183
42
16
9
0.925
127
26
12
0.950
69
20
0.975
64
0.950
0.700
38
11
8
7
5
4
0.800
29
18
12
8
6
0.900
236
54
20
11
0.925
164
33
15
0.950
88
25
0.975
82
0.975
0.700
53
14
11
9
7
5
0.800
40
25
16
11
8
0.900
332
75
27
15
0.925
229
46
20
0.950
122
34
0.975
114
0.990
0.700
73
19
15
12
9
7
0.800
54
34
22
14
10
0.900
463
103
37
20
0.925
320
63
28
0.950
170
46
0.975
159
m=3
pconf
pesp,lb
psp
0.800
0.900
0.925
0.950
0.975
0.990
0.800
0.700
7
2
2
2
1
1
0.800
5
4
3
2
2
0.900
34
9
4
2
0.925
24
6
3
0.950
14
5
0.975
13
0.900
0.700
13
4
3
3
2
2
0.800
10
6
4
3
2
0.900
74
17
7
4
0.925
51
11
5
0.950
28
8
0.975
26
0.925
0.700
15
5
4
3
2
2
0.800
12
8
5
4
3
0.900
92
21
8
5
0.925
64
13
6
0.950
35
10
0.975
32
0.950
0.700
19
6
4
4
3
2
0.800
15
9
6
4
3
0.900
118
27
10
6
0.925
82
17
8
0.950
44
13
0.975
41
0.975
0.700
27
7
6
5
4
3
0.800
20
13
8
6
4
0.900
166
38
14
8
0.925
115
23
10
0.950
61
17
0.975
57
0.990
0.700
37
10
8
6
5
4
0.800
27
17
11
7
5
0.900
232
52
19
10
0.925
160
32
14
0.950
85
23
0.975
80
m=4
pconf
pesp,lb
psp
0.800
0.900
0.925
0.950
0.975
0.990
0.800
0.700
5
2
2
1
1
1
0.800
4
3
2
1
1
0.900
23
6
3
2
0.925
16
4
2
0.950
9
3
0.975
9
0.900
0.700
9
3
2
2
2
1
0.800
7
4
3
2
2
0.900
49
12
5
3
0.925
34
8
4
0.950
19
6
0.975
18
0.925
0.700
10
3
3
2
2
2
0.800
8
5
4
3
2
0.900
61
14
6
3
0.925
43
9
4
0.950
23
7
0.975
22
0.950
0.700
13
4
3
3
2
2
0.800
10
6
4
3
2
0.900
79
18
7
4
0.925
55
11
5
0.950
30
9
0.975
28
0.975
0.700
18
5
4
3
3
2
0.800
14
9
6
4
3
0.900
111
25
9
5
0.925
77
16
7
0.950
41
12
0.975
38
0.990
0.700
25
7
5
4
3
3
0.800
18
12
8
5
4
0.900
155
35
13
7
0.925
107
21
10
0.950
57
16
0.975
53
m=5
pconf
pesp,lb
psp
0.800
0.900
0.925
0.950
0.975
0.990
0.800
0.700
4
1
1
1
1
1
0.800
3
2
2
1
1
0.900
17
5
2
1
0.925
12
3
2
0.950
7
3
0.975
7
0.900
0.700
7
2
2
2
1
1
0.800
5
3
2
2
1
0.900
37
9
4
2
0.925
26
6
3
0.950
14
4
0.975
13
0.925
0.700
8
3
2
2
1
1
0.800
6
4
3
2
2
0.900
46
11
4
3
0.925
32
7
3
0.950
18
5
0.975
16
0.950
0.700
10
3
2
2
2
1
0.800
8
5
3
2
2
0.900
59
14
5
3
0.925
41
9
4
0.950
22
7
0.975
21
0.975
0.700
14
4
3
3
2
2
0.800
10
7
4
3
2
0.900
83
19
7
4
0.925
58
12
5
0.950
31
9
0.975
29
0.990
0.700
19
5
4
3
3
2
0.800
14
9
6
4
3
0.900
116
26
10
5
0.925
80
16
7
0.950
43
12
0.975
40
