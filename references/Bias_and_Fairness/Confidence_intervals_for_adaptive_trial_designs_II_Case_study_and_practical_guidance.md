# Confidence intervals for adaptive trial designs II: Case study and practical guidance  
# David S. Robertson1*, Thomas Burnett2, Babak Choodari-Oskooei3, Munya Dimairo4 Michael Grayling5, Philip Pallmann6, Thomas Jaki1,7 
1 MRC Biostatistics Unit, University of Cambridge, UK  2 University of Bath, UK  3 MRC Clinical Trials Unit at UCL, UK  4 School of Health and Related Research (ScHARR), University of Sheffield, UK 5 Johnson & Johnson Innovative Medicine  6 Centre for Trials Research, Cardiff University, UK  7 University of Regensburg, Germany  * Corresponding author: david.robertson@mrc-bsu.cam.ac.uk 
# Abstract
In adaptive clinical trials, the conventional confidence interval (CI) for a treatment effect is  prone to undesirable properties such as undercoverage and potential inconsistency with the  final hypothesis testing decision. Accordingly, as is stated in recent regulatory guidance on  adaptive designs, there is the need for caution in the interpretation of CIs constructed during  and after an adaptive clinical trial. However, it may be unclear which of the available CIs in  the literature are preferable. This paper is the second in a two-part series that explores CIs for  adaptive trials. Part I provided a methodological review of approaches to construct CIs for  adaptive designs. In this paper (part II), we present an extended case study based around a twostage group sequential trial, including a comprehensive simulation study of the proposed CIs  for this setting. This facilitates an expanded description of considerations around what makes  for an effective CI procedure following an adaptive trial. We show that the CIs can have notably different properties. Finally, we propose a set of guidelines for researchers around the choice  of CIs and the reporting of CIs following an adaptive design. 
Keywords: Bootstrap; Conditional inference; Coverage; Estimation; Group sequential;  Interim analysis. 
# 1. Introduction
Clinical trials are traditionally run in a fixed manner that does not allow for interim looks at the data within the trial itself. In contrast, an adaptive design (AD) allows for pre-planned modifications to the course of the trial based on interim data analyses1–3. This added flexibility
● Early trial stopping: Group sequential designs (GSD) allow the trial to stop early at  interim looks for efficacy or futility/lack-of benefit. 
● Treatment selection: Multi-arm multi-stage (MAMS) designs test multiple treatment  options in parallel (typically against a common control arm), allowing the dropping of  treatment arm(s) that are not performing (as) well. 
# ● Population selection: Adaptive enrichment designs allow the clinical (sub)population of interest to be selected (‘enriched’) at interim looks, typically using pre-defined  patient subpopulations based on biomarker information. 
● Population selection: Adaptive enrichment designs allow the clinical (sub)population of interest to be selected (‘enriched’) at interim looks, typically using pre-defined  patient subpopulations based on biomarker information. 
● Changing randomisation probabilities: Response-adaptive randomisation (RAR)  allows updates of the randomisation probabilities based on patient responses, for  example to favour treatment arm(s) that are performing well6. 
● Changing trial sample size: Sample size re-estimation allows the sample size of the trial to be adjusted, for example based on interim conditional power calculations. 
Further educational material on all of these ADs can be found in Burnett et al. (2020)7,  Pallmann et al. (2018)1, and the PANDA online resource (https://panda.shef.ac.uk/)8. 
Regardless of the type of AD, it remains crucial that the integrity and validity of the trial is maintained. Appropriate estimation of treatment effects is a key part of trial validity, which includes not only point estimates (see Robertson et al. 2023a,b9,10) but also quantification of the uncertainty around the estimated treatment effects as given by confidence intervals (CIs). Intuitively, CIs capture this uncertainty by offering an interval that is expected to typically contain the unknown parameter of interest.    An important consideration in practice then is whether proposed methods to construct CIs have desirable properties. Most importantly, this relates to the CI having the desired coverage probability (i.e., the long-run probability that the CI contains the true unknown treatment effect of interest). However, there are numerous other considerations, including the width of the CI (all else being equal, narrower CIs are preferred as they are more informative), whether the CI will always contain an associated point estimate, and whether the CI will always be consistent with the decision rule (i.e., with an associated hypothesis test). The fundamental problem for ADs is that use of standard CI methodology (i.e., CIs constructed using methods that do not account for the fact an AD has been used) may not necessarily result in desirable properties. 
Regardless of the type of AD, it remains crucial that the integrity and validity of the trial is maintained. Appropriate estimation of treatment effects is a key part of trial validity, which includes not only point estimates (see Robertson et al. 2023a,b9,10) but also quantification of the uncertainty around the estimated treatment effects as given by confidence intervals (CIs). Intuitively, CIs capture this uncertainty by offering an interval that is expected to typically contain the unknown parameter of interest. 
An important consideration in practice then is whether proposed methods to construct CIs have desirable properties. Most importantly, this relates to the CI having the desired coverage probability (i.e., the long-run probability that the CI contains the true unknown treatment effect of interest). However, there are numerous other considerations, including the width of the CI (all else being equal, narrower CIs are preferred as they are more informative), whether the CI will always contain an associated point estimate, and whether the CI will always be consistent with the decision rule (i.e., with an associated hypothesis test). The fundamental problem for ADs is that use of standard CI methodology (i.e., CIs constructed using methods that do not account for the fact an AD has been used) may not necessarily result in desirable properties. 
Recent regulatory guidance highlights these concerns, with the U.S. Food and Drug  Administration (FDA) noting that “confidence intervals for the primary and secondary  endpoints may not have correct coverage probabilities for the true treatment effects” and thus “confidence intervals should be presented with appropriate cautions regarding their  interpretation”11. The same guidance also highlights the need to pre-specify methods used to  compute CIs after an AD, see also the Adaptive designs CONSORT Extension (ACE) guidance2,3. Meanwhile, the European Medicines Agency (EMA) guidance states that  “methods to … provide confidence intervals with pre-specified coverage probability are  required” if an AD is used in a regulated setting12.     These concerns and regulatory guidance motivate the growing body of literature proposing  ‘adjusted’ CIs that are specifically tailored for use with a particular AD. However, in our  experience there is at best limited uptake of adjusted CIs in practice (with the possible exception of ‘repeated’ CIs in GSDs; see part I of the paper series and Section 3.1 of this current  paper for a definition), with many adaptive trials continuing to only report the standard CI.  Evidence for this in the context of two-stage single-arm trials can be found in Grayling and  Mander (2021)13, where only 2% of 425 articles reported an adjusted CI. This limited uptake  of adjusted CIs for ADs is due to a number of reasons, including the lack of awareness of  methods in the literature, available software/code and guidance around the choice of adjusted CI in practice.    This paper is the second in a two-part series that explores the issue of CIs for ADs. In part I of 
Recent regulatory guidance highlights these concerns, with the U.S. Food and Drug Administration (FDA) noting that “confidence intervals for the primary and secondary endpoints may not have correct coverage probabilities for the true treatment effects” and thus “confidence intervals should be presented with appropriate cautions regarding their interpretation”11. The same guidance also highlights the need to pre-specify methods used to compute CIs after an AD, see also the Adaptive designs CONSORT Extension (ACE) guidance2,3. Meanwhile, the European Medicines Agency (EMA) guidance states that “methods to … provide confidence intervals with pre-specified coverage probability are required” if an AD is used in a regulated setting12.  
  These concerns and regulatory guidance motivate the growing body of literature proposing ‘adjusted’ CIs that are specifically tailored for use with a particular AD. However, in our experience there is at best limited uptake of adjusted CIs in practice (with the possible exception of ‘repeated’ CIs in GSDs; see part I of the paper series and Section 3.1 of this current paper for a definition), with many adaptive trials continuing to only report the standard CI. Evidence for this in the context of two-stage single-arm trials can be found in Grayling and Mander (2021)13, where only 2% of 425 articles reported an adjusted CI. This limited uptake of adjusted CIs for ADs is due to a number of reasons, including the lack of awareness of methods in the literature, available software/code and guidance around the choice of adjusted CI in practice. 
This paper is the second in a two-part series that explores the issue of CIs for ADs. In part I of the series, we reviewed and compared methods for constructing CIs for different classes of ADs and critically discussed different approaches. In the current paper (part II), we consider CIs for ADs from a practical perspective, and propose a set of guidelines for researchers around the choice and reporting of CIs following an AD. We first briefly describe performance measures of CIs in Section 2. We introduce the case study in Section 3 and show how to calculate different types of CIs (with R code provided). We use the case study as a basis for a simulation study in Section 4. We conclude with guidance for researchers and discussion in Sections 5 and 6, respectively. 
# 2. Performance measures for CIs
As introduced in part I of this paper series, different desirable properties for CIs have been proposed, which we recapitulate below. To fix ideas, suppose we have a random sample 𝑿 from a probability distribution with parameter 𝜃, which is the single parameter of interest in the trial. A CI for 𝜃 with confidence level 1 −𝛼 is a random interval (𝐿(𝑋), 𝑈(𝑋)) that has the following (claimed) property: 𝑃(𝐿(𝑋) <  𝜃< 𝑈(𝑋)) =  1 −𝛼  for all 𝜃. 
The coverage probability (often shortened to just ‘coverage’) of a confidence interval (L(X),  U(X)) is given by 𝑃(𝐿(𝑋) <  𝜃< 𝑈(𝑋)). This is the key performance measure for a CI, given 
# that the definition itself of a CI is based around actually having the claimed ‘nominal’  1 −𝛼 confidence level. 
# Alongside coverage, other criteria for CIs (generally, and specifically for ADs) have been  proposed in the literature. The main criteria/performance measures include:   
● Correct coverage (arguably essential)  ● Width (all other things being equal, a smaller width is desirable)  ● Consistency/compatibility with the hypothesis test (see below)  ● Contains the point estimate of interest  ● Is in fact an interval (i.e., not a union of disjoint intervals, or the empty set)  ● Is computationally feasible/simple 
A CI is consistent/compatible with the hypothesis testing decision if it excludes the parameter value(s) that are rejected by the hypothesis test, and conversely excludes the parameter value(s) that are not rejected by the hypothesis test. If a CI is not consistent/compatible with the hypothesis testing decision then this can lead to problems with study interpretation and the communication of results. 
# 3. Case study
As an example, we use the phase III MUSEC (multiple sclerosis and extract of cannabis) group sequential trial (Bauer et al., 201614; Zajicek et al., 201215). MUSEC investigated a standardised oral cannabis extract (CE), assessing its effect on muscle stiffness in adults with stable multiple sclerosis compared to placebo. Its primary outcome was whether or not a patient had “relief from muscle stiffness” after 12 weeks of treatments, based on a dichotomised 11-point category rating scale. MUSEC utilised a two-stage GSD, with early stopping for superiority assessed using an O’Brien-Fleming (OBF) type boundary. The unblinded interim analysis was planned after 200 participants (100 per arm) had completed the 12 week treatment course, with the final analysis planned after 400 patients (200 per arm) if the interim stopping rule was not met. The actual trial did plan for sample size re-estimation too, but for simplicity we will focus on the group sequential aspect here.    Ultimately, the trial did continue to its second stage; Table 1 summarises the study data at the interim and final analyses, as well as the OBF efficacy stopping boundaries. As can be seen, at the interim analysis the boundary for early rejection of the null hypothesis (no difference in the proportion of subjects with relief from muscle stiffness between treatment arms) was almost crossed, with the standardised test statistic being close to the stopping boundary. 
As an example, we use the phase III MUSEC (multiple sclerosis and extract of cannabis) group  sequential trial (Bauer et al., 201614; Zajicek et al., 201215). MUSEC investigated a standardised  oral cannabis extract (CE), assessing its effect on muscle stiffness in adults with stable multiple  sclerosis compared to placebo. Its primary outcome was whether or not a patient had “relief  from muscle stiffness” after 12 weeks of treatments, based on a dichotomised 11-point category rating scale. MUSEC utilised a two-stage GSD, with early stopping for superiority assessed  using an O’Brien-Fleming (OBF) type boundary. The unblinded interim analysis was planned  after 200 participants (100 per arm) had completed the 12 week treatment course, with the final  analysis planned after 400 patients (200 per arm) if the interim stopping rule was not met. The  actual trial did plan for sample size re-estimation too, but for simplicity we will focus on the  group sequential aspect here. 
Ultimately, the trial did continue to its second stage; Table 1 summarises the study data at the  interim and final analyses, as well as the OBF efficacy stopping boundaries. As can be seen, at the interim analysis the boundary for early rejection of the null hypothesis (no difference in the proportion of subjects with relief from muscle stiffness between treatment arms) was almost  crossed, with the standardised test statistic being close to the stopping boundary. 
 
Interim analysis 
Final analysis 
Placebo 
Cannabis 
extract 
Placebo 
Cannabis 
extract 
Number of patients with relief from 
muscle stiffness 
12 
27 
21 
42 
Total number of subjects 
97 
101 
134 
143 
Standardised test statistic 
2.540 
2.718 
O’Brien-Fleming stopping 
boundary  
2.797 
1.977 
 
Typically, at the final analysis a 100(1 −𝛼)% CI will be desired for the difference in the response rates in the placebo and CE  arms (the measure of the treatment effect of interest). A common method of achieving this is to use a standard two-sided interval, e.g., based on Wald’s methodology16. For MUSEC, denoting the final sample sizes in the two arms by 𝑛𝑃= 134 and 𝑛𝐶𝐸= 143, and the MLEs for the rates by 𝑝̂𝑃= 21/𝑛𝑃 and 𝑝̂𝐶𝐸= 42/𝑛𝐶𝐸, this gives for 𝛼= 0.05 
where Φ−1(𝑥) denotes the inverse cumulative distribution function (CDF) of a standard normal random variable. 
If the interim analysis was not present, this CI would have a number of desirable properties (as  discussed in Section 2): it would be guaranteed to be an interval containing the MLE for 𝑝̂𝐶𝐸− 𝑝̂𝑃, would (at least asymptotically) have the desired coverage, would be consistent with the  associated hypothesis test, and would evidently be easily computed. 
In this Section (as well as the simulation study in Section 4), we illustrate how different types of CIs (both unconditional and conditional) can be used in practice for a GSD, based on the MUSEC trial. We use a GSD as our case study in order to illustrate the widest range of different CIs. 
Using the observed data from the MUSEC trial, we now show how to calculate various CIs for the treatment difference, denoted 𝜃= 𝑝𝐶𝐸−𝑝𝑃, from both a conditional and unconditional perspective (see explanation on the difference below) when the trial continues to stage 2, as happened for the MUSEC trial. R code to obtain all CIs considered is provided in the data files
# As already seen above, the standard/naive CI for the treatment difference (i.e., the Wald  CI), is given by  
# As already seen above, the standard/naive CI for the treatment difference (i.e., the Wald 
𝜃̂ ± Φ−1(1 −𝛼/2)𝑉𝑎𝑟 ̂ (𝜃̂) = (𝑝̂𝐶𝐸−𝑝̂𝑃) ± Φ−1(1 −𝛼/2)√𝑝̂𝐶𝐸(1 −𝑝̂𝐶𝐸)/𝑛𝐶𝐸+ 𝑝̂𝑃(1 −𝑝̂𝑃)/𝑛𝑝 
𝜃̂ ± Φ−1(1 −𝛼/2)𝑉𝑎𝑟 ̂ (𝜃̂) = (𝑝̂𝐶𝐸−𝑝̂𝑃) ± Φ−1(1 −𝛼/2)√𝑝̂𝐶𝐸(1 −𝑝̂𝐶𝐸)/𝑛𝐶𝐸+ 𝑝̂𝑃(1 −𝑝̂𝑃)/𝑛𝑝 
Other methods than Wald are available to calculate standard/naive CIs for the difference of two proportions17,18, but none of these will be able to account for the AD used. 
From an unconditional perspective, we want to estimate θ regardless of the stage that the tria stops, and are interested in the properties of the CI as averaged over all possible stopping times (weighted by the respective stage-wise stopping probabilities). Note that the standard/naive CI above is an unconditional CI. 
In what follows, we let e1, e2 denote the efficacy stopping boundaries, I1, I2 the (observed) information, at stages 1 and stage 2, respectively, and T the stage the trial stopped at (so T = 2 for the MUSEC trial). The definitions of the information I1 and I2 for the MUSEC trial are given in Appendix A.1, which depend on the number of observed successes.  
The exact unconditional CI depends on a choice of the ordering of the sample space with respect to evidence against the null hypothesis. In what follows, we use stagewise ordering, which has desirable properties described by Jennison and Turnbull (1999)19. This allows the use of the p-value function P(θ) to find the lower and upper bounds for the 95% CI, 𝜃̂𝑙 and 𝜃̂𝑢, which are the solutions to the equations P(𝜃̂𝑙) = 0.025 and P(𝜃̂𝑢) = 0.975. The formula for the p-value function for stopping stage T = 2 (with observed second stage test statistic 𝑍2 = 𝑧2 ) is as follows: 
where 𝑓2((𝒙𝟏, 𝒙𝟐), 𝜇, 𝛴) is the density of a bivariate normal distribution with mean vector 𝜇 and covariance matrix 𝛴 evaluated at the vector (𝒙𝟏, 𝒙𝟐). Note that the associated point estimator of this method, the median unbiased estimator (MUE), 𝜃̂𝑀𝑈𝐸 ,  is the solution to the equation P(𝜃̂𝑀𝑈𝐸) = 0.5. If the distributional assumptions hold exactly (i.e., the joint canonical distribution of test statistics holds exactly), then the exact unconditional CI guarantees  consistency with the test decision. 
The repeated CI (RCI) follows a simple form: 𝜃̂ ± 𝑒𝑇 / √𝐼𝑇 , see Jennison and Turnbull (1999)19. Note that there is no explicit associated point estimator with this method (although one could of course just use the standard MLE 𝜃̂). Since 𝜃̂ = 𝑍𝑇/√𝐼𝑇 , the RCI guarantees consistency with the test decision.    The adjusted asymptotic CI adjusts the standard CI, giving a CI of the form 
The repeated CI (RCI) follows a simple form: 𝜃̂ ± 𝑒𝑇 / √𝐼𝑇 , see Jennison and Turnbull (1999)19. Note that there is no explicit associated point estimator with this method (although one could of course just use the standard MLE 𝜃̂). Since 𝜃̂ = 𝑍𝑇/√𝐼𝑇 , the RCI guarantees consistency with the test decision. 
# The adjusted asymptotic CI adjusts the standard CI, giving a CI of the form   (𝜃̂ −𝜇̂(𝜃̂)) ± Φ−1(1 −𝛼/2)𝜎̂(𝜃̂)/√𝐼𝑇 
where 𝜇̂(𝜃̂) and 𝜎̂(𝜃̂) are functions of 𝜃̂ given in Todd et al. (1996)20:  𝜇̂(𝜃̂) = 𝐸(𝜃̂) −𝜃̂ 𝐸(√𝐼𝑇)  𝜎̂(𝜃̂) = 𝐸(𝑍𝑇 2/𝐼𝑇) −2𝜃̂ 𝐸(𝑍𝑇) + 𝜃̂2 𝐸(𝐼𝑇) −𝜇̂(𝜃̂)2.   
# where 𝜇̂(𝜃̂) and 𝜎̂(𝜃̂) are functions of 𝜃̂ given in Todd et al. (1996)20:  𝜇̂(𝜃̂) = 𝐸(𝜃̂) −𝜃̂ 𝐸(√𝐼𝑇)  𝜎̂(𝜃̂) = 𝐸(𝑍𝑇 2/𝐼𝑇) −2𝜃̂ 𝐸(𝑍𝑇) + 𝜃̂2 𝐸(𝐼𝑇) −𝜇̂(𝜃̂)2.   
# here 𝜇̂(𝜃̂) and 𝜎̂(𝜃̂) are functions of 𝜃̂ given in Todd et al. (1996)20:
The associated point estimator for this method is the bias-adjusted estimator 𝜃̂ −𝜇̂(𝜃̂) .
One subtlety with the use of the adjusted asymptotic CI (for trials that continue to stage 2) is that it is possible for the information levels to actually decrease from stage 1 to stage 2 i.e., 𝐼2 < 𝐼1. This can happen in trials with a binary outcome when the pooled estimated response rate is very close to zero (or 1) at stage 1, but further away from zero (and 1) at stage 2. In this situation, it is not possible to calculate the adjusted asymptotic CI, although this happens very rarely (i.e., one or two times out of 105 simulation replicates) for trials with success rates similar to those observed in the MUSEC trial. It is similarly possible for this to occur in trials with, e.g., normal (if the estimated standard deviation differs greatly between stages) and time-toevent data (typically when the number of events that occur between analyses is smaller), though again is a rare occurrence. 
For the parametric bootstrap CI we use the a simple bootstrap algorithm to generate B bootstrap MLEs 𝜃̂(𝑏) for b = 1, …, B. In the interests of space, we defer the details to Appendix A.1. Note that the associated point estimator for this method is given by the mean of 𝜃̂(1), . . . , 𝜃̂(𝐵). 
Finally, the randomisation-based CI follows a different bootstrap procedure in order to calculate an adjusted p-value, based on the randomisation distribution (this being an example of randomisation-based inference i.e., a randomisation test). The idea is to reproduce the group sequential analysis for each allowable allocation of patients to treatments, but keeping the observed patient outcomes fixed. The adjusted p-value is then the proportion of these potential results that are as extreme or more extreme than the actual trial result. Note that at the interim analysis, exactly the same set of patients are used each time (although with different treatment allocations). 
More precisely, the procedure is defined as follows (Snapinn, 199421). Suppose a group sequential trial has been completed, and let X denote the set of all observed patient outcomes
Let the vector T denote the set of all possible allocations of patients to treatments (given the randomisation procedure used). Let 𝑡𝑖 denote a potential allocation of patients to treatments and 𝑡∗ denote the actual allocation used in the trial. Then let F(X, 𝑡𝑖) be the measure of strength of evidence against the null hypothesis. The adjusted p-value is then 
#    
#  [ Φ−1(1 −𝑝𝑎𝑑𝑗𝑢𝑠𝑡𝑒𝑑) ± Φ−1(1 −𝛼/2) ] √𝑝̂𝐶𝐸(1 −𝑝̂𝐶𝐸)/𝑛𝐶𝐸+ 𝑝̂𝑃(1 −𝑝̂𝑃)/𝑛𝑝 
 Φ−1(1 −𝑝𝑎𝑑𝑗𝑢𝑠𝑡𝑒𝑑)√𝑝̂𝐶𝐸(1 −𝑝̂𝐶𝐸)/𝑛𝐶𝐸+ 𝑝̂𝑃(1 −𝑝̂𝑃)/𝑛𝑝 
In practice, it is not feasible (except for small sample sizes) to use the entire set of possible random allocations T and we use N random samples from the set instead. However, a problem arises when 𝑝𝑎𝑑𝑗𝑢𝑠𝑡𝑒𝑑= 0 i.e., no allocations are sampled that give results as extreme as or more extreme than those actually observed, which gives  Φ−1(1 −𝑝𝑎𝑑𝑗𝑢𝑠𝑡𝑒𝑑) = ∞. Even when N = 10,000 this can occur as discussed later in the simulation study. 
Conditional CIs 
From a conditional perspective, we are interested in estimation conditional on the stage the tri stops at (so for the MUSEC trial, conditional on the trial continuing to stage 2). 
𝐶𝐼𝑐 = {𝜃∶ 𝛼/2 < 𝑃𝑟𝑜𝑏(𝜃̂  ≥𝜃̂𝑜𝑏𝑠 | 𝑇 = 𝑡, 𝜃) < 1 −𝛼/2}
where 𝜃̂𝑜𝑏𝑠 is the observed value of the MLE, see Fan and DeMets (2006)22. More explicitly,  for a trial continuing to stage 2  (so 𝑡= 2) the lower and upper and bounds for the CI, 𝜃̂𝑙 and  𝜃̂𝑢 , are the solutions to the equations 
     
and   
where 𝑓(𝜃̂ | 𝜃 , 𝑇= 2) is the conditional density of the MLE (conditional on continuing to stage 2), see Appendix A.1 for further details. The associated point estimator is the conditiona MUE, 𝜃̂𝐶𝑀𝑈𝐸 , which is the solution to the following equation: 
Like for the adjusted asymptotic CI, if the information levels decrease from stage 1 to stage 2  i.e., 𝐼2 < 𝐼1 , then the exact conditional CI (and hence the restricted exact conditional CI, see  below) cannot be calculated.  The restricted exact conditional CI, as the name suggests, restricts the range of the exact  conditional CI. Given the exact conditional CI (𝐶𝐼𝑐) defined above, the restricted exact  conditional CI is defined to be 𝐶𝐼𝑐∩𝐶𝐼𝑟 , where   𝐶𝐼𝑟= {𝜃∶ 𝑃𝑟𝑜𝑏(𝑇≤𝑡 | 𝜃) > 𝛼/2  and 𝑃𝑟𝑜𝑏(𝑇≥𝑡 | 𝜃) > 𝛼/2} ,  see Fan and DeMets (2006)22. For a trial continuing to stage 2 (so 𝑡= 2), the upper bound of  the restricted exact conditional CI is set equal to 𝑚𝑖𝑛{𝜃̂𝑢 ,(𝑒1 −𝛷−1(𝛼/2))/√𝐼1}, where 𝜃̂𝑢 is  the upper bound of 𝐶𝐼𝑐.  The conditional likelihood CI is based on the conditional MLE i.e., the maximiser of the  conditional log-likelihood. As shown in Marschner et al. (2022)23, the conditional loglikelihood, conditioning on the trial stopping at stage T = t is given by   
𝐿𝑐(𝜃 ; 𝑧𝑡 , 𝑡) = − 1 2 (𝑧𝑡−𝜃√𝐼𝑡 )2 −𝑙𝑜𝑔𝑃𝑟(𝑇= 𝑡 | 𝜃).
Using the results of Fan and DeMets (2006)22, we can show that the conditional MLE, 𝜃̂𝑐 , is the solution of the following equation when T = 2: 
𝜃̂𝑐 = 𝜃̂𝑜𝑏𝑠−√𝐼1 𝜙(𝑒1 −𝜃̂𝑐 √𝐼1) 𝐼2𝛷(𝑒1 −𝜃̂𝑐 √𝐼1)  
where 𝜙 and 𝛷 are the probability density function (pdf) and cdf of the standard normal distribution, respectively. The conditional MLE is the associated point estimator for the conditional likelihood CI. The conditional likelihood CI is calculated using a conditional bootstrap procedure, with full details provided in Appendix A.1. 
The penalised likelihood CI is equal to the conditional likelihood CI when the trial continues to stage 2, see Section 3.2 for details of how it is different when in fact the trial stops at stage 1. 
The penalised likelihood CI is equal to the conditional likelihood CI when the trial continues to stage 2, see Section 3.2 for details of how it is different when in fact the trial stops at stage
In this subsection we detail how the various CIs are calculated when the stopping stage is T  1 (rather than T = 2 as in Section 3.1), which will be needed for the Simulation study in Sectio 4.  The standard/naive CI for the treatment difference (i.e., the Wald CI) is given by   𝜃̂ ± Φ(1 −𝛼/2)𝑉𝑎𝑟 ̂(𝜃̂) = (𝑝̂𝐶𝐸−𝑝̂𝑃) ± Φ(1 −𝛼/2)√𝑝̂𝐶𝐸(1 −𝑝̂𝐶𝐸) 𝑛1,𝐶𝐸 + 𝑝̂𝑃(1 −𝑝̂𝑃) 𝑛1,𝑃     where 𝑝̂𝑃 and 𝑝̂𝐶𝐸 are the mean stage 1 response rates for the placebo and CE arms,  respectively. 
In this subsection we detail how the various CIs are calculated when the stopping stage is  1 (rather than T = 2 as in Section 3.1), which will be needed for the Simulation study in Sect 4. 
𝜃̂ ± Φ(1 −𝛼/2)𝑉𝑎𝑟 ̂(𝜃̂) = (𝑝̂𝐶𝐸−𝑝̂𝑃) ± Φ(1 −𝛼/2)√𝑝̂𝐶𝐸(1 −𝑝̂𝐶𝐸) 𝑛1,𝐶𝐸 + 𝑝̂𝑃(1 −𝑝̂𝑃) 𝑛1,𝑃
where 𝑝̂𝑃 and 𝑝̂𝐶𝐸 are the mean stage 1 response rates for the placebo and CE arms,  respectively. 
Unconditional CIs
For the exact unconditional CI we again use the p-value function P(θ) (based on stagewise ordering of the sample space) to find the lower and upper and bounds for the 95% CI, 𝜃̂𝑙 and 𝜃̂𝑢, which are the solutions to the equations P(𝜃̂𝑙) = 0.025 and P(𝜃̂𝑢) = 0.975. This admits a closed-form expression when T = 1, with 𝜃̂𝑙= Φ−1(𝛼/2)+Z1 √𝐼1  and 𝜃̂𝑢= Φ−1(1−𝛼/2)+Z1 √𝐼1 .  The repeated CI (RCI) and adjusted asymptotic CI are given in Section 3.1, since they are written in terms of a general stopping stage T.   The parametric bootstrap CI procedure is the same as before, except that now 𝑝̂𝐶𝐸 and 𝑝̂𝑃 represent the stage 1 response rate estimates on the CE and placebo arm, respectively.   Conditional CIs  When T = 1, the exact conditional CI has the same definition as given for 𝐶𝐼𝑐 in Section 3.1. More explicitly, for a trial stopping at stage 1, the lower and upper bounds for the CI, denoted 𝜃̂𝑙 and 𝜃̂𝑢 , are the solutions to the equations 𝛼/2 = 1−Φ(√I1[θ̂obs−θ̂l]) 1−Φ(e1−√I1θ̂l )   and  1 −𝛼/2 = 1−Φ(√I1[θ̂obs−θ̂u]) 1−Φ(e1−√I1θ̂u )  .   
For the exact unconditional CI we again use the p-value function P(θ) (based on stagewise  ordering of the sample space) to find the lower and upper and bounds for the 95% CI, 𝜃̂𝑙 and  𝜃̂𝑢, which are the solutions to the equations P(𝜃̂𝑙) = 0.025 and P(𝜃̂𝑢) = 0.975. This admits a  closed-form expression when T = 1, with 𝜃̂𝑙= Φ−1(𝛼/2)+Z1 √𝐼1  and 𝜃̂𝑢= Φ−1(1−𝛼/2)+Z1 √𝐼1 .  The repeated CI (RCI) and adjusted asymptotic CI are given in Section 3.1, since they are  written in terms of a general stopping stage T.   The parametric bootstrap CI procedure is the same as before, except that now 𝑝̂𝐶𝐸 and 𝑝̂𝑃  represent the stage 1 response rate estimates on the CE and placebo arm, respectively.   
Conditional CIs
When T = 1, the exact conditional CI has the same definition as given for 𝐶𝐼𝑐 in Section 3.1. More explicitly, for a trial stopping at stage 1, the lower and upper bounds for the CI, denoted 𝜃̂𝑙 and 𝜃̂𝑢 , are the solutions to the equations 𝛼/2 = 1−Φ(√I1[θ̂obs−θ̂l]) 1−Φ(e1−√I1θ̂l )   and  1 −𝛼/2 = 1−Φ(√I1[θ̂obs−θ̂u]) 1−Φ(e−√Iθ̂ )  . 
Similarly, the restricted exact conditional CI has the same definition 𝐶𝐼𝑐∩𝐶𝐼𝑟 with 𝐶𝐼𝑟 a given in Section 3.1. For a trial stopping at stage 1, the lower bound of the restricted exac conditional CI is set equal to 𝑚𝑎𝑥{𝜃̂𝑙 ,(𝑒1 −Φ−1(1 −𝛼/2))/√𝐼1}, where 𝜃̂𝑢 is the upper bound of 𝐶𝐼𝑐. 
# The conditional likelihood CI is based on the conditional log-likelihood as given in Section 3.1. Again using the results of Fan and DeMets (2006)22, the conditional MLE, 𝜃̂𝑐 , is the solution of the following equation when T = 1:  
𝜃̂𝑐 = 𝜃̂𝑜𝑏𝑠− 𝜙(𝑒1 −𝜃̂𝑐 √𝐼1) √𝐼1𝛷(𝑒1 −𝜃̂𝑐 √𝐼1)
The conditional likelihood CI is calculated using a conditional bootstrap procedure, with ful details given in Appendix A.1. 
Finally, the penalised likelihood CI is different from the conditional likelihood CI when the  trial stops at stage 1. As described in Marschner et al. (2022)23, the penalised log-likelihood is  given by 𝐿𝜆(𝜃 ; 𝑧𝑡 , 𝑡) = − 1 2 (𝑧𝑡−𝜃√𝐼𝑡 )2 −𝜆 𝑙𝑜𝑔 𝑃𝑟𝑜𝑏(𝑇= 𝑡 |  𝜃). Hence the MLE and  the conditional MLE correspond to maximising the penalised log-likelihood when 𝜆= 0 and  𝜆= 1, respectively. For a given choice of 𝜆∈[0,1] this gives a penalised likelihood estimate  𝑃(𝜆 , 𝑧𝑡 , 𝑡) = 𝑎𝑟𝑔𝑚𝑎𝑥𝜃 𝐿𝜆(𝜃 ; 𝑧𝑡 , 𝑡). The choice of 𝜆 proposed for t =1 is 𝜆∗= {𝜆∈[0,1] ∶  𝑃(𝜆 , 𝑒1 , 1) = 0}. With this choice, the penalised MLE is defined as 𝜃̂𝑝= 𝑃(𝜆∗ , 𝑧1 , 1). The  penalised likelihood CI is then calculated using the same bootstrap procedure as above for the  conditional likelihood CI, with the bootstrap conditional MLE 𝜃̂𝑐 (𝑏) replaced by the penalised  MLE 𝜃̂𝑝 (𝑏).  Note that by conditioning the bootstrap sampling on early stopping, each bootstrap  replication satisfies 𝜃̂𝑝 (𝑏) > 0, which therefore guarantees that the associated CI lies above zero,  consistent with the decision to stop the study early for benefit.   
# 3.3 Results from the MUSEC trial
Table 2 gives the values of all of the CIs described in Section 3.1, calculated using the observed data and OBF stopping boundaries from the MUSEC trial, with Figure 1 giving the graphical representation. When there is an associated point estimate (as described above in Section 2.1), this is also shown. For the methods requiring repeated sampling/simulation (i.e., the unconditional parametric bootstrap CI, unconditional randomisation-based CI and conditional (penalised) likelihood CI), we used N  = 106 trial replicates. 
Type of CI 
CI method 
Point estimate 
95% two-sided CI 
CI width 
Standard/naive 
Wald test 
0.137 [overall MLE] 
(0.040, 0.234) 
0.194 
Unconditional 
Exact 
0.134 [MUE] 
(0.034, 0.234) 
0.200 
Repeated 
- 
(0.037, 0.237) 
0.199 
Adjusted asymptotic 
0.137 
(0.039, 0.235) 
0.196 
Parametric bootstrap 
0.143 
(0.041, 0.253) 
0.212 
Randomisation-based 
0.130 
(0.033, 0.226) 
0.194 
Conditional 
Exact 
0.185 [conditional MUE] 
(0.052, 0.358) 
0.306 
Restricted exact 
0.185 [conditional MUE] 
(0.052, 0.269) 
0.217 
(Penalised) likelihood 
0.191 [conditional MLE] 
(0.034, 0.304) 
0.271 
Table 2: Confidence intervals (and associated point estimates) calculated using the observed data and O’BrienFleming efficacy stopping boundaries from the MUSEC trial. 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c7fc/c7fcd2f4-a981-44d9-8f9e-6c32535dbca5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Graphical representation of confidence intervals (and associated point estimates) calculated using th observed data from the MUSEC trial. RCI = Repeated Confidence Interval. </div>
The Wald (standard) CI is (0.040, 0.234) with a width of 0.194, and is the comparator for all  the other CIs in Table 2, since it is the conventional end-of-trial CI. Starting with the  unconditional CIs, all of them are very similar to the standard CI, with the exception of the  simple bootstrap method which is wider (an increase of 9%) than the standard CI. In contrast,  the conditional CIs all are substantially wider than the standard CI, with an increase of 58%,  12% and 40% for the conditional exact, restricted exact and (penalised) likelihood CIs,  respectively. This reflects the loss of information associated with conditioning on the stopping  stage T = 2. The conditional point estimators are also substantially higher than the  unconditional point estimators. This upward correction is intuitive from a conditional  perspective: there is downward ‘selection pressure’ on the MLE calculated at the end of stage 1, since if this is large then the trial does not continue to stage 2. For the conditional CIs, this  is also reflected in their upper confidence limits being substantially higher compared with the  standard CI. Finally, there is a marked asymmetry in the restricted exact CI around its  associated point estimate, reflecting how the upper confidence limit of the conditional exact CI  is adjusted. 
For the MUSEC trial data, the use of different methods can give noticeably different CIs for the treatment effect, particularly when considering a conditional versus unconditional perspective. This could influence the interpretation of the trial results, and highlights the importance of pre-specifying which CI(s) will be reported following an AD. The choice of CI(s) will depend on what the researchers wish to achieve regarding the estimand24 in question.
There will be pros and cons for the different CI methods, which we explore further in the simulation study in Section 4. We also note that there is a strong link between design and estimation - the CIs above depend on the design of the trial, and would be different if (fo example) the design had also included futility stopping boundaries. 
# 4. Simulation study
# 4.1 Simulation set-up
4.1 Simulation set-up
Since the CIs calculated above are one realisation of the trial data given the trial design, in this Section we carry out a simulation study to investigate the performance of the CIs under different scenarios. Note that we have not used assumed values for the underlying treatment effects to calculate the CIs given in Table 2. As can be seen from the formulae in Section 3.1, these CIs only depend on the observed data and efficacy stopping boundary.  To demonstrate the properties of the CIs when averaged over many trial realisations following the two-stage design of the MUSEC trial, we ran simulations under different values of 𝑝𝐶𝐸 , the assumed true value of the response rate for the CE arm. For simplicity, we kept the value
Since the CIs calculated above are one realisation of the trial data given the trial design, in this Section we carry out a simulation study to investigate the performance of the CIs under different scenarios. Note that we have not used assumed values for the underlying treatment effects to calculate the CIs given in Table 2. As can be seen from the formulae in Section 3.1, these CIs only depend on the observed data and efficacy stopping boundary. 
To demonstrate the properties of the CIs when averaged over many trial realisations following  the two-stage design of the MUSEC trial, we ran simulations under different values of 𝑝𝐶𝐸 ,  the assumed true value of the response rate for the CE arm. For simplicity, we kept the value  of 𝑝𝑃 , the assumed true value of the response rate for the placebo arm, equal to the value  observed in the MUSEC trial, i.e. 𝑝𝑃= 𝑝𝑝 ̂ = 21/134 ≈0.157. 
# We used the following procedure for our trial simulations: 
) Given the assumed value for 𝑝𝐶𝐸 , denoted 𝑝𝐶𝐸 ∗ , generate N stage 1 trial replicates S1,CE (1)  , . . . , S1,CE (N)    and  S1,P (1) , . . . , S1,P (N)  ,  where  S1,CE (i)  ~ 𝐵𝑖𝑛(𝑛1,𝐶𝐸, 𝑝𝐶𝐸 ∗ )  and 𝑆1,𝑃 (𝑖) ~ 𝐵𝑖𝑛(𝑛1,𝑃 , 𝑝̂𝑃 ). 
2) For i = 1, …, N calculate the bootstrap standardised stage 1 test statistic 𝑍1 (𝑖) from the bootstrap values S1,CE (i)   and S1,P (i)  .  a) If Z1 (i) > e1 then calculate all CIs (conditional and unconditional) with T = 1.  b) Otherwise, generate a stage 2 trial replicate 𝑆2,𝐶𝐸 (𝑖)   and 𝑆2,𝑃 (𝑖)  , where 𝑆2,𝐶𝐸 (𝑖)  ~ 𝐵𝑖𝑛(𝑛𝐶𝐸−𝑛1,𝐶𝐸 , 𝑝𝐶𝐸 ∗ ) and 𝑆2,𝑃 (𝑖) ~ 𝐵𝑖𝑛(𝑛𝑝−𝑛1,𝑃 , 𝑝̂𝑃 ). Then calculate all CIs (conditional and unconditional) with T = 2. 
For each value of 𝑝𝐶𝐸, we simulated N = 105 trial replicates. For the CI methods requiring a bootstrap procedure, we used B = 104 bootstrap samples. For the randomisation-based CI, even with 104 samples we still frequently ran into the issue of 𝑝𝑎𝑑𝑗𝑢𝑠𝑡𝑒𝑑= 0. Hence we did not consider this CI method further in the simulation study. 
For each of the CI methods considered, we evaluated the following properties: 
1) Mean coverage  2) Mean width and standard error (se)  3) Consistency  4) Lower coverage: probability that the lower confidence limit is above the true value of 𝜃, denoted 𝑃(𝐿(𝑋) > 𝜃).  5) Upper coverage: probability that the upper confidence limit is below the true value of 𝜃, denoted 𝑷(𝑼(𝑿) < 𝜽). 
6)
4.2 Simulation results
Table 3 shows the overall (unconditional) simulation results with the true success rates (𝑝𝑝 , 𝑝𝐶𝐸 ) on the two arms equal to the overall observed means in the MUSEC trial i.e., 𝑝𝑝= 21/134 ≈0.157 and 𝑝𝐶𝐸= 42/143 ≈0.294. The probability of stopping early for efficacy in stage 1 is 0.308. Note that with N = 105 trial replicates, the Monte Carlo standard error for the coverage and consistency results is less than 0.0016. 
<div style="text-align: center;">Overall (unconditional) results</div>
Type of CI 
CI method 
Coverage 
Mean width (se) 
Consistency 
𝑷(𝑳(𝑿)
> 𝜽) 
𝑷(𝑼(𝑿)
< 𝜽) 
Standard/naive 
Wald test 
0.945 
0.203 (0.017) 
0.989 
0.029 
0.026 
Unconditional 
Exact 
0.952 
0.211 (0.019) 
0.998 
0.022 
0.026 
Repeated 
0.973 
0.240 (0.063) 
1.000 
0.002 
0.025 
Adjusted asymptotic 
0.945 
0.205 (0.020) 
0.985 
0.022 
0.034 
Parametric bootstrap 
0.926 
0.210 (0.010) 
0.988 
0.056 
0.018 
Conditional 
Exact 
0.954 
0.386 (0.268) 
0.732 
0.024 
0.022 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/503e/503e40ef-7844-492e-9208-e410224475b2.png" style="width: 50%;"></div>
Table 3: Simulation results showing the performance of various CIs with 𝑝𝑝= 21/134 ≈0.157 and 𝑝𝐶𝐸= 42/143 ≈0.294. There were 105 trial replicates. The probability of stopping at stage 1 is 0.308. 
Starting with the coverage of the CIs, the standard CI has a very slight undercoverage, which  is caused by two factors: 1) the distributional assumptions underlying the Wald CI are no longer met due to the stopping rule, and 2) the quality of the (asymptotic) normal approximation used for binomial outcomes. As expected, the exact unconditional CI attains the nominal coverage of 95% (within Monte Carlo error). In contrast, the RCI has conservative coverage, which is  driven by the trial replicates that stop early at stage 1 (see the following tables). The adjusted asymptotic CI has the same coverage as the standard CI. The parametric bootstrap CI has  particularly low coverage of <93% in this scenario.     Turning to the conditional estimators, both the conditional exact CI and conditional restricted exact CI attain the nominal coverage of 95% (with a very slight overcoverage). This is to be  expected, since if a CI attains (or exceeds) the nominal coverage conditional on stopping at stage 1 and conditional on continuing to stage 2, then it will also attain (or exceed) the nominal coverage unconditionally (averaged over the two stopping possibilities). In contrast, the conditional likelihood and penalised likelihood have a notably conservative coverage, which is at least partly due to the choice of the conditional bootstrap procedure (using the bias-corrected bootstrap, for example, would give different results but is out of scope of this paper).    Looking at the mean CI width, the CI methods with higher mean coverage compared with the standard CI also have a higher mean width. However, even though the parametric bootstrap CI has a lower coverage than the standard CI, ist also has a slightly higher mean width. The mean widths of the unconditional exact, adjusted asymptotic and parametric bootstrap CIs are all  within +4% of the mean width of the standard CI. In contrast, the RCI has a substantial increase of +17%. As for the conditional CIs, what is striking is the huge increases in mean width for the conditional exact and conditional likelihood CIs, of +90% and +139%, respectively. These are also accompanied with very high standard errors, reflecting a very high variability in the  confidence limits. Such results have previously been noted in the literature22,23,25. In contrast, the conditional restricted exact and penalised likelihood CIs have mean widths much closer to
Looking at the mean CI width, the CI methods with higher mean coverage compared with the  standard CI also have a higher mean width. However, even though the parametric bootstrap CI  has a lower coverage than the standard CI, ist also has a slightly higher mean width. The mean  widths of the unconditional exact, adjusted asymptotic and parametric bootstrap CIs are all  within +4% of the mean width of the standard CI. In contrast, the RCI has a substantial increase  of +17%. As for the conditional CIs, what is striking is the huge increases in mean width for  the conditional exact and conditional likelihood CIs, of +90% and +139%, respectively. These  are also accompanied with very high standard errors, reflecting a very high variability in the  confidence limits. Such results have previously been noted in the literature22,23,25. In contrast,  the conditional restricted exact and penalised likelihood CIs have mean widths much closer to 
that of the standard CI, although there is still an increase of +12% and +33%, respectively. Again, this reflects the loss of information associated with conditioning on the stopping stage.
Again, this reflects the loss of information associated with conditioning on the stopping stage.    In terms of consistency, the standard CI has consistency just below 99%, with instances of nonconsistent CIs being caused by the same reasons as for the undercoverage. The unconditional exact CI has a coverage just below 100%, with the non-consistency caused by the the quality of the (asymptotic) normal approximation used for binomial outcomes, see also Lloyd (2021). The RCI is the only CI method with 100% consistency. Both the adjusted asymptotic and  unconditional parametric bootstrap CIs have very similar consistency to the standard CI. For the conditional CIs, again it is striking how low the consistency is for the conditional exact and conditional likelihood CIs, which is driven by how they are so wide (on average) that they often will include zero even when the null hypothesis of no treatment effect is rejected. In contrast, the conditional restricted exact and penalised likelihood CIs have consistencies much closer to 100%, comparable to the consistency of the standard CI.    Finally, looking at the upper and lower coverages, these are approximately equal for the standard, unconditional exact, conditional exact and conditional restricted exact CIs. The upper coverage is (sometimes substantially) greater than the lower coverage for the RCI, adjusted asymptotic, conditional likelihood and penalised likelihood CIs. The unconditional parametric bootstrap CI is the only method to have the lower coverage higher than the upper coverage.    In order to show more clearly the differences between the CIs in repeated realisations of the  MUSEC trial, Figure 2 shows the CIs from the first 10 simulation replicates of the simulation  study. Note that in simulation replicates 2, 3, 6, 8 and 10, the trial stopped early at stage 1 (and hence with rejection of the null hypothesis). In simulation replicates 1, 4, 5, 7 and 9, the trial continued to stage 2, with rejection of the null hypothesis in simulation replicates 1, 7 and 9.    Looking first at coverage (i.e., whether the CI contains the true value of the treatment  difference, shown by the red horizontal line), in simulation replicates 1, 2, 3, 6, 7, 8 and 9, all  CI methods contain the true value, whereas in simulation replicate 5 all CI methods do not contain the true value. However, in simulation replicate 4 we can see a discrepancy in the  coverage, with the standard, unconditional exact, RCI and adjusted asymptotic CIs not  containing the true value, whereas the other CIs do contain the true value. In simulation replicate 10, all CI methods contain the true value except for the bootstrap CI. Similarly, in  terms of consistency, in simulation replicates 2, 3, 6 and 8, the conditional exact and conditional likelihood CIs contain zero (with the conditional likelihood CI additionally containing zero in simulation replicate 10), despite the null hypothesis of no treatment effect being rejected.    Finally, in terms of CI width, in all of the simulation replicates where the trial stopped early in stage 1 (apart from simulation replicate 10), it is striking how much wider both the conditional exact and conditional likelihood CIs are compared to any of the others. In the trial replicates that continue to stage 2, the CI widths are much more similar, although the conditional CIs are wider than the unconditional ones (as would be expected).   
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d885/d88573e4-4d12-4efd-a411-b8932d4653a2.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Confidence intervals from the first 10 simulation replicates of the simulation study. The horizontal red  dashed line shows the true value of the treatment difference. RCI = Repeated Confidence Interval.    </div>
# Results conditional on stopping at stage 1
Apart from the overall (unconditional) results, it is informative to also report results conditional  on the stopping stage of the trial. Table 4 shows the simulation results conditional on stopping early at stage 1, with the true success rates (𝑝𝑝 , 𝑝𝐶𝐸 ) on the two arms again equal to the overall observed means in the MUSEC trial. With a probability of early stopping of 0.308, these results  are based on 3.08 × 104 simulation replicates, giving a Monte Carlo standard error of less than 0.0028 for the coverage and consistency results. 
 
Type of CI 
CI method 
Coverage 
Mean width (se) 
Consistency 
𝑷(𝑳(𝑿)
> 𝜽) 
𝑷(𝑼(𝑿)
< 𝜽) 
Standard/naive 
Wald test 
0.907 
0.225 (0.010) 
1.000 
0.093 
0.000 
Unconditional 
Exact 
0.930 
0.234 (0.011) 
1.000 
0.070 
0.000 
Repeated 
0.995 
0.334 (0.015) 
1.000 
0.005 
0.000 
Adjusted asymptotic 
0.930 
0.232 (0.011) 
1.000 
0.070 
0.000 
Parametric bootstrap 
0.820 
0.207 (0.010) 
1.000 
0.180 
0.000 
Conditional 
Exact 
0.970 
0.673 (0.331) 
0.179 
0.017 
0.013 
Restricted exact 
0.970 
0.242 (0.054) 
1.000 
0.017 
0.013 
Likelihood 
0.995 
0.992 (0.229) 
0.030 
0.005 
0.000 
Penalised likelihood 
0.993 
0.298 (0.017) 
1.000 
0.007 
0.000 
Table 4: Simulation results showing the performance of various CIs with 𝑝𝑝= 21/134 ≈0.157 and 𝑝𝐶𝐸= 42/143 ≈0.294, conditional on the trial stopping early at stage 1. There were 105 trial replicates. The probability  of stopping at stage 1 is 0.308. 
  Conditional on stopping at stage 1, the coverage of the standard CI is substantially below the  nominal, at less than 91%. The coverage of the unconditional exact and adjusted asymptotic  CIs also decreases to 93%, while the parametric bootstrap has the largest drop to only 82%. In  contrast, the RCI has a very conservative coverage of 99.5%. These results demonstrate that  even if unconditionally the unconditional CIs may have near nominal coverage, the conditional  coverage properties can be poor. In contrast, all of the conditional CI methods (as expected)  have coverage above the nominal 95%, with the conditional exact and restricted exact CIs  having conservative coverage of 97%, and the conditional likelihood and penalised likelihood  CIs having a very conservative coverage comparable to the RCI.    In terms of mean CI width, similar patterns are seen as for the unconditional results. Some  noticeable features are that the RCI has a substantially higher mean width than either the  conditional restricted exact or penalised likelihood CIs. The very large mean widths of the  conditional exact and conditional likelihood CIs are even more extreme conditional on stopping  at stage 1, with increases of +199% and +340%, respectively, compared with the mean width  of the standard CI. This agrees with the literature22,23 that does not recommend the use of these 
two methods when a group sequential trial stops early for benefit. The very large mean widths also correspond with very low consistencies of the test decision (which is always to reject the  null). In contrast, all other CI methods have 100% consistency (with this holding by design for the RCI and penalised likelihood CI). Finally, all CI methods except for the conditional exact  and restricted exact CIs have an upper coverage of zero. 
# Conditional on continuing to stage 2 
Table 5 shows the simulation results conditional on continuing to stage 2, with the true success rates (𝑝𝑝 , 𝑝𝐶𝐸 ) on the two arms again equal to the overall observed means in the MUSEC trial. With a probability of continuing to stage 2 of 69.2%, these results are based on 6.92 × 104 simulation replicates, giving a Monte Carlo standard error of less than 0.0019 for the coverage and consistency results. 
 
Type of CI 
CI method 
Coverage 
Mean width (se) 
Consistency 
𝑷(𝑳(𝑿)
> 𝜽) 
𝑷(𝑼(𝑿)
< 𝜽) 
Standard/naive 
Wald test 
0.962 
0.193 (0.008) 
0.984 
0.000 
0.038 
Unconditional 
Exact 
0.962 
0.200 (0.010) 
0.998 
0.001 
0.038 
Repeated 
0.963 
0.198 (0.008) 
1.000 
0.000 
0.036 
Adjusted asymptotic 
0.951 
0.193 (0.008) 
0.978 
0.000 
0.049 
Parametric bootstrap 
0.973 
0.211 (0.009) 
0.983 
0.000 
0.027 
Conditional 
Exact 
0.947 
0.258 (0.040) 
0.978 
0.027 
0.025 
Restricted exact 
0.947 
0.220 (0.031) 
0.978 
0.027 
0.025 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ebb8/ebb8b659-400f-4da9-8ba7-9e50faa1fd0c.png" style="width: 50%;"></div>
Table 5: Simulation results showing the performance of various CIs with 𝑝𝑝= 21/134 ≈0.157 and 𝑝𝐶𝐸= 42/143 ≈0.294, conditional on the trial continuing to stage 2. There were 105 trial replicates. The probability o continuing to stage 2 is 0.692. 
Conditional on continuing to stage 2, the standard CI and all the unconditional CIs now have  slightly conservative coverage of around 96 - 97% (with the exception of the adjusted  asymptotic CI which achieves the nominal 95% coverage). In contrast, the coverage of conditional exact and restricted exact CIs is just below the nominal 95%. The conditional  likelihood CI (which is the same as the penalised likelihood CI in this case) has the most  conservative coverage of 98.5%. What is noticeable is that the mean widths of the conditional exact and conditional (penalised) likelihood CIs are much lower than the mean widths  conditional on early stopping at stage 1, with increases of +14% and +34% compared with the standard CI. 
In terms of consistency, there is a drop of up to around 2% (compared with the 100%  consistency conditional on stopping at stage 1) for the standard CI, conditional restricted exact and all unconditional CIs, with the exception of the RCI which maintains 100% consistency and the unconditional exact CI which has almost 100% consistency (with inconsistency again  caused by the normal approximation). Meanwhile, the consistency of the conditional exact and conditional likelihood CIs are around 98 - 99%, compared with only 18% and 3% consistency, respectively,  conditional on stopping at stage 1. Finally, all CI methods except for the  conditional exact and restricted exact CIs now have a lower coverage of zero. 
If however we run the simulations with a higher true success rates for 𝑝𝐶𝐸 i.e., 𝑝𝐶𝐸= 42/143 + 0.08 ≈0.374 so that the probability of continuing to stage 2 is only 0.239, the coverage results  conditional on continuing to stage 2 look rather different. Table 6 shows these simulation  results, with the Monte Carlo standard error for the coverage and consistency results less than  0.0032.  
 
Type of CI 
CI method 
Coverage 
Mean width (se) 
Consistency 
𝑷(𝑳(𝑿)
> 𝜽) 
𝑷(𝑼(𝑿)
< 𝜽) 
Standard/naive 
Wald test 
0.897 
0.202 (0.007) 
0.994 
0.000 
0.103 
Unconditional 
Exact 
0.902 
0.218 (0.012) 
0.997 
0.000 
0.098 
Repeated 
0.910 
0.209 (0.007) 
1.000 
0.000 
0.090 
Adjusted asymptotic 
0.895 
0.205 (0.007) 
0.992 
0.000 
0.105 
Parametric bootstrap 
0.959 
0.221 (0.008) 
0.995 
0.000 
0.041 
Conditional 
Exact 
0.945 
0.310 (0.043) 
0.989 
0.031 
0.024 
Restricted exact 
0.945 
0.203 (0.055) 
0.989 
0.031 
0.024 
(Penalised) likelihood 
0.987 
0.288 (0.019) 
0.996 
0.000 
0.013 
Table 6: Simulation results showing the performance of various CIs with 𝑝𝑝= 21/134 ≈0.157 and 𝑝𝐶𝐸= 42/143 + 0.08 ≈0.374, conditional on continuing to stage 2. There were 105 trial replicates. The probability of  continuing to stage 2 is 0.239. 
Table 6: Simulation results showing the performance of various CIs with 𝑝𝑝= 21/134 ≈0.157 and 𝑝𝐶𝐸= 42/143 + 0.08 ≈0.374, conditional on continuing to stage 2. There were 105 trial replicates. The probability of  continuing to stage 2 is 0.239. 
This time, the standard CI and all the unconditional CIs, including the RCI, now have coverage  substantially less than the nominal 95%, ranging from 89 - 91%. The exception is the  parametric bootstrap CI, which has a slightly conservative coverage (96%). In contrast, the  coverage of conditional exact and restricted exact CIs remains just below the nominal 95%,  while the coverage of the conditional (penalised) likelihood CI remains rather conservative  (almost 99%). Again these results demonstrate that unconditional CIs can have poor coverage  properties conditional on the stopping stage (regardless of whether the trial is stopped early or  continues). Tables showing the simulation results unconditionally and conditional on early  stopping at stage 1 for this choice of values for 𝑝𝑝 and 𝑝𝐶𝐸 can be found in Appendix A.2.   
# Simulation results across a range of values for 𝒑𝑪𝑬
The results given in Tables 4-6 above are only for a particular fixed value of 𝑝𝐶𝐸 and so we  now explore the performance of the various CI methods across a range of values of 𝑝𝐶𝐸 from  𝑝̂𝐶𝐸−0.07 ≈0.224 to 𝑝̂𝐶𝐸+ 0.14 ≈0.434, which corresponds to a probability of early  stopping ranging from 0.05 to 0.94, as seen in Figure 3 below. 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f4da/f4da4024-2d56-486c-96de-2934b861009a.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Probability of early stopping at stage 1 as the value of 𝑝𝐶𝐸 varies from 𝑝̂𝐶𝐸−0.07 ≈0.224 to 𝑝̂𝐶𝐸+ 0.14 ≈0.434 . The value of 𝑝𝑝= 21/134 ≈0.157. </div>
For each value of 𝑝𝐶𝐸 (22 in total), we ran N = 105 trial replicates for each of the CI methods, except for the standard CI where we ran N = 106 trial replicates in order to more accurately assess whether there was any undercoverage. For the CI methods requiring a bootstrap procedure (unconditional parametric bootstrap, conditional likelihood and conditional penalised likelihood), we again used B = 104 bootstrap samples. Figure 4 shows the overall (unconditional) coverage of the CI methods as a function of 𝑝𝐶𝐸. The shaded areas around the lines for the CI methods correspond to ±1.96 times the Monte Carlo standard error. The red dashed line denotes the nominal 95% coverage. 
Starting with the standard CI, the coverage is (just) below the nominal 95% for most of the range of 𝑝𝐶𝐸 except for 𝑝𝐶𝐸> 0.40 when it goes just above 95%. In contrast, the unconditional exact CI has coverage (just) above the nominal 95% for the whole range of 𝑝𝐶𝐸 , with the coverage becoming increasingly conservative (up to 96%) as 𝑝𝐶𝐸 increases. The RCI has rather conservative coverage for the whole range of 𝑝𝐶𝐸 , with the coverage also becoming increasingly conservative (>98%) as 𝑝𝐶𝐸 increases. The adjusted asymptotic CI has very similar undercoverage to the standard CI for 𝑝𝐶𝐸< 0.33 but then has increasingly conservative coverage as 𝑝𝐶𝐸 increases above 0.33. The unconditional parametric bootstrap CI has rather low coverage (<93%) for 𝑝𝐶𝐸< 0.32 but quickly switches to having increasingly conservative coverage for 𝑝𝐶𝐸> 0.34.  These results demonstrate that whether a particular CI method has the correct coverage can strongly depend on the true (unknown) underlying parameter values. The conditional CIs all have conservative coverage, but the conditional exact and conditional
restricted exact CIs have coverage close to the nominal (95-96%). In contrast, the conditional likelihood and penalised likelihood CI have rather conservative coverage (98-99%). 
restricted exact CIs have coverage close to the nominal (95-96%). In contrast, the conditiona likelihood and penalised likelihood CI have rather conservative coverage (98-99%).   
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8e6a/8e6a1618-9022-4004-a1ae-02ead283a9d0.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4: Coverage as the value of 𝑝𝐶𝐸 varies from 𝑝̂𝐶𝐸−0.07 ≈0.224 to 𝑝̂𝐶𝐸+ 0.14 ≈0.434 . The value of 𝑝𝑝= 21/134 ≈0.157. The shaded areas around the lines for the CI methods correspond to ±1.96 times the Monte  Carlo standard error. The red dashed line denotes the nominal 95% coverage. The restricted conditional and  conditional exact lines are almost completely overlapping. </div>
Figure 5 shows the (unconditional) mean width of the CI methods as a function of 𝑝𝐶𝐸. The unconditional exact, adjusted asymptotic and unconditional parametric bootstrap CIs all have similar mean width as the standard CI (within ±6%). The conditional restricted exact CI has a slightly higher mean width than the standard CI (between 8-13%), with the penalised likelihood CI having a mean width between 23-33% higher than the standard CI. In contrast, the conditional exact and conditional likelihood CI mean widths are dramatically larger than the standard CI, up to 106% and 203% larger, respectively. Finally, the RCI has a similar mean width as the standard CI for small values of 𝑝𝐶𝐸 but becomes substantially higher as 𝑝𝐶𝐸 increases (up to 43% larger). 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c95c/c95cba50-67e0-4358-8100-60b596a18ba1.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: CI width as the value of 𝑝𝐶𝐸 varies from 𝑝̂𝐶𝐸−0.07 ≈0.224 to 𝑝̂𝐶𝐸+ 0.14 21/134 ≈0.157. </div>
As before, it is informative to also report results conditional on the stopping stage of the trial. Figure 6 shows the coverage of the CI methods conditional on early stopping at stage 1 as a function of 𝑝𝐶𝐸. This time, for smaller values of 𝑝𝐶𝐸 the coverage of the standard CI, adjusted asymptotic CI, unconditional exact and unconditional bootstrap CI is very low. The standard CI coverage can even be less than 50%. Note though that the lowest coverages are also achieved when there are the lowest probabilities of actually stopping at stage 1. The coverage of the standard CI does go above the nominal 95% when 𝑝𝐶𝐸> 0.33. In contrast, all the conditional CIs and the RCI have conservative coverage throughout the range of 𝑝𝐶𝐸, with the conditional exact and restricted exact CIs being (much) closer to the nominal level of coverage compared to the RCI, conditional likelihood and penalised likelihood CI.  
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6629/66292da4-e98e-48af-93de-c59d3746795c.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Coverage conditional on early stopping at stage 1 as the value of 𝑝𝐶𝐸 varies from 𝑝̂𝐶𝐸−0.07 ≈0.224 to 𝑝̂𝐶𝐸+ 0.14 ≈0.434 . The value of 𝑝𝑝= 21/134 ≈0.157. The shaded areas around the lines for the CI methods correspond to ±1.96 times the Monte Carlo standard error. The red dashed line denotes the nominal 95% coverage. The unconditional exact and adjusted asymptotic lines completely overlap, as do the restricted conditional and conditional exact lines. The conditional likelihood and penalised likelihood lines are also very close together. </div>
Figure 6: Coverage conditional on early stopping at stage 1 as the value of 𝑝𝐶𝐸 varies from 𝑝̂𝐶𝐸−0.07 ≈0.224 to 𝑝̂𝐶𝐸+ 0.14 ≈0.434 . The value of 𝑝𝑝= 21/134 ≈0.157. The shaded areas around the lines for the CI methods correspond to ±1.96 times the Monte Carlo standard error. The red dashed line denotes the nominal 95% coverage. The unconditional exact and adjusted asymptotic lines completely overlap, as do the restricted conditional and conditional exact lines. The conditional likelihood and penalised likelihood lines are also very close together. 
Figure 7 shows the mean width of the CI methods conditional on early stopping at stage 1 as a function of 𝑝𝐶𝐸. There are similar patterns as for the unconditional results, except that the RCI now has substantially greater mean width than the standard CI across the range of 𝑝𝐶𝐸 values (consistently between 47-52% larger). Also, the extreme widths for the conditional exact and conditional likelihood CIs are even more striking for smaller values of 𝑝𝐶𝐸. This reflects the results in the literature that these CIs have poor properties conditional on early stopping, with much better properties seen with the conditional restricted exact and penalised likelihood CIs. 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/98f6/98f674fd-fc25-467c-8235-a1673ca042e7.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: CI width conditional on early stopping at stage 1 as the value of 𝑝𝐶𝐸 varies from 𝑝̂𝐶𝐸−0.07 ≈0.224 to 𝑝̂𝐶𝐸+ 0.14 ≈0.434 . The value of 𝑝𝑝= 21/134 ≈0.157. The unconditional exact and adjusted asymptotic  lines are almost completely overlapping.  </div>
Figure 8 shows the coverage of the CI methods conditional on continuing to stage 2 as a  function of 𝑝𝐶𝐸. For larger values of 𝑝𝐶𝐸 the coverage of the standard, adjusted asymptotic,  unconditional exact and unconditional bootstrap CI is very low. The coverage of the standard  CI can be below 70%. This time, the RCI is no longer conservative throughout the whole range  of larger values of 𝑝𝐶𝐸, with also a very low coverage for larger values of 𝑝𝐶𝐸 (as low as 73%).  These lowest coverages are achieved when there are the lowest probabilities of actually  continuing to stage 2. The coverage of the standard CI is above the nominal 95% when 𝑝𝐶𝐸< 0.33. The conditional likelihood CI (which is the same as the penalised likelihood CI) and the  RCI again have conservative coverage throughout the range of 𝑝𝐶𝐸. This time the conditional  exact and restricted exact CIs essentially match the nominal 95% level of coverage. 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/82e8/82e8f1c3-28f0-44d7-97d6-5312fdc63521.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: Coverage conditional on continuing to stage 2  as the value of 𝑝𝐶𝐸 varies from 𝑝̂𝐶𝐸−0.07 ≈0.224 t 𝑝̂𝐶𝐸+ 0.14 ≈0.434 . The value of 𝑝𝑝= 21/134 ≈0.157. The shaded areas around the lines for the CI method correspond to ±1.96 times the Monte Carlo standard error. The red dashed line denotes the nominal 95%  coverage. The conditional exact and restricted conditional lines are almost completely overlapping. </div>
Finally, Figure 9 shows the mean width of the CI methods conditional on continuing to stage  2 as a function of 𝑝𝐶𝐸. The adjusted asymptotic CI has a very similar mean width as the standard CI (within ±2%). This time though, the RCI also has a similar mean width as the standard CI,  only up to 4% greater. Meanwhile, the unconditional exact CI has an increasingly large mean  width compared with the standard CI as 𝑝𝐶𝐸 increases (up to 12% larger). The unconditional  bootstrap CI has a consistently higher mean width than the standard CI of between 7-9%. The mean width of the conditional restricted exact CI has an interesting pattern, being up to 15%  greater than the standard CI for 𝑝𝐶𝐸< 0.38, but then having a smaller mean width than the standard CI for 𝑝𝐶𝐸> 0.38, with even up to a 16% decrease. In contrast, the conditional exact  and conditional likelihood CI mean widths remain much larger than the standard CI, up to 64% and 46% larger, respectively. This implies that the penalised likelihood CI also has much larger widths than the standard CI conditional on continuing to stage 2 (as it is equal to the conditional likelihood CI). 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c44d/c44d8531-6055-4692-9289-57da104195fb.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: CI width conditional on continuing to stage 2 as the value of 𝑝𝐶𝐸 varies from 𝑝̂𝐶𝐸−0.07 ≈0.224 t 𝑝̂𝐶𝐸+ 0.14 ≈0.434 . The value of 𝑝𝑝= 21/134 ≈0.157. </div>
# Summary
Our aim with the simulation study was not primarily to suggest that one method is the ‘best’  overall and should be used in the MUSEC trial context, as which method is ‘best’ very much  depends on the trial aims and relative importance of relevant metrics of interest (including  coverage, consistency and CI width), see the following Sections. However, we can draw the  following general observations: 
erall and should be used in the MUSEC trial context, as which method is ‘best’ very much  pends on the trial aims and relative importance of relevant metrics of interest (including  verage, consistency and CI width), see the following Sections. However, we can draw the  llowing general observations:  ● There is often (but not always) the following trade-off in metrics: a high(er) coverage  implies high(er) mean CI width, and vice-versa. The trade-off between coverage and  consistency is much less clear.  ● While unconditional CIs can have good performance unconditionally, the conditional  performance (especially in terms of coverage) may be poor.  ● To guarantee conditional performance in terms of coverage, conditional CIs are a must.  However, this can come at the price of (very) wide CIs.  ● For conditional CIs, as has been discussed in the literature, the restricted conditional  exact CI is to be preferred to the conditional exact CI, and the penalised likelihood CI  is to be preferred to the conditional likelihood CI (particularly for trials that stop early  at stage 1). 
● There is often (but not always) the following trade-off in metrics: a high(er) coverage  implies high(er) mean CI width, and vice-versa. The trade-off between coverage and  consistency is much less clear.  ● While unconditional CIs can have good performance unconditionally, the conditional  performance (especially in terms of coverage) may be poor.  ● To guarantee conditional performance in terms of coverage, conditional CIs are a must.  However, this can come at the price of (very) wide CIs.  ● For conditional CIs, as has been discussed in the literature, the restricted conditional  exact CI is to be preferred to the conditional exact CI, and the penalised likelihood CI  is to be preferred to the conditional likelihood CI (particularly for trials that stop early  at stage 1). 
● Some of the undercoverage and inconsistency is driven by the quality of the (asymptotic) normal approximation used. However, this is not unique to ADs, with such issues also seen for CIs for binary endpoints in fixed trial designs17.  ● The only CI methods that guarantee consistency in all cases in this trial context are the RCI, and the penalised likelihood CI (when stopping at stage 1).  ● The simulations highlight the importance of looking at a wide(r) region of the parameter space, as some CI methods may perform well in some parts of the parameter space and not in others.  ● More ‘extreme’ conditional results can be seen as the probability of stopping at that stage gets closer to zero or one. 
# 5. Guidance: best practice for CIs in ADs 
In this Section, we give guidance on the choice and reporting of CIs for ADs. This builds on the relevant parts of the FDA guidance for ADs11 and the ACE guidance2,3, and closely follows the guidance for best practices for point estimation in ADs given by Robertson et al. (2023a,b)9,10. The choice of CIs should be considered throughout an adaptive trial, from the planning stage to the final reporting and interpretation of the results. Indeed, the design and analysis of an adaptive trial are closely linked, and should ideally go hand-in-hand. In what follows