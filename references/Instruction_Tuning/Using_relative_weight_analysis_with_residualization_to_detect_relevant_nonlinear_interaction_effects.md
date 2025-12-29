# Using relative weight analysis with residualization to detect relevant nonlinear interaction effects in ordinary and logistic regression
Maikol Solís* Carlos Pasquier†
Abstract
Relative weight analysis is a classic tool for detecting whether one variable or interaction in a model is relevant. In this study, we focus on the construction of relative weights for non-linear interactions using restricted cubic splines. Our aim is to provide an accessible method to analyze a multivariate model and identify one subset with the most representative set of variables. Furthermore, we developed a procedure for treating control, fixed, free and interaction terms simultaneously in the residual weight analysis. The interactions are residualized properly against their main effects to maintain their true effects in the model. We tested this method using two simulated examples.
# 1 Introduction
Traditional fit in ordinary or logistic regressions estimates the best set of parameters for a dataset. We executed another analysis in tandem, like hypothesis testing about the nullity of coefficients; estimation of confidence intervals; check the distribution of the residuals; or checking the variability of the fit along with the data. These checks are mandatory in modern statistics. We can also check their influence on the model depending on the number of variables in the data. Here, we refer to the capacity of the variables to significantly impact the output. Non-influential variables should be excluded from the analysis. This analysis helps simplify the problem, analyze the results better, and choose better decisions.
*Corresponding author. Universidad de Costa Rica, Escuela de Matemática, Centro de Investigación en Matemática Pura y Aplicada. email: maikol.solis@ucr.ac.cr †Universidad de Costa Rica, Escuela de Matemática, Centro de Matemática Pura y Aplicada. email: carlos.pasquier@ucr.ac.cr
Carlos Pasquier†
General solutions to this analysis were proposed. Some examples use principal component analysis, stepwise regression, lasso, ridge, or elastic net regression (Hastie, Tibshirani, and Friedman 2011). However, these can detect and remove non-influential variables, but they hinder the impact of removing or adding this variable with respect to the other variables. For a detailed measurement of the influence of each variable, we can mention measures such as zero-order correlations, standardized regression weights, and semi-partial correlations (Johnson and Lebreton 2004). However, if multicollinearity exists among the variables, these measures are inadequate. Other techniques, such as dominance analysis (Budescu 1993) and residual weight analysis, have arisen to capture the complexity of a model (Johnson 2000). The dominance analysis and residual weights analysis aim to estimate the variable’s importance with respect to the output when multiple correlated predictors exist (Lebreton, Ployhart, and Ladd 2004). In this paper, we focus on the residual weight analysis originally proposed by Johnson (2000). Using ordinary least squares (OLS) regression, the technique creates a new set of predictors that are orthogonal representations of the original ones. In this new space, the importance standard score for each variable is estimated. Then, the scores return to the original space of variables through a transformation matrix. The method OLS regression method uses assumptions such as homoscedasticity, linearity and residual normality. However, in areas such as sociology or psychology, the dependent variable is categorical, and thus a logistic approach is necessary. All the aforementioned assumptions are violated in this case. The work in Tonidandel and LeBreton (2010) proposed a solution to this problem by estimating residual weights for logistic regression. The solution creates a new orthogonal space and estimates the standardized scores with the standard deviation of the logistic response variable. With interactions, the work of LeBreton, Tonidandel, and Krasikova (2013) proposed residualizing the interaction terms to account for only the true effect among the variables. Thus, they executed a local regression where the dependent variable is the interaction term and the covariates are the main effects. Then, the residual error captures the pure effect between variables. Regarding the type of model we use (OLS or logistic), the mentioned techniques intrinsically depend on the underlying way we adjust the predictors. The usual linear regression works for most problems. However, if the data have nonlinear structures; the linearity assumption is insufficient. Therefore, it can be extended using a nonlinear smoother. In this work, we perform a relative weight analysis where the dependent variable is either continuous or binary using restricted splines to model the predictors. Even if we have a clear knowledge about the main effects of the model, the interactions remain an obscure part of the problem, making it difficult to match two covariates that provide the most possible information. Thus, we add the most relevant interactions terms to the regression to capture all the information not seen by the main effects. Our procedure includes all the interactions among the main effects and then searches for the best subset through
a stepwise selection process. Even if there are some criticisms of the stepwise technique (Miller 2002; Harrell 2015), we get accurate results. To build a flexible procedure adjusted to multiple needs, we will allow the use of control variables, fixed variables, free variables and pairwise interactions. The control variables are static and remain only as main effects; the fixed and free variables are used to create interactions. The difference is the former ones remain in the final model, while the later ones can be removed in the stepwise process. The remainder of this paper is organized as follows. Section 2 presents a review of the preliminaries on setting regression models with restricted splines. In Section 3 resides the core of the paper, where we explain all the details to build a procedure of the Relative Weight Analysis with residualization. We devote Section 4 to testing the algorithm capabilities using two simulated examples. Finally, in Section 5, we present the conclusions with some discussion about future lines of research.
# 2 Preliminaries
In a generalized linear model, we can link a set of input variables X = (X1, . . . , X to an output variable Y through function C. We also have a set of n observations for each d-tuple of variables. Formally, we state the model as,
C(Y|X) = β0 + β1X1 + · · · βdXd + ε.
The parameters β0, . . . , βd, are estimated by an iterative reweighed the least-squared procedure. For a detailed review of logistic regression; we refer the reader to Hastie, Tibshirani, and Friedman (2011). In this work, we will use two cases for the function C(·)
Gaussian: The output variable Y = Yg ∈R and we set the link function as C(Yg|X) = Yg, and the ε are independent and normally distributed.
Binomial: The output variable is classified as 0 or 1 (Y = Yb = {0, 1}). The link function is defined as C(Yb|X) = logit(P (Yb = 1|X)), where logit(u) = log(u/(1 −u)) is a function defined in [0, 1] →R. The noise variable ε has an independent logistic distribution.
In the following sections, and to simplify the notation, we will use the parameters named as β in different contexts without implying that they are the same. Each equation has exactly the parameters that we want to present, unless otherwise stated. Additionally, we will use only Y to denote either Yg or Yb. In numerical examples, we will retake the notation to differentiate each case.
Quantile level (qi)
k
q1
q2
q3
q4
q5
3
0.1
0.5
0.9
4
0.05
0.35
0.65
0.95
5
0.05
0.275
0.5
0.725
0.95
Table 1: Quantile levels (qi) defining the knot positions for cubic restricted spline functions.
# 2.1 Control, fixed, and free variables
We assume a generalized linear model with the form,
C(Y|X) = β0 + c ∑ i=1 γiQi + dF ∑ i=1 fi(Xi) + p ∑ i=dF+1 fi(Xi).
The model is formed by Q1, . . . , Qc control variables, f1(X1), . . . , fdF(XdF) fixed variables and fdF+1(XdF+1), . . . , fp(Xn) free variables. The functions f represents the smoother used for each variable. In the classic setting, for a variable X we put f(X) = β1X. However, to allow a parsimonious structure to the model; we should to introduce nonlinear functions. We use restricted splines to model the functions f in Equation (1) (Stone and Koo 1985). These functions are linear at the endpoints and require fewer parameters than classic cubic splines. We call k the number of knots used to define restricted spline functions. We identify three cases according to the desired level of smoothness for each variable. Theoretically, it is possible to set any arbitrary positive integer value of k having the form,
for l = 2, . . . , k −2,
The model’s complexity increases in the same direction as k. To keep a sane number of parameters, the recommendation is to use k equal to 3, 4 or 5. The exact positions of the ti’s are defined as P(X ≤ti) = qi with the qi’s defined in Table 1 for a given number of knots k.
(1)
# 3 Methodology
In this section, we will explain a series of steps to determine the most relevant variables in a model combining the classic relative weight analysis, nonlinear smoother with restricted cubic spline and residualization of interactions. We can summarize our procedure in three steps:
In this section, we will explain a series of steps to determine the most relevant variables in a model combining the classic relative weight analysis, nonlinear smoother with restricted cubic spline and residualization of interactions. We can summarize our procedure in three steps: 1. Select the best submodel with given a set of control, free, fixed and interaction terms. 2. Residualize all the interaction terms to remove the effects of the main variables and keep only the pure interaction effect. 3. Apply the relative weight analysis to detect which variables are the most significant. The steps are explained thoroughly in the following sections.
3. Apply the relative weight analysis to detect which variables are the most significant.
The steps are explained thoroughly in the following sections.
# 3.1 Model selection with interactions
Given a model (Gaussian or logistic) and once defined the structure of the variables (linear or nonlinear), we will define the interactions pairwise between the variables. We will use restricted interaction multiplication. The restricted interaction will remove doubly nonlinear terms, allow us to remove no essential terms. For example, assume we decide to model A and B with a 3 knot spline. The interaction between A and B is denoted by
fA(A) × fB(B) = β1 AB + β2 AS(3) 1 (B) + β3 S(3) 1 (A)B,
for some constants β1, β2 and β3. A similar pattern follows the 4 knots case. Setting a full set of interaction with p variables, we should fit 1 2(p −1)(p) interactions. For a model with k-knots spline functions, we will need to fit 1 + kp + k 2(p −1)(p) distinct parameters. For example, if we have p = 10 variables, for a full interaction model with a 3-knots case, we will need to fit 166 parameters. If we decide to use the 4-knots case, the number increases to 221. Any model with such many parameters is inadequate. The overfitting leads to erroneous results, especially on small samples. To solve this issue, we opted by selecting a submodel from the full model. We use the classic stepwise method based on the change of Bayesian Information Criterion (BIC). Recall that BIC = −2 log(L) + ln(n)v. Here L represents the log-likelihood of the submodel and v the variables used to fit it. In the procedure, we search to minimize the BIC. The factor ln(n)v strongly penalize large models unless the value −2 log(L) is less than 2v. In our context, the BIC presents some advantages over the Akaike Information Criterion (AIC). The AIC is estimated with AIC = −2 log(L) + 2v. The criterion penalizes the submodel to exclude unnecessary variables as well as BIC using the factor 2v. However, it smooths the selection, allowing more
variables inside the final model. Such models are appropriate for prediction instead to be parsimonious (see Dziak et al. 2020). We focus on the method in the inference of the most relevant features of the data, considering the BIC a better value for the stepwise selection process. In our implementation, we start with a model using only main effects. Then, we continue adding or removing main effects or interactions until the algorithm cannot improve the BIC value. The implementation of the procedure was taken from the package MASS (Venables and Ripley 2002) using the function stepAIC. To use the BIC, as mentioned before, we set the parameter k = log(n). The function allows to include a parameter scope that consists of a list with lower and upper models. Given the structure of Equation (1), we define the lower and upper models as,
The implementation include in the lower model only the control and fixed variables. The upper model contains all the main effects and all the possible interactions. After selecting the most relevant variables on the model we remain with the subsets fi1, . . . , fidM of main effects and fi1(Xi1) × fj1(Xj1), . . . fidI (XidI ) × fjdI (XjdI ) of interactions. Here dM and dI represent the number of main effects and interactions selected and {is}dM s=1 and {(is, js)}dI s=1 are their respective indices. The final model after the stepwise selection is
Note finally that dM ≥dF and dI ≥0. If dM = dF then the model has only main effects. If dI = 0 then none of the interactions were added. For every interaction added, the main effects are also added automatically to preserve the hierarchical principle.
# 3.2 Residualized relative importance
Interactions play a key role in this work. We want to know if each interaction included, adds relevant information to the model. Otherwise, it is negligible compared to its main effects. Here, the interaction contains little information
(2)
blurring other results in the model. The objective is to separate the relevant effects from the main ones and interactions. Notice that in the case of simple linear interactions like Xi × Xj, there are three effect types
1. The effect solely from Xi. 2. The effect solely from Xj. 3. The effect solely from the
3. The effect solely from the interaction of Xi × Xj.
Interactions with restricted splines, are handle it equally. Except that the effects are more diffused across the terms. For example, for a 3 knot spline, the interaction between Xi and Xj is
fi(Xi) × fj(Xj) = β1XiXj + β2XiS(3) 1 (Xj) + β3S(3) 1 (Xi)Xj
The three terms contain mixed information about Xi and Xj. Therefore, if we apply the relative weight analysis to this interaction, the different effects are blurred. By controlling first by the other variables, we isolate the pure interaction effect. Suppose that we have a simpler model, where f1 and f2 are 3-knots restricted splines,
Logit(P(Y = 1)) = β0
+ β1X1 + β2S(3) 1 (X1) + β3X2 + β4S(3) 1 (X2) + β5X1X2 + β6X1S(3) 1 (X2) + β7S(3) 1 (X1)X2 + ǫ
The procedure proposed in LeBreton, Tonidandel, and Krasikova (2013) is the following:
1. Replace the higher order term with the residual obtained after regress the interaction with respect to their main effects.
X1X2 = ˜β1X1 + ˜β2X2 + rX1X2 X1S(3) 1 (X2) = ˜β3X1 + ˜β4S(3) 1 (X2) + rX1S(3) 1 (X2) S(3) 1 (X1)X2 = ˜β5S(3) 1 (X1) + ˜β6X2 + rS(3) 1 (X1)X2
Here the variables r represent the residual of the regressions. We perform the regression without an intercept to capture the full effects from the residual r12.
C(Y|(X1, X2)) = β0
1  + β5rX1X2 + β6rX1S(3) 1 (X2) + β7rS(3) 1 (X1)X2.
# It means that the interactions are replaced by the residuals of the step before. We denote the residualized interaction term as
�f1(X1) × �f2(X2) = β5rX1X2 + β6rX1S(3) 1 (X2) + β7rS(3) 1 (X)X
�f1(X1) × �f2(X2) = β5rX1X2 + β6rX1S(3) 1 (X2) + β7rS(3) 1 (X1)X2
The main advantage of residualizing the interaction effects, we create a new set of interaction variables uncorrelated with their main effects. The procedure separates the true synergy between the variables and the main effects. However, the effects could be correlated with each other. We apply this algorithm to all the restricted interactions of the reduced model. Therefore, the RWA was applied for the interactions over the residuals of an intermediate linear regression between the main effect.
# 3.3 Relative Weight Analysis
Relative Weight Analysis is tool used to separate the relevance of each variable or interaction. This technique creates a new set of predictors with are an orthogonal representation of the observed predictors (with one-to-one correspondence). Estimate the influence on this new space and then return the results to the original space. This way the problem presented with correlated models disappears given that the importance estimation occurs in this new space. We are interested in the importance of the main effects versus the interactions. The design matrix can have multiple patterns. For example, the right side of equation (3). Here a mix of linear and nonlinear variables is present. We are interested in determining the effect of the whole variables instead of their components. Therefore, we will define the matrix X as
(3)
Therefore, the matrix D contains the control variables and all the effects and interactions resulting from the selection model procedure. Also, the interactions contain only information from the pure synergy between the variables due to the residualization. The algorithm to estimate the relative weights is the following:
1. Start with design matrix D. 2. Standardize the columns of D.
3. Estimate the singular value decomposition D = A∆B⊤, wher
(a) A is the matrix of eigenvectors associated to DD⊤. (b) B is the matrix of eigenvectors associated to D⊤D. (c) ∆is the diagonal matrix of singular values, which is equivalent to the square root of eigenvalues of D⊤D.
4. Create the orthogonal version Z = AB⊤of D using the SVD decompo sition. The columns Z are one-to-one orthogonal representations of th columns of D.
6. Estimate the fully standardized coefficient for the model C(Y|Z) = βZ. However, depending on the link function the steps differ.
Gaussian: C(Y|X) = Y. (a) Estimate the standardized ordinary least square coefficient of Z with the formula β∗= (Z⊤Z)−1Z⊤Y
Logistic: C(Y|X) = logit(P (Y = 1|X)). (a) Obtain the unstandardized coefficients b of the model against the matrix Z. Here the b could be calculated using a least square procedure. (b) Compute R2 L which is.
Alternatively, regress Y against ˆY and recover the R2. (c) Estimate the standard deviation of the logit prediction, slogit( ˆY), where logit( ˆY), = ln[ ˆY/(1 −ˆY)]
# (d) Estimate the fully standardized coefficients
β∗= (b) (sZ) (RL) / � slogit( ˆY) � Here sZ is the standard deviation of each column of Z. 7. Create the link transformation Λ∗= (Z ′Z)−1Z ′X. 8. Estimate the relative weights with
β∗= (b) (sZ) (RL) / � slogit( ˆY) �
β∗= (b) (sZ) (RL) / � slogit( ˆY) �
Here sZ is the standard deviation of each column of Z. 7. Create the link transformation Λ∗= (Z ′Z)−1Z ′X. 8. Estimate the relative weights with
7. Create the link transformation Λ∗= (Z ′Z)−1Z ′X. 8. Estimate the relative weights with
The procedure projects the original columns of D to orthonormal space called Z with a correspondence one-to-one in their columns. In this new space, the fully standardized coefficients of the model C(Y|Z) = βZ. Those are named as β∗2. A particular property is β∗2 are the relative importance of the columns in Z. Given that Z is orthonormal, the weights refer to a uncorrelated variables set. To return those weights onto the original dimension space, we estimate a transformation matrix called Λ. The matrix Λ transforms the β∗2 into the original scale D. The sum of the ǫ are equal to the R2 O or R2 L depending on the case. Finally, each variable has associated a percentage contribution to the explained variance.
# 4 Results
We will show the results of our algorithm. To test the algorithm, we will generate a sample of n = 3000 of random variables determined by each example. In the simulated examples, we tested the capability of each method to determine the most relevant depending on if we are in the Gaussian or binomial case. We follow these steps to generate two outputs variable Yg and their respective dichotomized version Yb, with a model with d input variables X1, . . . , Xd: 1. Generate a variable Yg with the functional form of the variable. For example, Yg = X1 + X2. 2. Apply the inverse of the logistic function, p(L) = logit−1(Yg). 3. Generate a Bernoulli random variable ({0, 1}) with probability outcome p(Yg). 4. Repeat this procedure for each element of the sample. Call this vector Yb. The described step will generate a random variable Bernoulli Yb with values {0, 1} and generate with the underlying model given by Yg. For testing, we will use three classic models: The Ishigami model, the gmodel, and a syntethic model of 15 variables by Oakley and O’Hagan (2004).
# 4.1 Ishigami function
The Ishigami function is classic to test the sensitivity and reliability variables in models. It presents a strong non-linearity and non-monotonicity. The order of the variables in terms of relevance are X2, X1 and the interaction X1 with X3. All other variables and interaction have zero theoretical relevance’s. The form of the model is
where Xi ∼Uniform(−π, π) for i = 1, 2, 3, a = 7 and b = 0.1. We also set X4 −X8 ∼Uniform(−π, π) to add uncorrelated noise to the model. In the work of Ishigami and Homma (1990), they split the model’s variance to determine the relevance of each predictor. They concluded in this model that the percentage of variance due to X1, X2 and the interaction X13 are 31.38%, 44.24% and 24.36% respectively. The other effects have 0% on their contribution to the variance. We applied our procedure using the Yg and Yb cases. Table 2 presents the results for both cases. Notice how the variables X2, X1 and the interaction, X1 × X3 are captured correctly. For the three mentioned effects, we choose a restricted cubic spline with 5-knots. Recall that the model tested all the main effects and cross terms (X1 × X2, X1 × X3, X1 × X4, and so on). However, as we are using the BIC as a selection criterion, all the non-relevant terms were removed by the large penalty ln(n) in the criterion. In terms of goodness-of-fit, the R2 O = 0.8308 while R2 L = 0.4654. The results are expected. The Ishigami model is continuous and from it, we create two outputs Yg and Yb. In the binary case we loss power to predict the variance given the change of the dependent variable. Even in the case, active effects were detected.
Variable
Weight
Type
Continuous output Yg
X1
34.52%
Free
X2
48.27%
Free
X3
0.09%
Free
X1 × X3
17.12%
Interactions
Binary output Yb
X1
31.33%
Free
X2
53.25%
Free
X3
2.90%
Free
X1 × X3
12.52%
Interactions
Table 2: Relative weights for the Ishigami model when using the continuous output Yg and binary output Yb. We use 5-knots restricted cubic splines as smoother. To present the capabilities of the algorithm, we set two disfavorable scenarios for the Ishigami model. In these settings we will use only Yg.
Table 2: Relative weights for the Ishigami model when using the continuous output Yg and binary output Yb. We use 5-knots restricted cubic splines as smoother.
To present the capabilities of the algorithm, we set two disfavorable scenarios for the Ishigami model. In these settings we will use only Yg.
First, we set as fixed the variables X4, X5, X6, X7 and X8. This setting will cause that the model is forced to fit 5 noisy variables with little information for Yg. The model fit all the possible interactions, and removes those that worsen BIC. However, the variables X4 −X8 will remain in all models. The other scenario is when use X3 a control variable. Here, the variable X3 is in all models, but none interaction contain X3. This setting is useful when the study requires controlling by some variables, but you don’t want any further interference of it (e.g., gender, age, smoke, etc.). Table 3 presents the values on the mentioned scenarios. In the upper part of the table we notice how the inclusion of the non-important variables does not affect the weights compared with Table 2. In fact, all variables X4 −X8 present values between 0.04%-0.09%. If needed, the analyst can remove variables with weights in this order of magnitude without losing any inference power. The lower part presents the case when we control X3 in the model. Notices how the model relies on the weights of X1 and X2 because we restricted the use of the interaction X1 × X3. In this case
Variable
Weight
Type
Fixed X4, X5, X6,
X7 and X8
X1
35.57%
Free
X2
48.12%
Free
X3
0.06%
Free
X4
0.06%
Fixed
X5
0.05%
Fixed
X6
0.04%
Fixed
X7
0.09%
Fixed
X8
0.05%
Fixed
X1 × X3
15.96%
Interaction
Controlled X3
X1
42.69%
Free
X2
57.16%
Free
X3
0.15%
Control
Table 3: Relative weights when fixing the variables X4, X5, X6, X7 and X8 (upper) and when controlling by X3 (lower). All the results were estimated with the continuous output Yg
# 4.2 The Moon function
The work of Moon (2010) and Moon, Dean, and Santner (2012) established an algorithm to detect active and inactive variables on complex configurations. They propose a 20-dimensional function with 5 active main effects X1, X7, X12, X18, X19 and 4 active effects X1 × X18, X1 × X19 X7 × X12 and the quadratic term X2 19.
Assuming Xi ∼Uni f(0, 1), i = 1, . . . , 20, the explicit form of the model is, Ybase g = −19.71X1X18 + 23.72X1X19 −13.34 + X2 19 + 28.99X7X12 + small terms. They studied another complex configuration, including or removing the small terms and amplification of the active effects. To test if the procedure detects the relevant variables, we use the version where all the active effects were tripled while the small terms remain constant, YC3 g = −59.13X1X18 + 71.16X1X19 −40.02X2 19 + 86.97X7X12 + small terms. The small terms part consist of 189 elements between main, quadratic and interaction terms with low impact on the Yg. The coefficients can be found in Moon (2010) (Table 3.11). The aim setting Ybase g and YC3 g is to compare the changes on the relative weights in disfavorable and favorable scenarios. Table 4 presents the result using our algorithm with Ybase g and YC g 3. We estimated the absolute difference between both values to compare their change. The relative weights for Ybase g give significantly weight to non-important variables like X3 or X15. The behavior is due to the variable Ybase g presents a more blurred information about the true active effects. However, the effects X1 × X18, X1 × X19, X7 × X12 are well represented in the model. s With YC3 g all the non-important variables decrease their value. The true active ones increase their participation greatly. The variables X7, X12, X18 and X7 × X12 increase more than 10 percentage points with respect to the other model. Other terms like X1, X18, X1 × X18 and X1 × X19, increase in a small amount. The case for the quadratic term with X19 is interesting. Its value is low in both scenarios. However, recall that we are using a restricted cubic spline with 5 knots to fit the mains and interaction effects. Additionally, given the residualization step, most of the participation of X19 could be represented through X1 × X19 the residualization process to extract all the effects from the interactions to the main present lower, the effect We observe how the residual weights are more predominant for the C3 case than the base. The result is expected because small terms are stronger in the base case than in the C3.
# 5 Conclusion
In this paper, we explore a method to detect relevant variables using the relative weight analysis technique. The main contribution in our algorithm is the residualization of the interactions to capture the true effect of them. In a classic setting, the relative weights for the main effects and interactions are evaluated as is, without considering the real nature of the latter. In other words the interaction of two variables contains information on each variable separately and the remaining belongs to the true interaction effect. In this work, we emphasize
Relative weights
Variable
Ybase
g
YC3
g
∆(p.p.)
X1
1.26%
1.38%
0.13
X2
1.21%
0.13%
-1.08
X3
16.30%
2.97%
-13.33
X4
2.51%
0.36%
-2.15
X5
3.24%
0.56%
-2.68
X6
0.49%
0.05%
-0.44
X7
1.25%
12.21%
10.97
X8
1.40%
0.22%
-1.17
X9
1.97%
0.26%
-1.71
X12
2.20%
14.29%
12.10
X13
0.95%
0.13%
-0.81
X14
3.27%
0.71%
-2.57
X15
11.71%
1.69%
-10.02
X16
5.05%
0.95%
-4.10
X17
4.25%
0.56%
-3.69
X19
0.83%
2.62%
1.79
X20
3.25%
0.51%
-2.74
X18
10.19%
11.42%
1.24
X1 × X18
9.00%
10.18%
1.18
X1 × X19
6.77%
8.32%
1.56
X7 × X12
12.92%
30.46%
17.55
Table 4: Relative weights for the Moon model using the variable Ybase g and YC3 g . The absolute difference between both weights (∆) is estimated in percentage points (p.p.). We use 5-knots restricted cubic splines as smoother.
this technique following the works of Johnson (2000), Tonidandel and LeBreton (2010), and LeBreton, Tonidandel, and Krasikova (2013). One aim was to create a flexible algorithm beyond to the classic linear model. To this end, we include the restricted cubic spline as smothers to determine the relationship between the inputs and outputs. This feature allows us to detect nonlinear patterns in the data, even in complex settings. More work should be done to make the procedure flexible to a wide set of cases. We can find packages R performing Residual Weight Analysis like rwa (Chan 2020) or flipRegression (Displayr 2021). They rely on the analysis of main effects. In this study, we implemented a version that allows the user to set control, fixed, free and interactions in model. Besides, the predictors are modeled using restricted spline smothers. In this context, the analysis should be more accurate specially if the phenomenon is highly nonlinear. The great advantage with respect to other techniques is the analysts can include some variable in the model even if they are non-important. Or they can compare the weight
between main effects on interactions to choose a particular model. The stepwise procedure produced effective results, even if they are some points against it (e.g. Miller 2002; Harrell 2015). Other techniques to model selection were also considered like elastic nets (Zou and Hastie 2005) or PLS regression (Martens 2001; Mevik and Wehrens 2007). They represent interesting lines of research to follow in the future. In this study, we considered the implementation because of its simplicity. Stepwise regression combined with the Bayesian Information Criterion (BIC), is a fast and effective method to create the most relevant and simpler model. The BIC cut all the non-relevant variables in the first steps allowing the model to include those interactions adding some real value to the model. Other information criteria (Dziak et al. 2020) can be explored in the future. Finally, even if the results establish that our procedure identify correctly the relevant parts of our models, a proper validation must be done to check the dispersion of the relative weights. If we set a configuration over different realizations, the questions remaining are if the selected variables keep constant and how much the relative weights differ from each other. A proper validation of the results are beyond the scope of this paper, but, the work of Tonidandel, LeBreton, and Johnson 2009 conducted an extensive study of simulation with bootstrap confidence intervals. The implementation of the technique will be a priority for future developments.
# Declarations
# Funding
The authors acknowledge the financial support from Escuela de Matemática de la Universidad de Costa Rica, through CIMPA, Centro de Investigaciones en Matemática Pura y Aplicada through the projects 821–B8-A25.
# Code availability
All the calculations in this package were made using an own R-package residualrwa. The package is published on github on this address https://github.com/maikol-solis/residualrwa
All the calculations in this package were made using an own R-package residualrwa. The package is published on github on this address https://github.com/maik
# Conflict of interest
On behalf of all authors, the corresponding author states that there is no conflict of interest.
# References
Budescu, David V. (1993). “Dominance Analysis: A New Approach to the Problem of Relative Importance of Predictors in Multiple Regression.” en. In: Psychological Bulletin 114.3, pp. 542–551. ISSN: 1939-1455, 0033-2909. DOI: 10.1037/0033-2909.114.3.542. Chan, Martin (Nov. 2020). Rwa: Perform a Relative Weights Analysis. Displayr (Mar. 2021). flipRegression. Dziak, John J et al. (Mar. 2020). “Sensitivity and Specificity of Information Criteria”. In: Briefings in Bioinformatics 21.2, pp. 553–565. ISSN: 1477-4054. DOI: 10.1093/bib/bbz016. Harrell, Frank E. (2015). Regression Modeling Strategies: With Applications to Linear Models, Logistic and Ordinal Regression, and Survival Analysis. en. Springer Series in Statistics. Cham: Springer International Publishing. ISBN: 978-3319-19424-0 978-3-319-19425-7. DOI: 10.1007/978-3-319-19425-7. Hastie, Trevor, Robert Tibshirani, and Jerome Friedman (Dec. 2011). The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition. Vol. 1. Springer. ISBN: 978-0-387-84857-0. Ishigami, T. and T. Homma (Dec. 1990). “An Importance Quantification Technique in Uncertainty Analysis for Computer Models”. In: [1990] Proceedings. First International Symposium on Uncertainty Modeling and Analysis, pp. 398– 403. DOI: 10.1109/ISUMA.1990.151285. Johnson, Jeff W. (Jan. 2000). “A Heuristic Method for Estimating the Relative Weight of Predictor Variables in Multiple Regression”. In: Multivariate Behavioral Research 35.1, pp. 1–19. ISSN: 0027-3171. DOI: 10.1207/S15327906MBR3501_1. Johnson, Jeff W. and James M. Lebreton (July 2004). “History and Use of Relative Importance Indices in Organizational Research”. en. In: Organizational Research Methods 7.3, pp. 238–257. ISSN: 1094-4281. DOI: 10.1177/1094428104266510. Lebreton, James M., Robert E. Ployhart, and Robert T. Ladd (July 2004). “A Monte Carlo Comparison of Relative Importance Methodologies”. en. In: Organizational Research Methods 7.3, pp. 258–282. ISSN: 1094-4281. DOI: 10.1177/10944 LeBreton, James M., Scott Tonidandel, and Dina V. Krasikova (July 2013). “Residualized Relative Importance Analysis”. In: Organizational Research Methods 16.3, pp. 449–473. ISSN: 1094-4281. DOI: 10.1177/1094428113481065. Martens, Harald (Oct. 2001). “Reliable and Relevant Modelling of Real World Data: A Personal Account of the Development of PLS Regression”. en. In: Chemometrics and Intelligent Laboratory Systems. PLS Methods 58.2, pp. 85– 95. ISSN: 0169-7439. DOI: 10.1016/S0169-7439(01)00153-8. Mevik, Björn-Helge and Ron Wehrens (Jan. 2007). “The Pls Package: Principal Component and Partial Least Squares Regression in R”. en. In: Journal of Statistical Software 18.1, pp. 1–23. ISSN: 1548-7660. DOI: 10.18637/jss.v018.i02. Miller, Alan (2002). Subset Selection in Regression. en. Moon, Hyejung (2010). “Design and Analysis of Computer Experiments for Screening Input Variables”. PhD thesis. USA: Ohio State University.
Budescu, David V. (1993). “Dominance Analysis: A New Approach to the Problem of Relative Importance of Predictors in Multiple Regression.” en. In: Psychological Bulletin 114.3, pp. 542–551. ISSN: 1939-1455, 0033-2909. DOI: 10.1037/0033-2909.114.3.542. Chan, Martin (Nov. 2020). Rwa: Perform a Relative Weights Analysis. Displayr (Mar. 2021). flipRegression. Dziak, John J et al. (Mar. 2020). “Sensitivity and Specificity of Information Criteria”. In: Briefings in Bioinformatics 21.2, pp. 553–565. ISSN: 1477-4054. DOI: 10.1093/bib/bbz016. Harrell, Frank E. (2015). Regression Modeling Strategies: With Applications to Linear Models, Logistic and Ordinal Regression, and Survival Analysis. en. Springer Series in Statistics. Cham: Springer International Publishing. ISBN: 978-3319-19424-0 978-3-319-19425-7. DOI: 10.1007/978-3-319-19425-7. Hastie, Trevor, Robert Tibshirani, and Jerome Friedman (Dec. 2011). The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition. Vol. 1. Springer. ISBN: 978-0-387-84857-0. Ishigami, T. and T. Homma (Dec. 1990). “An Importance Quantification Technique in Uncertainty Analysis for Computer Models”. In: [1990] Proceedings. First International Symposium on Uncertainty Modeling and Analysis, pp. 398– 403. DOI: 10.1109/ISUMA.1990.151285. Johnson, Jeff W. (Jan. 2000). “A Heuristic Method for Estimating the Relative Weight of Predictor Variables in Multiple Regression”. In: Multivariate Behavioral Research 35.1, pp. 1–19. ISSN: 0027-3171. DOI: 10.1207/S15327906MBR3501_1 Johnson, Jeff W. and James M. Lebreton (July 2004). “History and Use of Relative Importance Indices in Organizational Research”. en. In: Organizational Research Methods 7.3, pp. 238–257. ISSN: 1094-4281. DOI: 10.1177/1094428104266510 Lebreton, James M., Robert E. Ployhart, and Robert T. Ladd (July 2004). “A Monte Carlo Comparison of Relative Importance Methodologies”. en. In: Organizational Research Methods 7.3, pp. 258–282. ISSN: 1094-4281. DOI: 10.1177/1094 LeBreton, James M., Scott Tonidandel, and Dina V. Krasikova (July 2013). “Residualized Relative Importance Analysis”. In: Organizational Research Methods 16.3, pp. 449–473. ISSN: 1094-4281. DOI: 10.1177/1094428113481065. Martens, Harald (Oct. 2001). “Reliable and Relevant Modelling of Real World Data: A Personal Account of the Development of PLS Regression”. en. In: Chemometrics and Intelligent Laboratory Systems. PLS Methods 58.2, pp. 85– 95. ISSN: 0169-7439. DOI: 10.1016/S0169-7439(01)00153-8. Mevik, Björn-Helge and Ron Wehrens (Jan. 2007). “The Pls Package: Principal Component and Partial Least Squares Regression in R”. en. In: Journal of Statistical Software 18.1, pp. 1–23. ISSN: 1548-7660. DOI: 10.18637/jss.v018.i02. Miller, Alan (2002). Subset Selection in Regression. en. Moon, Hyejung (2010). “Design and Analysis of Computer Experiments for Screening Input Variables”. PhD thesis. USA: Ohio State University.
Moon, Hyejung, Angela M. Dean, and Thomas J. Santner (Nov. 2012). “TwoStage Sensitivity-Based Group Screening in Computer Experiments”. en. In: Technometrics 54.4, pp. 376–387. ISSN: 0040-1706,1537-2723. DOI: 10.1080/00401706.2012.725994. Oakley, Jeremy E. and Anthony O’Hagan (Aug. 2004). “Probabilistic Sensitivity Analysis of Complex Models: A Bayesian Approach”. en. In: Journal of the Royal Statistical Society: Series B (Statistical Methodology) 66.3, pp. 751–769. DOI: 10.1111/j.1467-9868.2004.05304.x. Stone, Charles J and Cha-Yong Koo (1985). “Additive Splines in Statistics”. In: Statistical Computing Section,Proceedings of the American Statistical Association, pp. 45–48. Tonidandel, Scott and James M. LeBreton (2010). “Determining the Relative Importance of Predictors in Logistic Regression: An Extension of Relative Weight Analysis”. In: Organizational Research Methods 13.4, pp. 767–781. ISSN: 10944281. DOI: 10.1177/1094428109341993. Tonidandel, Scott, James M. LeBreton, and Jeff W. Johnson (Dec. 2009). “Determining the Statistical Significance of Relative Weights”. en. In: Psychological Methods 14.4, pp. 387–399. ISSN: 1939-1463,1082-989X. DOI: 10.1037/a0017735. Venables, W. N. and B. D. Ripley (2002). Modern Applied Statistics with S. Ed. by J. Chambers et al. Statistics and Computing. New York, NY: Springer New York. ISBN: 978-1-4419-3008-8978-0-387-21706-2. DOI: 10.1007/978-0-387-21706-2. Zou, Hui and Trevor Hastie (2005). “Regularization and Variable Selection via the Elastic Net”. en. In: Journal of the Royal Statistical Society: Series B (Statistical Methodology) 67.2, pp. 301–320. ISSN: 1467-9868. DOI: 10.1111/j.1467-9868.2005.00503.x.
