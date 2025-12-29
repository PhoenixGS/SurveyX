# Deep Learning with Nonsmooth Objectives
Vinesha Peiris, Vera Roshchina, Nadezda Sukhorukova
Abstract
We explore the potential for using a nonsmooth loss function based on the max-norm in the training of an artificial neural network. We hypothesise that this may lead to superior classification results in some special cases where the training data is either very small or unbalanced. Our numerical experiments performed on a simple artificial neural network with no hidden layers (a setting immediately amenable to standard nonsmooth optimisation techniques) appear to confirm our hypothesis that uniform approximation based approaches may be more suitable for the datasets with reliable training data that either is limited size or biased in terms of relative cluster sizes. keywords: quasiconvex functions, bisection method for quasiconvex minimisation, deep learning MSC 2010: 90C26, 90C90, 90C47, 65D15, 65K10
# 1 Introduction
Deep learning is a popular tool in the modern area of Artificial Intelligence. Deep learning has many practical applications, including data analysis, signal and image processing and many others [11, 16, 24]. Deep learning is based on solid mathematical modelling established in [6, 14,17,19]. These works demonstrate that deep learning solves approximation problems and, in its essence, relies on the results of the celebrated Kolmogorov-Arnold Theorem [1,15]. There is a massive amount of publications and internet resources dedicated to deep learning. One of the most comprehensive and thorough textbook on the modern view on deep learning can be found in [11]. This book also touches upon the optimisation side of the problem. In particular, the goal is to minimise the loss function, which is also the measure of “inaccuracy”. Overall, the goal of deep learning is to optimise weights in the network and therefore, this problem can be treated using modern optimisation tools [11–13,24]. It is customary to choose the mean least squares loss function to evaluate and optimise the performance of a neural network against a training set. There are several reasons for the popularity of the least squares formulation. From the optimisation perspective this model involves minimising a smooth quadratic objective function, and therefore basic optimisation techniques such as the gradient descent can be successfully employed. The least squares formulation also fares well with the assumption that the errors in the training set have a normal distribution. The goal of this work is to explore an alternative choice of loss functions and to analyse the impact these choices have on the quality of the training and the choice of the optimisation technique. The idea to use a different objective is not new, and was explored in the literature and applications. For instance, l1-norm may be useful in the settings when sparsity is sought after [2]. Other sources where alternative loss functions are explore are [11,18]. Our main hypothesis is that uniform approximation-based models may work better than least squares-based models whenever the size of the data available for training the model is small
[cs.LG]
while highly reliable or when the data is unbalanced. This is the case for the kind of datasets where each observation is a results of a very expensive procedure or experiment and therefore the availability of data is limited [23]. At the same time, the reliability of such data is high, since every experiment is carefully designed and analysed. Our preliminary numerical experiments appear to confirm the claim. We use an elementary model of an artificial neural network that has no hidden layers and only one output node. It was fairly easy to implement the necessary nonsmooth optimisation technique for this model to test our assumptions, moreover it is common to use such simple models in research literature: they represent building blocks for more complex models used in practice, while are easier to analyse. Since the proposed objective function is nonsmooth, we couldn’t use standard software for the training, and instead implemented a numerical routine from scratch. The relevant optimisation problem turns out to be quasiconvex, it can be solved by the bisection method, with each step requiring a linear feasibility problem to be solved. In [22] the authors explore the application of quasiconvexity in the case of least squares as well, but their approach is fundamentally different from ours, since our approach uses global characteristics of the functions, rather than local. We have implemented the algorithm in MATLAB and ran our numerical experiments in tandem alongside a standard implementation with the mean squared loss error function. Our experiments confirm that for a very small training set our max-norm model may perform better than the standard mean least squares formulation, moreover, adding an extra step that removes the outliers may improves the classification further. The paper is organised as follows. In Section 2 we explain a basic model of an artificial neural network and fix the notation that we use in subsequent sections. In Section 3 we recap some optimisation background related to quasiconvex functions and the bisection method that are used in Section 4 to perform the numerical experiments. Theorem 1 is the central mathematical result of this paper, connecting neural network models and quasiconvex optimisation. Section 5 reports the results of numerical experiments. Finally, Section 6 provides conclusions and future research directions.
# 2 Training a Simple Artificial Neural Network
We consider a basic model of an artificial neural network, and describe a training algorithm that can be used for the model with no hidden layers that is based on a bisection method for quasiconvex programming. A basic model of a neural network consists of several layers of nodes (artificial neurons), connected by directed edges. The network receives a (numerical) signal to its input layer and calculates the output on each subsequent layer using some real valued functions (propagation functions), effectively calculating a composition of these functions as the signal propagates through this network to the output layer. A classic example of an artificial neural network is a handwritten number recognition system, where the input is a grayscale image of a handwritten symbol, and the output is the digit that this handwritten symbol represents [16]. The propagation functions that pass the signal on to the next layer are usually selected from a parametrised
We consider a basic model of an artificial neural network, and describe a training algorithm that can be used for the model with no hidden layers that is based on a bisection method for
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9c71/9c71efb3-bd1b-4fbd-9d29-a1f84c52bcb8.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: A simple neural network without hidden layers</div>
family, with parameters (weights) adjusted during the training of the network. We will use the tuple x = (x1,...,xn) ∈Rn to represent the input of the neural network (for example, this can be the values of the grayscale pixels in the input image), and y ∈Rm represents the output that encodes the (classification) information extracted by the neural network (for instance, the digit that the handwritten symbol represents). An elementary example of a neural network consists of one input layer with n nodes and the output layer with m nodes (see Fig. 1). Usually the propagation function is chosen as a composition of some univariate function σ (activation function that plays the role of the sigmoid from models of biological neural networks) with a parametrised affine mapping. The (multilayered) network is then a composition of linear functions with the univariate function σ. In this work we will focus on a simple model with no hidden layers, and hence the input x is converted into the output y by a single composition of an activation function σ and some affine (linear) parametrised function, so
where wij ∈R for i ∈{1,...,m}, j ∈{0,...n} are the weights. The free parameter w0j is usually called the bias weight, as the model is contextualised as having an additional set of input nodes called biases that always take the value 1, so that the output function is linear on a lifted space with one additional input variable x0 = 1. Most commonly used activation functions are nondecreasing, and in fact it will be convenient for us to assume that σ is a continuous strictly increasing function. We can rewrite the model in a matrix notation as y(x) = ϕ(W,x) = σ( ¯Wx+w0), where
p ∈Rd, so that
    The training of a neural network normally consists of minimising a loss function that captures how closely the generated output describes the desired output values. Given a training set Z = {(¯x1, ¯y1),(¯x2, ¯y2),...,(¯xN, ¯yN)} that consists of pairs of inputs and desired outputs, we would like to choose our parameters W in such a way that the outputs produced by the neural network y(¯xi) = ϕ(W, ¯xi) = σ( ¯W ¯xi + w0) are as close as possible to the desired outputs ¯yj for j ∈{1,...,N}. A common loss (error) function used in this context is based on mean least squares formulation, N
where ∥·∥2 is the 2-norm,
 goal is to explore a different choice of the loss function, specific
L∞,∞(Z,W) = max i∈{1,...,N} max j∈{1,...,m}|¯yi j −ϕj(W, ¯xi)|,
which is effectively a composition of max-norms. We could have made different choices on both levels, but our interest lies in departing reasonably far from the mean least squares. The nature of this function ensures that we count the contribution from outliers in the dataset, rather than discount them via averaging.
# 3 Quasiconvex Functions and the Bisection Algorithm
The notion of quasiconvexity was originally introduced in the area of financial mathematics [10], where the author studied the behaviour of functions with convex sublevel sets, but the term quasiconvexity was introduced much later. Let D be a convex subset of Rn. A function f : D →R is quasiconvex if and only if its sublevel set
Sα = {x ∈D| f(x) ≤α}
is convex for any α ∈R. It is not difficult to observe that this definition is equivalent to requiring f(λx+(1−λ)y) ≤max{ f(x), f(y)} ∀x,y ∈D, λ ∈[0,1]. (2)
is convex for any α ∈R. It is not difficult to observe that this definition is equivalent to requiring
is convex for any α ∈R. It is not difficult to observe that this definition is equivalent to requiring
This characterisation is convenient for proving a number of important properties of quasiconvex functions, for instance, that the supremum of quasiconvex functions is quasiconvex.
(1)
(2)
A function f : D →R, where D is a convex subset of Rn, is called quasiconcave if −f is quasiconvex. Functions that are both quasiconvex and quasiconcave are called quasiaffine (quasilinear). A quasiconvex function does not need to be continuous, same applies to quasiconvcave and quasiaffine functions. In the case of univariate functions, quasiaffine functions are monotone. There are many studies devoted to quasiconvex optimisation [3,5,7,9,20,21]. In these studies, the notion of quasiconvexity appears as one of the possible generalisations of convexity. It is easy to see (e.g. using the characterisation (2)) that if g : Rn →R is quasiconvex and h : R →R is nondecreasing, then the composition f = h ◦g is quasiconvex. It appears that a similar statement is true when g is quasiaffine and h is monotone.
# Lemma 1 (Composition of monotone and quasiaffine functions) Assume that g : Rn →R is quasiaffine and h : R →R is monotone, then the composition f = h◦g is quasiaffine.
# osition of monotone and quasiaffine functions) Assume that g 
proof Assume that h is nondecreasing and g is quasiaffine, then f = h ◦g is quasiconvex. It remains to prove that −f = −h◦g is quasiconvex. Consider an auxiliary function ˜h : R →R, such that ˜h(t) = h(−t) and therefore h(t) = ˜h(−t). It is clear that if h is nondecreasing then ˜h is nonincreasing and vice versa. We have
−f = −h◦g = −˜h◦(−g).
Since −˜h is nondecreasing and −g is quasiconvex (g is quasiaffine), then −f is quasiconvex and therefore f is quasiaffine. It is also clear from the proof that h does not have to be non-decreasing, it is enough for h to be monotone. □ A standard technique for minimising quasiconvex functions is the bisection method (see [3, Section 4.2.5]). If a quasiconvex minimisation problem has an optimal solution and a lower and upper bounds on the optimal value of the objective are known, the method proceeds by bisecting the interval between the lower and upper bounds and solving a feasibility problem to detect whether the sublevel set corresponding to that midpoint is nonempty. If the sublevel set is empty, then the bisector becomes the new lower bound, otherwise the new upper bound is assigned the bisector value, and the process continues.
# 4 Training the Model
We consider a simple network without hidden layers (only input and output layers), furthermore we assume that the output layer consists of a single node. In this case on the input x ∈Rn the single output neuron produces the output
where σ is a monotone activation function, and W is reduced to a row vector, so we omit the first index and write wj for w1j with j ∈{1,...,N}. Since the function ∑n j=1 w jxi j +w0 is affine,
(3)
hence quasiaffine, from Lemma 1 we conclude that the composition ϕ(x) = σ( ¯wx + w0) is a quasiaffine function. Since changing the sign or adding a constant preserves the quasiaffinity the loss function (1) can be represented as the maximum of quasiaffine functions,
L∞,∞(Z,W) = max i∈{1,...,N}max{¯yi −ϕ(W, ¯xi),−¯yi +ϕ(W, ¯xi)}
Qiasiaffine functions are quasiconvex, and the maximum of quasiconvex functions is quasiconvex, therefore our loss function is quasiconvex in W. We have proved the following result. Theorem 1 In the case of a simple neural network consisting of an input layer and a single output node, the loss function L∞,∞is quasiconvex. Remark 1 Observe that Theorem 1 can be generalised to the case when the output contains more than one node: sandwiching one more layer of maxima in (4) again results in a quasiconvex function, as the proof is based on the same argument about a maximum of quasiaffine functions. Our next goal is to develop the implementation of the bisection method for our setting. We would like to solve the minimisation problem
Qiasiaffine functions are quasiconvex, and the maximum of quasiconvex functions is quasiconvex, therefore our loss function is quasiconvex in W. We have proved the following result.
�� �� Here w = (w0,w1,...,wn) = (w0, ¯w) ∈R×Rn are the weights to be decided, and Z = {(¯xi, ¯yi)}N i=1, with (¯xi, ¯yi) ∈Rn ×R, i ∈{1,...,N} is the training set. We know from Theorem 1 that the max function is quasiconvex, hence we can apply the bisection method to this minimisation problem. It is evident that the objective function is nonnegative, so we can choose l0 = 0 as the lower bound for the optimal value. For the upper bound we can substitute any value of the parameter w in the objective, for instance, w = 0 ∈Rn+1, then
�� �� We know that the optimal value of the objective function is between l0 and u0. Let L1 := l0+u0 2 . On each iteration we solve the feasibility problem
��� ��� If the problem is feasible, we let lk = lk−1, uk = Lk. Otherwise we let lk = Lk, uk = uk−1. We set a threshold ε > 0 and continue the process until the gap between the upper and the lower bound is smaller than this value, that is, our stopping criterion is
(4)
(5)
This is a simple procedure, which terminates in a finite number of steps. The convergence is linear [3] and therefore it is essential to start with the most accurate estimations for u0 and l0 to reduce the number of iterations. The feasibility problem (5) that we are required to solve on every iteration can be equivalently rewritten as
Under the assumption that σ is strictly increasing and hence has an inverse, this can be rewritten as
Notice that this problem is a linear feasibility problem (in other words, it is a system of linear inequalities with respect to w), hence it can be solved with any standard linear programming technique.
Notice that this problem is a linear feasibility problem (in other words, it is a system of linear inequalities with respect to w), hence it can be solved with any standard linear programming technique. Remark 2 The bisection method works in the case of non-decreasing functions (for example, classical Rectified Linear Unit (ReLU) activation function is not strictly monotone), since the problem remains quasiconvex. The only difference is that the left- and right-hand side values will correspond to the minima and maxima respectively over the set-valued inverse images of σ. Our assumption of strict monotonicity fares well with the choice of the Leaky Rectified Linear Unit (Leaky ReLU) activation function,
Notice that this problem is a linear feasibility problem (in other words, it is a system of linear inequalities with respect to w), hence it can be solved with any standard linear programming technique.
Remark 2 The bisection method works in the case of non-decreasing functions (for example, classical Rectified Linear Unit (ReLU) activation function is not strictly monotone), since the problem remains quasiconvex. The only difference is that the left- and right-hand side values will correspond to the minima and maxima respectively over the set-valued inverse images of σ. Our assumption of strict monotonicity fares well with the choice of the Leaky Rectified Linear Unit (Leaky ReLU) activation function,
This is a piecewise linear function with only two pieces, changing the linear slope at the origin:
A common choice of the parameter is α = 0.01, that is
It is easy to see that the inverse can be calculated explicitly,
hence the linear feasibility problem (5) can also be written and coded explicitly.
(6)
<div style="text-align: center;">Table 1: Original dataset: classification results</div>
Table 1: Original dataset: classification results
Test set
Method
classification
Confusion matrix
accuracy
MATLAB toolbox
89.7%
108
25
13
224
Uniform approximation
60.54%
77
56
90
147
# 5 Numerical Experiments
The goal of our numerical experiments is to test our hypothesis that using a max-type (uniform) loss function (1) may be beneficial in training an artificial neural network in the setting of a reliable but small training dataset. In our numerical experiments, we model this situation by reducing the size of the training set. We use HandOutlines dataset from [8]. This dataset contains 2 classes, the corresponding training set contains 1000 recordings (362 recordings in Class 1 and 638 recordings in Class 2) and the corresponding test set contains 370 recordings (133 recordings in Class 1 and 237 recordings in Class 2). Each record contains 2709 floating point values. The dataset contains the information about hand outlines of the subjects and their age (image type data). The data set was manually labelled as correct and incorrect hand outlines by three volunteers. If all three volunteers agree that a data point is valid, it is labelled as correct and hence, class 2 contains correctly identified data points whereas class 1 contains incorrectly identified data points. The size of the classes is not equal: Class 2 is almost twice the size of Class 1.
# 5.1 Experiments with the original dataset
We start with the original dataset (1000 points training set and 370 points test set) and compare the classification results obtained by MATLAB Deep learning toolbox and the results obtained via our uniform approximation algorithm (using bisection method, ε = 10−5). We use the default activation functions for MATLAB Deep learning experiments and Leaky ReLu function with α = 10−2 for uniform approximation. We report the classification accuracy and provide the confusion matrix. The classification accuracy gives the proportion of points that were assigned the correct class, and the confusion matrix gives more details: the diagonal entries give the number of elements from each of the two classes that were classified correctly, while the off-diagonal elements correspond to the number of misclassified points. In general, the rows of the confusion matrix correspond to the actual class and the columns correspond to the predicted class. For example, the element at the position (1,2) correspond to the number of points from Class 1 assigned to Class 2 (misclassified). The results of the experiments are shown in Table 1. From Table 1 one can see that MATLAB toolbox is much more accurate (almost 90%), while the uniform approximation is just above 60%.
<div style="text-align: center;">Table 2: Original dataset: classification results, training and test sets are swapped</div>
2: Original dataset: classification results, training and test sets are s
Test set
Method
classification
Confusion matrix
accuracy
MATLAB toolbox
84.5%
222
140
15
623
Uniform approximation
66.7%
195
167
166
472
Our intention is to demonstrate that uniform approximation is a better tool when the size of the corresponding training set is small. To demonstrate this numerically, we swap training and test sets and now the training set contains 370 points, while the test set contains 1000 points. The results are presented in Table 2. One can see from Table 2 that the classification accuracy for MATLAB toolbox decreased, which is not surprising, since the size of the training set is reduced and therefore less information is used to train the models. On the other hand, the classification accuracy in the case of uniform approximation has improved. Uniform approximation, due to its nature, treats smaller (underrepresented) groups as valid points, while least squares approximation tends to “average” and therefore under-represented groups tend to be “ignored”. This is a great advantage when the under-represented groups are outliers, but in many cases these points are valid data. On the other hand, the presence of ouliers may decrease may decrease the accuracy in the case of uniform approximation. Therefore, our hypothesis is that uniform approximation approach is preferable in the following cases.
2. Presence of under-represented groups of valid data or uneven distribution of data betwee the classes (that is, one class is significantly larger than others).
 Limited size of the available data, where most datapoints are val
The last case is very common in applications where each datapoint is a result of a very expensive experiment or procedure [23]. Based on our hypothesis, the improvement in the classification accuracy in the case of uniform approximation is due to the fact that many outliers are now removed. Overall, MATLAB toolbox is still more accurate, but this simple experiment encourages us to proceed with the reduction of the training set.
# 5.2 Experiments with reduced training set
ur next step is to reduce the training set even more. The experimental setup is as follows. 1. We use a reduced size test set (the exact size is specified in each experiment) to train the model and the original training set (1000 points) to test (we use this set for testing, since it is a larger set);
Table 3: Reduced dataset: classification results for even number of points from each training set
<div style="text-align: center;">Table 3: Reduced dataset: classification results for even number of points from each class in the</div>
Test set
Method
classification
Confusion matrix
accuracy
MATLAB toolbox
83.9%
218
144
17
621
Uniform approximation
70.60%
188
174
120
518
# 2. The tolerance ε in the bisection procedure is ε = 10−5.
# 2. The tolerance ε in the bisection procedure is ε = 10−5. Our experiments can be divided into two groups.
1. Equal vs unequal distribution of points between the classes in the training set. In this group of experiments, we are checking if it is harder to create accurate classification rules for unbalanced training sets. 2. Random vs non-random selection of reduced size training set. Random selection of training data is a common approach, since it reduces the chance of getting an “unusual” piece of data. At the same time, non-random selection of points (for example, top 10%) is helpful when one needs to recreate the experiments on the same piece of data.
1. Equal vs unequal distribution of points between the classes in the training set. In this group of experiments, we are checking if it is harder to create accurate classification rules for unbalanced training sets.
# Even number of representatives from each class in the train
In this experiment, we use 20 points for training: first 10 points from Class 1 and first 10 points from Class 2 (taken from 370 point set, which is the test set for the original dataset). The results are in Table 3. The results demonstrate that MATLAB toolbox is still more accurate, but the uniform approximati based approach is coming closer. Our next step is to consider situations, where the size of the training set remains at the same level, but one of the classes is underrepresented.
In this experiment, we use 20 points for training: first 10 points from Class 1 and first 10 points from Class 2 (taken from 370 point set, which is the test set for the original dataset). The results
# 2.2 Uneven number of representative from each class in the training set
Training set contains 40 points Consider the situation where the training set contains 40 points: 35 points from Class 1 and 5 points from Class 2. The results are in Table 4. The classification accuracy is still higher in the case MATLAB toolbox, but this difference
Training set contains 40 points Consider the situation where the training set contains 40 points 35 points from Class 1 and 5 points from Class 2. The results are in Table 4. The classification accuracy is still higher in the case MATLAB toolbox, but this difference is reducing. Consider now a symmetric situation where the training set contains 40 points: 5 points from Class 1 and 35 points from Class 2. The results are in Table 5. In this experiment, the classification accuracy is higher for uniform approximation.
<div style="text-align: center;">Table 4: Reduced dataset: classification results for uneven number of points from each class in the training set: 35 points in Class 1 and 5 points in Class 2</div>
t: 35 points in Class 1 and 5 points in Class 2
Test set
Method
classification
Confusion matrix
accuracy
MATLAB toolbox
74.6%
116
246
8
630
Uniform approximation
66.5%
128
234
101
537
<div style="text-align: center;">Table 5: Reduced dataset: classification results for uneven number of points from each class in the training set: 5 ponts in Class 1 and 35 points in Class 2.</div>
t: 5 ponts in Class 1 and 35 points in Class 2.
Test set
Method
classification
Confusion matrix
accuracy
MATLAB toolbox
64.3%
296
66
291
347
Uniform approximation
69.5%
193
169
136
502
<div style="text-align: center;">Table 6: Reduced dataset: classification results for uneven number of points from each class i the training set: 18 points in Class 1 and 2 points in Class 2</div>
Table 6: Reduced dataset: classification results for uneven number of points the training set: 18 points in Class 1 and 2 points in Class 2
: 18 points in Class 1 and 2 points in Class 2
Test set
Method
classification
Confusion matrix
accuracy
MATLAB toolbox
68.6%
65
297
17
621
Uniform approximation
63.30%
91
271
96
542
<div style="text-align: center;">Table 7: Reduced dataset: classification results for uneven number of points from each class  the training set: 2 points in Class 1 and 18 points in Class 2.</div>
: 2 points in Class 1 and 18 points in Class 2.
Test set
Method
classification
Confusion matrix
accuracy
MATLAB toolbox
39.5%
338
24
582
56
Uniform approximation
56.60%
267
95
339
299
<div style="text-align: center;">Table 8: Reduced dataset: classification results for randomly generated training set of 100 points repeated 10 times, the accuracy is averaged.</div>
ated 10 times, the accuracy is averaged.
Method
Test set classification accuracy
MATLAB toolbox
56.94%
Uniform approximation
73.19%
Training set contains 20 points Consider the situation where the training set contains 20 points 18 points from Class 1 and 2 points from Class 2. The results are in Table 6. The classification accuracy is still higher in the case MATLAB toolbox, but this difference is not very larger. Consider now a symmetric situation where the training set contains 20 points: 2 points from Class 1 and 18 points from Class 2. The results are in Table 7. In this experiment, the classification accuracy is low for both approaches, which is not surprising, given the size of the training set and uneven distribution of classes. However, the uniform approximation approach is more accurate.
Conclusions Overall, when the size of the training set is reducing, the uniform approximation approach becomes more efficient than MATLAB toolbox (based on MSE). This observation i especially significant when Class 1 is significantly underrepresented.
# 5.2.3 Random choice of training set points
In this section we present the numerical results when the training set points were chosen randomly from the original test set (370 points), while the testing was performed on the original training set (1000 points). The total size of the training sets are 100, 50 and 20. Since the points are chosen randomly, the distribution of the points between classes is also random. Each experiment was repeated ten times and the reported test set accuracy is the average. This approach is related to the commonly used 10-fold cross-validation approach with some adjustment to our problem.
Random training set size is 100 We start with the case where the training set contains 100 poin in total. The results are in Table 8. The classification accuracy is higher for uniform approximation.
Random training set size is 50 The training set contains 50 points in total. The results are in Table 9. The classification accuracy is higher for uniform approximation, but the gap is slightly decreasing compared to the experiment with 100 points.
Random training set size is 20 The training set contains 20 points in total. The results are in Table 10.
<div style="text-align: center;">Table 9: Reduced dataset: classification results for randomly generated training set of 50 points repeated 10 times, the accuracy is averaged.</div>
epeated 10 times, the accuracy is averaged.
Method
Test set classification accuracy
MATLAB toolbox
57.08%
Uniform approximation
71.58%
<div style="text-align: center;">Table 10: Reduced dataset: classification results for randomly generated training set of 100 points: repeated 10 times, the accuracy is averaged.</div>
ints: repeated 10 times, the accuracy is averaged.
Method
Test set classification accuracy
MATLAB toolbox
56.57%
Uniform approximation
69.21%
The classification accuracy is higher for uniform approximation. The gap between uniform approximation and MATLAB is quite significant.
Why the reduction of the training set leads to the improvement in the classification accuracy in the case of the uniform approximation-based approach? One possible explanation is that by removing a significant proportion of points, we also remove all (or almost all) outliers. If this is the case, then the removal of outliers make uniform approximation a better choice. Most outliers are recording or instrumental error and should be removed. At the same time, if recording or instrumental are common, it is unrealistic to avoid them is real-life applications. In the next section, we provide the results of numerical experiments where points with high absolute deviation from best uniform approximation are considered as outliers.
# 4 High absolute deviation point removal from the training s
In this section we assume that points with the highest absolute deviation from best uniform approximation are treated as outliers. The procedure contains two main steps (recall that we use the original test set for training and the training set for testing due to their size).
1. Training set reduction Find best uniform approximation using the original test set (370 point Identify the points whose absolute deviation is maximal or close to maximal with a specified tolerance ε. Refine the training set (370 points) by removing these points (outliers).
2. Actual training Treat the refined set as the training set and perform test classification on the original training set (1000 points).
Note that since there are several points with almost the same absolute deviation, it is hard to anticipate the threshold ε that gives a certain percentage of data reduction. We consider two cases. In the first case, the threshold ε = 10−7: 46 points out of 370 are removed. In the second case, our goal is to remove (approximately) half of the training set points: 181 points are treated as outliers and the remaining 189 points are used for training.
<div style="text-align: center;">Table 11: Reduced dataset: 324 valid points and 46 outliers.</div>
Table 11: Reduced dataset: 324 valid points and 46 outliers.
Test set
Method
classification
Confusion matrix
accuracy
MATLAB toolbox
57.3%
81
281
143
495
Uniform approximation
71%
216
146
144
494
<div style="text-align: center;">Table 12: Reduced dataset: 181 outliers and 189 valid points</div>
Table 12: Reduced dataset: 181 outliers and 189 valid points
Test set
Method
classification
Confusion matrix
accuracy
MATLAB toolbox
57.1%
92
270
159
479
Uniform approximation
72.4%
235
127
149
489
Training set size is 324 We start with the case where 46 points are treated as outliers and therefore they are removed. The results are in Table 11. The classification accuracy is significantly higher for uniform approximation. Training set size is 189 In this experiment, 181 points are treated as outliers while 189 points are counted as valid points. The results are in Table 12. The classification accuracy is significantly higher for uniform approximation and the gap is increasing compared to the situation where fewer outliers are removed.
Training set size is 324 We start with the case where 46 points are treated as outliers and therefore they are removed. The results are in Table 11. The classification accuracy is significantly higher for uniform approximation.
Training set size is 189 In this experiment, 181 points are treated as outliers while 189 points are counted as valid points. The results are in Table 12. The classification accuracy is significantly higher for uniform approximation and the gap is increasing compared to the situation where fewer outliers are removed.
# 6 Conclusions and Future Research
The results of the numerical experiments support our hypothesis that uniform approximationbased approach can be more efficient than mean least squares based approach when the training data is reliable while limited in size. This type of data are common in industrial, medical and research applications, where each datapoint is a result of an expensive experiment, medical procedure or expert evaluation that are not performed routinely. The size of such datasets can be very small, but each recording is carefully performed and therefore is a valid point. Based on our experiments, it may be beneficial to use a uniform approximation based approach, rather than the standard approach for this type of data. In the absence of an expert opinion it may be hard to distinguish between outliers and valid points that are underrepresented in a given dataset. This is one of the (many) reasons for gender, racial and other kinds of bias present in modern automatic decision making processes [4] that rely on the standard artificial neural network approach and implicitly assumes Gaussian data
distribution. This may lead to automatic discarding of under-represented data points as errors. This problem is not new, but it may be interesting to explore if replacing the mean least squares classifiers with uniform approximation may lead to useful results. Our work represents an initial step in researching the use of uniform approximation to address problems arising with data classification via standard artificial neural networks when a modest but high quality sample of classified data points is available. The following research themes are of future interest. 1. Extend our approach to artificial neural networks with several hidden layers. 2. Study other classification problems with limited or unbalanced training data. 3. Identify the types of activation functions that are efficient for particular datasets, rather than using “standard” ReLU activation.
distribution. This may lead to automatic discarding of under-represented data points as errors. This problem is not new, but it may be interesting to explore if replacing the mean least squares classifiers with uniform approximation may lead to useful results. Our work represents an initial step in researching the use of uniform approximation to address problems arising with data classification via standard artificial neural networks when a modest but high quality sample of classified data points is available. The following research themes are of future interest.
3. Identify the types of activation functions that are efficient for particular datasets, rather than using “standard” ReLU activation.
Finally, we would like to emphasise that the essence of deep learning is in approximation and optimisation and therefore there is a need for more robust mathematical models to tackle these problems. On the other hand, some fast heuristics may be used in the models where the size of complexity makes it hard to apply mathematical optimisation. Generally speaking, this models are not as reliable as those based on mathematical optimisation, but in some cases they are the only way we can handle such problems.
# Acknowledgement
This research was supported by the Australian Research Council (ARC), Solving hard Chebyshev approximation problems through nonsmooth analysis (Discovery Project DP180100602).
# References
[1] Arnold, V.: On functions of three variables. Dokl. Akad. Nauk SSSR 114, 679–681 (1957). English translation: Amer. Math. Soc. Transl., 28 (1963), pp. 51–54 [2] Bach, F., Jenatton, R., Mairal, J., Obozinski, G.: Convex Optimization with SparsityInducing Norms, chap. 2, pp. 19–53. MIT press (2011) [3] Boyd, S., Vandenberghe, L.: Convex Optimization, seventh edn. Cambridge University Press, New York, USA (2009) [4] Buolamwini, J., Gebru, T.: Gender shades: Intersectional accuracy disparities in commercial gender classification. Proceedings of Machine Learning Research 81, 1–15 (2018) [5] Crouzeix, J.P.: Conditions for convexity of quasiconvex functions. Mathematics of Operations Research 5(1), 120–125 (1980) [6] Cybenko, G.: Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems 2, 303–314 (1989)
[7] Daniilidis, A., Hadjisavvas, N., Martinez-Legaz, J.E.: An appropriate subdifferential for quasiconvex functions. SIAM Journal on Optimization 12(2), 407–420 (2002) [8] Dau, H.A., Keogh, E., Kamgar, K., Yeh, C.C.M., Zhu, Y., Gharghabi, S., Ratanamahatana, C.A., Yanping, Hu, B., Begum, N., Bagnall, A., Mueen, A., Batista, G., Hexagon-ML: The ucr time series classification archive (2018). https://www.cs.ucr.edu/~eamonn/ time_series_data_2018/ [9] Dutta, J., Rubinov, A.M.: Abstract convexity. Handbook of generalized convexity and generalized monotonicity 76, 293–333 (2005) [10] de Finetti, B.: Sulle stratificazioni convesse. Ann. Mat. Pura Appl. pp. 173–183 (1949) [11] Goodfellow, I., Bengio, Y., Courville, A.: Deep Learning. MIT Press (2016). http: //www.deeplearningbook.org [12] Gould, S., Hartley, R., Campbell, D.: Deep declarative networks: A new hope. CoRR abs/1909.04866 (2019). http://arxiv.org/abs/1909.04866 [13] Haeffele, B.D., Vidal, R.: Global optimality in neural network training. In: 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pp. 4390–4398. IEEE Computer Society (2017). doi 10.1109/CVPR.2017.467. https://doi.org/10.1109/CVPR.2017.467 [14] Hornik, K.: Approximation capabilities of multilayer feedforward networks. Neural networks 4(2), 251–257 (1991) [15] Kolmogorov, A.N.: On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition. Dokl. Akad. Nauk SSSR 114, 953–956 (1957) [16] LeCun, Y., Cortes, C., Burges, C.C.: The mnist database of handwritten digits (1998). http://yann.lecun.com/exdb/mnist/ [17] Leshno, M., Lin, V.Y., Pinkus, A., Schocken, S.: Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural Networks 6(6), 861–867 (1993). https://doi.org/10.1016/S0893-6080(05)80131-5 [18] Marcotte, P., Savard, G.: Novel approaches to the discrimination problem. Mathematical Methods of Operations Research 36, 517–545 (1992) [19] Pinkus, A.: Approximation theory of the MLP model in neural networks. Acta Numerica 8, 143–195 (1999). doi 10.1017/S0962492900002919 [20] Rubinov, A.M.: Abstract convexity and global optimization, vol. 44. Springer Science & Business Media (2013) [21] Rubinov, A.M., Simsek, B.: Conjugate quasiconvex nonnegative functions. Optimization 35(1), 1–22 (1995). doi 10.1080/02331939508844124
[22] Sinha, V.B., Kudugunta, S., Sankar, A.R., Chavali, S.T., Balasubramanian, V.N.: Dante: Deep alternations for training neural networks. Neural Networks 131, 127–143 (2020). doi https://doi.org/10.1016/j.neunet.2020.07.026. https://www.sciencedirect.com/ science/article/pii/S0893608020302677 [23] Steponaviˇc˙e, I., Hyndman, R., Smith-Miles, K., Villanova, L.: Efficient Identification of the Pareto Optimal Set. In: Pardalos P., Resende M., Vogiatzis C., Walteros J. (eds) Learning and Intelligent Optimization. LION 2014. Lecture Notes in Computer Science, vol. 8427. Springer, Cham (2014) [24] Sun, R.Y.: Optimization for deep learning: An overview. Journal of the Operations Research Society of China 8, 249–294 (2020)
