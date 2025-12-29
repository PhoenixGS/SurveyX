# LARGE-SCALE ONLINE FEATURE SELECTION FOR ULTRA-HIGH DIMENSIONAL SPARSE DATA
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/583d/583d01ed-66d9-4d4c-bc0e-f5fdc1a48ac1.png" style="width: 50%;"></div>
Yue Wu University of Science and Technology of China wye@mail.ustc.edu.cn
Tao Mei Microsoft Research, Beijing tmei@microsoft.com
19 Nov 2015
# ABSTRACT
Feature selection with large-scale high-dimensional data is important yet very challenging in machine learning and data mining. Online feature selection is a promising new paradigm that is more efficient and scalable than batch feature section methods, but the existing online approaches usually fall short in their inferior efficacy as compared with batch approaches. In this paper, we present a novel second-order online feature selection scheme that is simple yet effective, very fast and extremely scalable to deal with large-scale ultra-high dimensional sparse data streams. The basic idea is to improve the existing first-order online feature selection methods by exploiting second-order information for choosing the subset of important features with high confidence weights. However, unlike many second-order learning methods that often suffer from extra high computational cost, we devise a novel smart algorithm for second-order online feature selection using a MaxHeap-based approach, which is not only more effective than the existing first-order approaches, but also significantly more efficient and scalable for large-scale feature selection with ultra-high dimensional sparse data, as validated from our extensive experiments. Impressively, on a billion-scale synthetic dataset (1-billion dimensions, 1-billion nonzero features, and 1-million samples), our new algorithm took only 8 minutes on a single PC, which is orders of magnitudes faster than traditional batch approaches. http://arxiv.org/abs/1409.7794
arXiv:1409.7794v3
# INTRODUCTION
In machine learning and data mining, feature selection (FS) is the process of selecting a subset of relevant features and removing irrelevant and redundant features from data towards model construction. It is a very important technique in the era of big data today, and has found applications in a wide range of domains, particularly for scenarios with high-dimensional data. Feature selection has been extensively studied in which various algorithms have been proposed (Liu & Yu, 2005). Despite the extensive research efforts in literature, most existing feature selection methods are restricted to batch learning settings (Saeys et al., 2007), which have many critical drawbacks for big data applications. One drawback with batch learning is that they often require the entire training data set to be loaded in memory. This is obviously non-scalable when solving real-world applications with large-scale datasets that exceed memory capacity. Another drawback is that batch learning methods usually assume all training data and their full set of features must be made available prior to the learning task. This assumption does not always hold in many real-world applications where data arrives sequentially (e.g., internet data) and novel features may appear incrementally (e.g., spam email filtering). These drawbacks make traditional batch feature selection techniques non-practical for emerging big data applications.
To overcome the drawbacks of batch feature selection, online feature selection has been explored recently (Perkins & Theiler, 2003; Wang et al., 2014; Wu et al., 2010). One state-of-the-art scheme in (Wang et al., 2014) attempts to resolve feature selection by exploring online learning techniques. Although it is far more efficient and scalable than batch feature selection techniques, it still falls short in requiring linear time complexity with respect to feature dimensionality and sometimes failing to achieve satisfying learning accuracy when solving difficult tasks. In this paper, we argue that existing solutions are still not feasible due to high time and memory cost in real world applications with large-scale and ultra-high dimensional data. We propose a simple but smart second order online feature selection algorithm that is extremely efficient, scalable to large scale and ultra-high dimensionality, and effective to address this open challenge. Compared to existing FS methods, the complexity is significantly reduced to be linear to the average number of nonzero features per instance, rather than the full feature dimensionality. In particular, unlike the existing first-order online FS approaches, the proposed algorithm exploits the recent advances of second order online learning techniques (Dredze et al., 2008), trying to select the most confident weights while keeping the distribution close to the non-truncated distribution. It achieves highly competitive learning accuracy even compared with state-of-the-art batch FS methods. The rest of this paper is organized as follows: Section 2 reviews related work; Section 3 presents the proposed method in detail; Section 4 discusses our empirical studies; and finally Section 5 draws our conclusions. More extensive results are also included in the appendix section due to space limitation.
# 2 RELATED WORK
Feature selection methods have been extensively studied in literature (Kohavi & John, 1997; Liu & Yu, 2005; Zhao et al., 2013), which can be roughly grouped into three categories: Filter, Wrapper, and Embedded methods. Filter methods rely on characteristics of data such as correlation, distance, and information gain without assuming specific classifiers (Yu & Liu, 2003). Unlike the Filter methods that ignore the effect of selected features on the performance of the induction algorithm, wrapper methods employ a predetermined classifier to evaluate the quality of selected features (Kohavi & John, 1997). It searches for a subset of features and then evaluates their classification performances repeatedly. They often yield better performance for the chosen classifier, but are computationally intensive. Embedded methods integrate feature selection into the model training process (Xu et al., 2009), aiming to trade off between efficiency of filter methods and predictive accuracy of wrapper methods. However, their selected features might not be suitable for other classifiers. Many studies have attempted to address online FS in diverse ways. Some aim to handle streaming features arriving sequentially to the classifier (Glocer et al., 2005; Perkins & Theiler, 2003; Wu et al., 2010). Although they follow the stream learning setting and return a trained model at each time step given the observed features, they assume all the training instances must be given as a prior, making it unrealistic for many online applications. Our work is more closely related to another online FS setting in (Wang et al., 2014) that follows online learning methodology by assuming training data arrives sequentially. Despite its considerable advantages in efficiency and scalability over batch FS methods, it remains slow when being applied to large-scale FS tasks with ultra-high dimensionality. Our work is also related to online learning in machine learning literature (Crammer et al., 2006; Hoi et al., 2014), where a variety of online algorithms have been proposed, ranging from classical first-order algorithms (such as Passive-Aggressive learning (Crammer et al., 2006)) to recent second-order algorithms (Crammer et al., 2009). In general, these algorithms require to access and explore the full set of features. They are not directly applicable to online FS tasks for selecting a fixed number of active features. Another closely related online learning method is sparse online learning (Duchi et al., 2011; Langford et al., 2009), which aims to learn a sparse linear classifier from training data in high-dimensional space. Despite the extensive efforts, most of these works usually impose a soft constraint, such as ℓ1-regularization, onto the objective function for promoting sparsity, which do not directly solve an online FS task that requires a hard constraint on the number of active dimensions in the learned classifier. In this paper, we explore recent advances of online learning techniques in both second-order online learning and sparse online learning for advancing the state of the art of online feature selection tasks.
# 3 ONLINE FEATURE SELECTION
In this section, we present a novel online feature selection method. We first describe the problem setting and then briefly introduce the existing first-order online feature selection methods, followed by presenting the proposed second-order online feature selection method in detail.
# 3.1 PROBLEM SETTING
Without loss of generality, this paper first investigates the problem of online feature selection for binary classification tasks. Consider {(xt,yt)|t = 1,...,T} be a sequence of training data instances received sequentially over the training process, where each xt ∈Rd is a vector of d dimensions and yt ∈{+1,−1}. Generally, an online learner will learn a classifier with the same dimensionality w ∈Rd. In the setting of online feature selection, we need to select a relatively small number of elements in w and set the others to be zero. In other words, we impose the following constraint
where B is the predefined constant, and consequently at most B features of x will be used for prediction. Specifically, at each time t, a learner receives an incoming example xt ∈Rd, and then predicts its class label ˆyt ∈{−1,+1} based on its current model, i.e., a linear weight vector wt, as
  After making the prediction, the true label yt ∈{−1,+1} will be revealed, and the learner then can measure the loss lt(wt) suffered with respect to (xt,yt), which is the difference between the prediction outcome and the true label. At the end of each iteration, the learner will update the weight vector wt according to some learning rules. Throughout the paper, we assume ∥xt∥≤1,t = 1,...,T.
# 3.2 FIRST-ORDER ONLINE FEATURE SELECTION
One of most straightforward approaches to online feature selection is to apply the Perceptron algorithm via truncation (PET) (Wang et al., 2014). Specifically, at each step, the classifier first predicts the label ˆyt with wt. If ˆyt is correct, then wt+1 = wt; otherwise, the classifier will update wt by Perceptron rule to obtain ˆwt+1 = wt +ηtytxt, which will be further truncated by keeping the largest B absolute values of ˆwt+1 and setting the rest to zero. The truncated classifier, denoted by wB t or wt+1, will be used to predict the next observation. As analyzed in (Wang et al., 2014), the above simple approach does not work well in practice. In particular, it cannot guarantee a small number of mistakes since it fails to ensure the numerical values of truncated elements are sufficiently small, thus leading to a nontrivial loss of accuracy. Consequently, the authors in (Wang et al., 2014) proposed a novel first-order online feature selection scheme (FOFS) by exploring online gradient descent with a sparse projection scheme before truncation, which guarantees the resulting classifier wt to be restricted into an ℓ1-ball at each step. Algorithm 1 shows the details of their first-order OFS algorithm.
Algorithm 1 FOFS: First-order OFS via Sparse Projection
1: Input: B,η
2: Following the similar framework as PET but use constant learning rate η
3: ˜wt+1 = (1−λη)wt +ηytxt
4: ˆwt+1 = min{1,
1
√
λ
∥˜wt+1∥2 } ˜wt+1, where λ is a regularization parameter
5: wt+1 = Truncate( ˆwt+1,B)
# 3.3 SECOND-ORDER ONLINE FEATURE SELECTION
A key limitation of the above online feature selection algorithms is that they only exploit the firstorder information of the weight vector during the online feature selection process, which may lead to the loss of potentially informative features. To overcome the limitation, we propose a second-order
online feature selection method by exploring the recent advances of second-order online learnin techniques.
online feature selection method by exploring the recent advances of second-order online learning techniques.
The Confidence-Weighted (CW) method (Dredze et al., 2008) assumes the weight vector of the linear classifier follows a Gaussian distribution w ∼N (µµµ,Σ). Confidence of weights are represented by diagonal elements in covariance matrix Σj. The smaller Σ j, the more confidence we have in the mean value of weight µj. Before observing any samples, all the weights are of the same confidence or uncertainty. In the CW learning process, given an observed training example (xt,yt), CW makes an update by trying to stay close to the previous distribution and ensure that the probability of making correct prediction on xt is larger than a threshold η. The solution for the update can be cast into the following optimization:
( ˆµµµt+1,Σt+1) = argmin µµµ,Σ DKL(N (µµµ,Σ),N (µµµt,Σt)) s.t. Prw∼N (µµµ,Σ)[yt(w·xt) ≥0] ≥η (1)
The proposed second order online feature selection algorithm SOFS takes another step with similar idea to CW. With the goal to reduce the damage to classification ability while selecting features, SOFS tries to stay close to the updated distribution and ensure the L0 norm is less than B. The updated weights ˆwt+1 in equation (1) follows the distribution ˆwt+1 ∼N ( ˆµµµt+1,Σt+1). SOFS is cast into the following optimization:
µµµt+1 = argmin µµµ DKL(N (µµµ,Σt+1),N ( ˆµµµt+1,Σt+1)) s.t. ∥µµµ∥0 ≤B
In SOFS, only diagonal elements of the covariance matrix Σ are considered. This is because maintaining a full covariance matrix requires O(d2) memory space and O(d2) computational complexity, which is impractical for handling large-scale ultra-high dimensional data. By writing the KL divergence explicitly with the diagonal covariance matrix assumption, the above optimization is equivalent to:
µµµt+1 = argmin µµµ 1 2(µµµ −ˆµµµt+1)TΣ−1 t+1(µµµ −ˆµµµt+1) s.t. ∥µµµ∥0 ≤B.
uppose the selected feature indexes of the optimal solution µµµ∗to the optimization are s1,s2,...,sB. hus the rest feature weights with indexes sB+1,...,sd are set to zero. The KL divergence is:
  where Σt+1,si means the si-th diagonal element of the covariance matrix at iteration t. KL(µµµ∗, ˆµµµt+1) is the smallest among all the possible µµµt+1, it can be drawn that:
• si ∀ ∈ • Σt+1,si ≤Σt+1,s j,∀i ∈[1,B], j ∈[B+1,d].
  Note that the covariance matrix represents the confidence of weights. The above properties of the optimal solution indicate that the B most confidence features should be selected by exploiting the second-order information of the classifier. Specifically, in the online learning process, when the loss for a training instance (xt,yt) is non-zero, we update the weight vector only for the most confident B weight variables whose covariance values Σj are among the B smallest, and all the other weights are set to zero. By contrast, first order online feature selection algorithms select important features based on the magnitudes of the classifier weights. In this paper, we adopt the Adaptive Regularization of Weights (AROW) algorithm (Crammer et al., 2009) to solve the optimization problem in (1). It has been shown to be more robust in handling label noises than the original CW algorithms. The objective function of AROW is formulated as:
� � where γ > 0 is a regularization parameter. The problem in (5) can be solved with closed-form solutions as follows:
(1)
(2)
(3)
(4)
(5)
# 3.4 EFFICIENT SOFS ALGORITHMS
A common drawback with many existing second-order learning methods is the extra high computational cost incurred for exploiting the second-order information. In this section, we show that it is possible to devise a second-order OFS algorithm that is not only more effective but also considerably more efficient and scalable than the existing first-order approaches. Specifically, one of major time-consuming procedures in the above second-order feature selection method is to select top B elements from an array of length d (the diagonal vector of Σ in the secondorder OFS). Instead of sorting all the weights at each step as in the previous study (Wang et al., 2014), we propose a smart way to implement the proposed second-order online feature selection technique by employing a MaxHeap-based approach in exploiting the characteristics of SOFS, which can significantly reduce computational complexity to be linear with respect to the average number of nonzero features m of each example, rather than the original full dimensionality d (d ≫m). This makes it extremely fast and scalable when handling large-scale sparse high-dimensional data sets. Before presenting the proposed algorithm, we first introduce the following proposition for the monotonic decreasing property of Σt, a property that is critical to the proposed algorithm.
# Proposition 1 (monotonic decreasing) Given Σt computed by (6), ∀t and ∀j ∈[1,d], Σt+1, j ≤Σt,j.
It is not difficult to verify the above by noticing diag(xT
t xt)/γ is always non-negative. Using this
important property, we can develop a fast algorithm for the second-order OFS method.
Specifically, we build a MaxHeap data structure to store the B smallest diagonal values of covariance
Σt. The monotonic decreasing property of Σt implies the heap limit should decrease monotonically.
This leads to two major benefits in saving computational cost: (i) we do not need to check those
unchanged elements to see if they are smaller than the heap limit; and (ii) when updating elements
in the heap, only its child nodes need to be updated.
Algorithm 2 shows the details of the proposed fast algorithm for SOFS. Whenever a new feature
arrives and its covariance changes, we proceed to update as follows:
• If the corresponding covariance exists in the heap, adjust its position in the heap;
• Check if it is smaller than the heap limit; if so, replace the root node of the heap by the
current item and set the value of the original root node to be zero; otherwise,
• Simply set the corresponding weight to zero.
Algorithm 2 SOFS: Fast Algorithm for Second-order OFS
1: Input: γ, B
2: Initialize: µµµ1 = 0,Σ1 = I. MaxHeap H on Σ1 with size B
3: for t = 1,...,T
4:
if lt(µµµ) = max(0,1−yt(µµµ ·xt))2 > 0
5:
Calculate βt, gt by (6).
6:
for j = 1,...,d,xt, j ̸= 0
7:
µt+1, j = µt, j −1
2βtΣt, jgl
t,j,
Σ−1
t+1, j = Σ−1
t, j +
x2
t,j
γ
8:
if Σt+1, j ∈H
9:
adjust H to maintain the MaxHeap
10:
elseif Σt+1, j < Hmin
11:
replace Hmin by Σt+1, j and set the weight value of the original root node to be zero
12:
adjust H to maintain the MaxHeap
13:
else
14:
µt+1, j = 0
15: Output: weight vector µµµT and confidence ΣT
It is not difficult to verify the above by noticing diag(xT
t xt)/γ is always non-negative. Using this
important property, we can develop a fast algorithm for the second-order OFS method.
Specifically, we build a MaxHeap data structure to store the B smallest diagonal values of covariance
Σt. The monotonic decreasing property of Σt implies the heap limit should decrease monotonically.
This leads to two major benefits in saving computational cost: (i) we do not need to check those
unchanged elements to see if they are smaller than the heap limit; and (ii) when updating elements
in the heap, only its child nodes need to be updated.
Algorithm 2 shows the details of the proposed fast algorithm for SOFS. Whenever a new feature
arrives and its covariance changes, we proceed to update as follows:
• If the corresponding covariance exists in the heap, adjust its position in the heap;
• Check if it is smaller than the heap limit; if so, replace the root node of the heap by the
current item and set the value of the original root node to be zero; otherwise,
• Simply set the corresponding weight to zero.
Algorithm 2 SOFS: Fast Algorithm for Second-order OFS
1: Input: γ, B
2: Initialize: µµµ1 = 0,Σ1 = I. MaxHeap H on Σ1 with size B
3: for t = 1,...,T
4:
if lt(µµµ) = max(0,1−yt(µµµ ·xt))2 > 0
5:
Calculate βt, gt by (6).
6:
for j = 1,...,d,xt, j ̸= 0
7:
µt+1, j = µt, j −1
2βtΣt, jgl
t,j,
Σ−1
t+1, j = Σ−1
t, j +
x2
t,j
γ
8:
if Σt+1, j ∈H
9:
adjust H to maintain the MaxHeap
10:
elseif Σt+1, j < Hmin
11:
replace Hmin by Σt+1, j and set the weight value of the original root node to be zero
12:
adjust H to maintain the MaxHeap
13:
else
14:
µt+1, j = 0
15: Output: weight vector µµµT and confidence ΣT
# 3.5 ANALYSIS OF TIME AND SPACE COMPLEXITY
The above proposed technique significantly improves the efficiency of existing online feature selec tion techniques. We now analyze the computational complexity of the above algorithms.
Let us denote by d the dimensionality of the weight vector, and m the average number of nonzero features of each sample. For PET, each updating step has to calculate the loss (O(m)), update the model (O(m)), calculate absolute value of the model (O(m)), find the largest B elements according to their absolute values and then set the rest d −B to zero (O(d +d logB)). The overall computational complexity of PET at every step is O(3m+d +d logB). FOFS is similar to PET, with an extra normalization and sparse projection. The extra complexity is O(2d). Computational cost of calculating absolute value is also increased to O(d). Thus, the complexity of FOFS is O(2m + 4d + d logB), which is much more computationally expensive for high dimensional data. Our SOFS only needs to calculate the loss (O(m)), update weight vector and the covariance (O(2m)), and adjust the heap (O(mlogB)). The computational complexity of SOFS at each step is reduced to O(mlogB + 3m), making it far more efficient and scalable when handling ultra-high dimensional sparse data where m ≪d and B ≪d. Even in the worst case where m ≈d, our SOFS with complexity O(d logB+3d) is still more efficient than PET (O(4d +d logB)) and FOFS (O(6d +d logB)), where the improvement even only a constant can still save lots of training time for ultra-high dimensional data. For space complexity, we only consider the space required by the classifiers. Storages for data loading implementation are excluded here. Both PET and FOFS require to keep the weight vector w and its absolute vector v in memory, and thus have space complexity O(2d). SOFS also has space complexity O(2d) for keeping the weight vector and the diagonal elements of confidence matrix Σ in memory. Thus, SOFS shares the same space complexity as the first-order online FS algorithms.
# 4 EXPERIMENTS
In this section, we conduct extensive experiments to evaluate how the number of selected features affects the test accuracy and the training efficiency of different feature selection algorithms on both synthetic and real data on a large scale. We also evaluated the proposed algorithm on several public available medium-scale datasets. The results are shown in the supplementary material.
# 4.1 EXPERIMENTAL SETUP
For the family of online feature selection algorithms, we only run each algorithm by a single pass through the training data if without explicit indication. We compare the proposed algorithm with a set of state-of-the-art algorithms including both online and batch feature selection as follows: • PET: the baseline of OFS by Perceptron with truncation (Wang et al., 2014); • FOFS: the state-of-the-art first-order OFS via sparse projection (Wang et al., 2014); • mRMR: minimum Redundancy Maximum Relevance Feature Selection, a state-of-the-art batch feature selection method (Peng et al., 2005). • Liblinear: a famous library for large linear classification (Fan et al., 2008). We adopt l1SVM for the Embedded feature selection in our experiments. • FGM: a batch Embedded feature generating method (Tan et al., 2014).
For the family of online feature selection algorithms, we only run each algorithm by a single pass through the training data if without explicit indication. We compare the proposed algorithm with a set of state-of-the-art algorithms including both online and batch feature selection as follows:
• PET: the baseline of OFS by Perceptron with truncation (Wang et al., 2014); • FOFS: the state-of-the-art first-order OFS via sparse projection (Wang et al., 2014); • mRMR: minimum Redundancy Maximum Relevance Feature Selection, a state-of-the-art batch feature selection method (Peng et al., 2005). • Liblinear: a famous library for large linear classification (Fan et al., 2008). We adopt l1SVM for the Embedded feature selection in our experiments. • FGM: a batch Embedded feature generating method (Tan et al., 2014).
• For online algorithms, we use hinge loss as the loss function. A five-fold cross validation is conducted to identify the optimal parameters. The experiments were conducted over 10 times with a random permutation of a dataset. For l1-SVM in liblinear, we tune parameter C to select different number of features. For FGM, we follow the settings in (Tan et al., 2014) and set C = 10 for simplicity. For mRMR, we first select a specific number of features and then use the Perceptron to train a classifier. We exploited the advantage of online learning that processes data sequentially and implemented the program with two parallel threads, one for data loading and the other for learning. All experiments were conducted on a PC with Intel i7 CPU @ 3.3 GHz, 16 GB RAM 1.
# 4.2 EXPERIMENTS ON SYNTHETIC DATA
The goal of this set of experiments is to generate synthetic data with ultra high dimensionality in order to examine different aspects of our algorithm in an effective way. Synthetic Data. We follow the settings of FGM and generate two types of synthetic data, namely X1 ∈R100K×20K and X2 ∈R1M×1B to test efficacy, efficiency, and scalability of the algorithms for bi1The source codes for our experiments will be released after the paper is published.
The goal of this set of experiments is to generate synthetic data with ultra high dimensionality in order to examine different aspects of our algorithm in an effective way. Synthetic Data. We follow the settings of FGM and generate two types of synthetic data, namely X1 ∈R100K×20K and X2 ∈R1M×1B to test efficacy, efficiency, and scalability of the algorithms for bi-
The source codes for our experiments will be released after the paper is published
<div style="text-align: center;">Table 1: Summary of synthetic data (“K”,“M”,“B” are thousand, million, and billion, respectively.)</div>
DataSet
#Train
#Test
Dim
IDim1
NDim2
#Feat
X1
100K
10K
20K
200
400
60M
X2
1M
100K
1B
500
500
1B
1 
1 IDim is the dimension of informative features per instance 2 NDim is the dimension of noise features per instance
<div style="text-align: center;">1 IDim is the dimension of informative features per instance 2 NDim is the dimension of noise features per instance</div>
nary classification. Each entry is sampled from the i.i.d. Gaussian distribution N (0,1). To simulate real data, each sample is a sparse vector. The numbers of informative features for the two datasets are 200 and 500 respectively. For each sample, we randomly select 400 dimensions for X1 and 500 dimensions for X2 as noise. To generate labels, we sample a weight vector w∗from the Uniform distribution U (0,1) as the groundtruth weights for features. The label of each sample is determined by y = sign(w∗·x∗), where x∗is a sample without noise. Table 1 summarizes the synthetic datasets. Figure 1 shows the comparisons of accuracy and time cost. Accuracy. According to Figure 1(a), the proposed algorithm outperforms other online feature selection algorithms, showing its efficacy in exploiting informative features. SOFS is superior to FOFS and PET significantly when the number of selected features exceeds the number of informative features. mRMR performs the worst, similar to the observations in (Wang et al., 2014). Batch learning algorithms are superior to online algorithms when number of features is very limited. However, SOFS reaches the best and is comparable to batch feature selection algorithms when the number of selected features exceeds the number of informative feature (200 in X1). To conclude, the proposed algorithm is able to identify the groundtruth geometry of the data.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6a2a/6a2a30de-d4d9-4102-a615-30e5ae242219.png" style="width: 50%;"></div>
Time Cost. Although batch FS algorithms are often more effective, they are significantly slower than online FS algorithms. Among the algorithms, our SOFS can achieve comparable test accuracy as batch FS algorithms with the lowest time cost (only a few seconds). By contrast, liblinear is 10 times slower and FGM is more than 1,000 times slower than SOFS on the dataset. Among online FS algorithms, our method has the best accuracy but requires the least time cost. Scalability on Ultra-High Dimensional Data. Due to the ultra-high dimensionality and billionscale features of X2, we found that it would have to take days to run the existing FS algorithms. We thus only compare SOFT with two variants using two kinds of online learning algorithms on full sets of features (by choosing B = 500 for simplicity): Online Gradient Descent (OGD) and AROW (Crammer et al., 2009). Note that these two baselines were also implemented efficiently using the same framework of SOFS with efficient data structure, but without doing feature selection. Table 2 also shows the evaluation results on X2.
Algorithms
Time Cost (s)
Accuracy
Sparsity (%)
OGD
266.76
99.30
83.52
AROW
396.38
99.50
67.91
SOFS
480.95
99.69
99.995
As seen from the results, SOFT has improved the test accuracy as compared to the two baselines without explicit FS, which verifies that removing irrelevant or noisy features can improve predictive performance. Not only with higher accuracy, SOFS also uses significantly less features (only 0.1% as compared to 16% by OGD and 32% by AROW). In terms of time cost, SOFT took slightly more time cost due to the extra FS process, but only about 8 minutes to train a classifier on this dataset with billion-scale features. These encouraging results again validate that SOFT is efficient, scalable and effective in exploiting informative features on large-scale ultra-high dimensional data.
# 4.3 EXPERIMENTS ON LARGE-SCALE REAL-WORLD DATA SETS
In this part, we evaluate the performance of the proposed SOFS algorithm for three large-scale text classification tasks, as shown in Table 3. The first dataset ‘news‘ (for news group classification) is high dimensional, the second ‘rcv1‘ (for text categorization) is relatively large scale, and the last one ‘url‘ (for suspicious url detection) is large scale and high dimensional. In this experiment, for simplicity, we compare the proposed SOFS algorithm only with PET (due to its low time complexity) and FGM (due to its high accuracy).
<div style="text-align: center;">Table 3: Summary of large-scale real-world datasets in our experiment</div>
DataSet
Feat Dim
Train No.
Test No.
Feat No.
news
1,355,191
10,000
9,996
5,513,533
rcv1
47,152
781,265
23,149
59,155,144
url
3,231,961
2,000,000
396,130
231,249,028
Table 4 shows the experimental results of test accuracy and time cost of the three algorithms. We cannot show the results of FGM on “url” as it was too slow to run (took days to select 20% features). We observe that the performance of SOFS is very close to that of FGM. Both PET and FGM are far more computationally expensive, with FGM even more than an order of magnitude difference. The results further verify the significant advantage of SOFS on large-scale high-dimensional datasets.
<div style="text-align: center;">Table 4: Evaluation on large-scale high-dimensional datasets (ρ is the fraction of selected features).</div>
Table 4: Evaluation on large-scale high-dimensional datasets (ρ is the fraction of selected features
Dataset
ρ
0.005
0.05
0.1
0.2
news
PET
75.31%(52.95s)
71.71%(46.76s)
70.93%(54.27s)
76.55%(61.76s)
SOFS
78.48%(1.73s)
79.3%(2.12s)
79.36%(1.65s)
79.52%(1.45s)
FGM
79.3%(75.40s)
79.71%(751.01s)
79.72%(2540.64s)
79.63%(7587.04s)
rcv1
PET
80.14%(140.15s)
92.35%(79.90s)
93.71%(82.43s)
94.24%(99.95s)
SOFS
87.59%(14.71s)
93.65%(16.40s)
94.23%(15.62s)
94.61%(15.6s)
FGM
94.58%(501.4s)
94.71%(950.1s)
94.76%(1339.0s)
94.81%(2039.1s)
url
PET
98.16%(3816.8s)
98.24%(4750.4s)
98.29%(4957.3s)
98.29%(4878.7s)
SOFS
98.40%(66.7s)
98.62%(68.6s)
98.66%(68.9s)
98.71%(67.1s)
This paper addressed an open challenge of large-scale feature selection with large-scale ultra-high dimensional sparse data, and presented a novel scheme of Second-order Online Feature Selection (SOFS). In contrast to the existing online FS algorithms whose computational complexity is linear with respect to the total feature dimensions, the proposed new SOFS algorithm has a significantly lower computational complexity that is linearly dependent on the average number of nonzero features with each instance. We extensively evaluated empirical performance of the proposed algorithm by comparing it with state-of-the-art online and batch feature selection algorithms on both synthetic and large-scale real datasets. The promising results showed that our new method not only achieved highly competitive prediction accuracy, but also significantly improved computational efficiency, making our method practical for handling large-scale sparse data with ultra-high dimensionality.
REFERENCES
# REFERENCES
Crammer, Koby, Dekel, Ofer, Keshet, Joseph, Shalev-Shwartz, Shai, and Singer, Yoram. Online passive-aggressive algorithms. The Journal of Machine Learning Research, 7:551–585, 2006. Crammer, Koby, Kulesza, Alex, and Dredze, Mark. Adaptive regularization of weight vectors. Machine Learning, pp. 1–33, 2009. Dredze, Mark, Crammer, Koby, and Pereira, Fernando. Confidence-weighted linear classification. In Proceedings of the 25th international conference on Machine learning, pp. 264–271. ACM, 2008. Duchi, John, Hazan, Elad, and Singer, Yoram. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12:2121–2159, 2011. Fan, Rong-En, Chang, Kai-Wei, Hsieh, Cho-Jui, Wang, Xiang-Rui, and Lin, Chih-Jen. Liblinear: A library for large linear classification. The Journal of Machine Learning Research, 9:1871–1874, 2008. Glocer, Karen, Eads, Damian, and Theiler, James. Online feature selection for pixel classification. In Proceedings of the 22nd international conference on Machine learning, pp. 249–256. ACM, 2005. Hoi, Steven C. H., Wang, Jialei, and Zhao, Peilin. Libol: A library for online learning algorithms. The Journal of Machine Learning Research, 15:495–499, 2014. URL http: //LIBOL.stevenhoi.org. Kohavi, Ron and John, George H. Wrappers for feature subset selection. Artificial intelligence, 97 (1):273–324, 1997. Langford, John, Li, Lihong, and Zhang, Tong. Sparse online learning via truncated gradient. The Journal of Machine Learning Research, 10:777–801, 2009. Liu, Huan and Yu, Lei. Toward integrating feature selection algorithms for classification and clustering. IEEE Trans. on Knowledge and Data Engineering, 17(4):491–502, 2005. Peng, Hanchuan, Long, Fulmi, and Ding, Chris. Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 27(8):1226–1238, 2005. Perkins, Simon and Theiler, James. Online feature selection using grafting. In ICML, pp. 592–599, 2003. Saeys, Yvan, Inza, I˜naki, and Larra˜naga, Pedro. A review of feature selection techniques in bioinformatics. bioinformatics, 23(19):2507–2517, 2007. Tan, Mingkui, Tsang, Ivor W., and Wang, Li. Towards ultrahigh dimensional feature selection for big data. Journal of Machine Learning Research, 15(1):1371–1429, 2014. Wang, Jialei, Zhao, Peilin, Hoi, Steven CH, and Jin, Rong. Online feature selection and its applications. IEEE Transactions on Knowledge and Data Engineering, 26(3):698–710, 2014. Wu, Xindong, Yu, Kui, Wang, Hao, and Ding, Wei. Online streaming feature selection. In Proceedings of the 27th international conference on machine learning (ICML-10), pp. 1159–1166, 2010. Xu, Zenglin, Jin, Rong, Ye, Jieping, Lyu, Michael R, and King, Irwin. Non-monotonic feature selection. In Proceedings of the 26th Annual International Conference on Machine Learning, pp. 1145–1152. ACM, 2009. Yu, Lei and Liu, Huan. Feature selection for high-dimensional data: A fast correlation-based filter solution. In ICML, volume 3, pp. 856–863, 2003. Zhao, Zheng, Wang, Lei, Liu, Huan, and Ye, Jieping. On similarity preserving feature selection. Knowledge and Data Engineering, IEEE Transactions on, 25(3):619–632, 2013.
APPENDIX: MORE EXPERIMENTS ON MEDIUM-SCALE REAL DATA SETS In this appendix, we give more extensive experimental results of performance evaluations on a variety of medium-scale real-world datasets.
OVERVIEW OF MEDIUM-SCALE REAL DATA SETS
In this section, we evaluate the performance of online feature selection algorithms on a number of medium-scale public benchmark datasets, as shown in Table 5. The datasets can be downloaded either from Feature Selection website of Arizona State University2 or SVMLin3 (for sparse datasets).
<div style="text-align: center;">Table 5: Medium-scale real datasets in experiments</div>
DataSet
Feat Dim
Train No.
Test No.
Feat No.
relathe
4,322
1,000
427
87,352
pcmac
7,510
1,000
946
55,470
basehock
4,862
1,500
493
101,974
ccat
47,236
13,149
10,000
994,133
aut
20,072
40,000
22,581
1,969,407
real-sim
20,958
50,000
22,309
2,560,340
Figure 2 shows the test accuracy of different algorithms. By examining the online algorithms, we found that Perceptron (“PET”) with a simple truncation does not work well, while FOFS is much better than PET in most cases. However, we observe that performance of FOFS is not stable. The variance of FOFS is much larger than those of the other two online algorithms on half of the mediumscale datasets. The proposed SOFS method is able to learn a more compact classification model. With the same number of selected features, SOFS is able to achieve the higher test accuracy results.
<div style="text-align: center;">Table 6: Comparison of SOFS with mRMR on medium-scale datasets</div>
Dataset
B
100
200
300
400
500
relathe
mRMR
74.19
77.87
78.92
79.13
79.60
SOFS
71.38
78.81
81.34
82.39
82.91
pcmac
mRMR
87.95
90.34
89.93
91.49
91.10
SOFS
89.76
92.65
93.28
93.75
94.02
basehock
mRMR
93.78
95.15
95.03
95.25
94.89
SOFS
90.34
94.52
95.86
96.41
96.68
ccat
mRMR
82.75
85.71
86.42
86.94
87.40
SOFS
82.76
86.35
87.94
89.00
89.75
aut
mRMR
92.41
93.87
94.09
94.59
94.55
SOFS
74.72
81.71
85.89
97.67
99.75
real-sim
mRMR
85.44
88.51
89.71
90.84
94.55
SOFS
83.29
86.77
89.38
90.71
91.59
Besides, SOFS is comparable to batch FS algorithms when accuracy saturates with number of features. We find that FGM is able to perform well with rather few features. Liblinear in this case
2http://featureselection.asu.edu/datasets.php 3http://vikas.sindhwani.org/svmlin.html
shows a very interesting phenomenon in that the test accuracy first increases rapidly with more selected features, but after a certain stage where the accuracy of other algorithms begins to saturate, the accuracy of Liblinear tends to drop considerably. This implies that Liblinear may be more sensitive to irrelevant features or noises. We show the comparison of SOFS with mRMR separately in Table 6 (as mRMR was only able to output at most 500 selected features). From these results, we can observe that mRMR is better when the number of features is less. The accuracy of SOFS increases quickly and surpasses mRMR with more selected features. This is consistent to the above results. Note that mRMR is better than SOFS on the dataset “real-sim”. In Figure 2, all online FS algorithms fail to train a good model with only 500 features on “real-sim”. Their performance increases quickly and is expected to outperform mRMR with more features. The comparison again verifies the advantage of batch learning algorithms on very small number of selected features. However, when more features are selected, the proposed online feature selection becomes more accurate than mRMR.
shows a very interesting phenomenon in that the test accuracy first increases rapidly with more selected features, but after a certain stage where the accuracy of other algorithms begins to saturate, the accuracy of Liblinear tends to drop considerably. This implies that Liblinear may be more sensitive to irrelevant features or noises.
EVALUATION OF TIME COST
Figure 3 shows the time cost comparison of feature selection methods on medium-scale data. First of all, we observe that FOFS took slightly higher time cost than PET despite achieving better accuracy. The extra time cost is more obvious when data dimensions get higher. Further, we observe that the time cost first decreases and then increases with more selected features. This is due to the fact that when the number of selected features is too small, large number of mistakes are made and the model has to update frequently. With more features, the prediction accuracy can be improved and thus less update is performed, resulting in the decreased time costs. Note the time costs on the later three datasets, which are of relatively high dimension. It shows the great advantage of our proposed algorithm on high dimensional data. This is consistent with the analysis in the paper that complexity of SOFS is linearly dependent on the number of non-zero features, while PET and FOFS are linearly dependent on the feature dimension.
<div style="text-align: center;">le 7: Time Cost Comparison of SOFS with mRMR (#features = 500)(sec</div>
Dataset
relathe
pamac
basehock
ccat
aut
real-sim
SOFS
0.03
0.03
0.04
0.48
0.78
1.24
mRMR
1733
1429
1584
2205
1486
1403
As to batch learning algorithms, liblinear is quite similar to first-order online algorithms, but is much more than that of SOFS. Time cost of FGM is about an order of magnitude higher than liblinear. mRMR is the most inefficient among all the algorithms. We show time cost of mRMR and SOFS to select 500 features in TABLE 7. Even on the smaller dataset “relathe”, it takes over 1,700 seconds to select 500 features, while SOFS requres only 0.03 seconds. To conclude, SOFS is the most efficient one among all the algorithms in our experiments.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7ec5/7ec5bb70-4bae-46dc-8ae6-3073dcd1e265.png" style="width: 50%;"></div>
<div style="text-align: center;">2: Test Accuracy of Feature Selection Algorithms on Medium-scale real w</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f16a/f16adfb0-01c0-4058-bb5f-bab6e95a68f4.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Time Cost of Feature Selection Algorithms</div>
