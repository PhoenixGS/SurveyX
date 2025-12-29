# SecretGen: Privacy Recovery on Pre-Trained Models via Distribution Discrimination
Zhuowen Yuan1, Fan Wu1, Yunhui Long1, Chaowei Xiao2, and Bo Li1
1 University of Illinois Urbana-Champaign 2 Arizona State University
Abstract. Transfer learning through the use of pre-trained models has become a growing trend for the machine learning community. Consequently, numerous pre-trained models are released online to facilitate further research. However, it raises extensive concerns on whether these pretrained models would leak privacy-sensitive information of their training data. Thus, in this work, we aim to answer the following questions: “Can we effectively recover private information from these pre-trained models? What are the sufficient conditions to retrieve such sensitive information?” We first explore different statistical information which can discriminate the private training distribution from other distributions. Based on our observations, we propose a novel private data reconstruction framework, SecretGen, to effectively recover private information. Compared with previous methods which can recover private data with the ground truth label of the targeted recovery instance, SecretGen does not require such prior knowledge, making it more practical. We conduct extensive experiments on different datasets under diverse scenarios to compare SecretGen with other baselines and provide a systematic benchmark to better understand the impact of different auxiliary information and optimization operations. We show that without prior knowledge about true class prediction, SecretGen is able to recover private data with similar performance compared with the ones that leverage such prior knowledge. If the prior knowledge is given, SecretGen will significantly outperform baseline methods. We also propose several quantitative metrics to further quantify the privacy vulnerability of pre-trained models, which will help the model selection for privacy-sensitive applications. Our code is available at: https://github.com/AI-secure/SecretGen.
Keywords: Privacy; Pre-trained models; Transfer learning
# 1 Introduction
As machine learning has achieved great successes in different domains, such as robotics [24], audio recognition [7], and face recognition [15], how to train the learning models efficiently given the available large-scale dataset becomes a timely problem. Transfer learning, which focuses on transferring knowledge across domains, is a promising learning paradigm [2]. In particular, many pretrained models are available currently, such as TensorFlow Hubs [1] and PyTorch Hubs [22], which can be flexibly used for fine-tuning later for different
downstream tasks. As a result, the training paradigm with transfer learning has enabled efficient usage of the large-scale dataset without requiring training every model from scratch. However, such an efficient transfer learning paradigm also leads to additional privacy concerns. For instance, if the training data of the pre-trained models contain privacy-sensitive information, an adversary who downloads the pre-trained models could potentially perform different privacy attacks to infer the private information. In particular, membership inference attacks [18,19] have been studied to infer whether a private instance is in the training set, and model inversion attacks have been studied to reconstruct the private training instances under certain assumptions [28,11,10,26], which raises more privacy and safety concerns. To better understand the privacy vulnerabilities of such pre-trained models, a comprehensive analysis of different types of privacy attacks, especially the severe model inversion attacks, is required. Currently, there are several limitations of existing privacy model inversion attacks. First, the current state-of-the-art model inversion attack (i.e., GMI) [28] requires the ground truth label of the reconstructed instances, which is less practical. Furthermore, it is a known challenging problem to label the generated instances based on GANs [12]. Second, many existing model inversion attacks require whitebox access to the target pre-trained model, making it less practical in real-world applications. Thus, in this paper, we mainly aim to ask: Can we reconstruct private sensitive training instances without requiring such information? To answer it, we propose a general private data recovery framework SecretGen, which consists of a generation backbone, a pseudo label predictor, and a latent vector selector. We first use a pseudo label predictor to generate a pseudo label for each private instance. Specifically, we randomly sample latent vectors and feed them into the generation backbone to get recovered instances. To stabilize prediction quality, we apply different transformations (e.g.cutouts) to such instances before feeding them into the targeted model to get the final predicted pseudo labels. We then propose a latent vector selector via a proposed selection algorithm to further optimize and constrain the recovery space. Finally, we perform joint optimization to train the end-to-end framework as shown in Fig. 1. We conduct comprehensive experiments to evaluate the proposed SecretGen compared with multiple baselines. We show that SecretGen significantly outperforms baselines given the same ground truth label. Even without such information, SecretGen still achieves comparable performance compared to baselines which leverage the ground truth label information. In addition, to evaluate the performance of recovered data on downstream tasks, we propose different evaluation protocols considering different usage of the recovered private data, and we show that our observations are consistent for different protocols. We also evaluate the robustness of SecretGen against the purification defense [27]. Finally, we perform different ablation studies to show the effectiveness of our design choices. We make the following contributions:
# 2 Related Work
Revealing privacy-sensitive information from a trained model has aroused extensive research interest. Membership inference attacks and model inversion attacks are two major categories of such attacks. In membership inference attacks [18,19], the adversary aims to decide whether a sample is a member of the training set, while in model inversion attacks [28,11,10,26], the adversary attempts to reconstruct the training set under certain assumptions. [11] was the first to propose model inversion attacks aiming at recovering private training data. The authors demonstrated that personal genetic markers could be effectively recovered given the output of the model and auxiliary knowledge. [10] extended model inversion to more complex models, including shallow neural networks for face recognition. The recovered data with their proposed method are identified as the original person at a much higher rate than random guessing. However, the reconstructed images are blurry and not visually recognizable to humans. [26] proposed a training-based attack by training an auto-encoder on public data. The attack can be performed with blackbox accesses to the target model and partial (truncated) model predictions. More recently, [28] proposed generative model inversion attack (GMI). The authors distill public knowledge by training a conditional GAN on public data and then solve an optimization problem to maximize the probability of the recovered image for the ground truth class label. GMI significantly outperforms previous methods in re-identification rate of the recovered data, as well as guaranteeing the recovered data are visually plausible. However, they still require the ground truth label for the target image and whitebox access to the victim model, which is often not accessible to the adversary. Another recent work distributional model inversion attack (DMI) [3] recovers the private data distribution for each target class by constructing representative samples. However, DMI does not support recovering every private instance given its non-sensitive version (i.e. instance-level model inversion), which is the adversary’s goal in our setting.
# 3 Methodology
<div style="text-align: center;">3 Methodology</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4889/48896cd6-2df7-4162-a5d2-767e07e3db41.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 1. Overview of the proposed SecretGen. The blue modules represent the proposed algorithms. The Target Model could allow either whitebox or blackbox access.</div>
g. 1. Overview of the proposed SecretGen. The blue modules represent the prosed algorithms. The Target Model could allow either whitebox or blackbox access.
# 3.1 Problem Formulation
We focus on recovering the privacy-sensitive training data based on the trained classification models. Throughout the paper, we will refer to the model that is subjective to attacks as the target model F, which is trained on private training data Dtrain pri , aiming to perform evaluation on private test data Dtest pri . The target model returns a prediction vector F(x) given an input instance x. The prediction vector represents a probability distribution over C classes, where C denotes the number of classes of the whole private dataset Dpri = Dtrain pri ∪Dtest pri . The adversary’s goal is to recover the private training data Dtrain pri given the trained target model F and certain prior information, e.g., partially corrupted images from Dtrain pri . In particular, such corrupted images only contain non-sensitive background information (pixels) xns with the sensitive region xs cropped out. These corrupted images are usually easy to obtain, given that such corruption is often applied to protect the privacy of individuals in practice [28]. Specifically, in our evaluation, we consider cropping the whole face using two face datasets, leaving only the non-sensitive background regions (Section 4). Regarding the adversary’s ability, we consider (1) whitebox access to the target model, where all parameters and intermediate computations of the target model are visible to the adversary, and (2) blackbox access to the target model, where the adversary can only obtain the final prediction from the target model F. Additionally, we assume that the adversary also has access to some public data Dpub from the similar distribution for general training purpose.
# 3.2 Method Overview
An overview of SecretGen is illustrated in Fig. 1, where SecretGen takes nonsensitive information xns as input and returns the recovered images that contain privacy-sensitive training information (e.g., human faces). SecretGen is composed of three components: generation backbone, pseudo label predictor, and latent vector selector, which are jointly optimized under a unified framework. The generation backbone leverages a conditional GAN trained on public data as a backbone to generate realistic images based on the prior information (e.g., cropped images), and the generation process is controlled by the latent vector z sent to the GAN’s generator G. The pseudo label predictor predicts the most possible pseudo label for each recovered private image based on the distributional statistics of recovered images. The latent vector selector selects the optimal latent vector ˆz which is the most likely to contain privacy-sensitive information based on the proposed selection algorithm. Finally, we perform joint optimization on the selected ˆz, taking the pseudo label provided by the pseudo label predictor as the prediction target, to reconstruct image G(ˆz∗, xns). In the next following sections, we will describe each component in detail.
# 3.3 Generation Backbone of SecretGen
To recover the privacy-sensitive training data, we train a generation backbone for conditional image recovery on public data Dpub. In particular, we will start from certain prior knowledge, such as the corrupted private data containing only the nonsensitive information xns. We then perform the same corruption operation corr on Dpub to construct the training set for the generation backbone: Dpub corr = {corr(x)|x ∈Dpub}. Next, we train a conditional GAN which is composed of two networks: generator G and discriminator D. G is conditioned on xns ∈Dpub corr and z is the latent vector which is sampled from a prior distribution during training. Throughout the paper, we use the prior distribution as standard Gaussian distribution. We leverage the Wasserstein-GAN loss [13] for GAN training:
We also incorporate a diversity loss term Ldiv [25] for training the generator to prevent mode collapsing by sampling different latent vectors, say, z1 and z2:
where f is the feature extractor of the target model, which returns the feature embeddings of the input images in the whitebox setting. In the blackbox setting, we use a feature extractor trained on public data fpub for this process. The overall loss term for the generator is as following:
(1)
(2)
(3)
After the generation backbone is trained, we freeze the parameters for both G and D before we enter the next stage. We denote ˆx as the recovered image, i.e., ˆx = G(z, xns).
# 3.4 Pseudo Label Predictor
The main challenge in this data reconstruction process is that we have no knowledge about the ground truth label of the private images (related work assumes that they have access to the ground truth label [28,26,11,10], while we do not). To tackle this problem, we propose a pseudo label predictor which infers the label prediction with proposed discrimination metrics. We will first introduce the design of our discrimination metric, and then we elaborate on how the pseudo label predictor is optimized. Discrimination Metric. Given the certain prior knowledge xns, we randomly sample n latent vectors {zi}n i=1 from the prior distribution. We generate n recovered images using our generation backbone: {ˆxi}n i=1, where ˆxi = G(zi, xns). In order to improve the prediction stability, we consider prediction under different transformations. Concretely, let the list of considered transformation functions be T = {ti}m i=1. On each recovered image ˆxi, we perform m transformations independently to obtain m transformed images {˜xj i}m j=1, where ˜xj i = tj(ˆxi). We additionally define ˜x0 i = ˆxi. Let Fc(·) denote the model’s prediction confidence for class label c based on target model F. We define the discrimination metric M on label c as follows:
The discrimination metric returns a score indicating how likely it is for a label c to be the consistent prediction across different transformations. Based on existing studies of contrastive learning [4], we will select the class c with the highest discrimination metric score as the final label prediction. In particular, we define the list of transformations as a sequence of fix-sized cutouts. We split an image into fix-sized patches and define tj as the transformation that cuts out the j-th patch of the given image, as illustrated in Fig. 2. Intuitively, the discrimination metric M(c; n, m) should preserve the following properties. First, M(c; n, m) is likely to have a higher score when c equals the label associated with the corrupted image xns since the model has learned some correlation between the non-sensitive background information in xns and the label of the original image. Such correlation should be stronger if the target model is more overfitted to private training data. Second, when the recovered image ˜xj i is close to the training data, Fc(˜xj i) should be consistently higher on the correct label because training data are often more resistant to transformations than non-training data [6]. Based on these intuitions, we use the discrimination metric as the foundation of the pseudo label predictor in SecretGen.
(4)
Pseudo Label Predictor. Given the discrimination metric M, we next describe in detail how we leverage M to infer the pseudo prediction label considering different sampled latent vectors, which aims to approximate the ground truth. We first sample a set of n latent vectors randomly and compute M for all class labels. The pseudo label predictor chooses the label with the maximum discrimination metric score as the predicted label ˆc:
(5)
We defer the detailed algorithm for label prediction with M to the appendix. Note that there are various design choices for the discrimination metric M, e.g., the average confidence on only the recovered images without including their transformed versions. It is clear that more advanced M will provide more accurate pseudo label predictors. We will analyze the performance of the pseudo label predictor given different designs of M in Section 4.5.
# 3.5 Latent Vector Selector
In addition to the availability of ground truth labels, another challenge during private data recovery is that we may not have whitebox access to the target model. In systems where machine learning is used as a service (MLaaS), the adversary can only query the model and the prediction vector is returned from the service provider. All internal computations and model parameters are unknown to the adversary. In previous work [28], the adversary can directly optimize the latent vector z to maximize the target model’s confidence given a known ground truth label, which is less practical. Without the whitebox access, performing back-propagation with the target model is infeasible in our practical case. To tackle this problem, we design a latent vector selector to first randomly sample n random latent vectors, and then select the ones which lead to their recovered data classified as the predicted label from the pseudo label predictor. If there is no latent vector that leads to the recovered images which can be classified as the predicted label consistently, the selector returns a randomly sampled latent vector from the prior distribution. Otherwise, it returns the latent vector which has the highest confidence of the predicted label. We omit the detailed algorithm to the appendix.
# 3.6 SecretGen Optimization
To put every proposed component within SecretGen together, we perform joint optimization to maximize the consistent label prediction likelihood of recovered images indicated by the discrimination metric (i.e., identity loss), while keeping the recovered images realistic (i.e., discriminator loss). In the whitebox setting, we perform backpropagation on the target model with identity loss Lid. Lid
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/05b1/05b19216-d0f3-443a-adb6-9ad595f48cfe.png" style="width: 50%;"></div>
Fig. 2. Sequential cutout for the recovered image as transformations. The image is first split into m fix-sized patches. Operations of cutting out each patch are viewed as transformations respectively.
encourages the generated images to achieve consistently high label predict likelihood given the target model for class label c.
We utilize discriminator loss as regularization to penalize unrealistic images.
  h Then we initialize z with ˆz returned by our latent vector selector and optimize z with the following objective function:
In the blackbox setting, we perform the latent vector selection optimization only with the discriminator loss since the target model is not locally accessible:
Note that in the blackbox setting where we have the ground truth labels, the identity loss is still minimized by the latent vector selector through random sampling based on the prediction vector of the target model, guaranteeing that the recovered image is close to the density region of the ground truth identity.
# 3.7 Discussion
Our proposed SecretGen works under a wide range of scenarios regarding different types of prior knowledge. See Table 1 for the scenarios under which SecretGen and existing methods can be applied. Although EMI theoretically works in blackbox cases with ground truth labels, its performance and efficiency dramatically suffer against deep models. Although SecretGen still requires non-sensitive private data as prior knowledge, such assumption is realistic as image corruption is often leveraged for privacy protection by individuals [28]. Furthermore, if the knowledge of ground truth labels is available, it can be incorporated into SecretGen conveniently. More details are deferred to the appendix. In conclusion, SecretGen is more practical without requiring whitebox access to the target model or the ground truth label. In addition, SecretGen is very efficient and applicable to high-dimensional image data considering deep models as the target model as shown in our evaluation (Section 4).
Table 1. Comparison with existing methods on the information required by the adversary to recover private training data. The symbol ✓✗means that in theory, the method can work without the information, but the actual performance on deep models is bad.
Methods Non-sensitive Data? Whitebox Access? Ground Truth Label?
PII [25]
✓
✗
✗
EMI [10]
✗
✓✗
✓
GMI [28]
✓✗
✓
✓
SecretGen
✓
✗
✗
(6)
(8)
(9)
# 4 Experiments
In this section, we first present the experimental setup. Then, we introduce the evaluation protocols and report the attack performance respectively. We also evaluate the robustness of SecretGen against the purification defense [27]. In the end, we describe some ablation studies to better understand our method.
# 4.1 Experimental Setup
Datasets. We evaluate SecretGen on two face datasets: (1) CelebA [20] which contains 202,599 face images of 10,177 identities. We filter out those identities with 25 or fewer images and randomly select 25,000 images of 1,000 identities as private data Dpri. We also randomly select 50,000 images of 2,000 identities from the rest as adversary’s public data Dpub. There exist no overlapped identities between Dpub and Dpri. (2) FaceScrub [21] which consists of 106,863 face images of 530 identities. We use the images of 250 identities as Dpri and images of another 250 identities as Dpub. We further split Dpri into Dtrain pri and Dtest pri for training and testing. All the images are cropped and resized to 64 × 64. Prior Information. We consider two types of prior information that the adversary has access to: corrupted images by center mask and face T mask following the standard setting in [28]. Center mask blocks the center part of the private image, but the mouth information may still be exposed. Face T mask completely hides the identity revealing features of the face image. Model Architectures. We perform evaluation on target models with various architectures: (1) VGG16 [23]; (2) ResNet152 [15]; (3) face.evoLVe [5] with an IR50 backbone; (4) ViT-B 16 [9] We utilize IR152 [5] as the evaluation model to predict the identity of input images. Both VGG16 and ResNet152 are pretrained on ImageNet [8]. face.evoLVe and the evaluation model are pre-trained on MS-Celeb-1M [14]. ViT-B 16 is pre-trained on ImageNet21k [8]. The architecture of SecretGen generation backbone is adopted from [28]. Baselines. We compare SecretGen with the state-of-the-art model inversion attack GMI [28]. GMI assumes the adversary has access to the ground truth labels and performs optimization with identity loss and discriminator loss. We also compare our results with pure image inpainting (PII) [25], which only optimizes the discriminator loss for generating realistic images. Latent vectors of both GMI and PII are sampled randomly from Gaussian distribution. We do not compare with EMI [10], since it has been demonstrated in [28] that the effectiveness of EMI is quite limited against deep models. We defer additional details regarding model training and attack to the appendix.
# 4.2 Evaluation Protocols
We consider two principles for evaluating the privacy attack performance: “how much privacy sensitive identity information can be recovered” and “how well the recovered data can perform in downstream tasks”.
Corresponding to the two principles, we evaluate the privacy attack performance by attack accuracy under the following two protocols: – Protocol 1: Train the evaluation model on the private data, and evaluate on the recovered data. – Protocol 2: Train the evaluation model on the recovered data, and evaluate on the private data.
Protocol 1 was introduced in [28], which evaluates instance-level privacy recovery. However, we demonstrate that even if some instances are not recovered correctly, the recovered data can be used for downstream tasks, e.g., training another classification model. The adversary can potentially use the trained evaluation model for malicious purposes, e.g., performing unauthorized face recognition on private identities with significantly higher accuracy than the target model itself. Thus, we propose Protocol 2, which aims to evaluate distribution-level privacy recovery. In addition, a common goal of the adversary to reconstruct the private data is to leverage such data for other downstream tasks, and therefore Protocol 2 explicitly reflects the utility of the recovered data. For Protocol 1, we train the evaluation model on Dtrain pri and the resulting evaluation model achieves 98.0% classification accuracy over the private identities on Dtest pri . For Protocol 2, we first perform the attack on all corrupted private images Dpri—for each corrupted image xns ∈Dpri, we recover an image ˆx = G(ˆz∗, xns) via SecretGen, with label ˆc = arg maxc∈[1,C] Fc(ˆx). We then compose the recovered images into a recovered private set Drec, which is separated into Dtrain rec and Dvalid rec by 4:1. We train the evaluation model on Dtrain rec with Dvalid rec as the validation set. We then evaluate the model performance on Dtest pri . We also report Peak Signal-to-Noise Ratio (PSNR) [16] between original and recovered private data, which reflects the pixel-level reconstruction quality of our attack. Note that the recovered data can still reveal identity information even if the generated image is not close to the ground truth image pixel-wise. For example, the recovered images can exhibit variations in pose and light condition while keeping the identity.
# 4.3 Attack Performance
Whitebox Attacks. Table 2 compares the performance of SecretGen with baseline methods on CelebA. See the appendix for results on FaceScrub. We can see that SecretGen significantly outperforms GMI under both Protocol 1 and Protocol 2 if the ground truth label is given. Without such information, with the proposed pipeline especially the pseudo label predictor, SecretGen still achieves comparable performance with GMI under Protocol 1. Under Protocol 2, GMI with ground truth label performs better than SecretGen without ground truth label. The reason is that if the predicted pseudo label is incorrect, our pseudo label predictor and optimization push the recovery to be closer to the wrong identity. However, we still outperform PII by a large margin. We also observe that attack accuracy under Protocol 2 is much higher than that under Protocol 1. The reason is that Protocol 1 and 2 work at different
<div style="text-align: center;">Table 2. Whitebox attack performance on CelebA. See the Ground Truth Label column for whether ground truth label is provided for each attack method.</div>
Target Model Methods Ground Truth
Label
Center Mask
Face T Mask
Protocol 1 Protocol 2 PSNR Protocol 1 Protocol 2 PSNR
VGG16
PII
✗
0.423
0.561
27.583
0.166
0.363
26.276
GMI
✓
0.569
0.955
27.587
0.305
0.928
26.240
SecretGen
✗
0.584
0.928
27.955
0.312
0.793
26.632
SecretGen
✓
0.639
0.965
28.071
0.377
0.931
26.821
ResNet152
PII
✗
0.403
0.719
26.892
0.170
0.555
26.117
GMI
✓
0.556
0.965
27.177
0.295
0.946
26.482
SecretGen
✗
0.595
0.948
27.506
0.324
0.884
26.821
SecretGen
✓
0.618
0.971
27.587
0.349
0.945
26.967
face.evoLVe
PII
✗
0.267
0.455
27.317
0.122
0.343
26.356
GMI
✓
0.595
0.946
27.444
0.467
0.935
26.563
SecretGen
✗
0.551
0.841
27.613
0.274
0.630
26.562
SecretGen
✓
0.788
0.963
27.781
0.695
0.954
26.827
ViT
PII
✗
0.380
0.389
26.698
0.173
0.306
26.377
GMI
✓
0.482
0.893
24.907
0.214
0.715
24.624
SecretGen
✗
0.451
0.634
26.811
0.246
0.528
26.471
SecretGen
✓
0.551
0.950
26.607
0.326
0.913
26.609
levels: Protocol 1 evaluates how much “detailed” information the recovered images contain, while Protocol 2 evaluates how much distributional information we can recover by training another model based on the reconstructed data. Clearly, Protocol 2 is relatively easier by recovering distributional level information and thus achieves higher scores. We believe such observations will inspire interesting future work and narrow down such a gap. Blackbox Attacks. In the blackbox setting, the adversary is not capable of performing backpropagation with the target model. We make the following changes to our attack pipeline: (1) In Section 3.3, when training the generation backbone, we use a public feature extractor from [5] pre-trained on MS-Celeb-1M to substitute the target model for extracting the feature embeddings in computing the diversity loss (Ldiv, Eqn. (2)); (2) In Section 3.6, when performing SecretGen optimization, we remove the identity loss Lid and optimize the selected latent vector only with discriminator loss Ldisc. Table 3 compares our results with PII under the blackbox setting on CelebA. The only difference for PII under blackbox and whitebox scenarios is whether the target model is accessed when training the generation backbone. We can see that with the ground truth labels, SecretGen significantly outperforms PII. Without ground truth labels, which is the most general case, we still outperform PII by a large margin. As far as we are concerned, we are the first to propose an effective model inversion attack against deep classification models under the blackbox case without ground truth label. We note that GMI (with ground truth label) performs better on face.evoLVe than SecretGen (without ground truth label), as shown in Table 2 and Table 3. Under this setting, attack performance is largely dependent on the pseudo label predictor. We demonstrate that the label prediction accuracy of face.evoLVe is significantly lower than that of VGG16 and ResNet152 in the appendix. We believe the reason is that face.evoLVe is less overfitted due to the difference in
Table 3. Blackbox attack performance on CelebA. We report results for both cases where the adversary has or does not have ground truth labels. (Note: GMI does not support blackbox attack, and PII in the blackbox setting does not use the target model.)
Methods Target Model Ground Truth
Label
Center Mask
Face T Mask
Protocol 1 Protocol 2 PSNR Protocol 1 Protocol 2 PSNR
PII
Any
✗
0.216
0.759
27.319
0.081
0.484
25.705
SecretGen
VGG16
✗
0.351
0.915
27.638
0.164
0.837
26.045
✓
0.380
0.955
27.737
0.377
0.927
26.821
ResNet152
✗
0.334
0.933
27.737
0.152
0.765
26.144
✓
0.347
0.959
27.840
0.172
0.886
26.284
face.evoLVe
✗
0.447
0.711
27.568
0.156
0.353
25.787
✓
0.603
0.894
27.694
0.305
0.586
26.002
ViT
✗
0.285
0.709
27.480
0.119
0.685
25.828
✓
0.335
0.924
27.665
0.160
0.902
26.123
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/14fd/14fd24d1-c1bd-48ea-ad29-d5761d43f77d.png" style="width: 50%;"></div>
Fig. 3. Qualitative results of SecretGen on CelebA. “bb”/“wb” indicates the method requires blackbox/whitebox access to the model. “gt” indicates the method requires ground truth labels. pre-training datasets. (face.evoLVe is pre-trained on MS-Celeb-1M while others are on ImageNet.) Qualitative Results. In Fig. 3 we exhibit the images recovered with SecretGen on CelebA to demonstrate that our recovered images are both identityrevealing and visually plausible. We also show qualitative results of PII and GMI for comparison. From the figure, we see that although all of the three methods generate realistic images, PII cannot effectively recover the original identity of private data, while SecretGen is more effective in identity revealing. More examples are shown in the appendix.
# 4.4 Robustness Evaluation
We evaluate the robustness of our proposed method against purification defense [27], which has been shown to effectively defend against model inversion attacks while inducing negligible utility loss. We use Purifier I in [27] which is specialized for model inversion attacks. We follow the default architectures and settings for training the purifier. See Table 4 for quantitative results on CelebA against VGG16 under the blackbox setting. We also assume the ground truth label is not provided. We do not evaluate the whitebox setting because the ad-
<div style="text-align: center;">Table 4. Robustness evaluation for SecretGen against prediction purification on CelebA. Target model: VGG16. Blackbox setting.</div>
Methods
Center Mask
Face T Mask
Protocol 1 Protocol 2 PSNR Protocol 1 Protocol 2 PSNR
PII
0.216
0.759
27.319
0.081
0.484
25.705
SecretGen
0.351
0.915
27.638
0.164
0.837
26.045
SecretGen (purified)
0.328
0.913
27.590
0.151
0.747
26.007
versary can simply remove the purifier and directly attack the original private model. It can be seen that attack accuracy slightly decreases after the defense, but still outperforms the baseline by a large margin. Therefore, our method is robust against [27].
# 4.5 Ablation Studies
Discrimination Metrics. As discussed in Section 3.4, there may exist various choices for the discrimination metric. One intuitive choice may be derived by removing the transformations from our current discrimination metric M (Eqn. (4)), and the simplified discrimination metric is defined as follows:
We perform an end-to-end ablation study on face.evoLVe and CelebA. We remove the transformations in our pseudo label predictor and substitute M with M′. Quantitative results on face.evoLVe are shown in Table 5. See the appendix for results regarding other model architectures. We conclude that incorporating transformations improves the performance of our framework for most model architectures that we used for evaluation.
Table 5. Attack accuracy of SecretGen with and without transformations on CelebA Evaluated on 3,200 private instances under Protocol 1. Target model: face.evoLVe.
Metric
Center Mask
Face T Mask
Attack Acc PSNR Attack Acc PSNR
w/o transformation
0.528
27.505
0.256
26.527
w/ transformation
0.550
27.522
0.273
26.527
To further understand why and how transformations help, we compare the performance of pseudo label predictor equipped with M and M′. We evaluate the performance of pseudo label predictor using label prediction accuracy, which measures the percentage of the predicted labels matching the ground truth labels. We plot out the label prediction accuracy with M and M′ on 3,200 recovered images for face.evoLVe with varying n in Fig. 4. We observe that our pseudo label predictor can predict the pseudo labels more accurately if transformations are incorporated. See the appendix for results of other model architectures.
(10)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e1a9/e1a94ff5-be7d-47cd-b631-27d2d81e0f3e.png" style="width: 50%;"></div>
Fig. 4. Label prediction accuracy with and without transformations on CelebA. We plot the label prediction accuracy w.r.t. the number of random latent vectors n. Target model: face.evoLVe.
<div style="text-align: center;">Fig. 4. Label prediction accuracy with and without transformations on CelebA. We plot the label prediction accuracy w.r.t. the number of random latent vectors n. Target model: face.evoLVe.</div>
Data Transformations. Next, we discuss the performance of various data transformations on CelebA. We plot out label prediction accuracy w.r.t. n for various transformations including the proposed sequential cutout, horizontal flipping, gray-scale, and color jittering in Fig. 5. We also plot the results without transformations. We can see that sequential cutout performs better than other transformations in terms of label prediction accuracy. Although it is also possible to adopt other transformations within our pipeline, it is non-trivial to select the best hyper-parameters for other transformations (e.g., cropping and color jittering). We leave the analysis of how different transformations impact attack performance as future work. Overfitting Levels. We also evaluate the impact of higher overfitting levels of the target model on the performance of SecretGen, since the overfitting phenomenon is key to model inversion attacks. Note that results reported in Table 2 and Table 3 are based on standard well-trained models. We demonstrate that highly overfitted models are more vulnerable to our proposed attack. We describe the relevant experiment setup and quantitative results in the appendix.
# 5 Conclusion
In this paper, we propose an effective private data recovery framework SecretGen, which can effectively recover private information under a wide range of scenarios. To our full knowledge, we are the first to propose an effective model inversion attack without prior knowledge of ground truth labels, which can achieve comparable results with previous methods that require ground truth labels. If we are given such prior knowledge, we significantly outperform previous methods. Our attack can also be applied under the blackbox setting where the target model is provided as a service and not locally available. We perform a comprehensive analysis of the performance of SecretGen and our design choices. We also demonstrate that our attack is robust against the purification defense. We hope to raise people’s concerns about possible negative effects of releasing pre-trained models online. For future work, we are interested in whether we can perform privacy recovery simply with the target model and develop defenses against our attack. Acknowledgements. This work is partially supported by NSF grant No.1910100, NSF CNS No.2046726, C3 AI, and the Alfred P. Sloan Foundation.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c3c5/c3c564f2-8dcf-45f4-8dda-3de2dd2fadb6.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 5. Label prediction accuracy with different transformations on CelebA. We plot the label prediction accuracy w.r.t. the number of randomly sampled latent vectors n. Target model: face.evoLVe.</div>
Fig. 5. Label prediction accuracy with different transformations on CelebA. We plot the label prediction accuracy w.r.t. the number of randomly sampled latent vectors n. Target model: face.evoLVe.
# References
# A Additional Algorithm Details
In Section 3.4 and Section 3.5, we introduced pseudo label predictor and latent vector selector of SecretGen. In this section, we concretely demonstrate the detailed algorithms for these two components.
# A.1 Detailed Algorithm for Pseudo Label Predictor
Algorithm 1 elaborates how SecretGen’s pseudo label predictor infers a pseudo label for the target private image. The pseudo label predictor randomly samples n latent vectors from the prior distribution of the generation backbone iteratively. For each latent vector, we perform m transformations to the recovered image. We compute the average confidence for each class label c ∈[1, C] on the recovered image and their transformed versions. Then we select the class label ˆc with the highest average confidence as the predicted pseudo label.
Algorithm 1 Label Prediction
Input: xns: corrupted private data, T
= {ti}m
i=1: m transformations, F: target
model, n: number of randomly sampled latent vectors, C: number of classes
v = 0
\triangleright Initialize v to zero vector of length C
for i ←1 to n do
zi ∼N(0, 1)
\triangleright Randomly sample one latent vector from Gaussian
˜x0
i ←G(zi, xns)
\triangleright Generate the recovered image with the generation backbone
˜xj
i ←tj(˜x0
i ) for j = 1 . . . m
\triangleright Generate m transformed images
v ←v + �m
j=0 F(˜xj
i) \triangleright Sum over the prediction vectors of the m transformations
end for
v ←
1
n(m+1)v
ˆc ←arg maxc∈[1,c] vc
\triangleright Select the class c that has the highest value under measure M (Eq. 4)
return ˆc
# A.2 Detailed Algorithm for Latent Vector Selector
Algorithm 2 describes how we select the latent vector given the target label. We refer to the target label as the predicted pseudo label or ground truth label (if available). We first randomly sample n latent vectors and filter out the latent vectors that can generate recovered images that will be predicted as the target label by the target model. If there exist such latent vectors, we choose the latent vector which leads to the recovery with the highest confidence of the target label; otherwise, the algorithm returns a random latent vector sampled from Gaussian.
Algorithm 2 Latent Vector Selection
Input: xns: corrupted private data, F: target model, n: number of randomly sampled
latent vectors, y: predicted label or ground truth label
zbank ←sample n latent vectors from N(0, 1)
zbank y ←{z ∈zbank| arg maxc∈[1,C] Fc(G(z, xns)) = y}
\triangleright Select latent vectors that can generate recovered images that will be predicted to
have label y
if zbank y is not empty then
ˆz ←arg maxz∈zbank y Fy(G(z, xns))
\triangleright Select the latent vector that has the
highest confidence for label y
else
ˆz ∼N(0, 1)
end if
return ˆz
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5075/50758af9-b45f-4f8d-8a55-97c444d87110.png" style="width: 50%;"></div>
<div style="text-align: center;">Back propagation</div>
Fig. 6. Overview of the proposed SecretGen if ground truth label is available. The blue modules represent the proposed algorithms. The Target Model could allow either whitebox or blackbox access. The ground truth label is fed to the latent vector selector, which returns the initial latent vector for SecretGen optimization. In the whitebox setting, the ground truth label is also leveraged for identity loss.
# A.3 SecretGen with Known Ground Truth Labels
As we have presented in Section 3, SecretGen predicts a pseudo label for the target private image through a discrimination metric without requiring the knowledge of the ground truth label. Here, we consider the setting when the ground truth label is available, and we point out that the knowledge can be incorporated into our pipeline conveniently. See Fig. 6 for an overview of SecretGen if ground truth labels for target private images are known. Under this scenario, the pseudo label predictor is not required, and we substitute the input of the latent vector selector with the known ground truth label. Besides, the ground truth label is leveraged for computing identity loss during SecretGen optimization in the whitebox setting.
# B Additional Experimental Setup
Target Model Training. The target models are trained on Dtrain pri . For training ViT-B 16, we use the SGD optimizer with learning rate 3 × 10−2, batch size 64 for 10,000 steps. We use cosine annealing schedule with 500 warm-up steps. The other models are trained with the SGD optimizer with learning rate 10−2, batch size 64, momentum 0.9 and weight decay 10−4. Attack Procedures. We leverage WGAN-GP [13] and train our generation backbone with the Adam optimizer [17] with batch size 64, learning rate 4×10−3, β1 = 0.5, β2 = 0.999. We set λdiv in Eqn. (3) to 0.5 based on quantitative observations. For both our pseudo label predictor and latent vector selector, we set the number of randomly sampled latent vectors to 200. Although we demonstrate that our pipeline works better under larger n, the efficiency suffers. See Section 4.5 for concrete results. We set the patch size for cutouts to 16 × 16 and the resulting number of transformations m is 16. See Appendix C.2 for more discussion regarding the selection of m. At the SecretGen optimization stage, we set λid in Eqn. (8) to 100 and use the SGD optimizer to optimize the selected latent vector for 1,500 iterations with learning rate 0.02 and momentum 0.9.
# C Additional Results
In this section, we report additional results for ablation studies. We also show some additional qualitative results of SecretGen.
# C.1 Attack Performance on FaceScrub
Table 6 compares attack performance of SecretGen and baselines on FaceScrub. We use VGG16 as the target model. It can be seen that SecretGen still significantly outperforms PII and GMI. The conclusion aligns with that on CelebA.
<div style="text-align: center;">Table 6. Whitebox attack performance of SecretGen and baselines on FaceScrub. Target model: VGG16. The given prior information is corrupted images by center mask.</div>
Method
Ground Truth Label
Protocol 1
Protocol 2
PSNR
PII
✗
0.274
0.818
26.269
GMI
✓
0.398
0.972
26.028
SecretGen
✗
0.396
0.942
26.459
SecretGen
✓
0.453
0.973
26.516
# C.2 Ablation Studies on the Number of Transformations
We demonstrate that the number of transformations m in the discrimination metric (Eqn.(4)) affects the performance of our pseudo label predictor. We adjust
m by changing the patch size of our sequential cutout. Since our recovered image is 64 × 64, the resulting patch size is thus 64 √m × 64 √m. We fix n to 200 and plot label prediction accuracy on the recovered images w.r.t. m in Fig. 7. We observe that when m is small, the resulting label prediction accuracy suffers due to the aggressiveness of cutting out a large patch. However, efficiency suffers as m increases. See Table 7 for the average running time of pseudo label predictor for one target private image on a single NVIDIA RTX 2080 Ti GPU. Note that the relationship between m and label prediction accuracy is not consistent along different model architectures. Since the attacker has no idea about the optimal m for the target model, in practice we set m to 16 and we demonstrate that performance of SecretGen is promising in Section 4.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/56ac/56aca2cc-afe2-4d5a-9a77-9fd75f654bae.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 7. Label prediction accuracy of the pseudo label predictor w.r.t. different number of transformations on CelebA. For each private image, we sample n = 200 latent vectors, and then perform m transformations corresponding to each latent vector.</div>
Table 7. Average running time of pseudo label predictor for one target private image. For each private image, we sample n = 200 latent vectors and then perform m transformations corresponding to each latent vector. Target model: VGG16.
m
4
9
16
25
36
Running Time (s)
18.583
26.747
38.222
53.463
72.876
# C.3 Ablation Studies on Transformations
In this section, we present a series of quantitative results of the ablation studies on transformations. Performance without Transformations. In Section 4.5, we demonstrated that SecretGen performs better if transformations are incorporated in the discrimination metric for face.evoLVe. Here we present the results for other model architectures. We remove transformations from the original discrimination metric M (Eqn. (4)) and denote the resulting discrimination metric without transformations as M′ (Eqn. (10)). We evaluate end-to-end attack accuracy for M and M′ on the first 3,200 images from Dtrain pri under Protocol 1. The results are illustrated in Table 8, which demonstrate that incorporating transformations
improves attack accuracy for VGG16. For ResNet152, the results are similar. We also report PSNR scores for the recovered images for reference. Performance with Various Transformations. We further evaluate the performance of our pseudo label predictor incorporated with different kinds of transformations. We consider using more aggressive transformations including horizontal flipping, grayscale, and color jittering (brightness 1.5, contrast 1.5, saturation 1.5). We plot label prediction accuracy for the first 3,200 private images from Dtrain pri with different transformations w.r.t. the number of random latent vectors in Fig. 8. We also plot the label prediction accuracy without transformations for reference. We demonstrate that the performance of aggressive transformations is worse than the case when no transformations are incorporated, while sequential cutout which is used in our pipeline performs the best.
<div style="text-align: center;">Table 8. Attack accuracy of SecretGen with and without transformations on CelebA. Evaluated with the evaluation model under Protocol 1.</div>
Target Model
Setting
Center Mask
Face T Mask
Attack Acc PSNR Attack Acc PSNR
VGG16
w/o transformation
0.584
27.863
0.319
26.596
w/ transformation
0.599
27.874
0.328
26.598
ResNet152
w/o transformation
0.594
27.447
0.337
26.792
w/ transformation
0.592
27.442
0.338
26.788
face.evoLVe
w/o transformation
0.528
27.505
0.256
26.527
w/ transformation
0.550
27.522
0.273
26.527
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/dbc7/dbc7aec2-918b-4605-9485-6847febd9a22.png" style="width: 50%;"></div>
Fig. 8. Label prediction accuracy without transformation (denoted as “none” in le end) and with different kinds of transformations on CelebA. We plot label predictio accuracy w.r.t. the number of randomly sampled latent vectors.
<div style="text-align: center;">Fig. 8. Label prediction accuracy without transformation (denoted as “none” in legend) and with different kinds of transformations on CelebA. We plot label prediction accuracy w.r.t. the number of randomly sampled latent vectors.</div>
# C.4 Impact of Overfitting Levels
We evaluate target models of two architectures: VGG16 and ResNet152. We train the target model from scratch on Dtrain pri under the settings in Section 4.1 for 200 epochs and we save the model at epoch 50, 100, 200. Classification accuracy of overfitted models of the same architecture are similar, which are 0.746, 0.748, 0.748 (VGG16) and 0.675, 0.676, 0.675 (ResNet152), respectively. We substitute the target model with a public feature extractor from [5] when training the generation backbone to avoid training multiple generation backbones for each model. In Fig. 9 we plot label prediction accuracy w.r.t. different training epochs of the target model. The results indicate that target models of higher overfitted levels are more vulnerable to SecretGen pseudo label predictor. In the ideal case, the classification model learns a general mapping from face images to identity labels such that the non-sensitive information will be ignored. However, in reality, the correlation between the non-sensitive region and ground truth label grows stronger as the number of training epochs increases, as currently the overfitting problem has not been completely solved in machine learning.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bd09/bd091790-3675-4e04-8864-7e448685f832.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 9. Label prediction accuracy of pseudo label predictor against target models of different overfitting levels on CelebA. We plot label prediction accuracy w.r.t. the number of training epochs of the target model.</div>
# C.5 More Qualitative Results
We present some qualitative results of SecretGen in Fig. 10. We can see that when both ground truth label and whitebox access are available, SecretGen can produce recoveries that are closer to the target private image than GMI. In other settings, SecretGen also outperforms the baseline PII in reconstructing privacysensitive attributes of the target image. Besides, SecretGen generates visually plausible images under all settings.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c713/c7134ba9-1a03-4c7f-a448-a7278423a952.png" style="width: 50%;"></div>
Fig. 10. More qualitative results on CelebA. “bb”/“wb” indicates that the method requires blackbox/whitebox access to the model. “gt” indicates that the method requires ground truth labels. Images in the first column are ground truth private images. Images in the second column are prior information available to the adversary.
