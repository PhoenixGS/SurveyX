Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000. Digital Object Identifier 10.1109/ACCESS.2021.DOI
# Disease-oriented image embedding wit pseudo-scanner standardization for content-based image retrieval on 3D brain MRI 1
# Disease-oriented image embedding with pseudo-scanner standardization for content-based image retrieval on 3D
HAYATO ARAI1, YUTO ONGA1, KUMPEI IKUTA1, YUSUKE CHAYAMA1, HITOSHI IYATOMI1 (MEMBER, IEEE), AND KENICHI OISHI 2 for the Alzheimer’s Disease Neuroimaging Initiativ and the Parkinson’s Progression Markers Initiative. 1Department of Applied Informatics, Graduate School of Science and Engineering, Hosei University, Tokyo, 184-8584 Japan (e-mail: iyatomi@hosei.ac.jp) 2Department of Radiology and Radiological Science, Johns Hopkins University School of Medicine, Baltimore, MD 21205 USA (email:koishi2@jhmi.edu) Corresponding author: Hitoshi Iyatomi (e-mail: iyatomi@hosei.ac.jp). Data used in preparation of this article were obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf V]  14 Aug 20
Data used in preparation of this article were obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf V]  14
ABSTRACT To build a robust and practical content-based image retrieval (CBIR) system that is applicable to a clinical brain MRI database, we propose a new framework – Disease-oriented image embedding with pseudo-scanner standardization (DI-PSS) – that consists of two core techniques, data harmonization and a dimension reduction algorithm. Our DI-PSS uses skull stripping and CycleGAN-based image transformations that map to a standard brain followed by transformation into a brain image taken with a given reference scanner. Then, our 3D convolutioinal autoencoders (3D-CAE) with deep metric learning acquires a low-dimensional embedding that better reflects the characteristics of the disease. The effectiveness of our proposed framework was tested on the T1-weighted MRIs selected from the Alzheimer’s Disease Neuroimaging Initiative and the Parkinson’s Progression Markers Initiative. We confirmed that our PSS greatly reduced the variability of low-dimensional embeddings caused by different scanner and datasets. Compared with the baseline condition, our PSS reduced the variability in the distance from Alzheimer’s disease (AD) to clinically normal (CN) and Parkinson disease (PD) cases by 15.8–22.6% and 18.0–29.9%, respectively. These properties allow DI-PSS to generate lower dimensional representations that are more amenable to disease classification. In AD and CN classification experiments based on spectral clustering, PSS improved the average accuracy and macro-F1 by 6.2% and 10.7%, respectively. Given the potential of the DI-PSS for harmonizing images scanned by MRI scanners that were not used to scan the training data, we expect that the DI-PSS is suitable for application to a large number of legacy MRIs scanned in heterogeneous environments. arXiv:2108.06518v1  [cs
INDEX TERMS ADNI, CBIR, convolutional auto encoders, CycleGAN, Data harmonization, data standardization, metric learning, MRI, PPMI
# I. INTRODUCTION
I N the new era of Open Science [1], data sharing has become increasingly crucial for efficient and fair development of science and industry. Especially in the field of medical image science, various datasets have been released and used for the development of new methods and benchmarks. There have been attempts to create publicly open databases
VOLUME 4, 2016
consisting of medical images, demographic data, and clinical information, such as ADNI, AIBL, PPMI, 4RTN, PING, ABCD and UK BioBank. In the near future, clinical images acquired with medical indications will become available for research use.
Big data, consisting of large amounts of brain magnetic resonance (MR) images and corresponding medical records,
could provide new evidence for the diagnosis and treatment of various diseases. Clearly, search technology is essential for the practical and effective use of such big data. Currently, text-based searching is widely used for the retrieval of brain MR images. However, since this approach requires skills and experience during retrieval and data registration, there is a strong demand from the field to realize content-based image retrieval (CBIR) [2]. To build a CBIR system that is feasible for brain MR imaging (MRI) databases, obtaining an appropriate and robust low-dimensional representation of the original MR images that reflects the characteristics of the disease in focus is extremely important. Various methods have been proposed, including those based on classical feature description [3]– [5], anatomical phenotypes [6], and deep learning techniques [7]–[9]. The latter two techniques [8], [9] acquire similar low-dimensional representations for similar disease data by introducing the idea of distance metric learning [10] [11]. Their low-dimensional representations adequately capture disease characteristics rather than individual variations seen on gyrification patterns in the brain. However, the application of these methods to a heterogeneous database containing MRIs from various scanners and scan protocols is hampered by the scanner or protocol bias, which is not negligible. In brain MRI, such non-biological experimental variations (i.e., magnetic field strength, scanner manufacturer, reconstruction method) resulting from differences in scanner characteristics and protocols can affect the images in various ways and have a significant impact on the subsequent process [9], [12]–[16]. Wachinger et al. [16] analyzed 35,320 MR images from 17 open datasets and performed the ’Name That Dataset’ test, that is guessing which dataset it is based on the images alone. They reported a prediction accuracy of 71.5% based only on volume and thickness information from 70% of the training data. This is evidence that there are clear features left among datasets. Removing those variabilities is essential in multi-site and long-term studies and for building a robust CBIR system. There has been an increase in recent research on data harmonization, i.e., eliminating or reducing variation that is not intrinsically related to the brain’s biological features. Perhaps the most straightforward image harmonization approach is to reduce the variations in the intensity profile [17], [18]. In the methods in both [17] and [18], correction of the luminance distribution for each sub-region reduces the variability of the underlying statistics between images, whereas histogram equalization reduces the variability of neuroradiological features. However, these methods are limited to approximating rudimentary statistics that can be calculated from images, and they are based on the assumption that the intensity histogram is similar among images. This assumption is invalid when images that contain pathological findings that affect intensity profile are included. While some improvement in unintended image variability can be expected, the effect on practical tests that utilize data from multiple sites is unknown.
In the field of genomics, Johnson et al. [19] proposed an empirical Bayes-based correction method to reduce batch effects, which are non-biological differences originating from each batch of micro-array experiments obtained from multiple tests. This effective statistical bias reduction method is now called ComBat, and it has recently been published as a tool for MRI harmonization [20]. This tool has been applied to several studies [14], [16], [21], [22]. The ComBatbased methods standardize each cortical region based on an additive and use multiplicative linear transform to compensate for variability. Some limitations of these models have been pointed out, such as the following: (i) they might be insufficient for complex multi-site and area-level mapping, (ii) the assumption of certain prior probabilities (Gaussian or inverse gamma) is not always appropriate, and (iii) they are susceptible to outliers [23]. Recently, advancements in machine learning techniques [23]–[26] have provided practical solutions for MR image harmonization. DeepHarmony [24] uses a fully convolutional U-net to perform the harmonization of scanners. The researchers used an MRI dataset of multiple sclerosis patients in a longitudinal clinical setting to evaluate the effect of protocol changes on atrophy measures in a clinical study. As a result, DeepHarmony confirmed a significant improvement in the consistency of volume quantification across scanning protocols. This study was practical in that it aimed to directly standardize MR images using deep learning to achieve longterm, multi-institutional quantitative diagnosis. However, this model requires ”traveling head” (participants are scanned using multiple MRI scanners) to train the model. Zhao et al. [23] attempted to standardize a group of MR images of infants taken at multiple sites into a reference group using CycleGAN [27], which has a U-net structure in the generator. The experiment validated the evaluation of cortical thickness with several indices (i.e., ROI (region-of-interest)-base, distribution of low-dimensional representations). They argued that the retention of the patient’s age group was superior to ComBat in evaluating group difference. Moyer et al. [25] proposed a sophisticated training technique to reconstruct bias-free MR images by acquiring a lowdimensional representation independent of the scanner and condition. Their method is an hourglass-type unsupervised learning model based on variational autoencoders (VAE) with an encoder–decoder configuration. The input x and output x′ are the same MR images, and their low-dimensional representation is z (i.e., x →z →x′). The model is trained with the constraint that z and site- and scanner-specific information s are orthogonal (actually relaxed), such that the s in z is eliminated. They demonstrated the advantages of their method on diffusion MRI, but their technological framework is applicable to other modalities. Dinsdale et al. [26] also proposed a data harmonization method based on the idea of domain adaptation [28]. Their model uses adversarial learning, where the feature extractor consisting of convolutional neural networks (CNN) following the input is branched into a fully connected net for the
original task (e.g., segmentation and classification) and other fully connected nets for domain discriminators (e.g., scanner type or site prediction) to make the domain unknown while improving the accuracy of the original task. They have confirmed its effectiveness in age estimation and segmentation tasks. The methods developed by Moyer et al. and Dinsdale et al. aim to generate a low-dimensional representation with ”no site information”, and they are highly practical and generalizable techniques for data harmonization. Nevertheless, for CBIR, a method that is applicable for a large number of legacy images is necessary. Here, it is not realistic to collect images from each site and train the model to harmonize them. Practically, a method that can convert heterogeneous images in terms of variations in scanners and scan parameters into images scanned by a given pseudo-”standard” environment by applying a learned model is highly desired. In this paper, we propose a novel framework called disease-oriented image embedding with pseudo-scanner standardization (DI-PSS) to obtain a low-dimensional representation of MR images for practical CBIR implementation. The PSS, the key element of the proposal, corrects the bias caused by different scanning environments and converts the images so that it is as if the same equipment had scanned them. Our experiments on ADNI and PPMI datasets consisting of MR images captured by three manufacturers’ MRI systems confirmed that the proposed DI-PSS plays an important role in realizing CBIR. The highlights of this paper’s contribution are as follows: • To the best of the authors’ knowledge, this is the first study of the acquisition and quantitative evaluation of an effective low-dimensional representation of brain MR images for CBIR, including scanner harmonization. • Our DI-PSS framework reduces undesirable differences caused by differences in scanning environments (e.g., scanner, protocol, dataset) by converting MR images to images taken on a predefined pseudo-standard scanner, and a deep network using a metric learning acquires a low-dimensional representation that better represents the characteristics of the disease. • DI-PSS provides appropriately good low-dimensional representations for images from other vendors’ scanners, diseases, and datasets that are not used for learning image harmonization. This is an important feature for the practical and robust CBIR, which applies to a large amount of legacy MRIs scanned at heterogeneous environments.
# II. CLARIFICATION OF THE ISSUES ADDRESSED IN
# II. CLARIFICATION OF THE ISSUES ADDRESSED IN THIS PAPER
We begin by presenting the issues to be solved in this paper. As mentioned above, to realize CBIR for brain MRI, Onga et al. proposed a new technique called disease-oriented data concentration with metric learning (DDCML), which acquires low-dimensional representations of 3D brain MR
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1ec8/1ec82c87-dfa3-4736-a270-3065e97716b7.png" style="width: 50%;"></div>
<div style="text-align: center;">FIGURE 1: Plots of low-dimensional representations of 3D MRI obtained from different datasets. The impact of different scanners (CN⇔Control; they are medically equivalent) is greater than the impact of the disease (AD⇔CN).</div>
images that are focused on disease features rather than the features of the subject’s brain shape [9]. DDCML is composed of 3D convolutional autoencoders (3D-CAE) effectively combined with deep metric learning. Thanks to its metric learning, DDCML could acquire reasonable lowdimensional representations for unlearned diseases according to their severity, demonstrating the feasibility of CBIR for brain MR images. However, we found that such representations are highly sensitive to differences in datasets (i.e., differences in imaging environments, scanners, protocols, etc.), which is a serious challenge for CBIR. Figure 1 shows the low-dimensional distribution obtained by DDCML and visualized by t-SNE [29]. Here, DDCML was trained on Alzheimer’s disease (AD) and healthy cases (clinically normal; CN) in the ADNI2 dataset and evaluated ADNI2 cases not used for training and healthy control (Control – equivalent to CN) and Parkinson’s disease (PD) cases in the untrained PPMI dataset. From the perspective of CBIR, it is desirable to obtain similar low-dimensional representations for CN and Control. However, it can be confirmed that the obtained low-dimensional representations are more affected by the differences in the environment (dataset) than by the disease. As mentioned above, differences in imaging environments, including scanners, are a major problem in multi-center and time series analysis, and inconsistent lowdimensional representations because of such differences in datasets are a fatal problem in CBIR implementation. The purpose of this paper is reducing these differences and to obtain a low-dimensional representation that better captures the characteristics of the disease and is suitable for appropriate CBIR.
# B. OUR DATA HARMONIZATION STRATEGY FOR
# B. OUR DATA HARMONIZATION STRATEGY FOR REALIZING CBIR
In studies dealing with multi-site and long-term data, it is undoubtedly important to reduce non-biological bias origi-
nating from differences among sites and datasets. Since the methods of Moyer et. al. [25] and Dinsdale et al. [26] are theoretical and straightforward learning method that utilizes images of the target site to achieve data harmonization, their robustness to unexpected input (i.e. from another site or dataset) is questionable. Therefore, in principle, the images of all target sites (scanners, protocols) need to be learned in advance. Since CBIR requires more consideration of the use of images taken in the past, the number of environments that need to be addressed can be larger than for general data harmonization. It will be more difficult to implement a harmonization method that learns all the data of multiple environments in advance. Therefore, in contrast to their approaches, we aim to achieve data harmonization by converting images taken in each environment into images that can be regarded as having been taken in one predetermined ”standard” environment (e.g., the scanner currently used primarily at each site). However, in addition to the problems described above, it is practically impossible to build an image converter for each environment. With this background, we have developed a framework that combines CycleGAN, which realizes robust image transformation, with deep metric learning to achieve a certain degree of harmonization even for images in untrained environments. In this paper, we validate the feasibility of our framework, which converts MR images captured in various environments into pseudo standard environment images using only one type of image converter.
# III. DISEASE-ORIENTED IMAGE EMBEDDING WITH PSEUDO-SCANNER STANDARDIZATION (DI-PSS)
The aim of this study is to obtain a low-dimensional embedding of brain MRI that is independent of the MRI scanner and individual characteristics but dependent on the pathological features of the brain, to realize a practical CBIR system for brain MRI. To accomplish this, we propose a DI-PSS framework, which is composed of the three following components: (1) pre-process, (2) PSS, and (3) embedding acquisition.
# A. THE PRE-PROCESSING COMPONENT (SKULL STRIPPING WITH GEOMETRY AND INTENSITY
The pre-processing component performs the necessary preprocessing for future image scanner standardization processing and low-dimensional embedding acquisition processing. Specifically, for all 3D brain MR image data, skull stripping was performed using a multi-atlas label-fusion algorithm implemented in the MRICloud [30]. The skull-stripped images were linearly aligned to the JHU-MNI space using a 12-parameters affine transformation function implemented in the MRICloud, resulting in aligned brain images. This feature makes a significant contribution to the realization of the proposed PSS in the next stage. It is important to note here that since brain volume information is the feature that contributes most to the prediction of the dataset [16], the alignment to a standard brain with this skull stripping
technique should also contribute to the harmonization of the data. In addition, because the intensity and contrast of brain MR images are arbitrarily determined, there is a large inter-image variation. In brain MR image processing using machine learning, the variation in the average intensity confounds the results. Therefore, we standardized the intensity so that the average intensity value of each case was within mean µ = 18 and margin ϵ = 1.0 by performing an iterative gamma correction process, as in previous studies [31] [9].
1) The concept of PSS The proposed PSS is an image conversion scheme that converts a given raw MR image into a synthesized image that looks like an MR image scanned by a standard scanner and a protocol. Since there are numerous combinations of scanners and scan parameters, building scanner- and parameterspecific converters is not practical. Therefore, in our PSS scheme, we only construct a 1:1 image conversion model (i.e., PSS network) that converts images from a particular scanner Y to a standard scanner X. That is, a particular PSS network is used to convert images captured by other scanners (Z1, Z2, · · · ) as well. This strategy is in anticipation of the generalizability of the PSS network, backed by advanced deep learning techniques. In this paper, we evaluate the robustness of our image transformations provided by PSS on MR images taken by other vendors’ scanners and on images in different datasets. Figure 2 gives an overview of our PSS network that realizes the PSS. The PSS network makes effective use of CycleGAN [27], which has achieved excellent results in 1:1 image transformation. Here, training of CycleGAN generally requires a lot of training data, especially in the case of 3D data, because the degree of freedom of the model parameters is large. However, it is difficult to collect such a large amount of supervised labeled 3D MRI data to keep up with the increase. Since the position of any given slice is almost the same in our setting thanks to MRICloud in the skull stripping process, a 3D image can be treated as a set of 2D images containing position information. With these advantages, our PSS suppressed the problems an overwhelmingly insufficient amount of training data and the high degree of freedom of the transformation network. In sum, arbitrary slices are cut out from the input 3D image and converted to slices corresponding to the same position in the 3D image as the target domain using the PSS network based on common (2D) CycleGAN. Note that the PSS process is performed using the trained generator GX.
2) Implementation of the PSS network The structure of the PSS network that realizes the proposed PSS is explained according to the CycleGAN syntax, with images captured by a standard scanner as domain X and images captured by a certain different scanner as domain Y . Generator GY transforms (generates) an image y′ = GY (x) with the features of domain Y from an image x of the original
2) Implementation of the PSS network The structure of the PSS network that realizes the proposed PSS is explained according to the CycleGAN syntax, with images captured by a standard scanner as domain X and images captured by a certain different scanner as domain Y . Generator GY transforms (generates) an image y′ = GY (x) with the features of domain Y from an image x of the original
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1482/14824525-c207-42c0-9a6c-5c9953304812.png" style="width: 50%;"></div>
<div style="text-align: center;">FIGURE 2: Overview of pseudo-scanner standardization (PSS) network. Our PSS network is based on CycleGAN, and PSS is performed with trained generator GX.</div>
domain X. Discriminator DY determines the authenticity of the real image y belonging to domain Y or the generated y′ = GY (x). Similarly, the conversion from domain Y to domain X is performed by generator GY , and discriminator DX judges the authenticity of the image. The goal of this model is to learn maps of two domains X and Y given as training data. Note here again that we use the trained module GX (maps Y to X) as an image converter. The training of the model proceeds by repeating the transformation of the training data sample xi ∈X and the training data sample yj ∈Y . The overall objective function of the PSS network, LP SS to be minimized, consists of the three following loss components: adversarial loss (LGAN), cycle consistency loss (Leye), and identity mapping loss (Lidentity). This is expressed as follows:
(1)
The adversarial loss (LGAN) is defined based on the competition between the generator, which tries to produce the desired other domain image, and the discriminator, which sees through the fake generated image; this minimization implies a refinement of both. From the point of view of image transformation, the minimization of this loss means that the probability distribution generated by the generator is closer to the probability distribution of the counterpart domain, which means that a higher quality image can be obtained. This loss is defined in both directions, X →Y and Y →X, and these are expressed in order as follows:
(2)
LGAN(GX, DX) =Ex∼pdata(x)[(DX(x) −1)2] +Ey∼pdata(y)[(DX(GX(y))2].
(3)
The cycle consistency loss (Leye) is a constraint to guarantee that mutual transformation is possible by cycling two generators:
(4)
Finally, the identity mapping loss (Lidentity) is a constraint to maintain the original image features without performing any transformation when the image of the destination domain is input:
Lidentity(GX, GY ) =Ex∼pdata(x)||GX(x) −x||1 +Ey∼pdata(y)||GY (y) −y||1
(5)
It has been confirmed that the introduction of this constraint can suppress the learning of features that are not important in either domain, such as unneeded tints. Here, λ1 and λ2 are hyper-parameters and we set λ1 = 10.0 and λ2 = 0.5 as in the original setting.
3) The Embedding acquisition component In the embedding acquisition component, the lowdimensional embedding of 3D brain MRI images is obtained by our embedding network after the PSS process. Our embedding network is a 3D-CAE model consisting of encoders and decoders with distance metric learning, referring to Onga et al.’s DDCML [9]. Distance metric learning is a learning technique that reduces the Euclidean distance between feature representations of the same label and increases the distance between feature representations of different labels. Thanks to the introduction of metric learning, 3D-CAE has been found to yield embedding that is more focused on disease features. According to Hoffer’s criteria [11], the distance distribution in the low-dimensional embedding space for input x for
class i (i ∈1, · · · c; where c is the number of types of disease labels in the dataset) is calculated by
(6)
� Here, xi (i ∈1, · · · , c) is randomly sampled data from each class i, and f denotes the operation of the encoder (i.e., encoder part of the 3D-CAE in our implementation). This probability can be thought of as the probability that the data x belong to each class i. The loss function Ldist is calculated by the cross-entropy between the c-dimensional vector P described above and the c-dimensional one-hot vector I(x) with bits of the class to which x belongs as
(7)
Here, H(I(x), P (x; x1, · · · , xc)) takes a small value when the probability that the element firing in I(x) belongs to the class it represents is high, whereas it takes a large value when the probability is low. Thus, Ldist aims at the distribution of the sampled data at locations closer to the same class and farther from the different classes on the low-dimensional feature space. Finally, the objective function LCAE of our low-dimensional embedding acquisition network consisting of 3D-CAE and metric learning is finally expressed by the following equation:
Here, LRMSE is the pixel-wise root mean square error normalized by image size in CAE image reconstruction. Furthermore, α is a hyper-parameter set to 1/3 based on the results of preliminary experiments.
In CBIR, cases of the same disease should be able to acquire similar low-dimensional representations, regardless of the individual, scanner, or protocol. We investigated the effectiveness of the proposed DI-PSS by quantitatively evaluating how PSS changes the distribution of embeddings within and between data groups (i.e., combination of scanner type and disease). In addition, we compared the clustering performance of the obtained embeddings against diseases with and without PSS.
# A. DATASET
In this experiment, we used the ADNI2 and PPMI datasets, in which the vendor information of the scanners (Siemens [SI], GE Medical Systems [GE], Philips Medical Systems [PH]) was recorded along with the disease information. Statistics of those datasets used in the experiment are shown in Table 1. We used Alzheimer’s disease (ADNI-AD or AD) and clinically normal cases (ADNI-CN) from ADNI2 dataset with vendor information. From the PPMI dataset, we used two types of labeled images, Parkinson’s disease (PD) and Control. We did not utilize the scanner information for this
<div style="text-align: center;">TABLE 1: Dataset used in our study</div>
dataset
vendor
label
#used
#patients
#total
ADNI
CN
Siemens
CN_SI
92
103
439
GE
CN_GE
93
101
494
Philips
CN_PH
27
27
119
AD
Siemens
AD_SI
80
84
254
GE
AD_GE
80
92
302
Philips
AD_PH
20
24
73
PPMI
n/a
Control
75
75
114
PD
149
149
338
ADNI-CN and Control can be considered medically equivalent.
ADNI-CN and Control can be considered medically equivalent. There are no PD-related anatomical features observable on T1-weighted MRI.
dataset in evaluating the versatility of the proposed method. Note that ADNI-CN and Control can be considered medically equivalent. Furthermore, PD is known to show little or no difference in MRI from healthy cases [32] [33]. The ADNI and PPMI are longitudinal studies that include multiple time points, and the datasets contain multiple scans for each participant. To avoid duplication, one MRI was randomly selected from each participant. The MRICloud (https: //mricloud.org/) was used to skull strip the T1-weighted MRIs and affine transform to the JHU-MNI atlas [34]. A neurologist with more than 20 years of experience in brain MRI research performed the quality control of the MRIs and removed MRIs that the MRICloud did not appropriately pre-process. Due to the neural network model used in the experiments, the skull-stripped and affine-transformed brain MR images were converted to 160×160×192 pixels after cropping the background area. Training and evaluation of the PSS network and embedding network were performed using five-fold cross validation. In the evaluation experiments described below„ the evaluation data of each fold is not included in the training data for either the PSS network or the embedding network. Note that even skilled and experienced neuroradiologists cannot separate PD from CN or Control by visual inspection of the T1-weighted images. Therefore, we did not expect these two conditions to be separable by unsupervised clustering methods even after applying the DIPSS.
B. DETAIL OF THE PSS NETWORK AND ITS TRAINING Figures 3a and 3b show the architecture of the generator (GX, GY ) and the discriminator (DX, DY ), respectively of the PSS network. They are basically the same as the original CycleGAN for 2D images. Since PSS is to reduce the bias caused by variations in scanners and scan parameters, the disease-related anatomical variations should be minimized in the training images. Therefore, we used only ADNI-CN cases, in which disease features do not appear in the brain structure, to train the PSS network. In this experiment, we chose the Siemens scanner as the standard scanner because it has largest market share, and we chose the GE scanner as the specific vendor of image conversion source. In other words, our PSS network is designed to convert CN images taken by GE scanners from the ADNI2-dataset (CN_GE) to
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7e89/7e89e61a-a648-4199-b48a-3b0542a2c609.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) Generator</div>
<div style="text-align: center;">FIGURE 3: Architecture of (a) Generators (GX, GY ) and (b) Discriminators (DX, DY ) in the PSS network. (a) kernel size (f×f), stride size (s), padding size (p), × # of kernel + instance norm + ReLU* (b) convA : kernel size (f×f), stride size (s), padding size (p), × # of kernel + LeakyReLU convB : kernel size (f×f), stride size (s), padding size (p), × # of kernel + instanceNorm + LeakyReLU convC : kernel size (f×f), stride size (s), padding size (p), × # of kernel FC : fully connected layer (400→1)</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/641b/641bfa76-4804-4597-92b4-f09508b1004f.png" style="width: 50%;"></div>
<div style="text-align: center;">FIGURE 4: Architecture of embedding network. conv : kernel 3×3, stride size=1, padding size=1, × (# of kernel) + ReLU deconv : kernel 3×3, stride size=1, padding size=1, × (# of kernel) + ReLU average pooling, up-sampling (bi-linear interpolation): 2×2×2:1.</div>
synthetic images similar to those scanned by the Siemens scanners (CN_SI). We evaluated the applicability of the PSS to the diseased brain MRIs (AD and PD), as well as the generalizability to the non-GE scanners (see Section IV.D). In PSS network, we used coronal images for the training. The number of training images of each fold in the PSS network is (93+92)×4/5 (5-fold CV)×192 (slices).
# C. DETAIL OF THE EMBEDDING NETWORK AND ITS TRAINING
Figure 4 shows the architecture of our 3D-CAE-based embedding network. Our embedding network embeds each 3D brain MR image into 150-dimensional vectors. The size of the MRIs handled by the embedding network is halved at each side, as in DDCML [9], to improve the learning efficiency. Note that the compression ratio of our embedding network is (80×80×96):150 = 4,096:1. The embedding network was trained and evaluated using ADNI2 and PPMI datasets with the five-fold cross-validation strategy. As mentioned above, PD and CN cannot even be diagnosed from images by skilled neuroradiologists, so for training 3D-CAE to obtain low-dimensional representations, two classes of metric learning are used so that the representations of AD and (CN + Control) are separated. The lowdimensional representations of brain MR images are acquired by five-fold cross validation of 3D-CAE. In addition to AD, CN, and Control in each test fold, the low-dimensional
<div style="text-align: center;">(b) Discriminator</div>
representation of PD, which was not included in the training, is analyzed to quantitatively verify the effectiveness of the proposed DI-PSS evaluation. D. EVALUATION OF THE PSS To evaluate the effectiveness of the proposed DI-PSS framework, we evaluate the three following elements: 1) Changes in MR images 2) Distribution of the embedding. 3) Clustering performance of the embedding. In (1), we assess how the images are changed by our scanner standardization. We quantitatively evaluate the difference between the original (raw) image and the synthetic image with peak signal-to-noise ratio (PSNR), root mean squared error (RMSE), and structured similarity (SSIM). To ensure that the evaluation is not affected by differences in brain size, these evaluations were performed on brain regions only. Although MRICloud, which is used in skull stripping in this experiment, standardizes the brain size to the standard brain size, reducing the differences in brain size between cases, this method was adopted for a more rigorous evaluation. In (2), we quantitatively examine the effect of PSS by analyzing the distribution of the obtained low-dimensional representations. Specifically, for each category (e.g., CN_SI, AD_GE) we investigate the following: (i) variation (i.e., standard deviation) of the embedding and (ii) the mean and standard deviation of the distance from each embedding to the
representation of PD, which was not included in the training, is analyzed to quantitatively verify the effectiveness of the proposed DI-PSS evaluation.
# D. EVALUATION OF THE PSS
In (1), we assess how the images are changed by our scanner standardization. We quantitatively evaluate the difference between the original (raw) image and the synthetic image with peak signal-to-noise ratio (PSNR), root mean squared error (RMSE), and structured similarity (SSIM). To ensure that the evaluation is not affected by differences in brain size, these evaluations were performed on brain regions only. Although MRICloud, which is used in skull stripping in this experiment, standardizes the brain size to the standard brain size, reducing the differences in brain size between cases, this method was adopted for a more rigorous evaluation. In (2), we quantitatively examine the effect of PSS by analyzing the distribution of the obtained low-dimensional representations. Specifically, for each category (e.g., CN_SI, AD_GE) we investigate the following: (i) variation (i.e., standard deviation) of the embedding and (ii) the mean and standard deviation of the distance from each embedding to the
centroid of a different category, where the distance between the centroids of ADNI-CN_Siemens (CN_SI) and ADNIAD_Siemens (AD_SI) are normalized to 1. In addition, we visualize those distributions in 2D space using t-SNE [29] as supplemental results for intuitive understanding. In (3), we evaluate the separability of the resulting embeddings. In this study, we performed spectral clustering [35] to assess its potential quality for CBIR. In the spectral clustering, we used a normalized graph Laplacian based on 10-nearest neighbor graphs with a general Gaussian-type similarity measure. We set the number of clusters to be two (AD vs. CN + Control + PD), which is the number of disease categories to be classified. Here, the consistency of the distance between the embedded data because of the difference in folds is solved by standardizing the distance between CN_SI and AD_SI per fold to be 1, as mentioned above. The clustering performance was evaluated using two methodologies. The first was evaluation with six commonly used criteria (i.e., silhouette score, homogeneity, completeness, V-measure, adjusted Rand-index [ARI], and adjusted mutual information [AMI]) implemented on the scikit-learn machine learning library (https://scikit-learn.org/). The other is a diagnostic capability based on clustering results. Here, as with other clustering evaluations in the literature, we swap the columns so that each fold results in the optimal clustering result and then sum them.
# V. RESULTS
A. CHANGES IN MR IMAGES BY PSS Figure 5 shows an example of each MR image converted to an image taken on a pseudo-standard (= Siemens) scanner with PSS and the difference visualized. Table 2 summarizes the statistics of the degree of change in the images in the brain regions. Here, the background region was excluded from the calculation to eliminate the effect of differences in brain size. For the ADNI dataset, the differences obtained by the PSS image transformation were not significant between CN, AD, and scanner vendors, although the Philips scanners showed less variation on average. For the PPMI dataset that was not used for training, the change in the image because of PSS is clearly larger compared with ADNI (approx. × 1.5 in RMSE). In all categories, the amount of change because of PSS varied from case to case, but the PSS treatment did not cause any visually unnatural changes in the images. Figure 6 shows the cumulative intensity changes of images by PSS in each category. This time, the background areas other than the brain are also included in the evaluation. The number of pixels where the intensity has not changed because of PSS exceeds 80% for all categories, indicating that no undesired intensity changes have occurred for the background (as also seen in Figures. 5 and 6). There is no significant difference in the distribution of intensity change by vendor, and the PPMI dataset has a larger amount of intensity change overall.
<div style="text-align: center;">TABLE 2: Summary of image changes by PSS</div>
dataset
label
PSNR (db)
RMSE
SSIM
ADNI
CN_SI
31.52 ± 2.85
7.17 ± 2.57
0.9743 ± 0.0048
CN_GE
31.67 ± 2.45
6.94 ± 2.13
0.9748 ± 0.0041
CN_PH
32.18 ± 2.65
6.58 ± 2.02
0.9747 ± 0.0038
AD_SI
31.64 ± 3.04
7.13 ± 2.77
0.9746 ± 0.0043
AD_GE
31.65 ± 2.52
6.98 ± 2.32
0.9750 ± 0.0044
AD_PH
32.33 ± 2.13
6.36 ± 1.63
0.9751 ± 0.0031
PPMI
Control
30.16 ± 5.47
9.81 ± 8.32
0.9596 ± 0.0346
PD
29.40 ± 5.94
11.60 ± 10.89
0.9539 ± 0.0473
<div style="text-align: center;">TABLE 3: Variation (SD) of the embedding in category †</div>
 †
dataset
label
#data
baseline
with PSS
−SD (%)
ADNI
CN_SI
92
0.697
0.648
7.12
CN_GE
93
0.784
0.716
8.66
CN_PH
27
0.622
0.619
0.48
ADNI-CN
212
0.753
0.701
6.91
AD_SI
80
0.863
0.783
9.24
AD_GE
80
0.849
0.806
5.09
AD_PH
20
0.771
0.706
8.46
ADNI-AD
180
0.876
0.823
6.09
PPMI
Control
75
0.607
0.554
8.74
PD
149
0.603
0.515
14.71
both
CN
92
0.759
0.704
7.20
all
616
0.755
0.693
8.27
:Distance between CN_SI and AD_SI were normalized to 1.
†:Distance between CN_SI and AD_SI were normalized to 1.
# B. DISTRIBUTION OF LOW-DIMENSIONAL EMBEDDED DATA
Table 3 shows the variation (standard deviation; SD) of the 150-dimensional embedded representation in each category. Again, it should be noted here that CN_SI and AD_SI were normalized to 1. The average reduction in SD for all data by PSS was 8.27%. Tables 4 shows the statistics of distances from each embedding to the centroid of a different category. This shows the distribution of the data, considering the direction of variation, which is more practical for CBIR application. With PSS, the average distance between centroids across categories is almost unchanged, but the variability is greatly reduced for all categories.
2) Visualization of the distribution of the embedding Figures 7a and 7b show scatter plots of the embedding of test data with and without PSS, respectively in an arbitrary fold by t-SNE. Specifically, this is a scatter plot of the AD, CN, and Control test cases (data excluded from the training in the five-fold cross-validation) along with the untrained PD cases on the model. Here, PD has been randomly reduced to 1/5 for better visualization. Without PSS (baseline; 3D CAE + metric learning), AD and CN are properly separated, but the distribution of Control + PD (i.e., the difference in datasets) is separated from that of CN to a discernible degree (left). It can be confirmed that by performing PSS, the distribution of Control + PD becomes closer to that of CN, and the separation between AD and other categories becomes better (right).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/afcd/afcd9bf9-db05-4d54-a3ff-017934448309.png" style="width: 50%;"></div>
<div style="text-align: center;">FIGURE 5: Example of image change by PSS (in coronal plane). In each category, from left to right, the original image, the PSS processed image, and the difference between them.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ef6a/ef6aac5b-954c-4a77-bb37-af8a88b9b80c.png" style="width: 50%;"></div>
<div style="text-align: center;">(a) overall view</div>
<div style="text-align: center;">FIGURE 6: Cumulative intensity changes of MR images by PSS.</div>
C. CLUSTERING PERFORMANCE OF THE EMBEDDING In this section, we compare the separation ability of the obtained low-dimensional embedding of MR images with and without PSS (baseline). Tables 5 summarizes the clustering performance evaluated with six commonly used criteria. These are the silhouette score (silh), homogeneity score (homo), completeness score (comp), V-measure (harmonic mean of homogeneity and completeness; V) , ARI, and AMI implemented on the scikitlearn library. In each category, 1 is the best score and 0 is a score based on random clustering. It can be confirmed that
<div style="text-align: center;">(b) enlarged view</div>
PSS improved the clustering ability in all evaluation items. Table 6 is a summary of the clustering performance evaluated with the diagnostic ability. Table 6 (a) is a confusion matrix. Here, the numbers of CN, Control and AD cases are the sum of each fold in the cross-validation. In each fold, we tested all PD cases (not included in the training), and the number was divided by five and rounded to the nearest whole number. Tables 6b and 6c summarize the diagnostic performance calculated from Table 6 (a) without and with PD cases, respectively. It can be confirmed that PSS enhances the separation of AD and other categories (i.e., CN, Control and
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/796b/796b1c74-1c38-4346-a483-58c7e9308df3.png" style="width: 50%;"></div>
<div style="text-align: center;">FIGURE 7: Distribution of embedding visualized with t-SNE [29]: (left) baseline (3D-CAE+metric learning), (right) baseline with PSS.</div>
TABLE 4: Mean and variability of embedding across categories of data †
 †
baseline
with PSS
-SD (%)
from
to
mean
SD
mean
SD
ADNI-CN
AD
0.879
0.669
0.890
0.541
19.1
AD
ADNI-CN
0.875
0.729
16.6
Control
AD
1.354
0.745
1.329
0.537
28.0
AD
Control
0.907
0.702
22.6
PD
Control
0.256
0.469
0.297
0.312
33.5
Control
PD
0.414
0.368
11.2
ADNI-CN
PD
0.364
0.609
0.362
0.474
22.2
PD
ADNI-CN
0.373
0.255
31.4
AD
PD
1.164
0.939
1.091
0.770
18.0
PD
AD
0.620
0.434
29.9
CN
AD
0.996
0.753
0.997
0.583
22.6
AD
CN
0.917
0.773
15.8
CN
PD
0.249
0.593
0.269
0.466
21.4
PD
CN
0.349
0.264
24.2
†:Distance between CN_SI and AD_SI were normalized to 1.
†:Distance between CN_SI and AD_SI were normalized to 1.
TABLE 5: Clustering performance evaluated with common criteria †
TABLE 5: Clustering performance evaluated with common criteria †
TABLE 5: Clustering performance evaluated with common
 †
silh
homo
comp
V
ARI
AMI
baseline
0.236
0.220
0.301
0.250
0.251
0.241
+PSS
0.246
0.301
0.351
0.324
0.387
0.317
†: Score 1 is the best in each category. 0 is the score for random clustering.
†: Score 1 is the best in each category. 0 is the score for random clustering.
PD) in the low-dimensional representation. PSS improved the diagnostic performance by about 6.2% (from 73.7 to 79.9%) for micro-accuracy and about 10.7% (from 63.8 to 74.5%) for macro-F1. The specificity for PD was also improved by 6.1% (from 69.7% to 75.8%).
# VI. DISCUSSION
# A. CHANGES ON MR IMAGES BY PSS
A. CHANGES ON MR IMAGES BY PSS Our PSS network transforms healthy cases taken with GE scanners to those taken with Siemens scanners. As can be seen from Figure 6 and Table 2, the amount of change in the images because of PSS was almost the same for both AD and
CN images in the ADNI dataset, including the Philips case. The amount of conversion of the image for the PPMI dataset was larger than that for the ADNI dataset. This is thought to be due to the process of absorbing the differences in the datasets that exist in the image but are invisible to the eye. However, in all cases, the converted images have a natural appearance without destroying the brain structure. This can be objectively confirmed in SSIM, which evaluates the structural similarity on the image, maintains a high value. As discussed in detail below, PSS can reduce disease-specific variation in the resulting low-dimensional embedding, absorb differences among datasets and scanner vendors, and improve the separability of diseases. Given these factors, we can conclude that this PSS transformation was done properly. B. CONTRIBUTIONS OF DI-PSS FOR CBIR This section discusses the effects of our DI-PSS framework from the perspective of CBIR implementation.
B. CONTRIBUTIONS OF DI-PSS FOR CBIR This section discusses the effects of our DI-PSS framework from the perspective of CBIR implementation.
This section discusses the effects of our DI-PSS framework from the perspective of CBIR implementation.
1) Distribution of embedding
Based on the results in Tables 3 and 4, we first discuss the effectiveness of the proposed DI-PSS. From Table 3, PSS reduces the inter-cluster variability for all data categories. In particular, the SD of ADNI-CN and ADNI-AD, which are taken by scanners from three different companies in the same dataset, are reduced by 6.9% and 6.1%, respectively. This indicates that the PSS reduces the difference caused by different scanners. In addition, the SD of ALL_CN, which is a combination of ADNI-CN and Control from a different PPMI dataset, is also reduced by 7.2%, which clearly shows that the proposed PSS can absorb differences in datasets. This benefit can also be seen in Figure 7. The reduction of PD variability by PSS is more pronounced (−14.7%) than the others, and it is ultimately the category with the lowest variability. This is mentioned later in this section. From Table 4, PSS also succeeds in reducing the variability from each piece of data to all the different
<div style="text-align: center;">TABLE 6: Evaluation of clustering ability by diagnostic ability (a) Confusion matrix</div>
baseline
with PSS
CN+Control (+PD)
AD
CN+Control (+PD)
AD
CN+Control
284
3
274
13
(+PD)
(+104)
(+45)
(+113)
(+36)
AD
114
66
75
105
<div style="text-align: center;">(b) Clustering performance (excluded PD cases)</div>
CN+Control
AD
accuracy
macro-F1
precision
recall
F1
precision
recall
F1
baseline
71.36
98.95
82.92
95.65
36.67
53.01
74.9
68.0
+PSS
78.51
95.47
86.16
88.98
58.33
70.47
81.1
78.3
<div style="text-align: center;">(c) Clustering performance (included PD cases)</div>
CN+Control
AD
accuracy
macro-F1
Specificity
precision
recall
F1
precision
recall
F1
of PD
baseline
77.29
88.99
82.73
57.89
36.67
44.90
73.7
63.8
69.7
+PSS
83.77
88.76
86.19
68.18
58.33
62.87
79.9
74.5
75.8
cluster centers (inter-cluster variability). What is noteworthy here is the degree of decrease in the standard deviation, which reached an average of 22.6%. This ability to reduce not only the variability of data in the same category, but also the directional variability up to different data categories is an important feature in CBIR. In this experiment, we only built an image transformer (i.e. PSSnetwork) that converts CN_GE to CN_SI cases, but we could confirm that the harmonization is desirable for categories that are not included in the training in this way. This strongly suggests that the strategy we have adopted – that is, not having to build image harmonizers for all scanner types – may have sufficient harmonization effects for many types of scanners. Incidentally, the distances between PD and CN (ADNI-CN vs. PD and ALL-CN vs. PD) are closer than the distances between other categories. This supports the validity of the assumption we made in our experiment that PD and CN are outwardly indistinguishable, and therefore, they can be treated as the same class. In contrast, if we look closely, we can see that the distances of the gravity centers between PD and CN (0.249→0.269) and PD and Control (0.256→0.297) are slightly increased by PSS, and Table 3 shows that the variation of PD is greatly reduced by PSS. From this, we can say that the PSS is moving the PDs into smaller groups away from CN and Control. This can be taken as an indication that the model trained by DI-PSS tends to consider PD as a different class that is potentially separated from the CN category. Since the size of the dataset for this experiment was limited, we would like to run tests with a larger dataset in the future.
2) Separability of the embedding for CBIR Thanks to the harmonization of scanners by PSS, the proposed DI-PSS not only reduces the variability of lowdimensional representations of each disease category, which
2) Separability of the embedding for CBIR Thanks to the harmonization of scanners by PSS, the proposed DI-PSS not only reduces the variability of lowdimensional representations of each disease category, which
could not be reduced by deep metric learning learning alone as adopted in DCMML [9], but also reduces the differences among datasets, resulting in a significant performance improvement in the clustering ability of low-point representations. The PD data are different from the ADNI data used for training, and thus, it is an unknown dataset from our model. The improvement of clustering performance by the proposed DI-PSS for PD as well is an important and noteworthy result for the realization of CBIR.
# C. VALIDITY OF THE MODEL ARCHITECTURE
The recently proposed data harmonization methods for brain MR images by Moyer et al [25] and Dinsdale et al. [26] have been reported to be not only logically justified but also very effective. However, as mentioned above, these methods are difficult to apply to CBIR applications because images from all scanners are theoretically needed to train the model. Our DI-PSS is a new proposal to address these problems. Although DI-PSS only learned the transformation from CN_GE to CN_Siemens, the improvement of the properties of the obtained embeddings was confirmed even for combinations that included other companies’ scanners, such as the Philips scanner, and different disease categories (AD) that were not included in the training. The results are evidence of proper data harmonization. We think this is due to the combination of MRICloud, an advanced skull stripping algorithm that performs geometric and volumetric positioning, and CycleGAN’s generic style transformation capabilities and distance metric learning, which make up the PSS network. Experiments with large-scale data from more diverse disease classes are needed, but in this experiment, we could confirm the possibility of obtaining effective scanner standardization by building one model that translates into a standard scanner.
# LIMITATIONS OF THIS STUDY
The number of data and diversity of their conditions used in these experiments are limited. There is also a limit to the
number of diseases we considered. In the future, verification using more data is essential.
# VII. CONCLUSION
In this paper, we proposed a novel and effective MR image embedding method, DI-PSS, which is intended for application to CBIR. DI-PSS achieves data harmonization by transforming MR images to look like those captured with a predefined standard scanner, reducing the bias caused by variations in scanners and scan protocols, and obtaining a low-dimensional representation preserving disease-related anatomical features. The DI-PSS did not require training data that contained MRIs from all scanners and protocols; One set of image converters (i.e., CN_GE to CN_Siemens) was sufficient to train the model. In the future, we will continue the validation with more extensive and diverse data.
# ACKNOWLEDGMENT
This research was supported in part by the Ministry of Education, Science, Sports and Culture of Japan (JSPS KAKENHI), Grant-in-Aid for Scientific Research (C), 21K12656, 2021–2023. The MRI data collection and sharing for this project was funded by the Alzheimer’s Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12–2-0012). ADNI is funded by the National Institute on Aging, the National Institute of Biomedical Imaging and Bioengineering, and through generous contributions from the following: AbbVie, Alzheimer’s Association; Alzheimer’s Drug Discovery Foundation; Araclon Biotech; BioClinica, Inc.; Biogen; Bristol-Myers Squibb Company; CereSpir, Inc.; Cogstate; Eisai Inc.; Elan Pharmaceuticals, Inc.; Eli Lilly and Company; EuroImmun; F. Hoffmann-La Roche Ltd and its affiliated company Genentech, Inc.; Fujirebio; GE Healthcare; IXICO Ltd.; Janssen Alzheimer Immunotherapy Research & Development, LLC.; Johnson & Johnson Pharmaceutical Research & Development LLC.; Lumosity; Lundbeck; Merck & Co., Inc.; Meso Scale Diagnostics, LLC.; NeuroRx Research; Neurotrack Technologies; Novartis Pharmaceuticals Corporation; Pfizer Inc.; Piramal Imaging; Servier; Takeda Pharmaceutical Company; and Transition Therapeutics. The Canadian Institutes of Health Research is providing funds to support ADNI clinical sites in Canada. Private sector contributions are facilitated by the Foundation for the National Institutes of Health (www.fnih.org). The grantee organization is the Northern California Institute for Research and Education, and the study is coordinated by the Alzheimer’s Therapeutic Research Institute at the University of Southern California. ADNI data are disseminated by the Laboratory for Neuro Imaging at the University of Southern California. An additional MRI data used in the preparation of this article were obtained from the Parkinson’s Progression Markers Initiative (PPMI) database (www.ppmi-info.org/data). For up-to-date information on the study, visit www.ppmi-info.org. PPMI – a public-private partnership – is funded by the Michael J. Fox Foundation for Parkinson’s Research and funding partners, including AbbVie, Allergan, Avid Radiopharmaceuticals, Biogen, Biolegend, Bristol-Myers Squibb, Celgene, Denali, GE Healthcare, Genentech, GlaxoSmithKline, Lilly, Lundbeck, Merck, Meso Scale Discovery, Pfizer, Piramal, Prevail Therapeutics, Roche, Sanofi Genzyme, Servier, Takeda, Teva, UCB, Verily, Voyager Therapeutics, and Golub Capital.
# REFERENCES
[1] M. Woelfle, P. Olliaro, and M. H. Todd, “Open science is a research accelerator,” Nature chemistry, vol. 3, no. 10, pp. 745–748, 2011. [2] A. Kumar, J. Kim, W. Cai, M. Fulham, and D. Feng, “Content-based medical image retrieval: a survey of applications to multidimensional and multimodality data,” Journal of Digital Imaging, vol. 26, no. 6, pp. 1025– 1039, 2013. [3] Z. Tu and X. Bai, “Auto-context and its application to high-level vision tasks and 3D brain image segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 32, no. 10, pp. 1744–1757, 2010. [4] M. Huang, W. Yang, M. Yu, Z. Lu, Q. Feng, and W. Chen, “Retrieval of brain tumors with region-specific bag-of-visual-words representations in contrast-enhanced MRI images,” Computational and Mathematical Methods in Medicine, vol. 2012, p. 280538, 2012. [5] S. Murala and Q. M. J. Wu, “Local mesh patterns versus local binary patterns: Biomedical image indexing and retrieval,” IEEE Journal of Biomedical and Health Informatics, vol. 18, no. 3, pp. 929–938, 2014. [6] A. V. Faria, K. Oishi, S. Yoshida, A. Hillis, M. I. Miller, and S. Mori, “Content-based image retrieval for brain MRI: An image-searching engine and population-based analysis to utilize past clinical data for future diagnosis,” NeuroImage: Clinical, vol. 7, pp. 367–376, 2015. [7] K. Kruthika, Rajeswari, and H. Maheshappa, “CBIR system using capsule networks and 3D CNN for alzheimer’s disease diagnosis,” Informatics in Medicine Unlocked, vol. 14, pp. 59–68, 2019. [8] Z. N. K. Swati, Q. Zhao, M. Kabir, F. Ali, Z. Ali, S. Ahmed, and J. Lu, “Content-based brain tumor retrieval for MR images using transfer learning,” IEEE Access, vol. 7, pp. 17 809–17 822, 2019. [9] Y. Onga, S. Fujiyama, H. Arai, Y. Chayama, H. Iyatomi, and K. Oishi, “Efficient feature embedding of 3D brain MRI images for content-based image retrieval with deep metric learning,” pp. 3764–3769, 2019. [10] B. Alipanahi, M. Biggs, and A. Ghodsi, “Distance metric learning vs. fisher discriminant analysis,” Proceedings of the 23rd National Conference on Artificial Intelligence, vol. 2, pp. 598–603, 2008. [11] E. Hoffer and N. Ailon, “Semi-supervised deep learning by metric embedding,” arXiv preprint, p. 1611.01449, 2016. [12] K. A. Clark, R. P. Woods, D. A. Rottenberg, A. W. Toga, and J. C. Mazziotta, “Impact of acquisition protocols and processing streams on tissue segmentation of t1 weighted mr images,” NeuroImage, vol. 29, no. 1, pp. 185–202, 2006. [13] X. Han, J. Jovicich, D. Salat, A. van der Kouwe, B. Quinn, S. Czanner, E. Busa, J. Pacheco, M. Albert, R. Killiany, P. Maguire, D. Rosas, N. Makris, A. Dale, B. Dickerson, and B. Fischl, “Reliability of MRIderived measurements of human cerebral cortical thickness: the effects of field strength, scanner upgrade and manufacturer,” Neuroimage, vol. 32, no. 1, pp. 180–194, 2006. [14] M. Yu, K. A. Linn, P. A. Cook, M. L. Phillips, M. McInnis, M. Fava, M. H. Trivedi, M. H. Trivedi, R. T. Shinohara, and Y. I. Sheline, “Statistical harmonization corrects site effects in fuctional connectivity measures from multi-site fMRI data,” Human brain mapping, vol. 39, no. 11, pp. 4213– 4227, 2018. [15] K. Oishi, J. Chotiyanonta, D. Wu, M. I. Miller, and S. Mori, “Developmental trajectories of the human embryologic brain regions,” Neuroscience Letters, vol. 708, p. 134342, 2019. [16] C. Wachinger, A. Rieckmann, and S. Pölsterl, “Detect and correct bias in multi-site neuroimaging datasets,” Medical Image Analysis, vol. 67, p. 101879, 2021. [17] Y. Gao, J. Pan, Y. Guo, J. Yu, J. Zhang, D. Geng, and Y. Wang, “Optimised mri intensity standardisation based on multi-dimensional subregional point cloud registration,” Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization, vol. 7, no. 5-6, pp. 594–603, 2019. [18] H. Um, F. Tixier, D. Deasy, Bermudez, J. O, R. J. Young, and H. Veeraraghavan, “Impact of image preprocessing on the scanner dependence of multi-parametric MRI radiomic features and covariate shift in multiinstitutional glioblastoma datasets,” Physics in Medicine and Biology, vol. 64, no. 16, p. 165011, 2019. [19] W. E. Johnson, C. Li, and A. Rabinovic, “Adjusting batch effects in microarray expression data using empirical Bayes methods,” Biostatistics, vol. 8, no. 1, pp. 118–127, 2006. [20] J.-P. Fortin, D. Parker, B. Tunc, T. Watanabe, M. A. Elliott, K. Ruparel, D. R. Roalf, T. D. Satterthwaite, R. C. Gur, R. E. Gur, R. T. Schultz, R. Verma, and R. T. Shinohara, “Harmonization of multi-site diffusion tensor imaging data,” bioRxiv, 2017. [Online]. Available: http://biorxiv.org/content/early/2017/03/15/116541
[21] J.-P. Fortin, D. Parker, B. Tunç, T. Watanabe, M. A. Elliott, K. Ruparel, D. R. Roalf, T. D. Satterthwaite, R. C. Gur, R. E. Gur, R. T. Schultz, R. Verma, and R. T. Shinohara, “Harmonization of multi-site diffusion tensor imaging data,” NeuroImage, vol. 161, pp. 149–170, 2017. [22] J.-P. Fortin, N. Cullen, Y. I. Sheline, W. D. Taylor, I. Aselcioglu, P. A. Cook, P. Adams, C. Cooper, M. Fava, P. J. McGrath, M. McInnis, M. L. Phillips, M. H. Trivedi, and M. M. Weissman, “Harmonization of cortical thickness measurements across scanners and sites,” NeuroImage, vol. 167, pp. 104–120, 2017. [23] F. Zhao, Z. Wu, L. Wang, W. Lin, S. Xia, D. Shen, and G. Li, “Harmonization of infant cortical thickness using surface-to-surface cycle-consistent adversarial networks,” Med Image Comput Comput Assist Interv, vol. 11767, pp. 475–483, 2019. [24] C. Zhao, J. C. Reinhold, A. Carass, K. C. Fitzgerald, E. S. Sotirchos, S. Saidha, J. Oh, D. L. Pham, P. A. Calabresi, P. C. M. van Zijl, and J. L. Prince, “Deepharmony: A deep learning approach to contrast harmonization across scanner changes,” Magnetic Resonance Imaging, vol. 64, pp. 160–170, 2019. [25] D. Moyer, G. Ver Steeg, C. M. W. Tax, and T. P. M., “Scanner invariant representations for diffusion MRI harmonization,” Magnetic resonance in medicine, vol. 84, no. 4, pp. 2174–2189, 2020. [26] N. K. Dinsdale, M. Jenkinson, and A. I. L. Namburete, “Deep learningbased unlearning of dataset bias for MRI harmonisation and confound removal,” NeuroImage, vol. 228, p. 117689, 2021. [27] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” pp. 2242–2251, 2017. [28] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky, “Domain-adversarial training of neural networks,” Journal of Machine Learning Research, vol. 17, pp. 1–35, 2016. [29] M. van der Laurens and G. Hinton, “Visualizing data using t-SNE,” Journal of machine learning research, vol. 9, no. 86, pp. 2579–2605, 2008. [30] S. Mori, D. Wu, C. Ceritoglu, Y. Li, A. Kolasny, M. A. Vaillant, A. V. Faria, K. Oishi, and M. I. Miller, “MRICloud: Delivering high-throughput mri neuroinformatics as cloud-based software as a service,” Computing in Science Engineering, vol. 18, no. 5, pp. 21–35, 2016. [31] H. Arai, Y. Chayama, H. Iyatomi, and K. Oishi, “Significant dimension reduction of 3D brain MRI using 3D convolutional autoencoders,” pp. 5162–5165, 2018. [32] R. B. Postuma, D. Berg, M. Stern, W. Poewe, C. W. Olanow, W. Oertel, J. Obeso, K. Marek, I. Litvan, A. E. Lang, G. Halliday, C. G. Goetz, T. Gasser, B. Dubois, P. Chan, B. R. Bloem, C. H. Adler, and G. Deuschl, “MDS clinical diagnostic criteria for Parkinson’s disease,” Movement Disorders, vol. 30, no. 12, pp. 1591–601, 2015. [33] F. J. Meijera, B. Goraja, B. R. Bloemc, and R. A. Esselinkc, “Clinical application of brain mri in the diagnostic work-up of parkinsonism,” Journal of Parkinson’s Disease, vol. 7, pp. 211–217, 2017. [34] K. Oishi, A. Faria, H. Jiang, X. Li, K. Akhter, J. Zhang, J. T. Hsu, M. I. Miller, P. C. M. van Zijl, M. Albert, C. G. Lyketsos, R. Woods, A. W. Toga, G. B. Pike, P. Rosa Neto, A. Evans, J. Mazziotta, and S. Mori, “Atlas-based whole brain white matter analysis using large deformation diffeomorphic metric mapping: application to normal elderly and alzheimer’s disease participants,” Neuroimage, vol. 46, no. 2, pp. 486–499, 2009. [35] A. Y. Ng, M. I. Jordan, and Y. Weiss, “On spectral clustering: Analysis and an algorithm,” Proceedings of the 14th International Conference on Neural Information Processing Systems: Natural and Synthetic, pp. 849– 856, 2001.
