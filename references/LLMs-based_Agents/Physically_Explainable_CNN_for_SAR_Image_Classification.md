# Physically Explainable CNN for SAR Image Classification
Zhongling Huanga, Xiwen Yaoa,∗, Ying Liua, Corneliu Octavian Dumitrub, Mi Junwei Hana
Zhongling Huanga, Xiwen Yaoa,∗, Ying Liua, Corneliu Octavian Dumitrub, Mihai Datcub,c and Junwei Hana
athe BRain and Artificial INtelligence Lab (BRAIN LAB), School of Automation, Northwestern Polytechnical University, Xi’an, 710072, China bRemote Sensing Technology Institute (IMF), German Aerospace Center (DLR), Wessling, 82234, Germany cUniversity POLITEHNICA of Bucharest (UPB), Bucharest, 060042, Romania
A R T I C L E I N F O
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5912/59123de6-2f29-4218-a6cc-fc13429c8eff.png" style="width: 50%;"></div>
# 1. Introduction
Synthetic Aperture Radar (SAR) can work in all-day all-weather conditions as an active microwave sensing technology. Different from the optical remote sensing images close to the visual understanding system of human eyes, SAR images reflect the electromagnetic characteristics of objects and terrain. In order to understand SAR images in a more comprehensive way, the artificial intelligence approaches should pay close attention on not only the visual information, but also the physical properties of SAR. SAR image classification is a basic task, aiming to assign the semantic label to each SAR image patch. Some conventional theory-driven approaches were explored to extract the hand-crafted features based on the expertise of SAR, e.g., the statistic model based Leng et al. (2020); Gao et al. (2017) and the physical model based methods Leng et al. (2019). These model based approaches have strong interpretability, yet the feature selection and classifier design are time-consuming and lack flexibility. As a comparison, the data-driven deep learning approaches can build an end-to-end system to learn the hierarchical features automatically and predict the semantic labels simultaneously without human intervention, superior to the pure model-based methods on SAR image classification tasks Huang et al. (2017); Chen et al. (2016). Nevertheless, the current data-driven solutions for SAR image classification are still facing several challenges. The first is the contradiction between the data-hungry deep learning approaches and the expensive cost in manual annotation for SAR. At present, some pre-training related
methods are popularized to tackle the issue, such as transfer learning and self-supervised learning. The transfer learning methods utilize the models pre-trained on other data domains (like natural images, optical remote sensing imagery, etc) via fine-tuning Huang et al. (2017), domain adaptation Huang et al. (2020b), meta-learning Fu et al. (2022), etc. The self-supervised learning usually takes the current domain data without annotations to optimize a designed contrastive or pretext task, obtaining the pre-training model for the downstream classification Wen et al. (2021); Ren et al. (2021). Despite the good performance of transfer learning and self-supervised learning on SAR image classification, the prediction of most deep models is short of physical explanation. In consideration of the special physical characteristics underlying SAR images, it is important to develop the hybrid approaches that blend the deep learning algorithms with physical models in SAR domain to keep the prediction consistent with physics and expert knowledge, which still remains a big challenge in the current researches. At first, such feature fusion methods that combine the CNN features with the handcrafted description of SAR images were proposed Zhang et al. (2020); Wang et al. (2021); Sun et al. (2020), but there still exist limitations on providing explanations and physical insights of SAR. More advanced hybrid approaches are required to embed the prior scientific knowledge from physical model into the deep neural networks de Bézenac et al. (2019), particularly in SAR image classification domain, where the relevant studies are at the initial stage with a rising trend Huang et al. (2020a). To meet the above challenges in SAR image classification, we propose a novel physically explainable CNN that blends the data-driven and model-driven approach to perform
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c75a/c75aef90-fbe3-4be2-b3fc-c7015328aa97.png" style="width: 50%;"></div>
Figure 1: A surrogate task 푓푃퐺푁is built based on the physical information of SAR 푦푝ℎ푦derived from the explainable model 푓푋푀. Thus, the prior knowledge of physical model is embedded as feature representation 퐹푃퐴, which is successively injected into the main classification task via 푓푃퐼푁to learn the semantic label 푦푡푔푡.
impressive generalization with limited labeled data and achieves physically explainable predictions. Our motivations are two-fold, as depicted in Fig. 1. Firstly, we intend to build a physics-inspired data-driven model for SAR image classification such as Daw et al. (2020); Park and Park (2019); Svendsen et al. (2018) in other research fields, which embeds the knowledge prior of physical model into the neural network. Secondly, inspired by self-supervised learning where the semantic feature embeddings are learned without supervision, we set a surrogate task based on the physical model to leverage the unlabeled SAR images and further support the main classification task. The proposed method, namely physics guided and injected learning (PGIL), is composed of three modules:
1. Explainable Model (XM): We adopt the explainable model to generate the abstract physical representation for each SAR image patch in semantic level. 2. Physics Guided Network (PGN): We propose a novel unsupervised learning neural network based on the designed surrogate task under the guidance of the abstract physical representation. The knowledge prior in the explainable model is converted as feature embeddings with PGN, aware of physical properties of SAR. 3. Physics Injected Network (PIN): As for the main classification task, PIN is proposed to introduce the physics-aware features into the popular CNN pipeline. It captures more comprehensive representations and maintains the physical consistency of features, so as to prevent overfitting effectively with a few labeled samples available.
For evaluation, a hybrid Image-Physics dataset format is proposed, equipped with both SAR image patches and the corresponding physical scattering mechanisms. Sufficient experiments are conducted mainly on the Sentinel-1 seaice classification dataset and also the Gaofen-3 SAR data to demonstrate the effectiveness of each module in the proposed method. The results show that the proposed PGIL exhibits remarkable generalization performance compared with the counterpart CNN architecture with supervised learning, transfer learning, and self-supervised learning in case of limited labeled data. More importantly, the physical explanations are discussed to demonstrate how the prior knowledge constrains the network from training and how the physics consistency is maintained in the predictions. The contributions are summarized as follows:
1. A novel physically explainable deep learning method is proposed for SAR image classification that deeply integrates the data-driven and theory-driven approaches 2. By establishing a novel surrogate task based on explainable physical models, an unsupervised physics guided network is optimized to learn general features aware of prior knowledge. 3. A hybrid Image-Physics dataset formation is proposed for evaluation, which combines the image and physics information of SAR in a concise way. 4. We analyze the physics awareness of features, the good generalization on limited labeled data, and the explainability as well as the physics consistency of the predictions by sufficient experiments and discussions.
The rest of this paper is organized as follows. Section 2 reviews the background knowledge of physical models applied in this paper. Section 3 presents the physics guided and injected learning (PGIL) neural network for SAR image classification. The experiments and discussions are given in Section 4. Finally, Section 5 provides the conclusions.
# 2. Background
In this section, we introduce the background of explainable theory-driven models for SAR applied in the following proposed method. The first is the target decomposition model for PolSAR data to represent the target scattering by several basic scattering mechanisms. One of the well-known methods is the Cloude-Pottier decomposition for full-polarized SAR Cloude et al. (1997), with the entropy 퐻and the angle 훼calculated from coherency matrix. An 퐻∕훼plane is separated into nine zones to depict different scattering characteristics of full-pol SAR data, as shown in Fig. 2 (a). The scattering mechanism classification result can be obtained via the complex Wishart classifier proposed in Jong-Sen Lee et al. (1999). Afterwards, the Cloude-Pottier decomposition model has been improved for dual-polarized SAR images Ji and Wu (2015). In this paper, we employ the Cloude-Pottier decomposition for both full- and HH/HV dualpol SAR data Cloude et al. (1997); Ji and Wu (2015), and
Page 2 of 15
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7dcb/7dcb0dbf-cf3b-45d9-9c1f-34043eb01d8e.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Two different physical models of SAR. (a) shows the 2-dimensional H/훼classification space Cloude et al. (1997) to demonstrate the scattering mechanisms for full-polarized SAR data. (b) indicates the time-frequency analysis model in HDEC-TFA method Huang et al. (2021a) where the backscattering variations in different range and azimuth bandwidths of SAR targets are characterized.</div>
the scattering mechanism classification results are obtained by SNAP software. The polarimetric decomposition is no longer available in single channel SAR image data. The second one we introduce in this paper is based on the time-frequency analysis model for single-polarized SAR data, as shown in Fig. 2(b). The 2-dimensional short-time Fourier transform based timefrequency analysis on complex-valued high resolution SAR data characterizes the backscattering intensity variations of targets with different range and azimuth bandwidths, denoted as sub-band scattering pattern Huang et al. (2021a). Given a specific target with the position of (푥0, 푦0), and a segment 푠 centered in (푥0, 푦0). The sub-band scattering pattern of target (푥0, 푦0) is defined as
(1)
where 푤(푓푟, 푓푎) represents a series of bandpass filters centered on frequency pairs {(푓푟, 푓푎)} in both range and azimuth directions. The details can be found in Huang et al. (2021a). Fig. 2(b) gives some examples of the extracted subband scattering pattern for different targets. A learning based HDEC-TFA method was proposed in Huang et al. (2021a) to classify the scattering patterns. The above two different physical models will be applied in our proposed method. Fig. 3 (a) shows the visualized result of H/훼-Wishart classification Cloude et al. (1997) on full-polarized SAR data, where the labels are consistent with Fig. 2 (a). Fig. 3 (b) is the HDEC-TFA result Huang et al. (2021a) on single channel (HH) SAR image, where the given classes and colorization are in accordance with Huang et al. (2021a).
# 3. Physics Guided and Injected Learning
In most patch-wise SAR image classification methods, the processed SAR amplitude images denoted as 푥퐼are
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9194/9194355d-58f9-458e-9e99-63756da741d9.png" style="width: 50%;"></div>
(b)
(a)
Figure 3: Given a SAR image 푥, (a) shows the H/훼-Wishart classification result Cloude et al. (1997) on full-polarized SAR data, and (b) is the HDEC-TFA result Huang et al. (2021a) on single channel (HH) SAR data.
considered other than the original complex product 푥to predict the image label 푦푡푔푡. Thus, the intrinsic electromagnetic characteristics of SAR is not considered but desired. To this end, our proposed physics guided and injected learning (PGIL), as summarized in Fig. 1, leverages the visual friendly image data and the underlying prior knowledge in physical model. The basic motivation is to embed the physics knowledge into the neural network effectively. Three main modules are included in PGIL, that are explainable models (XM), physics guided learning network (PGN), and physics injected learning network (PIN). XM offers prior knowledge of physical model. PGN convert the prior knowledge into feature embedding, which is successively fused in PIN for label prediction. The overall framework is depicted in Fig. 4. XM acts on complex SAR image data 푥, where the explainable descriptor 푦푝ℎ푦is obtained to represent the physical scattering properties of SAR image:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2647/26479f4d-6d7f-4eec-9b4b-a1270dd0cee6.png" style="width: 50%;"></div>
<div style="text-align: center;"> 4: The physics guided and injected learning (PGIL) for SAR image clas</div>
푦푝ℎ푦plays a major role in establishing the surrogate task of PGN for optimization, therefore referred to physics guided signal. PGN follows an unsupervised learning manner and outputs the feature embedding 퐹푃퐴aware of prior physical knowledge, namely physics-aware features. The mapping function 푓푃퐺푁is written as:
(3)
Finally, PIN is proposed to complete the main classification task where the physics-aware feature 퐹푃퐴is injected, denoted as
(4)
# 3.2. Explainable Models
As introduced in Section 2, the physics-based H/훼Wishart Cloude et al. (1997); Ji and Wu (2015) and HDECTFA Huang et al. (2021a) models serving as a part of XM, are adopted to obtain the scattering mechanism of target in SAR image 푥, denoted as (푥). The discrete physical scattering labels (푥) either depict different zones in H/훼plane Cloude et al. (1997); Ji and Wu (2015), or refer to different targets with diverse scattering variation patterns (Fig. 2(b)) Huang et al. (2021a). Compared with SAR image label 푦푡푔푡, (푥) is too physics-specific to offer the semantic information. In view of optimizing PGN to obtain the feature embedding 퐹푃퐴 aware of physics knowledge as well as being semantic distinctive, we additionally employ the Latent Dirichlet Allocation (LDA) on (푥) to output the topic mixture 푦푝ℎ푦, denoted as 푦푝ℎ푦= ((푥)). In LDA topic modeling, the document formulated with bag-of-words representation is characterized by a distribution
over latent topics, and each topic is represented by a distribution over words in the vocabulary. The corpus is gathered from dataset to train the LDA model in an unsupervised pattern and the generative process is explainable. The details can be found in the related literature Blei et al. (2003); Rasiwasia and Vasconcelos (2013). We redefine the essential variables of LDA on the basis of SAR scattering characteristics as follows.
• corpus The corpus is collected for training the LDA model that formed as a matrix with the size 푁푣× 푁푑, where 푁푑is the number of document.
Finally, 푦푝ℎ푦is obtained as the topic mixture (namely Bag of Topics, BoT) of (푥), 
(5)
where 휑푘denotes the score of the 푘th topic. Generally, the summation of 휑푘equals 1.
# 3.3. Physics Guided Network
The role of PGN lies in embedding the prior physics knowledge in a neural network, so as to extract the physicsaware features with semantic discrimination beneficial to
Page 4 of 15
classification. The optimization of PGN is motivated by the pretext task setting in self-supervised learning Misra and Maaten (2020). T˘anase et al. T˘anase et al. (2017) pointed that the topic semantics of scattering properties are close to human semantics used for basic land-cover types. Correspondingly, we propose to build a surrogate task under the following assumption: the SAR image features and the topic mixture of physical scattering labels should share common attributes in semantic level. In other word, the physics descriptor 푦푝ℎ푦can be partly represented by high-level deep features extracted from SAR image 푥퐼. We apply the first three residual blocks of ResNet18 He et al. (2016) as the SAR image feature extractor, denoted as 푖푚푔(푥퐼). Since the weak relationship between physics-specific topics 푦푝ℎ푦and image-specific features, we design the physics mapping layer (PML) denoted as 푝푚푙 to narrow the knowledge gap. The PML is composed of a convolution module and a fully-connected layer, mapping the image representations to physics topic space, denoted as 휙퐼= 푝푚푙(푖푚푔(푥퐼)) where 휙퐼∈ℝ퐾. The following objective function describes the soft semantic relations between them, that
where 푲푣푖푠denotes the topics that can be represented by features from SAR vision domain. Equa. (6) is a relaxed constraint, where only the related semantics are considered to be similar. As a comparison, the hard constraint is
where the unrelated semantics (푘∉푲푣푖푠) are additionally required to be highly different. We choose the soft constraint in our method considering the semantic gap between SAR physics knowledge and the visual perception of image data. The follow-up experiments will discuss the differences of soft and hard constraints. In order to simplify the gradient descent optimization, we modify Equa. (6) as
(8)
∑ where 훿푘is the activation term with a value of 1 or 0. We choose the locations where 푦푘 푝ℎ푦⩾훼as activated ones (훿푘= 1) with a probability of 푝푎, otherwise deactivated (훿푘= 0). The parameter 훼filters the remarkable attributes in 푦푝ℎ푦, that is, only the significant semantic topics are considered as possibly related. The probability 푝푎decides only part of the semantic topics are selected to be related.
# 3.4. Physics Injected Network
The physics injected network (PIN) is designed to inject the physics-aware features obtained from the unsupervised
where 푐푖= 퐼푛푗(푟푠18(푥푖 퐼), 푡푟(푖푚푔(푥푖 퐼))). In order to ensure the physics-aware features to be more adaptive to the classification task, we add the small weighted soft constraint 퐿푠푟in Equa. (8) as an regularization term and fine-tune the PGN slightly in supervised classification training. The total loss function is written as
(10)
# 4. Experiments
In this section, we firstly introduce the hybrid ImagePhysics data format and the experimented datasets. Then, we conduct several experiments to prove the effectiveness of our proposed method with sufficient discussions.
# 4.1. Dataset and Experimental Setup
Most SAR image classification datasets, like OpenSARUrban Zhao et al. (2020), only provide the processed SAR amplitude images 푥퐼for a better visual understanding. For the purpose of leveraging the underlying physics knowledge in SAR and meanwhile preventing large storage space for complex data 푥, we propose the hybrid Image-Physics (Img-Phy) data format to integrate 푥퐼and (푥) in a concise way, to accomplish the proposed PGIL method. We mainly evaluate our method on a sea-ice classification dataset acquired by Sentinel-1, as shown in Fig. 5 (Left). The Sentinel-1 Interferometric Wide (IW) SAR data in polar region is downloaded 1, both single-look complex (SLC) and multi-looked Ground Range Detected High resolution
Page 5 of 15
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bd97/bd97656c-28ed-4b7a-948a-4f26b93e28ad.png" style="width: 50%;"></div>
<div style="text-align: center;">Img: Sentinel-1 GRD product (HH) ; Phy: H/alpha-Wishart on Dual-polarized SAR Data</div>
Figure 5: The hybrid Img-Phy data examples. Left: sea-ice classification dataset with manual annotations. The Img part is the Sentinel-1 grounded range detected (GRD) product and the Phy part is the H/훼-Wishart result from the single-look complex (SLC) product. Right: the Gaofen-3 SAR image patches in urban areas. Phy-1: the HDEC-TFA result Huang et al. (2021a), Phy-2: the H/훼-Wishart result Jong-Sen Lee et al. (2004), which are also given in Fig. 3.
Table 1
<div style="text-align: center;">The sea-ice classification dataset of Sentinel-1.</div>
class
YI
GL
FY
WT
IB
FI
OI
train
train-45, train-35, train-25, train-15, train-5
test
1399
51
129
406
154
76
1107
(GRDH) product (HH channel). Seven sea-ice types are annotated with the patch size of 256×256 for GRDH image, serving as 푥퐼. Besides, the dual-polarized SLC data is processed by SNAP software 2 to obtain the H/훼labels, serving as (푥). To ensure (푥) almost covers the same area with 푥퐼, some essential operations like multi-looking, grounded range projection, are required. Since the different pixel spacing between the processed SLC and GRDH, the Phy data is not square anymore, which is stored as a matrix with the size of about 187×139. For better visualization, Fig. 5 shows the resized square Phy patch in RGB format, where each color represents a scattering label. For a better evaluation, especially in case of limited labeled data, we randomly select 45, 35, 25, 15, and 5 samples from each class for training, denoted as train-45, train-35, train-25, train-15, and train-5, respectively. The test set is fixed, as shown in Table 1. In addition, a Gaofen-3 SAR scene image covering a wide urban area is experimented 3, shown in Fig. 5 (Right). Seven land cover and land use classes are annotated. The
<div style="text-align: center;">Img Phy-1 Phy-2</div>
class
WT
AL
MF
AP
HD
MD
MU
train
20
20
20
10
20
20
20
test
157
80
650
34
231
871
775
training/test details are listed in Table 2. We present two different types of (푥) introduced before, that are Phy-1 (the HDEC-TFA result Huang et al. (2021a) on single-polarized HH channel data) and Phy-2 (the H/훼-Wishart result JongSen Lee et al. (2004) on full-polarized data). In the following experiments, we will discuss the physics guided learning results with different Phy data. When using H/훼-Wishart result of the polarimetric SAR as the Phy data, the obtained (푥) is with 푁푠= 9 classes of physical scattering characteristics, corresponding to nine zones in H/훼plane shown in Fig. 2(a). While for HDEC-TFA result of the single channel (HH) SAR image, 푁푠is set to 15 according to Huang et al. (2021a). In the topic modeling for physics guided signals generation, the vocabulary size 푁푣 and the topic number 퐾of the LDA model are set to 500 and 175, respectively. Note that the topic number 퐾is a critical parameter in the algorithm, that will be under discussions in the following experiments. To determine 훿푘in Equation (8), we set 훼and 푝푎to 0.1 and 0.9, respectively. The following discussions will illustrate the strategy of parameter setting. The physics guided learning is optimized by stochastic gradient descent (SGD) with a fixed learning rate of 0.05,
푝푎.
퐾
train-45
train-35
train-25
train-15
train-5
25
77.57 / 69.34
77.21 / 69.34
77.24 / 68.85
77.33 / 68.85
74.14 / 66.64
50
81.88 / 75.98
80.70 / 74.29
80.85 / 74.66
78.51 / 72.05
78.00 / 70.78
100
83.20 / 78.54
81.88 / 76.26
81.88 / 75.52
80.34 / 74.00
78.45 / 71.46
150
84.14 / 78.79
83.53 / 78.39
82.33 / 76.28
82.00 / 75.47
77.81 / 70.86
175
84.18 / 79.24
83.62 / 78.51
82.03 / 76.95
81.19 / 75.05
78.39 / 71.53
200
82.66 / 77.20
82.51 / 76.99
82.45 / 76.78
82.03 / 76.20
78.12 / 72.29
and the momentum is set to 0.9 by default. All Img-Phy pairs in the dataset are fed into the PGN for training, lasting 200 epochs in total. The physics injected learning only takes annotated data to train 푟푠18 and 푡푟. The initial learning rate is set to 0.001 and the cosine annealing strategy is applied to decrease the learning rate to 10−8 in the last 3 epochs of 50 in total. The soft constraint regularization term in Equa. (10) is weighted by 휆set to 0.1. All experiments are conducted on a workstation of 64 bit Linux operating system, with 64G RAM and NVIDIA RTX 3090 graphics card of 24GB GDDR6X VRAM clocked at 1700 MHz.
# 4.2. Unsupervised Physics Guided Learning
The PGN learns physics-aware features 퐹푃퐴from all Img-Phy pairs. To evaluate the discriminative ability of 퐹푃퐴in semantic domain, we train a support vector machine (SVM) on 퐹푃퐴to predict 푦푡푔푡. In this section, we will firstly discuss how the topic number 퐾, and the activation strategy 훿푘effect the discriminative physics-aware feature learning based on the sea-ice classification dataset of Sentinel-1. Then, the characteristics of physics-aware features, as well as the differences between hard and soft constraint, are analyzed. At last, we additionally evaluate the effectiveness of the proposed physics guided learning approach on Gaofen-3 SAR data covering a wide urban area.
# 4.2.1. Hyperparameter Discussion
Firstly, we set 훼and 푝푎to 0.1 and 0.9, respectively, and discuss the topic number 퐾. The classification results are shown in Table 3, where the highest overall accuracy or F1-score are marked in red. We check out six different values of 퐾(25, 50, 100, 150, 175, 200) and train the SVM classifier on 5 different training sets. It can be observed from Table 3 that a larger 퐾almost leads to a better result. Since the fullyconnected layer in the PML module is determined by the topic number, a large 퐾would introduce plenty of parameters and increase the computation load. Consequently, we decide 퐾= 175 for a better trade-off. The topic number 퐾decides the shared attributes space ℝ퐾where the image representation is mapped. The assumption is proposed to build a bridge between 휙퐼∈ℝ퐾 and {휑푘}. A larger 퐾ensures more fine-grained physics attributes, so that the soft constraint in Equa. (6) can be more precise. We calculate the sparsity of BoT vector 흋, defined
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/227a/227a0cfe-2b71-4cf4-b445-b2163526500a.png" style="width: 50%;"></div>
Figure 6: The topic sparsity of LDA model in case of different topic number 퐾.
as
(11)
|||| where || ⋅||0 denotes the L0 norm. Fig. 6 plots the sparsity of BoT representation in case of different topic numbers. We find the fine-grained physics attributes lead to a more sparse representation of BoT encoding. The BoT sparsity is also highly relate to the physics-aware feature performance. Intuitively, a sparsity greater than 0.985 can be regarded as a good choice with the topic number 퐾no less than 150. Next, we will discuss how to fix the activated physics attributes 훿푘in Equa. (8). 훿푘is determined by 훼and 푝푎, where 훼filters the prominent attributes in 푦푝ℎ푦as the candidates, and 푝푎randomly select the potential attributes to calculate the constraint. Table 4 shows the SVM classification results of different 훼, with 퐾= 175 and 푝푎= 0.9. When 훼equals 0, some unconsidered attributes (휙푘close to 0) may be included to guide the network to learn the insignificant features. It is better to set 훼as a small value but greater than 0, e.g. 0.1 in our case. Table 5 discusses the values of 푝푎in the context of 훼= 0.1 and 퐾= 100. 푝푎is the probability for randomly selecting the potential attributes in 푦푝ℎ푦. The value of 푝푎 from 0.5 to 1.0 indicates the constraint becomes rigid. The result shows it is better to choose a greater values of 훼, which demonstrates a majority of remarkable attributes should be considered. Here, we choose 훼= 0.9 in our experiments.
훼
train-45
train-35
train-25
train-15
train-5
0
84.05 / 79.15
83.32 / 78.05
82.69 / 77.32
81.55 / 75.55
78.20 / 71.46
0.1
84.18 / 79.24
83.62 / 78.51
82.03 / 76.95
81.19 / 75.05
78.39 / 71.53
0.2
83.96 / 78.98
83.20 / 77.93
82.90 / 77.54
82.42 / 76.43
78.09 / 71.44
0.3
81.58 / 76.11
82.21 / 76.55
81.55 / 75.69
79.38 / 73.20
75.89 / 70.32
푝푎
train-45
train-35
train-25
train-15
train-5
1.0
83.20 / 78.54
81.88 / 76.26
81.88 / 75.52
80.34 / 74.00
78.45 / 71.46
0.9
83.35 / 78.09
83.02 / 77.37
82.48 / 76.00
81.13 / 74.74
79.23 / 71.87
0.8
83.87 / 78.73
82.93 / 77.51
82.45 / 76.64
81.10 / 74.60
78.24 / 71.84
0.7
83.17 / 77.93
82.93 / 77.66
81.58 / 76.30
79.74 / 73.70
78.27 / 71.83
0.6
82.18 / 76.81
81.52 / 76.54
80.67 / 75.25
79.74 / 73.65
77.39 / 71.32
0.5
80.34 / 74.39
80.79 / 74.71
79.86 / 73.17
79.02 / 72.58
75.80 / 68.96
To summarize, the topic number 퐾is the most important hyperparameter in PGN learning which can be determined by the sparsity of BoT representation. 훿푘controls a relax activation of physics attributes, decided by a relatively casual value of 훼and 푝푎. We recommend to set 훼to 0.1 or 0.2, and 푝푎to 0.9 or 0.8, respectively.
# 4.2.2. The Physics-Aware Features Discussion
The PGN is comprised of the ResNet-13 backbone and the 2-layer PML module. The ResNet-13 backbone 푖푚푔extracting hierarchical features from 푥퐼is basically image specific, with the higher-level features becoming closer to semantic meaning. The PML module transforms 푖푚푔(푥) to the physics attributes space ℝ퐾, building the semantic relation between image representation and physics knowledge. The physics-aware features are expected to be discriminative in the classification semantic domain and also with physics awareness. We analyze the outputs of different layers in PGN to demonstrate the semantic discrimination and the physics awareness of features. The SVM classification results are used to indicate the semantic discrimination, as shown in the first row in Table 6. The features from ResBlk-3 reach the highest classification accuracy of 84.18%, followed by an overall accuracy of 82.39% for the features from the convolution layer in PML module. The results demonstrate how the feature discrimination in semantic level changes with the physics guided neural network. Note that the physics BoT encoding only achieves 61.53% in classification, which indicates 푦푝ℎ푦is highly physics specific and the semantic gap is truly existed between 푦푝ℎ푦and 푦푡푔푡. Even so, 푦푝ℎ푦 can guide the PGN to learn the discriminative features close to 푦푡푔푡successfully, with the designed objective function. Intuitively, the first row in Fig. 7 displays the annotated labels
with different colors, indicating the feature discrimination in semantic level. We can observe that the BoT encodings are confused in understanding the semantic labels, since the SAR images in the same class may have different physics attributes. After the physics guided learning, the feature discrimination in semantic level has been improved from PML-fc to ResBlk-3 layer. Due to the lower level of ResBlk2, the features in Fig. 7(a) are not as discriminative as those of ResBlk-3 in Fig. 7(b). Additionally, we demonstrate the physics awareness of features by visualization and quantitative metrics, shown in the second row of Fig. 7 and Table 6, respectively. Given the BoT encodings of physics attributes, we apply a 푘means algorithm to cluster the BoT representations into 푘 classes as the color identification in the second row of Fig. 7. Thus, (f)(g)(h)(i)(j) indicate the feature discrimination in physics level, that is, the samples with the same color have similar physics attributes. We can observe how the features become physics aware with the help of PML module. Table 6 also lists the silhouette coefficient of features which reflect the separation between clusters. The silhouette coefficient values between -1 and 1, and a higher silhouette score indicates better discrimination of physics information in this feature space, that is, stronger physics awareness of features. It gradually decreases from physics topic space to image feature space, but still keep a positive value of 0.0351 in ResBlk-3. As a comparison, the silhouette score of features in ResBlk-3 of the traditional CNN learning model is -0.0085, indicating physics unawareness. Hence, we assert the PGN is able to learn the physics-aware features.
# 4.2.3. Hard and Soft Constraint Discussion
We discuss the soft and hard constraint objective func tions defined in Equa. (6) and (7). The soft constraint
<div style="text-align: center;"> SVM classification results (OA / F1-score), and the physics awareness analysis of features (quantified by silhouette score) m different layers in PGN. The soft and hard constraints in Equa. (6) and (7) are both explored.</div>
퐹푃퐴
Constraint
ResBlk-2
ResBlk-3
PML-conv
PML-fc
BoT
SVM result
OA / F1-score (%)
soft
79.05 / 72.75
84.18 / 79.24
82.39 / 76.84
77.90 / 71.74
61.53 / 51.86
hard
78.45 / 71.55
82.99 / 78.05
81.85 / 75.90
77.45 / 71.40
61.53 / 51.86
Physics Awareness
Silhouette score
soft
0.0157
0.0351
0.0904
0.1087
0.4766
hard
0.0181
0.0380
0.0949
0.1202
0.4766
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b42b/b42ba9a2-2685-4934-858b-a66489886169.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: The feature visualization of training data in different layers. Features in (a)(f), (b)(g), (c)(h), (d)(i), are from ResBlk-2, ResBlk-3, PML-conv, and PML-fc, respectively. (e)(j) are BoT representations. (a)(b)(c)(d)(e) are marked with true labels. (f)(g)(h)(i)(j) are marked with 푘-means cluster labels of BoT representations.</div>
only emphasizes the common semantics from physics and vision domain to be highly similar, while the hard constraint additionally restricts the specific ones to be different. The SVM classification results in Table 6 indicate that the soft constraint guides the PGN to learn more general features which achieve better performance in classification. The listed silhouette coefficients of hard constraint in Table 6 are larger than those of soft constraint, which demonstrate the the learned features by hard constraint are more physics specific but less semantic discriminative. The discussion explains the semantic gap between the physics attributes and the image features of SAR, encouraging us to find a trade-off in learning the physics-aware features.
We additionally take the other Gaofen-3 SAR data, as shown in Fig. 5 (right), to demonstrate the effectiveness and generalization ability of the proposed PGN, besides the polarimetric characteristics derived from H/훼-Wishart method Jong-Sen Lee et al. (1999). In our previous work Huang et al. (2021a), we verified that the HDEC-TFA method could automatically discover the time-frequency properties of SAR target in high resolution SAR images, especially for some man-made targets with characteristic scattering behaviors. It is based on the physics meanings of timefrequency analysis on complex SAR data, which reveals the scattering variation on different azimuth angles and range
Table 7 The SVM classification results of features in physics guided learning with different physics information, compared with the supervised end-to-end CNN training. (Overall Accuracy / F1-score (%))
Phy Data
None (CNN)
HDEC-TFA (Phy-1)
H/훼-Wishart (Phy-2)
Result
52.43 / 48.51
68.76 / 62.32
62.54 / 58.71
bandwidths. We apply different physics information on PGN to illustrate our proposed method can be integrated with various physical models. The SAR image in urban city with dense buildings and man-made targets is more complicated than polar area. With very limited annotation, it is difficult for supervised CNN training to learn generalized and discriminative features. Table 7 shows the CNN only achieves an accuracy and F1-score of 52.43% and 48.51%, respectively, with severe overfitting. The SVM classification is utilized to evaluate the physics-aware features by PGN with different Phy data. As recorded in Table 7, the classification result of physics-aware features guided by Phy-1 (HDEC-TFA) signals improves 16.33% in accuracy than CNN training result, 6.22% better than guided by H/훼-Wishart scattering characteristics. It also verifies the effectiveness of HDEC-TFA learning approach
Page 9 of 15
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f997/f997ebcb-437d-4824-856d-c8e026b9f4c4.png" style="width: 50%;"></div>
to extract significant physics properties under circumstance of no polarimetric information available. Fig. 8 visualizes the annotated ground truth, the test results of CNN, PGN with Phy-1 and Phy-2 on two classes, High-density residential area and Airport. The scattering in high-density residential area is strong and extremely complicated so that the visual interpretation is difficult. However, the physical characteristics in this area, as shown in Fig. 3, are more distinctive than other regions, especially in HDEC-TFA case. Consequently, PGN with Phy-1 (HDECTFA) achieves the best prediction result on high-density residential area with minimum false negative samples. There are 4 airports in the SAR image, as shown in Fig. 8, where only 10 annotated samples in Airport-1 are used for training. The PGN with Phy-1 can remarkably predict the remaining 3 unseen airports with only 3 false negative samples, which is much superior than traditional data-driven CNN model.
# 4.3. Supervised Physics Injected Learning
In this section, we discuss the effectiveness of PIN module on sea-ice classification dataset in Table 1 (train45 as the training data). The physics-aware representation 퐹푃퐴is considered as an off-the-shelf feature from PGN to be injected. The baseline is the traditional CNN method, that is, training 푟푠18 with only labeled amplitude SAR images. We both test the training from the scratch strategy and the transfer learning strategy using a SAR image pre-trained model proposed in Huang et al. (2021b). The classification results are shown in Table 8, where the retraining accuracy is only 74.55% and the transfer learning result is 82.57%. It shows that the limited labeled data is insufficient to train a very deep neural network from scratch. After the physics-aware features injection, however, the retraining results improve more than 10%, as shown in Table 8.
<div style="text-align: center;">The ablation study of physics injected learning. (OA (%))</div>
inj-2
inj-3
inj-4
retrain (%)
pre-trained (%)
-
-
-
74.55 ± 1.05
82.57 ± 0.82
√
-
-
82.56 ± 2.36
81.56 ± 1.89
-
√
-
84.52 ± 0.39
85.19 ± 0.75
-
-
√
82.15 ± 1.18
84.05 ± 0.89
√
√
-
82.11 ± 0.31
84.15 ± 0.83
-
√
√
82.77 ± 1.47
83.73 ± 1.35
√
√
√
84.96 ± 0.36
86.40 ± 0.35
We discuss the different locations where the physicsaware features are injected, including the ResBlk-2, ResBlk3, and ResBlk-4, denoted as inj-2, inj-3, and inj-4 in Table 8, respectively. With the same depth of 퐹푃퐴and the features of ResBlk-3, the injection in ResBlk-3 reaches the best performance of single-injection strategy, marked in blue. The results indicate that the obtained physics-aware feature is with abstract meanings, but still has semantic gap with the semantic features for target task. We also find that the multi-layer injection can improve the classification most in both retraining and transfer learning cases. The unsupervised PGN training costs about 4.45h, with the batchsize of 300 and training epochs of 200. Afterwards, the supervised PIN only takes 13.25 minutes for training, with the batchsize of 100 and training epochs of 100.
# 4.4. Ablation Study
In order to demonstrate the effectiveness of each module in the proposed method, we conduct the detailed ablation study of each module on sea-ice dataset (train-45 as the training data). As shown in Table 9, the baseline is set as retraining the CNN model from scratch, achieving an overall accuracy of
<div style="text-align: center;">The ablation study of different modules. (OA (%))</div>
XM
PGN
PIN
SAL
description
result (%)
Baseline
retrain CNN
74.55 ± 1.05
-
√
√
-
use (푥) as 푦푝ℎ푦
80.61 ± 1.18
√
-
√
-
inject Equa. (5)
78.78 ± 1.67
√
√
-
-
fine-tune PGN
82.59 ± 1.47
√
√
√
-
optimizing Equa. (9)
84.96 ± 0.36
√
√
√
√
optimizing Equa. (10)
85.27 ± 0.28
74.55% with strong overfitting. As a contrast, the proposed method reaches 85.27% in average, obtaining about 10.72% improvements. We discuss the effectiveness of the following four parts: 1. Explainable Models (XM): Based on the existing physical scattering labels, generating the abstract BoT encodings with LDA as the physics-guided signals. 2. Physics Guided Network (PGN): Learning the physicsaware features guided by physics BoT representations. 3. Physics Injected Network (PIN): Retraining the CNN with injecting physics-aware features. 4. Self-Adaptive Learning (SAL): Fine-tuning the physics guided network during the PIN training with a combined loss.
Table 9 shows the ablation experiments, excluding each part above-mentioned separately. It is clear that the XM and PGN play the most important roles in the proposed method. Generating an appropriate physics guided signal 푦푝ℎ푦in XM remarkably effects the quality of injected features learned by PGN, with the accuracy increasing from 80.61% to 84.96%. The PGN module ensures the injected knowledge more related to the target task. Otherwise, directly injecting the BoT representation in Equa. (5) only has an accuracy of 78.78% in average, and the proposed PGN contributes 6.18% improvement. The PIN learning makes further efforts on fusing the physics knowledge and vision features together, which has an improvement of about 2.37%. The SAL part which makes the physics-aware features more adaptive to the target task slightly improves the result. Additionally, we compare some self-supervised learning methods in computer vision field Wu et al. (2018); Chen et al. (2020) and also for PolSAR data Ren et al. (2021) with the proposed PGIL, since they all establish pretext tasks for unsupervised learning the feature embeddings. NPID Wu et al. (2018) learned the optimal feature via instance-level discrimination, while SimCLR Chen et al. (2020) conducted the contrastive learning based on dataaugmentation, both focusing on image contents. MI-SSL Ren et al. (2021) was proposed for PolSAR land cover classification, learning discriminative high-level features between multi-modal representations of PolSAR data. In order to adapt MI-SSL method to our case, we changed the SSL input of multi-modal features for full-polarized SAR data to our Img-Phy pairs. The results are listed in Table 10. Although NPID and SimCLR perform well in natural image classification, such as ImageNet, the results
<div style="text-align: center;">Table 10 The comparison with self-supervised learning methods on sea-ice classification dataset. (OA (%))</div>
<div style="text-align: center;">The comparison with self-supervised learning methods on sea-ice classification dataset. (OA (%))</div>
Method
Description
result (%)
NPID
Wu et al. (2018)
instance-level
discrimination
66.56 ± 2.95
SimCLR
Chen et al. (2020)
data-augmented
contrastive learning
74.02 ± 0.78
MI-SSL
Ren et al. (2021)
self-supervised
for PolSAR data
77.75 ± 1.41
PGIL
Proposed
85.27 ± 0.28
demonstrate the pretext tasks of instance-level discrimination and contrastive learning via data-augmentation almost fail in SAR image classification. MI-SSL performs better for comparison, because it considers the multiple representations of SAR data. Our proposed PGIL reaches the best among them.
# 4.5. Interpretability Discussion
In this section, we use the sea-ice classification case to demonstrate the physics explainability of the proposed method. The explainability lies in the following two aspects. Firstly, the topic modeling for physics information provides explainable representations for each SAR image patch. Secondly, the PGN and PIN maintain the explainable physics consistency of features to learn reasonable results and prevent overfitting during automatic training.
4.5.1. Physics Explanation of 푦푝ℎ푦 The PGN optimization is driven by the physics guided signals 푦푝ℎ푦, denoted as ((푥)) = {휙1, 휙2, ..., 휙퐾} in Equa. (5). The LDA topic modeling processing explains the physical scattering characteristics (푥) by a combination of topics. Each latent topic is represented by a set of specific words, that is, the physical scattering characteristics distribution in a small area of SAR image (defined as "word" in LDA). The weight assigned (휙푖) describes the probability of the SAR image patch belonging to the topic 푖. This benefits the understanding of hidden semantic structure between scattering labels of a large-scale SAR image area at an aggregate level. Fig. 10 presents the averaged physics topic distribution of training data for some selected sea-ice classes, with topic number 퐾= 175. Since we have discussed previously that most SAR patches are with highly sparse physics topic representations, the classes whose distribution concentrated in fewer topics, such as iceberg and glaciers shown in Fig. 10, are more characteristic. The iceberg is mainly represented by topic-73 weighted 0.6 and topic-119 weighted 0.13. The word distribution of each topic is also given in Fig. 10, where each word can be explained by the physical scattering properties. In this sea-ice hybrid Phy-Img dataset, word-6 and word-7 are mostly random surface and Bragg surface
Page 11 of 15
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4b20/4b207bd0-ba97-4540-8066-57467639d748.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: The confusion matrix and the classification metrics (precision, recall, f1-score, and overall accuracy) of CNN learning PGN + SVM, and PGIL performances are shown, respectively.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8bca/8bcaf0ce-0c7d-4e39-b6d0-3604dc57cc27.png" style="width: 50%;"></div>
scattering, respectively, and word-50 is a mixture of these two scattering properties. Also, we list the four dominated topics and the topic weights of water bodies and floating ice in Fig. 10. The result indicates the physics attributes of water bodies and the floating ice are similar in semantic level. The inference can be proved from the semantic definition of floating ice –
any form of ice found floating in water 4, that is to say, the SAR image patch with floating ice probably includes water bodies. Additionally, there are similar topics combinations in water bodies, floating ice, and young ice, shown in the zoom-in region of Fig. 10. The young ice has another specific topic of 62, indicating this class could have two kinds of representative physics attributes.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e976/e976c205-5f97-4f4e-b19b-dd9ca6fc424d.png" style="width: 50%;"></div>
# 4.5.2. Explanation of PGIL Results
Since the PGN is trained with the explainable BoT physics encoding, the above inferred information can explain the discrimination of physics-aware features in Fig. 7(b). With more characteristic topic distribution of glaciers and icebergs, their physics-aware features are the most discriminative, and the SVM classification results show the two classes achieve the best F1-score of 0.909 and 0.911, respectively, as shown in Fig. 9. In addition, the previous analysis explains the similar feature distribution of water bodies and floating ice shown in Fig. 7(b), and the features of young ice have two different characteristics, one of which is close to water bodies and floating ice. As a result, we can observe in the confusion matrix that the most misclassified test samples in water bodies are predicted as young ice and floating ice, with a number of 57 and 37, respectively. Among the 11 false negative samples in the floating ice class, 9 are classified to water bodies and 2 to young ice. Two cases are presented in Fig. 11 to demonstrate how the PIN preserves the physics consistency of features. We visualize the features of CNN supervised training, physicsaware features from unsupervised PGN, and the features after injection in PIN by t-sne Maaten and Hinton (2008).
As shown in Fig. 11(a), the CNN feature of instance 996 and 496 in glacier class are far away from the majority due to the different image contents. Based on the similar physics attributes, the middle figure in Fig. 11(a) shows the two samples are close to most glacier data in physics-aware features. In the visualization of features after physics injected learning, we can observe sample 496 and 996 are still close to each other due to the similar visual representation, and also, they maintain the closeness with sample 494, 793, and 1193 as they were in physics-aware features. As a result, we infer the physics injected learning can preserve the physics consistency during the network training. Another example is about sample 5013 and 4224, shown in Fig. 11(b). They have similar image features in CNN training while different from the other first year ice samples. The physics-aware features in the middle figure reveals that sample 5013 and 4224 have their own physics characteristics. The physics constraints existed in physics-aware features are, i.e, sample 5013 having similar physics properties with 4628, 5606, 5906, and sample 4224 being very close to sample 3925 from old ice class. PIN continues this kind of constraint, as shown in the right figure of Fig. 11(b). In a word, the proposed PGN and PIN represent
Page 13 of 15
the SAR images from a more comprehensive perspective than the traditional supervised CNN learning.
# 4.5.3. The Inspiration from Explainability
In addition, the explainability we discussed before can inspire us to improve our deep learning algorithm in the future work. For example, the above analysis indicates the floating ice has a similar physics BoT representation with water bodies so that the physics-aware features of them are not well discriminative in semantic level, as shown in Fig. 11. On the contrary, they can be discriminative based on the visual contents. The final PIN result shows injecting physics-aware features could not improve the performance of recognizing water bodies and floating ice, because their physics knowledge is not as helpful as other classes. We can see in Fig. 9 that the true positive samples of water bodies and floating ice in PGIL result are fewer than those in CNN training result. Thus, it inspires us to re-think the physics injected learning strategy that the constraint of physics consistency should be relaxed in such classes. In our future work, we will further improve the physics injected learning method in this direction to achieve a better result.
# 5. Conclusion
In this paper, we propose a novel physics guided and injected learning neural network for SAR image classification with limited labeled data, to explore the potential of physically explainable deep learning. Three components of PGIL are explainable model, physics guided network, and physics injected network. The prior knowledge in explainable models is encoded into the physics-aware features via unsupervised PGN learning, and then is injected in the classification pipeline through PIN, supervised by limited labeled data. The hybrid Img-Phy dataset format is proposed for evaluation and abundant experiments are conducted on Sentinel-1 and Gaofen-3 SAR data. The results demonstrate the semantic discrimination and the physics awareness of the learned features by PGN, as well as the good generalization. Additionally, we discuss the interpretability of the guided signals in the established surrogate task to prove the results are with physical constraint. The advantages of the proposed PGIL are (1) the unsupervised PGN is a plug-and-play module which can be integrated to any deep learning framework for physics-aware feature injection; (2) the physics knowledge injection is capable of preserving the physics consistency in the prediction and preventing overfitting in case of limited labeled data; (3) the results are explainable with the help of the distinct physical models and expertise to a certain extent, that inspire us to further improve the deep learning model in the right direction. The sea-ice dataset and source code are publicly in https: //github.com/Alien9427/XAI4SAR-PGIL.
# 6. Acknowledgment
This work was supported in part by the National Natural Science Foundation of China under Grant 62101459,
U20B2068, and in part by the China Postdoctoral Science Foundation under Grant BX2021248, and the Fundamental Research Funds for the Central Universities under Grant G2021KY05104.
# References
Page 14 of 15
Zhao, J., Zhang, Z., Yao, W., Datcu, M., Xiong, H., Yu, W., 2020. OpenSARUrban: A Sentinel-1 SAR Image Dataset for Urban Interpretation. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 13, 187–203. doi:10.1109/JSTARS.2019. 2954850.
