# REP WORKS IN SPEAKER VERIFICATION
# Yufeng Ma1, Miao Zhao1, Yiwei Ding1,2, Yu Zheng1, Min Liu1, Minqiang Xu1*
1 SpeakIn Technologies Co. Ltd. 2 Fudan University
ABSTRACT
Multi-branch convolutional neural network architecture has raised lots of attention in speaker verification since the aggregation of multiple parallel branches can significantly improve performance. However, this design is not efficient enough during the inference time due to the increase of model parameters and extra operations. In this paper, we present a new multi-branch network architecture RepSPKNet that uses a re-parameterization technique. With this technique, our backbone model contains an efficient VGG-like inference state while its training state is a complicated multi-branch structure. We first introduce the specific structure of RepVGG into speaker verification and propose several variants of this structure. The performance is evaluated on VoxCeleb-based test sets. We demonstrate that both the branch diversity and the branch capacity play important roles in RepSPKNet designing. Our RepSPKNet achieves state-ofthe-art performance with a 1.5982% EER and a 0.1374 minDCF on VoxCeleb1-H. Index Terms— speaker verification, speaker recognition, reparameterization
# 1. INTRODUCTION
Speaker verification aims to verify a speaker’s identity given an audio segment. In recent years, deep neural networks (DNNs) has improved the performance of speaker verification systems which outperform the traditional i-vector system [1]. Most DNN-based systems, such as x-vector [2], r-vector [3], and the recently proposed ECAPA-TDNN [4, 5], consist of three parts: (1) a network backbone to extract frame-level speaker representations, (2) a pooling layer to aggregate the frame-level information, and (3) a loss function. This paper focuses on the backbone architecture, which is the core part of the DNN models. The backbone architecture can be a 1-dimensional convolutional neural network (TDNN) [2], a 2-dimensional convolutional neural network (CNN) [3, 6], a recurrent neural network (RNN), and even a hybrid architecture that combines TDNN, CNN, RNN, and Transformer-like structures [7]. Several modifications of the backbone architecture are made to improve the performance. These modifications include adding a channel attention [8], transforming the custom convolution into a multi-scale convolution [9, 10], and aggregating multi-layer or multi-stage features [4, 11]. However, all these methods above only focus on the improvement of singlebranch structures and neglect a multi-branch way of designing neural networks. Adding parallel branches [12, 13, 14] can significantly enlarge the model capacity and enrich the feature space, which results in better model performance. Yu et al. [15] proposed a multi branch version of densely connected TDNN structure with a
selective kernel (D-TDNN-SS) and this model achieved competitive performance in speaker verification. Though the complicated multi-branch structure has proved its power, more parameters and connections usually lead to a slow inference speed. Recent researches [16, 17, 18] proposed a new technique called re-parameterization to solve the increasing inference cost. The main idea of this technique is to design a training time multi-branch structure which can be transformed to a single path with only one custom convolution during the inference time. This technique decouples training time and inference time architecture and ensures that the output remains the same. Inspired by this re-parameterization, we [19] first introduced the original RepVGG model into the speaker verification and obtained first place in both Track 1 and Track 2 of VoxSRC2021. The spectrogram whose shape is C × F × T is different from the image data that commonly serve as data input in computer vision tasks. C here means the feature maps (channel), F means the frequency features and T means the time axis. We presented that the design of the multi-branch structure can be heuristic due to this difference of input data. The structure is task-specific and various reparameterizable branches achieve different performances. To fully investigate how this re-parameterization works in speaker verification, we followed the work in [17, 19] and proposed several variants of the original RepVGG block. We evaluated the performance of these systems on Voxceleb1-O, VoxCeleb1-E, and Voxceleb1-H [20, 21]. Based on these results, we at the first time proposed a new re-parameterizable structure named RepSPKNet and demonstrated the importance of branch diversity and branch capacity in designing multi-branch structures. Our RepSPKNet can be transformed to a stack of simple K × K convolutions and ReLU layers during the inference time which results in a fast inference speed and competitive performance. The proposed RepSPKNet model achieved a 1.5982% EER and a 0.1374 minDCF on VoxCeleb1-H. The paper is organized as follows: Section 2 reviews the prior works related to re-parameterization. Section 3 presents our baseline system with a RepVGG backbone and other variants. In section 4, we discuss the experiment details and analyze the result. The analysis finally derives our carefully designed RepSPKNet. Section 5 concludes this paper.
# 2. RE-PARAMETERIZATION
Structural re-parameterization is used to avoid the extra parallel branch parameters and the slow inference speed via converting a multi-branch structure into a single path. ACNet [16] proposed a 1×3 kernel convolution with batch normalization (1×3 CONV-BN) and a 3 × 1 CONV-BN to strengthen the original 3 × 3 CONV-BN. RepVGG added a 1 × 1 CONV-BN and an identity batch normalization layer (ID-BN) in parallel. Furthermore, DBB [18] proposed a diverse branch block and gave more general transformations. It
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/293b/293b1911-ae31-4ffc-958d-8496cf8deb39.png" style="width: 50%;"></div>
Fig. 1. The baseline system. The a, b here denote the layer width parameter. Initial stride means the stride of first block of each stage. The input format is C × F × T.
is worth mentioning that the combinations of these transformations also satisfy the request for re-parameterization. Here we list the general transformations as follows: • CONV-BN fusion Batch normalization can be fused into its preceding convolution. Given a kernel weight F, the fused parameters can be formulated as:
(1)
where i denotes the i-th channel, and γ, µ, σ, β denote the scaling factor, mean, variance and bias of the BN layer. • Parallel conv addition Convolutions with different kernel size in different branches can be fused into one convolution by zero-padding small kernels and applying a simple elementwise addition. • Sequential convolutions fusion A sequence of 1×1 CONVBN and k × k CONV-BN can be fused into a k × k CONV whose parameters can be formulated as:
(3)
� � � where F (1), b(1) and F (2), b(2) denote the weights and bias of convolutions, and TRANS denotes transpose operation. • Average pooling transformation A kernel k average pooling can be transformed to a k × k convolution. The parameter is
(4)
where Ik×k is an identity matrix.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f718/f7187bc6-bbf4-4982-8fba-cd3f83c8b37a.png" style="width: 50%;"></div>
Fig. 2. Architecture of RepVGG block. Here (a) is the training time state. (b) demonstrates the process of CONV-BN fusion. (c) is the inference time state. ⊕denotes element-wise addition. A ReLU is added after branch addition.
# 3. OUR PROPOSED REPSPKNET SYSTEM
The RepVGG based speaker verification system has already shown its competitive performance [19]. This section describes our baseline system and our proposed variants.
# 3.1. Baseline system
Here we present our baseline system that consists of a RepVGG-A backbone, a statistical pooling [22], a 512-dimensional embedding layer and an additive margin softmax (AM-Softmax) loss function [23, 24]. The detailed topology is shown in Fig. 1. As Fig. 2 shows, the basic RepVGG block consists of three parallel branches: (1) a 3 × 3 CONV-BN, (2) a 1 × 1 CONV-BN, and (3) an ID-BN. The ID-BN branch exists only when the input channel equals the output channel. According to the transformations in Section 2, it is easy to verify that these three branches can be merged into one 3 × 3 convolution during the inference time. The RepVGG-A backbone consists of a stem layer and four stages. These stages contain 2, 4, 14, and 1 RepVGG blocks respectively and the stem layer is also a RepVGG block. The complexity of our backbone depends on the layer width parameters (a, b). For RepVGG-A0, we set a = 0.75, and b = 2.5. For RepVGG-A1, a = 1.0, and b = 2.5. For RepVGG-A2, a = 1.5, and b = 2.75. We slightly change the original stride setting [17] to make this backbone fit the speaker verification task. Both the first stage and the stem layer have a stride of 1. The other stages have a stride of 2. The format of input is 1×F ×T. The output of the backbone has a shape as 512b× F 8 × T 8 and is reshaped to (512b × F 8 ) × T 8 . The whole backbone can be transformed to a stack of 3×3 convolutions and ReLU layers during the inference time. A statistical pooling layer is applied to aggregate the speaker information. We calculate the mean and the standard deviation of the backbone output along the time axis. The mean and standard deviation are concatenated and then compressed to a 512-dimensional vector which serves as the speaker embedding. The AM-Sofmtax is used to classify speakers which can be formulated as:
(5)
where N denotes the number of samples, C denotes the number of speakers, s denotes the scaling factor, m denotes the margin penalty and cosθi,j denotes the angle between weight vector wj and the i-th sample.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a6de/a6de6e4e-e0b1-4a8e-97ab-58a7f7373554.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 3. Variants of the original RepVGG basic block. For convenience, the 3 × 3 CONV-BN and the ID-BN not depicted.</div>
As we introduced in Section 1, the original architecture of this RepVGG block is designed for computer vision tasks. Though this architecture has proved its superiority, it is task-specific and may not be the optimal structure in speaker verification. The 3×3 CONV-BN is the main branch of this architecture, while the ID-BN serves as a residual connection to avoid the gradient vanishing problem. To investigate the re-parameterization in speaker verification, we fixed the 3×3 CONV-BN and ID-BN branches and proposed several variants to replace the 1 × 1 CONV-BN. As Fig. 3 demonstrates, structure (a) is a duplicate 3 × 3 CONV-BN. Structure (b) and (c) are borrowed from ACNet. Structure (d) and (e) are borrowed from DBB. Structure (f) consists of a 3 × 3 convolution with a dilation of 2 and a batch normalization layer. The original RepVGG blocks are replaced with these variants of the original block to form new speaker verification models. According to the transformations in Section 2, all these variants (except (f)) can be transformed to a 3 × 3 convolution which means the inference bodies of these new models remain the same when compared to the original RepVGG inference time state.
# 4. EXPERIMENTS AND RESULTS
# 4.1. Dataset and features
All our models adopted the VoxCeleb2 development set [21] as our training set. This dataset contains 1,092,009 utterances and 5,994 speakers in total. Our data augmentation consisted of two parts: (1) A 3-fold speed augmentation [19, 25] was implemented at first to generate extra twice speakers based on the SoX speed function. (2) We followed the data augmentation method provided by the Kaldi VoxCeleb recipe. The RIRs [26] and MUSAN [27] dataset was used. After the augmentation process, 16,380,135 utterances from 17,982 speakers were generated. We extracted 81-dimensional log Mel filter bank energies based on Kaldi without voice activity detection (VAD). The window size is 25 ms, and the frame-shift is 10 ms. All the features were cepstral mean normalized.
# 4.2. Experiment setup
200 frames of each sample in one batch were randomly selected. The SGD optimizer with a momentum of 0.9 and a weight decay of 1e-3 was used. We used 8 GPUs with mini-batch as 1,024 and an initial learning rate of 0.08. We adopted ReduceLROnPlateau scheduler and the minimum learning rate is 1e-6. The margin of the AMSoftmax loss is set to 0.2 and the scale is 36. All our systems were evaluated on VoxCeleb1-O, VoxCeleb1-E, and VoxCeleb1-H. Trials were scored by cosine similarity of the 512-dimensional embeddings
Table 1. Ablation study on our baseline model RepVGG-A0. For convenience, we omitted the % sign of EER and used Var to denote the variant using the structures we proposed. Var f cannot be transformed to a custom 3 × 3 convolution.
formed to a custom 3 × 3 convolution.
Models
VoxCeleb1-O
VoxCeleb1-E
VoxCeleb1-H
EER
DCF0.01
EER
DCF0.01
EER
DCF0.01
A0
1.4310
0.1219
1.2900
0.1207
2.1700
0.1864
Var a
1.3940
0.1185
1.3100
0.1256
2.2100
0.1913
Var b
1.3890
0.1140
1.2980
0.1236
2.1802
0.1889
Var c
1.3091
0.1169
1.3110
0.1265
2.2213
0.1937
Var d
1.2671
0.1039
1.2602
0.1181
2.1126
0.1850
Var e
1.4263
0.1320
1.3164
0.1266
2.2201
0.1940
Var f
1.0821
0.1006
1.1204
0.1067
1.9342
0.1665
and no score normalization was implemented. The criterion is equal error rate (EER) and minimum decision cost function (DCF) where CF A = 1, CM = 1, and ptarget = 0.01.
# 4.3. Results and analysis
4.3.1. Ablation study of base model
As we mentioned in Section 3.2, the RepVGG block is task-specific. To compare our proposed variants with the original RepVGG structure, we selected RepVGG-A0 as our base model since it has a much faster training speed when compared with RepVGG-A1 and RepVGG-A2. To find out the most suitable re-parameterizable structure in speaker verification, we trained all the variants mentioned above. All the performances were presented in Table 1. As for Var a, it is rather intriguing that replacing the original 1×1 CONV-BN with an extra 3 × 3 CONV-BN bring performance decay on some complex test sets like VoxCeleb1-E and VoxCeleb1-H even the training time state has more parameters (larger branch capacity). We believed that this extra 3 × 3 CONV-BN structure tended to learn a representation that was similar to the main 3 × 3 CONV-BN. The lack of feature diversity caused the performance decay. Furthermore, Var d performed the best among all models (Var f not included) on all test sets. This structure added a 3 × 3 CONV-BN after the original 1 × 1 CONV-BN. On the contrary, Var e which consists of a 1 × 1 CONVBN and an average pooling performed the worst. Both these two structures had operators that control the balance between the branch diversity and branch capacity.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ec77/ec771da1-6c62-47e6-ae54-0eda56216d01.png" style="width: 50%;"></div>
Fig. 4. Branch similarity of each model. The layer number starts from the initial block to the final one.
Table 2. Performances of the RepSPKNet. RSBA denotes RepSPK-A Block and RSBB denotes RepSPK-B block. - 3 × 3 + 5 × 5 means replacing the main 3 × 3 CONV-BN branch with a 5 × 5 CONV-BN. The ECAPA denotes the ECAPA-TDNN(C=2048), and its results are referred from [5]. The ResNet34 model denotes our implementation of SOTA ResNet system. No score normalization is adopted except that the ECAPA system reports performances using adaptive s-norm.
the ECAPA system reports performances using adaptive s-norm.
Models
VoxCeleb1-O
VoxCeleb1-E
VoxCeleb1-H
EER
DCF0.01
EER
DCF0.01
EER
DCF0.01
ECAPA
0.8600
0.0960
1.0800
0.1223
2.0100
0.2004
ResNet34
1.0498
0.1045
1.0587
0.1008
1.8456
0.1619
A0
1.4310
0.1219
1.2900
0.1207
2.1700
0.1864
RSBA-A0
1.2671
0.1039
1.2602
0.1181
2.1126
0.1850
RSBB-A0
1.0821
0.1006
1.1204
0.1067
1.9342
0.1665
- 3 × 3 + 5 × 5
1.1771
0.0982
1.1041
0.1081
1.8960
0.1688
A1
1.2141
0.0913
1.1593
0.1054
1.9347
0.1655
RSBA-A1
1.1503
0.0872
1.1295
0.1013
1.8902
0.1605
RSBB-A1
0.9650
0.0795
1.0359
0.0936
1.7602
0.1512
A2
0.9546
0.0831
1.0143
0.0926
1.7149
0.1465
RSBA-A2
0.9122
0.0801
0.9939
0.0916
1.6809
0.1431
RSBB-A2
0.8430
0.0775
0.9637
0.0907
1.5982
0.1374
To verify that the multi-branch structure’s performance depends on the trade-off between branch diversity and branch capacity, we designed a branch Var f as shown in Fig. 3. This structure is a 3 × 3 CONV-BN with dilation as 2. This dilated convolution ensures diversity by constraining a different input and receptive field from the custom convolution of the main branch, and it also ensures capacity by increasing the parameters. We used the cosine similarity between the outputs of the main branch and our proposed branch to represent the branch similarity. The branch similarity of each layer was presented in Fig. 4. Var a, obviously had the highest branch similarities (around 0.9) as we speculated. Var e, on the contrary, had the lowest similarities (around 0.2). Both the two structures showed performance decay. The other two variants had similarities around 0.5 and outperformed the base model. Moreover, Var f with similarities closer to 0.5 achieved a relative 10.9% EER and a relative 10.5% minDCF improvement compared to the base RepVGG-A0 model.
4.3.2. RepSPKNet architecture
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4432/4432271c-0541-42e4-a1bd-d8ef64f11cb7.png" style="width: 50%;"></div>
Fig. 5. RepSPK block (RSB). (a) denotes the RSBA block and (b) denotes RSBB block. The ID-BN exists when input channel equals output channel.
It is easy to prove that a 3 × 3 convolution with a dilation of 2 can be transformed to a 5 × 5 convolution. According to the trans-
formations provided by Section.2, this Var f structure can also be re-parameterized to a single 5 × 5 custom convolution. Based on the results, we proposed the final architecture of the RepSPKNet. As depicted in Fig. 5, two blocks were presented. The RSBA block was composed of a 3 × 3 CONV-BN, a 1 × 1 CONV-BN followed by a 3 × 3 CONV-BN, and an ID-BN. The RSBB block was composed of a 3 × 3 CONV-BN, a 3 × 3 CONV-BN with a dilation of 2, and an ID-BN. To verify the stability and transferability of our proposed architecture, we compared our RepSPKNet with the original RepVGG-A1 and RepVGG-A2. Here we only replaced the RepVGG block with the RepSPK block (RSB). The model consisted of RSBA blocks was called RepSPKNet-A and the other consisted of RSBB blocks was called RepSPKNet-B. We also conducted an ablation study of the RSBB structure by replacing the 3 × 3 CONVBN main branch with a 5 × 5 CONV-BN. The results were presented in Table 2. The RepSPKNet-A and RepSPKNet-B both outperformed their corresponding base model. Compared to the performance of RepVGG-A2 on VoxCeleb1-H, the RSBA-A2 achieved relative improvements of 2.0% in EER and 2.3% in minDCF. Moreover, the RSBB-A2 achieved relative improvements of 6.8% in EER and 6.2% in minDCF. The result demonstrated that our RepSPKNets can achieve SOTA performance in speaker verification.
# 5. CONCLUSION
In this paper, we proposed two blocks as Fig.5 demonstrates. The RepSPKNet-A is composed of the RSBA block while the RepSPKNet-B is composed of the RSBB block. With the structral re-parameterization method, the RepSPKNet-A can be transformed to a stack of 3 × 3 convolution and ReLU and the RepSPKNet-B can be transformed to a stack of 5 × 5 convolution and ReLU. Ablation studies on various variants indicated that the performance of multi-branch structure depended on the branch diversity and branch capacity, which was a heuristic principle for designing multibranch models. Our proposed RepSPKNet outperformed the original RepVGG and achieved SOTA performance in speaker verification.
