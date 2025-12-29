# CADICA: a new dataset for coronary artery disease detection by using invasive coronary angiography
Ariadna Jim´enez-Partinen1,3, Miguel A. Molina-Cabello1,3, Karl Thurnhofer-Hemsi1,3,4, Esteban J. Palomo1,3, Jorge Rodr´ıguez-Capit´an2,3,4, Ana I. Molina-Ramos2,3,4, and Manuel Jim´enez-Navarro2,3,4,5
1 Department of Computer Languages and Computer Science. University of M´alaga, Bulevar Louis Pasteur, 35, M´alaga, Spain, 29071 2 Cardiology Deparment, Hospital Universitario Virgen de la Victoria, M´alaga, 29010, Spain 3 Instituto de Investigaci´on Biom´edica de M´alaga y Plataforma en Nanomedicina-IBIMA Plataforma BIONAND, C/ Severo Ochoa, 35, M´alaga TechPark, Campanillas, 29590, M´alaga, Spain 4 Centro de Investigaci´on Biom´edica en Red de Enfermedades Cardiovasculares (CIBERCV), Instituto de Salud Carlos III (ISCIII), Avenida Monforte de Lemos, 3-5. Pabell´on 11. Planta 0, 28029, Madrid, Spain 5 Facultad de Medicina, University of M´alaga, Bulevar Louis Pasteur, 37, 29071, M´alaga, Spain
1 Department of Computer Languages and Computer Science. University of M´alaga, Bulevar Louis Pasteur, 35, M´alaga, Spain, 29071 2 Cardiology Deparment, Hospital Universitario Virgen de la Victoria, M´alaga, 29010, Spain 3 Instituto de Investigaci´on Biom´edica de M´alaga y Plataforma en Nanomedicina-IBIMA Plataforma BIONAND, C/ Severo Ochoa, 35, M´alaga TechPark, Campanillas, 29590, M´alaga, Spain 4 Centro de Investigaci´on Biom´edica en Red de Enfermedades Cardiovasculares (CIBERCV), Instituto de Salud Carlos III (ISCIII), Avenida Monforte de Lemos, 3-5. Pabell´on 11. Planta 0, 28029, Madrid, Spain 5 Facultad de Medicina, University of M´alaga, Bulevar Louis Pasteur, 37, 29071, M´alaga, Spain
Abstract. Coronary artery disease (CAD) remains the leading cause of death globally and invasive coronary angiography (ICA) is considered the gold standard of anatomical imaging evaluation when CAD is suspected. However, risk evaluation based on ICA has several limitations, such as visual assessment of stenosis severity, which has significant interobserver variability. This motivates to development of a lesion classification system that can support specialists in their clinical procedures. Although deep learning classification methods are well-developed in other areas of medical imaging, ICA image classification is still at an early stage. One of the most important reasons is the lack of available and high-quality open-access datasets. In this paper, we reported a new annotated ICA images dataset, CADICA, to provide the research community with a comprehensive and rigorous dataset of coronary angiography consisting of a set of acquired patient videos and associated disease-related metadata. This dataset can be used by clinicians to train their skills in angiographic assessment of CAD severity, by computer scientists to create computer-aided diagnostic systems to help in such assessment, and to validate existing methods for CAD detection. In addition, baseline classification methods are proposed and analyzed, validating the functionality of CADICA with deep learning-based methods and giving the scientific community a starting point to improve CAD detection.
Keywords: Invasive coronary angiography dataset · cardiovascular artery disease · classification · deep learning · medical images.
# 1 Introduction
Coronary artery disease (CAD) remains the leading cause of death globally [1, 2]. Clinical presentations of CAD are currently categorized as either acute coronary syndromes or chronic coronary syndromes, and assessing patients with suspected CAD is a significant component of healthcare costs [3]. Invasive coronary angiography (ICA) is considered the gold standard of anatomical imaging evaluation when CAD is suspected [4, 5]. ICA acquisition is based on introducing radiocontrast through a catheter inserted by a percutaneous incision in the femoral or brachial artery, situated in the groin and the arm, respectively. The radiocontrast agent enhances the visibility of coronary arteries, with cardiac angiography equipment, X-ray-based, the state of the arteries is shown, allowing the clinicians to evaluate it and conclude if there is a luminal obstruction. The presence of obstructive CAD in ICA, usually defined as a lesion greater than 70 percent, has been recognized as an unequivocal sign of a bad cardiovascular prognosis. In contrast, it was initially proposed that non-obstructive CAD (usually defined as a lesion less than 70 percent) could constitute a condition related to a good cardiovascular prognosis [6], but subsequent evidence has increasingly shown that it confers an adverse prognosis when compared to the prognosis in the absence of CAD [7, 8, 9]. Consequently, it is currently accepted that cardiovascular risk increases when the degree of stenosis increases. However, risk evaluation based on ICA has several limitations. There is enough evidence indicating that the visual assessment of stenosis severity alone has significant interobserver variability, so this visual assessment alone does not provide us with enough information upon which to base decisions about revascularization in many patients [10]. In addition to this, angiographic assessment of CAD severity is limited in providing consistent information regarding the physiological significance of coronary lesions. Angiography is especially limited in coronary stenoses of intermediate severity (40–70 percent obstruction), where it predicts functional significance in less than 50 percent of lesions [11]. Visual assessment of coronary angiography fails to adequately determine lesion significance because lumen stenosis is only one variable out of many that influence the flow limitation of coronary lesions [12]. Lesion length, collateral flow, and the amount and health of the myocardial bed supplied are other essential factors that are not readily assessed by coronary angiography [13]. In order to overcome the aforementioned limitations, current guidelines recommend the routine assessment of vessel physiology in the form of indices derived from invasive pressure wire, such as fractional flow reserve and the instantaneous wave-free ratio [4, 5]. Despite these recommendations, the implantation of these functional tests in clinical practice has been especially low [14]. Many medical image datasets have been provided to the research community with the aim of developing an algorithm that can serve as a computer-aided diagnosis system [15, 16, 17, 18]. However, there is a lack of available and highquality open-access datasets regarding ICA images because most related studies use private image sets [19]. Some are provided by an associated medical center and used for image segmentation tasks [20, 21], while others are focused on
detection and classification [22, 23]. None of them provides access to other researchers to their data, which is necessary to achieve advances in this field. The main contributions of this work can be listed as follows:
– To provide the research community with a comprehensive and rigorous scientific coronary angiography dataset formed by a set of videos acquired from patients and metadata related to diseases associated with them. This dataset may serve medical doctors to train their skills in angiographic assessment of CAD severity, and computer scientists to create computer-aided diagnosis systems to help with that kind of evaluation and to validate and improve existing methods for CAD detection by training them on more data. – A set of resources for testing algorithms. Additionally, an exhaustive revision and expansion process of these tools will be carried out regularly. – To provide a study of the performance of known architectures using the dataset with the aim of classifying ICA images according to the presence of lesions. – To help the community to identify other related challenges to provide a focus for future research.
The rest of the paper is structured as follows: Section 2 describes the recent state-of-art works related to ICA. In Section 3, the most important details about the creation and organization of the CADICA dataset are given. The experimental results are shown in Section 4. A discussion is provided in Section 5. Finally, Section 6 is devoted to conclusions.
# 2 Related works
Deep learning has been thoroughly used for both classification and segmentation tasks in medical imaging, including in the area of cardiology, where the most common imaging modalities are MRI (magnetic resonance imaging), and X-raybased, such as CCTA (Coronary Computed Tomography Angiography) and ICA (Invasive Coronary Angiography) [24]. Specifically for ICA images, we found the work of Auet al. [25], which uses ICA images of the right coronary artery to detect and classify coronary stenosis. They implemented a complex method based on three phases with different networks to do it. First, they used a detection network, YOLONet, to localize the patch where is the lesion. Straightaway, the lesion is segmented using the U-Net model, and finally, it is classified by a small CNN as stenosis if the narrowing is higher than 70%. Nasr-Esfahani et al. [20] used patches from 44 coronary angiographies to set a system, composed of two CNNs based on learned kernels, which returns a segmentation probability map of the ICA images. Wu et al. [26] proposed a method to localize stenosis lesions in ICA image sequences based on three stages. For this study, 148 sequences of ICA images were employed. Firstly, the most appropriate frames from the complete sequence are selected using the U-Net architecture. Next, a deconvolutional single-shot
multibox detector localizes the possible boxes that contain stenosis, and finally, they designed a temporal module to determine which boxes are actually stenosis, considering the temporal sequence of the boxes selected. Zhang et al. [27] used the U-net architecture to implement the C-Unet model that is a deep learning-based solution to automatically extract the centerlines of blood vessels from ICA images, achieving a precision higher than 80%. Cong et al. [23] implemented a system completely automatically for the classification and location of ICA images, which had been clustered into the left coronary artery and right coronary artery, and the lesions were into categories depending on their grade. To start, an inception-v3 model classified the images according to the projection to which they belonged. Then, the ideal candidate frames were selected by a fusion between inception-v3 and LSTM (long-shortterm memory), these candidates were classified into predefined categories by another inception-v3 model adapted for it, and to conclude, they employed a class activation map to identify the regions in ICA images where the lesion could be located. Zhou et al. [22] proposed a method to classify ICA images with stenosis. To carry out the study, they employed 8731 right coronary artery images, and the procedure was formed by three stages. Firstly, by ResNet-18 structure was carried out to extract keyframes, which presents enough contrast and clarity, from video sequences, there were 6533 non-key frames and 2198 keyframes Secondly, a vessel segmentation was implemented by a U-Net model to get their masks. And thirdly, lesions were measured by the skeletonization of the vessels in masks. Moon et al. [28] classified the ICA images into normal and abnormal arteries if the narrowing is lower or higher than 50%, respectively. As in the previous work reported, in this study, the first phase was based on extracting key frames from sequences of ICA videos, 542 videos were used, and then, an inceptionv3 architecture was fed with these keyframes to can classify them into normal or abnormal. To finalize, stenosis lesions were visually localized using class activation mapping. These reported works have some aspects in common with the study that we had been carrying out, but most of them are focused on implementing segmentation or using patches to classify the images, instead of complete images. Also, some of them excluded images with more than one lesion or only classify obstructive lesions, they utilize smaller datasets, most of them are private datasets, the detail of the artery and projections used is omitted or a different annotation is implemented.
# 3 CADICA dataset
Next, the most important details about the creation and organization of the CADICA dataset are presented.
# .1 Patient Selection
The dataset proposed in this work consists of 668 invasive coronary angiography videos from 42 patients, acquired at Hospital Universitario Virgen de la Victoria, M´alaga, Spain. They have been included within the regulation set by the local ethical committee of the hospital and patient consent was waived, because this is a retrospective study with anonymized data. Prerequisites and data selection have not been performed in order to impose clinical fidelity. Therefore, a wide variability of cases and acquisition configurations were implied, while some cases were laborious to tag. This way, the dataset exhibits a high variety of pathological cases and image quality. Table 1 summarizes the baseline and demographic characteristics of patients included in the dataset.
Table 1. Demographic and baseline characteristics of the patients. Data are given as % or as median (interquartile range)
Age (years)
71.5 (58.25-78)
Sex (female-male)
47.6% - 52.4%
Diabetes mellitus
40.5%
Dyslipidemia
40.5%
Smoker
45.2%
High blood pressure
61.9%
Kidney failure
14.3%
Heart failure
14.3%
Atrial fibrillation
4.8%
Left ventricular ejection fraction
Normal (ejection fraction >55%)
68.2%
Mild dysfunction (ejection fraction 45%-55%)
9.8%
Moderate dysfunction (ejection fraction 45%-35%)
0%
Severe dysfunction (ejection fraction <35%)
22%
Clinical indication for angiography
Chronic coronary syndrome
4.9%
Non-ST segment elevation acute coronary syndrome
65.9%
ST segment elevation acute coronary syndrome
29.3%
Number of vessels affected
0
23.8%
1
50%
2
14.3%
3
11.9%
Maximum degree of the coronary artery involvement
<20%
14.3%
20-50%
66.7%
>70%
19%
# 3.2 Acquisition Protocol
The invasive coronary angiography videos were acquired as Digital Imaging and Communication in Medicine (DICOM) files recorded at 10 frames per second and with different duration (4-8 seconds) depending on the projection used, but they were converted to PNG images for effortless management. The frame size of each video is 512 × 512 pixels, while the length of the videos varies from 1 to 151 frames. The cardiac angiography equipment used was Artis Zee (Siemens AG, Muenchen, Germany). The dose of radiation administered in each projection ranges between 5-50 mGy. The protocol normally used in each angiography included five projections for the left coronary artery (LCA), such as right anterior oblique (RAO) and left anterior oblique (LAO), both with cranial and caudal angulation, with some additional projections in case of diagnostic difficulties. The projections used for the right coronary artery (RCA) are LAO and RAO, with cranial and caudal angulation. Fig. 1 and 2 show examples of projections for the left coronary artery and the right coronary artery, respectively.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2484/24849017-5336-4040-904a-78cb22461b6b.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 1. Examples for left coronary artery (LCA).</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9c39/9c39bc04-fb26-4629-a869-05d3db124c72.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 2. Examples of projections for right coronary artery (RCA)</div>
# 3.3 Label Protocol
A team of cardiologists was involved in the annotation of the dataset, assisted by computer scientists. For each frame, those regions of interest are delimited by a bounding box and classified into categories. This way, for each region of interest, is provided the location of the top left corner of its bounding box, its width, its height, and the label of that region of interest. The possible categories are itemized in Table 2.
<div style="text-align: center;">Table 2. Categories into which lesions have been divided.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1884/18840356-f939-4c71-ab13-35d7209a1709.png" style="width: 50%;"></div>
Label Lesion range
p0 20
<20%
p20 50
[20%, 49%]
p50 70
[50%, 69%]
p70 90
[70%, 89%]
p90 98
[90%, 98%]
p99
99%
p100
100%
For each video, a selection of keyframes is carried out. This selection contains the list of frames that exhibit a contrast with enough appearance in order to classify the patient correctly. Videos are organized by patients, where a certain number of videos have been collected for each patient. According to their coronary artery stenosis percentage, patients are grouped into three different categories: < 20% (mild), 20 - 50% (moderate), and > 70% (severe). Lesions of 100% imply a total occlusion of the vessel, while a 99% lesion presents a gap where the radiocontrast is imperceptible, but the continuation of the vessel is visible. Those lesions that had a narrowing between 50 to 70 percent are classified as obstructive in some studies [29] and non-obstructive in others [30], while lesions with a higher narrowing than 70 percent in the previous bibliography showed consensus that they should be taken as obstructive and classify as non-obstructive the lesions that are solidly taken as non-obstructive (20–50 per-cent). Fig. 3 exhibits some sample images from different patients according to their classification into these categories.
# 3.4 Dataset Organization
Video Selection The presented dataset provides a total of 668 ICA videos. However, in some frames of these videos where CAD is present, the lesion is indiscernible, being difficult to use them for diagnosis. In order to obtain the best videos for CAD classification, a selection of 382 videos was performed. These videos have been chosen by the medical team, where CAD can be visually classified correctly. Thus, videos in which radiocontrast does not perfuse have not been selected for the classification task. The specifications of CADICA dataset
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d241/d2415030-269f-47fb-a6c2-fd4a00d1a8a9.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 3. Samples of the 3 categories in which lesions are classified and delimited by a bounding box annotated.</div>
are reported in Table 3, where the number of patients, videos, images from selected videos and labels from “lesion” images are itemized. Please note that the number of labels is higher than the number of “lesion” images since it can be more than one lesion in an image.
Metadata The dataset also provides additional clinical data associated with each patient, such as if the patient suffers from diabetes, dyslipidemia, smoking, hypertension, another comorbidity (such as chronic obstructive pulmonary disease), renal insufficiency, heart failure, atrial fibrillation, or left ventricular ejection fraction. Other information such as age, gender, height, weight, and later event (such as non-cardiac death or heart attack) is also reported.
Structure The CADICA dataset becomes a directory that contains the metadata.xlsx file, which is the file where the clinical data is located, as well as two main folders that differentiate the videos selected by the medical team for each patient: nonselectedVideos and selectedVideos. Inside each folder, there are several sub-directories with the naming convention pX where X is the ID of each patient, and vY, where Y is the ID of the video of that patient. The folder pX contains the following information:
– vY : several sub-directories with the videos selected for that patient.
<div style="text-align: center;">(c) Severe</div>
<div style="text-align: center;">Table 3. Specifications of CADICA dataset: number of patients, videos, images from selected videos and labels from “lesion” images.</div>
Table 3. Specifications of CADICA dataset: number of patients, videos, images from elected videos and labels from “lesion” images.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e19e/e19ee863-0060-42b6-9952-98e0fecbed35.png" style="width: 50%;"></div>
<div style="text-align: center;">– lesionVideos.txt: contains the IDs of the selected videos where appears at least one lesion which is labeled. – nonlesionVideos.txt: contains the IDs of the selected videos where there are no visible lesions.</div>
– lesionVideos.txt: contains the IDs of the selected videos where appears at least one lesion which is labeled. – nonlesionVideos.txt: contains the IDs of the selected videos where there are no visible lesions.
# The folder vY contains the following information:
– input: a sub-directory containing a separate PNG file for each frame of the video. – pX vY selectedFrames.txt: contains the IDs of the keyframes for the medica team, for all the selected videos. – groundtruth: a sub-directory available only if there are lesions in that selected video.
The folder groundtruth contains the following information:
– pX vY 000ZZ.txt: contains the bounding boxes and their category in each row. There are such files as frames in pX vY selectedFrames.txt. Bounding boxes are specified in the format [x, y, w, h], where (x, y) are the pixel coordinates of the top left corner, w is the width and h is the height of the bounding box. – pX vY groundTruthTable.mat: contains a table with the ground truth information of that video.
# 4 Experiments
Given the provided dataset in this work, we did an exhaustive performance comparison to classify ICA images according to the presence of lesions, being a binary problem, where the method classifies between “non-lesion” or “lesion” (images with any label from Table 2).
# 4.1 Evaluation metrics
In order to measure the performance of a method that classifies coronary angiography images according to their coronary artery stenosis percentage, several well-known metrics have been proposed. Let us consider the true positives or number of hits (TP), true negatives or correct rejections (TN), false negatives or misses (FN), and false positives or false alarms (FP). The selected metrics and their definitions are as follows:
The most representative measures are the Accuracy (Acc), the F-measure (Fm, also known as F1 score), and the Balanced Accuracy (Bal), which provide a good overall evaluation of the performance of a given method. All these measures represent the percentage of hits of the system by providing values in the interval [0, 1], where higher is better. Meanwhile, other measures are also implicitly considered such as the precision (PR), the recall (RC), and the specificity (SP). In order to analyze these metrics, FN must be considered against FP (lower is better), PR against RC (higher is better).
# 4.2 Methods
Convolutional Neural Networks In this study, different Convolutional Neural Networks (CNNs) are used to compare their performance to classify ICA images from CADICA into two classes, “lesion”, which means that appears at least one lesion, and “non-lesion”. CNN is a type of deep learning model incorporating at least one convolutional layer, whose purpose is to extract the features from the input image, triggering under a specific condition [31]. CNNs have become successful methods with great versatility of applications in several areas, including medical images, and to solve different problems, such as segmentation, localization, or classification. Also, CNNs are characterized by their transferability of knowledge by applying the transfer learning technique, which is based on employing classification models trained on large datasets, also named pre-trained networks, which are re-trained with a specific dataset to specialize them to the particular problem [32]. In this study no layer was frozen, so all weights were updated according to the input dataset information. Five known pre-trained CNN architectures are used in this study: The Residual Networks (ResNets) family [33] introduces the residual connection to the model, these shortcut connections allow skipping some layers in the
(1)
(2)
(3)
process. In particular, in this study ResNet-18 and ResNet-50 networks are used, which are characterized by being composed of 18 and 50 layers deep, respectively. MobileNet-V2 [34] is a mobile neural network optimized to considerably reduce the number of parameters, compared with other architectures, which decreases the computational load. The MobileNet architecture is based on depthwise separable convolutions, which are a combination of two layers. The first is depthwise convolution, which applies a single filter to the input without extracting features, and the second is named pointwise convolution, which creates a linear combination output with new features [35]. NasNet-Mobile is the smallest version of NasNet models. NasNet models are CNNs based on Neural Architecture Search (NAS), which consist of basic building blocks, called cells, optimized by reinforcement learning method [36]. DenseNet-201 is a deep network based on ensuring the maximum information flow between layers by dense blocks. Dense blocks are blocks of layers, where each layer is connected to all former layers, instead only to the previous one [37].
Data Preprocessing In CADICA there are 382 selected videos in total, of which two subgroups were done differentiating between views of the left (LCA) and right (RCA) coronary arteries. The subgroup of LCA views was composed of 216 videos, a total of 3,228 images, where 1,003 images were labeled as “non-lesion” and 2,225 were labeled as “lesion”. The subgroup of RCA views consisted of 118 videos, which is 2,077 images in total, whose labels were distributed as 617 images labeled as “nonlesion” and 1,460 labeled as “lesion”. The input image of the pre-trained architectures selected is an RGB image of size 224 × 224 pixels. Thus, the first processing applied to all images was to resize them and use the color preprocessing to ensure that images have the number of channels required, in this case, three channels. To study the binary classification problem “lesion”/“non-lesion”, both sets had been divided into training (80%) and test (20%) sets. This division was done by videos, which means that 80% of the “non-lesion” and “lesion” videos were used for training and the 20% remaining for testing. This way, frames of the same video of the train set are unavailable for the test set, because frames of a video are very similar between them. Both sets have unbalanced distributions, which can cause the model to specialize in the majority class, in this case, “lesion”, and be relatively inefficient at classifying the minority class, in this case, “non-lesion”. To solve this issue a data augmentation strategy had been applied to the training sets. This data augmentation was done by using different random basic operations of the original images, detailed as follows:
– Translations in the x and y axis of [-25,25] pixels randomly, Fig. 4(b). – Rotation using a random angle between [-25º,25º], Fig. 4(c). – Scaling of the images with a random scale factor in a range of [0.8,1.7], Fig 4(d).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/661e/661e442d-b606-44a2-810f-82094f9813cf.png" style="width: 50%;"></div>
<div style="text-align: center;">ig. 4. Examples of the modifications applied to the training sets to augm</div>
Modifications were applied to the training sets of both classes, “lesion” and “non-lesion” images, and to both subsets, LCA and RCA, equalizing and increasing them. Fig. 5 shows the original class distributions for LCA and RCA sets, and the final distributions obtained after data augmentation was implemented, obtaining 3640 and 2342 images of each class in the LCA set and in the RCA set, respectively.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9137/9137f6e3-21c3-44eb-990f-a57f383caf31.png" style="width: 50%;"></div>
Fig. 5. Class distribution for Left coronary Artery (LCA) and Right Coronary Artery (RCA) sets.
Experimental Setup Several parameters can be tuned for training convolutional neural networks. The main ones that we were focusing on are reported below:
<div style="text-align: center;">(d) Scale</div>
– Validation frequency: it is the number of iterations between evaluations of the training process. – Maximum number of epochs: indicates the maximum times that the full dataset is passed to the model to update its weights. – Optimizer or solver: is the algorithm applied to update the weights of the network to reduce the loss function. – Initial Learning Rate: establishes the rate that is going to use to start the learning procedure. – Batch size: specifies the number of samples, in this case, images, that are processed by the model in one iteration.
Due to many options for possible combinations of training parameters, we started studying the behavior of the performance, establishing some values to tune the training parameters. To evaluate the progress in the tuning of the parameters, we used the performance metrics reported in section 4.1, focusing on F-measure, Balanced Accuracy, and Accuracy, because together they provide a global view of the performance. The LCA set was employed for this process because it is the largest and most complex set. Finally, the parameters selected were used to evaluate the RCA set too. The first parameters that we set were validation frequency in 50 iterations to evaluate the training process and the maximum number of epochs established in 10 epochs because the increase of it had an unsubstantial improvement compared to time-consuming. For the optimizer, we compared different algorithms: Adam (adaptive moment estimation), SGDM (stochastic gradient descent with momentum), and RMSProp (root mean square propagation). Besides, we proposed several rates for the initial learning rate: 0.01, 0.001, 0.0001, and 0.00001. For the batch size, two values were selected, 16 and 64. In total there are 24 possible combinations, to compare them, we implemented the stratified K-fold cross-validation. This technique is adequate to evaluate the performance in a reliable way because the results are averages from different partitions of the input dataset, so results are independent of the partition employed to validate. The proposed models were implemented in MATLAB R2022b on a computer system with an Intel Core i9-10900X processor, 128 GB of RAM, and NVIDIA GeForce RTX 3080 Ti GPU card. Firstly, we divided the training set of LCA subset into 5 boxes with samples of both classes, 80% for training the models and 20% as the validation set. For each model, the 24 combinations established were executed, except DenseNet201, which was only trained with batch size 16 because of memory settings. The obtained results for test sets, with the different possible combinations for each architecture, are shown in Tables 4, 5, and 6, where F-measure, Balanced Accuracy and Accuracy values for each combination are reported. In these tables, the best results for each model and batch size are shown in bold. By observing these tables, we can see how in general terms is better to choose low learning rates and a batch size of 16 instead of 64. Moreover, SGDM and RMSProp solvers obtained better results than Adam, especially for batch sizes of
Table 4. F-measure results obtained on the test set for LCA images using 5-fold stratified cross-validation, five convolutional network architectures, and different values for batch size, initial learning rate, and optimizer. The best performances are shown in bold.
Model
Initial
Learning Rate
Batch Size 16
Batch Size 64
Adam
SGDM
RMSProp
Adam
SGDM
RMSProp
MobileNet-V2
0.01
0.758 ± 0.033 0.776 ± 0.028
0.755 ± 0.096
0.767 ± 0.036
0.806 ± 0.013
0.720 ± 0.052
0.001
0.798 ± 0.035 0.800 ± 0.025
0.791 ± 0.034
0.812 ± 0.024
0.786 ± 0.018
0.807 ± 0.024
0.0001
0.809 ± 0.030 0.794 ± 0.021
0.818 ± 0.015
0.806 ± 0.016
0.800 ± 0.010
0.808 ± 0.015
0.00001
0.810 ± 0.014 0.785 ± 0.010 0.820 ± 0.015 0.810 ± 0.009 0.766 ± 0.011
0.789 ± 0.012
ResNet-18
0.01
0.737 ± 0.054 0.739 ± 0.038
0.730 ± 0.094
0.747 ± 0.047
0.789 ± 0.011
0.731 ± 0.065
0.001
0.782 ± 0.020 0.798 ± 0.020
0.765 ± 0.025
0.784 ± 0.013 0.794 ± 0.016 0.762 ± 0.017
0.0001
0.784 ± 0.022 0.780 ± 0.022 0.800 ± 0.021
0.783 ± 0.029
0.788 ± 0.026
0.775 ± 0.014
0.00001
0.783 ± 0.017 0.800 ± 0.029
0.789 ± 0.019
0.772 ± 0.016
0.770 ± 0.016
0.782 ± 0.012
ResNet-50
0.01
0.655 ± 0.057 0.777 ± 0.027
0.380 ± 0.418
0.755 ± 0.006
0.793 ± 0.013
0.713 ± 0.097
0.001
0.773 ± 0.020 0.774 ± 0.012
0.775 ± 0.031
0.772 ± 0.042
0.779 ± 0.021
0.746 ± 0.055
0.0001
0.785 ± 0.023 0.784 ± 0.025 0.823 ± 0.017
0.806 ± 0.024
0.802 ± 0.016 0.818 ± 0.023
0.00001
0.797 ± 0.037 0.796 ± 0.026
0.808 ± 0.025
0.782 ± 0.040
0.783 ± 0.034
0.777 ± 0.024
NasNet-Mobile
0.01
0.743 ± 0.018 0.785 ± 0.017
0.712 ± 0.105
0.723 ± 0.037 0.798 ± 0.011 0.717 ± 0.047
0.001
0.807 ± 0.026 0.811 ± 0.028 0.790 ± 0.032
0.790 ± 0.027
0.785 ± 0.019
0.753 ± 0.050
0.0001
0.794 ± 0.025 0.770 ± 0.036
0.803 ± 0.017
0.779 ± 0.015
0.750 ± 0.021
0.783 ± 0.016
0.00001
0.778 ± 0.008 0.714 ± 0.011
0.770 ± 0.016
0.773 ± 0.006
0.659 ± 0.049
0.765 ± 0.010
DenseNet-201
0.01
0.694 ± 0.020 0.736 ± 0.032
0.666 ± 0.103
0.001
0.761 ± 0.032 0.799 ± 0.019
0.758 ± 0.025
0.0001
0.797 ± 0.036 0.774 ± 0.012
0.801 ± 0.034
0.00001
0.806 ± 0.025 0.826 ± 0.017 0.794 ± 0.014
16, where the results with the Adam solver never overcome SGDM and RMSProp solvers. The configurations selected for each model are reported in Table 7. These configurations were chosen according to the results obtained in Tables 4, 5, and 6 with 5-fold cross-validation. The optimizer and the initial learning rate that more times get the highest results were selected. For instance, the best values for MobileNet-V2 with a batch size of 16 were achieved using the RMSProp solver and an initial learning rate of 0.00001 in the three tables. However, some models present different best configurations depending on the measure taken into account, e.g., ResNet-18 with a batch size of 16 obtained the best results using the RMSProp optimizer for F-measure and Balanced Accuracy, whereas the best optimizer was SGDM for Accuracy. In this case, the RMSProp optimizer is selected. Likewise, the chosen initial learning rate was 0.00001 for ResNet-18.
# 4.3 Results
The experiments carried out to compare the performance of the different pretrained models to evaluate the functionality of the implemented dataset based on the selected configurations (Table 7) are summarized here. These configurations were employed to implement a 10-fold stratified cross-validation, where 90% of images were used for training the models and 10% to validate the training process. The experimental results using different neural models, batch size and image subgroups (LCA and RCA) are shown in Table 8, where Balanced accuracy, F-measure and Accuracy obtained in the test set are reported. In Table 8 are shown in bold the highest values obtained for each measure in both subsets. Note that the best results for Balanced Accuracy and Accuracy were obtained
Table 5. Balanced Accuracy results obtained on the test set for LCA images using 5-fold stratified cross-validation, five convolutional network architectures, and different values for batch size, initial learning rate, and optimizer. The best performances are shown in bold.
Model
Initial
Learning Rate
Batch Size 16
Batch Size 64
Adam
SGDM
RMSProp
Adam
SGDM
RMSProp
MobileNet-V2
0.01
0.586 ± 0.059 0.597 ± 0.052
0.514 ± 0.029
0.608 ± 0.029
0.610 ± 0.024 0.578 ± 0.041
0.001
0.656 ± 0.040 0.644 ± 0.052
0.650 ± 0.044
0.660 ± 0.073
0.630 ± 0.022 0.648 ± 0.052
0.0001
0.667 ± 0.034 0.643 ± 0.035
0.671 ± 0.011
0.665 ± 0.014
0.638 ± 0.030 0.657 ± 0.018
0.00001
0.682 ± 0.020 0.618 ± 0.013 0.688 ± 0.024 0.678 ± 0.027 0.590 ± 0.020 0.662 ± 0.012
ResNet-18
0.01
0.551 ± 0.071 0.559 ± 0.028
0.492 ± 0.034
0.594 ± 0.053
0.625 ± 0.028 0.582 ± 0.120
0.001
0.609 ± 0.037 0.618 ± 0.023
0.605 ± 0.017
0.636 ± 0.050 0.656 ± 0.040 0.583 ± 0.033
0.0001
0.592 ± 0.014 0.633 ± 0.027
0.623 ± 0.017
0.610 ± 0.037
0.632 ± 0.037 0.583 ± 0.027
0.00001
0.640 ± 0.009 0.633 ± 0.061 0.649 ± 0.024
0.615 ± 0.025
0.618 ± 0.043 0.638 ± 0.014
ResNet-50
0.01
0.558 ± 0.026 0.592 ± 0.029
0.500 ± 0.000
0.578 ± 0.050
0.632 ± 0.027 0.547 ± 0.084
0.001
0.600 ± 0.022 0.613 ± 0.022
0.564 ± 0.055
0.626 ± 0.014
0.642 ± 0.034 0.577 ± 0.037
0.0001
0.636 ± 0.031 0.653 ± 0.033
0.645 ± 0.041
0.638 ± 0.057 0.651 ± 0.022 0.648 ± 0.030
0.00001
0.659 ± 0.060 0.623 ± 0.039 0.679 ± 0.033
0.650 ± 0.047
0.600 ± 0.066 0.642 ± 0.023
NasNet-Mobile
0.01
0.562 ± 0.020 0.620 ± 0.048
0.531 ± 0.039
0.568 ± 0.043
0.639 ± 0.018 0.597 ± 0.024
0.001
0.626 ± 0.045 0.686 ± 0.034
0.649 ± 0.053
0.610 ± 0.047 0.647 ± 0.034 0.597 ± 0.049
0.0001
0.625 ± 0.029 0.644 ± 0.048 0.649 ± 0.025
0.637 ± 0.033
0.618 ± 0.035 0.629 ± 0.017
0.00001
0.621 ± 0.010 0.583 ± 0.025
0.629 ± 0.010
0.620 ± 0.013
0.529 ± 0.022 0.607 ± 0.017
DenseNet-201
0.01
0.537 ± 0.084 0.597 ± 0.067
0.512 ± 0.056
0.001
0.576 ± 0.041 0.666 ± 0.020
0.611 ± 0.033
0.0001
0.623 ± 0.060 0.627 ± 0.022
0.669 ± 0.059
0.00001
0.642 ± 0.037 0.672 ± 0.025 0.638 ± 0.022
with the MobileNet-V2 model for LCA subset (0.673 and 0.732, respectively). However, for the RCA set the NasNet-Mobile model achieved the best Balanced Accuracy and Accuracy (0.658, and 0.744). According to the F-measure, the best results were obtained by the ResNet-50 in both subsets (0.814, and 0.830, respectively). To study the best suitable model for these inputs, a ranking was implemented to evaluate the results attained considering the three measures. The rankings obtained are reported in Fig. 6, scoring the methods by set and batch size. The scores were calculated by sorting the obtained values of a measure in ascending order since a higher value is better according to the considered measures, meaning that the highest value will be in the last position. The position indicates the obtained scores. There are five and four methods for batch sizes 16 and 64, respectively. Therefore, the maximum possible score is 15 points and 12, respectively, indicating that the method attained the highest values in the three measures. Focusing on the LCA set, Fig. 6(a) and 6(b), the best outcomes are produced by MobileNet-V2, obtaining 15 points and 11 points with a batch size of 16 and 64, respectively. Although comparing the results obtained in Table 8, the best result is produced with a batch size of 64, which reached the maximum balanced accuracy, 0.673, and the second best F-measure and Accuracy, 0.810 and 0.732, respectively. However, in the case of RCA set, Fig. 6(c) and 6(d), ResNet-18 and NasNetMobile are the architectures that obtain higher performance, being NasNetMobile with a batch size of 64 which returns the best Balanced Accuracy and
Table 6. Accuracy results obtained on the test set for LCA images using 5-fold stratified cross-validation, five convolutional network architectures, and different values for batch size, initial learning rate, and optimizer. The best performances are shown in bold.
Model
Initial
Learning Rate
Batch Size 16
Batch Size 64
Adam
SGDM
RMSProp
Adam
SGDM
RMSProp
MobileNet-V2
0.01
0.659 ± 0.037 0.678 ± 0.041
0.642 ± 0.074
0.674 ± 0.035
0.710 ± 0.017
0.626 ± 0.045
0.001
0.716 ± 0.040 0.713 ± 0.039
0.708 ± 0.044
0.730 ± 0.041
0.696 ± 0.018
0.722 ± 0.033
0.0001
0.729 ± 0.036 0.708 ± 0.029
0.739 ± 0.017
0.726 ± 0.019
0.712 ± 0.018
0.725 ± 0.017
0.00001
0.733 ± 0.018 0.692 ± 0.012 0.745 ± 0.021 0.732 ± 0.013 0.667 ± 0.014
0.708 ± 0.012
ResNet-18
0.01
0.629 ± 0.069 0.634 ± 0.037
0.611 ± 0.089
0.653 ± 0.055
0.698 ± 0.014
0.634 ± 0.091
0.001
0.687 ± 0.021 0.705 ± 0.025
0.671 ± 0.022
0.697 ± 0.023 0.712 ± 0.025 0.661 ± 0.017
0.0001
0.684 ± 0.021 0.692 ± 0.027
0.708 ± 0.023
0.688 ± 0.036
0.699 ± 0.033
0.673 ± 0.019
0.00001
0.697 ± 0.018 0.710 ± 0.043 0.705 ± 0.022
0.679 ± 0.021
0.679 ± 0.026
0.695 ± 0.015
ResNet-50
0.01
0.572 ± 0.030 0.678 ± 0.030
0.473 ± 0.209
0.653 ± 0.017
0.703 ± 0.019
0.617 ± 0.064
0.001
0.676 ± 0.023 0.681 ± 0.017
0.650 ± 0.027
0.685 ± 0.043
0.694 ± 0.029
0.646 ± 0.570
0.0001
0.698 ± 0.022 0.702 ± 0.030 0.737 ± 0.027
0.717 ± 0.038
0.718 ± 0.019 0.732 ± 0.029
0.00001
0.715 ± 0.050 0.705 ± 0.031
0.731 ± 0.032
0.699 ± 0.049
0.685 ± 0.050
0.692 ± 0.028
NasNet-Mobile
0.01
0.638 ± 0.016 0.692 ± 0.028
0.613 ± 0.074
0.623 ± 0.042 0.711 ± 0.015 0.630 ± 0.037
0.001
0.715 ± 0.037 0.736 ± 0.035 0.706 ± 0.040
0.695 ± 0.039
0.701 ± 0.025
0.659 ± 0.056
0.0001
0.704 ± 0.028 0.687 ± 0.044
0.718 ± 0.021
0.693 ± 0.021
0.661 ± 0.028
0.694 ± 0.018
0.00001
0.687 ± 0.008 0.621 ± 0.017
0.682 ± 0.016
0.682 ± 0.007
0.562 ± 0.038
0.670 ± 0.014
DenseNet-201
0.01
0.589 ± 0.041 0.643 ± 0.047
0.569 ± 0.081
0.001
0.658 ± 0.039 0.719 ± 0.020
0.666 ± 0.030
0.0001
0.705 ± 0.049 0.685 ± 0.015
0.723 ± 0.032
0.00001
0.719 ± 0.034 0.746 ± 0.024 0.706 ± 0.019
able 7. Selected configurations considering F-measure, Balanced Accuracy and Acuracy obtained with 5-fold stratified cross-validation.
<div style="text-align: center;">Table 7. Selected configurations considering F-measure, Balanced Accuracy and Accuracy obtained with 5-fold stratified cross-validation.</div>
Model
Batch Size Optimizer Initial Learning Rate
MobileNet-V2
16
RMSprop 0.00001
64
Adam
0.00001
ResNet-18
16
RMSprop 0.00001
64
SGDM
0.001
ResNet-50
16
RMSprop 0.00001
64
RMSprop 0.0001
NasNet-Mobile
16
SGDM
0.001
64
SGDM
0.01
DenseNet-201
16
SGDM
0.00001
Accuracy values, 0.658 and 0.744, respectively, and the second best F-measure, 0.826, according to the attained results reported in Table 8.
# 5 Discussion
In this section, some important aspects to be considered of our proposal are discussed.
In this section, some important aspects to be considered of our proposal are discussed.
– In this work, all degrees of lesions are considered, while other studies only include severe lesions [25, 38] or exclude those images in which more than one lesion appears [25]. Nevertheless, our results indicate a fair-to-high performance, which means that the dataset is functional, but also that this classification task is complex and challenging.
Table 8. Balanced Accuracy, F-measure, and Accuracy results obtained on the test set for LCA and RCA images using 10-fold stratified cross-validation, five convolutional network architectures, and different batch size. The highest values by columns are shown in bold.
Model
Batch
Size
LCA
RCA
Balanced Accuracy
F-measure
Accuracy
Balanced Accuracy
F-measure
Accuracy
MobileNet-V2
16
0.668 ± 0.014
0.805 ± 0.013
0.725 ± 0.015
0.648 ± 0.023
0.811 ± 0.018
0.726 ± 0.023
64
0.673 ± 0.026
0.810 ± 0.020 0.732 ± 0.025
0.641 ± 0.041
0.806 ± 0.026
0.719 ± 0.034
ResNet-18
16
0.642 ± 0.026
0.796 ± 0.024
0.710 ± 0.029
0.658 ± 0.043
0.825 ± 0.023
0.743 ± 0.031
64
0.632 ± 0.013
0.779 ± 0.009
0.691 ± 0.010
0.624 ± 0.045
0.792 ± 0.031
0.702 ± 0.039
ResNet-50
16
0.664 ± 0.035
0.793 ± 0.029
0.713 ± 0.036
0.618 ± 0.022
0.799 ± 0.009
0.705 ± 0.013
64
0.645 ± 0.044
0.814 ± 0.021 0.728 ± 0.029
0.620 ± 0.029
0.830 ± 0.025 0.738 ± 0.031
NasNet-Mobile
16
0.634 ± 0.047
0.789 ± 0.023
0.701 ± 0.033
0.651 ± 0.054
0.804 ± 0.058
0.723 ± 0.067
64
0.637 ± 0.043
0.783 ± 0.020
0.696 ± 0.029
0.658 ± 0.034
0.826 ± 0.033 0.744 ± 0.039
DenseNet-201
16
0.641 ± 0.024
0.802 ± 0.022
0.715 ± 0.027
0.633 ± 0.037
0.812 ± 0.029
0.723 ± 0.037
– Despite the fact that a Balanced Accuracy between 0.65 and 0.67 was obtained, it represents a good performance since this measure depends on specificity and recall, which quantifies an average of how correctly both classes are classified, and the “non-lesion” class is worse classified than the “lesion” class. This is due to the fact that there are fewer “non-lesion” images than “lesion” images, although data augmentation was applied to equalize both classes. Since the “lesion” class includes all degrees of lesion, non-severe lesions can be incorrectly classified as “non-lesion”. – Although the DenseNet model usually has a good performance [39, 40, 41], we could not evaluate it with a batch size of 64 due to memory requirements. Nevertheless, training this model with a batch size of 16 yielded good results in terms of F-measure. Therefore, better results will be expected by training with higher batch sizes. – Finally, for LCA images, the selected configurations outcomes similar results, where the MobileNet-V2 was the most suitable model independently of the batch size. However, the results reported a higher variability for RCA images, since the most suitable models depend on the batch size. Thus, ResNet-18 is the best model for batch size 16, whereas for batch size 64 it is the worst model. This could be because the selected configurations were chosen with 5fold cross-validation results for the LCA images, and these configurations were applied to classify the RCA images.
# 6 Conclusions
The coronary dataset published in this work aims to provide the research community with a conscientious and exhaustive scientific resource. It can serve as a benchmark for both algorithm implementations and medical staff to train their abilities on angiographic assessment of CAD severity. Considering the researchers’ feedback, a set of utilities and the already extensive dataset will be regularly revised and expanded. Experiments were designed to try the functionality of the dataset, which was divided into LCA and RCA images. Five well-known classification architec-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/afc1/afc10f67-e579-446e-9786-3186e4fa8519.png" style="width: 50%;"></div>
<div style="text-align: center;">ng of LCA set with batch size 16. (b) Ranking of LCA set with batch size</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d6ce/d6ce4ae1-0b9c-43b1-828f-a90c60ddcca3.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) Ranking of RCA set with batch size 16. (d) Ranking of RCA set with batch size 64.</div>
<div style="text-align: center;">Fig. 6. Ranking of methods considering F-measure, Balanced Accuracy and Accuracy obtained for the LCA and RCA subsets using batch sizes 16 and 64.</div>
tures were trained and tested using augmented data to get an overview of the performance classification of the “lesion” and “non-lesion” images. Experiments showed that the most suitable models to solve this problem were MobileNetV2 for LCA images and NasNet-Mobile for RCA images, getting fair-to-high outcomes, around 80% F-measure and Accuracy, and 65% Balanced Accuracy. These results were obtained considering a wide range of lesion levels and support the idea that this classification task is complex, setting up a challenge for physicians and computer-aided diagnosis systems. Therefore, the provided dataset gives the scientific community a starting point to improve CAD detection.
Given the complexity of the classification problem posed, other architectures could be tested in order to keep evaluating the performance of different models. Also, future works will focus on classifying severe lesions images, as well as trying to identify each type of severity. The use of image patches to detect and classify the arteries present in this region will help to improve classification rates.
# Declarations
Data availability
CADICA dataset is open-access available at the Mendeley Data repository with the data identification number: 10.17632/p9bpx9ctcv.1, and direct URL to data: https://data.mendeley.com/datasets/p9bpx9ctcv/1.
# Author Contibutions
All authors listed have made a substantial, direct, and intellectual contribution to the work, and approved it for publication.
# Declaration of competing interest
Declaration of competing interest
The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.
# Conflicts of Interest
Conflicts of Interest
The authors declare that they have no conflicts of interest to report regarding the present study
# Acknowledgment
This work is partially supported by the Autonomous Government of Andalusia (Spain) under project UMA20-FEDERJA-108, project name Detection, characterization and prognosis value of the non-obstructive coronary disease with deep learning, and also by the Ministry of Science and Innovation of Spain, grant number PID2022-136764OA-I00, project name Automated Detection of Non Lesional Focal Epilepsy by Probabilistic Diffusion Deep Neural Models. It includes funds from the European Regional Development Fund (ERDF). It is also partially supported by the University of M´alaga (Spain) under grants B1-2019 01, project name Anomaly detection on roads by moving cameras; B1-2019 02, project name Self-Organizing Neural Systems for Non-Stationary Environments; B1-2021 20, project name Detection of coronary stenosis using deep learning applied to coronary angiography; B4-2022, project name Intelligent Clinical Decision Support System for Non-Obstructive Coronary Artery Disease in Coronarographies; B1-2022 14, project name Detecci´on de trayectorias an´omalas de veh´ıculos en c´amaras de tr´afico; and by the Fundaci´on Unicaja under project PUNI-003 2023, project name Intelligent System to Help the Clinical Diagnosis of Non-Obstructive Coronary Artery Disease in Coronary Angiography. The authors thankfully acknowledge the computer resources, technical expertise and assistance provided by the SCBI (Supercomputing and Bioinformatics) center of the University of M´alaga. They also gratefully acknowledge the
support of NVIDIA Corporation with the donation of a RTX A6000 GPU with 48Gb. The authors also thankfully acknowledge the grant of the Universidad de M´alaga and the Instituto de Investigaci´on Biom´edica de M´alaga y Plataforma en Nanomedicina-IBIMA Plataforma BIONAND.
# References
[1] Sherry L Murphy et al. “Mortality in the United States, 2020”. In: NCHS Data Brief 427 (2021). [2] Konstantinos V. Voudris and Clifford J. Kavinsky. “Advances in Management of Stable Coronary Artery Disease: the Role of Revascularization?” eng. In: Current Treatment Options in Cardiovascular Medicine 21.3 (Mar. 2019), p. 15. issn: 1092-8464. doi: 10.1007/s11936-019-0720-9. [3] Antti Saraste et al. “Imaging in ESC clinical guidelines: chronic coronary syndromes”. eng. In: European Heart Journal. Cardiovascular Imaging 20.11 (Nov. 2019), pp. 1187–1197. issn: 2047-2412. doi: 10.1093/ehjci/ jez219. [4] Juhani Knuuti et al. “2019 ESC Guidelines for the diagnosis and management of chronic coronary syndromes: The Task Force for the diagnosis and management of chronic coronary syndromes of the European Society of Cardiology (ESC)”. In: European Heart Journal 41.3 (Aug. 2019), pp. 407–477. issn: 0195-668X. doi: 10.1093/eurheartj/ehz425. [5] Jean-Philippe Collet et al. “2020 ESC Guidelines for the management of acute coronary syndromes in patients presenting without persistent STsegment elevation”. en. In: European Heart Journal 42.14 (Apr. 2021), pp. 1289–1367. issn: 0195-668X, 1522-9645. doi: 10.1093/eurheartj/ ehaa575. (Visited on 07/30/2021). [6] H. G. Kemp et al. “Seven year survival of patients with normal or near normal coronary arteriograms: a CASS registry study”. eng. In: J Am Coll Cardiol 7.3 (Mar. 1986), pp. 479–483. issn: 0735-1097. doi: 10.1016/ s0735-1097(86)80456-9. [7] Jorge Rodr´ıguez-Capit´an et al. “Prognostic Implication of Non-Obstructive Coronary Lesions: A New Classification in Different Settings”. eng. In: Journal of Clinical Medicine 10.9 (Apr. 2021), p. 1863. issn: 2077-0383. doi: 10.3390/jcm10091863. [8] Zhi Jian Wang et al. “Prevalence and Prognosis of Nonobstructive Coronary Artery Disease in Patients Undergoing Coronary Angiography or Coronary Computed Tomography Angiography: A Meta-Analysis”. eng. In: Mayo Clin Proc 92.3 (Mar. 2017), pp. 329–346. issn: 1942-5546. doi: 10.1016/j.mayocp.2016.11.016. [9] Francesco Radico et al. “Determinants of long-term clinical outcomes in patients with angina but without obstructive coronary artery disease: a systematic review and meta-analysis”. eng. In: European Heart Journal 39.23 (June 2018), pp. 2135–2146. issn: 1522-9645. doi: 10.1093/eurheartj/ ehy185.
[1] Sherry L Murphy et al. “Mortality in the United States, 2020”. In: NCHS Data Brief 427 (2021). [2] Konstantinos V. Voudris and Clifford J. Kavinsky. “Advances in Management of Stable Coronary Artery Disease: the Role of Revascularization?” eng. In: Current Treatment Options in Cardiovascular Medicine 21.3 (Mar. 2019), p. 15. issn: 1092-8464. doi: 10.1007/s11936-019-0720-9. [3] Antti Saraste et al. “Imaging in ESC clinical guidelines: chronic coronary syndromes”. eng. In: European Heart Journal. Cardiovascular Imaging 20.11 (Nov. 2019), pp. 1187–1197. issn: 2047-2412. doi: 10.1093/ehjci/ jez219. [4] Juhani Knuuti et al. “2019 ESC Guidelines for the diagnosis and management of chronic coronary syndromes: The Task Force for the diagnosis and management of chronic coronary syndromes of the European Society of Cardiology (ESC)”. In: European Heart Journal 41.3 (Aug. 2019), pp. 407–477. issn: 0195-668X. doi: 10.1093/eurheartj/ehz425. [5] Jean-Philippe Collet et al. “2020 ESC Guidelines for the management of acute coronary syndromes in patients presenting without persistent STsegment elevation”. en. In: European Heart Journal 42.14 (Apr. 2021), pp. 1289–1367. issn: 0195-668X, 1522-9645. doi: 10.1093/eurheartj/ ehaa575. (Visited on 07/30/2021). [6] H. G. Kemp et al. “Seven year survival of patients with normal or near normal coronary arteriograms: a CASS registry study”. eng. In: J Am Coll Cardiol 7.3 (Mar. 1986), pp. 479–483. issn: 0735-1097. doi: 10.1016/ s0735-1097(86)80456-9. [7] Jorge Rodr´ıguez-Capit´an et al. “Prognostic Implication of Non-Obstructive Coronary Lesions: A New Classification in Different Settings”. eng. In: Journal of Clinical Medicine 10.9 (Apr. 2021), p. 1863. issn: 2077-0383. doi: 10.3390/jcm10091863. [8] Zhi Jian Wang et al. “Prevalence and Prognosis of Nonobstructive Coronary Artery Disease in Patients Undergoing Coronary Angiography or Coronary Computed Tomography Angiography: A Meta-Analysis”. eng. In: Mayo Clin Proc 92.3 (Mar. 2017), pp. 329–346. issn: 1942-5546. doi: 10.1016/j.mayocp.2016.11.016. [9] Francesco Radico et al. “Determinants of long-term clinical outcomes in patients with angina but without obstructive coronary artery disease: a systematic review and meta-analysis”. eng. In: European Heart Journal 39.23 (June 2018), pp. 2135–2146. issn: 1522-9645. doi: 10.1093/eurheartj/ ehy185.
[10] Nick Curzen et al. “Does routine pressure wire assessment influence management strategy at coronary angiography for diagnosis of chest pain?: the RIPCORD study”. eng. In: Circ Cardiovasc Interv 7.2 (Apr. 2014), pp. 248–255. issn: 1941-7632. doi: 10.1161/CIRCINTERVENTIONS.113. 000978. [11] Pim A. L. Tonino et al. “Fractional flow reserve versus angiography for guiding percutaneous coronary intervention”. eng. In: The New England Journal of Medicine 360.3 (Jan. 2009), pp. 213–224. issn: 1533-4406. doi: 10.1056/NEJMoa0807611. [12] E. J. Topol and S. E. Nissen. “Our preoccupation with coronary luminology. The dissociation between clinical and angiographic findings in ischemic heart disease”. eng. In: Circulation 92.8 (Oct. 1995), pp. 2333– 2342. issn: 0009-7322. doi: 10.1161/01.cir.92.8.2333. [13] David A. Halon. “Can angiography predict physiology?” eng. In: International Journal of Cardiology 270 (Nov. 2018), pp. 74–75. issn: 1874-1754. doi: 10.1016/j.ijcard.2018.07.029. [14] Lavinia Gabara et al. “Coronary Physiology Derived from Invasive Angiography: Will it be a Game Changer?” eng. In: Interv Cardiol 15 (Apr. 2020), e06. issn: 1756-1485. doi: 10.15420/icr.2019.25. [15] Philipp Tschandl, Cliff Rosendahl, and Harald Kittler. “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions”. In: Scientific data 5.1 (2018), pp. 1–9. [16] Noel CF Codella et al. “Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic)”. In: 2018 IEEE 15th international symposium on biomedical imaging (ISBI 2018). IEEE. 2018, pp. 168–172. [17] Fabio A Spanhol et al. “A dataset for breast cancer histopathological image classification”. In: Ieee transactions on biomedical engineering 63.7 (2015), pp. 1455–1462. [18] Damian J Matuszewski and Ida-Maria Sintorn. “TEM virus images: Benchmark dataset and deep learning classification”. In: Computer Methods and Programs in Biomedicine 209 (2021), p. 106318. [19] Emmanuel Ovalle-Magallanes et al. “Improving convolutional neural network learning based on a hierarchical bezier generative model for stenosis detection in X-ray images”. In: Computer Methods and Programs in Biomedicine 219 (2022), p. 106767. [20] Ebrahim Nasr-Esfahani et al. “Segmentation of vessels in angiograms using convolutional neural networks”. In: Biomedical Signal Processing and Control 40 (2018), pp. 240–251. [21] Kritika Iyer et al. “Angionet: a convolutional neural network for vessel segmentation in X-ray angiography”. In: Scientific Reports 11.1 (2021), p. 18066.
[22] Chengyang Zhou et al. “Automated deep learning analysis of angiography video sequences for coronary artery disease”. In: arXiv preprint arXiv:2101.12505 (2021). [23] Chao Cong et al. “Automated stenosis detection and classification in xray angiography using deep neural network”. In: 2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE. 2019, pp. 130 1308. [24] Yucheng Song et al. “Deep learning-based automatic segmentation of images in cardiac radiography: a promising challenge”. In: Computer Methods and Programs in Biomedicine (2022), p. 106821. [25] Benjamin Au et al. “Automated characterization of stenosis in invasive coronary angiography images with convolutional neural networks”. In: arXiv preprint arXiv:1807.10597 (2018). [26] Wei Wu et al. “Automatic detection of coronary artery stenosis by convolutional neural network with temporal constraint”. In: Computers in biology and medicine 118 (2020), p. 103657. [27] Xinyue Zhang et al. “X-ray coronary centerline extraction based on CUNet and a multifactor reconnection algorithm”. In: Computer Methods and Programs in Biomedicine 226 (2022), p. 107114. [28] Jong Hak Moon et al. “Automatic stenosis recognition from coronary angiography using convolutional neural networks”. In: Computer methods and programs in biomedicine 198 (2021), p. 105819. [29] Tom Finck et al. “10-year follow-up after coronary computed tomography angiography in patients with suspected coronary artery disease”. In: JACC: Cardiovascular Imaging 12.7 Part 2 (2019), pp. 1330–1338. [30] Se Hun Kang et al. “Long-term prognostic value of coronary CT angiography in asymptomatic type 2 diabetes mellitus”. In: JACC: Cardiovascular Imaging 9.11 (2016), pp. 1292–1300. [31] Jian Wang et al. “A review of deep learning on medical image analysis”. In: Mobile Networks and Applications 26.1 (2021), pp. 351–380. [32] Emmanuel Ovalle-Magallanes et al. “Transfer learning for stenosis detection in X-ray coronary angiography”. In: Mathematics 8.9 (2020), p. 1510. [33] Kaiming He et al. “Deep residual learning for image recognition”. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2016, pp. 770–778. [34] Mark Sandler et al. “Mobilenetv2: Inverted residuals and linear bottlenecks”. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2018, pp. 4510–4520. [35] Wannipa Sae-Lim, Wiphada Wettayaprasit, and Pattara Aiyarak. “Convolutional neural networks using MobileNet for skin lesion classification”. In: 2019 16th international joint conference on computer science and software engineering (JCSSE). IEEE. 2019, pp. 242–247. [36] Narsi Reddy, Ajita Rattani, and Reza Derakhshani. “Comparison of deep learning models for biometric-based mobile user authentication”. In: 2018
IEEE 9th international conference on biometrics theory, applications and systems (BTAS). IEEE. 2018, pp. 1–6. [37] Gao Huang et al. “Densely connected convolutional networks”. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2017, pp. 4700–4708. [38] Yiwen Shu and Xiwen Wu. “Deep Learning Based Coronary Angiography in Diagnosis of Myocardial Ischemia”. In: Scientific Programming 2021 (2021). [39] Sebastian Guendel et al. “Learning to recognize abnormalities in chest xrays with location-aware dense networks”. In: Iberoamerican Congress on Pattern Recognition. Springer. 2018, pp. 757–765. [40] Karl Thurnhofer-Hemsi and Enrique Dom´ınguez. “A convolutional neural network framework for accurate skin cancer detection”. In: Neural Processing Letters 53.5 (2021), pp. 3073–3093. [41] Tavishee Chauhan, Hemant Palivela, and Sarveshmani Tiwari. “Optimization and Fine-Tuning of DenseNet model for classification of Covid-19 cases in Medical Imaging”. In: International Journal of Information Management Data Insights 1.2 (2021), p. 100020.
