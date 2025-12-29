# Exploring Effective Factors for Improving Visual In-Context Learning
Yanpeng Sun1,2, Qiang Chen1, Jian Wang1, Jingdong Wang1, Zechao Li2* 1Baidu VIS 2School of Computer Science and Engineering, Nanjing University of Science and Technology
1Baidu VIS 2School of Computer Science and Engineering, Nanjing University of Science and Technology
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/79be/79be99db-45cc-45b8-abf8-833505638a09.png" style="width: 50%;"></div>
# Abstract
The In-Context Learning (ICL) is to understand a new task via a few demonstrations (aka. prompt) and predict new inputs without tuning the models. While it has been widely studied in NLP, it is still a relatively new area of research in computer vision. To reveal the factors influencing the performance of visual in-context learning, this paper shows that prompt selection and prompt fusion are two major factors that have a direct impact on the inference performance of visual context learning. Prompt selection is the process of identifying the most appropriate prompt or example to help the model understand new tasks. This is important because providing the model with relevant prompts can help it learn more effectively and efficiently. Prompt fusion involves combining knowledge from different positions within the large-scale visual model. By doing this, the model can leverage the diverse knowledge stored in different parts of the model to improve its performance on new tasks. Based these findings, we propose a simple framework prompt-SelF for visual in-context learning. Specifically, we first use the pixel-level retrieval method to select a suitable prompt, and then use different prompt fusion methods to activate all the knowledge stored in the large-scale model, and finally ensemble the prediction results obtained from different prompt fusion methods to obtain the final prediction results. And we conduct extensive experiments on single-object segmentation and detection tasks to demonstrate the effectiveness of prompt-SelF. Remarkably, the prompt-SelF has outperformed OSLSM based meta-learning in 1-shot segmentation for the first time. This indicated the great potential of visual in-context learning. The source code and models will be available at https: //github.com/syp2ysy/prompt-SelF.
# 1. Introduction
Benefiting from the large models and large scale datasets in NLP, researchers realized that the large models [28, 29, 4]
Figure 1: A framework for visual in-context learning. It involves the following steps. (a) Select the prompt pair corresponding to the task from the database based on the query image. (b) Fusion the query image and the selected prompt image to create a new image. (c) Feed new images into a pre-trained large-scale vision model to get predictions for query images. This paradigm enables the visual model to learn in-context by utilizing the knowledge captured from the prompt image to enhance the accuracy of the prediction for the demand image.
have a crucial emergent ablity, which is In-context Learning. The purpose of in-context learning is to assist the model in comprehending new tasks and making predictions for new examples based on provided prompt. Typically, the prompt is a concise, structured input that provides context for the task, such as a task description or an example of an input-label pair. As a well-known field in NLP [11, 39], in-context learning has just started in the field of vision. Indeed, visual in-context learning is becoming increasingly important in computer vision, particularly with the rise of large-scale models. While these models can achieve impressive results on many tasks [12], they often require massive amounts of data and computation to train, making them impractical for many real-world applications. As such, visual in-context learning is becoming increasingly important
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4a15/4a155636-6630-4577-b84d-4f7d6db6eab6.png" style="width: 50%;"></div>
<div style="text-align: center;">gure 2: (a) The selection of different prompts can affect the results of the same query image, therefore, choosing a high ality prompt can have a positive impact on visual in-context learning. (b) Different prompt fusion methods activate knowl ge at different positions in the large-scale model, leading to varying degrees of impact on visual in-context learning. Note at the red box in the figure is the prediction result, and the blue box is the prompt pair.</div>
for developing more efficient and accurate computer vision systems that can operate in real-world settings. However, these research is relatively limited, so we are concentrating on visual in-context learning and carrying out preliminary studies. Different from NLP, where the input is typically text, the input in visual tasks is often a single image. Therefore, the challenge of visual in-context learning lies in how to make the model understand the prompt and make predictions on the new image (aka. query image). Toward this end, recent studies [35, 3] propose unified visual in-context learning frameworks, illustrated in Figure 1. It involve selecting a prompt from a database based on query image, fusing it with the query image to create a combine images, and feeding this image into a large-scale model to obtain a prediction. Recent studies [31, 24, 23] have demonstrated that the quality of prompts is the primary determinant of performance in language in-context learning. In the context of visual in-context learning, we ask: what are the critical factors that influence downstream task performance? Under the premise of large-scale model determination, cue selection and fusion are the main determinants of visual context learning performance. We first analyze the prompt selection. Prior work in NLP [20] has shown that a highquality prompt can help the model better understand the task and yield improved results. We found that when different prompts were used for the same query image (as shown in Figure 2 (a)), the prediction results were highly sensitive to the prompt selection. A high-quality prompt can lead to a significantly increase in Intersection over Union (IoU) for the prediction result. Thus, one of the challenges in visual in-context learning is how to efficiently select high-quality prompt. In Figure 2 (a), we obtained the best results by using the query image itself as the prompt, indicating that high-quality prompts are spatially similar to the query image. For prompt selection, vpr [41] propose an image-level
retrieval framework. However, this approach does not account for certain characteristics of high-quality prompts.
Prompt fusion aims to merge the prompt and query image into a new image, which becomes the input for the large-scale model. The new image includes four subimages: the prompt image, prompt ground truth, query image, and prediction result. To ensure that the prompt pair is combined correctly (adjacent to each other), prompt fusion has a total of eight different arrangements (as shown in Figure 2 (b)). In this paper, we conduct a comprehensive investigation into the impact of different sub-image arrangements on in-context learning, and identify a critical issue: downstream performance is highly sensitive to variations in sub-image arrangements. Because most largescale vision models are based on masked image modeling (MIM) [12, 3, 6, 2], different prompt fusion methods activate knowledge at different locations in the large-scale model, affecting the downstream task performance. Thus, determining which prompt fusion method is the most effective for new downstream tasks is one of the major challenges in visual context learning. Toward this end, we propose a simple framework prompt-SelF for visual in-context learning. In prompt-SelF, we first use the pixel-level retrieval method to select a suitable prompt image that is similar to the input image in terms of visual content and context. This is important because a good prompt can provide relevant information and guidance to the model for understanding the new task. Once a suitable prompt image is selected, we then use different prompt fusion methods to activate all the knowledge stored in the large-scale model. This involves combining the prompt image with the input image in various ways to create a composite image that can better capture the relevant information needed for the task. By doing this, we can activate diverse knowledge in the large-scale model and improve the accuracy and efficiency of in-context learning. Finally,
we ensemble the prediction results obtained from different prompt fusion methods to obtain the final prediction results. This is done by using a majority voting strategy [17]. The ensemble strategy helps to further improve the performance of visual in-context learning by combining the strengths of different prompt fusion methods. We have conducted extensive experiments on single-object segmentation and detection tasks using visual in-context learning. Notably, our proposed approach, prompt-SelF, achieved state-of-the-art results and outperformed meta-learning-based methods for the first time on single-object segmentation tasks.
# 2. Related work
# 2.1. In-context learning
With the emergence of large models such as GPT-3 [4] and Instruction GPT [25], prompting these models more efficiently has become a concern for both academia and industry. As a result, the in-context learning method has gained popularity in the field of NLP. In-context learning aims to improve the performance of these models by providing additional context during inference. In recent years, in-context learning has been applied to a variety of NLP tasks [10, 23], including language modeling [13, 4], text classification [31, 24], question answering [26, 20, 14], and natural language generation [15]. However, the concept of in-context learning is still relatively new in computer vision [1]. The first step in implementing in-context learning is to unify different tasks into the same space. However, while some recent work [16, 5, 21] has attempted to unify different tasks, they have been limited in their ability to learn new tasks. To solve this problem, some recent works [35, 3, 41] activate the model’s ability to learn new tasks by combining prompt and query image fusion together. But it does not explore the factors that affect the performance of visual in-context learning. Therefore, we have conducted the first exploration of the main factors that affect the performance of visual in-context learning inference.
# 2.2. Optimization method of ICL in NLP
In the inference stage, optimizing ICL aims to improve the performance of the model on new tasks without altering the large-scale model [25]. The prompt, as an input to activate the ability of large models, has a significant impact on ICL. In natural language processing, prompts are often designed based on organization and format [7]. The organization method refers to how to select a suitable prompt. Existing work [15, 40, 22] can select similar results through text representation and mutual information, and prompts can even be directly generated by the language model itself. The format of the prompt can be divided into instruction [25, 37] and reasoning steps [38, 36]. The instruction
is mainly designed manually, and the reasoning step is designed to inspire the reasoning ability of the model by introducing reasoning steps. To advance the field of visual in-context learning, this paper presents the first exploration of optimization methods for visual in-context learning.
# 3. Methods
In this section, we start with the preliminaries on the visual in-context learning setting. Then, the analysis of two factors involved in visual in-context learning is presented, followed by a detailed introduction of the prompt-SelF process. Specifically, Prompt selection and Prompt fusion are discussed in sections 3.2 and section 3.3, respectively.
# 3.1. Visual In-context learning
In natural language processing, remarkable progress has been made in the development of in-context learning. Large-scale pre-trained language models, such as Instruction GPT [25], have introduced text prompts into the input, allowing the model to predict and learn different tasks without the need to update parameters. In computer vision, the concept of in-context learning is still relatively new. Previous studies [35, 3] have focused on combining images and labels into a new image and using MIM for pre-training, resulting in models with in-context learning capabilities. However, their performance on new tasks was not satisfactory. The primary goal of this paper is to investigate the key factors that impact visual context learning and improve the current frameworks. In new task, given a prompt dataset C “ tpx1, y1q, px2, y2q, ...., pxn, ynqu, in which there are n pairs of image-label pairs, pxk, ykq represents the k ´ th image-label pair. If we have an image xq and a model ficl with in-context learning capability, then the process of in-context learning can be expressed as follows:
(1)
where, f ˚ icl represents the visual model ficl with frozen parameters. pxi, yiq refers to the prompt pair selected from the prompt dataset C based on query image xq. Therefore, changing prompt has a direct impact on the performance of visual context learning. Additionally, the arrangement of prompt and query images is not fixed, and changing the arrangement can activate different knowledge from various positions in the large-scale visual model. As a result, we can conclude that prompt selection and prompt fusion are the main factors that affect visual in-context learning. Based on this analysis, we have designed a simple framework called Prompt-SelF.
Algorithm 1 Pseudo code of Prompt Selection
# Input: all images in the dataset ’X’, query image
’x_q’, the off-the-shelf feature extractor ’
img_encoder’
# Output: Output prompt index ’idx’
F_s = img_encoder.forward(X) # (N,C,H,W)
F_q = img_encoder.forward(x_q) # (1,C,H,W)
# normalization & reshape
F_s = (F_s / norm(F_s, dim=1)).reshape(N,-1)
F_q = F_q / norm(F_q, dim=1).reshape(1,-1)
# calculate similarity & get index
similarity = dot(F_q, F_s) # (1,N)
idx = argsort(similarity,dim=1)[0]
Algorithm 2 Pseudo code of Prompt Fusion
# Input: prompt pair ’(x_p,y_p)’, query image ’x_q’,
the visual generative model ’gen_model’
# Output: Output prediction result ’pred_q’
#Get the fusion image of 8 arrangements
arr_list = get_fusion_image(x_p,y_p,x_q)
#Predict results on 8 fusion images
pred_list = []
for fusion_image in arr_list:
sub_pred = gen_model.forward(fusion_image)
pred_list.append(sub_pred)
#get the final prediction result
pred_q = ensembel(pred_list)
# 3.2. Prompt Selection
The purpose of prompt selection is to automatically select a high-quality prompt for the query image. We ask, what are the characteristics of a high-quality prompt? The impact of different prompts on the prediction results of the query image is illustrated in Figure 2 (a). Our findings suggest that prompts with high spatial similarity to the query image can have a positive impact on the prediction results. Based on this finding, we propose the Prompt-SelF framework that leverages pixel-level retrieval to identify the suitable prompt for the query image. The detailed process of prompt selection based pixellevel retrieval is shown in Algorithm 1. First, we use the off-the-shelf feature extractor to obtain the spatial representation Fq of the query image and the spatial representation Fs of all images in the dataset C. Then, we normalize the image features using L2 normalization and calculate the pixel-level similarity between the query image and all images in the dataset following form.
(2)
Finally, we select the image-label pair with the highest similarity score as the prompt pxP , yP q. Once we have retrieved the appropriate prompt for the query image, the next step is to fuse the prompt and the query image, and use the largescale visual model to make predictions.
# 3.3. Prompt fusion
The purpose of prompt fusion is to combine the prompt and query image into a new input image for the large-scale model, to obtain the predicted result for the query image. There are eight possible ways to arrange the sub-images while ensuring that the prompt pair is adjacent to each other. Different arrangements have varying effects on visual incontext learning, as illustrated in Figure 2 (b). The reason for this phenomenon is that different arrangements of subimages activate different parts of the knowledge stored in the large-scale model. We ask, How can we leverage the
knowledge stored in the large-scale model to its fullest potential? Toward this end, Prompt-SelF proposes a simple method to maximizing the potential of large-scale vision models by employing different permutations of sub-images to produce multiple predictions, which are then merged to yield more precise outcomes. Algorithm 2 provides a detailed overview of the prompt fusion process in Prompt-SelF. It aims to fully leverage the diverse knowledge in large-scale visual models. To achieve this, we fuse the query image and prompt image using different arrangements to create eight new fused images, as illustrated in Figure 2 (b). These eight new images are then input into the frozen large-scale visual model to obtain eight prediction results based on the knowledge of different positions in the model. Finally, we ensemble these eight results to obtain the final prediction. In the final stage of ensemble, we use the simple voting strategy, where the default threshold in the paper is set to 4{8. This means that if more than four out of the eight predictions are in agreement, the final prediction is made based on their consensus. In summary, we analyzed the main factors that affect visual in-context learning and presented the implementation details of the proposed prompt-SelF framework. By selecting higher-quality prompts and leveraging the diverse knowledge captured by large-scale visual models, promptSelF can effectively improve the accuracy and efficiency of contextual learning. Subsequently, we conducted experiments to evaluate the effects of prompt selection and prompt fusion on visual in-context learning, and to validate the efficacy of Prompt-SelF.
# 4. Experiments
In this section, we perform comprehensive experiments on single-object segmentation and detection tasks to validate several crucial factors that affect visual context learning and demonstrated the effectiveness of prompt-SelF. In section 4.1, we introduce the experimental setup and implementation details. In section 4.2, we compare our proposed prompt-SelF method with state-of-the-art methods to verify its superiority and versatility. Finally, in section 4.3, we an-
<div style="text-align: center;">Table 1: Main results on different computer vision tasks. The best results based in-context learning are show in bold.</div>
Seg.(mIoU)Ò
Det.
(mIoU)Ò
Fold-0 Fold-1 Fold-2 Fold-3 Mean
Meta-learning
OSLSM [32]
33.60
55.30
40.90
33.50
40.80
-
co-FCN [30]
36.70
50.60
44.90
32.40
41.10
-
In-context learning
VP-Random † [3]
27.49
30.21
25.88
24.06
26.91
25.14
VPR-UsupPR † [41]
34.70
35.92
32.39
31.12
33.53
26.44
VPR-SupPR † [41]
37.08
38.43
34.40
32.32
35.56
28.22
prompt-SelF
42.48
43.34
39.76
38.50
41.02
29.83
alyze the characteristics of prompt-SelF to further demonstrate its effectiveness.
# 4.1. Setting
Downstream tasks and Datsets. To ensure fair comparison, we adopt the experimental settings of previous works [3, 41] for single-object segmentation, single object detection. Next, we will provide a detailed introduction to these tasks and the datasets used.
• single object segmentation: The goal is to extract the specific object from the query image based on the prompt. This experimental setup is similar to few-shot segmentation, where new object classes in the query image are recognized based on few of labeled samples. Thus, following previous research [3, 33, 34, 18], we evaluate our method on the pasacl-5i [32] dataset.
 single object detection: Different from traditional detection tasks, the goal here is to obtain fine-grained segmentation results based on coarse-grained bounding box annotations in prompt. Following the previous work [3], we conduct experiments on images and bounding boxes from the PASCAL VOC 2012 [9] dataset, specifically selecting images that contain only a single object and filtering out trivial images where the object covers more than 50% of the image.
Evaluation method. We conducted a comprehensive evaluation of the effectiveness of prompt-SelF by comparing it with several existing methods for learning visual context, including VP [3] and VPR [41]. Additionally, we compared prompt-SelF with few-shot segmentation methods based meta-learning [32, 30]. In the ablation experiment, we compared three prompt selection methods, namely random selection, image-level retrieval, and pixel-level retrieval. Moreover, we evaluated the impact of different prompt arrangements, denoted as Arrangement1 to Arrangement8, which correspond to A1-A8 in Figure 2 (b). We conduct this comparison to validate the significance
Table 2: Ablation study of different prompt selection retrieval methods. and represent the prompt fusion method of A1 and prompt-SelF respectively. The best results based in-context learning are show in bold.
prompt selection prompt fusion Fold-0 Fold-1 Fold-2 Fold-3 Means
random
27.49
30.21
25.88
24.06
26.91
33.91
36.12
31.97
29.37
32.84
image-level
34.70
35.92
32.92
31.12
33.53
41.07
41.32
38.14
36.44
39.24
pixel-level
36.42
38.47
34.56
34.12
35.89
42.48
43.34
39.76
38.50
41.02
of prompt selection and prompt fusion in visual in-context learning, while also ensuring a comprehensive and impartial assessment of the effectiveness of prompt-SelF. Implementation details. To perform in-context learning, not all large-scale models in computer vision are capable enough. Thus, we employ a pre-trained model named MAE-VQGAN [3] specifically designed for this purpose. In the prompt selection phase, we extract features using CLIP’s visual encoder [27], which yields two types of features: image-level features and pixel-level features. The former is obtained by using the classification header, while the latter is obtained without it. Since the pre-training model’s input size is 224 ˆ 224, we maintain the same size for image inputs in our experiments. We divide the image into sub-images of size 111 ˆ 111, with a spacing of 1 pixel between each sub-image.
# 4.2. Comparison with State-of-the-Art
To evaluate the superiority of prompt-SelF, we compared it not only with the existing in-context learning method, but also with a few-shot segmentation method [32, 30] based on meta-learning. The experimental results are presented in Table 1, indicating that our method significantly enhances the performance of visual in-context learning across different tasks. It’s important to highlight that promptSelF has demonstrated superior performance compared to an unsupervised method VPR-UsupPR [41] in different downstream tasks, particularly achieving a substantial 7.5 mIoU improvement in segmentation task. Additionally, our method also outperforms VPR-SupPR [41] that necessitate training. The remarkable performance of prompt-SelF in the segmentation task, surpassing that of meta-learning based methods for the first time, is particularly surprising. These results showcase both the superiority of prompt-SelF and the immense potential of visual context learning. By selecting high-quality prompts and leveraging the diverse knowledge captured by large-scale vision models, we can significantly enhance the accuracy and efficiency of context learning, leading to further advancements in the field of visual recognition and analysis.
able 3: Ablation study of different sub-images arrangements. Image-level and Pixel-level represent prompt selection methds based on image-level retrieval and pixel-level retrieval, respectively. The best results based in-context learning are show
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a1cf/a1cf8805-acc1-4b89-a828-d48aaf22aec6.png" style="width: 50%;"></div>
Image-level
Pixel-level
Fold-0
Fold-1
Fold-2
Fold-3
Mean
Fold-0
Fold-1
Fold-2
Fold-3
Mean
Arrangement1
34.70
35.92
32.92
31.12
33.53
36.42
38.47
34.56
34.12
35.89
Arrangement2
34.53
35.82
32.20
31.29
33.46
36.46
38.49
34.30
33.81
35.77
Arrangement3
36.40
38.06
34.19
34.07
35.68
37.93
40.48
35.90
36.07
37.60
Arrangement4
35.96
37.53
33.63
32.95
35.02
36.70
39.62
35.62
34.87
36.70
Arrangement5
33.15
38.25
31.83
32.39
33.91
33.55
39.21
33.86
34.80
35.36
Arrangement6
33.21
36.01
31.31
32.18
33.18
34.57
38.54
33.12
34.37
35.15
Arrangement7
35.71
36.81
35.49
30.77
34.70
37.89
39.24
37.16
34.16
37.11
Arrangement8
35.88
38.29
36.43
31.98
35.65
37.68
39.80
37.65
34.86
37.50
ensemble
41.07
41.32
38.14
36.44
39.24
42.48
43.34
39.76
38.50
41.02
A1
A2
A3
A4
A5
A6
A7
A8
Ensemble
IoU:62.59
IoU:66.18
IoU:65.49
IoU:70.36
IoU:68.28
IoU:67.94
IoU:71.49
IoU:65.48
IoU:79.39
IoU:70.54
IoU:70.78
IoU:66.89
IoU:70.83
IoU:69.67
IoU:72.29
IoU:64.89
IoU:59.65
IoU:78.13
<div style="text-align: center;">Figure 3: Visualization results of different prompt fusion methods. A1-A8 respectively correspond to Arrangement1Arrangement8 in Table 3, where the blue box is the prompt pair, and the red box is the prediction result. The best results are</div>
# 4.3. Ablation Study
In this section, we conduct an analysis of the factors that affect visual in-context learning and verify the effectiveness of prompt-SelF. All experiments below are performed on single-object segmentation tasks.
Prompt selection. It aim to select the best prompt for a query image, which can be done using various methods, such as random selection [3] or retrieval based on imagelevel features [41]. However, based on the analysis presented in Figure 2 (a), we identified the characteristic of high-quality prompts and proposed prompt-SelF, a retrieval method based on pixel-level features. To validate the effectiveness of this method, we compared it with the other two prompt selection methods. The random retrieval setting is identical to that of VP [3]. Based on the results presented in Table 2, we observe a significant improvement in the performance of single object segmentation task when using hints obtained by pixel-level retrieval. This suggests that
the prompts obtained through the retrieval method based on pixel-level features are better suited to the characteristics of high-quality prompts. Additionally, we also observed that changing the prompt fusion method did not compensate for the negative impact of prompt quality. These results demonstrate that prompt selection significantly impacts visual incontext learning, and prompt fusion is also an independent factor that affects visual in-context learning. Prompt fusion. The purpose of prompt fusion is to combine the prompt and query image into a new image, which is then fed into a large-scale visual model for prediction. Previous work followed the combination method shown as A1 in Figure 2 (b). However, there are a total of eight possible sub-images arrangements while ensuring that the prompt pair is adjacent. Next, we compared these eight arrangements while using the same prompt selection methods to evaluate the impact of different arrangements on visual context learning. The results are presented in Table 3, indicating that variations
Prompt selection
Domain shift
Prompt fusion
Fold-0
Fold-1
Fold-2
Fold-3
Means
Image-level
Pascal Ñ Pascal
34.70
35.92
32.92
31.12
33.53
COCOÑ Pascal
33.05
34.91
28.92
30.62
31.88p´1.65q
Pascal Ñ Pascal
41.07
41.32
38.14
36.44
39.24
COCOÑ Pascal
39.31
40.17
34.80
35.75
37.35p´1.89q
Pixel-level
Pascal Ñ Pascal
36.42
38.47
34.56
34.12
35.89
COCOÑ Pascal
33.94
37.45
32.45
33.21
34.26p´1.62q
Pascal Ñ Pascal
42.48
43.34
39.76
38.50
41.02
COCOÑ Pascal
40.13
42.14
37.84
38.52
39.80p´1.22q
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fa9a/fa9a8655-8057-43f9-a590-903acda0b1b0.png" style="width: 50%;"></div>
Figure 4: Ablation study of thresholds in ensemble strategy. The denominator and numerator of the threshold indicate the complete count of arrangements and the minimum count needed to establish the final outcome, correspondingly. in the arrangement of sub-images have varying effects on visual in-context learning. We hypothesize that the reason for this is the variation in the knowledge representation in different positions of the large-scale model. To substantiate this, we provide the visualization results of different arrangements, as depicted in Figure 3. The visualization outcomes demonstrate that it is arduous to establish the best arrangement for each query image. Moreover, a solitary sub-images arrangement in visual in-context learning fails to leverage the complete potential of knowledge present in large-scale visual models. Hence, Prompt-SelF proposes a straightforward ensemble strategy. We will discuss the ensemble strategy in prompt fusion in the following section. Ensemble strategy. To fully exploit the knowledge in large-scale visual models, we propose a simple ensemble strategy. This strategy integrates the prediction results of eight different arrangements and uses a simple voting strategy to determine the final result. Table 3 presents the experimental results, which demonstrate that the ensemble strategy can significantly enhance the performance of visual in-
context learning. This suggests that the ensemble strategy is effective in leveraging the diverse knowledge captured by the large-scale visual model and leads to more accurate predictions. Furthermore, we observed that the prompt selection strategy does not affect the success of the ensemble strategy. The visualizations presented in Figure 3 illustrate that the ensemble’s prediction outcomes can achieve more comprehensive object segmentation. We have set a decision threshold in our ensemble strategy to determine the final result. For instance, when the threshold is 2/8, the pixel will be classified as belonging to a certain class if at least two of the eight predictors predict the same class for the pixel. Next, we conducted ablation study on different thresholds to determine the optimal threshold value for Prompt-SelF. The experimental results are shown in Figure 4, which provides insights into the effect of varying threshold values on the performance of our ensemble strategy. Our experiments reveal that the optimal threshold value for the majority voting ensemble strategy is 4{8, as it provides the best performance on segmentation tasks. Therefore, in this paper, we set the threshold for ensemble to 4{8 by default. Alleviating domain shift. Domain shift is one of the key issues in visual in-context learning, as it can significantly impact the performance of the model. To alleviate the issue of domain shift, it is important to improve the robustness of the model to ensure its performance remains consistent and accurate in different environments. One approach to evaluating the robustness of a model is to test it on datasets that have different data distributions than the training dataset. We then investigate the robustness of prompt-SelF by switching the prompt dataset. Table 4 presents the experimental results, where ”PascalÑ Pascal” denotes that both the query and prompt datasets are from Pascal, ”COCOÑ Pascal” denotes that the prompt dataset is from MSCOCO [19] and the query dataset is from Pascal. The
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/586a/586a7db5-8d17-4bf9-bc75-f3ca249ee868.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5: Visualization results of more prompts and ensemble. Increasing the number of prompts can significantly increas performance. Note that the red box in the figure is the prediction result, and the blue box is the prompt pair.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2c3f/2c3fa070-52fe-4dc3-b7e0-6f8a972912d3.png" style="width: 50%;"></div>
Figure 6: Ablation study of different methods on introduce more prompts. pre means the method in VP [3] and VPR [41]. random and img respectively represent the prompt selection method of random and image-level retrieval. experiment results indicate that the pixel-level retrieval method can better mitigate the domain shift issue than the image-level retrieval methods. Moreover, in the ”COCOÑ Pascal” scenario, prompt-SelF achieved the best performance, indicating its robustness. The results indicate that prompt selection and prompt fusion are crucial elements that impact the robustness of visual context learning, and optimizing them can complementarily enhance the overall performance and robustness of the model.
Figure 6: Ablation study of different methods on introduce more prompts. pre means the method in VP [3] and VPR [41]. random and img respectively represent the prompt selection method of random and image-level retrieval. experiment results indicate that the pixel-level retrieval method can better mitigate the domain shift issue than the image-level retrieval methods. Moreover, in the ”COCOÑ Pascal” scenario, prompt-SelF achieved the best performance, indicating its robustness. The results indicate that prompt selection and prompt fusion are crucial elements that impact the robustness of visual context learning, and optimizing them can complementarily enhance the overall performance and robustness of the model. More prompts. The previous method [3, 41] to increase the number of prompts involved creating a larger grid by dividing the 224 ˆ 224 space into 16 grids to obtain up to 7 prompt pairs. However, this approach significantly reduces the resolution of sub-images, which reduces the effectiveness of each prompt and negatively impacts visual incontext learning. In contrast, we propose a straightforward
approach of combining the results from different prompts. Specifically, we sum the multiple prediction results and set a threshold of 0.5 to determine the final prediction result. Next, we conducted a comparison between the previous method and our proposed method on the impact of increasing the number of prompts on visual in-context learning. For this comparison, we selected two prompt selection methods: random and image-level retrieval, and the prompt fusion method A1. The results are presented in Figure 6, which shows that increasing the number of prompts positively affects visual in-context learning for all methods. However, previous methods significantly decrease the resolution of sub-images, resulting in 7 prompt pairs yielding worse results than our method with only 1 prompt. The visualization results in Figure 5 demonstrate that different prompts can have a significant impact on the prediction results. When a bad prompt is introduced, it negatively affects the prediction result. However, with the ensemble strategy, the negative impact of the bad prompt can be alleviated, leading to more robust prediction results.
# 5. Conclusion
This paper investigates the key factors affecting visual in-context learning, namely prompt selection and prompt fusion. We demonstrate that high-quality prompts can significantly improve the performance of visual in-context learning, and prompt fusion is essential for utilizing the full knowledge in large-scale visual models. Based on the above insights, we propose a simple framework, promptSelF, which utilizes pixel-level retrieval for prompt selection and ensemble prediction results from multiple prompt fusions to fully exploit the knowledge in the large model. After conducting a large number of experiments, we have demonstrated the effectiveness of prompt-SelF in addressing the two factors of prompt selection and prompt fusion that affect visual in-context learning. Notably, the impact
of prompt selection and prompt fusion on visual in-context learning is independent, and improving the ability of visual in-context learning requires optimizing both of these factors.
# References
Table 5: Ablation study of the input-label mapping. and indicate, respectively, whether the label corresponds to the prompt image. The best results based in-context learning are show in bold.
prompt selection correct label Fold-0 Fold-1 Fold-2 Fold-3
Means
random
27.49 30.21 25.88 24.06
26.91
22.16 26.24 22.45 19.60 22.61p´4.30q
image-level
34.70 35.92 32.92 31.12
33.53
27.81 31.37 26.64 25.76 27.90p´5.63q
# Appendix
# A. More Factors
In this section, we undertake a comprehensive analysis of additional factors that impact the performance of visual in-context learning.
# A.1. The input-label mapping
The input-label mapping for prompts plays a crucial role in NLP’s in-context learning [4, 25]. Prompts serve as the model’s input, and the effectiveness of the model’s understanding and processing of these inputs is largely determined by the quality of the input-label mapping. A welldesigned input-label mapping can enable the model to encode the inputs accurately and produce precise outputs. Thus, in order to achieve optimal performance, it is necessary to meticulously design and optimize the input-to-label mapping for prompts. Similarly, we strongly contend that the input-label mapping of prompts plays a critical role in visual in-context learning. As a result, we conducted an experiment to substitute the correct label of the prompt with an incorrect one, utilizing the prompt selection methods of random and image-level retrieval, as well as the arrangement method of A1. Table 5 presents the experimental results, which demonstrate the significant impact of inputlabel mapping on visual in-context learning. These findings indicate that an accurate input-label mapping can assist the model in comprehending new tasks with greater precision.
# A.2. Feature Extractor in Prompt Selection
The feature extractor used in prompt selection aims to extract features from both the prompt image and query image, and subsequently identify the most appropriate prompt for the query image based on these features. Therefore, it is imperative that the feature extractor can effectively convey semantic information across different regions of the image. In order to achieve this, we conducted experiments utilizing different feature extractors for both the pixel-level prompt selection and the prompt fusion method of A1. In order to conduct a comprehensive comparison, we selected three commonly-used feature extractors - VIT [8], Beit [2], and CLIP [27] - and the experimental results are presented in
Table 6: Ablation study of the feature extractor in Prompt Selection. The best results based in-context learning are show in bold.
Prompt Selection
Weight
Fold-0
Fold-1
Fold-2
Fold-3
Means
pixel-level
VIT
35.66
37.71
33.31
33.50
35.05
Beit
34.55
37.23
32.66
32.83
34.32
CLIP
36.42
38.47
34.56
34.12
35.89
Table 6. The results indicate that the multi-modal feature extractor contains a greater amount of semantic information and can more effectively accomplish the task of prompt selection.
# A.3. The Large-scale Visual Model
Undoubtedly, enhancing the capacity of large-scale visual models can greatly enhance visual in-context learning. Nevertheless, the majority of current visual largescale models [8, 6, 2] lack the capacity for visual in-context learning. This is primarily due to the fact that these largescale models are trained to comprehend the content of individual images, and therefore, are unable to directly capture the connections between different images. In light of this, prior research [3, 35] has endeavored to design largescale visual models equipped with the capacity for visual in-context learning. However, given the nascent nature of visual in-context learning, current research is insufficient to facilitate experiments aimed at assessing the impact of different large-scale visual models on in-context learning. Consequently, we anticipate that future research will produce more robust and insightful work on the subject, thus enabling a more comprehensive understanding of the influence of various large-scale visual models on visual incontext learning.
# B. More Visualization Results
To better visualize the benefits of different prompt fusion methods and more prompts for visual in-context learning, we present additional visualization results. Specifically, Figures 7, 8, and 9 depict the impact of incorporating 3, 5, and 7 prompts, respectively, on the final outcome. Moreover, these visualizations enable us to observe the effect of different prompts on the prediction of a single query image. These results demonstrate the critical role of prompt selection in visual in-context learning and highlight how amalgamating multiple predictions can substantially enhance the accuracy of query image predictions. Furthermore, the impact of different prompt fusions is illustrated in Figure 10. This further confirms that the arrangement of prompts and query images has a significant effect on the prediction results. Summarizing these results can greatly improve the performance of visual in-context learning.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fe71/fe71197e-740f-4efc-8585-fa66b497f387.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 7: Visualization results of three prompts and ensemble. Increasing the number of prompts can significantly increase performance. Note that the red box in the figure is the prediction result, and the blue box is the prompt pair.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/da06/da06cde0-c114-4b91-92ba-1360cad8bae5.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8: Visualization results of five prompts and ensemble. Increasing the number of prompts can significantly increase performance. Note that the red box in the figure is the prediction result, and the blue box is the prompt pair.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0688/0688c774-e6e4-4207-9195-297f1a0f6d9d.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9: Visualization results of seven prompts and ensemble. Increasing the number of prompts can significantly increase performance. Note that the red box in the figure is the prediction result, and the blue box is the prompt pair.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e5c7/e5c7ddcf-a003-4646-873f-7dfef6c78ebb.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10: Visualization results of different prompt fusion methods. A1-A8 represent different prompt fusion methods, where the blue box is the prompt pair, and the red box is the prediction result. The best results are show in bold.</div>
