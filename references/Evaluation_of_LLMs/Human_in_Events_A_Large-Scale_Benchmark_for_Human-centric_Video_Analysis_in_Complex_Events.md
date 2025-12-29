# HiEve: A Large-Scale Benchmark for Human-centric Video Analysis in Complex Events
Weiyao Lin · Huabin Liu · Shizhan Liu · Yuxi Li · Hongkai Xiong · Guojun Qi · Nicu Sebe
Abstract Along with the development of modern smart cities, human-centric video analysis has been encountering the challenge of analyzing diverse and complex events in real scenes. A complex event relates to dense crowds, anomalous individuals, or collective behaviors. However, limited by the scale and coverage of existing video datasets, few human analysis approaches have reported their performances on such complex events. To this end, we present a new large-scale dataset with comprehensive annotations, named Human-in-Events or HiEve (Human-centric video analysis in complex Events), for the understanding of human motions, poses, and actions in a variety of realistic events, especially in crowd & complex events. It contains a record number of poses (>1M), the largest number of action instances (>56k) under complex events, as well as one of the largest numbers of trajectories lasting for longer time (with an average trajectory length of >480 frames). Based on its diverse annotation, we present two simple baselines for action recognition and pose estimation, respectively. They leverage cross-label information during training to enhance the feature learning in corresponding visual tasks. Experiments show that they could boost the perXiv:2005.04490v6  [cs.CV]  13 Jul 
formance of existing action recognition and pose estimation pipelines. More importantly, they prove the widely ranged annotations in HiEve can improve various video tasks. Furthermore, we conduct extensive experiments to benchmark recent video analysis approaches together with our baseline methods, demonstrating HiEve is a challenging dataset for human-centric video analysis. We expect that the dataset will advance the development of cutting-edge techniques in human-centric analysis and the understanding of complex events. The dataset is available at http://humaninevents.org.
Keywords Complex events · Human-centric video analysis · Dataset and benchmark
# 1 Introduction
The development of smart cities highly relies on the advancement of fast and accurate visual understanding of multimedia [64,39,9]. To achieve this goal, many human-centered and event-driven visual understanding problems have been raised, such as human pose estimation [15], pedestrian tracking [12,44], and action recognition [55,49]. Recently, several public datasets (e.g., MSCOCO [33], PoseTrack [1], UCF-Crime [52]) have been proposed to benchmark the aforementioned tasks. However, they have some limitations when applied to real scenarios with complex events such as dining, earthquake escape, subway getting-off and collisions. First, most benchmarks focus on normal or relatively simple scenes. These scenes either have few occlusions or contain many easilypredictable motions and poses. Second, the coverage and scale of existing benchmarks are still limited. For example, although the UCF-Crime dataset [52] contains challenging scenes, it only has coarse video-level action
labels which may not be enough for fine-grained action recognition of human instance. Similarly, although the numbers of pose labels in MSCOCO [33] and PoseTrack [1] are sufficiently large for simple scenes with limited occlusions, these datasets lack realistic scenes containing crowded scenes and complex events. To this end, we present a new large-scale humancentric dataset, named Human-in-Events (HiEve), for understanding a hierarchy of human-centric information (motions, poses, and actions) in a variety of realistic complex events, especially in crowded and complex events. Among all datasets for realistic crowd scenarios, HiEve has substantially larger scales and complexity and contains a record number of poses (>1M), action labels (>56k) and long trajectories (with average trajectory length >480 frames). Compared with existing datasets, HiEve contains more comprehensive and larger-scale annotations in both generic and complex scenes, making it more adequate to develop new human-centric analysis techniques and evaluate them in realistic scenes. Table 1 provides a quantitative comparison of the HiEve dataset with related datasets in light of their nature and scale. One main feature of our HiEve dataset is the hierarchical and diverse information of human annotations under unified crowd scenes, which encourages to accomplish multiple human-centric visual tasks by integrating cross-annotation information. To make a tentative validation of this property, we explore combining the pose and action label in HiEve to present two simple baselines (1) a pose-aware action recognition algorithm and (2) an action-guided pose estimation algorithm. Specifically, the former promotes video action learning by encouraging the video feature to predict pose-aware motion patterns, while the latter refines the pose representation with action category prior knowledge. Experiments demonstrate that they can boost the performance of existing state-of-the-art pipelines on our HiEve dataset. We hope this exploration will foster further research in video understanding with diverse annotations of HiEve. Additionally, we build an online evaluation server available to the whole community in order to enable timely and scalable evaluation on the held-out test videos. We also evaluate existing state-of-the-art solutions on HiEve to benchmark their performance and analyze the corresponding oracle models, demonstrating that HiEve is challenging and of great value for advancing human-centric video analysis. In summary, we make the following main contributions:
2.1 Multi-object Tracking Datasets
Different from single-object tracking, multi-object tracking (MOT) does not solely depend on sophisticated appearance models to track objects in frames. In recent years, there is a corpus of datasets that provide multiobject bounding-box and track annotations in video sequences, which have fostered the development of this field. PETS [17] is an early proposed multi-sensor video dataset, it includes annotation of crowd person count and tracking of an individual within a crowd. Its sequences are all shot in the same scene, which leads to relatively simple samples. KITTI [18] tracking dataset features videos from a vehicle-mounted camera and focuses on street scenarios, it owns 2D & 3D boundingboxes and tracklets annotations. Meanwhile, is has a limited variety of video angles. The MOT-Challenge dataset [40] is the most widely-used benchmark for MOT tasks, primarily focusing on evaluating tracking performance in crowded environments. While the MOT-series (MOT-17, 19, and 20) datasets have fostered the development of various tracking algorithms, they exhibit certain shortcomings for current real-world applications. A key limitation of the MOT-Challenge dataset is its relatively narrow scope, as it predominantly features scenes with pedestrians in urban settings. This lack of diversity in scene types and events may hinder the generalization of tracking algorithms to more complex and varied scenarios. Compared to the latest MOT-20 [12] dataset, our HiEve dataset collects videos from various real-world scenes (12 scenes in total) and includes more complex events, such as fights, earthquakes, and robberies, presenting more significant challenges for realworld MOT tasks. Furthermore, as shown in Table 1, HiEve has longer video and track lengths than MOT20. Most importantly, HiEve offers a broad range of annotations, encompassing dense human poses, object tracking, and actions, making it a more comprehensive dataset for human-centric understanding tasks.
– We collect a new large-scale video dataset HiEve under various realistic complex events (e.g., dining, earth-quake escape, collision) for human-centric video analysis.
– Our HiEve provides a wide range of human annotations (track, pose, action) to enable analysis on various visual tasks, such as multi-object tracking, pose estimation, and action recognition. – By virtue of the diverse annotation in HiEve, we propose two enhanced baselines for action recognition and pose estimation, respectively. Experiments on them demonstrate the correlation between different types of human annotations could further boost the state-of-the-art methods on HiEve.
# 2 Related benchmarks and Comparison
Dataset
# pose
# box
# traj.(avg)
# action(class)
# total length (avg)
pose track
surveillance
complex events
MSCOCO [33]
105,698
105,698
NA
NA
NA
×
×
×
MPII [2]
14,993
14,993
NA
25,000
NA
×
×
×
CrowdPose [31]
∼80,000
∼80,000
NA
NA
NA
×
×
×
PoseTrack [1]
∼267,000
∼26,000
5,245(49)
NA
2,750s(2s)
√
×
×
MOT16[40]
NA
292,733
1,276(229)
NA
463s(33s)
×
√
×
MOT17
NA
901,119
3,993(226)
NA
1,389s(66s)
×
√
×
MOT20 [12]
NA
1,652,040
3457(478)
NA
535s(67s)
×
√
×
Avenue [36]
NA
NA
NA
37(37)
1,225s(33s)
×
√
×
UCF-Crime [52]
NA
NA
NA
1,900(13)
128h(4s)
×
√
√
UCF101-24 [51]
NA
NA
NA
44,716(24)
∼4h(7s)
×
×
×
JHMDB-21 [30]
NA
NA
NA
31,838(21)
∼5h(9s)
×
×
×
HiEve (Ours)
1,099,357
1,302,481
2,687(485)
56,643(14)
1,839s(57s)
√
√
√
Table 1: Comparison between HiEve and existing datasets. “NA” indicates not available. “∼” denotes approximated value. For “traj.(avg)”, the “traj.” means trajectory and “avg” indicates average trajectory length. For “action(class)”, “action” means action instance and “class” indicates the number of action category. For “total length (avg)”, “total length” denotes the total length of all videos while the “avg” means the average video length.
# 2.2 Pose Estimation and Tracking Datasets
lected from movies and online resources. The Kinetics [7] dataset, with 400/600/700 action categories and more than 300,000 clips, is currently one of the largest video datasets for action recognition. Researchers often use this dataset to provide prior action knowledge for downstream video backbones and tasks. The EpicKitchen [11] and Something-Something [22] datasets are unique in that they focus on human-object interactions and first-person visions. Epic-Kitchen collects videos in a daily kitchen setting, while Something-Something focuses on videos that record people performing actions with certain objects. Both of them pose new and significant challenges for action recognition. To recognize the anomaly actions, Avenue [36] and UCF-Crime [52] are further proposed. Aveue collects 37 videos with abnormal events from the campus, while UCF-Crime annotates 13 anomalies in real-world surveillance videos, such as fighting, accident, and robbery. However, most of the above datasets are collected from either less realistic drama scenes or uncrowded scenarios.
Human pose estimation in images has made great progres over the last few years. For single-person pose estimation, LSP [28], FLIC [47] are the two most representative benchmarks, the former focuses on sports scenes while the latter is collected from popular Hollywood movie sequences. Compared with LSP, FLIC only labels 10 upper body joints and owns a smaller data scale. WAF [14] is the first to establish a benchmark for multi-person pose estimation with simplified keypoint and body definition. Then, MPII [2] and MSCOCO [33] datasets were proposed to further advance the multiperson pose estimation task by their diversity and difficulty in the human pose. In particular, MSCOCO is regarded as the most widely used large-scale dataset with 105698 pose annotations in hundreds of daily activities. To evaluate the performance under crowded scenes, Crowdpose [31] selects crowded images from MPII, MSCOCO to form a subset for pose estimation under crowded scenes. Therefore, the scale of Crowdpose dataset is limited. Taking the tracking task into consideration, PoseTrack [1] builds a new video dataset which provides multi-person pose estimation and articulated tracking annotations. Compared with them, our HiEve provides more realistic scenarios for both pose estimation and pose tracking. Meanwhile, HiEve is dominated by crowded scenes, which is more challenging for current pose estimation algorithms.
The benchmarks mentioned above follow the regular video-level action recognition task, where each video is assigned only one action label. However, the action recognition task in our HiEve dataset focuses on a more complicated action detection task, where both the location and category of the action need to be recognized for each object. The previous action detection benchmarks, UCF101-24 [50,29] and JHMDB-21 [50,29], are more similar to our setting. The UCF101-24 dataset is a subset of the UCF101 [51] dataset, focusing specifically on 24 human action classes related to sports and human movements. This subset is annotated with spatiotemporal bounding boxes, making it suitable for the evaluation of both action recognition and action detection tasks. Similar to UCF101-24, the JHMDB-21 is a subset of the large HMDB [30] dataset, which selects 21 classes and annotate them with spatiotemporal bounding boxes. As shown in Table 1, compared to them, we contains a
# 2.3 Action Recognition Datasets
In recent years, action recognition has emerged as a popular research topic in computer vision. Meanwhile, the availability of large-scale video datasets has greatly facilitated the development of this field. UCF101 [51] and HMDB-51 [30] are two widely used datasets, which consist of various sports videos and daily activities col-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/600d/600d3742-5a67-4f56-83e3-523c5c379d13.png" style="width: 50%;"></div>
larger number of action instances and a much longer average video length (57s). Most importantly, the actions in HiEve dataset are performed under complex scenarios or abnormal events, which poses more significant challenges for action detection.
# 2.4 HiEve vs. other datasets
In summary, the related datasets mentioned above have served the community very well, but now they are confronting several limitations: (1) Most of them are focusing on normal or simple scenes (2) Their coverage and scales are limited. (3) They only contain a single aspect of human annotation (pose, track or action). Overall, compared with these datasets, our dataset has the following unique characteristics:
<div style="text-align: center;">Fig. 1: An example of the collection workflow of our HiEve dataset under street scene, where each scene contains videos captured at different sites with different types of events happening.</div>
Fig. 1: An example of the collection workflow of our HiEve dataset under street scene, where each scene contains videos captured at different sites with different types of events happening.
collection process (including participants recruitment, video shooting, and online video collection) follows the guidance of the IRB review of our institute, which ensures HiEve doesn’t violate individual privacy or other legal or ethical standards. In total, our video sequences are collected from 12 different scenes: airport, dining hall, factory, lounge, stadium, jail, mall, square, school, station and street. Fig. 6 shows the frame number of different scenes in HiEve. As illustrated in the workflow in Fig. 1, for each scene, we keep several videos captured at different sites and with different types of events happening to ensure the diversity of scenarios. Moreover, data redundancy is avoided through manual checking. Finally, 32 real-world video sequences in different scenes are collected (with 10 videos obtained by ourselves and 22 videos collected from online repositories), each containing one or more complex events. These video sequences are split into training and testing sets of 19 and 13 videos. Both our own collected videos and online resources videos have a roughly sixty-forty split in training and testing. The detailed training-setting split as well as the detailed information of each video (including FPS, resolution, frame number, and source) can be found in http://humaninevents.org/data.html Annotation We manually annotated the HiEve dataset by cooperating with a professional annotation company, which owns experienced data annotators and has provided annotation services to many well-known benchmarks. All the data are labeled under a standard procedure to ensure their quality. The annotation procedure is as follows: First, we annotate poses for each person in the entire video. Different from PoseTrack and COCO, our annotated pose for each body contains 14 keypoints
In a nutshell, our HiEve contains more comprehensive and larger-scale annotations in various complex-event scenes, making it more capable of evaluating the humancentric analyzing techniques in realistic scenes.
# 3 The HiEve dataset
3.1 Collection and Annotation
Collection We start by selecting several crowded places with complex and diverse events for video collection. The videos are collected from two sources. The first part of the videos was obtained by ourselves where the consents of participants were obtained in advance. The second part of the videos was collected from online repositories such as YouTube. We include them in our dataset according to the guidance of Fair use on YouTube. We also have verified that all personally identifiable information (e.g., faces) was blurred in these videos and cannot be used to identify a specific subject. Note that the video
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1f99/1f998a74-bbd4-4c72-bc02-4dcea89f312e.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 2: Samples of different actions from our training set and testing set.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e00b/e00ba0cc-c090-4f11-8f79-02578bf7a5cf.png" style="width: 50%;"></div>
double-checked to ensure their quality. Specifically, there are two groups of humans to conduct data annotation. All videos are first sent to one group for the 1st round of labeling following the standard annotation procedure. After the 1st round of annotation, the labeled data are then sent to another group for double-checking. During the double-check, annotations of each sample will be evaluated with a confidence score (value from 0 to 10), which indicates the confidence of labeling. Then, data with less than 9 confidence scores will be sent back to the forehead group for the 2nd round annotation. We repeat the above process until all annotations satisfy the rule of confidence score. Moreover, we set a maximum iterations (iter=4 in our annotation) for correcting the annotation cross-check process. It should be noted that in order to maintain the completeness and consistency in the annotation results for all objects in a scene, we annotate both visible and invisible keypoints & bounding boxes. For invisible keypoints and boxes, we infer their location from the motion cues from previous frames or by observations, and assign them with an additional ‘invisible’ label. However, during the performance evaluation stage, we only evaluate performances based on the visible keypoints & bounding boxes, while the ‘invisible’ keypoints & bounding boxes are not included. This can make our evaluation results more accurate and reliable.
<div style="text-align: center;">Fig. 3: (a) Keypoints definition (b) Example pose and bounding-box annotations from our dataset.</div>
(Fig. 3a): nose, chest, shoulders, elbows, wrists, hips, knees, ankles. Specially, we skip pose annotation which falls into any of the following conditions: (1) heavy occlusion (2) area of the bounding box is less than 500 pixels. Fig. 3b presents some pose and bounding-box annotation examples. Second, we annotate actions of all individuals every 20 frames in a video. For group actions, we assign the action label to each group member participating in this group activity. In total, we defined 14 action categories: walking-alone, walking-together, running-alone, running-together, riding, sitting-talking, sitting-alone, queuing, standing-alone, gathering, fighting, fall-over, walking-up-down-stairs, crouching-bowing. Fig. 2 shows some samples of different actions in HiEve. Third, In order to guarantee the quality of our annotation results, we also conduct a temporally sequential annotation process. Specifically, we inherit all annotations from the previous frame and then update the annotations according to object appearances in the current frame. This process can both maintain high temporal consistency in the annotation results and greatly reduce the annotation burden at the same time. Finally, all annotations are
# 3.2 HiEve Statistics
Our dataset contains 32 video sequences mostly longer than 900 frames. Their total length is 33 minutes and 18 seconds. Table 1 shows the basic statistics of our HiEve dataset: It contains 49,820 frames, a record number of poses (1,099,357), the largest number of action instances (56,643) under complex events, as well as one of the largest numbers of trajectories (2,687) lasting for longer time (with an average trajectory length of 485 frames).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2e24/2e24ac97-6da1-487c-8fb7-6f9c0f041226.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7865/7865bdb5-6985-48b7-9bab-58716a37e062.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 4: The classification of events. They are divided into three event categories.</div>
<div style="text-align: center;">Fig. 5: The distribution of events. Different colors represent different kinds of events.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ba12/ba12f07d-7882-49aa-9f45-0453e448512c.png" style="width: 50%;"></div>
<div style="text-align: center;">(b) MOT20</div>
<div style="text-align: center;">Fig. 7: The distribution of the number of people per frame in MOT17, MOT20, PoseTrack and  scenes in HiEve dataset owns more people.</div>
Fig. 7: The distribution of the number of people per frame in MOT17, MOT20, PoseTrack and HiEve dataset. The scenes in HiEve dataset owns more people.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1f33/1f33ddfb-b669-4326-8d74-acafd2c9ca76.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9bee/9bee9a66-105f-4148-9bbe-51438c198712.png" style="width: 50%;"></div>
Fig. 8: Number of disconnected and continuous tracks in training set.
<div style="text-align: center;">Fig. 8: Number of disconnected and continuous tracks in training set.</div>
Fig. 9: CrowdIndex distributions of MPII, MSCOCO, CrowdPose, and our HiEve dataset. MSCOCO is dominated by uncrowded images. while HiEve dataset pays more attention on crowded cases.
# To further illustrate the characteristics of our dataset, we conduct the following statistical analysis.
we further grouped these events into 3 categories: complex emergency event, complex daily event, and simple daily event. In this way, we can construct the relationship between action, event, and category with a bottom-up manner, where each event may contain multiple actions, and each event category includes multiple event types (cf. Fig. 4). This hierarchical structure also allows for better statistical analysis of our HiEve dataset. We first present the number of poses, objects, and tracks
First, we analysis some statistic information across different events. In terms of video content, we could group our video sequences into 11 events: fighting, quarreling, accident, robbery, after-school, shopping, gettingoff, dining, walking, playing and waiting. Each event contains different amount of participants and action types. Then, according to the complexity of these events,
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/20b2/20b24f88-c80d-4a2a-a8f6-fb9446370569.png" style="width: 50%;"></div>
Fig. 6: The frame number distribution of different scenes in HiEve dataset.
<div style="text-align: center;">(c) PoseTrack</div>
<div style="text-align: center;">(d) HiEve</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1c81/1c819d5e-d1ff-4e47-8e66-6c8d79b80030.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6882/6882f651-9a63-4a53-b231-1ec462a453d6.png" style="width: 50%;"></div>
Fig. 11: The distribution of the length of track in PoseTrack and HiEve dataset.
<div style="text-align: center;">Fig. 10: The number of tracks, objects and poses in events. Different colors represent different kinds of events.</div>
<div style="text-align: center;">Fig. 11: The distribution of the length of track in PoseTrack and HiEve dataset.</div>
while other benchmarks are dominated by uncrowded ones. This characteristic enables our HiEve to comprehensively evaluate various pose estimation methods, ranging from simple cases to hard crowded scenes. Moreover, we need to clarify that the CrowdPose dataset is carefully selected from three public datasets (MSCOCO, MPII, and AI Challenge) according to the CrowdIndex. In this way, it has a near-uniform distribution of CrowdIndex. On the contrary, our HiEve dataset is a newly collected large-scale dataset rather than a selected subset of available benchmarks. Fourth, we analyze the ratio of disconnected human tracks in our dataset. Disconnected human tracks are defined as trajectory annotations where the bounding boxes are not available on some frames due to: (1) One object temporally moves out of the camera view and moves back sometime later. (2) One object is severely occluded by foreground objects or certain obstacles for a long time so that annotators can not assign an approximate bounding box to it (as exemplified in Fig. 14). It is noticeable that in datasets like PoseTrack [1], the reappearance of one individual in the scene is considered as the start of a new trajectory instead of the continuation of the original track before disappearing, in this manner these datasets will contain more tracks with shorter endurance (as reflected in Fig. 11). In contrast, in HiEve we assign the tracks before and after disappearing with the same ID, so as to encourage algorithms which can properly handle long-term re-identification. The numbers of disconnected and continuous tracks in the training set are reported in Fig. 8. The statistical results show that the proportion of disconnected tracks is non-negligible supporting algorithms which could handle complex cases and crowded scenes. Finally, the distribution of all action classes in our dataset is shown in Fig. 12 and could be regarded as a long-tailed sample distribution. Fig. 13 demonstrates the complex events in our dataset have more concurrent
for the above 3 events in Fig. 10. From this figure, we can see that (1) In our HiEve dataset, complex events (including complex emergency and complex daily) contain more human-centric instances (i.e., tracks, objects, and poses) compared to simple events. (2) Among the three event categories, complex daily events exhibit the largest number of poses and objects. Meanwhile, complex emergency events also have a considerable number of poses and objects compared to simple daily events. Moreover, Fig. 5 presents the average frame number of each event category. It can be seen that both the complex daily event and complex emergency event contain a considerable number of video frames in our HiEve dataset, which further indicates that our dataset is dominated by complex events. All these observations demonstrate the significant challenges posed by our dataset. Second, we present the number of people per frame in our dataset in Fig. 7 demonstrating that the scenes in our video sequence have more people than MOT17 and PoseTrack [1], making our tracking task more difficult. Although MOT-20 [12] collects some video sequences with more people (up to 141 people), it only covers limited scenarios and human actions. Third, we adopt the Crowd Index defined in Crowdpose [31] to measure the crowding level of our dataset. For a given frame, its Crowd Index(CI) is computed as:
(1)
where n is the total number of persons in this frame. N a i denotes the number of joints from the ith human instance and N b i is the number of joints located in bounding-box of the ith human instance but not belonging to the ith person. We evaluate the Crowd Index distributions of our HiEve dataset and the pose dataset MSCOCO [33], MPII [2], and CrowdPose [31]. Fig. 9 shows that our HiEve dataset pays more attention to crowded scenes
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ec71/ec71126a-0617-4786-9609-7d599a2805b7.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bf2d/bf2d989b-a0b2-45b6-9b89-80036be6dab5.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 13: The distribution of the number of concurrent action in HiEve dataset. Different colors represent different kinds of events.</div>
<div style="text-align: center;">Fig. 12: Sample distribution of all action classes in the HiEve dataset. Note that present the log10 of number of samples for a better visualization.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3124/31246556-31c2-48d5-afbd-849902d569fe.png" style="width: 50%;"></div>
our HiEve dataset also includes the novel HOTA [37] (Higher Order Tracking Accuracy) metric for evaluating MOT performance. HOTA is a comprehensive metric that considers various aspects of multi-object tracking, such as detection, localization, identity preservation, and temporal consistency. We believe that the incorporation of these metrics will provide a more accurate and reliable evaluation of tracking algorithms on our dataset. Besides, in order to evaluate how algorithms perform on tracks with disconnected parts, we design a weighted MOTA (w-MOTA) metric. This metric is computed in a similar manner as MOTA except that we assign a higher weight γ to the ID switch cases happening in disconnected tracks, consequently the metric can be formulated as w-MOTA = 1−(Nfp+Nfn+Nsw+(γ−1)Nsw−dt)/Ngt where Nfp and Nfn are the number of false positive and false negative, Nsw is the total times of ID switch, Nsw−dt is the ID switch times happening in disconnected tracks and Ngt is the number of bounding boxes in annotations.
<div style="text-align: center;">Fig. 14: Examples of disconnected tracks (highlighted with bounding box)</div>
events, which means that the complexity and difficulty of identifying behaviors in such scenes will increase. Overall, these statistics further prove that HiEve is a large-scale and challenging dataset dominated by complex events.
# 4 Task and Metric
With the collected video data and available annotations, HiEve poses four tasks for the evaluation of video analysis algorithms. For each task, we adopt some widely used metrics. Meanwhile we also design some new metrics to measure the performance on crowded and complex scenes.
# 4.2 Multi-person pose estimation
4.2 Multi-person pose estimation
This task aims to estimate specific keypoints on human skeleton. Compared with MPII Pose and MSCOCO Keypoints, our dataset involves more real-scene pose patterns in various complex events. We adopt Average Precision (AP@α) for measuring multi-person pose accuracy. The evaluation protocol is similar to DeepCut [43], if a pose prediction has the highest PCKh [2] with a certain ground-truth, then it can be assigned to the ground truth. Unassigned predictions are counted as false positives. α is the specific distance threshold
# 4.1 Multi-person tracking
This task is proposed to estimate the location and corresponding trajectory of each identity throughout a video. Traditional metrics MOTA, MOTP [40], ID F1 Score, ID Sw [46], and ID Sw-DT are selected to perform evaluation. Apart from these traditional metrics,
in complex scenes. The frame mAP value calculated with these assigned weights is called weighted framemAP (wf-mAP). Similarly to f-mAP@avg, we also report wf-mAP@avg as an overall measurement of wf-mAP.
for computing PCKh. We take the average value of AP@0.5, AP@0.75, and AP@0.9 as an overall measurement AP@avg. To further avoid the methods only focusing on simple cases or uncrowded scenarios in the dataset (although Fig. 9 has shown that our dataset contains a large number of crowded and complex scenarios), we will assign larger weights to a test image during evaluation if it owns: (1) higher Crowd Index (2) anomalous behavior (e.g. fighting, fall-over, crouching-bowing). To be specific, the weights for the tth frame in one video sequence can be formulated as:
# 5 Enhanced baselines with cross-annotation
The main advantage of HiEve is that it provides a wide range of human-centric annotations (tracking, pose, action), thus encouraging researchers to design visual algorithms by utilizing annotations from different types and aspect. This results in more comprehensive and accurate human-centric visual analysis system. To validate the above ability of HiEve, we design two simple baselines for action recognition and pose estimation tasks based on HiEve in this section.
where CIt is the crowd index on tth frame calculated via Equation 1, Nt denotes the number of categories of anomalous actions. During our evaluation, the coefficients c1, c2 are set to 2, 1 respectively. The values of AP calculated with assigned weights are called weighted AP (w-AP). Besides, we calculate w-AP@avg in the similar way with AP@avg.
# 5.1 Pose-aware action recognition
Skeleton-based action recognition [34,13,56] has attracted much attention due to its innate ability to represent motion. Current skeleton-based algorithms are predominantly developed and evaluated using benchmarks with simple scenes, such as the NTU-RGB-D [48], which comprises only one or two individuals per frame. However, achieving accurate pose estimation in complex scenarios, particularly those with heavy occlusion, proves exceedingly difficult, limiting the application of skeleton-based methods. Therefore, the potential of skeleton representation under complex scenes for action recognition still remains under exploration. Leveraging the diverse annotations in HiEve, we establish an enhanced baseline for RGB-based action recognition, where skeleton information is implicitly learned and integrated into the video representation. Its overall architecture is illustrated in Fig. 16. It is worth noting that, unlike traditional skeleton-based approaches, we don’t require human poses during inference. Compared to RGB-based methods, the only additional information we employ is the pose annotation of training data provided by HiEve. In summary, our proposed paradigm enables us to utilize pose information to facilitate action recognition while concurrently avoiding incorrect pose estimation under complex events.
# 4.3 Pose tracking
This task requires to provide temporally consistent poses for all people visible in the videos. Compared with PoseTrack, our dataset is much larger in scale and includes more frequent occlusions. Evaluation metrics MOTA and MOTP are also adopted in this task.
# 4.4 Action recognition
The action recognition task requires participants to simultaneously detect specific individuals and assign correct action labels to it on every sampled frame. Compared with AVA challenge [24], our action recognition track does not only contain atomic level action definition but also involves more interactions and occlusion among individuals, making recognition more difficult. We adopt the frame mAP (f-mAP@α), which is widely used to evaluate spatial action detection accuracy on a single frame, as the basic metric in this task. α is the specific IOU threshold to determine true/false positive. We report the mean value of f-mAP@0.5, f-mAP@0.6, and f-mAP@0.75 as an overall measurement of f-mAP, we denote this measurement as f-mAP@avg. Furthermore, considering the unbalanced distribution of the action categories in the data set, it is appropriate to assign smaller weights to the test samples belonging to dominated categories. In addition, we assign a larger weight to frames under crowded and occluded scenarios to encourage models to perform better
# 5.1.1 Multi-level motion prediction
The skeleton sequence contains more pose motion patterns, whereas the video representation includes more appearance-related motion information. Based on the various annotation for training data in HiEve, we can
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/112f/112fb861-e9f2-4cd0-b663-2275d7c5d31a.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c738/c7383284-0d8b-470f-b664-74ab51ebfd21.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/40d7/40d75f8e-83af-456d-a11a-7b5f3b99b619.png" style="width: 50%;"></div>
<div style="text-align: center;">Fig. 17: The framework of action-guided pose estimation enhanced baseline.</div>
leverage the pose annotation to facilitate the video feature learning by providing complementary pose-aware motion. Given a video clip, the video-based pipeline extracts video features fv ∈Rd using a video-specific model (e.g., I3D [7] , SlowFast [16]). Meanwhile, we could resort to human pose annotation provided by HiEve to generate a corresponding skeleton graph sequence G for this clip. Since the graph convolution network (GCN) has been widely used to process the skeleton sequences, we also resort to the GCN module proposed in STGCN [65] to extract the skeleton feature.
The video features across layers in CNN models contain multi-level and multi-grained action patterns, so it’s promising for them to learn a robust motion representation. Therefore, we use video features from multiple stages of the model to conduct this prediction. For each feature map ml ∈Rdl output by the 3D CNN model in stage-l, we predict the corresponding pose-aware motion vector by linear transformation:
  where W c l ∈Rd×dl and bc l are the parameters of direction prediction, while W c s ∈R1×dl and bs l belong to the length prediction. We aggregate multiple predictions from multi-level features by:
# fp = STGCN(G)
(2)
where fp ∈Rd indicates the skeleton graph feature output by GCN, which we can name pose-aware feature. To empower the video network to obtain pose-aware motion by itself, we design a multi-level motion prediction task for the video stream. It encourages the video network to predict the pose-aware motion representation using multi-level video features. Meanwhile, we find it beneficial to predict the direction f c p and length ∥fp∥of fp separately. The fp vector can be decomposed into its direction and length, so we can re-write it as:
� Moreover, we add a prediction loss term to encourage the predicted motion vector r to be close to the fp:
Finally, the predicted feature vector is concatenated with the video feature fv, which provides the video feature with complementary pose-specific motion patterns.
(3)
(4)
(5)
(6)
5.1.2 Implementation Details
5.2.1 Action-guided domain alignment
Some special location relationships between human keypoints tend to indicate a certain anomalous behavior. For example, as illustrated in Fig. 15, a human skeleton yielding a dense and horizontal keypoints distribution is usually associated with the ‘fall-over’ action. Vice versa, the action category can provide reliable prior knowledge on keypoints location. Moreover, the incorrect keypoints location could be revised by these knowledge. With this observation, we propose an action-guided domain alignment module (ADAM), where we regard the pose and action as information from two different domains. The ADAM aims at building a mapping between them, such that the two domains are close in feature space. Follow the framework of top-down pose estimation, the pose feature Fp of single person is extracted by a base convolution network. Then, an encoder E with a series of down-sample operations squeezes the pose feature into a latent feature f p l ∈Rd. To extract action information, we embed the one-hot action label vector ˆya of this person into a latent feature f a l ∈Rd through a linear transformation T. The above process could be formulated as:
The Slowfast-ResNet50 [16] is chosen as our backbone for video feature extraction. Moreover, we follow the official setting of SlowFast to keep the same temporal resolution at different stages of ResNet. Regarding the additional overhead introduced by our baseline, it only adds approximately 25% GFLOPs to the vanilla SlowFast (an increase from 65.7 GFLOPs to 84.6 GFLOPs). Faster-RCNN detector is used to detect persons during testing. L = 3 in our default setting and the feature maps output by stage-1, 2, 3 are globally pooled to form as the multi-level feature m1, m2, m3. the final feature dimension d = 2034. We uniformly sample 16 frames for each video and each input frame is cropped into 256 × 256 during training and inference. The total loss for training is defined as:
L = Lcls + Lpred
(7)
where the Lcls is the cross-entropy loss adopted in classification task. During inference, since the pose annotation is not available, we straightly use the predicted poseaware motion feature as the input for classifier.
f p l = E(Fp), f a l = T(ˆya), f p l , f a l ∈Rd
# f p l = E(Fp), f a l = T(ˆya), f p l , f a l ∈Rd
Then, an alignment loss is calculated between latent features from two domains, which encourages feature consistency between them by minimizing their distance in the latent space:
5.2 Action-guided pose estimation
Although skeleton-based action recognition has been well developed, only a few methods [62,27] paid attention to its reverse paradigm, i.e., how action prior can help pose estimation. Luckily, thanks to the diverse annotations of HiEve, we build a simple yet effective baseline method for pose estimation, which enhances the pose learning stream by prior knowledge of action. As shown in Fig. 17, the algorithm mainly comprises two modules: action-guided domain alignment module (ADAM) and pose refinement module (PRM) module, where ADAM aligns the feature representation between the domain of action and pose, while PRM utilizes the aligned feature to refine the pose estimation results. Compared to previous approaches that attempt to leverage the action knowledge to facilitate the pose estimation, our method offers several advantages: First, it is free from utilizing additional action predictors during inference, which is necessary for most previous methods [62,27]. Second, we only added negligible overhead to the pose estimation stream. Thirdly, our method can be easily extended to most current pose estimation algorithms. It’s worth noting that some approaches integrate pose and action learning into a multi-task learning framework [38] or a unified model. Different from them, our focus remains on the pose estimation task.
Lalign = MSE(f p l , f a l )
However, there exists some variance among human poses even though they belong to the same action category. Aligning all of them to the same action embedding is not ideal. Moreover, for each individual in a complex event, action spatial-context (e.g., group activity, occlusion, or interaction with neighbors) also affects its human pose. Therefore, apart from input individual on itself, we also consider action information from its neighboring area U(on) and person om, m = 1, 2, . . . , |U(on)|, om ∈U(on) in this area. Then, we can utilize the self-attention mechanism [54] to get an instance-specific action embedding by aggregating the spatial-context action information. Specifically, we first embed their relative geo-position as:
where EP is positional encoding operation proposed in Transformer [54], x, y, w, h are the center coordinates, width, and height of person bounding box. Then, combining the action category embedding with relative-position
(8)
 (9)
embedding, we calculate spatial-context action correlations as:
The HRNet [53] pretrained on COCO is chosen as our backbone for pose feature extraction training. The proposed modules are appended after the last stage of HRNet. Our Encoder and Decoder use the corresponding downsample and upsample architecture in U-Net, respectively. For training, the whole network is trained on the HiEve training set. For a fair comparison, same as we described in 6.2, we take the Faster-RCNN [45] as person detector. As the actions are annotated every 20 frames in HiEve, we utilize interpolation to create action category labels for all individuals in every frame. We set different learning rates for the backbone HRNet and our proposed modules, which are 1e-4 and 1e-3 respectively. In our experiments, we will show that our model gains the ability of mining potential action information to refine the poses. During training phase, the total loss for training is defined as:
(10)
where WK, WQ ∈Rdk×d are projection matrices. Specially, we only consider people om who satisfy |dmn|2 ≤ 4.5 in the neighboring area U(on). The spatial-context are aggregated into the individual action information embedding in a residual sum manner as:
f a l = f a l + � m∈U(n) ωmn · (WV · (f a l )m) ,
(11)
where WV is projection matrix. The updated action embedding f a l is finally provided for f p l to perform alignment (Equation 8).
5.2.2 Pose refinement
To further improve the quality of pose estimation, we design a refinement module based on the latent pose features, which comprises two head structures: spatial refinement head (SR) and channel-wise refinement head (CR). In pose estimation, the position of keypoints is reflected by the local responses in the spatial feature maps. Therefore, the SR intends to re-weight the spatial feature map by emphasizing specific skeleton position and suppressing inaccurate keypoints response. Corresponding to the encoder in ADAM, the SR applies an decoder, which consists of a series of up-sampling operations to output an attention mask α from fp:
where the Lreg is the traditional heatmap regression L2 loss. During inference, the action label embedding process is removed, and the proposed modules are connected with the last stage’s output of HRNet.
# 6 Experiments and results
# α = σ(W1 s(D(f p l )))
# Baselines
where W1 S ∈RN×N are the parameters of a depth-wise separable 9×9 convolution, the output attention map α implicitly contains the keypoints prior from actionspecific knowledge. On the other hand, inspired by the SENet [26], the CR aims at performing channel-wise feature recalibration in a global sense, where the per-channel summary statistics are utilized to selectively emphasis informative feature maps as well as suppress useless ones. To be specific, the latent feature passes through two fully-connected layers and a sigmoid activation to obtain an attention vector β for each channel
# β = σ(W2 c · δ(W1 cf p l ))
  where σ(·) and δ represent the sigmoid and ReLU functions respectively, W1 C ∈Rd×N and W1 C ∈RN×N refer two fully-connection layers. The channel-wise and spatial attention guidance is then applied to refine pose feature as ˆFp = Fp ⊗(1 + β ⊗α)
– DeepSORT [59]. Based on the SORT [4] algorithm, it extracts person appearance features by a pre-trained model, then simple nearest neighbor query is performed to track pedestrians. – MOTDT [8]. MOTDT tackles unreliable detection by selecting candidates from outputs of both detection and tracks. Besides, a new scoring function for candidate selection is formulated by an efficient R-FCN. – IOUtracker [5]. IOUtracker proposes a very simple and efficient tracking algorithm, which only leverages the detection results and designs an IOU strategy to improve the performance of multi-objective tracking. – JDE [58]. JDE Tracker is the first joint pipeline for simultaneous detection and tracking, which produce the object embedding to accosiate persons across frames. – FairMOT [68]. FairMOT is another joint detectiontracking pipeline, which focuses on addressing spatial misalignment with under an anchor-free manner.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d140/d1409709-9dcf-40cb-967e-9c03ee8b1494.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 2: Results of multi-person tracking baselines.</div>
– TPM [42]. TPM proposes a tracklet-plane matching process to model and reducing the interference from noisy or confusing object detections. – CenterTrack [69]. A simple but efficient method, which applies a detection model to a pair of images and detections from the prior frame.
Implementation Details Faster R-CNN [45] is used to obtain the public results of bounding-boxes firstly. In MOTDT and DeepSORT, we use the train set of HiEve and the ground truth to fine-tune the official deep models in these methods. Then, we evaluate them in the HiEve test dataset with the public detection results. The threshold of detections is set to be 0.2.
The results of these baselines are shown in Table 2 and Fig. 18. We can observe that all of their performances are not ideal. This is because our dataset has complex scenes and a large number of overlapping targets, making identification and tracking more difficult. IOUtracker [5] performs best on our dataset, while MOTDT [8] and DeepSORT [59] have relatively worse performance. Meanwhile, the joint detection-and-tracking solution JDE [58], CenterTrack, and FairMOT [68] also performs worse than the simple IOU Tracker. The reason is that HiEve contains numerous crowded scenes and occlusions, so it’s hard to extract discriminative features to distinguish different object instances.
Fig. 18: Visualized results of MOT baselines and the ground-truth (GT).
6.2 Multi-pose estimation
# Baselines
# Implementation Details
For the above top-down methods, we take the same detection results of Faster-RCNN [45] as their input. For all mentioned methods, we use their official codes to conduct implementation and experiments. Specifically, we download their public COCO pre-trained weights as initialization and further fine-tune them on our HiEve
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/14ca/14cac341-e5fa-43ea-bc80-dbb942335687.png" style="width: 50%;"></div>
<div style="text-align: center;">(c) GT</div>
Net and then performs keypoints matching for each individual. – DEKR [19] It learns to directly regress different keypoints with distinctive adaptive convolutions, which could disentangle the representation for keypoints and obtain ideal performance under bottomup paradigm. – RSN [6] It devises a residual steps network to learn delicate local representations by intra-level feature fusion. – HRFormer [67] It adopts the idea of multi-resolution parallel in DHRN [53] to the Transformer [54] architecture. – Ours. Our proposed action-guided pose estimation baseline.
Method
w-AP@avg
w-AP@0.5
w-AP@0.75
w-AP@0.9
AP@avg
AP@0.5
AP@0.75
AP@0.9
DHRN [53]
52.78
61.73
50.73
45.91
56.40
64.89
54.56
49.76
Simple Baseline [61]
50.51
59.90
47.90
43.74
54.44
63.56
52.19
47.59
HigherHRNet [10]
22.03
25.65
21.37
19.06
24.92
28.74
24.23
21.77
RSN [6]
52.25
63.34
49.75
43.65
55.46
66.23
53.24
46.92
DEKR [19]
47.46
56.47
44.87
41.04
49.42
58.07
47.09
43.10
HRFormer [67]
51.03
60.77
48.33
44.00
54.67
64.07
52.21
47.74
Action-guided pose estimation (Ours)
53.92
63.72
51.67
46.36
57.68
67.15
55.60
50.30
<div style="text-align: center;">Table 3: Results of multi-person pose estimation.</div>
Results and Analysis We present the evaluation results in Table 3 and the visualization results in Fig. 19. It can be observed that DHRN [53] performs best excluded our proposed method. Interestingly, the performance of recently proposed HRFormer [67] falls between Simple-Baseline and DHRN. The reason is probably that transformer-based networks tend to overfit the training set. In fact, the performance of HRFormer on the validation set began to degrade earlier than other methods when we perform finetune on HiEve dataset. For bottom-up based methods, the recently proposed DEKR [19] surpasses the HigherHRNet [10] by a significant margin. The reason may be that the DEKR obtained disentangled representation for different keypoints using adaptive convolutions, which contributes to distinguishing the occlusion of human bodies. It can also be noticed that our proposed action-guided pose estimation further boosted the performance of DHRN by 1.13 w-AP. The comparisons manifest that by introducing action category information, our proposed simple baseline with aligned features and pose refine mechanisms could generate more accurate keypoint locations in crowded scenes. The success of this simple baseline also proves that leveraging the diverse annotation in the HiEve dataset could improve pose estimation.
Fig. 19: Visualized results of pose estimation baselines and the ground-truth (GT).
# Implementation Details
In LightTrack, YOLO v3, Siamese GCN, and MobileNet are selected as the keyframe detector, ReID module, and pose estimator respectively. We use DeepMatching to extract dense correspondences between adjacent frames in PoseFlow. All weights of model inherit from pretrained models on MSCOCO [33].
# 6.3 Pose tracking
# Baselines
– PoseFlow [63]. It’s an efficient pose tracker based on flows and top-down approaches RMPE [15]. An online optimization framework is designed to build the association of cross-frame poses and form pose flows (PF-Builder). Then, a novel pose flow non-maximum suppression (PF-NMS) is designed to robustly reduce redundant pose flows and re-link temporal disjoint ones. – LightTrack [41]. LightTrack is an effective lightweight framework for online human pose tracking. It unifies single-person pose tracking with multi-person identity association.
# Results and Analysis
The performance comparison of these three methods is presented in Table 4. As expected, the flow-based algorithm PoseFlow achieves higher performance while LightTrack [41] mainly aims to strike a balance between speed and accuracy. The Fig. 20 shows the visualization results of them, PoseFlow is able to track more people than LightTrack, but they all face the issue of losing objects and bad keypoints localization in crowded scenes.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/846c/846c70af-843b-4cd1-92e5-62b19a74859b.png" style="width: 50%;"></div>
<div style="text-align: center;">(d) GT</div>
<div style="text-align: center;">(c) Ours</div>
Method
MOTA
MOTP
AP
RMPE + PoseFlow [63]
44.17
48.33
60.10
LightTrack [41]
27.44
55.23
29.36
Ours + PoseFlow
45.36
49.97
63.16
<div style="text-align: center;">Table 4: Results of pose tracking baselines.</div>
– Our method + PoseFlow. Based on the pose estimation results of our algorithm, we adapted PoseFlow method to conduct human pose tracking across frames.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/b64a/b64a5f67-345e-43c1-940d-9071ae8263ad.png" style="width: 50%;"></div>
<div style="text-align: center;">(d) GT</div>
Fig. 20: Visualized results of pose tracking baselines and the ground-truth (GT).
<div style="text-align: center;">Fig. 20: Visualized results of pose tracking baselines and the ground-truth (GT).</div>
Implementation Details For all baselines except for SlowFast [16], we adopt the RGB-I3D [7] network with Inception-V1, initialized with Kinetics-pretrained weights, as a video feature extractor. The SlowFast takes pretrained inflated-ResNet50 [57] as backbone. In RPN+I3D, following [20], we generate region proposals by RPN on key-frame feature and implement action classification and box regression with I3D head. In Faster R-CNN+I3D and SlowFast, we use detection results of a Faster R-CNN detector as ROIs and perform action classification on RoI aligned features. In VTN, we use the same Faster R-CNN detection results as RoIs, but employ the transformer head in [21] for action classification. For ST-GCN, follow the [66], we utilize its official toolbox1 to generate skeleton locations for frames using OpenPose. For Video-Swin, we select the Swin-B2 model pretrained on Kinetics-400 as the classification model. In terms of the TimeSformer, we adopt the standard TimeSformer model3 pretrained on Kinetics-400 as the classification backbone. Results and Analysis The main results are shown in Table 5. The model employing I3D [7] with Faster R-CNN detector performs best on our dataset, outperforming that using I3D for both detection and classification. It’s probably because our dataset contains many crowded scenes, which is chal-
Enhanced by the accurate keypoints location of our proposed pose estimation algorithm, the performance of PoseFlow could be further improved.
6.4 Action recognition
# Baselines
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ecf8/ecf8562d-d65f-4e50-b8b4-79b87e3a6210.png" style="width: 50%;"></div>
<div style="text-align: center;">Table 5: Results of action recognition baselines.</div>
lenging for the detection stream. Therefore, utilizing a high-quality detector could significantly improve the detection performance. VTN [54] is superior on AVA [23] dataset but performs comparatively poor on our dataset. Meanwhile, both the FeatureBank and LSTC can also perform great on AVA by virtue of their feature memory mechanism. However, their performance in HiEve is not satisfying as the AVA dataset. The reason might be that the AVA dataset focuses on human-human and human-object interaction, while our dataset pays more attention to the individual action under complex event conditions. Moreover, the visualization results of first three baselines are shown in Fig. 21, we can observe that it’s difficult for these popular methods to recognize the anomalous actions in our dataset and none of them can tackle the prediction in crowded scenes well. The SlowFast owns the best performance in HiEve excluded the Transformer-based methods. Nevertheless, our proposed simple action recognition baseline still surpasses the vanilla Slowfast with 1.08 wf-mAP and 0.78 f-mAP. The difference in improvement on these two metrics indicates that combining the pose motion pattern can better address the action recognition under crowded scenes. The success of this simple baseline also proves that leveraging the diverse annotation in the HiEve dataset could improve the action recognition task. In terms of the Transformer-based methods (Video-Swin and TimeSformer), they significantly outperform all the above baselines, which is consistent with their great performance on other action detection datasets. Specifically, the performance of the Video-Swin model surpasses our proposed baseline (based on the SlowFast model) and achieves the best results. These findings demonstrate that more powerful long-term spatial-temporal modeling is beneficial for action recognition in our HiEve dataset. As for the skeleton-based method ST-GCN, we can observe that it is not ideal compared to most RGB-based methods. This can be attributed to the difficulty of obtaining accurate pose estimations in our HiEve dataset due to heavy occlusion and complex scenes. In con-
Fig. 21: Visualized results of action recognition baselines and the ground-truth (GT).
trast, commonly-used skeleton-based action recognition datasets (e.g., NTU-RGB+D dataset [48]) feature fixed and simple scenes (indoor settings with only a single person), allowing for relatively accurate pose estimation for subsequent action recognition. Furthermore, these observations also validate the rationality of our proposed method, which leverages ground-truth skeletons as auxiliary information during training to enhance the RGB-based action recognition backbone. This paradigm enables us to utilize pose information for action recognition while simultaneously avoiding inaccurate pose estimation.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/1668/1668282a-b9a2-4f81-9b6c-99fc52291599.png" style="width: 50%;"></div>
<div style="text-align: center;">(e) GT</div>
# 7 More Analysis and ablation study
Upper bound test All the human-centric video understanding tasks are tightly associated with object detection. To study the impact of detection accuracy in the HiEve dataset, we conduct the upper bound test on each task with specific oracle models, where the ground-truth bounding-boxes are directly used during testing, including multi-person tracking, pose estimation, and action recognition. We compared them with the normal setting that we described in section 6 without ground-truth. Table 9 lists the upper bound results for each track. It suggest that the tasks requiring temporal reasoning (Track1&3&4) rely more on the accuracy of the detection. In contrast, the pose estimation track is more dependent on the corresponding algorithm than the detection results. Ability for knowledge transfer HiEve covers large amounts of video frame data with a wide range of humancentric annotations, making it well suitable for model pretraining to inject these models with more comprehensive prior knowledge on downstream tasks. To demonstrate it, we conduct experiments on transfer learning from HiEve to other two related downstream tasks, human pose estimation and multiple object tracking. In detail, we apply HRNet [53] for pose estimation on COCO [33] and MOTDT [8] on MOT20 [12]. For each task, we compare the results with and without pretraining on our HiEve datasets in Table 8. For COCO we report the average AP value, for MOT20 we report the MOTA metric. It can be seen that for both downstream tasks, pretraining on HiEve can help improve the methods obtain better performance. Nevertheless, we can further observe a notable disparity in improvements between the two datasets, with a marginal improvement (0.4 AP) in COCO and a significant (1.2 MOTA) improvement in MOT20. Our HiEve primarily offers prior knowledge for recognition in complex scenes compared to existing datasets. Therefore, the contribution of pretraining on HiEve is related to the complexity of the downstream datasets. Since the COCO dataset predominantly consists of simple and uncrowded scenes, it is reasonable that knowledge transferred from HiEve to COCO yields modest improvements. Conversely, the MOT20 dataset includes more challenging and crowded scenes compared to COCO, so we can see more significant improvement.
In this section, we first conduct experiments to analyze the characteristics of our HiEve dataset. Then, the ablation studies of our proposed algorithm will be presented to evaluate different variants of our proposed algorithm.
7.1 Experimental characteristics
Group & fine-grained action First, to better understand the difficulty of action recognition on the HiEve, we calculate the per-class AP value for each action category. Fig. 22 displays the results obtained by SlowFast [16]. What stands out in this figure is the poor performance of some group behavior recognition, such as ‘gathering’, ‘running-together’, and ‘sitting-talking’. Besides, the performance encounters a marked decline when recognizing fine-grained actions. For example, it’s hard to distinguish the ‘running-alone’ from ‘walkingalone’. Compared to the vanilla SlowFast, our proposed action recognition baseline can effectively improve the accuracy of categories highly related to human skeletons. We also notice that our proposed baseline only gains slight improvement in these group-level and fine-grained categories. These results suggest that introducing pose information does improve action recognition under complex scenes. However, in our future work, specific measures need to be taken to further boost the performance of fine-grained & group action categories in the HiEve dataset. Hard video sequence First, we make a simple subjective analysis of the test video sequence. The CrowdIndex is calculated for each test video sequence to measure the crowding level of frames. The top-3 sequences with the highest CrowdIndex could be naturally regarded as relatively hard examples in the test set. Specifically, they are hm in bus (ID:21), hm in dining room2 (ID:22), and hm in subway station (ID:24). Furthermore, we report the weighted-AP of FT-HRNet[53] on each video sequence, since this metric pays more attention to crowded scenarios. As shown in Fig. 23, consistent with our assumption, the performance shows a sharp degradation in all of these three video sequences. This indicates that the crowded level is a major influence on video understanding tasks in HiEve. Surprisingly, the performance on video sequence hm in stair3 (ID:30) also meets a marked drop whereas its crowded level is relatively low among all sequences. The reason for this is that it was dominated by the overhead view. To sum up, the hard example in our data set are close to the real-world scenes, namely, the severe human occlusion and various video angles.
# 7.2 Ablation study on our proposed baselines
The multi-level feature prediction task enables the video network to learn the pose-specific motion patterns in the training and testing phase. In this section, we aim
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3f45/3f45fcb2-b70d-430f-a5a3-91e84ae9447e.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c76f/c76fd0cd-4110-4a11-a4b7-e1064ccb5eeb.png" style="width: 50%;"></div>
Fig. 22: The performance of SlowFast and ours on each action category in HiEve
Fig. 23: The performance of FT-HRNet on each video sequence in HiEve. Hard video examples (weighted-AP ≤50) are emphasized by red dashed boxes.
Modules
Performance
ADAM
PRM
w-AP@avg
AP@avg
52.78
56.40
✓
53.10
56.87
✓
✓
53.92
57.68
Table 7: Results by different refinement configurations
Table 6: Results of breakdown modules of our algorithm on HiEve dataset. ✓means the module is used
Pretraining ?
Downstream task
HRNet [53] on COCO
MODT [8] on MOT20
NO
74.4
46.4
YES
74.8
47.6
Table 8: Downstream task results with and without HiEve pretraining
Track
Methods
Normal
Oracle
1-human tracking
IOUTracker[5]
MOTA
38.59
97.70
2-pose estimation
DHRN[53]
w-AP@avg
52.78
53.34
3-pose tracking
PoseFlow[63]
MOTA
44.17
73.84
4-action recognition
SlowFast[16]
wf-mAP@avg
12.08
13.21
Table 10: Using features from different levels to predict. ml denotes feature output by stage-l in ResNet-50.
for learning the pose-aware patterns. The performance reaches its peak when we combine all the features from three stages to conduct prediction.
Table 9: The upper bound and normal setting results
to reveal the influence of multi-level feature selection. As shown in Table 10, we test different combinations of features across model stages to predict the pose-aware motion pattern. We can observe that using a single-level video feature is hard to conduct a precise prediction and only lead to a slight improvement. We also notice that the middle-level feature m2 is crucial in multi-level feature joint prediction. The reason may be that the middle-level feature contains both high-level semantic information and low-level texture, which is beneficial
The contributions of different modules in our model are first analyzed via experiments. Table 6 presents the breakdown results of the action-guided domain alignment (ADAM) and pose refinement module (PRM). We can observe that by introducing action category information as a kind of regularization, the performance can achieve a large improvement of 1.24 weighted-AP. Besides, the performance can be further boosted to 54.00 w-AP with the refinement module, which indicates that
Refinement Setting
Performance
SR
CR
w-AP@avg
AP@avg
✓
53.20
56.97
✓
53.65
57.25
✓
✓
53.92
57.68
Combination
wf-mAP@avg
m1
m2
m3
✓
12.36
✓
12.58
✓
12.44
✓
✓
12.79
✓
✓
12.61
✓
✓
12.57
✓
✓
✓
13.16
7.2.2 Study on action-guided pose estimation
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a9a7/a9a72c85-b391-41da-9fa1-38878755c283.png" style="width: 50%;"></div>
Fig. 24: Prediction of keypoints in a test video without (left)/with (right) PRM. Keypoints rectified by PRM are indicated by green arrow.
Table 11: Submissions selected from the offical leaderboard on the HiEve website.
the attention mask generated by the aligned latent feature fosters the pose feature revision and refinement. To further validate the effectiveness of the PRM, we first visualize the pose estimation results without/with PRM module. As presented in Fig. 24, PRM is able to rectify the position of some keypoints or replenish some hard keypoints that are not detected. Moreover, we also apply the SR and CR separately. As shown in Figure Table 7, each refinement plays an important role in the final performance. The application of single SR module gains 1.32 w-AP and 1.29 AP from the vanilla HRNet. With the combination of CR, the refinement module could provide the best performance. The contribution comparison demonstrates that the channel-wise refinement contributes more significantly to pose estimation refinement in crowded scenarios, which may be due to the difficulty of spatial attention modeling for severe occlusion scenes.
could provide a comprehensive evaluation and comparison among these SOTA methods or submissions. And we’ll show some real examples to further validate this. Table 11 presents submissions that selected from our public leaderboard on the HiEve website. As for the tracking task, we can observe that the submission ‘JiaRen.AI ’ have a very close AP with submission ‘Commander’. However, the ‘JiaRen.AI ’ marginally surpasses the ‘Commander’ on the w-MOTA. Our w-MOTA pays more attention to performance on disconnected tracks, which is a common problem in complex real-world scenes. Therefore, our leaderboard could provide a fair rank for these two methods and proves that the ‘JiaRen.AI ’ is a better choice for MOT task in complex scenes. Our proposed metric ‘wf-mAP’, which focuses more on frames with crowded or complex scenes, also contribute to a fair comparasion among action recognition methods. It can be seen from Table 11 that the submission ‘CF’ outperforms the submission ‘8A’ with a significant margin in the traditional frame-mAP metric. However, these two methods have similar performance on our wf-mAP metric. It demonstrates that the performance of ‘CF’ will rapidly drop under crowded scenes, while the ‘8A’ is more stable. Similar issues can be found in Table 11 for pose estimation with our proposed w-mAP metric. The above real example illustrates that our proposed metrics can provide a comprehensive evaluation for algorithm, especially for real-world complex events. Furthermore, apart from our newly-introduced weight metrics, we also maintain the original unweighted metrics in our evaluation besides our newly-introduced weighted metrics. They work together to ensure a comprehensive evaluation in the HiEve dataset.
7.3.1 Will they leak any information about GT?
Note that the detailed weights and parameters for our three weighted metrics are not available to the researcher. All evaluations are conducted on the HiEve online server. The only way researchers can do for improving performance on weighted metrics is by exploring efficient methods or modules to handle complex events (such as crowded scenes, and anomaly action) in our video.
7.3.2 How do they contribute to a comprehensive comparasion?
Our proposed weighted metrics aim to provide a comprehensive evaluation for various algorithms, especially their performance in real-world complex events. In most cases, the rank under these three metrics is consistent with the traditional metrics (as shown in Table 2, Table 3). However, when methods reach high performance with traditional metrics in HiEve, their performances will be too close to provide a fair comparison between them. Under this kind of condition, our proposed metrics
# 8 Conclusion
We present HiEve, a large-scale dataset for humancentric video analysis. The HiEve dataset covers a wide range of crowded scenes and complex events. We report the results of plenty of approaches in our dataset. Extensive experiments show that the HiEve is a challenging
Task
Submission name
Performance
Rank
Tracking
-
w-MOTA
MOTA
-
‘JiaRen.AI’
42.93
47.40
7
‘Commander’
42.47
47.41
8
Action
recognition
-
wf-mAP
f-mAP
-
‘CF’
15.31
20.63
2
‘8A’
15.09
16.25
3
Pose
Estimation
-
w-AP
AP
-
‘Commander’
52.25
55.47
10
‘DeepBlueAI’
52.05
56.33
11
dataset for pose estimation, multi-person tracking, and action recognition. Based on its diverse annotation, we propose two simple baselines, which use cross-annotation information to improve different visual tasks. Experiments on them validate that our HiEve dataset could facilitate multiple visual tasks by diverse annotations.
# 9 Declarations
Compliance with Ethical Standards The authors declare no conflicts of interest. All videos in this paper are either collected where the human participants were informed in advance and their consents for data publication were obtained, or obtained from online repositories where the publishing approvals from the video authors were obtained and the human identity information was guaranteed to be properly hidden or blurred. Data Availability Statement The datasets analyzed during the current study are all available publicly. Please refer to http://humaninevents.org for further details. Intended use of HiEve The authors do not condone with AI systems developed for malicious/unethical surveillance and tracking systems. Any use of the proposed video dataset must adhere to all relevant laws and regulations, including those related to data protection, privacy, and ethical considerations. The proposed video dataset is not to be used for any purpose that violates individual privacy or other legal or ethical standards. The authors are committed to ensuring that the proposed video dataset is used in ways that benefit society and do not cause harm.
# References
