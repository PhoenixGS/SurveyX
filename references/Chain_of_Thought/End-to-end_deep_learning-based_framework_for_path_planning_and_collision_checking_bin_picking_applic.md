# End-to-end deep learning-based framework for path planning and collision checking: bin picking application
Mehran Ghafarian Tamizia,∗, Homayoun Honarib,∗, Aleksey Nozdryn-Plotnickic, Homayoun Najjarana,b
Real-time and efficient path planning is critical for all robotic systems. In particular, it is of greater importance for industrial robots since the overall planning and execution time directly impact the cycle time and automation economics in production lines. While the problem may not be complex in static environments, classical approaches are inefficient in high-dimensional environments in terms of planning time and optimality. Collision checking poses another challenge in obtaining a real-time solution for path planning in complex environments. To address these issues, we propose an endto-end learning-based framework viz., Path Planning and Collision checking Network (PPCNet). The PPCNet generates the path by computing waypoints sequentially using two networks: the first network generates a waypoint, and the second one determines whether the waypoint is on a collision-free segment of the path. The end-to-end training process is based on imitation learning that uses data aggregation from the experience of an expert planner to train the two networks, simultaneously. We utilize two approaches for training a network that efficiently approximates the exact geometrical collision checking function. Finally, the PPCNet is evaluated in two different simulation environments and a practical implementation on a robotic arm for a bin-picking application. Compared to the state-of-the-art path planning methods, our results show significant improvement in performance by greatly reducing the planning time with comparable success rates and path lengths. Keywords: Path planning, Artificial Neural Network, Collision Checking, Bin Picking, Imitation Learning, Data Aggregation
# 1. Introduction
Bin-picking, which includes object detection, motion planning, and grasping, is a crucial part of automation in industry. Bin picking has been one of the trending research topics in recent years because of the challenges in image processing and path planning [1, 2]. Industrial bin-picking consists of a 2D/3D camera, which is used to collect the environmental information (object and obstacle detection), and a conveyor with the bins. The vision system is used to detect object positions and obstacle shapes, while the motion planning module calculates the path to the grasp position based on this information [3]. An efficient and real-time path for the manipulator can greatly improve the efficiency of mass production and assembly lines. In a typical cycle of bin-picking operations, a combination of machine vision, inverse kinematics, and other algorithms are used to find the best grasping configuration for the robot. Once a configuration is found, path planning algorithms generate a path from the robot’s fixed home state to the grasp, and then to a place state. Finally, the robot physically moves to complete the task. The serial and critical-path nature of path planning and movement means that planning time and execution time are essential factors that impact the overall cycle time and automation economics. It is crucial to ensure that robot paths are collision-free, considering both obstacles in the environment and self-collision. Classical planners can be used for bin-picking, but they often fail to take advantage of the repetitive nature of the task. In a static environment, where a robot cell operates for extended periods, path planning is limited to fixed initial configurations and a fixed subspace of possible end-effector poses within a bin. This paper proposes a more efficient path planner in terms of the planning time for bin-picking tasks. We introduce the Path Planning and Collision checking
Network (PPCNet), a deep neural network-based tool specifically designed for repetitive tasks like bin-picking. The PPCNet generates waypoints along the path and estimates collisions faster, making it a reliable solution for autonomous systems such as robotic arms. By using PPCNet, we can generate a safe and efficient path in a relatively short amount of time. Our proposed method outperforms conventional approaches in terms of planning time and stays competitive in path quality and success rate, making it a viable solution for industrial bin-picking applications. The following are the major features of this research:
• Collision detection is a critical and time-intensive aspect of path planning. PPCNet offers a faster solution, reducing planning time significantly through its ability to quickly detect collisions.
• We employ and compare two different methods for training the collision checking network to determine the best and safest approach for collision detection.
The paper is structured as follows: In Section 2, we provide a comprehensive overview of various path planning techniques, discussing their advantages, disadvantages, and limitations for this problem. Section 3 outlines the problem definition. We then introduce our framework, PPCNet, in Section 4, where we also discuss the training process. In Section 5, we evaluate and compare the performance of our approach against four state-of-the-art path planners. Finally, we conclude our study in Section 6.
# 2. Related work
# 2.1. Path planning
Generating collision-free paths for robotic arms is a key area of research in robotics. Path planning methods can broadly be classified into two categories: classical and learning-based [4]. Classical path planning approaches encompass a wide range of techniques, including artificial potential field (APF) [5], bio-inspired heuristic methods [6], and samplingbased path planners [7]. In contrast, learning-based path planners primarily utilize various machine-learning techniques to plan for the robot’s path. These methods have been gaining popularity in recent years due to their ability to handle complex and dynamic environments. Sampling-based methods are widely used in path planning. In this regard, they can be divided into two categories: Single-query algorithms and Multi-query algorithms. While single-query approaches aim to find a path between a single pair of initial and goal configurations, the multi-query ones try to be efficient when there are multiple pairs [8]. The Rapidly-exploring Random Tree (RRT) [9] and Probabilistic Road Map (PRM) [10] are two of the most well-known algorithms for single-query and multi-query approaches, respectively. PRM is a roadmap algorithm which aims to build an exhaustive roadmap of the configuration space to be able to path plan between any two configurations given to it. While PRM has been used for path planning of robotic arms before [11], it has several limitations. Firstly, it generates paths that are far from the optimal solution. Secondly, generating the roadmap is computationally expensive [12], making it inefficient for path planning in relatively high-dimensional environments, even for static tasks such as bin-picking [13]. Furthermore, given an initial and goal configuration pair, RRT grows a tree of waypoints from the initial configuration in order to connect the two configurations. An important characteristic of RRT algorithm is its probabilistically completeness, meaning that it is guaranteed to find a path if there exists one [8]. Due to RRT’s capability in non-convex high dimensional spaces, it has been extensively employed in many studies for path planning of robotic arms [14, 15, 16]. To increase the performance of RRT in different aspects, such as planning time and optimality, there are many variants of this method in the literature. In [17], the Bi-directional RRT (Bi-RRT) is introduced for path planning of the 7-DOF manipulator. Moreover, in [18] this algorithm used for distributed picking application. This method simultaneously builds two RRTs; one of them attempts to find the path from the initial configuration to the final configuration, and the other one attempts to find the path in the opposite direction and tries to connect these two trees in each iteration. Although this method is faster than RRT, they both often produce sub-optimal paths in practice and the quality of the path depends on the density of samples and the shape of the configuration space. RRT* is the optimal version of the RRT and finds the asymptotically optimal solution if one exists [12]. While that is a strong feature of RRT*, it is inefficient in high-dimensional spaces and has a relatively slow convergence rate. As a result, it is not a suitable solution for the real-time path planning of a robotic arm. To overcome the problems mentioned above, biased sampling is one of the promising methods which can improve the performance of sampling-based path planners,
due to the adaptive sampling of the configuration space of the robot. Batch-informed trees (BIT*) [19] and informed RRT* [20] are other significant variants of the RRT* that employ an informed search strategy and batch-processing approach to find the optimal path in a more efficient way by focusing the search on promising areas of the tree. Although this method is able to find the optimal solution, it is not suitable for real-time path planning since it requires a large amount of computational resources, which can lead to delays in the system’s response time. As mentioned above, since collision-checking is computationally expensive and the configuration space gets exponentially bigger with the increase of dimensionality, most of the classical sampling-based methods suffer from high computation and low convergence rates, making them impractical for use in a complex environment with high dimensional configuration space and real-time applications such as bin picking [21]. Ellekilde et al. [13] present a new algorithm for planning an efficient path in a bin-picking scenario based on using lookup tables. Although this method is almost instantaneous and faster than sampling-based planners, it is memory inefficient. To this end, learning-based path planners have emerged to deal with the limitations of classical methods. Learningbased techniques can solve the long run-time problem of sampling-based methods by providing biased sampling nodes, allowing them to run faster and more efficiently [22]. For instance, Qureshi et al. [23] proposed a deep neural networkbased sampler to generate nodes for sampling-based path planners. This work shows a significant improvement in the performance of sampling-based path planners in terms of planning time. Reinforcement learning (RL) is a subcategory of learning-based techniques for manipulator path planning in an unknown environment [24, 25]. Although RL-based approaches illustrate promising performance in path planning research, they need exhaustive interaction with the environment to acquire experience, and this is not applicable in many practical cases. Moreover, imitation learning is an alternative to these kinds of cases. In imitation learning, the training dataset is provided by an expert during the execution of the task. For instance, in [26], a recurrent neural network is trained by using demonstration from the virtual environment to perform a manipulation task. Neural planners are relatively new methods in the path planning context. These planners are based on deep neural networks, and their purpose is to solve the path planning problem as an efficient and fast alternative to sampling-based methods. Bency et al. [27] proposed a recurrent neural network to generate an end-to-end near-optimal path iteratively in a static environment. Moreover, in [28] and [29], motion planning networks (MPnet) is introduced. This method has a strong performance in unseen environments as the path can be learned directly.
# 2.2. Collision Approximation
Collision detection is a vital component of path planning in robotics and can consume a significant amount of computation time. In fact, it has been reported that it can take up to 90% of the total path planning time [7]. There are several methods for performing collision checking in path planning. Analytical methods like GJK (Gilbert-Johnson-Keerthi) algorithm [30] use mathematical equations and geometric models to determine if two objects will collide. They are often fast and efficient but can struggle with complex shapes and interactions. Grid-based Methods like Voxel Grid [31] divide the environment into a grid of cells and check for collisions by looking at the occupied cells. This is often faster than analytical methods but can result in a loss of precision. Another simple method for collision detection is by approximating objects with overlapping spheres [32]. In this approach, the number of spheres used to represent each object is a crucial hyperparameter. Additionally, this method is more instantaneous than other traditional collision detection algorithms. Recently, machine learning techniques have been utilized to overcome the limitations of traditional collision detection methods in robotics. For example, Support Vector Machines (SVM) [33] have been used to calculate precise collision boundaries in configuration space. Gaussian Mixture Models (GMM) were applied in another study [34], resulting in the path being generated five times faster than Bi-directional RRT. K-Nearest Neighbors (KNN) is another machine learning technique used in modeling configuration space for collision detection [35]. Fastron, a machine learning model, was proposed by Das et al. [36] to model configuration space as a proxy collision detector, decomposing it into subspaces and training a collision checkers for each region. The majority of previous studies in this field rely on decomposing the configuration space, which requires significant engineering and simplifications.
# 3. Problem definition
In this section, we will define the path-planning problem for the bin-picking application. Besides, the notation that we use in this paper will be introduced. The first important concept in path planning is the configuration space (Cspace). Cspace includes all positions and configurations of the robot. We can define q as the specific configuration for an n-joints robotic arm:
Where θi is the angular position of the i-th joint of the arm. C free can be defined as the subspace of Cspace in which the robot is collision-free. Let τ be a trajectory and the first and last element of it be qinitial and qtarget, respectively. The path planning problem is to find all of the elements between qinitial and qtarget in a way that all of the segments in the trajectory lie entirely in the C free space. In the context of path planning, a segment between two configurations is the space of the configurations inside the linear combination of the two configurations. In the other words:
� αqi + (1 −α)qi+1 | α ∈[0, 1] � ⊆C free, ∀i ∈{1, ..., N −1}
Where N is the number of waypoints. Bin-picking includes two different path-planning problems. The first one is finding the path between the initial configuration of the robot and the grasp configuration which is located inside the bin (Qpick); the second one is finding the path between the grasp position and the place configuration (Qplace). Figure 1 shows the pick-and-place operation overview. Now the problem is to design a planner that can satisfy the following objectives: • generating a safe and obstacle-free path. • answering the queries in real time. • generating a high-quality path with a high probability of success.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/2710/2710a807-4eb6-4482-ba22-53358f69bd42.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: Path planning for pick and place operation.</div>
# 4. Path planning and Collision checking framework
In this section, we will delve into the details of the proposed framework for path planning in bin-picking tasks. As illustrated in Figure 2, the proposed method is comprised of two consecutive networks, namely the planning network and the collision checking network. The first network is a modified version of MPNet [29]. The original version of the MPNet consists of two networks. The first one is the planning network which generates the waypoints in a sequential manner by imitating the behavior of an expert planner. The second one, the encoder network, encodes the environment to a fixed-length vector in order to give environment information to the planning network for better planning.
(2)

<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5567/55678595-267a-4679-9785-4b2a72b9876b.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: Path Planning and collision checking network (PPCNet).</div>
The primary obstacles of concern in this paper, are the bin, other permanent structures, and the robot itself. These are considered to be fully known and encoded within the planning environment such that unlike the MPNet, our proposed planner does not use a depth map or other sensory readings of the environment to encode information for the planning network. This allows for faster planning times, which is a key focus of our research. In a full bin-picking solution, the process that determines the grasp configuration will select grasps that are at low risk of collision during grasping and robot path. In our work, the planning network takes the current and goal configurations and generates the next joint space configuration to move into. Each configuration is represented by an n-dimensional column vector. Therefore, the path planning network’s input is a 2n-dimensional vector which is obtained by the concatenation of the current and goal configuration. In each iteration of decision-making, after getting the next configuration from the path planning network, the feasibility of the segment is checked using the second network. The second network, namely the collision checking network, gets the next waypoint and estimates how likely it is that segment between the two configurations will be inside C free. By following this approach, the process of decision-making (followed by collision-checking) in the configuration space can be done in a more efficient manner.
# 4.1. Training and dataset generation
The main purpose of the path planning network is to clone the behavior of an ”expert” planner. In cases like this, where it is feasible for an expert to demonstrate the desired behavior, imitation learning is a useful approach. Moreover, while RRT-based path planners are not themselves sequential decision-makers, they can be modeled as:
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/df38/df386644-8fde-4c55-8e41-c080aaac4121.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3: Post processing procedure: a) The generated path by the planner, b) Path after binary state contraction, c) Path after resampling.</div>
The purpose of the neural planner is to learn the function f. Behavior cloning, which focuses on using supervised learning to learn the expert’s behavior, is the most basic type of imitation learning. The process of behavioral cloning is rather straightforward. The expert’s demonstrations are broken down into state-action pairs, which are then used as independent and identically distributed (i.i.d.) training data for supervised learning. An important concern when using supervised learning for behavioral cloning is the risk of encountering out-of-distribution data. During the inference process, the neural planner may make decisions that were not previously observed in the dataset. Consequently, even minor inaccuracies in the replicated behavior can accumulate rapidly, resulting in a substantial failure rate in identifying viable paths. To mitigate this issue, we propose the data aggregation algorithm (DAGGER) in our study [37]. The training process of the neural planner and the collision checker is explained in Algorithm 1. At the beginning of the process, some initial demonstrations from the expert planner is required. To do so, RandomGoalCon f(T) function samples T goal configurations in the collision-free space of the bin. The generated goal configurations are given to the expert planner to generate the paths using GeneratePath function which requires initial and goal configurations. The expert planner outputs two datasets: the neural planner dataset (D) and the collision checker network dataset (C). D only consists of the final path that the expert planner generates; however, C includes every segment which was checked by the expert planner using the geometrical collision checker. In the case of an RRT-based algorithm, the final path will be stored in D and the whole RRT tree and all the instances of its corresponding steer function are stored in C. To train the neural planner, the demonstration dataset generated by the expert planner should be of high quality. To generate a high-quality dataset, post-processing is applied to the paths generated by the expert planner. As shown in Figure 4, the expert planner generates a collision-free path for a random query. Followed by that, the Binary State Contraction (BSC) is utilized to remove redundant and unnecessary waypoints, resulting in a shorter overall path. The sparsity of waypoints is desirable as it simplifies the subsequent processing, such as collision checking, communication with the robot controller, and trajectory planning. While the output of BSC is similar to the lazy state contraction presented in [28], BSC is computationally more efficient and reduces the run time. After BSC, the resampling function is used to ensure the smoothness and efficiency of the planned path. This involves discretizing the waypoints at regular intervals, which are guaranteed to be collision-free as they already exist on the collision-free path. Importantly, these additional waypoints do not increase the overall length of the path but do serve to reduce the mean and variance of the target step magnitudes, ||qt+1 −qt||. While larger steps require fewer neural network forward passes and collision checks during the inference phase, smaller steps offer several advantages. Firstly, they minimize the deviation from the planned path by reducing the magnitude of individual errors ||ˆqt+1−qt+1||. Secondly, they are more likely to be collision-free as they occupy less physical space. Finally, they offer a more normalized training target for the neural network, where the problem is reduced to predicting a direction in the limit. However, BSC is a crucial step before resampling, as it relies on dividing large steps into regular steps, and a dense path of short steps cannot be divided. Figure 3 illustrates this procedure in a 2D environment, highlighting how this method enhances the smoothness and quality of the path. The PostProcess procedure in Algorithm 1 presents the use of BSC and resampling on the generated paths.
# 4.1.1. Path planner training formulation
Once the dataset has been generated, it is time to train the network. The goal of the network is to predict the nex configuration, ˆqt+1, which should be as close as possible to the actual next configuration, qt+1, in the training dataset. The network uses the dataset to learn the underlying patterns and relationships between the input and output data.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c2d0/c2d0f5d7-0c3c-45b9-bdb1-1f7170db1964.png" style="width: 50%;"></div>
<div style="text-align: center;">igure 4: End-end-training process of PPCNet: Top row: The imitation learning and data aggregation processes for training the planner networ Bottom row: Population-based probability estimation and collision checker network training processes.</div>
To achieve the goal of accurate prediction, the network must minimize the difference, or loss, between the predicte and actual configurations. This is done by comparing the predicted output ˆqt+1 to the actual output qt+1 and adjusting th network’s parameters to minimize the difference. This process is known as backpropagation and it is typically done usin optimization algorithms such as stochastic gradient descent.
his equation, Nt is the number of trajectories and Nd is the number of all individual waypoints in the training dataset.
In this equation, Nt is the number of trajectories and Nd is the number of all individual waypoints in the training datas
4.1.2. Collision checker training formulation To ensure the safety of the robot, we experimented with two different approaches for training the collision checker and evaluated their performance. As mentioned earlier, the collision checker network takes the current and next configurations as input and estimates the probability of the segment being in-collision. Therefore, the training set for this network should include segments and corresponding labels indicating whether they are in collision or not. The optimization function utilized to train the collision checker for both of the approaches is called binary cross-entropy loss:
L( ˆp, ptrue) = ˆp log ptrue + (1 −ˆp) log (1 −ptrue)
where ˆp is the probability estimation made by the neural network and ptrue is the true probability. The reason for using binary cross entropy is due to the fact that, in essence, the collision checker is a binary classification network. Therefore, the standard approach in the literature is to use cross entropy due to it giving a better probability distribution over the
where ˆp is the probability estimation made by the neural network and ptrue is the true probability. The reason for using binary cross entropy is due to the fact that, in essence, the collision checker is a binary classification network. Therefore, the standard approach in the literature is to use cross entropy due to it giving a better probability distribution over the training data. Binary labels: The most basic approach to tackle the optimization of the network is to use the true labels given by the analytical collision checker. In that case, ptrue in Eq. 5 becomes either 0 (collision-free segment) or 1 (in-collision segment). In Figure 4 this approach has been named as Lbinary. The binary cross-entropy loss function measures the
Binary labels: The most basic approach to tackle the optimization of the network is to use the true labels given by the analytical collision checker. In that case, ptrue in Eq. 5 becomes either 0 (collision-free segment) or 1 (in-collision segment). In Figure 4 this approach has been named as Lbinary. The binary cross-entropy loss function measures the
(4)
(5)
Algorithm 1 Training process of PPCNet
Require expert planner π∗, number of initial demonstrations T, number of rollouts in each iteration T ′,number of state
samples S , required policy success rate ζ
Initialize Planner and Collision Dataset: D,C ←∅
Generate initial demonstrations:
GoalCon f ←RandomGoalCon f(T)
D,C ←GeneratePath(π∗, HomeCon f,GoalCon f)
Apply post-processing: PostProcess(D)
S uccessRate ←0
for i = 1, ... do
πi ←TrainPolicy(D)
⊲Start of DAGGER
GoalCon f ←RandomGoalCon f(T ′)
Dπi ←GeneratePath(πi, HomeCon f,GoalCon f)
S i ←S ampleS tates(Dπi, S )
Di,Ci ←GeneratePath(π∗, S i,GoalCon f)
Di ←PostProcess(Di)
D ←D ∪Di.
⊲End of DAGGER
C ←C ∪Ci.
ηi ←TrainCollisionChecker(C)
S uccessRate ←TestPolicy(πi, ηi)
if S uccessRate > ζ then
return πi, ηi
dissimilarity between the network’s predicted probability distribution and the true probability distribution in the binary classification problem. Population-based labels: The sparsity of the binary labels hinders the performance of the collision checker since it skews the output of the network towards the more common label. The motivation for this method is to make the binary labels continuous. In addition to the mentioned reasoning, making the labels continuous can help the network be better optimized in corner cases. For example, a collision-free segment inside the bin is mostly surrounded by incollision segments. Hence, the collision-free segment will not be of much help in optimizing the network compared to the massive existence of the in-collision segments. Making the labels continuous will amplify the effect of the collision-free segment. To make the binary labels continuous, we give a regional estimation of the probability of a specific segment being collision-free by retrieving the segments similar to it whose centers lie within the hypersphere (with a specific radius) surrounding the center of the segment. Finally, the population-based probability (denoted as ˆpre f to indicate it being a reference probability) estimates the probability of a segment being collision-free by dividing the number of collision-free segments by the total number of segments:
where 1 is the identity-indication function. The advantage of the proposed approach is that the probability labels generated by this method can provide a well-behaving continuous function where near the obstacles, the probability gets near zero and the further it gets from them, it approaches one. In practice, we utilize KD-Tree method (built with the centers of the segments) to find similar segments. Moreover, the more data this approach gets, the better the estimation will be. Therefore, our proposed algorithm stores every segment that was checked by the analytical collision checker during the path-generation process with the expert planner. In the case of an RRT-based expert planner, this corresponds to storing the whole tree and all of the in-collision segments generated by it. While this approach may have higher computational cost compared to the binary labels approach, our results indicate that it can achieve better prediction performance. For this approach, ˆpre f is used instead of ptrue in Eq. 5 and the corresponding loss function is called Lpopulation.
# 4.1.3. End-to-end training process
In this section, an overview of the end-to-end process of training the integrated model which consists of both the planner network and the collision checker network will be discussed. The training procedure is outlined in Algorithm 4.1. Initially, a set of trajectories are sampled using the expert planner as the initial demonstration, and the planner and collision dataset are filled. In each iteration, the path planner’s policy is trained using the demonstrations from an expert planner. Consequently, DAGGER process (as explained in Section 4.1) is executed which results in the expansion of
(6)
 Path generation process using PPCNet
Require Trained neural planner π and collision checker η, and initial and goal configurations qstart, qtarget
path ←{qstart}
qcurrent ←qstart
for i = 1, ..., smax do
if S teer(qcurrent, qtarget, η) == qtarget then
path ←path ∪qtarget
if IsFeasible(path) then
return path
else
⊲Apply Patching
segments ←InCollision(path)
for (qs, qe) in segments do
pathalt ←π∗(qs, qe)
if Failed(pathalt) then
return failure
path ←Patch(path, pathalt)
return path
qnext ←π(qcurrent, qtarget)
if S teer(qcurrent, qnext, η) then
qnext ←S teer(qcurrent, qnext, η)
path ←path ∪qnext
qcurrent ←qnext
oth of the datasets. Finally, the collision-checking dataset is trained and the model’s performance is evaluated in the estPolicy function. The function uses some random goal configurations and attempts to path plan between them (fol owing the process in Algorithm 2). If the success rate of the model was satisfying, the path planner and the collision hecker network weights are returned.
# 4.1.4. Path planning process
In this section, the end-to-end path planning process of the PPCNet is explained. As illustrated in Figure 2, the two networks, the planning network and collision checking network, do the decision-making sequentially. In the context of the bin-picking application, given the start, pick, and place configurations, the proposed model picks the object from the bin and places it in the place configuration. As outlined in Algorithm 2, in each iteration of the decision-making process the path planning network outputs the next configuration that the robotic arm must transition into. The current configuration and the next configuration are then checked with the collision-checking network. To do so, the S teer function is used. As explained in Algorithm 3, the path between the two waypoints is discretized into equal segments with lengths close to the resolution step size of the expert planner. This is done in order to have the segments be close to the collision checker training dataset’s distribution. Furthermore, in each segment, the collision-free probability of the segment is checked. In order to decide whether a segment is collision-free or not, a safety threshold is needed. If the probability is more the threshold, the algorithm continues to the next segment for collision checking. If the probability is less than the threshold,
Algorithm 3 Steer Function (qcurrent, qnext)
Require collision checking network η, safety threshold ¯η
segments ←Discretize(qcurrent, qnext)
q final ←qcurrent
for (qi, qe) in segments do
if η(qi, qe) > ¯η then
q final ←qe
else
if q final == qcurrent then
return failure
else
return success, q final
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/21f8/21f8df46-363d-407f-bdde-e447f8fccdc5.png" style="width: 50%;"></div>
<div style="text-align: center;">(c)</div>
then the segment is deemed to be in-collision. In that case, if the in-collision segment is the first segment of the path, then the planner has failed to output a good waypoint. Otherwise, if the in-collision segment is not the first segment, the planning network was successful and the tail of the last collision-free segment will be the output of the steer function. Moving on, if the steer function outputs success, then the next waypoint toward the goal configuration is added to the path. In the case that the steer function outputs failure, the planner network attempts to generate a new waypoint. For that purpose, stochasticity is needed inside the network. Therefore, dropout layers are used during inference to have the network be able to recover from failure cases [28]. Furthermore, in each iteration, if it is possible to directly connect the current configuration to the target configuration, then the target configuration is added to the path and it will be the final path generated by the PPCNet. At this point, the feasibility of the path is checked with the geometrical collision checker using IsFeasible function. If the path was collision-free, it will be returned; otherwise, the algorithm will attempt to patch the in-collision segments using the expert planner. To this end, the InCollision function outputs the segments in the path that are in-collision. It is important to note that if some consecutive segments were in-collision, all of the segments will be counted as one bigger in-collision segment and the head and tail of the bigger segment will be considered. This is done in order to have lesser expert planner calls and be more computationally efficient. Furthermore, for each segment, the expert planner attempts to generate an alternative path between the two configurations. If for all of the segments, the patching was done with success, then the proposed path planner would be successful and the path would be returned.
# 5. Experimental results
In this section, we will discuss the performance of our proposed algorithm, which employs PyTorch [38] to implement the planner network and collision checker network. The simulation environment we used is Pybullet planning [39, 40]. To evaluate the effectiveness of our approach compared to the state-of-the-art path planning techniques, we compared it against Bi-RRT [17], informed RRT* [20], BIT* [19], and MPNet [29] on three different environments (Figure 5). The first environment consists of a universal robot UR5 and a bin, the second one involves a UR5 robot and a wall as an obstacle, and the third one features a Kinova Gen3 robotic arm, which we implemented on both simulation and practical settings. The system used for training and testing has a 3.700 GHz AMD Ryzen 9 processor with 96 GB RAM and NVIDIA TITAN RTX GPU. Moreover, Table 1 shows the hyperparameter selection for each of the planners. Table 2 presents a comparative analysis of the performance of PPCNet against other planning methods. Our experiments demonstrate that PPCNet exhibits remarkable performance in terms of planning time in all three environments. Figure 6 provides a visual illustration of the planning time comparison. The hyperparameters for the competing RRTbased algorithms were tuned such that they produce paths of comparable length to PPCNet, while ensuring that they
<div style="text-align: center;">Table 1: Hyperparameters selection for planners.</div>
Methods
Hyperparameters
Values
Max iterations
1000
Bi-RRT
Max planning time
5 s
Resolution
0.1 rad
Number of batches
500
Resolution
0.05
Informed RRT*
γ
500
Goal probability
0.1
Max iterations
1000
Max planning time
10 s
Max iterations
1000
Max planning time
10 s
BIT*
Number of samples
500
η
20
Goal probability
0.1
Max iterations
100
learning rate
0.0001
MPNet
Expert planner
Bi-RRT
Network architecture
Fully connected
6 layers, 300 neurons
per layer
Post-processing
resample step size
0.1745 rad
Max iterations
100
Optimizer
Adam
PPCNet
Similarity distance
0.4 rad
Dagger iterations
30
Expert planner
Bi-RRT
Safety threshold (¯η)
0.8
Network architecture
(planning and , collision
checking network)
Fully connected
6 layers, 300 neurons
per layer
find a path for each planning task in less than 10 seconds and 1000 iterations. However, the results show that BIT* and Informed RRT* methods take longer to plan and have lower success rates than PPCNet. PPCNet has shown its ability to generate paths with comparable length and success rates compared to other algorithms, which is particularly important in evaluating the effectiveness of path-planning methods. Additionally, the post-processing method employed in dataset generation has allowed PPCNet to outperform its expert planner (Bi-RRT) in terms of path length. The BSC function plays a critical role in eliminating redundant and unnecessary waypoints, resulting in a reduction in path length and an improvement in path quality. The combination of the BSC function and resampling has significantly enhanced the quality of paths generated by PPCNet. Furthermore, our findings show that while the addition of the collision checking to the PPCNet has caused significant improvement compared to the MPNet in the expense of the marginal reduction in the PPCNet’s success rate. Finally, the study’s results indicate that the population-based training method for the collision-checking network performed better than the binary classification approach, resulting in a more accurate collision-checking model. This is evident from the decrease in the planning time due to a reduction in the need for patching. Hence, the trade-off between the computational cost of training the population-based collision checker against the accuracy it offers can be observed.
# 6. Conclusions
In this paper, we introduced a novel path-planning framework, PPCNet, and examined its efficacy for bin-picking applications. The robot path planning for the bin-picking application is typically a computationally expensive process
Environments
Methods
Planning time (s)
Path length (rad)
Success rate (%)
Bi-RRT
0.701 ± 0.246
7.516 ± 2.147
100
Informed RRT*
10.186 ± 3.192
5.304 ± 1.983
79.8
BIT*
8.621 ± 3.594
5.278±2.315
77.6
UR5
MPNet
0.384 ± 0.079
7.305 ± 2.298
94.4
PPCNet
Collision: Binary
0.101 ± 0.031
5.887 ± 1.038
90.2
PPCNet
Collision: Population
0.093 ± 0.033
5.679 ± 1.503
93.4
Bi-RRT
0.749 ± 0.182
7.925 ± 3.307
100
Informed RRT*
10.856 ± 3.425
5.813 ± 2.056
73.6
BIT*
9.299 ± 3.919
5.514 ± 2.747
70.4
UR5 with a wall
MPNet
0.392 ± 0.106
7.874 ± 3.011
93
PPCNet
Collision: Binary
0.106 ± 0.027
6.142 ± 1.979
89.8
PPCNet
Collision: Population
0.099 ± 0.034
6.016 ± 1.749
92.2
Bi-RRT
0.645 ± 0.281
6.812 ± 2.749
100
Informed RRT*
9.535 ± 4.122
5.124 ± 2.205
79.2
BIT*
8.496 ± 3.082
5.099 ± 2.464
79.6
Kinova Gen3
MPNet
0.227 ± 0.151
6.714 ± 2.812
90.6
PPCNet
Collision: Binary
0.090 ± 0.029
5.315 ± 1.482
88.2
PPCNet
Collision: Population
0.085 ± 0.032
5.188 ± 1.335
89.8
because the traditional algorithms need to compute the path in its entirety every time the target changes. Further, most existing algorithms lack the ability to learn from past experiences. The proposed PPCNet utilizes deep neural networks to find a real-time solution for path-planning and collision-checking problems. Specifically, we propose an end-to-end training algorithm that leverages the data aggregation algorithm to generate i.i.d. data for supervised learning. Using an expert planner in the training process, data aggregation can assist the agent in encountering unseen states. Another novel component of the proposed PPCNet algorithm includes two variations of the collision-checking network for approximating the exact geometrical collision function. We evaluate our proposed framework, PPCNet, by conducting simulation on two robotic arms namely UR5 and Kinova Gen3 in a bin-picking application. We have also examined the algorithm in a real-world scenario using Kinova Gen3 robot. Our results indicate that our approach significantly reduced planning time while producing paths with path lengths and success rates comparable to those of the state-of-the-art path planning methods.
# Acknowledgment
We would like to acknowledge the financial support of Apera AI and Mathematics of Information Technology an Complex Systems (MITACS) under IT16412 Mitacs Accelerate.
# References
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0a3c/0a3c47be-669f-4ea4-a98a-e130ab602dd6.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 6: Planning time comparison between different methods.</div>
