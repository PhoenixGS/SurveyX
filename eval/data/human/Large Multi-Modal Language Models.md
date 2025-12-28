# Large-scale Multi-Modal Pre-trained Models: A Comprehensive Survey  

Xiao Wang $^{1,2}$ , Guangyao Chen $^{1,3}$ , Guangwu Qian1, Pengcheng Gao $^{1}$ , Xiao-Yong Wei $^{1,4}$ , Yaowei $\mathrm{Wang}^{(\bigotimes)1}$ , Yonghong Tian( $\Join$ )1,3 and Wen Gao $^{1,3}$  

$^{1}$ Pengcheng Laboratory, Shenzhen 518055, China. $^2$ School of Computer Science and Technology, Anhui University, Hefei 230601, China. $^3$ School of Computer Science, Peking University, Beijing 100871, China. $^4$ College of Computer Science, Sichuan University, Chengdu 610065, China.  

# Abstract  

With the urgent demand for generalized deep models, many pre-trained big models are proposed, such as BERT, ViT, GPT, etc. Inspired by the success of these models in single domains (like computer vision and natural language processing), the multi-modal pre-trained big models have also drawn more and more attention in recent years. In this work, we give a comprehensive survey of these models and hope this paper could provide new insights and helps fresh researchers to track the most cuttingedge works. Specifcally, we frstly introduce the background of multi-modal pre-training by reviewing the conventional deep learning, pre-training works in natural language process, computer vision, and speech. Then, we introduce the task defnition, key challenges, and advantages of multi-modal pretraining models (MM-PTMs), and discuss the MM-PTMs with a focus on data, objectives, network architectures, and knowledge enhanced pre-training. After that, we introduce the downstream tasks used for the validation of large-scale MM-PTMs, including generative, classifcation, and regression tasks. We also give visualization and analysis of the model parameters and results on representative downstream tasks. Finally, we point out possible research directions for this topic that may beneft future works. In addition, we maintain a continuously updated paper list for large-scale pre-trained multi-modal big models: https://github.com/wangxiao5791509/MultiModal BigModels Survey.  

Keywords: Multi-modal, Pre-trained Model, Information Fusion, Representation Learning, Deep Learning  

# 1 Introduction  

Along with the breakthroughs of recognition performance of AlexNet [1] on the ImageNet competition [2], the artifcial intelligence have developed greatly. Many representative deep neural networks are proposed, such as VGG [3], ResNet [4], Inception [5], LSTM [6]. The researchers usually collect and annotate some samples for their task, and train their models based on pre-trained backbones on large-scale datasets (such as ImageNet [2] for computer vision, Glove [7] and Skip-thought vectors [8] for natural language processing). Many tasks can be solved well in such an end-to-end manner compared with traditional handcrafted features, such as object detection, segmentation, and recognition. However, the generalization ability of obtained deep model is still limited. Collecting and annotating a larger dataset can address these issues to some extent, but this procedure is expensive and tedious.  

To address this issue, Ashish et al. propose the Transformer network [9] which achieves new SOTA (State-Of-The-Art) performance on machine translation task. After that, the selfsupervised pre-training on large-scale corpus, then, fne-tuning on downstream tasks attracts more and more researchers’ attention. Many pretrained big models are proposed by following such paradigm, such as BERT [10], GPT [11, 12], T5 [13], XLNet [14] which also trigger new research highlights of pre-training in CV community. More and more large-scale NLP and CV models demonstrate the powerful efect by pretrain-and-fnetuning paradigm, including ViT [15] and Swin-Transformer [16].  

Although the progress brings new impetus to the development of artifcial intelligence, however, the issues caused by the defect of single modality are still hard to solve. Researchers attempt to incorporate more modalities to bridge the data gap for deep models. Many multi-modality fusion based tasks are also explored in a traditional deep learning manner, such as RGB, Depth, Natural Language, Point Cloud, Audio, Event stream, etc. Many large-scale pre-trained multi-modal models [17–23] are proposed which set new SOTA on downstream tasks one after another, as shown in Fig. 1. In this paper, we give a comprehensive review of these works which target to help the new researchers who are interested in this area to understand the history and latest developments quickly.  

Organization of our review. In this paper, we frstly review the background of multi-modal pre-training technique in Section 2, from the traditional deep learning paradigm to pre-training in single modality tasks, including natural language processing, computer vision, and automatic speech processing. Then, we focus on MM-PTMs and describe the task defnition, key challenges, and benefts, in Section 3.1 and 3.2. The key components are also reviewed in the following sub-sections, including large-scale data, network architectures, optimization objectives, and knowledge-enhanced pre-training. To validate the efectiveness of pre-trained models, many downstream tasks are used for quantitative assessment. In Section 4, we provide detailed reviews on the task defnition and evaluation metrics of these tasks. In Section 5, we review the model parameters and hardware for training and also report the experimental results of several representative downstream tasks. Finally, in Section 6, we conclude this survey and propose multiple research directions needed to be studied. The architecture of this survey is visualized in Fig. 2.  

Diference from existing reviews. Although there are already two surveys [24, 25] proposed for MM-PTMs, the diference between our survey and existing ones can be summarized as follows:  

• Scope: Existing multi-modal surveys [24, 25] focus on vision-language only, however, the multi-modal information problem is a wider research topic. This paper is more comprehensive than the aforementioned reviews by introducing more modalities, such as audio, video, table, etc.   
• Timeliness: This paper introduces the latest datasets and algorithms (from the year 2019 to June 2022) proposed for multi-modal pretraining which is a long survey, meanwhile, their work belongs to short paper.   
• New insights to MM-PTMs: By classifying and analyzing the existing MM-PTMs from diferent perspectives, this article can help readers master the cutting-edge methods and techniques from both detailed and high-level perspectives. In addition, our proposed research directions on the MM-PTMs are deliberate and will provide new clues for the follow-up research.  

# 2 Background  

# 2.1 Conventional Deep Learning  

With the release of AlexNet [1], a series of deep learning models are proposed in the artifcial intelligence community. These deep models show better capabilities for ftting complex data than conventional machine learning models. From the perspective of its development (LeNet [51] $\rightarrow$ AlexNet [1] $\longrightarrow$ VGG [3] $\longrightarrow$ ResNet [4] → DenseNet [52]), we can fnd that their architectures become deeper and deeper, and the corresponding performance accordingly becomes better. The success of these approaches is supported by large-scale annotated training data, such as the ImageNet [2] for the classifcation task. The scale of used data is much larger than traditional methods, but it’s still limited. The pursuit of robustness and generalization performance of machine learning models has never stopped.  

Table 1 Summary of related single- and multi-modal pre-training surveys. SC and DC denotes Single Column and Double Column. Pub. is short for Publication.   


<html><body><table><tr><td>No.</td><td>Title</td><td>Year</td><td>Pub.</td><td>Topic</td><td>Pages</td></tr><tr><td>01</td><td>A short survey of pre-trained language models for conversational ai-a new age in nlp [26]</td><td>2020</td><td>ACSWM</td><td>NLP</td><td>DC, 4</td></tr><tr><td>02</td><td>A Survey of Controllable Text Generation using Transformer-based Pre-trained Language Models [27]</td><td>2022</td><td>arXiv</td><td>NLP</td><td>SC, 34</td></tr><tr><td>03</td><td>A SurveyofKnowledge Enhanced Pre-trained Models [28</td><td>2021</td><td>arXiv</td><td>KE</td><td>DC, 20</td></tr><tr><td>04</td><td>A Survey of Knowledge-Intensive NLP with Pre-Trained Language Models [29]</td><td>2022</td><td>arXiv</td><td>KE</td><td>DC, 8</td></tr><tr><td>05</td><td>Commonsense Knowledge Reasoning and Generation with Pre-trained</td><td>2022</td><td>arXiv</td><td>KE</td><td>DC, 11</td></tr><tr><td>06</td><td>Language Models: A Survey [30] A survey on contextual embeddings 31 Pre-train, prompt, and predict:</td><td>2020</td><td>arXiv</td><td>NLP</td><td>DC, 13</td></tr><tr><td>07</td><td>A systematic survey of prompting methods in natural language processing [32]</td><td>2021</td><td>arXiv</td><td>NLP</td><td>SC, 46</td></tr><tr><td>08</td><td>Pre-trained Language Models in Biomedical Domain: A Systematic Survey [33]</td><td>2021</td><td>arXiv</td><td>NLP</td><td>SC, 46</td></tr><tr><td>09</td><td>Pre-trained models for natural language processing: A survey [34]</td><td>2020</td><td>SCTS</td><td>NLP</td><td>DC, 26</td></tr><tr><td>10</td><td>Pre-Trained Models: Past, Present and Future [35]</td><td>2021</td><td>AI Open</td><td>NLP, CV, MM</td><td>DC, 45</td></tr><tr><td>11</td><td>Recent Advances in Natural Language Processing via Large Pre-Trained Language Models: A Survey [35]</td><td>2021</td><td>arXiv</td><td>NLP</td><td>DC, 49</td></tr><tr><td>12</td><td>A Survey ofVision-Language Pre-Trained Models [36]</td><td>2022</td><td>arXiv</td><td>MM</td><td>DC, 9</td></tr><tr><td>13</td><td>Survey: Transformer based video-language pre-training [37</td><td>2022</td><td>AI Open</td><td>CV</td><td>DC, 13</td></tr><tr><td>14</td><td>Vision-LanguageIntelligence: Tasks, Representation Learning,</td><td>2022</td><td>arXiv</td><td>MM</td><td>DC, 19</td></tr><tr><td>15</td><td>and Large Models [38] A survey on vision transformer 39</td><td>2022</td><td>TPAMI</td><td>CV</td><td></td></tr><tr><td>16</td><td>Transformers in vision: A survey 40]</td><td>2021</td><td>CSUR</td><td>AO</td><td>DC, 23</td></tr><tr><td>17</td><td>A Survey of Visual Transformers 41</td><td>2021</td><td>arXiv</td><td>CV</td><td>SC,38</td></tr><tr><td>18</td><td>Video Transformers:A Survey 42</td><td>2022</td><td>arXiv</td><td>CV</td><td>DC, 21 DC, 24</td></tr><tr><td></td><td>Threats toPre-trained Language</td><td></td><td></td><td></td><td></td></tr><tr><td>19</td><td>Models: Survey and Taxonomy [43] A survey on bias in deep NLP 44</td><td>2022 2021</td><td>arXiv AS</td><td>NLP</td><td>DC, 8</td></tr><tr><td>20 21</td><td>A Survey of Controllable Text Generation using Transformer-based</td><td>2022</td><td>arXiv</td><td>NLP NLP</td><td>SC, 26 SC, 34</td></tr><tr><td>22</td><td>Pre-trained Language Models [27] AnEmpirical Survey of theEffectiveness of Debiasing Techniques for</td><td>2021</td><td>arXiv</td><td>NLP</td><td>DC, 21</td></tr><tr><td>23</td><td>Pre-Trained Language Models [45] A multi-layer bidirectional transformer encoder for pre-trained word embedding:</td><td>2020</td><td>CCDSE</td><td>NLP</td><td>DC,5</td></tr><tr><td>24</td><td>A survey of BERT [46] SurveyofPre-trainedModels</td><td>2021</td><td>ICEIB</td><td>NLP</td><td>DC, 4</td></tr><tr><td>25</td><td>for Natural Language Processing [47] A Roadmap for Big Model48</td><td>2022</td><td>arXiv</td><td>NLP, CV, MM</td><td>SC,200</td></tr><tr><td>26</td><td>Vision-and-LanguagePretrained Models: A Survey [ [49]</td><td>2022</td><td>IJCAI</td><td>MM</td><td>DC,8</td></tr><tr><td></td><td>Multimodal Learning with</td><td></td><td></td><td></td><td></td></tr><tr><td>27</td><td>Transformers: A Survey [50]</td><td>2022</td><td>arXiv</td><td>MM</td><td>DC, 23</td></tr></table></body></html>  

![](images/c8b8faecf0b7fa3a3927b98bc4ffe55b70c1831cc78e46e96e8cc0cc75d1e23c.jpg)  
Fig. 1 The chronological milestones on multi-modal pre-trained big models from 2019 to the present (June 2022), including multi-modal datasets (as shown by the orange arrow) and representative models (as shown by the blue arrow). The purple font indicates that the dataset contains Chinese text (other datasets contain English text). The models highlighted in wine red are trained on more than two modalities.  

![](images/b8a0174cca0e5bb30327bdf5b79af19ac7e630edab0890f4504795fe69ec2ee2.jpg)  
Fig. 2 The overall framework of this survey.  

Recently, the results of large-scale pre-trained models obtained by pre-training on massive data are constantly refreshing people’s cognition of artifcial intelligence. Compared with previous smallscale deep learning methods, pre-trained big models show obvious advantages in Natural Language Processing (NLP), Computer Vision (CV), and Multi-Modal felds. Such a pre-training scheme take full advantage of the large-scale unlabeled data, therefore, getting rid of expensive annotation costs. Therefore, the study of large-scale pre-trained models is a feasible and necessary way to explore real intelligence.  

# 2.2 Pre-training in Natural Language Processing  

The large-scale pre-trained models [29, 43, 44, 53– 56] frst appeared in the NLP feld. Their success is mainly attributed to self-supervised learning and network structures like Transformer [9]. Specifcally, the advent of Bidirectional Encoder Representations (BERT) [10] based on self-supervised learning has led to revolutionary performance improvements on a wide variety of downstream tasks by fne-tuned on fewer training data [57]. Generative Pre-trained Transformers (GPT) [12, 58, 59] further extends the number of parameters and the training data for better performance. Note that, the GPT-3 [12] has ten times more parameters than TuringNLP [60]. It can not only better fulfll the functions of general NLP tasks, but also has some mathematical calculation ability. The success of the GPT-3 model has made it widely used in various felds, such as search engines, chatbots, music composition, graphics, and coding. XLNet [14] is developed based on a generalized permutation language modeling objective, which achieves unsupervised language representation learning. PanGu- $\alpha$ [61] is a largescale pre-trained Chinese model with 200 billion parameters and implemented based on MindSpore Auto-parallel. NEZHA [62] is another Chinese pretrained big model based on BERT proposed by Wei et al. More large-scale pre-trained models for NLP can be found in surveys [27, 34].  

# 2.3 Pre-training in Computer Vision  

Inspired by the revolutionary advancement of Transformer for NLP tasks, many large-scale Transformer-based vision models are also proposed in recent years. Chen et al. [63] attempt to auto-regressively predict pixels using a sequence Transformer. The model obtained by pre-training on the low-resolution ImageNet dataset demonstrates strong image representations. The ViT (Vision Transformer) model [64] directly adopts the pure Transformer to handle the sequence of image patches for classifcation. Many new SOTA performances are achieved on several downstream CV tasks, including object detection [65], semantic segmentation [66], image processing [67], video understanding [67]. The Swin-Transformer [16] is another milestone for computer vision, as a hierarchical Transformer, it adopts shifted windows for representation learning.  

For the pre-training methods, the Masked Image Modeling (MIM) [63, 64] is proposed to learn rich visual representations via masked parts prediction by conditioning on visible context. MIM provides another direction for the exploration of the visual large-scale pre-training model. He et al. propose the MAE [68] to re-explore pixel regression in MIM and show more comparable performance on multiple image recognition tasks. BEiT [69] greatly improves MIM’s performance via masked visual token prediction, and PeCo [70] fnds injecting perceptual similarity during visual codebook learning benefts MIM pre-trained representation.  

# 2.4 Pre-training in Audio and Speech  

As one of the most popular modalities, the audio and speech based pre-training also draws the researcher’s attention. For example, the wav2vec [71] is the frst work that applies contrastive learning to improve supervised speech recognition by learning the future raw audio based on the past raw audio. The vq-wav2vec [71] uses context prediction tasks from wav2vec to learn the representations of audio segments. DiscreteBERT [72] is BERT-style model by fnetuning the pre-trained BERT models on transcribed speech. HuBERT [73] uses self-supervised speech learning where an ofine clustering step is used to generate discrete labels of masked speech signals. wav2vec 2.0 [74] solves a contrastive task to predict the masked latent representation. w2v-BERT [75] uses contrastive learning and masked speech modeling simultaneously, where a model predicts discretized speech tokens and another model solves a masked prediction task.  

![](images/5f755aeebc044dd544b87278a94e8ff220b646900388abb9ef9c1d4069a8eba4.jpg)  
Fig. 3 The detailed network architecture of Transformer network [9].  

# 3 Multi-Modal Pre-training  

# 3.1 Task Defnition and Key Challenges  

Task Defnition. Usually, the deep neural networks are trained on a large-scale dataset, for example, the widely used residual network [4] are pre-trained using a classifcation task on the ImageNet dataset [2]. In contrast, the multi-modal pre-training big models are usually trained on a massive training dataset. Usually, these data are not annotated with labels due to the scale are too large to annotate. On the other hand, the parameters need to reach a certain scale. As illustrated in Fig. 4, the multi-modal data, big model, and computing power are tightly connected. All in all, with the support of computing power, the multimodal pre-training usually denotes the task that the multi-modality model with huge parameters pre-trained on the massive multi-modal data in an unsupervised way.  

Key Challenges. It is challenging to attain a great multi-modal pre-training big model according to aforementioned process. More in detail, we summarize the following key challenging factors:  

Acquisition and clean of large-scale multi-modal data. The multi-modal data is one of the most important elements in MM-PTMs. The collection of multi-modal data is signifcantly harder than the single one, due to the scarce of multi-modal imaging devices. The frequently used multi-modal cameras are usually covers two modalities only, such as RGB-Depth, RGBThermal, RGB-Radar, RGB-Event cameras, etc. Most of current MM-PTMs are vision-language models, because of the easy access to image and text data from the Internet. But the additional cleaning of these data is also necessary due to the noisy samples.  

• Design of network architectures for large-scale multi-modal pre-training. The network architecture is another key component for multi-modal pre-training. The networks used for feature encoding of multiple input modalities are worthy carefully tailored, as diferent modalities may have their own features and particular networks are needed. For example, the Transformer or CNN are suggested for image and text modality, the spiking networks can be used for event streams. Another problem is the design of multimodal fusion or cross-modality matching modules. Whether similar modules designed for small-scale multi-modal tasks work for large-scale pre-trained models or not are still remain to be verifed.  

Design of pre-training objectives. Due to the massive unlabelled multi-modal data, the pre-training tasks usually need to be done in an unsupervised learning manner. Many current works adopt the masked region prediction for each modality as their learning objective. Obviously, the objectives for multi-modal tasks can be directly borrowed from single-modality pre-training, however, the pre-training objectives designed for the multi-modal tasks are also necessary, intuitive and efective. The widely used contrastive learning, modality based matching, and modality translation are all valid and meaningful attempts. How to design new multi-modal pretraining objectives is one of the most challenging tasks for MM-PTMs.  

• Support of large-scale computing power. The training for traditional deep neural networks can be executed on a server with limited number of GPUs. In contrast, the MM-PTMs needs more computing power due to the largescale multi-modal data and the super large-scale model parameters. Therefore, the frst thing is to prepare a supercomputing device and the subsequent model training also requires a lot of power to support.  

• Skills on parameter tuning. It is never a simple task to train an efective large model considering aforementioned challenging factors. The tricks used for training the neural networks are also very important. As the research and techniques for the small scale pre-training are relatively more mature, however, there is less accumulation of experience on large-scale pre-training techniques.  

# 3.2 Advantages of MM-PTMs  

Compared with single modality pre-trained big models, the MM-PTMs are more suitable for practical application scenarios. Specifcally, the problems like multi-modal collaborative generation, modal completion, cross-domain retrieval, etc, can be addressed well using MM-PTMs. Also, the multi-modal data contains more information which can make up for the defects of a single modality. Therefore, the MM-PTMs can help extracting the common features of multimodalities. Many recent works demonstrate that the utilization of MM-PTMs indeed brings in the additional prior knowledge [76–78].  

![](images/69a89a67f88f05cab987ad827b9d4d4fa1aa74edbaf1c6f9b9db5ee8c6106c08.jpg)  
Fig. 4 The relations between multi-modal data, model, and computing power.  

Compared with small-scale multi-modal models, the generalizability of MM-PTMs which are obtained by self-supervised/unsupervised learning can be improved signifcantly. As some prior knowledge is only contained in massive big data, and a small amount of artifcially selected annotated data is biased, therefore, it is hard for the small-scale models to master such knowledge.  

# 3.3 Pre-training Data  

As shown in Table 2, many large-scale multimodal datasets are proposed for the pre-training task. In this subsection, we will briefy introduce these datasets to help readers quickly master the data information for pre-training.  

• SBU Captions [79] is originally collected by querying Flickr 1 using plentiful query terms. Then, they flter the obtained large-scale but noisy samples to get the dataset, which contains more than 1M images with high-quality captions.  

• Flickr30k [80] is obtained by extending Hodosh et al. [110] ’s corpus with 31,783 photographs collected from Flickr. These images cover everyday activities, events, and scenes. Five sentences are annotated for each collected image via crowdsourcing, therefore, Flickr30k contains 158,915 captions.  

• COCO [111] is developed based on MSCOCO dataset [111] which contains 123,000 images. The authors recruit the Amazon Mechanical Turk 2 to annotate each image with fve sentences.  

• Visual Genome [82] is proposed to help develop machine learning models that can understand the image by mining the interactions and relationships between objects. Therefore, they perform well on the cognitive tasks, such as the image description and visual question answering, etc. Statistically, the Visual Genome dataset contains more than 108K images and each image has about 35 objects, 26 attributes, 21 pairwise relationships.  

• VQA v2.0 [83] is proposed to reduce the language biases that existed in previous VQA datasets which contain about 1.1M imagequestion samples and 13M associated answers on 200K visual images from the COCO dataset.  

• FashionGen [84] contains 325,536 highresolution images ( $1360\,\times\,1360$ ), each image has a paragraph-length descriptive captions sourced from experts. Six diferent angles are photographed for all fashion items.  

• CC3M [85] is a dataset annotated with conceptual captions proposed in 2018. The image-text samples are mainly collected from the web, then, about 3.3M image-description pairs remained after some necessary operations, such as extract, flter, and transform.  

CC12M [88] is the outcome of urgent need of MM-PTMs for large-scale data. The released CC3M dataset is far failed to meet the demand, therefore, the authors further relax the flters used in CC3M for the image and text cleaning. Correspondingly, a four times larger dataset CC12M can be obtained with a slight loss of accuracy.  

• GQA [86] is mainly proposed for visual reasoning and compositional question answering. A robust question engine is carefully refned by considering content and structure information. Then, the associated semantic representations are adopted to greatly reduce biases within the dataset and control for its question type composition. Finally, a balanced dataset with 1.7M samples is obtained.  

Table 2 An overview of multi-modal datasets proposed for large-scale pre-training. Lang. and Ava. is short for Language and Available, respectively.   


<html><body><table><tr><td>No.</td><td colspan="2">Datasets</td><td>Year</td><td>Scale</td><td>Modal</td><td>Lang.</td><td>Ava.</td><td>URL</td></tr><tr><td>01</td><td colspan="2">SBU Captions 79]</td><td>2011</td><td>1M</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>02</td><td colspan="2">Flickr30k [80]</td><td>2014</td><td>145K</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>03</td><td colspan="2">COCO [81]</td><td>2014</td><td>567K</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>04</td><td colspan="2">Visual Genome 82</td><td>2017</td><td>5.4M</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>05</td><td colspan="2">VQA v2.0 [83]</td><td>2017</td><td>1.1M</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>06</td><td colspan="2">FashionGen 84</td><td>2018</td><td>300k</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>07</td><td colspan="2">CC3M [85]</td><td>2018</td><td>3M</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>08</td><td colspan="2">GQA [86]</td><td>2019</td><td>1M</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>60</td><td colspan="2">LAIT [87]</td><td>2020</td><td>10M</td><td>image-text</td><td>English</td><td>×</td><td></td></tr><tr><td>10</td><td colspan="2">CC12M [88]</td><td>2021</td><td>12M</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>11</td><td colspan="2">AltText 89]</td><td>2021</td><td>1.8B</td><td>image-text</td><td>English</td><td>×</td><td></td></tr><tr><td>12</td><td colspan="2">TVQA [ 06</td><td>2018</td><td>21,793</td><td>video-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>13</td><td colspan="2">HT100M 91</td><td>2019</td><td>136M</td><td>video-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>14</td><td colspan="2">WebVid2M [92]</td><td>2021</td><td>2.5M</td><td>video-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>15</td><td colspan="2">YFCC-100M 93</td><td>2015</td><td>100M</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>16</td><td colspan="2">LAION-400M 94]</td><td>2021</td><td>400M</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>17</td><td colspan="2">RedCaps [95]</td><td>2021</td><td>12M</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>18</td><td colspan="2">Wukong [96]</td><td>2022</td><td>100M</td><td>image-text</td><td>Chinese</td><td></td><td>Link</td></tr><tr><td>19</td><td colspan="2">CxC [97</td><td>2021</td><td>24K</td><td>image-text</td><td>English</td><td></td><td>Link</td></tr><tr><td>20</td><td colspan="2">Product1M [98]</td><td>2021</td><td>1M</td><td>image-text</td><td>Chinese</td><td></td><td>Link</td></tr><tr><td>21</td><td colspan="2">WIT [99]</td><td>2021</td><td>37.5M</td><td>image-text</td><td>Multi-lingual</td><td></td><td>Link</td></tr><tr><td>22</td><td colspan="2">JFT-300M 100</td><td>2017</td><td>30M</td><td>image-text</td><td>English</td><td></td><td></td></tr><tr><td>23</td><td colspan="2">JFT-3B [101</td><td>2021</td><td>3000M</td><td>image-text</td><td></td><td></td><td>二</td></tr><tr><td>24</td><td>IG-3.5B-17k</td><td>[102]</td><td>2018</td><td>350M</td><td>image-text</td><td>English</td><td>×</td><td>二</td></tr><tr><td>25</td><td>M6-Corpus 103</td><td></td><td>2021</td><td>60M</td><td>image, image-text</td><td>English Chinese</td><td>× ×</td><td></td></tr><tr><td>26</td><td colspan="2">M5Product [104]</td><td>2021</td><td>6M</td><td>image, text, table</td><td>English</td><td></td><td>Link</td></tr><tr><td>27</td><td colspan="2">Localized</td><td>2020</td><td>849k</td><td>video, audio image, audio, text,</td><td>English</td><td></td><td>Link</td></tr><tr><td>28</td><td colspan="2">Narratives [105] RUC-CAS-WenLan 106]</td><td></td><td></td><td>mouse trace</td><td></td><td>√</td><td></td></tr><tr><td>29</td><td colspan="2">WuDaoMM [107]</td><td>2021 2022</td><td>30M 600M</td><td>image-text</td><td>Chinese Chinese</td><td>×</td><td>Link</td></tr><tr><td>30</td><td colspan="2">MEP-3M [108]</td><td>2021</td><td>3M</td><td>image-text</td><td>Chinese</td><td></td><td>Link</td></tr><tr><td>31</td><td colspan="2">WSCD [109]</td><td>2021</td><td>650M</td><td>image-text image-text</td><td>Chinese</td><td>×</td><td>二</td></tr></table></body></html>  

• LAIT [87] (Large-scale weAk-supervised Image-Text) is a large-scale image-text dataset collected from the Internet in a weak-supervised manner. It contains about 10M visual images, and each image has a corresponding natural language description which contains about 13 words.  

• AltText [89] is collected by following the rules for constructing Conceptual Captions dataset [85]. To get a large-scale dataset (1.8B image-text pairs), the authors only apply minimal frequency-based fltering for data cleaning. Although the obtained resulting dataset is noisy, the big models obtained by pre-training on this dataset still beats many SOTA works on many downstream tasks.  

• TVQA [90] is build based on six longrunning TV shows from 3 genres, including sitcoms, medical dramas, and crime drama. Then, the Amazon Mechanical Turk is used for VQA collection of video clips. Finally, this dataset contains about 152, 545 question-answer pairs from 21,793 video clips.  

• HT100M [91] contains about 136 million video clips, which are collected from 1.22 million narrated instructional videos. The content of these videos are mainly focus on humans with a total of 23,000 various tasks. The language description for each clip is an automatically transcribed narration. Therefore, the video and text are weakly-paired, compared with other captioning datasets.  

• WebVid2M [92] is a video-text captioning dataset which contains over two million video alt-text pairs. These data are collected from the Internet following a similar procedure to CC3M dataset. The authors fnd that more than $10\%$ of CC3M images are thumbnails from videos, therefore, they scrape these video sources (a total of 2.5M text-video pairs) and create the WebVid2M dataset.  

YFCC-100M [93] totally contains 100 million media objects (99.2 million photos, 0.8 million videos) collected from Flickr, the time span of these videos from 2004 and 2014. Note that the YFCC100M dataset is constantly evolving, various expansion packs are unscheduled released.  

LAION-400M [94] contains 400 million image-text pairs which is released for visionlanguage related pre-training. It is worthy to note that this dataset is fltered using CLIP [77] which is a very popular pre-trained vision-language model.  

• RedCaps [95] is a large-scale dataset with 12M image-text samples collected from 350 subreddits. The authors frstly defne the range of subreddit, then, flter the image post and clean the captions. The ethical issue is also considered when building the dataset, and the problematic images are fltered according to privacy, harmful stereotypes, etc.  

• Wukong [96] is the currently largest dataset collected from the Internet which contains 100 million image-text pairs. A list of 200K queries is maintained to ensure the collected samples cover diverse visual concepts. These queries are fed into the Baidu Image Search Engine, then, the image and its corresponding captions can be obtained. Note that each query can get at most 1000 samples to keep a balance between diferent queries and a series of fltering strategies are adopted for the fnal Wukong dataset.  

• CxC [97] is extended based on MS-COCO dataset by rating existing and new pairs with continuous (0-5) semantic similarity. In general, the CxC contains human ratings for 267,095 pairs which is a signifcant extension in scale and detail. It can be used for a variety of tasks, such as the image-text, text-text, and image-image retrieval, etc.  

• Product1M [98] contains 1,182,083 imagecaption pairs, 458 categories, 92,200 instance. Each image contains about 2.83 objects. Diferent from regular object detection benchmark datasets, this dataset obtains the instance locations in a paste manner. They frst segment the target object, then, paste them into other images based on a given bounding box. It can be used for multiple tasks, including weak-supervised, multi-modal, and instance-level retrieval.  

• WIT [99] is constructed by crawling on Wikipedia 3. Then, a set of rigorous fltering operations are executed on these data which fnally resulting the dataset containing over 37.5 million image-text sets. Note that, the WIT dataset contains multi-lingual, in contrast, other image-text datasets only contain single lingual (for example, English or Chinese).  

JFT-300M [100] contains about 300M images and 375M labels, and each image has about 1.26 labels. Note that, 18291 categories are annotated in this dataset, including 1165 animals and 5720 vehicles, etc. A rich hierarchy is formed according to these categories. It is worthy to note that this dataset is not available online.  

• JFT-3B [101] is also an internal Google dataset, which contains about 3 billion images. These samples are annotated in a semi-automatic way with a class hierarchy of 30,000 labels. In other words, this dataset contains large amount of noisy samples. Note that, this dataset is also not available online.  

• IG-3.5B-17k [102] is constructed for weakly supervised pre-training by collecting images from Instagram 4. Similar with JFT-300M [100] and JFT-3B [101], the dataset is also inaccessible and can only be used within the Facebook.  

• M6-Corpus [103] is specifcally constructed for the pre-training of vision-Chinese big model M6 [103]. The samples are collected from various sources, such as the product description, community question answering, forum, etc. It contains 60.5M images and 111.8B tokens.  

• M5Product [104] is a benchmark dataset specifcally proposed for E-commerce. It contains 6 million multi-modal samples which cover 6,000 categories, 5,000 attributes, and fve modalities, including the visual image, table, video, language description, and audio. It is worthy to note that the M5Product dataset is diferent from standard multimodal datasets which have completely paired samples, that is to say, each sample may only contain only a subset of modalities. It also has a challenging long-tailed distribution issue.  

• Localized Narratives [105] is proposed by Jordi et al. in 2020, which provides a new form of multi-modal image annotations for the connection of vision and language. The image and corresponding spoken description, textual description, and mouse trace are all embodied in this dataset which provides dense grounding between language and vision. It contains 849k images and covers the whole COCO, Flickr30k, and ADE20K [112] datasets and 671k images of Open Images.  

• RUC-CAS-WenLan [106] is obtained by crawling multi-source image-text data and totally contains about 30M image-text pairs. These samples covers a wide range of topics and categories, such as the sports, entertainment, news, art, and culture, etc. It plays a fundamental role in the WenLan project and supports the training of the BriVL model [106].  

• WSCD [109] (Weak Semantic Correlation Dataset) is a multi-source dataset, which contains large-scale image-text data samples (650 million). The English texts are all translated into Chinese to support the pre-training of BriVL.  

• MEP-3M [108] is a large-scale image-text dataset collected from several Chinese large Ecommerce platforms which contains 3 million image-text pairs of products and 599 classes. Another key feature of this dataset is the hierarchical category classifcation, in detail, it covers 14 classes, 599 sub-classes, and 13 sub-classes have further sub-subclasses.  

# 3.4 Pre-training Objectives  

How to design the learning objectives is a very important step for multi-modal pre-training. Currently, the following learning objectives are proposed, including contrastive loss, generative loss, etc.  

• Contrastive loss (CS) function usually constructs positive and negative training samples which is widely used in dual-modality. For example, CLIP [77], ALIGN [21] are all trained using contrastive learning loss. The authors of VinVL [113] adopt the 3-way contrastive loss for the pre-training to replace the binary contrastive loss function utilized in the Oscar model [17].  

The contrastive losses in ALIGN are defned as follows:  

$$
\begin{array}{l}{\displaystyle\mathcal{L}_{i2t}=-\frac{1}{N}\sum_{i}^{N}l o g\frac{e x p(x_{i}^{T}y_{i}/\sigma)}{\sum_{j=1}^{N}e x p(x_{i}^{T}y_{j}/\sigma)}}\\ {\displaystyle\mathcal{L}_{t2i}=-\frac{1}{N}\sum_{i}^{N}l o g\frac{e x p(y_{i}^{T}x_{i}/\sigma)}{\sum_{j=1}^{N}e x p(y_{i}^{T}x_{j}/\sigma)}}\\ {\displaystyle\mathcal{L}_{C L}=\mathcal{L}_{i2t}+\mathcal{L}_{t2i}}\end{array}
$$  

where $\mathcal{L}_{i2t},\mathcal{L}_{t2i},\mathcal{L}_{C L}$ are an image-to-text classifcation loss function, a text-to-image classifcation loss function and the total contrastive loss respectively. The $x_{i}$ is used to denote the normalized image embedding in the $i$ -th pair, while the $y_{j}$ denote the normalized embedding of text in the $j$ -th pair. The $N$ and $\sigma$ are batch size and temperature parameter.  

• Modality Matching loss (MML) is widely used in multi-modal pre-training big models due to the explicit or implicit alignment relationships between various modalities. For instance, Unicoder-VL [114] utilizes the Visuallinguistic Matching (VLM) for vision-language pre-training. They extract the positive and negative image-sentence pairs and train their model to predict whether the given sample pairs are aligned or not (in other words, to predict the matching scores). Diferent from regular negative image-text samples, the authors of InterBERT [115] design the image-text matching with hard negatives (i.e., ITM-hn) by selecting the highest TF-IDF similarities.  

• Masked Language Modeling (MLM) is another widely pre-training objective, usually, the researchers usually mask and fll the input words randomly using special tokens. The surrounding words and corresponding image regions can be used as a reference for the masked word prediction. Wang et al. train SIMVLM [116] using the Prefx Language Modeling (PrefxLM), which executes the bi-directional attention on the prefx sequence and auto-regressive factorization on the rest tokens, respectively. The words are denoted as $w\,=\,\{x_{1},\cdot\cdot\cdot\,,x_{K}\}$ , and the image regions as $v\,=\,\{v_{1},\cdot\cdot\cdot\,,v_{T}\}$ . For MLM, the input words is masked as $x_{m}$ by the mask indices $m$ by generated randomly with a probability of $p\%$ The optimizing goal is to predict the masked words based on all image regions $v$ and remaining words $x_{-m}$ , by minimizing the negative log-likelihood:  

![](images/3e78c30214b6de490106a225637a2709d1734bbff7fdb1ead0cd43f32addfcdb.jpg)  
Fig. 5 Representative pre-training objectives used in MM-PTMs.  

$$
\mathcal{L}_{M L M}(\theta)=-\mathbb{E}_{(x,v)}\log P_{\theta}(x_{m}|x_{-m},v),
$$  

where $\theta$ is the trainable parameters. Beside MLM, PrefxLM in SIMVLM can also be adopted to pretrain vision-language representation:  

$$
\mathcal{L}_{P r e f i x L M}(\theta)=-\mathbb{E}_{\mathbf{x}\sim D}\log P_{\theta}(\mathbf{x}_{\ge T_{p}}|\mathbf{x}_{<T_{p}}),
$$  

where $\textbf{\em x}$ is the given text sequence, $D$ is the pretraining data and $T_{p}$ is the length of a prefx sequence of tokens.  

• Masked Segment Modeling (MSM) masks a continuous segment of given text using the special token, meanwhile, the MLM masks random words.  

• Image Question Answering (QA) is used in LXMERT [117] to further expand the pretraining data, as many image-sentence pairs are image and question. The authors train their model to predict the answers as one of their pre-training objectives.  

• Masked Object Classifcation (MOC) mainly focuses on masking the visual images using zero values. Then, people often take the predicted labels by object detector as the ground truth labels. This pre-training objective is widely used, such as Unicoder-VL [114]. Similar to MLM, the image regions can be masked by masking their viusal feature with a prabability of $p\%$ . The goal is predict the object category of the masked image regions $v_{m}^{i}$ . The encoder output of the masked image regions $v_{m}^{i}$ is feed into an FC layer to predict the scores of $T$ object classes, which further goes through a softmax function to be be transformed into a normalized distribution $g_{\theta}(v_{m}^{i})$ . The fnal objective is:  

$$
\mathcal{L}_{M O C}(\theta)=-\mathbb{E}_{(w,v)}\sum_{i=1}^{M}C E(c(v_{m}^{i}),g_{\theta}(v_{m}^{i})),
$$  

where $c(v_{m}^{i})$ is the ground-truth label.  

• Masked Object Regression (MOR) is implemented to regress the masked feature or image regions. For example, the LXMERT [117] considers both MOC and MOR for their pretraining.  

• Image-Text Matching (ITM) aims to align the image-text data. Negative training data is generated by randomly sampling, including negative sentences for each image, and negative images for each sentence. $_y$ is denoted by the gourd truth label for each image-text pair $(v,t)$ . A binary classifcation loss function is used for optimization:  

$$
\begin{array}{r l}&{\mathcal{L}_{I T M}(\theta)=-\mathbb{E}_{(v,t)}[y\log s_{\theta}(v,t)}\\ &{\qquad\qquad\qquad+\mathrm{\Theta}(1-y)\log(1-s_{\theta}(v,t))],}\end{array}
$$  

where $s_{\theta}$ is the image-text similarity score.  

• Unidirectional LM (UiDT) Single direction history information is used for masked token prediction only, such as left-to-right and right-toleft language model objectives. Successful stories includes the ELMo [118], UNILM [119].  

Bidirectional LM (BiDT) Diferent from Unidirectional LM which predicts the masked token from a single direction only, the Bidirectional LM considers contextual information from both directions. Therefore, the contextual representations of text can be encoded more accurately. BERT [10], UNIML [119] and VLP [24] all adopt BiDT as one of their pre-training objective.  

• Sequence-to-Sequence LM (Seq2seq) is a pre-training objective used in VLP [24], etc. It treats the inputs as diferent parts, each part can attend to diferent contexts.  

• Word-Region Alignment (WRA) is used in UNITER [18] which target at explicitly achieves the fne-grained alignment between the multimodal inputs via Optimal Transport (OT) [120]. Specifcally, the authors learn a transport plan which is a 2D matrix to optimize the alignment and resort to the IPOT algorithm [121] for approximate OT distance estimation. Then, the authors take this distance as the WRA loss to optimize their networks.  

Action Prediction (AP) target at evaluating whether the agent developed for visionlanguage navigation (VLN) can select the right actions based on the current image and instruction [122].  

• Image-conditioned Denoising Autoencoding (IDA) is adopted in XGPT [11] to align the underlying image-text using an attention matrix. Even without the prior length of the masked fragment, the IDA could still reconstruct the whole sentence successfully.  

• Attribute Prediction (AttP) is used to recover the masked tokens of attribute pairs, as indicated in ERNIE-ViL [123].  

• Relation Prediction (RelP) is used in ERNIE-ViL [123] to predict the probability for each masked relation tokens to recover the masked relationship tokens.  

Aligned Kaleido Patch Modeling (AKPM) is proposed for the pre-training of Kaleido-BERT [124], which contains fve kaleido sub-tasks, i.e., Rotation Recognition (RR), Jigsaw Puzzle Solving (JPS), Camoufage Prediction (CP), Grey-to-Color Modeling (G2CM), and  

Blank-to-Color Modeling (B2CM):  

$$
\begin{array}{r l}&{\quad\mathcal{L}_{R R}=C E(y_{r},\mathcal{F}(T,K,\theta)_{K1.h i d d e n})}\\ &{\quad\mathcal{L}_{J P S}=C E(y_{j},\mathcal{F}(T,K,\theta)_{K2.h i d d e n})}\\ &{\quad\mathcal{L}_{C P}=C E(y_{c},\mathcal{F}(T,K,\theta)_{K3.h i d d e n})}\\ &{\mathcal{L}_{G2C M}=\displaystyle\sum K L D(k_{4i},\mathcal{F}(T,K,\theta)_{K4.h i d d e n})}\\ &{\mathcal{L}_{B2C M}=\displaystyle\sum K L D(k_{5i},\mathcal{F}(T,K,\theta)_{K5.h i d d e n})}\end{array}
$$  

where $C E$ represents the cross-entropy loss function, $y_{r}$ denotes the rotation angle, $K_{p}$ is the hidden output patch of size $p\times p$ , $K L D$ denotes the KL-divergence, and $K_{p}$ are kaleido patches, among which $k_{p i}$ is the masked out ones.  

• OBject Detection (OBD) is introduced in the [125] as a direct set prediction to enhance the pre-training. Also, the authors consider object attribute prediction to learn the fne-grained semantic information. A negative log-likelihood loss is defned for OBD as follows:  

$$
\begin{array}{c}{{\hat{\sigma}=\displaystyle\arg\operatorname*{min}_{\sigma\in\phi_{N}}\sum_{i}^{N}\mathcal{L}_{m a t c h}(y_{i},\hat{y}_{\sigma(i)})}}\\ {{\mathcal{L}_{O B D}(y,\hat{y})=\displaystyle\sum_{i=1}^{N}[-l o g\hat{p}_{\hat{\sigma}(i)}(a_{i})-l o g\hat{p}_{\hat{\sigma}(i)}(c_{i})}}\\ {{+\mathcal{L}_{b o x}(b_{i},\hat{b}_{\hat{\sigma}(i)}(i))]}}\end{array}
$$  

where $_y$ denotes the ground truth set of objects and $\hat{y}~=~\{\hat{y}_{i}\}_{i=1}^{N}$ , the number of elements is $N$ $\sigma$ is the cost of a permutation of $N$ elements, $\mathcal{L}_{m a t c h}\big(y_{i},\hat{y}_{\sigma(i)}\big)$ denotes the pair-wise matching loss between a prediction with index $\sigma(i)$ and ground truth $y_{i}$ , $\hat{p}_{\hat{\sigma}(i)}(a_{i}),\hat{p}_{\hat{\sigma}(i)}(c_{i})$ denotes the attribute and class probability, $\mathcal{L}_{b o x}(b_{i},\hat{b}_{\hat{\sigma}(i)}(i))$ is a normalized loss of bounding box regression.  

• Image-Text Generation (ITG) also plays an important role in the vision-language related pre-training tasks. The aligned image and text are capable of training a model for text generation based on a given image, for example, Xu et al. train the E2E-VLP [125] with ITG objective:  

$$
\mathcal{L}_{I T G}=-\sum_{(x,y)\in(\mathcal{X},\mathcal{y})}l o g\prod_{t=1}^{n}P\big(y_{t}|y_{<t},x\big)
$$  

where $\mathcal{X}$ represents the visual sequence with context, $\boldsymbol{y}$ denotes the generated set of text, and the length of tokens in text $_y$ is $n$ .  

• Video-Subtitle Matching (VSM) considers two targets for the video-text pre-training task, i.e., (i) local alignment, (ii) global alignment, as used in HERO [126]. The score functions and the corresponding loss functions are defned as follows:  

$$
\begin{array}{r l}&{S_{l o c a l}(s_{q},v)=V^{t e m p}q\in\mathbb{R}^{N_{v}}}\\ &{S_{g l o b a l}(s_{q},v)=m a x(\frac{V^{t e m p}}{\|V^{t e m p}\|}\frac{q}{\|q\|})}\\ &{\mathcal{L}_{h}(S_{p o s},S_{n e g})=m a x(0,\delta+S_{p o s}-S_{n e g})}\\ &{\mathcal{L}_{l o c a l}=-\mathbb{E}_{D}l o g(p_{s t}[y_{s t}]+l o g(p_{e d}[y_{e d}])}\\ &{\mathcal{L}_{g l o b a l}=-\mathbb{E}_{D}[\mathcal{L}_{h}(S_{g l o b a l}(s_{q},v),S_{g l o b a l}(s_{q},v))}\\ &{\quad\quad\quad\quad+\mathcal{L}_{h}(S_{g l o b a l}(s_{q},v),S_{g l o b a l}(s_{q},\hat{v}))]}\\ &{\mathcal{L}_{V S M}=\lambda_{1}\mathcal{L}_{l o c a l}+\lambda_{2}\mathcal{L}_{g l o b a l}}\end{array}
$$  

where $s_{q}$ denotes the sampled query from all subtitle sentences, $\pmb{v}$ is the whole video clip, $V^{t e m p}\in$ $\mathbb{R}^{N_{v}\times d}$ is the fnal visual frame representation generated by temporal transformer, $\textbf{\textit{q}}\in\~\mathbb{R}^{d}$ is the fnal query vector, $y_{s t},y_{e d}\;\in\;\{1,...,N_{v}\}$ are the start and end index respectively, $\pmb{p}_{s t},\pmb{p}_{e d}\,\in\,\mathbb{R}^{N_{v}}$ represent probability vectors generated from the scores, $p[y]$ indexes the $_y$ -th element of the vector $\textbf{\emph{p}}$ , $\mathcal{L}_{h}$ denotes the combined hinge loss over positive and negative query-video pairs, $(s_{q},v)$ is a positive pair while $(s_{q},\hat{\pmb{v}}),(\hat{s_{q}},\pmb{v})$ are negative ones replaced with one other sample in $\pmb{v}$ and $s_{q}$ respectively, $\delta$ is the margin hyper-parameter and $\lambda_{1},\lambda_{2}$ are balancing factors.  

• Frame Order Modeling (FOM) is treated as a classifcation problem in HERO [126], which targets reconstructing the timestamps of selected video frames. The objective of FOM is defned as follows:  

$$
\mathcal{L}_{F O M}=-\mathbb{E}_{D}\sum_{i=1}^{R}l o g{\cal P}[r_{i},t_{i}]
$$  

where the number of reordered frames is $R$ , $i\ \in$ $[1,R],t_{i}\in\{1,...,N_{v}\}$ , $r_{i}$ is the reorder index, $\pmb{P}\in$ RNv×Nv is the probability matrix.  

• Textual Aspect-Opinion Extraction (AOE) aims to extract aspect and opinion terms from the text, as noted in [127]. To handle the lack of label information required for supervised learning, the authors resort to other models for aspect extraction and opinion extraction. The obtained aspect and opinion terms are treated as labels for the AOE task.  

• Visual Aspect-Opinion Generation (AOG) targets at generating the aspect-opinion pair detected from the input image [127].  

Multimodal Sentiment Prediction (MSP) enhance the pre-trained models by capturing the subjective information from visionlanguage inputs [127].  

• Modality-Level Masking (MoLM) is used in [22] to learn the alignment among the text, vision, and audio. The authors mask out each modality independently with a certain probability.  

• Structural Knowledge Masking (SKM) is proposed in [128] which attempts to mask the tokens selectively based on the cue provided by the knowledge entry. The masking probabilities is calculated to obtain mask indices $M_{w}$ and $M_{r}$ for each knowledge entry, the two items denote the words of sentences and visual regions of images need to be masked, respectively. The loss function of Structural Knowledge Masking Language Model can be formulated as:  

$$
\mathcal{L}_{S K M L M}(\theta)=-\mathbb{E}_{(W,R)\sim D}l o g P_{\theta}(\mathcal{W}_{M_{w}}|\mathcal{W}_{\backslash M_{w}},\mathcal{R}_{\backslash M_{r}})
$$  

where $\theta$ is the parameters. ${\mathscr W}_{\backslash M_{w}}$ and $\mathcal{R}_{\backslash M_{r}}$ represent the non-masked words of sequences and the remaining regions of images, respectively.  

# 3.5 Pre-training Network Architecture  

# 3.5.1 Self-attention and Transformer  

In the large-scale pre-training era, most of current pre-trained models are inspired by the Transformer (which is mainly consisted of self-attention layers). It is originally developed for natural language processing tasks in 2017 [9] which sets new SOTA performance on many downstream tasks by a large margin. Such framework is also introduced into the computer vision community, therefore, the design of unifed network architectures for various tasks and inputs is the current research hotspot.  

Given the input x, an attention module $\mathrm{A}(\mathrm{x})$ is used to generate attention weights, then, some procedures are conducted based on input x and $\mathrm{A}(\mathrm{x})$ to get the attended input $\mathrm{\boldmath~x~}^{\!\prime}\,=\,\mathrm{f}(\mathrm{A}(\mathrm{x}),\mathrm{\boldmath~x~})$ . Many attention models are designed based on this idea, such as the channel attention, spatial attention, temporal attention, branch attention [129]. The self-attention scheme is a special case of attention mechanism, as shown in Fig. 6. More in detail,  

$$
\begin{array}{l}{{Q,K,V=L i n e a r(x)}}\\ {{A(x)=S o f t m a x(Q K)}}\\ {{f(A(x),x)=A(x)V}}\end{array}
$$  

where the Linear denotes fully connected layers. On the basis of self-attention, the work mechanism of multi-head attention is the aggregation of parallel attention layers. Mathematically speaking,  

$$
\begin{array}{l}{{M u l t i H e a d(Q,K,V)=[h e a d_{1},...,h e a d_{h}]W^{O}}}\\ {{\ }}\\ {{h e a d_{i}=A t t e n t i o n(Q W_{i}^{Q},K W_{i}^{K},V W_{i}^{V})}.}\end{array}
$$  

where [, ] denotes the concatenate operation, $W_{i}^{Q},W_{i}^{K},W_{i}^{V}$ and $W^{O}$ are parameter matrices.  

![](images/e73269b9b1717cda06e10acf5cf40e39b01b0f03225cb87e12ae8ccee24c6601.jpg)  
Fig. 6 An illustration of multi-head self-attention (MHSA) [9].  

# 3.5.2 Single- and Multi-stream  

The multi-layer transformer is widely used in many current MM-PTMs. The input of each modality is frst extracted as feature embeddings by the independent encoder and then interacted with other modalities. According to the manner of multi-modal information fusion, two categories of MM-PTMs can be concluded, i.e., single- and cross-stream. In this subsection, we will present these two architectures separately.  

• Single-stream Multi-modal inputs such as images and text are treated equally and fused in a unifed model. The uni-modal features extracted from each modality are tokenized and concatenated by the separators as the input of the multi-modal transformer for multi-modal fusion, as shown in Fig. 8(a). In the transformer, the MHSA (multi-head self-attention) mechanism is usually adopted to interactively fuse the unimodal features, then, the multi-modal fusion features are output from the class token of the transformer. Large-scale MM-PTMs based on single-stream structure includes VL PTMs (e.g., Oscar [17] and ALBEF [130]) and vision-languageaudio pre-training model OPT [22]. Single-stream pre-training models perform token-level matching based on strong semantic correlation, e.g. object features of the image are matched with semantic features of object tags. It provides realistic interaction between uni-modal features, and multi-modal fusion features contain information from diferent modalities with better characterization capability.  

Cross-stream Features of diferent modalities are extracted in parallel by independent models and then are aligned by self-supervised contrastive learning in cross-stream architecture. The pre-training models obtain aligned uni-modal features rather than fused multi-modal features. As shown in Fig. 8(b), multi-modal fusion features are obtained by concatenating uni-modal features and fed into a MLP (Multi-Layer Perceptron) for pretraining objective learning. Representative largescale MM-PTMs based on cross-stream structure include BriVL [106] and CLIP [77], etc. Compared with pre-training models based on single-stream, cross-stream models align diferent modality features into a consistent high-dimensional feature space, such as text semantics and visual image representation. Cross-stream pre-training models generally contain the CS pre-training objective and achieve embedding-level matching based on “weak semantic correlation” [106]. The structure of cross-stream models is more fexible, and modifying the branching structure of one modality of the model does not afect other modalities, making it easy to deploy in real scenarios. However, cross-stream models extract the aligned multimodal common features, and how to efectively exploit the information diferences and complementarity between multi-modal data is an issue to be studied.  

In addition, depending on the needs of the pretraining objectives, the structure of pre-training models can be divided into with and without a decoder. If pre-training objectives contain generative tasks, such as masked image reconstruction, generating matching images based on the text description, etc., the pre-training model adds a decoder after the encoder for converting multimodal fusion features into the corresponding output.  

# 3.5.3 Modality Interactive Learning  

Most of current large-scale pre-trained multimodal models adopt concatenate, add, Mergeattention, Co-attention, and Cross-attention [132] to achieve interactive learning between modalities. An introduction to these modules are given in the following paragraphs.  

• Merge-attention: As shown in Fig. 7 (a), a unifed feature representation is obtained by concatenating the input modalities. Then, this feature is fed into the fusion network. For example, the i-Code [131] fatten the visual inputs along the temporal and spatial dimensions. Note that the parameters of this attention model is shared by these input modalities.  

• Co-attention: For the co-attention module, as shown in Fig. 7, each input modality has its own self-attention layers for modality-specifc feature embedding. Then, the multiple embeddings are fused using a cross-attention layer.  

• Cross-attention: For the multi-modal task, the key step is how to design a fusion module to connect the multi-modality inputs efectively. For instance, the cross-attention layer is proposed by Suo et al. [132], which integrate the image and language subtly for visual question answering. Specifcally, they mutually input one modality into the Q-branch of another self-attention network. Then, the output of two modalities are concatenated as one unifed representation for fnal prediction.  

Tangled-transformer: The TaNgled Transformer (TNT) [133] is proposed to handle the action-, regional object-, and linguisticfeatures, simultaneously, using three Transformer modules. As shown in Fig. 7 (d), the authors inject one modality to the Transformer network designed for other modality to enhance the interactions.  

• Inter-Modality Contrastive Learning: The contrastive learning is widely used for intermodality relation modelling, such as the CLIP [77] and its following-up works [19, 104, 134–138]. The representative work SCALE [104] is trained with Self-harmonized Inter-Modality Contrastive Learning (SIMCL), which can be written as:  

$$
\mathcal{L}_{C L}(d_{i}^{(0)},d_{i}^{(1)})=-l o g\frac{e x p(S i m(f_{i}^{(0)},f_{i}^{(1)})/\tau)}{\sum_{m=0}^{1}\sum_{k=1}^{N}\mathbf{1}_{[k\neq i]}e x p(S i m(f_{i}^{(m)},f_{k}^{(1-m)})/\tau))}\,,
$$  

where $(d_{i}^{(0)},d_{i}^{(1)})$ is a positive pair, and the pairing of d(0) and other samples will bring us negative training data. $f_{i}^{(0)},f_{i}^{(1)}$ are feature embedding of $(d_{i}^{(0)},d_{i}^{(1)})$ respectively. The $S i m$ denotes the cosine similarity, $\mathbf{1}_{[k\neq i]}$ is the binary indicator function, $\tau$ is a temperature parameter.  

# 3.6 Pre-training using Knowledge  

Conventional pre-trained models sufer from poor logical reasoning and lack of interpretability. To alleviate those problems, it is straightforward to involve knowledge, deep understanding of data, in pre-training models, i.e., pre-training using knowledge also known as Knowledge Enhanced Pre-Trained Models (KEPTMs) shown in Fig. 9.  

Knowledge Representation Learning By learning to represent symbolic knowledge, usually in the form of entities and relations, knowledge representation learning enables neural network based models to fuse knowledge and improve their reasoning capabilities. Similarity-based models and graph neural network (GNN) models are two major methods of knowledge representation learning.  

Similarity-based Models Given similarity-based scoring functions, similaritybased models measure the similarity of latent semantics between two entities. Translation-based models are representatives of similarity-based models, as the distance in the vector space is often used to describe the similarity. TransE frstly models relations by translations, which operates on entity embeddings at low-dimension [197]. To deal with mapping properties of relations efciently in complex models, such as refexive, one-to-many, many-to-one and many-to-many,  

Table 3 The summary of mainstream multi-modal pre-trained big models (Part-I).   


<html><body><table><tr><td>No.</td><td>Model</td><td>Pub.</td><td>Modality</td><td>Architecture</td><td>Objective</td><td>Highlights</td><td>Parameters</td><td>Code</td></tr><tr><td>01</td><td>VisualBERT [139]</td><td>arXiv-2019</td><td>image-text</td><td>Trans, BERT</td><td>GR,MML</td><td>A simple and strong baseline for VLP</td><td>170M</td><td>URL</td></tr><tr><td>02</td><td>ViLBERT [140]</td><td>NeurIPS-2019</td><td>image-text</td><td>Trans</td><td>CS,GR</td><td>First adopt co-attention for MM pre-training</td><td>274M</td><td>URL</td></tr><tr><td>03</td><td>LXMERT [117]</td><td>EMNLP-2019</td><td>image-text</td><td>Trans</td><td>QA,MOR,MOC, MML,MLM</td><td>Propose a cross-modality encoder for vision-language pre-training</td><td>183M</td><td>URL</td></tr><tr><td>04</td><td>B2T2 [141]</td><td>EMNLP-2019</td><td>image-text</td><td>ResNet, BERT</td><td>MML,GR</td><td>Embed bounding box into text transformer in a early fusion manner</td><td></td><td>URL</td></tr><tr><td>05</td><td>Unicoder-VL [114]</td><td>AAAI-2020</td><td>image-text</td><td>Trans</td><td>GR,MML,MOC</td><td>Single transformer encoder for VLP</td><td>170M</td><td>URL</td></tr><tr><td>06</td><td>VL-BERT [142]</td><td>ICLR-2019</td><td>image-text</td><td>BERT</td><td>GR,MOC</td><td>MM PTMs and faster rcnn are jointly trained</td><td></td><td>URL</td></tr><tr><td>07</td><td>VLP [143]</td><td>AAAI-2020</td><td>image-text</td><td>Trans</td><td>BiDT, Seq2seq</td><td>Unified encoder-decoder network architecture</td><td></td><td>URL</td></tr><tr><td>08</td><td>UNITER [18]</td><td>ECCV-2020</td><td>image-text</td><td>Trans</td><td>MRA,MML</td><td>Propose an OT-based Word- Region Alignment objective Training jointly on 12 different</td><td>110M</td><td>URL</td></tr><tr><td>60</td><td>12-IN-1 [144]</td><td>CVPR-2020</td><td>image-text</td><td>Trans</td><td>CS,GR</td><td>datasets in a multi-task learning manner</td><td>270M</td><td>URL</td></tr><tr><td>10</td><td>VisDial-BERT [145]</td><td>ECCV-2020</td><td>image-text</td><td>Trans</td><td>MLM, NSP, MIR</td><td>Pre-training on image-text corpus and finetuning on visual dialog</td><td></td><td>URL</td></tr><tr><td>11</td><td>ImageBERT [87]</td><td>arXiv-2020</td><td>image-text</td><td>Trans</td><td>MOC, MLM, MML,MOR</td><td>Indicating that multi-stage pre-training works better</td><td>170M</td><td></td></tr><tr><td>12</td><td>PREVALENT [122]</td><td>CVPR-2020</td><td>image-text</td><td>Trans</td><td>MLM, AP</td><td>Pre-training for vision and language navigation Novel IDA pre-training;</td><td></td><td>URL</td></tr><tr><td>13</td><td>XGPT [11]</td><td>NLPCC-2021</td><td>image-text</td><td>Trans</td><td>IC, MLM, IDA, MOR</td><td>Share parameters between encoder and decoder</td><td></td><td></td></tr><tr><td>14</td><td>InterBERT [115]</td><td>arXiv-2020</td><td>image-text</td><td>Trans</td><td>MSM,MOC, ITM-hn</td><td>Finding that all-attention works better than co-attention for modal interaction</td><td>173M</td><td>URL</td></tr><tr><td>15</td><td>PixelBERT [20]</td><td>arXiv-2020</td><td>image-text</td><td>CNN, Trans</td><td>MLM, MML</td><td>First to align vision and language in pixel and text-level</td><td>142M</td><td></td></tr><tr><td>16</td><td>OSCAR [17]</td><td>ECCV-2020</td><td>image-text</td><td>Trans</td><td>CS,MLM</td><td>Align the visual patches with word embeddings by using object tags as anchor points</td><td>155M</td><td>URL</td></tr><tr><td>17</td><td>pyramidCLIP [146]</td><td>arXiv-2022</td><td>image-text</td><td>CNN+Trans</td><td>CS</td><td>Hierarchical image-text contrastive learning</td><td></td><td></td></tr><tr><td>18</td><td>FashionBERT [147]</td><td>RDIR-2020</td><td>image-text</td><td>BERT</td><td>MLM, MOR,MML</td><td>Use image patches for fashion domain instead of Rols</td><td></td><td>URL</td></tr><tr><td>19</td><td>VILLA [148]</td><td>NeurIPS-2020</td><td>image-text</td><td>Trans</td><td>MLM, MOR,MML</td><td>Pre-training with adversarial learning</td><td></td><td>URL</td></tr><tr><td>20</td><td>ERNIE-ViL[123]</td><td>AAAI-2021</td><td>image-text</td><td>Trans</td><td>MOC, AttP,RelP, MLM,MOR,MML</td><td>Use the knowledge obtained from scene graph</td><td></td><td>URL</td></tr><tr><td>21</td><td>KVL-BERT [149]</td><td>KBS-2021</td><td>image-text</td><td>BERT</td><td>MOC,MLM</td><td>Integrate commonsense knowledge for visual commonsense reasoning</td><td></td><td></td></tr><tr><td>22</td><td>VinVL [113]</td><td>CVPR-2021</td><td>image-text</td><td>Trans</td><td>MTL, 3-way CS</td><td>Verifying that visual feature matters in VLP, i.e., strong object detector brings</td><td>157M</td><td>URL</td></tr><tr><td>23</td><td>VL-T5 [150]</td><td>ICML-2021</td><td>image-text</td><td>Trans</td><td>MLM, VQA, MML, VG,GC</td><td>better results Unified framework for VL via generating texts</td><td>400M</td><td>URL</td></tr><tr><td>24</td><td>ViLT [151]</td><td>ICML-2021</td><td>image-text</td><td>Trans</td><td>MLM,MML</td><td>Use linear embedding only for Fast VL transformer</td><td>87M</td><td>URL</td></tr><tr><td>25</td><td>ALIGN [21]</td><td>ICML-2021</td><td>image-text</td><td>EfficientNet, BERT</td><td>CS</td><td>Milestone for image-text pre-training using noisy data</td><td>300M</td><td></td></tr><tr><td>26</td><td>Kaleido-BERT [124]</td><td>CVPR-2021</td><td>image-text</td><td>Trans</td><td>MLM,MML, AKPM</td><td>Use saliency detector to generate multi-grained patches</td><td></td><td>URL</td></tr><tr><td>27</td><td>MDETR [152]</td><td>ICCV-2021</td><td>image-text</td><td>CNN+Trans</td><td>STP, MML</td><td>A text-modulated detection system which can be trained in an end to end way</td><td></td><td>URL</td></tr><tr><td>28</td><td>SOHO [153]</td><td>CVPR-2021</td><td>image-text</td><td>CNN+Trans</td><td>MLM, MOR,MML</td><td>Use a dynamic-updated visual dictionary for vision-language alignment</td><td></td><td>URL</td></tr><tr><td>29</td><td>E2E-VLP [125]</td><td>ACL-2021</td><td>image-text</td><td>Trans</td><td>OBD,ITG</td><td>The first PTM for vision-language understanding and generation</td><td>94M</td><td></td></tr><tr><td>30</td><td>PIM [154]</td><td>NeurIPS-2021</td><td>image-text</td><td>Trans</td><td>MLM,MML,MOR</td><td>Measure and reveal the V+L fusion using the proposed</td><td>48M</td><td></td></tr><tr><td>31</td><td>CLIP-ViLp[137]</td><td>arXiv-2021</td><td>image-text</td><td>Trans</td><td>MLM, VQA, MML</td><td>inter-modality flow metric Take the CLIP visual encoder as its visual backbone</td><td></td><td>URL</td></tr><tr><td>32</td><td>ALBEF [130]</td><td>NeurIPS-2021</td><td>image-text</td><td>Trans</td><td>CS,GR</td><td>Design a momentum model to address noisy data Simple VL model using</td><td>210M</td><td>URL</td></tr><tr><td>33</td><td>SimVLM [116]</td><td>arXiv-2021</td><td>image-text</td><td>Trans</td><td>PrefixLM</td><td>single PrefixLM pre-training objective only</td><td></td><td></td></tr><tr><td>34</td><td>MURAL [155]</td><td>arXiv-2021</td><td>image-text</td><td>Trans</td><td>CS</td><td>Adopt multi-task contrastive learning objective (image-text,text-text)</td><td>430M</td><td></td></tr><tr><td>35</td><td>VLMo [156]</td><td>arXiv-2021</td><td>image-text</td><td>Trans</td><td>MLM,MML,CS</td><td>Jointly learns visual-,</td><td></td><td>URL</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>text-encoder and a fusion encoder</td><td></td><td></td></tr></table></body></html>  

Table 4 The summary of mainstream multi-modal pre-trained big models (Part-II).   


<html><body><table><tr><td>No.</td><td>Model</td><td>Pub.</td><td>Modality</td><td>Architecture</td><td>Objective</td><td>Highlights</td><td>Params</td><td></td></tr><tr><td>36</td><td>METER [157]</td><td>CVPR-2022</td><td>image-text</td><td>Trans</td><td>MLM, MOR, MOC,MML</td><td>An empirical study on VLP</td><td></td><td>URL</td></tr><tr><td>37</td><td>VideoBERT [158]</td><td>ICCV-2019</td><td>video-text</td><td>BERT</td><td>MLM</td><td>A simple model for video-text feature learning</td><td></td><td>URL</td></tr><tr><td>38</td><td>CBT [159]</td><td>arXiv-2019</td><td>video-text</td><td>Trans</td><td>NCE</td><td>Self-supervised contrastive bidirectional Transformer</td><td>15M</td><td></td></tr><tr><td>39</td><td>UniVL [160]</td><td>arXiv-2020</td><td>video-text</td><td>Trans</td><td>MLM, MFM, MML,ITG</td><td>A unified model for multimodal understanding and generation Hierarchical Transformer-based</td><td></td><td>URL</td></tr><tr><td>40</td><td>HERO [126]</td><td>EMNLP-2020</td><td>video-text</td><td>Trans</td><td>MLM, MFM, VSM,FOM</td><td>model trained with newly proposed VSM and FOM</td><td></td><td>URL</td></tr><tr><td>41</td><td>MMFT-BERT [161]</td><td>EMNLP-2020</td><td>image-text</td><td>BERT</td><td>Classification</td><td>Adopt multiModal fusion Transformer for modality fusion</td><td></td><td>URL</td></tr><tr><td>42</td><td>ActBERT[133]</td><td>CVPR-2020</td><td>image-text</td><td>Trans</td><td>CS,GR</td><td>Extract actions explicitly as one of the inputs</td><td></td><td></td></tr><tr><td>43</td><td>CLIP [77]</td><td>ICML-2021</td><td>image-text</td><td>Resnet, Trans</td><td>SO</td><td>Milestone for image-text pre-training using noisy data</td><td>88.6M</td><td>URL</td></tr><tr><td>44</td><td>Frozen [92]</td><td>ICCV-2021</td><td>video/image-text</td><td>Trans</td><td>MML</td><td>Jointly optimize the model on both images and videos</td><td>180.4M</td><td>URL</td></tr><tr><td>45</td><td>RegionLearner [162]</td><td>arXiv-2021</td><td>video-text</td><td>Trans</td><td>MML</td><td>Implicitly learning object region without position supervision Adapt to single-, multi-modal</td><td></td><td>URL</td></tr><tr><td>46</td><td>UNIMO [163]</td><td>arXiv-2020</td><td>image-text</td><td>Trans</td><td>CS</td><td>understanding and generation tasks effectively</td><td></td><td>URL</td></tr><tr><td>47</td><td>DALL-E [164]</td><td>ICML-2021</td><td>image-text</td><td>Trans</td><td>ELB</td><td>Achieve high quality image generation without using any of the training labels</td><td>12B</td><td>URL</td></tr><tr><td>48</td><td>BriVL [106]</td><td>arXiv-2021</td><td>image-text</td><td>Trans</td><td>InfoNCE</td><td>The first Chinese large-scale MM-PTMs</td><td>10B</td><td>URL</td></tr><tr><td>49</td><td>VLC [165]</td><td>arXiv-2022</td><td>image-text</td><td>ViT</td><td>MIM, MLM ITM</td><td>Built on top of MAE that does not require trained on ImageNet</td><td>87M</td><td>URL</td></tr><tr><td>50</td><td>M6 [103]</td><td>arXiv-2021</td><td>image-text</td><td>Trans</td><td>LM</td><td>The largest pretrained model in Chinese The first open-source</td><td>100B</td><td></td></tr><tr><td>51</td><td>CogView [166]</td><td>NeurIPS-2021</td><td>image-text</td><td>Trans</td><td>NLL</td><td>large text-to-image transformer</td><td>4B</td><td>URL</td></tr><tr><td>52</td><td>VATT [167]</td><td>NeurIPS-2021</td><td>Video,Audio, Text</td><td>Trans</td><td>NCE, MIL-NCE</td><td>Modality-specific or Modality-agnostic triplet modality pre-trained model</td><td>306.1M</td><td>URL</td></tr><tr><td>99</td><td>OPT [22]</td><td>arXiv-2021</td><td>image, Audio, Text</td><td>Trans</td><td>MLM, MVM, MoLM MAM, DTR, DIR</td><td>The first model pre-trained using triplet modalities</td><td></td><td></td></tr><tr><td>54</td><td>Florence [168]</td><td>arXiv-2021</td><td>image-text</td><td>CoSwin</td><td>UniCL</td><td>Multi-dimensional expansion of representations</td><td>893M</td><td></td></tr><tr><td>99</td><td>ROSITA [128]</td><td>MM-2021</td><td>image-text</td><td>Trans</td><td>SKM, MLM, MRM</td><td>Fuse the intra-, cross-modality knowledge, and SKM</td><td></td><td></td></tr><tr><td>56</td><td>VLCDoC [169]</td><td>arXiv-2022</td><td>image-text</td><td>Trans</td><td>CS</td><td>Contrastive Pre-Training for document classification Multimodality-guided visual</td><td></td><td></td></tr><tr><td>57</td><td>MVP [170]</td><td>arXiv-2022</td><td>image-text</td><td>ViT</td><td>MIM</td><td>pre-training leads to impressive gains</td><td></td><td></td></tr><tr><td>58</td><td>GilBERT [171]</td><td>IR-2021</td><td>image-text</td><td>BERT</td><td>MLM, MOR</td><td>Considers both realistic and synthetic data for VLP</td><td></td><td></td></tr><tr><td>59</td><td>COTS [172]</td><td>arXiv-2022</td><td>image-text</td><td>Trans</td><td>CS,KLD, MVLM</td><td>Token- and task-level interaction are proposed to enhance cross-modal interaction</td><td></td><td></td></tr><tr><td>60</td><td>U-VisualBERT [173]</td><td>NAACL-2021</td><td>image-text</td><td>Trans, BERT</td><td>GR, MML</td><td>Unpaired image-text data for pre-training</td><td></td><td>URL</td></tr><tr><td>61</td><td>Flamingo [174]</td><td>arXiv-2022</td><td>image-text</td><td>NFNet</td><td>CS</td><td>Pre-training on interleaved visual and text data as input</td><td>80B</td><td>URL</td></tr><tr><td>62</td><td>M3P [175]</td><td>CVPR-2021</td><td>image-text</td><td>BERT</td><td>xMLM, MC-MLM, MC-MRM</td><td>Multitask, Multilingual, Multimodal Pre-training</td><td></td><td>URL</td></tr><tr><td>89</td><td>BLIP [176]</td><td>arXiv-2022</td><td>image-text</td><td>BERT</td><td>CS,MML,MLM</td><td>Propose the multimodal mixture of encoder-decoder,and captioning-filtering scheme</td><td>224M</td><td>URL</td></tr><tr><td>64</td><td>NUWA [177]</td><td>arXiv-2021</td><td>image-text</td><td>Trans</td><td>T2I,T2V,V2V</td><td>A 3D transformer framework can handle image, text, and video, simultaneously</td><td>809M</td><td>URL</td></tr><tr><td>99</td><td>TCL [178]</td><td>CVPR-2022</td><td>image-text</td><td>BERT</td><td>CMA,IMC,LMI ITM, MLM</td><td>The first work considers local structure information for multi-modality</td><td>123.7M</td><td>URL</td></tr><tr><td>66</td><td>SCALE [179]</td><td>CVPR-2022</td><td>image,text, table video, audio</td><td>BERT</td><td>MRP, MLM, MEM MFP, MFP, MAM</td><td>representation learning A unified model to handle five modalities</td><td></td><td>URL</td></tr><tr><td>67</td><td>Clinical-BERT [180]</td><td>AAAI-2022</td><td>image-text</td><td>BERT</td><td>CD,MMM MLM, IMM</td><td>The first work to learn domain knowledge during pre-training for</td><td>102M</td><td></td></tr><tr><td>68</td><td>RegionCLIP [181]</td><td>CVPR-2022</td><td>image-text</td><td>Trans</td><td>Distillation loss, CS</td><td>the medical domain Learn region-levelvisual representations based on CLIP</td><td>=</td><td>URL</td></tr><tr><td>69</td><td>ProbES [182]</td><td>ACL-2022</td><td>image-text</td><td>LSTM,ViLBERT</td><td>Ranking loss</td><td>Prompt-based learning for VLN based on CLIP Unifying the object detection</td><td></td><td>URL</td></tr><tr><td>70</td><td>GLIP [183]</td><td>CVPR-2022</td><td>image-text</td><td>BERT</td><td>CS</td><td>and grounding into a</td><td>394M</td><td>URL</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>unified framework</td><td></td><td></td></tr></table></body></html>  

![](images/360620bceaa6e4bb7bff40f038bd7b675350f39bcb4151533941d7003b8d9a13.jpg)  
Fig. 7 The widely used modality interactive learning modules for MM-PTMs. (a) Merge-attention [131], (b) Co-attention [131], (c) Cross-attention [132], (d) Tangled-transformer [133], and (e) Contrastive learning [77].  

TransH is proposed to model a relation as a translation operation on a hyperplane [198]. TransR is proposed to embed entity and relation in a separated spaces to capture diferent aspects of entities over various relations [199]. Compared with TransR, not only the diversity of relations but also entities are considered in TransD [200]. To deal with heterogeneity and imbalance issues brought by knowledge graphs but ignored by aforementioned translation-based models, transfer matrices are replaced with adaptive sparse matrices in TranSparse, because the number of entities linked by relations determines sparse degrees [201]. Besides translation-based models, tensor or matrix factorization approaches have also been proposed for multi-relational data by introducing scoring or ranking functions to measure how likely the semantic matching is correct. With the latent components, RESCAL is capable of collective learning and can provide an efcient algorithm of the factorization of a three-way tensor [202]. NTN introduces an expressive neural tensor network for reasoning over relationships between two entities [203]. DistMult presents a general framework for multi-relational learning and shows the efectiveness of a simple bilinear formulation [204]. SME designs a new neural network architecture to encode multi-relational graphs or tensors into a fexible continuous vector space, so that multi-relational semantics can be learnt [205]. HolE is proposed to learn compositional vector space representations of entire knowledge graphs by employing holographic models of associative memory and circular correlation to create compositional representations [206].  

• Graph Neural Network Models To further leverage the structure of the graph rather than collections of triplets, graph neural network models are employed to embed entities and relations. As convolutional neural networks (CNNs) are extremely efcient architectures in recognition tasks over diferent domains, they are generalized to graphs based on hierarchical clustering of the domain and the spectrum of the graph Laplacian in [207]. Inspired by the pioneering work, further eforts have been done on graph convolutional networks (GCNs), such as semi-supervised classifcation [208], unsupervised learning based on the variational auto-encoder (VAE) [209], inductive representation learning to sample and aggregate features from a node’s local neighborhood [210], and attention mechanism by leveraging masked self-attentional layers [211]. Beyond GCNs, RGCNs is developed to deal with the highly multirelation data characteristic of realistic knowledge bases [212]. A structure-aware convolutional network (SACN) takes the beneft of GCN and ConvE [213] together, where GCN as the encoder utilizes knowledge graph node structure and ConvE as the decoder enables the translational feature [214]. To further enhance Graph Attention Networks (GATs) and capture both entity and relation features within any entity’s neighborhood, another model is proposed for attentionbased feature embedding [215]. To leverage various composition operations for embedding entities and relations in KGs and ever-increasing number of relations, a composition-based GCN named CompGCN is proposed to embed both nodes and relations jointly [216].  

le 5 The summary of mainstream multi-modal pre-trained big mode   


<html><body><table><tr><td>No.</td><td>Model</td><td>Pub.</td><td>Modality</td><td>Architecture</td><td>Objective</td><td>Highlights</td><td>Parameters</td><td>Code</td></tr><tr><td>71</td><td>VLP-MABSA [127]</td><td>ACL-2022</td><td>image-text</td><td>BERT</td><td>MLM,AOE, MRM AOG,MSP</td><td>Task-specific VL-PTMs for multimodal aspect-based sentiment analysis</td><td></td><td>URL</td></tr><tr><td>72</td><td>R2D2 [184]</td><td>arXiv-2022</td><td>image-text</td><td>ViT,BERT</td><td>GCPR,FGR,MLM</td><td>A two-way distillation strategy is proposed, i.e., target- and feature-guided distillation</td><td></td><td></td></tr><tr><td>73</td><td>DeCLIP [19]</td><td>ICLR-2022</td><td>image-text</td><td>ViT</td><td>InfoNCE,SS MVS, NNS</td><td>Learn generic visual features in a data efficient way</td><td>276M</td><td>URL</td></tr><tr><td>74</td><td>DeFILIP [136]</td><td>arXiv-2022</td><td>image-text</td><td>ViT,ResNet</td><td>CS</td><td>A benchmark for CLIP and its variants</td><td></td><td>URL</td></tr><tr><td>75</td><td>SLIP [185]</td><td>arXiv-2021</td><td>image-text</td><td>ViT</td><td>CS,InfoNCE</td><td>Combine the self-supervised learning and CLIP pre-training in a multi-task framework</td><td>38M</td><td>URL</td></tr><tr><td>76</td><td>FILIP [186]</td><td>arXiv-2021</td><td>image-text</td><td>ViT</td><td>CS</td><td>Cross-modal interactive learning for finer-level alignment</td><td></td><td></td></tr><tr><td>77</td><td>SemVLP [187]</td><td>arXiv-2021</td><td>image-text</td><td>Trans</td><td>MLM, MOP, ITM,QA</td><td>Fuse the single- and two-stream architectures</td><td>2.1B</td><td></td></tr><tr><td>78</td><td>CoCa [188]</td><td>arXiv-2022</td><td>image-text</td><td>Trans</td><td>CS,ITG</td><td>Jointly pre-train image text model with contrastive loss and captioning loss</td><td></td><td></td></tr><tr><td>79</td><td>HiVLP [189]</td><td>arXiv-2022</td><td>image-text</td><td>Trans</td><td>LRM,HRL, VLM</td><td>Accelerate image-text retrieval via hierarchical retrieval</td><td></td><td></td></tr><tr><td>80</td><td>CLIP-Event [135]</td><td>CVPR-2022</td><td>image-text</td><td>Trans</td><td>CS</td><td>Consider event structural knowledge and prompts in the pre-training phase.</td><td></td><td>URL</td></tr><tr><td>81</td><td>AudioCLIP [190]</td><td>ICASSP-2022</td><td>image-text-audio</td><td>Trans</td><td>CS</td><td>Build a triplet modality based PTMs like CLIP</td><td>30M</td><td>URL</td></tr><tr><td>82</td><td>VL-BEiT [191]</td><td>arXiv-2022</td><td>image-text</td><td>Trans</td><td>MLM, MIM, MVLM</td><td>Share the Transformer network on both monomodal- and multimodal-data</td><td></td><td>URL</td></tr><tr><td>83</td><td>MV-GPT [192]</td><td>arXiv-2022</td><td>image-text</td><td>BERT</td><td>MLM, LG</td><td>Pre-train both a multi-modal video encoder and a sentence decoder jointly.</td><td>117M</td><td></td></tr><tr><td>84</td><td>MMKD [193]</td><td>arXiv-2022</td><td>image-text</td><td>BERT</td><td>ITM</td><td>Iteratively execute knowledge discovery and model pre-training for continuous learning</td><td></td><td></td></tr><tr><td>85</td><td>GLIPv2 [194]</td><td>arXiv-2022</td><td>image-text</td><td>Swin,BERT</td><td>PGL,CS,MLM</td><td>Serves both the localization and understanding tasks.</td><td></td><td>URL</td></tr><tr><td>86</td><td>LIMoE [195]</td><td>arXiv-2022</td><td>image-text</td><td>Trans</td><td>CS</td><td>multi-modal pre-training with a sparse mixture of experts model</td><td>675M</td><td></td></tr><tr><td>87</td><td>VLMixer [196]</td><td>arXiv-2022</td><td>image-text</td><td>Trans</td><td>MLM,CMCL,MTM</td><td>Implicit cross-modal alignment learning in unpaired VLP.</td><td></td><td>URL</td></tr><tr><td>88</td><td>ProtoCLIP [138]</td><td>arXiv-2022</td><td>image-text</td><td>Trans</td><td>CS</td><td>Combine the CLIP loss and prototypical supervisions for VLP.</td><td></td><td>URL</td></tr><tr><td>89</td><td>i-Code [131]</td><td>arXiv-2022</td><td>image-text-audio</td><td>Trans</td><td>MLM, MVM MSM,CS</td><td>It can handle different combinations of modalities (such as single-, dual-, and triple-modality) into a single representation space.</td><td>906M</td><td></td></tr></table></body></html>  

![](images/b02d8a4351c68fe1e4b4e21f64c70e6c12ba04f1f7e9d481a0eadd2ea648ad04.jpg)  
(a) Architecture of single-stream pre-training multimodal model  

![](images/2f91ab9b1811e18651bd2a18fa45046ef95420185f80f8f68f40c90495a236b4.jpg)  
(b) Architecture of Cross-stream pre-training multimodal model   
Fig. 8 Pre-training network architecture.  

Knowledge Fusion Methods How to fuse knowledge into pre-trained models and improve their logical understanding of data after knowledge representation learning remains a challenge to researchers. According to the category of knowledge provided, KEPTMs roughly contain two categories: unstructured knowledge and structured knowledge enhanced pre-trained models.  

• Unstructured KEPTMs Unstructured knowledge often refers to the knowledge without structures involved, which is in the form of plain text, like the words or phrases. Although some literatures introduce entities as supervised data and achieve promising performance, structural information is ignored while only entities are used to enable PTMs to learn semantics or attain extra key features from them. Word-aligned attention aligns the character-level attention to the word level to exploit explicit word information in Chinese [217]. SentiLARE also introduces part-of-speech tag and sentiment polarity to build word-level linguistic knowledge [218]. As unstructured text trained neural language models can store knowledge implicitly, PTMs can be further fne-tuned to explicitly retrieve knowledge without access to external knowledge or context [219].  

• Structured KEPTMs Contrary to unstructured KEPTMs, structured KEPTMs take account of sorts of structural information, including syntax-tree, rules and knowledge graphs. Syntax-BERT incorporates syntax trees efectively and efciently into pre-trained Transformers [220]. LIMIT-BERT learns language representations across multiple linguistics tasks including constituent and dependency syntactic parsing [221]. Syntax-GNN is proposed to learn syntax representations by using dependency trees and fusing the embeddings into transformers [220]. Knowledge graphs (KGs) provide structural knowledge in the form of entities and relations between them. An enhanced language representation model ERNIE is trained by utilizing both large-scale textual corpora and knowledge graphs, so that it can simultaneously leverage lexical, syntactic and knowledge [222]. Similar work named KnowBert is also proposed for largescale models to embed multiple knowledge bases with entity linkers, which retrieves relevant entity embeddings and updates contextual word representations by the word-to-entity attention [223]. Moreover, the reasoning capability is also developed by fnding supporting-facts, based on a large external knowledge base [224, 225]. Rules, in the form of constraints or even logical expressions, are preferred due to their interpretability and accountability. HEX graphs are proposed to enhance existing models by capturing semantic relations between labels applied to the same object [226].  

Knowledge Evaluation Tasks Besides conventional performance metrics, more knowledgeoriented tasks are required to evaluate the capability of KEPTMs and inspect whether external knowledge really helps models understand data semantically. Knowledge evaluation tasks are severed as testbeds to ensure the efectiveness of knowledge fusion methods. Currently, knowledge evaluation tasks mainly focus on NLP tasks and can be categorized into two groups based on the types of required knowledge: factual knowledge and commonsense knowledge evaluation tasks.  

![](images/11a77e62a838f25eae0ca3a5148bce45bbce03e6ca85e12c3ff12fc6119e4803.jpg)  
9 The taxonomy of Knowledge Enhanced Pre-Trained Models (KEP  

• Factual Knowledge Evaluation Tasks Factual knowledge is the knowledge of facts, including specifc details and elements to describe the objective facts [28]. Factual knowledge evaluation tasks focus on testing models’ reasoning ability on factual knowledge over various domains, like answering questions by giving a fact or judging the correctness of a given fact. Natural Questions is the frst large publicly available dataset and robust metrics are also introduced to evaluate the performance of question answering (QA) systems [227]. HotpotQA, another QA dataset, provides supporting facts at sentence-level for reasoning and new factoid comparison questions [228]. Diferent from the above two open-domain QA tasks, BoolQ only involves yes/no naturally occurring questions, namely verifying facts generated in unprompted and unconstrained settings, but those queries involve with complicated and nonfactoid information so that make it unexpectedly challenging [229]. Another fact extraction and verifcation task FEVER is proposed and a new type of claims NotEnoughInfo is introduced beside Supported and Refuted [230]. Entity linking, linking entities from a knowledge base to the corresponding textual mentions in a corpus, can also evaluate how well a model understands the factual knowledge [231].  

• Commonsense Knowledge Evaluation Tasks Commonsense knowledge refers to the information generally accepted by the majority of people concerning everyday life, i.e. the practical knowledge about how the world works [29]. Like factual knowledge evaluation tasks, Commonsense QA also focuses on QA, but such QA requires prior knowledge outside the given document or context [232]. To extend the QA task Abductive Natural Language Inference ( $\alpha$ NLI), Abductive Natural Language Generation ( $\alpha$ NLG), a conditional generation task, is also proposed to explain given observations in natural language [233]. CommonGen further explicitly tests models for the ability of generative commonsense reasoning due to its rigorous requirements on both relation reasoning and compositional generalization [234]. Besides general commonsense evaluation tasks evaluating how well models understand daily scenarios, specifc commonsense knowledge ones are further designed for diferent scenarios. SocialIQA, a large-scale benchmark for social commonsense reasoning, is challenging even for PTMs [235]. Beside human interactions, physical interactions are also important in commonsense knowledge, hence the task of PIQA is introduced for physical commonsense reasoning [236]. Temporal commonsense is crucial for understanding the timing of events, for example duration, frequency, and order, leading to correct reasoning. McTaco defnes fve classes of temporal commonsense [237], while TRACIE evaluates models’ temporal understanding of implicit events [238].  

# 3.7 Characteristics of Diferent Pre-trained Big Models  

In the aforementioned paragraphs, we give a review to the main streams of multi-modal pretrained models and highlight the features of each model in Table 3, Table 4, and Table 5. In this subsection, we compare and analyze the characteristics of these models. Specifcally, the early multi-modal pre-trained big models usually design an interactive learning module, for example, the ViLBERT [140], LXMERT [117]. They integrate the co-attention or cross-attention mechanism into their framework to boost the feature representation between multiple inputs. Actually, these models obey the idea of interactive fusion of traditional small models. This allows for seamless integration with numerous downstream tasks and providing a high degree of fexibility. In contrast, many current big models directly process the inputs using projection layers and feed them into a unifed network like the Transformers, including UnicoderVL [114], VideoBERT [158], UniVL [160]. More and more works demonstrate that the powerful Transformer network can achieve comparable or event better performance.  

There are also some works make full use of existing big models and carry out secondary development to achieve a higher performance [181, 190]. To address the issues caused by shortage of paired multi-modal data, some researchers propose to training their model using unpaird data [173]. These models show the great potential of processing massive multi-modal data. Unlike general big models, some models are specifcally designed for a specifc task or domain, like the e-commerce, or Indoor navigation. This provides conditions and convenience for fully mining more detailed domain knowledge assist the pre-training process.  

# 4 Downstream Tasks  

After the pre-training phase, the researchers usually test their model on many downstream tasks to validate the powerful ability. Specifcally, the generative tasks, classifcation tasks, regression tasks are adopted for the validation which will be discussed below. As a new learning paradigm, the prompt learning which target at modifying the downstream tasks to ft the pre-trained big model draws more and more attention. In this part, several representative prompt learning algorithms are also reviewed. An overview of these downstream tasks are visualized in Fig. 10.  

# 4.1 Generative Tasks  

Image/Video Captioning attempt to describe content of input image or video using a couple of sentences. Usually, a visual encoder is used to encode the input image/video, then, a language decoder is adopted for sentence prediction in a word by word manner. NoCaps [239] is proposed by Agrawal et al. in 2019. It is also an image captioning task but focus on developing generalized captioning models.  

<html><body><table><tr><td colspan="2">Classification Tasks</td><td>GenerativeTasks</td></tr><tr><td>VisualQuestionAnswering Video-LanguageInference</td><td>Natural LanguageforVisualReasoning</td><td>Image/VideoCaptioning VisualDialogue Multi-modalMachineTranslation</td></tr><tr><td rowspan="2">VisualCommonsenseReasoning</td><td>VisualEntailment</td><td>RegressionTasks</td></tr><tr><td>CategoryRecognition</td><td>GroundingReferringExpressions</td></tr><tr><td></td><td>Multi-modalSentimentAnalysis</td><td>Spatio-TemporalVideoGrounding</td></tr><tr><td></td><td>Vision-LanguageRetrieval Vision-Language Navigation</td><td>PromptTuning&Others</td></tr><tr><td></td><td>OpticalCharacterRecognition</td><td></td></tr></table></body></html>  

Visual Dialogue (VD) attempt to let the AI agent to talk with humans by holding a meaningful dialog about the visual content [240].  

Multi-modal Machine Translation (MMT) is a task that targets translating the source sentence into a diferent language based on the paired image [241].  

# 4.2 Classifcation Tasks  

Visual Question Answering (VQA) model is provided with an image and a question, and asked to produce an answer [242]. The relations between GQA [86] and VQA is similar to the NoCaps and the standard captioning task. It is introduced to address key drawbacks of previous VQA datasets, and generate novel and diverse questions from a robust question engine, which sufciently considers the content and structure.  

Video-Language Inference (VLI) is proposed by Liu et al. [243] in year 2020, which aims at understanding the video and text multimodal data.  

Natural Language for Visual Reasoning (NLVR) can be seen as a binary classifcation problem. As noted in [244], the model needs to judge the authenticity of a statement for the image.  

Visual Entailment (VE) [245] is a triplet-label classifcation problem derived from Text Entailment (TE) task [246]. The VE model needs to predict whether the given image semantically entails the text. The three labels are entailment, neutral or contradiction.  

Visual Commonsense Reasoning (VCR) [247] is a variation of VQA, which require a machine to provide a rationale justifcation and answer correctly for the given challenging problem.  

Category Recognition (CR) is a classifcation problem which attempt to predict the category of given image. Many computer vision tasks are belong to this downstream task, such as pedestrian attribute recognition [248], action recognition [134].  

Multi-modal Sentiment Analysis (MSA) is a multi-modal fusion task proposed for sentiment analysis [249], which attempt to aggregate various homogeneous and/or heterogeneous modalities for more accurate reason. The modalities can be text, visual and acoustic, etc.  

Vision-Language Retrieval (VLR) can be used in many applications, such as text-based person search [250], or general object retrieval based on language [251].  

Vision-Language Navigation (VLN) [252, 253] is task that the agents learn to navigate in 3D indoor environments following the given natural language instruction. A benchmark for the popular VLN can be found at the following leaderboard.  

Optical Character Recognition (OCR) target at convert the images of Diverse text information into machine-encoded text. Usually, the OCR system contains both text detection and text recognition modules.  

# 4.3 Regression Tasks  

Grounding Referring Expressions (GRE) takes the visual image and language description as input, and output the location of target object described by the language [254–256]. Similar tasks defned on videos are termed Spatio-Temporal Video Grounding (STVG) [257] or Tracking by Natural Language [258–260].  

# 4.4 Prompt Learning  

To make full use of pre-trained big models, the prompt learning (also called prompt tuning) is proposed to re-formulate the downstream tasks to ft the objectives of pre-trained models, including CPT [261], CPL [262]. Also, some prompt tuning schemes are designed to fx the parameters of the large model and adjust the parameters as little as possible to achieve good results, such as the VPT [263], CoOp [264], CoCoOp [265]. To be specifc, the VPT [263] fxes the parameters of ViT models and integrates the prompt vectors as additional input. It achieves good performance even only tune the parameters of classifcation head and prompts. CoOp [264] achieves huge improvements by tuning the context words into a set of learnable prompt vectors. Conditional Context Optimization (CoCoOp) [265] is developed based on CoOp which learns an external network to generate input-conditional tokens for each image. It addresses the issue of class shift signifcantly using such dynamic prompts.  

# 5 Experimental Analysis  

Considering the complexity and numbers of MMPTMs, it is almost impossible to reproduce pretraining tasks in a short amount of time. Therefore, the experiments and related analyses of the pre-training are ignored in this paper. However, we still want to summarize a more complete review paper for the readers, thus, we extract the experimental results of the corresponding downstream tasks from their paper and compare them to the shared benchmark datasets. More detailed results can be found in Table 3 and Table 4.  

# 5.1 Model Parameters and Training Information  

As shown in Fig. 11 (a), the large-scale MM-PTMs are emerging in the year 2019 and the number of papers shows an increasing trend year by year 5 From the Fig. 11 (b), it is easy to fnd that current large-scale PTMs are optimized on servers with more than 8 GPUs. Also, many of them are trained using more than 100 GPUs, such as BriVL (128) [106], VLC (128) [165], M6 (128) [103], SimVLM (512) [116], MURAL (512) [155], CLIP (256) [19], VATT (256) [167], Florence (512) [168], FILIP (192) [186]. Some MM-PTMs are trained on TPUs with massive chips, for example, the largest model of Flamingo [174] is trained for 15 days on 1536 chips. From all these cases, we can see the huge demand of computing power for pre-trained big MM-PTMs.  

![](images/e9795cc822932cd912e8513efdec2e42100a3c0d8ba3ca5fa787955da1c1e471.jpg)  
Fig. 11 (a). Number of MM-PTMs papers published from year 2019 to 2022; (b). Number of GPUs used for pre-training of selected models; (c). Parameters of selected MM-PTMs.  

Based on Fig. 11 (c), it is also easy to fnd that many large-scale MM-PTMs are still with limited parameters, but some of them indeed reached new heights. For example, the DALLE-E [164] (12000 MB), BriVL [106] (10000 MB), M6 [103] (100000 MB), and CogView [166] (4000 MB). The reasons for this phenomenon may be as follows: 1). Many MM-PTMs are trained on several public datasets. The scale of parameters is greatly improved compared to traditional models, but not by a shocking amount. 2). The development of big models is also limited by the need for large-scale computing power, and only a few giant companies or research institutes have such computing power platforms.  

# 5.2 Performance on Representative Downstream Tasks  

Here, we report the experimental results of zeroshot image retrieval, image captioning, and visual question answering. From Fig. 12 (a), we can fnd that the performance of diferent MM-PTMs have a big diference on the zero-shot image retrieval task. The blue and red vertical bar denotes the results of Rank-1 and Rank-5, respectively. Some models achieve high performance on this task which demonstrates the efectiveness of large-scale pre-training. For example, the ALBEF [130] and METER [157] achieves 82.80, 96.30 and 79.60, 94.96 on both evaluation metric.  

For the image captioning task, we can fnd that the compared models achieved close performance on the COCO dataset according to Fig. 12 (b). Specifcally, OSCAR [17] obtains 41.7, 30.6, 140, 24.5; VinVL attains [113] 41, 31.1, 140.9,  

25.2; SimVLM achieves [116] 40.6, 33.7, 143.3, 25.4, respectively. These results are signifcantly better than traditional image captioning models pre-trained in a supervised manner through ImageNet [2] classifcation task. Similar results can also be concluded from Fig. 12 (c).  

# 6 Research Directions  

Although the multi-modal pre-trained big models have obtained huge development, however, it is still a young research direction. Many problems and opportunities are still waiting for researchers to solve. In this section, we summarize several research points which are worthy to be tried.  

Pre-training on More Modalities: Existing large-scale PTMs are usually pre-trained on two modalities, e.g., the vision and language. The missing of large amount aligned multi-modal data may be a key reason. As an old saying goes, “Sharpening your axe will not delay your job of chopping wood”. The acquirement of real multi-modal data is the most important thing for large-scale pre-training, as shown in Fig. 13, such as visual image, text, audio, radar, event streams, depth image, thermal image, etc. To the best of our knowledge, no imaging device can capture so many modalities at the same time. Therefore, the manufacture of multi-modal imaging equipment can be a very signifcant thing. The pre-trained big model based on these data may have a wider potential for applications.  

Incremental Learning based Pretraining: Currently, existing pre-trained big methods are used for downstream tasks through feature fnetuning or prompt learning [266]. This standard deep learning procedure works well in a short time, but pre-training is an expensive process. Specifcally, the collection and cleaning of data, the electric charge used for pre-training, and the hardware device all cost a huge amount of human and material resources. When we gathered another group of data, the pre-training on the mixed data are expensive, redundant, and not environmentally friendly. However, seldom of them consider incremental learning for big models, and it is still unclear if the incremental learning algorithms developed for traditional deep learning work well for big models.  

![](images/36249e8bb5374d8d34c8fa0ea2722682cde079e7664c238de5f83a814738fde9.jpg)  
Fig. 12 Experimental results of selected MM-PTMs on zero-shot image retrieval (Rank-1, Rank-5), image captioning (BLEU, METEOR, CIDEr, SPICE), and visual question answering (Test-std).  

![](images/40ddbe3a05503c3adb2af39a6a83ec1cc92e4c51a172b006b05d2c08e0fe2ac3.jpg)  
Fig. 13 Representative samples of mainstream modalities frequently used.  

In addition to the aforementioned data incremental learning, there are still many aspects that can be exploited for multi-modal pre-trained big modals. For example, the class (or category) incremental learning is a classical machine learning problem. Another interesting problem is modalityincremental learning, in another word, how to introduce and absorb the new modality into the already pre-trained multi-modal model. Because the new sensors (modalities) will appear at some indefnite time in the future, the designed multimodal big models should be fexible enough to handle this situation.  

• Knowledge Enhanced Multi-Modal Pre-training: Based on aforementioned reviews on MM-PTMs, we can fnd that the study of knowledge-assisted pre-training is still in the starting stage. Current works simply adopt external knowledge-graph or knowledge base in the pretraining phase, but they are usually single-modal, independent of multi-modal data, and limited to improving the understanding of data for models. Although commonsense knowledge is more ubiquitous, it is also abstract and introduces ambiguities, leading to challenges when applying to specifc data. Therefore, we believe that further explorations on knowledge enhanced multi-modal pretraining are worth investigating. First, specifed knowledge for multi-modal data is demanded to collect or extract through self-supervised learning. Second, more general knowledge fusion methods designed for multi-modal data are needed, beyond the limitations of vision and language modalities. Third, knowledge evaluation tasks specifc for pre-training are required to inspect the enhancement of knowledge at this early stage, because pre-training is the frst phase of the entire training procedure while downstream tasks are to be determined.  

Fine-grained Multi-Modal Pretraining: Most existing MM-PTMs are pre-trained from a global-view, for example, the researchers adopt the matching between the whole image and language as a supervised signal for the pre-training. The representative works are CLIP [77], ALIGN [21], etc. Note that, the fne-grained local information mining or instancelevel pre-training may further improve the overall performance of multi-modal pre-training. Some researchers have exploited the possibilities of fne-grained pre-training strategies [98]. We hope more researchers can focus on this direction to further boost the fnal results.  

• Multi-Modal Pre-trained Model based Prompt Learning: Current pre-trained big models are usually used in a “pretrain-fnetuning”  

way, specifcally, the users need to initialize their model using pre-trained weights, then, fnetune on downstream tasks. Although it works well in many tasks, however, the fnetune maybe not be the most direct way. Because current multi-modal big models are pre-trained via modality matching, masked token prediction, and the downstream tasks are usually classifcation and regression tasks. Therefore, it exists a gap between multimodal pre-training and fnetuning. Recently, a new framework (termed prompt learning) is developed for big model based downstream tasks, which slickly transforms the setting of downstream tasks to make them consistent with pretraining [266]. Many works have demonstrated its efectiveness [76, 135, 261, 264, 265] in CV and NLP tasks. The research in this direction is also interesting and has great potential.  

• Migration of techniques developed for small-scale models: The small-scale multimodal models have been exploited for many years, and many representative models are proposed for deep multi-modal tasks [267–269]. Among these works, difusion, cross-attention, and dynamic neural networks are useful for specifc multi-modal tasks. Part of these techniques is exploited in VL-PTMs, such as the cross-attention based ViLBERT [140]. There are still many algorithms or tricks that have not yet been explored on large model tasks. We believe the transfer from smallscale to large-scale PTMs is worthy to be studied.  

• Coupling and decoupling problems in cross-modal pre-training models: The coupling involves establishing the correlation between diferent modalities and the “cross” can be only realized through such correlation. The decoupling can further expand the modality dynamically. It is worth studying how to give feasible solutions to the two problems from the aspect of framework design.  

# 7 Conclusion  

We give a comprehensive review of large-scale Multi-Modal Pre-Trained Models (MM-PTMs) in this paper. Firstly, we introduce the background of MM-PTMs, with a focus on conventional deep learning, and pre-training in NLP, CV, and speech. Then, the task defnition, key challenges, and benefts of MM-PTMs are discussed. After that, we dive into the reviews of MM-PTMs and discuss the pre-training data, objectives, networks, knowledge enhanced pre-training, etc. We review the downstream tasks including generative, classifcation, and regression tasks, and also give an overview of model parameters of MM-PTMs and hardware for the pre-training. Experimental results of several representative tasks are also discussed and visualized. Finally, we point out some research directions that are worth to be focused on. We summarize this paper and hope our survey can provide some useful insights for the MM-PTMs.  

# Acknowledgement  

This work is supported by Key-Area Research and Development Program of Guangdong Province (No. 2021B0101400002), Peng Cheng Laboratory Key Research Project (No. PCL2021A07), Multisource Cross-platform Video Analysis and Understanding for Intelligent Perception in Smart City (No. U20B2052), National Natural Science Foundation of China (No. 61872256, 62102205).  

# Publish Information  

Name of the Journal: Machine Intelligence Research   
Links: https://link.springer.com/article/10. 1007/s11633-022-1410-8   
DOI: 10.1007/s11633-022-1410-8  

# References  

[1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classifcation with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.  

[2] Jia Deng, Wei Dong, Richard Socher, LiJia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.  

[3] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for largescale image recognition. arXiv preprint arXiv:1409.1556, 2014.  

[4] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.  

[5] Christian Szegedy, Sergey Iofe, Vincent Vanhoucke, and Alexander A Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning. In Thirty-frst AAAI conference on artifcial intelligence, 2017.  

[6] Sepp Hochreiter and Jirgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.  

[7] Jefrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 1532–1543, 2014.  

[8] Ryan Kiros, Yukun Zhu, Russ R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Skipthought vectors. Advances in neural information processing systems, 28, 2015.  

[9] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,  Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.  

[10] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pages 4171–4186, 2019.  

[11] Qiaolin Xia, Haoyang Huang, Nan Duan, Dongdong Zhang, Lei Ji, Zhifang Sui, Edward Cui, Taroon Bharti, and Ming Zhou. Xgpt: Cross-modal generative pretraining for image captioning. In CCF International Conference on Natural Language Processing and Chinese Computing, pages 786–797. Springer, 2021.  

[12] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.   
[13] Colin Rafel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unifed text-to-text transformer. Journal of Machine Learning Research, 21:1–67, 2020.   
[14] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language understanding. Advances in neural information processing systems, 32, 2019.   
[15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020.   
[16] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021.   
[17] Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al. Oscar: Object-semantics aligned pretraining for vision-language tasks. In European Conference on Computer Vision, pages 121–137. Springer, 2020.   
[18] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan,  

Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning. In European conference on computer vision, pages 104–120. Springer, 2020.  

[19] Yangguang Li, Feng Liang, Lichen Zhao, Yufeng Cui, Wanli Ouyang, Jing Shao, Fengwei Yu, and Junjie Yan. Supervision exists everywhere: A data efcient contrastive language-image pre-training paradigm. arXiv preprint arXiv:2110.05208, 2021.  

[20] Zhicheng Huang, Zhaoyang Zeng, Bei Liu, Dongmei Fu, and Jianlong Fu. Pixel-bert: Aligning image pixels with text by deep multi-modal transformers. arXiv preprint arXiv:2004.00849, 2020.  

[21] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and visionlanguage representation learning with noisy text supervision. In International Conference on Machine Learning, pages 4904– 4916. PMLR, 2021.  

[22] Jing Liu, Xinxin Zhu, Fei Liu, Longteng Guo, Zijia Zhao, Mingzhen Sun, Weining Wang, Hanqing Lu, Shiyu Zhou, Jiajun Zhang, et al. Opt: Omni-perception pre-trainer for cross-modal understanding and generation. arXiv preprint arXiv:2107.00249, 2021.  

[23] De Cheng, Jingyu Zhou, Nannan Wang, and Xinbo Gao. Hybrid dynamic contrast and probability distillation for unsupervised person re-id. IEEE Transactions on Image Processing, 31:3334–3346, 2022.  

[24] Feilong Chen, Duzhen Zhang, Minglun Han, Xiuyi Chen, Jing Shi, Shuang Xu, and Bo Xu. Vlp: A survey on visionlanguage pre-training. arXiv preprint arXiv:2202.09061, 2022.  

[25] Yifan Du, Zikang Liu, Junyi Li, and Wayne Xin Zhao. A survey of visionlanguage pre-trained models. arXiv preprint arXiv:2202.10936, 2022.  

[26] Munazza Zaib, Quan Z Sheng, and Wei Emma Zhang. A short survey of pretrained language models for conversational ai-a new age in nlp. In Proceedings of the Australasian Computer Science Week Multiconference, pages 1–4, 2020.   
[27] Hanqing Zhang, Haolin Song, Shaoyu Li, Ming Zhou, and Dawei Song. A survey of controllable text generation using transformer-based pre-trained language models. arXiv preprint arXiv:2201.05337, 2022.   
[28] Jian Yang, Gang Xiao, Yulong Shen, Wei Jiang, Xinyu Hu, Ying Zhang, and Jinghui Peng. A survey of knowledge enhanced pre-trained models. arXiv preprint arXiv:2110.00269, 2021.   
[29] Da Yin, Li Dong, Hao Cheng, Xiaodong Liu, Kai-Wei Chang, Furu Wei, and Jianfeng Gao. A survey of knowledge-intensive nlp with pre-trained language models. arXiv preprint arXiv:2202.08772, 2022.   
[30] Prajjwal Bhargava and Vincent Ng. Commonsense knowledge reasoning and generation with pre-trained language models: A survey. arXiv preprint arXiv:2201.12438, 2022.   
[31] Qi Liu, Matt J Kusner, and Phil Blunsom. A survey on contextual embeddings. arXiv preprint arXiv:2003.07278, 2020.   
[32] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. arXiv preprint arXiv:2107.13586, 2021.   
[33] Benyou Wang, Qianqian Xie, Jiahuan Pei, Prayag Tiwari, Zhao Li, et al. Pretrained language models in biomedical domain: A systematic survey. arXiv preprint arXiv:2110.05006, 2021.   
[34] Xipeng Qiu, Tianxiang Sun, Yige Xu, Yunfan Shao, Ning Dai, and Xuanjing Huang. Pre-trained models for natural language  

processing: A survey. Science China Tech  

nological Sciences, 63(10):1872–1897, 2020.   
[35] Xu Han, Zhengyan Zhang, Ning Ding, Yuxian Gu, Xiao Liu, Yuqi Huo, Jiezhong Qiu, Yuan Yao, Ao Zhang, Liang Zhang, et al. Pre-trained models: Past, present and future. AI Open, 2:225–250, 2021.   
[36] Yifan Du, Zikang Liu, Junyi Li, and Wayne Xin Zhao. A survey of visionlanguage pre-trained models. arXiv preprint arXiv:2202.10936, 2022.   
[37] Ludan Ruan and Qin Jin. Survey: Transformer based video-language pre-training. AI Open, 2022.   
[38] Feng Li, Hao Zhang, Yi-Fan Zhang, Shilong Liu, Jian Guo, Lionel M Ni, PengChuan Zhang, and Lei Zhang. Vision-language intelligence: Tasks, representation learning, and large models. arXiv preprint arXiv:2203.01922, 2022.   
[39] Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, et al. A survey on vision transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.   
[40] Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. Transformers in vision: A survey. ACM Computing Surveys (CSUR), 2021.   
[41] Yang Liu, Yao Zhang, Yixin Wang, Feng Hou, Jin Yuan, Jiang Tian, Yang Zhang, Zhongchao Shi, Jianping Fan, and Zhiqiang He. A survey of visual transformers. arXiv preprint arXiv:2111.06091, 2021.   
[42] Javier Selva, Anders S Johansen, Sergio Escalera, Kamal Nasrollahi, Thomas B Moeslund, and Albert Clap´es. Video transformers: A survey. arXiv preprint arXiv:2201.05991, 2022.   
[43] Shangwei Guo, Chunlong Xie, Jiwei Li, Lingjuan Lyu, and Tianwei Zhang. Threats to pre-trained language models: Survey and taxonomy. arXiv preprint arXiv:2202.06862, 2022.   
[44]  Ismael Garrido-Munoz,  Arturo  MontejoRaez, Fernando Martinez-Santiago, and L Alfonso Urena-Lopez. A survey on bias in deep nlp. Applied Sciences, 11(7):3184, 2021.   
[45] Nicholas Meade, Elinor Poole-Dayan, and Siva Reddy. An empirical survey of the efectiveness of debiasing techniques for pretrained language models. arXiv preprint arXiv:2110.08527, 2021.   
[46] Rohit Kumar Kaliyar. A multi-layer bidirectional transformer encoder for pre-trained word embedding: A survey of bert. In 2020 10th International Conference on Cloud Computing, Data Science & Engineering (Confuence), pages 336–340. IEEE, 2020.   
[47] Jiajia Peng and Kaixu Han. Survey of pre-trained models for natural language processing. In 2021 International Conference on Electronic Communications, Internet of Things and Big Data (ICEIB), pages 277– 280. IEEE, 2021.   
[48] Sha Yuan, Hanyu Zhao, Shuai Zhao, Jiahong Leng, Yangxiao Liang, Xiaozhi Wang, Jifan Yu, Xin Lv, Zhou Shao, Jiaao He, et al. A roadmap for big model. arXiv preprint arXiv:2203.14101, 2022.   
[49] Soyeon Caren Han Siqu Long, Feiqi Cao and Haiqing Yang. Vision-and-language pretrained models: A survey. In IJCAI, 2022.   
[50] Xu Peng, Zhu Xiatian, and A. Clifton David. Multimodal learning with transformers: A survey. arXiv preprint arXiv:2206.06488, 2022.   
[51] Y. Lecun, L. Bottou, Y. Bengio, and P. Hafner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.   
[52] Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.   
[53] Xipeng Qiu, Tianxiang Sun, Yige Xu, Yunfan Shao, Ning Dai, and Xuanjing Huang. Pre-trained models for natural language processing: A survey. Science China Technological Sciences, 63(10):1872–1897, 2020.   
[54] Munazza Zaib, Quan Z Sheng, and Wei Emma Zhang. A short survey of pretrained language models for conversational ai-a new age in nlp. In Proceedings of the Australasian Computer Science Week Multiconference, pages 1–4, 2020.   
[55] Bonan Min, Hayley Ross, Elior Sulem, Amir Pouran Ben Veyseh, Thien Huu Nguyen, Oscar Sainz, Eneko Agirre, Ilana Heinz, and Dan Roth. Recent advances in natural language processing via large pre-trained language models: A survey. arXiv preprint arXiv:2111.01243, 2021.   
[56] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. arXiv preprint arXiv:2107.13586, 2021.   
[57] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. In 7th International Conference on Learning Representations, ICLR 2019, 2019.   
[58] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pretraining. 2018.   
[59] Alec Radford, Jefrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9,  

2019.  

[60] Corby Rosset. Turing-nlg: A 17-billionparameter language model by microsoft. Microsoft Blog, 1(2), 2020.   
[61] Wei Zeng, Xiaozhe Ren, Teng Su, Hui Wang, Yi Liao, Zhiwei Wang, Xin Jiang, ZhenZhang Yang, Kaisheng Wang, Xiaoda Zhang, et al. Pangu-alpha: Large-scale autoregressive pretrained chinese language models with auto-parallel computation. arXiv preprint arXiv:2104.12369, 2021.   
[62] Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen, and Qun Liu. Nezha: Neural contextualized representation for chinese language understanding. arXiv preprint arXiv:1909.00204, 2019.   
[63] Mark Chen, Alec Radford, Rewon Child, Jefrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International Conference on Machine Learning, pages 1691–1703. PMLR, 2020.   
[64] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020.   
[65] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020.   
[66] Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip HS Torr, et al. Rethinking semantic segmentation from a sequence-tosequence perspective with transformers. In Proceedings of the IEEE/CVF conference  

on computer vision and pattern recognition, pages 6881–6890, 2021.  

[67] Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and Wen Gao. Pre-trained image processing transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12299–12310, 2021.  

[68] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable vision learners. arXiv preprint arXiv:2111.06377, 2021.  

[69] Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021.  

[70] Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, and Nenghai Yu. Peco: Perceptual codebook for bert pretraining of vision transformers. arXiv preprint arXiv:2111.12710, 2021.  

[71] Stefen Schneider, Alexei Baevski, Ronan Collobert, and Michael Auli. wav2vec: Unsupervised pre-training for speech recognition. arXiv preprint arXiv:1904.05862, 2019.  

[72] Alexei Baevski, Michael Auli, and Abdelrahman Mohamed. Efectiveness of selfsupervised pre-training for speech recognition. arXiv preprint arXiv:1911.03912, 2019.  

[73] Wei-Ning Hsu, Benjamin Bolte, YaoHung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3451–3460, 2021.  

[74] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in Neural Information Processing Systems, 33:12449– 12460, 2020.  

[75] Yu-An Chung, Yu Zhang, Wei Han, ChungCheng Chiu, James Qin, Ruoming Pang, and Yonghui Wu. W2v-bert: Combining contrastive learning and masked language modeling for self-supervised speech pretraining. arXiv preprint arXiv:2108.06209, 2021.  

[76] Peipei Zhu, Xiao Wang, Lin Zhu, Zhenglong Sun, Weishi Zheng, Yaowei Wang, and Changwen Chen. Prompt-based learning for unpaired image captioning. arXiv preprint arXiv:2205.13125, 2022.  

[77] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748– 8763. PMLR, 2021.  

[78] Yinghui Xing, Qirui Wu, De Cheng, Shizhou Zhang, Guoqiang Liang, and Yanning Zhang. Class-aware visual prompt tuning for vision-language pre-trained model. arXiv preprint arXiv:2208.08340, 2022.  

[79] Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. Advances in neural information processing systems, 24, 2011.  

[80] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014.  

[81] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.  

[82] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123(1):32–73, 2017.  

[83] Yash Goyal, Tejas Khot, Douglas SummersStay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017.  

[84] Negar Rostamzadeh, Seyedarian Hosseini, Thomas Boquet, Wojciech Stokowiec, Ying Zhang, Christian Jauvin, and Chris Pal. Fashion-gen: The generative fashion dataset and challenge. arXiv preprint arXiv:1806.08317, 2018.  

[85] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alttext dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2556–2565, 2018.  

[86] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.  

[87] Di Qi, Lin Su, Jia Song, Edward Cui, Taroon Bharti, and Arun Sacheti. Imagebert: Crossmodal pre-training with large-scale weaksupervised image-text data. arXiv preprint arXiv:2001.07966, 2020.  

[88] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3558–3568, 2021.  

[89] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and visionlanguage representation learning with noisy text supervision. In International Conference on Machine Learning, pages 4904– 4916. PMLR, 2021.  

[90] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara Berg. Tvqa: Localized, compositional video question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1369–1379, 2018.  

[91] Antoine Miech, Dimitri Zhukov, JeanBaptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2630–2640, 2019.  

[92] Max Bain, Arsha Nagrani, Gul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for endto-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738, 2021.  

[93] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016.  

[94] Christoph Schuhmann, Robert Kaczmarczyk, Aran Komatsuzaki, Aarush Katta, Richard Vencu, Romain Beaumont, Jenia Jitsev, Theo Coombes, and Clayton Mullis. Laion-400m: Open dataset of clip-fltered 400 million image-text pairs. In NeurIPS Workshop Datacentric AI, number FZJ2022-00923. Juilich Supercomputing Center,  

2021.  

[95] Karan Desai, Gaurav Kaul, Zubin Aysola, and Justin Johnson. RedCaps: Web-curated image-text data created by the people, for the people. In NeurIPS Datasets and Benchmarks, 2021.  

[96] Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Minzhe Niu, Hang Xu, Xiaodan Liang, Wei Zhang, Xin Jiang, and Chunjing Xu. Wukong: 100 million large-scale chinese cross-modal pre-training dataset and a foundation framework, 2022.  

[97] Zarana Parekh, Jason Baldridge, Daniel Cer, Austin Waters, and Yinfei Yang. Crisscrossed captions: Extended intramodal and intermodal semantic similarity judgments for ms-coco. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 2855–2870, 2021.  

[98] Xunlin Zhan, Yangxin Wu, Xiao Dong, Yunchao Wei, Minlong Lu, Yichi Zhang, Hang Xu, and Xiaodan Liang. Product1m: Towards weakly supervised instance-level product retrieval via cross-modal pretraining. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11782–11791, 2021.  

[99] Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2443–2449, 2021.  

[100] Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable efectiveness of data in deep learning era. In Proceedings of the IEEE international conference on computer vision, pages 843–852, 2017.  

[101] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan, and  

Jianfeng Gao. Focal self-attention for localglobal interactions in vision transformers. arXiv preprint arXiv:2107.00641, 2021.  

[102] Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens Van Der Maaten. Exploring the limits of weakly supervised pretraining. In Proceedings of the European conference on computer vision (ECCV), pages 181–196, 2018.  

[103] Junyang Lin, Rui Men, An Yang, Chang Zhou, Ming Ding, Yichang Zhang, Peng Wang, Ang Wang, Le Jiang, Xianyan Jia, et al. M6: A chinese multimodal pretrainer. arXiv preprint arXiv:2103.00823, 2021.  

[104] Xiao Dong, Xunlin Zhan, Yangxin Wu, Yunchao Wei, Xiaoyong Wei, Minlong Lu, and Xiaodan Liang. M5product: A multi-modal pretraining benchmark for e-commercial product downstream tasks. arXiv preprint arXiv:2109.04275, 2021.  

[105] Jordi Pont-Tuset, Jasper Uijlings, Soravit Changpinyo, Radu Soricut, and Vittorio Ferrari. Connecting vision and language with localized narratives. In European Conference on Computer Vision, pages 647–664. Springer, 2020.  

[106] Yuqi Huo, Manli Zhang, Guangzhen Liu, Haoyu Lu, Yizhao Gao, Guoxing Yang, Jingyuan Wen, Heng Zhang, Baogui Xu, Weihao Zheng, et al. Wenlan: Bridging vision and language by large-scale multi-modal pre-training. arXiv preprint arXiv:2103.06561, 2021.  

[107] Leng Jiahong Xue Zhao Zhao Hanyu Sha Yuan, Zhao Shuai and Tang Jie. Wudaomm: A large-scale multi-modal dataset for pre-training models. arXiv preprint arXiv:2203.11480, 2022.  

[108] Delong Chen, Fan Liu, Xiaoyu Du, Ruizhuo Gao, and Feng Xu. Mep-3m: A large-scale multi-modal e-commerce products dataset.  

[109] Nanyi Fei, Zhiwu Lu, Yizhao Gao, Guoxing Yang, Yuqi Huo, Jingyuan Wen, Haoyu Lu,  

Ruihua Song, Xin Gao, Tao Xiang, et al. Wenlan 2.0: Make ai imagine via a multimodal foundation model. arXiv preprint arXiv:2110.14378, 2021.  

[110] Micah Hodosh, Peter Young, and Julia Hockenmaier. Framing image description as a ranking task: Data, models and evaluation metrics. Journal of Artifcial Intelligence Research, 47:853–899, 2013.  

[111] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.  

[112] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641, 2017.  

[113] Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao. Vinvl: Revisiting visual representations in visionlanguage models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5579–5588, 2021.  

[114] Gen Li, Nan Duan, Yuejian Fang, Ming Gong, and Daxin Jiang. Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training. In Proceedings of the AAAI Conference on Artifcial Intelligence, volume 34, pages 11336–11344, 2020.  

[115] Junyang Lin, An Yang, Yichang Zhang, Jie Liu, Jingren Zhou, and Hongxia Yang. Interbert: Vision-and-language interaction for multi-modal pretraining. arXiv preprint arXiv:2003.13198, 2020.  

[116] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. arXiv preprint arXiv:2108.10904, 2021.  

[117] Hao Tan and Mohit Bansal. Lxmert: Learning cross-modality encoder representations from transformers. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLPIJCNLP), pages 5100–5111, 2019.  

[118] Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 2227–2237, New Orleans, Louisiana, June 2018. Association for Computational Linguistics.  

[119] Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, and Hsiao-Wuen Hon. Unifed language model pre-training for natural language understanding and generation. Advances in Neural Information Processing Systems, 32, 2019.  

[120] Gabriel Peyre´, Marco Cuturi, et al. Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning, 11(5-6):355– 607, 2019.  

[121] Yujia Xie, Xiangfeng Wang, Ruijia Wang, and Hongyuan Zha. A fast proximal point method for computing exact wasserstein distance. In Uncertainty in artifcial intelligence, pages 433–453. PMLR, 2020.  

[122] Weituo Hao, Chunyuan Li, Xiujun Li, Lawrence Carin, and Jianfeng Gao. Towards learning a generic agent for vision-andlanguage navigation via pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13137–13146, 2020.  

[123] Fei Yu, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. Ernie-vil: Knowledge enhanced visionlanguage representations through scene graphs. In Proceedings of the AAAI Conference on Artifcial Intelligence, volume 35, pages 3208–3216, 2021.  

[124] Mingchen Zhuge, Dehong Gao, Deng-Ping Fan, Linbo Jin, Ben Chen, Haoming Zhou, Minghui Qiu, and Ling Shao. Kaleido-bert: Vision-language pre-training on fashion domain. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12647–12657, 2021.  

[125] Haiyang Xu, Ming Yan, Chenliang Li, Bin Bi, Songfang Huang, Wenming Xiao, and Fei Huang. E2e-vlp: End-to-end vision-language pre-training enhanced by visual learning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 503–513, 2021.  

[126] Linjie Li, Yen-Chun Chen, Yu Cheng, Zhe Gan, Licheng Yu, and Jingjing Liu. Hero: Hierarchical encoder for video+ language omni-representation pre-training. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2046–2065, 2020.  

[127] Yan Ling, Rui Xia, et al. Vision-language pre-training for multimodal aspectbased sentiment analysis. arXiv preprint arXiv:2204.07955, 2022.  

[128] Yuhao Cui, Zhou Yu, Chunqi Wang, Zhongzhou Zhao, Ji Zhang, Meng Wang, and Jun Yu. Rosita: Enhancing visionand-language semantic alignments via crossand intra-modal knowledge integration. In Proceedings of the 29th ACM International Conference on Multimedia, pages 797–806, 2021.  

[129] Meng-Hao Guo, Tian-Xing Xu, Jiang-Jiang Liu, Zheng-Ning Liu, Peng-Tao Jiang, TaiJiang Mu, Song-Hai Zhang, Ralph R Martin, Ming-Ming Cheng, and Shi-Min Hu. Attention mechanisms in computer vision: A survey. Computational Visual Media, pages 1–38, 2022.  

[130] Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in Neural Information Processing Systems, 34, 2021.  

[131] Ziyi Yang, Yuwei Fang, Chenguang Zhu, Reid Pryzant, Dongdong Chen, Yu Shi, Yichong Xu, Yao Qian, Mei Gao, Yi-Ling Chen, et al. i-code: An integrative and composable multimodal learning framework. arXiv preprint arXiv:2205.01818, 2022.  

[132] Wei Suo, Mengyang Sun, Peng Wang, and Qi Wu. Proposal-free one-stage referring expression via grid-word cross-attention. In Zhi-Hua Zhou, editor, Proceedings of the Thirtieth International Joint Conference on Artifcial Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27 August 2021, pages 1032–1038. ijcai.org, 2021.  

[133] Linchao Zhu and Yi Yang. Actbert: Learning global-local video-text representations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8746–8755, 2020.  

[134] Mengmeng Wang, Jiazheng Xing, and Yong Liu. Actionclip: A new paradigm for video action recognition. arXiv preprint arXiv:2109.08472, 2021.  

[135] Manling Li, Ruochen Xu, Shuohang Wang, Luowei Zhou, Xudong Lin, Chenguang Zhu, Michael Zeng, Heng Ji, and Shih-Fu Chang. Clip-event: Connecting text and images with event structures. arXiv preprint arXiv:2201.05078, 2022.  

[136] Yufeng Cui, Lichen Zhao, Feng Liang, Yangguang Li, and Jing Shao. Democratizing contrastive language-image pre-training: A clip benchmark of data, model, and supervision. arXiv preprint arXiv:2203.05796, 2022.  

[137] Sheng Shen, Liunian Harold Li, Hao Tan, Mohit Bansal, Anna Rohrbach, Kai-Wei Chang, Zhewei Yao, and Kurt Keutzer. How much can clip beneft vision-and-language tasks? arXiv preprint arXiv:2107.06383, 2021.  

[138] Chen Delong, Wu Zhao, Liu Fan, Yang Zaiquan, Huang Yixiang, Bao Yiping, and Zhou Erjin. Prototypical contrastive language image pretraining. arXiv preprint arXiv:2206.10996, 2022.  

[139] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019.  

[140] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining taskagnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32, 2019.  

[141] Chris Alberti, Jefrey Ling, Michael Collins, and David Reitter. Fusion of detected objects in text for visual question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2131–2140, 2019.  

[142] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visual-linguistic representations. In International Conference on Learning Representations, 2019.  

[143] Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason Corso, and Jianfeng Gao. Unifed vision-language pre-training for image captioning and vqa. In Proceedings of the AAAI Conference on Artifcial Intelligence, volume 34, pages 13041–13049, 2020.  

[144] Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. 12-in-1: Multi-task vision and language representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10437–10446, 2020.  

[145] Vishvak Murahari, Dhruv Batra, Devi Parikh, and Abhishek Das. Large-scale pretraining for visual dialog: A simple state-ofthe-art baseline. In European Conference on Computer Vision, pages 336–352. Springer, 2020.  

[146] Gao Yuting, Liu Jinfeng, Xu Zihan, Zhang Jun, Li Ke, and Shen Chunhua. Pyramidclip: Hierarchical feature alignment for vision-language model pretraining. In arXiv:2204.14095, 2022.  

[147] Dehong Gao, Linbo Jin, Ben Chen, Minghui Qiu, Peng Li, Yi Wei, Yi Hu, and Hao Wang. Fashionbert: Text and image matching with adaptive loss for cross-modal retrieval. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2251–2260, 2020.  

[148] Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng, and Jingjing Liu. Largescale adversarial training for vision-andlanguage representation learning. Advances in Neural Information Processing Systems, 33:6616–6628, 2020.  

[149] Dandan Song, Siyi Ma, Zhanchen Sun, Sicheng Yang, and Lejian Liao. Kvl-bert: Knowledge enhanced visual-and-linguistic bert for visual commonsense reasoning. Knowledge-Based Systems, 230:107408, 2021.  

[150] Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. Unifying vision-and-language tasks via text generation. In International Conference on Machine Learning, pages 1931– 1942. PMLR, 2021.  

[151] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-and-language transformer without convolution or region supervision. In International Conference on Machine Learning, pages 5583–5594. PMLR, 2021.  

[152] Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. Mdetr-modulated detection for end-to-end multi-modal understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1780–1790, 2021.  

[153] Zhicheng Huang, Zhaoyang Zeng, Yupan Huang, Bei Liu, Dongmei Fu, and Jianlong Fu. Seeing out of the box: End-to-end pretraining for vision-language representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12976–12985, 2021.  

[154] Hongwei Xue, Yupan Huang, Bei Liu, Houwen Peng, Jianlong Fu, Houqiang Li, and Jiebo Luo. Probing inter-modality: Visual parsing with self-attention for visionand-language pre-training. Advances in Neural Information Processing Systems, 34, 2021.  

[155] Aashi Jain, Mandy Guo, Krishna Srinivasan, Ting Chen, Sneha Kudugunta, Chao Jia, Yinfei Yang, and Jason Baldridge. Mural: multimodal, multitask retrieval across languages. arXiv preprint arXiv:2109.05125, 2021.  

[156] Wenhui Wang, Hangbo Bao, Li Dong, and Furu Wei. Vlmo: Unifed vision-language pre-training with mixture-of-modalityexperts. arXiv preprint arXiv:2111.02358, 2021.  

[157] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Zicheng Liu, Michael Zeng, et al. An empirical study of training end-to-end vision-and-language transformers. arXiv preprint arXiv:2111.02387, 2021.  

[158] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid.  

Videobert: A joint model for video and language representation learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7464– 7473, 2019.  

[159] Chen Sun, Fabien Baradel, Kevin Murphy, and Cordelia Schmid. Learning video representations using contrastive bidirectional transformer. arXiv preprint arXiv:1906.05743, 2019.  

[160] Huaishao Luo, Lei Ji, Botian Shi, Haoyang Huang, Nan Duan, Tianrui Li, Jason Li, Taroon Bharti, and Ming Zhou. Univl: A unifed video and language pre-training model for multimodal understanding and generation. arXiv preprint arXiv:2002.06353, 2020.  

[161] Aisha Urooj, Amir Mazaheri, Mubarak Shah, et al. Mmft-bert: Multimodal fusion transformer with bert encodings for visual question answering. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4648–4660, 2020.  

[162] Rui Yan, Mike Zheng Shou, Yixiao Ge, Alex Jinpeng Wang, Xudong Lin, Guanyu Cai, and Jinhui Tang. Video-text pretraining with learned regions. arXiv preprint arXiv:2112.01194, 2021.  

[163] Wei Li, Can Gao, Guocheng Niu, Xinyan Xiao, Hao Liu, Jiachen Liu, Hua Wu, and Haifeng Wang. Unimo: Towards unifedmodal understanding and generation via cross-modal contrastive learning. arXiv preprint arXiv:2012.15409, 2020.  

[164] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zeroshot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021.  

[165] Alex Hauptmann Yonatan Bisk Jianfeng Gao Liangke Gui, Qiuyuan Huang. Training vision-language transformers from captions alone. arXiv preprint arXiv:2205.09256, 2022.  

[166] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering textto-image generation via transformers. Advances in Neural Information Processing Systems, 34, 2021.  

[167] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong. Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text. Advances in Neural Information Processing Systems, 34, 2021.  

[168] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432, 2021.  

[169] Mickael Coustaty Marcal Rusinol Oriol Ramos Terrades Souhail Bakkali, Zuheng Ming. Hivlp: Hierarchical visionlanguage pre-training for fast image-text retrieval. arXiv preprint arXiv:2205.12029, 2022.  

[170] Longhui Wei, Lingxi Xie, Wengang Zhou, Houqiang Li, and Qi Tian. Mvp: Multimodality-guided visual pre-training. arXiv preprint arXiv:2203.05175, 2022.  

[171] Weixiang Hong, Kaixiang Ji, Jiajia Liu, Jian Wang, Jingdong Chen, and Wei Chu. Gilbert: Generative vision-language pretraining for image-text retrieval. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1379–1388, 2021.  

[172] Yuqi Huo Yizhao Gao Zhiwu Lu JiRong Wen Haoyu Lu, Nanyi Fei. Cots: Collaborative two-stream vision-language pretraining model for cross-modal retrieval. In arXiv:2204.07441, 2022.  

[173] Liunian Harold Li, Haoxuan You, Zhecan Wang, Alireza Zareian, Shih-Fu Chang, and Kai-Wei Chang. Unsupervised visionand-language pre-training without parallel images and captions. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5339–5350, 2021.  

[174] Jean-Baptiste Alayrac, Jef Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. arXiv preprint arXiv:2204.14198, 2022.  

[175] Minheng Ni, Haoyang Huang, Lin Su, Edward Cui, Taroon Bharti, Lijuan Wang, Dongdong Zhang, and Nan Duan. M3p: Learning universal representations via multitask multilingual multimodal pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3977–3986, 2021.  

[176] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unifed vision-language understanding and generation, 2022.  

[177] Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, and Nan Duan. N\” uwa: Visual synthesis pre-training for neural visual world creation. arXiv preprint arXiv:2111.12417, 2021.  

[178] Jinyu Yang, Jiali Duan, Son Tran, Yi Xu, Sampath Chanda, Liqun Chen, Belinda Zeng, Trishul Chilimbi, and Junzhou Huang. Vision-language pre-training with triple contrastive learning. arXiv preprint arXiv:2202.10401, 2022.  

[179] Minlong Lu Wei, Yaowei Wang, and Xiaodan Liang. M5product: Self-harmonized contrastive learning for e-commercial multimodal pretraining.  

[180] Bin Yan and Mingtao Pei. Clinical-bert: Vision-language pre-training for radiograph diagnosis and reports generation. 2022.  

[181] Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, et al. Regionclip: Regionbased language-image pretraining. arXiv preprint arXiv:2112.09106, 2021.   
[182] Xiwen Liang, Fengda Zhu, Lingling Li, Hang Xu, and Xiaodan Liang. Visuallanguage navigation pretraining via promptbased environmental self-exploration. arXiv preprint arXiv:2203.04006, 2022.   
[183] Liunian Harold Li\*, Pengchuan Zhang $^*$ , Haotian Zhang $^*$ , Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded languageimage pre-training. In CVPR, 2022.   
[184] Xie Chunyu, Cai Heng, Song Jianfei, Li Jincheng, Kong Fanjing, Wu Xiaoyu, Morimitsu Henrique, Yao Lin, Wang Dexin, Leng Dawei, Ji Xiangyang, and Deng Yafeng. Zero and r2d2: A large-scale chinese cross-modal benchmark and a vision-language framework. arXiv preprint arXiv:2205.03860, 2022.   
[185] Norman Mu, Alexander Kirillov, David Wagner, and Saining Xie. Slip: Self-supervision meets language-image pretraining. arXiv preprint arXiv:2112.12750, 2021.   
[186] Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. arXiv preprint arXiv:2111.07783, 2021.   
[187] Chenliang Li, Ming Yan, Haiyang Xu, Fuli Luo, Wei Wang, Bin Bi, and Songfang Huang. Semvlp: Vision-language pretraining by aligning semantics at multiple levels. arXiv preprint arXiv:2103.07829, 2021.   
[188] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners  

are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022.  

[189] Jiaxin Shi Duzhen Zhang Jianlong Chang Feilong Chen, Xiuyi Chen and Qi Tian. Hivlp: Hierarchical vision-language pretraining for fast image-text retrieval. arXiv preprint arXiv:2205.12105, 2022.   
[190] Andrey Guzhov, Federico Raue, Jorn Hees, and Andreas Dengel. Audioclip: Extending clip to image, text and audio. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 976–980. IEEE, 2022.   
[191] Hangbo Bao, Wenhui Wang, Li Dong, and Furu Wei. Vl-beit: Generative visionlanguage pretraining. arXiv preprint arXiv:2206.01127, 2022.   
[192] Paul Hongsuck Seo, Arsha Nagrani, Anurag Arnab, and Cordelia Schmid. Endto-end generative pretraining for multimodal video captioning. arXiv preprint arXiv:2201.08264, 2022.   
[193] Fan Zhihao, Wei Zhongyu, Chen Jingjing, Wang Siyuan, Li Zejun, Xu Jiarong, and Huang Xuanjing. A unifed continuous learning framework for multi-modal knowledge discovery and pre-training. arXiv preprint arXiv:2206.05555, 2022.   
[194] Zhang Haotian, Zhang Pengchuan, Hu Xiaowei, Chen Yen-Chun, Harold Li Liunian, Dai Xiyang, Wang Lijuan, Yuan Lu, Hwang Jenq-Neng, and Gao Jianfeng. Glipv2: Unifying localization and visionlanguage understanding. arXiv preprint arXiv:2206.05836, 2022.   
[195] Mustafa Basil, Riquelme Carlos, Puigcerver Joan, Jenatton Rodolphe, and Houlsby Neil. Multimodal contrastive learning with limoe: the language-image mixture of experts. arXiv preprint arXiv:2206.02770, 2022.   
[196] Wang Teng, Jiang Wenhao, Lu Zhichao, Zheng Feng, Cheng Ran, Yin Chengguo,  

and Ping Luo. Vlmixer: Unpaired visionlanguage pre-training via cross-modal cutmix. arXiv preprint arXiv:2206.08919, 2022.  

[197] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In C.J. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc., 2013.  

[198] Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen. Knowledge graph embedding by translating on hyperplanes. Proceedings of the AAAI Conference on Artifcial Intelligence, 28(1), Jun. 2014.  

[199] Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, and Jun Zhao. Knowledge graph embedding via dynamic mapping matrix. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 687–696, Beijing, China, July 2015. Association for Computational Linguistics.  

[200] Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. Learning entity and relation embeddings for knowledge graph completion. In Proceedings of the Twenty-Ninth AAAI Conference on Artifcial Intelligence, AAAI’15, page 2181–2187. AAAI Press, 2015.  

[201] Guoliang Ji, Kang Liu, Shizhu He, and Jun Zhao. Knowledge graph completion with adaptive sparse transfer matrix. Proceedings of the AAAI Conference on Artifcial Intelligence, 30(1), Feb. 2016.  

[202] Maximilian Nickel, Volker Tresp, and HansPeter Kriegel. A three-way model for collective learning on multi-relational data. In Proceedings of the 28th International Conference on International Conference on Machine Learning, ICML’11, page 809–816, Madison, WI, USA, 2011. Omnipress.  

[203] Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng. Reasoning with neural tensor networks for knowledge base completion. In C.J. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc., 2013.  

[204] Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and relations for learning and inference in knowledge bases, 2014.  

[205] Antoine Bordes, Xavier Glorot, Jason Weston, and Yoshua Bengio. A semantic matching energy function for learning with multi-relational data. Machine Learning, 94(2):233–259, Feb 2014.  

[206] Maximilian Nickel, Lorenzo Rosasco, and Tomaso Poggio. Holographic embeddings of knowledge graphs. Proceedings of the AAAI Conference on Artifcial Intelligence, 30(1), Mar. 2016.  

[207] Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann Lecun. Spectral networks and locally connected networks on graphs. In International Conference on Learning Representations (ICLR2014), CBLS, April 2014, 2014.  

[208] Thomas N. Kipf and Max Welling. Semisupervised classifcation with graph convolutional networks. In International Conference on Learning Representations (ICLR), 2017.  

[209] Thomas N. Kipf and Max Welling. Variational graph auto-encoders, 2016.  

[210] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.  

[211] Petar Velicˇkovic´, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and  

Yoshua Bengio. Graph attention networks, 2017.  

[212] Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. Modeling relational data with graph convolutional networks. In Aldo Gangemi, Roberto Navigli, Maria-Esther Vidal, Pascal Hitzler, Raphae¨l Troncy, Laura Hollink, Anna Tordai, and Mehwish Alam, editors, The Semantic Web, pages 593–607, Cham, 2018. Springer International Publishing.  

[213] Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, and Bowen Zhou. End-toend structure-aware convolutional networks for knowledge base completion. Proceedings of the AAAI Conference on Artifcial Intelligence, 33(01):3060–3067, Jul. 2019.  

[214] Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. Convolutional 2d knowledge graph embeddings. Proceedings of the AAAI Conference on Artifcial Intelligence, 32(1), Apr. 2018.  

[215] Deepak Nathani, Jatin Chauhan, Charu Sharma, and Manohar Kaul. Learning attention-based embeddings for relation prediction in knowledge graphs. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2019.  

[216] Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, and Partha Talukdar. Compositionbased multi-relational graph convolutional networks. In International Conference on Learning Representations, 2020.  

[217] Yanzeng Li, Bowen Yu, Xue Mengge, and Tingwen Liu. Enhancing pre-trained Chinese character representation with wordaligned attention. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3442– 3448, Online, July 2020. Association for Computational Linguistics.  

[218] Pei Ke, Haozhe Ji, Siyang Liu, Xiaoyan Zhu, and Minlie Huang. SentiLARE: Sentimentaware language representation learning with linguistic knowledge. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6975–6988, Online, November 2020. Association for Computational Linguistics.  

[219] Adam Roberts, Colin Rafel, and Noam Shazeer. How much knowledge can you pack into the parameters of a language model? In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5418– 5426, Online, November 2020. Association for Computational Linguistics.  

[220] Devendra Sachan, Yuhao Zhang, Peng Qi, and William L. Hamilton. Do syntax trees help pre-trained transformers extract information? In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, Online, April 2021. Association for Computational Linguistics.  

[221] Junru Zhou, Zhuosheng Zhang, Hai Zhao, and Shuailiang Zhang. LIMIT-BERT : Linguistics informed multi-task BERT. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4450–4461, Online, November 2020. Association for Computational Linguistics.  

[222] Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun, and Qun Liu. ERNIE: Enhanced language representation with informative entities. In Proceedings of ACL 2019, 2019.  

[223] Matthew E. Peters, Mark Neumann, Robert Logan, Roy Schwartz, Vidur Joshi, Sameer Singh, and Noah A. Smith. Knowledge enhanced contextual word representations. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 43–54, Hong Kong, China, November 2019. Association for Computational Linguistics.  

[224] Peng Wang, Qi Wu, Chunhua Shen, Anthony Dick, and Anton van den Hengel. Explicit knowledge-based reasoning for visual question answering. In Proceedings of the Twenty-Sixth International Joint Conference on Artifcial Intelligence, IJCAI-17, pages 1290–1296, 2017.  

[225] Peng Wang, Qi Wu, Chunhua Shen, Anthony Dick, and Anton van den Hengel. Fvqa: Fact-based visual question answering. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(10):2413– 2427, 2018.  

[226] Jia Deng, Nan Ding, Yangqing Jia, Andrea Frome, Kevin Murphy, Samy Bengio, Yuan Li, Hartmut Neven, and Hartwig Adam. Large-scale object classifcation using label relation graphs. In David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, Computer Vision – ECCV 2014, pages 48–64, Cham, 2014. Springer International Publishing.  

[227] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfeld, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019.  

[228] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium, October-November 2018. Association for Computational Linguistics.  

[229] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difculty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924– 2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.  

[230] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a large-scale dataset for fact extraction and VERifcation. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809–819, New Orleans, Louisiana, June 2018. Association for Computational Linguistics.  

[231] Zhaochen Guo and Denilson Barbosa. Robust entity linking via random walks. In Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management, CIKM ’14, page 499–508, New York, NY, USA, 2014. Association for Computing Machinery.  

[232] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.  

[233] Chandra Bhagavatula, Ronan Le Bras, Chaitanya Malaviya, Keisuke Sakaguchi, Ari Holtzman, Hannah Rashkin, Doug Downey, Wen tau Yih, and Yejin Choi. Abductive commonsense reasoning. In International Conference on Learning Representations, 2020.  

[234] Bill Yuchen Lin, Wangchunshu Zhou, Ming Shen, Pei Zhou, Chandra Bhagavatula, Yejin Choi, and Xiang Ren. CommonGen: A constrained text generation challenge for generative commonsense reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1823–1840, Online, November 2020. Association for Computational Linguistics.  

[235] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa: Commonsense reasoning about social interactions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4463–4473, Hong Kong, China, November 2019. Association for Computational Linguistics.  

[236] Yonatan Bisk, Rowan Zellers, Ronan Le bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. Proceedings of the AAAI Conference on Artifcial Intelligence, 34(05):7432–7439, Apr. 2020.  

[237] Ben Zhou, Daniel Khashabi, Qiang Ning, and Dan Roth. “going on a vacation” takes longer than “going for a walk”: A study of temporal commonsense understanding. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3363–3369, Hong Kong, China, November 2019. Association for Computational Linguistics.  

[238] Ben Zhou, Kyle Richardson, Qiang Ning, Tushar Khot, Ashish Sabharwal, and Dan Roth. Temporal reasoning on implicit events from distant supervision. In NAACL, 2021.  

[239] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8948–8957, 2019.  

[240] Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, Jose´ MF Moura, Devi Parikh, and Dhruv Batra. Visual dialog. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 326–335, 2017.  

[241] Pengcheng Yang, Boxing Chen, Pei Zhang, and Xu Sun. Visual agreement regularized training for multi-modal machine translation. In Proceedings of the AAAI Conference on Artifcial Intelligence, volume 34, pages 9418–9425, 2020.  

[242] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433, 2015.  

[243] Jingzhou Liu, Wenhu Chen, Yu Cheng, Zhe Gan, Licheng Yu, Yiming Yang, and Jingjing Liu. Violin: A large-scale dataset for video-and-language inference. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10900–10910, 2020.  

[244] Alane Suhr, Mike Lewis, James Yeh, and Yoav Artzi. A corpus of natural language for visual reasoning. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 217–223, 2017.  

[245] Ning Xie, Farley Lai, Derek Doran, and Asim Kadav. Visual entailment: A novel task for fne-grained image understanding. arXiv preprint arXiv:1901.06706, 2019.  

[246] Ido Dagan, Oren Glickman, and Bernardo Magnini. The pascal recognising textual entailment challenge. In Machine Learning Challenges Workshop, pages 177–190. Springer, 2005.  

[247] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6720–6731, 2019.  

[248] Xiao Wang, Shaofei Zheng, Rui Yang, Aihua Zheng, Zhe Chen, Jin Tang, and Bin Luo. Pedestrian attribute recognition: A survey. Pattern Recognition, 121:108220, 2022.  

[249] Deepanway Ghosal, Md Shad Akhtar, Dushyant Chauhan, Soujanya Poria, Asif Ekbal, and Pushpak Bhattacharyya. Contextual inter-modal attention for multimodal sentiment analysis. In proceedings of the 2018 conference on empirical methods in natural language processing, pages 3454–3466, 2018.  

[250] Shuang Li, Tong Xiao, Hongsheng Li, Bolei Zhou, Dayu Yue, and Xiaogang Wang. Person search with natural language description. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1970–1979, 2017.  

[251] Wei Chen, Yang Liu, Weiping Wang, Erwin M Bakker, TK Georgiou, Paul Fieguth, Li Liu, and MSK Lew. Deep image retrieval: A survey. ArXiv, 2021.  

[252] Jing Gu, Eliana Stefani, Qi Wu, Jesse Thomason, and Xin Eric Wang. Visionand-language navigation: A survey of tasks, methods, and future directions. arXiv preprint arXiv:2203.12667, 2022.  

[253] Sang-Min Park and Young-Gab Kim. Visual language navigation: a survey and open challenges. Artifcial Intelligence Review, pages 1–63, 2022.  

[254] Hanwang Zhang, Yulei Niu, and Shih-Fu Chang. Grounding referring expressions in images by variational context. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4158–4166, 2018.  

[255] Sibei Yang, Guanbin Li, and Yizhou Yu. Cross-modal relationship inference for grounding referring expressions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4145–4154, 2019.   
[256] Xinpeng Ding, Nannan Wang, Shiwei Zhang, Ziyuan Huang, Xiaomeng Li, Mingqian Tang, Tongliang Liu, and Xinbo Gao. Exploring language hierarchy for video grounding. IEEE Transactions on Image Processing, 31:4693–4706, 2022.   
[257] Zongheng Tang, Yue Liao, Si Liu, Guanbin Li, Xiaojie Jin, Hongxu Jiang, Qian Yu, and Dong Xu. Human-centric spatio-temporal video grounding with visual transformers. IEEE Transactions on Circuits and Systems for Video Technology, 2021.   
[258] Xiao Wang, Xiujun Shu, Zhipeng Zhang, Bo Jiang, Yaowei Wang, Yonghong Tian, and Feng Wu. Towards more fexible and accurate object tracking with natural language: Algorithms and benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13763–13773, 2021.   
[259] Xiao Wang, Chenglong Li, Rui Yang, Tianzhu Zhang, Jin Tang, and Bin Luo. Describe and attend to track: Learning natural language guided structural representation and visual attention for object tracking. arXiv preprint arXiv:1811.10014, 2018.   
[260] Qi Feng, Vitaly Ablavsky, Qinxun Bai, and Stan Sclarof. Siamese natural language tracker: Tracking by natural language descriptions with siamese trackers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5851–5860, 2021.   
[261] Yuan Yao, Ao Zhang, Zhengyan Zhang, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. Cpt: Colorful prompt tuning for pre-trained vision-language models. arXiv preprint arXiv:2109.11797, 2021.   
[262] Xuehai He, Diji Yang, Weixi Feng, TsuJui Fu, Arjun Akula, Varun Jampani, Pradyumna Narayana, Sugato Basu, William Yang Wang, and Xin Eric Wang. Cpl: Counterfactual prompt learning for vision and language models. arXiv preprint arXiv:2210.10362, 2022.   
[263] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. arXiv preprint arXiv:2203.12119, 2022.   
[264] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337–2348, 2022.   
[265] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16816–16825, 2022.   
[266] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. arXiv preprint arXiv:2107.13586, 2021.   
[267] Qingzheng Wang, Shuai Li, Hong Qin, and Aimin Hao. Robust multi-modal medical image fusion via anisotropic heat difusion guided low-rank structural analysis. Information fusion, 26:103–121, 2015.   
[268] Xiao Wang, Xiujun Shu, Shilliang Zhang, Bo Jiang, Yaowei Wang, Yonghong Tian, and Feng Wu. Mfgnet: Dynamic modalityaware flter generation for rgb-t tracking. IEEE Transactions on Multimedia, 2022.   
[269] Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, and Xiaodong He. Stacked cross attention for image-text matching. In Proceedings of the European Conference on Computer Vision (ECCV), pages 201–216,  