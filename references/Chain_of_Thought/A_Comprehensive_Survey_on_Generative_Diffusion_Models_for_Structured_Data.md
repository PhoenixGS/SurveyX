# A COMPREHENSIVE SURVEY ON GENERATIVE DIFFUSION MODELS FOR STRUCTURED DATA
July 11, 2023
In recent years, generative diffusion models have achieved a rapid paradigm shift in deep generative models by showing groundbreaking performance across various applications. Meanwhile, structured data, encompassing tabular and time series data, has been received comparatively limited attention from the deep learning research community, despite its omnipresence and extensive applications. Thus, there is still a lack of literature and its reviews on structured data modelling via diffusion models, compared to other data modalities such as visual and textual data. To address this gap, we present a comprehensive review of recently proposed diffusion models in the field of structured data. First, this survey provides a concise overview of the score-based diffusion model theory, subsequently proceeding to the technical descriptions of the majority of pioneering works that used structured data in both data-driven general tasks and domain-specific applications. Thereafter, we analyse and discuss the limitations and challenges shown in existing works and suggest potential research directions. We hope this review serves as a catalyst for the research community, promoting developments in generative diffusion models for structured data.
arXiv:2306.04139v2
Keywords Survey · Diffusion Models · Generative Models · Structured Data · Tabular Data · Time Series Dat
# 1 Introduction
Structured data is characterised as its standardised format and ubiquitous across various domains. This type of data can be divided into two categories. The first is tabular data, where information is arranged into rows and columns, representing individual records and their respective attributes. The second category is time series data, which is sequential observations obtained at successive time intervals. The structured data has been extensively applied to many tasks, such as financial modelling [1], fraud detection [2], click-through rate (CTR) prediction [3], clinical event prediction [4], counterfactual estimation [5], and so forth. Enhancing predictive performance and robustness in these applications can provide significant benefits for both end users and organisations offering such solutions. Thus, structured data modelling has been a long-standing research topic in both academia and industry. Over the past decade, deep learning has revolutionised numerous fields, including computer vision and language modelling [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]. This substantial advancement can be largely attributed to data-driven deep learning technologies, which have received considerable focus from the research community [16]. Nevertheless, traditional machine learning techniques are still widely used for structured data and the volume of literature on structured data via deep learning is relatively insufficient. This is primarily due to the challenges that deep learning methodologies face when applied to structured data. First, datasets related to structured data are generally smaller than those for visual or textual data, thus limiting the full exploitation of deep learning’s expressiveness. As a result, traditional machine learning methods are still being broadly utilised [17, 18]. Moreover, dataset complications, such as mixed type of data (both continuous and categorical types), the absence of correlation amongst columns and rows, and the necessity for domain knowledge-guided feature engineering, have made the application of deep learning to structured data a complex task [19]. Kadra et al. [19] thus referred to structured data as the final unconquered castle in deep learning research community. However, structured data modelling via deep generative models including variational auto-encoders (VAE)
To Eun Kim Carnegie Mellon University toeunkim@cmu.edu
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8515/85150bcc-0a95-40f8-8231-b8a92baf8c28.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: A Process of Time Series Imputation via Diffusion Models. The reverse process pθ progressively removes random noise to generate plausible time series data, conditioned on observed values xco 0 . The dashed lines represent observed values, while the solid lines are targets for imputation, denoted as xta 0 , where t corresponds to a specific diffusion step out of the total step T. This illustration is derived from CSDI [39] to provide an intuitive understanding of time series modelling through diffusion models.</div>
[20] and generative adversarial networks (GAN) [21] has been continually explored for applications such as data synthesis against privacy concerns [22, 23], scenario-based simulations [24], and imputations [25]. Additionally, such generative modelling has also been explored to improve predictive performance on structured data. Score-based diffusion models [26, 27, 28] recently have become very prominent in various domains and applications, due to their superior capabilities compared to previous deep generative model families. Diffusion models were initially introduced by Sohl-Dickstein et al. [29], inspired by non-equilibrium statistical physics. Subsequently, they are further developed by [26, 27], verifying their potential in comparison with other state-of-the-art generative models in image synthesis tasks. Hereafter, they have shown exceptional performance across a variety of challenging tasks in different domains, e.g., inverse problems [7, 15], text driven image synthesis and editing [10, 11], language modelling [12, 14], 3D molecule generation [30, 31]. However, albeit its growing body of research, the attention on structured data modelling through diffusion models still remains insufficient. With the aims of promoting future research, we provide a holistic overview of the generative diffusion models for structured data in this paper. The remaining structure of this paper is outlined as follows. As preliminaries, a brief introduction on backgrounds of generative diffusion models is described in Section 2. Then, this survey dichotomises the existing works into two main categories: data-driven general tasks (Section 3) and domain-specific applications (Section 4). Each section provides an overview and describes cutting-edge works along with their key technical novelties. In Section 5, the limitations and challenges shown in existing works are discussed with potential research directions. Finally, we conclude the survey in Section 6. What Sets Our Survey Apart from Others There are several existing surveys on diffusion models, either covering algorithmic developments, various data modalities and applications [32, 33], or focusing on specific data modalities such as vision [34], language [35], and graphs [36], or concentrating on medical imaging [37] in a domain specific manner. Lin et al. [38] is covering structured data, but their work is limited to time series applications. Recognising the gap in the literature review concerning diffusion models for the structured data, this work aims to address this deficiency by presenting the first comprehensive survey dealing with structured data, including both tabular and time series data, and their related applications.
# 2 Backgrounds on Score-based Diffusion Models
Score-based diffusion models are a class of probabilistic generative models that learn to reversal of the data destruction processes that gradually injects noise, to yield high-quality and realistic synthesised data samples. In other words, the training procedure involves two steps: the forward diffusion process and the subsequent backward denoising process. Despite their diverse applications across various data modalities, the design of the forward and backward processes categorises the current research into three main frameworks: denoising diffusion probabilistic models (DDPMs) [27, 29], score-based models (SGMs) [26], and stochastic differential equations (SDEs) [28]. Therefore, this section provides a concise review on three subcategories of the diffusion models with mathematical formulae. We present only the essential derivations, thus we recommend referring to the original papers for comprehensive equations.
DDPMs design the forward and reverse processes via dual Markov chains [27, 29]. The forward process involves the diffusion of data with pre-determined noise, such as Gaussian noise, whilst the reverse process employs deep neural networks to sequentially eliminate noise and recovers the original data.
DDPMs design the forward and reverse processes via dual Markov chains [27, 29]. The forward process involves the diffusion of data with pre-determined noise, such as Gaussian noise, whilst the reverse process employs deep neural networks to sequentially eliminate noise and recovers the original data. Forward Pass Suppose that there is a clean data point x0 drawn from a data distribution q(x0). Then, the forward diffusion process progressively perturbs the clean original data distribution by adding Gaussian noise, ultimately converging towards the standard Gaussian distribution zT . During the diffusion step T, noised latent variables x1, x2, ... , xT are yielded. In other words, it generates xT using sequential transition kernel q(xt|xt−1), which can be formulated as below:
Forward Pass Suppose that there is a clean data point x0 drawn from a data distribution q(x0). Then, the forward diffusion process progressively perturbs the clean original data distribution by adding Gaussian noise, ultimately converging towards the standard Gaussian distribution zT . During the diffusion step T, noised latent variables x1 x2, ... , xT are yielded. In other words, it generates xT using sequential transition kernel q(xt|xt−1), which can be formulated as below:
� thus, the forward process is defined by using a series of transition kernels:
where βT ∈(0, 1) is a variance schedule that controls step sizes, and I is the identity matrix with the same dimension as input data x0. Also, N(x; µ, σ) is the Gaussian distribution of x with the mean µ and covariance σ. Let αt := 1 −βt and ¯αt := �t s=1 αs and ϵ as a Gaussian noise, it becomes feasible to yield a noisy sample from an arbitrary step from the distribution, conditioned on the original input x0 [40] as follows:
Reverse Process With the forward process, the reverse process removes the noise at each step in reverse time direction until the destroyed original data is reconstructed. We start with pθ(xT ) to generate pθ(x0) that obeys the true data distribution q(x0). Again, xT ∼N(0, I). A series of reverse Gaussian transition kernels pθ, where θ denotes the learnable parameters, is parameterised in the form of a deep neural network as:
Additionally, the model parameterises both the mean µθ(xt, t) and variance σθ(xt, t). The model learns to approximate the true data distribution during the reverse process, which is achieved through optimising variational upper bound on negative log-likelihood (NLL):
can be rewritten using Kullback–Leibler divergence (KL divergence) as:
� �� � � �� � � �� � The equation 8 is decomposed into three parts: LT , Lt−1, and L0. First, LT minimises the KL divergence betwe q(xT |x0) and a standard Gaussian distribution p(xT ). Next, in Lt−1, the pθ(xt−1|xt) can be computed agai
(2)
(3)
(4)
(5)
(7)
(8)
osteriors from the forward process. The last term L0 denotes a negative log-likelihood. Especially, we condition Lt−1 n x0 to make it tractable:
Considering the equation 6, 9 and 10, the term Lt−1 in equation 8 can be reformulated as:
Considering the equation 6, 9 and 10, the term Lt−1 in equation 8 can be reformulated as:
where C does not depend on θ, thus considered as a constant. Especially, Ho et al. [27] highlight that instead of parameterising the mean µθ(xt, t), predicting noise vector ϵ at each time step t in the forward process by parameterising ϵθ(xt, t) enhances training efficiency and improves sample quality. They rewrite the equation 11 as below:
where U(1, T) denotes a uniform distribution, λ(t) is a weighting function to adjust the scales of noise evenly, and ϵθ is a deep parameterised model that predicts the Gaussian noise ϵ ∼N(0, I). Once trained, it samples x0 that resembles the original data. Whilst the DDPM can only be applied to continuous data, such as image and audio, its application can be extended beyond these types. Hoogeboom et al. [41] have introduced a multinomial diffusion, specifically designed for categorical data. This model corrupts the data using a discrete Markovian process, then restores the original data through a reversal process. It is particularly crucial in certain data modalities and applications in which the usage of categorical data types is involved: tables, text, segmentation maps, and such [35, 41, 42]. We provide an explanation of multinomial diffusion in the Appendix A.
Whilst the DDPM can only be applied to continuous data, such as image and audio, its application can be extended beyond these types. Hoogeboom et al. [41] have introduced a multinomial diffusion, specifically designed for categorical data. This model corrupts the data using a discrete Markovian process, then restores the original data through a reversal process. It is particularly crucial in certain data modalities and applications in which the usage of categorical data types is involved: tables, text, segmentation maps, and such [35, 41, 42]. We provide an explanation of multinomial diffusion in the Appendix A.
# 2.2 Score-based Generative Models (SGMs)
Here, we employ a score function of a probabilistic density function, denoted as p(x). The score function, ∇x log p(x), is defined as the gradient of the logarithm of the probabilistic density with respect to the input x. In order to estimate the score function, we train a deep neural network sθ(x), where θ represents the learnable parameters and x is the input data, to approximate the score of the original data distribution p(x). It is mathematically formulated as following:
Nevertheless, it is computationally infeasible to obtain ∇x log p(x) in the context of high-dimensional data and deep neural networks. Thus, there are various works to address the issue: score matching [43], denoising score matching [26, 44], and sliced score matching [45]. In particular, noise-conditioned score network (NCSN) (a.k.a score matching with langevin dynamics, SMLD) [26] emphasise issues regarding manifold hypothesis. In circumstances where real-world data concentrate on the low-
Nevertheless, it is computationally infeasible to obtain ∇x log p(x) in the context of high-dimensional data and deep neural networks. Thus, there are various works to address the issue: score matching [43], denoising score matching [26, 44], and sliced score matching [45].
In particular, noise-conditioned score network (NCSN) (a.k.a score matching with langevin dynamics, SMLD) [26] emphasise issues regarding manifold hypothesis. In circumstances where real-world data concentrate on the lowdimensional manifolds that are embedded within a high-dimensional space, the estimated score functions are inevitably imprecise in regions of low density. Thus, they propose to inject the random Gaussian noise to the original data with a sequence of intensifying scale, making the data distribution more amenable to SGMs, and estimate the score corresponding to each noise level. Mathematically, we have a sequence of Gaussian noise scales 0 < σ1 < σ2 < · · · < σt < · · · < σT , thus pσ1(x) ≈p(x0), pσT (x) ≈N(x; 0, I) and pσt(xt|x) ≈N(xt; x, σ2 t I). Moreover, a single noise-conditioned score networks sθ(x, σt) aims to approximate the gradient logarithm density function ∇x log pσt(x). For a specific xt, the derivation of ∇x log pσt(x) is:
(11)
(12)
(13)
In particular, the directions of the gradients are towards regions where the density of samples is high. Then, th combination of the equation 13 and 14 with a weighting function λ (σt) derives a new equation as:
�� �� With the annealed Langevin dynamics, new samples are generated by a progressive denoising process from the prior Gaussian distribution. The annealed Langevin dynamics exploits a Markov chain Monte Carlo (MCMC) to draw a ample from the distribution p(x) using the score function ∇x log p(x). It recursively samples xi as follows:
where ωi ∼N(0, I), γ determines both magnitude and direction of the score update. After executing N iterations, the x becomes a sample derived from the original distribution p(x). Notably, in NCSN [26], the magnitude of ωi undergoes gradual decrement, thereby subtly introducing uncertainty and preventing the model from mode failure.
# 2.3 Stochastic Differential Equations (SDEs)
Since the objective forms of both SGMs [26] and DDPMs [27] are similar, Song et al. [28] have integrated and further generalised these into a single framework where the number of noise scales is extended to infinity via stochastic differential equations (SDEs). The corresponding continuous diffusion process can be described using Itô SDE as:
dx = f(x, t)dt + g(t)dw, t ∈[0, T].
where f is a drift coefficient of x(t) and g is a diffusion coefficient that is interwined with standard Wiener process w. Also, dt is a infinitesimal negative time step. Similar to DDPMs and SGMs, x0 and xT denote data samples from the clean distribution p0 and standard Gaussian distribution pT , respectively. Accordingly, it synthesises new samples from the known prior distribution pT , by solving the reverse-time SDE:
where f is a drift coefficient of x(t) and g is a diffusion coefficient that is interwined with standard Wiener process w Also, dt is a infinitesimal negative time step.
� � where ¯w is the reverse standard Brownian motion. The solution to reverse-time SDE is to approximate a time-dependent deep neural network sθ(x, t) to a score function ∇x log pt(x) via score matching objective function. Instead of directly approximating the score function which is computationally intractable, it estimates the transition probability ∇xtlog pt (xt|x0) that follows the Gaussian distribution during the forward diffusion process as:
� � Here, p0t (xt|x0) denotes the transition kernel of xt given x0 and λ(t) is the weighting function. Upon completion of the training process, we can generate samples employing various techniques such as the Euler-Maruyama (EM), Prediction-Correction (PC), or Probability Flow ODE method. EM solves the equation 18 by using a simple discretisation technique, where dx is substituted with △t and d ¯w is replaced by the Gaussian noise z ∼N(0, ∆tI). In PC method, it operates in a sequential manner, alternating between predictor and corrector. The predictor can employ any numerical solver for the reverse-time SDE following a fixed discretisation strategy, such as the EM method. Subsequently, the corrector can be any score-based Markov chain Monte Carlo (MCMC) method, like annealed Langevin dynamics. Thus, the equation 16 can be solved using Langevin dynamics. In Probability Flow ODE method, the equation 17 can be reformulated into an ODE given by:
This equation maintains the identical marginal probability density pt(t), as that of the SDE. Thus, sampling via solv the above reverse-time ODE is equivalent to solving the time reversal SDE.
(15)
(16)
(17)
(18)
(19)
(20)
# 3 Data-driven General Tasks
In this and subsequent sections, we delve into an in-depth review of diffusion-based methodologies for structured data, which are divided into two main categories: data-driven general tasks and domain-specific applications. For data-driven tasks, we focus on two data modalities: tabular and time series data. When it comes to tabular data modelling, we focus on generation and imputation, whilst for time series data, we delve into generation, imputation, and forecasting. Moving on to the domain-specific applications, we divide them into three categories: electronic health records (EHR), bioelectrical signal processing, and recommendation systems (RecSys). For EHR and bioelectrical signal processing, we explore both generation and specific tasks, with forecasting for EHR and enhancement for bioelectrical signal processing. Table 1 presents a quick summary on these categories along with the corresponding works. For further details, such as information on the generative modelling framework, specific datasets used in the experiments and links to accessible code, please refer to the Appendix B.
<div style="text-align: center;">Table 1: A hierarchical table on generative diffusion models for structured data</div>
Categories
Data Type
Task
Papers
Data-driven General Task
Tabular Data
Generation
TabDDPM [42]
SOS [46]
STaSy [47]
CoDi [48]
Imputation
TabCSDI [49]
Time Series Data
Generation
TSGM [50]
Imputation
CSDI [39]
SSSDS4 [51]
DSPD/CSPD [52]
Forecasting
TimeGrad [53]
ScoreGrad [54]
SSSDS4 [51]
D3VAE [55]
DSPD/CSPD [52]
Domain-specific Applications
Electronic Health Records
Generation
EHRDiff [56]
TabDDPM [57]
MedDiff [58]
EHR-DPM [59]
Forecasting
TDSTF [60]
Bioelectrical Signals
Generation
SSSD-ECG [61]
Enhancement
DeScoD-ECG [62]
DS-DDPM [63]
Recommendation Systems
CODIGEM [64]
DiffuRec [65]
DiffRec (Du et al.) [66]
DiffRec (Wang et al.) [67]
CDDRec [68]
# 3.1 Tabular Data
Tabular data, an organised data format in a rectangular grid of rows and columns, is one of the most universal data types in real-world applications. Nevertheless, handling tabular datasets directly using deep learning has been obstructed by various challenges, such as potential privacy issues and missing information due to data storage or human errors [47, 49]. Thus, deep generative models (including VAE [20] and GAN [21]) have been investigated for both tabular data synthesis [22, 69] or imputation [25, 70]. To further improve the performance, researchers have started to use diffusion models, a new paradigm in generative models, for the generation and imputation of tabular data.
# 3.1.1 Generation
TabDDPM [42] is the first pioneering work in the field of tabular data synthesis. To tackle mixed-type characteristics of tabular data, it employs the Gaussian diffusion to model continuous features and multinomial diffusion to model categorical features [41]. To pre-process the mixed-type of data, they convert the continuous features using min-max scaler whilst they convert categorical features through one-hot encoding and process each feature using a separate
diffusion process. During post-processing, they apply reverse scaling when generating continuous variables. For categorical features, they use softmax function, followed by a rounding operator. These pre- and post-processing techniques are widely used by the most of the follow-up works. TabDDPM [42] uses class-conditional model for classification datasets and inserts target values as an additional feature for regression datasets. A simple MLP architecture is optimised through a combination of mean squared error (MSE) and KL divergence, with each term tailored to continuous and categorical features. The equation is:
where C is the number of categorical features, Lmse t is inherently equation 12, and Li t minimises the KL divergence for each multinomial diffusion [41]. It outperforms other strong baselines in tabular data synthesis and even verified its efficacy on privacy criteria against SMOTE [71]. Next, to address the long-standing problem of class imbalance in tabular data, Kim et al. [46] propose the first work on Score-based Over Sampling (SOS). It utilises style transfer to transform samples from the majority classes to the minority classes. In detail, the samples from the majority classes are corrupted by a forward SDE and recovered using the reverse SDE that originally learns to sample minority classes. Class-conditional fine-tuning is optionally applied to further improve the over-sampling performance. STaSy [47] is another seminal work on tabular data synthesis. Kim et al. [47] highlight that directly applying SDE [28] to the tabular data makes it challenging to learn the joint probability of multiple columns, particularly when there is little to no correlation amongst them. To mitigate the issue, they design architectures comprised of MLP residual blocks, which are dataset-dependent. They further integrate self-paced learning with denoising score matching objective to improve the performance. Fake tabular samples are generated by solving reverse SDEs using probability flow ODE method. It is also shown that sampling quality improves after fine-tuning in general. CoDi [48] addresses the training challenges that arise from mixed-data types by adopting a dual diffusion model approach; one model is for modelling continuous features, while the other is for discrete (categorical) features. These models are comprised of UNet-based architecture where convolutional layers are replaced with linear layers [72] and trained in a co-evolutionary fashion, being conditioned reciprocally. In other words, they receive each other’s output as additional input. To elaborate, there is a pair of data, (xC 0 , xD 0 ), and the perturbed data xC t after t steps is a condition to xD t in the discrete diffusion model, and vice versa. Then, the diffusion objective function for model handling continuous features is updated as follows:
��� ��� For brevity, we only provide equations regarding the continuous features here. To further improve, triplet loss [73] is applied to each diffusion model independently. By learning a combination of two losses, it surpasses the other strong baselines in the real-world benchmark datasets.
# 3.1.2 Imputation
TabCSDI [49] is a novel approach on tabular data imputation built upon CSDI [39], which is for time-series data imputation and forecasting. To generate both continuous and categorical features of tabular data, it explores three conventional techniques: one-hot encoding, analog bits encoding [74], and feature tokenisation [75]. Especially, when utilising feature tokenisation, both continuous and categorical features undergo transformation via an embedding layer. Zheng et al. [49] emphasise the importance of feature tokenization as it mitigates the issue of column imbalance, which is commonly observed in the other two techniques. This approach leads to superior performance.
# 3.2 Time Series Data
Time series data consist of a sequence of observations recorded over regular intervals of time. It plays a crucial role in myriad fields, including finance, healthcare and climate. Its significance is particularly highlighted in decision-making processes, making time series forecasting methodologies evolve significantly from traditional statistical methods [76], to Recurrent Neural Networks (RNNs) [77], and Transformer-based models [78]. However, the performance of predictive tasks using these models can degrade due to the presence of missing data, often attributable to device failures or human error [79]. As such, time series imputation strategies have also been explored [79]. Concurrently, to facilitate scenario-based simulation and address privacy concerns, researchers have also focused on time series synthesis [23, 24].
(21)
(22)
Aiming to comprehensively address the three core challenges—generation, imputation, and forecasting—researcher have delved into the time series modelling with generative diffusion models in recent years1.
# 3.2.1 Generation
TSGM [50] is the first work on time-series synthesis. It generates fake time series data at each time step, based on the past synthesised observations. The framework follows a two-stage process: pre-training and main training, which are performed by encoder, decoder, and conditional score-matching network. During the pre-training stage, both the encoder and decoder are trained to reconstruct the input time series data with MSE loss. Next, in the main training stage, a conditional score matching network, based on UNet [72] with linear layers, is trained with the pre-trained encoder and decoder to synthesise the time series data. PC sampler [28] is harnessed to solve the time-reverse SDE and to yield the synthesised time series data. Its remarkable performance is demonstrated by surpassing prior baselines built upon the frameworks of VAE and GAN on five real-world datasets.
# 3.2.2 Imputation
Although CSDI [39] is mainly introduced for time-series data imputation, it is also applicable to forecasting. Tashiro et al. propose novel training and sampling strategies, which are inspired by masked language modelling [80]. During the training process, a subset of observed values is employed as conditional data, with the rest of the observed data for imputation targets. On the other hand, during sampling, all observed data points are utilised as conditional information, whereas all missing values are considered as targets for imputation. Let conditional observations xco 0 and imputation targets xta 0 , then the noisy target to be sampled can be written as xta t = √αtxta 0 + (1 −αt)ϵ. Accordingly, the imputation network ϵθ is trained by minimising the following objective function:
� � To effectively harness both temporal and feature-based dependencies of time series, they design a temporal and feature Transformer layer [80], instead of using a convolution layer. It outperforms other strong VAE-based baselines on both healthcare and environmental data. SSSDS4 [51] is a pioneering approach that integrates structured state-space models (SSM) [81] into the framework of diffusion models. Specifically, SSM is proven to be effective in handling long-term dependencies on time-series data. It is formulated as:
# x′(t) = Ax(t) + Bu(t) and y(t) = Cx(t) + Du(t)
where u(t) and y(t) are 1D input and output sequence, respectively, and x(t) is an N-dimensional hidden state. When learning with diffusion model framework, it denoises the segments, either in part or as a whole, making it suitable for both imputation and forecasting applications. First, for the imputation task, a given time series x0 and a binary mask M indicate observed elements, then the concatenated matrix is denoted as x0 c = concat(x0 ⊙M, M). They propose the modification of the objective function that incorporates a binary mask, built upon the DDPM framework [27]:
� �� ��� where ϵθ is the SSM and k is the diffusion step. In forecasting, x0 is replaced with (1 −M) ⊙x0. SSSDS4 employs SSSD and modifies the previous works [39, 82] to form a more direct representation of a diffusion process along the time axis. It has shown its competitiveness compared to most of the prior works on handling long sequence time series.
# 3.2.3 Forecasting
TimeGrad, presented by Rasul et al. [53], is an autoregressive multivariate probabilistic time series forecasting model on the basis of DDPM framework [27]. The time interval for prediction is perturbed using Gaussian noise, which is then gradually removed, conditioned on historical time series data. The historical data is encoded with a conditional distribution approximation using a hidden state, generated by an RNN-based module. At time point t, given the multivariate time series vector x0 t, the recurrent model RNN encodes the time series using covariates ct and the hidden state ht−1 from the previous step t −1 by:
(23)
(24)
(25)
Then, the conditional distribution is approximated as:
They follow a similar derivation introduced in Section 2.1, yielding an appropriate objective function using equation 27 After training, the iterative sampling process is conducted until the desired forecast length is attained.
After training, the iterative sampling process is conducted until the desired forecast length is attained. ScoreGrad [54] is built upon SDE framework [28] for time series foreceasting. It comprises a feature extraction module and conditional score-matching module. The feature extraction module, as well as target distribution, is identical to the previous work of TimeGrad [53] in that it encodes historical data and generates hidden states. Next, the conditional score-matching module has multiple residual blocks, each comprising a bidirectional dilated convolution layer, a gated activation unit, and a 1D convolutional neural network for generating output. PC sampler [28] is employed to forecast the future time datapoints by solving the time-reverse SDE. Li et al. [55] propose D3VAE, where a series of diffusion, denoising, and disentanglement is applied to bidirectional VAE (BVAE) [83] with the aim of improving both performance and interpretability in time series forecasting. First, to improve performance, it harnesses a coupled forward diffusion process for data augmentation on both input and target data. It reduces both epistemic and aleatoric uncertainties, where the former is induced by the model and the latter by the data. Meanwhile, the backward process, which encompasses prediction and further refinement of the disturbed prediction, is facilitated by leveraging the BVAE and multi-scale denoising score matching. It further effectively cleans the generated time series using a single-step gradient denoising jump [84]. Next, interpretability involves discerning the independent factors within the data [85]. It can be accomplished by disentangling the latent variables, which signify trends or seasonality. In this regard, D3VAE [55] minimises the Total Correlation (TC) [85] of the BVAE to disentangle latent variables. Therefore, its optimisation process is guided by a combination of four loss terms: 1) KL divergence between the estimated target distribution and target distribution, 2) denoising score matching, 3) minimisation of TC, and 4) MSE loss between prediction and ground truth. It surpasses previous VAE-based works by a significant margin. Bilovš et al. [52] approach time series diffusion modelling differently from the prior works, assuming that the time series data can be represented as a series of values derived from the underlying continuous function x(·). They propose a novel score-based generative diffusion framework where noise injection and removal processes are performed on the entire continuous function, instead of being applied to individual data points. In other words, the continuous function transitions to the prior stochastic process during the forward process, whilst the reverse process yields the new function samples. This means that this continuous function should be continuous and computationally tractable so as to facilitate both training and sampling. Bilovš et al. [52] fulfill these requirements by designing a Gaussian stochastic process with a covariate Σ as ϵ(·) ∼GP(0, Σ), instead of the conventional noise vector ϵ ∼N(0, I). They first introduce Discrete Stochastic Process Diffusion (DSPD), which is built upon the DDPM formulation [27]. The transition kernels in the forward process and backward process are modified from the original formulations of equations 3 and 6 to equations 28 and 29, respectively, as following:
# q(Xk|X0) = N(√¯αkX0, (1 −¯αk)Σ),
pθ(Xk−1|Xk) = N(µθ(Xk, k), (1 −αk)Σ)
where Xk = (xk(t0), ..., xk(tM−1)) represents the time-indexed sequence of points observed at M distinct timestamp ti, within the interval t ⊂[0, T], at k-th diffusion step. As aforementioned, these points are assumed to be derived from the continuous function x(·). Next, the objective function is modified from equation 12 to 30,
Ek,X0 F ,ϵ � λ(k) ��ϵ −ϵθ �√¯αkX0 F + √ 1 −¯αkϵk, X0 H, k ���2
� �� � ���� where X0 F and X0 H denote the input future and historical data. The DSPD can be applied to both forecasting and imputation tasks. First, the forecasting methodology of DSPD is similar to TimeGrad [53], with respect to objective function, architecture, and the sampling process. However, it advances in two points: 1) it is capable of predicting any
(26)
(27)
(28)
(29)
(30)
future point within a continuous time interval and 2) it facilitates simultaneous prediction of multiple time points in a single run, not in an autoregressive fashion. Second, DSPD assumes that observed time series are formed by the values of the continuous function x(·) at specific time points, which allows missing values to be computed by extracting the value of the continuous function at the corresponding time point. Thus, it can be repurposed for imputation by replacing the variance term with a covariance matrix, which is indicated in the equation on neural network of CSDI [39]. Bilovš et al. [52] extend DSPD to a continuous variant termed as Continuous Stochastic Process Diffusion (CSPD), developed from the SDE formulation [28]. This continuous variant can also be employed for both imputation and forecasting tasks, particularly useful in imputation tasks. This is because the continuous noise process is more natural when handling irregular time intervals, which is a prevalent issue in imputation tasks.
# 4 Domain-specific Applications
# 4.1 Electronic Health Records (EHR)
EHR encompasses a vast amount of patient-centered information, including patient’s medical history, diagnoses, medications and such. It has significantly benefited from the advancements in deep learning applications, such as predictive diagnosis [86], medication recommendation [87], and continuous monitoring in intensive care units (ICUs), which significantly diminished the likelihood of complications and mortality rates [88]. However, due to the privacy and ethical concerns surrounding the EHR, release of public data has been limited, which has constrained further research and development [89]. To mitigate these concerns, high-quality and realistic EHR synthesis using GAN [90] has been investigated amongst researchers. With more powerful generative diffusion models, researchers have explored both synthesis and forecasting of the EHR.
# 4.1.1 Generation
MedDiff [58] is the first literature that applies diffusion models to EHR. He et al. [58] propose novel three points: first, it accelerates the generation process via Anderson acceleration [91]. Next, to reflect label information to the generated samples, it utilises classifier-guided sampling process [92]. Lastly, it uses 1D convolutional layer to U-Net architecture to enhance the learning of feature correlations amongst neighbor features. Nonetheless, it is still limited in that it only generates continuous variables, as the authors confine the data to follow Gaussian distribution. To mitigate the aforementioned issue, Ceritli et al. [57] adopt TabDDPM framework [42] to generate 1) continuous data using the Gaussian multinomial diffusion processes and 2) categorical data using the multinomial diffusion processes. The results demonstrate that their work [57] surpasses previous works in data usability criteria but with the exception of privacy evaluation, raising the necessity to improve in such direction. EHRDiff [56] is built upon the work of SDE [28], especially solving the reverse process via Heun’s second order method ODEs to produce more precise and realistic EHR samples in a deterministic manner. Also, it utilises an adaptive parameterisation [93] to address the issue of amplified prediction error due to the variance of scale noise σ. For the architecture, it uses multilayer perceptron (MLP) layers. Similar to TabDDPM [57], it shows outstanding performance on data usability, but moderate performance on privacy criteria. Kuo et al. [59] propose a EHR-DPM. It introduces two auxiliary loss functions: 1) one-step reconstruction loss to alleviate training instability and 2) MSE loss between the clean data and its restored counterpart in a randomly projected latent space. Unlike MedDiff [58] that applies a 1D convolutional layer to individually denoise each feature, EHR-DPM adds linear layers. It involves two steps: 1) a linear layer is added to perform denoising at the latent level, as opposed to the variable level, 2) an additional linear layer is incorporated into each up-sampling and down-sampling 1D CNN, and the final up-sampling output.
# 4.1.2 Forecasting
TDSTF [60] is an initial work on EHR forecasting via diffusion models. The training and sampling process are the same as other traditional diffusion models for time series forecasting. However, to overcome data sparsity, Chang et al. [60] convert the data into triplet form instead of utilising conventional techniques such as aggregation and imputation. Also, its architecture consists of multiple residual layers including Transformer encoder and decoder [80]. It surpasses previous existing models in terms of both predictive performance on critical vital sign and inference speed by a large margin on MIMIC-III dataset [94].
Bioelectrical signals are measured through electrical potential differences across a cell or an organ [95]. They include Electrocardiogram (ECG), Electroencephalogram (EEG), Electromyogram (EMG) and Electrooculogram (EOG). These signals, expressing electrical activity in the heart, brain, muscles, and eyes, respectively, serve as non-invasive but informative diagnostic tools [96]. Recent advancements in deep learning methodologies have led to notable predictive capability improvements on bioelectrical signals. However, two significant hurdles persist [97]: restricted availability of public bioelectrical signal data due to privacy concerns and inevitable data corruption owing to noise from sources like patient respiration and body movements. These challenges impede further improvements of deep predictive models, thus researchers have explored deep generative models to both synthesise [98] and reduce wander or noise [99] from bioelectrical signal data.
# 4.2.1 Generation
SSSD-ECG [61] is built upon the work of SSSDS4 [51], which integrates diffusion models and structured state space models. It synthesises 12-lead ECG data in a multi-label fashion by conditioning on over 70 ECG statements. It outperforms previous GAN-based works in ECG synthesis in terms of both quantitative evaluation and qualitative analysis from domain experts.
# 4.2.2 Enhancement
Li et al. [62] propose a novel ECG enhancement model, DeScoD-ECG, that is designed for the removal of noise and wander. It operates conditionally on noisy observations and employs an iterative process to restore signals from Gaussian distributions. With a clean input signal ˜x and a noisy ECG signal x0, the equation 5 and 6 from DDPM [27] is modified as:
pθ (xt−1 | xt, ˜x) = N (xt−1; µθ (xt, t | ˜x) , σθ (xt, t | ˜x) I) .
Also, it utilises a self-ensemble strategy, wherein it averages multiple output results to enhance the performance of signal reconstruction. The architecture has two backbones that extract features from both the noisy observations and the latent variables, then they are combined through bridge modules. Inspired by the DeepFilter [99], two backbones conduct multi-scale feature aggregation and channel-wise concatenation processes. Also, the bridge modules are conditioned on noise level and perform encoding using sinusoidal positional embeddings [80], followed by a convolutional layer with a 1 x 1 kernel size. The absence of any up/downsampling procedures within the architecture ensures its applicability to ECG signals of any length. DeScoD-ECG [62] is validated on real-world datasets, including the QT database [100] and the MIT-BIH Noise Stress Test Database [101]. The results demonstrate the efficacy of stable DeScoD-ECG compared to other baselines, particularly in situations with extreme corruption. Duan et al. [63] explore the reconstruction of EEG signal under subject-specific variability. They propose DS-DDPM based on the assumption that noisy EEG signals (input) can be decomposed into domain specific noise and clean signals. Thus, they propose a novel diffusion and the reversal process, in which the domain specific noise and clean signal are divided and aggregated at every time step. As both noise and signal exist in orthogonal spaces, DS-DDPM learns using an Additive Angular Margin classification (Arc-Margin) loss [102] to improve intra-class cohesion and inter-class discrepancy. Additionally, they implement an input overlap segmentation strategy to minimise temporal differences in overlapping segments. It also adopts a classifier guidance strategy [92] that exploits human subject index to improve generalisation capability. It learns with modified UNet architecture [72] with multi-head attention [80]. The effectiveness of DS-DDPM is validated on the BCI Competition IV motor imagery decoding dataset [103]. It is also tested on downstream classification tasks to corroborate its reconstruction capability.
# 4.3 Recommendation Systems (RecSys)
RecSys aim at modelling personalised users’ preferences based on their previous user-item interactions such as clicks, ratings, or purchases [104]. Due to their profound significant value in industry, research on RecSys has progressed from traditional methods such as collaborative filtering (CF) based techniques [105] to deep learning methodologies [106]. However, they often show limited generalisation performance, on account of weak collaborative signals, inadequate latent representations, or noisy data scenarios [66]. Researchers have explored the generative models of VAE [107] and
(31)
(32)
GAN [108] to mitigate these challenges, but these models also have their own limitations of restricted capability in capturing personalised user preferences (from VAE-based models) and training instability (from GAN-based models) [67]. To address the issues, researchers have delved into diffusion models, owing to their strong representation capability and training stability. Walker et al. [64] propose CODIGEM, which is the first work exploiting diffusion models on RecSys. It generates strong collaborative signals and robust latent representations on user-item interactions, thus outperforming previous VAE-based works. However, the model falls short in handling sequential scenarios. DiffuRec [65] aims to address the limitations from modelling the item representations as fixed vectors. To handle latent representation of items and multi-level interests of users, they utilise diffusion models to represent them as distributions. During the diffusion process, a truncated linear schedule is used for noise addition. The introduction of noise functions as an uncertainty factor that steers the learning process towards enhanced robustness. Also, instead of using conventional MSE loss in DDPM [27], it learns with cross-entropy loss. This modification is driven by two reasons: the static nature of item embedding and the use of inner product for relevance calculation in the reverse process. They also exploit Transformer [80] as an approximator to reconstruct target item representation. These modifications enable their model to show remarkable performance on several benchmark datasets. Du et al. [66] develop another novel sequential diffusion model for RecSys. It introduces an additional transition to convert the items with discrete features. It perturbs the original item, instead of the whole item sequences, by injecting noise and it restores the target by MSE loss term. Also, it samples only important diffusion steps instead of the entire steps. Those strategies enable an efficient training and sampling. On the architectural side, it leverages a transformer-based encoder [80] with learnable positional embedding. It significantly surpasses preceding generative and contrastive learning based benchmarks on real world datasets. Wang et al. [67] design two novel diffusion recommendation models for sequential RecSys of L-DiffRec and T-DiffRec, highlighting two major challenges in RecSys: 1) high computational costs for large-scale item prediction and 2) temporal transition for user preference. L-DiffRec addresses the first challenge. It groups items into clusters, then compresses the user-item interaction using multiple VAEs and conducts diffusion processes in the latent space to generate top-K recommendations. It addresses the high computational demands arising from large-scale item prediction. Second, T-DiffRec mitigates the second issue, temporal shifts in user preferences. It introduces time-aware reweighting strategy to model user interactions based on the assumption that the recent interactions of a user may capture user preference more effectively. This enhances the model’s adaptability to dynamic user behaviour. The two variants of DiffRec outperform previous VAE-based baselines by a substantial margin on several public datasets. CDDRec [68] achieves diffusion models for sequential RecSys in conditional autoregressive and ranking aware manners. Its network comprises of three models: a step-wise diffuser, a sequence encoder and a cross-attentive conditional denoising decoder. First, the step-wise diffuser corrupts the original data using increasing Gaussian noise. The sequence encoder utilises a self-attention mechanism on historical interactions whilst the cross-attentive conditional denoising decoder adopts a direct-condition mechanism wherein the sequence embeddings from the encoder are directly conditioned to the decoder at every denoising step. It enables a conditional sample generation in an autoregressive fashion. Further, to mitigate the inherent long-tailed and sparse item distribution, CDDRec [68] is optimised through a combination of a cross-divergence loss and multi-view contrastive loss with denoising diffusion loss. The cross-divergence loss term utilises KL divergence to minimise the discrepancy between the forecasted mean and the target item embedding, as compared to the discrepancy between the predicted mean and unrelated target embeddings. It helps avoid a situation where the rank scores uniformly converge across all items for all users, enabling the model to function in a ranking-aware fashion. The multi-view contrastive loss with denoising diffusion loss applies contrastive loss to enhance the robustness against the noisy interference by maximising the agreement between the original view and its counterpart, achieved through random cropping, shuffling and masking randomly. The combination of two loss terms, therefore, guides the CDDRec to be more discriminative and robust.
# 5 Current Challenges and Future Research Directions
Despite the astonishing advancements in generative diffusion models for the structured data, there are still challenges that this field continues to grapple with. The objective of this section is to describe these challenges and provide potential future research directions. Customised Designs for Structured Data Generative diffusion models are largely influenced by factors such as architectural design, training strategy, and noise scheduling. However, most of the approaches on structured data
have directly adopted or slightly modified from existing seminal works from other data modalities. Customising these elements may potentially lead to significant improvements in modelling capability and overcoming generative modelling trilemma: sampling quality, diversity, and speed [109]. Causal Learning and Counterfactual Reasoning Causal learning aims to discern and identify the casual relationships or inter-dependencies amongst variables within a given dataset [110]. For instance, it allows us to model potential outcomes or the most indicative factor in financial forecasting or disease progression. On the other hand, counterfactual reasoning aims to predict an individual’s outcome under different circumstances [5]. Specifically, it predicts what the result would be like if certain variables were changed from their observed real-world values. Hence, the integration of causal learning or counterfactual reasoning methodologies into the diffusion models can potentially enhance their performance. By harnessing cause-and-effect relationships or counterfactual estimation rather than simply exploiting correlations or conditioning on variables, we can optimise diffusion models to be more reliable and robust. Bias in Dataset Another noteworthy challenge is the inherent bias present within the publicly available datasets. Specifically, the demographic features in the EHRs and the bioelectrical signals extracted from the subjects are often biased towards specific classes [63, 111]. This inevitable skewness in data consequently impacts the generalisability of the generative models and limits their applicability to related fields. To circumvent these issues, it is vital to utilise more diverse and balanced datasets. Otherwise, novel methods should be developed to neutralise this inherent bias. Extensions to Multi-modality Learning The fusion of structured data with other data modalities not only enhances the model performance but also expands the possible tasks, e.g., improving financial stock price prediction by jointly learning both text and time series data [112] and text synthesis from table [113]. In this regard, future research direction may focus on the development of novel methodologies adept at integrating multiple data modalities in an efficient and effective manner as well as exploring the potential untapped tasks. Miscellany In the realm of tabular data modelling, it is possible to devise an enhanced strategy that effectively models both continuous and categorical data types. Furthermore, a universal framework can be developed to encompass generation, imputation, and forecasting for both tabular and time-series modelling. For applications involving EHR and biological signal processing, current works may benefit from an incorporation of domain-specific knowledge or issues, e.g., medical ontology and irregular visit interval [114]. Furthermore, the long inference time compared to GAN-based methods poses a challenge for real-time or on-chip deployment in medical equipment, highlighting the need for improvements to facilitate practical use.
# 6 Conclusion
The generative diffusion models have verified themselves by exhibiting remarkable performance across diverse applications, and also have shown excellent performance on structured data. However, current research on generative diffusion models specifically tailored for structured data has not received active attention, compared to the other data modalities. To facilitate exploration and advancement in this field, we provide a comprehensive survey on generative diffusion models for structured data. Our survey encompasses a brief introduction to the underlying theory of scorebased diffusion models, followed by a concise review of existing literature categorized into data-driven general tasks and domain-specific applications. Additionally, we discuss current challenges and outline potential research directions for the future. We hope that this survey will serve as a valuable guide for those interested in this field, thereby fostering further research and advancement in the area.
# References
# Appendices
# A Multinomial Diffusion for Categorical Data
In this section, we introduce multinomial diffusion [41], which is designed to process categorical data. For K categorical data, each variable is encoded as a one-hot vector, denoted as xt ∈{0, 1}K. Using categorical distribution the multinomial diffusion process is formulated utilising the uniform diffusion noise schedule βt and categorica distribution C:
q(xt|xt−1) = C(xt; (1 −βt)xt−1 + βt/K).
Then, we can compute the probability of any xt given x0 with αt := 1 −βt and ¯αt := �t τ=0 ατ: q(xt|x0) = C(xt; ¯αtx0 + (1 −¯αt)/K).
q(xt|x0) = C(xt; ¯αtx0 + (1 −¯αt)/K).
According to Hoogeboom et al. [41], the distribution for the preceding time step t −1 can be expressed from the valu xt at the next step and the ground truth value x0 as following:
q(xt−1|xt, x0) = C(xt−1; ˜θ/A)
where ˜θ = [αtxt + (1 −αt)/K] ⊙[¯αt−1x0 + (1 −¯αt−1)/K] and A is a normalising constant that guarantees the cumulative total of all probabilities equals one. It is noteworthy that q(xt−1|xt, x0) simplifies to q(xt−1|xt), following the Markov property of the forward process. Lastly, the reverse process pθ(xt−1|xt) is also learned via a deep neural network, denoted as q(xt−1|xt, ˆx0(xt, t)) where ˆx0 is the predicted probabilities by the deep neural network. The model is trained by minimising KL divergence (the Lt−1 term in Equation 8) between the true distribution and predicted one as:
DKL(q(xt−1|xt, x0) || pθ(xt−1|xt)) = DKL(C(˜θ(xt, x0)) || C(˜θ(xt, ˆx0)
(33)
(34)
(35)
(36)
<div style="text-align: center;">Table 2: A detailed table on generative diffusion models for structured data, including their frameworks, datasets u for experiments and accessible code links</div>
Paper
Year
Task
Framework
Dataset
Code
TabDDPM [42]
2022
Tabular Generation
DDPM
Abalone, Adult ROC, Buddy, California Housing, Cardio,
Churn Modelling, Diabetes, Facebook Comments Volume,
Gesture Phase, Higgs Small, House 16H, Insurance, King,
MiniBooNE, Wilt
Link
SOS [46]
2022
Tabular Generation
SDE
Buddy, Default, Satimage, Shoppers, Surgical, WeatherAUS
Link
STaSy [47]
2023
Tabular Generation
SDE
Bean, Beijing, Credit, Crowdsource, Contraceptive, Default,
HTRU, Magic, News, Obesity, Phishing, Robot, Shoppers,
Shuttle, Spambase
Link
CoDi [48]
2023
Tabular Generation
DDPM
Absent, Bank, CMC, Customer, Drug, Faults, Heart, Insurance,
Obesity, Seismic, Stroke
Link
TabCSDI [49]
2022
Tabular Imputation
DDPM
Breast, Census, Concrete, COVID-19, Diabetes, Libras, Wine
Link
TSGM [50]
2023
Time Series Generation
SDE
Air, AI4I, Energy, Occupancy, Stocks
N/A
CSDI [39]
2021
Time Series Imputation
Time Series Forecasting
DDPM
Air Quality, PhysioNet 2012 Challenge
Link
SSSDS4 [51]
2022
Time Series Imputation
Time Series Forecasting
DDPM
ETTm1, MuJoCo, PTB-XL, Solar
Link
DSPD/CSPD [52]
2022
Time Series Imputation
Time Series Forecasting
SDE
CIR, Lorenz, OU, Predator-prey, Sine, Sink
N/A
TimeGrad [53]
2021
Time Series Forecasting
DDPM
Electricity, Exchange, Solar, Taxi, Traffic, Wiki
Link
ScoreGrad [54]
2021
Time Series Forecasting
SDE
Electricity, Exchange, Solar, Taxi, Traffic, Wiki
Link
D3VAE [55]
2022
Time Series Forecasting
DDPM
Electricity, ETTm1, ETTh1, Traffic, Weather, Wind
Link
EHRDiff [56]
2023
EHR Generation
SDE
MIMIC-III
Link
TabDDPM [57]
2023
EHR Generation
DDPM
MIMIC-III
N/A
MedDiff [58]
2023
EHR Generation
SDE
MIMIC-III, Patient Treatment Classification
N/A
EHR-DPM [59]
2023
EHR Generation
DDPM
MIMIC-III, EuResist
N/A
TDSTF [60]
2023
EHR Forecasting
DDPM
MIMIC-III
Link
SSSD-ECG [61]
2022
Biosignal Generation
DDPM
PTB-XL
Link
DeScoD-ECG [62]
2022
Biosignal Enhancement
DDPM
MIT-BIH Noise Stress Test Database, QT Database
Link
DS-DDPM [63]
2022
Biosignal Enhancement
DDPM
BCI-Competition-IV dataset
Link
CODIGEM [64]
2022
RecSys
DDPM
Amazon Electronics, MovieLens-1m, MovieLens-20m
Link
DiffuRec [65]
2023
RecSys
DDPM
Beauty, MovieLens-1M, Steam, Toys
N/A
DiffRec (Du et al.) [66]
2023
RecSys
DDPM
Beauty, MovieLens-1M, Toys
N/A
DiffRec (Wang et al.) [67]
2023
RecSys
DDPM
Amazon-book, MovieLens-1M, Yelp
Link
CDDRec [68]
2023
RecSys
DDPM
Beauty, Home, Office, Tools
N/A
