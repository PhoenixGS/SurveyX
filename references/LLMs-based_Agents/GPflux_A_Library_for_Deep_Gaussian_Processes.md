# GPflux: A Library for Deep Gaussian Processes
# Abstract
We introduce GPflux, a Python library for Bayesian deep learning with a strong emphasis on deep Gaussian processes (DGPs). Implementing DGPs is a challenging endeavour due to the various mathematical subtleties that arise when dealing with multivariate Gaussian distributions and the complex bookkeeping of indices. To date, there are no actively maintained, open-sourced and extendable libraries available that support research activities in this area. GPflux aims to fill this gap by providing a library with state-of-the-art DGP algorithms, as well as building blocks for implementing novel Bayesian and GP-based hierarchical models and inference schemes. GPflux is compatible with and built on top of the Keras deep learning eco-system. This enables practitioners to leverage tools from the deep learning community for building and training customised Bayesian models, and create hierarchical models that consist of Bayesian and standard neural network layers in a single coherent framework. GPflux relies on GPflow for most of its GP objects and operations, which makes it an efficient, modular and extensible library, while having a lean codebase.
Keywords: Bayesian deep learning, deep Gaussian processes, TensorFlow and GPflow
©2021 Dutordoir et al. ∗Work done while at Secondmind..
hugh.salimbeni@gmail.com eric.hambro@gmail.com johnangusmcleod@gmail.com felix.leibfried@gmail.com a.artemev20@imperial.ac.uk m.vdwilk@imperial.ac.uk james.hensman@gmail.com m.deisenroth@ucl.ac.uk st@secondmind.ai
st@secondmind.ai
# 1. Introduction
Deep neural networks (DNNs) are flexible parametric function approximators that can be used for supervised and unsupervised learning, especially in applications where there is an abundance of data, such as in computer vision [Krizhevsky et al., 2012], natural language processing [Vaswani et al., 2017], and planning [Schrittwieser et al., 2020]. However, in small-data and noisy settings, DNNs can overfit [Goodfellow et al., 2016], or simply make overconfident predictions, which prevents their use in safety-critical applications [Yuan et al., 2019]. To address this, Bayesian learning algorithms propose to replace the usual point estimates of parameters with probability distributions that quantify the uncertainty that remains due to a lack of data. The resulting Bayesian neural networks (BNNs) can be more robust to overfitting, and make predictions together with a measure of their reliability. Alternatively, instead of representing probability distributions in weight space, Gaussian processes (GPs) can be used to represent uncertainty directly in function space [Rasmussen and Williams, 2006]. Damianou and Lawrence [2013] first used Gaussian processes as layers to create the Deep Gaussian Process (DGP), which enables learning of more powerful representations through compositional features. Recent work has shown that DGPs are particularly promising because function-space Bayesian approximations seem to be of higher quality than those of weight-space BNNs [Foong et al., 2020; Damianou, 2015].
# 2. Motivation
Despite their advantages, DGPs have not been adopted nor studied as widely as (Bayesian) DNNs. A possible explanation for this is the non-existence of well-designed and extendable software libraries that underpin research activities. This is crucial, especially for DGPs, as implementing them is a challenging endeavour, even when relying on toolboxes that support automatic differentiation. This is due to, among other things, the implicit (and infinite) basis functions in GPs, the difficulty of keeping track of indices in the multi-layered, multioutput setting and the numerous numerical implementation subtleties when conditioning Gaussian distributions. To date, there are no actively maintained, open-sourced and extendable DGP libraries available. Some packages (see Table 1) exist, but they are written with a single use-case in mind, and only implement one variation of a model or inference scheme. None of them address the fact that much of the code in DGPs can be factored out and reused for building novel DGP models — these are exactly the abstractions GPflux is providing. Building on top of GPflow, GPflux is also designed to be efficient and modular. That is, the library allows new variants of models and approximations to be implemented without modifying the core GPflux source [Matthews et al., 2017; van der Wilk et al., 2020]. The aim of GPflux is twofold. First, it aims to provide researchers with reusable components to develop new DGP models and inference schemes. Second, it aims to provide practitioners with existing state-of-the-art (deep) GP algorithms and layers, such as latent variables, convolutional and multi-output models. GPflux is built on top of TensorFlow[Abadi et al., 2016; Dillon et al., 2017] and is compatible with Keras [Chollet et al., 2015]. This makes it possible to leverage on a plethora of tools developed by the deep learning community for building, training and deploying deep learning models.
# 3. Deep Gaussian Processes: Brief Overview of Model and Inference
Given a dataset {(xi, yi)}N i=1, a Deep Gaussian process [Damianou and Lawrence, 2013, DGP] is built by composing several GPs, where the output of one layer is fed as input to the next. For each layer fℓ(·), we assume that it is a-priori distributed according to a GP with a kernel kℓ(·, ·). The DGP is then defined as the composition F(·) = fL(. . . f2(f1(·))). We refer to the latent function evaluation of a datapoint xi at the ℓth GP as hi,ℓ= fℓ(hi,ℓ−1) with hi,0 = xi. We further assume a general likelihood yi | F, xi ∼p(yi | F(xi)). Salimbeni and Deisenroth [2017] introduced an elegant and scalable inference scheme for DGPs based on the work of Sparse Variational GPs [Hensman et al., 2015]. It defines L independent approximate posterior GPs q(fℓ(·)) of the form
q(fℓ(·)) = GP
�
kuℓ(·)K−1
uℓuℓmℓ
�
��
�
=:µℓ(·)
;
kℓ(·, ·) + k⊤
uℓ(·)K−1
uℓuℓ(Sℓ−Kuℓuℓ)K−1
uℓuℓkuℓ(·)
�
��
�
=:Σℓ(·)
�
,
where {mℓ, Sℓ}L ℓ=1 parameterises the variational approximate posterior q(uℓ) over inducin variables. Kuℓuℓand kuℓ(·) are covariance matrices computed using the kernel kℓ(·, ·). Th model is trained by optimising a lower bound (ELBO) on the log marginal likelihood
The complexity term in the ELBO can be computed in closed form because all of the distributions are Gaussian. An end-to-end differentiable and unbiased Monte-Carlo estimate of the data-fit term can be computed with samples from q(hi,L), which can be obtained by iteratively propagating datapoints through the layers using the reparametrisation trick: hℓ= µℓ(hℓ−1) + � Σ(hℓ−1) ǫ with ǫ ∼N(0, I). We refer the interested reader to van der Wilk et al. [2020] and Leibfried et al. [2020] for in-depth discussion of this method.
# 4. Key Features and Design
GPflux is designed as a deep learning library where functionality is packed into layers, and layers can be stacked on top of each other to form a hierarchical (i.e. deep) model. Next, we focus on the layers and subsequently show how to create models using these layers. We briefly highlight some useful tooling provided by Keras for training these Bayesian models.
# 4.1 Layers
The key building block in GPflux is the GPlayer, which represents the prior and posterior of a single (multi-output) GP, fℓ(·). It can be seen as the analogue of a standard fullyconnected (dense) layer in a DNN, but with an infinite number of basis functions. It is defined by a Kernel, InducingVariables, and MeanFunction, which are all GPflow objects. Adhering to the Keras design, a layer has to implement a call method which usually maps a Tensor to another Tensor. A GPLayer’s call is slightly different in that it takes the output of the previous layer, say hℓ−1, but returns an object that represents the complete Gaussian distribution of fℓ(hℓ−1) as given by eq. (1). If a subsequent layer
(1)
(2)
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a5cc/a5ccebfa-2982-49e0-890a-acd5a42cb67c.png" style="width: 50%;"></div>
is not able to use the previous layer’s distributional output, a sample will be taken usin the reparametrisation trick. This functionality is provided by TensorFlow Probability DistributionLambda layer [Dillon et al., 2017].
GPflux also provides other Bayesian and GP-based layers. A LatentVariableLayer implements a layer which augments its inputs hi,ℓ−1 with latent variables wi, usually through concatenation. This leads to more flexible DGPs that can model complex, non-Gaussian densities [Dutordoir et al., 2018; Salimbeni et al., 2019]. Convolutional layers can be used for temporal or spatially structured data [van der Wilk et al., 2017; Dutordoir et al., 2020]. GPflux also provides non-GP-specific layers, such as BayesianDenseLayer which implements a dense layer for variational Bayesian neural networks. Moreover, thanks to the compatibility of GPflux with the Keras eco-system, it is possible to naturally combine GPflux layers with standard DNN components, such as convolutional or fully-connected layers, as shown in Listing 1. This variety of different building blocks provided by GPflux in a single unified framework paves the way for systematic evaluation of these methods.
# 4.2 Models and Fitting
As shown in the second line of Listing 1, we can in most cases directly make use of Keras’ Sequential to combine the different GPflux and Keras layers into a hierarchical model. This can be convenient because we limit the number of wrappers around our core layer functionality. However, certain GPflux layers (e.g., LatentVariableLayer) require both features {xi} and the target {yi} in training, which is a functionality that Keras does not provide directly. For these use-cases GPflux provides the specialised DeepGP class. Deep neural networks (DNNs) are trained by minimising the prediction discrepancy for examples in a training set. This is a similar to the data-fit term in the ELBO (eq. (2)), which is passed to the framework using LikelihoodLoss(Gaussian()) in the listing. The KL complexity terms of the ELBO are added to the loss by the GPLayer calls. GPflux enables DGP models to reuse much of the tooling developed by the deep learning community. E.g., during training it can be advantageous to use Keras’ ReduceLROnPlateau to lower the learning rate when the ELBO converges. Other callbacks make it possible to monitor the optimisation trace using TensorBoard or save the optimal weights to disk — many of these features have not been leveraged in (deep) GP libraries before. Finally, adhering to the Keras interface also gives GPflux models a battle-tested interface (e.g., fit, predict, evaluate) which should ease its adoption in downstream applications.
# 5. Final Remarks
GPflux is a toolbox dedicated to Bayesian deep learning and Deep Gaussian processes. GPflux uses the mathematical building blocks from GPflow and combines these with the powerful layered deep learning API provided by Keras. This combination leads to a framework that can be used for: (i) researching new models, and (ii) building, training and deploying Bayesian models in a modern way. A number of steps have been taken to ensure the quality and usability of the project. All GPflux source code is available at http://github.com/secondmind-labs/GPflux/. We use continuous integration and have a test code coverage of over 97%. To learn more or get involved we encourage the reader to have a look at our documentation which contains tutorials and a thorough API reference.
# Acknowledgments
We want to thank Nicolas Durrande, Carl Rasmussen, Dongho Kim, and everybody else a Secondmind who was involved in the open-sourcing effort.
Appendix A. Existing open-source efforts
<div style="text-align: center;">Appendix A. Existing open-source efforts</div>
Package
Implements
Last Commit
Code Tests
SheffieldML/PyDeepGP
Damianou and Lawrence, 2013; Dai et al., 2015
Nov 2018
✗
FelixOpolka/Deep-Gaussian-Process
Salimbeni et al., 2019
Mar 2021
✗
ICL-SML/Doubly-Stochastic-DGP
Salimbeni and Deisenroth, 2017; Salimbeni et al., 2018
Feb 2019
✓
hughsalimbeni/DGPs with IWVI
Salimbeni et al., 2019
May 2019
✓
cambridge-mlg/sghmc dgp
Havasi et al., 2018
Feb 2019
✗
kekeblom/DeepCGP
Blomqvist et al., 2019
Sep 2019
✓
GPyTorch/DeepGP module
Salimbeni and Deisenroth, 2017, adapted to Conj. Gr.
Jul 2020
✓
Table 1: A summary of existing Python Deep GP libraries at the time of writing.
# References
References
M. Abadi et al. (2016). “TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems”. In: arXiv preprint arXiv:1603.04467. Kenneth Blomqvist, Samuel Kaski, and Markus Heinonen (2019). “Deep convolutional Gaussian processes”. In: Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer. Fran¸cois Chollet et al. (2015). Keras. https://keras.io. Zhenwen Dai, Andreas Damianou, Javier Gonz´alez, and Neil D. Lawrence (2015). “Variational auto-encoded deep Gaussian processes”. In: arXiv preprint arXiv:1511.06455. Andreas Damianou (2015). “Deep Gaussian Processes and Variational Propagation of Uncertainty”. PhD thesis. University of Sheffield.
Andreas Damianou and Neil D. Lawrence (2013). “Deep Gaussian processes”. In: Proceedings of the 16th International Conference on Artificial Intelligence and Statistics (AISTATS). Joshua V. Dillon, Ian Langmore, Dustin Tran, Eugene Brevdo, Srinivas Vasudevan, Dave Moore, Brian Patton, Alex Alemi, Matt Hoffman, and Rif A. Saurous (2017). “Tensorflow distributions”. In: arXiv preprint arXiv:1711.10604. Vincent Dutordoir, Hugh Salimbeni, James Hensman, and Marc P. Deisenroth (2018). “Gaussian Process Conditional Density Estimation”. In: Advances in Neural Information Processing Systems 31 (NeurIPS). Vincent Dutordoir, Mark van der Wilk, Artem Artemev, and James Hensman (2020). “Bayesian image classification with deep convolutional Gaussian processes”. In: Proceedings of the 23th International Conference on Artificial Intelligence and Statistics (AISTATS). Andrew Y. K. Foong, David R. Burt, Yingzhen Li, and Richard E. Turner (2020). “On the expressiveness of approximate inference in Bayesian neural networks”. In: Advances in Neural Information Processing Systems 33 (NeurIPS). Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016). Deep Learning. The MIT Press. Marton Havasi, Jos´e Miguel Hern´andez-Lobato, and Juan Jos´e Murillo-Fuentes (2018). “Inference in Deep Gaussian Processes using Stochastic Gradient Hamiltonian Monte Carlo”. In: Advances in Neural Information Processing Systems 31 (NeurIPS). James Hensman, Alexander G. de G. Matthews, and Zoubin Ghahramani (2015). “Scalable variational Gaussian process classification”. In: Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS). Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton (2012). “ImageNet classification with deep convolutional neural networks”. In: Advances in Neural Information Processing Systems 25 (NIPS). Felix Leibfried, Vincent Dutordoir, S. T. John, and Nicolas Durrande (2020). “A tutorial on sparse Gaussian processes and variational inference”. In: arXiv preprint arXiv:2012.13962 Alexander G. de G. Matthews, Mark van der Wilk, Tom Nickson, Keisuke Fujii, Alexis Boukouvalas, Pablo L’eon-Villagr’a, Zoubin Ghahramani, and James Hensman (2017). “GPflow: A Gaussian process library using TensorFlow”. In: Journal of Machine Learning Research. Carl E. Rasmussen and Christopher K. I. Williams (2006). Gaussian Processes for Machine Learning. MIT Press. Hugh Salimbeni and Marc P. Deisenroth (2017). “Doubly stochastic variational inference for deep Gaussian processes”. In: Advances in Neural Information Processing Systems 30 (NIPS).
Hugh Salimbeni, Vincent Dutordoir, James Hensman, and Marc P. Deisenroth (2019). “Deep Gaussian processes with importance-weighted variational inference”. In: Proceedings of the 36th International Conference on Machine Learning (ICML). Hugh Salimbeni, Stefanos Eleftheriadis, and James Hensman (2018). “Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models”. In: Proceedings of the 21th International Conference on Artificial Intelligence and Statistics (AISTATS). Julian Schrittwieser et al. (2020). “Mastering Atari, Go, Chess and Shogi by planning with a learned model”. In: Nature. Mark van der Wilk, Vincent Dutordoir, S. T. John, Artem Artemev, Vincent Adam, and James Hensman (2020). “A framework for interdomain and multioutput Gaussian processes”. In: arXiv preprint arXiv:2003.01115. Mark van der Wilk, Carl E. Rasmussen, and James Hensman (2017). “Convolutional Gaussian processes”. In: Advances in Neural Information Processing Systems 30 (NIPS). Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, �Lukasz Kaiser, and Illia Polosukhin (2017). “Attention is all you need”. In: Advances in Neural Information Processing Systems 30 (NIPS). Xiaoyong Yuan, Pan He, Qile Zhu, and Xiaolin Li (2019). “Adversarial examples: attacks and defenses for deep learning”. In: IEEE transactions on neural networks and learning systems.
