# Where does In-context Translation Happen in Large Language Models?
Suzanna Sia 1 David Mueller 1 Kevin Duh 1
# Abstract
Self-supervised large language models have demonstrated the ability to perform Machine Translation (MT) via in-context learning, but little is known about where the model performs the task with respect to prompt instructions and demonstration examples. In this work, we attempt to characterize the region where large language models transition from in-context learners to translation models. Through a series of layer-wise context-masking experiments on GPTNEO2.7B, BLOOM3B, LLAMA7B and LLAMA7B-CHAT, we demonstrate evidence of a "task recognition" point where the translation task is encoded into the input representations and attention to context is no longer necessary. We further observe correspondence between the low performance when masking out entire layers, and the task recognition layers. Taking advantage of this redundancy results in 45% computational savings when prompting with 5 examples, and task recognition achieved at layer 14 / 32. Our layer-wise fine-tuning experiments indicate that the most effective layers for MT fine-tuning are the layers critical to task recognition.
arXiv:2403.04510v1
In-context learning (ICL) refers to the phenomenon in which large generative pretrained transformers (GPTs) perform tasks with no gradient updates when shown task examples or descriptions in their context (Brown et al., 2020; Bommasani et al., 2021). While in-context learning in GPT models appears to be generally applicable to any natural language task, to study task location, we use Machine Translation (MT) as there is little to no ambiguity in evaluating whether the model has recognised the task, since it must generate tokens in a different language. While in-context MT has yet to reach parity with supervised neural MT models, it’s off-the-shelf translation performance is comparatively
1Johns Hopkins University. Correspondence to: Suzanna Sia <ssia1@jh.edu>.
1Johns Hopkins University. Correspondence to: Suzanna Sia <ssia1@jh.edu>.
strong and suggests a promising direction for the future of MT (Hendy et al., 2023; Garcia et al., 2023). Prior work on in-context MT has focused on prompt-engineering, treating GPT models as black boxes by focusing on which examples to provide in-context (Moslem et al., 2023). Agrawal et al. (2022) apply similarity-based retrieval to select in-context examples, while Sia & Duh (2023) suggest a coherencebased approach. However, these works apply surface level interventions leaving the internal mechanism of MT in GPT models largely not understood. In this work, we ask where does in-context Machine Translation occur in GPT models? We conduct an initial exploration into locating self-attention layers responsible for in-context MT in three base pre-trained and one instructiontuned open-source GPT models . Using causal masking over different parts of the context we demonstrate that there exists a "task-recognition" point after which attention to the context is no longer necessary (Section 3). The potential implications are large computational savings when the context is several times longer than the test source sentence (Section 5). Having identified the layers in which "task recognition" occurs, we study the extent to which subsequent layers are either redundant or corresponding to the "task recognition" layers. Simple layer-wise masking shows that for 3B parameter models, removing attention around the "task-recognition" layers can cause the model to fail to perform translation all-together, whereas layers towards the end of the model are much more redundant (Section 4.1). Next, we observe that very lightweight fine-tuning of LoRA parameters (Hu et al., 2021) are most effective at earlier layers of the model compared to the later ones (Section 6.2). This provides supports for the conjecture that earlier layers are more important for the task. We further investigate the extent of MT task redundancy using differentiable L0 regularisation to train discrete attention head gates (Section 6.5). We find that around 10% of the attention heads can be masked, which fundamentally differs from the literature in supervised NMT where attention heads are highly specialised for MT (Voita et al., 2019b; Michel et al., 2019; Behnke & Heafield, 2021).
# 1. Background
In-Context Learning was first demonstrated by Brown et al. (2020) who showed that GPT-3 could be used to perform a huge variety of tasks without any task-specific parameters or training, by conditioning the model’s generation on a prompt which included a few labeled examples of the task of interest. Since then, interest in using GPT models for ICL has grown significantly, with several recent works introducing methods such as instruction-tuning (Sanh et al., 2022; Wang et al., 2022) or chain-of-thought prompting (Wei et al., 2022) to improve downstream ICL accuracy.
In-Context Learning was first demonstrated by Brown et al. (2020) who showed that GPT-3 could be used to perform a huge variety of tasks without any task-specific parameters or training, by conditioning the model’s generation on a prompt which included a few labeled examples of the task of interest. Since then, interest in using GPT models for ICL has grown significantly, with several recent works introducing methods such as instruction-tuning (Sanh et al., 2022; Wang et al., 2022) or chain-of-thought prompting (Wei et al., 2022) to improve downstream ICL accuracy. Ostensibly, ICL can work for nearly any task that can be defined or described in natural language, and therefore has potential for incredibly broad impact. However, ICL can often still underperform supervised fine-tuning (Bhatia et al., 2023), prompting research in analyzing the mechanisms underlying ICL. One line of work studies in-context learning with linear functions, typically linear regression, characterizing the learnability of these functions with ICL (Li et al., 2023; Garg et al., 2022) and even the learning algorithm a transformer uses (Akyürek et al., 2022; Dai et al., 2023; von Oswald et al., 2023). A second body of work suggests that in-context learning locates existing latent concepts (tasks) which have been already learnt during pretraining (Xie et al., 2021; Wies et al., 2023). Finally, Wei et al. (2023) suggest that model size may change the mechanisms behind ICL from latent inference to actual learning algorithms as size increases. Our work which focuses on Machine Translation, fits into this recent chain of work by demonstrating that there exists a point in the model’s layers where the task has been located.
Ostensibly, ICL can work for nearly any task that can be defined or described in natural language, and therefore has potential for incredibly broad impact. However, ICL can often still underperform supervised fine-tuning (Bhatia et al., 2023), prompting research in analyzing the mechanisms underlying ICL. One line of work studies in-context learning with linear functions, typically linear regression, characterizing the learnability of these functions with ICL (Li et al., 2023; Garg et al., 2022) and even the learning algorithm a transformer uses (Akyürek et al., 2022; Dai et al., 2023; von Oswald et al., 2023). A second body of work suggests that in-context learning locates existing latent concepts (tasks) which have been already learnt during pretraining (Xie et al., 2021; Wies et al., 2023). Finally, Wei et al. (2023) suggest that model size may change the mechanisms behind ICL from latent inference to actual learning algorithms as size increases. Our work which focuses on Machine Translation, fits into this recent chain of work by demonstrating that there exists a point in the model’s layers where the task has been located.
Many works study layers of the model as a natural unit of analysis for interpretability (Hewitt & Liang, 2019; De Cao et al., 2020; Pasad et al., 2021; Durrani et al., 2022; BenShaul & Dekel, 2022; Sajjad et al., 2023). We highlight some of the work which is more closely related to task performance. Xie et al. (2022) study the layer-wise adaptability by a hidden-state variability ratio while Voita et al. (2019a) study evolution of representations in MT-supervised transformer models. Phang et al. (2021) studies when model layers can be skipped by feeding intermediate representations into the final output layer of a pre-trained supervised model. Our work adds to this body of work by considering the perspective of when and where layers are responsible for task location in in-context learning models.
In-Context Machine Translation While GPT models are strong few-shot learners, their pre-training data is historically dominated by English, limiting their ability to perform translation tasks (Hendy et al., 2023). Lin et al. (2022) find that an explicitly multilingual GPT significantly outperforms traditional english models such as GPT-3, and Garcia
et al. (2023) find that such models can even be competitive with supervised MT models in some settings. However, even with explicit multilingual pre-training, in-context MT has been found to be very sensitive to the examples used Liu et al. (2022) and their orders Lu et al. (2022). In response, recent work focuses on how to select prompts that elicit the best downstream MT performance (Agrawal et al., 2022; Sia & Duh, 2023). However, further improvement to translation with GPT models is limited by our understanding of how MT emerges in GPT models. Our work directly analyses when, in layer representations, a GPT model becomes a translation model via in-context learning, and how that may inform decisions around parameter tuning and redundancy.
# 2. Data and Settings
Models We use GPTNEO2.7B (Black et al., 2021), BLOOM3B (Scao et al., 2022), LLAMA7B and LLAMA7Bchat (Touvron et al., 2023), the instruction-tuned variant, in all of our experiments. GPTNEO2.7B has 32 layers and 20 heads, BLOOM3B has 30 layers and 32 heads, while LLAMA7B has 32 layers and 32 heads. The checkpoints we use are from the transformers library (Wolf et al., 2019). GPTNEO was trained on The PILE (Gao et al., 2020), an 825GB text dataset which consists of roughly 98% English data. Despite being mostly monolingual, The PILE contains Europarl which GPTNEO was trained on at a document level (rather than a sentence level). Conversely, BLOOM was trained on the ROOTS corpus (Laurençon et al., 2022), a composite collection of 498 datasets that were explicitly selected to be multilingual, representing 46 natural languages and 13 programming languages. LLAMA training data consists primarily of common crawl, C4, wikipedia, stackexchange as major sources. To our knowledge, there has not been any reports of sentence level parallel corpora in the training datasets of these models.
Data We test our models using FLORES (Goyal et al., 2021) en ↔fr which we report in the main paper, and a small study on extending Section 3 to en ↔pt in the Appendix. Prompt examples are drawn from the development set. We evaluate the generations using BLEU scores, following the implementation from Post (2018).
Prompt Format Our prompts may consist of instructions, examples, both, or none. Importantly, we adopt neutral delimiters, "Q:" and "A:" to separate the prompt and the start of machine generated text. This ensures that the models do not have any information from the delimiters on what the task is. 1
1In an earlier exploration, we found that supplying the model with language indicators only, e.g., "English:", "French:" was sufficient for strong models (llama7b, llama7b-chat) to perform
1In an earlier exploration, we found that supplying the model with language indicators only, e.g., "English:", "French:" was sufficient for strong models (llama7b, llama7b-chat) to perform
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/15c8/15c81be4-5140-4385-84c6-b1b708cd51f2.png" style="width: 50%;"></div>
Figure 1. Graphical explanation of Masking the Attention over Instructions and Examples. The leftmost image has instructions and masks examples (Instr, ExMask), while the right image has both instructions and examples masked (Instr, ExMask). In the interest of space we show only 2 out of 3 variants (see Section A.1 for all variants). In the table, the overline corresponds to the yellow highlights. N and Y refer to absence and presence of either Instruction of Examples. Instr: Instructions and Ex: Examples.
When no natural language instructions are used the model input will be Q: {source_sentence} A: Instructions are given in natural language and take the form: Translate from {L1} to {L2}: Q: {source_sentence} A:, where L1 = English and L2 = French if the source and target languages are English and French respectively. Examples are given after instructions, and similarly delimited by Q: and A:. See Appendix: Table 1 for an example.
# 3. Where does In-context MT happen?
3.1. Layer-from Context Masking
# 3.1. Layer-from Context Masking
In-context learning differs from task-specific supervised learning in that, during test time, the desired task must be identified from the context first, then executed. At what stage in the feed-forward computation does a GPT-style model transition from an in-context learner to a translation model? To explore this question, we introduce layer-from context-masking which masks out all attention weights to the context (instructions or prompts) from a certain layer onwards (see Figure 1 for a graphical description).
In-context learning differs from task-specific supervised learning in that, during test time, the desired task must be identified from the context first, then executed. At what stage in the feed-forward computation does a GPT-style model transition from an in-context learner to a translation model? To explore this question, we introduce layer-from context-masking which masks out all attention weights to the context (instructions or prompts) from a certain layer onwards (see Figure 1 for a graphical description). For Causal Decoder-only Transformer Language Models, given each position i, the Attention weight αij over context positions j, j < i can be computed by a αij = softmax( QKT √dk )ij. Each element in (QKT ) is the dot product between a query vector and key vector qi · kj, where qi = Wqxi, kj = Wkxj for trained weight matrices Wk and Wq.2 We apply the attention mask over the context so that the attention score is (qi·kj)+m(j, u). Here u is the context that we wish to mask, and m(j, u) = � 0 if xj /∈u −∞ if xj ∈u All masks operate from the j-th layer (ℓ) onwards, i.e.
All masks operate from the j-th layer (ℓj) onwards, i.e.
the task without seeing any instructions or examples in the context. 2Readers should note that there is a Wk and Wq weight matrix for each layer and each attention head, but we omit the notation on this for readability.
Under this causal masking treatment masking from layer ℓ, the model must rely on the representations of the target input sentence from layer ℓ+ 1 only to complete the task; if the target sentence representations do not already encode the target task (translation into a specific language) then the model will fail to generate translations.
In other words, the goal is to characterise where the model has "located” the task of translation. In all experiments we mask the examples provided in the context, but to control for the effect of semantic instructions, we ablate over different treatments of the instructions by removing instructions entirely (ExMask), leaving them unmasked (InstrExMask), or masking them together with the examples (InstrExMask). The overline notation indicates the context which are masking over. Also see Figure 1.
# 3.2. Results
# We discuss the central findings of the paper: Models do not need to maintain attention over all of the context across every layer to perform the task.
In all models we observe that when applying masking from {m(j, u)}ℓ:nℓover the context, performance plateaus before the final layer, i.e., when ℓ= nℓ. The results of our experiment for en→fr and fr→en are shown in Figure 2, and additional experiments for GPTNeo and Bloom on en→pt and pt→en are shown in Section A.3.
Different models reach this plateau point at different layers. In GPTNEO this point occurs around layer 25, in BLOOM this point occurs around layer 15-20, and in LLAMA models this occurs around layer 13-15. As English is the dominant language, as expected models can successfully perform translation into English upon earlier layers of masking, than
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d046/d046aafc-6d38-4b9e-9e4a-32780ae602cd.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2. Layer-from context-masking experiments for GPTNeo2.7B, BLOOM3B, Llama7b, Llama7b-chat on en↔fr. The graphs show translation performance when masking contexts from the jth layer onwards. Different lines indicate different treatments of the instruction as described in Figure 1. The dashed black line is the performance when shown both examples and instructions without masking.</div>
# translation out of English.
At this point, the models benefits only marginally, if at all, from attending to the context, suggesting most of the task "location" has already occurred.
There exists critical layers for task location. Prior to the task recognition point, around the middle layers of the models, moving the context mask up a layer results in a significant increase to performance. We consider these critical layers, as instead of a gradual increase in performance, we observe we can observe very steep jumps of over 20 bleu points across the different models. We conjecture that the model is locating the correct task during processing in these middle layers, after which the context is no longer necessary to perform the translation task.
Overall, our findings suggest a 3-phase process to in-context learning: in the first phase, moving the mask up makes little difference in performance, which is close to 0. This suggests that the context has not influenced task location at all. In the second phase, shifting the mask upwards makes a large difference in MT performance, suggesting that the model has started to locate the task but can improve significantly with more processing of the context. Finally, in the third phase, shifting the mask upwards again has little-to-no effect on MT performance, suggesting that the model has fully recognized the task as translation and no longer requires the context to interpret the task.
We provide further observations and ablations in the following sections.
# 3.3. Instruction-tuned vs Non-instruction Tuned Models
When comparing non-instruction tuned vs instruction-tuned LLAMA7B models, we do not observe any noticeable difference in where performance plateaus, i.e., where the model no longer requires attention over the context. This occurs around layers 18 for both LLAMA models in en →fr and around layer 14 for fr →en. The main difference is that instruction-tuned model is able to achieve better performance in the earlier layers for the setting where instructions are present and examples are masked (Instr, ExMask). This is to be expected as these models are tuned towards following instructions.
Overall we find that the observation of task recognition layers and a task recognition point is present across both non-instruction tuned and instruction tuned models, and that this presents itself similarly in both types of models.
# 3.4. The Role of Instructions vs Examples
In separate experiments, we found that when shown only instructions and no examples, GPTNEO and BLOOM models are unable to translate, and their performance is nearly at 0 BLEU Score. For GPTNEO and BLOOM we see that the behavior of the model is similar when no instructions are present (ExMask) and when instructions are masked (Instr,ExMask). However, if the model is given complete access to instructions (InstrExMask), it can use the intermediate processing of examples to reach baseline performance earlier.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fd20/fd2019a2-1532-4c93-89be-d14e744e0019.png" style="width: 50%;"></div>
Figure 3. Layer-from experiments for GPTNEO2.7B, BLOOM3B, LLAMA and LLAMA7B-CHAT on en →fr when masking out from layer j onwards. Orange and blue dashed lines refer to the baselines (without masking) of 0 and 5 prompts with instructions. In view of the smaller models failure to translate at all under the format Q: A: with no examples, we adopt "English:", "French:" as delimiters instead of QA in generating this figure.
# 3.5. Attention to the Context vs Attention to the Input
One possible explanation for the results in Figure 2 is that, rather than identifying the point at which the task is recognized, we have identified the point at which the model no longer requires attending to any other input tokens. To explore this, we run experiments in the en →fr direction where we mask attention to all inputs from a certain layer onwards. This does not include masking over the text the model has generated.
We plot the results in Figure 3; we find that for all models, the layer at which attention can be fully removed is much higher than the layer at which we can remove attention to the context. For GPTNEO and LLAMA, translation performance is never comparable to the baseline with no masking. Conversely, when masking only the context, translation performance improves as early as layer 10 and plateaus at the no-mask baseline much earlier. This supports the interpretation that the curves we observe in Figure 2 are due to the model still requiring attention to the source sentence input.
# 4. Characterising Redundancy in Layers
Recently, Sajjad et al. (2023) found that many layers in pre-trained transformers can be dropped with little harm to downstream tasks; moreover, it is well known neural MT transformer models are known have several redundant heads which are not necessary during test time (Voita et al., 2019b; Michel et al., 2019; Behnke & Heafield, 2021). However, it is not clear if the same trends hold for in-context MT models, and how that redundancy is related to task location versus task execution.
We study the contributions of individual attention-layers by performing a simple layer-wise masking of all self-attention heads for a single layer. When we mask layer j, we are masking the attention mechanism of layer j, that is the MLP of layer j acts directly on the output of layer j −1, rather than the output of the attention-head of layer j. Doing so allows us to study how critical each layer is, where critical
# layers is loosely defined as those that have a large negative impact when masked.
We plot results for each layer all models, using the three combinations of {0 examples, no instructions}, {5 examples, instructions}, {5 examples, no instructions} in Figure 4.3
# 4.1. Are “Critical" Layers Task Locating Layers?
In Section 3, we observed that there are layers for task location. In this section, we observe evidence that there are critical layers which correspond to the task locating layers, providing support for our earlier observations.
For instance for LLAMA7B en →fr, even in the scenarios when examples are provided, we can see a dip in performance around layer 15 to 18. Refering back to Figure 2, we see that this is where most of the task location with large jumps in performance had occurred.
For GPTNeo, we obseve a large set of contiguous layers which significantly decrease performance at around layer 10 to 15. This also corresponds to where most of the task location (large jumps in performance) had occurred for this model in Figure 2.
We note that the critical layers in different models have varying degrees of severity. It is not immediately clear why GPTNEO has such critical layers and suffers compared to the other models, although we note that this is unlikely to be due to size or model architecture as BLOOM is also around the same size as GPTNEO and performs more similarly to LLAMA. We suspect that it could be due to training data or some other factor related to the training dynamics but leave this for future work.
3The combination of {0 examples, no instructions} is not meaningful as the model only receives "Q: <source sentence> A:" as the input and is not expected to do the translation task.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/88c7/88c7208d-9ead-42cf-96c7-d750f4061bfc.png" style="width: 50%;"></div>
Figure 4. Layer-wise masking of self-attention heads for GPTNEO2.7B, BLOOM3B, LLAMA and LLAMA-CHAT on en ↔fr. The orange and blue dotted lines refer to the baselines (without masking) of 0 and 5 prompts with instructions. We observe critical layers nea he middle and redundant layers towards the end of the model.
<div style="text-align: center;">Figure 4. Layer-wise masking of self-attention heads for GPTNEO2.7B, BLOOM3B, LLAMA and LLAMA-CHAT on en ↔fr. Th orange and blue dotted lines refer to the baselines (without masking) of 0 and 5 prompts with instructions. We observe critical layers nea the middle and redundant layers towards the end of the model.</div>
model achieves close to baseline performance by layer-wise masking from ℓ15 for GPTNEO, ℓ26 for BLOOM and ℓ20 for LLAMA. This suggests that these later layers contain redundancy for translation. Overall, observing redundancy in layers is not suprising, and our main contribution is characterising the differences between redundant and critical layers. To explain why models can have redundant layers, we refer to Clark et al. (2019) who identify a phenomena where attention heads attend almost exclusively to delimiter and separator tokens such as [SEP], periods and commas. This is thought to act as a “no-op" as the value of such tokens in changing the current hidden representation is very small. Note that it is then possible to mask entire Transformer layers and still achieve a sensible output due to residual connections in the Transformer architecture at every layer.
model achieves close to baseline performance by layer-wise masking from ℓ15 for GPTNEO, ℓ26 for BLOOM and ℓ20 for LLAMA. This suggests that these later layers contain redundancy for translation.
# 5. Inference Efficiency
Speeding up transformer inference is of great interest to the community (Fournier et al., 2023). We highlight the potential of speeding up inference time as a direct consequence of identifying where task recognition occurs in the model and redundancy of self-attention processing. Our results indicate that we can achieve significant speedups in inference by removing the processing of context-tokens all-together after a certain point in the model, with little to no impact on downstream performance.
Let ℓr be the rth layer where we can mask out the attention of the context across subsequent layers and match the “ceiling" performance. Let k be the number of prompt examples, where each example consists of a pair of parallel sentences.
Then, for a model with nℓlayers, the amount of processing in terms of speed and memory saved is approximately (nℓ− r)/nℓ× (k/k + 1). Using the example of LLAMA7B (32 layers), we see from Figure 2 that the model is very close to it’s ceiling score after processing the examples at layer 14 (ℓ= 14). If we no longer need to process examples after ℓ= 14, under a prompt size of 5 the savings are approximately 45%. For instruction-tuned models which are typically deployed in production, even if we assume that no examples are provided, savings can be non-trivial as very long-form instructions are typically provided to the model in an attempt to control it’s behavior (prompt engineering).
Then, for a model with nℓlayers, the amount of processing in terms of speed and memory saved is approximately (nℓ− r)/nℓ× (k/k + 1).
Using the example of LLAMA7B (32 layers), we see from Figure 2 that the model is very close to it’s ceiling score after processing the examples at layer 14 (ℓ= 14). If we no longer need to process examples after ℓ= 14, under a prompt size of 5 the savings are approximately 45%.
For instruction-tuned models which are typically deployed in production, even if we assume that no examples are provided, savings can be non-trivial as very long-form instructions are typically provided to the model in an attempt to control it’s behavior (prompt engineering).
# 6. Further Analysis
In the following sections, we focus on GPTNEO and BLOOM to conduct deeper analysis on the main phenomena presented in the paper.
# 6.1. Does the Number of Prompts Affect Task Recognition?
In Section 3 we study context-masking with a fixed number of prompts. However, it is not clear if the number of prompts affects how fast, layer-wise, the model is able to recognize the task. We plot these results for en→fr in Figure 5, for both GPTNEO and BLOOM. In general, we find that the number of prompt examples has little effect on which layer the task is recognized at. While there is some variation in performance when the context is masked around the middle layers of the model, the final performance plateau occurs at the same layer regardless of the number of prompts.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ef00/ef004a12-c42f-4c84-a73e-97ab819e4c10.png" style="width: 50%;"></div>
Figure 5. Layer-from context-masking experiments for GPTNeo and BLOOM on en→fr investigating number of examples in the ExMask mask setting (described in Figure 8). The dashed black line refers to no instructions and no examples.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7d20/7d200c4e-36cf-4b64-ad2e-87c45ca4face.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6ed9/6ed9f166-d74c-40d9-b4c6-9d22f535b53d.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8ad5/8ad56e73-ba3f-4ad5-833d-ef94cdf70929.png" style="width: 50%;"></div>
Figure 6. Performance of no-instructions trained Lora layers for GPTNeo and BLOOM on en↔fr. The dashed black line refers to training of all layers together, while the orange (test without instructions) and blue (test with instructions) dashed lines refers to no training. The layers which are most amenable to lightweight fine-tuning occur in the earlier layers before the "task recognition" point.
# 6.2. The Adaptability of Task Layers
Intuitively, the layers prior to "task recognition" should contain information about locating the MT task. To test this intuition, we further explore the adaptability of these layers by lightweight fine-tuning experiments. We trained a single Low-rank Adaptation matrix (LoRA; Hu et al. (2021)) for each layer of the output projection while keeping the rest of the network frozen.4 The model was shown parallel sentences as input, and layers were trained with no explicit translation instructions. We split the dev set of FLORES into 800 training examples and 200 dev examples. Note that this setup is designed to tune the layers for task location. It is highly unlikely that the model can learn translation
4We also experimented with the training separate Key, Query and Value LoRA Layers but found this to be less effective.
knowledge from this small amount of supervision. The LoRA layers were trained for up to 50 epochs with early stopping patience= 5 and threshold= 0.001, with α = 32, r = 32 and dropout= 0.1. The cross-entropy loss was computed only on the target sentence (see Section A.5 for details) and we used the best checkpoint on the 200 held out dev examples for evaluation. We show the results of this experiment in Figure 6; while each layer can be trained to perform better than no finetuning at all, tuning different layers have different impacts on performance. In particular, we find that high performing layers occur at the earlier to middle parts of the network, with the peak often occurring near the start of the "tasklocating" layers from Section 3. In contrast to common fine-tuning wisdom, additional tuning on the later layers has a much smaller impact on final performance for en →fr.
# 6.3. Are There Specialised Attention Heads?
In Section 3, we found that the earlier part of the model is critical for task location from the prompt context, and in Section 4.1 we found both critical and redundant layers to the MT task. In this section, we increase the level of granularity to that of attention heads instead of layers.
A well established finding for supervised encoder-decoder MT models, is that up to 90% of the attention heads can be pruned while minimising fall in translation performance (Voita et al., 2019b; Behnke & Heafield, 2020; Michel et al., 2019). We note that asking about the extent of pruning is a slightly ill-formed research question, as it depends on the type of pruning technique used. However broad trends of highly prunable models have been observed in the supervised MT paradigm. In the in-context paradigm, there is no explicit supervision. Thus it is not clear if the task knowledge is spread across a much larger number of attention heads, or similarly specialised to a few heads. For instance, Bansal et al. (2023) studied attention-head importance for a broader set of ICL tasks, finding that the most important heads for ICL occur in the middle layers of the model.
# 6.4. Training Attention Head Gates with L0 regularisation
For a scalable approach to pruning, we opt to train selfattention head gates following Voita et al. (2019b) using the technique of differentiable L0 regularization (Louizos et al., 2017). Let the attention head gates g ∈Rnh×nℓbe a set of trainable parameters, where nh is the number of attention heads per layer, and nℓis the number of layers. Let the original output of each attention head be vj, gated outputs ˜vj are obtained by elementwise multiplication of the gate value gj, i.e., ˜vj = gj ⊙vj. For {(x, y)}n source sentence (x) and target sentence (y) training pairs, a model f and loss function L, Lp regularisation adds a λ weighted
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/375a/375a58e2-f084-4dc0-a27d-ce29958198cb.png" style="width: 50%;"></div>
Figure 7. Visualisation of attention head masks for GPTNeo and BLOOM, learned with L0(λ = 0.01) regularisation under a 0-prompt train scheme in en →fr. A value of 0 (in black) indicates that the attention head is effectively masked out by the trained attention gate. Around 10% of attention heads are masked out i.e., redundant, with a majority of them occuring at the later layers for GPTNeo and distributed across layers for BLOOM. fr →en is availble in Section A.7
penalty associated with the complexity of the parameters. 5 The L0 loss is non-differentiable as it involves raw counts of parameters. As a work around, g can be approximated with random variables drawn from a Binary concrete distribution (Maddison et al., 2016; Jang et al., 2016).6 We refer the reader to Louizos et al. (2017) for the relevant technical exposition. Details of training are provided in Section A.6.
# 6.5. Studying Redundancy via Compression
We noted that GPTNEO has some critical differences from BLOOM and LLAMA in terms of having critical layers (see Section 4.1). To what extent are there specialised attention heads for MT in the GPT-style models? If there were specialised heads, we would expect the model to be highly compressable/prunable to a select few heads. We plot a grid map of learned attention gate values for en →fr, where 0 indicates that the head is masked out (Figure 7). We find that most of the masked heads are distributed at the later layers for GPTNeo and are distributed across layers for BLOOM. This appears consistent with Section 4.1’s observations that redundancy is more focused at certain layers in GPTNeo, and more spread out across the layers for Bloom. In addition, we note that there are no "few" specialised heads, which directly contrasts with the literature on compression in supervised MT models (Voita et al., 2019b; Michel et al., 2019). Potential reasons for this difference might include data distribution and model architecture, or cross-entropy loss associated with task tuning for MT vs 5L2 regularisation has the effect of reducing the magnitude of all g, L1 regularisation has the effect of reducing the magnitude of several attention heads to a very small value (but not exactly 0), while L0 regularisation has the effect of driving g values to exactly 0. 6The class of Concrete distributions was invented to work around the problem of automatic differentiation of stochastic computation graphs.
We noted that GPTNEO has some critical differences from BLOOM and LLAMA in terms of having critical layers (see Section 4.1). To what extent are there specialised attention heads for MT in the GPT-style models? If there were specialised heads, we would expect the model to be highly compressable/prunable to a select few heads. We plot a grid map of learned attention gate values for en →fr, where 0 indicates that the head is masked out (Figure 7). We find that most of the masked heads are distributed at the later layers for GPTNeo and are distributed across layers for BLOOM. This appears consistent with Section 4.1’s observations that redundancy is more focused at certain layers in GPTNeo, and more spread out across the layers for Bloom.
In addition, we note that there are no "few" specialised heads, which directly contrasts with the literature on compression in supervised MT models (Voita et al., 2019b; Michel et al., 2019). Potential reasons for this difference might include data distribution and model architecture, or cross-entropy loss associated with task tuning for MT vs
5L2 regularisation has the effect of reducing the magnitude of all g, L1 regularisation has the effect of reducing the magnitude of several attention heads to a very small value (but not exactly 0), while L0 regularisation has the effect of driving g values to exactly 0. 6The class of Concrete distributions was invented to work around the problem of automatic differentiation of stochastic computation graphs.
non-specific training on large corpora. We leave this as an open question for future work.
# 7. Conclusion
We demonstrate evidence that In-context Causal Decoder models locate the translation task at a specific layers during forward inference. To study this, we introduced causal masking of self-attention over the context from layer ℓonwards (Section 3). The findings generalise across models of different sizes and in both non instruction-tuned and instruction-tuned models. We further identify certain layers as task critical, and show that this corresponds to the task recognition point of the model (Section 4.1) and is not influenced by increasing number of examples (Section 6.1) shown to the models. Our central finding that models do not need to maintain attention over all of the context across every layer has direct implications for inference efficiency of transformers, with estimated up to 45% cost-savings for llama model with 5 examples (Section 5).
We demonstrate evidence that In-context Causal Decoder models locate the translation task at a specific layers during forward inference. To study this, we introduced causal masking of self-attention over the context from layer ℓonwards (Section 3). The findings generalise across models of different sizes and in both non instruction-tuned and instruction-tuned models. We further identify certain layers as task critical, and show that this corresponds to the task recognition point of the model (Section 4.1) and is not influenced by increasing number of examples (Section 6.1) shown to the models.
Our central finding that models do not need to maintain attention over all of the context across every layer has direct implications for inference efficiency of transformers, with estimated up to 45% cost-savings for llama model with 5 examples (Section 5).
Contrary to common fine-tuning wisdom, we show that it is sometimes beneficial to target middle layers for fine-tuning the model which could be associated with task recognition ( Section 6.2). Finally, we trained attention head gates using differentiable L0 regularisation (Section 6.3), and found that around 10% of attention heads can be masked. These are mostly distributed across the later layers of the model, providing some support for the idea that later layers are redundant but layers are responsible for locating the translation task. Although we have characterised this phenomena using the example of translation we believe that the broad findings are likely to generalise to different tasks.
Limitations and Future Work We have conducted extensive investigations focusing on the task of translation on a high-resource language pair, with a small extension to en ↔pt. In future work, we hope to extend this analysis to other sequence or classification tasks as well as true novel tasks.
Reproducibility The MT dataset that we use, FLORES (Goyal et al., 2021), is fully open-source and well-known in the community. Models are open-source and freely available on Huggingface (Wolf et al., 2019). We used models of "reasonable" size (3B and 7B parameters) that can be run with consumer grade GPUs, making our reproducible to most academic institutions. Code to reproduce all the experiments will be made available subsequently.
Impact Statement (Ethics and Societal Consequences) There are no known ethical concerns as these are exploratory studies on open-source LLMs.
ACKNOWLEDGMENTS We would like to thank Daniel Kashabi and Marc Marone for feedback on earlier drafts.
We would like to thank Daniel Kashabi and Marc Marone for feedback on earlier drafts.
# References
Agrawal, S., Zhou, C., Lewis, M., Zettlemoyer, L., and Ghazvininejad, M. In-context examples selection for machine translation. arXiv preprint arXiv:2212.02437, 2022.
Akyürek, E., Schuurmans, D., Andreas, J., Ma, T., and Zhou, D. What learning algorithm is in-context learning? investigations with linear models. arXiv preprint arXiv:2211.15661, 2022.
Zhou, D. What learning algorithm is in-context learning? investigations with linear models. arXiv preprint arXiv:2211.15661, 2022. Bansal, H., Gopalakrishnan, K., Dingliwal, S., Bodapati, S., Kirchhoff, K., and Roth, D. Rethinking the role of scale for in-context learning: An interpretability-based case study at 66 billion scale. In Rogers, A., BoydGraber, J., and Okazaki, N. (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11833– 11856, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long. 660. URL https://aclanthology.org/2023. acl-long.660. Behnke, M. and Heafield, K. Losing heads in the lottery: Pruning transformer attention in neural machine translation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2664–2674, 2020. Behnke, M. and Heafield, K. Pruning neural machine translation for speed using group lasso. In Proceedings of the sixth conference on machine translation, pp. 1074–1086, 2021. Ben-Shaul, I. and Dekel, S. Nearest class-center simplification through intermediate layers. In Topological, Algebraic and Geometric Learning Workshops 2022, pp. 37–47. PMLR, 2022. Bhatia, K., Narayan, A., Sa, C. D., and Ré, C. Tart: A plugand-play transformer module for task-agnostic reasoning, 2023. Black, S., Leo, G., Wang, P., Leahy, C., and Biderman, S. GPT-Neo: Large scale autoregressive language modeling with Mesh-Tensorflow, March 2021. URL https:// doi.org/10.5281/zenodo.5297715. Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners.
ansal, H., Gopalakrishnan, K., Dingliwal, S., Bodapati, S., Kirchhoff, K., and Roth, D. Rethinking the role of scale for in-context learning: An interpretability-based case study at 66 billion scale. In Rogers, A., BoydGraber, J., and Okazaki, N. (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11833– 11856, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long. 660. URL https://aclanthology.org/2023. acl-long.660.
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners.
Advances in neural information processing systems, 33: 1877–1901, 2020. Clark, K., Khandelwal, U., Levy, O., and Manning, C. D. What does BERT look at? an analysis of BERT’s attention. In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pp. 276–286, Florence, Italy, August 2019. Association for Computational Linguistics. doi: 10.18653/v1/ W19-4828. URL https://aclanthology.org/ W19-4828. Dai, D., Sun, Y., Dong, L., Hao, Y., Ma, S., Sui, Z., and Wei, F. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers. In ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models, 2023. De Cao, N., Schlichtkrull, M. S., Aziz, W., and Titov, I. How do decisions emerge across layers in neural models? interpretation with differentiable masking. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 3243–3255, 2020. Durrani, N., Sajjad, H., Dalvi, F., and Alam, F. On the transformation of latent space in fine-tuned nlp models. arXiv preprint arXiv:2210.12696, 2022. Fournier, Q., Caron, G. M., and Aloise, D. A practical survey on faster and lighter transformers. ACM Computing Surveys, 55(14s):1–40, 2023. Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., et al. The Pile: An 800GB dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020. Garcia, X., Bansal, Y., Cherry, C., Foster, G., Krikun, M., Feng, F., Johnson, M., and Firat, O. The unreasonable effectiveness of few-shot learning for machine translation. arXiv preprint arXiv:2302.01398, 2023. Garg, S., Tsipras, D., Liang, P. S., and Valiant, G. What can transformers learn in-context? a case study of simple function classes. Advances in Neural Information Processing Systems, 35:30583–30598, 2022. Goyal, N., Gao, C., Chaudhary, V., Chen, P.-J., Wenzek, G., Ju, D., Krishnan, S., Ranzato, M., Guzmán, F., and Fan, A. The flores-101 evaluation benchmark for low-resource and multilingual machine translation. 2021. Hendy, A., Abdelrehim, M., Sharaf, A., Raunak, V., Gabr, M., Matsushita, H., Kim, Y. J., Afify, M., and Awadalla, H. H. How good are gpt models at machine translation? a comprehensive evaluation. arXiv preprint arXiv:2302.09210, 2023.
Hewitt, J. and Liang, P. Designing and interpreting probes with control tasks. arXiv preprint arXiv:1909.03368,
Laurençon, H., Saulnier, L., Wang, T., Akiki, C., Villanova del Moral, A., Le Scao, T., Von Werra, L., Mou, C., González Ponferrada, E., Nguyen, H., et al. The bigscience roots corpus: A 1.6 tb composite multilingual dataset. Advances in Neural Information Processing Systems, 35:31809–31826, 2022.
Li, S., Song, Z., Xia, Y., Yu, T., and Zhou, T. The closeness of in-context learning and weight shifting for softmax regression. arXiv preprint arXiv:2304.13276, 2023.
Lin, X. V., Mihaylov, T., Artetxe, M., Wang, T., Chen, S., Simig, D., Ott, M., Goyal, N., Bhosale, S., Du, J., Pasunuru, R., Shleifer, S., Koura, P. S., Chaudhary, V., O’Horo, B., Wang, J., Zettlemoyer, L., Kozareva, Z., Diab, M., Stoyanov, V., and Li, X. Few-shot learning with multilingual generative language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 9019– 9052, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.616. URL https:// aclanthology.org/2022.emnlp-main.616.
Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., and Chen, W. What makes good in-context examples for GPT-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pp. 100– 114, Dublin, Ireland and Online, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. deelio-1.10. URL https://aclanthology.org/ 2022.deelio-1.10.
Louizos, C., Welling, M., and Kingma, D. P. Learning sparse neural networks through l_0 regularization. arXiv preprint arXiv:1712.01312, 2017.
Louizos, C., Welling, M., and Kingma, D. P. Learning sparse neural networks through l_0 regularization. arXiv preprint arXiv:1712.01312, 2017. Lu, Y., Bartolo, M., Moore, A., Riedel, S., and Stenetorp, P. Fantastically ordered prompts and where to
Sajjad, H., Dalvi, F., Durrani, N., and Nakov, P. On the effect of dropping layers of pre-trained transformer models. Computer Speech & Language, 77:101429, 2023.
Sanh, V., Webson, A., Raffel, C., Bach, S., Sutawika, L., Alyafeai, Z., Chaffin, A., Stiegler, A., Raja, A., Dey, M., Bari, M. S., Xu, C., Thakker, U., Sharma, S. S., Szczechla, E., Kim, T., Chhablani, G., Nayak, N., Datta, D., Chang, J., Jiang, M. T.-J., Wang, H., Manica, M., Shen, S., Yong, Z. X., Pandey, H., Bawden, R., Wang, T., Neeraj, T., Rozen, J., Sharma, A., Santilli, A., Fevry, T., Fries, J. A., Teehan, R., Scao, T. L., Biderman, S., Gao, L., Wolf, T., and Rush, A. M. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations, 2022. URL https:// openreview.net/forum?id=9Vrb9D0WI4. Scao, T. L., Fan, A., Akiki, C., Pavlick, E., Ili´c, S., Hesslow, D., Castagné, R., Luccioni, A. S., Yvon, F., Gallé, M., et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022. Sia, S. and Duh, K. In-context learning as maintaining coherency: A study of on-the-fly machine translation using large language models. arXiv preprint arXiv:2305.03573, 2023.
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
Voita, E., Sennrich, R., and Titov, I. The bottom-up evolution of representations in the transformer: A study with machine translation and language modeling objectives. arXiv preprint arXiv:1909.01380, 2019a.
Voita, E., Talbot, D., Moiseev, F., Sennrich, R., and Titov, I. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 5797–5808, Florence, Italy, July 2019b. Association for Computational Linguistics. doi: 10.18653/v1/P19-1580. URL https://aclanthology.org/P19-1580.
von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov, M. Transformers learn in-context by gradient descent, 2023.
Wang, Y., Mishra, S., Alipoormolabashi, P., Kordi, Y., Mirzaei, A., Naik, A., Ashok, A., Dhanasekaran, A. S., Arunkumar, A., Stap, D., Pathak, E., Karamanolakis, G., Lai, H., Purohit, I., Mondal, I., Anderson, J., Kuznia, K., Doshi, K., Pal, K. K., Patel, M., Moradshahi, M., Parmar, M., Purohit, M., Varshney, N., Kaza, P. R., Verma, P., Puri, R. S., Karia, R., Doshi, S., Sampat, S. K., Mishra, S., Reddy A, S., Patro, S., Dixit, T., and Shen, X. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 5085– 5109, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.340. URL https:// aclanthology.org/2022.emnlp-main.340. Wei, J., Wang, X., Schuurmans, D., Bosma, M., brian ichter, Xia, F., Chi, E. H., Le, Q. V., and Zhou, D. Chain of thought prompting elicits reasoning in large language models. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/ forum?id=_VjQlMeSB_J. Wei, J., Wei, J., Tay, Y., Tran, D., Webson, A., Lu, Y., Chen, X., Liu, H., Huang, D., Zhou, D., and Ma, T. Larger language models do in-context learning differently, 2023. Wies, N., Levine, Y., and Shashua, A. The learnability of in-context learning. arXiv preprint arXiv:2303.07895, 2023.
Wies, N., Levine, Y., and Shashua, A. The learnability of in-context learning. arXiv preprint arXiv:2303.07895, 2023.
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al. Huggingface’s transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771, 2019. Xie, S., Qiu, J., Pasad, A., Du, L., Qu, Q., and Mei, H. Hidden state variability of pretrained language models can guide computation reduction for transfer learning. arXiv preprint arXiv:2210.10041, 2022. Xie, S. M., Raghunathan, A., Liang, P., and Ma, T. An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080, 2021.
# A. Appendix
# A.1. Graphical View of Context Masking Experiments
A.2. Prompt Format
A.3. Additional Results on English & Spanish
In addition to the language pairs en →fr and fr →en, we also run experiments on English and Spanish language pairs, both en →es and es →en. Due to space limitations, we plot the results of those experiments here. Overall, we see largely identical trends on both directions of English and Spanish to what we observe on English and French translation tasks, leading us to conclude that our conclusions generalize across different translation tasks.
# A.4. Autoregressive Decoder only Transformer
The transformer consists of stacked blocks of self-attention, which itself consists of smaller units of self-attention heads that are concatenated before being fed through a fully connected layer. In autoregressive decoder-only transformers, training and inference adopts a causal mask, where current positions are only able to attend to previous timesteps, instead of being able to attend to the entire input sequence. Unlike encoder-decoder NMT models where source and target sentence have separate processing transformer blocks, decoder-only means that the same model weights are both used to “encode" the source sentence and “decode" the target sentence in a single continuous sequence.
# A.5. Training with Autoregressive Translation
The original language modeling objective in GPT training involves predicting the entire input token sequence which consists of both the source and target sentence (shifted by 1 position). We found this to produce slightly worse results than only minimising the negative log likelihood of predicting the target sentence to be translated, and not the entire sequence. We consider this autoregressive translation training.
# A.6. L0 Attention Gate Training
Training Details For Section 6.5, We train using Adam Optimizer (β1 = 0.9, β2 = 0.999) with a batch size of 32, and learning rate of 0.001, early stopping patience of 10 and threshold of 0.01. We initialise attention head gates to be 1 instead of random or 0.5 as this leads to faster convergence. We experiment with two different training settings, the 0-prompts Train setting and the 5-prompts Train setting. As described in Section A.5, we train the model by predicting only the target sentence, conditioned on the context. In the 0-prompt setting, the context consists of the instructions and the source sentence to be translated. In the 5-prompt setting, the context con-
sists of the instructions, 5 prompt examples, and the source sentence to be translated.
In the 0-prompt setting, the conditional prefix consists of the instructions and the source sentence to be translated. In the 5-prompt setting, the conditional prefix consists of the instruction, 5 source target sentence pairs, and the source sentence to be translated.
Data We used the first 10,000 lines of en →fr from WMT06 Europarl (Koehn, 2005) for training.7 To test the generalisability of trained attention head gates, we use a different test domain, FLORES (Goyal et al., 2021) to reflect the scarcity of in-domain data. We also test an additional language direction en→pt in FLORES to see if training can generalise across languages.
Training Details We train using Adam Optimizer (β1 = 0.9, β2 = 0.999) with a batch size of 32, and learning rate of 0.001. We use a large early stopping patience of 10 and threshold of 0.01, and train for up to 100 epochs. This is due to the nature of L0 training; we do not expect performance to improve over many iterations and would like the attention gates to keep training as long as there is no large loss in performance. We initialise attention head gates to be 1 instead of random or 0.5 as this leads to much faster convergence and better performance. For the regularisation weight λ, we search over a hyperparameter set of {0.1, 0.01, 0.001, 0.0001} and found 0.01 performs best on the validation set.
# A.7. L0 head masking experiments.
# Additional experiments on L0 head masking in the fr→en and es→en direction.
Additional experiments on L0 head masking in the fr→en and es→en direction.
A.8. Generalisability of L0 gate training
# A.8. Generalisability of L0 gate training
We experiment with 0-prompts and 5-prompts in training and using λ = 0 (no regularisation) and λ = 0.01. L0 training for the 0-prompts shows some gains for the 0-prompts test case, and with no loss on the 5-prompts test case (Table 2). Notably, this persists in en→pt, a different language direction from training.
The robustness of translation performance under multiple testing conditions (number of prompts, datasets, language directions) gives some confidence that the trained discrete attention head gates from L0 support a general ability to translate (Table 2). In contrast, the soft attention head gates without regularisation (λ = 0) appear to overfit as they perform well on some conditions but deteriorate in others. We observe that 0-prompt training for L0(λ = 0.01) also 7Data available from https://www.statmt.org/ europarl/
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5611/5611a030-6c20-4736-81c8-da5d599294e1.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 8. Graphical explanation of Masking the Attention over Instructions and Examples. The leftmost picture has no instructions and masks examples (ExMask), the middle picture has instructions and masks examples (Instr, ExMask), and the rightmost picture masks both instructions and examples (Instr, ExMask).</div>
Translate English to French.
Q: A discomfort which lasts ..
A:
Un malaise qui dure
Q: HTML is a language for formatting
A:
HTML est un langage de formatage
...
...
Q: After you become comfortable with formatting ..
A:
Table 1. A single continuous input sequence presented to the model for decoding a single test source sentence “After you become comfortable with formatting..”. Given the entire sequence as input, the model proceeds to generate the target sequence.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d83b/d83b4d90-e08f-4386-a148-f729156993ca.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 9. Context-masking and Layer-masking results on the English ↔Portugese translation task. Critically, we see nearly identical trends to what we see in Figure 2 and Figure 4 on the English to French translation task, suggesting our results generalize across language</div>
outperforms 5-prompts which is slightly suprising since 5-prompts has more information in the prefix to locate the translation task. One possibility is that the model overfit to the Europarl domain where the training prompts were drawn from.
# A.9. Qualitative Analysis of Layer-wise Masking
GPTNEO Masking ℓ4:8 results in a drop in performance for the 0-prompt setting but not the 5-prompt setting (Figure 4), which suggests that ℓ4:8 are not related to the process-
ing of prompt examples. We emphasise that this interpretation mostly holds at an aggregate level and is not strictly for each instance. For Test Instance ID 575, the model still generates a copy of the English source sentence up to the masking of ℓ25 for the 0-prompts without instructions setting (Table 4). This suggests that uncertainty over the task is maintained across layers even though the contributions towards task location may be greater from specific layers.
BLOOM is observed to be more robust to masking of layers; suggesting that task location is more distributed.
Base
0-prompts
5-prompts
Base
0-prompts
5-prompts
λ=0
λ=.01
λ=0
λ=.01
λ=0
λ=.01
λ=0
λ=.01
0-prompts
18.3
21.4
20.1
18.9
19.3
6.7
15.7
8.6
13.2
6.4
5-prompts
24.3
24.5
24.1
23.6
24.2
25.9
19.6
25.8
24.3
26.0
Train:, Test:
Train:, Test:
Train: en→fr, Test: en→fr
Table 2. Performance when using trained attention head gates for L0 with regularisation λ = .01. λ = 0 refers to training w regularisation. 0 and 5 prompts were used in the context for training. We highlight values which are greater or worse than 0.5 points from baseline. Note that as these are compression experiments, we do not expect L0 to perform better than baseline.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/71d2/71d2bcb8-ff3c-41ba-a321-cf32e3fff925.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 10. Visualisation of attention head masks for GPTNeo and BLOOM, learned with L0(λ = 0.01) regularisation under a 0-prompt train scheme. A value of 0 (in black) indicates that the attention head is effectively masked out by the trained attention gate. A majority of them occuring at the later layers for GPTNeo and distributed across layers for BLOOM.</div>
For the 5-prompt setting, the performance only decreases very slightly. For the 0-prompt setting, we observe that similar to GPTNEO, performance drops when masking out the middle layers. At the aggregate level, BLOOM appears to still be translating (> 0 BLEU) even when layers are masked. However we observe that the drop in performance is due to around 40 to 50% of the test sentences scoring < 5 BLEU points. There is a clear failure to translate, not simply producing poorer translations.
Train: en→fr, Test: en→pt
<div style="text-align: center;">Train: en→fr, Test: en→pt</div>
layer
id
lang
BLEU
text
1
600
cy
0.00
uffose
1
575
ca
0.00
B marriages{
2
600
et
0.00
sses room ( I
2
575
no
0.00
NaN
3
600
fr
1.90
C’est la même chose que l’on a fait avec les virus.
3
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
4
600
no
0.00
NaN
4
575
no
0.00
NaN
5
600
fr
78.78
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
5
575
fr
72.98
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent toujours à un dinosaur.
6
600
fr
78.78
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
6
575
fr
60.29
Mais il y a beaucoup de choses à propos de oiseaux qui ressemblent encore à un dinosaur.
7
600
fr
13.94
Hershey et Chase ont implanté leur propre gène dans un bactérie.
7
575
fr
72.98
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent toujours à un dinosaur.
8
600
no
0.00
NaN
8
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosau.
9
600
no
0.00
NaN
9
575
fr
82.82
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent toujours à un dinosaure.
10
600
no
0.00
NaN
10
575
en
2.73
But there are a lot of things about birds that still look like a dinosaur.
11
600
en
4.78
Hershey and Chase used phages, or viruses, to implant their own DNA into a bacterium.
11
575
en
2.73
But there are a lot of things about birds that still look like a dinosaur.
12
600
no
0.00
NaN
12
575
no
0.00
NaN
13
600
no
0.00
NaN
13
575
fr
35.75
Mais il y a beaucoup de choses que je ne comprends pas.
14
600
en
4.78
Hershey and Chase used phages, or viruses, to implant their own DNA into a bacterium.
14
575
en
2.73
But there are a lot of things about birds that still look like a dinosaur.
15
600
fr
76.48
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bacillus.
15
575
en
2.73
But there are a lot of things about birds that still look like a dinosaur.
16
600
fr
78.78
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
16
575
fr
70.18
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent toujours comme un dinosaurof.
17
600
fr
82.32
Les Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans une bactérie.
17
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
18
600
fr
78.78
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre génome dans un bactérie.
18
575
fr
66.38
Mais il y a beaucoup de choses sur les oiseaux qui aussi ressemble à un dinosaures.
19
600
fr
59.33
Les héritiers de Hershey et de Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
19
575
fr
47.91
Mais il y a beaucoup de choses à propos de les oiseaux qui ressemblent toujours à un dinosaur.
20
600
fr
48.82
Hershey et Chase ont utilisé les phages, ou les virus, pour implanter leur propre gène dans un bactérie.
20
575
en
2.73
But there are a lot of things about birds that still look like a dinosaur.
21
600
fr
78.78
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
21
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
22
600
fr
48.82
Hershey et Chase ont utilisé les phages, ou les virus, pour implanter leur propre gène dans un bactérie.
22
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaurof.
23
600
fr
78.78
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
23
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
24
600
fr
78.78
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre génome dans un bactérie.
24
575
fr
62.72
Mais il y a beaucoup de choses à propos de les oiseaux qui ressemblent encore à un dinosaur.
25
600
fr
78.78
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
25
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
26
600
fr
78.78
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
26
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
27
600
fr
66.28
Hershey et Château ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
27
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
28
600
fr
78.78
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
28
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
29
600
fr
59.33
Les héritiers de Hershey et de Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
29
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaurof.
30
600
fr
78.78
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
30
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
31
600
fr
78.78
Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
31
575
fr
88.44
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
32
600
fr
6.44
Les héritiers de Hershey et de Chase ont été capables de l’implanter dans un bactérie.
32
575
fr
51.52
Mais il y a beaucoup de choses sur les oiseaux que sont encore aussi vus comme un dinosaures.
Table 3. 0-prompts with instructions, masking layer by layer of GPTNEO2.7B
Table 3. 0-prompts with instructions, masking layer by layer of GPTNEO2.7B
1
600
no
0.0
NaN
2
575
NaN
0.0
,
2
600
no
0.0
NaN
3
575
fr
2.3
C’est pas un oiseau, c’est un dinosaur.
3
600
fr
0.8
[phare] Phare, phare, phare, phare, phare, phare, phare, phare, phare, phare, phare, ...,
4
575
en
2.7
"I think it’s a dinosaur, I think it’s a dinosaur."
4
600
fr
2.6
Les virus, c’est-ce qu’on dit? C’un mot? C’est pas un mot? C’un mot? C’un’un? ...
5
575
fr
42.9
Mais il y a beaucoup de choses à propos de oiseaux qui ressemblent toujours comme un dinosaur.
5
600
fr
73.6
L’Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
6
575
fr
53.8
Mais il y a beaucoup de choses à propos de oiseaux qui ressemblent toujours à un dinosaure.
6
600
fr
74.9
Les Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
7
575
fr
76.2
Et il y a beaucoup de choses sur les oiseaux qui ressemblent toujours à un dinosaure.
7
600
fr
83.8
Les Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre ADN dans un bactérie.
8
575
fr
88.4
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosau.
8
600
fr
83.8
L’usine Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre ADN dans un bactérie.
9
575
en
1.5
The bird is a dinosaur.
9
600
fr
33.9
Les hémoglobine et Chase utilisent les phages, ou les virus, pour implanter leur propre gène dans un bactérie.
10
575
en
2.7
But there are a lot of things about birds that still look like a dinosaur.
10
600
fr
11.4
Les phages, ou virus, ont implanté leur propre gène dans un bactérie.
11
575
en
2.6
I think it’s a good idea to have a little bit of a bird in your pocket.
11
600
en
0.0
The French have a saying: "The French have a saying: "The French have a saying: "The French have a saying:...
12
575
en
2.7
But there are a lot of things about birds that still look like a dinosaur.
12
600
en
1.7
The bacterium was then able to use the phage to infect other bacteria.
13
575
en
2.7
But there are a lot of things about birds that still look like a dinosaur.
13
600
fr
18.7
L’entreprise Hershey a utilisé des phages pour implanter leur propre DNA dans leur bactérie.
14
575
en
2.7
But there are a lot of things about birds that still look like a dinosaur.
14
600
en
4.8
Hershey and Chase used phages, or viruses, to implant their own DNA into a bacterium.
15
575
fr
3.0
C’est pas un truc de poulet, c’est un truc de poulet.
15
600
fr
35.7
L’université de Paris-Sud a utilisé des phages, ou viraux, pour implanter leur propre gène dans un bacillus.
16
575
en
2.7
But there are a lot of things about birds that still look like a dinosaur.
16
600
fr
74.9
Les Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre génome dans un bactérie.
17
575
en
2.7
But there are a lot of things about birds that still look like a dinosaur.
17
600
fr
82.3
Les Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans une bactérie.
18
575
en
2.7
But there are a lot of things about birds that still look like a dinosaur.
18
600
fr
74.9
Les Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre génome dans un bactérie.
19
575
fr
44.2
Mais il y a beaucoup de choses à propos de oiseaux qui ressemblent toujours à un dinosaur.
19
600
fr
59.3
Les héritiers de Hershey et de Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
20
575
en
2.7
But there are a lot of things about birds that still look like a dinosaur.
20
600
fr
46.4
Les Hershey et Chase ont utilisé les phages, ou les virus, pour implanter leur propre gène dans un bactérie.
21
575
en
2.7
But there are a lot of things about birds that still look like a dinosaur.
21
600
fr
74.9
L’usine Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
22
575
en
2.7
But there are a lot of things about birds that still look like a dinosaur.
22
600
fr
56.3
Les Hershey et Chase ont utilisé les phages, ou les virus, pour implanter leur propre ADN dans un bactérie.
23
575
en
2.7
But there are a lot of things about birds that still look like a dinosaur.
23
600
fr
82.9
L’Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre ADN dans un bactérie.
24
575
fr
60.3
Mais il y a beaucoup de choses à propos de oiseaux qui ressemblent encore à un dinosaur.
24
600
fr
37.0
L’usine Hershey et Chase utilisaient les phages, ou les virus, pour implanter leur propre génome dans un bactérie.
25
575
en
2.7
But there are a lot of things about birds that still look like a dinosaur.
25
600
fr
74.9
Les Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
26
575
fr
71.6
Mais il y a beaucoup de choses à propos de oiseaux qui ressemblent encore à un dinosaure.
26
600
fr
73.6
L’Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
27
575
fr
88.4
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaurof.
27
600
fr
63.0
Les Hershey et Château ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
28
575
fr
44.2
Mais il y a beaucoup de choses à propos de oiseaux qui ressemblent toujours à un dinosaur.
28
600
fr
74.9
Les Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
29
575
fr
87.0
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaurofrench:
29
600
fr
53.4
L’entreprise de la filière Hershey a utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
30
575
fr
82.8
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent toujours à un dinosaure.
30
600
fr
74.9
Les Hershey et Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
31
575
fr
82.8
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent toujours à un dinosaure.
31
600
fr
59.3
Les hémoglobins de Hershey et de Chase ont utilisé des phages, ou des virus, pour implanter leur propre gène dans un bactérie.
32
575
fr
67.5
Mais il y a beaucoup de choses sur les oiseaux qui aussi ressemblent à un dinosaurof.
32
600
fr
6.7
L’Hershey et le Chase ont été capables de l’implanter leur propre gène dans un bactérie.
Table 4. 0-prompts without instructions, masking layer by layer of GPTNEO2.7B
1
902
en
0.0
: of
1
575
en
0.0
of
2
902
en
0.0
of(n, very very- ofS First
2
575
da
0.0
f(
3
902
fr
100.0
Les scènes sont affichées sur les pyramides et les différentes pyramides sont éclairées.
3
575
fr
88.4
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
4
902
en
0.0
the, the French, the French, the English, the, the, the, the, the, the, the, the, the, the,
4
575
no
0.0
NaN
5
902
fr
65.9
Les scènes sont affichées sur les pyramides et les différents pyramides sont éclairés.
5
575
fr
88.4
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
6
902
fr
100.0
Les scènes sont affichées sur les pyramides et les différentes pyramides sont éclairées.
6
575
fr
88.4
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaur.
7
902
fr
100.0
Les scènes sont affichées sur les pyramides et les différentes pyramides sont éclairées.
7
575
fr
100.0
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaure.
8
902
fr
100.0
Les scènes sont affichées sur les pyramides et les différentes pyramides sont éclairées.
8
575
fr
100.0
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaure.
9
902
fr
65.9
Les scènes sont affichées sur les pyramides et les différents pyramides sont éclairés.
9
575
fr
100.0
Mais il y a beaucoup de choses sur les oiseaux qui ressemblent encore à un dinosaure.
10
902
fr
100.0
Les scènes sont affichées sur les pyramides et les différentes pyramides sont éclairées.
10
575
fr
100.0
Mais il y a beaucoup de choses sur