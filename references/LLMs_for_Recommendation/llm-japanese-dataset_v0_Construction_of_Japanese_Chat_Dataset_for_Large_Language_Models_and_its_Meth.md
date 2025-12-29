# llm-japanese-dataset v0: Construction of Japanese Chat Dataset for Large Language Models and its Methodology
Masanori HIRANO, Masahiro SUZUKI, and Hiroki SAKAJI
Abstract This study constructed a Japanese chat dataset for tuning large language models (LLMs), which consist of about 8.4 million records. Recently, LLMs have been developed and gaining popularity. However, high-performing LLMs are usually mainly for English. There are two ways to support languages other than English by those LLMs: constructing LLMs from scratch or tuning existing models. However, in both ways, datasets are necessary parts. In this study, we focused on supporting Japanese in those LLMs and making a dataset for training or tuning LLMs in Japanese. The dataset we constructed consisted of various tasks, such as translation and knowledge tasks. In our experiment, we tuned an existing LLM using our dataset and evaluated the performance qualitatively. The results suggest that our dataset is possibly beneficial for LLMs. However, we also revealed some difficulties in constructing LLMs in languages other than English.
# 1 Introduction
Large language models (LLMs) have recently achieved remarkable progress in performance and generalization. Specifically, Transformer-based LLMs such as BERT [3] and the GPT series [17, 18, 1] have demonstrated high-performance thanks to their pre-training. Furthermore, models that have evolved from these, such as ChatGPT [14] and GPT4 [15], have gained popularity for their remarkable performance. Other models such as Bard [6], LLaMA [24], Dolly [2], Bloom [21], and Vicuna [26] have also emerged.
Some of those models are already provided to consumers as a web service. Moreover, via API, those models and services are also now available for sub-parts of web services, and many spin-off services are emerging. However, despite the prosperity of language models, there are still challenges in handling diverse prompts, including prompts written in languages other than English. For example, Alpaca [23] dataset has been proposed due to the incompleteness of LLaMA’s response. However, the dataset of Alpaca is only available in English, and the incompleteness pointed out by Alpaca has not been filled yet in the other languages. Moreover, LLaMA has difficultness to respond appropriately to some prompts in languages other than English. Considering these challenges, it is necessary to enhance models’ performances in languages other than English. However, it is not a good idea to study a specific model in terms of performance improvements in the other language. Moreover, model development is still ongoing and very competitive, and the situation is changing dramatically recently. It is also easy to assume that newer models with better performance will emerge in a few months or even 1-2 months. Therefore, enhancing datasets that support model training may be more useful than focusing on specific models. This approach may also lower the barrier to adapting new models to languages other than English. Therefore, this study constructed a new chat dataset in Japanese for LLM training, which contains approximately 8.4 million data points, and demonstrated the performance of the dataset qualitatively. The dataset and trained models are opensourced and publicly available. The details are as follows:
• Dataset: https://huggingface.co/datasets/izumi-lab/llm-japanese dataset • Trained Models (LLaMA 1 epoch): https://huggingface.co/izumi-lab/llama-13b-japanese-lorav0-1ep
The more details are explained in the following. Moreover, data expansion and additional model training are planned as future tasks.
# 2 Dataset Construction: izumi-lab/llm-japanese-dataset v0
In this study, we created a Japanese chat dataset. The dataset1 contains 8,393,726 data points. In the following, we describe the details of datasets and their creation
In this study, we created a Japanese chat dataset. The dataset1 contains 8,393,726 data points. In the following, we describe the details of datasets and their creation process. The dataset is publicly available at the following URLs:
• Hugging Face: https://huggingface.co/datasets/izumi-lab/llmjapanese-dataset • GitHub: https://github.com/masanorihirano/llm-japanese-dataset 1 In this paper, we mentioned v0, specifically v0.1.0. There is a possibility of future updates.
• Hugging Face: https://huggingface.co/datasets/izumi-lab/llmjapanese-dataset • GitHub: https://github.com/masanorihirano/llm-japanese-dataset
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/07c3/07c3c7ee-a5f8-46eb-a53a-075073cc427f.png" style="width: 50%;"></div>
# 2.1 Translation Task
In many LLM models, while the performance in English is good, the support for languages other than English, such as Japanese, is often immature. In such cases, if it were possible to link knowledge in English with other languages, such as Japanese, it would be possible to transfer the knowledge learned in English to multilingual settings. Therefore, we set up translation tasks from Japanese to English and from English to Japanese.
Translation Task Example 1
### Instruction:
Please translate the following Japanese into English.1
### Input:
He doesn’t see his family in his busy life.1
### Response:
He doesn’t see his family in his busy life.
Translation Task Example 2
### Instruction:
Please translate the title of the following English paper into Japanese. 1
### Input:
Odd viscosity in chiral active fluids
### Response:
Odd viscosity in chiral active fluids1
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8fb5/8fb51d83-08c3-4cb8-b81d-593019cdda18.png" style="width: 50%;"></div>
# 2.2 Knowledge-based Tasks
Next, to appropriately learn knowledge in Japanese, knowledge-based tasks were incorporated into the training dataset. In the following, we explain the datasets used for creating our dataset and the created dataset examples.
• Wikipedia8: By using the titles and first paragraphs of Wikipedia, the task of explaining words was created. The license is CC BY-SA 3.0. The dataset size created is 1,347,381. • Japanese WordNet [8]9: By using the word and definition parts, a chat was created that requests word explanations. The license is a BSD-like license unique to it. The dataset size created is 173,085. • AIO (AI King) Official Distribution Dataset Version 2.010: Since it is a quiz format dataset, it was incorporated directly as a chat dataset. The license is CC BY-SA 4.0. The dataset size created is 23,335. • Japanese Movie Recommendation Dialogue (JMRD) [10] 11: Only the metadata related to movies was extracted, and a chat dataset was created that shows movie descriptions and summaries. The license is CC BY-SA 4.0. The dataset size created is 260. • JCommonsenseQA [11]12: A Q&A dataset that questions common sense, which is part of the Japanese General Language Understanding Evaluation (JGLUE), was incorporated as its original form. The license is CC BY-SA 4.0. The dataset size created is 10,058. • Aozorra Paperback13: Only works of Japanese literature whose copyright has expired were selected from Aozora Paperback14, and questions were created that ask for the beginning of the sentence and the author and title from the beginning of the sentence. The dataset size created is 33,758.
<div style="text-align: center;">Combining them, the total number of datasets is 1,587,877. Knowledge-based Tasks: Wikipedia</div>
Knowledge-based Tasks: Wikipedia
### Instruction:
Please explain the input word.1
### Input:
The University of Tokyo1
### Response:
The University of Tokyo (Tokyo Daigaku) is a Japanese national university headquartered in
Bunkyo-ku, Tokyo. Its abbreviated name is Todai.1
Knowledge-based Tasks: Japanese WordNet
### Instruction:
What does ”possible” mean?1
### Response:
Have the methods, skills or know-how or rights necessary to do something1
Knowledge-based Tasks: AIO
### Instruction:
What bird is used to describe the personalities of three warlords, Nobunaga Oda, Hideyoshi
Toyotomi, and Ieyasu Tokugawa?1
### Response:
Cuculus poliocephalus1
Knowledge-based Tasks: JMRD (partially omitted)
### Instruction:
Tell us about My Neighbor Totoro.1
### Response:
My Neighbor Totoro is a movie produced in 1988 and directed by Hayao Miyazaki. Noriko
Hidaka and Chinatsu Sakamoto appear in the movie. The summary is as follows:
[Summary]
An elementary school student Satsuki and her younger sister Mei move with their father to a
farming village in early summer for their mother’s medical treatment. ...1
Knowledge-based Tasks: JCommonsenseQA
### Instruction:
Which of the following are primarily for children and contain illustrated stories?1
### Input:
World, photo book, illustrated book, research paper, picture book1
### Response:
Illustrated book1
Knowledge-based Tasks: Aozorra Paperback (1)
### Instruction:
What is the beginning of Ivan Turgenev’s work “First Love”?1
### Response:
It begins with “The guests had long since departed. The clock struck half-past twelve. There
remained in the room only the host, Sergy´ei Nikol´aevitch, and Vlad´ımir Petr´ovitch. The host
rang and ordered the remains of the supper to be removed.”1
Knowledge-based Tasks: Aozorra Paperback (2)
### Instruction:
Please tell us the author and title of the literary work that begins, “Once there were two
young gentlemen climbing up deep into the mountains. They dressed like British soldiers
and walking on the dry leaves carrying shining guns on their shoulders.”1
### Response:
“The Restaurant of Many Orders” written by Kenji Miyazawa.1
# 2.3 Other Tasks
# 2.3.1 Creating a Paper Title and Imagining an Abstract
We utilized the aforementioned ParaNatCom [25] to create tasks related to our research paper. The license for the dataset is CC BY 4.0, and the size of the created dataset is 1,732.
We utilized the aforementioned ParaNatCom [25] to create tasks related to our re-
search paper. The license for the dataset is CC BY 4.0, and the size of the created
dataset is 1,732.
Task Example 1 (partially omitted)
### Instruction:
Please make a title from the abstract of the paper. 1
### Input:
Superthin nanostructures, particularly with atomic-level thicknesses, typically display
unique optical properties because of their exceptional light-matter interactions. Here, we
report a facile strategy for the synthesis of sulfur-doped molybdenum oxide nanorings ...1
### Response:
Atomic-level
molybdenum
oxide
nanorings
with
full-spectrum
absorption
and
photoresponsive properties1
Task Example 2 (partially omitted)
### Instruction:
Imagine and write an abstract from the paper’s title.1
### Input:
Surface plasmon induced direct detection of long wavelength photons1
### Response:
Millimeter and terahertz wave photodetectors have long been of great interest due to a wide
range of applications, but they still face challenges in detection performance. Here, we
propose a new strategy for the direct detection of millimeter and terahertz wave photons
based on localized surface-plasmon-polariton (SPP)-induced non-equilibrium ...1
Task Example 1 (partially omitted)
### Instruction:
Please make a title from the abstract of the paper. 1
### Input:
Superthin nanostructures, particularly with atomic-level thicknesses, typically display
unique optical properties because of their exceptional light-matter interactions. Here, we
report a facile strategy for the synthesis of sulfur-doped molybdenum oxide nanorings ...1
### Response:
Atomic-level
molybdenum
oxide
nanorings
with
full-spectrum
absorption
and
photoresponsive properties1
photoresponsive properties
Task Example 2 (partially omitted)
### Instruction:
Imagine and write an abstract from the paper’s title.1
### Input:
Surface plasmon induced direct detection of long wavelength photons1
### Response:
Millimeter and terahertz wave photodetectors have long been of great interest due to a wide
range of applications, but they still face challenges in detection performance. Here, we
propose a new strategy for the direct detection of millimeter and terahertz wave photons
based on localized surface-plasmon-polariton (SPP)-induced non-equilibrium ...1
# 2.3.2 News Title Generation
We retrieved articles from the Japanese version of Wikinews15, and set the task of generating a title from the article body. The license is CC BY 2.5. The size of the created dataset is 4,265.
Task Example (partially omitted)
### Instruction:
Please write a title for the following news article.1
### Input:
On January 11, Iwanami Shoten released the 6th edition of Kojien, the first major revision of
the Kojien dictionary in 10 years. The sixth edition will be the first revision in 10 years ...1
### Response:
Kojien is revised for the first time in 10 years - Sixth edition released. 1
# 2.3.3 Answer Extraction Task
In addition to knowledge and translation tasks, we also adopted a task of extracting answers from the given context. As the source data, we used JSQuAD [11]16, which is part of JGLUE tasks. The license is CC BY-SA 4.0, and the dataset size created is 67,491.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/6a13/6a13ebd5-240d-476b-853e-0bf8823ea63c.png" style="width: 50%;"></div>
# 2.3.4 Easy Japanese Task
We incorporated the task of rewriting normal Japanese into simpler Japanese. We processed SNOW T15: Japanese Simplified Corpus with Core Vocabulary [13] and SNOW T23: Crowdsourced Corpus of Sentence Simplification with Core Vocabulary [9] as the original data. The license is CC BY 4.0. The size of the created dataset is 84,300.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7bb7/7bb75488-e7b8-49f3-92d9-676c014a23f3.png" style="width: 50%;"></div>
# 2.3.5 Japanese Translation of Existing Chat Dataset
In addition, we incorporated Japanese-translated versions of existing publicly available chat datasets. The following datasets were included:
In addition, we incorporated Japanese-translated versions of existing publicly available chat datasets. The following datasets were included: • Japanese-Alpaca-LoRA17: A translation of the Alpaca[23] dataset into Japanese. The license is Apache License 2.0. The dataset size is 52,002. • databricks-dolly-15k-ja18: A Japanese-translated version of the dataset used for training Dolly[2]. The license is CC BY-SA 3.0. The dataset size is 15,015.
• Japanese-Alpaca-LoRA17: A translation of the Alpaca[23] dataset into Japanese. The license is Apache License 2.0. The dataset size is 52,002. • databricks-dolly-15k-ja18: A Japanese-translated version of the dataset used for training Dolly[2]. The license is CC BY-SA 3.0. The dataset size is 15,015.
# 3 Model Construction and Evaluation Using Our Dataset
This study used LoRA [7] as a method to fine-tune LLMs without significant performance degradations. It is because building LLMs from scratch requires a massive amount of computational resources. Furthermore, LLMs with a large number of parameters require GPU resources not only for pre-training but also for fine-tuning. On the other hand, LoRA updates only small parts of LLM parameters. Therefore, LoRA is a feasible option for us to evaluate the benefits of our dataset. The main parameters used in the experiment are shown below.
• Base model: LLaMA 13B [24] • Learning rate: 3e-4 • Input length: 256 tokens • Batch size: 130 • Epochs: 1 • r of LoRA: 4 • α of LoRA: 16 • Dropout rate of LoRA: 0.05 • Tuning parameters in LoRA: q proj, v proj
We used PEFT[12] and DeepSpeed ZeRO 2[19] for the implementation. This tuned model is publicly available at https://huggingface.co/izum lab/llama-13b-japanese-lora-v0-1ep. In order to increase the reproducibility of the evaluation experiment, the temperature parameter for prompt generation was set to 0.0. Below are some qualitative comparisons we conducted to assess performance.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/14dc/14dcab65-7c08-448b-b299-b16f4457524e.png" style="width: 50%;"></div>
10
Masanori HIRANO, Masahiro SUZUKI, and Hiroki SAKAJI
The University of Tokyo is located at 1-1-1 Bunkyo, Bunkyo-ku, Tokyo.1
(Authors’ note: The correct address is 7-3-1 Hongo, Bunkyo-ku, Tokyo.)
Response Example (4)
### Input:
A phone rang. What should I do?1
### Output(LLaMA+LoRA):
To hang up the phone.1
### Output(LLaMA):
The phone rings. When the call is received, the person receiving the call should receive the
call.1
Response Example (5)
### Input:
What are the three major festivals in Kyoto?1
### Output(LLaMA+LoRA):
The three major festivals inKyoto are the spring festival, the summer festival, and the autumn
festival.1
(Authors’ note: Correct answer is Aoi, Gion, Jidai festivals.)
### Output(LLaMA):
What are the three major festivals in Kyoto?
What are the three major festivals in Kyoto?
What are the three major festivals in Kyoto?1
<div style="text-align: center;">The University of Tokyo is located at 1-1-1 Bunkyo, Bunkyo-ku, Tokyo.1 (Authors’ note: The correct address is 7-3-1 Hongo, Bunkyo-ku, Tokyo.)</div>
# 4 Discussion
In this study, we constructed a dataset, izumi-lab/llm-japanese-dataset v0. The dataset consists of approximately 8.4 million data points, mainly focusing on Japanese-English translation tasks. The purpose of this dataset was to improve the performance of Japanese language processing capability and connect English language processing capabilities with Japanese input and output. Therefore, we tested additional LoRA tuning to the LLM using our datasets to evaluate its benefits. Reviewing the actual prompt responses after LoRA tuning, there appears to be a slight improvement in performance. However, since we only trained LLaMA with one epoch of LoRA tuning, the difference may be small. Therefore, increasing the number of epochs could be a future work. Nevertheless, the results suggest that our new dataset is effective. For example, the responses of our model (LLaMA + LoRA) in examples (4) and (5) were incorrect, but LoRA tuning seems to achieve generating more appropriate Japanese outputs. Additionally, although the responses of our model (LLaMA + LoRA) in examples (2) and (3) were still somewhat unfluent, they have achieved generated correct responses in terms of facts. However, this qualitative comparison is not fully appropriate, and establishing a comparison method is also a possible future work. To evaluate more accurately, it is necessary to construct evaluation tasks and datasets specialized for Japanese,
and there may be a need for a Japanese version of the Language Model Evaluation Harness [5] evaluation scheme.
# 5 Conclusion
In this study, we constructed a Japanese chat dataset for training LLMs on chat responses. The dataset can be used for tuning chat tasks to LLMs created mainly in English, and we qualitatively confirmed the performance of improvement in the LLM tuned by our dataset and LoRA tuning. Our dataset is now publicly available. Through the construction of the model to check the performance of the dataset, we have also clarified some issues in the construction of LLMs for languages other than English. Such a trial to adapt LLMs built mainly for English into other languages is now essential to make LLMs available fluently in wider languages.
Acknowledgements This work was supported in part by JSPS KAKENHI Grant Number JP21K12010 and JST PRESTO Grant Number JPMJPR2267.
# References
