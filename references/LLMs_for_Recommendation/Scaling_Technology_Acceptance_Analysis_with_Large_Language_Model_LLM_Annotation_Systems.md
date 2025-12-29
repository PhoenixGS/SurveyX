# Scaling Technology Acceptance Analysis with Large Language Model (LLM) Annotation Systems
Pawel Robert Smolinski∗1, Joseph Januszewicz2, and Jacek Winiarski1
1University of Gdansk, Faculty of Economics, Sopot, Poland 2Geisel School of Medicine at Dartmouth, Hanover, New Hampshire, United States
# Abstract
Technology acceptance models effectively predict how users will adopt new technology products. Traditional surveys, often expensive and cumbersome, are commonly used for this assessment. As an alternative to surveys, we explore the use of large language models for annotating online user-generated content, like digital reviews and comments. Our research involved designing an LLM annotation system that transforms reviews into structured data based on the Unified Theory of Acceptance and Use of Technology model. We conducted two studies to validate the consistency and accuracy of the annotations. Results showed moderate-to-strong consistency of LLM annotation systems, improving further by lowering the model temperature. LLM annotations achieved close agreement with human expert annotations and outperformed the agreement between experts for UTAUT variables. These results suggest that LLMs can be an effective tool for analyzing user sentiment, offering a practical alternative to traditional survey methods and enabling deeper insights into technology design and adoption.
Keywords: Large Language Models (LLM), Technology Acceptance Models (TAM), Auto mated Annotation Systems, Natural Language Processing (NLP), GPT Models
# 1 Introduction
Technology acceptance is a recognized concept within the field of information science [2, 3]. It refers to the degree to which individuals perceive a new technology as useful and easy to use, and decide to use it [2]. Understanding technology acceptance is crucial for developers,
∗Corresponding author: pawel.smolinski@phdstud.ug.edu.pl 0This is a preprint of a paper accepted for the 32nd International Conference on Information Systems Development (ISD 2024), 26–28/08/2024, Gda´nsk, Poland
businesses [4], and policymakers [5, 6] to predict how new technologies will be adopted by target users. To translate this understanding into actionable insights, researchers have developed technology acceptance models. Technology acceptance models facilitate the collection of data for refining technology design, assessing and improving organizational readiness for new IT solutions, and conducting SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis of emerging technology products in market research. Technology acceptance data is typically collected through surveys and questionnaires that assess respondents’ perceptions of the usefulness, ease of use, social influence, and facilitating conditions related to the technology. These instruments are often based on the constructs defined in the TAM model and its variations (e.g. UTAUT). Responses are analyzed to predict attitudes toward the technology, intention to use it, and sometimes actual usage behavior. The results of such surveys provide valuable data for designing [7], implementing [8], and marketing [9] new technologies to improve their acceptance and adoption. However, designing questionnaires to measure technology acceptance can present many challenges, often requiring substantial effort and resources. A primary obstacle is the lack of standardization across technology acceptance questionnaires [10, 11]. Due to questionnaire heterogeneity, items researchers choose to represent key acceptance variables, such as Effort Expectancy, can differ widely from one study to another [12]. Moreover, collecting highquality data on users’ attitudes through questionnaires is a cumbersome process. It involves defining the target demographic for a given technological product, successfully reaching this group, and obtaining a sufficiently large sample size to ensure the reliability and validity of the TAM model. These steps are crucial for the accurate assessment of technology acceptance but can be challenging to execute in practice. The questionnaires themselves also possess general limitations that restrict their utility in technology acceptance research. For example, respondents may provide socially desirable answers rather than truthful responses. Furthermore, participants might interpret the questions differently and questionnaire items often fail to capture the complexity of human attitudes, leading to inconsistent data [13]. These challenges in questionnaire design, data collection, and accurate attitude inferences have prompted a move within technology acceptance research towards leveraging more natural or ecologically valid data sources for inferring attitudes [14]. Ecologically valid data refers to information gathered from real-world settings or contexts that resemble the actual environment in which the phenomenon of interest naturally occurs. In the case of technology acceptance research, ecologically valid data sources include user-generated content such as social media posts, product reviews, and forum discussions, which reflect users’ genuine attitudes and experiences with technologies in their everyday lives. The analysis of ecologically valid data has been accelerated by the development of Natural Language Processing (NLP) methods. NLP methods, such as sentiment analysis and Latent Dirichlet Allocation [15], enable computers to understand, interpret, and generate human language in meaningful ways. By applying NLP, researchers can analyze vast amounts of unstructured textual content— such as social media posts, product reviews, and forum discussions—to extract data about users’ attitudes toward technologies without several limitations of traditional survey methods [16]. For example, Liu Zhuchenyang (2022) was the first to use NLP to extract data from app store reviews in order to validate a Technology Acceptance Model (TAM) [17]. Their work demonstrated the potential of incorporating NLP-driven data into information
systems and technology acceptance research. In this paper, we aim to address the challenges associated with survey-based data collection in information system research by exploring the potential of Large Language Model (LLM) annotation systems as a consistent and scalable means of capturing user attitudes towards technology. Our research objectives include designing an LLM annotation system that converts unstructured user reviews into structured annotations and validating its consistency and accuracy. Through these objectives, we aim to demonstrate the potential of LLM annotation systems in replacing traditional questionnaire-based surveys and expert evaluations in technology and information science research.
# 2 Large Language Model (LLM) annotation systems
# 2 Large Language Model (LLM)
LLM annotation systems represent an emerging innovation in the field of Natural Language Processing [18, 19]. These systems leverage the capabilities of Large Language Models to convert textual data into a numerical format that can then be used for further statistical analysis [17, 20]. For instance, researchers interested in understanding how customer attitudes impact product market performance might traditionally rely on expensive market surveys, expert consultations, or focus groups. Now with increased proliferation of online data, researchers have the option to extract attitudes from customer reviews found on various forums and websites like Amazon. However, these textual reviews are initially in a text format and must be converted into a numerical format to be analyzable in statistical models, such as regression models. NLP annotation mechanisms serve this purpose by translating text into sentiment scores in the case of sentiment analysis and into any desired numerical output in the case of LLM annotation systems. Although LLM annotation systems are relatively new and thus not extensively tested, there is a growing body of research exploring their potential [21, 22]. Several studies have already shown the promise of GPT models in classifying unseen data without prior training on specific tasks, such as detecting hate speech [23], identifying misinformation [1], assessing the credibility of news sources [24], and medical reports [25]. A handful of researchers have also begun to validate the effectiveness of LLM annotations in accurately inferring attitudes from textual data, showing promising results [18, 24]. Despite these initial successes, the overarching consensus is that further research is needed to fully understand and optimize the use of LLM annotation systems in attitude analysis and beyond [26, 22].
# 3 Study objectives
In this paper, we aim to address the challenges associated with survey-based data collection in information system research. We recognize the potential of Natural Language Processing (NLP) to transform how we gather and analyze data and introduce LLM annotation systems for converting unstructured text data, specifically customer reviews, into analytically useful numeric format. Our research is driven by the goal of exploring the potential of LLM annotation systems as a consistent and scalable means of capturing user attitudes towards technology.
1. Designing an information system: The initial phase involves designing an LLM annotation system that converts unstructured user reviews into structured annotation reflecting user attitudes based on established technology acceptance models.
2. Consistency validation: This involves testing whether the LLM annotation system produces consistent results across multiple runs and across different technology acceptance variables. We aim to demonstrate that our LLM-based system can serve as a stable and dependable tool for data annotation.
3. Accuracy validation against human experts: The final step focuses on evaluating the accuracy of the LLM annotations in comparison to evaluations provided by human experts. Our hypothesis is that expert annotations will concur with those generated by the LLM system.
Through these objectives, our study aims to show that LLM annotation systems can replace traditional questionnaire-based surveys and expert evaluations in technology and information science research. By using ecologically valid data, we want to demonstrate the opportunity for more scalable methods of inferring user attitudes towards technology.
# 4 Methods
# 4.1 Designing an Information System for annotating customer r views with LLMs
# 4.1 Designing an Information System for annotating customer re-
We designed a simple LLM annotation system to evaluate customer reviews using Large Language Models (LLMs) and report their attitudes towards a product in a standardized Technology Acceptance Model (TAM) questionnaire format. Our system operates as follows: Initially, we input a custom prompt that instructs LLMs on annotating reviews based on a specified acceptance model. This prompt also details the required attitude scale (e.g. Likert scale) and the desired output format. In our case, the prompt given to the LLM was as follows:
Your task is to evaluate customer reviews of products based on specific variables that influence technology acceptance. You will use a 5-point Likert scale to rate each variable in the review. Variables: • Performance expectancy, • Effort expectancy, • Social influence, • Facilitating conditions Likert Scale:
• 1: The review clearly expresses a negative perception of the factor. • 2: The review suggests some negative aspects regarding the factor. • 3: The review does not lean clearly towards a positive or negative perception. • 4: The review suggests a positive perception of the factor. • 5: The review clearly expresses a positive perception of the factor.
If the review does not provide enough information to assess this factor, or the factor is irrelevant to the context of the review, will assign 0 to the respective variable.
If the review does not provide enough information to assess this factor, or the factor is irrelevant to the context of the review, will assign 0 to the respective
This prompt is adaptable to meet researchers’ needs. For instance, different variables can be introduced or even defined by the researcher within the prompt. A varied scale (e.g., a 7-point attitude scale) can be utilized, or different category ratings can be included or excluded (e.g., removing the ”no information” rating). However, prompt designers must be cautious because the output from the LLM will depend on the prompt given. Although previous research has demonstrated high consistency of results regardless of the prompt wording when it comes to annotations, we maintain that the quality of the prompt is crucial in obtaining the highest quality results [20]. Upon specifying the prompt, the system is initiated by receiving a list of customer reviews. For this study, we chose 15 refurbished iPhone 13 reviews from the Amazon website, selecting them for their relevance to technology and the extensive user feedback available on the website. Each review is individually processed to extract annotations related to the TAM variables specified in the prompt. This extraction process involves feeding the prompt into the model and parsing the model’s response. The parsed responses are standardized into the desired data format. We utilize regular expression syntax to extract this information from the LLM’s output and organize it into a data frame. The final result is a well-organized data frame with TAM variables as columns, reviews as rows, and the LLM annotations as elements. Using a GPT-4 LLM model to generate annotations for 15 selected Amazon iPhone 13 reviews, we obtained the following results in a single run:
Review:
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
Performance expectancy
2
3
1
1
5
1
5
2
4
2
3
2
3
1
1
Effort expectancy
4
4
1
4
5
1
5
3
2
3
2
1
2
3
1
Social influence
0
0
0
0
0
0
0
0
0
0
0
0
0
1
0
Facilitating conditions
3
2
1
2
5
1
4
1
2
1
1
2
2
1
1
The workflow generates evaluations as expected, annotating the Likert scale ratings where it thinks the attitude can be annotated and assigning 0 (no information category) where the model thinks that the review does not provide enough information to infer the user’s attitude. This is most prominent in the Social Influence variables, where the LLM did not provide attitude ratings for all but one review (review 14).
# 4.2 Consistency measure
We employed the weighted percentage agreement (WPA) to measure the level of consistency between different sets of annotations. This included comparisons between two LLMs annotations, the same LLM across different iterations (Study 1) and LLM and expert annotations (Study 2). WPA calculates the proportion of instances where the annotations matched (agreement), while also taking into account the severity of disagreement. Specifically, smaller disagreements are penalized less than larger disagreements. WPA can take values from 0 to 1 with closer to 1 meaning higher agreement. The equation for weighted percentage agreement is as follows:
Where:
• n is the total number of items (reviews) being compared.
• w(r1i, r2i) represents the penalty weight assigned to the disagreement between the tw sets of annotations on the i-th item.
• wmax is the maximum weight, representing the highest possible penalty for disagreemen within the custom weights matrix.
The sum �n i=1 w(r1i, r2i) calculates the total weighted disagreement over all items and the denominator n • wmax is the maximum weighted disagreement. The WPA metric requires a weight matrix that contains all possible penalties for the severity of disagreement. We opted for the following penalty matrix:
In this matrix, a penalty of 0 denotes complete agreement. Penalties increase in quadratic difference from 1 to 16 for disagreements among annotations 1 through 5. For example, a complete disagreement—annotating a review as clearly negative (1) versus clearly positive (5)—incurs the highest penalty of 16, while disagreements between adjacent annotations (e.g., 1 and 2) receive a minimal penalty of 1. Annotation of 0 is uniquely treated as a ”no information” category and is assigned specific penalties: [0, 16, 9, 4, 9, 16]. Disagreements between a 0 annotation and the extremes (ratings 1 and 5) are penalized most heavily (16), indicating a substantial gap in evaluation between annotating no information about the factor in a review and annotating a clear positive or negative review. Conversely, a comparison of 0 with a neutral annotation (3) attracts the lowest penalty (4), suggesting a somewhat similar lack of decisive positioning, although ”neither positive nor negative” is still not the same as ”no information about the
(1)
factor”. The penalties for comparisons between 0 and ratings 2 and 4 (9) are moderate, acknowledging that while there is a discernible gap in evaluations, it is less severe than the extreme annotations of 1 or 5. We determined that other popular metrics of agreement such as Cohen’s Kappa, Fleiss’ Kappa, and the Intraclass Correlation Coefficient (ICC) were not appropriate for our dataset. Our reason stems from the composition of our data, which blends categorical and ordinal ratings (categorical ”No information” rating annotated as 0, and the remaining ratings following a 1 to 5 Likert scale format). Additionally, we observed that some LLM generations (and experts) provided annotations with limited variability and that this could affect the results of more complex agreement metrics, as they require heterogeneous distributions.
# 5 Study 1: Consistency validation
In the first study, we evaluate run-to-run consistency of our workflow, which refers to whether each execution of the described information system produces consistent annotations for the same reviews. Should independent runs yield different annotations, the system’s utility would be undermined. It would suggest that the LLM lacks a genuine understanding of the attitudes in the reviews and generates ratings arbitrarily with each execution. To assess run-to-run consistency, we executed our workflow 50 times on the same set of reviews using two different LLM models (GPT-3.5 and GPT-4). We then calculated the weighted percentage agreement (WPA) for every possible pair of runs (e.g., run 1 with run 2, run 1 with run 3, run 2 with run 3, etc.). The weighted percentage agreement quantifies the proportion of similar annotations across two runs, incorporating a penalty for the severity of disagreement. This process was repeated for each TAM variable. Subsequently, we averaged all possible pairwise agreements to determine an average agreement score for each variable, thereby deriving the consistency ratings across 50 workflow runs.
# 5.1 Results
The assessment of run-to-run consistency yielded the following average weighted percentage agreement (WPA) scores for GPT-3.5: � WPAGP T−3.5 (P E) = 0.76 for Performance Expectancy, � WPAGP T−3.5 (EE) = 0.86 for Effort Expectancy, � WPAGP T−3.5 (SI) = 0.68 for Social Influence, and � WPAGP T−3.5 (F C) = 0.66 for Facilitating Conditions. For GPT4, the average WPA scores are: � WPAGP T−4 (P E) = 0.74 for Performance Expectancy, � WPAGP T−4 (EE) = 0.72 for Effort Expectancy, � WPAGP T−4 (SI) = 0.86 for Social Influence, and � WPAGP T−4 (F C) = 0.76 for Facilitating Conditions. According to McHugh (2012), these values indicate moderate to strong levels of consistency, where moderate is greater than 0.6 and strong exceeds 0.8 [27]. Such results suggest that while the annotations are generally consistent, there are instances where runs produce divergent annotations. To further analyze these instances of divergence, we compiled Table 1 and Table 2, which present four key statistics for each TAM variable and review: the mode annotation, the range of annotations, the proportion of cases in which the LLM annotated the mode, and the proportion of cases in which the LLM annotated within proximity of the mode (annotation
± 1 from the mode). The methodology for calculating these statistics is detailed in th Supplementary Materials.
<div style="text-align: center;"> 1 2 2 1 1 1 2 0 2  .9  .7 1  .6 1 1 1 1</div>
Annotation Consistency across LLM Runs (GPT-3.5)
1
2
3
4
5
6
7
8
9
10 11 12 13
4
4
1
2
5
1
4
2
4
1
3
2
3
3
3
1
2
4
1
4
3
2
1
2
2
3
3
0
0
0
4
0
5
0
2
1
2
2
0
0
3
1
1
4
1
4
0
2
1
2
2
0
1
0
1
1
0
0
1
2
0
1
2
1
1
1
1
1
1
0
0
1
1
0
2
1
1
2
3
3
1
4
4
1
5
3
3
3
3
3
3
4
4
0
1
1
0
2
4
1
1
1
3
3
.58
1.
.90 .74
1.
1.
.70 .66
1.
.86 .56 .94 .9
.82 .96 .90 .80
1.
1.
.96 .72
1.
.80 .84 .76 .5
.56 .66 .86 .54 .58 .98 .56 .90 .76 .94 .88 .50 .8
.38 .42
1.
.82 .82
1.
.76 .40 .98 .96 .80 .40 .8
1.
1.
1.
1.
1.
1.
1.
.96
1.
1.
1.
1.
1
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1
.56 .66
1.
.80 .80
1.
.86 .90 .76 .98 .88 .56 .8
.40 .78
1.
1.
1.
1.
1.
.72
1.
1.
1.
.94 .9
<div style="text-align: center;"> 14 1 2 1 1 1 2 1 1 .88 .48 .96 .88 1. 1. 1. 1.</div>
 Annotation Consistency across LLM Runs (GPT-4)
1
2
3
4
5
6
7
8
9
10 11 12 13
3
4
1
2
5
1
5
2
4
2
2
2
3
4
3
1
3
5
1
5
3
2
3
4
1
2
0
0
0
0
0
0
0
0
0
0
0
0
0
4
4
1
1
5
1
5
1
2
1
1
1
2
2
1
0
1
0
0
0
3
1
1
2
1
2
4
2
0
3
1
1
1
4
0
3
3
1
4
0
0
0
0
0
1
0
0
0
0
0
3
0
3
2
1
1
1
0
1
2
2
1
1
2
2
50 .74
1.
.90
1.
1.
1.
.90 .62 .82 .92 .98 .78
54 .56
1.
.48 .98 .96 .70 .56
1.
.48 .52 .74 .34
1.
1.
1.
1.
1.
.98
1.
1.
1.
1.
1.
.98
1.
40 .46 .98 .80 .66
1.
.68 .64 .88 .84 .76 .46 .84
1.
1.
1.
1.
1.
1.
1.
.98
1.
1.
.98
1.
1.
80 .86
1.
.80
1.
1.
1.
.90
1.
.48 .84
1.
.70
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
.98
1.
68 .78
1.
1.
1.
1.
1.
.92
1.
1.
1.
.84
1.
Key statistics from Table 1 and Table 1 that help identify potential divergences include the range and the proportion of annotations close to the mode. Ideally, annotations should not have a range exceeding 2. A range of 2 or less suggests a desired, narrow clustering of annotations. However, the range metric alone does not account for the frequency of deviations; a single outlier can easily push the range above 2. The proportion of annotations in proximity of the mode considers the percentage of annotations within a 2-unit range of the mode, providing a more nuanced view. Even in cases where a single outlier expands the range above 2, if the majority of annotations cluster around the mode (within this 2unit proximity), then this metric will remain high. We argue that values above 0.9 reflect consistent annotations, whereas values below 0.8 suggest inconsistency. Our results indicate that achieving perfect consistency in LLM annotations across multiple runs, variables, and stimuli (reviews) is challenging. No review (stimulus) has shown perfect consistency across runs. Only review 15 came closest to achieving near-perfect consistency in annotations, both for GPT-3.5 and GPT-4. In a comparative analysis, the GPT-4 model shows a slightly more consistent performance than GPT-3.5, indicated by less variability in the range and a higher proportion of annotations near the mode. However, this advantage is small and not consistent across all variables. GPT-4 performs better in annotating Social Influence but slightly worse in Effort Expectancy compared to GPT-3.5. Analyzing the weighted proportion agreement (WPA) between the mode annotations of GPT-3.5 and GPT-4 for all variables, we find: � WPABetween LLMs (P E) = 0.98 for Performance Expectancy, � WPABetween LLMs (EE) = 0.94 for Effort Expectancy, � WPABetween LLMs (SI) = 0.50 for Social Influence and � WPABetween LLMs (P E) = 0.81 for Facilitating Conditions. These findings indicate that for core technology acceptance variables like Performance Expectancy and Effort Expectancy, GPT-3.5 and GPT-4 produce remarkably similar annotations (the mode). The annotations for Facilitating Conditions also show considerable consistency. In contrast, annotations for Social Influence reveal some inconsistency between the models.
# 5.2 Improving consistency with model temperature
Previous research on LLM annotation systems has shown that the consistency of these models can be improved by adjusting the model temperature [22]. Model temperature refers to a hyperparameter that controls the diversity of the output generated by the model. A standard setting for many LLM models is a temperature of 1. This is considered the ”default” or ”neutral” setting, where the model generates text that balances randomness and predictability. Lowering the model temperature increases the model’s confidence in its predictions, leading to outputs that are less random and more predictable. However, setting the temperature too low can result in the model producing text that is repetitive or overly cautious. While a very low temperature is generally avoided in text generation due to its limiting effect on model performance, it could be beneficial for annotation systems where consistency is key. Although a temperature of 0 may be too low, potentially compromising the quality of model annotations due to a lack of randomness. Randomness in LLM outputs, introduced by higher temperature settings, allows the model to explore different possibilities and generate more diverse responses. This diversity can lead to improved reasoning and output quality, as the
model considers multiple perspectives and avoids being overly deterministic. However, in the case of annotation systems, where consistency is a primary goal, reducing randomness by lowering the temperature can be advantageous. In this paper, we explore a temperature setting of 0.25 to assess its impact on improving annotation consistency and to see whether it influences the mode of annotations. Table 3 displays the outcomes for an annotation system using the GPT-3.5 model at a reduced temperature setting.
<div style="text-align: center;">atu 15 1 1 0 1 0 0 0 0 1. 1. 1. 1. 1. 1. 1. 1.</div>
nsistency with Reduced Temperature (GPT-3.5, Tempe
2
3
4
5
6
7
8
9
10 11 12
13 14 
4
1
2
5
1
4
2
4
1
3
2
3
2
3
1
2
4
1
4
3
2
1
2
2
3
2
0
0
0
4
0
5
0
3
0
0
3
0
1
3
1
1
4
1
4
4
2
1
2
2
0
0
0
0
1
0
0
1
0
0
1
2
1
1
0
0
0
0
0
0
0
0
0
2
1
1
1
1
3
1
3
4
0
1
0
3
0
0
3
0
0
4
0
1
1
0
1
4
0
1
1
3
2
2
1.
1.
.98
1.
1. .80
1.
1.
.94 .88 .98 .98
1.
1.
1.
1.
1.
1.
1.
1.
1.
.90 .98 .86 .68 .98
.70 .98 .44 .50 1. .82
1.
.88
1.
1.
.64
1.
1.
.62
1.
.74 .86 1. .98 .40
1.
.98 .92 .58 .98 .50
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
1.
.70
1.
.80 .84 1.
1.
1.
.88
1.
1.
.64
1.
1.
.86
1.
1.
1.
1.
1.
.40
1.
1.
1.
.98 .98 .98
The results show that decreasing the model’s temperature leads to improved overall consistency in annotations. Variables such as Performance Expectancy and Effort Expectancy were annotated with almost perfect consistency, while the annotations for two other variables also saw significant improvements. Nevertheless, we observed minor changes in the mode of annotations. For instance, with a temperature of 1, the GPT-3.5 model categorized review 8 under the variable Facilitating Conditions as 0 (indicating no information), whereas at a temperature of 0.25, the same model assessed it as 4 (reflecting a strong positive attitude towards Facilitating conditions). This outcome becomes less surprising upon examining the range and frequencies of ratings across both runs, which reveal that this particular review is quite difficult to annotate, producing a diverse range of ratings in both instances.
# 6 Study 2: Accuracy validation against human experts
In the second study, we compared annotations derived from LLMs versus from human experts. We enlisted three independent experts who hold PhD degrees in information systems research and have published works on Technology Acceptance Models (the experts were not involved in the authorship of this paper). We presented them with the prompt from part one and asked them to evaluate the same 15 Amazon iPhone 13 reviews. Each review was annotated based on four Unified Theory of Acceptance and Use of Technology (UTAUT) variables, using a Likert scale. We then compared their annotations to those generated by LLMs.
# 6.1 Results
Table 4 presents the agreement between the evaluations of three human experts and mode evaluations generated by the GPT-4 model (mode evaluations taken from Table 2). The average WPA across all pairs (experts and LLM) is highest for Performance Expectancy (� WPAP E = 0.87) and Effort Expectancy (� WPAEE = 0.73). The mean WPA scores for Social Influence and Facilitating Conditions are on the threshold of moderate and weak, revealing a significantly lower level of consensus when it comes to annotating these factors in reviews (� WPASI = 0.67 and � WPAF C = 0.60).
<div style="text-align: center;">4: Agreement between Human Experts and GPT-4 Ann</div>
Variable / Comparison
Agreement Score
Performance Expectancy (PE)
Expert 1 vs Expert 2
.6625
Expert 1 vs Expert 3
.9542
Expert 1 vs GPT-4
.9458
Expert 2 vs Expert 3
.6833
Expert 2 vs GPT-4
.6917
Expert 3 vs GPT-4
.9833
Effort Expectancy (EE)
Expert 1 vs Expert 2
.6625
Expert 1 vs Expert 3
.8250
Expert 1 vs GPT-4
.6667
Expert 2 vs Expert 3
.6625
Expert 2 vs GPT-4
.6542
Expert 3 vs GPT-4
.8750
Social Influence (SI)
Expert 1 vs Expert 2
.2917
Expert 1 vs Expert 3
.8208
Expert 1 vs GPT-4
.9333
Expert 2 vs Expert 3
.4208
Expert 2 vs GPT-4
.3250
Expert 3 vs GPT-4
.7542
Facilitating Conditions (FC)
Expert 1 vs Expert 2
.4583
Expert 1 vs Expert 3
.4750
Expert 1 vs GPT-4
.3083
Expert 2 vs Expert 3
.6750
Expert 2 vs GPT-4
.6500
Expert 3 vs GPT-4
.8417
We can categorize the average WPA for each variable into two groups: the mean agreement among human experts alone (comparing expert 1 with expert 2, 2 with 3, and 1 with 3) and the mean agreement between all human experts and the LLM model (evaluations of experts 1, 2, and 3 with GPT-4). This allows us to assess if there is a significant difference between the consensus among human experts and their agreement with LLMs evaluations. For Performance Expectancy, the mean agreement between human experts was � WPABetween Human Experts (P E) = 0.77 while their agreement with GPT-4 evaluations was slightly higher at � WPAHuman Experts with GP T−4 (P E) = 0.87. This minor difference suggests that, on average, the evaluations provided by the LLM closely match those of the human experts. A similar pattern is observed for Effort Expectancy, where the agreement between human experts and between humans and the AI does not significantly diverge (� WPABetween Human Experts (EE) = 0.72, � WPAHuman Experts with GP T−4 (EE) = 0.73). The
trend holds for Facilitating Conditions as well, with agreements of (� WPABetween Human Experts (F C 0.54, � WPAHuman Experts with GP T−4 (F C) = 0.60). However, a notable exception is found in the case of Social Influence, where the agreement between human experts and the AI (� WPAHuman Experts with GP T−4 (SI) = 0.67) significantly surpasses the agreement among human experts alone (� WPABetween Human Experts (SI) = 0.51). This implies that, on average, human experts align more closely with the evaluations provided by LLMs than they do with each other’s assessments.
# 7 Discussion
Technology acceptance is a useful measure that helps companies design and target their products effectively. While consumer data has become increasingly accessible, the opportunity for analysis of consumer trends was previously limited by the reliance on traditional surveys and human expert reviewers. The development of Natural Language Processing with Large Language Model (LLM) annotation systems has provided researchers and companies with a powerful new tool for measuring technology acceptance in a scalable and cost-effective manner. In this paper, we demonstrated the success of LLMs, specifically GPT models, in this field. We find that LLM’s reliability and agreement with human experts are largely sufficient for current analysis tasks. Furthermore, as shown by the agreement scores, LLMs can be as useful as human experts when it comes to annotating attitudes in customer data. In our study, LLMs performed exceptionally well, with agreement scores between the AI and human experts coming very close to or even surpassing the agreement between human experts themselves. For example, we found that for Performance Expectancy, Facilitating Conditions, and Social Influence, the agreement is higher between the experts and the LLM than between the experts themselves. For the remaining category, Effort Expectancy, the agreement scores are almost identical. Statistically, a higher agreement score of the LLMs and experts means that scores generated by the LLM are more closely clustered around the mean of the human reviewers than the human reviews themselves. Such a result suggests that LLMs can be reliably and accurately used to analyze and annotate consumer data. The use of LLM annotation systems offers several advantages over traditional survey methods, including cost-effectiveness, scalability, and the ability to leverage existing usergenerated data from various online sources. Of these advantages, the most significant is the nearly unlimited LLM capacity to analyze reviews. For example, our analysis was limited by the time constraints of human reviewers with 15 reviews an acceptable middle ground between statistical power and feasibility per human reviewers. While the comparatively lower number of reviews does limit our analysis, it also shows the increased/superior capabilities of LLM analysis compared to traditional methods. However, it is important to note several potential limitations and challenges associated with LLM annotation systems, such as the need for prompt engineering, the closed-source nature of many popular LLMs, and the tradeoff between consistency and flexibility. Our findings have significant implications for researchers and practitioners in the field of information systems and technology acceptance. LLM annotation systems could be in-
tegrated into research and product development processes, providing a valuable tool for understanding user attitudes and preferences, refining technology design, and conducting market research. Despite these promising results, further research and development in this area are needed. This includes exploring different LLM models, fine-tuning techniques, and evaluating the performance of LLM annotation systems across different domains and technology products. Additionally, ethical considerations and potential risks associated with the use of LLM annotation systems, such as ethical data scraping, ethical model selection, and the need for transparency and accountability of LLMs, should be carefully addressed. With improved tuning, such as modifying the temperature or training models precisely for annotation tasks, LLMs are likely to be widely used to analyze the significant amounts of online consumer data. Our study shows that such an approach will likely be successful and should be an area of interest and further research, enabling deeper insights into technology design and adoption.
# 8 Recommendations for LLM Annotation Systems Use
Our research has demonstrated that Large Language Model (LLM) annotation systems can be a reliable and consistent tool for inferring user attitudes from unstructured text data. However, we observed some instances of small divergences from the mode in single-run annotations. To address this issue and ensure the highest quality of annotations, we propose the following recommendations for researchers interested in implementing LLM annotation systems in their work. We advise researchers to consider the mode of LLM annotations across multiple runs as the true sentiment inferred by the LLM model. While single-run annotations may exhibit minor variations, the mode represents the most frequently occurring annotation and is likely to be the most accurate representation of the model’s understanding of the text. To obtain the mode annotation, we recommend executing at least 10, preferably more, runs of the same LLM algorithm on the data. By conducting multiple runs, researchers can mitigate the impact of randomness imposed by the model temperature and obtain a more stable and reliable annotation. As demonstrated in our study, reducing the temperature to lower values, such as 0.25, can improve consistency without diminishing the model’s reasoning capabilities. For example, when using the GPT-3.5 model with a temperature of 0.25, we observed that the proportion of annotations matching the mode increased to 100% for Performance Expectancy and Effort Expectancy across all reviews. In addition to the mode, other metrics such as annotation range and proportions of the mode or proximity to the mode can be used to investigate the correctness and consistency of annotations. We recommend that future researchers employ these metrics in their implementations of LLM annotation systems. By examining the range of annotations, researchers can identify instances where the LLM model produces divergent annotations, potentially indicating areas of uncertainty or ambiguity in the text. Similarly, calculating the proportion of annotations that match or are close to the mode can provide a measure of the system’s consistency and reliability. In conclusion, we strongly recommend that future researchers employing LLM annotation systems in their work adopt a multi-run approach, using the mode annotation as the
true sentiment inferred by the model. By combining this approach with an examination of additional metrics and appropriate temperature settings, researchers can ensure the highest quality and consistency of annotations, unlocking the full potential of LLM annotation systems in technology acceptance research and beyond. Supplementary Materials: Additional details on the methodology and full numerical results are available in the supplementary materials at https://osf.io/pk63g/.
# References
[1] E. Hoes, S. Altay, and J. Bermeo, “Leveraging ChatGPT for Efficient Fact-Checking,” 2023. https://doi.org/10.31234/osf.io/qnjkf [2] F. D. Davis, “Perceived usefulness, perceived ease of use, and user acceptance of information technology,” MIS quarterly, pp. 319–340, 1989. [3] V. Venkatesh, M. G. Morris, G. B. Davis, and F. D. Davis, “User acceptance of information technology: Toward a unified view,” MIS quarterly, pp. 425–478, 2003. [4] J. Y. Thong, “An integrated model of information systems adoption in small businesses,” Journal of management information systems, vol. 15, no. 4, pp. 187–214, 1999. [5] Y. Kim, W. Kim, and M. Kim, “An international comparative analysis of public acceptance of nuclear energy,” Energy Policy, vol. 66, pp. 475–483, 2014. [6] P. R. Smolinski, J. Januszewicz, B. Pawlowska, and J. Winiarski, “Nuclear Energy Acceptance in Poland: From Societal Attitudes to Effective Policy Strategies – Network Modeling Approach,” 2023. https://doi.org/10.48550/arXiv.2309.14869 [7] L. Diamond, M. Busch, V. Jilch, and M. Tscheligi, “Using technology acceptance models for product development: case study of a smart payment card,” in Proceedings of the 20th International Conference on Human-Computer Interaction with Mobile Devices and Services Adjunct, 2018, pp. 400–409. [8] J. Park, F. Gunn, Y. Lee, and S. Shim, “Consumer acceptance of a revolutionary technology-driven product: The role of adoption in the industrial design development,” Journal of Retailing and Consumer Services, vol. 26, pp. 115–124, 2015. [9] L. Robinson, G. W. Marshall, and M. B. Stamps, “An empirical investigation of technology acceptance in a field sales force setting,” Industrial Marketing Management, vol. 34, no. 4, pp. 407–415, 2005. 10] W. W. Chin, N. Johnson, and A. Schwarz, “A fast form approach to measuring technology acceptance and other constructs,” MIS Quarterly, pp. 687–703, 2008. 11] P. R. Smolinski, M. Ma´nkowska, B. Pawlowska, and J. Winiarski, “The Determinants of Technology Acceptance for Social Media Messaging Applications–Fixed-Effect Questionnaire Design,” in Information Systems. Springer, 2023, pp. 381–396.
# 1] E. Hoes, S. Altay, and J. Bermeo, “Leveraging ChatGPT for Efficient Fact-Checking,” 2023. https://doi.org/10.31234/osf.io/qnjkf
2] F. D. Davis, “Perceived usefulness, perceived ease of use, and user acceptance of information technology,” MIS quarterly, pp. 319–340, 1989.
3] V. Venkatesh, M. G. Morris, G. B. Davis, and F. D. Davis, “User acceptance of information technology: Toward a unified view,” MIS quarterly, pp. 425–478, 2003.
[12] J. R. Lewis, “Comparison of Four TAM Item Formats: Effect of Response Option Labels and Order-JUX,” JUX-The Journal of Usability Studies, vol. 14, no. 2, 2019. [13] D. Chan, “So why ask me? Are self-report data really that bad?,” in Statistical and methodological myths and urban legends. Routledge, 2008, pp. 325–346. [14] H. Barki and I. Benbasat, “Quo vadis, TAM?,” Journal of the association for information systems, vol. 8, no. 4, p. 7, 2007. [15] D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent dirichlet allocation,” Journal of machine Learning research, vol. 3, no. Jan, pp. 993–1022, 2003. [16] J. Grimmer and B. M. Stewart, “Text as data: The promise and pitfalls of automatic content analysis methods for political texts,” Political analysis, vol. 21, no. 3, pp. 267– 297, 2013. [17] Z. Liu, “User Adoption Analysis of Social Media Applications Based on NLP and Technology Acceptance Model-A Case Study of Facebook,” in 2022 5th International Conference on Data Science and Information Technology (DSIT), 2022, pp. 1–6. [18] T. Kuzman, I. Mozetiˇc, and N. Ljubeˇsi´c, “ChatGPT: Beginning of an End of Manual Linguistic Data Annotation? Use Case of Automatic Genre Identification,” 2023. https://doi.org/10.48550/arXiv.2303.03953 [19] H. Kim, K. Mitra, R. L. Chen, S. Rahman, and D. Zhang, “MEGAnno+: A Human-LLM Collaborative Annotation System,” 2024. https://doi.org/10.48550/arXiv.2402.18050 [20] A. Kangale, S. K. Kumar, M. A. Naeem, M. Williams, and M. K. Tiwari, “Mining consumer reviews to generate ratings of different product attributes while producing feature-based review-summary,” International Journal of Systems Science, vol. 47, no. 13, pp. 3272–3286, 2016. [21] P. T¨ornberg, “ChatGPT-4 Outperforms Experts and Crowd Workers in Annotating Political Twitter Messages with Zero-Shot Learning,” 2023. https://doi.org/10.48550/arXiv.2304.06588 [22] M. V. Reiss, “Testing the Reliability of ChatGPT for Text Annotation and Classification: A Cautionary Remark,” 2023. https://doi.org/10.48550/arXiv.2304.11085 [23] F. Huang, H. Kwak, and J. An, “Is ChatGPT better than Human Annotators? Potential and Limitations of ChatGPT in Explaining Implicit Hate Speech,” in Companion Proceedings of the ACM Web Conference 2023, 2023, pp. 294–297. [24] J. Wei et al., “Long-form factuality in large language models,” 2024. https://doi.org/10.48550/arXiv.2403.18802 [25] A. Goel et al., “LLMs Accelerate Annotation for Medical Information Extraction,” 2023. https://doi.org/10.48550/arXiv.2312.02296
[26] P. T¨ornberg, “Best Practices for Text Annotation with Large Language Models,” 2024. https://doi.org/10.48550/arXiv.2402.05129 [27] M. L. McHugh, “Interrater reliability: the kappa statistic,” Biochemia medica, vol. 22, no. 3, pp. 276–282, 2012.
